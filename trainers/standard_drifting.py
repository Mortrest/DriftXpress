import os
import time
import csv
import argparse
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from common.config import UNetConfig, UNetLargeConfig, MultiResDriftConfig
from common.loss_reporting import count_feature_slots, format_scientific, scaled_loss_for_logging
from models.unet import UNet
from models.ema import EMA
from features.encoders import build_encoder
from methods.standard_drifting import compute_normalized_drift_multitemp_batched
from evaluation.sample import (
    drift_sample,
    drift_sample_from_noise,
    ensure_three_channels,
    load_sample_noise,
    sample_model_noise,
    save_sample_grid,
)
from evaluate_fid_10k_raw import (
    SUPPORTED_DATASETS,
    build_cifar_dataset,
    build_dataset_transform,
    configure_unet_cfg_for_dataset,
    default_ref_dir,
    evaluate_checkpoint,
    evaluate_loaded_model,
    get_dataset_spec,
    normalize_dataset_name,
    normalize_eval_weights,
    prepare_cifar_reference,
)


FID_EVAL_ARGS = SimpleNamespace(
    batch_size=256,
    fid_batch_size=50,
    fid_num_workers=0,
    fid_timeout=None,
    compile=False,
    num_samples=10_000,
)
TRAIN_SUPPORTED_DATASETS = (*SUPPORTED_DATASETS, "imagenet32")


def normalize_training_dataset_name(dataset_name):
    if dataset_name is None:
        return normalize_dataset_name(dataset_name)
    normalized = str(dataset_name).strip().lower()
    if normalized == "imagenet32":
        return normalized
    return normalize_dataset_name(normalized)


def normalize_eval_weight_list(eval_weights_spec):
    if eval_weights_spec is None:
        return ("raw", "ema")

    normalized = []
    seen = set()
    for part in str(eval_weights_spec).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        weight_name = normalize_eval_weights(stripped)
        if weight_name not in seen:
            normalized.append(weight_name)
            seen.add(weight_name)

    if not normalized:
        raise ValueError("fid_eval_weights must include at least one of: raw, ema.")
    return tuple(normalized)


def normalize_sample_weights(sample_weights):
    normalized = str(sample_weights).strip().lower()
    if normalized not in {"raw", "ema"}:
        raise ValueError(f"Unsupported sample weights '{sample_weights}'. Expected one of: raw, ema.")
    return normalized


def delegate_imagenet32_training(args):
    unsupported_flags = []
    if args.no_save_on_eval:
        unsupported_flags.append("--no-save-on-eval")
    if args.resume_from:
        unsupported_flags.append("--resume-from")
    if args.sample_every_seconds:
        unsupported_flags.append("--sample-every-seconds")
    if unsupported_flags:
        raise ValueError(
            "The ImageNet backend behind Standard Drifting "
            f"does not support {', '.join(unsupported_flags)}."
        )

    from trainers import imagenet_standard_drifting

    batch_size = 256 if args.batch_size is None else int(args.batch_size)
    delegate_args = argparse.Namespace(
        data_source=args.data_source,
        output_dir=args.output_dir,
        encoder=args.encoder,
        encoder_size=args.encoder_size,
        steps=args.steps,
        batch_size=batch_size,
        real_batch_size=args.real_batch_size,
        real_feature_batch_size=args.real_feature_batch_size,
        temps=args.temps,
        pool_size=args.pool_size,
        more_features=args.more_features,
        evaluate_every=args.evaluate_every,
        evaluate_every_seconds=args.evaluate_every_seconds,
        save_every=args.save_every,
        fid_eval_weights=args.fid_eval_weights,
        skip_final_fid=args.skip_final_fid,
        deadline_epoch=args.deadline_epoch,
        data_seed=args.data_seed,
        max_samples=args.max_samples,
        train_sample_ratio=args.train_sample_ratio,
        device=args.device,
        no_compile_encoder=args.no_compile_encoder,
        large=args.large,
    )
    return imagenet_standard_drifting.train(delegate_args)


def format_fid_summary(fids):
    pieces = []
    for weight_name in ("raw", "ema"):
        if weight_name in fids:
            pieces.append(f"{weight_name}={fids[weight_name]:.4f}")
    return ", ".join(pieces)


def fid_log_values(fids):
    return {
        "fid_raw": f"{fids['raw']:.6f}" if "raw" in fids else "",
        "fid_ema": f"{fids['ema']:.6f}" if "ema" in fids else "",
    }


def get_host_sync_timeout():
    timeout_seconds = int(os.environ.get("HOST_SYNC_TIMEOUT_SEC", "7200"))
    if timeout_seconds <= 0:
        raise ValueError("HOST_SYNC_TIMEOUT_SEC must be positive.")
    return timedelta(seconds=timeout_seconds)


HOST_SYNC_TIMEOUT = get_host_sync_timeout()


def append_training_log(log_path, step, loss="", elapsed="", images_per_sec="", fid_raw="", fid_ema=""):
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([step, loss, elapsed, images_per_sec, fid_raw, fid_ema])


def append_fid_log(log_path, step, fid_raw="", fid_ema=""):
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([step, fid_raw, fid_ema, fid_raw])


def ensure_csv_header(log_path, header, upgrade_row):
    path = Path(log_path)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return

    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if rows and rows[0] == header:
        return

    upgraded_rows = [upgrade_row(row) for row in rows[1:] if any(cell != "" for cell in row)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(upgraded_rows)


def upgrade_training_log_row(row):
    return [
        row[0] if len(row) > 0 else "",
        row[1] if len(row) > 1 else "",
        row[2] if len(row) > 2 else "",
        row[3] if len(row) > 3 else "",
        row[4] if len(row) > 4 else "",
        row[5] if len(row) > 5 else "",
    ]


def upgrade_fid_log_row(row):
    fid_raw = row[1] if len(row) > 1 else ""
    fid_ema = row[2] if len(row) > 2 else ""
    fid_alias = row[3] if len(row) > 3 else fid_raw
    return [
        row[0] if len(row) > 0 else "",
        fid_raw,
        fid_ema,
        fid_alias,
    ]


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def load_training_checkpoint(checkpoint_path, train_model, ema, optimizer, scaler, device):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_model.load_state_dict(checkpoint["model"])

    ema_state = checkpoint.get("ema")
    if ema_state is not None:
        ema.load_state_dict(ema_state)

    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_state_to_device(optimizer, device)

    scaler_state = checkpoint.get("scaler")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    return checkpoint_path, int(checkpoint.get("step", 0))


def build_fid_eval_args(eval_weights):
    return SimpleNamespace(
        batch_size=FID_EVAL_ARGS.batch_size,
        fid_batch_size=FID_EVAL_ARGS.fid_batch_size,
        fid_num_workers=FID_EVAL_ARGS.fid_num_workers,
        fid_timeout=FID_EVAL_ARGS.fid_timeout,
        compile=FID_EVAL_ARGS.compile,
        eval_weights=normalize_eval_weights(eval_weights),
        num_samples=FID_EVAL_ARGS.num_samples,
    )


def evaluate_fid_from_checkpoint(checkpoint_path, ref_dir, device, eval_weights):
    checkpoint_path = Path(checkpoint_path).resolve()
    eval_weights = normalize_eval_weights(eval_weights)
    num_samples = FID_EVAL_ARGS.num_samples
    fake_dir = checkpoint_path.parent.parent / "evaluation" / f"{checkpoint_path.stem}_fid_{num_samples}_{eval_weights}"
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        fake_dir=fake_dir,
        ref_dir=Path(ref_dir),
        device=device,
        args=build_fid_eval_args(eval_weights),
    )
    return result["fid"]


def evaluate_requested_fids_from_checkpoint(checkpoint_path, ref_dir, device, eval_weights):
    return {
        weight_name: evaluate_fid_from_checkpoint(checkpoint_path, ref_dir, device, weight_name)
        for weight_name in eval_weights
    }


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def broadcast_main_flag(flag, device):
    if not is_distributed():
        return bool(flag)
    tensor = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.broadcast(tensor, src=0)
    return bool(int(tensor.item()))


def broadcast_main_float(value, device):
    if not is_distributed():
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


def setup_runtime(use_ddp):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    host_sync_group = None
    if use_ddp:
        try:
            dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        try:
            host_sync_group = dist.new_group(backend="gloo", timeout=HOST_SYNC_TIMEOUT)
        except TypeError:
            host_sync_group = dist.new_group(backend="gloo")
    else:
        rank = 0
        world_size = 1
    return rank, world_size, local_rank, host_sync_group


def sync_workers(host_sync_group):
    if not is_distributed():
        return
    if host_sync_group is not None:
        dist.barrier(group=host_sync_group)
        return
    dist.barrier()


def cleanup():
    if is_distributed():
        dist.destroy_process_group()


def all_gather_flat(tensor):
    """All-gather tensors across all ranks. Returns concatenated along dim 0."""
    tensor = tensor.contiguous()
    if not is_distributed():
        return tensor
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def precompute_features(encoder, dataset, device, batch_size=256, rank=0, is_main=False):
    """Pre-compute features for all images in the dataset.

    Returns a list of tensors [N_total, L, C] for each feature group,
    plus the C_j values. Stored on CPU to save GPU memory.
    """
    if is_main:
        print("Pre-computing real image features...")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

    all_feats_by_group = None
    n_processed = 0

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for images, _ in loader:
            images = images.to(device)
            groups = encoder(ensure_three_channels(images))

            if all_feats_by_group is None:
                all_feats_by_group = [[] for _ in groups]

            for i, (feat, C_j) in enumerate(groups):
                all_feats_by_group[i].append(feat.float().cpu())

            n_processed += images.shape[0]
            if is_main and n_processed % 10000 == 0:
                print(f"  {n_processed}/{len(dataset)} images")

    # Concatenate and store
    result_feats = []
    result_cjs = []
    for i, feat_list in enumerate(all_feats_by_group):
        cat = torch.cat(feat_list, dim=0)  # [N_total, L, C]
        result_feats.append(cat)
        result_cjs.append(groups[i][1])

    if is_main:
        total_bytes = sum(f.numel() * f.element_size() for f in result_feats)
        print(f"  Pre-computed {n_processed} images, {total_bytes / 1e9:.2f} GB on CPU")
        for i, (f, c) in enumerate(zip(result_feats, result_cjs)):
            print(f"    Group {i}: shape={list(f.shape)}, C_j={c}")

    return result_feats, result_cjs


def build_live_fid_dir(out_dir, step, eval_weights):
    eval_weights = normalize_eval_weights(eval_weights)
    return (
        Path(out_dir)
        / "evaluation"
        / f"drift_step{int(step):07d}_fid_{FID_EVAL_ARGS.num_samples}_{eval_weights}"
    )


def evaluate_requested_fids_from_live_model(
    model,
    ref_dir,
    device,
    out_dir,
    step,
    dataset_name,
    eval_weights,
):
    if tuple(eval_weights) != ("raw",):
        raise ValueError(
            "Live-model FID evaluation without saving checkpoints only supports raw weights."
        )

    fake_dir = build_live_fid_dir(out_dir, step, "raw")
    eval_args = build_fid_eval_args("raw")
    result = evaluate_loaded_model(
        model=model,
        fake_dir=fake_dir,
        ref_dir=Path(ref_dir),
        device=device,
        args=eval_args,
        dataset_name=dataset_name,
        eval_weights="raw",
        metadata={"source": "live_model", "step": int(step)},
    )
    return {"raw": result["fid"]}


def multires_drift_loss_distributed(
    gen_groups,
    pos_groups,
    temps,
    rank,
    world_size,
    global_query_gather=False,
):
    """Compute multi-resolution drifting loss with optional full-batch feature gathering."""
    total_loss = 0.0

    for (gen_feat, C_j), (pos_feat, _) in zip(gen_groups, pos_groups):
        B_local = gen_feat.shape[0]

        with torch.no_grad():
            gen_det = gen_feat.detach()
            pos_det = pos_feat.detach()

            if global_query_gather:
                drift_gen = all_gather_flat(gen_det)
                drift_pos = all_gather_flat(pos_det)
                local_query_slice = slice(rank * B_local, (rank + 1) * B_local)
            else:
                drift_gen = gen_det
                drift_pos = pos_det
                local_query_slice = slice(0, B_local)

            gen_t = drift_gen.transpose(0, 1).contiguous()
            pos_t = drift_pos.transpose(0, 1).contiguous()

            gen_t_n, V, S = compute_normalized_drift_multitemp_batched(gen_t, pos_t, temps=temps)

            target = (gen_t_n + V)[:, local_query_slice, :].transpose(0, 1)

        gen_n = gen_feat / S
        loss_group = (gen_n - target).pow(2).mean(dim=(0, 2)).sum()
        total_loss = total_loss + loss_group

    return total_loss


def train(args):
    if normalize_training_dataset_name(args.dataset) == "imagenet32":
        return delegate_imagenet32_training(args)

    use_ddp = "RANK" in os.environ or "WORLD_SIZE" in os.environ
    rank, world_size, local_rank, host_sync_group = setup_runtime(use_ddp)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    if is_main:
        launch_mode = "DDP" if use_ddp else "single-process"
        print(f"Launch mode: {launch_mode}")
        if use_ddp:
            print(f"DDP: {world_size} GPUs")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = MultiResDriftConfig()
    cfg.evaluate_every = 10_000
    if args.steps:
        cfg.total_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.ema_decay is not None:
        cfg.ema_decay = float(args.ema_decay)
    if args.encoder:
        cfg.encoder = args.encoder
    if args.temps:
        cfg.temperatures = [float(t) for t in args.temps.split(",")]
    if args.pool_size:
        cfg.pool_size = args.pool_size
    if args.more_features:
        cfg.more_features = True
    cfg.dataset = normalize_training_dataset_name(args.dataset)
    if args.evaluate_every is not None:
        cfg.evaluate_every = args.evaluate_every
    if args.save_every is not None:
        cfg.save_every = args.save_every
    if args.sample_every is not None:
        cfg.sample_every = args.sample_every

    evaluate_every_seconds = int(args.evaluate_every_seconds or 0)
    sample_every_seconds = int(args.sample_every_seconds or 0)
    fid_eval_weights = normalize_eval_weight_list(args.fid_eval_weights)
    sample_weights = normalize_sample_weights(args.sample_weights)
    save_on_eval = not args.no_save_on_eval
    deadline_epoch = float(args.deadline_epoch) if args.deadline_epoch is not None else None
    sample_grid_cols = int(args.sample_grid_cols) if args.sample_grid_cols is not None else 8

    if cfg.evaluate_every < 0:
        raise ValueError("evaluate_every must be non-negative.")
    if cfg.sample_every < 0:
        raise ValueError("sample_every must be non-negative.")
    if cfg.save_every < 0:
        raise ValueError("save_every must be non-negative.")
    if evaluate_every_seconds < 0:
        raise ValueError("evaluate_every_seconds must be non-negative.")
    if sample_every_seconds < 0:
        raise ValueError("sample_every_seconds must be non-negative.")
    if sample_grid_cols <= 0:
        raise ValueError("sample_grid_cols must be positive.")
    if not (0.0 < cfg.ema_decay < 1.0):
        raise ValueError("ema_decay must be between 0 and 1.")
    if not save_on_eval and fid_eval_weights != ("raw",):
        raise ValueError("--no-save-on-eval only supports --fid-eval-weights raw.")

    encoder_input_size = args.encoder_size or 112
    data_root = args.data_root
    _, dataset_spec = get_dataset_spec(cfg.dataset)
    dataset_label = dataset_spec["pretty_name"]

    out_dir = args.output_dir
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # UNet
    unet_cfg = UNetLargeConfig() if args.large else UNetConfig()
    unet_cfg = configure_unet_cfg_for_dataset(unet_cfg, cfg.dataset)
    model = UNet(
        in_ch=unet_cfg.in_ch, out_ch=unet_cfg.out_ch, base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult, num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions, dropout=unet_cfg.dropout,
        num_heads=unet_cfg.num_heads,
        image_size=unet_cfg.image_size,
    ).to(memory_format=torch.channels_last).to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"UNet parameters: {n_params:,}")

    model = torch.compile(model)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
    train_model = model.module if use_ddp else model

    ema = EMA(train_model, decay=cfg.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
        weight_decay=0.0, fused=True,
    )
    scaler = torch.amp.GradScaler("cuda")

    def build_training_checkpoint(step_value):
        return {
            "step": int(step_value),
            "model": train_model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": {"unet": unet_cfg, "drift": cfg},
        }

    fixed_sample_noise = None
    if args.fixed_sample_noise_path:
        fixed_sample_noise = load_sample_noise(
            args.fixed_sample_noise_path,
            device=device,
            memory_format=torch.channels_last,
        )

    resume_checkpoint_path = None
    start_step = 0
    if args.resume_from:
        resume_checkpoint_path, start_step = load_training_checkpoint(
            checkpoint_path=args.resume_from,
            train_model=train_model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        if is_main:
            print(f"Loaded checkpoint from {resume_checkpoint_path}")
            print(f"  resume_step={start_step}")
        if use_ddp:
            sync_workers(host_sync_group)
        if start_step >= cfg.total_steps:
            if is_main:
                print(
                    f"Checkpoint step {start_step} is already >= requested total steps "
                    f"{cfg.total_steps}. Nothing to do."
                )
            cleanup()
            return

    # Multi-res encoder (built on rank 0 first to download weights)
    if use_ddp and rank == 0:
        feat_encoder = build_encoder(cfg.encoder, pool_size=cfg.pool_size,
                                     input_size=encoder_input_size,
                                     more_features=cfg.more_features).to(device)
    if use_ddp:
        sync_workers(host_sync_group)
    if not use_ddp or rank != 0:
        feat_encoder = build_encoder(cfg.encoder, pool_size=cfg.pool_size,
                                     input_size=encoder_input_size,
                                     more_features=cfg.more_features).to(device)
    if use_ddp:
        sync_workers(host_sync_group)
    feat_encoder.eval()

    # Compile the encoder for faster forward/backward
    feat_encoder = torch.compile(feat_encoder)

    if is_main:
        n_feat_params = sum(p.numel() for p in feat_encoder.parameters())
        print(f"Feature encoder: {cfg.encoder}, {n_feat_params:,} params (frozen, compiled)")
        print(f"  Input size: {encoder_input_size}x{encoder_input_size}")

    # ---- Pre-compute real features (the big optimization) ----
    if is_main:
        build_cifar_dataset(
            dataset_name=cfg.dataset,
            root=data_root,
            train=True,
            download=True,
            transform=None,
        )
    if use_ddp:
        sync_workers(host_sync_group)

    transform_real = build_dataset_transform(cfg.dataset)
    full_dataset = build_cifar_dataset(
        dataset_name=cfg.dataset,
        root=data_root,
        train=True,
        download=False,
        transform=transform_real,
    )

    precomp_feats, precomp_cjs = precompute_features(
        feat_encoder, full_dataset, device, batch_size=512, rank=rank, is_main=is_main
    )

    if is_main:
        print("Pre-computation complete. Real features cached on CPU.")

    # Logging
    log_path = os.path.join(out_dir, "loss_log.csv")
    fid_log_path = os.path.join(out_dir, "fid_log.csv")
    if is_main:
        ensure_csv_header(
            log_path,
            ["step", "loss", "time_s", "images_per_sec", "fid_raw", "fid_ema"],
            upgrade_training_log_row,
        )
        ensure_csv_header(
            fid_log_path,
            ["step", "fid_raw", "fid_ema", "fid"],
            upgrade_fid_log_row,
        )
        global_bs = cfg.batch_size * world_size
        print(f"Training Drifting (multi-res {cfg.encoder}) for {cfg.total_steps} steps")
        print(f"  dataset={dataset_label}")
        print(f"  per-GPU batch={cfg.batch_size}, global batch={global_bs}")
        print(f"  temps={cfg.temperatures}, pool_size={cfg.pool_size}")
        print(f"  more_features={cfg.more_features}")
        print(f"  encoder input={encoder_input_size}x{encoder_input_size}")
        print(f"  evaluate_every={cfg.evaluate_every}")
        print(f"  evaluate_every_seconds={evaluate_every_seconds}")
        print(f"  sample_every={cfg.sample_every}")
        print(f"  sample_every_seconds={sample_every_seconds}")
        print(f"  sample_weights={sample_weights}")
        print(f"  save_every={cfg.save_every}")
        print(f"  fid_eval_weights={','.join(fid_eval_weights)}")
        print(f"  save_on_eval={save_on_eval}")
        print(f"  skip_final_fid={args.skip_final_fid}")
        print(f"  deadline_epoch={deadline_epoch if deadline_epoch is not None else '<none>'}")
        print(f"  start_step={start_step}")
        if fixed_sample_noise is not None:
            print(f"  fixed_sample_noise_path={args.fixed_sample_noise_path}")
            print(f"  fixed_sample_count={fixed_sample_noise.shape[0]}")
            print(f"  sample_grid_cols={sample_grid_cols}")
        if resume_checkpoint_path is not None:
            print(f"  resume_from={resume_checkpoint_path}")

    fid_ref_dir = None
    if is_main:
        fid_ref_dir = prepare_cifar_reference(
            dataset_name=cfg.dataset,
            output_dir=default_ref_dir(cfg.dataset, FID_EVAL_ARGS.num_samples),
            data_root=data_root,
            num_images=FID_EVAL_ARGS.num_samples,
        )
    if use_ddp:
        sync_workers(host_sync_group)

    # Index tracking for pre-computed features
    # We need to know which dataset indices the DataLoader returns.
    # Simplest: use a custom dataset that returns (image, index)
    class IndexedImageDataset(torch.utils.data.Dataset):
        def __init__(self, dataset_name, root, train, transform):
            self.ds = build_cifar_dataset(
                dataset_name=dataset_name,
                root=root,
                train=train,
                download=False,
                transform=transform,
            )
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return img, idx

    indexed_dataset = IndexedImageDataset(
        cfg.dataset,
        data_root,
        train=True,
        transform=build_dataset_transform(cfg.dataset),
    )
    indexed_sampler = None
    if use_ddp:
        indexed_sampler = DistributedSampler(
            indexed_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        indexed_loader = DataLoader(
            indexed_dataset,
            batch_size=cfg.batch_size,
            sampler=indexed_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    else:
        indexed_loader = DataLoader(
            indexed_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    start_time = time.time()
    next_eval_time = None
    if evaluate_every_seconds > 0:
        initial_eval_time = start_time + evaluate_every_seconds
        next_eval_time = (
            broadcast_main_float(initial_eval_time if is_main else 0.0, device)
            if use_ddp
            else initial_eval_time
        )
    next_sample_time = None
    if sample_every_seconds > 0:
        initial_sample_time = start_time + sample_every_seconds
        next_sample_time = (
            broadcast_main_float(initial_sample_time if is_main else 0.0, device)
            if use_ddp
            else initial_sample_time
        )
    last_fid_step = None
    last_step = start_step
    epoch = 0
    data_iter = iter(indexed_loader)
    loss_log_slot_count = None

    for step in range(start_step + 1, cfg.total_steps + 1):
        last_step = step
        try:
            _, indices = next(data_iter)
        except StopIteration:
            epoch += 1
            if indexed_sampler is not None:
                indexed_sampler.set_epoch(epoch)
            data_iter = iter(indexed_loader)
            _, indices = next(data_iter)

        B = indices.shape[0]

        # Look up pre-computed real features for this batch.
        pos_groups = []
        for i, C_j in enumerate(precomp_cjs):
            feats = precomp_feats[i].index_select(0, indices)  # [B, L, C]
            pos_groups.append((feats.to(device), C_j))

        # Generate images from noise
        z = sample_model_noise(model, B, device, memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_images = model(z)

        # Extract features from generated images (with grad, through encoder)
        gen_f32 = gen_images.float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_groups_raw = feat_encoder(ensure_three_channels(gen_f32))
        # Cast back to float32 for drift computation
        gen_groups = [(f.float(), c) for f, c in gen_groups_raw]
        if loss_log_slot_count is None:
            loss_log_slot_count = count_feature_slots(gen_groups)
            if is_main:
                print(
                    f"  loss logging normalized by {loss_log_slot_count} feature slots "
                    f"(sum of encoder locations across groups)"
                )

        # Compute drift loss with global all-gather
        loss = multires_drift_loss_distributed(
            gen_groups, pos_groups,
            temps=tuple(cfg.temperatures),
            rank=rank,
            world_size=world_size,
            global_query_gather=cfg.drift_global_query_gather,
        )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        ema.update(train_model)

        if step % cfg.log_every == 0 and is_main:
            logged_loss, _ = scaled_loss_for_logging(loss, gen_groups)
            elapsed = time.time() - start_time
            completed_steps = step - start_step
            steps_per_sec = completed_steps / elapsed
            imgs_per_sec = steps_per_sec * cfg.batch_size * world_size
            eta_h = (cfg.total_steps - step) / steps_per_sec / 3600
            print(
                f"step {step:>7d}/{cfg.total_steps} | "
                f"loss {format_scientific(logged_loss)} | "
                f"{steps_per_sec:.1f} it/s ({imgs_per_sec:.0f} img/s) | "
                f"ETA {eta_h:.1f}h"
            )
            append_training_log(
                log_path,
                step,
                loss=format_scientific(logged_loss),
                elapsed=f"{elapsed:.1f}",
                images_per_sec=f"{imgs_per_sec:.0f}",
            )

        should_sample_step = (cfg.sample_every > 0 and step % cfg.sample_every == 0)
        main_should_sample_time = (
            next_sample_time is not None and time.time() >= next_sample_time
        ) if is_main else False
        should_sample_time = (
            broadcast_main_flag(main_should_sample_time, device)
            if use_ddp
            else main_should_sample_time
        )
        if should_sample_time:
            if is_main:
                current_time = time.time()
                while next_sample_time is not None and next_sample_time <= current_time:
                    next_sample_time += sample_every_seconds
            if use_ddp:
                next_sample_time = broadcast_main_float(next_sample_time if is_main else 0.0, device)
        should_sample = should_sample_step or should_sample_time

        if should_sample and is_main:
            sample_model = ema.shadow if sample_weights == "ema" else train_model
            if sample_weights == "raw":
                model.eval()
            else:
                sample_model.eval()
            if fixed_sample_noise is not None:
                samples = drift_sample_from_noise(sample_model, fixed_sample_noise)
            else:
                samples = drift_sample(sample_model, 64, device)
            save_sample_grid(
                samples,
                os.path.join(out_dir, "samples", f"drift_step{step:07d}.png"),
                nrow=sample_grid_cols,
            )
            if sample_weights == "raw":
                model.train()
            print(f"  Saved {sample_weights} sample grid at step {step}")
        if use_ddp and should_sample:
            sync_workers(host_sync_group)

        should_save = (cfg.save_every > 0 and step % cfg.save_every == 0)
        main_deadline_reached = (
            deadline_epoch is not None and time.time() >= deadline_epoch
        ) if is_main else False
        deadline_reached = (
            broadcast_main_flag(main_deadline_reached, device)
            if use_ddp
            else main_deadline_reached
        )
        if deadline_reached:
            if is_main:
                print(f"  Reached wall-clock deadline at step {step}; stopping training.")
            break

        should_evaluate_step = (cfg.evaluate_every > 0 and step % cfg.evaluate_every == 0)
        main_should_evaluate_time = (
            next_eval_time is not None and time.time() >= next_eval_time
        ) if is_main else False
        should_evaluate_time = (
            broadcast_main_flag(main_should_evaluate_time, device)
            if use_ddp
            else main_should_evaluate_time
        )
        if should_evaluate_time:
            if is_main:
                current_time = time.time()
                while next_eval_time is not None and next_eval_time <= current_time:
                    next_eval_time += evaluate_every_seconds
            if use_ddp:
                next_eval_time = broadcast_main_float(next_eval_time if is_main else 0.0, device)
        should_evaluate = should_evaluate_step or should_evaluate_time

        checkpoint_path = None
        if should_save and is_main:
            checkpoint_path = os.path.join(out_dir, "checkpoints", f"drift_step{step:07d}.pt")
            checkpoint_payload = build_training_checkpoint(step)
            torch.save(checkpoint_payload, checkpoint_path)
            torch.save(checkpoint_payload, os.path.join(out_dir, "checkpoints", "drift_latest.pt"))
            print(f"  Saved checkpoint at step {step}")

        if should_evaluate and is_main:
            eval_start = time.time()
            print(
                f"  Starting FID evaluation at step {step} "
                f"({','.join(fid_eval_weights)})."
            )
            if checkpoint_path is not None or save_on_eval:
                if checkpoint_path is None:
                    checkpoint_path = os.path.join(out_dir, "checkpoints", f"drift_step{step:07d}.pt")
                    checkpoint_payload = build_training_checkpoint(step)
                    torch.save(checkpoint_payload, checkpoint_path)
                    torch.save(
                        checkpoint_payload,
                        os.path.join(out_dir, "checkpoints", "drift_latest.pt"),
                    )
                    print(f"  Saved checkpoint at step {step}")
                fids = evaluate_requested_fids_from_checkpoint(
                    checkpoint_path,
                    fid_ref_dir,
                    device,
                    fid_eval_weights,
                )
            else:
                fids = evaluate_requested_fids_from_live_model(
                    model=train_model,
                    ref_dir=fid_ref_dir,
                    device=device,
                    out_dir=out_dir,
                    step=step,
                    dataset_name=cfg.dataset,
                    eval_weights=fid_eval_weights,
                )
            eval_elapsed = time.time() - eval_start
            print(
                f"  FID {format_fid_summary(fids)} "
                f"(eval {eval_elapsed / 60:.1f} min)."
            )
            fid_kwargs = fid_log_values(fids)
            append_training_log(
                log_path,
                step,
                elapsed=f"{time.time() - start_time:.1f}",
                fid_raw=fid_kwargs["fid_raw"],
                fid_ema=fid_kwargs["fid_ema"],
            )
            append_fid_log(
                fid_log_path,
                step,
                fid_raw=fid_kwargs["fid_raw"],
                fid_ema=fid_kwargs["fid_ema"],
            )
            last_fid_step = step
        if use_ddp and (should_save or should_evaluate):
            sync_workers(host_sync_group)

    if is_main:
        elapsed = time.time() - start_time
        completed_steps = last_step - start_step
        if last_step >= cfg.total_steps:
            print(
                f"\nTraining complete. Reached step {last_step} after "
                f"{completed_steps} step(s) in {elapsed/3600:.1f} hours"
            )
        else:
            print(
                f"\nTraining stopped at step {last_step} after "
                f"{completed_steps} step(s) in {elapsed/3600:.1f} hours"
            )
        final_path = os.path.join(out_dir, "checkpoints", "drift_final.pt")
        torch.save({
            "step": last_step,
            "model": train_model.state_dict(),
            "ema": ema.state_dict(),
            "config": {"unet": unet_cfg, "drift": cfg},
        }, final_path)
        if not args.skip_final_fid and last_step > 0 and last_fid_step != last_step:
            eval_start = time.time()
            print(
                f"  Starting final FID evaluation from {final_path} "
                f"({','.join(fid_eval_weights)})."
            )
            final_fids = evaluate_requested_fids_from_checkpoint(
                final_path,
                fid_ref_dir,
                device,
                fid_eval_weights,
            )
            eval_elapsed = time.time() - eval_start
            print(
                f"  Final FID {format_fid_summary(final_fids)} "
                f"(eval {eval_elapsed / 60:.1f} min)."
            )
            final_fid_kwargs = fid_log_values(final_fids)
            append_training_log(
                log_path,
                last_step,
                elapsed=f"{time.time() - start_time:.1f}",
                fid_raw=final_fid_kwargs["fid_raw"],
                fid_ema=final_fid_kwargs["fid_ema"],
            )
            append_fid_log(
                fid_log_path,
                last_step,
                fid_raw=final_fid_kwargs["fid_raw"],
                fid_ema=final_fid_kwargs["fid_ema"],
            )

    if use_ddp:
        sync_workers(host_sync_group)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/drift_multires")
    parser.add_argument("--encoder", type=str, default="dinov2-multires",
                        choices=["dinov2-multires", "convnextv2", "mocov2",
                                 "dinov3", "eva02", "siglip2", "clip"])
    parser.add_argument("--encoder-size", type=int, default=112,
                        help="Encoder input resolution (default 112, was 224)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU batch size")
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=TRAIN_SUPPORTED_DATASETS,
        help="Training dataset to use for real-image features and FID references.",
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--data-source",
        type=str,
        default="data/Imagenet32_train.zip",
        help="ImageNet32 source archive or extracted batch directory. Used only with --dataset imagenet32.",
    )
    parser.add_argument("--temps", type=str, default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--more-feautres", "--more-features", dest="more_features", action="store_true")
    parser.add_argument(
        "--real-batch-size",
        type=int,
        default=None,
        help="Real batch size per GPU for ImageNet32 exact training. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--real-feature-batch-size",
        type=int,
        default=None,
        help="Batch size used while building the cached real-feature bank for ImageNet32.",
    )
    parser.add_argument("--evaluate-every", type=int, default=None)
    parser.add_argument("--evaluate-every-seconds", type=int, default=None)
    parser.add_argument("--sample-every-seconds", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--fid-eval-weights", type=str, default="raw,ema")
    parser.add_argument("--sample-weights", type=str, default="raw", choices=("raw", "ema"))
    parser.add_argument("--no-save-on-eval", action="store_true")
    parser.add_argument("--skip-final-fid", action="store_true")
    parser.add_argument("--deadline-epoch", type=float, default=None)
    parser.add_argument(
        "--fixed-sample-noise-path",
        type=str,
        default=None,
        help="Optional shared latent-noise tensor to reuse for all saved sample grids.",
    )
    parser.add_argument(
        "--sample-grid-cols",
        type=int,
        default=None,
        help="Optional column count for saved sample grids.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="Base seed for per-epoch ImageNet32 shuffling.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on how many ImageNet32 samples to use per epoch for debugging.",
    )
    parser.add_argument(
        "--train-sample-ratio",
        type=float,
        default=None,
        help="Optional fraction or percent of ImageNet32 training samples to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override for single-process ImageNet32 runs.",
    )
    parser.add_argument(
        "--no-compile-encoder",
        action="store_true",
        help="Disable encoder compilation for ImageNet32 runs.",
    )
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()
    train(args)
