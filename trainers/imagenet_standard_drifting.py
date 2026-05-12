from __future__ import annotations

import argparse
import csv
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.imagenet import load_imagenet32_arrays
from trainers.imagenet_common import (
    clear_imagenet32_array_cache,
    iter_sharded_shuffled_index_batches,
    precompute_imagenet32_features,
    resolve_effective_max_samples,
    write_training_config_snapshot,
)
from trainers.imagenet_driftxpress import (
    DEFAULT_IMAGENET32_REF_DIR,
    DEFAULT_NUM_FID_SAMPLES,
    evaluate_all_fids_from_checkpoint,
    evaluate_requested_fids_from_checkpoint,
    fid_log_values,
    format_fid_summary,
    normalize_eval_weight_list,
    prepare_imagenet32_reference,
)
from common.config import MultiResDriftConfig, UNetConfig, UNetLargeConfig
from evaluation.sample import drift_sample, save_sample_grid
from models.ema import EMA
from models.unet import UNet
from methods.standard_drifting import (
    compute_normalized_drift_multitemp_batched,
    drifting_loss_multires,
)
from features.encoders import build_encoder
from common.loss_reporting import count_feature_slots, format_scientific, scaled_loss_for_logging


def append_training_log(log_path, step, loss="", elapsed="", images_per_sec="", fid_raw="", fid_ema=""):
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([step, loss, elapsed, images_per_sec, fid_raw, fid_ema])


def append_fid_log(log_path, step, fid_raw="", fid_ema=""):
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([step, fid_raw, fid_ema, fid_raw])


def get_host_sync_timeout():
    timeout_seconds = int(os.environ.get("HOST_SYNC_TIMEOUT_SEC", "7200"))
    if timeout_seconds <= 0:
        raise ValueError("HOST_SYNC_TIMEOUT_SEC must be positive.")
    return timedelta(seconds=timeout_seconds)


HOST_SYNC_TIMEOUT = get_host_sync_timeout()


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
    if torch.cuda.is_available():
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


def all_gather_flat(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    if not is_distributed():
        return tensor
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def multires_drift_loss_distributed(
    gen_groups,
    pos_groups,
    temps,
    rank,
    world_size,
    global_query_gather=False,
):
    total_loss = gen_groups[0][0].new_tensor(0.0)

    for (gen_feat, C_j), (pos_feat, _) in zip(gen_groups, pos_groups, strict=True):
        local_batch = gen_feat.shape[0]

        with torch.no_grad():
            gen_det = gen_feat.detach()
            pos_det = pos_feat.detach()

            if global_query_gather:
                drift_gen = all_gather_flat(gen_det)
                drift_pos = all_gather_flat(pos_det)
                local_query_slice = slice(rank * local_batch, (rank + 1) * local_batch)
            else:
                drift_gen = gen_det
                drift_pos = pos_det
                local_query_slice = slice(0, local_batch)

            gen_t = drift_gen.transpose(0, 1).contiguous()
            pos_t = drift_pos.transpose(0, 1).contiguous()
            gen_t_n, V, S = compute_normalized_drift_multitemp_batched(gen_t, pos_t, temps=temps)
            target = (gen_t_n + V)[:, local_query_slice, :].transpose(0, 1)

        gen_n = gen_feat / S
        loss_group = (gen_n - target).pow(2).mean(dim=(0, 2)).sum()
        total_loss = total_loss + loss_group

    return total_loss


def train(args):
    use_ddp = "RANK" in os.environ or "WORLD_SIZE" in os.environ
    rank, world_size, local_rank, host_sync_group = setup_runtime(use_ddp)
    device = (
        torch.device(f"cuda:{local_rank}")
        if use_ddp
        else torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    is_main = rank == 0
    real_batch_size = args.real_batch_size or args.batch_size

    try:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        cfg = MultiResDriftConfig()
        if args.steps is not None:
            cfg.total_steps = args.steps
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.encoder is not None:
            cfg.encoder = args.encoder
        if args.temps is not None:
            cfg.temperatures = [float(t) for t in args.temps.split(",")]
        if args.pool_size is not None:
            cfg.pool_size = args.pool_size
        if args.more_features:
            cfg.more_features = True
        if args.evaluate_every is not None:
            cfg.evaluate_every = args.evaluate_every
        if args.save_every is not None:
            cfg.save_every = args.save_every
        cfg.dataset = "imagenet32"
        evaluate_every_seconds = int(args.evaluate_every_seconds or 0)
        fid_eval_weights = normalize_eval_weight_list(args.fid_eval_weights)
        deadline_epoch = float(args.deadline_epoch) if args.deadline_epoch is not None else None
        if cfg.evaluate_every < 0:
            raise ValueError("evaluate_every must be non-negative.")
        if cfg.save_every < 0:
            raise ValueError("save_every must be non-negative.")
        if evaluate_every_seconds < 0:
            raise ValueError("evaluate_every_seconds must be non-negative.")

        encoder_input_size = args.encoder_size or 112
        out_dir = args.output_dir
        if is_main:
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

        if use_ddp:
            sync_workers(host_sync_group)

        if is_main:
            launch_mode = "DDP" if use_ddp else "single-process"
            print(f"Launch mode: {launch_mode}")
            if use_ddp:
                print(f"DDP: {world_size} GPUs")

        unet_cfg = UNetLargeConfig() if args.large else UNetConfig()
        model = UNet(
            in_ch=unet_cfg.in_ch,
            out_ch=unet_cfg.out_ch,
            base_ch=unet_cfg.base_ch,
            ch_mult=unet_cfg.ch_mult,
            num_res_blocks=unet_cfg.num_res_blocks,
            attn_resolutions=unet_cfg.attn_resolutions,
            dropout=unet_cfg.dropout,
            num_heads=unet_cfg.num_heads,
        ).to(memory_format=torch.channels_last).to(device)

        if is_main:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"UNet parameters: {n_params:,}")
        else:
            n_params = None

        model_compiled = False
        if device.type == "cuda":
            model = torch.compile(model)
            model_compiled = True
        if use_ddp:
            model = DDP(model, device_ids=[local_rank])
        train_model = model.module if use_ddp else model

        if use_ddp and rank == 0:
            feat_encoder = build_encoder(
                cfg.encoder,
                pool_size=cfg.pool_size,
                input_size=encoder_input_size,
                more_features=cfg.more_features,
            ).to(device)
        if use_ddp:
            sync_workers(host_sync_group)
        if not use_ddp or rank != 0:
            feat_encoder = build_encoder(
                cfg.encoder,
                pool_size=cfg.pool_size,
                input_size=encoder_input_size,
                more_features=cfg.more_features,
            ).to(device)
        if use_ddp:
            sync_workers(host_sync_group)
        feat_encoder.eval()

        feat_encoder_compiled = False
        if device.type == "cuda" and not args.no_compile_encoder:
            feat_encoder = torch.compile(feat_encoder)
            feat_encoder_compiled = True

        if is_main:
            n_feat_params = sum(p.numel() for p in feat_encoder.parameters())
            print(f"Feature encoder: {cfg.encoder}, {n_feat_params:,} params (frozen)")
            print(f"  Input size: {encoder_input_size}x{encoder_input_size}")
        else:
            n_feat_params = None

        data_source = Path(args.data_source)
        total_available_samples = int(load_imagenet32_arrays(data_source)[0].shape[0])
        effective_max_samples, normalized_train_sample_ratio = resolve_effective_max_samples(
            total_available_samples,
            max_samples=args.max_samples,
            train_sample_ratio=args.train_sample_ratio,
        )
        if effective_max_samples <= 0:
            raise ValueError("Effective ImageNet training sample count resolved to zero.")
        args.resolved_train_sample_ratio = normalized_train_sample_ratio
        args.effective_max_samples = effective_max_samples
        if is_main:
            print(f"Loaded ImageNet from {data_source}")
            write_training_config_snapshot(
                out_dir=out_dir,
                args=args,
                cfg=cfg,
                unet_cfg=unet_cfg,
                encoder_input_size=encoder_input_size,
                real_batch_size=real_batch_size,
                model_num_params=n_params,
                feat_encoder_num_params=n_feat_params,
                model_compiled=model_compiled,
                feat_encoder_compiled=feat_encoder_compiled,
            )

        ema = EMA(train_model, decay=cfg.ema_decay)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            fused=(device.type == "cuda"),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        def build_training_checkpoint(step_value):
            return {
                "step": int(step_value),
                "model": train_model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": {"unet": unet_cfg, "drift": cfg},
            }

        log_path = os.path.join(out_dir, "loss_log.csv")
        fid_log_path = os.path.join(out_dir, "fid_log.csv")
        if is_main:
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss", "time_s", "images_per_sec", "fid_raw", "fid_ema"])
            with open(fid_log_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "fid_raw", "fid_ema", "fid"])

            global_gen_batch = cfg.batch_size * world_size
            global_real_batch = real_batch_size * world_size
            print(f"Training Standard Drifting (multi-res {cfg.encoder}) on ImageNet for {cfg.total_steps} steps")
            print(f"  gen batch={cfg.batch_size}, real batch={real_batch_size}")
            print(f"  global gen batch={global_gen_batch}, global real batch={global_real_batch}")
            print(f"  temps={cfg.temperatures}, pool_size={cfg.pool_size}")
            print(f"  more_features={cfg.more_features}")
            print(f"  encoder input={encoder_input_size}x{encoder_input_size}")
            print(f"  real feature cache batch={int(args.real_feature_batch_size or 256)}")
            print(f"  evaluate_every={cfg.evaluate_every}")
            print(f"  evaluate_every_seconds={evaluate_every_seconds}")
            print(f"  save_every={cfg.save_every}")
            print(f"  fid_eval_weights={','.join(fid_eval_weights)}")
            print(f"  skip_final_fid={args.skip_final_fid}")
            print(f"  deadline_epoch={deadline_epoch if deadline_epoch is not None else '<none>'}")
            sample_scope = f"  train_samples={effective_max_samples:,}/{total_available_samples:,}"
            if normalized_train_sample_ratio is not None:
                sample_scope += f" ({normalized_train_sample_ratio * 100.0:.2f}%)"
            if args.max_samples is not None:
                sample_scope += f", max_samples={args.max_samples:,}"
            print(sample_scope)

        fid_ref_dir = DEFAULT_IMAGENET32_REF_DIR
        precomp_feats, precomp_cjs, cached_num_samples = precompute_imagenet32_features(
            encoder=feat_encoder,
            source=data_source,
            device=device,
            batch_size=int(args.real_feature_batch_size or 256),
            max_samples=effective_max_samples,
            is_main=is_main,
            use_autocast=(device.type == "cuda"),
            use_channels_last=(device.type == "cuda"),
        )
        if is_main:
            fid_ref_dir = prepare_imagenet32_reference(
                data_source=data_source,
                output_dir=DEFAULT_IMAGENET32_REF_DIR,
                num_images=DEFAULT_NUM_FID_SAMPLES,
            )
        if use_ddp:
            sync_workers(host_sync_group)
        clear_imagenet32_array_cache(data_source)

        start_time = time.time()
        next_eval_time = None
        if evaluate_every_seconds > 0:
            initial_eval_time = start_time + evaluate_every_seconds
            next_eval_time = (
                broadcast_main_float(initial_eval_time if is_main else 0.0, device)
                if use_ddp
                else initial_eval_time
            )
        last_fid_step = None
        last_step = 0
        loss_log_slot_count = None
        amp_ctx = (
            lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        epoch = 0
        current_seed = int(args.data_seed) + epoch
        real_batch_iter = iter_sharded_shuffled_index_batches(
            num_samples=cached_num_samples,
            batch_size=real_batch_size,
            epoch_seed=current_seed,
            max_samples=effective_max_samples,
            rank=rank,
            world_size=world_size,
        )

        for step in range(1, cfg.total_steps + 1):
            last_step = step
            try:
                batch_indices = next(real_batch_iter)
            except StopIteration:
                epoch += 1
                current_seed = int(args.data_seed) + epoch
                real_batch_iter = iter_sharded_shuffled_index_batches(
                    num_samples=cached_num_samples,
                    batch_size=real_batch_size,
                    epoch_seed=current_seed,
                    max_samples=effective_max_samples,
                    rank=rank,
                    world_size=world_size,
                )
                try:
                    batch_indices = next(real_batch_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        "ImageNet32 iterator produced no full global batches. "
                        "Reduce world size or real batch size, or increase max_samples."
                    ) from exc

            pos_groups = []
            for group_idx, c_j in enumerate(precomp_cjs):
                feats = precomp_feats[group_idx].index_select(0, batch_indices)
                pos_groups.append((feats.to(device=device, non_blocking=True), c_j))

            z = torch.randn(cfg.batch_size, 3, 32, 32, device=device).to(
                memory_format=torch.channels_last
            )
            with amp_ctx():
                gen_images = model(z)

            with amp_ctx():
                gen_groups_raw = feat_encoder(gen_images.float())
            gen_groups = [(feat.float(), c_j) for feat, c_j in gen_groups_raw]
            if loss_log_slot_count is None:
                loss_log_slot_count = count_feature_slots(gen_groups)
                if is_main:
                    print(
                        f"  loss logging normalized by {loss_log_slot_count} feature slots "
                        f"(sum of encoder locations across groups)"
                    )

            if use_ddp:
                loss = multires_drift_loss_distributed(
                    gen_groups,
                    pos_groups,
                    temps=tuple(cfg.temperatures),
                    rank=rank,
                    world_size=world_size,
                    global_query_gather=cfg.drift_global_query_gather,
                )
            else:
                loss = drifting_loss_multires(
                    gen_groups,
                    pos_groups,
                    temps=tuple(cfg.temperatures),
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            ema.update(train_model)

            if step % cfg.log_every == 0 and is_main:
                logged_loss, _ = scaled_loss_for_logging(loss, gen_groups)
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
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

            if step % cfg.sample_every == 0 and is_main:
                train_model.eval()
                samples = drift_sample(train_model, 64, device)
                save_sample_grid(
                    samples,
                    os.path.join(out_dir, "samples", f"drift_step{step:07d}.png"),
                )
                train_model.train()
                print(f"  Saved raw sample grid at step {step}")
            if use_ddp and step % cfg.sample_every == 0:
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
                ckpt = build_training_checkpoint(step)
                torch.save(ckpt, checkpoint_path)
                torch.save(ckpt, os.path.join(out_dir, "checkpoints", "drift_latest.pt"))
                print(f"  Saved checkpoint at step {step}")
                del ckpt
            if should_evaluate and is_main:
                if checkpoint_path is None:
                    checkpoint_path = os.path.join(out_dir, "checkpoints", f"drift_step{step:07d}.pt")
                    ckpt = build_training_checkpoint(step)
                    torch.save(ckpt, checkpoint_path)
                    torch.save(ckpt, os.path.join(out_dir, "checkpoints", "drift_latest.pt"))
                    print(f"  Saved checkpoint at step {step}")
                    del ckpt
                eval_start = time.time()
                print(f"  Starting FID evaluation at step {step} ({','.join(fid_eval_weights)}).")
                fids = evaluate_requested_fids_from_checkpoint(
                    checkpoint_path,
                    fid_ref_dir,
                    device,
                    fid_eval_weights,
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
            if last_step >= cfg.total_steps:
                print(f"\nTraining complete. Reached step {last_step} in {elapsed / 3600.0:.1f} hours")
            else:
                print(f"\nTraining stopped at step {last_step} in {elapsed / 3600.0:.1f} hours")
            final_path = os.path.join(out_dir, "checkpoints", "drift_final.pt")
            torch.save(
                {
                    "step": last_step,
                    "model": train_model.state_dict(),
                    "ema": ema.state_dict(),
                    "config": {"unet": unet_cfg, "drift": cfg},
                },
                final_path,
            )
            print(f"Saved final checkpoint to {final_path}")
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
    finally:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", type=str, default="data/Imagenet32_train.zip")
    parser.add_argument("--output-dir", type=str, default="outputs/imagenet32_exact_ddp")
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov3",
        choices=[
            "dinov2-multires",
            "convnextv2",
            "mocov2",
            "dinov3",
            "dinov3-l",
            "dinov3-h+",
            "aimv2-l",
            "aimv2-h",
            "radio",
            "eva02",
            "siglip2",
            "clip",
        ],
    )
    parser.add_argument(
        "--encoder-size",
        type=int,
        default=112,
        help="Encoder input resolution (default 112).",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256, help="Generated batch size per GPU.")
    parser.add_argument(
        "--real-batch-size",
        type=int,
        default=None,
        help="Real ImageNet32 batch size per GPU. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--real-feature-batch-size",
        type=int,
        default=256,
        help="Batch size used while building the cached real-feature bank.",
    )
    parser.add_argument("--temps", type=str, default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--more-features", "--more-feautres", dest="more_features", action="store_true")
    parser.add_argument("--evaluate-every", type=int, default=None)
    parser.add_argument("--evaluate-every-seconds", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--fid-eval-weights", type=str, default="raw,ema")
    parser.add_argument("--skip-final-fid", action="store_true")
    parser.add_argument("--deadline-epoch", type=float, default=None)
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
        help="Torch device. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument("--no-compile-encoder", action="store_true")
    parser.add_argument("--large", action="store_true")
    train(parser.parse_args())
