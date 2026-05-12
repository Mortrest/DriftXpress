from __future__ import annotations

import math
import os
import shlex
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from data.imagenet import (
    clear_imagenet32_array_cache,
    iter_imagenet32_batches as materialized_imagenet32_batches,
    list_imagenet32_batch_names as materialized_imagenet32_batch_names,
    load_imagenet32_arrays,
    uint8_images_to_model_input,
)


def write_training_config_snapshot(
    out_dir,
    args,
    cfg,
    unet_cfg,
    encoder_input_size,
    real_batch_size,
    model_num_params,
    feat_encoder_num_params,
    model_compiled,
    feat_encoder_compiled,
):
    resolved_cfg = asdict(cfg)
    resolved_cfg.pop("nystrom_landmarks_per_class", None)
    resolved_cfg.pop("nystrom_ridge", None)
    resolved_cfg.pop("nystrom_landmark_seed", None)
    resolved_cfg.pop("nystrom_kmeans_iters", None)
    resolved_cfg.pop("nystrom_repulsion", None)
    resolved_cfg["encoder_input_size"] = encoder_input_size
    resolved_cfg["output_dir"] = out_dir
    resolved_cfg["large_model"] = bool(args.large)
    resolved_cfg["dataset"] = "imagenet32"
    resolved_cfg["data_source"] = str(Path(args.data_source).resolve())
    resolved_cfg["training_mode"] = "exact_precomputed_real_features"
    resolved_cfg["real_batch_size"] = int(real_batch_size)
    resolved_cfg["real_feature_batch_size"] = int(
        getattr(args, "real_feature_batch_size", 256) or 256
    )
    resolved_cfg["data_seed"] = int(args.data_seed)
    resolved_cfg["max_samples"] = args.max_samples
    resolved_cfg["train_sample_ratio"] = getattr(args, "train_sample_ratio", None)
    resolved_cfg["resolved_train_sample_ratio"] = getattr(
        args, "resolved_train_sample_ratio", None
    )
    resolved_cfg["effective_max_samples"] = getattr(args, "effective_max_samples", None)

    sections = [
        ("command", {"argv": " ".join(shlex.quote(arg) for arg in sys.argv)}),
        ("cli_args", vars(args)),
        ("resolved_multires_config", resolved_cfg),
        ("unet_config", asdict(unet_cfg)),
        (
            "model_info",
            {
                "model_num_params": model_num_params,
                "feature_encoder_num_params": feat_encoder_num_params,
                "model_compiled": bool(model_compiled),
                "feature_encoder_compiled": bool(feat_encoder_compiled),
            },
        ),
    ]

    lines = ["ImageNet Standard Drifting Training Configuration", ""]
    for section_name, values in sections:
        lines.append(f"[{section_name}]")
        for key, value in values.items():
            lines.append(f"{key}: {value}")
        lines.append("")

    path = os.path.join(out_dir, "training_config.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved training config snapshot to {path}")


def normalize_train_sample_ratio(train_sample_ratio):
    if train_sample_ratio is None:
        return None
    ratio = float(train_sample_ratio)
    if ratio <= 0:
        raise ValueError("train_sample_ratio must be positive.")
    if ratio > 1.0:
        if ratio > 100.0:
            raise ValueError("train_sample_ratio must be in (0, 1] or (0, 100].")
        ratio = ratio / 100.0
    return ratio


def resolve_effective_max_samples(total_samples, max_samples=None, train_sample_ratio=None):
    effective = int(total_samples)
    normalized_ratio = normalize_train_sample_ratio(train_sample_ratio)
    if normalized_ratio is not None:
        ratio_cap = int(math.floor(total_samples * normalized_ratio))
        if total_samples > 0:
            ratio_cap = max(1, ratio_cap)
        effective = min(effective, ratio_cap)
    if max_samples is not None:
        effective = min(effective, int(max_samples))
    return max(0, effective), normalized_ratio


def list_imagenet32_batch_names(source: Path) -> list[str]:
    return materialized_imagenet32_batch_names(source)


def iter_imagenet32_batches(source: Path, batch_order: list[str] | None = None):
    yield from materialized_imagenet32_batches(source, batch_order=batch_order)


@torch.no_grad()
def precompute_imagenet32_features(
    encoder,
    source: Path,
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    is_main: bool,
    use_autocast: bool,
    use_channels_last: bool,
):
    data, _ = load_imagenet32_arrays(source)
    total_samples = int(data.shape[0])
    if max_samples is not None:
        total_samples = min(total_samples, int(max_samples))

    if is_main:
        print(
            f"Pre-computing ImageNet real features on CPU "
            f"(samples={total_samples:,}, batch_size={batch_size})..."
        )

    all_feats = None
    group_dims = None
    offset = 0
    amp_ctx = (
        lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_autocast
        else nullcontext()
    )

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        images = torch.from_numpy(data[start:end]).view(-1, 3, 32, 32)
        images = uint8_images_to_model_input(images).to(device=device, non_blocking=True).contiguous()
        if use_channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        with amp_ctx():
            groups = encoder(images)

        if all_feats is None:
            all_feats = []
            group_dims = []
            total_bytes = 0
            for feat, c_j in groups:
                storage = torch.empty(
                    (total_samples, feat.shape[1], feat.shape[2]),
                    dtype=torch.float32,
                    device="cpu",
                )
                all_feats.append(storage)
                group_dims.append(int(c_j))
                total_bytes += storage.numel() * storage.element_size()
            if is_main:
                print(f"  allocated {total_bytes / 1e9:.2f} GB for cached real features")

        chunk = end - start
        for group_idx, (feat, _) in enumerate(groups):
            all_feats[group_idx][offset : offset + chunk].copy_(feat.float().cpu())
        offset += chunk

        if is_main and offset % 10000 == 0:
            print(f"  cached {offset:,}/{total_samples:,} real images")

    if all_feats is None or group_dims is None:
        raise ValueError("No ImageNet samples were available for real-feature caching.")

    if is_main:
        print("  real-feature cache complete.")

    return all_feats, group_dims, total_samples


def iter_sharded_shuffled_index_batches(
    num_samples: int,
    batch_size: int,
    epoch_seed: int,
    max_samples: int | None,
    rank: int,
    world_size: int,
):
    generator = torch.Generator(device="cpu").manual_seed(int(epoch_seed))
    total = int(num_samples)
    if max_samples is not None:
        total = min(total, int(max_samples))

    global_batch_size = int(batch_size) * int(world_size)
    if total < global_batch_size:
        return

    usable_total = (total // global_batch_size) * global_batch_size
    if usable_total <= 0:
        return

    sample_order = torch.randperm(total, generator=generator)[:usable_total]
    for start in range(0, usable_total, global_batch_size):
        shard_start = start + rank * batch_size
        shard_end = shard_start + batch_size
        yield sample_order[shard_start:shard_end]


__all__ = [
    "clear_imagenet32_array_cache",
    "iter_imagenet32_batches",
    "iter_sharded_shuffled_index_batches",
    "list_imagenet32_batch_names",
    "precompute_imagenet32_features",
    "resolve_effective_max_samples",
    "uint8_images_to_model_input",
    "write_training_config_snapshot",
]
