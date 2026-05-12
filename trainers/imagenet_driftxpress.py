from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
import random
import shlex
import sys
import time
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm.auto import tqdm

from common.config import MultiResDriftConfig, UNetConfig, UNetLargeConfig
from evaluation.fid import compute_fid
from data.imagenet import (
    iter_imagenet32_batches as materialized_imagenet32_batches,
    uint8_images_to_model_input,
)
from evaluation.sample import (
    cleanup_fid_images,
    drift_sample,
    generate_fid_images,
    save_sample_grid,
)
from models.ema import EMA
from models.unet import UNet
from methods.driftxpress import NystromStats
from features.encoders import build_encoder
from common.loss_reporting import count_feature_slots, format_scientific, scaled_loss_for_logging


CACHE_FORMAT_VERSION = 3
DEFAULT_NUM_FID_SAMPLES = 10_000
SUPPORTED_EVAL_WEIGHTS = ("raw", "ema")
DEFAULT_IMAGENET32_REF_DIR = (
    Path(__file__).resolve().parents[1] / "fid_reference" / "imagenet32_train_10000"
)
FID_EVAL_ARGS = SimpleNamespace(
    batch_size=256,
    fid_batch_size=50,
    fid_num_workers=0,
    fid_timeout=None,
    compile=False,
)


def _alternative_temperature_scale(temperature, D, eps=1e-8):
    return max(float(temperature) * math.sqrt(float(D)), eps)


def _alternative_laplacian_kernel_batched(x, y, temperature, eps=1e-8):
    """Batched Laplacian kernel with the CIFAR alternative sqrt(D) temperature scale."""
    D = x.shape[-1]
    tau_tilde = _alternative_temperature_scale(temperature, D, eps=eps)
    dist = torch.cdist(x.float(), y.float(), p=2)
    return torch.exp(-dist / tau_tilde)


def _alternative_inverse_sqrt_psd_batched(mats, eps=1e-8):
    eigvals, eigvecs = torch.linalg.eigh(mats.float())
    inv_sqrt_eigs = eigvals.clamp_min(eps).rsqrt()
    return (eigvecs * inv_sqrt_eigs.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


def _subsample_landmarks_for_scale(landmarks, max_landmarks=256):
    M = int(landmarks.shape[1])
    if M <= max_landmarks:
        return landmarks
    idx = torch.linspace(
        0,
        M - 1,
        steps=max_landmarks,
        device=landmarks.device,
    ).round().long()
    return landmarks.index_select(1, idx)


def move_nystrom_stats_to_device(stats: NystromStats, device: torch.device | str) -> NystromStats:
    """Move a Nyström cache block and preserve the CIFAR alternative feature scale metadata."""
    device = torch.device(device)
    if (
        stats.landmarks.device == device
        and stats.A.device == device
        and stats.global_totals.device == device
        and stats.global_weighted_points.device == device
    ):
        return stats

    moved = NystromStats(
        landmarks=stats.landmarks.to(device=device, dtype=torch.float32, non_blocking=True),
        A=stats.A.to(device=device, dtype=torch.float32, non_blocking=True),
        global_totals=stats.global_totals.to(device=device, dtype=torch.float32, non_blocking=True),
        global_weighted_points=stats.global_weighted_points.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        temperature=float(stats.temperature),
    )
    if hasattr(stats, "feature_scale"):
        moved.feature_scale = torch.as_tensor(stats.feature_scale).detach().cpu()
    return moved


def _alternative_compute_exact_repulsive_barycenter_batched(
    x,
    temperature=0.05,
    eps=1e-8,
    mask_self=True,
):
    """Exact repulsive barycenter with the CIFAR alternative sqrt(D) temperature scale."""
    x_f = x.float()
    D = x.shape[-1]
    tau_tilde = _alternative_temperature_scale(temperature, D, eps=eps)
    dist = torch.cdist(x_f, x_f, p=2)

    if mask_self:
        N = x.shape[1]
        dist = dist + torch.eye(N, device=x.device, dtype=dist.dtype).unsqueeze(0) * 1e6

    weights = torch.exp(-dist / tau_tilde)
    den = weights.sum(dim=2, keepdim=True).clamp_min(eps)
    return torch.bmm(weights, x_f) / den


def prepare_nystrom_landmarks_batched(
    public_landmarks,
    temperature=0.05,
    ridge=1e-4,
    eps=1e-8,
    device=None,
):
    """Prepare landmark tensors and A = (W + ridge I)^(-1/2)."""
    if public_landmarks.ndim != 3:
        raise ValueError("Expected public_landmarks with shape [M, L, D].")

    if device is None:
        device = public_landmarks.device

    landmarks_t = public_landmarks.transpose(0, 1).contiguous().to(device=device, dtype=torch.float32)
    L, M, _ = landmarks_t.shape
    W = _alternative_laplacian_kernel_batched(
        landmarks_t,
        landmarks_t,
        temperature=temperature,
        eps=eps,
    )
    eye = torch.eye(M, device=device, dtype=W.dtype).unsqueeze(0).expand(L, -1, -1)
    A = _alternative_inverse_sqrt_psd_batched(W + ridge * eye, eps=eps)
    return landmarks_t, A


def compute_nystrom_features_batched(x, landmarks, A, temperature=0.05, eps=1e-8):
    """Explicit Nyström features phi(x) = K(x, U) (W + ridge I)^(-1/2)."""
    K_xu = _alternative_laplacian_kernel_batched(
        x,
        landmarks,
        temperature=temperature,
        eps=eps,
    )
    return torch.bmm(K_xu, A)


def _compute_positive_terms_batched(x, stats, eps=1e-8):
    """Compute attractive Nyström numerator and denominator terms."""
    stats = move_nystrom_stats_to_device(stats, x.device)
    phi = compute_nystrom_features_batched(
        x=x.float(),
        landmarks=stats.landmarks,
        A=stats.A,
        temperature=stats.temperature,
        eps=eps,
    )
    pos_num = torch.bmm(phi, stats.global_weighted_points.float())
    pos_den = torch.sum(
        phi * stats.global_totals.unsqueeze(1).float(),
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    return phi, pos_num, pos_den


@torch.no_grad()
def _estimate_feature_scale_from_landmarks(
    sensitive_points,
    public_landmarks,
    eps=1e-8,
    batch_size=512,
    max_landmarks=256,
    device=None,
):
    """Estimate one shared feature scale from real features and Nyström landmarks."""
    if sensitive_points.ndim != 3:
        raise ValueError("Expected sensitive_points with shape [Np, L, D].")
    if public_landmarks.ndim != 3:
        raise ValueError("Expected public_landmarks with shape [M, L, D].")

    if device is None:
        device = sensitive_points.device

    D = int(sensitive_points.shape[-1])
    landmarks_t = public_landmarks.transpose(0, 1).contiguous()
    landmarks_t = _subsample_landmarks_for_scale(
        landmarks_t,
        max_landmarks=max_landmarks,
    ).to(device=device, dtype=torch.float32, non_blocking=True)

    total = torch.zeros((), device=device, dtype=torch.float32)
    count = 0
    Np = int(sensitive_points.shape[0])
    for start in range(0, Np, batch_size):
        end = min(start + batch_size, Np)
        batch = sensitive_points[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        batch_t = batch.transpose(0, 1).contiguous()
        dists = torch.cdist(batch_t, landmarks_t, p=2)
        total = total + dists.sum()
        count += dists.numel()

    if count == 0:
        raise ValueError("Cannot estimate a feature scale from an empty sensitive set.")
    return (total / count / math.sqrt(D)).detach().clamp(min=eps)


def _alternative_normalize_drift_batched(V, D=None, eps=1e-8):
    """Normalize a whole feature group with one shared drift scale, matching CIFAR."""
    if D is None:
        D = V.shape[-1]
    lambda_j = torch.sqrt((V.float().pow(2).sum(dim=-1) / D).mean()).detach()
    return V / (lambda_j + eps)


def compute_nystrom_drift_batched(
    x,
    stats,
    eps=1e-8,
    max_drift_norm=None,
    mask_self=True,
    repulsion_mode="nystrom",
):
    """Compute Nyström drift V(x_i) = b_pos(x_i) - b_neg(x_i)."""
    x_f = x.float()
    phi, pos_num, pos_den = _compute_positive_terms_batched(x=x_f, stats=stats, eps=eps)
    b_pos = pos_num / pos_den

    if repulsion_mode == "nystrom":
        neg_totals = phi.sum(dim=1)
        neg_weighted_points = torch.bmm(phi.transpose(1, 2), x_f)
        neg_num = torch.bmm(phi, neg_weighted_points)
        neg_den = torch.sum(phi * neg_totals.unsqueeze(1), dim=-1, keepdim=True)

        if mask_self:
            self_mass = torch.sum(phi * phi, dim=-1, keepdim=True)
            neg_num = neg_num - self_mass * x_f
            neg_den = neg_den - self_mass

        neg_den = neg_den.clamp_min(eps)
        b_neg = neg_num / neg_den
    elif repulsion_mode == "exact":
        b_neg = _alternative_compute_exact_repulsive_barycenter_batched(
            x=x_f,
            temperature=stats.temperature,
            eps=eps,
            mask_self=mask_self,
        )
    else:
        raise ValueError(f"Unsupported repulsion_mode: {repulsion_mode}")

    V = (b_pos - b_neg).to(x.dtype)

    if max_drift_norm is not None:
        norms = torch.linalg.vector_norm(V.float(), dim=-1, keepdim=True)
        scale = torch.clamp(float(max_drift_norm) / (norms + eps), max=1.0)
        V = V * scale.to(V.dtype)

    return V


def compute_nystrom_sharded_positive_barycenter_batched(
    x,
    stats_by_shard,
    eps=1e-8,
    distributed_shards=False,
):
    """Combine shard-local attractive terms into a single barycenter."""
    pos_num_total = None
    pos_den_total = None

    for stats in stats_by_shard.values():
        _, pos_num, pos_den = _compute_positive_terms_batched(x=x, stats=stats, eps=eps)
        if pos_num_total is None:
            pos_num_total = pos_num
            pos_den_total = pos_den
        else:
            pos_num_total = pos_num_total + pos_num
            pos_den_total = pos_den_total + pos_den

    if pos_num_total is None or pos_den_total is None:
        raise ValueError("stats_by_shard must not be empty.")

    if distributed_shards:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "distributed_shards=True requires torch.distributed to be initialized."
            )
        dist.all_reduce(pos_num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(pos_den_total, op=dist.ReduceOp.SUM)

    return pos_num_total / pos_den_total.clamp_min(eps)


def gather_queries_global_batch(x):
    """Gather query features across ranks and return the local slice bounds."""
    local_batch = int(x.shape[1])
    if not dist.is_available() or not dist.is_initialized():
        return x, slice(0, local_batch)

    world_size = dist.get_world_size()
    if world_size == 1:
        return x, slice(0, local_batch)

    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.contiguous())
    rank = dist.get_rank()
    local_start = rank * local_batch
    local_end = local_start + local_batch
    return torch.cat(gathered, dim=1), slice(local_start, local_end)


def compute_nystrom_drift_multitemp_batched(
    x,
    stats_by_temp,
    temps=(0.02, 0.05, 0.2),
    eps=1e-8,
    max_drift_norm=None,
    repulsion_mode="nystrom",
    distributed_queries=False,
):
    """Aggregate Nyström drifts over multiple temperatures."""
    V_total = torch.zeros_like(x)
    x_eval = x
    local_query_slice = slice(0, int(x.shape[1]))
    if distributed_queries:
        x_eval, local_query_slice = gather_queries_global_batch(x)

    for temp in temps:
        temp = float(temp)
        if temp not in stats_by_temp:
            raise KeyError(f"Missing Nyström cache for temperature {temp}.")

        V_tau = compute_nystrom_drift_batched(
            x=x_eval,
            stats=stats_by_temp[temp],
            eps=eps,
            max_drift_norm=max_drift_norm,
            mask_self=True,
            repulsion_mode=repulsion_mode,
        )
        if distributed_queries:
            V_tau = V_tau[:, local_query_slice, :]
        V_total = V_total + _alternative_normalize_drift_batched(V_tau, eps=eps)

    return V_total


def compute_nystrom_drift_multitemp_sharded_batched(
    x,
    stats_by_shard,
    temps=(0.02, 0.05, 0.2),
    eps=1e-8,
    max_drift_norm=None,
    repulsion_mode="exact",
    distributed_shards=False,
    distributed_queries=False,
):
    """Aggregate class-sharded Nyström drifts over multiple temperatures."""
    if repulsion_mode != "exact":
        raise ValueError("Class-sharded Nyström currently supports only exact repulsion.")

    V_total = torch.zeros_like(x)
    x_global = x
    local_query_slice = slice(0, int(x.shape[1]))
    gather_queries = distributed_queries or distributed_shards
    if gather_queries:
        x_global, local_query_slice = gather_queries_global_batch(x)

    for temp in temps:
        temp = float(temp)
        shard_stats_at_temp = {}
        for shard_id, stats_by_temp in stats_by_shard.items():
            if temp not in stats_by_temp:
                raise KeyError(f"Missing Nyström cache for shard={shard_id}, temperature={temp}.")
            shard_stats_at_temp[shard_id] = stats_by_temp[temp]

        b_pos = compute_nystrom_sharded_positive_barycenter_batched(
            x=x_global,
            stats_by_shard=shard_stats_at_temp,
            eps=eps,
            distributed_shards=distributed_shards,
        )
        if gather_queries:
            b_pos = b_pos[:, local_query_slice, :]
        neg_queries = x_global if gather_queries else x
        b_neg = _alternative_compute_exact_repulsive_barycenter_batched(
            x=neg_queries.float(),
            temperature=temp,
            eps=eps,
            mask_self=True,
        )
        if gather_queries:
            b_neg = b_neg[:, local_query_slice, :]
        V_tau = (b_pos - b_neg).to(x.dtype)

        if max_drift_norm is not None:
            norms = torch.linalg.vector_norm(V_tau.float(), dim=-1, keepdim=True)
            scale = torch.clamp(float(max_drift_norm) / (norms + eps), max=1.0)
            V_tau = V_tau * scale.to(V_tau.dtype)

        V_total = V_total + _alternative_normalize_drift_batched(V_tau, eps=eps)

    return V_total


def _feature_scale_from_landmarks(
    x,
    landmarks,
    eps=1e-8,
    max_landmarks=256,
    query_chunk_size=64,
):
    """Approximate the normal baseline's shared feature scale using landmarks."""
    D = x.shape[-1]
    landmarks = _subsample_landmarks_for_scale(landmarks, max_landmarks=max_landmarks)
    landmarks = landmarks.to(device=x.device, dtype=torch.float32, non_blocking=True)
    total = torch.zeros((), device=x.device, dtype=torch.float32)
    count = 0

    for start in range(0, int(x.shape[1]), query_chunk_size):
        end = min(start + query_chunk_size, int(x.shape[1]))
        dists = torch.cdist(x[:, start:end].float(), landmarks, p=2)
        total = total + dists.sum()
        count += dists.numel()

    if count == 0:
        raise ValueError("Cannot compute feature scale from an empty query batch.")
    return (total / count / math.sqrt(D)).detach().clamp(min=eps)


def _feature_scale_from_stats_by_temp(
    x,
    stats_by_temp,
    temps,
    eps=1e-8,
    distributed_queries=False,
):
    for temp in temps:
        stats = stats_by_temp.get(float(temp))
        if stats is not None:
            scale = getattr(stats, "feature_scale", None)
            if scale is not None:
                return torch.as_tensor(scale, device=x.device, dtype=torch.float32).detach().clamp(min=eps)
            x_eval = x
            if distributed_queries:
                x_eval, _ = gather_queries_global_batch(x)
            return _feature_scale_from_landmarks(x_eval, stats.landmarks, eps=eps)
    first_stats = next(iter(stats_by_temp.values()))
    scale = getattr(first_stats, "feature_scale", None)
    if scale is not None:
        return torch.as_tensor(scale, device=x.device, dtype=torch.float32).detach().clamp(min=eps)
    x_eval = x
    if distributed_queries:
        x_eval, _ = gather_queries_global_batch(x)
    return _feature_scale_from_landmarks(x_eval, first_stats.landmarks, eps=eps)


def _feature_scale_from_sharded_stats(
    x,
    stats_by_shard,
    temps,
    eps=1e-8,
    distributed_queries=False,
):
    landmark_chunks = []
    for stats_by_temp in stats_by_shard.values():
        for temp in temps:
            stats = stats_by_temp.get(float(temp))
            if stats is not None:
                scale = getattr(stats, "feature_scale", None)
                if scale is not None:
                    return torch.as_tensor(scale, device=x.device, dtype=torch.float32).detach().clamp(min=eps)
                landmark_chunks.append(_subsample_landmarks_for_scale(stats.landmarks, max_landmarks=64))
                break

    if not landmark_chunks:
        raise ValueError("stats_by_shard must not be empty.")
    x_eval = x
    if distributed_queries:
        x_eval, _ = gather_queries_global_batch(x)
    landmarks = torch.cat(landmark_chunks, dim=1)
    return _feature_scale_from_landmarks(
        x_eval,
        landmarks,
        eps=eps,
        max_landmarks=256,
    )


def drifting_loss_multires_nystrom(
    gen_groups,
    stats_groups,
    temps=(0.02, 0.05, 0.2),
    eps=1e-8,
    max_drift_norm=None,
    repulsion_mode="nystrom",
    distributed_queries=False,
):
    """Multi-resolution Nyström loss in normalized feature space, matching CIFAR."""
    if len(gen_groups) != len(stats_groups):
        raise ValueError("gen_groups and stats_groups must have the same length.")

    total_loss = gen_groups[0][0].new_tensor(0.0)

    for (gen_feat, _), stats_by_temp in zip(gen_groups, stats_groups):
        with torch.no_grad():
            gen_t = gen_feat.detach().transpose(0, 1).contiguous()
            S = _feature_scale_from_stats_by_temp(
                gen_t,
                stats_by_temp,
                temps=temps,
                eps=eps,
                distributed_queries=distributed_queries,
            )
            V = compute_nystrom_drift_multitemp_batched(
                x=gen_t / S,
                stats_by_temp=stats_by_temp,
                temps=temps,
                eps=eps,
                max_drift_norm=max_drift_norm,
                repulsion_mode=repulsion_mode,
                distributed_queries=distributed_queries,
            )
            target = (gen_t / S + V).transpose(0, 1)

        gen_n = gen_feat / S
        loss_group = (gen_n - target).pow(2).mean(dim=(0, 2)).sum()
        total_loss = total_loss + loss_group

    return total_loss


def drifting_loss_multires_nystrom_sharded(
    gen_groups,
    stats_groups,
    temps=(0.02, 0.05, 0.2),
    eps=1e-8,
    max_drift_norm=None,
    repulsion_mode="exact",
    distributed_shards=False,
    distributed_queries=False,
):
    """Class-sharded multi-resolution Nyström loss in normalized feature space, matching CIFAR."""
    if len(gen_groups) != len(stats_groups):
        raise ValueError("gen_groups and stats_groups must have the same length.")

    total_loss = gen_groups[0][0].new_tensor(0.0)
    effective_distributed_queries = distributed_queries or distributed_shards

    for (gen_feat, _), stats_by_shard in zip(gen_groups, stats_groups):
        with torch.no_grad():
            gen_t = gen_feat.detach().transpose(0, 1).contiguous()
            S = _feature_scale_from_sharded_stats(
                gen_t,
                stats_by_shard,
                temps=temps,
                eps=eps,
                distributed_queries=effective_distributed_queries,
            )
            V = compute_nystrom_drift_multitemp_sharded_batched(
                x=gen_t / S,
                stats_by_shard=stats_by_shard,
                temps=temps,
                eps=eps,
                max_drift_norm=max_drift_norm,
                repulsion_mode=repulsion_mode,
                distributed_shards=distributed_shards,
                distributed_queries=effective_distributed_queries,
            )
            target = (gen_t / S + V).transpose(0, 1)

        gen_n = gen_feat / S
        loss_group = (gen_n - target).pow(2).mean(dim=(0, 2)).sum()
        total_loss = total_loss + loss_group

    return total_loss


def get_host_sync_timeout():
    timeout_seconds = int(os.environ.get("HOST_SYNC_TIMEOUT_SEC", "7200"))
    if timeout_seconds <= 0:
        raise ValueError("HOST_SYNC_TIMEOUT_SEC must be positive.")
    return timedelta(seconds=timeout_seconds)


HOST_SYNC_TIMEOUT = get_host_sync_timeout()


class NullProgressBar:
    def update(self, n=1):
        return None

    def set_postfix_str(self, s, refresh=True):
        return None

    def close(self):
        return None


def make_progress_bar(total: int | None, desc: str, unit: str = "img"):
    if not sys.stderr.isatty():
        return NullProgressBar()
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def write_training_config_snapshot(
    out_dir,
    args,
    cfg,
    unet_cfg,
    encoder_input_size,
    model_num_params,
    feat_encoder_num_params,
    summary_cache_path,
    selected_class_ids,
    rank0_selected_class_ids=None,
):
    """Write the exact resolved training config for this run."""
    resolved_cfg = asdict(cfg)
    resolved_cfg.pop("nystrom_landmarks_per_class", None)
    resolved_cfg.pop("nystrom_kmeans_iters", None)
    resolved_cfg["encoder_input_size"] = encoder_input_size
    resolved_cfg["output_dir"] = out_dir
    resolved_cfg["large_model"] = bool(args.large)
    resolved_cfg["dataset"] = "imagenet32"
    resolved_cfg["data_source"] = str(Path(args.data_source).resolve())
    resolved_cfg["feature_batch_size"] = int(args.feature_batch_size)
    resolved_cfg["nystrom_num_landmarks"] = int(args.nystrom_num_landmarks)
    resolved_cfg["nystrom_landmark_strategy"] = "random_class_subset_kmeans_per_class"
    resolved_cfg["nystrom_kernel_temperature_scale"] = "temperature * sqrt(D)"
    resolved_cfg["nystrom_feature_normalization"] = "shared feature_scale from real features and landmarks"
    resolved_cfg["nystrom_loss_aggregation"] = "sum per-location MSE-style group losses"
    resolved_cfg["nystrom_subset_num_classes"] = int(args.nystrom_subset_num_classes)
    resolved_cfg["nystrom_landmarks_per_class"] = int(args.nystrom_landmarks_per_class)
    resolved_cfg["nystrom_landmark_seed"] = int(cfg.nystrom_landmark_seed)
    resolved_cfg["nystrom_kmeans_iters"] = int(cfg.nystrom_kmeans_iters)
    resolved_cfg["selected_class_ids"] = [int(class_id) for class_id in selected_class_ids]
    if rank0_selected_class_ids is not None:
        resolved_cfg["rank0_selected_class_ids"] = [
            int(class_id) for class_id in rank0_selected_class_ids
        ]
    resolved_cfg["restrict_training_to_selected_classes"] = bool(
        args.restrict_training_to_selected_classes
    )
    resolved_cfg["max_samples"] = args.max_samples
    resolved_cfg["summary_cache_path"] = str(summary_cache_path)
    resolved_cfg["rebuild_summary_cache"] = bool(args.rebuild_summary_cache)

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
                "model_compiled": True,
                "feature_encoder_compiled": not args.no_compile_encoder,
            },
        ),
    ]

    lines = ["ImageNet DriftXpress Training Configuration", ""]
    for section_name, values in sections:
        lines.append(f"[{section_name}]")
        for key, value in values.items():
            lines.append(f"{key}: {value}")
        lines.append("")

    path = os.path.join(out_dir, "training_config.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved training config snapshot to {path}")


def append_training_log(log_path, step, loss="", elapsed="", images_per_sec="", fid_raw="", fid_ema=""):
    """Append a training metric row to the CSV log."""
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([step, loss, elapsed, images_per_sec, fid_raw, fid_ema])


def append_fid_log(log_path, step, fid_raw="", fid_ema=""):
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([step, fid_raw, fid_ema, fid_raw])


def normalize_eval_weights(eval_weights):
    normalized = str(eval_weights).strip().lower()
    if normalized not in SUPPORTED_EVAL_WEIGHTS:
        raise ValueError(
            f"Unsupported eval weights '{eval_weights}'. Expected one of {SUPPORTED_EVAL_WEIGHTS}."
        )
    return normalized


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


def _strip_prefix_if_present(state_dict, prefix="_orig_mod."):
    if not state_dict:
        return state_dict
    if not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


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


def is_distributed():
    return dist.is_available() and dist.is_initialized()


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


def set_process_seed(rank):
    base_seed = int(os.environ.get("TRAIN_SEED", "0"))
    seed = base_seed + int(rank)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def broadcast_object_from_main(value, is_main):
    if not is_distributed():
        return value
    object_list = [value if is_main else None]
    dist.broadcast_object_list(object_list, src=0)
    return object_list[0]


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


def prepare_imagenet32_reference(
    data_source: Path,
    output_dir: Path,
    num_images: int = DEFAULT_NUM_FID_SAMPLES,
):
    output_dir = Path(output_dir)
    existing = sorted(output_dir.glob("*.png"))
    if len(existing) == num_images:
        print(f"Using existing ImageNet reference set: {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    for png in existing:
        png.unlink()

    saved = 0
    for _, batch in iter_imagenet32_batches(Path(data_source)):
        images = torch.from_numpy(batch["data"]).view(-1, 3, 32, 32).float().div_(255.0)
        for image in images:
            save_image(image, output_dir / f"{saved:05d}.png")
            saved += 1
            if saved % 1000 == 0:
                print(f"  Saved {saved}/{num_images} ImageNet reference images")
            if saved >= num_images:
                return output_dir
        del batch, images

    raise ValueError(
        f"ImageNet source {data_source} only provided {saved} images, fewer than {num_images}."
    )


def load_eval_model_from_checkpoint(checkpoint_path, device, eval_weights):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    unet_cfg = checkpoint["config"]["unet"]
    model = UNet(
        in_ch=unet_cfg.in_ch,
        out_ch=unet_cfg.out_ch,
        base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult,
        num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions,
        dropout=unet_cfg.dropout,
        num_heads=unet_cfg.num_heads,
    ).to(device)

    eval_weights = normalize_eval_weights(eval_weights)
    state_key = "model" if eval_weights == "raw" else "ema"
    if state_key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain {eval_weights} weights.")

    state_dict = _strip_prefix_if_present(checkpoint[state_key])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_fid_eval_args(eval_weights):
    return SimpleNamespace(
        batch_size=FID_EVAL_ARGS.batch_size,
        fid_batch_size=FID_EVAL_ARGS.fid_batch_size,
        fid_num_workers=FID_EVAL_ARGS.fid_num_workers,
        fid_timeout=FID_EVAL_ARGS.fid_timeout,
        compile=FID_EVAL_ARGS.compile,
        eval_weights=normalize_eval_weights(eval_weights),
    )


def evaluate_fid_from_checkpoint(checkpoint_path, ref_dir, device, eval_weights):
    checkpoint_path = Path(checkpoint_path).resolve()
    eval_weights = normalize_eval_weights(eval_weights)
    fake_dir = checkpoint_path.parent.parent / "evaluation" / f"{checkpoint_path.stem}_fid_10000_{eval_weights}"
    fake_dir.mkdir(parents=True, exist_ok=True)
    for png in fake_dir.glob("*.png"):
        png.unlink()

    eval_args = build_fid_eval_args(eval_weights)
    model = load_eval_model_from_checkpoint(checkpoint_path, device=device, eval_weights=eval_weights)
    if eval_args.compile and device.type == "cuda":
        model = torch.compile(model)

    removed_generated_images = 0
    try:
        print(
            f"Generating {DEFAULT_NUM_FID_SAMPLES} samples using {eval_weights} weights to {fake_dir} ..."
        )
        generate_fid_images(
            model=model,
            n_images=DEFAULT_NUM_FID_SAMPLES,
            output_dir=str(fake_dir),
            device=device,
            sample_fn=drift_sample,
            batch_size=eval_args.batch_size,
        )

        device_str = f"{device.type}:0" if device.type == "cuda" and device.index is None else str(device)
        fid = compute_fid(
            str(Path(ref_dir).resolve()),
            str(fake_dir.resolve()),
            device=device_str,
            batch_size=eval_args.fid_batch_size,
            num_workers=eval_args.fid_num_workers,
            timeout=eval_args.fid_timeout,
        )
        if fid is None:
            raise RuntimeError(f"FID computation failed for checkpoint {checkpoint_path}.")
    finally:
        removed_generated_images = cleanup_fid_images(fake_dir)

    result = {
        "checkpoint": str(checkpoint_path),
        "weights": eval_weights,
        "dataset": "imagenet32",
        "reference_dir": str(Path(ref_dir).resolve()),
        "generated_dir": str(fake_dir.resolve()),
        "num_generated_images_removed": removed_generated_images,
        "num_samples": DEFAULT_NUM_FID_SAMPLES,
        "fid": fid,
    }
    result_path = fake_dir.parent / f"{fake_dir.name}_fid.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved FID result to {result_path}")
    return fid


def evaluate_all_fids_from_checkpoint(checkpoint_path, ref_dir, device):
    return {
        "raw": evaluate_fid_from_checkpoint(checkpoint_path, ref_dir, device, "raw"),
        "ema": evaluate_fid_from_checkpoint(checkpoint_path, ref_dir, device, "ema"),
    }


def evaluate_requested_fids_from_checkpoint(checkpoint_path, ref_dir, device, eval_weights):
    return {
        weight_name: evaluate_fid_from_checkpoint(checkpoint_path, ref_dir, device, weight_name)
        for weight_name in eval_weights
    }


def make_summary_cache_metadata(
    data_source: Path,
    cfg: MultiResDriftConfig,
    encoder_input_size: int,
    num_landmarks: int,
    max_samples: int | None,
    subset_num_classes: int,
    landmarks_per_class: int,
    selected_class_ids: list[int],
    restrict_training_to_selected_classes: bool,
) -> dict[str, object]:
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset": "imagenet32",
        "data_source": str(Path(data_source).resolve()),
        "encoder": cfg.encoder,
        "encoder_input_size": int(encoder_input_size),
        "pool_size": int(cfg.pool_size),
        "more_features": bool(cfg.more_features),
        "nystrom_num_landmarks": int(num_landmarks),
        "nystrom_landmark_strategy": "random_class_subset_kmeans_per_class",
        "nystrom_kernel_temperature_scale": "sqrtD",
        "nystrom_feature_normalization": "shared_feature_scale",
        "nystrom_loss_aggregation": "sum_per_location",
        "nystrom_subset_num_classes": int(subset_num_classes),
        "nystrom_landmarks_per_class": int(landmarks_per_class),
        "nystrom_landmark_seed": int(cfg.nystrom_landmark_seed),
        "nystrom_kmeans_iters": int(cfg.nystrom_kmeans_iters),
        "selected_class_ids": [int(class_id) for class_id in selected_class_ids],
        "restrict_training_to_selected_classes": bool(restrict_training_to_selected_classes),
        "temps": [float(temp) for temp in cfg.temperatures],
        "ridge": float(cfg.nystrom_ridge),
        "max_samples": max_samples,
    }
    if cfg.nystrom_shard_by_class:
        metadata["nystrom_shard_by_class"] = True
    return metadata


def make_summary_cache_path(
    metadata: dict[str, object],
    explicit_path: str | None,
) -> Path:
    if explicit_path:
        return Path(explicit_path)

    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    filename = (
        f"imagenet32_{metadata['encoder']}_m{metadata['nystrom_num_landmarks']}_"
        f"in{metadata['encoder_input_size']}_pool{metadata['pool_size']}_"
        f"mf{int(bool(metadata.get('more_features', False)))}_{digest}.pt"
    )
    return Path("outputs") / "nystrom_cache" / filename


def make_rank_local_explicit_cache_path(
    explicit_path: str | None,
    rank: int,
    world_size: int,
) -> str | None:
    if explicit_path is None or world_size <= 1:
        return explicit_path

    base_path = Path(explicit_path)
    suffix = "".join(base_path.suffixes)
    stem = base_path.name[: -len(suffix)] if suffix else base_path.name
    rank_name = f"{stem}_rank{rank}of{world_size}{suffix}"
    return str(base_path.with_name(rank_name))


def serialize_stats_by_temp(stats_by_temp: dict[float, NystromStats]) -> dict[str, object]:
    if not stats_by_temp:
        raise ValueError("Encountered an empty stats group while serializing.")

    first_stats = next(iter(stats_by_temp.values()))
    feature_scale = getattr(first_stats, "feature_scale", None)
    temps_payload: dict[float, dict[str, torch.Tensor | float]] = {}
    for temp, stats in stats_by_temp.items():
        temps_payload[float(temp)] = {
            "A": stats.A.detach().cpu(),
            "global_totals": stats.global_totals.detach().cpu(),
            "global_weighted_points": stats.global_weighted_points.detach().cpu(),
            "temperature": float(stats.temperature),
        }

    return {
        "landmarks": first_stats.landmarks.detach().cpu(),
        "feature_scale": None if feature_scale is None else torch.as_tensor(feature_scale).detach().cpu(),
        "temps": temps_payload,
    }


def restore_stats_by_temp(
    group_payload: dict[str, object],
    device: torch.device,
) -> dict[float, NystromStats]:
    landmarks_cpu = group_payload["landmarks"]
    if not isinstance(landmarks_cpu, torch.Tensor):
        raise TypeError("Invalid cache payload: missing landmarks tensor.")
    landmarks = landmarks_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    feature_scale = group_payload.get("feature_scale")

    temps_payload = group_payload["temps"]
    if not isinstance(temps_payload, dict):
        raise TypeError("Invalid cache payload: missing per-temperature stats.")

    stats_by_temp: dict[float, NystromStats] = {}
    for temp, temp_payload in temps_payload.items():
        temp_value = float(temp)
        if not isinstance(temp_payload, dict):
            raise TypeError("Invalid cache payload: malformed temperature entry.")
        stats = NystromStats(
            landmarks=landmarks,
            A=temp_payload["A"].to(device=device, dtype=torch.float32, non_blocking=True),
            global_totals=temp_payload["global_totals"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ),
            global_weighted_points=temp_payload["global_weighted_points"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ),
            temperature=float(temp_payload.get("temperature", temp_value)),
        )
        if feature_scale is not None:
            stats.feature_scale = torch.as_tensor(feature_scale).detach().cpu()
        stats_by_temp[temp_value] = stats

    return stats_by_temp


def serialize_stats_groups(
    stats_groups: list[dict[float, NystromStats]] | list[dict[int, dict[float, NystromStats]]],
) -> list[dict[str, object]]:
    serialized_groups = []
    if not stats_groups:
        return serialized_groups

    first_group = stats_groups[0]
    if not first_group:
        raise ValueError("Encountered an empty stats group while serializing.")
    first_value = next(iter(first_group.values()))

    if isinstance(first_value, NystromStats):
        for stats_by_temp in stats_groups:
            serialized_groups.append(serialize_stats_by_temp(stats_by_temp))
        return serialized_groups

    for stats_by_class in stats_groups:
        shards_payload: dict[int, dict[str, object]] = {}
        for class_id, stats_by_temp in stats_by_class.items():
            shards_payload[int(class_id)] = serialize_stats_by_temp(stats_by_temp)
        serialized_groups.append({"shards": shards_payload})
    return serialized_groups


def restore_stats_groups(
    serialized_groups: list[dict[str, object]],
    device: torch.device,
) -> list[dict[float, NystromStats]] | list[dict[int, dict[float, NystromStats]]]:
    stats_groups = []

    for group_payload in serialized_groups:
        if "shards" in group_payload:
            shards_payload = group_payload["shards"]
            if not isinstance(shards_payload, dict):
                raise TypeError("Invalid cache payload: malformed shard entry.")
            stats_by_class: dict[int, dict[float, NystromStats]] = {}
            for class_id, shard_payload in shards_payload.items():
                if not isinstance(shard_payload, dict):
                    raise TypeError("Invalid cache payload: malformed shard payload.")
                stats_by_class[int(class_id)] = restore_stats_by_temp(shard_payload, device=device)
            stats_groups.append(stats_by_class)
        else:
            stats_groups.append(restore_stats_by_temp(group_payload, device=device))

    return stats_groups


def save_summary_cache(
    cache_path: Path,
    metadata: dict[str, object],
    stats_groups: list[dict[float, NystromStats]] | list[dict[int, dict[float, NystromStats]]],
    summarized_samples: int,
) -> None:
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "metadata": metadata,
        "summarized_samples": int(summarized_samples),
        "stats_groups": serialize_stats_groups(stats_groups),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def load_summary_cache(
    cache_path: Path,
    expected_metadata: dict[str, object],
    device: torch.device,
) -> tuple[
    list[dict[float, NystromStats]] | list[dict[int, dict[float, NystromStats]]],
    int | None,
]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported cache format version: {payload.get('format_version')}"
        )

    cached_metadata = payload.get("metadata")
    if not isinstance(cached_metadata, dict):
        raise TypeError("Invalid cache payload: missing metadata.")

    mismatches = []
    for key, expected_value in expected_metadata.items():
        if cached_metadata.get(key) != expected_value:
            mismatches.append(
                f"{key} expected {expected_value!r}, found {cached_metadata.get(key)!r}"
            )
    if mismatches:
        raise ValueError("Cache metadata mismatch: " + "; ".join(mismatches))

    stats_groups = restore_stats_groups(payload["stats_groups"], device=device)
    summarized_samples = payload.get("summarized_samples")
    return stats_groups, int(summarized_samples) if summarized_samples is not None else None


def iter_imagenet32_batches(source: Path):
    yield from materialized_imagenet32_batches(source)


def count_samples(source: Path) -> int:
    total = 0
    for _, batch in iter_imagenet32_batches(source):
        total += int(len(batch["labels"]))
        del batch
    return total


def iter_model_input_batches(
    source: Path,
    batch_size: int,
    max_samples: int | None,
):
    yielded = 0
    for _, batch in iter_imagenet32_batches(source):
        flat = batch["data"]
        batch_total = int(flat.shape[0])
        if max_samples is not None:
            batch_total = min(batch_total, max_samples - yielded)
        for start in range(0, batch_total, batch_size):
            end = min(start + batch_size, batch_total)
            images = torch.from_numpy(flat[start:end]).view(-1, 3, 32, 32)
            yield uint8_images_to_model_input(images)
        yielded += batch_total
        del batch
        if max_samples is not None and yielded >= max_samples:
            break


def encode_landmark_groups(
    encoder: torch.nn.Module,
    landmark_images: torch.Tensor,
    device: torch.device,
    batch_size: int,
    progress_desc: str | None = "Encoding landmarks",
) -> list[torch.Tensor]:
    features_by_group: list[list[torch.Tensor]] | None = None
    amp_enabled = device.type == "cuda"
    pbar = (
        make_progress_bar(
            total=int(landmark_images.shape[0]),
            desc=progress_desc,
        )
        if progress_desc is not None else NullProgressBar()
    )

    try:
        with torch.no_grad():
            for start in range(0, landmark_images.shape[0], batch_size):
                end = min(start + batch_size, landmark_images.shape[0])
                images = uint8_images_to_model_input(landmark_images[start:end]).to(
                    device=device,
                    non_blocking=True,
                )
                with (
                    torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if amp_enabled else nullcontext()
                ):
                    groups = encoder(images)

                if features_by_group is None:
                    features_by_group = [[] for _ in groups]

                for group_idx, (feat, _) in enumerate(groups):
                    features_by_group[group_idx].append(feat.float().cpu())

                pbar.update(end - start)
                del images, groups
    finally:
        pbar.close()

    if features_by_group is None:
        raise RuntimeError("No landmark features were produced.")

    return [torch.cat(group_chunks, dim=0) for group_chunks in features_by_group]


def count_class_samples(
    source: Path,
    max_samples: int | None,
) -> tuple[dict[int, int], int]:
    class_counts: dict[int, int] = {}
    yielded = 0
    pbar = make_progress_bar(total=max_samples, desc="Counting classes")

    try:
        for _, batch in iter_imagenet32_batches(source):
            labels = batch["labels"]
            batch_total = int(len(labels))
            if max_samples is not None:
                batch_total = min(batch_total, max_samples - yielded)

            for label in labels[:batch_total]:
                class_id = int(label)
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

            yielded += batch_total
            pbar.update(batch_total)
            del batch
            if max_samples is not None and yielded >= max_samples:
                break
    finally:
        pbar.close()

    return class_counts, yielded


def select_random_class_subset(
    class_counts: dict[int, int],
    num_classes: int,
    min_samples_per_class: int,
    seed: int,
) -> list[int]:
    eligible_classes = [
        int(class_id)
        for class_id, count in sorted(class_counts.items())
        if int(count) >= int(min_samples_per_class)
    ]
    if len(eligible_classes) < int(num_classes):
        raise ValueError(
            f"Requested {num_classes} classes with at least {min_samples_per_class} samples each, "
            f"but only {len(eligible_classes)} classes are eligible."
        )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(len(eligible_classes), generator=generator).tolist()
    selected = [eligible_classes[idx] for idx in perm[: int(num_classes)]]
    return sorted(selected)


def select_random_class_partitions(
    class_counts: dict[int, int],
    num_classes_per_rank: int,
    min_samples_per_class: int,
    seed: int,
    world_size: int,
) -> tuple[list[int], list[list[int]]]:
    eligible_classes = [
        int(class_id)
        for class_id, count in sorted(class_counts.items())
        if int(count) >= int(min_samples_per_class)
    ]
    total_num_classes = int(num_classes_per_rank) * int(world_size)
    if len(eligible_classes) < total_num_classes:
        raise ValueError(
            f"Requested {total_num_classes} unique classes "
            f"({num_classes_per_rank} per rank across {world_size} ranks) with at least "
            f"{min_samples_per_class} samples each, but only {len(eligible_classes)} classes are eligible."
        )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(len(eligible_classes), generator=generator).tolist()
    selected_in_rank_order = [eligible_classes[idx] for idx in perm[:total_num_classes]]
    rank_class_ids = []
    for rank_idx in range(int(world_size)):
        start = rank_idx * int(num_classes_per_rank)
        end = start + int(num_classes_per_rank)
        rank_class_ids.append(sorted(selected_in_rank_order[start:end]))
    return sorted(selected_in_rank_order), rank_class_ids


def collect_class_images(
    source: Path,
    selected_class_ids: list[int],
    max_samples: int | None,
    total_samples: int | None,
) -> dict[int, torch.Tensor]:
    selected_set = set(int(class_id) for class_id in selected_class_ids)
    images_by_class = {int(class_id): [] for class_id in selected_class_ids}
    yielded = 0
    pbar = make_progress_bar(total=total_samples, desc="Collecting subset")

    try:
        for _, batch in iter_imagenet32_batches(source):
            labels = batch["labels"]
            flat = batch["data"]
            batch_total = int(len(labels))
            if max_samples is not None:
                batch_total = min(batch_total, max_samples - yielded)
            if batch_total <= 0:
                del batch
                break

            images = flat[:batch_total].reshape(-1, 3, 32, 32)
            for local_idx, label in enumerate(labels[:batch_total]):
                class_id = int(label)
                if class_id not in selected_set:
                    continue
                images_by_class[class_id].append(torch.from_numpy(images[local_idx]).clone())

            yielded += batch_total
            pbar.update(batch_total)
            del batch
            if max_samples is not None and yielded >= max_samples:
                break
    finally:
        pbar.close()

    stacked_images_by_class: dict[int, torch.Tensor] = {}
    for class_id in selected_class_ids:
        class_images = images_by_class[int(class_id)]
        if not class_images:
            raise RuntimeError(f"No images were collected for selected class {class_id}.")
        stacked_images_by_class[int(class_id)] = torch.stack(class_images, dim=0)

    return stacked_images_by_class


@torch.no_grad()
def _run_kmeans(
    points: torch.Tensor,
    num_clusters: int,
    seed: int,
    num_iters: int = 20,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.ndim != 2:
        raise ValueError("points must have shape [N, F].")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty.")

    num_points = points.shape[0]
    k = min(int(num_clusters), int(num_points))
    generator = torch.Generator(device=points.device).manual_seed(int(seed))
    init_idx = torch.randperm(num_points, generator=generator, device=points.device)[:k]
    centroids = points.index_select(0, init_idx).clone()

    for _ in range(int(num_iters)):
        assignments = torch.empty(num_points, dtype=torch.long, device=points.device)
        min_dists = torch.empty(num_points, dtype=torch.float32, device=points.device)

        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            dists = torch.cdist(points[start:end].float(), centroids.float(), p=2)
            min_dist, assign = dists.min(dim=1)
            assignments[start:end] = assign
            min_dists[start:end] = min_dist

        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(assignments, minlength=k)
        new_centroids.index_add_(0, assignments, points)

        non_empty = counts > 0
        if non_empty.any():
            new_centroids[non_empty] = new_centroids[non_empty] / counts[non_empty].unsqueeze(1)

        empty = ~non_empty
        if empty.any():
            replacement_idx = min_dists.topk(int(empty.sum()), largest=True).indices
            new_centroids[empty] = points.index_select(0, replacement_idx)

        if torch.allclose(new_centroids, centroids, atol=1e-4, rtol=1e-4):
            centroids = new_centroids
            break

        centroids = new_centroids

    assignments = torch.empty(num_points, dtype=torch.long, device=points.device)
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        dists = torch.cdist(points[start:end].float(), centroids.float(), p=2)
        assignments[start:end] = dists.argmin(dim=1)

    return centroids, assignments


@torch.no_grad()
def build_kmeans_landmarks_for_group_features(
    group_features: torch.Tensor,
    landmarks_per_class: int,
    seed: int,
    device: torch.device,
    num_iters: int = 20,
) -> torch.Tensor:
    group_features = group_features.to(device=device, dtype=torch.float32, non_blocking=True)
    num_points, num_locations, feat_dim = group_features.shape
    if num_points < int(landmarks_per_class):
        raise ValueError(
            f"Need at least {landmarks_per_class} points for k-means landmarks, "
            f"but only found {num_points}."
        )

    flat_features = group_features.reshape(num_points, num_locations * feat_dim)
    centroids, assignments = _run_kmeans(
        points=flat_features,
        num_clusters=landmarks_per_class,
        seed=seed,
        num_iters=num_iters,
    )

    landmarks = []
    for cluster_id in range(centroids.shape[0]):
        member_mask = assignments == cluster_id
        if member_mask.any():
            member_points = flat_features[member_mask]
            dists = torch.cdist(
                centroids[cluster_id : cluster_id + 1].float(),
                member_points.float(),
                p=2,
            )
            nearest_idx = dists.argmin(dim=1).item()
            landmarks.append(member_points[nearest_idx])
        else:
            dists = torch.cdist(
                centroids[cluster_id : cluster_id + 1].float(),
                flat_features.float(),
                p=2,
            )
            nearest_idx = dists.argmin(dim=1).item()
            landmarks.append(flat_features[nearest_idx])

    return torch.stack(landmarks, dim=0).reshape(-1, num_locations, feat_dim).cpu()


@torch.no_grad()
def build_kmeans_landmark_groups_from_class_images(
    encoder: torch.nn.Module,
    images_by_class: dict[int, torch.Tensor],
    selected_class_ids: list[int],
    device: torch.device,
    batch_size: int,
    landmarks_per_class: int,
    seed: int,
    kmeans_iters: int,
) -> list[torch.Tensor]:
    landmark_parts_by_group: list[list[torch.Tensor]] | None = None
    total_selected_samples = sum(int(images_by_class[class_id].shape[0]) for class_id in selected_class_ids)
    pbar = make_progress_bar(total=total_selected_samples, desc="Selecting landmarks")

    try:
        for class_id in selected_class_ids:
            class_groups = encode_landmark_groups(
                encoder=encoder,
                landmark_images=images_by_class[class_id],
                device=device,
                batch_size=batch_size,
                progress_desc=None,
            )
            if landmark_parts_by_group is None:
                landmark_parts_by_group = [[] for _ in class_groups]

            for group_idx, group_features in enumerate(class_groups):
                class_landmarks = build_kmeans_landmarks_for_group_features(
                    group_features=group_features,
                    landmarks_per_class=landmarks_per_class,
                    seed=int(seed) + int(class_id) * 1000 + int(group_idx),
                    device=device,
                    num_iters=kmeans_iters,
                )
                landmark_parts_by_group[group_idx].append(class_landmarks)

            pbar.update(int(images_by_class[class_id].shape[0]))
    finally:
        pbar.close()

    if landmark_parts_by_group is None:
        raise RuntimeError("No landmark groups were produced.")

    return [torch.cat(group_parts, dim=0) for group_parts in landmark_parts_by_group]


@torch.no_grad()
def build_kmeans_landmarks_by_class_from_class_images(
    encoder: torch.nn.Module,
    images_by_class: dict[int, torch.Tensor],
    selected_class_ids: list[int],
    device: torch.device,
    batch_size: int,
    landmarks_per_class: int,
    seed: int,
    kmeans_iters: int,
) -> list[dict[int, torch.Tensor]]:
    landmarks_by_group: list[dict[int, torch.Tensor]] | None = None
    total_selected_samples = sum(int(images_by_class[class_id].shape[0]) for class_id in selected_class_ids)
    pbar = make_progress_bar(total=total_selected_samples, desc="Selecting sharded landmarks")

    try:
        for class_id in selected_class_ids:
            class_groups = encode_landmark_groups(
                encoder=encoder,
                landmark_images=images_by_class[class_id],
                device=device,
                batch_size=batch_size,
                progress_desc=None,
            )
            if landmarks_by_group is None:
                landmarks_by_group = [{} for _ in class_groups]

            for group_idx, group_features in enumerate(class_groups):
                landmarks_by_group[group_idx][int(class_id)] = build_kmeans_landmarks_for_group_features(
                    group_features=group_features,
                    landmarks_per_class=landmarks_per_class,
                    seed=int(seed) + int(class_id) * 1000 + int(group_idx),
                    device=device,
                    num_iters=kmeans_iters,
                )

            pbar.update(int(images_by_class[class_id].shape[0]))
    finally:
        pbar.close()

    if landmarks_by_group is None:
        raise RuntimeError("No sharded landmark groups were produced.")

    return landmarks_by_group


def gather_landmark_groups_global(
    landmark_groups: list[torch.Tensor],
    device: torch.device,
    max_landmarks_per_rank: int = 256,
) -> list[torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return landmark_groups

    gathered_groups = []
    for landmarks in landmark_groups:
        local_landmarks_t = landmarks.transpose(0, 1).contiguous()
        local_landmarks_t = _subsample_landmarks_for_scale(
            local_landmarks_t,
            max_landmarks=max_landmarks_per_rank,
        )
        local_landmarks = local_landmarks_t.transpose(0, 1).contiguous()
        local_landmarks = local_landmarks.to(device=device, dtype=torch.float32, non_blocking=True)
        gathered = [torch.empty_like(local_landmarks) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_landmarks)
        gathered_groups.append(torch.cat(gathered, dim=0).cpu())
    return gathered_groups


def _init_feature_scale_state(
    landmark_groups: list[torch.Tensor],
    device: torch.device,
    max_landmarks: int = 256,
):
    scale_landmarks = []
    totals = []
    counts = []
    dims = []
    for public_landmarks in landmark_groups:
        if public_landmarks.ndim != 3:
            raise ValueError("Expected public_landmarks with shape [M, L, D].")
        landmarks_t = public_landmarks.transpose(0, 1).contiguous()
        landmarks_t = _subsample_landmarks_for_scale(
            landmarks_t,
            max_landmarks=max_landmarks,
        ).to(device=device, dtype=torch.float32, non_blocking=True)
        scale_landmarks.append(landmarks_t)
        totals.append(torch.zeros((), device=device, dtype=torch.float32))
        counts.append(0)
        dims.append(int(public_landmarks.shape[-1]))
    return scale_landmarks, totals, counts, dims


def _accumulate_feature_scale_state(
    group_features: torch.Tensor,
    scale_landmarks: torch.Tensor,
    total: torch.Tensor,
    count: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    num_points = int(group_features.shape[0])
    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        batch = group_features[start:end].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ).transpose(0, 1).contiguous()
        dists = torch.cdist(batch, scale_landmarks, p=2)
        total = total + dists.sum()
        count += dists.numel()
    return total, count


def _finalize_feature_scales(
    totals: list[torch.Tensor],
    counts: list[int],
    dims: list[int],
    eps: float = 1e-8,
    distributed: bool = False,
) -> list[torch.Tensor]:
    if distributed:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed=True requires torch.distributed to be initialized.")
        totals_t = torch.stack(totals)
        counts_t = torch.as_tensor(counts, device=totals_t.device, dtype=torch.float32)
        dist.all_reduce(totals_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)
        totals = [totals_t[i] for i in range(totals_t.shape[0])]
        counts = [int(counts_t[i].item()) for i in range(counts_t.shape[0])]

    feature_scales = []
    for total, count, dim in zip(totals, counts, dims, strict=True):
        if count <= 0:
            raise ValueError("Cannot estimate a feature scale from an empty sensitive set.")
        count_t = torch.as_tensor(count, device=total.device, dtype=total.dtype)
        feature_scales.append((total / count_t / math.sqrt(dim)).detach().clamp(min=eps).cpu())
    return feature_scales


@torch.no_grad()
def estimate_feature_scales_from_class_images(
    encoder: torch.nn.Module,
    images_by_class: dict[int, torch.Tensor],
    selected_class_ids: list[int],
    landmark_groups: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
    distributed: bool = False,
) -> list[torch.Tensor]:
    scale_landmarks, totals, counts, dims = _init_feature_scale_state(
        landmark_groups=landmark_groups,
        device=device,
    )
    total_selected_samples = sum(int(images_by_class[class_id].shape[0]) for class_id in selected_class_ids)
    pbar = make_progress_bar(total=total_selected_samples, desc="Estimating feature scale")

    try:
        for class_id in selected_class_ids:
            class_groups = encode_landmark_groups(
                encoder=encoder,
                landmark_images=images_by_class[class_id],
                device=device,
                batch_size=batch_size,
                progress_desc=None,
            )
            for group_idx, group_features in enumerate(class_groups):
                totals[group_idx], counts[group_idx] = _accumulate_feature_scale_state(
                    group_features=group_features,
                    scale_landmarks=scale_landmarks[group_idx],
                    total=totals[group_idx],
                    count=counts[group_idx],
                    device=device,
                    batch_size=batch_size,
                )

            pbar.update(int(images_by_class[class_id].shape[0]))
    finally:
        pbar.close()

    return _finalize_feature_scales(
        totals=totals,
        counts=counts,
        dims=dims,
        distributed=distributed,
    )


@torch.no_grad()
def estimate_feature_scales_streaming(
    encoder: torch.nn.Module,
    data_source: Path,
    landmark_groups: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    total_samples: int | None,
) -> list[torch.Tensor]:
    scale_landmarks, totals, counts, dims = _init_feature_scale_state(
        landmark_groups=landmark_groups,
        device=device,
    )
    amp_enabled = device.type == "cuda"
    pbar = make_progress_bar(total=total_samples, desc="Estimating feature scale")

    try:
        for images_cpu in iter_model_input_batches(
            data_source,
            batch_size=batch_size,
            max_samples=max_samples,
        ):
            images = images_cpu.to(device=device, non_blocking=True)
            with (
                torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if amp_enabled else nullcontext()
            ):
                groups = encoder(images)

            for group_idx, (feat, _) in enumerate(groups):
                totals[group_idx], counts[group_idx] = _accumulate_feature_scale_state(
                    group_features=feat.float(),
                    scale_landmarks=scale_landmarks[group_idx],
                    total=totals[group_idx],
                    count=counts[group_idx],
                    device=device,
                    batch_size=batch_size,
                )

            pbar.update(images.shape[0])
            del images_cpu, images, groups
    finally:
        pbar.close()

    return _finalize_feature_scales(totals=totals, counts=counts, dims=dims)


def accumulate_group_stats(
    group_features: torch.Tensor,
    stats_by_temp: dict[float, NystromStats],
    device: torch.device,
    batch_size: int,
):
    first_stats = next(iter(stats_by_temp.values()))
    feature_scale = torch.as_tensor(
        getattr(first_stats, "feature_scale"),
        device=device,
        dtype=torch.float32,
    ).detach().clamp(min=1e-8)

    for start in range(0, group_features.shape[0], batch_size):
        end = min(start + batch_size, group_features.shape[0])
        batch_points = group_features[start:end].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ).transpose(0, 1).contiguous()
        batch_points = batch_points / feature_scale

        for temp, stats in stats_by_temp.items():
            phi = compute_nystrom_features_batched(
                x=batch_points,
                landmarks=stats.landmarks,
                A=stats.A,
                temperature=float(temp),
            )
            stats.global_totals += phi.sum(dim=1)
            stats.global_weighted_points += torch.bmm(phi.transpose(1, 2), batch_points)


@torch.no_grad()
def build_nystrom_stats_groups_from_class_images(
    encoder: torch.nn.Module,
    images_by_class: dict[int, torch.Tensor],
    selected_class_ids: list[int],
    landmark_groups: list[torch.Tensor],
    temps: list[float],
    ridge: float,
    device: torch.device,
    batch_size: int,
    log_every: int,
) -> tuple[list[dict[float, NystromStats]], int]:
    feature_scales = estimate_feature_scales_from_class_images(
        encoder=encoder,
        images_by_class=images_by_class,
        selected_class_ids=selected_class_ids,
        landmark_groups=landmark_groups,
        device=device,
        batch_size=batch_size,
    )
    stats_groups = init_stats_groups(
        landmark_groups=landmark_groups,
        feature_scales=feature_scales,
        temps=temps,
        ridge=ridge,
        device=device,
    )

    total_selected_samples = sum(int(images_by_class[class_id].shape[0]) for class_id in selected_class_ids)
    processed = 0
    start_time = time.time()
    is_tty = sys.stderr.isatty()
    pbar = make_progress_bar(total=total_selected_samples, desc="Building cache")

    try:
        for class_id in selected_class_ids:
            class_groups = encode_landmark_groups(
                encoder=encoder,
                landmark_images=images_by_class[class_id],
                device=device,
                batch_size=batch_size,
                progress_desc=None,
            )
            for group_features, stats_by_temp in zip(class_groups, stats_groups, strict=True):
                accumulate_group_stats(
                    group_features=group_features,
                    stats_by_temp=stats_by_temp,
                    device=device,
                    batch_size=batch_size,
                )

            processed += int(images_by_class[class_id].shape[0])
            pbar.update(int(images_by_class[class_id].shape[0]))
            if log_every > 0 and processed % log_every == 0:
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1e-6)
                if is_tty:
                    pbar.set_postfix_str(f"{rate:.1f} img/s", refresh=False)
                else:
                    print(f"  summarized {processed:,} samples ({rate:.1f} img/s)")
    finally:
        pbar.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return stats_groups, processed


@torch.no_grad()
def build_subset_nystrom_stats_groups_from_class_images(
    encoder: torch.nn.Module,
    images_by_class: dict[int, torch.Tensor],
    selected_class_ids: list[int],
    temps: list[float],
    ridge: float,
    device: torch.device,
    batch_size: int,
    landmarks_per_class: int,
    seed: int,
    kmeans_iters: int,
    log_every: int,
) -> tuple[list[torch.Tensor], list[dict[float, NystromStats]], int]:
    landmark_groups = build_kmeans_landmark_groups_from_class_images(
        encoder=encoder,
        images_by_class=images_by_class,
        selected_class_ids=selected_class_ids,
        device=device,
        batch_size=batch_size,
        landmarks_per_class=landmarks_per_class,
        seed=seed,
        kmeans_iters=kmeans_iters,
    )
    stats_groups, processed = build_nystrom_stats_groups_from_class_images(
        encoder=encoder,
        images_by_class=images_by_class,
        selected_class_ids=selected_class_ids,
        landmark_groups=landmark_groups,
        temps=temps,
        ridge=ridge,
        device=device,
        batch_size=batch_size,
        log_every=log_every,
    )
    return landmark_groups, stats_groups, processed


def offload_stats_by_temp_to_cpu(
    stats_by_temp: dict[float, NystromStats],
) -> dict[float, NystromStats]:
    if not stats_by_temp:
        raise ValueError("Cannot offload an empty stats group.")

    first_stats = next(iter(stats_by_temp.values()))
    landmarks_cpu = first_stats.landmarks.detach().to(device="cpu", dtype=torch.float32)
    feature_scale = getattr(first_stats, "feature_scale", None)

    offloaded: dict[float, NystromStats] = {}
    for temp, stats in stats_by_temp.items():
        offloaded_stats = NystromStats(
            landmarks=landmarks_cpu,
            A=stats.A.detach().to(device="cpu", dtype=torch.float32),
            global_totals=stats.global_totals.detach().to(device="cpu", dtype=torch.float32),
            global_weighted_points=stats.global_weighted_points.detach().to(
                device="cpu",
                dtype=torch.float32,
            ),
            temperature=float(stats.temperature),
        )
        if feature_scale is not None:
            offloaded_stats.feature_scale = torch.as_tensor(feature_scale).detach().cpu()
        offloaded[float(temp)] = offloaded_stats
    return offloaded


@torch.no_grad()
def build_subset_nystrom_stats_groups_from_class_images_sharded(
    encoder: torch.nn.Module,
    images_by_class: dict[int, torch.Tensor],
    selected_class_ids: list[int],
    temps: list[float],
    ridge: float,
    device: torch.device,
    batch_size: int,
    landmarks_per_class: int,
    seed: int,
    kmeans_iters: int,
    log_every: int,
    cpu_offload: bool,
) -> tuple[list[dict[int, dict[float, NystromStats]]], int]:
    landmarks_by_group_by_class = build_kmeans_landmarks_by_class_from_class_images(
        encoder=encoder,
        images_by_class=images_by_class,
        selected_class_ids=selected_class_ids,
        device=device,
        batch_size=batch_size,
        landmarks_per_class=landmarks_per_class,
        seed=seed,
        kmeans_iters=kmeans_iters,
    )
    local_landmark_groups = [
        torch.cat(
            [landmarks_by_class[int(class_id)] for class_id in selected_class_ids],
            dim=0,
        )
        for landmarks_by_class in landmarks_by_group_by_class
    ]
    scale_landmark_groups = gather_landmark_groups_global(
        landmark_groups=local_landmark_groups,
        device=device,
    )
    feature_scales = estimate_feature_scales_from_class_images(
        encoder=encoder,
        images_by_class=images_by_class,
        selected_class_ids=selected_class_ids,
        landmark_groups=scale_landmark_groups,
        device=device,
        batch_size=batch_size,
        distributed=dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1,
    )
    stats_groups: list[dict[int, dict[float, NystromStats]]] = [
        {} for _ in landmarks_by_group_by_class
    ]
    total_selected_samples = sum(int(images_by_class[class_id].shape[0]) for class_id in selected_class_ids)
    processed = 0
    start_time = time.time()
    is_tty = sys.stderr.isatty()
    pbar = make_progress_bar(total=total_selected_samples, desc="Building sharded cache")

    try:
        for class_id in selected_class_ids:
            class_groups = encode_landmark_groups(
                encoder=encoder,
                landmark_images=images_by_class[class_id],
                device=device,
                batch_size=batch_size,
                progress_desc=None,
            )

            for group_idx, group_features in enumerate(class_groups):
                class_landmarks = landmarks_by_group_by_class[group_idx][int(class_id)]
                stats_by_temp = init_stats_groups(
                    landmark_groups=[class_landmarks],
                    feature_scales=[feature_scales[group_idx]],
                    temps=temps,
                    ridge=ridge,
                    device=device,
                )[0]
                accumulate_group_stats(
                    group_features=group_features,
                    stats_by_temp=stats_by_temp,
                    device=device,
                    batch_size=batch_size,
                )
                if cpu_offload:
                    stats_by_temp = offload_stats_by_temp_to_cpu(stats_by_temp)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                stats_groups[group_idx][int(class_id)] = stats_by_temp

            processed += int(images_by_class[class_id].shape[0])
            pbar.update(int(images_by_class[class_id].shape[0]))
            if log_every > 0 and processed % log_every == 0:
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1e-6)
                if is_tty:
                    pbar.set_postfix_str(f"{rate:.1f} img/s", refresh=False)
                else:
                    print(f"  summarized {processed:,} samples ({rate:.1f} img/s)")
    finally:
        pbar.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return stats_groups, processed


def allocate_balanced_landmarks_per_class(
    class_counts: dict[int, int],
    num_landmarks: int,
    seed: int,
) -> dict[int, int]:
    class_ids = sorted(class_counts)
    if not class_ids:
        raise ValueError("No classes found in the selected ImageNet32 split.")

    total_available = sum(class_counts.values())
    if num_landmarks > total_available:
        raise ValueError(
            f"Requested {num_landmarks} landmarks from only {total_available} available samples."
        )

    base = num_landmarks // len(class_ids)
    remainder = num_landmarks % len(class_ids)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(len(class_ids), generator=generator).tolist()

    targets = {class_id: base for class_id in class_ids}
    for idx in perm[:remainder]:
        targets[class_ids[idx]] += 1

    deficit = 0
    for class_id in class_ids:
        available = class_counts[class_id]
        if targets[class_id] > available:
            deficit += targets[class_id] - available
            targets[class_id] = available

    if deficit:
        for idx in perm:
            class_id = class_ids[idx]
            spare = class_counts[class_id] - targets[class_id]
            if spare <= 0:
                continue
            add = min(spare, deficit)
            targets[class_id] += add
            deficit -= add
            if deficit == 0:
                break

    if deficit:
        raise ValueError("Could not allocate the requested landmark budget across classes.")

    return targets


def sample_balanced_random_landmark_images(
    source: Path,
    targets_by_class: dict[int, int],
    seed: int,
    max_samples: int | None,
    total_samples: int | None = None,
) -> torch.Tensor:
    rng = random.Random(int(seed))
    reservoirs = {
        class_id: []
        for class_id, target in targets_by_class.items()
        if target > 0
    }
    seen_by_class = {class_id: 0 for class_id in reservoirs}
    yielded = 0
    pbar = make_progress_bar(total=total_samples, desc="Sampling landmarks")

    try:
        for _, batch in iter_imagenet32_batches(source):
            labels = batch["labels"]
            flat = batch["data"]
            batch_total = int(len(labels))
            if max_samples is not None:
                batch_total = min(batch_total, max_samples - yielded)

            images = flat[:batch_total].reshape(-1, 3, 32, 32)
            for local_idx, label in enumerate(labels[:batch_total]):
                class_id = int(label)
                target = targets_by_class.get(class_id, 0)
                if target <= 0:
                    continue

                seen_by_class[class_id] += 1
                reservoir = reservoirs[class_id]
                image = torch.from_numpy(images[local_idx]).clone()

                if len(reservoir) < target:
                    reservoir.append(image)
                    continue

                replace_idx = rng.randrange(seen_by_class[class_id])
                if replace_idx < target:
                    reservoir[replace_idx] = image

            yielded += batch_total
            pbar.update(batch_total)
            del batch
            if max_samples is not None and yielded >= max_samples:
                break
    finally:
        pbar.close()

    selected = []
    for class_id in sorted(reservoirs):
        target = targets_by_class[class_id]
        if len(reservoirs[class_id]) != target:
            raise RuntimeError(
                f"Collected {len(reservoirs[class_id])} landmarks for class {class_id}, "
                f"expected {target}."
            )
        selected.extend(reservoirs[class_id])

    if not selected:
        raise RuntimeError("No landmark images were sampled.")

    return torch.stack(selected, dim=0)


def init_stats_groups(
    landmark_groups: list[torch.Tensor],
    feature_scales: list[torch.Tensor],
    temps: list[float],
    ridge: float,
    device: torch.device,
) -> list[dict[float, NystromStats]]:
    stats_groups: list[dict[float, NystromStats]] = []
    if len(landmark_groups) != len(feature_scales):
        raise ValueError("landmark_groups and feature_scales must have the same length.")

    for public_landmarks, feature_scale in zip(landmark_groups, feature_scales, strict=True):
        stats_by_temp: dict[float, NystromStats] = {}
        shared_landmarks = None
        feature_scale = torch.as_tensor(
            feature_scale,
            device=device,
            dtype=torch.float32,
        ).detach().clamp(min=1e-8)
        public_landmarks_n = public_landmarks.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ) / feature_scale

        for temp in temps:
            landmarks, A = prepare_nystrom_landmarks_batched(
                public_landmarks=public_landmarks_n,
                temperature=float(temp),
                ridge=ridge,
                device=device,
            )
            if shared_landmarks is None:
                shared_landmarks = landmarks
            else:
                del landmarks

            l_count, m_count, feat_dim = shared_landmarks.shape
            stats = NystromStats(
                landmarks=shared_landmarks,
                A=A,
                global_totals=torch.zeros(
                    l_count,
                    m_count,
                    device=device,
                    dtype=torch.float32,
                ),
                global_weighted_points=torch.zeros(
                    l_count,
                    m_count,
                    feat_dim,
                    device=device,
                    dtype=torch.float32,
                ),
                temperature=float(temp),
            )
            stats.feature_scale = feature_scale.detach().cpu()
            stats_by_temp[float(temp)] = stats

        stats_groups.append(stats_by_temp)

    return stats_groups


@torch.no_grad()
def build_streaming_nystrom_stats_groups(
    encoder: torch.nn.Module,
    data_source: Path,
    landmark_groups: list[torch.Tensor],
    temps: list[float],
    ridge: float,
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    total_samples: int | None,
    log_every: int,
) -> tuple[list[dict[float, NystromStats]], int]:
    feature_scales = estimate_feature_scales_streaming(
        encoder=encoder,
        data_source=data_source,
        landmark_groups=landmark_groups,
        device=device,
        batch_size=batch_size,
        max_samples=max_samples,
        total_samples=total_samples,
    )
    stats_groups = init_stats_groups(
        landmark_groups=landmark_groups,
        feature_scales=feature_scales,
        temps=temps,
        ridge=ridge,
        device=device,
    )

    amp_enabled = device.type == "cuda"
    processed = 0
    start_time = time.time()
    is_tty = sys.stderr.isatty()
    pbar = make_progress_bar(total=total_samples, desc="Building cache")

    try:
        for images_cpu in iter_model_input_batches(
            data_source,
            batch_size=batch_size,
            max_samples=max_samples,
        ):
            images = images_cpu.to(device=device, non_blocking=True)
            with (
                torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if amp_enabled else nullcontext()
            ):
                groups = encoder(images)

            for (feat, _), stats_by_temp in zip(groups, stats_groups, strict=True):
                accumulate_group_stats(
                    group_features=feat.float(),
                    stats_by_temp=stats_by_temp,
                    device=device,
                    batch_size=batch_size,
                )

            processed += images.shape[0]
            pbar.update(images.shape[0])
            if log_every > 0 and processed % log_every == 0:
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1e-6)
                if is_tty:
                    pbar.set_postfix_str(f"{rate:.1f} img/s", refresh=False)
                else:
                    print(f"  summarized {processed:,} samples ({rate:.1f} img/s)")

            del images_cpu, images, groups
    finally:
        pbar.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return stats_groups, processed


def train(args):
    use_ddp = "RANK" in os.environ or "WORLD_SIZE" in os.environ
    rank, world_size, local_rank, host_sync_group = setup_runtime(use_ddp)
    device = (
        torch.device(f"cuda:{local_rank}")
        if use_ddp
        else torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    is_main = rank == 0
    seed = set_process_seed(rank)

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
        if args.nystrom_shard_by_class:
            cfg.nystrom_shard_by_class = True
        if args.nystrom_shard_cpu_offload:
            cfg.nystrom_shard_cpu_offload = True
        if args.nystrom_ridge is not None:
            cfg.nystrom_ridge = args.nystrom_ridge
        if args.nystrom_repulsion is not None:
            cfg.nystrom_repulsion = args.nystrom_repulsion
        if args.nystrom_landmark_seed is not None:
            cfg.nystrom_landmark_seed = args.nystrom_landmark_seed
        if args.nystrom_kmeans_iters is not None:
            cfg.nystrom_kmeans_iters = args.nystrom_kmeans_iters
        cfg.dataset = "imagenet32"
        evaluate_every_seconds = int(args.evaluate_every_seconds or 0)
        fid_eval_weights = normalize_eval_weight_list(args.fid_eval_weights)
        deadline_epoch = float(args.deadline_epoch) if args.deadline_epoch is not None else None

        expected_num_landmarks = (
            int(args.nystrom_subset_num_classes) * int(args.nystrom_landmarks_per_class)
        )
        if int(args.nystrom_num_landmarks) != expected_num_landmarks:
            raise ValueError(
                "--nystrom-num-landmarks must equal "
                "--nystrom-subset-num-classes * --nystrom-landmarks-per-class. "
                f"Got {args.nystrom_num_landmarks} vs {expected_num_landmarks}."
            )
        if cfg.nystrom_shard_by_class and cfg.nystrom_repulsion != "exact":
            raise ValueError("Class-sharded Nyström currently requires --nystrom-repulsion exact.")
        if cfg.nystrom_shard_cpu_offload and not cfg.nystrom_shard_by_class:
            raise ValueError("--nystrom-shard-cpu-offload requires --nystrom-shard-by-class.")
        if cfg.evaluate_every < 0:
            raise ValueError("evaluate_every must be non-negative.")
        if cfg.save_every < 0:
            raise ValueError("save_every must be non-negative.")
        if evaluate_every_seconds < 0:
            raise ValueError("evaluate_every_seconds must be non-negative.")

        partition_selected_classes_by_rank = bool(
            use_ddp and cfg.nystrom_shard_by_class and world_size > 1
        )

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
            print(f"Process seed: {seed}")

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

        if device.type == "cuda":
            model = torch.compile(model)
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
        if device.type == "cuda" and not args.no_compile_encoder:
            feat_encoder = torch.compile(feat_encoder)

        if is_main:
            n_feat_params = sum(p.numel() for p in feat_encoder.parameters())
            print(f"Feature encoder: {cfg.encoder}, {n_feat_params:,} params (frozen)")
            print(f"  Input size: {encoder_input_size}x{encoder_input_size}")
        else:
            n_feat_params = None

        data_source = Path(args.data_source)
        usable_samples = None
        num_classes_present = None
        global_selected_class_ids = None
        selected_class_partitions = None
        selected_sample_totals_by_rank = None

        if is_main:
            class_counts, usable_samples = count_class_samples(
                data_source,
                max_samples=args.max_samples,
            )
            num_classes_present = len(class_counts)
            print(f"ImageNet samples used: {usable_samples:,}")
            print(f"Classes present: {num_classes_present:,}")
            if cfg.nystrom_shard_by_class and not args.restrict_training_to_selected_classes:
                required_unique_classes = int(args.nystrom_subset_num_classes) * (
                    world_size if partition_selected_classes_by_rank else 1
                )
                if required_unique_classes != num_classes_present:
                    raise ValueError(
                        "Class-sharded Nyström currently requires --restrict-training-to-selected-classes "
                        "unless the selected class budget covers every class in the dataset."
                    )

            if partition_selected_classes_by_rank:
                global_selected_class_ids, selected_class_partitions = select_random_class_partitions(
                    class_counts=class_counts,
                    num_classes_per_rank=args.nystrom_subset_num_classes,
                    min_samples_per_class=args.nystrom_landmarks_per_class,
                    seed=cfg.nystrom_landmark_seed,
                    world_size=world_size,
                )
            else:
                rank0_class_ids = select_random_class_subset(
                    class_counts=class_counts,
                    num_classes=args.nystrom_subset_num_classes,
                    min_samples_per_class=args.nystrom_landmarks_per_class,
                    seed=cfg.nystrom_landmark_seed,
                )
                global_selected_class_ids = list(rank0_class_ids)
                selected_class_partitions = [rank0_class_ids]

            selected_sample_totals_by_rank = [
                sum(class_counts[class_id] for class_id in rank_class_ids)
                for rank_class_ids in selected_class_partitions
            ]
            global_selected_sample_total = sum(selected_sample_totals_by_rank)
            if partition_selected_classes_by_rank:
                print(
                    f"Selected {len(global_selected_class_ids):,} unique classes with seed="
                    f"{cfg.nystrom_landmark_seed}; global subset samples={global_selected_sample_total:,}"
                )
                print(
                    f"  per-rank classes={args.nystrom_subset_num_classes}, "
                    f"ranks={world_size}, rank0 subset samples={selected_sample_totals_by_rank[0]:,}"
                )
                print(f"  first rank0 classes: {selected_class_partitions[0][:10]}")
            else:
                print(
                    f"Selected {len(global_selected_class_ids):,} classes with seed="
                    f"{cfg.nystrom_landmark_seed}; subset samples={global_selected_sample_total:,}"
                )
                print(f"  first selected classes: {global_selected_class_ids[:10]}")

        usable_samples = broadcast_object_from_main(usable_samples, is_main)
        num_classes_present = broadcast_object_from_main(num_classes_present, is_main)
        global_selected_class_ids = broadcast_object_from_main(global_selected_class_ids, is_main)
        selected_class_partitions = broadcast_object_from_main(selected_class_partitions, is_main)
        selected_sample_totals_by_rank = broadcast_object_from_main(selected_sample_totals_by_rank, is_main)

        local_partition_index = rank if partition_selected_classes_by_rank else 0
        local_selected_class_ids = [
            int(class_id) for class_id in selected_class_partitions[local_partition_index]
        ]
        local_selected_sample_total = int(selected_sample_totals_by_rank[local_partition_index])
        summary_cache_metadata = make_summary_cache_metadata(
            data_source=data_source,
            cfg=cfg,
            encoder_input_size=encoder_input_size,
            num_landmarks=args.nystrom_num_landmarks,
            max_samples=args.max_samples,
            subset_num_classes=len(local_selected_class_ids),
            landmarks_per_class=args.nystrom_landmarks_per_class,
            selected_class_ids=local_selected_class_ids,
            restrict_training_to_selected_classes=args.restrict_training_to_selected_classes,
        )
        explicit_cache_path = args.summary_cache_path
        if partition_selected_classes_by_rank:
            explicit_cache_path = make_rank_local_explicit_cache_path(
                explicit_path=explicit_cache_path,
                rank=rank,
                world_size=world_size,
            )
        summary_cache_path = make_summary_cache_path(
            metadata=summary_cache_metadata,
            explicit_path=explicit_cache_path,
        )

        if is_main:
            if partition_selected_classes_by_rank:
                print(f"Rank-local Nyström summary cache (rank0): {summary_cache_path}")
            else:
                print(f"Nystrom summary cache: {summary_cache_path}")
            write_training_config_snapshot(
                out_dir=out_dir,
                args=args,
                cfg=cfg,
                unet_cfg=unet_cfg,
                encoder_input_size=encoder_input_size,
                model_num_params=n_params,
                feat_encoder_num_params=n_feat_params,
                summary_cache_path=summary_cache_path,
                selected_class_ids=global_selected_class_ids,
                rank0_selected_class_ids=(
                    local_selected_class_ids if partition_selected_classes_by_rank else None
                ),
            )

        stats_groups = None
        summarized_samples = None
        should_build_summary_on_this_rank = bool(partition_selected_classes_by_rank or is_main)
        summary_cache_device = torch.device(
            "cpu" if (cfg.nystrom_shard_by_class and cfg.nystrom_shard_cpu_offload) else device
        )
        if summary_cache_path.is_file() and not args.rebuild_summary_cache:
            if is_main:
                print(f"Loading cached Nystrom summary from {summary_cache_path}...")
            try:
                stats_groups, summarized_samples = load_summary_cache(
                    cache_path=summary_cache_path,
                    expected_metadata=summary_cache_metadata,
                    device=summary_cache_device,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if is_main:
                    if summarized_samples is None:
                        print("Loaded cached Nystrom summary.")
                    else:
                        print(
                            f"Loaded cached Nystrom summary built from {summarized_samples:,} samples."
                        )
            except Exception as exc:
                stats_groups = None
                if is_main:
                    print(f"  Failed to load cache ({exc}). Rebuilding summary cache...")

        if partition_selected_classes_by_rank:
            local_needs_build = torch.tensor(
                [1 if stats_groups is None else 0],
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(local_needs_build, op=dist.ReduceOp.MAX)
            if int(local_needs_build.item()) != 0:
                if stats_groups is not None:
                    stats_groups = None
                    summarized_samples = None
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                if is_main:
                    print("At least one rank needs a Nystrom summary rebuild; rebuilding on all ranks.")

        if stats_groups is None and should_build_summary_on_this_rank:
            if is_main:
                print(
                    f"Collecting {local_selected_sample_total:,} images from the selected "
                    f"{len(local_selected_class_ids):,} ImageNet classes..."
                )
                print(
                    f"  landmarks_per_class={args.nystrom_landmarks_per_class} "
                    f"kmeans_iters={cfg.nystrom_kmeans_iters} seed={cfg.nystrom_landmark_seed} "
                    f"restrict_training_to_selected_classes={args.restrict_training_to_selected_classes}"
                )
                if partition_selected_classes_by_rank:
                    print(
                        f"  distributed_class_partition=True unique_classes="
                        f"{len(global_selected_class_ids):,} per_rank_classes="
                        f"{len(local_selected_class_ids):,}"
                    )
                precompute_start = time.time()
            images_by_class = collect_class_images(
                source=data_source,
                selected_class_ids=local_selected_class_ids,
                max_samples=args.max_samples,
                total_samples=usable_samples,
            )

            if is_main:
                print(
                    f"Running per-class k-means over {len(local_selected_class_ids):,} selected classes "
                    f"to build {args.nystrom_num_landmarks:,} landmarks..."
                )
            if cfg.nystrom_shard_by_class:
                stats_groups, summarized_samples = build_subset_nystrom_stats_groups_from_class_images_sharded(
                    encoder=feat_encoder,
                    images_by_class=images_by_class,
                    selected_class_ids=local_selected_class_ids,
                    temps=list(cfg.temperatures),
                    ridge=cfg.nystrom_ridge,
                    device=device,
                    batch_size=args.feature_batch_size,
                    landmarks_per_class=args.nystrom_landmarks_per_class,
                    seed=cfg.nystrom_landmark_seed,
                    kmeans_iters=cfg.nystrom_kmeans_iters,
                    log_every=args.precompute_log_every,
                    cpu_offload=cfg.nystrom_shard_cpu_offload,
                )
                summary_scope = "selected-class-sharded"
                if is_main:
                    print(
                        f"  shard_by_class=True shard_cpu_offload={cfg.nystrom_shard_cpu_offload} "
                        f"selected_classes={len(local_selected_class_ids):,}"
                    )
            else:
                landmark_groups = build_kmeans_landmark_groups_from_class_images(
                    encoder=feat_encoder,
                    images_by_class=images_by_class,
                    selected_class_ids=local_selected_class_ids,
                    device=device,
                    batch_size=args.feature_batch_size,
                    landmarks_per_class=args.nystrom_landmarks_per_class,
                    seed=cfg.nystrom_landmark_seed,
                    kmeans_iters=cfg.nystrom_kmeans_iters,
                )
                if is_main:
                    for group_idx, group in enumerate(landmark_groups):
                        print(f"  landmark group {group_idx}: shape={list(group.shape)}")

                if args.restrict_training_to_selected_classes:
                    stats_groups, summarized_samples = build_nystrom_stats_groups_from_class_images(
                        encoder=feat_encoder,
                        images_by_class=images_by_class,
                        selected_class_ids=local_selected_class_ids,
                        landmark_groups=landmark_groups,
                        temps=list(cfg.temperatures),
                        ridge=cfg.nystrom_ridge,
                        device=device,
                        batch_size=args.feature_batch_size,
                        log_every=args.precompute_log_every,
                    )
                    summary_scope = "selected-class"
                else:
                    stats_groups, summarized_samples = build_streaming_nystrom_stats_groups(
                        encoder=feat_encoder,
                        data_source=data_source,
                        landmark_groups=landmark_groups,
                        temps=list(cfg.temperatures),
                        ridge=cfg.nystrom_ridge,
                        device=device,
                        batch_size=args.feature_batch_size,
                        max_samples=args.max_samples,
                        total_samples=usable_samples,
                        log_every=args.precompute_log_every,
                    )
                    summary_scope = "full-dataset"

            del images_by_class
            if not cfg.nystrom_shard_by_class:
                del landmark_groups
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if is_main:
                print(
                    f"Pre-computation complete. Built a {summary_scope} Nystrom summary from "
                    f"{summarized_samples:,} samples in {(time.time() - precompute_start) / 3600.0:.2f} h"
                )
            save_summary_cache(
                cache_path=summary_cache_path,
                metadata=summary_cache_metadata,
                stats_groups=stats_groups,
                summarized_samples=summarized_samples,
            )
            if is_main:
                print(f"Saved Nystrom summary cache to {summary_cache_path}")

        if use_ddp:
            sync_workers(host_sync_group)

        if stats_groups is None:
            stats_groups, summarized_samples = load_summary_cache(
                cache_path=summary_cache_path,
                expected_metadata=summary_cache_metadata,
                device=summary_cache_device,
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

        fid_ref_dir = DEFAULT_IMAGENET32_REF_DIR
        if is_main:
            fid_ref_dir = prepare_imagenet32_reference(
                data_source=data_source,
                output_dir=DEFAULT_IMAGENET32_REF_DIR,
                num_images=DEFAULT_NUM_FID_SAMPLES,
            )
        if use_ddp:
            sync_workers(host_sync_group)

        if is_main:
            global_batch_size = cfg.batch_size * world_size
            print(
                f"Training DriftXpress (multi-res {cfg.encoder}) on ImageNet for {cfg.total_steps} steps"
            )
            print(f"  batch={cfg.batch_size}, global_batch={global_batch_size}")
            print(f"  temps={cfg.temperatures}, pool_size={cfg.pool_size}")
            print(f"  more_features={cfg.more_features}")
            print(
                f"  shard_by_class={cfg.nystrom_shard_by_class}, "
                f"shard_cpu_offload={cfg.nystrom_shard_cpu_offload}"
            )
            if partition_selected_classes_by_rank:
                print(
                    f"  distributed_class_partition=True, "
                    f"unique_classes={len(global_selected_class_ids):,}, "
                    f"per_rank_classes={len(local_selected_class_ids):,}"
                )
            print(f"  encoder input={encoder_input_size}x{encoder_input_size}")
            print(f"  evaluate_every={cfg.evaluate_every}")
            print(f"  evaluate_every_seconds={evaluate_every_seconds}")
            print(f"  save_every={cfg.save_every}")
            print(f"  fid_eval_weights={','.join(fid_eval_weights)}")
            print(f"  skip_final_fid={args.skip_final_fid}")
            print(f"  deadline_epoch={deadline_epoch if deadline_epoch is not None else '<none>'}")

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

        for step in range(1, cfg.total_steps + 1):
            last_step = step
            z = torch.randn(
                cfg.batch_size,
                3,
                32,
                32,
                device=device,
            ).to(memory_format=torch.channels_last)
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

            if cfg.nystrom_shard_by_class:
                loss = drifting_loss_multires_nystrom_sharded(
                    gen_groups,
                    stats_groups,
                    temps=tuple(cfg.temperatures),
                    repulsion_mode=cfg.nystrom_repulsion,
                    distributed_shards=partition_selected_classes_by_rank,
                )
            else:
                loss = drifting_loss_multires_nystrom(
                    gen_groups,
                    stats_groups,
                    temps=tuple(cfg.temperatures),
                    repulsion_mode=cfg.nystrom_repulsion,
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

                if should_evaluate:
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/imagenet32_nystrom_kmeans_subset100x50",
    )
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
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=32,
        help="Batch size for subset feature encoding and summary construction.",
    )
    parser.add_argument("--evaluate-every", type=int, default=None)
    parser.add_argument("--evaluate-every-seconds", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--fid-eval-weights", type=str, default="raw,ema")
    parser.add_argument("--skip-final-fid", action="store_true")
    parser.add_argument("--deadline-epoch", type=float, default=None)
    parser.add_argument("--temps", type=str, default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--more-features", "--more-feautres", dest="more_features", action="store_true")
    parser.add_argument("--nystrom-num-landmarks", type=int, default=125*150)
    parser.add_argument(
        "--nystrom-subset-num-classes",
        type=int,
        default=125,
        help=(
            "How many ImageNet32 classes to sample per rank for the Nyström summary. "
            "In class-sharded DDP runs, the total unique class count becomes "
            "--nystrom-subset-num-classes * world_size."
        ),
    )
    parser.add_argument(
        "--nystrom-landmarks-per-class",
        type=int,
        default=150,
        help="How many k-means landmarks to keep per selected class.",
    )
    parser.add_argument("--nystrom-ridge", type=float, default=None)
    parser.add_argument("--nystrom-landmark-seed", type=int, default=None)
    parser.add_argument("--nystrom-kmeans-iters", type=int, default=None)
    parser.add_argument(
        "--nystrom-shard-by-class",
        action="store_true",
        help="Build one Nyström attractive cache shard per selected ImageNet32 class and combine shard numerators/denominators.",
    )
    parser.add_argument(
        "--nystrom-shard-cpu-offload",
        action="store_true",
        help="Store class-sharded Nyström caches on CPU after precompute and stream them back to the training device.",
    )
    parser.add_argument(
        "--restrict-training-to-selected-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, build the Nyström summary only from the randomly selected class subset. "
            "Use --no-restrict-training-to-selected-classes to build the summary over all "
            "ImageNet32 classes while still choosing landmarks from the selected subset."
        ),
    )
    parser.add_argument("--nystrom-repulsion", type=str, choices=["nystrom", "exact"], default=None)
    parser.add_argument(
        "--summary-cache-path",
        type=str,
        default=None,
        help="Optional path for the saved Nystrom summary cache. Defaults to outputs/nystrom_cache/<derived-name>.pt",
    )
    parser.add_argument(
        "--rebuild-summary-cache",
        action="store_true",
        help="Ignore any existing saved summary cache and rebuild it.",
    )
    parser.add_argument(
        "--precompute-log-every",
        type=int,
        default=50_000,
        help="Print progress every N real samples during summary construction.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on how many ImageNet32 samples to use for debugging.",
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
