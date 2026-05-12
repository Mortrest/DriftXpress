import argparse
import csv
import json
import math
import os
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from common.config import MultiResDriftConfig, UNetConfig, UNetLargeConfig
from common.loss_reporting import count_feature_slots, format_scientific, scaled_loss_for_logging
from evaluation.sample import (
    drift_sample,
    drift_sample_from_noise,
    ensure_three_channels,
    load_sample_noise,
    sample_model_noise,
    save_sample_grid,
)
from models.ema import EMA
from models.unet import UNet
from methods.driftxpress import (
    NystromStats,
    drifting_loss_multires_nystrom,
    drifting_loss_multires_nystrom_sharded,
    move_nystrom_stats_to_device,
    precompute_nystrom_statistics_multitemp_batched,
)
from features.encoders import build_encoder

from evaluate_fid_10k_raw import (
    SUPPORTED_DATASETS,
    build_cifar_dataset,
    build_dataset_transform,
    configure_unet_cfg_for_dataset,
    default_ref_dir,
    evaluate_checkpoint,
    evaluate_loaded_model,
    get_dataset_class_names,
    get_dataset_spec,
    get_dataset_targets,
    normalize_eval_weights,
    normalize_dataset_name,
    prepare_cifar_reference,
)

IMAGENET32_DEFAULT_SUBSET_NUM_CLASSES = 125
IMAGENET32_DEFAULT_NYSTROM_LANDMARKS_PER_CLASS = 150
IMAGENET32_DEFAULT_FEATURE_BATCH_SIZE = 32
IMAGENET32_DEFAULT_PRECOMPUTE_LOG_EVERY = 50_000
TRAIN_SUPPORTED_DATASETS = (*SUPPORTED_DATASETS, "imagenet32")


@dataclass
class FoldedNystromStats:
    landmarks: torch.Tensor
    summary_cat: torch.Tensor
    temperature: float


@dataclass
class PackedShardedFoldedNystromStats:
    landmarks: torch.Tensor
    summary_cat: torch.Tensor
    temperature: float


_PACKED_SHARDED_FOLDED_NYSTROM_CACHE: dict[
    tuple[int, float, str],
    PackedShardedFoldedNystromStats,
] = {}


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


def _device_key(device: torch.device) -> str:
    if device.index is None:
        return f"{device.type}:-1"
    return f"{device.type}:{device.index}"


def resolve_imagenet32_landmark_defaults(args):
    subset_num_classes = (
        int(args.nystrom_subset_num_classes)
        if args.nystrom_subset_num_classes is not None
        else IMAGENET32_DEFAULT_SUBSET_NUM_CLASSES
    )
    landmarks_per_class = (
        int(args.nystrom_landmarks_per_class)
        if args.nystrom_landmarks_per_class is not None
        else IMAGENET32_DEFAULT_NYSTROM_LANDMARKS_PER_CLASS
    )
    num_landmarks = args.nystrom_num_landmarks
    if num_landmarks is None:
        num_landmarks = args.nystrom_total_landmarks
    if num_landmarks is None:
        num_landmarks = subset_num_classes * landmarks_per_class
    return subset_num_classes, landmarks_per_class, int(num_landmarks)


def delegate_imagenet32_training(args):
    unsupported_flags = []
    if args.nystrom_landmark_strategy is not None:
        unsupported_flags.append("--nystrom-landmark-strategy")
    if args.nystrom_landmark_class is not None:
        unsupported_flags.append("--nystrom-landmark-class")
    if args.nystrom_facility_candidate_ratio is not None:
        unsupported_flags.append("--nystrom-facility-candidate-ratio")
    if args.nystrom_facility_eval_ratio is not None:
        unsupported_flags.append("--nystrom-facility-eval-ratio")
    if args.nystrom_density_knn_k is not None:
        unsupported_flags.append("--nystrom-density-knn-k")
    if args.nystrom_density_reference_size is not None:
        unsupported_flags.append("--nystrom-density-reference-size")
    if args.nystrom_folded_exact_attraction:
        unsupported_flags.append("--nystrom-folded-exact-attraction")
    if args.nystrom_classes_per_shard is not None:
        unsupported_flags.append("--nystrom-classes-per-shard")
    if args.no_save_on_eval:
        unsupported_flags.append("--no-save-on-eval")
    if args.sample_every_seconds:
        unsupported_flags.append("--sample-every-seconds")
    if unsupported_flags:
        raise ValueError(
            "The ImageNet backend behind DriftXpress does not support "
            f"{', '.join(unsupported_flags)}."
        )

    from trainers import imagenet_driftxpress

    subset_num_classes, landmarks_per_class, num_landmarks = resolve_imagenet32_landmark_defaults(args)
    batch_size = 256 if args.batch_size is None else int(args.batch_size)
    delegate_args = argparse.Namespace(
        data_source=args.data_source,
        output_dir=args.output_dir,
        encoder=args.encoder,
        encoder_size=args.encoder_size,
        steps=args.steps,
        batch_size=batch_size,
        feature_batch_size=args.feature_batch_size,
        evaluate_every=args.evaluate_every,
        evaluate_every_seconds=args.evaluate_every_seconds,
        save_every=args.save_every,
        fid_eval_weights=args.fid_eval_weights,
        skip_final_fid=args.skip_final_fid,
        deadline_epoch=args.deadline_epoch,
        temps=args.temps,
        pool_size=args.pool_size,
        more_features=args.more_features,
        nystrom_num_landmarks=num_landmarks,
        nystrom_subset_num_classes=subset_num_classes,
        nystrom_landmarks_per_class=landmarks_per_class,
        nystrom_ridge=args.nystrom_ridge,
        nystrom_landmark_seed=args.nystrom_landmark_seed,
        nystrom_kmeans_iters=args.nystrom_kmeans_iters,
        nystrom_shard_by_class=bool(args.nystrom_shard_by_class),
        nystrom_shard_cpu_offload=args.nystrom_shard_cpu_offload,
        restrict_training_to_selected_classes=args.restrict_training_to_selected_classes,
        nystrom_repulsion=args.nystrom_repulsion,
        summary_cache_path=args.summary_cache_path,
        rebuild_summary_cache=args.rebuild_summary_cache,
        precompute_log_every=args.precompute_log_every,
        max_samples=args.max_samples,
        device=args.device,
        no_compile_encoder=args.no_compile_encoder,
        large=args.large,
    )
    return imagenet_driftxpress.train(delegate_args)


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


def _alternative_temperature_scale(temperature, D, eps=1e-8):
    return max(float(temperature) * math.sqrt(float(D)), eps)


def _alternative_laplacian_kernel_batched(x, y, temperature, eps=1e-8):
    """Batched Laplacian kernel with the normalized-objective temperature scale."""
    D = x.shape[-1]
    tau_tilde = _alternative_temperature_scale(temperature, D, eps=eps)
    dist = torch.cdist(x.float(), y.float(), p=2)
    return torch.exp(-dist / tau_tilde)


def _alternative_inverse_sqrt_psd_batched(mats, eps=1e-8):
    eigvals, eigvecs = torch.linalg.eigh(mats.float())
    inv_sqrt_eigs = eigvals.clamp_min(eps).rsqrt()
    return (eigvecs * inv_sqrt_eigs.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


def move_nystrom_stats_to_device(stats: NystromStats, device: torch.device | str) -> NystromStats:
    """Move a Nyström cache block and preserve the alternative feature scale metadata."""
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
    """Exact repulsive barycenter with the normalized-objective temperature scale."""
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


@torch.no_grad()
def precompute_nystrom_statistics_batched(
    sensitive_points,
    public_landmarks,
    temperature=0.05,
    ridge=1e-4,
    eps=1e-8,
    batch_size=512,
    device=None,
    feature_scale=None,
):
    """Precompute normalized-space Nyström summaries with the alternative kernel scale."""
    if sensitive_points.ndim != 3:
        raise ValueError("Expected sensitive_points with shape [Np, L, D].")
    if public_landmarks.ndim != 3:
        raise ValueError("Expected public_landmarks with shape [M, L, D].")

    if device is None:
        device = sensitive_points.device
    if feature_scale is None:
        feature_scale = _estimate_feature_scale_from_landmarks(
            sensitive_points=sensitive_points,
            public_landmarks=public_landmarks,
            eps=eps,
            batch_size=batch_size,
            device=device,
        )
    feature_scale = torch.as_tensor(
        feature_scale,
        device=device,
        dtype=torch.float32,
    ).detach().clamp(min=eps)
    public_landmarks_n = public_landmarks.to(device=device, dtype=torch.float32, non_blocking=True)
    public_landmarks_n = public_landmarks_n / feature_scale

    landmarks, A = prepare_nystrom_landmarks_batched(
        public_landmarks=public_landmarks_n,
        temperature=temperature,
        ridge=ridge,
        eps=eps,
        device=device,
    )

    L, M, D = landmarks.shape
    global_totals = torch.zeros(L, M, device=device, dtype=torch.float32)
    global_weighted_points = torch.zeros(L, M, D, device=device, dtype=torch.float32)

    Np = sensitive_points.shape[0]
    for start in range(0, Np, batch_size):
        end = min(start + batch_size, Np)
        batch = sensitive_points[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        batch = batch / feature_scale
        batch_t = batch.transpose(0, 1).contiguous()
        phi = compute_nystrom_features_batched(
            batch_t,
            landmarks,
            A,
            temperature=temperature,
            eps=eps,
        )
        global_totals += phi.sum(dim=1)
        global_weighted_points += torch.bmm(phi.transpose(1, 2), batch_t)

    stats = NystromStats(
        landmarks=landmarks,
        A=A,
        global_totals=global_totals,
        global_weighted_points=global_weighted_points,
        temperature=float(temperature),
    )
    stats.feature_scale = feature_scale.detach().cpu()
    return stats


@torch.no_grad()
def precompute_nystrom_statistics_multitemp_batched(
    sensitive_points,
    public_landmarks,
    temps=(0.02, 0.05, 0.2),
    ridge=1e-4,
    eps=1e-8,
    batch_size=512,
    device=None,
    feature_scale=None,
):
    """Precompute private summaries for multiple temperatures."""
    if device is None:
        device = sensitive_points.device
    if feature_scale is None:
        feature_scale = _estimate_feature_scale_from_landmarks(
            sensitive_points=sensitive_points,
            public_landmarks=public_landmarks,
            eps=eps,
            batch_size=batch_size,
            device=device,
        )
    stats = {}
    for temp in temps:
        stats[float(temp)] = precompute_nystrom_statistics_batched(
            sensitive_points=sensitive_points,
            public_landmarks=public_landmarks,
            temperature=float(temp),
            ridge=ridge,
            eps=eps,
            batch_size=batch_size,
            device=device,
            feature_scale=feature_scale,
        )
    return stats


def _alternative_normalize_drift_batched(V, D=None, eps=1e-8):
    """Normalize a whole feature group with one shared drift scale."""
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
    gather_queries = distributed_queries
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
        neg_queries = x_global if distributed_queries else x
        b_neg = _alternative_compute_exact_repulsive_barycenter_batched(
            x=neg_queries.float(),
            temperature=temp,
            eps=eps,
            mask_self=True,
        )
        if distributed_queries:
            b_neg = b_neg[:, local_query_slice, :]
        V_tau = (b_pos - b_neg).to(x.dtype)

        if max_drift_norm is not None:
            norms = torch.linalg.vector_norm(V_tau.float(), dim=-1, keepdim=True)
            scale = torch.clamp(float(max_drift_norm) / (norms + eps), max=1.0)
            V_tau = V_tau * scale.to(V_tau.dtype)

        V_total = V_total + _alternative_normalize_drift_batched(V_tau, eps=eps)

    return V_total


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
    """Multi-resolution Nyström loss in normalized feature space."""
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
    """Class-sharded multi-resolution Nyström loss in normalized feature space."""
    if len(gen_groups) != len(stats_groups):
        raise ValueError("gen_groups and stats_groups must have the same length.")

    total_loss = gen_groups[0][0].new_tensor(0.0)

    for (gen_feat, _), stats_by_shard in zip(gen_groups, stats_groups):
        with torch.no_grad():
            gen_t = gen_feat.detach().transpose(0, 1).contiguous()
            S = _feature_scale_from_sharded_stats(
                gen_t,
                stats_by_shard,
                temps=temps,
                eps=eps,
                distributed_queries=distributed_queries,
            )
            V = compute_nystrom_drift_multitemp_sharded_batched(
                x=gen_t / S,
                stats_by_shard=stats_by_shard,
                temps=temps,
                eps=eps,
                max_drift_norm=max_drift_norm,
                repulsion_mode=repulsion_mode,
                distributed_shards=distributed_shards,
                distributed_queries=distributed_queries,
            )
            target = (gen_t / S + V).transpose(0, 1)

        gen_n = gen_feat / S
        loss_group = (gen_n - target).pow(2).mean(dim=(0, 2)).sum()
        total_loss = total_loss + loss_group

    return total_loss


def _ensure_nystrom_stats_on_device(stats: NystromStats, device: torch.device) -> NystromStats:
    if (
        stats.landmarks.device == device
        and stats.A.device == device
        and stats.global_totals.device == device
        and stats.global_weighted_points.device == device
    ):
        return stats

    moved = move_nystrom_stats_to_device(stats, device)
    feature_scale = getattr(stats, "feature_scale", None)
    if feature_scale is not None:
        moved.feature_scale = feature_scale
    return moved


def _get_folded_nystrom_stats(stats: NystromStats) -> FoldedNystromStats:
    cached = getattr(stats, "_folded_stats_exact", None)
    if cached is not None:
        return cached

    A = stats.A.float()
    A = 0.5 * (A + A.transpose(-1, -2))
    weighted_points = stats.global_weighted_points.float()
    totals = stats.global_totals.float().unsqueeze(-1)
    summary_cat = torch.cat(
        [
            torch.bmm(A, weighted_points),
            torch.bmm(A, totals),
        ],
        dim=-1,
    ).contiguous()

    cached = FoldedNystromStats(
        landmarks=stats.landmarks.float().contiguous(),
        summary_cat=summary_cat,
        temperature=float(stats.temperature),
    )
    stats._folded_stats_exact = cached
    return cached


def _get_packed_sharded_folded_nystrom_stats(
    stats_by_shard: dict[int, dict[float, NystromStats]],
    temp: float,
    device: torch.device,
) -> PackedShardedFoldedNystromStats:
    cache_key = (id(stats_by_shard), float(temp), _device_key(device))
    cached = _PACKED_SHARDED_FOLDED_NYSTROM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    shard_ids = sorted(stats_by_shard.keys())
    folded_shards = []
    for shard_id in shard_ids:
        stats_by_temp = stats_by_shard[shard_id]
        stats = _ensure_nystrom_stats_on_device(stats_by_temp[float(temp)], device)
        stats_by_temp[float(temp)] = stats
        folded_shards.append(_get_folded_nystrom_stats(stats))

    cached = PackedShardedFoldedNystromStats(
        landmarks=torch.stack([stats.landmarks for stats in folded_shards], dim=0).contiguous(),
        summary_cat=torch.stack([stats.summary_cat for stats in folded_shards], dim=0).contiguous(),
        temperature=float(temp),
    )
    _PACKED_SHARDED_FOLDED_NYSTROM_CACHE[cache_key] = cached
    return cached


def drifting_loss_multires_nystrom_folded_exact(
    gen_groups,
    stats_groups,
    temps=(0.02, 0.05, 0.2),
    eps=1e-8,
    max_drift_norm=None,
    repulsion_mode="exact",
    distributed_queries=False,
):
    """Exact folded-summary Nyström loss without explicit attractive phi materialization."""
    if repulsion_mode != "exact":
        raise ValueError("Folded exact attraction currently requires repulsion_mode='exact'.")
    if len(gen_groups) != len(stats_groups):
        raise ValueError("gen_groups and stats_groups must have the same length.")

    total_loss = gen_groups[0][0].new_tensor(0.0)

    for (gen_feat, _), stats_by_temp in zip(gen_groups, stats_groups, strict=True):
        with torch.no_grad():
            gen_t = gen_feat.detach().transpose(0, 1).contiguous()
            S = _feature_scale_from_stats_by_temp(
                gen_t,
                stats_by_temp,
                temps=temps,
                eps=eps,
                distributed_queries=distributed_queries,
            )
            x = gen_t / S
            x_eval = x
            local_query_slice = slice(0, int(x.shape[1]))
            if distributed_queries:
                x_eval, local_query_slice = gather_queries_global_batch(x)

            V_total = torch.zeros_like(x)
            x_eval_float = x_eval.float()
            for temp in temps:
                temp = float(temp)
                if temp not in stats_by_temp:
                    raise KeyError(f"Missing Nyström cache for temperature {temp}.")
                stats = _ensure_nystrom_stats_on_device(stats_by_temp[temp], x.device)
                stats_by_temp[temp] = stats
                folded = _get_folded_nystrom_stats(stats)

                k_xu = _alternative_laplacian_kernel_batched(
                    x_eval_float,
                    folded.landmarks,
                    temperature=temp,
                    eps=eps,
                )
                proj = torch.bmm(k_xu, folded.summary_cat)
                b_pos = proj[..., :-1] / proj[..., -1:].clamp_min(eps)
                if distributed_queries:
                    b_pos = b_pos[:, local_query_slice, :]

                neg_queries = x_eval_float if distributed_queries else x.float()
                b_neg = _alternative_compute_exact_repulsive_barycenter_batched(
                    x=neg_queries,
                    temperature=temp,
                    eps=eps,
                    mask_self=True,
                )
                if distributed_queries:
                    b_neg = b_neg[:, local_query_slice, :]
                V_tau = (b_pos - b_neg).to(x.dtype)

                if max_drift_norm is not None:
                    norms = torch.linalg.vector_norm(V_tau.float(), dim=-1, keepdim=True)
                    scale = torch.clamp(float(max_drift_norm) / (norms + eps), max=1.0)
                    V_tau = V_tau * scale.to(V_tau.dtype)

                V_total = V_total + _alternative_normalize_drift_batched(V_tau, eps=eps)

            target = (x + V_total).transpose(0, 1)

        gen_n = gen_feat / S
        loss_group = (gen_n - target).pow(2).mean(dim=(0, 2)).sum()
        total_loss = total_loss + loss_group

    return total_loss


def drifting_loss_multires_nystrom_sharded_folded_exact(
    gen_groups,
    stats_groups,
    temps=(0.02, 0.05, 0.2),
    eps=1e-8,
    max_drift_norm=None,
    repulsion_mode="exact",
    distributed_shards=False,
    distributed_queries=False,
):
    """Exact folded-summary loss for class-sharded Nyström attraction."""
    if repulsion_mode != "exact":
        raise ValueError("Folded exact attraction currently requires repulsion_mode='exact'.")
    if len(gen_groups) != len(stats_groups):
        raise ValueError("gen_groups and stats_groups must have the same length.")

    total_loss = gen_groups[0][0].new_tensor(0.0)

    for (gen_feat, _), stats_by_shard in zip(gen_groups, stats_groups, strict=True):
        with torch.no_grad():
            gen_t = gen_feat.detach().transpose(0, 1).contiguous()
            S = _feature_scale_from_sharded_stats(
                gen_t,
                stats_by_shard,
                temps=temps,
                eps=eps,
                distributed_queries=distributed_queries,
            )
            x = gen_t / S
            x_eval = x
            local_query_slice = slice(0, int(x.shape[1]))
            if distributed_queries:
                x_eval, local_query_slice = gather_queries_global_batch(x)

            V_total = torch.zeros_like(x)
            x_eval_float = x_eval.float()
            for temp in temps:
                temp = float(temp)
                packed = _get_packed_sharded_folded_nystrom_stats(
                    stats_by_shard=stats_by_shard,
                    temp=temp,
                    device=x.device,
                )
                k_xu_all = torch.vmap(
                    lambda shard_landmarks: _alternative_laplacian_kernel_batched(
                        x_eval_float,
                        shard_landmarks,
                        temperature=temp,
                        eps=eps,
                    )
                )(packed.landmarks)
                proj_all = torch.vmap(
                    lambda k_xu, summary_cat: torch.bmm(k_xu, summary_cat)
                )(k_xu_all, packed.summary_cat)
                proj_total = proj_all.sum(dim=0)

                if distributed_shards:
                    if not dist.is_available() or not dist.is_initialized():
                        raise RuntimeError(
                            "distributed_shards=True requires torch.distributed to be initialized."
                        )
                    dist.all_reduce(proj_total, op=dist.ReduceOp.SUM)

                b_pos = proj_total[..., :-1] / proj_total[..., -1:].clamp_min(eps)
                if distributed_queries:
                    b_pos = b_pos[:, local_query_slice, :]

                neg_queries = x_eval_float if distributed_queries else x.float()
                b_neg = _alternative_compute_exact_repulsive_barycenter_batched(
                    x=neg_queries,
                    temperature=temp,
                    eps=eps,
                    mask_self=True,
                )
                if distributed_queries:
                    b_neg = b_neg[:, local_query_slice, :]
                V_tau = (b_pos - b_neg).to(x.dtype)

                if max_drift_norm is not None:
                    norms = torch.linalg.vector_norm(V_tau.float(), dim=-1, keepdim=True)
                    scale = torch.clamp(float(max_drift_norm) / (norms + eps), max=1.0)
                    V_tau = V_tau * scale.to(V_tau.dtype)

                V_total = V_total + _alternative_normalize_drift_batched(V_tau, eps=eps)

            target = (x + V_total).transpose(0, 1)

        gen_n = gen_feat / S
        loss_group = (gen_n - target).pow(2).mean(dim=(0, 2)).sum()
        total_loss = total_loss + loss_group

    return total_loss


NYSTROM_CACHE_FORMAT_VERSION = 2
LANDMARK_STRATEGIES = (
    "random_global",
    "random_per_class",
    "kmeans_per_class",
    "kmeans_global",
    "kcenter_per_class",
    "kcenter_global",
    "facility_location_per_class",
    "facility_location_global",
    "density_weighted_kcenter_per_class",
)


def get_host_sync_timeout():
    timeout_seconds = int(os.environ.get("HOST_SYNC_TIMEOUT_SEC", "7200"))
    if timeout_seconds <= 0:
        raise ValueError("HOST_SYNC_TIMEOUT_SEC must be positive.")
    return timedelta(seconds=timeout_seconds)


HOST_SYNC_TIMEOUT = get_host_sync_timeout()


def write_training_config_snapshot(
    out_dir,
    args,
    cfg,
    unet_cfg,
    encoder_input_size,
    model_num_params,
    feat_encoder_num_params,
):
    """Write the exact resolved training config for this run."""
    resolved_cfg = asdict(cfg)
    resolved_cfg["encoder_input_size"] = encoder_input_size
    resolved_cfg["output_dir"] = out_dir
    resolved_cfg["large_model"] = bool(args.large)

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
                "feature_encoder_compiled": True,
            },
        ),
    ]

    lines = ["MultiRes Drift Training Configuration", ""]
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
    """Append FID rows to the CSV log.

    Keeps the legacy ``fid`` column as a raw-FID alias for backward compatibility.
    """
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([step, fid_raw, fid_ema, fid_raw])


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_fid_eval_args(eval_weights):
    return SimpleNamespace(
        batch_size=256,
        fid_batch_size=50,
        fid_num_workers=0,
        fid_timeout=None,
        compile=False,
        eval_weights=normalize_eval_weights(eval_weights),
    )


def evaluate_fid_from_checkpoint(checkpoint_path, ref_dir, device, eval_weights):
    checkpoint_path = Path(checkpoint_path).resolve()
    eval_weights = normalize_eval_weights(eval_weights)
    fake_dir = checkpoint_path.parent.parent / "evaluation" / f"{checkpoint_path.stem}_fid_10000_{eval_weights}"
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


def build_live_fid_dir(out_dir, step, eval_weights):
    eval_weights = normalize_eval_weights(eval_weights)
    return Path(out_dir) / "evaluation" / f"drift_step{int(step):07d}_fid_10000_{eval_weights}"


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

    eval_args = build_fid_eval_args("raw")
    result = evaluate_loaded_model(
        model=model,
        fake_dir=build_live_fid_dir(out_dir, step, "raw"),
        ref_dir=Path(ref_dir),
        device=device,
        args=eval_args,
        dataset_name=dataset_name,
        eval_weights="raw",
        metadata={"source": "live_model", "step": int(step)},
    )
    return {"raw": result["fid"]}


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


def serialize_stats_by_temp(stats_by_temp):
    """Move one feature-group cache to a CPU-serializable payload."""
    if not stats_by_temp:
        raise ValueError("Encountered an empty stats group while serializing.")

    first_stats = next(iter(stats_by_temp.values()))
    feature_scale = getattr(first_stats, "feature_scale", None)
    temps_payload = {}
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


def restore_stats_by_temp(group_payload, device):
    landmarks_cpu = group_payload["landmarks"]
    if not isinstance(landmarks_cpu, torch.Tensor):
        raise TypeError("Invalid cache payload: missing landmarks tensor.")
    landmarks = landmarks_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    feature_scale = group_payload.get("feature_scale")

    temps_payload = group_payload["temps"]
    if not isinstance(temps_payload, dict):
        raise TypeError("Invalid cache payload: missing per-temperature stats.")

    stats_by_temp = {}
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


def serialize_stats_groups(stats_groups):
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
        shards_payload = {}
        for class_id, stats_by_temp in stats_by_class.items():
            shards_payload[int(class_id)] = serialize_stats_by_temp(stats_by_temp)
        serialized_groups.append({"shards": shards_payload})
    return serialized_groups


def restore_stats_groups(serialized_groups, device):
    stats_groups = []

    for group_payload in serialized_groups:
        if "shards" in group_payload:
            shards_payload = group_payload["shards"]
            if not isinstance(shards_payload, dict):
                raise TypeError("Invalid cache payload: malformed shard entry.")
            stats_by_class = {}
            for class_id, shard_payload in shards_payload.items():
                if not isinstance(shard_payload, dict):
                    raise TypeError("Invalid cache payload: malformed shard payload.")
                stats_by_class[int(class_id)] = restore_stats_by_temp(shard_payload, device=device)
            stats_groups.append(stats_by_class)
        else:
            stats_groups.append(restore_stats_by_temp(group_payload, device=device))

    return stats_groups


def _normalize_shard_class_groups(shard_class_groups):
    if shard_class_groups is None:
        return None

    normalized_groups = []
    for shard_group in shard_class_groups:
        if isinstance(shard_group, (int, str)):
            normalized_groups.append([int(shard_group)])
            continue
        normalized_groups.append([int(class_id) for class_id in shard_group])
    return normalized_groups


def save_nystrom_cache(cache_path, stats_groups, total_landmarks, shard_class_groups):
    cache_path = Path(cache_path)
    normalized_groups = _normalize_shard_class_groups(shard_class_groups)
    singleton_ids = None
    if normalized_groups is not None and all(len(shard_group) == 1 for shard_group in normalized_groups):
        singleton_ids = [int(shard_group[0]) for shard_group in normalized_groups]
    payload = {
        "format_version": NYSTROM_CACHE_FORMAT_VERSION,
        "total_landmarks": int(total_landmarks),
        "shard_class_ids": singleton_ids,
        "shard_class_groups": normalized_groups,
        "stats_groups": serialize_stats_groups(stats_groups),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def load_nystrom_cache(cache_path, device):
    cache_path = Path(cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != NYSTROM_CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Nyström cache format version: {payload.get('format_version')}"
        )

    total_landmarks = int(payload["total_landmarks"])
    shard_class_groups = payload.get("shard_class_groups")
    if shard_class_groups is None:
        shard_class_ids = payload.get("shard_class_ids")
        if shard_class_ids is not None:
            shard_class_groups = [[int(class_id)] for class_id in shard_class_ids]
    else:
        shard_class_groups = _normalize_shard_class_groups(shard_class_groups)

    stats_groups = restore_stats_groups(payload["stats_groups"], device=device)
    return stats_groups, total_landmarks, shard_class_groups


def precompute_features(encoder, dataset, device, batch_size=256, verbose=False):
    """Pre-compute features for all images in the dataset."""
    if verbose:
        print("Pre-computing real image features...")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    all_feats_by_group = None
    n_processed = 0

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            groups = encoder(ensure_three_channels(images))

            if all_feats_by_group is None:
                all_feats_by_group = [[] for _ in groups]

            for i, (feat, _) in enumerate(groups):
                all_feats_by_group[i].append(feat.float().cpu())

            n_processed += images.shape[0]
            if verbose and n_processed % 10000 == 0:
                print(f"  {n_processed}/{len(dataset)} images")

    result_feats = []
    result_cjs = []
    for i, feat_list in enumerate(all_feats_by_group):
        cat = torch.cat(feat_list, dim=0)
        result_feats.append(cat)
        result_cjs.append(groups[i][1])

    if verbose:
        total_bytes = sum(f.numel() * f.element_size() for f in result_feats)
        print(f"  Pre-computed {n_processed} images, {total_bytes / 1e9:.2f} GB on CPU")
        for i, (feat, c_j) in enumerate(zip(result_feats, result_cjs)):
            print(f"    Group {i}: shape={list(feat.shape)}, C_j={c_j}")

    return result_feats, result_cjs


@torch.no_grad()
def _run_kmeans(points, num_clusters, seed, num_iters=20, chunk_size=1024):
    """Run deterministic Lloyd k-means on flattened points."""
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


def allocate_balanced_landmarks_per_class(class_counts, num_landmarks, seed):
    class_ids = sorted(int(class_id) for class_id in class_counts)
    if not class_ids:
        raise ValueError("No classes found while allocating landmark budgets.")

    total_available = sum(int(class_counts[class_id]) for class_id in class_ids)
    if int(num_landmarks) > total_available:
        raise ValueError(
            f"Requested {num_landmarks} total landmarks from only {total_available} available samples."
        )

    base = int(num_landmarks) // len(class_ids)
    remainder = int(num_landmarks) % len(class_ids)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(len(class_ids), generator=generator).tolist()

    targets = {class_id: base for class_id in class_ids}
    for idx in perm[:remainder]:
        targets[class_ids[idx]] += 1

    deficit = 0
    for class_id in class_ids:
        available = int(class_counts[class_id])
        if targets[class_id] > available:
            deficit += targets[class_id] - available
            targets[class_id] = available

    if deficit:
        for idx in perm:
            class_id = class_ids[idx]
            spare = int(class_counts[class_id]) - targets[class_id]
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


def build_balanced_class_shard_groups(labels, classes_per_shard):
    """Pack classes into the minimum number of shards with a fixed class cap per shard."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    class_ids = torch.unique(labels, sorted=True).tolist()
    if not class_ids:
        raise ValueError("No classes were found for class sharding.")

    classes_per_shard = int(classes_per_shard)
    if classes_per_shard <= 0:
        raise ValueError("classes_per_shard must be positive.")

    classes_per_shard = min(classes_per_shard, len(class_ids))
    shard_count = math.ceil(len(class_ids) / classes_per_shard)
    class_counts = {
        int(class_id): int(torch.count_nonzero(labels == class_id).item())
        for class_id in class_ids
    }

    shard_groups = [[] for _ in range(shard_count)]
    shard_sample_totals = [0 for _ in range(shard_count)]
    for class_id in sorted(class_ids, key=lambda cid: (-class_counts[int(cid)], int(cid))):
        eligible = [
            shard_id
            for shard_id, shard_group in enumerate(shard_groups)
            if len(shard_group) < classes_per_shard
        ]
        shard_id = min(
            eligible,
            key=lambda idx: (shard_sample_totals[idx], len(shard_groups[idx]), idx),
        )
        shard_groups[shard_id].append(int(class_id))
        shard_sample_totals[shard_id] += class_counts[int(class_id)]

    normalized_groups = []
    for shard_group in shard_groups:
        shard_group.sort()
        if shard_group:
            normalized_groups.append(shard_group)
    return normalized_groups


def _sample_random_indices(num_points, num_select, seed, device):
    if int(num_select) <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if int(num_select) >= int(num_points):
        return torch.arange(num_points, device=device, dtype=torch.long)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randperm(num_points, generator=generator, device=device)[: int(num_select)]


def _chunked_distances_to_centers(points, centers, chunk_size=1024):
    num_points = points.shape[0]
    dists = torch.empty(num_points, centers.shape[0], device=points.device, dtype=torch.float32)
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        dists[start:end] = torch.cdist(points[start:end].float(), centers.float(), p=2)
    return dists


def _chunked_distances_to_point(points, point, chunk_size=1024):
    num_points = points.shape[0]
    dists = torch.empty(num_points, device=points.device, dtype=torch.float32)
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        dists[start:end] = torch.cdist(points[start:end].float(), point.float(), p=2).squeeze(1)
    return dists


def _select_nearest_unused_point(points, centroid, selected_mask, chunk_size=1024):
    best_idx = None
    best_dist = None
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        chunk = points[start:end]
        dists = torch.cdist(chunk.float(), centroid.float(), p=2).squeeze(1)
        local_mask = ~selected_mask[start:end]
        if not local_mask.any():
            continue
        local_candidates = torch.nonzero(local_mask, as_tuple=False).squeeze(1)
        local_dists = dists.index_select(0, local_candidates)
        local_best_pos = int(local_dists.argmin().item())
        local_best_idx = int(local_candidates[local_best_pos].item()) + start
        local_best_dist = float(local_dists[local_best_pos].item())
        if best_dist is None or local_best_dist < best_dist:
            best_dist = local_best_dist
            best_idx = local_best_idx

    if best_idx is None:
        raise RuntimeError("Could not find an unused fallback point for landmark selection.")
    return best_idx


def _select_kmeans_indices(points, num_select, seed, num_iters=20, chunk_size=1024):
    if int(num_select) <= 0:
        return torch.empty(0, dtype=torch.long, device=points.device)
    if int(num_select) >= int(points.shape[0]):
        return torch.arange(points.shape[0], device=points.device, dtype=torch.long)

    centroids, assignments = _run_kmeans(
        points=points,
        num_clusters=int(num_select),
        seed=seed,
        num_iters=num_iters,
        chunk_size=chunk_size,
    )

    selected = []
    selected_mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for cluster_id in range(centroids.shape[0]):
        centroid = centroids[cluster_id : cluster_id + 1]
        member_idx = torch.nonzero(assignments == cluster_id, as_tuple=False).squeeze(1)
        chosen_idx = None

        if member_idx.numel() > 0:
            member_points = points.index_select(0, member_idx)
            dists = torch.cdist(member_points.float(), centroid.float(), p=2).squeeze(1)
            order = dists.argsort()
            for pos in order.tolist():
                candidate_idx = int(member_idx[pos].item())
                if not selected_mask[candidate_idx]:
                    chosen_idx = candidate_idx
                    break

        if chosen_idx is None:
            chosen_idx = _select_nearest_unused_point(
                points=points,
                centroid=centroid,
                selected_mask=selected_mask,
                chunk_size=chunk_size,
            )

        selected.append(chosen_idx)
        selected_mask[chosen_idx] = True

    return torch.as_tensor(selected, device=points.device, dtype=torch.long)


def _select_kcenter_indices(points, num_select, seed, density_weights=None, chunk_size=1024):
    if int(num_select) <= 0:
        return torch.empty(0, dtype=torch.long, device=points.device)

    num_points = int(points.shape[0])
    if int(num_select) >= num_points:
        return torch.arange(num_points, device=points.device, dtype=torch.long)

    generator = torch.Generator(device=points.device).manual_seed(int(seed))
    first_idx = int(torch.randint(num_points, (1,), generator=generator, device=points.device).item())
    selected = [first_idx]
    selected_mask = torch.zeros(num_points, dtype=torch.bool, device=points.device)
    selected_mask[first_idx] = True

    min_dists = _chunked_distances_to_point(
        points=points,
        point=points[first_idx : first_idx + 1],
        chunk_size=chunk_size,
    )

    density_weights = None if density_weights is None else density_weights.to(device=points.device, dtype=torch.float32)

    for _ in range(1, int(num_select)):
        scores = min_dists if density_weights is None else min_dists * density_weights
        scores = scores.clone()
        scores[selected_mask] = -1.0
        next_idx = int(scores.argmax().item())
        if float(scores[next_idx].item()) < 0:
            unused = torch.nonzero(~selected_mask, as_tuple=False).squeeze(1)
            next_idx = int(unused[0].item())

        selected.append(next_idx)
        selected_mask[next_idx] = True
        new_dists = _chunked_distances_to_point(
            points=points,
            point=points[next_idx : next_idx + 1],
            chunk_size=chunk_size,
        )
        min_dists = torch.minimum(min_dists, new_dists)

    return torch.as_tensor(selected, device=points.device, dtype=torch.long)


def _estimate_density_weights(points, seed, knn_k=32, reference_size=1024, chunk_size=1024):
    num_points = int(points.shape[0])
    if num_points <= 1:
        return torch.ones(num_points, device=points.device, dtype=torch.float32)

    k = max(1, min(int(knn_k), num_points - 1))
    ref_size = max(k + 1, min(num_points, int(reference_size)))
    generator = torch.Generator(device=points.device).manual_seed(int(seed))
    ref_idx = torch.randperm(num_points, generator=generator, device=points.device)[:ref_size]
    refs = points.index_select(0, ref_idx)
    refs_cover_all_points = (ref_size == num_points)

    mean_knn = torch.empty(num_points, device=points.device, dtype=torch.float32)
    take_k = min(ref_size, k + 1 if refs_cover_all_points else k)
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        dists = torch.cdist(points[start:end].float(), refs.float(), p=2)
        knn_vals = dists.topk(k=take_k, largest=False).values
        if refs_cover_all_points and knn_vals.shape[1] > 1:
            knn_vals = knn_vals[:, 1:]
        mean_knn[start:end] = knn_vals.mean(dim=1)

    density = (mean_knn + 1e-8).reciprocal()
    if num_points >= 8:
        q05 = torch.quantile(density, 0.05)
        q95 = torch.quantile(density, 0.95)
        density = density.clamp(min=q05, max=q95)
    density = density / density.max().clamp_min(1e-8)
    return density.clamp_min(0.25)


def _select_facility_location_indices(
    points,
    num_select,
    seed,
    temperature=0.05,
    candidate_ratio=4,
    eval_ratio=8,
    chunk_size=1024,
):
    if int(num_select) <= 0:
        return torch.empty(0, dtype=torch.long, device=points.device)

    num_points = int(points.shape[0])
    if int(num_select) >= num_points:
        return torch.arange(num_points, device=points.device, dtype=torch.long)

    candidate_size = min(num_points, max(int(num_select), int(candidate_ratio) * int(num_select)))
    eval_size = min(num_points, max(int(num_select), int(eval_ratio) * int(num_select)))

    generator = torch.Generator(device=points.device).manual_seed(int(seed))
    candidate_idx = torch.randperm(num_points, generator=generator, device=points.device)[:candidate_size]
    eval_idx = torch.randperm(num_points, generator=generator, device=points.device)[:eval_size]

    candidate_points = points.index_select(0, candidate_idx)
    eval_points = points.index_select(0, eval_idx)
    scale = _alternative_temperature_scale(temperature, points.shape[1])

    sims = torch.empty(eval_size, candidate_size, device=points.device, dtype=torch.float32)
    for start in range(0, eval_size, chunk_size):
        end = min(start + chunk_size, eval_size)
        dists = torch.cdist(eval_points[start:end].float(), candidate_points.float(), p=2)
        sims[start:end] = torch.exp(-dists / scale)

    selected_positions = []
    selected_mask = torch.zeros(candidate_size, dtype=torch.bool, device=points.device)
    best_cover = torch.zeros(eval_size, device=points.device, dtype=torch.float32)

    for _ in range(int(num_select)):
        gains = torch.clamp(sims - best_cover.unsqueeze(1), min=0).sum(dim=0)
        gains[selected_mask] = -1.0
        pos = int(gains.argmax().item())
        if float(gains[pos].item()) < 0:
            unused = torch.nonzero(~selected_mask, as_tuple=False).squeeze(1)
            pos = int(unused[0].item())
        selected_positions.append(pos)
        selected_mask[pos] = True
        best_cover = torch.maximum(best_cover, sims[:, pos])

    return candidate_idx.index_select(
        0,
        torch.as_tensor(selected_positions, device=points.device, dtype=torch.long),
    )


def _parse_landmark_strategy(strategy):
    if strategy not in LANDMARK_STRATEGIES:
        raise ValueError(
            f"Unsupported --nystrom-landmark-strategy={strategy}. "
            f"Expected one of {LANDMARK_STRATEGIES}."
        )
    if strategy.endswith("_per_class"):
        return strategy[: -len("_per_class")], "per_class"
    if strategy.endswith("_global"):
        return strategy[: -len("_global")], "global"
    raise ValueError(f"Could not parse landmark strategy suffix for {strategy}.")


def _select_strategy_indices_from_points(
    points,
    num_select,
    strategy_base,
    seed,
    kmeans_iters,
    primary_temp,
    facility_candidate_ratio,
    facility_eval_ratio,
    density_knn_k,
    density_reference_size,
):
    if strategy_base == "random":
        return _sample_random_indices(points.shape[0], num_select, seed=seed, device=points.device)
    if strategy_base == "kmeans":
        return _select_kmeans_indices(
            points=points,
            num_select=num_select,
            seed=seed,
            num_iters=kmeans_iters,
        )
    if strategy_base == "kcenter":
        return _select_kcenter_indices(points=points, num_select=num_select, seed=seed)
    if strategy_base == "facility_location":
        return _select_facility_location_indices(
            points=points,
            num_select=num_select,
            seed=seed,
            temperature=primary_temp,
            candidate_ratio=facility_candidate_ratio,
            eval_ratio=facility_eval_ratio,
        )
    if strategy_base == "density_weighted_kcenter":
        density_weights = _estimate_density_weights(
            points=points,
            seed=seed,
            knn_k=density_knn_k,
            reference_size=density_reference_size,
        )
        return _select_kcenter_indices(
            points=points,
            num_select=num_select,
            seed=seed,
            density_weights=density_weights,
        )
    raise ValueError(f"Unsupported landmark selector base strategy: {strategy_base}.")


def compute_landmark_metrics(
    flat_points,
    landmark_points,
    labels,
    selected_indices,
    temperature,
    ridge,
    class_names=None,
    chunk_size=1024,
):
    num_points = int(flat_points.shape[0])
    num_landmarks = int(landmark_points.shape[0])
    num_locations = int(landmark_points.shape[1])
    feat_dim = int(landmark_points.shape[2])
    flat_landmarks = landmark_points.reshape(num_landmarks, -1)
    flat_scale = _alternative_temperature_scale(temperature, flat_points.shape[1])

    min_dist_chunks = []
    max_kernel_chunks = []
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        dists = torch.cdist(flat_points[start:end].float(), flat_landmarks.float(), p=2)
        min_dist_chunks.append(dists.min(dim=1).values.cpu())
        max_kernel_chunks.append(torch.exp(-dists / flat_scale).max(dim=1).values.cpu())

    min_dists = torch.cat(min_dist_chunks, dim=0)
    max_kernels = torch.cat(max_kernel_chunks, dim=0)

    pairwise_dists = torch.cdist(flat_landmarks.float(), flat_landmarks.float(), p=2)
    if num_landmarks > 1:
        tri = torch.triu_indices(num_landmarks, num_landmarks, offset=1, device=pairwise_dists.device)
        landmark_spread = pairwise_dists[tri[0], tri[1]].cpu()
        mean_landmark_pairwise_dist = float(landmark_spread.mean().item())
        median_landmark_pairwise_dist = float(landmark_spread.median().item())
    else:
        mean_landmark_pairwise_dist = 0.0
        median_landmark_pairwise_dist = 0.0

    landmark_t = landmark_points.transpose(0, 1).contiguous()
    location_scale = _alternative_temperature_scale(temperature, feat_dim)
    W = torch.exp(-torch.cdist(landmark_t.float(), landmark_t.float(), p=2) / location_scale)
    eye = torch.eye(num_landmarks, device=W.device, dtype=W.dtype).unsqueeze(0).expand(num_locations, -1, -1)
    eigvals = torch.linalg.eigvalsh(W + float(ridge) * eye).cpu()
    cond = eigvals[:, -1] / eigvals[:, 0].clamp_min(1e-8)

    labels = torch.as_tensor(labels, dtype=torch.long)
    selected_labels = labels.index_select(0, selected_indices.detach().cpu())
    class_hist = torch.bincount(selected_labels, minlength=int(labels.max().item()) + 1)
    selected_class_hist = {}
    for class_id, count in enumerate(class_hist.tolist()):
        if count <= 0:
            continue
        key = str(class_id)
        if class_names is not None and class_id < len(class_names):
            key = f"{class_id}:{class_names[class_id]}"
        selected_class_hist[key] = int(count)

    return {
        "num_points": num_points,
        "num_landmarks": num_landmarks,
        "num_locations": num_locations,
        "feat_dim": feat_dim,
        "mean_min_dist": float(min_dists.mean().item()),
        "p95_min_dist": float(torch.quantile(min_dists, 0.95).item()),
        "max_min_dist": float(min_dists.max().item()),
        "mean_max_flat_kernel": float(max_kernels.mean().item()),
        "p05_max_flat_kernel": float(torch.quantile(max_kernels, 0.05).item()),
        "mean_landmark_pairwise_dist": mean_landmark_pairwise_dist,
        "median_landmark_pairwise_dist": median_landmark_pairwise_dist,
        "w_min_eig_mean": float(eigvals[:, 0].mean().item()),
        "w_min_eig_min": float(eigvals[:, 0].min().item()),
        "w_max_eig_mean": float(eigvals[:, -1].mean().item()),
        "w_condition_mean": float(cond.mean().item()),
        "w_condition_p95": float(torch.quantile(cond, 0.95).item()),
        "selected_class_hist": selected_class_hist,
    }


@torch.no_grad()
def build_landmarks_for_group(
    group_features,
    labels,
    strategy,
    seed,
    device,
    temperatures,
    ridge,
    kmeans_iters=20,
    landmarks_per_class=None,
    total_landmarks=None,
    landmark_class=None,
    facility_candidate_ratio=4,
    facility_eval_ratio=8,
    density_knn_k=32,
    density_reference_size=1024,
    class_names=None,
):
    if group_features.ndim != 3:
        raise ValueError("group_features must have shape [N, L, D].")

    labels_cpu = torch.as_tensor(labels, dtype=torch.long)
    num_points, num_locations, feat_dim = group_features.shape
    if labels_cpu.numel() != num_points:
        raise ValueError("labels must match the number of samples in group_features.")

    strategy_base, strategy_scope = _parse_landmark_strategy(strategy)
    if strategy_scope == "global" and landmark_class is not None:
        raise ValueError(
            f"--nystrom-landmark-class cannot be combined with global strategy {strategy}."
        )

    class_ids = torch.unique(labels_cpu, sorted=True).tolist()
    if landmark_class is not None:
        landmark_class = int(landmark_class)
        if landmark_class not in class_ids:
            raise ValueError(
                f"Requested landmark_class={landmark_class}, but available classes are {class_ids}."
            )
        class_ids = [landmark_class]

    points = group_features.to(device=device, dtype=torch.float32, non_blocking=True).reshape(num_points, -1)
    primary_temp = float(temperatures[0]) if temperatures else 0.05

    if total_landmarks is not None:
        total_budget = int(total_landmarks)
    else:
        if landmarks_per_class is None:
            raise ValueError(
                "Either --nystrom-total-landmarks or --nystrom-landmarks-per-class must be set."
            )
        if strategy_scope == "global":
            total_budget = int(landmarks_per_class) * len(class_ids)
        else:
            total_budget = int(landmarks_per_class) * len(class_ids)

    if total_budget <= 0:
        raise ValueError("Nyström landmark budget must be positive.")
    if total_budget > num_points:
        raise ValueError(
            f"Requested {total_budget} landmarks from only {num_points} available samples."
        )

    start_time = time.time()
    if strategy_scope == "global":
        selected_indices = _select_strategy_indices_from_points(
            points=points,
            num_select=total_budget,
            strategy_base=strategy_base,
            seed=seed,
            kmeans_iters=kmeans_iters,
            primary_temp=primary_temp,
            facility_candidate_ratio=facility_candidate_ratio,
            facility_eval_ratio=facility_eval_ratio,
            density_knn_k=density_knn_k,
            density_reference_size=density_reference_size,
        )
        target_counts = None
    else:
        class_counts = {
            int(class_id): int(torch.sum(labels_cpu == class_id).item())
            for class_id in class_ids
        }
        if total_landmarks is not None:
            if landmark_class is not None:
                target_counts = {int(landmark_class): total_budget}
            else:
                target_counts = allocate_balanced_landmarks_per_class(
                    class_counts=class_counts,
                    num_landmarks=total_budget,
                    seed=seed,
                )
        else:
            target_counts = {int(class_id): int(landmarks_per_class) for class_id in class_ids}

        selected_parts = []
        for class_id in class_ids:
            target = int(target_counts.get(int(class_id), 0))
            if target <= 0:
                continue
            class_indices_cpu = torch.nonzero(labels_cpu == class_id, as_tuple=False).squeeze(1)
            if target > int(class_indices_cpu.numel()):
                raise ValueError(
                    f"Requested {target} landmarks for class {class_id} but only "
                    f"{class_indices_cpu.numel()} points are available."
                )
            class_indices = class_indices_cpu.to(device=device, non_blocking=True)
            class_points = points.index_select(0, class_indices)
            local_indices = _select_strategy_indices_from_points(
                points=class_points,
                num_select=target,
                strategy_base=strategy_base,
                seed=int(seed) + int(class_id) * 1009,
                kmeans_iters=kmeans_iters,
                primary_temp=primary_temp,
                facility_candidate_ratio=facility_candidate_ratio,
                facility_eval_ratio=facility_eval_ratio,
                density_knn_k=density_knn_k,
                density_reference_size=density_reference_size,
            )
            selected_parts.append(class_indices.index_select(0, local_indices))

        if not selected_parts:
            raise ValueError("No landmarks were selected.")
        selected_indices = torch.cat(selected_parts, dim=0)

    selection_time_s = time.time() - start_time
    landmark_points = points.index_select(0, selected_indices).reshape(-1, num_locations, feat_dim)
    metrics = compute_landmark_metrics(
        flat_points=points,
        landmark_points=landmark_points,
        labels=labels_cpu,
        selected_indices=selected_indices,
        temperature=primary_temp,
        ridge=ridge,
        class_names=class_names,
    )
    metrics.update(
        {
            "strategy": strategy,
            "strategy_base": strategy_base,
            "strategy_scope": strategy_scope,
            "selection_time_s": float(selection_time_s),
            "landmark_budget": int(total_budget),
            "landmarks_per_class": None if landmarks_per_class is None else int(landmarks_per_class),
            "target_counts": None if target_counts is None else {str(k): int(v) for k, v in target_counts.items()},
        }
    )
    return landmark_points.cpu(), metrics


def build_selection_metrics_report(selection_metrics, cfg, total_landmarks):
    scalar_keys = (
        "selection_time_s",
        "mean_min_dist",
        "p95_min_dist",
        "max_min_dist",
        "mean_max_flat_kernel",
        "p05_max_flat_kernel",
        "mean_landmark_pairwise_dist",
        "median_landmark_pairwise_dist",
        "w_min_eig_mean",
        "w_min_eig_min",
        "w_max_eig_mean",
        "w_condition_mean",
        "w_condition_p95",
    )

    aggregate = {}
    if selection_metrics:
        for key in scalar_keys:
            values = [float(group[key]) for group in selection_metrics if key in group]
            if values:
                aggregate[key] = float(sum(values) / len(values))

    return {
        "strategy": cfg.nystrom_landmark_strategy,
        "temperatures": [float(temp) for temp in cfg.temperatures],
        "ridge": float(cfg.nystrom_ridge),
        "landmarks_per_class": int(cfg.nystrom_landmarks_per_class),
        "total_landmark_budget": (
            None if cfg.nystrom_total_landmarks is None else int(cfg.nystrom_total_landmarks)
        ),
        "total_landmarks_selected": int(total_landmarks),
        "groups": selection_metrics,
        "aggregate": aggregate,
    }


@torch.no_grad()
def build_kmeans_landmarks_from_points(group_features, num_landmarks, seed, device, num_iters=20):
    """Build k-means landmarks from a single pool of group features."""
    if group_features.ndim != 3:
        raise ValueError("group_features must have shape [N, L, D].")
    if group_features.shape[0] == 0:
        raise ValueError("group_features must not be empty.")

    group_features = group_features.to(device=device, dtype=torch.float32, non_blocking=True)
    n, l, d = group_features.shape
    flat_features = group_features.reshape(n, l * d)
    centroids, assignments = _run_kmeans(
        points=flat_features,
        num_clusters=num_landmarks,
        seed=seed,
        num_iters=num_iters,
    )

    landmarks = []
    for cluster_id in range(centroids.shape[0]):
        member_mask = assignments == cluster_id
        if member_mask.any():
            member_points = flat_features[member_mask]
            dists = torch.cdist(centroids[cluster_id:cluster_id + 1].float(), member_points.float(), p=2)
            nearest_idx = dists.argmin(dim=1).item()
            landmarks.append(member_points[nearest_idx])
        else:
            dists = torch.cdist(centroids[cluster_id:cluster_id + 1].float(), flat_features.float(), p=2)
            nearest_idx = dists.argmin(dim=1).item()
            landmarks.append(flat_features[nearest_idx])

    return torch.stack(landmarks, dim=0).reshape(-1, l, d)


@torch.no_grad()
def build_kmeans_landmarks(
    group_features,
    labels,
    landmarks_per_class,
    seed,
    device,
    num_iters=20,
    landmark_class=None,
):
    """Build per-class k-means landmarks from cluster member samples."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.ndim != 1:
        raise ValueError("labels must be 1D.")
    if group_features.shape[0] != labels.numel():
        raise ValueError("labels must match the number of samples in group_features.")

    landmarks = []
    class_ids = torch.unique(labels, sorted=True).tolist()

    if landmark_class is not None:
        landmark_class = int(landmark_class)
        if landmark_class not in class_ids:
            raise ValueError(
                f"Requested landmark_class={landmark_class}, "
                f"but available classes are {class_ids}."
            )
        class_ids = [landmark_class]

    for class_id in class_ids:
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).squeeze(1)
        if class_indices.numel() == 0:
            continue
        class_features = group_features.index_select(0, class_indices)
        landmarks.append(
            build_kmeans_landmarks_from_points(
                group_features=class_features,
                num_landmarks=landmarks_per_class,
                seed=int(seed) + int(class_id),
                device=device,
                num_iters=num_iters,
            )
        )

    if not landmarks:
        raise ValueError("No landmarks were produced.")

    return torch.cat(landmarks, dim=0)


def build_nystrom_stats_groups(
    feature_groups,
    labels,
    temps,
    landmarks_per_class,
    total_landmark_budget,
    landmark_strategy,
    landmark_class,
    ridge,
    landmark_seed,
    device,
    batch_size=512,
    kmeans_iters=20,
    facility_candidate_ratio=4,
    facility_eval_ratio=8,
    density_knn_k=32,
    density_reference_size=1024,
    class_names=None,
):
    """Build Nyström caches for each feature group using original-sample landmarks."""
    if not feature_groups:
        raise ValueError("feature_groups must not be empty.")

    num_points = feature_groups[0].shape[0]
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.numel() != num_points:
        raise ValueError("labels must match the number of samples in each feature group.")

    stats_groups = []
    total_landmarks = None
    selection_metrics = []

    for group_idx, features in enumerate(feature_groups):
        if features.shape[0] != num_points:
            raise ValueError("All feature groups must have the same number of samples.")

        public_landmarks, group_metrics = build_landmarks_for_group(
            group_features=features,
            labels=labels,
            strategy=landmark_strategy,
            temperatures=temps,
            ridge=ridge,
            landmarks_per_class=landmarks_per_class,
            total_landmarks=total_landmark_budget,
            seed=landmark_seed,
            device=device,
            kmeans_iters=kmeans_iters,
            landmark_class=landmark_class,
            facility_candidate_ratio=facility_candidate_ratio,
            facility_eval_ratio=facility_eval_ratio,
            density_knn_k=density_knn_k,
            density_reference_size=density_reference_size,
            class_names=class_names,
        )
        stats_by_temp = precompute_nystrom_statistics_multitemp_batched(
            sensitive_points=features,
            public_landmarks=public_landmarks,
            temps=temps,
            ridge=ridge,
            batch_size=batch_size,
            device=device,
        )
        stats_groups.append(stats_by_temp)
        if total_landmarks is None:
            total_landmarks = public_landmarks.shape[0]
        group_metrics["group_index"] = int(group_idx)
        selection_metrics.append(group_metrics)
        del public_landmarks

    return stats_groups, total_landmarks, selection_metrics


def build_nystrom_stats_groups_sharded(
    feature_groups,
    labels,
    temps,
    landmarks_per_class,
    classes_per_shard,
    ridge,
    landmark_seed,
    device,
    batch_size=512,
    kmeans_iters=20,
    cpu_offload=False,
):
    """Build Nyström cache shards from balanced groups of classes."""
    if not feature_groups:
        raise ValueError("feature_groups must not be empty.")

    num_points = feature_groups[0].shape[0]
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.numel() != num_points:
        raise ValueError("labels must match the number of samples in each feature group.")

    class_ids = torch.unique(labels, sorted=True).tolist()
    if not class_ids:
        raise ValueError("No classes were found for class-sharded Nyström.")
    shard_class_groups = build_balanced_class_shard_groups(
        labels=labels,
        classes_per_shard=classes_per_shard,
    )

    class_indices = {
        class_id: torch.nonzero(labels == class_id, as_tuple=False).squeeze(1)
        for class_id in class_ids
    }

    stats_groups = []
    total_landmarks = None

    for group_idx, features in enumerate(feature_groups):
        if features.shape[0] != num_points:
            raise ValueError("All feature groups must have the same number of samples.")

        print(
            f"Building class-sharded Nyström cache for feature group {group_idx + 1}/{len(feature_groups)} "
            f"across {len(class_ids)} classes into {len(shard_class_groups)} shards..."
        )
        stats_by_shard = {}
        group_total_landmarks = 0
        landmarks_by_class = {}

        for class_id in class_ids:
            class_features = features.index_select(0, class_indices[class_id])
            landmarks_by_class[class_id] = build_kmeans_landmarks_from_points(
                group_features=class_features,
                num_landmarks=landmarks_per_class,
                seed=int(landmark_seed) + int(class_id),
                device=device,
                num_iters=kmeans_iters,
            )
            group_total_landmarks += int(landmarks_by_class[class_id].shape[0])

        group_feature_scale = _estimate_feature_scale_from_landmarks(
            sensitive_points=features,
            public_landmarks=torch.cat(list(landmarks_by_class.values()), dim=0),
            batch_size=batch_size,
            device=device,
        )
        print(f"  shared_feature_scale={float(group_feature_scale.item()):.6f}")

        for shard_id, shard_class_ids in enumerate(shard_class_groups):
            shard_indices = torch.cat(
                [class_indices[class_id] for class_id in shard_class_ids],
                dim=0,
            )
            shard_features = features.index_select(0, shard_indices)
            public_landmarks = torch.cat(
                [landmarks_by_class[class_id] for class_id in shard_class_ids],
                dim=0,
            )
            stats_by_shard[shard_id] = precompute_nystrom_statistics_multitemp_batched(
                sensitive_points=shard_features,
                public_landmarks=public_landmarks,
                temps=temps,
                ridge=ridge,
                batch_size=batch_size,
                device=device,
                feature_scale=group_feature_scale,
            )
            if cpu_offload:
                stats_by_shard[shard_id] = {
                    float(temp): move_nystrom_stats_to_device(stats, "cpu")
                    for temp, stats in stats_by_shard[shard_id].items()
                }
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            print(
                f"  shard {shard_id}: classes={shard_class_ids} "
                f"samples={shard_features.shape[0]} "
                f"landmarks={public_landmarks.shape[0]} "
                f"storage={'cpu' if cpu_offload else device.type}"
            )
        del landmarks_by_class

        stats_groups.append(stats_by_shard)
        if total_landmarks is None:
            total_landmarks = group_total_landmarks
        elif total_landmarks != group_total_landmarks:
            raise ValueError("All feature groups must produce the same number of landmarks.")

    return stats_groups, total_landmarks, shard_class_groups


def train(args):
    if normalize_training_dataset_name(args.dataset) == "imagenet32":
        return delegate_imagenet32_training(args)

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires CUDA.")

    use_ddp = "RANK" in os.environ or "WORLD_SIZE" in os.environ
    rank, world_size, local_rank, host_sync_group = setup_runtime(use_ddp)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = MultiResDriftConfig()
    if args.steps is not None:
        cfg.total_steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.ema_decay is not None:
        cfg.ema_decay = float(args.ema_decay)
    if args.encoder is not None:
        cfg.encoder = args.encoder
    if args.temps is not None:
        cfg.temperatures = [float(t) for t in args.temps.split(",")]
    if args.pool_size is not None:
        cfg.pool_size = args.pool_size
    if args.more_features:
        cfg.more_features = True
    cfg.dataset = normalize_training_dataset_name(args.dataset)
    if args.nystrom_landmarks_per_class is not None:
        cfg.nystrom_landmarks_per_class = args.nystrom_landmarks_per_class
    if args.nystrom_total_landmarks is not None:
        cfg.nystrom_total_landmarks = args.nystrom_total_landmarks
    if args.nystrom_landmark_strategy is not None:
        cfg.nystrom_landmark_strategy = args.nystrom_landmark_strategy
    if args.nystrom_landmark_class is not None:
        cfg.nystrom_landmark_class = args.nystrom_landmark_class
    auto_enable_mnist_sharding = (
        cfg.dataset == "mnist"
        and args.nystrom_shard_by_class is None
        and args.nystrom_landmark_class is None
        and args.nystrom_total_landmarks is None
    )
    auto_enable_cifar100_sharding = (
        cfg.dataset == "cifar100"
        and args.nystrom_shard_by_class is None
        and args.nystrom_landmark_class is None
        and args.nystrom_total_landmarks is None
    )
    if args.nystrom_shard_by_class is None:
        cfg.nystrom_shard_by_class = auto_enable_mnist_sharding or auto_enable_cifar100_sharding
    else:
        cfg.nystrom_shard_by_class = bool(args.nystrom_shard_by_class)
    if args.nystrom_classes_per_shard is not None:
        cfg.nystrom_classes_per_shard = int(args.nystrom_classes_per_shard)
    elif cfg.dataset == "cifar100" and cfg.nystrom_shard_by_class:
        cfg.nystrom_classes_per_shard = 10
    if args.nystrom_shard_cpu_offload:
        cfg.nystrom_shard_cpu_offload = True
    if args.nystrom_ridge is not None:
        cfg.nystrom_ridge = args.nystrom_ridge
    if args.nystrom_kmeans_iters is not None:
        cfg.nystrom_kmeans_iters = args.nystrom_kmeans_iters
    if args.nystrom_landmark_seed is not None:
        cfg.nystrom_landmark_seed = args.nystrom_landmark_seed
    if args.nystrom_facility_candidate_ratio is not None:
        cfg.nystrom_facility_candidate_ratio = args.nystrom_facility_candidate_ratio
    if args.nystrom_facility_eval_ratio is not None:
        cfg.nystrom_facility_eval_ratio = args.nystrom_facility_eval_ratio
    if args.nystrom_density_knn_k is not None:
        cfg.nystrom_density_knn_k = args.nystrom_density_knn_k
    if args.nystrom_density_reference_size is not None:
        cfg.nystrom_density_reference_size = args.nystrom_density_reference_size
    if args.nystrom_repulsion is not None:
        cfg.nystrom_repulsion = args.nystrom_repulsion
    auto_enable_cifar100_folded_exact_attraction = (
        cfg.dataset == "cifar100"
        and args.nystrom_folded_exact_attraction is None
    )
    if args.nystrom_folded_exact_attraction is not None:
        cfg.nystrom_folded_exact_attraction = bool(args.nystrom_folded_exact_attraction)
    elif auto_enable_cifar100_folded_exact_attraction:
        cfg.nystrom_folded_exact_attraction = True
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
    if cfg.nystrom_landmark_strategy not in LANDMARK_STRATEGIES:
        raise ValueError(
            f"Unsupported --nystrom-landmark-strategy={cfg.nystrom_landmark_strategy}. "
            f"Expected one of {LANDMARK_STRATEGIES}."
        )
    if cfg.nystrom_total_landmarks is None and cfg.nystrom_landmarks_per_class <= 0:
        raise ValueError("nystrom_landmarks_per_class must be positive.")
    if cfg.nystrom_total_landmarks is not None and cfg.nystrom_total_landmarks <= 0:
        raise ValueError("nystrom_total_landmarks must be positive when set.")
    if cfg.nystrom_classes_per_shard <= 0:
        raise ValueError("nystrom_classes_per_shard must be positive.")
    if cfg.nystrom_kmeans_iters <= 0:
        raise ValueError("nystrom_kmeans_iters must be positive.")
    if cfg.nystrom_facility_candidate_ratio <= 0:
        raise ValueError("nystrom_facility_candidate_ratio must be positive.")
    if cfg.nystrom_facility_eval_ratio <= 0:
        raise ValueError("nystrom_facility_eval_ratio must be positive.")
    if cfg.nystrom_density_knn_k <= 0:
        raise ValueError("nystrom_density_knn_k must be positive.")
    if cfg.nystrom_density_reference_size <= 0:
        raise ValueError("nystrom_density_reference_size must be positive.")
    if cfg.nystrom_shard_by_class and cfg.nystrom_landmark_class is not None:
        raise ValueError(
            "--nystrom-shard-by-class cannot be combined with --nystrom-landmark-class. "
            "Use the former for all-class sharding or the latter for a single-class ablation."
        )
    if cfg.nystrom_shard_by_class and cfg.nystrom_total_landmarks is not None:
        raise ValueError("--nystrom-shard-by-class does not support --nystrom-total-landmarks.")
    if cfg.nystrom_shard_by_class and cfg.nystrom_landmark_strategy != "kmeans_per_class":
        raise ValueError(
            "--nystrom-shard-by-class currently supports only --nystrom-landmark-strategy "
            "kmeans_per_class."
        )
    if not cfg.nystrom_shard_by_class and cfg.nystrom_classes_per_shard != 1:
        raise ValueError("--nystrom-classes-per-shard requires --nystrom-shard-by-class.")
    if cfg.nystrom_shard_by_class and cfg.nystrom_repulsion != "exact":
        raise ValueError("Class-sharded Nyström currently requires --nystrom-repulsion exact.")
    if cfg.nystrom_folded_exact_attraction and cfg.nystrom_repulsion != "exact":
        raise ValueError(
            "--nystrom-folded-exact-attraction currently requires --nystrom-repulsion exact."
        )
    if cfg.nystrom_folded_exact_attraction and cfg.nystrom_shard_cpu_offload:
        raise ValueError(
            "--nystrom-folded-exact-attraction is incompatible with --nystrom-shard-cpu-offload."
        )
    if cfg.nystrom_shard_cpu_offload and not cfg.nystrom_shard_by_class:
        raise ValueError("--nystrom-shard-cpu-offload requires --nystrom-shard-by-class.")

    encoder_input_size = args.encoder_size or 112
    data_root = args.data_root
    _, dataset_spec = get_dataset_spec(cfg.dataset)
    dataset_label = dataset_spec["pretty_name"]

    out_dir = args.output_dir
    nystrom_cache_path = Path(out_dir) / "nystrom_summary_cache.pt"
    landmark_metrics_path = Path(out_dir) / "nystrom_landmark_metrics.json"
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    if use_ddp:
        sync_workers(host_sync_group)

    unet_cfg = UNetLargeConfig() if args.large else UNetConfig()
    unet_cfg = configure_unet_cfg_for_dataset(unet_cfg, cfg.dataset)
    model = UNet(
        in_ch=unet_cfg.in_ch,
        out_ch=unet_cfg.out_ch,
        base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult,
        num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions,
        dropout=unet_cfg.dropout,
        num_heads=unet_cfg.num_heads,
        image_size=unet_cfg.image_size,
    ).to(memory_format=torch.channels_last).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        launch_mode = "DDP" if use_ddp else "single-process"
        print(f"Launch mode: {launch_mode}")
        if use_ddp:
            print(f"DDP: {world_size} GPUs")
        print(f"UNet parameters: {n_params:,}")

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
    feat_encoder = torch.compile(feat_encoder)

    n_feat_params = sum(p.numel() for p in feat_encoder.parameters())
    if is_main:
        print(f"Feature encoder: {cfg.encoder}, {n_feat_params:,} params (frozen, compiled)")
        print(f"  Input size: {encoder_input_size}x{encoder_input_size}")
        write_training_config_snapshot(
            out_dir=out_dir,
            args=args,
            cfg=cfg,
            unet_cfg=unet_cfg,
            encoder_input_size=encoder_input_size,
            model_num_params=n_params,
            feat_encoder_num_params=n_feat_params,
        )

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

    full_dataset = build_cifar_dataset(
        dataset_name=cfg.dataset,
        root=data_root,
        train=True,
        download=False,
        transform=build_dataset_transform(cfg.dataset),
    )
    dataset_classes = get_dataset_class_names(cfg.dataset, full_dataset)
    if cfg.nystrom_landmark_class is not None:
        if not (0 <= int(cfg.nystrom_landmark_class) < len(dataset_classes)):
            raise ValueError(
                f"--nystrom-landmark-class must be in [0, {len(dataset_classes) - 1}] "
                f"for {dataset_label}. Got {cfg.nystrom_landmark_class}."
            )
    labels = get_dataset_targets(full_dataset)
    stats_device = torch.device("cpu") if cfg.nystrom_shard_cpu_offload else device

    if is_main:
        precomp_feats, _ = precompute_features(
            feat_encoder,
            full_dataset,
            device,
            batch_size=512,
            verbose=True,
        )

        shard_class_groups = None
        selection_metrics = None
        if cfg.nystrom_shard_by_class:
            stats_groups, total_landmarks, shard_class_groups = build_nystrom_stats_groups_sharded(
                feature_groups=precomp_feats,
                labels=labels,
                temps=tuple(cfg.temperatures),
                landmarks_per_class=cfg.nystrom_landmarks_per_class,
                classes_per_shard=cfg.nystrom_classes_per_shard,
                ridge=cfg.nystrom_ridge,
                landmark_seed=cfg.nystrom_landmark_seed,
                device=device,
                batch_size=512,
                kmeans_iters=cfg.nystrom_kmeans_iters,
                cpu_offload=cfg.nystrom_shard_cpu_offload,
            )
            landmark_class_desc = "sharded-by-class"
        else:
            stats_groups, total_landmarks, selection_metrics = build_nystrom_stats_groups(
                feature_groups=precomp_feats,
                labels=labels,
                temps=tuple(cfg.temperatures),
                landmarks_per_class=cfg.nystrom_landmarks_per_class,
                total_landmark_budget=cfg.nystrom_total_landmarks,
                landmark_strategy=cfg.nystrom_landmark_strategy,
                landmark_class=cfg.nystrom_landmark_class,
                ridge=cfg.nystrom_ridge,
                landmark_seed=cfg.nystrom_landmark_seed,
                device=device,
                batch_size=512,
                kmeans_iters=cfg.nystrom_kmeans_iters,
                facility_candidate_ratio=cfg.nystrom_facility_candidate_ratio,
                facility_eval_ratio=cfg.nystrom_facility_eval_ratio,
                density_knn_k=cfg.nystrom_density_knn_k,
                density_reference_size=cfg.nystrom_density_reference_size,
                class_names=dataset_classes,
            )
            landmark_class_desc = "all"
            if cfg.nystrom_landmark_class is not None:
                class_id = int(cfg.nystrom_landmark_class)
                landmark_class_desc = f"{class_id} ({dataset_classes[class_id]})"

        save_nystrom_cache(
            cache_path=nystrom_cache_path,
            stats_groups=stats_groups,
            total_landmarks=total_landmarks,
            shard_class_groups=shard_class_groups,
        )
        if selection_metrics is not None:
            metrics_payload = build_selection_metrics_report(
                selection_metrics=selection_metrics,
                cfg=cfg,
                total_landmarks=total_landmarks,
            )
            write_json(landmark_metrics_path, metrics_payload)
        del precomp_feats

        print("Pre-computation complete. Nyström caches built.")
        print(
            "  Nyström: "
            f"strategy={cfg.nystrom_landmark_strategy} "
            f"landmarks_per_class={cfg.nystrom_landmarks_per_class} "
            f"total_landmark_budget={cfg.nystrom_total_landmarks} "
            f"landmark_class={landmark_class_desc} "
            f"shard_by_class={cfg.nystrom_shard_by_class} "
            f"classes_per_shard={cfg.nystrom_classes_per_shard if cfg.nystrom_shard_by_class else '<n/a>'} "
            f"shard_cpu_offload={cfg.nystrom_shard_cpu_offload} "
            f"total_landmarks={total_landmarks} "
            f"ridge={cfg.nystrom_ridge} "
            f"seed={cfg.nystrom_landmark_seed} "
            f"kmeans_iters={cfg.nystrom_kmeans_iters} "
            f"repulsion={cfg.nystrom_repulsion} "
            "attractive_cache="
            f"{'alternative_sqrtD_folded_exact' if cfg.nystrom_folded_exact_attraction else 'alternative_sqrtD'}"
        )
        if selection_metrics is not None:
            aggregate_metrics = build_selection_metrics_report(
                selection_metrics=selection_metrics,
                cfg=cfg,
                total_landmarks=total_landmarks,
            )["aggregate"]
            print(f"  landmark_metrics_path={landmark_metrics_path}")
            if aggregate_metrics:
                print(
                    "  aggregate_landmark_metrics: "
                    f"mean_min_dist={aggregate_metrics.get('mean_min_dist', float('nan')):.4f} "
                    f"p95_min_dist={aggregate_metrics.get('p95_min_dist', float('nan')):.4f} "
                    f"mean_max_flat_kernel={aggregate_metrics.get('mean_max_flat_kernel', float('nan')):.4f} "
                    f"w_condition_mean={aggregate_metrics.get('w_condition_mean', float('nan')):.4f}"
                )
        if shard_class_groups is not None:
            shard_class_names = [
                [dataset_classes[class_id] for class_id in shard_group]
                for shard_group in shard_class_groups
            ]
            print(f"  shard_count={len(shard_class_groups)}")
            print(f"  shard_class_groups={shard_class_groups}")
            print(f"  shard_class_names={shard_class_names}")

    del full_dataset, labels
    if use_ddp:
        sync_workers(host_sync_group)
    if not is_main:
        stats_groups, total_landmarks, shard_class_groups = load_nystrom_cache(
            cache_path=nystrom_cache_path,
            device=stats_device,
        )
    elif cfg.nystrom_shard_cpu_offload:
        stats_groups, total_landmarks, shard_class_groups = load_nystrom_cache(
            cache_path=nystrom_cache_path,
            device=stats_device,
        )

    ema = EMA(train_model, decay=cfg.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        fused=True,
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

    log_path = os.path.join(out_dir, "loss_log.csv")
    fid_log_path = os.path.join(out_dir, "fid_log.csv")
    if is_main:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "time_s", "images_per_sec", "fid_raw", "fid_ema"])
        with open(fid_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "fid_raw", "fid_ema", "fid"])

    fid_ref_dir = None
    if is_main:
        fid_ref_dir = prepare_cifar_reference(
            dataset_name=cfg.dataset,
            output_dir=default_ref_dir(cfg.dataset),
            data_root=data_root,
        )
    if use_ddp:
        sync_workers(host_sync_group)

    if is_main:
        global_batch_size = cfg.batch_size * world_size
        print(f"Training Drifting (multi-res {cfg.encoder}) for {cfg.total_steps} steps")
        print(f"  dataset={dataset_label}")
        print(f"  per-GPU batch={cfg.batch_size}, global batch={global_batch_size}")
        print(f"  temps={cfg.temperatures}, pool_size={cfg.pool_size}")
        print(f"  more_features={cfg.more_features}")
        print(
            "  landmark_strategy="
            f"{cfg.nystrom_landmark_strategy}, total_landmark_budget={cfg.nystrom_total_landmarks}, "
            f"landmarks_per_class={cfg.nystrom_landmarks_per_class}"
        )
        if cfg.nystrom_shard_by_class:
            print(f"  classes_per_shard={cfg.nystrom_classes_per_shard}")
        print(f"  folded_exact_attraction={cfg.nystrom_folded_exact_attraction}")
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
        if fixed_sample_noise is not None:
            print(f"  fixed_sample_noise_path={args.fixed_sample_noise_path}")
            print(f"  fixed_sample_count={fixed_sample_noise.shape[0]}")
            print(f"  sample_grid_cols={sample_grid_cols}")
        if auto_enable_mnist_sharding:
            print("  note=auto-enabled --nystrom-shard-by-class for MNIST")
        elif auto_enable_cifar100_sharding:
            print("  note=auto-enabled --nystrom-shard-by-class for CIFAR-100")
            print(f"  note=auto-set --nystrom-classes-per-shard {cfg.nystrom_classes_per_shard} for CIFAR-100")
        elif cfg.dataset == "mnist" and not cfg.nystrom_shard_by_class:
            print(
                "  warning=MNIST Nyström without class sharding is known to collapse. "
                "Pass --nystrom-shard-by-class unless you are intentionally running the bad ablation."
            )
        if auto_enable_cifar100_folded_exact_attraction:
            print("  note=auto-enabled --nystrom-folded-exact-attraction for CIFAR-100")

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
    last_step = 0
    loss_log_slot_count = None

    for step in range(1, cfg.total_steps + 1):
        last_step = step
        batch_size = cfg.batch_size
        z = sample_model_noise(model, batch_size, device, memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_images = model(z)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_groups_raw = feat_encoder(ensure_three_channels(gen_images.float()))
        gen_groups = [(feat.float(), c_j) for feat, c_j in gen_groups_raw]
        if loss_log_slot_count is None:
            loss_log_slot_count = count_feature_slots(gen_groups)
            if is_main:
                print(
                    f"  loss logging normalized by {loss_log_slot_count} feature slots "
                    f"(sum of encoder locations across groups)"
                )

        if cfg.nystrom_shard_by_class:
            loss_fn = (
                drifting_loss_multires_nystrom_sharded_folded_exact
                if cfg.nystrom_folded_exact_attraction
                else drifting_loss_multires_nystrom_sharded
            )
            loss = loss_fn(
                gen_groups,
                stats_groups,
                temps=tuple(cfg.temperatures),
                repulsion_mode=cfg.nystrom_repulsion,
                distributed_queries=cfg.nystrom_global_query_gather,
            )
        else:
            loss_fn = (
                drifting_loss_multires_nystrom_folded_exact
                if cfg.nystrom_folded_exact_attraction
                else drifting_loss_multires_nystrom
            )
            loss = loss_fn(
                gen_groups,
                stats_groups,
                temps=tuple(cfg.temperatures),
                repulsion_mode=cfg.nystrom_repulsion,
                distributed_queries=cfg.nystrom_global_query_gather,
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
                train_model.train()
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
        if last_step >= cfg.total_steps:
            print(f"\nTraining complete. {last_step} steps in {elapsed / 3600:.1f} hours")
        else:
            print(f"\nTraining stopped at step {last_step} in {elapsed / 3600:.1f} hours")
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
                f"  FID {format_fid_summary(final_fids)} "
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
    parser.add_argument("--output-dir", type=str, default="outputs/alternative_nystrom")
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov3",
        choices=["dinov2-multires", "convnextv2", "mocov2", "dinov3", "eva02", "siglip2", "clip"],
    )
    parser.add_argument(
        "--encoder-size",
        type=int,
        default=112,
        help="Encoder input resolution (default 112, was 224)",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size per GPU when launched with torchrun",
    )
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=TRAIN_SUPPORTED_DATASETS,
        help="Training dataset to use for Nyström cache construction and FID references.",
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
        "--feature-batch-size",
        type=int,
        default=IMAGENET32_DEFAULT_FEATURE_BATCH_SIZE,
        help="Feature encoding batch size for ImageNet32 Nyström summary construction.",
    )
    parser.add_argument("--nystrom-landmarks-per-class", type=int, default=None)
    parser.add_argument(
        "--nystrom-num-landmarks",
        type=int,
        default=None,
        help="ImageNet32 Nyström landmark count. Defaults to subset classes times landmarks per class.",
    )
    parser.add_argument(
        "--nystrom-subset-num-classes",
        type=int,
        default=IMAGENET32_DEFAULT_SUBSET_NUM_CLASSES,
        help="How many ImageNet32 classes to sample when building the Nyström summary.",
    )
    parser.add_argument(
        "--nystrom-total-landmarks",
        type=int,
        default=None,
        help=(
            "Optional total Nyström landmark budget across the full dataset. "
            "When set, per-class strategies allocate this budget evenly across classes."
        ),
    )
    parser.add_argument(
        "--nystrom-landmark-strategy",
        type=str,
        default=None,
        choices=LANDMARK_STRATEGIES,
        help="Landmark selection strategy to use for Nyström cache construction.",
    )
    parser.add_argument(
        "--nystrom-landmark-class",
        type=int,
        default=None,
        help="Optional dataset class id to restrict Nyström landmarks to. Default uses all classes.",
    )
    parser.add_argument(
        "--nystrom-shard-by-class",
        dest="nystrom_shard_by_class",
        action="store_true",
        help=(
            "Build class-based Nyström attractive cache shards and combine shard numerators/denominators. "
            "Use --nystrom-classes-per-shard to pack multiple classes into each shard."
        ),
    )
    parser.add_argument(
        "--no-nystrom-shard-by-class",
        dest="nystrom_shard_by_class",
        action="store_false",
        help="Disable class-sharded Nyström attractive caches, even for datasets that auto-enable them.",
    )
    parser.set_defaults(nystrom_shard_by_class=None)
    parser.add_argument(
        "--nystrom-shard-cpu-offload",
        action="store_true",
        help="Store class-sharded Nyström caches on CPU after precompute and stream them back to GPU during training.",
    )
    parser.add_argument(
        "--restrict-training-to-selected-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For ImageNet32, restrict summary construction to the selected class subset.",
    )
    parser.add_argument("--nystrom-ridge", type=float, default=None)
    parser.add_argument("--nystrom-kmeans-iters", type=int, default=None)
    parser.add_argument("--nystrom-landmark-seed", type=int, default=None)
    parser.add_argument(
        "--nystrom-classes-per-shard",
        type=int,
        default=None,
        help=(
            "For class-sharded Nyström, pack this many classes into each shard. "
            "Each class still contributes nystrom_landmarks_per_class landmarks."
        ),
    )
    parser.add_argument("--nystrom-facility-candidate-ratio", type=int, default=None)
    parser.add_argument("--nystrom-facility-eval-ratio", type=int, default=None)
    parser.add_argument("--nystrom-density-knn-k", type=int, default=None)
    parser.add_argument("--nystrom-density-reference-size", type=int, default=None)
    parser.add_argument("--nystrom-repulsion", type=str, choices=["nystrom", "exact"], default=None)
    parser.add_argument(
        "--nystrom-folded-exact-attraction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the exact folded-summary attractive path for Nyström training. "
            "This preserves the exact sharded/unsharded exact objective while avoiding explicit attractive phi materialization."
        ),
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
        "--summary-cache-path",
        type=str,
        default=None,
        help="Optional ImageNet32 Nyström summary cache path.",
    )
    parser.add_argument(
        "--rebuild-summary-cache",
        action="store_true",
        help="Force rebuilding the ImageNet32 Nyström summary cache.",
    )
    parser.add_argument(
        "--precompute-log-every",
        type=int,
        default=IMAGENET32_DEFAULT_PRECOMPUTE_LOG_EVERY,
        help="ImageNet32 summary-construction progress interval in samples.",
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
        help="Torch device override for single-process ImageNet32 runs.",
    )
    parser.add_argument(
        "--no-compile-encoder",
        action="store_true",
        help="Disable encoder compilation for ImageNet32 runs.",
    )
    parser.add_argument("--large", action="store_true")
    train(parser.parse_args())
