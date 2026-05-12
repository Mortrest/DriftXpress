from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class NystromStats:
    """Cached Nyström summaries for one feature group and one temperature."""

    landmarks: torch.Tensor
    A: torch.Tensor
    global_totals: torch.Tensor
    global_weighted_points: torch.Tensor
    temperature: float


def move_nystrom_stats_to_device(stats: NystromStats, device: torch.device | str) -> NystromStats:
    """Move a Nyström cache block to the requested device if needed."""
    device = torch.device(device)
    if (
        stats.landmarks.device == device
        and stats.A.device == device
        and stats.global_totals.device == device
        and stats.global_weighted_points.device == device
    ):
        return stats

    return NystromStats(
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


def normalize_drift_batched(V, D=None, eps=1e-8):
    """Normalize drift per location so E[||V||^2 / D] ~ 1."""
    if D is None:
        D = V.shape[-1]
    lambda_j = torch.sqrt((V.float().pow(2).sum(dim=-1) / D).mean(dim=-1)).detach()
    return V / (lambda_j.unsqueeze(-1).unsqueeze(-1) + eps)


def _laplacian_kernel_batched(x, y, temperature, eps=1e-8):
    """Batched Laplacian kernel."""
    D = x.shape[-1]
    tau_tilde = max(float(temperature) * float(D), eps)
    dist = torch.cdist(x.float(), y.float(), p=2)
    return torch.exp(-dist / tau_tilde)


def _inverse_sqrt_psd_batched(mats, eps=1e-8):
    """Compute batched inverse square root for symmetric PSD matrices."""
    eigvals, eigvecs = torch.linalg.eigh(mats.float())
    inv_sqrt_eigs = eigvals.clamp_min(eps).rsqrt()
    return (eigvecs * inv_sqrt_eigs.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


def _compute_exact_repulsive_barycenter_batched(x, temperature=0.05, eps=1e-8, mask_self=True):
    """Compute an exact kernel repulsive barycenter over the current batch."""
    x_f = x.float()
    D = x.shape[-1]
    tau_tilde = max(float(temperature) * float(D), eps)
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
    W = _laplacian_kernel_batched(landmarks_t, landmarks_t, temperature=temperature, eps=eps)
    eye = torch.eye(M, device=device, dtype=W.dtype).unsqueeze(0).expand(L, -1, -1)
    A = _inverse_sqrt_psd_batched(W + ridge * eye, eps=eps)
    return landmarks_t, A


def compute_nystrom_features_batched(x, landmarks, A, temperature=0.05, eps=1e-8):
    """Explicit Nyström features phi(x) = K(x, U) (W + ridge I)^(-1/2)."""
    K_xu = _laplacian_kernel_batched(x, landmarks, temperature=temperature, eps=eps)
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
def precompute_nystrom_statistics_batched(
    sensitive_points,
    public_landmarks,
    temperature=0.05,
    ridge=1e-4,
    eps=1e-8,
    batch_size=512,
    device=None,
):
    """Precompute cached Nyström summaries over private points."""
    if sensitive_points.ndim != 3:
        raise ValueError("Expected sensitive_points with shape [Np, L, D].")
    if public_landmarks.ndim != 3:
        raise ValueError("Expected public_landmarks with shape [M, L, D].")

    if device is None:
        device = sensitive_points.device

    landmarks, A = prepare_nystrom_landmarks_batched(
        public_landmarks=public_landmarks,
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

    return NystromStats(
        landmarks=landmarks,
        A=A,
        global_totals=global_totals,
        global_weighted_points=global_weighted_points,
        temperature=float(temperature),
    )


@torch.no_grad()
def precompute_nystrom_statistics_multitemp_batched(
    sensitive_points,
    public_landmarks,
    temps=(0.02, 0.05, 0.2),
    ridge=1e-4,
    eps=1e-8,
    batch_size=512,
    device=None,
):
    """Precompute private summaries for multiple temperatures."""
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
        )
    return stats


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
        b_neg = _compute_exact_repulsive_barycenter_batched(
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
    local_batch = int(x.shape[1])
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
        V_total = V_total + normalize_drift_batched(V_tau, eps=eps)

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
        b_neg = _compute_exact_repulsive_barycenter_batched(
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

        V_total = V_total + normalize_drift_batched(V_tau, eps=eps)

    return V_total


def drifting_loss_multires_nystrom(
    gen_groups,
    stats_groups,
    temps=(0.02, 0.05, 0.2),
    eps=1e-8,
    max_drift_norm=None,
    repulsion_mode="nystrom",
    distributed_queries=False,
):
    """Multi-resolution drifting loss using Nyström drift."""
    if len(gen_groups) != len(stats_groups):
        raise ValueError("gen_groups and stats_groups must have the same length.")

    total_loss = gen_groups[0][0].new_tensor(0.0)

    for (gen_feat, _), stats_by_temp in zip(gen_groups, stats_groups):
        with torch.no_grad():
            gen_t = gen_feat.detach().transpose(0, 1).contiguous()
            V = compute_nystrom_drift_multitemp_batched(
                x=gen_t,
                stats_by_temp=stats_by_temp,
                temps=temps,
                eps=eps,
                max_drift_norm=max_drift_norm,
                repulsion_mode=repulsion_mode,
                distributed_queries=distributed_queries,
            )
            target = gen_feat.detach() + V.transpose(0, 1)

        total_loss = total_loss + F.mse_loss(gen_feat, target)

    return total_loss / len(gen_groups)


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
    """Multi-resolution drifting loss using class-sharded Nyström drift."""
    if len(gen_groups) != len(stats_groups):
        raise ValueError("gen_groups and stats_groups must have the same length.")

    total_loss = gen_groups[0][0].new_tensor(0.0)
    effective_distributed_queries = distributed_queries or distributed_shards

    for (gen_feat, _), stats_by_shard in zip(gen_groups, stats_groups):
        with torch.no_grad():
            gen_t = gen_feat.detach().transpose(0, 1).contiguous()
            V = compute_nystrom_drift_multitemp_sharded_batched(
                x=gen_t,
                stats_by_shard=stats_by_shard,
                temps=temps,
                eps=eps,
                max_drift_norm=max_drift_norm,
                repulsion_mode=repulsion_mode,
                distributed_shards=distributed_shards,
                distributed_queries=effective_distributed_queries,
            )
            target = gen_feat.detach() + V.transpose(0, 1)

        total_loss = total_loss + F.mse_loss(gen_feat, target)

    return total_loss / len(gen_groups)
