import math
import torch
import torch.nn.functional as F


def normalize_features(gen, pos, D):
    """Normalize 2D features with one shared scale.

    Args:
        gen: [N, D] generated features
        pos: [N_pos, D] positive (real) features
        D: feature dimensionality

    Returns:
        gen_n: [N, D] normalized generated features
        pos_n: [N_pos, D] normalized positive features
        S: scalar normalization scale (detached)
    """
    y_all = torch.cat([pos, gen], dim=0)
    dists = torch.cdist(gen.float(), y_all.float(), p=2)
    mean_dist = dists.mean()
    S = (mean_dist / math.sqrt(D)).detach().clamp(min=1e-8)
    return gen / S, pos / S, S


def _compute_drift_from_normalized(gen_n, pos_n, temp):
    """Compute coupled softmax drift in normalized feature space."""
    N, D = gen_n.shape
    N_pos = pos_n.shape[0]

    targets_n = torch.cat([pos_n, gen_n], dim=0)  # [N_pos + N, D]
    gen_f = gen_n.float()
    targets_f = targets_n.float()

    dist = torch.cdist(gen_f, targets_f, p=2)  # [N, N_pos + N]

    idx = torch.arange(N, device=gen_n.device)
    dist[idx, N_pos + idx] = 1e6

    tau_tilde = temp * math.sqrt(D)
    logit = -dist / tau_tilde
    A_row = torch.softmax(logit, dim=1)   # over targets
    A_col = torch.softmax(logit, dim=0)   # over generated samples
    A = torch.sqrt(A_row * A_col + 1e-30)

    A_pos = A[:, :N_pos]       # [N, N_pos]
    A_neg = A[:, N_pos:]       # [N, N]

    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)  # [N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)  # [N, N]

    drift_pos = W_pos @ targets_f[:N_pos]  # [N, D]
    drift_neg = W_neg @ targets_f[N_pos:]  # [N, D]
    return (drift_pos - drift_neg).to(gen_n.dtype)


def compute_drift(gen, pos, temp=0.05):
    """Compute coupled softmax drift from unnormalized features.

    Returns drift in normalized feature space.
    """
    D = gen.shape[-1]
    gen_n, pos_n, _ = normalize_features(gen, pos, D)
    return _compute_drift_from_normalized(gen_n, pos_n, temp=temp)


def normalize_drift(V, D):
    """Per-dimension drift normalization (Appendix A.6).

    Scales V so that E[||V||^2 / D] ~ 1.
    """
    lambda_j = torch.sqrt((V.float().pow(2).sum(dim=-1) / D).mean()).detach()
    return V / (lambda_j + 1e-8)


def compute_normalized_drift_multitemp(gen, pos, temps=(0.02, 0.05, 0.2)):
    """Compute normalized features and aggregated normalized drift.

    Each temperature's drift is independently normalized before summing.
    """
    D = gen.shape[-1]
    gen_n, pos_n, S = normalize_features(gen, pos, D)
    V_total = torch.zeros_like(gen_n)
    for temp in temps:
        V = _compute_drift_from_normalized(gen_n, pos_n, temp=temp)
        V = normalize_drift(V, D)
        V_total += V
    return gen_n, V_total, S


def compute_drift_multitemp(gen, pos, temps=(0.02, 0.05, 0.2)):
    """Compute drift field aggregated over multiple temperatures."""
    _, V_total, _ = compute_normalized_drift_multitemp(gen, pos, temps=temps)
    return V_total


def drifting_loss(gen_feats, pos_feats, temps=(0.02, 0.05, 0.2)):
    """Compute drifting loss.

    Gradient flows through gen_feats back to the generator.
    V is computed on detached features and target is detached.

    Args:
        gen_feats: [N, D] generated features (WITH gradient to generator)
        pos_feats: [N_pos, D] positive (real) features (detached)
        temps: temperatures for multi-scale drift

    Returns:
        loss: scalar MSE between normalized gen_feats and normalized target
    """
    with torch.no_grad():
        gen_n_det, V, S = compute_normalized_drift_multitemp(
            gen_feats.detach(), pos_feats.detach(), temps=temps
        )
        target = gen_n_det + V

    gen_n = gen_feats / S
    return F.mse_loss(gen_n, target)


# ---------------------------------------------------------------------------
# Batched versions: operate over L spatial locations simultaneously
# ---------------------------------------------------------------------------


def normalize_features_batched(gen, pos, D):
    """Batched feature normalization for one feature-map group.

    All locations in the same group share one normalization scale.

    Args:
        gen: [L, N, D] generated features
        pos: [L, N_pos, D] positive features
        D: feature dimensionality

    Returns:
        gen_n: [L, N, D] normalized generated features
        pos_n: [L, N_pos, D] normalized positive features
        S: scalar normalization scale (detached)
    """
    y_all = torch.cat([pos, gen], dim=1)  # [L, N_pos + N, D]
    dists = torch.cdist(gen.float(), y_all.float(), p=2)  # [L, N, N_pos + N]
    mean_dist = dists.mean()
    S = (mean_dist / math.sqrt(D)).detach().clamp(min=1e-8)
    return gen / S, pos / S, S


def normalize_drift_batched(V, D):
    """Batched drift normalization across L spatial locations.

    Shared lambda for the feature-map group.

    Args:
        V: [L, N, D] drift vectors at L locations
        D: feature dimensionality

    Returns:
        V_norm: [L, N, D]
    """
    lambda_j = torch.sqrt((V.float().pow(2).sum(dim=-1) / D).mean()).detach()
    return V / (lambda_j + 1e-8)


def _compute_drift_from_normalized_batched(gen_n, pos_n, temp=0.05):
    """Coupled softmax drift over L spatial locations in normalized space."""
    L, N, D = gen_n.shape
    N_pos = pos_n.shape[1]

    targets_n = torch.cat([pos_n, gen_n], dim=1)  # [L, N_pos + N, D]

    gen_f = gen_n.float()
    targets_f = targets_n.float()

    # Batched distances: [L, N, N_pos + N]
    dist = torch.cdist(gen_f, targets_f, p=2)

    # Self-exclusion: mask diagonal of gen-to-neg block
    mask = torch.eye(N, device=gen_n.device, dtype=dist.dtype).unsqueeze(0) * 1e6  # [1, N, N]
    # Pad to match target dim: zeros for pos columns, mask for neg columns
    pad = torch.zeros(1, N, N_pos, device=gen_n.device, dtype=dist.dtype)
    full_mask = torch.cat([pad, mask], dim=2)  # [1, N, N_pos + N]
    dist = dist + full_mask

    tau_tilde = temp * math.sqrt(D)
    logit = -dist / tau_tilde
    A_row = torch.softmax(logit, dim=2)   # over targets
    A_col = torch.softmax(logit, dim=1)   # over gen samples
    A = torch.sqrt(A_row * A_col + 1e-30)

    A_pos = A[:, :, :N_pos]   # [L, N, N_pos]
    A_neg = A[:, :, N_pos:]   # [L, N, N]

    W_pos = A_pos * A_neg.sum(dim=2, keepdim=True)  # [L, N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=2, keepdim=True)  # [L, N, N]

    drift_pos = torch.bmm(W_pos, targets_f[:, :N_pos])  # [L, N, D]
    drift_neg = torch.bmm(W_neg, targets_f[:, N_pos:])   # [L, N, D]

    return (drift_pos - drift_neg).to(gen_n.dtype)


def compute_drift_batched(gen, pos, temp=0.05):
    """Batched coupled softmax drift from unnormalized features."""
    D = gen.shape[-1]
    gen_n, pos_n, _ = normalize_features_batched(gen, pos, D)
    return _compute_drift_from_normalized_batched(gen_n, pos_n, temp=temp)


def compute_normalized_drift_multitemp_batched(gen, pos, temps=(0.02, 0.05, 0.2)):
    """Batched normalized feature + multi-temperature normalized drift.

    Each temperature's drift is independently normalized before summing.

    Args:
        gen: [L, N, D] generated features (detached)
        pos: [L, N_pos, D] positive features (detached)
        temps: tuple of temperatures

    Returns:
        gen_n: [L, N, D] normalized generated features
        V_total: [L, N, D] sum_tau V_tilde_tau
        S: scalar normalization scale
    """
    D = gen.shape[-1]
    gen_n, pos_n, S = normalize_features_batched(gen, pos, D)
    V_total = torch.zeros_like(gen_n)
    for temp in temps:
        V = _compute_drift_from_normalized_batched(gen_n, pos_n, temp=temp)
        V = normalize_drift_batched(V, D)
        V_total = V_total + V
    return gen_n, V_total, S


def compute_drift_multitemp_batched(gen, pos, temps=(0.02, 0.05, 0.2)):
    """Batched multi-temperature drift field."""
    _, V_total, _ = compute_normalized_drift_multitemp_batched(gen, pos, temps=temps)
    return V_total


def drifting_loss_multires(gen_groups, pos_groups, temps=(0.02, 0.05, 0.2)):
    """Compute multi-resolution drifting loss across feature groups.

    Each group is a (features[B, L, C], C_j) tuple. Groups are processed
    independently with batched drift computation. Losses are summed across
    feature slots as in the strong baseline.

    Gradient flows through gen features back to generator.

    Args:
        gen_groups: list of (features[B, L, C], C_j) from encoder on generated images
        pos_groups: list of (features[B, L, C], C_j) from encoder on real images
        temps: temperatures for multi-temperature drift

    Returns:
        loss: scalar, sum over feature terms
    """
    total_loss = 0.0

    for (gen_feat, C_j), (pos_feat, _) in zip(gen_groups, pos_groups):
        # gen_feat: [B, L, C_j] -- has gradient
        # pos_feat: [B, L, C_j] -- detached

        with torch.no_grad():
            # Transpose to [L, B, C_j] for batched computation
            gen_t = gen_feat.detach().transpose(0, 1).contiguous()
            pos_t = pos_feat.detach().transpose(0, 1).contiguous()

            # Compute normalized features and multi-temp drift: [L, B, C_j]
            gen_t_n, V, S = compute_normalized_drift_multitemp_batched(gen_t, pos_t, temps=temps)

            # Transpose back: [B, L, C_j]
            target = (gen_t_n + V).transpose(0, 1)

        gen_n = gen_feat / S
        loss_group = (gen_n - target).pow(2).mean(dim=(0, 2)).sum()
        total_loss = total_loss + loss_group

    return total_loss
