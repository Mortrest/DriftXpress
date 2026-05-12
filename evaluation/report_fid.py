from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from tqdm.auto import tqdm

from evaluation.sample import drift_sample, ensure_three_channels
from models.unet import UNet


SUPPORTED_EVAL_WEIGHTS = ("raw", "ema")
FID_FEATURE_DIMS = 2048


@dataclass(frozen=True)
class CachedStats:
    mu: np.ndarray
    sigma: np.ndarray
    num_items: int
    metadata: dict[str, Any]
    path: Path


class RunningFeatureStats:
    """Streaming mean/covariance estimator for Inception features."""

    def __init__(self, dims: int) -> None:
        self.dims = int(dims)
        self.count = 0
        self.sum = np.zeros(self.dims, dtype=np.float64)
        self.sum_outer = np.zeros((self.dims, self.dims), dtype=np.float64)

    def update(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.dims:
            raise ValueError(
                f"Expected features shaped [N, {self.dims}], got {tuple(features.shape)}."
            )
        feats = features.detach().cpu().numpy().astype(np.float64, copy=False)
        self.count += int(feats.shape[0])
        self.sum += feats.sum(axis=0)
        self.sum_outer += feats.T @ feats

    def finalize(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.count < 2:
            raise ValueError("Need at least two samples to compute FID statistics.")
        mu = self.sum / self.count
        sigma = (self.sum_outer - self.count * np.outer(mu, mu)) / (self.count - 1)
        return mu, sigma, self.count


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device(device_arg: str | None) -> torch.device:
    return torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))


def normalize_eval_weights(eval_weights: str) -> str:
    normalized = str(eval_weights).strip().lower()
    if normalized not in SUPPORTED_EVAL_WEIGHTS:
        raise ValueError(
            f"Unsupported eval weights '{eval_weights}'. Expected one of {SUPPORTED_EVAL_WEIGHTS}."
        )
    return normalized


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _strip_prefix_if_present(
    state_dict: dict[str, torch.Tensor],
    prefix: str = "_orig_mod.",
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    return torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)


def resolve_checkpoint_paths(checkpoint_arg: str | Path) -> tuple[list[Path], bool]:
    checkpoint_path = Path(checkpoint_arg).resolve()
    if checkpoint_path.is_dir():
        checkpoints = sorted(
            path for path in checkpoint_path.glob("*.pt") if path.name != "drift_latest.pt"
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint .pt files found in {checkpoint_path} after excluding drift_latest.pt."
            )
        return checkpoints, True

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    return [checkpoint_path], False


def infer_dataset_name_from_checkpoint(checkpoint: dict[str, Any]) -> str | None:
    config = checkpoint.get("config")
    drift_cfg = _cfg_get(config, "drift")
    dataset_name = _cfg_get(drift_cfg, "dataset")
    if dataset_name is None:
        return None
    return str(dataset_name).strip().lower()


def load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
    eval_weights: str,
) -> torch.nn.Module:
    eval_weights = normalize_eval_weights(eval_weights)
    unet_cfg = _cfg_get(checkpoint.get("config"), "unet")
    if unet_cfg is None:
        raise KeyError("Checkpoint does not contain config.unet.")

    model = UNet(
        in_ch=int(_cfg_get(unet_cfg, "in_ch", 3)),
        out_ch=int(_cfg_get(unet_cfg, "out_ch", 3)),
        base_ch=int(_cfg_get(unet_cfg, "base_ch")),
        ch_mult=tuple(_cfg_get(unet_cfg, "ch_mult")),
        num_res_blocks=int(_cfg_get(unet_cfg, "num_res_blocks")),
        attn_resolutions=tuple(_cfg_get(unet_cfg, "attn_resolutions")),
        dropout=float(_cfg_get(unet_cfg, "dropout", 0.0)),
        num_heads=int(_cfg_get(unet_cfg, "num_heads", 4)),
        num_classes=int(_cfg_get(unet_cfg, "num_classes", 0)),
        image_size=int(_cfg_get(unet_cfg, "image_size", 32)),
    ).to(device)

    state_key = "model" if eval_weights == "raw" else "ema"
    if state_key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain {eval_weights} weights.")

    state_dict = _strip_prefix_if_present(checkpoint[state_key])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_inception(device: torch.device, dims: int = FID_FEATURE_DIMS) -> InceptionV3:
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model


def _extract_inception_features(
    inception: InceptionV3,
    images: torch.Tensor,
) -> torch.Tensor:
    pred = inception(images)[0]
    if pred.shape[2] != 1 or pred.shape[3] != 1:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
    return pred.squeeze(3).squeeze(2)


def compute_stats_for_image_batches(
    batch_iterable: Iterable[torch.Tensor],
    *,
    inception: InceptionV3,
    device: torch.device,
    desc: str,
    total_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    stats = RunningFeatureStats(FID_FEATURE_DIMS)
    with torch.no_grad():
        with tqdm(total=total_images, desc=desc, unit="img") as progress:
            for images in batch_iterable:
                images = ensure_three_channels(images)
                images = images.to(device, non_blocking=True)
                features = _extract_inception_features(inception, images)
                stats.update(features)
                progress.update(int(images.shape[0]))
    return stats.finalize()


def compute_stats_for_generated_samples(
    *,
    model: torch.nn.Module,
    num_images: int,
    batch_size: int,
    inception: InceptionV3,
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    stats = RunningFeatureStats(FID_FEATURE_DIMS)
    generated = 0
    with torch.no_grad():
        with tqdm(total=num_images, desc=desc, unit="img") as progress:
            while generated < num_images:
                current_batch = min(batch_size, num_images - generated)
                samples = drift_sample(model, current_batch, device)
                samples = samples.add(1.0).mul_(0.5).clamp_(0.0, 1.0)
                samples = ensure_three_channels(samples)
                features = _extract_inception_features(inception, samples)
                stats.update(features)
                generated += current_batch
                progress.update(current_batch)
    return stats.finalize()


def load_cached_stats(stats_path: str | Path) -> CachedStats:
    stats_path = Path(stats_path)
    with np.load(stats_path, allow_pickle=False) as payload:
        if "mu" not in payload or "sigma" not in payload:
            raise ValueError(f"Cached stats at {stats_path} are missing 'mu'/'sigma' arrays.")
        mu = np.asarray(payload["mu"], dtype=np.float64)
        sigma = np.asarray(payload["sigma"], dtype=np.float64)
        if mu.shape != (FID_FEATURE_DIMS,):
            raise ValueError(f"Expected mu shape ({FID_FEATURE_DIMS},), got {mu.shape}.")
        if sigma.shape != (FID_FEATURE_DIMS, FID_FEATURE_DIMS):
            raise ValueError(
                f"Expected sigma shape ({FID_FEATURE_DIMS}, {FID_FEATURE_DIMS}), got {sigma.shape}."
            )
        raw_num_items = payload["num_items"] if "num_items" in payload else None
        num_items = int(raw_num_items) if raw_num_items is not None else -1
        raw_metadata = payload["metadata"] if "metadata" in payload else None
        metadata: dict[str, Any] = {}
        if raw_metadata is not None:
            metadata = json.loads(str(raw_metadata))
    return CachedStats(mu=mu, sigma=sigma, num_items=num_items, metadata=metadata, path=stats_path)


def save_cached_stats(
    stats_path: str | Path,
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    num_items: int,
    metadata: dict[str, Any],
) -> Path:
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_path,
        mu=np.asarray(mu, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        num_items=np.int64(num_items),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    return stats_path


def metadata_matches(
    cached_metadata: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for key, expected_value in expected_metadata.items():
        if cached_metadata.get(key) != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value!r}, found {cached_metadata.get(key)!r}"
            )
    return not mismatches, mismatches


def get_or_compute_real_stats(
    *,
    stats_path: str | Path,
    expected_metadata: dict[str, Any],
    inception: InceptionV3,
    device: torch.device,
    real_batch_iterator_factory: Callable[[], Iterable[torch.Tensor]],
    total_images: int | None,
    force_recompute: bool,
    desc: str,
) -> CachedStats:
    stats_path = Path(stats_path)
    if stats_path.is_file() and not force_recompute:
        cached = load_cached_stats(stats_path)
        matches, mismatches = metadata_matches(cached.metadata, expected_metadata)
        if matches:
            print(f"Using cached real-image FID stats: {stats_path}")
            return cached
        print(f"Cached stats metadata mismatch at {stats_path}; recomputing.")
        for mismatch in mismatches:
            print(f"  - {mismatch}")

    mu, sigma, num_items = compute_stats_for_image_batches(
        real_batch_iterator_factory(),
        inception=inception,
        device=device,
        desc=desc,
        total_images=total_images,
    )
    save_cached_stats(
        stats_path,
        mu=mu,
        sigma=sigma,
        num_items=num_items,
        metadata=expected_metadata,
    )
    print(f"Saved real-image FID stats to {stats_path}")
    return CachedStats(
        mu=mu,
        sigma=sigma,
        num_items=num_items,
        metadata=dict(expected_metadata),
        path=stats_path,
    )


def compute_fid_from_stats(
    real_mu: np.ndarray,
    real_sigma: np.ndarray,
    fake_mu: np.ndarray,
    fake_sigma: np.ndarray,
) -> float:
    return float(calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma))


def result_json_path(
    checkpoint_path: str | Path,
    *,
    dataset_name: str,
    eval_weights: str,
    num_samples: int,
    output_dir: str | Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    filename = (
        f"{checkpoint_path.stem}_fid_report_{dataset_name}_{normalize_eval_weights(eval_weights)}_"
        f"{int(num_samples)}.json"
    )
    if output_dir is not None:
        target_dir = Path(output_dir).resolve()
    else:
        target_dir = checkpoint_path.parent.parent / "evaluation"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename


def summary_json_path(
    checkpoint_dir: str | Path,
    *,
    dataset_name: str,
    eval_weights: str,
    num_samples: int,
    output_dir: str | Path | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    filename = (
        f"{checkpoint_dir.name}_fid_report_{dataset_name}_{normalize_eval_weights(eval_weights)}_"
        f"{int(num_samples)}_summary.json"
    )
    if output_dir is not None:
        target_dir = Path(output_dir).resolve()
    else:
        target_dir = checkpoint_dir.parent / "evaluation"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename


def evaluate_single_checkpoint(
    *,
    checkpoint_path: str | Path,
    dataset_name: str,
    expected_checkpoint_dataset: str | None,
    real_stats: CachedStats,
    num_samples: int,
    sample_batch_size: int,
    eval_weights: str,
    device: torch.device,
    compile_model: bool,
    seed: int,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    seed_everything(seed)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_dataset = infer_dataset_name_from_checkpoint(checkpoint)
    if (
        expected_checkpoint_dataset is not None
        and checkpoint_dataset is not None
        and checkpoint_dataset != expected_checkpoint_dataset
    ):
        print(
            f"Warning: checkpoint declares dataset '{checkpoint_dataset}', "
            f"but this script targets '{expected_checkpoint_dataset}'."
        )

    model = load_model_from_checkpoint(checkpoint, device=device, eval_weights=eval_weights)
    if compile_model:
        model = torch.compile(model)

    inception = create_inception(device)
    fake_mu, fake_sigma, fake_count = compute_stats_for_generated_samples(
        model=model,
        num_images=num_samples,
        batch_size=sample_batch_size,
        inception=inception,
        device=device,
        desc=f"Fake {dataset_name} activations",
    )
    fid = compute_fid_from_stats(real_stats.mu, real_stats.sigma, fake_mu, fake_sigma)

    result = {
        "checkpoint": str(checkpoint_path),
        "weights": normalize_eval_weights(eval_weights),
        "dataset": dataset_name,
        "checkpoint_dataset": checkpoint_dataset,
        "device": str(device),
        "seed": int(seed),
        "num_generated_samples": int(fake_count),
        "sample_batch_size": int(sample_batch_size),
        "real_stats_path": str(real_stats.path.resolve()),
        "real_num_images": int(real_stats.num_items),
        "fid": float(fid),
    }
    result_path = result_json_path(
        checkpoint_path,
        dataset_name=dataset_name,
        eval_weights=eval_weights,
        num_samples=num_samples,
        output_dir=output_dir,
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved report FID result to {result_path}")
    return result
