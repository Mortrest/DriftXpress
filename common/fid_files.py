from __future__ import annotations

from pathlib import Path

from common.release_profiles import release_root


def fid_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip().lower()
    if normalized == "imagenet":
        return "imagenet32"
    return normalized


def fid_reference_root() -> Path:
    return release_root() / "fid_reference"


def sample_reference_dir(dataset_name: str, num_samples: int) -> Path:
    dataset_key = fid_dataset_name(dataset_name)
    return fid_reference_root() / f"{dataset_key}_train_{int(num_samples)}"


def full_stats_path(dataset_name: str) -> Path:
    dataset_key = fid_dataset_name(dataset_name)
    return fid_reference_root() / f"{dataset_key}_train_full_inception_stats.npz"
