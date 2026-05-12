from __future__ import annotations

from pathlib import Path

import yaml


def release_root() -> Path:
    return Path(__file__).resolve().parents[1]


def release_profile_path() -> Path:
    return release_root() / "config" / "release_profiles.yaml"


def load_release_profiles() -> dict:
    return yaml.safe_load(release_profile_path().read_text())


def normalize_dataset_name(dataset_name: str) -> str:
    requested = str(dataset_name).strip().lower()
    datasets = load_release_profiles().get("datasets", {})
    for key, payload in datasets.items():
        aliases = {str(alias).strip().lower() for alias in payload.get("aliases", [])}
        aliases.add(str(key).strip().lower())
        if requested in aliases:
            return key
    raise ValueError(f"Unknown release dataset profile: {dataset_name}")


def dataset_cli_name(dataset_name: str) -> str:
    normalized = normalize_dataset_name(dataset_name)
    if normalized == "imagenet":
        return "imagenet32"
    return normalized


def get_release_profile(dataset_name: str) -> dict:
    normalized = normalize_dataset_name(dataset_name)
    return load_release_profiles()["datasets"][normalized]
