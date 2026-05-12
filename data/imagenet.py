from __future__ import annotations

import pickle
import re
import zipfile
from pathlib import Path

import numpy as np
import torch


IMAGENET32_BATCH_RE = re.compile(r"train_data_batch_(\d+)$")
_IMAGENET32_ARRAY_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def sorted_batch_names(names: list[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        match = IMAGENET32_BATCH_RE.search(Path(name).name)
        if match is None:
            raise ValueError(f"Unrecognized ImageNet32 batch name: {name}")
        return int(match.group(1)), name

    return sorted(names, key=key)


def _iter_archive_batches(source: Path):
    if source.is_dir():
        batch_paths = sorted_batch_names([str(path) for path in source.glob("train_data_batch_*")])
        if not batch_paths:
            raise FileNotFoundError(
                f"No train_data_batch_* files found in {source}. "
                "Use the original zip archive, an extracted batch directory, or train_data_full.npz."
            )
        for batch_path in batch_paths:
            path = Path(batch_path)
            with open(path, "rb") as handle:
                yield path.name, pickle.load(handle, encoding="latin1")
        return

    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            batch_names = sorted_batch_names(
                [name for name in archive.namelist() if IMAGENET32_BATCH_RE.search(Path(name).name)]
            )
            if not batch_names:
                raise FileNotFoundError(f"No train_data_batch_* entries found inside {source}.")
            for batch_name in batch_names:
                with archive.open(batch_name, "r") as handle:
                    yield Path(batch_name).name, pickle.load(handle, encoding="latin1")
        return

    raise FileNotFoundError(
        f"Unsupported ImageNet32 source: {source}. "
        "Use the original zip archive, an extracted batch directory, or train_data_full.npz."
    )


def load_imagenet32_arrays(source: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source_path = Path(source).expanduser().resolve()
    cache_key = str(source_path)
    cached = _IMAGENET32_ARRAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if source_path.is_file() and source_path.suffix == ".npz":
        with np.load(source_path, allow_pickle=False) as payload:
            if "data" not in payload or "labels" not in payload:
                raise KeyError(f"Expected 'data' and 'labels' arrays in {source_path}.")
            data = np.asarray(payload["data"], dtype=np.uint8)
            labels = np.asarray(payload["labels"], dtype=np.int64)
    else:
        total_samples = 0
        flat_dim = None
        for _, batch in _iter_archive_batches(source_path):
            batch_data = np.asarray(batch["data"], dtype=np.uint8)
            if batch_data.ndim != 2:
                raise ValueError(
                    f"Expected ImageNet32 batch data with shape [N, 3072], got {batch_data.shape}."
                )
            if flat_dim is None:
                flat_dim = int(batch_data.shape[1])
            elif int(batch_data.shape[1]) != flat_dim:
                raise ValueError("Encountered inconsistent flattened image sizes while loading ImageNet32.")
            total_samples += int(batch_data.shape[0])

        if total_samples <= 0 or flat_dim is None:
            raise ValueError(f"ImageNet32 source {source_path} did not yield any samples.")

        data = np.empty((total_samples, flat_dim), dtype=np.uint8)
        labels = np.empty((total_samples,), dtype=np.int64)
        offset = 0
        for _, batch in _iter_archive_batches(source_path):
            batch_data = np.asarray(batch["data"], dtype=np.uint8)
            batch_labels = np.asarray(batch["labels"], dtype=np.int64)
            batch_total = int(batch_data.shape[0])
            data[offset : offset + batch_total] = batch_data
            labels[offset : offset + batch_total] = batch_labels
            offset += batch_total

    _IMAGENET32_ARRAY_CACHE[cache_key] = (data, labels)
    return data, labels


def clear_imagenet32_array_cache(source: str | Path | None = None) -> None:
    if source is None:
        _IMAGENET32_ARRAY_CACHE.clear()
        return
    source_path = Path(source).expanduser().resolve()
    _IMAGENET32_ARRAY_CACHE.pop(str(source_path), None)


def list_imagenet32_batch_names(source: str | Path) -> list[str]:
    load_imagenet32_arrays(source)
    return ["train_data_full"]


def iter_imagenet32_batches(source: str | Path, batch_order: list[str] | None = None):
    data, labels = load_imagenet32_arrays(source)
    if batch_order not in (None, ["train_data_full"]):
        raise ValueError(
            "ImageNet32 batches are materialized eagerly now; batch_order must be omitted or "
            "contain the single synthetic batch name 'train_data_full'."
        )
    yield "train_data_full", {"data": data, "labels": labels}


def uint8_images_to_model_input(images: torch.Tensor) -> torch.Tensor:
    return images.float().div_(127.5).sub_(1.0)
