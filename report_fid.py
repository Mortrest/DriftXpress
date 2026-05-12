from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.fid_files import fid_dataset_name, full_stats_path
from data.imagenet import load_imagenet32_arrays
from evaluation.report_fid import (
    SUPPORTED_EVAL_WEIGHTS,
    create_inception,
    default_device,
    evaluate_single_checkpoint,
    get_or_compute_real_stats,
    resolve_checkpoint_paths,
    summary_json_path,
)


DEFAULT_NUM_SAMPLES = 50_000
TORCHVISION_DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "svhn": datasets.SVHN,
}


def normalize_report_dataset(dataset_name: str) -> str:
    normalized = str(dataset_name).strip().lower()
    if normalized in {"imagenet", "imagenet32"}:
        return "imagenet"
    if normalized not in TORCHVISION_DATASETS:
        raise ValueError(f"Unsupported report dataset: {dataset_name}")
    return normalized


def build_torchvision_iterator(dataset_name: str, data_root: str, real_batch_size: int, num_workers: int, device):
    dataset_name = normalize_report_dataset(dataset_name)
    if dataset_name == "svhn":
        dataset = datasets.SVHN(
            root=data_root,
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )
    else:
        dataset = TORCHVISION_DATASETS[dataset_name](
            root=data_root,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

    def iterator():
        loader = DataLoader(
            dataset,
            batch_size=real_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        for images, _ in loader:
            yield images

    return dataset, iterator


def build_imagenet_iterator(data_source: str, real_batch_size: int):
    data_source_path = Path(data_source).resolve()
    flat_images, _ = load_imagenet32_arrays(data_source_path)

    def iterator():
        total = int(flat_images.shape[0])
        for start in range(0, total, real_batch_size):
            batch = flat_images[start : start + real_batch_size]
            yield torch.from_numpy(batch).view(-1, 3, 32, 32).float().div_(255.0)

    return data_source_path, iterator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("cifar10", "cifar100", "svhn", "imagenet", "imagenet32"))
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt file or directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for JSON outputs.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--data-source", type=str, default="data/Imagenet32_train.zip")
    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--force-recompute-real-stats", action="store_true")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--real-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--eval-weights",
        "--weights",
        dest="eval_weights",
        type=str,
        default="ema",
        choices=SUPPORTED_EVAL_WEIGHTS,
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report_dataset = normalize_report_dataset(args.dataset)
    checkpoint_dataset = fid_dataset_name(report_dataset)
    device = default_device(args.device)
    stats_path = args.stats_path or str(full_stats_path(report_dataset))

    if report_dataset == "imagenet":
        data_source_path, real_batch_iterator_factory = build_imagenet_iterator(
            data_source=args.data_source,
            real_batch_size=args.real_batch_size,
        )
        expected_metadata = {
            "dataset": checkpoint_dataset,
            "split": "train",
            "data_source": str(data_source_path),
        }
        total_images = None
    else:
        dataset, real_batch_iterator_factory = build_torchvision_iterator(
            dataset_name=report_dataset,
            data_root=args.data_root,
            real_batch_size=args.real_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        expected_metadata = {
            "dataset": checkpoint_dataset,
            "split": "train",
            "data_root": str(Path(args.data_root).resolve()),
            "num_images": len(dataset),
        }
        total_images = len(dataset)

    inception = create_inception(device)
    real_stats = get_or_compute_real_stats(
        stats_path=stats_path,
        expected_metadata=expected_metadata,
        inception=inception,
        device=device,
        real_batch_iterator_factory=real_batch_iterator_factory,
        total_images=total_images,
        force_recompute=args.force_recompute_real_stats,
        desc=f"Real {checkpoint_dataset} activations",
    )
    del inception
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint_paths, multi_checkpoint = resolve_checkpoint_paths(args.checkpoint)
    results = []
    for checkpoint_path in checkpoint_paths:
        results.append(
            evaluate_single_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset_name=checkpoint_dataset,
                expected_checkpoint_dataset=checkpoint_dataset,
                real_stats=real_stats,
                num_samples=args.num_samples,
                sample_batch_size=args.sample_batch_size,
                eval_weights=args.eval_weights,
                device=device,
                compile_model=args.compile,
                seed=args.seed,
                output_dir=args.output_dir,
            )
        )

    if multi_checkpoint:
        summary = {
            "checkpoint_dir": str(Path(args.checkpoint).resolve()),
            "dataset": checkpoint_dataset,
            "weights": args.eval_weights,
            "num_generated_samples": int(args.num_samples),
            "real_stats_path": str(Path(stats_path).resolve()),
            "results": results,
        }
        summary_path = summary_json_path(
            args.checkpoint,
            dataset_name=checkpoint_dataset,
            eval_weights=args.eval_weights,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"Saved combined FID summary to {summary_path}")


if __name__ == "__main__":
    main()
