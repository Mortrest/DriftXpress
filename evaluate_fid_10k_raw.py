import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from common.fid_files import sample_reference_dir
from evaluation.fid import compute_fid
from evaluation.sample import (
    cleanup_fid_images,
    drift_sample,
    generate_fid_images,
    save_dataset_images,
)
from models.unet import UNet


DEFAULT_NUM_SAMPLES = 10_000
SUPPORTED_EVAL_WEIGHTS = ("raw", "ema")
DATASET_SPECS = {
    "cifar10": {
        "dataset_cls": datasets.CIFAR10,
        "pretty_name": "CIFAR-10",
        "image_size": 32,
        "model_channels": 3,
        "pad": None,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
        "class_names": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    },
    "cifar100": {
        "dataset_cls": datasets.CIFAR100,
        "pretty_name": "CIFAR-100",
        "image_size": 32,
        "model_channels": 3,
        "pad": None,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
        "class_names": None,
    },
    "mnist": {
        "dataset_cls": datasets.MNIST,
        "pretty_name": "MNIST",
        "image_size": 32,
        "model_channels": 1,
        "pad": 2,
        "normalize_mean": (0.5,),
        "normalize_std": (0.5,),
        "class_names": [str(i) for i in range(10)],
    },
    "svhn": {
        "dataset_cls": datasets.SVHN,
        "pretty_name": "SVHN",
        "image_size": 32,
        "model_channels": 3,
        "pad": None,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
        "class_names": [str(i) for i in range(10)],
    },
}
SUPPORTED_DATASETS = tuple(DATASET_SPECS.keys())
DEFAULT_DATASET = "cifar10"
DEFAULT_REF_DIR = sample_reference_dir("cifar10", DEFAULT_NUM_SAMPLES)


def _strip_prefix_if_present(state_dict, prefix="_orig_mod."):
    if not state_dict:
        return state_dict
    if not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def normalize_eval_weights(eval_weights):
    normalized = str(eval_weights).strip().lower()
    if normalized not in SUPPORTED_EVAL_WEIGHTS:
        raise ValueError(
            f"Unsupported eval weights '{eval_weights}'. Expected one of {SUPPORTED_EVAL_WEIGHTS}."
        )
    return normalized


def normalize_dataset_name(dataset_name):
    if dataset_name is None:
        return DEFAULT_DATASET
    normalized = str(dataset_name).strip().lower()
    if normalized not in DATASET_SPECS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Expected one of {SUPPORTED_DATASETS}."
        )
    return normalized


def get_dataset_spec(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    return dataset_name, DATASET_SPECS[dataset_name]


def dataset_constructor_kwargs(dataset_name, train):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "svhn":
        return {"split": "train" if train else "test"}
    return {"train": bool(train)}


def default_ref_dir(dataset_name, num_samples=DEFAULT_NUM_SAMPLES):
    dataset_name = normalize_dataset_name(dataset_name)
    return sample_reference_dir(dataset_name, int(num_samples))


def build_dataset_transform(dataset_name):
    dataset_name, spec = get_dataset_spec(dataset_name)
    ops = []
    if spec["pad"] is not None:
        ops.append(transforms.Pad(spec["pad"]))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec["normalize_mean"], spec["normalize_std"]),
        ]
    )
    return transforms.Compose(ops)


def dataset_model_channels(dataset_name):
    dataset_name, spec = get_dataset_spec(dataset_name)
    return int(spec["model_channels"])


def dataset_image_size(dataset_name):
    dataset_name, spec = get_dataset_spec(dataset_name)
    return int(spec["image_size"])


def configure_unet_cfg_for_dataset(unet_cfg, dataset_name):
    unet_cfg.in_ch = dataset_model_channels(dataset_name)
    unet_cfg.out_ch = dataset_model_channels(dataset_name)
    unet_cfg.image_size = dataset_image_size(dataset_name)
    return unet_cfg


def build_cifar_dataset(dataset_name, root, train, download, transform):
    dataset_name, spec = get_dataset_spec(dataset_name)
    return spec["dataset_cls"](
        root=root,
        download=download,
        transform=transform,
        **dataset_constructor_kwargs(dataset_name, train),
    )


def get_dataset_targets(dataset):
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long)
    if hasattr(dataset, "labels"):
        return torch.as_tensor(dataset.labels, dtype=torch.long)
    raise AttributeError(
        f"Dataset of type {type(dataset).__name__} does not expose targets or labels."
    )


def get_dataset_class_names(dataset_name, dataset=None):
    dataset_name, spec = get_dataset_spec(dataset_name)
    if dataset is not None and hasattr(dataset, "classes"):
        return [str(name) for name in dataset.classes]

    class_names = spec.get("class_names")
    if class_names is not None:
        return [str(name) for name in class_names]

    if dataset is None:
        raise ValueError(
            f"Dataset {dataset_name!r} does not define builtin class names; provide a dataset instance."
        )

    targets = get_dataset_targets(dataset)
    num_classes = int(targets.max().item()) + 1 if targets.numel() else 0
    return [str(class_id) for class_id in range(num_classes)]


def load_model_from_checkpoint(ckpt, device, eval_weights="raw"):
    unet_cfg = ckpt["config"]["unet"]
    model = UNet(
        in_ch=unet_cfg.in_ch,
        out_ch=unet_cfg.out_ch,
        base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult,
        num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions,
        dropout=unet_cfg.dropout,
        num_heads=unet_cfg.num_heads,
        image_size=getattr(unet_cfg, "image_size", 32),
    ).to(device)

    eval_weights = normalize_eval_weights(eval_weights)
    state_key = "model" if eval_weights == "raw" else "ema"
    if state_key not in ckpt:
        raise KeyError(f"Checkpoint does not contain {eval_weights} weights.")

    state_dict = _strip_prefix_if_present(ckpt[state_key])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_cifar_reference(dataset_name, output_dir, data_root="./data", num_images=DEFAULT_NUM_SAMPLES):
    dataset_name, spec = get_dataset_spec(dataset_name)
    output_dir = Path(output_dir)
    existing = sorted(output_dir.glob("*.png"))
    if len(existing) == num_images:
        print(f"Using existing {spec['pretty_name']} reference set: {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    for png in existing:
        png.unlink()

    dataset = build_cifar_dataset(
        dataset_name=dataset_name,
        root=data_root,
        train=True,
        download=True,
        transform=build_dataset_transform(dataset_name),
    )
    save_dataset_images(
        dataset=dataset,
        output_dir=str(output_dir),
        n_images=num_images,
        dataset_name=spec["pretty_name"],
    )
    return output_dir


def prepare_cifar10_reference(output_dir, data_root="./data", num_images=DEFAULT_NUM_SAMPLES):
    return prepare_cifar_reference(
        dataset_name="cifar10",
        output_dir=output_dir,
        data_root=data_root,
        num_images=num_images,
    )


def default_fake_dir(checkpoint_path, eval_weights="raw", num_samples=DEFAULT_NUM_SAMPLES):
    checkpoint_path = Path(checkpoint_path).resolve()
    eval_weights = normalize_eval_weights(eval_weights)
    return checkpoint_path.parent.parent / "evaluation" / f"{checkpoint_path.stem}_fid_{int(num_samples)}_{eval_weights}"


def infer_dataset_name_from_checkpoint(ckpt, default=DEFAULT_DATASET):
    drift_cfg = ckpt.get("config", {}).get("drift")
    dataset_name = None
    if isinstance(drift_cfg, dict):
        dataset_name = drift_cfg.get("dataset")
    elif drift_cfg is not None:
        dataset_name = getattr(drift_cfg, "dataset", None)

    if dataset_name is None:
        return normalize_dataset_name(default)
    return normalize_dataset_name(dataset_name)


def summary_path_for_checkpoint_dir(checkpoint_dir, output_dir, eval_weights="raw", num_samples=DEFAULT_NUM_SAMPLES):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    eval_weights = normalize_eval_weights(eval_weights)
    if output_dir is not None:
        summary_dir = Path(output_dir)
    else:
        summary_dir = checkpoint_dir.parent / "evaluation"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / f"{checkpoint_dir.name}_fid_{int(num_samples)}_{eval_weights}_summary.json"


def resolve_checkpoint_paths(checkpoint_arg):
    checkpoint_path = Path(checkpoint_arg).resolve()
    if checkpoint_path.is_dir():
        checkpoints = sorted(
            path for path in checkpoint_path.glob("*.pt")
            if path.name != "drift_latest.pt"
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint .pt files found in {checkpoint_path} after excluding drift_latest.pt."
            )
        return checkpoints, True

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    return [checkpoint_path], False


def fake_dir_for_checkpoint(
    checkpoint_path,
    output_dir,
    multi_checkpoint,
    eval_weights="raw",
    num_samples=DEFAULT_NUM_SAMPLES,
):
    if output_dir is None:
        return default_fake_dir(checkpoint_path, eval_weights=eval_weights, num_samples=num_samples)

    base_dir = Path(output_dir)
    if multi_checkpoint:
        return base_dir / checkpoint_path.stem
    return base_dir


def evaluate_loaded_model(
    model,
    fake_dir,
    ref_dir,
    device,
    args,
    *,
    dataset_name=None,
    eval_weights="raw",
    checkpoint_path=None,
    metadata=None,
):
    eval_weights = normalize_eval_weights(eval_weights)
    num_samples = int(getattr(args, "num_samples", DEFAULT_NUM_SAMPLES))
    dataset_name = normalize_dataset_name(dataset_name or DEFAULT_DATASET)
    fake_dir = Path(fake_dir)
    ref_dir = Path(ref_dir)
    fake_dir.mkdir(parents=True, exist_ok=True)
    for png in fake_dir.glob("*.png"):
        png.unlink()

    removed_generated_images = 0
    fid = None
    was_training = bool(model.training)
    model.eval()
    try:
        print(f"Generating {num_samples} samples using {eval_weights} weights to {fake_dir} ...")
        generate_fid_images(
            model=model,
            n_images=num_samples,
            output_dir=str(fake_dir),
            device=device,
            sample_fn=drift_sample,
            batch_size=args.batch_size,
        )

        device_str = f"{device.type}:0" if device.type == "cuda" and device.index is None else str(device)
        fid = compute_fid(
            str(ref_dir),
            str(fake_dir),
            device=device_str,
            batch_size=args.fid_batch_size,
            num_workers=args.fid_num_workers,
            timeout=args.fid_timeout,
        )
        if fid is None:
            source_label = checkpoint_path if checkpoint_path is not None else "live model"
            raise RuntimeError(f"FID computation failed for {source_label}.")
    finally:
        if was_training:
            model.train()
        removed_generated_images = cleanup_fid_images(fake_dir)

    result = {
        "checkpoint": (
            str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None
        ),
        "weights": eval_weights,
        "dataset": dataset_name,
        "reference_dir": str(ref_dir.resolve()),
        "generated_dir": str(fake_dir.resolve()),
        "num_generated_images_removed": removed_generated_images,
        "num_samples": num_samples,
        "fid": fid,
    }
    if metadata:
        result.update(metadata)
    result_path = fake_dir.parent / f"{fake_dir.name}_fid.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved FID result to {result_path}")
    return result


def evaluate_checkpoint(checkpoint_path, fake_dir, ref_dir, device, args):
    eval_weights = normalize_eval_weights(getattr(args, "eval_weights", "raw"))
    ckpt = load_checkpoint(checkpoint_path)
    dataset_name = infer_dataset_name_from_checkpoint(ckpt)
    model = load_model_from_checkpoint(ckpt=ckpt, device=device, eval_weights=eval_weights)
    if args.compile:
        model = torch.compile(model)
    return evaluate_loaded_model(
        model=model,
        fake_dir=fake_dir,
        ref_dir=ref_dir,
        device=device,
        args=args,
        dataset_name=dataset_name,
        eval_weights=eval_weights,
        checkpoint_path=checkpoint_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt file or directory of checkpoints.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--ref-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fid-batch-size", type=int, default=50)
    parser.add_argument("--fid-num-workers", type=int, default=0)
    parser.add_argument("--fid-timeout", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--eval-weights",
        "--weights",
        dest="eval_weights",
        type=str,
        default="raw",
        choices=SUPPORTED_EVAL_WEIGHTS,
        help="Which checkpoint weights to use for generation and FID.",
    )
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_paths, multi_checkpoint = resolve_checkpoint_paths(args.checkpoint)
    first_ckpt = load_checkpoint(checkpoint_paths[0])
    dataset_name = (
        normalize_dataset_name(args.dataset)
        if args.dataset is not None
        else infer_dataset_name_from_checkpoint(first_ckpt)
    )
    ref_dir_arg = (
        Path(args.ref_dir)
        if args.ref_dir is not None
        else default_ref_dir(dataset_name, args.num_samples)
    )

    ref_dir = prepare_cifar_reference(
        dataset_name=dataset_name,
        output_dir=ref_dir_arg,
        data_root=args.data_root,
        num_images=args.num_samples,
    )

    results = []
    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating checkpoint: {checkpoint_path}")
        fake_dir = fake_dir_for_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=args.output_dir,
            multi_checkpoint=multi_checkpoint,
            eval_weights=args.eval_weights,
            num_samples=args.num_samples,
        )
        results.append(
            evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                fake_dir=fake_dir,
                ref_dir=ref_dir,
                device=device,
                args=args,
            )
        )

    if multi_checkpoint:
        summary = {
            "checkpoint_dir": str(Path(args.checkpoint).resolve()),
            "weights": args.eval_weights,
            "dataset": dataset_name,
            "reference_dir": str(ref_dir.resolve()),
            "num_samples": args.num_samples,
            "results": results,
        }
        summary_path = summary_path_for_checkpoint_dir(
            args.checkpoint,
            args.output_dir,
            eval_weights=args.eval_weights,
            num_samples=args.num_samples,
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"Saved combined FID summary to {summary_path}")


if __name__ == "__main__":
    main()
