import os
from pathlib import Path

import torch
import numpy as np
from torchvision.utils import make_grid, save_image


def ensure_three_channels(images):
    """Repeat grayscale images across RGB channels when needed."""
    if images.ndim == 3:
        if images.shape[0] == 3:
            return images
        if images.shape[0] == 1:
            return images.repeat(3, 1, 1)
    elif images.ndim == 4:
        if images.shape[1] == 3:
            return images
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected 1- or 3-channel image tensor, got shape {tuple(images.shape)}")


def sample_model_noise(model, n, device, memory_format=None):
    """Sample latent noise matching the model's input shape."""
    module = getattr(model, "module", model)
    module = getattr(module, "_orig_mod", module)
    channels = getattr(module, "in_ch", module.conv_in.in_channels)
    image_size = getattr(module, "image_size", 32)
    z = torch.randn(n, channels, image_size, image_size, device=device)
    if memory_format is not None:
        z = z.to(memory_format=memory_format)
    return z


def load_sample_noise(path, device, memory_format=None):
    """Load a saved sample-noise tensor or payload and move it onto ``device``."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    noise = payload.get("noise") if isinstance(payload, dict) else payload
    if not torch.is_tensor(noise):
        raise TypeError(f"Expected tensor noise payload at {path}, got {type(noise)!r}")
    if noise.ndim != 4:
        raise ValueError(f"Expected saved sample noise with shape [N, C, H, W], got {tuple(noise.shape)}")
    noise = noise.to(device=device, dtype=torch.float32, non_blocking=True)
    if memory_format is not None:
        noise = noise.contiguous(memory_format=memory_format)
    return noise


def drift_sample_from_noise(model, noise):
    """Generate samples from a provided latent-noise tensor."""
    with torch.no_grad():
        x = model(noise)
    return x.clamp(-1, 1)


def drift_sample(model, n, device):
    """Generate samples from drifting model (1 forward pass)."""
    z = sample_model_noise(model, n, device)
    return drift_sample_from_noise(model, z)


def save_sample_grid(samples, path, nrow=8):
    """Save a grid of samples to disk. Expects samples in [-1, 1]."""
    grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)
    save_image(grid, path)


def generate_fid_images(model, n_images, output_dir, device, sample_fn, batch_size=256):
    """Generate images for FID computation and save as individual PNGs.

    Args:
        model: the generative model
        n_images: total number of images to generate
        output_dir: directory to save images
        device: torch device
        sample_fn: callable(model, batch_size, device) -> [B, C, H, W] in [-1, 1]
        batch_size: batch size for generation
    """
    os.makedirs(output_dir, exist_ok=True)
    idx = 0
    while idx < n_images:
        bs = min(batch_size, n_images - idx)
        samples = sample_fn(model, bs, device)
        # Convert to [0, 1] for saving
        samples = (samples + 1) / 2
        samples = ensure_three_channels(samples)

        for i in range(bs):
            save_image(samples[i], os.path.join(output_dir, f"{idx:05d}.png"))
            idx += 1

        if idx % 1000 == 0:
            print(f"  Generated {idx}/{n_images} images")


def cleanup_fid_images(output_dir, pattern="*.png", remove_empty_dir=True):
    """Delete temporary per-image files written for pytorch-fid."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0

    removed = 0
    for path in output_dir.glob(pattern):
        if not path.is_file():
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            print(f"  Warning: could not remove temporary FID image {path}: {exc}")

    if remove_empty_dir:
        try:
            output_dir.rmdir()
        except OSError:
            pass

    if removed:
        print(f"  Removed {removed} temporary FID images from {output_dir}")
    return removed


def save_dataset_images(dataset, output_dir, n_images=10000, dataset_name="dataset"):
    """Save dataset images as individual PNGs for FID reference."""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(n_images, len(dataset))):
        img, _ = dataset[i]
        # img is already a tensor in [-1, 1] if transformed
        img = (img + 1) / 2
        img = ensure_three_channels(img)
        save_image(img, os.path.join(output_dir, f"{i:05d}.png"))
        if (i + 1) % 1000 == 0:
            print(f"  Saved {i + 1}/{n_images} {dataset_name} reference images")


def save_cifar10_images(dataset, output_dir, n_images=10000):
    """Backward-compatible wrapper for CIFAR-10 reference generation."""
    save_dataset_images(
        dataset=dataset,
        output_dir=output_dir,
        n_images=n_images,
        dataset_name="CIFAR-10",
    )
