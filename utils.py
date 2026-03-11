"""Dataset and transform utilities for VQ-VAE-2 training on brain MRI."""

import enum
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    Spacingd,
    ToTensord,
)

# Suppress nibabel's noisy "pixdim[0] (qfac) should be 1 or -1" info messages.
# Our pipeline reorients to RAS via Orientationd, so the default qfac=1 is safe.
# Must be set *after* MONAI import (which initialises nibabel loggers).
logging.getLogger("nibabel").setLevel(logging.WARNING)

log = logging.getLogger(__name__)


class TBSummaryTypes(str, enum.Enum):
    """Keys for TensorBoard summary dictionaries."""
    SCALAR = "scalar"
    IMAGE = "image"
    HISTOGRAM = "histogram"


def get_spatial_size(spacing):
    """Compute target spatial size for a given isotropic voxel spacing.

    Reference: 1mm images are approximately 182x218x182 voxels.
    """
    if spacing == 1.0:
        return (182, 218, 182)
    elif spacing == 2.0:
        return (91, 109, 91)
    else:
        return tuple(int(s / spacing) for s in (182, 218, 182))


def build_transforms(spacing=2.0, crop_margin=0):
    """Build deterministic + random transform pipelines for CacheDataset.

    The deterministic transforms (load, resample, orient, normalize) are
    executed once and cached in RAM by MONAI's CacheDataset.  The random
    augmentation transforms are applied on-the-fly each epoch.

    Args:
        spacing: Isotropic voxel spacing in mm.
        crop_margin: Voxels to crop from each edge.

    Returns:
        (det_transform, rand_transform) — pass det_transform to
        ``build_cached_dataset`` and rand_transform for training only.
    """
    spatial_size = get_spatial_size(spacing)
    if crop_margin > 0:
        spatial_size = tuple(s - 2 * crop_margin for s in spatial_size)

    log.info(f"Voxel spacing: {spacing}mm, spatial size: {spatial_size}")

    # Deterministic preprocessing (cached after first pass)
    det = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    ]
    if spacing != 1.0:
        det.append(
            Spacingd(keys=["image"], pixdim=(spacing, spacing, spacing), mode="bilinear")
        )
    det.extend([
        Orientationd(keys=["image"], axcodes="RAS"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])
    det_transform = Compose(det)

    # Random augmentation (applied on-the-fly, never cached)
    rand_transform = Compose([
        RandAffined(
            keys=["image"],
            rotate_range=[-0.05, 0.05],
            shear_range=[0.001, 0.05],
            scale_range=[0, 0.05],
            mode="bilinear",
            padding_mode="zeros",
            prob=0.5,
        ),
        RandShiftIntensityd(keys=["image"], offsets=(-0.1, 0.1), prob=0.2),
    ])

    return det_transform, rand_transform


def load_data(df, data_dir, label_map):
    """Scan data_dir for T1 images matching subjects in the dataframe."""
    exts = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".npy"]
    items, missing = [], []
    for _, row in df.iterrows():
        subj = str(row["Subject"])
        found = None
        t1_dir = os.path.join(data_dir, subj, "t1")
        if os.path.isdir(t1_dir):
            for f in os.listdir(t1_dir):
                if any(f.endswith(ext) for ext in exts):
                    found = os.path.join(t1_dir, f)
                    break
        if found:
            items.append({"image": found, "label": label_map[row["Group"]], "subject": subj})
        else:
            missing.append(subj)
    if missing:
        log.warning(f"Missing T1 for {len(missing)} subjects")
    log.info(f"Loaded {len(items)} subjects, {len(missing)} missing")
    return items, missing


def load_items(data_dir, csv_path):
    """Load all data items from CSV + data directory."""
    df = pd.read_csv(csv_path)
    missing_cols = {"Subject", "Group"} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV at {csv_path} is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    label_values = sorted(df["Group"].unique())
    label_map = {v: i for i, v in enumerate(label_values)}
    items, _ = load_data(df, data_dir, label_map)
    return items


class ADNIDataset(torch.utils.data.Dataset):
    """Brain MRI dataset returning preprocessed T1 volumes."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        data = self.transform({"image": item["image"]})
        return {"image": data["image"], "label": item["label"], "index": idx}


class CachedAugDataset(torch.utils.data.Dataset):
    """Wraps a MONAI CacheDataset and applies random augmentation on the fly.

    CacheDataset caches the expensive deterministic transforms (load, resample,
    orient, normalize) in RAM.  This wrapper applies lightweight random
    augmentation (affine jitter, intensity shift) every time ``__getitem__``
    is called, so each epoch sees different augmented views.
    """

    def __init__(self, cache_ds, rand_transform=None):
        self.cache_ds = cache_ds
        self.rand_transform = rand_transform

    def __len__(self):
        return len(self.cache_ds)

    def __getitem__(self, idx):
        data = self.cache_ds[idx]
        if self.rand_transform is not None:
            data = self.rand_transform(data)
        return data


def build_cached_dataset(data_dicts, det_transform, rand_transform=None,
                         cache_rate=1.0, num_workers=4):
    """Create a MONAI CacheDataset with optional on-the-fly augmentation.

    Args:
        data_dicts: List of dicts, each with at least an ``"image"`` key
                    pointing to a NIfTI path.
        det_transform: Deterministic Compose (load → normalize → tensor).
        rand_transform: Random Compose for training augmentation (or None for val).
        cache_rate: Fraction of data to cache in RAM (1.0 = all).
        num_workers: Workers used *during initial caching* (not DataLoader workers).

    Returns:
        A ``CachedAugDataset`` (if rand_transform) or plain ``CacheDataset``.
    """
    log.info(f"Building CacheDataset ({len(data_dicts)} items, "
             f"cache_rate={cache_rate}, workers={num_workers}) …")
    cache_ds = CacheDataset(
        data=data_dicts,
        transform=det_transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    if rand_transform is not None:
        return CachedAugDataset(cache_ds, rand_transform)
    return cache_ds


def save_decoded_images(model, data, args, step: int, save_dir: Path) -> None:
    """
    Encode then decode the first sample and write original + reconstruction as NIfTI files.

    Args:
        model: VQVAE model.
        data: Current batch dictionary (must contain key ``"image"``).
        args: Parsed argument namespace.
        step: Current training step (used in the output filename).
        save_dir: Directory to save the NIfTI files.
    """
    import nibabel as nib

    with torch.no_grad():
        samples = data["image"]
        img = samples[0:1]  # (1, C, D, H, W) — first sample, keep batch dim

        decoded = model(img.to(next(model.parameters()).device), return_recon=True)[0]

        decoded_np = decoded.squeeze().cpu().numpy()
        original_np = img.squeeze().cpu().numpy()

        affine_2mm = np.diag([2.0, 2.0, 2.0, 1.0])
        os.makedirs(save_dir, exist_ok=True)

        nib.save(
            nib.Nifti1Image(original_np, affine=affine_2mm),
            f"{save_dir}/step_{step:05d}_original.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(decoded_np, affine=affine_2mm),
            f"{save_dir}/step_{step:05d}_decoded.nii.gz",
        )
        print(f"[SAVED] Decoded images at step {step} to {save_dir}/", flush=True)