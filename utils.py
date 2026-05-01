"""Dataset and transform utilities for VQ-VAE-2 training on brain MRI."""

import enum
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from monai.data import CacheDataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandShiftIntensityd,
    Resized,
    ResizeWithPadOrCropd,
    Spacingd,
    ToTensord,
    CreateBrainMaskd,
    ApplyBrainMaskd,
    IntersectMasksd,
     RandScaleIntensityd,
     RandBiasFieldd,
     RandAdjustContrastd,
     RandGaussianNoised,
     RandGaussianSmoothd,
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

    .. deprecated::
        Prefer :func:`probe_spatial_size` which reads the actual image
        dimensions instead of relying on hardcoded lookup tables.
    """
    if spacing == 1.0:
        return (182, 218, 182)
    elif spacing == 2.0:
        return (91, 109, 91)
    else:
        return tuple(int(s / spacing) for s in (182, 218, 182))


def probe_spatial_size(sample_path, spacing=1.0):
    """Load a single image and return its spatial size after resampling.

    This avoids hardcoded dimension tables — the size is read directly
    from the data at whatever voxel spacing is requested.

    Args:
        sample_path: Path to one NIfTI (or other MONAI-readable) image.
        spacing: Isotropic voxel spacing in mm.

    Returns:
        Tuple (D, H, W) after resampling + RAS orientation.
    """
    probe = [
        LoadImaged(keys=["image"], image_only=True),
        Lambdad(keys=["image"], func=_ensure_3d_image),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    ]
    if spacing != 1.0:
        probe.append(
            Spacingd(keys=["image"], pixdim=(spacing, spacing, spacing), mode="bilinear")
        )
    probe.append(Orientationd(keys=["image"], axcodes="RAS"))
    result = Compose(probe)({"image": sample_path})
    return tuple(result["image"].shape[1:])  # (D, H, W) after channel dim


def _ensure_3d_image(x):
    """Convert loaded image arrays to 3D before orientation transforms.

    Handles common cases like (1, D, H, W), (D, H, W, 1), and arbitrary
    higher-dimensional arrays by squeezing singleton axes and selecting the
    first volume along trailing extra dimensions.
    """
    x = np.asarray(x)
    x = np.squeeze(x)
    while x.ndim > 3:
        x = x[..., 0]
    return x


def _build_single_image_transforms(spacing=2.0, crop_margin=0, downsample=1.0,
                                   sample_path=None):
    """Build deterministic + random transform pipelines for CacheDataset.

    The deterministic transforms (load, resample, orient, normalize) are
    executed once and cached in RAM by MONAI's CacheDataset.  The random
    augmentation transforms are applied on-the-fly each epoch.

    Args:
        spacing: Isotropic voxel spacing in mm.
        crop_margin: Voxels to crop from each edge (0 = no crop).
        downsample: Extra spatial downsampling factor applied after
            resampling/cropping (e.g. 0.5 = halve each dim, 1.0 = no change).
        sample_path: Path to one image used to probe native spatial size.
            If provided, dimensions are read from the data instead of using
            the hardcoded lookup table in :func:`get_spatial_size`.

    Returns:
        (det_transform, rand_transform) — pass det_transform to
        ``build_cached_dataset`` and rand_transform for training only.
    """
    # ── Determine native spatial size ────────────────────────────────────
    if sample_path is not None:
        native_size = probe_spatial_size(sample_path, spacing=spacing)
        log.info(f"Probed spatial size from data: {native_size} "
                 f"(spacing={spacing}mm)")
    else:
        native_size = get_spatial_size(spacing)
        log.info(f"Using hardcoded spatial size: {native_size} "
                 f"(spacing={spacing}mm)")

    # ── Apply crop margin ────────────────────────────────────────────────
    if crop_margin > 0:
        spatial_size = tuple(s - 2 * crop_margin for s in native_size)
        assert all(s > 0 for s in spatial_size), (
            f"crop_margin={crop_margin} is too large for native size {native_size}"
        )
        log.info(f"After cropping {crop_margin}px per edge: {spatial_size}")
    else:
        spatial_size = native_size

    # ── Apply downsampling ───────────────────────────────────────────────
    if downsample < 1.0:
        spatial_size = tuple(max(1, int(s * downsample)) for s in spatial_size)
        log.info(f"After downsampling (factor {downsample}): {spatial_size}")

    log.info(f"Final spatial size: {spatial_size}")

    # ── Deterministic preprocessing (cached after first pass) ────────────
    det = [
        LoadImaged(keys=["image"], image_only=True),
        # Ensure exactly 3 spatial dims before Orientationd("RAS").
        Lambdad(keys=["image"], func=_ensure_3d_image),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    ]
    if spacing != 1.0:
        det.append(
            Spacingd(keys=["image"], pixdim=(spacing, spacing, spacing), mode="bilinear")
        )
    det.append(Orientationd(keys=["image"], axcodes="RAS"))

    # Crop first (at native resolution), then downsample if requested
    if crop_margin > 0:
        cropped_size = tuple(s - 2 * crop_margin for s in native_size)
        det.append(CenterSpatialCropd(keys=["image"], roi_size=cropped_size))

    if downsample < 1.0:
        det.append(
            Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear")
        )
    else:
        # Pad/crop to uniform size (handles minor per-subject size variation)
        det.append(
            ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size)
        )

    det.extend([
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"], track_meta=False),
    ])
    det_transform = Compose(det)

    # Random augmentation (applied on-the-fly, never cached)
    rand_transform = Compose([
        RandAffined(
            keys=["image"],
            rotate_range=[0.05, 0.05, 0.05],          # 3 axes for 3D
            shear_range=[0.05, 0.05, 0.05,             # 6 values for 3D:
                         0.05, 0.05, 0.05],             # one per axis-pair
            scale_range=[0.05, 0.05, 0.05],            # 3 axes for 3D
            mode="bilinear",
            padding_mode="zeros",
            prob=0.5,
        ),
        RandShiftIntensityd(keys=["image"], offsets=(-0.1, 0.1), prob=0.2),
    ])

    return det_transform, rand_transform


def build_transforms(*args, **kwargs):
    """Backward-compatible wrapper for the single-image transform builder.

    Prefer :func:`transforms` for the paired-modality pipeline.  This wrapper
    preserves the older API used by the training script and external callers.
    """
    warnings.warn(
        "build_transforms is deprecated; use transforms for paired-modality "
        "pipelines or _build_single_image_transforms for the legacy single-image "
        "pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_single_image_transforms(*args, **kwargs)


def transforms(
    spacing=2.0,
    crop_margin=0,
    spatial_size=None,
    masks_from_disk=False,
    asymmetric_aug=False,
):
    """
    Create training and validation transforms for brain MRI images.

    Args:
        spacing: Isotropic voxel spacing in mm.
                 - 1.0: Original resolution (~182x218x182)
                 - 2.0: Downsampled (~91x109x91)
        crop_margin: Number of voxels to crop from each edge (all 6 sides).
                     E.g., crop_margin=4 removes 4 voxels from each side,
                     reducing each dimension by 8.
        spatial_size: Explicit (D, H, W) tuple.  When provided, overrides the
                      size derived from *spacing* and *crop_margin*.
        asymmetric_aug: When True, apply independent intensity augmentations per view
                        (T1 and T2 receive different random draws for shift, scale,
                        bias field, gamma, noise, and smoothing). Spatial augmentations
                        remain synced across views.
    Returns:
        train_transforms, val_transforms
    """
    if spatial_size is not None:
        spatial_size = tuple(spatial_size)
        if crop_margin > 0:
            logging.warning(
                f"--spatial-size {spatial_size} is set — --crop-margin {crop_margin} "
                f"will be applied on top (reducing each dim by {2 * crop_margin})."
            )
            spatial_size = tuple(s - 2 * crop_margin for s in spatial_size)
    else:
        # Calculate spatial size based on spacing
        # Original 1mm images are approximately 182x218x182
        if spacing == 1.0:
            spatial_size = (182, 218, 182)
        elif spacing == 2.0:
            spatial_size = (91, 109, 91)
        else:
            # Calculate proportionally from 1mm reference
            spatial_size = tuple(int(s / spacing) for s in (182, 218, 182))

        # Apply cropping: reduce each dimension by 2*crop_margin
        if crop_margin > 0:
            spatial_size = tuple(s - 2 * crop_margin for s in spatial_size)
            logging.info(f"Cropping {crop_margin} voxels from each edge")

    logging.info(f"Using voxel spacing: {spacing}mm, spatial size: {spatial_size}")

    # Common transforms list builder
    def build_transforms(is_training=False):
        if masks_from_disk:
            load_keys = ["image", "mask"]
            transforms_list = [
                LoadImaged(keys=load_keys),
                EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel"),
            ]
        else:
            transforms_list = [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                # Create brain mask BEFORE resampling (where original > 0)
                CreateBrainMaskd(keys=["image"], mask_keys=["mask"]),
            ]

        # Only add spacing transforms if not using original 1mm
        if spacing != 1.0:
            transforms_list.extend(
                [
                    Spacingd(
                        keys=["image"],
                        pixdim=(spacing, spacing, spacing),
                        mode="bilinear",
                    ),
                    Spacingd(
                        keys=["mask"],
                        pixdim=(spacing, spacing, spacing),
                        mode="nearest",
                    ),
                ]
            )

        transforms_list.extend(
            [
                Orientationd(keys=["image", "mask"], axcodes="RAS"),
                ResizeWithPadOrCropd(
                    keys=["image", "mask"],
                    spatial_size=spatial_size,
                ),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ]
        )
        transforms_list.append(
            ApplyBrainMaskd(
                keys=["image"],
                mask_keys=["mask"],
                threshold=0.5,
            )
        )

        # Add augmentations for training only
        if is_training:
            # Spatial transform is shared across views to preserve voxel-level
            # correspondence (required by patch-InfoNCE and content alignment).
            if asymmetric_aug:
                # Also transform masks so they stay aligned after the re-apply step below.
                transforms_list.append(
                    RandAffined(
                        keys=["image", "mask"],
                        mode=["bilinear", "nearest"],
                        rotate_range=[-0.05, 0.05],
                        shear_range=[0.001, 0.05],
                        scale_range=[0, 0.05],
                        padding_mode="zeros",
                        prob=0.5,
                    )
                )
            else:
                transforms_list.append(
                    RandAffined(
                        keys=["image", "mask"],
                        rotate_range=[-0.05, 0.05],
                        shear_range=[0.001, 0.05],
                        scale_range=[0, 0.05],
                        mode="bilinear",
                        padding_mode="zeros",
                        prob=0.5,
                    )
                )

            if asymmetric_aug:
                # Independent intensity augs per view — each view gets its own random draw.
                # Each transform instance has its own RNG, so calling the same transform
                # separately on image and mask yields different samples.
                for view_key in ("image", "mask"):
                    transforms_list.extend(
                        [
                            RandShiftIntensityd(keys=[view_key], offsets=(-0.1, 0.1), prob=0.5),
                            RandScaleIntensityd(keys=[view_key], factors=0.1, prob=0.5),
                            RandBiasFieldd(keys=[view_key], coeff_range=(0.0, 0.1), prob=0.3),
                            RandAdjustContrastd(keys=[view_key], gamma=(0.7, 1.5), prob=0.3),
                            RandGaussianNoised(keys=[view_key], std=0.05, prob=0.3),
                            RandGaussianSmoothd(
                                keys=[view_key],
                                sigma_x=(0.25, 1.0),
                                sigma_y=(0.25, 1.0),
                                sigma_z=(0.25, 1.0),
                                prob=0.2,
                            ),
                        ]
                    )
                # Re-apply brain mask so intensity augs don't leak signal into background.
                transforms_list.append(
                    ApplyBrainMaskd(
                        keys=["image"],
                        mask_keys=["mask"],
                        threshold=0.5,
                    )
                )
            else:
                transforms_list.append(
                    RandShiftIntensityd(keys=["image"], offsets=(-0.1, 0.1), prob=0.2)
                )

        transforms_list.append(ToTensord(keys=["image", "label"]))
        return Compose(transforms_list)

    train_transforms = build_transforms(is_training=True)
    val_transforms = build_transforms(is_training=False)

    return train_transforms, val_transforms



def _find_brain_mask(dir_path, require_substr=None):
    """Return path to ``*_brain_mask.nii.gz`` in ``dir_path``, or ``None``."""
    if not os.path.exists(dir_path):
        return None
    for file in os.listdir(dir_path):
        if not file.endswith("_brain_mask.nii.gz"):
            continue
        if require_substr is not None and require_substr not in file:
            continue
        return os.path.join(dir_path, file)
    return None

def load_data(df, data_dir, label_map, load_masks=False):
    """Scan data_dir for T1 images matching subjects in the dataframe."""
    exts = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".npy"]
    items, missing = [], []
    for _, row in df.iterrows():
        subj = str(row["Subject"])
        found_img = None
        found_mask = None
        t1_dir = os.path.join(data_dir, subj, "t1")
        if os.path.isdir(t1_dir):
            if load_masks:
                found_mask = _find_brain_mask(t1_dir)
            for f in os.listdir(t1_dir):
                if any(f.endswith(ext) for ext in exts) and not f.endswith("_brain_mask.nii.gz"):
                    found_img = os.path.join(t1_dir, f)
                    break
        if found_img and (not load_masks or found_mask):
            item = {"image": found_img, "label": label_map[row["Group"]], "subject": subj}
            if load_masks:
                item["mask"] = found_mask
            items.append(item)
        else:
            missing.append(subj)
    if missing:
        log.warning(f"Missing items for {len(missing)} subjects")
    log.info(f"Loaded {len(items)} subjects, {len(missing)} missing")
    return items, missing


def load_items(data_dir, csv_path, load_masks=False):
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
    items, _ = load_data(df, data_dir, label_map, load_masks=load_masks)
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


def save_decoded_images(data, recon, args, step: int, save_dir: Path) -> None:
    """
    Write original + pre-computed reconstruction as NIfTI files.

    Args:
        data: Current batch dictionary (must contain key ``"image"``).
        recon: Pre-computed reconstruction tensor for the first sample,
               shape ``(1, C, D, H, W)`` (already on CPU).
        args: Parsed argument namespace.
        step: Current training step (used in the output filename).
        save_dir: Directory to save the NIfTI files.
    """
    import nibabel as nib

    img = data["image"][0:1]  # (1, C, D, H, W) — first sample

    decoded_np = recon.float().squeeze().cpu().numpy()
    original_np = img.squeeze().cpu().numpy()

    spacing = getattr(args, "image_spacing", 2.0)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    os.makedirs(save_dir, exist_ok=True)

    nib.save(
        nib.Nifti1Image(original_np, affine=affine),
        f"{save_dir}/step_{step:05d}_original.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(decoded_np, affine=affine),
        f"{save_dir}/step_{step:05d}_decoded.nii.gz",
    )
    print(f"[SAVED] Decoded images at step {step} to {save_dir}/", flush=True)