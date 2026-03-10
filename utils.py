"""Dataset and transform utilities for VQ-VAE-2 training on brain MRI."""

import enum
import logging
import os

import numpy as np
import pandas as pd
import torch

# Suppress nibabel's noisy "pixdim[0] (qfac) should be 1 or -1" info messages.
# Our pipeline reorients to RAS via Orientationd, so the default qfac=1 is safe.
logging.getLogger("nibabel.nifti1").setLevel(logging.WARNING)
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
    """Build MONAI training and validation transform pipelines.

    Args:
        spacing: Isotropic voxel spacing in mm.
        crop_margin: Voxels to crop from each edge.

    Returns:
        (train_transforms, val_transforms)
    """
    spatial_size = get_spatial_size(spacing)
    if crop_margin > 0:
        spatial_size = tuple(s - 2 * crop_margin for s in spatial_size)

    log.info(f"Voxel spacing: {spacing}mm, spatial size: {spatial_size}")

    def _build(is_training=False):
        t = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        ]
        if spacing != 1.0:
            t.append(
                Spacingd(keys=["image"], pixdim=(spacing, spacing, spacing), mode="bilinear")
            )
        t.extend([
            Orientationd(keys=["image"], axcodes="RAS"),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ])
        if is_training:
            t.extend([
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
        t.append(ToTensord(keys=["image"]))
        return Compose(t)

    return _build(is_training=True), _build(is_training=False)


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
