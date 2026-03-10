import monai 
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CreateBrainMaskd,
    Spacingd,
    Orientationd,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
    ApplyBrainMaskd,
    RandAffined,
    RandShiftIntensityd,
    ToTensord,
)
import logging
import os
import pandas as pd


class ADNIDataset():

    def __init__(
        self,
        data_dir: str,
        change_lists=None,
        mode="train",
        transform=None,
        spacing=2.0,
        crop_margin=0,
        **kwargs,
    ):
        self.mode = mode
        self.data_dir = data_dir
        self.change_lists = change_lists or []
        self.spacing = spacing
        self.crop_margin = crop_margin

        # Load CSV and build item list
        csv_path = "/nfs/home/nglazman/cluster/labels_cleaned_3class.csv"
        df = pd.read_csv(csv_path)
        label_values = sorted(df["Group"].unique())
        label_map = {v: i for i, v in enumerate(label_values)}

        # Load data using utils.load_data
        self.items, missing = load_data(df, data_dir, label_map)
        self.num_samples = len(self.items)

        train_transforms, val_transforms = transform(spacing=self.spacing, crop_margin=self.crop_margin)
        self.monai_transform = train_transforms if mode == "train" else val_transforms

    def __len__(self):
        return self.num_samples

    def sample(self, size, random_state=None):
        """Sample for DCI evaluation - returns empty since we don't have latent factors."""
        return np.array([[]]), []

    def __getitem__(self, idx):
        """Return dict with T1 and T2 as two views."""
        item = self.items[idx]

        data_dict = {
            "image_t1": item["image"],
            "label": item["label"],
        }
        transformed = self.monai_transform(data_dict)

        img_t1 = transformed["image_t1"] 

        if idx == 0:
            print("[Dataset] Image dimensions after transforms:")
            print(f"  T1 shape: {img_t1.shape}")
        return {
            "image": [img_t1],
            "index": idx,
        }




def transforms(spacing=2.0, crop_margin=0):
    """
    Create training and validation transforms for brain MRI images.

    Args:
        spacing: Isotropic voxel spacing in mm.
                 - 1.0: Original resolution (~182x218x182)
                 - 2.0: Downsampled (~91x109x91)
        crop_margin: Number of voxels to crop from each edge (all 6 sides).
                     E.g., crop_margin=4 removes 4 voxels from each side,
                     reducing each dimension by 8.

    Returns:
        train_transforms, val_transforms
    """
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
        transforms_list = [
            LoadImaged(keys=["image_t1", "image_t2"]),
            EnsureChannelFirstd(keys=["image_t1", "image_t2"], channel_dim="no_channel"),
            # Create brain mask BEFORE resampling (where original > 0)
            CreateBrainMaskd(keys=["image_t1", "image_t2"], mask_keys=["mask_t1", "mask_t2"]),
        ]

        # Only add spacing transforms if not using original 1mm
        if spacing != 1.0:
            transforms_list.extend(
                [
                    Spacingd(
                        keys=["image_t1", "image_t2"],
                        pixdim=(spacing, spacing, spacing),
                        mode="bilinear",
                    ),
                    Spacingd(
                        keys=["mask_t1", "mask_t2"],
                        pixdim=(spacing, spacing, spacing),
                        mode="nearest",
                    ),
                ]
            )

        transforms_list.extend(
            [
                Orientationd(keys=["image_t1", "image_t2", "mask_t1", "mask_t2"], axcodes="RAS"),
                ResizeWithPadOrCropd(
                    keys=["image_t1", "image_t2", "mask_t1", "mask_t2"],
                    spatial_size=spatial_size,
                ),
                NormalizeIntensityd(keys=["image_t1", "image_t2"], nonzero=True, channel_wise=True),
                ApplyBrainMaskd(
                    keys=["image_t1", "image_t2"],
                    mask_keys=["mask_t1", "mask_t2"],
                    threshold=0.5,
                ),
            ]
        )

        # Add augmentations for training only
        if is_training:
            transforms_list.extend(
                [
                    RandAffined(
                        keys=["image_t1", "image_t2"],
                        rotate_range=[-0.05, 0.05],
                        shear_range=[0.001, 0.05],
                        scale_range=[0, 0.05],
                        mode="bilinear",
                        padding_mode="zeros",
                        prob=0.5,
                    ),
                    RandShiftIntensityd(keys=["image_t1", "image_t2"], offsets=(-0.1, 0.1), prob=0.2),
                ]
            )

        transforms_list.append(ToTensord(keys=["image_t1", "image_t2", "label"]))
        return Compose(transforms_list)

    train_transforms = build_transforms(is_training=True)
    val_transforms = build_transforms(is_training=False)

    return train_transforms, val_transforms



def load_data(df_filtered, data_dir, label_map):
    exts = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".npy"]
    missing = []
    items = []
    for _, row in df_filtered.iterrows():
        subj = str(row["Subject"])
        found_t1 = None

        # Find T1 image
        for ext in exts:
            candidate = os.path.join(data_dir, subj, "t1")
            if os.path.exists(candidate):
                candidate_files = os.listdir(candidate)
                for file in candidate_files:
                    if file.endswith(ext):
                        found_t1 = os.path.join(candidate, file)
                        break
            if found_t1:
                break

        if found_t1:
            items.append(
                {
                    "image": found_t1,
                    "label": label_map[row["Group"]],
                    "subject": subj,
                }
            )
        else:
            missing.append(subj)
            if not found_t1:
                logging.warning(f"Missing T1 for subject {subj}")

    logging.info(f"Loaded {len(items)} subjects with both T1 and T2. Missing: {len(missing)}")
    return items, missing