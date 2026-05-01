import re

with open('train.py', 'r') as f:
    train_code = f.read()

train_code = re.sub(
    r'det_transform, rand_transform = build_transforms\([\s\S]*?shared_brain_mask=args.shared_brain_mask,\s*\)',
    '''train_transform, val_transform = transforms(
        spacing=args.image_spacing,
        crop_margin=args.crop_margin,
        asymmetric_aug=getattr(args, 'asymmetric_aug', False),
    )''',
    train_code
)

train_code = re.sub(
    r'train_set = build_cached_dataset\([\s\S]*?num_workers=args\.workers,\s*\)',
    '''from monai.data import Dataset, CacheDataset
    # Standard Dataset (no cache) because train_transform includes random augmentations
    train_set = Dataset(data=train_dicts, transform=train_transform)''',
    train_code
)

train_code = re.sub(
    r'val_set = build_cached_dataset\([\s\S]*?num_workers=args\.workers,\s*\)',
    '''val_set = CacheDataset(
        data=val_dicts, transform=val_transform,
        cache_rate=args.val_cache_rate, num_workers=args.workers,
    )''',
    train_code
)

# And similarly for the evaluate block:
train_code = re.sub(
    r'det_transform, _ = build_transforms\([\s\S]*?sample_path=items\[0\]\["image"\],\s*\)',
    '''_, val_transform = transforms(
        spacing=args.image_spacing,
        crop_margin=args.crop_margin,
    )''',
    train_code
)

train_code = re.sub(
    r'val_set = build_cached_dataset\([\s\S]*?cache_rate=args\.val_cache_rate, num_workers=args\.workers,\s*\)',
    '''from monai.data import CacheDataset
    val_set = CacheDataset(
        data=val_dicts, transform=val_transform,
        cache_rate=args.val_cache_rate, num_workers=args.workers,
    )''',
    train_code
)

with open('train.py', 'w') as f:
    f.write(train_code)

