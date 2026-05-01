with open('train.py', 'r') as f:
    code = f.read()

train_tform = """
    train_transform, val_transform = transforms(
        spacing=args.image_spacing,
        crop_margin=args.crop_margin,
        asymmetric_aug=getattr(args, 'asymmetric_aug', False),
    )"""
val_tform = """
    _, val_transform = transforms(
        spacing=args.image_spacing,
        crop_margin=args.crop_margin,
    )"""

code = code.replace(train_tform.strip(), """train_transform, val_transform = transforms(
        spacing=args.image_spacing,
        crop_margin=args.crop_margin,
        spatial_size=getattr(args, "spatial_size", None),
        asymmetric_aug=getattr(args, 'asymmetric_aug', False),
    )""")

code = code.replace(val_tform.strip(), """_, val_transform = transforms(
        spacing=args.image_spacing,
        crop_margin=args.crop_margin,
        spatial_size=getattr(args, "spatial_size", None),
    )""")

with open('train.py', 'w') as f:
    f.write(code)
