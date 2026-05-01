with open('train.py', 'r') as f:
    code = f.read()

code = code.replace(
    'items = load_items(args.dataroot, args.csv_path)',
    'items = load_items(args.dataroot, args.csv_path, load_masks=getattr(args, "masks_from_disk", False))'
)
code = code.replace(
    'train_dicts = [{"image": it["image"]} for it in train_items]',
    'train_dicts = [{"image": it["image"], "mask": it["mask"]} if "mask" in it else {"image": it["image"]} for it in train_items]'
)
code = code.replace(
    'val_dicts = [{"image": it["image"]} for it in val_items]',
    'val_dicts = [{"image": it["image"], "mask": it["mask"]} if "mask" in it else {"image": it["image"]} for it in val_items]'
)

with open('train.py', 'w') as f:
    f.write(code)
