with open('utils.py', 'r') as f:
    code = f.read()

code = code.replace(
    'transforms_list.append(ToTensord(keys=["image", "label"]))',
    'transforms_list.append(ToTensord(keys=["image", "mask"] if masks_from_disk or not is_training else ["image"]))'
)

with open('utils.py', 'w') as f:
    f.write(code)
