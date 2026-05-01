with open('train.py', 'r') as f:
    code = f.read()

code = code.replace(
    'from torch.utils.data import DataLoader',
    'from torch.utils.data import DataLoader\nfrom monai.data import Dataset, CacheDataset'
)

code = code.replace(
    'from monai.data import Dataset, CacheDataset\n    # Standard Dataset (no cache)',
    '# Standard Dataset (no cache)'
)
code = code.replace(
    'from monai.data import CacheDataset\n    val_set = CacheDataset',
    'val_set = CacheDataset'
)

with open('train.py', 'w') as f:
    f.write(code)
