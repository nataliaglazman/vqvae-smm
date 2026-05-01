import re

with open('utils.py', 'r') as f:
    content = f.read()

# 1. Insert _find_brain_mask before load_data
find_mask_func = """
def _find_brain_mask(dir_path, require_substr=None):
    \"\"\"Return path to ``*_brain_mask.nii.gz`` in ``dir_path``, or ``None``.\"\"\"
    if not os.path.exists(dir_path):
        return None
    for file in os.listdir(dir_path):
        if not file.endswith("_brain_mask.nii.gz"):
            continue
        if require_substr is not None and require_substr not in file:
            continue
        return os.path.join(dir_path, file)
    return None

def load_data"""

content = content.replace("def load_data", find_mask_func)

# 2. Replace load_data implementation
old_load_data = """def load_data(df, data_dir, label_map):
    \"\"\"Scan data_dir for T1 images matching subjects in the dataframe.\"\"\"
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
    return items, missing"""

new_load_data = """def load_data(df, data_dir, label_map, load_masks=False):
    \"\"\"Scan data_dir for T1 images matching subjects in the dataframe.\"\"\"
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
    return items, missing"""

content = content.replace(old_load_data, new_load_data)

# 3. Replace load_items implementation
old_load_items = """def load_items(data_dir, csv_path):
    \"\"\"Load all data items from CSV + data directory.\"\"\"
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
    return items"""

new_load_items = """def load_items(data_dir, csv_path, load_masks=False):
    \"\"\"Load all data items from CSV + data directory.\"\"\"
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
    return items"""

content = content.replace(old_load_items, new_load_items)

with open('utils.py', 'w') as f:
    f.write(content)
