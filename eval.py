# TO DO:
# - Add classifiers for downstream tasks (e.g. AD vs CN, MCI vs CN, etc.)
# - Add Dan's methods for segmentations

import random
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from pathlib import Path

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    ToTensord,
)
from monai.data import Dataset, DataLoader as MonaiDataLoader

from vqvae2 import VQVAE
from loss import BaselineLoss
from utils import get_spatial_size


def get_transforms(spacing=2.0, spatial_size=None):
    """Get MONAI transforms for 3D medical image preprocessing."""
    if spatial_size is None:
        spatial_size = get_spatial_size(spacing)
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    ]
    if spacing != 1.0:
        transforms.append(
            Spacingd(keys=["image"], pixdim=(spacing, spacing, spacing), mode="bilinear")
        )
    transforms.extend([
        Orientationd(keys=["image"], axcodes="RAS"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])
    return Compose(transforms)


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """Recreate a VQVAE model from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    a = ckpt.get("args", {})
    model = VQVAE(
        in_channels=1,
        hidden_channels=a.get("vqvae_hidden_channels", 64),
        res_channels=a.get("vqvae_res_channels", 32),
        nb_levels=a.get("vqvae_nb_levels", 3),
        embed_dim=a.get("vqvae_embed_dim", 32),
        nb_entries=a.get("vqvae_nb_entries", 384),
        scaling_rates=a.get("vqvae_scaling_rates", [2, 2, 2]),
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_evaluate(checkpoint, dataroot, device, spacing=2.0, batch_size=4):
    """Run validation loss on all NIfTI images in a directory.

    Can be called standalone or from the CLI via ``--mode evaluate``.
    """
    model = load_model_from_checkpoint(checkpoint, device)

    data_dir = Path(dataroot)
    image_files = sorted(data_dir.glob("**/*.nii.gz")) + sorted(data_dir.glob("**/*.nii"))
    if not image_files:
        raise FileNotFoundError(f"No NIfTI files found in {data_dir}")
    data_dicts = [{"image": str(f)} for f in image_files]

    transforms = get_transforms(spacing=spacing)
    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = MonaiDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    loss_fn = BaselineLoss().to(device)

    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            recon, diffs, *_ = model(images)

            net_out = {"reconstruction": [recon], "quantization_losses": diffs}
            loss = loss_fn(net_out, images)
            total_loss += loss.item()
            n += 1

    avg_loss = total_loss / max(n, 1)
    print(f"Evaluated {n} batches ({len(image_files)} images)")
    print(f"Average Loss: {avg_loss:.4f}")

    # Codebook utilization
    for i, codebook in enumerate(model.codebooks):
        util = codebook.codebook_utilization()
        print(f"  Codebook {i}: {util['active_codes']}/{codebook.n_embed} active "
              f"({util['utilization']:.1%}), perplexity={util['perplexity']:.1f}")

    return avg_loss


# ──────────────────────────────────────────────────────────────────────────────
# Feature hooks
# ──────────────────────────────────────────────────────────────────────────────


class FeatureHook:
    """Captures the output tensor of any nn.Module via a forward hook."""
    def __init__(self):
        self.output = None

    def hook_fn(self, module, input, output):
        self.output = output[0] if isinstance(output, tuple) else output

    def register(self, module):
        return module.register_forward_hook(self.hook_fn)


def attach_hooks(model):
    """
    Attach hooks to all encoder levels and codebooks for the 3D VQ-VAE-2.
    Returns (hooks dict, handle list) — call handle.remove() to clean up.
    """
    hooks = {}
    handles = []

    for i, encoder in enumerate(model.encoders):
        hook = FeatureHook()
        hooks[f"encoder_{i}"] = hook
        handles.append(hook.register(encoder))

    for i, codebook in enumerate(model.codebooks):
        hook = FeatureHook()
        hooks[f"codebook_{i}"] = hook
        handles.append(hook.register(codebook))

    for i, decoder in enumerate(model.decoders):
        hook = FeatureHook()
        hooks[f"decoder_{i}"] = hook
        handles.append(hook.register(decoder))

    return hooks, handles


# ──────────────────────────────────────────────────────────────────────────────
# Volume / slice helpers
# ──────────────────────────────────────────────────────────────────────────────


def to_grid_3d(feat: torch.Tensor, max_channels: int = 16, slice_axis: int = 2) -> np.ndarray:
    """
    feat: (1, C, D, H, W)  →  grid image (grid_H, grid_W) normalised to [0,1]
    Takes the middle slice along slice_axis (default: axial/z-axis).
    """
    feat = feat.detach().cpu().float()
    C = min(feat.shape[1], max_channels)
    feat = feat[0, :C]  # (C, D, H, W)

    if slice_axis == 0:
        mid = feat.shape[1] // 2
        feat = feat[:, mid, :, :]
    elif slice_axis == 1:
        mid = feat.shape[2] // 2
        feat = feat[:, :, mid, :]
    else:
        mid = feat.shape[3] // 2
        feat = feat[:, :, :, mid]

    mn = feat.flatten(1).min(1).values[:, None, None]
    mx = feat.flatten(1).max(1).values[:, None, None]
    feat = (feat - mn) / (mx - mn + 1e-8)

    cols = math.ceil(math.sqrt(C))
    rows = math.ceil(C / cols)
    H, W = feat.shape[1], feat.shape[2]
    grid = np.zeros((rows * H, cols * W), dtype=np.float32)
    for idx in range(C):
        r, c = divmod(idx, cols)
        grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = feat[idx].numpy()
    return grid


def quant_to_heatmap_3d(quant_feat: torch.Tensor, slice_axis: int = 2) -> np.ndarray:
    """
    quant_feat: (1, C, D, H, W) → middle slice → mean across channels → (H, W) normalised
    """
    q = quant_feat.detach().cpu().float()[0]

    if slice_axis == 0:
        mid = q.shape[1] // 2
        q = q[:, mid, :, :]
    elif slice_axis == 1:
        mid = q.shape[2] // 2
        q = q[:, :, mid, :]
    else:
        mid = q.shape[3] // 2
        q = q[:, :, :, mid]

    hm = q.mean(0).numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm


def volume_to_slices(vol: torch.Tensor) -> dict:
    """
    vol: (1, 1, D, H, W) → dict with 'axial', 'coronal', 'sagittal' middle slices as np.ndarray
    """
    vol = vol.detach().cpu().float().squeeze()
    if vol.ndim == 4:
        vol = vol[0]

    slices = {
        "sagittal": vol[vol.shape[0] // 2, :, :].numpy(),
        "coronal": vol[:, vol.shape[1] // 2, :].numpy(),
        "axial": vol[:, :, vol.shape[2] // 2].numpy(),
    }

    for k in slices:
        s = slices[k]
        slices[k] = (s - s.min()) / (s.max() - s.min() + 1e-8)

    return slices


def _load_volume(path, spacing=2.0):
    """Load a single NIfTI volume and return (tensor, display_slices)."""
    transforms = get_transforms(spacing=spacing)
    data = transforms({"image": path})
    tensor = data["image"].unsqueeze(0)  # (1, C, D, H, W)
    display = volume_to_slices(tensor)
    return tensor, display


def _tensor_to_display(vol: torch.Tensor) -> dict:
    """Convert a (1, 1, D, H, W) reconstruction tensor to display slices."""
    return volume_to_slices(vol)


def _sample_random_image(dataroot, seed=None):
    """Pick a random NIfTI file from a data directory.

    Args:
        dataroot: Root directory to search recursively for .nii/.nii.gz files.
        seed: Optional random seed for reproducibility.

    Returns:
        Path string of the randomly selected image.
    """
    data_dir = Path(dataroot)
    image_files = sorted(data_dir.glob("**/*.nii.gz")) + sorted(data_dir.glob("**/*.nii"))
    if not image_files:
        raise FileNotFoundError(f"No NIfTI files found in {data_dir}")
    if seed is not None:
        random.seed(seed)
    chosen = random.choice(image_files)
    print(f"Randomly sampled: {chosen} (from {len(image_files)} images)")
    return str(chosen)


# ──────────────────────────────────────────────────────────────────────────────
# Feature map visualization (plot_features — existing)
# ──────────────────────────────────────────────────────────────────────────────


def plot_features(input_slices, hooks, recon_slices, nb_levels, save_path=None):
    """Plot feature maps from all encoder and codebook levels."""
    n_cols = max(nb_levels + 1, 4)
    fig = plt.figure(figsize=(5 * n_cols, 10), facecolor="#0e1118")
    fig.suptitle("VQ-VAE-2  ·  Feature Map Inspector",
                 color="#c8d0e8", fontsize=14, fontweight="bold",
                 fontfamily="monospace", y=0.98)

    gs = gridspec.GridSpec(2, n_cols, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.04, right=0.97, top=0.92, bottom=0.05)

    def styled_ax(gs_pos, title, subtitle=""):
        ax = fig.add_subplot(gs_pos, facecolor="#0e1118")
        ax.set_title(f"{title}\n{subtitle}", color="#9aa8c8", fontsize=8,
                     fontfamily="monospace", loc="left", pad=6)
        ax.axis("off")
        return ax

    ax = styled_ax(gs[0, 0], "INPUT", "axial slice")
    ax.imshow(input_slices["axial"], cmap="gray")

    cmaps = ["viridis", "plasma", "inferno", "cividis"]
    for i in range(nb_levels):
        key = f"encoder_{i}"
        if key in hooks and hooks[key].output is not None:
            feat = hooks[key].output
            grid = to_grid_3d(feat, max_channels=16)
            ax = styled_ax(gs[0, i + 1], f"ENCODER {i}",
                           f"feat maps · {feat.shape[1]} ch")
            ax.imshow(grid, cmap=cmaps[i % len(cmaps)], interpolation="nearest")
            cb = plt.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=6, colors="#6a7090")

    ax = styled_ax(gs[0, n_cols - 1], "RECONSTRUCTION", "axial slice")
    ax.imshow(recon_slices["axial"], cmap="gray")

    for i in range(nb_levels):
        key = f"codebook_{i}"
        if key in hooks and hooks[key].output is not None:
            feat = hooks[key].output
            hm = quant_to_heatmap_3d(feat)
            ax = styled_ax(gs[1, i + 1], f"CODEBOOK {i}",
                           f"quantized · {feat.shape[2]}×{feat.shape[3]}×{feat.shape[4]}")
            im = ax.imshow(hm, cmap=cmaps[i % len(cmaps)], interpolation="nearest")
            cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=6, colors="#6a7090")

    ax = fig.add_subplot(gs[1, 0], facecolor="#0e1118")
    ax.set_title("ACTIVATION DISTRIBUTION\nper encoder level",
                 color="#9aa8c8", fontsize=8, fontfamily="monospace", loc="left", pad=6)
    level_colors = ["#73e8b5", "#7eb8f7", "#f7a87e", "#e87373"]
    for i in range(nb_levels):
        key = f"encoder_{i}"
        if key in hooks and hooks[key].output is not None:
            vals = hooks[key].output.detach().cpu().float().flatten().numpy()
            ax.hist(vals, bins=80, alpha=0.65, color=level_colors[i % len(level_colors)],
                    label=f"level {i}", histtype="stepfilled", linewidth=0.5,
                    edgecolor=level_colors[i % len(level_colors)])
    ax.legend(fontsize=7, facecolor="#161a26", edgecolor="#2a3050",
              labelcolor="#9aa8c8", framealpha=0.8)
    ax.set_facecolor("#0e1118")
    ax.tick_params(colors="#4a5070", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3050")

    ax = styled_ax(gs[1, n_cols - 1], "RESIDUAL |x - x̂|", "axial slice")
    if input_slices is not None and recon_slices is not None:
        diff = np.abs(input_slices["axial"] - recon_slices["axial"])
        im = ax.imshow(diff, cmap="magma", vmin=0, vmax=0.3, interpolation="bilinear")
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors="#6a7090")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Enhanced feature map export + activation statistics
# ──────────────────────────────────────────────────────────────────────────────


def compute_activation_stats(hooks, nb_levels):
    """Compute per-channel activation statistics for each encoder level.

    Returns:
        dict mapping level index to dict of numpy arrays:
            {0: {"mean": (C,), "std": (C,), "sparsity": (C,), "l2_norm": (C,)}, ...}
    """
    stats = {}
    for i in range(nb_levels):
        key = f"encoder_{i}"
        if key not in hooks or hooks[key].output is None:
            continue
        feat = hooks[key].output.detach().cpu().float()  # (1, C, D, H, W)
        feat_flat = feat[0].reshape(feat.shape[1], -1)  # (C, N)

        stats[i] = {
            "mean": feat_flat.mean(dim=1).numpy(),
            "std": feat_flat.std(dim=1).numpy(),
            "sparsity": (feat_flat <= 0).float().mean(dim=1).numpy(),
            "l2_norm": feat_flat.norm(dim=1).numpy(),
        }
    return stats


def save_feature_maps(hooks, nb_levels, save_dir, save_nifti=False, spacing=2.0):
    """Save per-level feature maps as individual PNG figures and optionally NIfTI files.

    Each PNG contains a channel grid on the left and per-channel statistics bars on
    the right (mean activation, sparsity).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_activation_stats(hooks, nb_levels)

    for i in range(nb_levels):
        key = f"encoder_{i}"
        if key not in hooks or hooks[key].output is None:
            continue

        feat = hooks[key].output  # (1, C, D, H, W)
        C = feat.shape[1]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0e1118",
                                 gridspec_kw={"width_ratios": [3, 1]})

        # Left: channel grid
        grid = to_grid_3d(feat, max_channels=min(C, 64))
        axes[0].imshow(grid, cmap="viridis", interpolation="nearest")
        axes[0].set_title(f"Encoder {i} — {C} channels, shape {list(feat.shape[2:])}",
                          color="#c8d0e8", fontsize=10, fontfamily="monospace")
        axes[0].axis("off")

        # Right: per-channel sparsity bar chart
        if i in stats:
            channels = np.arange(C)
            axes[1].barh(channels, stats[i]["sparsity"], color="#7eb8f7", alpha=0.7)
            axes[1].set_xlabel("Sparsity (frac <= 0)", color="#9aa8c8", fontsize=8)
            axes[1].set_ylabel("Channel", color="#9aa8c8", fontsize=8)
            axes[1].set_title("Per-channel sparsity", color="#c8d0e8", fontsize=9,
                              fontfamily="monospace")
            axes[1].set_facecolor("#0e1118")
            axes[1].tick_params(colors="#4a5070", labelsize=6)
            axes[1].invert_yaxis()
            for spine in axes[1].spines.values():
                spine.set_edgecolor("#2a3050")

        fig.tight_layout()
        fig.savefig(save_dir / f"encoder_{i}_features.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved encoder {i} features → {save_dir / f'encoder_{i}_features.png'}")

        # NIfTI export
        if save_nifti:
            import nibabel as nib
            vol = feat[0].detach().cpu().numpy()  # (C, D, H, W)
            vol = vol.transpose(1, 2, 3, 0)  # (D, H, W, C) — 4D NIfTI convention
            affine = np.diag([spacing, spacing, spacing, 1.0])
            nib.save(nib.Nifti1Image(vol, affine=affine),
                     str(save_dir / f"encoder_{i}_features.nii.gz"))
            print(f"Saved encoder {i} NIfTI → {save_dir / f'encoder_{i}_features.nii.gz'}")

    # Print summary table
    if stats:
        print(f"\n{'Level':<7} {'Channels':<10} {'Mean act':<12} {'Mean sparsity':<15} {'Mean L2':<12}")
        print("-" * 56)
        for i in sorted(stats):
            s = stats[i]
            print(f"{i:<7} {len(s['mean']):<10} {s['mean'].mean():<12.4f} "
                  f"{s['sparsity'].mean():<15.3f} {s['l2_norm'].mean():<12.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# Analytical receptive field
# ──────────────────────────────────────────────────────────────────────────────


def _extract_conv_layers(module):
    """Recursively extract all Conv3d layers from a module in forward-pass order.

    Handles the model's module hierarchy:
        Encoder.layers (Sequential) → Conv3d / Sequential(Conv3d+BN+ReLU) / ResidualStack
        ResidualStack.stack (Sequential) → ReZero blocks
        ReZero.layers (Sequential) → Conv3d layers
    """
    convs = []
    if isinstance(module, nn.Conv3d):
        convs.append(module)
    elif isinstance(module, nn.Sequential):
        for child in module:
            convs.extend(_extract_conv_layers(child))
    elif hasattr(module, "stack"):
        # ResidualStack: self.stack is nn.Sequential of ReZero blocks
        convs.extend(_extract_conv_layers(module.stack))
    elif hasattr(module, "layers") and hasattr(module, "alpha"):
        # ReZero: self.layers is nn.Sequential, self.alpha is the scaling param
        convs.extend(_extract_conv_layers(module.layers))
    elif hasattr(module, "layers"):
        # Generic module with .layers attribute (e.g. Encoder)
        convs.extend(_extract_conv_layers(module.layers))
    else:
        for child in module.children():
            convs.extend(_extract_conv_layers(child))
    return convs


def compute_analytical_rf(model):
    """Compute the theoretical receptive field at each encoder level output.

    Traces through all Conv3d layers in the encoder chain, accumulating
    RF size and stride (jump) using:
        RF_new = RF_prev + (kernel_size - 1) * jump_prev
        jump_new = jump_prev * stride

    Returns:
        dict with keys:
            "per_level": list of dicts with rf_size, total_stride
            "layer_trace": list of dicts documenting each layer's contribution
    """
    rf = 1
    jump = 1
    layer_trace = []
    per_level = []

    for level_idx, encoder in enumerate(model.encoders):
        conv_layers = _extract_conv_layers(encoder)
        for conv in conv_layers:
            k = conv.kernel_size[0]
            s = conv.stride[0]
            rf = rf + (k - 1) * jump
            layer_trace.append({
                "level": level_idx,
                "type": type(conv).__name__,
                "kernel": k,
                "stride": s,
                "rf_after": rf,
                "jump_after": jump * s,
            })
            jump = jump * s

        per_level.append({
            "rf_size": rf,
            "total_stride": jump,
        })

    return {"per_level": per_level, "layer_trace": layer_trace}


def print_rf_summary(rf_result, spacing=2.0):
    """Print a formatted receptive field summary table."""
    print("\nAnalytical Receptive Field")
    print("=" * 60)
    print(f"{'Level':<7} {'RF (vox)':<12} {'RF (mm)':<12} {'Stride':<10} {'Conv layers':<12}")
    print("-" * 60)

    for i, level in enumerate(rf_result["per_level"]):
        n_layers = sum(1 for t in rf_result["layer_trace"] if t["level"] == i)
        print(f"{i:<7} {level['rf_size']:<12} {level['rf_size'] * spacing:<12.1f} "
              f"{level['total_stride']:<10} {n_layers:<12}")

    print()
    # Detailed layer trace
    print("Layer-by-layer trace:")
    print(f"  {'Level':<6} {'Type':<12} {'Kernel':<8} {'Stride':<8} {'RF after':<10} {'Jump after':<10}")
    print("  " + "-" * 54)
    for t in rf_result["layer_trace"]:
        print(f"  {t['level']:<6} {t['type']:<12} {t['kernel']:<8} {t['stride']:<8} "
              f"{t['rf_after']:<10} {t['jump_after']:<10}")


def plot_analytical_rf(rf_result, input_slices, spacing=2.0, save_path=None):
    """Visualize analytical receptive fields as overlaid rectangles on brain slices.

    Draws concentric colored rectangles centered on each slice, one per encoder
    level, sized proportionally to the computed RF.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0e1118")
    fig.suptitle("Analytical Receptive Fields",
                 color="#c8d0e8", fontsize=14, fontweight="bold",
                 fontfamily="monospace", y=0.97)

    view_names = ["sagittal", "coronal", "axial"]
    level_colors = ["#73e8b5", "#7eb8f7", "#f7a87e", "#e87373"]

    for ax_idx, view in enumerate(view_names):
        ax = axes[ax_idx]
        ax.imshow(input_slices[view], cmap="gray")
        H, W = input_slices[view].shape
        center_y, center_x = H / 2, W / 2

        for level_idx, level_info in enumerate(rf_result["per_level"]):
            rf_vox = level_info["rf_size"]
            half_rf = rf_vox / 2
            color = level_colors[level_idx % len(level_colors)]

            # Clamp rectangle to image bounds
            rect_w = min(rf_vox, W)
            rect_h = min(rf_vox, H)
            rect = plt.Rectangle(
                (center_x - rect_w / 2, center_y - rect_h / 2),
                rect_w, rect_h,
                linewidth=2, edgecolor=color, facecolor="none", linestyle="--",
                label=f"Level {level_idx}: {rf_vox} vox ({rf_vox * spacing:.0f}mm)",
            )
            ax.add_patch(rect)

        ax.set_title(view.capitalize(), color="#9aa8c8", fontsize=11, fontfamily="monospace")
        ax.axis("off")
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper left", facecolor="#161a26",
                      edgecolor="#2a3050", labelcolor="#9aa8c8", framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved analytical RF → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Empirical (gradient-based) receptive field
# ──────────────────────────────────────────────────────────────────────────────


def compute_empirical_rf(model, input_tensor, level=0, position=None, channel=None):
    """Compute the empirical receptive field via gradient backpropagation.

    Enables gradients on the input, runs a forward pass, selects a single neuron
    at the center of the chosen encoder level, and backpropagates. The gradient
    magnitude on the input reveals the effective receptive field.

    Args:
        model: VQVAE model (will be set to eval mode)
        input_tensor: (1, 1, D, H, W) input tensor
        level: encoder level index (0 to nb_levels-1)
        position: (d, h, w) spatial position in the encoder feature map (None = center)
        channel: channel index to probe (None = sum across all channels)

    Returns:
        dict with gradient_map, gradient_map_normalized, position_used, feature_shape
    """
    model.eval()

    # Freeze model parameters — we only need gradients w.r.t. the input
    param_requires_grad = {}
    for name, p in model.named_parameters():
        param_requires_grad[name] = p.requires_grad
        p.requires_grad_(False)

    try:
        inp = input_tensor.clone().detach().requires_grad_(True)

        hook = FeatureHook()
        handle = hook.register(model.encoders[level])

        # Forward pass with gradients enabled only for input
        _ = model(inp)

        feat = hook.output  # (1, C, D', H', W')
        C, D, H, W = feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4]

        if position is None:
            d, h, w = D // 2, H // 2, W // 2
        else:
            d, h, w = position

        if channel is not None:
            target = feat[0, channel, d, h, w]
        else:
            target = feat[0, :, d, h, w].sum()

        target.backward()

        grad = inp.grad[0, 0].abs().detach().cpu().numpy()  # (D, H, W)
        grad_norm = grad / (grad.max() + 1e-8)

        handle.remove()
    finally:
        # Restore original requires_grad state
        for name, p in model.named_parameters():
            p.requires_grad_(param_requires_grad[name])

    return {
        "gradient_map": grad,
        "gradient_map_normalized": grad_norm,
        "position_used": (d, h, w),
        "feature_shape": (C, D, H, W),
    }


def compute_all_empirical_rfs(model, input_tensor):
    """Compute empirical RF for all encoder levels."""
    results = {}
    for level in range(model.nb_levels):
        results[level] = compute_empirical_rf(model, input_tensor, level=level)
    return results


def plot_empirical_rf(erf_result, input_slices, level, save_path=None):
    """Overlay the empirical RF gradient map on brain slices.

    Top row: anatomical + gradient overlay. Bottom row: gradient-only heatmaps.
    """
    grad = erf_result["gradient_map_normalized"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="#0e1118")
    fig.suptitle(
        f"Empirical Receptive Field — Encoder Level {level}\n"
        f"Feature shape: {erf_result['feature_shape']}, "
        f"Position: {erf_result['position_used']}",
        color="#c8d0e8", fontsize=13, fontweight="bold",
        fontfamily="monospace", y=0.97,
    )

    slices_spec = [
        ("sagittal", lambda v: v[v.shape[0] // 2, :, :]),
        ("coronal", lambda v: v[:, v.shape[1] // 2, :]),
        ("axial", lambda v: v[:, :, v.shape[2] // 2]),
    ]

    for col, (name, slicer) in enumerate(slices_spec):
        grad_slice = slicer(grad)

        # Top row: overlay
        ax = axes[0, col]
        ax.imshow(input_slices[name], cmap="gray")
        ax.imshow(grad_slice, cmap="hot", alpha=0.5, vmin=0, vmax=max(grad_slice.max(), 1e-8))
        ax.set_title(f"{name.capitalize()} — overlay", color="#9aa8c8", fontsize=10,
                     fontfamily="monospace")
        ax.axis("off")

        # Bottom row: gradient only
        ax = axes[1, col]
        im = ax.imshow(grad_slice, cmap="hot", vmin=0, vmax=max(grad_slice.max(), 1e-8))
        ax.set_title(f"{name.capitalize()} — gradient magnitude", color="#9aa8c8",
                     fontsize=10, fontfamily="monospace")
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors="#6a7090")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved empirical RF → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_all_empirical_rfs(erf_results, input_slices, save_path=None):
    """Combined comparison of empirical RFs across all encoder levels (axial view)."""
    nb_levels = len(erf_results)
    fig, axes = plt.subplots(2, nb_levels, figsize=(6 * nb_levels, 12), facecolor="#0e1118")
    fig.suptitle("Empirical Receptive Fields — All Levels (axial)",
                 color="#c8d0e8", fontsize=14, fontweight="bold",
                 fontfamily="monospace", y=0.97)

    if nb_levels == 1:
        axes = axes.reshape(2, 1)

    for level in range(nb_levels):
        grad = erf_results[level]["gradient_map_normalized"]
        grad_slice = grad[:, :, grad.shape[2] // 2]

        # Top: overlay
        ax = axes[0, level]
        ax.imshow(input_slices["axial"], cmap="gray")
        ax.imshow(grad_slice, cmap="hot", alpha=0.5, vmin=0, vmax=max(grad_slice.max(), 1e-8))
        ax.set_title(f"Level {level} — overlay\nshape: {erf_results[level]['feature_shape']}",
                     color="#9aa8c8", fontsize=9, fontfamily="monospace")
        ax.axis("off")

        # Bottom: gradient only
        ax = axes[1, level]
        im = ax.imshow(grad_slice, cmap="hot", vmin=0, vmax=max(grad_slice.max(), 1e-8))
        ax.set_title(f"Level {level} — gradient", color="#9aa8c8", fontsize=9,
                     fontfamily="monospace")
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors="#6a7090")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved all empirical RFs → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_rf_comparison(rf_analytical, erf_results, input_slices, spacing=2.0, save_path=None):
    """Side-by-side comparison of analytical and empirical receptive fields.

    Left column: analytical RF rectangles on axial slice.
    Right columns: one empirical RF heatmap per level.
    """
    nb_levels = len(erf_results)
    fig, axes = plt.subplots(1, nb_levels + 1, figsize=(6 * (nb_levels + 1), 6),
                             facecolor="#0e1118")
    fig.suptitle("Receptive Field Comparison — Analytical vs Empirical",
                 color="#c8d0e8", fontsize=14, fontweight="bold",
                 fontfamily="monospace", y=0.97)

    level_colors = ["#73e8b5", "#7eb8f7", "#f7a87e", "#e87373"]

    # Analytical RF
    ax = axes[0]
    ax.imshow(input_slices["axial"], cmap="gray")
    H, W = input_slices["axial"].shape
    cy, cx = H / 2, W / 2
    for i, level_info in enumerate(rf_analytical["per_level"]):
        rf_vox = level_info["rf_size"]
        color = level_colors[i % len(level_colors)]
        rect_w, rect_h = min(rf_vox, W), min(rf_vox, H)
        rect = plt.Rectangle((cx - rect_w / 2, cy - rect_h / 2), rect_w, rect_h,
                              linewidth=2, edgecolor=color, facecolor="none", linestyle="--",
                              label=f"L{i}: {rf_vox}vox ({rf_vox * spacing:.0f}mm)")
        ax.add_patch(rect)
    ax.legend(fontsize=7, loc="upper left", facecolor="#161a26",
              edgecolor="#2a3050", labelcolor="#9aa8c8", framealpha=0.9)
    ax.set_title("Analytical RF", color="#9aa8c8", fontsize=11, fontfamily="monospace")
    ax.axis("off")

    # Empirical RFs
    for level in range(nb_levels):
        ax = axes[level + 1]
        grad = erf_results[level]["gradient_map_normalized"]
        grad_slice = grad[:, :, grad.shape[2] // 2]
        ax.imshow(input_slices["axial"], cmap="gray")
        ax.imshow(grad_slice, cmap="hot", alpha=0.5, vmin=0, vmax=max(grad_slice.max(), 1e-8))
        ax.set_title(f"Empirical RF — Level {level}", color="#9aa8c8", fontsize=11,
                     fontfamily="monospace")
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved RF comparison → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="VQ-VAE-2 evaluation and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes:
  evaluate       Compute validation loss + codebook utilization on a directory of NIfTI images
  visualize      Feature map inspector (encoder grids, codebook heatmaps, reconstruction)
  features       Export per-level feature maps as PNG + optional NIfTI
  rf-analytical  Compute theoretical receptive field from model architecture
  rf-empirical   Compute empirical receptive field via gradient backpropagation
  rf-both        Combined analytical + empirical RF comparison

examples:
  python eval.py --mode evaluate --checkpoint ckpt.pt --dataroot /path/to/data
  python eval.py --mode visualize --checkpoint ckpt.pt --image brain.nii.gz --save features.png
  python eval.py --mode visualize --checkpoint ckpt.pt --dataroot /path/to/data --save features.png
  python eval.py --mode rf-both --checkpoint ckpt.pt --dataroot /path/to/data --seed 42 --save rf.png
""",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to model .pt checkpoint")
    parser.add_argument("--device", default="cpu", help="cuda / mps / cpu")
    parser.add_argument("--spacing", type=float, default=2.0, help="Isotropic voxel spacing in mm")
    parser.add_argument(
        "--mode", default="visualize",
        choices=["evaluate", "visualize", "features", "rf-analytical", "rf-empirical", "rf-both"],
        help="Analysis mode (default: visualize)",
    )
    # Data source
    parser.add_argument("--dataroot", default=None,
                        help="Data directory (for evaluate mode, or to randomly sample an image)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluate mode")
    # Visualization modes
    parser.add_argument("--image", default=None,
                        help="Path to input NIfTI image (if omitted, randomly samples from --dataroot)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible image sampling")
    parser.add_argument("--save", default=None, help="Save figure to this path instead of showing")
    parser.add_argument("--save-dir", default=None, help="Directory for saving feature maps")
    parser.add_argument("--save-nifti", action="store_true", help="Export feature maps as NIfTI")
    # RF options
    parser.add_argument("--rf-level", type=int, default=None,
                        help="Encoder level for empirical RF (None = all levels)")
    parser.add_argument("--rf-position", type=int, nargs=3, default=None,
                        metavar=("D", "H", "W"),
                        help="Spatial position in feature map for empirical RF probe")
    parser.add_argument("--rf-channel", type=int, default=None,
                        help="Channel index for empirical RF (None = average all)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # ── evaluate: runs loss on a directory of images ─────────────────────────
    if args.mode == "evaluate":
        if not args.checkpoint:
            parser.error("--checkpoint is required for evaluate mode")
        if not args.dataroot:
            parser.error("--dataroot is required for evaluate mode")
        run_evaluate(args.checkpoint, args.dataroot, device,
                     spacing=args.spacing, batch_size=args.batch_size)
        return

    # Load model
    print("Loading model…")
    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint, device)
    else:
        print("No checkpoint given — using random weights (demo mode).")
        model = VQVAE().to(device)
        model.eval()
    nb_levels = model.nb_levels

    # ── rf-analytical: no image needed ────────────────────────────────────────
    if args.mode == "rf-analytical":
        rf_result = compute_analytical_rf(model)
        print_rf_summary(rf_result, spacing=args.spacing)
        img_path = args.image
        if not img_path and args.dataroot:
            img_path = _sample_random_image(args.dataroot, seed=args.seed)
        if img_path:
            _, input_slices = _load_volume(img_path, spacing=args.spacing)
            plot_analytical_rf(rf_result, input_slices, spacing=args.spacing,
                               save_path=args.save)
        return

    # All other modes require an image — resolve from --image or --dataroot
    image_path = args.image
    if not image_path:
        if args.dataroot:
            image_path = _sample_random_image(args.dataroot, seed=args.seed)
        else:
            parser.error("--image or --dataroot is required for this mode")

    print(f"Loading image: {image_path}")
    img_tensor, input_slices = _load_volume(image_path, spacing=args.spacing)
    img_tensor = img_tensor.to(device)

    if args.mode == "visualize":
        hooks, handles = attach_hooks(model)
        with torch.no_grad():
            recon, diffs, *_ = model(img_tensor)
        recon_slices = _tensor_to_display(recon)
        plot_features(input_slices, hooks, recon_slices, nb_levels, save_path=args.save)
        for h in handles:
            h.remove()

    elif args.mode == "features":
        hooks, handles = attach_hooks(model)
        with torch.no_grad():
            model(img_tensor)
        save_dir = args.save_dir or "feature_maps"
        save_feature_maps(hooks, nb_levels, save_dir,
                          save_nifti=args.save_nifti, spacing=args.spacing)
        for h in handles:
            h.remove()

    elif args.mode == "rf-empirical":
        if args.rf_level is not None:
            erf = compute_empirical_rf(
                model, img_tensor, level=args.rf_level,
                position=tuple(args.rf_position) if args.rf_position else None,
                channel=args.rf_channel,
            )
            plot_empirical_rf(erf, input_slices, args.rf_level, save_path=args.save)
        else:
            erf_results = compute_all_empirical_rfs(model, img_tensor)
            plot_all_empirical_rfs(erf_results, input_slices, save_path=args.save)

    elif args.mode == "rf-both":
        rf_analytical = compute_analytical_rf(model)
        print_rf_summary(rf_analytical, spacing=args.spacing)
        erf_results = compute_all_empirical_rfs(model, img_tensor)
        plot_rf_comparison(rf_analytical, erf_results, input_slices,
                           spacing=args.spacing, save_path=args.save)


if __name__ == "__main__":
    main()
