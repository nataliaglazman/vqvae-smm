# TO DO:
# - Add feature maps 
# - Add classifiers for downstream tasks (e.g. AD vs CN, MCI vs CN, etc.) and evaluate on those tasks as well
# - Add Dan's methods for segmentations or whatever

import torch
from torch.utils.data import DataLoader
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from pathlib import Path

# MONAI imports for 3D medical imaging
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    Resized,
    ToTensord,
)
from monai.data import Dataset, DataLoader as MonaiDataLoader

from vqvae2 import VQVAE
from loss import BaselineLoss


def get_transforms(spacing=(1.0, 1.0, 1.0), spatial_size=(128, 128, 128)):
    """Get MONAI transforms for 3D medical image preprocessing."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=5),
        Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
        ToTensord(keys=["image"]),
    ])


def evaluate(args):
    device = torch.device(args.device)

    # Prepare data list for MONAI
    data_dir = Path(args.dataroot)
    image_files = list(data_dir.glob("**/*.nii.gz")) + list(data_dir.glob("**/*.nii"))
    data_dicts = [{"image": str(f)} for f in image_files]

    transforms = get_transforms(
        spacing=tuple(args.spacing) if hasattr(args, 'spacing') else (1.0, 1.0, 1.0),
        spatial_size=tuple(args.spatial_size) if hasattr(args, 'spatial_size') else (128, 128, 128),
    )
    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = MonaiDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = VQVAE().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {args.checkpoint}")

    recon_loss_fn = BaselineLoss().to(device)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            recon_images, diffs, encoder_features, decoder_outputs, id_outputs = model(images)

            # Total loss
            loss = recon_loss_fn(recon_images, images)
            vq_loss = sum(diffs) * args.vq_loss_weight
            loss += vq_loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average Validation Loss: {avg_loss:.4f}")






class FeatureHook:
    """Captures the output tensor of any nn.Module via a forward hook."""
    def __init__(self):
        self.output = None

    def hook_fn(self, module, input, output):
        # output may be a tuple (e.g. quantizer returns (quant, loss, indices))
        self.output = output[0] if isinstance(output, tuple) else output

    def register(self, module):
        return module.register_forward_hook(self.hook_fn)


def attach_hooks(model):
    """
    Attach hooks to all encoder levels and codebooks for the 3D VQ-VAE-2.
    Returns (hooks dict, handle list) — call handle.remove() to clean up.

    Adapted for vqvae2.py which uses ModuleLists for encoders/codebooks.
    """
    hooks = {}
    handles = []

    # Attach hooks to all encoder levels
    for i, encoder in enumerate(model.encoders):
        hook = FeatureHook()
        hooks[f"encoder_{i}"] = hook
        handles.append(hook.register(encoder))

    # Attach hooks to all codebook levels
    for i, codebook in enumerate(model.codebooks):
        hook = FeatureHook()
        hooks[f"codebook_{i}"] = hook
        handles.append(hook.register(codebook))

    # Attach hooks to all decoder levels
    for i, decoder in enumerate(model.decoders):
        hook = FeatureHook()
        hooks[f"decoder_{i}"] = hook
        handles.append(hook.register(decoder))

    return hooks, handles



def to_grid_3d(feat: torch.Tensor, max_channels: int = 16, slice_axis: int = 2) -> np.ndarray:
    """
    feat: (1, C, D, H, W)  →  grid image (grid_H, grid_W) normalised to [0,1]
    Takes the middle slice along slice_axis (default: axial/z-axis).
    """
    feat = feat.detach().cpu().float()
    C = min(feat.shape[1], max_channels)
    feat = feat[0, :C]  # (C, D, H, W)

    # Take middle slice along the specified axis
    if slice_axis == 0:  # Sagittal
        mid = feat.shape[1] // 2
        feat = feat[:, mid, :, :]  # (C, H, W)
    elif slice_axis == 1:  # Coronal
        mid = feat.shape[2] // 2
        feat = feat[:, :, mid, :]  # (C, D, W)
    else:  # Axial (default)
        mid = feat.shape[3] // 2
        feat = feat[:, :, :, mid]  # (C, D, H)

    # Per-channel min-max normalise
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
    q = quant_feat.detach().cpu().float()[0]  # (C, D, H, W)

    # Take middle slice
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
    vol = vol.detach().cpu().float().squeeze()  # (D, H, W)
    if vol.ndim == 4:
        vol = vol[0]  # Remove channel dim if present

    slices = {
        "sagittal": vol[vol.shape[0] // 2, :, :].numpy(),
        "coronal": vol[:, vol.shape[1] // 2, :].numpy(),
        "axial": vol[:, :, vol.shape[2] // 2].numpy(),
    }

    # Normalise each slice to [0, 1]
    for k in slices:
        s = slices[k]
        slices[k] = (s - s.min()) / (s.max() - s.min() + 1e-8)

    return slices


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Main plotting routine
# ──────────────────────────────────────────────────────────────────────────────

def plot_features(input_display, hooks, recon, save_path=None):
    fig = plt.figure(figsize=(18, 10), facecolor="#0e1118")
    fig.suptitle("VQ-VAE-2  ·  Feature Map Inspector",
                 color="#c8d0e8", fontsize=14, fontweight="bold",
                 fontfamily="monospace", y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.04, right=0.97, top=0.92, bottom=0.05)

    ax_cfg = dict(facecolor="#0e1118")

    def styled_ax(gs_pos, title, subtitle=""):
        ax = fig.add_subplot(gs_pos, **ax_cfg)
        ax.set_title(f"{title}\n{subtitle}", color="#9aa8c8", fontsize=8,
                     fontfamily="monospace", loc="left", pad=6)
        ax.axis("off")
        return ax

    # ── Input ────────────────────────────────────────────────────────────────
    ax = styled_ax(gs[0, 0], "INPUT", "original image")
    ax.imshow(input_display)

    # ── Bottom encoder feature maps ──────────────────────────────────────────
    if hooks["enc_bottom"].output is not None:
        grid_b = to_grid(hooks["enc_bottom"].output, max_channels=16)
        ax = styled_ax(gs[0, 1], "BOTTOM ENCODER  z_b",
                       f"feat maps  ·  {hooks['enc_bottom'].output.shape[1]} ch")
        ax.imshow(grid_b, cmap="viridis", interpolation="nearest")
        cb = plt.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors="#6a7090")

    # ── Top encoder feature maps ─────────────────────────────────────────────
    if hooks["enc_top"].output is not None:
        grid_t = to_grid(hooks["enc_top"].output, max_channels=16)
        ax = styled_ax(gs[0, 2], "TOP ENCODER  z_t",
                       f"feat maps  ·  {hooks['enc_top'].output.shape[1]} ch")
        ax.imshow(grid_t, cmap="plasma", interpolation="nearest")
        cb = plt.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors="#6a7090")

    # ── Reconstruction ───────────────────────────────────────────────────────
    ax = styled_ax(gs[0, 3], "RECONSTRUCTION  x̂", "decoded output")
    ax.imshow(recon)

    # ── Bottom quantized codes ───────────────────────────────────────────────
    if hooks["quant_bottom"].output is not None:
        hm_b = quant_to_heatmap(hooks["quant_bottom"].output)
        ax = styled_ax(gs[1, 1], "BOTTOM CODEBOOK",
                       f"quantized  ·  {hooks['quant_bottom'].output.shape[-2]}×{hooks['quant_bottom'].output.shape[-1]}")
        im = ax.imshow(hm_b, cmap="inferno", interpolation="nearest")
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors="#6a7090")

    # ── Top quantized codes ──────────────────────────────────────────────────
    if hooks["quant_top"].output is not None:
        hm_t = quant_to_heatmap(hooks["quant_top"].output)
        ax = styled_ax(gs[1, 2], "TOP CODEBOOK",
                       f"quantized  ·  {hooks['quant_top'].output.shape[-2]}×{hooks['quant_top'].output.shape[-1]}")
        im = ax.imshow(hm_t, cmap="coolwarm", interpolation="nearest")
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors="#6a7090")

    # ── Per-channel activation histogram ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0], facecolor="#0e1118")
    ax.set_title("ACTIVATION DISTRIBUTION\nbottom vs top encoder",
                 color="#9aa8c8", fontsize=8, fontfamily="monospace", loc="left", pad=6)
    for feat, label, color in [
        (hooks["enc_bottom"].output, "bottom", "#73e8b5"),
        (hooks["enc_top"].output,    "top",    "#7eb8f7"),
    ]:
        if feat is not None:
            vals = feat.detach().cpu().float().flatten().numpy()
            ax.hist(vals, bins=80, alpha=0.65, color=color, label=label,
                    histtype="stepfilled", linewidth=0.5, edgecolor=color)
    ax.legend(fontsize=7, facecolor="#161a26", edgecolor="#2a3050",
              labelcolor="#9aa8c8", framealpha=0.8)
    ax.set_facecolor("#0e1118")
    ax.tick_params(colors="#4a5070", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3050")

    # ── Difference map ───────────────────────────────────────────────────────
    ax = styled_ax(gs[1, 3], "RESIDUAL  |x - x̂|", "per-channel mean")
    if recon is not None and input_display is not None:
        inp_f  = input_display.astype(np.float32) / 255.0
        rec_f  = recon.astype(np.float32) / 255.0
        diff   = np.abs(inp_f - rec_f).mean(-1)
        im = ax.imshow(diff, cmap="magma", vmin=0, vmax=0.3, interpolation="bilinear")
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors="#6a7090")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()

def load_model(checkpoint: str | None, device: torch.device):
    """
    Returns a VQ-VAE-2 model.
    Replace the body of this function with your own model loading logic.
    """
    try:
        from vqvae2 import VQVAE  # rosinality/vq-vae-2-pytorch
        model = VQVAE()
        if checkpoint:
            state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded weights from {checkpoint}")
        else:
            print("No checkpoint given — using random weights (demo mode).")
        model.eval().to(device)
        return model
    except ImportError:
        raise ImportError(
            "Could not import vqvae. Install it with:\n"
            "  pip install git+https://github.com/rosinality/vq-vae-2-pytorch\n"
            "or point load_model() at your own model class."
        )

def main():
    parser = argparse.ArgumentParser(description="VQ-VAE-2 feature map visualizer")
    parser.add_argument("--image",      required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default=None,  help="Path to model .pt checkpoint")
    parser.add_argument("--save",       default=None, help="Save figure to this path instead of showing")
    parser.add_argument("--device",     default="cpu", help="cuda / mps / cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading model…")
    model = load_model(args.checkpoint, device)

    print(f"Loading image: {args.image}")
    img_tensor, img_display = load_image(args.image, args.size)
    img_tensor = img_tensor.to(device)

    print("Attaching hooks…")
    hooks, handles = attach_hooks(model)

    print("Running forward pass…")
    with torch.no_grad():
        recon, _ = model(img_tensor)          # adjust if your model returns differently
    recon_display = tensor_to_image(recon)

    print("Plotting…")
    plot_features(img_display, hooks, recon_display, save_path=args.save)

    # Clean up hooks
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()