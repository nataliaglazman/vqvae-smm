"""Training script for VQ-VAE-2 on 3D brain MRI."""

import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset, CacheDataset

from config import parse_args
from helper import get_device, get_parameter_count
from loss import BaselineLoss, CliffLoss
from utils import (
    TBSummaryTypes, build_cached_dataset, transforms, load_items, save_decoded_images,
)
from vqvae2 import VQVAE, CodeLayer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CSV Logger ────────────────────────────────────────────────────────────────


class CSVLogger:
    """Append-friendly CSV logger that writes one row per logged step.

    On resume, new rows are simply appended to the existing file so that no
    history is lost.  If the file doesn't exist, a header row is written first.
    """

    TRAIN_COLUMNS = [
        "step", "epoch", "elapsed_s", "lr",
        "total_loss", "pixel_loss", "fft_loss", "perceptual_loss",
    ]
    # vq_loss_0 … vq_loss_{N-1} are added dynamically based on nb_levels

    VAL_COLUMNS = ["step", "epoch", "elapsed_s", "val_loss"]

    CLIFF_COLUMNS = [
        "Loss-Cliff-Univariate", "Loss-Cliff-Bivariate",
        "Loss-Cliff-AntiCollapse", "Loss-Cliff-Total",
    ]

    def __init__(self, out_dir: Path, nb_levels: int, use_cliff: bool = False):
        self.out_dir = out_dir
        self.nb_levels = nb_levels

        # Build full train header (with per-level VQ columns)
        self.train_columns = list(self.TRAIN_COLUMNS)
        for i in range(nb_levels):
            self.train_columns.append(f"vq_loss_{i}")
        if use_cliff:
            self.train_columns.extend(self.CLIFF_COLUMNS)

        self.train_path = out_dir / "train_losses.csv"
        self.val_path = out_dir / "val_losses.csv"

        # Write headers only if the files don't exist yet
        self._ensure_header(self.train_path, self.train_columns)
        self._ensure_header(self.val_path, self.VAL_COLUMNS)

    @staticmethod
    def _ensure_header(path: Path, columns: list):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(columns)

    def log_train(self, row: dict):
        """Append a training row.  Missing keys are written as empty strings."""
        with open(self.train_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.train_columns, extrasaction="ignore")
            w.writerow(row)

    def log_val(self, row: dict):
        """Append a validation row."""
        with open(self.val_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.VAL_COLUMNS, extrasaction="ignore")
            w.writerow(row)


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, best_val_loss, args):
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )
    log.info(f"Checkpoint saved -> {path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    step = ckpt["step"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    log.info(f"Resumed from step {step} (best val loss: {best_val_loss:.4f})")
    return step, best_val_loss


def load_model_from_checkpoint(path, device="cpu"):
    """Recreate a VQVAE model from a training checkpoint (useful for eval/inference)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
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
    # Strip _orig_mod. prefix added by torch.compile
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Validation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, loader, device, amp_enabled, commitment_weight=0.25):
    model.eval()
    total, n = 0.0, 0
    first_sample = None
    first_recon = None
    # Run validation in eager mode regardless of whether torch.compile is active.
    # The inductor backend recompiles for eval-mode (different BatchNorm graph) and
    # that recompilation requires a C compiler — which may not be on PATH in cluster
    # environments.  Eager mode is fast enough for a validation pass.
    _model_eval = torch._dynamo.disable(model) if hasattr(torch, "_dynamo") else model
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        images = images.to(memory_format=torch.channels_last_3d)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            recon, diffs, *_ = _model_eval(images)
            # Use only cheap pixel (L1) loss for validation, masked if available.
            if "mask" in batch:
                mask = batch["mask"].to(device, non_blocking=True)
                loss = torch.nn.functional.l1_loss(recon * mask, images * mask, reduction="sum") / mask.sum().clamp(min=1e-5)
            else:
                loss = torch.nn.functional.l1_loss(recon, images)
            for d in diffs:
                loss = loss + d.float() * commitment_weight
        if first_sample is None:
            first_sample = batch
            first_recon = recon[0:1].detach().cpu()
        total += loss.item()
        n += 1
    return first_sample, first_recon, total / max(n, 1)


# ── Training ──────────────────────────────────────────────────────────────────


def train(args):
    device = get_device(args.no_cuda)
    amp_enabled = args.use_amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # TF32: ~3× faster float32 matmuls/convs on Ampere+ GPUs (A100, RTX 30xx/40xx).
        # Precision is reduced from 23 to 10 mantissa bits — negligible impact on
        # training quality for this model.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Output directory
    out_dir = Path(args.model_dir) / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dir = out_dir / "decoded_images"
    save_dir.mkdir(exist_ok=True)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    items = load_items(args.dataroot, args.csv_path, load_masks=getattr(args, "masks_from_disk", False))
    if not items:
        log.error("No data items found. Check --dataroot and --csv-path.")
        return

    train_transform, val_transform = transforms(
        spacing=args.image_spacing,
        crop_margin=args.crop_margin,
        asymmetric_aug=getattr(args, 'asymmetric_aug', False),
    )

    # Train / val split (stratified by class label)
    if args.val_size < 1:
        val_count = int(len(items) * args.val_size)
    else:
        val_count = int(args.val_size)
    val_count = max(1, min(val_count, len(items) // 2))

    rng = np.random.RandomState(args.seed)
    labels = np.array([it["label"] for it in items])
    unique_labels = np.unique(labels)
    train_items, val_items = [], []
    for lbl in unique_labels:
        lbl_indices = np.where(labels == lbl)[0]
        rng.shuffle(lbl_indices)
        n_val = max(1, int(len(lbl_indices) * val_count / len(items)))
        val_items.extend(items[i] for i in lbl_indices[:n_val])
        train_items.extend(items[i] for i in lbl_indices[n_val:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    # Build MONAI data dicts (CacheDataset expects list-of-dicts with file paths)
    train_dicts = [{"image": it["image"], "mask": it["mask"]} if "mask" in it else {"image": it["image"]} for it in train_items]
    val_dicts = [{"image": it["image"], "mask": it["mask"]} if "mask" in it else {"image": it["image"]} for it in val_items]

    # CacheDataset caches deterministic transforms in RAM (load once).
    # Random augmentation is applied on-the-fly by CachedAugDataset wrapper.
    # Standard Dataset (no cache) because train_transform includes random augmentations
    train_set = Dataset(data=train_dicts, transform=train_transform)
    val_set = CacheDataset(
        data=val_dicts, transform=val_transform,
        cache_rate=args.val_cache_rate, num_workers=args.workers,
    )
    log.info(f"Dataset split -- train: {len(train_set)}, val: {len(val_set)}")

    if len(train_set) < args.batch_size:
        log.error(
            f"Training set ({len(train_set)}) smaller than batch size ({args.batch_size}). "
            "Reduce --batch-size or add more data."
        )
        return

    loader_kwargs = dict(
        pin_memory=(device.type == "cuda" and not args.no_pin_memory),
        num_workers=args.workers,
        persistent_workers=(args.persistent_workers and args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VQVAE(
        in_channels=1,
        hidden_channels=args.vqvae_hidden_channels,
        res_channels=args.vqvae_res_channels,
        nb_levels=args.vqvae_nb_levels,
        embed_dim=args.vqvae_embed_dim,
        nb_entries=args.vqvae_nb_entries,
        scaling_rates=args.vqvae_scaling_rates,
        use_checkpoint=args.gradient_checkpointing,
        entropy_weight=args.entropy_weight,
    ).to(device)
    log.info(f"Parameters: {get_parameter_count(model):,}")

    # Channels-last memory layout: cuDNN picks faster conv kernels on modern GPUs.
    model = model.to(memory_format=torch.channels_last_3d)

    # torch.compile (PyTorch 2.0+): fuses ops, reduces kernel launches.
    if args.compile:
        log.info(f"Compiling model with torch.compile (backend={args.compile_backend}, first step will be slow)…")
        model = torch.compile(model, backend=args.compile_backend)

    # ── Loss / optimiser / scheduler ──────────────────────────────────────────
    loss_fn = BaselineLoss(commitment_weight=args.vq_commitment_weight).to(device)

    cliff_fn = None
    if args.use_cliff_loss:
        # Input dim = hidden_channels per encoder level × nb_levels
        cliff_in_dim = args.vqvae_hidden_channels * args.vqvae_nb_levels
        cliff_fn = CliffLoss(
            lambda_uni=args.cliff_lambda_uni,
            lambda_biv=args.cliff_lambda_biv,
            lambda_kl_uni=args.cliff_lambda_kl_uni,
            sigma=args.cliff_sigma,
            K=args.cliff_K,
            M=args.cliff_M,
            latent_dim=args.cliff_latent_dim,
            in_dim=cliff_in_dim,
        ).to(device)
        log.info(
            f"Cliff loss enabled (scale={args.scale_cliff_loss}, "
            f"projection {cliff_in_dim} -> {args.cliff_latent_dim})"
        )

    # Optimise model params + cliff projection (if present)
    optim_params = list(model.parameters())
    if cliff_fn is not None:
        optim_params += list(cliff_fn.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr)

    total_opt_steps = max(args.train_steps // args.gradient_accumulation_steps, 1)
    warmup_steps = min(args.warmup_steps, total_opt_steps // 5)  # cap at 20% of training
    cosine_steps = max(total_opt_steps - warmup_steps, 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    if args.resume_training and ckpt_latest.exists():
        start_step, best_val_loss = load_checkpoint(
            ckpt_latest, model, optimizer, scheduler, scaler, device,
        )

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out_dir / "tb")
    except ImportError:
        log.warning("tensorboard not installed -- logging to stdout only")

    # ── CSV Logger ────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(out_dir, nb_levels=args.vqvae_nb_levels, use_cliff=args.use_cliff_loss)
    log.info(f"CSV logs -> {csv_logger.train_path}, {csv_logger.val_path}")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step = start_step
    epoch = 0
    optimizer.zero_grad(set_to_none=True)

    log.info(f"Training: step {step} -> {args.train_steps}")
    t0 = time.time()

    while step < args.train_steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.train_steps:
                break

            images = batch["image"].to(device, non_blocking=True)
            images = images.to(memory_format=torch.channels_last_3d)

            # Optionally skip reconstruction for memory / codebook-only steps
            skip_recon = args.skip_recon_ratio > 0 and random.random() < args.skip_recon_ratio

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                recon, diffs, _enc_feat, _dec_out, _ids, enc_pools = model(images, return_recon=not skip_recon)
                vq_loss = sum(d.float() for d in diffs) * args.vq_commitment_weight
                if skip_recon:
                    loss = vq_loss
                else:
                    # Compute reconstruction losses (pixel + fft + perceptual + gdl)
                    # separately from VQ loss so scale_recon_loss doesn't also
                    # scale the commitment cost.
                    net_out = {"reconstruction": [recon], "quantization_losses": []}
                    if "mask" in batch:
                        net_out["mask"] = batch["mask"].to(device, non_blocking=True)
                    recon_loss = loss_fn(net_out, images) * args.scale_recon_loss
                    loss = recon_loss + vq_loss

                # Cliff disentanglement regularizer on pooled encoder latents
                if cliff_fn is not None:
                    # enc_pools: list of (B, C) per encoder level — concatenate
                    # across levels to form (B, d) latent vector z.
                    z = torch.cat([p.float() for p in enc_pools], dim=1)  # (B, d)
                    cliff_loss = cliff_fn(z) * args.scale_cliff_loss
                    loss = loss + cliff_loss

            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # ── Logging ───────────────────────────────────────────────────
            if step % args.log_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0

                if writer:
                    writer.add_scalar("train/loss", loss.item(), step)
                    writer.add_scalar("train/lr", lr, step)
                    for i, d in enumerate(diffs):
                        writer.add_scalar(f"train/vq_loss_{i}", d.item(), step)
                    if not skip_recon:
                        for name, val in loss_fn.get_summaries().get(TBSummaryTypes.SCALAR, {}).items():
                            v = val.item() if torch.is_tensor(val) else val
                            writer.add_scalar(f"train/{name}", v, step)
                    if cliff_fn is not None:
                        for name, val in cliff_fn.get_summaries().get(TBSummaryTypes.SCALAR, {}).items():
                            v = val.item() if torch.is_tensor(val) else val
                            writer.add_scalar(f"train/{name}", v, step)

                # CSV row
                csv_row = {
                    "step": step, "epoch": epoch,
                    "elapsed_s": f"{elapsed:.1f}", "lr": f"{lr:.2e}",
                    "total_loss": f"{loss.item():.6f}",
                }
                if not skip_recon:
                    summaries = loss_fn.get_summaries().get(TBSummaryTypes.SCALAR, {})
                    for name, val in summaries.items():
                        v = val.item() if torch.is_tensor(val) else val
                        if "MAE" in name:
                            csv_row["pixel_loss"] = f"{v:.6f}"
                        elif "Jukebox" in name:
                            csv_row["fft_loss"] = f"{v:.6f}"
                        elif name == "Loss-Perceptual-Reconstruction":
                            csv_row["perceptual_loss"] = f"{v:.6f}"
                for i, d in enumerate(diffs):
                    csv_row[f"vq_loss_{i}"] = f"{d.item():.6f}"
                if cliff_fn is not None:
                    cliff_summaries = cliff_fn.get_summaries().get(TBSummaryTypes.SCALAR, {})
                    for name, val in cliff_summaries.items():
                        v = val.item() if torch.is_tensor(val) else val
                        csv_row[name] = f"{v:.6f}"
                csv_logger.log_train(csv_row)

                # Build a readable summary line with component losses
                parts = [
                    f"step {step:>7d}/{args.train_steps}",
                    f"loss {loss.item():.4f}",
                ]
                if not skip_recon:
                    summaries = loss_fn.get_summaries().get(TBSummaryTypes.SCALAR, {})
                    for name, val in summaries.items():
                        v = val.item() if torch.is_tensor(val) else val
                        if "MAE" in name:
                            parts.append(f"pix {v:.4f}")
                        elif "Jukebox" in name:
                            parts.append(f"fft {v:.4f}")
                        elif name == "Loss-Perceptual-Reconstruction":
                            parts.append(f"perc {v:.4f}")
                        elif "GDL" in name:
                            parts.append(f"gdl {v:.4f}")
                vq_vals = [f"{d.item():.4f}" for d in diffs]
                parts.append(f"vq [{','.join(vq_vals)}]")
                if cliff_fn is not None:
                    cliff_total = cliff_fn.get_summaries().get(TBSummaryTypes.SCALAR, {}).get("Loss-Cliff-Total")
                    if cliff_total is not None:
                        v = cliff_total.item() if torch.is_tensor(cliff_total) else cliff_total
                        parts.append(f"cliff {v * args.scale_cliff_loss:.4f}")
                parts.extend([f"lr {lr:.2e}", f"ep {epoch}", f"{elapsed:.0f}s"])
                log.info(" | ".join(parts))

            # ── Validation + checkpoint ───────────────────────────────────
            if step > 0 and step % args.checkpoint_steps == 0:
                val_sample, val_recon, val_loss = validate(
                    model, val_loader, device, amp_enabled,
                    commitment_weight=args.vq_commitment_weight,
                )
                save_decoded_images(
                    data=val_sample,
                    recon=val_recon,
                    args=args,
                    step=step,
                    save_dir=save_dir,
                )
                

                csv_logger.log_val({
                    "step": step, "epoch": epoch,
                    "elapsed_s": f"{time.time() - t0:.1f}",
                    "val_loss": f"{val_loss:.6f}",
                })
                log.info(f"  val loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

                if writer:
                    writer.add_scalar("val/loss", val_loss, step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        out_dir / "checkpoint_best.pt",
                        model, optimizer, scheduler, scaler, step, best_val_loss, args,
                    )

                save_checkpoint(
                    ckpt_latest,
                    model, optimizer, scheduler, scaler, step, best_val_loss, args,
                )

                if args.save_all_checkpoints:
                    save_checkpoint(
                        out_dir / f"checkpoint_{step}.pt",
                        model, optimizer, scheduler, scaler, step, best_val_loss, args,
                    )

                model.train()

            step += 1

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(
        out_dir / "checkpoint_final.pt",
        model, optimizer, scheduler, scaler, step, best_val_loss, args,
    )
    if writer:
        writer.close()
    log.info(f"Done -- {step} steps in {time.time() - t0:.0f}s")


def run_evaluation(args):
    """Run full evaluation on the validation set using a trained checkpoint."""
    device = get_device(args.no_cuda)
    amp_enabled = args.use_amp and device.type == "cuda"

    out_dir = Path(args.model_dir) / args.model_id
    ckpt_path = args.checkpoint or (out_dir / "checkpoint_best.pt")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        return

    log.info(f"Loading checkpoint: {ckpt_path}")
    model = load_model_from_checkpoint(str(ckpt_path), device)

    # Build validation data
    items = load_items(args.dataroot, args.csv_path, load_masks=getattr(args, "masks_from_disk", False))
    if not items:
        log.error("No data items found.")
        return

    _, val_transform = transforms(
        spacing=args.image_spacing,
        crop_margin=args.crop_margin,
    )

    if args.val_size < 1:
        val_count = int(len(items) * args.val_size)
    else:
        val_count = int(args.val_size)
    val_count = max(1, min(val_count, len(items) // 2))

    rng = np.random.RandomState(args.seed)
    labels = np.array([it["label"] for it in items])
    unique_labels = np.unique(labels)
    val_items = []
    for lbl in unique_labels:
        lbl_indices = np.where(labels == lbl)[0]
        rng.shuffle(lbl_indices)
        n_val = max(1, int(len(lbl_indices) * val_count / len(items)))
        val_items.extend(items[i] for i in lbl_indices[:n_val])
    rng.shuffle(val_items)
    val_dicts = [{"image": it["image"], "mask": it["mask"]} if "mask" in it else {"image": it["image"]} for it in val_items]

    val_set = CacheDataset(
        data=val_dicts, transform=val_transform,
        cache_rate=args.val_cache_rate, num_workers=args.workers,
    )
    loader_kwargs = dict(
        pin_memory=(device.type == "cuda" and not args.no_pin_memory),
        num_workers=args.workers,
        persistent_workers=(args.persistent_workers and args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    loss_fn = BaselineLoss(commitment_weight=args.vq_commitment_weight).to(device)
    log.info(f"Evaluating on {len(val_set)} samples...")

    val_sample, val_recon, val_loss = validate(
        model, val_loader, device, amp_enabled,
        commitment_weight=args.vq_commitment_weight,
    )
    log.info(f"Validation loss: {val_loss:.4f}")

    # Log per-level codebook utilization (EMA-based from training)
    for i, codebook in enumerate(model.codebooks):
        util = codebook.codebook_utilization()
        log.info(f"  Codebook {i} (EMA): {util['active_codes']}/{codebook.n_embed} active "
                 f"({util['utilization']:.1%}), perplexity={util['perplexity']:.1f}")

    # Also compute actual inference-based utilization by collecting indices
    log.info("Computing inference-based codebook utilization...")
    all_ids = [[] for _ in range(model.nb_levels)]
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            images = images.to(memory_format=torch.channels_last_3d)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                _, _, _, _, id_outputs, _ = model(images, return_recon=True)
            for lvl in range(model.nb_levels):
                all_ids[lvl].append(id_outputs[lvl].cpu())
    for i, codebook in enumerate(model.codebooks):
        ids_cat = torch.cat(all_ids[i], dim=0)
        util = CodeLayer.codebook_utilization_from_indices(ids_cat, codebook.n_embed)
        log.info(f"  Codebook {i} (inference): {util['active_codes']}/{codebook.n_embed} active "
                 f"({util['utilization']:.1%}), perplexity={util['perplexity']:.1f}")

    # Save example reconstructions
    save_dir = Path(args.model_dir) / args.model_id / "eval_images"
    save_dir.mkdir(exist_ok=True)
    save_decoded_images(data=val_sample, recon=val_recon, args=args, step=0, save_dir=save_dir)
    log.info(f"Example reconstructions saved → {save_dir}")


def main():
    parser = parse_args()
    args = parser.parse_args()
    if args.evaluate:
        run_evaluation(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
