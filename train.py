"""Training script for VQ-VAE-2 on 3D brain MRI."""

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import nibabel as nib

from config import parse_args
from helper import get_device, get_parameter_count
from loss import BaselineLoss
from utils import ADNIDataset, TBSummaryTypes, build_transforms, load_items, save_decoded_images
from vqvae2 import VQVAE


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
    ckpt = torch.load(path, map_location=device, weights_only=False)
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
    ckpt = torch.load(path, map_location=device, weights_only=False)
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
    return model


# ── Validation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp_enabled):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            recon, diffs, *_ = model(images)
            loss = loss_fn({"reconstruction": [recon], "quantization_losses": diffs}, images)
        total += loss.item()
        n += 1
    return recon, total / max(n, 1)


# ── Training ──────────────────────────────────────────────────────────────────


def train(args):
    device = get_device(args.no_cuda)
    amp_enabled = args.use_amp and device.type == "cuda"

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
    items = load_items(args.dataroot, args.csv_path)
    if not items:
        log.error("No data items found. Check --dataroot and --csv-path.")
        return

    train_t, val_t = build_transforms(spacing=args.image_spacing, crop_margin=args.crop_margin)

    # Train / val split
    if args.val_size < 1:
        val_count = int(len(items) * args.val_size)
    else:
        val_count = int(args.val_size)
    val_count = max(1, min(val_count, len(items) // 2))

    np.random.shuffle(items)
    train_set = ADNIDataset(items[val_count:], train_t)
    val_set = ADNIDataset(items[:val_count], val_t)
    log.info(f"Dataset split -- train: {len(train_set)}, val: {len(val_set)}")

    if len(train_set) < args.batch_size:
        log.error(
            f"Training set ({len(train_set)}) smaller than batch size ({args.batch_size}). "
            "Reduce --batch-size or add more data."
        )
        return

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
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
    ).to(device)
    log.info(f"Parameters: {get_parameter_count(model):,}")

    # ── Loss / optimiser / scheduler ──────────────────────────────────────────
    loss_fn = BaselineLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_opt_steps = max(args.train_steps // args.gradient_accumulation_steps, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

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

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step = start_step
    epoch = 0
    optimizer.zero_grad()

    log.info(f"Training: step {step} -> {args.train_steps}")
    t0 = time.time()

    while step < args.train_steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.train_steps:
                break

            images = batch["image"].to(device, non_blocking=True)

            # Optionally skip reconstruction for memory / codebook-only steps
            skip_recon = args.skip_recon_ratio > 0 and random.random() < args.skip_recon_ratio

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                recon, diffs, *_ = model(images, return_recon=not skip_recon)
                if skip_recon:
                    loss = sum(diffs) * args.vq_commitment_weight
                else:
                    net_out = {"reconstruction": [recon], "quantization_losses": diffs}
                    loss = loss_fn(net_out, images) * args.scale_recon_loss

            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
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

                log.info(
                    f"step {step:>7d}/{args.train_steps} | loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | epoch {epoch} | {elapsed:.0f}s"
                )

            # ── Validation + checkpoint ───────────────────────────────────
            if step > 0 and step % args.checkpoint_steps == 0:
                recon, val_loss = validate(model, val_loader, loss_fn, device, amp_enabled)
                # save first validation recon for visual sanity check
                if step == args.checkpoint_steps:
                    save_decoded_images(
                        model=model,
                        data=batch,
                        args=args,
                        step=step,
                        save_dir = save_dir
                    )
                

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


def main():
    parser = parse_args()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
