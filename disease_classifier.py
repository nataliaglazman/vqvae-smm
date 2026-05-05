"""Train 3D DenseNet heads on frozen VQ-VAE-2 spatial features and compare.

Loads a checkpoint via :func:`eval.load_model_from_checkpoint` (which reads the
training config from ``ckpt["args"]`` and rebuilds the encoder), freezes it,
then trains one MONAI ``DenseNet121`` (3D) per requested encoder level on the
spatial feature map ``(B, hidden_channels, D, H, W)`` returned by that level.
This lets you compare disease-classification accuracy across encoder depths.

Usage:
    python disease_classifier.py \\
        --checkpoint results/checkpoint_best.pt \\
        --dataroot /path/to/ADNI \\
        --csv-path merged_data.csv \\
        --feature-levels 0 1 2 \\
        --task ad_cn \\
        --classifier-epochs 50 --classifier-batch-size 8

Speed: the encoder runs ONCE over train + val and the resulting spatial maps
are held in CPU RAM (fp16 by default) or memmap'd to disk. Every probed level
trains on those cached tensors, so encoder cost is paid once regardless of how
many levels are compared.

Tasks:
  --task all    — multi-class over every Group label found in the labels CSV
  --task ad_cn  — binary AD vs CN (CN→0, AD→1); MCI rows are dropped

Outputs (per level tag):
    <out-dir>/<out-name>_<task>__L<level>_best.pt — best DenseNet head
    <out-dir>/<out-name>_metrics.json             — combined metrics file
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from eval import FeatureHook, load_model_from_checkpoint
from utils import ADNIDataset, load_items, transforms
from vqvae2 import VQVAE

logger = logging.getLogger("disease_classifier")


# ---------------------------------------------------------------------------
# Dataset construction — mirrors the train/val stratified split used in
# train.py so we evaluate disease prediction on the same val subjects the
# VQ-VAE never reconstructed in training.
# ---------------------------------------------------------------------------


def _resolve_task(task: str, label_map: dict) -> tuple[dict, str]:
    """Return (remap: orig_int → new_int, task_name).

    `label_map` is the {Group_string: int} dict built by load_items. Original
    ints not present in the returned remap are dropped.
    """
    if task == "all":
        # Identity over the full int range.
        return {v: v for v in label_map.values()}, ",".join(label_map.keys())
    if task == "ad_cn":
        ad_int = next((v for k, v in label_map.items() if k.upper().startswith("AD")), None)
        cn_int = next((v for k, v in label_map.items() if k.upper().startswith("CN")), None)
        if ad_int is None or cn_int is None:
            raise RuntimeError(
                f"--task ad_cn requires AD and CN groups in the CSV; found {list(label_map.keys())}"
            )
        return {cn_int: 0, ad_int: 1}, "CN_vs_AD"
    raise ValueError(task)


def _build_label_map(csv_path: str) -> dict:
    """Re-derive the Group→int map exactly as load_items does."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    label_values = sorted(df["Group"].unique())
    return {v: i for i, v in enumerate(label_values)}


def _split_items(items: list, val_size: float, seed: int) -> tuple[list, list]:
    """Stratified train/val split matching train.py:222-241."""
    val_count = int(len(items) * val_size) if val_size < 1 else int(val_size)
    val_count = max(1, min(val_count, len(items) // 2))
    rng = np.random.RandomState(seed)
    labels = np.array([it["label"] for it in items])
    train_items, val_items = [], []
    for lbl in np.unique(labels):
        lbl_indices = np.where(labels == lbl)[0]
        rng.shuffle(lbl_indices)
        n_val = max(1, int(len(lbl_indices) * val_count / len(items)))
        val_items.extend(items[i] for i in lbl_indices[:n_val])
        train_items.extend(items[i] for i in lbl_indices[n_val:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


# ---------------------------------------------------------------------------
# Feature precomputation — encoder runs ONCE per dataset, results held in
# RAM (or memmap) keyed by level. All heads then train on cached tensors.
# ---------------------------------------------------------------------------


@dataclass
class FeatureCache:
    """All-channel encoder features per requested level + labels.

    ``feats[L]`` is a CPU tensor, ``np.memmap``, or ``_MemmapRowView``.
    """

    feats: dict  # level → tensor | np.memmap | _MemmapRowView
    labels: torch.Tensor  # (N,) long

    def __len__(self) -> int:
        return self.labels.shape[0]

    def read_batch(self, level: int, batch_rows: torch.Tensor) -> torch.Tensor:
        """Return ``(B, C, D, H, W)`` for the given row indices, as float32."""
        feat = self.feats[level]
        rows_np = batch_rows.numpy() if isinstance(batch_rows, torch.Tensor) else np.asarray(batch_rows)
        if isinstance(feat, _MemmapRowView):
            block = feat.gather(rows_np)
        elif isinstance(feat, np.memmap):
            block = torch.from_numpy(np.ascontiguousarray(feat[rows_np]))
        else:
            block = feat[torch.as_tensor(rows_np)]
        return block.float()


def _bytes_human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


@torch.no_grad()
def _precompute_features(
    encoder: VQVAE,
    dataset,
    levels: list[int],
    device,
    batch_size: int,
    workers: int,
    dtype: torch.dtype,
    cache_dir: Optional[str] = None,
    cache_tag: str = "train",
) -> FeatureCache:
    """One encoder pass; writes per-level spatial maps into a preallocated
    buffer (RAM tensor or np.memmap on disk).
    """
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    n_rows = len(dataset)

    # Capture encoder-level spatial maps via forward hooks. The model's forward
    # nulls `encoder_outputs[l]` after consumption to save memory, so the
    # returned features list is all-None when `pool_only=False`. Hooks read
    # the output before it's released.
    hooks = {L: FeatureHook() for L in levels}
    handles = [hooks[L].register(encoder.encoders[L]) for L in levels]

    try:
        # Probe one batch to discover per-level shapes.
        probe_iter = iter(loader)
        probe_batch = next(probe_iter)
        probe_img = probe_batch["image"][:1].to(device, non_blocking=True)
        encoder(probe_img, return_recon=False)
        shapes = {L: tuple(hooks[L].output.shape[1:]) for L in levels}
        del probe_img

        # Allocate one buffer per level.
        feats_by_level: dict = {}
        is_memmap = cache_dir is not None
        if is_memmap:
            os.makedirs(cache_dir, exist_ok=True)
            np_dtype = np.float16 if dtype == torch.float16 else np.float32
        for L in levels:
            nbytes = n_rows * int(np.prod(shapes[L])) * (2 if dtype == torch.float16 else 4)
            logger.info(
                f"  Allocating L{L} cache: ({n_rows}, {shapes[L]}) ≈ {_bytes_human(nbytes)} "
                f"({'disk' if is_memmap else 'RAM'})"
            )
            if is_memmap:
                path = os.path.join(cache_dir, f"feats_{cache_tag}_L{L}.dat")
                feats_by_level[L] = np.memmap(path, dtype=np_dtype, mode="w+", shape=(n_rows, *shapes[L]))
            else:
                feats_by_level[L] = torch.empty((n_rows, *shapes[L]), dtype=dtype)

        labels_all = torch.empty(n_rows, dtype=torch.long)
        write_idx = 0

        def _consume(batch):
            nonlocal write_idx
            bsz = batch["label"].shape[0]
            img = batch["image"].to(device, non_blocking=True)
            encoder(img, return_recon=False)
            for L in levels:
                f = hooks[L].output.to(dtype).cpu()
                if is_memmap:
                    feats_by_level[L][write_idx : write_idx + bsz] = f.numpy()
                else:
                    feats_by_level[L][write_idx : write_idx + bsz].copy_(f)
            labels_all[write_idx : write_idx + bsz] = batch["label"].long()
            write_idx += bsz

        _consume(probe_batch)
        for batch in probe_iter:
            _consume(batch)
            if write_idx % (10 * batch_size) < batch_size:
                logger.info(f"  precomputed {write_idx}/{n_rows}")

        if is_memmap:
            for L in levels:
                feats_by_level[L].flush()
    finally:
        for h in handles:
            h.remove()

    return FeatureCache(feats=feats_by_level, labels=labels_all)


def _filter_cache(cache: FeatureCache, valid_mask: torch.Tensor) -> FeatureCache:
    """Apply a row-mask. memmap caches are wrapped in a row-indexed view so
    we don't physically rewrite (which would double disk usage)."""
    keep_idx = torch.where(valid_mask)[0].numpy()
    new_feats = {}
    for L, t in cache.feats.items():
        if isinstance(t, np.memmap):
            new_feats[L] = _MemmapRowView(t, keep_idx)
        else:
            new_feats[L] = t[valid_mask]
    return FeatureCache(feats=new_feats, labels=cache.labels[valid_mask])


class _MemmapRowView:
    """np.memmap restricted to a row-index list."""

    def __init__(self, mm: np.memmap, row_idx: np.ndarray):
        self.mm = mm
        self.row_idx = row_idx
        self.shape = (len(row_idx), *mm.shape[1:])
        self.dtype = mm.dtype

    def __len__(self):
        return self.shape[0]

    def gather(self, batch_rows: np.ndarray) -> torch.Tensor:
        physical_rows = self.row_idx[batch_rows]
        block = np.asarray(self.mm[physical_rows])
        return torch.from_numpy(np.ascontiguousarray(block))


# ---------------------------------------------------------------------------
# Cached-feature epoch
# ---------------------------------------------------------------------------


def _epoch_cached(
    head,
    cache: FeatureCache,
    level: int,
    opt,
    device,
    batch_size: int,
    train: bool,
    use_amp: bool,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    head.train(train)
    N = len(cache)
    order = torch.randperm(N) if train else torch.arange(N)

    losses, preds, trues = [], [], []
    autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda")

    for start in range(0, N, batch_size):
        idx = order[start : start + batch_size]
        x = cache.read_batch(level, idx).to(device, non_blocking=True)
        y = cache.labels[idx].to(device, non_blocking=True)

        with autocast_ctx:
            logits = head(x)
            loss = F.cross_entropy(logits, y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        losses.append(float(loss.detach()))
        preds.append(logits.argmax(1).detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(trues) if trues else np.array([])
    bacc = balanced_accuracy_score(y_true, y_pred) if len(y_true) else float("nan")
    return float(np.mean(losses) if losses else float("nan")), bacc, y_true, y_pred


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to VQ-VAE-2 checkpoint .pt")
    p.add_argument("--dataroot", required=True, help="Root directory of ADNI data")
    p.add_argument("--csv-path", required=True, help="merged_data.csv (Subject, Group, ...)")
    p.add_argument("--out-dir", default=None, help="Where to write metrics + heads (default: dirname(checkpoint))")
    p.add_argument("--out-name", default="disease_classifier", help="Filename prefix")
    p.add_argument(
        "--feature-levels", type=int, nargs="+", default=[0],
        help="Encoder level(s) to probe. One classifier is trained per level.",
    )
    p.add_argument(
        "--task", default="ad_cn", choices=["all", "ad_cn"],
        help="`all`: multi-class over every Group. `ad_cn`: binary AD vs CN (MCI dropped).",
    )
    # Image preprocessing — must match how the encoder was trained.
    p.add_argument("--image-spacing", type=float, default=2.0)
    p.add_argument("--crop-margin", type=int, default=0)
    p.add_argument("--spatial-size", type=int, nargs="+", default=None)
    p.add_argument("--val-size", type=float, default=0.15, help="Validation fraction (0-1) or absolute count (>=1)")
    # Classifier
    p.add_argument("--classifier-epochs", type=int, default=50)
    p.add_argument("--classifier-lr", type=float, default=1e-4)
    p.add_argument("--classifier-batch-size", type=int, default=8)
    p.add_argument("--classifier-weight-decay", type=float, default=1e-4)
    # Cache
    p.add_argument("--cache-dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--extract-batch-size", type=int, default=4)
    p.add_argument(
        "--cache-mode", default="memmap", choices=["memmap", "memory"],
        help="memmap: features paged from disk. memory: held in RAM.",
    )
    p.add_argument("--feature-cache-dir", default=None,
                   help="Directory for memmap files. Defaults to <out-dir>/.disease_classifier_cache/")
    p.add_argument("--keep-cache", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42, help="Same as train.py default — picks the same val split.")
    return p.parse_args()


def main(cli):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    device = torch.device(cli.device)

    out_dir = cli.out_dir or os.path.dirname(os.path.abspath(cli.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Outputs → {out_dir}")

    logger.info(f"Loading VQ-VAE-2 from {cli.checkpoint}")
    encoder = load_model_from_checkpoint(cli.checkpoint, device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    logger.info(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}  nb_levels={encoder.nb_levels}")

    logger.info("Building dataset")
    items = load_items(cli.dataroot, cli.csv_path, load_masks=False)
    if not items:
        raise RuntimeError("No data items found — check --dataroot / --csv-path.")
    label_map = _build_label_map(cli.csv_path)
    logger.info(f"  Total items: {len(items)} | label_map={label_map}")

    train_items, val_items = _split_items(items, cli.val_size, cli.seed)
    logger.info(f"  Stratified split: train={len(train_items)} val={len(val_items)}")

    _, val_transform = transforms(
        spacing=cli.image_spacing,
        crop_margin=cli.crop_margin,
        spatial_size=cli.spatial_size,
        asymmetric_aug=False,
    )
    # Probing → no augmentation on the train side either.
    train_ds = ADNIDataset(train_items, transform=val_transform)
    val_ds = ADNIDataset(val_items, transform=val_transform)

    label_remap, task_name = _resolve_task(cli.task, label_map)
    logger.info(f"  Task: {cli.task} ({task_name}) | label_remap={label_remap}")

    cache_dtype = torch.float16 if cli.cache_dtype == "fp16" else torch.float32

    if cli.cache_mode == "memory":
        cache_dir = None
        logger.info("  Cache mode: in-memory")
    else:
        cache_dir = cli.feature_cache_dir or os.path.join(out_dir, ".disease_classifier_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"  Cache mode: memmap → {cache_dir}")

    levels = sorted(set(cli.feature_levels))
    invalid = [L for L in levels if L < 0 or L >= encoder.nb_levels]
    if invalid:
        raise ValueError(f"--feature-levels {invalid} out of range for nb_levels={encoder.nb_levels}")

    logger.info("Precomputing features (one encoder pass over train+val) ...")
    train_cache = _precompute_features(
        encoder, train_ds, levels, device,
        batch_size=cli.extract_batch_size, workers=cli.workers,
        dtype=cache_dtype, cache_dir=cache_dir, cache_tag="train",
    )
    val_cache = _precompute_features(
        encoder, val_ds, levels, device,
        batch_size=cli.extract_batch_size, workers=cli.workers,
        dtype=cache_dtype, cache_dir=cache_dir, cache_tag="val",
    )

    def _apply_task(cache: FeatureCache) -> FeatureCache:
        valid = torch.tensor([int(l.item()) in label_remap for l in cache.labels])
        cache = _filter_cache(cache, valid)
        cache.labels = torch.tensor([label_remap[int(l.item())] for l in cache.labels], dtype=torch.long)
        return cache

    train_cache = _apply_task(train_cache)
    val_cache = _apply_task(val_cache)
    num_classes = int(max(train_cache.labels.max().item(), val_cache.labels.max().item())) + 1
    logger.info(
        f"  After task filter: train={len(train_cache.labels)} val={len(val_cache.labels)} "
        f"num_classes={num_classes}"
    )
    if len(train_cache.labels) == 0 or len(val_cache.labels) == 0:
        raise RuntimeError("Empty train or val set after task filter.")

    # Encoder no longer needed.
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    use_amp = not cli.no_amp
    comparison: dict[str, dict] = {}

    for level in levels:
        tag = f"{task_name}__L{level}"
        logger.info("")
        logger.info(f"=== Probing {tag} ===")
        torch.manual_seed(cli.seed)
        np.random.seed(cli.seed)

        feat_shape = tuple(train_cache.feats[level].shape[1:])
        in_channels = feat_shape[0]
        logger.info(
            f"  Feature shape per sample: {feat_shape}  train rows={len(train_cache)} val rows={len(val_cache)}"
        )

        head = DenseNet121(
            spatial_dims=3, in_channels=in_channels, out_channels=num_classes, dropout_prob=0.2,
        ).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=cli.classifier_lr, weight_decay=cli.classifier_weight_decay)
        logger.info(f"  Head params: {sum(p.numel() for p in head.parameters()):,}")

        best_bacc = -1.0
        best_report: dict = {}
        best_cm: list = []
        best_epoch = 0
        history: list = []
        best_path = os.path.join(out_dir, f"{cli.out_name}_{tag}_best.pt")

        for epoch in range(1, cli.classifier_epochs + 1):
            tr_loss, tr_bacc, _, _ = _epoch_cached(
                head, train_cache, level, opt, device,
                cli.classifier_batch_size, train=True, use_amp=use_amp,
            )
            with torch.no_grad():
                va_loss, va_bacc, y_true, y_pred = _epoch_cached(
                    head, val_cache, level, opt, device,
                    cli.classifier_batch_size, train=False, use_amp=use_amp,
                )
            logger.info(
                f"  [{tag}] epoch {epoch:3d}/{cli.classifier_epochs} | "
                f"train loss={tr_loss:.4f} bacc={tr_bacc:.3f} | "
                f"val loss={va_loss:.4f} bacc={va_bacc:.3f}"
            )
            history.append({
                "epoch": epoch,
                "train_loss": tr_loss, "train_bacc": tr_bacc,
                "val_loss": va_loss, "val_bacc": va_bacc,
            })
            if va_bacc > best_bacc:
                best_bacc = va_bacc
                best_epoch = epoch
                torch.save({
                    "head_state_dict": head.state_dict(),
                    "epoch": epoch,
                    "val_bacc": va_bacc,
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "level": level,
                    "task": cli.task,
                }, best_path)
                best_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                best_cm = confusion_matrix(y_true, y_pred).tolist()

        logger.info(f"  [{tag}] best val bACC = {best_bacc:.4f} @ epoch {best_epoch} → {best_path}")
        comparison[tag] = {
            "task": cli.task,
            "level": level,
            "in_channels": in_channels,
            "feat_shape": list(feat_shape),
            "best_epoch": best_epoch,
            "best_val_bacc": best_bacc,
            "best_classification_report": best_report,
            "best_confusion_matrix": best_cm,
            "history": history,
            "ckpt_path": best_path,
        }

        del head, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    logger.info("")
    logger.info("=== Comparison summary (best val balanced accuracy) ===")
    for tag, res in sorted(comparison.items(), key=lambda kv: -kv[1]["best_val_bacc"]):
        logger.info(f"  {tag:<24} bACC={res['best_val_bacc']:.4f}  channels={res['in_channels']}")

    metrics_path = os.path.join(out_dir, f"{cli.out_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "comparison": comparison,
            "summary": {tag: res["best_val_bacc"] for tag, res in comparison.items()},
            "config": vars(cli),
        }, f, indent=2)
    logger.info(f"Metrics written to {metrics_path}")

    if cache_dir is not None and not cli.keep_cache:
        del train_cache, val_cache
        import gc
        gc.collect()
        for fn in os.listdir(cache_dir):
            if fn.startswith("feats_") and fn.endswith(".dat"):
                try:
                    os.remove(os.path.join(cache_dir, fn))
                except OSError as e:
                    logger.warning(f"  Could not remove {fn}: {e}")
        try:
            os.rmdir(cache_dir)
        except OSError:
            pass
        logger.info(f"Cache cleaned ({cache_dir})")


if __name__ == "__main__":
    main(parse_args())
