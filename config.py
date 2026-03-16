# flake8: noqa
"""Argument parsing and dataset-specific configuration for multiview-CRL."""

import argparse


def parse_args() -> argparse.ArgumentParser:
    """
    Build and return the argument parser.

    Returns:
        argparse.ArgumentParser: Parser (call ``.parse_args()`` to get the namespace).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="/nfs/home/nglazman/ADNI_registered")
    parser.add_argument("--model-dir", type=str, default="results")
    parser.add_argument("--model-id", type=str, default="vqvae-upscaled-improved")
    parser.add_argument("--encoding-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=300001)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=1000)
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation instead of training")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for evaluation (defaults to checkpoint_best.pt in model dir)",
    )
    parser.add_argument(
        "--val-size", default=0.15, type=float,
        help="Validation fraction (0-1) or absolute count (>=1)",
    )
    parser.add_argument("--test-size", default=25000, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers. For 3D MRI with pin_memory, each worker holds "
        "prefetch_factor batches in pinned memory (~330 MB each). Keep this low (4-8).",
    )
    parser.add_argument(
        "--cache-rate",
        type=float,
        default=0.25,
        help="Fraction of training samples cached in RAM (0-1). Lower reduces CPU memory usage.",
    )
    parser.add_argument(
        "--val-cache-rate",
        type=float,
        default=0.1,
        help="Fraction of validation samples cached in RAM (0-1).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=1,
        help="Batches prefetched per DataLoader worker. Lower reduces CPU/pinned memory.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader worker processes alive between epochs (faster, higher RAM).",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pinned host memory in DataLoader to reduce CPU RAM usage.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision (fp16) to reduce memory",
    )
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume training from last checkpoint if available",
    )
    parser.add_argument("--load-args", action="store_true")
    parser.add_argument("--collate-random-pair", action="store_true")
    parser.add_argument(
        "--scale-recon-loss",
        type=float,
        default=1,
        help="Scale factor for the reconstruction loss",
    )
    # VQ-VAE-2 specific
    parser.add_argument("--vqvae-hidden-channels", type=int, default=64)
    parser.add_argument("--vqvae-res-channels", type=int, default=32)
    parser.add_argument("--vqvae-nb-levels", type=int, default=3)
    parser.add_argument("--vqvae-embed-dim", type=int, default=32)
    parser.add_argument("--vqvae-nb-entries", type=int, default=384)
    parser.add_argument("--vqvae-scaling-rates", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--vq-commitment-weight", type=float, default=0.25)
    parser.add_argument(
        "--entropy-weight", type=float, default=0.1,
        help="Codebook entropy regularization weight (0 to disable). "
        "Encourages uniform codebook usage to prevent collapse.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for memory in residual blocks",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for fused kernels (PyTorch 2.0+, ~20-50%% faster)",
    )
    parser.add_argument(
        "--compile-backend",
        type=str,
        default="inductor",
        help="torch.compile backend. Use 'inductor' (default, fastest, needs gcc) or "
             "'aot_eager' (no C compiler required, useful on clusters without gcc).",
    )
    parser.add_argument(
        "--skip-recon-ratio",
        type=float,
        default=0.0,
        help="Fraction of steps to skip reconstruction (0–1)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps (effective batch = batch_size × N)",
    )
    # Image preprocessing
    parser.add_argument("--image-spacing", type=float, default=1.0, help="Isotropic voxel spacing in mm")
    parser.add_argument("--crop-margin", type=int, default=0, help="Voxels to crop from each edge")
    parser.add_argument(
        "--csv-path", type=str,
        default="/nfs/home/nglazman/cluster/labels_cleaned_3class.csv",
        help="Path to CSV file with Subject and Group columns",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument(
        "--warmup-steps", type=int, default=500,
        help="Linear LR warmup steps before cosine decay (0 to disable)",
    )
    parser.add_argument("--change-lists", default=[[4, 5, 6, 8, 9, 10]])
    parser.add_argument("--faiss-omp-threads", type=int, default=16)
    # Evaluation
    return parser