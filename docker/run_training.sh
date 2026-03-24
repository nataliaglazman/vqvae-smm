#!/usr/bin/env bash
set -euo pipefail

python /nfs/home/nglazman/vqvae-smm/train.py \
    --dataroot /nfs/home/nglazman/ADNI_scalarmomentum \
    --csv-path /nfs/home/nglazman/cluster/labels_cleaned_3class.csv \
    --model-id vqvae-scalar-momentum-cropped \
    --batch-size 2 \
    --image-spacing 1.0 \
    --lr 1e-4 \
    --train-steps 30000 \
    --gradient-checkpointing \
    --compile \
    --use-amp \
    --vqvae-scaling-rates 2 2 2 \
    --skip-recon-ratio 0.3 \
    --entropy-weight 0.1 \
    --vq-commitment-weight 0.25
