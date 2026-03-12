#!/usr/bin/env bash
set -euo pipefail

python /nfs/home/nglazman/vqvae-smm/train.py --model-id vqvae-feature-maps \
    --dataroot /nfs/home/nglazman/ADNI_registered \
    --csv-path /nfs/home/nglazman/cluster/labels_cleaned_3class.csv \
    --batch-size 4 \
    --image-spacing 1.0 \
    --lr 1e-4 \
    --train-steps 30000 \
    --gradient-checkpointing \
    --compile \
    --use-amp