#!/usr/bin/env bash
set -euo pipefail

python /nfs/home/nglazman/vqvae-smm/train.py
    --dataroot /nfs/home/nglazman/ADNI_stripped \
    --csv-path /nfs/home/nglazman/cluster/labels_cleaned_3class.csv \
    --model-id vqvae-stripped \
    --batch-size 4 \
    --image-spacing 1.0 \
    --lr 1e-4 \
    --train-steps 30000 \
    --gradient-checkpointing \
    --compile \
    --use-amp