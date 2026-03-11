#!/usr/bin/env bash
set -euo pipefail

python /nfs/home/nglazman/vqvae-smm/train.py --evaluate --checkpoint /nfs/home/nglazman/results/vqvae/checkpoint_best.pt \
    --dataroot /nfs/home/nglazman/ADNI_registered \
    --csv-path /nfs/home/nglazman/cluster/labels_cleaned_3class.csv