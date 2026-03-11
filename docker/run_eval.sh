#!/usr/bin/env bash
set -euo pipefail

python /nfs/home/nglazman/vqvae-smm/eval.py --mode rf-both \
    --dataroot /nfs/home/nglazman/ADNI_registered \
    --checkpoint /nfs/home/nglazman/results/vqvae/checkpoint_best.pt \
    --save rf_combined.png