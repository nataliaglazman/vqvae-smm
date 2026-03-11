#!/usr/bin/env bash
set -euo pipefail

python /nfs/home/nglazman/vqvae-smm/eval.py --mode rf-both \
    --checkpoint /nfs/home/nglazman/results/vqvae/checkpoint_best.pt \
    --save rf_combined.png