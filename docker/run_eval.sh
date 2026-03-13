#!/usr/bin/env bash
set -euo pipefail

python /nfs/home/nglazman/vqvae-smm/eval.py --mode evaluate \
    --dataroot /nfs/home/nglazman/ADNI_stripped \
    --checkpoint /nfs/home/nglazman/results/vqvae-stripped/checkpoint_latest.pt \
    --device cuda \
    --spacing 1.0 \