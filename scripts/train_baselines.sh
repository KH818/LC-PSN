#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/baselines/TCN_MUSIC"
export PYTHONPATH=.
python exe/train_models/one_bit/sep8/generate_sep8_data.py
for seed in 0 1 2; do
  for model in trans tcn da; do
    python exe/train_models/one_bit/sep8/train_sep8.py --model "$model" --seed "$seed" --epochs 100
  done
done
