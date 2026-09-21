#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/lcpsn"
python generate_paper_data.py
OUT="$ROOT/results/retrained/lcpsn"
mkdir -p "$OUT"
cd "$OUT"
for seed in 0 1 2; do
  for variant in uncovON uncovOFF; do
    if [[ "$variant" == "uncovON" ]]; then lambda=0.001; else lambda=0; fi
    python "$ROOT/lcpsn/train.py" \
      --seed "$seed" --run-name "lcpsn_sep8_${variant}_seed${seed}" \
      --train-file "$ROOT/data/generated/lcpsn/train_sep8_1bit.h5" \
      --val-file "$ROOT/data/generated/lcpsn/val_sep8_1bit.h5" \
      --snr-list 0 5 10 15 20 --min-sep-deg 8 \
      --num-train 256000 --num-val 25600 --epochs 100 --batch-size 128 \
      --cov-mode proposed --lambda-uncov "$lambda"
  done
done
