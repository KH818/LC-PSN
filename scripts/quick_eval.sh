#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/evaluation"
PYTHONPATH=. python run_eval.py \
  --checkpoint-set sep8 \
  --seeds 0 \
  --lcpsn-variants uncovON \
  --num-tests 64 \
  --batch 16 \
  --min-seps 8 \
  --out-prefix smoke_paper_models
