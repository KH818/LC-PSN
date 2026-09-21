#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/evaluation"
PYTHONPATH=. python run_eval.py --checkpoint-set sep8 --seeds 0 1 2 \
  --lcpsn-variants uncovON uncovOFF --min-seps 8 --out-prefix reproduced_sep8
PYTHONPATH=. python run_eval.py --checkpoint-set music --min-seps 8 \
  --out-prefix reproduced_music
