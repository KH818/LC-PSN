"""
Generate the LCPSN sep8 dataset ONCE, shared by all six training runs.

Mirrors TCN_MUSIC/exe/train_models/one_bit/sep8/generate_sep8_data.py so the
4-model comparison sees the same signal distribution:
  * SNR {0,5,10,15,20} dB, 256,000 train / 25,600 val (51,200 / 5,120 per SNR)
  * K cycles over {2,3,4,5}, snapshots T=200, ULA M=8
  * minimum pairwise DoA separation 8 deg (resample-until)
  * h5 stores the complex analog signal; the 1-bit sign() is applied at load
    time by dataset.load_and_preprocess_data -> noise precedes quantization.

dataset.sample_thetas_with_min_sep draws from the GLOBAL numpy RNG, so the
dataset must be built here with a fixed seed rather than inside train.py (which
seeds globally with --seed and would give every training seed different data).

Run:
    cd <repository>/lcpsn
    python sep8/generate_sep8_data.py
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np

from train import create_snr_mix_h5, SNAPSHOTS

SNR_LIST = (0, 5, 10, 15, 20)
MIN_SEP_DEG = 8.0
NUM_TRAIN = 256000
NUM_VAL = 25600
TRAIN_SEED = 1234          # same convention as the 3-model sep8 data
VAL_SEED = 4321

PROJECT_ROOT = os.path.abspath(os.path.join(REPO, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "generated", "lcpsn")
os.makedirs(DATA_DIR, exist_ok=True)
TRAIN_OUT = os.path.join(DATA_DIR, "train_sep8_1bit")
VAL_OUT = os.path.join(DATA_DIR, "val_sep8_1bit")


def build(out_base, total, seed):
    if os.path.exists(f"{out_base}.h5"):
        print(f"[SKIP] {out_base}.h5 already exists", flush=True)
        return
    np.random.seed(seed)                      # angle sampling uses the global RNG
    create_snr_mix_h5(
        out_name=out_base,
        total_size=total,
        snapshots=SNAPSHOTS,
        snr_list=SNR_LIST,
        num_sources=[2, 3, 4, 5],
        coherent=False,
        seed=seed,
        min_sep_deg=MIN_SEP_DEG,
    )


if __name__ == "__main__":
    build(TRAIN_OUT, NUM_TRAIN, TRAIN_SEED)
    build(VAL_OUT, NUM_VAL, VAL_SEED)
    print("[DONE] sep8 datasets ready", flush=True)
