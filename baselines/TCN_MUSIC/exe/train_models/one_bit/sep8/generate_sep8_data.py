"""
Generate one-bit training/validation data with a MINIMUM PAIRWISE DoA
SEPARATION of 8 degrees (all other settings identical to
train_one_bit_*_regressor.py / DeepAugmentedMUSIC.generate_data):

  * SNR {0,5,10,15,20} dB, 51200 train / 5120 val samples per SNR
  * K cycles over {2,3,4,5}, snapshots L=200, ULA M=8
  * z = Q1(Ax + n)  (noise before the 1-bit quantizer)
  * targets padded with pi to length M

The dataset is generated ONCE with a fixed seed (DATA_SEED) and shared by
all training seeds, so only training randomness varies between runs.

Run:
    cd <repository>/baselines/TCN_MUSIC
    PYTHONPATH=. python exe/train_models/one_bit/sep8/generate_sep8_data.py
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import h5py
from tqdm import tqdm

from utils.uniform_linear_array import UniformLinearArray
from utils.one_bit_signal_receiver import OneBitSignalReceiver

ANGLE_GRIDS = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
M, L = 8, 200
SNR_DB = list(range(0, 21, 5))
KS = [2, 3, 4, 5]
MIN_SEP_DEG = 8.0
DATA_SEED = 1234

TRAIN_PER_SNR = 51200
VAL_PER_SNR = 5120
PROJECT_ROOT = os.path.abspath(os.path.join(REPO, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "generated", "baselines")
os.makedirs(DATA_DIR, exist_ok=True)
TRAIN_PATH = os.path.join(DATA_DIR, "one_bit_training_data_sep8.h5")
VAL_PATH = os.path.join(DATA_DIR, "one_bit_validation_data_sep8.h5")


def sample_angles_min_sep(k, min_sep_rad, low, high, max_trials=2000):
    """Uniform DoAs with pairwise separation >= min_sep_rad (resample-until)."""
    for _ in range(max_trials):
        th = np.random.uniform(low, high, size=k)
        if k == 1 or np.all(np.diff(np.sort(th)) >= min_sep_rad):
            return th
    # fallback: sorted grid pick then push apart (mirrors unified_eval/signal_gen.py)
    th = np.sort(np.random.uniform(low + min_sep_rad, high - min_sep_rad, size=k))
    for i in range(1, k):
        if th[i] - th[i - 1] < min_sep_rad:
            th[i] = min(th[i - 1] + min_sep_rad, high)
    return th


def generate(receiver, samples_per_snr, path):
    n_total = samples_per_snr * len(SNR_DB)
    data = np.zeros((n_total, 2 * M, L), np.float32)
    targets = np.zeros((n_total, M), np.float32)
    min_sep = np.deg2rad(MIN_SEP_DEG)

    with tqdm(total=n_total) as bar:
        for snr_index, snr in enumerate(SNR_DB):
            for sample_index in range(samples_per_snr):
                k = KS[sample_index % len(KS)]
                angles = sample_angles_min_sep(
                    k, min_sep, ANGLE_GRIDS[0], ANGLE_GRIDS[-1])
                z = receiver.receive(angles, k, snr_db=snr)
                idx = snr_index * samples_per_snr + sample_index
                data[idx, :M, :] = np.real(z)
                data[idx, M:, :] = np.imag(z)
                targets[idx, :] = np.pad(angles, (0, M - k), 'constant',
                                         constant_values=np.pi)
                bar.update()

    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('targets', data=targets)
    print(f"[OK] wrote {path}  data={data.shape}", flush=True)


if __name__ == '__main__':
    receiver = OneBitSignalReceiver(L, UniformLinearArray(M))
    np.random.seed(DATA_SEED)
    if os.path.exists(TRAIN_PATH):
        print(f"[SKIP] {TRAIN_PATH} already exists")
    else:
        generate(receiver, TRAIN_PER_SNR, TRAIN_PATH)
    if os.path.exists(VAL_PATH):
        print(f"[SKIP] {VAL_PATH} already exists")
    else:
        generate(receiver, VAL_PER_SNR, VAL_PATH)
