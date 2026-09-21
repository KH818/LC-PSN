"""
Train one 1-bit model (regressor + classifier) on the 8-deg-min-sep dataset
with a given training seed. Data is FIXED (generate_sep8_data.py); only the
training randomness (model init, DataLoader shuffle, cuDNN) follows --seed.

Everything else matches the original one_bit training scripts:
Adam(1e-3, (0.9,0.999)), batch 64, 100 epochs, RMSPE / CE losses.

Outputs:
  saved_models/sep8/one_bit_{model}_music_sep8_seed{S}.pt
  saved_models/sep8/one_bit_{model}_music_classifier_sep8_seed{S}.pt
  results/sep8/train_log_{model}_regressor_sep8_seed{S}.csv
  results/sep8/train_log_{model}_classifier_sep8_seed{S}.csv

Run:
    cd <repository>/baselines/TCN_MUSIC
    PYTHONPATH=. python exe/train_models/one_bit/sep8/train_sep8.py --model trans --seed 0
"""
import os
import sys
import csv
import argparse

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import h5py

from utils.uniform_linear_array import UniformLinearArray
from utils.deterministic_results import deterministic_results
from Estimators.Trans_MUSIC.transformer_music import TransformerMUSIC
from Estimators.DA_MUSIC.da_music import DeepAugmentedMUSIC
from Estimators.TCN_MUSIC.tcn_music import TemporalConvolutionalNetworkMUSIC

MODELS = {
    "trans": (TransformerMUSIC, "1-Bit Trans-MUSIC (sep8)", "one_bit_trans_music"),
    "da": (DeepAugmentedMUSIC, "1-Bit DA-MUSIC (sep8)", "one_bit_da_music"),
    "tcn": (TemporalConvolutionalNetworkMUSIC, "1-Bit TCN-MUSIC (sep8)", "one_bit_tcn_music"),
}

PROJECT_ROOT = os.path.abspath(os.path.join(REPO, "..", ".."))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "generated", "baselines", "one_bit_training_data_sep8.h5")
VAL_PATH = os.path.join(PROJECT_ROOT, "data", "generated", "baselines", "one_bit_validation_data_sep8.h5")


def load_h5(path):
    with h5py.File(path, "r") as f:
        return np.array(f.get("data")), np.array(f.get("targets"))


def save_csv(path, columns):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(columns.keys()))
        for i in range(len(next(iter(columns.values())))):
            w.writerow([i + 1] + [f"{columns[c][i]:.6f}" for c in columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS.keys(), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--skip-classifier", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing retrained checkpoint")
    args = ap.parse_args()

    cls, display_name, base = MODELS[args.model]
    model_dir = os.path.join(PROJECT_ROOT, "results", "retrained", "baselines", "checkpoints")
    result_dir = os.path.join(PROJECT_ROOT, "results", "training_logs", "baselines")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    reg_path = os.path.join(model_dir, f"{base}_sep8_seed{args.seed}.pt")
    clf_path = os.path.join(model_dir, f"{base}_classifier_sep8_seed{args.seed}.pt")

    print(f"=== {display_name} | seed={args.seed} | epochs={args.epochs} ===", flush=True)
    training_data, training_targets = load_h5(TRAIN_PATH)
    validation_data, validation_targets = load_h5(VAL_PATH)
    print(f"train={training_data.shape} val={validation_data.shape}", flush=True)

    deterministic_results(args.seed)

    angle_grids = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    rx_ula = UniformLinearArray(8)
    est = cls(angle_grids, 200, rx_ula, device="cuda")
    est.system_name(f"{display_name} seed{args.seed}")

    # --- regressor ---
    if os.path.exists(reg_path):
        if not args.overwrite:
            raise FileExistsError(f"{reg_path} exists; pass --overwrite to replace it")
        os.remove(reg_path)
    tr_loss, val_loss = est.train_regressor(
        training_data, training_targets,
        validation_data, validation_targets,
        learning_rate=0.001, betas=(0.9, 0.999),
        batch_size=64, number_of_epochs=args.epochs,
        model_path=reg_path,
    )
    save_csv(os.path.join(result_dir, f"train_log_{args.model}_regressor_sep8_seed{args.seed}.csv"),
             {"training_loss": tr_loss, "validation_loss": val_loss})
    print(f"[OK] regressor saved -> {reg_path}", flush=True)

    if args.skip_classifier:
        return

    # --- classifier (frozen/loaded regressor features, same as original) ---
    est.load_regressor_model(reg_path)
    if os.path.exists(clf_path):
        if not args.overwrite:
            raise FileExistsError(f"{clf_path} exists; pass --overwrite to replace it")
        os.remove(clf_path)
    tr_loss, tr_acc, val_loss, val_acc = est.train_classifier(
        training_data, training_targets,
        validation_data, validation_targets,
        learning_rate=0.001, betas=(0.9, 0.999),
        batch_size=64, number_of_epochs=args.epochs,
        model_path=clf_path,
    )
    save_csv(os.path.join(result_dir, f"train_log_{args.model}_classifier_sep8_seed{args.seed}.csv"),
             {"training_loss": tr_loss, "training_accuracy": tr_acc,
              "validation_loss": val_loss, "validation_accuracy": val_acc})
    print(f"[OK] classifier saved -> {clf_path}", flush=True)


if __name__ == "__main__":
    main()
