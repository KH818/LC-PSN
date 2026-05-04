import argparse
import csv
import math
import os
import random
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import LCPSN
from dataset import get_dataloader, create_complete_dataset
from criterion import build_angle_grid_rad, permutation_best_errors_deg
from physics import ULA_action_vector


M = 8
SNAPSHOTS = 200
BATCH_SIZE = 128
ANGLES_COUNT = 360
K_MIN = 2
K_MAX = 5
VARIANCE_FLOOR = 5e-2
NUM_SOURCES = [2, 3, 4, 5]
TEST_SIZE_PER_SNR = 2000
EVAL_BASE_SEED = 9000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42069):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_seed_for_snr(snr: int) -> int:
    return EVAL_BASE_SEED + int(snr) + 1000


def ensure_test_file_for_snr(snr: int, force_regenerate: bool = False) -> str:
    fname = f"test_snr_{snr}.h5"
    if force_regenerate and os.path.exists(fname):
        os.remove(fname)

    if not os.path.exists(fname):
        seed = eval_seed_for_snr(snr)
        print(f"[INFO] Creating {fname} with eval_seed={seed}")
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            create_complete_dataset(
                name=f"test_snr_{snr}",
                size=TEST_SIZE_PER_SNR,
                snr=snr,
                snapshots=SNAPSHOTS,
                m=M,
                num_sources=NUM_SOURCES,
                coherent=False,
                save=True,
            )
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
    else:
        print(f"[INFO] Using existing {fname} (eval_seed={eval_seed_for_snr(snr)} if regenerated)")
    return fname


def precompute_steering_vectors(m=M, angles_count=ANGLES_COUNT):
    angles = np.linspace(-np.pi / 2, np.pi / 2, angles_count, endpoint=False)
    a_matrix = torch.zeros((m, angles_count), dtype=torch.complex64)
    for i, ang in enumerate(angles):
        a_matrix[:, i] = torch.from_numpy(ULA_action_vector(ang, m)).to(torch.complex64)
    return a_matrix.to(DEVICE)


def build_model() -> LCPSN:
    return LCPSN(
        m=M,
        snapshots=SNAPSHOTS,
        angles_count=ANGLES_COUNT,
        d_model=96,
        nhead=8,
        num_layers=4,
        ff_dim=256,
        dropout=0.10,
        spectrum_hidden=512,
        k_min=K_MIN,
        k_max=K_MAX,
        variance_floor=VARIANCE_FLOOR,
    ).to(DEVICE)


def load_model(ckpt_path: str) -> LCPSN:
    model = build_model()
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def select_topk_separated_indices(
    score_np: np.ndarray,
    grid_np: np.ndarray,
    k_hat: int,
    min_separation_deg: float,
) -> List[int]:
    min_sep = math.radians(min_separation_deg)
    candidate_idx = []
    for i in range(1, len(score_np) - 1):
        if score_np[i] > score_np[i - 1] and score_np[i] > score_np[i + 1]:
            candidate_idx.append(i)

    ranked = sorted(candidate_idx, key=lambda idx: score_np[idx], reverse=True)
    if len(ranked) == 0:
        ranked = list(np.argsort(score_np)[::-1])

    selected = []
    for idx in ranked:
        if all(abs(grid_np[idx] - grid_np[j]) >= min_sep for j in selected):
            selected.append(idx)
        if len(selected) >= k_hat:
            break

    if len(selected) < k_hat:
        for idx in list(np.argsort(score_np)[::-1]):
            if idx in selected:
                continue
            if all(abs(grid_np[idx] - grid_np[j]) >= min_sep for j in selected):
                selected.append(idx)
            if len(selected) >= k_hat:
                break

    return sorted(selected[:k_hat])


def local_peak_count(values: np.ndarray, threshold: float) -> int:
    if len(values) < 3:
        return 0
    peaks = 0
    for i in range(1, len(values) - 1):
        if values[i] >= threshold and values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks += 1
    return peaks


def halfmax_width_bins(values: np.ndarray) -> int:
    idx = int(np.argmax(values))
    peak = float(values[idx])
    if peak <= 0.0:
        return 0
    half = 0.5 * peak
    left = idx
    while left > 0 and values[left] >= half:
        left -= 1
    right = idx
    while right < len(values) - 1 and values[right] >= half:
        right += 1
    return int(right - left)


def decode_lcpsn_numpy(mu, sigma2, k_logits, angle_grid_rad, min_separation_deg):
    score = mu / (np.sqrt(np.maximum(sigma2, 1e-8)) + 1e-8)
    k_hat = int(np.argmax(k_logits) + K_MIN)
    k_hat = max(K_MIN, min(K_MAX, k_hat))
    selected_idx = select_topk_separated_indices(score, angle_grid_rad, k_hat, min_separation_deg)
    pred = [float(angle_grid_rad[i]) for i in selected_idx]
    return score, k_hat, pred, selected_idx


def plot_example(
    out_path,
    angle_grid_deg,
    true_angles_rad,
    pred_angles_rad,
    mu,
    sigma2,
    score,
    s_base,
    k_true,
    k_hat,
    snr,
    rmse_deg,
):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    panels = [
        (axes[0], mu, "mu", "tab:blue"),
        (axes[1], score, "score = mu / sqrt(sigma2)", "tab:green"),
        (axes[2], sigma2, "sigma2", "tab:orange"),
    ]
    for ax, values, label, color in panels:
        ax.plot(angle_grid_deg, values, color=color, linewidth=1.6, label=label)
        for i, theta in enumerate(true_angles_rad):
            ax.axvline(np.rad2deg(theta), color="tab:red", linestyle="--", linewidth=1.5, alpha=0.9, label="true" if i == 0 else None)
        for i, theta in enumerate(pred_angles_rad):
            ax.axvline(np.rad2deg(theta), color="black", linestyle=":", linewidth=1.8, alpha=0.9, label="pred" if i == 0 else None)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].plot(angle_grid_deg, s_base, color="tab:purple", linewidth=1.0, alpha=0.55, label="S_base")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Angle (deg)")
    fig.suptitle(f"SNR={snr} dB | trueK={k_true} predK={k_hat} | RMSE={rmse_deg:.3f} deg")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def collect_diagnostics(
    model,
    snr: int,
    a_steering,
    angle_grid_rad_t,
    sample_limit: int,
    min_separation_deg: float,
    out_dir: str,
    num_examples: int,
    force_regenerate: bool,
):
    fname = ensure_test_file_for_snr(snr, force_regenerate=force_regenerate)
    loader = get_dataloader(fname, batch_size=BATCH_SIZE, shuffle=False, use_1bit=True)
    angle_grid_rad = angle_grid_rad_t.detach().cpu().numpy()
    angle_grid_deg = np.rad2deg(angle_grid_rad)

    example_dir = os.path.join(out_dir, "examples")
    os.makedirs(example_dir, exist_ok=True)

    rows = []
    saved_examples = 0
    seen = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x, a_steering)

            mu_b = out["mu"].detach().cpu().numpy()
            sigma2_b = out["sigma2"].detach().cpu().numpy()
            klog_b = out["K_logits"].detach().cpu().numpy()
            sbase_b = out["S_base"].detach().cpu().numpy()
            y_b = y.numpy()

            for b in range(mu_b.shape[0]):
                true_angles = y_b[b][np.abs(y_b[b] - np.pi) > 1e-6]
                score, k_hat, pred_angles, selected_idx = decode_lcpsn_numpy(
                    mu_b[b],
                    sigma2_b[b],
                    klog_b[b],
                    angle_grid_rad,
                    min_separation_deg,
                )
                mae_deg, rmse_deg, _ = permutation_best_errors_deg(pred_angles, np.sort(true_angles))
                if np.isnan(rmse_deg):
                    rmse_deg = float("nan")
                    mae_deg = float("nan")

                k_true = int(len(true_angles))
                k_prob = torch.softmax(torch.from_numpy(klog_b[b]), dim=-1).numpy()
                top_score = np.sort(score)[::-1]
                row = {
                    "snr": snr,
                    "idx": seen,
                    "k_true": k_true,
                    "k_hat": k_hat,
                    "k_correct": int(k_true == k_hat),
                    "mae_deg": float(mae_deg),
                    "rmse_deg": float(rmse_deg),
                    "mu_max": float(mu_b[b].max()),
                    "mu_mean": float(mu_b[b].mean()),
                    "score_max": float(score.max()),
                    "score_mean": float(score.mean()),
                    "sigma2_mean": float(sigma2_b[b].mean()),
                    "sigma2_min": float(sigma2_b[b].min()),
                    "sigma2_max": float(sigma2_b[b].max()),
                    "score_peak_count_015": local_peak_count(score, 0.15),
                    "score_peak_count_030": local_peak_count(score, 0.30),
                    "mu_peak_count_015": local_peak_count(mu_b[b], 0.15),
                    "mu_peak_count_030": local_peak_count(mu_b[b], 0.30),
                    "score_halfmax_width_bins": halfmax_width_bins(score),
                    "top2_over_top1": float(top_score[1] / (top_score[0] + 1e-8)) if len(top_score) > 1 else 0.0,
                    "k_conf": float(k_prob[k_hat - K_MIN]),
                    "k_prob_2": float(k_prob[0]),
                    "k_prob_3": float(k_prob[1]),
                    "k_prob_4": float(k_prob[2]),
                    "k_prob_5": float(k_prob[3]),
                }
                rows.append(row)

                if saved_examples < num_examples:
                    out_path = os.path.join(example_dir, f"snr{snr}_idx{seen}_k{k_true}_pred{k_hat}.png")
                    plot_example(
                        out_path,
                        angle_grid_deg,
                        true_angles,
                        pred_angles,
                        mu_b[b],
                        sigma2_b[b],
                        score,
                        sbase_b[b],
                        k_true,
                        k_hat,
                        snr,
                        row["rmse_deg"],
                    )
                    saved_examples += 1

                seen += 1
                if seen >= sample_limit:
                    break
            if seen >= sample_limit:
                break

    return rows


def write_rows_csv(path: str, rows):
    if len(rows) == 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    if len(rows) == 0:
        return
    keys = [
        "k_correct",
        "rmse_deg",
        "mae_deg",
        "mu_max",
        "mu_mean",
        "score_peak_count_015",
        "score_peak_count_030",
        "sigma2_mean",
        "top2_over_top1",
        "k_conf",
    ]
    print("[SUMMARY]")
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        print(f"  {key}: mean={np.nanmean(vals):.4f} std={np.nanstd(vals):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best_lcpsn_1bit_S4_noBSC.pt")
    parser.add_argument("--snrs", type=int, nargs="*", default=[0, 5, 10])
    parser.add_argument("--outdir", type=str, default="./diagnostics_lcpsn_spectrum")
    parser.add_argument("--sample-limit", type=int, default=256)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--min-separation-deg", type=float, default=4.0)
    parser.add_argument("--force-regenerate", action="store_true")
    args = parser.parse_args()

    seed_everything(42069)
    os.makedirs(args.outdir, exist_ok=True)

    model = load_model(args.ckpt)
    a_steering = precompute_steering_vectors(M, ANGLES_COUNT)
    angle_grid_rad_t = build_angle_grid_rad(ANGLES_COUNT, device=DEVICE)

    all_rows = []
    for snr in args.snrs:
        print(f"[INFO] Diagnosing SNR={snr} dB")
        rows = collect_diagnostics(
            model=model,
            snr=snr,
            a_steering=a_steering,
            angle_grid_rad_t=angle_grid_rad_t,
            sample_limit=args.sample_limit,
            min_separation_deg=args.min_separation_deg,
            out_dir=args.outdir,
            num_examples=args.num_examples,
            force_regenerate=args.force_regenerate,
        )
        write_rows_csv(os.path.join(args.outdir, f"diagnostics_snr{snr}.csv"), rows)
        print_summary(rows)
        all_rows.extend(rows)

    write_rows_csv(os.path.join(args.outdir, "diagnostics_all.csv"), all_rows)
    print(f"[OK] Saved diagnostics to: {args.outdir}")


if __name__ == "__main__":
    main()
