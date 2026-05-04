import os
import math
import random
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import LCPSN
from dataset import get_dataloader, create_complete_dataset
from criterion import build_angle_grid_rad, permutation_best_errors_deg, real_augmented_bsc_loss
from physics import ULA_action_vector


def seed_everything(seed=42069):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


M = 8
SNAPSHOTS = 200
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANGLES_COUNT = 360
K_MIN = 2
K_MAX = 5
VARIANCE_FLOOR = 5e-2
CKPT_PATH = "best_lcpsn_1bit.pt"
SNR_LIST = [0, 5, 10]
EVAL_BASE_SEED = 9000
NUM_SOURCES = [2, 3, 4, 5]
TEST_SIZE_PER_SNR = 2000
SUCCESS_THRESH_DEG = 5.0
OUT_DIR = "./eval_outputs_lcpsn"
os.makedirs(OUT_DIR, exist_ok=True)


def deg_to_rad(x: float) -> float:
    return x * np.pi / 180.0


def precompute_steering_vectors(m, angles_count=360):
    angles = np.linspace(-np.pi / 2, np.pi / 2, angles_count, endpoint=False)
    a_matrix = torch.zeros((m, angles_count), dtype=torch.complex64)
    for i, ang in enumerate(angles):
        a_matrix[:, i] = torch.from_numpy(ULA_action_vector(ang, m)).to(torch.complex64)
    return a_matrix.to(DEVICE)


def eval_seed_for_snr(snr: int) -> int:
    return EVAL_BASE_SEED + int(snr) + 1000


def select_topk_separated_peaks(
    score: torch.Tensor,
    angle_grid_rad: torch.Tensor,
    k_hat: int,
    min_separation_deg: float = 8.0,
):
    """
    score: [G]
    returns list[float] selected DOAs in radians
    """
    score_np = score.detach().cpu().numpy()
    grid_np = angle_grid_rad.detach().cpu().numpy()
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

    selected = sorted(selected)
    return [float(grid_np[i]) for i in selected[:k_hat]]


def decode_lcpsn_outputs(
    mu: torch.Tensor,
    sigma2: torch.Tensor,
    K_logits: torch.Tensor,
    angle_grid_rad: torch.Tensor,
    min_separation_deg: float = 8.0,
    eps: float = 1e-8,
):
    """
    K_hat = argmax(K_logits) + K_MIN.
    DOAs are top-K separated peaks from mu / (sqrt(sigma2) + eps).
    sigma2-based confidence weighting is disabled because sigma2 collapsed
    to a near-constant value across samples (std ~5e-3 on validation).
    """
    score = mu
    k_hat = torch.argmax(K_logits, dim=-1) + K_MIN
    results = []
    for b in range(score.size(0)):
        k_b = int(k_hat[b].item())
        k_b = max(K_MIN, min(K_MAX, k_b))
        results.append(select_topk_separated_peaks(score[b], angle_grid_rad, k_b, min_separation_deg))
    return results, k_hat.detach().cpu().numpy()


def project_hermitian(R: torch.Tensor) -> torch.Tensor:
    return 0.5 * (R + R.conj().transpose(-1, -2))


def apply_r_tta(
    model: LCPSN,
    x: torch.Tensor,
    a_steering: torch.Tensor,
    steps: int,
    lr: float,
    anchor_weight: float,
    psd_weight: float,
):
    """
    Refine only R_ref with L_BSC(R_ref, x), then recompute LC-PSN outputs.

    x: [B, 2M, T] one-bit real/imag sign snapshots
    """
    with torch.no_grad():
        initial_out = model(x, a_steering)
        R_tta = project_hermitian(initial_out["R_ref"].detach()).clone()
        R_init = R_tta.clone()
        bsc_before = real_augmented_bsc_loss(R_tta, x).detach()

    R_tta.requires_grad_(True)
    optimizer = torch.optim.Adam([R_tta], lr=float(lr))

    for _ in range(max(int(steps), 0)):
        optimizer.zero_grad(set_to_none=True)
        R_current = project_hermitian(R_tta)
        bsc_loss = real_augmented_bsc_loss(R_current, x)
        anchor_loss = torch.mean(torch.abs(R_current - R_init) ** 2).real
        eigvals = torch.linalg.eigvalsh(R_current)
        psd_loss = eigvals.clamp_max(0.0).abs().mean()
        loss = bsc_loss + float(anchor_weight) * anchor_loss + float(psd_weight) * psd_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            R_tta.copy_(project_hermitian(R_tta))

    with torch.no_grad():
        R_final = project_hermitian(R_tta.detach())
        bsc_after = real_augmented_bsc_loss(R_final, x).detach()
        out = model.forward_from_cov(R_final, a_steering)

    return out, bsc_before, bsc_after


def summarize_metrics(all_k_true, all_k_pred, all_doa_true, all_doa_pred):
    mae_deg_list, rmse_deg_list = [], []
    mae_deg_kcorr, rmse_deg_kcorr = [], []
    success_flags, success_flags_kcorr = [], []

    for kt, kp, dt, dp in zip(all_k_true, all_k_pred, all_doa_true, all_doa_pred):
        mae_deg, rmse_deg, _ = permutation_best_errors_deg(dp, dt)
        mae_deg_list.append(mae_deg)
        rmse_deg_list.append(rmse_deg)
        success_flags.append(1 if (not np.isnan(rmse_deg) and rmse_deg <= SUCCESS_THRESH_DEG) else 0)

        if kp == kt:
            mae_deg_k, rmse_deg_k, _ = permutation_best_errors_deg(dp, dt)
            mae_deg_kcorr.append(mae_deg_k)
            rmse_deg_kcorr.append(rmse_deg_k)
            success_flags_kcorr.append(1 if (not np.isnan(rmse_deg_k) and rmse_deg_k <= SUCCESS_THRESH_DEG) else 0)

    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(all_k_true, all_k_pred):
        if 2 <= t <= 5 and 2 <= p <= 5:
            cm[t - 2, p - 2] += 1

    doa_mae_deg = float(np.nanmean(mae_deg_list)) if len(mae_deg_list) > 0 else np.nan
    doa_rmse_deg = float(np.nanmean(rmse_deg_list)) if len(rmse_deg_list) > 0 else np.nan
    doa_mae_deg_kcorr = float(np.nanmean(mae_deg_kcorr)) if len(mae_deg_kcorr) > 0 else np.nan
    doa_rmse_deg_kcorr = float(np.nanmean(rmse_deg_kcorr)) if len(rmse_deg_kcorr) > 0 else np.nan

    return {
        "k_acc": float(np.mean(np.array(all_k_true) == np.array(all_k_pred))),
        "doa_mae_deg": doa_mae_deg,
        "doa_rmse_deg": doa_rmse_deg,
        "success_rate": float(np.mean(success_flags)) if len(success_flags) > 0 else np.nan,
        "doa_mae_deg_kcorrect": doa_mae_deg_kcorr,
        "doa_rmse_deg_kcorrect": doa_rmse_deg_kcorr,
        "success_rate_kcorrect": float(np.mean(success_flags_kcorr)) if len(success_flags_kcorr) > 0 else np.nan,
        "doa_mae_rad": deg_to_rad(doa_mae_deg) if not np.isnan(doa_mae_deg) else np.nan,
        "doa_rmse_rad": deg_to_rad(doa_rmse_deg) if not np.isnan(doa_rmse_deg) else np.nan,
        "doa_mae_rad_kcorrect": deg_to_rad(doa_mae_deg_kcorr) if not np.isnan(doa_mae_deg_kcorr) else np.nan,
        "doa_rmse_rad_kcorrect": deg_to_rad(doa_rmse_deg_kcorr) if not np.isnan(doa_rmse_deg_kcorr) else np.nan,
        "k_confusion_matrix": cm,
    }


def ensure_test_file_for_snr(snr):
    dataset_name = f"test_snr_{snr}_T{SNAPSHOTS}"
    fname = f"{dataset_name}.h5"
    if not os.path.exists(fname):
        seed = eval_seed_for_snr(snr)
        print(f"[INFO] Creating {fname} with eval_seed={seed}")
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            create_complete_dataset(
                name=dataset_name,
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


def evaluate_one_snr(
    model,
    a_steering,
    angle_grid_rad,
    snr,
    tta=False,
    tta_steps=5,
    tta_lr=0.01,
    tta_anchor=1.0,
    tta_psd=0.05,
):
    fname = ensure_test_file_for_snr(snr)
    loader = get_dataloader(fname, batch_size=BATCH_SIZE, shuffle=False, use_1bit=True)
    model.eval()

    all_k_true, all_k_pred = [], []
    all_doa_true, all_doa_pred = [], []
    eval_bsc_sum = 0.0
    eval_bsc_count = 0
    tta_bsc_before, tta_bsc_after = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        if tta:
            out, bsc_before, bsc_after = apply_r_tta(
                model,
                x,
                a_steering,
                steps=tta_steps,
                lr=tta_lr,
                anchor_weight=tta_anchor,
                psd_weight=tta_psd,
            )
            tta_bsc_before.append(float(bsc_before.item()))
            tta_bsc_after.append(float(bsc_after.item()))
        else:
            with torch.no_grad():
                out = model(x, a_steering)

        with torch.no_grad():
            bsc_eval = real_augmented_bsc_loss(out["R_ref"], x)
            eval_bsc_sum += float(bsc_eval.item()) * x.size(0)
            eval_bsc_count += int(x.size(0))

            decoded, k_pred_batch = decode_lcpsn_outputs(
                out["mu"],
                out["sigma2"],
                out["K_logits"],
                angle_grid_rad,
                min_separation_deg=8.0,
            )
            y_np = y.cpu().numpy()
            for b in range(y_np.shape[0]):
                true_angles = y_np[b][np.abs(y_np[b] - np.pi) > 1e-6]
                pred_angles = np.array(decoded[b], dtype=np.float32)
                all_k_true.append(len(true_angles))
                all_k_pred.append(int(k_pred_batch[b]))
                all_doa_true.append(np.sort(true_angles))
                all_doa_pred.append(np.sort(pred_angles))

    metrics = summarize_metrics(all_k_true, all_k_pred, all_doa_true, all_doa_pred)
    metrics["eval_bsc_loss"] = float(eval_bsc_sum / max(eval_bsc_count, 1))
    if tta and len(tta_bsc_before) > 0:
        metrics["tta_bsc_before"] = float(np.mean(tta_bsc_before))
        metrics["tta_bsc_after"] = float(np.mean(tta_bsc_after))
    return metrics


def print_metrics_block(metrics: dict):
    print(f"K Accuracy              : {metrics['k_acc']:.4f}")
    print(f"DOA MAE (deg)           : {metrics['doa_mae_deg']:.4f}")
    print(f"DOA RMSE (deg)          : {metrics['doa_rmse_deg']:.4f}")
    print(f"Success Rate @5.0deg    : {metrics['success_rate']:.4f}")
    print(f"DOA MAE (K-correct)     : {metrics['doa_mae_deg_kcorrect']:.4f}")
    print(f"DOA RMSE (K-correct)    : {metrics['doa_rmse_deg_kcorrect']:.4f}")
    print(f"SR @5.0deg K-correct    : {metrics['success_rate_kcorrect']:.4f}")
    print(f"Eval BSC L(R_ref, x)    : {metrics['eval_bsc_loss']:.6f}")
    print("K Confusion Matrix [rows=true K 2~5, cols=pred K 2~5]")
    print(metrics["k_confusion_matrix"])
    if "tta_bsc_before" in metrics:
        print(f"R-TTA BSC before        : {metrics['tta_bsc_before']:.6f}")
        print(f"R-TTA BSC after         : {metrics['tta_bsc_after']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LC-PSN with optional R-TTA.")
    parser.add_argument("--tta", action="store_true", help="Enable covariance-only R-TTA.")
    parser.add_argument("--tta-steps", type=int, default=5, help="Number of R-TTA optimization steps.")
    parser.add_argument("--tta-lr", type=float, default=0.01, help="Learning rate for R-TTA on R_ref.")
    parser.add_argument("--tta-anchor", type=float, default=1.0, help="Anchor weight for ||R_tta - R_init||^2.")
    parser.add_argument("--tta-psd", type=float, default=0.05, help="PSD violation penalty weight for R-TTA.")
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH, help="Checkpoint path.")
    parser.add_argument("--snrs", type=int, nargs="+", default=SNR_LIST, help="SNR list to evaluate.")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42069)
    model = LCPSN(
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

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    state = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    a_steering = precompute_steering_vectors(M, ANGLES_COUNT)
    angle_grid_rad = build_angle_grid_rad(ANGLES_COUNT, device=DEVICE)

    snr_range, k_acc_list = [], []
    doa_mae_rad_list, doa_rmse_rad_list = [], []
    doa_mae_rad_kcorr_list, doa_rmse_rad_kcorr_list = [], []
    eval_bsc_loss_list = []

    print("========== Evaluation (LC-PSN, SNR-wise) ==========")
    if args.tta:
        print(
            f"[INFO] R-TTA enabled: steps={args.tta_steps}, lr={args.tta_lr}, "
            f"anchor={args.tta_anchor}, psd={args.tta_psd}"
        )
    for snr in args.snrs:
        metrics = evaluate_one_snr(
            model,
            a_steering,
            angle_grid_rad,
            snr,
            tta=args.tta,
            tta_steps=args.tta_steps,
            tta_lr=args.tta_lr,
            tta_anchor=args.tta_anchor,
            tta_psd=args.tta_psd,
        )
        print(f"\n================ SNR = {snr} dB ================")
        print_metrics_block(metrics)
        snr_range.append(snr)
        k_acc_list.append(metrics["k_acc"])
        doa_mae_rad_list.append(metrics["doa_mae_rad"])
        doa_rmse_rad_list.append(metrics["doa_rmse_rad"])
        doa_mae_rad_kcorr_list.append(metrics["doa_mae_rad_kcorrect"])
        doa_rmse_rad_kcorr_list.append(metrics["doa_rmse_rad_kcorrect"])
        eval_bsc_loss_list.append(metrics["eval_bsc_loss"])

    np.savez(
        os.path.join(OUT_DIR, "lcpsn_results.npz"),
        snr_range=np.array(snr_range),
        k_acc=np.array(k_acc_list),
        doa_mae_rad=np.array(doa_mae_rad_list),
        doa_rmse_rad=np.array(doa_rmse_rad_list),
        doa_mae_rad_kcorrect=np.array(doa_mae_rad_kcorr_list),
        doa_rmse_rad_kcorrect=np.array(doa_rmse_rad_kcorr_list),
        eval_bsc_loss=np.array(eval_bsc_loss_list),
    )

    plt.figure(figsize=(6, 4))
    plt.plot(snr_range, k_acc_list, marker='o')
    plt.xlabel("SNR [dB]")
    plt.ylabel("K Accuracy")
    plt.title("LC-PSN: K Accuracy vs SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "k_acc_vs_snr.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(snr_range, doa_mae_rad_list, marker='o', label="All")
    plt.plot(snr_range, doa_mae_rad_kcorr_list, marker='s', label="K-correct only")
    plt.xlabel("SNR [dB]")
    plt.ylabel("DOA MAE [rad]")
    plt.title("LC-PSN: DOA MAE vs SNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "doa_mae_rad_vs_snr.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(snr_range, doa_rmse_rad_list, marker='o', label="All")
    plt.plot(snr_range, doa_rmse_rad_kcorr_list, marker='s', label="K-correct only")
    plt.xlabel("SNR [dB]")
    plt.ylabel("DOA RMSE [rad]")
    plt.title("LC-PSN: DOA RMSE vs SNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "doa_rmse_rad_vs_snr.png"), dpi=150)
    plt.close()

    print(f"\nSaved results and plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
