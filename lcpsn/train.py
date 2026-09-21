"""Train LCPSN with the paper data distribution and multi-task objective."""

import os
import math
import random
import h5py
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from model import LCPSN
from dataset import get_dataloader, create_complete_dataset
from criterion import (
    build_angle_grid_rad,
    make_target_spectrum_batch,
    get_sigma_deg,
    SpectrumLossWithFalsePeakPenalty,
    gaussian_nll_spectrum_loss,
    uncertainty_weighted_covariance_loss,
    covariance_consistency_loss,
)
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
LR = 2e-4
EPOCHS = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42069

NUM_TRAIN = 50000
NUM_VAL = 5000

TRAIN_FILE = "train_mini_1bit.h5"
VAL_FILE = "val_mini_1bit.h5"

ANGLES_COUNT = 360
MIN_SEP_DEG = 8.0

SIGMA_START = 2.0
SIGMA_END = 1.2
SIGMA_WARMUP_RATIO = 0.15

POS_WEIGHT = 10.0
FOCAL_GAMMA = 2.0
NEG_PEAK_WEIGHT = 0.35
NEG_TARGET_THRESHOLD = 0.05

K_MIN = 2
K_MAX = 5
LAMBDA_K = 0.5
LAMBDA_GNLL = 0.01
LAMBDA_VAR = 5e-3
LAMBDA_UNCOV = 0.001
VARIANCE_FLOOR = 5e-2
RUN_NAME = "lcpsn_uncov0001"


def precompute_steering_vectors(m, angles_count=360):
    angles = np.linspace(-np.pi / 2, np.pi / 2, angles_count, endpoint=False)
    a_matrix = torch.zeros((m, angles_count), dtype=torch.complex64)
    for i, ang in enumerate(angles):
        a_matrix[:, i] = torch.from_numpy(ULA_action_vector(ang, m)).to(torch.complex64)
    return a_matrix.to(DEVICE)


def create_snr_mix_h5(
    out_name,
    total_size,
    snapshots,
    snr_list=(0, 5, 10),
    num_sources=[2, 3, 4, 5],
    coherent=False,
    seed=1234,
    min_sep_deg=MIN_SEP_DEG,
):
    rng = np.random.default_rng(seed)

    per = total_size // len(snr_list)
    sizes = [per] * len(snr_list)
    sizes[-1] += total_size - sum(sizes)

    Xs, Ys = [], []
    for snr, sz in zip(snr_list, sizes):
        X_part, Y_part = create_complete_dataset(
            name=f"_tmp_{out_name}_snr{snr}",
            size=sz,
            snr=snr,
            snapshots=snapshots,
            m=M,
            num_sources=num_sources,
            coherent=coherent,
            save=False,
            min_sep_deg=min_sep_deg,
        )
        Xs.append(X_part)
        Ys.append(Y_part)

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    idx = rng.permutation(total_size)
    X = X[idx]
    Y = Y[idx]

    with h5py.File(f"{out_name}.h5", "w") as hf:
        hf.create_dataset("X", data=X)
        hf.create_dataset("Y", data=Y)

    print(
        f"[OK] Saved SNR-mix dataset: {out_name}.h5 | "
        f"size={total_size} | snr_list={snr_list} | snapshots={int(snapshots)}"
    )


def make_k_class_labels(true_doa: torch.Tensor, k_min: int = K_MIN, k_max: int = K_MAX) -> torch.Tensor:
    valid = torch.abs(true_doa - math.pi) > 1e-6
    k_true = valid.sum(dim=1).long().clamp(min=k_min, max=k_max)
    return k_true - k_min


def average_num_peaks(spectrum: torch.Tensor, threshold: float) -> torch.Tensor:
    if spectrum.size(-1) < 3:
        return torch.zeros((), device=spectrum.device)
    mid = spectrum[:, 1:-1]
    peaks = (mid > spectrum[:, :-2]) & (mid > spectrum[:, 2:]) & (mid >= threshold)
    return peaks.sum(dim=1).float().mean()


def lcpsn_loss(
    out,
    y,
    target_spec,
    spectrum_criterion,
    lambda_uncov: float = LAMBDA_UNCOV,
    lambda_gnll: float = LAMBDA_GNLL,
    lambda_var: float = LAMBDA_VAR,
    cov_mode: str = "proposed",
    lambda_diag: float | None = None,
    lambda_off: float | None = None,
):
    """Total objective. The three lambdas are arguments rather than constants so
    each auxiliary term can be switched off from the command line for ablation;
    the heads themselves are always built and run, so `--lambda-gnll 0
    --lambda-var 0` removes the training signal without changing the network."""
    l_spec, spec_stats = spectrum_criterion(out["mu_logits"], target_spec)
    l_gnll = gaussian_nll_spectrum_loss(out["mu"], out["sigma2"], target_spec)
    k_labels = make_k_class_labels(y)
    l_k = F.cross_entropy(out["K_logits"], k_labels, label_smoothing=0.1)
    l_var = out["log_var"].pow(2).mean()
    l_cov_weighted, cov_stats = covariance_consistency_loss(
        out["R_ref"], out["R0"], out["U0"], M,
        mode=cov_mode,
        lambda_cov=lambda_uncov,
        lambda_diag=lambda_diag,
        lambda_off=lambda_off,
    )
    l_uncov = cov_stats["full_weighted"]

    loss = (
        l_spec
        + lambda_gnll * l_gnll
        + LAMBDA_K * l_k
        + lambda_var * l_var
        + l_cov_weighted
    )
    monitor = l_spec + LAMBDA_K * l_k
    stats = {
        "spec": l_spec.detach(),
        "gnll": l_gnll.detach(),
        "monitor": monitor.detach(),
        "focal_bce": spec_stats["focal_bce"].detach(),
        "false_peak": spec_stats["false_peak"].detach(),
        "k": l_k.detach(),
        "var": l_var.detach(),
        "uncov": l_uncov.detach(),
        "cov_weighted": l_cov_weighted.detach(),
        "cov_diag": cov_stats["diag"].detach(),
        "cov_off_weighted": cov_stats["off_weighted"].detach(),
        "cov_full_uniform": cov_stats["full_uniform"].detach(),
        "cov_off_uniform": cov_stats["off_uniform"].detach(),
    }
    return loss, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train LCPSN with U0 uncertainty-weighted covariance consistency.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-file", type=str, default=TRAIN_FILE)
    parser.add_argument("--val-file", type=str, default=VAL_FILE)
    parser.add_argument("--run-name", type=str, default=RUN_NAME)
    parser.add_argument("--lambda-uncov", type=float, default=LAMBDA_UNCOV)
    parser.add_argument("--cov-mode", choices=["none", "uniform", "diag", "diag_uniform", "proposed"],
                        default="proposed", help="covariance ablation mode")
    parser.add_argument("--lambda-diag", type=float, default=None,
                        help="diagonal covariance coefficient for diag_uniform mode")
    parser.add_argument("--lambda-off", type=float, default=None,
                        help="off-diagonal covariance coefficient for diag_uniform mode")
    parser.add_argument("--lambda-gnll", type=float, default=LAMBDA_GNLL,
                        help="weight of the Gaussian-NLL spectrum term; 0 ablates it")
    parser.add_argument("--lambda-var", type=float, default=LAMBDA_VAR,
                        help="weight of the variance regularizer; 0 ablates it")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--metric-every", type=int, default=5,
                        help="Every N epochs, decode K-acc/MAE on validation batches for epoch curves.")
    parser.add_argument("--metric-batches", type=int, default=8,
                        help="Number of validation batches used for the periodic decode metrics.")
    parser.add_argument("--snr-list", type=int, nargs="+", default=[0, 5, 10],
                        help="Training/validation SNR mix (dB).")
    parser.add_argument("--min-sep-deg", type=float, default=MIN_SEP_DEG,
                        help="Min DoA separation for data gen (0 = unconstrained).")
    parser.add_argument("--num-train", type=int, default=NUM_TRAIN)
    parser.add_argument("--num-val", type=int, default=NUM_VAL)
    return parser.parse_args()


def _decode_topk_peaks(spec_np, grid_np, k, min_sep_rad):
    """Top-k separated local maxima of a spectrum (mirrors eval.py decoding)."""
    cand = [i for i in range(1, len(spec_np) - 1)
            if spec_np[i] > spec_np[i - 1] and spec_np[i] > spec_np[i + 1]]
    ranked = sorted(cand, key=lambda i: spec_np[i], reverse=True) or list(np.argsort(spec_np)[::-1])
    sel = []
    for idx in ranked:
        if all(abs(grid_np[idx] - grid_np[j]) >= min_sep_rad for j in sel):
            sel.append(idx)
        if len(sel) >= k:
            break
    for idx in np.argsort(spec_np)[::-1]:
        if len(sel) >= k:
            break
        if idx not in sel and all(abs(grid_np[idx] - grid_np[j]) >= min_sep_rad for j in sel):
            sel.append(int(idx))
    return [grid_np[i] for i in sorted(sel)]


@torch.no_grad()
def quick_val_decode(model, loader, a_steering, angle_grid_rad, max_batches=8):
    """Lightweight K-accuracy / permutation MAE on a few validation batches."""
    from criterion import permutation_best_errors_deg
    model.eval()
    grid_np = angle_grid_rad.detach().cpu().numpy()
    k_corr, n, mae_list = 0, 0, []
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(DEVICE)
        out = model(x, a_steering)
        k_hat = torch.argmax(out["K_logits"], dim=-1).cpu().numpy() + K_MIN
        mu = out["mu"].cpu().numpy()
        y_np = y.numpy()
        for b in range(y_np.shape[0]):
            true = y_np[b][np.abs(y_np[b] - math.pi) > 1e-6]
            k = int(np.clip(k_hat[b], K_MIN, K_MAX))
            pred = _decode_topk_peaks(mu[b], grid_np, k, math.radians(8.0))
            k_corr += int(k == len(true)); n += 1
            mae, _, _ = permutation_best_errors_deg(pred, list(np.sort(true)))
            if not np.isnan(mae):
                mae_list.append(mae)
    return {"val_k_acc": k_corr / max(n, 1),
            "val_mae_deg": float(np.nanmean(mae_list)) if mae_list else float("nan")}


def train(args=None):
    if args is None:
        args = parse_args()

    seed_everything(args.seed)

    if not os.path.exists(args.train_file):
        print(f"Creating training dataset ({args.num_train} samples) | "
              f"snr={args.snr_list} min_sep={args.min_sep_deg} deg...")
        create_snr_mix_h5(
            out_name=os.path.splitext(args.train_file)[0],
            total_size=args.num_train,
            snapshots=SNAPSHOTS,
            snr_list=tuple(args.snr_list),
            num_sources=[2, 3, 4, 5],
            coherent=False,
            seed=1234,
            min_sep_deg=args.min_sep_deg,
        )

    if not os.path.exists(args.val_file):
        print(f"Creating validation dataset ({args.num_val} samples) | "
              f"snr={args.snr_list} min_sep={args.min_sep_deg} deg...")
        create_snr_mix_h5(
            out_name=os.path.splitext(args.val_file)[0],
            total_size=args.num_val,
            snapshots=SNAPSHOTS,
            snr_list=tuple(args.snr_list),
            num_sources=[2, 3, 4, 5],
            coherent=False,
            seed=4321,
            min_sep_deg=args.min_sep_deg,
        )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = get_dataloader(
        args.train_file,
        batch_size=args.batch_size,
        shuffle=True,
        use_1bit=True,
        generator=train_generator,
    )
    valid_loader = get_dataloader(
        args.val_file,
        batch_size=args.batch_size,
        shuffle=False,
        use_1bit=True,
    )

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = SpectrumLossWithFalsePeakPenalty(
        pos_weight=POS_WEIGHT,
        gamma=FOCAL_GAMMA,
        neg_peak_weight=NEG_PEAK_WEIGHT,
        neg_target_threshold=NEG_TARGET_THRESHOLD,
    )

    a_steering = precompute_steering_vectors(M, ANGLES_COUNT)
    angle_grid_rad = build_angle_grid_rad(ANGLES_COUNT, device=DEVICE)

    best_val_monitor = float("inf")
    print(
        f"[CONFIG] train_file={args.train_file} val_file={args.val_file} "
        f"run_name={args.run_name} seed={args.seed} "
        f"lambda_uncov={args.lambda_uncov} cov_mode={args.cov_mode} "
        f"lambda_diag={args.lambda_diag} lambda_off={args.lambda_off}"
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] total parameters = {n_params:,}")

    import csv as _csv
    log_path = f"train_log_{args.run_name}.csv"
    log_f = open(log_path, "w", newline="")
    log_w = _csv.writer(log_f)
    log_w.writerow(["epoch", "sigma_deg",
                    "train_total", "train_monitor", "train_spec", "train_gnll",
                    "train_k", "train_var", "train_uncov", "train_cov_weighted",
                    "val_total", "val_monitor", "val_spec", "val_gnll",
                    "val_k", "val_var", "val_uncov", "val_cov_weighted",
                    "val_k_acc", "val_mae_deg"])
    log_f.flush()
    print(f"[LOG] per-epoch CSV -> {log_path}")

    for epoch in range(args.epochs):
        sigma_deg = get_sigma_deg(
            epoch=epoch,
            total_epochs=args.epochs,
            start_sigma=SIGMA_START,
            end_sigma=SIGMA_END,
            warmup_ratio=SIGMA_WARMUP_RATIO,
        )

        model.train()
        train_total = 0.0
        train_monitor = 0.0
        train_spec = 0.0
        train_gnll = 0.0
        train_focal = 0.0
        train_fp = 0.0
        train_k = 0.0
        train_var = 0.0
        train_uncov = 0.0
        train_cov_weighted = 0.0

        for batch in train_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            target_spec = make_target_spectrum_batch(y, angle_grid_rad, sigma_deg=sigma_deg)

            optimizer.zero_grad()
            out = model(x, a_steering)
            loss, loss_stats = lcpsn_loss(
                out,
                y,
                target_spec,
                criterion,
                lambda_uncov=args.lambda_uncov,
                lambda_gnll=args.lambda_gnll,
                lambda_var=args.lambda_var,
                cov_mode=args.cov_mode,
                lambda_diag=args.lambda_diag,
                lambda_off=args.lambda_off,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += loss.item()
            train_monitor += loss_stats["monitor"].item()
            train_spec += loss_stats["spec"].item()
            train_gnll += loss_stats["gnll"].item()
            train_focal += loss_stats["focal_bce"].item()
            train_fp += loss_stats["false_peak"].item()
            train_k += loss_stats["k"].item()
            train_var += loss_stats["var"].item()
            train_uncov += loss_stats["uncov"].item()
            train_cov_weighted += loss_stats["cov_weighted"].item()

        scheduler.step()

        model.eval()
        val_total = 0.0
        val_monitor = 0.0
        val_spec = 0.0
        val_gnll = 0.0
        val_focal = 0.0
        val_fp = 0.0
        val_k = 0.0
        val_var = 0.0
        val_uncov = 0.0
        val_cov_weighted = 0.0
        val_common_monitor = 0.0
        val_spectrum_max = 0.0
        val_spectrum_mean = 0.0
        val_peaks_015 = 0.0
        val_peaks_030 = 0.0
        val_conf_mean = 0.0
        val_conf_max = 0.0
        val_sigma2_mean = 0.0
        val_u0_mean = 0.0
        val_u0_median = 0.0
        val_u0_min = 0.0
        val_u0_max = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                target_spec = make_target_spectrum_batch(y, angle_grid_rad, sigma_deg=sigma_deg)
                target_spec_common = make_target_spectrum_batch(y, angle_grid_rad, sigma_deg=SIGMA_END)

                out = model(x, a_steering)
                loss, loss_stats = lcpsn_loss(
                    out,
                    y,
                    target_spec,
                    criterion,
                    lambda_uncov=args.lambda_uncov,
                    lambda_gnll=args.lambda_gnll,
                    lambda_var=args.lambda_var,
                    cov_mode=args.cov_mode,
                    lambda_diag=args.lambda_diag,
                    lambda_off=args.lambda_off,
                )
                val_total += loss.item()
                val_monitor += loss_stats["monitor"].item()
                val_spec += loss_stats["spec"].item()
                val_gnll += loss_stats["gnll"].item()
                val_focal += loss_stats["focal_bce"].item()
                val_fp += loss_stats["false_peak"].item()
                val_k += loss_stats["k"].item()
                val_var += loss_stats["var"].item()
                val_uncov += loss_stats["uncov"].item()
                val_cov_weighted += loss_stats["cov_weighted"].item()
                common_loss, common_stats = lcpsn_loss(
                    out, y, target_spec_common, criterion,
                    lambda_uncov=args.lambda_uncov,
                    lambda_gnll=args.lambda_gnll,
                    lambda_var=args.lambda_var,
                    cov_mode=args.cov_mode,
                    lambda_diag=args.lambda_diag,
                    lambda_off=args.lambda_off,
                )
                val_common_monitor += common_stats["monitor"].item()

                spectrum = out["mu"]
                pred_confs = torch.softmax(out["K_logits"], dim=-1).max(dim=-1).values
                val_spectrum_max += spectrum.max(dim=-1).values.mean().item()
                val_spectrum_mean += spectrum.mean().item()
                val_peaks_015 += average_num_peaks(spectrum, threshold=0.15).item()
                val_peaks_030 += average_num_peaks(spectrum, threshold=0.30).item()
                val_conf_mean += pred_confs.mean().item()
                val_conf_max += pred_confs.max().item()
                val_sigma2_mean += out["sigma2"].mean().item()
                val_u0_mean += out["U0"].mean().item()
                val_u0_median += out["U0"].median().item()
                val_u0_min += out["U0"].amin().item()
                val_u0_max += out["U0"].amax().item()

        ntr = max(len(train_loader), 1)
        nva = max(len(valid_loader), 1)
        avg_val = val_total / nva
        avg_val_monitor = val_common_monitor / nva

        print(
            f"[Epoch {epoch:03d}] "
            f"sigma={sigma_deg:.3f} | "
            f"Train total={train_total/ntr:.4f} monitor={train_monitor/ntr:.4f} "
            f"spec={train_spec/ntr:.4f} gnll={train_gnll/ntr:.4f} "
            f"(focal={train_focal/ntr:.4f}, fp={train_fp/ntr:.4f}) "
            f"k={train_k/ntr:.4f} var={train_var/ntr:.4f} "
            f"uncov={train_uncov/ntr:.4f} covW={train_cov_weighted/ntr:.4f} | "
            f"Val total={avg_val:.4f} common_monitor={avg_val_monitor:.4f} "
            f"spec={val_spec/nva:.4f} gnll={val_gnll/nva:.4f} "
            f"(focal={val_focal/nva:.4f}, fp={val_fp/nva:.4f}) "
            f"k={val_k/nva:.4f} var={val_var/nva:.4f} "
            f"uncov={val_uncov/nva:.4f} covW={val_cov_weighted/nva:.4f} | "
            f"dbg smax={val_spectrum_max/nva:.4f} smean={val_spectrum_mean/nva:.4f} "
            f"pk015={val_peaks_015/nva:.2f} pk030={val_peaks_030/nva:.2f} "
            f"conf_mean={val_conf_mean/nva:.4f} conf_max={val_conf_max/nva:.4f} "
            f"sigma2={val_sigma2_mean/nva:.4f} "
            f"u0_mean={val_u0_mean/nva:.6f} u0_med={val_u0_median/nva:.6f} "
            f"u0_min={val_u0_min/nva:.6f} u0_max={val_u0_max/nva:.6f}"
        )

        dm = {"val_k_acc": float("nan"), "val_mae_deg": float("nan")}
        if (epoch + 1) % args.metric_every == 0 or epoch == args.epochs - 1:
            dm = quick_val_decode(model, valid_loader, a_steering, angle_grid_rad,
                                  max_batches=args.metric_batches)
            print(f"    [decode] val K_acc={dm['val_k_acc']:.4f} "
                  f"val MAE={dm['val_mae_deg']:.4f} deg", flush=True)

        log_w.writerow([epoch, f"{sigma_deg:.4f}",
                        f"{train_total/ntr:.6f}", f"{train_monitor/ntr:.6f}",
                        f"{train_spec/ntr:.6f}", f"{train_gnll/ntr:.6f}",
                        f"{train_k/ntr:.6f}", f"{train_var/ntr:.6f}",
                        f"{train_uncov/ntr:.6f}", f"{train_cov_weighted/ntr:.6f}",
                        f"{avg_val:.6f}", f"{avg_val_monitor:.6f}",
                        f"{val_spec/nva:.6f}", f"{val_gnll/nva:.6f}",
                        f"{val_k/nva:.6f}", f"{val_var/nva:.6f}",
                        f"{val_uncov/nva:.6f}", f"{val_cov_weighted/nva:.6f}",
                        f"{dm['val_k_acc']:.6f}", f"{dm['val_mae_deg']:.6f}"])
        log_f.flush()

        if avg_val_monitor < best_val_monitor:
            best_val_monitor = avg_val_monitor
            torch.save(model.state_dict(), f"best_{args.run_name}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_monitor": best_val_monitor,
                "args": vars(args),
            }, f"best_{args.run_name}.ckpt")
            print("=> Model Saved")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": args.epochs - 1,
        "best_val_monitor": best_val_monitor,
        "args": vars(args),
    }, f"last_{args.run_name}.ckpt")
    log_f.close()
    print(f"[LOG] epoch curves saved to {log_path}")


if __name__ == "__main__":
    train(parse_args())
