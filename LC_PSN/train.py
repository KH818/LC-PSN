import os
import math
import random
import h5py
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
    covariance_alignment_loss,
    hermitian_consistency_loss,
    real_augmented_bsc_loss,
    gaussian_nll_spectrum_loss,
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
LAMBDA_BSC_MAX = 0.0
BSC_START_EPOCH = 15
BSC_WARMUP_EPOCHS = 30
LAMBDA_COV = 0.0
LAMBDA_HERM = 0.0
LAMBDA_PSD = 0.05
VARIANCE_FLOOR = 5e-2


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
            min_sep_deg=MIN_SEP_DEG,
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

    print(f"[OK] Saved SNR-mix dataset: {out_name}.h5 | size={total_size} | snr_list={snr_list}")


def make_k_class_labels(true_doa: torch.Tensor, k_min: int = K_MIN, k_max: int = K_MAX) -> torch.Tensor:
    valid = torch.abs(true_doa - math.pi) > 1e-6
    k_true = valid.sum(dim=1).long().clamp(min=k_min, max=k_max)
    return k_true - k_min


def get_bsc_weight(epoch: int) -> float:
    if LAMBDA_BSC_MAX <= 0.0 or epoch < BSC_START_EPOCH:
        return 0.0
    ramp = (epoch - BSC_START_EPOCH + 1) / max(BSC_WARMUP_EPOCHS, 1)
    return float(LAMBDA_BSC_MAX * min(max(ramp, 0.0), 1.0))


def average_num_peaks(spectrum: torch.Tensor, threshold: float) -> torch.Tensor:
    if spectrum.size(-1) < 3:
        return torch.zeros((), device=spectrum.device)
    mid = spectrum[:, 1:-1]
    peaks = (mid > spectrum[:, :-2]) & (mid > spectrum[:, 2:]) & (mid >= threshold)
    return peaks.sum(dim=1).float().mean()


def psd_violation(R_ref: torch.Tensor) -> torch.Tensor:
    R_herm = 0.5 * (R_ref + R_ref.conj().transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(R_herm).real
    min_eigs = eigvals.min(dim=-1).values
    return (-min_eigs).clamp_min(0.0).mean()


def lcpsn_loss(out, x, y, target_spec, spectrum_criterion, lambda_bsc: float):
    l_spec, spec_stats = spectrum_criterion(out["mu_logits"], target_spec)
    l_gnll = gaussian_nll_spectrum_loss(out["mu"], out["sigma2"], target_spec)
    k_labels = make_k_class_labels(y)
    l_k = F.cross_entropy(out["K_logits"], k_labels, label_smoothing=0.1)
    l_bsc = real_augmented_bsc_loss(out["R_ref"], x)
    l_var = out["log_var"].pow(2).mean()
    l_cov = covariance_alignment_loss(out["R_ref"], y, M)
    l_herm = hermitian_consistency_loss(out["R_ref"])
    l_psd = psd_violation(out["R_ref"])

    loss = (
        l_spec
        + LAMBDA_GNLL * l_gnll
        + LAMBDA_K * l_k
        + lambda_bsc * l_bsc
        + LAMBDA_VAR * l_var
        + LAMBDA_COV * l_cov
        + LAMBDA_HERM * l_herm
        + LAMBDA_PSD * l_psd
    )
    monitor = (
        l_spec
        + LAMBDA_K * l_k
        + lambda_bsc * l_bsc
        + LAMBDA_COV * l_cov
        + LAMBDA_HERM * l_herm
        + LAMBDA_PSD * l_psd
    )
    stats = {
        "spec": l_spec.detach(),
        "gnll": l_gnll.detach(),
        "monitor": monitor.detach(),
        "focal_bce": spec_stats["focal_bce"].detach(),
        "false_peak": spec_stats["false_peak"].detach(),
        "k": l_k.detach(),
        "bsc": l_bsc.detach(),
        "var": l_var.detach(),
        "cov": l_cov.detach(),
        "herm": l_herm.detach(),
        "psd": l_psd.detach(),
    }
    return loss, stats


def train():
    seed_everything(42069)

    if not os.path.exists(TRAIN_FILE):
        print(f"Creating mini training dataset ({NUM_TRAIN} samples)...")
        create_snr_mix_h5(
            out_name="train_mini_1bit",
            total_size=NUM_TRAIN,
            snapshots=SNAPSHOTS,
            snr_list=(0, 5, 10),
            num_sources=[2, 3, 4, 5],
            coherent=False,
            seed=1234,
        )

    if not os.path.exists(VAL_FILE):
        print(f"Creating mini validation dataset ({NUM_VAL} samples)...")
        create_snr_mix_h5(
            out_name="val_mini_1bit",
            total_size=NUM_VAL,
            snapshots=SNAPSHOTS,
            snr_list=(0, 5, 10),
            num_sources=[2, 3, 4, 5],
            coherent=False,
            seed=4321,
        )

    train_loader = get_dataloader(TRAIN_FILE, batch_size=BATCH_SIZE, shuffle=True, use_1bit=True)
    valid_loader = get_dataloader(VAL_FILE, batch_size=BATCH_SIZE, shuffle=False, use_1bit=True)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = SpectrumLossWithFalsePeakPenalty(
        pos_weight=POS_WEIGHT,
        gamma=FOCAL_GAMMA,
        neg_peak_weight=NEG_PEAK_WEIGHT,
        neg_target_threshold=NEG_TARGET_THRESHOLD,
    )

    a_steering = precompute_steering_vectors(M, ANGLES_COUNT)
    angle_grid_rad = build_angle_grid_rad(ANGLES_COUNT, device=DEVICE)

    best_val_monitor = float("inf")

    for epoch in range(EPOCHS):
        lambda_bsc = get_bsc_weight(epoch)
        sigma_deg = get_sigma_deg(
            epoch=epoch,
            total_epochs=EPOCHS,
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
        train_bsc = 0.0
        train_var = 0.0
        train_cov = 0.0
        train_herm = 0.0
        train_psd = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            target_spec = make_target_spectrum_batch(y, angle_grid_rad, sigma_deg=sigma_deg)

            optimizer.zero_grad()
            out = model(x, a_steering)
            loss, loss_stats = lcpsn_loss(out, x, y, target_spec, criterion, lambda_bsc)
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
            train_bsc += loss_stats["bsc"].item()
            train_var += loss_stats["var"].item()
            train_cov += loss_stats["cov"].item()
            train_herm += loss_stats["herm"].item()
            train_psd += loss_stats["psd"].item()

        scheduler.step()

        model.eval()
        val_total = 0.0
        val_monitor = 0.0
        val_spec = 0.0
        val_gnll = 0.0
        val_focal = 0.0
        val_fp = 0.0
        val_k = 0.0
        val_bsc = 0.0
        val_var = 0.0
        val_cov = 0.0
        val_herm = 0.0
        val_psd_loss = 0.0
        val_spectrum_max = 0.0
        val_spectrum_mean = 0.0
        val_peaks_015 = 0.0
        val_peaks_030 = 0.0
        val_conf_mean = 0.0
        val_conf_max = 0.0
        val_sigma2_mean = 0.0
        val_psd = 0.0

        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                target_spec = make_target_spectrum_batch(y, angle_grid_rad, sigma_deg=sigma_deg)

                out = model(x, a_steering)
                loss, loss_stats = lcpsn_loss(out, x, y, target_spec, criterion, lambda_bsc)
                val_total += loss.item()
                val_monitor += loss_stats["monitor"].item()
                val_spec += loss_stats["spec"].item()
                val_gnll += loss_stats["gnll"].item()
                val_focal += loss_stats["focal_bce"].item()
                val_fp += loss_stats["false_peak"].item()
                val_k += loss_stats["k"].item()
                val_bsc += loss_stats["bsc"].item()
                val_var += loss_stats["var"].item()
                val_cov += loss_stats["cov"].item()
                val_herm += loss_stats["herm"].item()
                val_psd_loss += loss_stats["psd"].item()

                spectrum = out["mu"]
                pred_confs = torch.softmax(out["K_logits"], dim=-1).max(dim=-1).values
                val_spectrum_max += spectrum.max(dim=-1).values.mean().item()
                val_spectrum_mean += spectrum.mean().item()
                val_peaks_015 += average_num_peaks(spectrum, threshold=0.15).item()
                val_peaks_030 += average_num_peaks(spectrum, threshold=0.30).item()
                val_conf_mean += pred_confs.mean().item()
                val_conf_max += pred_confs.max().item()
                val_sigma2_mean += out["sigma2"].mean().item()
                val_psd += psd_violation(out["R_ref"]).item()

        ntr = max(len(train_loader), 1)
        nva = max(len(valid_loader), 1)
        avg_val = val_total / nva
        avg_val_monitor = val_monitor / nva

        print(
            f"[Epoch {epoch:03d}] "
            f"sigma={sigma_deg:.3f} lambda_bsc={lambda_bsc:.4f} | "
            f"Train total={train_total/ntr:.4f} monitor={train_monitor/ntr:.4f} "
            f"spec={train_spec/ntr:.4f} gnll={train_gnll/ntr:.4f} "
            f"(focal={train_focal/ntr:.4f}, fp={train_fp/ntr:.4f}) "
            f"k={train_k/ntr:.4f} bsc={train_bsc/ntr:.4f} var={train_var/ntr:.4f} "
            f"cov={train_cov/ntr:.4f} herm={train_herm/ntr:.6f} psd={train_psd/ntr:.6f} | "
            f"Val total={avg_val:.4f} monitor={avg_val_monitor:.4f} "
            f"spec={val_spec/nva:.4f} gnll={val_gnll/nva:.4f} "
            f"(focal={val_focal/nva:.4f}, fp={val_fp/nva:.4f}) "
            f"k={val_k/nva:.4f} bsc={val_bsc/nva:.4f} var={val_var/nva:.4f} "
            f"cov={val_cov/nva:.4f} herm={val_herm/nva:.6f} psd={val_psd_loss/nva:.6f} | "
            f"dbg smax={val_spectrum_max/nva:.4f} smean={val_spectrum_mean/nva:.4f} "
            f"pk015={val_peaks_015/nva:.2f} pk030={val_peaks_030/nva:.2f} "
            f"conf_mean={val_conf_mean/nva:.4f} conf_max={val_conf_max/nva:.4f} "
            f"sigma2={val_sigma2_mean/nva:.4f} psd_viol={val_psd/nva:.6f}"
        )

        if avg_val_monitor < best_val_monitor:
            best_val_monitor = avg_val_monitor
            torch.save(model.state_dict(), "best_lcpsn_1bit.pt")
            print("=> Model Saved")


if __name__ == "__main__":
    train()
