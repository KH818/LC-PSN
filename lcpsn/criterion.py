"""Define the LCPSN training losses, spectrum targets, and DOA decoding helpers."""

import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_angle_grid_rad(angles_count: int = 360, device=None) -> torch.Tensor:
    """Angle grid the TARGET spectrum is built on: CLOSED [-pi/2, +pi/2].

    Spacing is pi/(angles_count-1) = 0.50139 deg, NOT 0.5 deg -- this is a
    different grid from precompute_steering_vectors() in train.py/eval.py, which
    is half-open (endpoint=False, 0.50000 deg) and only feeds the Bartlett prior.

    Anything that turns a spectrum bin index back into an angle must use THIS
    grid, because that is what the network was trained against; decoding on the
    steering grid instead mislabels every bin and injects a systematic offset of
    about -0.21 deg. Keep in sync with unified_eval/adapters.py:ANGLES_DECODE.
    """
    return torch.linspace(-math.pi / 2, math.pi / 2, steps=angles_count, device=device)


def doa_valid_mask(true_doa: torch.Tensor) -> torch.Tensor:
    return torch.abs(true_doa - math.pi) > 1e-6


def make_target_spectrum_batch(true_doa: torch.Tensor, angle_grid_rad: torch.Tensor, sigma_deg: float = 2.0) -> torch.Tensor:
    """
    true_doa: [B, L], padded with pi
    angle_grid_rad: [G]
    returns [B, G]
    """
    sigma = math.radians(float(sigma_deg))
    sigma = max(sigma, 1e-6)
    valid = doa_valid_mask(true_doa)
    outs = []
    for b in range(true_doa.size(0)):
        doas = true_doa[b][valid[b]]
        if doas.numel() == 0:
            outs.append(torch.zeros_like(angle_grid_rad))
            continue
        diff = angle_grid_rad[None, :] - doas[:, None]
        gauss = torch.exp(-0.5 * (diff / sigma) ** 2)
        target = gauss.max(dim=0).values
        outs.append(target.clamp(0.0, 1.0))
    return torch.stack(outs, dim=0)


def get_sigma_deg(epoch: int, total_epochs: int, start_sigma: float = 2.0, end_sigma: float = 1.2, warmup_ratio: float = 0.15) -> float:
    """
    Piecewise-linear curriculum:
      - keep start_sigma during warmup
      - then decrease linearly to end_sigma
    """
    total_epochs = max(int(total_epochs), 1)
    epoch = max(int(epoch), 0)

    warmup_epochs = int(round(total_epochs * warmup_ratio))
    warmup_epochs = min(max(warmup_epochs, 0), total_epochs - 1) if total_epochs > 1 else 0

    if epoch <= warmup_epochs:
        return float(start_sigma)

    decay_epochs = max(total_epochs - 1 - warmup_epochs, 1)
    t = min((epoch - warmup_epochs) / decay_epochs, 1.0)
    return float(start_sigma + (end_sigma - start_sigma) * t)


class SpectrumLossWithFalsePeakPenalty(nn.Module):
    """
    Main spectrum loss:
      1) focal BCE on the soft target spectrum
      2) extra penalty on high responses in regions where target is near zero
    """
    def __init__(
        self,
        pos_weight: float = 10.0,
        gamma: float = 2.0,
        neg_peak_weight: float = 0.35,
        neg_target_threshold: float = 0.05,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.gamma = float(gamma)
        self.neg_peak_weight = float(neg_peak_weight)
        self.neg_target_threshold = float(neg_target_threshold)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
            pos_weight=pos_weight,
        )

        prob = torch.sigmoid(logits)
        pt = target * prob + (1.0 - target) * (1.0 - prob)
        focal = (1.0 - pt).clamp_min(1e-6).pow(self.gamma)
        focal_bce = (focal * bce).mean()

        neg_mask = (target < self.neg_target_threshold).to(logits.dtype)
        false_peak_penalty = (neg_mask * prob.pow(2)).mean()

        total = focal_bce + self.neg_peak_weight * false_peak_penalty
        stats = {
            "focal_bce": focal_bce.detach(),
            "false_peak": false_peak_penalty.detach(),
        }
        return total, stats


def gaussian_nll_spectrum_loss(
    mu: torch.Tensor,
    sigma2: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood for probabilistic spectrum regression.

    mu: [B, G] predicted spectrum mean
    sigma2: [B, G] predicted variance
    target: [B, G] soft target spectrum
    """
    sigma2_safe = sigma2.clamp_min(eps)
    sq_error = (target - mu).pow(2)
    return 0.5 * (sq_error / sigma2_safe + torch.log(sigma2_safe)).mean()


def uncertainty_weighted_covariance_loss(
    R_ref: torch.Tensor,
    R0: torch.Tensor,
    U0: torch.Tensor,
    m: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Reliability-weighted covariance consistency for Stage B.

    R_ref, R0: [B, M, M] complex covariance matrices
    U0: [B, 2M, 2M] real-augmented arcsine covariance uncertainty

    The real-augmented U0 entries are folded into approximate variances for
    Re(R_ij) and Im(R_ij). This keeps U0 structurally active without changing
    the probabilistic interpretation of the neural spectrum heads.
    """
    if U0.shape[-2:] != (2 * m, 2 * m):
        raise ValueError(f"U0 must have trailing shape [{2 * m}, {2 * m}]")

    U_rr = U0[..., :m, :m]
    U_ri = U0[..., :m, m:]
    U_ir = U0[..., m:, :m]
    U_ii = U0[..., m:, m:]
    var_re = 0.25 * (U_rr + U_ii)
    var_im = 0.25 * (U_ir + U_ri)

    diff = R_ref - R0.detach()
    weighted_re = diff.real.pow(2) / var_re.clamp_min(eps)
    weighted_im = diff.imag.pow(2) / var_im.clamp_min(eps)
    return (weighted_re + weighted_im).mean()



def covariance_loss_components(
    R_ref: torch.Tensor,
    R0: torch.Tensor,
    U0: torch.Tensor,
    m: int,
    eps: float = 1e-6,
):
    """Return covariance-consistency pieces, each normalized over all M^2 entries."""
    if U0.shape[-2:] != (2 * m, 2 * m):
        raise ValueError(f"U0 must have trailing shape [{2 * m}, {2 * m}]")

    U_rr = U0[..., :m, :m]
    U_ri = U0[..., :m, m:]
    U_ir = U0[..., m:, :m]
    U_ii = U0[..., m:, m:]
    var_re = 0.25 * (U_rr + U_ii)
    var_im = 0.25 * (U_ir + U_ri)

    diff = R_ref - R0.detach()
    sq_re = diff.real.pow(2)
    sq_im = diff.imag.pow(2)
    weighted = sq_re / var_re.clamp_min(eps) + sq_im / var_im.clamp_min(eps)
    uniform = sq_re + sq_im

    eye = torch.eye(m, device=R_ref.device, dtype=weighted.dtype).view(1, m, m)
    off = 1.0 - eye
    return {
        "diag": (weighted * eye).mean(),
        "off_weighted": (weighted * off).mean(),
        "full_weighted": weighted.mean(),
        "off_uniform": (uniform * off).mean(),
        "full_uniform": uniform.mean(),
    }


def covariance_consistency_loss(
    R_ref: torch.Tensor,
    R0: torch.Tensor,
    U0: torch.Tensor,
    m: int,
    mode: str = "proposed",
    lambda_cov: float = 1e-3,
    lambda_diag: float | None = None,
    lambda_off: float | None = None,
    eps: float = 1e-6,
):
    """Weighted covariance objective used by the ablation plan.

    Modes:
      none         -> 0
      uniform      -> lambda_cov * L_full,u
      diag         -> lambda_cov * L_diag
      diag_uniform -> lambda_diag * L_diag + lambda_off * L_off,u
      proposed     -> lambda_cov * (L_diag + L_off,w)
    """
    mode = str(mode).lower().replace("-", "_")
    comps = covariance_loss_components(R_ref, R0, U0, m, eps=eps)
    zero = comps["full_uniform"].new_zeros(())
    lambda_diag = float(lambda_cov if lambda_diag is None else lambda_diag)
    lambda_off = float(lambda_cov if lambda_off is None else lambda_off)
    lambda_cov = float(lambda_cov)

    if mode in {"none", "off", "no", "zero"}:
        loss = zero
    elif mode in {"proposed", "weighted", "uncov", "uncovon"}:
        loss = lambda_cov * (comps["diag"] + comps["off_weighted"])
    elif mode in {"uniform", "full_uniform", "l2"}:
        loss = lambda_cov * comps["full_uniform"]
    elif mode in {"diag", "diagonal", "diag_only"}:
        loss = lambda_cov * comps["diag"]
    elif mode in {"diag_uniform", "same_diag_uniform_off", "uniform_off"}:
        loss = lambda_diag * comps["diag"] + lambda_off * comps["off_uniform"]
    else:
        raise ValueError(f"unknown covariance loss mode {mode!r}")

    stats = dict(comps)
    stats["cov_weighted"] = loss
    return loss, stats


def angle_wrap_diff_rad(a, b):
    return (a - b + math.pi / 2) % math.pi - math.pi / 2


def decode_spectrum_peaks(
    spectrum_prob: torch.Tensor,
    angle_grid_rad: torch.Tensor,
    threshold: float = 0.22,
    relative_threshold: float = 0.60,
    min_separation_deg: float = 5.0,
    max_sources: int = 5,
    min_sources: int = 2,
):
    """
    spectrum_prob: [B,G] or [G]
    returns list[list[float]] angles in radians
    """
    if spectrum_prob.ndim == 1:
        spectrum_prob = spectrum_prob.unsqueeze(0)

    min_sep = math.radians(min_separation_deg)
    results = []

    for spec in spectrum_prob:
        spec_np = spec.detach().cpu().numpy()
        grid_np = angle_grid_rad.detach().cpu().numpy()

        top_val = float(spec_np.max())
        thr = max(threshold, relative_threshold * top_val)

        candidate_idx = []
        for i in range(1, len(spec_np) - 1):
            if (
                spec_np[i] >= thr
                and spec_np[i] > spec_np[i - 1]
                and spec_np[i] > spec_np[i + 1]
            ):
                candidate_idx.append(i)

        if len(candidate_idx) == 0:
            candidate_idx = [int(np.argmax(spec_np))]

        selected = []
        for idx in sorted(candidate_idx, key=lambda x: spec_np[x], reverse=True):
            if all(abs(grid_np[idx] - grid_np[j]) >= min_sep for j in selected):
                selected.append(idx)
            if len(selected) >= max_sources:
                break

        if len(selected) < min_sources:
            for idx in list(np.argsort(spec_np)[::-1]):
                if idx in selected:
                    continue
                if all(abs(grid_np[idx] - grid_np[j]) >= min_sep for j in selected):
                    selected.append(idx)
                if len(selected) >= min_sources:
                    break

        selected = sorted(selected)
        results.append([float(grid_np[i]) for i in selected[:max_sources]])

    return results


def permutation_best_errors_deg(pred, true):
    if len(pred) == 0 or len(true) == 0:
        return np.nan, np.nan, np.array([])
    n = min(len(pred), len(true))
    best_rmse = None
    best_mae = None
    best_pred = None
    for perm in itertools.permutations(range(len(pred)), n):
        p = np.array(pred)[list(perm)]
        diff = angle_wrap_diff_rad(np.array(true[:n]), p)
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_pred = p.copy()
    return np.rad2deg(best_mae), np.rad2deg(best_rmse), best_pred
