import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_angle_grid_rad(angles_count: int = 360, device=None) -> torch.Tensor:
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
      - warmup 구간에서는 start_sigma 유지
      - 이후 end_sigma까지 선형 감소
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


def complex_cov_to_real_augmented(R_ref: torch.Tensor) -> torch.Tensor:
    """
    Convert complex covariance to real-augmented covariance.

    R_ref: [B, M, M] complex
    returns Sigma: [B, 2M, 2M] real

    Sigma = 0.5 * [[Re(R), -Im(R)],
                   [Im(R),  Re(R)]]
    """
    R_re = R_ref.real
    R_im = R_ref.imag
    top = torch.cat([R_re, -R_im], dim=-1)
    bottom = torch.cat([R_im, R_re], dim=-1)
    return 0.5 * torch.cat([top, bottom], dim=-2)


def covariance_to_correlation(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize real covariance to correlation.

    Sigma: [B, N, N] real
    returns C: [B, N, N] real correlation, clamped for asin stability.
    """
    diag = torch.diagonal(Sigma, dim1=-2, dim2=-1).clamp_min(eps)  # [B,N]
    inv_std = torch.rsqrt(diag)
    C = Sigma * inv_std.unsqueeze(-1) * inv_std.unsqueeze(-2)
    return C.clamp(-1.0 + eps, 1.0 - eps)


def sign_pair_same_probability_from_cov(R_ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute p_pq = P[s_p(t) and s_q(t) have the same sign].

    R_ref: [B, M, M] complex covariance
    returns p_same: [B, 2M, 2M]
    """
    Sigma = complex_cov_to_real_augmented(R_ref)       # [B,2M,2M]
    C = covariance_to_correlation(Sigma, eps=eps)      # [B,2M,2M]
    sign_corr = (2.0 / math.pi) * torch.asin(C)
    p_same = 0.5 * (1.0 + sign_corr)
    return p_same.clamp(eps, 1.0 - eps)


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


def observed_sign_pair_same_events(x: torch.Tensor) -> torch.Tensor:
    """
    Build y_pq(t) from the real-augmented sign vector.

    x: [B, 2M, T], ordered as [sign(Re z_1..M), sign(Im z_1..M)]
    returns y_same: [B, 2M, 2M, T]
    """
    s = torch.sign(x)
    s = torch.where(s == 0, torch.ones_like(s), s)
    pair_prod = s.unsqueeze(2) * s.unsqueeze(1)  # [B,2M,2M,T]
    return (pair_prod > 0).to(x.dtype)


def real_augmented_bsc_loss(
    R_ref: torch.Tensor,
    x: torch.Tensor,
    eps: float = 1e-6,
    exclude_diagonal: bool = True,
    upper_triangle_only: bool = True,
    return_stats: bool = False,
):
    """
    Real-augmented Bernoulli self-consistency loss.

    This follows the one-bit sign-pair likelihood from AGENTS.md:
      s(t) = [sign(Re z_1..M), sign(Im z_1..M)]
      Sigma = 0.5 * [[Re(R), -Im(R)], [Im(R), Re(R)]]
      C = corr(Sigma)
      p_pq = 0.5 * (1 + (2/pi) * asin(C_pq))
      y_pq(t) = 1 if s_p(t) * s_q(t) == +1 else 0

    R_ref: [B, M, M] complex covariance
    x: [B, 2M, T] one-bit real/imag sign snapshots
    """
    if R_ref.ndim != 3 or not torch.is_complex(R_ref):
        raise ValueError("R_ref must be a complex tensor with shape [B, M, M].")
    if x.ndim != 3:
        raise ValueError("x must have shape [B, 2M, T].")
    if x.size(0) != R_ref.size(0) or x.size(1) != 2 * R_ref.size(1):
        raise ValueError("x shape must match R_ref batch and real-augmented dimension.")

    p_same = sign_pair_same_probability_from_cov(R_ref, eps=eps)  # [B,2M,2M]
    y_same = observed_sign_pair_same_events(x).to(p_same.dtype)   # [B,2M,2M,T]
    n = p_same.size(-1)
    pair_mask = torch.ones((n, n), device=R_ref.device, dtype=torch.bool)
    if exclude_diagonal:
        pair_mask.fill_diagonal_(False)
    if upper_triangle_only:
        pair_mask = torch.triu(pair_mask, diagonal=1 if exclude_diagonal else 0)

    p_masked = p_same[:, pair_mask].unsqueeze(-1).expand(-1, -1, y_same.size(-1))
    y_masked = y_same[:, pair_mask, :]
    loss = F.binary_cross_entropy(p_masked, y_masked, reduction="mean")

    if not return_stats:
        return loss

    stats = {
        "bsc_loss": loss.detach(),
        "bsc_p_mean": p_same[:, pair_mask].mean().detach(),
        "bsc_y_mean": y_same[:, pair_mask, :].mean().detach(),
        "bsc_num_pairs": torch.tensor(int(pair_mask.sum().item()), device=R_ref.device),
    }
    return loss, stats


def hermitian_consistency_loss(R_ref: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(R_ref - R_ref.conj().transpose(-1, -2)) ** 2).real


def covariance_alignment_loss(R_ref: torch.Tensor, true_doa: torch.Tensor, m: int) -> torch.Tensor:
    """
    Weak physics regularizer: encourages R_ref to match ideal covariance subspace
    from the true steering manifold. Works on simulation where true DOAs are known.
    """
    device = R_ref.device
    B = true_doa.size(0)
    valid = doa_valid_mask(true_doa)
    losses = []
    array = torch.arange(m, device=device, dtype=torch.float32)

    for b in range(B):
        doas = true_doa[b][valid[b]]
        if doas.numel() == 0:
            continue
        A = torch.exp(-1j * math.pi * array[:, None] * torch.sin(doas)[None, :])  # [M,K]
        R_tgt = A @ A.conj().transpose(-1, -2)
        R_tgt = R_tgt / (torch.trace(R_tgt).real + 1e-8)
        R_hat = R_ref[b] / (torch.trace(R_ref[b]).real + 1e-8)
        losses.append(torch.mean(torch.abs(R_hat - R_tgt) ** 2).real)

    if len(losses) == 0:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


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
