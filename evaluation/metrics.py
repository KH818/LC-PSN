"""
Canonical metrics — applied IDENTICALLY to all four models so the comparison is fair.

RMSPE definition (K-hat decoding, the unknown-K deployment scenario):
    * pred : the K_hat angles the model actually decoded (regressor slice or spectrum peaks)
    * true : the K_true ground-truth angles
    * match n = min(K_hat, K_true) angles by the permutation that minimises error
    * angular error uses mod-pi wrapping to [-pi/2, +pi/2]  (periodicity, same as RMSPE loss)
    * value returned in RADIANS  (deg = rad * 180/pi)

This is the single agreed metric ("K-hat decoding only"); it is scored the same way
for Trans / TCN / DA (regressor outputs) and LCPSN (spectrum peaks).
"""
import itertools
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:                                  # metric still usable without torch
    torch = None


def wrap_pi(d):
    """Wrap angular differences (rad) to [-pi/2, +pi/2]."""
    return (d + np.pi / 2.0) % np.pi - np.pi / 2.0


def rmspe_rad(pred, true):
    """Permutation-min RMSPE (rad) over n=min(len(pred),len(true)) matched angles."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if len(pred) == 0 or len(true) == 0:
        return np.nan
    n = min(len(pred), len(true))
    best = None
    # permute the longer set's indices, keep the first n of the shorter as anchor
    if len(pred) <= len(true):
        anchor, other = pred[:n], true
    else:
        anchor, other = true[:n], pred
    for perm in itertools.permutations(range(len(other)), n):
        diff = wrap_pi(anchor - other[list(perm)])
        r = np.sqrt(np.mean(diff ** 2))
        if best is None or r < best:
            best = r
    return best


_SEL_CACHE = {}


def _selections(n_other, n, device):
    """Ordered n-subsets of range(n_other) — the permutations rmspe_rad iterates."""
    key = (n_other, n, str(device))
    if key not in _SEL_CACHE:
        _SEL_CACHE[key] = torch.tensor(
            list(itertools.permutations(range(n_other), n)),
            dtype=torch.long, device=device)
    return _SEL_CACHE[key]


def rmspe_rad_batch(pred_list, true_list, device=None, with_mae=False):
    """Vectorized rmspe_rad over a whole batch — identical results, ~100x faster.

    Equivalent to [rmspe_rad(p, t) for p, t in zip(pred_list, true_list)].
    Samples are grouped by (len(pred), len(true)) so one tensor op covers every
    sample sharing a shape. float64 throughout, so it matches the numpy
    reference to machine precision rather than merely closely.

    with_mae=True also returns the per-scene mean ABSOLUTE error, evaluated at
    the SAME permutation that minimises the RMSPE. Taking the argmin permutation
    rather than separately minimising the MAE keeps the two metrics describing
    one assignment, which is what "permutation-matched MAE and RMSE" means.
    """
    n = len(pred_list)
    out = np.full(n, np.nan, dtype=np.float64)
    mae = np.full(n, np.nan, dtype=np.float64)
    if torch is None:
        for i, (p, t) in enumerate(zip(pred_list, true_list)):
            out[i] = rmspe_rad(p, t)
        return (out, mae) if with_mae else out
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    groups = defaultdict(list)
    for i, (p, t) in enumerate(zip(pred_list, true_list)):
        if len(p) == 0 or len(t) == 0:
            continue
        groups[(len(p), len(t))].append(i)

    for (lp, lt), idx in groups.items():
        P = torch.tensor(np.stack([np.asarray(pred_list[i], dtype=np.float64) for i in idx]),
                         dtype=torch.float64, device=device)
        T = torch.tensor(np.stack([np.asarray(true_list[i], dtype=np.float64) for i in idx]),
                         dtype=torch.float64, device=device)
        # anchor = the shorter set (used whole); select/permute n of the longer one
        anchor, other = (P, T) if lp <= lt else (T, P)
        sel = _selections(other.shape[1], min(lp, lt), device)        # (S, n)
        cand = other[:, sel]                                          # (B, S, n)
        diff = torch.remainder(anchor[:, None, :] - cand + np.pi / 2.0, np.pi) - np.pi / 2.0
        best = torch.sqrt(torch.mean(diff ** 2, dim=2)).min(dim=1)    # values, indices
        out[np.asarray(idx)] = best.values.cpu().numpy()
        if with_mae:
            rows = torch.arange(diff.shape[0], device=device)
            picked = diff[rows, best.indices]                         # (B, n)
            mae[np.asarray(idx)] = picked.abs().mean(dim=1).cpu().numpy()
    return (out, mae) if with_mae else out


SR_THRESHOLD_DEG = 5.0


def summarize(k_true, k_pred, doa_true, doa_pred):
    """
    k_true/k_pred : [N] ints
    doa_true/doa_pred : list of length N, each an array of angles (rad)

    All angular metrics share ONE assignment per scene: the permutation that
    minimises that scene's RMSPE. Scene-level values are averaged over scenes
    (not pooled over angles), so every scene carries equal weight regardless of
    how many sources it contains.

    returns dict with
      k_accuracy      P(K_hat == K_true)
      rmspe_rad/deg   mean over scenes of the per-scene matched RMSE
      mae_deg         mean over scenes of the per-scene matched MAE
      sr5             fraction of scenes whose per-scene matched RMSE < 5 deg
      rmspe_deg_kok   rmspe_deg restricted to scenes with K_hat == K_true
      mae_deg_kok     mae_deg   restricted to scenes with K_hat == K_true
      num_tests, num_kok
    """
    k_true = np.asarray(k_true)
    k_pred = np.asarray(k_pred)
    pred_len = np.asarray([len(p) for p in doa_pred], dtype=np.int64)
    true_len = np.asarray([len(t) for t in doa_true], dtype=np.int64)
    rlist, mlist = rmspe_rad_batch(doa_pred, doa_true, with_mae=True)
    rdeg = np.degrees(rlist)
    mdeg = np.degrees(mlist)
    kok = (k_true == k_pred)

    def _mean(a, mask=None):
        a = a if mask is None else a[mask]
        return float(np.nanmean(a)) if a.size and np.any(~np.isnan(a)) else np.nan

    valid = ~np.isnan(rdeg)
    sr5 = float(np.mean(rdeg[valid] < SR_THRESHOLD_DEG)) if np.any(valid) else np.nan
    joint = valid & kok & (rdeg < SR_THRESHOLD_DEG)
    rmspe = _mean(rlist)
    return {
        "num_tests": int(len(k_true)),
        "num_kok": int(kok.sum()),
        "k_accuracy": float(np.mean(kok)),
        "joint_success": float(np.mean(joint)) if len(k_true) else np.nan,
        "mean_pred_len": float(np.mean(pred_len)) if pred_len.size else np.nan,
        "mean_true_len": float(np.mean(true_len)) if true_len.size else np.nan,
        "short_pred_rate": float(np.mean(pred_len < k_pred)) if pred_len.size else np.nan,
        "rmspe_rad": rmspe,
        "rmspe_deg": float(np.degrees(rmspe)) if not np.isnan(rmspe) else np.nan,
        "mae_deg": _mean(mdeg),
        "sr5": sr5,
        "rmspe_deg_kok": _mean(rdeg, kok),
        "mae_deg_kok": _mean(mdeg, kok),
    }
