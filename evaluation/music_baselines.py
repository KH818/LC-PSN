"""
Classical MUSIC baselines for the unified 1-bit harness — one-bit first.

Three covariance front-ends, all fed by the SAME signal batch the learned models
see (`signal_gen.make_onebit_batch`), so the only thing that differs between a
baseline row and an LCPSN row is the estimator:

    naive    R_z = (1/T) z z^H            one-bit sample covariance.
                                          One-bit MUSIC (Huang & Liao, SPL'19)
                                          applies MUSIC to it directly: the
                                          signal/noise subspaces survive the
                                          scaling + identity shift of the sign
                                          nonlinearity.
    arcsine  R_0 = sin((pi/2) C_z)        Van Vleck inverse on the unit-diagonal
                                          normalization of R_z, applied to the
                                          real and imaginary parts separately.
                                          This is byte-for-byte the statistic
                                          LCPSN starts from (`model.py:
                                          onebit_arcsine_cov`), so the arcsine
                                          rows isolate "what the network adds on
                                          top of its own input".
    analog   R   = (1/T) y y^H            unquantized sample covariance. Not a
                                          one-bit method — the full-precision
                                          anchor that says how much the sign
                                          quantizer costs.

and three enumeration rules, because "unknown K" is the whole point:

    mdl / aic   Wax & Kailath information-theoretic criteria on the eigenvalues
                of whichever covariance the row uses.
    oracle      the true K is handed to the estimator. Not a competitor — it
                separates the two error sources: with oracle K the remaining
                error is purely angular, so (oracle - mdl) is exactly what
                blind enumeration costs that estimator.

Decoding is deliberately identical to `adapters.LCPSNAdapter._peaks`: top-K_hat
local maxima of the pseudospectrum on the same 360-point grid with the same 2 deg
separation floor. Same grid, same peak rule, same metric -> the comparison is
about the estimator, not about the decoder.

Caveat carried by the arcsine rows: sin((pi/2) C_z) is applied entry-wise and is
not projected back onto the PSD cone, so R_0 can be indefinite at finite T. Its
eigenvalues are clipped at `EIG_FLOOR` before the log-ratio in AIC/MDL, which is
the standard plug-in fix and is reported as such rather than hidden.
"""
import math
import numpy as np

from decoder import separated_topk

ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
K_MIN, K_MAX = 2, 5
EIG_FLOOR = 1e-10
COV_KINDS = ("naive", "arcsine", "analog")
ENUM_RULES = ("mdl", "aic", "oracle")


# ----------------------------------------------------------------- covariances
def _to_complex(x1bit):
    """[N,2M,T] real-augmented signs -> [N,M,T] complex."""
    m = x1bit.shape[1] // 2
    return (x1bit[:, :m, :] + 1j * x1bit[:, m:, :]).astype(np.complex128)


def cov_naive(x1bit):
    """One-bit sample covariance R_z = (1/T) z z^H, [N,M,M]."""
    z = _to_complex(x1bit)
    T = z.shape[-1]
    R = z @ np.conj(np.swapaxes(z, -1, -2)) / float(T)
    return 0.5 * (R + np.conj(np.swapaxes(R, -1, -2)))


def cov_arcsine(x1bit):
    """Van Vleck-corrected covariance, identical to LCPSN's own front-end.

    Mirrors `lcpsn/model.py: LCPSN.onebit_arcsine_cov` exactly: normalize R_z to
    unit diagonal, then invert the arcsine law on the real and imaginary parts
    separately, then Hermitian-symmetrize.
    """
    R = cov_naive(x1bit)
    diag = np.maximum(np.real(np.diagonal(R, axis1=-2, axis2=-1)), 1e-8)
    inv_std = 1.0 / np.sqrt(diag)                                  # [N,M]
    Cz = R * inv_std[:, :, None] * inv_std[:, None, :]
    Rx = np.sin((math.pi / 2.0) * Cz.real) + 1j * np.sin((math.pi / 2.0) * Cz.imag)
    return 0.5 * (Rx + np.conj(np.swapaxes(Rx, -1, -2)))


def cov_analog(analog):
    """Full-precision sample covariance from the unquantized snapshots."""
    y = np.asarray(analog, dtype=np.complex128)
    T = y.shape[-1]
    R = y @ np.conj(np.swapaxes(y, -1, -2)) / float(T)
    return 0.5 * (R + np.conj(np.swapaxes(R, -1, -2)))


def covariance(kind, x1bit=None, analog=None):
    if kind == "naive":
        return cov_naive(x1bit)
    if kind == "arcsine":
        return cov_arcsine(x1bit)
    if kind == "analog":
        if analog is None:
            raise ValueError("cov kind 'analog' needs the unquantized snapshots")
        return cov_analog(analog)
    raise ValueError(f"unknown covariance kind {kind!r}")


# ----------------------------------------------------------------- enumeration
def eigh_desc(R):
    """Eigen-decomposition sorted by DESCENDING eigenvalue. -> (vals [N,M], vecs [N,M,M])."""
    w, V = np.linalg.eigh(R)                 # ascending
    return w[:, ::-1], V[:, :, ::-1]


def info_criterion_k(eigvals_desc, snapshots, rule):
    """Wax-Kailath AIC / MDL source enumeration.

    For k = 0..M-1 the noise cluster is the trailing M-k eigenvalues, and
        L(k) = T (M-k) log( arithmetic_mean / geometric_mean )
        AIC(k) = 2 L(k) + 2 k (2M - k)
        MDL(k) = L(k) + 0.5 k (2M - k) log T
    The returned k is the minimizer, clipped to the [K_MIN, K_MAX] range every
    method in this study is restricted to.
    """
    lam = np.maximum(np.asarray(eigvals_desc, dtype=np.float64), EIG_FLOOR)
    n, m = lam.shape
    T = float(snapshots)
    ks = np.arange(m)                                          # 0..M-1
    # trailing statistics: sums / log-sums of lam[:, k:]
    rev_sum = np.cumsum(lam[:, ::-1], axis=1)[:, ::-1]         # [N,M] sum_{i>=k}
    rev_logsum = np.cumsum(np.log(lam)[:, ::-1], axis=1)[:, ::-1]
    cnt = (m - ks).astype(np.float64)                          # M-k
    log_arith = np.log(rev_sum / cnt[None, :])
    log_geo = rev_logsum / cnt[None, :]
    ll = T * cnt[None, :] * (log_arith - log_geo)              # >= 0
    free = ks * (2 * m - ks)
    if rule == "mdl":
        crit = ll + 0.5 * free[None, :] * math.log(T)
    elif rule == "aic":
        crit = 2.0 * ll + 2.0 * free[None, :]
    else:
        raise ValueError(f"unknown enumeration rule {rule!r}")
    khat = np.argmin(crit, axis=1)
    return np.clip(khat, K_MIN, K_MAX).astype(np.int64)


# ----------------------------------------------------------------- pseudospectrum
def steering_grid(m, angles=ANGLES):
    """[M,G] ULA manifold, same convention as signal_gen.steering_matrix."""
    n = np.arange(m)[:, None]
    return np.exp(-1j * np.pi * n * np.sin(angles)[None, :])


def music_spectrum(vecs_desc, khat, a_grid):
    """MUSIC pseudospectrum 1 / ||E_n^H a(theta)||^2 -> [N,G].

    K_hat varies per sample, so samples are grouped by K_hat and each group is
    one batched einsum (M is 8; the whole batch is a handful of tiny contractions).
    """
    n, m, _ = vecs_desc.shape
    g = a_grid.shape[1]
    out = np.empty((n, g), dtype=np.float64)
    for k in np.unique(khat):
        idx = np.nonzero(khat == k)[0]
        En = vecs_desc[idx][:, :, int(k):]                      # [n_k, M, M-k]
        proj = np.einsum('nml,mg->nlg', np.conj(En), a_grid)    # [n_k, M-k, G]
        denom = np.sum(np.abs(proj) ** 2, axis=1)               # [n_k, G]
        out[idx] = 1.0 / np.maximum(denom, 1e-12)
    return out


def top_k_peaks(score, khat, grid=ANGLES, min_sep_deg=2.0):
    """Top-K_hat separated local maxima — same rule as LCPSNAdapter._peaks.

    Local maxima first, ranked by height, greedily accepted while at least
    `min_sep_deg` away from every already-accepted peak; if the spectrum has no
    interior local maximum the whole grid is ranked instead. Every candidate is
    inspected (no truncation), so this is the LCPSN decoder applied to a MUSIC
    pseudospectrum rather than a cheaper approximation of it.
    """
    return separated_topk(score, khat, grid, min_sep_deg=min_sep_deg)


# ----------------------------------------------------------------- estimator
def music_predict(R, snapshots, rule, ktrue=None, min_sep_deg=2.0, a_grid=None):
    """Full blind MUSIC pass: eig -> enumerate -> pseudospectrum -> peaks.

    rule='oracle' uses `ktrue` instead of an information criterion.
    Returns (khat [N], doa_pred list of arrays in rad).
    """
    vals, vecs = eigh_desc(R)
    if rule == "oracle":
        if ktrue is None:
            raise ValueError("oracle enumeration needs ktrue")
        khat = np.clip(np.asarray(ktrue, dtype=np.int64), K_MIN, K_MAX)
    else:
        khat = info_criterion_k(vals, snapshots, rule)
    if a_grid is None:
        a_grid = steering_grid(R.shape[-1])
    spec = music_spectrum(vecs, khat, a_grid)
    doa = [top_k_peaks(spec[i], int(khat[i]), min_sep_deg=min_sep_deg)
           for i in range(len(khat))]
    return khat, doa
