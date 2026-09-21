"""
Unified 1-bit signal generator shared by ALL four models
(Trans-MUSIC / TCN-MUSIC / DA-MUSIC / LCPSN).

Signal model (paper-faithful, matches TCN_MUSIC/utils OneBitSignalReceiver):
    x(t) = A s(t)                      sources s ~ CN(0, I), unit power
    y(t) = x(t) + n(t)                 AWGN, noise power = 10^(-SNR/10)
    z(t) = Q1(y(t)),  Q1 = (1/sqrt2)(sign(Re) + j sign(Im))   [threshold 0]

  * ULA steering: a_n(theta) = exp(-j * pi * n * sin(theta)), n = 0..M-1
    (identical to TCN_MUSIC UniformLinearArray and lcpsn physics.ULA_action_vector)
  * signal power = 1, noise scaled by SNR  -> reproduces the exact distribution the
    3 models were trained on. LCPSN is scale-invariant (arcsine cov + sign()), so the
    same tensor feeds all four models.

`min_sep_deg`:
  * None  -> DoAs drawn i.i.d. uniform in (-pi/2, pi/2)  (paper Fig.13 / 3-model default)
  * 8.0   -> DoAs resampled until pairwise separation >= 8 deg (lcpsn default)

Returned per batch:
    x1bit : float32 [N, 2M, T]  = [Re(z); Im(z)],  values in {-1/sqrt2, +1/sqrt2}
    truth : float32 [N, KMAX]   true DoAs (rad), padded with pi
    ktrue : int64   [N]         number of sources
"""
import numpy as np

M_DEFAULT = 8
KMAX = 6                     # padding width (matches both codebases' Y layout)
ONE_BIT_SCALE = 1.0 / np.sqrt(2.0)


def steering_matrix(thetas, m=M_DEFAULT):
    """A: [M, K] complex, half-wavelength ULA."""
    n = np.arange(m)[:, None]                       # [M,1]
    return np.exp(-1j * np.pi * n * np.sin(thetas)[None, :])   # [M,K]


def sample_angles(k, rng, min_sep_deg=None, max_trials=2000):
    """Draw k DoAs (rad) in (-pi/2, pi/2), optionally with a min pairwise gap."""
    if min_sep_deg is None:
        return np.sort(np.pi * (rng.random(k) - 0.5))
    min_sep = np.deg2rad(min_sep_deg)
    for _ in range(max_trials):
        th = np.sort(np.pi * (rng.random(k) - 0.5))
        if k == 1 or np.all(np.diff(th) >= min_sep):
            return th
    # fallback: evenly spaced jittered grid
    grid = np.linspace(-np.pi / 2 + min_sep, np.pi / 2 - min_sep, 2000)
    idx = np.sort(rng.choice(len(grid), size=k, replace=False))
    th = np.sort(grid[idx])
    for i in range(1, len(th)):
        if th[i] - th[i - 1] < min_sep:
            th[i] = min(th[i - 1] + min_sep, np.pi / 2 - min_sep)
    return th


def make_onebit_batch(ktrue, snr_db, snapshots, rng, m=M_DEFAULT, min_sep_deg=None,
                      return_analog=False):
    """
    ktrue : iterable of ints (per-sample source counts)
    returns x1bit [N,2M,T], truth [N,KMAX], ktrue [N]
            (+ analog [N,M,T] complex64 when return_analog=True)

    `return_analog` only keeps the pre-quantization y = A s + n that is computed
    anyway; it draws no extra random numbers, so a run that asks for it sees the
    exact same waveforms as one that does not. That is what lets the classical
    full-precision baseline be scored on the same batch as the one-bit models.
    """
    ktrue = np.asarray(ktrue, dtype=np.int64)
    n = len(ktrue)
    T = int(snapshots)
    x1bit = np.zeros((n, 2 * m, T), dtype=np.float32)
    truth = np.full((n, KMAX), np.pi, dtype=np.float32)
    analog = np.zeros((n, m, T), dtype=np.complex64) if return_analog else None

    noise_power = 10.0 ** (-snr_db / 10.0)           # signal power fixed at 1
    n_sigma = np.sqrt(noise_power / 2.0)

    for i in range(n):
        k = int(ktrue[i])
        thetas = sample_angles(k, rng, min_sep_deg=min_sep_deg)
        A = steering_matrix(thetas, m)                                    # [M,k]
        s = np.sqrt(0.5) * (rng.standard_normal((k, T)) + 1j * rng.standard_normal((k, T)))
        noise = n_sigma * (rng.standard_normal((m, T)) + 1j * rng.standard_normal((m, T)))
        y = A @ s + noise                                                 # [M,T], noisy (pre-quant)
        z = ONE_BIT_SCALE * (np.where(y.real > 0, 1.0, -1.0)
                             + 1j * np.where(y.imag > 0, 1.0, -1.0))      # Q1(Ax+n)
        x1bit[i, :m, :] = z.real
        x1bit[i, m:, :] = z.imag
        truth[i, :k] = thetas
        if return_analog:
            analog[i] = y
    if return_analog:
        return x1bit, truth, ktrue, analog
    return x1bit, truth, ktrue
