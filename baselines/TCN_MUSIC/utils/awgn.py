"""Add complex white Gaussian noise at a requested signal-to-noise ratio."""

import numpy as np


def awgn(signal, snr_db, signal_power_in_db, seed=None):
    if seed is not None:
        np.random.seed(seed)

    noise_power_db = signal_power_in_db - snr_db
    noise_power_linear = 10 ** (noise_power_db / 10)
    real_noise = np.random.normal(0, np.sqrt(noise_power_linear/2), size=signal.shape)
    imag_noise = np.random.normal(0, np.sqrt(noise_power_linear/2), size=signal.shape)
    noise = real_noise + 1j * imag_noise
    noisy_signal = signal + noise

    return noisy_signal
