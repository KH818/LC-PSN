"""Provide the ULA steering vector and synthetic narrowband signal models."""

import numpy as np


M = 8


def ULA_action_vector(theta, m=M):
    """Return the half-wavelength ULA steering vector at ``theta`` radians."""
    sensor_index = np.arange(m)
    return np.exp(-1j * np.pi * sensor_index * np.sin(theta))


def construct_signal(
    thetas,
    snr,
    snapshots,
    m=M,
    var_signal_power=1.0,
    var_noise=1.0,
):
    """Generate independent complex Gaussian sources for a ULA."""
    source_count = len(thetas)
    source_scale = np.sqrt(var_signal_power) * np.sqrt(10 ** (snr / 10))
    sources = source_scale * (
        np.random.randn(source_count, snapshots)
        + 1j * np.random.randn(source_count, snapshots)
    )

    steering = np.array([ULA_action_vector(theta, m) for theta in thetas])
    noise = np.sqrt(var_noise) * (
        np.random.randn(m, snapshots) + 1j * np.random.randn(m, snapshots)
    )
    return steering.T @ sources + noise, sources


def construct_coherent_signal(
    thetas,
    snr,
    snapshots,
    m=M,
    var_signal_power=1.0,
    var_noise=1.0,
):
    """Generate fully coherent complex Gaussian sources for a ULA."""
    source_count = len(thetas)
    source_scale = np.sqrt(var_signal_power) * (10 ** (snr / 10))
    shared_source = source_scale * (
        np.random.randn(1, snapshots) + 1j * np.random.randn(1, snapshots)
    )
    sources = np.repeat(shared_source, source_count, axis=0)

    steering = np.array([ULA_action_vector(theta, m) for theta in thetas])
    noise = np.sqrt(var_noise) * (
        np.random.randn(m, snapshots) + 1j * np.random.randn(m, snapshots)
    )
    return steering.T @ sources + noise, sources
