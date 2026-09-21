"""Approximate entry-wise uncertainty of an arcsine-corrected one-bit covariance."""

import math

import torch


def _as_signs(x: torch.Tensor) -> torch.Tensor:
    """Map zero entries to +1 so sign-pair products stay in {-1, +1}."""
    signs = torch.sign(x)
    return torch.where(signs == 0, torch.ones_like(signs), signs)


def arcsine_covariance_uncertainty_from_signs(
    x: torch.Tensor,
    eps: float = 1e-6,
    u_max: float = 10.0,
):
    """
    First-order finite-snapshot uncertainty approximation for the real-augmented arcsine covariance.

    x: real-augmented one-bit signs, shape [B, 2M, T]
    returns:
      tau_hat: empirical sign-pair correlation, [B, 2M, 2M]
      C_hat: arcsine correlation estimate, [B, 2M, 2M]
      U0: delta-method variance of C_hat, [B, 2M, 2M]
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, 2M, T], got {tuple(x.shape)}")

    T = x.shape[-1]
    if T <= 0:
        raise ValueError("snapshot dimension T must be positive")
    s = _as_signs(x)
    tau_hat = torch.matmul(s, s.transpose(-1, -2)) / float(T)
    tau_hat = tau_hat.clamp(min=-1.0 + eps, max=1.0 - eps)

    angle = (math.pi / 2.0) * tau_hat
    C_hat = torch.sin(angle)

    deriv_sq = ((math.pi / 2.0) * torch.cos(angle)) ** 2
    tau_var = (1.0 - tau_hat.square()).clamp_min(0.0) / float(T)
    U0 = (deriv_sq * tau_var).clamp(min=eps, max=u_max)

    # Keep exact algebraic symmetries after clamp/elementwise operations.
    tau_hat = 0.5 * (tau_hat + tau_hat.transpose(-1, -2))
    C_hat = 0.5 * (C_hat + C_hat.transpose(-1, -2))
    U0 = 0.5 * (U0 + U0.transpose(-1, -2))
    return tau_hat, C_hat, U0


def real_augmented_cov_to_complex(C: torch.Tensor, m: int) -> torch.Tensor:
    """
    Convert real-augmented covariance [Re, Im] blocks to complex covariance.

    C: [B, 2M, 2M] real covariance/correlation estimate
    returns: [B, M, M] complex matrix using
      Re(R_ij) = 0.5 * (C_rr(i,j) + C_ii(i,j))
      Im(R_ij) = 0.5 * (C_ir(i,j) - C_ri(i,j))
    """
    if C.shape[-2:] != (2 * m, 2 * m):
        raise ValueError(f"C must have trailing shape [{2 * m}, {2 * m}], got {tuple(C.shape[-2:])}")

    C_rr = C[..., :m, :m]
    C_ri = C[..., :m, m:]
    C_ir = C[..., m:, :m]
    C_ii = C[..., m:, m:]
    R = torch.complex(0.5 * (C_rr + C_ii), 0.5 * (C_ir - C_ri))
    return 0.5 * (R + R.conj().transpose(-1, -2))


def complex_arcsine_covariance_with_uncertainty(
    z: torch.Tensor,
    eps: float = 1e-6,
    u_max: float = 10.0,
):
    """
    Complex one-bit wrapper for Stage A diagnostics.

    z: complex one-bit snapshots, shape [B, M, T]
    returns:
      R0: complex arcsine covariance-like estimate, [B, M, M]
      U0: real-augmented arcsine covariance uncertainty, [B, 2M, 2M]
      tau_hat: empirical real-augmented sign correlation, [B, 2M, 2M]
      C_hat: real-augmented arcsine correlation estimate, [B, 2M, 2M]
    """
    if z.ndim != 3 or not torch.is_complex(z):
        raise ValueError(f"z must be complex with shape [B, M, T], got {tuple(z.shape)}")

    x = torch.cat([z.real, z.imag], dim=1)
    tau_hat, C_hat, U0 = arcsine_covariance_uncertainty_from_signs(x, eps=eps, u_max=u_max)
    R0 = real_augmented_cov_to_complex(C_hat, z.shape[1])
    return R0, U0, tau_hat, C_hat
