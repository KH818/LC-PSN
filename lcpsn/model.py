"""Implement LCPSN from one-bit covariance recovery to spectrum and source-count outputs."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from uncertainty import arcsine_covariance_uncertainty_from_signs


class CovarianceTokenizer(nn.Module):
    """Map each complex covariance row to one real Transformer token."""
    def __init__(self, m: int, d_model: int):
        super().__init__()
        self.m = int(m)
        self.norm = nn.LayerNorm(2 * self.m)
        self.proj = nn.Linear(2 * self.m, d_model)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        # R: [B, M, M] complex Hermitian covariance.
        tok = torch.cat([R.real, R.imag], dim=-1)
        return self.proj(self.norm(tok))


class LCPSN(nn.Module):
    """
    Spectrum network for one-bit unknown-K DOA estimation.

    This is the retained RA-PSN-Cov path:
      one-bit signs -> arcsine covariance R0 and uncertainty U0
      -> neural covariance refinement R_ref
      -> spectrum mean/variance and K logits

    U0 is exposed for the uncertainty-weighted covariance consistency loss.
    It is not concatenated as an input feature.
    """

    def __init__(
        self,
        m=8,
        snapshots=200,
        angles_count=360,
        d_model=96,
        nhead=8,
        num_layers=4,
        ff_dim=256,
        dropout=0.1,
        spectrum_hidden=512,
        k_min=2,
        k_max=5,
        variance_floor=5e-2,
    ):
        super().__init__()
        self.m = int(m)
        self.snapshots = int(snapshots)
        self.angles_count = int(angles_count)
        self.d_model = int(d_model)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_classes = self.k_max - self.k_min + 1
        self.variance_floor = float(variance_floor)

        self.tokenizer = CovarianceTokenizer(self.m, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.row_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * self.m),
        )

        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        head_in = d_model + angles_count
        self.mu_head = nn.Sequential(
            nn.Linear(head_in, spectrum_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spectrum_hidden, spectrum_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spectrum_hidden, angles_count),
        )
        self.log_var_head = nn.Sequential(
            nn.Linear(head_in, spectrum_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spectrum_hidden, spectrum_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spectrum_hidden, angles_count),
        )
        k_head_in = head_in + self.m + (self.m - 1)
        self.K_head = nn.Sequential(
            nn.Linear(k_head_in, max(spectrum_hidden // 2, d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(spectrum_hidden // 2, d_model), self.k_classes),
        )

        self.alpha = nn.Parameter(torch.tensor(0.10))
        self.beta = nn.Parameter(torch.tensor(0.75))

    def realimag_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        xr = x[:, : self.m, :]
        xi = x[:, self.m :, :]
        return torch.complex(xr, xi)

    def onebit_arcsine_cov(self, z: torch.Tensor) -> torch.Tensor:
        """Recover a normalized covariance surrogate with the complex arcsine law."""
        _, _, T = z.shape
        Rz = z @ z.conj().transpose(-1, -2) / float(T)

        diag = torch.real(torch.diagonal(Rz, dim1=-2, dim2=-1)).clamp_min(1e-8)
        inv_std = 1.0 / torch.sqrt(diag)
        Dinv = torch.diag_embed(inv_std).to(Rz.dtype)

        Cz = Dinv @ Rz @ Dinv
        Rx = torch.complex(
            torch.sin((math.pi / 2.0) * Cz.real),
            torch.sin((math.pi / 2.0) * Cz.imag),
        )
        return 0.5 * (Rx + Rx.conj().transpose(-1, -2))

    def onebit_arcsine_uncertainty(self, x: torch.Tensor):
        """Estimate the real-augmented reliability weights used only by the loss."""
        return arcsine_covariance_uncertainty_from_signs(x)

    def encode_covariance(self, R: torch.Tensor):
        """Encode covariance rows and mean-pool them into one scene feature."""
        feat = self.encoder(self.tokenizer(R))
        pooled = self.pool(feat.mean(dim=1))
        return feat, pooled

    def refine_covariance(self, R0: torch.Tensor):
        """Predict a Hermitian residual and add it to the physical covariance estimate."""
        feat, pooled = self.encode_covariance(R0)

        row_delta = self.row_out(feat)
        d_re = row_delta[..., : self.m]
        d_im = row_delta[..., self.m :]
        delta = torch.complex(d_re, d_im)
        delta = 0.5 * (delta + delta.conj().transpose(-1, -2))

        R_ref = R0 + self.alpha * delta
        R_ref = 0.5 * (R_ref + R_ref.conj().transpose(-1, -2))
        return R_ref, pooled

    def bartlett_spectrum_from_cov(
        self,
        R: torch.Tensor,
        a_steering: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a normalized Bartlett prior on the fixed steering grid."""
        Ra = torch.matmul(R, a_steering.unsqueeze(0))
        quad = torch.sum(torch.conj(a_steering).unsqueeze(0) * Ra, dim=1).real
        quad = quad.clamp_min(1e-8)
        return quad / (quad.amax(dim=1, keepdim=True) + 1e-8)

    def probabilistic_heads_from_features(
        self,
        pooled: torch.Tensor,
        S_base: torch.Tensor,
        R_ref: torch.Tensor,
    ):
        """Predict spectrum mean/variance and source-count logits from shared features."""
        feat_cat = torch.cat([pooled, self.beta * S_base], dim=1)
        prior_logits = torch.logit(S_base.clamp(1e-4, 1.0 - 1e-4))

        mu_logits = self.mu_head(feat_cat) + self.beta * prior_logits
        mu = torch.sigmoid(mu_logits)

        log_var = self.log_var_head(feat_cat).clamp(-10.0, 5.0)
        sigma2 = F.softplus(log_var) + self.variance_floor

        eigvals = torch.linalg.eigvalsh(R_ref).real
        eigvals_sorted = torch.sort(eigvals, dim=-1, descending=True).values
        log_eig = torch.log(eigvals_sorted.clamp_min(1e-6))
        log_eig_gap = log_eig[:, :-1] - log_eig[:, 1:]
        K_logits = self.K_head(torch.cat([feat_cat, log_eig, log_eig_gap], dim=-1))
        return mu, sigma2, mu_logits, log_var, K_logits

    def forward(self, x: torch.Tensor, a_steering: torch.Tensor):
        # Stage 1: recover a physical covariance and its reliability weights.
        z = self.realimag_to_complex(x)
        R0 = self.onebit_arcsine_cov(z)
        tau_hat, _, U0 = self.onebit_arcsine_uncertainty(x)
        # Stage 2: learn a small Hermitian correction to the recovered covariance.
        R_ref, pooled = self.refine_covariance(R0)
        # Stage 3: combine the Bartlett prior with learned scene features.
        S_base = self.bartlett_spectrum_from_cov(R_ref, a_steering)
        mu, sigma2, mu_logits, log_var, K_logits = self.probabilistic_heads_from_features(
            pooled,
            S_base,
            R_ref,
        )
        return {
            "mu": mu,
            "sigma2": sigma2,
            "mu_logits": mu_logits,
            "log_var": log_var,
            "K_logits": K_logits,
            "R_ref": R_ref,
            "S_base": S_base,
            "R0": R0,
            "U0": U0,
            "tau_hat": tau_hat,
        }
