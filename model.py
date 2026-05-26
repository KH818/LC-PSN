import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceTokenizer(nn.Module):
    def __init__(self, m: int, d_model: int):
        super().__init__()
        self.m = m
        self.norm = nn.LayerNorm(2 * m)
        self.proj = nn.Linear(2 * m, d_model)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        # R: [B, M, M] complex
        tok = torch.cat([R.real, R.imag], dim=-1)   # [B, M, 2M]
        tok = self.norm(tok)
        tok = self.proj(tok)                        # [B, M, d_model]
        return tok


class TransMusic(nn.Module):
    """
    Spectrum-only TransMUSIC-style model for 1-bit / unknown-K DOA.

    Input : x [B, 2M, T]
    Output:
      spec_logits: [B, G]
      spectrum_prob: [B, G]
      aux: dict with refined covariance / physics prior spectrum
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
    ):
        super().__init__()
        self.m = m
        self.snapshots = snapshots
        self.angles_count = angles_count
        self.d_model = d_model

        self.tokenizer = CovarianceTokenizer(m, d_model)

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
            nn.Linear(d_model, 2 * m),
        )

        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.spec_head = nn.Sequential(
            nn.Linear(d_model + angles_count, spectrum_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spectrum_hidden, spectrum_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spectrum_hidden, angles_count),
        )

        # refinement strength / physics-prior strength
        self.alpha = nn.Parameter(torch.tensor(0.10))
        self.beta = nn.Parameter(torch.tensor(0.75))

    def realimag_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        xr = x[:, :self.m, :]
        xi = x[:, self.m:, :]
        return torch.complex(xr, xi)  # [B, M, T]

    def onebit_arcsine_cov(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, M, T] complex quantized observations
        Returns corrected covariance-like matrix [B, M, M] complex.
        """
        _, _, T = z.shape
        Rz = z @ z.conj().transpose(-1, -2) / T  # [B,M,M]

        diag = torch.real(torch.diagonal(Rz, dim1=-2, dim2=-1)).clamp_min(1e-8)
        inv_std = 1.0 / torch.sqrt(diag)
        Dinv = torch.diag_embed(inv_std).to(Rz.dtype)

        Cz = Dinv @ Rz @ Dinv
        Rx_re = torch.sin((math.pi / 2.0) * Cz.real)
        Rx_im = torch.sin((math.pi / 2.0) * Cz.imag)
        Rx = torch.complex(Rx_re, Rx_im)
        Rx = 0.5 * (Rx + Rx.conj().transpose(-1, -2))
        return Rx

    def refine_covariance(self, R0: torch.Tensor):
        feat, pooled = self.encode_covariance(R0)

        row_delta = self.row_out(feat)     # [B,M,2M]
        d_re = row_delta[..., :self.m]
        d_im = row_delta[..., self.m:]
        Delta = torch.complex(d_re, d_im)
        Delta = 0.5 * (Delta + Delta.conj().transpose(-1, -2))

        R_ref = R0 + self.alpha * Delta
        R_ref = 0.5 * (R_ref + R_ref.conj().transpose(-1, -2))

        return R_ref, pooled

    def encode_covariance(self, R: torch.Tensor):
        """
        Encode a complex covariance matrix into row and pooled features.

        R: [B, M, M] complex
        returns:
          feat: [B, M, d_model]
          pooled: [B, d_model]
        """
        tok = self.tokenizer(R)      # [B,M,d]
        feat = self.encoder(tok)     # [B,M,d]
        pooled = self.pool(feat.mean(dim=1))
        return feat, pooled

    def bartlett_spectrum_from_cov(self, R: torch.Tensor, a_steering: torch.Tensor) -> torch.Tensor:
        """
        R: [B,M,M] complex Hermitian
        a_steering: [M,G] complex
        returns normalized Bartlett spectrum in [0, 1], [B,G]
        """
        Ra = torch.matmul(R, a_steering.unsqueeze(0))                # [B,M,G]
        quad = torch.sum(torch.conj(a_steering).unsqueeze(0) * Ra, dim=1).real  # [B,G]
        quad = quad.clamp_min(1e-8)
        quad = quad / (quad.amax(dim=1, keepdim=True) + 1e-8)
        return quad

    def covariance_features_from_snapshots(self, x: torch.Tensor, a_steering: torch.Tensor):
        """
        Shared TransMUSIC/LC-PSN front-end.

        x: [B, 2M, T] one-bit real/imag sign snapshots
        a_steering: [M, G] complex steering dictionary
        returns:
          R0: [B, M, M] complex arcsine covariance
          R_ref: [B, M, M] complex refined covariance
          pooled: [B, d_model] neural covariance feature
          S_base: [B, G] Bartlett prior spectrum
        """
        z = self.realimag_to_complex(x)     # [B,M,T]
        R0 = self.onebit_arcsine_cov(z)     # [B,M,M]
        R_ref, pooled = self.refine_covariance(R0)
        S_base = self.bartlett_spectrum_from_cov(R_ref, a_steering)  # [B,G]
        return R0, R_ref, pooled, S_base

    def spectrum_logits_from_features(self, pooled: torch.Tensor, S_base: torch.Tensor) -> torch.Tensor:
        """
        Spectrum-only TransMUSIC head.

        Kept separate so LC-PSN can reuse the covariance/Bartlett front-end
        without changing the baseline training and evaluation interface.
        """
        feat_cat = torch.cat([pooled, self.beta * S_base], dim=1)    # [B,d+G]
        spec_residual = self.spec_head(feat_cat)

        prior_logits = torch.logit(S_base.clamp(1e-4, 1 - 1e-4))
        return spec_residual + self.beta * prior_logits

    def forward(self, x: torch.Tensor, a_steering: torch.Tensor):
        R0, R_ref, pooled, S_base = self.covariance_features_from_snapshots(x, a_steering)
        spec_logits = self.spectrum_logits_from_features(pooled, S_base)
        spectrum = torch.sigmoid(spec_logits)

        aux = {
            "R0": R0,
            "R_ref": R_ref,
            "base_spec": S_base,
            "S_base": S_base,
        }
        return spec_logits, spectrum, aux


class LCPSN(TransMusic):
    """
    LC-PSN: Likelihood-Consistent Probabilistic Spectrum Network.

    Input : x [B, 2M, T]
    Output: dict with mu, sigma2, mu_logits, log_var, K_logits, R_ref, S_base
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
        super().__init__(
            m=m,
            snapshots=snapshots,
            angles_count=angles_count,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            spectrum_hidden=spectrum_hidden,
        )
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_classes = self.k_max - self.k_min + 1
        self.variance_floor = float(variance_floor)

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
        k_head_in = head_in + m + (m - 1)
        self.K_head = nn.Sequential(
            nn.Linear(k_head_in, max(spectrum_hidden // 2, d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(spectrum_hidden // 2, d_model), self.k_classes),
        )

    def probabilistic_heads_from_features(self, pooled: torch.Tensor, S_base: torch.Tensor, R_ref: torch.Tensor):
        """
        Probabilistic spectrum and K heads.

        pooled: [B, d_model]
        S_base: [B, G]
        returns mu, sigma2, mu_logits, log_var, K_logits
        """
        feat_cat = torch.cat([pooled, self.beta * S_base], dim=1)  # [B,d+G]
        prior_logits = torch.logit(S_base.clamp(1e-4, 1.0 - 1e-4))

        mu_residual = self.mu_head(feat_cat)
        mu_logits = mu_residual + self.beta * prior_logits         # [B,G]
        mu = torch.sigmoid(mu_logits)                              # [B,G]

        log_var = self.log_var_head(feat_cat).clamp(-10.0, 5.0)    # [B,G]
        sigma2 = F.softplus(log_var) + self.variance_floor         # [B,G]

        eigvals = torch.linalg.eigvalsh(R_ref).real                   # [B, M]
        eigvals_sorted = torch.sort(eigvals, dim=-1, descending=True).values
        log_eig = torch.log(eigvals_sorted.clamp_min(1e-6))           # [B, M]
        log_eig_gap = log_eig[:, :-1] - log_eig[:, 1:]                # [B, M-1]
        k_feat = torch.cat([feat_cat, log_eig, log_eig_gap], dim=-1)
        K_logits = self.K_head(k_feat)                                # [B,K_classes]
        return mu, sigma2, mu_logits, log_var, K_logits

    def forward_from_cov(self, R_ref: torch.Tensor, A: torch.Tensor):
        """
        Recompute LC-PSN outputs from a refined covariance.

        R_ref: [B, M, M] complex refined covariance
        A: [M, G] complex steering dictionary
        """
        R_ref = 0.5 * (R_ref + R_ref.conj().transpose(-1, -2))
        _, pooled = self.encode_covariance(R_ref)
        S_base = self.bartlett_spectrum_from_cov(R_ref, A)         # [B,G]
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
        }

    def forward(self, x: torch.Tensor, a_steering: torch.Tensor):
        z = self.realimag_to_complex(x)     # [B,M,T]
        R0 = self.onebit_arcsine_cov(z)     # [B,M,M]
        R_ref, pooled = self.refine_covariance(R0)
        S_base = self.bartlett_spectrum_from_cov(R_ref, a_steering)  # [B,G]
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
        }
