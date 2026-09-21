"""Transformer-based one-bit MUSIC regressor used as a learned baseline."""

import torch
import torch.nn as nn
import numpy as np
from utils.uniform_linear_array import UniformLinearArray
from utils.music_spectrum import MUSICSpectrum
from utils.positional_encoding import PositionalEncoding


class TransMUSICModel(nn.Module):
    """Encode snapshot tokens, form a learned subspace, and regress DOAs."""
    def __init__(self,
                 angle_grids=np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False),
                 rx_ula=UniformLinearArray(8),
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(TransMUSICModel, self).__init__()
        self.device = device
        self.number_of_ula_elements = rx_ula.number_of_elements
        # Precompute the array manifold used to form the MUSIC spectrum.
        steering_vector_set = torch.from_numpy(rx_ula.steering_vector_radian(angle_grids))
        self.steering_vector_set = torch.complex(steering_vector_set.real.float(),
                                                 steering_vector_set.imag.float()).to(self.device)
        self.music_spectrum = MUSICSpectrum(self.steering_vector_set, self.device)
        self.batch_normalization = torch.nn.BatchNorm1d(16).to(device)
        self.positional_encoder = PositionalEncoding(16, device=device)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=16,
            nhead=8,
            dim_feedforward=1024,
            dropout=0,
            activation="relu",
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=False,
            device=None,
            dtype=None).to(device)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=3,
            norm=None).to(device)
        self.input_linear = nn.Linear(in_features=16, out_features=128).to(device)
        self.peak_finder = nn.Sequential(
            nn.Linear(in_features=360, out_features=16),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=16, out_features=6)
        ).to(device)

    def forward(self, x):
        # [batch_size, 2M=16, T=200]
        batch_size = x.shape[0]
        x = self.batch_normalization(x).to(self.device)
        # [batch_size, 2M=16, T=200] -> [T=200, batch_size, 2*M=16]
        x = x.permute(2, 0, 1).float().to(self.device)
        # [T=200, batch_size, 2*M=16]
        x = self.positional_encoder(x).to(self.device)
        # [T=200, batch_size, 2*M=16] -> [batch_size, T=200, 2M=16]
        x = x.permute(1, 0, 2).float().to(self.device)
        # [batch_size, T=200, 2M=16]
        x1 = self.encoder(x)  # Transformer Encoder network output becomes [size, 200,16]
        # [batch_size, 2M=16]
        x2 = torch.mean(x1, dim=1)  # Output becomes [size, 16]
        # [batch_size, 2*(M**2)=128]
        x3 = self.input_linear(x2).to(self.device)
        vector = x3  # Reused by the separate source-count classifier.
        # [batch_size, 2*(M**2)=128] -> [batch_size, 2M=16, M=8]
        x4 = x3.reshape(batch_size, 16, 8).to(self.device)
        # [batch_size, 2M=16, M=8] -> [batch_size, M=8, M=8]
        x5 = torch.complex(x4[:, :8, :], x4[:, 8:, :]).to(self.device)
        # Convert the learned complex matrix into a MUSIC pseudospectrum.
        spectrum = self.music_spectrum(x5).to(self.device)
        # Regress the fixed-length DOA vector from the spectrum.
        x8 = self.peak_finder(spectrum)
        return x8, vector


if __name__ == '__main__':
    from thop import profile
    model = TransMUSICModel()
    _, number_of_parameters = profile(model, inputs=(torch.randn(1, 16, 200).to(model.device),), verbose=False)
    print(f'number of parameters: {number_of_parameters}')
