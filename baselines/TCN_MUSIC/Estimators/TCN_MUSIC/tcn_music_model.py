"""Temporal-convolutional one-bit MUSIC regressor used as a learned baseline."""

import numpy as np
import torch
import torch.nn as nn
from utils.uniform_linear_array import UniformLinearArray
from utils.temporal_conv_net import TemporalConvNet
from utils.music_spectrum import MUSICSpectrum


class TemporalConvolutionalNetworkMUSICModel(nn.Module):
    """Summarize sign snapshots with a TCN, then regress DOAs through MUSIC features."""
    def __init__(self,
                 angle_grids=np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False),
                 rx_ula=UniformLinearArray(8),
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super(TemporalConvolutionalNetworkMUSICModel, self).__init__()
        self.device = device
        self.number_of_ula_elements = rx_ula.number_of_elements
        # Precompute the array manifold used to form the MUSIC spectrum.
        steering_vector_set = torch.from_numpy(rx_ula.steering_vector_radian(angle_grids))
        self.steering_vector_set = torch.complex(steering_vector_set.real.float(),
                                                 steering_vector_set.imag.float()).to(self.device)
        self.music_spectrum = MUSICSpectrum(self.steering_vector_set, self.device)
        self.batch_norm = nn.BatchNorm1d(16).to(device)
        self.temp_conv = TemporalConvNet(16, [64], 2, dropout=0)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = torch.nn.Flatten()
        self.reshaper_dense = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
        ).to(device)

        self.peak_finder = nn.Sequential(
            nn.Linear(in_features=360, out_features=32),
            nn.PReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.PReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.PReLU(),
            nn.Linear(in_features=32, out_features=6),
        ).to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        y = self.batch_norm(x)
        # Extract local temporal patterns before pooling across all snapshots.
        y = self.temp_conv(y)
        y = self.gap(y)
        y = self.flatten(y)
        y = self.reshaper_dense(y)
        feature_for_number_of_sources_classification = y  # Shared with the K classifier.
        real_noise_subspace = y.reshape(batch_size, 16, 8).to(self.device)
        complex_noise_subspace = \
            torch.complex(real_noise_subspace[:, :8, :].to(self.device),
                          real_noise_subspace[:, 8:, :].to(self.device))

        spectrum = self.music_spectrum(complex_noise_subspace)
        y = self.peak_finder(spectrum)
        return y, feature_for_number_of_sources_classification


if __name__ == '__main__':
    from thop import profile
    model = TemporalConvolutionalNetworkMUSICModel()
    _, number_of_parameters = profile(model, inputs=(torch.randn(1, 16, 200).to(model.device),), verbose=False)
    print(f'number of parameters: {number_of_parameters}')
