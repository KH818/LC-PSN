"""Recurrent DA-MUSIC regressor used as a learned comparison model."""

import numpy as np
import torch
import torch.nn as nn
from utils.uniform_linear_array import UniformLinearArray
from utils.eigenvalue_decomposition import EigenValueDecomposition
from utils.music_spectrum import MUSICSpectrum


class DeepAugmentedMUSICModel(nn.Module):
    def __init__(self,
                 angle_grids=np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False),
                 rx_ula=UniformLinearArray(8),
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(DeepAugmentedMUSICModel, self).__init__()
        self.device = device
        self.number_of_ula_elements = rx_ula.number_of_elements
        # Precompute the array manifold used to form the MUSIC spectrum.
        steering_vector_set = torch.from_numpy(rx_ula.steering_vector_radian(angle_grids))
        self.steering_vector_set = torch.complex(steering_vector_set.real,
                                                 steering_vector_set.imag).to(self.device)
        self.music_spectrum = MUSICSpectrum(self.steering_vector_set, self.device)
        self.batch_normalization = torch.nn.BatchNorm1d(2 * self.number_of_ula_elements).to(self.device)
        self.evd = EigenValueDecomposition(self.number_of_ula_elements, self.device).to(self.device)
        self.gru_layer = nn.GRU(2 * self.number_of_ula_elements,
                                2 * self.number_of_ula_elements,
                                batch_first=True).to(self.device)
        self.linear_reshaper = nn.Linear(in_features=16, out_features=128).to(self.device)
        self.mlp_in_noise_subspace_selector = nn.Sequential(
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=16, out_features=8),
            nn.Sigmoid()
        ).to(self.device)
        self.peak_finder = nn.Sequential(
            nn.Linear(in_features=360, out_features=16),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=16, out_features=6)
        ).to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.batch_normalization(x)
        # [batch_size, 2M=16, T=200] -> [batch_size, T=200, 2M=16]
        x = x.permute(0, 2, 1).float().to(self.device)
        # [batch_size, T=200, 2M=16] -> [batch_size, 2M=16]
        _, x = self.gru_layer(x)
        # [batch_size, 2M=16] -> [batch_size, 2*(M**2)=128]
        x = self.linear_reshaper(x)
        # [batch_size, 2*(M**2)=128] -> [batch_size, 2M=16, M=8]
        x = x.reshape(batch_size, 16, 8).to(self.device)
        # eigenvalue decomposition
        eigenvalues, eigenvector = self.evd(x)
        # [batch_size, 2*(M**2)=128] -> [batch_size, 2M=16]
        eigenvalues = torch.cat([eigenvalues.real,
                                 eigenvalues.imag], dim=1).to(self.device)
        # noise subspace selector
        # [batch_size, 2M=16] -> [batch_size, M=8]
        q = self.mlp_in_noise_subspace_selector(eigenvalues)
        # [batch_size, M=8] -> [batch_size, M=8, M=8]
        q = torch.diag_embed(q)
        # [batch_size, M=8, M=8]
        q = torch.complex(q, torch.zeros([batch_size, 8, 8]).float().to(self.device))
        # [batch_size, M=8, M=8] * [batch_size, M=8, M=8] -> [batch_size, M=8, M=8]
        noise_subspace = torch.matmul(q, eigenvector).to(self.device)
        # calculate music spectrum
        spectrum = self.music_spectrum(noise_subspace)
        # peak finder
        estimated_peaks = self.peak_finder(spectrum)

        return estimated_peaks, eigenvalues


if __name__ == '__main__':
    from thop import profile
    model = DeepAugmentedMUSICModel()
    _, number_of_parameters = profile(model, inputs=(torch.randn(1, 16, 200).to(model.device),), verbose=False)
    print(f'number of parameters: {number_of_parameters}')
