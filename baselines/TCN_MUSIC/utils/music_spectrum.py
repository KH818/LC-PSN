"""Evaluate a MUSIC pseudospectrum from a learned complex subspace matrix."""

import torch
import torch.nn as nn


class MUSICSpectrum(nn.Module):
    def __init__(self,
                 steering_vector_set,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(MUSICSpectrum, self).__init__()
        self.device = device
        self.steering_vector_set = steering_vector_set.to(self.device)
        self.steering_vector_set_conj = torch.conj(self.steering_vector_set).to(self.device)

    def forward(self, noise_subspace):
        x1 = torch.matmul(noise_subspace @ torch.conj(noise_subspace.permute(0, 2, 1).to(self.device)),
                          self.steering_vector_set).to(self.device)
        x2 = torch.mul(x1, self.steering_vector_set_conj)
        x3 = torch.sum(x2, dim=1)
        spectrum = (1.0 / torch.abs(x3)).to(self.device)
        return spectrum
