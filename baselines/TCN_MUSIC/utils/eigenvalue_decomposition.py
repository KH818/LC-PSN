"""Compute batched covariance eigenvalues and eigenvectors for DA-MUSIC."""

import torch
import torch.nn as nn


class EigenValueDecomposition(nn.Module):
    def __init__(self,
                 matrix_size,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(EigenValueDecomposition, self).__init__()
        self.matrix_size = matrix_size
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        # [batch_size, 2*size=16, size=8] -> [batch_size, size=8, size=8]
        x = torch.complex(x[:, :self.matrix_size, :],
                          x[:, self.matrix_size:, :]).to(self.device)
        # Eigenvalue decomposition (NOT GUARANTEED TO BE IN ANY SPECIFIC ORDER)
        # [batch_size, size=8, size=8] -> [batch_size, size=8], [batch_size, size=8, size=8] respectively
        x_eigenvalues, x_eigenvectors = torch.linalg.eig(x)
        # [batch_size, size=8] -> [batch_size, size=8]
        abs_x_eigenvalues = torch.abs(x_eigenvalues).to(self.device)
        # [batch_size, size=8] -> [batch_size, size=8]
        _, idx = torch.sort(abs_x_eigenvalues, dim=1)
        x_eigenvectors = x_eigenvectors.to(self.device)
        sorted_eigenvalues = torch.zeros([batch_size, self.matrix_size],
                                         dtype=torch.complex64).to(self.device)
        sorted_eigenvectors = torch.zeros([batch_size,
                                           self.matrix_size,
                                           self.matrix_size], dtype=torch.complex64).to(self.device)
        for sample_index_in_batch in range(batch_size):
            sorted_eigenvectors[sample_index_in_batch, :, :] = \
                x_eigenvectors[sample_index_in_batch].index_select(dim=1,
                                                                   index=idx[sample_index_in_batch]).unsqueeze(dim=0)
            sorted_eigenvalues[sample_index_in_batch, :] = \
                x_eigenvalues[sample_index_in_batch].index_select(dim=0,
                                                                  index=idx[sample_index_in_batch]).unsqueeze(dim=0)

        return sorted_eigenvalues, sorted_eigenvectors
