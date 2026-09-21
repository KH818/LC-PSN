"""Compute permutation-invariant wrapped angular error for DOA regression."""

import torch
import math
import numpy as np
from utils.permutation_matrices_of_size import permutation_matrices_of_size


class MinimalPermutationRMSPE:
    def __init__(self,
                 maximum_size=5,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.permutations = []
        for size in range(1, maximum_size + 1):
            permutation = permutation_matrices_of_size(size)
            permutation = permutation.reshape(math.factorial(size) * size,
                                              size).swapaxes(0, 1)
            self.permutations.append(permutation)

    def minimal_permutation_rmspe(self, estimations, true):
        number_of_sources = true.shape[1]
        all_permutations = torch.matmul(true,
                                        torch.from_numpy(self.permutations[number_of_sources - 1])
                                        .float().to(self.device))
        all_permutations = all_permutations.reshape(math.factorial(number_of_sources),
                                                    number_of_sources)
        differences_mod = torch.fmod(all_permutations - estimations + np.pi / 2, np.pi) - np.pi / 2
        rspe, _ = torch.min(torch.mean(differences_mod ** 2, dim=1) ** 0.5, dim=0)
        rmspe = torch.mean(rspe)
        return rmspe

    def loss(self, estimations, true):
        batch_size = true.shape[0]
        true_number_of_sources = torch.argmax(true, dim=1)
        loss_summation = torch.zeros(1).to(self.device)
        for batch_index, number_of_sources in enumerate(true_number_of_sources):
            loss_summation += \
                self.minimal_permutation_rmspe(estimations[batch_index, :number_of_sources].unsqueeze(dim=0),
                                               true[batch_index, :number_of_sources].unsqueeze(dim=0))
        batch_loss = loss_summation / batch_size
        return batch_loss
