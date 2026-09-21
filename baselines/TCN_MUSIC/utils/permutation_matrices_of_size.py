"""Generate permutation matrices used by the baseline matching loss."""

import numpy as np
from itertools import permutations


def permutation_matrices_of_size(n):
    # Get all permutations of row indices
    perm_list = list(permutations(range(n)))

    # Generate a permutation matrix for each permutation of indices
    permutation_matrices = []
    for perm in perm_list:
        matrix = np.zeros((n, n), dtype=int)
        # Place a 1 in each row at the column index specified by the permutation
        for row, col in enumerate(perm):
            matrix[row, col] = 1
        permutation_matrices.append(matrix)

    return np.array(permutation_matrices)
