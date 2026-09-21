"""Implement the conventional MUSIC estimator used by baseline utilities."""

import numpy as np
from scipy.signal import find_peaks


class MUSIC:
    def __init__(self,
                 angle_grids,
                 rx_ula,
                 default_threshold=0.25,
                 ):
        self.angle_grids = angle_grids
        self.number_of_angle_grids = angle_grids.shape[0]
        self.rx_ula = rx_ula
        self.number_of_ula_elements = rx_ula.number_of_elements
        self.steering_vector_set = rx_ula.steering_vector_radian(self.angle_grids)
        self.steering_vector_set_conjugated_transpose = self.steering_vector_set.conj().T
        self.default_threshold = default_threshold

    def estimate(self, received_signal, number_of_sources=None, threshold=None):
        if threshold is None:
            threshold = self.default_threshold
        covariance_matrix_of_received_signal = np.cov(received_signal)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix_of_received_signal)
        if number_of_sources is None:
            number_of_sources = np.sum(np.abs(eigen_values) > (np.abs(eigen_values[-1]) + threshold))
        noise_subspace = eigen_vectors[:, number_of_sources:]
        spectrum = np.zeros(self.number_of_angle_grids)
        for angle_grid_index in range(self.number_of_angle_grids):
            spectrum[angle_grid_index] = \
                np.abs(1./(self.steering_vector_set_conjugated_transpose[angle_grid_index, :] @ noise_subspace
                           @ noise_subspace.conj().T @ self.steering_vector_set[:, angle_grid_index] + 1e-8))
        peak_indices, _ = find_peaks(spectrum)
        peak_indices = peak_indices[np.argsort(spectrum[peak_indices])][::-1]
        estimated_angles = self.angle_grids[peak_indices]

        return estimated_angles, number_of_sources
