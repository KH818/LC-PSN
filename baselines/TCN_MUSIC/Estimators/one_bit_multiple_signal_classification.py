"""Apply conventional MUSIC to covariance formed from one-bit snapshots."""

import numpy as np
from Estimators.multiple_signal_classification import MUSIC
from scipy.signal import find_peaks


class OneBitMUSIC(MUSIC):
    def estimate(self, one_bit_quantized_received_signal, number_of_sources=None, threshold=None):
        if threshold is None:
            threshold = self.default_threshold
        covariance_matrix_of_received_signal = np.cov(one_bit_quantized_received_signal)
        covariance_matrix_of_received_signal = np.pi / 2 * covariance_matrix_of_received_signal
        covariance_matrix_of_received_signal = np.sin(np.real(covariance_matrix_of_received_signal)) \
            + 1j * np.sin(np.imag(covariance_matrix_of_received_signal))

        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix_of_received_signal)
        if number_of_sources is None:
            number_of_sources = np.sum(np.abs(eigen_values) > (np.abs(eigen_values[-1]) + threshold))
        noise_subspace = eigen_vectors[:, number_of_sources:]
        spectrum = np.zeros(self.number_of_angle_grids)
        for angle_grid_index in range(self.number_of_angle_grids):
            spectrum[angle_grid_index] = \
                np.abs(1. / (self.steering_vector_set_conjugated_transpose[angle_grid_index, :] @ noise_subspace
                             @ noise_subspace.conj().T @ self.steering_vector_set[:, angle_grid_index] + 1e-8))
        peak_indices, _ = find_peaks(spectrum)
        peak_indices = peak_indices[np.argsort(spectrum[peak_indices])][::-1]
        estimated_angles = self.angle_grids[peak_indices]

        return estimated_angles, number_of_sources
