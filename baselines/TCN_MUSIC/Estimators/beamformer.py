"""Implement a conventional beamformer used by legacy baseline utilities."""

import numpy as np
from scipy.signal import find_peaks


class Beamformer:
    def __init__(self,
                 angle_grids,
                 rx_ula,
                 default_threshold=0.1,
                 ):
        self.angle_grids = angle_grids
        self.number_of_angle_grids = angle_grids.shape[0]
        self.rx_ula = rx_ula
        self.steering_vector_set = rx_ula.steering_vector_radian(self.angle_grids)
        self.steering_vector_set_conjugated_transpose = self.steering_vector_set.conj().T
        self.default_threshold = default_threshold

    def estimate(self, received_signal, number_of_sources=None, threshold=None):
        if threshold is None:
            threshold = self.default_threshold
        covariance_matrix_of_received_signal = np.cov(received_signal)
        spectrum = np.zeros(self.number_of_angle_grids)
        for angle_grid_index in range(self.number_of_angle_grids):
            spectrum[angle_grid_index] = \
                np.abs(self.steering_vector_set_conjugated_transpose[angle_grid_index, :] @
                       covariance_matrix_of_received_signal @ self.steering_vector_set[:, angle_grid_index])

        peak_indices, properties = find_peaks(spectrum, prominence=(None, None))
        if number_of_sources is None:
            prominence_threshold = threshold * np.max(properties["prominences"])
            dominant_peaks_indices = peak_indices[properties["prominences"] >= prominence_threshold]
            number_of_sources = dominant_peaks_indices.shape[0]
        else:
            dominant_peaks_indices = peak_indices[np.argsort(properties["prominences"])[::-1]]

        estimated_angles = self.angle_grids[dominant_peaks_indices]

        return estimated_angles, number_of_sources
