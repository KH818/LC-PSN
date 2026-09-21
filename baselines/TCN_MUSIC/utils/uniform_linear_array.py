"""Represent a half-wavelength uniform linear array and its steering vectors."""

import numpy as np


class UniformLinearArray:
    def __init__(self,
                 number_of_elements,
                 spacing=None,
                 wave_length=None,
                 imperfection_magnitude=None,
                 ):
        self.number_of_elements = number_of_elements
        if spacing is not None and wave_length is not None:
            self.spacing = spacing
            self.wave_length = wave_length
            self.constant_phase = np.expand_dims(2 * np.pi * self.spacing / self.wave_length
                                                 * np.arange(0, self.number_of_elements, 1), axis=-1)
        else:
            self.constant_phase = np.expand_dims(np.pi * np.arange(0, self.number_of_elements, 1), axis=-1)

        self.imperfection_magnitude = imperfection_magnitude

    def steering_vector_degree(self, directions_of_arrival):
        ideal_response = np.exp(-1j * self.constant_phase @ np.expand_dims(np.sin(np.deg2rad(directions_of_arrival)),
                                                                           axis=-1).T, dtype=np.complex64)
        if self.imperfection_magnitude is None:
            return ideal_response
        else:
            return self.imperfect_response(ideal_response)

    def steering_vector_radian(self, directions_of_arrival):
        ideal_response = np.exp(-1j * self.constant_phase @ np.expand_dims(np.sin(directions_of_arrival), axis=-1).T,
                                dtype=np.complex64)
        if self.imperfection_magnitude is None:
            return ideal_response
        else:
            return self.imperfect_response(ideal_response)

    def imperfect_response(self, ideal_response):
        noise = \
            (np.random.normal(0, np.sqrt(self.imperfection_magnitude / 2), size=ideal_response.shape)
             + 1j * np.random.normal(0, np.sqrt(self.imperfection_magnitude / 2), size=ideal_response.shape))\
            .astype(np.complex64)

        imperfect_response = ideal_response + noise

        return imperfect_response

