"""Generate narrowband complex snapshots received by a uniform linear array."""

import numpy as np
from utils.uniform_linear_array import UniformLinearArray
from utils.awgn import awgn


class SignalReceiver:
    def __init__(self,
                 number_of_snapshots=200,
                 rx_ula=UniformLinearArray(8),
                 default_correlation_coefficient=None):
        self.number_of_snapshots = number_of_snapshots
        self.rx_ula = rx_ula
        if default_correlation_coefficient is not None and not -1 <= default_correlation_coefficient <= 1:
            raise ValueError("Correlation coefficient must be between -1 and 1.")
        self.default_correlation_coefficient = default_correlation_coefficient

    def receive(self,
                directions_of_arrival,
                number_of_sources=None,
                seed=None,
                correlation_coefficient=None,
                snr_db=None):
        directions_of_arrival = np.atleast_1d(directions_of_arrival)

        if correlation_coefficient is not None and not -1 <= correlation_coefficient <= 1:
            raise ValueError("Correlation coefficient must be between -1 and 1.")

        if correlation_coefficient is None:
            correlation_coefficient = self.default_correlation_coefficient

        if number_of_sources is None:
            number_of_sources = directions_of_arrival.size

        if seed is not None:
            np.random.seed(seed)

        if correlation_coefficient is None:
            emitted_signals = np.random.normal(0, np.sqrt(0.5), (number_of_sources, self.number_of_snapshots)) \
                + 1j * np.random.normal(0, np.sqrt(0.5), (number_of_sources, self.number_of_snapshots))
        else:
            covariance_matrix = np.zeros((2 * number_of_sources, 2 * number_of_sources))

            for i in range(number_of_sources):
                # Variance of real and imaginary parts is 1/2
                covariance_matrix[2 * i, 2 * i] = 0.5  # Var(Re(signal_i))
                covariance_matrix[2 * i + 1, 2 * i + 1] = 0.5  # Var(Im(signal_i))

                for j in range(i + 1, number_of_sources):
                    # Covariance between real parts of signal i and j
                    covariance_matrix[2 * i, 2 * j] = 0.5 * correlation_coefficient
                    covariance_matrix[2 * j, 2 * i] = 0.5 * correlation_coefficient

                    # Covariance between imaginary parts of signal i and j
                    covariance_matrix[2 * i + 1, 2 * j + 1] = 0.5 * correlation_coefficient
                    covariance_matrix[2 * j + 1, 2 * i + 1] = 0.5 * correlation_coefficient

                    # Assuming no correlation between real part of one and imaginary part of another
                    covariance_matrix[2 * i, 2 * j + 1] = 0
                    covariance_matrix[2 * j + 1, 2 * i] = 0
                    covariance_matrix[2 * i + 1, 2 * j] = 0
                    covariance_matrix[2 * j, 2 * i + 1] = 0

            # Generate the correlated real and imaginary parts
            correlated_real_imag = np.random.multivariate_normal(
                mean=np.zeros(2 * number_of_sources),
                cov=covariance_matrix,
                size=self.number_of_snapshots,
                check_valid='warn',
                tol=1e-8
            ).T

            # Combine real and imaginary parts to form complex signals
            emitted_signals = np.zeros((number_of_sources, self.number_of_snapshots), dtype=complex)
            for i in range(number_of_sources):
                emitted_signals[i, :] = (
                        correlated_real_imag[2 * i, :] + 1j * correlated_real_imag[2 * i + 1, :]
                )

        steering_vectors = self.rx_ula.steering_vector_radian(directions_of_arrival)
        received_signal = \
            steering_vectors @ emitted_signals

        received_signal = received_signal.astype(np.complex64)

        # Additive white Gaussian noise is applied to the (unquantized) received
        # signal so that any downstream quantization sees the *noisy* signal,
        # i.e. y = A x + n  (paper eq. (1)/(2)). This keeps the noise strictly
        # before any 1-bit quantization: z = Q1(A x + n).
        if snr_db is not None:
            received_signal = awgn(received_signal, snr_db, 0).astype(np.complex64)

        return received_signal
