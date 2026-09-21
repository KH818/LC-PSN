"""Generate noisy one-bit array snapshots by quantizing after noise is added."""

import numpy as np
from utils.signal_receiver import SignalReceiver


class OneBitSignalReceiver(SignalReceiver):
    def receive(self,
                directions_of_arrival,
                number_of_sources=None,
                seed=None,
                snr_db=None,
                one_bit_quantization_threshold=0):
        # Paper model (TransMUSIC eq. (2) / TCN-MUSIC eq. (16)):
        #     z(t) = Q1( A x(t) + n(t) )
        # The additive noise MUST be injected before the one-bit quantizer.
        # We therefore let the base receiver add the AWGN (when snr_db is given)
        # and only then apply the sign-based one-bit quantization.
        received_signal = super().receive(directions_of_arrival,
                                           number_of_sources,
                                           seed,
                                           snr_db=snr_db)
        one_bit_quantized_received_signal = 1 / np.sqrt(2) \
            * (np.where(received_signal.real > one_bit_quantization_threshold, 1, -1)
               + 1j * np.where(received_signal.imag > one_bit_quantization_threshold, 1, -1))
        return one_bit_quantized_received_signal
