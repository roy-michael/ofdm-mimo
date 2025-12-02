import itertools

import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qpsk, _to_symbols

"""
Produce the SER curve for 2X2, 4X2 SM with ML and ZF
detection
"""


def main(s_symbols, num_symbols, rx, tx):
    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 40, 2)
    sm_ser_ml = []
    sm_ser_zf = []

    for snr_db in snr_db_range:
        # Convert SNR from dB to linear and calculate required noise power
        snr_linear = 10 ** (snr_db / 10.0)
        rho = np.sqrt(1 / rx) / np.sqrt(snr_linear)  # Noise scaling factor

        # Count a vector error if any symbol in the detected vector is wrong.
        sm_detected_symbols_ml = sm_step(s_symbols, num_symbols, rho, rx, tx=tx, detector=_sm_ml_detect)
        sm_detected_symbols_zf = sm_step(s_symbols, num_symbols, rho, rx, tx=tx, detector=_sm_zf_detect)

        total_symbol_errors = np.sum(s_symbols != sm_detected_symbols_ml)
        sm_ser_ml.append(total_symbol_errors / (num_symbols * tx))

        sm_total_errors = np.sum(s_symbols != sm_detected_symbols_zf)
        sm_ser_zf.append(sm_total_errors / (num_symbols * tx))

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    plt.semilogy(snr_db_range, sm_ser_ml, 'r-', label=f'SM ML {tx}x{rx}')
    plt.semilogy(snr_db_range, sm_ser_zf, 'g--', label=f'SM ZF {tx}x{rx}')

    plt.title('Symbol Error Rate (SER) vs. SNR for Spatial Multiplexing')
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10 ** -5, 1)  # Set y-axis limits for log scale
    plt.legend()
    plt.grid(True, which='both')


def _sm_ml_detect(received_symbols, H_tilde, constellation, tx):
    """
    Performs Maximum Likelihood (ML) detection for Spatial Multiplexing.
    """

    # Create all possible transmitted symbol vectors
    possible_s_options = np.array(list(itertools.product(constellation, repeat=tx)))  # tx * 2^4 (for QAM4)

    # Calculate the expected received signal for each possible transmitted vector.
    possible_options = H_tilde @ possible_s_options.T  # num_symbols * tx * tx^4

    # Calculate squared Euclidean distance: ||y - H*s_candidate||^2 for all candidates.
    distances = np.linalg.norm(received_symbols - possible_options, axis=1) ** 2

    min_idx = np.argmin(distances, axis=1)
    return possible_s_options[min_idx]


def sm_step(
        s_symbols,
        num_symbols,
        rho,
        rx,
        tx,
        constellation=_to_symbols(np.arange(0, 4)),
        detector=_sm_ml_detect
):
    # Generate a batch of channels, one for each symbol
    # H has shape (num_symbols, rx, tx)
    H = _get_complex(num_symbols, col=tx, rows=rx)  # num_symbols * rx * tx Channels
    H_tilde = (1 / np.sqrt(rx)) * H

    # Transmit through the channel.
    # H is (N, rx, tx), s is (N, tx, 1). y is (N, rx, 1).
    y = H_tilde @ s_symbols[:, :, np.newaxis]

    # Add AWGN
    n = _get_complex(num_symbols, col=rx).reshape(num_symbols, rx, 1)
    y += rho * n

    # Detections
    return detector(y, H_tilde, constellation, tx)


def _sm_zf_detect(received_symbols, H_tilde, constellation, tx):
    """
    Performs Zero Forcing detection for Spatial Multiplexing.
    """

    H_tilde_inv = (
            np.sqrt(tx) *
            np.linalg.pinv(
                np.conj(H_tilde).transpose(0, 2, 1) @
                H_tilde
            ) @
            np.conj(H_tilde).transpose(0, 2, 1))
    s_hat = (H_tilde_inv @ received_symbols).squeeze(axis=-1)

    return _ls_detect(s_hat, constellation)


# Find the closest constellation symbol for each received symbol
def _ls_detect(received_symbols, constellation):
    distances = np.abs(received_symbols[:, :, np.newaxis] - constellation) ** 2

    # Find the index of the minimum distance for each symbol in the vector
    # Return the detected symbols from the constellation
    min_indices = np.argmin(distances, axis=2)
    return constellation[min_indices]


if __name__ == '__main__':

    num_symbols = int(1e6)
    tx_rx_configs = [(2, 2), (2, 4)]

    _, s_symbols = _get_qpsk(num_symbols)

    for tx, rx in tx_rx_configs:
        s_symbols_mtx = np.vstack([s_symbols[i::tx] for i in range(tx)]).T
        main(s_symbols_mtx, num_symbols // tx, rx, tx)

    plt.show()
