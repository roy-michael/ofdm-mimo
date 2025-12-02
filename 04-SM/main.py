import itertools

import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qpsk, _to_symbols

"""
Produce the SER curve for 2X2 SM with ML detection
Compare with STC 2X1
"""


def main(s_symbols, num_symbols):
    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 40, 2)
    sm_ser = []
    stc_ser = []

    for snr_db in snr_db_range:

        # Convert SNR from dB to linear and calculate required noise power
        snr_linear = 10 ** (snr_db / 10.0)
        rho = np.sqrt(0.5) / np.sqrt(snr_linear)  # Noise scaling factor

        # --- SM Path ---
        # Count a vector error if any symbol in the detected vector is wrong.
        sm_detected_symbols = sm_step(s_symbols, num_symbols, rho, 2, 2)
        sm_total_errors = np.sum(s_symbols != sm_detected_symbols)
        sm_ser.append(sm_total_errors / (num_symbols * 2))

        # --- STC Path ---
        stc_detected_symbols = stc_step(s_symbols, num_symbols, rho, 2)
        stc_total_errors = np.sum(s_symbols != stc_detected_symbols)
        stc_ser.append(stc_total_errors / num_symbols * 2)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    plt.semilogy(snr_db_range, sm_ser, 'r-', label='SM 2x2')
    plt.semilogy(snr_db_range, stc_ser, 'g--', label='STC 2x1')

    plt.title('Symbol Error Rate (SER) vs. SNR for Spatial Multiplexing')
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10 ** -5, 1)  # Set y-axis limits for log scale
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


def sm_step(s_symbols, num_symbols, rho, rx, tx, constellation=_to_symbols(np.arange(0, 4))):
    # Generate a batch of channels, one for each symbol
    # H has shape (num_symbols, rx, tx)
    H = _get_complex(num_symbols, rx, tx)  # num_symbols * tx * rx Channels
    H_tilde = (1 / np.sqrt(tx)) * H

    # Transmit through the channel.
    # H is (N, rx, tx), x is (N, tx, 1). y is (N, rx, 1).
    y = H_tilde @ s_symbols[:, :, np.newaxis]

    # Add AWGN
    n = _get_complex(num_symbols, col=rx).reshape(num_symbols, rx, 1)
    y += rho * n

    # Detections
    return _sm_ml_detect(y, H_tilde, constellation, tx)


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


def stc_step(s_stc_symbols, num_stc, rho, tx, constellation=_to_symbols(np.arange(0, 4))):
    """
    Implements Alamouti STC using the effective matrix form:
    y_tilde = (1/sqrt(2)) * H_eff * s + rho * n_tilde
    """
    rx = 1

    # Channel Setup
    # h has shape (num_stc, rx, tx).
    h = _get_complex(num_stc, rx, tx)
    h0, h1 = h[:, 0, 0], h[:, 0, 1]

    # Construct the Effective Channel Matrix H
    H = np.array([
        [h0, h1],
        [np.conj(h1), -np.conj(h0)]
    ]).transpose(2, 0, 1)

    # Construct the Signal Vector s
    s_vec = s_stc_symbols[:, :, np.newaxis]  # Shape (num_stc, 2, 1)

    # Construct the Effective Noise Vector
    n0 = _get_complex(num_stc)
    n1 = _get_complex(num_stc)

    n_tilde = np.array([n0, np.conj(n1)]).T[:, :, np.newaxis]

    # Compute the Received Signal y_tilde
    y_tilde = (1 / np.sqrt(2)) * (H @ s_vec) + rho * n_tilde

    # Detection (Least Squares)
    H_inv = np.linalg.pinv(H)
    s_stc_hat = (H_inv @ y_tilde).squeeze()

    return _ls_detect(s_stc_hat, constellation)


# Find the closest constellation symbol for each received symbol
def _ls_detect(received_symbols, constellation):

    distances = np.abs(received_symbols[:, :, np.newaxis] - constellation) ** 2

    # Find the index of the minimum distance for each symbol in the vector
    # Return the detected symbols from the constellation
    min_indices = np.argmin(distances, axis=2)
    return constellation[min_indices]


if __name__ == '__main__':
    num_symbols = int(1e6)
    M = 2
    N = 2

    _, s_symbols = _get_qpsk(num_symbols)

    s_symbols = np.vstack((s_symbols[::M], s_symbols[1::M])).T
    main(s_symbols, num_symbols // 2)
