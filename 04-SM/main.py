"""
Produce the SER curve for 2X2 SM with ML detection
Compare with STC 2X1
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qpsk, _to_symbols

"""
Repeat the Eigen BF case for 2Rx X 2Tx and 2Rx X 4Tx
Compare with the STC 2X2
"""

num_symbols = 100000


def main(s_int, s_symbols, num_symbols):

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
        sm_errors = np.sum(np.any(s_symbols != sm_detected_symbols, axis=1))
        sm_ser.append(sm_errors / num_symbols)

        # --- STC Path ---
        stc_detected_symbols = stc_step(s_symbols, num_symbols, rho, 2)
        stc_errors = np.sum(np.any(s_symbols != stc_detected_symbols, axis=1))
        stc_ser.append(stc_errors / num_symbols)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    plt.semilogy(snr_db_range, sm_ser, 'r-', label='SM 2x2')
    plt.semilogy(snr_db_range, stc_ser, 'g--', label='STC 2x1')

    plt.title('Symbol Error Rate (SER) vs. SNR for QPSK with Eigen-Beamforming')
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
    y = (H_tilde @ s_symbols[:, :, np.newaxis]).squeeze()

    # Add AWGN
    n = _get_complex(num_symbols, col=rx)
    y += rho * n

    # Detections
    return _sm_detect(y, H_tilde, constellation, tx)


def _sm_detect(received_symbols, H_tilde, constellation, tx):

    # Create all possible transmitted symbol vectors
    possible_s_options = np.array(list(itertools.product(constellation, repeat=tx)))    #  tx * 2^4 (for QAM4)

    # Calculate the expected received signal for each possible transmitted vector.
    # || y - H*s_candidate ||^2 for all candidates.
    possible_options = H_tilde @ possible_s_options.T # num_symbols * tx * tx^4

    # Calculate squared Euclidean distance.
    distances = np.pow(np.linalg.norm(received_symbols[:,:,np.newaxis] - possible_options, axis=1), 2)

    min_idx = np.argmin(distances, axis=1)
    return possible_s_options[min_idx]



def stc_step(s_stc_symbols, num_stc, rho, tx, constellation=_to_symbols(np.arange(0, 4))):
    # Alamouti STC is for 2 Tx antennas and is typically demonstrated with 1 Rx antenna.
    rx = 1
    # Channel h is (num_stc, rx, tx)
    h = _get_complex(num_stc, rx, tx)
    h1, h2 = h[:, 0, 0], h[:, 0, 1] # Channel gains for Tx1 and Tx2

    # Construct the effective Alamouti channel matrix H_eff for each symbol
    # H_eff has shape (num_stc, 2, 2)
    H = np.array([[h1, h2], [np.conj(h2), -np.conj(h1)]]).transpose(2, 0, 1)

    n_stc = np.array([_get_complex(num_stc), np.conj(_get_complex(num_stc))])

    # STC Received signals (2x2 MISO)
    y_stc = ((H / np.sqrt(2)) @ s_stc_symbols[:, :, np.newaxis]).squeeze(axis=-1)
    y_stc += (rho * n_stc).T  # Add noise
    y_stc[:, 1] = y_stc[:, 1].conj()

    # Equalization/Detection
    s_stc_hat = (np.linalg.pinv(H) @ y_stc[:, :, np.newaxis]).squeeze(axis=-1)

    return _to_symbols(_ls_detect(s_stc_hat, constellation))


# Find the closest constellation symbol for each received symbol
def _ls_detect(received_symbols, constellation):
    distances = np.abs(received_symbols[:, np.newaxis, :] - np.array([np.array([constellation] * 2).transpose()])) ** 2

    return np.argmin(distances, axis=1)


if __name__ == '__main__':
    num_symbols = int(100e3)
    M = 2
    N = 2

    s_int, s_symbols = _get_qpsk(num_symbols)

    s_symbols = np.vstack((s_symbols[::M], s_symbols[1::M])).T
    s_int = np.vstack((s_int[::M], s_int[1::M])).T

    main(s_int, s_symbols, num_symbols // 2)
