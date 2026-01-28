import itertools

import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qpsk, _to_symbols

"""
1. In a 2X2 system, employ SVD precoding x=1/sqrt(2)*V*s
    - Decode with ML and ZF and show identical results
    - Plot the SER curve of each stream separately
"""


def _ml_detect(received_symbols, H_tilde, constellation, tx):
    """
    Performs Maximum Likelihood (ML) detection for Spatial Multiplexing.
    """

    # Create all possible transmitted symbol vectors
    possible_s_options = np.array(
        list(itertools.product(constellation, repeat=tx))
    )  # tx * 2^4 (for QAM4)

    # Calculate the expected received signal for each possible transmitted vector.
    possible_options = H_tilde @ possible_s_options.T  # num_symbols * tx * tx^4

    # Calculate squared Euclidean distance: ||y - H*s_candidate||^2 for all candidates.
    distances = np.linalg.norm(received_symbols - possible_options, axis=1) ** 2

    min_idx = np.argmin(distances, axis=1)
    return possible_s_options[min_idx]


def main(s_symbols, num_symbols, rx, tx, num_streams):
    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 40, 2)
    ser_array = np.zeros((len(snr_db_range), 6))

    for snr_db_idx, snr_db in enumerate(snr_db_range):
        # Convert SNR from dB to linear and calculate required noise power
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = 1 / snr_linear
        rho = np.sqrt(noise_power / 2)  # Noise scaling factor

        # Count a vector error if any symbol in the detected vector is wrong.
        detected_ml, detected_zf = bfm_step(
            s_symbols,
            num_symbols,
            rho,
            rx,
            num_streams=num_streams,
            tx=tx,
            detectors=[_ml_detect, _zf_detect],
        )
        errors_ml = s_symbols != detected_ml
        errors_zf = s_symbols != detected_zf

        ser_array[snr_db_idx][0] = np.sum(errors_ml) / (num_symbols * num_streams)
        ser_array[snr_db_idx][1] = np.sum(errors_zf) / (num_symbols * num_streams)
        ser_array[snr_db_idx][2] = np.sum(errors_ml[:, 0]) / num_symbols
        ser_array[snr_db_idx][3] = np.sum(errors_zf[:, 0]) / num_symbols
        ser_array[snr_db_idx][4] = np.sum(errors_ml[:, 1]) / num_symbols
        ser_array[snr_db_idx][5] = np.sum(errors_zf[:, 1]) / num_symbols

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    styles = [
        ("Precoded SM - ML 2x2", "r-"),
        ("Precoded SM - ZF 2x2", "b--"),
        ("Precoded SM-ML Stream-1", "g-"),
        ("Precoded SM-ZF Stream-1", "y--"),
        ("Precoded SM-ML Stream-2", "c-"),
        ("Precoded SM-ZF Stream-2", "m--"),
    ]
    for tp_idx, tp_ser in enumerate(ser_array.T):
        plt.semilogy(
            snr_db_range, tp_ser, styles[tp_idx][1], label=f"{styles[tp_idx][0]}"
        )

    plt.title("Symbol Error Rate (SER) vs. SNR for 2x2 QPSK with SVD Precoded")
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10**-5, 1)  # Set y-axis limits for log scale
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


def bfm_step(
    s_symbols,
    num_symbols,
    rho,
    rx,
    tx,
    num_streams,
    constellation=_to_symbols(np.arange(0, 4)),
    detectors=[_ml_detect],
):
    # Generate a batch of channels, one for each symbol
    # H has shape (num_symbols, rx, tx)
    H = _get_complex(num_symbols, col=tx, rows=rx)  # num_symbols * tx * rx Channels

    # Perform SVD on the batch of channels
    U, S, Vh = np.linalg.svd(H)

    # Precode the symbols.
    # x will have shape (num_symbols, tx)
    x, precode_matrix = precode(s_symbols, Vh, num_streams, tx)

    # Transmit through the channel.
    # H is (N, rx, tx), x is (N, tx, 1). y is (N, rx, 1).
    y = H @ x

    # Add AWGN
    n = _get_complex(num_symbols, col=rx).reshape(num_symbols, rx, 1)
    y += rho * n

    # For detection, we need the effective channel.
    H_eff = (1 / np.sqrt(tx)) * H @ precode_matrix

    # Detections
    return [detector(y, H_eff, constellation, num_streams) for detector in detectors]


# Precode the symbols.
# employ SVD precoding x=1/sqrt(2)*V*s
# x will have shape (num_symbols, tx)
def precode(s_symbols, Vh, num_streams, tx):
    V = np.conj(Vh).transpose(0, 2, 1)
    precode_matrix = V[
        :, :, :num_streams
    ]  # making sure precode matrix size according to stream num
    return (1 / np.sqrt(tx)) * precode_matrix @ s_symbols[
        :, :, np.newaxis
    ], precode_matrix


def _zf_detect(received_symbols, H_tilde, constellation, tx):
    """
    Performs Zero Forcing detection for Spatial Multiplexing.
    """

    H_tilde_h = np.conj(H_tilde).transpose(0, 2, 1)
    H_tilde_inv = np.linalg.pinv(H_tilde_h @ H_tilde) @ H_tilde_h
    s_hat = (H_tilde_inv @ received_symbols).squeeze(axis=-1)

    return _ls_detect(s_hat, constellation)


# Find the closest constellation symbol for each received symbol
def _ls_detect(received_symbols, constellation):
    distances = np.abs(received_symbols[:, :, np.newaxis] - constellation) ** 2

    # Find the index of the minimum distance for each symbol in the vector
    # Return the detected symbols from the constellation
    min_indices = np.argmin(distances, axis=2)
    return constellation[min_indices]


if __name__ == "__main__":

    num_symbols = int(1e6)
    tx, rx = (2, 2)

    _, s_symbols = _get_qpsk(num_symbols)
    num_streams = min(tx, rx)
    s_symbols_mtx = np.vstack([s_symbols[i::num_streams] for i in range(num_streams)]).T
    main(s_symbols_mtx, num_symbols // num_streams, rx, tx, num_streams)

    plt.show()
