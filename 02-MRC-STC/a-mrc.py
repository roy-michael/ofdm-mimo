import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qpsk, _to_symbols, _ml_detect

"""
Repeat the MRC case (Tx, channel, Rx) and reproduce the
SER curve for N = 2, 4 Rx antennas

This script simulates a Multiple-Input Multiple-Output (MIMO) system using Maximal Ratio Combining (MRC)
for a QPSK modulation scheme. It calculates and plots the Symbol Error Rate (SER) versus Signal-to-Noise Ratio (SNR)
for different numbers of receive antennas (N=2 and N=4).
"""

num_symbols = 100000
amplitude = 1


def main():
    """
    Main function to run the MRC simulation.
    It generates QPSK symbols, simulates the channel with AWGN, applies MRC, and calculates SER.
    """
    s_int, s_symbols = _get_qpsk(num_symbols)

    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 40, 2)

    mrc_ns = [2, 4]
    ser_array = np.zeros((len(snr_db_range), len(mrc_ns)))

    for mrc_idx, mrc_n in enumerate(mrc_ns):

        # --- Run Simulation for each SNR value ---
        for snr_db_idx, snr_db in enumerate(snr_db_range):
            # Convert SNR from dB to linear and calculate required noise power
            snr_linear = 10 ** (snr_db / 10.0)
            # Signal power is 0.5 (from _to_symbols in common.py), so N0 = 0.5 / SNR
            rho = np.sqrt(0.5) / np.sqrt(snr_linear)  # Noise scaling factor

            n = _get_complex(num_symbols, mrc_n)  # Generate AWGN for N receive antennas
            h = _get_complex(num_symbols, mrc_n)  # Generate Rayleigh fading channel coefficients for N receive antennas

            y = s_symbols.reshape(-1, 1) * h + rho * n  # Received signals at each antenna

            # Apply Maximal Ratio Combining (MRC)
            s_hat = np.sum(np.conj(h) * y, axis=1) / np.sum((np.abs(h) ** 2), axis=1)

            # Perform Maximum Likelihood (ML) detection
            constellation = _to_symbols(np.arange(0, 4))
            s_int_detected = _ml_detect(s_hat, constellation)

            # Calculate Symbol Error Rate (SER)
            ser_array[snr_db_idx, mrc_idx] = np.sum(s_int != s_int_detected) / num_symbols

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    styles = ['r-', 'b-']
    for mrc_idx, mrc_n in enumerate(mrc_ns):
        plt.semilogy(snr_db_range, ser_array[:, mrc_idx], styles[mrc_idx], label=f'MRC (N={mrc_n})')

    plt.title('Symbol Error Rate (SER) vs. SNR for QPSK with MRC')
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10 ** -5, 1)  # Set y-axis limits for log scale
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


if __name__ == '__main__':
    main()
