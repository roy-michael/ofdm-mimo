"""
This script simulates a Single-Input Single-Output (SISO) communication system
with QPSK modulation over AWGN and Rayleigh fading channels.
It calculates and plots the Symbol Error Rate (SER) versus Signal-to-Noise Ratio (SNR).
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt

num_symbols = 500000


# Main simulation function

def main():
    # QPSK symbols
    s_int = np.random.randint(0, 4, num_symbols)
    s_degrees = s_int * 360 / 4.0 + 45  # 45, 135, 225, 315 degrees
    s_radians = np.deg2rad(s_degrees)
    s_symbols = np.exp(1j * s_radians)  # normalized QPSK complex symbols (Power=1)

    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 40, 2)

    ser_awgn = []
    ser_rayleigh = []

    # --- Run Simulation for each SNR value ---
    for snr_db in snr_db_range:
        # Convert SNR from dB to linear and calculate required noise power
        snr_linear = 10 ** (snr_db / 10.0)
        rho = 1 / np.sqrt(snr_linear)

        # Generate AWGN with the calculated power
        # Noise power is normalized to 1 before scaling by rho
        n = np.squeeze(np.random.normal(size=(num_symbols, 2), loc=0, scale=1).view(np.complex128) / np.sqrt(2))  # AWGN

        # Channels
        h_awgn = 1
        h_rayleigh = np.squeeze(
            np.random.normal(size=(num_symbols, 2), loc=0, scale=1).view(np.complex128) / np.sqrt(2))

        # Received signals
        y_awgn = s_symbols * h_awgn + rho * n
        y_rayleigh = s_symbols * h_rayleigh + rho * n

        # Equalization
        s_hat_awgn = y_awgn / h_awgn
        s_hat_rayleigh = y_rayleigh / h_rayleigh

        # Detections
        constellation = _to_symbols(np.arange(0, 4))
        s_int_detected_awgn = _ml_detect(s_hat_awgn, constellation)
        s_int_detected_rayleigh = _ml_detect(s_hat_rayleigh, constellation)

        # Count the errors and calculate SER
        ser_awgn.append(np.sum(s_int != s_int_detected_awgn) / num_symbols)
        ser_rayleigh.append(np.sum(s_int != s_int_detected_rayleigh) / num_symbols)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ser_awgn, 'b--', label='SISO AWGN')
    plt.semilogy(snr_db_range, ser_rayleigh, 'r-', label='SISO Rayleigh')

    plt.title('Symbol Error Rate (SER) vs. SNR for QPSK')
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10 ** -5, 1)  # Set y-axis limits for log scale
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


# Helper functions
def _to_symbols(int_array: np.ndarray) -> np.ndarray:
    """
    Converts an array of integers to QPSK complex symbols.

    Args:
        int_array (List[int]): An array of integers representing the QPSK symbols (0, 1, 2, 3).

    Returns:
        List[complex]: An array of normalized QPSK complex symbols.
    """

    degrees = int_array * 360 / 4.0 + 45  # 45, 135, 225, 315 degrees
    radians = np.deg2rad(degrees)
    symbols = np.exp(1j * radians)  # normalized QPSK complex symbols

    return symbols


def _ml_detect(received_symbols, constellation):
    """
    Performs Maximum Likelihood (ML) detection by finding the closest constellation point.

    Args:
        received_symbols (np.ndarray): Array of received complex symbols.
        constellation (np.ndarray): Array of ideal constellation points.

    Returns:
        np.ndarray: Array of detected integer symbols (indices of the closest constellation points).
    """
    # Find the closest constellation symbol for each received symbol
    distances = np.abs(received_symbols[:, np.newaxis] - constellation)
    return np.argmin(distances, axis=1)


if __name__ == '__main__':
    main()
