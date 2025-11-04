from typing import List

import numpy as np
import matplotlib.pyplot as plt

num_symbols = 1000
amplitude = 1

def main():

    # QPSK symbols
    s_int = np.random.randint(0, 4, num_symbols)
    s_degrees = s_int * 360/4.0 + 45 # 45, 135, 225, 315 degrees
    s_radians = np.deg2rad(s_degrees)
    s_symbols = amplitude * np.exp(1j * s_radians) / np.sqrt(2) # normalized QPSK complex symbols

    # Calculate signal power (Es)
    signal_power = np.mean(np.abs(s_symbols) ** 2)

    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 40, 1)

    ser_awgn = []
    ser_rayleigh = []

    # --- Run Simulation for each SNR value ---
    for snr_db in snr_db_range:

        # Convert SNR from dB to linear and calculate required noise power
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # Generate AWGN with the calculated power
        noise_std = np.sqrt(noise_power / 2)  # Per dimension (I and Q)
        # n = np.random.normal(loc=0, scale=noise_std, size=(num_symbols, 2)).view(np.complex128)
        n = np.squeeze(np.random.normal(size=(num_symbols, 2), loc=0, scale=noise_std).view(np.complex128) / np.sqrt(2)) # AWGN

        # Channels
        h_awgn = 1
        h_rayleigh = np.squeeze(np.random.normal(size=(num_symbols, 2), loc=0, scale=1).view(np.complex128) / np.sqrt(2))

        # Received signals
        y_awgn = s_symbols * h_awgn +  n
        y_rayleigh = s_symbols * h_rayleigh + n

        # Equalization
        s_hat_awgn = y_awgn / h_awgn
        s_hat_rayleigh = y_rayleigh / h_rayleigh

        # Detections
        options = [45, 135, 225, 315]
        s_int_detected_awgn = _ml_detect(s_hat_awgn, options)
        s_int_detected_rayleigh = _ml_detect(s_hat_rayleigh, options)

        # Count the errors and calculate SER
        ser_awgn.append(np.sum(s_int != s_int_detected_awgn) / num_symbols)
        ser_rayleigh.append(np.sum(s_int != s_int_detected_rayleigh) / num_symbols)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    snr_linear_range = 10 ** (snr_db_range / 10.0)
    plt.semilogy(snr_db_range, ser_awgn, 'r--', label='SISO AWGN')
    plt.semilogy(snr_db_range, ser_rayleigh, 'b--', label='SISO Rayleigh')

    plt.title('Symbol Error Rate (SER) vs. SNR for QPSK')
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10 ** -5, 1)  # Set y-axis limits for log scale
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


def _to_symbols(int_array: List[int]) -> List[complex]:

    degrees = int_array * 360 / 4.0 + 45  # 45, 135, 225, 315 degrees
    radians = np.deg2rad(degrees)
    symbols = amplitude * np.exp(1j * radians) / np.sqrt(2)  # normalized QPSK complex symbols

    return symbols


def _from_symbols(array: List[complex]) -> List[int]:

    degrees = np.angle(array, deg=True)
    int_array = np.round((degrees - 45) * 4 / 360)

    return int_array

def _ml_detect(symbols, options):
    degrees = np.angle(symbols, deg=True) % 360

    new_symbols = degrees[:, np.newaxis]
    distances = np.square(np.abs(new_symbols - options))
    return np.argmin(distances, axis=1)

if __name__ == '__main__':
    main()