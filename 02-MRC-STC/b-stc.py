import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qpsk, _to_symbols, _ls_detect, _ml_detect

"""
Repeat the STC case (Tx, channel, Rx) and reproduce the
SER curve for 2 Tx and 1 Rx antennas
Compare with MRC 2
"""

num_symbols = int(1e6)


def main():

    s_int, s_symbols = _get_qpsk(num_symbols)

    # into a 2 x (num_symbols / 2) matrices
    s_stc_symbols = np.vstack((s_symbols[::2], s_symbols[1::2])).T
    s_stc_int = np.vstack((s_int[::2], s_int[1::2])).T
    num_stc = num_symbols // 2

    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 30, 2)

    ser_stc_array = np.zeros((len(snr_db_range)))
    ser_mrc_array = np.zeros((len(snr_db_range)))

    for snr_db_idx, snr_db in enumerate(snr_db_range):

        # Convert SNR from dB to linear and calculate required noise power
        snr_linear = 10 ** (snr_db / 10.0)
        rho = np.sqrt(0.5) / np.sqrt(snr_linear)  # Noise scaling factor

        # AWGN
        n_stc = np.array([_get_complex(num_stc), np.conj(_get_complex(num_stc))])
        n_mrc = _get_complex(num_symbols, col=2)

        # Channels
        h_stc = _get_complex(num_stc, col=2) # Channel remains constant over two symbol periods
        h_mrc = _get_complex(num_symbols, col=2)

        H = np.array(
            [
                [h_stc[:, 0], np.conj(h_stc[:, 1])],
                [h_stc[:, 1], -np.conj(h_stc[:, 0])]
            ]).transpose()  # n_stc * 2 * 2  matrix

        # STC Received signals (2x1 MISO)
        y_stc = ((H / np.sqrt(2)) @ s_stc_symbols[:, :, np.newaxis]).squeeze(axis=-1)
        y_stc += (rho * n_stc).T  # Add noise

        y_mrc = s_symbols.reshape(-1, 1) * h_mrc + rho * n_mrc  # MRC Received signals (1x2 SIMO)

        # Equalization/Detection
        s_mrc_hat = np.sum(np.conj(h_mrc) * y_mrc, axis=1) / np.sum((np.abs(h_mrc) ** 2), axis=1)
        
        # STC Detection:
        # y_stc = H/sqrt(2) * s + n
        # inv(H) * y_stc = s/sqrt(2) + inv(H)*n
        # We need to scale by sqrt(2) to recover the scale of s
        s_stc_hat = (np.linalg.inv(H) @ y_stc[:, :, np.newaxis]).squeeze(axis=-1) * np.sqrt(2)

        # Detections
        constellation = _to_symbols(np.arange(0, 4))
        s_stc_detected = _ls_detect(s_stc_hat, constellation)
        s_mrc_detected = _ml_detect(s_mrc_hat, constellation)

        # Count the errors and calculate SER
        ser_stc_array[snr_db_idx] = np.sum(s_stc_int != s_stc_detected) / num_symbols
        ser_mrc_array[snr_db_idx] = np.sum(s_int != s_mrc_detected) / num_symbols

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    plt.semilogy(snr_db_range, ser_mrc_array, "r--", label=f'MRC 1X2')
    plt.semilogy(snr_db_range, ser_stc_array, "b-", label=f'STC 2X1')

    plt.title('Symbol Error Rate (SER) vs. SNR for QPSK with MRC/STC')
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10 ** -5, 1)  # Set y-axis limits for log scale
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


if __name__ == '__main__':
    main()
