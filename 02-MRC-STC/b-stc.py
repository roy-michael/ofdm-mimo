import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qpsk, _to_symbols, _ls_detect, _ml_detect

"""
Repeat the STC case (Tx, channel, Rx) and reproduce the
SER curve for 2 Tx and 1 Rx antennas
Compare with MRC 2
"""

num_symbols = 100000


def main():
    s0_int, s0_symbols = _get_qpsk(num_symbols * 2)
    s_mrc_int, s_mrc_symbols = _get_qpsk(num_symbols)

    s_stc_symbols = s0_symbols.reshape(num_symbols, 2)
    s_stc_int = s0_int.reshape(num_symbols, 2)

    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 40, 2)

    ser_stc_array = np.zeros((len(snr_db_range)))
    ser_mrc_array = np.zeros((len(snr_db_range)))

    for snr_db_idx, snr_db in enumerate(snr_db_range):
        # Convert SNR from dB to linear and calculate required noise power
        snr_linear = 10 ** (snr_db / 10.0)
        rho = 1. / np.sqrt(snr_linear)

        n_stc = np.array([_get_complex(num_symbols), np.conj(_get_complex(num_symbols))])  # AWGN
        n_mrc = _get_complex(num_symbols, 2)
        h = _get_complex(num_symbols, 2)  # Channels

        ha = np.array(
            [[h[:, 0], np.conj(h[:, 1])], [h[:, 1], -np.conj(h[:, 0])]]).transpose()  # A 50000 * 2 * 2  matrix

        # STC Received signals
        y_stc = (ha @ s_stc_symbols[:, :, np.newaxis]).squeeze(axis=-1)
        y_stc += (rho * n_stc).transpose()

        y_mrc = s_mrc_symbols.reshape(-1, 1) * h + rho * n_mrc  # Received signals

        # Equalization
        s_mrc_hat = np.sum(np.conj(h) * y_mrc, axis=1) / np.sum((np.abs(h) ** 2), axis=1)
        s_stc_hat = (np.linalg.inv(ha) @ y_stc[:, :, np.newaxis]).squeeze(axis=-1)

        # Detections
        constellation = _to_symbols(np.arange(0, 4))
        s_stc_detected = _ls_detect(s_stc_hat, constellation)
        s_mrc_detected = _ml_detect(s_mrc_hat, constellation)

        # Count the errors and calculate SER
        ser_stc_array[snr_db_idx] = np.sum(s_stc_int != s_stc_detected) / num_symbols
        ser_mrc_array[snr_db_idx] = np.sum(s_mrc_int != s_mrc_detected) / num_symbols

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    plt.semilogy(snr_db_range, ser_mrc_array, "r--", label=f'MRC 1X2')
    plt.semilogy(snr_db_range, ser_stc_array, "b-", label=f'STC 2X1')

    plt.title('Symbol Error Rate (SER) vs. SNR for QPSK with MRC')
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10 ** -5, 1)  # Set y-axis limits for log scale
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


if __name__ == '__main__':
    main()
