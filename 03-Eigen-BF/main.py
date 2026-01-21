import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qpsk, _to_symbols, _ls_detect, _ml_detect

"""
Repeat the Eigen BF case for 2Rx X 2Tx and 2Rx X 4Tx
Compare with the STC 2X2
"""


def main(s_int, s_symbols, num_symbols):
    s_stc_symbols = np.vstack((s_symbols[::2], s_symbols[1::2])).T
    s_stc_int = np.vstack((s_int[::2], s_int[1::2])).T
    num_stc = num_symbols // 2

    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 40, 2)
    ser_array = np.zeros((len(snr_db_range), 3))

    for snr_db_idx, snr_db in enumerate(snr_db_range):
        # Convert SNR from dB to linear and calculate required noise power
        snr_linear = 10 ** (snr_db / 10.0)
        rho = np.sqrt(0.5) / np.sqrt(snr_linear)  # Noise scaling factor

        ser_array[snr_db_idx] = [
            np.sum(s_int != bfm_step(s_symbols, num_symbols, rho, 2, 2)) / num_symbols,
            np.sum(s_int != bfm_step(s_symbols, num_symbols, rho, 2, 4)) / num_symbols,
            np.sum(s_stc_int != stc_step(s_stc_symbols, num_stc, rho, 2)) / num_symbols,
        ]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    styles = [("Eigen BF 2x2", 'r-'), ("Eigen BF 4x2", 'b--'), ("STC 2x2", 'g:')]
    for tp_idx, tp_ser in enumerate(ser_array.T):
        plt.semilogy(snr_db_range, tp_ser, styles[tp_idx][1], label=f'{styles[tp_idx][0]}')

    plt.title('Symbol Error Rate (SER) vs. SNR for QPSK with Eigen-Beamforming')
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10 ** -5, 1)  # Set y-axis limits for log scale
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


def bfm_step(s_symbols, num_symbols, rho, rx, tx, constellation=_to_symbols(np.arange(0, 4))):
    # Generate a batch of channels, one for each symbol
    # H has shape (num_symbols, rx, tx)
    H = _get_complex(size=(num_symbols, rx, 2 * tx))  # num_symbols * tx * rx Channels

    # Perform SVD on the batch of channels
    U, S, Vh = np.linalg.svd(H)

    # Select the first right singular vector for each channel as the precoding vector
    # w has shape (num_symbols, tx)
    w = np.conj(Vh[:, 0, :])

    # Precode the symbols.
    # x will have shape (num_symbols, tx)
    x = w * s_symbols[:, np.newaxis]

    # Transmit through the channel.
    # H is (N, rx, tx), x is (N, tx, 1). y is (N, rx, 1).
    y = H @ x[:, :, np.newaxis]

    # Add AWGN
    n = _get_complex(num_symbols, col=rx).reshape(num_symbols, rx, 1)
    y += rho * n

    # Hw has shape (num_symbols, rx, 1)
    Hw = H @ w[:, :, np.newaxis]

    # Equalization/Detection using MRC
    # Combine and equalize: s_hat = (Hw_H * y) / (Hw_H * Hw)
    s_hat = (np.conj(Hw).transpose(0, 2, 1) @ y) / (np.conj(Hw).transpose(0, 2, 1) @ Hw)

    # Detections
    return _ml_detect(s_hat.flatten(), constellation)


def stc_step(s_stc_symbols, num_stc, rho, tx, constellation=_to_symbols(np.arange(0, 4))):
    n_stc = np.array([
        _get_complex(num_stc), np.conj(_get_complex(num_stc)),
        _get_complex(num_stc), np.conj(_get_complex(num_stc))
    ])
    h = _get_complex(num_stc, col=tx)  # Channels
    g = _get_complex(num_stc, col=tx)  # Channels

    H = np.array(
        [
            [h[:, 0], np.conj(h[:, 1]), g[:, 0], np.conj(g[:, 1])],
            [h[:, 1], -np.conj(h[:, 0]), g[:, 1], -np.conj(g[:, 0])]
        ]).transpose()  # A 50000 * 4 * 2  matrix

    # STC Received signals (2x2 MISO)
    y_stc = ((H / np.sqrt(2)) @ s_stc_symbols[:, :, np.newaxis]).squeeze(axis=-1)
    y_stc += (rho * n_stc).T  # Add noise

    # Equalization/Detection
    s_stc_hat = (np.linalg.pinv(H) @ y_stc[:, :, np.newaxis]).squeeze(axis=-1)

    return _ls_detect(s_stc_hat, constellation)


if __name__ == '__main__':
    num_symbols = int(1e6)
    s_int, s_symbols = _get_qpsk(num_symbols)

    main(s_int, s_symbols, num_symbols)
