import numpy as np
import matplotlib.pyplot as plt

from common import (
    _get_complex,
    _get_qpsk,
    _to_symbols,
)

"""
Consider a receiver with 4 antennas and the model
y = h*s + sqrt(pg)*g*r + rho + n,
where h, g and r are CN(0,1) iid and pg = 0.3162 (5dB SIR)
    • Decode with MRC
    • Decode with MVDR
    • Compare the MVDR with MRC3 without interference
"""


def main(s_symbols, r_symbols, num_symbols, rx, tx):
    # Define SNR range in dB for the simulation and plots
    snr_db_range = np.arange(0, 40, 2)
    ser_array = np.zeros((len(snr_db_range), 3))

    for snr_db_idx, snr_db in enumerate(snr_db_range):
        # Convert SNR from dB to linear and calculate required noise power
        snr_linear = 10 ** (snr_db / 10.0)

        # rho = np.sqrt(0.5) / np.sqrt(snr_linear)  # Noise scaling factor
        # rho = 1 / snr_linear  # Noise scaling factor
        rho = np.sqrt((1 / snr_linear) / 2)

        # Noise power is 1/snr_linear assuming symbol power is 1
        noise_power = 1 / snr_linear

        # Count a vector error if any symbol in the detected vector is wrong.
        detections = [
            step(
                s_symbols,
                r_symbols,
                num_symbols,
                rho,
                rx,
                tx=tx,
                decoder="mrc",
                detector=_ml_detect,
            ),
            step(
                s_symbols,
                r_symbols,
                num_symbols,
                rho,
                rx,
                tx=tx,
                decoder="mvdr",
                detector=_ls_detect,
            ),
            step(
                s_symbols,
                r_symbols,
                num_symbols,
                rho,
                3,
                tx=tx,
                decoder="mrc",
                with_interference=False,
                detector=_ml_detect,
            ),
        ]
        errors = [(s_symbols.squeeze() != det.squeeze()) for det in detections]
        ser_array[snr_db_idx] = [np.sum(err) / num_symbols for err in errors]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    styles = [("MRC", "b-"), ("MVDR", "r-"), ("MRC3 (no interference)", "g--")]
    for tp_idx, tp_ser in enumerate(ser_array.T):
        plt.semilogy(
            snr_db_range, tp_ser, styles[tp_idx][1], label=f"{styles[tp_idx][0]}"
        )

    plt.title("MRC and MVDR Rayleigh 4 Antennas at SIR=5dB")
    plt.xlabel("SNR (Es/N0) [dB]")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.ylim(10**-5, 1)
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


def step(
    s_symbols,
    r_symbols,
    num_symbols,
    rho,
    rx,
    tx,
    detector,
    decoder="mvdr",
    with_interference=True,
):
    constellation = _to_symbols(np.arange(0, 4))

    pg = 0.3162  # -5dB SIR

    H = _get_complex(num_symbols, col=tx, rows=rx)  # num_symbols * tx * rx Channels
    G = _get_complex(num_symbols, col=tx, rows=rx)  # num_symbols * tx * rx Channels

    yh = H @ s_symbols[:, np.newaxis]
    yg = np.sqrt(pg) * G @ r_symbols[:, np.newaxis]

    # Add AWGN
    n = _get_complex(num_symbols, col=tx, rows=rx)

    y = yh

    if with_interference:
        y += yg

    # Final y
    y += rho * n

    # Decoding
    s_hat = decode_mvdr(y, H, G, pg, rho, rx) if decoder == "mvdr" else decode_mrc(H, y)

    # Detections
    return detector(s_hat, constellation)


def decode_mrc(h, y):
    # Apply Maximal Ratio Combining (MRC)

    axis = 1
    return np.sum(np.conj(h) * y, axis=axis) / np.sum((np.abs(h) ** 2), axis=axis)


def decode_mvdr(y, h, g, pg, noise_power, rx):
    g_h = np.conj(g).transpose(0, 2, 1)
    h_h = np.conj(h).transpose(0, 2, 1)

    C = pg * g @ g_h + noise_power**2 * np.eye(rx)
    C_inv = np.linalg.inv(C)

    return (h_h @ C_inv @ y) / (h_h @ C_inv @ h)


# Find the closest constellation symbol for each received symbol
def _ls_detect(received_symbols, constellation):
    distances = np.abs(received_symbols - constellation) ** 2

    # Find the index of the minimum distance for each symbol in the vector
    # Return the detected symbols from the constellation
    min_indices = np.argmin(distances, axis=2)
    return constellation[min_indices]


# Find the closest constellation symbol for each received symbol
def _ml_detect(received_symbols, constellation):
    distances = np.abs(received_symbols - constellation)
    min_distance = np.argmin(distances, axis=1)

    return constellation[min_distance]


if __name__ == "__main__":
    num_symbols = int(1e6)
    tx, rx = (1, 4)
    num_streams = min(tx, rx)

    _, s_symbols = _get_qpsk(num_symbols)
    _, r_symbols = _get_qpsk(num_symbols)

    s_symbols = s_symbols.reshape(-1, 1)
    r_symbols = r_symbols.reshape(-1, 1)

    main(s_symbols, r_symbols, num_symbols // num_streams, rx, tx)

    plt.show()
