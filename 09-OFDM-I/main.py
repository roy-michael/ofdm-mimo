import numpy as np
import matplotlib.pyplot as plt

from common import _get_qpsk, _get_complex

"""
Create an OFDM system with $N_{FFT}$=256 with $L_{CP}$=16
  1. Assume the $1^{st}$ symbol is known to Rx (Preamble) for channel estimation followed by 20 payload symbols
  2. Consider the channels $h_1=\delta(n)$, $h_2=\delta(n)+0.9j\delta(n-9)$
  3. Add 20dB SNR (to the time domain signal)
  4. Do channel estimation and equalize all payload symbols
  5. Compute average EVM $mean|\hat s-s|^2$
  6. Repeat d-e for perfect channel knowledge $H(k)=\sum_{n}h(n)e^{-j\frac{2\pi}{N}kn}$
"""


def main():
    n_fft = 256
    l_cp = 16
    n_payload_symbols = 20
    n_total_symbols = 1 + n_payload_symbols  # 1 Preamble + 20 Payload

    # Define active subcarriers (-100 to 100, excluding DC)
    active_indices = np.concatenate([np.arange(-100, 0), np.arange(1, 101)])
    n_active = len(active_indices)

    preamble_data = np.ones((1, n_active), dtype=complex)
    _, payload_data = _get_qpsk(num_symbols=(n_active * n_payload_symbols))
    payload_data = np.reshape(payload_data, (n_payload_symbols, n_active))
    all_data_freq = np.vstack([preamble_data, payload_data])

    d_k_mtx = np.zeros((n_total_symbols, n_fft), dtype=complex)
    # Map active subcarriers into the grid
    d_k_mtx[:, active_indices] = all_data_freq

    s_n_mtx = _ofdm_modulate(d_k_mtx, n_fft, l_cp)
    tx_signal = s_n_mtx.flatten()

    h1, h2 = _get_channels(n_fft)
    # apply the channel
    # Convolve and truncate to original length
    rx_h1_clean = np.convolve(tx_signal, h1, mode='full')[:len(tx_signal)]
    rx_h2_clean = np.convolve(tx_signal, h2, mode='full')[:len(tx_signal)]

    # Calculate signal power for SNR
    signal_power = np.mean(np.abs(s_n_mtx) ** 2)

    # Add AWGN
    # Noise should match the signal shape (n_fft + l_cp)
    rx_h1_noise = _get_noise(rx_h1_clean, 20).flatten()
    rx_h2_noise = _get_noise(rx_h2_clean, 20).flatten()

    # Received signal
    rx_h1 = rx_h1_clean + rx_h1_noise
    rx_h2 = rx_h2_clean + rx_h2_noise

    # Reshape back to (n_fft (w/o l_cp), n_symbols)
    rx_h1_blk = rx_h1.reshape(n_total_symbols, n_fft + l_cp)[:, l_cp:]
    rx_h2_blk = rx_h2.reshape(n_total_symbols, n_fft + l_cp)[:, l_cp:]

    # Batch FFT and shift to bring 0Hz back to center
    rx_h1_freq = np.fft.fftshift(np.fft.fft(rx_h1_blk, axis=1) / np.sqrt(n_fft), axes=1)
    rx_h2_freq = np.fft.fftshift(np.fft.fft(rx_h2_blk, axis=1) / np.sqrt(n_fft), axes=1)

    rx_h1_active = rx_h1_freq[:, active_indices]
    rx_h2_active = rx_h2_freq[:, active_indices]

    # LS Estimation
    H1_est = rx_h1_active[0, :] / all_data_freq[0, :]
    H2_est = rx_h2_active[0, :] / all_data_freq[0, :]

    # 2. Perfect Channel Knowledge (Section E)
    H1_perf_full = np.fft.fftshift(np.fft.fft(h1, n_fft))
    H2_perf_full = np.fft.fftshift(np.fft.fft(h2, n_fft))
    H1_perf = H1_perf_full[active_indices]
    H2_perf = H2_perf_full[active_indices]

    # Equalize
    s_h1_est = rx_h1_active[1:, :] / H1_est
    s_h2_est = rx_h2_active[1:, :] / H2_est

    s_h1_perf = rx_h1_active[1:, :] / H1_perf
    s_h2_perf = rx_h2_active[1:, :] / H2_perf

    s_h1_evm_est = _calculate_evm(s_h1_est, payload_data)
    s_h2_evm_est = _calculate_evm(s_h2_est, payload_data)
    s_h1_evm_perf = _calculate_evm(s_h1_perf, payload_data)
    s_h2_evm_perf = _calculate_evm(s_h2_perf, payload_data)

    print("-" * 35)
    print(f"EVM RESULTS")
    print(f"Average EVM (Perfect H1):   {s_h1_evm_perf * 100:.2f}%")
    print(f"Average EVM (Perfect H2):   {s_h2_evm_perf * 100:.2f}%")
    print(f"Average EVM (Estimated H1): {s_h1_evm_est * 100:.2f}%")
    print(f"Average EVM (Estimated H2): {s_h2_evm_est * 100:.2f}%")
    print("-" * 35)

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(np.real(s_h2_est), np.imag(s_h2_est), s=2, alpha=0.6, color='blue')
    plt.title("QPSK Constellation (Estimated H2)")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.abs(H1_perf_full), label='True |H1(k)|')
    plt.plot(np.abs(H2_perf_full), label='True |H2(k)|')
    plt.title("Frequency Response Magnitude")
    plt.xlabel("Subcarrier Index")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- E/F. EVM Calculation ---
def _calculate_evm(recovered, ideal):
    errors = recovered - ideal
    return np.sqrt(np.mean(np.abs(errors) ** 2) / np.mean(np.abs(ideal) ** 2))


def _get_channels(n_fft):
    h1 = np.zeros(n_fft, dtype=np.complex128)
    h1[0] = 1

    h2 = np.zeros(n_fft, dtype=np.complex128)
    h2[0] = 1
    h2[9] = 0.9j

    return h1, h2


def _get_noise(rx_signal, snr_db):
    sig_pwr = np.mean(np.abs(rx_signal) ** 2)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))

    n = _get_complex(len(rx_signal))

    return np.sqrt(noise_pwr / 2) * n


# 2. OFDM Modulation Function
def _ofdm_modulate(d_k_mtx, n_fft, l_cp):
    """
    Transforms d_k (centered indices) to a_k (IFFT indices)
    and performs IFFT + CP.
    """

    # Rearrangement to a_k format, with positive/negative indices
    a_k_mtx = np.fft.ifftshift(d_k_mtx, axes=1)
    s_n_mtx = np.fft.ifft(a_k_mtx, n_fft, axis=1) * np.sqrt(n_fft)

    # Add Cyclic Prefix. Take last L_cp samples and prepend to the start
    cp = s_n_mtx[:, -l_cp:]
    ofdm_symbol = np.hstack([cp, s_n_mtx])

    return ofdm_symbol


if __name__ == '__main__':
    main()
