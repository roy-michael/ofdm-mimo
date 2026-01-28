import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qams, _ofdm_modulate, _calc_papr, _as_ofdm_symbols, _as_ofdm_centered_symbols, \
    _from_ofdm_centered_symbols

"""
Create an OFDMA system with **N_FFT=256** with **L_CP=16** and **4 Rx antennas** at AP.
* a. Assume 2 STAs transmitting 200syms UL (QPSK):
  1. STA0 transmitting a single stream over **SCs 0-49**.
  2. STA1 transmitting 2 streams (**SU-MIMO**) over **SCs 50-69**.
* b. Pilots:
  1. STA0 has a single preamble symbol.
  2. STA1 has 2 preamble symbols (one for each Tx ant.).
* c. Channels – for each STA consider a channel with:
  1. 10 equal power paths (uniform phase).
  2. Each path with random uniform delay up to CP duration (integer sample).
  3. Each path with uniform (0, 2*pi] DoA and DoD.
  4. Assume lambda / 2 ULAs.
* **d. Perform channel estimation and demodulate both STAs with linear detection.**
* **e. Show perfect reconstruction without noise.**
"""


def main(n_fft, l_cp, n_symbols, tx, rx):
    n_sta_scs = (
        (n_symbols, 50, 0, 2, 0),
        (n_symbols // 2, 20, 0, 2, 50),
        (n_symbols // 2, 20, 1, 2, 50))  # (number of symbols, preamble position, number of streams, sc start)
    qam_streams = []
    n_paths = 10
    ofdm_symbols = []
    s_mtx = []

    for i in range(len(n_sta_scs)):
        n_stream, n_stream_scs, preamble_pos, n_streams, sc_start = n_sta_scs[i]

        # 1. Generate QAMs
        _, stream = _get_qams(n_stream, qam_size=4)
        qam_streams.append(stream)

        # 2. SC-FDMA (DFT Precoding)
        # M is n_stream_scs
        ofdma_coefficients = _ofdma_modulate(stream, n_stream_scs)

        # 3. Add Preamble (as first symbol)
        # Note: stream is clipped to n_symbols to maintain assignment constraints
        preamble = _create_preamble(n_stream_scs, n_streams, preamble_pos)

        # 4. IFFT and CP (Generation of time-domain symbols)
        # Returns matrix of [Num_Symbols x (N_FFT + L_CP)]
        preamble_matrix = _as_ofdm_centered_symbols(n_stream_scs, n_fft, l_cp, 1, preamble, sc_start)
        symbol_matrix = _as_ofdm_centered_symbols(n_stream_scs, n_fft, l_cp, 1, ofdma_coefficients, sc_start)
        ofdm_symbols.append(np.vstack([preamble_matrix, symbol_matrix]))
        s_mtx.append(ofdm_symbols[i].flatten())

    # s_mtx = [symbol.flatten() for symbol in ofdm_symbols]

    H_sta0 = _get_channel_mtx(n_fft, l_cp, 1, rx, n_paths)
    H_sta1 = _get_channel_mtx(n_fft, l_cp, tx, rx, n_paths)
    b0 = H_sta0 @ s_mtx[0][:, np.newaxis].T
    b1 = H_sta1 @ np.vstack((s_mtx[1], s_mtx[2]))

    b0 = np.hstack((b0, np.zeros((rx, b1.shape[1] - b0.shape[1]))))

    y = b0 + b1

    # Reshape back to (rx, n_symbols, n_fft + l_cp)
    y_symbols = y.reshape(y.shape[0], -1, n_fft + l_cp)

    # 2. Extract the preamble symbols
    # Let's say indices 0 and 1 are the preambles for STA1
    preamble1_time = y_symbols[:, 0, :]
    preamble2_time = y_symbols[:, 1, :]

    # 3. Move to Frequency Domain (Remove CP first)
    preamble1_freq = np.fft.fftshift(np.fft.fft(preamble1_time[:, l_cp:], axis=1) / np.sqrt(n_fft), axes=1)
    preamble2_freq = np.fft.fftshift(np.fft.fft(preamble2_time[:, l_cp:], axis=1) / np.sqrt(n_fft), axes=1)

    # 4. Extract H(k) for subcarriers 50-69
    # Column 1 of H comes from Preamble 1, Column 2 from Preamble 2
    H_STA0_est = np.zeros((256, 4, 1), dtype=complex)
    H_STA0_est[0:50, :, 0] = preamble1_freq[:, 0:50].T

    # 4. Extract H(k) for subcarriers 50-69
    # Column 1 of H comes from Preamble 1, Column 2 from Preamble 2
    H_STA1_est = np.zeros((256, 4, 2), dtype=complex)
    H_STA1_est[50:70, :, 0] = preamble1_freq[:, 50:70].T
    H_STA1_est[50:70, :, 1] = preamble2_freq[:, 50:70].T

    H_STA0_est_inv = np.linalg.pinv(H_STA0_est)
    H_STA1_est_inv = np.linalg.pinv(H_STA1_est)

    y_no_cp = y_symbols[:, 2:, l_cp:]
    np_fft_fft = np.fft.fft(y_no_cp, axis=2)
    bk_freq = np.fft.fftshift(np.fft.fft(y_no_cp, axis=1) / np.sqrt(n_fft), axes=1)

    ak0_est = H_STA0_est_inv @ bk_freq.transpose(2, 0, 1)
    ak1_est = H_STA1_est_inv @ bk_freq.transpose(2, 0, 1)

    ak_ast = np.hstack((ak0_est, ak1_est))

    est_qam_streams = []

    for i in range(len(n_sta_scs)):
        n_stream, n_stream_scs, preamble_pos, n_streams, sc_start = n_sta_scs[i]

        ofdma_coefficients = _from_ofdm_centered_symbols(n_stream_scs, n_fft, ak_ast[:, i, :], sc_start)
        qam_stream = _ofdma_demodulate(ofdma_coefficients, n_stream, n_stream_scs)
        est_qam_streams.append(qam_stream)

    for i in range(len(n_sta_scs)):
        print(np.sum(qam_streams[i] == est_qam_streams[i]))

    # 6. Plotting the Magnitude
    # for i in range(len(ofdm_symbols)):
    #     # ofdm_symbols[i] is now (Time_Slots, 272)
    #     # We plot the first data symbol (index 1)
    #     # If testing with only preamble, use index 0
    #     symbol_to_plot = ofdm_symbols[i][1, :] if ofdm_symbols[i].shape[0] > 1 else ofdm_symbols[i][0, :]
    #
    #     magnitude = np.abs(symbol_to_plot)
    #     plt.plot(magnitude, label=labels[i])
    #
    # plt.axvline(x=l_cp, color='r', linestyle='--', label='End of CP')
    # plt.title('Corrected SC-FDMA Magnitude')
    # plt.legend()
    # plt.show()


def _plot(papr_ccdf, title, over_sampling):
    plt.figure(figsize=(8, 6))
    for (m, n_fft, qam_size), (papr_vals, ccdf_probs) in papr_ccdf.items():
        plt.semilogy(papr_vals, ccdf_probs, linewidth=2,
                     label=f'M={m}, N_FFT={n_fft}, {qam_size}QAM (OS={over_sampling})')

    plt.title(title)
    plt.xlabel("PAPR threshold (dB)")
    plt.ylabel("Probability (PAPR > threshold)")
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.ylim(1e-4, 1)  # Limit Y-axis to match standard plots
    plt.legend()
    plt.tight_layout()


def calculate_ccdf(n_fft, l_cp, qam_size, n_allocation_tones, n_oversampling, n_ofdm_symbols=10000):
    """
    Simulates OFDM symbols and returns the sorted PAPR values and their CCDF probabilities.
    """

    _, payload_data = _get_qams(num_symbols=(n_allocation_tones * n_ofdm_symbols), qam_size=qam_size)
    ofdm_symbols = _as_ofdm_symbols(qam_payload=payload_data,
                                    allocation_tones=n_allocation_tones,
                                    n_fft=n_fft,
                                    l_cp=l_cp,
                                    over_sample=n_oversampling)

    papr_db_values = _calc_papr(ofdm_symbols)

    # 5. Calculate CCDF using the Sorting Method
    # Sort values from low to high
    sorted_papr = np.sort(papr_db_values)

    # Generate the Y-axis (Probability of exceeding)
    # If we have N items, the probability of exceeding the k-th smallest item
    # is roughly (N - k) / N
    y_axis = 1.0 - np.arange(1, len(sorted_papr) + 1) / len(sorted_papr)

    return sorted_papr, y_axis


def _create_preamble(n_sc_symbols, n_streams, preamble_pos):
    preamble_data = np.zeros((n_streams, n_sc_symbols), dtype=complex)
    preamble_data[preamble_pos] = np.ones(n_sc_symbols, dtype=complex)
    return preamble_data


def _add_preamble(payload_data, n_sc_symbols, n_streams, preamble_pos):
    preamble_data = np.zeros((n_streams, n_sc_symbols), dtype=complex)
    preamble_data[preamble_pos] = np.ones(n_sc_symbols, dtype=complex)
    return np.hstack([preamble_data.flatten(), payload_data])


def _create_stream(n_symbols, n_sc):
    return _get_qams(n_symbols * n_sc, qam_size=4)


def _get_channel_mtx(n_fft, l_cp, tx, rx, n_paths, fc=2.5e9):
    """
    * c. Channels – for each STA consider a channel with:
      1. 10 equal power paths (uniform phase).
      2. Each path with random uniform delay up to CP duration (integer sample).
      3. Each path with uniform $(0, 2*pi] DoA and DoD.
      4. Assume lambda / 2 ULAs.
    """

    delays = np.random.uniform(0, l_cp, n_paths)
    phases = np.random.uniform(0, 2 * np.pi, n_paths)
    doas = np.random.uniform(0, 2 * np.pi, n_paths)
    dods = np.random.uniform(0, 2 * np.pi, n_paths)

    coefficients = _get_channel_coefficients(phases, delays, fc)

    x_doa_theta = _as_steering_mtx(rx, doas)
    x_dod_phi = _as_steering_mtx(tx, dods)

    # Multiply 3D matrix by 3D matrix
    H = np.sum(coefficients[:, np.newaxis, np.newaxis] *
               (x_doa_theta.T[:, :, np.newaxis] @ x_dod_phi.T[:, np.newaxis, :].conj()), axis=0)

    return H


def _as_steering_vector(size, phases):
    n = np.arange(size)
    return np.sum(np.exp(-1j * np.pi * n[:, np.newaxis] * np.sin(phases)), axis=1)


def _as_steering_mtx(size, phases):
    n = np.arange(size)
    return np.exp(-1j * np.pi * n[:, np.newaxis] * np.sin(phases))


def _ofdma_modulate(stream, n_sta_sc):
    n_ofdma_symbols = len(stream) // n_sta_sc
    ofdma_symbols = np.zeros((n_ofdma_symbols, n_sta_sc), dtype=complex)

    for i in range(n_ofdma_symbols):
        sub_stream = stream[i * n_sta_sc: (i + 1) * n_sta_sc]
        ofdma_symbols[i, :] = np.fft.fft(sub_stream, n_sta_sc)

    return ofdma_symbols


def _ofdma_demodulate(ofdma_symbols, n_symbols, n_sta_sc):
    n_ofdma_symbols = ofdma_symbols.shape[1]
    streams = np.zeros((n_ofdma_symbols, n_sta_sc), dtype=complex)

    for i in range(n_ofdma_symbols):
        # sub_stream = ofdma_symbols[i * n_sta_sc: (i + 1) * n_sta_sc]
        sub_stream = np.fft.ifft(ofdma_symbols[:, i], n_sta_sc)
        streams[i, :] = sub_stream

    return streams.flatten()[:n_symbols]


def _get_channel_coefficients(phases, delays, fc, avg_power=1, v=100):
    c = 3e8
    fm = fc * v / c

    delay_spreads = avg_power * np.exp(-2j * np.pi * fc * delays)
    freq_shifts = np.exp(-2j * np.pi * fm * np.cos(phases))

    return delay_spreads * freq_shifts


if __name__ == '__main__':
    n_fft = 256
    l_cp = 16
    rx = 4
    tx = 2
    n_symbols = 200

    allocation_tones = 100
    over_sampling = 4
    qam_sizes = [4, 16, 64, 256]
    num_ofdm_symbols = int(10e3)

    main(n_fft, l_cp, n_symbols, tx, rx)
