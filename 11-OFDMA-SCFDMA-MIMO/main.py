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
        (n_symbols, 20, 0, 2, 50),
        (n_symbols, 20, 1, 2, 50))  # (number of symbols, preamble position, number of streams, sc start)
    qam_streams = []
    n_paths = 10
    ofdm_symbols = []
    s_mtx = []

    # 1. TRANSMITTER SIDE
    for i in range(len(n_sta_scs)):
        n_stream, n_stream_scs, preamble_pos, n_streams, sc_start = n_sta_scs[i]

        # 1. Generate QAMs
        _, stream = _get_qams(n_stream, qam_size=4)
        qam_streams.append(stream)

        # 2. SC-FDMA (DFT Precoding). M is n_stream_scs
        ofdma_coefficients = _ofdma_modulate(stream, n_stream_scs)

        # 3. Add Preamble (as symbols)
        preamble = _create_preamble(n_stream_scs, n_streams, preamble_pos)

        # 4. IFFT and CP (Generation of time-domain symbols)
        # Returns matrix of [Num_Symbols x (N_FFT + L_CP)]
        preamble_matrix = _as_ofdm_centered_symbols(n_stream_scs, n_fft, l_cp, 1, preamble, sc_start)
        symbol_matrix = _as_ofdm_centered_symbols(n_stream_scs, n_fft, l_cp, 1, ofdma_coefficients, sc_start)
        ofdm_symbols.append(np.vstack([preamble_matrix, symbol_matrix]))
        s_mtx.append(ofdm_symbols[i].flatten())

    # Find the maximum signal length to pad others
    max_len = max(len(s) for s in s_mtx)
    padded_tx_signals = [np.pad(s, (0, max_len - len(s))) for s in s_mtx]
    X_tx = np.vstack(padded_tx_signals)

    # 2. RECEIVER SIDE
    h_total = _get_channel_mtx(rx, tx, n_paths, l_cp)
    # Apply physical multi-path convolution
    total_samples = X_tx.shape[1]
    y_received = np.zeros((rx, total_samples), dtype=complex)
    for r in range(rx):
        for t in range(tx):
            # Applying physical multi-path convolution
            conv_res = np.convolve(X_tx[t, :], h_total[r, t, :], mode='full')
            y_received[r, :] += conv_res[:total_samples]


    # Reshape back to (rx, n_symbols, n_fft + l_cp)
    y_symbols = y_received.reshape(rx, -1, n_fft + l_cp)

    # 2. Extract the preamble symbols
    # Let's say indices 0 and 1 are the preambles for STA1
    preamble1_time = y_symbols[:, 0, :]
    preamble2_time = y_symbols[:, 1, :]

    # 3. Move to Frequency Domain (Remove CP first)
    preamble1_freq = np.fft.fftshift(np.fft.fft(preamble1_time[:, l_cp:], axis=1) / np.sqrt(n_fft), axes=1)
    preamble2_freq = np.fft.fftshift(np.fft.fft(preamble2_time[:, l_cp:], axis=1) / np.sqrt(n_fft), axes=1)

    # 4. Channel Estimation
    # Column 1 of H comes from Preamble 1, Column 2 from Preamble 2
    mid_fft = n_fft // 2
    H_est = np.zeros((n_fft, rx, tx), dtype=complex)
    H_est[mid_fft:mid_fft+50, :, 0] = preamble1_freq[:, 0:50].T
    H_est[mid_fft+50:mid_fft+70, :, 0] = preamble1_freq[:, 50:70].T
    H_est[mid_fft+50:mid_fft+70, :, 1] = preamble2_freq[:, 50:70].T

    y_no_cp = y_symbols[:, 2:, l_cp:]
    bk_freq = np.fft.fftshift(np.fft.fft(y_no_cp, axis=1) / np.sqrt(n_fft), axes=1)

    # 5. Linear Detection
    est_qams = []

    # User 0 (STA0) - Original data was 4 slots
    s0_ak = np.zeros((50, 4), dtype=complex)
    for k in range(mid_fft, mid_fft + 50):
        H_k_sc = H_est[k, :, [0]]
        # Extract received matrix for this carrier across the 4 data slots
        Y_k_slots = bk_freq.T[k, 0:4, :]  # (4_slots, 4_rx)
        # Multiply: (1x4) @ (4x4) -> (1x4)
        s0_ak[k - mid_fft, :] = (np.linalg.pinv(H_k_sc).T @ Y_k_slots).flatten()

    est_qams.append(_ofdma_demodulate(s0_ak, n_symbols, 50))

    # User 1 (STA1)
    # H_k_sc: (4, 2), pinv(H): (2, 4)
    # Y_data_grid[k, 0:10, :]: (10, 4) - 10 slots, 4 rx antennas
    # We want: (2, 4) @ (4, 10) -> (2, 10) result
    s1_ak = np.zeros((20, 10, 2), dtype=complex)
    for k in range(mid_fft + 50, mid_fft + 70):
        H_k_sc = H_est[k, :, 1:3]
        Y_k_slots = bk_freq.T[k, 0:10, :]  # (10_slots, 4_rx)
        # Result is (2, 10)
        res = np.linalg.pinv(H_k_sc) @ Y_k_slots.T
        s1_ak[k - (mid_fft + 50), :, :] = res.T  # Store as (SCs, Slots, Streams)

    est_qams.append(_ofdma_demodulate(s1_ak[:, :, 0], n_symbols, 20))
    est_qams.append(_ofdma_demodulate(s1_ak[:, :, 1], n_symbols, 20))

    # Verification
    for i in range(3):
        err = np.sum(np.abs(qam_streams[i] - est_qams[i]) > 1e-9)
        print(f"Stream {i} Reconstruction: {'PERFECT' if err == 0 else 'ERRORS: ' + str(np.sum(np.abs(qam_streams[i] - est_qams[i])))}")

    est_qam_streams = []

    # for i in range(len(n_sta_scs)):
    #     n_stream, n_stream_scs, preamble_pos, n_streams, sc_start = n_sta_scs[i]
    #
    #     ofdma_coefficients = _from_ofdm_centered_symbols(n_stream_scs, n_fft, ak_ast[:, i, :], sc_start)
    #     qam_stream = _ofdma_demodulate(ofdma_coefficients, n_stream, n_stream_scs)
    #     est_qam_streams.append(qam_stream)
    #
    # for i in range(len(n_sta_scs)):
    #     print(np.sum(qam_streams[i] == est_qam_streams[i]))

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


def _create_preamble(n_sc_symbols, n_streams, preamble_pos):
    preamble_data = np.zeros((n_streams, n_sc_symbols), dtype=complex)
    preamble_data[preamble_pos] = np.ones(n_sc_symbols, dtype=complex)
    return preamble_data


def _add_preamble(payload_data, n_sc_symbols, n_streams, preamble_pos):
    preamble_data = np.zeros((n_streams, n_sc_symbols), dtype=complex)
    preamble_data[preamble_pos, :] = 1.0
    return np.hstack([preamble_data.flatten(), payload_data])


def _create_stream(n_symbols, n_sc):
    return _get_qams(n_symbols * n_sc, qam_size=4)


def _get_channel_mtx(rx, tx, n_paths, l_cp):

    H_k = np.zeros((rx, tx, l_cp), dtype=complex)

    for t in range(tx):
        delays = np.random.uniform(0, l_cp, n_paths)
        phases = np.random.uniform(0, 2 * np.pi, n_paths)
        doas = np.random.uniform(0, 2 * np.pi, n_paths)
        dods = np.random.uniform(0, 2 * np.pi, n_paths)

        x_doa = _as_steering_mtx(rx, doas)  # (rx, 10)
        x_dod = _as_steering_mtx(tx, dods)  # (tx, 10)

        for p in range(n_paths):
            # Apply gain, phase, DoA, and DoD (x_dod is 1x1 here since we model per-stream tx)
            # In a full MIMO model, this would be x_doa @ x_dod.H
            delay_idx = int(delays[p])
            path_gain = np.exp(1j * phases[p]) * x_dod[0, p].conj()
            H_k[:, t, delay_idx] += path_gain * x_doa[:, p]

    return H_k


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
    tx = 3  # total streams
    n_symbols = 200

    allocation_tones = 100
    over_sampling = 4
    qam_sizes = [4, 16, 64, 256]
    num_ofdm_symbols = int(10e3)

    main(n_fft, l_cp, n_symbols, tx, rx)
