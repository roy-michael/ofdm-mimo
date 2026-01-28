import numpy as np

# --- MAIN SIMULATION ---

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
    # Configuration: (n_syms, sc_count, preamble_slot, tx_stream_index, sc_start)
    # tx_stream_index maps to the columns of the global MIMO channel
    n_sta_scs = (
        (n_symbols, 50, 0, 0, 0),  # STA0 (Stream 0)
        (n_symbols, 20, 0, 1, 50),  # STA1 (Stream 1)
        (n_symbols, 20, 1, 2, 50))  # STA1 (Stream 2)

    qam_streams = []
    tx_signals = []
    mid_fft = n_fft // 2

    # 1. TRANSMITTER SIDE
    for i in range(len(n_sta_scs)):
        n_s, n_sc, pre_slot, tx_idx, sc_start = n_sta_scs[i]
        _, stream = _get_qams(n_s, qam_size=4)
        qam_streams.append(stream)

        # A. SC-FDMA (DFT Precoding)
        ofdma_coefficients = _ofdma_modulate(stream, n_sc)

        # B. Preamble Generation (Using centered symbols logic)
        pre_payload = np.zeros((2, n_sc), dtype=complex)
        pre_payload[pre_slot, :] = 1.0  # Known Pilots
        pre_matrix = _as_ofdm_centered_symbols(n_sc, n_fft, l_cp, 1, pre_payload, sc_start)

        # C. Data Generation
        data_matrix = _as_ofdm_centered_symbols(n_sc, n_fft, l_cp, 1, ofdma_coefficients, sc_start)

        # Combine and Flatten
        tx_signals.append(np.concatenate([pre_matrix.flatten(), data_matrix.flatten()]))

    # Padding and stacking for simultaneous Multi-User transmission
    max_len = max(len(s) for s in tx_signals)
    X_tx = np.vstack([np.pad(s, (0, max_len - len(s))) for s in tx_signals])

    # 2. CHANNEL APPLICATION
    h_total = _get_channel_mtx(rx, tx, 10, l_cp)
    total_samples = X_tx.shape[1]
    y_received = np.zeros((rx, total_samples), dtype=complex)
    for r in range(rx):
        for t in range(tx):
            conv_res = np.convolve(X_tx[t, :], h_total[r, t, :], mode='full')
            y_received[r, :] += conv_res[:total_samples]

    # 3. RECEIVER SIDE
    slot_len = n_fft + l_cp
    y_slots = y_received.reshape(rx, -1, slot_len)

    def _fft_and_shift(time_data, n_fft):
        # time_data is (rx, n_slots, n_fft)
        freq = np.fft.fft(time_data, axis=2) / np.sqrt(n_fft)
        return np.fft.fftshift(freq, axes=2)

    # A. Extraction
    Y_pre = _fft_and_shift(y_slots[:, 0:2, l_cp:], n_fft).transpose(2, 1, 0)  # (SCs, Slots, RX)
    Y_data = _fft_and_shift(y_slots[:, 2:, l_cp:], n_fft).transpose(2, 1, 0)  # (SCs, Slots, RX)

    # B. Channel Estimation
    H_est = np.zeros((n_fft, rx, tx), dtype=complex)
    # STA0 (Stream 0) and STA1-S1 (Stream 1) from Slot 0
    H_est[mid_fft:mid_fft + 50, :, 0] = Y_pre[mid_fft:mid_fft + 50, 0, :]
    H_est[mid_fft + 50:mid_fft + 70, :, 1] = Y_pre[mid_fft + 50:mid_fft + 70, 0, :]
    # STA1-S2 (Stream 2) from Slot 1
    H_est[mid_fft + 50:mid_fft + 70, :, 2] = Y_pre[mid_fft + 50:mid_fft + 70, 1, :]

    # C. Linear Detection
    est_qams = []

    # User 0 (STA0) - Detect 4 slots (200 syms / 50 SCs)
    s0_ak = np.zeros((50, 4), dtype=complex)
    for k in range(mid_fft, mid_fft + 50):
        H_k = H_est[k, :, 0].reshape(rx, 1)
        s0_ak[k - mid_fft, :] = np.linalg.pinv(H_k) @ Y_data[k, 0:4, :].T
    est_qams.append(_ofdma_demodulate(s0_ak, n_symbols, 50))

    # User 1 (STA1) - Detect 10 slots for both streams (200 syms / 20 SCs)
    s1_ak = np.zeros((20, 10, 2), dtype=complex)
    for k in range(mid_fft + 50, mid_fft + 70):
        H_k = H_est[k, :, 1:3]
        res = np.linalg.pinv(H_k) @ Y_data[k, 0:10, :].T
        s1_ak[k - (mid_fft + 50), :, :] = res.T

    est_qams.append(_ofdma_demodulate(s1_ak[:, :, 0], n_symbols, 20))
    est_qams.append(_ofdma_demodulate(s1_ak[:, :, 1], n_symbols, 20))

    # D. Verification
    for i in range(tx):
        err = np.sum(np.abs(qam_streams[i] - est_qams[i]) > 1e-8)
        print(f"Stream {i} Reconstruction: {'PERFECT' if err == 0 else 'ERRORS: ' + str(err)}")


# --- HELPER FUNCTIONS ---

def _get_channel_mtx(rx_count, tx_count, n_paths, cp_len):
    h_n = np.zeros((rx_count, tx_count, cp_len), dtype=complex)
    for t in range(tx_count):
        delays = np.random.randint(0, cp_len, n_paths)
        phases = np.random.uniform(0, 2 * np.pi, n_paths)
        doas = np.random.uniform(0, 2 * np.pi, n_paths)
        dods = np.random.uniform(0, 2 * np.pi, n_paths)
        x_doa = np.exp(-1j * np.pi * np.arange(rx_count)[:, np.newaxis] * np.sin(doas))
        x_dod = np.exp(-1j * np.pi * np.arange(1)[:, np.newaxis] * np.sin(dods))
        for p in range(n_paths):
            h_n[:, t, int(delays[p])] += np.exp(1j * phases[p]) * x_doa[:, p] * x_dod[0, p].conj()
    return h_n


def get_gray_to_pam_map(bits_per_axis):
    num_levels = 2 ** bits_per_axis
    gray_codes = [i ^ (i >> 1) for i in range(num_levels)]
    levels = np.arange(-num_levels + 1, num_levels, 2)
    gray_to_level = {gray_val: levels[i] for i, gray_val in enumerate(gray_codes)}
    return gray_to_level


def get_qam_constellation(symbol_int, qam_size):
    if np.log2(qam_size) % 2 != 0:
        raise ValueError("Only Square QAM supported.")
    k = int(np.log2(qam_size))
    k_axis = k // 2
    pam_map = get_gray_to_pam_map(k_axis)
    i_bits = symbol_int >> k_axis
    q_bits = symbol_int & ((1 << k_axis) - 1)
    return pam_map[i_bits] + 1j * pam_map[q_bits]


def _get_qams(num_symbols, qam_size):
    s_int = np.random.randint(low=0, high=qam_size, size=int(num_symbols))
    v_func = np.vectorize(get_qam_constellation)
    return s_int, v_func(s_int, qam_size)


def _ofdm_modulate(d_k_mtx, n_fft, l_cp, over_sample):
    a_k_mtx = np.fft.ifftshift(d_k_mtx, axes=0)
    s_n_mtx = np.fft.ifft(a_k_mtx, n_fft * over_sample, axis=0) * np.sqrt(n_fft)
    cp = s_n_mtx[-l_cp:, :]
    ofdm_symbols_time = np.vstack([cp, s_n_mtx])
    return ofdm_symbols_time.T


def _as_ofdm_centered_symbols(allocation_tones, n_fft, l_cp, over_sample, payload_data_mtx, sc_at_position=0):
    data_to_map = payload_data_mtx.T
    mid_nfft = n_fft // 2
    n_total_symbols = data_to_map.shape[1]
    d_k_mtx = np.zeros((n_fft, n_total_symbols), dtype=complex)
    indices = (np.arange(allocation_tones) + mid_nfft + sc_at_position) % n_fft
    d_k_mtx[indices, :] = data_to_map
    return _ofdm_modulate(d_k_mtx, n_fft, l_cp, over_sample)


def _ofdma_modulate(stream, n_sta_sc):
    n_blocks = len(stream) // n_sta_sc
    blocks = stream.reshape(n_blocks, n_sta_sc)
    return np.fft.fft(blocks, axis=1)


def _ofdma_demodulate(ofdma_symbols, n_symbols, n_sta_sc):
    streams = np.fft.ifft(ofdma_symbols, axis=0)
    return streams.T.flatten()[:n_symbols]


if __name__ == '__main__':
    main(n_fft=256, l_cp=16, n_symbols=200, tx=3, rx=4)
