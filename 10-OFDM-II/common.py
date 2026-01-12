import numpy as np


def _get_qams(num_symbols, qam_size):
    s_int = np.random.randint(low=0, high=qam_size, size=int(num_symbols))
    return s_int, np.vectorize(get_qam_constellation)(s_int, qam_size)


def _get_complex(num_elements=None, rows=None, col=1, mean=0, std=1):
    size = (num_elements, rows, 2 * col) if rows else (num_elements, 2 * col)

    val = np.random.normal(size=size, loc=mean, scale=std).view(
        np.complex128
    ) / np.sqrt(2)

    return val if rows else np.squeeze(val)


def get_gray_to_pam_map(bits_per_axis):
    """
    Generates a mapping from bits (integer) to PAM amplitude levels
    using Gray coding.

    Returns: A dictionary {integer_bits: amplitude_level}
    """
    num_levels = 2 ** bits_per_axis  # Number of levels per axis (e.g., 4 for 16-QAM)

    # 1. Generate standard Gray codes for these bits
    # The formula (i ^ (i >> 1)) generates the sequence of gray codes
    gray_codes = [i ^ (i >> 1) for i in range(num_levels)]

    # 2. Define the Amplitude levels
    levels = np.arange(-num_levels + 1, num_levels, 2)

    # 3. Create the map: Gray_Value -> Amplitude
    gray_to_level = {}
    for i, gray_val in enumerate(gray_codes):
        gray_to_level[gray_val] = levels[i]

    return gray_to_level


def get_qam_constellation(symbol_int, qam_size):
    """
    Generates the full constellation and mapping for Square M-QAM.
    """
    if np.log2(qam_size) % 2 != 0:
        raise ValueError("This code only supports Square QAM (M = 4, 16, 64, 256...)")

    k = int(np.log2(qam_size))  # Total bits per symbol
    k_axis = k // 2  # Bits per axis (I and Q)

    # Get the Gray coding map for one axis
    pam_map = get_gray_to_pam_map(k_axis)

    # 1. Bitwise operations to split the integer
    # Upper half of bits -> Real (I)
    # Lower half of bits -> Imag (Q)
    i_bits = symbol_int >> k_axis
    q_bits = symbol_int & ((1 << k_axis) - 1)

    # 2. Map bits to PAM levels
    i_level = pam_map[i_bits]
    q_level = pam_map[q_bits]

    # 3. Create complex symbol
    complex_point = i_level + 1j * q_level

    return complex_point


def _as_ofdm_symbols(qam_payload, allocation_tones, n_fft, l_cp, over_sample, with_preamble=False):
    mid_nfft = n_fft / 2
    inactive_bw = np.round((n_fft - allocation_tones) / 2).astype(int)
    n_total_symbols = int(len(qam_payload) / allocation_tones)  # QAM Payload

    # Define active subcarriers (-n_fft/2 to n_fft/2, excluding DC)
    active_indices = np.concatenate(
        [np.arange(-(mid_nfft - inactive_bw), 0), np.arange(1, (mid_nfft - inactive_bw) + 1)]).astype(int)
    n_active = len(active_indices)

    payload_data = np.reshape(qam_payload, (n_total_symbols, allocation_tones)).astype(complex)
    all_data_freq = payload_data

    if with_preamble:
        n_total_symbols += 1
        preamble_data = np.ones((1, n_active), dtype=complex)
        all_data_freq = np.vstack([preamble_data, payload_data])

    d_k_mtx = np.zeros((n_total_symbols, n_fft), dtype=complex)
    # Map active subcarriers into the grid
    d_k_mtx[:, active_indices] = all_data_freq

    return _ofdm_modulate(d_k_mtx, n_fft, l_cp, over_sample)


def _ofdm_modulate(d_k_mtx, n_fft, l_cp, over_sample):
    """
    Transforms d_k (centered indices) to a_k (IFFT indices)
    and performs IFFT + CP.
    """

    # Rearrangement to a_k format, with positive/negative indices
    a_k_mtx = np.fft.ifftshift(d_k_mtx, axes=1)
    s_n_mtx = np.fft.ifft(a_k_mtx, n_fft * over_sample, axis=1) * np.sqrt(n_fft)

    # Add Cyclic Prefix. Take last L_cp samples and prepend to the start
    cp = s_n_mtx[:, -l_cp:]
    ofdm_symbol = np.hstack([cp, s_n_mtx])

    return ofdm_symbol


def _calc_papr(s):
    s_power = np.abs(s) ** 2

    return 10 * np.log10(np.max(s_power, axis=1) / np.mean(s_power, axis=1))
