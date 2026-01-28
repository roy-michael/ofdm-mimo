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

def _as_ofdm_centered_symbols(allocation_tones, n_fft, l_cp, over_sample, payload_data_mtx, sc_at_position=0):

    data_to_map = payload_data_mtx.T
    mid_nfft = n_fft // 2
    n_total_symbols = data_to_map.shape[1]

    d_k_mtx = np.zeros((n_fft, n_total_symbols), dtype=complex)
    # Map active subcarriers into the grid
    indices = np.arange(allocation_tones) + mid_nfft + sc_at_position
    indices = indices % n_fft

    d_k_mtx[indices, :] = data_to_map

    return _ofdm_modulate(d_k_mtx, n_fft, l_cp, over_sample)


def _from_ofdm_centered_symbols(allocation_tones, n_fft, ofdm_symbols_mtx, sc_at_position=0):

    data_to_map = ofdm_symbols_mtx.T
    mid_nfft = n_fft // 2
    n_total_symbols = data_to_map.shape[1]

    zp = np.fft.fftshift(np.fft.fft(ofdm_symbols_mtx, axis=1), axes=1)  # / TODO: np.sqrt(n_fft)??

    return zp[sc_at_position:sc_at_position+allocation_tones, :]



def _as_ofdm_symbols(allocation_tones, n_fft, l_cp, over_sample, payload_data=None, payload_data_mtx=None, with_preamble=False, preamble_at_position=0, sc_at_position=0):
    mid_nfft = n_fft / 2
    inactive_bw = np.round((n_fft - allocation_tones) / 2).astype(int)

    # Define active subcarriers (-n_fft/2 to n_fft/2, excluding DC)
    active_indices = np.concatenate(
        [
            np.arange(-(mid_nfft - inactive_bw), 0),
            np.arange(1, (mid_nfft - inactive_bw) + 1)
        ]).astype(int)
    n_active = len(active_indices)

    if payload_data_mtx is None:
        n_total_symbols = int(len(payload_data) / allocation_tones)  # QAM Payload
        payload_data_mtx = np.reshape(payload_data, (n_total_symbols, allocation_tones)).astype(complex)
    else:
        n_total_symbols = payload_data_mtx.shape[1]

    all_data_freq = payload_data_mtx

    if with_preamble:
        n_total_symbols += 1 + preamble_at_position
        preamble_data = np.ones((1, n_active), dtype=complex)
        all_data_freq = np.vstack([preamble_data, payload_data_mtx])

    d_k_mtx = np.zeros((n_fft, n_total_symbols), dtype=complex)
    # Map active subcarriers into the grid
    d_k_mtx[active_indices, :] = all_data_freq

    return _ofdm_modulate(d_k_mtx, n_fft, l_cp, over_sample)


def _ofdm_demodulate(y_no_cp, n_fft, l_cp):
    """
    Performs -CP and FFT.
    Transforms a_k (IFFT indices) to z_p (centered indices)
    """

    # y_no_cp = rx_mtx[:, l_cp:]
    bk_freq = np.fft.fftshift(np.fft.fft(y_no_cp, axis=1) / np.sqrt(n_fft), axes=1)

    return bk_freq.T


def _ofdm_modulate(d_k_mtx, n_fft, l_cp, over_sample):
    """
    Transforms d_k (centered indices) to a_k (IFFT indices)
    and performs IFFT + CP.
    """

    # Rearrangement to a_k format, with positive/negative indices
    a_k_mtx = np.fft.ifftshift(d_k_mtx, axes=0)
    s_n_mtx = np.fft.ifft(a_k_mtx, n_fft * over_sample, axis=0) * np.sqrt(n_fft)

    # Add Cyclic Prefix. Take last L_cp samples and prepend to the start
    cp = s_n_mtx[-l_cp:, :]
    ofdm_symbols_time = np.vstack([cp, s_n_mtx])

    return ofdm_symbols_time.T


def _calc_papr(s):
    s_power = np.abs(s) ** 2

    return 10 * np.log10(np.max(s_power, axis=1) / np.mean(s_power, axis=1))
