import itertools

import numpy as np


# QPSK symbols
def _get_qpsk(num_symbols):
    s_int = np.random.randint(low=0, high=4, size=int(num_symbols))
    s_symbols = _to_symbols(s_int)

    return s_int, s_symbols


def _to_symbols(int_array, amplitude=1):
    degrees = int_array * 360 / 4.0 + 45  # 45, 135, 225, 315 degrees
    radians = np.deg2rad(degrees)
    symbols = amplitude * np.exp(1j * radians) / np.sqrt(2)  # normalized QPSK complex symbols

    return symbols


def _from_symbols(array):
    degrees = np.angle(array, deg=True)
    int_array = np.round((degrees - 45) * 4 / 360)

    return int_array


def _zf_detect(received_symbols, H_tilde, constellation, tx):
    """
    Performs Zero Forcing detection for Spatial Multiplexing.
    """

    H_tilde_inv = (
        np.sqrt(tx)
        * np.linalg.pinv(np.conj(H_tilde).transpose(0, 2, 1) @ H_tilde)
        @ np.conj(H_tilde).transpose(0, 2, 1)
    )
    s_hat = (H_tilde_inv @ received_symbols).squeeze(axis=-1)

    return _ls_detect(s_hat, constellation)


# Find the closest constellation symbol for each received symbol
def _ml_detect(received_symbols, constellation):
    distances = np.abs(received_symbols - constellation)
    min_distance = np.argmin(distances, axis=1)

    return constellation[min_distance]


def _ml_detect2(received_symbols, H_tilde, constellation, tx):
    """
    Performs Maximum Likelihood (ML) detection for Spatial Multiplexing.
    """

    # Create all possible transmitted symbol vectors
    possible_s_options = np.array(list(itertools.product(constellation, repeat=tx)))  # tx * 2^4 (for QAM4)

    # Calculate the expected received signal for each possible transmitted vector.
    possible_options = H_tilde @ possible_s_options.T  # num_symbols * tx * tx^4

    # Calculate squared Euclidean distance: ||y - H*s_candidate||^2 for all candidates.
    distances = np.linalg.norm(received_symbols - possible_options, axis=1) ** 2

    min_idx = np.argmin(distances, axis=1)
    return possible_s_options[min_idx]


# Find the closest constellation symbol for each received symbol
def _ls_detect(received_symbols, constellation):
    distances = np.abs(received_symbols - constellation) ** 2

    # Find the index of the minimum distance for each symbol in the vector
    # Return the detected symbols from the constellation
    min_indices = np.argmin(distances, axis=2)
    return constellation[min_indices]


def _get_complex(num_elements=None, rows=None, col=1, mean=0, std=1):
    size = (num_elements, rows, 2 * col) if rows else (num_elements, 2 * col)

    val = np.random.normal(size=size, loc=mean, scale=std).view(np.complex128) / np.sqrt(2)

    return val if rows else np.squeeze(val)
