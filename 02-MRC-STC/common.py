import numpy as np


# QPSK symbols
def _get_qpsk(num_symbols):
    s_int = np.random.randint(low=0, high=4, size=num_symbols)
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


# Find the closest constellation symbol for each received symbol
def _ml_detect(received_symbols, constellation):
    distances = np.abs(received_symbols[:, np.newaxis] - constellation)
    return np.argmin(distances, axis=1)


# Find the closest constellation symbol for each received symbol
def _ls_detect(received_symbols, constellation):
    distances = np.abs(received_symbols[:, np.newaxis, :] - np.array([np.array([constellation] * 2).transpose()])) ** 2

    return np.argmin(distances, axis=1)


def _get_complex(num_elements, col=1, mean=0, std=1):
    return np.squeeze(
        np.random.normal(
            size=(num_elements, (2 * col)),
            loc=mean,
            scale=std)
        .view(np.complex128) / np.sqrt(2))
