from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt

"""
Simulate the Channel A in the table below
    • Compute the response at some t0
    • Compare the response at t0 + dt1, dt1 << Tc
    • Compare the response at t0 + dt2, dt2 >> Tc
"""

NS = 1e-9


@dataclass
class Tap:
    relative_delay_ns: int
    avg_power_db: float


def _get_jakes_psd_filter(num_samples, fs, f_max):
    freqs = np.fft.fftfreq(num_samples, 1 / fs)  # generate frequency array for the mask
    mask = abs(freqs) < f_max
    psd = np.zeros(len(freqs))

    # Calculate the denominator terms
    f_norm = freqs[mask] / f_max
    denominator = f_max * np.sqrt(1 - f_norm ** 2)
    psd[mask] = 1.0 / denominator

    return np.sqrt(psd)


def _channel_phases(taps, fc):
    def as_signal(delay_sec):
        return np.exp(-1j * 2 * np.pi * fc * delay_sec)

    delay_phases = np.array([as_signal(t.relative_delay_ns * NS) for t in taps])
    return delay_phases


def main(taps, T, fs, fc, f_max, Tc):

    print(f"response for {len(taps)} taps, with Tc = {Tc} seconds and Fmax = {f_max}Hz")

    samples = np.linspace(0, T, int(T * fs), endpoint=False)
    num_samples = len(samples)

    psd_filter = _get_jakes_psd_filter(num_samples, fs, f_max)
    coefficients = _channel_coefficients(taps, num_samples, psd_filter)
    channel_phases = _channel_phases(taps, fc)

    channel_matrix = coefficients * channel_phases
    channel_matrix = channel_matrix.T

    # ---  Perform the Assignment Checks ---
    print(f"Simulation: Tc = {Tc * 1000:.2f} ms")

    # Define Indices for check
    # t0 = 0.1s
    idx_t0 = int(0.1 * fs)

    # dt1 = 1% of Tc (Should be highly correlated)
    dt1_sec = 0.01 * Tc
    idx_dt1 = idx_t0 + int(dt1_sec * fs)

    # dt2 = 500% of Tc (Should be uncorrelated)
    dt2_sec = 5.0 * Tc
    idx_dt2 = idx_t0 + int(dt2_sec * fs)

    # Extract Vectors (The "Response" at specific times)
    h_t0 = channel_matrix[:, idx_t0]
    h_dt1 = channel_matrix[:, idx_dt1]
    h_dt2 = channel_matrix[:, idx_dt2]

    # Calculate Correlations
    rho_short = _calculate_correlation(h_t0, h_dt1)
    rho_long = _calculate_correlation(h_t0, h_dt2)

    print(f"\n--- Results ---")
    print(f"Correlation at dt1 ({dt1_sec * 1000:.2f} ms): {rho_short:.4f} (Expected ~1.0)")
    print(f"Correlation at dt2 ({dt2_sec * 1000:.2f} ms): {rho_long:.4f}")

    # --- Plotting one tap magnitude over time ---
    time_axis = np.arange(num_samples) / fs
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, 20 * np.log10(np.abs(channel_matrix[0, :])))
    plt.title("Fading Profile of Tap 1 (0ns) over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()


def _calculate_correlation(h_t0, h_dt):
    dot_p = np.vdot(h_t0, h_dt)
    norm_t0 = np.linalg.norm(h_t0)
    norm_dt = np.linalg.norm(h_dt)

    return np.abs(dot_p) / (norm_t0 * norm_dt)


def _channel_coefficients(taps, num_samples, psd_filter):
    num_taps = len(taps)

    w = (np.random.randn(num_taps, num_samples) +
         1j * np.random.randn(num_taps, num_samples)) / np.sqrt(2)

    w_freq = np.fft.fft(w)

    w_filtered = np.fft.fft(w_freq * psd_filter)

    # Each tap process must have average power = 1.0 before we apply tap weights
    row_powers = np.mean(np.abs(w_filtered) ** 2, axis=1, keepdims=True)
    w_normalized = w_filtered / np.sqrt(row_powers)

    avg_power_lin = [np.sqrt(10 ** (t.avg_power_db / 10)) for t in taps]

    return w_normalized.T * avg_power_lin


def _from_db(power_db):
    return 10 ** (power_db / 10.0)


def _channel_frequency_response(t_ns, taps, f_c):
    delays_sec: List[float] = [((t_ns + t.relative_delay_ns) * NS) for t in taps]
    cfr = np.array([np.exp(-1j * 2 * np.pi * f_c * delay_sec) for delay_sec in delays_sec])

    # return np.fft.ifft(cfr)
    return cfr


if __name__ == '__main__':
    f_max = 10  # 10Hz
    fc = 2.5e9  # 2.5GHz
    T = 1  # duration
    fs = 10e3  # sampling rate
    Tc = 9 / (16 * np.pi * f_max)

    # _plot_jakes(T, f_s, f_max)

    taps = [
        Tap(0, 0.0),
        Tap(310, -1.0),
        Tap(710, -9.0),
        Tap(1090, -10.0),
        Tap(1730, -15.0),
        Tap(2510, -20.0)
    ]

    main(taps, T, fs, fc, f_max, Tc)
