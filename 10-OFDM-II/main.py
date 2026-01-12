import numpy as np
import matplotlib.pyplot as plt

from common import _get_complex, _get_qams, _ofdm_modulate, _calc_papr, _as_ofdm_symbols

"""
OFDM PAPR
a) Build an OFDM signal with
    • Flexible modulation (QPSK – 256QAM)
    • Flexible N_FFT
    • Flexible Oversampling (through longer IFFT)
b) Compute the PAPR of each OFDM symbol
c) Plot the CCDF of the PAPR
d) Repeat for N_FFT=64, 256, 1024
e) Repeat for QPSK, 16QAM, 256QAM
"""


def main():
    n_fft = 128
    l_cp = 16
    allocation_tones = 100
    over_sampling = 4
    qam_sizes = [4, 16, 64, 256]
    num_ofdm_symbols = int(10e3)

    papr_vals = []
    ccdf_probs = []

    papr_ccdf = {}

    for qam_size in qam_sizes:
        # _, payload_data = _get_qams(num_symbols=(allocation_tones * num_ofdm_symbols), qam_size=qam_size)
        # ofdm_symbols = _as_ofdm_symbols(payload_data, allocation_tones=allocation_tones, n_fft=n_fft, l_cp=l_cp,
        #                            over_sample=over_sampling)

        paprs, ccdfs = calculate_ccdf(n_fft, l_cp, qam_size, allocation_tones, over_sampling, num_ofdm_symbols)

        # papr_vals.append(paprs)
        # ccdf_probs.append(ccdfs)
        papr_ccdf[qam_size] = (paprs, ccdfs)

        # print(f"papr: {papr}")

    # ==========================================
    # Plotting
    # ==========================================
    plt.figure(figsize=(8, 6))
    for qam_size, (papr_vals, ccdf_probs) in papr_ccdf.items():
        plt.semilogy(papr_vals, ccdf_probs, linewidth=2, label=f'QAM Size: {qam_size}')

    plt.title("PAPR CCDF (M=100, N_FFT=128, OS=4)")
    plt.xlabel("PAPR threshold (dB)")
    plt.ylabel("Probability (PAPR > threshold)")
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.ylim(1e-4, 1)  # Limit Y-axis to match standard plots
    plt.legend()
    plt.tight_layout()


    """
    d) Repeat for N_FFT=64, 256, 1024  
    e) Repeat for QPSK, 16QAM, 256QAM
    """
    papr_ccdf = {}
    qam_size = 16
    for m, n_fft in [(10, 64), (200, 256), (1000, 1024)]:
        paprs, ccdfs = calculate_ccdf(n_fft, l_cp, qam_size, m, over_sampling, num_ofdm_symbols)
        papr_ccdf[(m, n_fft, qam_size)] = (paprs, ccdfs)
        print(f"done papr: {n_fft}, {qam_size}")
    _plot(papr_ccdf, "PAPR CCDF: OFDM, 16QAM with M allocated tones", over_sampling)

    papr_ccdf = {}
    m = 100
    n_fft = 128
    for qam_size in [4, 16, 256]:
        paprs, ccdfs = calculate_ccdf(n_fft, l_cp, qam_size, m, over_sampling, num_ofdm_symbols)
        papr_ccdf[(m, n_fft, qam_size)] = (paprs, ccdfs)
        print(f"done papr: {n_fft}, {qam_size}")
    _plot(papr_ccdf, "PAPR CCDF: OFDM with M=100 allocated tones, Different Modulation orders", over_sampling)

    plt.show()

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


if __name__ == '__main__':
    main()
