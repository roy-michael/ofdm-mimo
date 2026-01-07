import numpy as np
import matplotlib.pyplot as plt

"""
Apply the MRC for a desired DoA to all DoAs in the range -90:90 and compute the post MRC response.
    • Consider 4 Rx antennas
    • Consider 16 Rx antennas
    • Consider different desired DoAs
"""


def main(fc):
    c = 3e8
    wavelength = c / fc

    theta_deg = 30
    phi_deg = np.linspace(-90, 90, 181, endpoint=True)

    phi_rad = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)
    get_mag_vectorized = np.vectorize(_get_magnitude)

    plt.figure(figsize=(10, 6))

    for N in {4, 16}:
        mags = get_mag_vectorized(N, theta_rad, phi_rad, wavelength)
        decibels = 20 * np.log10(mags)
        plt.plot(phi_deg, decibels, label=f"{N} Rx antennas", alpha=0.8)

    plt.title(f"MRC Response Comparison. Desired at {theta_deg} degrees")
    plt.xlabel("DoA (degrees)")
    plt.ylabel("Response (dB)")
    plt.ylim(-30, 0)
    plt.legend()
    plt.grid(True)
    plt.show()


def _build_steering_vector(N, ang, lambda_):
    d = lambda_ / 2

    return np.array([np.exp(-1j * 2 * np.pi * ((n * d * np.sin(ang)) / lambda_)) for n in range(N)])


def _get_magnitude(N, theta, phi, lambda_):
    theta_steer = _build_steering_vector(N, theta, lambda_)
    phi_steer = _build_steering_vector(N, phi, lambda_)

    return np.abs(np.vdot(theta_steer, phi_steer)) / N


if __name__ == '__main__':
    fc = 2.5e9  # 2.5GHz
    T = 1  # duration sec

    main(fc)
