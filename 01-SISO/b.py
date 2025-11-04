import numpy as np
from matplotlib import pyplot as plt

multipath = 10
bins = 100
fc = 2.5 * 10e9 # 2.5GHz
tries = 1000

def main():

    re_h_fc = []
    im_h_fc = []

    for i in range(tries):
        re, im = _calc_channel_tao()

        re_h_fc.append(re)
        im_h_fc.append(im)

    # 2. Plot the histogram directly
    plt.figure(figsize=(8, 5))
    plt.hist(re_h_fc, bins=bins, edgecolor='black', alpha=0.5, label="Real Part")
    plt.hist(im_h_fc, bins=bins, edgecolor='black', alpha=0.5, label="Imaginary Part")
    plt.title('The Rayleigh Channel Multipath')
    plt.xlabel('Value')
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.5)
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


def _calc_channel_tao():

    taos = np.random.uniform(low=50, high=100, size=multipath) * 1 / (3 * 10e8)
    h = np.sum(np.exp(-1j * 2 * np.pi * fc * taos))

    return h.real, h.imag

def _calc_channel():

    phis = np.random.uniform(0, 2 * np.pi, multipath)

    re_h_fc = np.sum(np.cos(phis))
    im_h_fc = -np.sum(np.sin(phis))

    return re_h_fc, im_h_fc

if __name__ == '__main__':
    main()