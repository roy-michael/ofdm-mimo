# Assignment

Create an OFDM system with $N_{FFT}$=256 with $L_{CP}$=16
  1. Assume the $1^{st}$ symbol is known to Rx (Preamble) for channel estimation followed by 20 payload symbols
  2. Consider the channels $h_1=\delta(n)$, $h_2=\delta(n)+0.9j\delta(n-9)$
  3. Add 20dB SNR (to the time domain signal)
  4. Do channel estimation and equalize all payload symbols
  5. Compute average EVM $mean|\hat s-s|^2$
  6. Repeat d-e for perfect channel knowledge $H(k)=\sum_{n}h(n)e^{-j\frac{2\pi}{N}kn}$

# Results

The simulation implements an OFDM system with 256 subcarriers and a cyclic prefix of 16. It compares the performance 
(EVM) of channel estimation using a preamble versus perfect channel knowledge for two different channels.

**Channel Models:**
*   **H1:** A flat fading channel (single tap). $h_1[n] = \delta[n]$. In the frequency domain, this is a constant magnitude response.
*   **H2:** A frequency-selective fading channel (two taps). $h_2[n] = \delta[n] + 0.9j\delta[n-9]$. This creates deep nulls in the frequency spectrum.

**EVM Results (at 20dB SNR):**
*   **Perfect H1 (6.46%):** This represents the baseline performance limited only by the AWGN noise.
*   **Estimated H1 (8.11%):** The EVM increases slightly due to the estimation error from the noisy preamble.
*   **Perfect H2 (19.88%):** The EVM is significantly higher than H1 even with perfect channel knowledge. This is 
because H2 has deep spectral nulls where the SNR is very low (noise enhancement during equalization).
*   **Estimated H2 (29.41%):** The combination of channel estimation error and noise enhancement in deep fades leads to 
the highest EVM.
