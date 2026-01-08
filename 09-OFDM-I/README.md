### Assignment

Create an OFDM system with $N_{FFT}$=256 with $L_{CP}$=16
  1. Assume the $1^{st}$ symbol is known to Rx (Preamble) for channel estimation followed by 20 payload symbols
  2. Consider the channels $h_1=\delta(n)$, $h_2=\delta(n)+0.9j\delta(n-9)$
  3. Add 20dB SNR (to the time domain signal)
  4. Do channel estimation and equalize all payload symbols
  5. Compute average EVM $mean|\hat s-s|^2$
  6. Repeat d-e for perfect channel knowledge $H(k)=\sum_{n}h(n)e^{-j\frac{2\pi}{N}kn}$