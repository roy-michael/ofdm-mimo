1. Repeat the SISO case (Tx, channel, noise addition, Rx) and produce the SER plots in the previous slide (Rayleigh and AWGN)

2. Consider a channel of the form
   
$$h(\tau) = \sum_{i=1}^{10} 1 \cdot \delta(\tau - \tau_i)$$

where $\tau_i$ are uniformly distributed $[50m, 100m] \cdot \frac{1}{3 \cdot 10^8 m/\sec}$

Show the distribution of the real and imaginary part of
$$H(f = 2.5GHz)$$

---
Use the built-in functions randsrc to create QPSK signals easily
rand (for uniform RV) and hist (for plotting historgrams)
