# Assignment

Create an OFDMA system with **N_FFT=256** with **L_CP=16** and **4 Rx antennas** at AP.
* a. Assume 2 STAs transmitting 200syms UL (QPSK):
  1. STA0 transmitting a single stream over **SCs 0-49**.
  2. STA1 transmitting 2 streams (**SU-MIMO**) over **SCs 50-69**.

* b. Pilots:
  1. STA0 has a single preamble symbol.
  2. STA1 has 2 preamble symbols (one for each Tx ant.).

* c. Channels – for each STA consider a channel with:
  1. 10 equal power paths (uniform phase).
  2. Each path with random uniform delay up to CP duration (integer sample).
  3. Each path with uniform $(0, 2\pi]$ DoA and DoD.
  4. Assume $\lambda / 2$ ULAs.

* **d. Perform channel estimation and demodulate both STAs with linear detection.**
* **e. Show perfect reconstruction without noise.**

---


# Results
