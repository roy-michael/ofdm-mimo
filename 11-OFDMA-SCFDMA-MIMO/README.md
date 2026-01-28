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

The simulation implements an Uplink OFDMA system with SC-FDMA (DFT-spread OFDM) and Multi-User MIMO.

1.  **System Configuration:**
    *   **STA0:** Transmits a single stream on 50 subcarriers (SISO-OFDMA).
    *   **STA1:** Transmits 2 spatial streams on 20 subcarriers (MIMO-OFDMA).
    *   **Receiver:** The Access Point (AP) has 4 antennas.

2.  **Processing Steps:**
    *   **Channel Estimation:** The AP estimates the frequency-domain channel response for each user and stream using orthogonal preamble slots.
    *   **Multi-User Detection:**
        *   Users are separated in the frequency domain (OFDMA).
        *   STA1's two spatial streams are separated using Zero-Forcing (ZF) detection on each subcarrier.
    *   **SC-FDMA Demodulation:** The received frequency-domain symbols are transformed back to the time domain (IDFT) to recover the QPSK symbols.

3.  **Performance Verification:**
    *   The simulation is performed in a noise-free environment to verify the correctness of the transceiver chain.
    *   **Perfect Reconstruction:** The script compares the transmitted QPSK symbols with the received and demodulated symbols.
    *   **Output:** The console output confirms "PERFECT" reconstruction (zero bit errors) for all three streams (STA0, STA1-Stream1, STA1-Stream2). This validates that the OFDMA subcarrier mapping, MIMO spatial multiplexing, and SC-FDMA precoding are correctly implemented and reversed at the receiver.
