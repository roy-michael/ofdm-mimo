# Assignment
1. In a 2x2 system, employ SVD precoding $x = \frac{1}{\sqrt{2}}\mathbf{Vs}$.
   * Decode with ML and ZF and show identical results.
   * Plot the SER curve of each stream separately.


2. Consider a receiver with 4 antennas and the model:

$ \mathbf{y} = \mathbf{h}s + \sqrt{P_g}\mathbf{g}r + \rho\mathbf{n} $

where $\mathbf{h}$, $\mathbf{g}$ and $r$ are $CN(0,1)$ iid and $P_g = 0.3162$ (5dB SIR).
   * Decode with MRC.
   * Decode with MVDR.
   * Compare the MVDR with MRC without interference.