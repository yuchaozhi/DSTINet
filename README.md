# DSTINet

**Complexity Analysis.** The vanilla spatial attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}(Q K^T) V,
$$

where $Q, K, V \in \mathbb{R}^{N \times C}$ with $N$ nodes and $C$ feature dimensions. The computational steps are: (1) $Q K^T$ computes an $N \times N$ attention matrix with complexity $O(N^2 C)$; (2) applying softmax to the $N \times N$ matrix has complexity $O(N^2)$; (3) multiplying the $N \times N$ matrix with $V$ has complexity $O(N^2 C)$. The total complexity is $O(N^2 C)$, which is quadratic in $N$.

In contrast, our CMI module leverages random feature mapping to achieve linear complexity:

$$
\text{CMI}(Q, K, V) = \tau(Q) \big(\tau(K)^T V\big),
$$

where $Q \in \mathbb{R}^{N \times C}$, $K \in \mathbb{R}^{N \times 2C}$, $V \in \mathbb{R}^{N \times C}$, and $\tau(\cdot)$ represents the softmax feature map. The computational steps are: (1) transpose and apply feature mapping: $\tau(K)^T \in \mathbb{R}^{2C \times N}$ with complexity $O(N C)$; (2) compute $\tau(K)^T V$: $(2C \times N) \times (N \times C) = 2C \times C$ matrix with complexity $O(N C^2)$; (3) apply feature mapping to $Q$: $\tau(Q) \in \mathbb{R}^{N \times C}$ with complexity $O(N C)$; (4) compute $\tau(Q)(\tau(K)^T V)$: $(N \times C) \times (C \times 2C) = N \times 2C$ with complexity $O(N C^2)$. The total complexity is $O(N C^2)$, which is linear in $N$.

The key optimization lies in the computation order: standard attention first computes the $N \times N$ attention matrix (complexity $O(N^2)$), while CMI first computes the $C \times C$ intermediate matrix (complexity $O(C^2)$). When $N \gg C$ (typical in traffic networks where $N$ ranges from hundreds to thousands while $C$ is typically 64-128), this reordering dramatically reduces computational cost. For example, with $N=1000$ and $C=64$: standard attention requires $1000^2 \times 64 = 64{,}000{,}000$ operations, while CMI requires $1000 \times 64^2 = 4{,}096{,}000$ operations, achieving a $15.6\times$ speedup.

Note that while the trainable adjacency matrix $A_{\text{bias}}$ involves computing $E_1 E_2^T$ (complexity $O(N^2 C)$), this matrix is precomputed during training and stored as a fixed parameter. During inference, $A_{\text{bias}}$ is simply loaded and used, contributing only $O(N C)$ complexity for the multiplication $\kappa(A_{\text{bias}})V$. Therefore, the overall inference complexity of CMI remains $O(N C^2)$.
