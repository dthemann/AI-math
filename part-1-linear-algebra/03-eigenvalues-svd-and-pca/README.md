# Module 03 — Eigenvalues, SVD and PCA

> After this module you can take any matrix apart into its "natural axes", use that to compress data, rank web pages, run PCA from scratch, solve rank-deficient least-squares problems, and explain why badly scaled features make optimisation slow.

## Why this matters for ML and AI

Module 02 treated a matrix as a machine that moves vectors around. This module asks a sharper question: *along which directions does that machine act simply?* The answer — eigenvectors for square matrices, singular vectors for any matrix — is the single most reused idea in applied linear algebra. Once you can read a matrix through its spectrum you can see at a glance whether a dynamical system blows up or settles down, how much of a dataset lives in a handful of directions, and how badly rounding errors will be amplified when you solve a system.

The applications are not academic. Google's original ranking of the web was the dominant eigenvector of a Markov chain with one state per web page, computed by the power iteration you will implement here. The Netflix Prize was won by teams who factorised a 480,189 × 17,770 ratings matrix into a few dozen latent factors — truncated SVD in spirit. `sklearn.decomposition.PCA` is literally the SVD of the centred data matrix. LoRA, the dominant method for fine-tuning large language models, is a bet that weight updates are low-rank. And every time an optimiser crawls across a long narrow valley, the villain is the condition number of a Hessian — the ratio of its largest to smallest eigenvalue.

The module builds up in a straight line: eigenvectors (what a square matrix preserves) → diagonalisation (why matrix powers are easy) → symmetric matrices (where everything is real and orthogonal) → the SVD (the same story for *every* matrix) → low-rank approximation and PCA (SVD as a compression tool) → the pseudo-inverse (SVD as a solver) → the condition number (SVD as a diagnostic).

## Learning objectives

By the end of this module you will be able to:

- Compute eigenvalues and eigenvectors of a 2×2 matrix by hand, and explain what they mean geometrically.
- Diagonalise a matrix, use $A^k = PD^kP^{-1}$ to predict long-run behaviour, and implement power iteration (including PageRank with damping).
- State the spectral theorem and use it to classify quadratic forms as positive (semi)definite.
- Compute the SVD of a small matrix by hand, relate it to the eigendecomposition of $A^\top A$, and read off rank and the four fundamental subspaces.
- Apply the Eckart–Young theorem to build the best rank-$k$ approximation and quantify the error it leaves.
- Implement PCA from scratch (centring, covariance, eigenvectors or SVD, explained variance, projection, reconstruction) and match scikit-learn.
- Build the Moore–Penrose pseudo-inverse from the SVD and explain why it gives the minimum-norm least-squares solution.
- Compute a condition number, predict how much error a linear solve can amplify, and connect conditioning to gradient-descent speed and feature scaling.

## Prerequisites

- [Module 01 — Vectors and geometry](../01-vectors-and-geometry/README.md): norms, dot products, orthogonality, projections, orthonormal bases.
- [Module 02 — Matrices and linear maps](../02-matrices-and-linear-maps/README.md): matrix multiplication, rank, null space, determinants, orthogonal matrices ($Q^\top Q = I$), least squares and the normal equations, change of basis ($P^{-1}AP$), Markov transition matrices.

Recall in particular that an orthogonal matrix preserves lengths and angles, that $\det(A) = 0$ exactly when $A$ squashes some direction to zero, and that similar matrices $P^{-1}AP$ represent the same map in different coordinates.

## 1. Eigenvectors and eigenvalues

**Intuition.** Picture a square matrix $A$ as a machine that takes every arrow in the plane and moves it. Most arrows get rotated *and* stretched. A few special arrows are only stretched (or shrunk, or flipped) — they keep pointing along the same line. Those directions are the eigenvectors, and the stretch factor along each is its eigenvalue. If you know these directions, you know the machine: every other vector is a combination of them, and the machine acts on each piece separately.

**Definition.** For a square matrix $A \in \mathbb{R}^{n\times n}$, a non-zero vector $v$ is an *eigenvector* with *eigenvalue* $\lambda$ if

$$A v = \lambda v .$$

Rewriting as $(A - \lambda I)v = 0$: a non-zero solution exists only if $A - \lambda I$ is singular, so the eigenvalues are the roots of the *characteristic polynomial* $\det(A - \lambda I) = 0$. For a $2\times 2$ matrix this is

$$\lambda^2 - \operatorname{tr}(A)\,\lambda + \det(A) = 0,$$

so the two eigenvalues always add up to the trace and multiply to the determinant. All eigenvectors for one $\lambda$ form a subspace (the *eigenspace*). The polynomial may have complex roots: a rotation by 90°, $R = \begin{pmatrix} 0 & -1 \\ 1 & 0\end{pmatrix}$, gives $\lambda^2 + 1 = 0$, so $\lambda = \pm i$ — no real direction survives a rotation unchanged, and complex eigenvalues always signal a rotational component. A matrix can also have too few eigenvectors: the shear $\begin{pmatrix} 1 & 1 \\ 0 & 1\end{pmatrix}$ has the double eigenvalue $1$ but only one eigenvector direction, $(1, 0)$.

**Worked example.** Take $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$. The trace is $7$ and the determinant is $12 - 2 = 10$, so

$$\lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2) = 0 \quad\Rightarrow\quad \lambda_1 = 5,\ \lambda_2 = 2.$$

For $\lambda_1 = 5$: $A - 5I = \begin{pmatrix} -1 & 1 \\ 2 & -2\end{pmatrix}$, whose null space is spanned by $v_1 = (1, 1)$. Check: $A v_1 = (5, 5) = 5 v_1$. For $\lambda_2 = 2$: $A - 2I = \begin{pmatrix} 2 & 1 \\ 2 & 1\end{pmatrix}$, null space $v_2 = (1, -2)$. Check: $A v_2 = (4 - 2,\ 2 - 6) = (2, -4) = 2 v_2$. Any vector $x = a v_1 + b v_2$ is mapped to $A x = 5a\,v_1 + 2b\,v_2$: the component along $v_1$ grows two and a half times faster than the component along $v_2$. Note that $v_1$ and $v_2$ are *not* orthogonal — that only happens for symmetric matrices (Section 3).

**Real-life example (population biology).** A Leslie matrix tracks an age-structured population. Suppose juveniles produce on average 1 offspring per season and adults 2, and half of the juveniles survive to adulthood: with state $(\text{juveniles}, \text{adults})$,

$$L = \begin{pmatrix} 1 & 2 \\ 0.5 & 0 \end{pmatrix}, \qquad \lambda^2 - \lambda - 1 = 0 \ \Rightarrow\ \lambda_1 = \tfrac{1+\sqrt 5}{2} \approx 1.618,\ \lambda_2 \approx -0.618 .$$

The dominant eigenvalue says the population grows by 61.8 % per season in the long run (the golden ratio, because this is the Fibonacci recurrence in disguise), and its eigenvector $(0.764, 0.236)$ is the *stable age structure*: whatever the starting mix, after a dozen seasons about 76 % of the population are juveniles. The same mathematics gives the resonance frequencies of a bridge (eigenvalues of its stiffness–mass system) and the steady states of chemical reaction networks.

**In ML/AI this is …** the eigenvalues of the loss Hessian, which set the largest safe learning rate for gradient descent ($\eta < 2/\lambda_{\max}$ on a quadratic, Module 05); the eigenvalues of a recurrent weight matrix $W_{hh}$, whose repeated application makes gradients grow or shrink like $\lambda^t$ (exploding and vanishing gradients, Module 16); and the eigenvectors of a graph Laplacian used in spectral clustering. In NumPy: `np.linalg.eig` (general) and `np.linalg.eigh` (symmetric).

## 2. Diagonalisation and matrix powers

**Intuition.** In the eigenvector coordinate system, $A$ is nothing but a set of independent scalings. "Change to eigen-coordinates, scale each coordinate, change back" is all a diagonalisable matrix ever does. The payoff is that $A$ applied a thousand times is just the same scalings raised to the thousandth power — and the largest one dominates everything else.

**Definition.** If $A$ has $n$ linearly independent eigenvectors, stack them as columns of $P$ and the eigenvalues in a diagonal $D$. Then $AP = PD$, so

$$A = P D P^{-1}, \qquad A^k = P D^k P^{-1}, \qquad A^k x = \sum_{i=1}^{n} c_i\,\lambda_i^k\, v_i \ \text{ where } x = \sum_i c_i v_i .$$

If $|\lambda_1| > |\lambda_2| \ge \dots$, the term $c_1 \lambda_1^k v_1$ eventually dwarfs the rest, so $A^k x$ aligns with the *dominant eigenvector* $v_1$. **Power iteration** turns this into an algorithm: start from a random $x_0$ and repeat $x_{k+1} = A x_k / \|A x_k\|$. The angle to $v_1$ shrinks like $|\lambda_2/\lambda_1|^k$, and $\lambda_1 \approx x_k^\top A x_k$ (the Rayleigh quotient of Section 3). A column-stochastic transition matrix $T$ (columns sum to one) always has the eigenvalue $1$ and no eigenvalue of larger modulus; for an irreducible, aperiodic chain it is strictly dominant, and its eigenvector, normalised to sum to one, is the *stationary distribution* $\pi$ with $T\pi = \pi$.

**Worked example (two-state Markov chain).** Two brands compete. Each month 90 % of brand A's customers stay and 10 % switch; 50 % of brand B's customers switch to A and 50 % stay. With market shares as a column vector,

$$T = \begin{pmatrix} 0.9 & 0.5 \\ 0.1 & 0.5\end{pmatrix}, \qquad \lambda^2 - 1.4\lambda + 0.4 = (\lambda - 1)(\lambda - 0.4) = 0 .$$

For $\lambda = 1$: $(T - I) = \begin{pmatrix} -0.1 & 0.5 \\ 0.1 & -0.5\end{pmatrix}$ gives $v \propto (5, 1)$, so $\pi = (5/6, 1/6) \approx (0.833, 0.167)$. For $\lambda = 0.4$: $v_2 = (1, -1)$. Start with all customers at B, $x_0 = (0, 1) = \pi + c\,v_2$ with $c = -5/6$. Then

$$x_k = \pi - \tfrac{5}{6}\,(0.4)^k\,(1, -1), \qquad x_3 = \left(\tfrac56 - \tfrac56\cdot 0.064,\ \tfrac16 + \tfrac56 \cdot 0.064\right) = (0.78, 0.22).$$

Multiplying directly confirms it: $x_1 = (0.5, 0.5)$, $x_2 = (0.7, 0.3)$, $x_3 = (0.78, 0.22)$. Since $0.4^{10} \approx 10^{-4}$, the shares are within 0.01 percentage points of $(83.3\,\%, 16.7\,\%)$ after ten months, whatever the start — the eigenvalue $0.4$ *is* the speed of forgetting.

**Real-life example (web search).** PageRank models a surfer who follows a random out-link with probability $d = 0.85$ and jumps to a uniformly random page otherwise. With $M$ the column-stochastic link matrix ($M_{ij} = 1/\text{outdeg}(j)$ if page $j$ links to $i$; a page with no out-links is treated as linking everywhere), the "Google matrix" is

$$G = d\,M + \frac{1-d}{n}\,\mathbf{1}\mathbf{1}^\top ,$$

and the PageRank vector is its stationary distribution. Because every entry of $G$ is positive, the Perron–Frobenius theorem guarantees a unique positive stationary vector, and one can show the second eigenvalue satisfies $|\lambda_2| \le d$. So power iteration converges at least as fast as $0.85^k$: about 50 iterations ($0.85^{52} \approx 2\times10^{-4}$) give four digits, regardless of whether the web has ten pages or ten billion. The lesson notebook runs this on a six-page web.

**In ML/AI this is …** PageRank and personalised PageRank (recommendation, graph preprocessing); the Lanczos/Arnoldi solvers behind `scipy.sparse.linalg.eigs`, which are power iteration with memory; spectral normalisation in GANs, which runs one power-iteration step per training step to estimate each weight matrix's largest singular value; and the stationary distributions that MCMC (Module 18) and value iteration (Module 19) converge to.

## 3. Symmetric matrices and the spectral theorem

**Intuition.** Symmetric matrices ($A = A^\top$) are the well-behaved family: their eigenvalues are always real and their eigenvectors can always be chosen perpendicular to each other. Geometrically, a symmetric matrix is a pure stretch along a set of mutually orthogonal axes — no rotation, no shear. Covariance matrices, Hessians, kernel matrices and graph Laplacians are all symmetric, which is why so much of ML lives in this comfortable world.

**Definition.** The **spectral theorem**: every real symmetric $A$ can be written as

$$A = Q \Lambda Q^\top = \sum_{i=1}^n \lambda_i\, q_i q_i^\top ,$$

with $Q$ orthogonal ($Q^\top Q = I$, columns $q_i$ orthonormal eigenvectors) and $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ real. The short proof that eigenvectors for *distinct* eigenvalues are orthogonal: if $Au = \lambda_1 u$ and $Av = \lambda_2 v$, then $\lambda_1 u^\top v = (Au)^\top v = u^\top A v = \lambda_2 u^\top v$, so $(\lambda_1 - \lambda_2)\,u^\top v = 0$.

The **quadratic form** $x^\top A x$ becomes a weighted sum of squares in eigen-coordinates $y = Q^\top x$:

$$x^\top A x = \sum_i \lambda_i\, y_i^2 .$$

Hence $A$ is **positive semidefinite** (PSD, written $A \succeq 0$: $x^\top A x \ge 0$ for all $x$) exactly when all $\lambda_i \ge 0$, and **positive definite** ($A \succ 0$) when all $\lambda_i > 0$. Two important PSD families: Gram matrices $X^\top X$, since $x^\top X^\top X x = \|Xx\|^2 \ge 0$; and covariance matrices $\Sigma$, since $w^\top \Sigma w = \operatorname{Var}(w^\top z) \ge 0$ (Module 09). The **Rayleigh quotient** $R(x) = x^\top A x / x^\top x$ satisfies $\lambda_{\min} \le R(x) \le \lambda_{\max}$, with the maximum attained at the top eigenvector — the fact PCA rests on.

**Worked example.** $S = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$ has trace 4 and determinant 3, so $\lambda^2 - 4\lambda + 3 = 0$ gives $\lambda_1 = 3$, $\lambda_2 = 1$, with unit eigenvectors $q_1 = (1, 1)/\sqrt2$ and $q_2 = (1, -1)/\sqrt2$ (orthogonal, as promised). Reconstruct:

$$3\,q_1 q_1^\top + 1\,q_2 q_2^\top = 3\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5\end{pmatrix} + \begin{pmatrix} 0.5 & -0.5 \\ -0.5 & 0.5\end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}. \checkmark$$

The quadratic form $x^\top S x = 2x_1^2 + 2x_1x_2 + 2x_2^2$ becomes $3y_1^2 + y_2^2$ with $y_1 = (x_1 + x_2)/\sqrt2$, $y_2 = (x_1 - x_2)/\sqrt2$; at $x = (1, 0)$ both give $2$. The level set $x^\top S x = 1$ is an ellipse with semi-axes $1/\sqrt3 \approx 0.577$ along $q_1$ and $1$ along $q_2$: the matrix is "stiffest" along $(1,1)$. Both eigenvalues are positive, so $S \succ 0$.

**Real-life example (finance).** The risk of a portfolio with weights $w$ over assets with covariance $\Sigma$ is the quadratic form $w^\top \Sigma w$. Two assets with annual volatilities 20 % and 30 % and correlation 0.2 have

$$\Sigma = \begin{pmatrix} 0.04 & 0.012 \\ 0.012 & 0.09\end{pmatrix}, \qquad w = (0.5, 0.5):\ w^\top\Sigma w = 0.01 + 0.0225 + 0.006 = 0.0385,$$

a volatility of $\sqrt{0.0385} \approx 19.6\,\%$ — below the 25 % average of the two, the diversification effect. The eigenvalues $0.0927$ and $0.0373$ say 71 % of the total variance sits in one direction (eigenvector $\approx (0.22, 0.98)$, dominated by the riskier asset), and the eigenvector $(0.98, -0.22)$ is the lowest-risk way to trade the two against each other. Engineers do the identical computation with a stiffness matrix $K$: the elastic energy $\tfrac12 x^\top K x$ must be non-negative, so $K \succeq 0$.

**In ML/AI this is …** the covariance matrix (Module 09) and the Hessian (Modules 04–05): a positive-definite Hessian certifies a local minimum, and its eigenvalue ratio is the condition number of Section 8. Kernel (Gram) matrices in SVMs and Gaussian processes must be PSD for the kernel to be valid (Mercer's condition). In code: `np.linalg.eigh`, which exploits symmetry to return real, sorted eigenvalues and an orthonormal $Q$.

## 4. The singular value decomposition

**Intuition.** Eigendecomposition needs a square matrix and still may deliver complex numbers or too few eigenvectors. The SVD fixes all of this by allowing *two* different orthonormal bases, one for the input space and one for the output space. Every matrix — rectangular, rank-deficient, anything — is then "rotate, stretch along the axes, rotate again". A unit circle in the input space always comes out as an ellipse whose semi-axes are the singular values.

**Definition.** Any $A \in \mathbb{R}^{m\times n}$ factors as

$$A = U \Sigma V^\top = \sum_{i=1}^{r} \sigma_i\, u_i v_i^\top ,$$

with $U \in \mathbb{R}^{m\times m}$ and $V \in \mathbb{R}^{n\times n}$ orthogonal, and $\Sigma \in \mathbb{R}^{m\times n}$ zero except for the *singular values* $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ on its diagonal, where $r = \operatorname{rank}(A)$. Equivalently $A v_i = \sigma_i u_i$: the right singular vector $v_i$ is sent to $\sigma_i$ times the left singular vector $u_i$. The link to eigenvalues is

$$A^\top A = V \Sigma^\top\Sigma V^\top, \qquad A A^\top = U \Sigma \Sigma^\top U^\top ,$$

so the $v_i$ are eigenvectors of $A^\top A$, the $u_i$ eigenvectors of $AA^\top$, and $\sigma_i^2$ are the shared non-zero eigenvalues — real and non-negative because both matrices are PSD. Consequences: $\|A\|_2 = \sigma_1$ (the largest stretch), $\|A\|_F^2 = \sum_i \sigma_i^2$, $|\det A| = \prod_i \sigma_i$ for square $A$, and the SVD lays out the **four fundamental subspaces**: $\operatorname{col}(A) = \operatorname{span}\{u_1,\dots,u_r\}$, $\operatorname{null}(A^\top) = \operatorname{span}\{u_{r+1},\dots,u_m\}$, $\operatorname{row}(A) = \operatorname{span}\{v_1,\dots,v_r\}$, $\operatorname{null}(A) = \operatorname{span}\{v_{r+1},\dots,v_n\}$. The *thin* SVD keeps only the first $r$ (or $\min(m,n)$) columns; `np.linalg.svd(A, full_matrices=False)` returns it.

**Worked example.** $A = \begin{pmatrix} 3 & 0 \\ 4 & 5\end{pmatrix}$. Then $A^\top A = \begin{pmatrix} 25 & 20 \\ 20 & 25\end{pmatrix}$ with eigenvalues $45$ and $5$ (trace 50, det 225), so

$$\sigma_1 = \sqrt{45} = 3\sqrt5 \approx 6.708, \qquad \sigma_2 = \sqrt5 \approx 2.236, \qquad v_1 = \tfrac{1}{\sqrt2}(1, 1),\ v_2 = \tfrac{1}{\sqrt2}(1, -1).$$

Left singular vectors from $u_i = A v_i / \sigma_i$: $A v_1 = (3, 9)/\sqrt2$, so $u_1 = (1, 3)/\sqrt{10}$; $A v_2 = (3, -1)/\sqrt2$, so $u_2 = (3, -1)/\sqrt{10}$. Check $\sigma_1\sigma_2 = \sqrt{225} = 15 = \det A$. Note the eigenvalues of this (triangular) matrix are $3$ and $5$ — *not* the singular values; only their product agrees. The unit circle is mapped to an ellipse with semi-axes $6.71$ along $u_1$ and $2.24$ along $u_2$.

**Real-life example (computer vision and structural biology).** Aligning two 3-D point clouds — a scanned part to its CAD model, or two protein structures — means finding the rotation $R$ minimising $\sum_i \|R p_i - q_i\|^2$ after centring. The Kabsch algorithm computes the $3\times3$ cross-covariance $H = \sum_i p_i q_i^\top$, takes its SVD $H = U\Sigma V^\top$, and returns $R = V U^\top$ (with one sign flipped if $\det < 0$, to exclude a reflection). Three singular values tell you how well-determined the alignment is: a nearly flat point cloud has $\sigma_3 \approx 0$ and its rotation about the normal is ambiguous.

**In ML/AI this is …** `np.linalg.svd` / `torch.linalg.svd`; the spectral norm $\sigma_1$ of a weight matrix, which bounds how much a layer can amplify a perturbation (Lipschitz constant, adversarial robustness, spectral normalisation); orthogonal weight initialisation (take $U$ from the SVD of a Gaussian matrix); whitening transforms; and the observation that the attention score matrix $W_Q W_K^\top$ in a transformer head has rank at most $d_{\text{head}}$ (Module 17).

## 5. Low-rank approximation

**Intuition.** The SVD writes $A$ as a sum of rank-one layers $\sigma_i u_i v_i^\top$, ordered from most to least important. Keeping the first $k$ layers and discarding the rest is the best possible compression of $A$ into $k$ "patterns" — and the discarded singular values tell you exactly how much you lost. In data, the small singular values are typically noise or fine detail; in weight matrices, they are the part a cheap update does not need.

**Definition.** The **truncated SVD** $A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^\top = U_k \Sigma_k V_k^\top$ has rank $k$. The **Eckart–Young–Mirsky theorem**: for every matrix $B$ of rank at most $k$,

$$\|A - B\|_F \ \ge\ \|A - A_k\|_F = \sqrt{\sum_{i>k} \sigma_i^2}, \qquad \|A - B\|_2 \ \ge\ \|A - A_k\|_2 = \sigma_{k+1} .$$

No rank-$k$ matrix is closer to $A$, in either norm. The **energy** (squared Frobenius norm) captured by $A_k$ is $\sum_{i\le k}\sigma_i^2 / \sum_i \sigma_i^2$. Storage drops from $mn$ numbers to $k(m + n + 1)$. The proof hinges on the fact that any rank-$k$ $B$ has a null space of dimension $\ge n-k$, which must intersect $\operatorname{span}\{v_1,\dots,v_{k+1}\}$ in some unit vector $w$, and then $\|A - B\|_2 \ge \|(A-B)w\| = \|Aw\| \ge \sigma_{k+1}$.

**Worked example.** Continue with $A = \begin{pmatrix} 3 & 0 \\ 4 & 5\end{pmatrix}$. Its best rank-one approximation is

$$A_1 = \sigma_1 u_1 v_1^\top = 3\sqrt5 \cdot \frac{1}{\sqrt{10}}\begin{pmatrix}1\\3\end{pmatrix}\frac{1}{\sqrt2}\begin{pmatrix}1 & 1\end{pmatrix} = \begin{pmatrix} 1.5 & 1.5 \\ 4.5 & 4.5\end{pmatrix},$$

with error $\|A - A_1\|_F = \sigma_2 = \sqrt5 \approx 2.236$ and energy captured $45/50 = 90\,\%$. For a $512\times512$ grey-scale image, keeping $k = 20$ layers stores $20 \cdot (512 + 512 + 1) = 20{,}500$ numbers instead of $262{,}144$ — 7.8 % — and typically reproduces the picture recognisably, because natural images have fast-decaying singular values.

**Real-life example (e-commerce recommendation).** The Netflix Prize ratings matrix had 480,189 users and 17,770 films but only 100,480,507 ratings — 1.2 % of the cells. A rank-50 factorisation $R \approx P Q^\top$ assigns each user a 50-dimensional taste vector $p_u$ and each film a 50-dimensional profile $q_i$, predicting $\hat r_{ui} = p_u^\top q_i$; that is about 25 million parameters standing in for 8.5 billion cells. The latent dimensions are learned, not labelled, but they typically line up with recognisable axes such as "art-house versus blockbuster" or "appeals to children". Because most entries are missing, the winning systems fitted the factors by gradient descent on the observed cells only — the mini-project in `exercises.ipynb` builds the SVD-with-imputation version. The same construction underlies latent semantic analysis, where a term-by-document count matrix is truncated to a few hundred "topics".

**In ML/AI this is …** `sklearn.decomposition.TruncatedSVD` (LSA), randomised SVD for very large matrices, denoising by truncation, and **LoRA** (Module 17): instead of updating a $d\times k$ weight matrix $W_0$, train a rank-$r$ update $W_0 + BA$ with $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$. For $d = k = 4096$ and $r = 8$ that is $8 \cdot 8192 = 65{,}536$ trainable numbers instead of $16.8$ million — 0.4 %. Compressing trained weight matrices by truncated SVD is also a standard model-compression step.

## 6. Principal component analysis

**Intuition.** A cloud of data points in $d$ dimensions is usually not spread evenly: it is elongated in some directions and almost flat in others. PCA rotates the coordinate system so the first axis points along the direction of greatest spread, the second along the greatest remaining spread perpendicular to the first, and so on. Keeping the first $k$ axes gives the best $k$-dimensional flat summary of the data, and the dropped axes measure what the summary misses.

**Definition.** Let $X \in \mathbb{R}^{n\times d}$ hold $n$ observations in rows. Centre it, $X_c = X - \mathbf{1}\mu^\top$ with $\mu$ the column means, and form the sample covariance $C = X_c^\top X_c / (n-1)$. The **principal components** are the orthonormal eigenvectors $w_1, \dots, w_d$ of $C$ ordered by eigenvalue $\lambda_1 \ge \dots \ge \lambda_d \ge 0$; $\lambda_i$ is the variance of the data along $w_i$ and $\lambda_i / \sum_j \lambda_j$ the **explained variance ratio**. The first component solves

$$\max_{\|w\| = 1}\ \operatorname{Var}(X_c w) = \max_{\|w\| = 1}\ w^\top C w ,$$

which by the Rayleigh-quotient fact of Section 3 is the top eigenvector. Through the SVD $X_c = U \Sigma V^\top$ the same objects appear without ever forming $C$: the components are the columns of $V$, $\lambda_i = \sigma_i^2/(n-1)$, and the **scores** (new coordinates) are $Z = X_c V_k = U_k \Sigma_k$. The rank-$k$ **reconstruction** $\hat X = Z V_k^\top + \mathbf{1}\mu^\top$ is exactly the Eckart–Young optimum for the centred data, with total squared error $(n-1)\sum_{i>k}\lambda_i$. Choosing $k$: a scree plot of $\lambda_i$, a cumulative-variance threshold (90–95 %), or the downstream task's validation score. If the features have different units, standardise each column first — otherwise the component simply follows whichever feature has the biggest numbers. Sign of each $w_i$ is arbitrary.

**Worked example.** Five points: $(3,1), (4,3), (5,2), (6,4), (7,5)$. Mean $\mu = (5, 3)$; centred rows $(-2,-2), (-1,0), (0,-1), (1,1), (2,2)$. Sums of products: $\sum x_1^2 = 10$, $\sum x_2^2 = 10$, $\sum x_1x_2 = 9$, so

$$C = \frac{1}{4}\begin{pmatrix} 10 & 9 \\ 9 & 10\end{pmatrix} = \begin{pmatrix} 2.5 & 2.25 \\ 2.25 & 2.5\end{pmatrix}, \qquad \lambda_1 = 4.75,\ \lambda_2 = 0.25,$$

with $w_1 = (1,1)/\sqrt2$ and $w_2 = (1,-1)/\sqrt2$. The first component explains $4.75/5 = 95\,\%$ of the variance. Scores $z_i = x_{c,i}^\top w_1$: $-2.83, -0.71, -0.71, 1.41, 2.83$; their variance is $4.75 = \lambda_1$, as it must be. Reconstructing the second point from one component: $\mu + z_2 w_1 = (5,3) + (-0.5, -0.5) = (4.5, 2.5)$ instead of $(4, 3)$, a residual of $(-0.5, 0.5)$ that lies exactly along $w_2$. The total squared residual over the five points is $1.0 = 4 \cdot \lambda_2$.

**Real-life example (healthcare and genomics).** A gene-expression study measures 20,000 genes in 100 tumour samples: a $100\times20{,}000$ matrix nobody can look at. PCA to two components routinely separates tumour subtypes that respond to different drugs, and the loadings of $w_1$ name the genes driving the separation. Two cautions the practitioners learned the hard way: the first component often tracks a technical artefact (sequencing batch or library size) rather than biology, and genes must be log-transformed and standardised, or a few very highly expressed genes dominate everything. Finance uses the identical tool on bond yields — the first three components of yield-curve changes are "level, slope and curvature" and typically explain more than 95 % of the variation — and engineers use it to compress hundreds of correlated sensor channels into a handful of health indicators.

**In ML/AI this is …** `sklearn.decomposition.PCA` (which centres and runs an SVD; its `explained_variance_ratio_` is $\lambda_i/\sum_j\lambda_j$), whitening of inputs, eigenfaces, visualising embedding spaces before applying t-SNE or UMAP, and a theorem worth remembering: a linear autoencoder with a $k$-unit bottleneck trained with squared error learns exactly the top-$k$ principal subspace (Baldi and Hornik, 1989) — PCA is the simplest representation-learning algorithm.

## 7. The pseudo-inverse and least squares via the SVD

**Intuition.** You cannot undo a stretch by zero. The pseudo-inverse undoes every stretch the matrix *does* perform (divide by each non-zero $\sigma_i$) and simply declines to act along the directions the matrix destroyed. The result is the most sensible "inverse" a rectangular or singular matrix can have, and it hands you least-squares solutions even when the normal equations are singular.

**Definition.** With $A = U\Sigma V^\top$, the **Moore–Penrose pseudo-inverse** is

$$A^{+} = V \Sigma^{+} U^\top, \qquad \Sigma^{+}_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > 0 \\ 0 & \sigma_i = 0 \end{cases},$$

an $n\times m$ matrix. If $A$ has full column rank, $A^+ = (A^\top A)^{-1}A^\top$ — the familiar normal-equations solution; if $A$ is square and invertible, $A^+ = A^{-1}$. In general $x^{+} = A^{+} b$ is the least-squares solution of **minimum norm**: among all $x$ minimising $\|Ax - b\|$, it has the smallest $\|x\|$. In coordinates,

$$x^{+} = \sum_{\sigma_i > 0} \frac{u_i^\top b}{\sigma_i}\, v_i .$$

Each term projects $b$ onto a left singular direction, divides by the stretch, and places the result along the matching right singular direction; components of $b$ outside $\operatorname{col}(A)$ are dropped (they are the residual), and nothing is placed in $\operatorname{null}(A)$ (that is what makes the norm minimal). **Ridge regression** (Module 13) changes the filter to $\sigma_i/(\sigma_i^2 + \lambda)$: directions with $\sigma_i^2 \gg \lambda$ pass almost unchanged, directions with $\sigma_i^2 \ll \lambda$ are shrunk towards zero — a smooth version of the hard cut-off that `np.linalg.pinv(A, rcond=...)` applies.

**Worked example.** $A = \begin{pmatrix} 1 & 1 \\ 1 & 1\end{pmatrix}$ has rank one: $\sigma_1 = 2$, $u_1 = v_1 = (1,1)/\sqrt2$, $\sigma_2 = 0$. So

$$A^{+} = v_1 \frac{1}{\sigma_1} u_1^\top = \frac12 \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5\end{pmatrix} = \begin{pmatrix} 0.25 & 0.25 \\ 0.25 & 0.25\end{pmatrix}.$$

For $b = (1, 1)$ (in the column space) $x^+ = (0.5, 0.5)$: every $x$ with $x_1 + x_2 = 1$ solves $Ax = b$ exactly, and $(0.5, 0.5)$ is the shortest of them. For $b = (1, 0)$ (not in the column space) $x^+ = (0.25, 0.25)$ and $Ax^+ = (0.5, 0.5)$, the orthogonal projection of $b$ onto the line spanned by $(1,1)$ — the least-squares fit. The normal equations $A^\top A x = A^\top b$ are singular here and cannot be solved by inversion at all.

**Real-life example (manufacturing calibration).** A production line regresses product hardness on sensor readings, and two of the sensors are thermocouples that record the same temperature, one in °C and one in °F. The two columns are exactly collinear ($F = 1.8\,C + 32$), $X^\top X$ is singular, and `np.linalg.inv` fails or returns garbage. The pseudo-inverse returns the minimum-norm coefficients, which split the temperature effect evenly between the two sensors — a stable answer that predicts correctly. The lesson for the analyst is the diagnostic one: a zero (or tiny) singular value in $X$ is the fingerprint of redundant features.

**In ML/AI this is …** `np.linalg.lstsq` and `np.linalg.pinv`, both SVD-based, and therefore `sklearn.linear_model.LinearRegression`, which returns the minimum-norm solution whenever there are more features than samples. That minimum-norm interpolator is the object studied in "double descent" (Module 15), and the ridge filter factors above are the cleanest way to see what regularisation does to a linear model (Module 13).

## 8. Condition number and numerical stability

**Intuition.** A matrix that stretches one direction by 10,000 and another by 1 mixes wildly different scales. When you solve $Ax = b$, any error in $b$ along the weak direction gets multiplied by 10,000 relative to the strong direction. The condition number measures this worst-case amplification; a large value means that rounding errors, measurement noise and optimiser steps all behave badly, and that the fix is usually to rescale the problem.

**Definition.** The (2-norm) **condition number** of $A$ is

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} \ \ (\ge 1), \qquad \kappa(A) = \frac{|\lambda_{\max}|}{|\lambda_{\min}|}\ \text{ for symmetric } A .$$

If $Ax = b$ and $A(x + \delta x) = b + \delta b$, then

$$\frac{\|\delta x\|}{\|x\|} \ \le\ \kappa(A)\, \frac{\|\delta b\|}{\|b\|} ,$$

with equality for the worst-case directions ($b$ along $u_1$, $\delta b$ along $u_{\min}$). Rule of thumb: you lose about $\log_{10}\kappa$ significant digits. Two consequences matter constantly in ML. First, $\kappa(A^\top A) = \kappa(A)^2$, so forming the normal equations squares the damage — solve least squares by QR or SVD (`lstsq`), not by `inv(X.T @ X)`. Second, gradient descent on a quadratic $\tfrac12 x^\top A x - b^\top x$ with the best fixed step size shrinks the error by a factor $(\kappa - 1)/(\kappa + 1)$ per step (Module 05): for $\kappa = 100$ that is $0.98$, about 345 steps to gain three digits; for $\kappa = 10^4$ it is $0.9998$, tens of thousands of steps. **Feature scaling** attacks $\kappa$ directly: standardising columns makes the singular values comparable, and centring removes the near-collinearity between a constant column and a slowly varying feature.

**Worked example.** $A = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001\end{pmatrix}$ has singular values $2.00005$ and $0.00005$, so $\kappa \approx 40{,}002$. With $b = (2, 2.0001)$ the solution is $x = (1, 1)$. Change $b$ by one part in $10^{-4}$ to $b' = (2, 2.0002)$ and the solution becomes $x' = (0, 2)$: a relative input change of $3.5\times10^{-5}$ produced a relative output change of $1.0$, an amplification of about $28{,}000$ — below the bound $\kappa$, because this perturbation was not exactly along the worst direction, but devastating all the same. A correlation of 0.9 between two standardised features already gives the Gram matrix $\begin{pmatrix} 1 & 0.9 \\ 0.9 & 1\end{pmatrix}$ a condition number of $19$.

**Real-life example (climate).** Fitting a quadratic trend to annual temperature anomalies for 1990–2020 with the design matrix columns $(1, t, t^2)$ and $t$ the raw calendar year gives $\kappa \approx 2\times10^{11}$: the columns $1.99\times10^3$ and $3.96\times10^6$ are nearly parallel and single precision cannot represent the answer at all. Replacing $t$ by $t - 2005$ (centring) gives $\kappa \approx 161$ — the same model, the same fitted curve, and nine orders of magnitude less error amplification. Every polynomial-features pipeline should do this, and `sklearn.preprocessing.StandardScaler` does it for you.

**In ML/AI this is …** `StandardScaler` before linear models, SVMs and neural networks; the `rcond` argument of `np.linalg.lstsq`/`pinv`, which discards singular values below $\text{rcond}\cdot\sigma_{\max}$; and the condition number of the Hessian, which is why momentum, Adam and second-order methods exist (Module 05) and part of why batch normalisation helps (Module 16).

## Common pitfalls and misconceptions

1. **"Every matrix has eigenvalues, and they are real."** Only square matrices have eigenvalues, and real matrices can have complex ones (any rotation). → For a non-square matrix use the SVD; for a symmetric one use `np.linalg.eigh`, which guarantees real, sorted output.
2. **"The eigenvectors returned by `eig` are orthogonal."** Only for symmetric (more generally, normal) matrices. For $A = \begin{pmatrix} 4 & 1 \\ 2 & 3\end{pmatrix}$ they are $(1,1)$ and $(1,-2)$, whose lines meet at about 72°, not 90°. → Check $Q^\top Q = I$ before relying on it; use `eigh` when the matrix is symmetric.
3. **Forgetting to centre before an SVD-based PCA.** Without centring the first singular vector points roughly at the mean of the data, not along its largest variance. → Subtract the column means (scikit-learn does this for you; your own NumPy code must).
4. **Running PCA on unscaled features.** A feature measured in grams next to one measured in kilometres owns the first component by sheer magnitude. → Standardise columns when units differ; PCA is not scale-invariant.
5. **"The top principal components are the best features for prediction."** PCA is unsupervised; it finds directions of large *variance*, and the direction that separates your classes may be a low-variance one. → Validate downstream performance against $k$, or use a supervised projection (LDA, PLS).
6. **Reading principal components as features.** A component is a direction — a weighted combination of all features — and its sign is arbitrary. → Look at the loadings to interpret, and never compare the sign of a component across runs or libraries.
7. **Computing an SVD or PCA through $A^\top A$.** It squares the condition number, so small singular values drown in rounding noise (a $\sigma$ of $10^{-5}$ becomes an eigenvalue of $10^{-10}$, below single-precision resolution). → Call `np.linalg.svd` on $A$ itself.
8. **Inverting $X^\top X$ for least squares.** Slow, unstable, and undefined for collinear features. → Use `np.linalg.lstsq` (SVD) or QR, which also return the minimum-norm solution when the system is rank-deficient.

## Real-world case studies

### Case study 1 — Eigenfaces: face recognition with PCA (computer vision)

A classic face database holds 400 grey-scale photographs (40 people, 10 each) of $92\times112$ pixels. Each image is a vector in $\mathbb{R}^{10{,}304}$, so the data matrix is $400\times10{,}304$ and the covariance matrix would be $10{,}304\times10{,}304$ — 106 million entries, far too many for the hardware of 1991 when Turk and Pentland published the method. The trick: with only 400 samples the covariance has rank at most 399, and the non-zero eigenvalues of $X_c^\top X_c$ equal those of the tiny $400\times400$ matrix $X_c X_c^\top$. If $X_c X_c^\top a = \lambda a$ then $X_c^\top a$ is an eigenvector of $X_c^\top X_c$ with the same $\lambda$ — a one-line consequence of Section 4. Today you simply call the thin SVD of the $400\times10{,}304$ matrix.

The resulting eigenvectors, reshaped to $92\times112$, are ghostly "eigenfaces": the first captures overall lighting, the next few capture face width, hair line and glasses. Keeping $k = 50$ components typically retains around 90 % of the variance; each face is then described by 50 coordinates instead of 10,304, a 206-fold compression, and storing the model costs $50\times10{,}304 + 400\times50 \approx 535{,}000$ numbers instead of 4.1 million. Recognition is nearest-neighbour search in the 50-dimensional score space. The limitations are instructive: because PCA follows variance, the leading components track illumination and pose rather than identity, so eigenfaces fail under changed lighting — one of the motivations for the supervised projections (Fisherfaces) and, eventually, learned convolutional features that replaced them.

### Case study 2 — Latent factors for recommendation (e-commerce)

The Netflix Prize data (100,480,507 ratings, 480,189 users, 17,770 films) is 98.8 % empty, so a plain SVD is not even defined: there is no matrix to decompose until the gaps are filled. Two routes exist. The *imputation* route fills missing cells with a baseline (global mean plus user and item offsets), takes a rank-$k$ truncated SVD, writes the reconstruction back into the missing cells, and repeats — each pass is an Eckart–Young projection, and the iterations converge to a low-rank completion. The *direct* route, used by the winning teams, minimises $\sum_{(u,i)\ \text{observed}} (r_{ui} - \mu - b_u - b_i - p_u^\top q_i)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2)$ by stochastic gradient descent over the observed cells only; the regulariser is the ridge shrinkage of Section 7, and it is essential because a user with three ratings cannot support a 50-dimensional taste vector.

The arithmetic explains why the method works: a rank-50 model has $(480{,}189 + 17{,}770)\times50 \approx 25$ million parameters, four times fewer than the 100 million observed ratings, so it must generalise rather than memorise. Netflix's own system scored a root-mean-square error of about 0.95 stars on held-out ratings, and the competition's goal was a 10 % improvement, to about 0.86; latent-factor models were the single largest contributor. The mini-project in `exercises.ipynb` reproduces the imputation route on a synthetic ratings matrix and shows the characteristic U-shaped curve of held-out RMSE against $k$: too few factors underfit, too many fit the noise in the imputed cells.

### Case study 3 — Empirical orthogonal functions in climate science

Climate scientists have run PCA on gridded fields since the 1950s under the name *empirical orthogonal function* (EOF) analysis. Take monthly sea-surface-temperature anomalies over the tropical Pacific on a $2°$ grid: roughly $30$ latitude bands $\times$ $85$ longitude bands $= 2{,}550$ grid cells, observed for 40 years $= 480$ months. The data matrix is $480\times2{,}550$ (time by space); its right singular vectors are spatial patterns (the EOFs), the left singular vectors are their time series (the principal components), and $\sigma_i^2$ is the variance each pattern accounts for. The leading EOF of this region is the El Niño–Southern Oscillation pattern — warm anomalies along the equator fanning out from South America — and its time series is a standard El Niño index.

Three practical details map directly onto this module. Grid cells near the poles cover less area than those at the equator, so each column is weighted by $\sqrt{\cos(\text{latitude})}$ before the SVD — a change of the inner product. The thin SVD of the $480\times2{,}550$ matrix costs on the order of $480^2\times2{,}550 \approx 6\times10^8$ flops, trivial today, whereas the $2{,}550\times2{,}550$ covariance route was the bottleneck for decades. And the sign and ordering of EOFs are arbitrary: two eigenvalues that are close together (within sampling error) can swap order or mix between datasets, so researchers test the separation of neighbouring eigenvalues before interpreting a pattern — Section 8's warning about nearly equal singular values in a different costume.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Eigenvalues of the Hessian | Gradient descent, Newton's method (Module 05) | $\lambda_{\max}$ caps the learning rate; $\kappa = \lambda_{\max}/\lambda_{\min}$ sets the convergence speed |
| Dominant eigenvector / power iteration | PageRank, spectral normalisation, Lanczos solvers | The largest eigen- or singular value is often all you need, and it is cheap |
| Stationary distributions | MCMC (Module 18), value iteration (Module 19) | Markov chains converge to the eigenvector with eigenvalue 1 |
| Spectral theorem, PSD | Covariance matrices (Module 09), kernel matrices, Hessians | Guarantees real eigenvalues and orthogonal axes; PSD validates kernels and certifies minima |
| SVD | `lstsq`, `pinv`, PCA, orthogonal init, spectral norm | The universal decomposition; exists for every matrix |
| Eckart–Young / truncated SVD | LSA, model compression, LoRA (Module 17) | Best rank-$k$ approximation with a known error |
| PCA | Preprocessing, visualisation, eigenfaces, linear autoencoders | Optimal linear dimensionality reduction |
| Pseudo-inverse | `LinearRegression`, minimum-norm interpolation (Modules 13, 15) | Least squares that survive rank deficiency |
| Ridge filter factors $\sigma^2/(\sigma^2+\lambda)$ | Ridge regression (Module 13), weight decay | Regularisation shrinks the directions the data barely constrain |
| Condition number | Feature scaling, batch norm (Module 16), optimiser design (Module 05) | Governs error amplification and optimisation speed |

## Cheat sheet

| Fact | Formula / statement |
|---|---|
| Eigen-equation | $Av = \lambda v$, $v \ne 0$; eigenvalues solve $\det(A - \lambda I) = 0$ |
| $2\times2$ characteristic polynomial | $\lambda^2 - \operatorname{tr}(A)\lambda + \det(A) = 0$; $\sum\lambda_i = \operatorname{tr}A$, $\prod\lambda_i = \det A$ |
| Diagonalisation | $A = PDP^{-1}$, $A^k = PD^kP^{-1}$, $A^k x = \sum_i c_i\lambda_i^k v_i$ |
| Power iteration | $x_{k+1} = Ax_k/\lVert Ax_k\rVert$; error $\propto \lvert\lambda_2/\lambda_1\rvert^k$; $\lambda_1 \approx x_k^\top A x_k$ |
| Google matrix | $G = dM + \frac{1-d}{n}\mathbf{1}\mathbf{1}^\top$; PageRank = stationary vector; $\lvert\lambda_2\rvert \le d$ |
| Spectral theorem | $A = A^\top \Rightarrow A = Q\Lambda Q^\top$, $Q^\top Q = I$, $\Lambda$ real |
| Quadratic form | $x^\top A x = \sum_i \lambda_i y_i^2$ with $y = Q^\top x$; PSD $\Leftrightarrow$ all $\lambda_i \ge 0$ |
| Rayleigh quotient | for symmetric $A$: $\lambda_{\min} \le \dfrac{x^\top A x}{x^\top x} \le \lambda_{\max}$ |
| SVD | $A = U\Sigma V^\top = \sum_i \sigma_i u_i v_i^\top$; $Av_i = \sigma_i u_i$; $\sigma_i^2 = \operatorname{eig}(A^\top A)$ |
| Norms from the SVD | $\lVert A\rVert_2 = \sigma_1$, $\lVert A\rVert_F^2 = \sum_i\sigma_i^2$, $\operatorname{rank} A = \#\{\sigma_i > 0\}$ |
| Eckart–Young | $\min_{\operatorname{rank} B \le k}\lVert A - B\rVert_F = \sqrt{\sum_{i>k}\sigma_i^2}$, attained by $A_k = U_k\Sigma_kV_k^\top$ |
| PCA | Centre; $C = X_c^\top X_c/(n-1)$; components = eigenvectors of $C$ = columns of $V$; $\lambda_i = \sigma_i^2/(n-1)$ |
| PCA scores and reconstruction | $Z = X_cV_k$; $\hat X = ZV_k^\top + \mathbf{1}\mu^\top$; error $= (n-1)\sum_{i>k}\lambda_i$ |
| Pseudo-inverse | $A^+ = V\Sigma^+U^\top$; $x^+ = A^+b = \sum_{\sigma_i>0}\dfrac{u_i^\top b}{\sigma_i}v_i$ (min-norm least squares) |
| Ridge filter | $x_\lambda = \sum_i \dfrac{\sigma_i}{\sigma_i^2 + \lambda}(u_i^\top b)\,v_i$ |
| Condition number | $\kappa = \sigma_{\max}/\sigma_{\min}$; $\dfrac{\lVert\delta x\rVert}{\lVert x\rVert} \le \kappa\dfrac{\lVert\delta b\rVert}{\lVert b\rVert}$; $\kappa(A^\top A) = \kappa(A)^2$ |
| GD on a quadratic | Error factor per step $(\kappa-1)/(\kappa+1)$ with the optimal fixed step |

## Self-assessment

1. **A $2\times2$ matrix has trace 6 and determinant 8. What are its eigenvalues, and is it diagonalisable?**
   <details><summary>Answer</summary>

   $\lambda^2 - 6\lambda + 8 = (\lambda-2)(\lambda-4)$, so $\lambda = 2, 4$. Distinct eigenvalues always give linearly independent eigenvectors, hence the matrix is diagonalisable (though the eigenvectors need not be orthogonal unless the matrix is symmetric).

   </details>

2. **Why does power iteration on a column-stochastic matrix converge, and what controls the speed?**
   <details><summary>Answer</summary>

   A column-stochastic matrix has eigenvalue 1 (since $\mathbf{1}^\top T = \mathbf{1}^\top$) and all other eigenvalues have modulus at most 1. If the chain is irreducible and aperiodic the eigenvalue 1 is simple and strictly dominant, so $T^k x_0 \to \pi$. The error decays like $\lvert\lambda_2\rvert^k$; for the Google matrix $\lvert\lambda_2\rvert \le d$, which is why damping guarantees fast convergence.

   </details>

3. **Is $\begin{pmatrix} 1 & 2 \\ 2 & 1\end{pmatrix}$ positive semidefinite?**
   <details><summary>Answer</summary>

   No. Its eigenvalues solve $\lambda^2 - 2\lambda - 3 = 0$, giving $3$ and $-1$. Equivalently $x = (1, -1)$ gives $x^\top A x = 1 - 4 + 1 = -2 < 0$. A positive diagonal is not enough; the off-diagonal entries must be dominated.

   </details>

4. **How are the singular values of $A$ related to the eigenvalues of $A^\top A$, and why are singular values never negative or complex?**
   <details><summary>Answer</summary>

   $\sigma_i = \sqrt{\lambda_i(A^\top A)}$. $A^\top A$ is symmetric and positive semidefinite ($x^\top A^\top Ax = \lVert Ax\rVert^2 \ge 0$), so its eigenvalues are real and non-negative, and their square roots are real and non-negative too.

   </details>

5. **A $1000\times1000$ matrix has singular values $100, 50, 10, 1, 1, 1, \dots$ (the rest are 1). What relative Frobenius error does the rank-3 approximation make?**
   <details><summary>Answer</summary>

   Dropped energy: $997 \times 1^2 = 997$. Total energy: $10{,}000 + 2{,}500 + 100 + 997 = 13{,}597$. Relative Frobenius error $= \sqrt{997/13{,}597} \approx 0.27$. Even though each dropped value is tiny, there are many of them — the long tail matters.

   </details>

6. **Your colleague runs PCA on a dataset with columns "age in years" and "annual income in euros" and reports that the first component explains 99.99 % of the variance. What happened?**
   <details><summary>Answer</summary>

   Income varies by tens of thousands, age by tens; the covariance matrix is dominated by the income variance, so the first component is essentially the income axis. The 99.99 % says nothing about structure in the data. Standardise each column (zero mean, unit variance) and repeat; on the correlation matrix the split will be far more balanced.

   </details>

7. **Explain in one sentence each why $A^+b$ (a) minimises $\lVert Ax - b\rVert$ and (b) has minimum norm among the minimisers.**
   <details><summary>Answer</summary>

   (a) $AA^+b = U_rU_r^\top b$ is the orthogonal projection of $b$ onto $\operatorname{col}(A)$, which is the closest point in the column space. (b) $A^+b$ lies in $\operatorname{row}(A) = \operatorname{span}\{v_1,\dots,v_r\}$, orthogonal to $\operatorname{null}(A)$; any other minimiser is $A^+b + z$ with $z \in \operatorname{null}(A)$, and by Pythagoras $\lVert A^+b + z\rVert^2 = \lVert A^+b\rVert^2 + \lVert z\rVert^2$.

   </details>

8. **Ridge regression replaces $1/\sigma_i$ by $\sigma_i/(\sigma_i^2+\lambda)$. Which directions does it change most, and what happens as $\lambda \to 0$ and $\lambda \to \infty$?**
   <details><summary>Answer</summary>

   For $\sigma_i^2 \gg \lambda$ the factor is $\approx 1/\sigma_i$ (unchanged); for $\sigma_i^2 \ll \lambda$ it is $\approx \sigma_i/\lambda \to 0$, so the small-singular-value directions — the ones that amplify noise most — are shrunk hardest. As $\lambda \to 0$ ridge becomes the pseudo-inverse solution; as $\lambda \to \infty$ all coefficients go to zero.

   </details>

9. **Why does gradient descent slow down on an ill-conditioned problem, and how does feature scaling help?**
   <details><summary>Answer</summary>

   The step size must be below $2/\lambda_{\max}$ to avoid diverging along the stiff direction, but progress along the flat direction is then proportional to $\lambda_{\min}/\lambda_{\max} = 1/\kappa$ per step. Standardising features equalises the scales of the columns of $X$, bringing the eigenvalues of $X^\top X$ closer together and shrinking $\kappa$, so one step size works well for every direction.

   </details>

## Further reading

- Gilbert Strang, *Introduction to Linear Algebra* (5th ed., 2016), Chapters 6–7, and his MIT 18.06 lectures (free on MIT OpenCourseWare) — the clearest treatment of eigenvalues and the SVD.
- Gilbert Strang, *Linear Algebra and Learning from Data* (2019) — the same author aiming squarely at ML: low rank, PCA, condition numbers.
- Lloyd Trefethen and David Bau, *Numerical Linear Algebra* (SIAM, 1997), Lectures 4–5 and 12 — the SVD, conditioning and stability done properly.
- Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020; free PDF at mml-book.com), Chapter 4 (matrix decompositions) and Chapter 10 (PCA, including the autoencoder view).
- Jonathon Shlens, "A Tutorial on Principal Component Analysis" (2014, arXiv:1404.1100) — short, concrete, and careful about assumptions.
- Lawrence Page, Sergey Brin, Rajeev Motwani and Terry Winograd, "The PageRank Citation Ranking: Bringing Order to the Web" (Stanford technical report, 1999).
- Yehuda Koren, Robert Bell and Chris Volinsky, "Matrix Factorization Techniques for Recommender Systems", *IEEE Computer*, 2009 — the latent-factor model of Case study 2.
- Matthew Turk and Alex Pentland, "Eigenfaces for Recognition", *Journal of Cognitive Neuroscience*, 1991.
- Edward J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021, arXiv:2106.09685) — low-rank approximation as a fine-tuning method.
- 3Blue1Brown, *Essence of Linear Algebra*, the episode on eigenvectors and eigenvalues (free video) — for the geometric picture.

## What's in the notebooks

**`lesson.ipynb`** walks through every section with code you can edit: a grid-transform picture showing which directions a matrix only stretches; power iteration from scratch compared with `np.linalg.eig`, with its convergence rate measured; a two-state Markov chain and a six-page PageRank with damping; the spectral decomposition of a symmetric matrix, its ellipse of constant quadratic form, and a PSD check; the SVD of a low-rank-plus-noise matrix with a scree plot and the rotate–stretch–rotate picture; compression of a synthetic 64×64 image at ranks 1, 2, 5, 10 and 20 with the Eckart–Young error curve; PCA from scratch on the Iris data matched against `sklearn.decomposition.PCA`, plus a demonstration of what unscaled features do; the pseudo-inverse built from the SVD, the minimum-norm solution, and ridge filter factors; and an experiment relating condition number to error amplification and to the number of gradient-descent steps.

**`exercises.ipynb`** has twelve exercises with instant self-checks: eigenvalues of a 2×2 by the characteristic polynomial, the Rayleigh quotient, spectral-theorem and PSD checks, power iteration, PageRank with damping on a given graph (hidden answer), truncated-SVD reconstruction error, PCA from scratch versus scikit-learn, the pseudo-inverse via SVD and the minimum-norm property, condition numbers and worst-case error amplification, two conceptual questions (unscaled PCA, and computing an SVD through $A^\top A$), and a mini-project: a latent-factor recommender built by iterated truncated SVD on a synthetic ratings matrix with missing entries, evaluated by held-out RMSE against the rank $k$. `solutions.ipynb` contains reference implementations and explanations.
