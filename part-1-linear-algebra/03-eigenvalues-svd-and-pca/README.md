# Module 03 — Eigenvalues, SVD and PCA

> After this module you can take any matrix apart into its "natural axes", use that to compress data, rank web pages, run PCA from scratch, solve rank-deficient least-squares problems, and explain why badly scaled features make optimisation slow.

## Why this matters for ML and AI

Module 02 treated a matrix as a machine that moves vectors around. This module asks a sharper question: *along which directions does that machine act simply?* The answer — eigenvectors for square matrices, singular vectors for any matrix — is the single most reused idea in applied linear algebra. Once you can read a matrix through its spectrum you can see at a glance whether a dynamical system blows up or settles down, how much of a dataset lives in a handful of directions, and how badly rounding errors will be amplified when you solve a system.

The applications are not academic. Google's original ranking of the web was the dominant eigenvector of a Markov chain with one state per web page, computed by the power iteration you will implement here. The Netflix Prize was won by teams who factorised a 480,189 × 17,770 ratings matrix into a few dozen latent factors — truncated SVD in spirit. `sklearn.decomposition.PCA` computes the SVD of the centred data matrix, up to signs and rounding (depending on the data's shape it may get there through the equivalent eigendecomposition of the covariance matrix, or settle for a close randomised approximation when only a few components of a large matrix are wanted). LoRA, the dominant method for fine-tuning large language models, is a bet that weight updates are low-rank. And every time an optimiser crawls across a long narrow valley, the villain is the condition number of a Hessian — the ratio of its largest to smallest eigenvalue.

The module builds up in a straight line: eigenvectors (what a square matrix preserves) → diagonalisation (why matrix powers are easy) → symmetric matrices (where everything is real and orthogonal) → the SVD (the same story for *every* matrix) → low-rank approximation and PCA (SVD as a compression tool) → the pseudo-inverse (SVD as a solver) → the condition number (SVD as a diagnostic).

## Learning objectives

By the end of this module you will be able to:

- Compute eigenvalues and eigenvectors of a 2×2 matrix by hand, and explain what they mean geometrically.
- Diagonalise a matrix, use $A^k = PD^kP^{-1}$ to predict long-run behaviour, and implement power iteration (including PageRank with damping).
- Decide from algebraic and geometric multiplicities whether a matrix is diagonalisable, and read growth, decay, sign flips and rotation off its eigenvalues.
- State the spectral theorem and use it to classify quadratic forms as positive (semi)definite.
- Compute the SVD of a small matrix by hand, relate it to the eigendecomposition of $A^\top A$, and read off rank and the four fundamental subspaces.
- Apply the Eckart–Young theorem to build the best rank-$k$ approximation and quantify the error it leaves.
- Implement PCA from scratch (centring, covariance, eigenvectors or SVD, explained variance, projection, reconstruction) and match scikit-learn.
- Explain why the top eigenvector of the covariance matrix both maximises the variance of the projected data and minimises the squared reconstruction error.
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

so the two eigenvalues always add up to the trace and multiply to the determinant. All eigenvectors for one $\lambda$, together with $0$, form a subspace (the *eigenspace*): the null space of $A - \lambda I$. The polynomial may have complex roots: a rotation by 90°, $R = \begin{pmatrix} 0 & -1 \\ 1 & 0\end{pmatrix}$, gives $\lambda^2 + 1 = 0$, so $\lambda = \pm i$ — no real direction keeps its line under this rotation (Figure 1(b); only the rotations by 0° and 180° have real eigenvalues, $1$ and $-1$), and complex eigenvalues always signal a rotational component. A matrix can also have too few eigenvectors: the shear $\begin{pmatrix} 1 & 1 \\ 0 & 1\end{pmatrix}$ has the double eigenvalue $1$ but only one eigenvector direction, $(1, 0)$ (Figure 1(c)); Section 2 explains why that matters.

*Finding the eigenvectors.* Once $\lambda$ is known, its eigenvectors are the non-zero solutions of $(A - \lambda I)v = 0$, found by elimination like any null space ([Module 02](../02-matrices-and-linear-maps/README.md), Section 5). A $2\times 2$ matrix has a shortcut: $A - \lambda I$ is singular, so its two rows are proportional, and one non-zero row $(a, b)$ says everything — it reads $a x_1 + b x_2 = 0$, which $v = (b, -a)$ solves. (If both rows are zero, $A = \lambda I$ and every vector is an eigenvector.) Any non-zero multiple of an eigenvector is again one, so what the equation really determines is a line through the origin; software returns a representative of length 1, with whatever sign its algorithm happens to produce.

*Reading an eigenvalue.* Along an eigenvector, applying $A$ $k$ times multiplies by $\lambda^k$, so each eigenvalue is a growth factor per step: $|\lambda| > 1$ grows, $|\lambda| < 1$ decays, $\lambda < 0$ also flips the sign at every step, and $\lambda = 1$ marks a direction that stays put. A complex pair $r e^{\pm i\theta}$ belongs to a plane in which $A$ turns vectors by $\theta$ and scales them by $r$ per step (literally so for a scaled rotation such as $R$, with $r = 1$ and $\theta = 90°$; for other matrices the turning happens in skewed coordinates), so $|\lambda|$ is the growth rate even when $\lambda$ is complex. This module's examples cover the range: $5$ and $2$ in the worked example below (both directions grow, one faster), $1.618$ and $-0.618$ for the Leslie matrix (growth, plus an oscillation in the age mix that dies out) and $1$ and $0.4$ for the Markov chain of Section 2 (a steady state, and the rate at which the start is forgotten). Whether a repeatedly applied map blows up or settles down is decided in the long run by the largest $|\lambda|$, the *spectral radius*; the largest stretch in a single step is the largest singular value of Section 4, which can be bigger.

*Computing eigenpairs in practice.* `w, V = np.linalg.eig(A)` returns the eigenvalues in `w` (in no guaranteed order) and unit-length eigenvectors as the *columns* of `V` — complex arrays as soon as one eigenvalue is complex. Check the output with `np.allclose(A @ V, V * w)`, which tests $Av_i = \lambda_i v_i$ for every column at once, and find the dominant pair with `np.argmax(np.abs(w))`. For the worked example below NumPy returns $w = (5, 2)$ with the columns $(0.707, 0.707) = v_1/\sqrt2$ and $(-0.447, 0.894) = -v_2/\sqrt5$, the second with its sign flipped. For symmetric matrices use `np.linalg.eigh` (real eigenvalues in ascending order, orthonormal columns; Section 3). And `eig` does not warn about missing eigenvectors: for the shear it returns two columns that both lie along $(1, 0)$, so `V` has a condition number of about $10^{16}$ — the numerical signature of a matrix that cannot be diagonalised.

**Worked example.** Take $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$. The trace is $7$ and the determinant is $12 - 2 = 10$, so

$$\lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2) = 0 \quad\Rightarrow\quad \lambda_1 = 5,\ \lambda_2 = 2.$$

For $\lambda_1 = 5$: $A - 5I = \begin{pmatrix} -1 & 1 \\ 2 & -2\end{pmatrix}$, whose first row says $-x_1 + x_2 = 0$ (the second row is $-2$ times the first), so its null space is spanned by $v_1 = (1, 1)$. Check: $A v_1 = (5, 5) = 5 v_1$. For $\lambda_2 = 2$: $A - 2I = \begin{pmatrix} 2 & 1 \\ 2 & 1\end{pmatrix}$, whose rows both say $2x_1 + x_2 = 0$, so $v_2 = (1, -2)$. Check: $A v_2 = (4 - 2,\ 2 - 6) = (2, -4) = 2 v_2$. Any vector $x = a v_1 + b v_2$ is mapped to $A x = 5a\,v_1 + 2b\,v_2$: the component along $v_1$ grows two and a half times faster than the component along $v_2$. Figure 1(a) draws the two eigen-lines: $v_1$ and $v_2$ stay on them, while a vector off them, such as $e_1 = (1, 0)$, is turned. Note that $v_1$ and $v_2$ are *not* orthogonal — their lines meet at $71.6°$ — and orthogonal eigenvectors are guaranteed only for symmetric matrices (more generally, normal ones; Section 3).

![Three panels. Left: the matrix with rows 4, 1 and 2, 3. Its eigen-lines through the origin, along (1, 1) and (1, -2), carry unit vectors that are stretched five times and twice along their own lines, while e1 = (1, 0) is turned to (4, 2); an arc marks the 71.6 degree angle between the lines. Middle: a rotation by 90 degrees turns three grey unit vectors into red ones a quarter turn further on, so no direction keeps its line. Right: a shear turns the unit square into a parallelogram; e1 stays on the horizontal line, while (0, 1) and (0, -1) slide sideways to (1, 1) and (-1, -1).](figures/eigen-directions.svg)

**Figure 1.** (a) The worked example: the unit eigenvectors $\hat v_1 = v_1/\sqrt2$ and $\hat v_2 = v_2/\sqrt5$ stay on their lines and are only stretched, by $5$ and by $2$, whereas $e_1 = (1, 0)$ is turned to $Ae_1 = (4, 2)$, at $26.6°$. The eigen-lines meet at $71.6°$, not $90°$ (pitfall 2). (b) The rotation $R$ by $90°$ turns every unit vector (grey) a quarter turn (red): no real direction keeps its line, matching $\lambda = \pm i$. (c) The shear $\begin{pmatrix} 1 & 1 \\ 0 & 1\end{pmatrix}$ fixes $e_1$, its only eigen-direction, while every vector off the horizontal axis slides along it: $(0, \pm 1) \mapsto \pm(1, 1)$. The eigenvalue $1$ is double, yet there is one eigen-direction, not two.

**Real-life example (population biology).** A Leslie matrix tracks an age-structured population. Suppose juveniles produce on average 1 offspring per season and adults 2, and half of the juveniles survive to adulthood: with state $(\text{juveniles}, \text{adults})$,

$$L = \begin{pmatrix} 1 & 2 \\ 0.5 & 0 \end{pmatrix}, \qquad \lambda^2 - \lambda - 1 = 0 \ \Rightarrow\ \lambda_1 = \tfrac{1+\sqrt 5}{2} \approx 1.618,\ \lambda_2 \approx -0.618 .$$

The dominant eigenvalue says the population grows by 61.8 % per season in the long run (the golden ratio, because this is the Fibonacci recurrence in disguise), and its eigenvector $(0.764, 0.236)$ is the *stable age structure*: whatever the starting mix, after a dozen seasons about 76 % of the population are juveniles. The negative eigenvalue shapes the approach: starting from 100 adults and no juveniles, the juvenile share runs $0, 1, 0.67, 0.80, 0.75, 0.77, 0.76, \dots$, overshooting and undershooting $0.764$ with a gap that changes sign every season and shrinks by about $|\lambda_2/\lambda_1| = 0.382$, while the totals $100, 200, 300, 500, 800, 1300, \dots$ are 100 times the Fibonacci numbers. The same mathematics gives the resonance frequencies of a bridge (eigenvalues of its stiffness–mass system) and the steady states of chemical reaction networks.

**In ML/AI this is …** the eigenvalues of the loss Hessian, which set the largest safe learning rate for gradient descent ($\eta < 2/\lambda_{\max}$ on a quadratic, Module 05); the eigenvalues of a recurrent weight matrix $W_{hh}$, whose repeated application makes gradients grow or shrink like $\lambda^t$ (exploding and vanishing gradients, Module 16); and the eigenvectors of a graph Laplacian used in spectral clustering. In NumPy: `np.linalg.eig` (general) and `np.linalg.eigh` (symmetric).

## 2. Diagonalisation and matrix powers

**Intuition.** In the eigenvector coordinate system, $A$ is nothing but a set of independent scalings. "Change to eigen-coordinates, scale each coordinate, change back" is all a diagonalisable matrix ever does. The payoff is that $A$ applied a thousand times is just the same scalings raised to the thousandth power — and the largest one dominates everything else.

**Definition.** If $A$ has $n$ linearly independent eigenvectors, stack them as columns of $P$ and the eigenvalues in a diagonal $D$. Then $AP = PD$, so

$$A = P D P^{-1}, \qquad A^k = P D^k P^{-1}, \qquad A^k x = \sum_{i=1}^{n} c_i\,\lambda_i^k\, v_i \ \text{ where } x = \sum_i c_i v_i .$$

*When does $P$ exist?* Each eigenvalue has an *algebraic multiplicity* (how often it is a root of the characteristic polynomial) and a *geometric multiplicity* (the dimension of its eigenspace, $\dim\operatorname{null}(A - \lambda I)$), and the second never exceeds the first. $A$ is diagonalisable exactly when the geometric multiplicities add up to $n$. Two guarantees cover most matrices met in practice. Distinct eigenvalues give independent eigenvectors: if $v_2 = c\,v_1$ for eigenvectors with $\lambda_1 \ne \lambda_2$, then $\lambda_2 v_2 = A v_2 = c\lambda_1 v_1 = \lambda_1 v_2$, impossible for $v_2 \ne 0$ (induction extends the argument to any number of eigenvectors), so an $n\times n$ matrix with $n$ distinct eigenvalues is diagonalisable (with complex $P$ and $D$ when some eigenvalues are complex, as for the rotation $R$ of Section 1). And a symmetric matrix is always diagonalisable, with orthogonal eigenvectors (Section 3). The shear of Section 1 is the failing case (Figure 1(c)): its eigenvalue $1$ has algebraic multiplicity 2 but geometric multiplicity 1, so $P$ would need two independent eigenvectors where only one direction exists. The identity matrix has the same trace, determinant and eigenvalues but geometric multiplicity 2, so the characteristic polynomial alone cannot tell the two apart. In code, `n - np.linalg.matrix_rank(A - lam * np.eye(n))` is the geometric multiplicity of an exactly known eigenvalue `lam`: 1 for the shear, 2 for the identity.

*Why the dominant eigenvalue wins.* Factor the largest term out of the expansion above:

$$A^k x = c_1 \lambda_1^k \Big[\, v_1 + \sum_{i \ge 2} \frac{c_i}{c_1} \Big(\frac{\lambda_i}{\lambda_1}\Big)^k v_i \Big].$$

If $|\lambda_1| > |\lambda_2| \ge \dots$ and $c_1 \ne 0$, every term in the bracket except $v_1$ dies out, the slowest like $|\lambda_2/\lambda_1|^k$, so the direction of $A^k x$ approaches the *dominant eigenvector* $v_1$ while the prefactor only sets the length. For the worked example of Section 1, $e_1 = \tfrac23 v_1 + \tfrac13 v_2$, so $A^k e_1 = \tfrac23\, 5^k \big[v_1 + \tfrac12 (2/5)^k\, v_2\big]$: the vectors $Ae_1 = (4, 2)$ (Figure 1(a)), $A^2 e_1 = (18, 14)$, $A^3 e_1 = (86, 78)$ and $A^4 e_1 = (422, 406)$ point at $26.6°$, $37.9°$, $42.2°$ and $43.9°$, closing in on the $45°$ of $v_1$ with the gap shrinking by a factor of about $0.4 = \lambda_2/\lambda_1$ per step. **Power iteration** turns this into an algorithm: start from a random $x_0$ and repeat $x_{k+1} = A x_k / \|A x_k\|$ (the normalisation only keeps the numbers in range). The angle to $v_1$ shrinks like $|\lambda_2/\lambda_1|^k$, and $\lambda_1 \approx x_k^\top A x_k$ (the Rayleigh quotient of Section 3). Three things can spoil it: a start with $c_1 = 0$ never sees $v_1$ (a random start avoids this with probability one); a second eigenvalue of the same modulus, such as $-1$ beside $1$ for a periodic chain, or a complex pair, keeps the bracket from settling; and a negative $\lambda_1$ flips the sign of $x_k$ at every step, so compare directions up to sign.

*Stochastic matrices.* A *column-stochastic* matrix $T$ has non-negative entries and columns that sum to one, $\mathbf{1}^\top T = \mathbf{1}^\top$: it moves probability mass around without creating or destroying any. ([Module 02](../02-matrices-and-linear-maps/README.md) writes Markov chains with the row-stochastic $P = T^\top$ acting on row vectors; $T\pi = \pi$ and $\pi^\top P = \pi^\top$ are the same equation.) $T$ always has the eigenvalue $1$: $\mathbf{1}^\top T = \mathbf{1}^\top$ says that $\mathbf{1}$ is an eigenvector of $T^\top$ with eigenvalue $1$, and $T^\top$ has the same characteristic polynomial as $T$. No eigenvalue is larger in modulus, because $T$ cannot increase the $L_1$ norm: $\|Tx\|_1 = \sum_i \big|\sum_j T_{ij} x_j\big| \le \sum_j \big(\sum_i T_{ij}\big) |x_j| = \|x\|_1$, so $Tv = \lambda v$ forces $|\lambda| \le 1$. For an irreducible, aperiodic chain the Perron–Frobenius theorem adds that the eigenvalue $1$ is simple and strictly dominant, with an eigenvector that can be taken positive; normalised to sum to one, it is the *stationary distribution* $\pi$ with $T\pi = \pi$, and the distance to $\pi$ shrinks like $|\lambda_2|^k$ — the rate behind the mixing time of MCMC in [Module 18](../../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md).

**Worked example (two-state Markov chain).** Two brands compete. Each month 90 % of brand A's customers stay and 10 % switch; 50 % of brand B's customers switch to A and 50 % stay. With market shares as a column vector (the numbers are those of Module 02's weather chain, transposed),

$$T = \begin{pmatrix} 0.9 & 0.5 \\ 0.1 & 0.5\end{pmatrix}, \qquad \lambda^2 - 1.4\lambda + 0.4 = (\lambda - 1)(\lambda - 0.4) = 0 .$$

For $\lambda = 1$: $(T - I) = \begin{pmatrix} -0.1 & 0.5 \\ 0.1 & -0.5\end{pmatrix}$ gives $v \propto (5, 1)$, so $\pi = (5/6, 1/6) \approx (0.833, 0.167)$. For $\lambda = 0.4$: $v_2 = (1, -1)$. Start with all customers at B, $x_0 = (0, 1) = \pi + c\,v_2$ with $c = -5/6$. Then

$$x_k = \pi - \tfrac{5}{6}\,(0.4)^k\,(1, -1), \qquad x_3 = \left(\tfrac56 - \tfrac56\cdot 0.064,\ \tfrac16 + \tfrac56 \cdot 0.064\right) = (0.78, 0.22).$$

Multiplying directly confirms it: $x_1 = (0.5, 0.5)$, $x_2 = (0.7, 0.3)$, $x_3 = (0.78, 0.22)$. Since $0.4^{10} \approx 10^{-4}$, the shares are within 0.01 percentage points of $(83.3\,\%, 16.7\,\%)$ after ten months, whatever the start — the eigenvalue $0.4$ *is* the speed of forgetting. Figure 2 shows both halves of the formula: the share vectors walk along the segment $x_1 + x_2 = 1$ towards $\pi$, and their offset from $\pi$ points along $v_2$ and shrinks by the factor $0.4$ every month.

![Two panels. Left: market-share vectors plotted with the share of brand A across and the share of brand B up. All of them lie on a grey segment from (1, 0) to (0, 1); starting at (0, 1), the shares step to (0.5, 0.5), (0.7, 0.3) and (0.78, 0.22), closing in on a red star at (5/6, 1/6) that lies on a dashed red line from the origin, the eigenvalue-1 direction. Right: the distance to the star on a logarithmic axis against the month, a straight line of dots falling from about 1.2 to about 0.0001 after ten months, beside a much flatter dashed orange line.](figures/markov-forgetting.svg)

**Figure 2.** The two-brand chain of the worked example. (a) Every share vector lies on the segment $x_1 + x_2 = 1$; from $x_0 = (0, 1)$ the shares move through $x_1 = (0.5, 0.5)$, $x_2 = (0.7, 0.3)$ and $x_3 = (0.78, 0.22)$ towards $\pi = (5/6, 1/6)$, which lies on the eigenvalue-1 direction $(5, 1)$ (red dashed line). The offset $x_k - \pi$ always points along $v_2 = (1, -1)$ and shrinks by the factor $0.4$ each month. (b) The distance $\lVert x_k - \pi\rVert = \tfrac56\sqrt2 \cdot 0.4^k$ (dots: direct multiplication) is a straight line on a log scale, down from $1.18$ to about $1.2\times10^{-4}$ at month 10. The dashed orange line falls by the factor $0.85$ per step, the guaranteed rate for the Google matrix of the PageRank example below — a different, much larger chain, drawn from the same start for comparison: $0.85^k$ needs 57 steps to drop below $10^{-4}$, where $0.4^k$ needs 11.

**Real-life example (web search).** PageRank models a surfer who follows a random out-link with probability $d = 0.85$ and jumps to a uniformly random page otherwise. With $M$ the column-stochastic link matrix ($M_{ij} = 1/\text{outdeg}(j)$ if page $j$ links to $i$; a page with no out-links is treated as linking everywhere), the "Google matrix" is

$$G = d\,M + \frac{1-d}{n}\,\mathbf{1}\mathbf{1}^\top ,$$

and the PageRank vector is its stationary distribution. Because every entry of $G$ is positive, the Perron–Frobenius theorem guarantees a unique positive stationary vector, and one can show the second eigenvalue satisfies $|\lambda_2| \le d$. So power iteration converges at least as fast as $0.85^k$ (the dashed orange line of Figure 2(b)): about 60 iterations ($0.85^{57} \approx 9.5\times10^{-5}$) give four digits, regardless of whether the web has ten pages or ten billion. The lesson notebook runs this on a six-page web.

**In ML/AI this is …** PageRank and personalised PageRank (recommendation, graph preprocessing); the Lanczos/Arnoldi solvers behind `scipy.sparse.linalg.eigs`, which are power iteration with memory; spectral normalisation in GANs, which runs one power-iteration step per training step to estimate each weight matrix's largest singular value; and the stationary distributions that MCMC (Module 18) converges to. The averaging argument that caps every $|\lambda|$ at $1$ also drives reinforcement learning: a row-stochastic matrix averages a value vector, so the discounted updates of policy evaluation and value iteration contract by the factor $\gamma$ per step ([Module 19](../../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)).

## 3. Symmetric matrices and the spectral theorem

**Intuition.** Symmetric matrices ($A = A^\top$) are the well-behaved family: their eigenvalues are always real and their eigenvectors can always be chosen perpendicular to each other. Geometrically, a symmetric matrix is a pure stretch along a set of mutually orthogonal axes — no rotation, no shear. Covariance matrices, Hessians, kernel matrices and graph Laplacians are all symmetric, which is why so much of ML lives in this comfortable world.

**Definition.** The **spectral theorem**: every real symmetric $A$ can be written as

$$A = Q \Lambda Q^\top = \sum_{i=1}^n \lambda_i\, q_i q_i^\top ,$$

with $Q$ orthogonal ($Q^\top Q = I$, columns $q_i$ orthonormal eigenvectors) and $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ real. The short proof that eigenvectors for *distinct* eigenvalues are orthogonal: if $Au = \lambda_1 u$ and $Av = \lambda_2 v$, then $\lambda_1 u^\top v = (Au)^\top v = u^\top A v = \lambda_2 u^\top v$, so $(\lambda_1 - \lambda_2)\,u^\top v = 0$.

The **quadratic form** $x^\top A x$ becomes a weighted sum of squares in eigen-coordinates $y = Q^\top x$:

$$x^\top A x = \sum_i \lambda_i\, y_i^2 .$$

Hence $A$ is **positive semidefinite** (PSD, written $A \succeq 0$: $x^\top A x \ge 0$ for all $x$) exactly when all $\lambda_i \ge 0$, and **positive definite** ($A \succ 0$) when all $\lambda_i > 0$. Two important PSD families: Gram matrices $X^\top X$, since $x^\top X^\top X x = \|Xx\|^2 \ge 0$; and covariance matrices $\Sigma$, since $w^\top \Sigma w = \operatorname{Var}(w^\top z) \ge 0$ (Module 09). The **Rayleigh quotient** $R(x) = x^\top A x / x^\top x$ satisfies $\lambda_{\min} \le R(x) \le \lambda_{\max}$, with the maximum attained at the top eigenvector — the fact PCA rests on. (In eigen-coordinates $R(x) = \sum_i \lambda_i y_i^2 / \sum_i y_i^2$ is a weighted average of the eigenvalues, so it cannot leave their range.)

**Worked example.** $S = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$ has trace 4 and determinant 3, so $\lambda^2 - 4\lambda + 3 = 0$ gives $\lambda_1 = 3$, $\lambda_2 = 1$, with unit eigenvectors $q_1 = (1, 1)/\sqrt2$ and $q_2 = (1, -1)/\sqrt2$ (orthogonal, as promised). Reconstruct:

$$3\,q_1 q_1^\top + 1\,q_2 q_2^\top = 3\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5\end{pmatrix} + \begin{pmatrix} 0.5 & -0.5 \\ -0.5 & 0.5\end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}. \checkmark$$

The quadratic form $x^\top S x = 2x_1^2 + 2x_1x_2 + 2x_2^2$ becomes $3y_1^2 + y_2^2$ with $y_1 = (x_1 + x_2)/\sqrt2$, $y_2 = (x_1 - x_2)/\sqrt2$; at $x = (1, 0)$ both give $2$. The level set $x^\top S x = 1$ is an ellipse with semi-axes $1/\sqrt3 \approx 0.577$ along $q_1$ and $1$ along $q_2$: the matrix is "stiffest" along $(1,1)$. Both eigenvalues are positive, so $S \succ 0$. Figure 3 sets three readings side by side: $S$ maps the unit circle to an ellipse with semi-axes $3$ along $q_1$ and $1$ along $q_2$ (Figure 3(a)); on the unit circle the form equals $2 + \sin 2\theta$, swinging between $\lambda_{\min} = 1$ and $\lambda_{\max} = 3$ exactly at the eigenvectors (Figure 3(b)); and the level set is the ellipse of Figure 3(c).

![Three panels. Left: the unit circle and its image under the symmetric matrix with rows 2, 1 and 1, 2, a tilted ellipse that reaches 3 along the diagonal direction (1, 1) and 1 along (1, -1), with the two perpendicular eigenvectors drawn. Middle: the value of the quadratic form for unit vectors, plotted against their angle from -90 to 180 degrees, a wave between 1 and 3 with its maximum 3 at 45 degrees, its minimum 1 at 135 and -45 degrees, and the value 2 at 0 degrees. Right: the level set where the quadratic form equals 1, an ellipse inside the unit circle that reaches only 0.577 along (1, 1) and touches the circle along (1, -1).](figures/quadratic-form.svg)

**Figure 3.** Three views of $S = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$. (a) $S$ stretches the unit circle by $3$ along $q_1 = (1, 1)/\sqrt2$ and by $1$ along $q_2 = (1, -1)/\sqrt2$: a pure stretch along perpendicular axes, with no rotation. (b) On the unit circle $x = (\cos\theta, \sin\theta)$ the quadratic form is $x^\top S x = 2 + \sin 2\theta$. It stays between $\lambda_{\min} = 1$ and $\lambda_{\max} = 3$, reaching the maximum at $q_1$ ($45°$) and the minimum at $q_2$ ($-45°$) and $-q_2$ ($135°$) — the Rayleigh-quotient bounds — and equals $2$ at $x = (1, 0)$, as in the worked example. (c) The level set $x^\top S x = 1$ has semi-axes $1/\sqrt3 \approx 0.577$ along $q_1$ and $1$ along $q_2$. Panels (a) and (c) show different ellipses: the stiff direction $q_1$ is the *long* axis of the image of the circle but the *short* axis of the level set, because a large eigenvalue makes $x^\top S x$ reach $1$ sooner.

**Real-life example (finance).** The risk of a portfolio with weights $w$ over assets with covariance $\Sigma$ is the quadratic form $w^\top \Sigma w$. Two assets with annual volatilities 20 % and 30 % and correlation 0.2 have

$$\Sigma = \begin{pmatrix} 0.04 & 0.012 \\ 0.012 & 0.09\end{pmatrix}, \qquad w = (0.5, 0.5):\ w^\top\Sigma w = 0.01 + 0.0225 + 0.006 = 0.0385,$$

a volatility of $\sqrt{0.0385} \approx 19.6\,\%$ — below the 25 % average of the two, the diversification effect. The eigenvalues $0.0927$ and $0.0373$ say 71 % of the total variance sits in one direction (eigenvector $\approx (0.22, 0.98)$, dominated by the riskier asset), and the eigenvector $(0.98, -0.22)$ is the lowest-risk way to trade the two against each other. Engineers do the identical computation with a stiffness matrix $K$: the elastic energy $\tfrac12 x^\top K x$ must be non-negative, so $K \succeq 0$.

**In ML/AI this is …** the covariance matrix (Module 09) and the Hessian (Modules 04–05): a positive-definite Hessian at a stationary point ($\nabla f = 0$) certifies a strict local minimum, a Hessian that is PSD everywhere (a convex function) makes every stationary point a global minimum, and the Hessian's eigenvalue ratio is the condition number of Section 8. Kernel (Gram) matrices in SVMs and Gaussian processes must be PSD for the kernel to be valid (Mercer's condition). In code: `np.linalg.eigh`, which exploits symmetry to return real, sorted eigenvalues and an orthonormal $Q$.

## 4. The singular value decomposition

**Intuition.** Eigendecomposition needs a square matrix and still may deliver complex numbers or too few eigenvectors. The SVD fixes all of this by allowing *two* different orthonormal bases, one for the input space and one for the output space. Every matrix — rectangular, rank-deficient, anything — is then "rotate, stretch along the axes, rotate again" (where a "rotation" may include a reflection, as explained below). A unit circle in the input space always comes out as an ellipse whose semi-axes are the singular values.

**Definition.** Any $A \in \mathbb{R}^{m\times n}$ factors as

$$A = U \Sigma V^\top = \sum_{i=1}^{r} \sigma_i\, u_i v_i^\top ,$$

with $U \in \mathbb{R}^{m\times m}$ and $V \in \mathbb{R}^{n\times n}$ orthogonal, and $\Sigma \in \mathbb{R}^{m\times n}$ zero except for the *singular values* $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ on its diagonal, where $r = \operatorname{rank}(A)$.

*Two bases, not one.* Read $AV = U\Sigma$ column by column: $A v_i = \sigma_i u_i$ for $i \le r$ and $A v_i = 0$ for $i > r$. The right singular vector $v_i$ is sent to $\sigma_i$ times the left singular vector $u_i$, so $A$ takes the orthonormal input directions $v_1, \dots, v_n$ to mutually orthogonal output directions, stretching the first $r$ by $\sigma_1, \dots, \sigma_r$ and squashing the rest to zero. Compare $A = PDP^{-1}$ (Section 2), which uses one basis for input and output and does not always exist, and $A = Q\Lambda Q^\top$ (Section 3), which uses one orthonormal basis but needs a symmetric matrix: allowing a second basis is what makes the SVD exist for every matrix.

*Where the SVD comes from.* $A^\top A$ is symmetric and PSD ($x^\top A^\top A x = \|Ax\|^2 \ge 0$), so by the spectral theorem it has orthonormal eigenvectors $v_1, \dots, v_n$ with eigenvalues $\sigma_1^2 \ge \dots \ge \sigma_n^2 \ge 0$, exactly $r$ of them positive. For each positive one set $u_i = A v_i/\sigma_i$. These vectors are orthonormal automatically, because $v_j$ is an eigenvector of $A^\top A$:

$$u_i^\top u_j = \frac{v_i^\top A^\top A\, v_j}{\sigma_i \sigma_j} = \frac{\sigma_j^2\; v_i^\top v_j}{\sigma_i \sigma_j} = \begin{cases} 1 & i = j, \\ 0 & i \ne j, \end{cases}$$

and for the zero eigenvalues $\|A v_i\|^2 = v_i^\top A^\top A v_i = 0$. Complete $u_1, \dots, u_r$ to an orthonormal basis of $\mathbb{R}^m$; then $AV = U\Sigma$ holds column by column, and multiplying on the right by $V^\top$ gives $A = U\Sigma V^\top$. The same factors diagonalise both Gram matrices,

$$A^\top A = V \Sigma^\top\Sigma V^\top, \qquad A A^\top = U \Sigma \Sigma^\top U^\top ,$$

so the $v_i$ are eigenvectors of $A^\top A$, the $u_i$ eigenvectors of $AA^\top$, and $\sigma_i^2$ are the shared non-zero eigenvalues — real and non-negative because both matrices are PSD. This is a proof by construction, not how software computes an SVD: forming $A^\top A$ squares the condition number (Section 8, pitfall 7), so libraries work on $A$ directly.

*What the SVD reads off.* $\|A\|_2 = \sigma_1$, the largest stretch ($\|Ax\|^2 = \sum_i \sigma_i^2 (v_i \cdot x)^2 \le \sigma_1^2 \|x\|^2$, with equality at $x = v_1$); $\|A\|_F^2 = \sum_i \sigma_i^2$, because orthogonal factors do not change the Frobenius norm; $|\det A| = \prod_i \sigma_i$ for square $A$, because $\det U$ and $\det V$ are $\pm 1$; and orthonormal bases of the **four fundamental subspaces** of [Module 02](../02-matrices-and-linear-maps/README.md) (Section 5). Every output $Ax = \sum_{i \le r} \sigma_i (v_i \cdot x)\, u_i$ is a combination of $u_1, \dots, u_r$, and each $u_i = A(v_i/\sigma_i)$ is reached, so $\operatorname{col}(A) = \operatorname{span}\{u_1,\dots,u_r\}$; and $Ax = 0$ exactly when $x$ has no component along $v_1, \dots, v_r$, so $\operatorname{null}(A) = \operatorname{span}\{v_{r+1},\dots,v_n\}$, of dimension $n - r$ as rank–nullity demands. The same argument for $A^\top = V\Sigma^\top U^\top$ gives $\operatorname{row}(A) = \operatorname{span}\{v_1,\dots,v_r\}$ and $\operatorname{null}(A^\top) = \operatorname{span}\{u_{r+1},\dots,u_m\}$, and the rank is the number of non-zero singular values. For the rank-one $\begin{pmatrix} 1 & 1 \\ 1 & 1\end{pmatrix}$ of Section 7 all four subspaces are lines, drawn in Figure 7: $\operatorname{col}(A)$ and $\operatorname{row}(A)$ along $(1, 1)$, $\operatorname{null}(A)$ and $\operatorname{null}(A^\top)$ along $(1, -1)$.

*Thin SVD, and what NumPy returns.* Only the first $\min(m, n)$ columns of $U$ and $V$ ever meet a diagonal entry of $\Sigma$; the *thin* SVD keeps just those, with a square $\Sigma$, and the *compact* SVD keeps only the first $r$. `U, s, Vt = np.linalg.svd(A, full_matrices=False)` returns the thin version, with `s` a 1-D array in decreasing order and `Vt` $= V^\top$, whose *rows* are the $v_i$; `(U * s) @ Vt` rebuilds $A$. Three quick checks that the output has been read correctly: `np.allclose(A @ Vt.T, U * s)` ($Av_i = \sigma_i u_i$ for every $i$), `np.allclose(U.T @ U, np.eye(len(s)))` and `np.isclose(np.linalg.norm(A, 2), s[0])`.

*Signs, repeats and the spectral theorem.* The singular values are unique; the vectors are not quite. Each pair $(u_i, v_i)$ can be replaced by $(-u_i, -v_i)$, since $\sigma_i(-u_i)(-v_i)^\top = \sigma_i u_i v_i^\top$ — for the worked example below NumPy returns exactly the negatives of $u_1, u_2, v_1, v_2$ — and when a singular value repeats, any orthonormal basis of its subspace will do (an orthogonal matrix has every $\sigma_i = 1$, and any orthonormal basis can serve as $V$); Case study 3 meets this in noisy form. The sign freedom also decides whether "rotate" in the intuition is literal. Orthogonal matrices include reflections, and with the signs printed below $\det V = \det U = -1$; flipping $v_2$ and $u_2$ together turns $V$ into the rotation by $45°$ and $U$ into the rotation by $71.6°$. Such a choice exists whenever $\det A > 0$, while for $\det A < 0$ one of the two factors must be a reflection. For a symmetric PSD matrix the SVD *is* the spectral decomposition ($\sigma_i = \lambda_i$, $u_i = v_i = q_i$); for a symmetric matrix with a negative eigenvalue the sign moves into $U$: $\begin{pmatrix} 1 & 2 \\ 2 & 1\end{pmatrix}$ (self-assessment 3) has eigenvalues $3$ and $-1$ but singular values $3$ and $1$, the second with $v = q = (1, -1)/\sqrt2$ and $u = -q$.

**Worked example.** $A = \begin{pmatrix} 3 & 0 \\ 4 & 5\end{pmatrix}$. Then $A^\top A = \begin{pmatrix} 25 & 20 \\ 20 & 25\end{pmatrix}$ with eigenvalues $45$ and $5$ (trace 50, det 225), so

$$\sigma_1 = \sqrt{45} = 3\sqrt5 \approx 6.708, \qquad \sigma_2 = \sqrt5 \approx 2.236, \qquad v_1 = \tfrac{1}{\sqrt2}(1, 1),\ v_2 = \tfrac{1}{\sqrt2}(1, -1).$$

Left singular vectors from $u_i = A v_i / \sigma_i$: $A v_1 = (3, 9)/\sqrt2$, so $u_1 = (3, 9)/\sqrt{90} = (1, 3)/\sqrt{10}$; $A v_2 = (3, -1)/\sqrt2$, so $u_2 = (3, -1)/\sqrt{10}$; and $u_1 \cdot u_2 = (3 - 3)/10 = 0$, as the construction promised. Check $\sigma_1\sigma_2 = \sqrt{225} = 15 = \det A$. Note the eigenvalues of this (triangular) matrix are $3$ and $5$ — *not* the singular values; only their product agrees. The unit circle is mapped to an ellipse with semi-axes $6.71$ along $u_1$ and $2.24$ along $u_2$; Figure 4 follows it through $V^\top$, $\Sigma$ and $U$ in turn.

![Three panels following the unit circle through the SVD of the matrix with rows 3, 0 and 4, 5. Left: the unit circle with the perpendicular right singular vectors v1 at 45 degrees and v2 at -45 degrees, and a green dot at (0, 1). Middle: after the first two factors the circle has become an ellipse lying along the axes, 6.71 to each side horizontally and 2.24 vertically, and the green dot sits at (4.74, -1.58). Right: after the last factor the ellipse is tilted, its long axis along u1 at 71.6 degrees, and the green dot lands at (0, 5), on the ellipse but not on one of its axes.](figures/svd-rotate-stretch.svg)

**Figure 4.** The worked example $A = \begin{pmatrix} 3 & 0 \\ 4 & 5\end{pmatrix} = U\Sigma V^\top$ applied to the unit circle one factor at a time. (a) The input, with the perpendicular right singular vectors $v_1$ ($45°$) and $v_2$ ($-45°$). (b) $V^\top$ moves $v_1$ and $v_2$ onto the axes, and $\Sigma$ stretches the axes by $\sigma_1 = 6.71$ and $\sigma_2 = 2.24$ (dashed: the unit circle, for scale). (c) $U$ moves the axes onto $u_1$ ($71.6°$) and $u_2$, giving the ellipse that $A$ maps the circle to, with semi-axes $\sigma_1 u_1 = Av_1$ and $\sigma_2 u_2 = Av_2$. The green dot follows $(0, 1)$, the eigenvector of $A$ for $\lambda = 5$: it lands on the ellipse at $A(0, 1) = (0, 5)$ but on neither axis, because eigenvectors and singular vectors are different things. With the signs printed in the worked example $\det V = \det U = -1$, so the first and last steps are reflections; flipping $v_2$ and $u_2$ together makes them rotations, by $-45°$ and $71.6°$, without changing $A$.

**Real-life example (computer vision and structural biology).** Aligning two 3-D point clouds — a scanned part to its CAD model, or two protein structures — means finding the rotation $R$ minimising $\sum_i \|R p_i - q_i\|^2$ after centring. The Kabsch algorithm computes the $3\times3$ cross-covariance $H = \sum_i p_i q_i^\top$, takes its SVD $H = U\Sigma V^\top$, and returns $R = V U^\top$ (with one sign flipped if $\det < 0$, to exclude a reflection). The three singular values tell you how well-determined the alignment is. A nearly collinear (needle-shaped) cloud has $\sigma_2 \approx \sigma_3 \approx 0$, and its rotation about its own axis is ambiguous. A merely flat cloud, with only $\sigma_3 \approx 0$, still pins the rotation down, but it fits its mirror image almost as well — a flat object reflected in its own plane is unchanged — so it is the determinant check that keeps the answer a rotation.

**In ML/AI this is …** `np.linalg.svd` / `torch.linalg.svd`; the spectral norm $\sigma_1$ of a weight matrix, which bounds how much a layer can amplify a perturbation (Lipschitz constant, adversarial robustness, spectral normalisation); orthogonal weight initialisation (take $U$ from the SVD of a Gaussian matrix); whitening transforms; and the observation that the attention score matrix $W_Q W_K^\top$ in a transformer head has rank at most $d_{\text{head}}$ (Module 17).

## 5. Low-rank approximation

**Intuition.** The SVD writes $A$ as a sum of rank-one layers $\sigma_i u_i v_i^\top$, ordered from most to least important. Keeping the first $k$ layers and discarding the rest is the best possible compression of $A$ into $k$ "patterns" — and the discarded singular values tell you exactly how much you lost. In data, the small singular values are typically noise or fine detail; in weight matrices, they are the part a cheap update does not need.

**Definition.** The **truncated SVD** $A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^\top = U_k \Sigma_k V_k^\top$ has rank $k$. The **Eckart–Young–Mirsky theorem**: for every matrix $B$ of rank at most $k$,

$$\|A - B\|_F \ \ge\ \|A - A_k\|_F = \sqrt{\sum_{i>k} \sigma_i^2}, \qquad \|A - B\|_2 \ \ge\ \|A - A_k\|_2 = \sigma_{k+1} .$$

No rank-$k$ matrix is closer to $A$, in either norm. The **energy** (squared Frobenius norm) captured by $A_k$ is $\sum_{i\le k}\sigma_i^2 / \sum_i \sigma_i^2$. Storage drops from $mn$ numbers to $k(m + n + 1)$. The proof hinges on the fact that any rank-$k$ $B$ has a null space of dimension $\ge n-k$, which must intersect $\operatorname{span}\{v_1,\dots,v_{k+1}\}$ in some unit vector $w$, and then $\|A - B\|_2 \ge \|(A-B)w\| = \|Aw\| \ge \sigma_{k+1}$.

**Worked example.** Continue with $A = \begin{pmatrix} 3 & 0 \\ 4 & 5\end{pmatrix}$. Its best rank-one approximation is

$$A_1 = \sigma_1 u_1 v_1^\top = 3\sqrt5 \cdot \frac{1}{\sqrt{10}}\begin{pmatrix}1\\3\end{pmatrix}\frac{1}{\sqrt2}\begin{pmatrix}1 & 1\end{pmatrix} = \begin{pmatrix} 1.5 & 1.5 \\ 4.5 & 4.5\end{pmatrix},$$

with error $\|A - A_1\|_F = \sigma_2 = \sqrt5 \approx 2.236$ and energy captured $45/50 = 90\,\%$. Column by column, $A_1 = u_1 u_1^\top A$ (because $A^\top u_1 = \sigma_1 v_1$): each column of $A_1$ is the orthogonal projection of the matching column of $A$ onto the line along $u_1$. Both columns, $(3, 4)$ and $(0, 5)$, land on $(1.5, 4.5)$, with squared residuals $2.5 + 2.5 = 5 = \sigma_2^2$. A matrix of rank at most one is exactly one whose columns lie on a single line through the origin, so here Eckart–Young says that the line along $u_1$ is the best such line: dropping the columns onto the $x_2$-axis instead would leave a squared error of $9$, and onto the line along $(1, 1)$ one of $13$ (Figure 5(a)). For a $512\times512$ grey-scale image, keeping $k = 20$ layers stores $20 \cdot (512 + 512 + 1) = 20{,}500$ numbers instead of $262{,}144$ — 7.8 %, drawn to scale in Figure 5(b) — and typically reproduces the picture recognisably, because natural images have fast-decaying singular values.

![Two panels. Left: the two columns of the matrix with rows 3, 0 and 4, 5 drawn as points, (3, 4) and (0, 5), and a line through the origin along u1; dashed perpendiculars drop both points onto the line, where they land on the same point (1.5, 4.5). Right: a to-scale block diagram of a 512 by 512 image matrix beside its rank-20 factors, a tall 512 by 20 strip, a tiny 20 by 20 square and a wide 20 by 512 strip, which together hold 20,500 numbers instead of 262,144.](figures/low-rank-projection.svg)

**Figure 5.** (a) The best rank-one approximation of the worked example, column by column: $A_1 = u_1 u_1^\top A$ drops each column of $A$ perpendicularly onto the line along $u_1 = (1, 3)/\sqrt{10}$. Both land on $(1.5, 4.5)$, so both columns of $A_1$ are $(1.5, 4.5)$, and the squared lengths of the dashed residuals add up to $2.5 + 2.5 = 5 = \sigma_2^2 = \lVert A - A_1\rVert_F^2$; any other line does worse — $9$ for the $x_2$-axis, $13$ for $(1, 1)$. (b) The truncated SVD of a $512\times512$ image at rank $k = 20$, drawn to scale: $U_k$, $\Sigma_k$ and $V_k^\top$ hold $10{,}240 + 20 + 10{,}240 = 20{,}500$ numbers, 7.8 % of the $262{,}144$ in $A$.

**Real-life example (e-commerce recommendation).** The Netflix Prize ratings matrix had 480,189 users and 17,770 films but only 100,480,507 ratings — 1.2 % of the cells. A rank-50 factorisation $R \approx P Q^\top$ assigns each user a 50-dimensional taste vector $p_u$ and each film a 50-dimensional profile $q_i$, predicting $\hat r_{ui} = p_u^\top q_i$; that is about 25 million parameters standing in for 8.5 billion cells. The latent dimensions are learned, not labelled, but they typically line up with recognisable axes such as "art-house versus blockbuster" or "appeals to children". Because most entries are missing, the winning systems fitted the factors by gradient descent on the observed cells only — the mini-project in `exercises.ipynb` builds the SVD-with-imputation version. The same construction underlies latent semantic analysis, where a term-by-document count matrix is truncated to a few hundred "topics".

**In ML/AI this is …** `sklearn.decomposition.TruncatedSVD` (LSA), randomised SVD for very large matrices, denoising by truncation, and **LoRA** (Module 17): instead of updating a $d\times k$ weight matrix $W_0$, train a rank-$r$ update $W_0 + BA$ with $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$. For $d = k = 4096$ and $r = 8$ that is $8 \cdot 8192 = 65{,}536$ trainable numbers instead of $16.8$ million — 0.4 %. Compressing trained weight matrices by truncated SVD is also a standard model-compression step.

## 6. Principal component analysis

**Intuition.** A cloud of data points in $d$ dimensions is usually not spread evenly: it is elongated in some directions and almost flat in others. PCA rotates the coordinate system so the first axis points along the direction of greatest spread, the second along the greatest remaining spread perpendicular to the first, and so on. Keeping the first $k$ axes gives the best $k$-dimensional flat summary of the data, and the dropped axes measure what the summary misses.

**Definition.** Let $X \in \mathbb{R}^{n\times d}$ hold $n$ observations in rows. Centre it, $X_c = X - \mathbf{1}\mu^\top$ with $\mu$ the column means, and form the sample covariance $C = X_c^\top X_c / (n-1)$. The **principal components** are the orthonormal eigenvectors $w_1, \dots, w_d$ of $C$ ordered by eigenvalue $\lambda_1 \ge \dots \ge \lambda_d \ge 0$; $\lambda_i$ is the variance of the data along $w_i$ and $\lambda_i / \sum_j \lambda_j$ the **explained variance ratio**. The first component solves

$$\max_{\|w\| = 1}\ \operatorname{Var}(X_c w) = \max_{\|w\| = 1}\ w^\top C w ,$$

which by the Rayleigh-quotient fact of Section 3 is the top eigenvector.

*Variance along a direction.* The coordinate of a centred point $x_c$ along a unit vector $w$ is its scalar projection $x_c \cdot w$ ([Module 01](../01-vectors-and-geometry/README.md), Section 5), and the $n$ coordinates together form $X_c w$. They have mean zero, so their sample variance is

$$\frac{1}{n-1}\sum_{i=1}^{n} (x_{c,i} \cdot w)^2 = \frac{\|X_c w\|^2}{n-1} = w^\top \frac{X_c^\top X_c}{n-1}\, w = w^\top C\, w ,$$

a quadratic form in the covariance matrix (Section 3) — which is why the two maximisation problems above are the same. For the five points of the worked example below, the variance is $2.5$ along $(1, 0)$ (the variance of $x_1$), $4.75$ along $w_1 = (1, 1)/\sqrt2$ and $0.25$ along $w_2 = (1, -1)/\sqrt2$.

*Why the top eigenvector wins.* Expand a unit vector in the eigenvectors of $C$, $w = \sum_i c_i w_i$ with $\sum_i c_i^2 = 1$; then $w^\top C w = \sum_i \lambda_i c_i^2$, a weighted average of the eigenvalues. It is at most $\lambda_1$, attained at $w = w_1$ (and, if $\lambda_1 > \lambda_2$, only there, up to sign) — the Rayleigh-quotient bound of Section 3, traced around a circle in Figure 3(b). For example, $(1, 0) = (w_1 + w_2)/\sqrt2$ puts weight $\tfrac12$ on each eigenvalue, which gives $(4.75 + 0.25)/2 = 2.5$. The second component maximises the same variance over the directions perpendicular to $w_1$ ($c_1 = 0$), which the same expansion shows is $w_2$, and so on; $k$ components together keep $\lambda_1 + \dots + \lambda_k$ of the *total variance* $\operatorname{tr}(C) = \sum_i \lambda_i$, the sum of the variances of the original features.

*Loadings, scores and the SVD route.* The entries of $w_i$ are its **loadings**, the weights of the original features in the component. For the four Iris measurements in centimetres, the first component's loadings on sepal length, sepal width, petal length and petal width are $(0.36, -0.08, 0.86, 0.36)$: mostly petal length, the measurement that varies most, and the component explains 92 % of the variance. The new coordinates of a point are its **scores** $z = W_k^\top x_c$, the rows of $Z = X_c W_k$ (with $w_1, \dots, w_k$ as the columns of $W_k$), and a plot of $\lambda_i$ against $i$ is the *scree plot*. In practice the components come from the SVD $X_c = U \Sigma V^\top$ rather than from $C$: it gives $C = V\,\Sigma^\top\Sigma\,V^\top/(n-1)$, so the columns of $V$ are the components ($V_k = W_k$), $\lambda_i = \sigma_i^2/(n-1)$, and the scores are $Z = X_c V_k = U_k \Sigma_k$. No covariance matrix is ever formed, which is better conditioned (pitfall 7) and feasible even when $d = 10{,}304$ (Case study 1). Dividing each column of $Z$ by $\sqrt{\lambda_i}$ *whitens* the data: the new coordinates are uncorrelated and have unit variance.

*Variance kept is error avoided.* For each centred point, Pythagoras splits the squared length into the part along $w$ and the residual ([Module 01](../01-vectors-and-geometry/README.md), Section 5): $\|x_c\|^2 = (x_c \cdot w)^2 + \|x_c - (x_c \cdot w)\,w\|^2$. Summing over the points and dividing by $n - 1$,

$$\operatorname{tr}(C) = w^\top C\, w + \frac{1}{n-1}\sum_{i=1}^{n} \big\|x_{c,i} - (x_{c,i} \cdot w)\, w\big\|^2 :$$

total variance = variance kept + squared reconstruction error$/(n-1)$. The left side does not depend on $w$, so the direction that keeps the most variance is also the line through $\mu$ with the smallest squared reconstruction error. With $k$ components the same split holds for the projection onto their span: the rank-$k$ **reconstruction** $\hat X = Z V_k^\top + \mathbf{1}\mu^\top$ is exactly the Eckart–Young optimum for the centred data (Section 5), with total squared error $(n-1)\sum_{i>k}\lambda_i$. In the worked example $5 = 4.75 + 0.25$, and the total squared residual is $4 \cdot 0.25 = 1.0$ (Figure 6(a)).

*Choosing $k$, and two preprocessing traps.* Look for the elbow of the scree plot, keep enough components for a cumulative-variance threshold (90–95 % is common), or — the only criterion that measures what you care about — tune $k$ on the downstream task's validation score (pitfall 5). PCA depends on where the origin is and on the units: without centring, the top singular vector of $X$ points roughly at the mean (pitfall 3, Figure 6(b)); and if the features have different units, the first component simply follows whichever feature has the biggest numbers (pitfall 4, Figure 6(c)), so standardise each column first, which is the same as running PCA on the correlation matrix. The sign of each $w_i$ is arbitrary (pitfall 6).

*Checking against scikit-learn.* `pca = PCA(n_components=k).fit(X)` centres `X` itself; `pca.components_` holds the $w_i$ as *rows*, `pca.explained_variance_` the $\lambda_i$ (with the $n - 1$ divisor), `pca.transform(X)` the scores and `pca.inverse_transform` the reconstruction. To test your own code, take `U, s, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)` and compare `np.abs(Vt[:k])` with `np.abs(pca.components_)` — absolute values, because of the sign freedom — and `s[:k]**2 / (len(X) - 1)` with `pca.explained_variance_`. The covariance matrix itself, with correlation matrices and whitening, is the subject of [Module 09](../../part-3-probability/09-multivariate-probability/README.md).

**Worked example.** Five points: $(3,1), (4,3), (5,2), (6,4), (7,5)$. Mean $\mu = (5, 3)$; centred rows $(-2,-2), (-1,0), (0,-1), (1,1), (2,2)$. Sums of products: $\sum x_1^2 = 10$, $\sum x_2^2 = 10$, $\sum x_1x_2 = 9$, so

$$C = \frac{1}{4}\begin{pmatrix} 10 & 9 \\ 9 & 10\end{pmatrix} = \begin{pmatrix} 2.5 & 2.25 \\ 2.25 & 2.5\end{pmatrix}, \qquad \lambda_1 = 4.75,\ \lambda_2 = 0.25,$$

with $w_1 = (1,1)/\sqrt2$ and $w_2 = (1,-1)/\sqrt2$. The first component explains $4.75/5 = 95\,\%$ of the variance. Scores $z_i = x_{c,i}^\top w_1$: $-2.83, -0.71, -0.71, 1.41, 2.83$; their variance is $4.75 = \lambda_1$, as it must be. Reconstructing the second point from one component: $\mu + z_2 w_1 = (5,3) + (-0.5, -0.5) = (4.5, 2.5)$ instead of $(4, 3)$, a residual of $(-0.5, 0.5)$ that lies exactly along $w_2$. The total squared residual over the five points is $1.0 = 4 \cdot \lambda_2$.

The same five points show the two preprocessing traps of the pitfalls. Skip the centring, and the top right singular vector of the raw $X$ points at $32.3°$ through the origin — close to the direction of the mean $\mu$ ($31.0°$), not to the $45°$ of $w_1$ — and accounts for 99.0 % of $\sum_i \|x_i\|^2$, a number about distance from the origin rather than spread (Figure 6(b)). Record $x_2$ in units ten times finer, and the first component turns to $84.9°$, with loadings $(0.09, 1.00)$, and "explains" 99.8 % of the variance: it follows the column with the big numbers (Figure 6(c)). Standardising both columns gives the correlation matrix $\begin{pmatrix} 1 & 0.9 \\ 0.9 & 1\end{pmatrix}$, with eigenvalues $1.9$ and $0.1$: $45°$ and 95 % again.

![Three panels with the same five points, (3, 1), (4, 3), (5, 2), (6, 4) and (7, 5). Left: the first principal axis runs through their mean (5, 3) at 45 degrees and passes through three of the points; the other two are joined by short red residuals to the same reconstructed point (4.5, 2.5). Middle: without centring, the top singular direction is a line through the origin at 32.3 degrees, almost along the grey arrow to the mean at 31.0 degrees, while the true first axis through the mean is dashed. Right: with the second coordinate multiplied by ten the points form a nearly vertical streak, and the first principal axis follows it at 84.9 degrees.](figures/pca-centring-scaling.svg)

**Figure 6.** (a) The worked example: the first principal axis passes through $\mu = (5, 3)$ along $w_1 = (1, 1)/\sqrt2$ and keeps $\lambda_1 = 4.75$ of the total variance $5$ (95 %). Three points lie on it; $(4, 3)$ and $(5, 2)$ are both reconstructed as $(4.5, 2.5)$ (hollow circle), with residuals $\mp(0.5, -0.5)$ along $w_2$, the direction of the remaining variance $\lambda_2 = 0.25$. (b) Pitfall 3: without centring, the top right singular vector of the raw $X$ is a line through the *origin* at $32.3°$, close to the direction of the mean ($31.0°$, grey arrow) rather than to the spread of the data ($45°$ through $\mu$, dashed); its "99.0 %" is a share of $\sum_i \lVert x_i\rVert^2$, not of the variance. (c) Pitfall 4: the same points with $x_2$ in units ten times finer. The first component turns to $84.9°$ (loadings $(0.09, 1.00)$) and "explains" 99.8 % of the variance because it follows the column with the big numbers; standardising both columns restores (a), $45°$ and 95 %.

**Real-life example (healthcare and genomics).** A gene-expression study measures 20,000 genes in 100 tumour samples: a $100\times20{,}000$ matrix nobody can look at. PCA to two components routinely separates tumour subtypes that respond to different drugs, and the loadings of $w_1$ name the genes driving the separation. Two cautions the practitioners learned the hard way: the first component often tracks a technical artefact (sequencing batch or library size) rather than biology, and genes must be log-transformed and standardised, or a few very highly expressed genes dominate everything. Finance uses the identical tool on bond yields — the first three components of yield-curve changes are "level, slope and curvature" and typically explain more than 95 % of the variation — and engineers use it to compress hundreds of correlated sensor channels into a handful of health indicators.

**In ML/AI this is …** `sklearn.decomposition.PCA` (which centres the data and uses an SVD-equivalent solver; its `explained_variance_ratio_` is $\lambda_i/\sum_j\lambda_j$), whitening of inputs, eigenfaces, visualising embedding spaces before applying t-SNE or UMAP, and a theorem worth remembering: a linear autoencoder with a $k$-unit bottleneck trained with squared error learns exactly the top-$k$ principal subspace (Baldi and Hornik, 1989) — PCA is the simplest representation-learning algorithm.

## 7. The pseudo-inverse and least squares via the SVD

**Intuition.** You cannot undo a stretch by zero. The pseudo-inverse undoes every stretch the matrix *does* perform (divide by each non-zero $\sigma_i$) and simply declines to act along the directions the matrix destroyed. The result is the most sensible "inverse" a rectangular or singular matrix can have, and it hands you least-squares solutions even when the normal equations are singular.

**Definition.** With $A = U\Sigma V^\top$, the **Moore–Penrose pseudo-inverse** is

$$A^{+} = V \Sigma^{+} U^\top, \qquad \Sigma^{+}_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > 0 \\ 0 & \sigma_i = 0 \end{cases},$$

an $n\times m$ matrix. If $A$ has full column rank, $A^+ = (A^\top A)^{-1}A^\top$ — the familiar normal-equations solution; if $A$ is square and invertible, $A^+ = A^{-1}$. In general $x^{+} = A^{+} b$ is the least-squares solution of **minimum norm**: among all $x$ minimising $\|Ax - b\|$, it has the smallest $\|x\|$. In coordinates,

$$x^{+} = \sum_{\sigma_i > 0} \frac{u_i^\top b}{\sigma_i}\, v_i .$$

Each term projects $b$ onto a left singular direction, divides by the stretch, and places the result along the matching right singular direction; components of $b$ outside $\operatorname{col}(A)$ are dropped (they are the residual), and nothing is placed in $\operatorname{null}(A)$ (that is what makes the norm minimal). **Ridge regression** (Module 13) changes the filter to $\sigma_i/(\sigma_i^2 + \lambda)$: directions with $\sigma_i^2 \gg \lambda$ pass almost unchanged, directions with $\sigma_i^2 \ll \lambda$ are shrunk towards zero — a smooth version of the hard cut-off that `np.linalg.pinv(A, rcond=...)` applies.

**Worked example.** $A = \begin{pmatrix} 1 & 1 \\ 1 & 1\end{pmatrix}$ has rank one: $\sigma_1 = 2$, $u_1 = v_1 = (1,1)/\sqrt2$, $\sigma_2 = 0$. So

$$A^{+} = v_1 \frac{1}{\sigma_1} u_1^\top = \frac12 \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5\end{pmatrix} = \begin{pmatrix} 0.25 & 0.25 \\ 0.25 & 0.25\end{pmatrix}.$$

For $b = (1, 1)$ (in the column space) $x^+ = (0.5, 0.5)$: every $x$ with $x_1 + x_2 = 1$ solves $Ax = b$ exactly, and $(0.5, 0.5)$ is the shortest of them — length $0.71$, against $1$ for $(1, 0)$ and $1.58$ for $(1.5, -0.5)$ (Figure 7(b)). For $b = (1, 0)$ (not in the column space) $x^+ = (0.25, 0.25)$ and $Ax^+ = (0.5, 0.5)$, the orthogonal projection of $b$ onto the line spanned by $(1,1)$ — the least-squares fit, with the residual $(0.5, -0.5)$ perpendicular to the column space (Figure 7(a)). The normal equations $A^\top A x = A^\top b$ are singular here and cannot be solved by inversion at all.

![Two panels for the rank-one matrix whose four entries all equal 1. Left, the output plane: the column space is the line along (1, 1); the target (1, 0) lies off it, and a dashed residual (0.5, -0.5) meets the line at a right angle at the fitted point (0.5, 0.5), while the target (1, 1) lies on the line. Right, the input plane: the solutions for the target (1, 1) form the line x1 + x2 = 1, parallel to the null space along (1, -1); the shortest solution (0.5, 0.5), of length 0.71, meets that line at a right angle, while the solutions (1, 0) and (1.5, -0.5) are longer, 1 and 1.58.](figures/pseudo-inverse.svg)

**Figure 7.** The worked example $A = \begin{pmatrix} 1 & 1 \\ 1 & 1\end{pmatrix}$, with the output plane and the input plane drawn separately. (a) Least squares: $\operatorname{col}(A)$ is the line along $(1, 1)$. For $b = (1, 0)$, off that line, $Ax^+ = (0.5, 0.5)$ is the foot of the perpendicular from $b$, and the residual $(0.5, -0.5)$ lies in $\operatorname{null}(A^\top)$; $b = (1, 1)$ lies in $\operatorname{col}(A)$ and is fitted exactly. (b) Minimum norm: the solutions of $Ax = (1, 1)$ fill the line $x_1 + x_2 = 1$, a copy of $\operatorname{null}(A)$ shifted off the origin. $x^+ = (0.5, 0.5)$ is the point of that line nearest the origin (length $0.71$, against $1$ for $(1, 0)$ and $1.58$ for $(1.5, -0.5)$) and lies in $\operatorname{row}(A)$; every other solution adds a null-space vector $z$ and is longer by Pythagoras (self-assessment 7). The small red dot is $x^+ = (0.25, 0.25)$ for $b = (1, 0)$, also in $\operatorname{row}(A)$.

**Real-life example (manufacturing calibration).** A production line regresses product hardness on sensor readings, and two of the sensors are thermocouples that record the same temperature, one in °C and one in °F. Together with the intercept column of ones, the columns are exactly linearly dependent ($F = 1.8\,C + 32 \cdot \mathbf{1}$; after centring, the °C and °F columns are exactly proportional), so $X^\top X$ is singular, and `np.linalg.inv` fails or returns garbage. The pseudo-inverse of the centred columns — which is what `LinearRegression` computes — returns the minimum-norm coefficients, which are proportional to the column scales, $\beta_F = 1.8\,\beta_C$; since the °F column also varies 1.8 times as much, it carries $1.8^2/(1 + 1.8^2) = 3.24/4.24 \approx 76\,\%$ of the temperature effect and the °C column the other 24 % (two sensors in the same unit, or standardised columns, would split it evenly). It is a stable answer that predicts correctly. The lesson for the analyst is the diagnostic one: a zero (or tiny) singular value in $X$ is the fingerprint of redundant features.

**In ML/AI this is …** `np.linalg.lstsq` and `np.linalg.pinv`, both SVD-based, and therefore `sklearn.linear_model.LinearRegression`, which returns the minimum-norm solution whenever there are more features than samples. That minimum-norm interpolator is the object studied in "double descent" (Module 15), and the ridge filter factors above are the cleanest way to see what regularisation does to a linear model (Module 13).

## 8. Condition number and numerical stability

**Intuition.** A matrix that stretches one direction by 10,000 and another by 1 mixes wildly different scales. When you solve $Ax = b$, any error in $b$ along the weak direction gets multiplied by 10,000 relative to the strong direction. The condition number measures this worst-case amplification; a large value means that rounding errors, measurement noise and optimiser steps all behave badly, and that the fix is usually to rescale the problem.

**Definition.** The (2-norm) **condition number** of $A$ is

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} \ \ (\ge 1), \qquad \kappa(A) = \frac{|\lambda_{\max}|}{|\lambda_{\min}|}\ \text{ for symmetric } A .$$

If $Ax = b$ and $A(x + \delta x) = b + \delta b$, then

$$\frac{\|\delta x\|}{\|x\|} \ \le\ \kappa(A)\, \frac{\|\delta b\|}{\|b\|} ,$$

with equality for the worst-case directions ($b$ along $u_1$, $\delta b$ along $u_{\min}$). Rule of thumb: you lose about $\log_{10}\kappa$ significant digits. Two consequences matter constantly in ML. First, $\kappa(A^\top A) = \kappa(A)^2$, so forming the normal equations squares the damage — solve least squares by QR or SVD (`lstsq`), not by `inv(X.T @ X)`. Second, gradient descent on a quadratic $\tfrac12 x^\top A x - b^\top x$ with the best fixed step size shrinks the error by a factor $(\kappa - 1)/(\kappa + 1)$ per step (Module 05): for $\kappa = 100$ that is $0.98$, about 345 steps to gain three digits; for $\kappa = 10^4$ it is $0.9998$, tens of thousands of steps. **Feature scaling** attacks $\kappa$ directly: standardising columns removes the part of $\kappa$ that comes from mismatched units, and centring removes the near-collinearity between a constant column and a slowly varying feature (neither can remove a genuine collinearity between features).

**Worked example.** $A = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001\end{pmatrix}$ has singular values $2.00005$ and $0.00005$, so $\kappa \approx 40{,}002$. With $b = (2, 2.0001)$ the solution is $x = (1, 1)$. Change the second entry of $b$ by $10^{-4}$ (one part in 20,000), to $b' = (2, 2.0002)$, and the solution becomes $x' = (0, 2)$: a relative input change of $\|\delta b\|/\|b\| = 3.5\times10^{-5}$ produced a relative output change of $1.0$, an amplification of about $28{,}000$ — below the bound $\kappa$, because this perturbation was not exactly along the worst direction, but devastating all the same.

Milder numbers show the mechanism. A correlation of 0.9 between two standardised features already gives the Gram matrix $G = \begin{pmatrix} 1 & 0.9 \\ 0.9 & 1\end{pmatrix}$ — the correlation matrix of the five points of Section 6, and the collinear pair of the ridge example in [Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md) — a condition number of $19$: its eigenvalues are $1.9$ along $(1, 1)/\sqrt2$ and $0.1$ along $(1, -1)/\sqrt2$, so it squashes the unit circle into a thin ellipse (Figure 8(a)). Solving $Gx = b$ divides by these eigenvalues, so an error in $b$ along $(1, -1)$ is multiplied by $10$ and one along $(1, 1)$ by $1/1.9 \approx 0.53$. Gradient descent on $\tfrac12 x^\top G x$ with the best fixed step $\eta = 2/(\lambda_{\max} + \lambda_{\min}) = 1$ multiplies the error by $(19 - 1)/(19 + 1) = 0.9$ per step, so three digits take 66 steps, and the iterates zig-zag across the narrow valley (Figure 8(b)).

![Two panels for the matrix with ones on the diagonal and 0.9 off it. Left: the unit circle and its image, a long thin ellipse that reaches 1.9 along (1, 1) but only 0.1 along (1, -1). Right: contour lines of the quadratic bowl form a long narrow valley along (1, -1); gradient-descent iterates starting at (1.5, -0.5) zig-zag back and forth across the valley floor and approach the minimum at the origin only slowly.](figures/condition-number.svg)

**Figure 8.** The figure draws the $\kappa = 19$ Gram matrix $G = \begin{pmatrix} 1 & 0.9 \\ 0.9 & 1\end{pmatrix}$ from the paragraph after the worked example, not the $\kappa \approx 40{,}002$ matrix, whose ellipse would be indistinguishable from a line segment. (a) $G$ maps the unit circle to an ellipse with semi-axes $1.9$ along $q_1 = (1, 1)/\sqrt2$ and $0.1$ along $q_2 = (1, -1)/\sqrt2$. Solving $Gx = b$ runs this backwards, so errors in $b$ along $(1, -1)$ are multiplied by $10$ and errors along $(1, 1)$ by $0.53$; the ratio $19$ is $\kappa$. (b) Gradient descent on $\tfrac12 x^\top G x$ with the best fixed step $\eta = 2/(\lambda_{\max} + \lambda_{\min}) = 1$, from $x_0 = (1.5, -0.5)$ through $x_1 = (0.45, -1.35)$ and $x_2 = (1.215, -0.405)$ to $x_{12} = (0.42, -0.14)$. The iterates zig-zag across the narrow valley whose floor runs along $(1, -1)$ ($\lambda_{\min} = 0.1$), and the distance to the minimum shrinks by exactly $(\kappa - 1)/(\kappa + 1) = 0.9$ per step: 66 steps for three digits, against about 345 for $\kappa = 100$.

**Real-life example (climate).** Fitting a quadratic trend to annual temperature anomalies for 1990–2020 with the design matrix columns $(1, t, t^2)$ and $t$ the raw calendar year gives $\kappa \approx 2\times10^{11}$: the columns $1.99\times10^3$ and $3.96\times10^6$ are nearly parallel and single precision cannot represent the answer at all. Replacing $t$ by $t - 2005$ (centring) gives $\kappa \approx 161$ — the same model, the same fitted curve, and nine orders of magnitude less error amplification. Every polynomial-features pipeline should do this: put `sklearn.preprocessing.StandardScaler` *before* `PolynomialFeatures`, which gives $\kappa \approx 2.8$ here. Standardising the finished columns instead still leaves $\kappa \approx 10^3$, because $t^2$ remains almost a multiple of $t$ (their correlation is $0.999998$).

**In ML/AI this is …** `StandardScaler` before linear models, SVMs and neural networks; the `rcond` argument of `np.linalg.lstsq`/`pinv`, which discards singular values below $\text{rcond}\cdot\sigma_{\max}$; and the condition number of the Hessian, which is why momentum, Adam and second-order methods exist (Module 05) and part of why batch normalisation helps (Module 16).

## Common pitfalls and misconceptions

1. **"Every matrix has eigenvalues, and they are real."** Only square matrices have eigenvalues, and real matrices can have complex ones (any rotation other than by 0° or 180°). → For a non-square matrix use the SVD; for a symmetric one use `np.linalg.eigh`, which guarantees real, sorted output.
2. **"The eigenvectors returned by `eig` are orthogonal."** Only for symmetric (more generally, normal) matrices. For $A = \begin{pmatrix} 4 & 1 \\ 2 & 3\end{pmatrix}$ they are $(1,1)$ and $(1,-2)$, whose lines meet at about 72°, not 90° (Figure 1(a)). → Check $Q^\top Q = I$ before relying on it; use `eigh` when the matrix is symmetric.
3. **Forgetting to centre before an SVD-based PCA.** Without centring the first singular vector points roughly at the mean of the data, not along its largest variance (Figure 6(b)). → Subtract the column means (scikit-learn does this for you; your own NumPy code must).
4. **Running PCA on unscaled features.** A feature measured in grams next to one measured in kilometres owns the first component by sheer magnitude (Figure 6(c)). → Standardise columns when units differ; PCA is not scale-invariant.
5. **"The top principal components are the best features for prediction."** PCA is unsupervised; it finds directions of large *variance*, and the direction that separates your classes may be a low-variance one. → Validate downstream performance against $k$, or use a supervised projection (LDA, PLS).
6. **Reading principal components as features.** A component is a direction — a weighted combination of all features — and its sign is arbitrary. → Look at the loadings to interpret, and never compare the sign of a component across runs or libraries.
7. **Computing an SVD or PCA through $A^\top A$.** It squares the condition number, so small singular values drown in rounding noise (a $\sigma$ of $10^{-5}$ becomes an eigenvalue of $10^{-10}$, below single-precision resolution). → Call `np.linalg.svd` on $A$ itself.
8. **Inverting $X^\top X$ for least squares.** Slow, unstable, and undefined for collinear features. → Use `np.linalg.lstsq` (SVD-based, so it also returns the minimum-norm solution when the system is rank-deficient) or, for full-column-rank problems, QR; plain QR breaks down on a rank-deficient $X$.

## Real-world case studies

### Case study 1 — Eigenfaces: face recognition with PCA (computer vision)

A classic face database holds 400 grey-scale photographs (40 people, 10 each) of $92\times112$ pixels. Each image is a vector in $\mathbb{R}^{10{,}304}$, so the data matrix is $400\times10{,}304$ and the covariance matrix would be $10{,}304\times10{,}304$ — 106 million entries, far too many for the hardware of 1991 when Turk and Pentland published the method. The trick: with only 400 samples the covariance has rank at most 399, and the non-zero eigenvalues of $X_c^\top X_c$ equal those of the tiny $400\times400$ matrix $X_c X_c^\top$. If $X_c X_c^\top a = \lambda a$ then $X_c^\top a$ is an eigenvector of $X_c^\top X_c$ with the same $\lambda$ — a one-line consequence of Section 4. Today you simply call the thin SVD of the $400\times10{,}304$ matrix.

The resulting eigenvectors, reshaped to $92\times112$, are ghostly "eigenfaces": the first captures overall lighting, the next few capture face width, hair line and glasses. Keeping $k = 50$ components typically retains around 90 % of the variance; each face is then described by 50 coordinates instead of 10,304, a 206-fold compression, and storing the model costs $50\times10{,}304 + 400\times50 \approx 535{,}000$ numbers instead of 4.1 million. Recognition is nearest-neighbour search in the 50-dimensional score space. The limitations are instructive: because PCA follows variance, the leading components track illumination and pose rather than identity, so eigenfaces fail under changed lighting — one of the motivations for the supervised projections (Fisherfaces) and, eventually, learned convolutional features that replaced them.

### Case study 2 — Latent factors for recommendation (e-commerce)

The Netflix Prize data (100,480,507 ratings, 480,189 users, 17,770 films) is 98.8 % empty, so a plain SVD is not even defined: there is no matrix to decompose until the gaps are filled. Two routes exist. The *imputation* route fills missing cells with a baseline (global mean plus user and item offsets), takes a rank-$k$ truncated SVD, writes the reconstruction back into the missing cells, and repeats — each pass is an Eckart–Young projection, and the iterations converge to a low-rank completion. The *direct* route, used by the winning teams, minimises $\sum_{(u,i)\ \text{observed}} (r_{ui} - \mu - b_u - b_i - p_u^\top q_i)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2)$ by stochastic gradient descent over the observed cells only; the regulariser is the ridge shrinkage of Section 7, and it is essential because a user with three ratings cannot support a 50-dimensional taste vector.

The arithmetic explains why the method works: a rank-50 model has $(480{,}189 + 17{,}770)\times50 \approx 25$ million parameters, four times fewer than the 100 million observed ratings, so it cannot simply memorise them (although, as the U-shaped curve below shows, too many factors still overfit). Netflix's own system scored a root-mean-square error of about 0.95 stars on held-out ratings, and the competition's goal was a 10 % improvement, to about 0.86; latent-factor models were the single largest contributor. The mini-project in `exercises.ipynb` reproduces the imputation route on a synthetic ratings matrix and shows the characteristic U-shaped curve of held-out RMSE against $k$: too few factors underfit, too many fit the noise in the imputed cells.

### Case study 3 — Empirical orthogonal functions in climate science

Climate scientists have run PCA on gridded fields since the 1950s under the name *empirical orthogonal function* (EOF) analysis. Take monthly sea-surface-temperature anomalies over the tropical Pacific on a $2°$ grid: roughly $30$ latitude bands $\times$ $85$ longitude bands $= 2{,}550$ grid cells, observed for 40 years $= 480$ months. The data matrix is $480\times2{,}550$ (time by space); its right singular vectors are spatial patterns (the EOFs), the left singular vectors scaled by $\sigma_i$ are their time series (the principal components), and $\sigma_i^2/(n-1)$ is the variance each pattern accounts for ($\sigma_i^2/\sum_j \sigma_j^2$ is its share of the total). The leading EOF of this region is the El Niño–Southern Oscillation pattern — warm anomalies along the equator fanning out from South America — and its time series is a standard El Niño index.

Three practical details map directly onto this module. Grid cells near the poles cover less area than those at the equator, so each column is weighted by $\sqrt{\cos(\text{latitude})}$ before the SVD — a change of the inner product. The thin SVD of the $480\times2{,}550$ matrix costs on the order of $480^2\times2{,}550 \approx 6\times10^8$ flops, trivial today, whereas the $2{,}550\times2{,}550$ covariance route was the bottleneck for decades. And the sign of each EOF is arbitrary, and the order of nearly equal eigenvalues is unreliable: two eigenvalues that are close together (within sampling error) can swap order or mix between datasets, because a perturbation of size $\varepsilon$ can rotate their eigenvectors into each other when their gap is smaller than about $\varepsilon$. Researchers therefore test the separation of neighbouring eigenvalues before interpreting a pattern (North's rule of thumb) — the non-uniqueness of singular vectors for repeated singular values (Section 4), blurred by sampling noise.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Eigenvalues of the Hessian | Gradient descent, Newton's method (Module 05) | $\lambda_{\max}$ caps the learning rate; $\kappa = \lambda_{\max}/\lambda_{\min}$ sets the convergence speed |
| Dominant eigenvector / power iteration | PageRank, spectral normalisation, Lanczos solvers | The largest eigen- or singular value is often all you need, and it is cheap |
| Spectral radius $\max_i \lvert\lambda_i\rvert$ | RNN gradients (Module 16), linear dynamical systems, Markov-chain mixing (Module 18) | Repeated multiplication grows when it exceeds 1 and dies out when it is below 1; $\lvert\lambda_2\rvert$ sets how fast a chain forgets its start |
| Stationary distributions, stochastic matrices | MCMC (Module 18), policy evaluation and value iteration (Module 19) | Markov chains converge to the eigenvector with eigenvalue 1; a stochastic matrix only averages ($\lvert\lambda\rvert \le 1$), so discounted value updates contract |
| Spectral theorem, PSD | Covariance matrices (Module 09), kernel matrices, Hessians | Guarantees real eigenvalues and orthogonal axes; PSD validates kernels, and a PD Hessian at a stationary point certifies a local minimum |
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
| Eigenvector of a $2\times2$ | a non-zero row $(a, b)$ of $A - \lambda I$ gives $v = (b, -a)$ |
| Diagonalisation | $A = PDP^{-1}$, $A^k = PD^kP^{-1}$, $A^k x = \sum_i c_i\lambda_i^k v_i$ |
| Diagonalisable | $\iff$ the geometric multiplicities sum to $n$; guaranteed by $n$ distinct eigenvalues (complex $P$ if they are complex) or by $A = A^\top$ |
| Power iteration | $x_{k+1} = Ax_k/\lVert Ax_k\rVert$; error $\propto \lvert\lambda_2/\lambda_1\rvert^k$ (needs $c_1 \ne 0$); $\lambda_1 \approx x_k^\top A x_k$ |
| Column-stochastic $T$ | $\mathbf{1}^\top T = \mathbf{1}^\top \Rightarrow \lambda = 1$ is an eigenvalue; $\lVert Tx\rVert_1 \le \lVert x\rVert_1 \Rightarrow$ every $\lvert\lambda\rvert \le 1$ |
| Google matrix | $G = dM + \frac{1-d}{n}\mathbf{1}\mathbf{1}^\top$; PageRank = stationary vector; $\lvert\lambda_2\rvert \le d$ |
| Spectral theorem | $A = A^\top \Rightarrow A = Q\Lambda Q^\top$, $Q^\top Q = I$, $\Lambda$ real |
| Quadratic form | for symmetric $A$: $x^\top A x = \sum_i \lambda_i y_i^2$ with $y = Q^\top x$; PSD $\Leftrightarrow$ all $\lambda_i \ge 0$ |
| Rayleigh quotient | for symmetric $A$: $\lambda_{\min} \le \dfrac{x^\top A x}{x^\top x} \le \lambda_{\max}$ |
| SVD | $A = U\Sigma V^\top = \sum_i \sigma_i u_i v_i^\top$; $Av_i = \sigma_i u_i$; $\sigma_i^2 = \operatorname{eig}(A^\top A)$, $u_i = Av_i/\sigma_i$ |
| Norms from the SVD | $\lVert A\rVert_2 = \sigma_1$, $\lVert A\rVert_F^2 = \sum_i\sigma_i^2$, $\operatorname{rank} A = \#\{\sigma_i > 0\}$ |
| Subspaces from the SVD | $\operatorname{col}A = \operatorname{span}\{u_1,\dots,u_r\}$, $\operatorname{null}A^\top = \operatorname{span}\{u_{r+1},\dots,u_m\}$, $\operatorname{row}A = \operatorname{span}\{v_1,\dots,v_r\}$, $\operatorname{null}A = \operatorname{span}\{v_{r+1},\dots,v_n\}$ |
| Eckart–Young | $\min_{\operatorname{rank} B \le k}\lVert A - B\rVert_F = \sqrt{\sum_{i>k}\sigma_i^2}$, attained by $A_k = U_k\Sigma_kV_k^\top$ |
| PCA | Centre; $C = X_c^\top X_c/(n-1)$; components = eigenvectors of $C$ = columns of $V$; $\lambda_i = \sigma_i^2/(n-1)$ |
| Variance along a unit $w$ | $\operatorname{Var}(X_c w) = w^\top C w$; $\operatorname{tr} C$ = variance kept + squared residuals$/(n-1)$ |
| PCA scores and reconstruction | $Z = X_cV_k$; $\hat X = ZV_k^\top + \mathbf{1}\mu^\top$; error $= (n-1)\sum_{i>k}\lambda_i$ |
| Pseudo-inverse | $A^+ = V\Sigma^+U^\top$; $x^+ = A^+b = \sum_{\sigma_i>0}\dfrac{u_i^\top b}{\sigma_i}v_i$ (min-norm least squares) |
| Ridge filter | $x_\lambda = \sum_i \dfrac{\sigma_i}{\sigma_i^2 + \lambda}(u_i^\top b)\,v_i$ |
| Condition number | $\kappa = \sigma_{\max}/\sigma_{\min}$; $\dfrac{\lVert\delta x\rVert}{\lVert x\rVert} \le \kappa\dfrac{\lVert\delta b\rVert}{\lVert b\rVert}$; $\kappa(A^\top A) = \kappa(A)^2$ |
| GD on a quadratic | Error factor per step $(\kappa-1)/(\kappa+1)$ with the optimal fixed step |

## Self-assessment

1. **A $2\times2$ matrix has trace 6 and determinant 8. What are its eigenvalues, and is it diagonalisable?**
   <details><summary>Answer</summary>

   $\lambda^2 - 6\lambda + 8 = (\lambda-2)(\lambda-4)$, so $\lambda = 2, 4$. Distinct eigenvalues always give linearly independent eigenvectors (Section 2), hence the matrix is diagonalisable (though the eigenvectors need not be orthogonal unless the matrix is symmetric).

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

   (a) $AA^+b = U_rU_r^\top b$ is the orthogonal projection of $b$ onto $\operatorname{col}(A)$, which is the closest point in the column space. (b) $A^+b$ lies in $\operatorname{row}(A) = \operatorname{span}\{v_1,\dots,v_r\}$, orthogonal to $\operatorname{null}(A)$; any other minimiser is $A^+b + z$ with $z \in \operatorname{null}(A)$, and by Pythagoras $\lVert A^+b + z\rVert^2 = \lVert A^+b\rVert^2 + \lVert z\rVert^2$. Figure 7 draws both halves for $A = \begin{pmatrix} 1 & 1 \\ 1 & 1\end{pmatrix}$.

   </details>

8. **Ridge regression replaces $1/\sigma_i$ by $\sigma_i/(\sigma_i^2+\lambda)$. Which directions does it change most, and what happens as $\lambda \to 0$ and $\lambda \to \infty$?**
   <details><summary>Answer</summary>

   For $\sigma_i^2 \gg \lambda$ the factor is $\approx 1/\sigma_i$ (unchanged); for $\sigma_i^2 \ll \lambda$ it is $\approx \sigma_i/\lambda \to 0$, so the small-singular-value directions — the ones that amplify noise most — are shrunk hardest. As $\lambda \to 0$ ridge becomes the pseudo-inverse solution; as $\lambda \to \infty$ all coefficients go to zero.

   </details>

9. **Why does gradient descent slow down on an ill-conditioned problem, and how does feature scaling help?**
   <details><summary>Answer</summary>

   The step size must be below $2/\lambda_{\max}$ to avoid diverging along the stiff direction, but progress along the flat direction is then proportional to $\lambda_{\min}/\lambda_{\max} = 1/\kappa$ per step. Standardising features equalises the scales of the columns of $X$, bringing the eigenvalues of $X^\top X$ closer together and shrinking $\kappa$, so one step size works well for every direction. Figure 8(b) shows the zig-zag for $\kappa = 19$, where even the best fixed step shrinks the error only by the factor $0.9$ per step.

   </details>

10. **The shear $\begin{pmatrix} 1 & 1 \\ 0 & 1\end{pmatrix}$ and the identity matrix have the same trace, determinant and eigenvalues. Why is only one of them diagonalisable?**
    <details><summary>Answer</summary>

    Both have the double eigenvalue $1$ (algebraic multiplicity 2). For the identity, $I - 1\cdot I = 0$, so every vector is an eigenvector: the geometric multiplicity is 2 and any basis diagonalises it. For the shear, $A - I = \begin{pmatrix} 0 & 1 \\ 0 & 0\end{pmatrix}$ has rank 1, so its null space — the eigenspace — is only the line through $(1, 0)$: geometric multiplicity 1, no second independent eigenvector, and no invertible $P$ (Figure 1(c)). The characteristic polynomial records only algebraic multiplicities, so it cannot tell the two matrices apart.

    </details>

11. **In the construction of the SVD from $A^\top A$, why are the vectors $u_i = Av_i/\sigma_i$ automatically orthonormal?**
    <details><summary>Answer</summary>

    Because $v_j$ is an eigenvector of $A^\top A$ with eigenvalue $\sigma_j^2$: $u_i^\top u_j = v_i^\top A^\top A v_j/(\sigma_i\sigma_j) = \sigma_j^2\, v_i^\top v_j/(\sigma_i\sigma_j)$. The $v_i$ are orthonormal, so this is $0$ for $i \ne j$ and $\sigma_i^2/\sigma_i^2 = 1$ for $i = j$. In the worked example of Section 4, $u_1 \cdot u_2 = \big(1\cdot 3 + 3\cdot(-1)\big)/10 = 0$.

    </details>

12. **Why does the first principal component both maximise the variance of the projected data and minimise the squared reconstruction error?**
    <details><summary>Answer</summary>

    For each centred point, Pythagoras gives $\lVert x_c\rVert^2 = (x_c\cdot w)^2 + \lVert x_c - (x_c\cdot w)\,w\rVert^2$. Summed over the points and divided by $n - 1$, the left side is the total variance $\operatorname{tr}(C)$, which is the same for every $w$; so variance kept plus squared reconstruction error$/(n-1)$ is a constant, and making one as large as possible makes the other as small as possible. For the five points of Section 6: $5 = 4.75 + 0.25$.

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

**`lesson.ipynb`** walks through every section with code you can edit: a grid-transform picture showing which directions a matrix only stretches; power iteration from scratch compared with `np.linalg.eig`, with its convergence rate measured; a two-state Markov chain and a six-page PageRank with damping, with both chains' convergence measured against their second eigenvalues and the PageRank of every page traced as the damping factor varies; the spectral decomposition of a symmetric matrix, its ellipse of constant quadratic form, a PSD check, and the Rayleigh quotient traced around the unit circle and sampled in 10,000 random directions; the SVD of a 2×2 matrix drawn as rotate–stretch–rotate (the unit circle mapped to an ellipse), then a 60×40 low-rank-plus-noise matrix whose scree plot, on log and linear axes, reveals its rank; compression of a synthetic 64×64 image at ranks 1, 2, 5, 10, 12 and 20 with the Eckart–Young error curve, and a measurement showing that truncating at the clean image's rank 12 removes noise; PCA from scratch on the Iris data matched against `sklearn.decomposition.PCA`, plus a demonstration of what unscaled features do to the loadings and variance shares; the pseudo-inverse built from the SVD, the minimum-norm solution as the shortest of a line of equally good fits, and ridge filter factors; and an experiment relating condition number to error amplification and to the number of gradient-descent steps, with the gradient-descent zig-zag drawn on the level sets, and a demonstration that centring and standardising features shrink the condition number.

**`exercises.ipynb`** has twelve exercises, nine of them with instant self-checks: eigenvalues of a 2×2 by the characteristic polynomial, spectral-theorem and PSD checks, truncated-SVD reconstruction error, power iteration with a Rayleigh-quotient estimate of the eigenvalue, PageRank with damping on a given graph (hidden answer), PCA from scratch versus scikit-learn, a conceptual question on unscaled PCA, the pseudo-inverse via SVD and the minimum-norm property, condition numbers and worst-case error amplification, ridge regression as singular-value shrinkage, a second conceptual question (computing an SVD through $A^\top A$), and a mini-project: a latent-factor recommender built by iterated truncated SVD on a synthetic ratings matrix with missing entries, evaluated by held-out RMSE against the rank $k$. `solutions.ipynb` contains reference implementations and explanations.
