# Module 02 — Matrices and Linear Maps

> After this module you can read a matrix as a *function*, multiply and compose such functions, tell when a system $A\mathbf{x} = \mathbf{b}$ has a solution (and how to get it without ever inverting a matrix), fit a least-squares line by hand, and recognise every one of these operations inside a neural network, a recommender, or a Markov chain.

## Why this matters for ML and AI

Almost everything a modern model does is a matrix multiplication. A dense layer computes $\mathbf{h} = W\mathbf{x} + \mathbf{b}$; the attention mechanism in a transformer computes $QK^\top$ and then multiplies the resulting weights into $V$; an embedding lookup is a one-hot vector times a matrix; a convolution can be unrolled into a (very sparse) matrix product. When NVIDIA reports that an H100 GPU delivers roughly a petaflop of tensor throughput, the number refers to one operation: multiplying matrices. When you read that GPT-class models have "175 billion parameters", most of those parameters are entries of weight matrices. If you understand what a matrix *does* to a vector — rotate it, stretch it, project it, squash a direction to zero — you understand what each layer of a network does to its input.

The same objects also carry the classical statistics that machine learning is built on. Ordinary least squares, the first model everybody fits, is the statement "project the target vector $\mathbf{y}$ onto the column space of the data matrix $X$", and its closed-form solution is a $d \times d$ linear system. Collinear features — two columns of $X$ that are nearly multiples of each other — make that system ill-conditioned, which is why ridge regression exists and why `statsmodels` warns you about the condition number. The dummy-variable trap in one-hot encoding is a rank deficiency. The "same map in different coordinates" idea behind change of basis is what PCA does to a dataset and what the Fourier transform does to a signal.

Finally, matrices describe *dynamics*. Powers of a transition matrix propagate a probability distribution through time; the stationary distribution of such a chain is Google's original PageRank score; the same machinery underlies Markov decision processes in reinforcement learning (Module 19) and the forward process of diffusion models (Module 18). This module is the grammar for all of that. Module 01 gave you vectors; here you learn what can be *done* to them.

## Learning objectives

By the end of this module you will be able to:

- Interpret $A\mathbf{x}$ in four equivalent ways (column combination, row dot-products, function, system of equations) and switch between them as needed.
- Multiply matrices by hand for small sizes, explain why $AB \neq BA$ in general, and state the $O(n^3)$ cost and why batched products dominate GPU workloads.
- Construct rotation, scaling, shear, reflection and projection matrices in 2-D, and read the determinant as a signed area-scaling factor.
- Recognise identity, diagonal, symmetric, orthogonal, triangular, permutation and positive (semi)definite matrices and state what each guarantees.
- Compute the rank, column space and null space of a small matrix, apply rank–nullity, and diagnose collinear features in a data matrix.
- Decide whether $A\mathbf{x} = \mathbf{b}$ has zero, one or infinitely many solutions; solve small systems by elimination; explain why `np.linalg.solve` beats `np.linalg.inv`.
- Derive the normal equations $A^\top A\,\hat{\mathbf{x}} = A^\top \mathbf{b}$ both geometrically and by calculus, fit a line to a handful of points by hand, and verify that the hat matrix is a projection.
- Express a vector in a new basis, form $P^{-1}AP$, and explain why "same map, different coordinates" is the key to eigendecomposition and PCA.
- Use adjacency and transition matrices to count paths and propagate distributions, and compute a stationary distribution.

## Prerequisites

You need [Module 01 — Vectors and geometry](../01-vectors-and-geometry/README.md): vectors as arrows and as data points, the dot product $\mathbf{u}\cdot\mathbf{v} = \sum_i u_i v_i$, the Euclidean norm $\lVert \mathbf{v} \rVert = \sqrt{\mathbf{v}\cdot\mathbf{v}}$, orthogonality ($\mathbf{u}\cdot\mathbf{v} = 0$), and the projection of one vector onto another. Recall in particular that the projection of $\mathbf{b}$ onto the line spanned by $\mathbf{a}$ is $\dfrac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{a}\cdot\mathbf{a}}\,\mathbf{a}$ — this module generalises that formula from a line to a whole subspace. In Python you should be comfortable with NumPy arrays, `.shape`, slicing, and the `@` operator.

## 1. Matrices as tables and as functions

**Intuition.** A matrix is a rectangular grid of numbers, and most people first meet it as a *table*: rows are patients, columns are lab measurements. That view is useful for data. But the view that unlocks everything else is the matrix as a *machine*: feed it a vector, and it returns another vector. The two views coexist peacefully. The data matrix $X$ with $n$ rows and $d$ columns is a table of $n$ observations; multiplying it by a weight vector $\mathbf{w}$ is a machine that turns each row into a single prediction.

**Definition.** An $m \times n$ matrix $A$ has $m$ rows and $n$ columns; its entry in row $i$, column $j$ is $A_{ij}$ (rows first, always). For a vector $\mathbf{x} \in \mathbb{R}^n$ the product $A\mathbf{x} \in \mathbb{R}^m$ is defined by

$$
(A\mathbf{x})_i = \sum_{j=1}^{n} A_{ij}\, x_j .
$$

There are four ways to read this one formula, and you will use all of them:

| View | What $A\mathbf{x}$ means | When it helps |
|---|---|---|
| **Column picture** | $A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_n \mathbf{a}_n$, a linear combination of the columns $\mathbf{a}_j$ with weights $x_j$ | Existence of solutions, column space, least squares |
| **Row picture** | $(A\mathbf{x})_i = \mathbf{r}_i \cdot \mathbf{x}$, the dot product of row $i$ with $\mathbf{x}$ | Predictions per data point, neuron activations |
| **Function** | $\mathbf{x} \mapsto A\mathbf{x}$ is a map $\mathbb{R}^n \to \mathbb{R}^m$ | Geometry, composition, neural network layers |
| **System of equations** | $A\mathbf{x} = \mathbf{b}$ is $m$ simultaneous linear equations in $n$ unknowns | Solving, elimination, rank |

A map $T$ is **linear** if $T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$ for all vectors and scalars. Every matrix defines a linear map, and — the key fact — *every* linear map $\mathbb{R}^n \to \mathbb{R}^m$ is given by exactly one matrix: its $j$-th column is $T(\mathbf{e}_j)$, the image of the $j$-th standard basis vector. Linearity means knowing what happens to the $n$ basis vectors tells you what happens to everything. Note what linearity rules out: $T(\mathbf{0}) = \mathbf{0}$ always, so a translation $\mathbf{x} \mapsto \mathbf{x} + \mathbf{c}$ is *not* linear (it is *affine*). That is why a neural network layer $W\mathbf{x} + \mathbf{b}$ needs a separate bias term.

**Worked example.** Let

$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \qquad \mathbf{x} = \begin{pmatrix} 2 \\ -1 \end{pmatrix}.
$$

Row picture: first entry $1\cdot 2 + 2\cdot(-1) = 0$; second entry $3 \cdot 2 + 4\cdot(-1) = 2$. So $A\mathbf{x} = (0, 2)^\top$.
Column picture: $2\begin{pmatrix}1\\3\end{pmatrix} + (-1)\begin{pmatrix}2\\4\end{pmatrix} = \begin{pmatrix}2-2\\6-4\end{pmatrix} = \begin{pmatrix}0\\2\end{pmatrix}$. Same answer, different story: we walked two steps along column one and one step back along column two.
Function picture: $A$ sends $\mathbf{e}_1 = (1,0)^\top$ to $(1,3)^\top$ and $\mathbf{e}_2 = (0,1)^\top$ to $(2,4)^\top$ — those are its columns. Everything else follows by linearity.

**Real-life example (healthcare).** A hospital stores a data matrix $X$ with $n = 5{,}000$ patients (rows) and $d = 3$ features (columns): age in decades, systolic blood pressure in units of 10 mmHg, and BMI. A risk score is $\mathbf{s} = X\mathbf{w}$ with $\mathbf{w} = (0.3, 0.5, 0.2)^\top$. In the row picture, patient $i$ gets the score $\mathbf{r}_i \cdot \mathbf{w}$ — a weighted sum of their three measurements; for a 60-year-old with 140 mmHg and BMI 30 this is $0.3\cdot 6 + 0.5 \cdot 14 + 0.2 \cdot 30 = 1.8 + 7 + 6 = 14.8$. In the column picture, the whole score vector is $0.3\,(\text{age column}) + 0.5\,(\text{BP column}) + 0.2\,(\text{BMI column})$: the score lives in the span of the three feature columns, and no choice of weights can produce a score pattern outside that span. That limitation is exactly what least squares will have to live with in Section 7.

**In ML/AI this is …** the forward pass of a linear layer: `nn.Linear` computes $\mathbf{h} = W\mathbf{x} + \mathbf{b}$, and in the row picture each neuron $h_i$ is the dot product of its weight row with the input. The data matrix convention $X \in \mathbb{R}^{n \times d}$ (rows = samples) is why scikit-learn's `fit(X, y)` and `predict(X)` compute `X @ coef_ + intercept_`. An embedding lookup `E[token_id]` is the column picture with a one-hot $\mathbf{x}$: $E^\top \mathbf{e}_k$ selects column $k$.

## 2. Matrix multiplication

**Intuition.** If a matrix is a machine, then the product $AB$ is "run machine $B$, then run machine $A$ on the result". Composition of functions is written right-to-left, and so is matrix multiplication: $(AB)\mathbf{x} = A(B\mathbf{x})$. Everything about the product — its shape rules, its non-commutativity, its cost — follows from this one reading.

**Definition.** For $A \in \mathbb{R}^{m \times p}$ and $B \in \mathbb{R}^{p \times n}$ the product $C = AB \in \mathbb{R}^{m \times n}$ has entries

$$
C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj} \quad = \quad (\text{row } i \text{ of } A) \cdot (\text{column } j \text{ of } B).
$$

The inner dimensions must match ($p$ with $p$); the outer ones give the result's shape ($m \times n$). Equivalent readings: column $j$ of $AB$ is $A$ applied to column $j$ of $B$; row $i$ of $AB$ is row $i$ of $A$ times $B$; and $AB = \sum_k \mathbf{a}_k \mathbf{b}_k^\top$, a sum of $p$ rank-one outer products. Properties: associative, $(AB)C = A(BC)$; distributive; $(AB)^\top = B^\top A^\top$ (order flips); but **not commutative**: $AB \neq BA$ in general, and $BA$ may not even be defined.

Cost: each of the $mn$ entries needs $p$ multiply-adds, so a square $n \times n$ product costs $\Theta(n^3)$ flops. (Strassen-type algorithms are asymptotically faster but rarely used in practice.) The reason GPUs are so good at this is that the $n^3$ operations are embarrassingly parallel and every loaded number is reused $n$ times — arithmetic intensity is high. In NumPy, `A @ B` on two 2-D arrays is the matrix product; `A * B` is the *elementwise* (Hadamard) product and needs identical shapes. With a 3-D array of shape `(batch, m, p)` and a matrix of shape `(p, n)`, `@` broadcasts over the leading batch dimension and returns `(batch, m, n)` — one call, many products.

**Worked example.** With

$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}:
$$

$AB$: entry $(1,1) = 1\cdot 0 + 2 \cdot 1 = 2$; $(1,2) = 1\cdot 1 + 2\cdot 0 = 1$; $(2,1) = 3\cdot 0 + 4 \cdot 1 = 4$; $(2,2) = 3 \cdot 1 + 4 \cdot 0 = 3$. So $AB = \begin{pmatrix} 2 & 1 \\ 4 & 3 \end{pmatrix}$ — $B$ swaps the *columns* of $A$.
$BA$: $(1,1) = 0\cdot 1 + 1 \cdot 3 = 3$; $(1,2) = 0 \cdot 2 + 1 \cdot 4 = 4$; $(2,1) = 1$; $(2,2) = 2$. So $BA = \begin{pmatrix} 3 & 4 \\ 1 & 2 \end{pmatrix}$ — $B$ swaps the *rows* of $A$. Different matrices: multiplying by a permutation on the right permutes columns, on the left permutes rows.

**Real-life example (finance).** A bank quotes conversion rates as matrices. $B$ converts amounts in (EUR, GBP) into (USD, CHF): $B = \begin{pmatrix} 1.08 & 1.27 \\ 0.95 & 1.12 \end{pmatrix}$, so a position of $\mathbf{x}$ = (100 EUR, 50 GBP) becomes $B\mathbf{x} = (108 + 63.5,\; 95 + 56)^\top = (171.5\text{ USD}, 151\text{ CHF})^\top$. A second table $A$ then converts (USD, CHF) into (JPY, SGD). Converting a whole portfolio in two steps is $A(B\mathbf{x})$; precomputing $AB$ gives a single EUR/GBP → JPY/SGD table. For a portfolio of $10^6$ positions, computing $AB$ once (a $2\times2$ product) and applying it saves half the work compared with applying $B$ and then $A$ to every position — the same reason a compiler fuses consecutive linear layers.

**In ML/AI this is …** the reason a stack of linear layers with no nonlinearity is useless: $W_2 (W_1 \mathbf{x}) = (W_2 W_1)\mathbf{x}$ collapses to a single matrix, so depth buys nothing without an activation function in between. It is also the attention score computation $QK^\top$ (shape `(seq, d) @ (d, seq)` → `(seq, seq)`), the batched `torch.bmm` behind multi-head attention, and the FLOP budget of a forward pass: a dense layer with input $d_{in}$ and output $d_{out}$ on a batch of $n$ costs about $2\,n\,d_{in}\,d_{out}$ flops.

## 3. Geometric transformations and the determinant

**Intuition.** Draw the unit square with corners $(0,0), (1,0), (1,1), (0,1)$ and apply a $2\times 2$ matrix to every corner. The square becomes a parallelogram whose edges are the two columns of the matrix. Rotations turn it, scalings stretch it, shears slant it, reflections flip it over, and projections flatten it onto a line. The determinant is the *area* of that parallelogram, with a sign that records whether the orientation was flipped. When the determinant is zero the parallelogram has collapsed into a line segment — a whole dimension of information has been destroyed and cannot be recovered.

**Definition.** The standard $2 \times 2$ building blocks are

$$
R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix},\quad
S = \begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix},\quad
H_k = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix},\quad
F = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix},\quad
P_{\mathbf{u}} = \mathbf{u}\mathbf{u}^\top \ (\lVert\mathbf{u}\rVert = 1),
$$

for rotation by $\theta$ counter-clockwise, axis scaling, horizontal shear, reflection in the $x$-axis, and orthogonal projection onto the unit vector $\mathbf{u}$. The **determinant** of a $2\times2$ matrix is $\det\begin{pmatrix} a & b \\ c & d\end{pmatrix} = ad - bc$; for a $3\times 3$ it is the signed volume of the parallelepiped spanned by the columns, computed by cofactor expansion along any row. Key facts: $\det(AB) = \det(A)\det(B)$, $\det(A^\top) = \det A$, $\det(A^{-1}) = 1/\det A$, $\det(cA) = c^n \det A$ for an $n\times n$ matrix, and

$$
\det A = 0 \iff A \text{ is singular (not invertible)} \iff A \text{ squashes some direction to } \mathbf{0}.
$$

A matrix with $\lvert\det A\rvert = 1$ preserves volume; with $\det A < 0$ it reverses orientation.

**Worked example.** Rotation by $90^\circ$: $R = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$, $\det R = 0\cdot 0 - (-1)(1) = 1$: area and orientation preserved. Shear $H_2 = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$ sends the square to the parallelogram with corners $(0,0),(1,0),(3,1),(2,1)$; $\det H_2 = 1 - 0 = 1$, so the slanted shape still has area 1 (base 1, height 1). Scaling $S = \mathrm{diag}(2, 3)$ has $\det = 6$: area multiplied by six. Projection onto $\mathbf{u} = (1,0)^\top$: $P = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$, $\det P = 0$: the square becomes the segment from $(0,0)$ to $(1,0)$ with zero area, and every point $(x, y)$ with the same $x$ lands on the same image — the $y$ coordinate is lost. Composition: rotating then scaling gives $SR$ with $\det(SR) = \det S \det R = 6$.

**Real-life example (computer vision).** Image augmentation libraries build their random affine transforms exactly this way. A random rotation of $\pm 15^\circ$, a horizontal flip (reflection), a small shear and a zoom of $0.9$–$1.1$ are each a $2\times2$ matrix (plus a translation, handled in homogeneous coordinates as a $3\times 3$ matrix); they are composed into one matrix and applied to every pixel coordinate. The determinant tells you how much the object grew: a zoom of $1.1$ in both axes has $\det = 1.21$, so the object covers 21 % more pixels. In robotics the same matrices express a camera's coordinate frame relative to the robot base, and a chain of joints is a product of rotation matrices — each with determinant $+1$ — so the composed map is also a pure rotation.

**In ML/AI this is …** the `RandomAffine` / `RandomRotation` transforms in computer-vision pipelines, the rotary positional encoding in modern transformers (RoPE applies a $2\times2$ rotation $R_{m\theta}$ to each pair of embedding dimensions, Module 17), and the change-of-variables rule in normalising flows, where the log-likelihood picks up a $\log\lvert\det J\rvert$ term because the Jacobian's determinant is how much a transformation inflates or compresses probability volume.

## 4. Special matrices

**Intuition.** Most matrices you meet in practice are not generic. They have structure — zeros in the right places, symmetry, columns of unit length — and each structure is a *promise*: a diagonal matrix promises that the coordinates do not interact; an orthogonal matrix promises that lengths and angles survive; a triangular matrix promises that you can solve a system by back-substitution in $O(n^2)$ instead of $O(n^3)$. Learning to spot the structure is learning what you can get away with.

**Definitions.** For a square $n \times n$ matrix:

| Name | Condition | What it guarantees |
|---|---|---|
| Identity $I$ | $I_{ij} = 1$ if $i = j$, else $0$ | $I\mathbf{x} = \mathbf{x}$; $AI = IA = A$ |
| Diagonal $D$ | $D_{ij} = 0$ for $i \neq j$ | Scales axis $i$ by $D_{ii}$; $D^{-1} = \mathrm{diag}(1/D_{ii})$; $\det D = \prod_i D_{ii}$ |
| Symmetric | $A^\top = A$ | Real eigenvalues, orthogonal eigenvectors (Module 03); arises as $X^\top X$, covariance, kernels |
| Orthogonal $Q$ | $Q^\top Q = Q Q^\top = I$ | Columns are orthonormal; $\lVert Q\mathbf{x}\rVert = \lVert\mathbf{x}\rVert$; $Q^{-1} = Q^\top$; $\det Q = \pm 1$ |
| Upper / lower triangular | Zeros below / above the diagonal | Solve by back-/forward-substitution; $\det = \prod_i A_{ii}$ |
| Permutation $P$ | One 1 per row and column, else 0 | Reorders coordinates; orthogonal; $\det = \pm 1$ |
| Positive semidefinite (PSD) | Symmetric and $\mathbf{x}^\top A \mathbf{x} \ge 0$ for all $\mathbf{x}$ | A valid covariance / kernel / Hessian at a minimum; eigenvalues $\ge 0$. Positive definite if $> 0$ for $\mathbf{x} \neq \mathbf{0}$ |

The **transpose** swaps rows and columns, $(A^\top)_{ij} = A_{ji}$, and obeys $(A + B)^\top = A^\top + B^\top$, $(AB)^\top = B^\top A^\top$, $(A^\top)^\top = A$, and $\mathbf{u}^\top \mathbf{v} = \mathbf{u}\cdot\mathbf{v}$. For any $A$ (not necessarily square), $A^\top A$ is symmetric and PSD: $\mathbf{x}^\top A^\top A\mathbf{x} = (A\mathbf{x})^\top (A\mathbf{x}) = \lVert A\mathbf{x}\rVert^2 \ge 0$. That two-line argument is used constantly.

Why orthogonal matrices preserve length: $\lVert Q\mathbf{x}\rVert^2 = (Q\mathbf{x})^\top(Q\mathbf{x}) = \mathbf{x}^\top Q^\top Q\,\mathbf{x} = \mathbf{x}^\top \mathbf{x} = \lVert\mathbf{x}\rVert^2$. Dot products are preserved by the same calculation, so angles are too. Rotations and reflections are the orthogonal matrices.

**Worked example.** $Q = \frac{1}{\sqrt 2}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$ (rotation by $45^\circ$). Check: $Q^\top Q = \frac12 \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix} = \frac12\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = I$. Apply to $\mathbf{x} = (3, 4)^\top$ with $\lVert\mathbf{x}\rVert = 5$: $Q\mathbf{x} = \frac{1}{\sqrt2}(3 - 4,\; 3 + 4)^\top = \frac{1}{\sqrt 2}(-1, 7)^\top$, whose squared norm is $\frac12(1 + 49) = 25$. Length preserved. Now $A = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$ is symmetric; is it positive definite? $\mathbf{x}^\top A\mathbf{x} = 2x_1^2 + 2x_1x_2 + 2x_2^2 = (x_1 + x_2)^2 + x_1^2 + x_2^2 > 0$ for $\mathbf{x} \neq \mathbf{0}$. Yes. By contrast $\begin{pmatrix} 1 & 2 \\ 2 & 1\end{pmatrix}$ gives $\mathbf{x}^\top A \mathbf{x} = -2$ at $\mathbf{x} = (1, -1)^\top$, so it is symmetric but *not* PSD — it cannot be a covariance matrix.

**Real-life example (manufacturing).** A quality-control lab measures three dimensions of a machined part and reports their covariance matrix $\Sigma$. $\Sigma$ must be symmetric (the covariance of length with width equals that of width with length) and PSD (the variance of any weighted combination $\mathbf{w}^\top \mathbf{x}$ is $\mathbf{w}^\top \Sigma \mathbf{w}$, which cannot be negative). When an engineer hand-edits a few entries "to make them look right" and the result has a negative eigenvalue, the downstream Gaussian model produces `nan` likelihoods. The fix is to project back to the PSD cone — a routine `scipy`/`sklearn` step (`sklearn.covariance` shrinkage estimators guarantee a PSD result).

**In ML/AI this is …** the orthogonal initialisation of recurrent weight matrices (`torch.nn.init.orthogonal_`), chosen because repeated multiplication by an orthogonal matrix neither explodes nor vanishes the signal; the symmetric PSD Gram matrix $X^\top X$ in ridge regression and the kernel matrix $K$ in Gaussian processes and SVMs; the diagonal preconditioners in Adam and RMSProp, which scale each parameter axis independently; and the permutation matrices implicit in shuffling a batch or in the pivoting inside `np.linalg.solve`.

## 5. Column space, null space and rank

**Intuition.** The column space of $A$ is every output the machine can produce: all vectors $A\mathbf{x}$. The null space is every input the machine destroys: all $\mathbf{x}$ with $A\mathbf{x} = \mathbf{0}$. Rank counts how many *genuinely different* directions the columns point in. A $3 \times 3$ matrix whose third column is twice the first plus the second has only two independent columns; it maps $\mathbb{R}^3$ onto a plane and kills a whole line of inputs. Information is lost, and the loss is exactly the null space.

**Definitions.** For $A \in \mathbb{R}^{m\times n}$:

- **Column space** $\mathcal{C}(A) = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\} \subseteq \mathbb{R}^m$, the span of the columns. $A\mathbf{x} = \mathbf{b}$ is solvable iff $\mathbf{b} \in \mathcal{C}(A)$.
- **Null space** $\mathcal{N}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\} \subseteq \mathbb{R}^n$. If $\mathbf{x}_p$ solves $A\mathbf{x} = \mathbf{b}$, so does $\mathbf{x}_p + \mathbf{z}$ for every $\mathbf{z} \in \mathcal{N}(A)$.
- **Rank** $r = \operatorname{rank} A = \dim \mathcal{C}(A)$ = number of linearly independent columns = number of linearly independent rows (column rank equals row rank — the "pivot" count in elimination). Always $r \le \min(m, n)$; $A$ has **full rank** when equality holds, otherwise it is **rank deficient**.
- **Rank–nullity theorem:** $\operatorname{rank} A + \dim\mathcal{N}(A) = n$ (the number of columns). Every column either contributes a new direction or contributes a null-space dimension.

For square $A$: full rank $\iff$ invertible $\iff \det A \neq 0 \iff \mathcal{N}(A) = \{\mathbf{0}\}$. A rank-$r$ matrix can be written as a sum of $r$ outer products $\mathbf{u}_k\mathbf{v}_k^\top$ and no fewer — this is the seed of low-rank approximation in Module 03. Numerically, rank is decided by counting singular values above a tolerance (`np.linalg.matrix_rank`); a column that is 99.9 % a combination of the others is "numerically almost rank deficient", which is where the condition number (Section 6) takes over.

**Worked example.** Take

$$
M = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 0 & 1 \end{pmatrix}.
$$

Column 3 equals column 1 plus column 2: $(1, 2, 1) + (2, 4, 0) = (3, 6, 1)$. So the columns span at most a plane. Columns 1 and 2 are not multiples of each other (the third entries $1$ and $0$ differ), so $\operatorname{rank} M = 2$ and the column space is the plane spanned by $(1,2,1)^\top$ and $(2,4,0)^\top$. By rank–nullity, $\dim\mathcal{N}(M) = 3 - 2 = 1$. The relation $\mathbf{c}_1 + \mathbf{c}_2 - \mathbf{c}_3 = \mathbf{0}$ says $M (1, 1, -1)^\top = \mathbf{0}$; check row 3: $1 + 0 - 1 = 0$. The null space is the line $t\,(1, 1, -1)^\top$. Consequently $M\mathbf{x} = \mathbf{b}$ has either no solution (if $\mathbf{b}$ is off the plane) or a whole line of them. Row 2 is twice row 1, confirming the rank from the row side.

**Real-life example (e-commerce).** An online shop encodes the "device" feature of each session with one-hot columns `is_mobile`, `is_desktop`, `is_tablet`, and its regression design matrix also contains an all-ones intercept column. Since every session is exactly one device, `is_mobile + is_desktop + is_tablet = 1` = the intercept column for every row. The four columns span only three dimensions: the design matrix is rank deficient, $X^\top X$ is singular, and the normal equations have infinitely many solutions — you can add any constant to the three device coefficients and subtract it from the intercept without changing a single prediction. This is the **dummy-variable trap**; the fix is to drop one category (`drop_first=True` in pandas, `drop='first'` in `OneHotEncoder`) or to regularise. The same thing happens with redundant sensors: two thermometers reading the same temperature in °C and °F give exactly collinear features (one is an affine function of the other).

**In ML/AI this is …** the warning `LinAlgWarning: Ill-conditioned matrix` from scikit-learn when features are collinear, the reason ridge regression adds $\lambda I$ to $X^\top X$ (it raises the rank to full), the "rank" $r$ in LoRA fine-tuning, where a weight update $\Delta W = BA$ with $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r \times k}$ is forced to have rank $\le r$ (Module 17), and the rank-collapse diagnostics used to detect when a transformer's attention outputs lose dimensionality.

## 6. Solving $A\mathbf{x} = \mathbf{b}$

**Intuition.** Solving a linear system is asking: which input did the machine turn into $\mathbf{b}$? If the machine is invertible there is exactly one answer. If it squashed a direction, either no input works (the target is off the column space) or infinitely many do (any null-space component is invisible). Gaussian elimination answers the question by subtracting multiples of equations from one another until the system is triangular, then solving from the bottom up. The inverse matrix is a *concept* for reasoning; computing it explicitly is almost always the wrong move in code.

**Definitions and facts.** For square $A \in \mathbb{R}^{n\times n}$, exactly one of the following holds: $A$ is invertible, and $A\mathbf{x} = \mathbf{b}$ has the unique solution $\mathbf{x} = A^{-1}\mathbf{b}$ for every $\mathbf{b}$; or $A$ is singular, and the system has no solution or infinitely many depending on whether $\mathbf{b} \in \mathcal{C}(A)$. For a $2\times 2$ matrix,

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix},
$$

and $(AB)^{-1} = B^{-1}A^{-1}$ (order flips, like the transpose). Gaussian elimination with partial pivoting factors $PA = LU$ (permutation, unit lower triangular, upper triangular) in about $\tfrac{2}{3}n^3$ flops; afterwards each new right-hand side costs only $O(n^2)$ via two triangular solves. Computing $A^{-1}$ explicitly costs roughly $2n^3$ — three times as much — and then multiplying $A^{-1}\mathbf{b}$ adds rounding error that the triangular solves avoid. Rule: write $A^{-1}\mathbf{b}$ in maths, call `np.linalg.solve(A, b)` in code (or `scipy.linalg.lu_factor` + `lu_solve` for many right-hand sides).

How much can rounding hurt? The **condition number** $\kappa(A) = \lVert A\rVert\,\lVert A^{-1}\rVert$ (equal to the ratio of the largest to the smallest singular value, Module 03) bounds the damage: a relative perturbation $\varepsilon$ in $\mathbf{b}$ can change the solution by a relative amount up to $\kappa(A)\,\varepsilon$. With double precision ($\varepsilon \approx 10^{-16}$), a matrix with $\kappa = 10^{10}$ leaves you about six trustworthy digits; $\kappa = 10^{16}$ leaves none. Singular means $\kappa = \infty$; "nearly singular" means $\kappa$ is huge.

**Worked example (2×2).** Solve $\begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}\mathbf{x} = \begin{pmatrix} 5 \\ 10 \end{pmatrix}$. Elimination: subtract $\tfrac12 \times$ row 1 from row 2 to get $0\,x_1 + 2.5\,x_2 = 7.5$, so $x_2 = 3$; back-substitute: $2x_1 + 3 = 5$, $x_1 = 1$. Via the inverse formula: $\det = 6 - 1 = 5$, $A^{-1} = \tfrac15\begin{pmatrix} 3 & -1 \\ -1 & 2\end{pmatrix}$, $A^{-1}\mathbf{b} = \tfrac15 (15 - 10,\; -5 + 20)^\top = (1, 3)^\top$. Same answer.

**Worked example (3×3, conditioning).** Consider $A = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001\end{pmatrix}$. With $\mathbf{b} = (2, 2.0001)^\top$ the solution is $(1, 1)^\top$. Change the last digit of $\mathbf{b}$ to $2.0002$ — a relative change of $5\times 10^{-5}$ — and the solution becomes $(0, 2)^\top$: a 100 % change. Indeed $\kappa(A) \approx 4\times 10^{4}$, and $4\times10^4 \cdot 5\times 10^{-5} \approx 2$, matching the observed amplification. The two columns are nearly parallel, so the parallelogram is a sliver and tiny nudges to $\mathbf{b}$ slide the solution a long way along it.

**Real-life example (manufacturing / food).** A juice company blends three concentrates to hit a target of 7 g sugar, 8 g fibre and 12 g vitamin C per serving. Per decilitre, concentrate 1 supplies (1, 2, 1) of those nutrients, concentrate 2 supplies (1, 1, 3) and concentrate 3 supplies (2, 1, 1). With $\mathbf{x}$ the decilitres of each, the nutrient totals are $A\mathbf{x}$ with the concentrates as columns:

$$
\begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 1 \\ 1 & 3 & 1 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} = \begin{pmatrix} 7 \\ 8 \\ 12 \end{pmatrix}.
$$

Eliminate: row 2 − 2·row 1 gives $-x_2 - 3x_3 = -6$; row 3 − row 1 gives $2x_2 - x_3 = 5$. From the first of these, $x_2 = 6 - 3x_3$; substitute: $12 - 6x_3 - x_3 = 5$, so $x_3 = 1$, $x_2 = 3$, and from row 1, $x_1 = 7 - 3 - 2 = 2$. The recipe is 2 dl, 3 dl, 1 dl. The same structure — "which combination of ingredients hits the targets" — is balancing a chemical equation (atoms as rows, molecules as columns), Kirchhoff's laws in circuit analysis (nodes as rows, currents as unknowns) and portfolio construction (factor exposures as rows, assets as columns).

**In ML/AI this is …** the closed-form ridge and OLS solvers (`solver='cholesky'` in `sklearn.linear_model.Ridge` solves $(X^\top X + \lambda I)\mathbf{w} = X^\top \mathbf{y}$ with a triangular factorisation, never an inverse), the Newton step $H\,\Delta\boldsymbol\theta = -\nabla L$ in second-order optimisers and in `scipy.optimize` (Module 05), the Gaussian-process posterior mean $K_*\,K^{-1}\mathbf{y}$ computed by a Cholesky solve (Module 09), and the policy-evaluation step of an MDP, $(I - \gamma P_\pi)\mathbf{v} = \mathbf{r}$ (Module 19). The condition-number warning you get from a badly scaled feature matrix is the practical reason for feature standardisation.

## 7. Least squares and the normal equations

**Intuition.** Four data points almost never lie exactly on a line, so the system "line passes through all four" has more equations than unknowns and no exact solution. Least squares asks instead for the line whose predictions are *closest* to the observations in the Euclidean sense. Geometrically: the vector of observations $\mathbf{y}$ sticks out of the plane of achievable predictions (the column space of $A$); the best achievable prediction is the *shadow* of $\mathbf{y}$ on that plane, and the residual — the part of $\mathbf{y}$ left over — is perpendicular to the plane. "Perpendicular to every column" is one short equation, and that equation is the normal equations.

**Definition and derivation.** Given $A \in \mathbb{R}^{m\times n}$ with $m > n$ (more observations than parameters) and full column rank, the least-squares solution is

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \lVert A\mathbf{x} - \mathbf{y} \rVert^2 .
$$

*Geometric derivation.* The residual $\mathbf{r} = \mathbf{y} - A\hat{\mathbf{x}}$ must be orthogonal to every column of $A$ (otherwise moving along that column would shrink it). Orthogonal to every column means $A^\top \mathbf{r} = \mathbf{0}$, i.e. $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}$, which rearranges to the **normal equations**

$$
A^\top A\,\hat{\mathbf{x}} = A^\top \mathbf{y}, \qquad \hat{\mathbf{x}} = (A^\top A)^{-1}A^\top\mathbf{y}.
$$

*Calculus derivation (preview of Module 04).* Expand $L(\mathbf{x}) = \lVert A\mathbf{x} - \mathbf{y}\rVert^2 = \mathbf{x}^\top A^\top A\mathbf{x} - 2\mathbf{y}^\top A\mathbf{x} + \mathbf{y}^\top\mathbf{y}$; the gradient is $\nabla L = 2A^\top A\mathbf{x} - 2A^\top\mathbf{y}$; set it to zero and you get the same equations. $A^\top A$ is PSD (Section 4), so this stationary point is a minimum.

The fitted values are $\hat{\mathbf{y}} = A\hat{\mathbf{x}} = H\mathbf{y}$ with the **hat matrix** $H = A(A^\top A)^{-1}A^\top$. $H$ is the orthogonal projector onto $\mathcal{C}(A)$: it is symmetric ($H^\top = H$) and idempotent ($H^2 = H$ — projecting twice changes nothing), its trace equals the rank $n$ (the number of parameters, which is why residual degrees of freedom are $m - n$), and $I - H$ projects onto the orthogonal complement, giving the residuals. In code, do *not* form $(A^\top A)^{-1}$: `np.linalg.lstsq(A, y)` uses an SVD and `np.linalg.qr` gives $A = QR$ so that $\hat{\mathbf{x}}$ solves the triangular system $R\hat{\mathbf{x}} = Q^\top\mathbf{y}$. Both avoid squaring the condition number — $\kappa(A^\top A) = \kappa(A)^2$ — which is the practical weakness of the normal equations.

**Worked example.** Fit $y = c + m\,t$ to the points $(t, y) = (0, 1), (1, 2), (2, 2), (3, 4)$. The design matrix has an intercept column and a $t$ column:

$$
A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 1 \\ 2 \\ 2 \\ 4 \end{pmatrix}, \quad
A^\top A = \begin{pmatrix} 4 & 6 \\ 6 & 14 \end{pmatrix}, \quad A^\top \mathbf{y} = \begin{pmatrix} 9 \\ 18 \end{pmatrix}.
$$

($A^\top A$: $4$ ones; $\sum t = 6$; $\sum t^2 = 0+1+4+9 = 14$. $A^\top\mathbf{y}$: $\sum y = 9$; $\sum t y = 0 + 2 + 4 + 12 = 18$.) Solve $4c + 6m = 9$, $6c + 14m = 18$: multiply the first by $1.5$ to get $6c + 9m = 13.5$; subtract from the second: $5m = 4.5$, $m = 0.9$; then $4c = 9 - 5.4 = 3.6$, $c = 0.9$. The line is $\hat y = 0.9 + 0.9\,t$. Fitted values $(0.9, 1.8, 2.7, 3.6)$, residuals $(0.1, 0.2, -0.7, 0.4)$. Check orthogonality: residuals sum to $0.1 + 0.2 - 0.7 + 0.4 = 0$ (orthogonal to the ones column) and $\sum t\, r = 0 + 0.2 - 1.4 + 1.2 = 0$ (orthogonal to the $t$ column). The residual sum of squares is $0.01 + 0.04 + 0.49 + 0.16 = 0.70$, and no other line does better.

**Real-life example (healthcare / pharmacology).** In a dose-finding study, four groups receive 0, 10, 20 and 30 mg of a drug and show mean biomarker reductions of 1, 2, 2 and 4 units (same numbers as above, with $t$ in units of 10 mg). The fitted slope of $0.9$ units per 10 mg is the estimated dose–response, and the intercept $0.9$ is the placebo effect. The hat matrix tells you how much each observation influences its own fitted value: its diagonal entries (the *leverages*) for this design are $0.7, 0.3, 0.3, 0.7$ — the outermost doses have the most pull, which is why dose-ranging designs put replicates at the extremes. The identical calculation calibrates a sensor (voltage readings against known reference concentrations) or converts a thermistor's resistance to temperature.

**In ML/AI this is …** `sklearn.linear_model.LinearRegression`, which calls `scipy.linalg.lstsq` under the hood; the closed form that gradient descent on mean-squared error converges to (Module 05 shows that the Hessian of this loss is $2A^\top A/m$, so its conditioning sets the learning-rate budget); the "OLS as projection" view developed fully in [Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md); and the last-layer readout in many self-supervised evaluation protocols, where a linear probe is just least squares (or logistic regression) on frozen features.

## 8. Change of basis and similarity

**Intuition.** A vector is a geometric object; its *coordinates* depend on which rulers you lay down. The point two steps east and one step north is $(2, 1)$ in the compass basis but $(1.5, 0.5)$ in a basis whose first axis points north-east and second points south-east. Nothing about the point changed — only the description. The same is true of linear maps: a shear that looks messy in standard coordinates can look like a plain scaling in a cleverly chosen basis. Finding that basis is the entire business of eigendecomposition (Module 03), and PCA is the special case where the clever basis is chosen from the data itself.

**Definition.** Let the columns of $P \in \mathbb{R}^{n\times n}$ be a basis $\mathbf{p}_1, \dots, \mathbf{p}_n$ (so $P$ is invertible). A vector with standard coordinates $\mathbf{x}$ has coordinates $\mathbf{c}$ in the new basis when

$$
\mathbf{x} = P\mathbf{c} = c_1\mathbf{p}_1 + \dots + c_n\mathbf{p}_n, \qquad\text{so}\qquad \mathbf{c} = P^{-1}\mathbf{x}.
$$

If $P$ is orthogonal this is simply $\mathbf{c} = P^\top\mathbf{x}$: each new coordinate is a dot product with a basis vector, which is exactly the projection formula from Module 01. A linear map with standard matrix $A$ has, in the new coordinates, the matrix

$$
A' = P^{-1} A P \qquad (\text{read right to left: convert } \mathbf{c} \to \mathbf{x},\ \text{apply } A,\ \text{convert back}).
$$

$A$ and $A'$ are called **similar**. Similar matrices share everything that does not depend on coordinates: determinant, trace, rank, eigenvalues. The goal of diagonalisation is to find $P$ such that $P^{-1}AP = D$ is diagonal — then the map is "stretch along $\mathbf{p}_1$ by $d_1$, along $\mathbf{p}_2$ by $d_2$, …", and $A^k = PD^kP^{-1}$ costs almost nothing.

**Worked example.** Take the basis $\mathbf{p}_1 = (1, 1)^\top$, $\mathbf{p}_2 = (1, -1)^\top$, so $P = \begin{pmatrix} 1 & 1 \\ 1 & -1\end{pmatrix}$ with $P^{-1} = \tfrac12\begin{pmatrix} 1 & 1 \\ 1 & -1\end{pmatrix}$ (check: $\det P = -2$, use the 2×2 formula). The vector $\mathbf{x} = (3, 1)^\top$ has new coordinates $\mathbf{c} = P^{-1}\mathbf{x} = \tfrac12(4, 2)^\top = (2, 1)^\top$, and indeed $2\,(1,1) + 1\,(1,-1) = (3, 1)$. Now the symmetric map $A = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$: $AP = \begin{pmatrix} 3 & 1 \\ 3 & -1\end{pmatrix}$, and $P^{-1}AP = \tfrac12\begin{pmatrix} 1 & 1 \\ 1 & -1\end{pmatrix}\begin{pmatrix} 3 & 1 \\ 3 & -1 \end{pmatrix} = \tfrac12\begin{pmatrix} 6 & 0 \\ 0 & 2\end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & 1\end{pmatrix}$. In the $(\mathbf{p}_1, \mathbf{p}_2)$ basis, $A$ is a pure scaling: ×3 along the diagonal direction, ×1 along the anti-diagonal. Trace check: $2 + 2 = 3 + 1$; determinant check: $4 - 1 = 3 \cdot 1$. We have just diagonalised $A$ by hand; Module 03 explains how to *find* such a $P$ in general.

**Real-life example (physics / signal processing).** An accelerometer in a phone reports acceleration in the phone's own axes. To know whether the phone is being lifted or slid, the software must re-express the vector in the room's axes — a change of basis by a rotation matrix $R$ obtained from the gyroscope: $\mathbf{a}_{\text{room}} = R\,\mathbf{a}_{\text{phone}}$. In audio, the discrete Fourier transform is a change of basis from "amplitude at each time sample" to "amplitude of each frequency": the new basis vectors are sampled sinusoids, the transform matrix is (up to scaling) orthogonal, and a low-pass filter that is a dense convolution in time becomes a *diagonal* matrix in frequency — the same map, made simple by the right coordinates.

**In ML/AI this is …** PCA (`sklearn.decomposition.PCA`): `transform(X)` computes $(X - \bar{\mathbf{x}})\,W$ where the columns of $W$ are the principal directions, i.e. the data re-expressed in the covariance eigenbasis (Module 03); whitening, where $P = \Sigma^{-1/2}$ makes the covariance the identity; the diagonalisation of the Hessian that explains why gradient descent's speed is governed by the ratio of its largest to smallest eigenvalue (Module 05); and the Fourier-domain view of convolutional layers.

## 9. Matrices as graphs and Markov chains

**Intuition.** A matrix can also be a *network*. Put a 1 in entry $(i, j)$ when there is an edge from node $i$ to node $j$ and you have the adjacency matrix. Multiply it by itself and entry $(i, j)$ counts the two-step walks from $i$ to $j$ — matrix multiplication *is* path counting, because $\sum_k A_{ik}A_{kj}$ sums over intermediate nodes $k$. Replace the ones by probabilities of moving along each edge and the matrix becomes a transition matrix; multiplying a distribution by it moves the probability mass one step forward in time. Keep multiplying and, for well-behaved chains, the distribution forgets where it started.

**Definition.** For a graph on $n$ nodes the **adjacency matrix** $A \in \{0,1\}^{n\times n}$ has $A_{ij} = 1$ iff there is an edge $i \to j$; $(A^k)_{ij}$ counts walks of length exactly $k$ from $i$ to $j$. A **(row-)stochastic transition matrix** $P$ has $P_{ij} = \Pr(\text{next} = j \mid \text{now} = i) \ge 0$ with every row summing to 1. If $\boldsymbol\pi_t$ is the row vector of state probabilities at time $t$,

$$
\boldsymbol\pi_{t+1} = \boldsymbol\pi_t P, \qquad \boldsymbol\pi_t = \boldsymbol\pi_0 P^t, \qquad (P^k)_{ij} = \Pr(\text{state } j \text{ after } k \text{ steps} \mid \text{start } i).
$$

A **stationary distribution** satisfies $\boldsymbol\pi P = \boldsymbol\pi$ — an eigenvector with eigenvalue 1 (Module 03). If the chain is irreducible and aperiodic, $\boldsymbol\pi_0 P^t \to \boldsymbol\pi$ for every starting $\boldsymbol\pi_0$ and the rows of $P^t$ all converge to $\boldsymbol\pi$.

**Worked example.** Two-state weather: from Sunny, tomorrow is Sunny with probability 0.9; from Rainy, tomorrow is Sunny with probability 0.5. $P = \begin{pmatrix} 0.9 & 0.1 \\ 0.5 & 0.5 \end{pmatrix}$. Start rainy, $\boldsymbol\pi_0 = (0, 1)$. Then $\boldsymbol\pi_1 = (0.5, 0.5)$; $\boldsymbol\pi_2 = (0.5\cdot0.9 + 0.5\cdot0.5,\ 0.5\cdot0.1 + 0.5\cdot0.5) = (0.70, 0.30)$; $\boldsymbol\pi_3 = (0.63 + 0.15,\ 0.07 + 0.15) = (0.78, 0.22)$. Stationary: solve $\pi_S = 0.9\pi_S + 0.5\pi_R$, i.e. $0.1\pi_S = 0.5\pi_R$, with $\pi_S + \pi_R = 1$: $\boldsymbol\pi = (5/6, 1/6) \approx (0.833, 0.167)$. Each step closes the gap to $\boldsymbol\pi$ by a factor of $0.4$ (the second eigenvalue, $0.9 + 0.5 - 1$). For path counting, the three-node path graph $1 - 2 - 3$ has $A = \begin{pmatrix} 0&1&0\\1&0&1\\0&1&0\end{pmatrix}$ and $A^2 = \begin{pmatrix} 1&0&1\\0&2&0\\1&0&1\end{pmatrix}$: two 2-step walks return to node 2 (via 1 or via 3), one walk goes from 1 to 3.

**Real-life example (sport / e-commerce).** A football analytics team models possession as a chain with states {defence, midfield, attack, shot, lost}; the transition matrix is estimated from event data and $P^k$ gives the probability of a shot within $k$ passes from each zone — the basis of "expected threat" metrics. An e-commerce team models a shopper's journey through {home, category, product, cart, purchase, exit} the same way; the stationary distribution of a model with "exit" looping back to "home" is the long-run share of page views, and raising $P_{\text{cart} \to \text{purchase}}$ by one percentage point can be priced in revenue by recomputing the absorption probabilities.

**In ML/AI this is …** PageRank, the stationary distribution of a random surfer on the web's adjacency matrix (Module 03 computes it by power iteration); the message-passing step $A X W$ in graph neural networks, where multiplying by the (normalised) adjacency matrix aggregates each node's neighbours; the forward process $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ of a diffusion model and the MCMC transition kernels of Module 18; and the transition tensor $P(s' \mid s, a)$ of a Markov decision process, whose policy-evaluation equation is a linear system (Module 19).

## Common pitfalls and misconceptions

1. **"`A * B` multiplies matrices."** In NumPy `*` is elementwise and silently broadcasts; the matrix product is `A @ B` (or `np.matmul`). The fix: use `@` for linear algebra, `*` only when you *mean* Hadamard products (e.g. masking).
2. **Treating $AB$ and $BA$ as interchangeable.** They differ in general — the 2×2 example in Section 2 swaps columns in one order and rows in the other — and one may not even be defined. Always ask "which map runs first?" and write the composition right-to-left.
3. **Computing `inv(A) @ b`.** It costs about three times as much as a solve, loses accuracy for ill-conditioned $A$, and fails outright for rectangular $A$. Use `np.linalg.solve(A, b)`, or `np.linalg.lstsq` / QR when $A$ is not square. Reserve `inv` for when you genuinely need the inverse's entries (rare — standard errors in OLS are one legitimate case).
4. **Believing $\det A \approx 0$ means "nearly singular".** The determinant scales with $c^n$: $0.1\,I_{10}$ has determinant $10^{-10}$ yet is perfectly conditioned, and $\mathrm{diag}(10^{8}, 10^{-8})$ has determinant 1 yet $\kappa = 10^{16}$. Use the condition number (`np.linalg.cond`) or the singular values to judge near-singularity.
5. **Mistaking rank for "number of nonzero columns".** $M$ in Section 5 has three nonzero columns and rank 2. Rank counts *independent* directions; compute it with `np.linalg.matrix_rank` and watch the tolerance when data are noisy.
6. **Assuming a solution always exists for $A\mathbf{x} = \mathbf{b}$ because $A$ "has enough equations".** Existence needs $\mathbf{b} \in \mathcal{C}(A)$; uniqueness needs a trivial null space. A tall system typically has no exact solution — that is why least squares exists — and a wide one typically has infinitely many — that is why regularisation exists.
7. **Reading $P^{-1}AP$ as a different map.** It is the *same* map in different coordinates; trace, determinant, rank and eigenvalues are unchanged. What changes is only how simple the matrix looks. Conversely, two matrices with the same eigenvalues are not necessarily similar (e.g. $I$ and $\begin{pmatrix}1&1\\0&1\end{pmatrix}$).
8. **Forgetting the bias.** A linear map fixes the origin, so a translation cannot be expressed as $W\mathbf{x}$. Neural layers are affine, $W\mathbf{x} + \mathbf{b}$; regression needs an intercept column; image warps use $3\times 3$ homogeneous coordinates. Dropping the constant forces every fit through the origin.

## Real-world case studies

### Case study 1 — Multicollinearity in a clinical risk model

A hospital fits a linear model predicting 30-day readmission risk from five features: age, systolic blood pressure, BMI, weight (kg) and height (cm). The analyst also adds "weight in pounds" exported from a legacy system. The design matrix $X$ is $n \times 7$ (six features plus intercept). Weight-in-pounds is $2.2046 \times$ weight-in-kg, so two columns are exact multiples: $\operatorname{rank}(X) = 6 < 7$, $X^\top X$ is singular, and `np.linalg.solve` raises `LinAlgError: Singular matrix`. Switching to `lstsq` "works" — it returns the minimum-norm solution — but the two weight coefficients are now arbitrary: adding $+5$ to one and $-5/2.2046$ to the other leaves every prediction unchanged, so neither coefficient can be interpreted or reported in a paper. After dropping the duplicate column the rank is 6 and the fit is unique. The subtler problem is BMI = weight / height², which is a *nonlinear* function of two other columns; the columns are not exactly collinear, but on adults they are highly correlated, and the condition number of $X^\top X$ comes out at about $3 \times 10^{7}$. The standard errors of the BMI and weight coefficients inflate roughly as $\sqrt{\text{VIF}}$, with variance inflation factors above 50. A ridge penalty $\lambda I$ with $\lambda = 1$ on standardised features lowers the condition number to around $10^{3}$ and stabilises the coefficients at the cost of a small bias. The lesson: rank deficiency is a hard failure, *near* rank deficiency is a soft one, and both are properties of the data matrix, not of the algorithm.

### Case study 2 — Calibrating a spectrometer with least squares

A manufacturing lab calibrates a near-infrared spectrometer to predict the moisture content of grain. For 60 calibration samples of known moisture (8 %–20 %, measured by oven drying) it records absorbance at 4 wavelengths, giving $A \in \mathbb{R}^{60\times 5}$ (four absorbances plus an intercept column) and $\mathbf{y} \in \mathbb{R}^{60}$. The normal equations are a $5\times 5$ system; solving them by QR gives coefficients such as $\hat{\mathbf{x}} = (3.1, 42.7, -18.2, 9.5, 0.8)^\top$ with a residual standard deviation of 0.35 percentage points. Two checks from this module make the calibration trustworthy. First, the hat matrix: its trace is 5 (the number of parameters) and the residual degrees of freedom are $60 - 5 = 55$; a sample with leverage $h_{ii} = 0.31$ — six times the average $5/60 \approx 0.083$ — is flagged as influential and turns out to be a mislabelled wet sample. Second, conditioning: wavelengths 2 and 3 are adjacent bands with correlation 0.998, so $\kappa(A) \approx 2{,}000$ and $\kappa(A^\top A) \approx 4\times 10^6$; forming $A^\top A$ explicitly and inverting it loses six digits, which here is survivable but would not be with 50 wavelengths. In production, the lab uses `np.linalg.lstsq` (SVD-based) and re-estimates once a month on fresh samples — a prediction for a new spectrum $\mathbf{a}$ is simply the dot product $\mathbf{a}^\top \hat{\mathbf{x}}$, i.e. one row of the row picture.

### Case study 3 — Customer-state Markov chain for an insurer

An insurer tracks policyholders in three states each month: Active (A), Lapsed (L — payment missed but reinstatable) and Churned (C — gone for good, an absorbing state). From historical data the monthly transition matrix is

$$
P = \begin{pmatrix} 0.80 & 0.15 & 0.05 \\ 0.30 & 0.50 & 0.20 \\ 0 & 0 & 1 \end{pmatrix}.
$$

Starting from 1,000 active customers, $\boldsymbol\pi_0 = (1000, 0, 0)$: after one month $\boldsymbol\pi_1 = (800, 150, 50)$; after two months $\boldsymbol\pi_2 = (800\cdot0.8 + 150\cdot0.3,\ 800\cdot0.15 + 150\cdot0.5,\ 50 + 40 + 30) = (685, 195, 120)$; after twelve months $\boldsymbol\pi_0P^{12} \approx (254, 93, 653)$. Because C is absorbing, everyone churns eventually; the useful quantities are the *expected time to churn*. Writing $Q$ for the $2\times 2$ block of transitions among the transient states, the fundamental matrix $N = (I - Q)^{-1}$ gives expected visits: $I - Q = \begin{pmatrix} 0.2 & -0.15 \\ -0.3 & 0.5\end{pmatrix}$, $\det = 0.10 - 0.045 = 0.055$, so $N = \frac{1}{0.055}\begin{pmatrix} 0.5 & 0.15 \\ 0.3 & 0.2 \end{pmatrix} \approx \begin{pmatrix} 9.09 & 2.73 \\ 5.45 & 3.64\end{pmatrix}$. Row sums: an Active customer stays a further $9.09 + 2.73 \approx 11.8$ months on average before churning, a Lapsed one $9.1$ months. With a monthly premium of EUR 40, the expected remaining lifetime value of an active customer is about EUR 364 (counting only Active months). A retention campaign that cuts $P_{L\to C}$ from 0.20 to 0.10 (and raises $P_{L\to L}$ to 0.60) changes $I - Q$ to $\begin{pmatrix} 0.2 & -0.15 \\ -0.3 & 0.4\end{pmatrix}$ with $\det = 0.08 - 0.045 = 0.035$, so $N = \frac{1}{0.035}\begin{pmatrix} 0.4 & 0.15 \\ 0.3 & 0.2\end{pmatrix} \approx \begin{pmatrix} 11.43 & 4.29 \\ 8.57 & 5.71\end{pmatrix}$: an active customer now stays $15.7$ months in total, of which $11.4$ are paying months, and the expected lifetime value rises from EUR 364 to about EUR 457 — a 26 % gain that can be weighed against the campaign's cost. Every step of this analysis is a $2\times 2$ or $3\times 3$ matrix operation from this module.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| $A\mathbf{x}$, row/column pictures | Dense layers, `X @ w`, embedding lookup | Every prediction is a dot product; outputs live in the column space |
| Matrix product, composition | Stacked layers, $QK^\top V$ attention, batched `bmm` | Linear layers collapse without nonlinearities; FLOP budgets are $2 n d_{in} d_{out}$ |
| Rotation / shear / scaling | Data augmentation, RoPE, robotics frames | Composable geometry; determinant = volume change (normalising flows) |
| Orthogonal matrices | Orthogonal init, QR in lstsq, Householder in SVD | Preserve norms → stable signal propagation and numerics |
| PSD matrices | Covariance, kernel matrices, Hessians at minima | Guarantee non-negative variances, convexity, valid Gaussians |
| Rank, null space | Collinear features, dummy-variable trap, LoRA rank $r$ | Identifiability of coefficients; parameter-efficient fine-tuning |
| Solving $A\mathbf{x}=\mathbf{b}$, conditioning | Ridge/OLS solvers, Newton steps, GP posteriors, policy evaluation | `solve` not `inv`; condition number sets achievable precision and GD speed |
| Normal equations, hat matrix | `LinearRegression`, linear probes, leverage diagnostics | Closed-form fit; projection view explains residual orthogonality and degrees of freedom |
| Change of basis, similarity | PCA, whitening, Fourier view of convolutions | Choose coordinates in which the map (or data) is diagonal |
| Adjacency / transition matrices | PageRank, GNN message passing, MDPs, diffusion forward process | Matrix powers propagate paths and probabilities |

## Cheat sheet

| Item | Formula / fact |
|---|---|
| Matrix–vector product | $(A\mathbf{x})_i = \sum_j A_{ij}x_j$; $A\mathbf{x} = \sum_j x_j\,\mathbf{a}_j$ (column picture) |
| Matrix product | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$; $(m\times p)(p\times n) = m\times n$; cost $O(mpn)$ |
| Transpose rules | $(AB)^\top = B^\top A^\top$, $(A^\top)^\top = A$, $(A+B)^\top = A^\top + B^\top$ |
| Inverse rules | $(AB)^{-1} = B^{-1}A^{-1}$, $(A^\top)^{-1} = (A^{-1})^\top$; 2×2: $\frac{1}{ad-bc}\begin{pmatrix} d & -b\\ -c & a\end{pmatrix}$ |
| Determinant | $\det\begin{pmatrix} a&b\\c&d\end{pmatrix} = ad - bc$; $\det(AB) = \det A \det B$; $\det A^\top = \det A$; $\det(cA) = c^n\det A$; $\det A = 0 \iff$ singular |
| Rotation by $\theta$ | $\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$, $\det = 1$, orthogonal |
| Orthogonal $Q$ | $Q^\top Q = I$, $\lVert Q\mathbf{x}\rVert = \lVert\mathbf{x}\rVert$, $Q^{-1} = Q^\top$ |
| PSD | $A = A^\top$ and $\mathbf{x}^\top A\mathbf{x} \ge 0$; $A^\top A$ is always PSD |
| Rank–nullity | $\operatorname{rank} A + \dim\mathcal{N}(A) = n$ (columns); row rank = column rank |
| Solvability | $A\mathbf{x} = \mathbf{b}$ solvable $\iff \mathbf{b} \in \mathcal{C}(A)$; unique $\iff \mathcal{N}(A) = \{\mathbf{0}\}$ |
| Condition number | $\kappa(A) = \lVert A\rVert\lVert A^{-1}\rVert = \sigma_{\max}/\sigma_{\min}$; relative error amplification $\le \kappa\,\varepsilon$ |
| Normal equations | $A^\top A\hat{\mathbf{x}} = A^\top\mathbf{y}$; $\hat{\mathbf{x}} = (A^\top A)^{-1}A^\top\mathbf{y}$ (compute via QR/SVD) |
| Hat matrix | $H = A(A^\top A)^{-1}A^\top$; $H^2 = H = H^\top$; $\operatorname{tr} H = \operatorname{rank} A$; residual $= (I - H)\mathbf{y} \perp \mathcal{C}(A)$ |
| Change of basis | $\mathbf{x} = P\mathbf{c}$, $\mathbf{c} = P^{-1}\mathbf{x}$; $A' = P^{-1}AP$ preserves det, trace, rank, eigenvalues |
| Markov chain | $\boldsymbol\pi_{t+1} = \boldsymbol\pi_t P$, rows of $P$ sum to 1; stationary $\boldsymbol\pi P = \boldsymbol\pi$; $(A^k)_{ij}$ counts $k$-step walks |
| NumPy | `A @ B` product, `A * B` elementwise, `np.linalg.solve`, `np.linalg.lstsq`, `np.linalg.qr`, `np.linalg.matrix_rank`, `np.linalg.cond`, `np.linalg.det` |

## Self-assessment

1. **Why does the column picture of $A\mathbf{x}$ immediately tell you when $A\mathbf{x} = \mathbf{b}$ has a solution?**
   <details><summary>Answer</summary>Because $A\mathbf{x}$ is a linear combination of the columns with weights $x_j$, the set of all possible $A\mathbf{x}$ is exactly the span of the columns — the column space. A solution exists if and only if $\mathbf{b}$ lies in that span. If $\mathbf{b}$ is outside it, no choice of weights reaches it and you need least squares instead.</details>

2. **$W_1$ is $4\times 3$ and $W_2$ is $5\times 4$. Which products are defined, and what is the shape of the composed map applied to a batch $X$ of shape $(n, 3)$ stored in rows?**
   <details><summary>Answer</summary>$W_2W_1$ is defined ($5\times4$ times $4\times3$ gives $5\times3$); $W_1W_2$ is not (inner dimensions 3 and 5 disagree). With rows as samples you apply transposes: $X W_1^\top$ has shape $(n, 4)$ and $X W_1^\top W_2^\top = X (W_2W_1)^\top$ has shape $(n, 5)$. Without a nonlinearity the two layers equal a single $5\times 3$ layer.</details>

3. **A $2\times2$ matrix has determinant $-3$. What two things does this tell you about the unit square's image?**
   <details><summary>Answer</summary>Its area is 3 (area scaled by $\lvert\det\rvert$) and its orientation is reversed (negative sign): the matrix includes a reflection. The matrix is invertible since $\det \neq 0$.</details>

4. **Why is $Q^\top$ the inverse of an orthogonal matrix, and why does this make orthogonal matrices numerically pleasant?**
   <details><summary>Answer</summary>Orthonormal columns mean $Q^\top Q = I$ by definition of the dot products between columns; for a square matrix a one-sided inverse is the inverse, so $Q^{-1} = Q^\top$. Inverting is then free and exact (no elimination, no rounding), and because $\lVert Q\mathbf{x}\rVert = \lVert\mathbf{x}\rVert$ the condition number is 1 — errors are neither amplified nor hidden.</details>

5. **A data matrix has 1,000 rows and 12 columns, and `np.linalg.matrix_rank` returns 11. What does that mean practically, and what are two fixes?**
   <details><summary>Answer</summary>One column is an exact linear combination of the others (for example a one-hot group that sums to the intercept, or a unit-converted duplicate), so $X^\top X$ is singular and the regression coefficients are not unique — the null space direction can be added to the coefficient vector at no cost. Fixes: drop the redundant column (or one one-hot level), or add a ridge penalty $\lambda I$, which makes $X^\top X + \lambda I$ full rank.</details>

6. **Give two reasons to prefer `solve(A, b)` over `inv(A) @ b`.**
   <details><summary>Answer</summary>Cost: an LU solve is about $\tfrac23 n^3$ flops versus roughly $2n^3$ for the inverse plus the extra product. Accuracy: the triangular solves commit only the unavoidable rounding error, whereas forming the inverse and multiplying can amplify error by an extra factor related to $\kappa(A)$. A third reason is generality — `solve` extends naturally to `lstsq` for rectangular systems, where no inverse exists.</details>

7. **Derive the normal equations in one sentence from the geometry of projection.**
   <details><summary>Answer</summary>The best approximation $A\hat{\mathbf{x}}$ is the orthogonal projection of $\mathbf{y}$ onto the column space, so the residual $\mathbf{y} - A\hat{\mathbf{x}}$ is orthogonal to every column of $A$, which is the statement $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}$, i.e. $A^\top A\hat{\mathbf{x}} = A^\top\mathbf{y}$.</details>

8. **The hat matrix of a regression with 8 parameters on 200 observations has trace 8. Why, and what is the average leverage?**
   <details><summary>Answer</summary>$H$ is the orthogonal projector onto an 8-dimensional column space; the trace of a projector equals the dimension of the space it projects onto (its eigenvalues are eight 1s and 192 0s). The average diagonal entry (leverage) is therefore $8/200 = 0.04$; observations with leverage several times that are influential.</details>

9. **$A' = P^{-1}AP$ with $P$ invertible. Which of $\det$, trace, rank, the entries, and the eigenvalues can differ between $A$ and $A'$?**
   <details><summary>Answer</summary>Only the entries. Determinant ($\det P^{-1}\det A\det P = \det A$), trace (cyclic property), rank (invertible factors do not change rank) and eigenvalues (if $A\mathbf{v} = \lambda\mathbf{v}$ then $A'(P^{-1}\mathbf{v}) = \lambda P^{-1}\mathbf{v}$) are all invariant — they describe the map, not the coordinates.</details>

10. **For the transition matrix $P = \begin{pmatrix} 0.9 & 0.1 \\ 0.5 & 0.5 \end{pmatrix}$, why is $(1, 1)^\top$ an eigenvector of $P$ with eigenvalue 1, and what does the stationary distribution have to do with eigenvectors?**
    <details><summary>Answer</summary>Each row sums to 1, so $P\mathbf{1} = \mathbf{1}$: the all-ones vector is a right eigenvector with eigenvalue 1. The stationary distribution is the corresponding *left* eigenvector, $\boldsymbol\pi P = \boldsymbol\pi$ with $\sum_i \pi_i = 1$; here $\boldsymbol\pi = (5/6, 1/6)$. Powers $P^t$ converge to a matrix whose rows are all $\boldsymbol\pi$, which is why the starting state is forgotten.</details>

## Further reading

- Gilbert Strang, *Introduction to Linear Algebra* (Wellesley-Cambridge Press) — the four fundamental subspaces, the column picture and least squares as projection, in Strang's inimitable style; his MIT 18.06 lectures are free on MIT OpenCourseWare.
- Gilbert Strang, *Linear Algebra and Learning from Data* (Wellesley-Cambridge Press, 2019) — the same author aiming squarely at ML readers.
- Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020; free PDF at mml-book.github.io) — Chapters 2–4 cover this module's material with ML notation.
- Lloyd N. Trefethen and David Bau, *Numerical Linear Algebra* (SIAM, 1997) — why you solve rather than invert, QR for least squares, conditioning and stability, in forty short lectures.
- Sheldon Axler, *Linear Algebra Done Right* (Springer) — the coordinate-free view of linear maps, for readers who want the "same map, different basis" idea made rigorous.
- Stephen Boyd and Lieven Vandenberghe, *Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares* (Cambridge University Press, 2018; free PDF at the authors' Stanford page) — applications-first, with least squares, classification and dynamics examples.
- 3Blue1Brown, *Essence of Linear Algebra* (YouTube series) — the animated geometric intuition for linear maps, determinants and change of basis that Sections 3 and 8 describe in words.
- Ian Goodfellow, Yoshua Bengio and Aaron Courville, *Deep Learning* (MIT Press, 2016; free at deeplearningbook.org) — Chapter 2 is a compact linear-algebra review written for deep-learning readers.
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022; free PDF) — Chapter 7 on linear algebra, including the SVD and least-squares connections you will need next.

## What's in the notebooks

**`lesson.ipynb`** walks through the module with code and figures: the unit square under rotation, shear, scaling and projection with the determinant printed for each; the column and row pictures of $A\mathbf{x}$ computed side by side; a loop-based `matmul` verified against `@`, with timing to show the cubic cost and the collapse of two linear layers into one; rank detection on a synthetic feature matrix with a duplicated and a one-hot-trapped column; solving systems with `solve` versus `inv` and the blow-up on an ill-conditioned matrix as $\kappa$ grows; least squares three ways (normal equations, QR, `lstsq`) on noisy line and quadratic data, with the hat matrix checked for symmetry and idempotence; a change of basis that diagonalises a symmetric matrix; and a Markov transition matrix whose powers converge to the stationary distribution.

**`exercises.ipynb`** (with `solutions.ipynb`) tests the same ideas in twelve exercises: the row and column pictures of $A\mathbf{x}$; matrix multiplication with three loops; rotation, shear and scaling matrices and the determinant of their composition; the rank, null space and solvability of a given $3\times4$ matrix; a $3\times 3$ system by hand and by `solve`; least squares via the normal equations checked against `lstsq`; the hat matrix as a symmetric idempotent projector with its leverages; detecting exactly collinear columns in a customer feature matrix; propagating a Markov chain and finding its stationary distribution with a linear solve; two conceptual questions (layer shapes and $AB$ versus $BA$; why `solve` beats `inv` and what collinearity does to coefficients); and a mini-project fitting a multi-feature linear regression to synthetic housing data with your own normal-equations solver, interpreting the coefficients and matching `sklearn.linear_model.LinearRegression`.

Next: [Module 03 — Eigenvalues, SVD and PCA](../03-eigenvalues-svd-and-pca/README.md), where the "same map, different coordinates" idea becomes the eigendecomposition and the SVD.
