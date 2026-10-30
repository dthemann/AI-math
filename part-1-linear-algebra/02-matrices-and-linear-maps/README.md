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
- Compute the rank, column space and null space of a small matrix by elimination, name the four fundamental subspaces and the two orthogonality relations between them, apply rank–nullity, and diagnose collinear features in a data matrix.
- Decide whether $A\mathbf{x} = \mathbf{b}$ has zero, one or infinitely many solutions; solve small systems by elimination; explain why `np.linalg.solve` beats `np.linalg.inv`.
- Derive the normal equations $A^\top A\,\hat{\mathbf{x}} = A^\top \mathbf{b}$ both geometrically and by calculus, fit a line to a handful of points by hand, verify that the hat matrix is a projection, state when $A^\top A$ is invertible, and read $R^2$ off the Pythagoras split taken around the mean, $\mathrm{TSS} = \mathrm{ESS} + \mathrm{RSS}$.
- Express a vector in a new basis, form $P^{-1}AP$ (and know when $P^\top$ may stand in for $P^{-1}$), show why trace, determinant, rank and eigenvalues survive the change, and explain why "same map, different coordinates" is the key to eigendecomposition and PCA.
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

Row picture: first entry $1\cdot 2 + 2\cdot(-1) = 0$; second entry $3 \cdot 2 + 4\cdot(-1) = 2$. So $A\mathbf{x} = (0, 2)^\top$. Read as a system, the two rows are the equations $x_1 + 2x_2 = 0$ and $3x_1 + 4x_2 = 2$, and $\mathbf{x} = (2, -1)^\top$ is the point where the two lines cross (Figure 1(b)).
Column picture: $2\begin{pmatrix}1\\3\end{pmatrix} + (-1)\begin{pmatrix}2\\4\end{pmatrix} = \begin{pmatrix}2-2\\6-4\end{pmatrix} = \begin{pmatrix}0\\2\end{pmatrix}$. Same answer, different story: we walked two steps along column one and one step back along column two (Figure 1(a)).
Function picture: $A$ sends $\mathbf{e}_1 = (1,0)^\top$ to $(1,3)^\top$ and $\mathbf{e}_2 = (0,1)^\top$ to $(2,4)^\top$ — those are its columns. Everything else follows by linearity.

![Two panels. Left, output space: the columns a1 = (1, 3) and a2 = (2, 4) as arrows from the origin, a lighter walk of two steps along a1 up to (2, 6) and then one step back along a2 down to (0, 2), and the result Ax = (0, 2) as a bold arrow up the vertical axis. Right, input space: the two lines x1 + 2 x2 = 0 and 3 x1 + 4 x2 = 2, crossing at the dot x = (2, -1).](figures/column-vs-row-picture.svg)

**Figure 1.** The worked example in its two pictures. (a) Column picture, drawn in the *output* space where $\mathbf{b}$ lives: $A\mathbf{x} = 2\mathbf{a}_1 - 1\mathbf{a}_2$ is a walk of two steps along $\mathbf{a}_1 = (1, 3)$ followed by one step back along $\mathbf{a}_2 = (2, 4)$, ending at $(0, 2)$. (b) Row picture, drawn in the *input* space where $\mathbf{x}$ lives: each row of $A$ is one linear equation, hence one line, and $\mathbf{x} = (2, -1)$ is the one point on both lines. Same $A$, same $\mathbf{x}$, same answer — one picture shows where outputs can land, the other where the input sits.

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

**Worked example.** Rotation by $90^\circ$: $R = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$, $\det R = 0\cdot 0 - (-1)(1) = 1$: area and orientation preserved. Shear $H_2 = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$ sends the square to the parallelogram with corners $(0,0),(1,0),(3,1),(2,1)$; $\det H_2 = 1 - 0 = 1$, so the slanted shape still has area 1 (base 1, height 1; Figure 2(a)). Scaling $S = \mathrm{diag}(2, 3)$ has $\det = 6$: area multiplied by six (Figure 2(b)). Projection onto $\mathbf{u} = (1,0)^\top$: $P = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$, $\det P = 0$: the square becomes the segment from $(0,0)$ to $(1,0)$ with zero area, and every point $(x, y)$ with the same $x$ lands on the same image — the $y$ coordinate is lost; in Figure 2(c) the inputs $(0.5, 0.2)$ and $(0.5, 0.8)$ both land on $(0.5, 0)$. Composition: rotating then scaling gives $SR$ with $\det(SR) = \det S \det R = 6$.

![Three panels, each with the dashed unit square and its filled image. Left: the shear sends the square to the parallelogram with corners (0, 0), (1, 0), (3, 1), (2, 1); a dotted vertical marks its height 1 over base 1. Middle: the scaling sends it to a 2 by 3 rectangle. Right: the projection flattens it onto the segment from (0, 0) to (1, 0); two red input points (0.5, 0.2) and (0.5, 0.8) drop onto the same output (0.5, 0).](figures/unit-square-determinant.svg)

**Figure 2.** The unit square (dashed) and its image (filled) under three of the worked-example matrices; the blue and orange arrows are the images of $\mathbf{e}_1$ and $\mathbf{e}_2$, i.e. the columns (in (c) the image of $\mathbf{e}_2$ is the orange dot at the origin). (a) The shear $H_2$ slants the square into the parallelogram with corners $(0,0), (1,0), (3,1), (2,1)$, but base 1 and height 1 keep the area at $\det H_2 = 1$. (b) $\mathrm{diag}(2, 3)$ stretches it into a $2\times 3$ rectangle: area $\times 6 = \det S$. (c) The projection onto $\mathbf{e}_1$ flattens it onto a segment of area $0 = \det P$, and the two inputs $(0.5, 0.2)$ and $(0.5, 0.8)$ — any two points with the same $x$ — land on the same output $(0.5, 0)$: the $y$ coordinate cannot be recovered. The $90^\circ$ rotation $R$ (not drawn) merely turns the square, $\det R = 1$, and composing it with (b) gives $\det(SR) = 6\cdot 1 = 6$.

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

Why orthogonal matrices preserve length: $\lVert Q\mathbf{x}\rVert^2 = (Q\mathbf{x})^\top(Q\mathbf{x}) = \mathbf{x}^\top Q^\top Q\,\mathbf{x} = \mathbf{x}^\top \mathbf{x} = \lVert\mathbf{x}\rVert^2$. Dot products are preserved by the same calculation, so angles are too. In 2-D the orthogonal matrices are exactly the rotations and reflections (in higher dimensions, products of these).

**Worked example.** $Q = \frac{1}{\sqrt 2}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$ (rotation by $45^\circ$). Check: $Q^\top Q = \frac12 \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix} = \frac12\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = I$. Apply to $\mathbf{x} = (3, 4)^\top$ with $\lVert\mathbf{x}\rVert = 5$: $Q\mathbf{x} = \frac{1}{\sqrt2}(3 - 4,\; 3 + 4)^\top = \frac{1}{\sqrt 2}(-1, 7)^\top$, whose squared norm is $\frac12(1 + 49) = 25$. Length preserved (Figure 3(a)). Now $A = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$ is symmetric; is it positive definite? $\mathbf{x}^\top A\mathbf{x} = 2x_1^2 + 2x_1x_2 + 2x_2^2 = (x_1 + x_2)^2 + x_1^2 + x_2^2 > 0$ for $\mathbf{x} \neq \mathbf{0}$. Yes: the form is $6$ at $(1, 1)^\top$ and $2$ at $(1, -1)^\top$, and its level curves are nested ellipses — a bowl (Figure 3(b)). By contrast $\begin{pmatrix} 1 & 2 \\ 2 & 1\end{pmatrix}$ gives $\mathbf{x}^\top A \mathbf{x} = -2$ at $\mathbf{x} = (1, -1)^\top$ (and $+6$ at $(1, 1)^\top$), so it is symmetric but *not* PSD — its form is a saddle (Figure 3(c)), and it cannot be a covariance matrix.

![Three panels. Left: the vector x = (3, 4) and its image Q x = (-1, 7) over root 2 both end on a dashed arc of the circle of radius 5, with a 45-degree arc between them. Middle: nested ellipses, the level curves 1, 2, 4, 6 of the quadratic form of the matrix with rows (2, 1) and (1, 2), long axis along (1, -1), with the values 6 at (1, 1) and 2 at (1, -1) marked. Right: the level curves of the quadratic form of the matrix with rows (1, 2) and (2, 1), a saddle: two straight lines through the origin where it is zero, hyperbolas on either side, the negative region shaded, with the values 6 at (1, 1) and -2 at (1, -1) marked.](figures/quadratic-forms.svg)

**Figure 3.** (a) The orthogonal $Q$ rotates $\mathbf{x} = (3, 4)$ by $45^\circ$ to $Q\mathbf{x} = (-1, 7)/\sqrt 2$; both arrows end on the dashed arc of the circle of radius 5, because $\lVert Q\mathbf{x}\rVert^2 = \tfrac12(1 + 49) = 25 = \lVert\mathbf{x}\rVert^2$. (b) For the positive definite $A = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$ the quadratic form $f(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x} = 2x_1^2 + 2x_1x_2 + 2x_2^2$ is a bowl: its level curves $f = 1, 2, 4, 6$ are nested ellipses, $f(1, 1) = 6$ along the short axis, $f(1, -1) = 2$ along the long axis, and $f > 0$ everywhere except at the origin. (c) For $\begin{pmatrix} 1 & 2 \\ 2 & 1\end{pmatrix}$, symmetric but not PSD, $g(\mathbf{x}) = x_1^2 + 4x_1x_2 + x_2^2$ is a saddle: $g = 0$ on two lines through the origin (black), $g(1, 1) = 6$ on one side and $g(1, -1) = -2$ in the shaded region where $g < 0$. A covariance or kernel matrix must look like (b), never like (c). The axes of the bowl are the directions $\mathbf{p}_1 = (1, 1)$ and $\mathbf{p}_2 = (1, -1)$ in which Section 8 diagonalises this very $A$.

**Real-life example (manufacturing).** A quality-control lab measures three dimensions of a machined part and reports their covariance matrix $\Sigma$. $\Sigma$ must be symmetric (the covariance of length with width equals that of width with length) and PSD (the variance of any weighted combination $\mathbf{w}^\top \mathbf{x}$ is $\mathbf{w}^\top \Sigma \mathbf{w}$, which cannot be negative). When an engineer hand-edits a few entries "to make them look right" and the result has a negative eigenvalue, the downstream Gaussian model produces `nan` likelihoods. The fix is to project back to the PSD cone — a routine `scipy`/`sklearn` step (`sklearn.covariance` shrinkage estimators guarantee a PSD result).

**In ML/AI this is …** the orthogonal initialisation of recurrent weight matrices (`torch.nn.init.orthogonal_`), chosen because repeated multiplication by an orthogonal matrix neither explodes nor vanishes the signal; the symmetric PSD Gram matrix $X^\top X$ in ridge regression and the kernel matrix $K$ in Gaussian processes and SVMs; the diagonal preconditioners in Adam and RMSProp, which scale each parameter axis independently; and the permutation matrices implicit in shuffling a batch or in the pivoting inside `np.linalg.solve`.

## 5. Column space, null space and rank

**Intuition.** The column space of $A$ is every output the machine can produce: all vectors $A\mathbf{x}$. The null space is every input the machine destroys: all $\mathbf{x}$ with $A\mathbf{x} = \mathbf{0}$. Rank counts how many *genuinely different* directions the columns point in. A $3 \times 3$ matrix whose third column is twice the first plus the second has only two independent columns; it maps $\mathbb{R}^3$ onto a plane and kills a whole line of inputs. Information is lost, and the loss is exactly the null space.

**Definitions.** For $A \in \mathbb{R}^{m\times n}$, with the worked matrix $M$ below as the running example:

*Column space and solvability.* The **column space** $\mathcal{C}(A) = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\} \subseteq \mathbb{R}^m$ is the span of the columns — in the column picture of Section 1, every output the machine can produce — so $A\mathbf{x} = \mathbf{b}$ is solvable iff $\mathbf{b} \in \mathcal{C}(A)$. The test can be made concrete. Every column of $M$ has second entry twice its first, so every combination of the columns does too: $\mathcal{C}(M)$ is the plane $x_2 = 2x_1$ in $\mathbb{R}^3$ (Figure 4(a)), and $M\mathbf{x} = \mathbf{b}$ is solvable exactly when $b_2 = 2b_1$ — yes for $\mathbf{b} = (3, 6, 1)^\top$, no for $\mathbf{b} = (2, 1, 1)^\top$, and three equations in three unknowns do not change that (pitfall 6).

*Null space.* The **null space** $\mathcal{N}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\} \subseteq \mathbb{R}^n$ collects the inputs the machine destroys. It always contains $\mathbf{0}$, and it lives in the *input* space, whereas $\mathcal{C}(A)$ lives in the output space. In the language of [Module 01](../01-vectors-and-geometry/README.md) (Section 1), a non-zero null vector is a closed loop — a non-trivial combination of the columns that returns to the origin — so $\mathcal{N}(A) \neq \{\mathbf{0}\}$ says exactly that the columns are linearly dependent.

*Rank and rank–nullity.* The **rank** $r = \operatorname{rank} A = \dim \mathcal{C}(A)$ is the number of linearly independent columns, which equals the number of linearly independent rows (shown below). Always $r \le \min(m, n)$; $A$ has **full rank** when equality holds, otherwise it is **rank deficient**. The **rank–nullity theorem** says $\operatorname{rank} A + \dim\mathcal{N}(A) = n$ (the number of columns): every column either contributes a new direction or contributes a null-space dimension.

*Finding the null space by elimination.* When no relation among the columns is visible by eye, row-reduce. Subtracting a multiple of one row from another, or scaling a row by a nonzero constant, does not change the solutions of $A\mathbf{x} = \mathbf{0}$ (the new equation is a combination of old ones, and the step can be undone), so the reduced matrix has the same null space as $A$. For $M$: row 2 minus twice row 1 gives $(0, 0, 0)$; row 3 minus row 1 gives $(0, -2, -2)$, which we scale to $(0, 1, 1)$; then row 1 minus twice that gives $(1, 0, 1)$. The reduced form is

$$
M \;\longrightarrow\; \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix},
$$

with a **pivot** (the leading 1 of a nonzero row) in columns 1 and 2; column 3 has no pivot and is **free**. Set the free variable $x_3 = t$ and read the pivot rows as $x_1 + t = 0$ and $x_2 + t = 0$: $\mathcal{N}(M) = \{t\,(-1, -1, 1)^\top\}$, the same line as $t\,(1, 1, -1)^\top$. The rule: one null-space direction per free column, found by setting that free variable to 1 and the other free variables to 0. This is rank–nullity in action — 2 pivot columns + 1 free column = 3 columns.

*A basis for the column space.* The pivot columns of $A$ *itself* form a basis of $\mathcal{C}(A)$: for $M$, columns 1 and 2, $(1, 2, 1)^\top$ and $(2, 4, 0)^\top$. Not the columns of the reduced matrix — elimination mixes rows, which changes the column space: the reduced columns $(1, 0, 0)^\top$ and $(0, 1, 0)^\top$ span the $x_1x_2$-plane, which does not even contain $(1, 2, 1)^\top$.

*Row space, and why row rank equals column rank.* The **row space** is the span of the rows, a subspace of the input space $\mathbb{R}^n$ (it is $\mathcal{C}(A^\top)$). Elimination replaces rows by combinations of rows, so it leaves the row space unchanged, and the nonzero reduced rows — $(1, 0, 1)$ and $(0, 1, 1)$ for $M$ — are a basis of it. The number of pivots therefore counts the independent rows; it also counts the independent columns (the pivot columns); so the two ranks agree. Row 2 of $M$ is twice row 1, which is the same rank-2 verdict seen from the row side.

*The left null space.* The fourth subspace is $\mathcal{N}(A^\top) = \{\mathbf{y} \in \mathbb{R}^m : A^\top\mathbf{y} = \mathbf{0}\}$, the vectors orthogonal to every column. For $M$, $\mathbf{y} = (-2, 1, 0)^\top$ works: $M^\top\mathbf{y} = (-2 + 2,\; -4 + 4,\; -6 + 6)^\top = \mathbf{0}$. It encodes "row 2 $= 2\times$ row 1" from the output side, and it is the normal vector of the plane $x_2 = 2x_1$ — which is why the solvability test above is the single equation $\mathbf{y}\cdot\mathbf{b} = -2b_1 + b_2 = 0$. In general, $\mathbf{b} \in \mathcal{C}(A)$ iff $\mathbf{b}$ is orthogonal to every vector of $\mathcal{N}(A^\top)$: "is $\mathbf{b}$ in the span?" becomes a handful of dot products.

*Two orthogonal pairs.* $A\mathbf{x} = \mathbf{0}$ says that $\mathbf{x}$ has zero dot product with every row, so $\mathcal{N}(A)$ is perpendicular to the row space — for $M$, $(1, 1, -1)\cdot(1, 0, 1) = 0$ and $(1, 1, -1)\cdot(0, 1, 1) = 0$ — and likewise $\mathcal{N}(A^\top)$ is perpendicular to $\mathcal{C}(A)$. The dimensions add up: row space ($r$) plus null space ($n - r$) fill $\mathbb{R}^n$; column space ($r$) plus left null space ($m - r$) fill $\mathbb{R}^m$. These are Strang's **four fundamental subspaces**. The SVD of [Module 03](../03-eigenvalues-svd-and-pca/README.md) hands you orthonormal bases of all four at once, and the SVD view of ridge regression in [Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md) shrinks exactly the coefficient directions in which the design matrix has little or no variance — its near-null-space directions — while leaving the high-variance directions, which the data determine, almost untouched.

*Solution sets.* If $\mathbf{x}_p$ solves $A\mathbf{x} = \mathbf{b}$, so does $\mathbf{x}_p + \mathbf{z}$ for every $\mathbf{z} \in \mathcal{N}(A)$, because $A(\mathbf{x}_p + \mathbf{z}) = \mathbf{b} + \mathbf{0}$; conversely, two solutions differ by a null vector. The full solution set is therefore $\mathbf{x}_p + \mathcal{N}(A)$ — a copy of the null space shifted away from the origin, for $M$ a line parallel to $t\,(1, 1, -1)^\top$ (Figure 4(b)) — and the solution is unique iff $\mathcal{N}(A) = \{\mathbf{0}\}$. Among infinitely many solutions, the one of **minimum norm** is the point of that line (or flat) closest to the origin, equivalently the one with no null-space component, i.e. the one lying in the row space. This is why a rank-deficient design matrix (the dummy-variable trap below, Case study 1) has infinitely many coefficient vectors with identical predictions, and why `np.linalg.lstsq` returns the minimum-norm one.

For square $A$: full rank $\iff$ invertible $\iff \det A \neq 0 \iff \mathcal{N}(A) = \{\mathbf{0}\}$. A rank-$r$ matrix can be written as a sum of $r$ outer products $\mathbf{u}_k\mathbf{v}_k^\top$ and no fewer — this is the seed of low-rank approximation in Module 03.

*In practice.* `scipy.linalg.null_space(M)` returns an orthonormal basis of the null space, here the single column $(-1, -1, 1)^\top/\sqrt 3$ (up to sign); `np.linalg.matrix_rank(M)` counts the singular values above a tolerance (for $M$ they are $8.44$, $0.92$ and $0$, so the rank is 2); and `np.linalg.lstsq(M, b)` returns the minimum-norm solution when there are many, or the least-squares one (Section 7) when there are none. With measured data an exact rank is the wrong question: a column that is 99.9 % a combination of the others is "numerically almost rank deficient", and the condition number of Section 6 takes over from the rank.

**Worked example.** Take

$$
M = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 0 & 1 \end{pmatrix}.
$$

Column 3 equals column 1 plus column 2: $(1, 2, 1) + (2, 4, 0) = (3, 6, 1)$. So the columns span at most a plane. Columns 1 and 2 are not multiples of each other (the third entries $1$ and $0$ differ), so $\operatorname{rank} M = 2$ and the column space is the plane spanned by $(1,2,1)^\top$ and $(2,4,0)^\top$ — the plane $x_2 = 2x_1$ of Figure 4(a), since both spanning vectors satisfy that equation. By rank–nullity, $\dim\mathcal{N}(M) = 3 - 2 = 1$. The relation $\mathbf{c}_1 + \mathbf{c}_2 - \mathbf{c}_3 = \mathbf{0}$ says $M (1, 1, -1)^\top = \mathbf{0}$; check row 3: $1 + 0 - 1 = 0$. The null space is the line $t\,(1, 1, -1)^\top$ — the line that elimination found above without any spotting. Row 2 is twice row 1, confirming the rank from the row side, and the left null vector $\mathbf{y} = (-2, 1, 0)^\top$ records that fact.

Is $\mathbf{b} = (2, 1, 1)^\top$ in $\mathcal{C}(M)$? $\mathbf{y}\cdot\mathbf{b} = -4 + 1 = -3 \neq 0$, so no: $b_2 = 1$ is not $2b_1 = 4$, $\mathbf{b}$ sticks out of the plane (Figure 4(a); the nearest point of the plane is $(0.8, 1.6, 1.0)$, at distance $\lvert\mathbf{y}\cdot\mathbf{b}\rvert/\lVert\mathbf{y}\rVert = 3/\sqrt 5 \approx 1.34$), and $M\mathbf{x} = \mathbf{b}$ has no solution despite three equations in three unknowns. For $\mathbf{b} = \mathbf{c}_3 = (3, 6, 1)^\top$ instead, $\mathbf{y}\cdot\mathbf{b} = -6 + 6 = 0$, and $\mathbf{x}_p = (0, 0, 1)^\top$ is one solution (it picks out column 3), so the full solution set is $(0, 0, 1)^\top + t\,(1, 1, -1)^\top$: the points $(0, 0, 1)$, $(0.5, 0.5, 0.5)$ and $(1, 1, 0)$ all satisfy $M\mathbf{x} = (3, 6, 1)^\top$ — check the last one: $(1 + 2,\; 2 + 4,\; 1 + 0) = (3, 6, 1)$ — and so does every other point of the dashed line in Figure 4(b). `np.linalg.lstsq(M, c3)` returns the point of that line closest to the origin, $(\tfrac13, \tfrac13, \tfrac23)^\top$, the one orthogonal to the null direction: $\tfrac13 + \tfrac13 - \tfrac23 = 0$. Consequently $M\mathbf{x} = \mathbf{b}$ has either no solution ($\mathbf{b}$ off the plane) or a whole line of them — never exactly one.

![Two three-dimensional panels. Left, output space: a translucent vertical plane with the three column vectors c1 = (1, 2, 1), c2 = (2, 4, 0) and c3 = (3, 6, 1) drawn inside it, a dashed path showing c3 as c1 followed by c2, and a red vector b = (2, 1, 1) leaving the plane with a dotted perpendicular to its foot on the plane. Right, input space: a red line through the origin in the direction (1, 1, -1), and a parallel dashed green line through (0, 0, 1) with dots at (0, 0, 1), (0.5, 0.5, 0.5) and (1, 1, 0).](figures/column-space-and-null-space.svg)

**Figure 4.** (a) Output space $\mathbb{R}^3$: all three columns of $M$ lie in the plane $x_2 = 2x_1$ — "row 2 $= 2\times$ row 1" seen from the output side — so three nonzero columns give only rank 2 (pitfall 5), and $\mathbf{c}_3$ is reached tip-to-tail as $\mathbf{c}_1 + \mathbf{c}_2$ (dashed). $\mathbf{b} = (2, 1, 1)$ leaves the plane (dotted: the perpendicular to its nearest point $(0.8, 1.6, 1.0)$, length $1.34$), so $M\mathbf{x} = \mathbf{b}$ has no solution although there are as many equations as unknowns (pitfall 6). (b) Input space $\mathbb{R}^3$: the null space is the solid line $t\,(1, 1, -1)$ — two pivot directions plus this one null direction account for the three columns — and the solutions of $M\mathbf{x} = \mathbf{c}_3$ form the parallel dashed line $(0, 0, 1) + t\,(1, 1, -1)$, every point of which maps to $(3, 6, 1)$; `np.linalg.lstsq` returns the point of it nearest the origin.

**Real-life example (e-commerce).** An online shop encodes the "device" feature of each session with one-hot columns `is_mobile`, `is_desktop`, `is_tablet`, and its regression design matrix also contains an all-ones intercept column. Since every session is exactly one device, `is_mobile + is_desktop + is_tablet = 1` = the intercept column for every row. The four columns span only three dimensions: the design matrix is rank deficient, $X^\top X$ is singular, and the normal equations have infinitely many solutions — you can add any constant to the three device coefficients and subtract it from the intercept without changing a single prediction. This is the **dummy-variable trap**; the fix is to drop one category (`drop_first=True` in pandas, `drop='first'` in `OneHotEncoder`) or to regularise. The same thing happens with redundant sensors: two thermometers reading the same temperature in °C and °F give features that are exactly collinear with the intercept column (one is an affine function of the other, $F = 1.8\,C + 32$, so the ones column, the °C column and the °F column are linearly dependent).

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

**Worked example (2×2, conditioning).** Consider $A = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001\end{pmatrix}$. With $\mathbf{b} = (2, 2.0001)^\top$ the solution is $(1, 1)^\top$. Change the last digit of $\mathbf{b}$ to $2.0002$ — a relative change of $5\times 10^{-5}$ — and the solution becomes $(0, 2)^\top$: a 100 % change (Figure 5(b)). Indeed $\kappa(A) \approx 4\times 10^{4}$, and $4\times10^4 \cdot 5\times 10^{-5} \approx 2$ is an upper bound consistent with the observed 100 % change — the actual amplification, $1 / (5\times10^{-5}) = 2\times 10^4$, is about half of $\kappa$. The two columns are nearly parallel, so the parallelogram is a sliver and tiny nudges to $\mathbf{b}$ slide the solution a long way along it. The row picture says the same thing: the two equations are lines only $0.003^\circ$ apart, and nudging either line slides their crossing point far along the shared direction. For comparison, the rows of the $2\times2$ example above cross at $45^\circ$ and $\kappa = 2.6$: nudging its $b_2$ from $10$ to $10.5$ — a 5 % change, visible to the naked eye — moves the solution only from $(1, 3)^\top$ to $(0.9, 3.2)^\top$, a 7 % change (Figure 5(a)). Note that $\det A = 0.0001$ is *not* what makes $A$ fragile (pitfall 4); its nearly parallel rows are, and for this $A$ that is exactly what $\kappa$ picks up.

![Two panels, each showing a 2 by 2 system as crossing lines in the input plane. Left: the lines 2 x1 + x2 = 5 and x1 + 3 x2 = 10 cross at a wide angle at the dot x = (1, 3); a dashed copy of the second line nudged to 10.5 moves the crossing only to the open circle at (0.9, 3.2). Right: the lines x1 + x2 = 2 and x1 + 1.0001 x2 = 2.0001 lie on top of each other; nudging the second to 2.0002 slides the crossing from the dot at (1, 1) to the open circle at (0, 2), a move of 1.41 marked by a red arrow beside the line.](figures/ill-conditioning.svg)

**Figure 5.** Row pictures of a well-conditioned and an ill-conditioned $2\times2$ system, each with its $b_2$ nudged. (a) The $2\times 2$ worked example: the rows $2x_1 + x_2 = 5$ and $x_1 + 3x_2 = 10$ cross at $45^\circ$, $\kappa = 2.6$, and raising $b_2$ from $10$ to $10.5$ (5 %) moves the solution from $(1, 3)$ to $(0.9, 3.2)$ — by $0.22$, or 7 %. (b) The conditioning example: the rows $x_1 + x_2 = 2$ and $x_1 + 1.0001\,x_2 = 2.0001$ are only $0.003^\circ$ apart, so all three lines coincide on screen; $\kappa = 4\times 10^4$, and the invisible nudge of $b_2$ to $2.0002$ (0.005 %) slides the crossing from $(1, 1)$ to $(0, 2)$, a move of $1.41$ — 100 %. The nudge in (a) is the control: the same kind of change, harmless because the rows are far from parallel.

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

*From a line to a plane.* With a single column $\mathbf{a}$ this is the projection problem of [Module 01](../01-vectors-and-geometry/README.md) (Section 5): $\lVert x\mathbf{a} - \mathbf{y}\rVert^2$ is a parabola in the scalar $x$, minimised where the residual is perpendicular to $\mathbf{a}$, i.e. $(\mathbf{a}\cdot\mathbf{a})\,\hat x = \mathbf{a}\cdot\mathbf{y}$ — the projection coefficient quoted in the Prerequisites (for the $t$ column of the worked example alone, $14\,\hat x = 18$). With several columns the same condition must hold for every column at once: $\mathbf{a}_j\cdot(\mathbf{y} - A\hat{\mathbf{x}}) = 0$ for each $j$, and stacking those $n$ equations is $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}$. The columns need not be orthogonal to each other — in the worked example $A^\top A$ has off-diagonal entries $6$ — and the normal equations handle that coupling, which the one-column formula applied column by column cannot.

*Geometric derivation: why the residual must be orthogonal.* The residual $\mathbf{r} = \mathbf{y} - A\hat{\mathbf{x}}$ must be orthogonal to every column of $A$. Here is the reason: nudge $\hat{\mathbf{x}}$ by a small $s$ along $\mathbf{e}_j$, so that the prediction moves by $s\,\mathbf{a}_j$ and the residual becomes $\mathbf{r} - s\mathbf{a}_j$, with squared length $\lVert\mathbf{r}\rVert^2 - 2s\,(\mathbf{a}_j\cdot\mathbf{r}) + s^2\lVert\mathbf{a}_j\rVert^2$. If $\mathbf{a}_j\cdot\mathbf{r} \neq 0$, a small $s$ of the right sign makes the middle term dominate and the error *drops*, so $\hat{\mathbf{x}}$ was not the minimiser; at the minimum every slope $-2\,\mathbf{a}_j\cdot\mathbf{r}$ is zero. Orthogonal to every column means $A^\top \mathbf{r} = \mathbf{0}$, i.e. $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}$, which rearranges to the **normal equations**

$$
A^\top A\,\hat{\mathbf{x}} = A^\top \mathbf{y}, \qquad \hat{\mathbf{x}} = (A^\top A)^{-1}A^\top\mathbf{y}.
$$

*Calculus derivation (preview of Module 04).* Expand $L(\mathbf{x}) = \lVert A\mathbf{x} - \mathbf{y}\rVert^2 = \mathbf{x}^\top A^\top A\mathbf{x} - 2\mathbf{y}^\top A\mathbf{x} + \mathbf{y}^\top\mathbf{y}$; the gradient is $\nabla L = 2A^\top A\mathbf{x} - 2A^\top\mathbf{y}$; set it to zero and you get the same equations. $A^\top A$ is PSD (Section 4), so this stationary point is a minimum.

*The hat matrix.* The fitted values are $\hat{\mathbf{y}} = A\hat{\mathbf{x}} = H\mathbf{y}$ with the **hat matrix** $H = A(A^\top A)^{-1}A^\top$, the orthogonal projector onto $\mathcal{C}(A)$: $\hat{\mathbf{y}}$ is the shadow of $\mathbf{y}$ on the column space (Figure 6(b)). Three properties, each with a two-line reason. *Symmetric:* $H^\top = \big(A(A^\top A)^{-1}A^\top\big)^\top = A\big((A^\top A)^{-1}\big)^\top A^\top = H$, because $A^\top A$ is symmetric and so is its inverse. *Idempotent:* $H^2 = A(A^\top A)^{-1}(A^\top A)(A^\top A)^{-1}A^\top = H$ — the middle collapses — which says geometrically that projecting a vector already in the plane changes nothing, so $H\hat{\mathbf{y}} = \hat{\mathbf{y}}$. *Trace equals rank:* $\operatorname{tr} H = \operatorname{tr}\big(A(A^\top A)^{-1}A^\top\big) = \operatorname{tr}\big((A^\top A)^{-1}A^\top A\big) = \operatorname{tr} I_n = n$, using $\operatorname{tr}(XY) = \operatorname{tr}(YX)$; so the trace is the number of parameters, and the residual degrees of freedom are $m - n$, the dimension of the space the residual lives in. The diagonal entries $h_{ii}$ are the **leverages**: $h_{ii}$ is how much observation $i$ pulls its own fitted value, and they average $n/m$. Finally $I - H$ projects onto the orthogonal complement of $\mathcal{C}(A)$, giving the residuals $\mathbf{r} = (I - H)\mathbf{y}$.

*Pythagoras and $R^2$.* Because $\mathbf{r}$ is orthogonal to $\hat{\mathbf{y}}$ (which lies in $\mathcal{C}(A)$), $\lVert\mathbf{y}\rVert^2 = \lVert\hat{\mathbf{y}}\rVert^2 + \lVert\mathbf{r}\rVert^2$: for the worked example, $25 = 24.3 + 0.7$ (Figure 6(b)). With an intercept column the ones vector lies in $\mathcal{C}(A)$, so the residuals sum to zero and the same split holds around the mean $\bar y$: $\sum_i (y_i - \bar y)^2 = \sum_i (\hat y_i - \bar y)^2 + \sum_i r_i^2$, here $4.75 = 4.05 + 0.70$. The explained share $R^2 = 1 - 0.70/4.75 = 0.853$ is the coefficient of determination that [Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md) reports — in Module 01's language, the squared cosine of the angle between the centred $\mathbf{y}$ and its projection.

*When the normal equations fail.* Since $\mathbf{x}^\top (A^\top A)\mathbf{x} = \lVert A\mathbf{x}\rVert^2$, $(A^\top A)\mathbf{x} = \mathbf{0}$ forces $A\mathbf{x} = \mathbf{0}$ and conversely: $\mathcal{N}(A^\top A) = \mathcal{N}(A)$. So $A^\top A$ is invertible exactly when $A$ has full column rank — independent columns. If they are dependent, the projection $\hat{\mathbf{y}}$ of $\mathbf{y}$ onto $\mathcal{C}(A)$ is still unique (a plane's shadow does not depend on which vectors you used to describe the plane), but $\hat{\mathbf{x}}$ is not: any null vector of $A$ can be added to it without moving $\hat{\mathbf{y}}$. That is precisely the dummy-variable trap of Section 5 — and the duplicated weight column of Case study 1 — and the reason `lstsq` has to pick one solution (the minimum-norm one).

*Conditioning.* $\kappa(A^\top A) = \kappa(A)^2$, because the singular values of $A^\top A$ are the squares of those of $A$ (Module 03): for the worked design, $\kappa(A) = 3.76$ and $\kappa(A^\top A) = 14.1$ (eigenvalues $16.81$ and $1.19$). Harmless here; for the 50-wavelength design that Case study 2 warns against, the squaring is what turns a survivable $\kappa$ into a fatal one. This is the practical weakness of the normal equations.

*In practice.* Do *not* form $(A^\top A)^{-1}$. `np.linalg.lstsq(A, y)` uses an SVD, and `Q, R = np.linalg.qr(A)` gives $A = QR$ with orthonormal columns in $Q$, so that $\hat{\mathbf{x}}$ solves the triangular system $R\hat{\mathbf{x}} = Q^\top\mathbf{y}$ (`np.linalg.solve(R, Q.T @ y)`). Both work with $A$ itself, never with $A^\top A$, so the condition number is not squared. Form $H$ only for diagnostics on small $m$ (it is $m \times m$), and get the leverages without it as `np.sum(Q**2, axis=1)`, the squared row norms of $Q$ — because $H = QQ^\top$ — which for the worked design gives $(0.7, 0.3, 0.3, 0.7)$.

**Worked example.** Fit $y = c + m\,t$ to the points $(t, y) = (0, 1), (1, 2), (2, 2), (3, 4)$. The design matrix has an intercept column and a $t$ column:

$$
A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 1 \\ 2 \\ 2 \\ 4 \end{pmatrix}, \quad
A^\top A = \begin{pmatrix} 4 & 6 \\ 6 & 14 \end{pmatrix}, \quad A^\top \mathbf{y} = \begin{pmatrix} 9 \\ 18 \end{pmatrix}.
$$

($A^\top A$: $4$ ones; $\sum t = 6$; $\sum t^2 = 0+1+4+9 = 14$. $A^\top\mathbf{y}$: $\sum y = 9$; $\sum t y = 0 + 2 + 4 + 12 = 18$.) Solve $4c + 6m = 9$, $6c + 14m = 18$: multiply the first by $1.5$ to get $6c + 9m = 13.5$; subtract from the second: $5m = 4.5$, $m = 0.9$; then $4c = 9 - 5.4 = 3.6$, $c = 0.9$. The line is $\hat y = 0.9 + 0.9\,t$. Fitted values $(0.9, 1.8, 2.7, 3.6)$, residuals $(0.1, 0.2, -0.7, 0.4)$. Check orthogonality: residuals sum to $0.1 + 0.2 - 0.7 + 0.4 = 0$ (orthogonal to the ones column) and $\sum t\, r = 0 + 0.2 - 1.4 + 1.2 = 0$ (orthogonal to the $t$ column). The residual sum of squares is $0.01 + 0.04 + 0.49 + 0.16 = 0.70$, and no other line does better: the line $y = 1 + t$, for instance, has fitted values $(1, 2, 3, 4)$, residuals $(0, 0, -1, 0)$ and RSS $= 1.00$ (Figure 6(a)).

The hat matrix for this design: $(A^\top A)^{-1} = \tfrac{1}{20}\begin{pmatrix} 14 & -6 \\ -6 & 4\end{pmatrix}$ (determinant $4\cdot 14 - 36 = 20$, 2×2 formula), and row $i$ of $A$ is $(1, t_i)$, so $H_{ij} = \big(14 - 6(t_i + t_j) + 4\,t_i t_j\big)/20$:

$$
H = \frac{1}{10}\begin{pmatrix} 7 & 4 & 1 & -2 \\ 4 & 3 & 2 & 1 \\ 1 & 2 & 3 & 4 \\ -2 & 1 & 4 & 7 \end{pmatrix}.
$$

Check the first fitted value: row 1 times $\mathbf{y}$ is $0.7\cdot 1 + 0.4\cdot 2 + 0.1\cdot 2 - 0.2\cdot 4 = 0.7 + 0.8 + 0.2 - 0.8 = 0.9$. $H$ is symmetric by inspection, its diagonal $(0.7, 0.3, 0.3, 0.7)$ holds the leverages, and its trace is $2$, the number of parameters. Pythagoras: $\lVert\mathbf{y}\rVert^2 = 1 + 4 + 4 + 16 = 25$, $\lVert\hat{\mathbf{y}}\rVert^2 = 0.81 + 3.24 + 7.29 + 12.96 = 24.3$, $\lVert\mathbf{r}\rVert^2 = 0.70$, and $24.3 + 0.7 = 25$ (Figure 6(b)); around the mean $\bar y = 2.25$, $\mathrm{TSS} = 4.75$ and $\mathrm{ESS} = 4.05$, so $R^2 = 1 - 0.70/4.75 = 0.853$.

![Two panels. Left: four data points at (0, 1), (1, 2), (2, 2), (3, 4), the fitted line through them with short red vertical residual segments labelled +0.1, +0.2, -0.7, +0.4, and a grey dashed competing line y = 1 + t. Right: a right triangle drawn true to scale: a black arrow y of length 5 from the origin, its blue shadow y-hat of length 4.93 along a light horizontal band marked as the column space seen edge-on, and a red vertical residual arrow of length 0.84 from the tip of y-hat up to the tip of y with a right-angle marker; a grey dot at 5.48 on the horizontal axis marks the competing fit (1, 2, 3, 4).](figures/least-squares-projection.svg)

**Figure 6.** The worked example twice. (a) Data space: the four points, the least-squares line $\hat y = 0.9 + 0.9\,t$ with its residuals $+0.1, +0.2, -0.7, +0.4$ (RSS $0.70$, summing to zero), and the competing line $y = 1 + t$, whose RSS of $1.00$ is worse. (b) The same numbers as vectors in $\mathbb{R}^4$, drawn true to scale in the plane spanned by $\hat{\mathbf{y}}$ and $\mathbf{r}$: $\mathbf{y}$ has length 5, its shadow $\hat{\mathbf{y}} = H\mathbf{y}$ on the column space has length $\sqrt{24.3} = 4.93$, and the residual $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$, of length $\sqrt{0.7} = 0.84$, meets the column space at a right angle, so $25 = 24.3 + 0.7$ and the angle between $\mathbf{y}$ and $\hat{\mathbf{y}}$ is $9.6^\circ$. The shaded band is only a visual cue: the two-dimensional column space is seen edge-on in this slice. The four residuals of (a) are the components of the single vector $\mathbf{r}$ in (b). The competing fit $(1, 2, 3, 4)$ happens to lie on the $\hat{\mathbf{y}}$ axis because $\hat{\mathbf{y}} = 0.9\,(1, 2, 3, 4)$ exactly; its distance to $\mathbf{y}$, $1.00$, is longer than $0.84$, as it is for every other point of $\mathcal{C}(A)$.

**Real-life example (healthcare / pharmacology).** In a dose-finding study, four groups receive 0, 10, 20 and 30 mg of a drug and show mean biomarker reductions of 1, 2, 2 and 4 units (same numbers as above, with $t$ in units of 10 mg). The fitted slope of $0.9$ units per 10 mg is the estimated dose–response, and the intercept $0.9$ is the placebo effect. The hat matrix tells you how much each observation influences its own fitted value: its diagonal entries (the *leverages*) for this design are $0.7, 0.3, 0.3, 0.7$ — the outermost doses have the most pull, which is why dose-ranging designs put replicates at the extremes. The identical calculation calibrates a sensor (voltage readings against known reference concentrations) or converts a thermistor's resistance to temperature.

**In ML/AI this is …** `sklearn.linear_model.LinearRegression`, which calls `scipy.linalg.lstsq` under the hood; the closed form that gradient descent on mean-squared error converges to (Module 05 shows that the Hessian of this loss is $2A^\top A/m$, so its conditioning sets the learning-rate budget); the "OLS as projection" view developed fully in [Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md); and the last-layer readout in many self-supervised evaluation protocols, where a linear probe is just least squares (or logistic regression) on frozen features.

## 8. Change of basis and similarity

**Intuition.** A vector is a geometric object; its *coordinates* depend on which rulers you lay down. The point two steps east and one step north is $(2, 1)$ in the compass basis but $(1.5, 0.5)$ in a basis whose first axis points north-east and second points south-east. Nothing about the point changed — only the description. The same is true of linear maps: a shear that looks messy in standard coordinates can look like a plain scaling in a cleverly chosen basis. Finding that basis is the entire business of eigendecomposition (Module 03), and PCA is the special case where the clever basis is chosen from the data itself.

**Definition.** Let the columns of $P \in \mathbb{R}^{n\times n}$ be a basis $\mathbf{p}_1, \dots, \mathbf{p}_n$ (so $P$ is invertible). A vector with standard coordinates $\mathbf{x}$ has coordinates $\mathbf{c}$ in the new basis when

$$
\mathbf{x} = P\mathbf{c} = c_1\mathbf{p}_1 + \dots + c_n\mathbf{p}_n, \qquad\text{so}\qquad \mathbf{c} = P^{-1}\mathbf{x}.
$$

*Coordinates are coefficients.* Recall [Module 01](../01-vectors-and-geometry/README.md), Section 1: a basis gives every vector exactly one recipe as a combination of the basis vectors, and the coefficients of that recipe are its coordinates. $P$ collects the basis as columns, so $P\mathbf{c}$ is the recipe evaluated — the column picture of Section 1. With the worked basis below, $P(2, 1)^\top = 2\,(1, 1) + 1\,(1, -1) = (3, 1)$ (Figure 7(a)).

*Two directions, one rule.* $P$ converts $\mathbf{p}$-coordinates into standard ones — a combination, no solving — and $P^{-1}$ converts standard coordinates into $\mathbf{p}$-coordinates, which is a linear system: $\mathbf{c}$ is `np.linalg.solve(P, x)`. So the matrix whose columns are the *new* basis vectors turns *new* coordinates into *old* ones. Most people guess the opposite at first; the check is a basis vector itself: $\mathbf{c} = \mathbf{e}_1$ must give $\mathbf{x} = \mathbf{p}_1$, and indeed $P\mathbf{e}_1$ is the first column of $P$.

*Orthonormal bases are the easy case.* If $P^\top P = I$ then $P^{-1} = P^\top$ and $c_i = \mathbf{p}_i\cdot\mathbf{x}$: each new coordinate is a dot product with a basis vector, exactly the projection formula from Module 01 (Section 6). Watch the hypothesis. The worked $\mathbf{p}_1, \mathbf{p}_2$ are orthogonal but each has length $\sqrt 2$, so $P^\top P = 2I$ and $P^{-1} = \tfrac12 P^\top$ — which is where the $\tfrac12$ in the worked example comes from; applying the shortcut $\mathbf{c} = P^\top\mathbf{x}$ to $\mathbf{x} = (3, 1)^\top$ gives $(4, 2)^\top$, twice the right answer. Normalising the basis to $(1, 1)/\sqrt 2$ and $(1, -1)/\sqrt 2$ makes $P$ orthogonal (the coordinates of $\mathbf{x}$ become $(2\sqrt 2, \sqrt 2)$ — the same arrow measured with shorter rulers) and leaves the diagonal matrix found below unchanged.

*Deriving $A' = P^{-1}AP$ by following one vector.* A linear map with standard matrix $A$ has, in the new coordinates, the matrix

$$
A' = P^{-1} A P \qquad (\text{read right to left: convert } \mathbf{c} \to \mathbf{x},\ \text{apply } A,\ \text{convert back}).
$$

Follow the worked $\mathbf{c} = (2, 1)^\top$ through the three steps: convert, $\mathbf{x} = P\mathbf{c} = (3, 1)^\top$; apply, $A\mathbf{x} = (7, 5)^\top$; convert back, $P^{-1}(7, 5)^\top = \tfrac12(12, 2)^\top = (6, 1)^\top$. In $\mathbf{p}$-coordinates the map sent $(2, 1)$ to $(6, 1) = (3\cdot 2,\ 1\cdot 1)$: the first coordinate tripled, the second stayed, and the matrix that does this is $\mathrm{diag}(3, 1) = P^{-1}AP$ (Figure 7(b)). Nothing happened to the map; only its description changed.

*Why the invariants are invariant.* $A$ and $A'$ are called **similar**, and similar matrices share everything that does not depend on coordinates. Determinant: $\det(P^{-1}AP) = \det(P)^{-1}\det(A)\det(P) = \det A$. Trace: $\operatorname{tr}\big(P^{-1}(AP)\big) = \operatorname{tr}\big((AP)P^{-1}\big) = \operatorname{tr} A$ by $\operatorname{tr}(XY) = \operatorname{tr}(YX)$. Rank: multiplying by an invertible matrix neither creates nor destroys directions, so $\operatorname{rank}(P^{-1}AP) = \operatorname{rank} A$. Eigenvalues: if $A\mathbf{v} = \lambda\mathbf{v}$ then $A'(P^{-1}\mathbf{v}) = P^{-1}AP\,P^{-1}\mathbf{v} = P^{-1}A\mathbf{v} = \lambda\,(P^{-1}\mathbf{v})$ — the same eigenvalue, with the eigenvector merely re-expressed in the new coordinates. The entries themselves are *not* invariant, and the converse fails: equal eigenvalues do not make two matrices similar (pitfall 7).

*What a diagonal form buys.* The goal of diagonalisation is to find $P$ such that $P^{-1}AP = D$ is diagonal — then the map is "stretch along $\mathbf{p}_1$ by $d_1$, along $\mathbf{p}_2$ by $d_2$, …", and powers are cheap: $A = PDP^{-1}$ gives $A^k = (PDP^{-1})(PDP^{-1})\cdots(PDP^{-1}) = PD^kP^{-1}$, because every inner $P^{-1}P$ cancels and $D^k$ just raises each diagonal entry to the $k$-th power. The transition-matrix powers of Section 9 and the gradient-descent analysis of [Module 05](../../part-2-calculus-and-optimization/05-optimization/README.md) use exactly this.

*In practice.* `A_new = np.linalg.solve(P, A @ P)` — never `np.linalg.inv(P) @ A @ P` (Section 6). `np.linalg.eig(A)` returns the eigenvalues together with a matrix whose columns diagonalise $A$: for the worked $A$ it returns $3$ and $1$ with the columns $(1, 1)/\sqrt 2$ and $(-1, 1)/\sqrt 2$ (up to sign and order), the normalised $\mathbf{p}_1$ and $-\mathbf{p}_2$. That is how [Module 03](../03-eigenvalues-svd-and-pca/README.md) *finds* the basis this section assumes.

**Worked example.** Take the basis $\mathbf{p}_1 = (1, 1)^\top$, $\mathbf{p}_2 = (1, -1)^\top$, so $P = \begin{pmatrix} 1 & 1 \\ 1 & -1\end{pmatrix}$ with $P^{-1} = \tfrac12\begin{pmatrix} 1 & 1 \\ 1 & -1\end{pmatrix}$ (check: $\det P = -2$, use the 2×2 formula). The vector $\mathbf{x} = (3, 1)^\top$ has new coordinates $\mathbf{c} = P^{-1}\mathbf{x} = \tfrac12(4, 2)^\top = (2, 1)^\top$, and indeed $2\,(1,1) + 1\,(1,-1) = (3, 1)$ (Figure 7(a)). Now the symmetric map $A = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$: $AP = \begin{pmatrix} 3 & 1 \\ 3 & -1\end{pmatrix}$, and $P^{-1}AP = \tfrac12\begin{pmatrix} 1 & 1 \\ 1 & -1\end{pmatrix}\begin{pmatrix} 3 & 1 \\ 3 & -1 \end{pmatrix} = \tfrac12\begin{pmatrix} 6 & 0 \\ 0 & 2\end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & 1\end{pmatrix}$. In the $(\mathbf{p}_1, \mathbf{p}_2)$ basis, $A$ is a pure scaling: ×3 along the diagonal direction, ×1 along the anti-diagonal. Trace check: $2 + 2 = 3 + 1$; determinant check: $4 - 1 = 3 \cdot 1$. Following $\mathbf{x}$ itself (Figure 7(b)): $A\mathbf{x} = (7, 5)^\top$ has $\mathbf{p}$-coordinates $P^{-1}(7, 5)^\top = (6, 1)^\top = (3\cdot 2,\ 1\cdot 1)$, as $\mathrm{diag}(3, 1)$ predicts from $\mathbf{c} = (2, 1)^\top$. Powers come for free: $A^2 = P\,\mathrm{diag}(9, 1)\,P^{-1} = \tfrac12\begin{pmatrix} 9 & 1 \\ 9 & -1\end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1\end{pmatrix} = \begin{pmatrix} 5 & 4 \\ 4 & 5\end{pmatrix}$, matching the direct product $\begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}^2 = \begin{pmatrix} 4 + 1 & 2 + 2 \\ 2 + 2 & 1 + 4\end{pmatrix}$. We have just diagonalised $A$ by hand; Module 03 explains how to *find* such a $P$ in general.

![Two panels with the same axes, both overlaid with the skewed grey grid of integer combinations of p1 = (1, 1) and p2 = (1, -1). Left: arrows p1 and p2, a lighter walk of two steps along p1 to (2, 2) and one step along p2 to (3, 1), and the bold red arrow x = (3, 1). Right: arrows A p1 = (3, 3) and A p2 = (1, -1), a lighter walk of six steps along p1 to (6, 6) and one step along p2 to (7, 5), the bold red arrow Ax = (7, 5), and a faint copy of x.](figures/change-of-basis.svg)

**Figure 7.** The worked example on the grid of the $\mathbf{p}$-basis (grey lines: the grid of integer combinations of $\mathbf{p}_1 = (1, 1)$ and $\mathbf{p}_2 = (1, -1)$). (a) One arrow, two descriptions: $\mathbf{x} = (3, 1)$ in standard coordinates is the walk $2\mathbf{p}_1$ then $1\mathbf{p}_2$, so its $\mathbf{p}$-coordinates are $(2, 1)$. (b) The same map $A = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$ applied: $A\mathbf{p}_1 = (3, 3) = 3\mathbf{p}_1$ and $A\mathbf{p}_2 = (1, -1) = 1\mathbf{p}_2$, so $A\mathbf{x} = (7, 5)$ is the walk $6\mathbf{p}_1$ then $1\mathbf{p}_2$ — $\mathbf{p}$-coordinates $(6, 1) = (3\cdot 2,\ 1\cdot 1)$. Follow the chain $\mathbf{c} = (2, 1) \to P\mathbf{c} = (3, 1) \to A(3, 1) = (7, 5) \to P^{-1}(7, 5) = (6, 1)$: the first $\mathbf{p}$-coordinate tripled and the second stayed, which is exactly $\mathrm{diag}(3, 1) = P^{-1}AP$. Same map, different coordinates (pitfall 7).

**Real-life example (physics / signal processing).** An accelerometer in a phone reports acceleration in the phone's own axes. To know whether the phone is being lifted or slid, the software must re-express the vector in the room's axes — a change of basis by a rotation matrix $R$ obtained from the gyroscope: $\mathbf{a}_{\text{room}} = R\,\mathbf{a}_{\text{phone}}$. In audio, the discrete Fourier transform is a change of basis from "amplitude at each time sample" to "amplitude of each frequency": the new basis vectors are sampled sinusoids, the transform matrix is (up to scaling) unitary — the complex-valued analogue of orthogonal, so it too preserves lengths and dot products — and a low-pass filter that is a dense convolution in time becomes a *diagonal* matrix in frequency — the same map, made simple by the right coordinates.

**In ML/AI this is …** PCA (`sklearn.decomposition.PCA`): `transform(X)` computes $(X - \bar{\mathbf{x}})\,W$ where the columns of $W$ are the principal directions, i.e. the data re-expressed in the covariance eigenbasis (Module 03); whitening, where the new coordinates are $\mathbf{c} = \Sigma^{-1/2}\mathbf{x}$ (so $P = \Sigma^{1/2}$), which makes the covariance the identity; the diagonalisation of the Hessian that explains why gradient descent's speed is governed by the ratio of its largest to smallest eigenvalue (Module 05); and the Fourier-domain view of convolutional layers.

## 9. Matrices as graphs and Markov chains

**Intuition.** A matrix can also be a *network*. Put a 1 in entry $(i, j)$ when there is an edge from node $i$ to node $j$ and you have the adjacency matrix. Multiply it by itself and entry $(i, j)$ counts the two-step walks from $i$ to $j$ — matrix multiplication *is* path counting, because $\sum_k A_{ik}A_{kj}$ sums over intermediate nodes $k$. Replace the ones by probabilities of moving along each edge and the matrix becomes a transition matrix; multiplying a distribution by it moves the probability mass one step forward in time. Keep multiplying and, for well-behaved chains, the distribution forgets where it started.

**Definition.** For a graph on $n$ nodes the **adjacency matrix** $A \in \{0,1\}^{n\times n}$ has $A_{ij} = 1$ iff there is an edge $i \to j$; $(A^k)_{ij}$ counts walks of length exactly $k$ from $i$ to $j$. A **(row-)stochastic transition matrix** $P$ has $P_{ij} = \Pr(\text{next} = j \mid \text{now} = i) \ge 0$ with every row summing to 1. If $\boldsymbol\pi_t$ is the row vector of state probabilities at time $t$,

$$
\boldsymbol\pi_{t+1} = \boldsymbol\pi_t P, \qquad \boldsymbol\pi_t = \boldsymbol\pi_0 P^t, \qquad (P^k)_{ij} = \Pr(\text{state } j \text{ after } k \text{ steps} \mid \text{start } i).
$$

A **stationary distribution** satisfies $\boldsymbol\pi P = \boldsymbol\pi$ — an eigenvector with eigenvalue 1 (Module 03). If the chain is irreducible and aperiodic, $\boldsymbol\pi_0 P^t \to \boldsymbol\pi$ for every starting $\boldsymbol\pi_0$ and the rows of $P^t$ all converge to $\boldsymbol\pi$.

**Worked example.** Two-state weather: from Sunny, tomorrow is Sunny with probability 0.9; from Rainy, tomorrow is Sunny with probability 0.5. $P = \begin{pmatrix} 0.9 & 0.1 \\ 0.5 & 0.5 \end{pmatrix}$. Start rainy, $\boldsymbol\pi_0 = (0, 1)$. Then $\boldsymbol\pi_1 = (0.5, 0.5)$; $\boldsymbol\pi_2 = (0.5\cdot0.9 + 0.5\cdot0.5,\ 0.5\cdot0.1 + 0.5\cdot0.5) = (0.70, 0.30)$; $\boldsymbol\pi_3 = (0.63 + 0.15,\ 0.07 + 0.15) = (0.78, 0.22)$ (Figure 8(b)). Stationary: solve $\pi_S = 0.9\pi_S + 0.5\pi_R$, i.e. $0.1\pi_S = 0.5\pi_R$, with $\pi_S + \pi_R = 1$: $\boldsymbol\pi = (5/6, 1/6) \approx (0.833, 0.167)$. Each step closes the gap to $\boldsymbol\pi$ by a factor of $0.4$ (the second eigenvalue, $0.9 + 0.5 - 1$): the gap in $\pi_t(\text{Sunny})$ from the rainy start is $0.833, 0.333, 0.133, 0.053, \dots$, and from the sunny start $\boldsymbol\pi_0 = (1, 0)$ it is $0.167, 0.067, 0.027, \dots$ — the same factor from either side (Figure 8(c)), and by day 10 both gaps are below $10^{-4}$. For path counting, the three-node path graph $1 - 2 - 3$ has $A = \begin{pmatrix} 0&1&0\\1&0&1\\0&1&0\end{pmatrix}$ and $A^2 = \begin{pmatrix} 1&0&1\\0&2&0\\1&0&1\end{pmatrix}$: two 2-step walks return to node 2 (via 1 or via 3), one walk goes from 1 to 3.

![Three panels. Left: two circles labelled Sunny and Rainy joined by curved arrows labelled 0.1 from Sunny to Rainy and 0.5 from Rainy to Sunny, each with a self-loop labelled 0.9 and 0.5. Middle: the probability of Sunny over days 0 to 8 from a rainy start, rising 0, 0.5, 0.70, 0.78 and levelling off, and from a sunny start, falling 1, 0.9, 0.86 and levelling off, both approaching a dashed line at 5/6. Right: the distance to 5/6 on a logarithmic axis, two parallel straight lines of points, one per start: the rainy-start points along the dashed line 5/6 times 0.4 to the power t, the sunny-start points on a parallel line below it.](figures/markov-chain.svg)

**Figure 8.** The two-state weather chain. (a) The transition matrix as a diagram: the arrows out of each state carry its row of $P = \begin{pmatrix} 0.9 & 0.1 \\ 0.5 & 0.5\end{pmatrix}$ and sum to 1. (b) $\pi_t(\text{Sunny})$ from the rainy start $(0, 1)$ — $0, 0.5, 0.70, 0.78, 0.812, 0.825, \dots$ — and from the sunny start $(1, 0)$ — $1, 0.9, 0.86, 0.844, \dots$; both settle on the stationary value $5/6 \approx 0.833$ (dashed), so the chain forgets where it started. (c) The gap $\lvert\pi_t(\text{Sunny}) - 5/6\rvert$ on a log scale: from the rainy start it is $0.833, 0.333, 0.133, 0.053, \dots$, each day exactly $0.4$ times the day before, so both series are straight lines of slope $\log 0.4$ and the dashed line is $\tfrac56\cdot 0.4^t$. Days 4 onwards extend the worked example's $\boldsymbol\pi_1, \boldsymbol\pi_2, \boldsymbol\pi_3$. The same multiplication applied to an adjacency matrix counts paths, as $A^2$ for the path graph above shows.

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

A hospital fits a linear model predicting 30-day readmission risk from five features: age, systolic blood pressure, BMI, weight (kg) and height (cm). The analyst also adds "weight in pounds" exported from a legacy system. The design matrix $X$ is $n \times 7$ (six features plus intercept). Weight-in-pounds is $2.2046 \times$ weight-in-kg, so two columns are exact multiples: $\operatorname{rank}(X) = 6 < 7$, $X^\top X$ is singular, and `np.linalg.solve` either raises `LinAlgError: Singular matrix` or — more often, because rounding while forming $X^\top X$ leaves tiny non-zero pivots — silently returns a meaningless coefficient vector; `np.linalg.matrix_rank(X)` (6, not 7) and `np.linalg.cond` (around $10^{16}$) expose the problem, and `scipy.linalg.solve` at least emits a `LinAlgWarning`. Switching to `lstsq` "works" — it returns the minimum-norm solution — but the two weight coefficients are now arbitrary: adding $+5$ to one and $-5/2.2046$ to the other leaves every prediction unchanged, so neither coefficient can be interpreted or reported in a paper. After dropping the duplicate column the rank is 6 and the fit is unique. The subtler problem is BMI = weight / height², which is a *nonlinear* function of two other columns; the columns are not exactly collinear, but on adults they are highly correlated, and the condition number of $X^\top X$ comes out at about $3 \times 10^{7}$. The standard errors of the BMI and weight coefficients inflate roughly as $\sqrt{\text{VIF}}$, with variance inflation factors above 50. Standardising the features already brings the condition number down to the hundreds (most of the $3\times10^{7}$ came from the mismatched units and the intercept column); a ridge penalty $\lambda I$ on top lowers it further to $(\lambda_{\max} + \lambda)/(\lambda_{\min} + \lambda)$, where $\lambda_{\max}$ and $\lambda_{\min}$ are the extreme eigenvalues of the standardised $X^\top X$, and stabilises the coefficients at the cost of a small bias. The lesson: rank deficiency is a hard failure, *near* rank deficiency is a soft one, and both are properties of the data matrix, not of the algorithm.

### Case study 2 — Calibrating a spectrometer with least squares

A manufacturing lab calibrates a near-infrared spectrometer to predict the moisture content of grain. For 60 calibration samples of known moisture (8 %–20 %, measured by oven drying) it records absorbance at 4 wavelengths, giving $A \in \mathbb{R}^{60\times 5}$ (four absorbances plus an intercept column) and $\mathbf{y} \in \mathbb{R}^{60}$. The normal equations are a $5\times 5$ system; solving them by QR gives coefficients such as $\hat{\mathbf{x}} = (3.1, 42.7, -18.2, 9.5, 0.8)^\top$ with a residual standard deviation of 0.35 percentage points. Two checks from this module make the calibration trustworthy. First, the hat matrix: its trace is 5 (the number of parameters) and the residual degrees of freedom are $60 - 5 = 55$; a sample with leverage $h_{ii} = 0.31$ — almost four times the average $5/60 \approx 0.083$ — is flagged as influential and turns out to be a mislabelled wet sample. Second, conditioning: wavelengths 2 and 3 are adjacent bands with correlation 0.998, so $\kappa(A) \approx 2{,}000$ and $\kappa(A^\top A) \approx 4\times 10^6$; forming $A^\top A$ explicitly and inverting it loses six digits, which here is survivable but would not be with 50 wavelengths. In production, the lab uses `np.linalg.lstsq` (SVD-based) and re-estimates once a month on fresh samples — a prediction for a new spectrum $\mathbf{a}$ is simply the dot product $\mathbf{a}^\top \hat{\mathbf{x}}$, i.e. one row of the row picture.

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
| Four fundamental subspaces, solution sets | Minimum-norm `lstsq`, SVD bases (Module 03), identifiable regression coefficients (Module 13) | Row space = directions the data determine, null space = directions they cannot; $\mathcal{N}(A^\top)$ tests solvability |
| Solving $A\mathbf{x}=\mathbf{b}$, conditioning | Ridge/OLS solvers, Newton steps, GP posteriors, policy evaluation | `solve` not `inv`; condition number sets achievable precision and GD speed |
| Normal equations, hat matrix | `LinearRegression`, linear probes, leverage diagnostics, $R^2$ | Closed-form fit; projection view explains residual orthogonality, degrees of freedom and $R^2$; $A^\top A$ invertible iff columns independent |
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
| Rank–nullity | $\operatorname{rank} A + \dim\mathcal{N}(A) = n$ (columns); row rank = column rank = number of pivots |
| Null space by elimination | Reduce to echelon form; one null vector per free column (that free variable 1, the others 0); the pivot columns of the *original* $A$ are a basis of $\mathcal{C}(A)$ |
| Four fundamental subspaces | $\mathcal{C}(A^\top) \perp \mathcal{N}(A)$ in $\mathbb{R}^n$ (dimensions $r$, $n - r$); $\mathcal{C}(A) \perp \mathcal{N}(A^\top)$ in $\mathbb{R}^m$ (dimensions $r$, $m - r$) |
| Solvability | $A\mathbf{x} = \mathbf{b}$ solvable $\iff \mathbf{b} \in \mathcal{C}(A) \iff \mathbf{b} \perp \mathcal{N}(A^\top)$; unique $\iff \mathcal{N}(A) = \{\mathbf{0}\}$; all solutions $\mathbf{x}_p + \mathcal{N}(A)$ |
| Condition number | $\kappa(A) = \lVert A\rVert\lVert A^{-1}\rVert = \sigma_{\max}/\sigma_{\min}$; relative error amplification $\le \kappa\,\varepsilon$ |
| Normal equations | $A^\top A\hat{\mathbf{x}} = A^\top\mathbf{y}$; $\hat{\mathbf{x}} = (A^\top A)^{-1}A^\top\mathbf{y}$ (compute via QR/SVD); unique iff the columns are independent, since $\mathcal{N}(A^\top A) = \mathcal{N}(A)$; $\kappa(A^\top A) = \kappa(A)^2$ |
| Hat matrix | $H = A(A^\top A)^{-1}A^\top = QQ^\top$; $H^2 = H = H^\top$; $\operatorname{tr} H = \operatorname{rank} A$; leverages $h_{ii}$ average $n/m$; residual $= (I - H)\mathbf{y} \perp \mathcal{C}(A)$ |
| Pythagoras, $R^2$ | $\lVert\mathbf{y}\rVert^2 = \lVert\hat{\mathbf{y}}\rVert^2 + \lVert\mathbf{r}\rVert^2$; with an intercept, $\mathrm{TSS} = \mathrm{ESS} + \mathrm{RSS}$ and $R^2 = 1 - \mathrm{RSS}/\mathrm{TSS}$ |
| Change of basis | $\mathbf{x} = P\mathbf{c}$, $\mathbf{c} = P^{-1}\mathbf{x}$ ($= P^\top\mathbf{x}$ only when $P^\top P = I$); $A' = P^{-1}AP$ preserves det, trace, rank, eigenvalues; $P^{-1}AP = D \Rightarrow A^k = PD^kP^{-1}$ |
| Markov chain | $\boldsymbol\pi_{t+1} = \boldsymbol\pi_t P$, rows of $P$ sum to 1; stationary $\boldsymbol\pi P = \boldsymbol\pi$; $(A^k)_{ij}$ counts $k$-step walks |
| NumPy | `A @ B` product, `A * B` elementwise, `np.linalg.solve`, `np.linalg.lstsq`, `np.linalg.qr`, `np.linalg.matrix_rank`, `np.linalg.cond`, `np.linalg.det`, `scipy.linalg.null_space` |

## Self-assessment

1. **Why does the column picture of $A\mathbf{x}$ immediately tell you when $A\mathbf{x} = \mathbf{b}$ has a solution?**
   <details><summary>Answer</summary>

   Because $A\mathbf{x}$ is a linear combination of the columns with weights $x_j$, the set of all possible $A\mathbf{x}$ is exactly the span of the columns — the column space. A solution exists if and only if $\mathbf{b}$ lies in that span. If $\mathbf{b}$ is outside it, no choice of weights reaches it and you need least squares instead.

   </details>

2. **$W_1$ is $4\times 3$ and $W_2$ is $5\times 4$. Which products are defined, and what is the shape of the composed map applied to a batch $X$ of shape $(n, 3)$ stored in rows?**
   <details><summary>Answer</summary>

   $W_2W_1$ is defined ($5\times4$ times $4\times3$ gives $5\times3$); $W_1W_2$ is not (inner dimensions 3 and 5 disagree). With rows as samples you apply transposes: $X W_1^\top$ has shape $(n, 4)$ and $X W_1^\top W_2^\top = X (W_2W_1)^\top$ has shape $(n, 5)$. Without a nonlinearity the two layers equal a single $5\times 3$ layer.

   </details>

3. **A $2\times2$ matrix has determinant $-3$. What two things does this tell you about the unit square's image?**
   <details><summary>Answer</summary>

   Its area is 3 (area scaled by $\lvert\det\rvert$) and its orientation is reversed (negative sign): the matrix includes a reflection. The matrix is invertible since $\det \neq 0$.

   </details>

4. **Why is $Q^\top$ the inverse of an orthogonal matrix, and why does this make orthogonal matrices numerically pleasant?**
   <details><summary>Answer</summary>

   Orthonormal columns mean $Q^\top Q = I$ by definition of the dot products between columns; for a square matrix a one-sided inverse is the inverse, so $Q^{-1} = Q^\top$. Inverting is then free and exact (no elimination, no rounding), and because $\lVert Q\mathbf{x}\rVert = \lVert\mathbf{x}\rVert$ the condition number is 1 — errors are neither amplified nor hidden.

   </details>

5. **A data matrix has 1,000 rows and 12 columns, and `np.linalg.matrix_rank` returns 11. What does that mean practically, and what are two fixes?**
   <details><summary>Answer</summary>

   One column is an exact linear combination of the others (for example a one-hot group that sums to the intercept, or a unit-converted duplicate), so $X^\top X$ is singular and the regression coefficients are not unique — the null space direction can be added to the coefficient vector at no cost. Fixes: drop the redundant column (or one one-hot level), or add a ridge penalty $\lambda I$, which makes $X^\top X + \lambda I$ full rank.

   </details>

6. **Give two reasons to prefer `solve(A, b)` over `inv(A) @ b`.**
   <details><summary>Answer</summary>

   Cost: an LU solve is about $\tfrac23 n^3$ flops versus roughly $2n^3$ for the inverse plus the extra product. Accuracy: the triangular solves commit only the unavoidable rounding error, whereas forming the inverse and multiplying can amplify error by an extra factor related to $\kappa(A)$. A third reason is generality — `solve` extends naturally to `lstsq` for rectangular systems, where no inverse exists.

   </details>

7. **Derive the normal equations in one sentence from the geometry of projection.**
   <details><summary>Answer</summary>

   The best approximation $A\hat{\mathbf{x}}$ is the orthogonal projection of $\mathbf{y}$ onto the column space, so the residual $\mathbf{y} - A\hat{\mathbf{x}}$ is orthogonal to every column of $A$, which is the statement $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}$, i.e. $A^\top A\hat{\mathbf{x}} = A^\top\mathbf{y}$.

   </details>

8. **The hat matrix of a regression with 8 parameters on 200 observations has trace 8. Why, and what is the average leverage?**
   <details><summary>Answer</summary>

   $H$ is the orthogonal projector onto an 8-dimensional column space; the trace of a projector equals the dimension of the space it projects onto (its eigenvalues are eight 1s and 192 0s). The average diagonal entry (leverage) is therefore $8/200 = 0.04$; observations with leverage several times that are influential.

   </details>

9. **$A' = P^{-1}AP$ with $P$ invertible. Which of $\det$, trace, rank, the entries, and the eigenvalues can differ between $A$ and $A'$?**
   <details><summary>Answer</summary>

   Only the entries. Determinant ($\det P^{-1}\det A\det P = \det A$), trace (cyclic property), rank (invertible factors do not change rank) and eigenvalues (if $A\mathbf{v} = \lambda\mathbf{v}$ then $A'(P^{-1}\mathbf{v}) = \lambda P^{-1}\mathbf{v}$) are all invariant — they describe the map, not the coordinates.

   </details>

10. **For the transition matrix $P = \begin{pmatrix} 0.9 & 0.1 \\ 0.5 & 0.5 \end{pmatrix}$, why is $(1, 1)^\top$ an eigenvector of $P$ with eigenvalue 1, and what does the stationary distribution have to do with eigenvectors?**
    <details><summary>Answer</summary>

    Each row sums to 1, so $P\mathbf{1} = \mathbf{1}$: the all-ones vector is a right eigenvector with eigenvalue 1. The stationary distribution is the corresponding *left* eigenvector, $\boldsymbol\pi P = \boldsymbol\pi$ with $\sum_i \pi_i = 1$; here $\boldsymbol\pi = (5/6, 1/6)$. Powers $P^t$ converge to a matrix whose rows are all $\boldsymbol\pi$, which is why the starting state is forgotten.

    </details>

11. **A $3\times 4$ matrix row-reduces to a form with pivots in columns 1 and 3. Give the dimensions of its four fundamental subspaces, say how many null-space vectors you read off and how, and name a basis of the column space.**
    <details><summary>Answer</summary>

    Two pivots, so $r = 2$. Row space and null space live in $\mathbb{R}^4$ with dimensions $2$ and $4 - 2 = 2$; column space and left null space live in $\mathbb{R}^3$ with dimensions $2$ and $3 - 2 = 1$. The two free columns (2 and 4) give two null-space vectors: set one free variable to 1 and the other to 0, and read the pivot variables off the reduced rows. A basis of the column space is columns 1 and 3 of the *original* matrix, not of the reduced one. Solvability of $A\mathbf{x} = \mathbf{b}$ is then a single test: $\mathbf{b}$ must be orthogonal to the one vector spanning $\mathcal{N}(A^\top)$.

    </details>

12. **You fit $y \approx c + m\,t$ by the normal equations and get a unique $\hat{\mathbf{x}}$. A colleague appends a third column equal to $2t + 1$ (a unit conversion of $t$). What happens to $A^\top A$, to $\hat{\mathbf{x}}$, to $\hat{\mathbf{y}}$ and to the RSS?**
    <details><summary>Answer</summary>

    The new column is $2\,(t\text{ column}) + 1\,(\text{ones column})$, so the columns are dependent and $\mathcal{N}(A)$ is the line $s\,(1, 2, -1)$ (check: $1\cdot\mathbf{1} + 2\,\mathbf{t} - 1\cdot(2\mathbf{t} + \mathbf{1}) = \mathbf{0}$). Because $\mathcal{N}(A^\top A) = \mathcal{N}(A)$, $A^\top A$ is singular and $\hat{\mathbf{x}}$ is no longer unique — any multiple of $(1, 2, -1)$ can be added to it. But the column space is the same plane as before, so the projection $\hat{\mathbf{y}}$ of $\mathbf{y}$ onto it and the RSS are unchanged; `lstsq` returns the minimum-norm $\hat{\mathbf{x}}$. This is the dummy-variable trap in miniature.

    </details>

13. **For the worked basis $\mathbf{p}_1 = (1, 1)^\top$, $\mathbf{p}_2 = (1, -1)^\top$, a learner computes the new coordinates of $\mathbf{x} = (3, 1)^\top$ as $\mathbf{c} = P^\top\mathbf{x} = (4, 2)^\top$. What went wrong, and what are two ways to fix it?**
    <details><summary>Answer</summary>

    The shortcut $\mathbf{c} = P^\top\mathbf{x}$ needs $P^\top P = I$; here $P^\top P = 2I$ because each basis vector has length $\sqrt 2$, so $P^\top\mathbf{x} = 2\mathbf{c}$. Fix 1: solve $P\mathbf{c} = \mathbf{x}$ properly, $\mathbf{c} = P^{-1}\mathbf{x} = \tfrac12 P^\top\mathbf{x} = (2, 1)^\top$. Fix 2: normalise the basis to $(1, 1)/\sqrt 2$ and $(1, -1)/\sqrt 2$, after which $P$ is orthogonal and $\mathbf{c} = P^\top\mathbf{x} = (2\sqrt 2, \sqrt 2)^\top$ — different numbers because the rulers are shorter, the same arrow, and the same $P^{-1}AP = \mathrm{diag}(3, 1)$ for the worked $A$.

    </details>

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

**`lesson.ipynb`** walks through the module with code and figures: the column and row pictures of $A\mathbf{x}$ computed side by side; a loop-based `matmul` verified against `@`, with timing to show the cubic cost, the speed-up of NumPy's `@` over the loops, and the collapse of two linear layers into one; the unit square under rotation, scaling, shear, reflection and projection with the determinant printed for each, then an asymmetric letter pushed through the same rotation and shear in both orders, so that $AB \neq BA$ is visible while the determinants agree; orthogonal, Gram and triangular matrices and what they promise (lengths preserved, positive semidefiniteness, back-substitution); rank detection on a synthetic feature matrix with a duplicated and a one-hot-trapped column, and a loss-surface picture of what the null-space direction does (a flat valley for an exact duplicate, a long thin bowl for a sloppy one); solving systems with `solve` versus `inv`, the blow-up on an ill-conditioned matrix as $\kappa$ grows, a round ball of noise in $\mathbf{b}$ turning into a sliver of candidate solutions, and Hilbert matrices on which `inv(A) @ b` visibly fails to solve the system while `solve` does not; least squares three ways (normal equations, QR, `lstsq`) on noisy line and quadratic data, with the hat matrix checked for symmetry and idempotence, a degree-6 fit compared with the true curve and with fresh noise, and an exact-data experiment in which the normal equations lose digits much faster than QR and `lstsq`; a change of basis that diagonalises a symmetric matrix; and walk counting on a small graph and a Markov transition matrix whose powers converge to the stationary distribution, with trajectories from all three starting states and the $|\lambda_2|^t$ rate at which the distance to it shrinks.

**`exercises.ipynb`** (with `solutions.ipynb`) tests the same ideas in twelve exercises: the row and column pictures of $A\mathbf{x}$; matrix multiplication with three loops; rotation, shear and scaling matrices and the determinant of their composition; the rank, null space, left null space and solvability of a given $3\times4$ matrix; a $3\times 3$ system by hand and by `solve`; least squares via the normal equations checked against `lstsq`; the hat matrix as a symmetric idempotent projector (with test matrices showing that both properties are needed) and its leverages; detecting exactly collinear columns in a customer feature matrix; propagating a Markov chain and finding its stationary distribution with a linear solve; two conceptual questions (layer shapes and $AB$ versus $BA$; why `solve` beats `inv` and what collinearity does to coefficients); and a mini-project fitting a multi-feature linear regression to synthetic housing data with your own normal-equations solver, interpreting the coefficients and matching `sklearn.linear_model.LinearRegression`.

Next: [Module 03 — Eigenvalues, SVD and PCA](../03-eigenvalues-svd-and-pca/README.md), where the "same map, different coordinates" idea becomes the eigendecomposition and the SVD.
