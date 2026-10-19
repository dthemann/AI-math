#!/usr/bin/env python3
"""Builder for Module 02 — Matrices and Linear Maps.

Generates lesson.ipynb, exercises.ipynb and solutions.ipynb into
part-1-linear-algebra/02-matrices-and-linear-maps.  Run from the repo root:

    python3 _authoring/builders/build_02.py
    python3 tools/run_notebooks.py part-1-linear-algebra/02-matrices-and-linear-maps
"""
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import ExerciseSet, Notebook, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-1-linear-algebra/02-matrices-and-linear-maps"

# ============================================================================
# lesson.ipynb
# ============================================================================
nb = Notebook(None)

nb.md(r"""
# Module 02 — Matrices and Linear Maps

This notebook is the runnable companion to the concept guide. It shows a matrix acting as a
*function* on vectors, composes such functions with `@`, draws what rotations, shears and
projections do to the unit square, detects rank deficiency in a feature matrix, solves linear
systems the right way (and watches the wrong way fail on an ill-conditioned matrix), fits least
squares three different ways, diagonalises a map by changing basis, and runs a Markov chain to
its stationary distribution.

Read the [concept guide](README.md) first, or keep it open alongside: each section below mirrors
a numbered section there.
""")

nb.code(r"""
import time
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
""")

# ---------------------------------------------------------------- section 1
nb.md(r"""
## 1. Four ways to read $A\mathbf{x}$

The product of an $m \times n$ matrix with a vector in $\mathbb{R}^n$ is defined entry by entry,
$(A\mathbf{x})_i = \sum_j A_{ij} x_j$, but you should be able to *see* it in four ways:

* **row picture** — entry $i$ is the dot product of row $i$ with $\mathbf{x}$;
* **column picture** — $A\mathbf{x} = x_1 \mathbf{a}_1 + \dots + x_n \mathbf{a}_n$, a linear combination of the columns;
* **function** — $\mathbf{x} \mapsto A\mathbf{x}$ is a linear map $\mathbb{R}^n \to \mathbb{R}^m$, fully determined by where it sends the basis vectors (the columns);
* **system of equations** — $A\mathbf{x} = \mathbf{b}$ is $m$ equations in $n$ unknowns.

The cell below computes all three numeric versions for $A = \begin{pmatrix}1&2\\3&4\end{pmatrix}$,
$\mathbf{x} = (2, -1)^\top$, then applies the row picture to a small patient data matrix.
""")

nb.code(r"""
A = np.array([[1., 2.],
              [3., 4.]])
x = np.array([2., -1.])

row_picture = np.array([A[i] @ x for i in range(A.shape[0])])          # dot product per row
col_picture = sum(x[j] * A[:, j] for j in range(A.shape[1]))           # weighted sum of columns
print("row picture   :", row_picture)
print("column picture:", col_picture)
print("A @ x         :", A @ x)
print("images of the basis vectors e1, e2 are the columns:", A @ np.array([1., 0.]), A @ np.array([0., 1.]))

# A data matrix X (n patients x d features) times a weight vector = one score per patient
X = np.array([[6., 14., 30.],     # age (decades), systolic BP (x10 mmHg), BMI
              [4., 12., 22.],
              [7., 16., 28.],
              [5., 13., 35.]])
w = np.array([0.3, 0.5, 0.2])
scores = X @ w
print("\nrisk scores (row picture, one dot product per patient):", scores)
print("same scores as a combination of feature columns   :", 0.3 * X[:, 0] + 0.5 * X[:, 1] + 0.2 * X[:, 2])
""")

nb.md(r"""
Notice that the three computations agree to the last digit — they are the same sum, grouped
differently. The *column* grouping is the one to remember for the rest of this module: every
score vector `X @ w` you can ever produce is a combination of the three feature columns, so it
lives in a 3-dimensional subspace of $\mathbb{R}^4$. If the true risks do not lie in that
subspace (they never exactly do), no weight vector reproduces them and the best you can do is
the *closest* point in the subspace — that is least squares, Section 7.
""")

# ---------------------------------------------------------------- section 2
nb.md(r"""
## 2. Matrix multiplication as composition

$(AB)_{ij} = \sum_k A_{ik} B_{kj}$ is the dot product of row $i$ of $A$ with column $j$ of $B$,
and $(AB)\mathbf{x} = A(B\mathbf{x})$: apply $B$, then $A$. Three consequences you can test:

1. the product is **not commutative** ($AB \ne BA$ in general);
2. the cost of a square $n \times n$ product grows like $n^3$ — doubling $n$ costs about $8\times$;
3. two linear layers in a row collapse into one matrix: $W_2(W_1\mathbf{x}) = (W_2 W_1)\mathbf{x}$.

We first write the triple loop ourselves, then check it against NumPy's `@`.
""")

nb.code(r"""
def matmul_loops(A, B):
    m, p = A.shape
    p2, n = B.shape
    assert p == p2, "inner dimensions must match"
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i, j] += A[i, k] * B[k, j]
    return C

A = np.array([[1., 2.], [3., 4.]])
B = np.array([[0., 1.], [1., 0.]])          # a permutation (swap) matrix
print("A @ B =\n", A @ B, "\nB @ A =\n", B @ A)
print("loops agree with @ :", np.allclose(matmul_loops(A, B), A @ B))

# cubic cost: time the loop version for n, 2n, 4n
for n in (30, 60, 120):
    M = rng.standard_normal((n, n))
    t0 = time.perf_counter(); matmul_loops(M, M); dt = time.perf_counter() - t0
    print(f"n = {n:4d}: {n**3:>10,d} multiply-adds, loops took {dt*1000:7.1f} ms")

# a two-layer *linear* network is one matrix
W1 = rng.standard_normal((5, 3))      # 3 inputs -> 5 hidden
W2 = rng.standard_normal((2, 5))      # 5 hidden -> 2 outputs
x = rng.standard_normal(3)
print("\nW2 @ (W1 @ x) =", W2 @ (W1 @ x))
print("(W2 @ W1) @ x =", (W2 @ W1) @ x, "   combined shape:", (W2 @ W1).shape)

# batching: one call multiplies a whole stack of matrices
batch = rng.standard_normal((8, 4, 3))    # 8 matrices of shape 4x3
print("batch @ W  ->", (batch @ rng.standard_normal((3, 2))).shape)
""")

nb.md(r"""
Notice that $AB$ swaps the *columns* of $A$ while $BA$ swaps its *rows* — the same permutation
matrix does different things depending on which side it sits. The timings roughly multiply by 8
each time $n$ doubles, the signature of $n^3$ (NumPy's `@` does the same count of operations in
optimised, parallel BLAS code, which is why the same product on a GPU is thousands of times
faster). And `W2 @ W1` reproduces the two-step computation exactly: without a nonlinearity between
them, two layers are one layer with fewer degrees of freedom than their parameter count suggests.

**Try it yourself:** change the batch shape to `(8, 4, 3)` and multiply by `W1.T` (shape `(3, 5)`) —
predict the output shape before you run it. Then try `A * B` instead of `A @ B` and compare.
""")

# ---------------------------------------------------------------- section 3
nb.md(r"""
## 3. Geometry: what a $2\times2$ matrix does to the unit square

Apply a matrix to the corners of the unit square and you get a parallelogram whose edges are
the two columns. Its signed area is the determinant $\det\begin{pmatrix}a&b\\c&d\end{pmatrix} = ad - bc$:
$|\det|$ is the area-scaling factor, a negative sign means the orientation flipped, and
$\det = 0$ means the square was squashed onto a line — a dimension was lost.

$$
R_\theta = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix},\quad
S = \begin{pmatrix}s_x & 0\\ 0 & s_y\end{pmatrix},\quad
H_k = \begin{pmatrix}1 & k\\ 0 & 1\end{pmatrix},\quad
F = \begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix},\quad
P_{\mathbf u} = \mathbf{u}\mathbf{u}^\top .
$$
""")

nb.code(r"""
def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def scaling(sx, sy):
    return np.array([[sx, 0.], [0., sy]])

def shear(k):
    return np.array([[1., k], [0., 1.]])

def reflection_x():
    return np.array([[1., 0.], [0., -1.]])

def projection(u):
    u = np.asarray(u, dtype=float); u = u / np.linalg.norm(u)
    return np.outer(u, u)

def det2(M):
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

square = np.array([[0, 1, 1, 0, 0],      # x coordinates of the corners (closed polygon)
                   [0, 0, 1, 1, 0.]])    # y coordinates
grid = np.array([[gx, gy] for gx in np.linspace(0, 1, 6) for gy in np.linspace(0, 1, 6)]).T

maps = {"rotation 30°": rotation(np.pi / 6), "scaling (2, 0.5)": scaling(2, 0.5),
        "shear k=1": shear(1.0), "reflection in x-axis": reflection_x(),
        "projection onto (1,1)": projection([1, 1]), "rotation 30° then scaling": scaling(2, 0.5) @ rotation(np.pi / 6)}

fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
for ax, (name, M) in zip(axes.ravel(), maps.items()):
    img, g_img = M @ square, M @ grid
    ax.plot(square[0], square[1], "k--", lw=1, label="unit square")
    ax.fill(img[0], img[1], alpha=0.3, color="tab:blue")
    ax.plot(img[0], img[1], color="tab:blue", lw=2, label="image")
    ax.scatter(g_img[0], g_img[1], s=8, color="tab:red")
    ax.arrow(0, 0, *M[:, 0], head_width=0.08, color="tab:green", length_includes_head=True)
    ax.arrow(0, 0, *M[:, 1], head_width=0.08, color="tab:orange", length_includes_head=True)
    ax.set_title(f"{name}\ndet = {det2(M):+.3f}  (NumPy: {np.linalg.det(M):+.3f})", fontsize=9)
    ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.5, 2); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7, loc="upper left")
plt.suptitle("Linear maps acting on the unit square (green/orange arrows = the two columns)")
plt.tight_layout(); plt.show()

print("det(S @ R) = det(S) * det(R):", det2(scaling(2, 0.5) @ rotation(np.pi / 6)), "=", det2(scaling(2, 0.5)) * det2(rotation(np.pi / 6)))
""")

nb.md(r"""
Notice that the rotation and the shear both have determinant $+1$ — the slanted parallelogram has
exactly the same area as the square. The reflection has determinant $-1$: same area, but the
corners now run clockwise. The projection has determinant $0$ and the red grid points, which
were spread over the square, now all lie on one line: pairs of different inputs have become the
same output, and no matrix can undo that. Composition multiplies determinants.
""")

# ---------------------------------------------------------------- section 4
nb.md(r"""
## 4. Special matrices and what they promise

* **Orthogonal** $Q$: $Q^\top Q = I$, so $Q^{-1} = Q^\top$ and $\lVert Q\mathbf{x}\rVert = \lVert\mathbf{x}\rVert$ — lengths and angles survive.
* **Symmetric PSD**: $A = A^\top$ and $\mathbf{x}^\top A\mathbf{x} \ge 0$; every $A^\top A$ is one, because $\mathbf{x}^\top A^\top A\mathbf{x} = \lVert A\mathbf{x}\rVert^2$.
* **Triangular**: solve by back-substitution in $O(n^2)$ instead of $O(n^3)$.
* **Transpose rule**: $(AB)^\top = B^\top A^\top$ — the order flips.
""")

nb.code(r"""
Q = rotation(np.pi / 4)                                   # rotations are orthogonal
v = rng.standard_normal(2)
print("Q^T Q =\n", Q.T @ Q)
print("||v|| = %.4f   ||Qv|| = %.4f" % (np.linalg.norm(v), np.linalg.norm(Q @ v)))

M = rng.standard_normal((5, 3))
G = M.T @ M                                               # Gram matrix: symmetric and PSD
print("\nG symmetric:", np.allclose(G, G.T), "   eigenvalues of G:", np.linalg.eigvalsh(G))
for _ in range(3):
    z = rng.standard_normal(3)
    print("  z^T G z = %.3f  (= ||Mz||^2 = %.3f)" % (z @ G @ z, np.sum((M @ z) ** 2)))

K = np.array([[1., 2.], [2., 1.]])                        # symmetric but NOT PSD
z = np.array([1., -1.])
print("\nz^T K z for K=[[1,2],[2,1]], z=(1,-1):", z @ K @ z, "  eigenvalues:", np.linalg.eigvalsh(K))

A, B = rng.standard_normal((3, 2)), rng.standard_normal((2, 4))
print("\n(AB)^T == B^T A^T :", np.allclose((A @ B).T, B.T @ A.T))

# back-substitution on an upper-triangular system
U = np.array([[2., 1., -1.], [0., 3., 2.], [0., 0., 4.]])
b = np.array([3., 11., 8.])
x = np.zeros(3)
for i in range(2, -1, -1):
    x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
print("back-substitution x =", x, "  check U @ x =", U @ x)
""")

nb.md(r"""
Notice that $\mathbf{z}^\top G\mathbf{z}$ equals $\lVert M\mathbf{z}\rVert^2$ for every draw — that is the whole
proof that Gram matrices are positive semidefinite, and it is why covariance matrices (which are
Gram matrices of centred data) always have non-negative eigenvalues. The matrix `K` looks harmless
but gives a negative quadratic form, so it cannot be a covariance matrix. The back-substitution
loop solved a $3\times3$ system with only $3 + 2 + 1 = 6$ multiplications: the triangular structure
is what `np.linalg.solve` buys itself by first factoring $A = LU$.

**Try it yourself:** replace `rotation(np.pi/4)` with `shear(1.0)` and re-run. Is $Q^\top Q$ still the
identity? Which norms change?
""")

# ---------------------------------------------------------------- section 5
nb.md(r"""
## 5. Rank, column space and null space — collinear features

$\operatorname{rank} A$ is the number of independent columns; $\dim\mathcal{N}(A) = n - \operatorname{rank} A$
(rank–nullity). In a data matrix a rank deficiency means some feature is an exact linear
combination of others, so the regression coefficients are not identifiable. Below, a synthetic
feature matrix with 1,000 rows contains an exact duplicate (a unit conversion) and a one-hot
group that sums to the intercept — the *dummy-variable trap*.
""")

nb.code(r"""
n = 1000
age = rng.uniform(20, 80, n)
weight_kg = rng.normal(75, 12, n)
height_cm = rng.normal(172, 9, n)
weight_lb = 2.2046 * weight_kg                                   # exact duplicate information
device = rng.integers(0, 3, n)                                   # 0 = mobile, 1 = desktop, 2 = tablet
one_hot = np.eye(3)[device]                                      # three 0/1 columns
intercept = np.ones(n)

X_good = np.column_stack([intercept, age, weight_kg, height_cm, one_hot[:, 1:]])   # drop one level
X_bad = np.column_stack([intercept, age, weight_kg, height_cm, weight_lb, one_hot])  # both traps

for name, X in [("X_good", X_good), ("X_bad", X_bad)]:
    r = np.linalg.matrix_rank(X)
    print(f"{name}: shape {X.shape}, rank {r}, null-space dimension {X.shape[1] - r}")

# null space of X_bad from the SVD: right-singular vectors with (numerically) zero singular value
_, s, Vt = np.linalg.svd(X_bad, full_matrices=False)
print("\nsingular values of X_bad:", s)
null_basis = Vt[s < 1e-8 * s.max()]

def tidy_basis(N, tol=1e-9):
    # row-reduce a basis of the null space so that each vector spells out ONE dependency
    N = N.copy(); r = 0
    for c in range(N.shape[1]):
        if r == N.shape[0]:
            break
        piv = r + np.argmax(np.abs(N[r:, c]))
        if abs(N[piv, c]) < tol:
            continue
        N[[r, piv]] = N[[piv, r]]; N[r] /= N[r, c]
        for i in range(N.shape[0]):
            if i != r:
                N[i] -= N[i, c] * N[r]
        r += 1
    N[np.abs(N) < 1e-6] = 0.0
    return N

cols = ["intercept", "age", "weight_kg", "height_cm", "weight_lb", "mobile", "desktop", "tablet"]
for z in tidy_basis(null_basis):
    relation = " ".join(f"{v:+.4f}*{c}" for c, v in zip(cols, z) if v != 0)
    print("null vector:", relation, "= 0   -> ||X_bad z|| = %.1e" % np.linalg.norm(X_bad @ z))
""")

nb.md(r"""
Notice that `X_bad` has 8 columns but rank 6: the null space is two-dimensional, and the two
null vectors spell out exactly which columns are redundant — one says
`weight_kg - 0.4536 * weight_lb = 0` (i.e. $1/2.2046 = 0.4536$), the other says
`intercept - mobile - desktop - tablet = 0`. Any multiple of these vectors can be
added to a coefficient vector without changing a single prediction, so the "effect of weight" or
"effect of being on mobile" is meaningless until one column from each group is dropped (or a ridge
penalty is added). `X_good` does that and has full column rank.
""")

# ---------------------------------------------------------------- section 6
nb.md(r"""
## 6. Solving $A\mathbf{x} = \mathbf{b}$: `solve`, not `inv`

For an invertible square $A$ the solution is $\mathbf{x} = A^{-1}\mathbf{b}$ *on paper*. In code,
`np.linalg.solve` factors $A = LU$ and does two triangular solves — about $\tfrac23 n^3$ flops versus
roughly $2n^3$ for the explicit inverse — and commits less rounding error. How much rounding error
you can afford is governed by the **condition number** $\kappa(A)$: relative perturbations of the
data can be amplified by up to $\kappa(A)$ in the solution.

We solve the juice-blending system from the guide, time `solve` against `inv`, and then push a
$2\times2$ matrix towards singularity to watch the error grow with $\kappa$.
""")

nb.code(r"""
A = np.array([[1., 1., 2.],
              [2., 1., 1.],
              [1., 3., 1.]])
b = np.array([7., 8., 12.])
x = np.linalg.solve(A, b)
print("blend (dl of each concentrate):", x, "  residual:", np.linalg.norm(A @ x - b))

# cost: solve vs inverse on a 600x600 system
M = rng.standard_normal((600, 600)); rhs = rng.standard_normal(600)
t0 = time.perf_counter(); [np.linalg.solve(M, rhs) for _ in range(5)]; t_solve = (time.perf_counter() - t0) / 5
t0 = time.perf_counter(); [np.linalg.inv(M) @ rhs for _ in range(5)]; t_inv = (time.perf_counter() - t0) / 5
print(f"600x600: solve {t_solve*1000:.1f} ms   inv(A) @ b {t_inv*1000:.1f} ms   ratio {t_inv / t_solve:.1f}x")

# sensitivity: a family of increasingly ill-conditioned 2x2 matrices.
# We perturb b by a *relative* 1e-8 (think: measurement noise in the 8th digit) and watch
# how far the solution moves; the condition number predicts the amplification.
x_true = np.array([0.3, 0.7])
delta = 1e-8
eps_values = 10.0 ** -np.arange(1, 13)
kappas, err_pert, err_round = [], [], []
for eps in eps_values:
    A2 = np.array([[1., 1.], [1., 1. + eps]])
    b2 = A2 @ x_true
    b_noisy = b2 * (1 + delta * np.array([1., -1.]))          # relative perturbation of size delta
    kappas.append(np.linalg.cond(A2))
    err_pert.append(np.linalg.norm(np.linalg.solve(A2, b_noisy) - x_true) / np.linalg.norm(x_true))
    err_round.append(np.linalg.norm(np.linalg.solve(A2, b2) - x_true) / np.linalg.norm(x_true))
kappas = np.array(kappas)

plt.loglog(kappas, err_pert, "o-", label=r"relative error, $b$ perturbed by $10^{-8}$")
plt.loglog(kappas, kappas * delta, "k:", label=r"bound $\kappa(A)\cdot 10^{-8}$")
plt.loglog(kappas, np.maximum(err_round, 1e-17), "s--", alpha=0.6, label=r"relative error, exact $b$ (rounding only)")
plt.loglog(kappas, kappas * 2.2e-16, "k-.", alpha=0.6, label=r"bound $\kappa(A)\cdot\epsilon_{machine}$")
plt.xlabel(r"condition number $\kappa(A)$"); plt.ylabel("relative error in the solution")
plt.title("Perturbations of b are amplified by up to the condition number")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()
print("kappa * 1e-8 exceeds 1 (no correct digits left) once kappa > 1e8; here kappa reaches %.0e" % kappas[-1])

print("eps = 1e-4: kappa = %.1e" % kappas[3])
A4 = np.array([[1., 1.], [1., 1.0001]])
print("b = (2, 2.0001) -> x =", np.linalg.solve(A4, [2., 2.0001]))
print("b = (2, 2.0002) -> x =", np.linalg.solve(A4, [2., 2.0002]), "  (a 5e-5 relative change in b)")
""")

nb.md(r"""
Notice how the circles hug the dotted line $\kappa(A)\cdot 10^{-8}$: a perturbation in the eighth
digit of $\mathbf{b}$ is amplified by almost exactly the condition number, and once
$\kappa > 10^{8}$ the relative error in $\mathbf{x}$ exceeds 1 — not a single digit of the answer is
reliable. The rounding-only curve (exact $\mathbf{b}$) sits far lower but follows the same slope with
$\epsilon_{\text{machine}} \approx 2\times10^{-16}$ in place of $10^{-8}$: with $\kappa = 10^{10}$ you keep about
six digits, with $10^{16}$ none. Conditioning is a property of the *problem*, not the algorithm; `inv`
cannot fix it and costs roughly three times as much. The last two lines are the guide's example: a
change in the fifth decimal of $\mathbf{b}$ moves the solution from $(1, 1)$ to $(0, 2)$, because the
two columns of $A$ are almost parallel.

**Try it yourself:** change `delta` to `1e-4` (noise in the fourth digit) and read off the largest
$\kappa$ for which the solution still has two correct digits. Then try `np.linalg.solve` on
`X_bad.T @ X_bad` from Section 5 and read the error message.
""")

# ---------------------------------------------------------------- section 7
nb.md(r"""
## 7. Least squares three ways

With more observations than parameters, $A\mathbf{x} = \mathbf{y}$ has no exact solution. The
least-squares solution minimises $\lVert A\mathbf{x} - \mathbf{y}\rVert^2$; the residual must be orthogonal to
every column, $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}$, giving the **normal equations**

$$
A^\top A\,\hat{\mathbf{x}} = A^\top\mathbf{y}, \qquad \hat{\mathbf{y}} = H\mathbf{y}, \quad H = A(A^\top A)^{-1}A^\top .
$$

We fit a line to noisy data by (a) the normal equations from scratch, (b) a QR factorisation
$A = QR$, so $R\hat{\mathbf{x}} = Q^\top\mathbf{y}$, and (c) `np.linalg.lstsq`, then check that the hat
matrix is a symmetric projection.
""")

nb.code(r"""
t = np.linspace(0, 10, 30)
y = 2.0 + 0.5 * t + rng.normal(0, 0.8, t.size)          # true line: intercept 2, slope 0.5
A = np.column_stack([np.ones_like(t), t])              # design matrix (30 x 2)

# (a) normal equations, solved (not inverted)
x_normal = np.linalg.solve(A.T @ A, A.T @ y)
# (b) QR: A = Q R  ->  R x = Q^T y  (triangular, well conditioned)
Qm, Rm = np.linalg.qr(A)
x_qr = np.linalg.solve(Rm, Qm.T @ y)
# (c) library call (SVD-based)
x_lstsq, residual_ss, rank, sv = np.linalg.lstsq(A, y, rcond=None)
print("normal eq :", x_normal, "\nQR        :", x_qr, "\nlstsq     :", x_lstsq)
print("rank of A :", rank, "   kappa(A) = %.1f   kappa(A^T A) = %.1f" % (np.linalg.cond(A), np.linalg.cond(A.T @ A)))

# the hat matrix is a symmetric, idempotent projector with trace = number of parameters
H = A @ np.linalg.solve(A.T @ A, A.T)
r = y - H @ y
print("\nH symmetric:", np.allclose(H, H.T), "  H @ H == H:", np.allclose(H @ H, H), "  trace(H) =", np.trace(H).round(6))
print("residual orthogonal to the columns: A^T r =", A.T @ r)
print("leverages (diag of H) range: %.3f .. %.3f, mean %.3f = p/n" % (np.diag(H).min(), np.diag(H).max(), np.diag(H).mean()))

plt.scatter(t, y, s=18, label="data")
plt.plot(t, A @ x_lstsq, "r", label=f"fit: {x_lstsq[0]:.2f} + {x_lstsq[1]:.2f} t")
plt.plot(t, 2.0 + 0.5 * t, "k--", lw=1, label="true line")
plt.vlines(t, A @ x_lstsq, y, color="gray", alpha=0.5, lw=1)
plt.xlabel("t"); plt.ylabel("y"); plt.title("Least-squares line: residuals are the grey segments")
plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

nb.md(r"""
Notice that all three routes give the same coefficients — they solve the same problem — but
$\kappa(A^\top A) = \kappa(A)^2$: forming the Gram matrix squares the condition number, which is
why `lstsq` and QR are preferred for anything wider than a handful of columns. The hat matrix
passes both projection tests, its trace is exactly 2 (the number of parameters), and
$A^\top\mathbf{r} = \mathbf{0}$ to rounding error: the residual vector really is perpendicular to the
column space. "Linear" least squares means linear *in the coefficients*: the same machinery fits a
quadratic once we add a $t^2$ column.
""")

nb.code(r"""
y2 = 1.0 + 0.3 * t - 0.05 * t ** 2 + rng.normal(0, 0.3, t.size)      # curved truth
designs = {"line": np.column_stack([np.ones_like(t), t]),
           "quadratic": np.column_stack([np.ones_like(t), t, t ** 2]),
           "degree-6 polynomial": np.column_stack([t ** k for k in range(7)])}
tt = np.linspace(0, 10, 200)
plt.scatter(t, y2, s=18, color="k", label="data")
for name, A_d in designs.items():
    coef = np.linalg.lstsq(A_d, y2, rcond=None)[0]
    A_tt = np.column_stack([tt ** k for k in range(A_d.shape[1])])
    rss = np.sum((y2 - A_d @ coef) ** 2)
    print(f"{name:20s}: {A_d.shape[1]} parameters, RSS = {rss:.3f}, kappa(A) = {np.linalg.cond(A_d):.1e}")
    plt.plot(tt, A_tt @ coef, label=f"{name} (RSS {rss:.2f})")
plt.ylim(y2.min() - 1, y2.max() + 1)
plt.xlabel("t"); plt.ylabel("y"); plt.title("Polynomial fits are still linear least squares")
plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

nb.md(r"""
Notice that the residual sum of squares can only fall as columns are added — the column space
grows, so the projection gets closer — but the degree-6 fit buys very little over the quadratic
while its design matrix has a condition number millions of times larger. More columns means a
better fit to *these* points, not a better model; Module 13 turns that tension into the
bias–variance trade-off, and Module 03's SVD explains the exploding condition number.

**Try it yourself:** increase the polynomial degree to 12 and look at both `kappa(A)` and the
curve between the data points.
""")

# ---------------------------------------------------------------- section 8
nb.md(r"""
## 8. Change of basis: the same map in better coordinates

If the columns of $P$ form a basis, a vector with standard coordinates $\mathbf{x}$ has new
coordinates $\mathbf{c} = P^{-1}\mathbf{x}$ (so that $\mathbf{x} = P\mathbf{c}$), and a map with matrix $A$
becomes $A' = P^{-1}AP$ in the new coordinates. For $A = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ and the
basis $\mathbf{p}_1 = (1,1)^\top$, $\mathbf{p}_2 = (1,-1)^\top$ the new matrix is diagonal: $A$ just
stretches by 3 along $\mathbf{p}_1$ and by 1 along $\mathbf{p}_2$.
""")

nb.code(r"""
A = np.array([[2., 1.], [1., 2.]])
P = np.array([[1., 1.], [1., -1.]])                  # columns = new basis vectors
x = np.array([3., 1.])
c = np.linalg.solve(P, x)                            # coordinates of x in the new basis
print("x =", x, " new coordinates c =", c, " reconstruct P @ c =", P @ c)
A_new = np.linalg.solve(P, A @ P)                    # P^{-1} A P without forming the inverse
print("P^-1 A P =\n", A_new)
print("invariants  trace: %.1f vs %.1f   det: %.1f vs %.1f" % (np.trace(A), np.trace(A_new), np.linalg.det(A), np.linalg.det(A_new)))

# picture: the unit circle and its image, with the new basis directions
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])
ellipse = A @ circle
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(circle[0], circle[1], "k--", lw=1, label="unit circle")
ax.plot(ellipse[0], ellipse[1], "tab:blue", lw=2, label="image under A")
for i, (col, lab) in enumerate(zip(["tab:green", "tab:orange"], ["p1 direction (x3)", "p2 direction (x1)"])):
    p = P[:, i] / np.linalg.norm(P[:, i])
    ax.arrow(0, 0, *p, head_width=0.08, color=col, length_includes_head=True)
    ax.arrow(0, 0, *(A @ p), head_width=0.08, color=col, alpha=0.4, length_includes_head=True, label=lab)
ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("A stretches along p1 by 3 and along p2 by 1")
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.show()
""")

nb.md(r"""
Notice that the circle becomes an ellipse whose axes are exactly the new basis directions: along
$\mathbf{p}_1$ the arrow triples, along $\mathbf{p}_2$ it is unchanged, and $P^{-1}AP$ records nothing but
those two numbers. Trace and determinant are the same in both coordinate systems — they belong
to the map, not to the matrix. Finding the basis that makes $A$ diagonal is the eigenvalue
problem, and doing it for a covariance matrix is PCA (Module 03).
""")

# ---------------------------------------------------------------- section 9
nb.md(r"""
## 9. Matrices as graphs and Markov chains

An adjacency matrix $A$ has $A_{ij} = 1$ when there is an edge $i \to j$, and $(A^k)_{ij}$ counts
walks of length $k$. A row-stochastic transition matrix $P$ propagates a distribution:
$\boldsymbol\pi_{t+1} = \boldsymbol\pi_t P$. For a well-behaved chain $\boldsymbol\pi_t$ converges to the
stationary distribution $\boldsymbol\pi$ with $\boldsymbol\pi P = \boldsymbol\pi$, whatever the start.
""")

nb.code(r"""
# path counting on a small undirected graph: 0-1, 1-2, 2-3, 3-0, 0-2
Adj = np.array([[0, 1, 1, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [1, 0, 1, 0]])
print("walks of length 2 (A^2):\n", Adj @ Adj)
print("walks of length 3 from node 0 to node 3:", np.linalg.matrix_power(Adj, 3)[0, 3])

# a three-state customer chain: Active, Lapsed, Churned-but-returning
Pm = np.array([[0.80, 0.15, 0.05],
               [0.30, 0.50, 0.20],
               [0.10, 0.20, 0.70]])
print("\nrows sum to 1:", Pm.sum(axis=1))
pi = np.array([1., 0., 0.])                           # everyone starts Active
history = [pi]
for step in range(25):
    pi = pi @ Pm
    history.append(pi)
history = np.array(history)

# stationary distribution two ways: a large power, and the eigenvector of P^T for eigenvalue 1
pi_power = np.linalg.matrix_power(Pm, 200)[0]
evals, evecs = np.linalg.eig(Pm.T)
v = np.real(evecs[:, np.argmin(np.abs(evals - 1))]); pi_eig = v / v.sum()
print("pi after 25 steps :", history[-1])
print("row of P^200      :", pi_power)
print("eigenvector method:", pi_eig, "   check pi P = pi:", np.allclose(pi_eig @ Pm, pi_eig))
print("eigenvalues of P  :", np.sort(np.abs(evals))[::-1], " (second one sets the convergence rate)")

for j, lab in enumerate(["Active", "Lapsed", "Churned"]):
    plt.plot(history[:, j], "o-", ms=3, label=lab)
    plt.axhline(pi_eig[j], color="gray", ls=":", lw=1)
plt.xlabel("month"); plt.ylabel("probability"); plt.title("Distribution converges to the stationary distribution (dotted)")
plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

nb.md(r"""
Notice that after about 15 steps the three curves have flattened onto the dotted stationary
values, and that the $200$-th power of $P$ has that same vector as its row — every starting state
leads to the same long-run mix. The gap closes geometrically at the rate of the second-largest
eigenvalue modulus, which is the quantity PageRank and MCMC practitioners care about (Modules 03
and 18). The $A^2$ entries are plain path counts: matrix multiplication is doing combinatorics.
""")

# ---------------------------------------------------------------- summary
nb.md(r"""
## Summary

* $A\mathbf{x}$ is simultaneously a set of dot products (rows), a combination of columns, a linear
  function and a system of equations; the column picture tells you what outputs are reachable.
* `A @ B` composes maps right-to-left, is not commutative, costs $O(n^3)$, and collapses stacked
  linear layers into one.
* Rotation, scaling, shear, reflection and projection are $2\times2$ matrices; the determinant is the
  signed area factor, and $\det = 0$ means a dimension was destroyed.
* Orthogonal matrices preserve lengths; Gram matrices $A^\top A$ are symmetric PSD; triangular
  systems solve in $O(n^2)$.
* Rank counts independent columns; duplicated or one-hot-trapped features give a non-trivial null
  space and unidentifiable coefficients.
* Use `np.linalg.solve`, never `inv(A) @ b`; the condition number bounds how many digits survive.
* Least squares = projection onto the column space; the normal equations, QR and `lstsq` agree,
  but QR/SVD avoid squaring $\kappa$; the hat matrix is a symmetric idempotent projector.
* $P^{-1}AP$ is the same map in new coordinates — diagonal when $P$ holds the eigenvectors.
* Powers of adjacency matrices count walks; powers of transition matrices converge to the
  stationary distribution.
""")

nb.md(r"""
**Next:** open `exercises.ipynb` and work through the twelve exercises; `solutions.ipynb` has
reference implementations and explanations. Then continue with
[Module 03 — Eigenvalues, SVD and PCA](../03-eigenvalues-svd-and-pca/README.md).
""")

nb.save(MOD / "lesson.ipynb")
print("wrote lesson.ipynb with", len(nb.cells), "cells")

# ============================================================================
# exercises.ipynb / solutions.ipynb
# ============================================================================


def run_solution(src: str) -> dict:
    """Execute a solution snippet in a fresh namespace (so hashes come from the real code)."""
    ns = {"np": np}
    exec(textwrap.dedent(src), ns)
    return ns


def dd(*parts: str) -> str:
    """Dedent each part separately, then join — so column-0 and indented snippets can be mixed."""
    return "\n".join(textwrap.dedent(p).strip("\n") for p in parts) + "\n"


def safe_sha(value, decimals=3):
    """sha_of, but refuse values whose rounding at `decimals` is ambiguous."""
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    scaled = arr * 10 ** decimals
    frac = np.abs(scaled - np.floor(scaled) - 0.5)
    assert np.all(frac > 0.05), f"value {value} too close to a rounding boundary for hashing"
    return sha_of(value, decimals)


ex = ExerciseSet(
    "Module 02 — Exercises",
    intro_md=r"""
    Twelve exercises on matrices as linear maps: multiplication from scratch, geometric
    transformations and determinants, rank and null spaces, solving systems the right way,
    least squares and the hat matrix, collinearity detection, Markov chains, Gaussian elimination,
    and a mini-project fitting a multi-feature regression with your own solver. Only NumPy (plus
    matplotlib and scikit-learn for the project) is needed. Work through them in order — later
    exercises reuse ideas from earlier ones.
    """,
    setup_code=r"""
    import numpy as np
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    np.set_printoptions(precision=4, suppress=True)
    """,
)

# ---------------------------------------------------------------- 1: pictures
SOL_1 = r"""
def matvec_rows(A, x):
    # entry i is the dot product of row i with x
    return np.array([A[i] @ x for i in range(A.shape[0])])

def matvec_cols(A, x):
    # weighted sum of the columns, weights = entries of x
    out = np.zeros(A.shape[0])
    for j in range(A.shape[1]):
        out = out + x[j] * A[:, j]
    return out
"""
ex.add(
    title="The row picture and the column picture", difficulty=1,
    prompt_md=r"""
    Implement the matrix–vector product twice, once per picture:

    * `matvec_rows(A, x)` — entry $i$ of the result is the dot product of **row** $i$ of `A` with `x`
      (you may use `@` between two 1-D vectors, but not `A @ x`);
    * `matvec_cols(A, x)` — the result is the **linear combination of the columns** of `A`,
      $\sum_j x_j\,\mathbf{a}_j$ (use scalar-times-vector and vector addition only).

    Both take a 2-D array `A` of shape `(m, n)` and a 1-D array `x` of length `n`, and return a 1-D
    array of length `m`. They must agree with `A @ x` for rectangular `A` as well as square `A`.
    """,
    starter=r"""
    def matvec_rows(A, x):
        # YOUR CODE HERE
        return None

    def matvec_cols(A, x):
        # YOUR CODE HERE
        return None
    """,
    solution=SOL_1,
    check=r"""
    A = np.array([[1., 2.], [3., 4.]]); x = np.array([2., -1.])
    B = np.array([[1., 0., 2.], [0., 1., -1.], [3., 3., 3.], [1., 1., 1.]]); z = np.array([1., 2., 3.])
    try:
        got_r, got_c = matvec_rows(A, x), matvec_cols(A, x)
    except Exception:
        got_r = got_c = None
    check_close('row picture, 2x2 example gives (0, 2)', got_r, A @ x)
    check_close('column picture, 2x2 example gives (0, 2)', got_c, A @ x)
    try:
        got_r, got_c = matvec_rows(B, z), matvec_cols(B, z)
    except Exception:
        got_r = got_c = None
    check_close('row picture, 4x3 matrix', got_r, B @ z)
    check_close('column picture, 4x3 matrix', got_c, B @ z)
    check('result has shape (4,) for a 4x3 matrix', got_r is not None and np.shape(got_r) == (4,))
    """,
    explanation_md=r"""
    Both functions compute the same sums $\sum_j A_{ij}x_j$, grouped differently. Row-wise you fix
    $i$ and sum over $j$ — one dot product per output entry, which is how a neuron computes its
    pre-activation. Column-wise you fix $j$ and accumulate $x_j$ times the whole column — which makes
    it obvious that every possible output is a combination of the columns, i.e. lies in the column
    space. The common mistake is to loop over the wrong axis (`A[:, i] @ x` instead of `A[i] @ x`),
    which only "works" for square symmetric matrices and fails the rectangular test.
    """,
)

# ---------------------------------------------------------------- 2: matmul
SOL_2 = r"""
def matmul_loops(A, B):
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("inner dimensions must match")
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C
"""
ex.add(
    title="Matrix multiplication with three loops", difficulty=1,
    prompt_md=r"""
    Implement `matmul_loops(A, B)` computing $C_{ij} = \sum_k A_{ik}B_{kj}$ with explicit Python
    loops over $i$, $j$ and $k$ — no `@`, `np.dot`, `np.matmul`, `np.einsum` or `np.sum`.
    `A` has shape `(m, p)`, `B` has shape `(p, n)`, the result has shape `(m, n)`; raise a
    `ValueError` when the inner dimensions disagree. Your implementation should reproduce `A @ B`
    to floating-point precision for rectangular inputs, and you will use it to confirm that
    $AB \neq BA$ for the swap matrix $B = \begin{pmatrix}0&1\\1&0\end{pmatrix}$.
    """,
    starter=r"""
    def matmul_loops(A, B):
        m, p = A.shape
        p2, n = B.shape
        # YOUR CODE HERE
        return None
    """,
    solution=SOL_2,
    check=r"""
    A = np.array([[1., 2.], [3., 4.]]); S = np.array([[0., 1.], [1., 0.]])
    try:
        AS, SA = matmul_loops(A, S), matmul_loops(S, A)
    except Exception:
        AS = SA = None
    check_close('A @ S swaps the columns of A', AS, A @ S)
    check_close('S @ A swaps the rows of A', SA, S @ A)
    check('A S and S A differ (non-commutative)', AS is not None and SA is not None and not np.allclose(AS, SA))
    M1 = rng.standard_normal((3, 4)); M2 = rng.standard_normal((4, 2))
    try:
        got = matmul_loops(M1, M2)
    except Exception:
        got = None
    check_close('rectangular (3x4)(4x2) matches @', got, M1 @ M2)
    try:
        matmul_loops(M1, M1)   # inner dims 4 and 3 do not match
        raised = False
    except ValueError:
        raised = True
    except Exception:
        raised = False
    check('mismatched inner dimensions raise ValueError', raised)
    """,
    explanation_md=r"""
    The triple loop is the definition made literal: $m n$ output entries, each a sum of $p$ products,
    so $mnp$ multiply–adds — $n^3$ for square matrices. NumPy's `@` performs the same arithmetic in
    compiled BLAS code with cache blocking and SIMD, which is why it is hundreds of times faster but
    not asymptotically cheaper. The swap matrix shows that order matters: multiplying on the right
    rearranges columns, on the left rearranges rows. A typical slip is indexing `B[j, k]` instead of
    `B[k, j]`, which computes $AB^\top$ — the rectangular test catches it.
    """,
)

# ---------------------------------------------------------------- 3: geometry
SOL_3 = r"""
def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def shear(k):
    return np.array([[1., k], [0., 1.]])

def scaling(sx, sy):
    return np.array([[sx, 0.], [0., sy]])

def det2(M):
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
"""
_ns3 = run_solution(SOL_3)
_composite = _ns3["scaling"](2., 3.) @ _ns3["shear"](1.) @ _ns3["rotation"](np.pi / 2)
HASH_3_DET = safe_sha(float(_ns3["det2"](_composite)))          # 6.0
ex.add(
    title="Rotation, shear, scaling — and their determinants", difficulty=1,
    prompt_md=r"""
    Write four small functions returning $2\times2$ NumPy arrays / floats:

    * `rotation(theta)` — counter-clockwise rotation by `theta` radians;
    * `shear(k)` — horizontal shear $\begin{pmatrix}1&k\\0&1\end{pmatrix}$;
    * `scaling(sx, sy)` — $\mathrm{diag}(s_x, s_y)$;
    * `det2(M)` — the determinant $ad - bc$ of a $2\times2$ matrix, without `np.linalg.det`.

    Then predict, before running the check: what is the determinant of
    `scaling(2, 3) @ shear(1) @ rotation(np.pi/2)`? (Think "product of determinants", not
    "multiply the matrices".)
    """,
    starter=r"""
    def rotation(theta):
        # YOUR CODE HERE
        return None

    def shear(k):
        # YOUR CODE HERE
        return None

    def scaling(sx, sy):
        # YOUR CODE HERE
        return None

    def det2(M):
        # YOUR CODE HERE
        return None
    """,
    solution=SOL_3,
    check=rf"""
    def _try(f):
        try:
            return f()
        except Exception:
            return None
    check_close('rotation by 90° sends e1 to e2', _try(lambda: rotation(np.pi / 2) @ np.array([1., 0.])), np.array([0., 1.]))
    check_close('rotation(a) @ rotation(b) == rotation(a + b)', _try(lambda: rotation(0.3) @ rotation(1.1)), _try(lambda: rotation(1.4)) if _try(lambda: rotation(1.4)) is not None else np.full((2, 2), np.nan))
    check_close('rotations are orthogonal: R^T R = I', _try(lambda: rotation(0.7).T @ rotation(0.7)), np.eye(2))
    check_close('shear(3) maps (0,1) to (3,1)', _try(lambda: shear(3.) @ np.array([0., 1.])), np.array([3., 1.]))
    check_close('det of a shear is 1 (area preserved)', _try(lambda: det2(shear(3.))), 1.0)
    check_close('det2 agrees with np.linalg.det', _try(lambda: det2(np.array([[2., 5.], [1., 4.]]))), np.linalg.det(np.array([[2., 5.], [1., 4.]])))
    check_close('scaling(2, 3) is diag(2, 3)', _try(lambda: scaling(2., 3.)), np.diag([2., 3.]))
    d = _try(lambda: float(det2(scaling(2., 3.) @ shear(1.) @ rotation(np.pi / 2))))
    check_hash('hidden: det of scaling(2,3) @ shear(1) @ rotation(90°)', d, '{HASH_3_DET}')
    """,
    explanation_md=r"""
    The columns of each matrix are the images of $\mathbf{e}_1$ and $\mathbf{e}_2$: a rotation sends
    them to $(\cos\theta, \sin\theta)$ and $(-\sin\theta, \cos\theta)$; a shear leaves $\mathbf{e}_1$
    alone and tilts $\mathbf{e}_2$ to $(k, 1)$. Determinants multiply under composition, so the hidden
    answer is $\det S \cdot \det H \cdot \det R = 6 \cdot 1 \cdot 1 = 6$ regardless of the order of the
    factors — even though the matrices themselves depend on the order. A frequent sign error is to
    put $+\sin\theta$ in the top-right entry, which produces a *clockwise* rotation and fails the
    $\mathbf{e}_1 \mapsto \mathbf{e}_2$ test.
    """,
)

# ---------------------------------------------------------------- 4: conceptual warm-up
ex.add_conceptual(
    title="Shapes, order and the collapse of linear layers", difficulty=1,
    prompt_md=r"""
    A network has two linear layers with weight matrices $W_1 \in \mathbb{R}^{4\times3}$ and
    $W_2 \in \mathbb{R}^{5\times4}$ and no activation function between them. A batch of $n$ inputs is
    stored as the rows of $X \in \mathbb{R}^{n\times3}$.

    1. Which of $W_1W_2$ and $W_2W_1$ is defined, and what is its shape?
    2. Write the batch output as a single matrix expression in $X$, $W_1$, $W_2$ (watch the
       transposes), and give its shape.
    3. Why does the absence of an activation make the second layer pointless? How many free
       parameters does the two-layer network have, and how many does the equivalent single
       layer have?
    4. Give a concrete $2\times2$ example where $AB \ne BA$, and one (with $A, B \ne I$ and
       $A \ne B$) where $AB = BA$.
    """,
    answer_md=r"""
    1. Only $W_2W_1$ is defined: $(5\times4)(4\times3) = 5\times3$. $W_1W_2$ would need the inner
       dimensions $3$ and $5$ to agree.
    2. Per sample $\mathbf{h} = W_2 W_1 \mathbf{x}$; with samples as rows, $H = X W_1^\top W_2^\top = X (W_2 W_1)^\top$,
       shape $(n, 5)$. The transposes appear because $(W_2W_1\mathbf{x})^\top = \mathbf{x}^\top W_1^\top W_2^\top$.
    3. The composition of two linear maps is a linear map, so the network computes $\mathbf{x} \mapsto M\mathbf{x}$
       with $M = W_2W_1 \in \mathbb{R}^{5\times3}$ — nothing a single layer could not do. The two-layer version has
       $5\cdot4 + 4\cdot3 = 32$ parameters (plus biases) but only $15$ effective ones, and if the hidden width
       were smaller than 3 it would even *restrict* the maps to rank $\le$ hidden width. A nonlinearity
       between the layers breaks the collapse — that is its entire job.
    4. $A = \begin{pmatrix}1&2\\3&4\end{pmatrix}$, $B = \begin{pmatrix}0&1\\1&0\end{pmatrix}$:
       $AB$ swaps the columns of $A$, $BA$ swaps its rows, and they differ. Commuting examples: any
       two diagonal matrices, e.g. $A = \mathrm{diag}(2, 3)$ and $B = \mathrm{diag}(5, 7)$; or a matrix with
       its own powers or inverse, e.g. $A$ and $A^2$; or two rotations in the plane, $R_\alpha R_\beta = R_\beta R_\alpha = R_{\alpha+\beta}$.
    """,
)

# ---------------------------------------------------------------- 5: rank & null space
M4 = np.array([[1., 2., 3., 4.],
               [2., 4., 6., 8.],
               [1., 0., 1., 0.]])
_rank4 = int(np.linalg.matrix_rank(M4))
HASH_4_RANK = sha_of(_rank4)                   # 2
HASH_4_NULLITY = sha_of(M4.shape[1] - _rank4)  # 2
assert _rank4 == 2
ex.add(
    title="Rank and null space of a given matrix", difficulty=2,
    prompt_md=r"""
    Consider

    $$
    M = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 6 & 8 \\ 1 & 0 & 1 & 0 \end{pmatrix}.
    $$

    1. Work out **by hand** which columns are combinations of which others, and set `rank_M` (an
       `int`) and `nullity_M` (an `int`, the dimension of the null space). Only then confirm with
       `np.linalg.matrix_rank`.
    2. Set `null_vec` to any **non-zero** NumPy array of shape `(4,)` with $M\,\mathbf{v} = \mathbf{0}$,
       read off from the column relations you found (no `scipy.linalg.null_space`).
    3. Set `b_solvable` to `True` or `False`: does $M\mathbf{x} = (3, 6, 1)^\top$ have a solution?
       And `b2_solvable` for $M\mathbf{x} = (1, 1, 1)^\top$? (Hint: is the right-hand side in the
       column space? Compare rows 1 and 2.)
    """,
    starter=r"""
    M = np.array([[1., 2., 3., 4.],
                  [2., 4., 6., 8.],
                  [1., 0., 1., 0.]])

    rank_M = None        # YOUR ANSWER HERE (int)
    nullity_M = None     # YOUR ANSWER HERE (int)
    null_vec = None      # YOUR CODE HERE: np.array of shape (4,), non-zero, with M @ null_vec == 0
    b_solvable = None    # True / False for b = (3, 6, 1)
    b2_solvable = None   # True / False for b = (1, 1, 1)
    """,
    solution=r"""
    M = np.array([[1., 2., 3., 4.],
                  [2., 4., 6., 8.],
                  [1., 0., 1., 0.]])

    # column 3 = column 1 + column 2;  column 4 = 2 * column 2;  columns 1 and 2 are independent
    rank_M = 2
    nullity_M = 4 - rank_M
    null_vec = np.array([1., 1., -1., 0.])      # c1 + c2 - c3 = 0  (another choice: (0, 2, 0, -1))
    b_solvable = True       # (3, 6, 1) is column 3 itself, so x = (0, 0, 1, 0) works
    b2_solvable = False     # row 2 of M is twice row 1, so any M x has second entry = 2 * first entry; 1 != 2
    print("matrix_rank:", np.linalg.matrix_rank(M), "  M @ null_vec =", M @ null_vec)
    """,
    check=rf"""
    check_hash('hidden: rank of M', rank_M, '{HASH_4_RANK}')
    check_hash('hidden: nullity of M', nullity_M, '{HASH_4_NULLITY}')
    try:
        v = np.asarray(null_vec, dtype=float)
        ok = v.shape == (4,) and np.linalg.norm(v) > 1e-9 and np.allclose(M @ v, 0.0)
    except Exception:
        ok = False
    check('null_vec is a non-zero vector in the null space of M', ok)
    check_hash('hidden: M x = (3, 6, 1) solvable?', b_solvable, '{sha_of(True)}')
    check_hash('hidden: M x = (1, 1, 1) solvable?', b2_solvable, '{sha_of(False)}')
    """,
    explanation_md=r"""
    Column 3 is column 1 plus column 2 and column 4 is twice column 2, so only two columns are
    independent: rank 2, and by rank–nullity the null space has dimension $4 - 2 = 2$. Each column
    relation is a null vector — $\mathbf{c}_1 + \mathbf{c}_2 - \mathbf{c}_3 = \mathbf{0}$ gives
    $(1, 1, -1, 0)$ and $2\mathbf{c}_2 - \mathbf{c}_4 = \mathbf{0}$ gives $(0, 2, 0, -1)$; together they
    span the null space. Solvability is membership in the column space: $(3, 6, 1)$ *is* a column,
    while $(1, 1, 1)$ violates the constraint that every output has second entry equal to twice its
    first (row 2 is twice row 1). Looking only at "three non-zero rows, so rank 3" is the classic
    mistake — rank counts *independent* rows, and here there are two.
    """,
)

# ---------------------------------------------------------------- 5: solve 3x3
E5 = np.array([[2., 1., -1.],
               [-3., -1., 2.],
               [-2., 1., 2.]])
b5 = np.array([8., -11., -3.])
_x5 = np.linalg.solve(E5, b5)            # (2, 3, -1)
HASH_5_X = safe_sha(_x5)
HASH_5_DET = safe_sha(float(np.linalg.det(E5)))   # -1
ex.add(
    title="Solve a 3×3 system (by hand, then by `solve`)", difficulty=2,
    prompt_md=r"""
    Solve

    $$
    \begin{aligned}
    2x + y - z &= 8 \\
    -3x - y + 2z &= -11 \\
    -2x + y + 2z &= -3
    \end{aligned}
    $$

    First by hand with elimination (write the steps on paper — it is three rows), then in code.
    Set `A5` and `b5` to the coefficient matrix and right-hand side, compute `x5 = np.linalg.solve(A5, b5)`
    and `det5 = np.linalg.det(A5)`. **Do not** use `np.linalg.inv`. Finally set `n_solutions` to the
    string `'one'`, `'none'` or `'infinite'` describing the solution set (which the determinant tells you).
    """,
    starter=r"""
    A5 = None            # YOUR CODE HERE: 3x3 coefficient matrix
    b5 = None            # YOUR CODE HERE: right-hand side
    x5 = None            # YOUR CODE HERE: np.linalg.solve(...)
    det5 = None          # YOUR CODE HERE
    n_solutions = None   # 'one', 'none' or 'infinite'
    """,
    solution=r"""
    A5 = np.array([[2., 1., -1.],
                   [-3., -1., 2.],
                   [-2., 1., 2.]])
    b5 = np.array([8., -11., -3.])
    x5 = np.linalg.solve(A5, b5)
    det5 = np.linalg.det(A5)
    n_solutions = 'one'          # det != 0  ->  A5 is invertible  ->  exactly one solution
    print("x =", x5, "  det =", round(det5, 6), "  residual:", np.linalg.norm(A5 @ x5 - b5))
    """,
    check=rf"""
    try:
        resid = float(np.linalg.norm(np.asarray(A5, float) @ np.asarray(x5, float) - np.asarray(b5, float)))
    except Exception:
        resid = None
    check('A5 @ x5 reproduces b5 (residual < 1e-9)', resid is not None and resid < 1e-9)
    check_hash('hidden: the solution vector (x, y, z)', (np.round(np.asarray(x5, float), 6) if x5 is not None else None), '{HASH_5_X}')
    check_hash('hidden: det(A5)', (float(det5) if det5 is not None else None), '{HASH_5_DET}')
    check_hash('hidden: number of solutions', n_solutions, '{sha_of("one")}')
    """,
    explanation_md=r"""
    Elimination: add $\tfrac32\times$ row 1 to row 2 and row 1 to row 3 to clear the $x$ column,
    giving $\tfrac12 y + \tfrac12 z = 1$ and $2y + z = 5$; then $y = 2 - z$ substituted into the second
    gives $4 - 2z + z = 5$, so $z = -1$, $y = 3$, and from row 1, $2x + 3 + 1 = 8$, $x = 2$. The
    determinant is $-1 \neq 0$, so the matrix is invertible and the solution is unique. `np.linalg.solve`
    does exactly this elimination (with row swaps for stability) in $\tfrac23 n^3$ flops; forming the
    inverse first would triple the work and, for ill-conditioned systems, lose accuracy.
    """,
)

# ---------------------------------------------------------------- 6: least squares
SOL_6 = r"""
def lstsq_normal(A, y):
    # normal equations  (A^T A) x = A^T y, solved with a factorisation (no inverse)
    return np.linalg.solve(A.T @ A, A.T @ y)
"""
_ns6 = run_solution(SOL_6)
A6 = np.column_stack([np.ones(4), np.arange(4.)])
y6 = np.array([1., 2., 2., 4.])
HASH_6_COEF = safe_sha(_ns6["lstsq_normal"](A6, y6))        # (0.9, 0.9)
_r6 = y6 - A6 @ _ns6["lstsq_normal"](A6, y6)
HASH_6_RSS = safe_sha(float(_r6 @ _r6))                      # 0.7
ex.add(
    title="Least squares via the normal equations", difficulty=2,
    prompt_md=r"""
    Implement `lstsq_normal(A, y)` returning the least-squares coefficients $\hat{\mathbf{x}}$ that
    minimise $\lVert A\mathbf{x} - \mathbf{y}\rVert^2$, by forming and solving the normal equations
    $A^\top A\,\hat{\mathbf{x}} = A^\top\mathbf{y}$ with `np.linalg.solve`. Do **not** use
    `np.linalg.lstsq`, `np.linalg.inv` or `np.linalg.pinv`.

    Then fit the line $y = c + m\,t$ to the four points $(0, 1), (1, 2), (2, 2), (3, 4)$ by building
    the design matrix `A` with an intercept column and a `t` column; store the coefficients in
    `coef_line` and the residual sum of squares $\sum_i r_i^2$ in `rss_line`. Work the $2\times2$
    normal equations out by hand first and compare.
    """,
    starter=r"""
    def lstsq_normal(A, y):
        # YOUR CODE HERE
        return None

    t = np.array([0., 1., 2., 3.])
    y_pts = np.array([1., 2., 2., 4.])
    A_line = None        # YOUR CODE HERE: design matrix with columns [1, t]
    coef_line = None     # YOUR CODE HERE
    rss_line = None      # YOUR CODE HERE: residual sum of squares
    """,
    solution=dd(SOL_6, r"""
    t = np.array([0., 1., 2., 3.])
    y_pts = np.array([1., 2., 2., 4.])
    A_line = np.column_stack([np.ones_like(t), t])
    coef_line = lstsq_normal(A_line, y_pts)
    r = y_pts - A_line @ coef_line
    rss_line = float(r @ r)
    print("coefficients (intercept, slope):", coef_line, "  RSS:", round(rss_line, 6))
    print("residual orthogonal to the columns, A^T r =", A_line.T @ r)
    """),
    check=rf"""
    Ar = np.column_stack([np.ones(30), rng.uniform(0, 10, 30), rng.standard_normal(30)])
    yr = Ar @ np.array([1.5, -0.3, 2.0]) + rng.normal(0, 0.5, 30)
    try:
        got = lstsq_normal(Ar, yr)
    except Exception:
        got = None
    ref = np.linalg.lstsq(Ar, yr, rcond=None)[0]
    check_close('agrees with np.linalg.lstsq on a 30x3 design', got, ref, tol=1e-6)
    try:
        orth = np.linalg.norm(Ar.T @ (yr - Ar @ got))
    except Exception:
        orth = None
    check('residual is orthogonal to every column (A^T r = 0)', orth is not None and orth < 1e-8)
    check_hash('hidden: (intercept, slope) of the 4-point line', (np.round(np.asarray(coef_line, float), 6) if coef_line is not None else None), '{HASH_6_COEF}')
    check_hash('hidden: residual sum of squares of the 4-point line', (float(rss_line) if rss_line is not None else None), '{HASH_6_RSS}')
    """,
    explanation_md=r"""
    The residual of the best fit must be orthogonal to the column space — otherwise sliding along
    some column would reduce it — and "orthogonal to every column" is $A^\top\mathbf{r} = \mathbf{0}$,
    i.e. the normal equations. For the four points $A^\top A = \begin{pmatrix}4&6\\6&14\end{pmatrix}$ and
    $A^\top\mathbf{y} = (9, 18)^\top$, giving $c = m = 0.9$ and residuals $(0.1, 0.2, -0.7, 0.4)$ with
    sum of squares $0.70$. The normal equations are fine for a handful of well-scaled columns, but
    $\kappa(A^\top A) = \kappa(A)^2$, so for wide or badly scaled designs prefer QR or `lstsq`
    (SVD), which never form $A^\top A$. Forgetting the intercept column forces the line through the
    origin and is the most common wrong turn.
    """,
)

# ---------------------------------------------------------------- 8: hat matrix
SOL_8 = r"""
def hat_matrix(A):
    # H = A (A^T A)^{-1} A^T, computed with a solve instead of an explicit inverse
    return A @ np.linalg.solve(A.T @ A, A.T)

def is_projection(H, tol=1e-8):
    symmetric = np.allclose(H, H.T, atol=tol)
    idempotent = np.allclose(H @ H, H, atol=tol)
    return bool(symmetric and idempotent)
"""
_ns8 = run_solution(SOL_8)
_A8 = np.column_stack([np.ones(4), np.arange(4.)])
_H8 = _ns8["hat_matrix"](_A8)
HASH_8_TRACE = safe_sha(float(np.trace(_H8)))                 # 2.0
HASH_8_LEV = safe_sha(np.diag(_H8))                            # (0.7, 0.3, 0.3, 0.7)
ex.add(
    title="The hat matrix is a projection", difficulty=2,
    prompt_md=r"""
    The fitted values of least squares are $\hat{\mathbf{y}} = H\mathbf{y}$ with the **hat matrix**
    $H = A(A^\top A)^{-1}A^\top$.

    1. Implement `hat_matrix(A)` returning $H$ for a full-column-rank design matrix `A` of shape
       `(m, n)`. Do not call `np.linalg.inv` — use `np.linalg.solve(A.T @ A, A.T)` to obtain
       $(A^\top A)^{-1}A^\top$ directly.
    2. Implement `is_projection(H, tol=1e-8)` returning `True` iff `H` is symmetric *and*
       idempotent ($H^2 = H$) to within `tol`.
    3. For the 4-point design matrix of the previous exercise (columns $\mathbf{1}$ and $t = 0,1,2,3$)
       store `trace_H = np.trace(H)` and `leverages = np.diag(H)`. Before running, predict the
       trace from the number of parameters and say which observations have the most leverage.
    """,
    starter=r"""
    def hat_matrix(A):
        # YOUR CODE HERE
        return None

    def is_projection(H, tol=1e-8):
        # YOUR CODE HERE
        return None

    A_line = np.column_stack([np.ones(4), np.arange(4.)])
    trace_H = None       # YOUR CODE HERE
    leverages = None     # YOUR CODE HERE (np.diag of the hat matrix)
    """,
    solution=dd(SOL_8, r"""
    A_line = np.column_stack([np.ones(4), np.arange(4.)])
    H_line = hat_matrix(A_line)
    trace_H = float(np.trace(H_line))
    leverages = np.diag(H_line)
    print("H =\n", H_line.round(3))
    print("trace(H) =", round(trace_H, 6), "  leverages =", leverages.round(3), "  projection:", is_projection(H_line))
    print("I - H is also a projection:", is_projection(np.eye(4) - H_line))
    """),
    check=rf"""
    Ar = np.column_stack([np.ones(12), rng.standard_normal(12), rng.standard_normal(12)])
    try:
        H = hat_matrix(Ar)
        ok_shape = H.shape == (12, 12)
        ok_proj = bool(is_projection(H))
        ok_fit = np.allclose(H @ Ar, Ar)
    except Exception:
        ok_shape = ok_proj = ok_fit = False
    check('hat matrix has shape (m, m)', ok_shape)
    check('hat matrix is symmetric and idempotent', ok_proj)
    check('H leaves the columns of A unchanged (H A = A)', ok_fit)
    try:
        shear_verdict = is_projection(np.array([[1., 1.], [0., 1.]]))
    except Exception:
        shear_verdict = None
    check('is_projection rejects a shear matrix (and accepted the hat matrix above)', ok_proj and shear_verdict is not None and not bool(shear_verdict))
    check_hash('hidden: trace of the 4-point hat matrix', (float(trace_H) if trace_H is not None else None), '{HASH_8_TRACE}')
    check_hash('hidden: leverages of the 4 points', (np.asarray(leverages, float) if leverages is not None else None), '{HASH_8_LEV}')
    """,
    explanation_md=r"""
    $H$ projects onto the column space of $A$: it is symmetric because $(A(A^\top A)^{-1}A^\top)^\top$
    reproduces itself (the middle factor is symmetric), and idempotent because
    $A(A^\top A)^{-1}A^\top A(A^\top A)^{-1}A^\top$ collapses the inner $(A^\top A)^{-1}A^\top A$ to $I$.
    Projecting a vector already in the column space — any column of $A$ — changes nothing, hence
    $HA = A$. The trace of a projector equals the dimension of its range, here 2 (intercept and
    slope), so the leverages $h_{ii}$ average $2/4 = 0.5$; the end points $t = 0$ and $t = 3$ carry
    $0.7$ each because a line is pinned down by its extremes. A shear is idempotent only if $k = 0$,
    and is not symmetric, so `is_projection` must reject it.
    """,
)

# ---------------------------------------------------------------- 9: conceptual — why not invert
ex.add_conceptual(
    title="Why `solve`, not `inv`?", difficulty=2,
    prompt_md=r"""
    A colleague's code computes `w = np.linalg.inv(X.T @ X) @ X.T @ y` for a regression with
    $n = 10^5$ rows and $d = 2{,}000$ features, and then reports that two features whose columns are
    exact unit conversions of each other "got weird coefficients around $\pm 10^{12}$".

    1. Explain why `inv(A) @ b` is a poor choice even when it works: compare the flop counts and
       the error behaviour with `np.linalg.solve(A, b)`.
    2. Diagnose the $\pm 10^{12}$ coefficients. What property of $X$ causes them, what would
       `np.linalg.matrix_rank(X)` report, and why does `inv` not simply raise an error?
    3. Suppose the two columns are *not* exactly collinear but have correlation $0.9999$. Does the
       diagnosis change? Which quantity would you compute to quantify the risk, and name two remedies.
    """,
    answer_md=r"""
    1. Solving via LU factorisation costs about $\tfrac23 d^3$ flops; forming the inverse costs about
       $2d^3$ (an LU plus $d$ triangular solves) and then the product $A^{-1}\mathbf{b}$ adds another
       $2d^2$. Worse, each entry of $A^{-1}$ carries rounding error of relative size $\kappa(A)\varepsilon$,
       and the multiplication by $\mathbf{b}$ mixes them; the triangular solves applied directly to
       $\mathbf{b}$ are backward stable, so `solve` gives a result that is the exact solution of a
       problem perturbed by only $O(\varepsilon)$. For $d = 2{,}000$ the inverse route wastes roughly
       $3\times$ the time and some digits for nothing.
    2. Two exactly proportional columns make $X$ rank deficient: `matrix_rank(X)` returns $1{,}999$
       (or fewer), and $X^\top X$ is singular in exact arithmetic. In floating point the two columns are
       proportional only up to rounding, so $X^\top X$ has a tiny but non-zero smallest eigenvalue
       (around $10^{-16}$ times the largest) rather than an exact zero; `inv` therefore does not detect
       singularity and returns entries of order $1/\lambda_{\min}$, i.e. astronomically large, with opposite
       signs on the two redundant features so that their contributions cancel. Any vector in the null
       space can be added to $\mathbf{w}$ without changing $X\mathbf{w}$, so those coefficients are not
       identifiable at all.
    3. The diagnosis becomes "nearly rank deficient": $X^\top X$ is invertible but ill-conditioned. The
       quantity to compute is the condition number $\kappa(X)$ (or the variance inflation factor of
       the two columns); with $\kappa(X^\top X) = \kappa(X)^2$ the coefficient estimates have standard
       errors inflated by roughly $1/\sqrt{1 - 0.9999^2} \approx 70$. Remedies: drop or merge one of the two
       columns; use ridge regression, which solves $(X^\top X + \lambda I)\mathbf{w} = X^\top\mathbf{y}$ and
       bounds the smallest eigenvalue below by $\lambda$; or compute the fit with `lstsq`/QR, which work on
       $X$ directly and never square the condition number — though that only fixes the numerics, not
       the identifiability.
    """,
)

# ---------------------------------------------------------------- 10: collinearity detection
SOL_10 = r"""
def dependent_columns(X, tol=1e-8):
    # columns taking part in an exact linear dependency = support of the null space of X
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    null_basis = Vt[s <= tol * s.max()]            # right-singular vectors with ~zero singular value
    if null_basis.size == 0:
        return []
    involved = np.any(np.abs(null_basis) > 1e-6, axis=0)
    return sorted(int(j) for j in np.flatnonzero(involved))
"""
_ns10 = run_solution(SOL_10)
_g10 = np.random.default_rng(7)
_n10 = 300
_income = _g10.normal(50, 10, _n10)
_age10 = _g10.uniform(20, 70, _n10)
_tenure = _g10.uniform(0, 30, _n10)
_region = _g10.integers(0, 4, _n10)
_X10 = np.column_stack([np.ones(_n10), _income, _age10, _tenure, _income * 1000,
                        np.eye(4)[_region], _g10.normal(size=_n10)])
_rank10 = int(np.linalg.matrix_rank(_X10))
_dep10 = _ns10["dependent_columns"](_X10)
assert _rank10 == 8 and _dep10 == [0, 1, 4, 5, 6, 7, 8], (_rank10, _dep10)
HASH_10_RANK = sha_of(_rank10)
HASH_10_DEP = sha_of(_dep10)
ex.add(
    title="Detect collinear features in a data matrix", difficulty=2,
    prompt_md=r"""
    The cell below builds a customer feature matrix `X_cust` with 300 rows and these 10 columns:

    | index | column | index | column |
    |---|---|---|---|
    | 0 | intercept (all ones) | 5 | region = North (0/1) |
    | 1 | income (kEUR) | 6 | region = East (0/1) |
    | 2 | age | 7 | region = South (0/1) |
    | 3 | tenure (years) | 8 | region = West (0/1) |
    | 4 | income (EUR) | 9 | noise feature |

    1. Set `rank_X = np.linalg.matrix_rank(X_cust)` and predict it *before* running: how many
       independent columns are there?
    2. Implement `dependent_columns(X, tol=1e-8)` returning the sorted `list` of column indices that
       take part in an exact linear dependency — the columns on which some non-zero null-space
       vector has a non-zero entry. Hint: `np.linalg.svd(X, full_matrices=False)` returns `U, s, Vt`;
       the rows of `Vt` whose singular value `s` is $\le$ `tol * s.max()` span the null space. Treat
       entries of magnitude below `1e-6` as zero.
    3. Set `drop_cols` to a list of column indices such that removing exactly those columns leaves a
       matrix of full column rank, using as few columns as possible.
    """,
    starter=r"""
    g = np.random.default_rng(7)
    n = 300
    income = g.normal(50, 10, n); age = g.uniform(20, 70, n); tenure = g.uniform(0, 30, n)
    region = g.integers(0, 4, n)
    X_cust = np.column_stack([np.ones(n), income, age, tenure, income * 1000,
                              np.eye(4)[region], g.normal(size=n)])

    rank_X = None            # YOUR CODE HERE

    def dependent_columns(X, tol=1e-8):
        # YOUR CODE HERE
        return None

    drop_cols = None         # YOUR ANSWER HERE: list of column indices to drop
    """,
    solution=dd(r"""
    g = np.random.default_rng(7)
    n = 300
    income = g.normal(50, 10, n); age = g.uniform(20, 70, n); tenure = g.uniform(0, 30, n)
    region = g.integers(0, 4, n)
    X_cust = np.column_stack([np.ones(n), income, age, tenure, income * 1000,
                              np.eye(4)[region], g.normal(size=n)])

    rank_X = int(np.linalg.matrix_rank(X_cust))     # 10 columns, 2 exact dependencies -> 8
    """, SOL_10, r"""
    # one column from each dependency: income-in-EUR duplicates income-in-kEUR,
    # and the four region dummies sum to the intercept (dummy-variable trap)
    drop_cols = [4, 8]
    print("rank:", rank_X, "  dependent columns:", dependent_columns(X_cust))
    print("rank after dropping", drop_cols, ":", np.linalg.matrix_rank(np.delete(X_cust, drop_cols, axis=1)))
    """),
    check=rf"""
    check_hash('hidden: rank of X_cust', rank_X, '{HASH_10_RANK}')
    try:
        got = dependent_columns(X_cust)
        got = [int(j) for j in got]
    except Exception:
        got = None
    check_hash('hidden: columns involved in exact dependencies', got, '{HASH_10_DEP}')
    try:
        full = dependent_columns(np.column_stack([np.ones(20), rng.standard_normal((20, 3))]))
        ok_full = list(full) == []
    except Exception:
        ok_full = False
    check('a full-rank matrix has no dependent columns', ok_full)
    try:
        X_small = np.array([[1., 2., 3.], [1., 0., 1.], [2., 2., 4.], [0., 1., 1.]])   # col3 = col1 + col2
        ok_small = [int(j) for j in dependent_columns(X_small)] == [0, 1, 2]
    except Exception:
        ok_small = False
    check('finds c1 + c2 = c3 in a 4x3 example', ok_small)
    try:
        X_red = np.delete(X_cust, list(drop_cols), axis=1)
        ok_drop = len(drop_cols) == X_cust.shape[1] - np.linalg.matrix_rank(X_cust) and np.linalg.matrix_rank(X_red) == X_red.shape[1]
    except Exception:
        ok_drop = False
    check('drop_cols removes the minimum number of columns and leaves full column rank', ok_drop)
    """,
    explanation_md=r"""
    Two exact dependencies hide in ten columns: income in EUR is $1000\times$ income in kEUR, and the
    four region indicators sum to the intercept column (every customer is in exactly one region).
    Hence the rank is $10 - 2 = 8$, the null space is two-dimensional, and its vectors are supported
    on columns $\{1, 4\}$ and $\{0, 5, 6, 7, 8\}$ — seven columns in total, which is what the SVD
    reveals through the right-singular vectors with zero singular value. Age, tenure and the noise
    column are not involved. Dropping one column from each group (e.g. 4 and 8 — any member of a
    group works) restores full column rank; dropping two from the same group would not. This is the
    `drop_first=True` of one-hot encoding, derived rather than memorised.
    """,
)

# ---------------------------------------------------------------- 11: Markov chain (stretch)
SOL_11 = r"""
def propagate(P, pi0, k):
    pi = np.asarray(pi0, dtype=float)
    for _ in range(k):
        pi = pi @ P
    return pi

def stationary(P):
    # solve pi (P - I) = 0  together with  sum(pi) = 1:
    # transpose to (P^T - I) pi^T = 0, then replace the last equation by the normalisation
    n = P.shape[0]
    M = P.T - np.eye(n)
    M[-1, :] = 1.0
    rhs = np.zeros(n); rhs[-1] = 1.0
    return np.linalg.solve(M, rhs)
"""
_ns11 = run_solution(SOL_11)
P11 = np.array([[0.70, 0.20, 0.10],
                [0.20, 0.60, 0.20],
                [0.30, 0.30, 0.40]])
_pi11 = _ns11["stationary"](P11)
_pi3 = _ns11["propagate"](P11, np.array([1., 0., 0.]), 3)
assert np.allclose(_pi11 @ P11, _pi11)
HASH_11_PI = safe_sha(_pi11)
HASH_11_P3 = safe_sha(_pi3)
ex.add(
    title="Markov chain: propagate, then find the stationary distribution", difficulty=3,
    prompt_md=r"""
    A subscription business classifies customers each month as **Active**, **At risk** or
    **Dormant**, with monthly transition matrix (rows = current state, columns = next state)

    $$
    P = \begin{pmatrix} 0.70 & 0.20 & 0.10 \\ 0.20 & 0.60 & 0.20 \\ 0.30 & 0.30 & 0.40 \end{pmatrix}.
    $$

    1. Implement `propagate(P, pi0, k)` returning the row vector $\boldsymbol\pi_0 P^k$ (the state
       distribution after `k` months) using only `@` in a loop — no `np.linalg.matrix_power`.
    2. Implement `stationary(P)` returning the stationary distribution $\boldsymbol\pi$ with
       $\boldsymbol\pi P = \boldsymbol\pi$ and $\sum_i \pi_i = 1$ **by solving a linear system** with
       `np.linalg.solve` — not by iterating and not with `np.linalg.eig`. Hint: $\boldsymbol\pi(P - I) = \mathbf{0}$
       is a homogeneous system with a one-dimensional solution set; transpose it to $(P^\top - I)\boldsymbol\pi^\top = \mathbf{0}$
       and replace one of its (redundant) equations by the normalisation $\sum_i \pi_i = 1$.
    3. Store `pi_3 = propagate(P, [1, 0, 0], 3)` and `pi_inf = stationary(P)`.
    """,
    starter=r"""
    P = np.array([[0.70, 0.20, 0.10],
                  [0.20, 0.60, 0.20],
                  [0.30, 0.30, 0.40]])

    def propagate(P, pi0, k):
        # YOUR CODE HERE
        return None

    def stationary(P):
        # YOUR CODE HERE
        return None

    pi_3 = None      # YOUR CODE HERE
    pi_inf = None    # YOUR CODE HERE
    """,
    solution=dd(r"""
    P = np.array([[0.70, 0.20, 0.10],
                  [0.20, 0.60, 0.20],
                  [0.30, 0.30, 0.40]])
    """, SOL_11, r"""
    pi_3 = propagate(P, [1., 0., 0.], 3)
    pi_inf = stationary(P)
    print("after 3 months:", pi_3, "   stationary:", pi_inf)
    print("check pi P = pi:", np.allclose(pi_inf @ P, pi_inf), "   after 50 months from any start:", propagate(P, [0., 0., 1.], 50))
    """),
    check=rf"""
    try:
        got1 = propagate(P, np.array([0., 1., 0.]), 1)
    except Exception:
        got1 = None
    check_close('one step from "At risk" is the second row of P', got1, P[1])
    check_hash('hidden: distribution after 3 months from Active', (np.asarray(pi_3, float) if pi_3 is not None else None), '{HASH_11_P3}')
    try:
        s = np.asarray(stationary(P), float)
        ok_stat = s.shape == (3,) and np.allclose(s @ P, s) and abs(s.sum() - 1) < 1e-9 and np.all(s > 0)
    except Exception:
        ok_stat = False
    check('stationary(P) satisfies pi P = pi and sums to 1', ok_stat)
    check_hash('hidden: the stationary distribution', (np.asarray(pi_inf, float) if pi_inf is not None else None), '{HASH_11_PI}')
    try:
        P2 = np.array([[0.9, 0.1], [0.5, 0.5]])
        ok2 = np.allclose(stationary(P2), [5 / 6, 1 / 6])
    except Exception:
        ok2 = False
    check('two-state weather chain gives (5/6, 1/6)', ok2)
    try:
        far = propagate(P, [0., 0., 1.], 60)
        ok_conv = np.allclose(far, pi_inf, atol=1e-6)
    except Exception:
        ok_conv = False
    check('propagating 60 steps from any start converges to the stationary distribution', ok_conv)
    """,
    explanation_md=r"""
    Propagation is just repeated multiplication by $P$ on the right — one month per product.
    The stationary distribution is a left eigenvector for eigenvalue 1, but you do not need an
    eigen-solver: $(P^\top - I)\boldsymbol\pi^\top = \mathbf{0}$ has rank $n - 1$ (its rows sum to zero because
    the rows of $P$ sum to one), so one equation is redundant and can be swapped for
    $\sum_i\pi_i = 1$, which makes the system square and invertible. Trying `np.linalg.solve(P.T - I, 0)`
    without that swap returns the useless all-zeros vector or raises a singularity error — the
    classic wrong turn. Because this chain is irreducible and aperiodic (every entry of $P$ is
    positive), $\boldsymbol\pi_0 P^k \to \boldsymbol\pi$ for every start, which the 60-step check confirms.
    """,
)

# ---------------------------------------------------------------- 12: mini-project
PROJECT_DATA = r"""
# --- synthetic housing data (run this cell as given) -----------------------------------
g = np.random.default_rng(123)
n = 200
size_m2 = g.uniform(40, 160, n)                                   # living area
rooms = np.clip(np.round(size_m2 / 35 + g.normal(0, 0.7, n)), 1, 6)   # correlated with size
age_years = g.uniform(0, 60, n)
price_k = 50 + 2.5 * size_m2 + 15 * rooms - 1.2 * age_years + g.normal(0, 25, n)   # kEUR
"""
ex.add_project(
    title="Mini-project — multi-feature linear regression with your own solver", difficulty=3,
    prompt_md=r"""
    The data cell generates 200 synthetic apartments with living area `size_m2`, number of `rooms`,
    `age_years` and a sale price `price_k` in kEUR. The price was generated as
    $50 + 2.5\,\text{size} + 15\,\text{rooms} - 1.2\,\text{age} + \varepsilon$ with
    $\varepsilon \sim \mathcal{N}(0, 25^2)$ — pretend you do not know that.

    Build an ordinary least-squares model of price from the three features **using only NumPy**:

    1. Assemble the design matrix $X$ (with an intercept column) and target $\mathbf{y}$; report the
       shape and rank of $X$ and the condition numbers $\kappa(X)$ and $\kappa(X^\top X)$.
    2. Fit the coefficients three ways — normal equations with `np.linalg.solve`, QR
       ($R\hat{\mathbf{w}} = Q^\top\mathbf{y}$) and `np.linalg.lstsq` — and confirm they agree.
    3. Interpret every coefficient in words with its units (kEUR per m², per room, per year), and
       compare with the generating values.
    4. Compute the fitted values via the hat matrix, the residual sum of squares, the residual
       standard deviation $\sqrt{\mathrm{RSS}/(n - p)}$ and $R^2 = 1 - \mathrm{RSS}/\sum_i (y_i - \bar y)^2$;
       check that the residuals are orthogonal to every column.
    5. Verify against `sklearn.linear_model.LinearRegression` (`coef_`, `intercept_`).
    6. Add a fourth feature `size_sqft = size_m2 * 10.764` and show what happens to the rank,
       the condition number and the coefficients — then explain what you see in one sentence.
    7. Plot predicted versus actual prices (title, axis labels, a 45° reference line).
    """,
    rubric_md=r"""
    * The design matrix has an intercept column, shape `(200, 4)`, rank 4, and you report both condition numbers and explain why $\kappa(X^\top X) = \kappa(X)^2$.
    * Normal equations, QR and `lstsq` agree to at least 8 significant digits and match scikit-learn's `coef_`/`intercept_`.
    * Each coefficient is stated with units and compared with the true values $(50, 2.5, 15, -1.2)$; you note that `rooms` is correlated with `size_m2` ($\rho \approx 0.77$) and say what that does to the precision of those two coefficients.
    * Residuals satisfy $X^\top\mathbf{r} \approx \mathbf{0}$; the residual standard deviation is close to the noise level 25 and $R^2$ is reported.
    * Adding `size_sqft` drops the rank to 4 of 5 columns, `np.linalg.solve` fails or returns absurd coefficients, and you explain this as an exact collinearity (null-space direction) rather than a numerical accident.
    * The predicted-vs-actual plot is labelled and includes the $y = x$ reference line.
    """,
    starter=dd(PROJECT_DATA, r"""

# Your mini-project workspace — follow steps 1 to 7 of the prompt.
"""),
    solution_sketch_md=r"""
    Stack `[1, size_m2, rooms, age_years]` into $X$ (200 × 4). All three solvers give
    $\hat{\mathbf{w}} \approx (53.1,\ 2.53,\ 13.4,\ -1.24)$: about 2.5 kEUR per extra square metre,
    13 kEUR per additional room and a loss of 1.2 kEUR per year of age, close to the generating
    values; the intercept and the `rooms` coefficient are the least precise because `rooms` is
    strongly correlated with `size_m2`. $\kappa(X) \approx 400$ and $\kappa(X^\top X) \approx 1.6\times10^5$
    — harmless here, but it shows why the normal equations square the conditioning. The hat matrix
    gives the fitted values, $X^\top \mathbf{r}$ is zero to rounding, the residual standard deviation is
    close to 25 and $R^2 \approx 0.95$. Appending the square-foot column makes column 5 exactly
    $10.764\times$ column 2: the rank stays 4 while the matrix has 5 columns, `solve` raises
    `LinAlgError` (or, if rounding makes $X^\top X$ barely invertible, returns coefficients of order
    $10^{10}$ with opposite signs), and `lstsq` returns the minimum-norm solution that splits the size
    effect arbitrarily between the two columns. scikit-learn's `LinearRegression` agrees with the
    full-rank fit exactly, because it calls `scipy.linalg.lstsq` under the hood.
    """,
    solution_code=dd(PROJECT_DATA, r"""
from sklearn.linear_model import LinearRegression

# 1. design matrix
X = np.column_stack([np.ones(n), size_m2, rooms, age_years])
y = price_k
names = ["intercept", "size_m2", "rooms", "age_years"]
print("X shape:", X.shape, " rank:", np.linalg.matrix_rank(X))
print("kappa(X) = %.1f   kappa(X^T X) = %.1f   (ratio = kappa(X)^2: %.1f)" % (np.linalg.cond(X), np.linalg.cond(X.T @ X), np.linalg.cond(X) ** 2))
print("corr(size, rooms) = %.3f" % np.corrcoef(size_m2, rooms)[0, 1])

# 2. three solvers
w_normal = np.linalg.solve(X.T @ X, X.T @ y)
Q, R = np.linalg.qr(X)
w_qr = np.linalg.solve(R, Q.T @ y)
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
print("\nmax |normal - qr| = %.2e,  max |normal - lstsq| = %.2e" % (np.abs(w_normal - w_qr).max(), np.abs(w_normal - w_lstsq).max()))

# 3. interpretation
true_w = [50, 2.5, 15, -1.2]
units = ["kEUR", "kEUR per m2", "kEUR per room", "kEUR per year"]
for nm, w_hat, w_true, u in zip(names, w_normal, true_w, units):
    print(f"  {nm:10s} = {w_hat:8.3f} {u:15s} (true {w_true})")

# 4. hat matrix, residuals, fit quality
H = X @ np.linalg.solve(X.T @ X, X.T)
y_hat = H @ y
r = y - y_hat
rss = float(r @ r)
p = X.shape[1]
sigma_hat = np.sqrt(rss / (n - p))
r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
print("\ntrace(H) = %.3f,  X^T r = %s" % (np.trace(H), np.round(X.T @ r, 8)))
print("RSS = %.1f, residual sd = %.2f (noise sd was 25), R^2 = %.3f" % (rss, sigma_hat, r2))

# 5. scikit-learn comparison
lr = LinearRegression().fit(X[:, 1:], y)
print("\nsklearn intercept %.4f coef %s" % (lr.intercept_, np.round(lr.coef_, 4)))
print("matches our solver:", np.allclose(np.r_[lr.intercept_, lr.coef_], w_normal))

# 6. an exactly collinear extra column
X5 = np.column_stack([X, size_m2 * 10.764])
print("\nwith size_sqft: shape", X5.shape, " rank", np.linalg.matrix_rank(X5), " kappa(X5) = %.2e" % np.linalg.cond(X5))
try:
    w5 = np.linalg.solve(X5.T @ X5, X5.T @ y)
    print("solve returned", np.round(w5, 3), "(garbage of huge magnitude: the system is singular up to rounding)")
except np.linalg.LinAlgError as e:
    print("np.linalg.solve raised LinAlgError:", e)
w5_lstsq = np.linalg.lstsq(X5, y, rcond=None)[0]
print("lstsq minimum-norm coefficients:", np.round(w5_lstsq, 4))
print("size effect is split: 2.5 ~= coef_m2 + 10.764 * coef_sqft =", round(w5_lstsq[1] + 10.764 * w5_lstsq[4], 4))

# 7. plot
plt.scatter(y, y_hat, s=14)
lo, hi = y.min(), y.max()
plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
plt.xlabel("actual price (kEUR)"); plt.ylabel("predicted price (kEUR)")
plt.title(f"OLS on synthetic apartments: R$^2$ = {r2:.3f}, residual sd = {sigma_hat:.1f} kEUR")
plt.legend(); plt.grid(alpha=0.3); plt.show()
"""),
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
print("wrote exercises.ipynb and solutions.ipynb with", len(ex.items), "exercises")
