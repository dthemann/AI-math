#!/usr/bin/env python3
"""Builder for Module 02 — Matrices and Linear Maps.

Generates lesson.ipynb, exercises.ipynb and solutions.ipynb into
part-1-linear-algebra/02-matrices-and-linear-maps.  Run from the repo root:

    python3 tools/builders/build_02.py
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
**Notice that** the three computations agree to the last digit — they are the same sum, grouped
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

# cubic cost: time the loop version for n, 2n, 4n -- and NumPy's @ for the very same product
for n in (30, 60, 120):
    M = rng.standard_normal((n, n))
    t0 = time.perf_counter(); matmul_loops(M, M); dt = time.perf_counter() - t0
    M @ M                                                 # warm-up call, so start-up costs are not timed
    t0 = time.perf_counter()
    for _ in range(100):
        M @ M
    dt_np = (time.perf_counter() - t0) / 100              # seconds per product, averaged over 100 calls
    print(f"n = {n:4d}: {n**3:>10,d} multiply-adds, loops took {dt*1000:7.1f} ms, M @ M took {dt_np*1e6:6.1f} µs  (~{dt / dt_np:,.0f}x faster)")

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
**Notice that** $AB$ swaps the *columns* of $A$ while $BA$ swaps its *rows* — the same permutation
matrix does different things depending on which side it sits. The loop timings roughly multiply by 8
each time $n$ doubles, the signature of $n^3$. NumPy's `@` performs the same $n^3$ multiply–adds in
compiled, cache-blocked BLAS code, and the last figure on each line shows what that buys: typically
hundreds to tens of thousands of times faster at these sizes, with the gap widening as $n$ grows —
the algorithm is the same, the machine code is not. And `W2 @ W1` reproduces the two-step
computation (up to rounding): without a nonlinearity between them, two layers are one layer with
fewer degrees of freedom than their parameter count suggests.

**Try it yourself:** multiply `batch` (shape `(8, 4, 3)`) by `W1.T` (shape `(3, 5)`) instead of the
random `(3, 2)` matrix, and predict the output shape before you run it — which dimensions does the
product use up, and which does it carry along? Then try `A * B` instead of `A @ B` and compare.
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

maps = {"rotation 30°": rotation(np.pi / 6), "scaling (2, 1.5)": scaling(2, 1.5),
        "shear k=1": shear(1.0), "reflection in x-axis": reflection_x(),
        "projection onto (1,1)": projection([1, 1]), "rotation 30° then scaling": scaling(2, 1.5) @ rotation(np.pi / 6)}

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax, (name, M) in zip(axes.ravel(), maps.items()):
    img, g_img = M @ square, M @ grid
    ax.plot(square[0], square[1], "k--", lw=1, label="unit square")
    ax.fill(img[0], img[1], alpha=0.3, color="tab:blue")
    ax.plot(img[0], img[1], color="tab:blue", lw=2, label="image of the square")
    ax.scatter(g_img[0], g_img[1], s=8, color="tab:red", label="images of grid points", zorder=3)
    for k, (col, lab) in enumerate([("tab:green", "column 1 = image of e1"), ("tab:orange", "column 2 = image of e2")]):
        ax.annotate("", xy=M[:, k], xytext=(0, 0), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, shrinkA=0, shrinkB=0))
        ax.plot([], [], color=col, lw=2.2, label=lab)               # an empty line gives the arrow its legend entry
    names_at = {}                                                    # number the image corners 0 -> 1 -> 2 -> 3 (merge coinciding ones)
    for k, corner in enumerate(img[:, :4].T):
        names_at.setdefault(tuple(np.round(corner, 6)), []).append(str(k))
    for corner, ks in names_at.items():
        ax.annotate(",".join(ks), corner, textcoords="offset points", xytext=(4, 4), fontsize=7, color="tab:blue", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))      # white backing so the numbers stay legible over dots and outlines
    ax.set_title(f"{name}\ndet = {det2(M):+.3f}  (NumPy: {np.linalg.det(M):+.3f})", fontsize=9)
    ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.5, 2.2); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=6.5, loc="lower left")
plt.suptitle("Linear maps acting on the unit square (image corners numbered 0-3 in the order they are visited)")
plt.tight_layout(); plt.show()

S_, R_ = scaling(2, 1.5), rotation(np.pi / 6)
print(f"det(S @ R) = {det2(S_ @ R_):.4f}   det(S) * det(R) = {det2(S_) * det2(R_):.4f}   (= 3.0 x 1.0)")
""")

nb.md(r"""
**Notice that** the rotation and the shear both have determinant $+1$ — the slanted parallelogram has
exactly the same area as the square — while the scaling by $(2, 1.5)$ has determinant $+3$: its image
is a $2 \times 1.5$ rectangle, three times the area. The reflection has determinant $-1$: same area,
but the corners numbered 0, 1, 2, 3 now run clockwise instead of counter-clockwise, because the second
column lies clockwise of the first. The projection has determinant $0$: its corners 1 and 3 land on
the same point, and the red grid points, which were spread over the square, now all lie on one line.
Pairs of different inputs have become the same output, and no matrix can undo that. Composition
multiplies determinants: the last panel (rotate by $30^\circ$, then scale) has determinant
$1 \times 3 = 3$, as the printed line confirms.
""")

nb.code(r"""
# The order of two maps matters: the same rotation and shear applied to an asymmetric letter F, in both orders
F_shape = np.array([[0, .3, .3, .7, .7, .3, .3, .9, .9, 0, 0],
                    [0, 0, .4, .4, .6, .6, .8, .8, 1, 1, 0]])       # a closed polygon without any symmetry
R30, S1 = rotation(np.pi / 6), shear(1.0)                             # helpers from the cell above
orders_ = [("S @ R: rotate by 30° first, then shear", S1 @ R30, R30, "after the rotation only"),
           ("R @ S: shear first, then rotate by 30°", R30 @ S1, S1, "after the shear only")]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharex=True, sharey=True)
for ax, (title, M_, first_, lab_) in zip(axes, orders_):
    ax.plot(*F_shape, color="0.6", ls="--", lw=1, label="original F")
    ax.plot(*(first_ @ F_shape), color="tab:orange", ls=":", lw=1.8, label=lab_)
    ax.fill(*(M_ @ F_shape), color="tab:blue", alpha=0.35)
    ax.plot(*(M_ @ F_shape), color="tab:blue", lw=2, label="final image")
    ax.set_title(f"{title}\ndet = {det2(M_):+.3f}", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
plt.suptitle("Same two maps, opposite order: different pictures, equal determinants")
plt.tight_layout(); plt.show()
print("S @ R =\n", S1 @ R30, "\nR @ S =\n", R30 @ S1, "\nequal:", np.allclose(S1 @ R30, R30 @ S1))
""")

nb.md(r"""
**Notice that** the same two maps give two different pictures when they are applied in the opposite
order: `S @ R` rotates first and shears second (a product acts on its right factor first), `R @ S`
does it the other way round, and the printed products differ in their diagonal entries. Yet both have
determinant $+1$ — determinants are plain numbers and commute, matrices do not. The orange dotted
letter is the intermediate step, so read each panel from the dashed original, through the orange
letter, to the blue one: exactly the right-to-left order in which you read the product.
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
**Notice that** $\mathbf{z}^\top G\mathbf{z}$ equals $\lVert M\mathbf{z}\rVert^2$ for every draw — that is the whole
proof that Gram matrices are positive semidefinite, and it is why covariance matrices (which are
Gram matrices of centred data) always have non-negative eigenvalues. The matrix `K` looks harmless
but gives a negative quadratic form, so it cannot be a covariance matrix. The back-substitution
loop solved a $3\times3$ system with only $3 + 2 + 1 = 6$ multiplications and divisions in all
($n(n+1)/2$ for an $n\times n$ system): the triangular structure is what `np.linalg.solve` buys
itself by first factoring $A = LU$.

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
print("\nsingular values of X_bad:", s[:6], "  smallest two: %.1e and %.1e (rounding noise, i.e. zero)" % tuple(s[6:]))
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
**Notice that** `X_bad` has 8 columns but rank 6: the null space is two-dimensional, and the two
null vectors spell out exactly which columns are redundant — one says
`weight_kg - 0.4536 * weight_lb = 0` (i.e. $1/2.2046 = 0.4536$), the other says
`intercept - mobile - desktop - tablet = 0`. Any linear combination of these two vectors can be
added to a coefficient vector without changing a single prediction, so the "effect of weight" or
"effect of being on mobile" is meaningless until one column from each group is dropped (or a ridge
penalty is added). `X_good` does that and has full column rank.
""")

nb.code(r"""
# What a null-space direction does to the loss: a flat valley (exact duplicate) vs a long thin bowl (sloppy duplicate)
from matplotlib.patches import Patch
g_ = np.random.default_rng(11)                                # its own generator: the lesson's rng stream stays untouched
y_dem = 0.5 * weight_kg + g_.normal(0, 4.0, len(weight_kg))   # an outcome that depends on weight
yc_ = y_dem - y_dem.mean()
lb_sloppy = weight_lb + g_.normal(0, 6.0, len(weight_kg))     # the same weight read off a second, sloppy scale (±6 lb)
zs = lambda v: (v - v.mean()) / v.std()                       # per-standard-deviation units, so both axes are comparable
cases_ = [("exact duplicate: weight_lb = 2.2046 · weight_kg", np.column_stack([zs(weight_kg), zs(weight_lb)])),
          ("sloppy duplicate: weight_lb read with ±6 lb noise", np.column_stack([zs(weight_kg), zs(lb_sloppy)]))]
gw_ = np.linspace(-4, 8, 241); W1_, W2_ = np.meshgrid(gw_, gw_)
band_levels = [1, 1.01, 1.05, 1.25, 2, 1e9]                   # bands of RSS(w) / RSS_min
band_cols = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff"]
fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), layout="constrained")
for ax, (title, Z_) in zip(axes, cases_):
    w_min = np.linalg.lstsq(Z_, yc_, rcond=None)[0]           # the minimum-norm minimiser (what lstsq returns)
    G_ = Z_.T @ Z_; rss_min = np.sum((yc_ - Z_ @ w_min) ** 2)
    D1, D2 = W1_ - w_min[0], W2_ - w_min[1]
    rel = 1 + (G_[0, 0] * D1 ** 2 + 2 * G_[0, 1] * D1 * D2 + G_[1, 1] * D2 ** 2) / rss_min     # RSS(w) / RSS_min
    ax.contourf(W1_, W2_, rel, levels=band_levels, colors=band_cols)
    ax.contour(W1_, W2_, rel, levels=band_levels[1:-1], colors="k", linewidths=0.6)
    ax.plot(*w_min, "r*", ms=14)
    s_ = np.linalg.svd(Z_, compute_uv=False)
    ax.set_title(f"{title}\nsingular values of the two columns: {s_[0]:.1f} and {s_[1]:.1e}", fontsize=9)
    ax.set_xlabel("coefficient of weight_kg (per sd)"); ax.set_ylabel("coefficient of weight_lb (per sd)")
    ax.set_xlim(-4, 8); ax.set_ylim(-4, 8); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    print(f"{title.split(':')[0]:17s}: lstsq answer {np.round(w_min, 2)}, sum of the two coefficients {w_min.sum():.2f}")
    if title.startswith("exact"):
        ax.annotate("", xy=w_min + [2.2, -2.2], xytext=w_min - [2.2, -2.2], arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
        ax.text(-3.8, -3.6, "red arrow = null direction (1, -1):\nmoving along it changes no prediction", color="red", fontsize=7,
                bbox=dict(fc="white", ec="0.8"))
band_names = ["within 1% of the best RSS", "1-5% worse", "5-25% worse", "25-100% worse", "more than 100% worse"]
star_ = plt.Line2D([], [], color="r", marker="*", ls="", ms=11, label="lstsq answer")
axes[0].legend(handles=[star_], fontsize=7, loc="upper right")
axes[1].legend(handles=[star_] + [Patch(facecolor=c, edgecolor="k", linewidth=0.5, label=nm) for c, nm in zip(band_cols, band_names)],
               fontsize=7, loc="upper right", title="RSS(w) compared with its minimum", title_fontsize=7)
plt.suptitle("Loss RSS(w) over the two weight coefficients: a flat valley versus a long thin bowl")
plt.show()
""")

nb.md(r"""
**Notice that** with an exact duplicate the loss has a flat-bottomed valley: every pair of
coefficients with the same sum fits equally well, so no amount of data can say how the effect of
weight is split between the two columns, and `lstsq` merely picks the pair of smallest norm. The red
arrow is the null vector found above — $(1, -0.4536)$ in raw units, $(1, -1)$ per standard deviation —
and the second singular value in the title, zero up to rounding noise, is that flat direction. A sloppy
duplicate closes the valley into a long thin ellipse: there is now a unique minimum, but it is pinned
down only *across* the valley, not *along* it — the situation Section 6 measures with the condition
number.
""")

# ---------------------------------------------------------------- section 6
nb.md(r"""
## 6. Solving $A\mathbf{x} = \mathbf{b}$: `solve`, not `inv`

For an invertible square $A$ the solution is $\mathbf{x} = A^{-1}\mathbf{b}$ *on paper*. In code,
`np.linalg.solve` factors $A = LU$ and does two triangular solves — about $\tfrac23 n^3$ flops versus
roughly $2n^3$ for the explicit inverse — and returns an $\mathbf{x}$ that solves the system you wrote
down to rounding level, which `inv(A) @ b` does not guarantee. How many digits of the answer survive
is governed by the **condition number** $\kappa(A)$: relative perturbations of the data can be
amplified by up to $\kappa(A)$ in the solution.

We solve the juice-blending system from the guide, time `solve` against `inv`, push a $2\times2$
matrix towards singularity to watch the error grow with $\kappa$, draw *why* it grows (a round ball of
noise becomes a sliver), and finish with Hilbert matrices, where `inv(A) @ b` visibly fails and
`solve` does not.
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
kappas, err_pert, err_ulp = [], [], []
for eps in eps_values:
    A2 = np.array([[1., 1.], [1., 1. + eps]])
    b2 = A2 @ x_true
    b_noisy = b2 * (1 + delta * np.array([1., -1.]))          # relative perturbation of size delta
    b_ulp = b2 + np.spacing(b2) * np.array([1., -1.])         # the smallest possible change: one unit in the last place
    kappas.append(np.linalg.cond(A2))
    err_pert.append(np.linalg.norm(np.linalg.solve(A2, b_noisy) - x_true) / np.linalg.norm(x_true))
    err_ulp.append(np.linalg.norm(np.linalg.solve(A2, b_ulp) - x_true) / np.linalg.norm(x_true))
kappas, err_ulp = np.array(kappas), np.array(err_ulp)

plt.figure(figsize=(8, 4.8))
plt.loglog(kappas, err_pert, "o-", label=r"relative error, $b$ perturbed by $10^{-8}$")
plt.loglog(kappas, kappas * delta, "k:", label=r"bound $\kappa(A)\cdot 10^{-8}$")
plt.loglog(kappas, err_ulp, "s--", alpha=0.7, label=r"relative error, $b$ changed by one unit in the last place")
plt.loglog(kappas, kappas * 2.2e-16, "k-.", alpha=0.6, label=r"bound $\kappa(A)\,\varepsilon_{\mathrm{machine}}$")
plt.axhline(1, color="tab:red", ls="--", lw=1, label="relative error = 1 (no correct digit)")
plt.xlabel(r"condition number $\kappa(A)$"); plt.ylabel("relative error in the solution")
plt.title("Perturbations of b are amplified by up to the condition number")
plt.legend(fontsize=7.5, loc="lower right"); plt.grid(alpha=0.3, which="both"); plt.show()
print("kappa * 1e-8 exceeds 1 (no correct digits left) once kappa > 1e8; here kappa reaches %.0e" % kappas[-1])
print("one-ulp error / (kappa * 2.2e-16) lies between %.2f and %.2f at all %d points" % ((err_ulp / (kappas * 2.2e-16)).min(), (err_ulp / (kappas * 2.2e-16)).max(), len(kappas)))

print("\neps = 1e-4: kappa = %.1e" % kappas[3])
A4 = np.array([[1., 1.], [1., 1.0001]])
b_a, b_b = np.array([2., 2.0001]), np.array([2., 2.0002])
x_a, x_b = np.linalg.solve(A4, b_a), np.linalg.solve(A4, b_b)
print("b = (2, 2.0001) -> x =", x_a)
print("b = (2, 2.0002) -> x =", x_b)
db = b_b - b_a
print("b changed by %.0e in its second entry: %.1e relative to that entry, %.1e relative to ||b|| -- yet x moved by %.0f%%"
      % (db[1], db[1] / b_a[1], np.linalg.norm(db) / np.linalg.norm(b_a), 100 * np.linalg.norm(x_b - x_a) / np.linalg.norm(x_a)))
""")

nb.md(r"""
**Notice that** the circles hug the dotted line $\kappa(A)\cdot 10^{-8}$: a perturbation in the eighth
digit of $\mathbf{b}$ is amplified by almost exactly the condition number, and once
$\kappa > 10^{8}$ the relative error in $\mathbf{x}$ exceeds 1 (the red line) — not a single digit of
the answer is reliable. The squares show the smallest change of $\mathbf{b}$ a computer can even
represent, one unit in the last place (a relative change of about $10^{-16}$): they sit on the
dash-dotted line $\kappa(A)\,\varepsilon_{\text{machine}}$ with $\varepsilon_{\text{machine}} \approx 2\times10^{-16}$ —
the same slope, eight orders lower. With $\kappa = 10^{10}$ you keep about six digits, with $10^{16}$ none.
Conditioning is a property of the *problem*, not of the algorithm; `inv` cannot fix it and typically
costs two to five times as much as `solve` (three by the flop count). The last lines are the guide's
example: a change in the last printed digit of $\mathbf{b}$ (the fourth decimal) moves the solution
from $(1, 1)$ to $(0, 2)$, because the two columns of $A$ are almost parallel.
""")

nb.code(r"""
# kappa as an aspect ratio: a round ball of noise in b becomes a long sliver of candidate solutions x
A_c = np.array([[1., 1.], [1., 1.1]])                        # the eps = 0.1 member of the family above
x_c = np.array([0.3, 0.7]); b_c = A_c @ x_c
U_c, s_c, Vt_c = np.linalg.svd(A_c)
radius = 0.02                                                # size of the measurement noise in b
phi = np.linspace(0, 2 * np.pi, 361)
ball = radius * np.vstack([np.cos(phi), np.sin(phi)])        # every perturbation of 2-norm 0.02
cloud = np.linalg.solve(A_c, b_c[:, None] + ball)            # the solutions that these noisy right-hand sides lead to
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
ax1.plot(*(b_c[:, None] + ball), color="tab:blue", lw=2, label=r"$b+\Delta b$, $\|\Delta b\| = 0.02$")
ax1.plot(*b_c, "k.", ms=9, label="exact b")
ax1.set_xlabel("$b_1$"); ax1.set_ylabel("$b_2$"); ax1.set_title("Output space: a round ball of noise around b", fontsize=10)
ax1.set_aspect("equal"); ax1.margins(0.25); ax1.legend(fontsize=8, loc="upper left"); ax1.grid(alpha=0.3)
ax2.plot(*cloud, color="tab:red", lw=2, label=r"$x = A^{-1}(b + \Delta b)$")
ax2.plot(*x_c, "k.", ms=9, label="true x")
for k_, col_ in enumerate(["tab:green", "tab:orange"]):
    ax2.annotate("", xy=x_c + Vt_c[k_] * radius / s_c[k_], xytext=x_c, arrowprops=dict(arrowstyle="-|>", color=col_, lw=2, shrinkA=0, shrinkB=0))
    ax2.plot([], [], color=col_, lw=2, label=r"semi-axis %.3f = 0.02 / $\sigma_%d$" % (radius / s_c[k_], k_ + 1)
             + (" (too short to see)" if k_ == 0 else ""))
ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
ax2.set_title(rf"Input space: a sliver {s_c[0] / s_c[1]:.0f} times longer than wide ($\kappa$ = {np.linalg.cond(A_c):.0f})", fontsize=10)
ax2.set_aspect("equal"); ax2.legend(fontsize=7.5, loc="upper right"); ax2.grid(alpha=0.3)
plt.show()
print("sigma_1 = %.3f, sigma_2 = %.4f, kappa = sigma_1 / sigma_2 = %.1f;  long axis of the sliver along v_2 = %s" % (s_c[0], s_c[1], s_c[0] / s_c[1], np.round(Vt_c[1], 2)))
print("noise radius %.2f is %.1f%% of ||b||, yet x_1 can be anywhere in [%.2f, %.2f] although the true value is %.1f"
      % (radius, 100 * radius / np.linalg.norm(b_c), cloud[0].min(), cloud[0].max(), x_c[0]))
""")

nb.md(r"""
**Notice that** a perfectly round ball of measurement noise in $\mathbf{b}$ becomes a sliver $42$ times
longer than wide in $\mathbf{x}$: its semi-axes are the noise radius divided by the singular values
$\sigma_1 = 2.05$ and $\sigma_2 = 0.049$, so their ratio is $\sigma_1/\sigma_2 = \kappa(A)$. The first
coordinate $x_1$ can therefore be anywhere between $0$ and $0.6$ although its true value is $0.3$ —
an uncertainty of about $100\%$ from noise of $1.4\%$ — and the long direction, $\mathbf{v}_2$ (up to sign
$(-0.72, 0.69)$), is where the two almost equal columns of $A$ trade off against each other. This is
the circle-to-ellipse picture of Section 8 with $A^{-1}$ in place of $A$: the axis ratio of the ellipse is
a ratio of singular values, which the SVD of Module 03 makes precise. At $\kappa = 4\times10^{12}$ the
sliver would be four trillion times longer than wide.
""")

nb.code(r"""
# Same problem, two algorithms: solve returns an x that solves the system you wrote down, inv(A) @ b does not (Hilbert matrices)
orders_h = np.arange(4, 13)
kap_h, res_s, res_i, err_s, err_i = [], [], [], [], []
for m_ in orders_h:
    H_h = 1.0 / (np.arange(m_)[:, None] + np.arange(m_)[None, :] + 1.0)      # Hilbert matrix 1/(i+j+1): kappa grows like e^(3.5 m)
    b_h = H_h @ np.ones(m_)                                                   # chosen so that the exact solution is all ones
    xs_, xi_ = np.linalg.solve(H_h, b_h), np.linalg.inv(H_h) @ b_h
    kap_h.append(np.linalg.cond(H_h))
    res_s.append(np.linalg.norm(H_h @ xs_ - b_h) / np.linalg.norm(b_h)); res_i.append(np.linalg.norm(H_h @ xi_ - b_h) / np.linalg.norm(b_h))
    err_s.append(np.linalg.norm(xs_ - 1) / np.sqrt(m_)); err_i.append(np.linalg.norm(xi_ - 1) / np.sqrt(m_))
kap_h = np.array(kap_h)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
ax1.loglog(kap_h, np.maximum(res_s, 1e-17), "o-", label="solve(H, b)")
ax1.loglog(kap_h, np.maximum(res_i, 1e-17), "s-", label="inv(H) @ b")
ax1.set_xlabel(r"condition number $\kappa(H)$"); ax1.set_ylabel(r"relative residual $\|Hx - b\| / \|b\|$")
ax1.set_title("Residual: does x solve the system you wrote down?", fontsize=10)
ax2.loglog(kap_h, err_s, "o-", label="solve(H, b)")
ax2.loglog(kap_h, err_i, "s-", label="inv(H) @ b")
ax2.loglog(kap_h, kap_h * np.finfo(float).eps, "k:", label=r"$\kappa(H)\,\varepsilon$")
ax2.axhline(1, color="tab:red", ls="--", lw=1, label="relative error 1 (no correct digit)")
ax2.set_xlabel(r"condition number $\kappa(H)$"); ax2.set_ylabel("relative error of the solution")
ax2.set_title("Error: how far is x from the true solution (all ones)?", fontsize=10)
for ax in (ax1, ax2):
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
plt.suptitle("Hilbert matrices of order 4 to 12: same problem, two algorithms")
plt.show()
i10 = list(orders_h).index(10)
print("order 10: kappa = %.1e, relative residual: solve %.0e, inv(H) @ b %.0e" % (kap_h[i10], res_s[i10], res_i[i10]))
""")

nb.md(r"""
**Notice that** the residual of `solve` stays at rounding level (about $10^{-16}$) however
ill-conditioned the Hilbert matrix becomes, while `inv(H) @ b` misses the target by about $6\times10^{-5}$
at order 10 and by roughly $9\%$ at order 12 (the exact values depend on the LAPACK build, the pattern
does not). Neither error rises much above $\kappa\,\varepsilon$ (the dotted line): `inv(H) @ b` hovers
around it, `solve` sits well below it. Conditioning is a property of the *problem*: the rounding already
present in the stored entries of $H$ is amplified by $\kappa$ whatever algorithm follows. But only
`solve` hands back an $\mathbf{x}$ that solves the system you actually wrote down: its answer is the
exact solution of a system whose matrix differs from $H$ only in the last digits (*backward
stability*), and `inv(H) @ b` offers no such guarantee.

**Try it yourself:** change `delta` in the sensitivity cell to `1e-4` (noise in the fourth digit) and read
off the largest $\kappa$ for which the solution still has two correct digits. Then, back with the
matrix `X_bad` of Section 5, solve `X_bad.T @ X_bad` against `X_bad.T @ y` for
`y = 50 + 10 * np.random.default_rng(0).standard_normal(1000)` (its own generator, so the lesson's `rng`
stays untouched), once as it is and once with the right-hand side multiplied by `(1 + 1e-10)`. You will
probably see no error at all — the Gram matrix is singular only up to rounding (its condition number is
of order $10^{17}$ or more), which LAPACK does not necessarily notice, so do not count on an exception —
yet the intercept moves by several units (about $-8.3$ in the reference run; the exact amount depends on
rounding and on the LAPACK build) while each of the three dummy coefficients moves by the same amount in
the opposite direction, `weight_kg` and `weight_lb` move in the ratio $1 : -0.4536$, and the predictions
`X_bad @ w` change by only about $10^{-10}$ in relative terms: the two null vectors above at work.
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
**Notice that** all three routes give the same coefficients — they solve the same problem — but
$\kappa(A^\top A) = \kappa(A)^2$: forming the Gram matrix squares the condition number, which is
why `lstsq` and QR are preferred whenever $\kappa(A)$ is large (roughly $10^4$ or more), however many
columns there are. The hat matrix passes both projection tests, its trace is exactly 2 (the number of
parameters), and $A^\top\mathbf{r} = \mathbf{0}$ to rounding error: the residual vector really is
perpendicular to the column space. "Linear" least squares means linear *in the coefficients*: the same
machinery fits a quadratic once we add a $t^2$ column.
""")

nb.code(r"""
truth = lambda s: 1.0 + 0.3 * s - 0.05 * s ** 2                               # the curve behind the data
y2 = truth(t) + rng.normal(0, 0.3, t.size)                                    # curved truth plus noise
y_new = truth(t) + np.random.default_rng(99).normal(0, 0.3, (2000, t.size))   # 2000 fresh noise draws at the same t (own generator)
designs = {"line": np.column_stack([np.ones_like(t), t]),
           "quadratic": np.column_stack([np.ones_like(t), t, t ** 2]),
           "degree-6 polynomial": np.column_stack([t ** k for k in range(7)])}
tt = np.linspace(0, 10, 200)
print(f"{'':20s}  params     RSS  mean RSS, 2000 fresh draws  RMS gap to true curve    kappa(A)")
plt.scatter(t, y2, s=18, color="k", label="data")
fresh = {}                                                                    # per model: the RSS on each of the fresh draws
for name, A_d in designs.items():
    coef = np.linalg.lstsq(A_d, y2, rcond=None)[0]
    A_tt = np.column_stack([tt ** k for k in range(A_d.shape[1])])
    rss = np.sum((y2 - A_d @ coef) ** 2)
    fresh[name] = np.sum((y_new - A_d @ coef) ** 2, axis=1)              # the same fitted curve, judged on each fresh draw
    rss_new = fresh[name].mean()                                         # mean RSS of the fitted curve on the fresh draws
    gap = np.sqrt(np.mean((A_tt @ coef - truth(tt)) ** 2))               # RMS distance from the true curve on [0, 10]
    print(f"{name:20s}  {A_d.shape[1]:6d}  {rss:6.3f}  {rss_new:26.3f}  {gap:21.3f}  {np.linalg.cond(A_d):10.1e}")
    plt.plot(tt, A_tt @ coef, label=f"{name} (RSS {rss:.2f})")
print("on %.0f%% of the 2000 fresh draws the degree-6 curve has a larger RSS than the quadratic" % (100 * np.mean(fresh["degree-6 polynomial"] > fresh["quadratic"])))
plt.plot(tt, truth(tt), "k--", lw=1, label="true curve")
plt.ylim(y2.min() - 1, y2.max() + 1)
plt.xlabel("t"); plt.ylabel("y"); plt.title("Polynomial fits are still linear least squares")
plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

nb.md(r"""
**Notice that** the residual sum of squares can only fall as columns are added — the column space
grows, so the projection gets closer — but the degree-6 fit buys very little over the quadratic
(RSS $2.21 \to 1.70$) while its design matrix has a condition number of about $6\times10^6$, some
$46{,}000$ times that of the quadratic ($140$). The dashed true curve shows what the extra columns
really bought: the degree-6 curve bends to chase the noise and is on average about twice as far from
the truth as the quadratic (RMS gap $0.15$ against $0.08$), and on fresh noise at the same $t$ values
(averaged over 2000 draws) it fits *worse* than the quadratic (RSS $3.392$ against $2.875$, as in the
table; worse on $89\%$ of the draws, as the last printed line says). More columns means a better fit to
*these* points, not a better model; Module 13 turns that tension into the bias–variance trade-off,
and Module 03's SVD explains the exploding condition number.
""")

nb.code(r"""
# The same exact problem, three solvers: forming A^T A makes the normal equations lose digits much faster than QR or lstsq
eps_mach = np.finfo(float).eps
kap_, err_ = [], {"normal equations": [], "QR": [], "lstsq (SVD)": []}
for deg in range(1, 10):
    A_p = np.column_stack([t ** k for k in range(deg + 1)])   # monomial design on the lesson's t in [0, 10]
    c_true = np.ones(deg + 1)
    y_p = A_p @ c_true                                        # noise-free data, so the exact least-squares answer is c_true
    Q_p, R_p = np.linalg.qr(A_p)
    sols = {"normal equations": np.linalg.solve(A_p.T @ A_p, A_p.T @ y_p),
            "QR": np.linalg.solve(R_p, Q_p.T @ y_p),
            "lstsq (SVD)": np.linalg.lstsq(A_p, y_p, rcond=None)[0]}
    kap_.append(np.linalg.cond(A_p))
    for k_, v_ in sols.items():
        err_[k_].append(np.linalg.norm(v_ - c_true) / np.linalg.norm(c_true))
kap_ = np.array(kap_)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for (k_, e_), mk in zip(err_.items(), ["s-", "o-", "^-"]):
    ax.loglog(kap_, e_, mk, alpha=0.8, label=k_)
ax.loglog(kap_, kap_ * eps_mach, "k:", label=r"$\kappa(A)\,\varepsilon$")
ax.loglog(kap_, kap_ ** 2 * eps_mach, "k-.", label=r"$\kappa(A)^2\,\varepsilon$")
ax.axhline(1, color="tab:red", ls="--", lw=1, label="relative error 1 (no correct digit)")
for deg in (2, 4, 6, 8):
    ax.annotate(f"degree {deg}", (kap_[deg - 1], err_["QR"][deg - 1]), textcoords="offset points", xytext=(0, -14), fontsize=7, ha="center",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
ax.set_ylim(1e-17, 1e2)
ax.set_xlabel(r"condition number $\kappa(A)$ of the degree-$d$ monomial design"); ax.set_ylabel("relative error of the coefficients")
ax.set_title(r"Exact data, three solvers: forming $A^\top A$ squares the damage")
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3, which="both"); plt.show()
for deg in (6, 8):
    print("degree %d: kappa = %.1e, correct digits  normal equations %.1f   QR %.1f   lstsq %.1f"
          % (deg, kap_[deg - 1], *(-np.log10(err_[k_][deg - 1]) for k_ in err_)))
""")

nb.md(r"""
**Notice that** QR and `lstsq` stay close to the line $\kappa\,\varepsilon$ while the normal equations
lose digits much faster, between $\kappa\,\varepsilon$ and $\kappa^2\varepsilon$: on the degree-6
design of the previous cell ($\kappa = 6\times10^6$) they return about 6 correct digits against 10, and
at degree 8 ($\kappa = 2\times10^9$) only 2 against 7 or 8. The data and the exact answer are
identical — forming $A^\top A$ is what squares the damage, while QR and `lstsq` work on $A$ itself.

**Try it yourself:** increase the polynomial degree of the fitting cell to 12 and look at both
`kappa(A)` and the curve between the data points.
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
print("P^-1 A P =\n", np.round(A_new, 12) + 0.0)            # (+ 0.0 turns a printed -0. into 0.)
print("invariants  trace: %.1f vs %.1f   det: %.1f vs %.1f" % (np.trace(A), np.trace(A_new), np.linalg.det(A), np.linalg.det(A_new)))

# picture: the unit circle and its image, with the new basis directions
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])
ellipse = A @ circle
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(circle[0], circle[1], "k--", lw=1, label="unit circle")
ax.plot(ellipse[0], ellipse[1], "tab:blue", lw=2, label="image under A")
for i, (col, nm) in enumerate(zip(["tab:green", "tab:orange"], ["p1", "p2"])):
    p = P[:, i] / np.linalg.norm(P[:, i])                       # unit vector along the i-th new basis direction
    ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=A @ p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, alpha=0.45, shrinkA=0, shrinkB=0))
    ax.plot([], [], color=col, lw=2.5, label=f"{nm} (unit length)")
    ax.plot([], [], color=col, lw=1.5, alpha=0.45, label=f"A {nm} = {np.linalg.norm(A @ p):.0f} x {nm}")
ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("A stretches along p1 by 3 and along p2 by 1")
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.show()
""")

nb.md(r"""
**Notice that** the circle becomes an ellipse whose axes are exactly the new basis directions: along
$\mathbf{p}_1$ the arrow triples (solid green $\to$ faint green), along $\mathbf{p}_2$ it is unchanged
(solid and faint orange coincide), and $P^{-1}AP$ records nothing but those two numbers. Trace and
determinant are the same in both coordinate systems — they belong to the map, not to the matrix.
Finding the basis that makes $A$ diagonal is the eigenvalue problem, and doing it for a covariance
matrix is PCA (Module 03).
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
plt.plot([], [], color="gray", ls=":", label="stationary values (%.2f, %.2f, %.2f)" % tuple(pi_eig))   # legend entry for the dotted lines
plt.xlabel("month"); plt.ylabel("probability"); plt.title("Distribution converges to the stationary distribution (dotted)")
plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

nb.md(r"""
**Notice that** after about 15 steps the three curves have flattened onto the dotted stationary
values (two of them coincide at $0.25$), and that the $200$-th power of $P$ has that same vector as
its row — every starting state leads to the same long-run mix. The gap closes geometrically at the
rate of the second-largest eigenvalue modulus, which is the quantity PageRank and MCMC practitioners
care about (Modules 03 and 18); the next figure shows it for all three starting states. The $A^2$
entries are plain path counts: matrix multiplication is doing combinatorics.
""")

nb.code(r"""
# Three starts, one destination: trajectories in the probability simplex, and the distance to pi on a log axis
lam2 = np.sort(np.abs(evals))[::-1][1]                       # second-largest eigenvalue modulus of Pm
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
gaps_ = []
for j, (name_, col_) in enumerate(zip(["Active", "Lapsed", "Churned"], ["tab:blue", "tab:orange", "tab:green"])):
    traj_ = [np.eye(3)[j]]                                   # everyone starts in state j
    for _ in range(25):
        traj_.append(traj_[-1] @ Pm)
    traj_ = np.array(traj_)
    gaps_.append(np.abs(traj_ - pi_eig).sum(axis=1))         # L1 distance to the stationary distribution
    ax1.plot(traj_[:, 1], traj_[:, 2], "o-", ms=3, color=col_, label=f"start: everyone {name_}")
    ax1.plot(traj_[0, 1], traj_[0, 2], "s", ms=9, color=col_)
    ax2.semilogy(gaps_[-1], "o-", ms=3, color=col_, label=f"start: everyone {name_}")
ax1.plot([0, 1], [1, 0], "k--", lw=1, label="edge P(Active) = 0")   # the hypotenuse of the simplex
ax1.plot(pi_eig[1], pi_eig[2], "r*", ms=16, label=r"stationary $\pi$")
ax1.set_xlabel("P(Lapsed)"); ax1.set_ylabel("P(Churned)")
ax1.set_title("Distributions move through the simplex\n(P(Active) = 1 - x - y; squares = starting points)", fontsize=10)
ax1.set_aspect("equal"); ax1.legend(fontsize=7, loc="upper right"); ax1.grid(alpha=0.3)
months_ = np.arange(26)
ax2.semilogy(months_, 2 * max(g_[20] / lam2 ** 20 for g_ in gaps_) * lam2 ** months_, "k--", lw=1.5,
             label=rf"reference slope $|\lambda_2|^t$ with $|\lambda_2|$ = {lam2:.3f}")
ax2.set_xlabel("month t"); ax2.set_ylabel(r"distance $\sum_j |\pi_t(j) - \pi(j)|$")
ax2.set_title("Distance to the stationary distribution", fontsize=10)
ax2.legend(fontsize=7); ax2.grid(alpha=0.3, which="both")
plt.show()
print("each month the distance shrinks by a factor of about %.3f, i.e. it halves every %.2f months" % (lam2, np.log(2) / -np.log(lam2)))
""")

nb.md(r"""
**Notice that** all three starting points — the corners of the simplex, "everyone Active", "everyone
Lapsed", "everyone Churned" — flow into the same point, the red star: the distribution
$(0.5, 0.25, 0.25)$, which sits at $(0.25, 0.25)$ in these coordinates. On the log axis the three
distances become parallel straight lines with the slope of $|\lambda_2|^t$: every month the distance
to $\boldsymbol\pi$ shrinks by a factor of about $0.673$, so it halves roughly every $1.75$ months.
The starting point only changes the height of the line, never its slope.
""")

# ---------------------------------------------------------------- summary
nb.md(r"""
## Summary

* $A\mathbf{x}$ is simultaneously a set of dot products (rows), a combination of columns, a linear
  function and a system of equations; the column picture tells you what outputs are reachable.
* `A @ B` composes maps right-to-left, is not commutative (the order changes the picture, not the
  determinant), costs $O(n^3)$, and collapses stacked linear layers into one.
* Rotation, scaling, shear, reflection and projection are $2\times2$ matrices; the determinant is the
  signed area factor, and $\det = 0$ means a dimension was destroyed.
* Orthogonal matrices preserve lengths; Gram matrices $A^\top A$ are symmetric PSD; triangular
  systems solve in $O(n^2)$.
* Rank counts independent columns; duplicated or one-hot-trapped features give a non-trivial null
  space — a flat valley of the loss — and unidentifiable coefficients.
* Use `np.linalg.solve`, never `inv(A) @ b`; the condition number bounds how many digits survive
  whatever the algorithm, but `solve` keeps the residual at rounding level where `inv` does not.
* Least squares = projection onto the column space; the normal equations, QR and `lstsq` agree on
  well-conditioned problems, but QR/SVD avoid squaring $\kappa$ and keep many more digits when it is
  large; the hat matrix is a symmetric idempotent projector.
* $P^{-1}AP$ is the same map in new coordinates — diagonal when $P$ holds the eigenvectors.
* Powers of adjacency matrices count walks; powers of transition matrices converge to the
  stationary distribution, at the rate $|\lambda_2|^t$.
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
    compiled BLAS code with cache blocking and SIMD, which is why, at the sizes timed in the lesson
    ($n = 30$ to $120$), it is hundreds to tens of thousands of times faster, but not asymptotically
    cheaper. The swap matrix shows that order matters: multiplying on the right rearranges columns,
    on the left rearranges rows. A typical slip is indexing `B[j, k]` instead of `B[k, j]`, which
    computes $AB^\top$ — the rectangular test catches it.
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
M5 = np.array([[1., 2., 3., 4.],
               [2., 4., 6., 8.],
               [1., 0., 1., 0.]])
_rank5 = int(np.linalg.matrix_rank(M5))
HASH_5_RANK = sha_of(_rank5)                      # 2
HASH_5_NULLITY = sha_of(M5.shape[1] - _rank5)     # 2  (rank and nullity coincide for this M ...)
HASH_5_LEFT = sha_of(M5.shape[0] - _rank5)        # 1  (... which is why the left null space is asked for as well)
assert _rank5 == 2 and HASH_5_LEFT != HASH_5_NULLITY
ex.add(
    title="Rank and null space of a given matrix", difficulty=2,
    prompt_md=r"""
    Consider

    $$
    M = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 6 & 8 \\ 1 & 0 & 1 & 0 \end{pmatrix}.
    $$

    1. Work out **by hand** which columns are combinations of which others, and set `rank_M` (an
       `int`), `nullity_M` (an `int`, the dimension of the null space $\mathcal{N}(M)$) and
       `left_nullity_M` (an `int`, the dimension of the null space of $M^\top$: how many independent
       combinations of the *rows* add up to the zero row). Only then confirm with
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
    left_nullity_M = None   # YOUR ANSWER HERE (int): dimension of the null space of M^T
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
    nullity_M = 4 - rank_M          # rank-nullity: rank + nullity = number of columns (4)
    left_nullity_M = 3 - rank_M     # rank + left nullity = number of rows (3): one relation among the rows
    null_vec = np.array([1., 1., -1., 0.])      # c1 + c2 - c3 = 0  (another choice: (0, 2, 0, -1))
    b_solvable = True       # (3, 6, 1) is column 3 itself, so x = (0, 0, 1, 0) works
    b2_solvable = False     # row 2 of M is twice row 1, so any M x has second entry = 2 * first entry; 1 != 2
    print("matrix_rank:", np.linalg.matrix_rank(M), "  M @ null_vec =", M @ null_vec)
    print("row relation 2*row1 - row2 - 0*row3 =", np.array([2., -1., 0.]) @ M, "  (a null vector of M^T)")
    """,
    check=rf"""
    check_hash('hidden: rank of M', rank_M, '{HASH_5_RANK}')
    check_hash('hidden: nullity of M', nullity_M, '{HASH_5_NULLITY}')
    check_hash('hidden: dimension of the null space of M^T', left_nullity_M, '{HASH_5_LEFT}')
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
    independent: rank 2, and by rank–nullity the null space has dimension $4 - 2 = 2$. (Rank and
    nullity happen to coincide for this $M$; in general rank plus nullity is the number of *columns*, 4,
    while rank plus the dimension of $\mathcal{N}(M^\top)$ is the number of *rows*, 3 — so the left null
    space here has dimension $3 - 2 = 1$, spanned by the row relation $2\,\mathbf{r}_1 - \mathbf{r}_2 = \mathbf{0}$,
    i.e. the vector $(2, -1, 0)$.) Each column relation is a null vector — $\mathbf{c}_1 + \mathbf{c}_2 - \mathbf{c}_3 = \mathbf{0}$ gives
    $(1, 1, -1, 0)$ and $2\mathbf{c}_2 - \mathbf{c}_4 = \mathbf{0}$ gives $(0, 2, 0, -1)$; together they
    span the null space. Solvability is membership in the column space: $(3, 6, 1)$ *is* a column,
    while $(1, 1, 1)$ violates the constraint that every output has second entry equal to twice its
    first (row 2 is twice row 1). Looking only at "three non-zero rows, so rank 3" is the classic
    mistake — rank counts *independent* rows, and here there are two.
    """,
)

# ---------------------------------------------------------------- 6: solve 3x3
E6 = np.array([[2., 1., -1.],
               [-3., -1., 2.],
               [-2., 1., 2.]])
b6 = np.array([8., -11., -3.])
_x6 = np.linalg.solve(E6, b6)            # (2, 3, -1)
HASH_6_X = safe_sha(_x6)
HASH_6_DET = safe_sha(float(np.linalg.det(E6)))   # -1
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
    Set `A6` and `b6` to the coefficient matrix and right-hand side, compute `x6 = np.linalg.solve(A6, b6)`
    and `det6 = np.linalg.det(A6)`. **Do not** use `np.linalg.inv`. Finally set `n_solutions` to the
    string `'one'`, `'none'` or `'infinite'` describing the solution set (which the determinant tells you).
    """,
    starter=r"""
    A6 = None            # YOUR CODE HERE: 3x3 coefficient matrix
    b6 = None            # YOUR CODE HERE: right-hand side
    x6 = None            # YOUR CODE HERE: np.linalg.solve(...)
    det6 = None          # YOUR CODE HERE
    n_solutions = None   # 'one', 'none' or 'infinite'
    """,
    solution=r"""
    A6 = np.array([[2., 1., -1.],
                   [-3., -1., 2.],
                   [-2., 1., 2.]])
    b6 = np.array([8., -11., -3.])
    x6 = np.linalg.solve(A6, b6)
    det6 = np.linalg.det(A6)
    n_solutions = 'one'          # det != 0  ->  A6 is invertible  ->  exactly one solution
    print("x =", x6, "  det =", round(det6, 6), "  residual:", np.linalg.norm(A6 @ x6 - b6))
    """,
    check=rf"""
    try:
        resid = float(np.linalg.norm(np.asarray(A6, float) @ np.asarray(x6, float) - np.asarray(b6, float)))
    except Exception:
        resid = None
    check('A6 @ x6 reproduces b6 (residual < 1e-9)', resid is not None and resid < 1e-9)
    check_hash('hidden: the solution vector (x, y, z)', (np.round(np.asarray(x6, float), 6) if x6 is not None else None), '{HASH_6_X}')
    check_hash('hidden: det(A6)', (float(det6) if det6 is not None else None), '{HASH_6_DET}')
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

# ---------------------------------------------------------------- 7: least squares
SOL_7 = r"""
def lstsq_normal(A, y):
    # normal equations  (A^T A) x = A^T y, solved with a factorisation (no inverse)
    return np.linalg.solve(A.T @ A, A.T @ y)
"""
_ns7 = run_solution(SOL_7)
A7 = np.column_stack([np.ones(4), np.arange(4.)])
y7 = np.array([1., 2., 2., 4.])
HASH_7_COEF = safe_sha(_ns7["lstsq_normal"](A7, y7))        # (0.9, 0.9)
_r7 = y7 - A7 @ _ns7["lstsq_normal"](A7, y7)
HASH_7_RSS = safe_sha(float(_r7 @ _r7))                      # 0.7
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
    solution=dd(SOL_7, r"""
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
    check_hash('hidden: (intercept, slope) of the 4-point line', (np.round(np.asarray(coef_line, float), 6) if coef_line is not None else None), '{HASH_7_COEF}')
    check_hash('hidden: residual sum of squares of the 4-point line', (float(rss_line) if rss_line is not None else None), '{HASH_7_RSS}')
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
       idempotent ($H^2 = H$) to within `tol`. Both conditions matter: $\begin{pmatrix}1&1\\0&0\end{pmatrix}$
       is idempotent but not symmetric (an *oblique* projection) and $\mathrm{diag}(1, 2)$ is symmetric but
       not idempotent.
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
    try:
        oblique_verdict = is_projection(np.array([[1., 1.], [0., 0.]]))     # idempotent, but not symmetric
    except Exception:
        oblique_verdict = None
    try:
        diag_verdict = is_projection(np.diag([1., 2.]))                     # symmetric, but not idempotent
    except Exception:
        diag_verdict = None
    check('is_projection rejects an idempotent but non-symmetric matrix', ok_proj and oblique_verdict is not None and not bool(oblique_verdict))
    check('is_projection rejects a symmetric but non-idempotent matrix', ok_proj and diag_verdict is not None and not bool(diag_verdict))
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
    and is not symmetric, so `is_projection` must reject it. The two other test matrices show that
    both conditions are needed: $\begin{pmatrix}1&1\\0&0\end{pmatrix}$ squares to itself but is not
    symmetric (it projects *obliquely*, not orthogonally), and $\mathrm{diag}(1, 2)$ is symmetric but
    squares to $\mathrm{diag}(1, 4)$ — either test alone would give a wrong verdict.
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
       the two columns). Two separate effects appear: the standard errors of the two coefficients are
       inflated by $\sqrt{\mathrm{VIF}} = 1/\sqrt{1 - 0.9999^2} \approx 71$, a statistical statement about
       how little the data can say about each coefficient on its own; and $\kappa(X^\top X) = \kappa(X)^2$
       means that forming $X^\top X$ squares the numerical difficulty, a statement about lost digits.
       Remedies: drop or merge one of the two
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
    eigen-solver: $(P^\top - I)\boldsymbol\pi^\top = \mathbf{0}$ has rank $n - 1$ (its *columns* sum to zero because
    the rows of $P$ sum to one, so its rows add up to the zero vector), so one equation is redundant and can be swapped for
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
    * Adding `size_sqft` leaves the rank at 4 although there are now 5 columns and makes $\kappa$ explode; you note that `np.linalg.solve` on the normal equations may raise, return enormous numbers or return a plausible-looking vector whose split between the two size columns is arbitrary, and you explain this as an exact collinearity (a null-space direction) rather than a numerical accident.
    * The predicted-vs-actual plot is labelled and includes the $y = x$ reference line.
    """,
    starter=dd(PROJECT_DATA, r"""

# Your mini-project workspace — follow steps 1 to 7 of the prompt.
"""),
    solution_sketch_md=r"""
    Stack `[1, size_m2, rooms, age_years]` into $X$ (200 × 4). All three solvers give
    $\hat{\mathbf{w}} \approx (53.1,\ 2.53,\ 13.4,\ -1.24)$: about 2.5 kEUR per extra square metre,
    13 kEUR per additional room and a loss of 1.2 kEUR per year of age, close to the generating
    values (every estimate is within about one standard error of the truth). The intercept is the
    least precise (standard error about 7 kEUR, because it extrapolates to a flat of size 0 and age 0;
    centring the features cuts it to about 2), and `rooms` and `size_m2` are each about 1.6 times less
    precise than they would be if the features were uncorrelated, because $\rho \approx 0.77$
    (variance inflation factor about 2.5). $\kappa(X) \approx 400$ and
    $\kappa(X^\top X) \approx 1.6\times10^5$ — harmless here, but it shows why the normal equations
    square the conditioning. The hat matrix gives the fitted values, $X^\top \mathbf{r}$ is zero to
    rounding, the residual standard deviation is about 26 (close to the true noise level 25) and
    $R^2 \approx 0.94$. Appending the square-foot column makes column 5 exactly $10.764\times$ column 2:
    the rank stays 4 while the matrix has 5 columns and $\kappa$ jumps to about $10^{16}$. What
    `np.linalg.solve` does with the normal equations then depends on rounding and on the LAPACK
    build: it may raise `LinAlgError`, return enormous numbers or — as in the reference run — return a
    perfectly ordinary-looking vector. The system is consistent, so the 'solution' is one arbitrary
    member of the family of exact fits: re-solve with $\mathbf{y}$ scaled by $1 + 10^{-14}$ and the split
    between `size_m2` and `size_sqft` changes, while $\text{coef}_{m^2} + 10.764\,\text{coef}_{ft^2} \approx 2.53$
    does not. `lstsq` returns the minimum-norm member of that family. None of them is a meaningful
    'effect of one more square metre versus one more square foot': the null-space direction
    $(10.764, -1)$ (in the coefficients of `size_m2` and `size_sqft`) can be added freely.
    scikit-learn's `LinearRegression` agrees with the full-rank fit to rounding error, because it
    calls `scipy.linalg.lstsq` under the hood.
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
print("\ntrace(H) = %.3f,  X^T r = %s" % (np.trace(H), np.round(X.T @ r, 8) + 0.0))
print("RSS = %.1f, residual sd = %.2f (noise sd was 25), R^2 = %.3f" % (rss, sigma_hat, r2))

# 5. scikit-learn comparison
lr = LinearRegression().fit(X[:, 1:], y)
print("\nsklearn intercept %.4f coef %s" % (lr.intercept_, np.round(lr.coef_, 4)))
print("matches our solver:", np.allclose(np.r_[lr.intercept_, lr.coef_], w_normal))

# 6. an exactly collinear extra column
X5 = np.column_stack([X, size_m2 * 10.764])
print("\nwith size_sqft: shape", X5.shape, " rank", np.linalg.matrix_rank(X5), " kappa(X5) = %.2e" % np.linalg.cond(X5))
G5 = X5.T @ X5
print("cond(X5^T X5) = %.1e  -- singular up to rounding" % np.linalg.cond(G5))
try:
    w5 = np.linalg.solve(G5, X5.T @ y)
    print("np.linalg.solve raised no exception and returned", np.round(w5, 3))
    w5b = np.linalg.solve(G5, X5.T @ (y * (1 + 1e-14)))
    print("  y scaled by 1 + 1e-14 gives          ", np.round(w5b, 3), " <- the size_m2 / size_sqft split is arbitrary")
    print("  but coef_m2 + 10.764 * coef_sqft is determined: %.4f and %.4f" % (w5[1] + 10.764 * w5[4], w5b[1] + 10.764 * w5b[4]))
except np.linalg.LinAlgError as e:
    print("np.linalg.solve raised LinAlgError:", e)
w5_lstsq = np.linalg.lstsq(X5, y, rcond=None)[0]
print("lstsq minimum-norm coefficients:", np.round(w5_lstsq, 4))
print("only the combination is meaningful: coef_m2 + 10.764 * coef_sqft =", round(w5_lstsq[1] + 10.764 * w5_lstsq[4], 4), "(the full-rank fit gave %.4f)" % w_normal[1])

# 7. plot
plt.figure(figsize=(5.5, 5.5))
plt.scatter(y, y_hat, s=14)
lim = (min(y.min(), y_hat.min()) - 10, max(y.max(), y_hat.max()) + 10)       # the same limits on both axes ...
plt.plot(lim, lim, "k--", lw=1, label="y = x")
plt.xlim(lim); plt.ylim(lim); plt.gca().set_aspect("equal", adjustable="box")   # ... so y = x is a true 45-degree line
plt.xlabel("actual price (kEUR)"); plt.ylabel("predicted price (kEUR)")
plt.title(f"OLS on synthetic apartments: R$^2$ = {r2:.3f}, residual sd = {sigma_hat:.1f} kEUR")
plt.legend(); plt.grid(alpha=0.3); plt.show()
"""),
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
print("wrote exercises.ipynb and solutions.ipynb with", len(ex.items), "exercises")
