"""Builder for Module 01 — Vectors and Geometry.

Generates lesson.ipynb, exercises.ipynb and solutions.ipynb into
part-1-linear-algebra/01-vectors-and-geometry.  Hidden answers are computed by
*executing* the reference solutions below, never typed by hand.
"""
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import Notebook, ExerciseSet, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-1-linear-algebra/01-vectors-and-geometry"
MOD.mkdir(parents=True, exist_ok=True)


def safe_round(x, decimals=3):
    """Assert that x is not within 1e-6 of a rounding boundary at `decimals`."""
    frac = abs(x * 10 ** decimals - math.floor(x * 10 ** decimals) - 0.5)
    assert frac > 1e-6, f"value {x!r} sits on a rounding boundary at {decimals} decimals"
    return x


# ============================================================================
# LESSON NOTEBOOK
# ============================================================================
nb = Notebook("Module 01 — Vectors and Geometry")

nb.md(r"""
This notebook is the hands-on companion to the concept guide [README.md](README.md) — read the guide first, or keep it open alongside. Every object is built **from scratch in NumPy** (norms, dot products, projections, Gram–Schmidt, hyperplanes), the hand-worked numbers from the guide are re-computed, and the last section runs the simulations that make high-dimensional geometry believable.

The eight sections mirror the guide: vectors → norms → dot product → cosine similarity → projections → orthonormal bases → hyperplanes → high dimensions.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
""")

# ---------------------------------------------------------------- Section 1
nb.md(r"""
## 1. Vectors as arrows, points and data records

A vector $\mathbf{v} \in \mathbb{R}^n$ is an ordered list of $n$ numbers. The same object is an **arrow** (a displacement), a **point** (the arrow's tip) and a **data record** (one row of a dataset). Two operations define everything:

$$\mathbf{u} + \mathbf{v} = (u_1 + v_1, \dots, u_n + v_n), \qquad c\,\mathbf{v} = (c v_1, \dots, c v_n).$$

Below, one and the same `np.array` plays all three roles: first as arrows in the plane, then as a patient record whose year-on-year *change* is a vector subtraction.
""")

nb.code(r"""
u = np.array([1., 2.])
v = np.array([3., -1.])
print("u + v  =", u + v)
print("2u     =", 2 * u)
print("2u - v =", 2 * u - v)

# the same arithmetic on a data record: one patient visit = (age, systolic BP, cholesterol, BMI)
p_2024 = np.array([54., 130., 210., 27.5])
p_2025 = np.array([55., 142., 198., 28.1])
print("change 2024 -> 2025:", p_2025 - p_2024, "  (BP up, cholesterol down)")

# draw u, v, u + v and 2u as arrows from the origin; the faint arrow shows v placed tip-to-tail after u
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for vec, col, lab in [(2 * u, "C3", "2u"), (v, "C1", "v"), (u + v, "C2", "u + v"), (u, "C0", "u")]:   # u drawn last, on top of 2u
    ax.quiver(0, 0, vec[0], vec[1], angles="xy", scale_units="xy", scale=1, color=col, width=0.012)
    ax.annotate(lab, vec, textcoords="offset points", xytext=(6, 4), color=col, fontsize=11)
ax.quiver(u[0], u[1], v[0], v[1], angles="xy", scale_units="xy", scale=1, color="C1", alpha=0.35, width=0.008)
ax.set_xlim(-1, 5); ax.set_ylim(-2, 5); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Vector addition and scaling in the plane")
plt.show()
""")

nb.md(r"""
**Notice that** $\mathbf{u} + \mathbf{v}$ is where you land if you walk along $\mathbf{u}$ and then along $\mathbf{v}$ (the faint arrow): tip-to-tail addition and component-wise addition are the same thing. Scaling by 2 keeps the direction and doubles the length.

**Linear independence.** Vectors are independent when no non-trivial combination gives $\mathbf{0}$ — equivalently, when the matrix with those vectors as columns has full rank. Writing a third vector $\mathbf{w}$ in terms of $\mathbf{u}$ and $\mathbf{v}$ means solving two equations in two unknowns. The guide did this by hand for $\mathbf{w} = (5, 0)$ and found $5/7$ and $10/7$.
""")

nb.code(r"""
M = np.column_stack([u, v])                       # 2x2 matrix with u and v as columns
rank = np.linalg.matrix_rank(M)
print("rank of [u v] =", rank, "->", "independent" if rank == 2 else "dependent")

w = np.array([5., 0.])
coeffs = np.linalg.solve(M, w)                    # a*u + b*v = w
print("w = %.4f * u + %.4f * v    (by hand: 5/7 = %.4f, 10/7 = %.4f)" % (coeffs[0], coeffs[1], 5/7, 10/7))
print("check a*u + b*v =", coeffs[0] * u + coeffs[1] * v)

# any three vectors in R^2 are dependent: the rank cannot exceed 2
print("rank of [u v w] =", np.linalg.matrix_rank(np.column_stack([u, v, w])))
""")

nb.md(r"""
> **Try it yourself:** replace `v` by `np.array([2., 4.])` (a multiple of `u`) and re-run the two cells above. The rank drops to 1 and `np.linalg.solve` raises `LinAlgError: Singular matrix` — the two arrows lie on one line, so they cannot reach $(5, 0)$.
""")

# ---------------------------------------------------------------- Section 2
nb.md(r"""
## 2. Norms and distance

A norm is a ruler. The three you will use constantly:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}, \qquad \|\mathbf{x}\|_1 = \sum_i |x_i|, \qquad \|\mathbf{x}\|_\infty = \max_i |x_i|,$$

and in general $\|\mathbf{x}\|_p = \big(\sum_i |x_i|^p\big)^{1/p}$. A **unit vector** has norm 1; **normalising** $\mathbf{x}$ means dividing by its norm. The **distance** between two points is the norm of their difference. For $\mathbf{x} = (3, -4, 12)$ the guide found $\|\mathbf{x}\|_2 = 13$, $\|\mathbf{x}\|_1 = 19$ and $\|\mathbf{x}\|_\infty = 12$ — let us compute the $L_2$ norm three ways and confirm the ordering $\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt{n}\,\|\mathbf{x}\|_2$.
""")

nb.code(r"""
x = np.array([3., -4., 12.])

l2_by_hand = np.sqrt(np.sum(x ** 2))     # the definition
l2_dot     = np.sqrt(x @ x)              # via the dot product: ||x||^2 = x . x
l2_numpy   = np.linalg.norm(x)           # the library call
print("L2 three ways      :", l2_by_hand, l2_dot, l2_numpy)

l1, l1_np     = np.sum(np.abs(x)), np.linalg.norm(x, 1)
linf, linf_np = np.max(np.abs(x)), np.linalg.norm(x, np.inf)
print("L1   (ours, numpy) :", l1, l1_np)
print("Linf (ours, numpy) :", linf, linf_np)
print("Linf <= L2 <= L1 <= sqrt(n) L2 :", linf <= l2_dot <= l1 <= np.sqrt(len(x)) * l2_dot,
      "  (sqrt(3) * 13 = %.2f)" % (np.sqrt(3) * 13))

x_hat = x / l2_dot
print("unit vector x_hat  :", x_hat, "  norm:", np.linalg.norm(x_hat))

a, b = np.array([1., 2., 3.]), np.array([4., 6., 3.])
print("distance a <-> b   : L2 = %.3f   L1 = %.3f   Linf = %.3f"
      % (np.linalg.norm(a - b), np.linalg.norm(a - b, 1), np.linalg.norm(a - b, np.inf)))
""")

nb.md(r"""
The *unit ball* $\{\mathbf{x} : \|\mathbf{x}\|_p \le 1\}$ reveals each norm's personality. We trace its boundary by taking points on the ordinary circle and rescaling each one to have $L_p$ norm exactly 1.
""")

nb.code(r"""
def lp_unit_circle(p, n=721):
    # points on the L2 circle, each rescaled so that its Lp norm equals 1
    t = np.linspace(0, 2 * np.pi, n)
    pts = np.column_stack([np.cos(t), np.sin(t)])
    norms_p = (np.abs(pts) ** p).sum(axis=1) ** (1 / p)
    return pts / norms_p[:, None]

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for p, lab in [(1, "p = 1  (diamond)"), (2, "p = 2  (circle)"), (4, "p = 4")]:
    c = lp_unit_circle(p)
    ax.plot(c[:, 0], c[:, 1], label=lab)
square = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]], dtype=float)
ax.plot(square[:, 0], square[:, 1], label="p = inf  (square)")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Unit balls of the $L_p$ norms")
plt.show()
""")

nb.md(r"""
**Notice that** the $L_1$ ball has corners on the coordinate axes — points where one coordinate is exactly zero. When a smooth loss is minimised subject to an $L_1$ constraint, the solution tends to land on such a corner, which is the geometric reason the lasso produces exact zeros (Module 13). The round $L_2$ ball has no corners, so ridge regression only shrinks coefficients. The $L_\infty$ ball is a square: it only cares about the single largest coordinate.

> **Try it yourself:** call `lp_unit_circle(0.5)` and plot it. For $p < 1$ the "ball" is not convex (it caves in towards the origin), which is why $\|\cdot\|_p$ is not a norm for $p < 1$ — the triangle inequality fails.
""")

# ---------------------------------------------------------------- Section 3
nb.md(r"""
## 3. The inner (dot) product

$$\mathbf{u}\cdot\mathbf{v} = \sum_i u_i v_i = \|\mathbf{u}\|\,\|\mathbf{v}\|\cos\theta.$$

The algebraic side is a *weighted sum*; the geometric side measures *alignment*. The guide proved the two agree via the law of cosines, $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\,\mathbf{u}\cdot\mathbf{v}$. We check that identity on a thousand random pairs, together with the two inequalities that follow from it: **Cauchy–Schwarz** $|\mathbf{u}\cdot\mathbf{v}| \le \|\mathbf{u}\|\|\mathbf{v}\|$ and the **triangle inequality** $\|\mathbf{u} + \mathbf{v}\| \le \|\mathbf{u}\| + \|\mathbf{v}\|$.
""")

nb.code(r"""
u, v = np.array([3., 4.]), np.array([4., 3.])
dot = u @ v
cos_theta = dot / (np.linalg.norm(u) * np.linalg.norm(v))
print("u . v = %.1f   cos(theta) = %.3f   theta = %.2f degrees" % (dot, cos_theta, np.degrees(np.arccos(cos_theta))))

# law-of-cosines identity on 1000 random pairs of 5-D vectors (one pair per column)
U = rng.normal(size=(5, 1000)); V = rng.normal(size=(5, 1000))
lhs = np.sum((U - V) ** 2, axis=0)
rhs = np.sum(U ** 2, axis=0) + np.sum(V ** 2, axis=0) - 2 * np.sum(U * V, axis=0)
print("law of cosines: max |lhs - rhs| over 1000 pairs = %.2e" % np.max(np.abs(lhs - rhs)))

dots = np.sum(U * V, axis=0)
nu, nv = np.linalg.norm(U, axis=0), np.linalg.norm(V, axis=0)
print("Cauchy-Schwarz holds for every pair :", bool(np.all(np.abs(dots) <= nu * nv + 1e-12)))
print("triangle inequality holds for every pair:", bool(np.all(np.linalg.norm(U + V, axis=0) <= nu + nv + 1e-12)))
print("largest |u.v| / (||u|| ||v||) observed: %.3f   (equality would need parallel vectors)" % np.max(np.abs(dots) / (nu * nv)))

# orthogonality: rotating (a, b) by 90 degrees gives (-b, a), and the dot product vanishes
perp = np.array([-u[1], u[0]])
print("u =", u, " perp =", perp, " u . perp =", u @ perp)
""")

nb.md(r"""
**Notice that** the identity holds to floating-point precision ($\sim 10^{-14}$) for every pair, and that the Cauchy–Schwarz ratio never reaches 1 for random vectors — it only equals 1 when one vector is a multiple of the other.

The *weighted-sum* reading is what a linear model computes: features $\mathbf{x}$, weights $\mathbf{w}$, prediction $\mathbf{w}\cdot\mathbf{x} + b$. The *alignment* reading says the same number is large when the feature vector points along $\mathbf{w}$. And a whole dataset is scored with one matrix–vector product `X @ w`: one dot product per row.
""")

nb.code(r"""
x = np.array([54., 130., 210., 27.5])        # one patient: age, systolic BP, cholesterol, BMI
w = np.array([0.04, 0.02, 0.01, 0.05])       # illustrative risk weights per unit of each feature
b = -8.0
score = w @ x + b
theta = np.arccos((w @ x) / (np.linalg.norm(w) * np.linalg.norm(x)))
print("risk score  w.x + b             = %.3f" % score)
print("same number ||w|| ||x|| cos(theta) + b = %.3f   (theta = %.1f degrees)"
      % (np.linalg.norm(w) * np.linalg.norm(x) * np.cos(theta) + b, np.degrees(theta)))

X = np.array([[54., 130., 210., 27.5],
              [40., 118., 180., 22.0],
              [67., 155., 240., 31.2]])
print("scores for three patients, X @ w + b =", X @ w + b)
""")

# ---------------------------------------------------------------- Section 4
nb.md(r"""
## 4. Cosine similarity and a toy semantic search

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|} \in [-1, 1]$$

is the dot product of the *normalised* vectors, so it ignores length and compares direction only. Here are eight tiny "documents" as bag-of-words count vectors over a ten-word vocabulary. We search them with the query *"model data loss"* twice — once ranking by cosine similarity and once by Euclidean distance — and watch the two rankings disagree exactly where document length differs.

All eight cosine scores come out of **one matrix–vector product**: normalise the rows of the document matrix $D$, normalise the query, and compute $\hat{D}\hat{\mathbf{q}}$. That is what a vector database does, at scale.
""")

nb.code(r"""
vocab = ["model", "data", "train", "loss", "goal", "match", "team", "coach", "price", "market"]
docs = {
    "ML note (short)":          [2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "ML article (long)":        [8, 4, 4, 2, 0, 0, 0, 0, 0, 0],
    "Football report":          [0, 0, 0, 0, 3, 2, 2, 1, 0, 0],
    "Football feature (long)":  [0, 0, 1, 0, 6, 5, 6, 2, 0, 0],
    "Market update":            [0, 1, 0, 0, 0, 0, 0, 0, 3, 3],
    "ML for trading":           [2, 2, 0, 0, 0, 0, 0, 0, 2, 2],
    "Sports analytics":         [1, 2, 0, 0, 0, 1, 2, 0, 0, 0],
    "Coach interview":          [0, 0, 2, 0, 0, 0, 2, 3, 0, 0],
}
names = list(docs)
D = np.array([docs[n] for n in names], dtype=float)                 # 8 x 10 document-term matrix
query = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)       # "model data loss"

def normalize_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n == 0, 1.0, n)                             # leave all-zero rows alone

cos_scores = normalize_rows(D) @ (query / np.linalg.norm(query))    # all eight cosines at once
euc_dists  = np.linalg.norm(D - query, axis=1)

order_cos, order_euc = np.argsort(-cos_scores), np.argsort(euc_dists)
print(f"{'rank':<6}{'by cosine similarity':<26}{'cos':>6}      {'by Euclidean distance':<26}{'dist':>6}")
print("-" * 80)
for r in range(len(names)):
    i, j = order_cos[r], order_euc[r]
    print(f"{r + 1:<6}{names[i]:<26}{cos_scores[i]:6.3f}      {names[j]:<26}{euc_dists[j]:6.3f}")
print("\nnorm of each document vector:", np.linalg.norm(D, axis=1))
""")

nb.md(r"""
**Notice that** cosine ranks the long ML article at the top — it is the most "about" the query — while Euclidean distance pushes it far down the list, below *Sports analytics* and *ML for trading*, purely because its vector is long ($\|\mathbf{d}\| \approx 10$ versus $\approx 2.4$ for the short note). Euclidean distance mixes "different topic" with "different length"; cosine measures topic only.

The full pairwise picture is the matrix $\hat{D}\hat{D}^\top$ of all document–document cosines.
""")

nb.code(r"""
Dn = normalize_rows(D)
S = Dn @ Dn.T                                      # 8 x 8 matrix of pairwise cosine similarities

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(S, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8); ax.set_yticklabels(names, fontsize=8)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if S[i, j] < 0.6 else "k")
plt.colorbar(im, ax=ax, label="cosine similarity")
ax.set_title("Pairwise cosine similarity of the eight documents"); ax.set_xlabel("document"); ax.set_ylabel("document")
plt.tight_layout(); plt.show()

print("ML note vs ML article: cos = %.3f  (same topic, 4x the length)" % S[0, 1])
print("ML note vs Football report: cos = %.3f  (no shared words)" % S[0, 2])
""")

nb.md(r"""
> **Try it yourself:** change the query to *"goal team price"* (`[0,0,0,0,1,0,1,0,1,0]`) and re-run the search. Then multiply the query by 10 — which of the two rankings changes, and why?
""")

# ---------------------------------------------------------------- Section 5
nb.md(r"""
## 5. Projections

The projection of $\mathbf{a}$ onto the line through $\mathbf{b}$, and the residual left over:

$$\mathbf{p} = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b}, \qquad \mathbf{r} = \mathbf{a} - \mathbf{p}, \qquad \mathbf{r}\cdot\mathbf{b} = 0.$$

$\mathbf{p}$ is the *best approximation* of $\mathbf{a}$ among all multiples $t\mathbf{b}$: the squared error $\|\mathbf{a} - t\mathbf{b}\|^2$ is a parabola in $t$ whose minimum sits at $t^\star = \mathbf{a}\cdot\mathbf{b}/\mathbf{b}\cdot\mathbf{b}$. The guide's example is $\mathbf{a} = (2, 3)$, $\mathbf{b} = (4, 2)$, giving $t^\star = 0.7$, $\mathbf{p} = (2.8, 1.4)$ and $\mathbf{r} = (-0.8, 1.6)$.
""")

nb.code(r"""
a, b = np.array([2., 3.]), np.array([4., 2.])
t_star = (a @ b) / (b @ b)
p = t_star * b
r = a - p
print("t* = %.3f   p = %s   r = %s   r . b = %.2e" % (t_star, p, r, r @ b))
print("Pythagoras: ||p||^2 + ||r||^2 = %.3f   ||a||^2 = %.3f" % (p @ p + r @ r, a @ a))
print("fraction of ||a||^2 explained by direction b: %.3f  (= cos^2 theta)" % ((p @ p) / (a @ a)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for vec, col, lab in [(a, "C0", "a"), (b, "C1", "b"), (p, "C2", "p = proj_b a")]:
    ax1.quiver(0, 0, vec[0], vec[1], angles="xy", scale_units="xy", scale=1, color=col, width=0.012, label=lab)
ax1.quiver(p[0], p[1], r[0], r[1], angles="xy", scale_units="xy", scale=1, color="C3", width=0.012, label="r = a - p")
ax1.plot([-1, 5], [-0.5, 2.5], "k:", lw=1, label="line through b")
ax1.set_xlim(-1, 5); ax1.set_ylim(-1, 4); ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc="upper left")
ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$"); ax1.set_title("Projection of a onto b, and the residual")

ts = np.linspace(-0.5, 2.0, 200)
errs = [np.sum((a - t * b) ** 2) for t in ts]
ax2.plot(ts, errs)
ax2.axvline(t_star, color="C2", ls="--", label="t* = a.b / b.b = %.2f" % t_star)
ax2.set_xlabel("t"); ax2.set_ylabel("squared error  ||a - t b||^2"); ax2.set_title("Squared error is a parabola in t")
ax2.legend(); ax2.grid(alpha=0.3)
plt.show()
""")

nb.md(r"""
**Notice that** the residual arrow meets the dotted line at a right angle — $\mathbf{r}\cdot\mathbf{b}$ is zero to machine precision — and that the parabola's minimum is exactly at $t^\star$. Replace the single direction $\mathbf{b}$ by several feature columns and the same picture becomes *least squares*: the fitted values are the projection of the target onto the span of the columns, and the residual vector is orthogonal to every column (Module 02).
""")

# ---------------------------------------------------------------- Section 6
nb.md(r"""
## 6. Orthonormal bases and Gram–Schmidt

Vectors $\mathbf{q}_1, \dots, \mathbf{q}_n$ are **orthonormal** when $\mathbf{q}_i\cdot\mathbf{q}_j = 1$ if $i = j$ and $0$ otherwise; as columns of $Q$, this reads $Q^\top Q = I$. Coordinates are then just dot products, and lengths are preserved (Parseval):

$$\mathbf{x} = \sum_i c_i\,\mathbf{q}_i, \qquad c_i = \mathbf{x}\cdot\mathbf{q}_i, \qquad \|\mathbf{x}\|^2 = \sum_i c_i^2 .$$

**Gram–Schmidt** turns any independent set into an orthonormal one: take each new vector, subtract its projections onto the directions already built, normalise. Below is the *modified* variant (each subtraction uses the already-reduced vector, which is numerically a little kinder); `np.linalg.qr` does the same job with Householder reflections, so its columns may differ from ours by a sign.
""")

nb.code(r"""
def gram_schmidt(A):
    # Columns of A (linearly independent) -> Q with orthonormal columns and the same span.
    A = np.asarray(A, dtype=float)
    Q = np.zeros_like(A)
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i in range(j):
            v -= (v @ Q[:, i]) * Q[:, i]            # remove the component along q_i
        Q[:, j] = v / np.linalg.norm(v)
    return Q

A = np.column_stack([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.]])      # a1, a2, a3 from the guide
Q = gram_schmidt(A)
print("Q =\n", Q)
print("by hand: q1 = (1,1,0)/sqrt2, q2 = (1,-1,2)/sqrt6, q3 = (-1,1,1)/sqrt3 ->\n",
      np.column_stack([[1, 1, 0] / np.sqrt(2), [1, -1, 2] / np.sqrt(6), [-1, 1, 1] / np.sqrt(3)]))
print("Q^T Q =\n", Q.T @ Q)

Q_qr, R = np.linalg.qr(A)
print("agrees with np.linalg.qr up to column signs:", np.allclose(np.abs(Q), np.abs(Q_qr)))

x = np.array([3., 1., 2.])
c = Q.T @ x                                        # coordinates = dot products with the basis vectors
print("coordinates of x = (3, 1, 2) in the new basis:", c)
print("reconstruction Q c =", Q @ c, "   Parseval: sum c_i^2 = %.3f, ||x||^2 = %.3f" % (c @ c, x @ x))
""")

nb.md(r"""
**Notice that** the third coordinate of $\mathbf{x}$ is zero: $\mathbf{x}$ lies in the plane spanned by $\mathbf{a}_1$ and $\mathbf{a}_2$ (indeed $\mathbf{x} = \mathbf{a}_1 + 2\mathbf{a}_2$), and the orthonormal basis makes that visible at a glance. The same idea organises colour: a pixel's $(R, G, B)$ values re-expressed in a brightness/chroma basis put almost all of their energy into the brightness coordinate — which is why image codecs compress chroma far more aggressively than luma.
""")

nb.code(r"""
pixel = np.array([200., 150., 100.])                       # (R, G, B)
B = np.column_stack([np.ones(3) / np.sqrt(3),              # brightness direction
                     np.array([1., 0., -1.]) / np.sqrt(2),  # red-versus-blue
                     np.array([1., -2., 1.]) / np.sqrt(6)]) # green-versus-magenta
coords = B.T @ pixel
print("B^T B = I :", np.allclose(B.T @ B, np.eye(3)))
print("coordinates (brightness, red-blue, green-magenta):", coords)
print("share of the pixel's energy in the brightness coordinate: %.1f%%" % (100 * coords[0] ** 2 / (pixel @ pixel)))
print("Parseval: %.1f = %.1f" % (coords @ coords, pixel @ pixel))
""")

nb.md(r"""
> **Try it yourself:** feed `gram_schmidt` three *nearly* dependent columns, e.g. `np.column_stack([[1,1,1], [1,1,1+1e-8], [1,1+1e-8,1]])`, and compare `Q.T @ Q` with the result of `np.linalg.qr`. How far from the identity is each? (This is why production code uses QR or SVD rather than textbook Gram–Schmidt.)
""")

# ---------------------------------------------------------------- Section 7
nb.md(r"""
## 7. Hyperplanes and half-spaces

The set $\{\mathbf{x} : \mathbf{w}\cdot\mathbf{x} + b = 0\}$ is a hyperplane with **normal vector** $\mathbf{w}$. The **signed distance** of any point to it is

$$\delta(\mathbf{x}) = \frac{\mathbf{w}\cdot\mathbf{x} + b}{\|\mathbf{w}\|},$$

positive on the side $\mathbf{w}$ points to. A linear classifier predicts $\operatorname{sign}(\mathbf{w}\cdot\mathbf{x} + b)$, and the **margin** of a labelled point is $y\,\delta(\mathbf{x})$.

The simplest linear classifier is *nearest centroid*: assign a point to whichever class mean is closer. Working out $\|\mathbf{x} - \boldsymbol{\mu}_1\|^2 < \|\mathbf{x} - \boldsymbol{\mu}_0\|^2$ shows that this is a hyperplane with $\mathbf{w} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$ passing through the midpoint of the two centroids, i.e. $b = -\mathbf{w}\cdot(\boldsymbol{\mu}_0 + \boldsymbol{\mu}_1)/2$.
""")

nb.code(r"""
n = 60
X0 = rng.normal(loc=[1.0, 1.0], scale=0.7, size=(n, 2))      # class -1
X1 = rng.normal(loc=[3.5, 3.0], scale=0.7, size=(n, 2))      # class +1
X = np.vstack([X0, X1]); y = np.array([-1] * n + [1] * n)

mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
w = mu1 - mu0                                   # normal vector, pointing from class -1 towards class +1
b = -w @ (mu0 + mu1) / 2                        # boundary through the midpoint of the centroids
signed = (X @ w + b) / np.linalg.norm(w)        # signed distances, all points at once
pred = np.where(signed > 0, 1, -1)
margins = y * signed
print("w =", w, "  b = %.3f  ||w|| = %.3f" % (b, np.linalg.norm(w)))
print("training accuracy: %.3f    misclassified: %d" % (np.mean(pred == y), np.sum(margins <= 0)))
print("smallest positive margin: %.3f" % margins[margins > 0].min())
print("rescaling (w, b) by 10 leaves every signed distance unchanged:",
      np.allclose((X @ (10 * w) + 10 * b) / np.linalg.norm(10 * w), signed))

fig, ax = plt.subplots(figsize=(6.5, 5.5))
sc = ax.scatter(X[:, 0], X[:, 1], c=signed, cmap="coolwarm", vmin=-3, vmax=3, edgecolor="k", lw=0.3)
plt.colorbar(sc, ax=ax, label="signed distance to the boundary")
xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 50)
ax.plot(xs, -(w[0] * xs + b) / w[1], "k-", label="w . x + b = 0")
mid = (mu0 + mu1) / 2
ax.quiver(mid[0], mid[1], w[0], w[1], angles="xy", scale_units="xy", scale=1, color="k", width=0.01, label="normal vector w")
ax.scatter(mu0[0], mu0[1], marker="X", s=160, c="C0", edgecolor="k", label="centroid, class -1")
ax.scatter(mu1[0], mu1[1], marker="X", s=160, c="C3", edgecolor="k", label="centroid, class +1")
ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("A linear classifier is a hyperplane with normal w")
plt.show()
""")

nb.md(r"""
**Notice that** $\mathbf{w}$ is *perpendicular* to the decision line, not along it, and that the colour (signed distance) changes only as you move in the direction of $\mathbf{w}$ — moving parallel to the boundary changes nothing. Points near the line have small margins and are the ones a slightly different training set would flip; a support vector machine chooses $\mathbf{w}, b$ to make the smallest margin as large as possible. Scaling $(\mathbf{w}, b)$ changes the raw scores but not the boundary or the distances — which is why logistic regression needs a regulariser to pin down the scale of $\mathbf{w}$.
""")

# ---------------------------------------------------------------- Section 8
nb.md(r"""
## 8. High-dimensional geometry

Everything so far holds in any dimension, but the *typical* behaviour of random points changes completely as $d$ grows. For two random vectors with independent standard-normal coordinates,

$$\mathbb{E}[\cos(\mathbf{u}, \mathbf{v})] = 0, \qquad \operatorname{sd}[\cos(\mathbf{u}, \mathbf{v})] = \frac{1}{\sqrt d},$$

so random directions become nearly orthogonal. We draw 300 random vectors for $d = 2, 10, 100, 1000$, compute **all pairwise cosines in one matrix product** $\hat{Z}\hat{Z}^\top$, and compare the spread with $1/\sqrt d$.
""")

nb.code(r"""
dims, n = [2, 10, 100, 1000], 300
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
print(f"{'d':>6} {'std of cosine':>14} {'1/sqrt(d)':>10} {'max |cos|':>10}")
for ax, d in zip(axes.ravel(), dims):
    Z = rng.normal(size=(n, d))
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    C = Zn @ Zn.T                                        # all n x n cosines at once
    cosines = C[np.triu_indices(n, k=1)]                 # each unordered pair exactly once
    print(f"{d:>6} {cosines.std():>14.4f} {1 / np.sqrt(d):>10.4f} {np.abs(cosines).max():>10.3f}")
    ax.hist(cosines, bins=80, range=(-1, 1), density=True, color="C0", alpha=0.85)
    ax.set_title(f"d = {d}: cosine of {n * (n - 1) // 2} random pairs"); ax.set_xlabel("cosine similarity"); ax.set_ylabel("density")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** the measured standard deviation tracks $1/\sqrt d$ almost perfectly, and at $d = 1000$ no pair out of 44,850 reaches a cosine of even $0.15$. (In $d = 2$ the histogram piles up at $\pm 1$: the cosine of a uniformly random angle has an arcsine-shaped density, so in the plane random directions are *more* likely to be nearly parallel than nearly perpendicular — the opposite of high dimensions.) This is the baseline against which any "similarity" must be judged: a cosine of $0.3$ is nothing in 10 dimensions and a ten-sigma event in 1000.

The second surprise is **distance concentration**. For random points in the unit cube, $\|\mathbf{x} - \mathbf{y}\|^2$ is a sum of $d$ independent terms, so its relative spread shrinks like $1/\sqrt d$ — the nearest and the farthest neighbour end up at almost the same distance.
""")

nb.code(r"""
dims, n = [2, 5, 10, 20, 50, 100, 200, 500, 1000], 500
contrast, rel_spread = [], []
for d in dims:
    P = rng.random((n, d))                               # uniform in the unit cube
    dist = np.linalg.norm(P[1:] - P[0], axis=1)          # distances from the first point to all others
    contrast.append((dist.max() - dist.min()) / dist.min())
    rel_spread.append(dist.std() / dist.mean())

fig, ax = plt.subplots()
ax.plot(dims, contrast, "o-", label="(farthest - nearest) / nearest")
ax.plot(dims, rel_spread, "s-", label="std / mean of the distances")
ax.plot(dims, 0.59 / np.sqrt(np.array(dims, float)), "k:", label="0.59 / sqrt(d)  (theory for std/mean)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
ax.set_xlabel("dimension d"); ax.set_ylabel("relative spread"); ax.set_title("Distances from a random point concentrate as d grows")
plt.show()
for d, c, s in zip(dims, contrast, rel_spread):
    print(f"d = {d:>5}: farthest neighbour is {100 * c:7.1f}% farther than the nearest;  std/mean = {s:.3f}")
""")

nb.md(r"""
**Notice that** in two dimensions the farthest point is tens of times farther away than the nearest, while at $d = 1000$ the difference is a few percent. A nearest-neighbour method on raw 1000-dimensional noise is essentially picking at random. (The dotted theory line: the relative spread of the *squared* distance is $\approx 1.18/\sqrt d$, and taking the square root halves relative fluctuations, giving $\approx 0.59/\sqrt d$ for the distance itself.)

Third surprise: **volume lives in the corners.** The unit ball has volume $V_d = \pi^{d/2}/\Gamma(d/2 + 1)$, the cube $[-1, 1]^d$ around it has volume $2^d$. We compare the exact ratio with a Monte Carlo estimate.
""")

nb.code(r"""
from scipy.special import gamma

def ball_volume(d, r=1.0):
    return np.pi ** (d / 2) / gamma(d / 2 + 1) * r ** d

ds = np.arange(1, 21)
ratio = np.array([ball_volume(d) / 2 ** d for d in ds])
mc = []
for d in range(1, 11):                                   # Monte Carlo: fraction of cube points inside the ball
    P = rng.uniform(-1, 1, size=(100_000, d))
    mc.append(np.mean(np.sum(P ** 2, axis=1) <= 1))

fig, ax = plt.subplots()
ax.semilogy(ds, ratio, "o-", label="exact  V_d / 2^d")
ax.semilogy(range(1, 11), mc, "x", ms=9, color="C3", label="Monte Carlo, 100k points")
ax.set_xlabel("dimension d"); ax.set_ylabel("fraction of the cube inside the inscribed ball")
ax.set_title("The inscribed ball vanishes: volume lives in the corners"); ax.legend(); ax.grid(alpha=0.3, which="both")
plt.show()
print("fraction of the cube inside the ball:  d=2: %.4f   d=3: %.4f   d=10: %.5f   d=20: %.2e" % (ratio[1], ratio[2], ratio[9], ratio[19]))
print("fraction of a ball within 90%% of its radius, 0.9^d:  d=10: %.3f   d=100: %.1e" % (0.9 ** 10, 0.9 ** 100))
""")

nb.md(r"""
**Where this bites in deep learning.** Attention scores are dot products $\mathbf{q}\cdot\mathbf{k}$ of $d_k$-dimensional vectors. For roughly unit-variance coordinates that dot product has standard deviation $\sqrt{d_k}$, so without correction the scores for different keys are spread over tens of units and the softmax saturates onto a single key. Dividing by $\sqrt{d_k}$ brings the spread back to one — this is the only reason the $\sqrt{d_k}$ appears in the transformer's attention formula.
""")

nb.code(r"""
d_k, n_keys = 64, 10
K = rng.normal(size=(n_keys, d_k))
Qs = rng.normal(size=(2000, d_k))                         # many queries, to measure the spread of q.k reliably
all_scores = Qs @ K.T                                     # (2000, 10): one dot product per query-key pair
print("std of the raw scores q.k over 20000 pairs: %.2f   (theory: sqrt(d_k) = %.1f)" % (all_scores.std(), np.sqrt(d_k)))
print("std after dividing by sqrt(d_k)           : %.2f" % (all_scores / np.sqrt(d_k)).std())

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

scores = all_scores[0]                                    # the ten scores of the first query
print("attention weights, raw scores        :", softmax(scores))
print("attention weights, scores / sqrt(d_k):", softmax(scores / np.sqrt(d_k)))
print("largest weight: raw %.3f  vs  scaled %.3f" % (softmax(scores).max(), softmax(scores / np.sqrt(d_k)).max()))
""")

nb.md(r"""
> **Try it yourself:** set `d_k = 512` and re-run. The raw-score softmax should become a one-hot vector (all attention on one key, gradients near zero for the rest), while the scaled version stays spread out. Then try dividing by `d_k` instead of `np.sqrt(d_k)` — why is that too much?
""")

# ---------------------------------------------------------------- Summary
nb.md(r"""
## Summary

- A vector is simultaneously an arrow, a point and a data record; addition and scaling are component-wise, and `rank` tells you whether a set of vectors is independent.
- Norms are rulers: $L_2$ (round ball), $L_1$ (diamond — corners give sparsity), $L_\infty$ (square); distances depend on the norm *and* on how features are scaled.
- The dot product is a weighted sum and a measure of alignment at once: $\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$, verified numerically along with Cauchy–Schwarz and the triangle inequality.
- Cosine similarity compares direction only; it ranked the long ML article first where Euclidean distance buried it. All similarities to a query come from one matrix–vector product.
- Projection $\mathbf{p} = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\mathbf{b}$ is the best approximation on a line; the residual is orthogonal; the squared error is a parabola.
- Gram–Schmidt builds an orthonormal basis by subtracting projections; coordinates are then dot products and Parseval preserves length; `np.linalg.qr` agrees up to signs.
- A linear classifier is a hyperplane $\mathbf{w}\cdot\mathbf{x} + b = 0$ with normal $\mathbf{w}$; signed distance $(\mathbf{w}\cdot\mathbf{x} + b)/\|\mathbf{w}\|$ is invariant to rescaling $(\mathbf{w}, b)$.
- In high dimension random vectors are nearly orthogonal (cosine spread $1/\sqrt d$), distances concentrate ($\approx 0.59/\sqrt d$ relative spread), and the inscribed ball's share of the cube vanishes — hence $\sqrt{d_k}$ in attention and the failure of $k$-NN on raw high-dimensional inputs.
""")

nb.md(r"""
**Next:** open `exercises.ipynb` and implement these objects yourself — norms without `np.linalg.norm`, top-$k$ cosine retrieval, projections, Gram–Schmidt, signed distances, a distance-concentration experiment, and a small recommender built from user–user cosine similarity. Reference solutions are in `solutions.ipynb`.
""")

nb.save(MOD / "lesson.ipynb")
print("lesson cells:", len(nb.cells))


# ============================================================================
# EXERCISES — reference solutions (executed here to compute hidden hashes)
# ============================================================================
SOL_E1 = r"""
def norms(v):
    # Return (l1, l2, linf) as Python floats, without np.linalg.norm.
    v = np.asarray(v, dtype=float)
    l1 = float(np.sum(np.abs(v)))
    l2 = float(np.sqrt(v @ v))
    linf = float(np.max(np.abs(v)))
    return (l1, l2, linf)
"""

SOL_E2 = r"""
def angle_between(u, v):
    # Angle in degrees, in [0, 180]; clip the cosine so rounding noise cannot break arccos.
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    c = (u @ v) / (np.sqrt(u @ u) * np.sqrt(v @ v))
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))
"""

SOL_E3 = r"""
def normalize_rows(E):
    # Each row divided by its L2 norm; all-zero rows are returned unchanged (as zeros).
    E = np.asarray(E, dtype=float)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.where(n == 0, 1.0, n)
"""

SOL_E5 = r"""
def top_k_cosine(E, q, k):
    # Indices of the k rows of E with the largest cosine similarity to q, best first.
    E = np.asarray(E, dtype=float); q = np.asarray(q, dtype=float)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    scores = En @ (q / np.linalg.norm(q))          # all cosines in one matrix-vector product
    return np.argsort(-scores)[:k]
"""

SOL_E6 = r"""
def project(a, b):
    # Vector projection of a onto the line through b, and the residual a - p.
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    p = (a @ b) / (b @ b) * b
    r = a - p
    return p, r
"""

SOL_E7 = r"""
def gram_schmidt(A):
    # Columns of A (independent) -> Q with orthonormal columns, same span, q_j . a_j > 0.
    A = np.asarray(A, dtype=float)
    Q = np.zeros_like(A)
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i in range(j):
            v = v - (v @ Q[:, i]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q
"""

SOL_E8 = r"""
def signed_distance(X, w, b):
    # Signed distance of every row of X (shape (n, d)) to the hyperplane w.x + b = 0 -> shape (n,).
    X = np.asarray(X, dtype=float); w = np.asarray(w, dtype=float)
    return (X @ w + b) / np.linalg.norm(w)
"""

SOL_E10 = r"""
def distance_concentration(d, n=500, seed=0):
    # Sample n points uniformly in [0,1]^d, measure distances from the first point to the others.
    g = np.random.default_rng(seed)
    P = g.random((n, d))
    dist = np.linalg.norm(P[1:] - P[0], axis=1)
    contrast = float((dist.max() - dist.min()) / dist.min())
    sq = dist ** 2
    rel_std_sq = float(sq.std() / sq.mean())
    return contrast, rel_std_sq
"""

SOL_E11 = r"""
def reflect(X, w, b):
    # Mirror image of every row of X across the hyperplane w.x + b = 0.
    X = np.asarray(X, dtype=float); w = np.asarray(w, dtype=float)
    w_hat = w / np.linalg.norm(w)
    delta = (X @ w + b) / np.linalg.norm(w)            # signed distances, shape (n,)
    return X - 2.0 * delta[:, None] * w_hat
"""

ns = {"np": np}
for src in (SOL_E1, SOL_E2, SOL_E3, SOL_E5, SOL_E6, SOL_E7, SOL_E8, SOL_E10, SOL_E11):
    exec(src, ns)

H = {}
# E1 — hidden L1 norm of a seeded 50-vector
z = np.random.default_rng(11).normal(size=50)
H["E1A"] = sha_of(safe_round(ns["norms"](z)[0]))
# E2 — hidden angle between (1,2,3) and (-1,0,1)
H["E2A"] = sha_of(safe_round(ns["angle_between"](np.array([1., 2., 3.]), np.array([-1., 0., 1.]))))
# E3 — hidden sum of all entries of the normalised seeded matrix
M3 = np.random.default_rng(5).normal(size=(6, 4)); M3[2] = 0.0
H["E3A"] = sha_of(safe_round(float(ns["normalize_rows"](M3).sum())))
# E5 — hidden top-5 indices for a seeded embedding table
g5 = np.random.default_rng(123); E5 = g5.normal(size=(50, 8)); q5 = g5.normal(size=8)
H["E5A"] = sha_of(np.asarray(ns["top_k_cosine"](E5, q5, 5), dtype=int))
# E6 — hidden fraction of ||a||^2 explained by b
p6, r6 = ns["project"](np.array([1., 2., 3., 4.]), np.array([1., 1., 0., 1.]))
H["E6A"] = sha_of(safe_round(float((p6 @ p6) / 30.0)))
# E7 — hidden sum of |Q| for a seeded 5x3 matrix (sign-independent)
A7 = np.random.default_rng(9).normal(size=(5, 3))
H["E7A"] = sha_of(safe_round(float(np.abs(ns["gram_schmidt"](A7)).sum())))
# E8 — hidden count of points on the positive side, and the smallest |distance|
P8 = np.random.default_rng(21).normal(size=(200, 3)); w8 = np.array([1., -2., 2.]); b8 = 0.5
sd8 = ns["signed_distance"](P8, w8, b8)
H["E8A"] = sha_of(int(np.sum(sd8 > 0)))
H["E8B"] = sha_of(safe_round(float(np.min(np.abs(sd8)))))
# E11 — hidden sum of the reflected seeded points
P11 = np.random.default_rng(33).normal(size=(20, 5)); w11 = np.array([1., 2., 0., -1., 3.]); b11 = -0.7
H["E11A"] = sha_of(safe_round(float(ns["reflect"](P11, w11, b11).sum())))

# sanity: E10 ranges used by the checks must hold for the reference solution
c2, s2 = ns["distance_concentration"](2)
c10, _ = ns["distance_concentration"](10)
c100, s100 = ns["distance_concentration"](100)
c1000, _ = ns["distance_concentration"](1000)
assert c2 > 1.0 and c1000 < 0.35 and c2 > c10 > c100 > c1000, (c2, c10, c100, c1000)
assert 0.08 < s100 < 0.16, s100


def fill(check_src: str) -> str:
    for key, val in H.items():
        check_src = check_src.replace(f"__H_{key}__", val)
    assert "__H_" not in check_src, "unfilled hash placeholder"
    return check_src


# ============================================================================
# EXERCISE SET
# ============================================================================
ex = ExerciseSet(
    "Module 01 — Exercises",
    intro_md=r"""
These exercises practise the geometry of vectors from the [concept guide](README.md): norms, angles, cosine retrieval, projections, Gram–Schmidt, hyperplanes and the behaviour of random vectors in high dimension. The mini-project at the end builds a small recommender from user–user cosine similarity.

Only NumPy and matplotlib are needed. Where an exercise forbids a library call (e.g. `np.linalg.norm`), the point is to build the object from its definition once — afterwards, use the library.
""",
    setup_code=r"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def _safe(thunk):
    # Run thunk() and return None instead of raising; the check cells use this so they never crash.
    try:
        return thunk()
    except Exception:
        return None
""",
)

# ---------------------------------------------------------------- E1
ex.add(
    title="Three norms from their definitions", difficulty=1,
    prompt_md=r"""
Implement `norms(v)` for a 1-D NumPy array `v`. It must return a tuple of three Python floats `(l1, l2, linf)`:

$$\|\mathbf{v}\|_1 = \sum_i |v_i|, \qquad \|\mathbf{v}\|_2 = \sqrt{\mathbf{v}\cdot\mathbf{v}}, \qquad \|\mathbf{v}\|_\infty = \max_i |v_i|.$$

Use only elementary NumPy (`np.abs`, `np.sum`, `np.sqrt`, `np.max`, `@`) — **no** `np.linalg.norm`.
""",
    starter=r"""
def norms(v):
    # Return (l1, l2, linf) as Python floats, without np.linalg.norm.
    # YOUR CODE HERE
    return None
""",
    solution=SOL_E1,
    check=fill(r"""
v = np.array([3., -4., 12.])
got = _safe(lambda: norms(v))
check_close('L1 norm of (3, -4, 12) is 19', _safe(lambda: got[0]), 19.0)
check_close('L2 norm of (3, -4, 12) is 13', _safe(lambda: got[1]), 13.0)
check_close('L-infinity norm of (3, -4, 12) is 12', _safe(lambda: got[2]), 12.0)

z = np.random.default_rng(11).normal(size=50)
got_z = _safe(lambda: norms(z))
check_close('L2 agrees with np.linalg.norm on a random 50-vector', _safe(lambda: got_z[1]), np.linalg.norm(z))
check_hash('hidden: L1 norm of the random 50-vector', _safe(lambda: got_z[0]), '__H_E1A__')
check('ordering  Linf <= L2 <= L1  holds', _safe(lambda: got_z[2] <= got_z[1] <= got_z[0]))
"""),
    explanation_md=r"""
Each norm is a one-liner once you read the definition literally: absolute values summed, squares summed and square-rooted (equivalently $\sqrt{\mathbf{v}\cdot\mathbf{v}}$), or the largest absolute value. The common wrong turns are forgetting the absolute value in the $L_1$ norm (negative entries would then cancel) and forgetting the square root in the $L_2$ norm (which returns the *squared* length). The ordering $\|\mathbf{v}\|_\infty \le \|\mathbf{v}\|_2 \le \|\mathbf{v}\|_1$ holds for every vector: the max is one of the terms in the sum of squares, and $\sqrt{\sum v_i^2} \le \sum |v_i|$ because the cross terms $2|v_i||v_j|$ are non-negative.
""",
)

# ---------------------------------------------------------------- E2
ex.add(
    title="Angle between two vectors", difficulty=1,
    prompt_md=r"""
Implement `angle_between(u, v)` returning the angle in **degrees** (a float in $[0, 180]$) between two non-zero vectors, from

$$\cos\theta = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}.$$

Floating-point noise can push the cosine to $1.0000000002$, where `np.arccos` returns `nan`; clip the cosine to $[-1, 1]$ before taking the arc-cosine.
""",
    starter=r"""
def angle_between(u, v):
    # Angle in degrees in [0, 180]. Remember to clip the cosine to [-1, 1].
    # YOUR CODE HERE
    return None
""",
    solution=SOL_E2,
    check=fill(r"""
check_close('angle between (3,4) and (4,3) is 16.26 degrees', _safe(lambda: angle_between(np.array([3., 4.]), np.array([4., 3.]))), 16.2602, tol=1e-3)
check_close('orthogonal vectors give 90 degrees', _safe(lambda: angle_between(np.array([1., 0.]), np.array([0., 1.]))), 90.0, tol=1e-6)
check_close('anti-parallel vectors give 180 degrees (needs the clip)', _safe(lambda: angle_between(np.array([1., 2.]), np.array([-2., -4.]))), 180.0, tol=1e-6)
check_close('parallel vectors give 0 degrees (needs the clip)', _safe(lambda: angle_between(np.array([0.1, 0.2, 0.3]), np.array([1., 2., 3.]))), 0.0, tol=1e-6)
check_hash('hidden: angle between (1,2,3) and (-1,0,1)', _safe(lambda: angle_between(np.array([1., 2., 3.]), np.array([-1., 0., 1.]))), '__H_E2A__')
"""),
    explanation_md=r"""
The geometric definition of the dot product, solved for $\theta$. The clip matters: for parallel vectors the computed cosine can be $1 + 10^{-16}$, and `np.arccos` of anything outside $[-1, 1]$ is `nan`. Returning degrees rather than radians is just `np.degrees`. A frequent slip is to divide by $\|\mathbf{u}\|\|\mathbf{v}\|$ *after* the arc-cosine, or to divide by $\mathbf{u}\cdot\mathbf{u}$ (the squared norm) instead of the norm.
""",
)

# ---------------------------------------------------------------- E3
ex.add(
    title="Normalise the rows of an embedding table", difficulty=1,
    prompt_md=r"""
Implement `normalize_rows(E)` for a 2-D array `E` of shape `(n, d)`: return an array of the same shape in which every non-zero row has been divided by its $L_2$ norm (so it becomes a unit vector) and every all-zero row is left as zeros (no division by zero, no `nan`). Use broadcasting — no Python loop over rows. `np.linalg.norm(..., axis=1, keepdims=True)` is allowed here.
""",
    starter=r"""
def normalize_rows(E):
    # Unit-length rows; all-zero rows stay zero. No loops.
    # YOUR CODE HERE
    return None
""",
    solution=SOL_E3,
    check=fill(r"""
M = np.random.default_rng(5).normal(size=(6, 4)); M[2] = 0.0
got = _safe(lambda: normalize_rows(M))
check('output has the same shape as the input', _safe(lambda: got.shape == (6, 4)))
check_close('non-zero rows have norm 1', _safe(lambda: np.linalg.norm(got[[0, 1, 3, 4, 5]], axis=1)), np.ones(5))
check_close('the all-zero row stays zero (no nan)', _safe(lambda: got[2]), np.zeros(4))
check_close('directions are preserved (row 0 is a positive multiple of the original)', _safe(lambda: got[0] * np.linalg.norm(M[0])), M[0])
check_hash('hidden: sum of all entries of the normalised matrix', _safe(lambda: float(got.sum())), '__H_E3A__')
"""),
    explanation_md=r"""
`np.linalg.norm(E, axis=1, keepdims=True)` gives an `(n, 1)` column of row norms, which broadcasts against `(n, d)` when you divide. The only subtlety is the zero row: dividing by zero produces `nan`, so replace zero norms by 1 before dividing (`np.where(n == 0, 1.0, n)`) — a zero vector has no direction, and leaving it at zero is the honest choice. Forgetting `keepdims=True` yields shape `(n,)`, which NumPy would try to broadcast along the *last* axis and fail (or silently do the wrong thing when `n == d`).
""",
)

# ---------------------------------------------------------------- E4 (conceptual)
ex.add_conceptual(
    title="Why cosine rather than Euclidean distance for text?", difficulty=1,
    prompt_md=r"""
A colleague builds a document search engine on bag-of-words count vectors and ranks results by Euclidean distance to the query. A long, highly relevant article keeps ranking below short, barely related notes. Explain (a) what is going wrong geometrically, (b) why cosine similarity fixes it, (c) the exact relationship between cosine similarity and Euclidean distance *after* normalisation, and (d) one situation where you would deliberately keep the raw dot product or Euclidean distance instead.
""",
    answer_md=r"""
(a) Count vectors scale with document length: a 3,000-word article on the query's topic points in (roughly) the same direction as a 300-word note but has about ten times the norm. Euclidean distance $\|\mathbf{q} - \mathbf{d}\|$ mixes two kinds of difference — direction (topic) and magnitude (length) — and for long documents the magnitude term dominates, so the relevant long article looks "far away" and an unrelated short note looks "close" (the guide's example: the football report beats the long ML article as neighbour of the ML note).

(b) Cosine similarity $\mathbf{q}\cdot\mathbf{d}/(\|\mathbf{q}\|\|\mathbf{d}\|)$ divides out both norms, so it depends only on the angle. Duplicating a document (doubling every count) leaves its cosine to every query unchanged.

(c) For unit vectors, $\|\hat{\mathbf{q}} - \hat{\mathbf{d}}\|^2 = 2 - 2\cos(\mathbf{q}, \mathbf{d})$, a monotone decreasing function of the cosine — so ranking by cosine is *identical* to ranking by Euclidean distance between the normalised vectors. The real choice is not "cosine versus Euclidean" but "normalise or not".

(d) Keep magnitude when it carries meaning: sensor readings or coordinates, where a larger vector really is a different state; embeddings trained with a metric loss that encodes confidence in the norm; or recommender item vectors whose length encodes popularity, where the raw dot product deliberately favours popular items. Attention scores also use the raw (scaled) dot product, because the model learns to use the norms of queries and keys.
""",
)

# ---------------------------------------------------------------- E5
ex.add(
    title="Top-k retrieval by cosine similarity", difficulty=2,
    prompt_md=r"""
Implement `top_k_cosine(E, q, k)`: given an embedding table `E` of shape `(n, d)` (no zero rows), a query vector `q` of shape `(d,)` and an integer `k`, return a NumPy integer array of the `k` row indices with the **largest cosine similarity** to `q`, in decreasing order of similarity.

Constraint: compute all `n` similarities with a **single matrix–vector product** (normalise the rows and the query first) — no Python loop over rows. `np.argsort` is allowed.
""",
    starter=r"""
def top_k_cosine(E, q, k):
    # Indices of the k rows most similar to q (by cosine), best first. One matrix-vector product, no loops.
    # YOUR CODE HERE
    return None
""",
    solution=SOL_E5,
    check=fill(r"""
E_small = np.array([[1., 0.], [0., 1.], [2., 0.1], [-1., 0.]])
q_small = np.array([1., 0.])
got = _safe(lambda: np.asarray(top_k_cosine(E_small, q_small, 2), dtype=int))
check_close('top-2 for the toy table is rows [0, 2] (row 2 is long but well aligned)', got, np.array([0, 2]))
got3 = _safe(lambda: np.asarray(top_k_cosine(E_small, 10 * q_small, 3), dtype=int))
check_close('scaling the query does not change the ranking', got3, np.array([0, 2, 1]))

g = np.random.default_rng(123); E_big = g.normal(size=(50, 8)); q_big = g.normal(size=8)
idx = _safe(lambda: np.asarray(top_k_cosine(E_big, q_big, 5), dtype=int))
check('returns exactly 5 indices', _safe(lambda: idx.shape == (5,)))
ref_scores = np.array([E_big[i] @ q_big / (np.linalg.norm(E_big[i]) * np.linalg.norm(q_big)) for i in range(50)])
check('similarities of the returned rows are in decreasing order', _safe(lambda: bool(np.all(np.diff(ref_scores[idx]) <= 0))))
check_hash('hidden: the five indices for the seeded table', idx, '__H_E5A__')
"""),
    explanation_md=r"""
Normalising the rows of $E$ and the query turns every cosine into a plain dot product, and the dot products of one vector with every row are exactly the matrix–vector product $\hat{E}\hat{\mathbf{q}}$ — one BLAS call instead of $n$ Python iterations. `np.argsort(-scores)[:k]` sorts in decreasing order (NumPy sorts ascending, hence the minus sign). Note that you do not even need to normalise the query to get the *ranking* right — dividing every score by the same positive constant changes nothing — but you do need to normalise the rows, otherwise long rows win. The toy table shows this: row 2 is twice as long as row 0 yet ranks second because its direction is slightly off.
""",
)

# ---------------------------------------------------------------- E6
ex.add(
    title="Projection and residual", difficulty=2,
    prompt_md=r"""
Implement `project(a, b)` returning a tuple `(p, r)` where `p` is the vector projection of `a` onto the line through `b` and `r = a - p` is the residual:

$$\mathbf{p} = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b}, \qquad \mathbf{r} = \mathbf{a} - \mathbf{p}.$$

Both outputs are NumPy arrays with the shape of `a`. Do **not** assume `b` is a unit vector.
""",
    starter=r"""
def project(a, b):
    # Return (p, r): projection of a onto b, and the residual a - p.
    # YOUR CODE HERE
    return None
""",
    solution=SOL_E6,
    check=fill(r"""
a, b = np.array([2., 3.]), np.array([4., 2.])
got = _safe(lambda: project(a, b))
check_close('p for a=(2,3), b=(4,2) is (2.8, 1.4)', _safe(lambda: got[0]), np.array([2.8, 1.4]))
check_close('r for a=(2,3), b=(4,2) is (-0.8, 1.6)', _safe(lambda: got[1]), np.array([-0.8, 1.6]))

g = np.random.default_rng(3); a5, b5 = g.normal(size=5), 3.0 * g.normal(size=5)
got5 = _safe(lambda: project(a5, b5))
check_close('p + r reconstructs a (5-D random case)', _safe(lambda: got5[0] + got5[1]), a5)
check_close('residual is orthogonal to b (r . b = 0)', _safe(lambda: float(got5[1] @ b5)), 0.0, tol=1e-9)
check_close('p is parallel to b (cross-check: p = (p.b / b.b) b)', _safe(lambda: got5[0]), _safe(lambda: (got5[0] @ b5) / (b5 @ b5) * b5))
p4 = _safe(lambda: project(np.array([1., 2., 3., 4.]), np.array([1., 1., 0., 1.]))[0])
check_hash('hidden: fraction of ||a||^2 explained by b for a=(1,2,3,4), b=(1,1,0,1)', _safe(lambda: float((p4 @ p4) / 30.0)), '__H_E6A__')
"""),
    explanation_md=r"""
The coefficient $t^\star = \mathbf{a}\cdot\mathbf{b}/\mathbf{b}\cdot\mathbf{b}$ is the minimiser of $\|\mathbf{a} - t\mathbf{b}\|^2$, and multiplying it by $\mathbf{b}$ gives the closest point on the line. The residual is orthogonal by construction: $\mathbf{r}\cdot\mathbf{b} = \mathbf{a}\cdot\mathbf{b} - t^\star\,\mathbf{b}\cdot\mathbf{b} = 0$. The classic mistake is to write $(\mathbf{a}\cdot\mathbf{b})\,\mathbf{b}$ without the denominator, which is only right when $\|\mathbf{b}\| = 1$ — the 5-D check with $\mathbf{b}$ scaled by 3 would then fail by a factor of 9. The explained fraction $\|\mathbf{p}\|^2/\|\mathbf{a}\|^2$ equals $\cos^2\theta$, the $R^2$ of a one-variable regression through the origin.
""",
)

# ---------------------------------------------------------------- E7
ex.add(
    title="Gram–Schmidt orthonormalisation", difficulty=2,
    prompt_md=r"""
Implement `gram_schmidt(A)` for a matrix `A` of shape `(m, n)` with $n \le m$ linearly independent columns $\mathbf{a}_1, \dots, \mathbf{a}_n$. Return `Q` of the same shape whose columns are orthonormal and span the same space, built in order:

$$\mathbf{v}_j = \mathbf{a}_j - \sum_{i<j} (\mathbf{v}_j\cdot\mathbf{q}_i)\,\mathbf{q}_i, \qquad \mathbf{q}_j = \mathbf{v}_j / \|\mathbf{v}_j\|.$$

Because you build from $\mathbf{a}_j$ and only subtract, each $\mathbf{q}_j$ satisfies $\mathbf{q}_j\cdot\mathbf{a}_j > 0$ (so the signs are determined — `np.linalg.qr` may differ by a sign). Loops over columns are fine; do not call `np.linalg.qr` inside.
""",
    starter=r"""
def gram_schmidt(A):
    # Columns of A -> orthonormal columns Q with the same span; q_j . a_j > 0.
    # YOUR CODE HERE
    return None
""",
    solution=SOL_E7,
    check=fill(r"""
A = np.column_stack([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.]])
Q = _safe(lambda: gram_schmidt(A))
check_close('Q^T Q = I for the 3x3 example', _safe(lambda: Q.T @ Q), np.eye(3), tol=1e-9)
check_close('first column is a1 / ||a1||', _safe(lambda: Q[:, 0]), np.array([1., 1., 0.]) / np.sqrt(2))
check_close('second column is (1, -1, 2) / sqrt(6)', _safe(lambda: Q[:, 1]), np.array([1., -1., 2.]) / np.sqrt(6))
check_close('agrees with np.linalg.qr up to column signs', _safe(lambda: np.abs(Q)), np.abs(np.linalg.qr(A)[0]), tol=1e-9)

A5 = np.random.default_rng(9).normal(size=(5, 3))
Q5 = _safe(lambda: gram_schmidt(A5))
check('shape is preserved for a 5x3 input', _safe(lambda: Q5.shape == (5, 3)))
check_close('Q^T Q = I for the 5x3 input', _safe(lambda: Q5.T @ Q5), np.eye(3), tol=1e-9)
check_close('same span: Q Q^T a = a for a column a of A', _safe(lambda: Q5 @ (Q5.T @ A5[:, 2])), A5[:, 2])
check('sign convention q_j . a_j > 0', _safe(lambda: bool(np.all(np.sum(Q5 * A5, axis=0) > 0))))
check_hash('hidden: sum of |Q| for the 5x3 input', _safe(lambda: float(np.abs(Q5).sum())), '__H_E7A__')
"""),
    explanation_md=r"""
Each step is the projection exercise applied repeatedly: subtracting $(\mathbf{v}\cdot\mathbf{q}_i)\,\mathbf{q}_i$ removes exactly the component along $\mathbf{q}_i$, and because the earlier $\mathbf{q}_i$ are mutually orthogonal, removing one component never re-introduces another. Normalising at the end makes the length 1. Using the running vector $\mathbf{v}$ (rather than the original $\mathbf{a}_j$) in each dot product is the *modified* Gram–Schmidt variant, which loses less orthogonality to rounding. The usual bugs: normalising before subtracting, subtracting projections onto the un-normalised $\mathbf{a}_i$ instead of $\mathbf{q}_i$, or mutating `A` in place (hence the `.copy()`).
""",
)

# ---------------------------------------------------------------- E8
ex.add(
    title="Signed distance to a hyperplane, vectorised", difficulty=2,
    prompt_md=r"""
Implement `signed_distance(X, w, b)` for a data matrix `X` of shape `(n, d)`, a normal vector `w` of shape `(d,)` and a scalar `b`. Return the array of shape `(n,)` of signed distances of every row to the hyperplane $\mathbf{w}\cdot\mathbf{x} + b = 0$:

$$\delta(\mathbf{x}) = \frac{\mathbf{w}\cdot\mathbf{x} + b}{\|\mathbf{w}\|}.$$

No Python loop over rows — one matrix–vector product.
""",
    starter=r"""
def signed_distance(X, w, b):
    # Signed distances of the rows of X to the hyperplane w.x + b = 0, shape (n,).
    # YOUR CODE HERE
    return None
""",
    solution=SOL_E8,
    check=fill(r"""
w, b = np.array([3., 4.]), -12.0
pts = np.array([[4., 3.], [0., 0.], [2., 1.5]])
got = _safe(lambda: signed_distance(pts, w, b))
check_close('distances for the guide example are (2.4, -2.4, 0)', got, np.array([2.4, -2.4, 0.0]))
check_close('rescaling (w, b) by 10 changes nothing', _safe(lambda: signed_distance(pts, 10 * w, 10 * b)), got)
check('output shape is (n,)', _safe(lambda: got.shape == (3,)))

P = np.random.default_rng(21).normal(size=(200, 3)); w3, b3 = np.array([1., -2., 2.]), 0.5
sd = _safe(lambda: signed_distance(P, w3, b3))
check_hash('hidden: number of points on the positive side', _safe(lambda: int(np.sum(sd > 0))), '__H_E8A__')
check_hash('hidden: distance of the point closest to the plane', _safe(lambda: float(np.min(np.abs(sd)))), '__H_E8B__')
"""),
    explanation_md=r"""
`X @ w` computes $\mathbf{w}\cdot\mathbf{x}$ for every row at once; adding `b` broadcasts the scalar; dividing by $\|\mathbf{w}\|$ converts raw scores into geometric distances. The sign tells you the side (positive where $\mathbf{w}$ points), and the invariance check matters conceptually: $(10\mathbf{w}, 10b)$ is the *same* hyperplane, so a classifier's confidence should be read from the distance, not from the raw score. Forgetting to divide by $\|\mathbf{w}\|$ gives the raw score — fine for the predicted label, wrong for the margin.
""",
)

# ---------------------------------------------------------------- E9 (conceptual)
ex.add_conceptual(
    title="Predict the effect of the square root of d_k in attention", difficulty=2,
    prompt_md=r"""
In a transformer, the attention weight of key $j$ for query $i$ is $\operatorname{softmax}_j(\mathbf{q}_i\cdot\mathbf{k}_j / \sqrt{d_k})$. Suppose the coordinates of $\mathbf{q}$ and $\mathbf{k}$ are independent with mean 0 and variance 1, and $d_k = 512$.

(a) What are the mean and standard deviation of the raw score $\mathbf{q}\cdot\mathbf{k}$? (b) Predict what the softmax over 20 keys looks like with raw scores, and what goes wrong for learning. (c) Why divide by $\sqrt{d_k}$ specifically, rather than by $d_k$?
""",
    answer_md=r"""
(a) $\mathbf{q}\cdot\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ is a sum of $d_k$ independent terms, each with mean $\mathbb{E}[q_i]\mathbb{E}[k_i] = 0$ and variance $\mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$. So the raw score has mean 0 and variance $d_k$, i.e. standard deviation $\sqrt{512} \approx 22.6$.

(b) Twenty raw scores spread over roughly $\pm 45$ differ from each other by tens of units; after exponentiation the largest dominates completely and the softmax is essentially one-hot. The attention weights then barely depend on the other keys, and the gradient through the softmax — whose Jacobian entries are proportional to $p_j(1 - p_j)$ — is nearly zero: the layer stops learning (saturation).

(c) Dividing by $\sqrt{d_k}$ brings the standard deviation back to exactly 1 regardless of $d_k$, so the softmax operates in its sensitive range for any head size. Dividing by $d_k$ would shrink the scores to standard deviation $1/\sqrt{d_k} \approx 0.044$: all weights become nearly equal and attention cannot discriminate between keys. $\sqrt{d_k}$ is the unique scale that removes the dependence on dimension — the same $1/\sqrt d$ that governs the cosine of random vectors in the lesson's simulation.
""",
)

# ---------------------------------------------------------------- E10 (stretch)
ex.add(
    title="Distance concentration experiment", difficulty=3,
    prompt_md=r"""
Implement `distance_concentration(d, n=500, seed=0)`:

1. Create `g = np.random.default_rng(seed)` and sample `n` points uniformly in the unit cube $[0, 1]^d$ with `g.random((n, d))`.
2. Compute the Euclidean distances from the **first** point to the other `n - 1` points.
3. Return the tuple `(contrast, rel_std_sq)` where `contrast = (max - min) / min` of those distances and `rel_std_sq` is the standard deviation divided by the mean of the **squared** distances.

Theory says `rel_std_sq` $\approx 1.18/\sqrt d$ and that `contrast` collapses towards 0 as $d$ grows. The checks test these qualitative facts, not exact numbers, so you may use `np.linalg.norm`.
""",
    starter=r"""
def distance_concentration(d, n=500, seed=0):
    # Return (contrast, rel_std_sq) for n uniform points in [0,1]^d; see the prompt for the exact recipe.
    # YOUR CODE HERE
    return None
""",
    solution=SOL_E10,
    check=fill(r"""
res = {d: _safe(lambda d=d: distance_concentration(d)) for d in (2, 10, 100, 1000)}
check('returns a pair (contrast, rel_std_sq)', _safe(lambda: len(res[2]) == 2))
check('in 2-D the farthest neighbour is more than twice as far as the nearest (contrast > 1)', _safe(lambda: res[2][0] > 1.0))
check('in 1000-D nearest and farthest differ by less than 35% (contrast < 0.35)', _safe(lambda: res[1000][0] < 0.35))
check('contrast decreases monotonically for d = 2, 10, 100, 1000', _safe(lambda: res[2][0] > res[10][0] > res[100][0] > res[1000][0]))
check('relative spread of squared distances at d = 100 is close to 1.18 / sqrt(100) = 0.118', _safe(lambda: 0.08 < res[100][1] < 0.16))
check('relative spread shrinks by roughly sqrt(10) from d = 100 to d = 1000', _safe(lambda: 2.0 < res[100][1] / res[1000][1] < 5.0))
"""),
    explanation_md=r"""
The squared distance is $\sum_{i=1}^d (x_i - y_i)^2$, a sum of $d$ independent terms with mean $1/6$ and variance $7/180$, so its mean grows like $d$ while its standard deviation grows like $\sqrt d$ — the relative spread is $\sqrt{7/180}\,/\,(1/6)\,/\sqrt d \approx 1.18/\sqrt d$. With all distances squeezed into a band of relative width $\sim 1/\sqrt d$, the gap between the nearest and the farthest neighbour collapses too: that is the `contrast` you measured. Two practical notes: `P[1:] - P[0]` broadcasts the first point against all others in one step, and distances from a single reference point are enough — using all $\binom{n}{2}$ pairs gives the same picture with more correlated samples.
""",
)

# ---------------------------------------------------------------- E11 (stretch)
ex.add(
    title="Reflect points across a hyperplane", difficulty=3,
    prompt_md=r"""
Implement `reflect(X, w, b)` returning, for every row $\mathbf{x}$ of `X` (shape `(n, d)`), its mirror image across the hyperplane $\mathbf{w}\cdot\mathbf{x} + b = 0$:

$$\mathbf{x}' = \mathbf{x} - 2\,\delta(\mathbf{x})\,\hat{\mathbf{w}}, \qquad \delta(\mathbf{x}) = \frac{\mathbf{w}\cdot\mathbf{x} + b}{\|\mathbf{w}\|}, \quad \hat{\mathbf{w}} = \frac{\mathbf{w}}{\|\mathbf{w}\|}.$$

Return an array of shape `(n, d)`; vectorise with broadcasting (hint: `delta[:, None]`). Reflecting twice must give back `X`, and the signed distance of the reflected points must be the negative of the original.
""",
    starter=r"""
def reflect(X, w, b):
    # Mirror image of each row of X across the hyperplane w.x + b = 0, shape (n, d).
    # YOUR CODE HERE
    return None
""",
    solution=SOL_E11,
    check=fill(r"""
w, b = np.array([3., 4.]), -12.0
pts = np.array([[4., 3.], [2., 1.5], [0., 0.]])
got = _safe(lambda: reflect(pts, w, b))
check_close('(4,3) reflects to (1.12, -0.84) across 3x + 4y = 12', _safe(lambda: got[0]), np.array([1.12, -0.84]))
check_close('a point on the plane is its own mirror image', _safe(lambda: got[1]), np.array([2., 1.5]))
check_close('reflecting twice is the identity', _safe(lambda: reflect(got, w, b)), pts)
_sd = lambda Z: (Z @ w + b) / np.linalg.norm(w)
check_close('signed distances flip sign', _safe(lambda: _sd(got)), -_sd(pts))

P = np.random.default_rng(33).normal(size=(20, 5)); w5, b5 = np.array([1., 2., 0., -1., 3.]), -0.7
R5 = _safe(lambda: reflect(P, w5, b5))
check('shape (20, 5) is preserved', _safe(lambda: R5.shape == (20, 5)))
check_close('reflection preserves distances between points (it is an isometry)', _safe(lambda: np.linalg.norm(R5[0] - R5[1])), np.linalg.norm(P[0] - P[1]))
check_hash('hidden: sum of all reflected coordinates (5-D seeded case)', _safe(lambda: float(R5.sum())), '__H_E11A__')
"""),
    explanation_md=r"""
Moving a point by $-\delta(\mathbf{x})\hat{\mathbf{w}}$ drops it onto the hyperplane (the foot of the perpendicular); moving it by twice that lands on the mirror image. The signed distance is a scalar per row, so `delta[:, None] * w_hat` broadcasts an `(n, 1)` column against the `(d,)` normal to produce the `(n, d)` correction. Reflections are isometries (they preserve all distances and angles) and are their own inverse; in matrix form the map is $I - 2\hat{\mathbf{w}}\hat{\mathbf{w}}^\top$, a *Householder reflection* — exactly what `np.linalg.qr` uses internally instead of Gram–Schmidt, which is why its signs can differ from yours in the previous exercise.
""",
)

# ---------------------------------------------------------------- E12 (mini-project)
PROJECT_STARTER = r"""
# ---- synthetic ratings data (keep this block as it is) -------------------------------------
g = np.random.default_rng(7)
n_users, n_items, n_factors = 80, 30, 3
U_true = g.normal(size=(n_users, n_factors))             # hidden "taste" of each user
V_true = g.normal(size=(n_items, n_factors))             # hidden "profile" of each item
user_bias = g.normal(scale=0.5, size=n_users)            # generous vs harsh raters
item_bias = g.normal(scale=0.5, size=n_items)            # popular vs unpopular items
R_full = np.clip(np.round(3.2 + user_bias[:, None] + item_bias[None, :]
                          + (U_true @ V_true.T) / np.sqrt(n_factors)
                          + g.normal(scale=0.4, size=(n_users, n_items))), 1, 5)
observed = g.random((n_users, n_items)) < 0.7            # about 70% of the ratings are known
test_mask = observed & (g.random((n_users, n_items)) < 0.2)
train_mask = observed & ~test_mask
R_train = np.where(train_mask, R_full, 0.0)              # 0 = unknown
print("train ratings:", train_mask.sum(), " test ratings:", test_mask.sum())

# ---- your work ------------------------------------------------------------------------------
# 1. baseline predictions for the test entries (global mean, then each user's mean) and their RMSE
# 2. user-user cosine similarity on mean-centred training rows (unknown entries contribute 0)
# 3. predict a held-out rating as the similarity-weighted average of the k most similar users who rated the item
# 4. RMSE of the recommender vs the baselines; plot RMSE against k
"""

PROJECT_SOLUTION = r"""
g = np.random.default_rng(7)
n_users, n_items, n_factors = 80, 30, 3
U_true = g.normal(size=(n_users, n_factors))
V_true = g.normal(size=(n_items, n_factors))
user_bias = g.normal(scale=0.5, size=n_users)
item_bias = g.normal(scale=0.5, size=n_items)
R_full = np.clip(np.round(3.2 + user_bias[:, None] + item_bias[None, :]
                          + (U_true @ V_true.T) / np.sqrt(n_factors)
                          + g.normal(scale=0.4, size=(n_users, n_items))), 1, 5)
observed = g.random((n_users, n_items)) < 0.7
test_mask = observed & (g.random((n_users, n_items)) < 0.2)
train_mask = observed & ~test_mask
R_train = np.where(train_mask, R_full, 0.0)
print("train ratings:", train_mask.sum(), " test ratings:", test_mask.sum())

def rmse(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)))

test_u, test_i = np.where(test_mask)
truth = R_full[test_mask]

# --- baselines ---------------------------------------------------------------------------------
global_mean = R_train[train_mask].mean()
counts = train_mask.sum(axis=1)
user_mean = np.where(counts > 0, R_train.sum(axis=1) / np.maximum(counts, 1), global_mean)
print("RMSE global mean : %.3f" % rmse(np.full(len(truth), global_mean), truth))
print("RMSE user mean   : %.3f" % rmse(user_mean[test_u], truth))

# --- user-user cosine similarity on mean-centred rows ------------------------------------------
C = np.where(train_mask, R_train - user_mean[:, None], 0.0)      # centred, unknown -> 0
norms = np.linalg.norm(C, axis=1, keepdims=True)
Cn = C / np.where(norms == 0, 1.0, norms)
S = Cn @ Cn.T                                                     # (n_users x n_users) cosine matrix
np.fill_diagonal(S, 0.0)                                          # a user is not their own neighbour

def predict(u, i, k):
    rated = train_mask[:, i].copy(); rated[u] = False             # users who rated item i (not u)
    cand = np.where(rated)[0]
    if len(cand) == 0:
        return user_mean[u]
    sims = S[u, cand]
    top = cand[np.argsort(-sims)[:k]]
    w = S[u, top]
    w = np.where(w > 0, w, 0.0)                                   # ignore dissimilar users
    if w.sum() == 0:
        return user_mean[u]
    # weighted average of the neighbours' *deviations* from their own means, added to u's mean
    return user_mean[u] + np.sum(w * (R_train[top, i] - user_mean[top])) / w.sum()

ks = [1, 2, 3, 5, 8, 12, 20, 40, n_users - 1]
scores = []
for k in ks:
    pred = np.clip([predict(u, i, k) for u, i in zip(test_u, test_i)], 1, 5)
    scores.append(rmse(np.array(pred), truth))
    print("k = %2d neighbours: RMSE = %.3f" % (k, scores[-1]))

fig, ax = plt.subplots()
ax.plot(ks, scores, "o-", label="user-user cosine CF")
ax.axhline(rmse(user_mean[test_u], truth), color="C1", ls="--", label="user-mean baseline")
ax.axhline(rmse(np.full(len(truth), global_mean), truth), color="C3", ls=":", label="global-mean baseline")
ax.set_xlabel("k (number of neighbours)"); ax.set_ylabel("RMSE on held-out ratings"); ax.set_title("Collaborative filtering with cosine similarity")
ax.legend(); ax.grid(alpha=0.3)
plt.show()
"""

ex.add_project(
    title="Mini-project — a tiny recommender from cosine similarity", difficulty=3,
    prompt_md=r"""
Build a user-based collaborative-filtering recommender on a synthetic users × items ratings matrix and evaluate it honestly.

The starter cell generates the data: 80 users, 30 items, ratings 1–5 of which about 70% are observed; roughly a fifth of the observed ratings are hidden in `test_mask` for evaluation, the rest are in `train_mask` / `R_train` (0 marks "unknown"). The ratings come from hidden user tastes and item profiles (a dot product, as in Section 3 of the guide) plus a per-user generosity bias, a per-item popularity bias and noise.

1. **Baselines.** Predict every held-out rating by the global mean of the training ratings, then by each user's own mean. Report the RMSE of each.
2. **Similarity.** Mean-centre each user's training row (subtract that user's mean from their known ratings; unknown entries stay 0) and compute the full user–user cosine-similarity matrix with one matrix product.
3. **Prediction.** For a held-out pair (user $u$, item $i$), take the $k$ most similar users who *have* rated item $i$ and predict $u$'s own mean plus the similarity-weighted average of those neighbours' deviations from their means. Fall back to the user mean when no neighbour qualifies. Clip to $[1, 5]$.
4. **Evaluation.** Compute the RMSE on the held-out ratings for several $k$ (e.g. 1, 2, 3, 5, 8, 12, 20, 40, all) and plot RMSE against $k$ with the two baselines as horizontal lines.
""",
    rubric_md=r"""
- Compute both baselines and state their RMSE before touching similarities — a recommender that does not beat the user-mean baseline is not working.
- Explain why the rows are mean-centred before computing cosine (it turns cosine into a Pearson-style correlation and stops "generous raters" looking similar to everyone).
- Obtain the whole similarity matrix with one matrix product of row-normalised data, and zero the diagonal so a user is not their own neighbour.
- Use only neighbours who actually rated the item, weight by similarity, and handle the empty-neighbour case without crashing.
- Produce the RMSE-versus-$k$ plot with titled axes and read it: very small $k$ is noisy, very large $k$ averages in dissimilar users, the best $k$ is in between.
- Say what you would change for real data (sparsity, shrinkage of similarities computed on few co-rated items, item–item variant, matrix factorisation — Module 03).
""",
    solution_sketch_md=r"""
Mean-centring makes the cosine between two users depend only on how they deviate from their personal averages, so a user who rates everything 4–5 and one who rates everything 2–3 but with the same *pattern* become similar. With zeros for unknown entries, $\hat{C}\hat{C}^\top$ gives every pairwise cosine in one product (unrated items contribute nothing). Prediction adds the similarity-weighted average of the neighbours' centred ratings back onto the target user's mean; only positively similar users are used. On this data the user-mean baseline improves a little on the global mean (it captures the generosity bias), and the recommender beats both clearly for moderate $k$ (RMSE around 0.74 at $k \approx 8$ versus 1.10 for the user mean); the curve rises again for large $k$ as dissimilar users are averaged in, and $k = 1$ is noisy because a single neighbour's rating carries all the weight. The remaining error comes from the rounding noise in the ratings and from the item bias, which user-based CF only captures indirectly — an item–item variant or matrix factorisation (Module 03) would pick it up.
""",
    solution_code=PROJECT_SOLUTION,
    starter=PROJECT_STARTER,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = {}
for it in ex.items:
    kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
print("exercises:", len(ex.items), kinds)
print("hashes:", H)
