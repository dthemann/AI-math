#!/usr/bin/env python3
"""Builder for Module 03 — Eigenvalues, SVD and PCA.

Generates lesson.ipynb, exercises.ipynb and solutions.ipynb into
part-1-linear-algebra/03-eigenvalues-svd-and-pca. Hidden answers for the
exercise self-checks are computed here by *executing the solution code*.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import Notebook, ExerciseSet, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-1-linear-algebra/03-eigenvalues-svd-and-pca"


def safe_to_hash(x: float, decimals: int = 3, margin: float = 2e-4) -> float:
    """Assert that x is not within `margin` of a rounding boundary at `decimals` places."""
    frac = abs(x) * 10 ** decimals
    dist = abs(frac - np.floor(frac) - 0.5)   # distance (in units of the last digit) to the .5 boundary
    assert dist > margin * 10 ** decimals, f"hidden value {x!r} too close to a rounding boundary"
    return x

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 03 — Eigenvalues, SVD and PCA")

nb.md(r"""
This notebook walks through the ideas of the module with code you can edit and re-run:
eigenvectors as the directions a matrix only stretches, power iteration and PageRank, the
spectral theorem, the singular value decomposition, low-rank approximation and image
compression, PCA from scratch on the Iris data, the pseudo-inverse, and the condition number.

Read the [concept guide](README.md) first, or keep it open alongside — every section here
mirrors a numbered section there. Code cells build each object **from scratch in NumPy**
first and then check against the library call that does the same.
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
## 1. Eigenvectors and eigenvalues — the directions a matrix only stretches

A square matrix $A$ moves every vector. A vector $v \ne 0$ with

$$A v = \lambda v$$

is only *scaled* (by the eigenvalue $\lambda$), never turned. For a $2\times2$ matrix the
eigenvalues solve the characteristic polynomial $\lambda^2 - \operatorname{tr}(A)\lambda + \det(A) = 0$.

Below we take $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$, solve the polynomial by hand,
compare with `np.linalg.eig`, and draw what $A$ does to a grid. Watch the two coloured arrows:
they are the eigenvectors, and they come out pointing the same way they went in.
""")

nb.code(r"""
A = np.array([[4., 1.],
              [2., 3.]])

# --- eigenvalues by hand: lambda^2 - tr*lambda + det = 0
tr, det = np.trace(A), np.linalg.det(A)
disc = np.sqrt(tr**2 - 4 * det)
lam_hand = np.array([(tr + disc) / 2, (tr - disc) / 2])
print(f"characteristic polynomial:  lambda^2 - {tr:.0f} lambda + {det:.0f} = 0")
print("eigenvalues by hand:       ", lam_hand)

# --- NumPy
w, V = np.linalg.eig(A)
order = np.argsort(-w.real)              # sort descending so the two agree
w, V = w[order].real, V[:, order].real
print("np.linalg.eig eigenvalues: ", w)
print("eigenvectors (unit length, as columns):")
print(V)
for i in range(2):
    print(f"  A v{i+1} = {A @ V[:, i]}   lambda{i+1} v{i+1} = {w[i] * V[:, i]}")
print("angle between eigenvectors: %.1f degrees  (not orthogonal: A is not symmetric)"
      % np.degrees(np.arccos(abs(V[:, 0] @ V[:, 1]))))

# --- the grid picture
ts = np.linspace(-1, 1, 11)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, M, title, lim in [(axes[0], np.eye(2), "before: unit grid", 1.6),
                          (axes[1], A, "after: A applied to every point", 6.5)]:
    for t in ts:
        for seg in (np.array([[t, t], [-1, 1]]), np.array([[-1, 1], [t, t]])):
            p = M @ seg
            ax.plot(p[0], p[1], color="lightgray", lw=0.8)
    g = M @ np.array([1., 0.])          # a generic vector: it gets turned
    ax.annotate("", xy=g, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:green", lw=2))
    ax.text(g[0] * 1.1, g[1] + 0.15, "e1 (generic)", color="tab:green")
    for i, c in enumerate(["tab:red", "tab:blue"]):
        u = M @ V[:, i]
        ax.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        ax.text(u[0] * 1.12, u[1] * 1.12, f"v{i+1} (lambda={w[i]:.0f})", color=c)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(title); ax.set_xlabel("x1"); ax.set_ylabel("x2")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Notice that the green vector $e_1$ is rotated as well as stretched, while the red and blue
eigenvectors keep their direction: $v_1 = (1, 1)/\sqrt2$ is stretched by $5$, and
$v_2 = (1, -2)/\sqrt5$ by $2$. The grid squares become parallelograms whose area is
multiplied by $\det A = 10 = \lambda_1 \lambda_2$. The two eigenvector *lines* meet at about
72 degrees, not 90 — orthogonality is a privilege reserved for symmetric matrices (Section 3).

**Try it yourself:** replace `A` with the rotation matrix `np.array([[0., -1.], [1., 0.]])`
(remove the `.real` calls first). The eigenvalues come out as $\pm i$ — no real direction
survives a rotation, so complex eigenvalues always signal a rotational component.
""")

# ---------------------------------------------------------------- Section 2
nb.md(r"""
## 2. Diagonalisation, matrix powers and power iteration

With eigenvectors as columns of $P$ and eigenvalues in a diagonal $D$,
$A = PDP^{-1}$ and $A^k = PD^kP^{-1}$. Writing $x = \sum_i c_i v_i$,

$$A^k x = \sum_i c_i \lambda_i^k v_i ,$$

so the dominant eigenvalue takes over: $A^k x$ lines up with $v_1$, and the angle to $v_1$
shrinks like $|\lambda_2/\lambda_1|^k$. **Power iteration** exploits this: repeat
$x \leftarrow Ax / \|Ax\|$ and estimate $\lambda_1$ by the Rayleigh quotient $x^\top A x$.
""")

nb.code(r"""
def power_iteration(A, n_iter=60, x0=None):
    x = np.ones(A.shape[0]) if x0 is None else np.asarray(x0, float)
    x = x / np.linalg.norm(x)
    estimates = []
    for _ in range(n_iter):
        y = A @ x
        x = y / np.linalg.norm(y)
        estimates.append(x @ A @ x)          # Rayleigh quotient
    return estimates[-1], x, np.array(estimates)

lam_est, v_est, hist = power_iteration(A, n_iter=40, x0=[1., 0.])
print("power iteration:  lambda_1 ~ %.10f,  v_1 ~ %s" % (lam_est, v_est))
print("np.linalg.eig:    lambda_1 = %.10f,  v_1 = %s" % (w[0], V[:, 0]))

err = np.abs(hist - w[0])
k = np.arange(1, len(hist) + 1)
plt.semilogy(k, err, "o-", label="|estimate - lambda_1|")
plt.semilogy(k, err[0] * (w[1] / w[0]) ** (k - 1), "--", label="C * (lambda_2/lambda_1)^k")
plt.title("Power iteration converges geometrically at rate lambda_2/lambda_1 = 0.4")
plt.xlabel("iteration k"); plt.ylabel("error in eigenvalue estimate"); plt.legend(); plt.show()
""")

nb.md(r"""
The error falls by a factor of $0.4 = \lambda_2/\lambda_1$ per step until it hits
floating-point precision near $10^{-16}$. If the two largest eigenvalues were close
(ratio 0.99) the same picture would need hundreds of iterations — the *gap* controls the speed.

### A two-state Markov chain

A column-stochastic matrix $T$ (columns sum to one) has eigenvalue $1$, and its eigenvector —
scaled to sum to one — is the stationary distribution $\pi$. Repeatedly applying $T$ *is*
power iteration, and the second eigenvalue sets how fast the chain forgets its start.
""")

nb.code(r"""
T = np.array([[0.9, 0.5],     # column j = where brand j's customers go next month
              [0.1, 0.5]])
w_T, P = np.linalg.eig(T)
print("eigenvalues of T:", w_T.real)
pi = P[:, np.argmax(w_T.real)].real
pi = pi / pi.sum()
print("stationary distribution pi =", pi, "  (5/6, 1/6)")

x = np.array([0., 1.])                      # everyone starts with brand B
print("\nmonth   share A   share B   distance to pi")
for month in range(1, 11):
    x = T @ x
    print(f"{month:5d}   {x[0]:7.4f}   {x[1]:7.4f}   {np.abs(x - pi).sum():.2e}")

# A^k via the diagonalisation: P D^k P^-1  equals the direct matrix power
D = np.diag(w_T.real)
Tk_diag = P.real @ np.diag(w_T.real ** 10) @ np.linalg.inv(P.real)
print("\nmax |P D^10 P^-1 - T^10| =", np.abs(Tk_diag - np.linalg.matrix_power(T, 10)).max())
print("0.4^10 =", 0.4 ** 10, " -> the second eigenvalue is the speed of forgetting")
""")

nb.md(r"""
Both brands end at the shares $(5/6, 1/6)$ whatever the starting split, and the distance to
$\pi$ shrinks by exactly $0.4$ per month — the second eigenvalue.

### PageRank on a six-page web

PageRank is the stationary distribution of a random surfer who follows a random out-link
with probability $d = 0.85$ and jumps to a uniformly random page otherwise. With $M$ the
column-stochastic link matrix (a page with no out-links is treated as linking to every page),

$$G = d\,M + \frac{1-d}{n}\,\mathbf{1}\mathbf{1}^\top, \qquad \pi = G\pi .$$

Every entry of $G$ is positive, so the stationary vector is unique and $|\lambda_2| \le d$:
power iteration converges at least as fast as $0.85^k$, however large the web.
""")

nb.code(r"""
links = {0: [1, 2], 1: [2], 2: [0], 3: [0, 2], 4: [3, 5], 5: []}   # page 5 is a dead end
n = len(links)

M = np.zeros((n, n))                       # M[i, j] = prob. of going j -> i
for j, outs in links.items():
    if outs:
        M[outs, j] = 1.0 / len(outs)
    else:
        M[:, j] = 1.0 / n                  # dangling page: jump anywhere
d = 0.85
G = d * M + (1 - d) / n * np.ones((n, n))
print("columns of G sum to one:", np.allclose(G.sum(axis=0), 1))

x = np.ones(n) / n
for k in range(1, 200):
    x_new = G @ x
    if np.abs(x_new - x).sum() < 1e-12:
        break
    x = x_new
pagerank = x_new
print(f"converged in {k} iterations (0.85^{k} = {0.85**k:.1e})")
print("PageRank:", pagerank, " sum =", pagerank.sum())
print("ranking (best first):", np.argsort(-pagerank))

# cross-check with the eigenvector for eigenvalue 1
w_G, V_G = np.linalg.eig(G)
v1 = V_G[:, np.argmin(np.abs(w_G - 1))].real
print("max |power iteration - eig| =", np.abs(pagerank - v1 / v1.sum()).max())
print("second-largest |eigenvalue| of G = %.3f  (<= d = %.2f)" % (np.sort(np.abs(w_G))[-2], d))

plt.bar(range(n), pagerank, color="tab:blue")
plt.title("PageRank of a six-page web (d = 0.85)")
plt.xlabel("page"); plt.ylabel("stationary probability"); plt.show()
""")

nb.md(r"""
Pages 2 and 0 share the top with about 35 % each. Page 2 is linked to by three pages (0, 1
and 3); page 0 is linked to by *one* page only — but that page is page 2, the most important
one, and it sends page 0 its entire out-link weight. That is the whole idea of PageRank: a
link is worth what its source is worth. Pages 3, 4 and 5 are linked to from nobody or only
from the weak page 4, so they live mostly on the teleportation term $(1-d)/n$.

**Try it yourself:** set `d = 0.5` and then `d = 0.99`. Convergence speeds up or slows down
with $d$, and as $d \to 1$ the loop 0→1→2→0 soaks up almost all of the probability mass while
the dead-end page 5 and its only referrer, page 4, drop towards zero.
""")

# ---------------------------------------------------------------- Section 3
nb.md(r"""
## 3. Symmetric matrices and the spectral theorem

For a symmetric matrix ($A = A^\top$) the eigenvalues are real and the eigenvectors can be
chosen orthonormal:

$$A = Q\Lambda Q^\top = \sum_i \lambda_i q_i q_i^\top, \qquad Q^\top Q = I .$$

In the eigen-coordinates $y = Q^\top x$ the quadratic form becomes a weighted sum of squares,
$x^\top A x = \sum_i \lambda_i y_i^2$, so $A$ is positive semidefinite exactly when every
$\lambda_i \ge 0$. The level set $x^\top A x = 1$ is an ellipse whose axes are the
eigenvectors and whose semi-axis lengths are $1/\sqrt{\lambda_i}$.
""")

nb.code(r"""
S = np.array([[2., 1.],
              [1., 2.]])
lam, Q = np.linalg.eigh(S)                  # eigh: symmetric input -> real, ascending
lam, Q = lam[::-1], Q[:, ::-1]              # descending
print("eigenvalues:", lam)
print("Q (orthonormal eigenvectors as columns):"); print(Q)
print("Q^T Q = I ?", np.allclose(Q.T @ Q, np.eye(2)))
recon = sum(lam[i] * np.outer(Q[:, i], Q[:, i]) for i in range(2))
print("sum_i lambda_i q_i q_i^T reconstructs S ?", np.allclose(recon, S))

# the ellipse x^T S x = 1 drawn via eigen-coordinates
theta = np.linspace(0, 2 * np.pi, 400)
y = np.vstack([np.cos(theta) / np.sqrt(lam[0]), np.sin(theta) / np.sqrt(lam[1])])
x = Q @ y
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(x[0], x[1], label="x^T S x = 1")
for i, c in enumerate(["tab:red", "tab:blue"]):
    a = Q[:, i] / np.sqrt(lam[i])
    ax.annotate("", xy=a, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax.text(a[0] * 1.15, a[1] * 1.15, f"q{i+1}, length 1/sqrt({lam[i]:.0f})", color=c)
ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
ax.set_title("Level set of the quadratic form: axes = eigenvectors")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend(loc="lower right"); plt.show()
""")

nb.md(r"""
The ellipse is short along $q_1 = (1,1)/\sqrt2$ (semi-axis $1/\sqrt3 \approx 0.58$): the form
grows fastest there, because $\lambda_1 = 3$ is the larger eigenvalue. Now the same facts
on a larger random symmetric matrix, and the two things we most often need in ML: a
covariance matrix is PSD, and the Rayleigh quotient is squeezed between $\lambda_{\min}$ and
$\lambda_{\max}$.
""")

nb.code(r"""
B = rng.normal(size=(5, 5))
S5 = B @ B.T                                # Gram matrices are symmetric PSD
lam5, Q5 = np.linalg.eigh(S5)
print("eigenvalues of B B^T:", lam5, "-> all >= 0, so PSD")
print("max |Q^T Q - I| = %.1e,   max |Q L Q^T - S| = %.1e"
      % (np.abs(Q5.T @ Q5 - np.eye(5)).max(), np.abs(Q5 @ np.diag(lam5) @ Q5.T - S5).max()))

# a covariance matrix of real-looking data is PSD too
data = rng.normal(size=(200, 4)) @ rng.normal(size=(4, 4))
C = np.cov(data, rowvar=False)
print("eigenvalues of a covariance matrix:", np.linalg.eigvalsh(C))

# Rayleigh quotient of 10,000 random directions never leaves [lambda_min, lambda_max]
X = rng.normal(size=(10000, 5))
rq = np.einsum("ij,jk,ik->i", X, S5, X) / np.einsum("ij,ij->i", X, X)
print("Rayleigh quotients: min %.4f, max %.4f;  eigenvalues: min %.4f, max %.4f"
      % (rq.min(), rq.max(), lam5.min(), lam5.max()))
""")

nb.md(r"""
The random Rayleigh quotients stay strictly inside $[\lambda_{\min}, \lambda_{\max}]$ and
never quite reach the ends — the extremes are attained only *at* the eigenvectors. That
"maximise $w^\top C w$ over unit $w$" problem is exactly what PCA solves in Section 6.
""")

# ---------------------------------------------------------------- Section 4
nb.md(r"""
## 4. The singular value decomposition

Any $m\times n$ matrix factors as

$$A = U\Sigma V^\top = \sum_{i=1}^{r} \sigma_i u_i v_i^\top, \qquad A v_i = \sigma_i u_i ,$$

with orthonormal $u_i$ (left singular vectors), orthonormal $v_i$ (right singular vectors)
and $\sigma_1 \ge \sigma_2 \ge \dots > 0$. The $v_i$ are eigenvectors of $A^\top A$, the $u_i$
eigenvectors of $AA^\top$, and $\sigma_i^2$ are their shared eigenvalues. Geometrically:
rotate ($V^\top$), stretch along the axes ($\Sigma$), rotate ($U$) — so the unit circle
always becomes an ellipse with semi-axes $\sigma_i u_i$.
""")

nb.code(r"""
A2 = np.array([[3., 0.],
               [4., 5.]])
U, s, Vt = np.linalg.svd(A2)
print("singular values:", s, "   by hand: sqrt(45), sqrt(5) =", np.sqrt(45), np.sqrt(5))
print("sqrt of eigenvalues of A^T A:", np.sqrt(np.linalg.eigvalsh(A2.T @ A2)[::-1]))
print("eigenvalues of A itself:   ", np.sort(np.linalg.eigvals(A2).real)[::-1], "(different!)")
print("sigma_1 * sigma_2 = %.4f = |det A| = %.4f" % (s.prod(), abs(np.linalg.det(A2))))
print("U^T U = I:", np.allclose(U.T @ U, np.eye(2)), "  V^T V = I:", np.allclose(Vt @ Vt.T, np.eye(2)))
print("U S V^T reconstructs A:", np.allclose(U @ np.diag(s) @ Vt, A2))
for i in range(2):
    print(f"A v{i+1} = {A2 @ Vt[i]}   sigma{i+1} u{i+1} = {s[i] * U[:, i]}")

theta = np.linspace(0, 2 * np.pi, 400)
circle = np.vstack([np.cos(theta), np.sin(theta)])
image = A2 @ circle
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].plot(circle[0], circle[1]); axes[0].set_title("input: unit circle with v1, v2")
axes[1].plot(image[0], image[1]); axes[1].set_title("output: ellipse with semi-axes sigma_i u_i")
for i, c in enumerate(["tab:red", "tab:blue"]):
    axes[0].annotate("", xy=Vt[i], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    axes[0].text(Vt[i][0] * 1.15, Vt[i][1] * 1.15, f"v{i+1}", color=c)
    axes[1].annotate("", xy=s[i] * U[:, i], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    axes[1].text(s[i] * U[0, i] * 1.1, s[i] * U[1, i] * 1.1, f"sigma{i+1} u{i+1}", color=c)
for ax, lim in zip(axes, (1.5, 7.5)):
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The circle's diameters along $v_1$ and $v_2$ map to the ellipse's axes: $v_1$ goes to
$\sigma_1 u_1$ (length 6.71) and $v_2$ to $\sigma_2 u_2$ (length 2.24). The eigenvalues of
this matrix, 5 and 3, are *not* its singular values — only the product agrees. Signs of the
singular vectors are arbitrary (NumPy may flip a pair), which never changes the picture.

### Singular values reveal rank — even under noise

A matrix built from three rank-one pieces has exactly three non-zero singular values. Add
noise and the zeros become small-but-non-zero: the *scree plot* shows a cliff after the true
rank, which is how one decides how many components to keep.
""")

nb.code(r"""
m, n_cols, r_true = 60, 40, 3
L = rng.normal(size=(m, r_true)) @ rng.normal(size=(r_true, n_cols))   # exact rank 3
noise = 0.3 * rng.normal(size=(m, n_cols))
s_clean = np.linalg.svd(L, compute_uv=False)
s_noisy = np.linalg.svd(L + noise, compute_uv=False)
print("rank of L:", np.linalg.matrix_rank(L), "   rank of L + noise:", np.linalg.matrix_rank(L + noise))
print("first 6 singular values, clean:", s_clean[:6])
print("first 6 singular values, noisy:", s_noisy[:6])

plt.semilogy(np.arange(1, 41), s_clean + 1e-16, "o-", label="exact rank-3 matrix")
plt.semilogy(np.arange(1, 41), s_noisy, "s-", label="rank-3 + noise (sd 0.3)")
plt.title("Scree plot: singular values of a 60 x 40 matrix")
plt.xlabel("index i"); plt.ylabel("sigma_i (log scale)"); plt.legend(); plt.show()
""")

nb.md(r"""
The clean matrix has three large singular values and the rest are at the level of rounding
error ($10^{-14}$ — `matrix_rank` correctly says 3). With noise, every singular value is
non-zero (`matrix_rank` now says 40), but the plot still shows a cliff: $\sigma_3 \approx 35$
against $\sigma_4 \approx 4$. The largest singular value of a pure-noise matrix with entries
of standard deviation $\tau$ is roughly $\tau(\sqrt{m} + \sqrt{n})$ — here
$0.3\,(\sqrt{60} + \sqrt{40}) \approx 4.2$ — a useful rule of thumb for deciding whether a
singular value is "signal".

**Try it yourself:** raise the noise standard deviation to `1.5`. At what level does the
cliff disappear, so that the true rank is no longer readable from the plot?
""")

# ---------------------------------------------------------------- Section 5
nb.md(r"""
## 5. Low-rank approximation — image compression

Keeping the $k$ largest terms of the SVD gives $A_k = \sum_{i \le k} \sigma_i u_i v_i^\top$,
and the **Eckart–Young theorem** says no rank-$k$ matrix is closer:

$$\|A - A_k\|_F = \sqrt{\textstyle\sum_{i>k}\sigma_i^2}, \qquad \|A - A_k\|_2 = \sigma_{k+1} .$$

Storing $A_k$ costs $k(m + n + 1)$ numbers instead of $mn$. We compress a synthetic $64\times64$
image and check the theorem's error formula against the directly measured error.
""")

nb.code(r"""
yy, xx = np.mgrid[0:64, 0:64]
img = 0.3 * xx / 63 + 0.2 * yy / 63                       # smooth gradient (rank 2)
img = img + 0.6 * (((xx - 42) ** 2 + (yy - 22) ** 2) < 12 ** 2)   # a disc
img[12:52, 8:16] += 0.5                                    # a vertical bar
img = img + 0.12 * np.sin((xx + yy) / 4.0)                 # diagonal stripes (rank 2)
img = img + 0.02 * rng.normal(size=img.shape)              # a little sensor noise

U, s, Vt = np.linalg.svd(img)
def rank_k(U, s, Vt, k):
    return (U[:, :k] * s[:k]) @ Vt[:k]

ks = [1, 2, 5, 10, 20]
fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
axes = axes.ravel()
axes[0].imshow(img, cmap="gray"); axes[0].set_title("original (rank %d)" % np.linalg.matrix_rank(img))
for ax, k in zip(axes[1:], ks):
    ax.imshow(rank_k(U, s, Vt, k), cmap="gray", vmin=img.min(), vmax=img.max())
    ax.set_title(f"rank {k}: {k * (64 + 64 + 1) / 64**2:.0%} of storage")
for ax in axes:
    ax.set_xlabel("column"); ax.set_ylabel("row"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Truncated-SVD compression of a 64 x 64 image"); plt.tight_layout(); plt.show()
""")

nb.code(r"""
total_energy = (s ** 2).sum()
ks_all = np.arange(1, 65)
measured = np.array([np.linalg.norm(img - rank_k(U, s, Vt, k)) for k in ks_all])
eckart_young = np.sqrt(np.cumsum(s[::-1] ** 2)[::-1][1:].tolist() + [0.0])   # sqrt(sum_{i>k} sigma_i^2)
print("max |measured - Eckart-Young formula| =", np.abs(measured - eckart_young).max())
for k in ks:
    print(f"k = {k:2d}: relative Frobenius error {measured[k-1] / np.linalg.norm(img):.3f},"
          f"  energy captured {(s[:k]**2).sum() / total_energy:.1%}")

plt.plot(ks_all, measured / np.linalg.norm(img), "o-", label="measured ||A - A_k||_F / ||A||_F")
plt.plot(ks_all, eckart_young / np.linalg.norm(img), "--", label="sqrt(sum_{i>k} sigma_i^2) / ||A||_F")
plt.title("Approximation error vs rank: the Eckart-Young formula is exact")
plt.xlabel("rank k"); plt.ylabel("relative Frobenius error"); plt.legend(); plt.show()
""")

nb.md(r"""
Rank 1 is a blur of row and column averages; rank 5 shows every shape; rank 10 is visually
indistinguishable from the original at 31 % of the storage. The measured error curve sits
exactly on the Eckart–Young formula — the theorem is not an approximation. The clean image
has rank 12 (a gradient of rank 2, stripes of rank 2, a disc and a bar), so the singular
values beyond the twelfth are pure sensor noise; the error curve bends at $k \approx 12$ and
then declines slowly, and truncating there actually *denoises* the picture.
""")

# ---------------------------------------------------------------- Section 6
nb.md(r"""
## 6. PCA from scratch — and against scikit-learn

For a centred data matrix $X_c$ ($n$ rows, $d$ features), the principal components are the
eigenvectors of the covariance $C = X_c^\top X_c/(n-1)$, or equivalently the right singular
vectors of $X_c$ with $\lambda_i = \sigma_i^2/(n-1)$. The explained variance ratio is
$\lambda_i / \sum_j \lambda_j$; scores are $Z = X_c V_k$; the reconstruction is
$\hat X = Z V_k^\top + \mu$.

We do all of it on the Iris data (150 flowers, 4 measurements in cm) both ways, then call
`sklearn.decomposition.PCA` and confirm it agrees to machine precision.
""")

nb.code(r"""
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X, y_lab = iris.data, iris.target
n_samples, d = X.shape
mu = X.mean(axis=0)
Xc = X - mu

# route 1: eigenvectors of the covariance matrix
C = Xc.T @ Xc / (n_samples - 1)
lam, W = np.linalg.eigh(C)
lam, W = lam[::-1], W[:, ::-1]
evr_cov = lam / lam.sum()

# route 2: SVD of the centred data (never forms C)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
lam_svd = s ** 2 / (n_samples - 1)
evr_svd = lam_svd / lam_svd.sum()

# route 3: scikit-learn
pca = PCA().fit(X)

print("explained variance ratio, covariance route:", evr_cov)
print("explained variance ratio, SVD route:       ", evr_svd)
print("explained variance ratio, sklearn:         ", pca.explained_variance_ratio_)
print("components agree up to sign (cov vs sklearn):", np.allclose(np.abs(W.T), np.abs(pca.components_)))
print("components agree up to sign (SVD vs sklearn):", np.allclose(np.abs(Vt), np.abs(pca.components_)))
print("first component (loadings on sepal len, sepal wid, petal len, petal wid):", Vt[0])

k = 2
Z = Xc @ Vt[:k].T                         # scores = new coordinates
X_hat = Z @ Vt[:k] + mu                   # rank-2 reconstruction
resid = ((X - X_hat) ** 2).sum()
print(f"\nk = {k}: variance explained {evr_svd[:k].sum():.2%}")
print("total squared reconstruction error %.4f  =  (n-1) * sum of dropped eigenvalues %.4f"
      % (resid, (n_samples - 1) * lam_svd[k:].sum()))

for cls, name in enumerate(iris.target_names):
    plt.scatter(Z[y_lab == cls, 0], Z[y_lab == cls, 1], s=18, label=name)
plt.title("Iris in the first two principal components")
plt.xlabel(f"PC1 ({evr_svd[0]:.1%} of variance)"); plt.ylabel(f"PC2 ({evr_svd[1]:.1%} of variance)")
plt.legend(); plt.show()
""")

nb.md(r"""
All three routes give the same numbers: PC1 carries 92.5 % of the variance and PC2 another
5.3 %, so the 2-D picture keeps 97.8 % of the spread of the 4-D cloud. The loadings of PC1
are all the same sign and largest on petal length: the first component is essentially
"overall flower size". The three species separate well on PC1 — but remember PCA never saw
the labels; it found the direction of largest *variance*, which here happens to coincide with
species.

### What happens without scaling

PCA is not scale-invariant. Suppose one lab recorded petal length in micrometres
(a factor 10,000 larger) rather than centimetres.
""")

nb.code(r"""
X_bad = X.copy()
X_bad[:, 2] *= 10_000                       # petal length now in micrometres
evr_bad = PCA().fit(X_bad).explained_variance_ratio_
comp_bad = PCA().fit(X_bad).components_[0]
print("explained variance ratio with one feature in micrometres:", evr_bad)
print("first component:", comp_bad, "-> it is just the petal-length axis")

X_std = (X_bad - X_bad.mean(axis=0)) / X_bad.std(axis=0, ddof=1)   # standardise every column
evr_std = PCA().fit(X_std).explained_variance_ratio_
print("\nafter standardising each column:", evr_std)
print("first component:", PCA().fit(X_std).components_[0])
print("(the same as standardising the original cm data:", np.allclose(
      evr_std, PCA().fit((X - mu) / X.std(axis=0, ddof=1)).explained_variance_ratio_), ")")
""")

nb.md(r"""
With one badly scaled feature, "PC1 explains 99.99999 % of the variance" — and it is just
that feature. Standardising restores a sensible split (73 %, 23 %, ...) that does not depend
on the units anyone chose. When features have different units, standardise first.

**Try it yourself:** compute the cumulative explained variance `np.cumsum(evr_std)` and find
the smallest $k$ that captures 99 %. Then repeat for `load_wine()` (13 features) — how many
components does it need?
""")

# ---------------------------------------------------------------- Section 7
nb.md(r"""
## 7. The pseudo-inverse and least squares via the SVD

$$A^{+} = V\Sigma^{+}U^\top, \qquad \Sigma^{+}_{ii} = 1/\sigma_i \ \text{ if } \sigma_i > 0,\ \text{ else } 0,
\qquad x^{+} = A^{+}b = \sum_{\sigma_i>0}\frac{u_i^\top b}{\sigma_i}\,v_i .$$

$x^{+}$ minimises $\|Ax - b\|$ and, among all minimisers, has the smallest norm. It exists
even when $A^\top A$ is singular — the case the normal equations cannot handle. We build it
for a calibration problem with two perfectly collinear sensors (the same temperature in
Celsius and Fahrenheit).
""")

nb.code(r"""
def pinv_svd(A, rcond=1e-12):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_plus = np.where(s > rcond * s.max(), 1.0 / np.where(s > 0, s, 1.0), 0.0)
    return Vt.T @ (s_plus[:, None] * U.T)

celsius = np.array([20., 25., 30., 35., 40., 45.])
fahrenheit = 1.8 * celsius + 32                       # exactly collinear with [1, celsius]
hardness = 50 + 0.8 * celsius + rng.normal(0, 0.5, size=6)
Xd = np.column_stack([np.ones(6), celsius, fahrenheit])  # design matrix, rank 2

print("singular values of the design matrix:", np.linalg.svd(Xd, compute_uv=False))
print("rank:", np.linalg.matrix_rank(Xd), " -> X^T X is singular, cond =", np.linalg.cond(Xd.T @ Xd))

coef_min = pinv_svd(Xd) @ hardness
print("\nour pinv agrees with np.linalg.pinv:", np.allclose(pinv_svd(Xd), np.linalg.pinv(Xd)))
print("lstsq gives the same coefficients:", np.allclose(coef_min, np.linalg.lstsq(Xd, hardness, rcond=None)[0]))
print("minimum-norm coefficients [intercept, C, F]:", coef_min, " norm = %.4f" % np.linalg.norm(coef_min))

# any null-space direction gives the SAME fit but a LONGER coefficient vector
null_dir = np.array([32., 1.8, -1.])      # 32*1 + 1.8*C - F = 0 for every row
print("X @ null_dir = 0 ?", np.allclose(Xd @ null_dir, 0))
coef_other = coef_min + 0.5 * null_dir
print("same predictions ?", np.allclose(Xd @ coef_other, Xd @ coef_min),
      "  norm = %.4f (larger)" % np.linalg.norm(coef_other))
print("residual of the fit: %.4f" % np.linalg.norm(Xd @ coef_min - hardness))
""")

nb.md(r"""
The third singular value is zero (up to rounding), so $X^\top X$ cannot be inverted — but
`pinv_svd` simply skips that direction and returns the shortest coefficient vector that fits
the data. Adding any multiple of the null-space direction $(32, 1.8, -1)$ leaves every
prediction unchanged and only makes the coefficients longer. That is why `np.linalg.lstsq`
and `sklearn.linear_model.LinearRegression` are SVD-based.

**Ridge regression** softens the hard cut-off: instead of $1/\sigma_i$ (or $0$) it uses the
filter $\sigma_i/(\sigma_i^2 + \lambda)$, which leaves large singular values alone and
shrinks small ones towards zero.
""")

nb.code(r"""
sig = np.logspace(-2, 2, 200)
for lam_ridge in [0.01, 1.0, 100.0]:
    plt.semilogx(sig, sig ** 2 / (sig ** 2 + lam_ridge), label=f"lambda = {lam_ridge:g}")
plt.axhline(1.0, color="gray", ls=":", label="pseudo-inverse (lambda = 0)")
plt.title("Ridge filter factors sigma^2 / (sigma^2 + lambda)")
plt.xlabel("singular value sigma_i"); plt.ylabel("fraction of the OLS component kept"); plt.legend(); plt.show()
print("A direction with sigma = 0.1 and lambda = 1 keeps only %.1f%% of its OLS coefficient."
      % (100 * 0.1 ** 2 / (0.1 ** 2 + 1)))
""")

nb.md(r"""
Directions with $\sigma_i^2 \gg \lambda$ pass unchanged; directions with $\sigma_i^2 \ll
\lambda$ — the ones the data barely constrain and that amplify noise most — are nearly
removed. Module 13 derives this and shows why it reduces variance.
""")

# ---------------------------------------------------------------- Section 8
nb.md(r"""
## 8. Condition number and numerical stability

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}, \qquad
\frac{\|\delta x\|}{\|x\|} \le \kappa(A)\,\frac{\|\delta b\|}{\|b\|}.$$

We manufacture matrices with a prescribed $\kappa$ (random orthogonal $U$, $V$ and chosen
singular values), perturb $b$ by one part in $10^8$, and measure how much the solution of
$Ax = b$ moves — for random perturbations and for the worst-case direction.
""")

nb.code(r"""
def random_orthogonal(n):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q

def matrix_with_condition(n, kappa):
    U, V = random_orthogonal(n), random_orthogonal(n)
    s = np.logspace(0, -np.log10(kappa), n)       # sigma from 1 down to 1/kappa
    return U @ np.diag(s) @ V.T, U, V, s

n = 8
kappas = np.logspace(0, 8, 9)
amp_random, amp_worst = [], []
for kappa in kappas:
    A_k, U, V, s = matrix_with_condition(n, kappa)
    # random perturbation of a random right-hand side
    x_true = rng.normal(size=n); b = A_k @ x_true
    db = rng.normal(size=n); db = 1e-8 * np.linalg.norm(b) * db / np.linalg.norm(db)
    x_pert = np.linalg.solve(A_k, b + db)
    amp_random.append((np.linalg.norm(x_pert - x_true) / np.linalg.norm(x_true)) / 1e-8)
    # worst case: b along u_1 (strong direction), db along u_n (weak direction)
    b_w = U[:, 0]; db_w = 1e-8 * U[:, -1]
    x_w = np.linalg.solve(A_k, b_w); x_wp = np.linalg.solve(A_k, b_w + db_w)
    amp_worst.append((np.linalg.norm(x_wp - x_w) / np.linalg.norm(x_w)) / 1e-8)
    print(f"kappa = {kappa:9.0e}:  np.linalg.cond = {np.linalg.cond(A_k):9.2e},"
          f"  amplification random {amp_random[-1]:9.2e},  worst case {amp_worst[-1]:9.2e}")

plt.loglog(kappas, amp_worst, "o-", label="worst-case direction")
plt.loglog(kappas, amp_random, "s-", label="random direction")
plt.loglog(kappas, kappas, "k--", label="bound: amplification = kappa")
plt.title("Error amplification when solving Ax = b vs condition number")
plt.xlabel("condition number kappa"); plt.ylabel("(relative error in x) / (relative error in b)")
plt.legend(); plt.show()
""")

nb.md(r"""
The worst-case amplification sits exactly on the line $\kappa$ — the bound is tight. Random
perturbations do less damage but still grow with $\kappa$: with $\kappa = 10^8$ a change in
the eighth decimal of $b$ moves $x$ in the first decimal. Rule of thumb: you lose
$\log_{10}\kappa$ digits.

### Why ill-conditioning slows gradient descent

On the quadratic $f(x) = \tfrac12 x^\top A x$ with eigenvalues $\lambda_{\max}$ and
$\lambda_{\min}$, gradient descent with the best fixed step $\eta = 2/(\lambda_{\max} +
\lambda_{\min})$ shrinks the error by $(\kappa - 1)/(\kappa + 1)$ per step. Let's count steps.
""")

nb.code(r"""
def gd_steps_to_tolerance(kappa, tol=1e-3, max_steps=200_000):
    A_q = np.diag([1.0, kappa])                    # eigenvalues 1 and kappa
    eta = 2.0 / (1.0 + kappa)                      # optimal fixed step size
    x = np.array([1.0, 1.0]); x0_norm = np.linalg.norm(x)
    for step in range(1, max_steps + 1):
        x = x - eta * (A_q @ x)                    # gradient of 0.5 x^T A x is A x
        if np.linalg.norm(x) < tol * x0_norm:
            return step
    return max_steps

kappas_gd = np.array([1, 2, 5, 10, 30, 100, 300, 1000, 3000, 10000])
steps = np.array([gd_steps_to_tolerance(k) for k in kappas_gd])
ratio = (kappas_gd - 1) / (kappas_gd + 1)
predicted = np.ones_like(kappas_gd, dtype=float)
predicted[kappas_gd > 1] = np.log(1e-3) / np.log(ratio[kappas_gd > 1])   # kappa = 1 converges in one step
for k_, s_, p_ in zip(kappas_gd, steps, predicted):
    print(f"kappa = {k_:6d}: {s_:6d} steps to shrink the error 1000x   (theory ~ {p_:8.0f})")

plt.loglog(kappas_gd, steps, "o-", label="measured steps")
plt.loglog(kappas_gd, predicted, "--", label="log(1e-3) / log((kappa-1)/(kappa+1))")
plt.title("Gradient descent needs ~ kappa steps per factor of e")
plt.xlabel("condition number kappa"); plt.ylabel("steps to reduce error 1000x"); plt.legend(); plt.show()
""")

nb.md(r"""
The step count grows linearly with $\kappa$: ten times worse conditioning, ten times more
iterations. Module 05 shows how momentum and Adam attack this; the cheapest fix of all is to
make $\kappa$ small in the first place by **scaling the features**.
""")

nb.code(r"""
# A polynomial-trend design matrix on raw calendar years vs centred years
years = np.arange(1990, 2021, dtype=float)
X_raw = np.column_stack([np.ones_like(years), years, years ** 2])
X_cen = np.column_stack([np.ones_like(years), years - 2005, (years - 2005) ** 2])
print("condition number, raw years:     %.2e" % np.linalg.cond(X_raw))
print("condition number, centred years: %.2e" % np.linalg.cond(X_cen))

# Correlated features with wildly different scales, before and after standardising
Z = rng.normal(size=(500, 3)) @ np.array([[1, 0.9, 0.0], [0, 0.4, 0.0], [0, 0, 1.0]])
Z = Z * np.array([1.0, 1000.0, 0.01])        # different units
Z_std = (Z - Z.mean(axis=0)) / Z.std(axis=0)
print("\ncondition number of X^T X, raw features:          %.2e" % np.linalg.cond(Z.T @ Z))
print("condition number of X^T X, standardised features: %.2e" % np.linalg.cond(Z_std.T @ Z_std))
print("(the remaining conditioning comes from the 0.9 correlation, which scaling cannot remove)")
""")

nb.md(r"""
Centring the year variable cuts the condition number from about $10^{11}$ to about $10^2$ —
same model, same fitted curve, nine orders of magnitude less error amplification.
Standardising the three-feature design matrix removes the damage done by units; what remains
is genuine correlation between features, which only regularisation (Section 7, Module 13)
or dropping features can address.

**Try it yourself:** in `gd_steps_to_tolerance`, use the "safe" step `eta = 1.0 / kappa`
instead of the optimal one. How does the step count change?
""")

# ---------------------------------------------------------------- Summary
nb.md(r"""
## Summary

- Eigenvectors are the directions a square matrix only stretches; for a $2\times2$ matrix
  the eigenvalues solve $\lambda^2 - \operatorname{tr}A\,\lambda + \det A = 0$.
- $A^k = PD^kP^{-1}$: the dominant eigenvalue takes over, which is why power iteration
  works, why Markov chains converge to the eigenvector with eigenvalue 1, and why PageRank
  with damping converges in a few dozen steps.
- Symmetric matrices have real eigenvalues and orthonormal eigenvectors ($A = Q\Lambda Q^\top$);
  positive semidefinite means all $\lambda_i \ge 0$, and the Rayleigh quotient lives in
  $[\lambda_{\min}, \lambda_{\max}]$.
- Every matrix has an SVD $A = U\Sigma V^\top$: rotate, stretch, rotate. Singular values
  reveal rank, norms and noise floors.
- Truncating the SVD at rank $k$ is the best rank-$k$ approximation (Eckart–Young), with
  error exactly $\sqrt{\sum_{i>k}\sigma_i^2}$ — the basis of compression, denoising and LoRA.
- PCA = eigenvectors of the covariance = right singular vectors of the centred data;
  scikit-learn agrees to machine precision; scale your features first.
- The pseudo-inverse $V\Sigma^+U^\top$ gives the minimum-norm least-squares solution even
  for rank-deficient problems; ridge regression is a soft version of its cut-off.
- The condition number $\sigma_{\max}/\sigma_{\min}$ bounds error amplification exactly and
  sets the number of gradient-descent steps; centring and standardising shrink it.

## Next: open `exercises.ipynb`

Twelve exercises with instant self-checks, from eigenvalues by hand to a latent-factor
recommender built from truncated SVDs.
""")

nb.save(MOD / "lesson.ipynb")
print("lesson cells:", len(nb.cells))

# =============================================================================
# EXERCISES — solution code is executed here to compute the hidden answers
# =============================================================================
SETUP = r"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
"""

ex = ExerciseSet(
    "Module 03 — Exercises",
    intro_md=r"""
Exercises for **Module 03 — Eigenvalues, SVD and PCA**. They follow the order of the
[concept guide](README.md): eigenvalues by hand, the spectral theorem, truncated SVD, power
iteration and PageRank, PCA, the pseudo-inverse, conditioning, ridge regression through the
SVD, two conceptual questions and a mini-project on latent-factor recommendation.
Use only NumPy (plus the scikit-learn objects named in a prompt) and keep everything seeded.
""",
    setup_code=SETUP,
)

ns: dict = {}
exec(SETUP, ns)  # gives the builder the same np/rng/PCA names as the notebook


def run_solution(src: str) -> None:
    """Execute solution code inside the shared namespace (hashes come from *running* it)."""
    exec(src, ns)


# ---------------------------------------------------------------- Ex 1
SOL1 = r"""
def eig2x2(A):
    tr = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    disc = np.sqrt(tr ** 2 - 4 * det)
    return np.array([(tr + disc) / 2, (tr - disc) / 2])
"""
run_solution(SOL1)
h1 = sha_of(ns["eig2x2"](np.array([[1., 4.], [2., 3.]])))

ex.add(
    title="Eigenvalues of a 2x2 matrix by hand",
    difficulty=1,
    prompt_md=r"""
Implement `eig2x2(A)` that returns the two eigenvalues of a real $2\times2$ matrix `A`
as a NumPy array **sorted in descending order**, using only the characteristic polynomial

$$\lambda^2 - \operatorname{tr}(A)\,\lambda + \det(A) = 0$$

and the quadratic formula. Do **not** call anything from `np.linalg`. You may assume the
eigenvalues are real (discriminant $\ge 0$).
""",
    starter=r"""
def eig2x2(A):
    # YOUR CODE HERE  (trace, determinant, quadratic formula; return np.array([larger, smaller]))
    return None
""",
    solution=SOL1,
    check=r"""
try:
    got = eig2x2(np.array([[4., 1.], [2., 3.]]))
except Exception:
    got = None
check_close('eigenvalues of [[4,1],[2,3]] are 5 and 2', got, np.array([5., 2.]))
try:
    got_sym = eig2x2(np.array([[2., 1.], [1., 2.]]))
except Exception:
    got_sym = None
check_close('eigenvalues of [[2,1],[1,2]] are 3 and 1', got_sym, np.array([3., 1.]))
try:
    got_hidden = eig2x2(np.array([[1., 4.], [2., 3.]]))
except Exception:
    got_hidden = None
check_hash('hidden: eigenvalues of [[1,4],[2,3]]', got_hidden, 'HASH1')
""".replace("HASH1", h1),
    explanation_md=r"""
For a $2\times2$ matrix the characteristic polynomial is fully determined by the trace and
the determinant, and the quadratic formula gives $\lambda = \tfrac12\left(\operatorname{tr} \pm
\sqrt{\operatorname{tr}^2 - 4\det}\right)$. The eigenvalues sum to the trace and multiply to
the determinant — a quick sanity check on any hand calculation. The hidden case has a
negative eigenvalue ($5$ and $-1$ for trace 4, determinant $-5$); a common wrong turn is to
sort by absolute value or to forget that the determinant can be negative.
""",
)

# ---------------------------------------------------------------- Ex 2
SOL2 = r"""
def spectral(A):
    lam, Q = np.linalg.eigh(A)
    order = np.argsort(-lam)
    return lam[order], Q[:, order]

def is_psd(A, tol=1e-10):
    lam, _ = spectral(A)
    return bool(lam.min() >= -tol)
"""
run_solution(SOL2)
h2 = sha_of(float(ns["spectral"](np.array([[1., 2.], [2., 1.]]))[0].min()))

ex.add(
    title="Spectral theorem and positive semidefiniteness",
    difficulty=1,
    prompt_md=r"""
Implement two functions for a real **symmetric** matrix `A`:

* `spectral(A)` returns `(lam, Q)` where `lam` holds the eigenvalues **sorted in descending
  order** and the columns of `Q` are the matching **orthonormal** eigenvectors, so that
  `Q @ np.diag(lam) @ Q.T` reconstructs `A`. Use `np.linalg.eigh` (which returns ascending order).
* `is_psd(A, tol=1e-10)` returns a Python `bool`: `True` if every eigenvalue is $\ge -\text{tol}$.
""",
    starter=r"""
def spectral(A):
    # YOUR CODE HERE  (np.linalg.eigh, then reorder to descending)
    return None

def is_psd(A, tol=1e-10):
    # YOUR CODE HERE
    return None
""",
    solution=SOL2,
    check=r"""
_g = np.random.default_rng(11)
_B = _g.normal(size=(4, 4)); _S = _B @ _B.T
try:
    lam_s, Q_s = spectral(_S)
except Exception:
    lam_s, Q_s = None, None
check('eigenvalues sorted in descending order', lam_s is not None and np.all(np.diff(lam_s) <= 0))
check('Q is orthonormal (Q^T Q = I)', Q_s is not None and np.allclose(Q_s.T @ Q_s, np.eye(4)))
check('Q diag(lam) Q^T reconstructs A', Q_s is not None and np.allclose(Q_s @ np.diag(lam_s) @ Q_s.T, _S))
try:
    psd_a = is_psd(np.array([[2., 1.], [1., 2.]]))
    psd_b = is_psd(np.array([[1., 2.], [2., 1.]]))
    psd_c = is_psd(_S)
except Exception:
    psd_a = psd_b = psd_c = None
check('[[2,1],[1,2]] is PSD', psd_a is True)
check('[[1,2],[2,1]] is NOT PSD', psd_b is False)
check('a Gram matrix B B^T is PSD', psd_c is True)
try:
    smallest = float(spectral(np.array([[1., 2.], [2., 1.]]))[0].min())
except Exception:
    smallest = None
check_hash('hidden: smallest eigenvalue of [[1,2],[2,1]]', smallest, 'HASH2')
""".replace("HASH2", h2),
    explanation_md=r"""
`np.linalg.eigh` already returns real eigenvalues and an orthonormal `Q`, but in *ascending*
order; `np.argsort(-lam)` (or slicing with `[::-1]`) flips it. Reordering the eigenvalues and
the columns of `Q` together keeps the pairing intact, so the reconstruction still holds.
Positive semidefiniteness is a statement about *all* eigenvalues — a positive diagonal is not
enough: $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ has eigenvalues $3$ and $-1$, and the vector
$(1,-1)$ gives the negative quadratic form $-2$. The tolerance matters because rounding can
turn a true zero eigenvalue into $-10^{-16}$.
""",
)

# ---------------------------------------------------------------- Ex 3
SOL3 = r"""
def rank_k(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]

def rel_frobenius_error(A, k):
    return float(np.linalg.norm(A - rank_k(A, k)) / np.linalg.norm(A))
"""
run_solution(SOL3)
_g = np.random.default_rng(7)
_M = _g.normal(size=(20, 3)) @ _g.normal(size=(3, 15)) + 0.5 * _g.normal(size=(20, 15))
_rel3 = ns["rel_frobenius_error"](_M, 3)
safe_to_hash(_rel3)
h3 = sha_of(_rel3)

ex.add(
    title="Truncated SVD and the Eckart-Young error",
    difficulty=1,
    prompt_md=r"""
Implement

* `rank_k(A, k)` — the best rank-$k$ approximation $A_k = U_k\Sigma_kV_k^\top$ built from
  `np.linalg.svd(A, full_matrices=False)`;
* `rel_frobenius_error(A, k)` — the relative error $\|A - A_k\|_F / \|A\|_F$ as a Python `float`.

Eckart–Young says this error equals $\sqrt{\sum_{i>k}\sigma_i^2 / \sum_i \sigma_i^2}$; the
check compares your measured error with that formula.
""",
    starter=r"""
def rank_k(A, k):
    # YOUR CODE HERE
    return None

def rel_frobenius_error(A, k):
    # YOUR CODE HERE
    return None
""",
    solution=SOL3,
    check=r"""
_g = np.random.default_rng(7)
_M = _g.normal(size=(20, 3)) @ _g.normal(size=(3, 15)) + 0.5 * _g.normal(size=(20, 15))
_s = np.linalg.svd(_M, compute_uv=False)
try:
    _Ak = rank_k(_M, 2)
except Exception:
    _Ak = None
check('rank_k(M, 2) has shape (20, 15)', _Ak is not None and _Ak.shape == (20, 15))
check('rank_k(M, 2) has rank 2', _Ak is not None and np.linalg.matrix_rank(_Ak) == 2)
try:
    _full = rank_k(_M, 15)
except Exception:
    _full = None
check_close('rank_k(M, 15) reproduces M exactly', _full, _M, tol=1e-8)
try:
    _e2 = rel_frobenius_error(_M, 2)
except Exception:
    _e2 = None
check_close('k = 2 error matches sqrt(sum_{i>2} s_i^2 / sum s_i^2)', _e2,
            np.sqrt((_s[2:] ** 2).sum() / (_s ** 2).sum()), tol=1e-8)
try:
    _e3 = rel_frobenius_error(_M, 3)
except Exception:
    _e3 = None
check_hash('hidden: relative error at k = 3 (3 decimals)', _e3, 'HASH3')
""".replace("HASH3", h3),
    explanation_md=r"""
`U[:, :k] * s[:k]` scales the first $k$ left singular vectors by their singular values
(broadcasting over columns), and multiplying by `Vt[:k]` sums the $k$ rank-one layers
$\sigma_i u_i v_i^\top$. Because the layers are orthogonal to each other, the squared Frobenius
norm of what is dropped is exactly $\sum_{i>k}\sigma_i^2$, which is the Eckart–Young error.
At $k = 3$ the error is about 26 % — the matrix is rank 3 plus noise, and the noise is what
remains. A common wrong turn is `U[:k]` (rows instead of columns) or using `np.diag(s)` with
`full_matrices=True`, where the shapes no longer match.
""",
)

# ---------------------------------------------------------------- Ex 4
SOL4 = r"""
def power_iteration(A, n_iter=500, x0=None):
    n = A.shape[0]
    x = np.ones(n) if x0 is None else np.asarray(x0, dtype=float)
    x = x / np.linalg.norm(x)
    for _ in range(n_iter):
        y = A @ x
        x = y / np.linalg.norm(y)
    lam = float(x @ A @ x)
    return lam, x
"""
run_solution(SOL4)
_g = np.random.default_rng(18)
_B = _g.normal(size=(5, 5)); _S5 = _B @ _B.T
_lam4, _v4 = ns["power_iteration"](_S5)
safe_to_hash(_lam4)
h4 = sha_of(_lam4)

ex.add(
    title="Power iteration",
    difficulty=2,
    prompt_md=r"""
Implement `power_iteration(A, n_iter=500, x0=None)` returning `(lam, v)`: the dominant
eigenvalue (as a Python `float`, estimated by the Rayleigh quotient $v^\top A v$) and the
corresponding **unit-length** eigenvector. Start from `x0` if given, otherwise from the
all-ones vector; normalise after every multiplication. Do **not** call `np.linalg.eig`,
`eigh`, `eigvals` or `svd`.
""",
    starter=r"""
def power_iteration(A, n_iter=500, x0=None):
    # YOUR CODE HERE
    return None
""",
    solution=SOL4,
    check=r"""
_g = np.random.default_rng(18)
_B = _g.normal(size=(5, 5)); _S5 = _B @ _B.T
_lam_true, _Q_true = np.linalg.eigh(_S5)
try:
    lam_pi, v_pi = power_iteration(_S5)
except Exception:
    lam_pi, v_pi = None, None
check_close('dominant eigenvalue of a 5x5 Gram matrix', lam_pi, _lam_true[-1], tol=1e-8)
check('eigenvector has unit length', v_pi is not None and abs(np.linalg.norm(v_pi) - 1) < 1e-9)
check('eigenvector matches eigh up to sign', v_pi is not None and abs(abs(v_pi @ _Q_true[:, -1]) - 1) < 1e-6)
try:
    lam_A, v_A = power_iteration(np.array([[4., 1.], [2., 3.]]), x0=np.array([1., 0.]))
except Exception:
    lam_A, v_A = None, None
check_close('works on the non-symmetric [[4,1],[2,3]]: lambda_1 = 5', lam_A, 5.0, tol=1e-8)
check_hash('hidden: dominant eigenvalue of the Gram matrix (3 decimals)', lam_pi, 'HASH4')
""".replace("HASH4", h4),
    explanation_md=r"""
Each multiplication by $A$ amplifies the component along $v_1$ by $\lambda_1$ and the others
by smaller factors, so after normalisation the vector turns towards $v_1$ at the geometric rate
$|\lambda_2/\lambda_1|$. Normalising every step prevents overflow or underflow; the Rayleigh
quotient $x^\top A x$ of a unit vector then estimates $\lambda_1$ (for symmetric matrices its
error is the *square* of the vector's error, which is why it converges so fast). Forgetting
to normalise, or returning `y` instead of `x`, are the usual slips. Power iteration cannot find
$\lambda_1$ if the start vector is exactly orthogonal to $v_1$ or if $|\lambda_1| = |\lambda_2|$.
""",
)

# ---------------------------------------------------------------- Ex 5
SOL5 = r"""
def pagerank(links, d=0.85, n_iter=200):
    n = len(links)
    M = np.zeros((n, n))
    for j in range(n):
        outs = links[j]
        if len(outs) == 0:
            M[:, j] = 1.0 / n
        else:
            M[outs, j] = 1.0 / len(outs)
    G = d * M + (1.0 - d) / n * np.ones((n, n))
    x = np.ones(n) / n
    for _ in range(n_iter):
        x = G @ x
    return x / x.sum()
"""
run_solution(SOL5)
WEB = {0: [1, 3], 1: [2], 2: [0, 1], 3: [2, 4], 4: [5], 5: [3, 4], 6: []}
_pr = ns["pagerank"](WEB)
_pr_sorted = np.sort(_pr)
assert np.min(np.diff(_pr_sorted)) > 1e-3, "PageRank values too close for a stable ranking"
h5_top = sha_of(int(np.argmax(_pr)))
h5_order = sha_of([int(i) for i in np.argsort(-_pr)])

ex.add(
    title="PageRank with damping",
    difficulty=2,
    prompt_md=r"""
Implement `pagerank(links, d=0.85, n_iter=200)` for a web graph given as a dictionary
`links[j] = list of pages that page j links to` (pages are numbered `0 .. n-1`). Build the
column-stochastic matrix $M$ with $M_{ij} = 1/\text{outdeg}(j)$ if $j$ links to $i$, treating a
page with **no out-links as linking to every page** (including itself) with probability $1/n$.
Form the Google matrix

$$G = d\,M + \frac{1-d}{n}\,\mathbf{1}\mathbf{1}^\top ,$$

start from the uniform vector and apply $G$ `n_iter` times. Return the PageRank vector
(a NumPy array that sums to one).

The hidden checks use this seven-page web (page 6 is a dead end):

```python
web = {0: [1, 3], 1: [2], 2: [0, 1], 3: [2, 4], 4: [5], 5: [3, 4], 6: []}
```
""",
    starter=r"""
def pagerank(links, d=0.85, n_iter=200):
    n = len(links)
    # YOUR CODE HERE  (build M column by column, then G, then iterate)
    return None

web = {0: [1, 3], 1: [2], 2: [0, 1], 3: [2, 4], 4: [5], 5: [3, 4], 6: []}
""",
    solution=SOL5 + r"""
web = {0: [1, 3], 1: [2], 2: [0, 1], 3: [2, 4], 4: [5], 5: [3, 4], 6: []}
print("PageRank:", pagerank(web))
print("ranking (best first):", np.argsort(-pagerank(web)))
""",
    check=r"""
try:
    pr_two = pagerank({0: [1], 1: [0]})
except Exception:
    pr_two = None
check_close('two pages linking to each other share rank equally', pr_two, np.array([0.5, 0.5]), tol=1e-9)
try:
    pr = pagerank(web)
except Exception:
    pr = None
check('PageRank sums to one', pr is not None and abs(pr.sum() - 1) < 1e-9)
check('every page has positive rank (damping)', pr is not None and np.all(pr > 0))
check('the dead-end page 6 gets the lowest rank', pr is not None and int(np.argmin(pr)) == 6)
check_hash('hidden: the top-ranked page', int(np.argmax(pr)) if pr is not None else None, 'HASH5A')
check_hash('hidden: the full ranking, best first', [int(i) for i in np.argsort(-pr)] if pr is not None else None, 'HASH5B')
""".replace("HASH5A", h5_top).replace("HASH5B", h5_order),
    explanation_md=r"""
Column $j$ of $M$ is the probability distribution of where a surfer on page $j$ goes next, so
each column must sum to one — including the dead-end column, which the uniform rule fixes.
Mixing in the teleport matrix $\frac{1-d}{n}\mathbf{1}\mathbf{1}^\top$ makes every entry of
$G$ positive, so by Perron–Frobenius the stationary vector is unique and power iteration
converges at least as fast as $d^k$: 200 iterations are far more than needed. Nobody links to
page 6, so it lives only on teleportation and ranks last. The classic bug is building $M$
row-stochastic (rows sum to one) and then multiplying from the wrong side; another is
forgetting the dangling-page rule, which leaks probability mass out of the chain.
""",
)

# ---------------------------------------------------------------- Ex 6
SOL6 = r"""
def pca_fit(X, k):
    mean = X.mean(axis=0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = s ** 2 / (X.shape[0] - 1)
    return Vt[:k], lam[:k] / lam.sum(), mean

def pca_transform(X, components, mean):
    return (X - mean) @ components.T

wine = load_wine().data
wine_std = (wine - wine.mean(axis=0)) / wine.std(axis=0, ddof=1)
_, evr_wine, _ = pca_fit(wine_std, wine_std.shape[1])
k95 = int(np.argmax(np.cumsum(evr_wine) >= 0.95) + 1)
print("components needed for 95% of the variance of the standardised wine data:", k95)
"""
run_solution(SOL6)
h6 = sha_of(int(ns["k95"]))

ex.add(
    title="PCA from scratch versus scikit-learn",
    difficulty=2,
    prompt_md=r"""
Implement PCA with NumPy only:

* `pca_fit(X, k)` returns `(components, explained_variance_ratio, mean)` where `components`
  is a $k\times d$ array whose **rows** are the first $k$ principal components (unit length),
  `explained_variance_ratio` has shape `(k,)`, and `mean` is the column mean of `X`. Centre
  the data, then use either the eigendecomposition of the covariance matrix (with $n-1$) or
  the SVD of the centred data.
* `pca_transform(X, components, mean)` returns the $n\times k$ matrix of scores.

Then standardise the wine data `load_wine().data` (zero mean, unit variance per column),
run your PCA with all 13 components, and set `k95` to the **smallest** number of components
whose cumulative explained variance reaches 95 % (a Python `int`).
""",
    starter=r"""
def pca_fit(X, k):
    # YOUR CODE HERE
    return None

def pca_transform(X, components, mean):
    # YOUR CODE HERE
    return None

wine = load_wine().data
k95 = None   # YOUR CODE HERE
""",
    solution=SOL6,
    check=r"""
_X = load_iris().data
_ref = PCA(n_components=2).fit(_X)
try:
    comps, evr, mu_ = pca_fit(_X, 2)
except Exception:
    comps, evr, mu_ = None, None, None
check('components has shape (2, 4)', comps is not None and comps.shape == (2, 4))
check_close('explained variance ratio matches sklearn', evr, _ref.explained_variance_ratio_, tol=1e-8)
check('components match sklearn up to sign', comps is not None and np.allclose(np.abs(comps), np.abs(_ref.components_), atol=1e-6))
try:
    Z = pca_transform(_X, comps, mu_)
except Exception:
    Z = None
check('scores have shape (150, 2)', Z is not None and Z.shape == (150, 2))
check('scores match sklearn up to sign', Z is not None and np.allclose(np.abs(Z), np.abs(_ref.transform(_X)), atol=1e-6))
_lam_all = np.linalg.eigvalsh(np.cov(_X, rowvar=False))[::-1]
try:
    _resid = ((_X - (Z @ comps + mu_)) ** 2).sum()
except Exception:
    _resid = None
check_close('reconstruction error equals (n-1) * sum of dropped eigenvalues', _resid, 149 * _lam_all[2:].sum(), tol=1e-6)
check_hash('hidden: components for 95% of variance (standardised wine)', k95 if isinstance(k95, (int, np.integer)) else None, 'HASH6')
""".replace("HASH6", h6),
    explanation_md=r"""
Centring is the step people forget: without it the first singular vector points at the mean
of the data rather than along its largest variance. With $X_c = U\Sigma V^\top$, the rows of
$V^\top$ are the components and $\sigma_i^2/(n-1)$ are the variances along them, so the
explained-variance ratio is $\sigma_i^2/\sum_j\sigma_j^2$ — no covariance matrix needed.
Components are only defined up to sign, which is why the checks compare absolute values.
The wine data needs 10 of 13 standardised components to reach 95 %: its features are only
moderately correlated, so PCA compresses it far less than Iris.
""",
)

# ---------------------------------------------------------------- Ex 7 (conceptual)
ex.add_conceptual(
    title="Why PCA on unscaled features is misleading",
    difficulty=2,
    prompt_md=r"""
A hospital dataset has the columns *age (years)*, *systolic blood pressure (mmHg)*,
*cholesterol (mg/dL)* and *daily steps (count)*. A colleague runs PCA on the raw data and
reports that "the first component explains 99.8 % of the variance, so the data is essentially
one-dimensional".

1. Explain what the first component almost certainly is and why the 99.8 % figure says
   nothing about the structure of the data.
2. State precisely what preprocessing fixes the problem and what PCA then operates on.
3. Name one situation in which you would deliberately **not** standardise before PCA.
""",
    answer_md=r"""
1. Daily step counts vary by thousands, while age varies by tens and blood pressure and
   cholesterol by tens to hundreds. The total variance is therefore dominated by the steps
   column, and the first principal component is essentially the steps axis (a loading near 1
   on steps and near 0 elsewhere). Its 99.8 % is a statement about *units*, not about the
   data: re-recording steps in thousands of steps would change the result completely. PCA
   maximises variance, and variance is not scale-invariant.
2. Standardise every column — subtract its mean and divide by its standard deviation — so
   that each feature has unit variance. PCA then diagonalises the **correlation matrix**
   instead of the covariance matrix; each feature starts with an equal share (1 of $d$) of the
   variance, and the components reflect correlations between features rather than their
   magnitudes.
3. When the features share a unit and their scale *is* meaningful: pixel intensities in an
   image, gene-expression values after a common log transform, or returns of several assets in
   percent. Standardising would then inflate the noise in nearly constant features (a dead
   pixel, a gene that is barely expressed) to the same importance as real signal. In such
   cases run PCA on the covariance matrix, possibly after a variance-stabilising transform.
""",
)

# ---------------------------------------------------------------- Ex 8
SOL8 = r"""
def pinv_svd(A, rcond=1e-12):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > rcond * s.max()
    s_plus = np.zeros_like(s)
    s_plus[keep] = 1.0 / s[keep]
    return Vt.T @ (s_plus[:, None] * U.T)
"""
run_solution(SOL8)
_A_under = np.array([[1., 2., 3.], [4., 5., 6.]])
_x_min = ns["pinv_svd"](_A_under) @ np.array([1., 1.])
_norm_min = float(np.linalg.norm(_x_min))
safe_to_hash(_norm_min)
h8 = sha_of(_norm_min)

ex.add(
    title="The pseudo-inverse via the SVD",
    difficulty=2,
    prompt_md=r"""
Implement `pinv_svd(A, rcond=1e-12)` returning the Moore–Penrose pseudo-inverse
$A^{+} = V\Sigma^{+}U^\top$ for any real $m\times n$ matrix, where $\Sigma^{+}$ inverts every
singular value with $\sigma_i > \text{rcond}\cdot\sigma_{\max}$ and sets the rest to zero.
Use `np.linalg.svd` only — no `np.linalg.pinv`, `inv`, `solve` or `lstsq`.

Then compute `x_min = pinv_svd(A) @ b` for the under-determined system

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}, \qquad b = (1, 1),$$

which has infinitely many exact solutions; `x_min` is the one with the smallest norm.
""",
    starter=r"""
def pinv_svd(A, rcond=1e-12):
    # YOUR CODE HERE
    return None

A_under = np.array([[1., 2., 3.], [4., 5., 6.]])
b_under = np.array([1., 1.])
x_min = None   # YOUR CODE HERE  (use pinv_svd)
""",
    solution=SOL8 + r"""
A_under = np.array([[1., 2., 3.], [4., 5., 6.]])
b_under = np.array([1., 1.])
x_min = pinv_svd(A_under) @ b_under
print("minimum-norm solution:", x_min, " norm = %.4f" % np.linalg.norm(x_min), " residual = %.1e" % np.linalg.norm(A_under @ x_min - b_under))
""",
    check=r"""
_g = np.random.default_rng(5)
_T = _g.normal(size=(6, 3))                               # tall, full column rank
_R = _g.normal(size=(4, 2)) @ _g.normal(size=(2, 4))      # 4x4 of rank 2
try:
    P_T = pinv_svd(_T); P_R = pinv_svd(_R)
except Exception:
    P_T = P_R = None
check_close('matches np.linalg.pinv on a 6x3 full-rank matrix', P_T, np.linalg.pinv(_T), tol=1e-8)
check_close('matches np.linalg.pinv on a rank-deficient 4x4 matrix', P_R, np.linalg.pinv(_R), tol=1e-8)
check('Moore-Penrose identity A A+ A = A (rank-deficient case)', P_R is not None and np.allclose(_R @ P_R @ _R, _R, atol=1e-8))
check('full column rank: A+ equals (A^T A)^-1 A^T', P_T is not None and np.allclose(P_T, np.linalg.solve(_T.T @ _T, _T.T), atol=1e-8))
check('x_min solves the under-determined system exactly', x_min is not None and np.allclose(A_under @ x_min, b_under, atol=1e-9))
_x_other = (x_min + 0.3 * np.array([1., -2., 1.])) if x_min is not None else None    # (1,-2,1) spans null(A)
check('any other solution x_min + z, z in null(A), is longer', _x_other is not None and np.linalg.norm(_x_other) > np.linalg.norm(x_min))
check_hash('hidden: norm of the minimum-norm solution (3 decimals)', float(np.linalg.norm(x_min)) if x_min is not None else None, 'HASH8')
""".replace("HASH8", h8),
    explanation_md=r"""
With the thin SVD, $A^{+} = V\Sigma^{+}U^\top$ inverts each non-zero stretch and ignores the
directions the matrix destroyed; the `rcond` threshold decides what "zero" means in floating
point, which is why the rank-deficient case matches `np.linalg.pinv` instead of blowing up.
For full column rank the formula collapses to $(A^\top A)^{-1}A^\top$. The minimum-norm property
comes from $x^{+}$ lying in the row space (spanned by the $v_i$ with $\sigma_i>0$): any other
solution adds a null-space vector $z$, orthogonal to the row space, and Pythagoras gives
$\|x^{+}+z\|^2 = \|x^{+}\|^2 + \|z\|^2$. Here $x^{+} = (-0.5, 0, 0.5)$ with norm $1/\sqrt2$.
Dividing by *all* singular values, including the tiny ones, is the classic error — it
produces enormous, noise-dominated coefficients.
""",
)

# ---------------------------------------------------------------- Ex 9
SOL9 = r"""
def cond_svd(A):
    s = np.linalg.svd(A, compute_uv=False)
    return float(s.max() / s.min())

def amplification(A, b, db):
    x = np.linalg.solve(A, b)
    dx = np.linalg.solve(A, b + db) - x
    return float((np.linalg.norm(dx) / np.linalg.norm(x)) / (np.linalg.norm(db) / np.linalg.norm(b)))

H4 = np.array([[1.0 / (i + j + 1) for j in range(4)] for i in range(4)])   # the 4x4 Hilbert matrix
kappa_H4 = cond_svd(H4)
print("condition number of the 4x4 Hilbert matrix: %.3f" % kappa_H4)
"""
run_solution(SOL9)
_kH = ns["kappa_H4"]
safe_to_hash(_kH)
h9 = sha_of(_kH)

ex.add(
    title="Condition number and error amplification",
    difficulty=2,
    prompt_md=r"""
Implement

* `cond_svd(A)` — the 2-norm condition number $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ from the
  singular values (a Python `float`; do not call `np.linalg.cond`);
* `amplification(A, b, db)` — solve $Ax = b$ and $A\tilde{x} = b + \delta b$ with `np.linalg.solve`
  and return the ratio $\dfrac{\|\tilde x - x\|/\|x\|}{\|\delta b\|/\|b\|}$ as a `float`.

Then set `kappa_H4 = cond_svd(H4)` for the $4\times4$ Hilbert matrix $H_{ij} = 1/(i+j-1)$ (1-based
indices), a famous ill-conditioned matrix. The theory says `amplification` can never exceed
`cond_svd(A)` and reaches it when $b$ lies along $u_1$ and $\delta b$ along $u_{\min}$.
""",
    starter=r"""
def cond_svd(A):
    # YOUR CODE HERE
    return None

def amplification(A, b, db):
    # YOUR CODE HERE
    return None

H4 = np.array([[1.0 / (i + j + 1) for j in range(4)] for i in range(4)])   # the 4x4 Hilbert matrix
kappa_H4 = None   # YOUR CODE HERE
""",
    solution=SOL9,
    check=r"""
try:
    k_H4 = cond_svd(H4)
except Exception:
    k_H4 = None
check_close('cond_svd matches np.linalg.cond on the Hilbert matrix', k_H4, np.linalg.cond(H4), tol=1e-6)
check_hash('hidden: condition number of the 4x4 Hilbert matrix (3 decimals)', kappa_H4 if isinstance(kappa_H4, float) else None, 'HASH9')
_g = np.random.default_rng(21)
_Q1, _ = np.linalg.qr(_g.normal(size=(5, 5))); _Q2, _ = np.linalg.qr(_g.normal(size=(5, 5)))
_s = np.array([1.0, 0.5, 0.1, 0.01, 1e-4])
_A = _Q1 @ np.diag(_s) @ _Q2.T                             # kappa = 1e4 by construction
try:
    amp_rand = amplification(_A, _g.normal(size=5), 1e-6 * _g.normal(size=5))
    amp_worst = amplification(_A, _Q1[:, 0], 1e-6 * _Q1[:, -1])
except Exception:
    amp_rand = amp_worst = None
check('random perturbation: amplification <= kappa', amp_rand is not None and 1.0 <= amp_rand <= 1e4 * (1 + 1e-6))
check_close('worst-case directions: amplification equals kappa = 1e4', amp_worst, 1e4, tol=1e-4)
""".replace("HASH9", h9),
    explanation_md=r"""
$\kappa$ is the ratio of the largest stretch to the smallest, so it is read straight off the
singular values; the Hilbert matrix has $\kappa \approx 1.55\times10^4$ already at size 4
(and about $10^{13}$ at size 10). For the error bound, $\delta x = A^{-1}\delta b$ is at most
$\|\delta b\|/\sigma_{\min}$ long, while $\|x\| \ge \|b\|/\sigma_{\max}$; dividing gives
$\|\delta x\|/\|x\| \le \kappa\,\|\delta b\|/\|b\|$, with equality when $b$ is along $u_1$
(so $x$ is as short as possible) and $\delta b$ along $u_{\min}$ (so $\delta x$ is as long as
possible). A random perturbation spreads over all directions and is amplified less, which is
why the bound is sometimes called pessimistic — but the worst case does occur in practice,
for instance when noise happens to align with a nearly collinear feature direction.
""",
)

# ---------------------------------------------------------------- Ex 10 (stretch)
SOL10 = r"""
def ridge_svd(X, y, lam):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    filt = s / (s ** 2 + lam)                 # filter factors sigma_i / (sigma_i^2 + lambda)
    return Vt.T @ (filt * (U.T @ y))
"""
run_solution(SOL10)

ex.add(
    title="Ridge regression as singular-value shrinkage",
    difficulty=3,
    prompt_md=r"""
Ridge regression minimises $\|Xw - y\|^2 + \lambda\|w\|^2$ and has the closed form
$w_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$. Show that it is a *filtered* pseudo-inverse by
implementing it through the SVD $X = U\Sigma V^\top$:

$$w_\lambda = \sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda}\,(u_i^\top y)\,v_i .$$

Implement `ridge_svd(X, y, lam)` using `np.linalg.svd(X, full_matrices=False)` only (no
`inv`, `solve` or `lstsq`). The checks compare against the closed form, verify that
$\lambda = 0$ recovers the least-squares solution, and that larger $\lambda$ gives a shorter
coefficient vector.
""",
    starter=r"""
def ridge_svd(X, y, lam):
    # YOUR CODE HERE
    return None
""",
    solution=SOL10,
    check=r"""
_g = np.random.default_rng(13)
_X = _g.normal(size=(40, 6)) @ (np.eye(6) + 0.8 * np.ones((6, 6)))   # correlated features
_w_true = np.array([1., -2., 0.5, 0., 3., -1.])
_y = _X @ _w_true + 0.5 * _g.normal(size=40)
def _closed_form(lam):
    return np.linalg.solve(_X.T @ _X + lam * np.eye(6), _X.T @ _y)
try:
    w1 = ridge_svd(_X, _y, 1.0); w10 = ridge_svd(_X, _y, 10.0); w0 = ridge_svd(_X, _y, 0.0)
except Exception:
    w1 = w10 = w0 = None
check_close('lambda = 1 matches (X^T X + lambda I)^-1 X^T y', w1, _closed_form(1.0), tol=1e-8)
check_close('lambda = 10 matches the closed form', w10, _closed_form(10.0), tol=1e-8)
check_close('lambda = 0 recovers ordinary least squares', w0, np.linalg.lstsq(_X, _y, rcond=None)[0], tol=1e-8)
check('coefficients shrink as lambda grows', w0 is not None and np.linalg.norm(w10) < np.linalg.norm(w1) < np.linalg.norm(w0))
""",
    explanation_md=r"""
Substitute $X = U\Sigma V^\top$ into the closed form: $X^\top X + \lambda I = V(\Sigma^2 +
\lambda I)V^\top$ (using $V^\top V = I$ for the thin SVD and the fact that $V$ is square and
orthogonal when $X$ has full column rank), whose inverse is $V(\Sigma^2+\lambda I)^{-1}V^\top$,
and $X^\top y = V\Sigma U^\top y$. Multiplying gives $w_\lambda = V\,\operatorname{diag}
\!\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right)U^\top y$. At $\lambda = 0$ the factor is
$1/\sigma_i$ — the pseudo-inverse — and as $\lambda$ grows each coefficient along $v_i$ is
multiplied by $\sigma_i^2/(\sigma_i^2+\lambda) < 1$, hitting the small-$\sigma_i$ directions
hardest. Because the $v_i$ are orthonormal, $\|w_\lambda\|$ is monotonically decreasing in
$\lambda$. A frequent error is to write `U.T @ y` as `U @ y` (wrong shape) or to forget that
`filt * (U.T @ y)` must multiply element-wise before the final `Vt.T @`.
""",
)

# ---------------------------------------------------------------- Ex 11 (conceptual)
ex.add_conceptual(
    title="Diagnose a numerically unsafe SVD",
    difficulty=2,
    prompt_md=r"""
A colleague computes singular values with the recipe "form $A^\top A$, call `np.linalg.eigh`,
take square roots". It agrees with `np.linalg.svd` on random matrices, so they conclude the
two are equivalent.

Explain why the recipe is numerically unsafe in general. Use the $3\times2$ matrix

$$A = \begin{pmatrix} 1 & 1 \\ \varepsilon & 0 \\ 0 & \varepsilon \end{pmatrix}, \qquad \varepsilon = 10^{-8},$$

work out its exact singular values, and predict what the recipe returns in double precision
(about 16 significant digits). What does this have to do with the condition number?
""",
    answer_md=r"""
Exactly, $A^\top A = \begin{pmatrix} 1+\varepsilon^2 & 1 \\ 1 & 1+\varepsilon^2\end{pmatrix}$ has
eigenvalues $2 + \varepsilon^2$ and $\varepsilon^2$, so the singular values are
$\sigma_1 = \sqrt{2+\varepsilon^2} \approx 1.414$ and $\sigma_2 = \varepsilon = 10^{-8}$. In
double precision, however, $1 + \varepsilon^2 = 1 + 10^{-16}$ rounds to exactly $1$, so the
computed $A^\top A$ is $\begin{pmatrix}1&1\\1&1\end{pmatrix}$ with eigenvalues $2$ and $0$ (or a
tiny negative number, whose square root is `nan`). The recipe returns $\sigma_2 = 0$ and declares
$A$ rank-deficient, although its true rank is 2 and `np.linalg.svd`, which works on $A$
directly, returns $10^{-8}$ correctly.

The mechanism is squaring: the singular values of $A^\top A$ are $\sigma_i^2$, so
$\kappa(A^\top A) = \kappa(A)^2$. Any singular value below about $\sqrt{\epsilon_{\text{mach}}}
\,\sigma_1 \approx 10^{-8}\sigma_1$ is pushed below the rounding floor $\epsilon_{\text{mach}}
\sigma_1^2 \approx 10^{-16}\sigma_1^2$ of $A^\top A$ and is lost. On random matrices all singular
values are of similar size, so the two methods agree — the difference only shows up on the
ill-conditioned matrices where accuracy matters most. The same argument says: solve least
squares via QR or SVD of $X$, never by inverting $X^\top X$.
""",
)

# ---------------------------------------------------------------- Ex 12 (project)
PROJECT_DATA = r"""
# ---- synthetic ratings: 200 users x 100 items, true latent rank 4, 25% observed ----
g = np.random.default_rng(0)
n_users, n_items, k_true = 200, 100, 4
P_true = g.normal(size=(n_users, k_true))
Q_true = g.normal(size=(n_items, k_true))
user_bias = g.normal(0, 0.5, size=n_users)
item_bias = g.normal(0, 0.5, size=n_items)
R_true = np.clip(3.5 + user_bias[:, None] + item_bias[None, :] + P_true @ Q_true.T / np.sqrt(k_true), 1, 5)
R_noisy = R_true + g.normal(0, 0.3, size=R_true.shape)
observed = g.random(R_true.shape) < 0.25           # which ratings exist
test_mask = observed & (g.random(R_true.shape) < 0.2)
train_mask = observed & ~test_mask
print(f"observed ratings: {observed.sum()} of {R_true.size} cells,"
      f" train {train_mask.sum()}, held-out test {test_mask.sum()}")
"""

ex.add_project(
    title="Mini-project: a latent-factor recommender with truncated SVD",
    difficulty=3,
    prompt_md=r"""
Recommender systems predict the ratings a user *has not* given from the ones they have. A
ratings matrix with missing entries has no SVD, so we fill the gaps and iterate:

1. Fill every missing cell with the mean of the observed **training** ratings.
2. Repeat ~30 times: take the rank-$k$ truncated SVD of the filled matrix, then overwrite the
   *missing* cells with their rank-$k$ reconstruction while keeping the observed training
   ratings fixed (each pass is an Eckart–Young projection; the iteration converges to a
   low-rank completion).
3. Predict the held-out **test** ratings from the final rank-$k$ reconstruction and report
   the RMSE.

Run this for $k \in \{1, 2, 3, 4, 5, 6, 8, 10, 15, 20\}$, plot test RMSE (and training
RMSE) against $k$, and compare with the baseline that predicts the global training mean for
everything. The cell below generates the data: 200 users, 100 items, a true latent rank of 4,
observation noise, and 25 % of cells observed; 20 % of those are held out for testing.
""",
    rubric_md=r"""
- The imputation loop keeps observed training ratings fixed and only updates missing cells.
- Training RMSE decreases monotonically with $k$, while test RMSE is U-shaped — state at
  which $k$ it is minimal and relate that to the true rank 4.
- The best model beats the global-mean baseline by a wide margin (roughly 0.45 vs 1.04).
- The plot has a title, axis labels and a legend distinguishing train, test and baseline.
- You can explain in two sentences why large $k$ overfits even though the SVD is "optimal":
  it reproduces the noise *and the imputed values* in the observed/missing cells.
- (Bonus) Replace the global-mean fill with user-mean or item-mean fills, or add the user and
  item biases, and report whether the test RMSE improves.
""",
    starter=PROJECT_DATA + r"""

# YOUR CODE HERE
# def impute_svd(R, mask, k, n_iter=30): ...
# for k in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]: fit on train_mask, evaluate RMSE on test_mask
# plot RMSE vs k with the global-mean baseline as a horizontal line
""",
    solution_sketch_md=r"""
Write `impute_svd(R, mask, k, n_iter)` that starts from `np.where(mask, R, mu)` with `mu` the
training mean, and loops: thin SVD, rank-$k$ reconstruction `(U[:, :k] * s[:k]) @ Vt[:k]`,
then `X = np.where(mask, R, X_k)`. After the loop return `X_k`. RMSE on a mask is
`np.sqrt(np.mean((pred[mask] - R[mask]) ** 2))`. The test RMSE drops steeply to about 0.45 at
$k = 5$ (one above the true rank, because the biases add low-rank structure of their own) and
then rises again as the model starts to fit noise and its own imputations; the training RMSE
keeps falling towards zero. The global-mean baseline is about 1.04.
""",
    solution_code=PROJECT_DATA + r"""

def impute_svd(R, mask, k, n_iter=30):
    mu = R[mask].mean()
    X = np.where(mask, R, mu)
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X_k = (U[:, :k] * s[:k]) @ Vt[:k]
        X = np.where(mask, R, X_k)           # keep observed ratings, update the gaps
    return X_k

def rmse(pred, R, mask):
    return float(np.sqrt(np.mean((pred[mask] - R[mask]) ** 2)))

ks = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
train_rmse, test_rmse = [], []
for k in ks:
    pred = impute_svd(R_noisy, train_mask, k)
    train_rmse.append(rmse(pred, R_noisy, train_mask))
    test_rmse.append(rmse(pred, R_noisy, test_mask))
    print(f"k = {k:2d}: train RMSE {train_rmse[-1]:.3f}   test RMSE {test_rmse[-1]:.3f}")

baseline = rmse(np.full_like(R_noisy, R_noisy[train_mask].mean()), R_noisy, test_mask)
best = ks[int(np.argmin(test_rmse))]
print(f"\nglobal-mean baseline test RMSE: {baseline:.3f}")
print(f"best rank k = {best} with test RMSE {min(test_rmse):.3f}  (true latent rank 4 + biases)")

plt.plot(ks, train_rmse, "s--", label="train RMSE")
plt.plot(ks, test_rmse, "o-", label="held-out test RMSE")
plt.axhline(baseline, color="gray", ls=":", label="global-mean baseline")
plt.title("Latent-factor recommender: RMSE vs rank k")
plt.xlabel("rank k of the truncated SVD"); plt.ylabel("RMSE (stars)"); plt.legend(); plt.show()
""",
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print("exercises:", len(ex.items), "| code:", kinds.count("code"), "| concept:", kinds.count("concept"),
      "| project:", kinds.count("project"))
print("difficulties:", [it["difficulty"] for it in ex.items])
