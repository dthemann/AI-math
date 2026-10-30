"""
build_09.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for

    part-3-probability/09-multivariate-probability

Run from the repository root:

    python3 _authoring/builders/build_09.py
    python3 tools/run_notebooks.py part-3-probability/09-multivariate-probability

All hidden answers are computed here by *executing* the reference solutions and hashing
the result with `sha_of`, so the exercises notebook never contains the answers in clear text.
"""
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import ExerciseSet, Notebook, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-3-probability/09-multivariate-probability"

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 09 — Multivariate Probability")

nb.md(r"""
This notebook is the runnable companion to the concept guide: read [README.md](README.md) first, or
keep it open alongside. Every section mirrors a numbered section of the guide. We take a joint table
apart into marginals and conditionals, measure linear dependence with covariance and correlation (and
watch correlation miss a perfectly deterministic relationship), build covariance matrices from scratch,
sample a two-dimensional Gaussian with a hand-made Cholesky factor and draw its ellipses from the
eigendecomposition of $\Sigma$, visualise Gaussian conditioning, verify the law of total variance,
construct Simpson's paradox, and finish with the two famous applications of Gaussian conditioning:
a Kalman filter and Gaussian-process regression.

Everything is plain NumPy first; `scipy.stats` and scikit-learn appear only to confirm the hand computation.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
""")

# ---------------------------------------------------------------- 1. joint tables
nb.md(r"""
## 1. Joint, marginal and conditional distributions — a table taken apart

The joint PMF of (age group, purchases last month) from the guide. Marginalising is summing along an
axis, conditioning is dividing a row (or column) by its sum, and independence means the table equals
the outer product of its marginals:

$$p_X(x) = \sum_y p(x, y), \qquad p(y \mid x) = \frac{p(x, y)}{p_X(x)}, \qquad \text{independent} \iff p(x, y) = p_X(x)\,p_Y(y).$$
""")

nb.code(r"""
ages = ["18-34", "35-54", "55+"]
purchases = ["none", "one", "several"]
J = np.array([[0.20, 0.10, 0.05],
              [0.15, 0.15, 0.10],
              [0.15, 0.07, 0.03]])
print("joint table sums to", J.sum())

p_age = J.sum(axis=1)            # marginal of X: sum out Y (along each row)
p_purch = J.sum(axis=0)          # marginal of Y: sum out X (down each column)
print("P(age)      =", dict(zip(ages, p_age.round(3))))
print("P(purchase) =", dict(zip(purchases, p_purch.round(3))))

cond_Y_given_X = J / p_age[:, None]        # renormalise each row
print("\nP(purchase | age) — every row sums to 1:")
for a, row in zip(ages, cond_Y_given_X):
    print(f"  {a:6s}", dict(zip(purchases, row.round(3))))
cond_X_given_Y = J / p_purch[None, :]      # renormalise each column  (= Bayes' theorem in table form)
print(f"\nP(age = 55+ | several) = {cond_X_given_Y[2, 2]:.4f}   vs the prior P(55+) = {p_age[2]:.2f}")

indep = np.outer(p_age, p_purch)           # what the table would be if age and purchases were independent
print("\ntable under independence (outer product of the marginals):\n", indep.round(3))
print("largest |joint - independent| =", np.abs(J - indep).max().round(3), " -> dependent")
print("rank of the joint table:", np.linalg.matrix_rank(J), "   rank of the independent table:", np.linalg.matrix_rank(indep))
""")

nb.code(r"""
# conditioning is *filtering*: draw customers from the joint table, keep the 35-54 group, count purchases
n = 100_000
cell = rng.choice(J.size, size=n, p=J.ravel())
age_idx, purch_idx = np.divmod(cell, 3)
keep = age_idx == 1
emp_cond = np.array([(purch_idx[keep] == k).mean() for k in range(3)])
print(f"{keep.sum()} of {n} simulated customers are 35-54  (exact marginal 0.40)")
print("empirical P(purchase | 35-54) by filtering:", emp_cond.round(3), "   exact:", cond_Y_given_X[1].round(3))
""")

nb.code(r"""
# independence is a statement about a whole table: compare the joint with the outer product of its marginals
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
panels = [(J, "joint table p(age, purchases)", "Blues", 0.0, 0.20, "{:.3f}"),
          (indep, "outer product p(age) p(purchases)", "Blues", 0.0, 0.20, "{:.3f}"),
          (J - indep, "difference: joint - outer product", "RdBu_r", -0.05, 0.05, "{:+.3f}")]
for ax, (table, title, cmap, vmin, vmax, fmt) in zip(axes, panels):
    im = ax.imshow(table, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(3):
        for j in range(3):
            dark = abs(table[i, j]) > 0.6 * vmax               # white text on the dark cells
            ax.text(j, i, fmt.format(table[i, j]), ha="center", va="center", fontsize=9, color="white" if dark else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(purchases); ax.set_yticks(range(3)); ax.set_yticklabels(ages)
    ax.set_xlabel("purchases last month"); ax.set_ylabel("age group"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** the difference panel is nowhere zero, and that the 35–54 row carries the biggest departures
($-0.05$ under "none", $+0.028$ under "several"). That is exactly the row whose conditional
$(0.375, 0.375, 0.25)$ differs most from the marginal $(0.50, 0.32, 0.18)$: dependence *is* the gap between the
table and the outer product.
""")

nb.md(r"""
The middle row renormalised by $0.40$ gives $(0.375, 0.375, 0.25)$, and filtering the simulated
customers lands close to it: $(0.371, 0.379, 0.250)$. The outer-product table differs from the joint in every cell (by up to $0.05$, in the
35–54/none cell; the smallest gap is $0.01$), and
it has rank 1 while the real table has rank 3 — a joint table of independent variables is always a rank-1
matrix, because every row is a rescaled copy of the marginal $p_Y$.

**Try it yourself:** replace `J` by `indep` in the conditioning cell — every conditional row should now
equal `p_purch`.
""")

# ---------------------------------------------------------------- 2. covariance and correlation
nb.md(r"""
## 2. Covariance and correlation — and what correlation cannot see

$$\operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)], \qquad \rho = \frac{\operatorname{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1].$$

Four scatter plots: three Gaussian pairs with different $\rho$, and the pair $Y = X^2$, which is
perfectly dependent yet has correlation zero. Spearman's rank correlation is shown alongside.
""")

nb.code(r"""
n = 400
def corr_pair(rho, n):
    z = rng.standard_normal((n, 2))
    return z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]      # exact correlation rho

cases = [("rho = 0.9", corr_pair(0.9, n)), ("rho = -0.5", corr_pair(-0.5, n)), ("rho = 0 (independent)", corr_pair(0.0, n))]
x = rng.standard_normal(n)
x = np.concatenate([x[:n // 2], -x[:n // 2]])                  # mirror half the sample, so the cloud is exactly symmetric
cases.append(("Y = X^2 (dependent!)", (x, x**2)))

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, (name, (x, y)) in zip(axes, cases):
    r, rs = np.corrcoef(x, y)[0, 1], stats.spearmanr(x, y)[0]
    ax.scatter(x, y, s=6, alpha=0.6)
    ax.set_title(f"{name}\nPearson r = {r:.2f}, Spearman = {rs:.2f}", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout(); plt.show()
""")

nb.code(r"""
# the guide's height/weight example, step by step
h = np.array([160, 165, 170, 175, 180.]); w = np.array([55, 62, 66, 70, 82.])
dh, dw = h - h.mean(), w - w.mean()
print("deviations  h:", dh, "  w:", dw)
print("products   :", dh * dw, "  sum =", (dh * dw).sum())
cov_hw = (dh * dw).sum() / (len(h) - 1)
r_hw = (dh * dw).sum() / np.sqrt((dh**2).sum() * (dw**2).sum())
print(f"sample covariance = {cov_hw:.2f} cm*kg    correlation = {r_hw:.4f}    np.corrcoef gives {np.corrcoef(h, w)[0, 1]:.4f}")
print(f"measured in metres and grams: covariance = {np.cov(h/100, w*1000)[0, 1]:.2f}  but correlation = {np.corrcoef(h/100, w*1000)[0, 1]:.4f}")

# zero correlation without independence, exactly
X = np.array([-1, 0, 1]); Y = X**2
print(f"\nX uniform on {{-1, 0, 1}}, Y = X^2:   E[XY] = {np.mean(X*Y):.3f}   E[X]E[Y] = {X.mean()*Y.mean():.3f}   -> Cov(X, Y) = 0")
print(f"but P(X=1, Y=1) = 1/3 = {1/3:.3f}   while   P(X=1) P(Y=1) = {(1/3)*(2/3):.3f}   -> dependent")

# bilinearity checked by simulation
A = rng.standard_normal(50_000); B = rng.standard_normal(50_000); C = 0.5*A + rng.standard_normal(50_000)
lhs = np.cov(2*A + 3*B, C)[0, 1]; rhs = 2*np.cov(A, C)[0, 1] + 3*np.cov(B, C)[0, 1]
print(f"\nbilinearity: Cov(2A + 3B, C) = {lhs:.4f}    2 Cov(A, C) + 3 Cov(B, C) = {rhs:.4f}")
print(f"Var(A + C) = {np.var(A + C, ddof=1):.4f}    Var A + Var C + 2 Cov = {np.var(A, ddof=1) + np.var(C, ddof=1) + 2*np.cov(A, C)[0, 1]:.4f}")
""")

nb.md(r"""
The fourth panel is the point of the section: $Y$ is a deterministic function of $X$, yet Pearson's $r$
and even Spearman's rank correlation are both zero (we drew a sample symmetric about $0$, so each point cancels
with its mirror image; an ordinary sample of 400 would wobble a little around zero), because the relationship
is neither linear nor monotone. Correlation measures how well a *straight line* describes the cloud, nothing more. Covariance
changes with the units (77.5 cm·kg becomes 775 m·g); correlation does not — it is the covariance of the
standardised variables.

**Try it yourself:** in `corr_pair`, generate $y = \sin(3x)$ plus a little noise and look at Pearson vs
Spearman; then try $y = e^{x}$, which is monotone — Spearman should jump to 1 while Pearson stays below.
""")

# ---------------------------------------------------------------- 3. covariance matrices
nb.md(r"""
## 3. Random vectors and covariance matrices

$$\Sigma = \mathbb{E}\big[(X - \mu)(X - \mu)^\top\big], \qquad \operatorname{Var}(a^\top X) = a^\top \Sigma a \ge 0, \qquad S = \frac{1}{n-1}\mathbf{X}_c^\top \mathbf{X}_c .$$

We generate 2,000 observations of three correlated "asset returns", build the sample covariance from the
centred data matrix, and compare with `np.cov`. Then the three-asset portfolio of the guide, a
positive-semidefiniteness check, and whitening.
""")

nb.code(r"""
sd = np.array([0.20, 0.10, 0.30])                              # volatilities of three assets
R = np.array([[1.0, 0.3, 0.5],
              [0.3, 1.0, 0.0],
              [0.5, 0.0, 1.0]])                                # correlation matrix
Sigma_true = np.outer(sd, sd) * R                              # Sigma_ij = rho_ij sigma_i sigma_j
mu_true = np.array([0.08, 0.03, 0.10])
X = rng.multivariate_normal(mu_true, Sigma_true, size=2000)    # rows = observations, columns = variables

Xc = X - X.mean(axis=0)                                        # centre every column
S = Xc.T @ Xc / (len(X) - 1)                                   # sample covariance, from scratch
print("sample covariance S (from scratch):\n", S.round(4))
print("np.cov(X, rowvar=False):\n", np.cov(X, rowvar=False).round(4))
print("largest difference:", np.abs(S - np.cov(X, rowvar=False)).max())
print("true Sigma:\n", Sigma_true.round(4))
print("\nsample correlation matrix (D^-1/2 S D^-1/2):\n", (S / np.sqrt(np.outer(np.diag(S), np.diag(S)))).round(3))
""")

nb.code(r"""
w = np.array([0.5, 0.3, 0.2])                                  # portfolio weights
print("Sigma w =", (Sigma_true @ w).round(4))
var_p = w @ Sigma_true @ w
print(f"portfolio variance w'Sigma w = {var_p:.4f}   volatility = {np.sqrt(var_p):.4f}")
print(f"weighted average of the volatilities = {w @ sd:.4f};  if the assets were uncorrelated: {np.sqrt(w**2 @ sd**2):.4f}")
print(f"empirical variance of the simulated portfolio returns X w: {np.var(X @ w, ddof=1):.4f}   vs  w'S w = {w @ S @ w:.4f}")

# a covariance matrix is positive semidefinite: a'Sigma a = Var(a'X) >= 0 for *every* a
print("\neigenvalues of Sigma:", np.linalg.eigvalsh(Sigma_true).round(4), "  all >= 0")
dirs = rng.standard_normal((5000, 3))
quad = np.einsum("ij,jk,ik->i", dirs, Sigma_true, dirs)        # a'Sigma a for 5000 random directions
print(f"smallest a'Sigma a over 5000 random directions: {quad.min():.6f}")
bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
print("eigenvalues of an impossible 'correlation matrix' (rho12 = rho13 = 0.9, rho23 = -0.9):", np.linalg.eigvalsh(bad).round(3))
a = np.array([1, -1, -1.0]); print(f"  it would give Var(X1 - X2 - X3) = a'Ra = {a @ bad @ a:.2f} < 0")

# crisis scenario: all correlations rise to 0.8
Sigma_crisis = np.outer(sd, sd) * np.where(np.eye(3) == 1, 1.0, 0.8)
print(f"\ncrisis (all rho = 0.8): volatility = {np.sqrt(w @ Sigma_crisis @ w):.4f}   diversification benefit shrinks from "
      f"{w @ sd - np.sqrt(var_p):.4f} to {w @ sd - np.sqrt(w @ Sigma_crisis @ w):.4f}")
""")

nb.code(r"""
# whitening: Z = Sigma^{-1/2} (X - mu) has identity covariance
lam, U = np.linalg.eigh(S)
W = U @ np.diag(lam ** -0.5) @ U.T                             # symmetric inverse square root S^{-1/2}
Z = Xc @ W.T
print("covariance of the whitened data:\n", np.cov(Z, rowvar=False).round(3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.scatter(Xc[:, 0], Xc[:, 2], s=4, alpha=0.4)
ax1.set_title(f"centred assets 1 and 3 (r = {np.corrcoef(X[:, 0], X[:, 2])[0, 1]:.2f})"); ax1.set_xlabel("asset 1 return"); ax1.set_ylabel("asset 3 return"); ax1.set_aspect("equal")
ax2.scatter(Z[:, 0], Z[:, 2], s=4, alpha=0.4)
ax2.set_title(f"after whitening (r = {np.corrcoef(Z[:, 0], Z[:, 2])[0, 1]:.2f})"); ax2.set_xlabel("whitened coordinate 1"); ax2.set_ylabel("whitened coordinate 3"); ax2.set_aspect("equal")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The from-scratch $S$ matches `np.cov(X, rowvar=False)` to machine precision (remember: `rowvar=False` or
`np.cov(X.T)` — NumPy's default treats *rows* as variables). The portfolio volatility, $14.9\%$, lies
between the uncorrelated value ($12.0\%$) and the weighted average ($19\%$); the two positive correlations
cost almost three points, and in the crisis scenario the benefit of diversifying shrinks to less than a third
(from $4.1$ to $1.2$ points of volatility). The "correlation matrix" with $0.9, 0.9, -0.9$ has a negative
eigenvalue and would assign a *negative variance* ($-2.4$) to $X_1 - X_2 - X_3$, the direction of its $-0.8$
eigenvector — it cannot describe any random vector. Whitening by $S^{-1/2}$ turns the tilted cloud into
a round one with identity covariance.

**Try it yourself:** set `size=3` in the sampling call and look at `np.linalg.eigvalsh(S)` — with no more
observations than dimensions ($n \le d$) the centred data matrix has rank at most $n - 1 < d$, so the sample
covariance is singular (an eigenvalue of zero, up to round-off). Try `size=4` too: the smallest eigenvalue
is no longer zero, though still well below the true $0.0083$ (about $0.001$ on this seed).
""")

# ---------------------------------------------------------------- 4. multivariate Gaussian
nb.md(r"""
## 4. The multivariate Gaussian: Cholesky sampling and the geometry of $\Sigma$

$$f(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\Big(-\tfrac12 (x-\mu)^\top \Sigma^{-1} (x-\mu)\Big), \qquad X = \mu + L z \sim \mathcal{N}(\mu, LL^\top).$$

For $\Sigma = \begin{pmatrix} 2 & 1 \\ 1 & 1\end{pmatrix}$ we compute the Cholesky factor by hand, draw
4,000 samples, and overlay the ellipses $\Delta^2 = 1, 4, 5.99$ whose axes are the eigenvectors of $\Sigma$
scaled by $\sqrt{c\lambda_i}$.
""")

nb.code(r"""
Sigma = np.array([[2.0, 1.0], [1.0, 1.0]])
a, b, c = Sigma[0, 0], Sigma[0, 1], Sigma[1, 1]
L = np.array([[np.sqrt(a), 0.0],
              [b / np.sqrt(a), np.sqrt(c - b**2 / a)]])     # Cholesky of a 2x2 matrix, by hand
print("L by hand:\n", L.round(4), "\nnp.linalg.cholesky:\n", np.linalg.cholesky(Sigma).round(4))
print("L L^T =\n", L @ L.T)

z = rng.standard_normal((4000, 2))
X2 = z @ L.T                                                   # every row is x = L z   (mu = 0)
print("\nsample mean:", X2.mean(axis=0).round(3), "   sample covariance:\n", np.cov(X2, rowvar=False).round(3))

lam, U = np.linalg.eigh(Sigma)                                 # eigenvalues ascending
u_major = U[:, 1] * np.sign(U[0, 1])                           # fix the arbitrary sign
print(f"\neigenvalues: {lam.round(4)}   = (3 -/+ sqrt5)/2 = {((3 - np.sqrt(5))/2):.4f}, {((3 + np.sqrt(5))/2):.4f}")
print(f"major axis direction {u_major.round(3)} at {np.degrees(np.arctan2(u_major[1], u_major[0])):.1f} degrees;  1-sd semi-axes = sqrt(lambda) = {np.sqrt(lam).round(3)}")

def ellipse(mu, Sigma, c, n_pts=200):
    # points x with (x - mu)' Sigma^{-1} (x - mu) = c:  the unit circle mapped by U diag(sqrt(c lambda))
    lam, U = np.linalg.eigh(Sigma)
    t = np.linspace(0, 2*np.pi, n_pts)
    return mu[:, None] + U @ np.diag(np.sqrt(c * lam)) @ np.stack([np.cos(t), np.sin(t)])

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(X2[:, 0], X2[:, 1], s=3, alpha=0.3, label="4000 samples x = Lz")
for cval, lab in [(1.0, "1 sd  (Δ² = 1)"), (4.0, "2 sd  (Δ² = 4)"), (stats.chi2.ppf(0.95, 2), "95%  (Δ² = 5.99)")]:
    e = ellipse(np.zeros(2), Sigma, cval); ax.plot(e[0], e[1], lw=2, label=lab)
for i in range(2):
    ax.annotate("", xy=tuple(np.sqrt(lam[i]) * U[:, i] * np.sign(U[0, i])), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=2))
ax.plot([], [], "k-", marker=">", lw=2, label="eigenvectors scaled by sqrt(λ)")
ax.set_aspect("equal"); ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("N(0, [[2,1],[1,1]]) — ellipses from the eigendecomposition of Σ"); plt.show()

d2 = np.einsum("ij,jk,ik->i", X2, np.linalg.inv(Sigma), X2)    # squared Mahalanobis distance of every sample
print(f"fraction of samples inside the 95% ellipse: {(d2 <= stats.chi2.ppf(0.95, 2)).mean():.3f}   (Δ² ~ chi-square with 2 degrees of freedom)")
""")

nb.code(r"""
# sweep the direction a around the half circle: a and -a give the same variance, so 180 degrees is enough
th = np.linspace(0, np.pi, 181)
A = np.stack([np.cos(th), np.sin(th)])                         # 181 unit vectors, as columns
formula = np.einsum("ij,jk,ki->i", A.T, Sigma, A)              # a'Sigma a for every direction
empirical = np.var(X2 @ A, axis=0, ddof=1)                     # variance of the projected samples
deg = np.degrees(th)
ang_max = np.degrees(np.arctan2(U[1, 1], U[0, 1])) % 180
ang_min = np.degrees(np.arctan2(U[1, 0], U[0, 0])) % 180

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(deg, formula, "k", lw=2, label="a'Σa (formula)")
ax.plot(deg, empirical, color="C1", ls="--", lw=2, label="Var(a'x) of the 4000 samples")
for ang, val, name, y_txt in [(ang_max, lam[1], "major", 2.78), (ang_min, lam[0], "minor", 0.14)]:
    ax.axvline(ang, color="grey", ls=":", lw=1); ax.axhline(val, color="grey", ls=":", lw=1)
    ax.text(ang, y_txt, f"{name} axis: λ = {val:.3f} at {ang:.1f}°", ha="center", fontsize=9, bbox=dict(fc="white", ec="none", pad=1))
ax.set_xlim(0, 180); ax.set_ylim(0, 3)
ax.set_xlabel("direction of a = (cos θ, sin θ): θ in degrees"); ax.set_ylabel("variance of the projection a'X")
ax.set_title("Var(a'X) = a'Σa is never negative and is extreme along the eigenvectors"); ax.legend(fontsize=8, loc="lower left")
plt.show()
""")

nb.md(r"""
**Notice that** the sample variances of the projected data trace the quadratic form closely, that the curve never
dips below zero (Section 3's positive semidefiniteness made visible), and that its maximum and minimum occur
precisely at the eigenvector angles with the eigenvalues as values — the same numbers that set the ellipse's
semi-axes.
""")

nb.code(r"""
def mvn_logpdf(x, mu, Sigma):
    # log density via the Cholesky factor: never invert Sigma explicitly
    L = np.linalg.cholesky(Sigma)
    y = np.linalg.solve(L, x - mu)                 # L^{-1}(x - mu); then y'y = (x-mu)' Sigma^{-1} (x-mu)
    log_det = 2.0 * np.log(np.diag(L)).sum()       # log|Sigma| = 2 sum_i log L_ii
    return -0.5 * (len(mu) * np.log(2 * np.pi) + log_det + y @ y)

ref = stats.multivariate_normal(np.zeros(2), Sigma)
for x in (np.array([1.0, 1.0]), np.array([1.0, -1.0]), np.zeros(2)):
    print(f"x = {x}:  Mahalanobis^2 = {x @ np.linalg.inv(Sigma) @ x:.1f}   density = {np.exp(mvn_logpdf(x, np.zeros(2), Sigma)):.4f}   scipy: {ref.pdf(x):.4f}")
ratio = np.exp(mvn_logpdf(np.array([1.0, 1.0]), np.zeros(2), Sigma) - mvn_logpdf(np.array([1.0, -1.0]), np.zeros(2), Sigma))
print(f"f(1, 1) / f(1, -1) = {ratio:.3f} = e^2:  same Euclidean length, very different probability")
""")

nb.md(r"""
Roughly 95% of the samples fall inside the $\Delta^2 = 5.99$ ellipse, as the $\chi^2_2$ law predicts. The
arrows are the eigenvectors scaled by $\sqrt{\lambda_i}$: the long axis points along $(0.85, 0.53)$ because
the correlation is positive, and the ratio of the semi-axes is $\sqrt{2.618/0.382} = 2.6$. The two points
$(1, 1)$ and $(1, -1)$ are equally far from the origin in Euclidean terms, but one lies along the cloud and
the other across it — the Mahalanobis distances $1$ and $5$ (hence densities differing by $e^2$) are what the
Gaussian actually sees.

**Try it yourself:** change the off-diagonal of `Sigma` to $-1$ and re-run; then to $1.5$ ($1.4$ is still, just
barely, positive definite, with determinant $0.04$) — what happens to the Cholesky factor when $\Sigma$ stops
being positive definite?
""")

# ---------------------------------------------------------------- 5. conditioning
nb.md(r"""
## 5. Marginalisation and conditioning in Gaussians

$$\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2), \qquad \Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}.$$

We implement the formula for any split of the coordinates, apply it to $\Sigma = \begin{pmatrix} 2 & 1 \\ 1 & 1\end{pmatrix}$,
and check it against brute force: filter 200,000 samples to those with $x_2 \approx 1.5$ and look at their $x_1$.
""")

nb.code(r"""
def gaussian_condition(mu, Sigma, obs_idx, obs_val):
    # distribution of the remaining coordinates given X[obs_idx] = obs_val
    mu = np.asarray(mu, float); Sigma = np.asarray(Sigma, float)
    obs_idx = list(obs_idx); free = [i for i in range(len(mu)) if i not in obs_idx]
    S11 = Sigma[np.ix_(free, free)]; S12 = Sigma[np.ix_(free, obs_idx)]; S22 = Sigma[np.ix_(obs_idx, obs_idx)]
    gain = S12 @ np.linalg.inv(S22)                            # Sigma_12 Sigma_22^{-1}: the regression coefficients
    mu_c = mu[free] + gain @ (np.asarray(obs_val, float) - mu[obs_idx])
    Sigma_c = S11 - gain @ S12.T                               # Schur complement
    return mu_c, Sigma_c

mu_c, S_c = gaussian_condition(np.zeros(2), Sigma, [1], [1.5])
print(f"X1 | X2 = 1.5  ~  N({mu_c[0]:.3f}, {S_c[0, 0]:.3f})       marginal of X1 is N(0, 2)")
mu_c2, S_c2 = gaussian_condition(np.zeros(2), Sigma, [0], [2.0])
print(f"X2 | X1 = 2.0  ~  N({mu_c2[0]:.3f}, {S_c2[0, 0]:.3f})       marginal of X2 is N(0, 1)")
print(f"both variances shrink by the factor 1 - rho^2 = {1 - Sigma[0, 1]**2 / (Sigma[0, 0] * Sigma[1, 1]):.3f}")

Xbig = rng.standard_normal((200_000, 2)) @ L.T
band = np.abs(Xbig[:, 1] - 1.5) < 0.05                        # brute force: keep samples with x2 close to 1.5
print(f"\n{band.sum()} samples have x2 within 1.5 +/- 0.05:  mean of x1 = {Xbig[band, 0].mean():.3f},  variance = {Xbig[band, 0].var():.3f}")
far = np.abs(Xbig[:, 1] - 3.0) < 0.05                         # an extreme observation: same variance, shifted mean
print(f"{far.sum()} samples have x2 within 3.0 +/- 0.05:  mean of x1 = {Xbig[far, 0].mean():.3f},  variance = {Xbig[far, 0].var():.3f}   (formula: 3.0, 1.0)")
""")

nb.code(r"""
g = np.linspace(-4, 4, 200); G1, G2 = np.meshgrid(g, g)
dens = ref.pdf(np.stack([G1.ravel(), G2.ravel()], axis=1)).reshape(G1.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.contour(G1, G2, dens, levels=8)
ax1.axhline(1.5, color="r", lw=2, label="observe x2 = 1.5")
ax1.set_xlabel("x1"); ax1.set_ylabel("x2"); ax1.set_title("joint density and the conditioning slice"); ax1.legend(); ax1.set_aspect("equal")
ax2.plot(g, stats.norm.pdf(g, 0, np.sqrt(2)), label="marginal of X1: N(0, 2)")
ax2.plot(g, stats.norm.pdf(g, mu_c[0], np.sqrt(S_c[0, 0])), "r", lw=2, label=f"X1 | X2 = 1.5: N({mu_c[0]:.1f}, {S_c[0, 0]:.1f})")
ax2.hist(Xbig[band, 0], bins=40, density=True, alpha=0.3, color="r", label="filtered samples")
ax2.set_xlabel("x1"); ax2.set_ylabel("density"); ax2.set_title("conditioning shifts the mean and narrows the spread"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.code(r"""
# height (cm) and weight (kg): mean (170, 70), sds (10, 12), correlation 0.6
mu_hw = np.array([170.0, 70.0]); S_hw = np.array([[100.0, 72.0], [72.0, 144.0]])
for h_obs in (155, 170, 185):
    m, v = gaussian_condition(mu_hw, S_hw, [0], [h_obs])
    print(f"height {h_obs} cm  ->  weight ~ N({m[0]:.1f}, sd {np.sqrt(v[0, 0]):.2f})")
m, v = gaussian_condition(mu_hw, S_hw, [1], [90.0])
print(f"weight 90 kg   ->  height ~ N({m[0]:.1f}, sd {np.sqrt(v[0, 0]):.2f})")
slope_w_on_h = S_hw[0, 1] / S_hw[0, 0]; slope_h_on_w = S_hw[0, 1] / S_hw[1, 1]
print(f"\nregression slopes: weight on height {slope_w_on_h:.2f} kg/cm,  height on weight {slope_h_on_w:.2f} cm/kg,  product = {slope_w_on_h * slope_h_on_w:.2f} = rho^2 (not 1)")
""")

nb.code(r"""
# the two regression lines and the long axis of the ellipse are three different lines
rng_hw = np.random.default_rng(5)                              # a private generator: the shared rng stays untouched
HW = mu_hw + rng_hw.standard_normal((1500, 2)) @ np.linalg.cholesky(S_hw).T
lam_hw, U_hw = np.linalg.eigh(S_hw)
slope_axis = U_hw[1, 1] / U_hw[0, 1]
hh = np.linspace(140, 200, 2)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(HW[:, 0], HW[:, 1], s=4, alpha=0.3, label="1500 people")
e = ellipse(mu_hw, S_hw, stats.chi2.ppf(0.95, 2)); ax.plot(e[0], e[1], "k", lw=1, label="95% ellipse")
ax.plot(hh, 70 + slope_axis * (hh - 170), "k--", lw=1.5, label=f"long axis of the ellipse: slope {slope_axis:.2f}")
ax.plot(hh, 70 + slope_w_on_h * (hh - 170), "r", lw=2, label=f"E[W | H]: slope {slope_w_on_h:.2f} kg/cm")
ax.plot(hh, 70 + (hh - 170) / slope_h_on_w, "g", lw=2, label=f"E[H | W] read sideways: slope 1/{slope_h_on_w:.2f} = {1 / slope_h_on_w:.1f}")
edges = np.arange(145, 200, 5.0)
idx = np.digitize(HW[:, 0], edges)
centres = [edges[k - 1] + 2.5 for k in range(1, len(edges)) if (idx == k).sum() >= 5]
means = [HW[idx == k, 1].mean() for k in range(1, len(edges)) if (idx == k).sum() >= 5]
ax.plot(centres, means, "s", color="r", ms=7, mec="k", label="mean weight in 5-cm height bins")
ax.set_xlim(135, 205); ax.set_ylim(25, 132)
ax.set_xlabel("height H (cm)"); ax.set_ylabel("weight W (kg)")
ax.set_title("Regression to the mean: E[W | H] is flatter than the cloud's axis", fontsize=11); ax.legend(fontsize=8, loc="upper left")
plt.show()
""")

nb.md(r"""
**Notice that** the binned means (squares) sit on the flatter red line, not on the ellipse's long axis: for any
given height the centre of the cloud is pulled back towards the mean weight — that is regression to the mean. The
green line is the other regression (height on weight) read sideways; the two cross only at the mean, and the
regression slopes ($0.72$ kg/cm and $0.50$ cm/kg, the latter drawn here as $1/0.50 = 2.0$) multiply to
$\rho^2 = 0.36$, so neither line is the inverse of the other.
""")

nb.md(r"""
The filtered samples near $x_2 = 1.5$ have mean $\approx 1.5$ and variance $\approx 1$ — half the marginal
variance of 2 — and the samples near the much rarer $x_2 = 3$ have the *same* variance with a shifted mean:
in a Gaussian the conditional covariance never depends on the observed value. In the two-panel plot the red
conditional is both displaced and narrower than the blue marginal. The height/weight numbers show
regression to the mean: a person $1.5$ standard deviations tall is predicted to be only $0.9$ standard
deviations heavy, and the two regression slopes multiply to $\rho^2 = 0.36$, so you cannot invert one
regression line to get the other.
""")

# ---------------------------------------------------------------- 6. conditional expectation, total variance
nb.md(r"""
## 6. Conditional expectation and the law of total variance

$$\operatorname{Var}(Y) = \mathbb{E}\big[\operatorname{Var}(Y \mid X)\big] + \operatorname{Var}\big(\mathbb{E}[Y \mid X]\big), \qquad
\mathbb{E}[Y \mid X] = \arg\min_g \mathbb{E}\big[(Y - g(X))^2\big].$$

First the insurer's two segments from the guide (60% with $\mathcal{N}(10, 4)$, 40% with $\mathcal{N}(20, 9)$):
the within and between parts should sum to $6 + 24 = 30$. Then a nonlinear regression function, where we
compare the mean squared error of several predictors with the true $\mathbb{E}[Y \mid X]$.
""")

nb.code(r"""
n = 200_000
in_B = rng.random(n) < 0.4                                     # segment B with probability 0.4
Y = np.where(in_B, rng.normal(20, 3, n), rng.normal(10, 2, n))
cond_mean = np.where(in_B, 20.0, 10.0)                         # E[Y | X] as a random variable
cond_var = np.where(in_B, 9.0, 4.0)                            # Var(Y | X) as a random variable
print(f"Var(Y) by simulation            = {Y.var():.3f}")
print(f"E[Var(Y|X)] (within)            = {cond_var.mean():.3f}     exact 0.6*4 + 0.4*9 = 6")
print(f"Var(E[Y|X]) (between)           = {cond_mean.var():.3f}    exact 0.6*16 + 0.4*36 = 24")
print(f"within + between                = {cond_var.mean() + cond_mean.var():.3f}")
print(f"tower property: E[E[Y|X]] = {cond_mean.mean():.3f} = E[Y] = {Y.mean():.3f}")
print(f"fraction of variance explained by the segment: {cond_mean.var() / Y.var():.3f}")

# Gaussian check: for Sigma = [[2,1],[1,1]], Var(X1|X2) = 1 and Var(E[X1|X2]) = Sigma_12^2 / Sigma_22 = 1, summing to Sigma_11 = 2
between_g = Sigma[0, 1]**2 / Sigma[1, 1]                       # Var(E[X1|X2]) = Sigma_12^2 / Sigma_22
print(f"\nGaussian: E[Var(X1|X2)] + Var(E[X1|X2]) = {S_c[0, 0]:.1f} + {between_g:.1f} = {S_c[0, 0] + between_g:.1f} = Sigma_11")
""")

nb.code(r"""
# the same decomposition as a picture (no new random numbers: Y and in_B from the cell above)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2.2, 1]})
ax1.hist([Y[~in_B], Y[in_B]], bins=np.linspace(0, 32, 129), stacked=True, color=["C0", "C1"],
         label=["segment A: 60%, N(10, 4)", "segment B: 40%, N(20, 9)"])
ax1.set_ylim(0, 8200)
for xv, col, txt, ha, dx in [(10, "C0", "E[Y | A] = 10", "right", -0.2), (14, "k", "E[Y] = 14", "left", 0.2), (20, "C1", "E[Y | B] = 20", "left", 0.2)]:
    ax1.axvline(xv, color=col, ls="--", lw=1.2)
    ax1.text(xv + dx, 7650, txt, color=col, ha=ha, fontsize=8)
ax1.annotate("", xy=(20, 6600), xytext=(10, 6600), arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
ax1.text(15, 6830, "'between': the two means differ", ha="center", fontsize=8, bbox=dict(fc="white", ec="none", pad=1))
ax1.annotate("", xy=(12, 3640), xytext=(8, 3640), arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
ax1.text(7.8, 3640, "'within': sd 2", ha="right", va="center", fontsize=8)
ax1.annotate("", xy=(23, 1610), xytext=(17, 1610), arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
ax1.text(23.2, 1610, "'within': sd 3", ha="left", va="center", fontsize=8)
ax1.set_xlabel("claim size Y"); ax1.set_ylabel("count")
ax1.set_title("Var(Y) = spread inside the segments + spread of the segment means", fontsize=10); ax1.legend(fontsize=8, loc="center right")

within = [cond_var.mean(), 6.0]; between = [cond_mean.var(), 24.0]
ax2.bar(["simulation", "exact"], within, color="C0", label="E[Var(Y | X)] (within)")
ax2.bar(["simulation", "exact"], between, bottom=within, color="C1", label="Var(E[Y | X]) (between)")
for k in range(2):
    ax2.text(k, within[k] / 2, f"{within[k]:.1f}", ha="center", va="center", color="white", fontsize=9)
    ax2.text(k, within[k] + between[k] / 2, f"{between[k]:.1f}", ha="center", va="center", color="white", fontsize=9)
ax2.axhline(Y.var(), color="k", ls=":", label=f"Var(Y) by simulation = {Y.var():.1f}")
ax2.set_ylim(0, 38); ax2.set_ylabel("variance"); ax2.set_title("within + between = total", fontsize=10); ax2.legend(fontsize=7, loc="upper center", ncol=1)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** each hump has only the "within" spread (sd $2$ and $3$), while the gap between their centres is the
"between" part — and that the bar shows the between part dominating: $24$ of the $30$ units of variance come from
*which* segment a customer is in, which is why knowing the segment explains $80\%$ of the variance.
""")

nb.code(r"""
n = 5000
x = rng.uniform(-2, 2, n)
m_true = np.sin(1.5 * x) + 0.3 * x**2                          # the regression function E[Y | X = x]
y = m_true + rng.normal(0, 0.4, n)                             # Var(Y | X) = 0.16, the noise floor

predictors = {"constant E[Y]": np.full(n, y.mean()),
              "best straight line": np.polyval(np.polyfit(x, y, 1), x),
              "cubic polynomial": np.polyval(np.polyfit(x, y, 3), x),
              "true E[Y | X]": m_true}
for name, pred in predictors.items():
    print(f"{name:20s} MSE = {np.mean((y - pred)**2):.4f}")
print(f"{'noise floor':20s} E[Var(Y|X)] = 0.4^2 = {0.16:.4f}")

order = np.argsort(x)
plt.scatter(x, y, s=3, alpha=0.25, label="data")
plt.plot(x[order], predictors["best straight line"][order], lw=2, label="best straight line")
plt.plot(x[order], predictors["cubic polynomial"][order], lw=2, label="cubic polynomial")
plt.plot(x[order], m_true[order], "k", lw=2, label="E[Y | X] (best possible)")
plt.xlabel("x"); plt.ylabel("y"); plt.title("The conditional mean is the best MSE predictor"); plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
Within plus between reproduces the total variance, and the segment explains $24/30 = 80\%$ of it. In the
second experiment no predictor beats the true conditional mean, whose MSE sits at the noise floor
$\mathbb{E}[\operatorname{Var}(Y \mid X)] = 0.16$; the straight line is far above it because
$\mathbb{E}[Y \mid X]$ is not linear here. Every regressor you will ever train is trying to approximate the
black curve — and no amount of data or model capacity gets its out-of-sample error below $0.16$.

**Try it yourself:** increase the noise standard deviation from $0.4$ to $1.0$ and watch the gap between
the straight line and the truth shrink *relative* to the noise floor — when is a linear model "good enough"?
""")

# ---------------------------------------------------------------- 7. Simpson's paradox
nb.md(r"""
## 7. Simpson's paradox

$$P(R \mid T) = \sum_s P(R \mid T, s)\,P(s \mid T):$$

the aggregate rate mixes the stratum rates with each treatment's *own* severity mix. We reproduce the
guide's table, then build a continuous version: three age groups in which the within-group trend is
negative while the overall trend is positive.
""")

nb.code(r"""
import pandas as pd
rows = [("A", "mild", 20, 18), ("A", "severe", 80, 48), ("B", "mild", 80, 68), ("B", "severe", 20, 10)]
df = pd.DataFrame(rows, columns=["treatment", "severity", "patients", "recovered"])
df["rate"] = df.recovered / df.patients
strat = df.pivot(index="treatment", columns="severity", values="rate")[["mild", "severe"]]
print("recovery rate within each severity stratum:\n", strat.round(3))

agg = df.groupby("treatment")[["patients", "recovered"]].sum()
agg["rate"] = agg.recovered / agg.patients
print("\naggregate recovery rate:\n", agg)

mix = df.pivot(index="treatment", columns="severity", values="patients")[["mild", "severe"]]
mix = mix.div(mix.sum(axis=1), axis=0)
print("\nseverity mix P(s | T) — the culprit:\n", mix.round(2))

pooled = df.groupby("severity").patients.sum() / df.patients.sum()
standardised = (strat * pooled[strat.columns]).sum(axis=1)
print("\nstandardised rate with the pooled mix (50/50):\n", standardised.round(3))
print("\nA wins in every stratum, B wins overall: P(R|A) = 0.9*0.2 + 0.6*0.8 =", round(0.9*0.2 + 0.6*0.8, 2),
      "  P(R|B) = 0.85*0.8 + 0.5*0.2 =", round(0.85*0.8 + 0.5*0.2, 2))
""")

nb.code(r"""
# continuous Simpson: weekly exercise hours vs cholesterol, three age groups
groups = {"age 20-35": (2.0, 180.0), "age 36-50": (4.0, 200.0), "age 51-65": (6.0, 220.0)}
all_x, all_y = [], []
for name, (mx, my) in groups.items():
    xg = mx + rng.normal(0, 0.8, 60)
    yg = my - 6.0 * (xg - mx) + rng.normal(0, 5, 60)           # within each group: more exercise, lower cholesterol
    slope = np.polyfit(xg, yg, 1)[0]
    plt.scatter(xg, yg, s=12, alpha=0.7, label=f"{name}: slope {slope:+.1f}")
    xs_ = np.linspace(xg.min(), xg.max(), 2); plt.plot(xs_, np.polyval(np.polyfit(xg, yg, 1), xs_), lw=1.5)
    all_x.append(xg); all_y.append(yg)
all_x, all_y = np.concatenate(all_x), np.concatenate(all_y)
coef = np.polyfit(all_x, all_y, 1)
xs_ = np.linspace(all_x.min(), all_x.max(), 2)
plt.plot(xs_, np.polyval(coef, xs_), "k--", lw=2, label=f"all data pooled: slope {coef[0]:+.1f}")
plt.xlabel("exercise (hours per week)"); plt.ylabel("cholesterol (mg/dL)")
plt.title("Simpson's paradox: negative within every group, positive overall"); plt.legend(fontsize=8); plt.show()
print(f"pooled correlation = {np.corrcoef(all_x, all_y)[0, 1]:+.2f};  within-group correlations are all negative")
""")

nb.md(r"""
In the table, A beats B in both strata (90% vs 85%, 60% vs 50%) and loses overall (66% vs 78%) because 80%
of A's patients were severe against 20% of B's. Standardising both treatments to the same 50/50 mix restores
the stratified verdict. In the scatter plot the confounder is age: older people exercise more *and* have
higher cholesterol in this (synthetic) population, so pooling the groups produces a positive slope even
though exercise lowers cholesterol within every group. Which analysis is right is a causal question — here
age is a confounder, so the within-group slopes are the ones to believe.
""")

# ---------------------------------------------------------------- 8a. Kalman filter
nb.md(r"""
## 8a. A one-dimensional Kalman filter

Hidden position $x_t = x_{t-1} + v + w_t$, $w_t \sim \mathcal{N}(0, q)$; measurement $z_t = x_t + r_t$, $r_t \sim \mathcal{N}(0, r)$.

$$\text{predict: } \mu^- = \mu + v,\ P^- = P + q; \qquad \text{update: } K = \frac{P^-}{P^- + r},\ \mu = \mu^- + K(z_t - \mu^-),\ P = (1 - K)P^- .$$

We track an object for 60 steps with $v = 1$, $q = 1$, $r = 3$ (the guide's numbers), compare the filter's
error with the raw sensor, and watch the gain converge to its steady state.
""")

nb.code(r"""
T, v, q, r = 60, 1.0, 1.0, 3.0
x_true = np.empty(T); x_true[0] = 10 + rng.normal(0, 2)
for t in range(1, T):
    x_true[t] = x_true[t - 1] + v + rng.normal(0, np.sqrt(q))
z = x_true + rng.normal(0, np.sqrt(r), T)                      # noisy sensor readings

mu, P = 10.0, 4.0                                              # prior belief about x_0
mus, Ps, Ks = np.empty(T), np.empty(T), np.empty(T)
for t in range(T):
    if t > 0:
        mu, P = mu + v, P + q                                  # predict
    K = P / (P + r)                                            # Kalman gain
    mu, P = mu + K * (z[t] - mu), (1 - K) * P                  # update = Gaussian conditioning
    mus[t], Ps[t], Ks[t] = mu, P, K

rmse = lambda est: np.sqrt(np.mean((est - x_true)**2))
P_minus_inf = (q + np.sqrt(q**2 + 4*q*r)) / 2
print(f"RMSE of the raw sensor: {rmse(z):.3f}     RMSE of the Kalman estimate: {rmse(mus):.3f}")
print(f"filter's own sd at the end: {np.sqrt(Ps[-1]):.3f}   steady-state formula sqrt(P- r/(P- + r)) = {np.sqrt(P_minus_inf * r / (P_minus_inf + r)):.3f}   sensor sd = {np.sqrt(r):.3f}")
print(f"gain at the end: {Ks[-1]:.4f}   steady-state K = P-/(P- + r) = {P_minus_inf / (P_minus_inf + r):.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(x_true, "k", lw=2, label="true position")
ax1.plot(z, "o", ms=3, alpha=0.5, label="sensor")
ax1.plot(mus, "r", lw=2, label="Kalman estimate")
ax1.fill_between(np.arange(T), mus - 2*np.sqrt(Ps), mus + 2*np.sqrt(Ps), color="r", alpha=0.2, label="±2 sd")
ax1.set_xlabel("time step"); ax1.set_ylabel("position"); ax1.set_title("Tracking a noisy sensor"); ax1.legend(fontsize=8)
ax2.plot(Ks, lw=2, label="gain K_t"); ax2.axhline(P_minus_inf / (P_minus_inf + r), color="k", ls="--", label="steady-state gain")
ax2.annotate(f"K_0 = 4/{4 + r:g} = {Ks[0]:.3f}", xy=(0, Ks[0]), xytext=(6, Ks[0] - 0.012), fontsize=9, arrowprops=dict(arrowstyle="->", color="k"))
ax2.set_xlabel("time step"); ax2.set_ylabel("Kalman gain K"); ax2.set_title("The gain settles within a few steps"); ax2.legend()
plt.tight_layout(); plt.show()
""")

nb.code(r"""
# one update step *is* the conditioning formula of Section 5
mu_pred, P_pred, z_obs = 11.0, 5.0, 13.0
joint_mu = np.array([mu_pred, mu_pred])                        # (x_t, z_t) before seeing z_t
joint_S = np.array([[P_pred, P_pred], [P_pred, P_pred + r]])   # Cov(x, z) = Var(x) = P-
m, Vc = gaussian_condition(joint_mu, joint_S, [1], [z_obs])
K = P_pred / (P_pred + r)
print(f"Gaussian conditioning:  mean {m[0]:.4f}, variance {Vc[0, 0]:.4f}")
print(f"Kalman formulas:        mean {mu_pred + K*(z_obs - mu_pred):.4f}, variance {(1 - K)*P_pred:.4f}   (gain K = {K:.3f})")
print(f"precisions add: 1/{P_pred} + 1/{r} = {1/P_pred + 1/r:.4f} = 1/{1/(1/P_pred + 1/r):.4f}")
""")

nb.code(r"""
# the same update step as three densities: prediction, measurement, and the posterior that combines them
grid = np.linspace(4, 20, 400)
post = stats.norm.pdf(grid, m[0], np.sqrt(Vc[0, 0]))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(grid, stats.norm.pdf(grid, mu_pred, np.sqrt(P_pred)), "C0", lw=2, label=f"prediction N({mu_pred:.0f}, {P_pred:.0f})")
ax.plot(grid, stats.norm.pdf(grid, z_obs, np.sqrt(r)), "C1", lw=2, label=f"measurement N({z_obs:.0f}, {r:.0f})")
ax.plot(grid, post, "r", lw=2, label=f"posterior N({m[0]:.2f}, {Vc[0, 0]:.3f})"); ax.fill_between(grid, post, color="r", alpha=0.3)
for xv, col in [(mu_pred, "C0"), (z_obs, "C1"), (m[0], "r")]:
    ax.axvline(xv, color=col, ls=":", lw=1)
ax.annotate("", xy=(m[0], 0.018), xytext=(mu_pred, 0.018), arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
ax.annotate(f"K = {K:.3f} of the way\nto the measurement", xy=(11.6, 0.022), xytext=(4.3, 0.13), fontsize=9,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.8))
ax.set_xlabel("position"); ax.set_ylabel("density")
ax.set_title("One Kalman update: the posterior is the renormalised product of the two Gaussians", fontsize=10); ax.legend(fontsize=9)
plt.show()
""")

nb.md(r"""
**Notice that** the posterior is narrower than *both* curves it came from — precisions add,
$1/5 + 1/3 = 1/1.875$ — and that its peak sits five-eighths of the way from the prediction to the measurement,
which is exactly the gain $K = P^-/(P^- + r)$.
""")

nb.md(r"""
The filter cuts the sensor's error by about a quarter on this short run (RMSE $1.11$ against $1.46$); in steady
state the theory promises a third — sd $\sqrt{1.30} = 1.14$ against the sensor's $\sqrt3 = 1.73$ — which is to say
the error variance is more than halved ($1.30$ against $3$). The $\pm 2$ sd band contains the true track at 57 of
the 60 steps (95%, just what a calibrated filter should do); the misses are at $t = 10$, $11$ and $52$.
The gain starts high (the prior is vague, $P = 4$) and settles to $0.434$ in about five steps; from then on
every estimate is $43\%$ measurement and $57\%$ prediction. The one-step cell shows the update is nothing
but Section 5's formula applied to the joint Gaussian of (state, measurement).

**Try it yourself:** set `r = 30` (a terrible sensor) and then `r = 0.1` (an excellent one) — how do the
gain and the band react? Then make the true velocity $1.3$ while the filter still assumes $v = 1$ and watch
the model error show up as a lag.
""")

# ---------------------------------------------------------------- 8b. Gaussian processes
nb.md(r"""
## 8b. Gaussian-process regression

Function values at any set of inputs are jointly Gaussian with covariance $K_{ij} = k(x_i, x_j)$, here the
RBF kernel $k(x, x') = \sigma_f^2 \exp\big(-\frac{(x - x')^2}{2\ell^2}\big)$. Conditioning on six noisy observations gives

$$\mathbb{E}[f_* \mid y] = k_*^\top (K + \sigma_n^2 I)^{-1} y, \qquad \operatorname{Var}(f_* \mid y) = k(x_*, x_*) - k_*^\top (K + \sigma_n^2 I)^{-1} k_* .$$

Left: three functions drawn from the prior. Right: the posterior mean, the 95% band, three posterior samples,
and a check against scikit-learn.
""")

nb.code(r"""
def rbf(a, b, ell=1.0, sf=1.0):
    return sf**2 * np.exp(-(a[:, None] - b[None, :])**2 / (2 * ell**2))

f_true = lambda x: np.sin(x) + 0.3 * x
X_tr = np.array([-4.0, -2.5, -1.0, 0.5, 2.0, 3.5])
y_tr = f_true(X_tr) + rng.normal(0, 0.1, len(X_tr))
xs = np.linspace(-6, 6, 300)
ell, sf, sn2 = 1.0, 1.0, 0.01

K = rbf(X_tr, X_tr, ell, sf) + sn2 * np.eye(len(X_tr))        # K + sigma_n^2 I
Ks = rbf(X_tr, xs, ell, sf)                                    # k_* for every test point (columns)
Kss = rbf(xs, xs, ell, sf)
alpha = np.linalg.solve(K, y_tr)
post_mean = Ks.T @ alpha                                       # Sigma_12 Sigma_22^{-1} (y - 0)
post_cov = Kss - Ks.T @ np.linalg.solve(K, Ks)                 # Schur complement
post_sd = np.sqrt(np.clip(np.diag(post_cov), 0, None))

jitter = 1e-8 * np.eye(len(xs))
prior_samples = np.linalg.cholesky(Kss + jitter) @ rng.standard_normal((len(xs), 3))
post_samples = post_mean[:, None] + np.linalg.cholesky(post_cov + jitter) @ rng.standard_normal((len(xs), 3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
prior_lines = ax1.plot(xs, prior_samples, lw=1.5); prior_lines[0].set_label("3 prior samples")
ax1.fill_between(xs, -2*sf, 2*sf, alpha=0.15, label="prior ±2σ_f band")
ax1.set_title("three functions from the GP prior (RBF, ℓ = 1)"); ax1.set_xlabel("x"); ax1.set_ylabel("f(x)"); ax1.legend(fontsize=8, loc="lower center", ncol=2)
ax2.fill_between(xs, post_mean - 1.96*post_sd, post_mean + 1.96*post_sd, alpha=0.25, label="95% band")
post_lines = ax2.plot(xs, post_samples, lw=1, alpha=0.7); post_lines[0].set_label("3 posterior samples")
ax2.plot(xs, post_mean, "r", lw=2, label="posterior mean")
ax2.plot(xs, f_true(xs), "k--", lw=1, label="true function")
ax2.plot(X_tr, y_tr, "ko", label="observations")
ax2.set_title("posterior after conditioning on 6 points"); ax2.set_xlabel("x"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()

for xq in (0.5, 1.25, 5.5):
    i = np.argmin(np.abs(xs - xq))
    print(f"x = {xs[i]:5.2f}: posterior mean {post_mean[i]:+.3f}, sd {post_sd[i]:.3f}   (prior sd {sf:.1f})")

# the same posterior from scikit-learn (no hyperparameter optimisation, so the kernels match exactly)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
gpr = GaussianProcessRegressor(kernel=ConstantKernel(sf**2, "fixed") * RBF(ell, "fixed"), alpha=sn2, optimizer=None).fit(X_tr[:, None], y_tr)
sk_mean, sk_sd = gpr.predict(xs[:, None], return_std=True)
print(f"\nmax |mean - sklearn| = {np.abs(post_mean - sk_mean).max():.2e}   max |sd - sklearn| = {np.abs(post_sd - sk_sd).max():.2e}")
""")

nb.md(r"""
Near the observations the posterior standard deviation collapses to about $0.1$ (the noise level); between
them it grows, and about two length-scales past the last observation (by $x \approx 5.5$) it is back at the
prior value $1$ — the band widens exactly where there is no data, which is what makes Gaussian processes useful for deciding where to measure next. The posterior
samples are plausible functions consistent with the data, not just a mean curve. Our thirty lines agree with
`GaussianProcessRegressor` to machine precision: the library is the conditioning formula plus a Cholesky solve.
""")

# ---------------------------------------------------------------- 9. beyond Gaussians
nb.md(r"""
## 9. Dependence beyond Gaussians: copulas and mutual information

Two pairs with *identical* standard-normal marginals and nearly identical correlation, one tied together by
a Gaussian copula and one by a $t_3$-copula. Then a histogram estimate of mutual information
$I(X; Y) = \sum p(x, y)\log_2\frac{p(x, y)}{p(x)p(y)}$ for the zero-correlation pair $Y = X^2$.
""")

nb.code(r"""
n, rho, nu = 100_000, 0.5, 3
Lc = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
zg = rng.standard_normal((n, 2)) @ Lc.T                        # bivariate normal with correlation rho
u_gauss = stats.norm.cdf(zg)                                   # Gaussian copula: uniform marginals
zt = zg / np.sqrt(rng.chisquare(nu, n) / nu)[:, None]         # bivariate t_3 with the same rho
u_t = stats.t.cdf(zt, nu)                                      # t copula
x_gauss, x_t = stats.norm.ppf(u_gauss), stats.norm.ppf(u_t)    # both get standard-normal marginals

print(f"Pearson correlation:  Gaussian copula {np.corrcoef(x_gauss.T)[0, 1]:.3f}   t copula {np.corrcoef(x_t.T)[0, 1]:.3f}")
print(f"Spearman correlation: Gaussian copula {stats.spearmanr(x_gauss)[0]:.3f}   t copula {stats.spearmanr(x_t)[0]:.3f}")

from scipy import integrate
def joint_gauss(q):
    # P(both below their q-quantile), Gaussian copula: integrate over asset 1, using the conditional normal of asset 2
    a = stats.norm.ppf(q)
    return integrate.quad(lambda t: stats.norm.pdf(t) * stats.norm.cdf((a - rho * t) / np.sqrt(1 - rho**2)), -np.inf, a)[0]
def joint_t(q):
    # the same for the t copula: given asset 1 = t, asset 2 is t with nu + 1 dof, centre rho t and the scale below
    a = stats.t.ppf(q, nu)
    scale = lambda t: np.sqrt((nu + t**2) * (1 - rho**2) / (nu + 1))
    return integrate.quad(lambda t: stats.t.pdf(t, nu) * stats.t.cdf((a - rho * t) / scale(t), nu + 1), -np.inf, a)[0]

print("\nP(asset 2 below its q-quantile | asset 1 below its q-quantile):   [simulated / exact]")
for qtl in (0.05, 0.01, 0.001):
    sim_g = ((u_gauss[:, 0] < qtl) & (u_gauss[:, 1] < qtl)).mean() / qtl
    sim_t = ((u_t[:, 0] < qtl) & (u_t[:, 1] < qtl)).mean() / qtl
    ex_g, ex_t = joint_gauss(qtl) / qtl, joint_t(qtl) / qtl
    print(f"  q = {qtl:5.1%}:  Gaussian copula {sim_g:.3f} / {ex_g:.3f}     t copula {sim_t:.3f} / {ex_t:.3f}     (independence: {qtl:.3f})")
tail_lim = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
print(f"as q -> 0 the Gaussian conditional probability goes to 0 (no tail dependence); the t copula's tends to the positive limit {tail_lim:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
for ax, data, name in zip(axes, (x_gauss, x_t), ("Gaussian copula", "t copula (3 dof)")):
    q1 = stats.norm.ppf(0.01)
    ax.scatter(data[:8000, 0], data[:8000, 1], s=3, alpha=0.4)
    ax.axvline(q1, color="r", ls=":", label="1% quantile of each asset"); ax.axhline(q1, color="r", ls=":")
    n_joint = ((data[:8000, 0] < q1) & (data[:8000, 1] < q1)).sum()
    ax.text(0.97, 0.03, f"joint crashes among these 8000: {n_joint}", transform=ax.transAxes, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{name}: same marginals, r = {np.corrcoef(data.T)[0, 1]:.2f}"); ax.set_xlabel("asset 1"); ax.set_ylabel("asset 2")
plt.tight_layout(); plt.show()

def mutual_info_hist(x, y, bins=30):
    H, _, _ = np.histogram2d(x, y, bins=bins); p = H / H.sum()
    px, py = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())

x = rng.standard_normal(50_000)
for name, y in (("independent N(0,1)", rng.standard_normal(50_000)),
                ("Gaussian, rho = 0.707", 0.707*x + 0.707*rng.standard_normal(50_000)),
                ("Y = X^2", x**2)):
    print(f"{name:22s} correlation = {np.corrcoef(x, y)[0, 1]:+.3f}   mutual information ~ {mutual_info_hist(x, y):.3f} bits")
print("exact MI for a Gaussian with rho = 0.707: -0.5 log2(1 - 0.5) = 0.5 bits;  for Y = X^2 it is unbounded (Y is a function of X)")
""")

nb.code(r"""
# how the conditional crash probability changes with the threshold: exact curves, simulated points
qs = np.logspace(-3, np.log10(0.5), 15)
exact_g = [joint_gauss(q_) / q_ for q_ in qs]
exact_t = [joint_t(q_) / q_ for q_ in qs]
q_pts = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
sim_g = [((u_gauss[:, 0] < q_) & (u_gauss[:, 1] < q_)).mean() / q_ for q_ in q_pts]
sim_t = [((u_t[:, 0] < q_) & (u_t[:, 1] < q_)).mean() / q_ for q_ in q_pts]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.semilogx(qs, exact_g, "C0", lw=2, label="Gaussian copula, exact")
ax.semilogx(qs, exact_t, "C3", lw=2, label="t copula (3 dof), exact")
ax.plot(q_pts, sim_g, "o", mfc="none", color="C0", label="Gaussian copula, simulated (100 000 pairs)")
ax.plot(q_pts, sim_t, "s", mfc="none", color="C3", label="t copula, simulated (100 000 pairs)")
ax.axhline(tail_lim, color="k", ls="--", lw=1, label=f"t-copula tail dependence λ = {tail_lim:.2f}")
ax.plot(qs, qs, "k:", lw=1, label="independence: P = q")
ax.invert_xaxis(); ax.set_ylim(0, 0.75)
ax.set_xlabel("threshold q (both assets below their q-quantile)"); ax.set_ylabel("P(asset 2 crashes | asset 1 crashes)")
ax.set_title("Tail dependence: the Gaussian copula forgets, the t copula remembers", fontsize=11); ax.legend(fontsize=8)
plt.show()
""")

nb.md(r"""
**Notice that** the Gaussian curve keeps sliding towards zero as the threshold moves into the tail, while the $t$
curve flattens onto its tail-dependence coefficient $\lambda \approx 0.31$. The two copulas agree in the bulk
($q = 0.5$) and disagree more the further out you look, which is why a correlation estimated from ordinary days
says nothing about joint crashes. The red squares run a little above their curve on this seed: all the points
reuse the same 100,000 pairs, so one lucky batch of joint crashes lifts them together.
""")

nb.md(r"""
The two scatter plots have the same marginals and nearly the same correlation, yet joint extremes (both
below the red lines) are clearly more common under the $t$-copula. The conditional probability "asset 2
crashes given that asset 1 crashed" tells the story: under the Gaussian copula it *falls* as the threshold
moves further into the tail (towards zero — no tail dependence), while under the $t$-copula it stays around
a third no matter how extreme the event (the exact column is the one to trust at $q = 0.1\%$, where the
simulation rests on just 38 joint crashes under the $t$-copula and 6 under the Gaussian). At the 0.1% level the
joint crash is already about five to six times more likely. No univariate check and no correlation matrix would
detect the difference — this is the mechanism behind the 2008 mispricing described in the guide. Mutual
information, by contrast, sees what correlation misses: it is about $0$ for the independent pair, close to the
exact $0.5$ bits for the $\rho = 0.707$ Gaussian (binning loses a little information, so the estimate falls just
short of $0.5$; the finite-sample bias goes the other way and is what the independent pair's $0.01$ bits is made
of), and large for $Y = X^2$ despite its zero correlation.
""")

nb.md(r"""
## Summary

- A joint table contains everything: marginals are axis sums, conditionals are renormalised rows, independence is "the table is the outer product of its marginals" (rank 1).
- Covariance measures linear co-movement and depends on units; correlation is its unit-free version in $[-1, 1]$; zero correlation does not mean independence ($Y = X^2$).
- The covariance matrix is symmetric positive semidefinite because $a^\top \Sigma a = \operatorname{Var}(a^\top X) \ge 0$; linear maps act by $A \Sigma A^\top$; the sample version is $\mathbf{X}_c^\top \mathbf{X}_c/(n-1)$ and whitening by $\Sigma^{-1/2}$ decorrelates.
- The multivariate Gaussian has ellipsoidal level sets with axes from the eigendecomposition of $\Sigma$, is sampled as $\mu + Lz$ with $LL^\top = \Sigma$, and its squared Mahalanobis distance is $\chi^2_d$.
- Conditioning a Gaussian shifts the mean linearly in the observation and shrinks the covariance by the Schur complement — independently of the observed value; regression slopes multiply to $\rho^2$.
- $\mathbb{E}[Y \mid X]$ is the best MSE predictor and the target of every regressor; $\operatorname{Var} Y$ = within + between.
- Simpson's paradox: differing stratum mixes can reverse an aggregate comparison; standardise or randomise, but decide causally which table to trust.
- The Kalman filter and Gaussian-process regression are the conditioning formula applied over time and over space; both matched reference implementations exactly.
- Copulas separate marginals from dependence; tail dependence and mutual information capture what correlation cannot.

**Next:** open `exercises.ipynb` and work through the self-checked problems; the solutions notebook has reference implementations.
""")

nb.save(MOD / "lesson.ipynb")
print(f"lesson.ipynb: {len(nb.cells)} cells")

# =============================================================================
# EXERCISES — hidden answers are computed by executing the reference solutions
# =============================================================================
SETUP = r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)
"""


def run(solution_src: str, extra_src: str = "") -> dict:
    """Execute the setup, a solution and optional extra code in a fresh namespace; return it."""
    ns: dict = {}
    exec(textwrap.dedent(SETUP), ns)
    exec(textwrap.dedent(solution_src), ns)
    if extra_src:
        exec(textwrap.dedent(extra_src), ns)
    return ns


ex = ExerciseSet(
    "Module 09 — Exercises",
    intro_md=r"""
These exercises follow the order of the concept guide: joint tables, covariance and correlation, covariance
matrices, the multivariate Gaussian and its conditioning formulas, Simpson's paradox, the Mahalanobis distance,
Gaussian processes, and finally a Kalman-filter mini-project. Several checks compare your answer against a
*hidden* value, so compute carefully rather than guessing. Use only NumPy (and SciPy where a prompt says so).
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- 1. joint table (warm-up)
JOINT_DATA = r"""
# joint PMF of (weather, bike-share demand) for a city: rows = weather, columns = demand level
weather = ["sunny", "cloudy", "rainy"]
demand = ["low", "medium", "high"]
J = np.array([[0.05, 0.15, 0.25],
              [0.10, 0.15, 0.10],
              [0.12, 0.06, 0.02]])
"""
SOL_1 = r"""
def marginals(J):
    return J.sum(axis=1), J.sum(axis=0)

def conditional_cols_given_row(J, i):
    return J[i] / J[i].sum()

def conditional_rows_given_col(J, j):
    return J[:, j] / J[:, j].sum()

def is_independent(J, tol=1e-9):
    pr, pc = marginals(J)
    return bool(np.allclose(J, np.outer(pr, pc), atol=tol))
"""
ns1 = run(JOINT_DATA + SOL_1, r"""
p_w, p_d = marginals(J)
c_rainy = conditional_cols_given_row(J, 2)
c_high = conditional_rows_given_col(J, 2)
p_high_given_sunny = conditional_cols_given_row(J, 0)[2]
p_sunny_given_high = c_high[0]
indep_flag = is_independent(J)
indep_outer = is_independent(np.outer(p_w, p_d))
""")
ex.add(
    title="Marginals, conditionals and an independence test from a joint table", difficulty=1,
    prompt_md=r"""
The table `J` below is the joint PMF of the weather $W$ (rows) and the bike-share demand $D$ (columns) on a
random day. Implement, using only NumPy array operations (no loops needed):

* `marginals(J)` → a tuple `(p_rows, p_cols)` of the two marginal PMFs (arrays of length 3);
* `conditional_cols_given_row(J, i)` → the conditional PMF $p(D \mid W = w_i)$ as an array of length 3;
* `conditional_rows_given_col(J, j)` → the conditional PMF $p(W \mid D = d_j)$ (Bayes' theorem in table form);
* `is_independent(J, tol=1e-9)` → `True` if `J` equals the outer product of its marginals to within `tol`, else `False`.

Then think: is a high-demand day more likely to be sunny than a random day is?
""" + "\n```python\n" + textwrap.dedent(JOINT_DATA).strip() + "\n```\n",
    starter=JOINT_DATA + r"""

def marginals(J):
    # YOUR CODE HERE
    return None

def conditional_cols_given_row(J, i):
    # YOUR CODE HERE
    return None

def conditional_rows_given_col(J, j):
    # YOUR CODE HERE
    return None

def is_independent(J, tol=1e-9):
    # YOUR CODE HERE
    return None
""",
    solution=JOINT_DATA + "\n" + SOL_1,
    check=rf"""
try:
    p_w, p_d = marginals(J)
    p_w, p_d = np.asarray(p_w, float), np.asarray(p_d, float)
except Exception:
    p_w, p_d = None, None
check_close('marginal of the weather', p_w, np.array([0.45, 0.35, 0.20]))
check_hash('marginal of the demand', p_d, '{sha_of(ns1["p_d"])}')
try:
    c = np.asarray(conditional_cols_given_row(J, 2), float)
    ok_sum = abs(c.sum() - 1) < 1e-9
except Exception:
    c, ok_sum = None, False
check('P(demand | rainy) sums to 1', ok_sum)
check_hash('P(demand | rainy)', c, '{sha_of(ns1["c_rainy"])}')
try:
    got = float(conditional_cols_given_row(J, 0)[2])
except Exception:
    got = None
check_hash('P(high demand | sunny)', got, '{sha_of(ns1["p_high_given_sunny"])}')
try:
    got = float(conditional_rows_given_col(J, 2)[0])
except Exception:
    got = None
check_hash('P(sunny | high demand)  (compare with P(sunny) = 0.45)', got, '{sha_of(ns1["p_sunny_given_high"])}')
try:
    got_dep = is_independent(J)
    got_ind = is_independent(np.outer(J.sum(axis=1), J.sum(axis=0)))
    ok_dep = got_dep is not None and not got_dep
    ok_ind = got_ind is not None and bool(got_ind)
except Exception:
    ok_dep, ok_ind = False, False
check('the table is detected as dependent', ok_dep)
check('the outer-product table is detected as independent', ok_ind)
""",
    explanation_md=r"""
Marginalising is summing along an axis: `J.sum(axis=1)` adds across each row (summing out the demand) and
gives the weather marginal $(0.45, 0.35, 0.20)$. Conditioning is renormalising a slice: the rainy row
$(0.12, 0.06, 0.02)$ divided by $0.20$ gives $(0.6, 0.3, 0.1)$. Conditioning on a *column* is Bayes' theorem in
table form — $P(\text{sunny} \mid \text{high}) = 0.25/0.37 \approx 0.68$, far above the prior $0.45$, so high
demand is strong evidence of sunshine. Independence means every cell factorises, which `np.allclose` against
the outer product checks in one line; the common mistake is to test a single cell, which can match by
coincidence even in a dependent table.
""",
)

# ---------------------------------------------------------------- 2. covariance & correlation by hand (warm-up)
SOL_2 = r"""
def cov_xy(x, y, ddof=1):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - ddof))

def corr_xy(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx, dy = x - x.mean(), y - y.mean()
    return float(np.sum(dx * dy) / np.sqrt(np.sum(dx**2) * np.sum(dy**2)))

def rank_corr(x, y):
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    return corr_xy(rx, ry)
"""
ns2 = run(SOL_2, r"""
temp = np.array([12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0])
sales = np.array([30.0, 38.0, 46.0, 60.0, 78.0, 105.0, 150.0])
cov_ts = cov_xy(temp, sales)
cov_ts0 = cov_xy(temp, sales, ddof=0)
r_ts = corr_xy(temp, sales)
rs_ts = rank_corr(temp, sales)
xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]); ys = xs**2
r_sq = corr_xy(xs, ys)
r_lin = corr_xy(temp, 3*temp + 7)
r_neg = corr_xy(temp, -0.5*temp + 1)
""")
ex.add(
    title="Covariance and correlation from scratch", difficulty=1,
    prompt_md=r"""
Implement, without `np.cov`, `np.corrcoef` or `scipy.stats.pearsonr`:

* `cov_xy(x, y, ddof=1)` — the sample covariance $\frac{1}{n - \text{ddof}}\sum_i (x_i - \bar x)(y_i - \bar y)$;
* `corr_xy(x, y)` — Pearson's correlation coefficient (the `ddof` cancels, so none is needed);
* `rank_corr(x, y)` — Spearman's rank correlation, i.e. `corr_xy` applied to the ranks (`scipy.stats.rankdata` is allowed).

Test data: daily temperatures `temp = [12, 15, 18, 21, 24, 27, 30]` (°C) and ice-cream sales
`sales = [30, 38, 46, 60, 78, 105, 150]`. The checks also test the exact-line cases $y = 3x + 7$ and
$y = -0.5x + 1$ and the symmetric parabola $x = (-2, -1, 0, 1, 2)$, $y = x^2$.
""",
    starter=r"""
def cov_xy(x, y, ddof=1):
    # YOUR CODE HERE
    return None

def corr_xy(x, y):
    # YOUR CODE HERE
    return None

def rank_corr(x, y):
    # YOUR CODE HERE  (hint: stats.rankdata)
    return None

temp = np.array([12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0])
sales = np.array([30.0, 38.0, 46.0, 60.0, 78.0, 105.0, 150.0])
""",
    solution=SOL_2 + r"""
temp = np.array([12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0])
sales = np.array([30.0, 38.0, 46.0, 60.0, 78.0, 105.0, 150.0])
""",
    check=rf"""
try:
    got = cov_xy(temp, sales)
except Exception:
    got = None
check_hash('sample covariance of temperature and sales (ddof = 1)', got, '{sha_of(ns2["cov_ts"], decimals=2)}', decimals=2)
try:
    got = cov_xy(temp, sales, ddof=0)
except Exception:
    got = None
check_hash('population covariance (ddof = 0)', got, '{sha_of(ns2["cov_ts0"], decimals=2)}', decimals=2)
try:
    got = corr_xy(temp, sales)
except Exception:
    got = None
check_hash('Pearson correlation of temperature and sales', got, '{sha_of(ns2["r_ts"], decimals=3)}', decimals=3)
try:
    got = rank_corr(temp, sales)
except Exception:
    got = None
check_close('Spearman correlation of temperature and sales is exactly 1 (monotone)', got, 1.0)
try:
    got_pos = corr_xy(temp, 3*temp + 7); got_neg = corr_xy(temp, -0.5*temp + 1)
except Exception:
    got_pos, got_neg = None, None
check_close('y = 3x + 7 has correlation +1', got_pos, 1.0)
check_close('y = -0.5x + 1 has correlation -1', got_neg, -1.0)
try:
    xs_ = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]); got = corr_xy(xs_, xs_**2)
except Exception:
    got = None
check_close('y = x^2 on a symmetric grid has correlation 0', got, 0.0, tol=1e-9)
""",
    explanation_md=r"""
Centre both variables, multiply the deviations pairwise and average — with $n - 1$ for the unbiased sample
version, $n$ for the population version (NumPy's `np.cov` defaults to the former, `np.var` to the latter).
Correlation divides by the product of the two root-sum-of-squares, so any common normalisation cancels, and
the result is scale-free: $y = 3x + 7$ gives exactly $+1$ and $y = -0.5x + 1$ exactly $-1$. The ice-cream
data are convex (sales accelerate with temperature), so Pearson's $r \approx 0.95$ is below the Spearman value
of exactly $1$ — ranks only see the monotone ordering. On the symmetric parabola the deviations $x_i - \bar x$
are odd and $y_i - \bar y$ even, so every product cancels with its mirror image and the correlation is zero
despite perfect dependence.
""",
)

# ---------------------------------------------------------------- 3. conceptual: uncorrelated vs independent (warm-up)
ex.add_conceptual(
    title="Uncorrelated or independent?", difficulty=1,
    prompt_md=r"""
A colleague inspects the correlation matrix of a feature set, finds that feature 7 has correlation $0.01$ with
the target, and drops it: "it carries no information about $y$."

1. Give a concrete example of a feature $X$ and a target $Y$ with $\operatorname{Corr}(X, Y) = 0$ where $Y$ is nevertheless
   a deterministic function of $X$. Prove the correlation is zero.
2. For which family of joint distributions *does* zero correlation imply independence, and why does the argument
   not extend to other distributions?
3. Name two quantities that would detect the dependence your example hides from correlation, and say briefly what
   each one measures.
""",
    answer_md=r"""
1. Take $X \sim \mathcal{N}(0, 1)$ (or any distribution symmetric about zero) and $Y = X^2$. Then
   $\operatorname{Cov}(X, Y) = \mathbb{E}[X^3] - \mathbb{E}[X]\,\mathbb{E}[X^2] = 0 - 0 \cdot 1 = 0$, because odd
   moments of a symmetric distribution vanish. Yet $Y$ is completely determined by $X$: knowing $X$ tells you $Y$
   exactly, so $P(Y \le 1 \mid X = 2) = 0 \ne P(Y \le 1) \approx 0.68$. A model that drops $X$ cannot predict $Y$ at
   all; a model with $X^2$ as a feature predicts it perfectly.
2. For *jointly Gaussian* vectors. If $\Sigma$ is diagonal the density $\exp(-\tfrac12 (x - \mu)^\top \Sigma^{-1}(x - \mu))$
   splits into a product of one-dimensional factors, which is the definition of independence. The argument uses the
   specific form of the Gaussian density — the *exponential* of a quadratic form, so a diagonal $\Sigma$ turns the sum
   in the exponent into a product of factors. Depending on $x$ only through a quadratic form is not enough: a
   multivariate $t$ with diagonal scale does too, and its coordinates are uncorrelated yet dependent. The Gaussian
   is the standard example; a pair of binary variables is another, because a $2 \times 2$ table has only one degree
   of freedom of dependence. In general, though, a diagonal covariance says nothing about the joint beyond its
   second moments.
3. *Mutual information* $I(X; Y) = D_{\mathrm{KL}}(p_{XY} \| p_X p_Y)$ is zero if and only if $X$ and $Y$ are
   independent and would be large (formally infinite for a deterministic continuous relationship) in the example.
   *Spearman's rank correlation* catches monotone non-linear dependence, though not this symmetric one; a *scatter
   plot*, a *binned conditional mean* $\mathbb{E}[Y \mid X \in \text{bin}]$, or a *distance correlation* also expose the
   U-shape. The practical lesson: inspect $\mathbb{E}[Y \mid X]$, not just the linear coefficient, before discarding a feature.
""",
)

# ---------------------------------------------------------------- 4. covariance matrix, PSD, a'Sigma a (core)
SOL_4 = r"""
def sample_cov(X):
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (X.shape[0] - 1)

def is_psd(S, tol=1e-10):
    S = np.asarray(S, float)
    return bool(np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() >= -tol)

def portfolio_vol(S, w):
    w = np.asarray(w, float)
    return float(np.sqrt(w @ np.asarray(S, float) @ w))

def cov_to_corr(S):
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)
"""
ns4 = run(SOL_4, r"""
sd = np.array([0.25, 0.15, 0.30, 0.10])
R = np.array([[1.0, 0.4, 0.6, 0.1],
              [0.4, 1.0, 0.2, 0.0],
              [0.6, 0.2, 1.0, 0.3],
              [0.1, 0.0, 0.3, 1.0]])
Sigma = np.outer(sd, sd) * R
X = np.random.default_rng(4).multivariate_normal(np.zeros(4), Sigma, size=500)
S = sample_cov(X)
w_eq = np.full(4, 0.25)
vol_true_eq = portfolio_vol(Sigma, w_eq)
vol_sample_eq = portfolio_vol(S, w_eq)
w_tilt = np.array([0.1, 0.2, 0.3, 0.4])
vol_true_tilt = portfolio_vol(Sigma, w_tilt)
bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
psd_sigma = is_psd(Sigma); psd_bad = is_psd(bad)
corr_sample_13 = cov_to_corr(S)[0, 2]
""")
ex.add(
    title=r"Covariance matrices: build, check positive semidefiniteness, compute $a^\top\Sigma a$", difficulty=2,
    prompt_md=r"""
Implement, with NumPy only (no `np.cov`, no `np.corrcoef`):

* `sample_cov(X)` → the $d \times d$ sample covariance $\frac{1}{n-1}\mathbf{X}_c^\top\mathbf{X}_c$ of an $n \times d$ data matrix `X` (rows = observations);
* `is_psd(S, tol=1e-10)` → `True` if `S` is symmetric and all eigenvalues are $\ge -\text{tol}$ (`np.linalg.eigvalsh` is allowed);
* `portfolio_vol(S, w)` → $\sqrt{w^\top S w}$ as a float;
* `cov_to_corr(S)` → the correlation matrix $D^{-1/2} S D^{-1/2}$.

The data: four assets with volatilities `sd = [0.25, 0.15, 0.30, 0.10]` and the correlation matrix `R` below,
so $\Sigma_{ij} = \rho_{ij}\sigma_i\sigma_j$; `X` holds 500 simulated annual returns (the cell is seeded, do not change it).
```python
sd = np.array([0.25, 0.15, 0.30, 0.10])
R = np.array([[1.0, 0.4, 0.6, 0.1],
              [0.4, 1.0, 0.2, 0.0],
              [0.6, 0.2, 1.0, 0.3],
              [0.1, 0.0, 0.3, 1.0]])
Sigma = np.outer(sd, sd) * R
X = np.random.default_rng(4).multivariate_normal(np.zeros(4), Sigma, size=500)
```
""",
    starter=r"""
sd = np.array([0.25, 0.15, 0.30, 0.10])
R = np.array([[1.0, 0.4, 0.6, 0.1],
              [0.4, 1.0, 0.2, 0.0],
              [0.6, 0.2, 1.0, 0.3],
              [0.1, 0.0, 0.3, 1.0]])
Sigma = np.outer(sd, sd) * R
X = np.random.default_rng(4).multivariate_normal(np.zeros(4), Sigma, size=500)   # own seed: do not change

def sample_cov(X):
    # YOUR CODE HERE
    return None

def is_psd(S, tol=1e-10):
    # YOUR CODE HERE
    return None

def portfolio_vol(S, w):
    # YOUR CODE HERE
    return None

def cov_to_corr(S):
    # YOUR CODE HERE
    return None
""",
    solution=r"""
sd = np.array([0.25, 0.15, 0.30, 0.10])
R = np.array([[1.0, 0.4, 0.6, 0.1],
              [0.4, 1.0, 0.2, 0.0],
              [0.6, 0.2, 1.0, 0.3],
              [0.1, 0.0, 0.3, 1.0]])
Sigma = np.outer(sd, sd) * R
X = np.random.default_rng(4).multivariate_normal(np.zeros(4), Sigma, size=500)   # own seed: do not change
""" + SOL_4,
    check=rf"""
try:
    S = np.asarray(sample_cov(X), float)
    ok_shape = S.shape == (4, 4)
except Exception:
    S, ok_shape = None, False
check('sample covariance has shape (4, 4)', ok_shape)
check_close('sample covariance matches np.cov(X, rowvar=False)', S, np.cov(X, rowvar=False), tol=1e-9)
try:
    got = portfolio_vol(Sigma, np.full(4, 0.25))
except Exception:
    got = None
check_hash('equal-weight portfolio volatility from the true Sigma', got, '{sha_of(ns4["vol_true_eq"], decimals=4)}', decimals=4)
try:
    got = portfolio_vol(Sigma, np.array([0.1, 0.2, 0.3, 0.4]))
except Exception:
    got = None
check_hash('tilted portfolio (0.1, 0.2, 0.3, 0.4) volatility from the true Sigma', got, '{sha_of(ns4["vol_true_tilt"], decimals=4)}', decimals=4)
try:
    got = portfolio_vol(S, np.full(4, 0.25))
    emp = float(np.std(X @ np.full(4, 0.25), ddof=1))
except Exception:
    got, emp = None, None
check_close('sqrt(w^T S w) equals the empirical standard deviation of the portfolio returns X w', got, emp, tol=1e-9)
try:
    got_good = is_psd(Sigma); got_bad = is_psd(np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]))
    ok_good = got_good is not None and bool(got_good)
    ok_bad = got_bad is not None and not got_bad
except Exception:
    ok_good, ok_bad = False, False
check('Sigma is recognised as positive semidefinite', ok_good)
check('the impossible correlation matrix (0.9, 0.9, -0.9) is rejected', ok_bad)
try:
    C = np.asarray(cov_to_corr(Sigma), float)
    ok_diag = np.allclose(np.diag(C), 1.0)
except Exception:
    C, ok_diag = None, False
check('correlation matrix has ones on the diagonal', ok_diag)
check_close('cov_to_corr(Sigma) recovers R', C, R, tol=1e-9)
""",
    explanation_md=r"""
Centre the columns and form $\mathbf{X}_c^\top\mathbf{X}_c/(n - 1)$: a single matrix product gives all
$d(d+1)/2$ variances and covariances at once. $w^\top S w$ is the variance of the one-dimensional series
$\mathbf{X}w$ — the check confirms this identity exactly, which is the content of $\operatorname{Var}(a^\top X) = a^\top\Sigma a$.
The equal-weight volatility of about $14.7\%$ sits well below the average volatility of $20\%$ because the
correlations are moderate. The matrix with correlations $0.9, 0.9, -0.9$ is symmetric with a unit diagonal but
has eigenvalue $-0.8$: assets 2 and 3 cannot both move with asset 1 and against each other, and `is_psd` must
reject it. A common slip is to use `np.linalg.eigvals`, which returns complex numbers with tiny imaginary parts;
`eigvalsh` exploits symmetry and returns sorted real eigenvalues.
""",
)

# ---------------------------------------------------------------- 5. MVN log-density (core)
SOL_5 = r"""
def mvn_logpdf(x, mu, Sigma):
    x = np.atleast_2d(np.asarray(x, float)); mu = np.asarray(mu, float); Sigma = np.asarray(Sigma, float)
    d = len(mu)
    L = np.linalg.cholesky(Sigma)
    Y = np.linalg.solve(L, (x - mu).T)                 # columns: L^{-1}(x_i - mu)
    maha = np.sum(Y**2, axis=0)                        # squared Mahalanobis distances
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    out = -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
    return float(out[0]) if out.shape[0] == 1 else out

def mahalanobis_sq(x, mu, Sigma):
    x = np.atleast_2d(np.asarray(x, float)); mu = np.asarray(mu, float)
    L = np.linalg.cholesky(np.asarray(Sigma, float))
    Y = np.linalg.solve(L, (x - mu).T)
    out = np.sum(Y**2, axis=0)
    return float(out[0]) if out.shape[0] == 1 else out
"""
ns5 = run(SOL_5, r"""
mu3 = np.array([1.0, -1.0, 0.5])
S3 = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 1.5], [1.0, 1.5, 2.0]])
lp_a = mvn_logpdf(np.array([2.0, 0.0, 1.0]), mu3, S3)
lp_b = mvn_logpdf(mu3, mu3, S3)
m_a = mahalanobis_sq(np.array([2.0, 0.0, 1.0]), mu3, S3)
S2 = np.array([[2.0, 1.0], [1.0, 1.0]])
lp_11 = mvn_logpdf(np.array([1.0, 1.0]), np.zeros(2), S2)
lp_1m1 = mvn_logpdf(np.array([1.0, -1.0]), np.zeros(2), S2)
""")
ex.add(
    title="The multivariate Gaussian log-density via Cholesky", difficulty=2,
    prompt_md=r"""
Implement the log-density of $\mathcal{N}(\mu, \Sigma)$,

$$\log f(x) = -\tfrac12\Big[d\log(2\pi) + \log|\Sigma| + (x - \mu)^\top\Sigma^{-1}(x - \mu)\Big],$$

**without** calling `np.linalg.inv` or `np.linalg.det`. Use the Cholesky factor $\Sigma = LL^\top$:
$\log|\Sigma| = 2\sum_i \log L_{ii}$, and with $y = L^{-1}(x - \mu)$ (a triangular solve, `np.linalg.solve(L, ...)`)
the quadratic form is $y^\top y$.

* `mvn_logpdf(x, mu, Sigma)` → a float for a single point `x` of shape `(d,)`, or an array of length $n$ for an $n \times d$ array of points;
* `mahalanobis_sq(x, mu, Sigma)` → the squared Mahalanobis distance(s) $(x - \mu)^\top\Sigma^{-1}(x - \mu)$, same conventions.

The checks compare against `scipy.stats.multivariate_normal` and against the hand-computed values of the guide
for $\Sigma = \begin{pmatrix} 2 & 1 \\ 1 & 1\end{pmatrix}$.
""",
    starter=r"""
def mvn_logpdf(x, mu, Sigma):
    # YOUR CODE HERE  (Cholesky: L = np.linalg.cholesky(Sigma); no inv, no det)
    return None

def mahalanobis_sq(x, mu, Sigma):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_5,
    check=rf"""
S2 = np.array([[2.0, 1.0], [1.0, 1.0]])
try:
    got = mvn_logpdf(np.array([1.0, 1.0]), np.zeros(2), S2)
except Exception:
    got = None
check_close('log f(1, 1) for Sigma = [[2,1],[1,1]] equals log(e^-0.5 / 2pi)', got, float(np.log(np.exp(-0.5) / (2 * np.pi))))
try:
    got = mvn_logpdf(np.array([1.0, -1.0]), np.zeros(2), S2)
except Exception:
    got = None
check_close('log f(1, -1) matches scipy', got, float(stats.multivariate_normal(np.zeros(2), S2).logpdf([1.0, -1.0])))
try:
    got = mahalanobis_sq(np.array([1.0, -1.0]), np.zeros(2), S2)
except Exception:
    got = None
check_close('squared Mahalanobis distance of (1, -1) is 5', got, 5.0)
mu3 = np.array([1.0, -1.0, 0.5]); S3 = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 1.5], [1.0, 1.5, 2.0]])
try:
    got = mvn_logpdf(np.array([2.0, 0.0, 1.0]), mu3, S3)
except Exception:
    got = None
check_hash('3-D log-density at (2, 0, 1)', got, '{sha_of(ns5["lp_a"], decimals=4)}', decimals=4)
check_close('3-D log-density at (2, 0, 1) matches scipy', got, float(stats.multivariate_normal(mu3, S3).logpdf([2.0, 0.0, 1.0])))
try:
    pts = rng.multivariate_normal(mu3, S3, size=50)
    got = np.asarray(mvn_logpdf(pts, mu3, S3), float)
    ok = got.shape == (50,)
except Exception:
    got, ok = None, False
check('vectorised call returns one value per row', ok)
check_close('vectorised log-densities match scipy', got, (stats.multivariate_normal(mu3, S3).logpdf(pts) if ok else None), tol=1e-8)
try:
    big = 3000 * np.eye(5)
    got = mvn_logpdf(np.zeros(5), np.zeros(5), big)
except Exception:
    got = None
check_close('stable for a large-variance 5-D case', got, float(stats.multivariate_normal(np.zeros(5), 3000 * np.eye(5)).logpdf(np.zeros(5))))
""",
    explanation_md=r"""
Factor $\Sigma = LL^\top$ once; then $\Sigma^{-1} = L^{-\top}L^{-1}$, so $(x - \mu)^\top\Sigma^{-1}(x - \mu) = \|L^{-1}(x - \mu)\|^2$,
and $|\Sigma| = |L|^2 = \prod_i L_{ii}^2$ gives $\log|\Sigma| = 2\sum_i\log L_{ii}$. This costs one $O(d^3)$ factorisation plus
an $O(d^2)$ triangular solve per point, it never forms the inverse (the factor $L$ has only the square root of $\Sigma$'s condition number, so the triangular solve is both cheaper and more accurate), and it never
underflows the determinant — for $d = 1{,}000$ features with variances of $0.01$, $|\Sigma| = 10^{-2000}$ underflows to
$0$ in `np.linalg.det`, while the sum of logs is harmless. The values $\log f(1, 1) = \log(e^{-1/2}/2\pi)$
and $\Delta^2(1, -1) = 5$ are the guide's hand computation. `scipy.stats.multivariate_normal` does the same thing
internally (via an eigendecomposition, so it also handles singular $\Sigma$).
""",
)

# ---------------------------------------------------------------- 6. Gaussian conditioning (core)
SOL_6 = r"""
def gaussian_condition(mu, Sigma, obs_idx, obs_val):
    mu = np.asarray(mu, float); Sigma = np.asarray(Sigma, float)
    obs_idx = list(obs_idx); free = [i for i in range(len(mu)) if i not in obs_idx]
    S11 = Sigma[np.ix_(free, free)]
    S12 = Sigma[np.ix_(free, obs_idx)]
    S22 = Sigma[np.ix_(obs_idx, obs_idx)]
    gain = S12 @ np.linalg.inv(S22)
    mu_c = mu[free] + gain @ (np.asarray(obs_val, float) - mu[obs_idx])
    Sigma_c = S11 - gain @ S12.T
    return mu_c, Sigma_c
"""
ns6 = run(SOL_6, r"""
mu_hw = np.array([170.0, 70.0]); S_hw = np.array([[100.0, 72.0], [72.0, 144.0]])
m1, v1 = gaussian_condition(mu_hw, S_hw, [0], [185.0])
mu3 = np.array([1.0, -1.0, 0.5])
S3 = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 1.5], [1.0, 1.5, 2.0]])
m2, v2 = gaussian_condition(mu3, S3, [2], [2.0])
m3, v3 = gaussian_condition(mu3, S3, [0, 2], [3.0, 0.0])
""")
ex.add(
    title="Gaussian conditioning in any dimension", difficulty=2,
    prompt_md=r"""
Implement `gaussian_condition(mu, Sigma, obs_idx, obs_val)` returning the tuple `(mu_cond, Sigma_cond)` of the
conditional distribution of the *unobserved* coordinates (in their original order) given that the coordinates
in the list `obs_idx` take the values `obs_val`:

$$\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2), \qquad \Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}.$$

`np.ix_` is handy for extracting blocks. The function must work for any dimension and any number of observed
coordinates. Test cases: the guide's height/weight example ($\mu = (170, 70)$, $\Sigma = \begin{pmatrix}100 & 72\\72 & 144\end{pmatrix}$,
observe height 185) and a three-dimensional Gaussian with one and with two observed coordinates.
""",
    starter=r"""
def gaussian_condition(mu, Sigma, obs_idx, obs_val):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_6,
    check=rf"""
mu_hw = np.array([170.0, 70.0]); S_hw = np.array([[100.0, 72.0], [72.0, 144.0]])
try:
    m, v = gaussian_condition(mu_hw, S_hw, [0], [185.0])
    m, v = np.asarray(m, float).ravel(), np.asarray(v, float)
except Exception:
    m, v = None, None
check_close('weight | height = 185: mean 80.8', m, np.array([80.8]))
check_close('weight | height = 185: variance 92.16', (v.ravel() if v is not None else None), np.array([92.16]))
try:
    m, v = gaussian_condition(mu_hw, S_hw, [1], [90.0])
    m = np.asarray(m, float).ravel()
except Exception:
    m, v = None, None
check_close('height | weight = 90: mean 180', m, np.array([180.0]))
mu3 = np.array([1.0, -1.0, 0.5]); S3 = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 1.5], [1.0, 1.5, 2.0]])
try:
    m, v = gaussian_condition(mu3, S3, [2], [2.0])
    m, v = np.asarray(m, float), np.asarray(v, float)
    ok = m.shape == (2,) and v.shape == (2, 2)
except Exception:
    m, v, ok = None, None, False
check('3-D, one observed coordinate: shapes (2,) and (2, 2)', ok)
check_hash('3-D conditional mean given x3 = 2', m, '{sha_of(ns6["m2"])}')
check_hash('3-D conditional covariance given x3 = 2', v, '{sha_of(ns6["v2"])}')
try:
    m, v = gaussian_condition(mu3, S3, [0, 2], [3.0, 0.0])
    m, v = np.asarray(m, float).ravel(), np.asarray(v, float).ravel()
except Exception:
    m, v = None, None
check_hash('3-D conditional mean of x2 given x1 = 3, x3 = 0', m, '{sha_of(ns6["m3"].ravel())}')
check_hash('3-D conditional variance of x2 given x1 = 3, x3 = 0', v, '{sha_of(ns6["v3"].ravel())}')
try:
    _, v_a = gaussian_condition(mu3, S3, [2], [2.0]); _, v_b = gaussian_condition(mu3, S3, [2], [-7.0])
    same = bool(np.allclose(v_a, v_b))
    _, v_full = gaussian_condition(mu3, S3, [2], [2.0])
    shrinks = bool(np.all(np.linalg.eigvalsh(S3[:2, :2] - np.asarray(v_full)) >= -1e-12))
except Exception:
    same, shrinks = None, None
check('the conditional covariance does not depend on the observed value', same is True)
check('conditioning never increases uncertainty (Sigma_11 - Sigma_1|2 is PSD)', shrinks is True)
""",
    explanation_md=r"""
Slice the four blocks with `np.ix_`, compute the gain $\Sigma_{12}\Sigma_{22}^{-1}$ once, and apply it to the
innovation $x_2 - \mu_2$; the Schur complement is $\Sigma_{11}$ minus the gain times $\Sigma_{21}$. Two structural
facts are checked: the conditional covariance is the same whether $x_3 = 2$ or $x_3 = -7$ (only the mean moves), and
$\Sigma_{11} - \Sigma_{1|2} = \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ is PSD, so conditioning can only shrink the
uncertainty ellipsoid. For the height/weight case the gain is $72/100 = 0.72$ kg per cm and the residual variance
$144 - 72^2/100 = 92.16$. The usual bugs are forgetting the transpose in $\Sigma_{21}$ (harmless for symmetric
slices but wrong for general blocks) and conditioning on indices without keeping the unobserved coordinates in their
original order.
""",
)

# ---------------------------------------------------------------- 7. conceptual: conditional variance claim (core)
ex.add_conceptual(
    title="Diagnose the argument: 'an extreme reading should make us less sure'", difficulty=2,
    prompt_md=r"""
A sensor-fusion engineer models two thermometers on the same machine as jointly Gaussian with correlation
$\rho = 0.8$, standard deviations $\sigma_1 = \sigma_2 = 2$ °C around a shared mean of 60 °C. Thermometer 2 fails,
and she predicts its reading from thermometer 1 with the conditional distribution. A colleague objects:
"When thermometer 1 reads 70 °C — five standard deviations out — the prediction for thermometer 2 should carry a
much *wider* uncertainty than when it reads 61 °C. Your formula gives the same variance in both cases, so it must
be wrong."

1. Compute the conditional mean and standard deviation of thermometer 2 for readings of 61 °C and 70 °C.
2. Is the colleague right that the formula is wrong? What property of the Gaussian is at stake?
3. Describe one realistic situation in which the colleague's intuition *is* correct, and what it says about the model.
""",
    answer_md=r"""
1. $\mathbb{E}[X_2 \mid x_1] = 60 + \rho\frac{\sigma_2}{\sigma_1}(x_1 - 60) = 60 + 0.8(x_1 - 60)$, so $60.8$ °C for a
   reading of 61 and $68$ °C for a reading of 70 (regression to the mean: the prediction is pulled 20% back
   towards 60). The conditional variance is $\sigma_2^2(1 - \rho^2) = 4 \times 0.36 = 1.44$, standard deviation
   $1.2$ °C — in both cases.
2. The formula is right *for the model stated*. In a jointly Gaussian vector the Schur complement
   $\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ does not involve $x_2$ at all: the residual
   $X_2 - \mathbb{E}[X_2 \mid X_1]$ is independent of $X_1$, so its spread is the same whatever $X_1$ happens to be.
   This is a property (homoscedasticity) of the Gaussian, not a bug in the derivation.
3. The intuition is correct whenever the joint distribution is *not* Gaussian — for instance if a reading of 70 °C
   is far more likely to come from a faulty thermometer 1 than from a genuinely hot machine, or if the machine's
   behaviour at high temperatures is itself more erratic (heteroscedastic noise). Then the conditional variance
   really does grow with $|x_1 - 60|$, and the right response is to change the model: a mixture (normal operation
   versus sensor fault), a heavy-tailed joint such as a multivariate $t$, or a heteroscedastic regression that
   models $\operatorname{Var}(X_2 \mid x_1)$ explicitly. A five-sigma reading under the stated model has probability
   below $10^{-6}$, which is itself the signal that the Gaussian assumption, not the conditioning formula, deserves
   the scrutiny.
""",
)

# ---------------------------------------------------------------- 8. Cholesky sampling (core)
SOL_8 = r"""
def sample_mvn(mu, Sigma, n, rng):
    mu = np.asarray(mu, float); Sigma = np.asarray(Sigma, float)
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, len(mu)))
    return mu + Z @ L.T

def empirical_cov(X):
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (X.shape[0] - 1)
"""
ns8 = run(SOL_8, r"""
mu_t = np.array([1.0, -2.0, 0.0])
S_t = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 1.5], [1.0, 1.5, 2.0]])
Xs = sample_mvn(mu_t, S_t, 20_000, np.random.default_rng(7))
err_cov = np.abs(empirical_cov(Xs) - S_t).max()
err_mean = np.abs(Xs.mean(axis=0) - mu_t).max()
L_true = np.linalg.cholesky(S_t)
""")
ex.add(
    title="Sampling a multivariate Gaussian with a Cholesky factor", difficulty=2,
    prompt_md=r"""
Implement

* `sample_mvn(mu, Sigma, n, rng)` → an $n \times d$ array of samples from $\mathcal{N}(\mu, \Sigma)$ built as
  $x = \mu + Lz$ with $z \sim \mathcal{N}(0, I)$ and $LL^\top = \Sigma$ (`np.linalg.cholesky`), drawing the standard
  normals with **exactly one call** `rng.standard_normal((n, d))`; do **not** use `rng.multivariate_normal`;
* `empirical_cov(X)` → the sample covariance of the rows of `X` (you may reuse your code from Exercise 4).

The checks draw 20,000 samples with `np.random.default_rng(7)` from $\mu = (1, -2, 0)$,
$\Sigma = \begin{pmatrix} 4 & 2 & 1 \\ 2 & 3 & 1.5 \\ 1 & 1.5 & 2\end{pmatrix}$ and compare the empirical mean and
covariance with the targets, and they verify that your samples are the *exact* deterministic function
$\mu + Lz$ of the standard normals (so the implementation, not just its statistics, is tested).
""",
    starter=r"""
def sample_mvn(mu, Sigma, n, rng):
    # YOUR CODE HERE  (one call to rng.standard_normal((n, d)); then mu + Z @ L.T)
    return None

def empirical_cov(X):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_8,
    check=rf"""
mu_t = np.array([1.0, -2.0, 0.0]); S_t = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 1.5], [1.0, 1.5, 2.0]])
try:
    Xs = np.asarray(sample_mvn(mu_t, S_t, 20_000, np.random.default_rng(7)), float)
    ok_shape = Xs.shape == (20_000, 3)
except Exception:
    Xs, ok_shape = None, False
check('samples have shape (20000, 3)', ok_shape)
try:
    err_mean = float(np.abs(Xs.mean(axis=0) - mu_t).max())
    err_cov = float(np.abs(np.cov(Xs, rowvar=False) - S_t).max())
except Exception:
    err_mean, err_cov = None, None
check('empirical mean within 0.05 of mu in every coordinate', err_mean is not None and err_mean < 0.05)
check('empirical covariance within 0.1 of Sigma in every entry', err_cov is not None and err_cov < 0.1)
try:
    Z = np.random.default_rng(7).standard_normal((20_000, 3))
    exact = bool(np.allclose(Xs, mu_t + Z @ np.linalg.cholesky(S_t).T))
except Exception:
    exact = None
check('samples are exactly mu + L z for the standard normals drawn', exact is True)
check_hash('first sample (seed 7), rounded', (Xs[0] if ok_shape else None), '{sha_of(ns8["Xs"][0], decimals=3)}', decimals=3)
try:
    got = np.asarray(empirical_cov(Xs), float)
except Exception:
    got = None
check_close('empirical_cov agrees with np.cov', got, (np.cov(Xs, rowvar=False) if ok_shape else None), tol=1e-9)
try:
    iso = sample_mvn(np.zeros(2), np.eye(2), 5, np.random.default_rng(1))
    same = bool(np.allclose(iso, np.random.default_rng(1).standard_normal((5, 2))))
except Exception:
    same = None
check('with Sigma = I the samples are the raw standard normals', same is True)
""",
    explanation_md=r"""
If $z \sim \mathcal{N}(0, I)$ then $\mu + Lz \sim \mathcal{N}(\mu, LL^\top)$ by the affine-closure rule
$\operatorname{Cov}(Lz) = L\,I\,L^\top$. With rows as samples the matrix form is `mu + Z @ L.T` — the transpose is the
classic slip: `Z @ L` gives covariance $L^\top L$, which is a different matrix unless $L$ is diagonal. The check that
the samples equal $\mu + Lz$ *exactly* rules out `rng.multivariate_normal` (which draws the normals in a different
order and uses an SVD by default). Any square root $A$ with $AA^\top = \Sigma$ would also work — the symmetric
$\Sigma^{1/2}$ from an eigendecomposition, for instance — but the Cholesky factor is the cheapest ($d^3/3$ flops) and
is what `rng.multivariate_normal(method="cholesky")` uses. Sampling error of a covariance entry with 20,000 samples
is of order $\sqrt{2}\sigma_i\sigma_j/\sqrt{n} \approx 0.03$, comfortably inside the tolerance.
""",
)

# ---------------------------------------------------------------- 9. Simpson's paradox (core)
SIMPSON_DATA = r"""
# kidney-stone data (Charig et al., 1986): counts[treatment][stone size] = (patients, successes)
counts = {
    "open surgery": {"small": (87, 81),  "large": (263, 192)},
    "PCNL":         {"small": (270, 234), "large": (80, 55)},
}
"""
SOL_9 = r"""
def stratified_rates(counts):
    return {t: {s: n_s / n for s, (n, n_s) in d.items()} for t, d in counts.items()}

def aggregate_rates(counts):
    return {t: sum(n_s for n, n_s in d.values()) / sum(n for n, n_s in d.values()) for t, d in counts.items()}

def standardised_rates(counts):
    strata = list(next(iter(counts.values())).keys())
    tot = {s: sum(counts[t][s][0] for t in counts) for s in strata}
    N = sum(tot.values())
    weights = {s: tot[s] / N for s in strata}
    rates = stratified_rates(counts)
    return {t: sum(weights[s] * rates[t][s] for s in strata) for t in counts}

def is_simpson(counts):
    rates = stratified_rates(counts); agg = aggregate_rates(counts)
    (ta, ra), (tb, rb) = list(agg.items())
    strata = list(rates[ta].keys())
    a_wins_all = all(rates[ta][s] > rates[tb][s] for s in strata)
    b_wins_all = all(rates[tb][s] > rates[ta][s] for s in strata)
    return bool((a_wins_all and ra < rb) or (b_wins_all and rb < ra))
"""
ns9 = run(SIMPSON_DATA + SOL_9, r"""
strat = stratified_rates(counts)
agg = aggregate_rates(counts)
std = standardised_rates(counts)
simp = is_simpson(counts)
open_small = strat["open surgery"]["small"]; pcnl_large = strat["PCNL"]["large"]
agg_open = agg["open surgery"]; agg_pcnl = agg["PCNL"]
std_open = std["open surgery"]; std_pcnl = std["PCNL"]
guide = {"A": {"mild": (20, 18), "severe": (80, 48)}, "B": {"mild": (80, 68), "severe": (20, 10)}}
simp_guide = is_simpson(guide)
no_para = {"A": {"mild": (50, 45), "severe": (50, 30)}, "B": {"mild": (50, 40), "severe": (50, 25)}}
simp_none = is_simpson(no_para)
""")
ex.add(
    title="Simpson's paradox: stratified, aggregate and standardised rates", difficulty=2,
    prompt_md=r"""
The dictionary `counts` holds the kidney-stone data from the guide: for each treatment and each stone size, the
number of patients and the number of successes. Implement four functions that take such a dictionary (any number
of strata, exactly two treatments):

* `stratified_rates(counts)` → `{treatment: {stratum: success rate}}`;
* `aggregate_rates(counts)` → `{treatment: overall success rate}` (pool the strata);
* `standardised_rates(counts)` → `{treatment: rate}` where each treatment's stratum rates are weighted by the
  **pooled** stratum shares $P(s) = \frac{\text{patients in } s \text{ (both treatments)}}{\text{all patients}}$;
* `is_simpson(counts)` → `True` if one treatment has the strictly higher rate in *every* stratum but the strictly
  lower aggregate rate, else `False`.
""" + "\n```python\n" + textwrap.dedent(SIMPSON_DATA).strip() + "\n```\n",
    starter=SIMPSON_DATA + r"""

def stratified_rates(counts):
    # YOUR CODE HERE
    return None

def aggregate_rates(counts):
    # YOUR CODE HERE
    return None

def standardised_rates(counts):
    # YOUR CODE HERE
    return None

def is_simpson(counts):
    # YOUR CODE HERE
    return None
""",
    solution=SIMPSON_DATA + "\n" + SOL_9,
    check=rf"""
try:
    strat = stratified_rates(counts)
    got = float(strat["open surgery"]["small"])
except Exception:
    got = None
check_close('open surgery, small stones: 81/87', got, 81/87)
try:
    got = float(strat["PCNL"]["large"])
except Exception:
    got = None
check_hash('PCNL, large stones', got, '{sha_of(ns9["pcnl_large"], decimals=4)}', decimals=4)
try:
    agg = aggregate_rates(counts)
    got_o, got_p = float(agg["open surgery"]), float(agg["PCNL"])
except Exception:
    got_o, got_p = None, None
check_hash('aggregate rate, open surgery', got_o, '{sha_of(ns9["agg_open"], decimals=4)}', decimals=4)
check_hash('aggregate rate, PCNL', got_p, '{sha_of(ns9["agg_pcnl"], decimals=4)}', decimals=4)
try:
    std = standardised_rates(counts)
    got_o, got_p = float(std["open surgery"]), float(std["PCNL"])
except Exception:
    got_o, got_p = None, None
check_hash('standardised rate, open surgery', got_o, '{sha_of(ns9["std_open"], decimals=4)}', decimals=4)
check_hash('standardised rate, PCNL', got_p, '{sha_of(ns9["std_pcnl"], decimals=4)}', decimals=4)
check('standardisation restores the stratified ordering (open surgery ahead)', got_o is not None and got_p is not None and got_o > got_p)
try:
    s1 = is_simpson(counts)
    s2 = is_simpson({{"A": {{"mild": (20, 18), "severe": (80, 48)}}, "B": {{"mild": (80, 68), "severe": (20, 10)}}}})
    s3 = is_simpson({{"A": {{"mild": (50, 45), "severe": (50, 30)}}, "B": {{"mild": (50, 40), "severe": (50, 25)}}}})
    ok_s1 = s1 is not None and bool(s1)
    ok_s2 = s2 is not None and bool(s2)
    ok_s3 = s3 is not None and not s3
except Exception:
    ok_s1, ok_s2, ok_s3 = False, False, False
check('kidney-stone data show the paradox', ok_s1)
check("the guide's treatment table shows the paradox", ok_s2)
check('balanced groups (50/50 in both arms) show no paradox', ok_s3)
""",
    explanation_md=r"""
Stratified rates are ratios inside each cell; aggregate rates pool the numerators and denominators; the
standardised rate re-weights each treatment's stratum rates by a *common* mix (here the pooled shares
$P(\text{small}) = 357/700 = 0.51$). Open surgery wins in both strata ($0.93 > 0.87$ and $0.73 > 0.69$) but loses
overall ($0.78 < 0.83$) because 75% of its patients had large stones against 23% for PCNL; with the common mix it
is ahead again ($0.83 > 0.78$). The paradox disappears as soon as both arms have the same stratum composition —
which is precisely what randomisation guarantees in expectation. The classic coding error is to average the stratum
rates with equal weights instead of the pooled patient shares; that is also a standardisation, but to a different
(and arbitrary) reference population.
""",
)

# ---------------------------------------------------------------- 10. Mahalanobis anomaly detection (stretch)
MAHA_DATA = r"""
# two correlated process sensors (standardised); 300 normal readings plus 6 injected anomalies
mu_s = np.array([0.0, 0.0])
Sigma_s = np.array([[1.0, 0.9], [0.9, 1.0]])
normal = np.random.default_rng(10).multivariate_normal(mu_s, Sigma_s, size=300)   # own seed: do not change
anomalies = np.array([[ 2.0, -2.0], [-1.5,  1.5], [ 2.8,  2.8], [ 0.0,  2.5], [-3.6, -3.4], [ 1.0, -0.8]])
readings = np.vstack([normal, anomalies])      # rows 300..305 are the injected points
"""
SOL_10 = r"""
def mahalanobis_sq(X, mu, Sigma):
    X = np.atleast_2d(np.asarray(X, float))
    L = np.linalg.cholesky(np.asarray(Sigma, float))
    Y = np.linalg.solve(L, (X - np.asarray(mu, float)).T)
    return np.sum(Y**2, axis=0)

def flag_anomalies(X, mu, Sigma, alpha=0.01):
    d = len(mu)
    thresh = stats.chi2.ppf(1 - alpha, d)
    return np.flatnonzero(mahalanobis_sq(X, mu, Sigma) > thresh)

def flag_euclidean(X, mu, alpha=0.01):
    d = len(mu)
    thresh = stats.chi2.ppf(1 - alpha, d)
    return np.flatnonzero(np.sum((np.asarray(X, float) - mu)**2, axis=1) > thresh)
"""
ns10 = run(MAHA_DATA + SOL_10, r"""
idx_m = flag_anomalies(readings, mu_s, Sigma_s, 0.01)
idx_e = flag_euclidean(readings, mu_s, 0.01)
inj_caught_m = sorted(set(idx_m.tolist()) & set(range(300, 306)))
inj_caught_e = sorted(set(idx_e.tolist()) & set(range(300, 306)))
fa_m = int((idx_m < 300).sum()); fa_e = int((idx_e < 300).sum())
d2_anom = mahalanobis_sq(anomalies, mu_s, Sigma_s)
idx_m_str = " ".join(map(str, idx_m.tolist()))
caught_m_str = " ".join(map(str, inj_caught_m)); caught_e_str = " ".join(map(str, inj_caught_e))
""")
ex.add(
    title="Anomaly detection with the Mahalanobis distance", difficulty=3,
    prompt_md=r"""
Two standardised sensors on a reactor have correlation $0.9$ in normal operation. The array `readings` holds 300
normal readings followed by six injected anomalies (rows 300–305), generated by the seeded cell below (do not
change it). Implement

* `mahalanobis_sq(X, mu, Sigma)` → the squared Mahalanobis distance of every row of `X` (array of length $n$);
* `flag_anomalies(X, mu, Sigma, alpha=0.01)` → the sorted indices of the rows whose squared distance exceeds the
  $\chi^2_d$ quantile $\chi^2_{d, 1-\alpha}$ (`stats.chi2.ppf(1 - alpha, d)`), as an integer array;
* `flag_euclidean(X, mu, alpha=0.01)` → the same rule applied to the squared *Euclidean* distance $\|x - \mu\|^2$ with
  the same threshold (the naive detector that ignores the correlation).

Then compare: which injected anomalies does each rule catch, and how many normal readings does each flag?
""" + "\n```python\n" + textwrap.dedent(MAHA_DATA).strip() + "\n```\n",
    starter=MAHA_DATA + r"""

def mahalanobis_sq(X, mu, Sigma):
    # YOUR CODE HERE
    return None

def flag_anomalies(X, mu, Sigma, alpha=0.01):
    # YOUR CODE HERE
    return None

def flag_euclidean(X, mu, alpha=0.01):
    # YOUR CODE HERE
    return None
""",
    solution=MAHA_DATA + "\n" + SOL_10,
    check=rf"""
try:
    d2 = np.asarray(mahalanobis_sq(anomalies, mu_s, Sigma_s), float).ravel()
    ok = d2.shape == (6,)
except Exception:
    d2, ok = None, False
check('mahalanobis_sq returns one value per row', ok)
check_close('Mahalanobis^2 of the first anomaly (2, -2) is 80', (d2[0] if ok else None), 80.0, tol=1e-6)
check_hash('Mahalanobis^2 of all six injected points', (d2 if ok else None), '{sha_of(ns10["d2_anom"], decimals=2)}', decimals=2)
try:
    idx_m = np.asarray(flag_anomalies(readings, mu_s, Sigma_s, 0.01)).astype(int)
    idx_m_str = " ".join(map(str, sorted(idx_m.tolist())))
    caught_m = " ".join(map(str, sorted(set(idx_m.tolist()) & set(range(300, 306)))))
    fa_m = int((idx_m < 300).sum())
except Exception:
    idx_m_str, caught_m, fa_m = None, None, None
check_hash('all indices flagged by the Mahalanobis rule (alpha = 0.01)', idx_m_str, '{sha_of(ns10["idx_m_str"])}')
check_hash('injected anomalies caught by the Mahalanobis rule', caught_m, '{sha_of(ns10["caught_m_str"])}')
check('Mahalanobis rule flags at most 8 of the 300 normal readings (expected about 3 at alpha = 0.01)', fa_m is not None and fa_m <= 8)
try:
    idx_e = np.asarray(flag_euclidean(readings, mu_s, 0.01)).astype(int)
    caught_e = " ".join(map(str, sorted(set(idx_e.tolist()) & set(range(300, 306)))))
    fa_e = int((idx_e < 300).sum())
except Exception:
    caught_e, fa_e = None, None
check_hash('injected anomalies caught by the Euclidean rule', caught_e, '{sha_of(ns10["caught_e_str"])}')
check('the Euclidean rule misses the "across-the-correlation" anomalies that Mahalanobis catches',
      caught_m is not None and caught_e is not None and len(caught_m.split()) > len(caught_e.split()))
""",
    explanation_md=r"""
With $\rho = 0.9$ the inverse covariance is $\frac{1}{0.19}\begin{pmatrix} 1 & -0.9 \\ -0.9 & 1\end{pmatrix}$, so
$\Delta^2(x) = 5.26\,(x_1^2 - 1.8x_1x_2 + x_2^2)$: points *along* the diagonal are cheap, points *across* it are
expensive. The reading $(2, -2)$ has $\Delta^2 = 80$ against a 99% threshold of $9.21$ — an unmistakable anomaly
that the Euclidean rule (distance $2.83$, squared $8 < 9.21$) lets through, as it does $(-1.5, 1.5)$ and $(1, -0.8)$.
Conversely $(2.8, 2.8)$ lies along the correlation ($\Delta^2 = 8.25$) and is *not* flagged by Mahalanobis although
it is far out in Euclidean terms. Under the model $\Delta^2 \sim \chi^2_2$, so about 1% of normal readings (3 of 300)
are expected to be flagged — the false-alarm rate is controlled by construction. The Euclidean rule with the same
threshold cannot claim that, because $\|x\|^2$ does not follow $\chi^2_2$ for correlated data: on this draw it
flags 7 normal readings (all of them far out *along* the diagonal) while catching only two of the six injected points. This is Hotelling's
$T^2$ chart, and `sklearn.covariance.EllipticEnvelope` fits $\mu$ and $\Sigma$ robustly before applying the same test.
""",
)

# ---------------------------------------------------------------- 11. GP regression (stretch)
SOL_11 = r"""
def rbf_kernel(A, B, ell=1.0, sf=1.0):
    A = np.asarray(A, float).reshape(-1, 1); B = np.asarray(B, float).reshape(1, -1)
    return sf**2 * np.exp(-(A - B)**2 / (2 * ell**2))

def gp_posterior(X_train, y_train, X_test, ell=1.0, sf=1.0, noise=0.01):
    K = rbf_kernel(X_train, X_train, ell, sf) + noise * np.eye(len(X_train))
    Ks = rbf_kernel(X_train, X_test, ell, sf)
    Kss = rbf_kernel(X_test, X_test, ell, sf)
    alpha = np.linalg.solve(K, y_train)
    mean = Ks.T @ alpha
    cov = Kss - Ks.T @ np.linalg.solve(K, Ks)
    return mean, np.sqrt(np.clip(np.diag(cov), 0.0, None))
"""
ns11 = run(SOL_11, r"""
X_tr = np.array([-3.0, -1.5, 0.0, 1.0, 2.5])
y_tr = np.array([-0.6, 0.9, 0.1, -1.2, 0.4])
X_te = np.array([-2.0, 0.5, 4.0])
m, s = gp_posterior(X_tr, y_tr, X_te, ell=1.0, sf=1.0, noise=0.01)
m1, s1 = gp_posterior(np.array([0.0]), np.array([1.0]), np.array([0.5, 2.0]), 1.0, 1.0, 0.1)
m_long, s_long = gp_posterior(X_tr, y_tr, X_te, ell=3.0, sf=1.0, noise=0.01)
""")
ex.add(
    title="Gaussian-process regression as Gaussian conditioning", difficulty=3,
    prompt_md=r"""
Implement Gaussian-process regression with the RBF kernel $k(x, x') = \sigma_f^2\exp\big(-\frac{(x - x')^2}{2\ell^2}\big)$, with NumPy only:

* `rbf_kernel(A, B, ell=1.0, sf=1.0)` → the $|A| \times |B|$ matrix $k(a_i, b_j)$ for 1-D input arrays;
* `gp_posterior(X_train, y_train, X_test, ell=1.0, sf=1.0, noise=0.01)` → a tuple `(mean, sd)` of arrays over `X_test`:
  $$\text{mean} = K_*^\top (K + \text{noise}\, I)^{-1} y, \qquad \text{sd} = \sqrt{\operatorname{diag}\big(K_{**} - K_*^\top(K + \text{noise}\, I)^{-1}K_*\big)}.$$
  Use `np.linalg.solve`, not `np.linalg.inv`, and clip tiny negative variances to 0 before the square root.

Checks: the guide's one-point example (observe $y = 1$ at $x = 0$, noise $0.1$: mean $0.802$ and sd $0.540$ at
$x_* = 0.5$, mean $0.123$ and sd $0.992$ at $x_* = 2$), a five-point data set, and agreement with
`sklearn.gaussian_process.GaussianProcessRegressor` with fixed hyperparameters.
""",
    starter=r"""
def rbf_kernel(A, B, ell=1.0, sf=1.0):
    # YOUR CODE HERE
    return None

def gp_posterior(X_train, y_train, X_test, ell=1.0, sf=1.0, noise=0.01):
    # YOUR CODE HERE
    return None, None
""",
    solution=SOL_11,
    check=rf"""
try:
    Kt = np.asarray(rbf_kernel(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])), float)
    ok = Kt.shape == (2, 3)
except Exception:
    Kt, ok = None, False
check('rbf_kernel returns a |A| x |B| matrix', ok)
check_close('k(0, 0) = 1 and k(0, 2) = e^-2', (np.array([Kt[0, 0], Kt[0, 2]]) if ok else None), np.array([1.0, np.exp(-2.0)]))
try:
    m, s = gp_posterior(np.array([0.0]), np.array([1.0]), np.array([0.5, 2.0]), 1.0, 1.0, 0.1)
    m, s = np.asarray(m, float).ravel(), np.asarray(s, float).ravel()
except Exception:
    m, s = None, None
check_close("the guide's one-point example: posterior means", m, np.array([np.exp(-0.125) / 1.1, np.exp(-2.0) / 1.1]), tol=1e-6)
check_close("the guide's one-point example: posterior sds", s, np.sqrt(np.array([1 - np.exp(-0.125)**2 / 1.1, 1 - np.exp(-2.0)**2 / 1.1])), tol=1e-6)
X_tr = np.array([-3.0, -1.5, 0.0, 1.0, 2.5]); y_tr = np.array([-0.6, 0.9, 0.1, -1.2, 0.4]); X_te = np.array([-2.0, 0.5, 4.0])
try:
    m, s = gp_posterior(X_tr, y_tr, X_te, ell=1.0, sf=1.0, noise=0.01)
    m, s = np.asarray(m, float).ravel(), np.asarray(s, float).ravel()
except Exception:
    m, s = None, None
check_hash('five-point data: posterior mean at (-2, 0.5, 4)', m, '{sha_of(ns11["m"], decimals=3)}', decimals=3)
check_hash('five-point data: posterior sd at (-2, 0.5, 4)', s, '{sha_of(ns11["s"], decimals=3)}', decimals=3)
check('far from the data (x = 4) the sd has almost returned to the prior value 1', s is not None and s.shape == (3,) and s[2] > 0.9)
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    gpr = GaussianProcessRegressor(kernel=ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed"), alpha=0.01, optimizer=None).fit(X_tr[:, None], y_tr)
    sk_m, sk_s = gpr.predict(X_te[:, None], return_std=True)
except Exception:
    sk_m, sk_s = None, None
check_close('posterior mean agrees with scikit-learn', m, sk_m, tol=1e-6)
check_close('posterior sd agrees with scikit-learn', s, sk_s, tol=1e-6)
try:
    m_long, s_long = gp_posterior(X_tr, y_tr, X_te, ell=3.0, sf=1.0, noise=0.01)
    s_long = np.asarray(s_long, float).ravel()
except Exception:
    s_long = None
check('a longer length-scale (3.0) makes the far point (x = 4) less uncertain', s_long is not None and s is not None and s_long.shape == (3,) and s.shape == (3,) and s_long[2] < s[2])
""",
    explanation_md=r"""
The training outputs and the test values are one big Gaussian vector with covariance built from the kernel; the
posterior is Section 5's conditioning formula with $\Sigma_{12} = K_*$, $\Sigma_{22} = K + \sigma_n^2 I$ and
$\Sigma_{11} = K_{**}$. Solving $K\alpha = y$ once and reusing $\alpha$ is both faster and more stable than forming the
inverse. The one-point example shows the mechanics: $k_* = e^{-0.125} = 0.88$ at distance $0.5$ gives a mean of
$0.88/1.1$ and a variance of $1 - 0.88^2/1.1$, while two length-scales away $k_* = e^{-2}$ and the prior is almost
untouched. A longer $\ell$ makes distant points more correlated, so the posterior at $x = 4$ tightens — the
length-scale encodes how far information travels. Clipping the diagonal before the square root matters because the
Schur complement can produce $-10^{-16}$ at training points through round-off. With `optimizer=None` and fixed
kernel parameters scikit-learn computes the identical formula (via a Cholesky factor), hence the agreement to $10^{-6}$.
""",
)

# ---------------------------------------------------------------- 12. mini-project: Kalman filter (★★★)
PROJECT_CODE = r"""
rng = np.random.default_rng(2024)
T, dt = 200, 1.0
v_true, q, r = 0.5, 0.05, 4.0                      # true velocity, process noise variance, sensor noise variance

# --- synthetic track and measurements ---
x_true = np.empty(T); x_true[0] = 0.0
for t in range(1, T):
    x_true[t] = x_true[t - 1] + v_true * dt + rng.normal(0, np.sqrt(q))
z = x_true + rng.normal(0, np.sqrt(r), T)
z[120] += 15.0                                     # one gross outlier (sensor glitch)

def kalman_1d(z, v, q, r, mu0, P0):
    mus, Ps, Ks = np.empty(len(z)), np.empty(len(z)), np.empty(len(z))
    mu, P = mu0, P0
    for t, zt in enumerate(z):
        if t > 0:
            mu, P = mu + v * dt, P + q                 # predict
        K = P / (P + r)                                # gain
        mu, P = mu + K * (zt - mu), (1 - K) * P        # update
        mus[t], Ps[t], Ks[t] = mu, P, K
    return mus, Ps, Ks

def moving_average(z, window):
    out = np.empty_like(z)
    for t in range(len(z)):
        out[t] = z[max(0, t - window + 1): t + 1].mean()    # causal (uses only past and present)
    return out

rmse = lambda est: float(np.sqrt(np.mean((est - x_true)**2)))
mus, Ps, Ks = kalman_1d(z, v_true, q, r, mu0=0.0, P0=10.0)
print(f"RMSE  raw sensor      : {rmse(z):.3f}")
for wdw in (3, 5, 10, 20):
    print(f"RMSE  moving avg w={wdw:2d}: {rmse(moving_average(z, wdw)):.3f}")
print(f"RMSE  Kalman filter   : {rmse(mus):.3f}")

P_inf = (q + np.sqrt(q**2 + 4*q*r)) / 2
print(f"\nsteady state: P- = {P_inf:.3f}, K = {P_inf/(P_inf + r):.3f}, filtered sd = {np.sqrt(P_inf*r/(P_inf + r)):.3f}  vs sensor sd {np.sqrt(r):.3f}")
print(f"filter's final gain {Ks[-1]:.3f}, final sd {np.sqrt(Ps[-1]):.3f}")
inside = np.mean(np.abs(mus - x_true) <= 2 * np.sqrt(Ps))
print(f"fraction of time steps with the truth inside the +/-2 sd band: {inside:.3f}  (about 0.95 if the filter is calibrated)")

# --- misspecified velocity: the filter lags behind ---
mus_wrong, _, _ = kalman_1d(z, 0.0, q, r, 0.0, 10.0)
print(f"\nKalman with the wrong velocity model (v = 0): RMSE {rmse(mus_wrong):.3f}, mean error (bias) {np.mean(mus_wrong - x_true):+.3f}")
mus_bigq, _, _ = kalman_1d(z, 0.0, 2.0, r, 0.0, 10.0)
print(f"... same wrong model but q raised to 2.0:    RMSE {rmse(mus_bigq):.3f}, bias {np.mean(mus_bigq - x_true):+.3f}  (more trust in the sensor, less lag)")

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(x_true, "k", lw=2, label="true position")
axes[0].plot(z, ".", ms=3, alpha=0.5, label="sensor")
axes[0].plot(moving_average(z, 10), "g", lw=1.2, label="moving average (w = 10)")
axes[0].plot(mus, "r", lw=1.5, label="Kalman")
axes[0].fill_between(np.arange(T), mus - 2*np.sqrt(Ps), mus + 2*np.sqrt(Ps), color="r", alpha=0.15)
axes[0].set_ylabel("position"); axes[0].set_title("Tracking a noisy sensor: Kalman filter vs moving average"); axes[0].legend(fontsize=8)
axes[1].plot(z - x_true, ".", ms=3, alpha=0.5, label="sensor error")
axes[1].plot(moving_average(z, 10) - x_true, "g", lw=1.2, label="moving-average error")
axes[1].plot(mus - x_true, "r", lw=1.5, label="Kalman error")
axes[1].axhline(0, color="k", lw=0.8); axes[1].set_xlabel("time step"); axes[1].set_ylabel("error"); axes[1].legend(fontsize=8)
axes[1].set_title(f"Errors against the true track (the {z[120] - x_true[120]:+.1f} spike is the sensor glitch at t = 120)")
plt.tight_layout(); plt.show()
print(f"\nerror right after the outlier at t = 120: sensor {z[120] - x_true[120]:+.2f}, Kalman {mus[120] - x_true[120]:+.2f}, "
      f"Kalman 5 steps later {mus[125] - x_true[125]:+.2f}  (the gain K = {Ks[120]:.2f} damps the glitch and the filter recovers)")
"""
ex.add_project(
    title="Mini-project: a one-dimensional Kalman filter for a noisy sensor track",
    prompt_md=r"""
An object moves along a line with (approximately) constant velocity; a sensor reports its position once per time
step with large noise. Build a Kalman filter for the track and compare it with the two naive alternatives — the raw
sensor and a moving average.

**The model** (use `rng = np.random.default_rng(2024)`, $T = 200$ steps, $\Delta t = 1$):

1. **Truth:** $x_t = x_{t-1} + v\,\Delta t + w_t$ with $v = 0.5$, $w_t \sim \mathcal{N}(0, q)$, $q = 0.05$, $x_0 = 0$.
2. **Sensor:** $z_t = x_t + r_t$ with $r_t \sim \mathcal{N}(0, r)$, $r = 4$. Add one gross outlier: `z[120] += 15`.
3. **Filter:** implement `kalman_1d(z, v, q, r, mu0, P0)` returning the arrays of posterior means, posterior
   variances and gains over time (predict $\mu^- = \mu + v\Delta t$, $P^- = P + q$; update with $K = P^-/(P^- + r)$).
   Start from $\mu_0 = 0$, $P_0 = 10$.
4. **Baseline:** a *causal* moving average of the measurements over the last $w$ steps, for $w \in \{3, 5, 10, 20\}$.

**Deliverables**

* RMSE against the true track for the raw sensor, each moving average and the Kalman filter;
* the steady-state gain and filtered standard deviation from the formula $P^-_\infty = \frac{q + \sqrt{q^2 + 4qr}}{2}$,
  compared with the filter's final values;
* a calibration check: the fraction of time steps at which the truth lies inside the filter's $\pm 2$ sd band;
* a plot of truth, measurements, moving average and Kalman estimate with its band, plus an error plot;
* a misspecification experiment: run the filter with the wrong velocity $v = 0$ and report the bias (mean error);
  then show that raising $q$ reduces the lag at the price of a noisier estimate;
* a comment on how the filter handles the outlier at $t = 120$ and why.
""",
    rubric_md=r"""
* The simulation is seeded and the truth, measurements and outlier are generated exactly as specified.
* The filter alternates predict and update steps, stores mean, variance and gain at every step, and the update is
  recognisably the Gaussian conditioning formula (gain $= P^-/(P^- + r)$, variance $(1 - K)P^-$).
* The moving average is causal (no future measurements) and is compared at several window lengths.
* RMSEs are reported for all methods; the Kalman filter beats the raw sensor by a wide margin and at least matches
  the best moving average.
* The steady-state gain and variance are computed from the formula and agree with the filter's late-time values;
  the $\pm 2$ sd band contains the truth roughly 95% of the time.
* The misspecified filter ($v = 0$) shows a systematic lag (negative bias), and the trade-off when inflating $q$ is
  explained: a larger process noise means a larger gain, more weight on the sensor, less lag but more noise.
* The plots have titles, axis labels and legends.
""",
    solution_sketch_md=r"""
Generate the truth with a loop (each step depends on the previous one) and the measurements in one vectorised line.
The filter is ten lines: predict, gain, update, store. With $q = 0.05$ and $r = 4$ the steady-state prediction variance
is $P^-_\infty = (0.05 + \sqrt{0.0025 + 0.8})/2 \approx 0.47$, the gain is $K_\infty \approx 0.105$ and the filtered
standard deviation is about $0.65$ — a threefold improvement on the sensor's $2.0$. The RMSE of the raw sensor comes
out near $2.2$ (the outlier adds a little). A causal moving average faces a dilemma on a *moving* target: a short
window ($w = 3$ or $5$, RMSE about $1.4$) stays noisy, a long window lags behind by about $v(w-1)/2$ — five units for
$w = 20$ — and is *worse* than the raw sensor. The Kalman filter reaches about $0.8$ because it knows the object moves
at $v$ and never lags (the RMSE is above the steady-state $0.65$ mainly because of the start-up transient and the
outlier). The truth lies inside the $\pm 2$ sd band at roughly 93–95% of the steps, so the filter's own uncertainty
is honest. With $v = 0$ the filter systematically trails the object by about $v(1 - K_\infty)/K_\infty \approx 4$ units
(bias about $-3.9$); inflating $q$ to $2$ raises the gain, cuts the bias to about $-0.4$ but lets more sensor noise
through. The outlier at $t = 120$ is multiplied by the gain ($K \approx 0.1$), so a $+15$ glitch moves the estimate
by only about $1.5$ and the error decays geometrically over the following steps; a robust filter would additionally
gate measurements whose innovation $z_t - \mu^-$ exceeds a few $\sqrt{P^- + r}$.
""",
    solution_code=PROJECT_CODE,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print(f"exercises: {len(ex.items)} items — {kinds.count('code')} coding, {kinds.count('concept')} conceptual, {kinds.count('project')} project")
print("difficulties:", [it["difficulty"] for it in ex.items])
