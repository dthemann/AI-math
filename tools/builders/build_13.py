#!/usr/bin/env python3
"""Builder for Module 13 — Linear Models and Regularisation.

Generates lesson.ipynb, exercises.ipynb and solutions.ipynb into
part-5-learning-theory/13-linear-models-and-regularization. Hidden answers for
the exercise self-checks are computed here by *executing the solution code*.
"""
import pathlib
import sys
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import Notebook, ExerciseSet, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-5-learning-theory/13-linear-models-and-regularization"
warnings.filterwarnings("ignore")


def safe_to_hash(x: float, decimals: int = 3, margin: float = 2e-4) -> float:
    """Assert that x is not within `margin` of a rounding boundary at `decimals` places."""
    frac = abs(x) * 10 ** decimals
    dist = abs(frac - np.floor(frac) - 0.5)   # distance (in units of the last digit) to the .5 boundary
    assert dist > margin * 10 ** decimals, f"hidden value {x!r} too close to a rounding boundary"
    return x


# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 13 — Linear Models and Regularisation")

nb.md(r"""
This notebook walks through the ideas of the module with code you can edit and re-run:
empirical versus true risk, ordinary least squares derived three ways, polynomial features
and overfitting, ridge regression through the SVD, the lasso by coordinate descent, logistic
and Poisson regression fitted by Newton's method, the bias–variance decomposition measured by
simulation, cross-validation and learning curves, and the diagnostics that tell you when a
fit is lying.

Read the [concept guide](README.md) first, or keep it open alongside — every section here
mirrors a numbered section there. Code cells build each object **from scratch in NumPy**
first and then check against the scikit-learn call that does the same.
""")

nb.code(r"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
from sklearn.datasets import load_breast_cancer, load_diabetes
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
warnings.filterwarnings("ignore")
""")

# ---------------------------------------------------------------- Section 1
nb.md(r"""
## 1. Empirical risk versus true risk

Supervised learning picks a function $f$ from a hypothesis class to minimise the
**empirical risk** $\hat R_n(f) = \frac1n\sum_i \ell(f(x_i), y_i)$, hoping that the
**true risk** $R(f) = \mathbb{E}[\ell(f(X), Y)]$ on fresh data is small too. The two are
not the same thing, and the gap between them grows with the flexibility of the class.

We generate $y = \sin(2\pi x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, 0.3^2)$,
fit polynomials of increasing degree on $n = 30$ training points, and estimate the true risk
on 20,000 fresh points (a stand-in for the expectation we cannot compute).
""")

nb.code(r"""
def f_true(x):
    return np.sin(2 * np.pi * x)

sigma = 0.3
n_train = 30
x_tr = rng.uniform(0, 1, n_train)
y_tr = f_true(x_tr) + sigma * rng.normal(size=n_train)
x_fresh = rng.uniform(0, 1, 20_000)                     # "the population"
y_fresh = f_true(x_fresh) + sigma * rng.normal(size=20_000)

degrees = np.arange(0, 10)
emp_risk, true_risk = [], []
for d in degrees:
    coef = np.polyfit(x_tr, y_tr, d)                    # least squares in the polynomial basis
    emp_risk.append(np.mean((np.polyval(coef, x_tr) - y_tr) ** 2))
    true_risk.append(np.mean((np.polyval(coef, x_fresh) - y_fresh) ** 2))
emp_risk, true_risk = np.array(emp_risk), np.array(true_risk)

print("degree  empirical risk (train MSE)   true risk (fresh MSE)   gap")
for d, e, t in zip(degrees, emp_risk, true_risk):
    print(f"{d:6d}  {e:26.4f}   {t:21.4f}   {t - e:6.3f}")
print(f"\nirreducible noise sigma^2 = {sigma**2:.3f}")

plt.semilogy(degrees, emp_risk, "s--", label="empirical risk (training data)")
plt.semilogy(degrees, true_risk, "o-", label="true risk (20,000 fresh points)")
plt.axhline(sigma ** 2, color="gray", ls=":", label="noise floor sigma^2")
plt.title("Training error flatters the model; fresh data does not")
plt.xlabel("polynomial degree"); plt.ylabel("mean squared error (log scale)"); plt.legend(); plt.show()
""")

nb.md(r"""
Training error falls monotonically with the degree — a richer class always fits the data
it was fitted on at least as well. The true risk falls to a minimum around degree 3–5 and then
rises as the polynomial starts fitting the noise; nothing ever gets below the noise floor
$\sigma^2 = 0.09$. Everything that follows in this module is about understanding, controlling
and *estimating* that gap.

**Try it yourself:** set `n_train = 300` and re-run. The gap shrinks for every degree — more
data is the one regulariser that costs no bias.
""")

# ---------------------------------------------------------------- Section 2
nb.md(r"""
## 2. Ordinary least squares, three ways

For the model $y = \beta_0 + \beta_1 x$ on the five points of the concept guide, the OLS
coefficients can be reached by

1. **geometry** — the normal equations $X^\top X\hat\beta = X^\top y$ (the residual is
   orthogonal to the columns of $X$);
2. **calculus** — gradient descent on $\operatorname{RSS}(\beta) = \|y - X\beta\|^2$ with
   gradient $-2X^\top(y - X\beta)$;
3. **probability** — numerical maximum likelihood under $y_i \sim \mathcal{N}(\beta^\top x_i, \sigma^2)$.

We also compute the hat matrix $H$, leverages, $R^2$, $\hat\sigma^2 = \operatorname{RSS}/(n-p)$
and the standard error of the slope.
""")

nb.code(r"""
x5 = np.array([1., 2., 3., 4., 5.])
y5 = np.array([2., 3., 5., 4., 6.])
X5 = np.column_stack([np.ones(5), x5])
n, p = X5.shape

# (1) geometry: normal equations
beta_ne = np.linalg.solve(X5.T @ X5, X5.T @ y5)

# (2) calculus: gradient descent on RSS
beta_gd = np.zeros(2)
eta = 0.03                                  # must be < 2 / lambda_max(2 X^T X) ~ 0.0169*2
for _ in range(3000):
    beta_gd -= eta * (-2 * X5.T @ (y5 - X5 @ beta_gd)) / n

# (3) probability: maximise the Gaussian log-likelihood numerically over (beta0, beta1, log sigma)
def neg_loglik(theta):
    b, log_s = theta[:2], theta[2]
    return -np.sum(stats.norm.logpdf(y5, loc=X5 @ b, scale=np.exp(log_s)))
res = optimize.minimize(neg_loglik, x0=np.array([0., 0., 0.]), method="BFGS")
beta_ml, sigma_ml = res.x[:2], np.exp(res.x[2])

print("normal equations:   ", beta_ne)
print("gradient descent:   ", beta_gd)
print("maximum likelihood: ", beta_ml, "  sigma_MLE^2 = %.4f" % sigma_ml**2)
print("np.linalg.lstsq:    ", np.linalg.lstsq(X5, y5, rcond=None)[0])

# hat matrix, residuals, fit statistics
H = X5 @ np.linalg.solve(X5.T @ X5, X5.T)
y_hat = H @ y5
e = y5 - y_hat
rss, tss = e @ e, ((y5 - y5.mean()) ** 2).sum()
sigma2_hat = rss / (n - p)
se = np.sqrt(np.diag(sigma2_hat * np.linalg.inv(X5.T @ X5)))
print("\nfitted values:", y_hat, "  residuals:", e)
print("X^T e = ", X5.T @ e, " (residual orthogonal to the columns)")
print("leverages h_ii:", np.diag(H), "  sum = %.1f = p" % np.trace(H))
print("RSS = %.2f, TSS = %.2f, R^2 = %.3f" % (rss, tss, 1 - rss / tss))
print("sigma^2 (unbiased, RSS/(n-p)) = %.4f   vs MLE RSS/n = %.4f" % (sigma2_hat, rss / n))
print("SE(beta) =", se, "  t for slope = %.2f, two-sided p = %.3f"
      % (beta_ne[1] / se[1], 2 * stats.t.sf(beta_ne[1] / se[1], df=n - p)))

xx = np.linspace(0.5, 5.5, 50)
plt.scatter(x5, y5, color="k", zorder=3, label="data")
plt.plot(xx, beta_ne[0] + beta_ne[1] * xx, label="OLS fit  y = 1.3 + 0.9 x")
plt.vlines(x5, y_hat, y5, color="tab:red", label="residuals")
plt.title("OLS: the fitted line makes the residuals orthogonal to (1, x)")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
""")

nb.md(r"""
All three routes land on $\hat\beta = (1.3, 0.9)$, and `lstsq` agrees. Note the two noise
estimates: maximum likelihood divides the RSS by $n$ and is biased low; the unbiased estimate
divides by $n - p = 3$. The leverages $0.6, 0.3, 0.2, 0.3, 0.6$ sum to $p = 2$ and are largest
at the ends of the $x$-range — those points pull hardest on the line, which matters again in
Sections 9 and 10.
""")

# ---------------------------------------------------------------- Section 3
nb.md(r"""
## 3. Basis expansion and overfitting

A model that is linear in its *coefficients* can be as curved as you like in $x$: replace $x$
by features $\phi(x) = (1, x, x^2, \dots, x^k)$ and run the same least squares. Two things go
wrong when $k$ is large. Statistically, a degree-9 polynomial through 10 points interpolates
them exactly and oscillates wildly in between. Numerically, the columns $x^k$ have wildly
different scales, so the design matrix is badly conditioned (Module 03) unless you
**standardise** $x$ first.
""")

nb.code(r"""
x10 = np.linspace(0, 1, 10)
y10 = f_true(x10) + sigma * rng.normal(size=10)
x_test = rng.uniform(0, 1, 2000)
y_test = f_true(x_test) + sigma * rng.normal(size=2000)
grid = np.linspace(0, 1, 300)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, d in zip(axes, [1, 3, 9]):
    coef = np.polyfit(x10, y10, d)
    tr_rmse = np.sqrt(np.mean((np.polyval(coef, x10) - y10) ** 2))
    te_rmse = np.sqrt(np.mean((np.polyval(coef, x_test) - y_test) ** 2))
    ax.plot(grid, f_true(grid), "k:", label="truth sin(2 pi x)")
    ax.plot(grid, np.polyval(coef, grid), label=f"degree {d} fit")
    ax.scatter(x10, y10, color="tab:red", zorder=3, label="10 training points")
    ax.set_ylim(-2, 2); ax.set_title(f"degree {d}: train RMSE {tr_rmse:.2f}, test RMSE {te_rmse:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Polynomial features: underfit (1), about right (3), interpolate the noise (9)")
plt.tight_layout(); plt.show()

# conditioning of the design matrix, raw versus standardised x
def poly_design(x, degree):
    return np.column_stack([x ** k for k in range(degree + 1)])
x_raw = np.linspace(0, 10, 11)
x_std = (x_raw - x_raw.mean()) / x_raw.std()
for d in [3, 6, 9]:
    print(f"degree {d}: condition number raw x = {np.linalg.cond(poly_design(x_raw, d)):10.3e}"
          f"   standardised x = {np.linalg.cond(poly_design(x_std, d)):10.3e}")
""")

nb.md(r"""
The degree-9 curve passes through every training point (train RMSE 0.00) yet its error on
fresh data (0.49) is far worse than the cubic's (0.31): it has memorised the noise, and the
wiggles between the points are the price. Degree 1 cannot bend at all and is worst of the
three on both counts. The condition numbers show the second problem: with raw $x \in [0, 10]$
the degree-9 design matrix is conditioned like $10^{11}$, so you lose about eleven digits in
the solve; standardising brings it down by seven orders of magnitude. **Always standardise
before adding powers or a penalty** — the next two sections rely on it.

**Try it yourself:** change the training sample to 30 points (`np.linspace(0, 1, 30)`).
Degree 9 is no longer an interpolant; how much does its test RMSE improve?
""")

# ---------------------------------------------------------------- Section 4
nb.md(r"""
## 4. Ridge regression: shrinkage along singular directions

Ridge minimises $\|y - X\beta\|^2 + \lambda\|\beta\|^2$ and has the closed form
$\hat\beta_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$. Through the SVD $X = UDV^\top$,

$$\hat\beta_\lambda = \sum_j \frac{d_j}{d_j^2 + \lambda}\,\langle u_j, y\rangle\,v_j
\qquad\text{so each OLS component is multiplied by the filter factor } \frac{d_j^2}{d_j^2 + \lambda},$$

and the effective degrees of freedom are $\operatorname{df}(\lambda) = \sum_j d_j^2/(d_j^2+\lambda)$.

The design below has six standardised features of which $x_1$ is almost a copy of $x_0$
(correlation 0.98) and $x_3$ is moderately correlated with $x_2$. The true coefficients are
$(2, 0, -1.5, 0.8, 0, 0)$.
""")

nb.code(r"""
from sklearn.linear_model import Ridge

def make_design(n=60, seed=0):
    g = np.random.default_rng(seed)
    Z = g.normal(size=(n, 6))
    Z[:, 1] = Z[:, 0] + 0.2 * g.normal(size=n)          # x1 is a noisy copy of x0
    Z[:, 3] = 0.6 * Z[:, 2] + 0.8 * g.normal(size=n)    # x3 partly overlaps x2
    X = (Z - Z.mean(axis=0)) / Z.std(axis=0)            # standardise
    beta_true = np.array([2., 0., -1.5, 0.8, 0., 0.])
    y = X @ beta_true + 1.5 * g.normal(size=n)
    return X, y - y.mean(), beta_true                   # centred y: no intercept needed

Xr, yr, beta_true = make_design()
print("correlation(x0, x1) = %.3f" % np.corrcoef(Xr[:, 0], Xr[:, 1])[0, 1])

def ridge_closed_form(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def ridge_svd(X, y, lam):
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((d / (d ** 2 + lam)) * (U.T @ y))

beta_ols = ridge_closed_form(Xr, yr, 0.0)
print("true coefficients:", beta_true)
print("OLS:              ", beta_ols, "  (x0 and x1 fight over the shared signal)")
for lam in [1., 10., 100.]:
    b = ridge_closed_form(Xr, yr, lam)
    assert np.allclose(b, ridge_svd(Xr, yr, lam))
    assert np.allclose(b, Ridge(alpha=lam, fit_intercept=False).fit(Xr, yr).coef_)
    print(f"ridge lambda={lam:5.0f}:", b, " ||beta|| = %.2f" % np.linalg.norm(b))
print("(closed form == SVD formula == sklearn.linear_model.Ridge for every lambda)")

# coefficient path, filter factors and effective degrees of freedom
lams = np.logspace(-2, 3, 60)
path = np.array([ridge_closed_form(Xr, yr, lam) for lam in lams])
U, d, Vt = np.linalg.svd(Xr, full_matrices=False)
print("\nsingular values of X:", d)
print("smallest right singular vector v_6 (the direction the data barely constrain):", Vt[-1])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for j in range(6):
    axes[0].semilogx(lams, path[:, j], label=f"beta_{j} (true {beta_true[j]:g})")
axes[0].axhline(0, color="gray", lw=0.5)
axes[0].set_title("Ridge coefficient path"); axes[0].set_xlabel("lambda"); axes[0].set_ylabel("coefficient")
axes[0].legend(fontsize=7, ncol=2)
for j in range(6):
    axes[1].semilogx(lams, d[j] ** 2 / (d[j] ** 2 + lams), label=f"d_{j+1} = {d[j]:.2f}")
axes[1].semilogx(lams, (d[:, None] ** 2 / (d[:, None] ** 2 + lams[None, :])).sum(0) / 6, "k--", label="df(lambda) / 6")
axes[1].set_title("Filter factors d_j^2 / (d_j^2 + lambda)"); axes[1].set_xlabel("lambda"); axes[1].set_ylabel("fraction of OLS component kept")
axes[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
OLS splits the shared signal between $x_0$ and $x_1$ almost arbitrarily ($3.60$ and $-1.46$
where the truth is $2$ and $0$ — note the opposite signs). The smallest singular value,
$d_6 \approx 0.94$, belongs to the direction $v_6 \approx (x_1 - x_0)/\sqrt2$ — exactly the
"difference" the data cannot resolve — and its filter factor ($0.47$ at $\lambda = 1$, $0.08$
at $\lambda = 10$) is the first to collapse as $\lambda$ grows. By $\lambda = 10$ the two
coefficients are $1.14$ and $0.74$: both positive, both wrong individually, but their *sum*
($1.88$, the quantity the data actually determine) is close to the true $2$. The other
directions, with $d_j^2$ between 22 and 129, keep 70–93 % of their weight at $\lambda = 10$ and
are only seriously shrunk when $\lambda$ reaches their scale. The effective degrees of freedom
slide from 6 towards 0: $5.4$ at $\lambda = 1$, $4.3$ at $\lambda = 10$, $1.9$ at $\lambda = 100$.

**Try it yourself:** in the Bayesian reading $\lambda = \sigma^2/\tau^2$. The noise here has
$\sigma = 1.5$; which prior standard deviation $\tau$ on the coefficients corresponds to
$\lambda = 10$? Does it look reasonable next to the true coefficients?
""")

# ---------------------------------------------------------------- Section 5
nb.md(r"""
## 5. Lasso: soft-thresholding and coordinate descent

The lasso minimises $\tfrac12\|y - X\beta\|^2 + \lambda\|\beta\|_1$. For an orthonormal design
the solution is the **soft-thresholding** of the OLS coefficients,
$S_\lambda(z) = \operatorname{sign}(z)\max(|z| - \lambda, 0)$; in general **coordinate descent**
applies the same operator one coordinate at a time to the partial residual:

$$\beta_j \leftarrow \frac{S_\lambda\!\left(x_j^\top r^{(j)}\right)}{\|x_j\|^2},
\qquad r^{(j)} = y - \sum_{k\ne j} x_k\beta_k .$$

scikit-learn's `Lasso(alpha)` minimises $\frac{1}{2n}\|y - X\beta\|^2 + \alpha\|\beta\|_1$,
so $\lambda = n\alpha$.
""")

nb.code(r"""
from sklearn.linear_model import Lasso

def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

def lasso_cd(X, y, lam, n_iter=500):
    n, p = X.shape
    beta = np.zeros(p)
    r = y.copy()                                    # full residual y - X beta
    col_sq = (X ** 2).sum(axis=0)
    for _ in range(n_iter):
        for j in range(p):
            r += X[:, j] * beta[j]                  # partial residual without feature j
            beta[j] = soft_threshold(X[:, j] @ r, lam) / col_sq[j]
            r -= X[:, j] * beta[j]
    return beta

print("soft_threshold([3, -0.5, 1.2], 1) =", soft_threshold(np.array([3., -0.5, 1.2]), 1.0))
for lam in [5., 20., 60.]:
    b_cd = lasso_cd(Xr, yr, lam)
    b_sk = Lasso(alpha=lam / len(yr), fit_intercept=False, tol=1e-10, max_iter=100_000).fit(Xr, yr).coef_
    print(f"lasso lambda={lam:4.0f}: {b_cd}   max |ours - sklearn| = {np.abs(b_cd - b_sk).max():.1e}")
print("true:              ", beta_true)

# lasso path on the diabetes data (10 standardised features, centred target)
Xd, yd = load_diabetes(return_X_y=True)
names = load_diabetes().feature_names
Xd_s = (Xd - Xd.mean(axis=0)) / Xd.std(axis=0)
yd_c = yd - yd.mean()
alphas = np.logspace(-1, 1.8, 40)
lasso_path = np.array([lasso_cd(Xd_s, yd_c, a * len(yd_c), n_iter=300) for a in alphas])
n_nonzero = (np.abs(lasso_path) > 1e-8).sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
z = np.linspace(-3, 3, 200)
axes[0].plot(z, z, "k:", label="OLS (identity)")
axes[0].plot(z, soft_threshold(z, 1.0), label="lasso: soft threshold, lambda = 1")
axes[0].plot(z, z / 2, label="ridge: z / (1 + lambda), lambda = 1")
axes[0].set_title("Orthonormal design: how each penalty moves a coefficient")
axes[0].set_xlabel("OLS coefficient"); axes[0].set_ylabel("penalised coefficient"); axes[0].legend(fontsize=8)
for j in range(10):
    axes[1].semilogx(alphas, lasso_path[:, j], label=names[j])
axes[1].axhline(0, color="gray", lw=0.5)
axes[1].set_title("Lasso path on the diabetes data (coordinate descent)")
axes[1].set_xlabel("alpha (sklearn convention, lambda = n alpha)"); axes[1].set_ylabel("standardised coefficient")
axes[1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.show()
for a, k in zip(alphas[::8], n_nonzero[::8]):
    print(f"alpha = {a:6.2f}: {k:2d} non-zero coefficients")
""")

nb.md(r"""
Our twenty-line coordinate descent matches scikit-learn to about $10^{-10}$. On the synthetic
design the lasso does what ridge cannot: already at $\lambda = 5$ it sets the coefficient of
the near-duplicate $x_1$ to exactly zero and gives $x_0$ its full weight ($2.00$, the truth);
at $\lambda = 20$ three more irrelevant coefficients are exactly zero while the survivors are
shrunk. Which of two near-duplicates survives is decided by tiny differences in correlation
with $y$ — the *choice* between them is arbitrary, the *selection* of the pair is not. On the
diabetes data the path shows the order in which features enter as the penalty relaxes:
`bmi` and `s5` (a serum triglyceride measure) first, then blood pressure and `s3`, then sex,
`s6` and `s1`; `s4`, `s2` and `age` stay at zero for most of the path because they are highly
correlated with features already in the model (`s2` with `s1` at $0.90$, `s4` with `s3`) and add
little once those are in.

**Try it yourself:** add a ridge term to turn `lasso_cd` into an elastic net — the update
becomes $S_\lambda(x_j^\top r^{(j)})/(\|x_j\|^2 + \lambda_2)$. With $\lambda_2 = 20$, do $x_0$
and $x_1$ now share the weight?
""")

# ---------------------------------------------------------------- Section 6
nb.md(r"""
## 6. Logistic regression by Newton's method (IRLS)

For binary $y$, $p(x) = \sigma(\beta^\top x)$ and the negative log-likelihood
$\ell(\beta) = \sum_i[\log(1 + e^{\beta^\top x_i}) - y_i\beta^\top x_i]$ has

$$\nabla\ell = X^\top(p - y), \qquad \nabla^2\ell = X^\top W X, \quad W = \operatorname{diag}(p_i(1-p_i)).$$

Newton's step $\beta \leftarrow \beta - (X^\top WX)^{-1}X^\top(p - y)$ is a weighted least-squares
fit, hence *iteratively reweighted least squares*. We add an L2 penalty $\frac{\lambda}{2}\|\beta_{1:}\|^2$
(not on the intercept) and fit the standardised breast-cancer data (569 tumours, 30 features,
target 1 = benign). scikit-learn's `LogisticRegression(C)` solves the same problem with
$\lambda = 1/C$.
""")

nb.code(r"""
from sklearn.linear_model import LogisticRegression

bc = load_breast_cancer()
Xb = (bc.data - bc.data.mean(axis=0)) / bc.data.std(axis=0)
yb = bc.target.astype(float)
Xb1 = np.column_stack([np.ones(len(yb)), Xb])          # intercept column first

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_nll(beta, X, y, lam, pen):
    z = X @ beta
    return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(pen * beta ** 2)

def logistic_newton(X, y, lam=1.0, n_iter=50, tol=1e-10):
    beta = np.zeros(X.shape[1])
    pen = np.ones(X.shape[1]); pen[0] = 0.0            # do not penalise the intercept
    history = [logistic_nll(beta, X, y, lam, pen)]
    for it in range(n_iter):
        p = sigmoid(X @ beta)
        grad = X.T @ (p - y) + lam * pen * beta
        W = p * (1 - p)
        Hess = X.T @ (W[:, None] * X) + lam * np.diag(pen)
        step = np.linalg.solve(Hess, grad)
        beta = beta - step
        history.append(logistic_nll(beta, X, y, lam, pen))
        if np.abs(step).max() < tol:
            break
    return beta, np.array(history)

lam = 1.0
beta_nt, hist = logistic_newton(Xb1, yb, lam=lam)
print("penalised negative log-likelihood per Newton iteration:")
print(np.round(hist, 4))

clf = LogisticRegression(C=1.0 / lam, tol=1e-10, max_iter=10_000).fit(Xb, yb)
beta_sk = np.r_[clf.intercept_, clf.coef_.ravel()]
print("\nmax |Newton - sklearn| = %.1e" % np.abs(beta_nt - beta_sk).max())
p_hat = sigmoid(Xb1 @ beta_nt)
print("training accuracy = %.4f" % np.mean((p_hat > 0.5) == yb))
print("calibration: mean predicted probability %.4f  vs  observed rate %.4f" % (p_hat.mean(), yb.mean()))
order = np.argsort(-np.abs(beta_nt[1:]))[:5]
print("\nlargest standardised effects (odds ratio per standard deviation of the feature):")
for j in order:
    print(f"  {bc.feature_names[j]:24s} beta = {beta_nt[1 + j]:7.3f}   odds ratio = {np.exp(beta_nt[1 + j]):.3f}")
""")

nb.md(r"""
Newton converges in about ten iterations — the quadratic convergence shows as the number of
correct digits roughly doubling each step near the end — and lands on the same coefficients as
scikit-learn. The calibration identity $\sum_i p_i = \sum_i y_i$ holds exactly: with an
intercept, the score equation forces the predicted probabilities to average to the observed
rate. Each coefficient is a log odds ratio per standard deviation: a tumour one standard
deviation higher in "worst texture" has about $0.27\times$ the odds of being benign, all else
equal.

### Separable data: why the unpenalised MLE does not exist

If a hyperplane separates the classes perfectly, scaling $\beta$ up only makes the likelihood
better, so plain gradient ascent never stops. We compare eight points that are separable with
the same points after two labels are swapped.
""")

nb.code(r"""
xs = np.array([-2., -1.5, -1., -0.5, 0.5, 1., 1.5, 2.])
Xs1 = np.column_stack([np.ones(8), xs])
y_sep = (xs > 0).astype(float)                         # perfectly separable at x = 0
y_over = y_sep.copy(); y_over[2] = 1.0; y_over[5] = 0.0  # two labels swapped: overlapping

def gradient_ascent(X, y, lam=0.0, n_iter=3000, lr=0.5):
    beta = np.zeros(2); norms = []
    for _ in range(n_iter):
        p = sigmoid(X @ beta)
        beta += lr * (X.T @ (y - p) - lam * np.r_[0.0, beta[1]])
        norms.append(np.linalg.norm(beta))
    return beta, np.array(norms)

runs = {"separable, lambda = 0": gradient_ascent(Xs1, y_sep),
        "overlapping, lambda = 0": gradient_ascent(Xs1, y_over),
        "separable, lambda = 1": gradient_ascent(Xs1, y_sep, lam=1.0)}
for name, (b, norms) in runs.items():
    p = sigmoid(Xs1 @ b)
    nll = -np.sum(np.where(name.startswith("sep"), y_sep, y_over) * np.log(p)
                  + (1 - np.where(name.startswith("sep"), y_sep, y_over)) * np.log(1 - p))
    print(f"{name:26s} after 3000 steps: beta = {b},  ||beta|| = {norms[-1]:6.2f},  NLL = {nll:.4f}")
    plt.semilogx(np.arange(1, 3001), norms, label=name)
plt.title("Gradient ascent on the log-likelihood: separable data never converges")
plt.xlabel("iteration"); plt.ylabel("||beta||"); plt.legend(); plt.show()
""")

nb.md(r"""
On the overlapping data the coefficients settle within a hundred steps. On the separable
data the norm keeps growing — slowly, because the gradient $X^\top(y - p)$ shrinks as the
probabilities saturate, but without limit; the negative log-likelihood creeps towards zero and
`np.linalg.solve` in a Newton implementation would eventually hit a singular $X^\top WX$.
The L2 penalty makes the objective strictly convex and the fit stops at a finite $\beta$. This
is why scikit-learn regularises by default and why "my logistic regression coefficients are
huge" usually means "my classes are (nearly) separable".

**Try it yourself:** separable data in two dimensions is common when $p$ is large relative
to $n$. Fit `logistic_newton` with `lam=0.0` to the first 100 rows of `Xb1` only; watch the
coefficients and the number of iterations.
""")

# ---------------------------------------------------------------- Section 7
nb.md(r"""
## 7. Generalised linear models: Poisson regression

A GLM pairs an exponential-family distribution with a link function; with the canonical link
the score equations are always $X^\top(y - \mu) = 0$ and Newton's method is IRLS with
$W_{ii} = \operatorname{Var}(y_i)$. For counts, $y_i \sim \text{Poisson}(\mu_i)$ with
$\log\mu_i = \log t_i + \beta^\top x_i$, where $t_i$ is the **exposure** (an offset with
coefficient fixed to 1). Below: synthetic motor-insurance claims — each policy has an exposure
between 0.2 and 1 years, a standardised driver-age score and an urban indicator, with true
coefficients $(-0.3, -0.35, 0.5)$. We fit by plain gradient ascent, by Newton, and with
`sklearn.linear_model.PoissonRegressor`, then compute the deviance.
""")

nb.code(r"""
from sklearn.linear_model import PoissonRegressor

n_pol = 600
age_z = rng.normal(size=n_pol)
urban = rng.binomial(1, 0.4, size=n_pol).astype(float)
exposure = rng.uniform(0.2, 1.0, size=n_pol)
Xp = np.column_stack([np.ones(n_pol), age_z, urban])
beta_pois_true = np.array([-0.3, -0.35, 0.5])
claims = rng.poisson(exposure * np.exp(Xp @ beta_pois_true)).astype(float)
offset = np.log(exposure)
print("policies: %d, total claims: %d, share with zero claims: %.2f" % (n_pol, claims.sum(), (claims == 0).mean()))

def poisson_loglik(beta, X, y, offset):
    eta = offset + X @ beta
    return np.sum(y * eta - np.exp(eta))               # up to the constant -sum(log y!)

# (a) gradient ascent on the log-likelihood: gradient X^T (y - mu)
beta_ga = np.zeros(3); ll_hist = []
for it in range(2000):
    mu = np.exp(offset + Xp @ beta_ga)
    beta_ga += 0.05 * Xp.T @ (claims - mu) / n_pol
    ll_hist.append(poisson_loglik(beta_ga, Xp, claims, offset))

# (b) Newton / IRLS: Hessian X^T diag(mu) X
beta_nt = np.zeros(3)
for it in range(25):
    mu = np.exp(offset + Xp @ beta_nt)
    step = np.linalg.solve(Xp.T @ (mu[:, None] * Xp), Xp.T @ (claims - mu))
    beta_nt += step
    if np.abs(step).max() < 1e-12:
        break

# (c) scikit-learn: model the *rate* y / t with sample weights t (equivalent to the offset)
pr = PoissonRegressor(alpha=0.0, tol=1e-12, max_iter=10_000).fit(Xp[:, 1:], claims / exposure, sample_weight=exposure)
beta_sk = np.r_[pr.intercept_, pr.coef_]

print("true coefficients:      ", beta_pois_true)
print("gradient ascent (2000): ", beta_ga)
print("Newton (%2d iterations):  " % (it + 1), beta_nt)
print("sklearn PoissonRegressor:", beta_sk)
print("rate ratios exp(beta):   ", np.exp(beta_nt), " -> urban policies claim %.0f%% more often" % (100 * (np.exp(beta_nt[2]) - 1)))

def poisson_deviance(y, mu):
    terms = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0) - (y - mu)
    return 2 * terms.sum()

mu_hat = np.exp(offset + Xp @ beta_nt)
mu_null = exposure * claims.sum() / exposure.sum()     # intercept-only model: one overall rate
print("\nresidual deviance = %.1f on %d degrees of freedom   (null deviance %.1f)"
      % (poisson_deviance(claims, mu_hat), n_pol - 3, poisson_deviance(claims, mu_null)))

plt.plot(ll_hist)
plt.title("Poisson regression: log-likelihood during gradient ascent")
plt.xlabel("iteration"); plt.ylabel("log-likelihood (up to a constant)"); plt.show()
""")

nb.md(r"""
Three fits, one answer. Newton needs a handful of iterations where gradient ascent needs
hundreds, because the Hessian $X^\top\operatorname{diag}(\mu)X$ rescales each direction
correctly (Module 05). The coefficients are read as **rate ratios**: $e^{\hat\beta_{\text{urban}}}$
is the factor by which the claim rate is multiplied for urban drivers. The residual deviance is
close to its degrees of freedom, as it should be for data that really are Poisson; for real
claims it is typically 1.5–2 times larger (overdispersion), a signal to inflate the standard
errors.
""")

# ---------------------------------------------------------------- Section 8
nb.md(r"""
## 8. The bias–variance decomposition, measured

For $y = f(x) + \varepsilon$ and a model $\hat f$ fitted on a random training set $D$,

$$\mathbb{E}\big[(y - \hat f(x))^2\big] = \sigma^2 + \big(f(x) - \mathbb{E}_D[\hat f(x)]\big)^2 + \mathbb{E}_D\big[(\hat f(x) - \mathbb{E}_D[\hat f(x)])^2\big].$$

The expectation over $D$ is exactly what a simulation can approximate: draw 200 training sets
of $n = 30$ points, fit a polynomial of each degree to each, and look at the spread of the 200
fitted curves on a test grid. We also measure the test error directly and check that the
three terms add up to it.
""")

nb.code(r"""
def bias_variance(degree, n_train=30, n_rep=200, sigma=0.3, seed=0):
    g = np.random.default_rng(seed)
    x_grid = np.linspace(0, 1, 101)
    f_grid = f_true(x_grid)
    preds = np.empty((n_rep, len(x_grid)))
    test_mse = np.empty(n_rep)
    for r in range(n_rep):
        x = g.uniform(0, 1, n_train)
        y = f_true(x) + sigma * g.normal(size=n_train)
        coef = np.polyfit(x, y, degree)
        preds[r] = np.polyval(coef, x_grid)
        y_grid = f_grid + sigma * g.normal(size=len(x_grid))          # fresh noisy targets
        test_mse[r] = np.mean((y_grid - preds[r]) ** 2)
    bias2 = np.mean((preds.mean(axis=0) - f_grid) ** 2)
    var = np.mean(preds.var(axis=0))
    return bias2, var, test_mse.mean(), preds, x_grid

degs = np.arange(0, 11)
table = np.array([bias_variance(d)[:3] for d in degs])
print("degree   bias^2   variance   noise   bias^2+var+noise   measured test MSE")
for d, (b2, v, m) in zip(degs, table):
    print(f"{d:6d}  {b2:7.4f}  {v:9.4f}  {sigma**2:6.3f}  {b2 + v + sigma**2:17.4f}  {m:18.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(degs, table[:, 0], "o-", label="bias^2")
axes[0].semilogy(degs, table[:, 1], "s-", label="variance")
axes[0].semilogy(degs, np.full_like(degs, sigma ** 2, dtype=float), "k:", label="noise sigma^2")
axes[0].semilogy(degs, table[:, 2], "^-", label="test MSE")
axes[0].set_title("Bias^2 falls, variance rises: the U-curve"); axes[0].set_xlabel("polynomial degree"); axes[0].set_ylabel("error (log scale)")
axes[0].legend(fontsize=8)
for d, c in [(1, "tab:blue"), (3, "tab:green"), (9, "tab:red")]:
    _, _, _, preds, x_grid = bias_variance(d)
    for r in range(15):
        axes[1].plot(x_grid, preds[r], color=c, alpha=0.25, lw=0.8)
    axes[1].plot(x_grid, preds.mean(axis=0), color=c, lw=2, label=f"degree {d}: mean of 200 fits")
axes[1].plot(x_grid, f_true(x_grid), "k--", label="truth")
axes[1].set_ylim(-2.5, 2.5); axes[1].set_title("15 fitted curves per degree (thin) and their mean (thick)")
axes[1].set_xlabel("x"); axes[1].set_ylabel("prediction"); axes[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The decomposition is exact up to Monte Carlo error: the sum of the three estimated terms
tracks the measured test MSE in every row. Degree 1 is all bias — the mean of its 200 fits is
a straight line through a sine. Degree 9 is all variance — the individual fits swing by
several units near the edges of the interval, though their *average* is close to the truth.
Degree 3 to 5 balances the two. Regularisation moves along the same trade-off without
changing the degree: increasing $\lambda$ in ridge raises bias and lowers variance exactly
as decreasing the degree does here.

**Try it yourself:** set `n_train=300` inside the call. The variance of every degree drops
by roughly a factor of 10 (it scales like $p/n$) and the bottom of the U moves to a higher
degree — with more data you can afford a more flexible model.
""")

# ---------------------------------------------------------------- Section 9
nb.md(r"""
## 9. Cross-validation, the leave-one-out shortcut and learning curves

$k$-fold cross-validation estimates the true risk by holding out each fold in turn. We use it
to choose $\lambda$ for ridge on the diabetes data with all 65 degree-2 polynomial features
(10 linear, 10 squares, 45 interactions) — a setting where regularisation matters. Error bars
are the standard error of the fold means; the **one-standard-error rule** picks the largest
$\lambda$ whose error is within one standard error of the minimum.
""")

nb.code(r"""
from sklearn.preprocessing import PolynomialFeatures

P = PolynomialFeatures(degree=2, include_bias=False).fit_transform(Xd)
P = (P - P.mean(axis=0)) / P.std(axis=0)
n_d = len(yd)
print("design matrix with degree-2 features:", P.shape)

def kfold_indices(n, k, g):
    return np.array_split(g.permutation(n), k)

def ridge_with_intercept(Xtr, ytr, lam):
    mu, ym = Xtr.mean(axis=0), ytr.mean()
    b = np.linalg.solve((Xtr - mu).T @ (Xtr - mu) + lam * np.eye(Xtr.shape[1]), (Xtr - mu).T @ (ytr - ym))
    return b, mu, ym

def cv_ridge(X, y, lam, folds):
    fold_mse = []
    for f in range(len(folds)):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(len(folds)) if g != f])
        b, mu, ym = ridge_with_intercept(X[train], y[train], lam)   # centring learned on the training fold only
        pred = (X[test] - mu) @ b + ym
        fold_mse.append(np.mean((y[test] - pred) ** 2))
    return np.array(fold_mse)

folds = kfold_indices(n_d, 10, np.random.default_rng(0))
lams = np.logspace(-2, 4, 31)
cv = np.array([cv_ridge(P, yd, lam, folds) for lam in lams])          # shape (31, 10)
cv_mean, cv_se = cv.mean(axis=1), cv.std(axis=1, ddof=1) / np.sqrt(10)
i_best = int(np.argmin(cv_mean))
i_1se = int(np.max(np.where(cv_mean <= cv_mean[i_best] + cv_se[i_best])[0]))
print(f"lambda minimising CV error: {lams[i_best]:.1f}  (CV RMSE {np.sqrt(cv_mean[i_best]):.2f})")
print(f"one-standard-error rule:    {lams[i_1se]:.1f}  (CV RMSE {np.sqrt(cv_mean[i_1se]):.2f})")
print(f"for comparison, OLS on the 65 features (lambda -> 0): CV RMSE {np.sqrt(cv_mean[0]):.2f}")

plt.errorbar(lams, np.sqrt(cv_mean), yerr=cv_se / (2 * np.sqrt(cv_mean)), fmt="o-", capsize=3, label="10-fold CV RMSE +/- 1 SE")
plt.axvline(lams[i_best], color="tab:green", ls="--", label="minimum")
plt.axvline(lams[i_1se], color="tab:orange", ls=":", label="one-SE rule")
plt.xscale("log"); plt.title("Choosing lambda for ridge on 65 polynomial features (diabetes data)")
plt.xlabel("lambda"); plt.ylabel("cross-validated RMSE"); plt.legend(); plt.show()
""")

nb.md(r"""
The curve is the U of Section 8 seen through cross-validation: too little penalty and the 65
features overfit; too much and the model is pulled towards the mean. The minimum sits around
$\lambda \approx 100$, and the one-standard-error rule picks a larger, simpler model that is
statistically indistinguishable from it.

### The leave-one-out shortcut

For any linear smoother $\hat y = Hy$ the leave-one-out error needs no refitting:
$\operatorname{CV}_{\text{LOO}} = \frac1n\sum_i\big(\frac{y_i - \hat y_i}{1 - h_{ii}}\big)^2$.
We verify it against brute force (442 refits) for ridge; `RidgeCV` uses the same identity.
""")

nb.code(r"""
import time
from sklearn.linear_model import RidgeCV

def loo_ridge_shortcut(X, y, lam):
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    H = Xc @ np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T)
    h = np.diag(H) + 1.0 / len(y)                    # + 1/n: the intercept's share of the leverage
    e = yc - H @ yc
    return np.mean((e / (1 - h)) ** 2)

def loo_ridge_brute(X, y, lam):
    errs = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b, mu, ym = ridge_with_intercept(X[m], y[m], lam)
        errs.append((y[i] - ((X[i] - mu) @ b + ym)) ** 2)
    return np.mean(errs)

lam = 100.0
t0 = time.time(); loo_fast = loo_ridge_shortcut(P, yd, lam); t_fast = time.time() - t0
t0 = time.time(); loo_slow = loo_ridge_brute(P, yd, lam); t_slow = time.time() - t0
print(f"LOO MSE at lambda = {lam:g}: shortcut {loo_fast:.6f} ({t_fast*1e3:.1f} ms)   brute force {loo_slow:.6f} ({t_slow*1e3:.0f} ms)")
loo_curve = np.array([loo_ridge_shortcut(P, yd, l) for l in lams])
rcv = RidgeCV(alphas=lams).fit(P, yd)                # efficient LOO internally
print(f"LOO-optimal lambda: ours {lams[np.argmin(loo_curve)]:.1f}   sklearn RidgeCV {rcv.alpha_:.1f}")
""")

nb.md(r"""
The shortcut agrees with 442 explicit refits to every printed digit and runs hundreds of times
faster. (The extra $1/n$ in the leverage is the intercept's contribution: the fitted value
$\bar y + \hat\beta^\top(x_i - \bar x)$ is a linear smoother whose hat matrix is
$\frac1n\mathbf{1}\mathbf{1}^\top + H_c$.)

### Learning curves

Finally, plot training and validation error against the training-set size $n$ for an
underfitting, a reasonable and an overfitting model.
""")

nb.code(r"""
def learning_curve(degree, sizes, n_rep=40, sigma=0.3, seed=1):
    g = np.random.default_rng(seed)
    tr, va = [], []
    for n in sizes:
        tr_n, va_n = [], []
        for _ in range(n_rep):
            x = g.uniform(0, 1, n); y = f_true(x) + sigma * g.normal(size=n)
            coef = np.polyfit(x, y, degree)
            tr_n.append(np.mean((np.polyval(coef, x) - y) ** 2))
            xv = g.uniform(0, 1, 500); yv = f_true(xv) + sigma * g.normal(size=500)
            va_n.append(np.mean((np.polyval(coef, xv) - yv) ** 2))
        tr.append(np.mean(tr_n)); va.append(np.median(va_n))     # median: degree 9 at n = 12 has wild outliers
    return np.array(tr), np.array(va)

sizes = np.array([12, 15, 20, 30, 50, 80, 120, 200, 400])
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, d in zip(axes, [1, 3, 9]):
    tr, va = learning_curve(d, sizes)
    ax.semilogx(sizes, tr, "s--", label="training error")
    ax.semilogx(sizes, va, "o-", label="validation error")
    ax.axhline(sigma ** 2, color="gray", ls=":", label="noise floor")
    ax.set_ylim(0, 0.6); ax.set_title(f"degree {d}"); ax.set_xlabel("training-set size n")
    ax.set_xticks([12, 20, 50, 100, 200, 400]); ax.set_xticklabels(["12", "20", "50", "100", "200", "400"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
axes[0].set_ylabel("mean squared error"); axes[0].legend(fontsize=8)
fig.suptitle("Learning curves: high bias (left) vs high variance (right)")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Degree 1: the two curves meet almost immediately at a high plateau — more data cannot help a
model that cannot represent the truth. Degree 9: a large gap at small $n$ that closes slowly as
$n$ grows — the signature of variance, and the case where more data *does* help. Degree 3
converges quickly to the noise floor. When a model disappoints, this plot tells you whether to
collect data or to change the model.

**Try it yourself:** the CV curve above used a single random fold assignment. Re-run the
CV cell with `np.random.default_rng(1)` for the folds. How much does the chosen $\lambda$
move? That variability is why nested CV is needed before you *report* the score of the
winner.
""")

# ---------------------------------------------------------------- Section 10
nb.md(r"""
## 10. Diagnostics: residuals, leverage and multicollinearity

Three quick checks after any fit: residuals against fitted values (should be a structureless
band), leverage (points far out in feature space, $h_{ii} > 2p/n$), and variance inflation
factors $\operatorname{VIF}_j = 1/(1 - R_j^2)$ for correlated features.
""")

nb.code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# (a) heteroscedastic noise: the funnel
xa = rng.uniform(1, 10, 150)
ya = 2 + 0.5 * xa + 0.15 * xa * rng.normal(size=150)      # noise grows with x
Xa = np.column_stack([np.ones(150), xa]); ba = np.linalg.lstsq(Xa, ya, rcond=None)[0]
axes[0].scatter(Xa @ ba, ya - Xa @ ba, s=12); axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_title("(a) funnel: variance grows with the fit"); axes[0].set_xlabel("fitted value"); axes[0].set_ylabel("residual")

# (b) a missing non-linearity: the curve
xb_ = rng.uniform(-2, 2, 150)
yb_ = 1 + xb_ + 0.8 * xb_ ** 2 + 0.4 * rng.normal(size=150)
Xb_ = np.column_stack([np.ones(150), xb_]); bb = np.linalg.lstsq(Xb_, yb_, rcond=None)[0]
axes[1].scatter(Xb_ @ bb, yb_ - Xb_ @ bb, s=12); axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_title("(b) curve: the model is missing x^2"); axes[1].set_xlabel("fitted value"); axes[1].set_ylabel("residual")

# (c) one high-leverage point steers the whole line
x_lev = np.r_[x5, 10.0]; y_lev = np.r_[y5, 4.0]
X_lev = np.column_stack([np.ones(6), x_lev])
H_lev = X_lev @ np.linalg.solve(X_lev.T @ X_lev, X_lev.T)
b_with = np.linalg.lstsq(X_lev, y_lev, rcond=None)[0]
b_without = np.linalg.lstsq(X5, y5, rcond=None)[0]
xx = np.linspace(0, 11, 50)
axes[2].scatter(x5, y5, color="k", zorder=3); axes[2].scatter([10], [4], color="tab:red", zorder=3, label=f"new point, leverage {H_lev[-1, -1]:.2f}")
axes[2].plot(xx, b_without[0] + b_without[1] * xx, "k--", label="fit without it")
axes[2].plot(xx, b_with[0] + b_with[1] * xx, "tab:red", label="fit with it")
axes[2].set_title("(c) leverage: 2p/n = %.2f" % (2 * 2 / 6)); axes[2].set_xlabel("x"); axes[2].set_ylabel("y"); axes[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
print("leverages with the extra point:", np.diag(H_lev))

# variance inflation factors on the diabetes features
def vif(X):
    out = []
    for j in range(X.shape[1]):
        others = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        bj = np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        r2 = 1 - ((X[:, j] - others @ bj) ** 2).sum() / ((X[:, j] - X[:, j].mean()) ** 2).sum()
        out.append(1 / (1 - r2))
    return np.array(out)
v = vif(Xd_s)
b_ols_d = np.linalg.lstsq(np.column_stack([np.ones(n_d), Xd_s]), yd, rcond=None)[0][1:]
print("\nfeature   VIF     OLS coefficient (standardised)")
for name, vj, bj in zip(names, v, b_ols_d):
    print(f"{name:7s} {vj:7.2f}   {bj:8.2f}" + ("   <- strongly collinear" if vj > 10 else ""))
print("correlation(s1, s2) = %.3f" % np.corrcoef(Xd_s[:, 4], Xd_s[:, 5])[0, 1])
""")

nb.md(r"""
Panel (a) is heteroscedasticity: the OLS coefficients are still unbiased but the standard
errors and prediction intervals are wrong (too narrow where the noise is large). Panel (b)
says the model is missing a term — a quadratic fixes it. Panel (c) shows a single point with
leverage $0.84$ (the threshold $2p/n$ is $0.67$) rotating the fitted line; a point like this
deserves a check of the data before it is allowed to vote. The VIF table explains the diabetes
coefficients: `s1` and `s2` are correlated at $0.90$, inflate each other's variance by factors
of 59 and 39, and receive large OLS coefficients of opposite sign — the same symptom as
$x_0, x_1$ in Section 4, and the reason the lasso path in Section 5 kept only one of them.
""")

# ---------------------------------------------------------------- Summary
nb.md(r"""
## Summary

- Training error estimates the empirical risk and flatters flexible models; only fresh data
  (or cross-validation) estimates the true risk.
- OLS is one answer with three derivations — projection, zero gradient, Gaussian maximum
  likelihood; the hat matrix $H$ gives fitted values, leverages and the LOO shortcut.
- Basis expansion keeps the model linear in its coefficients while bending in $x$; standardise
  before adding powers or penalties.
- Ridge multiplies each singular-direction component by $d_j^2/(d_j^2 + \lambda)$: the directions
  the data barely constrain are shrunk first; $\operatorname{df}(\lambda)$ measures what is left.
- Lasso is soft-thresholding; coordinate descent solves it in a few lines and produces exact
  zeros; among correlated features the choice is arbitrary but the selection is not.
- Logistic and Poisson regression are GLMs: score $X^\top(y - \mu)$, Newton = IRLS; on
  separable data the unpenalised MLE does not exist.
- Expected squared error = noise + bias² + variance, measurable by simulation; complexity and
  regularisation move along the same trade-off.
- $k$-fold CV with error bars, the one-standard-error rule and learning curves tell you how
  much to regularise and whether more data would help; nested CV keeps the reported score honest.
- Residual plots, leverage and VIF catch heteroscedasticity, influential points and
  collinearity before they reach production.

## Next: open `exercises.ipynb`

Twelve exercises with instant self-checks, from the normal equations to an end-to-end
regularised regression with nested cross-validation and a bootstrap confidence interval.
""")

nb.save(MOD / "lesson.ipynb")
print("lesson cells:", len(nb.cells))

# =============================================================================
# EXERCISES — solution code is executed here to compute the hidden answers
# =============================================================================
SETUP = r"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso, PoissonRegressor
from sklearn.preprocessing import PolynomialFeatures
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
warnings.filterwarnings("ignore")

def make_design(n=60, seed=0):
    # six standardised features; x1 is a noisy copy of x0, x3 overlaps x2; centred target
    g = np.random.default_rng(seed)
    Z = g.normal(size=(n, 6))
    Z[:, 1] = Z[:, 0] + 0.2 * g.normal(size=n)
    Z[:, 3] = 0.6 * Z[:, 2] + 0.8 * g.normal(size=n)
    X = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    beta_true = np.array([2., 0., -1.5, 0.8, 0., 0.])
    y = X @ beta_true + 1.5 * g.normal(size=n)
    return X, y - y.mean(), beta_true

def make_poly_diabetes():
    # all 65 degree-2 features of the diabetes data, standardised; centred target
    X, y = load_diabetes(return_X_y=True)
    P = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    P = (P - P.mean(axis=0)) / P.std(axis=0)
    return P, y - y.mean()
"""

ex = ExerciseSet(
    "Module 13 — Exercises",
    intro_md=r"""
Exercises for **Module 13 — Linear models and regularisation**. They follow the order of the
[concept guide](README.md): the normal equations and $R^2$, polynomial features and
standardisation, soft-thresholding, ridge in closed form, the lasso by coordinate descent,
the logistic-regression gradient and Newton step, the bias–variance decomposition by
simulation, $k$-fold cross-validation, a conceptual question on standardisation, Poisson
regression with an offset, the leave-one-out shortcut, and a mini-project on the diabetes
data. Use only NumPy (plus the scikit-learn objects named in a prompt) and keep everything
seeded.
""",
    setup_code=SETUP,
)

ns: dict = {}
exec(SETUP, ns)  # gives the builder the same np/rng/helper names as the notebook


def run_solution(src: str) -> None:
    """Execute solution code inside the shared namespace (hashes come from *running* it)."""
    exec(src, ns)


# ---------------------------------------------------------------- Ex 1
SOL1 = r"""
def ols_fit(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

def r_squared(y, y_hat):
    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - rss / tss)
"""
run_solution(SOL1)
_g = np.random.default_rng(3)
_X1 = np.column_stack([np.ones(50), _g.normal(size=(50, 3))])
_y1 = _X1 @ np.array([1.0, 2.0, -1.0, 0.5]) + _g.normal(size=50)
_r2_hidden = ns["r_squared"](_y1, _X1 @ ns["ols_fit"](_X1, _y1))
safe_to_hash(_r2_hidden)
h1 = sha_of(_r2_hidden)

ex.add(
    title="OLS via the normal equations and R^2",
    difficulty=1,
    prompt_md=r"""
Implement

* `ols_fit(X, y)` — the least-squares coefficients $\hat\beta = (X^\top X)^{-1}X^\top y$ for a
  design matrix `X` (which already contains the intercept column if one is wanted). Use
  `np.linalg.solve` on the normal equations; do **not** call `lstsq`, `pinv` or `inv`.
* `r_squared(y, y_hat)` — the coefficient of determination $R^2 = 1 - \operatorname{RSS}/\operatorname{TSS}$
  as a Python `float`, with $\operatorname{TSS} = \sum_i (y_i - \bar y)^2$.

The first checks use the five-point example of the concept guide, $x = (1,2,3,4,5)$,
$y = (2,3,5,4,6)$, whose answer you can verify by hand.
""",
    starter=r"""
def ols_fit(X, y):
    # YOUR CODE HERE  (normal equations with np.linalg.solve)
    return None

def r_squared(y, y_hat):
    # YOUR CODE HERE
    return None
""",
    solution=SOL1,
    check=r"""
_x5 = np.array([1., 2., 3., 4., 5.]); _y5 = np.array([2., 3., 5., 4., 6.])
_X5 = np.column_stack([np.ones(5), _x5])
try:
    b5 = ols_fit(_X5, _y5)
except Exception:
    b5 = None
check_close('five-point example: beta = (1.3, 0.9)', b5, np.array([1.3, 0.9]), tol=1e-9)
try:
    r2_5 = r_squared(_y5, _X5 @ b5)
except Exception:
    r2_5 = None
check_close('five-point example: R^2 = 0.81', r2_5, 0.81, tol=1e-9)
_g = np.random.default_rng(3)
_X = np.column_stack([np.ones(50), _g.normal(size=(50, 3))])
_y = _X @ np.array([1.0, 2.0, -1.0, 0.5]) + _g.normal(size=50)
try:
    b = ols_fit(_X, _y)
except Exception:
    b = None
check_close('matches np.linalg.lstsq on a 50 x 4 design', b, np.linalg.lstsq(_X, _y, rcond=None)[0], tol=1e-8)
check('residual is orthogonal to the columns (X^T e = 0)', b is not None and np.allclose(_X.T @ (_y - _X @ b), 0, atol=1e-8))
try:
    r2 = r_squared(_y, _X @ b)
except Exception:
    r2 = None
check('R^2 is a Python float between 0 and 1', isinstance(r2, float) and 0 < r2 < 1)
check_hash('hidden: R^2 of the 50-point fit (3 decimals)', r2, 'HASH1')
""".replace("HASH1", h1),
    explanation_md=r"""
The normal equations $X^\top X\hat\beta = X^\top y$ say the residual is orthogonal to every
column of $X$; `np.linalg.solve` solves them without forming an explicit inverse, which is
both faster and more accurate. $R^2$ compares the residual sum of squares with the spread of
$y$ around its mean, so it is the fraction of variance the model explains; with an intercept
in the model it lies in $[0, 1]$. For the five-point data the hand calculation gives
$\hat\beta = (1.3, 0.9)$, $\operatorname{RSS} = 1.9$, $\operatorname{TSS} = 10$. A common
wrong turn is to compute TSS around zero instead of around $\bar y$, which inflates $R^2$.
""",
)

# ---------------------------------------------------------------- Ex 2
SOL2 = r"""
def poly_features(x, degree):
    x = np.asarray(x, dtype=float)
    return np.column_stack([x ** k for k in range(degree + 1)])

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std
"""
run_solution(SOL2)

ex.add(
    title="Polynomial features and standardisation",
    difficulty=1,
    prompt_md=r"""
Implement

* `poly_features(x, degree)` — for a 1-D array `x` of length $n$, the $n\times(\text{degree}+1)$
  design matrix with columns $x^0, x^1, \dots, x^{\text{degree}}$ (in that order). Do not use
  `np.vander` or `PolynomialFeatures`.
* `standardize(X)` — return `(Z, mean, std)` where `Z` has each column of `X` centred and
  divided by its (population, `ddof=0`) standard deviation.

The checks confirm that a degree-9 polynomial interpolates 10 points exactly and that
standardising shrinks the condition number of a polynomial design matrix.
""",
    starter=r"""
def poly_features(x, degree):
    # YOUR CODE HERE
    return None

def standardize(X):
    # YOUR CODE HERE  (return Z, mean, std)
    return None
""",
    solution=SOL2,
    check=r"""
_x = np.array([0.5, 1.0, 2.0])
try:
    Phi = poly_features(_x, 3)
except Exception:
    Phi = None
check('poly_features(x, 3) has shape (3, 4)', Phi is not None and Phi.shape == (3, 4))
check_close('columns are 1, x, x^2, x^3', Phi, np.vander(_x, 4, increasing=True), tol=1e-12)
_x10 = np.linspace(0, 1, 10); _y10 = np.sin(2 * np.pi * _x10)
try:
    _V = poly_features(_x10, 9)
    _c = np.linalg.solve(_V, _y10)
    _interp_err = float(np.abs(_V @ _c - _y10).max())
except Exception:
    _interp_err = None
check('degree 9 interpolates 10 points exactly', _interp_err is not None and _interp_err < 1e-6)
_g = np.random.default_rng(5)
_X = _g.normal(size=(40, 3)) * np.array([1.0, 100.0, 0.01]) + np.array([5.0, -3.0, 0.2])
try:
    Z, mu, sd = standardize(_X)
except Exception:
    Z, mu, sd = None, None, None
check('standardised columns have mean 0', Z is not None and np.allclose(Z.mean(axis=0), 0, atol=1e-10))
check('standardised columns have standard deviation 1 (ddof=0)', Z is not None and np.allclose(Z.std(axis=0), 1, atol=1e-10))
check('mean and std returned correctly', mu is not None and np.allclose(mu, _X.mean(axis=0)) and np.allclose(sd, _X.std(axis=0)))
try:
    _xr = np.linspace(0, 10, 11)
    _cond_raw = np.linalg.cond(poly_features(_xr, 5))
    _cond_std = np.linalg.cond(poly_features(standardize(_xr[:, None])[0].ravel(), 5))
except Exception:
    _cond_raw = _cond_std = None
check('standardising x shrinks the condition number of the degree-5 design by > 100x',
      _cond_raw is not None and _cond_std is not None and _cond_raw > 100 * _cond_std)
""",
    explanation_md=r"""
A list comprehension over the powers, stacked as columns, is all that `poly_features` needs;
the column of ones is $x^0$ and plays the role of the intercept. Because $n$ distinct points
give an invertible $n\times n$ Vandermonde matrix, a degree-$(n-1)$ polynomial fits them
exactly — zero training error, and (Section 3) useless predictions between the points. The
standardisation uses the population standard deviation (`ddof=0`), the convention of
`StandardScaler`; whichever you choose, the same `mean` and `std` must be applied to new data,
which is why the function returns them. The condition-number check shows the numerical reason
to standardise: on $x \in [0, 10]$ the powers $1$ to $x^5$ span five orders of magnitude.
""",
)

# ---------------------------------------------------------------- Ex 3
SOL3 = r"""
def soft_threshold(z, lam):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)
"""
run_solution(SOL3)
_g = np.random.default_rng(8)
_z3 = _g.normal(size=25)
_n_nonzero3 = int((ns["soft_threshold"](_z3, 0.8) != 0).sum())
h3 = sha_of(_n_nonzero3)

ex.add(
    title="The soft-thresholding operator",
    difficulty=1,
    prompt_md=r"""
Implement the soft-thresholding operator, vectorised over a NumPy array `z`:

$$S_\lambda(z) = \operatorname{sign}(z)\,\max(|z| - \lambda,\ 0).$$

`soft_threshold(z, lam)` must work for arrays and for Python scalars, and must return exactly
`0.0` (not a tiny number) wherever $|z| \le \lambda$. This is the lasso solution for an
orthonormal design and the building block of coordinate descent in Exercise 5.
""",
    starter=r"""
def soft_threshold(z, lam):
    # YOUR CODE HERE
    return None
""",
    solution=SOL3,
    check=r"""
try:
    got = soft_threshold(np.array([3.0, -0.5, 1.2]), 1.0)
except Exception:
    got = None
check_close('S_1(3, -0.5, 1.2) = (2, 0, 0.2)', got, np.array([2.0, 0.0, 0.2]), tol=1e-12)
try:
    got0 = soft_threshold(np.array([3.0, -0.5, 1.2]), 0.0)
except Exception:
    got0 = None
check_close('lambda = 0 is the identity', got0, np.array([3.0, -0.5, 1.2]), tol=1e-12)
try:
    gs = soft_threshold(-2.5, 1.0)
except Exception:
    gs = None
check_close('works on a scalar: S_1(-2.5) = -1.5', gs, -1.5, tol=1e-12)
try:
    gz = soft_threshold(np.array([0.3, -0.7]), 0.7)
except Exception:
    gz = None
check('exactly zero inside the threshold', gz is not None and np.all(gz == 0.0))
_g = np.random.default_rng(8)
_z = _g.normal(size=25)
try:
    _k = int((soft_threshold(_z, 0.8) != 0).sum())
except Exception:
    _k = None
check_hash('hidden: number of survivors of S_0.8 on 25 N(0,1) draws', _k, 'HASH3')
""".replace("HASH3", h3),
    explanation_md=r"""
`np.maximum(np.abs(z) - lam, 0.0)` clips at zero and `np.sign(z)` restores the sign, so values
inside $[-\lambda, \lambda]$ map to exactly $0.0$ and everything else moves towards zero by
exactly $\lambda$. Compare ridge in the same orthonormal setting, $z/(1+\lambda)$, which
scales every coefficient and never produces an exact zero. Writing the operator with
`np.where(z > lam, z - lam, np.where(z < -lam, z + lam, 0.0))` is equivalent; the common slip
is `np.sign(z) * (np.abs(z) - lam)` without the clip, which pushes small coefficients *past*
zero to the wrong sign.
""",
)

# ---------------------------------------------------------------- Ex 4
SOL4 = r"""
def ridge_fit(X, y, lam):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)

def ridge_df(X, lam):
    d = np.linalg.svd(X, compute_uv=False)
    return float(np.sum(d ** 2 / (d ** 2 + lam)))
"""
run_solution(SOL4)
_X4, _y4, _ = ns["make_design"]()
_df4 = ns["ridge_df"](_X4, 20.0)
safe_to_hash(_df4)
h4 = sha_of(_df4)

ex.add(
    title="Ridge regression in closed form and its degrees of freedom",
    difficulty=2,
    prompt_md=r"""
With `X` standardised and `y` centred (so no intercept is needed), implement

* `ridge_fit(X, y, lam)` — the ridge coefficients $(X^\top X + \lambda I)^{-1}X^\top y$ via
  `np.linalg.solve`;
* `ridge_df(X, lam)` — the effective degrees of freedom
  $\operatorname{df}(\lambda) = \sum_j d_j^2/(d_j^2 + \lambda)$ from the singular values of `X`
  (`np.linalg.svd(X, compute_uv=False)`), as a Python `float`.

The checks use `make_design()` from the setup cell (six correlated standardised features,
60 rows) and compare with `sklearn.linear_model.Ridge(alpha=lam, fit_intercept=False)` and
with the trace of the ridge hat matrix $X(X^\top X + \lambda I)^{-1}X^\top$.
""",
    starter=r"""
def ridge_fit(X, y, lam):
    # YOUR CODE HERE
    return None

def ridge_df(X, lam):
    # YOUR CODE HERE  (singular values -> sum of d^2 / (d^2 + lam))
    return None
""",
    solution=SOL4,
    check=r"""
_X, _y, _bt = make_design()
try:
    b1 = ridge_fit(_X, _y, 1.0); b10 = ridge_fit(_X, _y, 10.0); b0 = ridge_fit(_X, _y, 0.0)
except Exception:
    b1 = b10 = b0 = None
check_close('lambda = 1 matches sklearn Ridge(alpha=1, fit_intercept=False)', b1, Ridge(alpha=1.0, fit_intercept=False).fit(_X, _y).coef_, tol=1e-8)
check_close('lambda = 10 matches sklearn', b10, Ridge(alpha=10.0, fit_intercept=False).fit(_X, _y).coef_, tol=1e-8)
check_close('lambda = 0 recovers OLS', b0, np.linalg.lstsq(_X, _y, rcond=None)[0], tol=1e-8)
check('coefficient norm shrinks as lambda grows', b0 is not None and np.linalg.norm(b10) < np.linalg.norm(b1) < np.linalg.norm(b0))
try:
    df0 = ridge_df(_X, 0.0); df10 = ridge_df(_X, 10.0); df20 = ridge_df(_X, 20.0)
except Exception:
    df0 = df10 = df20 = None
check_close('df(0) = p = 6', df0, 6.0, tol=1e-9)
_H10 = _X @ np.linalg.solve(_X.T @ _X + 10.0 * np.eye(6), _X.T)
check_close('df(10) equals the trace of the ridge hat matrix', df10, np.trace(_H10), tol=1e-8)
check_hash('hidden: df(20) on make_design() (3 decimals)', df20 if isinstance(df20, float) else None, 'HASH4')
""".replace("HASH4", h4),
    explanation_md=r"""
Adding $\lambda I$ to $X^\top X$ raises every eigenvalue by $\lambda$, so the system is always
solvable — even for $p > n$ — and the solution is shorter than OLS. The SVD makes the effect
visible: $X^\top X + \lambda I = V(D^2 + \lambda I)V^\top$, so the fitted values are
$\hat y = \sum_j \frac{d_j^2}{d_j^2+\lambda}\langle u_j, y\rangle u_j$ and the trace of the hat
matrix is the sum of the filter factors. With $\lambda = 10$ on this design the direction with
$d_6^2 \approx 1.06$ keeps less than a tenth of its weight while the leading directions keep
over 75 %, and the effective degrees of freedom drop from 6 to about 4.4. The common error is
to regularise the intercept, which is why the data here are centred.
""",
)

# ---------------------------------------------------------------- Ex 5
SOL5 = r"""
def lasso_cd(X, y, lam, n_iter=500):
    n, p = X.shape
    beta = np.zeros(p)
    r = y.copy()
    col_sq = (X ** 2).sum(axis=0)
    for _ in range(n_iter):
        for j in range(p):
            r += X[:, j] * beta[j]
            beta[j] = soft_threshold(X[:, j] @ r, lam) / col_sq[j]
            r -= X[:, j] * beta[j]
    return beta
"""
run_solution(SOL5)
_b5 = ns["lasso_cd"](_X4, _y4, 20.0)
_k5 = int((np.abs(_b5) > 1e-10).sum())
_lasso_sk5 = ns["Lasso"](alpha=20.0 / len(_y4), fit_intercept=False, tol=1e-12, max_iter=200_000).fit(_X4, _y4).coef_
assert np.abs(_b5 - _lasso_sk5).max() < 1e-6, "coordinate descent disagrees with sklearn"
assert np.min(np.abs(_b5[np.abs(_b5) > 1e-10])) > 1e-2, "a lasso coefficient is dangerously close to zero"
h5 = sha_of(_k5)

ex.add(
    title="Lasso by coordinate descent",
    difficulty=2,
    prompt_md=r"""
Implement `lasso_cd(X, y, lam, n_iter=500)` that minimises
$\tfrac12\|y - X\beta\|^2 + \lambda\|\beta\|_1$ by cyclic coordinate descent, starting from
$\beta = 0$: for each coordinate $j$ in turn, form the partial residual
$r^{(j)} = y - \sum_{k\ne j}x_k\beta_k$ and set

$$\beta_j \leftarrow \frac{S_\lambda\big(x_j^\top r^{(j)}\big)}{\|x_j\|^2}$$

using your `soft_threshold` from Exercise 3. Keep a running residual rather than recomputing
$X\beta$ from scratch each time. Return the coefficient vector (no intercept; `X` is
standardised and `y` centred).

Convention reminder: scikit-learn's `Lasso(alpha)` minimises $\frac{1}{2n}\|y - X\beta\|^2 +
\alpha\|\beta\|_1$, so your `lam` corresponds to `alpha = lam / n`.
""",
    starter=r"""
def lasso_cd(X, y, lam, n_iter=500):
    n, p = X.shape
    beta = np.zeros(p)
    # YOUR CODE HERE
    return None
""",
    solution=SOL5,
    check=r"""
_X, _y, _bt = make_design()
_n = len(_y)
# orthonormal design: the answer must be the soft-thresholded OLS coefficients
_Q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(30, 4)))
_yq = _Q @ np.array([2.0, -0.5, 1.0, 0.1]) + 0.05 * np.random.default_rng(3).normal(size=30)
_ols_q = _Q.T @ _yq
try:
    bq = lasso_cd(_Q, _yq, 0.4)
except Exception:
    bq = None
check_close('orthonormal design: lasso = soft-threshold of OLS', bq, np.sign(_ols_q) * np.maximum(np.abs(_ols_q) - 0.4, 0), tol=1e-8)
try:
    b5 = lasso_cd(_X, _y, 5.0); b20 = lasso_cd(_X, _y, 20.0); b_big = lasso_cd(_X, _y, 1e4)
except Exception:
    b5 = b20 = b_big = None
check_close('lambda = 5 matches sklearn Lasso(alpha=5/n)', b5, Lasso(alpha=5.0 / _n, fit_intercept=False, tol=1e-12, max_iter=200_000).fit(_X, _y).coef_, tol=1e-5)
check_close('lambda = 20 matches sklearn Lasso(alpha=20/n)', b20, Lasso(alpha=20.0 / _n, fit_intercept=False, tol=1e-12, max_iter=200_000).fit(_X, _y).coef_, tol=1e-5)
check('a huge lambda zeroes every coefficient', b_big is not None and np.all(b_big == 0.0))
check('lambda = 20 produces exact zeros (sparsity)', b20 is not None and np.any(b20 == 0.0))
check_hash('hidden: number of non-zero coefficients at lambda = 20', int((np.abs(b20) > 1e-10).sum()) if b20 is not None else None, 'HASH5')
""".replace("HASH5", h5),
    explanation_md=r"""
Each coordinate update solves the one-dimensional lasso problem exactly: with the other
coefficients fixed, the objective in $\beta_j$ is $\frac12\|r^{(j)} - x_j\beta_j\|^2 +
\lambda|\beta_j|$, whose minimiser is the soft-thresholded least-squares coefficient
$x_j^\top r^{(j)}/\|x_j\|^2$ with threshold $\lambda/\|x_j\|^2$ — equivalently
$S_\lambda(x_j^\top r^{(j)})/\|x_j\|^2$. The objective never increases, and because the
non-smooth part is a sum over coordinates the cycle converges to the global minimum. Keeping a
running residual makes each sweep $O(np)$; recomputing $X\beta$ inside the loop makes it
$O(np^2)$. The exact zeros are not rounding: the operator returns $0.0$ whenever the partial
correlation is below $\lambda$, which is also the KKT condition $|x_j^\top r| \le \lambda$ for
an inactive coefficient.
""",
)

# ---------------------------------------------------------------- Ex 6
SOL6 = r"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_nll(beta, X, y):
    z = X @ beta
    return float(np.sum(np.logaddexp(0.0, z) - y * z))

def logistic_grad(beta, X, y):
    return X.T @ (sigmoid(X @ beta) - y)

def logistic_hessian(beta, X, y):
    p = sigmoid(X @ beta)
    return X.T @ ((p * (1 - p))[:, None] * X)

def newton_step(beta, X, y):
    return beta - np.linalg.solve(logistic_hessian(beta, X, y), logistic_grad(beta, X, y))
"""
run_solution(SOL6)
_x6 = np.array([-2., -1., 0., 1., 2.]); _y6 = np.array([0., 1., 0., 1., 1.])
_X6 = np.column_stack([np.ones(5), _x6])
_b6 = np.zeros(2)
for _ in range(25):
    _b6 = ns["newton_step"](_b6, _X6, _y6)
_boundary6 = float(-_b6[0] / _b6[1])
_nll6 = ns["logistic_nll"](_b6, _X6, _y6)
safe_to_hash(_boundary6); safe_to_hash(_nll6)
h6a = sha_of(_boundary6)
h6b = sha_of(_nll6)

ex.add(
    title="Logistic regression: gradient, Hessian and a Newton step",
    difficulty=2,
    prompt_md=r"""
For the negative log-likelihood of logistic regression,
$\ell(\beta) = \sum_i\big[\log(1 + e^{\beta^\top x_i}) - y_i\,\beta^\top x_i\big]$, implement

* `sigmoid(z)`;
* `logistic_nll(beta, X, y)` — $\ell(\beta)$ as a `float` (use `np.logaddexp(0, z)` for
  $\log(1 + e^z)$ so that large $|z|$ does not overflow);
* `logistic_grad(beta, X, y)` — $\nabla\ell = X^\top(p - y)$ with $p = \sigma(X\beta)$;
* `logistic_hessian(beta, X, y)` — $\nabla^2\ell = X^\top W X$, $W = \operatorname{diag}(p_i(1-p_i))$;
* `newton_step(beta, X, y)` — one Newton update $\beta - (\nabla^2\ell)^{-1}\nabla\ell$.

The checks use the five-point example of the concept guide, $x = (-2,-1,0,1,2)$,
$y = (0,1,0,1,1)$ with an intercept: from $\beta = 0$ the gradient is $(-0.5, -2)$, the Hessian
is $\operatorname{diag}(1.25, 2.5)$ and the first Newton step lands on $(0.4, 0.8)$. Iterating
to convergence gives the MLE $\hat\beta \approx (0.623, 1.090)$.
""",
    starter=r"""
def sigmoid(z):
    # YOUR CODE HERE
    return None

def logistic_nll(beta, X, y):
    # YOUR CODE HERE
    return None

def logistic_grad(beta, X, y):
    # YOUR CODE HERE
    return None

def logistic_hessian(beta, X, y):
    # YOUR CODE HERE
    return None

def newton_step(beta, X, y):
    # YOUR CODE HERE
    return None
""",
    solution=SOL6,
    check=r"""
_x = np.array([-2., -1., 0., 1., 2.]); _y = np.array([0., 1., 0., 1., 1.])
_X = np.column_stack([np.ones(5), _x])
_b0 = np.zeros(2)
try:
    g0 = logistic_grad(_b0, _X, _y); H0 = logistic_hessian(_b0, _X, _y); s1 = newton_step(_b0, _X, _y)
except Exception:
    g0 = H0 = s1 = None
check_close('gradient at beta = 0 is (-0.5, -2.0)', g0, np.array([-0.5, -2.0]), tol=1e-10)
check_close('Hessian at beta = 0 is diag(1.25, 2.5)', H0, np.diag([1.25, 2.5]), tol=1e-10)
check_close('first Newton step gives (0.4, 0.8)', s1, np.array([0.4, 0.8]), tol=1e-10)
# gradient versus finite differences at a random point
_bt = np.array([0.3, -0.7])
try:
    _fd = np.array([(logistic_nll(_bt + 1e-6 * np.eye(2)[k], _X, _y) - logistic_nll(_bt - 1e-6 * np.eye(2)[k], _X, _y)) / 2e-6 for k in range(2)])
    _ga = logistic_grad(_bt, _X, _y)
except Exception:
    _fd, _ga = None, None
check_close('gradient matches finite differences of the NLL', _ga, _fd, tol=1e-5)
try:
    b = np.zeros(2)
    for _ in range(25):
        b = newton_step(b, _X, _y)
    nll_hat = logistic_nll(b, _X, _y)
except Exception:
    b, nll_hat = None, None
check_close('Newton converges to the MLE (0.623, 1.090)', b, np.array([0.6227, 1.0904]), tol=2e-3)
check('the NLL at the MLE is below the NLL at beta = 0 (5 ln 2 = 3.466)', nll_hat is not None and nll_hat < 5 * np.log(2))
check_hash('hidden: decision boundary -beta0/beta1 (3 decimals)', float(-b[0] / b[1]) if b is not None else None, 'HASH6A')
check_hash('hidden: minimised negative log-likelihood (3 decimals)', nll_hat, 'HASH6B')
""".replace("HASH6A", h6a).replace("HASH6B", h6b),
    explanation_md=r"""
Differentiating $\log(1 + e^{z}) - yz$ with respect to $z$ gives $\sigma(z) - y$, and the chain
rule through $z = X\beta$ supplies the $X^\top$: the gradient is $X^\top(p - y)$, the same
"prediction minus target" form as for least squares. Differentiating once more,
$\sigma'(z) = \sigma(z)(1 - \sigma(z))$, gives the weights $W$. At $\beta = 0$ every
$p_i = \tfrac12$, so $W = \tfrac14 I$ and the Hessian is $\tfrac14 X^\top X$ — which is why the
first Newton step is so easy by hand. Newton converges in about six steps here; the
hidden answers are the decision boundary $x = -\hat\beta_0/\hat\beta_1 \approx -0.57$ (where
$p = \tfrac12$) and the minimised NLL. The classic sign error is to return $X^\top(y - p)$,
the gradient of the *log*-likelihood, and then step in the wrong direction.
""",
)

# ---------------------------------------------------------------- Ex 7
SOL7 = r"""
def bias_variance(degree, n_train=30, n_rep=200, sigma=0.3, seed=0):
    g = np.random.default_rng(seed)
    x_grid = np.linspace(0, 1, 101)
    f_grid = np.sin(2 * np.pi * x_grid)
    preds = np.empty((n_rep, len(x_grid)))
    for r in range(n_rep):
        x = g.uniform(0, 1, n_train)
        y = np.sin(2 * np.pi * x) + sigma * g.normal(size=n_train)
        coef = np.polyfit(x, y, degree)
        preds[r] = np.polyval(coef, x_grid)
    bias2 = float(np.mean((preds.mean(axis=0) - f_grid) ** 2))
    variance = float(np.mean(preds.var(axis=0)))
    return bias2, variance
"""
run_solution(SOL7)
_bv = {d: ns["bias_variance"](d) for d in (1, 3, 9)}
assert 0.15 < _bv[1][0] < 0.26 and _bv[1][1] < 0.06
assert _bv[3][0] < 0.03 and 0.005 < _bv[3][1] < 0.05
assert _bv[9][1] > 1.0

ex.add(
    title="Bias^2 and variance by simulation",
    difficulty=2,
    prompt_md=r"""
Measure the bias–variance decomposition for polynomial regression on
$f(x) = \sin(2\pi x)$, $x \sim \mathcal{U}(0,1)$, $y = f(x) + \varepsilon$,
$\varepsilon \sim \mathcal{N}(0, \sigma^2)$.

Implement `bias_variance(degree, n_train=30, n_rep=200, sigma=0.3, seed=0)` that

1. creates `g = np.random.default_rng(seed)`;
2. repeats `n_rep` times: draw a training set of `n_train` points, fit a polynomial of the given
   degree with `np.polyfit`, and store its predictions on the test grid
   `x_grid = np.linspace(0, 1, 101)`;
3. returns `(bias2, variance)` as Python floats, where `bias2` is the mean over the grid of
   $(\bar{\hat f}(x) - f(x))^2$ ($\bar{\hat f}$ = average prediction over the repetitions) and
   `variance` is the mean over the grid of the variance of the predictions across repetitions.

The checks are range checks (the exact values depend on how you draw the random numbers):
degree 1 should be bias-dominated, degree 9 variance-dominated, with the noise
$\sigma^2 = 0.09$ as the third term.
""",
    starter=r"""
def bias_variance(degree, n_train=30, n_rep=200, sigma=0.3, seed=0):
    g = np.random.default_rng(seed)
    x_grid = np.linspace(0, 1, 101)
    f_grid = np.sin(2 * np.pi * x_grid)
    # YOUR CODE HERE  (loop over n_rep training sets, collect predictions on x_grid)
    return None
""",
    solution=SOL7 + r"""

for d in (1, 3, 9):
    b2, v = bias_variance(d)
    print(f"degree {d}: bias^2 = {b2:.4f}, variance = {v:.4f}, noise = 0.09, total = {b2 + v + 0.09:.4f}")
""",
    check=r"""
try:
    b1, v1 = bias_variance(1); b3, v3 = bias_variance(3); b9, v9 = bias_variance(9)
except Exception:
    b1 = v1 = b3 = v3 = b9 = v9 = None
check('returns two Python floats', isinstance(b1, float) and isinstance(v1, float))
check('degree 1: bias^2 is about 0.2 (a line cannot follow a sine)', b1 is not None and 0.15 < b1 < 0.26)
check('degree 1: variance is small (< 0.06)', v1 is not None and v1 < 0.06)
check('degree 3: bias^2 is small (< 0.03)', b3 is not None and b3 < 0.03)
check('degree 3: variance between 0.005 and 0.05', v3 is not None and 0.005 < v3 < 0.05)
check('degree 9: variance explodes (> 1)', v9 is not None and v9 > 1.0)
check('bias^2 decreases with degree, variance increases', None not in (b1, b3, v3, v9) and b1 > b3 and v9 > v3)
""",
    explanation_md=r"""
The decomposition is a statement about the distribution of $\hat f(x)$ over training sets,
so the simulation must hold the test point fixed and vary the training set — which is exactly
what the loop does, with `preds[r]` holding one fitted curve per training set. The column
mean of `preds` estimates $\bar f(x)$, its squared distance to the truth is the bias², and the
column variance is the variance term; adding $\sigma^2 = 0.09$ reproduces the test error. The
straight line has bias² near $0.20$ (the best linear approximation to a full period of a sine
is nearly flat), the cubic reduces it to about $0.005$ at the price of a slightly larger
variance, and degree 9 has variance in the units — the fitted curves swing wildly near the
edges of $[0, 1]$ where the training points are sparse. A common error is to average the
squared errors first and then look for the terms; the bias needs the *average prediction*.
""",
)

# ---------------------------------------------------------------- Ex 8
SOL8 = r"""
def cv_mse_ridge(X, y, lam, folds):
    fold_mse = []
    k = len(folds)
    for f in range(k):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(k) if g != f])
        beta = np.linalg.solve(X[train].T @ X[train] + lam * np.eye(X.shape[1]), X[train].T @ y[train])
        fold_mse.append(np.mean((y[test] - X[test] @ beta) ** 2))
    return float(np.mean(fold_mse))

Xp, yp = make_poly_diabetes()
folds = np.array_split(np.random.default_rng(0).permutation(len(yp)), 5)
lam_grid = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])
cv_errors = np.array([cv_mse_ridge(Xp, yp, lam, folds) for lam in lam_grid])
best_lam = float(lam_grid[np.argmin(cv_errors)])
print("CV MSE per lambda:", np.round(cv_errors, 1))
print("best lambda:", best_lam, " CV RMSE: %.2f" % np.sqrt(cv_errors.min()))
"""
run_solution(SOL8)
_cv8 = ns["cv_errors"]
_order8 = np.sort(_cv8)
assert (_order8[1] - _order8[0]) / _order8[0] > 1e-3, "best lambda is a near tie"
assert 0 < int(np.argmin(_cv8)) < len(_cv8) - 1, "CV minimum is not interior"
h8 = sha_of(float(ns["best_lam"]))

ex.add(
    title="k-fold cross-validation to choose lambda",
    difficulty=2,
    prompt_md=r"""
Implement `cv_mse_ridge(X, y, lam, folds)` that returns the $k$-fold cross-validated mean
squared error (a Python `float`) of ridge regression without intercept: for each fold `f`,
fit $(X_{\text{train}}^\top X_{\text{train}} + \lambda I)^{-1}X_{\text{train}}^\top y_{\text{train}}$ on
the other folds, predict the held-out fold, record its MSE, and average the $k$ fold MSEs.

Then, on the 65 standardised degree-2 features of the diabetes data from
`make_poly_diabetes()` (setup cell), with the five folds and the grid given in the starter
cell, compute `cv_errors` (one CV MSE per grid value) and set `best_lam` to the grid value
with the smallest CV error. The data are standardised and centred globally for simplicity;
the mini-project does it properly inside the folds.
""",
    starter=r"""
def cv_mse_ridge(X, y, lam, folds):
    # YOUR CODE HERE
    return None

Xp, yp = make_poly_diabetes()
folds = np.array_split(np.random.default_rng(0).permutation(len(yp)), 5)   # five index arrays
lam_grid = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])
cv_errors = None   # YOUR CODE HERE  (array with one CV MSE per lambda in lam_grid)
best_lam = None    # YOUR CODE HERE
""",
    solution=SOL8,
    check=r"""
# reference CV error for lambda = 10 computed directly
def _ref_cv(lam):
    out = []
    for f in range(5):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(5) if g != f])
        b = np.linalg.solve(Xp[tr].T @ Xp[tr] + lam * np.eye(Xp.shape[1]), Xp[tr].T @ yp[tr])
        out.append(np.mean((yp[te] - Xp[te] @ b) ** 2))
    return float(np.mean(out))
try:
    got10 = cv_mse_ridge(Xp, yp, 10.0, folds)
except Exception:
    got10 = None
check_close('cv_mse_ridge at lambda = 10 matches the reference', got10, _ref_cv(10.0), tol=1e-6)
check('cv_errors has one entry per grid value', cv_errors is not None and np.shape(cv_errors) == lam_grid.shape)
check('the CV curve is U-shaped (minimum strictly inside the grid)', cv_errors is not None and 0 < int(np.argmin(cv_errors)) < len(lam_grid) - 1)
check('too little regularisation is worse than the best lambda by > 5%', cv_errors is not None and cv_errors[0] > 1.05 * np.min(cv_errors))
check_hash('hidden: the selected lambda', float(best_lam) if best_lam is not None else None, 'HASH8')
""".replace("HASH8", h8),
    explanation_md=r"""
Each fold plays the role of fresh data once; averaging the five held-out errors estimates the
true risk of "ridge with this $\lambda$" far better than the training error, which decreases
monotonically as $\lambda \to 0$. With 65 features and 442 observations the curve is clearly
U-shaped: at small $\lambda$ the many interaction terms fit noise, at very large $\lambda$ the
model collapses towards predicting the mean. The minimiser is $\lambda = 100$ — about 1.5
standard deviations of the target are explained — and the CV RMSE there is roughly 56 against
59 for near-OLS. Two refinements matter in practice: standardising *inside* each training fold
(done in the mini-project), and reporting the error of the chosen $\lambda$ from a *nested*
loop, because the minimum of twelve noisy estimates is optimistic.
""",
)

# ---------------------------------------------------------------- Ex 9 (conceptual)
ex.add_conceptual(
    title="Why standardise before ridge and lasso?",
    difficulty=2,
    prompt_md=r"""
A colleague fits a lasso to predict house prices from *floor area in square metres*,
*distance to the city centre in kilometres* and *number of rooms*, without standardising, and
reports that the lasso "selected" distance and rooms but set the floor-area coefficient to
zero — "so area does not matter once you know the number of rooms". Floor areas run from 40 to
250; distances from 0.5 to 30; rooms from 1 to 7.

1. Explain why the penalty $\lambda\sum_j|\beta_j|$ treats these three coefficients unfairly,
   and in which direction the unfairness goes.
2. Explain what standardising the features changes about the fitted model and about the
   interpretation of the coefficients.
3. The same colleague standardised the *target* as well and penalised the intercept. What goes
   wrong?
""",
    answer_md=r"""
1. The penalty charges the same $\lambda$ per unit of $|\beta_j|$ regardless of the units of
   $x_j$. A feature with a large numerical range (area, up to 250) needs only a *small*
   coefficient to have a large effect on the prediction, so its coefficient is cheap to keep and
   cheap to shrink; a feature with a small range (rooms, 1–7) needs a large coefficient for the
   same effect and is penalised far more. The direction here is subtle: area's coefficient is
   small in absolute terms and so is *easily* driven to exactly zero by soft-thresholding, even
   when its effect on the price is large, while the rooms coefficient survives. The selection
   reflects units, not importance — re-measuring area in square feet would change the outcome.
2. Standardising ($z_j = (x_j - \bar x_j)/s_j$) puts every feature on the scale "one standard
   deviation", so $\lambda$ penalises *effect per standard deviation* equally across features;
   the same penalty then means the same thing for every coefficient. The fitted *predictions*
   of an unpenalised model are unchanged by standardising (OLS is equivariant to rescaling),
   but the penalised fit genuinely changes, and the coefficients become comparable: "one
   standard deviation more area adds this many euros". To report effects in original units,
   divide each standardised coefficient by $s_j$.
3. Standardising the target is harmless in itself (it rescales $\lambda$). Penalising the
   intercept is not: the intercept carries the mean of $y$, and shrinking it towards zero biases
   every prediction downwards (or towards whatever the arbitrary zero of the target is) for no
   statistical benefit — there is no variance to trade for it, since $\bar y$ is estimated from
   all $n$ points. Centre $X$ and $y$ (or use `fit_intercept=True`) so the intercept is $\bar y$
   and is left alone.
""",
)

# ---------------------------------------------------------------- Ex 10 (stretch)
SOL10 = r"""
def poisson_newton(X, y, offset, n_iter=25, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        mu = np.exp(offset + X @ beta)
        step = np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (y - mu))
        beta = beta + step
        if np.abs(step).max() < tol:
            break
    return beta

def poisson_deviance(y, mu):
    terms = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0) - (y - mu)
    return float(2.0 * np.sum(terms))
"""
run_solution(SOL10)
_g = np.random.default_rng(20)
_n10 = 500
_X10 = np.column_stack([np.ones(_n10), _g.normal(size=_n10), _g.binomial(1, 0.4, size=_n10).astype(float)])
_t10 = _g.uniform(0.2, 1.0, size=_n10)
_y10 = _g.poisson(_t10 * np.exp(_X10 @ np.array([-0.3, -0.35, 0.5]))).astype(float)
_b10 = ns["poisson_newton"](_X10, _y10, np.log(_t10))
_pr10 = ns["PoissonRegressor"](alpha=0.0, tol=1e-12, max_iter=10_000).fit(_X10[:, 1:], _y10 / _t10, sample_weight=_t10)
assert np.abs(np.r_[_pr10.intercept_, _pr10.coef_] - _b10).max() < 1e-5
_rr10 = float(np.exp(_b10[2]))
_dev10 = ns["poisson_deviance"](_y10, np.exp(np.log(_t10) + _X10 @ _b10))
safe_to_hash(_rr10); safe_to_hash(_dev10)
h10a = sha_of(_rr10)
h10b = sha_of(_dev10)

ex.add(
    title="Poisson regression with an exposure offset",
    difficulty=3,
    prompt_md=r"""
Counts $y_i$ observed over exposures $t_i$ follow $y_i \sim \text{Poisson}(\mu_i)$ with
$\log\mu_i = \log t_i + \beta^\top x_i$. Implement

* `poisson_newton(X, y, offset, n_iter=25, tol=1e-10)` — Newton's method for the maximum
  likelihood estimate, starting from $\beta = 0$, with $\mu = \exp(\text{offset} + X\beta)$,
  score $X^\top(y - \mu)$ and Hessian (of the negative log-likelihood) $X^\top\operatorname{diag}(\mu)X$;
* `poisson_deviance(y, mu)` — $D = 2\sum_i\big[y_i\log(y_i/\mu_i) - (y_i - \mu_i)\big]$ as a
  `float`, with the convention $0\log 0 = 0$.

The checks generate 500 synthetic insurance policies (intercept, a standardised age score and
an urban indicator; exposures between 0.2 and 1 years) and compare with
`PoissonRegressor(alpha=0)` fitted to the *rates* `y / t` with `sample_weight=t`, which is
the scikit-learn way of expressing an offset.
""",
    starter=r"""
def poisson_newton(X, y, offset, n_iter=25, tol=1e-10):
    beta = np.zeros(X.shape[1])
    # YOUR CODE HERE
    return None

def poisson_deviance(y, mu):
    # YOUR CODE HERE  (remember 0 * log 0 = 0)
    return None
""",
    solution=SOL10,
    check=r"""
_g = np.random.default_rng(20)
_n = 500
_X = np.column_stack([np.ones(_n), _g.normal(size=_n), _g.binomial(1, 0.4, size=_n).astype(float)])
_t = _g.uniform(0.2, 1.0, size=_n)
_y = _g.poisson(_t * np.exp(_X @ np.array([-0.3, -0.35, 0.5]))).astype(float)
try:
    b = poisson_newton(_X, _y, np.log(_t))
except Exception:
    b = None
_pr = PoissonRegressor(alpha=0.0, tol=1e-12, max_iter=10_000).fit(_X[:, 1:], _y / _t, sample_weight=_t)
check_close('Newton matches sklearn PoissonRegressor with the offset expressed as rates + weights', b, np.r_[_pr.intercept_, _pr.coef_], tol=1e-5)
check('score equations hold at the MLE: X^T (y - mu) = 0', b is not None and np.allclose(_X.T @ (_y - np.exp(np.log(_t) + _X @ b)), 0, atol=1e-6))
try:
    dev_toy = poisson_deviance(np.array([2., 0., 3.]), np.array([1.5, 0.5, 2.5]))
    dev_sat = poisson_deviance(np.array([2., 0., 3.]), np.array([2., 1e-9, 3.]))
except Exception:
    dev_toy = dev_sat = None
check_close('deviance of y = (2, 0, 3) against mu = (1.5, 0.5, 2.5) is 1.245', dev_toy, 1.24466, tol=1e-4)
check('a (near-)saturated fit has deviance about 0', dev_sat is not None and abs(dev_sat) < 1e-6)
check_hash('hidden: rate ratio exp(beta_urban) (3 decimals)', float(np.exp(b[2])) if b is not None else None, 'HASH10A')
try:
    dev_fit = poisson_deviance(_y, np.exp(np.log(_t) + _X @ b))
except Exception:
    dev_fit = None
check_hash('hidden: residual deviance of the fitted model (3 decimals)', dev_fit, 'HASH10B')
""".replace("HASH10A", h10a).replace("HASH10B", h10b),
    explanation_md=r"""
With the canonical log link the Poisson log-likelihood is $\sum_i[y_i\eta_i - e^{\eta_i}]$
(plus a constant) with $\eta_i = \log t_i + \beta^\top x_i$; its gradient is
$X^\top(y - \mu)$ and its Hessian $-X^\top\operatorname{diag}(\mu)X$ — the GLM pattern where
$W_{ii} = \operatorname{Var}(y_i) = \mu_i$. Newton therefore converges in a handful of steps
from $\beta = 0$. The offset enters only through $\mu$ and is never multiplied by a
coefficient; the scikit-learn equivalent models the rate $y_i/t_i$ with weight $t_i$, which
gives the same likelihood up to a constant. The deviance compares the fit to the saturated
model that sets $\mu_i = y_i$; a zero count contributes $2\mu_i$ (its $y\log y$ term is zero by
convention), which is why `np.where` is needed to avoid `0 * log(0)`. For well-specified Poisson
data the deviance is close to $n - p$; a much larger value signals overdispersion.
""",
)

# ---------------------------------------------------------------- Ex 11 (stretch)
SOL11 = r"""
def loo_ridge_mse(X, y, lam):
    H = X @ np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T)
    e = y - H @ y
    return float(np.mean((e / (1.0 - np.diag(H))) ** 2))
"""
run_solution(SOL11)
_grid11 = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
_loo11 = np.array([ns["loo_ridge_mse"](_X4, _y4, l) for l in _grid11])
_sorted11 = np.sort(_loo11)
assert (_sorted11[1] - _sorted11[0]) / _sorted11[0] > 1e-3, "LOO-optimal lambda is a near tie"
_best11 = float(_grid11[np.argmin(_loo11)])
_loo_at_3 = ns["loo_ridge_mse"](_X4, _y4, 3.0)
safe_to_hash(_loo_at_3)
h11a = sha_of(_best11)
h11b = sha_of(_loo_at_3)

ex.add(
    title="The leave-one-out shortcut for ridge",
    difficulty=3,
    prompt_md=r"""
For any linear smoother $\hat y = Hy$ the leave-one-out cross-validation error is

$$\operatorname{CV}_{\text{LOO}} = \frac1n\sum_{i=1}^{n}\left(\frac{y_i - \hat y_i}{1 - h_{ii}}\right)^2,$$

with no refitting. Implement `loo_ridge_mse(X, y, lam)` for ridge regression without
intercept (`X` standardised, `y` centred), using $H_\lambda = X(X^\top X + \lambda I)^{-1}X^\top$,
and return the LOO mean squared error as a `float`. The checks compare it with the brute-force
version (60 refits) on `make_design()`, then ask for the LOO-optimal $\lambda$ on the grid
`[0.1, 0.3, 1, 3, 10, 30, 100]`.
""",
    starter=r"""
def loo_ridge_mse(X, y, lam):
    # YOUR CODE HERE  (hat matrix, residuals, divide by 1 - h_ii)
    return None
""",
    solution=SOL11 + r"""

X_, y_, _ = make_design()
grid_ = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
loo_ = np.array([loo_ridge_mse(X_, y_, l) for l in grid_])
print("LOO MSE per lambda:", loo_)
print("LOO-optimal lambda:", grid_[np.argmin(loo_)])
""",
    check=r"""
_X, _y, _bt = make_design()
def _loo_brute(X, y, lam):
    errs = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b = np.linalg.solve(X[m].T @ X[m] + lam * np.eye(X.shape[1]), X[m].T @ y[m])
        errs.append((y[i] - X[i] @ b) ** 2)
    return float(np.mean(errs))
try:
    s1 = loo_ridge_mse(_X, _y, 1.0); s30 = loo_ridge_mse(_X, _y, 30.0)
except Exception:
    s1 = s30 = None
check_close('shortcut equals brute force at lambda = 1', s1, _loo_brute(_X, _y, 1.0), tol=1e-8)
check_close('shortcut equals brute force at lambda = 30', s30, _loo_brute(_X, _y, 30.0), tol=1e-8)
check('returns a Python float', isinstance(s1, float))
_grid = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
try:
    _loo = np.array([loo_ridge_mse(_X, _y, l) for l in _grid])
    _best = float(_grid[np.argmin(_loo)])
    _at3 = loo_ridge_mse(_X, _y, 3.0)
except Exception:
    _best = _at3 = None
check_hash('hidden: LOO-optimal lambda on the grid', _best, 'HASH11A')
check_hash('hidden: LOO MSE at lambda = 3 (3 decimals)', _at3, 'HASH11B')
""".replace("HASH11A", h11a).replace("HASH11B", h11b),
    explanation_md=r"""
Deleting observation $i$ and refitting changes the prediction at $x_i$ by exactly
$e_i h_{ii}/(1 - h_{ii})$, so the leave-one-out residual is $e_i/(1 - h_{ii})$ — a consequence
of the Sherman–Morrison formula for a rank-one update of $(X^\top X + \lambda I)^{-1}$. The
shortcut costs one fit instead of $n$, and for ridge the hat matrix is available from the same
solve that gives the coefficients; `RidgeCV` uses this identity (and its GCV variant with
$h_{ii}$ replaced by $\operatorname{tr}(H)/n$). High-leverage points have their residuals
inflated most, which is why they dominate the LOO error. A frequent slip is to forget that the
formula requires the *same* $H$ to produce $\hat y$ and the $h_{ii}$: mixing the OLS hat matrix
with ridge residuals breaks the identity.
""",
)

# ---------------------------------------------------------------- Ex 12 (project)
PROJECT_DATA = r"""
# ---- diabetes data: 442 patients, 10 baseline features, target = disease progression after one year ----
from sklearn.datasets import load_diabetes
diab = load_diabetes()
X_all, y_all, feature_names = diab.data, diab.target, list(diab.feature_names)
g = np.random.default_rng(0)
perm = g.permutation(len(y_all))
test_idx, train_idx = perm[:89], perm[89:]                    # ~20% held out, touched once at the end
X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
print(f"train {X_train.shape}, test {X_test.shape}; target mean {y_all.mean():.1f}, sd {y_all.std():.1f}")
"""

ex.add_project(
    title="Mini-project: regularised regression on the diabetes data, done honestly",
    difficulty=3,
    prompt_md=r"""
Build an end-to-end regularised regression on the diabetes data (442 patients, 10 features,
target = a measure of disease progression one year after baseline):

1. **Preprocessing inside the folds.** Write a `fit_predict(X_tr, y_tr, X_te, method, lam)`
   helper that standardises the features and centres the target using *training* statistics
   only, fits OLS / ridge (closed form) / lasso (`sklearn.linear_model.Lasso` or your own
   coordinate descent), and returns predictions for `X_te` on the original target scale.
2. **Nested cross-validation.** For ridge and for lasso, run 5-fold *outer* CV on the training
   set; inside each outer training fold, choose $\lambda$ from a logarithmic grid by 5-fold
   *inner* CV using the **one-standard-error rule** (the largest $\lambda$ whose inner-CV error
   is within one standard error of the minimum), then score the outer test fold. Report the
   outer-CV RMSE of OLS, ridge and lasso.
3. **Final models.** Choose $\lambda$ for ridge and lasso by 5-fold CV (same rule) on the whole
   training set, refit, and compare the three coefficient vectors (standardised scale) in a
   bar chart. Which features does the lasso drop? Relate it to the correlation between `s1`
   and `s2`.
4. **Honest test error with uncertainty.** Evaluate the final models on the 89 held-out test
   patients and report the test RMSE with a 95 % bootstrap confidence interval (resample the test
   residuals 2,000 times, Module 10).
""",
    rubric_md=r"""
- Standardisation and centring are fitted on the training part of every split — no test or
  validation statistics leak into any fit.
- The nested loop is genuinely nested: the $\lambda$ scored on an outer fold was chosen without
  that fold.
- OLS, ridge and lasso are compared on the same folds, and you state which differences are
  within the fold-to-fold noise (the three RMSEs are within a unit of each other here, while
  the fold-to-fold standard deviation is about 5).
- The lasso sets some coefficients to exactly zero and you explain the pattern via
  collinearity (`s1`/`s2` are correlated at 0.90 and their OLS coefficients nearly cancel, so
  dropping them costs almost nothing once the other serum measures are in).
- The test RMSE is reported with a bootstrap confidence interval, and you note that an interval
  from 89 patients is wide (roughly ±7 units), so the models cannot be separated on the test set.
- (Bonus) Repeat with all 65 degree-2 polynomial features. Does ridge or lasso gain more from
  regularisation when the design is wider?
""",
    starter=PROJECT_DATA + r"""

# YOUR CODE HERE
# def fit_predict(X_tr, y_tr, X_te, method, lam): ...   (standardise with training stats; method in {"ols", "ridge", "lasso"})
# def cv_fold_mse(X, y, method, lam, folds): ...        (one held-out MSE per fold)
# def choose_lam(X, y, method, grid, folds): ...        (one-standard-error rule)
# nested CV: outer 5 folds; inner 5 folds choose lam from a log grid
# final fits, coefficient bar chart, test RMSE with a bootstrap CI
""",
    solution_sketch_md=r"""
`fit_predict` computes `mu, sd = X_tr.mean(0), X_tr.std(0)` and `ym = y_tr.mean()`, fits on
`(X_tr - mu) / sd` and `y_tr - ym`, and returns `((X_te - mu) / sd) @ beta + ym`. Ridge is
`np.linalg.solve(Z.T @ Z + lam * I, Z.T @ yc)`; for the lasso use
`Lasso(alpha=lam / n, fit_intercept=False)` on the centred data. A `cv_fold_mse` helper over
a list of fold index arrays is reused at both levels: `choose_lam` turns its per-fold MSEs
into a mean and a standard error and applies the one-SE rule; the outer loop calls it on an
inner `np.array_split` of the outer-training indices, then scores the outer test fold with the
chosen $\lambda$. Expect outer-CV RMSEs of about 55 for all three methods — regularisation
helps little on 10 well-chosen features with 353 patients, which is itself a finding (the CV
curve is almost flat up to $\lambda \approx 100$ for ridge and $\lambda \approx 1000$ for the
lasso, which is why the one-SE rule picks large values). The final lasso keeps only `sex`,
`bmi`, `bp`, `s3` and `s5` and drops `age`, `s1`, `s2`, `s4` and `s6`: `s1` and `s2` are
correlated at $0.90$ (VIFs 59 and 39) and their large, opposite-signed OLS coefficients
cancel almost perfectly, so the lasso can zero both at little cost once `s3` and `s5` carry
the serum information. The test RMSE is about 54 with a bootstrap 95 % interval of width
roughly 14 — far wider than the differences between methods.
""",
    solution_code=PROJECT_DATA + r"""

from sklearn.linear_model import Lasso

def fit_predict(X_tr, y_tr, X_te, method, lam=0.0):
    mu, sd, ym = X_tr.mean(axis=0), X_tr.std(axis=0), y_tr.mean()     # training statistics only
    Z, yc = (X_tr - mu) / sd, y_tr - ym
    if method == "ols":
        beta = np.linalg.lstsq(Z, yc, rcond=None)[0]
    elif method == "ridge":
        beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ yc)
    elif method == "lasso":
        beta = Lasso(alpha=lam / len(yc), fit_intercept=False, tol=1e-8, max_iter=50_000).fit(Z, yc).coef_
    return ((X_te - mu) / sd) @ beta + ym, beta

def cv_fold_mse(X, y, method, lam, folds):
    out = []
    for f in range(len(folds)):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(len(folds)) if k != f])
        pred, _ = fit_predict(X[tr], y[tr], X[te], method, lam)
        out.append(np.mean((y[te] - pred) ** 2))
    return np.array(out)                                               # one MSE per fold

def choose_lam(X, y, method, grid, folds):
    # one-standard-error rule: the largest lambda whose CV MSE is within one SE of the minimum
    fold_mse = np.array([cv_fold_mse(X, y, method, lam, folds) for lam in grid])
    mean, se = fold_mse.mean(axis=1), fold_mse.std(axis=1, ddof=1) / np.sqrt(len(folds))
    i_min = int(np.argmin(mean))
    return float(grid[int(np.max(np.where(mean <= mean[i_min] + se[i_min])[0]))])

grids = {"ridge": np.logspace(-1, 3.5, 19), "lasso": np.logspace(-1, 3.5, 19)}   # wide enough that the choice is interior
outer = np.array_split(g.permutation(len(y_train)), 5)
results = {"ols": [], "ridge": [], "lasso": []}
chosen = {"ridge": [], "lasso": []}
for f in range(5):
    te = outer[f]; tr = np.concatenate([outer[k] for k in range(5) if k != f])
    pred, _ = fit_predict(X_train[tr], y_train[tr], X_train[te], "ols")
    results["ols"].append(np.sqrt(np.mean((y_train[te] - pred) ** 2)))
    inner = np.array_split(g.permutation(len(tr)), 5)                  # inner folds *inside* the outer training set
    for method in ("ridge", "lasso"):
        lam = choose_lam(X_train[tr], y_train[tr], method, grids[method], inner)
        chosen[method].append(lam)
        pred, _ = fit_predict(X_train[tr], y_train[tr], X_train[te], method, lam)
        results[method].append(np.sqrt(np.mean((y_train[te] - pred) ** 2)))
print("nested 5-fold CV RMSE (mean +/- sd over outer folds):")
for method, r in results.items():
    extra = "" if method == "ols" else f"   inner-chosen lambdas: {np.round(chosen[method], 2)}"
    print(f"  {method:6s} {np.mean(r):6.2f} +/- {np.std(r):4.2f}{extra}")

# final models: lambda by 5-fold CV on the whole training set, then refit
folds_final = np.array_split(g.permutation(len(y_train)), 5)
lam_final = {m: choose_lam(X_train, y_train, m, grids[m], folds_final) for m in ("ridge", "lasso")}
print("\nfinal lambdas:", lam_final)
coefs, test_pred = {}, {}
for method in ("ols", "ridge", "lasso"):
    test_pred[method], coefs[method] = fit_predict(X_train, y_train, X_test, method, lam_final.get(method, 0.0))
print("\nstandardised coefficients (feature: OLS / ridge / lasso):")
for j, name in enumerate(feature_names):
    print(f"  {name:4s} {coefs['ols'][j]:8.2f} {coefs['ridge'][j]:8.2f} {coefs['lasso'][j]:8.2f}" + ("   <- dropped by lasso" if coefs["lasso"][j] == 0 else ""))
Zs = (X_train - X_train.mean(0)) / X_train.std(0)
print("correlation(s1, s2) on the training set: %.3f" % np.corrcoef(Zs[:, 4], Zs[:, 5])[0, 1])

w = 0.27
plt.bar(np.arange(10) - w, coefs["ols"], w, label="OLS")
plt.bar(np.arange(10), coefs["ridge"], w, label=f"ridge (lambda = {lam_final['ridge']:.1f})")
plt.bar(np.arange(10) + w, coefs["lasso"], w, label=f"lasso (lambda = {lam_final['lasso']:.1f})")
plt.xticks(range(10), feature_names); plt.axhline(0, color="k", lw=0.6)
plt.title("Diabetes data: standardised coefficients of the three final models")
plt.xlabel("feature"); plt.ylabel("coefficient (per standard deviation)"); plt.legend(); plt.show()

# honest test error with a bootstrap confidence interval
gb = np.random.default_rng(1)
print("\nheld-out test RMSE with 95% bootstrap CI (2,000 resamples of the 89 test residuals):")
for method in ("ols", "ridge", "lasso"):
    resid = y_test - test_pred[method]
    rmse = np.sqrt(np.mean(resid ** 2))
    boot = np.array([np.sqrt(np.mean(gb.choice(resid, size=len(resid), replace=True) ** 2)) for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {method:6s} RMSE = {rmse:5.2f}   95% CI [{lo:5.2f}, {hi:5.2f}]")
print("The intervals overlap almost completely: 89 test patients cannot rank these models.")
""",
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print("exercises:", len(ex.items), "| code:", kinds.count("code"), "| concept:", kinds.count("concept"),
      "| project:", kinds.count("project"))
print("difficulties:", [it["difficulty"] for it in ex.items])
