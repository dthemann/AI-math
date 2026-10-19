"""
build_11.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for

    part-4-statistics/11-likelihood-and-bayesian-inference

Run from the repository root:

    python3 _authoring/builders/build_11.py
    python3 tools/run_notebooks.py part-4-statistics/11-likelihood-and-bayesian-inference

All hidden answers are computed here by *executing* the reference solutions and hashing
the result with `sha_of`, so the exercises notebook never contains the answers in clear text.
"""
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import ExerciseSet, Notebook, sha_of  # noqa: E402

import numpy as np  # noqa: E402,F401  (used when executing solution code)

MOD = ROOT / "part-4-statistics/11-likelihood-and-bayesian-inference"

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 11 — Likelihood and Bayesian Inference")

nb.md(r"""
This notebook is the runnable companion to the concept guide: read [README.md](README.md) first, or keep
it open alongside. Each section mirrors a numbered section of the guide. We draw the likelihood curve of a
coin, derive and check the closed-form MLEs, fit a Gamma distribution numerically three different ways,
verify the Fisher-information formula for standard errors by simulation, show by code that minimising MSE
*is* Gaussian maximum likelihood and that cross-entropy *is* the Bernoulli log-likelihood, update Beta and
Normal posteriors, turn a Gaussian prior into ridge regression, compare predictive intervals with plug-in
ones, and finish with a Bayesian A/B test.

Everything is built from scratch in NumPy first; `scipy` and `scikit-learn` appear afterwards to confirm the
hand computation.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy import stats, optimize, special
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
""")

# ---------------------------------------------------------------- 1. likelihood
nb.md(r"""
## 1. The likelihood function

Seven heads in ten flips. Under a Binomial model with head probability $p$,

$$L(p) = \binom{10}{7}p^7(1-p)^3, \qquad \ell(p) = \log L(p).$$

We tabulate $L$ on a grid of $p$, plot both $L$ and $\ell$, and check two claims from the guide: the
curve peaks at $p = 0.7$, and $L$ is **not** a probability distribution over $p$ (its integral is $1/11$,
not 1).
""")

nb.code(r"""
k, n = 7, 10
p_grid = np.linspace(0.001, 0.999, 999)

def binom_lik(p, k, n):
    return comb(n, k) * p**k * (1 - p)**(n - k)

L = binom_lik(p_grid, k, n)
logL = np.log(L)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.plot(p_grid, L); ax1.axvline(k / n, color="r", ls="--", label=f"MLE = k/n = {k/n}")
ax1.set_xlabel("p"); ax1.set_ylabel("L(p)"); ax1.set_title("Likelihood of 7 heads in 10 flips"); ax1.legend()
ax2.plot(p_grid, logL); ax2.axvline(k / n, color="r", ls="--")
ax2.set_ylim(-12, 0); ax2.set_xlabel("p"); ax2.set_ylabel("log L(p)"); ax2.set_title("Log-likelihood (same argmax)")
plt.tight_layout(); plt.show()

for p in (0.5, 0.6, 0.7, 0.8, 0.9):
    print(f"p = {p}:  L = {binom_lik(p, k, n):.4f}   log L = {np.log(binom_lik(p, k, n)):.4f}")
print(f"\nargmax on the grid: p = {p_grid[np.argmax(L)]:.3f}")
print(f"likelihood ratio L(0.7)/L(0.5) = {binom_lik(0.7, k, n)/binom_lik(0.5, k, n):.3f}")
print(f"integral of L over p in [0,1] = {np.trapezoid(L, p_grid):.4f}   (= 1/11 = {1/11:.4f}, so L is not a density in p)")
""")

nb.md(r"""
Both curves peak at the same $p$ — the logarithm is monotone, so maximising $\ell$ and maximising $L$ are
the same problem, and $\ell$ is the one you can differentiate and sum. The integral of $L$ over $p$ is
$1/11$: the likelihood ranks parameter values but assigns no probability to them. The ratio
$L(0.7)/L(0.5) = 2.28$ says the data prefer a 70% coin to a fair one, but only mildly.

**Try it yourself:** change `k, n` to `70, 100` and re-run. How much narrower is the peak, and what is the
likelihood ratio now?
""")

# ---------------------------------------------------------------- 2. analytic MLE
nb.md(r"""
## 2. Analytic maximum likelihood

Setting the score (derivative of $\ell$) to zero gives closed forms: Normal $\hat\mu = \bar x$,
$\hat\sigma^2 = \frac1n\sum(x_i - \bar x)^2$; Poisson $\hat\lambda = \bar x$; Exponential
$\hat\lambda = 1/\bar x$. We compute each by hand on the guide's data and confirm with
`scipy.optimize.minimize` on the negative log-likelihood — the general-purpose route of Section 3.
""")

nb.code(r"""
x_norm = np.array([2.1, 3.4, 1.9, 4.0, 2.6])
calls = np.array([3, 5, 2, 4, 6, 4])
repair = np.array([2.0, 0.5, 1.5, 4.0])

# closed forms
mu_hat, var_hat = x_norm.mean(), x_norm.var()                   # ddof=0: the MLE
print(f"Normal:      mu_hat = {mu_hat:.3f}   sigma^2_hat (MLE, /n) = {var_hat:.4f}   unbiased s^2 (/(n-1)) = {x_norm.var(ddof=1):.4f}")
print(f"Poisson:     lambda_hat = mean = {calls.mean():.3f} calls/hour")
print(f"Exponential: lambda_hat = 1/mean = {1/repair.mean():.3f} per hour;  by invariance P(T > 3h) = exp(-3 lambda_hat) = {np.exp(-3/repair.mean()):.4f}")

# the same answers from a numerical optimiser on the negative log-likelihood
def nll_normal(theta, x):
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)                                    # reparameterise so sigma > 0
    return 0.5 * len(x) * np.log(2 * np.pi * sigma**2) + np.sum((x - mu)**2) / (2 * sigma**2)

res = optimize.minimize(nll_normal, x0=[0.0, 0.0], args=(x_norm,), method="BFGS")
print(f"\nnumeric Normal MLE: mu = {res.x[0]:.4f}, sigma^2 = {np.exp(2*res.x[1]):.4f}   (closed form {mu_hat:.4f}, {var_hat:.4f})")

nll_poisson = lambda log_lam, x: -np.sum(x * log_lam - np.exp(log_lam) - special.gammaln(x + 1))
res_p = optimize.minimize(nll_poisson, x0=[0.0], args=(calls,), method="BFGS")
print(f"numeric Poisson MLE: lambda = {np.exp(res_p.x[0]):.4f}   (closed form {calls.mean():.4f})")

# the bias of the variance MLE, seen by simulation
sims = rng.normal(0, 1, size=(20_000, 5))
print(f"\nE[sigma^2_hat] for n = 5 and sigma^2 = 1: simulated {sims.var(axis=1).mean():.4f}   theory (n-1)/n = {4/5:.4f}")
""")

nb.md(r"""
Optimiser and formula agree to four decimals, which is the point: closed forms are a luxury, the NLL is
always available. Note the variance: the MLE divides by $n$ and averages $0.8\sigma^2$ over repeated samples
of size 5 — biased low by the factor $(n-1)/n$, exactly as the guide derives. Note also the
reparameterisation `log_sigma`: the optimiser explores all of $\mathbb{R}$ while $\sigma$ stays positive.
""")

# ---------------------------------------------------------------- 3. numeric MLE: Gamma
nb.md(r"""
## 3. Numerical maximum likelihood: the Gamma distribution

For Gamma$(\alpha, \beta)$ data, $\hat\beta = \hat\alpha/\bar x$ in closed form, but $\hat\alpha$ solves
$\log\alpha - \psi(\alpha) = \log\bar x - \overline{\log x}$ with $\psi$ the digamma function — no closed
form. Three routes to the same answer: Newton's method on this one-dimensional profile equation, BFGS on the
two-parameter NLL in $(\log\alpha, \log\beta)$, and `scipy.stats.gamma.fit`. Then we draw the
log-likelihood surface.
""")

nb.code(r"""
x_g = np.array([1.2, 2.5, 0.8, 3.1, 1.9])
xbar, s = x_g.mean(), np.log(x_g.mean()) - np.log(x_g).mean()
print(f"xbar = {xbar:.3f}   s = log(xbar) - mean(log x) = {s:.4f}")

# (a) Newton's method on the profile equation f(alpha) = log(alpha) - digamma(alpha) - s = 0
f  = lambda a: np.log(a) - special.digamma(a) - s
fp = lambda a: 1/a - special.polygamma(1, a)
alpha = xbar**2 / x_g.var()                       # method-of-moments start
for it in range(6):
    print(f"  Newton iteration {it}: alpha = {alpha:.6f}   f(alpha) = {f(alpha):+.2e}")
    step = f(alpha) / fp(alpha)
    alpha -= step
    if abs(step) < 1e-10:
        break
alpha_newton, beta_newton = alpha, alpha / xbar
print(f"(a) Newton:        alpha = {alpha_newton:.5f}, beta = {beta_newton:.5f}")

# (b) BFGS on the full two-parameter negative log-likelihood, in log-parameters
def nll_gamma(theta, x):
    a, b = np.exp(theta)
    return -np.sum(stats.gamma.logpdf(x, a, scale=1/b))

res_g = optimize.minimize(nll_gamma, x0=[0.0, 0.0], args=(x_g,), method="BFGS")
alpha_bfgs, beta_bfgs = np.exp(res_g.x)
print(f"(b) BFGS on NLL:   alpha = {alpha_bfgs:.5f}, beta = {beta_bfgs:.5f}   ({res_g.nfev} function evaluations)")

# (c) scipy's built-in fit (floc=0 fixes the location at zero; it returns shape, loc, scale = 1/beta)
a_sp, _, scale_sp = stats.gamma.fit(x_g, floc=0)
print(f"(c) scipy fit:     alpha = {a_sp:.5f}, beta = {1/scale_sp:.5f}")
""")

nb.code(r"""
A, B = np.meshgrid(np.linspace(0.5, 12, 200), np.linspace(0.2, 6, 200))
LL = np.array([[-nll_gamma(np.log([a, b]), x_g) for a in A[0]] for b in B[:, 0]])
LL_max = LL.max()
cs = plt.contour(A, B, LL, levels=LL_max - np.array([20, 10, 5, 3, 2, 1, 0.5, 0.1]), cmap="viridis")
plt.clabel(cs, fmt=lambda v: f"{v - LL_max:.1f}", fontsize=7)
plt.plot(alpha_bfgs, beta_bfgs, "r*", ms=12, label="MLE")
plt.plot(np.linspace(0.5, 12, 50), np.linspace(0.5, 12, 50) / xbar, "k--", lw=1, label=r"profile: $\beta = \alpha/\bar{x}$")
plt.xlabel(r"shape $\alpha$"); plt.ylabel(r"rate $\beta$"); plt.title("Gamma log-likelihood surface (levels = drop below maximum)")
plt.legend(loc="lower right"); plt.show()
""")

nb.md(r"""
Three methods, one summit: $\hat\alpha = 4.620$, $\hat\beta = 2.432$. The contour plot explains why the
one-dimensional Newton route works: the dashed line $\beta = \alpha/\bar x$ is the ridge of the surface, and
the full MLE lies on it. Notice the long diagonal valley — $\alpha$ and $\beta$ are strongly correlated in
the likelihood (a larger shape with a larger rate fits almost as well), so with five data points neither is
pinned down precisely. Section 4 turns that curvature into standard errors.

**Try it yourself:** generate `x_g = rng.gamma(4.62, 1/2.43, size=500)` and re-run — watch the contours
tighten around the truth.
""")

# ---------------------------------------------------------------- 4. Fisher information
nb.md(r"""
## 4. Fisher information and standard errors, checked by simulation

The guide derives $i(p) = 1/(p(1-p))$ for a Bernoulli trial and $i(\lambda) = 1/\lambda$ for a Poisson
count, so the MLE's standard error should be $\operatorname{SE} = 1/\sqrt{n\,i(\theta)}$. We generate 5,000
datasets of each size, compute the MLE on each, and compare the empirical standard deviation of the MLEs with
the formula; then we check that the Wald interval $\hat\theta \pm 1.96\,\operatorname{SE}(\hat\theta)$
covers the truth about 95% of the time.
""")

nb.code(r"""
p_true, lam_true, reps = 0.3, 4.0, 5_000
ns = np.array([10, 30, 100, 300, 1_000])
print(f"{'n':>6s} | {'sd(p_hat)':>10s} {'1/sqrt(n i(p))':>15s} {'Wald coverage':>14s} | {'sd(lam_hat)':>12s} {'sqrt(lam/n)':>12s} {'Wald coverage':>14s}")
sd_p, sd_l = [], []
for n_ in ns:
    p_hat = rng.binomial(n_, p_true, size=reps) / n_
    se_p = np.sqrt(np.clip(p_hat * (1 - p_hat), 1e-12, None) / n_)             # SE evaluated at the estimate
    cover_p = np.mean(np.abs(p_hat - p_true) <= 1.96 * se_p)
    lam_hat = rng.poisson(lam_true, size=(reps, n_)).mean(axis=1)
    se_l = np.sqrt(lam_hat / n_)
    cover_l = np.mean(np.abs(lam_hat - lam_true) <= 1.96 * se_l)
    sd_p.append(p_hat.std()); sd_l.append(lam_hat.std())
    print(f"{n_:6d} | {p_hat.std():10.4f} {np.sqrt(p_true*(1-p_true)/n_):15.4f} {cover_p:14.3f} | {lam_hat.std():12.4f} {np.sqrt(lam_true/n_):12.4f} {cover_l:14.3f}")

plt.loglog(ns, sd_p, "o-", label=r"empirical sd of $\hat p$ (Bernoulli, p = 0.3)")
plt.loglog(ns, np.sqrt(p_true*(1-p_true)/ns), "k--", label=r"$1/\sqrt{n\,i(p)} = \sqrt{p(1-p)/n}$")
plt.loglog(ns, sd_l, "s-", label=r"empirical sd of $\hat\lambda$ (Poisson, $\lambda$ = 4)")
plt.loglog(ns, np.sqrt(lam_true/ns), "k:", label=r"$1/\sqrt{n\,i(\lambda)} = \sqrt{\lambda/n}$")
plt.xlabel("sample size n"); plt.ylabel("standard deviation of the MLE"); plt.title("Fisher information predicts the spread of the MLE")
plt.legend(fontsize=8); plt.show()

# the Cramer-Rao bound is attained exactly by the sample proportion: its variance IS p(1-p)/n
print(f"\nobserved information for 7 heads in 10 at p_hat = 0.7: 7/0.7^2 + 3/0.3^2 = {7/0.49 + 3/0.09:.2f} = n/(p(1-p)) = {10/0.21:.2f}")
print(f"-> SE = 1/sqrt(47.6) = {1/np.sqrt(10/0.21):.4f};  95% Wald interval [{0.7 - 1.96/np.sqrt(10/0.21):.3f}, {0.7 + 1.96/np.sqrt(10/0.21):.3f}]")
""")

nb.md(r"""
The empirical standard deviations sit on the dashed theory lines at every $n$, with the familiar slope
$-1/2$ on log–log axes. Coverage of the Wald interval is close to 95% once $n$ is in the hundreds; at
$n = 10$ it is visibly off (the normal approximation to $\hat p$ is poor, and $\hat p = 0$ gives a
zero-width interval), which is why small-sample intervals need the exact Beta posterior of Section 6 or
the methods of [Module 12](../12-hypothesis-testing-and-experiments/README.md).
""")

# ---------------------------------------------------------------- 5. losses
nb.md(r"""
## 5. MLE gives you your loss functions

Two demonstrations. First, for a linear model $y = wx + \varepsilon$ with Gaussian noise, the Gaussian
negative log-likelihood and the mean squared error are different functions of $w$ — but they have the **same
argmin**, because the NLL is an affine function of the SSE. Second, the binary cross-entropy of a
classifier is exactly $-\frac1n\sum_i \log p(y_i \mid p_i)$ under a Bernoulli model.
""")

nb.code(r"""
n_pts = 50
x_lin = rng.uniform(-2, 2, size=n_pts)
y_lin = 1.5 * x_lin + rng.normal(0, 0.8, size=n_pts)

w_grid = np.linspace(0, 3, 601)
sse = np.array([np.sum((y_lin - w * x_lin)**2) for w in w_grid])
mse = sse / n_pts
sigma2 = 0.8**2
nll_gauss = 0.5 * n_pts * np.log(2 * np.pi * sigma2) + sse / (2 * sigma2)

fig, ax1 = plt.subplots()
ax1.plot(w_grid, mse, "C0", label="MSE(w)"); ax1.set_xlabel("slope w"); ax1.set_ylabel("MSE", color="C0")
ax2 = ax1.twinx(); ax2.plot(w_grid, nll_gauss, "C3", label="Gaussian NLL(w)"); ax2.set_ylabel("negative log-likelihood", color="C3")
for ax, arr, c in ((ax1, mse, "C0"), (ax2, nll_gauss, "C3")):
    ax.axvline(w_grid[np.argmin(arr)], color=c, ls="--", lw=1)
plt.title("Different curves, identical argmin: least squares is Gaussian MLE"); fig.tight_layout(); plt.show()

w_mse, w_nll = w_grid[np.argmin(mse)], w_grid[np.argmin(nll_gauss)]
w_closed = (x_lin @ y_lin) / (x_lin @ x_lin)
print(f"argmin MSE = {w_mse:.3f}   argmin Gaussian NLL = {w_nll:.3f}   closed-form least squares = {w_closed:.4f}")
print(f"NLL = {0.5*n_pts*np.log(2*np.pi*sigma2):.3f} + SSE/{2*sigma2:.2f}  -> an affine function of SSE, so the minimiser cannot differ")

# Laplace noise -> MAE, and the robustness difference on an outlier
r = np.array([0.5, -1.0, 0.2, 2.0, -0.3]); r_out = r.copy(); r_out[3] = 4.0
print(f"\nresiduals {r}: SSE = {np.sum(r**2):.2f}, MAE = {np.mean(np.abs(r)):.2f}")
print(f"one residual 2.0 -> 4.0:   SSE = {np.sum(r_out**2):.2f} (x{np.sum(r_out**2)/np.sum(r**2):.1f}),  MAE = {np.mean(np.abs(r_out)):.2f} (x{np.mean(np.abs(r_out))/np.mean(np.abs(r)):.1f})")
""")

nb.code(r"""
# binary cross-entropy IS the negative mean Bernoulli log-likelihood
def bce(y, p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def neg_mean_bernoulli_loglik(y, p):
    lik_each = p**y * (1 - p)**(1 - y)            # p(y_i | p_i) for each example
    return -np.mean(np.log(lik_each))

y_demo = np.array([1, 0, 1, 1, 0]); p_demo = np.array([0.9, 0.2, 0.6, 0.3, 0.1])
print(f"BCE = {bce(y_demo, p_demo):.5f}   -mean log Bernoulli likelihood = {neg_mean_bernoulli_loglik(y_demo, p_demo):.5f}")
from sklearn.metrics import log_loss
print(f"sklearn log_loss = {log_loss(y_demo, p_demo):.5f}")
print(f"\ncost of assigning 0.7 to the true class: {-np.log(0.7):.3f} nats;  0.1: {-np.log(0.1):.3f} nats;  0.001: {-np.log(0.001):.3f} nats (confident mistakes are punished hardest)")

# logistic regression = numerical MLE of the Bernoulli model, trained by gradient ascent on the log-likelihood
X_cls = rng.normal(size=(200, 2))
y_cls = (X_cls @ np.array([1.5, -1.0]) + 0.5 + rng.normal(scale=1.0, size=200) > 0).astype(float)
Xb = np.c_[np.ones(200), X_cls]                       # bias column
sigmoid = lambda z: 1 / (1 + np.exp(-z))

w = np.zeros(3); eta = 0.1; history = []
for step in range(2_000):
    p = sigmoid(Xb @ w)
    grad_ll = Xb.T @ (y_cls - p)                      # d(log-likelihood)/dw = X^T (y - p)   (Module 04/05)
    w += eta * grad_ll / 200
    history.append(np.sum(y_cls * np.log(p) + (1 - y_cls) * np.log(1 - p)))

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=np.inf, max_iter=1000).fit(X_cls, y_cls)    # C = inf: no penalty, pure MLE
print(f"\ngradient ascent: bias {w[0]:.4f}, weights {w[1:]}")
print(f"sklearn (unpenalised): bias {clf.intercept_[0]:.4f}, weights {clf.coef_[0]}")
plt.plot(history); plt.xlabel("gradient-ascent step"); plt.ylabel("log-likelihood of the labels")
plt.title("Logistic regression: climbing the Bernoulli log-likelihood"); plt.show()
""")

nb.md(r"""
The BCE and the negative mean log-likelihood agree to every digit (and so does `log_loss`): they are the
same formula. Gradient ascent on $\ell(w)$ with the gradient $X^\top(y - p)$ converges to the same weights
scikit-learn finds, because `LogisticRegression` is a numerical MLE too. The outlier experiment shows the
practical content of the Gaussian assumption — one residual doubled multiplies the SSE by 3.2 but the MAE
by only 1.5.

**Try it yourself:** make the classes separable (`y_cls = (X_cls @ np.array([1.5, -1.0]) > 0)`, no noise)
and re-run the gradient-ascent cell: the weights keep growing and the log-likelihood keeps creeping toward
zero — the MLE does not exist.
""")

# ---------------------------------------------------------------- 6. Bayesian inference
nb.md(r"""
## 6. Bayesian inference for a click-through rate

Two clicks in forty impressions. First with the guide's three-point prior
$P(p = 0.02, 0.05, 0.10) = (0.25, 0.5, 0.25)$: posterior $\propto$ prior $\times$ likelihood, normalised by
the evidence. Then with a uniform prior on a fine grid of $p$ — the same arithmetic with 1,000 hypotheses —
which we compare with the exact Beta$(3, 39)$ posterior and with the Wald interval.
""")

nb.code(r"""
k_c, n_c = 2, 40
p_vals = np.array([0.02, 0.05, 0.10]); prior3 = np.array([0.25, 0.50, 0.25])
lik3 = comb(n_c, k_c) * p_vals**k_c * (1 - p_vals)**(n_c - k_c)
joint3 = prior3 * lik3
evidence3 = joint3.sum()
post3 = joint3 / evidence3
print("three-point prior")
print(f"  likelihoods     {lik3.round(4)}")
print(f"  prior x lik     {joint3.round(4)}   evidence p(D) = {evidence3:.4f}")
print(f"  posterior       {post3.round(3)}   posterior mean = {(post3 * p_vals).sum():.4f}")

# grid posterior with a uniform prior (1,000 hypotheses) vs the exact Beta(3, 39)
grid = np.linspace(0.0005, 0.9995, 1000)
prior_g = np.ones_like(grid) / len(grid)
lik_g = comb(n_c, k_c) * grid**k_c * (1 - grid)**(n_c - k_c)
post_g = prior_g * lik_g; post_g /= post_g.sum()
post_exact = stats.beta(k_c + 1, n_c - k_c + 1)
cdf_g = np.cumsum(post_g)
ci_grid = (grid[np.searchsorted(cdf_g, 0.025)], grid[np.searchsorted(cdf_g, 0.975)])
se_wald = np.sqrt((k_c/n_c) * (1 - k_c/n_c) / n_c)

plt.plot(grid, post_g * len(grid), label="grid posterior (uniform prior)", lw=3, alpha=0.5)
plt.plot(grid, post_exact.pdf(grid), "k--", label="exact Beta(3, 39)")
plt.axvline(k_c / n_c, color="r", ls=":", label=f"MLE = {k_c/n_c}")
plt.axvspan(*post_exact.ppf([0.025, 0.975]), alpha=0.15, color="C0", label="95% credible interval")
plt.xlim(0, 0.3); plt.xlabel("click-through rate p"); plt.ylabel("posterior density"); plt.title("Posterior for p after 2 clicks in 40 impressions"); plt.legend(fontsize=8); plt.show()

print(f"\ngrid: mean {np.sum(grid * post_g):.4f}, 95% CI [{ci_grid[0]:.4f}, {ci_grid[1]:.4f}]")
print(f"Beta(3,39): mean {post_exact.mean():.4f}, median {post_exact.median():.4f}, mode {2/40:.4f}, 95% CI {post_exact.ppf([0.025, 0.975]).round(4)}")
print(f"P(p > 0.10 | data) = {1 - post_exact.cdf(0.10):.4f}     P(p > 0.04 | data) = {1 - post_exact.cdf(0.04):.4f}")
print(f"Wald interval: {k_c/n_c:.3f} +/- 1.96 x {se_wald:.4f} = [{k_c/n_c - 1.96*se_wald:.4f}, {k_c/n_c + 1.96*se_wald:.4f}]   <- includes negative rates")
""")

nb.md(r"""
The grid posterior and the exact Beta coincide — Bayes' theorem on a fine grid *is* the integral, done
numerically, and it works for any prior you can tabulate. The posterior is skewed: mode $0.05$ (the MLE),
median $0.065$, mean $0.071$, and a 21% probability that the true rate exceeds 10%. The Wald interval
$[-0.018, 0.118]$ is a reminder that the asymptotics of Section 4 have not arrived at $n = 40$, while the
posterior is exact at any $n$.
""")

# ---------------------------------------------------------------- 7. conjugacy
nb.md(r"""
## 7. Conjugate updating: Beta–Binomial and Normal–Normal

A Beta$(a, b)$ prior updated with $k$ successes in $n$ trials gives Beta$(a + k, b + n - k)$, whose mean is
a weighted average of the prior mean and $k/n$ with data weight $n/(a + b + n)$. We start from the sceptical
prior Beta$(2, 8)$ (mean 0.2, worth 10 pseudo-flips) and feed in flips from a coin with $p = 0.7$ —
snapshots after 0, 5, 20 and 100 flips — then verify that updating one flip at a time ends at the same
posterior as one batch update. Finally the Normal–Normal case: precisions add, means are precision-weighted.
""")

nb.code(r"""
a0, b0, p_coin = 2, 8, 0.7
flips = rng.random(100) < p_coin
p_axis = np.linspace(0, 1, 500)

fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
for ax, m in zip(axes, (0, 5, 20, 100)):
    k_m = flips[:m].sum()
    a_m, b_m = a0 + k_m, b0 + m - k_m
    ax.plot(p_axis, stats.beta.pdf(p_axis, a_m, b_m))
    ax.axvline(p_coin, color="r", ls="--", lw=1)
    ax.fill_between(p_axis, 0, stats.beta.pdf(p_axis, a_m, b_m), where=(p_axis > stats.beta.ppf(0.025, a_m, b_m)) & (p_axis < stats.beta.ppf(0.975, a_m, b_m)), alpha=0.2)
    ax.set_title(f"after {m} flips: Beta({a_m}, {b_m})", fontsize=9); ax.set_xlabel("p")
    w_data = m / (a0 + b0 + m)
    data_part = f"{w_data:.3f} x {k_m/m:.3f} (data)" if m else "no data yet"
    print(f"after {m:3d} flips ({k_m:2d} heads): posterior mean {a_m/(a_m+b_m):.4f} = {data_part} + {1-w_data:.3f} x {a0/(a0+b0):.1f} (prior);"
          f"  95% CI [{stats.beta.ppf(0.025, a_m, b_m):.3f}, {stats.beta.ppf(0.975, a_m, b_m):.3f}]")
axes[0].set_ylabel("posterior density"); fig.suptitle("Beta-Binomial updating from a Beta(2, 8) prior; red line = true p = 0.7", y=1.03)
plt.tight_layout(); plt.show()

# sequential == batch
a_seq, b_seq = a0, b0
for f_ in flips:
    a_seq, b_seq = a_seq + int(f_), b_seq + 1 - int(f_)
print(f"\nsequential (100 single-flip updates): Beta({a_seq}, {b_seq});  batch: Beta({a0 + flips.sum()}, {b0 + 100 - flips.sum()})  -> identical")
""")

nb.code(r"""
# Normal-Normal with known sensor noise: prior N(100, 10^2), four readings with sigma = 4 averaging 108
mu0, tau2, sigma2_s, n_r, xbar_r = 100.0, 10.0**2, 4.0**2, 4, 108.0
prec_prior, prec_data = 1/tau2, n_r/sigma2_s
prec_post = prec_prior + prec_data
mu_post = (prec_prior * mu0 + prec_data * xbar_r) / prec_post
print(f"precisions: prior {prec_prior:.4f} + data {prec_data:.4f} = posterior {prec_post:.4f}  ->  posterior sd {np.sqrt(1/prec_post):.3f}")
print(f"posterior mean = {mu_post:.3f}   (data weight {prec_data/prec_post:.3f})")

t_axis = np.linspace(80, 125, 600)
plt.plot(t_axis, stats.norm.pdf(t_axis, mu0, np.sqrt(tau2)), label="prior N(100, 10^2)")
plt.plot(t_axis, stats.norm.pdf(t_axis, xbar_r, np.sqrt(sigma2_s/n_r)), label=r"likelihood of $\bar x$ = 108 (sd 4/$\sqrt{4}$ = 2)")
plt.plot(t_axis, stats.norm.pdf(t_axis, mu_post, np.sqrt(1/prec_post)), "k", lw=2, label=f"posterior N({mu_post:.1f}, {np.sqrt(1/prec_post):.2f}^2)")
plt.xlabel("furnace temperature"); plt.ylabel("density"); plt.title("Normal-Normal: the posterior is a precision-weighted compromise"); plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
The four panels tell the shrinkage story: the prior pulls the first few estimates toward 0.2, but by 100
flips the data carry $100/110 = 91\%$ of the weight and the posterior sits on the true $p = 0.7$ with a
narrow 95% band. Sequential and batch updates land on the same Beta — the likelihood factorises, so the
order of arrival is irrelevant. In the Normal case the posterior is narrower than *both* the prior and the
likelihood (precisions add), and its mean is almost the data mean because the sensor is far more precise
than the prior.

**Try it yourself:** change the prior to Beta$(20, 80)$ (same mean, ten times the pseudo-count) — how many
flips does it take before the posterior mean passes 0.5?
""")

# ---------------------------------------------------------------- 8. MAP = ridge
nb.md(r"""
## 8. MAP with a Gaussian prior is ridge regression

Polynomial regression of degree 9 on 15 noisy points — a recipe for overfitting under maximum likelihood.
Put a Gaussian prior $w_j \sim \mathcal{N}(0, \tau^2)$ on the coefficients. The MAP estimate minimises
$\|y - Xw\|^2 + \lambda\|w\|^2$ with $\lambda = \sigma^2/\tau^2$, and we confirm that our closed-form
$(X^\top X + \lambda I)^{-1}X^\top y$ equals `sklearn.linear_model.Ridge(alpha=λ)` as $\tau$ varies.
""")

nb.code(r"""
from sklearn.linear_model import Ridge

n_poly, degree, sigma_noise = 15, 9, 0.3
x_p = np.sort(rng.uniform(-1, 1, size=n_poly))
f_true = lambda x: np.sin(3 * x)
y_p = f_true(x_p) + rng.normal(0, sigma_noise, size=n_poly)
Phi = np.vander(x_p, degree + 1, increasing=True)[:, 1:]        # x, x^2, ..., x^9 (intercept handled separately)
xx = np.linspace(-1, 1, 300); Phi_xx = np.vander(xx, degree + 1, increasing=True)[:, 1:]

def map_fit(Phi, y, lam):
    # centre so that the intercept is unpenalised (matches Ridge(fit_intercept=True))
    Pm, ym = Phi.mean(axis=0), y.mean()
    Pc, yc = Phi - Pm, y - ym
    w = np.linalg.solve(Pc.T @ Pc + lam * np.eye(Phi.shape[1]), Pc.T @ yc)
    return w, ym - Pm @ w

plt.scatter(x_p, y_p, color="k", zorder=3, label="data"); plt.plot(xx, f_true(xx), "k:", label="true function")
print(f"{'tau':>8s} {'lambda = sigma^2/tau^2':>22s} {'||w||':>8s} {'max |w_ours - w_sklearn|':>26s}")
for tau in (1e3, 1.0, 0.1, 0.01):
    lam = sigma_noise**2 / tau**2
    w_map, b_map = map_fit(Phi, y_p, lam)
    ridge = Ridge(alpha=lam).fit(Phi, y_p)
    plt.plot(xx, Phi_xx @ w_map + b_map, label=f"tau = {tau:g}  (lambda = {lam:.2g})")
    print(f"{tau:8g} {lam:22.3g} {np.linalg.norm(w_map):8.2f} {np.max(np.abs(w_map - ridge.coef_)):26.2e}")
plt.ylim(-2, 2); plt.xlabel("x"); plt.ylabel("y"); plt.title("Degree-9 polynomial: MAP under N(0, tau^2) priors = ridge"); plt.legend(fontsize=7); plt.show()
""")

nb.md(r"""
With $\tau = 1000$ the prior is effectively flat, $\lambda \approx 10^{-7}$, and the MAP is the MLE: a
wiggly degree-9 curve that threads the noise, with coefficients in the hundreds. Tightening $\tau$ to
$0.1$ gives $\lambda = 9$ and a smooth fit close to the truth; $\tau = 0.01$ over-shrinks toward a flat
line. The last column shows our normal-equations solution matching scikit-learn's `Ridge` to machine
precision for every $\lambda$ — the regulariser *is* the prior.

**Try it yourself:** multiply the data fivefold (`n_poly = 75`) and keep $\tau = 0.1$. The same prior now
shrinks far less, because the data term has five times as many summands.
""")

# ---------------------------------------------------------------- 9. predictive
nb.md(r"""
## 9. Posterior predictive versus plug-in prediction

With the Beta$(3, 39)$ posterior from Section 6, how many clicks will the next 100 impressions bring? The
plug-in answer is Binomial$(100, \hat p)$ with $\hat p$ the posterior mean. The honest answer integrates
over $p$: a Beta-Binomial distribution, which we obtain both from `scipy.stats.betabinom` and by brute
force — sample $p$ from the posterior, then sample clicks given $p$. We also revisit the zero-failure
problem.
""")

nb.code(r"""
a_post, b_post, m_future = 3, 39, 100
p_bar = a_post / (a_post + b_post)
plug_in = stats.binom(m_future, p_bar)
predictive = stats.betabinom(m_future, a_post, b_post)

# the same predictive by simulation: draw p from the posterior, then clicks given p
p_draws = rng.beta(a_post, b_post, size=200_000)
clicks_sim = rng.binomial(m_future, p_draws)

ks = np.arange(0, 31)
plt.bar(ks - 0.2, plug_in.pmf(ks), width=0.4, label=f"plug-in Binomial(100, {p_bar:.3f})")
plt.bar(ks + 0.2, predictive.pmf(ks), width=0.4, label="posterior predictive (Beta-Binomial)")
plt.plot(ks, np.bincount(clicks_sim, minlength=200)[:31] / len(clicks_sim), "k.", label="simulated predictive")
plt.xlabel("clicks in the next 100 impressions"); plt.ylabel("probability"); plt.title("Integrating over p widens the prediction"); plt.legend(fontsize=8); plt.show()

print(f"plug-in:    mean {plug_in.mean():.2f}, sd {plug_in.std():.2f}, 95% interval {plug_in.ppf([0.025, 0.975]).astype(int)}")
print(f"predictive: mean {predictive.mean():.2f}, sd {predictive.std():.2f}, 95% interval {predictive.ppf([0.025, 0.975]).astype(int)}")
print(f"simulated:  mean {clicks_sim.mean():.2f}, sd {clicks_sim.std():.2f}, 95% interval {np.percentile(clicks_sim, [2.5, 97.5]).astype(int)}")
print(f"variance inflation factor (a+b+m)/(a+b+1) = {(a_post + b_post + m_future)/(a_post + b_post + 1):.3f}")

# zero failures in 20 tests: plug-in says failure is impossible, the posterior predictive does not
print("\nzero failures in 20 stress tests, uniform prior -> posterior Beta(1, 21)")
print(f"  P(next fails)              rule of succession (k+1)/(n+2) = {1/22:.4f}      plug-in MLE = 0")
print(f"  P(no failure in next 10)   predictive = B(1,31)/B(1,21) = {21/31:.4f}        plug-in = 1")
print(f"  95% upper credible bound for the failure rate: {stats.beta.ppf(0.95, 1, 21):.4f}   (rule of three 3/n = {3/20:.3f})")
""")

nb.md(r"""
The predictive distribution has the same mean as the plug-in but a standard deviation of 4.7 instead of
2.6, and a 95% interval of 1–18 clicks instead of 3–13: the plug-in interval is less than half as wide as it
should be, because it pretends $p$ is known. The simulated histogram (dots) sits on the Beta-Binomial bars —
"sample the parameter, then sample the data" is the posterior predictive integral done by Monte Carlo, and
it works for any model, conjugate or not.
""")

# ---------------------------------------------------------------- 10. A/B
nb.md(r"""
## 10. Bayesian A/B testing

A: 120 conversions in 2,400 visitors; B: 150 in 2,400. With uniform priors the posteriors are
Beta$(121, 2281)$ and Beta$(151, 2251)$. We estimate $P(p_B > p_A \mid \mathcal{D})$ by Monte Carlo, compute
the expected loss of each decision, and put the frequentist two-proportion $z$-test next to it to see how the
two answers differ in kind.
""")

nb.code(r"""
kA, nA, kB, nB = 120, 2400, 150, 2400
postA, postB = stats.beta(1 + kA, 1 + nA - kA), stats.beta(1 + kB, 1 + nB - kB)

S = 400_000
pA, pB = postA.rvs(S, random_state=rng), postB.rvs(S, random_state=rng)
prob_B_better = np.mean(pB > pA)
loss_choose_B = np.mean(np.maximum(pA - pB, 0))     # conversion forgone if B is actually worse
loss_choose_A = np.mean(np.maximum(pB - pA, 0))

print(f"posterior means: A {postA.mean():.4f}, B {postB.mean():.4f};  posterior sds: A {postA.std():.4f}, B {postB.std():.4f}")
print(f"P(B > A | data) = {prob_B_better:.4f}  (Monte Carlo SE {np.sqrt(prob_B_better*(1-prob_B_better)/S):.4f})")
print(f"expected loss if we ship B: {loss_choose_B:.5f}   if we keep A: {loss_choose_A:.5f}   (ratio {loss_choose_A/loss_choose_B:.0f}x)")
print(f"95% credible interval for the lift p_B - p_A: [{np.percentile(pB - pA, 2.5):.4f}, {np.percentile(pB - pA, 97.5):.4f}]")

# frequentist two-proportion z-test for comparison (Module 12)
p_pool = (kA + kB) / (nA + nB)
z = (kB/nB - kA/nA) / np.sqrt(p_pool * (1 - p_pool) * (1/nA + 1/nB))
print(f"\nfrequentist: z = {z:.3f}, two-sided p-value = {2 * (1 - stats.norm.cdf(z)):.4f}  -> 'not significant at 5%'")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
grid_ab = np.linspace(0.03, 0.09, 500)
ax1.plot(grid_ab, postA.pdf(grid_ab), label="A: Beta(121, 2281)"); ax1.plot(grid_ab, postB.pdf(grid_ab), label="B: Beta(151, 2251)")
ax1.set_xlabel("conversion rate"); ax1.set_ylabel("posterior density"); ax1.set_title("Two posteriors"); ax1.legend()
ax2.hist(pB - pA, bins=100, density=True, alpha=0.7); ax2.axvline(0, color="k", ls="--")
ax2.set_xlabel("p_B - p_A"); ax2.set_ylabel("density"); ax2.set_title(f"Posterior of the lift: P(B > A) = {prob_B_better:.3f}")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Both analyses are correct and they answer different questions. The posterior says: B is better with
probability 0.97, and the expected cost of shipping it if that is wrong is 0.008 percentage points, roughly
160 times smaller than the expected cost of keeping A. The $z$-test says the data would be this extreme 6% of
the time if the rates were equal — not below 5%, so by that rule you would not reject equality. A decision
maker who needs to choose *now* wants the first pair of numbers; [Module 12](../12-hypothesis-testing-and-experiments/README.md)
explains when and why the second matters.

**Try it yourself:** multiply all four counts by 4 (same rates, four times the traffic). What happens to
$P(B > A)$, to the expected loss, and to the $p$-value?
""")

nb.md(r"""
## Summary

- The likelihood $L(\theta) = \prod_i p(x_i \mid \theta)$ ranks parameters by how well they explain the data; it is not a distribution over $\theta$ (the coin example integrates to $1/11$).
- Closed-form MLEs: $\hat p = k/n$, $\hat\lambda = \bar x$ (Poisson), $\hat\lambda = 1/\bar x$ (Exponential), $\hat\mu = \bar x$, $\hat\sigma^2 = \frac1n\sum(x_i - \bar x)^2$ — biased by $(n-1)/n$, confirmed by simulation.
- Without a closed form (Gamma shape, logistic regression), minimise the NLL numerically; reparameterise to respect constraints; Newton, BFGS and `scipy.stats.*.fit` agree.
- The Fisher information predicts the spread of the MLE: empirical standard deviations match $1/\sqrt{n\,i(\theta)}$ and Wald intervals cover ~95% for moderate $n$.
- MSE and the Gaussian NLL have the same argmin; BCE equals the negative mean Bernoulli log-likelihood; gradient ascent on $\ell$ reproduces `LogisticRegression`.
- Posterior $\propto$ prior $\times$ likelihood, computable on a grid for any prior; conjugate pairs make it arithmetic on pseudo-counts, and sequential = batch.
- MAP with a Gaussian prior is ridge with $\lambda = \sigma^2/\tau^2$, verified against `Ridge` for several $\tau$.
- The posterior predictive is wider than the plug-in prediction (variance factor $(a+b+m)/(a+b+1)$) and never predicts "impossible" after zero failures.
- Bayesian A/B testing gives $P(B > A \mid \mathcal{D})$ and expected losses — a different question from the $p$-value, answered on the same data.

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
from math import comb
from scipy import stats, optimize, special
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
    "Module 11 — Exercises",
    intro_md=r"""
These exercises follow the order of the concept guide: the likelihood and analytic MLEs, Fisher information,
numerical MLE, loss functions as negative log-likelihoods, conjugate posteriors, MAP as regularisation, the
posterior predictive, Bayesian A/B testing, and finally empirical-Bayes shrinkage. Several checks compare
your answer against a *hidden* value, so derive carefully rather than guessing.
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- 1. Bernoulli log-likelihood (warm-up)
SOL_1 = r"""
def bernoulli_loglik(p, k, n):
    p = np.asarray(p, dtype=float)
    return k * np.log(p) + (n - k) * np.log(1 - p)

def mle_bernoulli(k, n):
    return k / n

def log_lik_ratio(p1, p2, k, n):
    return float(bernoulli_loglik(p1, k, n) - bernoulli_loglik(p2, k, n))
"""
ns1 = run(SOL_1, r"""
llr_a = log_lik_ratio(0.7, 0.5, 7, 10)
llr_b = log_lik_ratio(0.7, 0.5, 70, 100)
ll_grid = bernoulli_loglik(np.linspace(0.01, 0.99, 99), 7, 10)
""")
ex.add(
    title="The Bernoulli log-likelihood and its maximiser", difficulty=1,
    prompt_md=r"""
For $k$ successes in $n$ independent Bernoulli trials the log-likelihood (dropping the binomial coefficient,
which does not depend on $p$) is $\ell(p) = k\log p + (n - k)\log(1 - p)$. Implement

* `bernoulli_loglik(p, k, n)` — $\ell(p)$; `p` may be a float or a NumPy array (return an array of the same shape);
* `mle_bernoulli(k, n)` — the maximiser $\hat p$ (derive it by setting $\ell'(p) = 0$; do not search numerically);
* `log_lik_ratio(p1, p2, k, n)` — $\ell(p_1) - \ell(p_2)$ as a float, the log of the likelihood ratio.

Use only NumPy (`np.log`).
""",
    starter=r"""
def bernoulli_loglik(p, k, n):
    # YOUR CODE HERE
    return None

def mle_bernoulli(k, n):
    # YOUR CODE HERE
    return None

def log_lik_ratio(p1, p2, k, n):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_1,
    check=rf"""
try:
    got = bernoulli_loglik(0.5, 7, 10)
except Exception:
    got = None
check_close('log-likelihood at p = 0.5 for 7/10 is 10 log(0.5)', got, 10 * np.log(0.5))
try:
    arr = bernoulli_loglik(np.linspace(0.01, 0.99, 99), 7, 10)
    ok_shape = np.shape(arr) == (99,)
    argmax_p = float(np.linspace(0.01, 0.99, 99)[np.argmax(arr)]) if ok_shape else None
except Exception:
    ok_shape, argmax_p = False, None
check('vectorised over an array of p', ok_shape)
check_close('the grid maximum is at p = 0.70', argmax_p, 0.70, tol=1e-9)
try:
    got = mle_bernoulli(7, 10)
except Exception:
    got = None
check_close('MLE of 7/10 is 0.7', got, 0.7)
try:
    got = log_lik_ratio(0.7, 0.5, 7, 10)
except Exception:
    got = None
check_hash('log likelihood ratio 0.7 vs 0.5 for 7 heads in 10', got, '{sha_of(ns1["llr_a"])}')
try:
    got = log_lik_ratio(0.7, 0.5, 70, 100)
except Exception:
    got = None
check_hash('log likelihood ratio 0.7 vs 0.5 for 70 heads in 100', got, '{sha_of(ns1["llr_b"])}')
""",
    explanation_md=r"""
$\ell'(p) = k/p - (n-k)/(1-p) = 0$ gives $k(1-p) = (n-k)p$, i.e. $\hat p = k/n$ — the sample proportion.
The binomial coefficient is a constant in $p$ and cancels in every ratio, which is why it can be dropped.
The log likelihood ratio grows linearly with the data: ten times the flips at the same proportion gives ten
times the log-ratio ($0.823 \to 8.23$), turning weak evidence into overwhelming evidence. A common slip is
to compute the *difference* of likelihoods rather than of log-likelihoods, or to divide by $n$ — the
likelihood ratio is a ratio of total probabilities, not of averages.
""",
)

# ---------------------------------------------------------------- 2. Exponential MLE (warm-up)
SOL_2 = r"""
def mle_exponential(x):
    x = np.asarray(x, dtype=float)
    return float(len(x) / x.sum())

def exp_loglik(lam, x):
    x = np.asarray(x, dtype=float)
    return float(len(x) * np.log(lam) - lam * x.sum())

def mle_tail_prob(x, t):
    return float(np.exp(-mle_exponential(x) * t))
"""
ns2 = run(SOL_2, r"""
x_rep = np.array([2.0, 0.5, 1.5, 4.0])
lam_hat = mle_exponential(x_rep)
tail3 = mle_tail_prob(x_rep, 3.0)
x_big = rng.exponential(scale=1/2.5, size=400)
lam_big = mle_exponential(x_big)
ll_at_mle = exp_loglik(lam_hat, x_rep)
ll_near = exp_loglik(lam_hat * 1.1, x_rep)
""")
ex.add(
    title="Derive and code the Exponential MLE", difficulty=1,
    prompt_md=r"""
Waiting times $x_1, \dots, x_n$ follow an Exponential distribution with rate $\lambda$, density
$f(x) = \lambda e^{-\lambda x}$. Derive the log-likelihood, set its derivative to zero, and implement

* `exp_loglik(lam, x)` — $\ell(\lambda)$ for an array `x`, as a float;
* `mle_exponential(x)` — $\hat\lambda$ as a float (closed form, no optimiser);
* `mle_tail_prob(x, t)` — the MLE of $P(X > t) = e^{-\lambda t}$, using the invariance property.

Test data: `x = np.array([2.0, 0.5, 1.5, 4.0])` and `t = 3.0`. The second hidden check draws 400 waiting
times with `rng.exponential(scale=1/2.5, size=400)` right after the setup cell's `rng` is created (so run the
setup cell first, then only this exercise's cells, in order).
""",
    starter=r"""
def exp_loglik(lam, x):
    # YOUR CODE HERE
    return None

def mle_exponential(x):
    # YOUR CODE HERE
    return None

def mle_tail_prob(x, t):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_2,
    check=rf"""
x_rep = np.array([2.0, 0.5, 1.5, 4.0])
try:
    got = mle_exponential(x_rep)
except Exception:
    got = None
check_close('MLE rate for the repair times is 1/mean = 0.5', got, 0.5)
try:
    got = mle_tail_prob(x_rep, 3.0)
except Exception:
    got = None
check_hash('MLE of P(X > 3) by invariance', got, '{sha_of(ns2["tail3"], decimals=4)}', decimals=4)
try:
    ll0 = exp_loglik(0.5, x_rep); ll1 = exp_loglik(0.55, x_rep); ll2 = exp_loglik(0.45, x_rep)
    ok_max = (ll0 is not None) and ll0 > ll1 and ll0 > ll2
except Exception:
    ok_max = False
check('the log-likelihood is larger at the MLE than 10% either side', ok_max)
try:
    got = exp_loglik(0.5, x_rep)
except Exception:
    got = None
check_close('log-likelihood at lambda = 0.5 equals 4 log 0.5 - 4', got, 4 * np.log(0.5) - 4.0)
rng = np.random.default_rng(0)
x_big = rng.exponential(scale=1/2.5, size=400)
try:
    got = mle_exponential(x_big)
except Exception:
    got = None
check_hash('MLE rate for the 400 simulated waiting times', got, '{sha_of(ns2["lam_big"], decimals=3)}', decimals=3)
check('that estimate is within 0.3 of the true rate 2.5', (got is not None) and abs(got - 2.5) < 0.3)
""",
    explanation_md=r"""
$\ell(\lambda) = \sum_i [\log\lambda - \lambda x_i] = n\log\lambda - \lambda\sum_i x_i$, so
$\ell'(\lambda) = n/\lambda - \sum_i x_i = 0$ gives $\hat\lambda = n/\sum x_i = 1/\bar x$; the second
derivative $-n/\lambda^2 < 0$ confirms a maximum. For the repair times $\bar x = 2$, $\hat\lambda = 0.5$.
Invariance says the MLE of any function of $\lambda$ is that function of $\hat\lambda$, so
$\widehat{P}(X > 3) = e^{-1.5} \approx 0.223$ needs no new optimisation. The usual wrong turn is to
estimate the *mean* $1/\lambda$ and forget to invert, or to use $\bar x$ itself as the rate.
""",
)

# ---------------------------------------------------------------- 3. Fisher information for Poisson (core)
SOL_3 = r"""
def fisher_info_poisson(lam, n):
    return float(n / lam)

def poisson_mle_se(x):
    x = np.asarray(x, dtype=float)
    lam_hat = x.mean()
    return float(lam_hat), float(np.sqrt(lam_hat / len(x)))

def wald_ci_poisson(x, z=1.96):
    lam_hat, se = poisson_mle_se(x)
    return (lam_hat - z * se, lam_hat + z * se)

def observed_info_poisson(x):
    x = np.asarray(x, dtype=float)
    lam_hat = x.mean()
    return float(x.sum() / lam_hat**2)       # -l''(lam_hat) = sum(x)/lam^2
"""
ns3 = run(SOL_3, r"""
calls = np.array([3, 5, 2, 4, 6, 4])
I_calls = fisher_info_poisson(calls.mean(), len(calls))
lam_c, se_c = poisson_mle_se(calls)
ci_c = wald_ci_poisson(calls)
J_calls = observed_info_poisson(calls)
counts_big = rng.poisson(7.0, size=250)
ci_big = wald_ci_poisson(counts_big)
J_big = observed_info_poisson(counts_big)
""")
ex.add(
    title="Fisher information and a Wald interval for a Poisson rate", difficulty=2,
    prompt_md=r"""
For a single Poisson count $x$ with rate $\lambda$, $\log p(x \mid \lambda) = x\log\lambda - \lambda - \log x!$.
Derive the Fisher information $i(\lambda) = -\mathbb{E}[\partial^2_\lambda \log p]$ by hand, then implement

* `fisher_info_poisson(lam, n)` — $I_n(\lambda) = n\,i(\lambda)$ for $n$ i.i.d. counts, as a float;
* `poisson_mle_se(x)` — a tuple `(lam_hat, se)` with the MLE and its standard error $1/\sqrt{I_n(\hat\lambda)}$;
* `wald_ci_poisson(x, z=1.96)` — the tuple `(lower, upper)` of the Wald interval $\hat\lambda \pm z\,\operatorname{SE}$;
* `observed_info_poisson(x)` — the observed information $J(\hat\lambda) = -\ell''(\hat\lambda)$ computed from the
  data (not from the expectation formula). For the Poisson it should coincide with $I_n(\hat\lambda)$.

Test data: the six hourly call counts `[3, 5, 2, 4, 6, 4]`. A second check uses 250 counts drawn with
`rng.poisson(7.0, size=250)` from a freshly seeded `np.random.default_rng(0)`.
""",
    starter=r"""
def fisher_info_poisson(lam, n):
    # YOUR CODE HERE
    return None

def poisson_mle_se(x):
    # YOUR CODE HERE
    return None

def wald_ci_poisson(x, z=1.96):
    # YOUR CODE HERE
    return None

def observed_info_poisson(x):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_3,
    check=rf"""
calls = np.array([3, 5, 2, 4, 6, 4])
try:
    got = fisher_info_poisson(4.0, 6)
except Exception:
    got = None
check_close('I_6(lambda = 4) = 6/4 = 1.5', got, 1.5)
try:
    lam_hat, se = poisson_mle_se(calls)
except Exception:
    lam_hat, se = None, None
check_close('MLE for the call counts is 4.0', lam_hat, 4.0)
check_hash('standard error of the call-rate MLE', se, '{sha_of(ns3["se_c"], decimals=4)}', decimals=4)
try:
    lo, hi = wald_ci_poisson(calls)
except Exception:
    lo, hi = None, None
check_hash('upper end of the 95% Wald interval for the call rate', hi, '{sha_of(ns3["ci_c"][1], decimals=3)}', decimals=3)
check_close('the interval is symmetric about the MLE', (None if lo is None else (lo + hi) / 2), 4.0)
try:
    got = observed_info_poisson(calls)
except Exception:
    got = None
check_close('observed information equals n/lambda_hat for the call counts', got, 6 / 4.0)
rng = np.random.default_rng(0)
counts_big = rng.poisson(7.0, size=250)
try:
    lo, hi = wald_ci_poisson(counts_big)
except Exception:
    lo, hi = None, None
check_hash('lower end of the 95% Wald interval for the 250 counts', lo, '{sha_of(ns3["ci_big"][0], decimals=3)}', decimals=3)
check('that interval contains the true rate 7.0', (lo is not None) and lo < 7.0 < hi)
""",
    explanation_md=r"""
$\partial_\lambda \log p = x/\lambda - 1$ and $\partial^2_\lambda \log p = -x/\lambda^2$; taking the
expectation with $\mathbb{E}[x] = \lambda$ gives $i(\lambda) = 1/\lambda$, so $I_n = n/\lambda$ and
$\operatorname{SE}(\hat\lambda) = \sqrt{\hat\lambda/n}$ — for the calls, $\sqrt{4/6} = 0.816$ and a Wald
interval of $[2.40, 5.60]$. The observed information is $-\ell''(\hat\lambda) = \sum x_i/\hat\lambda^2 =
n\bar x/\bar x^2 = n/\bar x$, identical to $I_n(\hat\lambda)$ because the Poisson's second derivative depends
on the data only through $\sum x_i$. (Only six counts: the interval is wide, and its lower end is not far
from zero — a reminder that the Wald interval can go negative for small counts, which the Gamma posterior of
the guide's Section 7 never does.) A frequent error is to forget the $n$: the information for one count is
$1/\lambda$, for $n$ counts $n/\lambda$.
""",
)

# ---------------------------------------------------------------- 4. numerical Gamma MLE (core)
SOL_4 = r"""
def gamma_nll(theta, x):
    log_a, log_b = theta
    a, b = np.exp(log_a), np.exp(log_b)
    x = np.asarray(x, dtype=float)
    return float(-(len(x) * a * np.log(b) - len(x) * special.gammaln(a) + (a - 1) * np.log(x).sum() - b * x.sum()))

def fit_gamma_mle(x):
    x = np.asarray(x, dtype=float)
    # method-of-moments start, in log-parameters
    a0 = x.mean()**2 / x.var(); b0 = a0 / x.mean()
    res = optimize.minimize(gamma_nll, x0=np.log([a0, b0]), args=(x,), method="BFGS")
    a_hat, b_hat = np.exp(res.x)
    return float(a_hat), float(b_hat)
"""
ns4 = run(SOL_4, r"""
x_small = np.array([1.2, 2.5, 0.8, 3.1, 1.9])
a_s, b_s = fit_gamma_mle(x_small)
x_sim = rng.gamma(shape=2.5, scale=1/1.5, size=300)
a_hat, b_hat = fit_gamma_mle(x_sim)
a_sp, _, scale_sp = stats.gamma.fit(x_sim, floc=0)
nll_sim = gamma_nll(np.log([a_hat, b_hat]), x_sim)
""")
ex.add(
    title="Numerical MLE for the Gamma distribution", difficulty=2,
    prompt_md=r"""
The Gamma$(\alpha, \beta)$ density (shape $\alpha$, **rate** $\beta$) is
$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha - 1}e^{-\beta x}$, so for a sample $x_1, \dots, x_n$

$$\ell(\alpha, \beta) = n\alpha\log\beta - n\log\Gamma(\alpha) + (\alpha - 1)\sum_i\log x_i - \beta\sum_i x_i .$$

There is no closed form for $\hat\alpha$. Implement

* `gamma_nll(theta, x)` — the **negative** log-likelihood as a float, where `theta = (log_alpha, log_beta)` are
  the *logarithms* of the parameters (so the optimiser can roam over all of $\mathbb{R}^2$). Use
  `scipy.special.gammaln` for $\log\Gamma$.
* `fit_gamma_mle(x)` — returns the tuple `(alpha_hat, beta_hat)` by minimising `gamma_nll` with
  `scipy.optimize.minimize` (any method; a method-of-moments start $\alpha_0 = \bar x^2/\hat\sigma^2$,
  $\beta_0 = \alpha_0/\bar x$ is a good idea).

Checks compare with the guide's hand-worked five-point example and with `scipy.stats.gamma.fit(x, floc=0)`
(which returns `shape, loc, scale` with `scale = 1/beta`) on 300 points drawn via
`rng.gamma(shape=2.5, scale=1/1.5, size=300)` from a fresh `np.random.default_rng(0)`.
""",
    starter=r"""
def gamma_nll(theta, x):
    log_a, log_b = theta
    # YOUR CODE HERE
    return None

def fit_gamma_mle(x):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_4,
    check=rf"""
x_small = np.array([1.2, 2.5, 0.8, 3.1, 1.9])
try:
    got = gamma_nll(np.log([2.0, 1.0]), x_small)
except Exception:
    got = None
check_close('NLL at alpha = 2, beta = 1 equals -sum(log x) + sum(x)', got, float(-(np.log(x_small).sum()) + x_small.sum()), tol=1e-6)
try:
    a_s, b_s = fit_gamma_mle(x_small)
except Exception:
    a_s, b_s = None, None
check_close("shape MLE for the guide's five points is 4.620", a_s, 4.6201, tol=2e-3)
check_close('rate MLE equals shape / mean', (None if a_s is None else b_s), (None if a_s is None else a_s / x_small.mean()), tol=2e-3)
rng = np.random.default_rng(0)
x_sim = rng.gamma(shape=2.5, scale=1/1.5, size=300)
try:
    a_hat, b_hat = fit_gamma_mle(x_sim)
    a_sp, _, scale_sp = stats.gamma.fit(x_sim, floc=0)
except Exception:
    a_hat, b_hat, a_sp, scale_sp = None, None, 1.0, 1.0
check_close('shape agrees with scipy.stats.gamma.fit', a_hat, a_sp, tol=1e-3)
check_close('rate agrees with 1/scale from scipy', b_hat, 1 / scale_sp, tol=1e-3)
try:
    nll_mle = gamma_nll(np.log([a_hat, b_hat]), x_sim)
    nll_off = gamma_nll(np.log([a_hat * 1.2, b_hat]), x_sim)
    ok = nll_mle < nll_off
except Exception:
    ok = False
check('the NLL is lower at the fitted parameters than at a perturbed shape', ok)
""",
    explanation_md=r"""
Working in $(\log\alpha, \log\beta)$ turns a constrained problem into an unconstrained one, so BFGS never
tries a negative shape. The NLL is smooth and convex in $\log\beta$ for fixed $\alpha$, and the profile in
$\alpha$ is unimodal, so any reasonable start converges — the method-of-moments start just saves iterations.
`scipy.stats.gamma.fit(x, floc=0)` solves the same problem (with `loc` fixed at 0, as our model assumes),
which is why the two agree to about $10^{-6}$; without `floc=0` scipy would also fit a shift parameter and
land somewhere else. Forgetting `gammaln` and using `np.log(special.gamma(a))` overflows for $a \gtrsim 170$;
forgetting the $n$ in $n\alpha\log\beta$ and $n\log\Gamma(\alpha)$ is the classic bookkeeping bug.
""",
)

# ---------------------------------------------------------------- 5. cross-entropy from likelihood (core)
SOL_5 = r"""
def bce_from_likelihood(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    lik = p**y * (1 - p)**(1 - y)            # Bernoulli p(y_i | p_i)
    return float(-np.mean(np.log(lik)))

def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def categorical_nll(logits, labels):
    p = softmax(logits)
    labels = np.asarray(labels)
    return float(-np.mean(np.log(p[np.arange(len(labels)), labels])))

def ce_grad_logits(logits, labels):
    p = softmax(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(labels)), labels] = 1.0
    return (p - onehot) / len(labels)
"""
ns5 = run(SOL_5, r"""
y_demo = np.array([1, 0, 1, 1, 0]); p_demo = np.array([0.9, 0.2, 0.6, 0.3, 0.1])
bce_demo = bce_from_likelihood(y_demo, p_demo)
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, -1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 3.0]])
labels = np.array([0, 1, 2, 1])
nll_demo = categorical_nll(logits, labels)
g = ce_grad_logits(logits, labels)
g00 = float(g[0, 0]); g31 = float(g[3, 1])
""")
ex.add(
    title="Cross-entropy is the negative log-likelihood of the labels", difficulty=2,
    prompt_md=r"""
Build the classification losses *from the likelihood*, not from a formula you remember:

* `bce_from_likelihood(y, p)` — for binary labels `y` and predicted probabilities `p`, compute the Bernoulli
  likelihood of each label, $p_i^{y_i}(1 - p_i)^{1 - y_i}$, take logs, and return $-\frac1n\sum_i \log(\cdot)$
  as a float. It must equal the binary cross-entropy.
* `softmax(z)` — row-wise softmax of a 2-D array of logits (subtract the row maximum first for stability).
* `categorical_nll(logits, labels)` — for an $n \times K$ array of logits and integer `labels` in $\{0, \dots, K-1\}$,
  return $-\frac1n\sum_i \log \text{softmax}(z_i)_{y_i}$ as a float.
* `ce_grad_logits(logits, labels)` — the gradient of `categorical_nll` with respect to the logits, an $n \times K$
  array. Derive it: for the mean loss it is $(p_i - \mathbf{e}_{y_i})/n$ in row $i$.

Use only NumPy. The checks compare with `sklearn.metrics.log_loss` and with a finite-difference gradient.
""",
    starter=r"""
def bce_from_likelihood(y, p):
    # YOUR CODE HERE
    return None

def softmax(z):
    # YOUR CODE HERE
    return None

def categorical_nll(logits, labels):
    # YOUR CODE HERE
    return None

def ce_grad_logits(logits, labels):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_5,
    check=rf"""
from sklearn.metrics import log_loss
y_demo = np.array([1, 0, 1, 1, 0]); p_demo = np.array([0.9, 0.2, 0.6, 0.3, 0.1])
try:
    got = bce_from_likelihood(y_demo, p_demo)
except Exception:
    got = None
check_close('BCE from the Bernoulli likelihood equals sklearn log_loss', got, log_loss(y_demo, p_demo), tol=1e-9)
try:
    got = bce_from_likelihood(np.array([1, 1]), np.array([0.5, 0.5]))
except Exception:
    got = None
check_close('two labels at p = 0.5 cost log 2 each', got, np.log(2.0))
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, -1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 3.0]])
labels = np.array([0, 1, 2, 1])
try:
    sm = softmax(logits)
    ok_rows = np.allclose(sm.sum(axis=1), 1.0) and sm.shape == (4, 3) and np.all(sm > 0)
except Exception:
    ok_rows = False
check('softmax rows are positive and sum to 1', ok_rows)
try:
    got = categorical_nll(logits, labels)
except Exception:
    got = None
check_close('categorical NLL equals sklearn log_loss on the softmax probabilities', got, (log_loss(labels, softmax(logits), labels=[0, 1, 2]) if ok_rows else None), tol=1e-9)
check_hash('categorical NLL of the four examples', got, '{sha_of(ns5["nll_demo"], decimals=4)}', decimals=4)
try:
    g = ce_grad_logits(logits, labels)
    eps = 1e-6; g_fd = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            d = np.zeros_like(logits); d[i, j] = eps
            g_fd[i, j] = (categorical_nll(logits + d, labels) - categorical_nll(logits - d, labels)) / (2 * eps)
    ok_grad = g.shape == (4, 3) and np.allclose(g, g_fd, atol=1e-6)
    ok_rowsum = np.allclose(g.sum(axis=1), 0.0, atol=1e-12)
except Exception:
    ok_grad, ok_rowsum = False, False
check('analytic gradient (p - y)/n matches finite differences', ok_grad)
check('each row of the gradient sums to zero', ok_rowsum)
try:
    got = float(ce_grad_logits(logits, labels)[3, 1])
except Exception:
    got = None
check_hash('gradient entry for example 4, true class 1', got, '{sha_of(ns5["g31"], decimals=4)}', decimals=4)
""",
    explanation_md=r"""
Writing $p(y_i \mid p_i) = p_i^{y_i}(1 - p_i)^{1 - y_i}$ and taking the log gives
$y_i\log p_i + (1 - y_i)\log(1 - p_i)$ — the negative mean of this *is* binary cross-entropy, so the two
functions agree to machine precision. For $K$ classes the likelihood of the label is the softmax probability
of the true class, and its negative log is the cross-entropy with a one-hot target. The gradient is the
famous $p - y$: because $\partial\log\text{softmax}(z)_y/\partial z_j = \mathbf{1}[j = y] - p_j$, each row of
the gradient is $(p_i - \mathbf{e}_{y_i})/n$, which sums to zero (the softmax cannot raise all logits at
once). Subtracting the row maximum before `exp` prevents overflow and does not change the softmax, since
the shift cancels in the ratio. The common bug is forgetting the $1/n$ in the gradient after averaging the
loss, which the finite-difference check catches.
""",
)

# ---------------------------------------------------------------- 6. Beta posterior (core)
SOL_6 = r"""
def beta_posterior(a, b, k, n):
    return (a + k, b + n - k)

def beta_posterior_summary(a, b, k, n, level=0.95):
    a_post, b_post = beta_posterior(a, b, k, n)
    post = stats.beta(a_post, b_post)
    mean = a_post / (a_post + b_post)
    alpha = (1 - level) / 2
    lo, hi = post.ppf(alpha), post.ppf(1 - alpha)
    return dict(a=a_post, b=b_post, mean=float(mean), lower=float(lo), upper=float(hi))

def prob_rate_above(a, b, k, n, threshold):
    a_post, b_post = beta_posterior(a, b, k, n)
    return float(1 - stats.beta.cdf(threshold, a_post, b_post))
"""
ns6 = run(SOL_6, r"""
s1 = beta_posterior_summary(1, 1, 2, 40)
s2 = beta_posterior_summary(2, 8, 7, 10)
s3 = beta_posterior_summary(5, 95, 2, 15)
pa = prob_rate_above(1, 1, 2, 40, 0.10)
pb = prob_rate_above(5, 95, 20, 150, 0.08)
""")
ex.add(
    title="Beta-Binomial posterior, posterior mean and credible interval", difficulty=2,
    prompt_md=r"""
Implement the conjugate update for a success probability:

* `beta_posterior(a, b, k, n)` — the tuple `(a_post, b_post)` of the posterior Beta parameters after $k$
  successes in $n$ trials with a Beta$(a, b)$ prior;
* `beta_posterior_summary(a, b, k, n, level=0.95)` — a dict with keys `a`, `b`, `mean` (posterior mean) and
  `lower`, `upper` (the equal-tailed `level` credible interval, via `scipy.stats.beta.ppf`);
* `prob_rate_above(a, b, k, n, threshold)` — $P(p > \text{threshold} \mid \mathcal{D})$ as a float.

Test cases: (i) uniform prior, 2 clicks in 40 impressions; (ii) Beta$(2, 8)$ prior, 7 heads in 10;
(iii) the surgeon from the guide — Beta$(5, 95)$ prior, 2 complications in 15 operations.
""",
    starter=r"""
def beta_posterior(a, b, k, n):
    # YOUR CODE HERE
    return None

def beta_posterior_summary(a, b, k, n, level=0.95):
    # YOUR CODE HERE
    return None

def prob_rate_above(a, b, k, n, threshold):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_6,
    check=rf"""
try:
    got = beta_posterior(1, 1, 2, 40)
except Exception:
    got = None
check('uniform prior + 2/40 gives Beta(3, 39)', got is not None and tuple(got) == (3, 39))
try:
    s = beta_posterior_summary(1, 1, 2, 40)
    mean, lo, hi = s["mean"], s["lower"], s["upper"]
except Exception:
    mean, lo, hi = None, None, None
check_close('posterior mean 3/42', mean, 3/42)
check_hash('upper end of the 95% credible interval (2 in 40, uniform prior)', hi, '{sha_of(ns6["s1"]["upper"], decimals=4)}', decimals=4)
check('the interval is inside [0, 1] and contains the posterior mean', (lo is not None) and 0 < lo < mean < hi < 1)
try:
    s = beta_posterior_summary(2, 8, 7, 10)
    got = s["mean"]
except Exception:
    got = None
check_close('Beta(2,8) prior + 7/10 -> mean 0.45', got, 0.45)
try:
    s = beta_posterior_summary(5, 95, 2, 15)
    got = s["mean"]
except Exception:
    got = None
check_hash('shrunken complication rate of the surgeon (Beta(5,95) prior, 2 in 15)', got, '{sha_of(ns6["s3"]["mean"], decimals=4)}', decimals=4)
try:
    got = prob_rate_above(1, 1, 2, 40, 0.10)
except Exception:
    got = None
check_hash('P(p > 0.10 | 2 clicks in 40, uniform prior)', got, '{sha_of(ns6["pa"], decimals=3)}', decimals=3)
try:
    got = prob_rate_above(5, 95, 20, 150, 0.08)
except Exception:
    got = None
check_hash('P(rate > 8%) for the surgeon with 20 complications in 150', got, '{sha_of(ns6["pb"], decimals=3)}', decimals=3)
""",
    explanation_md=r"""
Multiplying the Beta$(a, b)$ kernel $p^{a-1}(1-p)^{b-1}$ by the Binomial kernel $p^k(1-p)^{n-k}$ gives
$p^{a+k-1}(1-p)^{b+n-k-1}$, the kernel of Beta$(a + k, b + n - k)$ — the whole update is two additions. The
posterior mean $\frac{a+k}{a+b+n}$ is the data proportion pulled toward the prior mean with weight
$\frac{a+b}{a+b+n}$: for the surgeon, $7/115 = 6.1\%$ instead of the raw $13.3\%$, because 100 pseudo-cases
outweigh 15 real ones. Credible intervals come straight from the Beta quantiles and can never leave $[0, 1]$,
unlike the Wald interval. A typical error is to add $n$ to the second parameter instead of $n - k$ — the
prior failures plus the observed failures, not the total trials.
""",
)

# ---------------------------------------------------------------- 7. Normal-Normal posterior (core)
SOL_7 = r"""
def normal_normal_posterior(mu0, tau2, sigma2, x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    prec_post = 1 / tau2 + n / sigma2
    mu_post = (mu0 / tau2 + x.sum() / sigma2) / prec_post
    return float(mu_post), float(1 / prec_post)

def sequential_normal_update(mu0, tau2, sigma2, x):
    mu, var = float(mu0), float(tau2)
    for xi in np.asarray(x, dtype=float):
        mu, var = normal_normal_posterior(mu, var, sigma2, [xi])
    return mu, var
"""
ns7 = run(SOL_7, r"""
readings = np.array([106.0, 110.5, 107.2, 108.3])
mu_b, var_b = normal_normal_posterior(100.0, 100.0, 16.0, readings)
mu_s, var_s = sequential_normal_update(100.0, 100.0, 16.0, readings)
mu_c, var_c = normal_normal_posterior(0.0, 1.0, 4.0, rng.normal(1.5, 2.0, size=25))
""")
ex.add(
    title="Normal-Normal updating: precisions add", difficulty=2,
    prompt_md=r"""
A quantity $\mu$ has prior $\mathcal{N}(\mu_0, \tau^2)$; observations $x_i \sim \mathcal{N}(\mu, \sigma^2)$ with
**known** $\sigma^2$. Implement

* `normal_normal_posterior(mu0, tau2, sigma2, x)` — returns `(mu_post, var_post)` for the array `x`, using the
  precision form $\frac{1}{\sigma_n^2} = \frac{1}{\tau^2} + \frac{n}{\sigma^2}$,
  $\mu_n = \sigma_n^2\big(\frac{\mu_0}{\tau^2} + \frac{\sum_i x_i}{\sigma^2}\big)$;
* `sequential_normal_update(mu0, tau2, sigma2, x)` — the same posterior obtained by updating **one observation at
  a time**, feeding each posterior in as the next prior; it must return the same `(mu_post, var_post)`.

Test case: prior $\mathcal{N}(100, 10^2)$, sensor noise $\sigma = 4$, readings `[106.0, 110.5, 107.2, 108.3]`.
A second check uses 25 draws `rng.normal(1.5, 2.0, size=25)` from a fresh `np.random.default_rng(0)` with prior
$\mathcal{N}(0, 1)$ and $\sigma^2 = 4$.
""",
    starter=r"""
def normal_normal_posterior(mu0, tau2, sigma2, x):
    # YOUR CODE HERE
    return None

def sequential_normal_update(mu0, tau2, sigma2, x):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_7,
    check=rf"""
readings = np.array([106.0, 110.5, 107.2, 108.3])
try:
    mu_b, var_b = normal_normal_posterior(100.0, 100.0, 16.0, readings)
except Exception:
    mu_b, var_b = None, None
check_close('posterior variance 1/(0.01 + 0.25)', var_b, 1 / 0.26)
check_hash('posterior mean for the furnace readings', mu_b, '{sha_of(ns7["mu_b"], decimals=3)}', decimals=3)
check('posterior mean lies between the prior mean and the sample mean', (mu_b is not None) and 100.0 < mu_b < readings.mean())
try:
    mu_s, var_s = sequential_normal_update(100.0, 100.0, 16.0, readings)
except Exception:
    mu_s, var_s = None, None
check_close('sequential update gives the same mean as the batch update', mu_s, mu_b, tol=1e-9)
check_close('sequential update gives the same variance as the batch update', var_s, var_b, tol=1e-9)
rng = np.random.default_rng(0)
x25 = rng.normal(1.5, 2.0, size=25)
try:
    mu_c, var_c = normal_normal_posterior(0.0, 1.0, 4.0, x25)
except Exception:
    mu_c, var_c = None, None
check_hash('posterior mean for the 25 draws', mu_c, '{sha_of(ns7["mu_c"], decimals=3)}', decimals=3)
check_close('posterior variance 1/(1 + 25/4)', var_c, 1 / (1 + 25 / 4))
""",
    explanation_md=r"""
The prior contributes precision $1/\tau^2 = 0.01$ and the four readings $n/\sigma^2 = 0.25$; the posterior
precision is their sum $0.26$ (variance $3.85$, sd $1.96$) and the posterior mean is the precision-weighted
average $(0.01 \cdot 100 + 0.25 \cdot 108)/0.26 = 107.7$ — 96% of the way to the data because the sensor is
far more informative than the prior. Updating one reading at a time gives exactly the same answer: each step
adds $1/\sigma^2$ to the precision and $x_i/\sigma^2$ to the weighted sum, and addition is associative. This
is the scalar Kalman filter's measurement update. The usual mistake is to average *variances* or to weight
by standard deviations instead of precisions.
""",
)

# ---------------------------------------------------------------- 8. MAP = ridge (core)
SOL_8 = r"""
def map_gaussian_prior(X, y, sigma2, tau2):
    lam = sigma2 / tau2
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def mle_least_squares(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)
"""
ns8 = run(SOL_8, r"""
from sklearn.linear_model import Ridge
X8 = rng.normal(size=(40, 6)); w_true = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 1.5])
y8 = X8 @ w_true + rng.normal(0, 0.5, size=40)
w_map = map_gaussian_prior(X8, y8, sigma2=0.25, tau2=0.01)
w_mle = mle_least_squares(X8, y8)
w_ridge = Ridge(alpha=25.0, fit_intercept=False).fit(X8, y8).coef_
norm_map = float(np.linalg.norm(w_map)); norm_mle = float(np.linalg.norm(w_mle))
w_map_loose = map_gaussian_prior(X8, y8, sigma2=0.25, tau2=1e6)
X80 = np.vstack([X8] * 10); y80 = np.concatenate([y8] * 10)
w_map_big = map_gaussian_prior(X80, y80, sigma2=0.25, tau2=0.01)
""")
ex.add(
    title="MAP with a Gaussian prior is ridge regression with lambda = sigma^2 / tau^2", difficulty=2,
    prompt_md=r"""
For $y = Xw + \varepsilon$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, with prior $w_j \sim \mathcal{N}(0, \tau^2)$,
the guide derives $\hat w_{\text{MAP}} = \arg\min_w \|y - Xw\|^2 + \lambda\|w\|^2$ with $\lambda = \sigma^2/\tau^2$.
Implement, with NumPy only (`np.linalg.solve`, no scikit-learn inside the functions):

* `map_gaussian_prior(X, y, sigma2, tau2)` — the MAP weight vector via the closed form $(X^\top X + \lambda I)^{-1}X^\top y$;
* `mle_least_squares(X, y)` — the MLE / ordinary least-squares weights $(X^\top X)^{-1}X^\top y$.

The checks generate `X8 = rng.normal(size=(40, 6))`, `y8 = X8 @ w_true + rng.normal(0, 0.5, size=40)` with
`w_true = [2, -1, 0.5, 0, 0, 1.5]` from a fresh `np.random.default_rng(0)`, compare your MAP with
`Ridge(alpha=sigma2/tau2, fit_intercept=False)`, and verify two consequences: the MAP has a smaller norm than
the MLE, and stacking the data ten times (same prior) moves the MAP toward the MLE.
""",
    starter=r"""
def map_gaussian_prior(X, y, sigma2, tau2):
    # YOUR CODE HERE
    return None

def mle_least_squares(X, y):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_8,
    check=rf"""
from sklearn.linear_model import Ridge
rng = np.random.default_rng(0)
X8 = rng.normal(size=(40, 6)); w_true = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 1.5])
y8 = X8 @ w_true + rng.normal(0, 0.5, size=40)
try:
    w_map = np.asarray(map_gaussian_prior(X8, y8, sigma2=0.25, tau2=0.01), dtype=float)
    w_mle = np.asarray(mle_least_squares(X8, y8), dtype=float)
    ok_shape = w_map.shape == (6,) and w_mle.shape == (6,)
except Exception:
    w_map, w_mle, ok_shape = None, None, False
check('both weight vectors have shape (6,)', ok_shape)
check_close('MAP equals Ridge(alpha = sigma^2/tau^2 = 25)', w_map, Ridge(alpha=25.0, fit_intercept=False).fit(X8, y8).coef_, tol=1e-8)
check_close('MLE equals np.linalg.lstsq', w_mle, np.linalg.lstsq(X8, y8, rcond=None)[0], tol=1e-8)
check('the MAP has a smaller norm than the MLE (shrinkage)', ok_shape and np.linalg.norm(w_map) < np.linalg.norm(w_mle))
check_hash('norm of the MAP weights', (float(np.linalg.norm(w_map)) if ok_shape else None), '{sha_of(ns8["norm_map"], decimals=3)}', decimals=3)
try:
    w_loose = map_gaussian_prior(X8, y8, sigma2=0.25, tau2=1e6)
    ok_loose = np.allclose(w_loose, w_mle, atol=1e-4)
except Exception:
    ok_loose = False
check('a very wide prior (tau^2 = 1e6) recovers the MLE', ok_loose)
try:
    X80 = np.vstack([X8] * 10); y80 = np.concatenate([y8] * 10)
    w_big = map_gaussian_prior(X80, y80, sigma2=0.25, tau2=0.01)
    ok_big = np.linalg.norm(w_big - w_mle) < np.linalg.norm(w_map - w_mle)
except Exception:
    ok_big = False
check('ten times the data, same prior: the MAP moves toward the MLE', ok_big)
""",
    explanation_md=r"""
$-\log p(w \mid \mathcal{D}) = \frac{1}{2\sigma^2}\|y - Xw\|^2 + \frac{1}{2\tau^2}\|w\|^2 + \text{const}$;
multiplying by $2\sigma^2$ gives the ridge objective with $\lambda = \sigma^2/\tau^2 = 0.25/0.01 = 25$, and
setting the gradient $-2X^\top(y - Xw) + 2\lambda w$ to zero gives the normal equations with $\lambda I$ added
— identical to what `Ridge` solves, hence agreement to $10^{-8}$. Shrinkage is visible in the smaller norm;
with $\tau^2 = 10^6$ the prior is flat and the MAP collapses onto the MLE. Stacking the data ten times
multiplies $X^\top X$ and $X^\top y$ by ten while $\lambda I$ stays put, so the prior's share of the solution
drops tenfold: the same $\lambda$ regularises less on more data, which is why `alpha` must be re-tuned when
the dataset grows. (Stacking copies is only a device to scale $n$ without changing the data's direction;
the point stands for genuinely new data.)
""",
)

# ---------------------------------------------------------------- 9. conceptual (core)
ex.add_conceptual(
    title="Credible interval or confidence interval?", difficulty=2,
    prompt_md=r"""
A data scientist runs an experiment and reports: "The 95% confidence interval for the conversion rate is
$[4.1\%, 5.9\%]$. So there is a 95% probability that the true rate is between 4.1% and 5.9%." A colleague
replies: "You can only say that if you computed a credible interval — and then you'd have to tell me your
prior." Who is right? In your answer (a) explain precisely what the 95% in a confidence interval refers to,
(b) explain what a 95% credible interval claims and what it requires, (c) say under which circumstances the
two intervals are numerically almost identical, and (d) give one situation from this module where they differ
sharply and say which one you would trust.
""",
    answer_md=r"""
The colleague is right about the logic, even if the numbers may be close. **(a)** A confidence interval is
the output of a *procedure*; "95%" is the long-run frequency with which that procedure captures the fixed,
unknown parameter over hypothetical repetitions of the experiment. Once the data are in and the interval is
$[4.1\%, 5.9\%]$, the parameter is either inside it or not — no probability attaches to this particular
interval. **(b)** A 95% credible interval $[a, b]$ satisfies $P(a \le \theta \le b \mid \mathcal{D}) = 0.95$,
a direct probability statement about the parameter *given these data*; it requires a prior, because without
one there is no posterior to take quantiles of. **(c)** With a flat (or weak) prior and a large sample, the
posterior is approximately $\mathcal{N}(\hat\theta, \operatorname{SE}^2)$ — the same normal that the Wald
interval is built from — so the two intervals nearly coincide numerically (for thousands of visitors, both
are about $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$). The sentence becomes *correct* if it is read as a
statement about the flat-prior posterior. **(d)** They differ sharply for small $n$ or rates near 0 or 1: for
2 clicks in 40 impressions the Wald interval is $[-1.8\%, 11.8\%]$ — it includes impossible negative rates
and understates the right tail — while the Beta$(3, 39)$ credible interval is $[1.5\%, 16.5\%]$. Here the
credible interval is the one to trust, because it is exact for any $n$ and respects the parameter space,
whereas the Wald interval relies on asymptotics that have not kicked in. (A frequentist would reach for an
exact or Wilson interval instead, [Module 12](../12-hypothesis-testing-and-experiments/README.md).)
""",
)

# ---------------------------------------------------------------- 10. posterior predictive (core)
SOL_10 = r"""
def beta_binomial_pmf(j, m, a, b):
    j = np.asarray(j)
    log_pmf = (special.gammaln(m + 1) - special.gammaln(j + 1) - special.gammaln(m - j + 1)
               + special.betaln(j + a, m - j + b) - special.betaln(a, b))
    return np.exp(log_pmf)

def predictive_interval(a, b, m, level=0.95):
    j = np.arange(m + 1)
    cdf = np.cumsum(beta_binomial_pmf(j, m, a, b))
    alpha = (1 - level) / 2
    lo = int(j[np.searchsorted(cdf, alpha)])
    hi = int(j[np.searchsorted(cdf, 1 - alpha)])
    return lo, hi

def plug_in_interval(a, b, m, level=0.95):
    p_bar = a / (a + b)
    alpha = (1 - level) / 2
    return int(stats.binom.ppf(alpha, m, p_bar)), int(stats.binom.ppf(1 - alpha, m, p_bar))

def prob_no_events(a, b, m):
    return float(np.exp(special.betaln(a, b + m) - special.betaln(a, b)))
"""
ns10 = run(SOL_10, r"""
pmf_check = beta_binomial_pmf(np.arange(101), 100, 3, 39)
pi_pred = predictive_interval(3, 39, 100)
pi_plug = plug_in_interval(3, 39, 100)
pne = prob_no_events(1, 21, 10)
pred_mean = float(np.sum(np.arange(101) * pmf_check))
pred_var = float(np.sum((np.arange(101) - pred_mean)**2 * pmf_check))
pi_pred_b = predictive_interval(9, 11, 50)
""")
ex.add(
    title="Posterior predictive versus plug-in prediction", difficulty=2,
    prompt_md=r"""
After observing $k$ successes in $n$ trials with a Beta$(a_0, b_0)$ prior, the posterior is Beta$(a, b)$ and the
number of successes $J$ in $m$ *future* trials has the Beta-Binomial predictive distribution

$$P(J = j \mid \mathcal{D}) = \binom{m}{j}\frac{B(j + a,\; m - j + b)}{B(a, b)}, \qquad j = 0, \dots, m,$$

with $B$ the Beta function. Implement, using `scipy.special.betaln` and `gammaln` (work in logs; do **not**
use `scipy.stats.betabinom` — build it yourself):

* `beta_binomial_pmf(j, m, a, b)` — the PMF for an integer or integer array `j`;
* `predictive_interval(a, b, m, level=0.95)` — the tuple `(lo, hi)` of the smallest $j$ with CDF $\ge 0.025$ and the
  smallest $j$ with CDF $\ge 0.975$ (use `np.searchsorted` on the cumulative sum);
* `plug_in_interval(a, b, m, level=0.95)` — the same quantiles of the plug-in Binomial$(m, \bar p)$ with
  $\bar p = a/(a + b)$, via `scipy.stats.binom.ppf`;
* `prob_no_events(a, b, m)` — the predictive probability of zero successes in the next $m$ trials, $\frac{B(a, b + m)}{B(a, b)}$.

Test case: posterior Beta$(3, 39)$, $m = 100$ (the click-through example); and the zero-failure posterior
Beta$(1, 21)$ with $m = 10$.
""",
    starter=r"""
def beta_binomial_pmf(j, m, a, b):
    # YOUR CODE HERE
    return None

def predictive_interval(a, b, m, level=0.95):
    # YOUR CODE HERE
    return None

def plug_in_interval(a, b, m, level=0.95):
    # YOUR CODE HERE
    return None

def prob_no_events(a, b, m):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_10,
    check=rf"""
try:
    pmf = np.asarray(beta_binomial_pmf(np.arange(101), 100, 3, 39), dtype=float)
    ok_pmf = pmf.shape == (101,) and abs(pmf.sum() - 1) < 1e-9 and np.all(pmf >= 0)
except Exception:
    pmf, ok_pmf = None, False
check('the predictive PMF is non-negative and sums to 1', ok_pmf)
check_close('matches scipy.stats.betabinom', pmf, stats.betabinom.pmf(np.arange(101), 100, 3, 39), tol=1e-10)
try:
    got = float(beta_binomial_pmf(7, 100, 3, 39))
except Exception:
    got = None
check_hash('P(J = 7 | Beta(3, 39), m = 100)', got, '{sha_of(float(ns10["pmf_check"][7]), decimals=4)}', decimals=4)
try:
    lo_pred, hi_pred = predictive_interval(3, 39, 100)
    lo_plug, hi_plug = plug_in_interval(3, 39, 100)
except Exception:
    lo_pred, hi_pred, lo_plug, hi_plug = None, None, None, None
check_hash('upper end of the 95% predictive interval', hi_pred, '{sha_of(ns10["pi_pred"][1])}')
check_hash('upper end of the 95% plug-in interval', hi_plug, '{sha_of(ns10["pi_plug"][1])}')
check('the predictive interval is wider than the plug-in interval', (hi_pred is not None) and (hi_pred - lo_pred) > (hi_plug - lo_plug))
try:
    got = prob_no_events(1, 21, 10)
except Exception:
    got = None
check_close('P(no failure in 10 more tests | Beta(1, 21)) = 21/31', got, 21/31)
try:
    got = prob_no_events(3, 39, 100)
except Exception:
    got = None
check_close('consistency: prob_no_events equals the PMF at j = 0', got, (float(pmf[0]) if ok_pmf else None), tol=1e-10)
""",
    explanation_md=r"""
The predictive PMF is the Binomial PMF averaged over the Beta posterior:
$\int_0^1 \binom{m}{j}p^j(1-p)^{m-j}\,\frac{p^{a-1}(1-p)^{b-1}}{B(a,b)}\,dp = \binom{m}{j}\frac{B(j+a, m-j+b)}{B(a,b)}$,
because the integrand is an unnormalised Beta kernel. Working with `gammaln` and `betaln` keeps everything
finite for $m = 100$, where $\binom{100}{50} \approx 10^{29}$. The predictive has the same mean as the
plug-in Binomial but variance inflated by $(a + b + m)/(a + b + 1) = 142/43 \approx 3.3$, so its 95% interval
is 1–18 clicks against 3–13 for the plug-in. The zero-success probability is the $j = 0$ term,
$B(a, b + m)/B(a, b)$, which for Beta$(1, 21)$ and $m = 10$ is $21/31$ — the honest answer to "will the
component survive ten more tests?", instead of the plug-in's certainty. Forgetting the binomial coefficient
and forgetting to normalise by $B(a, b)$ are the two standard slips; the sum-to-one check catches both.
""",
)

# ---------------------------------------------------------------- 11. Bayesian A/B (stretch)
SOL_11 = r"""
def bayesian_ab_test(kA, nA, kB, nB, n_draws=200_000, a=1, b=1, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    pA = rng.beta(a + kA, b + nA - kA, size=n_draws)
    pB = rng.beta(a + kB, b + nB - kB, size=n_draws)
    return dict(prob_B_better=float(np.mean(pB > pA)),
                expected_loss_B=float(np.mean(np.maximum(pA - pB, 0))),
                expected_loss_A=float(np.mean(np.maximum(pB - pA, 0))),
                lift_mean=float(np.mean(pB - pA)))

def prob_B_better_exact(kA, nA, kB, nB, a=1, b=1):
    # P(pB > pA) = sum over j of P(pB = ... ) -- closed form via a sum over the Beta-Binomial (Cook's formula)
    aA, bA, aB, bB = a + kA, b + nA - kA, a + kB, b + nB - kB
    j = np.arange(int(aB))
    log_terms = (special.betaln(aA + j, bA + bB) - np.log(bB + j) - special.betaln(1 + j, bB) - special.betaln(aA, bA))
    return float(np.sum(np.exp(log_terms)))
"""
ns11 = run(SOL_11, r"""
res = bayesian_ab_test(120, 2400, 150, 2400)
exact = prob_B_better_exact(120, 2400, 150, 2400)
res_small = bayesian_ab_test(5, 100, 9, 100)
exact_small = prob_B_better_exact(5, 100, 9, 100)
exact_tiny = prob_B_better_exact(1, 5, 2, 5)
""")
ex.add(
    title="Bayesian A/B test by Monte Carlo, with an exact check", difficulty=3,
    prompt_md=r"""
Implement

* `bayesian_ab_test(kA, nA, kB, nB, n_draws=200_000, a=1, b=1, rng=None)` — with Beta$(a, b)$ priors on both
  conversion rates, draw `n_draws` samples from each posterior (use `rng.beta`; create
  `rng = np.random.default_rng(0)` when `rng is None`) and return a dict with keys `prob_B_better`
  ($P(p_B > p_A \mid \mathcal{D})$), `expected_loss_B` ($\mathbb{E}[\max(p_A - p_B, 0)]$, what you lose on average by
  shipping B if it is worse), `expected_loss_A` ($\mathbb{E}[\max(p_B - p_A, 0)]$) and `lift_mean` ($\mathbb{E}[p_B - p_A]$).
* `prob_B_better_exact(kA, nA, kB, nB, a=1, b=1)` — the *exact* probability, using the closed form for two
  independent Beta posteriors $p_A \sim \text{Beta}(\alpha_A, \beta_A)$, $p_B \sim \text{Beta}(\alpha_B, \beta_B)$ with
  integer $\alpha_B$:

  $$P(p_B > p_A) = \sum_{j=0}^{\alpha_B - 1}\frac{B(\alpha_A + j,\; \beta_A + \beta_B)}{(\beta_B + j)\,B(1 + j,\; \beta_B)\,B(\alpha_A, \beta_A)} .$$

  Compute each term in log space with `scipy.special.betaln` and exponentiate.

Test cases: the guide's A/B data (A: 120/2400, B: 150/2400) and a small test (A: 5/100, B: 9/100).
""",
    starter=r"""
def bayesian_ab_test(kA, nA, kB, nB, n_draws=200_000, a=1, b=1, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    # YOUR CODE HERE
    return None

def prob_B_better_exact(kA, nA, kB, nB, a=1, b=1):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_11,
    check=rf"""
try:
    res = bayesian_ab_test(120, 2400, 150, 2400)
    pBA = res["prob_B_better"]; lossB = res["expected_loss_B"]; lossA = res["expected_loss_A"]
except Exception:
    res, pBA, lossB, lossA = None, None, None, None
check('P(B > A) for the guide data is between 0.96 and 0.98', (pBA is not None) and 0.96 < pBA < 0.98)
check('expected loss of shipping B is below 0.0002', (lossB is not None) and 0 <= lossB < 2e-4)
check('expected loss of keeping A is above 0.011', (lossA is not None) and lossA > 0.011)
check_close('lift_mean equals the difference of posterior means (within MC error)', (None if res is None else res["lift_mean"]), 151/2402 - 121/2402, tol=2e-4)
try:
    got = prob_B_better_exact(120, 2400, 150, 2400)
except Exception:
    got = None
check_hash('exact P(B > A) for the guide data', got, '{sha_of(ns11["exact"], decimals=3)}', decimals=3)
check('Monte Carlo and exact agree within 0.003', (got is not None) and (pBA is not None) and abs(got - pBA) < 0.003)
try:
    got = prob_B_better_exact(5, 100, 9, 100)
except Exception:
    got = None
check_hash('exact P(B > A) for 5/100 vs 9/100', got, '{sha_of(ns11["exact_small"], decimals=3)}', decimals=3)
try:
    got = prob_B_better_exact(1, 5, 2, 5)
    brute = np.mean(np.random.default_rng(1).beta(2, 5, 400_000) < np.random.default_rng(2).beta(3, 4, 400_000))
except Exception:
    got, brute = None, None
check('exact formula agrees with brute force on a tiny example', (got is not None) and abs(got - brute) < 0.003)
try:
    got = prob_B_better_exact(10, 100, 10, 100)
except Exception:
    got = None
check('symmetric data give P(B > A) = 0.5', (got is not None) and abs(got - 0.5) < 1e-9)
""",
    explanation_md=r"""
Monte Carlo makes the two-dimensional integral $\iint \mathbf{1}[p_B > p_A]\,p(p_A)\,p(p_B)$ trivial: sample
each posterior, compare, average. With 200,000 draws the standard error of a probability near 0.97 is
$\sqrt{0.97 \times 0.03/200{,}000} \approx 0.0004$, which is why Monte Carlo and the exact formula agree to
three decimals. The exact sum (due to John Cook) comes from writing $P(p_B > p_A) = \mathbb{E}_{p_A}[1 -
F_B(p_A)]$ and expanding the Beta CDF with integer first parameter as a finite sum; it is handy for unit tests
but the Monte Carlo approach generalises to any number of variants and any prior. The expected losses are the
decision-relevant numbers: here shipping B risks about 0.008 percentage points on average, keeping A forfeits
about 1.25. A common mistake is to compute $P(p_B > p_A)$ from the *means* alone, or to use the same random
stream for both posteriors (which induces a spurious correlation).
""",
)

# ---------------------------------------------------------------- 12. mini-project (★★★)
PROJECT_CODE = r"""
rng = np.random.default_rng(11)
n_stores = 20

# --- synthetic ground truth: true conversion rates around 5%, very unequal traffic ---
true_rates = rng.beta(6, 114, size=n_stores)                 # mean 0.05, sd about 0.02
visits = rng.integers(20, 400, size=n_stores)
conversions = rng.binomial(visits, true_rates)
raw = conversions / visits

# --- empirical-Bayes Beta prior by the method of moments ---
m = raw.mean()
var_raw = raw.var(ddof=1)
sampling_noise = np.mean(raw * (1 - raw) / visits)             # average binomial variance of the raw rates
v_between = max(var_raw - sampling_noise, 1e-6)               # variance of the TRUE rates, net of noise
strength = m * (1 - m) / v_between - 1                        # a + b
a_eb, b_eb = m * strength, (1 - m) * strength
print(f"ensemble mean {m:.4f};  var of raw rates {var_raw:.6f} = between-store {v_between:.6f} + sampling noise {sampling_noise:.6f}")
print(f"empirical-Bayes prior: Beta({a_eb:.2f}, {b_eb:.2f})  ->  prior strength a + b = {strength:.1f} pseudo-visits")

# --- shrink every store toward the prior mean ---
shrunk = (a_eb + conversions) / (strength + visits)
weight_data = visits / (strength + visits)

mse_raw = np.mean((raw - true_rates) ** 2)
mse_shrunk = np.mean((shrunk - true_rates) ** 2)
mse_pooled = np.mean((m - true_rates) ** 2)
print(f"\nMSE vs true rates:  raw {mse_raw:.3e}   shrunken {mse_shrunk:.3e}   fully pooled (everyone = {m:.3f}) {mse_pooled:.3e}")
print(f"shrinkage cuts the MSE to {mse_shrunk / mse_raw:.0%} of the raw estimates'")

order = np.argsort(visits)
print(f"\n{'store':>5s} {'visits':>7s} {'conv':>5s} {'raw':>7s} {'shrunk':>7s} {'true':>7s} {'data weight':>12s}")
for i in order:
    print(f"{i:5d} {visits[i]:7d} {conversions[i]:5d} {raw[i]:7.3f} {shrunk[i]:7.3f} {true_rates[i]:7.3f} {weight_data[i]:12.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for i in range(n_stores):
    ax1.plot([0, 1], [raw[i], shrunk[i]], color="gray", lw=0.8)
ax1.scatter(np.zeros(n_stores), raw, s=visits / 4, label="raw k/n (size = visits)")
ax1.scatter(np.ones(n_stores), shrunk, s=visits / 4, label="shrunken")
ax1.axhline(m, color="k", ls=":", label="ensemble mean")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["raw", "empirical Bayes"]); ax1.set_ylabel("conversion rate")
ax1.set_title("Small stores move a lot, big stores barely"); ax1.legend(fontsize=8)
ax2.scatter(true_rates, raw, s=visits / 4, label=f"raw (MSE {mse_raw:.1e})")
ax2.scatter(true_rates, shrunk, s=visits / 4, marker="s", label=f"shrunken (MSE {mse_shrunk:.1e})")
lims = [0, max(raw.max(), true_rates.max()) * 1.05]
ax2.plot(lims, lims, "k--", lw=1); ax2.set_xlabel("true rate"); ax2.set_ylabel("estimate"); ax2.set_title("Estimates vs truth"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()

# --- ranking: which store is really the best? ---
print(f"\nbest store by raw rate:     #{np.argmax(raw)} (raw {raw.max():.3f}, {visits[np.argmax(raw)]} visits, true {true_rates[np.argmax(raw)]:.3f})")
print(f"best store by shrunken rate: #{np.argmax(shrunk)} (shrunk {shrunk.max():.3f}, {visits[np.argmax(shrunk)]} visits, true {true_rates[np.argmax(shrunk)]:.3f})")
print(f"truly best store:            #{np.argmax(true_rates)} (true {true_rates.max():.3f})")

# --- robustness: repeat the whole experiment 200 times ---
ratios = []
for rep in range(200):
    r = np.random.default_rng(1000 + rep)
    tr = r.beta(6, 114, size=n_stores); vi = r.integers(20, 400, size=n_stores); co = r.binomial(vi, tr); rw = co / vi
    mm = rw.mean(); vb = max(rw.var(ddof=1) - np.mean(rw * (1 - rw) / vi), 1e-6); st = mm * (1 - mm) / vb - 1
    sh = (mm * st + co) / (st + vi)
    ratios.append(np.mean((sh - tr) ** 2) / np.mean((rw - tr) ** 2))
ratios = np.array(ratios)
print(f"\nover 200 replications: median MSE ratio shrunken/raw = {np.median(ratios):.2f};  shrinkage helps in {np.mean(ratios < 1):.0%} of replications")
"""
ex.add_project(
    title="Mini-project: empirical-Bayes shrinkage of 20 store conversion rates",
    prompt_md=r"""
A retailer has 20 stores with very different traffic. Each store's conversion rate is estimated from its own
visits, and head office wants to rank the stores and spot genuinely strong or weak ones. Raw rates $k_j/n_j$
are noisy for small stores; empirical Bayes borrows strength across stores by estimating a Beta prior *from
the ensemble* and shrinking each store toward it.

**The model** (use `rng = np.random.default_rng(11)`):

1. Draw 20 true rates `true_rates = rng.beta(6, 114, size=20)` (mean 5%), visit counts
   `visits = rng.integers(20, 400, size=20)`, and `conversions = rng.binomial(visits, true_rates)`.
2. Fit a Beta$(a, b)$ prior by the **method of moments**: let $m$ be the mean of the raw rates and let $v$ be their
   variance (`ddof=1`) **minus** the average binomial sampling variance $\overline{\hat p_j(1 - \hat p_j)/n_j}$
   (floor $v$ at $10^{-6}$); then $a + b = \frac{m(1 - m)}{v} - 1$, $a = m(a + b)$, $b = (1 - m)(a + b)$.
3. Shrink each store to its posterior mean $\frac{a + k_j}{a + b + n_j}$.

**Deliverables**

* the fitted prior and its strength $a + b$ in pseudo-visits, with a one-sentence interpretation;
* the mean squared error of the raw and of the shrunken estimates **against the true rates**, and the MSE of
  the fully pooled estimate (every store $= m$) for comparison;
* a table or plot showing how far each store moves, with the data weight $n_j/(a + b + n_j)$, demonstrating
  that small stores move a lot and large stores barely;
* which store is "best" by raw rate, by shrunken rate, and in truth — and a comment on why they can differ;
* a robustness check: repeat the whole experiment for at least 100 random seeds and report the median ratio of
  shrunken-to-raw MSE and the fraction of replications in which shrinkage helps.
""",
    rubric_md=r"""
* The sampling-noise correction is applied before fitting the prior (subtracting the average $\hat p(1-\hat p)/n$),
  and the prior strength $a + b$ is reported in pseudo-visits.
* Shrinkage is the Beta posterior mean $\frac{a + k_j}{a + b + n_j}$ — a weighted average with data weight
  $n_j/(a + b + n_j)$ — and the per-store movement is shown to depend on $n_j$.
* MSE is computed against the *true* rates (this is synthetic data; that is the point) for raw, shrunken and
  fully pooled estimates, and the ordering is explained: pooling throws away real differences, raw estimates
  keep all the noise, shrinkage sits between.
* The ranking comparison identifies that the raw "best" store is typically a small store with a lucky sample.
* The robustness loop reports a median MSE ratio and a success fraction over $\ge 100$ seeds, with reproducible
  seeding.
* Figures have titles and axis labels.
""",
    solution_sketch_md=r"""
With seed 11 the twenty stores have between 35 and 399 visits. The raw rates have variance about
$6.6 \times 10^{-4}$, of which almost half ($3.2 \times 10^{-4}$) is binomial sampling noise, leaving a
between-store variance of about $3.5 \times 10^{-4}$ and a prior Beta$(5.9, 120.7)$ — a prior worth about 127
pseudo-visits centred at 4.6%. A store with 35 visits keeps only $35/162 \approx 22\%$ of its own evidence; a
store with 399 visits keeps 76%. Against the true rates, the shrunken estimates have an MSE of
$1.7 \times 10^{-4}$ against $4.9 \times 10^{-4}$ for the raw rates (ratio $\approx 0.35$) and
$2.0 \times 10^{-4}$ for full pooling: pooling beats the raw rates here because the noise is so large, but
shrinkage beats both, since it keeps the real differences that the data support. In this draw the raw and
shrunken rankings agree on the top store (#17, 19 of 173, raw 11.0% shrunk to 8.3%), yet its true rate is an
ordinary 5.2% — a lucky sample that even shrinkage cannot fully undo with 173 visits — while the truly best
store (#14, true 7.0%) had 1 conversion in 60 visits and looks terrible under both estimates. That is the
limit of the method: it corrects for *noise* on average, it cannot recover information that a small sample
never contained. Across 200 replications the median MSE ratio is about 0.55 and shrinkage helps in about 89% of
them — not a free lunch in every single dataset (when the true rates happen to be widely spread, the fitted
prior is weak and the gain small), but a large improvement on average, which is what Stein's paradox promises
for three or more simultaneous estimates.
""",
    solution_code=PROJECT_CODE,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print(f"exercises: {len(ex.items)} items — {kinds.count('code')} coding, {kinds.count('concept')} conceptual, {kinds.count('project')} project")
print("difficulties:", [it["difficulty"] for it in ex.items])
