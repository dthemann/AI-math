"""
build_18.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for

    part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models

Run from the repository root:

    python3 _authoring/builders/build_18.py
    python3 tools/run_notebooks.py part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models

All hidden answers are computed here by *executing* the reference solutions and hashing
the result with `sha_of`, so the exercises notebook never contains the answers in clear text.
Everything is NumPy/SciPy only — no deep-learning frameworks.
"""
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import ExerciseSet, Notebook, sha_of  # noqa: E402

import numpy as np  # noqa: E402,F401  (used when executing solution code)

MOD = ROOT / "part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models"

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 18 — Sampling, MCMC and Generative Models")

nb.md(r"""
This notebook is the runnable companion to the concept guide: read [README.md](README.md) first, or keep
it open alongside. Every section mirrors a numbered section of the guide. We start with the exact samplers
that turn uniform random numbers into any distribution (inverse CDF, cumulative sums, Box–Muller,
reparameterisation), then correct samples from a convenient proposal with rejection and importance
sampling, build Markov chains whose long-run distribution is a target we choose (Metropolis–Hastings,
Gibbs, Langevin dynamics), turn inference into optimisation with the ELBO and a NumPy variational
autoencoder, and finish with a one-dimensional diffusion model — forward noising, the closed-form marginal,
and reverse sampling with both the exact score and a small trained denoiser — and the Fréchet distance
used to evaluate generative models.

Everything is plain NumPy/SciPy, built from the formulas in the guide; library calls appear only to confirm
the hand computation.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
""")

# ---------------------------------------------------------------- 1. why sampling
nb.md(r"""
## 1. Why sampling: an integral becomes an average

The Monte Carlo estimator $\hat\mu_n = \frac1n\sum_i f(X_i)$ of $\mu = \mathbb{E}_p[f(X)]$ is unbiased
with standard deviation $\sigma_f/\sqrt{n}$, whatever the dimension. We test it on a case with a known
answer: $X \sim \mathcal{N}(0, 1)$, $f(x) = e^{-x^2}$, for which $\mathbb{E}[f(X)] = 1/\sqrt{3}$.
Repeating the whole experiment 200 times for each $n$ shows the error falling as $1/\sqrt{n}$.
""")

nb.code(r"""
exact = 1 / np.sqrt(3)
x = rng.standard_normal(10_000)
f = np.exp(-x ** 2)
print(f"Monte Carlo estimate with n = 10,000: {f.mean():.4f}   exact 1/sqrt(3) = {exact:.4f}")
print(f"sample sd of f(X) = {f.std():.4f}  ->  standard error = {f.std() / np.sqrt(len(f)):.4f}")

ns = np.array([10, 100, 1_000, 10_000, 100_000])
rmse = np.array([np.sqrt(np.mean([(np.exp(-rng.standard_normal(n) ** 2).mean() - exact) ** 2
                                  for _ in range(200)])) for n in ns])
plt.loglog(ns, rmse, "o-", label="observed RMSE (200 repetitions)")
plt.loglog(ns, f.std() / np.sqrt(ns), "k--", label=r"theory $\sigma_f/\sqrt{n}$")
plt.xlabel("number of samples n"); plt.ylabel("root-mean-square error")
plt.title(r"Monte Carlo error of $\mathbb{E}[e^{-X^2}]$ falls like $1/\sqrt{n}$"); plt.legend(); plt.show()
for n, r in zip(ns, rmse):
    print(f"n = {n:>7d}: RMSE {r:.5f}")
""")

nb.md(r"""
Every factor of 100 in $n$ buys one more decimal digit. The whole module is about the two words the
estimator takes for granted — "$X_i \sim p$" — which are easy for a Gaussian and hard for a posterior
or an image distribution.
""")

# ---------------------------------------------------------------- 2. exact samplers
nb.md(r"""
## 2. Exact samplers from uniform random numbers

**Inverse CDF:** if $U \sim \text{Uniform}(0,1)$ then $X = F^{-1}(U)$ has CDF $F$. For the Exponential($\lambda$),
$X = -\ln(1 - U)/\lambda$. **Discrete:** return the first index whose cumulative probability exceeds $U$
(`np.searchsorted` on the cumulative sums). **Box–Muller:** $Z_1 = \sqrt{-2\ln U_1}\cos 2\pi U_2$,
$Z_2 = \sqrt{-2\ln U_1}\sin 2\pi U_2$ are independent standard normals. **Reparameterisation:**
$X = \mu + \sigma Z \sim \mathcal{N}(\mu, \sigma^2)$ — the randomness lives in $Z$, the parameters enter
through a differentiable formula.
""")

nb.code(r"""
n = 20_000
u = rng.random(n)

# --- inverse CDF for the Exponential(lambda = 2)
lam = 2.0
x_exp = -np.log(1 - u) / lam
print(f"inverse CDF, u = 0.3 -> x = {-np.log(1 - 0.3) / lam:.4f};  u = 0.9 -> x = {-np.log(1 - 0.9) / lam:.4f}")
print(f"Exponential(2): sample mean {x_exp.mean():.4f} (exact 0.5), sample var {x_exp.var():.4f} (exact 0.25)")

# --- discrete sampling by cumulative sums
p = np.array([0.2, 0.5, 0.3])
cdf = np.cumsum(p)
k = np.searchsorted(cdf, u)                 # smallest k with cdf[k] >= u
freq = np.bincount(k, minlength=3) / n
print(f"cumulative sums {cdf};  u = 0.65 -> category {np.searchsorted(cdf, 0.65)}")
print(f"discrete sampler frequencies {freq.round(4)} vs p = {p}")

# --- Box-Muller
u1, u2 = rng.random(n // 2), rng.random(n // 2)
r = np.sqrt(-2 * np.log(u1))
z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
print(f"Box-Muller with u1 = 0.5, u2 = 0.25: R = {np.sqrt(-2 * np.log(0.5)):.4f}, z1 = 0, z2 = {np.sqrt(-2 * np.log(0.5)):.4f}")
print(f"Box-Muller normals: mean {z.mean():.4f}, var {z.var():.4f}, P(|Z| > 1.96) = {np.mean(np.abs(z) > 1.96):.4f} (exact 0.05)")

# --- reparameterisation: N(3, 2^2) from the same z
x_rep = 3 + 2 * z
print(f"3 + 2 Z: mean {x_rep.mean():.4f}, sd {x_rep.std():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
grid = np.linspace(0, 3, 200)
axes[0].hist(x_exp, bins=60, density=True, alpha=0.6, label="inverse-CDF samples")
axes[0].plot(grid, lam * np.exp(-lam * grid), "r", label=r"$2e^{-2x}$")
axes[0].set_xlabel("x"); axes[0].set_ylabel("density"); axes[0].set_title("Exponential(2) via inverse CDF"); axes[0].legend()
axes[1].bar(np.arange(3) - 0.2, p, width=0.4, label="p")
axes[1].bar(np.arange(3) + 0.2, freq, width=0.4, label="sample frequency")
axes[1].set_xlabel("category"); axes[1].set_ylabel("probability"); axes[1].set_title("Discrete sampling via cumulative sums"); axes[1].legend()
grid = np.linspace(-4, 4, 200)
axes[2].hist(z, bins=60, density=True, alpha=0.6, label="Box-Muller")
axes[2].plot(grid, stats.norm.pdf(grid), "r", label="N(0,1) density")
axes[2].set_xlabel("z"); axes[2].set_ylabel("density"); axes[2].set_title("Box-Muller normals"); axes[2].legend()
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Three different distributions, one source of randomness: a stream of uniforms. The discrete sampler is
exactly what `rng.choice(3, p=p)` does, and what a language model's decoder does over a 100,000-word
vocabulary after the softmax. The reparameterised draw `3 + 2 * z` is the formula a VAE differentiates
through in Section 9.

**Try it yourself:** replace `-np.log(1 - u)` by `-np.log(u)` — the histogram does not change, because
$1 - U$ is also uniform. Then sample a Uniform(2, 5) and a Cauchy ($F^{-1}(u) = \tan(\pi(u - 1/2))$) by
inverse CDF and look at their sample means as $n$ grows.
""")

# ---------------------------------------------------------------- 3. rejection sampling
nb.md(r"""
## 3. Rejection sampling: Beta(2, 5) under a uniform envelope

Target $p(x) = 30\,x(1-x)^4$ on $[0, 1]$, proposal $q = \text{Uniform}(0,1)$. The smallest valid envelope
constant is the maximum of $p$, reached at $x^* = 0.2$: $M = p(0.2) = 2.4576$. Propose $x \sim q$,
$u \sim \text{Uniform}(0, 1)$, accept if $u \le p(x)/(Mq(x))$. The acceptance rate is $1/M \approx 0.407$ and
accepted samples have density exactly $p$ — the picture below is the proof: dots are uniform under the
envelope, and the kept dots are uniform under $p$.
""")

nb.code(r"""
def beta25_pdf(x):
    return 30 * x * (1 - x) ** 4

M = beta25_pdf(0.2)
n_prop = 100_000
x_prop = rng.random(n_prop)
u_acc = rng.random(n_prop)
accepted = u_acc <= beta25_pdf(x_prop) / M
x_acc = x_prop[accepted]
print(f"envelope constant M = {M:.4f}  ->  predicted acceptance rate 1/M = {1 / M:.4f}")
print(f"observed acceptance rate {accepted.mean():.4f}  ({accepted.sum()} of {n_prop} proposals kept)")
print(f"mean of accepted samples {x_acc.mean():.4f} (exact 2/7 = {2 / 7:.4f}),  var {x_acc.var():.4f} (exact {2 * 5 / (49 * 8):.4f})")
print(f"single proposals: x = 0.5, u = 0.5 -> ratio {beta25_pdf(0.5) / M:.3f} -> rejected;  x = 0.15, u = 0.5 -> ratio {beta25_pdf(0.15) / M:.3f} -> accepted")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
show = slice(0, 3000)
ax1.scatter(x_prop[show][accepted[show]], (u_acc * M)[show][accepted[show]], s=3, label="accepted")
ax1.scatter(x_prop[show][~accepted[show]], (u_acc * M)[show][~accepted[show]], s=3, alpha=0.4, label="rejected")
grid = np.linspace(0, 1, 300)
ax1.plot(grid, beta25_pdf(grid), "k", lw=2, label="target p(x)")
ax1.axhline(M, color="r", ls="--", label="envelope M q(x)")
ax1.set_xlabel("x"); ax1.set_ylabel("u M q(x)"); ax1.set_title("Darts under the envelope; keep those under p"); ax1.legend(fontsize=8)
ax2.hist(x_acc, bins=50, density=True, alpha=0.6, label="accepted samples")
ax2.plot(grid, stats.beta.pdf(grid, 2, 5), "r", label="Beta(2,5) density")
ax2.set_xlabel("x"); ax2.set_ylabel("density"); ax2.set_title("Accepted samples follow the target exactly"); ax2.legend()
plt.tight_layout(); plt.show()

# why it fails in high dimension: target N(0, I_d), proposal N(0, sigma^2 I_d), best M = sigma^d
print("\nacceptance rate sigma^-d for proposal N(0, sigma^2 I) on target N(0, I):")
for sigma in (1.05, 1.1, 1.5):
    print(f"  sigma = {sigma}: " + "  ".join(f"d={d}: {sigma ** -d:.2e}" for d in (1, 10, 50, 100, 500)))
""")

nb.md(r"""
About 40.4% of 100,000 proposals survive, within noise of the predicted 40.7%, and the histogram sits on
the Beta(2, 5) density. The table is the bad news: a proposal only 10% wider than the target accepts one
proposal in 14,000 at $d = 100$, because the mismatch is raised to the power of the dimension. Rejection
sampling is a building block for one-dimensional distributions and for *speculative decoding* in LLM
serving, not a tool for posteriors.

**Try it yourself:** use $M = 4$ instead of the tight $2.4576$ — the accepted samples are still exactly
Beta(2, 5) distributed, but what happens to the acceptance rate? Then try $M = 2$ and compare the
histogram near $x = 0.2$ with the density: an invalid envelope clips the target.
""")

# ---------------------------------------------------------------- 4. importance sampling
nb.md(r"""
## 4. Importance sampling: a tail probability with weight diagnostics

$\mathbb{E}_p[f] = \mathbb{E}_q[f\,p/q]$. We estimate $\mu = P(X > 4)$ for $X \sim \mathcal{N}(0, 1)$
(exact $\Phi(-4) = 3.17 \times 10^{-5}$), where plain Monte Carlo with $n = 10{,}000$ expects $0.3$ hits.
Proposals: $\mathcal{N}(4, 1)$ centred on the boundary, with weights $w = \phi(x)/\phi(x - 4) = e^{8 - 4x}$;
a shifted exponential $x = 4 + Y$, $Y \sim \text{Exponential}(4)$, which matches the shape of the tail; and, as a
warning, a *light-tailed* proposal $\mathcal{N}(0, 0.6^2)$ for the simpler integrand $f(x) = x^2$. The effective
sample size $\text{ESS} = (\sum w_i)^2 / \sum w_i^2$ tells us how many "clean" samples each estimate rests on.
""")

nb.code(r"""
def ess(w):
    w = np.asarray(w, dtype=float)
    return w.sum() ** 2 / (w ** 2).sum()

mu_true = stats.norm.sf(4)
n = 10_000
print(f"target P(X > 4) = {mu_true:.4e}")

# plain Monte Carlo
hits = (rng.standard_normal(n) > 4).sum()
print(f"plain MC: {hits} hits in {n} samples -> estimate {hits / n:.1e}, relative se per sample {np.sqrt((1 - mu_true) / mu_true):.0f}")

# proposal 1: N(4, 1)
x1 = rng.normal(4, 1, n)
w1 = np.exp(8 - 4 * x1)                      # phi(x) / phi(x - 4)
fw1 = (x1 > 4) * w1
se1 = fw1.std() / np.sqrt(n)
print(f"IS with N(4,1): estimate {fw1.mean():.4e} +/- {se1:.1e}   ESS of f*w = {ess(fw1):.0f}   "
      f"max weight share {fw1.max() / fw1.sum():.4f}   (ESS of the raw weights w: {ess(w1):.1f})")

# proposal 2: shifted exponential, x = 4 + Y, Y ~ Exp(rate 4)
y = rng.exponential(1 / 4, n)
x2 = 4 + y
w2 = stats.norm.pdf(x2) / (4 * np.exp(-4 * y))
print(f"IS with 4 + Exp(4): estimate {w2.mean():.4e} +/- {w2.std() / np.sqrt(n):.1e}   ESS = {ess(w2):.0f}   max weight share {w2.max() / w2.sum():.4f}")
print(f"variance reduction vs plain MC (N(4,1) proposal): factor {(mu_true * (1 - mu_true)) / fw1.var():.0f}")

# the danger: light-tailed proposal N(0, 0.6^2) for E[X^2] under N(0,1) (exact 1)
x3 = rng.normal(0, 0.6, n)
w3 = stats.norm.pdf(x3) / stats.norm.pdf(x3, 0, 0.6)
print(f"\nlight-tailed proposal for E[X^2]: estimate {np.mean(w3 * x3 ** 2):.4f} (exact 1.0)   ESS = {ess(w3):.0f} of {n}   "
      f"max weight share {w3.max() / w3.sum():.4f}   largest weight {w3.max():.1f}")
print("four weights (0.1, 0.1, 0.1, 0.7): ESS =", round(ess([0.1, 0.1, 0.1, 0.7]), 3))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
axes[0].hist(fw1[x1 > 4], bins=50, alpha=0.7)
axes[0].set_title("N(4,1) proposal: weights of the x > 4 samples"); axes[0].set_xlabel("f(x) w(x)"); axes[0].set_ylabel("count")
axes[1].hist(w2, bins=50, alpha=0.7, color="C2")
axes[1].set_title("shifted-exponential proposal: weights"); axes[1].set_xlabel("w(x)"); axes[1].set_ylabel("count")
axes[2].hist(w3, bins=50, alpha=0.7, color="C3")
axes[2].set_yscale("log"); axes[2].set_title("light-tailed proposal: a few huge weights"); axes[2].set_xlabel("w(x)"); axes[2].set_ylabel("count (log)")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The $\mathcal{N}(4, 1)$ proposal turns an impossible estimate into one with 2% relative error from 10,000
samples; its weights are small and well spread (ESS about 1,800), and the shifted exponential is better
still (ESS near 9,900) because its shape $e^{-4(x - 4)}$ follows the Gaussian tail. The light-tailed
proposal looks fine at first glance — the estimate is near 1 — but its ESS is a fraction of $n$ and the
weight histogram has a long right tail: a single sample in the far tail of $\mathcal{N}(0,1)$ carries a weight
in the thousands, and the variance of such an estimator is infinite. The raw-weight ESS of the first proposal
(about 9) is a reminder to compute ESS for the quantity you average, $f \cdot w$, when $f$ is zero on most
of the proposal's support.

**Try it yourself:** move the first proposal to $\mathcal{N}(3, 1)$ and to $\mathcal{N}(6, 1)$ and compare ESS
and standard error. Which is worse, undershooting or overshooting the boundary?
""")

# ---------------------------------------------------------------- 5. Markov chains
nb.md(r"""
## 5. Markov chains: the stationary distribution three ways

For the weather chain of the guide we find $\pi$ with $\pi = \pi P$ as the left eigenvector of $P$ for
eigenvalue 1, by power iteration $\pi^{(t+1)} = \pi^{(t)} P$ from a sunny start, and by simulating one long
trajectory and counting visits. The second eigenvalue, $|\lambda_2| = 0.473$, sets the speed at which the
start is forgotten, and we test detailed balance $\pi_i P_{ij} = \pi_j P_{ji}$ — which this chain fails,
although it is perfectly stationary.
""")

nb.code(r"""
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3],
              [0.2, 0.3, 0.5]])
states = ["sunny", "cloudy", "rainy"]

# (a) eigenvector: pi P = pi  <=>  P^T pi^T = pi^T
evals, evecs = np.linalg.eig(P.T)
k = np.argmin(np.abs(evals - 1))
pi = np.real(evecs[:, k]); pi = pi / pi.sum()
print("eigenvalues of P:", np.round(np.sort(np.abs(evals))[::-1], 4))
print("stationary distribution (eigenvector):", pi.round(5), "  exact (21, 13, 12)/46 =", (np.array([21, 13, 12]) / 46).round(5))
print("check pi P - pi =", np.abs(pi @ P - pi).max())

# (b) power iteration from a sunny day
dist = np.array([1.0, 0.0, 0.0])
dists = [dist]
for t in range(15):
    dist = dist @ P
    dists.append(dist)
dists = np.array(dists)
print("\nt :  " + "  ".join(f"{t:>6d}" for t in (0, 1, 2, 3, 5, 10, 15)))
for j, s in enumerate(states):
    print(f"{s:7s}" + "  ".join(f"{dists[t, j]:6.4f}" for t in (0, 1, 2, 3, 5, 10, 15)))

# (c) one long simulated trajectory
n_steps = 100_000
traj = np.empty(n_steps, dtype=int); s = 0
cum = np.cumsum(P, axis=1)
for t in range(n_steps):
    s = np.searchsorted(cum[s], rng.random())   # the discrete sampler of Section 2, row by row
    traj[t] = s
print("\nvisit frequencies in a 100,000-step trajectory:", (np.bincount(traj, minlength=3) / n_steps).round(4))

# detailed balance?
flows = pi[:, None] * P
print("\nprobability flows pi_i P_ij:\n", flows.round(4))
print("detailed balance pi_i P_ij = pi_j P_ji holds?", np.allclose(flows, flows.T), "   e.g. pi_1 P_12 =", round(flows[0, 1], 4), " pi_2 P_21 =", round(flows[1, 0], 4))

err = np.abs(dists - pi).sum(axis=1)
print("\nL1 distance to pi after t days, and the ratio err[t]/err[t-1] (should approach |lambda_2| = 0.4732):")
for t in (1, 2, 3, 5, 8, 12):
    print(f"  t = {t:2d}: distance {err[t]:.2e}   ratio {err[t] / err[t - 1]:.4f}")
""")

nb.md(r"""
All three routes agree on $\pi \approx (0.4565, 0.2826, 0.2609)$. The distance to $\pi$ shrinks by a factor
that settles at $|\lambda_2| = 0.473$ per day — after ten days the forecast has forgotten today's weather. The flow matrix
is not symmetric ($0.0913$ vs $0.0848$), so this chain is stationary but *not* reversible: detailed balance
is sufficient for stationarity, not necessary. Every sampler in the next sections is built by *imposing*
detailed balance with respect to the target, which is the easy direction.
""")

# ---------------------------------------------------------------- 6. Metropolis-Hastings
nb.md(r"""
## 6. Metropolis–Hastings on a bimodal target: three step sizes

Target (up to a constant): $\tilde p(x) = \tfrac12\mathcal{N}(x; -2, 0.6^2) + \tfrac12\mathcal{N}(x; 2, 0.6^2)$.
Random-walk proposal $x' = x + s\,\varepsilon$; accept with probability $\min\{1, \tilde p(x')/\tilde p(x)\}$,
computed in log space. We run 10,000 steps for $s \in \{0.2, 1, 5\}$ and look at the trace, the histogram,
the autocorrelation function $\rho_k$ and the effective sample size $n/\tau$ with
$\tau = 1 + 2\sum_k \rho_k$ (summed up to the first negative autocorrelation).
""")

nb.code(r"""
def log_bimodal(x):
    # log of 0.5 N(x; -2, 0.6^2) + 0.5 N(x; 2, 0.6^2), dropping the common constant
    return np.logaddexp(-(x + 2) ** 2 / (2 * 0.36), -(x - 2) ** 2 / (2 * 0.36))

def random_walk_mh(log_target, x0, step, n, rng):
    x = x0
    lp = log_target(x)
    chain = np.empty(n)
    accepted = 0
    for i in range(n):
        prop = x + step * rng.standard_normal()
        lp_prop = log_target(prop)
        if np.log(rng.random()) < lp_prop - lp:      # accept with probability min(1, p(prop)/p(x))
            x, lp = prop, lp_prop
            accepted += 1
        chain[i] = x
    return chain, accepted / n

def autocorr(x, max_lag):
    x = x - x.mean()
    denom = x @ x
    return np.array([x[:len(x) - k] @ x[k:] / denom for k in range(max_lag + 1)])

def ess_mcmc(x, max_lag=2000):
    # integrated autocorrelation time tau = 1 + 2 sum rho_k, summing until the first negative rho_k
    rho = autocorr(x, min(max_lag, len(x) - 1))
    tau = 1.0
    for k in range(1, len(rho)):
        if rho[k] < 0:
            break
        tau += 2 * rho[k]
    return len(x) / tau, tau

n_mh = 10_000
steps = (0.2, 1.0, 5.0)
chains = {}
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
grid = np.linspace(-4.5, 4.5, 400)
target_pdf = 0.5 * stats.norm.pdf(grid, -2, 0.6) + 0.5 * stats.norm.pdf(grid, 2, 0.6)
for row, s in enumerate(steps):
    chain, acc = random_walk_mh(log_bimodal, 0.0, s, n_mh, rng)
    chains[s] = chain
    n_eff, tau = ess_mcmc(chain)
    print(f"step {s:3.1f}: acceptance {acc:.3f}   mean {chain.mean():+.3f} (exact 0)   var {chain.var():.3f} (exact 4.36)   "
          f"fraction x > 0: {np.mean(chain > 0):.3f} (exact 0.5)   tau = {tau:6.1f}   ESS = {n_eff:6.0f}")
    axes[row, 0].plot(chain[:2000], lw=0.5); axes[row, 0].set_title(f"trace, step {s} (first 2000 of {n_mh})"); axes[row, 0].set_xlabel("iteration"); axes[row, 0].set_ylabel("x")
    axes[row, 1].hist(chain, bins=60, density=True, alpha=0.6); axes[row, 1].plot(grid, target_pdf, "r"); axes[row, 1].set_title(f"histogram, step {s}"); axes[row, 1].set_xlabel("x"); axes[row, 1].set_ylabel("density")
    axes[row, 2].plot(autocorr(chain, 100)); axes[row, 2].axhline(0, color="k", lw=0.5); axes[row, 2].set_title(f"autocorrelation, step {s}"); axes[row, 2].set_xlabel("lag k"); axes[row, 2].set_ylabel(r"$\rho_k$")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Three regimes in one picture. Step $0.2$: 90% acceptance and a beautifully smooth trace — but the chain
crosses the gap between the modes only a handful of times in 10,000 steps, so the two bumps in its
histogram have the wrong weights, the autocorrelation barely decays within 100 lags, and the ESS is in
the single digits: ten thousand draws worth fewer than ten. Step $1$: the chain crosses more often (about a
hundred times), but each crossing is a slow diffusion through the low-density valley, so the autocorrelation
stays high and the ESS is still only around a hundred. Step $5$: only a quarter of proposals are accepted
(the flat stretches in the trace), but each accepted move is a genuine jump, often straight into the other
mode; the autocorrelation dies within a few lags and the ESS is over a thousand. Acceptance rate alone is a
terrible diagnostic — the worst chain accepts most happily. Trace plots catch a stuck chain only if you
*know* there is another mode; several chains from dispersed starts (the $\hat R$ idea) catch it without that
knowledge.

**Try it yourself:** run the step-0.2 chain for only 2,000 steps from `x0 = -2.0` and again from
`x0 = 2.0` and overlay the two histograms — the two "posteriors" disagree completely, which is exactly what
$\hat R$ measures.
""")

nb.md(r"""
### A Bayesian logistic regression by MH

Forty synthetic points with $P(y = 1 \mid x) = \sigma(\beta_0 + \beta_1 x)$, true $(\beta_0, \beta_1) = (-0.5, 1.5)$,
and a weak $\mathcal{N}(0, 10^2)$ prior on each coefficient. The log posterior is the log-likelihood
$\sum_i [y_i z_i - \log(1 + e^{z_i})]$ with $z_i = \beta_0 + \beta_1 x_i$, plus the log prior. A two-dimensional
random-walk MH chain of 20,000 steps gives posterior histograms that we compare with the maximum-likelihood
estimate (found with `scipy.optimize.minimize`).
""")

nb.code(r"""
from scipy.optimize import minimize

n_data = 40
x_data = rng.normal(0, 1.5, n_data)
beta_true = np.array([-0.5, 1.5])
y_data = (rng.random(n_data) < 1 / (1 + np.exp(-(beta_true[0] + beta_true[1] * x_data)))).astype(float)
X_design = np.column_stack([np.ones(n_data), x_data])

def log_lik(beta):
    z = X_design @ beta
    return np.sum(y_data * z - np.logaddexp(0, z))

def log_post(beta):
    return log_lik(beta) - 0.5 * np.sum(beta ** 2) / 10 ** 2        # N(0, 10^2) prior on each coefficient

mle = minimize(lambda b: -log_lik(b), np.zeros(2)).x

def random_walk_mh_vec(log_target, x0, step, n, rng):
    x = np.array(x0, dtype=float); lp = log_target(x)
    chain = np.empty((n, len(x))); accepted = 0
    for i in range(n):
        prop = x + step * rng.standard_normal(len(x))
        lp_prop = log_target(prop)
        if np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop; accepted += 1
        chain[i] = x
    return chain, accepted / n

chain_lr, acc = random_walk_mh_vec(log_post, np.zeros(2), 0.9, 20_000, rng)
burn = 2_000
post = chain_lr[burn:]
print(f"acceptance rate {acc:.3f};  burn-in {burn} discarded")
for j, name in enumerate(["intercept", "slope"]):
    print(f"{name:9s}: MLE {mle[j]:+.3f}   posterior mean {post[:, j].mean():+.3f}   posterior sd {post[:, j].std():.3f}   "
          f"95% interval [{np.quantile(post[:, j], 0.025):+.3f}, {np.quantile(post[:, j], 0.975):+.3f}]   ESS {ess_mcmc(post[:, j])[0]:.0f}")
print(f"P(slope > 1 | data) = {np.mean(post[:, 1] > 1):.3f}    P(slope > 0 | data) = {np.mean(post[:, 1] > 0):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for j, name in enumerate(["intercept beta_0", "slope beta_1"]):
    axes[j].hist(post[:, j], bins=60, density=True, alpha=0.6, label="MH posterior samples")
    axes[j].axvline(mle[j], color="r", ls="--", label="MLE")
    axes[j].axvline(beta_true[j], color="k", ls=":", label="true value")
    axes[j].set_xlabel(name); axes[j].set_ylabel("posterior density"); axes[j].set_title(f"posterior of the {name.split()[0]}"); axes[j].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The posterior means sit near the MLE but are pulled *away from zero* — with 40 points and a steep true
slope the likelihood is flat on the high side, and the posterior mean of the slope lands above the MLE while
its histogram is visibly right-skewed, a shape no Gaussian approximation around the MLE would show. The
posterior standard deviations are the Bayesian counterpart of the MLE's standard errors. Questions like
$P(\beta_1 > 1 \mid \text{data})$ are plain averages of indicators over the chain. The ESS tells you how many
independent draws your 18,000 correlated samples are worth (a few thousand here, at an acceptance rate near
the two-dimensional optimum); a hierarchical model with 30 groups uses exactly this machinery in 32 dimensions.
""")

# ---------------------------------------------------------------- 7. Gibbs and Langevin
nb.md(r"""
## 7. Gibbs sampling and Langevin dynamics

**Gibbs** on a correlated Gaussian, $\rho = 0.9$: alternate $x_1 \sim \mathcal{N}(\rho x_2, 1 - \rho^2)$ and
$x_2 \sim \mathcal{N}(\rho x_1, 1 - \rho^2)$ (Gaussian conditioning, Module 09). Every move is accepted, but
the path zig-zags along the ridge and the lag-$k$ autocorrelation is $\rho^{2k}$.
**Langevin dynamics** needs only the score $\nabla_x \log p(x)$: $x_{t+1} = x_t + \frac{\epsilon}{2}\nabla\log p(x_t)
+ \sqrt{\epsilon}\,\xi_t$. We write the analytic score of the bimodal mixture of Section 6, check it against
finite differences of the log density, and run 3,000 particles in parallel.
""")

nb.code(r"""
# --- Gibbs for a 2-D Gaussian with correlation rho
rho = 0.9
n_gibbs = 5_000
gibbs = np.zeros((n_gibbs, 2)); x1, x2 = 2.0, 2.0
path = [(x1, x2)]
for i in range(n_gibbs):
    x1 = rho * x2 + np.sqrt(1 - rho ** 2) * rng.standard_normal(); path.append((x1, x2))
    x2 = rho * x1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(); path.append((x1, x2))
    gibbs[i] = (x1, x2)
path = np.array(path)
acf_g = autocorr(gibbs[:, 0], 6)
print("Gibbs: sample covariance\n", np.cov(gibbs.T).round(3))
print("lag-k autocorrelation of x1:", acf_g.round(3))
print("theory rho^(2k)            :", (rho ** (2 * np.arange(7))).round(3))
print(f"tau = (1 + rho^2)/(1 - rho^2) = {(1 + rho ** 2) / (1 - rho ** 2):.2f}  ->  ESS of {n_gibbs} sweeps ~ {n_gibbs * (1 - rho ** 2) / (1 + rho ** 2):.0f}   (measured ESS {ess_mcmc(gibbs[:, 0])[0]:.0f})")

# --- Langevin dynamics with the analytic score of a Gaussian mixture
w_mix, m_mix, s_mix = np.array([0.5, 0.5]), np.array([-2.0, 2.0]), np.array([0.6, 0.6])

def mixture_logpdf(x, w, m, s):
    x = np.asarray(x, dtype=float)[..., None]
    comp = np.log(w) - 0.5 * np.log(2 * np.pi * s ** 2) - (x - m) ** 2 / (2 * s ** 2)
    return logsumexp(comp, axis=-1)

def mixture_score(x, w, m, s):
    # d/dx log sum_k w_k N(x; m_k, s_k^2) = sum_k r_k(x) * (-(x - m_k)/s_k^2), r_k = responsibilities
    x = np.asarray(x, dtype=float)[..., None]
    comp = np.log(w) - 0.5 * np.log(2 * np.pi * s ** 2) - (x - m) ** 2 / (2 * s ** 2)
    resp = np.exp(comp - logsumexp(comp, axis=-1, keepdims=True))
    return np.sum(resp * (-(x - m) / s ** 2), axis=-1)

pts = np.array([-3.0, -1.0, 0.3, 2.5]); h = 1e-5
fd = (mixture_logpdf(pts + h, w_mix, m_mix, s_mix) - mixture_logpdf(pts - h, w_mix, m_mix, s_mix)) / (2 * h)
print("\nscore at", pts, ": analytic", mixture_score(pts, w_mix, m_mix, s_mix).round(4), " finite differences", fd.round(4))

eps = 0.05
particles = 3 * rng.standard_normal(3_000)        # start far from the modes
for t in range(800):
    particles = particles + 0.5 * eps * mixture_score(particles, w_mix, m_mix, s_mix) + np.sqrt(eps) * rng.standard_normal(len(particles))
print(f"Langevin after 800 steps: fraction x > 0 = {np.mean(particles > 0):.3f} (0.5), mean |x| = {np.mean(np.abs(particles)):.3f} (2.0), var = {particles.var():.3f} (target 4.36)")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].plot(path[:60, 0], path[:60, 1], "-o", ms=3, lw=0.8, color="C0", label="first 30 Gibbs sweeps")
ellipse_t = np.linspace(0, 2 * np.pi, 200); L = np.linalg.cholesky([[1, rho], [rho, 1]])
for r_ in (1, 2):
    e = r_ * L @ np.vstack([np.cos(ellipse_t), np.sin(ellipse_t)]); axes[0].plot(e[0], e[1], "k", lw=0.6)
axes[0].set_aspect("equal"); axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2"); axes[0].set_title("Gibbs zig-zags along the ridge (rho = 0.9)"); axes[0].legend(fontsize=8)
axes[1].plot(acf_g, "o-", label="measured"); axes[1].plot(rho ** (2 * np.arange(7)), "k--", label=r"$\rho^{2k}$")
axes[1].set_xlabel("lag k (sweeps)"); axes[1].set_ylabel("autocorrelation of x1"); axes[1].set_title("Gibbs autocorrelation"); axes[1].legend()
axes[2].hist(particles, bins=60, density=True, alpha=0.6, label="Langevin particles")
axes[2].plot(grid, target_pdf, "r", label="target mixture"); axes[2].set_xlabel("x"); axes[2].set_ylabel("density"); axes[2].set_title("Langevin dynamics with the analytic score"); axes[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Gibbs never rejects, yet 5,000 sweeps are worth only about 500 independent draws because consecutive
samples share 81% of their variance ($\rho^2$). Rotating to the principal axes would make the coordinates
independent and the sampler exact in one sweep — correlation, not acceptance, is the enemy. The Langevin
particles, started from a wide cloud, settle onto both modes with the right weights after a few hundred
gradient-plus-noise steps, and the finite-difference check confirms the score formula. This update rule,
with a *learned* score and a slowly changing target, is the diffusion sampler of Section 10.

**Try it yourself:** set `eps = 0.5` in the Langevin loop — the particles still find both modes, but the
histogram is too wide (the discretisation bias grows with $\epsilon$); a Metropolis correction (MALA) would
remove it.
""")

# ---------------------------------------------------------------- 8. ELBO
nb.md(r"""
## 8. The ELBO on a conjugate toy problem: the gap *is* the KL

Model: $z \sim \mathcal{N}(0, 1)$, $x_i \mid z \sim \mathcal{N}(z, 1)$, data $x = (0.4, 1.6, 0.9, 1.1)$. The exact
posterior is $\mathcal{N}\big(\frac{n\bar x}{n+1}, \frac{1}{n+1}\big) = \mathcal{N}(0.8, 0.2)$ and the evidence
$p(x)$ is the density of $x$ under $\mathcal{N}(0, I + \mathbf{1}\mathbf{1}^\top)$. For a Gaussian
$q = \mathcal{N}(m, s^2)$ every term of
$\mathcal{L}(q) = \mathbb{E}_q[\log p(x \mid z)] + \mathbb{E}_q[\log p(z)] + \mathbb{H}(q)$ is available in closed form, so we
can draw the whole ELBO surface and check $\log p(x) - \mathcal{L}(q) = \text{KL}(q \,\|\, p(z \mid x))$ exactly.
Then we compare the two gradient estimators of Section 8 of the guide on $f(z) = z^2$.
""")

nb.code(r"""
x_obs = np.array([0.4, 1.6, 0.9, 1.1]); n_obs = len(x_obs)
post_mean, post_var = n_obs * x_obs.mean() / (n_obs + 1), 1 / (n_obs + 1)
log_evidence = stats.multivariate_normal(mean=np.zeros(n_obs), cov=np.eye(n_obs) + np.ones((n_obs, n_obs))).logpdf(x_obs)

def elbo(m, s2):
    e_loglik = -0.5 * n_obs * np.log(2 * np.pi) - 0.5 * (np.sum((x_obs - m) ** 2) + n_obs * s2)
    e_logprior = -0.5 * np.log(2 * np.pi) - 0.5 * (m ** 2 + s2)
    entropy = 0.5 * np.log(2 * np.pi * np.e * s2)
    return e_loglik + e_logprior + entropy

def kl_gauss(m, s2, m2, s2_2):
    # KL( N(m, s2) || N(m2, s2_2) )
    return 0.5 * (np.log(s2_2 / s2) + (s2 + (m - m2) ** 2) / s2_2 - 1)

print(f"exact posterior N({post_mean:.2f}, {post_var:.2f});  log evidence = {log_evidence:.4f}")
for m, s2 in [(0.8, 0.2), (1.0, 0.2), (0.8, 0.5), (0.0, 1.0)]:
    gap = log_evidence - elbo(m, s2)
    print(f"q = N({m:.1f}, {s2:.1f}): ELBO = {elbo(m, s2):.4f}   gap = {gap:.4f}   KL(q || posterior) = {kl_gauss(m, s2, post_mean, post_var):.4f}")

# Monte Carlo check of the ELBO at one q, via a single-sample estimator averaged over many samples
m, s2 = 1.0, 0.2
z = m + np.sqrt(s2) * rng.standard_normal(100_000)
loglik = np.sum(stats.norm.logpdf(x_obs[None, :], z[:, None], 1), axis=1)
mc_elbo = np.mean(loglik + stats.norm.logpdf(z) - stats.norm.logpdf(z, m, np.sqrt(s2)))
print(f"Monte Carlo ELBO at q = N(1.0, 0.2): {mc_elbo:.4f}  (closed form {elbo(m, s2):.4f})")

M_grid, S_grid = np.meshgrid(np.linspace(0.2, 1.4, 121), np.linspace(0.05, 0.8, 121))
E_grid = np.vectorize(elbo)(M_grid, S_grid)
plt.contourf(M_grid, S_grid, E_grid, levels=30); plt.colorbar(label="ELBO")
plt.plot(post_mean, post_var, "r*", ms=14, label=f"exact posterior (ELBO = log p(x) = {log_evidence:.3f})")
plt.xlabel("variational mean m"); plt.ylabel("variational variance s^2"); plt.title("The ELBO is maximised at the true posterior"); plt.legend(loc="upper right"); plt.show()

# score-function vs reparameterisation gradient of E_q[z^2], q = N(mu, 1), at mu = 1 (exact gradient 2 mu = 2)
mu = 1.0
eps_ = rng.standard_normal(200_000)
g_reparam = 2 * (mu + eps_)                 # d/dmu (mu + eps)^2
z_ = mu + eps_
g_score = z_ ** 2 * (z_ - mu)               # f(z) * d/dmu log N(z; mu, 1)
print(f"\ngradient of E[z^2] wrt mu at mu = 1 (exact 2):  reparameterisation mean {g_reparam.mean():.3f}, variance {g_reparam.var():.2f};"
      f"   score-function mean {g_score.mean():.3f}, variance {g_score.var():.2f}")
""")

nb.md(r"""
The gap between $\log p(x)$ and the ELBO equals the KL divergence from $q$ to the exact posterior in every
row, to all printed digits, and the surface peaks exactly at the posterior, where the bound is tight. Note
the *shape* of the surface: it falls off much faster when $s^2$ is too large than when it is too small —
the reverse KL is forgiving of an under-dispersed $q$, which is the mode-seeking bias in action. The two
gradient estimators have the same mean, but the score-function version has about eight times the
variance; with a neural-network decoder in place of $z^2$ the ratio is far worse, which is why VAEs use the
reparameterisation trick.
""")

# ---------------------------------------------------------------- 9. VAE
nb.md(r"""
## 9. A linear VAE in NumPy

Data: 400 points on a noisy line in the plane, $x = w z + b + 0.4\,\eta$. Encoder $\mu(x) = Ex + e$ with a
learned global $\log\sigma_z$; decoder $\hat x(z) = Dz + d$ with a learned $\log\sigma_x$; prior $\mathcal{N}(0, 1)$.
Loss per point: $\frac{\|x - \hat x\|^2}{2\sigma_x^2} + 2\log\sigma_x + \log 2\pi$ (Gaussian reconstruction)
$+\ \frac12(\mu^2 + \sigma_z^2 - 1 - \log\sigma_z^2)$ (closed-form KL), with $z = \mu + \sigma_z\varepsilon$.
The gradients are written by hand (chain rule through the reparameterised sample) and verified by finite
differences, then Adam trains the six parameter blocks for 1,500 steps.
""")

nb.code(r"""
n_pts = 400
z_true = rng.standard_normal(n_pts)
X_vae = np.outer(z_true, [2.0, 1.0]) + np.array([1.0, -0.5]) + 0.4 * rng.standard_normal((n_pts, 2))

def init_vae(rng):
    return dict(E=0.1 * rng.standard_normal((1, 2)), e=np.zeros(1), log_sz=np.zeros(1),
                D=0.1 * rng.standard_normal((2, 1)), d=np.zeros(2), log_sx=np.zeros(1))

def vae_loss_and_grads(p, X, eps):
    n = X.shape[0]
    mu = X @ p["E"].T + p["e"]                       # (n, 1) encoder mean
    sz = np.exp(p["log_sz"])                         # encoder sd (shared)
    z = mu + sz * eps                                # reparameterised sample, (n, 1)
    x_hat = z @ p["D"].T + p["d"]                    # (n, 2) decoder mean
    r = x_hat - X
    sx2 = np.exp(2 * p["log_sx"])
    recon = 0.5 * np.sum(r ** 2, axis=1) / sx2 + 2 * p["log_sx"] + np.log(2 * np.pi)   # -log N(x; x_hat, sx^2 I)
    kl = 0.5 * (mu[:, 0] ** 2 + sz ** 2 - 1 - 2 * p["log_sz"])                      # KL(N(mu, sz^2) || N(0, 1))
    loss = np.mean(recon + kl)
    # --- backward pass (mean over the batch) ---
    d_xhat = r / sx2 / n                             # d loss / d x_hat
    g = {}
    g["D"] = d_xhat.T @ z
    g["d"] = d_xhat.sum(axis=0)
    g["log_sx"] = np.array([np.mean(-np.sum(r ** 2, axis=1) / sx2 + 2)])
    d_z = d_xhat @ p["D"]                            # (n, 1) through the decoder
    d_mu = d_z + mu / n                              # reconstruction path + KL term mu^2/2
    g["E"] = d_mu.T @ X
    g["e"] = d_mu.sum(axis=0)
    g["log_sz"] = np.array([np.sum(d_z * sz * eps) + (sz ** 2 - 1)[0]])   # chain rule: dz/dlog_sz = sz*eps; KL: sz^2 - 1
    return loss, g, recon.mean(), kl.mean()

params = init_vae(rng)
eps0 = rng.standard_normal((n_pts, 1))
loss0, grads0, _, _ = vae_loss_and_grads(params, X_vae, eps0)
print("gradient check (first entry of each block):")
for k in params:
    h = 1e-6
    p_plus = {kk: v.copy() for kk, v in params.items()}; p_plus[k].flat[0] += h
    p_minus = {kk: v.copy() for kk, v in params.items()}; p_minus[k].flat[0] -= h
    num = (vae_loss_and_grads(p_plus, X_vae, eps0)[0] - vae_loss_and_grads(p_minus, X_vae, eps0)[0]) / (2 * h)
    print(f"  {k:7s} analytic {grads0[k].flat[0]:+.6f}   numeric {num:+.6f}")

def adam_step(params, grads, m, v, t, lr=0.02, b1=0.9, b2=0.999):
    for k in params:
        m[k] = b1 * m[k] + (1 - b1) * grads[k]
        v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
        params[k] -= lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + 1e-8)

m_adam = {k: np.zeros_like(v) for k, v in params.items()}; v_adam = {k: np.zeros_like(v) for k, v in params.items()}
history = []
for t in range(1, 1501):
    eps_t = rng.standard_normal((n_pts, 1))                       # fresh noise every step: single-sample ELBO estimate
    loss, grads, rec, kl = vae_loss_and_grads(params, X_vae, eps_t)
    adam_step(params, grads, m_adam, v_adam, t)
    history.append((loss, rec, kl))
history = np.array(history)
print(f"\nfinal loss {history[-1, 0]:.4f} = reconstruction {history[-1, 1]:.4f} + KL {history[-1, 2]:.4f}")
print(f"learned decoder direction D = {params['D'].ravel().round(3)}, offset d = {params['d'].round(3)}, sigma_x = {np.exp(params['log_sx'])[0]:.3f} (true noise 0.4), sigma_z = {np.exp(params['log_sz'])[0]:.3f}")
""")

nb.code(r"""
# compare with PCA and with the exact probabilistic-PCA marginal likelihood
Xc = X_vae - X_vae.mean(axis=0)
_, S, Vt = np.linalg.svd(Xc, full_matrices=False)
D_dir = params["D"].ravel() / np.linalg.norm(params["D"])
print(f"first principal component {Vt[0].round(4)}   decoder direction {D_dir.round(4)}   |cosine| = {abs(Vt[0] @ D_dir):.6f}")

# the decoder defines p(x) = N(d, D D^T + sigma_x^2 I) exactly (linear-Gaussian model) -> compare ELBO with the true log-likelihood
cov_model = params["D"] @ params["D"].T + np.exp(2 * params["log_sx"]) * np.eye(2)
avg_loglik = stats.multivariate_normal(mean=params["d"], cov=cov_model).logpdf(X_vae).mean()
print(f"average -ELBO per point {history[-1, 0]:.4f}   exact average -log p(x) of the learned decoder {-avg_loglik:.4f}   gap (= avg KL(q || true posterior)) {history[-1, 0] + avg_loglik:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].plot(history[:, 0], label="loss = -ELBO"); axes[0].plot(history[:, 1], label="reconstruction"); axes[0].plot(history[:, 2], label="KL")
axes[0].set_xlabel("Adam step"); axes[0].set_ylabel("nats per point"); axes[0].set_title("VAE training curves"); axes[0].legend()
z_prior = rng.standard_normal((300, 1)); x_gen = z_prior @ params["D"].T + params["d"] + np.exp(params["log_sx"]) * rng.standard_normal((300, 2))
axes[1].scatter(X_vae[:, 0], X_vae[:, 1], s=6, alpha=0.4, label="data")
axes[1].scatter(x_gen[:, 0], x_gen[:, 1], s=6, alpha=0.6, label="generated: decode z ~ N(0,1)")
axes[1].set_aspect("equal"); axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2"); axes[1].set_title("Samples from the trained VAE"); axes[1].legend(fontsize=8)
mu_enc = (X_vae @ params["E"].T + params["e"])[:, 0]
axes[2].scatter(z_true, mu_enc, s=6, alpha=0.5); axes[2].set_xlabel("true latent z"); axes[2].set_ylabel("encoder mean mu(x)"); axes[2].set_title("Encoder recovers the latent (up to sign/scale)")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The hand-written gradients match finite differences to six decimals, the decoder direction aligns with
the first principal component to a cosine of $0.99999$, and the learned $\sigma_x$ recovers the true
noise level. The linear-Gaussian VAE *is* probabilistic PCA, which is why we can compute the exact
marginal likelihood of the learned decoder and see that the ELBO sits only a few hundredths of a nat below
it — the amortised Gaussian encoder with a shared variance is nearly the exact posterior here. Generated
points (decode a prior sample, add decoder noise) land on the data cloud. The KL term stays well above
zero — the code carries about 1.8 nats of information per point — so there is no posterior collapse.

**Try it yourself:** multiply the KL term by `beta = 5` in the loss (a $\beta$-VAE) and watch
$\sigma_z$ grow toward 1 while the reconstruction worsens; with `beta = 50` the KL goes to zero and the
generated points ignore the data — posterior collapse on demand.
""")

# ---------------------------------------------------------------- 10. diffusion
nb.md(r"""
## 10. A one-dimensional diffusion model

Data: two modes at $\pm 2$ with standard deviation $0.3$. Schedule: $T = 50$, $\beta_t$ linear from $10^{-3}$ to
$0.2$, so $\bar\alpha_T \approx 0.0045$ — the endpoint is (almost) pure noise. We check the closed-form
marginal $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, 1 - \bar\alpha_t)$ against simulation of the
step-by-step chain, then sample backwards with the **exact** score — available because a Gaussian mixture
stays a Gaussian mixture under Gaussian noise: at level $t$ the marginal has means $\sqrt{\bar\alpha_t}\mu_k$ and
variances $\bar\alpha_t\sigma_k^2 + 1 - \bar\alpha_t$ — using $\varepsilon = -\sqrt{1 - \bar\alpha_t}\,\nabla\log q_t$
and the DDPM update $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\varepsilon\big) + \sigma_t\xi$.
""")

nb.code(r"""
T = 50
betas = np.linspace(1e-3, 0.2, T)
alphas = 1 - betas
alpha_bar = np.cumprod(alphas)
print(f"alpha_bar at t = 1, 10, 25, 50: {alpha_bar[0]:.4f}, {alpha_bar[9]:.4f}, {alpha_bar[24]:.4f}, {alpha_bar[-1]:.4f}")
print(f"signal-to-noise ratio alpha_bar/(1 - alpha_bar) at t = 10, 25, 50: {alpha_bar[9] / (1 - alpha_bar[9]):.2f}, {alpha_bar[24] / (1 - alpha_bar[24]):.2f}, {alpha_bar[-1] / (1 - alpha_bar[-1]):.4f}")

# worked example from the guide: beta = (0.1, 0.2, 0.3), x0 = 2, x3 = 1.0
b3 = np.array([0.1, 0.2, 0.3]); ab3 = np.cumprod(1 - b3)
eps_hat = (1.0 - np.sqrt(ab3[2]) * 2) / np.sqrt(1 - ab3[2])
mu_rev = (1.0 - b3[2] / np.sqrt(1 - ab3[2]) * eps_hat) / np.sqrt(1 - b3[2])
print(f"guide example: alpha_bar = {ab3.round(3)}, implied eps = {eps_hat:.3f}, reverse mean = {mu_rev:.3f}, beta_tilde_3 = {(1 - ab3[1]) / (1 - ab3[2]) * b3[2]:.3f}")

# data and forward process
n_data = 2_000
w_d, m_d, s_d = np.array([0.5, 0.5]), np.array([-2.0, 2.0]), np.array([0.3, 0.3])
comp = rng.random(n_data) < 0.5
x0 = np.where(comp, m_d[0], m_d[1]) + 0.3 * rng.standard_normal(n_data)

def q_sample(x0, t, eps):
    # closed-form marginal: x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps  (t is a 0-based index)
    return np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps

# step-by-step chain vs one-shot marginal at t = 25
x_chain = x0.copy()
for t in range(25):
    x_chain = np.sqrt(alphas[t]) * x_chain + np.sqrt(betas[t]) * rng.standard_normal(n_data)
x_oneshot = q_sample(x0, 24, rng.standard_normal(n_data))
print(f"\nt = 25: step-by-step chain mean {x_chain.mean():+.3f}, var {x_chain.var():.3f};  one-shot marginal mean {x_oneshot.mean():+.3f}, var {x_oneshot.var():.3f};"
      f"  theory var = alpha_bar Var(x0) + 1 - alpha_bar = {alpha_bar[24] * x0.var() + 1 - alpha_bar[24]:.3f}")

fig, axes = plt.subplots(1, 5, figsize=(14, 2.8), sharey=True)
for ax, t in zip(axes, (0, 5, 15, 30, 49)):
    xt = x0 if t == 0 else q_sample(x0, t, rng.standard_normal(n_data))
    ax.hist(xt, bins=50, density=True, alpha=0.7, range=(-5, 5)); ax.set_title(f"t = {t + 1 if t else 0}, alpha_bar = {alpha_bar[t] if t else 1:.3f}"); ax.set_xlabel("x")
axes[0].set_ylabel("density"); fig.suptitle("Forward diffusion: the two modes melt into a standard normal"); plt.tight_layout(); plt.show()
""")

nb.code(r"""
def score_marginal(x, t):
    # exact score of q_t(x) = sum_k w_k N(x; sqrt(ab_t) m_k, ab_t s_k^2 + 1 - ab_t)
    means = np.sqrt(alpha_bar[t]) * m_d
    var = alpha_bar[t] * s_d ** 2 + (1 - alpha_bar[t])
    return mixture_score(x, w_d, means, np.sqrt(var))

def eps_from_score(x, t):
    return -np.sqrt(1 - alpha_bar[t]) * score_marginal(x, t)

def ancestral_sample(eps_model, n, rng):
    x = rng.standard_normal(n)                      # x_T ~ N(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = eps_model(x, t)
        mean = (x - betas[t] / np.sqrt(1 - alpha_bar[t]) * eps_hat) / np.sqrt(alphas[t])
        if t > 0:
            sigma_t = np.sqrt((1 - alpha_bar[t - 1]) / (1 - alpha_bar[t]) * betas[t])   # beta_tilde_t
            x = mean + sigma_t * rng.standard_normal(n)
        else:
            x = mean                                # no noise at the last step
    return x

x_gen_exact = ancestral_sample(eps_from_score, 4_000, rng)
print(f"samples with the exact score: fraction > 0 = {np.mean(x_gen_exact > 0):.3f} (0.5),  mean |x| = {np.mean(np.abs(x_gen_exact)):.3f} (2.0),  "
      f"sd within the right mode = {x_gen_exact[x_gen_exact > 0].std():.3f} (0.3)")

# the reverse chain, a few trajectories
n_traj = 8
traj = np.empty((T + 1, n_traj)); x = rng.standard_normal(n_traj); traj[T] = x
for t in range(T - 1, -1, -1):
    eps_hat = eps_from_score(x, t)
    mean = (x - betas[t] / np.sqrt(1 - alpha_bar[t]) * eps_hat) / np.sqrt(alphas[t])
    x = mean + (np.sqrt((1 - alpha_bar[t - 1]) / (1 - alpha_bar[t]) * betas[t]) * rng.standard_normal(n_traj) if t > 0 else 0)
    traj[t] = x
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
ax1.plot(np.arange(T + 1), traj, lw=1); ax1.invert_xaxis(); ax1.set_xlabel("diffusion step t (reverse: right to left)"); ax1.set_ylabel("x_t"); ax1.set_title("Reverse chain: noise becomes two modes")
grid_x = np.linspace(-4, 4, 300)
ax2.hist(x_gen_exact, bins=60, density=True, alpha=0.6, label="exact-score samples")
ax2.plot(grid_x, 0.5 * stats.norm.pdf(grid_x, -2, 0.3) + 0.5 * stats.norm.pdf(grid_x, 2, 0.3), "r", label="data density")
ax2.set_xlabel("x"); ax2.set_ylabel("density"); ax2.set_title("Ancestral sampling with the exact score"); ax2.legend()
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The one-shot marginal matches the step-by-step chain in mean and variance — that is the closed-form
$q(x_t \mid x_0)$, and it is what makes training cheap (any noise level in one draw). Run backwards with the
exact score, 50 small denoising steps turn standard-normal noise into the two data modes with the right
weights and widths. In a real model the score is not available; the next cell *learns* it.

### Learning the noise predictor

A small MLP $\varepsilon_\theta(x_t, t)$ (inputs: $x_t$ and a sinusoidal embedding of $t$; two hidden layers
of 64 tanh units) is trained on the simplified loss $\mathbb{E}\,\|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}x_0 +
\sqrt{1 - \bar\alpha_t}\varepsilon, t)\|^2$: for each mini-batch pick random data points, random steps $t$ and random
noise, corrupt in one shot, regress the noise. Forward and backward passes are written by hand
(Module 16), Adam does the updates.
""")

nb.code(r"""
def time_features(x, t):
    tt = (np.asarray(t, dtype=float) + 1) / T
    return np.column_stack([x, tt, np.sin(2 * np.pi * tt), np.cos(2 * np.pi * tt), np.sin(4 * np.pi * tt), np.cos(4 * np.pi * tt)])

H = 64
net = dict(W1=rng.standard_normal((6, H)) / np.sqrt(6), b1=np.zeros(H),
           W2=rng.standard_normal((H, H)) / np.sqrt(H), b2=np.zeros(H),
           W3=rng.standard_normal((H, 1)) / np.sqrt(H), b3=np.zeros(1))

def mlp_forward(p, F):
    h1 = np.tanh(F @ p["W1"] + p["b1"]); h2 = np.tanh(h1 @ p["W2"] + p["b2"])
    return (h2 @ p["W3"] + p["b3"])[:, 0], (F, h1, h2)

def mlp_backward(p, cache, d_out):
    F, h1, h2 = cache; d_out = d_out[:, None]
    g = {"W3": h2.T @ d_out, "b3": d_out.sum(0)}
    d_h2 = (d_out @ p["W3"].T) * (1 - h2 ** 2)
    g["W2"] = h1.T @ d_h2; g["b2"] = d_h2.sum(0)
    d_h1 = (d_h2 @ p["W2"].T) * (1 - h1 ** 2)
    g["W1"] = F.T @ d_h1; g["b1"] = d_h1.sum(0)
    return g

m_net = {k: np.zeros_like(v) for k, v in net.items()}; v_net = {k: np.zeros_like(v) for k, v in net.items()}
batch, losses = 256, []
for it in range(1, 2501):
    idx = rng.integers(0, n_data, batch)
    t_b = rng.integers(0, T, batch)
    eps_b = rng.standard_normal(batch)
    x_t = q_sample(x0[idx], t_b, eps_b)                       # one-shot corruption
    pred, cache = mlp_forward(net, time_features(x_t, t_b))
    losses.append(np.mean((pred - eps_b) ** 2))
    grads = mlp_backward(net, cache, 2 * (pred - eps_b) / batch)
    adam_step(net, grads, m_net, v_net, it, lr=3e-3)
print(f"noise-prediction loss: first 50 steps {np.mean(losses[:50]):.3f}  ->  last 200 steps {np.mean(losses[-200:]):.3f}")

eps_learned = lambda x, t: mlp_forward(net, time_features(x, np.full(len(x), t)))[0]
x_gen_learned = ancestral_sample(eps_learned, 4_000, rng)
print(f"samples with the learned denoiser: fraction > 0 = {np.mean(x_gen_learned > 0):.3f} (0.5),  mean |x| = {np.mean(np.abs(x_gen_learned)):.3f} (2.0),  "
      f"sd within the right mode = {x_gen_learned[x_gen_learned > 0].std():.3f} (0.3)")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].plot(np.convolve(losses, np.ones(50) / 50, mode="valid")); axes[0].set_xlabel("training step"); axes[0].set_ylabel("MSE (50-step moving average)"); axes[0].set_title("Simplified diffusion loss")
for t_show, c in ((5, "C0"), (24, "C1"), (45, "C2")):
    axes[1].plot(grid_x, eps_from_score(grid_x, t_show), c, label=f"exact, t = {t_show + 1}")
    axes[1].plot(grid_x, eps_learned(grid_x, t_show), c + "--", label=f"learned, t = {t_show + 1}")
axes[1].set_xlabel("x_t"); axes[1].set_ylabel("predicted noise eps(x_t, t)"); axes[1].set_title("Learned vs exact noise predictor"); axes[1].legend(fontsize=7)
axes[2].hist(x_gen_learned, bins=60, density=True, alpha=0.6, label="learned-denoiser samples")
axes[2].plot(grid_x, 0.5 * stats.norm.pdf(grid_x, -2, 0.3) + 0.5 * stats.norm.pdf(grid_x, 2, 0.3), "r", label="data density")
axes[2].set_xlabel("x"); axes[2].set_ylabel("density"); axes[2].set_title("Samples from the trained diffusion model"); axes[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The learned noise predictor tracks the exact one closely where the data live and deviates only in the far
tails that the training set never visits, and the samples it generates reproduce both modes with
approximately the right weights and widths. The loss cannot go to zero: at high noise levels the noise is
genuinely unpredictable from $x_t$, and the minimum achievable MSE is the posterior variance of
$\varepsilon$ given $x_t$. This is, in miniature, the entire training and sampling loop of a modern image
diffusion model; the differences are a U-Net or transformer in place of the 64-unit MLP, images in place of
scalars, and 1,000 steps in place of 50.

**Try it yourself:** sample with only every 5th step (keep `t` in `range(T-1, -1, -5)` and use
$\bar\alpha$ of the skipped-to step) — ancestral sampling degrades quickly, which is what DDIM's
deterministic update was designed to fix.
""")

# ---------------------------------------------------------------- 12. FID
nb.md(r"""
## 12. Evaluating generative models: the Fréchet distance between two Gaussians

$\text{FID} = \|\mu_r - \mu_g\|^2 + \operatorname{tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\big)$.
We implement it with `scipy.linalg.sqrtm`, verify the hand-worked values ($\mathcal{N}(0,1)$ vs $\mathcal{N}(1,4)$
gives 2; the 2-D example gives 2.25), estimate it from samples in five dimensions, and show the
finite-sample bias: two samples from the *same* distribution do not give zero. Finally we score the two
diffusion samplers of Section 10 against the data.
""")

nb.code(r"""
from scipy import linalg

def frechet_distance(mu1, S1, mu2, S2):
    S1, S2 = np.atleast_2d(S1), np.atleast_2d(S2)
    covmean = np.real(linalg.sqrtm(S1 @ S2))
    return float(np.sum((np.asarray(mu1) - np.asarray(mu2)) ** 2) + np.trace(S1 + S2 - 2 * covmean))

def fid_from_samples(A, B):
    A, B = np.atleast_2d(A.T).T, np.atleast_2d(B.T).T
    return frechet_distance(A.mean(0), np.cov(A.T), B.mean(0), np.cov(B.T))

print(f"N(0,1) vs N(1,4): {frechet_distance([0.0], [[1.0]], [1.0], [[4.0]]):.4f}   (hand: 1 + 1 = 2)")
print(f"2-D example:      {frechet_distance([0, 0], np.eye(2), [1, 0.5], np.diag([4.0, 1.0])):.4f}   (hand: 1.25 + 1 = 2.25)")

d = 5
A = rng.multivariate_normal(np.zeros(d), np.eye(d), 5_000)
B = rng.multivariate_normal(0.5 * np.ones(d), 1.5 * np.eye(d), 5_000)
A2 = rng.multivariate_normal(np.zeros(d), np.eye(d), 5_000)
print(f"5-D from 5,000 samples each: {fid_from_samples(A, B):.4f}   exact {frechet_distance(np.zeros(d), np.eye(d), 0.5 * np.ones(d), 1.5 * np.eye(d)):.4f}")
print(f"two samples from the SAME distribution: {fid_from_samples(A, A2):.4f}  (not zero: finite-sample bias)")
for n_s in (100, 1_000, 5_000):
    vals = [fid_from_samples(rng.standard_normal((n_s, d)), rng.standard_normal((n_s, d))) for _ in range(20)]
    print(f"   same-distribution FID with n = {n_s:>5d}: mean {np.mean(vals):.4f}")

# score the diffusion samplers against held-out data (1-D, so 'features' are the raw values)
x_test = np.where(rng.random(4_000) < 0.5, -2.0, 2.0) + 0.3 * rng.standard_normal(4_000)
print(f"\n1-D Frechet distance to fresh data: exact-score sampler {fid_from_samples(x_test[:, None], x_gen_exact[:, None]):.4f},"
      f"  learned denoiser {fid_from_samples(x_test[:, None], x_gen_learned[:, None]):.4f},  pure noise N(0,1) {fid_from_samples(x_test[:, None], rng.standard_normal(4_000)[:, None]):.4f}")
""")

nb.md(r"""
The formula reproduces the hand calculations exactly and recovers the true distance from samples to two
decimals, but two samples from the same distribution score a small positive number that shrinks like
$1/n$ — always compare models at the same sample size. Both diffusion samplers score far below the
pure-noise baseline; the exact-score sampler is better than the learned one, as it should be. Note what the
metric does *not* see: a Gaussian fit cannot tell a two-mode distribution from a one-mode one with the same
mean and variance, which is why precision/recall-style metrics exist alongside FID.
""")

nb.md(r"""
## Summary

- Any expectation becomes an average with error $\sigma_f/\sqrt{n}$ once you can sample from $p$; the module is about making "$X_i \sim p$" true.
- Uniforms are turned into exact samples by the inverse CDF, cumulative sums (the LLM decoder), Box–Muller and the reparameterisation $\mu + \sigma z$.
- Rejection sampling is exact with acceptance rate $1/M$ and collapses like $\sigma^{-d}$ in high dimension; importance sampling reweights instead, and its ESS $(\sum w)^2/\sum w^2$ is the diagnostic that catches light-tailed proposals.
- A Markov chain's stationary distribution solves $\pi = \pi P$; detailed balance implies stationarity (not conversely), and $|\lambda_2|$ sets the mixing speed.
- Metropolis–Hastings accepts with $\min\{1, \tilde p(x')q(x \mid x')/(\tilde p(x)q(x' \mid x))\}$ and satisfies detailed balance for any $\tilde p$; tune the step for 0.2–0.5 acceptance and judge by autocorrelation and ESS, never by acceptance rate alone.
- Gibbs updates exact conditionals (acceptance 1) but crawls along correlated ridges ($\rho_k = \rho^{2k}$); Langevin dynamics is gradient ascent on $\log p$ plus noise of variance $\epsilon$ and needs only the score.
- The ELBO is a lower bound on $\log p(x)$ whose gap is $\text{KL}(q \,\|\, p(z \mid x))$ — mode-seeking; the pathwise gradient has far lower variance than the score-function gradient.
- A VAE minimises reconstruction $+ \frac12\sum(\mu^2 + \sigma^2 - 1 - \log\sigma^2)$ through one reparameterised sample; the linear case is probabilistic PCA.
- Diffusion: $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1 - \bar\alpha_t)I)$, train on $\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$, sample by iterated denoising; the noise predictor is the score and each reverse step is a Langevin step.
- FID is the Wasserstein-2 distance between Gaussian fits in feature space; it is biased at small $n$ and blind to anything beyond means and covariances.

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
    "Module 18 — Exercises",
    intro_md=r"""
These exercises follow the order of the concept guide: exact samplers, rejection and importance sampling,
Markov chains, Metropolis–Hastings, the VAE's Gaussian KL, the diffusion forward process and the score, and
finally a Bayesian mini-project with MCMC. Several checks compare against *hidden* values of seeded
computations, so follow the specified recipe (which `rng` call, in which order) exactly.
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- 1. inverse CDF + discrete sampler (warm-up)
SOL_1 = r"""
def sample_exponential(lam, n, rng):
    u = rng.random(n)
    return -np.log(1 - u) / lam

def sample_discrete(p, n, rng):
    cdf = np.cumsum(np.asarray(p, dtype=float))
    u = rng.random(n)
    return np.searchsorted(cdf, u)
"""
ns1 = run(SOL_1, r"""
xs = sample_exponential(2.0, 50_000, np.random.default_rng(1))
m, v = xs.mean(), xs.var()
ks = sample_discrete([0.2, 0.5, 0.3], 50_000, np.random.default_rng(2))
freq = np.bincount(ks, minlength=3) / len(ks)
x_fixed = sample_exponential(2.0, 3, np.random.default_rng(7))
k_fixed = sample_discrete([0.1, 0.2, 0.3, 0.4], 5, np.random.default_rng(7))
""")
ex.add(
    title="Exact samplers: inverse CDF and cumulative sums", difficulty=1,
    prompt_md=r"""
Implement two samplers that use **only** `rng.random` (uniforms) plus NumPy arithmetic — no
`rng.exponential`, no `rng.choice`:

* `sample_exponential(lam, n, rng)` → array of `n` draws from Exponential($\lambda$) via the inverse CDF
  $X = -\ln(1 - U)/\lambda$, where `U = rng.random(n)` is drawn **once, in that form**;
* `sample_discrete(p, n, rng)` → array of `n` integer category indices for probabilities `p`, obtained
  by drawing `U = rng.random(n)` and returning the smallest `k` with `cumsum(p)[k] >= U`
  (`np.searchsorted` on the cumulative sums does exactly this).

Because the checks reproduce your draws with a fixed seed, use exactly one call `rng.random(n)` in each function.
""",
    starter=r"""
def sample_exponential(lam, n, rng):
    # YOUR CODE HERE  (u = rng.random(n); then transform)
    return None

def sample_discrete(p, n, rng):
    # YOUR CODE HERE  (cumulative sums + np.searchsorted)
    return None
""",
    solution=SOL_1,
    check=rf"""
try:
    xs = sample_exponential(2.0, 50_000, np.random.default_rng(1))
    m, v = float(np.mean(xs)), float(np.var(xs))
except Exception:
    xs, m, v = None, None, None
check('Exponential(2): sample mean within 2% of 1/2', m is not None and abs(m - 0.5) < 0.01)
check('Exponential(2): sample variance within 5% of 1/4', v is not None and abs(v - 0.25) < 0.0125)
check('Exponential(2): all samples positive', xs is not None and bool(np.all(np.asarray(xs) > 0)))
try:
    got = np.round(sample_exponential(2.0, 3, np.random.default_rng(7)), 4)
except Exception:
    got = None
check_hash('hidden: the first three Exponential(2) draws with seed 7', got, '{sha_of(np.round(ns1["x_fixed"], 4), decimals=4)}', decimals=4)
try:
    ks = np.asarray(sample_discrete([0.2, 0.5, 0.3], 50_000, np.random.default_rng(2)))
    freq = np.bincount(ks, minlength=3) / len(ks)
except Exception:
    ks, freq = None, None
check('discrete sampler: frequencies within 0.01 of p', freq is not None and bool(np.all(np.abs(freq - np.array([0.2, 0.5, 0.3])) < 0.01)))
check('discrete sampler returns integer indices in {{0, 1, 2}}', ks is not None and set(np.unique(ks).tolist()) <= {{0, 1, 2}})
try:
    got = np.asarray(sample_discrete([0.1, 0.2, 0.3, 0.4], 5, np.random.default_rng(7))).astype(int)
except Exception:
    got = None
check_hash('hidden: five categorical draws with seed 7', got, '{sha_of(ns1["k_fixed"].astype(int))}')
""",
    explanation_md=r"""
If $U$ is uniform then $P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$, so $X = F^{-1}(U)$ has CDF $F$; for the
Exponential, $F(x) = 1 - e^{-\lambda x}$ inverts to $-\ln(1 - u)/\lambda$. The discrete version is the same
statement for a step CDF: the interval $(c_{k-1}, c_k]$ has length $p_k$, so `searchsorted` returns category
$k$ with probability $p_k$. The common wrong turn is `np.searchsorted(cdf, u, side='right')`, which differs
from the default only on the measure-zero event $U = c_k$ but would change the hidden hash if it did occur;
another is using `rng.random()` inside a Python loop, which changes the order of draws.
""",
)

# ---------------------------------------------------------------- 2. rejection sampler (warm-up)
SOL_2 = r"""
def beta25_pdf(x):
    return 30 * x * (1 - x) ** 4

def rejection_beta25(n_proposals, rng):
    M = beta25_pdf(0.2)
    x = rng.random(n_proposals)
    u = rng.random(n_proposals)
    keep = u <= beta25_pdf(x) / M
    return x[keep], float(keep.mean())
"""
ns2 = run(SOL_2, r"""
samples, rate = rejection_beta25(200_000, np.random.default_rng(3))
M = beta25_pdf(0.2)
""")
ex.add(
    title="Rejection sampling for Beta(2, 5)", difficulty=1,
    prompt_md=r"""
The Beta(2, 5) density is $p(x) = 30\,x(1-x)^4$ on $[0, 1]$; its maximum is at $x^* = 0.2$. Implement

* `beta25_pdf(x)` — the density (vectorised);
* `rejection_beta25(n_proposals, rng)` — rejection sampling with the uniform proposal $q = \text{Uniform}(0,1)$
  and the **tightest** valid envelope constant $M = p(0.2)$. Draw `x = rng.random(n_proposals)` first, then
  `u = rng.random(n_proposals)`, accept where `u <= p(x) / M`, and return the tuple
  `(accepted_samples, acceptance_rate)` with the rate as a float.

Only NumPy; do not use `scipy.stats.beta` to generate samples.
""",
    starter=r"""
def beta25_pdf(x):
    # YOUR CODE HERE
    return None

def rejection_beta25(n_proposals, rng):
    # YOUR CODE HERE  (x = rng.random(n); u = rng.random(n); accept where u <= p(x)/M)
    return None, None
""",
    solution=SOL_2,
    check=rf"""
try:
    got = float(beta25_pdf(0.2))
except Exception:
    got = None
check_close('density at the mode p(0.2) = 2.4576', got, 2.4576, tol=1e-9)
try:
    got = float(beta25_pdf(0.5))
except Exception:
    got = None
check_close('density at 0.5 agrees with scipy', got, float(stats.beta.pdf(0.5, 2, 5)), tol=1e-9)
try:
    samples, rate = rejection_beta25(200_000, np.random.default_rng(3))
    samples = np.asarray(samples, dtype=float); rate = float(rate)
except Exception:
    samples, rate = None, None
check('acceptance rate within 0.01 of 1/M = 0.4069', rate is not None and abs(rate - 1 / 2.4576) < 0.01)
check('number of accepted samples matches the rate', samples is not None and abs(len(samples) - rate * 200_000) < 1)
check('accepted samples have mean within 0.01 of 2/7', samples is not None and abs(samples.mean() - 2 / 7) < 0.01)
check('accepted samples have variance within 10% of 10/392', samples is not None and abs(samples.var() - 10 / 392) < 0.1 * 10 / 392)
check_hash('hidden: exact acceptance rate for seed 3', (round(rate, 4) if rate is not None else None), '{sha_of(round(ns2["rate"], 4), decimals=4)}', decimals=4)
""",
    explanation_md=r"""
With a uniform proposal the envelope is a flat line at height $M$, and the tightest choice is the maximum
of the target: $p'(x) = 30(1-x)^3(1 - 5x) = 0$ at $x = 0.2$, $M = 2.4576$. Accepted points are uniform under
the curve $p$, so their $x$-coordinates have density $p$ exactly, and the fraction accepted is the area
under $p$ (one) over the area under the envelope ($M$), i.e. $1/M \approx 40.7\%$. A larger $M$ would still
be *valid* (the samples stay exactly Beta-distributed) but wasteful; a smaller one is *invalid* and clips the
density near the mode — the acceptance check `u <= p/M` would then saturate at 1 there.
""",
)

# ---------------------------------------------------------------- 3. importance sampling (core)
SOL_3 = r"""
def is_tail_estimate(a, n, shift, rng):
    x = rng.normal(shift, 1.0, n)                     # proposal N(shift, 1)
    w = np.exp(-0.5 * x ** 2 + 0.5 * (x - shift) ** 2)   # phi(x) / phi(x - shift)
    fw = (x > a) * w
    return float(fw.mean()), fw

def effective_sample_size(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())
"""
ns3 = run(SOL_3, r"""
est, fw = is_tail_estimate(4.0, 20_000, 4.0, np.random.default_rng(5))
ess_val = effective_sample_size(fw)
est3, fw3 = is_tail_estimate(4.0, 20_000, 3.0, np.random.default_rng(5))
est5, fw5 = is_tail_estimate(4.0, 20_000, 5.0, np.random.default_rng(5))
truth = stats.norm.sf(4.0)
ess_small = effective_sample_size([0.1, 0.1, 0.1, 0.7])
""")
ex.add(
    title="Importance sampling for a Gaussian tail probability", difficulty=2,
    prompt_md=r"""
Estimate $\mu = P(X > a)$ for $X \sim \mathcal{N}(0, 1)$ with importance sampling from the shifted proposal
$q = \mathcal{N}(\text{shift}, 1)$. Implement

* `is_tail_estimate(a, n, shift, rng)` → the tuple `(estimate, fw)`, where the samples are
  `x = rng.normal(shift, 1.0, n)`, the weights are $w(x) = \phi(x)/\phi(x - \text{shift})$ (compute them from
  the formula $\exp\!\big(-x^2/2 + (x - \text{shift})^2/2\big)$ or with `scipy.stats.norm.pdf`), `fw` is the
  array of the $n$ weighted integrands $\mathbf{1}[x_i > a]\,w(x_i)$, and `estimate` is `fw.mean()` as a float;
* `effective_sample_size(w)` → $(\sum_i w_i)^2 / \sum_i w_i^2$ as a float.

Test case: `a = 4`, `n = 20_000`, `shift = 4`, `rng = np.random.default_rng(5)`; the truth is `stats.norm.sf(4)`.
""",
    starter=r"""
def is_tail_estimate(a, n, shift, rng):
    # YOUR CODE HERE  (x = rng.normal(shift, 1.0, n); weights phi(x)/phi(x - shift); fw = indicator * weights)
    return None, None

def effective_sample_size(w):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_3,
    check=rf"""
truth = stats.norm.sf(4.0)
try:
    est, fw = is_tail_estimate(4.0, 20_000, 4.0, np.random.default_rng(5))
    est = float(est); fw = np.asarray(fw, dtype=float)
except Exception:
    est, fw = None, None
check('IS estimate of P(X > 4) within 5% of the truth', est is not None and abs(est - truth) / truth < 0.05)
check('fw has one entry per proposal and is zero where x <= 4 (about half of them)', fw is not None and fw.shape == (20_000,) and 0.4 < np.mean(fw == 0) < 0.6)
check_hash('hidden: the estimate for seed 5 (rounded to 7 decimals)', (round(est, 7) if est is not None else None), '{sha_of(round(ns3["est"], 7), decimals=7)}', decimals=7)
try:
    got = effective_sample_size([0.1, 0.1, 0.1, 0.7])
except Exception:
    got = None
check_close('ESS of weights (0.1, 0.1, 0.1, 0.7) is 1/0.52', got, 1 / 0.52)
try:
    got = effective_sample_size(np.ones(50))
except Exception:
    got = None
check_close('ESS of 50 equal weights is 50', got, 50.0)
try:
    e = effective_sample_size(fw)
except Exception:
    e = None
check('ESS of the weighted integrands is between 3,000 and 4,500 (about 18% of n)', e is not None and 3_000 < e < 4_500)
try:
    e3 = effective_sample_size(is_tail_estimate(4.0, 20_000, 3.0, np.random.default_rng(5))[1])
    e5 = effective_sample_size(is_tail_estimate(4.0, 20_000, 5.0, np.random.default_rng(5))[1])
except Exception:
    e3, e5 = None, None
check('shifting the proposal to 3 or to 5 lowers the ESS compared with 4', e is not None and e3 is not None and e5 is not None and e3 < e and e5 < e)
""",
    explanation_md=r"""
The identity $\mathbb{E}_p[f] = \mathbb{E}_q[f\,p/q]$ makes the weighted average unbiased for any $q$ that
covers the tail. Shifting the proposal to the boundary puts half the samples where the indicator is 1, and
their weights $e^{8 - 4x}$ are at most $e^{-8}$, so the variance per sample drops by a factor of several
thousand relative to plain Monte Carlo (which expects 0.6 hits in 20,000 draws). The ESS of the weighted
integrands is about 18% of $n$ — the theoretical value is $\mu^2/(e^{16}\Phi(-8)) \approx 0.18$ — and it is
*lower* for shifts of 3 (too few samples in the tail) or 5 (weights become more variable), so the ESS is
also a tool for choosing the proposal. The typical error is to compute the ESS of the raw weights $w$ rather
than of $f \cdot w$: for this problem the raw ESS is tiny (about 9) because the weights are huge where
$x < 4$, yet those samples do not matter since the indicator kills them.
""",
)

# ---------------------------------------------------------------- 4. stationary distribution + detailed balance (core)
SOL_4 = r"""
def stationary_distribution(P):
    P = np.asarray(P, dtype=float)
    evals, evecs = np.linalg.eig(P.T)
    k = np.argmin(np.abs(evals - 1.0))
    pi = np.real(evecs[:, k])
    pi = pi / pi.sum()
    return pi

def satisfies_detailed_balance(P, pi, tol=1e-9):
    P = np.asarray(P, dtype=float); pi = np.asarray(pi, dtype=float)
    flows = pi[:, None] * P
    return bool(np.allclose(flows, flows.T, atol=tol))
"""
ns4 = run(SOL_4, r"""
P_w = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
pi_w = stationary_distribution(P_w)
P_q = np.array([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]])      # birth-death chain: reversible
pi_q = stationary_distribution(P_q)
P_c = np.array([[0.1, 0.9, 0.0], [0.0, 0.1, 0.9], [0.9, 0.0, 0.1]])        # cyclic chain: not reversible
pi_c = stationary_distribution(P_c)
db_w, db_q, db_c = satisfies_detailed_balance(P_w, pi_w), satisfies_detailed_balance(P_q, pi_q), satisfies_detailed_balance(P_c, pi_c)
""")
ex.add(
    title="Stationary distributions and detailed balance", difficulty=2,
    prompt_md=r"""
Implement

* `stationary_distribution(P)` → the stationary distribution $\pi$ of the row-stochastic matrix `P`, solving
  $\pi = \pi P$ (hint: $\pi^\top$ is the eigenvector of $P^\top$ with eigenvalue 1; take the real part and
  normalise so the entries sum to 1), returned as a 1-D NumPy array;
* `satisfies_detailed_balance(P, pi, tol=1e-9)` → `True` if $\pi_i P_{ij} = \pi_j P_{ji}$ for all $i, j$
  (to tolerance `tol`), else `False`.

Test chains: the weather chain of the guide
`[[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]]`; the birth–death chain
`[[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]]`; and the cyclic chain
`[[0.1, 0.9, 0], [0, 0.1, 0.9], [0.9, 0, 0.1]]`.
""",
    starter=r"""
def stationary_distribution(P):
    # YOUR CODE HERE  (left eigenvector of P for eigenvalue 1, normalised to sum 1)
    return None

def satisfies_detailed_balance(P, pi, tol=1e-9):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_4,
    check=rf"""
P_w = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
P_q = np.array([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]])
P_c = np.array([[0.1, 0.9, 0.0], [0.0, 0.1, 0.9], [0.9, 0.0, 0.1]])
try:
    pi_w = np.asarray(stationary_distribution(P_w), dtype=float).reshape(3)
except Exception:
    pi_w = None
check_close('weather chain: pi = (21, 13, 12)/46', pi_w, np.array([21, 13, 12]) / 46, tol=1e-6)
check_close('weather chain: pi P = pi', (pi_w @ P_w if pi_w is not None else None), pi_w, tol=1e-9)
try:
    pi_q = np.asarray(stationary_distribution(P_q), dtype=float).reshape(3)
except Exception:
    pi_q = None
check_hash('hidden: stationary distribution of the birth-death chain', (np.round(pi_q, 4) if pi_q is not None else None), '{sha_of(np.round(ns4["pi_q"], 4), decimals=4)}', decimals=4)
try:
    pi_c = np.asarray(stationary_distribution(P_c), dtype=float).reshape(3)
except Exception:
    pi_c = None
check_close('cyclic chain: uniform stationary distribution', pi_c, np.ones(3) / 3, tol=1e-6)
try:
    db = [bool(satisfies_detailed_balance(P, pi)) for P, pi in ((P_w, pi_w), (P_q, pi_q), (P_c, pi_c))]
except Exception:
    db = None
check('detailed balance: weather False, birth-death True, cyclic False', db == [False, True, False])
try:
    got = satisfies_detailed_balance(np.eye(2), np.array([0.3, 0.7]))
except Exception:
    got = None
check('the identity chain satisfies detailed balance for any pi', got is True)
""",
    explanation_md=r"""
$\pi = \pi P$ is $P^\top\pi^\top = \pi^\top$, an eigenvector problem; `np.linalg.eig` may return the vector
with an arbitrary sign and complex dtype, hence the real part and the normalisation. All three chains have
a unique stationary distribution (they are irreducible and aperiodic), but only the birth–death chain
satisfies $\pi_i P_{ij} = \pi_j P_{ji}$: for the cyclic chain probability flows around the cycle
($\pi_1 P_{12} = 0.3$ but $\pi_2 P_{21} = 0$), and for the weather chain the flows are merely unequal
($0.0913$ vs $0.0848$). Detailed balance is sufficient for stationarity, which is why MCMC algorithms impose
it, but it is not necessary — a common confusion.
""",
)

# ---------------------------------------------------------------- 5. MH sampler (core)
SOL_5 = r"""
def log_gamma3(x):
    return 2 * np.log(x) - x if x > 0 else -np.inf

def random_walk_mh(log_target, x0, step, n, rng):
    x = float(x0)
    lp = log_target(x)
    chain = np.empty(n)
    n_acc = 0
    for i in range(n):
        prop = x + step * rng.standard_normal()
        lp_prop = log_target(prop)
        if np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop
            n_acc += 1
        chain[i] = x
    return chain, n_acc / n

def acceptance_probability(log_target, x, prop):
    return float(min(1.0, np.exp(log_target(prop) - log_target(x))))
"""
ns5 = run(SOL_5, r"""
chain, rate = random_walk_mh(log_gamma3, 3.0, 2.5, 30_000, np.random.default_rng(11))
m, v = chain[2000:].mean(), chain[2000:].var()
a1 = acceptance_probability(log_gamma3, 1.0, 2.0)
a2 = acceptance_probability(log_gamma3, 2.0, 3.0)
a3 = acceptance_probability(log_gamma3, 2.0, 6.0)
chain_s, rate_s = random_walk_mh(log_gamma3, 3.0, 0.3, 30_000, np.random.default_rng(11))
chain_l, rate_l = random_walk_mh(log_gamma3, 3.0, 12.0, 30_000, np.random.default_rng(11))
""")
ex.add(
    title="Random-walk Metropolis–Hastings for a Gamma target", difficulty=2,
    prompt_md=r"""
The target is Gamma(shape 3, rate 1), known only up to a constant: $\log\tilde p(x) = 2\ln x - x$ for $x > 0$
and $-\infty$ otherwise. Implement

* `log_gamma3(x)` — the unnormalised log density for a scalar `x` (return `-np.inf` for `x <= 0`);
* `acceptance_probability(log_target, x, prop)` → $\min\{1, \tilde p(\text{prop})/\tilde p(x)\}$ for a symmetric
  proposal, computed from the log target;
* `random_walk_mh(log_target, x0, step, n, rng)` → `(chain, acceptance_rate)`: `n` iterations starting from
  `x0`; at each iteration draw `prop = x + step * rng.standard_normal()`, then `u = rng.random()`, accept
  if `np.log(u) < log_target(prop) - log_target(x)`, and record the current state (whether or not the move was
  accepted) in a NumPy array of length `n`. Exactly these two `rng` calls per iteration, in this order.

Test: `random_walk_mh(log_gamma3, 3.0, 2.5, 30_000, np.random.default_rng(11))`, discarding the first 2,000
samples as burn-in. Gamma(3, 1) has mean 3 and variance 3.
""",
    starter=r"""
def log_gamma3(x):
    # YOUR CODE HERE
    return None

def acceptance_probability(log_target, x, prop):
    # YOUR CODE HERE
    return None

def random_walk_mh(log_target, x0, step, n, rng):
    # YOUR CODE HERE  (per iteration: prop = x + step*rng.standard_normal(); u = rng.random())
    return None, None
""",
    solution=SOL_5,
    check=rf"""
try:
    got = [float(log_gamma3(1.0)), float(log_gamma3(2.0)), float(log_gamma3(-1.0))]
    ok_log = abs(got[0] + 1) < 1e-12 and abs(got[1] - (2 * np.log(2) - 2)) < 1e-12 and got[2] == -np.inf
except Exception:
    ok_log = False
check('log target: log p(1) = -1, log p(2) = 2 ln 2 - 2, log p(-1) = -inf', ok_log)
try:
    a1 = acceptance_probability(log_gamma3, 1.0, 2.0)
    a2 = acceptance_probability(log_gamma3, 2.0, 3.0)
    a3 = acceptance_probability(log_gamma3, 2.0, 6.0)
except Exception:
    a1 = a2 = a3 = None
check_close('uphill move 1 -> 2 (towards the mode at 2) is accepted with probability 1', a1, 1.0)
check_hash('hidden: acceptance probability of the downhill move 2 -> 3', (round(a2, 6) if a2 is not None else None), '{sha_of(round(ns5["a2"], 6), decimals=6)}', decimals=6)
check_hash('hidden: acceptance probability of the downhill move 2 -> 6', (round(a3, 6) if a3 is not None else None), '{sha_of(round(ns5["a3"], 6), decimals=6)}', decimals=6)
try:
    chain, rate = random_walk_mh(log_gamma3, 3.0, 2.5, 30_000, np.random.default_rng(11))
    chain = np.asarray(chain, dtype=float); rate = float(rate); post = chain[2000:]
except Exception:
    chain, rate, post = None, None, None
check('chain has length 30,000 and stays positive', chain is not None and chain.shape == (30_000,) and bool(np.all(chain > 0)))
check('acceptance rate between 0.45 and 0.65 for step 2.5', rate is not None and 0.45 < rate < 0.65)
check('posterior mean within 0.1 of 3', post is not None and abs(post.mean() - 3) < 0.1)
check('posterior variance within 0.3 of 3', post is not None and abs(post.var() - 3) < 0.3)
check_hash('hidden: exact acceptance rate for seed 11, step 2.5', (round(rate, 4) if rate is not None else None), '{sha_of(round(ns5["rate"], 4), decimals=4)}', decimals=4)
try:
    rate_small = random_walk_mh(log_gamma3, 3.0, 0.3, 30_000, np.random.default_rng(11))[1]
    rate_large = random_walk_mh(log_gamma3, 3.0, 12.0, 30_000, np.random.default_rng(11))[1]
except Exception:
    rate_small = rate_large = None
check('tiny step accepts > 85%, huge step accepts < 25%', rate_small is not None and rate_large is not None and rate_small > 0.85 and rate_large < 0.25)
""",
    explanation_md=r"""
Only *ratios* of the target enter, so the Gamma normalising constant $1/\Gamma(3) = 1/2$ never has to be known,
and working with $\log\tilde p$ keeps the comparison `log(u) < log p(prop) - log p(x)` safe from under- and
overflow. A proposal below zero has $\log\tilde p = -\infty$ and is rejected automatically. The step $2.5$
(about 1.4 target standard deviations) gives an acceptance rate near 0.55 and a chain whose mean and variance
match Gamma(3, 1) after burn-in; the step $0.3$ accepts almost everything but moves slowly, and the step $12$
is rejected most of the time. Recording the *current* state after a rejection — not skipping it — is
essential: the repeated values are what give low-density regions their correct (small) weight.
""",
)

# ---------------------------------------------------------------- 6. conceptual
ex.add_conceptual(
    title="Dimension kills rejection sampling; the direction of the KL matters", difficulty=2,
    prompt_md=r"""
1. A colleague proposes to sample a 200-dimensional posterior by rejection sampling with a Gaussian proposal
   whose covariance is "about right, within 10% in each direction". Estimate the acceptance rate and explain
   why no amount of tuning will save the plan.
2. Variational inference minimises $\text{KL}(q \,\|\, p)$ (reverse KL). Suppose the true posterior $p$ is a
   50/50 mixture of two well-separated Gaussians and $q$ is a single Gaussian. Describe the $q$ that minimises
   the reverse KL and the $q$ that would minimise the forward KL $\text{KL}(p \,\|\, q)$, and explain why VI
   uses the reverse direction anyway.
""",
    answer_md=r"""
1. For target $\mathcal{N}(0, I_d)$ and proposal $\mathcal{N}(0, \sigma^2 I_d)$ with $\sigma = 1.1$ (the proposal
   must be *wider* than the target or no finite envelope exists), the tightest constant is
   $M = \sup_x p(x)/q(x) = \sigma^d$, so the acceptance rate is $\sigma^{-d} = 1.1^{-200} \approx 5 \times 10^{-9}$
   — one accepted sample per 200 million proposals. The failure is structural: a per-coordinate mismatch of
   $\sigma$ becomes $\sigma^d$ in the density ratio, and even $\sigma = 1.01$ gives $1.01^{-200} \approx 0.14$ only
   because the proposal is then essentially the target, which we cannot construct without already knowing the
   posterior. Any error in the proposal is amplified exponentially in $d$, which is why high-dimensional
   sampling uses Markov chains (local moves, no global envelope) or variational methods.
2. The reverse KL $\text{KL}(q \,\|\, p) = \mathbb{E}_q[\log q - \log p]$ is huge wherever $q$ puts mass but $p$ is
   tiny, so the optimal single Gaussian sits on *one* of the two modes, matching its width (**mode-seeking**,
   variance underestimated: it ignores half the posterior). The forward KL $\text{KL}(p \,\|\, q) = \mathbb{E}_p[\log p - \log q]$
   penalises $q$ wherever $p$ has mass and $q$ does not, so its minimiser is a broad Gaussian covering both modes
   with the correct overall mean and inflated variance (**mass-covering**). VI uses the reverse direction because
   its expectation is under $q$, which we can sample and differentiate through, and because it leads to the
   ELBO, which needs only the unnormalised $p(x, z)$; the forward KL requires expectations under the unknown
   posterior — the very thing we are trying to compute.
""",
)

# ---------------------------------------------------------------- 7. Gaussian KL (core)
SOL_7 = r"""
def kl_gaussian_to_standard(mu, log_sigma):
    mu = np.asarray(mu, dtype=float); log_sigma = np.asarray(log_sigma, dtype=float)
    sigma2 = np.exp(2 * log_sigma)
    return float(np.sum(0.5 * (mu ** 2 + sigma2 - 1 - 2 * log_sigma)))

def kl_monte_carlo(mu, log_sigma, n, rng):
    mu = np.asarray(mu, dtype=float); sigma = np.exp(np.asarray(log_sigma, dtype=float))
    eps = rng.standard_normal((n, len(mu)))
    z = mu + sigma * eps
    log_q = np.sum(stats.norm.logpdf(z, mu, sigma), axis=1)
    log_p = np.sum(stats.norm.logpdf(z, 0.0, 1.0), axis=1)
    return float(np.mean(log_q - log_p))
"""
ns7 = run(SOL_7, r"""
kl_a = kl_gaussian_to_standard([0.5, -1.0], np.log([0.8, 0.5]))
kl_b = kl_gaussian_to_standard([1.0, 2.0, -0.5], [0.0, -0.5, 0.3])
mc = kl_monte_carlo([0.5, -1.0], np.log([0.8, 0.5]), 200_000, np.random.default_rng(9))
""")
ex.add(
    title="The VAE's closed-form KL, verified by Monte Carlo", difficulty=2,
    prompt_md=r"""
For a diagonal Gaussian encoder output $q = \mathcal{N}(\mu, \operatorname{diag}\sigma^2)$ and prior $p = \mathcal{N}(0, I)$,
implement

* `kl_gaussian_to_standard(mu, log_sigma)` → $\text{KL}(q \,\|\, p) = \sum_j \frac12\big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big)$
  as a float, where `log_sigma` holds $\log\sigma_j$ (the quantity a VAE encoder outputs);
* `kl_monte_carlo(mu, log_sigma, n, rng)` → the Monte Carlo estimate $\frac1n\sum_i[\log q(z_i) - \log p(z_i)]$
  from `n` reparameterised samples `z = mu + sigma * rng.standard_normal((n, k))`, using `scipy.stats.norm.logpdf`
  for the two log densities (sum over the `k` coordinates), as a float.

Test: $\mu = (0.5, -1.0)$, $\sigma = (0.8, 0.5)$ — the worked example of the guide, where the KL is $0.9863$.
""",
    starter=r"""
def kl_gaussian_to_standard(mu, log_sigma):
    # YOUR CODE HERE
    return None

def kl_monte_carlo(mu, log_sigma, n, rng):
    # YOUR CODE HERE  (z = mu + sigma * rng.standard_normal((n, k)); average of log q(z) - log p(z))
    return None
""",
    solution=SOL_7,
    check=rf"""
try:
    got = kl_gaussian_to_standard([0.5, -1.0], np.log([0.8, 0.5]))
except Exception:
    got = None
check_close('KL for mu = (0.5, -1), sigma = (0.8, 0.5) is 0.98629', got, 0.986290731874155, tol=1e-6)
try:
    got = kl_gaussian_to_standard([0.0, 0.0], [0.0, 0.0])
except Exception:
    got = None
check_close('KL is zero when q equals the prior', got, 0.0, tol=1e-12)
try:
    got = kl_gaussian_to_standard([1.0, 2.0, -0.5], [0.0, -0.5, 0.3])
except Exception:
    got = None
check_hash('hidden: KL for a 3-D case', (round(got, 5) if got is not None else None), '{sha_of(round(ns7["kl_b"], 5), decimals=5)}', decimals=5)
try:
    got = kl_gaussian_to_standard([0.0], [np.log(0.05)])
except Exception:
    got = None
check('a near-deterministic code (sigma = 0.05) costs more than 2 nats', got is not None and got > 2.0)
try:
    mc = kl_monte_carlo([0.5, -1.0], np.log([0.8, 0.5]), 200_000, np.random.default_rng(9))
except Exception:
    mc = None
check('Monte Carlo KL within 0.01 of the closed form', mc is not None and abs(mc - 0.986290731874155) < 0.01)
check_hash('hidden: the Monte Carlo estimate for seed 9 (3 decimals)', (round(mc, 3) if mc is not None else None), '{sha_of(round(ns7["mc"], 3), decimals=3)}', decimals=3)
""",
    explanation_md=r"""
$\text{KL}(q\,\|\,p) = \mathbb{E}_q[\log q - \log p]$ with both densities Gaussian: $\mathbb{E}_q[-\log\sigma - (z-\mu)^2/(2\sigma^2)] = -\log\sigma - \tfrac12$
and $\mathbb{E}_q[z^2/2] = (\mu^2 + \sigma^2)/2$, giving $\frac12(\mu^2 + \sigma^2 - 1 - \log\sigma^2)$ per dimension;
independent dimensions add. The Monte Carlo estimate is what a VAE *would* compute if the KL had no closed
form, and it agrees to two decimals with 200,000 samples — in practice the closed form is used because it
is exact and has zero variance. The penalty $-\log\sigma^2$ explodes as $\sigma \to 0$: a deterministic code is
infinitely expensive, which is how the KL keeps the latent space smooth. A frequent bug is to pass
$\log\sigma^2$ where $\log\sigma$ is expected (PyTorch tutorials use `logvar`), which halves or doubles the
log term.
""",
)

# ---------------------------------------------------------------- 8. diffusion schedule (core)
SOL_8 = r"""
def alpha_bar(betas):
    return np.cumprod(1 - np.asarray(betas, dtype=float))

def forward_marginal(x0, t, betas):
    ab = alpha_bar(betas)[t - 1]
    return float(np.sqrt(ab) * x0), float(1 - ab)

def noisy_sample(x0, t, betas, eps):
    mean, var = forward_marginal(x0, t, betas)
    return mean + np.sqrt(var) * eps

def implied_noise(x0, t, betas, x_t):
    mean, var = forward_marginal(x0, t, betas)
    return float((x_t - mean) / np.sqrt(var))
"""
ns8 = run(SOL_8, r"""
b3 = [0.1, 0.2, 0.3]
ab3 = alpha_bar(b3)
m3, v3 = forward_marginal(2.0, 3, b3)
e3 = implied_noise(2.0, 3, b3, 1.0)
lin = np.linspace(1e-4, 0.02, 1000)
ab_lin = alpha_bar(lin)
m500, v500 = forward_marginal(1.0, 500, lin)
snr_500 = ab_lin[499] / (1 - ab_lin[499])
ab_T = ab_lin[-1]
""")
ex.add(
    title="The diffusion forward process in closed form", difficulty=2,
    prompt_md=r"""
With $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s \le t}\alpha_s$, the forward marginal is
$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\ 1 - \bar\alpha_t)$ (one dimension). Implement, with `t`
**1-based** (so `t = 1` uses $\beta_1$ only):

* `alpha_bar(betas)` → the array $(\bar\alpha_1, \dots, \bar\alpha_T)$;
* `forward_marginal(x0, t, betas)` → the tuple `(mean, variance)` of $q(x_t \mid x_0)$ as floats;
* `noisy_sample(x0, t, betas, eps)` → $\sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$ for a given noise value `eps`;
* `implied_noise(x0, t, betas, x_t)` → the $\varepsilon$ that produced `x_t` from `x0`, i.e. $(x_t - \sqrt{\bar\alpha_t}x_0)/\sqrt{1 - \bar\alpha_t}$ — the regression target of the diffusion loss.

Tests: the guide's schedule $\beta = (0.1, 0.2, 0.3)$ with $x_0 = 2$, and the DDPM schedule
`np.linspace(1e-4, 0.02, 1000)`.
""",
    starter=r"""
def alpha_bar(betas):
    # YOUR CODE HERE
    return None

def forward_marginal(x0, t, betas):
    # YOUR CODE HERE  (t is 1-based)
    return None, None

def noisy_sample(x0, t, betas, eps):
    # YOUR CODE HERE
    return None

def implied_noise(x0, t, betas, x_t):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_8,
    check=rf"""
b3 = [0.1, 0.2, 0.3]
try:
    got = np.asarray(alpha_bar(b3), dtype=float)
except Exception:
    got = None
check_close('alpha_bar for beta = (0.1, 0.2, 0.3) is (0.9, 0.72, 0.504)', got, [0.9, 0.72, 0.504], tol=1e-12)
try:
    m, v = forward_marginal(2.0, 3, b3)
except Exception:
    m, v = None, None
check_close('marginal mean at t = 3 from x0 = 2 is 2 sqrt(0.504)', m, 2 * np.sqrt(0.504), tol=1e-9)
check_close('marginal variance at t = 3 is 0.496', v, 0.496, tol=1e-9)
try:
    got = noisy_sample(2.0, 3, b3, -0.5)
except Exception:
    got = None
check_close('noisy sample with eps = -0.5', got, 2 * np.sqrt(0.504) - 0.5 * np.sqrt(0.496), tol=1e-9)
try:
    got = implied_noise(2.0, 3, b3, 1.0)
except Exception:
    got = None
check_hash('hidden: implied noise for x_3 = 1.0 (the guide example)', (round(got, 4) if got is not None else None), '{sha_of(round(ns8["e3"], 4), decimals=4)}', decimals=4)
try:
    got = implied_noise(2.0, 3, b3, noisy_sample(2.0, 3, b3, 0.7))
except Exception:
    got = None
check_close('implied_noise inverts noisy_sample', got, 0.7, tol=1e-9)
lin = np.linspace(1e-4, 0.02, 1000)
try:
    ab = np.asarray(alpha_bar(lin), dtype=float)
    abT = float(ab[-1]); snr500 = float(ab[499] / (1 - ab[499]))
except Exception:
    ab, abT, snr500 = None, None, None
check('DDPM linear schedule: alpha_bar_T below 1e-4 (pure noise at the end)', abT is not None and abT < 1e-4)
check_hash('hidden: signal-to-noise ratio at t = 500 (3 decimals)', (round(snr500, 3) if snr500 is not None else None), '{sha_of(round(ns8["snr_500"], 3), decimals=3)}', decimals=3)
""",
    explanation_md=r"""
Each forward step scales the signal by $\sqrt{\alpha_t}$ and adds variance $\beta_t$; because a Gaussian plus
an independent Gaussian is Gaussian, the composition of $t$ steps is a single Gaussian whose mean coefficient
is $\prod\sqrt{\alpha_s} = \sqrt{\bar\alpha_t}$ and whose variance is $1 - \bar\alpha_t$ (check: variance
$\alpha_2(1 - \alpha_1) + (1 - \alpha_2) = 1 - \alpha_1\alpha_2$ after two steps). That closed form is what makes
training cheap: any $x_t$ is one draw away from $x_0$, and the noise used in that draw is the regression target
$\varepsilon$. For the DDPM schedule, $\bar\alpha_{1000} = 4 \times 10^{-5}$, so $x_T$ is indistinguishable from
$\mathcal{N}(0, 1)$, while at $t = 500$ the SNR is already below 0.09 — half the steps operate in near-noise.
The usual off-by-one error is indexing $\bar\alpha$ with a 0-based `t`, which silently shifts the schedule.
""",
)

# ---------------------------------------------------------------- 9. score of a Gaussian mixture (stretch)
SOL_9 = r"""
def mixture_logpdf(x, w, m, s):
    x = np.asarray(x, dtype=float)[..., None]
    w, m, s = (np.asarray(a, dtype=float) for a in (w, m, s))
    comp = np.log(w) - 0.5 * np.log(2 * np.pi * s ** 2) - (x - m) ** 2 / (2 * s ** 2)
    mx = comp.max(axis=-1, keepdims=True)
    return (mx + np.log(np.sum(np.exp(comp - mx), axis=-1, keepdims=True)))[..., 0]

def mixture_score(x, w, m, s):
    x = np.asarray(x, dtype=float)[..., None]
    w, m, s = (np.asarray(a, dtype=float) for a in (w, m, s))
    comp = np.log(w) - 0.5 * np.log(2 * np.pi * s ** 2) - (x - m) ** 2 / (2 * s ** 2)
    mx = comp.max(axis=-1, keepdims=True)
    resp = np.exp(comp - mx)
    resp = resp / resp.sum(axis=-1, keepdims=True)
    return np.sum(resp * (-(x - m) / s ** 2), axis=-1)

def langevin(score_fn, x0, eps, n_steps, rng):
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(n_steps):
        x = x + 0.5 * eps * score_fn(x) + np.sqrt(eps) * rng.standard_normal(x.shape)
    return x
"""
ns9 = run(SOL_9, r"""
w, m, s = [0.3, 0.7], [-1.0, 2.0], [0.5, 1.0]
pts = np.array([-1.5, 0.0, 0.8, 3.0])
sc = mixture_score(pts, w, m, s)
sc_single = mixture_score(np.array([0.5, 2.0]), [1.0], [0.0], [1.0])
x_final = langevin(lambda x: mixture_score(x, w, m, s), 4 * np.random.default_rng(1).standard_normal(5000), 0.05, 600, np.random.default_rng(1))
frac = float(np.mean(x_final > 0.5))
""")
ex.add(
    title="The score of a Gaussian mixture and Langevin sampling", difficulty=3,
    prompt_md=r"""
For the mixture $p(x) = \sum_k w_k\,\mathcal{N}(x; m_k, s_k^2)$ implement

* `mixture_logpdf(x, w, m, s)` → $\log p(x)$, vectorised over an array `x` (use the log-sum-exp trick);
* `mixture_score(x, w, m, s)` → $\nabla_x\log p(x) = \sum_k r_k(x)\,\frac{-(x - m_k)}{s_k^2}$ with responsibilities
  $r_k(x) = w_k\mathcal{N}(x; m_k, s_k^2)/p(x)$, vectorised over `x`;
* `langevin(score_fn, x0, eps, n_steps, rng)` → the particles after `n_steps` updates
  $x \leftarrow x + \frac{\epsilon}{2}\,\text{score}(x) + \sqrt{\epsilon}\,\xi$ with `xi = rng.standard_normal(x.shape)`
  (one such call per step), starting from the array `x0`.

Test mixture: `w = [0.3, 0.7]`, `m = [-1.0, 2.0]`, `s = [0.5, 1.0]`. The score is checked against central
finite differences of your `mixture_logpdf` **and** against a hidden reference.
""",
    starter=r"""
def mixture_logpdf(x, w, m, s):
    # YOUR CODE HERE
    return None

def mixture_score(x, w, m, s):
    # YOUR CODE HERE  (responsibilities times the per-component scores)
    return None

def langevin(score_fn, x0, eps, n_steps, rng):
    # YOUR CODE HERE  (x <- x + eps/2 * score(x) + sqrt(eps) * rng.standard_normal(x.shape))
    return None
""",
    solution=SOL_9,
    check=rf"""
w, m, s = [0.3, 0.7], [-1.0, 2.0], [0.5, 1.0]
pts = np.array([-1.5, 0.0, 0.8, 3.0])
try:
    lp = np.asarray(mixture_logpdf(pts, w, m, s), dtype=float)
    ref = np.log(0.3 * stats.norm.pdf(pts, -1, 0.5) + 0.7 * stats.norm.pdf(pts, 2, 1))
except Exception:
    lp, ref = None, None
check_close('log density matches scipy at four points', lp, ref, tol=1e-9)
try:
    got = np.asarray(mixture_score(np.array([0.5, 2.0]), [1.0], [0.0], [1.0]), dtype=float)
except Exception:
    got = None
check_close('single standard normal: score is -x', got, [-0.5, -2.0], tol=1e-9)
try:
    sc = np.asarray(mixture_score(pts, w, m, s), dtype=float)
    h = 1e-5
    fd = (np.asarray(mixture_logpdf(pts + h, w, m, s)) - np.asarray(mixture_logpdf(pts - h, w, m, s))) / (2 * h)
except Exception:
    sc, fd = None, None
check_close('score matches finite differences of the log density', sc, fd, tol=1e-5)
check_hash('hidden: score at the four test points (4 decimals)', (np.round(sc, 4) if sc is not None else None), '{sha_of(np.round(ns9["sc"], 4), decimals=4)}', decimals=4)
try:
    x_final = np.asarray(langevin(lambda x: mixture_score(x, w, m, s),
                                  4 * np.random.default_rng(1).standard_normal(5000), 0.05, 600, np.random.default_rng(1)), dtype=float)
    if x_final.shape != (5000,):
        raise ValueError("wrong shape")
    frac = float(np.mean(x_final > 0.5)); mean_right = float(x_final[x_final > 0.5].mean())
except Exception:
    x_final, frac, mean_right = None, None, None
check('Langevin: about 70% of particles end in the right-hand mode', frac is not None and 0.64 < frac < 0.76)
check('Langevin: right-hand particles average near 2', mean_right is not None and abs(mean_right - 2.0) < 0.15)
check_hash('hidden: exact fraction in the right mode for seed 1 (3 decimals)', (round(frac, 3) if frac is not None else None), '{sha_of(round(ns9["frac"], 3), decimals=3)}', decimals=3)
""",
    explanation_md=r"""
Differentiating $\log\sum_k w_k\mathcal{N}_k(x)$ gives $\frac{\sum_k w_k\mathcal{N}_k(x)\,\nabla\log\mathcal{N}_k(x)}{\sum_k w_k\mathcal{N}_k(x)}$:
a responsibility-weighted average of the per-component scores $-(x - m_k)/s_k^2$. Computing the responsibilities
in log space (subtract the maximum before exponentiating) avoids underflow far from the modes. Langevin
dynamics with this score needs no normalising constant and no accept/reject step; after 600 steps of size
$\epsilon = 0.05$ the particle cloud carries about 70% of its mass in the right-hand mode, as the mixture weights
say, with a small bias from the finite step size. The step `0.5 * eps * score` — not `eps * score` — is the
one that makes $\sqrt\epsilon$ the right noise scale; the diffusion sampler of the lesson is this update with
a learned score and a slowly changing target.
""",
)

# ---------------------------------------------------------------- 10. mini-project (★★★)
PROJECT_CODE = r"""
rng = np.random.default_rng(2024)
y = np.array([3, 5, 2, 4, 6, 3, 4, 5, 2, 3])
n, S = len(y), y.sum()
a0, b0 = 2.0, 1.0                                   # Gamma(shape 2, rate 1) prior on the rate lambda
a_post, b_post = a0 + S, b0 + n                     # exact conjugate posterior Gamma(39, 11)
print(f"data: n = {n}, sum = {S};  exact posterior Gamma(shape {a_post:.0f}, rate {b_post:.0f}): mean {a_post / b_post:.4f}, sd {np.sqrt(a_post) / b_post:.4f}")

def log_post(lam):
    # log prior + log likelihood, up to a constant; Poisson likelihood sum(y) log lam - n lam
    if lam <= 0:
        return -np.inf
    return (a0 - 1) * np.log(lam) - b0 * lam + S * np.log(lam) - n * lam

def random_walk_mh(log_target, x0, step, n_iter, rng):
    x, lp = float(x0), log_target(x0)
    chain, n_acc = np.empty(n_iter), 0
    for i in range(n_iter):
        prop = x + step * rng.standard_normal()
        lp_prop = log_target(prop)
        if np.log(rng.random()) < lp_prop - lp:
            x, lp, n_acc = prop, lp_prop, n_acc + 1
        chain[i] = x
    return chain, n_acc / n_iter

def autocorr(x, max_lag):
    x = x - x.mean(); d = x @ x
    return np.array([x[:len(x) - k] @ x[k:] / d for k in range(max_lag + 1)])

def ess(x, max_lag=1000):
    rho = autocorr(x, min(max_lag, len(x) - 1)); tau = 1.0
    for k in range(1, len(rho)):
        if rho[k] < 0:
            break
        tau += 2 * rho[k]
    return len(x) / tau, tau

n_iter, burn = 20_000, 2_000
steps = [0.05, 0.3, 1.0, 3.0, 10.0]
results = {}
fig, axes = plt.subplots(len(steps), 2, figsize=(11, 2.2 * len(steps)))
grid = np.linspace(1.5, 6.0, 300)
for row, s in enumerate(steps):
    chain, acc = random_walk_mh(log_post, 1.0, s, n_iter, rng)       # deliberately poor start at lambda = 1
    post = chain[burn:]
    n_eff, tau = ess(post)
    results[s] = dict(acc=acc, mean=post.mean(), sd=post.std(), ess=n_eff, tau=tau, chain=chain)
    axes[row, 0].plot(chain[:3000], lw=0.6); axes[row, 0].axhline(a_post / b_post, color="r", ls="--")
    axes[row, 0].set_title(f"step {s}: acceptance {acc:.2f}, ESS {n_eff:.0f}"); axes[row, 0].set_xlabel("iteration"); axes[row, 0].set_ylabel("lambda")
    axes[row, 1].plot(autocorr(post, 150)); axes[row, 1].axhline(0, color="k", lw=0.5)
    axes[row, 1].set_title(f"autocorrelation, tau = {tau:.1f}"); axes[row, 1].set_xlabel("lag"); axes[row, 1].set_ylabel("rho_k")
plt.tight_layout(); plt.show()

print(f"\n{'step':>6s} {'accept':>7s} {'mean':>7s} {'sd':>7s} {'tau':>7s} {'ESS':>7s}   (exact mean {a_post / b_post:.4f}, sd {np.sqrt(a_post) / b_post:.4f})")
for s in steps:
    r = results[s]
    print(f"{s:6.2f} {r['acc']:7.3f} {r['mean']:7.4f} {r['sd']:7.4f} {r['tau']:7.1f} {r['ess']:7.0f}")
best = max(steps, key=lambda s: results[s]["ess"])
print(f"\nbest step by ESS: {best}  (acceptance {results[best]['acc']:.2f}; optimal step ~ 2.4 * posterior sd = {2.4 * np.sqrt(a_post) / b_post:.2f})")

# compare the best chain with the exact posterior
post = results[best]["chain"][burn:]
plt.hist(post, bins=60, density=True, alpha=0.6, label=f"MH samples (step {best})")
plt.plot(grid, stats.gamma.pdf(grid, a_post, scale=1 / b_post), "r", label="exact Gamma(39, 11) posterior")
plt.xlabel("lambda (events per interval)"); plt.ylabel("posterior density"); plt.title("MH posterior vs the exact conjugate posterior"); plt.legend(); plt.show()
q_mc = np.quantile(post, [0.025, 0.5, 0.975]); q_ex = stats.gamma.ppf([0.025, 0.5, 0.975], a_post, scale=1 / b_post)
print("2.5%, 50%, 97.5% quantiles:  MH", q_mc.round(3), "  exact", q_ex.round(3))
print(f"P(lambda > 4 | data):  MH {np.mean(post > 4):.4f} +/- {np.sqrt(np.mean(post > 4) * (1 - np.mean(post > 4)) / results[best]['ess']):.4f} (MC error from the ESS)   exact {stats.gamma.sf(4, a_post, scale=1 / b_post):.4f}")

# burn-in matters when the start is bad: compare the first 200 iterations of the small-step chain with the posterior mean
bad = results[0.05]["chain"]
print(f"\nsmall step 0.05 from lambda = 1: mean of iterations 1-200 = {bad[:200].mean():.3f}, of 2000-20000 = {bad[2000:].mean():.3f}  (exact {a_post / b_post:.3f}) -> burn-in is essential")

# R-hat from 4 chains with dispersed starts, for the best step
chains = np.array([random_walk_mh(log_post, x0, best, 5_000, rng)[0][500:] for x0 in (0.5, 2.0, 5.0, 9.0)])
m_c, n_c = chains.shape
W = chains.var(axis=1, ddof=1).mean(); B = n_c * chains.mean(axis=1).var(ddof=1)
var_hat = (n_c - 1) / n_c * W + B / n_c
print(f"R-hat over 4 chains (step {best}): {np.sqrt(var_hat / W):.4f}  (values near 1.00 indicate the chains agree)")
"""
ex.add_project(
    title="Mini-project: Bayesian inference for a Poisson rate with Metropolis–Hastings",
    prompt_md=r"""
A hospital ward records $y = (3, 5, 2, 4, 6, 3, 4, 5, 2, 3)$ emergency admissions on ten consecutive nights,
modelled as $y_i \sim \text{Poisson}(\lambda)$ with prior $\lambda \sim \text{Gamma}(\text{shape } 2, \text{rate } 1)$.
Because the model is conjugate (Module 11), the exact posterior is $\text{Gamma}(2 + \sum y_i,\ 1 + n) = \text{Gamma}(39, 11)$
— which lets you *grade your own sampler*.

**Tasks** (use `rng = np.random.default_rng(2024)`):

1. Write the unnormalised log posterior $\log\tilde p(\lambda) = (a_0 - 1)\ln\lambda - b_0\lambda + \sum_i y_i\ln\lambda - n\lambda$
   for $\lambda > 0$ (and $-\infty$ otherwise) and a random-walk MH sampler.
2. Run 20,000 iterations from the deliberately poor start $\lambda_0 = 1$ for step sizes
   $s \in \{0.05, 0.3, 1, 3, 10\}$. For each: acceptance rate, trace plot, autocorrelation function, integrated
   autocorrelation time $\tau$ and $\text{ESS} = n/\tau$ after discarding 2,000 burn-in iterations.
3. Pick the step with the largest ESS and compare its samples with the exact Gamma(39, 11) posterior:
   histogram against the density, posterior mean and sd, the 2.5%/50%/97.5% quantiles, and
   $P(\lambda > 4 \mid y)$ with a Monte Carlo standard error computed from the ESS (not from $n$).
4. Show why burn-in matters: compare the mean of the first 200 iterations of the smallest-step chain with the
   exact posterior mean.
5. Run four shorter chains (5,000 iterations) from the starts $\{0.5, 2, 5, 9\}$ with your best step and compute
   $\hat R = \sqrt{\hat V / W}$, where $W$ is the mean within-chain variance, $B = n\cdot\operatorname{Var}(\text{chain means})$
   and $\hat V = \frac{n-1}{n}W + \frac{B}{n}$.
""",
    rubric_md=r"""
* The log posterior is correct (prior **and** likelihood, $-\infty$ for $\lambda \le 0$) and the sampler accepts with
  probability $\min\{1, \tilde p(\lambda')/\tilde p(\lambda)\}$ computed in log space, recording the current state after
  a rejection.
* Acceptance rate, $\tau$ and ESS are reported for every step size, and the best step (largest ESS, acceptance
  roughly 0.3–0.6, close to $2.4\times$ the posterior sd) is chosen by ESS rather than by acceptance rate.
* The MH posterior mean and sd agree with the exact $39/11 = 3.545$ and $\sqrt{39}/11 = 0.568$ to within Monte
  Carlo error, and the histogram lies on the Gamma(39, 11) density.
* The standard error of $P(\lambda > 4 \mid y)$ uses the ESS, and the exact value from `scipy.stats.gamma.sf` is
  within about two standard errors.
* The burn-in comparison shows the early iterations of the slow chain are biased toward the start, and
  $\hat R$ from four dispersed chains is close to 1.
* All figures have titles and axis labels.
""",
    solution_sketch_md=r"""
The posterior is proportional to $\lambda^{a_0 - 1 + \sum y}e^{-(b_0 + n)\lambda}$, so the sampler only needs
$\log\tilde p(\lambda) = 38\ln\lambda - 11\lambda$. With a posterior sd of $0.57$, a step of $0.05$ accepts
about 97% of proposals but has $\tau$ in the hundreds (ESS a few hundred, and its first 200 iterations average
well below the posterior mean because the chain is still climbing from $\lambda = 1$); a step of $1$ or $3$
accepts 30–60% and gives $\tau \approx 5$–$8$, ESS around 3,000; a step of $10$ is rejected about 90% of the
time and the ESS falls again. The optimal step is near $2.4\sigma \approx 1.4$. The best chain reproduces the
exact mean $3.545$ and sd $0.568$ to two decimals, its quantiles match `stats.gamma.ppf`, and
$P(\lambda > 4 \mid y) \approx 0.21$ agrees with `stats.gamma.sf(4, 39, scale=1/11)` within the ESS-based
standard error. Four chains from dispersed starts give $\hat R \approx 1.00$. The lesson of the project: a
conjugate problem lets you *see* every MCMC pathology — slow mixing, start-up bias, wasted rejections — against
the truth, so that you recognise them when there is no truth to compare with.
""",
    solution_code=PROJECT_CODE,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print(f"exercises: {len(ex.items)} items — {kinds.count('code')} coding, {kinds.count('concept')} conceptual, {kinds.count('project')} project")
print("difficulties:", [it["difficulty"] for it in ex.items])
