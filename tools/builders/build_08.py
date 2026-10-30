"""
build_08.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for

    part-3-probability/08-distributions-and-limit-theorems

Run from the repository root:

    python3 tools/builders/build_08.py
    python3 tools/run_notebooks.py part-3-probability/08-distributions-and-limit-theorems

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

MOD = ROOT / "part-3-probability/08-distributions-and-limit-theorems"

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 08 — Distributions and Limit Theorems")

nb.md(r"""
This notebook is the runnable companion to the concept guide: read [README.md](README.md) first, or keep
it open alongside. Each section mirrors a numbered section of the guide. We draw a gallery of the named
distributions, check every mean and variance formula against `scipy.stats`, simulate the main families, watch the Binomial turn into the
Poisson, check memorylessness numerically, test the 68–95–99.7 rule with a Q–Q plot, grow a Beta posterior,
build the Gamma, Chi-square and Student-t from their ingredients, confirm an MGF identity, watch the law of
large numbers work (and fail for the Cauchy), see the central limit theorem emerge at different speeds, and
finish with a Monte Carlo integral whose error bars shrink like $1/\sqrt{n}$.

Everything is computed in plain NumPy first; `scipy.stats` appears afterwards to confirm the hand computation.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
from math import comb, exp, factorial, log, sqrt, pi
from scipy import stats
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
""")

# ---------------------------------------------------------------- 1. gallery
nb.md(r"""
## 1. A distribution is a modelling assumption — the gallery

Each named distribution is a *story* about how one data point is generated, plus a few parameters. The
first gallery shows discrete PMFs, the second continuous PDFs, each with its parameters varied. Watch how a
single parameter moves the mass around, and keep the stories in mind: Binomial = successes in $n$ trials,
Geometric = trials until the first success, Poisson = rare events in a window, Exponential = waiting time at
constant rate, Normal = sum of many small effects, Gamma = waiting time for several events, Beta = a
distribution over a probability.
""")

nb.code(r"""
fig, axes = plt.subplots(3, 4, figsize=(14, 9))

k = np.arange(0, 21)
for n, p in [(20, 0.1), (20, 0.5), (20, 0.8)]:
    axes[0, 0].plot(k, stats.binom.pmf(k, n, p), "o-", ms=3, label=f"Binomial({n}, {p})")
axes[0, 0].set_title("Binomial: successes in n trials")

k = np.arange(1, 21)
for p in (0.1, 0.3, 0.6):
    axes[0, 1].plot(k, stats.geom.pmf(k, p), "o-", ms=3, label=f"Geometric({p})")
axes[0, 1].set_title("Geometric: trial of the first success")

k = np.arange(0, 21)
for lam in (1, 3, 10):
    axes[0, 2].plot(k, stats.poisson.pmf(k, lam), "o-", ms=3, label=f"Poisson({lam})")
axes[0, 2].set_title("Poisson: rare events in a window")

k = np.arange(0, 41)
for r, p in [(3, 0.3), (3, 0.5), (10, 0.5)]:
    axes[0, 3].plot(k, stats.nbinom.pmf(k, r, p), "o-", ms=3, label=f"NegBin(r={r}, p={p})")
axes[0, 3].set_title("Negative Binomial: failures before r-th success")

x = np.linspace(-0.5, 8, 500)
for a, b in [(0, 1), (2, 8), (3, 4)]:
    axes[1, 0].plot(x, stats.uniform.pdf(x, loc=a, scale=b - a), label=f"Uniform({a}, {b})")
axes[1, 0].set_title("Uniform")

for lam in (0.5, 1, 2):
    axes[1, 1].plot(x, stats.expon.pdf(x, scale=1 / lam), label=f"Exponential(λ={lam})")
axes[1, 1].set_title("Exponential  (scipy: scale = 1/λ)")

xn = np.linspace(-6, 6, 500)
for mu, sd in [(0, 1), (0, 2), (2, 0.5)]:
    axes[1, 2].plot(xn, stats.norm.pdf(xn, mu, sd), label=f"Normal(μ={mu}, σ={sd})")
axes[1, 2].set_title("Normal  (scipy: scale = σ, not σ²)")

xg = np.linspace(0, 15, 500)
for a, beta in [(1, 1), (3, 1), (3, 0.5), (9, 1)]:
    axes[1, 3].plot(xg, stats.gamma.pdf(xg, a=a, scale=1 / beta), label=f"Gamma(α={a}, β={beta})")
axes[1, 3].set_title("Gamma  (shape α, rate β)")

xb = np.linspace(0.001, 0.999, 500)
for a, b in [(1, 1), (2, 5), (5, 2), (0.5, 0.5), (20, 20)]:
    axes[2, 0].plot(xb, stats.beta.pdf(xb, a, b), label=f"Beta({a}, {b})")
axes[2, 0].set_ylim(0, 6); axes[2, 0].set_title("Beta: a distribution over a probability")

for df in (1, 3, 30):
    axes[2, 1].plot(xn, stats.t.pdf(xn, df), label=f"Student-t(ν={df})")
axes[2, 1].plot(xn, stats.norm.pdf(xn), "k--", label="Normal(0, 1)")
axes[2, 1].set_title("Student-t: heavier tails for small ν")

xc = np.linspace(0, 20, 500)
for df in (1, 2, 4, 8):
    axes[2, 2].plot(xc, stats.chi2.pdf(xc, df), label=f"Chi-square({df})")
axes[2, 2].set_ylim(0, 0.6); axes[2, 2].set_title("Chi-square: sum of k squared Normals")

xl = np.linspace(0.01, 6, 500)
for s in (0.25, 0.5, 1.0):
    axes[2, 3].plot(xl, stats.lognorm.pdf(xl, s=s, scale=1.0), label=f"Log-normal(μ=0, σ={s})")
axes[2, 3].set_title("Log-normal: ln Y is Normal")

for ax in axes.ravel():
    ax.legend(fontsize=7); ax.set_xlabel("x"); ax.set_ylabel("probability / density")
fig.suptitle("The gallery: one story and a few parameters each")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Two conventions to burn in: `scipy.stats` continuous families take `loc` and `scale`, so the Exponential
with rate $\lambda$ is `expon(scale=1/lam)`, the Normal with standard deviation $\sigma$ is
`norm(loc=mu, scale=sigma)` and the Gamma with rate $\beta$ is `gamma(a=alpha, scale=1/beta)`. The next
cell checks each object's `.mean()` and `.var()` against the formulas in the guide's cheat sheet — the
quickest way to catch a parametrisation mistake before it poisons everything downstream.
""")

nb.code(r"""
rows = [
    ("Binomial(20, 0.1)",     stats.binom(20, 0.1),              20 * 0.1,        20 * 0.1 * 0.9),
    ("Geometric(0.1)",        stats.geom(0.1),                    1 / 0.1,         0.9 / 0.01),
    ("NegBin(r=3, p=0.1)",    stats.nbinom(3, 0.1),               3 * 0.9 / 0.1,   3 * 0.9 / 0.01),
    ("Poisson(3)",            stats.poisson(3),                   3,               3),
    ("Uniform(2, 8)",         stats.uniform(loc=2, scale=6),      5,               36 / 12),
    ("Exponential(λ=0.2)",    stats.expon(scale=1 / 0.2),         5,               25),
    ("Normal(165, 7²)",       stats.norm(loc=165, scale=7),       165,             49),
    ("Gamma(α=3, β=0.2)",     stats.gamma(a=3, scale=1 / 0.2),    15,              75),
    ("Beta(9, 45)",           stats.beta(9, 45),                  9 / 54,          9 * 45 / (54**2 * 55)),
    ("Chi-square(4)",         stats.chi2(4),                      4,               8),
    ("Student-t(5)",          stats.t(5),                         0,               5 / 3),
]
print(f"{'distribution':22s} {'scipy mean':>11s} {'formula':>9s} {'scipy var':>11s} {'formula':>9s}")
for name, d, m, v in rows:
    print(f"{name:22s} {d.mean():11.4f} {m:9.4f} {d.var():11.4f} {v:9.4f}")

# the five methods every family shares, on Binomial(20, 0.1)
B = stats.binom(20, 0.1)
print(f"\npmf(2) = {B.pmf(2):.4f}   cdf(2) = {B.cdf(2):.4f}   sf(2) = P(X > 2) = {B.sf(2):.4f}   "
      f"ppf(0.95) = {B.ppf(0.95):.0f}   rvs(5) = {B.rvs(5, random_state=rng)}")
""")

# ---------------------------------------------------------------- 2. Bernoulli / Binomial / Multinomial
nb.md(r"""
## 2. Bernoulli, Binomial and Multinomial — by hand, by simulation, by SciPy

$P(S = k) = \binom{n}{k}p^k(1-p)^{n-k}$ with $\mathbb{E}[S] = np$ and $\operatorname{Var}(S) = np(1-p)$.
We implement the PMF from `comb`, compare it with `scipy.stats.binom`, then *simulate* the story — 20
visitors, each buying with probability 0.1 — 100,000 times and compare the empirical frequencies, mean and
variance with the formulas. Finally a Multinomial: 10 support tickets split over three categories.
""")

nb.code(r"""
def binom_pmf(k, n, p):
    return comb(n, k) * p**k * (1 - p) ** (n - k)

n, p = 20, 0.1
ks = np.arange(0, n + 1)
pmf_hand = np.array([binom_pmf(k, n, p) for k in ks])
print("max |hand - scipy| over all k:", np.abs(pmf_hand - stats.binom.pmf(ks, n, p)).max())
print(f"P(S <= 2) = {pmf_hand[:3].sum():.4f}    P(S >= 3) = {pmf_hand[3:].sum():.4f}    P(S >= 5) = {pmf_hand[5:].sum():.4f}")

# simulate the story: 100,000 groups of 20 Bernoulli(0.1) visitors
trials = rng.random((100_000, n)) < p            # Bernoulli(p) as 'uniform < p'
S = trials.sum(axis=1)
emp = np.array([(S == k).mean() for k in ks])
print(f"simulated mean {S.mean():.4f} vs np = {n*p};   simulated var {S.var():.4f} vs np(1-p) = {n*p*(1-p)}")

plt.bar(ks - 0.2, pmf_hand, width=0.4, label="Binomial PMF (formula)")
plt.bar(ks + 0.2, emp, width=0.4, label="100,000 simulated groups of 20 visitors")
plt.xlabel("k = number of purchases among 20 visitors"); plt.ylabel("probability")
plt.title("Binomial(20, 0.1): formula vs simulation"); plt.legend(); plt.show()

# Multinomial: 10 tickets, categories bug / question / feature with probabilities (0.7, 0.2, 0.1)
pi_ = np.array([0.7, 0.2, 0.1])
counts = rng.multinomial(10, pi_, size=100_000)
p_721 = factorial(10) / (factorial(7) * factorial(2) * factorial(1)) * 0.7**7 * 0.2**2 * 0.1
print(f"\nMultinomial: P(7, 2, 1) formula = {p_721:.4f}   simulated = {np.mean(np.all(counts == [7, 2, 1], axis=1)):.4f}")
print(f"means {counts.mean(axis=0)} vs n*pi = {10*pi_};  Cov(N_bug, N_question) = {np.cov(counts[:, 0], counts[:, 1])[0, 1]:.3f} vs -n pi_1 pi_2 = {-10*0.7*0.2:.3f}")
""")

nb.md(r"""
The hand PMF and `scipy.stats.binom` agree to floating-point precision, and the simulation agrees with
both to within Monte Carlo noise ($\pm 0.003$ on each bar). The mean of 2 purchases hides a lot of spread:
3 or more purchases happen in 32% of groups. In the Multinomial the covariance between two categories is
*negative* — a ticket filed as a bug is one that cannot be a question.

**Try it yourself:** change `p` to 0.5 and re-run — the PMF becomes symmetric; then set `n = 200`,
`p = 0.01` and compare the bars with `stats.poisson.pmf(ks, 2)`.
""")

# ---------------------------------------------------------------- 3. Geometric / Negative Binomial
nb.md(r"""
## 3. Geometric and Negative Binomial — waiting for successes

Count the visitors until the first purchase ($p = 0.1$). The guide says $P(X = k) = 0.9^{k-1}\cdot 0.1$,
$\mathbb{E}[X] = 10$, $\operatorname{Var}(X) = 90$, and — memorylessness — $P(X > 8 \mid X > 5) = P(X > 3) = 0.729$.
We simulate the trials literally (keep drawing until a success), then the failures before the *third*
success, whose variance far exceeds its mean: the over-dispersion signature.
""")

nb.code(r"""
p = 0.1
n_sims = 100_000
# trial number of the first success: inverse-CDF for the geometric, equivalent to drawing until success
u = rng.random(n_sims)
X = np.floor(np.log(u) / np.log(1 - p)).astype(int) + 1
print(f"Geometric(0.1): simulated mean {X.mean():.3f} (1/p = 10),  var {X.var():.2f} ((1-p)/p² = 90),  median {np.median(X):.0f} (ln 0.5 / ln 0.9 = {log(0.5)/log(0.9):.2f})")
print(f"P(X = 5) sim {np.mean(X == 5):.4f} vs 0.9^4 * 0.1 = {0.9**4 * 0.1:.4f};   P(X > 10) sim {np.mean(X > 10):.4f} vs 0.9^10 = {0.9**10:.4f}")

# memorylessness by filtering: among runs that survived 5 trials, how many survive 8?
survived5 = X > 5
print(f"memorylessness: P(X > 8 | X > 5) = {np.mean(X[survived5] > 8):.4f}   P(X > 3) = {np.mean(X > 3):.4f}   0.9^3 = {0.9**3:.4f}")

# failures before the 3rd success = sum of three independent (failures-before-success) geometrics
Y = sum(np.floor(np.log(rng.random(n_sims)) / np.log(1 - p)).astype(int) for _ in range(3))
print(f"\nNegBin(r=3, p=0.1): simulated mean {Y.mean():.2f} (27),  var {Y.var():.1f} (270)  ->  var/mean = {Y.var()/Y.mean():.1f}  (Poisson would give 1)")
print(f"P(Y <= 10) sim {np.mean(Y <= 10):.4f}   scipy nbinom.cdf(10, 3, 0.1) = {stats.nbinom.cdf(10, 3, 0.1):.4f}")
print(f"P(Y = 0) sim {np.mean(Y == 0):.4f}   0.1^3 = {0.1**3:.4f}")
""")

nb.code(r"""
# The same simulated runs as pictures: no new random numbers are drawn here
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# left: the remaining wait of the runs that already failed five times, against the original PMF
wait = np.arange(1, 31)
rest = X[X > 5] - 5
ax1.bar(wait, 0.9 ** (wait - 1) * 0.1, color="C0", alpha=0.6, label="P(X = k) = 0.9^(k−1) · 0.1: exact")
ax1.plot(wait, np.bincount(rest, minlength=31)[1:31] / rest.size, "o", color="C3", ms=4,
         label="P(X − 5 = k | X > 5): runs that survived 5 trials")
ax1.axvline(10, color="k", ls="--", lw=1, label="mean 10")
ax1.axvline(7, color="k", ls=":", lw=1, label="median 7")
ax1.set_xlabel("k = trial of the first success"); ax1.set_ylabel("probability")
ax1.set_title("Geometric(0.1) is memoryless: the survivors start over"); ax1.legend(fontsize=8)

# right: a Negative Binomial and a Poisson with the same mean
fails = np.arange(0, 101)
ax2.bar(fails, np.bincount(Y, minlength=101)[:101] / Y.size, width=1.0, color="C0", alpha=0.6,
        label="NegBin(r=3, p=0.1), simulated: mean 27, var 270")
ax2.plot(fails, stats.poisson.pmf(fails, 27), color="C3", lw=1.5, label="Poisson(27): mean 27, var 27")
ax2.set_xlabel("k = failures before the 3rd success"); ax2.set_ylabel("probability")
ax2.set_title("Over-dispersion: variance 10 times the mean"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** the red dots, the distribution of the remaining wait among runs that had already failed
five times, sit on the unconditional bars: the survivors start from scratch. On the right the two
distributions share the mean 27, but the Poisson is a narrow hump while the Negative Binomial spreads from
0 to beyond 80. That width is what "variance ten times the mean" looks like.
""")

nb.md(r"""
Conditioning on "no success in the first 5 trials" and then asking about 3 *further* failures gives the
same probability as asking about 3 failures from scratch — the trials have no memory, so a machine that has
run clean for 60 days is not "overdue". The Negative Binomial's variance-to-mean ratio of 10 is what real
count data (claims, reads, clicks) often look like; a Poisson model of such counts would be badly
over-confident.
""")

# ---------------------------------------------------------------- 4. Poisson
nb.md(r"""
## 4. Poisson — the limit of many rare trials, and the Poisson process

$P(N = k) = e^{-\lambda}\lambda^k/k!$ with mean and variance both $\lambda$. We implement the PMF, then
watch $\text{Binomial}(n, \lambda/n)$ converge to $\text{Poisson}(\lambda)$ as $n$ grows with the mean held
at $\lambda = 3$, and finally simulate a Poisson *process*: Exponential gaps between calls, counted per hour.
""")

nb.code(r"""
def poisson_pmf(k, lam):
    return exp(-lam) * lam**k / factorial(k)

lam = 3
ks = np.arange(0, 16)
pmf_p = np.array([poisson_pmf(k, lam) for k in ks])
print("max |hand - scipy|:", np.abs(pmf_p - stats.poisson.pmf(ks, lam)).max())
print(f"P(N >= 4) = {pmf_p[4:].sum():.4f}  (1 - 13 e^-3 = {1 - 13*exp(-3):.4f})")

fig, ax = plt.subplots(figsize=(8, 4))
print(f"\n{'n':>6s} {'p = λ/n':>9s} {'max_k |Binomial - Poisson|':>28s}")
for n in (5, 10, 30, 180, 1000):
    pb = stats.binom.pmf(ks, n, lam / n)
    ax.plot(ks, pb, "o-", ms=3, alpha=0.8, label=f"Binomial(n={n}, p={lam/n:.4f})")
    print(f"{n:6d} {lam/n:9.4f} {np.abs(pb - pmf_p).max():28.5f}")
ax.plot(ks, pmf_p, "k--", lw=2, label="Poisson(3)")
ax.set_xlabel("k"); ax.set_ylabel("P(X = k)"); ax.set_title("Binomial(n, 3/n) → Poisson(3) as n grows")
ax.legend(fontsize=8); plt.show()
""")

nb.code(r"""
# A Poisson process at rate 3 per hour: exponential gaps, counted per hour over 20,000 hours
rate = 3.0
hours = 20_000
gaps = rng.exponential(1 / rate, size=int(rate * hours * 1.2))   # more than enough gaps
arrivals = np.cumsum(gaps)
arrivals = arrivals[arrivals < hours]
counts = np.bincount(arrivals.astype(int), minlength=hours)      # calls in each hour
print(f"calls per hour: mean {counts.mean():.3f}, var {counts.var():.3f}  (Poisson: both = 3)")
emp = np.array([(counts == k).mean() for k in ks])
print("max |empirical - Poisson(3) pmf|:", np.abs(emp - pmf_p).max())
print(f"gaps between calls (minutes): mean {60*gaps.mean():.2f} (20),  P(gap > 30 min) = {np.mean(gaps > 0.5):.4f} (e^-1.5 = {exp(-1.5):.4f})")
# thinning: keep each call independently with probability 1/3 -> Poisson process of rate 1
kept = arrivals[rng.random(arrivals.size) < 1/3]
kc = np.bincount(kept.astype(int), minlength=hours)
print(f"thinned (keep 1/3): mean {kc.mean():.3f}, var {kc.var():.3f}  (Poisson(1))")
""")

nb.code(r"""
# One process, three views: the timeline, the gaps between its ticks, and the counts per hour
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))

ax = axes[0]
first, first_kept = arrivals[arrivals < 6], kept[kept < 6]
ax.eventplot([first, first_kept], lineoffsets=[1, 0], linelengths=0.6, colors=["C0", "C2"])
for h in range(7):
    ax.axvline(h, color="0.8", lw=0.8, zorder=0)
for h in range(6):
    ax.text(h + 0.5, 1.4, str(np.sum(first.astype(int) == h)), color="C0", ha="center", fontsize=10)
    ax.text(h + 0.5, 0.4, str(np.sum(first_kept.astype(int) == h)), color="C2", ha="center", fontsize=10)
ax.set_xlim(0, 6); ax.set_ylim(-0.5, 1.75)
ax.set_yticks([0, 1]); ax.set_yticklabels(["thinned\n(keep 1/3)", "all calls"])
ax.set_xlabel("time (hours)"); ax.set_ylabel("stream")
ax.set_title("Six hours: a tick per call, counts per hour")

ax = axes[1]
tgrid = np.linspace(0, 120, 300)
ax.hist(60 * gaps, bins=60, range=(0, 120), density=True, alpha=0.6, label="gaps between calls")
ax.plot(tgrid, (3 / 60) * np.exp(-(3 / 60) * tgrid), "r", label="Exponential, rate 3/60 per minute")
ax.axvline(20, color="k", ls="--", lw=1, label="mean gap 20 min")
ax.set_xlabel("gap (minutes)"); ax.set_ylabel("density")
ax.set_title("Exponential gaps ..."); ax.legend(fontsize=8)

ax = axes[2]
show = ks[:9]
ax.bar(show - 0.2, [(counts == k).mean() for k in show], width=0.4, color="C0", alpha=0.7, label="calls per hour, simulated")
ax.plot(show - 0.2, pmf_p[:9], "ko", ms=4, label="Poisson(3) PMF")
ax.bar(show + 0.2, [(kc == k).mean() for k in show], width=0.4, color="C2", alpha=0.7, label="thinned stream, simulated")
ax.plot(show + 0.2, stats.poisson.pmf(show, 1), "k^", ms=4, label="Poisson(1) PMF")
ax.set_xlabel("k = calls in an hour"); ax.set_ylabel("probability")
ax.set_title("... produce Poisson counts"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** the numbers printed above the timeline are just the ticks between neighbouring grid lines.
The gaps between those ticks follow the Exponential in the middle panel, and the counts over all 20,000 hours
follow the Poisson(3) PMF on the right: three views of one process. Thinning keeps each tick with probability
1/3, and its counts land on the Poisson(1) markers, still with mean equal to variance.
""")

nb.md(r"""
At $n = 180$ (the guide's "180 customers each calling with probability 1/60") the largest gap between the
Binomial and the Poisson PMF is below $0.002$; by $n = 1000$ it is below $0.0004$. The process simulation
closes the loop between Sections 4 and 5: Exponential gaps *produce* Poisson counts with mean equal to
variance, and thinning the stream at random keeps it Poisson.

**Try it yourself:** replace the Exponential gaps by `rng.uniform(0, 2/rate, size=...)` (same mean gap,
but regular) and look at the variance of the hourly counts — it collapses well below the mean, because
regular arrivals are *under*-dispersed.
""")

# ---------------------------------------------------------------- 5. Uniform / Exponential
nb.md(r"""
## 5. Uniform and Exponential — inverse-CDF sampling and memorylessness

If $U \sim \text{Uniform}(0, 1)$ then $T = -\ln(U)/\lambda$ has $P(T > t) = P(U < e^{-\lambda t}) = e^{-\lambda t}$:
it is $\text{Exponential}(\lambda)$. We sample this way (no `rng.exponential`), compare the histogram with
$\lambda e^{-\lambda t}$, read off the median and 90th percentile, check memorylessness by filtering, and
estimate the hazard rate $h(t) = f(t)/P(T > t)$ from the sample — it should be flat.
""")

nb.code(r"""
lam = 0.2                                   # buses: one every 5 minutes on average
u = rng.random(100_000)
T = -np.log(u) / lam                        # inverse-CDF sampling

grid = np.linspace(0, 30, 300)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.hist(T, bins=80, range=(0, 30), density=True, alpha=0.6, label="samples  -ln(U)/λ")
ax1.plot(grid, lam * np.exp(-lam * grid), "r", label=r"$\lambda e^{-\lambda t}$")
ax1.set_xlabel("waiting time t (minutes)"); ax1.set_ylabel("density"); ax1.set_title("Exponential(0.2) by inverse CDF"); ax1.legend()

# empirical hazard: fraction of survivors at t that 'fail' in the next dt, divided by dt
edges = np.arange(0, 20, 1.0)
haz = [np.mean((T >= a) & (T < a + 1)) / np.mean(T >= a) for a in edges]
ax2.step(edges, haz, where="post", label="empirical hazard per minute")
ax2.axhline(1 - exp(-lam), color="r", ls="--", label=f"constant: 1 - e^(-λ·1) = {1-exp(-lam):.3f}")
ax2.set_ylim(0, 0.4); ax2.set_xlabel("t (minutes)"); ax2.set_ylabel("P(fail in [t, t+1) | survived to t)")
ax2.set_title("The hazard of an Exponential is flat"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"mean {T.mean():.3f} (5),  sd {T.std():.3f} (5),  median {np.median(T):.3f} (ln2/λ = {log(2)/lam:.3f}),  90th pct {np.quantile(T, 0.9):.2f} (-ln(0.1)/λ = {-log(0.1)/lam:.2f})")
print(f"P(T > 5) = {np.mean(T > 5):.4f} (e^-1 = {exp(-1):.4f});   P(2 < T < 4) = {np.mean((T > 2) & (T < 4)):.4f} ({exp(-0.4)-exp(-0.8):.4f})")
print(f"memorylessness: P(T > 10 | T > 5) = {np.mean(T[T > 5] > 10):.4f}   vs   P(T > 5) = {np.mean(T > 5):.4f}")
print(f"scipy: expon(scale=1/lam).ppf(0.9) = {stats.expon(scale=1/lam).ppf(0.9):.2f};  rng.exponential(1/lam) gives the same distribution")
""")

nb.md(r"""
The hazard plot is the real content: for an Exponential, the chance of the event in the next minute does
not depend on how long you have already waited. Any lifetime whose hazard rises with age (tyres, bearings)
or falls (infant mortality of electronics) is *not* Exponential, and the plot would show a slope.
The median $3.47$ sits well below the mean $5$ — a skewed distribution's typical value is not its mean.
""")

# ---------------------------------------------------------------- 6. Normal
nb.md(r"""
## 6. The Normal — 68–95–99.7, standardisation, a Q–Q plot, and sums

Heights $\sim \mathcal{N}(165, 7^2)$. We check the 68–95–99.7 rule on samples, compute $P(X > 180)$ by
standardising, verify that the *sum* of two independent Normals is exactly Normal, and draw a **Q–Q plot**:
sorted sample quantiles against the theoretical Normal quantiles. Points on the diagonal mean "Normal";
the same plot for Exponential and Student-t data shows what skew and heavy tails look like.
""")

nb.code(r"""
mu, sd = 165, 7
X = rng.normal(mu, sd, 100_000)
for k in (1, 2, 3):
    print(f"P(|X - μ| <= {k}σ): sample {np.mean(np.abs(X - mu) <= k*sd):.4f}   exact {stats.norm.cdf(k) - stats.norm.cdf(-k):.4f}")
z = (180 - mu) / sd
print(f"\nP(X > 180): z = {z:.3f}, 1 - Φ(z) = {stats.norm.sf(z):.4f}, sample {np.mean(X > 180):.4f}")
print(f"99th percentile: μ + 2.326σ = {mu + stats.norm.ppf(0.99)*sd:.2f}, sample {np.quantile(X, 0.99):.2f}")

# sums of independent Normals are exactly Normal
Y = rng.normal(mu, sd, 100_000)
S = X + Y
print(f"\nX + Y: mean {S.mean():.2f} (330), var {S.var():.1f} (98);  P(S > 345) sample {np.mean(S > 345):.4f} vs N(330, 98): {stats.norm.sf(345, 330, sqrt(98)):.4f}")

# log-normal: mean is e^(mu + sigma^2/2), not e^mu
L = np.exp(rng.normal(log(50_000), 0.5, 100_000))
print(f"\nlog-normal income: median {np.median(L):,.0f} (50,000), mean {L.mean():,.0f} (e^(μ+σ²/2) = {50_000*exp(0.125):,.0f}), P(> 100k) = {np.mean(L > 100_000):.4f} ({stats.norm.sf(log(2)/0.5):.4f})")
""")

nb.code(r"""
# The numbers above as pictures: the exact density of X, and the same samples S and L
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))

ax = axes[0]
xs = np.linspace(140, 192, 400)
pdf = stats.norm.pdf(xs, mu, sd)
for k, alpha, share in [(3, 0.12, "99.7%"), (2, 0.2, "95.4%"), (1, 0.3, "68.3%")]:
    inside = np.abs(xs - mu) <= k * sd
    ax.fill_between(xs[inside], pdf[inside], color="C0", alpha=alpha, label=f"±{k}σ: {share}")
ax.fill_between(xs[xs >= 180], pdf[xs >= 180], color="C3", alpha=0.8, label="X > 180 cm: 1.6%")
ax.plot(xs, pdf, "k", lw=1.2)
ax.set_xlabel("height (cm)"); ax.set_ylabel("density")
ax.set_title("N(165, 7²): the 68–95–99.7 rule"); ax.legend(fontsize=8)

ax = axes[1]
xs = np.linspace(290, 370, 300)
ax.hist(S, bins=80, density=True, alpha=0.6, label="X + Y, two independent N(165, 7²)")
ax.plot(xs, stats.norm.pdf(xs, 330, sqrt(98)), "r", label="N(330, 98) pdf: exact")
ax.set_ylim(0, 0.055); ax.set_xlabel("combined height of two people (cm)"); ax.set_ylabel("density")
ax.set_title("A sum of Normals is exactly Normal"); ax.legend(fontsize=8, loc="upper center")

ax = axes[2]
ax.hist(L / 1000, bins=100, range=(0, 200), density=True, alpha=0.6, label="log-normal incomes")
ax.axvline(50_000 * exp(-0.25) / 1000, color="k", ls=":", label=f"mode {50 * exp(-0.25):.1f}k")
ax.axvline(50, color="k", ls="--", label="median 50k")
ax.axvline(50 * exp(0.125), color="k", ls="-", lw=1, label=f"mean {50 * exp(0.125):.1f}k")
ax.set_xlabel("income (thousands)"); ax.set_ylabel("density")
ax.set_title("Log-normal: mode < median < mean"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** the three shaded bands nest: one, two and three standard deviations capture 68%, 95% and
99.7% of the heights, and the red sliver beyond 180 cm is the 1.6% obtained above by standardising. The
histogram of $X + Y$ lies on the $\mathcal{N}(330, 98)$ curve with no approximation involved. The income
panel shows the signature of right skew, mode 38.9k below median 50k below mean 56.7k, so $e^{\mu}$ is the
median, not the mean.
""")

nb.code(r"""
def qq_plot(ax, sample, title):
    s = np.sort(sample)
    n = len(s)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = stats.norm.ppf(probs)
    zs = (s - s.mean()) / s.std()
    ax.plot(theo, zs, ".", ms=2, label="sample, standardised")
    ax.plot([-4, 4], [-4, 4], "r--", lw=1, label="Normal reference y = x")
    ax.set_xlim(-4, 4); ax.set_ylim(-5, 8)
    ax.set_xlabel("theoretical Normal quantile"); ax.set_ylabel("standardised sample quantile"); ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
qq_plot(axes[0], rng.normal(size=5000), "Normal data: on the line")
qq_plot(axes[1], rng.exponential(size=5000), "Exponential data: skewed (bent)")
qq_plot(axes[2], rng.standard_t(3, size=5000), "Student-t(3) data: heavy tails (S-shape)")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
In the Q–Q plot a right-skewed sample bends upward on the right (its large quantiles are larger than a
Normal's) and a heavy-tailed sample peels away at *both* ends (its most extreme points lie off the chart).
This is the single most useful diagnostic before you trust a Gaussian assumption — on residuals of a
regression, on the per-example losses of a model, or on the sampling distribution of a metric.

**Try it yourself:** feed `np.log(L)` (the log of the log-normal incomes) into `qq_plot` — the log
transform turns the skewed sample into a straight line.
""")

# ---------------------------------------------------------------- 7. Gamma, Beta, chi-square, t
nb.md(r"""
## 7. Gamma, Beta, Chi-square and Student-t — built from their ingredients

Each of these is *made of* simpler pieces: the Gamma($k$, $\lambda$) is a sum of $k$ Exponentials, the
Chi-square($k$) is a sum of $k$ squared standard Normals, and the Student-t($\nu$) is $Z/\sqrt{V/\nu}$. We
construct each one from its ingredients and overlay the `scipy.stats` density. Then the Beta: a grid of
shapes, and the posterior $\text{Beta}(2 + 7, 2 + 43)$ for a conversion rate after 7 successes in 50 trials.
""")

nb.code(r"""
n_s = 100_000
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

# Gamma(3, rate 0.2) = sum of three Exponential(0.2): time until the third bus
G = rng.exponential(1 / 0.2, size=(n_s, 3)).sum(axis=1)
xg = np.linspace(0, 60, 300)
axes[0].hist(G, bins=80, density=True, alpha=0.6, label="sum of 3 Exponential(0.2)")
axes[0].plot(xg, stats.gamma.pdf(xg, a=3, scale=1 / 0.2), "r", label="Gamma(α=3, β=0.2) pdf")
axes[0].set_title("Gamma = sum of Exponentials"); axes[0].set_xlabel("minutes until 3rd bus")
print(f"Gamma: mean {G.mean():.2f} (15), sd {G.std():.2f} ({sqrt(3)/0.2:.2f}), P(> 20) = {np.mean(G > 20):.4f}  (= P(Poisson(4) <= 2) = {stats.poisson.cdf(2, 4):.4f})")

# Chi-square(4) = sum of four squared standard Normals
C = (rng.normal(size=(n_s, 4)) ** 2).sum(axis=1)
xc = np.linspace(0, 20, 300)
axes[1].hist(C, bins=80, range=(0, 20), density=True, alpha=0.6, label="Z1² + Z2² + Z3² + Z4²")
axes[1].plot(xc, stats.chi2.pdf(xc, 4), "r", label="Chi-square(4) pdf")
axes[1].set_title("Chi-square = sum of squared Normals"); axes[1].set_xlabel("value")
print(f"Chi-square(4): mean {C.mean():.3f} (4), var {C.var():.2f} (8), 95th pct {np.quantile(C, 0.95):.3f} (scipy {stats.chi2.ppf(0.95, 4):.3f})")

# Student-t(3) = Z / sqrt(V/3)
Z = rng.normal(size=n_s); V = (rng.normal(size=(n_s, 3)) ** 2).sum(axis=1)
Tt = Z / np.sqrt(V / 3)
xt = np.linspace(-6, 6, 300)
axes[2].hist(Tt, bins=200, range=(-6, 6), density=True, alpha=0.6, label="Z / sqrt(V/3)")
axes[2].plot(xt, stats.t.pdf(xt, 3), "r", label="Student-t(3) pdf")
axes[2].plot(xt, stats.norm.pdf(xt), "k--", lw=1, label="Normal(0,1)")
axes[2].set_yscale("log"); axes[2].set_ylim(1e-4, 1); axes[2].set_title("Student-t: heavy tails (log scale)"); axes[2].set_xlabel("value")
print(f"Student-t(3): P(|T| > 3) = {np.mean(np.abs(Tt) > 3):.4f} (exact {2*stats.t.sf(3, 3):.4f});  Normal: {2*stats.norm.sf(3):.4f}  ->  ratio {stats.t.sf(3, 3)/stats.norm.sf(3):.1f}x")
print(f"t quantiles 97.5%: ν=4 {stats.t.ppf(0.975, 4):.3f}, ν=9 {stats.t.ppf(0.975, 9):.3f}, ν=29 {stats.t.ppf(0.975, 29):.3f}, Normal {stats.norm.ppf(0.975):.3f}")
for ax in axes:
    ax.legend(fontsize=7); ax.set_ylabel("density")
plt.tight_layout(); plt.show()
""")

nb.code(r"""
# Beta shapes, and a Beta posterior for a conversion rate
theta = np.linspace(0.001, 0.999, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
for a, b in [(1, 1), (0.5, 0.5), (2, 5), (5, 2), (2, 2), (10, 10)]:
    ax1.plot(theta, stats.beta.pdf(theta, a, b), label=f"Beta({a}, {b}): mean {a/(a+b):.2f}")
ax1.set_ylim(0, 4.5); ax1.set_xlabel("θ"); ax1.set_ylabel("density"); ax1.set_title("Beta shapes"); ax1.legend(fontsize=7)

# prior Beta(2, 2); data: 7 conversions in 50 visits; posterior Beta(9, 45)
a0, b0, s, f = 2, 2, 7, 43
post = stats.beta(a0 + s, b0 + f)
ax2.plot(theta, stats.beta.pdf(theta, a0, b0), "gray", ls="--", label="prior Beta(2, 2)")
ax2.plot(theta, post.pdf(theta), "C3", label=f"posterior Beta({a0+s}, {b0+f})")
ax2.axvline(s / (s + f), color="k", ls=":", label=f"raw rate {s/(s+f):.2f}")
ax2.set_xlim(0, 0.6); ax2.set_xlabel("conversion rate θ"); ax2.set_ylabel("density"); ax2.set_title("Beta–Binomial updating: 7 of 50 converted"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"posterior mean {post.mean():.4f} (9/54), sd {post.std():.4f}, mode {(a0+s-1)/(a0+s+b0+f-2):.4f}")
print(f"P(θ > 0.2 | data) = {post.sf(0.2):.4f};   95% credible interval {post.ppf(0.025):.3f} – {post.ppf(0.975):.3f}")
# the posterior IS the normalised likelihood x prior: check numerically on the grid
unnorm = theta**(a0 + s - 1) * (1 - theta)**(b0 + f - 1)
print(f"grid check of the density at θ=0.15: {unnorm[np.argmin(np.abs(theta-0.15))] / np.trapezoid(unnorm, theta):.3f} vs scipy {post.pdf(0.15):.3f}")

# Bayesian A/B: page B converted 12 of 50 -> Beta(14, 40); P(θ_B > θ_A) by sampling both posteriors
a_s = rng.beta(9, 45, 200_000); b_s = rng.beta(14, 40, 200_000)
print(f"P(θ_B > θ_A) = {np.mean(b_s > a_s):.3f}")
""")

nb.md(r"""
The Gamma histogram built from three Exponentials sits exactly on `gamma.pdf`, and its tail probability
$P(G > 20)$ equals the Poisson probability of at most 2 arrivals in 20 minutes — the same event described
twice. On the log scale the Student-t(3) density is still $2 \times 10^{-3}$ at $\pm 6$, where the Normal
($6 \times 10^{-9}$) has long left the chart: a power-law tail decays polynomially, not like $e^{-x^2/2}$. (On
log–log axes, as in the survival plot of Section 8, it would be a straight line.) Beyond $|x| = 3$ the
$t_3$ has 21 times the mass of the Normal. The Beta posterior
has simply added the counts to the prior parameters; the result is a full distribution for the conversion
rate, from which "probability the rate exceeds 20%" or "probability B beats A" are one line each.
""")

# ---------------------------------------------------------------- 8. moments / MGF / tails
nb.md(r"""
## 8. Moments, an MGF identity, and heavy tails

Skewness $\gamma_1 = \mathbb{E}[(X-\mu)^3]/\sigma^3$ and excess kurtosis $\gamma_2 = \mathbb{E}[(X-\mu)^4]/\sigma^4 - 3$
from samples; the MGF $M(t) = \mathbb{E}[e^{tX}]$ estimated by averaging $e^{tX}$ and compared with the
closed form $e^{\lambda(e^t - 1)}$ for the Poisson; the sum of two Poissons checked against
$\text{Poisson}(\lambda_1 + \lambda_2)$; and a look at how a Pareto tail differs from an Exponential tail
with the *same mean*.
""")

nb.code(r"""
def skew_kurt(x):
    z = (x - x.mean()) / x.std()
    return np.mean(z**3), np.mean(z**4) - 3

samples = {
    "Normal":            rng.normal(size=200_000),
    "Uniform":           rng.random(200_000),
    "Exponential":       rng.exponential(size=200_000),
    "Bernoulli(0.1)":    (rng.random(200_000) < 0.1).astype(float),
    "Poisson(3)":        rng.poisson(3, 200_000).astype(float),
    "Student-t(5)":      rng.standard_t(5, 200_000),
}
theory = {"Normal": (0, 0), "Uniform": (0, -1.2), "Exponential": (2, 6), "Bernoulli(0.1)": ((1 - 0.2) / sqrt(0.09), (1 - 6 * 0.09) / 0.09),
          "Poisson(3)": (1 / sqrt(3), 1 / 3), "Student-t(5)": (0, 6)}
print(f"{'distribution':16s} {'skew (sample)':>14s} {'skew (theory)':>14s} {'kurt (sample)':>14s} {'kurt (theory)':>14s}")
for name, x in samples.items():
    s, k = skew_kurt(x); ts, tk = theory[name]
    print(f"{name:16s} {s:14.3f} {ts:14.3f} {k:14.3f} {tk:14.3f}")

# MGF of Poisson(3): E[e^{tX}] vs exp(lam (e^t - 1)); and the sum of Poisson(2) + Poisson(3)
N3 = samples["Poisson(3)"]
for t in (0.1, 0.25, 0.5):
    print(f"t = {t}: mean of e^(tX) = {np.mean(np.exp(t * N3)):.4f}   exp(3(e^t - 1)) = {exp(3 * (exp(t) - 1)):.4f}")
N2 = rng.poisson(2, 200_000)
S = N2 + N3.astype(int)
ks = np.arange(0, 16)
print("\nmax |empirical pmf of Poisson(2)+Poisson(3)  -  Poisson(5) pmf| =",
      np.abs(np.array([(S == k).mean() for k in ks]) - stats.poisson.pmf(ks, 5)).max())
""")

nb.code(r"""
# light vs heavy tails with the same mean 3: Exponential(mean 3) vs Pareto(alpha=1.5, x_min=1)
n = 200_000
E = rng.exponential(3.0, n)
P = (rng.pareto(1.5, n) + 1.0)           # numpy's pareto is the Lomax; +1 gives the classical Pareto with x_min = 1
print(f"means: Exponential {E.mean():.3f}, Pareto {P.mean():.3f}  (both 3 in theory)")
print(f"P(X > 30):  Exponential {np.mean(E > 30):.2e} (e^-10 = {exp(-10):.2e}),  Pareto {np.mean(P > 30):.2e} (30^-1.5 = {30**-1.5:.2e})")
print(f"P(X > 300): Exponential {np.mean(E > 300):.1e} (e^-100),           Pareto {np.mean(P > 300):.2e} (300^-1.5 = {300**-1.5:.2e})")
print(f"largest of {n} samples: Exponential {E.max():.1f}, Pareto {P.max():.0f};  share of total in the top 1%: Exponential {np.sort(E)[-n//100:].sum()/E.sum():.3f}, Pareto {np.sort(P)[-n//100:].sum()/P.sum():.3f}")
print(f"sample kurtosis: Exponential {skew_kurt(E)[1]:.1f} (6),  Pareto {skew_kurt(P)[1]:.0f} (infinite: it grows with n and never settles)")

xs = np.logspace(0, 3, 200)
plt.loglog(xs, np.exp(-xs / 3), label="Exponential(mean 3): P(X > x) = e^(-x/3)")
plt.loglog(xs, xs ** -1.5, label="Pareto(α=1.5): P(X > x) = x^(-1.5)")
plt.loglog(xs, [np.mean(P > x) for x in xs], "k.", ms=3, label="Pareto sample survival")
plt.ylim(1e-6, 1); plt.xlabel("x (log scale)"); plt.ylabel("P(X > x)  (log scale)")
plt.title("Same mean, different tails: exponential decay vs power law"); plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
The sample skewness and kurtosis match the theory for every light-tailed distribution (the Student-t(5)
kurtosis is the noisiest: its population value 6 exists, but the *variance of the estimate* involves the
eighth moment, which the $t_5$ does not have, so even 200,000 samples leave it a little off); for the Pareto
the sample kurtosis is a number that *grows with the sample size* because the population value is infinite. On
the log–log survival plot a power law is a straight line and an exponential plunges: the Pareto puts
$2 \times 10^{-4}$ of its mass beyond 100 times its mean, where the Exponential puts $e^{-100}$. The MGF
check confirms $\mathbb{E}[e^{tN}] = e^{\lambda(e^t - 1)}$, and the sum of two independent Poissons is
indistinguishable from a Poisson with the summed rate — the MGF proof in the guide made concrete.
""")

# ---------------------------------------------------------------- 9. LLN
nb.md(r"""
## 9. The law of large numbers — running averages, and the Cauchy exception

$P(|\bar X_n - \mu| \ge \varepsilon) \le \sigma^2/(n\varepsilon^2)$. We plot the running average of
Exponential(1) samples for several seeds together with the Chebyshev band and the (much tighter) CLT band,
then do the same for a Student-t(2) sample, whose variance is infinite, and for a Cauchy sample, whose mean
does not exist — that running average never settles.
""")

nb.code(r"""
n = 20_000
ns = np.arange(1, n + 1)
fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(15, 4))
for seed in range(6):
    r = np.random.default_rng(seed)
    ax1.plot(ns, np.cumsum(r.exponential(1.0, n)) / ns, lw=0.8, alpha=0.8)
    ax2.plot(ns, np.cumsum(r.standard_cauchy(n)) / ns, lw=0.8, alpha=0.8)
    ax3.plot(ns, np.cumsum(r.standard_t(2, n)) / ns, lw=0.8, alpha=0.8)
ax1.axhline(1, color="k", ls="--", lw=1)
ax1.fill_between(ns, 1 - 1.96 / np.sqrt(ns), 1 + 1.96 / np.sqrt(ns), color="C0", alpha=0.15, label="CLT 95% band ±1.96σ/√n")
ax1.plot(ns, 1 + np.sqrt(1 / (ns * 0.05)), "r:", lw=1, label="Chebyshev 95% band ±σ/√(0.05 n)")
ax1.plot(ns, 1 - np.sqrt(1 / (ns * 0.05)), "r:", lw=1)
ax1.set_xscale("log"); ax1.set_ylim(0, 2); ax1.set_xlabel("n (log scale)"); ax1.set_ylabel("running average")
ax1.set_title("Exponential(1): six running averages → 1"); ax1.legend(fontsize=8, loc="upper right")
ax3.set_xscale("log"); ax3.set_ylim(-2, 2); ax3.axhline(0, color="k", ls="--", lw=1, label="true mean 0")
ax3.set_xlabel("n (log scale)"); ax3.set_ylabel("running average"); ax3.set_title("Student-t(2): converges, but in jumps"); ax3.legend(fontsize=8, loc="upper right")
ax2.set_xscale("log"); ax2.set_ylim(-6, 6); ax2.axhline(0, color="k", ls="--", lw=1, label="centre 0 (no mean exists)")
ax2.set_xlabel("n (log scale)"); ax2.set_ylabel("running average"); ax2.set_title("Cauchy: the average never settles"); ax2.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.show()

# the gambler's fallacy in numbers: the proportion converges while the count drifts
flips = rng.integers(0, 2, size=(1000, 10_000))
heads = flips.cumsum(axis=1)
for m in (100, 1_000, 10_000):
    excess = heads[:, m - 1] - m / 2
    print(f"after {m:>6d} flips: sd of (heads - n/2) = {excess.std():6.2f} (√n/2 = {sqrt(m)/2:6.2f});   sd of proportion = {(excess / m).std():.4f} (1/(2√n) = {1/(2*sqrt(m)):.4f})")
""")

nb.code(r"""
# The same 30 coin sequences read twice: as an excess count, and as a proportion
flip_n = np.arange(1, 10_001)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for i in range(30):
    ax1.plot(flip_n, heads[i] - flip_n / 2, lw=0.5, alpha=0.6)
    ax2.plot(flip_n, heads[i] / flip_n - 0.5, lw=0.5, alpha=0.6)
ax1.plot(flip_n, np.sqrt(flip_n), "k--", lw=1.2, label="±2 · √n/2 (95% envelope)")
ax1.plot(flip_n, -np.sqrt(flip_n), "k--", lw=1.2)
ax1.set_xlabel("n flips"); ax1.set_ylabel("heads − n/2")
ax1.set_title("The excess count drifts like √n/2: luck is never repaid"); ax1.legend(fontsize=8, loc="upper left")
ax2.plot(flip_n, 1 / np.sqrt(flip_n), "k--", lw=1.2, label="±2 · 1/(2√n) (95% envelope)")
ax2.plot(flip_n, -1 / np.sqrt(flip_n), "k--", lw=1.2)
ax2.set_xscale("log"); ax2.set_xlim(10, 10_000); ax2.set_ylim(-0.3, 0.3)
ax2.set_xlabel("n flips (log scale)"); ax2.set_ylabel("heads/n − 1/2")
ax2.set_title("The proportion converges like 1/(2√n): luck is diluted"); ax2.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** the same 30 coin sequences fan out on the left, where the excess count keeps growing like
$\sqrt{n}/2$ with nothing pulling it back to zero, and funnel in on the right, because dividing by $n$ shrinks
that same excess like $1/(2\sqrt{n})$. The law of large numbers is the right panel, not the left.
""")

nb.md(r"""
Every Exponential trajectory ends inside the CLT band; the Chebyshev band is far wider — it is a guarantee
valid for *any* distribution, paid for with looseness. The Cauchy panel is the warning: each trajectory
is occasionally hit by a sample so large that the average jumps, and no amount of data cures it, because
the mean does not exist. The Student-t with 2 degrees of freedom in the middle panel (finite mean, infinite
variance) sits between the two: its average converges, but more erratically than the Exponential's and
somewhat more slowly (its error shrinks like $\sqrt{\ln n / n}$, roughly 3 to 4 times the Exponential's spread at
$n = 20{,}000$). In the coin table the *count* of excess heads grows like $\sqrt{n}/2$ while the
*proportion* shrinks like $1/(2\sqrt{n})$: early luck is diluted, never repaid.
""")

# ---------------------------------------------------------------- 10. CLT
nb.md(r"""
## 10. The central limit theorem — how fast, for which source?

$Z_n = \sqrt{n}(\bar X_n - \mu)/\sigma \to \mathcal{N}(0, 1)$. For each of three sources — Uniform
(symmetric), Exponential (skewness 2) and Bernoulli(0.05) (skewness 4.1) — we draw 20,000 means of
$n = 1, 2, 5, 30$ observations, standardise, and overlay the standard Normal. The **Kolmogorov distance**
$\sup_z |F_n(z) - \Phi(z)|$ measures how far each histogram is from the bell, and the Berry–Esseen bound
$0.4748\,\rho/(\sigma^3\sqrt{n})$ says how far it is *allowed* to be.
""")

nb.code(r"""
def standardised_means(sampler, n, reps, mu, sigma):
    x = sampler((reps, n)).mean(axis=1)
    return np.sqrt(n) * (x - mu) / sigma

def kolmogorov_distance(z):
    z = np.sort(z); m = len(z); F = stats.norm.cdf(z)
    return max(np.max(np.arange(1, m + 1) / m - F), np.max(F - np.arange(0, m) / m))

p_rare = 0.05
sources = {
    "Uniform(0,1)":     (lambda s: rng.random(s),                        0.5,     sqrt(1 / 12)),
    "Exponential(1)":   (lambda s: rng.exponential(1.0, s),              1.0,     1.0),
    "Bernoulli(0.05)":  (lambda s: (rng.random(s) < p_rare).astype(float), p_rare, sqrt(p_rare * (1 - p_rare))),
}
ns = (1, 2, 5, 30)
reps = 20_000
zgrid = np.linspace(-4, 5, 200)
fig, axes = plt.subplots(3, 4, figsize=(13, 8))
kd = {}
for i, (name, (sampler, mu, sigma)) in enumerate(sources.items()):
    for j, n in enumerate(ns):
        z = standardised_means(sampler, n, reps, mu, sigma)
        kd[(name, n)] = kolmogorov_distance(z)
        ax = axes[i, j]
        if name.startswith("Bernoulli"):
            # a lattice distribution: one bar per number of successes, its area the exact probability
            spacing = 1 / (sigma * sqrt(n))
            atoms = np.sqrt(n) * (np.arange(n + 1) / n - mu) / sigma
            ax.bar(atoms, stats.binom.pmf(np.arange(n + 1), n, p_rare) / spacing, width=spacing,
                   alpha=0.6, edgecolor="white", linewidth=0.5)
            ax.set_xlim(-4, 5)
        else:
            ax.hist(z, bins=np.linspace(-4, 4, 81), density=True, alpha=0.6, label="standardised means")
            ax.set_xlim(-4, 4)
        ax.plot(zgrid, stats.norm.pdf(zgrid), "r", lw=1.2, label="N(0, 1)")
        ax.set_title(f"{name}, n = {n}\nKolmogorov distance {kd[(name, n)]:.3f}", fontsize=9)
        ax.set_ylim(0, 0.9)
        if j == 0: ax.set_ylabel("density")
        if i == 2: ax.set_xlabel("standardised mean z")
axes[0, 0].legend(fontsize=7)
fig.suptitle("Histograms of √n(X̄ - μ)/σ against the standard Normal (Bernoulli row: one bar per number of successes)")
plt.tight_layout(); plt.show()
""")

nb.code(r"""
# how fast: Kolmogorov distance vs n, with the Berry-Esseen bound for the exponential source
ns_long = np.array([1, 2, 5, 10, 30, 100, 300, 1000])
rho_exp = 2.4146      # E|X - 1|^3 for Exponential(1)
rho_bern = p_rare * (1 - p_rare) * ((1 - p_rare)**2 + p_rare**2)
sig3_bern = (p_rare * (1 - p_rare)) ** 1.5
print(f"{'n':>5s} {'Uniform':>9s} {'Exponential':>12s} {'BE bound (exp)':>15s} {'Bernoulli(0.05)':>16s} {'BE bound (bern)':>16s}")
rows = []
for n in ns_long:
    d = [kolmogorov_distance(standardised_means(s, n, reps, mu, sg)) for s, mu, sg in sources.values()]
    rows.append(d)
    print(f"{n:5d} {d[0]:9.4f} {d[1]:12.4f} {0.4748*rho_exp/sqrt(n):15.4f} {d[2]:16.4f} {0.4748*rho_bern/sig3_bern/sqrt(n):16.4f}")
rows = np.array(rows)
plt.figure(figsize=(11, 4.5))
plt.loglog(ns_long, rows[:, 1], "o-", label="Exponential: observed distance")
plt.loglog(ns_long, rows[:, 2], "s-", label="Bernoulli(0.05): observed distance")
plt.loglog(ns_long, rows[:, 0], "^-", label="Uniform: observed distance")
plt.loglog(ns_long, 0.4748 * rho_exp / np.sqrt(ns_long), "k--", lw=1, label="Berry–Esseen bound, Exponential")
plt.loglog(ns_long, 0.3 / np.sqrt(ns_long), "k:", lw=1, label="reference slope 1/√n")
plt.axhline(0.8687 / np.sqrt(reps), color="grey", ls="--", lw=1, label="simulation noise ≈ 0.87/√20,000 (the average score of a perfect Normal)")
plt.xlabel("n"); plt.ylabel("sup |F_n - Φ|"); plt.title("Distance to the Normal shrinks like 1/√n, down to the simulation noise")
plt.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left"); plt.tight_layout(); plt.show()

skew = lambda p: (1 - 2 * p) / sqrt(p * (1 - p))
print(f"\nskewness of the source: Uniform 0, Exponential 2, Bernoulli(0.05) {skew(0.05):.2f};  of the mean of 30: {2/sqrt(30):.2f} and {skew(0.05)/sqrt(30):.2f}")
print(f"n needed for the skewness of the mean to fall below 0.3: Exponential {(2/0.3)**2:.0f}, Bernoulli(0.05) {(skew(0.05)/0.3)**2:.0f}, Bernoulli(0.01) {(skew(0.01)/0.3)**2:.0f}")
""")

nb.md(r"""
The Uniform is bell-shaped by $n = 2$ (the triangular distribution) and nearly indistinguishable from the
Normal at $n = 5$; from $n = 10$ on its Kolmogorov distance is pure simulation noise: a perfect Normal
sample of 20,000 scores $0.87/\sqrt{20{,}000} \approx 0.006$ on average, give or take about $0.002$, and the
table's $0.004$–$0.005$ lies within that. The Exponential needs $n = 30$ to look
symmetric. The Bernoulli(0.05) mean of 30 is still a coarse skewed staircase, one bar per number of successes
0, 1, 2, … (the four tallest, for 0–3 successes, carry 94% of the mass), and only at $n = 1000$ does it
approach the bell. (That row plots the exact probabilities, one bar per number of successes, because a
fixed-width histogram cannot show a lattice honestly.) On the log–log plot the Bernoulli curve has slope $-1/2$
from $n = 10$ on and the Exponential curve has it up to about $n = 30$; beyond that simulation noise adds to its
small remaining distance and the curve flattens towards that noise level, around which the Uniform curve already scatters. The
Berry–Esseen bound for the Exponential sits above the Exponential curve with exactly the slope $-1/2$ (that it
runs along the Bernoulli curve is a coincidence; the Bernoulli's own bound, in the table, is larger). The *rate*
is universal, the *constant* is set by skewness. This is why "$n = 30$ is enough" is a rule of thumb for roughly
symmetric data and a trap for rare events.

**Try it yourself:** set `p_rare = 0.01` and re-run both cells — then read the number of samples a
"significant" A/B test on a 1%-conversion page really needs.

### When the CLT fails: Cauchy means and the polling margin of error

Averages of $n$ Cauchy variables are again Cauchy — the same distribution for every $n$. And on the
success side: a poll of 1,000 voters has a sample proportion that is Normal to high accuracy, so the 95%
margin of error $1.96\sqrt{p(1-p)/n}$ is trustworthy.
""")

nb.code(r"""
# Cauchy: the distribution of the mean of n samples does not depend on n
means = {}
for n in (1, 10, 100, 1000):
    m = rng.standard_cauchy((20_000, n)).mean(axis=1)
    means[n] = m
    q = np.quantile(m, [0.25, 0.5, 0.75])
    print(f"n = {n:>5d}: quartiles of the sample mean {q[0]:7.3f} {q[1]:7.3f} {q[2]:7.3f}   (Cauchy quartiles: -1, 0, 1)")

# polling: sample proportion from 1,000 voters, true p = 0.52
p_true, n_poll = 0.52, 1000
props = (rng.random((50_000, n_poll)) < p_true).mean(axis=1)
moe = 1.96 * sqrt(p_true * (1 - p_true) / n_poll)
print(f"\npoll of {n_poll}: sd of the proportion {props.std():.4f} (√(p(1-p)/n) = {sqrt(p_true*(1-p_true)/n_poll):.4f}),  margin of error ±{100*moe:.1f} points")
print(f"coverage of p ± margin: {np.mean(np.abs(props - p_true) <= moe):.4f} (CLT promises 0.95)")
print(f"sample size for a ±2-point margin: n = (1.96 · 0.5 / 0.02)² = {(1.96*0.5/0.02)**2:.0f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# 8% of a Cauchy lies outside ±8, so each bar is scaled by ALL 20,000 values, not just those shown
edges = np.linspace(-8, 8, 81)
for n, m in means.items():
    ax1.hist(m, bins=edges, weights=np.full(m.size, 1 / (m.size * (edges[1] - edges[0]))),
             histtype="step", lw=1.3, label=f"mean of n = {n}")
xs = np.linspace(-8, 8, 400)
ax1.plot(xs, 1 / (pi * (1 + xs**2)), "k--", lw=1, label="Cauchy pdf 1/(π(1+x²))")
ax1.set_xlabel("sample mean"); ax1.set_ylabel("density")
ax1.set_title("Cauchy: the mean has the same distribution for every n"); ax1.legend(fontsize=8)

# proportions live on a 0.001 lattice: bin edges fall between lattice points, or the histogram shows a false comb
ax2.hist(props, bins=np.arange(0.4605, 0.5805, 0.003), density=True, alpha=0.6, label="sample proportion, 50,000 polls of 1,000")
xs = np.linspace(0.46, 0.58, 300)
ax2.plot(xs, stats.norm.pdf(xs, p_true, sqrt(p_true * (1 - p_true) / n_poll)), "r", label="N(0.52, 0.0158²)")
ax2.axvline(p_true - moe, color="k", ls=":", lw=1.2, label="p ± 3.1 points (95%)")
ax2.axvline(p_true + moe, color="k", ls=":", lw=1.2)
ax2.set_ylim(0, 34); ax2.set_xlabel("proportion for A"); ax2.set_ylabel("density")
ax2.set_title("Poll of 1,000: the proportion is Normal"); ax2.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** the four Cauchy histograms are one curve: averaging 1,000 samples has narrowed nothing, which
is what "the mean does not exist" looks like. The poll's proportions form the bell
$\mathcal{N}(0.52, 0.0158^2)$, whose central 95% is the $\pm 3.1$-point margin of error drawn on it.
""")

nb.md(r"""
The quartiles of the Cauchy sample mean are $-1, 0, 1$ whether you average 1 observation or 1,000: there
is no concentration at all, and any "standard error" computed from such data is a fiction. The poll is the
opposite extreme — a bounded, nearly symmetric source — and the 95% interval covers the truth 95% of
the time, as promised.
""")

# ---------------------------------------------------------------- 11. Monte Carlo
nb.md(r"""
## 11. Monte Carlo — error bars that shrink like $1/\sqrt{n}$, and antithetic variates

Estimate $I = \int_0^1 e^{-x^2}\,dx = 0.74682$ as $\mathbb{E}[e^{-U^2}]$ with $U \sim \text{Uniform}(0, 1)$.
For each $n$ we repeat the whole estimate 200 times to measure the actual error, compare it with the
reported standard error $\hat\sigma/\sqrt{n}$, and then use **antithetic variates** — average
$g(U)$ and $g(1-U)$ — to cut the variance without extra samples.
""")

nb.code(r"""
g = lambda u: np.exp(-u ** 2)
I_exact = 0.7468241328124271          # from scipy.integrate.quad

def mc_estimate(n, rng, antithetic=False):
    if antithetic:
        u = rng.random(n // 2)
        vals = 0.5 * (g(u) + g(1 - u))          # n/2 antithetic pairs = n function evaluations
    else:
        vals = g(rng.random(n))
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))

est, se = mc_estimate(10_000, rng)
print(f"n = 10,000:  estimate {est:.5f} ± {se:.5f} (standard error),  exact {I_exact:.5f},  error {est - I_exact:+.5f}")
est_a, se_a = mc_estimate(10_000, rng, antithetic=True)
print(f"antithetic:  estimate {est_a:.5f} ± {se_a:.5f},  error {est_a - I_exact:+.5f}   -> variance ratio ≈ {(se / se_a)**2:.2g}x")

ns = np.array([100, 300, 1_000, 3_000, 10_000, 30_000, 100_000])
reps = 200
rmse, rmse_a, reported = [], [], []
for n in ns:
    e = np.array([mc_estimate(n, rng)[0] for _ in range(reps)])
    ea = np.array([mc_estimate(n, rng, antithetic=True)[0] for _ in range(reps)])
    rmse.append(np.sqrt(np.mean((e - I_exact) ** 2)))
    rmse_a.append(np.sqrt(np.mean((ea - I_exact) ** 2)))
    reported.append(np.mean([mc_estimate(n, rng)[1] for _ in range(20)]))
rmse, rmse_a, reported = map(np.array, (rmse, rmse_a, reported))
sigma_g = sqrt(0.0404)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.errorbar(ns, [mc_estimate(n, np.random.default_rng(n))[0] for n in ns],
             yerr=[1.96 * mc_estimate(n, np.random.default_rng(n))[1] for n in ns], fmt="o", capsize=3, label="estimate ± 1.96 SE")
ax1.axhline(I_exact, color="r", ls="--", label="exact value")
ax1.set_xscale("log"); ax1.set_xlabel("n"); ax1.set_ylabel("estimate of ∫₀¹ e^(-x²) dx"); ax1.set_title("Error bars shrink with n"); ax1.legend()
ax2.loglog(ns, rmse, "o-", label="actual RMSE, plain")
ax2.loglog(ns, reported, "x--", label="reported SE  σ̂/√n")
ax2.loglog(ns, sigma_g / np.sqrt(ns), "k:", label="theory σ_g/√n")
ax2.loglog(ns, rmse_a, "s-", label="actual RMSE, antithetic")
ax2.set_xlabel("n (function evaluations)"); ax2.set_ylabel("error"); ax2.set_title("Monte Carlo error ∝ 1/√n; antithetic ≈ 25x less variance"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
for n, r, ra, rep in zip(ns, rmse, rmse_a, reported):
    print(f"n = {n:>7d}: RMSE plain {r:.5f}   reported SE {rep:.5f}   RMSE antithetic {ra:.5f}")
""")

nb.code(r"""
# Why antithetic pairs work on one integrand and not on another (a local generator: the rng stream is untouched)
u_pairs = np.random.default_rng(8).random(2000)
h = lambda u: np.sin(np.pi * u)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(g(u_pairs), g(1 - u_pairs), ".", ms=3, alpha=0.5, label="2,000 pairs (U, 1 − U)")
ax1.set_xlabel("g(U) = exp(−U²)"); ax1.set_ylabel("g(1 − U)")
ax1.set_title(f"Monotone integrand: corr = {np.corrcoef(g(u_pairs), g(1 - u_pairs))[0, 1]:.2f}, pairs cancel noise"); ax1.legend(fontsize=8)
ax2.plot(h(u_pairs), h(1 - u_pairs), ".", ms=3, alpha=0.5, label="2,000 pairs (U, 1 − U)")
ax2.set_xlabel("h(U) = sin(πU)"); ax2.set_ylabel("h(1 − U)")
ax2.set_title(f"Symmetric integrand: corr = {np.corrcoef(h(u_pairs), h(1 - u_pairs))[0, 1]:+.2f}, pairs are copies"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** for the monotone integrand the pairs lie on a falling curve: when $g(U)$ is large, $g(1-U)$ is
small, so the pair average has far less spread than either member. A correlation $\rho = -0.96$ gives the
variance ratio $1/(1+\rho) = 25$ measured above. The symmetric $\sin(\pi U)$ sends $U$ and $1-U$ to the same
value, so the pair average is just $h(U)$ and the second evaluation adds no information.
""")

nb.md(r"""
The reported standard error $\hat\sigma/\sqrt{n}$ tracks the actual root-mean-square error at every $n$ —
that is the CLT delivering a trustworthy error bar for an integral whose value we pretended not to know.
Both curves have slope $-1/2$: 100 times the samples for one more digit. Antithetic sampling moves the
whole curve down by a factor of about 5 in error (25 in variance) at zero extra cost, because
$e^{-u^2}$ is monotone and $g(U)$, $g(1-U)$ are strongly negatively correlated. Module 18 adds
importance sampling and MCMC for distributions you cannot sample directly.

**Try it yourself:** replace `g` by `lambda u: np.sin(np.pi * u)` (symmetric about 1/2) and `I_exact` by
`2 / np.pi`: the antithetic pair $g(U) + g(1-U)$ is now *perfectly* correlated and the pair average is just $g(U)$.
The trick does worse than nothing: $n$ evaluations now give only $n/2$ independent values, so the variance doubles
(the printed ratio falls to about 0.5, and the antithetic RMSE curve rises above the plain one).
""")

nb.md(r"""
## Summary

- A named distribution is a generative story plus a few parameters; `pmf/pdf`, `cdf`, `sf`, `ppf`, `rvs` are the five verbs, and `scipy.stats` takes `scale`, never a variance or a rate.
- Bernoulli → Binomial → Multinomial describe labels and counts of successes; their negative log-likelihoods are the cross-entropy losses.
- Geometric and Exponential waiting times are memoryless; the Negative Binomial and Gamma are their sums; over-dispersion (variance ≫ mean) calls for the Negative Binomial instead of the Poisson.
- Binomial($n, \lambda/n$) → Poisson($\lambda$); Exponential gaps produce Poisson counts; the Poisson has mean = variance.
- The Normal is closed under sums and standardisation; the 68–95–99.7 rule and the Q–Q plot are the quick checks; the log-normal's mean is $e^{\mu + \sigma^2/2}$.
- Gamma, Chi-square and Student-t are built from Exponentials and Normals; the Beta is a distribution over a probability and updates by adding counts.
- MGFs multiply under independent sums (Poisson + Poisson = Poisson); heavy tails show as straight lines on a log–log survival plot and as kurtosis that never settles.
- The LLN makes averages converge at rate $\sigma/\sqrt{n}$ — unless the mean does not exist (Cauchy); luck is diluted, not repaid.
- The CLT makes standardised averages Normal at rate $1/\sqrt{n}$ with a constant set by skewness; rare events need thousands of samples, Cauchy data never get there.
- Monte Carlo turns expectations into averages with error bar $\hat\sigma/\sqrt{n}$; antithetic variates shrink $\hat\sigma$ for monotone integrands.

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
from math import comb, exp, factorial, log, sqrt, pi
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
    "Module 08 — Exercises",
    intro_md=r"""
These exercises follow the order of the concept guide: the discrete families (Binomial, Poisson), the
continuous ones (Exponential, Normal, Beta, Gamma), then the limit theorems and Monte Carlo. Several checks
compare your answer against a *hidden* value, so compute carefully rather than guessing; `scipy.stats` is
allowed wherever the prompt does not forbid it.
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- 1. Binomial & Poisson probabilities (warm-up)
SOL_1 = r"""
p_exactly_3 = float(stats.binom.pmf(3, 20, 0.1))
p_at_most_3 = float(stats.binom.cdf(3, 20, 0.1))
p_at_least_5 = float(stats.binom.sf(4, 20, 0.1))
k_95 = int(stats.binom.ppf(0.95, 20, 0.1))
p_pois_0 = float(stats.poisson.pmf(0, 3))
p_pois_ge_6 = float(stats.poisson.sf(5, 3))
lam_1pct = float(-log(0.01))
"""
ns1 = run(SOL_1)
ex.add(
    title="Binomial and Poisson probabilities with scipy.stats", difficulty=1,
    prompt_md=r"""
A product page converts 10% of visitors; 20 visitors arrive. A help desk receives on average 3 calls per
hour. Using `scipy.stats.binom` and `scipy.stats.poisson` (methods `pmf`, `cdf`, `sf`, `ppf`), assign these
**floats** (and one **int**):

* `p_exactly_3` — exactly 3 of the 20 visitors buy;
* `p_at_most_3` — at most 3 buy;
* `p_at_least_5` — at least 5 buy (careful: `sf(k)` is $P(X > k)$, strictly greater);
* `k_95` — the smallest integer $k$ with $P(S \le k) \ge 0.95$ (an `int`);
* `p_pois_0` — no call in a given hour;
* `p_pois_ge_6` — six or more calls in an hour;
* `lam_1pct` — the hourly rate $\lambda$ at which the probability of a call-free hour would be exactly 1% (solve $e^{-\lambda} = 0.01$ by hand).
""",
    starter=r"""
p_exactly_3 = None
p_at_most_3 = None
p_at_least_5 = None
k_95 = None
p_pois_0 = None
p_pois_ge_6 = None
lam_1pct = None
# YOUR CODE HERE
""",
    solution=SOL_1,
    check=rf"""
check_close('P(exactly 3 of 20 buy) = C(20,3) 0.1^3 0.9^17', p_exactly_3, comb(20, 3) * 0.1**3 * 0.9**17)
check_hash('P(at most 3 buy)', p_at_most_3, '{sha_of(ns1["p_at_most_3"], decimals=4)}', decimals=4)
check_hash('P(at least 5 buy)', p_at_least_5, '{sha_of(ns1["p_at_least_5"], decimals=4)}', decimals=4)
try:
    k_int = int(k_95) if float(k_95) == int(k_95) else None     # a whole-number float, which is what scipy's ppf returns, is just as right as an int
except Exception:
    k_int = None
check_hash('95% quantile of the purchase count', k_int, '{sha_of(ns1["k_95"])}')
check_close('P(no call in an hour) = e^-3', p_pois_0, exp(-3))
check_hash('P(6 or more calls)', p_pois_ge_6, '{sha_of(ns1["p_pois_ge_6"], decimals=4)}', decimals=4)
check_hash('rate for a 1% chance of a call-free hour', lam_1pct, '{sha_of(ns1["lam_1pct"], decimals=3)}', decimals=3)
try:
    consistent = abs((1 - p_at_most_3) - (stats.binom.pmf(4, 20, 0.1) + p_at_least_5)) < 1e-9
except Exception:
    consistent = False
check('consistency: 1 - P(<= 3) = P(= 4) + P(>= 5)', consistent)
""",
    explanation_md=r"""
`pmf`, `cdf` and `sf` are the three ways of slicing a discrete distribution, and the off-by-one trap is the
survival function: `binom.sf(4, 20, 0.1)` is $P(S > 4) = P(S \ge 5)$, so "at least 5" needs `sf(4)`, not
`sf(5)`. `ppf(0.95)` returns the smallest $k$ whose CDF reaches 0.95 — here 4, because
$P(S \le 3) = 0.867 < 0.95 \le P(S \le 4) = 0.957$. For the Poisson, $P(N = 0) = e^{-\lambda}$, so a 1% chance of
a quiet hour means $\lambda = -\ln 0.01 = 4.605$ calls per hour. The consistency check guards against the
most common slip, mixing up strict and non-strict inequalities.
""",
)

# ---------------------------------------------------------------- 2. Poisson PMF from scratch (warm-up)
SOL_2 = r"""
def poisson_pmf(k, lam):
    k = np.asarray(k)
    log_p = -lam + k * np.log(lam) - np.array([sum(np.log(np.arange(1, kk + 1))) for kk in np.atleast_1d(k)]).reshape(k.shape)
    return np.exp(log_p)

def poisson_cdf(k, lam):
    return float(np.sum(poisson_pmf(np.arange(0, int(k) + 1), lam)))

def binomial_pmf(k, n, p):
    return comb(n, k) * p**k * (1 - p) ** (n - k)

diff_180 = max(abs(binomial_pmf(k, 180, 1/60) - float(poisson_pmf(k, 3))) for k in range(0, 40))
"""
ns2 = run(SOL_2, r"""
p_big = float(poisson_pmf(200, 150))
""")
ex.add(
    title="Implement the Poisson PMF from scratch and watch the Binomial converge", difficulty=1,
    prompt_md=r"""
Implement, with NumPy only (no `scipy.stats`, no `math.factorial` — which overflows for large $k$):

* `poisson_pmf(k, lam)` → $e^{-\lambda}\lambda^k/k!$ for an integer or integer array `k`; compute it in log
  space as $\exp(-\lambda + k\ln\lambda - \ln k!)$ with $\ln k! = \sum_{j=1}^{k}\ln j$ so that
  `poisson_pmf(200, 150)` works without overflow;
* `poisson_cdf(k, lam)` → $P(N \le k)$ as a float, by summing the PMF;
* `binomial_pmf(k, n, p)` → $\binom{n}{k}p^k(1-p)^{n-k}$ (here `comb` is fine).

Then compute `diff_180`, the largest absolute difference over $k = 0, \dots, 39$ between
`binomial_pmf(k, 180, 1/60)` and `poisson_pmf(k, 3)` — the guide's "180 customers each calling with
probability 1/60" example.
""",
    starter=r"""
def poisson_pmf(k, lam):
    # YOUR CODE HERE
    return None

def poisson_cdf(k, lam):
    # YOUR CODE HERE
    return None

def binomial_pmf(k, n, p):
    # YOUR CODE HERE
    return None

diff_180 = None
# YOUR CODE HERE
""",
    solution=SOL_2,
    check=rf"""
try:
    got = np.asarray(poisson_pmf(np.arange(0, 16), 3), dtype=float)
except Exception:
    got = None
check_close('poisson_pmf(k, 3) matches scipy for k = 0..15', got, stats.poisson.pmf(np.arange(0, 16), 3), tol=1e-9)
try:
    got = float(poisson_pmf(200, 150))
except Exception:
    got = None
check_close('poisson_pmf(200, 150) computed in log space (no overflow)', got, float(stats.poisson.pmf(200, 150)), tol=1e-12)
try:
    got = poisson_cdf(3, 3)
except Exception:
    got = None
check_hash('P(N <= 3) for lambda = 3', got, '{sha_of(ns2["poisson_cdf"](3, 3), decimals=4)}', decimals=4)
try:
    got = binomial_pmf(2, 20, 0.1)
except Exception:
    got = None
check_close('binomial_pmf(2, 20, 0.1)', got, 190 * 0.01 * 0.9**18)
check_hash('largest |Binomial(180, 1/60) - Poisson(3)| over k = 0..39', diff_180, '{sha_of(ns2["diff_180"], decimals=4)}', decimals=4)
check('the two PMFs differ by less than 0.002 everywhere', diff_180 is not None and diff_180 < 0.002)
""",
    explanation_md=r"""
Working in log space is the professional habit: $150^{200}$ and $200!$ both overflow a double, but
$-150 + 200\ln 150 - \ln 200! \approx -11.1$ is a perfectly ordinary number whose exponential is
$1.5 \times 10^{-5}$. (`scipy.special.gammaln(k + 1)` is the library version of $\ln k!$.) The Binomial–Poisson
comparison shows the limit in Section 4 of the guide at work: with $n = 180$ and $p = 1/60$ the largest gap
over all $k$ is below $0.002$, so for "many rare trials" the one-parameter Poisson replaces the two-parameter
Binomial at essentially no cost. A common slip is to apply `np.log` to the integer array `k` directly for
$\ln k!$ — you need the cumulative sum of $\ln j$, not $\ln k$.
""",
)

# ---------------------------------------------------------------- 3. conceptual: choose the distribution (warm-up)
ex.add_conceptual(
    title="Choose the distribution", difficulty=1,
    prompt_md=r"""
For each quantity, name the most appropriate named distribution, its parameters (in words), and one
assumption that must hold for it to be right:

1. The number of defective screws in a box of 200 from a line with a 1.5% defect rate.
2. The number of e-mails arriving in a 10-minute window at a server that receives 300 per hour.
3. The time until the next e-mail arrives at that server.
4. The number of cold calls a salesperson makes until the first sale, if each call succeeds with probability 0.04.
5. The time until the third e-mail arrives.
6. Your uncertainty about a coin's probability of heads after seeing 12 heads in 20 flips, starting from no prior preference.
7. The class label (one of 10 digits) that an image classifier predicts, as a random variable.
8. The average of 50 independent measurements of a resistor, each with a slightly skewed error distribution.
9. The number of insurance claims per policyholder per year, when the sample variance of the counts is five times the sample mean.
""",
    answer_md=r"""
1. Binomial($n = 200$, $p = 0.015$) — screws must be defective independently with the same probability; since
   $n$ is large and $p$ small, Poisson($3$) is an excellent approximation.
2. Poisson($\lambda = 300 \times 10/60 = 50$) — arrivals must be independent at a constant rate (no bursts, no
   batch deliveries); if the variance of the counts exceeds the mean, use a Negative Binomial.
3. Exponential(rate $= 300$ per hour $= 5$ per minute) — same Poisson-process assumption; mean 12 seconds.
4. Geometric($p = 0.04$) on the trial number — calls independent with constant success probability; mean 25 calls.
5. Gamma(shape 3, rate 5 per minute) — the sum of three independent Exponential inter-arrival times; mean 36 seconds.
6. Beta($1 + 12, 1 + 8$) $=$ Beta(13, 9) — a Uniform (Beta(1, 1)) prior updated with 12 successes and 8
   failures; it requires the flips to be independent given the coin's bias.
7. Categorical over 10 classes with probability vector given by the softmax output; the one-hot label is
   its sample, and the multinomial would describe the class counts over many images.
8. Approximately Normal($\mu$, $\sigma^2/50$) by the central limit theorem — the measurements must be
   independent with finite variance; skewness $\gamma_1$ of a single measurement leaves a residual skewness of
   $\gamma_1/\sqrt{50}$ in the average, which is small for mild skew.
9. Negative Binomial (mean $\mu$, dispersion $r$ with $\mu + \mu^2/r = 5\mu$) — a Poisson would force variance
   $=$ mean and understate every tail probability; the heterogeneity across policyholders (a Gamma-distributed
   rate) is exactly what the Negative Binomial models.
""",
)

# ---------------------------------------------------------------- 4. Exponential quantiles & memorylessness (core)
SOL_4 = r"""
def expon_quantile(q, lam):
    return float(-np.log(1 - q) / lam)

def expon_tail(t, lam):
    return float(np.exp(-lam * t))

def conditional_tail(s, t, lam):
    return float(np.exp(-lam * (s + t)) / np.exp(-lam * s))

def sample_expon(n, lam, rng):
    return -np.log(rng.random(n)) / lam
"""
ns4 = run(SOL_4, r"""
med = expon_quantile(0.5, 0.2)
q90 = expon_quantile(0.9, 0.2)
tail10 = expon_tail(10, 0.2)
cond = conditional_tail(5, 10, 0.2)
s = sample_expon(50_000, 0.2, np.random.default_rng(3))
emp_cond = float(np.mean(s[s > 5] > 15))
""")
ex.add(
    title="Exponential quantiles, tails and memorylessness", difficulty=2,
    prompt_md=r"""
Buses arrive as a Poisson process with rate $\lambda = 0.2$ per minute. Implement, with NumPy only:

* `expon_quantile(q, lam)` → the $q$-quantile of $\text{Exponential}(\lambda)$, by inverting $F(t) = 1 - e^{-\lambda t}$;
* `expon_tail(t, lam)` → $P(T > t)$;
* `conditional_tail(s, t, lam)` → $P(T > s + t \mid T > s)$ computed *from the definition* of conditional
  probability (ratio of two tail probabilities), not by assuming memorylessness;
* `sample_expon(n, lam, rng)` → `n` samples via the inverse-CDF method `-np.log(rng.random(n)) / lam`.

Then verify numerically that `conditional_tail(5, 10, 0.2)` equals `expon_tail(10, 0.2)`, and that the
empirical version — among 50,000 simulated waits longer than 5 minutes, the fraction longer than 15 — agrees.
""",
    starter=r"""
def expon_quantile(q, lam):
    # YOUR CODE HERE
    return None

def expon_tail(t, lam):
    # YOUR CODE HERE
    return None

def conditional_tail(s, t, lam):
    # YOUR CODE HERE
    return None

def sample_expon(n, lam, rng):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_4,
    check=rf"""
try:
    got = expon_quantile(0.5, 0.2)
except Exception:
    got = None
check_hash('median waiting time (minutes)', got, '{sha_of(ns4["med"], decimals=3)}', decimals=3)
try:
    got = expon_quantile(0.9, 0.2)
except Exception:
    got = None
check_hash('90th percentile of the wait', got, '{sha_of(ns4["q90"], decimals=3)}', decimals=3)
try:
    got = expon_quantile(0.9, 0.2)
    ok_scipy = abs(got - stats.expon(scale=1/0.2).ppf(0.9)) < 1e-9
except Exception:
    ok_scipy = False
check('quantile agrees with scipy expon(scale=1/lam).ppf', ok_scipy)
try:
    got = expon_tail(10, 0.2)
except Exception:
    got = None
check_close('P(T > 10) = e^-2', got, exp(-2))
try:
    got = conditional_tail(5, 10, 0.2)
except Exception:
    got = None
check_hash('P(T > 15 | T > 5)', got, '{sha_of(ns4["cond"], decimals=4)}', decimals=4)
try:
    memoryless = abs(conditional_tail(5, 10, 0.2) - expon_tail(10, 0.2)) < 1e-12
except Exception:
    memoryless = False
check('memorylessness: P(T > 15 | T > 5) == P(T > 10)', memoryless)
try:
    s = np.asarray(sample_expon(50_000, 0.2, np.random.default_rng(3)), dtype=float)
    ok_shape = s.shape == (50_000,) and np.all(s > 0)
    emp = float(np.mean(s[s > 5] > 15))
    ok_mean = abs(s.mean() - 5) < 0.1
except Exception:
    ok_shape, emp, ok_mean = False, None, False
check('sampler returns 50,000 positive values with mean close to 5', ok_shape and ok_mean)
check('empirical conditional tail within 0.02 of e^-2', emp is not None and abs(emp - exp(-2)) < 0.02)
""",
    explanation_md=r"""
Inverting the CDF gives $Q(q) = -\ln(1-q)/\lambda$: the median is $\ln 2/0.2 = 3.47$ minutes and the 90th
percentile $-\ln(0.1)/0.2 = 11.5$ minutes — the mean of 5 minutes is neither. The conditional tail is
$e^{-\lambda(s+t)}/e^{-\lambda s} = e^{-\lambda t}$, which is exactly $P(T > t)$: that cancellation *is*
memorylessness, and the simulation confirms it by filtering (keep the waits above 5 and ask about 15). The
sampler is inverse-CDF sampling with $U$ in place of $1 - U$ (same distribution). The classic error is to
write the tail as $e^{-t/\lambda}$ — mixing up rate and scale — which the comparison with
`expon(scale=1/lam)` catches.
""",
)

# ---------------------------------------------------------------- 5. Normal z-scores, quality control (core)
SOL_5 = r"""
def z_score(x, mu, sigma):
    return (x - mu) / sigma

def p_below(x, mu, sigma):
    return float(stats.norm.cdf(z_score(x, mu, sigma)))

def p_between(a, b, mu, sigma):
    return float(stats.norm.cdf(z_score(b, mu, sigma)) - stats.norm.cdf(z_score(a, mu, sigma)))

def sigma_for_tail(limit, mu, target_prob):
    # largest sigma such that P(X < limit) <= target_prob, for limit < mu
    return float((mu - limit) / stats.norm.ppf(1 - target_prob))

def two_sided_interval(mu, sigma, coverage):
    z = stats.norm.ppf(0.5 + coverage / 2)
    return (float(mu - z * sigma), float(mu + z * sigma))
"""
ns5 = run(SOL_5, r"""
pb = p_below(490, 500, 4)
pbet = p_between(495, 505, 500, 4)
sig = sigma_for_tail(490, 500, 0.001)
lo, hi = two_sided_interval(500, 4, 0.99)
p_tall = 1 - p_below(180, 165, 7)
""")
ex.add(
    title="Normal z-scores and a quality-control problem", difficulty=2,
    prompt_md=r"""
A filling machine dispenses $\mathcal{N}(500, 4^2)$ grams. Implement (you may use `scipy.stats.norm`, but
all arguments must be converted to $z$-scores first — the check inspects `z_score`):

* `z_score(x, mu, sigma)` → $(x - \mu)/\sigma$;
* `p_below(x, mu, sigma)` → $P(X < x)$;
* `p_between(a, b, mu, sigma)` → $P(a < X < b)$;
* `sigma_for_tail(limit, mu, target_prob)` → the largest $\sigma$ such that $P(X < \text{limit}) \le$ `target_prob`
  (for `limit < mu`): solve $\Phi\big((\text{limit} - \mu)/\sigma\big) = $ `target_prob` for $\sigma$;
* `two_sided_interval(mu, sigma, coverage)` → the tuple $(\mu - z\sigma, \mu + z\sigma)$ containing the central
  `coverage` fraction of the mass (e.g. $z = 1.96$ for 0.95).

Questions answered by the checks: the fraction of packets below 490 g; the fraction within $\pm 5$ g; the
$\sigma$ needed so that at most 0.1% of packets fall below 490 g; the central 99% interval; and the
fraction of adults taller than 180 cm if heights are $\mathcal{N}(165, 7^2)$.
""",
    starter=r"""
def z_score(x, mu, sigma):
    # YOUR CODE HERE
    return None

def p_below(x, mu, sigma):
    # YOUR CODE HERE
    return None

def p_between(a, b, mu, sigma):
    # YOUR CODE HERE
    return None

def sigma_for_tail(limit, mu, target_prob):
    # YOUR CODE HERE
    return None

def two_sided_interval(mu, sigma, coverage):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_5,
    check=rf"""
try:
    got = z_score(490, 500, 4)
except Exception:
    got = None
check_close('z-score of 490 g is -2.5', got, -2.5)
try:
    got = p_below(490, 500, 4)
except Exception:
    got = None
check_hash('fraction of packets below 490 g', got, '{sha_of(ns5["pb"], decimals=4)}', decimals=4)
try:
    got = p_between(495, 505, 500, 4)
except Exception:
    got = None
check_hash('fraction within +/- 5 g', got, '{sha_of(ns5["pbet"], decimals=4)}', decimals=4)
try:
    got = p_between(496, 504, 500, 4)
except Exception:
    got = None
check_close('68.27% within one sigma', got, stats.norm.cdf(1) - stats.norm.cdf(-1))
try:
    got = sigma_for_tail(490, 500, 0.001)
    ok_round_trip = abs(p_below(490, 500, got) - 0.001) < 1e-9
except Exception:
    got, ok_round_trip = None, False
check_hash('sigma needed for at most 0.1% below 490 g', got, '{sha_of(ns5["sig"], decimals=3)}', decimals=3)
check('round trip: p_below(490, 500, sigma_for_tail(...)) == 0.001', ok_round_trip)
try:
    lo, hi = two_sided_interval(500, 4, 0.99)
except Exception:
    lo, hi = None, None
check_hash('central 99% interval', (lo, hi), '{sha_of((ns5["lo"], ns5["hi"]), decimals=3)}', decimals=3)
try:
    got = 1 - p_below(180, 165, 7)
except Exception:
    got = None
check_hash('fraction of adults taller than 180 cm', got, '{sha_of(ns5["p_tall"], decimals=4)}', decimals=4)
""",
    explanation_md=r"""
Everything reduces to the standard Normal through $z = (x - \mu)/\sigma$. $490$ g is $z = -2.5$, so
$\Phi(-2.5) = 0.0062$ of packets are underweight; $\pm 5$ g is $\pm 1.25\sigma$, covering $0.789$. For the
engineering target invert the problem: $\Phi(-10/\sigma) = 0.001$ means $10/\sigma = \Phi^{-1}(0.999) = 3.09$,
so $\sigma = 3.24$ g — a 19% reduction in variability. The central 99% interval uses $z = \Phi^{-1}(0.995) = 2.576$,
not $2.326$ (that is the one-sided 99% point) — the most frequent mistake in this exercise. The height
question is the same arithmetic with $z = 15/7 = 2.14$, giving $1.6\%$.
""",
)

# ---------------------------------------------------------------- 6. Beta posterior (core)
SOL_6 = r"""
def beta_posterior(a0, b0, successes, failures):
    return (a0 + successes, b0 + failures)

def beta_mean(a, b):
    return a / (a + b)

def beta_sd(a, b):
    return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))

def credible_interval(a, b, level=0.95):
    lo = float(stats.beta.ppf((1 - level) / 2, a, b))
    hi = float(stats.beta.ppf(1 - (1 - level) / 2, a, b))
    return (lo, hi)

def prob_b_beats_a(a_A, b_A, a_B, b_B, n_samples, rng):
    return float(np.mean(rng.beta(a_B, b_B, n_samples) > rng.beta(a_A, b_A, n_samples)))
"""
ns6 = run(SOL_6, r"""
a, b = beta_posterior(2, 2, 7, 43)
m = beta_mean(a, b); s = beta_sd(a, b)
lo, hi = credible_interval(a, b, 0.95)
p_gt = float(stats.beta.sf(0.2, a, b))
pb = prob_b_beats_a(9, 45, 14, 40, 200_000, np.random.default_rng(11))
m_big = beta_mean(*beta_posterior(2, 2, 70, 430))
""")
ex.add(
    title="A Beta posterior for a conversion rate", difficulty=2,
    prompt_md=r"""
A landing page is believed, before any data, to convert with a rate described by a $\text{Beta}(2, 2)$ prior.
It then converts 7 of 50 visitors. Implement

* `beta_posterior(a0, b0, successes, failures)` → the tuple `(a, b)` of the posterior Beta parameters;
* `beta_mean(a, b)` and `beta_sd(a, b)` → the mean $\frac{a}{a+b}$ and standard deviation $\sqrt{\frac{ab}{(a+b)^2(a+b+1)}}$ as floats;
* `credible_interval(a, b, level=0.95)` → the tuple of the $\frac{1-\text{level}}{2}$ and $1 - \frac{1-\text{level}}{2}$ quantiles (`scipy.stats.beta.ppf`);
* `prob_b_beats_a(a_A, b_A, a_B, b_B, n_samples, rng)` → the Monte Carlo probability that a draw from
  $\text{Beta}(a_B, b_B)$ exceeds a draw from $\text{Beta}(a_A, b_A)$, using `rng.beta(a_B, b_B, n_samples)` **first** and
  `rng.beta(a_A, b_A, n_samples)` second.

Then compare the posterior mean with the raw rate $7/50$ and the prior mean $0.5$, and compute the posterior
mean you would get from 70 conversions in 500 visits (same rate, ten times the data).
""",
    starter=r"""
def beta_posterior(a0, b0, successes, failures):
    # YOUR CODE HERE
    return None

def beta_mean(a, b):
    # YOUR CODE HERE
    return None

def beta_sd(a, b):
    # YOUR CODE HERE
    return None

def credible_interval(a, b, level=0.95):
    # YOUR CODE HERE
    return None

def prob_b_beats_a(a_A, b_A, a_B, b_B, n_samples, rng):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_6,
    check=rf"""
try:
    a, b = beta_posterior(2, 2, 7, 43)
except Exception:
    a, b = None, None
check('posterior parameters are (9, 45)', (a, b) == (9, 45))
try:
    got = beta_mean(9, 45)
except Exception:
    got = None
check_hash('posterior mean', got, '{sha_of(ns6["m"], decimals=4)}', decimals=4)
try:
    got = beta_sd(9, 45)
except Exception:
    got = None
check_hash('posterior standard deviation', got, '{sha_of(ns6["s"], decimals=4)}', decimals=4)
try:
    got = beta_sd(1, 1)
except Exception:
    got = None
check_close('sd of Beta(1, 1) = Uniform is 1/sqrt(12)', got, 1/sqrt(12))
try:
    got = credible_interval(9, 45, 0.95)
except Exception:
    got = None
check_hash('95% credible interval', got, '{sha_of((ns6["lo"], ns6["hi"]), decimals=3)}', decimals=3)
try:
    got = prob_b_beats_a(9, 45, 14, 40, 200_000, np.random.default_rng(11))
except Exception:
    got = None
check('P(theta_B > theta_A) is close to 0.885 (within 0.01)', got is not None and abs(got - 0.885) < 0.01)
try:
    got = beta_mean(*beta_posterior(2, 2, 70, 430))
except Exception:
    got = None
check_hash('posterior mean with ten times the data', got, '{sha_of(ns6["m_big"], decimals=4)}', decimals=4)
check('more data pulls the mean to within 0.01 of the raw rate 0.14', got is not None and abs(got - 0.14) < 0.01)
""",
    explanation_md=r"""
Beta–Binomial conjugacy makes the update pure bookkeeping: successes are added to $a$, failures to $b$, so
$\text{Beta}(2, 2)$ becomes $\text{Beta}(9, 45)$ with mean $9/54 = 0.167$ — between the raw rate $0.14$ and the
prior mean $0.5$, because the prior counts as 4 pseudo-observations against 50 real ones. With 500 visits the
prior is 4 against 500 and the posterior mean $72/504 = 0.143$ is almost the raw rate; the standard deviation
shrinks by about $\sqrt{10}$. The credible interval is read straight off the posterior quantiles, and
$P(\theta_B > \theta_A)$ is a one-line Monte Carlo estimate — compare the contortions a frequentist test
needs for the same question. The typical slip is to add the number of *trials* to $b$ instead of the number of
failures.
""",
)

# ---------------------------------------------------------------- 7. Gamma waiting times & Gamma–Poisson identity (core)
SOL_7 = r"""
def gamma_pdf(x, k, lam):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = lam**k * x[pos] ** (k - 1) * np.exp(-lam * x[pos]) / factorial(k - 1)
    return out

def gamma_tail_via_poisson(t, k, lam):
    # P(k-th event after time t) = P(at most k-1 events in [0, t]) with count ~ Poisson(lam t)
    mu = lam * t
    return float(sum(np.exp(-mu) * mu**j / factorial(j) for j in range(k)))

def sample_gamma_as_sum(n, k, lam, rng):
    return rng.exponential(1 / lam, size=(n, k)).sum(axis=1)
"""
ns7 = run(SOL_7, r"""
pdf10 = float(gamma_pdf(10.0, 3, 0.2))
tail20 = gamma_tail_via_poisson(20, 3, 0.2)
tail_scipy = float(stats.gamma.sf(20, a=3, scale=5))
g = sample_gamma_as_sum(100_000, 3, 0.2, np.random.default_rng(5))
emp_tail = float(np.mean(g > 20))
""")
ex.add(
    title="Gamma waiting times and the Gamma–Poisson identity", difficulty=2,
    prompt_md=r"""
Buses arrive at rate $\lambda = 0.2$ per minute; the time until the $k$-th bus is $\text{Gamma}(k, \lambda)$ with
integer shape $k$ (an *Erlang* distribution). Implement, using NumPy and `math.factorial` only (no `scipy.stats`
inside the functions):

* `gamma_pdf(x, k, lam)` → the density $\frac{\lambda^k}{(k-1)!}x^{k-1}e^{-\lambda x}$ for $x > 0$ (0 otherwise), vectorised over `x`;
* `gamma_tail_via_poisson(t, k, lam)` → $P(\text{$k$-th bus arrives after } t)$ computed as the Poisson
  probability of **at most $k - 1$ arrivals** in $[0, t]$, $\sum_{j=0}^{k-1} e^{-\lambda t}(\lambda t)^j/j!$;
* `sample_gamma_as_sum(n, k, lam, rng)` → `n` samples as sums of `k` independent Exponentials, drawn with a single
  call `rng.exponential(1/lam, size=(n, k))` summed along axis 1.

Then confirm that `gamma_tail_via_poisson(20, 3, 0.2)` equals `scipy.stats.gamma.sf(20, a=3, scale=5)` and that the
empirical tail of 100,000 simulated third-bus times agrees.
""",
    starter=r"""
def gamma_pdf(x, k, lam):
    # YOUR CODE HERE
    return None

def gamma_tail_via_poisson(t, k, lam):
    # YOUR CODE HERE
    return None

def sample_gamma_as_sum(n, k, lam, rng):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_7,
    check=rf"""
try:
    got = float(np.asarray(gamma_pdf(10.0, 3, 0.2)).ravel()[0])
except Exception:
    got = None
check_hash('Gamma(3, 0.2) density at x = 10', got, '{sha_of(ns7["pdf10"], decimals=4)}', decimals=4)
try:
    xs = np.linspace(0.5, 40, 80)
    got = np.asarray(gamma_pdf(xs, 3, 0.2), dtype=float)
except Exception:
    got = None
check_close('gamma_pdf matches scipy gamma.pdf on a grid', got, stats.gamma.pdf(np.linspace(0.5, 40, 80), a=3, scale=5), tol=1e-9)
try:
    got = np.asarray(gamma_pdf(np.array([-1.0, 0.0]), 3, 0.2), dtype=float)
except Exception:
    got = None
check_close('density is 0 for x <= 0', got, np.array([0.0, 0.0]))
try:
    got = gamma_tail_via_poisson(20, 3, 0.2)
except Exception:
    got = None
check_hash('P(third bus after 20 minutes)', got, '{sha_of(ns7["tail20"], decimals=4)}', decimals=4)
check_close('Gamma tail equals Poisson CDF: matches scipy gamma.sf(20, a=3, scale=5)', got, float(stats.gamma.sf(20, a=3, scale=5)), tol=1e-9)
try:
    got = gamma_tail_via_poisson(5, 1, 0.2)
except Exception:
    got = None
check_close('k = 1 reduces to the Exponential tail e^-1', got, exp(-1))
try:
    g = np.asarray(sample_gamma_as_sum(100_000, 3, 0.2, np.random.default_rng(5)), dtype=float)
    ok = g.shape == (100_000,) and abs(g.mean() - 15) < 0.15 and abs(np.mean(g > 20) - stats.gamma.sf(20, a=3, scale=5)) < 0.01
except Exception:
    ok = False
check('simulated third-bus times: mean about 15 and tail about 0.238', ok)
""",
    explanation_md=r"""
"The third bus arrives after time $t$" and "at most two buses arrived by time $t$" are the same event, so the
Gamma tail is a Poisson CDF: $P(G > 20) = e^{-4}(1 + 4 + 8) = 0.238$. This identity is how the Gamma, the
Exponential ($k = 1$) and the Poisson process fit together, and it is also how `scipy` could evaluate the
Erlang tail without any special function. The density needs $(k-1)!$, not $k!$ — the Gamma function
satisfies $\Gamma(k) = (k-1)!$ — and must be zero for $x \le 0$; the simulation (sums of Exponentials) is the
generative story made literal, and its mean $k/\lambda = 15$ confirms the parametrisation.
""",
)

# ---------------------------------------------------------------- 8. CLT standardised means (core)
SOL_8 = r"""
def standardised_means(sampler, n, reps, mu, sigma, rng):
    x = np.asarray(sampler(rng, (reps, n)), dtype=float).mean(axis=1)
    return np.sqrt(n) * (x - mu) / sigma

def kolmogorov_distance(z):
    z = np.sort(np.asarray(z, dtype=float))
    m = len(z)
    F = stats.norm.cdf(z)
    return float(max(np.max(np.arange(1, m + 1) / m - F), np.max(F - np.arange(0, m) / m)))

def skewness(z):
    z = np.asarray(z, dtype=float)
    c = z - z.mean()
    return float(np.mean(c**3) / np.mean(c**2) ** 1.5)
"""
ns8 = run(SOL_8, r"""
r = np.random.default_rng(1)
z_exp30 = standardised_means(lambda rng, size: rng.exponential(1.0, size), 30, 20_000, 1.0, 1.0, r)
z_exp1 = standardised_means(lambda rng, size: rng.exponential(1.0, size), 1, 20_000, 1.0, 1.0, np.random.default_rng(2))
kd1 = kolmogorov_distance(z_exp1)
sk30 = skewness(z_exp30)
sk_small = skewness(np.array([1.0, 2.0, 3.0, 4.0, 10.0]))
""")
ex.add(
    title="Verify the central limit theorem (and watch it fail)", difficulty=2,
    prompt_md=r"""
Implement

* `standardised_means(sampler, n, reps, mu, sigma, rng)` → a NumPy array of `reps` values
  $\sqrt{n}\,(\bar X_n - \mu)/\sigma$, where `sampler(rng, size)` returns an array of the given shape (call it
  **once** with `size=(reps, n)` and average along axis 1);
* `kolmogorov_distance(z)` → $\sup_z |F_{\text{emp}}(z) - \Phi(z)|$ between the empirical CDF of the sample and the
  standard Normal CDF (sort the sample; the sup is attained just before or at a data point — compare both
  $\frac{i}{m} - \Phi(z_{(i)})$ and $\Phi(z_{(i)}) - \frac{i-1}{m}$);
* `skewness(z)` → the sample skewness $\frac{\frac1m\sum(z_i - \bar z)^3}{\big(\frac1m\sum(z_i - \bar z)^2\big)^{3/2}}$.

Then test the CLT: for an Exponential(1) source ($\mu = \sigma = 1$) with $n = 30$ and 20,000 repetitions the
standardised means should have mean $\approx 0$, variance $\approx 1$, Kolmogorov distance below 0.04 and skewness
near $2/\sqrt{30} = 0.365$. For a Cauchy source (use $\mu = 0$, $\sigma = 1$ as *nominal* values) the variance
of the "standardised means" should blow up and the Kolmogorov distance should stay large for every $n$.
""",
    starter=r"""
def standardised_means(sampler, n, reps, mu, sigma, rng):
    # YOUR CODE HERE
    return None

def kolmogorov_distance(z):
    # YOUR CODE HERE
    return None

def skewness(z):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_8,
    check=rf"""
expo = lambda rng, size: rng.exponential(1.0, size)
try:
    z = np.asarray(standardised_means(expo, 30, 20_000, 1.0, 1.0, np.random.default_rng(1)), dtype=float)
    ok_shape = z.shape == (20_000,)
except Exception:
    z, ok_shape = None, False
check('returns 20,000 standardised means', ok_shape)
check('mean of standardised means is within 0.03 of 0', ok_shape and abs(z.mean()) < 0.03)
check('variance of standardised means is within 0.05 of 1', ok_shape and abs(z.var() - 1) < 0.05)
try:
    kd = kolmogorov_distance(z)
except Exception:
    kd = None
check('Kolmogorov distance to N(0,1) below 0.04 for exponential means of 30', kd is not None and kd < 0.04)
try:
    kd_normal = kolmogorov_distance(stats.norm.ppf((np.arange(1, 1001) - 0.5) / 1000))
except Exception:
    kd_normal = None
check_close('Kolmogorov distance of perfect Normal quantiles is 0.0005', kd_normal, 0.0005, tol=1e-9)
try:
    kd1 = kolmogorov_distance(standardised_means(expo, 1, 20_000, 1.0, 1.0, np.random.default_rng(2)))
except Exception:
    kd1 = None
check('n = 1 (no averaging): distance is large, about 0.16 (within 0.02)', kd1 is not None and abs(kd1 - 0.1587) < 0.02)
try:
    sk = skewness(z)
except Exception:
    sk = None
check('skewness of the means is near 2/sqrt(30) = 0.365 (within 0.08)', sk is not None and abs(sk - 2/sqrt(30)) < 0.08)
try:
    got = skewness(np.array([1.0, 2.0, 3.0, 4.0, 10.0]))
except Exception:
    got = None
check_hash('skewness of [1, 2, 3, 4, 10]', got, '{sha_of(ns8["sk_small"], decimals=4)}', decimals=4)
try:
    cauchy = lambda rng, size: rng.standard_cauchy(size)
    zc = standardised_means(cauchy, 1000, 20_000, 0.0, 1.0, np.random.default_rng(3))
    ok_cauchy = np.var(zc) > 100 and kolmogorov_distance(zc) > 0.2
except Exception:
    ok_cauchy = False
check('Cauchy source: variance explodes and the distance to the Normal stays large', ok_cauchy)
""",
    explanation_md=r"""
Standardising by the *true* $\mu$ and $\sigma$ and multiplying by $\sqrt{n}$ removes the $1/\sqrt{n}$ shrinkage,
so what remains is pure shape: for the Exponential at $n = 30$ the result has mean 0, variance 1 and a residual
skewness of $\gamma_1/\sqrt{n} = 0.365$, which is the whole content of the Berry–Esseen story. The Kolmogorov
distance is the natural ruler — $0.16$ at $n = 1$ (an Exponential is not a Normal), about $0.025$ at $n = 30$,
heading to 0 like $1/\sqrt{n}$. For the Cauchy there is no $\mu$ and no $\sigma$; plugging in nominal values
produces "standardised means" whose variance is enormous and grows with `reps`, and whose distance to the
Normal never shrinks — the CLT's finite-variance condition is not a technicality. A common slip in the
Kolmogorov distance is to compare only $i/m$ with $\Phi(z_{(i)})$, missing the jump just below each data point.
""",
)

# ---------------------------------------------------------------- 9. conceptual: misuse of the LLN (core)
ex.add_conceptual(
    title="Diagnose the argument: 'the law of large numbers guarantees it'", difficulty=2,
    prompt_md=r"""
Three colleagues each invoke the law of large numbers. For each, say whether the conclusion follows, and if
not, name the error precisely and give the correct statement.

1. "Our roulette table has paid out more than expected this week. By the law of large numbers, next week it
   will pay out *less* than expected so that the long-run average is restored."
2. "Our fraud model's per-transaction loss has a heavy tail — a few transactions cost millions. But we
   average the loss over 10 million transactions a day, so by the law of large numbers the daily average is
   essentially constant and we can budget on it."
3. "We evaluate on 200 test examples. By the central limit theorem the accuracy estimate is Normal, so its
   95% interval is $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/200}$ — fine for our model with $\hat p = 0.995$."
""",
    answer_md=r"""
1. Gambler's fallacy. The LLN says the *proportion* $\bar X_n$ converges; it says nothing about the *sum*
   $S_n - n\mu$, which does not revert and in fact wanders further (its standard deviation grows like
   $\sqrt{n}$). Next week's spins are independent of this week's and have the same expectation; the excess is
   diluted by future spins, not repaid. Correct statement: the fraction of the long-run total that this week
   represents shrinks toward zero.
2. Two problems. First, the LLN needs a finite *mean*, and the rate of convergence needs a finite *variance*:
   if the loss tail is a power law with index $\alpha \le 2$, the daily average has infinite variance, is
   dominated by the single largest loss, and does not stabilise like $\sigma/\sqrt{n}$ (if $\alpha \le 1$ it does
   not converge at all). Second, the 10 million transactions are not independent (fraud rings, system outages
   produce clusters), so the effective $n$ is far smaller. Correct approach: model the tail explicitly
   (generalised Pareto), budget on a high quantile of the daily total, and reinsure or cap exposures.
3. The CLT is asymptotic and its speed depends on skewness: with $p = 0.995$ and $n = 200$ the expected number of
   errors is just 1, the Bernoulli skewness is $(1 - 2p)/\sqrt{p(1-p)} \approx -14$, and the sampling distribution
   of $\hat p$ is a few discrete spikes, not a bell. The Wald interval $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$
   is badly miscalibrated here (if 0 errors are observed it collapses to a point). Correct approach: use an exact
   Binomial (Clopper–Pearson) or Wilson interval, which for 1 error in 200 gives roughly $[0.97, 1.0]$ — far
   wider than the Wald interval suggests — and recognise that distinguishing 99.5% from 99% needs thousands of
   examples, not 200.
""",
)

# ---------------------------------------------------------------- 10. Monte Carlo integral with antithetic variates (stretch)
SOL_10 = r"""
def mc_integral(f, a, b, n, rng):
    u = rng.random(n)
    x = a + (b - a) * u
    vals = (b - a) * f(x)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n))

def mc_integral_antithetic(f, a, b, n, rng):
    u = rng.random(n // 2)
    x1 = a + (b - a) * u
    x2 = a + (b - a) * (1 - u)
    vals = 0.5 * (b - a) * (f(x1) + f(x2))
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))

def samples_needed(sigma, target_se):
    return int(np.ceil((sigma / target_se) ** 2))
"""
ns10 = run(SOL_10, r"""
f = lambda x: np.exp(-x**2)
est, se = mc_integral(f, 0.0, 1.0, 10_000, np.random.default_rng(7))
est_a, se_a = mc_integral_antithetic(f, 0.0, 1.0, 10_000, np.random.default_rng(7))
est2, se2 = mc_integral(lambda x: np.sin(x) / x, 1.0, 5.0, 20_000, np.random.default_rng(8))
nn = samples_needed(0.201, 0.0002)
""")
ex.add(
    title="Monte Carlo integration with an error bar, plus antithetic variates", difficulty=3,
    prompt_md=r"""
Implement

* `mc_integral(f, a, b, n, rng)` → the tuple `(estimate, standard_error)` for $\int_a^b f(x)\,dx$, using
  `n` uniform points `x = a + (b - a) * rng.random(n)`: the estimate is the mean of $(b-a)f(x_i)$ and the standard
  error is their sample standard deviation (`ddof=1`) divided by $\sqrt{n}$;
* `mc_integral_antithetic(f, a, b, n, rng)` → the same with **antithetic variates**: draw `u = rng.random(n // 2)`,
  evaluate $f$ at both $a + (b-a)u$ and $a + (b-a)(1-u)$, average each pair into one value
  $\frac{b-a}{2}[f(x_1) + f(x_2)]$, and return the mean and standard error (`ddof=1`, divided by $\sqrt{n/2}$)
  of those $n/2$ pair values;
* `samples_needed(sigma, target_se)` → the smallest integer $n$ with $\sigma/\sqrt{n} \le$ `target_se`.

Targets: $\int_0^1 e^{-x^2}\,dx = 0.746824$ and $\int_1^5 \frac{\sin x}{x}\,dx = 0.603848$. Your estimates must be
within 4 standard errors of the truth, and the antithetic standard error for $e^{-x^2}$ must be at least 3 times
smaller than the plain one (the guide predicts about 5 times).
""",
    starter=r"""
def mc_integral(f, a, b, n, rng):
    # YOUR CODE HERE
    return None

def mc_integral_antithetic(f, a, b, n, rng):
    # YOUR CODE HERE
    return None

def samples_needed(sigma, target_se):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_10,
    check=rf"""
I1, I2 = 0.7468241328124271, 0.6038481290459576
f1 = lambda x: np.exp(-x**2)
f2 = lambda x: np.sin(x) / x
try:
    est, se = mc_integral(f1, 0.0, 1.0, 10_000, np.random.default_rng(7))
except Exception:
    est, se = None, None
check('plain estimate of int_0^1 exp(-x^2) within 4 SE of 0.746824', est is not None and se is not None and abs(est - I1) < 4 * se)
check('its standard error is about 0.002 (within 0.0005)', se is not None and abs(se - 0.00201) < 0.0005)
check_hash('seeded plain estimate (rng seed 7, n = 10,000)', est, '{sha_of(ns10["est"], decimals=4)}', decimals=4)
try:
    est2, se2 = mc_integral(f2, 1.0, 5.0, 20_000, np.random.default_rng(8))
except Exception:
    est2, se2 = None, None
check('estimate of int_1^5 sin(x)/x within 4 SE of 0.603848', est2 is not None and se2 is not None and abs(est2 - I2) < 4 * se2)
check('the (b - a) factor is applied (estimate is not 0.15)', est2 is not None and est2 > 0.5)
try:
    est_a, se_a = mc_integral_antithetic(f1, 0.0, 1.0, 10_000, np.random.default_rng(7))
except Exception:
    est_a, se_a = None, None
check('antithetic estimate within 4 SE of the truth', est_a is not None and se_a is not None and abs(est_a - I1) < 4 * se_a)
check('antithetic SE at least 3x smaller than plain SE', se is not None and se_a is not None and se_a > 0 and se / se_a > 3)
check_hash('seeded antithetic estimate', est_a, '{sha_of(ns10["est_a"], decimals=4)}', decimals=4)
try:
    # equally correct formulas can differ by 1 in floating point here: compare to the nearest thousand
    got = round(int(samples_needed(0.201, 0.0002)), -3)
except Exception:
    got = None
check_hash('samples needed for SE 0.0002 with sigma 0.201 (to the nearest thousand)', got, '{sha_of(round(ns10["nn"], -3))}')
try:
    got = samples_needed(1.0, 0.1)
except Exception:
    got = None
check_close('samples_needed(1, 0.1) = 100', got, 100)
try:
    got = samples_needed(1.0, 0.3)
except Exception:
    got = None
check_close('samples_needed(1, 0.3) = 12 (11.1 rounds up, not down)', got, 12)
""",
    explanation_md=r"""
Writing $\int_a^b f = (b-a)\,\mathbb{E}[f(U)]$ with $U \sim \text{Uniform}(a, b)$ turns integration into averaging,
and the CLT supplies the error bar $\hat\sigma/\sqrt{n}$ — forgetting the factor $(b - a)$ is the classic
mistake (the second integral would come out near 0.15 instead of 0.60). For $e^{-x^2}$, $\sigma_g = 0.201$, so
$n = 10{,}000$ gives a standard error of $0.002$ and a target of $0.0002$ needs $(0.201/0.0002)^2 \approx 1{,}010{,}000$
samples. Antithetic variates exploit the monotonicity of $e^{-x^2}$ on $[0, 1]$: $g(U)$ and $g(1-U)$ are
negatively correlated (covariance $-0.0388$ against a variance of $0.0404$), so each pair average has variance
$\frac12(0.0404 - 0.0388) = 0.0008$, roughly 25 times smaller than two independent draws — the same accuracy
for 1/25 of the samples. The standard error of the antithetic estimator must be computed from the *pair
averages*, not from the $n$ individual evaluations, which are no longer independent.
""",
)

# ---------------------------------------------------------------- 11. mini-project: call centre (★★★)
PROJECT_CODE = r"""
rng = np.random.default_rng(2024)
RATE = 2.0            # calls per minute (Poisson arrivals)
MEAN_SERVICE = 4.0    # minutes (Exponential service)
HOURS = 8.0
WARMUP = 60.0         # minutes discarded so the system forgets its empty start
TARGET = 1 / 3        # 20 seconds

def simulate_day(c, rng, rate=RATE, mean_service=MEAN_SERVICE, hours=HOURS, warmup=WARMUP):
    T = hours * 60 + warmup
    gaps = rng.exponential(1 / rate, int(rate * T * 1.5) + 50)
    arrivals = np.cumsum(gaps)
    arrivals = arrivals[arrivals < T]
    service = rng.exponential(mean_service, arrivals.size)
    free_at = np.zeros(c)                     # when each agent becomes free
    waits = np.empty(arrivals.size)
    for i, (a, s) in enumerate(zip(arrivals, service)):
        j = np.argmin(free_at)                # the agent who is free earliest takes the call
        start = max(a, free_at[j])
        waits[i] = start - a
        free_at[j] = start + s
    return waits[arrivals >= warmup]

def erlang_c_service_level(lam, mu, c, t):
    # analytic M/M/c steady state: P(wait <= t)
    A = lam / mu
    if c <= A:
        return 0.0
    num = A**c / factorial(c) * c / (c - A)
    den = sum(A**k / factorial(k) for k in range(c)) + num
    p_wait = num / den
    return 1 - p_wait * exp(-(c * mu - lam) * t)

# sanity checks on the ingredients
test_gaps = rng.exponential(1 / RATE, 100_000)
print(f"arrival gaps: mean {test_gaps.mean():.3f} min (0.5), memoryless check P(gap > 1 | gap > 0.5) = {np.mean(test_gaps[test_gaps > 0.5] > 1.0):.3f} vs P(gap > 0.5) = {np.mean(test_gaps > 0.5):.3f}")
print(f"offered load = rate x mean service = {RATE * MEAN_SERVICE:.1f} agent-equivalents -> fewer than 9 agents cannot keep up\n")

results = {}
print(f"{'agents':>6s} {'occupancy':>9s} {'P(wait=0)':>9s} {'P(wait<=20s)':>13s} {'Erlang C':>9s} {'mean wait':>10s} {'90% wait':>9s} {'99% wait':>9s}")
for c in range(9, 16):
    w = np.concatenate([simulate_day(c, rng) for _ in range(60)])
    results[c] = w
    sl = np.mean(w <= TARGET)
    print(f"{c:6d} {RATE*MEAN_SERVICE/c:9.3f} {np.mean(w == 0):9.3f} {sl:13.3f} {erlang_c_service_level(RATE, 1/MEAN_SERVICE, c, TARGET):9.3f} "
          f"{w.mean():10.2f} {np.quantile(w, 0.9):9.2f} {np.quantile(w, 0.99):9.2f}")

staff = min(c for c, w in results.items() if np.mean(w <= TARGET) >= 0.9)
print(f"\nsmallest staffing with >= 90% of calls answered within 20 s: {staff} agents (occupancy {RATE*MEAN_SERVICE/staff:.0%})")

# waiting-time distribution for the chosen staffing and one agent fewer
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for c, col in [(staff - 1, "C3"), (staff, "C0"), (staff + 1, "C2")]:
    w = results[c]
    axes[0].hist(w[w > 0], bins=60, range=(0, 10), density=True, alpha=0.5, color=col, label=f"{c} agents (P(wait>0) = {np.mean(w > 0):.2f})")
    ws = np.sort(w)
    axes[1].plot(ws, np.arange(1, len(ws) + 1) / len(ws), color=col, label=f"{c} agents")
axes[0].set_xlabel("waiting time of calls that waited (minutes)"); axes[0].set_ylabel("density"); axes[0].set_title("Conditional waiting-time distribution"); axes[0].legend(fontsize=8)
axes[1].axhline(0.9, color="k", ls=":", lw=1); axes[1].axvline(TARGET, color="k", ls=":", lw=1)
axes[1].set_xlim(0, 5); axes[1].set_xlabel("waiting time t (minutes)"); axes[1].set_ylabel("P(wait <= t)"); axes[1].set_title("Service level: fraction answered within t"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# standard error of the service-level estimate for the chosen staffing: binomial-ish but calls within a day are correlated,
# so estimate it from day-to-day variation instead of p(1-p)/n
daily = np.array([np.mean(simulate_day(staff, rng) <= TARGET) for _ in range(60)])
print(f"service level at {staff} agents: {daily.mean():.3f} +/- {daily.std(ddof=1)/np.sqrt(len(daily)):.3f} (standard error over 60 simulated days)")
print(f"naive p(1-p)/n standard error would claim +/- {np.sqrt(daily.mean()*(1-daily.mean())/results[staff].size):.4f} -- too optimistic, because calls that queue together are correlated")

# sensitivity: 10% more traffic
w_busy = np.concatenate([simulate_day(staff, rng, rate=RATE * 1.1) for _ in range(60)])
print(f"\nwith 10% more calls and the same {staff} agents: service level {np.mean(w_busy <= TARGET):.3f}, mean wait {w_busy.mean():.2f} min")
"""
ex.add_project(
    title="Mini-project: staffing a call centre with Poisson arrivals and Exponential service",
    prompt_md=r"""
A call centre receives calls as a Poisson process at **2 calls per minute** during an 8-hour day; each call
occupies one agent for an **Exponential** time with mean **4 minutes**. Calls that find every agent busy wait in
a single first-come-first-served queue. Build a simulation and answer the manager's question: *how many
agents are needed so that at least 90% of calls are answered within 20 seconds?*

**The model** (use `rng = np.random.default_rng(2024)`):

1. Generate arrival times as the cumulative sum of Exponential(rate 2) gaps over the day (add a one-hour warm-up
   whose calls you discard, so the empty start of the day does not bias the results); generate an Exponential
   service time for each call.
2. Keep an array `free_at` with the time each of the $c$ agents becomes free. Process calls in arrival order:
   the earliest-free agent takes the call, it starts at `max(arrival, free_at[agent])`, the wait is the
   difference, and the agent is busy until `start + service`.
3. Simulate 60 days for each $c$ from 9 to 15 agents and pool the waiting times.

**Deliverables**

* a table over $c$ of the occupancy (offered load $2 \times 4 = 8$ divided by $c$), the fraction of calls that wait at
  all, the service level $P(\text{wait} \le 20\text{ s})$, and the mean, 90th and 99th percentile waiting times;
* the smallest $c$ meeting the 90% target, and a histogram of waiting times for that $c$ and its neighbours;
* a sanity check that the arrival gaps are memoryless, and a comparison of your simulated service level with the
  analytic **Erlang C** formula for the $M/M/c$ queue:
  $P(\text{wait} > 0) = \frac{A^c/c! \cdot c/(c-A)}{\sum_{k<c} A^k/k! + A^c/c! \cdot c/(c-A)}$ with $A = \lambda/\mu = 8$,
  and $P(\text{wait} > t) = P(\text{wait} > 0)\,e^{-(c\mu - \lambda)t}$;
* a standard error for the service-level estimate computed from the **day-to-day** variation of 60 daily service
  levels, compared with the naive $\sqrt{p(1-p)/n}$ over all pooled calls — explain why they differ;
* a sensitivity check: the service level at your chosen $c$ if the call rate rises by 10%.
""",
    rubric_md=r"""
* Arrivals are generated as a Poisson process (Exponential gaps, cumulative sum) and service times as Exponentials,
  all from one seeded generator; a warm-up period is discarded.
* The queue is simulated correctly: the earliest-free agent serves the next call, waits are `start - arrival`, and
  a call that finds a free agent has wait 0.
* The table reports occupancy, $P(\text{wait} > 0)$, the service level and waiting-time quantiles for each staffing
  level, and the smallest $c$ meeting the 90%-within-20-seconds target is identified (12 agents in simulation, service
  level about 0.905; Erlang C gives 0.8998, a hair under the target, so 13 with that reasoning is also correct).
* The simulated service levels agree with the Erlang C formula to within about 0.02 for every $c$ (a few standard
  errors of the 60-day estimate).
* The histogram has a title, axis labels and a legend, and shows how one agent more or fewer changes the
  waiting-time distribution.
* The standard error is computed from independent simulated days, and the answer explains that pooled calls are
  positively correlated (a busy spell delays many calls at once), so $\sqrt{p(1-p)/n}$ over calls understates the
  uncertainty.
""",
    solution_sketch_md=r"""
The offered load is $\lambda \times \mathbb{E}[\text{service}] = 2 \times 4 = 8$ agent-equivalents, so at least 9 agents
are needed for the queue to be stable at all, and at 9 agents (89% occupancy) only about 40% of calls are answered
within 20 seconds. Each additional agent helps enormously at first and then less: 10 agents give about 65%,
11 about 81%, **12 agents about 90%** (occupancy 67%; Erlang C gives 0.8998, so 12 sits right at the target and clears it only
within simulation noise), 13 about 95%. The simulation matches the Erlang C
formula to within simulation noise, which confirms both the queue logic and the memorylessness of the inputs
(Erlang C assumes exactly this $M/M/c$ model). The waiting-time histograms show that most calls at 12 agents do
not wait at all and the conditional wait of those that do is roughly Exponential with mean $1/(c\mu - \lambda)$
$= 1$ minute. The service-level standard error from 60 independent days is a few tenths of a percent, several
times larger than the naive binomial figure, because calls arriving during the same congested spell share
their fate. A 10% rise in traffic (offered load 8.8) drops the 12-agent service level to around 80% — the
system is non-linear near full utilisation, which is why call centres plan for peaks, not averages.
""",
    solution_code=PROJECT_CODE,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print(f"exercises: {len(ex.items)} items — {kinds.count('code')} coding, {kinds.count('concept')} conceptual, {kinds.count('project')} project")
print("difficulties:", [it["difficulty"] for it in ex.items])
