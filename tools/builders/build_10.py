"""
build_10.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for

    part-4-statistics/10-estimation-and-sampling

Run from the repository root:

    python3 _authoring/builders/build_10.py
    python3 tools/run_notebooks.py part-4-statistics/10-estimation-and-sampling

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

MOD = ROOT / "part-4-statistics/10-estimation-and-sampling"

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 10 — Estimation and Sampling")

nb.md(r"""
This notebook is the runnable companion to the concept guide: read [README.md](README.md) first,
or keep it open alongside. Every section below mirrors a numbered section of the guide. We write out
the sampling distribution of a mean by enumerating every possible sample, watch a biased survey stay
biased no matter how many people we mail, compute robust and non-robust summaries of a latency sample,
compare the mean and the median as estimators, measure bias, variance and MSE by simulation, see the
standard error shrink like $1/\sqrt{n}$, run a coverage experiment for confidence intervals, expose the
failure of the Wald interval, build the bootstrap from scratch (and break it), plan sample sizes, and
finally put honest error bars on a classifier's test accuracy.

Everything is plain NumPy first; library calls (`scipy.stats`, `sklearn`) appear only afterwards, to
confirm the hand computation.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
""")

# ---------------------------------------------------------------- 1. population vs sample
nb.md(r"""
## 1. Population, sample, and the sampling distribution written out in full

A population of five warehouse workers packed $\{2, 4, 4, 6, 9\}$ orders in an hour. We draw a simple
random sample of $n = 2$ *without replacement*. There are only $\binom{5}{2} = 10$ possible samples, so we
can list every one, compute its mean, and look at the **sampling distribution** of $\bar{X}_2$ directly.
The guide predicts: the mean of the ten sample means is $\mu$, and their variance is
$\frac{\sigma^2}{n}\cdot\frac{N-n}{N-1}$ (the finite population correction).
""")

nb.code(r"""
from itertools import combinations

population = np.array([2, 4, 4, 6, 9], dtype=float)
N, n = len(population), 2
mu, sigma2 = population.mean(), population.var()           # population mean and variance (divisor N)

samples = list(combinations(range(N), n))
means = np.array([population[list(s)].mean() for s in samples])
print(f"population {population}:  mu = {mu},  sigma^2 = {sigma2:.2f}")
print(f"{len(samples)} equally likely samples of size {n}, their means: {means}")
print(f"mean of the sample means     = {means.mean():.3f}   (= mu: the sample mean is unbiased)")
fpc = (N - n) / (N - 1)
print(f"variance of the sample means = {means.var():.3f}   formula sigma^2/n * (N-n)/(N-1) = {sigma2/n:.2f} * {fpc:.2f} = {sigma2/n*fpc:.3f}")

# sampling WITH replacement: 25 ordered samples, independent draws, no correction
with_repl = np.array([population[[i, j]].mean() for i in range(N) for j in range(N)])
print(f"with replacement:            = {with_repl.var():.3f}   = sigma^2/n = {sigma2/n:.3f}")
""")

nb.md(r"""
The ten means **are** the sampling distribution: a histogram of them is what you would get by repeating
the sampling forever. Sampling without replacement is slightly *more* precise than with replacement
(variance 2.1 versus 2.8), because a sample cannot accidentally pick the same worker twice.

### A biased sample does not get better with size

An app has 70% casual users (about 2 hours/month) and 30% engaged users (about 20 hours/month), so the
true mean usage is $0.7 \times 2 + 0.3 \times 20 = 7.4$ hours. We email a survey; casual users respond
with probability 0.10, engaged users with 0.40. We compare the respondents' mean with the mean of a
*random* sample of the same size from the same mailing, for mailings from 200 to 200,000 users, each
repeated 30 times.
""")

nb.code(r"""
def mail_survey(n_mailed, rng, p_engaged=0.3, hours=(2.0, 20.0), response=(0.10, 0.40)):
    engaged = rng.random(n_mailed) < p_engaged
    usage = np.where(engaged, hours[1], hours[0]) + rng.normal(0, 0.5, n_mailed)
    responds = rng.random(n_mailed) < np.where(engaged, response[1], response[0])
    k = responds.sum()                                      # number of respondents
    random_subset = rng.choice(n_mailed, size=k, replace=False)   # a random sample of the same size
    return usage[responds].mean(), usage[random_subset].mean()

true_mean = 0.7 * 2.0 + 0.3 * 20.0
expected_biased = (0.7 * 0.10 * 2.0 + 0.3 * 0.40 * 20.0) / (0.7 * 0.10 + 0.3 * 0.40)
sizes = np.array([200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000])
reps = 30
biased = np.array([[mail_survey(m, rng)[0] for _ in range(reps)] for m in sizes])
random_ = np.array([[mail_survey(m, rng)[1] for _ in range(reps)] for m in sizes])

plt.errorbar(sizes, biased.mean(1), yerr=biased.std(1), fmt="o-", capsize=3, label="respondents (self-selected)")
plt.errorbar(sizes, random_.mean(1), yerr=random_.std(1), fmt="s-", capsize=3, label="random sample of the same size")
plt.axhline(true_mean, color="k", ls="--", label=f"true mean {true_mean} h")
plt.axhline(expected_biased, color="C0", ls=":", label=f"expected respondent mean {expected_biased:.1f} h")
plt.xscale("log"); plt.xlabel("number of users mailed (log scale)"); plt.ylabel("estimated mean usage (hours/month)")
plt.title("Sampling noise shrinks with n; selection bias does not"); plt.legend(fontsize=8); plt.show()

for m, b, r in zip(sizes[[0, 4, 9]], biased[[0, 4, 9]], random_[[0, 4, 9]]):
    print(f"mailed {m:>7d}: respondents {b.mean():6.2f} +/- {b.std():.2f}    random sample {r.mean():5.2f} +/- {r.std():.2f}    truth {true_mean}")
""")

nb.md(r"""
The error bars (one standard deviation over 30 repetitions) collapse as the mailing grows — that is the
$1/\sqrt{n}$ law. The random sample converges to 7.4; the respondents converge to 13.4, which is
exactly the $\frac{0.07 \times 2 + 0.12 \times 20}{0.19}$ the guide computes. Eighty percent too high,
with a tiny error bar: *precisely wrong*. A sample that is a thousand times larger buys nothing.

**Try it yourself:** set both response probabilities to 0.25 and re-run. Then make the casual users the
more likely responders. What determines the sign of the bias?
""")

# ---------------------------------------------------------------- 2. descriptive statistics
nb.md(r"""
## 2. Descriptive statistics, robust and otherwise

Ten API response times in milliseconds, nine ordinary and one that hit a cold cache. We compute every
summary from Section 2 of the guide from scratch — mean, median, trimmed mean, both variances, quartiles,
IQR and MAD — and then check against NumPy and SciPy.
""")

nb.code(r"""
times = np.array([120, 135, 140, 150, 155, 160, 170, 180, 210, 2400], dtype=float)

def my_median(v):
    s = np.sort(v); n = len(s); m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])

def trimmed_mean(v, alpha):
    s = np.sort(v); k = int(np.floor(alpha * len(s)))          # drop k smallest and k largest
    return s[k:len(s) - k].mean()

def variance(v, ddof=0):
    return ((v - v.mean()) ** 2).sum() / (len(v) - ddof)

def quantile(v, q):                                            # linear interpolation, NumPy's default
    s = np.sort(v); pos = q * (len(s) - 1); lo = int(np.floor(pos)); hi = int(np.ceil(pos))
    return s[lo] + (pos - lo) * (s[hi] - s[lo])

def mad(v):
    return my_median(np.abs(v - my_median(v)))

rows = [
    ("mean",                times.mean(),                np.mean(times)),
    ("median",              my_median(times),            np.median(times)),
    ("10% trimmed mean",    trimmed_mean(times, 0.10),   stats.trim_mean(times, 0.10)),
    ("variance, divisor n", variance(times, 0),          np.var(times, ddof=0)),
    ("variance, divisor n-1", variance(times, 1),        np.var(times, ddof=1)),
    ("sd s (ddof=1)",       np.sqrt(variance(times, 1)), np.std(times, ddof=1)),
    ("Q1 (25%)",            quantile(times, 0.25),       np.quantile(times, 0.25)),
    ("Q3 (75%)",            quantile(times, 0.75),       np.quantile(times, 0.75)),
    ("IQR",                 quantile(times, 0.75) - quantile(times, 0.25), stats.iqr(times)),
    ("MAD",                 mad(times),                  stats.median_abs_deviation(times)),
    ("1.4826 * MAD",        1.4826 * mad(times),         stats.median_abs_deviation(times, scale="normal")),
]
print(f"{'statistic':24s}{'from scratch':>14s}{'library':>14s}")
for name, mine, lib in rows:
    print(f"{name:24s}{mine:14.3f}{lib:14.3f}")
clean = times[:-1]
print(f"\nwithout the outlier: mean {clean.mean():.1f}, sd {clean.std(ddof=1):.1f}  ->  the median, trimmed mean and scaled MAD were close; mean and sd were not")
""")

nb.md(r"""
### Shape: histogram, kernel density estimate and empirical CDF

For shape we need more than ten points. We generate 200 latencies from a log-normal distribution (a
realistic right-skewed model: median 150 ms, $\sigma = 0.6$ on the log scale) with its own seeded
generator so the same sample can be reused in the bootstrap section. The KDE is
$\hat f(x) = \frac{1}{nh}\sum_i K\!\big(\frac{x - x_i}{h}\big)$ with a Gaussian kernel and Silverman's
bandwidth $h = 0.9\min(s, \text{IQR}/1.34)\,n^{-1/5}$; the ECDF is $\hat F_n(x) = \frac1n\sum_i \mathbf 1[x_i \le x]$.
""")

nb.code(r"""
lat_rng = np.random.default_rng(7)
latency = lat_rng.lognormal(mean=np.log(150), sigma=0.6, size=200)
n_lat = len(latency)

def kde(data, grid):
    s = data.std(ddof=1); iqr_ = np.subtract(*np.quantile(data, [0.75, 0.25]))
    h = 0.9 * min(s, iqr_ / 1.34) * len(data) ** (-1 / 5)                 # Silverman's rule of thumb
    return np.exp(-0.5 * ((grid[:, None] - data[None, :]) / h) ** 2).sum(1) / (len(data) * h * np.sqrt(2 * np.pi))

grid = np.linspace(0, 600, 400)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
ax1.hist(latency, bins=30, density=True, alpha=0.5, label="histogram (density=True)")
ax1.plot(grid, kde(latency, grid), "C3", label="KDE from scratch")
ax1.plot(grid, stats.gaussian_kde(latency)(grid), "k:", label="scipy gaussian_kde")
for q, c in [(0.5, "C2"), (0.95, "C1")]:
    ax1.axvline(np.quantile(latency, q), color=c, ls="--", label=f"{int(q*100)}th percentile = {np.quantile(latency, q):.0f} ms")
ax1.set_xlabel("latency (ms)"); ax1.set_ylabel("density"); ax1.set_title("200 latencies: histogram and KDE"); ax1.legend(fontsize=7)

xs = np.sort(latency); ecdf = np.arange(1, n_lat + 1) / n_lat
ax2.step(xs, ecdf, where="post", label="ECDF")
ax2.plot(grid, stats.lognorm.cdf(grid, s=0.6, scale=150), "k--", label="true CDF")
ax2.axhline(0.5, color="C2", ls=":"); ax2.axhline(0.95, color="C1", ls=":")
ax2.set_xlabel("latency (ms)"); ax2.set_ylabel("F(x)"); ax2.set_title("Empirical CDF vs the truth"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"mean {latency.mean():.1f}   median {np.median(latency):.1f}   p95 {np.quantile(latency, 0.95):.1f}   sd {latency.std(ddof=1):.1f}   (true median 150, true mean {150*np.exp(0.18):.1f})")
""")

nb.code(r"""
# breakdown points: move k of the nine clean response times to a huge value
for k in range(0, 6):
    v = clean.copy(); v[:k] = 1e6
    print(f"{k} of 9 corrupted:  mean {v.mean():>10.1f}   sd {v.std(ddof=1):>10.1f}   median {my_median(v):>9.1f}   MAD {mad(v):>7.1f}   10%-trimmed {trimmed_mean(v, 0.1):>10.1f}   20%-trimmed {trimmed_mean(v, 0.2):>10.1f}")
""")

nb.md(r"""
One corrupted value wrecks the mean and the standard deviation (breakdown point 0). The 10%-trimmed mean
trims $\lfloor 0.1 \times 9 \rfloor = 0$ values per side, so at $n = 9$ it *is* the mean and fails just as
fast; the 20%-trimmed mean trims one value per side and survives exactly one corrupted point (breakdown
point $\alpha$). The median and the MAD survive four corrupted values out of nine and break at five —
their 50% breakdown point. The ECDF needs no bins or bandwidth and converges uniformly to the true CDF;
the KDE depends on $h$ (set it to 5 or 60 in `kde` to see too spiky and too smooth), the histogram on
the bin edges.
""")

# ---------------------------------------------------------------- 3. estimators
nb.md(r"""
## 3. Estimators are random variables: mean vs median, and three variance estimators

For $n = 25$ observations we compare the sample mean and the sample median as estimators of the centre,
for a Normal and for a (heavy-tailed) Laplace population, by drawing 20,000 samples of each. The guide
predicts $\operatorname{Var}(\text{median}) / \operatorname{Var}(\bar X) \approx \pi/2 \approx 1.57$ for
normal data and $\approx 1/2$ for Laplace data.
""")

nb.code(r"""
n, reps = 25, 20_000
populations = {"Normal(0, 1)": (rng.normal(0, 1, size=(reps, n)), 1.0),
               "Laplace(0, 1)": (rng.laplace(0, 1, size=(reps, n)), 2.0)}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (name, (X, var_pop)) in zip(axes, populations.items()):
    xbar, med = X.mean(1), np.median(X, 1)
    ax.hist(xbar, bins=60, alpha=0.6, density=True, label=f"sample mean, sd {xbar.std():.3f}")
    ax.hist(med, bins=60, alpha=0.6, density=True, label=f"sample median, sd {med.std():.3f}")
    ax.set_title(f"sampling distributions, {name}, n = {n}"); ax.set_xlabel("estimate of the centre"); ax.legend(fontsize=8)
    print(f"{name:14s} Var(mean) {xbar.var():.4f} (theory sigma^2/n = {var_pop/n:.4f})   Var(median) {med.var():.4f}   "
          f"ratio median/mean = {med.var()/xbar.var():.2f}   bias of mean {xbar.mean():+.4f}, of median {med.mean():+.4f}")
axes[0].set_ylabel("density"); plt.tight_layout(); plt.show()
print(f"theory: pi/2 = {np.pi/2:.2f} for normal data; the median is about twice as efficient for Laplace data")
""")

nb.md(r"""
Both estimators are unbiased in both cases (the histograms are centred on 0). For normal data the median
is wider — you would need about 57% more observations to match the mean. For Laplace data the picture
flips: the median is the tighter one. *Which estimator is better depends on the population's tails.*

### Bias, variance and MSE of three variance estimators

$\hat\sigma^2_c = c\sum_i (X_i - \bar X)^2$ with $c = \frac{1}{n-1}, \frac1n, \frac1{n+1}$ for $n = 10$
normal observations with $\sigma^2 = 1$. Since $\sum_i (X_i - \bar X)^2 \sim \chi^2_{9}$ has mean 9 and
variance 18, the theory is $\text{bias} = 9c - 1$ and $\text{variance} = 18c^2$.
""")

nb.code(r"""
n, reps = 10, 50_000
X = rng.normal(0, 1, size=(reps, n))
ss = ((X - X.mean(1, keepdims=True)) ** 2).sum(1)              # sum of squared deviations, one per sample
print(f"{'divisor':8s}{'bias':>9s}{'theory':>9s}{'variance':>10s}{'theory':>9s}{'MSE':>9s}{'bias^2+var':>12s}")
for name, c in [("n-1", 1 / (n - 1)), ("n", 1 / n), ("n+1", 1 / (n + 1))]:
    est = c * ss
    bias, var, mse = est.mean() - 1.0, est.var(), ((est - 1.0) ** 2).mean()
    print(f"{name:8s}{bias:9.4f}{(n-1)*c - 1:9.4f}{var:10.4f}{2*(n-1)*c**2:9.4f}{mse:9.4f}{bias**2 + var:12.4f}")
print("\nunbiased (n-1) has the largest MSE; shrinking towards 0 (n+1) trades a little bias for less variance")
""")

nb.md(r"""
### Shrinkage in many dimensions: the James–Stein estimator

Observe one noisy vector $X \sim \mathcal N(\theta, I_p)$ with $p = 10$. The obvious estimate is
$\hat\theta = X$ (total MSE $= p = 10$). James–Stein shrinks it toward the origin,
$\hat\theta_{JS} = \big(1 - \frac{p-2}{\|X\|^2}\big)X$, and the guide claims it wins for *every* $\theta$.
We check it along a range of $\|\theta\|$.
""")

nb.code(r"""
p, reps = 10, 20_000
norms = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
mse_mle, mse_js, mse_js_plus = [], [], []
for r in norms:
    theta = np.zeros(p); theta[0] = r                           # only ||theta|| matters, by symmetry
    X = theta + rng.normal(size=(reps, p))
    shrink = 1 - (p - 2) / (X ** 2).sum(1, keepdims=True)
    js = shrink * X
    js_plus = np.maximum(shrink, 0) * X                         # positive-part James-Stein: never flip the sign
    mse_mle.append(((X - theta) ** 2).sum(1).mean())
    mse_js.append(((js - theta) ** 2).sum(1).mean())
    mse_js_plus.append(((js_plus - theta) ** 2).sum(1).mean())

plt.plot(norms, mse_mle, "o-", label=r"$\hat\theta = X$ (unbiased), MSE = p")
plt.plot(norms, mse_js, "s-", label="James-Stein")
plt.plot(norms, mse_js_plus, "^--", label="positive-part James-Stein")
plt.xlabel(r"$\|\theta\|$ (distance of the truth from the shrinkage target)"); plt.ylabel("total MSE over the 10 coordinates")
plt.title("A biased estimator that beats the unbiased one everywhere (p = 10)"); plt.legend(); plt.show()
for r, a, b in zip(norms, mse_mle, mse_js):
    print(f"||theta|| = {r:4.1f}:  MSE(X) = {a:6.2f}   MSE(JS) = {b:5.2f}")
""")

nb.md(r"""
At $\theta = 0$ the shrinkage estimator has MSE about 2 instead of 10; far from the origin the gain fades
but never becomes a loss. This is the mathematical ancestor of ridge regression and weight decay
([Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md)): accept a
little bias, remove a lot of variance.

**Try it yourself:** set `p = 2` (where the theorem no longer applies) — does James–Stein still win?
""")

# ---------------------------------------------------------------- 4. standard error
nb.md(r"""
## 4. Sampling distributions and the standard error

The sample mean of $n$ i.i.d. observations has standard deviation — the **standard error** —
$\sigma/\sqrt n$; a sample proportion has $\sqrt{p(1-p)/n}$. We draw from a very skewed population,
$\text{Exp}(1)$, to see two things at once: the sampling distribution becomes normal as $n$ grows (the
CLT), and its width follows the formula exactly.
""")

nb.code(r"""
ns = np.array([2, 5, 10, 30, 100, 300, 1_000])
reps, p_true = 5_000, 0.3
sd_mean = np.array([rng.exponential(1.0, size=(reps, m)).mean(1).std() for m in ns])
sd_prop = np.array([(rng.random(size=(reps, m)) < p_true).mean(1).std() for m in ns])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
grid = np.linspace(0, 3, 300)
for m, c in [(2, "C0"), (30, "C1")]:
    xb = rng.exponential(1.0, size=(reps, m)).mean(1)
    ax1.hist(xb, bins=60, density=True, alpha=0.5, color=c, label=f"mean of n = {m} Exp(1) draws")
    ax1.plot(grid, stats.norm.pdf(grid, 1, 1 / np.sqrt(m)), color=c, lw=1.5)
ax1.set_xlabel("sample mean"); ax1.set_ylabel("density"); ax1.set_title("CLT: the mean of a skewed population"); ax1.legend(fontsize=8)
ax2.loglog(ns, sd_mean, "o", label="observed sd of the mean (Exp(1))")
ax2.loglog(ns, 1 / np.sqrt(ns), "C0-", label=r"$\sigma/\sqrt{n}$, $\sigma = 1$")
ax2.loglog(ns, sd_prop, "s", color="C1", label=f"observed sd of a proportion (p = {p_true})")
ax2.loglog(ns, np.sqrt(p_true * (1 - p_true) / ns), "C1-", label=r"$\sqrt{p(1-p)/n}$")
ax2.set_xlabel("sample size n"); ax2.set_ylabel("standard error"); ax2.set_title("Standard error falls like 1/sqrt(n)"); ax2.legend(fontsize=7)
plt.tight_layout(); plt.show()
for m, a, b in zip(ns, sd_mean, sd_prop):
    print(f"n = {m:>5d}: sd(mean) {a:.4f} vs {1/np.sqrt(m):.4f}     sd(proportion) {b:.4f} vs {np.sqrt(p_true*(1-p_true)/m):.4f}")
""")

nb.code(r"""
# the poll and the filling machine from the guide, plus scipy's standard error of the mean
p_hat, n_poll = 0.52, 1_000
se_p = np.sqrt(p_hat * (1 - p_hat) / n_poll)
print(f"poll: p_hat = {p_hat}, n = {n_poll}:  SE = {se_p:.4f},  95% margin = {1.96*se_p:.3f}   (the famous +/- 3 points)")
print(f"      the LEAD 2p-1 = {2*p_hat-1:.2f} has SE {2*se_p:.4f} and margin +/- {2*1.96*se_p:.3f}  -> a 52-48 lead is ~1.3 SE from a tie")

sigma_fill, target = 4.0, 500.0
for n_pack, xbar_ in [(16, 497.5), (4, 497.5)]:
    se = sigma_fill / np.sqrt(n_pack); zscore = (xbar_ - target) / se
    print(f"filling machine: n = {n_pack:2d}, xbar = {xbar_}: SE = {se:.2f} g, z = {zscore:+.2f},  P(|Xbar - 500| >= {target-xbar_}) = {2*stats.norm.sf(abs(zscore)):.4f}")

print(f"\nscipy.stats.sem(latency) = {stats.sem(latency):.3f}   by hand s/sqrt(n) = {latency.std(ddof=1)/np.sqrt(n_lat):.3f}")
print(f"sd of the 200 latencies = {latency.std(ddof=1):.1f} ms (spread of individual requests; does not shrink with n)")
""")

nb.md(r"""
Even for $n = 2$ draws from an exponential the formula $\sigma/\sqrt n$ is exact — it is only the *shape*
that needs the CLT, and by $n = 30$ the histogram is close to the normal curve. In the right panel the
points sit on the theory lines across three decades of $n$: a straight line of slope $-1/2$ on log–log
axes. Keep the two numbers at the end apart: the standard deviation (87 ms) describes requests, the
standard error (6 ms) describes our knowledge of the mean.
""")

# ---------------------------------------------------------------- 5. confidence intervals
nb.md(r"""
## 5. Confidence intervals: a coverage experiment

A 95% interval is a *procedure* that captures the parameter in 95% of repeated samples. We test three
procedures on 10,000 samples of $n = 5$ from $\mathcal N(10, 2^2)$: the $z$-interval with $\sigma$ known,
the $t$-interval with $s$ estimated, and the common mistake — a $z$-interval that plugs in $s$. The guide
predicts the last one covers only $2\,F_{t_4}(1.96) - 1 \approx 88\%$ at $n = 5$.
""")

nb.code(r"""
mu, sigma, n, reps = 10.0, 2.0, 5, 10_000
X = rng.normal(mu, sigma, size=(reps, n))
xbar, s = X.mean(1), X.std(1, ddof=1)
z = stats.norm.ppf(0.975); t = stats.t.ppf(0.975, df=n - 1)
intervals = {
    "z, sigma known":        (xbar - z * sigma / np.sqrt(n), xbar + z * sigma / np.sqrt(n)),
    "t, s estimated":        (xbar - t * s / np.sqrt(n),     xbar + t * s / np.sqrt(n)),
    "z with s  (too narrow)": (xbar - z * s / np.sqrt(n),     xbar + z * s / np.sqrt(n)),
}
print(f"n = {n}: z_0.975 = {z:.3f}, t_(4, 0.975) = {t:.3f}")
for name, (lo, hi) in intervals.items():
    covered = (lo <= mu) & (mu <= hi)
    print(f"{name:24s} coverage {covered.mean():.4f}   mean width {np.mean(hi - lo):.3f}")
print(f"theory for 'z with s' at n = 5: 2 F_t4(1.96) - 1 = {2*stats.t.cdf(z, df=n-1) - 1:.4f}")

lo, hi = intervals["t, s estimated"]; k = 100
covered = (lo[:k] <= mu) & (mu <= hi[:k])
plt.figure(figsize=(9, 3.6))
plt.vlines(np.arange(k), lo[:k], hi[:k], colors=np.where(covered, "C0", "C3"), lw=1.5)
plt.plot(np.arange(k), xbar[:k], "k.", ms=3)
plt.axhline(mu, color="k", ls="--", label=f"true mu = {mu}")
plt.xlabel("sample number"); plt.ylabel("95% t-interval for mu")
plt.title(f"100 of the {reps} intervals: {int((~covered).sum())} miss the truth (red); expected about 5")
plt.legend(); plt.show()
""")

nb.md(r"""
The $z$-interval with known $\sigma$ and the $t$-interval both hit 95% (within Monte Carlo error); the
$z$-with-$s$ interval is about 30% narrower than the $t$-interval and misses the truth about 12–13% of
the time. Each red segment is an interval that was computed correctly and is nevertheless wrong — the
95% belongs to the procedure, not to any one of them.

**Try it yourself:** change `n` to 30. The three procedures should become nearly indistinguishable — why?

### Wald vs Wilson for a proportion: exact coverage

For a proportion we do not need to simulate: $\hat p = k/n$ takes only $n + 1$ values, so the coverage
of any interval at a given $p$ is $\sum_{k} P(K = k)\,\mathbf 1[\text{interval}(k, n) \ni p]$. We compute
it for $p = 0.05$ (rare events) and $p = 0.5$ as a function of $n$.
""")

nb.code(r"""
def wald_ci(k, n, conf=0.95):
    z = stats.norm.ppf(0.5 + conf / 2); p = k / n
    h = z * np.sqrt(p * (1 - p) / n)
    return p - h, p + h

def wilson_ci(k, n, conf=0.95):
    z = stats.norm.ppf(0.5 + conf / 2); p = k / n; z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z / (1 + z2 / n) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return centre - half, centre + half

def exact_coverage(ci_fn, n, p, conf=0.95):
    ks = np.arange(n + 1); pmf = stats.binom.pmf(ks, n, p)
    lo, hi = ci_fn(ks, n, conf)
    return pmf[(lo <= p) & (p <= hi)].sum()

ns = np.arange(10, 1001, 5)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, p in zip(axes, (0.05, 0.5)):
    ax.plot(ns, [exact_coverage(wald_ci, m, p) for m in ns], label="Wald", lw=1)
    ax.plot(ns, [exact_coverage(wilson_ci, m, p) for m in ns], label="Wilson", lw=1)
    ax.axhline(0.95, color="k", ls="--", lw=1); ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("n"); ax.set_title(f"exact coverage of nominal 95% intervals, p = {p}"); ax.legend()
axes[0].set_ylabel("coverage probability"); plt.tight_layout(); plt.show()

for m in (20, 50, 100, 200, 500, 1000):
    print(f"p = 0.05, n = {m:>4d}: Wald {exact_coverage(wald_ci, m, 0.05):.3f}   Wilson {exact_coverage(wilson_ci, m, 0.05):.3f}")

print()
for k, m in [(9, 200), (0, 20), (2, 150)]:
    sp = stats.binomtest(k, m).proportion_ci(confidence_level=0.95, method="wilson")
    print(f"k = {k:>2d}/{m:<4d} Wald [{wald_ci(k, m)[0]:+.4f}, {wald_ci(k, m)[1]:.4f}]   Wilson [{wilson_ci(k, m)[0]:.4f}, {wilson_ci(k, m)[1]:.4f}]"
          f"   scipy wilson [{sp.low:.4f}, {sp.high:.4f}]")
print(f"rule of three for 0/20: 3/n = {3/20:.3f}   Wilson upper limit z^2/(n+z^2) = {z*z/(20+z*z):.3f}")
""")

nb.md(r"""
For $p = 0.05$ the Wald interval's coverage oscillates far below 95% — at $n = 20$ it is 64%, and it
does not settle near the nominal level until $n$ is around 1,000 — while Wilson stays close to 95%
throughout. At $p = 0.5$ the two are nearly identical and both fine: the Wald interval is not wrong,
it is wrong *near the boundaries*. With $k = 0$ it has zero width, and for $2/150$ its lower limit is
negative; Wilson's never leaves $[0, 1]$. The hand-written Wilson formula matches `scipy.stats.binomtest`.
""")

# ---------------------------------------------------------------- 6. bootstrap
nb.md(r"""
## 6. The bootstrap, from scratch

There is no standard-error formula for a median or a 95th percentile that works well at $n = 200$. So we
resample: draw 200 values *with replacement* from the 200 latencies of Section 2, compute the statistic,
repeat $B = 10{,}000$ times. The spread of the replicates estimates the standard error, and their 2.5%
and 97.5% quantiles form the **percentile interval**. We also bootstrap the mean, where we *can* check
the answer against $s/\sqrt n$ and the $t$-interval.
""")

nb.code(r"""
boot_rng = np.random.default_rng(0)
B = 10_000
idx = boot_rng.integers(0, n_lat, size=(B, n_lat))             # B bootstrap samples, as index arrays
resamples = latency[idx]                                        # shape (B, n): each row is one resample

def percentile_ci(replicates, conf=0.95):
    return np.quantile(replicates, [(1 - conf) / 2, (1 + conf) / 2])

statistics = {"mean": (resamples.mean(1), latency.mean()),
              "median": (np.median(resamples, axis=1), np.median(latency)),
              "95th percentile": (np.quantile(resamples, 0.95, axis=1), np.quantile(latency, 0.95))}
for name, (reps_, point) in statistics.items():
    lo, hi = percentile_ci(reps_)
    print(f"{name:16s} estimate {point:6.1f} ms   bootstrap SE {reps_.std(ddof=1):5.2f}   95% percentile CI [{lo:6.1f}, {hi:6.1f}]")
t199 = stats.t.ppf(0.975, df=n_lat - 1); se_formula = latency.std(ddof=1) / np.sqrt(n_lat)
print(f"\ncheck on the mean: formula SE s/sqrt(n) = {se_formula:.2f},  t-interval [{latency.mean() - t199*se_formula:.1f}, {latency.mean() + t199*se_formula:.1f}]")
print(f"each resample leaves out on average {np.mean([len(np.setdiff1d(np.arange(n_lat), row)) for row in idx[:500]]) / n_lat:.3f} of the data   (theory (1 - 1/n)^n = {(1 - 1/n_lat)**n_lat:.3f})")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, name in zip(axes, ("median", "95th percentile")):
    reps_, point = statistics[name]; lo, hi = percentile_ci(reps_)
    ax.hist(reps_, bins=60, alpha=0.7)
    ax.axvline(point, color="k", label=f"sample {name} = {point:.1f}")
    ax.axvline(lo, color="C3", ls="--", label=f"95% percentile CI [{lo:.0f}, {hi:.0f}]"); ax.axvline(hi, color="C3", ls="--")
    ax.set_xlabel(f"bootstrap replicate of the {name} (ms)"); ax.set_ylabel("count"); ax.set_title(f"bootstrap distribution of the {name}"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# the same thing with scipy (different random stream, so expect agreement up to Monte Carlo error)
res = stats.bootstrap((latency,), np.median, confidence_level=0.95, n_resamples=B, method="percentile", random_state=1)
print(f"scipy.stats.bootstrap, median: [{res.confidence_interval.low:.1f}, {res.confidence_interval.high:.1f}],  SE {res.standard_error:.2f}")
""")

nb.md(r"""
For the mean the bootstrap reproduces the formula (SE 6.2 vs 6.1 ms, interval matching the $t$-interval
to within a millisecond) — reassuring, and a good unit test for bootstrap code. For the median and the
95th percentile it produces intervals that no formula in this module could. Notice the p95 histogram:
lumpy and more than three times wider than the median's, because a 95th percentile of 200 values is
decided by the ten largest observations. The bootstrap distribution also reveals skew — the interval for the
median is not symmetric around the estimate.

### A statistic of pairs, and a statistic where the bootstrap fails

For a correlation coefficient the resampling unit is the *pair* $(x_i, y_i)$ — resampling $x$ and $y$
separately would destroy the very thing being measured. We compare the bootstrap interval with the
classical Fisher-$z$ interval $\tanh\big(\operatorname{artanh}(r) \pm 1.96/\sqrt{n-3}\big)$. Then we
bootstrap the sample **maximum** of 50 uniform draws and compare with the true sampling distribution,
which we know because we know the population.
""")

nb.code(r"""
n_pairs, rho = 50, 0.6
Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_pairs)
r_hat = np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]
pair_idx = rng.integers(0, n_pairs, size=(B, n_pairs))
xs, ys = Z[pair_idx, 0], Z[pair_idx, 1]                          # resample PAIRS together
xs = xs - xs.mean(1, keepdims=True); ys = ys - ys.mean(1, keepdims=True)
r_boot = (xs * ys).sum(1) / np.sqrt((xs ** 2).sum(1) * (ys ** 2).sum(1))
fisher = np.tanh(np.arctanh(r_hat) + np.array([-1, 1]) * 1.96 / np.sqrt(n_pairs - 3))
print(f"correlation: r = {r_hat:.3f} (true rho = {rho})   bootstrap SE {r_boot.std(ddof=1):.3f}   percentile CI [{np.quantile(r_boot, 0.025):.3f}, {np.quantile(r_boot, 0.975):.3f}]"
      f"   Fisher-z CI [{fisher[0]:.3f}, {fisher[1]:.3f}]")

# the maximum: bootstrap distribution vs the true sampling distribution
u = rng.random(50); u_max = u.max()
true_max = rng.random(size=(B, 50)).max(1)                      # resample from the POPULATION (possible only in a simulation)
boot_max = u[rng.integers(0, 50, size=(B, 50))].max(1)          # resample from the SAMPLE
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(true_max, bins=50, range=(0.85, 1), alpha=0.6, density=True, label="true sampling distribution of max of 50 Uniform(0,1)")
ax.hist(boot_max, bins=50, range=(0.85, 1), alpha=0.6, density=True, label="bootstrap distribution (resampling the sample)")
ax.axvline(u_max, color="k", ls="--", label=f"observed maximum {u_max:.4f}")
ax.set_xlabel("maximum"); ax.set_ylabel("density"); ax.set_title("The bootstrap fails for the sample maximum"); ax.legend(fontsize=8); plt.show()
print(f"fraction of bootstrap maxima equal to the observed maximum: {(boot_max == u_max).mean():.3f}   (theory 1 - (1 - 1/50)^50 = {1 - (1 - 1/50)**50:.3f})")
print(f"true sampling sd of the max {true_max.std():.4f}   bootstrap sd {boot_max.std():.4f}   -> wrong shape, wrong width, and the bootstrap can never exceed the observed max")
""")

nb.md(r"""
The paired bootstrap for the correlation agrees with the Fisher interval to within a few hundredths,
without needing to know that $\operatorname{artanh}(r)$ is approximately normal. For the maximum the bootstrap distribution is a spike
at the observed value (63% of the replicates) with a left tail — nothing like the truth, which lives
mostly *above* the sample maximum. The plug-in principle fails because the maximum depends on the
extreme tail of $F$, where the empirical distribution is a poor substitute.

**Try it yourself:** increase `B` from 10,000 to 50,000 for the median — does the interval change much?
(It should not: the bootstrap's error is dominated by $n$, not by $B$.) Then try `n_lat = 20` latencies.
""")

# ---------------------------------------------------------------- 7. sample size
nb.md(r"""
## 7. Sample-size planning

Invert the standard-error formulas: to estimate a mean to within $\pm e$ with confidence $1 - \alpha$,
$n \ge (z_{\alpha/2}\sigma/e)^2$; for a proportion, $n \ge z_{\alpha/2}^2\,p(1-p)/e^2$, with $p = 1/2$ as
the worst case. The cost of precision grows like $1/e^2$.
""")

nb.code(r"""
def n_for_mean(sigma, e, conf=0.95):
    z = stats.norm.ppf(0.5 + conf / 2)
    return int(np.ceil((z * sigma / e) ** 2))

def n_for_proportion(e, conf=0.95, p=0.5):
    z = stats.norm.ppf(0.5 + conf / 2)
    return int(np.ceil(z * z * p * (1 - p) / e ** 2))

print(f"mean delivery time to +/- 2 min, sigma ~ 15: n = {n_for_mean(15, 2)};  to +/- 1 min: {n_for_mean(15, 1)}")
print(f"proportion to +/- 3 points, no prior guess:  n = {n_for_proportion(0.03)};  +/- 1 point: {n_for_proportion(0.01)};  at 99% confidence, +/- 3: {n_for_proportion(0.03, conf=0.99)}")
print(f"conversion rate near 2% to +/- 1 point: n = {n_for_proportion(0.01, p=0.02)} visitors per variant")
print(f"distribution-free (Hoeffding) for +/- 3 points at 95%: n = {int(np.ceil(np.log(2/0.05) / (2 * 0.03**2)))}  vs CLT {n_for_proportion(0.03)}")

print("\ntest-set size for a 95% interval on a model's accuracy:")
margins = [0.05, 0.02, 0.01, 0.005]
print(f"{'accuracy':>10s}" + "".join(f"{'+/- ' + str(int(e*1000)/10) + ' pt':>12s}" for e in margins))
for acc in (0.5, 0.7, 0.9, 0.95, 0.99):
    print(f"{acc:10.2f}" + "".join(f"{n_for_proportion(e, p=acc):12,d}" for e in margins))

es = np.logspace(-2.5, -1, 100)
for acc in (0.5, 0.9, 0.99):
    plt.loglog(es * 100, [n_for_proportion(e, p=acc) for e in es], label=f"accuracy near {acc}")
plt.loglog(es * 100, [n_for_mean(15, e * 500) for e in es], "k--", label=r"mean, $\sigma = 15$, $e$ in units of 500")
plt.xlabel("required half-width of the 95% interval (percentage points)"); plt.ylabel("required sample size n")
plt.title("Sample size grows like 1/e^2"); plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3); plt.show()
""")

nb.md(r"""
Every line has slope $-2$ on log–log axes: a tenfold tighter interval costs a hundredfold more data.
Near 50% accuracy — the regime of hard benchmarks — a ± 1 point claim needs about 9,600 examples; a
100-prompt evaluation has a margin of about ± 10 points. Plan the test set *before* labelling.
""")

# ---------------------------------------------------------------- 8. model performance
nb.md(r"""
## 8. A test-set accuracy is a sample proportion

We train two simple classifiers on half of `load_breast_cancer` (569 tumours, 30 features) and evaluate
on the other half ($n = 285$): a **nearest-centroid** classifier on standardised features, and a weaker
**one-feature threshold** on mean radius. For each accuracy we compute the formula standard error
$\sqrt{\hat a(1 - \hat a)/n}$, the Wilson interval and a bootstrap interval (resampling test *examples*),
and then compare the two models the right way — on their per-example disagreements.
""")

nb.code(r"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X_all, y_all = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.5, random_state=0, stratify=y_all)
n_te = len(y_te)

# model A: nearest centroid on standardised features (statistics from the training half only)
mu_tr, sd_tr = X_tr.mean(0), X_tr.std(0)
Z_tr, Z_te = (X_tr - mu_tr) / sd_tr, (X_te - mu_tr) / sd_tr
c0, c1 = Z_tr[y_tr == 0].mean(0), Z_tr[y_tr == 1].mean(0)
pred_A = (((Z_te - c1) ** 2).sum(1) < ((Z_te - c0) ** 2).sum(1)).astype(int)

# model B: a single threshold on feature 0 (mean radius), chosen on the training half
cands = np.quantile(X_tr[:, 0], np.linspace(0.05, 0.95, 91))
train_acc = [((X_tr[:, 0] < c).astype(int) == y_tr).mean() for c in cands]
thr = cands[int(np.argmax(train_acc))]
pred_B = (X_te[:, 0] < thr).astype(int)

correct_A, correct_B = pred_A == y_te, pred_B == y_te
boot_idx = rng.integers(0, n_te, size=(10_000, n_te))
for name, correct in [("A: nearest centroid", correct_A), ("B: one-feature threshold", correct_B)]:
    acc = correct.mean(); se = np.sqrt(acc * (1 - acc) / n_te)
    lo_w, hi_w = wilson_ci(correct.sum(), n_te)
    boot_acc = correct[boot_idx].mean(1)
    print(f"{name:26s} accuracy {acc:.4f} = {correct.sum()}/{n_te}   SE {se:.4f}   Wilson [{lo_w:.3f}, {hi_w:.3f}]"
          f"   bootstrap [{np.quantile(boot_acc, 0.025):.3f}, {np.quantile(boot_acc, 0.975):.3f}]")

# paired comparison: only the disagreements carry information about the difference
n10 = int((correct_A & ~correct_B).sum()); n01 = int((~correct_A & correct_B).sum())
d = (n10 - n01) / n_te
se_paired = np.sqrt(((n10 + n01) / n_te - d ** 2) / n_te)
aA, aB = correct_A.mean(), correct_B.mean()
se_unpaired = np.sqrt(aA * (1 - aA) / n_te + aB * (1 - aB) / n_te)
boot_diff = (correct_A[boot_idx].astype(float) - correct_B[boot_idx]).mean(1)
print(f"\nA right & B wrong: {n10},  B right & A wrong: {n01},  agree on {n_te - n10 - n01} of {n_te}")
print(f"difference A - B = {d:+.4f}   paired SE {se_paired:.4f}  [{d - 1.96*se_paired:+.3f}, {d + 1.96*se_paired:+.3f}]"
      f"   paired bootstrap [{np.quantile(boot_diff, 0.025):+.3f}, {np.quantile(boot_diff, 0.975):+.3f}]")
print(f"unpaired SE (ignores that both models see the same examples) {se_unpaired:.4f} -> interval {se_unpaired/se_paired:.2f}x too wide")
print(f"for a +/- 2 point interval on A's accuracy you would need n = {n_for_proportion(0.02, p=aA)} test examples; on the difference n = {int(np.ceil(1.96**2 * ((n10 + n01)/n_te - d**2) / 0.02**2))}")
""")

nb.md(r"""
Both accuracies carry an SE of about 1.6–2 points on 285 examples, and the formula, Wilson and bootstrap
intervals agree. The difference between the models is estimated more precisely from the *paired* data
than the unpaired formula suggests, and the paired bootstrap confirms the paired formula. The last line
is the sobering one: to pin an accuracy down to ± 2 points needs several hundred more examples than we
have.

### The best of $k$ seeds is biased upward

Suppose $k$ training runs (or checkpoints, or hyper-parameter settings) all have the *same* true
accuracy 0.80 and are each scored on a validation set of 200 examples. Reporting the best one
overstates the truth by roughly $\operatorname{SE} \times \mathbb E[\max \text{ of } k \text{ standard normals}]$.
""")

nb.code(r"""
from scipy import integrate

def expected_max_normals(k):           # E[max of k iid N(0,1)] = integral of x * k * phi(x) * Phi(x)^(k-1)
    return integrate.quad(lambda x: x * k * stats.norm.pdf(x) * stats.norm.cdf(x) ** (k - 1), -10, 10)[0]

n_val, acc_true, reps = 200, 0.80, 50_000
se_val = np.sqrt(acc_true * (1 - acc_true) / n_val)
print(f"validation SE at accuracy {acc_true}, n = {n_val}: {se_val:.4f}\n")
print(f"{'k':>4s}{'E[best score]':>16s}{'optimism':>10s}{'SE x E[max]':>14s}")
best_of = {}
for k in (1, 3, 5, 10, 24):
    scores = rng.binomial(n_val, acc_true, size=(reps, k)) / n_val
    best_of[k] = scores.max(1)
    print(f"{k:4d}{best_of[k].mean():16.4f}{best_of[k].mean() - acc_true:+10.4f}{se_val * expected_max_normals(k):14.4f}")

plt.hist(best_of[1], bins=np.arange(0.70, 0.93, 0.005), alpha=0.6, density=True, label="a single run (k = 1)")
plt.hist(best_of[24], bins=np.arange(0.70, 0.93, 0.005), alpha=0.6, density=True, label="best of k = 24 runs")
plt.axvline(acc_true, color="k", ls="--", label="true accuracy of every run")
plt.xlabel("reported validation accuracy"); plt.ylabel("density"); plt.title("Selecting the best of k equally good models")
plt.legend(); plt.show()
""")

nb.md(r"""
Twenty-four equally good configurations, and the winner reports about 85% when every one of them is
an 80% model: 5 points of pure selection. The $\sigma\,\mathbb E[\max]$ approximation tracks the
simulation closely. Choose on a validation set, report on a separate test set, and state the number of
seeds and the spread.

**Try it yourself:** set `n_val = 2000`. The optimism shrinks by about $\sqrt{10}$ — selection bias is
a sampling-noise phenomenon, so more validation data *does* cure it.
""")

nb.md(r"""
## Summary

- A statistic is a random variable; its sampling distribution can be written out for tiny populations and simulated for everything else. Random sampling is what makes inference valid; a biased mechanism stays biased at any $n$.
- Mean and standard deviation have breakdown point 0; median, MAD and trimmed means resist outliers. `ddof=0` versus `ddof=1` differ by the factor $(n-1)/n$.
- $\text{MSE} = \text{bias}^2 + \text{variance}$; the unbiased variance estimator has the largest MSE of the three divisors, and James–Stein beats the unbiased mean vector for every $\theta$ when $p \ge 3$.
- The standard error of a mean is $\sigma/\sqrt n$ and of a proportion $\sqrt{p(1-p)/n}$, exactly; the CLT supplies the normal shape.
- A 95% confidence interval is a procedure with 95% coverage; $z$-with-$s$ under-covers at small $n$, and the Wald interval under-covers near $p = 0$ or $1$ — use $t$ and Wilson.
- The bootstrap resamples the sample to estimate standard errors and percentile intervals for statistics without formulas; it reproduces $s/\sqrt n$ for the mean, handles medians, quantiles and correlations, and fails for the maximum.
- Sample size grows like $1/e^2$: ± 3 points needs about 1,000 observations, ± 1 point about 10,000.
- A test accuracy is a proportion with an SE; compare models on paired disagreements; the best of $k$ noisy scores is biased upward by about $\text{SE} \times \mathbb E[\max_k]$.

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


def assert_hash_margin(value: float, decimals: int, name: str, min_frac: float = 0.02):
    """Guard: a hashed float must not sit within `min_frac` of a unit in the last kept digit of a
    rounding boundary (x.5 in that digit), so floating-point noise cannot flip the hash."""
    scaled = value * 10 ** decimals
    frac = abs(scaled - np.floor(scaled) - 0.5)
    assert frac > min_frac, f"{name} = {value!r} is too close to a rounding boundary at {decimals} decimals"


ex = ExerciseSet(
    "Module 10 — Exercises",
    intro_md=r"""
These exercises follow the order of the concept guide: descriptive statistics, estimators and their bias,
standard errors and confidence intervals (including Wilson's), a coverage simulation, the bootstrap,
sample-size planning, the MSE decomposition, and finally the paired comparison of two classifiers.
Several checks compare your answer against a *hidden* value, so compute carefully rather than guessing.
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- 1. descriptive statistics (warm-up)
SOL_1 = r"""
def my_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x) / len(x))

def my_median(x):
    s = np.sort(np.asarray(x, dtype=float)); n = len(s); m = n // 2
    return float(s[m]) if n % 2 else float(0.5 * (s[m - 1] + s[m]))

def my_var(x, ddof=0):
    x = np.asarray(x, dtype=float); mu = my_mean(x)
    return float(np.sum((x - mu) ** 2) / (len(x) - ddof))

def my_mad(x):
    x = np.asarray(x, dtype=float); m = my_median(x)
    return my_median(np.abs(x - m))

def my_trimmed_mean(x, alpha):
    s = np.sort(np.asarray(x, dtype=float)); k = int(np.floor(alpha * len(s)))
    return float(np.sum(s[k:len(s) - k]) / (len(s) - 2 * k))
"""
DATA_1 = r"""
times = np.array([120, 135, 140, 150, 155, 160, 170, 180, 210, 2400], dtype=float)
v_even = np.array([3., 1., 4., 1., 5., 9.])
v_odd = np.array([2., 7., 1., 8., 2., 8., 1.])
"""
ns1 = run(SOL_1, DATA_1 + r"""
mad_times = my_mad(times)
trim_times = my_trimmed_mean(times, 0.10)
""")
ex.add(
    title="Descriptive statistics from scratch", difficulty=1,
    prompt_md=r"""
Implement five summaries using only `np.sort`, `np.sum`, `np.abs`, `np.floor`, `len` and indexing —
**not** `np.mean`, `np.median`, `np.var`, `np.std`, `np.quantile` or anything from `scipy.stats`
(those are used only in the checks):

* `my_mean(x)`;
* `my_median(x)` — the middle value for odd $n$, the average of the two middle values for even $n$;
* `my_var(x, ddof=0)` — $\frac{1}{n - \text{ddof}}\sum_i (x_i - \bar x)^2$;
* `my_mad(x)` — the median absolute deviation $\operatorname{median}_i\,|x_i - \operatorname{median}(x)|$ (unscaled);
* `my_trimmed_mean(x, alpha)` — drop the $\lfloor \alpha n \rfloor$ smallest and the $\lfloor \alpha n \rfloor$ largest values, average the rest.

All return Python floats. Test data: the ten response times from the guide (`times`), plus an even-length
and an odd-length vector.
""",
    starter=DATA_1 + r"""

def my_mean(x):
    # YOUR CODE HERE
    return None

def my_median(x):
    # YOUR CODE HERE
    return None

def my_var(x, ddof=0):
    # YOUR CODE HERE
    return None

def my_mad(x):
    # YOUR CODE HERE
    return None

def my_trimmed_mean(x, alpha):
    # YOUR CODE HERE
    return None
""",
    solution=DATA_1 + "\n" + SOL_1,
    check=rf"""
def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None
check_close('mean of times', _safe(my_mean, times), np.mean(times))
check_close('median of times (even n)', _safe(my_median, times), np.median(times))
check_close('median of an odd-length vector', _safe(my_median, v_odd), np.median(v_odd))
check_close('variance with divisor n', _safe(my_var, v_even, 0), np.var(v_even, ddof=0))
check_close('variance with divisor n-1', _safe(my_var, v_even, 1), np.var(v_even, ddof=1))
check_close('MAD of v_odd matches scipy', _safe(my_mad, v_odd), stats.median_abs_deviation(v_odd))
check_hash('hidden: MAD of times', _safe(my_mad, times), '{sha_of(ns1["mad_times"])}')
check_close('10% trimmed mean of v_even matches scipy', _safe(my_trimmed_mean, v_even, 0.10), stats.trim_mean(v_even, 0.10))
check_hash('hidden: 10% trimmed mean of times', _safe(my_trimmed_mean, times, 0.10), '{sha_of(ns1["trim_times"])}')
check('trimmed mean with alpha = 0 is the mean', (lambda a, b: a is not None and b is not None and abs(a - b) < 1e-9)(_safe(my_trimmed_mean, times, 0.0), _safe(my_mean, times)))
""",
    explanation_md=r"""
Sorting does all the work: the median reads off the middle of the sorted array (two middles for even
$n$), the trimmed mean slices off $k = \lfloor \alpha n \rfloor$ values at each end, and the MAD applies
the median twice. For `times` the mean is 382 while the median is 157.5, the 10%-trimmed mean 162.5 and
the MAD 20: the one outlier (2400 ms) moves the mean by more than 200 ms and the robust summaries by a few.
The common wrong turn in `my_var` is dividing by $n$ regardless of `ddof`, or computing deviations from the
median instead of the mean. Note that `np.var` defaults to `ddof=0` while pandas defaults to `ddof=1`.
""",
)

# ---------------------------------------------------------------- 2. variance bias by simulation (warm-up)
SOL_2 = r"""
def var_ddof(x, ddof):
    x = np.asarray(x, dtype=float)
    return float(((x - x.mean()) ** 2).sum() / (len(x) - ddof))

def estimate_bias(estimator, n, n_reps, rng, true_value=1.0):
    samples = rng.normal(0.0, 1.0, size=(n_reps, n))
    estimates = np.array([estimator(row) for row in samples])
    return float(estimates.mean() - true_value)
"""
ns2 = run(SOL_2, r"""
b0 = estimate_bias(lambda s: var_ddof(s, 0), 5, 20_000, np.random.default_rng(1))
b1 = estimate_bias(lambda s: var_ddof(s, 1), 5, 20_000, np.random.default_rng(1))
""")
assert -0.23 < ns2["b0"] < -0.17 and -0.03 < ns2["b1"] < 0.03, (ns2["b0"], ns2["b1"])
ex.add(
    title="The biased and unbiased variance estimators", difficulty=1,
    prompt_md=r"""
1. Implement `var_ddof(x, ddof)` $= \frac{1}{n - \text{ddof}}\sum_i (x_i - \bar x)^2$ (a float).
2. Implement `estimate_bias(estimator, n, n_reps, rng, true_value=1.0)`: draw `n_reps` samples of size
   `n` from $\mathcal N(0, 1)$ with `rng.normal(0.0, 1.0, size=(n_reps, n))`, apply `estimator` (a function
   of a 1-D array) to each row, and return the average estimate minus `true_value` — the Monte Carlo
   estimate of the bias.

With $n = 5$ the guide predicts a bias of $-\sigma^2/n = -0.2$ for `ddof=0` and $0$ for `ddof=1`.
""",
    starter=r"""
def var_ddof(x, ddof):
    # YOUR CODE HERE
    return None

def estimate_bias(estimator, n, n_reps, rng, true_value=1.0):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_2,
    check=r"""
try:
    got0 = var_ddof([1., 2., 3., 4.], 0); got1 = var_ddof([1., 2., 3., 4.], 1)
except Exception:
    got0 = got1 = None
check_close('var_ddof([1,2,3,4], 0) = 1.25', got0, 1.25)
check_close('var_ddof([1,2,3,4], 1) = 5/3', got1, 5 / 3)
try:
    b0 = estimate_bias(lambda s: var_ddof(s, 0), 5, 20_000, np.random.default_rng(1))
    b1 = estimate_bias(lambda s: var_ddof(s, 1), 5, 20_000, np.random.default_rng(1))
except Exception:
    b0 = b1 = None
check('bias of the ddof=0 estimator at n=5 is about -0.2', b0 is not None and -0.23 < b0 < -0.17, hint='expected -sigma^2/n = -0.2')
check('bias of the ddof=1 estimator is about 0', b1 is not None and -0.03 < b1 < 0.03)
try:
    bm = estimate_bias(lambda s: s.mean(), 5, 20_000, np.random.default_rng(2), true_value=0.0)
except Exception:
    bm = None
check('the sample mean is unbiased for the mean', bm is not None and abs(bm) < 0.02)
""",
    explanation_md=r"""
The deviations from $\bar x$ are smaller than the deviations from $\mu$ because $\bar x$ was chosen to
minimise them: $\mathbb E\sum(X_i - \bar X)^2 = (n - 1)\sigma^2$, so dividing by $n$ gives
$\mathbb E[\hat\sigma^2] = \frac{n-1}{n}\sigma^2$, a bias of $-\sigma^2/n = -0.2$ at $n = 5$, which the
simulation reproduces to within its Monte Carlo error of about $\sqrt{\operatorname{Var}(\hat\sigma^2)/20000} \approx 0.004$.
Dividing by $n - 1$ removes the bias exactly. The common wrong turn is to compute the bias of the
*standard deviation* instead — $s$ is not unbiased for $\sigma$ even with `ddof=1`.
""",
)

# ---------------------------------------------------------------- 3. SE and z-/t-interval (warm-up)
SOL_3 = r"""
def standard_error(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))

def z_interval(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    z = stats.norm.ppf(0.5 + conf / 2)
    h = z * standard_error(x)
    return (float(x.mean() - h), float(x.mean() + h))

def t_interval(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    t = stats.t.ppf(0.5 + conf / 2, df=len(x) - 1)
    h = t * standard_error(x)
    return (float(x.mean() - h), float(x.mean() + h))
"""
DATA_3 = r"""
x_conc = np.array([12.1, 9.8, 11.4, 10.6, 12.9, 11.1, 10.2, 11.7])   # eight assay readings (mg/l)
"""
ns3 = run(SOL_3, DATA_3 + r"""
se3 = standard_error(x_conc)
lo3, hi3 = z_interval(x_conc)
tlo3, thi3 = t_interval(x_conc)
width_ratio = (thi3 - tlo3) / (hi3 - lo3)
""")
assert_hash_margin(ns3["se3"], 4, "se3")
assert_hash_margin(ns3["lo3"], 3, "lo3")
assert_hash_margin(ns3["hi3"], 3, "hi3")
ex.add(
    title="Standard error, z-interval and t-interval", difficulty=1,
    prompt_md=r"""
For the eight assay readings `x_conc` implement

* `standard_error(x)` $= s/\sqrt n$ with $s$ the sample standard deviation (`ddof=1`);
* `z_interval(x, conf=0.95)` → the tuple `(lo, hi)` $= \bar x \pm z_{\alpha/2}\, s/\sqrt n$, using
  `stats.norm.ppf` for the quantile;
* `t_interval(x, conf=0.95)` → the same with $t_{n-1,\,\alpha/2}$ from `stats.t.ppf`.

The hidden checks hash the standard error to 4 decimals and the $z$-interval bounds to 3 decimals.
""",
    starter=DATA_3 + r"""

def standard_error(x):
    # YOUR CODE HERE
    return None

def z_interval(x, conf=0.95):
    # YOUR CODE HERE
    return None

def t_interval(x, conf=0.95):
    # YOUR CODE HERE
    return None
""",
    solution=DATA_3 + "\n" + SOL_3,
    check=rf"""
try:
    se = standard_error(x_conc)
except Exception:
    se = None
check_hash('hidden: standard error of x_conc', se, '{sha_of(ns3["se3"], decimals=4)}', decimals=4)
check_close('SE matches scipy.stats.sem', se, stats.sem(x_conc))
try:
    lo, hi = z_interval(x_conc)
except Exception:
    lo = hi = None
check_hash('hidden: lower z bound', lo, '{sha_of(ns3["lo3"], decimals=3)}', decimals=3)
check_hash('hidden: upper z bound', hi, '{sha_of(ns3["hi3"], decimals=3)}', decimals=3)
try:
    tlo, thi = t_interval(x_conc)
    ref = stats.t.interval(0.95, df=len(x_conc) - 1, loc=x_conc.mean(), scale=stats.sem(x_conc))
except Exception:
    tlo = thi = None; ref = (0.0, 0.0)
check_close('t-interval matches scipy.stats.t.interval', (tlo, thi), ref)
check('t-interval is wider than the z-interval', (tlo is not None and lo is not None) and (thi - tlo) > (hi - lo))
try:
    lo99, hi99 = z_interval(x_conc, conf=0.99)
except Exception:
    lo99 = hi99 = None
check('99% interval is wider than 95%', lo99 is not None and (hi99 - lo99) > (hi - lo))
""",
    explanation_md=r"""
$\bar x = 11.225$, $s = 1.022$, so the standard error is $1.022/\sqrt 8 = 0.361$. The $z$-interval is
$11.225 \pm 1.96 \times 0.361 = [10.517, 11.933]$. With only 8 observations the $t$ quantile is
$t_{7} = 2.365$ rather than 1.96, so the honest interval is about 21% wider: the $z$-interval pretends
that $s$ is $\sigma$. The common wrong turn is `ddof=0` in the standard deviation (too small by a factor
$\sqrt{7/8}$) or using `stats.norm.ppf(conf)` instead of `stats.norm.ppf(0.5 + conf/2)`, which gives a
one-sided quantile (1.645 for 95%).
""",
)

# ---------------------------------------------------------------- 4. Wilson interval (core)
SOL_4 = r"""
def wilson_interval(k, n, conf=0.95):
    z = stats.norm.ppf(0.5 + conf / 2); z2 = z * z
    p = k / n
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = (z / (1 + z2 / n)) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (float(centre - half), float(centre + half))

def wald_interval(k, n, conf=0.95):
    z = stats.norm.ppf(0.5 + conf / 2); p = k / n
    h = z * np.sqrt(p * (1 - p) / n)
    return (float(p - h), float(p + h))
"""
ns4 = run(SOL_4, r"""
w9 = wilson_interval(9, 200); w0 = wilson_interval(0, 20); w141 = wilson_interval(141, 150); w20 = wilson_interval(20, 20)
w2_99 = wilson_interval(2, 150, conf=0.99)
""")
ex.add(
    title="The Wilson score interval", difficulty=2,
    prompt_md=r"""
Implement `wilson_interval(k, n, conf=0.95)` for $k$ successes in $n$ trials, returning `(lo, hi)`:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \;\pm\; \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}, \qquad \hat p = k/n,\; z = z_{\alpha/2}.$$

Also implement the plain `wald_interval(k, n, conf=0.95)` $= \hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ so you can
compare. The checks compare against `scipy.stats.binomtest(k, n).proportion_ci(method="wilson")` and probe
the boundary cases $k = 0$ and $k = n$.
""",
    starter=r"""
def wilson_interval(k, n, conf=0.95):
    # YOUR CODE HERE
    return None

def wald_interval(k, n, conf=0.95):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_4,
    check=r"""
def _sp(k, n, conf=0.95):
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=conf, method="wilson")
    return (ci.low, ci.high)
def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:
        return None
check_close('Wilson 9/200 matches scipy', _safe(wilson_interval, 9, 200), _sp(9, 200))
check_close('Wilson 0/20 matches scipy', _safe(wilson_interval, 0, 20), _sp(0, 20))
check_close('Wilson 20/20 matches scipy', _safe(wilson_interval, 20, 20), _sp(20, 20))
check_close('Wilson 141/150 matches scipy', _safe(wilson_interval, 141, 150), _sp(141, 150))
check_close('Wilson 2/150 at 99% matches scipy', _safe(wilson_interval, 2, 150, conf=0.99), _sp(2, 150, 0.99))
w0 = _safe(wilson_interval, 0, 20)
check('Wilson lower bound at k = 0 is 0 and the upper bound is positive', w0 is not None and abs(w0[0]) < 1e-9 and 0.1 < w0[1] < 0.2)
wd = _safe(wald_interval, 0, 20)
check('Wald interval at k = 0 has zero width (that is the problem)', wd is not None and abs(wd[1] - wd[0]) < 1e-12)
wd2 = _safe(wald_interval, 2, 150)
check('Wald interval for 2/150 has a negative lower bound', wd2 is not None and wd2[0] < 0)
""",
    explanation_md=r"""
Wilson keeps the unknown $p$ inside the standard error and solves $|\hat p - p| \le z\sqrt{p(1-p)/n}$ as a
quadratic in $p$; the two roots are the interval. Its centre $(\hat p + z^2/2n)/(1 + z^2/n)$ is pulled toward
$1/2$, which is what gives it a positive upper bound at $k = 0$ (about $z^2/(n + z^2) = 0.161$ for $n = 20$,
the rule of three in disguise) and keeps it inside $[0, 1]$ at $k = n$. The Wald interval plugs $\hat p$ into
the SE, so at $k = 0$ it collapses to $[0, 0]$ and for $2/150$ it dips below zero. The usual bug is to forget
the factor $1/(1 + z^2/n)$ on the half-width or to use $z^2/(4n)$ instead of $z^2/(4n^2)$ under the root.
""",
)

# ---------------------------------------------------------------- 5. coverage simulation (core)
HELPERS_5 = r"""
def interval_z_with_s(x, conf=0.95):          # the common mistake: z quantile, estimated s
    z = stats.norm.ppf(0.5 + conf / 2); h = z * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h

def interval_t(x, conf=0.95):                 # the correct small-sample interval
    t = stats.t.ppf(0.5 + conf / 2, df=len(x) - 1); h = t * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h
"""
SOL_5 = r"""
def coverage(make_interval, n, mu, sigma, n_reps, rng):
    hits = 0
    for _ in range(n_reps):
        x = rng.normal(mu, sigma, size=n)
        lo, hi = make_interval(x)
        hits += (lo <= mu <= hi)
    return hits / n_reps
"""
ns5 = run(HELPERS_5 + SOL_5, r"""
cz = coverage(interval_z_with_s, 4, 10.0, 2.0, 20_000, np.random.default_rng(123))
ct = coverage(interval_t, 4, 10.0, 2.0, 20_000, np.random.default_rng(123))
cz30 = coverage(interval_z_with_s, 30, 10.0, 2.0, 20_000, np.random.default_rng(5))
""")
assert 0.84 < ns5["cz"] < 0.87 and 0.94 < ns5["ct"] < 0.96 and 0.93 < ns5["cz30"] < 0.955, (ns5["cz"], ns5["ct"], ns5["cz30"])
ex.add(
    title="A coverage experiment", difficulty=2,
    prompt_md=r"""
Two interval procedures are given below (do not change them). Implement

`coverage(make_interval, n, mu, sigma, n_reps, rng)`

which draws `n_reps` samples of size `n` from $\mathcal N(\mu, \sigma^2)$ with `rng.normal(mu, sigma, size=n)`,
calls `make_interval(x)` on each to get `(lo, hi)`, and returns the fraction of intervals with
`lo <= mu <= hi` (a float). Then predict before you run: at $n = 4$, what coverage does the $z$-with-$s$
interval achieve? (The guide says $2F_{t_3}(1.96) - 1$.) And at $n = 30$?
""",
    starter=HELPERS_5 + r"""

def coverage(make_interval, n, mu, sigma, n_reps, rng):
    # YOUR CODE HERE
    return None
""",
    solution=HELPERS_5 + "\n" + SOL_5,
    check=r"""
try:
    cz = coverage(interval_z_with_s, 4, 10.0, 2.0, 20_000, np.random.default_rng(123))
    ct = coverage(interval_t, 4, 10.0, 2.0, 20_000, np.random.default_rng(123))
    cz30 = coverage(interval_z_with_s, 30, 10.0, 2.0, 20_000, np.random.default_rng(5))
except Exception:
    cz = ct = cz30 = None
check('coverage returns a float in [0, 1]', isinstance(cz, float) and 0.0 <= cz <= 1.0)
check('t-interval at n = 4 covers about 95%', ct is not None and 0.94 < ct < 0.96, hint='expected ~0.95')
check('z-with-s at n = 4 covers only about 85-86%', cz is not None and 0.84 < cz < 0.87, hint='expected 2*F_t3(1.96)-1 = 0.855')
check('z-with-s at n = 30 is nearly fine (about 94%)', cz30 is not None and 0.93 < cz30 < 0.955)
try:
    always = coverage(lambda x: (-np.inf, np.inf), 4, 10.0, 2.0, 100, np.random.default_rng(0))
except Exception:
    always = None
check_close('an infinite interval covers 100%', always, 1.0)
""",
    explanation_md=r"""
Coverage is just a Monte Carlo estimate of $P(L \le \mu \le U)$: simulate, count, divide. The $t$-interval
achieves its nominal 95% at every $n$ because the $t$ quantile was built for exactly this situation; the
$z$-with-$s$ interval reaches only $2F_{t_3}(1.96) - 1 \approx 0.855$ at $n = 4$, because $s$ is a noisy
stand-in for $\sigma$ and 1.96 is too small a multiplier (the right one is $t_3 = 3.18$). By $n = 30$ the
two quantiles differ by 4% and the coverage gap has mostly closed. With 20,000 repetitions the Monte Carlo
standard error of a coverage estimate near 0.95 is $\sqrt{0.95 \times 0.05/20000} \approx 0.0015$, so the
check ranges are generous. A common wrong turn is to test `lo <= x.mean() <= hi`, which is always true.
""",
)

# ---------------------------------------------------------------- 6. bootstrap percentile CI (core)
SOL_6 = r"""
def bootstrap_ci(x, stat, B, rng, conf=0.95):
    x = np.asarray(x, dtype=float); n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    replicates = np.array([stat(x[row]) for row in idx])
    lo, hi = np.quantile(replicates, [(1 - conf) / 2, (1 + conf) / 2])
    return (float(lo), float(hi), float(replicates.std(ddof=1)))
"""
DATA_6 = r"""
lat30 = np.array([152.6, 296.0, 276.7, 116.2, 129.2, 115.2, 199.4, 145.9, 217.9, 59.6,
                  328.3, 142.9, 210.8, 140.1, 124.1, 189.1, 226.5, 135.6, 139.0, 211.3,
                  97.1, 70.3, 182.8, 107.3, 57.4, 99.8, 118.7, 82.6, 71.1, 152.8])   # 30 request latencies (ms)
"""
ns6 = run(SOL_6, DATA_6 + r"""
lo6, hi6, se6 = bootstrap_ci(lat30, np.median, 2000, np.random.default_rng(0))
mlo, mhi, mse6 = bootstrap_ci(lat30, np.mean, 2000, np.random.default_rng(0))
se_formula = lat30.std(ddof=0) / np.sqrt(len(lat30))
""")
assert 110 < ns6["lo6"] < 125 and 160 < ns6["hi6"] < 180 and 10 < ns6["se6"] < 14, (ns6["lo6"], ns6["hi6"], ns6["se6"])
assert abs(ns6["mse6"] / ns6["se_formula"] - 1) < 0.12, (ns6["mse6"], ns6["se_formula"])
ex.add(
    title="A general bootstrap percentile interval", difficulty=2,
    prompt_md=r"""
Implement `bootstrap_ci(x, stat, B, rng, conf=0.95)` returning the tuple `(lo, hi, se)`:

* draw `B` bootstrap samples of size $n$ from `x` *with replacement* (e.g. `rng.integers(0, n, size=(B, n))`
  as index arrays, or `rng.choice(x, size=n, replace=True)` in a loop);
* apply `stat` (a function of a 1-D array, e.g. `np.median`) to each;
* `lo, hi` are the $\frac{1-\text{conf}}{2}$ and $\frac{1+\text{conf}}{2}$ quantiles of the replicates (`np.quantile`),
  `se` is their standard deviation (`ddof=1`).

Apply it to the 30 request latencies `lat30` for the median (the checks use `B = 2000` and a fresh
`np.random.default_rng(0)`; exact values depend on how you draw, so the checks test ranges) and for the
mean, where the bootstrap SE should agree with the plug-in formula $\hat\sigma/\sqrt n$ (with `ddof=0`)
to within about 10%.
""",
    starter=DATA_6 + r"""

def bootstrap_ci(x, stat, B, rng, conf=0.95):
    # YOUR CODE HERE
    return None
""",
    solution=DATA_6 + "\n" + SOL_6,
    check=r"""
try:
    lo, hi, se = bootstrap_ci(lat30, np.median, 2000, np.random.default_rng(0))
except Exception:
    lo = hi = se = None
check('median: lower bound in a plausible range (110-125 ms)', lo is not None and 110 < lo < 125)
check('median: upper bound in a plausible range (160-180 ms)', hi is not None and 160 < hi < 180)
check('median: bootstrap SE about 11-13 ms', se is not None and 10 < se < 14)
check('interval contains the sample median', lo is not None and lo <= np.median(lat30) <= hi)
try:
    mlo, mhi, mse = bootstrap_ci(lat30, np.mean, 2000, np.random.default_rng(0))
    ratio = mse / (lat30.std(ddof=0) / np.sqrt(len(lat30)))
except Exception:
    mlo = mhi = mse = ratio = None
check('mean: bootstrap SE within 12% of sigma_hat/sqrt(n)', ratio is not None and 0.88 < ratio < 1.12)
check('mean: interval roughly symmetric about the sample mean', mlo is not None and abs((mhi - lat30.mean()) - (lat30.mean() - mlo)) < 0.5 * mse)
try:
    wlo, whi, _ = bootstrap_ci(lat30, np.median, 2000, np.random.default_rng(0), conf=0.80)
except Exception:
    wlo = whi = None
check('an 80% interval is narrower than the 95% one', wlo is not None and (whi - wlo) < (hi - lo))
""",
    explanation_md=r"""
The empirical distribution stands in for the population, so resampling rows of indices with replacement
and recomputing the statistic simulates the sampling distribution; quantiles of the replicates give the
percentile interval and their standard deviation the bootstrap SE. For the mean this reproduces the plug-in
formula $\hat\sigma_n/\sqrt n$ (divisor $n$, not $n - 1$, because the bootstrap resamples from $\hat F_n$),
which is the standard unit test for bootstrap code. For the median of 30 right-skewed latencies the
interval is noticeably asymmetric — the bootstrap captures that for free. Two classic bugs: sampling
*without* replacement (every replicate is then a permutation and the statistic never changes), and
resampling `B` values instead of `n`.
""",
)

# ---------------------------------------------------------------- 7. sample size (core)
SOL_7 = r"""
def n_for_mean(sigma, e, conf=0.95):
    z = stats.norm.ppf(0.5 + conf / 2)
    return int(np.ceil((z * sigma / e) ** 2))

def n_for_proportion(e, conf=0.95, p=0.5):
    z = stats.norm.ppf(0.5 + conf / 2)
    return int(np.ceil(z * z * p * (1 - p) / e ** 2))
"""
ns7 = run(SOL_7, r"""
a = n_for_mean(15, 2); b = n_for_mean(15, 1); c = n_for_proportion(0.03); d = n_for_proportion(0.01, p=0.9)
e_ = n_for_proportion(0.03, conf=0.99); f = n_for_proportion(0.02, p=0.75)
""")
assert (ns7["a"], ns7["b"], ns7["c"], ns7["d"], ns7["e_"]) == (217, 865, 1068, 3458, 1844), ns7
ex.add(
    title="Sample-size planning", difficulty=2,
    prompt_md=r"""
Implement two planners that return the smallest **integer** $n$ (round *up* with `np.ceil`, return `int`):

* `n_for_mean(sigma, e, conf=0.95)` $= \lceil (z_{\alpha/2}\,\sigma/e)^2 \rceil$ — the sample size that makes
  the $(1-\alpha)$ interval for a mean have half-width at most $e$;
* `n_for_proportion(e, conf=0.95, p=0.5)` $= \lceil z_{\alpha/2}^2\,p(1-p)/e^2 \rceil$, with $p = 0.5$ as the
  worst-case default.

Take $z_{\alpha/2}$ from `stats.norm.ppf(0.5 + conf / 2)` rather than typing 1.96 — a rounded quantile can
change the integer answer.

Then answer (the hidden checks hash the integers): how many deliveries to estimate a mean delivery time
to ± 2 minutes if $\sigma \approx 15$? How many respondents for a poll to ± 3 points, and to ± 1 point?
How many test images to estimate an accuracy near 90% to ± 1 point?
""",
    starter=r"""
def n_for_mean(sigma, e, conf=0.95):
    # YOUR CODE HERE
    return None

def n_for_proportion(e, conf=0.95, p=0.5):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_7,
    check=rf"""
def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:
        return None
check_hash('hidden: deliveries for +/- 2 min with sigma 15', _safe(n_for_mean, 15, 2), '{sha_of(ns7["a"])}')
check_hash('hidden: deliveries for +/- 1 min', _safe(n_for_mean, 15, 1), '{sha_of(ns7["b"])}')
check_hash('hidden: poll to +/- 3 points (worst case p)', _safe(n_for_proportion, 0.03), '{sha_of(ns7["c"])}')
check_hash('hidden: test images for accuracy near 0.9 to +/- 1 point', _safe(n_for_proportion, 0.01, p=0.9), '{sha_of(ns7["d"])}')
check_hash('hidden: poll to +/- 3 points at 99% confidence', _safe(n_for_proportion, 0.03, conf=0.99), '{sha_of(ns7["e_"])}')
check('results are Python ints', isinstance(_safe(n_for_mean, 15, 2), int) and isinstance(_safe(n_for_proportion, 0.03), int))
r1, r2 = _safe(n_for_proportion, 0.01), _safe(n_for_proportion, 0.02)
check('halving the margin roughly quadruples n', r1 is not None and r2 is not None and 3.9 < r1 / r2 < 4.1)
""",
    explanation_md=r"""
Set the half-width $z\,\sigma/\sqrt n$ (or $z\sqrt{p(1-p)/n}$) equal to $e$ and solve for $n$, then round
up because a fraction of an observation does not exist. Delivery times: $(1.96 \times 15/2)^2 = 216.1 \to 217$;
to ± 1 minute, 865 — four times as many for half the margin. The poll: $0.9604/0.0009 = 1067.1 \to 1068$,
the origin of the "thousand-person poll"; ± 1 point needs 9,604. An accuracy near 0.9 to ± 1 point needs
$3.8415 \times 0.09/0.0001 = 3457.3 \to 3458$ images. At 99% confidence $z = 2.576$ and the poll needs 1,844.
The common wrong turn is rounding to the nearest integer (217 vs 216 would fail the precision target) or
quoting the margin as a full width instead of a half-width.
""",
)

# ---------------------------------------------------------------- 8. conceptual (core)
ex.add_conceptual(
    title="Reading intervals and choosing estimators", difficulty=2,
    prompt_md=r"""
Three short questions; write your answers down before opening the model answer.

1. A product manager reads "the 95% confidence interval for the conversion rate is $[2.1\%, 3.4\%]$" and
   says: "So there is a 95% probability that the true rate is between 2.1% and 3.4%." What is wrong with
   this sentence, what is the correct frequentist reading, and under what additional assumption would the
   manager's sentence become literally true?
2. On a sample of 20 user sessions none crashed. A colleague reports "crash rate 0% (95% CI: 0% to 0%)".
   Diagnose the error and give a defensible upper bound.
3. For normal data the sample mean has lower MSE than the sample median; for Laplace or Student-$t_3$ data
   the median wins. Explain *why* in terms of the two ingredients of the MSE, and name the property of the
   data that decides.
""",
    answer_md=r"""
1. Once the interval is computed, nothing is random: the true rate is a fixed number that is either inside
   $[2.1\%, 3.4\%]$ or not. The 95% is a property of the *procedure* — over many repeated samples, 95% of the
   intervals it produces contain the truth — so the honest sentence is "this interval was produced by a method
   that captures the true rate 95% of the time". To make a probability statement about the rate itself you
   need a prior distribution for it; the Bayesian credible interval then has exactly the manager's reading
   (Module 11), and with a flat prior and this much data it is numerically almost the same interval.
2. The colleague used the Wald interval $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$ with $\hat p = 0$, which has
   zero width — it plugs the estimate into its own standard error, and the estimate happens to be at the
   boundary. Twenty crash-free sessions are perfectly consistent with a crash rate of several percent: the
   Wilson interval gives $[0, z^2/(n + z^2)] = [0, 0.161]$, the rule of three gives an upper bound of about
   $3/20 = 0.15$, and the exact (Clopper–Pearson) 95% one-sided bound is $1 - 0.05^{1/20} \approx 0.139$.
3. Both estimators are unbiased for a symmetric population, so the MSE *is* the variance. The variance of the
   mean is $\sigma^2/n$, driven by the tails (large deviations are squared); the variance of the median is
   $\approx 1/(4f(m)^2 n)$, driven only by how much probability mass sits near the centre. Light tails and a
   flat centre (normal) favour the mean; heavy tails with a sharp peak (Laplace, $t_3$) favour the median,
   because the extra tail mass inflates $\sigma^2$ but not $f(m)$. For the Cauchy distribution $\sigma^2$ is
   infinite and the mean does not converge at all, while the median behaves perfectly well. The deciding
   property is the tail weight of the data-generating distribution.
""",
)

# ---------------------------------------------------------------- 9. MSE decomposition (stretch)
SOL_9 = r"""
def mse_decomposition(estimates, theta):
    estimates = np.asarray(estimates, dtype=float)
    bias = float(estimates.mean() - theta)
    var = float(estimates.var())                      # divisor n_reps: the Monte Carlo variance of the estimator
    mse = float(((estimates - theta) ** 2).mean())
    return {"bias": bias, "var": var, "mse": mse}

def compare_variance_estimators(n, n_reps, rng, sigma2=1.0):
    X = rng.normal(0.0, np.sqrt(sigma2), size=(n_reps, n))
    ss = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    out = {}
    for name, divisor in (("n-1", n - 1), ("n", n), ("n+1", n + 1)):
        out[name] = mse_decomposition(ss / divisor, sigma2)
    return out
"""
ns9 = run(SOL_9, r"""
res = compare_variance_estimators(10, 50_000, np.random.default_rng(7))
dec = mse_decomposition(np.array([1.0, 2.0, 4.0]), 2.0)
""")
r9 = ns9["res"]
assert 0.205 < r9["n-1"]["mse"] < 0.24 and 0.17 < r9["n+1"]["mse"] < 0.195 and -0.115 < r9["n"]["bias"] < -0.085, r9
assert r9["n-1"]["mse"] > r9["n"]["mse"] > r9["n+1"]["mse"], r9
ex.add(
    title="Bias, variance and MSE by simulation", difficulty=3,
    prompt_md=r"""
1. Implement `mse_decomposition(estimates, theta)`: given a 1-D array of estimates (many realisations of
   one estimator) and the true value `theta`, return the dict
   `{"bias": mean(estimates) - theta, "var": var(estimates), "mse": mean((estimates - theta)^2)}`
   (use `ddof=0` for the variance, so that the identity $\text{mse} = \text{bias}^2 + \text{var}$ holds exactly).
2. Implement `compare_variance_estimators(n, n_reps, rng, sigma2=1.0)`: draw `n_reps` samples of size `n`
   from $\mathcal N(0, \sigma^2)$ with `rng.normal(0.0, np.sqrt(sigma2), size=(n_reps, n))`, compute the sum of
   squared deviations from each sample's mean, and return a dict with keys `"n-1"`, `"n"`, `"n+1"` mapping to
   the `mse_decomposition` of $\hat\sigma^2 = \text{SS}/\text{divisor}$ against `sigma2`.

With $n = 10$ the guide's table predicts MSEs of about 0.222, 0.190 and 0.182 and biases $0, -0.1, -0.182$.
Which estimator would you use if MSE is what you care about?
""",
    starter=r"""
def mse_decomposition(estimates, theta):
    # YOUR CODE HERE
    return None

def compare_variance_estimators(n, n_reps, rng, sigma2=1.0):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_9,
    check=r"""
try:
    dec = mse_decomposition(np.array([1.0, 2.0, 4.0]), 2.0)
    ok_keys = set(dec) == {"bias", "var", "mse"}
except Exception:
    dec, ok_keys = None, False
check('returns a dict with keys bias, var, mse', ok_keys)
check_close('bias of [1, 2, 4] about theta = 2 is 1/3', (dec["bias"] if ok_keys else None), 1 / 3)
check_close('mse = bias^2 + var exactly', (dec["mse"] if ok_keys else None), (dec["bias"] ** 2 + dec["var"] if ok_keys else None), tol=1e-12)
try:
    res = compare_variance_estimators(10, 50_000, np.random.default_rng(7))
    m1, m0, m2 = res["n-1"]["mse"], res["n"]["mse"], res["n+1"]["mse"]
    b1, b0, b2 = res["n-1"]["bias"], res["n"]["bias"], res["n+1"]["bias"]
except Exception:
    m1 = m0 = m2 = b1 = b0 = b2 = None
check('n-1: bias about 0', b1 is not None and abs(b1) < 0.015)
check('n: bias about -0.1', b0 is not None and -0.115 < b0 < -0.085)
check('n-1: MSE about 0.222', m1 is not None and 0.205 < m1 < 0.24)
check('n+1: MSE about 0.182', m2 is not None and 0.17 < m2 < 0.195)
check('MSE ordering n-1 > n > n+1 (the unbiased estimator loses)', m1 is not None and m1 > m0 > m2)
""",
    explanation_md=r"""
`mse_decomposition` is the identity $\mathbb E[(\hat\theta - \theta)^2] = \operatorname{Var}(\hat\theta) +
(\mathbb E\hat\theta - \theta)^2$ evaluated on Monte Carlo draws; with the population-style variance
(`ddof=0`) it holds to machine precision, with `ddof=1` only approximately. For the variance estimators,
$\text{SS} = \sum(X_i - \bar X)^2 \sim \sigma^2\chi^2_{n-1}$ has mean $(n-1)\sigma^2$ and variance
$2(n-1)\sigma^4$, so $\text{SS}/c$ has bias $(n-1)/c - 1$ and variance $2(n-1)/c^2$: dividing by $n + 1$
minimises the MSE within this family (0.182 vs 0.222 for the unbiased choice at $n = 10$). If MSE is the goal,
unbiasedness is not sacred — shrinking toward zero pays. The common wrong turn is to compute the sum of
squares around the *true* mean 0 instead of the sample mean, which removes the bias entirely (and the lesson).
""",
)

# ---------------------------------------------------------------- 10. paired comparison (stretch)
SOL_10 = r"""
def paired_accuracy_diff(correct_a, correct_b, conf=0.95):
    a = np.asarray(correct_a, dtype=bool); b = np.asarray(correct_b, dtype=bool); n = len(a)
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
    d = (n10 - n01) / n
    se_paired = float(np.sqrt(((n10 + n01) / n - d ** 2) / n))
    pa, pb = a.mean(), b.mean()
    se_unpaired = float(np.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n))
    z = stats.norm.ppf(0.5 + conf / 2)
    return {"d": float(d), "n10": n10, "n01": n01, "se_paired": se_paired, "se_unpaired": se_unpaired,
            "ci": (float(d - z * se_paired), float(d + z * se_paired))}
"""
DATA_10 = r"""
# per-example correctness of two classifiers on the same 1000 test examples
correct_a = np.r_[np.ones(700), np.ones(90), np.zeros(70), np.zeros(140)].astype(bool)
correct_b = np.r_[np.ones(700), np.zeros(90), np.ones(70), np.zeros(140)].astype(bool)
# a second pair that disagrees on far fewer examples
correct_c = np.r_[np.ones(700), np.ones(15), np.zeros(3), np.zeros(282)].astype(bool)
correct_d = np.r_[np.ones(700), np.zeros(15), np.ones(3), np.zeros(282)].astype(bool)
"""
ns10 = run(SOL_10, DATA_10 + r"""
r = paired_accuracy_diff(correct_a, correct_b)
r2 = paired_accuracy_diff(correct_c, correct_d)
""")
r10, r10b = ns10["r"], ns10["r2"]
assert_hash_margin(r10["se_paired"], 4, "se_paired")
assert_hash_margin(r10["se_unpaired"], 4, "se_unpaired")
assert_hash_margin(r10["ci"][0], 3, "ci lo")
assert_hash_margin(r10["ci"][1], 3, "ci hi")
assert_hash_margin(r10b["se_paired"], 4, "se_paired 2")
ex.add(
    title="Comparing two classifiers on the same test set", difficulty=3,
    prompt_md=r"""
Two classifiers are scored on the same $n$ test examples; `correct_a[i]` and `correct_b[i]` say whether
each got example $i$ right. Implement `paired_accuracy_diff(correct_a, correct_b, conf=0.95)` returning a
dict with

* `"n10"`: the number of examples A gets right and B wrong; `"n01"`: the reverse;
* `"d"`: the accuracy difference $\hat d = (n_{10} - n_{01})/n$;
* `"se_paired"` $= \sqrt{\frac1n\big(\frac{n_{10} + n_{01}}{n} - \hat d^{\,2}\big)}$ — the standard error of the
  mean of the per-example differences $c^A_i - c^B_i \in \{-1, 0, 1\}$ (plug-in variance, divisor $n$);
* `"se_unpaired"` $= \sqrt{\hat a_A(1 - \hat a_A)/n + \hat a_B(1 - \hat a_B)/n}$ — what you would (wrongly) get by
  treating the two accuracies as independent;
* `"ci"`: the tuple $\hat d \pm z_{\alpha/2}\,\text{se}_{\text{paired}}$.

Run it on `(correct_a, correct_b)` and on `(correct_c, correct_d)`. Both pairs have accuracies about 1–2
points apart; which difference is statistically clear, and why?
""",
    starter=DATA_10 + r"""

def paired_accuracy_diff(correct_a, correct_b, conf=0.95):
    # YOUR CODE HERE
    return None
""",
    solution=DATA_10 + "\n" + SOL_10,
    check=rf"""
try:
    r = paired_accuracy_diff(correct_a, correct_b)
    d, sep, seu, ci, n10, n01 = r["d"], r["se_paired"], r["se_unpaired"], r["ci"], r["n10"], r["n01"]
except Exception:
    d = sep = seu = ci = n10 = n01 = None
check('n10 = 90 and n01 = 70', n10 == 90 and n01 == 70)
check_close('d = 0.02', d, 0.02)
check_hash('hidden: paired SE', sep, '{sha_of(r10["se_paired"], decimals=4)}', decimals=4)
check_hash('hidden: unpaired SE', seu, '{sha_of(r10["se_unpaired"], decimals=4)}', decimals=4)
check('paired SE is smaller than unpaired SE', sep is not None and seu is not None and sep < seu)
check_hash('hidden: lower CI bound', (ci[0] if ci is not None else None), '{sha_of(r10["ci"][0], decimals=3)}', decimals=3)
check_hash('hidden: upper CI bound', (ci[1] if ci is not None else None), '{sha_of(r10["ci"][1], decimals=3)}', decimals=3)
try:
    r2 = paired_accuracy_diff(correct_c, correct_d)
    sep2, ci2 = r2["se_paired"], r2["ci"]
except Exception:
    sep2 = ci2 = None
check_hash('hidden: paired SE for the second pair', sep2, '{sha_of(r10b["se_paired"], decimals=4)}', decimals=4)
check('second pair: the interval excludes 0 (a clear difference despite a smaller d)', ci2 is not None and ci2[0] > 0)
check('first pair: the interval includes 0 (inconclusive)', ci is not None and ci[0] < 0 < ci[1])
""",
    explanation_md=r"""
The per-example difference $d_i = c^A_i - c^B_i$ is $+1$ on the $n_{10}$ examples, $-1$ on the $n_{01}$, and
$0$ wherever the models agree, so its mean is $\hat d$ and its plug-in variance is
$\frac{n_{10} + n_{01}}{n} - \hat d^{\,2}$; the standard error of the mean follows by dividing by $n$ and
taking the root. Only the disagreements contribute. For the first pair ($90$ vs $70$ of $1000$) the paired SE is
0.0126 versus 0.0185 unpaired, and the interval $[-0.005, 0.045]$ still includes zero. The second pair differs
by only 1.2 points, but the models disagree on just 18 examples: the paired SE is 0.0042 and the interval
$[0.004, 0.020]$ excludes zero — a smaller difference, yet a clearer one. The unpaired formula cannot see this
because it ignores that both models are graded on the same questions. The common bug is to divide the
variance by $n - 1$ or to compute `se_paired` from the two marginal accuracies.
""",
)

# ---------------------------------------------------------------- 11. mini-project (★★★)
PROJECT_CODE = r"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(2024)
X_all, y_all = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.5, random_state=0, stratify=y_all)
n = len(y_te)
mu_tr, sd_tr = X_tr.mean(0), X_tr.std(0)
Z_tr, Z_te = (X_tr - mu_tr) / sd_tr, (X_te - mu_tr) / sd_tr

# model A: nearest centroid on standardised features; model B: logistic regression
c0, c1 = Z_tr[y_tr == 0].mean(0), Z_tr[y_tr == 1].mean(0)
pred_A = (((Z_te - c1) ** 2).sum(1) < ((Z_te - c0) ** 2).sum(1)).astype(int)
pred_B = LogisticRegression(max_iter=5000).fit(Z_tr, y_tr).predict(Z_te)

def accuracy(pred, y):
    return (pred == y).mean()

def f1_score_(pred, y):
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

def wilson(k, n, z=1.959964):
    p = k / n; z2 = z * z
    c = (p + z2 / (2 * n)) / (1 + z2 / n); h = z / (1 + z2 / n) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return c - h, c + h

B = 5000
idx = rng.integers(0, n, size=(B, n))                      # one set of resamples shared by both models (paired)
print(f"test set: n = {n}, positives (benign = 1): {y_te.sum()}\n")
results = {}
for name, pred in [("A: nearest centroid", pred_A), ("B: logistic regression", pred_B)]:
    acc = accuracy(pred, y_te); f1 = f1_score_(pred, y_te)
    boot_acc = np.array([accuracy(pred[i], y_te[i]) for i in idx])
    boot_f1 = np.array([f1_score_(pred[i], y_te[i]) for i in idx])
    results[name] = (acc, f1, boot_acc, boot_f1)
    k = int((pred == y_te).sum())
    print(f"{name:24s} accuracy {acc:.4f}  SE {np.sqrt(acc*(1-acc)/n):.4f}  Wilson [{wilson(k, n)[0]:.3f}, {wilson(k, n)[1]:.3f}]"
          f"  bootstrap [{np.quantile(boot_acc, 0.025):.3f}, {np.quantile(boot_acc, 0.975):.3f}]")
    print(f"{'':24s} F1       {f1:.4f}  bootstrap SE {boot_f1.std(ddof=1):.4f}  bootstrap [{np.quantile(boot_f1, 0.025):.3f}, {np.quantile(boot_f1, 0.975):.3f}]")

# paired comparison
cA, cB = pred_A == y_te, pred_B == y_te
n10, n01 = int((cA & ~cB).sum()), int((~cA & cB).sum())
d = (n01 - n10) / n                                        # B minus A
se_paired = np.sqrt(((n10 + n01) / n - d ** 2) / n)
se_unpaired = np.sqrt(cA.mean() * (1 - cA.mean()) / n + cB.mean() * (1 - cB.mean()) / n)
boot_d = results["B: logistic regression"][2] - results["A: nearest centroid"][2]
boot_df1 = results["B: logistic regression"][3] - results["A: nearest centroid"][3]
print(f"\ndisagreements: A right/B wrong {n10}, B right/A wrong {n01}, agree on {n - n10 - n01}")
print(f"accuracy difference B - A = {d:+.4f}   paired SE {se_paired:.4f}  CI [{d - 1.96*se_paired:+.4f}, {d + 1.96*se_paired:+.4f}]"
      f"   paired bootstrap CI [{np.quantile(boot_d, 0.025):+.4f}, {np.quantile(boot_d, 0.975):+.4f}]   unpaired SE {se_unpaired:.4f}")
print(f"F1 difference B - A = {boot_df1.mean():+.4f}   paired bootstrap CI [{np.quantile(boot_df1, 0.025):+.4f}, {np.quantile(boot_df1, 0.975):+.4f}]")

# how large a test set for +/- 2 points?
z = 1.959964
for name, (acc, _, _, _) in results.items():
    print(f"n for a +/- 2 point 95% interval on the accuracy of {name}: {int(np.ceil(z*z*acc*(1-acc)/0.02**2))}")
var_diff = (n10 + n01) / n - d ** 2
print(f"n for a +/- 2 point interval on the DIFFERENCE (same disagreement rate): {int(np.ceil(z*z*var_diff/0.02**2))}")
print(f"n for a +/- 1 point interval on the difference: {int(np.ceil(z*z*var_diff/0.01**2))}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (metric, j) in zip(axes, [("accuracy", 2), ("F1", 3)]):
    for name, res in results.items():
        ax.hist(res[j], bins=40, alpha=0.6, density=True, label=name)
    ax.set_xlabel(f"bootstrap replicate of test {metric}"); ax.set_ylabel("density"); ax.set_title(f"bootstrap distributions of {metric} (n = {n})"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""
ex.add_project(
    title="Mini-project: honest error bars for two classifiers",
    prompt_md=r"""
Evaluate two classifiers on a held-out half of `load_breast_cancer` and report their performance the way
this module says it should be reported.

**Setup.** Split with `train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)`; standardise
with the training half's mean and standard deviation. Model A: the **nearest-centroid** classifier from the
lesson (assign each test point to the class whose training-set centroid is closer in standardised space).
Model B: `sklearn.linear_model.LogisticRegression(max_iter=5000)` on the standardised features (or any
second classifier you like). Use `rng = np.random.default_rng(2024)` for the bootstrap.

**Deliverables**

1. For each model: test **accuracy** with its formula standard error $\sqrt{\hat a(1-\hat a)/n}$, a Wilson
   95% interval, and a bootstrap 95% percentile interval obtained by resampling the test *examples*
   ($B \ge 2000$).
2. For each model: test **F1 score** (computed by hand from TP, FP, FN) with a bootstrap SE and interval —
   there is no formula for this one.
3. A **paired** comparison: the accuracy difference B − A with the paired SE from the disagreement counts,
   the paired bootstrap interval for the accuracy difference *and* for the F1 difference (use the *same*
   resample indices for both models), and the unpaired SE for contrast.
4. **Sample-size planning**: how many test examples would be needed for a ± 2 point 95% interval on each
   accuracy, and on the difference (assume the observed disagreement rate)?
5. A figure overlaying the bootstrap distributions of the two models' accuracies (and F1), with title and
   axis labels.

Write two or three sentences of conclusion: is B better than A, and what would you need to be sure?
""",
    rubric_md=r"""
* The split, standardisation and both models are implemented correctly, with statistics computed on the training half only.
* Every reported metric carries an interval; the formula SE, Wilson and bootstrap intervals for accuracy agree to within a few tenths of a point.
* F1 is computed by hand and its bootstrap replicates reuse the same resampled indices as the accuracy replicates.
* The paired comparison uses the disagreement counts; the paired SE is smaller than the unpaired one and the paired bootstrap interval agrees with the paired formula.
* The sample-size answers follow from $n = z^2 \hat p(1-\hat p)/e^2$ (and the analogous formula for the difference), rounded up.
* The figure has a title, axis labels and a legend, and the conclusion distinguishes "B is better on this sample" from "B is better".
""",
    solution_sketch_md=r"""
With `random_state=0` the test half has 285 tumours. Nearest centroid reaches about 92% accuracy, logistic
regression about 98%; formula, Wilson and bootstrap intervals for each agree closely (roughly ± 3 points for A,
± 2 for B). F1 is a little above accuracy for both because the positive (benign) class is the majority. The
models disagree on 23 examples, 20 of them in B's favour, so the paired SE of the difference is about
1.7 points — only a little below the unpaired 1.8 here, because A's errors are mostly examples B gets
right, so the two models' mistakes are not strongly correlated — and the paired interval for B − A,
roughly $[+2.7, +9.2]$ points, excludes zero: B is better on this test set *and* the difference is
statistically clear. For a ± 2 point interval on B's accuracy you need about 200 examples (its accuracy is
high, so $p(1-p)$ is small); for A about 710; and for the *difference* about 740 at the observed
disagreement rate. The remaining caveat is not statistical: the 569 tumours are a
convenience sample from one institution, so the intervals describe sampling error, not generalisation to
other scanners or populations.
""",
    solution_code=PROJECT_CODE,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print(f"exercises: {len(ex.items)} items — {kinds.count('code')} coding, {kinds.count('concept')} conceptual, {kinds.count('project')} project")
print("difficulties:", [it["difficulty"] for it in ex.items])
