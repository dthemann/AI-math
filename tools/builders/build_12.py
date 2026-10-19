"""
build_12.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for

    part-4-statistics/12-hypothesis-testing-and-experiments

Run from the repository root:

    python3 _authoring/builders/build_12.py
    python3 tools/run_notebooks.py part-4-statistics/12-hypothesis-testing-and-experiments

All hidden answers are computed here by *executing* the reference solutions and hashing
the result with `sha_of`, so the exercises notebook never contains the answers in clear text.
statsmodels is not available in the course environment, so every multiple-comparison
procedure (Bonferroni, Holm, Benjamini–Hochberg), McNemar's test and the Nadeau–Bengio
correction are implemented directly in NumPy / scipy.stats.
"""
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import ExerciseSet, Notebook, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-4-statistics/12-hypothesis-testing-and-experiments"


def safe_sha(value, decimals=3):
    """sha_of, but refuse values whose rounding could flip from floating-point noise."""
    arr = np.atleast_1d(np.asarray(value, dtype=float)) if not isinstance(value, (str, bool)) else None
    if arr is not None and np.issubdtype(arr.dtype, np.floating):
        for x in arr.ravel():
            frac = abs(x * 10 ** decimals) % 1.0
            if abs(frac - 0.5) < 1e-4 and not float(x).is_integer():
                raise ValueError(f"value {x!r} is too close to a rounding boundary at {decimals} decimals")
    return sha_of(value, decimals)


# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 12 — Hypothesis Testing and Experiments")

nb.md(r"""
This notebook is the runnable companion to the concept guide: read [README.md](README.md) first,
or keep it open alongside. Every section below mirrors a numbered section of the guide. We test a
coin exactly and approximately, watch p-values come out uniform under the null, draw power curves and
check them by simulation, build Welch's and the paired t-test from scratch, do a chi-square test by hand,
run a permutation test on skewed revenue data, verify that confidence intervals and tests are the same
thing, implement Bonferroni, Holm and Benjamini–Hochberg and measure what they control, see what
randomisation and blocking buy in an experiment, quantify how peeking inflates false positives, and
finally compare two classifiers properly with McNemar's test and a corrected cross-validation test.

Everything is plain NumPy first; `scipy.stats` and scikit-learn appear afterwards to confirm the hand
computation.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy import stats
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
""")

# ---------------------------------------------------------------- 1. the logic of a test
nb.md(r"""
## 1. The logic of a test: is the coin fair after 60 heads in 100 flips?

$H_0: \theta = 0.5$ against $H_1: \theta \ne 0.5$, test statistic $X$ = number of heads. Under $H_0$,
$X \sim \text{Binomial}(100, 0.5)$, so the two-sided p-value is the probability of a count at least as far
from 50 as the observed 60, computed *assuming the coin is fair*:

$$p = P_{H_0}(X \ge 60) + P_{H_0}(X \le 40) = 2\sum_{k=60}^{100}\binom{100}{k}2^{-100}.$$

We compute it exactly from the PMF, then with the normal approximation (with and without the continuity
correction), and finally we find the exact level-0.05 rejection region of this discrete test.
""")

nb.code(r"""
n, k_obs, theta0 = 100, 60, 0.5
ks = np.arange(n + 1)
pmf0 = np.array([comb(n, k) * theta0**k * (1 - theta0)**(n - k) for k in ks])   # null PMF, by hand

p_upper = pmf0[ks >= k_obs].sum()
p_lower = pmf0[ks <= n - k_obs].sum()           # mirror-image tail (the binomial(100, .5) is symmetric)
p_exact = p_upper + p_lower
print(f"P(X >= 60 | fair) = {p_upper:.4f}   two-sided exact p-value = {p_exact:.4f}")
print(f"scipy.stats.binomtest : {stats.binomtest(k_obs, n, theta0).pvalue:.4f}")

mu, sd = n * theta0, np.sqrt(n * theta0 * (1 - theta0))
z_plain = (k_obs - mu) / sd
z_cc = (abs(k_obs - mu) - 0.5) / sd                              # continuity correction
print(f"\nnormal approximation: z = {z_plain:.2f} -> p = {2 * stats.norm.sf(abs(z_plain)):.4f}")
print(f"with continuity correction: z = {z_cc:.2f} -> p = {2 * stats.norm.sf(z_cc):.4f}   (exact: {p_exact:.4f})")

# the exact level-0.05 rejection region: the smallest c such that P(|X - 50| >= c) <= 0.05
for c in range(8, 13):
    size = pmf0[np.abs(ks - mu) >= c].sum()
    flag = "  <- first region with size <= 0.05" if size <= 0.05 and pmf0[np.abs(ks - mu) >= c - 1].sum() > 0.05 else ""
    print(f"reject if |X - 50| >= {c:2d}:  size = {size:.4f}{flag}")

colors = np.where(np.abs(ks - mu) >= k_obs - mu, "C3", "C0")
plt.bar(ks, pmf0, color=colors, width=0.9)
plt.axvline(k_obs, color="k", ls="--", label="observed X = 60")
plt.xlim(25, 75); plt.xlabel("number of heads X"); plt.ylabel("P(X = k) under H0: fair coin")
plt.title("Null distribution; red bars = outcomes at least as extreme as 60 (p = 0.057)")
plt.legend(); plt.show()
""")

nb.md(r"""
Notice three things. The exact two-sided p-value is $0.057$: a fair coin produces a split at least this
lopsided about once in 18 blocks of 100 flips, so at $\alpha = 0.05$ we do **not** reject. The crude normal
approximation says $0.046$ and *would* reject — approximations matter exactly at the boundary, and the
continuity correction repairs most of the damage. And because $X$ is discrete, no region has size exactly
$0.05$: the level-0.05 test is "reject if $X \le 39$ or $X \ge 61$", whose true size is only $0.035$.

### p-values are uniform under the null

If $H_0$ is true and the statistic is continuous, the p-value is $\text{Uniform}(0, 1)$ — so exactly a fraction
$\alpha$ of true nulls are rejected. Under $H_1$ the p-values pile up near 0; the mass below $\alpha$ is the power.
We run 20,000 Welch t-tests on two samples of 20 drawn from the *same* normal distribution, then 20,000
more where the second sample is shifted by half a standard deviation.
""")

nb.code(r"""
n_rep, n_s = 20_000, 20
x = rng.normal(0.0, 1.0, size=(n_rep, n_s))
y_null = rng.normal(0.0, 1.0, size=(n_rep, n_s))
y_alt = rng.normal(0.5, 1.0, size=(n_rep, n_s))
p_null = stats.ttest_ind(x, y_null, axis=1, equal_var=False).pvalue
p_alt = stats.ttest_ind(x, y_alt, axis=1, equal_var=False).pvalue

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
ax1.hist(p_null, bins=20, range=(0, 1), density=True, alpha=0.8); ax1.axhline(1, color="k", ls="--")
ax1.set_title("H0 true: p-values are Uniform(0, 1)"); ax1.set_xlabel("p-value"); ax1.set_ylabel("density")
ax2.hist(p_alt, bins=20, range=(0, 1), density=True, alpha=0.8, color="C3")
ax2.set_title("H1 true (shift 0.5 sd): p-values pile up near 0"); ax2.set_xlabel("p-value")
plt.tight_layout(); plt.show()

for a in (0.01, 0.05, 0.10):
    print(f"alpha = {a:.2f}:  fraction rejected under H0 = {(p_null <= a).mean():.4f}   under H1 = {(p_alt <= a).mean():.4f}")
""")

nb.md(r"""
Under $H_0$ the histogram is flat and the fraction of p-values below $0.05$ is $0.05$ — the false-alarm rate is
built into the procedure, whatever the data look like. Under $H_1$ the fraction below $0.05$ is the power,
here only about $0.34$: with 20 observations per group a half-standard-deviation effect is missed two times
in three. That number is the subject of the next section.

**Try it yourself:** change the shift `0.5` to `1.0` and the group size `n_s` to 40 — which change raises the
power more?
""")

# ---------------------------------------------------------------- 2. errors and power
nb.md(r"""
## 2. Errors and power

For the two-sample z-test with known $\sigma$, $n$ per arm and a two-sided level $\alpha$, the difference of
means has standard error $\sigma\sqrt{2/n}$ and

$$1 - \beta = \Phi\!\left(\frac{\delta}{\sigma\sqrt{2/n}} - z_{1-\alpha/2}\right) + \Phi\!\left(-\frac{\delta}{\sigma\sqrt{2/n}} - z_{1-\alpha/2}\right),
\qquad n_{\text{per arm}} = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2\sigma^2}{\delta^2}.$$

We implement both, draw power against $n$ for three effect sizes, and check the curve by simulating the
test thousands of times.
""")

nb.code(r"""
def power_two_sample_z(delta, sigma, n, alpha=0.05):
    se = sigma * np.sqrt(2.0 / n)
    zc = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.cdf(-zc - delta / se) + stats.norm.sf(zc - delta / se)

def n_per_arm(delta, sigma, power=0.8, alpha=0.05):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * (za + zb)**2 * sigma**2 / delta**2))

def mde(sigma, n, power=0.8, alpha=0.05):
    return (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sigma * np.sqrt(2.0 / n)

sigma, delta = 10.0, 4.0
for n in (25, 50, 100, 200):
    print(f"n = {n:3d} per arm: analytic power = {power_two_sample_z(delta, sigma, n):.3f}")
print(f"\nrequired n per arm for 80% power: {n_per_arm(delta, sigma)}   (formula gives {2 * (1.96 + 0.8416)**2 * sigma**2 / delta**2:.1f})")
print(f"minimum detectable effect at n = 50, 80% power: {mde(sigma, 50):.2f} minutes")

# Monte Carlo check: simulate the experiment many times and count rejections
def simulated_power(delta, sigma, n, reps=5_000, alpha=0.05):
    a = rng.normal(0.0, sigma, size=(reps, n)); b = rng.normal(delta, sigma, size=(reps, n))
    z = (b.mean(axis=1) - a.mean(axis=1)) / (sigma * np.sqrt(2.0 / n))
    return (np.abs(z) > stats.norm.ppf(1 - alpha / 2)).mean()

ns = np.arange(10, 301, 5)
for d, c in zip((2.0, 4.0, 6.0), ("C0", "C1", "C2")):
    plt.plot(ns, power_two_sample_z(d, sigma, ns), color=c, label=f"delta = {d:.0f} (analytic)")
sim_ns = np.array([25, 50, 100, 150, 200, 300])
plt.plot(sim_ns, [simulated_power(4.0, sigma, n) for n in sim_ns], "o", color="C1", label="delta = 4, simulated")
plt.axhline(0.8, color="gray", ls=":"); plt.axhline(0.05, color="gray", ls=":")
plt.xlabel("n per arm"); plt.ylabel("power = P(reject H0 | H1)")
plt.title("Power of the two-sample z-test (sigma = 10, alpha = 0.05)")
plt.legend(); plt.show()
""")

nb.md(r"""
The simulated points fall on the analytic curve. Read the plot the way an experimenter does: with
$n = 50$ a 4-minute effect is caught only half the time, and halving the effect to 2 minutes drops the power
at $n = 50$ to $0.17$ — power grows with $\sqrt n$, so a half-size effect needs *four times* the data. The
dotted line at $0.05$ is where every curve starts: at $\delta = 0$ the "power" is just the false-positive rate.

**Try it yourself:** call `n_per_arm(2.0, 10.0)` and `n_per_arm(4.0, 10.0, power=0.9)` — what do halving the
effect and raising the power from 80% to 90% each cost in sample size?
""")

# ---------------------------------------------------------------- 3. t-tests
nb.md(r"""
## 3. Welch's t-test from scratch, and why pairing increases power

$$t = \frac{\bar x_1 - \bar x_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}, \qquad
\nu = \frac{(s_1^2/n_1 + s_2^2/n_2)^2}{\dfrac{(s_1^2/n_1)^2}{n_1-1} + \dfrac{(s_2^2/n_2)^2}{n_2-1}}.$$

We apply it to the cycle-time data of the guide (old process $n_1 = 8$, new process $n_2 = 6$), report
Cohen's $d$ and a 95% confidence interval, and compare with `scipy.stats.ttest_ind(..., equal_var=False)`.
Then the paired design: six machines before and after a change, analysed paired and (wrongly) unpaired.
""")

nb.code(r"""
def welch_t(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1) / n1, y.var(ddof=1) / n2
    t = (x.mean() - y.mean()) / np.sqrt(v1 + v2)
    df = (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p

old = np.array([52., 48, 55, 50, 53, 49, 51, 54])
new = np.array([47., 50, 44, 49, 46, 48])
t, df, p = welch_t(old, new)
print(f"means {old.mean():.2f} vs {new.mean():.2f};  sample variances {old.var(ddof=1):.3f}, {new.var(ddof=1):.3f}")
print(f"Welch by hand : t = {t:.4f}, df = {df:.3f}, p = {p:.5f}")
res = stats.ttest_ind(old, new, equal_var=False)
print(f"scipy         : t = {res.statistic:.4f}, df = {res.df:.3f}, p = {res.pvalue:.5f}")
res_pooled = stats.ttest_ind(old, new, equal_var=True)
print(f"Student pooled: t = {res_pooled.statistic:.4f}, df = {res_pooled.df:.0f}, p = {res_pooled.pvalue:.5f}")

se = np.sqrt(old.var(ddof=1) / len(old) + new.var(ddof=1) / len(new))
tcrit = stats.t.ppf(0.975, df)
diff = old.mean() - new.mean()
s_pooled = np.sqrt(((len(old) - 1) * old.var(ddof=1) + (len(new) - 1) * new.var(ddof=1)) / (len(old) + len(new) - 2))
print(f"\ndifference {diff:.2f} min, 95% CI [{diff - tcrit * se:.2f}, {diff + tcrit * se:.2f}], Cohen's d = {diff / s_pooled:.2f}")

# paired: six machines before/after
before = np.array([31.0, 28.5, 35.2, 30.1, 27.8, 33.4])
after = np.array([29.1, 27.9, 33.0, 29.6, 27.0, 31.5])
d = before - after
t_pair = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
p_pair = 2 * stats.t.sf(abs(t_pair), len(d) - 1)
print(f"\npaired differences {d}: mean {d.mean():.3f}, sd {d.std(ddof=1):.3f}")
print(f"paired t by hand: t = {t_pair:.3f}, df = {len(d) - 1}, p = {p_pair:.4f}    scipy ttest_rel: p = {stats.ttest_rel(before, after).pvalue:.4f}")
print(f"the SAME numbers analysed unpaired (Welch): p = {welch_t(before, after)[2]:.4f}   (before/after correlation = {np.corrcoef(before, after)[0, 1]:.3f})")
""")

nb.md(r"""
Hand computation and scipy agree to every digit, and Student's pooled test is close here because the two
variances happen to be similar. The paired example is the dramatic one: $p = 0.008$ paired, $p = 0.40$
unpaired, from the same twelve numbers. The machines differ from each other by about 2.9 minutes while the
change moves each machine by about 1.3; pairing subtracts the machine effect
($\operatorname{Var}(\bar x - \bar y) = (\sigma_x^2 + \sigma_y^2 - 2\rho\sigma_x\sigma_y)/n$ with $\rho = 0.98$),
the unpaired analysis cannot. The next cell measures that gain as a function of $\rho$.
""")

nb.code(r"""
def paired_vs_unpaired_power(rho, n=15, delta=0.5, reps=4_000, alpha=0.05):
    z1 = rng.standard_normal((reps, n)); z2 = rng.standard_normal((reps, n))
    x = z1
    y = rho * z1 + np.sqrt(1 - rho**2) * z2 + delta          # correlated partner, shifted by delta
    p_paired = stats.ttest_rel(x, y, axis=1).pvalue
    p_unpaired = stats.ttest_ind(x, y, axis=1, equal_var=False).pvalue
    return (p_paired < alpha).mean(), (p_unpaired < alpha).mean()

rhos = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95])
pw = np.array([paired_vs_unpaired_power(r) for r in rhos])
plt.plot(rhos, pw[:, 0], "o-", label="paired t-test")
plt.plot(rhos, pw[:, 1], "s--", label="unpaired Welch t-test")
plt.xlabel("correlation rho between paired measurements"); plt.ylabel("power")
plt.title("n = 15 pairs, effect 0.5 sd: pairing pays off as rho grows"); plt.legend(); plt.show()
for r, (pp, pu) in zip(rhos, pw):
    print(f"rho = {r:.2f}: power paired {pp:.3f}   unpaired {pu:.3f}")
""")

nb.md(r"""
At $\rho = 0$ the two tests are nearly equal (the paired test even loses a little, having $n - 1$ instead of
about $2n - 2$ degrees of freedom); at $\rho = 0.9$ the paired test detects the effect almost always while the
unpaired one still misses it more than half the time. This is why before/after designs, repeated measures,
and model comparisons on *shared* folds or test examples are analysed paired.

**Try it yourself:** set `delta = 0.3` and `n = 30` — does the crossover point move?
""")

# ---------------------------------------------------------------- 4. proportions and chi-square
nb.md(r"""
## 4. Proportions and the chi-square test

Conversions by variant: A 200/4000, B 240/4000. Two-proportion z-test with the pooled rate
$\bar\theta = (x_1 + x_2)/(n_1 + n_2)$, and the same data as a $2 \times 2$ table with Pearson's
$\chi^2 = \sum (O - E)^2/E$, $E_{ij} = R_i C_j / N$. For a $2 \times 2$ table $\chi^2 = z^2$ exactly.
We also run a small-sample table through Fisher's exact test and Yates's correction.
""")

nb.code(r"""
def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pbar = (x1 + x2) / (n1 + n2)
    se0 = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))          # pooled SE, valid under H0
    z = (p2 - p1) / se0
    se_ci = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)      # unpooled SE for the interval
    return z, 2 * stats.norm.sf(abs(z)), (p2 - p1 - 1.96 * se_ci, p2 - p1 + 1.96 * se_ci)

def chi2_independence(table):
    table = np.asarray(table, float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    chi2 = ((table - expected)**2 / expected).sum()
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    return chi2, stats.chi2.sf(chi2, dof), dof, expected

z, p_z, ci = two_prop_z(200, 4000, 240, 4000)
print(f"two-proportion z-test: z = {z:.4f}, p = {p_z:.4f}, 95% CI for the lift [{ci[0]:.5f}, {ci[1]:.5f}]")

table = np.array([[200, 3800], [240, 3760]])
chi2, p_chi, dof, E = chi2_independence(table)
print(f"\nexpected counts under independence:\n{E}")
print(f"chi-square by hand: chi2 = {chi2:.4f}, dof = {dof}, p = {p_chi:.4f}    z^2 = {z**2:.4f}")
sp = stats.chi2_contingency(table, correction=False)
print(f"scipy chi2_contingency(correction=False): chi2 = {sp.statistic:.4f}, p = {sp.pvalue:.4f}")

# small-sample table: nausea 30/200 treated vs 18/200 placebo
small = np.array([[30, 170], [18, 182]])
chi2_s, p_s, _, E_s = chi2_independence(small)
print(f"\nside-effect table: expected = {E_s[0].round(1)} per row;  chi2 = {chi2_s:.3f}, p = {p_s:.4f}")
print(f"with Yates's correction: p = {stats.chi2_contingency(small, correction=True).pvalue:.4f}")
print(f"Fisher's exact test   : p = {stats.fisher_exact(small).pvalue:.4f}")
""")

nb.md(r"""
$z^2 = 3.848 = \chi^2$ to all shown digits, and the p-values coincide: the chi-square test of a $2 \times 2$
table *is* the two-sided two-proportion z-test. The confidence interval for the lift,
$[0.00001, 0.020]$, excludes zero by a hair just as $p = 0.0498$ sits a hair under $0.05$ — Section 6 explains
why these must agree. On the small side-effect table the three p-values ($0.065$, $0.091$, $0.090$) disagree
noticeably; none reaches $0.05$, but the interval for the excess nausea rate stretches up to 12 percentage
points, which is the number a safety review should read.
""")

# ---------------------------------------------------------------- 5. permutation test
nb.md(r"""
## 5. Permutation and rank tests on skewed revenue data

Revenue per user is 95% zeros plus a long right tail. Under $H_0$ (treatment changes nothing) the labels
are arbitrary, so we shuffle them, recompute the difference of means, and read the p-value off the
shuffled distribution:

$$\hat p = \frac{1 + \#\{b : |T^{(b)}| \ge |t_{\text{obs}}|\}}{B + 1}.$$

We then compute the Mann–Whitney $U$ statistic by counting pairs, apply its tie-corrected normal
approximation, and compare both with scipy and with Welch's t-test.
""")

nb.code(r"""
def revenue(n, p_buy, rng):
    buys = rng.random(n) < p_buy
    return np.where(buys, rng.lognormal(mean=np.log(60), sigma=1.0, size=n), 0.0)

n_users = 2_000
rev_a = revenue(n_users, 0.050, rng)        # control: 5% buy
rev_b = revenue(n_users, 0.065, rng)        # treatment: 6.5% buy, same basket distribution
t_obs = rev_b.mean() - rev_a.mean()
print(f"mean revenue per user: A = {rev_a.mean():.2f}, B = {rev_b.mean():.2f}, difference = {t_obs:.2f}")
print(f"share of zeros: {np.mean(rev_a == 0):.3f} / {np.mean(rev_b == 0):.3f};  largest order = {max(rev_a.max(), rev_b.max()):.0f}")

pooled = np.concatenate([rev_a, rev_b])
B = 5_000
perm_diffs = np.empty(B)
for b in range(B):
    perm = rng.permutation(pooled)
    perm_diffs[b] = perm[n_users:].mean() - perm[:n_users].mean()
p_perm = (1 + np.sum(np.abs(perm_diffs) >= abs(t_obs))) / (B + 1)
print(f"\npermutation test: p = {p_perm:.4f}   (B = {B} shuffles)")
print(f"Welch t-test    : p = {stats.ttest_ind(rev_b, rev_a, equal_var=False).pvalue:.4f}")

plt.hist(perm_diffs, bins=60, alpha=0.8, label="difference of means under shuffled labels")
plt.axvline(t_obs, color="C3", ls="--", label=f"observed difference = {t_obs:.2f}")
plt.axvline(-t_obs, color="C3", ls=":")
plt.xlabel("mean(B) - mean(A) after shuffling"); plt.ylabel("count")
plt.title("Permutation null distribution of the revenue difference"); plt.legend(); plt.show()
""")

nb.code(r"""
def mann_whitney_u(a, b):
    # U = number of (a_i, b_j) pairs with a_i > b_j, ties counting one half
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return gt + 0.5 * eq

def mann_whitney_p(a, b):
    n1, n2 = len(a), len(b); N = n1 + n2
    u1 = mann_whitney_u(a, b)
    u = max(u1, n1 * n2 - u1)                                   # two-sided: the larger of U_A, U_B
    mu = n1 * n2 / 2
    _, ties = np.unique(np.concatenate([a, b]), return_counts=True)
    var = n1 * n2 / 12 * ((N + 1) - (ties**3 - ties).sum() / (N * (N - 1)))   # tie-corrected variance
    z = (u - mu - 0.5) / np.sqrt(var)                           # continuity correction
    return u1, 2 * stats.norm.sf(z)

u1, p_mw = mann_whitney_p(rev_b, rev_a)
sp = stats.mannwhitneyu(rev_b, rev_a, alternative="two-sided", method="asymptotic")
print(f"Mann-Whitney by hand: U = {u1:.1f}, p = {p_mw:.4f}    scipy: U = {sp.statistic:.1f}, p = {sp.pvalue:.4f}")
print(f"P(a random B user out-spends a random A user) = U / (n1 n2) = {u1 / (n_users**2):.4f}   (0.5 under H0)")

# a tiny example where the exact permutation distribution can be enumerated
from itertools import combinations
A3, B3 = np.array([12., 15, 11]), np.array([9., 8, 10])
all6 = np.concatenate([A3, B3]); obs = A3.mean() - B3.mean()
diffs = np.array([all6[list(idx)].mean() - np.delete(all6, list(idx)).mean() for idx in combinations(range(6), 3)])
print(f"\n3 vs 3 example: {len(diffs)} splits, observed diff {obs:.2f}, exact two-sided p = {np.mean(np.abs(diffs) >= abs(obs) - 1e-12):.3f}")
print(f"Welch t-test on the same six numbers: p = {stats.ttest_ind(A3, B3, equal_var=False).pvalue:.3f}  (leans on normality of three points)")
""")

nb.md(r"""
Two lessons in one output. First, the treatment *does* raise the purchase rate (6.5% against 5%), yet with
2,000 users per arm neither test comes close to significance: a handful of large orders dominates the means
(one order of about 1,000 against an average of 5–6), so the standard error of the difference is huge. This is
the everyday reality of revenue metrics — they need far more users than conversion metrics, or a more robust
statistic. Second, the permutation p-value and Welch's p-value are close here because the central limit
theorem has had 4,000 observations to work with even for this ugly distribution; the permutation test's
advantage is that it would remain exact with 50 users per arm, or for a trimmed mean or a median, where the
t-test's approximation fails. Mann–Whitney asks a different question — does a random B user out-spend a random
A user more than half the time? — and the hand computation matches scipy's once ties and the continuity
correction are handled. The 3-vs-3 example shows the granularity of exact tests: the smallest possible
two-sided p-value is $2/20 = 0.10$, so no result from three observations per group can be "significant at 5%".

**Try it yourself:** replace the statistic in the permutation loop by the difference of *medians* or of
20%-trimmed means (`stats.trim_mean(x, 0.2)`) — the code does not change otherwise. That is the point.
""")

# ---------------------------------------------------------------- 6. CI–test duality, TOST
nb.md(r"""
## 6. Confidence intervals and tests are the same thing

A level-$\alpha$ test of $\theta = \theta_0$ rejects exactly when $\theta_0$ lies outside the $(1 - \alpha)$
confidence interval built from the same statistic. We verify this on 5,000 simulated two-sample
experiments. Then the reverse question — *are two things equivalent?* — via Schuirmann's two one-sided
tests (TOST): equivalence within $\pm\Delta$ at level $\alpha$ holds iff the $(1 - 2\alpha)$ interval lies
inside $(-\Delta, \Delta)$.
""")

nb.code(r"""
reps, n = 5_000, 30
x = rng.normal(0.0, 1.0, size=(reps, n)); y = rng.normal(0.3, 1.5, size=(reps, n))
res = stats.ttest_ind(x, y, axis=1, equal_var=False)
diff = x.mean(axis=1) - y.mean(axis=1)
se = np.sqrt(x.var(axis=1, ddof=1) / n + y.var(axis=1, ddof=1) / n)
tcrit = stats.t.ppf(0.975, res.df)
lo, hi = diff - tcrit * se, diff + tcrit * se
reject = res.pvalue < 0.05
excludes_zero = (lo > 0) | (hi < 0)
print(f"experiments where test and interval agree: {np.mean(reject == excludes_zero):.4f}   (rejections: {reject.mean():.3f})")
print(f"interval covers the true difference -0.3 in {np.mean((lo <= -0.3) & (-0.3 <= hi)):.4f} of experiments")

def tost(diff, se, margin, alpha=0.05):
    # H0: theta <= -margin  OR  theta >= margin ; reject both to conclude equivalence
    p_lower = stats.norm.sf((diff + margin) / se)
    p_upper = stats.norm.sf((margin - diff) / se)
    zc = stats.norm.ppf(1 - alpha)
    ci90 = (diff - zc * se, diff + zc * se)
    return max(p_lower, p_upper), (p_lower < alpha) and (p_upper < alpha), ci90

for se_ in (0.4, 0.8, 1.2, 2.0):
    p_tost, equiv, ci90 = tost(0.5, se_, margin=2.0)
    p_classic = 2 * stats.norm.sf(abs(0.5 / se_))
    print(f"diff 0.5, SE {se_:.1f}: classic p = {p_classic:.3f} (never rejects 'no difference');  TOST p = {p_tost:.4f}, "
          f"90% CI [{ci90[0]:+.2f}, {ci90[1]:+.2f}] -> equivalent within +/-2: {equiv}")

ses = np.array([0.4, 0.8, 1.2, 2.0])
plt.axhspan(-2, 2, color="C2", alpha=0.15, label="equivalence margin +/-2")
for i, se_ in enumerate(ses):
    _, equiv, ci90 = tost(0.5, se_, 2.0)
    plt.errorbar(i, 0.5, yerr=[[0.5 - ci90[0]], [ci90[1] - 0.5]], fmt="o", capsize=5, color="C2" if equiv else "C3")
plt.axhline(0, color="k", lw=0.8)
plt.xticks(range(len(ses)), [f"SE = {s}" for s in ses]); plt.ylabel("difference (90% CI)")
plt.title("TOST: equivalence (green) iff the 90% CI fits inside the margin"); plt.legend(loc="upper left"); plt.show()
""")

nb.md(r"""
Test and interval agree in 100% of experiments, because they are two readings of one inequality. The TOST
rows show why "not significant" is not "equivalent": all four classic tests fail to reject the null of no
difference, yet only the two precise studies demonstrate equivalence within $\pm 2$; the imprecise ones are
simply inconclusive. A generic drug's bioequivalence, or a quantised model's "no loss of accuracy", must be
shown this way, with the margin fixed in advance.
""")

# ---------------------------------------------------------------- 7. multiple comparisons
nb.md(r"""
## 7. Multiple comparisons: Bonferroni, Holm and Benjamini–Hochberg

With $m$ independent true nulls tested at $\alpha$, $P(\text{at least one false rejection}) = 1 - (1 - \alpha)^m$.
We confirm this by simulation, implement the three standard corrections as *adjusted p-values*
(reject when $\tilde p \le \alpha$), apply them to the ten p-values of the guide, and then measure what each
procedure actually controls — FWER, FDR and power — on a simulated screen of 100 hypotheses of which 20 are
real.

* Bonferroni: $\tilde p_i = \min(m\,p_i, 1)$.
* Holm (step-down): sort; $\tilde p_{(i)} = \min\big(1, \max_{j \le i}(m - j + 1)\,p_{(j)}\big)$.
* Benjamini–Hochberg (step-up): sort; $\tilde p_{(i)} = \min\big(1, \min_{j \ge i} \tfrac{m}{j}\,p_{(j)}\big)$.
""")

nb.code(r"""
def bonferroni_adjust(p):
    p = np.asarray(p, float)
    return np.minimum(p * len(p), 1.0)

def holm_adjust(p):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p)
    scaled = p[order] * (m - np.arange(m))                       # (m - i + 1) p_(i), i = 1..m
    adj_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)  # enforce monotonicity downwards
    adj = np.empty(m); adj[order] = adj_sorted
    return adj

def bh_adjust(p):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)                  # (m / i) p_(i)
    adj_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)   # enforce monotonicity upwards
    adj = np.empty(m); adj[order] = adj_sorted
    return adj

# FWER blow-up under the global null
for m in (1, 2, 5, 10, 20, 50, 100):
    pvals = rng.random((20_000, m))
    print(f"m = {m:3d}: simulated P(any p < 0.05) = {(pvals.min(axis=1) < 0.05).mean():.3f}   formula 1 - 0.95^m = {1 - 0.95**m:.3f}")

p10 = np.array([0.001, 0.011, 0.012, 0.019, 0.040, 0.045, 0.200, 0.350, 0.600, 0.900])
print(f"\n{'p':>8s} {'Bonferroni':>11s} {'Holm':>8s} {'BH':>8s}")
for p_, b_, h_, q_ in zip(p10, bonferroni_adjust(p10), holm_adjust(p10), bh_adjust(p10)):
    print(f"{p_:8.3f} {b_:11.4f} {h_:8.4f} {q_:8.4f}")
print("rejections at 0.05 -> uncorrected:", (p10 <= 0.05).sum(), " Bonferroni:", (bonferroni_adjust(p10) <= 0.05).sum(),
      " Holm:", (holm_adjust(p10) <= 0.05).sum(), " BH:", (bh_adjust(p10) <= 0.05).sum())
""")

nb.code(r"""
# what does each procedure control? 100 hypotheses, 20 real effects (z-shift 3), 2000 repetitions
m, m1, reps = 100, 20, 2_000
is_alt = np.zeros(m, bool); is_alt[:m1] = True
methods = {"uncorrected": lambda p: p, "Bonferroni": bonferroni_adjust, "Holm": holm_adjust, "BH (q = 0.05)": bh_adjust}
stats_ = {k: dict(fwer=0.0, fdr=0.0, power=0.0) for k in methods}
for _ in range(reps):
    z = rng.standard_normal(m) + 3.0 * is_alt
    p = 2 * stats.norm.sf(np.abs(z))
    for name, adjust in methods.items():
        rej = adjust(p) <= 0.05
        V, R = np.sum(rej & ~is_alt), rej.sum()
        stats_[name]["fwer"] += V >= 1
        stats_[name]["fdr"] += V / R if R else 0.0
        stats_[name]["power"] += np.mean(rej[is_alt])
print(f"{'method':14s} {'FWER':>7s} {'FDR':>7s} {'power':>7s}")
for name, s in stats_.items():
    print(f"{name:14s} {s['fwer'] / reps:7.3f} {s['fdr'] / reps:7.3f} {s['power'] / reps:7.3f}")

ms = np.arange(1, 101)
plt.plot(ms, 1 - 0.95**ms, label="FWER of uncorrected tests, 1 - 0.95^m")
plt.plot(ms, 1 - (1 - 0.05 / ms)**ms, label="FWER with Bonferroni (alpha / m)")
plt.axhline(0.05, color="gray", ls=":")
plt.xlabel("number of independent tests m"); plt.ylabel("P(at least one false rejection)")
plt.title("Testing many hypotheses without correction guarantees false alarms"); plt.legend(); plt.show()
""")

nb.md(r"""
On the ten p-values Bonferroni and Holm reject only the smallest, while BH rejects four — including
$p = 0.011$, which exceeded its own threshold $0.010$ but is swept in by the step-up rule. The simulation
shows the trade: uncorrected testing has a family-wise error rate near 1 (some false discovery in almost
every screen); Bonferroni and Holm hold it at $0.05$ but find only about 40% of the real effects; BH lets the
FDR be about $0.04$ (close to its guarantee $q\,m_0/m = 0.04$) and in exchange finds about 70% of them. Which
guarantee you want depends on the cost of one false lead versus one missed discovery.

**Try it yourself:** set `m1 = 50` (half the hypotheses real) and re-run — BH's power rises while the FDR
bound $q\,m_0/m$ falls to $0.025$; Bonferroni's power does not move. Why?
""")

# ---------------------------------------------------------------- 8. designing experiments
nb.md(r"""
## 8. Designing experiments: what randomisation and blocking buy

Outcome: minutes on site. Mobile users average 2 minutes, desktop users 6 minutes (sd 1.5 within device),
and the new design adds **+0.5 minutes for everyone**. In the observational roll-out the feature was opted
into mostly by mobile users; in the experiment, assignment is a coin flip. We compare the naive estimates,
then measure how much *stratified* randomisation (blocking on device) tightens the estimate.
""")

nb.code(r"""
N = 20_000
mobile = rng.random(N) < 0.5
base = np.where(mobile, 2.0, 6.0)
true_effect = 0.5

# observational roll-out: mobile users opt in 80% of the time, desktop users 20%
treated_obs = rng.random(N) < np.where(mobile, 0.8, 0.2)
y_obs = base + true_effect * treated_obs + rng.normal(0, 1.5, N)
naive = y_obs[treated_obs].mean() - y_obs[~treated_obs].mean()
print(f"observational: treated mean {y_obs[treated_obs].mean():.2f} vs control {y_obs[~treated_obs].mean():.2f} -> naive effect {naive:+.2f} (truth {true_effect:+.2f})")
for dev, mask in (("mobile", mobile), ("desktop", ~mobile)):
    eff = y_obs[mask & treated_obs].mean() - y_obs[mask & ~treated_obs].mean()
    print(f"   within {dev:7s}: effect {eff:+.2f}   (treated share {treated_obs[mask].mean():.2f})")

# randomised experiment on the same population
treated_rct = rng.random(N) < 0.5
y_rct = base + true_effect * treated_rct + rng.normal(0, 1.5, N)
print(f"randomised   : effect {y_rct[treated_rct].mean() - y_rct[~treated_rct].mean():+.2f}   (mobile share treated {mobile[treated_rct].mean():.3f} vs control {mobile[~treated_rct].mean():.3f})")

# blocking: simple vs stratified randomisation, n = 400 users, 2000 replications
def one_experiment(stratified, n=400):
    mob = rng.random(n) < 0.5
    if stratified:
        t = np.zeros(n, bool)
        for mask in (mob, ~mob):                      # exactly half of each device group treated
            idx = np.flatnonzero(mask); t[rng.permutation(idx)[:len(idx) // 2]] = True
    else:
        t = rng.random(n) < 0.5
    y = np.where(mob, 2.0, 6.0) + true_effect * t + rng.normal(0, 1.5, n)
    return y[t].mean() - y[~t].mean()

est_simple = np.array([one_experiment(False) for _ in range(2_000)])
est_strat = np.array([one_experiment(True) for _ in range(2_000)])
print(f"\nestimate of the effect over 2000 experiments (n = 400):")
print(f"   simple randomisation    : mean {est_simple.mean():.3f}, sd {est_simple.std():.3f}")
print(f"   stratified by device    : mean {est_strat.mean():.3f}, sd {est_strat.std():.3f}")
print(f"   theory: sd simple = sqrt(2 (1.5^2 + 2^2) / 200) = {np.sqrt(2 * (1.5**2 + 2.0**2) / 200):.3f};  stratified = sqrt(2 * 1.5^2 / 200) = {np.sqrt(2 * 1.5**2 / 200):.3f}")

# A/A test: two arms with identical treatment should reject 5% of the time
p_aa = np.array([stats.ttest_ind(*np.split(rng.permutation(y_rct[~treated_rct][:2_000]), 2), equal_var=False).pvalue for _ in range(2_000)])
print(f"\nA/A test rejection rate at alpha = 0.05: {(p_aa < 0.05).mean():.3f}")
""")

nb.md(r"""
The observational estimate has the *wrong sign* ($-1.9$ against a truth of $+0.5$) even though the effect is
positive within each device type: treated users are mostly mobile users, who spend less time regardless —
Simpson's paradox, produced by confounded assignment. Randomisation makes the treated and control groups
have the same device mix and recovers $+0.5$. Blocking then removes the device variance from the comparison:
the standard deviation of the estimate drops from about $0.25$ to $0.15$, equivalent to nearly tripling the
sample size for free. The A/A test rejecting at the nominal 5% is the health check that the pipeline is honest.
""")

# ---------------------------------------------------------------- 9. peeking
nb.md(r"""
## 9. Peeking: checking a running test and stopping early

Two arms, standard normal noise, 1,000 users per arm in total. We look at the z-statistic after every
batch and stop as soon as $|Z| > 1.96$. Under $H_0$ the planned single look rejects 5% of the time; with
repeated looks the running statistic is a random walk that gets many chances to cross. Then the fix:
Pocock's constant boundary for a pre-specified number of looks.
""")

nb.code(r"""
def peeking_fpr(n_looks, n_per_look, n_sims, z_crit=1.96):
    a = rng.standard_normal((n_sims, n_looks, n_per_look)).sum(axis=2)
    b = rng.standard_normal((n_sims, n_looks, n_per_look)).sum(axis=2)
    n_cum = np.arange(1, n_looks + 1) * n_per_look
    z = (b.cumsum(axis=1) - a.cumsum(axis=1)) / np.sqrt(2 * n_cum)   # z-statistic at each look
    any_cross = np.any(np.abs(z) > z_crit, axis=1)
    return any_cross.mean(), (np.abs(z[:, -1]) > z_crit).mean(), z

looks = [1, 2, 5, 10, 20, 50]
armitage = {1: 0.05, 2: 0.083, 5: 0.142, 10: 0.193, 20: 0.248, 50: 0.320}   # Armitage, McPherson & Rowe (1969)
fprs = []
for L in looks:
    fpr_any, fpr_final, _ = peeking_fpr(L, 1000 // L, 3_000)
    fprs.append(fpr_any)
    print(f"{L:2d} look(s): P(stop early with 'significant' result | H0) = {fpr_any:.3f}   (tabulated {armitage[L]:.3f}; at the final look alone {fpr_final:.3f})")

plt.plot(looks, fprs, "o-", label="simulated, stop at first |Z| > 1.96")
plt.plot(looks, [armitage[L] for L in looks], "k--", label="Armitage et al. (1969)")
plt.axhline(0.05, color="gray", ls=":", label="nominal 0.05")
plt.xscale("log"); plt.xlabel("number of looks at the data"); plt.ylabel("false-positive rate under H0")
plt.title("Optional stopping inflates the type I error"); plt.legend(); plt.show()

# the fix: Pocock boundaries for a pre-specified number of looks
for L, zb in ((5, 2.413), (10, 2.555), (20, 2.672)):
    fpr_any, _, _ = peeking_fpr(L, 1000 // L, 3_000, z_crit=zb)
    print(f"Pocock boundary |Z| > {zb} with {L:2d} looks: false-positive rate = {fpr_any:.3f}")
_, _, z20 = peeking_fpr(20, 50, 3_000)
stopped = np.argmax(np.abs(z20) > 1.96, axis=1); ever = np.any(np.abs(z20) > 1.96, axis=1)
print(f"\namong null experiments stopped early, median stopping look = {np.median(stopped[ever]) + 1:.0f} of 20;"
      f" mean |Z| at stopping = {np.abs(z20[ever, stopped[ever]]).mean():.2f} (an 'effect' that does not exist)")
""")

nb.md(r"""
The simulation reproduces the classical table: one look, 5%; five looks, 14%; twenty looks, 25%. Note the
last column — at the *final* look alone the rate is still 5%; the inflation comes entirely from the option
to stop earlier. Pocock's boundaries restore the nominal rate at the price of needing $|Z| > 2.4$ to $2.7$
instead of $1.96$; O'Brien–Fleming boundaries spend less $\alpha$ early and are the clinical-trial default.
The stopped experiments report large "effects" that are pure noise — the winner's curse of optional stopping.

**Try it yourself:** replace `z_crit=1.96` by the Bonferroni-per-look value `stats.norm.ppf(1 - 0.025 / L)`
for `L = 10` — it is valid, but how much stricter than Pocock's $2.555$ is it?
""")

# ---------------------------------------------------------------- 10. comparing ML models
nb.md(r"""
## 10. Comparing two classifiers properly

Logistic regression versus $k$-nearest neighbours on the bundled breast-cancer data. First a single
held-out set: the two models' per-example results are *paired*, so we use **McNemar's test** on the
discordant counts $b$ (A right, B wrong) and $c$ (B right, A wrong),

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \qquad \text{exact: } b \sim \text{Binomial}(b + c, \tfrac12).$$

Then repeated stratified cross-validation ($10 \times 5$-fold, $J = 50$ paired differences) with the naive
paired t-test and the **Nadeau–Bengio** correction, which inflates the variance by
$(1/J + n_{\text{test}}/n_{\text{train}})/(1/J)$ to account for the overlap between training sets.
""")

nb.code(r"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
model_a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5_000))
model_b = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

def mcnemar(y_true, pred_a, pred_b):
    right_a, right_b = pred_a == y_true, pred_b == y_true
    b = int(np.sum(right_a & ~right_b)); c = int(np.sum(~right_a & right_b))
    chi2_cc = (abs(b - c) - 1)**2 / (b + c) if b + c > 0 else 0.0
    p_cc = stats.chi2.sf(chi2_cc, 1)
    p_exact = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, chi2_cc, p_cc, p_exact

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
pred_a = model_a.fit(X_tr, y_tr).predict(X_te)
pred_b = model_b.fit(X_tr, y_tr).predict(X_te)
acc_a, acc_b = np.mean(pred_a == y_te), np.mean(pred_b == y_te)
b, c, chi2_cc, p_cc, p_exact = mcnemar(y_te, pred_a, pred_b)
print(f"held-out set of {len(y_te)} examples: accuracy logistic = {acc_a:.4f}, kNN = {acc_b:.4f}, gap = {acc_a - acc_b:+.4f}")
print(f"discordant examples: logistic right & kNN wrong b = {b}, kNN right & logistic wrong c = {c}  (both right/wrong: {len(y_te) - b - c})")
print(f"McNemar: chi2 (continuity-corrected) = {chi2_cc:.3f}, p = {p_cc:.3f};  exact binomial p = {p_exact:.3f}")
se_single = np.sqrt(acc_a * (1 - acc_a) / len(y_te))
print(f"for scale: the standard error of one accuracy on this test set is {se_single:.4f}")
""")

nb.code(r"""
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
diffs = []
for tr, te in cv.split(X, y):
    a = model_a.fit(X[tr], y[tr]).score(X[te], y[te])
    b_ = model_b.fit(X[tr], y[tr]).score(X[te], y[te])
    diffs.append(a - b_)
diffs = np.array(diffs); J = len(diffs)
n_test, n_train = len(te), len(tr)

t_naive = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(J))
p_naive = 2 * stats.t.sf(abs(t_naive), J - 1)
rho = n_test / n_train
t_nb = diffs.mean() / np.sqrt((1 / J + rho) * diffs.var(ddof=1))
p_nb = 2 * stats.t.sf(abs(t_nb), J - 1)
print(f"{J} paired fold differences (logistic - kNN): mean {diffs.mean():+.4f}, sd {diffs.std(ddof=1):.4f}, folds favouring logistic: {(diffs > 0).sum()}")
print(f"naive paired t-test        : t = {t_naive:.2f}, p = {p_naive:.4f}   (scipy ttest_1samp: p = {stats.ttest_1samp(diffs, 0).pvalue:.4f})")
print(f"Nadeau-Bengio corrected    : t = {t_nb:.2f}, p = {p_nb:.4f}   (variance inflated by factor {(1 / J + rho) / (1 / J):.1f})")

plt.hist(diffs, bins=15, alpha=0.8)
plt.axvline(0, color="k", lw=1); plt.axvline(diffs.mean(), color="C3", ls="--", label=f"mean difference {diffs.mean():+.4f}")
plt.xlabel("accuracy(logistic) - accuracy(kNN) per fold"); plt.ylabel("count")
plt.title("10 x 5-fold CV: paired accuracy differences"); plt.legend(); plt.show()
""")

nb.md(r"""
The two tests tell one story once done properly. On the single split the models differ on only nine of 228
examples, split 6:3, and McNemar cannot distinguish them ($p \approx 0.5$) — the 1.3-point accuracy gap is
about one standard error of a single accuracy. In repeated cross-validation logistic regression wins 33 of 50
folds and the *naive* paired t-test declares $p = 0.0001$; but it treats the 50 overlapping folds as
independent. The Nadeau–Bengio correction multiplies the variance by
$1 + J\,n_{\text{test}}/n_{\text{train}} \approx 13.5$, the t-statistic shrinks by $\sqrt{13.5} \approx 3.7$, and
$p \approx 0.26$: a plausible 1-point edge for logistic regression that this dataset of 569 examples cannot
confirm. If you must claim that one model beats another, report the paired difference with its corrected
interval, over several seeds — not a single number with three decimals.
""")

nb.md(r"""
## Summary

- A p-value is $P(\text{data at least this extreme} \mid H_0)$, uniform under the null, and never $P(H_0 \mid \text{data})$; the exact binomial test and the normal approximation disagree exactly where it matters.
- Power depends on $\delta/\sigma$, $n$ and $\alpha$ through $\Phi(\delta/(\sigma\sqrt{2/n}) - z_{1-\alpha/2})$; halving the effect quadruples the required sample; underpowered tests "find nothing" about everything.
- Welch's t-test is the default two-sample test; pairing removes shared variation and can turn $p = 0.40$ into $p = 0.008$ on the same numbers.
- The chi-square test of a $2 \times 2$ table equals the two-proportion z-test squared; small tables call for Fisher's exact test.
- Permutation tests shuffle labels to obtain an exact null distribution for any statistic; Mann–Whitney is a permutation test on ranks.
- Confidence intervals and tests are one object; equivalence needs TOST with a pre-specified margin.
- Bonferroni and Holm control FWER, Benjamini–Hochberg controls FDR with far more power; hyperparameter search and dashboards are multiple-comparison problems.
- Randomisation defeats confounding (Simpson's paradox), blocking removes known variance, A/A tests validate the pipeline.
- Peeking inflates the false-positive rate to 25% with 20 looks; Pocock or O'Brien–Fleming boundaries restore it.
- Compare models with McNemar on paired predictions or the Nadeau–Bengio-corrected test on repeated CV; naive fold t-tests are several times too confident.

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
from itertools import combinations
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
    "Module 12 — Exercises",
    intro_md=r"""
These exercises follow the order of the concept guide: the logic of a test and p-values, power and sample
size, t-tests (Welch and paired), proportions and chi-square, permutation tests, multiple comparisons,
experimental design, peeking, and finally comparing machine-learning models. Several checks compare your
answer against a *hidden* value, so compute carefully rather than guessing. `statsmodels` is not available
(and not needed): everything is NumPy plus `scipy.stats`.
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- 1. exact binomial p-value (warm-up)
SOL_1 = r"""
def binom_two_sided_p(k, n, theta0=0.5):
    lower = stats.binom.cdf(k, n, theta0)          # P(X <= k)
    upper = stats.binom.sf(k - 1, n, theta0)       # P(X >= k)
    return float(min(1.0, 2 * min(lower, upper)))

def normal_approx_p(k, n, theta0=0.5, continuity=True):
    mu = n * theta0
    sd = np.sqrt(n * theta0 * (1 - theta0))
    dev = abs(k - mu) - (0.5 if continuity else 0.0)
    dev = max(dev, 0.0)
    return float(2 * stats.norm.sf(dev / sd))
"""
ns1 = run(SOL_1, r"""
a = binom_two_sided_p(60, 100)
b = normal_approx_p(60, 100, continuity=True)
c = normal_approx_p(60, 100, continuity=False)
d = binom_two_sided_p(27, 40, 0.5)
e = binom_two_sided_p(14, 30, 0.3)
""")
ex.add(
    title="Exact binomial p-value versus the normal approximation", difficulty=1,
    prompt_md=r"""
Implement two functions for testing $H_0: \theta = \theta_0$ with $k$ successes in $n$ trials:

* `binom_two_sided_p(k, n, theta0=0.5)` — the exact two-sided p-value as the **doubled smaller tail**,
  $\min\big(1,\ 2\min(P(X \le k),\ P(X \ge k))\big)$ with $X \sim \text{Binomial}(n, \theta_0)$. Use
  `stats.binom.cdf` / `stats.binom.sf` (careful: `sf(k)` is $P(X > k)$, not $P(X \ge k)$). Return a float.
* `normal_approx_p(k, n, theta0=0.5, continuity=True)` — the two-sided normal-approximation p-value,
  $2\big(1 - \Phi(z)\big)$ with $z = \big(|k - n\theta_0| - \tfrac12\big)/\sqrt{n\theta_0(1-\theta_0)}$ when
  `continuity=True` (clip the numerator at 0) and without the $\tfrac12$ otherwise.
""",
    starter=r"""
def binom_two_sided_p(k, n, theta0=0.5):
    # YOUR CODE HERE
    return None

def normal_approx_p(k, n, theta0=0.5, continuity=True):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_1,
    check=rf"""
try:
    got = binom_two_sided_p(50, 100)
except Exception:
    got = None
check_close('50 heads in 100 flips: p = 1', got, 1.0)
try:
    got = binom_two_sided_p(60, 100)
except Exception:
    got = None
check_hash('exact two-sided p for 60/100', got, '{safe_sha(ns1["a"], 4)}', decimals=4)
try:
    got = binom_two_sided_p(40, 100)
except Exception:
    got = None
check_hash('symmetry: 40/100 gives the same p', got, '{safe_sha(ns1["a"], 4)}', decimals=4)
try:
    got = normal_approx_p(60, 100, continuity=True)
except Exception:
    got = None
check_hash('normal approximation with continuity correction, 60/100', got, '{safe_sha(ns1["b"], 4)}', decimals=4)
try:
    got = normal_approx_p(60, 100, continuity=False)
except Exception:
    got = None
check_close('normal approximation without correction, 60/100', got, 2 * stats.norm.sf(2.0), tol=1e-6)
try:
    got = binom_two_sided_p(14, 30, 0.3)
except Exception:
    got = None
check_hash('asymmetric case: 14/30 under theta0 = 0.3', got, '{safe_sha(ns1["e"], 4)}', decimals=4)
""",
    explanation_md=r"""
The p-value is the null probability of a result at least as extreme as the observed one; for a two-sided
alternative on a symmetric binomial, doubling the smaller tail is the natural definition and coincides
with scipy's `binomtest` when $\theta_0 = 0.5$ (for asymmetric $\theta_0$ the two conventions can differ
slightly — that is not a bug, two-sided p-values for discrete tests are genuinely a matter of definition).
The most common wrong turn is `stats.binom.sf(k, n, theta0)`, which is $P(X > k)$ and silently drops the
observed outcome itself; use `sf(k - 1)` or `1 - cdf(k - 1)`. The continuity correction moves the normal
approximation from $0.046$ to $0.057$, within $0.001$ of the exact $0.057$.
""",
)

# ---------------------------------------------------------------- 2. power and sample size (warm-up)
SOL_2 = r"""
def power_two_sample_z(delta, sigma, n, alpha=0.05):
    se = sigma * np.sqrt(2.0 / n)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(-zc - delta / se) + stats.norm.sf(zc - delta / se))

def n_per_arm(delta, sigma, power=0.8, alpha=0.05):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * (za + zb)**2 * sigma**2 / delta**2))

def mde(sigma, n, power=0.8, alpha=0.05):
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sigma * np.sqrt(2.0 / n))
"""
ns2 = run(SOL_2, r"""
a = power_two_sample_z(4.0, 10.0, 50)
b = n_per_arm(4.0, 10.0)
c = n_per_arm(0.5, 2.0, power=0.9)
d = mde(10.0, 50)
e = power_two_sample_z(0.0, 10.0, 50)
""")
ex.add(
    title="Power, required sample size and minimum detectable effect", difficulty=1,
    prompt_md=r"""
For the two-sample z-test (known $\sigma$, $n$ per arm, two-sided level $\alpha$) implement

* `power_two_sample_z(delta, sigma, n, alpha=0.05)` — the power
  $\Phi\big(\tfrac{\delta}{\sigma\sqrt{2/n}} - z_{1-\alpha/2}\big) + \Phi\big(-\tfrac{\delta}{\sigma\sqrt{2/n}} - z_{1-\alpha/2}\big)$ as a float;
* `n_per_arm(delta, sigma, power=0.8, alpha=0.05)` — the required sample size per arm,
  $\big\lceil 2(z_{1-\alpha/2} + z_{1-\beta})^2\sigma^2/\delta^2 \big\rceil$, as an **int**;
* `mde(sigma, n, power=0.8, alpha=0.05)` — the minimum detectable effect $(z_{1-\alpha/2} + z_{1-\beta})\,\sigma\sqrt{2/n}$ as a float.

Use `stats.norm.ppf` for the quantiles and `stats.norm.cdf` / `stats.norm.sf` for $\Phi$.
""",
    starter=r"""
def power_two_sample_z(delta, sigma, n, alpha=0.05):
    # YOUR CODE HERE
    return None

def n_per_arm(delta, sigma, power=0.8, alpha=0.05):
    # YOUR CODE HERE
    return None

def mde(sigma, n, power=0.8, alpha=0.05):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_2,
    check=rf"""
try:
    got = power_two_sample_z(0.0, 10.0, 50)
except Exception:
    got = None
check_close('no effect: power equals alpha = 0.05', got, 0.05, tol=1e-6)
try:
    got = power_two_sample_z(4.0, 10.0, 100)
except Exception:
    got = None
check_close('delta 4, sigma 10, n 100: power ~ 0.807', got, 0.8074, tol=1e-3)
try:
    got = power_two_sample_z(4.0, 10.0, 50)
except Exception:
    got = None
check_hash('power at n = 50', got, '{safe_sha(ns2["a"], 3)}', decimals=3)
try:
    got = n_per_arm(4.0, 10.0)
    ok_int = isinstance(got, (int, np.integer))
except Exception:
    got, ok_int = None, False
check('n_per_arm returns an integer', ok_int)
check_hash('n per arm for delta 4, sigma 10, 80% power', got, '{safe_sha(ns2["b"])}')
try:
    got = n_per_arm(0.5, 2.0, power=0.9)
except Exception:
    got = None
check_hash('n per arm for delta 0.5, sigma 2, 90% power', got, '{safe_sha(ns2["c"])}')
try:
    got = mde(10.0, 50)
except Exception:
    got = None
check_hash('MDE for sigma 10, n 50', got, '{safe_sha(ns2["d"], 3)}', decimals=3)
try:
    got = power_two_sample_z(mde(10.0, 50), 10.0, 50)
except Exception:
    got = None
check_close('consistency: power at the MDE is 0.8', got, 0.8, tol=1e-3)
""",
    explanation_md=r"""
Under $H_1$ the standardised difference is $\mathcal N(\delta/\text{SE}, 1)$ with $\text{SE} = \sigma\sqrt{2/n}$,
and the test rejects when it falls outside $\pm z_{1-\alpha/2}$; the two $\Phi$ terms are the two tails of that
shifted normal. Setting the main term to $1 - \beta$ gives $\delta/\text{SE} = z_{1-\alpha/2} + z_{1-\beta}$, which
solves to the sample-size and MDE formulas (the tiny wrong-direction term is dropped, so the power at the MDE
is $0.8$ to three decimals rather than exactly). Common slips: forgetting the factor 2 (two arms contribute
variance), using $z_{1-\alpha}$ instead of $z_{1-\alpha/2}$ for a two-sided test, and returning a float where a
sample size must be rounded *up* to an integer.
""",
)

# ---------------------------------------------------------------- 3. Welch (warm-up)
SOL_3 = r"""
def welch_t(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1) / n1, y.var(ddof=1) / n2
    t = (x.mean() - y.mean()) / np.sqrt(v1 + v2)
    df = (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return float(t), float(df), float(p)
"""
ex.add(
    title="Welch's t-test from scratch", difficulty=1,
    prompt_md=r"""
Implement `welch_t(x, y)` returning the tuple `(t, df, p)` for two independent samples:

$$t = \frac{\bar x - \bar y}{\sqrt{s_x^2/n_x + s_y^2/n_y}}, \qquad
\nu = \frac{(s_x^2/n_x + s_y^2/n_y)^2}{\dfrac{(s_x^2/n_x)^2}{n_x - 1} + \dfrac{(s_y^2/n_y)^2}{n_y - 1}}, \qquad
p = 2\,P(t_\nu \ge |t|).$$

Use sample variances with `ddof=1` and `stats.t.sf` for the tail probability. Do **not** call
`stats.ttest_ind` inside the function (the checks compare against it).
""",
    starter=r"""
def welch_t(x, y):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_3,
    check=r"""
old = np.array([52., 48, 55, 50, 53, 49, 51, 54]); new = np.array([47., 50, 44, 49, 46, 48])
ref = stats.ttest_ind(old, new, equal_var=False)
try:
    t, df, p = welch_t(old, new)
except Exception:
    t, df, p = None, None, None
check_close('cycle times: t statistic matches scipy', t, ref.statistic)
check_close('cycle times: Welch-Satterthwaite df matches scipy', df, ref.df)
check_close('cycle times: p-value matches scipy', p, ref.pvalue)
_r = np.random.default_rng(3)
x2 = _r.normal(0, 1, 12); y2 = _r.normal(0.5, 4, 40)          # very unequal variances and sizes
ref2 = stats.ttest_ind(x2, y2, equal_var=False)
try:
    t2, df2, p2 = welch_t(x2, y2)
except Exception:
    t2, df2, p2 = None, None, None
check_close('unequal variances: p-value matches scipy', p2, ref2.pvalue)
check('df lies between min(n) - 1 and n1 + n2 - 2', (df2 is not None) and (11 <= df2 <= 50))
""",
    explanation_md=r"""
Welch's statistic standardises the difference of means by the *unpooled* standard error, and the
Welch–Satterthwaite formula picks the degrees of freedom for which a scaled chi-square best matches the
distribution of that estimated variance; the result is fractional and lies between $\min(n_x, n_y) - 1$ and
$n_x + n_y - 2$. Typical mistakes: `ddof=0` (population variance), pooling the variances (that is Student's
test), or using `stats.t.cdf` without the factor 2 for a two-sided p-value.
""",
)

# ---------------------------------------------------------------- 4. conceptual: p-values (warm-up)
ex.add_conceptual(
    title="What a p-value is not", difficulty=1,
    prompt_md=r"""
An A/B test on a new checkout page reports a lift of 0.6 percentage points with $p = 0.03$. For each of the
following statements say whether it is correct, and if not, what the correct statement is:

1. "There is a 3% probability that the new page is no better than the old one."
2. "There is a 97% probability that the new page is better."
3. "If the new page were truly no better, a lift at least this large (in either direction) would occur in about 3% of such experiments."
4. "If we repeated the experiment, it would come out significant 97% of the time."
5. "The result is more trustworthy than a competitor's $p = 0.04$ result from an experiment ten times larger."
""",
    answer_md=r"""
Only statement 3 is correct — it is the definition of a two-sided p-value: a probability about *data*
computed under $H_0$. Statement 1 reverses the conditioning: $P(\text{data} \mid H_0) \ne P(H_0 \mid \text{data})$;
the latter needs a prior over hypotheses and Bayes' theorem (with 10% of ideas being real improvements and
power 0.8, a significant result is a false positive about a third of the time). Statement 2 is the same error
for $H_1$. Statement 4 confuses the p-value with power: the replication probability depends on the true
effect and the sample size and, for an effect just large enough to give $p = 0.03$, is only around 50%.
Statement 5 is wrong twice: p-values from different sample sizes are not comparable as "trustworthiness",
and the larger experiment's $p = 0.04$ corresponds to a much *smaller* estimated effect with a much
*narrower* interval — compare effect sizes and confidence intervals, not p-values.
""",
)

# ---------------------------------------------------------------- 5. paired t-test (core)
SOL_5 = r"""
def paired_t(before, after):
    d = np.asarray(before, float) - np.asarray(after, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / se if se > 0 else np.sign(d.mean()) * np.inf     # all differences equal -> infinite t
    df = n - 1
    p = 2 * stats.t.sf(abs(t), df)
    return float(t), int(df), float(p)

before = np.array([31.0, 28.5, 35.2, 30.1, 27.8, 33.4])
after = np.array([29.1, 27.9, 33.0, 29.6, 27.0, 31.5])
t_paired, df_paired, p_paired = paired_t(before, after)
p_unpaired = float(stats.ttest_ind(before, after, equal_var=False).pvalue)
"""
ns5 = run(SOL_5)
ex.add(
    title="Paired t-test and the cost of ignoring the pairing", difficulty=2,
    prompt_md=r"""
Six machines were timed before and after a maintenance change (minutes per cycle):

```
before = [31.0, 28.5, 35.2, 30.1, 27.8, 33.4]
after  = [29.1, 27.9, 33.0, 29.6, 27.0, 31.5]
```

1. Implement `paired_t(before, after)` returning `(t, df, p)` for the paired t-test on the differences
   $d_i = \text{before}_i - \text{after}_i$: $t = \bar d/(s_d/\sqrt n)$, $\nu = n - 1$ (an int), two-sided $p$.
   Do not call `stats.ttest_rel` inside the function.
2. Assign `t_paired, df_paired, p_paired = paired_t(before, after)`.
3. Assign `p_unpaired` = the two-sided p-value of **Welch's** test on the same two arrays (you may use
   `stats.ttest_ind(..., equal_var=False)` here), to see what ignoring the pairing costs.
""",
    starter=r"""
def paired_t(before, after):
    # YOUR CODE HERE
    return None

before = np.array([31.0, 28.5, 35.2, 30.1, 27.8, 33.4])
after = np.array([29.1, 27.9, 33.0, 29.6, 27.0, 31.5])
t_paired, df_paired, p_paired = None, None, None
p_unpaired = None
# YOUR CODE HERE
""",
    solution=SOL_5,
    check=rf"""
check_hash('paired t statistic', t_paired, '{safe_sha(ns5["t_paired"], 3)}', decimals=3)
check_hash('degrees of freedom', df_paired, '{safe_sha(ns5["df_paired"])}')
check_hash('paired p-value', p_paired, '{safe_sha(ns5["p_paired"], 3)}', decimals=3)
try:
    got = paired_t(before, after)[2]
except Exception:
    got = None
check_close('matches stats.ttest_rel', got, stats.ttest_rel(before, after).pvalue)
check('unpaired Welch p-value is above 0.3 on the same numbers', (p_unpaired is not None) and p_unpaired > 0.3)
check('pairing makes the p-value more than 20 times smaller', (p_paired is not None and p_unpaired is not None) and p_unpaired / p_paired > 20)
try:
    got = paired_t(np.array([1., 2., 3., 4.]), np.array([0., 1., 2., 3.]) )
    ok = got is not None and np.isinf(got[0]) or (got is not None and abs(got[0]) > 1e6)
except Exception:
    ok = False
check('constant differences give an infinite (or huge) t', ok)
""",
    explanation_md=r"""
The paired test is a one-sample test on the differences: $\bar d = 1.317$, $s_d = 0.763$,
$t = 1.317/(0.763/\sqrt 6) = 4.23$ on 5 degrees of freedom, $p = 0.008$. The unpaired Welch test on the same
twelve numbers gives $p \approx 0.40$ because the between-machine spread (about 2.9 minutes) enters the
standard error, whereas the differences remove it; the before/after correlation is $0.98$. When every
difference is identical, $s_d = 0$ and $t = \pm\infty$ — a degenerate but correct answer (guard the division
so NumPy does not warn). The classic bug is to compute `before.std() - after.std()` or to pool variances
as if the samples were independent.
""",
)

# ---------------------------------------------------------------- 6. chi-square 2x2 (core)
SOL_6 = r"""
def chi2_2x2(table):
    table = np.asarray(table, float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    chi2 = ((table - expected)**2 / expected).sum()
    p = stats.chi2.sf(chi2, 1)
    return float(chi2), float(p), expected

def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pbar = (x1 + x2) / (n1 + n2)
    return float((p2 - p1) / np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2)))
"""
ns6 = run(SOL_6, r"""
chi2_side, p_side, E_side = chi2_2x2([[30, 170], [18, 182]])
z_conv = two_prop_z(200, 4000, 240, 4000)
""")
ex.add(
    title="Chi-square test of a 2 x 2 table and its identity with the z-test", difficulty=2,
    prompt_md=r"""
Implement

* `chi2_2x2(table)` — for a $2 \times 2$ array of counts, return `(chi2, p, expected)` where
  `expected[i, j] = R_i C_j / N` (row total times column total over the grand total),
  $\chi^2 = \sum_{ij}(O_{ij} - E_{ij})^2/E_{ij}$ **without** continuity correction, and $p = P(\chi^2_1 \ge \chi^2)$;
* `two_prop_z(x1, n1, x2, n2)` — the two-proportion z-statistic $(\hat\theta_2 - \hat\theta_1)/\sqrt{\bar\theta(1-\bar\theta)(1/n_1 + 1/n_2)}$ with the pooled rate $\bar\theta$.

Then convince yourself numerically that $\chi^2 = z^2$ for the conversion table `[[200, 3800], [240, 3760]]`
(rows = variants A and B, columns = converted / not converted).
""",
    starter=r"""
def chi2_2x2(table):
    # YOUR CODE HERE
    return None

def two_prop_z(x1, n1, x2, n2):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_6,
    check=rf"""
conv = np.array([[200, 3800], [240, 3760]])
ref = stats.chi2_contingency(conv, correction=False)
try:
    chi2, p, E = chi2_2x2(conv)
except Exception:
    chi2, p, E = None, None, None
check_close('conversion table: chi2 matches scipy (no correction)', chi2, ref.statistic)
check_close('conversion table: p matches scipy', p, ref.pvalue)
check_close('expected counts', E, ref.expected_freq)
try:
    z = two_prop_z(200, 4000, 240, 4000)
except Exception:
    z = None
check_hash('two-proportion z for the conversion table', z, '{safe_sha(ns6["z_conv"], 3)}', decimals=3)
check_close('chi2 equals z squared', (z**2 if z is not None else None), chi2 if chi2 is not None else -1.0, tol=1e-9)
try:
    chi2s, ps, Es = chi2_2x2([[30, 170], [18, 182]])
except Exception:
    chi2s, ps = None, None
check_hash('side-effect table: chi2 statistic', chi2s, '{safe_sha(ns6["chi2_side"], 3)}', decimals=3)
check_hash('side-effect table: p-value', ps, '{safe_sha(ns6["p_side"], 3)}', decimals=3)
""",
    explanation_md=r"""
Under independence the expected count in a cell is (row share) $\times$ (column share) $\times N$, i.e.
$R_i C_j/N$. For a $2 \times 2$ table all four $|O - E|$ are equal, and a few lines of algebra turn
$\sum (O - E)^2/E$ into $(\hat\theta_2 - \hat\theta_1)^2/[\bar\theta(1-\bar\theta)(1/n_1 + 1/n_2)] = z^2$ — the two
tests are one test, which is why their p-values agree exactly ($0.0498$ here). The degrees of freedom are
$(2-1)(2-1) = 1$. Watch out for `np.outer(table.sum(0), table.sum(1))`, which transposes the expected
counts (harmless for symmetric margins, wrong in general), and for applying Yates's correction when the
exercise asks for the plain statistic.
""",
)

# ---------------------------------------------------------------- 7. permutation test (core)
SOL_7 = r"""
def exact_perm_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b]); n1 = len(a)
    obs = abs(a.mean() - b.mean())
    count, total = 0, 0
    for idx in combinations(range(len(pooled)), n1):
        idx = list(idx)
        rest = np.delete(pooled, idx)
        diff = abs(pooled[idx].mean() - rest.mean())
        count += diff >= obs - 1e-12
        total += 1
    return count / total

def perm_test_mc(a, b, n_perm, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b]); n1 = len(a)
    obs = abs(a.mean() - b.mean())
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        count += abs(perm[:n1].mean() - perm[n1:].mean()) >= obs - 1e-12
    return (1 + count) / (n_perm + 1)
"""
PERM_DATA = r"""
_r = np.random.default_rng(7)
spend_a = np.round(_r.exponential(20.0, 40), 2)          # control: skewed spend, mean 20
spend_b = np.round(_r.exponential(28.0, 40), 2)          # treatment: mean 28
"""
ns7 = run(SOL_7, PERM_DATA + r"""
p_exact_small = exact_perm_p([12., 15, 11], [9., 8, 10])
p_ref = perm_test_mc(spend_a, spend_b, 200_000, np.random.default_rng(123))
p_welch = float(stats.ttest_ind(spend_a, spend_b, equal_var=False).pvalue)
""")
print(f"permutation reference p (200k perms) = {ns7['p_ref']:.4f}, Welch p = {ns7['p_welch']:.4f}")
ex.add(
    title="Permutation tests: exact and Monte Carlo", difficulty=2,
    prompt_md=r"""
Implement two permutation tests for the two-sided null "the two samples come from the same distribution",
using the absolute difference of means as the statistic:

* `exact_perm_p(a, b)` — enumerate **all** $\binom{n_a + n_b}{n_a}$ ways to choose which pooled observations
  form group $a$ (`itertools.combinations`), and return the fraction of splits whose
  $|\bar a^* - \bar b^*|$ is $\ge$ the observed value (use a tolerance such as `>= obs - 1e-12` to be safe with
  floating point). Only for tiny samples.
* `perm_test_mc(a, b, n_perm, rng)` — shuffle the pooled data `n_perm` times with `rng.permutation`, take the
  first $n_a$ values as the new group $a$, and return $\dfrac{1 + \#\{|T^*| \ge |t_{\text{obs}}|\}}{n_{\text{perm}} + 1}$.

Data for the Monte Carlo test (skewed spend per customer, 40 per arm) are generated below with a fixed seed.
""" + "\n```python\n" + textwrap.dedent(PERM_DATA).strip() + "\n```\n",
    starter=PERM_DATA + r"""

def exact_perm_p(a, b):
    # YOUR CODE HERE
    return None

def perm_test_mc(a, b, n_perm, rng):
    # YOUR CODE HERE
    return None
""",
    solution=PERM_DATA + "\n" + SOL_7,
    check=rf"""
try:
    got = exact_perm_p([12., 15, 11], [9., 8, 10])
except Exception:
    got = None
check_hash('exact two-sided p for the 3-vs-3 example', got, '{safe_sha(ns7["p_exact_small"], 3)}', decimals=3)
try:
    got = exact_perm_p([1., 2., 3.], [1., 2., 3.])
except Exception:
    got = None
check_close('identical groups: every split is at least as extreme, p = 1', got, 1.0)
try:
    got = exact_perm_p([5., 6., 7., 8.], [1., 2., 3., 4.])
except Exception:
    got = None
check_close('completely separated 4-vs-4 groups: p = 2/70', got, 2 / 70)
try:
    p_mc = perm_test_mc(spend_a, spend_b, 5_000, np.random.default_rng(1))
except Exception:
    p_mc = None
check('Monte Carlo p is a probability in (0, 1]', (p_mc is not None) and 0 < p_mc <= 1)
check('Monte Carlo p within 0.015 of the reference value', (p_mc is not None) and abs(p_mc - {ns7["p_ref"]:.4f}) < 0.015,
      hint='did you compare |T*| with |t_obs| and add 1 to numerator and denominator?')
try:
    p_same = perm_test_mc(spend_a, spend_a, 2_000, np.random.default_rng(2))
except Exception:
    p_same = None
check('a sample against itself: p = 1', (p_same is not None) and abs(p_same - 1.0) < 1e-12)
""",
    explanation_md=r"""
Under the null the group labels are exchangeable, so every assignment of the pooled values to the two
groups is equally likely and the fraction of assignments with a statistic at least as extreme as the one
observed *is* the p-value — no distributional assumption anywhere. For 3 vs 3 there are 20 splits, and
only the observed one and its mirror image reach $|{\bar a} - {\bar b}| = 3.67$, so $p = 0.10$; for the
separated 4-vs-4 case the two extreme splits out of $\binom 84 = 70$ give $2/70$. The Monte Carlo version
estimates the same fraction with random permutations; adding 1 to numerator and denominator counts the
observed arrangement itself and guarantees the estimated p-value is never 0 and remains valid. Comparing a
sample against itself gives $p = 1$ because every permutation ties the observed statistic of exactly 0.
The reference value for the skewed spend data is close to Welch's p-value here (40 per arm is enough for
the CLT), but the permutation test would stay exact at 8 per arm, where the t-test would not.
""",
)

# ---------------------------------------------------------------- 8. multiple comparisons (core)
SOL_8 = r"""
def bonferroni_reject(p, alpha=0.05):
    p = np.asarray(p, float)
    return p <= alpha / len(p)

def holm_reject(p, alpha=0.05):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p)
    thresholds = alpha / (m - np.arange(m))                 # alpha/m, alpha/(m-1), ..., alpha
    passed = p[order] <= thresholds
    k = np.argmin(passed) if not passed.all() else m       # index of the first failure
    reject = np.zeros(m, bool)
    reject[order[:k]] = True
    return reject

def bh_reject(p, q=0.05):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p)
    thresholds = q * np.arange(1, m + 1) / m               # q i / m
    passed = np.flatnonzero(p[order] <= thresholds)
    reject = np.zeros(m, bool)
    if len(passed):
        k = passed.max() + 1                               # largest i with p_(i) <= q i / m
        reject[order[:k]] = True
    return reject
"""
PVALS = r"""
pvals = np.array([0.040, 0.350, 0.001, 0.019, 0.600, 0.012, 0.200, 0.045, 0.011, 0.900, 0.004, 0.030])
"""
ns8 = run(SOL_8, PVALS + r"""
rb = np.flatnonzero(bonferroni_reject(pvals)).tolist()
rh = np.flatnonzero(holm_reject(pvals)).tolist()
rq = np.flatnonzero(bh_reject(pvals)).tolist()
rq10 = np.flatnonzero(bh_reject(pvals, q=0.10)).tolist()
""")
print("multiple comparisons: Bonferroni", ns8["rb"], "Holm", ns8["rh"], "BH", ns8["rq"], "BH q=0.1", ns8["rq10"])
ex.add(
    title="Bonferroni, Holm and Benjamini–Hochberg", difficulty=2,
    prompt_md=r"""
Implement three functions that take an array of (unsorted) p-values and return a **boolean array** of the
same length, `True` where the hypothesis is rejected:

* `bonferroni_reject(p, alpha=0.05)` — reject if $p_i \le \alpha/m$;
* `holm_reject(p, alpha=0.05)` — sort the p-values; compare $p_{(i)}$ with $\alpha/(m - i + 1)$ for
  $i = 1, 2, \dots$; reject all hypotheses **before the first failure** and nothing after it (step-down);
* `bh_reject(p, q=0.05)` — sort; find the **largest** $k$ with $p_{(k)} \le kq/m$; reject the $k$ hypotheses with
  the smallest p-values (step-up), or none if no such $k$ exists.

Remember to map the decisions back to the original order (use `np.argsort`). Apply them to the twelve
p-values below; the checks hash the sorted *indices* of the rejected hypotheses.
""" + "\n```python\n" + textwrap.dedent(PVALS).strip() + "\n```\n",
    starter=PVALS + r"""

def bonferroni_reject(p, alpha=0.05):
    # YOUR CODE HERE
    return None

def holm_reject(p, alpha=0.05):
    # YOUR CODE HERE
    return None

def bh_reject(p, q=0.05):
    # YOUR CODE HERE
    return None
""",
    solution=PVALS + "\n" + SOL_8,
    check=rf"""
def _idx(fn, p, *args):
    try:
        r = np.asarray(fn(p, *args))
        return np.flatnonzero(r).tolist() if r.dtype == bool and r.shape == (len(p),) else None
    except Exception:
        return None
rb, rh, rq = _idx(bonferroni_reject, pvals), _idx(holm_reject, pvals), _idx(bh_reject, pvals)
check('functions return boolean arrays of the right length', rb is not None and rh is not None and rq is not None)
check_hash('Bonferroni rejections (indices)', rb, '{safe_sha(ns8["rb"])}')
check_hash('Holm rejections (indices)', rh, '{safe_sha(ns8["rh"])}')
check_hash('Benjamini-Hochberg rejections at q = 0.05 (indices)', rq, '{safe_sha(ns8["rq"])}')
check_hash('Benjamini-Hochberg rejections at q = 0.10 (indices)', _idx(bh_reject, pvals, 0.10), '{safe_sha(ns8["rq10"])}')
check('Bonferroni is a subset of Holm, Holm a subset of BH', (rb is not None and rh is not None and rq is not None) and set(rb) <= set(rh) <= set(rq))
check('nothing rejected when all p-values are 0.5', _idx(bh_reject, np.full(5, 0.5)) == [] and _idx(holm_reject, np.full(5, 0.5)) == [])
check('everything rejected when all p-values are tiny', _idx(bonferroni_reject, np.full(6, 1e-6)) == [0, 1, 2, 3, 4, 5])
""",
    explanation_md=r"""
Sorted, the twelve p-values are $0.001, 0.004, 0.011, 0.012, 0.019, 0.030, 0.040, 0.045, 0.2, \dots$.
Bonferroni's threshold is $0.05/12 = 0.0042$, so it rejects the two smallest. Holm compares $0.001$ with
$0.05/12$, $0.004$ with $0.05/11 = 0.0045$, then $0.011$ with $0.05/10 = 0.005$ — a failure, so it stops with the
same two rejections (Holm never rejects less than Bonferroni and often more). BH thresholds are
$0.05\,i/12$: $0.0042, 0.0083, 0.0125, 0.0167, 0.0208, 0.025, 0.029, 0.033, \dots$; the largest $i$ with
$p_{(i)} \le 0.05\,i/12$ is $i = 5$ ($0.019 \le 0.0208$), so BH rejects the five smallest *including* $0.011$ and
$0.012$, which exceeded their own thresholds — the step-up rule looks for the last success, not the first
failure. At $q = 0.10$ the thresholds double and more hypotheses pass. The standard bug is forgetting to
undo the sort: compute decisions on `p[order]` and write them back with `reject[order[:k]] = True`.
""",
)

# ---------------------------------------------------------------- 9. conceptual: randomisation (core)
ex.add_conceptual(
    title="Why randomise, and what blocking adds", difficulty=2,
    prompt_md=r"""
A streaming service lets users opt in to a redesigned home screen. After a month, opted-in users watch
on average 38 minutes per day and the others 52 minutes, so the product team concludes the redesign
*reduces* watch time. You notice that 80% of the opted-in users are on mobile, where average watch time is
30 minutes, while 80% of the others are on TVs, where it is 55 minutes; within each device type the
opted-in users actually watch about 3 minutes *more*.

1. Name the phenomenon and explain in two or three sentences why the overall comparison is misleading.
2. Explain why randomising the assignment (instead of letting users opt in) fixes the problem — including for
   confounders nobody has measured.
3. The team now plans a randomised test. What does *stratifying the randomisation by device* add on top of
   plain randomisation, and roughly when is it worth doing?
4. Name two further design safeguards you would put in the test plan and the failure each one prevents.
""",
    answer_md=r"""
1. This is Simpson's paradox caused by confounding: device type affects both who opts in (mobile users
   opt in far more) and the outcome (mobile sessions are shorter). The aggregate contrast compares a
   mostly-mobile treated group with a mostly-TV control group, so it measures the device effect, not the
   redesign; stratified by device, the redesign adds about 3 minutes.
2. Random assignment makes treatment independent of *every* user characteristic — device, age, tenure,
   enthusiasm, and whatever else nobody thought of — so in expectation the two groups have the same
   distribution of all covariates and the difference in means is unbiased for the average treatment effect.
   Adjusting for measured confounders in observational data can only fix the ones you measured.
3. Stratified (blocked) randomisation forces an exact 50:50 split *within* each device group, so the realised
   groups are balanced on device rather than balanced only on average, and the device-driven variance drops
   out of the comparison; the standard error shrinks by a factor involving the share of variance explained
   by the stratum (in the lesson, from about 0.25 to 0.15, like nearly tripling $n$). It is worth doing when
   a covariate is known in advance and strongly predicts the outcome; otherwise plain randomisation (or
   post-hoc covariate adjustment) does nearly as well.
4. Any two of: fix the sample size from a power calculation and a pre-registered primary metric (prevents
   peeking and metric shopping); an A/A test and a sample-ratio-mismatch check (prevents a broken randomiser
   or logging from masquerading as an effect); run for whole weekly cycles (prevents novelty and day-of-week
   effects); randomise by user rather than by session and watch for interference, e.g. shared household
   accounts (prevents contamination between arms); Holm or BH adjustment for guardrail metrics (prevents
   one chance "significant" guardrail out of many from driving the decision).
""",
)

# ---------------------------------------------------------------- 10. peeking simulation (stretch)
SOL_10 = r"""
def peeking_fpr(n_looks, n_per_look, n_sims, rng, alpha=0.05, z_crit=None):
    zc = stats.norm.ppf(1 - alpha / 2) if z_crit is None else z_crit
    a = rng.standard_normal((n_sims, n_looks, n_per_look)).sum(axis=2)
    b = rng.standard_normal((n_sims, n_looks, n_per_look)).sum(axis=2)
    n_cum = np.arange(1, n_looks + 1) * n_per_look
    z = (b.cumsum(axis=1) - a.cumsum(axis=1)) / np.sqrt(2 * n_cum)
    return float(np.any(np.abs(z) > zc, axis=1).mean())
"""
ex.add(
    title="How much does peeking inflate the false-positive rate?", difficulty=3,
    prompt_md=r"""
Implement `peeking_fpr(n_looks, n_per_look, n_sims, rng, alpha=0.05, z_crit=None)` that simulates an A/B test
**under the null** (both arms standard normal, known $\sigma = 1$) analysed sequentially:

* each simulated experiment collects `n_per_look` new observations per arm at each of `n_looks` looks;
* at look $k$ (cumulative $n_k = k \cdot$ `n_per_look` per arm) the statistic is
  $Z_k = (\bar b_k - \bar a_k)/\sqrt{2/n_k}$, equivalently $(\text{sum}_b - \text{sum}_a)/\sqrt{2n_k}$;
* the experiment "stops with a significant result" if $|Z_k| > z_c$ at **any** look, where
  $z_c = z_{1-\alpha/2}$ by default or the supplied `z_crit`;
* return the fraction of the `n_sims` experiments that stopped with a significant result (a float).

Vectorise: draw arrays of shape `(n_sims, n_looks, n_per_look)`, sum over the last axis, take cumulative sums
along the look axis. Use the `rng` passed in, not the global one.
""",
    starter=r"""
def peeking_fpr(n_looks, n_per_look, n_sims, rng, alpha=0.05, z_crit=None):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_10,
    check=r"""
try:
    f1 = peeking_fpr(1, 1000, 4000, np.random.default_rng(10))
except Exception:
    f1 = None
check('one look: false-positive rate close to 0.05', (f1 is not None) and 0.035 < f1 < 0.065)
try:
    f5 = peeking_fpr(5, 200, 4000, np.random.default_rng(11))
except Exception:
    f5 = None
check('five looks: rate between 0.11 and 0.18 (Armitage: 0.142)', (f5 is not None) and 0.11 < f5 < 0.18)
try:
    f20 = peeking_fpr(20, 50, 4000, np.random.default_rng(12))
except Exception:
    f20 = None
check('twenty looks: rate between 0.20 and 0.30 (Armitage: 0.248)', (f20 is not None) and 0.20 < f20 < 0.30)
check('more looks, more false positives', (f1 is not None and f5 is not None and f20 is not None) and f1 < f5 < f20)
try:
    fp = peeking_fpr(5, 200, 4000, np.random.default_rng(13), z_crit=2.413)
except Exception:
    fp = None
check("Pocock's boundary 2.413 for 5 looks restores roughly 0.05", (fp is not None) and 0.03 < fp < 0.075)
try:
    f_strict = peeking_fpr(5, 200, 4000, np.random.default_rng(14), alpha=0.01)
except Exception:
    f_strict = None
check('alpha = 0.01 with five looks: rate between 0.02 and 0.05', (f_strict is not None) and 0.02 < f_strict < 0.05)
""",
    explanation_md=r"""
The cumulative z-statistics form a correlated sequence (each look reuses all earlier data), and the
probability that a random walk *ever* leaves $\pm 1.96$ grows with the number of chances: about 0.05, 0.14
and 0.25 for 1, 5 and 20 looks, matching the classical table of Armitage, McPherson and Rowe. Pocock's
constant boundary 2.413 for five looks brings the overall rate back to 0.05, at the price that even the
final analysis needs $|Z| > 2.41$. Implementation pitfalls: using the *difference of means* with the
standard error of a single arm (forgetting the factor $\sqrt 2$), or testing only at the final look (which
gives 0.05 regardless of `n_looks` and misses the entire point).
""",
)

# ---------------------------------------------------------------- 11. McNemar + Nadeau–Bengio (stretch)
SOL_11 = r"""
def mcnemar(y_true, pred_a, pred_b):
    right_a = np.asarray(pred_a) == np.asarray(y_true)
    right_b = np.asarray(pred_b) == np.asarray(y_true)
    b = int(np.sum(right_a & ~right_b))
    c = int(np.sum(~right_a & right_b))
    chi2_cc = (abs(b - c) - 1)**2 / (b + c)
    p_exact = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, float(chi2_cc), p_exact

def nadeau_bengio_t(diffs, n_test, n_train):
    d = np.asarray(diffs, float); J = len(d)
    var = d.var(ddof=1) * (1 / J + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return float(t), float(p)
"""
MC_DATA = r"""
_r = np.random.default_rng(11)
y_true = _r.integers(0, 2, 800)
right_a = _r.random(800) < 0.90                                   # classifier A is right 90% of the time
right_b = np.where(right_a, _r.random(800) < 0.94, _r.random(800) < 0.45)   # B tends to be right when A is
pred_a = np.where(right_a, y_true, 1 - y_true)
pred_b = np.where(right_b, y_true, 1 - y_true)
cv_diffs = np.round(_r.normal(0.006, 0.015, 50), 4)             # 50 paired fold differences, 10 x 5-fold CV
"""
ns11 = run(SOL_11, MC_DATA + r"""
b, c, chi2_cc, p_exact = mcnemar(y_true, pred_a, pred_b)
t_nb, p_nb = nadeau_bengio_t(cv_diffs, 100, 400)
t_naive = float(stats.ttest_1samp(cv_diffs, 0).statistic)
""")
print(f"McNemar data: b={ns11['b']}, c={ns11['c']}, chi2={ns11['chi2_cc']:.3f}, p_exact={ns11['p_exact']:.4f}; NB t={ns11['t_nb']:.3f} p={ns11['p_nb']:.3f}, naive t={ns11['t_naive']:.3f}")
ex.add(
    title="McNemar's test and the Nadeau–Bengio correction", difficulty=3,
    prompt_md=r"""
Two ways to compare classifiers honestly.

1. `mcnemar(y_true, pred_a, pred_b)` → `(b, c, chi2_cc, p_exact)` for two classifiers evaluated on the **same**
   test set: `b` = number of examples A gets right and B wrong, `c` = the reverse (both ints),
   `chi2_cc` $= (|b - c| - 1)^2/(b + c)$ (the continuity-corrected statistic), and `p_exact` = the exact
   two-sided p-value of `stats.binomtest(min(b, c), b + c, 0.5)`.
2. `nadeau_bengio_t(diffs, n_test, n_train)` → `(t, p)`: for $J$ paired per-fold score differences $d_j$,
   $t = \bar d \big/ \sqrt{(1/J + n_{\text{test}}/n_{\text{train}})\,s_d^2}$ with $s_d^2$ the sample variance
   (`ddof=1`), and $p = 2P(t_{J-1} \ge |t|)$.

The test data below are generated with a fixed seed: 800 test examples, and 50 fold differences from a
$10 \times 5$-fold cross-validation (so `n_test = 100`, `n_train = 400` for a dataset of 500 examples).
""" + "\n```python\n" + textwrap.dedent(MC_DATA).strip() + "\n```\n",
    starter=MC_DATA + r"""

def mcnemar(y_true, pred_a, pred_b):
    # YOUR CODE HERE
    return None

def nadeau_bengio_t(diffs, n_test, n_train):
    # YOUR CODE HERE
    return None
""",
    solution=MC_DATA + "\n" + SOL_11,
    check=rf"""
try:
    b, c, chi2_cc, p_exact = mcnemar(y_true, pred_a, pred_b)
except Exception:
    b, c, chi2_cc, p_exact = None, None, None, None
check_hash('discordant count b (A right, B wrong)', b, '{safe_sha(ns11["b"])}')
check_hash('discordant count c (B right, A wrong)', c, '{safe_sha(ns11["c"])}')
check_hash('continuity-corrected McNemar statistic', chi2_cc, '{safe_sha(ns11["chi2_cc"], 3)}', decimals=3)
check_hash('exact McNemar p-value', p_exact, '{safe_sha(ns11["p_exact"], 3)}', decimals=3)
try:
    got = mcnemar(np.array([0, 1, 0, 1, 1, 0]), np.array([0, 1, 0, 1, 0, 1]), np.array([0, 1, 1, 1, 1, 0]))
except Exception:
    got = None
check_close('tiny example: b = 1, c = 2, chi2 = 0', (got[:3] if got is not None else None), (1, 2, 0.0))
try:
    t_nb, p_nb = nadeau_bengio_t(cv_diffs, 100, 400)
except Exception:
    t_nb, p_nb = None, None
check_hash('Nadeau-Bengio t statistic', t_nb, '{safe_sha(ns11["t_nb"], 3)}', decimals=3)
check_hash('Nadeau-Bengio p-value', p_nb, '{safe_sha(ns11["p_nb"], 3)}', decimals=3)
try:
    t_naive_like, _ = nadeau_bengio_t(cv_diffs, 0, 400)        # n_test = 0 switches the correction off
except Exception:
    t_naive_like = None
check_close('with n_test = 0 the statistic is the naive paired t', t_naive_like, stats.ttest_1samp(cv_diffs, 0).statistic)
check('the correction shrinks |t| by a factor between 3 and 4 here', (t_nb is not None and t_naive_like is not None) and 3 < abs(t_naive_like / t_nb) < 4)
""",
    explanation_md=r"""
McNemar ignores the examples both classifiers get right or wrong — they carry no information about the
*difference* — and asks whether the discordant ones split evenly: under $H_0$, $b \sim \text{Binomial}(b + c, \tfrac12)$,
so the exact p-value is a binomial test and $(|b - c| - 1)^2/(b + c)$ is its chi-square approximation. In the
tiny example $b = 1$, $c = 2$ and $(|1 - 2| - 1)^2 = 0$. For cross-validation, the $J$ fold differences are
correlated because training sets overlap; Nadeau and Bengio approximate that correlation by
$\rho \approx n_{\text{test}}/n_{\text{train}}$ and inflate the variance from $s_d^2/J$ to
$(1/J + \rho)\,s_d^2$ — with $J = 50$ and $\rho = 1/4$ a factor $13.5$, so $|t|$ shrinks by about $3.7$. Setting
$n_{\text{test}} = 0$ recovers the naive paired t-test, which is the comparison the last check makes. The
common mistake is to apply the correction factor to the standard error rather than to the variance, which
under-corrects by a square root.
""",
)

# ---------------------------------------------------------------- 12. mini-project (★★★)
PROJECT_CODE = r"""
rng = np.random.default_rng(2024)

# ---- 1. design: baseline 5%, minimum detectable relative lift 10%, alpha 0.05 two-sided, power 0.8
base, rel_mde = 0.05, 0.10
p_a, p_b_min = base, base * (1 + rel_mde)
delta_min = p_b_min - p_a
alpha, power = 0.05, 0.80
za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
n = int(np.ceil((za + zb)**2 * (p_a * (1 - p_a) + p_b_min * (1 - p_b_min)) / delta_min**2))
visitors_per_day = 5_000
print(f"design: detect {p_a:.3f} -> {p_b_min:.4f} (delta = {delta_min:.4f}) with power {power} at alpha {alpha}")
print(f"required n per arm = {n:,}  ->  {2 * n:,} visitors in total = {2 * n / visitors_per_day:.1f} days at {visitors_per_day:,}/day")

# ---- 2. run the experiment once (the truth: a 10% relative lift) and analyse it
true_p_b = p_a * 1.10
x_a, x_b = rng.binomial(n, p_a), rng.binomial(n, true_p_b)
p1, p2 = x_a / n, x_b / n
pbar = (x_a + x_b) / (2 * n)
z = (p2 - p1) / np.sqrt(pbar * (1 - pbar) * 2 / n)
p_val = 2 * stats.norm.sf(abs(z))
se_ci = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
lo, hi = (p2 - p1) - 1.96 * se_ci, (p2 - p1) + 1.96 * se_ci
print(f"\nresult: A {x_a}/{n} = {p1:.4f}, B {x_b}/{n} = {p2:.4f}")
print(f"z = {z:.2f}, p = {p_val:.4f}; absolute lift {p2 - p1:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]; relative lift {(p2 / p1 - 1) * 100:+.1f}%"
      f" (CI roughly [{lo / p1 * 100:+.1f}%, {hi / p1 * 100:+.1f}%])")
print("decision:", "ship" if p_val < alpha and lo > 0 else "do not ship")

# ---- 3. check the design by simulation: power at the MDE and false-positive rate under H0
reps = 4_000
xa = rng.binomial(n, p_a, reps); xb = rng.binomial(n, p_b_min, reps)
pb_ = (xa + xb) / (2 * n); zz = (xb / n - xa / n) / np.sqrt(pb_ * (1 - pb_) * 2 / n)
xa0 = rng.binomial(n, p_a, reps); xb0 = rng.binomial(n, p_a, reps)
pb0 = (xa0 + xb0) / (2 * n); zz0 = (xb0 / n - xa0 / n) / np.sqrt(pb0 * (1 - pb0) * 2 / n)
print(f"\nsimulated power at the MDE: {(np.abs(zz) > 1.96).mean():.3f}   false-positive rate under H0: {(np.abs(zz0) > 1.96).mean():.3f}")

# ---- 4. what daily peeking would have done: 14 looks, stop at the first p < 0.05
days = 14
per_day = n // days
def peek(p_b_true, sims=3_000):
    ca = rng.binomial(per_day, p_a, (sims, days)).cumsum(axis=1)
    cb = rng.binomial(per_day, p_b_true, (sims, days)).cumsum(axis=1)
    n_cum = np.arange(1, days + 1) * per_day
    pa_, pb_ = ca / n_cum, cb / n_cum
    pooled = (ca + cb) / (2 * n_cum)
    zd = (pb_ - pa_) / np.sqrt(pooled * (1 - pooled) * 2 / n_cum)
    sig = np.abs(zd) > 1.96
    stopped = sig.any(axis=1)
    first = np.argmax(sig, axis=1)
    lift_at_stop = (pb_ - pa_)[np.arange(sims), first]
    return stopped, first, lift_at_stop, zd

stopped0, first0, lift0, zd0 = peek(p_a)
print(f"\npeeking daily under H0 (no real effect): declared 'significant' at some point in {stopped0.mean():.3f} of experiments (nominal 0.05)")
print(f"   median stopping day among those: {np.median(first0[stopped0]) + 1:.0f}; mean |estimated lift| at stopping: {np.abs(lift0[stopped0]).mean():.4f} (truth 0)")
stopped1, first1, lift1, _ = peek(true_p_b)
early = stopped1 & (first1 < days - 1)
print(f"peeking daily under the true 10% lift: stopped before day 14 in {early.mean():.3f} of experiments;"
      f" mean estimated lift when stopping early {lift1[early].mean():+.4f} vs truth {true_p_b - p_a:+.4f} (winner's curse)")

fig, ax = plt.subplots(figsize=(8, 4))
for i in range(6):
    ax.plot(np.arange(1, days + 1), zd0[i], marker=".", lw=1)
ax.axhline(1.96, color="C3", ls="--"); ax.axhline(-1.96, color="C3", ls="--")
ax.set_xlabel("day of the experiment"); ax.set_ylabel("running z-statistic")
ax.set_title("Six null experiments watched daily: random walks that sometimes cross +/-1.96")
plt.show()
"""
ex.add_project(
    title="Mini-project: design and analyse an A/B test end to end",
    prompt_md=r"""
You are the analyst for a checkout redesign. Baseline conversion is 5.0%; the smallest lift worth shipping
is a **10% relative** improvement; the site has 5,000 visitors a day. Use `rng = np.random.default_rng(2024)`
and the two-proportion z-test throughout.

1. **Design.** Compute the minimum detectable absolute effect, the required $n$ per arm for $\alpha = 0.05$
   (two-sided) and power 0.8 using the proportion version of the sample-size formula,
   $n = (z_{1-\alpha/2} + z_{1-\beta})^2\,[\theta_1(1-\theta_1) + \theta_2(1-\theta_2)]/\delta^2$, and the number of
   days the test must run.
2. **Run and analyse.** Simulate the experiment once with a true 10% relative lift (`rng.binomial` for each
   arm). Report the z-statistic, the p-value, the absolute and relative lift with a 95% confidence interval
   (unpooled standard error) and a ship / do-not-ship decision.
3. **Validate the design.** Simulate 4,000 experiments at the minimum detectable effect and 4,000 under $H_0$;
   report the empirical power and false-positive rate.
4. **Peeking.** Split the run into 14 daily batches. Under $H_0$, simulate 3,000 experiments that check the
   z-test every day and stop at the first $p < 0.05$: report the false-positive rate, the median stopping day
   and the mean absolute "lift" reported by the stopped experiments. Under the true lift, report how often a
   peeking analyst would stop before day 14 and how inflated the estimated lift is at those early stops.
5. Plot the running z-statistic of a few null experiments with the $\pm 1.96$ lines.
""",
    rubric_md=r"""
* The sample size comes from the formula with the correct $z$ quantiles ($1.96$ and $0.84$) and is rounded up; the duration follows from it (about 12.5 days, so plan two full weeks).
* The analysis reports effect size **and** interval **and** p-value, and the decision rule is stated before the result is looked at.
* The validation simulation is vectorised (arrays of binomial counts, not Python loops over visitors) and reproduces power $\approx 0.8$ and false-positive rate $\approx 0.05$.
* The peeking simulation uses cumulative counts per day and the pooled standard error at each look; it shows a false-positive rate near 0.2 for 14 looks and an inflated effect estimate at early stops under the alternative.
* The figure has a title, axis labels and the $\pm 1.96$ boundaries.
* The write-up draws the right conclusion: fix the horizon in advance (or use a sequential boundary) and report the interval rather than a verdict.
""",
    solution_sketch_md=r"""
With $\theta_1 = 0.05$, $\theta_2 = 0.055$ and $\delta = 0.005$, the formula gives
$7.85 \times (0.0475 + 0.0520)/0.000025 \approx 31{,}231$ visitors per arm, about 62,500 in total — 12.5 days,
so a 14-day test. The single run with `default_rng(2024)` is instructive: the true lift *is* the MDE, yet the
observed lift is only $0.21$ percentage points with $z = 1.20$, $p = 0.23$ — a type II error, which a design with
80% power commits one time in five. Read the interval, not the verdict: $[-0.14,\ +0.57]$ points still contains
the $0.5$-point effect the test was built to detect, so the honest conclusion is "inconclusive", not "no
effect", and a longer run is justified. (Re-run with another seed and most runs give $z$ around 2.5 to 3 and an
interval like $[0.1, 0.9]$ points; even then the *relative* lift is pinned down only to within about
$\pm 7$ percentage points of relative lift — 80% power is power to detect, not to measure precisely.) The
validation simulation returns power close to $0.80$ and a false-positive rate close to $0.05$. Daily peeking
under the null declares significance at some point in roughly 22% of experiments (14 looks), typically
stopping around day 3 to 5 with an apparent lift of close to a percentage point that does not exist; under
the true lift, early stops report an inflated effect (about $0.8$ points against a truth of $0.5$) — the winner's
curse of optional stopping. The plot of running z-statistics shows null experiments wandering across
$\pm 1.96$ and back. The fix is procedural: pre-register the horizon and the primary metric, or use a
group-sequential boundary (Pocock's constant boundary is about $2.61$ for 14 looks; O'Brien–Fleming spends less
early) or an always-valid sequential test.
""",
    solution_code=PROJECT_CODE,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print(f"exercises: {len(ex.items)} items — {kinds.count('code')} coding, {kinds.count('concept')} conceptual, {kinds.count('project')} project")
print("difficulties:", [it["difficulty"] for it in ex.items])
