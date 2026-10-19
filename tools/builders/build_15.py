#!/usr/bin/env python3
"""Builder for Module 15 — Generalisation and Statistical Learning Theory.

Generates lesson.ipynb, exercises.ipynb and solutions.ipynb into
part-5-learning-theory/15-generalization-and-learning-theory. Hidden answers for
the exercise self-checks are computed here by *executing the solution code*.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import Notebook, ExerciseSet, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-5-learning-theory/15-generalization-and-learning-theory"


def safe_to_hash(x: float, decimals: int = 3, margin: float = 2e-4) -> float:
    """Assert that x is not within `margin` of a rounding boundary at `decimals` places."""
    frac = abs(x) * 10 ** decimals
    dist = abs(frac - np.floor(frac) - 0.5)   # distance (in units of the last digit) to the .5 boundary
    assert dist > margin * 10 ** decimals, f"hidden value {x!r} too close to a rounding boundary"
    return x


# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 15 — Generalisation and Statistical Learning Theory")

nb.md(r"""
This notebook turns the ideas of the module into experiments you can edit and re-run: a
memorising learner whose training error lies, the Hoeffding bound against the real tail of a
test accuracy, the optimism of picking the best of many models, shattering pictures and the
VC bound, the $1/n$ learning curve PAC theory predicts, margins and early stopping as capacity
control, double descent with random features, gradient descent finding the minimum-norm
interpolator, learning curves with a fitted power law, covariate shift, and a leakage trap.

Read the [concept guide](README.md) first, or keep it open alongside — every section here
mirrors a numbered section there. Everything is built **from scratch in NumPy**; scikit-learn
appears only in the last section, to show the leakage-proof `Pipeline`.
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
## 1. The generalisation question — a report card the model wrote itself

True risk $R(h) = \mathbb{E}[\ell(h(x), y)]$ is what you care about; the training error
$\hat R_S(h)$ is what you can see. For a hypothesis chosen *using* the sample the two can be
arbitrarily far apart. The extreme case is a **memoriser**: it stores every training point and
predicts the majority class elsewhere, so its training error is zero whatever the labels.

Data: $x \sim \text{Uniform}[0, 6]$, label $y = \mathbf{1}[x \ge 2.5]$ with 10 % of labels
flipped, so the Bayes risk is $0.1$. We compare the memoriser with an honest **threshold
learner** (ERM over all cut points) and estimate each one's *true* risk on a fresh sample of
200,000 points — something you can do with synthetic data and never with real data.
""")

nb.code(r"""
def make_data(n, g, flip=0.10):
    x = g.uniform(0, 6, n)
    y = (x >= 2.5).astype(int)
    flipped = g.random(n) < flip
    y[flipped] = 1 - y[flipped]
    return x, y

class Memoriser:
    def fit(self, x, y):
        self.table = dict(zip(x.tolist(), y.tolist()))
        self.majority = int(np.round(y.mean()))
        return self
    def predict(self, x):
        return np.array([self.table.get(v, self.majority) for v in x.tolist()])

class ThresholdLearner:                    # ERM over h_a(x) = 1[x >= a]
    def fit(self, x, y):
        xs = np.sort(x)
        cands = np.concatenate([[xs[0] - 1], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1]])
        errs = [np.mean((x >= a).astype(int) != y) for a in cands]
        self.a = cands[int(np.argmin(errs))]
        return self
    def predict(self, x):
        return (x >= self.a).astype(int)

x_tr, y_tr = make_data(50, rng)
x_fresh, y_fresh = make_data(200_000, rng)          # stands in for the population
print(f"{'learner':18s} {'training error':>15s} {'true risk (fresh)':>18s}")
for model in (Memoriser(), ThresholdLearner()):
    model.fit(x_tr, y_tr)
    tr = np.mean(model.predict(x_tr) != y_tr)
    te = np.mean(model.predict(x_fresh) != y_fresh)
    print(f"{type(model).__name__:18s} {tr:15.3f} {te:18.3f}")
print(f"\nthreshold found: a = {model.a:.3f}  (truth 2.5);  Bayes risk = 0.100")
""")

nb.md(r"""
The memoriser reports a perfect training score and has a true risk of about $0.43$ — the error
of always predicting the majority class ($P(y = 1) = 0.567$ here), which is what it does on every
point it has not seen. It has learned nothing. The threshold learner's training error (0.12) is close to its true risk (0.10)
because the class it searches is tiny: there is little room to fit the sample's accidents.
The rest of the module quantifies "little room".

**Try it yourself:** set `flip=0.3` in both `make_data` calls. The memoriser still scores
zero on training; what happens to the threshold learner's gap?
""")

# ---------------------------------------------------------------- Section 2
nb.md(r"""
## 2. Approximation error versus estimation error

Excess risk $= [R(h^*_{\mathcal H}) - R^*] + [R(\hat h_S) - R(h^*_{\mathcal H})]$: what the
best member of the class misses (approximation) plus what the finite sample costs (estimation).
We regress $y = \sin(2\pi x) + \varepsilon$, $\sigma = 0.3$, with polynomial classes of
increasing degree and $n = 20$ points, measuring each term directly:

* approximation error = population MSE of the best polynomial (fitted on a dense noiseless grid);
* estimation error = average excess risk of the least-squares fit over 200 training sets, minus the approximation error.
""")

nb.code(r"""
f_true = lambda x: np.sin(2 * np.pi * x)
grid = np.linspace(0, 1, 2001)
n_train, sigma, n_sets = 20, 0.3, 200

degrees = [0, 1, 2, 3, 4, 5, 7, 9]
approx, estim = [], []
for deg in degrees:
    best_c = np.polyfit(grid, f_true(grid), deg)                  # best in class (population)
    approx.append(np.mean((np.polyval(best_c, grid) - f_true(grid)) ** 2))
    risks = []
    for _ in range(n_sets):
        x = rng.uniform(0, 1, n_train)
        y = f_true(x) + sigma * rng.normal(size=n_train)
        c = np.polyfit(x, y, deg)
        risks.append(np.mean((np.polyval(c, grid) - f_true(grid)) ** 2))   # excess risk (noise excluded)
    estim.append(np.mean(risks) - approx[-1])
approx, estim = np.array(approx), np.array(estim)

print(f"{'degree':>6s} {'approximation':>14s} {'estimation':>11s} {'excess risk':>12s}")
for d, a, e in zip(degrees, approx, estim):
    print(f"{d:6d} {a:14.4f} {e:11.4f} {a + e:12.4f}")

plt.semilogy(degrees, approx + 1e-6, "s-", label="approximation error (bias)")
plt.semilogy(degrees, estim, "o-", label="estimation error (variance)")
plt.semilogy(degrees, approx + estim, "k--", label="excess risk = sum")
plt.title(f"Approximation vs estimation error, polynomial classes, n = {n_train}")
plt.xlabel("polynomial degree (capacity)"); plt.ylabel("mean squared error"); plt.legend(); plt.show()
""")

nb.md(r"""
Approximation error collapses once the class contains a good approximation of the sine (degree
3 and above); estimation error grows with degree and explodes past degree 5 on 20 points. Their
sum is the classical U-curve — the bias–variance trade-off of Module 13 in the language of
learning theory. Notice that the estimation error of degree 0 and 1 is not zero: even a class
that is far too small pays a sample-size price.
""")

# ---------------------------------------------------------------- Section 3
nb.md(r"""
## 3. Concentration: Hoeffding versus the real tail of a test accuracy

For a *fixed* classifier with true accuracy $p$, each test example is a Bernoulli$(p)$ trial,
and the measured accuracy $\hat p$ is a mean of $n$ variables in $[0, 1]$. Hoeffding:

$$P(|\hat p - p| \ge t) \le 2e^{-2nt^2}, \qquad \text{Chebyshev: } P(|\hat p - p| \ge t) \le \frac{p(1-p)}{nt^2}.$$

We simulate 20,000 test sets of size $n = 200$ for a model with $p = 0.7$ and compare the
empirical tail with both bounds.
""")

nb.code(r"""
n, p_true, n_rep = 200, 0.7, 20_000
acc = rng.binomial(n, p_true, size=n_rep) / n
ts = np.linspace(0.005, 0.12, 60)
empirical = np.array([np.mean(np.abs(acc - p_true) >= t) for t in ts])
hoeffding = np.minimum(1, 2 * np.exp(-2 * n * ts ** 2))
chebyshev = np.minimum(1, p_true * (1 - p_true) / (n * ts ** 2))

plt.semilogy(ts, empirical, "k-", lw=2, label="empirical tail (20,000 test sets)")
plt.semilogy(ts, hoeffding, "--", label="Hoeffding  2 exp(-2 n t^2)")
plt.semilogy(ts, chebyshev, ":", label="Chebyshev  p(1-p)/(n t^2)")
plt.ylim(1e-4, 1.5)
plt.title(f"P(|accuracy - p| >= t) for n = {n} test examples, p = {p_true}")
plt.xlabel("deviation t"); plt.ylabel("probability (log scale)"); plt.legend(); plt.show()

for t in (0.05, 0.08, 0.10):
    i = np.argmin(np.abs(ts - t))
    print(f"t = {t:.2f}: empirical {empirical[i]:.4f}   Hoeffding {hoeffding[i]:.4f}   Chebyshev {chebyshev[i]:.4f}")

eps, delta = 0.02, 0.05
print(f"\nsample size for +-{eps} at {1-delta:.0%}:  Hoeffding n >= {int(np.ceil(np.log(2/delta)/(2*eps**2))):,}"
      f"   Chebyshev n >= {int(np.ceil(1/(4*delta*eps**2) - 1e-9)):,}   normal approx n ~ {int(np.ceil(1.96**2*0.25/eps**2 - 1e-9)):,}")
""")

nb.md(r"""
Both bounds are valid (they sit above the black curve everywhere) and both are loose. For small
$t$ Chebyshev is actually tighter; once $nt^2$ exceeds about one, Hoeffding's exponential
decay takes over and leaves Chebyshev's $1/t^2$ far behind. That exponential tail is what makes
the union bound affordable in the next section: multiplying an exponentially small probability
by a large number of hypotheses costs only a logarithm.

**Try it yourself:** change `p_true` to `0.95`. The empirical tail tightens a lot (the
variance $p(1-p)$ shrinks), Chebyshev follows it, Hoeffding does not — it only knows the range.
""")

# ---------------------------------------------------------------- Section 4
nb.md(r"""
## 4. The union bound and the optimism of the best of $m$ models

Evaluate $m$ models on a validation set of $n$ examples and keep the best score. Even if all
$m$ have the *same* true accuracy $a$, the maximum of $m$ noisy scores sits above $a$. Two
predictions:

* **expected optimism** $\approx \mathbb{E}[\max \text{ of } m \text{ standard normals}] \cdot \sqrt{a(1-a)/n}$
  (the normal approximation, with $0.56, 1.16, 1.54, 2.25, 2.51$ for $m = 2, 5, 10, 50, 100$);
* **union-bound envelope**: $P(\max_j \hat a_j - a \ge t) \le m\,e^{-2nt^2}$, so with probability
  $\ge 0.95$ the optimism is below $\sqrt{\ln(m/0.05)/(2n)}$.
""")

nb.code(r"""
from scipy import stats, integrate

def expected_max_normal(m):
    f = lambda z: z * m * stats.norm.pdf(z) * stats.norm.cdf(z) ** (m - 1)
    return integrate.quad(f, -10, 10)[0]

a_true, n_val, n_rep = 0.8, 1000, 2000
ms = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200])
se = np.sqrt(a_true * (1 - a_true) / n_val)
optimism = np.zeros(len(ms)); q95 = np.zeros(len(ms))
for j, m in enumerate(ms):
    best = rng.binomial(n_val, a_true, size=(n_rep, m)).max(axis=1) / n_val
    optimism[j] = np.mean(best) - a_true
    q95[j] = np.quantile(best, 0.95) - a_true
theory = np.array([expected_max_normal(m) for m in ms]) * se
envelope = np.sqrt(np.log(ms / 0.05) / (2 * n_val))

print(f"true accuracy of every model: {a_true}, validation size {n_val}, SE of one score {se:.4f}\n")
print(f"{'m':>4s} {'mean optimism':>14s} {'E[max] theory':>14s} {'95% quantile':>13s} {'union bound':>12s}")
for m, o, t, q, e in zip(ms, optimism, theory, q95, envelope):
    print(f"{m:4d} {o:14.4f} {t:14.4f} {q:13.4f} {e:12.4f}")

plt.semilogx(ms, optimism, "o-", label="simulated mean optimism of the best")
plt.semilogx(ms, theory, "--", label="E[max of m normals] * SE")
plt.semilogx(ms, q95, "s-", label="simulated 95% quantile")
plt.semilogx(ms, envelope, "k:", label="union bound sqrt(ln(m/0.05)/(2n))")
plt.title("Winner's curse: how much the best of m equal models looks better")
plt.xlabel("number of models m compared"); plt.ylabel("best validation accuracy - true accuracy"); plt.legend(); plt.show()
""")

nb.md(r"""
Picking the best of ten identical models buys about two points of phantom accuracy on a
1,000-example validation set; the best of a hundred, over three. The normal-approximation line
tracks the simulation closely, and the union-bound envelope — distribution-free, worst-case —
sits above the simulated 95 % quantile everywhere while growing with the same
$\sqrt{\ln m}$ shape. This is `GridSearchCV.best_score_` being optimistic, and the reason the
final model gets a fresh test set (Section 10).

**Try it yourself:** set `n_val = 10_000`. The optimism shrinks by $\sqrt{10} \approx 3.2$ —
the only cure for the winner's curse, besides looking less often, is a bigger validation set.
""")

# ---------------------------------------------------------------- Section 5
nb.md(r"""
## 5. Shattering, the growth function and the VC bound

A class **shatters** $n$ points if it realises all $2^n$ labellings of them; the VC dimension
is the largest such $n$. For lines in the plane, three points in general position are shattered
(all 8 patterns) and four are not: the XOR labelling of four points in convex position cannot be
cut by any line. So $d_{\mathrm{VC}} = 3 = d + 1$.

Below we check every labelling by brute force: for each pattern we ask whether a linear
classifier $\operatorname{sign}(w^\top x + b)$ separates it, by solving a small linear feasibility
problem (does some $(w, b)$ satisfy $y_i(w^\top x_i + b) \ge 1$ for all $i$?).
""")

nb.code(r"""
from itertools import product
from scipy.optimize import linprog

def linearly_separable(X, y):
    # feasibility of  y_i (w.x_i + b) >= 1  <=>  -y_i (w.x_i + b) <= -1 ; variables (w1, w2, b)
    A = -(y[:, None] * np.column_stack([X, np.ones(len(X))]))
    res = linprog(c=np.zeros(3), A_ub=A, b_ub=-np.ones(len(X)), bounds=[(None, None)] * 3, method="highs")
    return res.status == 0

def realised_dichotomies(X):
    return [np.array(lab) for lab in product([-1, 1], repeat=len(X)) if linearly_separable(X, np.array(lab))]

P3 = np.array([[0., 0.], [1., 0.], [0.5, 0.9]])
P4 = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
dich3, dich4 = realised_dichotomies(P3), realised_dichotomies(P4)
print(f"3 points: {len(dich3)} of {2**3} labellings realised by lines -> shattered: {len(dich3) == 8}")
print(f"4 points (square): {len(dich4)} of {2**4} realised -> shattered: {len(dich4) == 16}")
missing = [lab for lab in product([-1, 1], repeat=4) if not linearly_separable(P4, np.array(lab))]
print("labellings no line can realise on the square:", missing)

fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for ax, lab in zip(axes.ravel(), dich3):
    ax.scatter(P3[lab == 1, 0], P3[lab == 1, 1], c="tab:red", s=80, marker="o", label="+1")
    ax.scatter(P3[lab == -1, 0], P3[lab == -1, 1], c="tab:blue", s=80, marker="x", label="-1")
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.2); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("".join("+" if l == 1 else "-" for l in lab)); ax.set_xlabel("x1"); ax.set_ylabel("x2")
fig.suptitle("Three points in the plane: all 8 labellings are linearly separable (shattered)")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Every one of the eight patterns on three points is separable; on the four corners of a square
exactly the two XOR patterns ($+,-,+,-$ and $-,+,-,+$) are not. Fourteen of sixteen is also
what the general-position formula $\Pi(n) = n^2 - n + 2$ predicts for lines on $n = 4$ points.

### Growth function versus $2^n$, Sauer's lemma, and the bounds as curves in $n$

Sauer's lemma: if $d_{\mathrm{VC}} = d$, then $\Pi(n) \le \sum_{i=0}^{d}\binom{n}{i} \le (en/d)^d$ — a
polynomial, not an exponential. We count the dichotomies lines realise on random point sets
directly (every realisable labelling of points in general position can be obtained from a line
through two of the points, perturbed) and compare. Then we draw the uniform-deviation bounds
themselves as functions of $n$: finite class $\sqrt{\ln(2|\mathcal H|/\delta)/(2n)}$, VC class
$\sqrt{2d\ln(en/d)/n} + \sqrt{\ln(1/\delta)/(2n)}$.
""")

nb.code(r"""
from math import comb

def count_line_dichotomies(P):
    # all labellings realisable by a line through points i, j (with i, j labelled either way), plus the two constant ones
    n = len(P); seen = {tuple([1] * n), tuple([0] * n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = P[j] - P[i]; normal = np.array([-d[1], d[0]])
            side = ((P - P[i]) @ normal > 0).astype(int)
            for li in (0, 1):
                for lj in (0, 1):
                    lab = side.copy(); lab[i], lab[j] = li, lj
                    seen.add(tuple(lab)); seen.add(tuple(1 - lab))
    return len(seen)

ns_pts = np.arange(2, 13)
counted = np.array([count_line_dichotomies(rng.normal(size=(n, 2))) for n in ns_pts])
formula = ns_pts ** 2 - ns_pts + 2
sauer = np.array([sum(comb(int(n), i) for i in range(4)) for n in ns_pts])       # d = 3
print(f"{'n':>3s} {'lines realise':>14s} {'n^2-n+2':>8s} {'Sauer bound (d=3)':>18s} {'2^n':>6s}")
for n, c, f_, s_ in zip(ns_pts, counted, formula, sauer):
    print(f"{n:3d} {c:14d} {f_:8d} {s_:18d} {2**n:6d}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].semilogy(ns_pts, 2.0 ** ns_pts, "k--", label="2^n (all labellings)")
axes[0].semilogy(ns_pts, sauer, "s-", label="Sauer's lemma, d = 3")
axes[0].semilogy(ns_pts, counted, "o-", label="lines in the plane: realised dichotomies")
axes[0].set_title("Growth function of lines in 2-D: polynomial, not exponential")
axes[0].set_xlabel("number of points n"); axes[0].set_ylabel("number of dichotomies (log)"); axes[0].legend(fontsize=8)

ns = np.logspace(2, 6, 100); delta = 0.05
for H in (4_000, 1_000_000):
    axes[1].loglog(ns, np.sqrt(np.log(2 * H / delta) / (2 * ns)), label=f"finite class, |H| = {H:,}")
for d in (3, 101):
    vc = np.sqrt(2 * d * np.log(np.e * ns / d) / ns) + np.sqrt(np.log(1 / delta) / (2 * ns))
    axes[1].loglog(ns, vc, "--", label=f"VC bound, d = {d}")
axes[1].loglog(ns, np.sqrt(np.log(2 / delta) / (2 * ns)), "k:", label="one fixed hypothesis (Hoeffding)")
axes[1].axhline(1, color="gray", lw=0.8)
axes[1].set_title("Uniform deviation bounds vs sample size (delta = 0.05)")
axes[1].set_xlabel("n"); axes[1].set_ylabel("bound on sup |R(h) - R_hat(h)|"); axes[1].legend(fontsize=7)
plt.tight_layout(); plt.show()

for n in (1_000, 10_000, 100_000):
    vc3 = np.sqrt(2 * 3 * np.log(np.e * n / 3) / n) + np.sqrt(np.log(1 / delta) / (2 * n))
    print(f"n = {n:>7,d}: Hoeffding {np.sqrt(np.log(2/delta)/(2*n)):.4f}   |H| = 4,000: {np.sqrt(np.log(8000/delta)/(2*n)):.4f}   VC d=3: {vc3:.4f}")
""")

nb.md(r"""
Left: up to $n = 3$ the class realises everything; from $n = 4$ it falls further and further
below $2^n$, and Sauer's cubic bound holds with room to spare. Learning is possible because the
number of *behaviours* a class can show on a sample is tiny compared with the number of
labellings, so the union bound over behaviours (rather than hypotheses) is affordable.

Right: all bounds fall like $1/\sqrt{n}$ (slope $-1/2$ on the log–log plot); the price of a
class is a vertical offset — a factor, not a change of rate. A million hypotheses cost barely
twice the deviation of one. The VC bound for linear classifiers in 100 dimensions ($d = 101$)
is above 1, i.e. vacuous, until $n \approx 630$, and still $0.35$ at $n = 10{,}000$ — true to
its worst-case nature, and far more pessimistic than what you will see in Section 6.
""")

# ---------------------------------------------------------------- Section 6
nb.md(r"""
## 6. PAC learning: the realisable rate is $1/n$

PAC theory says the sample complexity of a class with finite VC dimension scales like
$d/\varepsilon$ when some hypothesis is perfect (realisable), and like $d/\varepsilon^2$ otherwise.
Turned around: the error of ERM should fall like $1/n$ in the realisable case.

The class of intervals ($d_{\mathrm{VC}} = 2$) makes this exact. The ERM rule "tightest interval
around the positive examples" has true risk equal to the length of the two uncovered slivers,
which we can compute exactly instead of estimating.
""")

nb.code(r"""
a_true, b_true = 0.3, 0.7                         # target interval; x ~ Uniform[0, 1]

def erm_interval(x, y):
    pos = x[y == 1]
    return (pos.min(), pos.max()) if len(pos) else (0.0, 0.0)

ns = np.array([10, 20, 40, 80, 160, 320, 640, 1280])
risk = np.zeros(len(ns)); q95 = np.zeros(len(ns))
for i, n in enumerate(ns):
    r = []
    for _ in range(400):
        x = rng.uniform(0, 1, n); y = ((x >= a_true) & (x <= b_true)).astype(int)
        a, b = erm_interval(x, y)
        r.append((a - a_true) + (b_true - b))          # exact true risk of the learned interval
    risk[i], q95[i] = np.mean(r), np.quantile(r, 0.95)
slope = np.polyfit(np.log(ns), np.log(risk), 1)[0]

plt.loglog(ns, risk, "o-", label="mean true risk of ERM (intervals)")
plt.loglog(ns, q95, "s--", label="95% quantile of the risk")
plt.loglog(ns, 2 / ns, "k:", label="2/n")
plt.title(f"Realisable learning curve: risk falls like 1/n (fitted slope {slope:.2f})")
plt.xlabel("training set size n"); plt.ylabel("true risk"); plt.legend(); plt.show()
print("n       :", ns)
print("risk    :", np.round(risk, 4))
print("risk * n:", np.round(risk * ns, 2), "  -> roughly constant, as 1/n predicts")
""")

nb.md(r"""
The fitted slope is about $-1$: doubling the data halves the error, the realisable PAC rate.
With label noise the same experiment gives slope $-1/2$ (the agnostic rate), because the
learner must then *average out* noise rather than merely *locate* a boundary. Compare the
actual risk at $n = 1{,}000$ (about $0.002$) with the VC bound's $0.24$ from Section 5: the
bound's shape is right, its constant is a hundred times too pessimistic.

**Try it yourself:** flip 10 % of the labels inside the loop (`y[rng.random(n) < 0.1] ^= 1`)
and replace the exact risk by an estimate on fresh points. What slope do you get now?
""")

# ---------------------------------------------------------------- Section 7
nb.md(r"""
## 7. Capacity control: margins and early stopping

### Margin, not dimension

For linear separators with $\|x\| \le r$ and normalised margin $\rho = \min_i y_i w^\top x_i / \|w\|$,
the margin bound says the capacity is $(r/\rho)^2$ — independent of the dimension. We generate
separable data in $\mathbb{R}^{10}$ with a true margin of $0.15$, collect several hundred
*different* separators that all classify the 30 training points perfectly, and plot each one's
test error against its margin. Then we run gradient descent on the logistic loss, whose direction
converges to the maximum-margin separator.
""")

nb.code(r"""
def separable_data(n, g, d=10, margin=0.15):
    w_star = g.normal(size=d); w_star /= np.linalg.norm(w_star)
    X = g.normal(size=(8 * n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)   # r = 1
    X = X[np.abs(X @ w_star) >= margin][:n]
    return X, np.sign(X @ w_star), w_star

def perceptron(X, y, order):
    w = np.zeros(X.shape[1])
    for _ in range(200):
        mistakes = 0
        for i in order:
            if y[i] * (X[i] @ w) <= 0:
                w += y[i] * X[i]; mistakes += 1
        if mistakes == 0:
            break
    return w

def margin_of(w, X, y):
    return np.min(y * (X @ w)) / np.linalg.norm(w)

X_tr, y_tr, w_star = separable_data(30, rng)
X_te = rng.normal(size=(5000, 10)); X_te /= np.linalg.norm(X_te, axis=1, keepdims=True)
y_te = np.sign(X_te @ w_star)

separators = [perceptron(X_tr, y_tr, rng.permutation(30)) for _ in range(300)]
for _ in range(3000):                                 # random consistent separators in dual form
    alpha = rng.exponential(size=30) * (rng.random(30) < 0.3)
    w = (alpha * y_tr) @ X_tr
    if np.linalg.norm(w) > 0 and np.all(y_tr * (X_tr @ w) > 0):
        separators.append(w)
separators = np.array(separators)
margins = np.array([margin_of(w, X_tr, y_tr) for w in separators])
test_err = np.array([np.mean(np.sign(X_te @ w) != y_te) for w in separators])

def logistic_gd(X, y, steps, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w += lr * (X.T @ (y / (1 + np.exp(y * (X @ w))))) / len(y)
    return w

plt.scatter(margins, test_err, s=10, alpha=0.5, label=f"{len(separators)} separators with zero training error")
for steps, c in ((50, "tab:orange"), (500, "tab:green"), (5000, "tab:red")):
    w = logistic_gd(X_tr, y_tr, steps)
    plt.scatter(margin_of(w, X_tr, y_tr), np.mean(np.sign(X_te @ w) != y_te), s=120, marker="*", c=c,
                label=f"logistic-loss GD, {steps} steps")
plt.axvline(margin_of(w_star, X_tr, y_tr), color="gray", ls=":", label="true separator's margin")
plt.title("Same training error, different margins: larger margin, lower test error")
plt.xlabel("normalised margin on the training set"); plt.ylabel("test error"); plt.legend(fontsize=8); plt.show()
print(f"correlation(margin, test error) = {np.corrcoef(margins, test_err)[0, 1]:.3f}")
for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.2)):
    m = (margins >= lo) & (margins < hi)
    print(f"margin in [{lo:.2f}, {hi:.2f}): {m.sum():3d} separators, mean test error {test_err[m].mean():.3f}")
""")

nb.md(r"""
Every separator here has *zero* training error and lives in the same 10-dimensional class, so
VC dimension cannot tell them apart. Margin can: the ones that separate the data with room to
spare generalise better, and gradient descent on the logistic loss drifts towards the
maximum-margin direction as training continues (the stars move right and down). This is
Section 7 of the guide in one picture: capacity $(r/\rho)^2$, not $d + 1$.

### Early stopping is ridge in disguise

Gradient descent on least squares from $w_0 = 0$ with step $\eta$ reaches, after $t$ steps,
the fraction $1 - (1 - \eta\sigma_i^2)^t$ of the OLS coefficient along each singular direction
of $X$. Ridge reaches $\sigma_i^2/(\sigma_i^2 + \lambda)$. We measure the GD fractions directly.
""")

nb.code(r"""
X = rng.normal(size=(100, 20)) @ np.diag(np.logspace(0, -1.5, 20))   # singular values from ~10 down to ~0.3
beta = rng.normal(size=20)
y = X @ beta + 0.5 * rng.normal(size=100)
U, s, Vt = np.linalg.svd(X, full_matrices=False)
w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
eta = 1.0 / s.max() ** 2

w = np.zeros(20); snapshots = {}
for t in range(1, 2001):
    w -= eta * X.T @ (X @ w - y)
    if t in (10, 100, 1000):
        snapshots[t] = w.copy()

for t, w_t in snapshots.items():
    measured = (Vt @ w_t) / (Vt @ w_ols)                        # fraction of OLS reached, per singular direction
    predicted = 1 - (1 - eta * s ** 2) ** t
    plt.plot(s, measured, "o", label=f"GD, t = {t} (measured)")
    plt.plot(s, predicted, "k-", lw=0.8)
    lam = 1 / (eta * t)
    plt.plot(s, s ** 2 / (s ** 2 + lam), "--", label=f"ridge, lambda = 1/(eta t) = {lam:.1f}")
    print(f"t = {t:4d}: max |measured - predicted filter| = {np.abs(measured - predicted).max():.2e}")
plt.xscale("log")
plt.title("Early stopping filter 1 - (1 - eta sigma^2)^t  vs  ridge filter sigma^2/(sigma^2 + lambda)")
plt.xlabel("singular value sigma_i of X"); plt.ylabel("fraction of the OLS coefficient kept"); plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
The measured fractions sit exactly on the predicted curve $1 - (1 - \eta\sigma_i^2)^t$, and that
curve has the same shape and the same cut-off as ridge with $\lambda \approx 1/(\eta t)$:
directions with large singular values are fitted first, the weakly determined directions —
the ones that carry mostly noise — are fitted last. Stopping early is choosing $\lambda$.

**Try it yourself:** make the noise ten times larger (`5.0 * rng.normal`) and plot the test
error of `snapshots[t]` on fresh data against $t$: it will go down and then up, the
U-curve in the time domain.
""")

# ---------------------------------------------------------------- Section 8
nb.md(r"""
## 8. Double descent and the implicit bias of gradient descent

### Double descent with random ReLU features

Features $\phi_j(x) = \max(0, w_j^\top x + b_j)$ with random frozen $w_j, b_j$ — a one-hidden-layer
network whose first layer is never trained. We fit the output weights by *minimum-norm* least
squares (`lstsq`, which is the pseudo-inverse) and vary the number of features $p$ across the
interpolation threshold $p = n$. Target: $f(x) = \sin x_1 + \tfrac12 x_2 x_3$ in $\mathbb{R}^5$,
$n = 40$ training points, noise $\sigma = 0.3$. The variance of the test target is about $0.66$,
so predicting zero would score roughly that.
""")

nb.code(r"""
def target(X):
    return np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]

def relu_features(X, W, b):
    return np.maximum(0.0, X @ W + b)

n_train, n_test, d_in, sigma = 40, 400, 5, 0.3
ps = np.unique(np.concatenate([np.arange(2, 60, 2), np.arange(60, 401, 20)]))
reps = 20
test_mse = np.zeros((reps, len(ps))); train_mse = np.zeros_like(test_mse); w_norm = np.zeros_like(test_mse)
for r in range(reps):
    X_tr = rng.normal(size=(n_train, d_in)); y_tr = target(X_tr) + sigma * rng.normal(size=n_train)
    X_te = rng.normal(size=(n_test, d_in)); y_te = target(X_te)
    W = rng.normal(size=(d_in, ps.max())); b = rng.uniform(-1, 1, ps.max())
    for j, p in enumerate(ps):
        Phi, Phi_te = relu_features(X_tr, W[:, :p], b[:p]), relu_features(X_te, W[:, :p], b[:p])
        w = np.linalg.lstsq(Phi, y_tr, rcond=None)[0]           # minimum-norm least squares
        test_mse[r, j] = np.mean((Phi_te @ w - y_te) ** 2)
        train_mse[r, j] = np.mean((Phi @ w - y_tr) ** 2)
        w_norm[r, j] = np.linalg.norm(w)
med_test, med_train, med_norm = (np.median(a, axis=0) for a in (test_mse, train_mse, w_norm))

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.semilogy(ps, med_test, "o-", label="test MSE (median over 20 runs)")
ax1.semilogy(ps, np.maximum(med_train, 1e-3), "s-", label="training MSE (floored at 1e-3; exactly 0 for p >= n)")
ax1.axvline(n_train, color="gray", ls=":", label="interpolation threshold p = n")
ax1.set_ylim(5e-4, 200)
ax1.set_xlabel("number of random ReLU features p"); ax1.set_ylabel("mean squared error (log)")
ax2 = ax1.twinx(); ax2.semilogy(ps, med_norm, "^--", color="tab:red", alpha=0.6, label="||w|| of the min-norm solution")
ax2.set_ylabel("||w||", color="tab:red")
ax1.set_title(f"Double descent: random-feature regression, n = {n_train}")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8); plt.show()

for p in (10, 20, 36, 40, 44, 80, 200, 400):
    j = int(np.where(ps == p)[0][0])
    print(f"p = {p:3d}: median test MSE {med_test[j]:8.3f}   train MSE {med_train[j]:.1e}   ||w|| {med_norm[j]:6.2f}")
under = ps < n_train
print(f"\nbest under-parameterised test MSE {med_test[under].min():.3f} at p = {ps[under][np.argmin(med_test[under])]};"
      f"   at p = 400: {med_test[-1]:.3f}")
""")

nb.md(r"""
The classical U-curve is there for $p < n$: test error is minimal around $p \approx 18$ and
rises as $p \to n$. At $p = n$ the design matrix is square and nearly singular, the minimum-norm
solution must use enormous weights to interpolate, and the test error explodes (training error
is exactly zero from here on). Beyond the threshold the extra features give the interpolator
room to fit the data with *small* weights again — $\|w\|$ falls steadily — and the test error
descends a second time, ending **below** the best under-parameterised model. The red curve is
the mechanism: past the peak, more parameters mean a smaller, smoother interpolator.

**Try it yourself:** set `sigma = 0.0`. Without label noise the peak at $p = n$ shrinks
dramatically — the spike is the noise being interpolated with huge weights.

### Gradient descent from zero finds the minimum-norm interpolator

With $p > n$ there are infinitely many $w$ with $Xw = y$. Gradient descent on
$\tfrac12\|Xw - y\|^2$ from $w_0 = 0$ only ever adds multiples of rows of $X$, so it stays in
the row space and converges to the unique interpolator there: $X^{+}y$.
""")

nb.code(r"""
n, p = 20, 50
X = rng.normal(size=(n, p)); w_true = rng.normal(size=p); y = X @ w_true     # noiseless, p > n
w_pinv = np.linalg.pinv(X) @ y
lr = 1.0 / np.linalg.norm(X, 2) ** 2

def gd_least_squares(X, y, w0, lr, steps=3000):
    w = w0.copy(); dist = []
    for _ in range(steps):
        w -= lr * X.T @ (X @ w - y)
        dist.append(np.linalg.norm(w - w_pinv))
    return w, np.array(dist)

w_gd0, dist0 = gd_least_squares(X, y, np.zeros(p), lr)
w0 = rng.normal(size=p)
w_gd1, dist1 = gd_least_squares(X, y, w0, lr)
null_part = w0 - np.linalg.pinv(X) @ (X @ w0)       # component of w0 in the null space of X

print(f"GD from zero:        max |w_GD - pinv(X) y| = {np.abs(w_gd0 - w_pinv).max():.1e},  ||w|| = {np.linalg.norm(w_gd0):.4f}")
print(f"pinv solution:       ||w|| = {np.linalg.norm(w_pinv):.4f}   (w_true has ||w|| = {np.linalg.norm(w_true):.4f} and is NOT recovered)")
print(f"GD from random w0:   interpolates? {np.allclose(X @ w_gd1, y)}   ||w|| = {np.linalg.norm(w_gd1):.4f}")
print(f"                     max |w_GD - (pinv(X) y + null-space part of w0)| = {np.abs(w_gd1 - (w_pinv + null_part)).max():.1e}")
plt.semilogy(dist0, label="start at 0")
plt.semilogy(dist1, label="start at random w0")
plt.title("Distance of the GD iterate to the minimum-norm solution pinv(X) y")
plt.xlabel("gradient step"); plt.ylabel("||w_t - pinv(X) y||"); plt.legend(); plt.show()
""")

nb.md(r"""
From zero, GD lands on `pinv(X) @ y` to machine precision: the optimiser *is* the regulariser.
From a random start it still interpolates, but it keeps the null-space component of its starting
point forever (the gradient can never see it), so the limit is the minimum-norm solution plus
that frozen component — larger in norm and no better on new data. Initialisation scale is a
regularisation choice too.
""")

# ---------------------------------------------------------------- Section 8b: learning curves / power law
nb.md(r"""
### Learning curves and a fitted power law

Test error against $n$ for two polynomial classes on $y = \sin(2\pi x) + \varepsilon$. For a
well-specified class with $p$ parameters, theory says the generalisation *gap* (test minus train
MSE) is about $2\sigma^2 p/n$, i.e. a power law with exponent $-1$. A misspecified class sees
its own approximation error as extra noise, so the fair comparison uses the residual variance
$\sigma_{\text{eff}}^2 = \sigma^2 + \text{approximation error}$ in place of $\sigma^2$. We fit
$c\,n^{-\gamma}$ to the measured gap and overlay $2\sigma_{\text{eff}}^2 p/n$.
""")

nb.code(r"""
f_true = lambda x: np.sin(2 * np.pi * x)
sigma = 0.3
ns = np.array([20, 40, 80, 160, 320, 640])
grid = np.linspace(0, 1, 2001)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for deg, c in ((1, "tab:blue"), (5, "tab:red")):
    approx_err = np.mean((np.polyval(np.polyfit(grid, f_true(grid), deg), grid) - f_true(grid)) ** 2)
    sigma_eff2 = sigma ** 2 + approx_err                      # what a misspecified model sees as noise
    test, gap = [], []
    for n in ns:
        te, g = [], []
        for _ in range(100):
            x = rng.uniform(0, 1, n); y = f_true(x) + sigma * rng.normal(size=n)
            coef = np.polyfit(x, y, deg)
            xt = rng.uniform(0, 1, 2000); yt = f_true(xt) + sigma * rng.normal(size=2000)
            tr_e = np.mean((np.polyval(coef, x) - y) ** 2); te_e = np.mean((np.polyval(coef, xt) - yt) ** 2)
            te.append(te_e); g.append(te_e - tr_e)
        test.append(np.mean(te)); gap.append(np.mean(g))
    test, gap = np.array(test), np.array(gap)
    slope, icpt = np.polyfit(np.log(ns), np.log(gap), 1)
    axes[0].plot(ns, test, "o-", color=c, label=f"degree {deg}")
    axes[1].loglog(ns, gap, "o", color=c, label=f"degree {deg}: gap ~ {np.exp(icpt):.1f} n^({slope:.2f})")
    axes[1].loglog(ns, np.exp(icpt) * ns ** slope, "-", color=c, lw=0.8)
    axes[1].loglog(ns, 2 * sigma_eff2 * (deg + 1) / ns, ":", color=c, lw=1.2, label=f"theory 2 sigma_eff^2 p / n, p = {deg+1}")
    print(f"degree {deg}: approximation error {approx_err:.4f}, sigma_eff^2 = {sigma_eff2:.4f};  test MSE {np.round(test, 3)};"
          f"  fitted gap ~ {np.exp(icpt):.2f} n^({slope:.2f}) vs theory {2 * sigma_eff2 * (deg + 1):.2f} / n")
axes[0].axhline(sigma ** 2, color="gray", ls=":", label="noise floor sigma^2")
axes[0].set_title("Learning curves: test MSE vs n"); axes[0].set_xlabel("n"); axes[0].set_ylabel("test MSE"); axes[0].legend()
axes[1].set_title("Generalisation gap follows a power law"); axes[1].set_xlabel("n"); axes[1].set_ylabel("test MSE - train MSE"); axes[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The straight line (degree 1) plateaus far above the noise floor: its approximation error
($\approx 0.20$) never goes away. Degree 5 is misspecified too, but barely, and its curve
approaches $\sigma^2$. Both gaps follow power laws with exponents near $-1$, and for $n \ge 80$
they sit on the $2\sigma_{\text{eff}}^2 p/n$ line once the approximation error is counted as noise.
The fitted constants come out larger than the theory's because the smallest sample sizes pull
the fit up: with a uniform design and few points the variance term exceeds its $p/n$
approximation, and $p = 6$ parameters on $n = 20$ points is already near the $p \approx n$
instability of Section 8. The large-language-model scaling laws
of the guide — $L = E + A/N^\alpha + B/D^\beta$ — are this picture with far shallower
exponents ($\beta \approx 0.28$) on far harder targets.
""")

# ---------------------------------------------------------------- Section 9
nb.md(r"""
## 9. Covariate shift: the same function, a different population

Train on $x \sim \mathcal N(0.3, 0.5^2)$, deploy on $x \sim \mathcal N(1.0, 0.25^2)$; the regression
function $f(x) = x^3 - x$ never changes, so this is pure **covariate shift**. A straight line is
misspecified, and the best line for one input distribution is a bad line for the other: around
$x = 0.3$ the cubic is almost flat, around $x = 1$ it climbs steeply.
**Importance weighting** — multiplying each training loss by $w(x) = p_{\text{test}}(x)/p_{\text{train}}(x)$ —
turns the training ERM into an unbiased estimate of the test risk.
""")

nb.code(r"""
from scipy.stats import norm

f_shift = lambda x: x ** 3 - x
mu_tr, sd_tr, mu_te, sd_te, n = 0.3, 0.5, 1.0, 0.25, 100
mse = {"in-distribution test": [], "shifted test, unweighted fit": [], "shifted test, importance-weighted fit": []}
for _ in range(200):
    x_tr = rng.normal(mu_tr, sd_tr, n); y_tr = f_shift(x_tr) + 0.3 * rng.normal(size=n)
    x_same, x_shift = rng.normal(mu_tr, sd_tr, 4000), rng.normal(mu_te, sd_te, 4000)
    c = np.polyfit(x_tr, y_tr, 1)
    w_imp = norm.pdf(x_tr, mu_te, sd_te) / norm.pdf(x_tr, mu_tr, sd_tr)        # density ratio
    c_w = np.polyfit(x_tr, y_tr, 1, w=np.sqrt(w_imp))                            # weighted least squares
    mse["in-distribution test"].append(np.mean((np.polyval(c, x_same) - f_shift(x_same)) ** 2))
    mse["shifted test, unweighted fit"].append(np.mean((np.polyval(c, x_shift) - f_shift(x_shift)) ** 2))
    mse["shifted test, importance-weighted fit"].append(np.mean((np.polyval(c_w, x_shift) - f_shift(x_shift)) ** 2))
for k, v in mse.items():
    print(f"{k:40s} mean MSE {np.mean(v):.4f}  (sd over runs {np.std(v):.4f})")
print(f"effective sample size of the last weighted fit: {w_imp.sum()**2 / (w_imp**2).sum():.1f} of {n}")

xs = np.linspace(-1, 2, 300)
plt.plot(xs, f_shift(xs), "k", label="true f(x) = x^3 - x")
plt.scatter(x_tr, y_tr, s=10, alpha=0.5, label="training data  x ~ N(0.3, 0.5^2)")
plt.plot(xs, np.polyval(c, xs), "--", label="unweighted line")
plt.plot(xs, np.polyval(c_w, xs), "-.", label="importance-weighted line")
plt.fill_between(xs, -2, 4, where=(np.abs(xs - mu_te) < 2 * sd_te), color="tab:green", alpha=0.08, label="deployment region x ~ N(1, 0.25^2)")
plt.ylim(-1.2, 3.5); plt.title("Covariate shift: a misspecified model fitted for the wrong population")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
The unweighted line is tuned to where the training data are dense ($x \approx 0.3$, where the
cubic is nearly flat) and is almost horizontal; in the deployment region the true function has
slope about $2$, and the shifted-test MSE is more than double the in-distribution one. Re-weighting
the training points by the density ratio fits the line for the deployment population and cuts
the shifted error by a factor of about five, at the cost of a smaller effective sample size
(about 20 of 100 — a few points in the right tail carry most of the weight). The i.i.d. test
set from the training distribution — the first row — would never have revealed the problem;
only data from the deployment distribution can. With a well-specified model (try degree 3)
the unweighted fit is already fine under the shift: covariate shift hurts *because* the class
is wrong somewhere, and importance weighting then trades bias for variance.
""")

# ---------------------------------------------------------------- Section 10
nb.md(r"""
## 10. A leakage trap: feature selection outside the folds

100 samples, 5,000 features of pure noise, random labels — nothing is learnable. Select the 50
features most correlated with the label *using all the data*, then cross-validate a classifier
on them: the score is spectacular and completely wrong, because every test fold's labels were
used to choose the features. Putting the selection inside a `Pipeline` repeats it on each
training fold only.
""")

nb.code(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

n, p = 100, 5000
X_noise = rng.normal(size=(n, p)); y_rand = rng.integers(0, 2, n)
cv = KFold(10, shuffle=True, random_state=0)

corr = np.abs((X_noise - X_noise.mean(0)).T @ (y_rand - y_rand.mean()))
top50 = np.argsort(-corr)[:50]                               # selection sees ALL labels -> leakage
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_noise[:, top50], y_rand, cv=cv)

proper = cross_val_score(make_pipeline(SelectKBest(f_classif, k=50), LogisticRegression(max_iter=1000)),
                         X_noise, y_rand, cv=cv)              # selection re-fitted inside every training fold
print(f"leaky protocol:  10-fold CV accuracy {leaky.mean():.3f} +- {leaky.std():.3f}   (data are pure noise!)")
print(f"proper pipeline: 10-fold CV accuracy {proper.mean():.3f} +- {proper.std():.3f}   (chance = 0.5)")
""")

nb.md(r"""
The leaky protocol reports over 90 % accuracy on data with no signal; the honest one reports
chance. Every bound in this module assumes the test data were untouched by the training
process — including by preprocessing, feature selection and hyper-parameter choices. The
`Pipeline` is how you make that assumption true in code.
""")

# ---------------------------------------------------------------- Summary
nb.md(r"""
## Summary

- Training error is biased by selection; a test set restores the fixed-hypothesis setting in
  which $\hat R_S(h)$ is an unbiased, concentrating estimate of $R(h)$.
- Excess risk = approximation + estimation (+ optimisation); the U-curve is their sum.
- Hoeffding: $P(|\hat p - p| \ge t) \le 2e^{-2nt^2}$ — valid, loose, and exponential, which is
  what makes the union bound cheap. About 4,600 examples pin an accuracy to ±0.02 at 95 %.
- The best of $m$ models on a validation set is optimistic by roughly
  $\sqrt{2\ln m}\cdot\text{SE}$; the union bound gives the distribution-free envelope.
- Lines in the plane shatter 3 points but not 4; Sauer's lemma makes the growth function
  polynomial; VC bounds fall like $\sqrt{d\ln n/n}$ with pessimistic constants.
- Realisable learning curves fall like $1/n$; agnostic ones like $1/\sqrt n$.
- Among zero-training-error separators, larger margin means lower test error, independently
  of dimension; early stopping reproduces ridge's filter with $\lambda \approx 1/(\eta t)$.
- Random-feature regression shows double descent with a spike at $p = n$ and a second
  descent driven by shrinking $\|w\|$; GD from zero converges to the minimum-norm interpolator.
- Covariate shift breaks a misspecified model even though $P(y \mid x)$ is unchanged;
  importance weighting repairs it at the price of effective sample size.
- Feature selection outside the folds turns pure noise into 90 %+ accuracy — use a `Pipeline`.

## Next: open `exercises.ipynb`

Eleven exercises with instant self-checks, from Hoeffding sample sizes to a mini-project that
fits a power law to the generalisation gap.
""")

nb.save(MOD / "lesson.ipynb")
print("lesson cells:", len(nb.cells))

# =============================================================================
# EXERCISES — solution code is executed here to compute the hidden answers
# =============================================================================
SETUP = r"""
import numpy as np
import matplotlib.pyplot as plt
from math import comb, log, sqrt, exp
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
"""

ex = ExerciseSet(
    "Module 15 — Exercises",
    intro_md=r"""
Exercises for **Module 15 — Generalisation and Statistical Learning Theory**. They follow the
order of the [concept guide](README.md): Hoeffding sample sizes, Chebyshev versus Hoeffding,
the finite-class bound, growth functions and VC dimension by brute force, the optimism of
selecting the best of many models, gradient descent versus the pseudo-inverse, a double-descent
experiment, two conceptual questions, and a mini-project that fits a power law to the
generalisation gap. Use only NumPy and the standard library (plus `scipy` where a prompt says
so) and keep everything seeded.
""",
    setup_code=SETUP,
)

ns: dict = {}
exec(SETUP, ns)  # gives the builder the same np/rng names as the notebook


def run_solution(src: str) -> None:
    """Execute solution code inside the shared namespace (hashes come from *running* it)."""
    exec(src, ns)


# ---------------------------------------------------------------- Ex 1 (warm-up)
SOL1 = r"""
def hoeffding_n(eps, delta):
    return int(np.ceil(np.log(2.0 / delta) / (2.0 * eps ** 2)))

def hoeffding_eps(n, delta):
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * n)))

n_benchmark = hoeffding_n(0.01, 0.01)
eps_benchmark = hoeffding_eps(5000, 0.05)
"""
run_solution(SOL1)
h1a = sha_of(ns["n_benchmark"])
safe_to_hash(ns["eps_benchmark"])
h1b = sha_of(ns["eps_benchmark"])

ex.add(
    title="Hoeffding sample size and deviation",
    difficulty=1,
    prompt_md=r"""
For a *fixed* classifier evaluated on $n$ i.i.d. test examples, Hoeffding's inequality gives,
with probability at least $1 - \delta$, $|\hat a - a| \le \sqrt{\ln(2/\delta)/(2n)}$.

Implement

* `hoeffding_n(eps, delta)` — the smallest **integer** $n$ (use `np.ceil`, return a Python `int`)
  guaranteeing $|\hat a - a| \le \varepsilon$ with probability $\ge 1-\delta$;
* `hoeffding_eps(n, delta)` — the deviation $\varepsilon$ guaranteed by $n$ examples (a `float`).

Then set `n_benchmark = hoeffding_n(0.01, 0.01)` (a benchmark that pins accuracy to one point
at 99 %) and `eps_benchmark = hoeffding_eps(5000, 0.05)`.
""",
    starter=r"""
def hoeffding_n(eps, delta):
    # YOUR CODE HERE
    return None

def hoeffding_eps(n, delta):
    # YOUR CODE HERE
    return None

n_benchmark = None     # YOUR CODE HERE
eps_benchmark = None   # YOUR CODE HERE
""",
    solution=SOL1,
    check=r"""
try:
    got_n = hoeffding_n(0.02, 0.05)
except Exception:
    got_n = None
check('n for eps = 0.02, delta = 0.05 is 4612', got_n == 4612, hint="ln(40) / (2 * 0.0004), rounded up")
try:
    got_eps = hoeffding_eps(4612, 0.05)
except Exception:
    got_eps = None
check('eps for n = 4612, delta = 0.05 is just under 0.02', got_eps is not None and 0.0199 < got_eps <= 0.02)
check_hash('hidden: n_benchmark', n_benchmark if isinstance(n_benchmark, (int, np.integer)) else None, 'HASH1A')
check_hash('hidden: eps_benchmark (3 decimals)', eps_benchmark if isinstance(eps_benchmark, float) else None, 'HASH1B')
""".replace("HASH1A", h1a).replace("HASH1B", h1b),
    explanation_md=r"""
Set $2e^{-2n\varepsilon^2} = \delta$ and solve: $n = \ln(2/\delta)/(2\varepsilon^2)$, rounded *up* because
$n$ must be an integer and the inequality must still hold. The deviation form is the same
equation solved for $\varepsilon$. Note the asymmetry: halving $\varepsilon$ quadruples $n$, while
cutting $\delta$ from 0.05 to 0.01 only raises $\ln(2/\delta)$ from 3.69 to 5.30 — confidence
is cheap, precision is expensive. The hidden benchmark size is in the tens of thousands, which
is why 1,000-example benchmarks cannot resolve one-point differences.
""",
)

# ---------------------------------------------------------------- Ex 2 (warm-up)
SOL2 = r"""
def chebyshev_tail(n, t, p):
    return float(min(1.0, p * (1 - p) / (n * t ** 2)))

def hoeffding_tail(n, t):
    return float(min(1.0, 2.0 * np.exp(-2.0 * n * t ** 2)))

def crossover_n(t, p):
    n = 1
    while hoeffding_tail(n, t) >= chebyshev_tail(n, t, p):
        n += 1
    return n

ratio_1000 = chebyshev_tail(1000, 0.05, 0.8) / hoeffding_tail(1000, 0.05)
n_cross = crossover_n(0.05, 0.8)
"""
run_solution(SOL2)
safe_to_hash(ns["ratio_1000"])
h2a = sha_of(ns["ratio_1000"])
h2b = sha_of(ns["n_cross"])

ex.add(
    title="Chebyshev versus Hoeffding",
    difficulty=1,
    prompt_md=r"""
For a sample proportion $\hat p$ of $n$ Bernoulli$(p)$ trials, two bounds on $P(|\hat p - p| \ge t)$:

$$\text{Chebyshev: } \frac{p(1-p)}{n t^2}, \qquad \text{Hoeffding: } 2e^{-2nt^2}.$$

Implement `chebyshev_tail(n, t, p)` and `hoeffding_tail(n, t)` (both capped at 1 with `min`,
returning `float`), and `crossover_n(t, p)` — the smallest integer $n \ge 1$ at which the
Hoeffding bound becomes **strictly smaller** than the Chebyshev bound (loop upward from 1).

Then set `ratio_1000 = chebyshev_tail(1000, 0.05, 0.8) / hoeffding_tail(1000, 0.05)` and
`n_cross = crossover_n(0.05, 0.8)`.
""",
    starter=r"""
def chebyshev_tail(n, t, p):
    # YOUR CODE HERE
    return None

def hoeffding_tail(n, t):
    # YOUR CODE HERE
    return None

def crossover_n(t, p):
    # YOUR CODE HERE
    return None

ratio_1000 = None   # YOUR CODE HERE
n_cross = None      # YOUR CODE HERE
""",
    solution=SOL2,
    check=r"""
try:
    c = chebyshev_tail(100, 0.1, 0.8); h = hoeffding_tail(100, 0.1)
except Exception:
    c = h = None
check_close('Chebyshev n=100, t=0.1, p=0.8 is 0.16', c, 0.16)
check_close('Hoeffding n=100, t=0.1 is 2 exp(-2)', h, 2 * np.exp(-2))
try:
    capped = chebyshev_tail(10, 0.01, 0.5)
except Exception:
    capped = None
check('bounds are capped at 1', capped == 1.0)
check_hash('hidden: ratio Chebyshev/Hoeffding at n=1000, t=0.05 (3 decimals)', ratio_1000 if isinstance(ratio_1000, float) else None, 'HASH2A')
check_hash('hidden: crossover n for t=0.05, p=0.8', n_cross if isinstance(n_cross, (int, np.integer)) else None, 'HASH2B')
""".replace("HASH2A", h2a).replace("HASH2B", h2b),
    explanation_md=r"""
Chebyshev uses the variance $p(1-p)/n$ of the proportion and decays like $1/(nt^2)$; Hoeffding
uses only the range $[0, 1]$ and decays like $e^{-2nt^2}$. For small $nt^2$ the constant 2 and
the ignored variance make Hoeffding the *looser* bound (at $n = 100$, $t = 0.1$: 0.27 against
0.16); the exponential wins once $nt^2$ is a few units — the crossover for $t = 0.05$,
$p = 0.8$ is in the low hundreds — and by $n = 1{,}000$ Chebyshev is several times larger.
A common slip is to forget the factor 2 in Hoeffding's two-sided form, or to use $1/4$
instead of $p(1-p)$ in Chebyshev (which is also valid but looser).
""",
)

# ---------------------------------------------------------------- Ex 3 (warm-up)
SOL3 = r"""
def finite_class_bound(n, size_H, delta):
    return float(np.sqrt(np.log(2.0 * size_H / delta) / (2.0 * n)))

def finite_class_n(eps, size_H, delta):
    return int(np.ceil(np.log(2.0 * size_H / delta) / (2.0 * eps ** 2)))

n_stumps = 20 * 100 * 2
bound_stumps = finite_class_bound(2000, n_stumps, 0.05)
n_needed_million = finite_class_n(0.03, 10 ** 6, 0.01)
"""
run_solution(SOL3)
safe_to_hash(ns["bound_stumps"])
h3a = sha_of(ns["bound_stumps"])
h3b = sha_of(ns["n_needed_million"])

ex.add(
    title="The finite-class uniform bound",
    difficulty=1,
    prompt_md=r"""
Union bound + Hoeffding: with probability $\ge 1 - \delta$, **every** $h$ in a finite class
$\mathcal H$ satisfies $|R(h) - \hat R_S(h)| \le \sqrt{\ln(2|\mathcal H|/\delta)/(2n)}$.

Implement `finite_class_bound(n, size_H, delta)` (a `float`) and `finite_class_n(eps, size_H, delta)`
— the smallest integer $n$ that makes the bound $\le \varepsilon$ (a Python `int`).

Then: decision stumps on 20 features with 100 thresholds and 2 orientations each form a class
of `n_stumps = 20 * 100 * 2` hypotheses. Set `bound_stumps = finite_class_bound(2000, n_stumps, 0.05)`
and `n_needed_million = finite_class_n(0.03, 10**6, 0.01)`.
""",
    starter=r"""
def finite_class_bound(n, size_H, delta):
    # YOUR CODE HERE
    return None

def finite_class_n(eps, size_H, delta):
    # YOUR CODE HERE
    return None

n_stumps = 20 * 100 * 2
bound_stumps = None        # YOUR CODE HERE
n_needed_million = None    # YOUR CODE HERE
""",
    solution=SOL3,
    check=r"""
try:
    b1 = finite_class_bound(1000, 1, 0.05)
except Exception:
    b1 = None
check_close('|H| = 1 recovers Hoeffding: sqrt(ln(40)/2000)', b1, np.sqrt(np.log(40) / 2000))
try:
    n1 = finite_class_n(0.02, 1, 0.05)
except Exception:
    n1 = None
check('|H| = 1, eps = 0.02 needs 4612 examples', n1 == 4612)
try:
    bigger = finite_class_bound(1000, 10**6, 0.05) > finite_class_bound(1000, 10**3, 0.05)
except Exception:
    bigger = False
check('a larger class gives a larger bound', bigger)
check_hash('hidden: bound for 4,000 stumps, n = 2000 (3 decimals)', bound_stumps if isinstance(bound_stumps, float) else None, 'HASH3A')
check_hash('hidden: n needed for a million hypotheses, eps = 0.03, delta = 0.01', n_needed_million if isinstance(n_needed_million, (int, np.integer)) else None, 'HASH3B')
""".replace("HASH3A", h3a).replace("HASH3B", h3b),
    explanation_md=r"""
Each hypothesis fails Hoeffding with probability at most $2e^{-2n\varepsilon^2}$; the union bound
multiplies by $|\mathcal H|$; setting $2|\mathcal H|e^{-2n\varepsilon^2} = \delta$ and solving gives both
formulas. The class enters only through $\ln|\mathcal H|$: 4,000 stumps cost a factor
$\sqrt{\ln(160{,}000)/\ln 40} \approx 1.8$ over a single hypothesis, and a *million* hypotheses
at $\varepsilon = 0.03$, $\delta = 0.01$ need around ten thousand examples — about 60 % more than
a thousand hypotheses would. People sometimes multiply $n$ by $|\mathcal H|$ instead of adding
$\ln|\mathcal H|$; that is the mistake the logarithm saves you from.
""",
)

# ---------------------------------------------------------------- Ex 4 (core)
SOL4 = r"""
from itertools import product

def growth_function(hypotheses, points):
    seen = set()
    for h in hypotheses:
        seen.add(tuple(int(h(x)) for x in points))
    return len(seen)

def threshold_hypotheses(points):
    xs = np.sort(np.asarray(points, dtype=float))
    cuts = np.concatenate([[xs[0] - 1.0], (xs[:-1] + xs[1:]) / 2.0, [xs[-1] + 1.0]])
    return [(lambda x, a=a: x >= a) for a in cuts]

def threshold_growth(n):
    pts = np.arange(n, dtype=float)
    return growth_function(threshold_hypotheses(pts), pts)

growth_thresholds_12 = threshold_growth(12)
"""
run_solution(SOL4)
h4 = sha_of(ns["growth_thresholds_12"])

ex.add(
    title="Growth function by brute force: thresholds",
    difficulty=2,
    prompt_md=r"""
The growth function $\Pi_{\mathcal H}(n)$ counts the distinct labellings (dichotomies) a class
realises on $n$ points. Implement

* `growth_function(hypotheses, points)` — `hypotheses` is a list of callables `h(x) -> bool`;
  return the number of **distinct** tuples `(h(x_1), ..., h(x_n))` over the list;
* `threshold_hypotheses(points)` — a list of threshold rules $h_a(x) = \mathbf 1[x \ge a]$ that
  together realise every dichotomy thresholds can produce on `points` (hint: cut points below
  the smallest point, between consecutive sorted points, and above the largest — one rule each;
  bind `a` as a default argument, `lambda x, a=a: x >= a`);
* `threshold_growth(n)` — the growth function of thresholds on `n` distinct points.

Set `growth_thresholds_12 = threshold_growth(12)`. Every function should work for any
distinct real-valued points.
""",
    starter=r"""
from itertools import product

def growth_function(hypotheses, points):
    # YOUR CODE HERE  (collect label tuples in a set, return its size)
    return None

def threshold_hypotheses(points):
    # YOUR CODE HERE  (one lambda per cut point; use `lambda x, a=a: x >= a`)
    return None

def threshold_growth(n):
    # YOUR CODE HERE
    return None

growth_thresholds_12 = None   # YOUR CODE HERE
""",
    solution=SOL4,
    check=r"""
try:
    g3 = threshold_growth(3); g5 = threshold_growth(5)
except Exception:
    g3 = g5 = None
check('thresholds realise n + 1 dichotomies on 3 points', g3 == 4)
check('thresholds realise n + 1 dichotomies on 5 points', g5 == 6)
try:
    pts = np.array([0.3, -1.2, 4.0, 2.2])
    const = growth_function([lambda x: True, lambda x: False, lambda x: x > -100], pts)
except Exception:
    const = None
check('growth_function counts DISTINCT dichotomies (two constant rules + a duplicate -> 2)', const == 2)
check_hash('hidden: growth function of thresholds on 12 points', growth_thresholds_12 if isinstance(growth_thresholds_12, (int, np.integer)) else None, 'HASH4')
""".replace("HASH4", h4),
    explanation_md=r"""
On $n$ sorted points a threshold produces a labelling of the form $0\cdots01\cdots1$: the
switch can happen before point 1, between any two consecutive points, or after point $n$ —
$n+1$ possibilities, so $\Pi(n) = n + 1$, consistent with $d_{\mathrm{VC}} = 1$ and Sauer's
lemma $\binom n0 + \binom n1$. The brute-force counter is deliberately general: in the next
exercise it is reused for intervals. The classic Python pitfall is building lambdas in a loop
without binding the loop variable (`lambda x: x >= a` captures the *last* `a` for every rule);
the default-argument trick `a=a` freezes it.
""",
)

# ---------------------------------------------------------------- Ex 5 (core)
SOL5 = r"""
def interval_realisable(labels):
    labels = [int(v) for v in labels]
    ones = [i for i, v in enumerate(labels) if v == 1]
    if not ones:
        return True
    return all(labels[i] == 1 for i in range(ones[0], ones[-1] + 1))

def interval_growth(n):
    return sum(1 for lab in product([0, 1], repeat=n) if interval_realisable(lab))

def interval_vc_dimension(max_n=8):
    d = 0
    for n in range(1, max_n + 1):
        if interval_growth(n) == 2 ** n:
            d = n
    return d

vc_intervals = interval_vc_dimension()
growth_intervals_10 = interval_growth(10)
"""
run_solution(SOL5)
h5a = sha_of(ns["vc_intervals"])
h5b = sha_of(ns["growth_intervals_10"])

ex.add(
    title="Shattering and the VC dimension of intervals",
    difficulty=2,
    prompt_md=r"""
The class of intervals $h_{a,b}(x) = \mathbf 1[a \le x \le b]$ on sorted points realises a
labelling exactly when its 1s form a single contiguous block (or there are none).

Implement

* `interval_realisable(labels)` — `labels` is a sequence of 0/1 for points **already sorted by
  position**; return `True` iff some interval realises it;
* `interval_growth(n)` — the number of realisable labellings of $n$ points (iterate over
  `product([0, 1], repeat=n)` from `itertools`);
* `interval_vc_dimension(max_n=8)` — the largest $n \le$ `max_n` with `interval_growth(n) == 2**n`.

Set `vc_intervals = interval_vc_dimension()` and `growth_intervals_10 = interval_growth(10)`.
The closed form you should recover is $\Pi(n) = \binom{n}{2} + n + 1$.
""",
    starter=r"""
from itertools import product

def interval_realisable(labels):
    # YOUR CODE HERE  (the 1s must form one contiguous block, or be absent)
    return None

def interval_growth(n):
    # YOUR CODE HERE
    return None

def interval_vc_dimension(max_n=8):
    # YOUR CODE HERE
    return None

vc_intervals = None          # YOUR CODE HERE
growth_intervals_10 = None   # YOUR CODE HERE
""",
    solution=SOL5,
    check=r"""
try:
    r1 = interval_realisable([0, 1, 1, 0]); r2 = interval_realisable([1, 0, 1]); r3 = interval_realisable([0, 0, 0])
except Exception:
    r1 = r2 = r3 = None
check('0110 is realisable by an interval', r1 is True)
check('101 is NOT realisable (needs two intervals)', r2 is False)
check('all-zero labelling is realisable (empty interval)', r3 is True)
try:
    g3 = interval_growth(3); g6 = interval_growth(6)
except Exception:
    g3 = g6 = None
check('growth function on 3 points is 7', g3 == 7)
check('growth function matches C(n,2) + n + 1 at n = 6', g6 == comb(6, 2) + 6 + 1)
check_hash('hidden: VC dimension of intervals', vc_intervals if isinstance(vc_intervals, (int, np.integer)) else None, 'HASH5A')
check_hash('hidden: growth function on 10 points', growth_intervals_10 if isinstance(growth_intervals_10, (int, np.integer)) else None, 'HASH5B')
""".replace("HASH5A", h5a).replace("HASH5B", h5b),
    explanation_md=r"""
An interval picks out a contiguous run of sorted points, so a labelling is realisable iff its
1s are contiguous: choose the first and last 1 among $n$ points ($\binom n2$ ways with two
distinct endpoints, $n$ ways with one point) or no 1 at all — $\binom n2 + n + 1$. Two points
are shattered ($4 = \binom22 + 2 + 1$), three are not ($7 < 8$, the pattern $101$ fails), so
$d_{\mathrm{VC}} = 2$. Sauer's lemma with $d = 2$ gives exactly $\binom n0 + \binom n1 + \binom n2$,
so for intervals the lemma is tight. The brute-force search over $2^n$ labellings is fine up
to $n \approx 16$; a frequent bug is treating the all-zero labelling as unrealisable.
""",
)

# ---------------------------------------------------------------- Ex 6 (core)
SOL6 = r"""
def best_of_m_optimism(m, n, acc, n_rep, g):
    best = g.binomial(n, acc, size=(n_rep, m)).max(axis=1) / n
    return float(np.mean(best) - acc)

opt = {m: best_of_m_optimism(m, 1000, 0.8, 4000, np.random.default_rng(1)) for m in (1, 10, 100)}
"""
run_solution(SOL6)

ex.add(
    title="Simulating the winner's curse",
    difficulty=2,
    prompt_md=r"""
Implement `best_of_m_optimism(m, n, acc, n_rep, g)`: simulate `n_rep` rounds in which $m$ models
with the **same** true accuracy `acc` are each scored on an independent validation set of `n`
examples (`g.binomial(n, acc, size=(n_rep, m)) / n` draws all scores at once; `g` is a
`numpy.random.Generator`), keep the best score in each round, and return the average of
`best - acc` as a `float`.

Then build the dictionary `opt = {m: best_of_m_optimism(m, 1000, 0.8, 4000, np.random.default_rng(1)) for m in (1, 10, 100)}`.
The theory predicts about $0$, $1.54\cdot\text{SE}$ and $2.51\cdot\text{SE}$ with
$\text{SE} = \sqrt{0.8 \cdot 0.2/1000} \approx 0.0126$.
""",
    starter=r"""
def best_of_m_optimism(m, n, acc, n_rep, g):
    # YOUR CODE HERE
    return None

opt = None   # YOUR CODE HERE  ({m: ... for m in (1, 10, 100)})
""",
    solution=SOL6,
    check=r"""
try:
    o1 = opt[1]; o10 = opt[10]; o100 = opt[100]
except Exception:
    o1 = o10 = o100 = None
se = np.sqrt(0.8 * 0.2 / 1000)
check('m = 1: no optimism (|mean - acc| < 0.002)', o1 is not None and abs(o1) < 0.002)
check('m = 10: optimism within 20% of 1.54 SE', o10 is not None and 0.8 * 1.539 * se < o10 < 1.2 * 1.539 * se)
check('m = 100: optimism within 20% of 2.51 SE', o100 is not None and 0.8 * 2.508 * se < o100 < 1.2 * 2.508 * se)
check('optimism grows with m', None not in (o1, o10, o100) and o1 < o10 < o100)
try:
    big_n = best_of_m_optimism(10, 100_000, 0.8, 500, np.random.default_rng(2))
except Exception:
    big_n = None
check('a 100x larger validation set shrinks the optimism ~10x', big_n is not None and big_n < 0.004)
""",
    explanation_md=r"""
Each validation score is a Binomial proportion with standard error $\sqrt{a(1-a)/n}$; the
maximum of $m$ such scores exceeds $a$ by roughly $\mathbb E[\max \text{ of } m \text{ normals}]$
standard errors — $1.54$ for $m = 10$, $2.51$ for $m = 100$ — because selection keeps the lucky
draws. The optimism shrinks like $1/\sqrt n$ but only grows like $\sqrt{\ln m}$, which is why
"compare fewer models" is a weaker lever than "use more validation data". Common wrong turns:
averaging the best *accuracy* instead of the *excess* over `acc`, or reusing one score matrix
across the three values of $m$ (which is fine) while forgetting that `m = 1` must come out near
zero (if it does not, the subtraction is wrong).
""",
)

# ---------------------------------------------------------------- Ex 7 (core)
SOL7 = r"""
def gd_least_squares(X, y, steps=5000, lr=None):
    n, p = X.shape
    if lr is None:
        lr = 1.0 / np.linalg.norm(X, 2) ** 2
    w = np.zeros(p)
    for _ in range(steps):
        w = w - lr * X.T @ (X @ w - y)
    return w
"""
run_solution(SOL7)

ex.add(
    title="Gradient descent from zero equals the pseudo-inverse",
    difficulty=2,
    prompt_md=r"""
Implement `gd_least_squares(X, y, steps=5000, lr=None)`: run plain gradient descent on
$\tfrac12\|Xw - y\|^2$ **starting from `w = 0`**, with step size `lr` (default
`1 / np.linalg.norm(X, 2)**2`, i.e. $1/\sigma_{\max}^2$), for `steps` iterations, and return `w`.
Use only NumPy array operations — no `lstsq`, `pinv` or `solve`.

The checks verify that for $p > n$ your iterate (i) interpolates the data, (ii) coincides with
`np.linalg.pinv(X) @ y`, and (iii) is shorter than any other interpolator; and that for $p < n$
it recovers ordinary least squares.
""",
    starter=r"""
def gd_least_squares(X, y, steps=5000, lr=None):
    # YOUR CODE HERE  (w = 0; repeat w <- w - lr * X^T (X w - y))
    return None
""",
    solution=SOL7,
    check=r"""
_g = np.random.default_rng(7)
_X = _g.normal(size=(20, 50)); _y = _X @ _g.normal(size=50)          # p > n, noiseless
try:
    _w = gd_least_squares(_X, _y)
except Exception:
    _w = None
check('p > n: GD interpolates the data', _w is not None and np.allclose(_X @ _w, _y, atol=1e-6))
check_close('p > n: GD from zero equals pinv(X) y', _w, np.linalg.pinv(_X) @ _y, tol=1e-6)
try:
    _other = _w + (np.eye(50) - np.linalg.pinv(_X) @ _X) @ _g.normal(size=50)   # add a null-space vector
    shorter = np.linalg.norm(_w) < np.linalg.norm(_other) and np.allclose(_X @ _other, _y, atol=1e-6)
except Exception:
    shorter = False
check('p > n: the GD solution is the shortest interpolator', shorter)
_X2 = _g.normal(size=(60, 5)); _y2 = _X2 @ np.arange(1., 6.) + 0.1 * _g.normal(size=60)
try:
    _w2 = gd_least_squares(_X2, _y2)
except Exception:
    _w2 = None
check_close('p < n: GD recovers ordinary least squares', _w2, np.linalg.lstsq(_X2, _y2, rcond=None)[0], tol=1e-5)
""",
    explanation_md=r"""
The update $w \leftarrow w - \eta X^\top(Xw - y)$ adds a vector in the row space of $X$ at every
step, so from $w_0 = 0$ every iterate lies in $\operatorname{row}(X)$. For $\eta < 2/\sigma_{\max}^2$
the iteration converges to a minimiser of the convex loss — when $p > n$, an exact interpolator.
The interpolators form $X^{+}y + \operatorname{null}(X)$, and $\operatorname{null}(X) \perp \operatorname{row}(X)$, so the
only interpolator in the row space is $X^{+}y$, which is also the shortest (Pythagoras). This is
the implicit regularisation of gradient descent: nothing in the loss asks for a small norm, yet
the algorithm delivers it. Starting from a random $w_0$ would break the argument — the
null-space component of $w_0$ is never touched — which is exactly what the check with a
null-space perturbation illustrates.
""",
)

# ---------------------------------------------------------------- Ex 8 (stretch)
SOL8 = r"""
def relu_features(X, W, b):
    return np.maximum(0.0, X @ W + b)

def double_descent_errors(n=40, ps=(20, 40, 200), reps=10, sigma=0.3, seed=0):
    g = np.random.default_rng(seed)
    target = lambda X: np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]
    errs = {p: [] for p in ps}
    for _ in range(reps):
        X_tr = g.normal(size=(n, 5)); y_tr = target(X_tr) + sigma * g.normal(size=n)
        X_te = g.normal(size=(400, 5)); y_te = target(X_te)
        W = g.normal(size=(5, max(ps))); b = g.uniform(-1, 1, max(ps))
        for p in ps:
            Phi, Phi_te = relu_features(X_tr, W[:, :p], b[:p]), relu_features(X_te, W[:, :p], b[:p])
            w = np.linalg.lstsq(Phi, y_tr, rcond=None)[0]
            errs[p].append(np.mean((Phi_te @ w - y_te) ** 2))
    return {p: float(np.median(v)) for p, v in errs.items()}

dd = double_descent_errors()
"""
run_solution(SOL8)

ex.add(
    title="A double-descent experiment",
    difficulty=3,
    prompt_md=r"""
Reproduce double descent with random ReLU features. Implement

* `relu_features(X, W, b)` returning $\max(0, XW + b)$ (shape `(n, p)`);
* `double_descent_errors(n=40, ps=(20, 40, 200), reps=10, sigma=0.3, seed=0)`: with
  `g = np.random.default_rng(seed)`, repeat `reps` times — draw `X_tr` of shape `(n, 5)` from
  `g.normal`, labels `y_tr = target(X_tr) + sigma * g.normal(size=n)` with
  `target(X) = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]`, a noiseless test set `X_te` of 400 rows
  (`y_te = target(X_te)`), then `W = g.normal(size=(5, max(ps)))` and `b = g.uniform(-1, 1, max(ps))`
  (draw in exactly this order); for each `p` in `ps` use the **first** `p` columns of `W` / entries
  of `b`, fit the output weights by `np.linalg.lstsq(Phi, y_tr, rcond=None)[0]` (minimum-norm),
  and record the test MSE. Return `{p: median test MSE over reps}`.

Set `dd = double_descent_errors()`. The checks test the *shape*: the error at $p = n$ (the
interpolation threshold) must be far above both $p = n/2$ and $p = 5n$, and $p = 5n$ must beat
$p = n/2$ — the second descent.
""",
    starter=r"""
def relu_features(X, W, b):
    # YOUR CODE HERE
    return None

def double_descent_errors(n=40, ps=(20, 40, 200), reps=10, sigma=0.3, seed=0):
    # YOUR CODE HERE
    return None

dd = None   # YOUR CODE HERE
""",
    solution=SOL8,
    check=r"""
try:
    e_half, e_n, e_5n = dd[20], dd[40], dd[200]
except Exception:
    e_half = e_n = e_5n = None
check('under-parameterised (p = n/2): test MSE below 0.8', e_half is not None and 0.1 < e_half < 0.8)
check('interpolation threshold (p = n): test MSE at least 5x the p = n/2 value', None not in (e_half, e_n) and e_n > 5 * e_half)
check('over-parameterised (p = 5n): test MSE below 0.6', e_5n is not None and e_5n < 0.6)
check('second descent: p = 5n beats p = n/2', None not in (e_half, e_5n) and e_5n < e_half)
try:
    Phi = relu_features(np.array([[1., -2.]]), np.array([[1., 0.], [0., 1.]]), np.array([0., 0.]))
except Exception:
    Phi = None
check_close('relu_features clips negatives: [[1, -2]] -> [[1, 0]]', Phi, np.array([[1., 0.]]))
""",
    explanation_md=r"""
With $p < n$ the least-squares fit has the usual variance $\propto p/(n-p)$, which blows up as
$p \to n$: at the interpolation threshold the feature matrix is square and nearly singular, and
the only way to interpolate noisy labels is with huge coefficients — the test error spikes.
Beyond $p = n$ there are many interpolators and `lstsq` returns the one of minimum norm; extra
features let it fit the same data with smaller, smoother weights, so the error falls again and,
because random ReLU features of a smooth target are a benign feature family, ends below the
classical minimum. The hidden medians for the default seed are roughly $0.5$, $30$ and $0.3$.
Using the mean instead of the median still passes but is dominated by a few catastrophic runs
at $p = n$; forgetting to reuse the *same* `W` across `p` within a run is harmless for the shape
but breaks the "nested classes" interpretation.
""",
)

# ---------------------------------------------------------------- Ex 9 (conceptual)
ex.add_conceptual(
    title="Why VC theory does not explain deep learning",
    difficulty=2,
    prompt_md=r"""
A ReLU network with $W = 2.5\times10^7$ weights and $L = 50$ layers has VC dimension of order
$WL\log W \approx 3\times10^{10}$. It is trained on $1.28\times10^6$ images, reaches zero training
error, and scores 76 % top-1 accuracy on the test set. A colleague says: "VC theory predicts
this network cannot generalise, so the theory is wrong."

Explain (a) what the VC bound actually says here and why that is *not* a prediction of failure;
(b) which experimental fact (Zhang et al., 2017) shows that *any* uniform-convergence argument
over the whole architecture class must be vacuous; (c) two ingredients the modern account adds
to recover a non-vacuous story.
""",
    answer_md=r"""
(a) The VC bound reads $R(h) \le \hat R_S(h) + \text{(something} \gtrsim \sqrt{d_{\mathrm{VC}}/n})$, and with
$d_{\mathrm{VC}}/n \approx 2\times10^4$ the right-hand side exceeds 1. A bound above 1 is *vacuous*, not
false: it says the theory offers no guarantee, which is different from predicting failure. The
bound is a worst case over all distributions and all hypotheses in the class; it cannot be
tightened for this architecture because (b) the same network reaches zero training error on
CIFAR-10 with randomly permuted labels. If the class can fit arbitrary labels on $n$ points,
there *are* hypotheses in it whose generalisation gap is essentially 1, so any statement of the
form "for all $h$ in the class, $R(h) - \hat R_S(h)$ is small" is simply untrue; uniform convergence
over the architecture class is dead on arrival. (c) The modern account restricts attention to
the hypotheses the *algorithm actually produces*: implicit regularisation (gradient descent from
small initialisation prefers low-norm, low-complexity, large-margin solutions — the min-norm
result for least squares, max-margin for logistic loss), so the effective class is far smaller
than the architecture class and norm-based or margin-based bounds replace parameter counts; and
benign overfitting / double descent (in the over-parameterised regime the interpolator can
absorb label noise in many weak directions while fitting the signal in a few strong ones). A
third ingredient is algorithmic stability or PAC-Bayes bounds, which are about the learned
distribution over weights rather than the worst member of the class. The honest summary is
that VC theory is correct but answers a question ("what holds for every function the
architecture can represent?") that is not the one deep learning poses.
""",
)

# ---------------------------------------------------------------- Ex 10 (conceptual)
ex.add_conceptual(
    title="What no free lunch does and does not say",
    difficulty=2,
    prompt_md=r"""
Two claims, both citing the no-free-lunch theorem:

1. "Averaged over all possible target functions, no algorithm beats random guessing on unseen
   inputs — so benchmarking learning algorithms against each other is meaningless."
2. "Since convolutional networks beat fully connected networks on images, the no-free-lunch
   theorem must be false."

Diagnose each claim. Then explain, with one concrete example, what an inductive bias is and
why its *mismatch* with a problem is the mechanism by which NFL bites in practice. Finally,
state which assumption of this entire module is itself an inductive bias and name one way it
fails in deployment.
""",
    answer_md=r"""
Claim 1 is a misreading of the averaging. NFL averages uniformly over *all* $2^{|\mathcal X|}$
labelings of the domain, almost all of which are incompressible noise with no relation between
nearby inputs. Real problems are not drawn from that uniform measure — images, text and
molecules have structure — and on any *particular* distribution algorithms differ enormously.
Benchmarks measure performance on the structured problems we care about, which is exactly
what NFL leaves unconstrained. Claim 2 is the mirror error: NFL is a theorem about the uniform
average, and a CNN's advantage on images is consistent with it because a CNN must be
correspondingly *worse* than a fully connected network on the (never-seen, unnatural) targets
whose labels change arbitrarily under translation. The theorem says every advantage on some
targets is paid for on others; it does not say the advantages do not exist. An inductive bias
is the set of assumptions that let a learner prefer some hypotheses over others consistent
with the data: a convolution assumes the same local pattern means the same thing anywhere in
the image (translation equivariance, locality), which is why it needs 36,864 weights for a
$3\times3$, 64-to-64-channel layer instead of billions — and why it will fail on a task where
the position of a pattern is what matters, unless positional information is added back. NFL
bites exactly when the bias is wrong for the problem: a smoothness prior on a discontinuous
target, a bag-of-words model on a task where word order carries the meaning, a linear model
on XOR. The module-wide assumption that is itself a bias is that training and deployment data
are i.i.d. draws from the same distribution $P$. It fails under covariate shift (a different
case mix at a new hospital), label shift (a different prevalence), or concept shift (the
meaning of the features changes, as in fraud patterns after a policy change) — and every bound
in the module is silent the moment it does.
""",
)

# ---------------------------------------------------------------- Ex 11 (project)
PROJECT_DATA = r"""
# ---- setting: cubic target, polynomial regression of the correct degree, noise sigma = 0.5 ----
g = np.random.default_rng(0)
coef_true = np.array([2.0, 0.0, -1.0, 0.5])        # highest power first (np.polyval convention): 2x^3 - x + 0.5
f_true = lambda x: np.polyval(coef_true, x)
sigma, degree = 0.5, 3
ns = np.array([20, 40, 80, 160, 320, 640, 1280])
n_reps, n_test = 200, 2000
print("number of parameters p =", degree + 1, "; noise variance sigma^2 =", sigma ** 2)
"""

ex.add_project(
    title="Mini-project: the generalisation gap as a power law",
    difficulty=3,
    prompt_md=r"""
Scaling laws say loss falls as a power of data; classical theory says the generalisation gap
of a well-specified $p$-parameter model falls like $p\sigma^2/n$. Measure one and compare with
the other.

1. For each $n$ in `ns`, repeat `n_reps` times: draw $x \sim \text{Uniform}[-1, 1]$, labels
   $y = f(x) + \sigma\varepsilon$; fit a degree-3 polynomial by least squares (`np.polyfit`); record
   the training MSE, the test MSE on `n_test` fresh noisy points, and their difference (the gap).
2. Average over repetitions and plot test MSE and training MSE against $n$ (log $x$-axis) with
   the noise floor $\sigma^2$ marked; on a second log–log panel plot the mean gap against $n$.
3. Fit a power law $\text{gap} \approx c\,n^{-\gamma}$ by linear regression of $\log(\text{gap})$ on
   $\log n$; report $c$ and $\gamma$.
4. Compare with theory: for OLS with $p$ parameters, $\mathbb E[\text{train MSE}] \approx \sigma^2(1 - p/n)$
   and $\mathbb E[\text{test MSE}] \approx \sigma^2(1 + p/n)$, so the gap is $\approx 2\sigma^2 p/n$:
   $\gamma = 1$ and $c = 2\sigma^2 p = 2.0$. Overlay this line.
5. Fit the Chinchilla-style form $\text{test MSE} \approx E + B\,n^{-\beta}$ with
   `scipy.optimize.curve_fit` and compare $E$ with $\sigma^2$ and $\beta$ with $\gamma$.

The cell below fixes the setting. Keep the total runtime modest (the defaults take a few seconds).
""",
    rubric_md=r"""
- Training MSE rises towards $\sigma^2$ and test MSE falls towards it from above; both are
  plotted against $n$ with the noise floor marked, with titles, axis labels and a legend.
- The fitted exponent $\gamma$ is close to $1$ (roughly $0.9$–$1.2$) and you can say *why* it is
  $1$ and not $1/2$ (variance of a well-specified estimator scales like $1/n$; the $1/\sqrt n$ of
  Hoeffding is a worst-case *bound* on deviations, not the expected gap).
- The fitted constant $c$ is of the order of $2\sigma^2 p = 2$ (the realised value is somewhat
  larger because the uniform design makes $\mathbb E[x^\top(X^\top X)^{-1}x]$ exceed $p/n$ at small $n$).
- The scaling-law fit returns $E \approx \sigma^2 = 0.25$ — the irreducible error, i.e. the
  Bayes risk — and you can map $E$, $B n^{-\beta}$ onto the Chinchilla terms $E$, $B/D^\beta$.
- You can state what changes if the degree is *too low* (test MSE plateaus above $\sigma^2$:
  approximation error; the gap still shrinks) and *too high* (gap larger, same exponent).
- (Bonus) Repeat with degree 1 and degree 8 and overlay the three gap curves.
""",
    starter=PROJECT_DATA + r"""

# YOUR CODE HERE
# for n in ns: repeat n_reps times -> train MSE, test MSE, gap; average
# plot test/train MSE vs n (semilogx) with sigma^2 marked; plot gap vs n (loglog)
# fit log(gap) = log(c) - gamma * log(n); overlay 2 sigma^2 p / n
# fit test MSE ~ E + B n^(-beta) with scipy.optimize.curve_fit
""",
    solution_sketch_md=r"""
Loop over `ns`, and inside over `n_reps` draws; `np.polyfit(x, y, degree)` then `np.polyval` on
the training points and on 2,000 fresh test points. Average the three quantities. A linear fit
of `np.log(gap)` on `np.log(ns)` gives slope $-\gamma \approx -1.0$ and intercept $\log c$ with
$c \approx 2.5$–$3$, against the theory line $2\sigma^2 p/n = 2/n$. `curve_fit` on
`lambda n, E, B, beta: E + B * n**(-beta)` with `p0=[0.25, 2.0, 1.0]` returns $E \approx 0.25$
(the noise variance), and $\beta$ somewhat above 1 because the test MSE excess over $\sigma^2$
is $\sigma^2 p/n$ plus a small-$n$ correction. The exponent is $1$ because the estimator is
well specified and its excess risk is pure variance $\propto 1/n$; the $1/\sqrt n$ in the
Hoeffding and VC bounds is a high-probability deviation bound, not an expected excess risk.
""",
    solution_code=PROJECT_DATA + r"""

from scipy.optimize import curve_fit

train_mse = np.zeros(len(ns)); test_mse = np.zeros(len(ns)); gap = np.zeros(len(ns))
for i, n in enumerate(ns):
    tr, te = [], []
    for _ in range(n_reps):
        x = g.uniform(-1, 1, n); y = f_true(x) + sigma * g.normal(size=n)
        c = np.polyfit(x, y, degree)
        x_test = g.uniform(-1, 1, n_test); y_test = f_true(x_test) + sigma * g.normal(size=n_test)
        tr.append(np.mean((np.polyval(c, x) - y) ** 2))
        te.append(np.mean((np.polyval(c, x_test) - y_test) ** 2))
    train_mse[i], test_mse[i] = np.mean(tr), np.mean(te)
    gap[i] = test_mse[i] - train_mse[i]

slope, intercept = np.polyfit(np.log(ns), np.log(gap), 1)
gamma_fit, c_fit = -slope, np.exp(intercept)
p = degree + 1
print(f"{'n':>6s} {'train MSE':>10s} {'test MSE':>10s} {'gap':>8s} {'theory 2 sigma^2 p/n':>22s}")
for n, a, b, gp in zip(ns, train_mse, test_mse, gap):
    print(f"{n:6d} {a:10.4f} {b:10.4f} {gp:8.4f} {2 * sigma**2 * p / n:22.4f}")
print(f"\nfitted power law: gap ~ {c_fit:.2f} * n^(-{gamma_fit:.2f})     theory: {2*sigma**2*p:.2f} * n^(-1)")

law = lambda n, E, B, beta: E + B * n ** (-beta)
(E_fit, B_fit, beta_fit), _ = curve_fit(law, ns, test_mse, p0=[0.25, 2.0, 1.0], maxfev=20000)
print(f"scaling-law fit of the test MSE: E = {E_fit:.4f} (sigma^2 = {sigma**2}), B = {B_fit:.2f}, beta = {beta_fit:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].semilogx(ns, test_mse, "o-", label="test MSE")
axes[0].semilogx(ns, train_mse, "s-", label="training MSE")
axes[0].semilogx(ns, law(ns, E_fit, B_fit, beta_fit), "k--", lw=0.8, label=f"E + B n^(-beta), E = {E_fit:.3f}")
axes[0].axhline(sigma ** 2, color="gray", ls=":", label="noise floor sigma^2")
axes[0].set_title("Learning curves of a well-specified cubic fit"); axes[0].set_xlabel("n"); axes[0].set_ylabel("MSE"); axes[0].legend(fontsize=8)
axes[1].loglog(ns, gap, "o", label="measured gap (test - train)")
axes[1].loglog(ns, c_fit * ns ** (-gamma_fit), "-", label=f"fit {c_fit:.2f} n^(-{gamma_fit:.2f})")
axes[1].loglog(ns, 2 * sigma ** 2 * p / ns, "k:", label=f"theory 2 sigma^2 p / n = {2*sigma**2*p:.1f} / n")
axes[1].set_title("Generalisation gap vs n is a power law"); axes[1].set_xlabel("n"); axes[1].set_ylabel("gap"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""",
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print("exercises:", len(ex.items), "| code:", kinds.count("code"), "| concept:", kinds.count("concept"),
      "| project:", kinds.count("project"))
print("difficulties:", [it["difficulty"] for it in ex.items])
