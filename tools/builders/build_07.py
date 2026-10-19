"""
build_07.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for

    part-3-probability/07-probability-foundations

Run from the repository root:

    python3 _authoring/builders/build_07.py
    python3 tools/run_notebooks.py part-3-probability/07-probability-foundations

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

MOD = ROOT / "part-3-probability/07-probability-foundations"

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 07 — Probability Foundations")

nb.md(r"""
This notebook is the runnable companion to the concept guide: read [README.md](README.md) first,
or keep it open alongside. Every section below mirrors a numbered section of the guide. We simulate
dice and coins to watch frequencies settle on probabilities, work through the two-stage manufacturing
tree and the diagnostic-test example (including a plot of PPV against prevalence), simulate Monty Hall,
draw PMFs, PDFs and CDFs, check linearity of expectation on *dependent* variables, test Chebyshev's
inequality, estimate $\pi$ by Monte Carlo and watch the error fall like $1/\sqrt{n}$, and finally build a
naive Bayes spam filter by hand and verify it against scikit-learn.

Everything is plain NumPy first; library calls appear only afterwards, to confirm the hand computation.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
from math import comb
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
""")

# ---------------------------------------------------------------- 1. frequencies
nb.md(r"""
## 1. Frequencies settle on probabilities

The frequentist reading of $P(\text{heads}) = 0.5$ is a statement about the long run: the proportion of
heads in $n$ flips approaches $0.5$ as $n$ grows. Section 8 of the guide tells us *how fast*: the
proportion of successes in $n$ trials has standard deviation $\sqrt{p(1-p)/n} \le \frac{1}{2\sqrt{n}}$,
so we draw that band around the true value.
""")

nb.code(r"""
n_flips = 20_000
flips = rng.integers(0, 2, size=n_flips)                 # 1 = heads
n = np.arange(1, n_flips + 1)
running_freq = np.cumsum(flips) / n

plt.plot(n, running_freq, lw=1, label="running proportion of heads")
plt.axhline(0.5, color="k", ls="--", label="true probability 0.5")
plt.fill_between(n, 0.5 - 1/(2*np.sqrt(n)), 0.5 + 1/(2*np.sqrt(n)), alpha=0.2,
                 label=r"$0.5 \pm 1/(2\sqrt{n})$")
plt.xscale("log"); plt.ylim(0.2, 0.8)
plt.xlabel("number of flips n (log scale)"); plt.ylabel("proportion of heads")
plt.title("Relative frequency of heads settles on the probability")
plt.legend(loc="upper right"); plt.show()

for k in (10, 100, 1_000, 10_000, 20_000):
    print(f"after {k:>6d} flips: proportion of heads = {running_freq[k-1]:.4f}   "
          f"(band half-width 1/(2 sqrt n) = {1/(2*np.sqrt(k)):.4f})")
""")

nb.code(r"""
faces = np.arange(1, 7)
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax, n_rolls in zip(axes, (60, 600, 60_000)):
    rolls = rng.integers(1, 7, size=n_rolls)
    freq = np.array([(rolls == f).mean() for f in faces])
    ax.bar(faces, freq, alpha=0.8)
    ax.axhline(1/6, color="k", ls="--")
    ax.set_title(f"{n_rolls} rolls"); ax.set_xlabel("face")
axes[0].set_ylabel("relative frequency")
fig.suptitle("Empirical PMF of a fair die vs the true value 1/6 (dashed)")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Notice that the running proportion wanders a lot for the first few hundred flips and then hugs $0.5$
ever more tightly — but the band narrows only like $1/\sqrt{n}$: going from 100 to 10,000 flips
(100 times the work) shrinks the typical error by a factor of 10, from about $0.05$ to about $0.005$.
With the die, 60 rolls give a lumpy histogram, 600 a recognisable one, 60,000 a nearly flat one.
""")

# ---------------------------------------------------------------- 2. axioms
nb.md(r"""
## 2. Sample spaces, events and the axioms — two dice by brute force

With equally likely outcomes, probability *is* counting: $P(A) = |A| / |\Omega|$. We enumerate all 36
outcomes of two dice as a Python set, define events as subsets, and check the rules of Section 2 of
the guide — complement, disjoint additivity, inclusion–exclusion and the union bound — numerically.
""")

nb.code(r"""
from itertools import product

omega = [(i, j) for i, j in product(range(1, 7), repeat=2)]
N = len(omega)

def prob(event):                        # equally likely outcomes: probability = |A| / |Omega|
    return len(event) / N

A = {w for w in omega if sum(w) == 7}          # sum is 7
B = {w for w in omega if w[0] == w[1]}         # doubles
C = {w for w in omega if 6 in w}               # at least one six

print(f"|Omega| = {N}")
print(f"P(A: sum = 7)          = {len(A)}/36 = {prob(A):.4f}")
print(f"P(B: doubles)          = {len(B)}/36 = {prob(B):.4f}")
print(f"P(C: at least one six) = {len(C)}/36 = {prob(C):.4f}   complement rule: 1 - 25/36 = {1 - 25/36:.4f}")
print()
print(f"A and B disjoint? {len(A & B) == 0}  ->  P(A u B) = {prob(A | B):.4f}  =  P(A) + P(B) = {prob(A) + prob(B):.4f}")
print(f"A and C overlap:  P(A n C) = {prob(A & C):.4f}")
print(f"inclusion-exclusion: P(A u C) = {prob(A | C):.4f}  =  P(A) + P(C) - P(A n C) = {prob(A) + prob(C) - prob(A & C):.4f}")
print(f"union bound:         P(A u C) = {prob(A | C):.4f} <= P(A) + P(C) = {prob(A) + prob(C):.4f}")
print()
print(f"cards: P(ace or heart) = 4/52 + 13/52 - 1/52 = {(4 + 13 - 1)/52:.4f}")
print(f"cards: P(two aces in a two-card hand) = C(4,2)/C(52,2) = {comb(4, 2)}/{comb(52, 2)} = {comb(4, 2)/comb(52, 2):.5f} = 1/221")
""")

nb.md(r"""
The intersection term in inclusion–exclusion is exactly what stops you from double-counting
$(1, 6)$ and $(6, 1)$, which belong to both "sum is 7" and "at least one six". The union bound
gives up that correction and is therefore loose — but it never needs to know how the events overlap.

**Try it yourself:** define the event "the sum is at least 10" and the event "the two faces differ
by exactly 1"; compute $P$ of each, of their intersection and of their union, and confirm
inclusion–exclusion by counting.
""")

# ---------------------------------------------------------------- 3. conditional probability
nb.md(r"""
## 3. Conditional probability, total probability and (non-)independence

The two-stage manufacturing example from the guide: $P(D_1) = 0.04$, $P(D_2 \mid D_1) = 0.10$,
$P(D_2 \mid D_1^c) = 0.02$. We compute the four leaves of the probability tree with the
multiplication rule, get $P(D_2)$ from the law of total probability, invert with
$P(D_1 \mid D_2) = P(D_1 \cap D_2) / P(D_2)$, and then simulate 200,000 parts to confirm.
""")

nb.code(r"""
p_d1 = 0.04
p_d2_given_d1, p_d2_given_not_d1 = 0.10, 0.02

leaves = {
    "D1 and D2":         p_d1 * p_d2_given_d1,
    "D1 and not D2":     p_d1 * (1 - p_d2_given_d1),
    "not D1 and D2":     (1 - p_d1) * p_d2_given_not_d1,
    "not D1 and not D2": (1 - p_d1) * (1 - p_d2_given_not_d1),
}
for name, p in leaves.items():
    print(f"P({name:17s}) = {p:.4f}")
print(f"sum of the four leaves = {sum(leaves.values()):.10f}")

p_d2 = leaves["D1 and D2"] + leaves["not D1 and D2"]      # law of total probability
p_defective = 1 - leaves["not D1 and not D2"]              # complement of 'no damage anywhere'
p_d1_given_d2 = leaves["D1 and D2"] / p_d2                 # definition of conditional probability
print(f"\nP(D2) = {p_d2:.4f}    P(defective) = {p_defective:.4f}    P(D1 | D2) = {p_d1_given_d2:.4f}")
print(f"independence test: P(D1) P(D2) = {p_d1 * p_d2:.5f}  vs  P(D1 and D2) = {leaves['D1 and D2']:.5f}  -> dependent")

# simulate the two stages for 200,000 parts
n_parts = 200_000
d1 = rng.random(n_parts) < p_d1
d2 = rng.random(n_parts) < np.where(d1, p_d2_given_d1, p_d2_given_not_d1)
print(f"\nsimulation: P(D2) ~ {d2.mean():.4f}   P(D1 | D2) ~ {d1[d2].mean():.4f}   P(defective) ~ {(d1 | d2).mean():.4f}")
""")

nb.md(r"""
Two things to notice. First, the simulation of "conditioning" is literally *filtering*: `d1[d2]`
keeps only the parts for which $D_2$ happened and asks how often $D_1$ holds among them — the new,
smaller sample space. Second, $P(D_1 \mid D_2) \approx 0.17$ is more than four times $P(D_1) = 0.04$:
knowing about stage-2 damage changes what you believe about stage 1, which is exactly what
"dependent" means.
""")

# ---------------------------------------------------------------- 4. Bayes
nb.md(r"""
## 4. Bayes' theorem — the diagnostic test

Sensitivity $P(+ \mid D) = 0.99$, specificity $P(- \mid D^c) = 0.95$, prevalence $P(D) = 0.01$.
Bayes' theorem gives

$$P(D \mid +) = \frac{P(+ \mid D)\,P(D)}{P(+ \mid D)\,P(D) + P(+ \mid D^c)\,P(D^c)} .$$

We write the formula as a function, build the 10,000-person natural-frequency table, and redo the
calculation in odds form — prior odds $\times$ likelihood ratio — including a second, conditionally
independent positive test.
""")

nb.code(r"""
import pandas as pd

def ppv(se, sp, prev):
    # positive predictive value P(D | +): true positives over all positives
    tp = se * prev
    fp = (1 - sp) * (1 - prev)
    return tp / (tp + fp)

def npv(se, sp, prev):
    # negative predictive value P(not D | -)
    tn = sp * (1 - prev)
    fn = (1 - se) * prev
    return tn / (tn + fn)

se, sp, prev = 0.99, 0.95, 0.01
pop = 10_000
table = pd.DataFrame({"test +": [se * prev * pop, (1 - sp) * (1 - prev) * pop],
                      "test -": [(1 - se) * prev * pop, sp * (1 - prev) * pop]},
                     index=["diseased", "healthy"]).round(0).astype(int)
table.loc["total"] = table.sum()
table["total"] = table.sum(axis=1)
print("natural-frequency table for 10,000 people:")
print(table)

print(f"\nPPV = P(D | +)     = {ppv(se, sp, prev):.4f}   (= 99 / 594 = 1/6)")
print(f"NPV = P(not D | -) = {npv(se, sp, prev):.5f}")

prior_odds = prev / (1 - prev)
lr_pos = se / (1 - sp)                     # likelihood ratio of a positive result
post_odds = prior_odds * lr_pos
print(f"\nodds form: prior odds {prior_odds:.4f} x LR+ {lr_pos:.1f} = posterior odds {post_odds:.3f} "
      f"-> probability {post_odds / (1 + post_odds):.4f}")
post_odds_2 = post_odds * lr_pos           # sequential update with a second independent positive test
print(f"after a second positive test: odds {post_odds_2:.3f} -> probability {post_odds_2 / (1 + post_odds_2):.4f}")
print(f"prevalence at which PPV reaches 50%: (1 - sp) / (se + 1 - sp) = {(1 - sp) / (se + 1 - sp):.4f}")
""")

nb.code(r"""
prevs = np.logspace(-4, -0.0001, 400)
for s, p_ in [(0.99, 0.95), (0.99, 0.99), (0.85, 0.93), (0.99, 0.999)]:
    plt.plot(prevs * 100, ppv(s, p_, prevs), label=f"sensitivity {s}, specificity {p_}")
plt.axhline(0.5, color="k", ls=":", lw=1)
plt.axvline(1, color="gray", ls=":", lw=1)
plt.xscale("log")
plt.xlabel("prevalence in % (log scale)"); plt.ylabel("PPV = P(disease | positive test)")
plt.title("A positive test means little when the condition is rare")
plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
At 1% prevalence the 99%/95% test (blue) sits at a PPV of $1/6$; the curve for specificity 0.99
is far higher at every prevalence, while raising *sensitivity* from 0.85 to 0.99 barely moves it. At
low prevalence the false positives come from the huge healthy population, so specificity is what
counts. Each curve crosses $0.5$ at prevalence $(1 - \text{sp})/(\text{se} + 1 - \text{sp})$.

**Try it yourself:** change the specificity in the first code cell to 0.99 and re-run — how does the
PPV change, and how many false positives remain among 10,000 people?

### Monty Hall by simulation

You pick one of three doors; the host, who knows where the car is, opens a *different* door with a
goat; you may switch to the remaining door. The guide argues that switching wins with probability
$2/3$. Rather than argue, we play the game 50,000 times with the host's behaviour coded explicitly.
""")

nb.code(r"""
def monty_hall(n_games, switch, rng):
    car = rng.integers(0, 3, size=n_games)
    pick = rng.integers(0, 3, size=n_games)
    opened = np.empty(n_games, dtype=int)
    for g in range(n_games):
        # the host opens a door that is neither the pick nor the car (random choice if both qualify)
        options = [d for d in range(3) if d != pick[g] and d != car[g]]
        opened[g] = options[rng.integers(len(options))]
    final = (3 - pick - opened) if switch else pick     # door indices 0 + 1 + 2 = 3
    return (final == car).mean()

n_games = 50_000
p_stay = monty_hall(n_games, switch=False, rng=rng)
p_switch = monty_hall(n_games, switch=True, rng=rng)
se_mc = np.sqrt(p_switch * (1 - p_switch) / n_games)
print(f"P(win | stay)   ~ {p_stay:.4f}")
print(f"P(win | switch) ~ {p_switch:.4f}  +/- {se_mc:.4f}   (exact: 2/3 = {2/3:.4f})")
""")

nb.md(r"""
The simulation lands within a couple of standard errors of $2/3$. The reason: your first pick is the
car with probability $1/3$ no matter what the host does afterwards, so the *other* unopened door
carries the remaining $2/3$. The host's opening reveals information only about the door he did not
open — a conditional-probability argument that the simulation makes impossible to dispute.
""")

# ---------------------------------------------------------------- 5. random variables
nb.md(r"""
## 5. Random variables: PMF, CDF, PDF and a change of variables

A random variable is a function from outcomes to numbers. The sum $S$ of two dice maps the 36
outcomes to the integers $2, \dots, 12$; its PMF is $p_S(s) = (6 - |s - 7|)/36$ and its CDF is the
running sum. We compute both exactly from the sample space of Section 2, then draw 20,000 samples
and compare.
""")

nb.code(r"""
values = np.arange(2, 13)
pmf_exact = np.array([sum(1 for w in omega if sum(w) == s) for s in values]) / N
cdf_exact = np.cumsum(pmf_exact)

samples = rng.integers(1, 7, size=(20_000, 2)).sum(axis=1)
pmf_emp = np.array([(samples == s).mean() for s in values])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.bar(values - 0.2, pmf_exact, width=0.4, label="exact PMF")
ax1.bar(values + 0.2, pmf_emp, width=0.4, label="20,000 samples")
ax1.set_xlabel("s = sum of two dice"); ax1.set_ylabel("P(S = s)"); ax1.set_title("PMF of the dice sum"); ax1.legend()
ax2.step(np.r_[1, values], np.r_[0, cdf_exact], where="post")
ax2.axhline(0.5, color="gray", ls=":"); ax2.set_xlabel("s"); ax2.set_ylabel("F(s) = P(S <= s)")
ax2.set_title("CDF of the dice sum (median = 7)")
plt.tight_layout(); plt.show()

print("s      :", " ".join(f"{s:5d}" for s in values))
print("PMF*36 :", " ".join(f"{int(round(p*36)):5d}" for p in pmf_exact))
print("CDF    :", " ".join(f"{c:5.3f}" for c in cdf_exact))
median = values[np.searchsorted(cdf_exact, 0.5)]        # smallest s with F(s) >= 0.5
print(f"\nF(4) = {cdf_exact[values == 4][0]:.4f} (= 6/36)    median = {median}    P(S > 9) = 1 - F(9) = {1 - cdf_exact[values == 9][0]:.4f}")
""")

nb.md(r"""
### A continuous variable: densities are not probabilities

Let $X \sim \text{Uniform}(0, 1)$ and $Y = X^2$. The change-of-variables formula with
$g^{-1}(y) = \sqrt{y}$ gives $f_Y(y) = \frac{1}{2\sqrt{y}}$ on $(0, 1]$, a density that *exceeds 1*
near 0 and even diverges, yet integrates to 1. The CDF is $F_Y(y) = \sqrt{y}$ and the quantile
function $Q_Y(u) = u^2$, so the median is $0.25$. We check all of this against 100,000 samples.
""")

nb.code(r"""
x = rng.random(100_000)
y = x ** 2
grid = np.linspace(0.005, 1, 400)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.hist(y, bins=50, density=True, alpha=0.6, label="histogram of Y = X^2 (density=True)")
ax1.plot(grid, 1 / (2 * np.sqrt(grid)), "r", label=r"$f_Y(y) = 1/(2\sqrt{y})$")
ax1.set_ylim(0, 6); ax1.set_xlabel("y"); ax1.set_ylabel("density"); ax1.set_title("PDF of Y = X², X ~ Uniform(0,1)"); ax1.legend(fontsize=8)
ys = np.sort(y); ecdf = np.arange(1, len(ys) + 1) / len(ys)
ax2.plot(ys[::100], ecdf[::100], label="empirical CDF")
ax2.plot(grid, np.sqrt(grid), "r--", label=r"$F_Y(y) = \sqrt{y}$")
ax2.set_xlabel("y"); ax2.set_ylabel("F(y)"); ax2.set_title("CDF of Y"); ax2.legend()
plt.tight_layout(); plt.show()

print(f"density at y = 0.01: f_Y = {1/(2*np.sqrt(0.01)):.1f}  (> 1, allowed)   but P(Y = 0.01) = 0")
print(f"P(0 < Y <= 0.01) = integral of f_Y = sqrt(0.01) = {np.sqrt(0.01):.3f}   simulation: {(y <= 0.01).mean():.4f}")
print(f"median of Y: exact Q(0.5) = 0.5^2 = 0.25   simulation: {np.median(y):.4f}")
print(f"90th percentile: exact 0.81   simulation: {np.quantile(y, 0.9):.4f}")
print(f"check the density integrates to 1: {np.trapezoid(1/(2*np.sqrt(np.linspace(1e-9, 1, 2_000_001))), np.linspace(1e-9, 1, 2_000_001)):.4f}")
""")

nb.md(r"""
Squaring pushes mass towards 0: half of the $Y$ values are below $0.25$. The histogram with
`density=True` is the empirical density; its bars near $y = 0$ rise above 1 just as the formula says.
Keep the three objects apart: the PDF $f_Y$ (a rate), the CDF $F_Y$ (a probability), and the quantile
function $Q_Y = F_Y^{-1}$ (a value) — `np.quantile` computes the last one from samples.

**Try it yourself:** replace `y = x ** 2` by `y = -np.log(x)` and plot the histogram against
$f_Y(y) = e^{-y}$ — this is how an exponential random variable is sampled from a uniform one.
""")

# ---------------------------------------------------------------- 6. expectation
nb.md(r"""
## 6. Expectation, LOTUS and linearity — even for dependent variables

$\mathbb{E}[X] = \sum_x x\,p(x)$; by LOTUS $\mathbb{E}[g(X)] = \sum_x g(x)\,p(x)$. Linearity,
$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$, holds for **any** pair of random variables.
We test it with three increasingly dependent pairs: two independent dice, $Y = 7 - X$ (perfectly
negatively dependent), and $Y = X$ (perfectly positively dependent). Then the negative-expectation game.
""")

nb.code(r"""
pmf_die = np.full(6, 1/6)
E_X = float(faces @ pmf_die)                       # sum x p(x)
E_X2 = float((faces ** 2) @ pmf_die)               # LOTUS with g(x) = x^2
print(f"E[X] = {E_X:.4f}    E[X^2] = {E_X2:.4f}    (E[X])^2 = {E_X**2:.4f}    -> E[g(X)] != g(E[X])")

n = 200_000
X = rng.integers(1, 7, size=n)
pairs = {"independent second die": rng.integers(1, 7, size=n), "Y = 7 - X (dependent)": 7 - X, "Y = X (dependent)": X}
print(f"\n{'pair':28s} {'mean X':>8s} {'mean Y':>8s} {'mean X+Y':>10s} {'E X + E Y':>10s}")
for name, Y in pairs.items():
    print(f"{name:28s} {X.mean():8.4f} {Y.mean():8.4f} {(X + Y).mean():10.4f} {X.mean() + Y.mean():10.4f}")

# E[XY] = E[X] E[Y] only under independence
print(f"\nE[XY]: independent {np.mean(X * pairs['independent second die']):.3f} vs E[X]E[Y] = {E_X**2:.3f};"
      f"   Y = X: {np.mean(X * X):.3f} (= E[X^2])")

# indicator variables: E[1_A] = P(A); expected number of shared-birthday pairs among 23 people
print(f"\nindicator: E[1(six)] = P(six) = {np.mean(X == 6):.4f}")
print(f"expected number of birthday-sharing pairs among 23 people = C(23,2)/365 = {comb(23, 2)/365:.4f}")

# a game with negative expectation: pay 1 to roll, win 5 on a six
gain = np.where(X == 6, 4, -1)
print(f"\ngame: exact E[gain] = 4/6 - 5/6 = {4/6 - 5/6:.4f} per play;  simulated average over {n} plays = {gain.mean():.4f}")
print(f"average cumulative loss after 600 plays ~ {600 * (4/6 - 5/6):.1f}")
""")

nb.code(r"""
plays = 600
n_players = 200
gains = np.where(rng.integers(1, 7, size=(n_players, plays)) == 6, 4, -1).cumsum(axis=1)
plt.plot(gains.T, color="C0", alpha=0.08)
plt.plot(np.arange(1, plays + 1) * (-1/6), "r", lw=2, label="expected cumulative gain -n/6")
plt.xlabel("number of plays"); plt.ylabel("cumulative gain (USD)")
plt.title("200 players of a negative-expectation game")
plt.legend(); plt.show()
print(f"players ahead after {plays} plays: {(gains[:, -1] > 0).sum()} of {n_players}")
""")

nb.md(r"""
The column "mean X+Y" equals "E X + E Y" in every row, including the perfectly dependent pairs — that
is linearity. The *product* rule, by contrast, fails as soon as dependence enters:
$\mathbb{E}[X \cdot X] = 15.17 \neq 12.25$. In the game plot, individual players drift around, a few are
briefly ahead, but the cloud follows the red line $-n/6$; after 600 plays almost nobody is in profit.
""")

# ---------------------------------------------------------------- 7. variance
nb.md(r"""
## 7. Variance, standardisation and Chebyshev's inequality

$\operatorname{Var}(X) = \mathbb{E}[X^2] - \mu^2$, $\operatorname{Var}(aX + b) = a^2\operatorname{Var}(X)$, and
$\operatorname{Var}(X + Y) = \operatorname{Var}X + \operatorname{Var}Y + 2\operatorname{Cov}(X, Y)$. For the
same three pairs as above we see the covariance term at work, then we check Chebyshev's bound
$P(|X - \mu| \ge k\sigma) \le 1/k^2$ against the truth for a skewed distribution.
""")

nb.code(r"""
var_X = E_X2 - E_X ** 2
print(f"die: Var(X) = 91/6 - 3.5^2 = {var_X:.4f}   sd = {np.sqrt(var_X):.4f}")
print(f"scaling: Var(2X + 10) exact = 4 Var(X) = {4*var_X:.4f}   simulated = {np.var(2*X + 10):.4f}")

print(f"\n{'pair':28s} {'Var X':>7s} {'Var Y':>7s} {'Var(X+Y)':>9s} {'VarX+VarY':>10s} {'2 Cov':>7s}")
for name, Y in pairs.items():
    cov = np.mean((X - X.mean()) * (Y - Y.mean()))
    print(f"{name:28s} {X.var():7.3f} {Y.var():7.3f} {(X + Y).var():9.3f} {X.var() + Y.var():10.3f} {2*cov:7.3f}")

# standardisation
Z = (X - E_X) / np.sqrt(var_X)
print(f"\nz-scores of the die: mean {Z.mean():.4f}, var {Z.var():.4f};  a six has z = {(6 - E_X)/np.sqrt(var_X):.3f}")

# averages: Var(mean of n) = sigma^2 / n
for m in (1, 4, 16, 64):
    means = rng.integers(1, 7, size=(50_000, m)).mean(axis=1)
    print(f"average of {m:2d} dice: sd = {means.std():.4f}   sigma/sqrt(n) = {np.sqrt(var_X/m):.4f}")
""")

nb.code(r"""
# Chebyshev vs truth for a skewed variable: Y = X^2 with X ~ Uniform(0,1)  (mean 1/3, var 4/45)
mu_y, sd_y = 1/3, np.sqrt(4/45)
ks = np.linspace(1, 4, 50)
truth = np.array([np.mean(np.abs(y - mu_y) >= k * sd_y) for k in ks])
plt.plot(ks, 1 / ks**2, "k--", label="Chebyshev bound 1/k²")
plt.plot(ks, truth, label="true P(|Y - mu| >= k sigma), Y = X²")
plt.plot(ks, [np.mean(np.abs(rng.integers(1, 7, 100_000) - 3.5) >= k * np.sqrt(35/12)) for k in ks], label="true value for one die")
plt.xlabel("k (number of standard deviations)"); plt.ylabel("probability")
plt.title("Chebyshev's inequality is valid for every distribution — and loose")
plt.legend(); plt.show()
for k in (1.5, 2, 3):
    print(f"k = {k}: bound {1/k**2:.3f}   Y = X^2: {np.mean(np.abs(y - mu_y) >= k*sd_y):.4f}   die: {np.mean(np.abs(X - 3.5) >= k*np.sqrt(35/12)):.4f}")
""")

nb.md(r"""
In the variance table the first row (independent dice) has a covariance near 0 and the variances add;
the second row has $\operatorname{Cov} = -\operatorname{Var}X$ and $X + Y = 7$ has *zero* variance; the
third has $\operatorname{Var}(2X) = 4\operatorname{Var}X$, twice what naive addition would give. The
standard deviation of an average of $n$ dice falls like $1/\sqrt{n}$, exactly $\sigma/\sqrt{n}$.
The Chebyshev plot shows the bound sitting well above both true curves — it is a guarantee, not an
estimate — and at $k = 3$ it still allows $11\%$ in the tails where the die has $0\%$.

**Try it yourself:** in the Chebyshev cell, generate a heavy-tailed sample such as
`rng.standard_t(df=3, size=100_000)` and compare its tail probabilities with the bound.
""")

# ---------------------------------------------------------------- 8. Monte Carlo
nb.md(r"""
## 8. Monte Carlo: estimating $\pi$ and the $1/\sqrt{n}$ law

Throw $n$ uniform points into the unit square. The fraction landing inside the quarter circle
$x^2 + y^2 \le 1$ estimates $\pi/4$. The estimate is an average of indicators, so its standard
deviation is $\sqrt{p(1-p)/n}$ with $p = \pi/4$; multiplied by 4 the error of $\hat\pi$ is about
$1.64/\sqrt{n}$. We check this by repeating the whole experiment 200 times for each $n$.
""")

nb.code(r"""
def estimate_pi(n, rng):
    pts = rng.random((n, 2))
    inside = (pts ** 2).sum(axis=1) <= 1.0
    return 4 * inside.mean()

pts = rng.random((3000, 2)); inside = (pts ** 2).sum(axis=1) <= 1
plt.figure(figsize=(4, 4))
plt.scatter(pts[inside, 0], pts[inside, 1], s=3, label="inside")
plt.scatter(pts[~inside, 0], pts[~inside, 1], s=3, label="outside")
plt.gca().set_aspect("equal"); plt.xlabel("x"); plt.ylabel("y")
plt.title(f"3000 points: pi estimate = {4*inside.mean():.3f}"); plt.legend(loc="lower left", fontsize=8); plt.show()

ns = np.array([10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000])
reps = 200
rmse = []
for n_pts in ns:
    ests = np.array([estimate_pi(n_pts, rng) for _ in range(reps)])
    rmse.append(np.sqrt(np.mean((ests - np.pi) ** 2)))
rmse = np.array(rmse)
theory = 4 * np.sqrt((np.pi/4) * (1 - np.pi/4) / ns)

plt.loglog(ns, rmse, "o-", label="observed RMSE over 200 repetitions")
plt.loglog(ns, theory, "k--", label=r"theory: $4\sqrt{p(1-p)/n} \approx 1.64/\sqrt{n}$")
plt.xlabel("number of points n"); plt.ylabel("root-mean-square error of pi estimate")
plt.title("Monte Carlo error falls like 1/sqrt(n)"); plt.legend(); plt.show()
for n_pts, r, t in zip(ns, rmse, theory):
    print(f"n = {n_pts:>6d}: RMSE {r:.4f}   theory {t:.4f}")
""")

nb.md(r"""
On log–log axes the error is a straight line of slope $-1/2$: every factor of 100 in $n$ buys one
decimal digit. This is the same $\sigma/\sqrt{n}$ you saw for averages of dice, and it is the reason
Monte Carlo is cheap for two digits and ruinously expensive for six. The birthday problem works the
same way — the exercises ask you to simulate it and compare with the exact product formula.
""")

# ---------------------------------------------------------------- 9. naive Bayes
nb.md(r"""
## 9. A naive Bayes spam filter, by hand

For a message with words $w_1, \dots, w_m$:

$$\hat c = \arg\max_c \Big[\log P(c) + \sum_j \log \hat P(w_j \mid c)\Big], \qquad
\hat P(w \mid c) = \frac{\text{count}(w, c) + 1}{\text{tokens in } c + V}.$$

We build a 12-message corpus, count words per class, apply Laplace smoothing, and classify new
messages in log space. The per-word log-likelihood ratio $\log \frac{P(w \mid \text{spam})}{P(w \mid \text{ham})}$
tells us which words drive each decision. Finally we check against `sklearn.naive_bayes.MultinomialNB`.
""")

nb.code(r"""
from collections import Counter

corpus = [
    ("free money now click here",          "spam"),
    ("win a free prize now",                "spam"),
    ("claim your free offer today",         "spam"),
    ("cheap meds money back guarantee",     "spam"),
    ("you win money click now",             "spam"),
    ("exclusive offer limited time click",  "spam"),
    ("team meeting moved to tomorrow",      "ham"),
    ("can you review the project report",   "ham"),
    ("lunch today at noon",                 "ham"),
    ("notes from the meeting attached",     "ham"),
    ("project deadline is friday",          "ham"),
    ("call me back when you are free",      "ham"),
]
classes = ("spam", "ham")
docs = {c: [m.split() for m, lab in corpus if lab == c] for c in classes}
counts = {c: Counter(w for d in docs[c] for w in d) for c in classes}
vocab = sorted(set(w for d in docs["spam"] + docs["ham"] for w in d))
V = len(vocab)
n_tokens = {c: sum(counts[c].values()) for c in classes}
log_prior = {c: np.log(len(docs[c]) / len(corpus)) for c in classes}
print(f"vocabulary size V = {V};  tokens: {n_tokens};  log priors: { {c: round(v, 3) for c, v in log_prior.items()} }")

def log_word_prob(word, c, alpha=1.0):
    # Laplace-smoothed estimate of P(word | class), in log space
    return np.log((counts[c][word] + alpha) / (n_tokens[c] + alpha * V))

def classify(message, alpha=1.0, verbose=False):
    words = [w for w in message.split() if w in vocab]      # ignore words never seen in training
    scores = {c: log_prior[c] + sum(log_word_prob(w, c, alpha) for w in words) for c in classes}
    # normalise to a posterior with the log-sum-exp trick
    m = max(scores.values())
    post = {c: np.exp(s - m) / sum(np.exp(v - m) for v in scores.values()) for c, s in scores.items()}
    if verbose:
        print(f"'{message}'")
        for w in words:
            print(f"    {w:10s} log P(w|spam) = {log_word_prob(w, 'spam', alpha):7.3f}   log P(w|ham) = {log_word_prob(w, 'ham', alpha):7.3f}"
                  f"   log-likelihood ratio = {log_word_prob(w, 'spam', alpha) - log_word_prob(w, 'ham', alpha):+.3f}")
        print(f"    log scores: spam {scores['spam']:.3f}, ham {scores['ham']:.3f}   ->  P(spam | message) = {post['spam']:.3f}")
    return max(scores, key=scores.get), post["spam"]

for msg in ["free money offer", "meeting about the project report", "click to win a prize", "free lunch at the meeting"]:
    classify(msg, verbose=True)
""")

nb.code(r"""
# the words that carry the most weight, as log-likelihood ratios
llr = {w: log_word_prob(w, "spam") - log_word_prob(w, "ham") for w in vocab}
top = sorted(llr.items(), key=lambda kv: kv[1])
words_sorted = [w for w, _ in top]; vals = [v for _, v in top]
plt.figure(figsize=(7, 6))
plt.barh(words_sorted, vals, color=["C0" if v < 0 else "C3" for v in vals])
plt.axvline(0, color="k", lw=1)
plt.xlabel("log P(w | spam) - log P(w | ham)   (Laplace-smoothed)"); plt.ylabel("word")
plt.title("Which words push a message towards spam (red) or ham (blue)?")
plt.tight_layout(); plt.show()

# without smoothing a single unseen word vetoes a class
def unsmoothed_log_prob(word, c):
    p = counts[c][word] / n_tokens[c]
    return np.log(p) if p > 0 else -np.inf

msg = "win free money at the meeting".split()
for c in classes:
    vetoes = [w for w in msg if w in vocab and counts[c][w] == 0]
    score = log_prior[c] + sum(unsmoothed_log_prob(w, c) for w in msg if w in vocab)
    print(f"unsmoothed log score for '{' '.join(msg)}' under {c:4s}: {score:>6}   vetoed by {vetoes}")
print("-> every class containing one unseen word scores -inf, so the strong evidence in the other words is thrown away;")
print("   with Laplace smoothing the same message gets P(spam) =", f"{classify(' '.join(msg))[1]:.3f}")
""")

nb.code(r"""
# verify against scikit-learn's MultinomialNB (alpha=1 is Laplace smoothing)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vec = CountVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b")
X_train = vec.transform([m for m, _ in corpus])
y_train = [lab for _, lab in corpus]
clf = MultinomialNB(alpha=1.0).fit(X_train, y_train)

tests = ["free money offer", "meeting about the project report", "click to win a prize", "free lunch at the meeting"]
spam_col = list(clf.classes_).index("spam")
sk_post = clf.predict_proba(vec.transform(tests))[:, spam_col]
for msg, p_sk in zip(tests, sk_post):
    label, p_hand = classify(msg)
    print(f"{msg:35s} hand: P(spam) = {p_hand:.4f}   sklearn: {p_sk:.4f}   agree: {abs(p_hand - p_sk) < 1e-9}")
""")

nb.md(r"""
Hand computation and `MultinomialNB` agree to machine precision, because they *are* the same
formula: class log-priors from class frequencies, smoothed word log-probabilities from counts. The
bar chart is the model's entire knowledge — a single log-likelihood ratio per word — which is why
naive Bayes is so easy to interpret and so easy to fool (repeat "meeting" ten times in a spam message).
"free lunch at the meeting" shows the model weighing evidence: "free" argues for spam, "lunch" and
"meeting" for ham, and the sum decides.

**Try it yourself:** set `alpha=0.1` in `classify` — does the posterior for the ambiguous message
become more or less extreme? Then add a new ham message containing the word "free" to the corpus
and watch its log-likelihood ratio move.
""")

nb.md(r"""
## Summary

- Relative frequencies converge to probabilities at rate $1/\sqrt{n}$ — the same rate that governs every Monte Carlo estimate.
- With equally likely outcomes, probability is counting; complements, inclusion–exclusion and the union bound follow from the three axioms.
- Conditioning is filtering the sample space; the law of total probability sums over a partition; dependence shows up as $P(A \mid B) \ne P(A)$.
- Bayes' theorem inverts conditionals; at low prevalence the PPV is driven by specificity, and sequential evidence multiplies likelihood ratios in odds form.
- PMFs sum to 1, PDFs integrate to 1 and may exceed 1, CDFs give probabilities of intervals, quantile functions invert them; $f_Y = f_X(g^{-1})\,|(g^{-1})'|$ for monotone transformations.
- Expectation is linear for *all* random variables; $\mathbb{E}[g(X)] \ne g(\mathbb{E}[X])$ and $\mathbb{E}[XY] \ne \mathbb{E}[X]\mathbb{E}[Y]$ in general.
- Variances add only up to a covariance term; standardisation gives $z$-scores; Chebyshev bounds tails for every distribution with finite variance.
- Naive Bayes is Bayes' theorem plus conditional independence plus Laplace smoothing, computed in log space — and it matches `MultinomialNB` exactly.

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
from math import comb, factorial
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
    "Module 07 — Exercises",
    intro_md=r"""
These exercises follow the order of the concept guide: counting and the axioms, conditional probability
and Bayes' theorem, random variables, expectation and variance, Monte Carlo, and finally naive Bayes.
Several checks compare your answer against a *hidden* value, so compute carefully rather than guessing.
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- 1. compound events (warm-up)
SOL_1 = r"""
from itertools import product
omega = [(i, j) for i, j in product(range(1, 7), repeat=2)]

def p_event(predicate):
    return sum(1 for w in omega if predicate(w)) / len(omega)

p_sum_ge_10 = p_event(lambda w: sum(w) >= 10)
p_diff_1 = p_event(lambda w: abs(w[0] - w[1]) == 1)
p_sum_ge_10_or_diff_1 = p_event(lambda w: sum(w) >= 10 or abs(w[0] - w[1]) == 1)
p_two_aces = comb(4, 2) / comb(52, 2)
p_at_least_one_ace_5cards = 1 - comb(48, 5) / comb(52, 5)
"""
ns1 = run(SOL_1)
ex.add(
    title="Compound events by counting", difficulty=1,
    prompt_md=r"""
Two fair dice are rolled; a 52-card deck is shuffled. Using only counting (enumerate the 36 outcomes of
the dice as pairs; use `comb` for the cards), assign these five **floats**:

* `p_sum_ge_10` — the sum of the two dice is at least 10;
* `p_diff_1` — the two faces differ by exactly 1;
* `p_sum_ge_10_or_diff_1` — the union of the two events above (check your answer against inclusion–exclusion);
* `p_two_aces` — a two-card hand consists of two aces;
* `p_at_least_one_ace_5cards` — a five-card hand contains at least one ace (hint: complement).
""",
    starter=r"""
from itertools import product
omega = [(i, j) for i, j in product(range(1, 7), repeat=2)]   # all 36 outcomes

p_sum_ge_10 = None
p_diff_1 = None
p_sum_ge_10_or_diff_1 = None
p_two_aces = None
p_at_least_one_ace_5cards = None
# YOUR CODE HERE
""",
    solution=SOL_1,
    check=rf"""
check_close('P(sum >= 10) = 6/36', p_sum_ge_10, 6/36)
check_hash('P(|difference| = 1)', p_diff_1, '{sha_of(ns1["p_diff_1"])}')
check_hash('P(sum >= 10 or |difference| = 1)', p_sum_ge_10_or_diff_1, '{sha_of(ns1["p_sum_ge_10_or_diff_1"])}')
try:
    ie = p_sum_ge_10 + p_diff_1 - p_sum_ge_10_or_diff_1
except Exception:
    ie = None
check_hash('the intersection implied by inclusion-exclusion', ie, '{sha_of(ns1["p_sum_ge_10"] + ns1["p_diff_1"] - ns1["p_sum_ge_10_or_diff_1"])}')
check_close('P(two aces) = 1/221', p_two_aces, 1/221)
check_hash('P(at least one ace in 5 cards)', p_at_least_one_ace_5cards, '{sha_of(ns1["p_at_least_one_ace_5cards"])}')
""",
    explanation_md=r"""
With equally likely outcomes, probability is a ratio of counts. "Sum $\ge 10$" has 6 outcomes
$\{(4,6),(5,5),(6,4),(5,6),(6,5),(6,6)\}$; "differ by 1" has 10; they share $(5,6)$ and $(6,5)$, so the
union has $6 + 10 - 2 = 14$ outcomes — inclusion–exclusion in action. For the cards,
$\binom{4}{2}/\binom{52}{2} = 6/1326 = 1/221$, and "at least one ace" is easiest through its complement
"no ace": $1 - \binom{48}{5}/\binom{52}{5} \approx 0.341$. The common wrong turn is to compute
$P(\text{at least one ace}) = 5 \times 4/52$, which double-counts hands with several aces (that is the union
bound, an overestimate).
""",
)

# ---------------------------------------------------------------- 2. total probability (warm-up)
SOL_2 = r"""
def p_defective(shares, defect_rates):
    shares = np.asarray(shares, dtype=float); defect_rates = np.asarray(defect_rates, dtype=float)
    return float(shares @ defect_rates)

def p_supplier_given_defective(shares, defect_rates):
    shares = np.asarray(shares, dtype=float); defect_rates = np.asarray(defect_rates, dtype=float)
    joint = shares * defect_rates
    return joint / joint.sum()
"""
ns2 = run(SOL_2, r"""
pd_ = p_defective([0.5, 0.3, 0.2], [0.01, 0.02, 0.05])
post = p_supplier_given_defective([0.5, 0.3, 0.2], [0.01, 0.02, 0.05])
""")
ex.add(
    title="Law of total probability: three suppliers", difficulty=1,
    prompt_md=r"""
A factory buys the same component from three suppliers, who deliver shares $P(S_1), P(S_2), P(S_3)$ of
the stock with defect rates $P(D \mid S_i)$. Implement

* `p_defective(shares, defect_rates)` → the overall defect probability $P(D) = \sum_i P(D \mid S_i)P(S_i)$ as a float;
* `p_supplier_given_defective(shares, defect_rates)` → a NumPy array with $P(S_i \mid D)$ for each supplier (Bayes' theorem, one entry per supplier, summing to 1).

Both take Python lists or arrays of equal length. Test case: shares `[0.5, 0.3, 0.2]`, defect rates
`[0.01, 0.02, 0.05]`.
""",
    starter=r"""
def p_defective(shares, defect_rates):
    # YOUR CODE HERE
    return None

def p_supplier_given_defective(shares, defect_rates):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_2,
    check=rf"""
try:
    got = p_defective([0.5, 0.3, 0.2], [0.01, 0.02, 0.05])
except Exception:
    got = None
check_hash('P(defective) for the three suppliers', got, '{sha_of(ns2["pd_"], decimals=4)}', decimals=4)
try:
    got_single = p_defective([1.0], [0.07])
except Exception:
    got_single = None
check_close('single supplier: P(D) = its defect rate', got_single, 0.07)
try:
    post = np.asarray(p_supplier_given_defective([0.5, 0.3, 0.2], [0.01, 0.02, 0.05]), dtype=float)
    ok_shape = post.shape == (3,)
except Exception:
    post, ok_shape = None, False
check('posterior has shape (3,)', ok_shape)
check_hash('P(S_3 | defective)', (float(post[2]) if ok_shape else None), '{sha_of(float(ns2["post"][2]))}')
check_close('posterior sums to 1', (float(post.sum()) if ok_shape else None), 1.0)
""",
    explanation_md=r"""
The suppliers partition the stock, so $P(D) = 0.5 \cdot 0.01 + 0.3 \cdot 0.02 + 0.2 \cdot 0.05 = 0.021$.
Bayes' theorem then divides each joint probability $P(D \mid S_i)P(S_i)$ by that total: supplier 3 delivers
only 20% of the parts but $0.010/0.021 \approx 48\%$ of the defects. A dot product does the whole law of
total probability in one line; normalising the vector of joints gives all three posteriors at once — the
pattern `joint / joint.sum()` is Bayes' theorem for any finite set of hypotheses.
""",
)

# ---------------------------------------------------------------- 3. PPV (core)
SOL_3 = r"""
def ppv(sensitivity, specificity, prevalence):
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    return tp / (tp + fp)

def prevalence_for_ppv(sensitivity, specificity, target=0.5):
    # solve se*pi / (se*pi + (1-sp)(1-pi)) = target for pi
    fpr = 1 - specificity
    return target * fpr / (sensitivity * (1 - target) + target * fpr)

def ppv_after_two_positives(sensitivity, specificity, prevalence):
    prior_odds = prevalence / (1 - prevalence)
    lr = sensitivity / (1 - specificity)
    post_odds = prior_odds * lr * lr
    return post_odds / (1 + post_odds)
"""
ns3 = run(SOL_3, r"""
a = ppv(0.90, 0.97, 0.002)
b = prevalence_for_ppv(0.90, 0.97, 0.5)
c = ppv_after_two_positives(0.90, 0.97, 0.002)
d = prevalence_for_ppv(0.85, 0.93, 0.9)
""")
ex.add(
    title="Bayes' theorem: PPV, the break-even prevalence and a second test", difficulty=2,
    prompt_md=r"""
Implement three functions (all arguments and results are floats):

* `ppv(sensitivity, specificity, prevalence)` — $P(D \mid +)$;
* `prevalence_for_ppv(sensitivity, specificity, target=0.5)` — the prevalence at which the PPV equals `target` (solve the PPV formula for the prevalence; for `target=0.5` the answer is $\frac{1 - \text{sp}}{\text{se} + 1 - \text{sp}}$);
* `ppv_after_two_positives(sensitivity, specificity, prevalence)` — the posterior probability of disease after **two** positive results from conditionally independent tests (use the odds form: multiply the prior odds by the likelihood ratio twice).

Test case: a screening test with sensitivity 0.90 and specificity 0.97 for a condition with prevalence 0.2%.
""",
    starter=r"""
def ppv(sensitivity, specificity, prevalence):
    # YOUR CODE HERE
    return None

def prevalence_for_ppv(sensitivity, specificity, target=0.5):
    # YOUR CODE HERE
    return None

def ppv_after_two_positives(sensitivity, specificity, prevalence):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_3,
    check=rf"""
try:
    got = ppv(0.99, 0.95, 0.01)
except Exception:
    got = None
check_close('the guide example: PPV = 1/6', got, 1/6, tol=1e-6)
try:
    got = ppv(0.90, 0.97, 0.002)
except Exception:
    got = None
check_hash('PPV for se 0.90, sp 0.97, prevalence 0.002', got, '{sha_of(ns3["a"], decimals=4)}', decimals=4)
try:
    got = prevalence_for_ppv(0.90, 0.97, 0.5)
except Exception:
    got = None
check_hash('prevalence at which PPV = 50%', got, '{sha_of(ns3["b"], decimals=4)}', decimals=4)
try:
    got = ppv(0.90, 0.97, prevalence_for_ppv(0.90, 0.97, 0.5))
except Exception:
    got = None
check_close('consistency: ppv at that prevalence is 0.5', got, 0.5)
try:
    got = prevalence_for_ppv(0.85, 0.93, 0.9)
except Exception:
    got = None
check_hash('prevalence needed for PPV = 90% with se 0.85, sp 0.93', got, '{sha_of(ns3["d"], decimals=4)}', decimals=4)
try:
    got = ppv_after_two_positives(0.90, 0.97, 0.002)
except Exception:
    got = None
check_hash('posterior after two positive tests', got, '{sha_of(ns3["c"], decimals=4)}', decimals=4)
""",
    explanation_md=r"""
PPV is Bayes' theorem with the evidence expanded by total probability. Setting it equal to a target $t$ and
solving gives $\pi = \frac{t(1 - \text{sp})}{\text{se}(1 - t) + t(1 - \text{sp})}$, which reduces to
$\frac{1 - \text{sp}}{\text{se} + 1 - \text{sp}}$ at $t = 0.5$. For the second test the odds form is
cleanest: prior odds $0.002/0.998$, likelihood ratio $0.90/0.03 = 30$, so two positives multiply the odds
by 900 and take the posterior from about 5.7% to about 64%. The usual mistake is to feed the first
posterior back into `ppv` as the new prevalence — which is correct and gives the same number — but to then
*also* forget that this assumes the two test results are conditionally independent given the disease
status.
""",
)

# ---------------------------------------------------------------- 4. birthday problem (core)
SOL_4 = r"""
def birthday_exact(n, days=365):
    p_all_different = 1.0
    for i in range(n):
        p_all_different *= (days - i) / days
    return 1.0 - p_all_different

def birthday_simulated(n, n_trials, rng, days=365):
    draws = rng.integers(0, days, size=(n_trials, n))
    draws.sort(axis=1)
    shared = (np.diff(draws, axis=1) == 0).any(axis=1)
    return float(shared.mean())

def smallest_group_for(threshold, days=365):
    n = 1
    while birthday_exact(n, days) < threshold:
        n += 1
    return n
"""
ns4 = run(SOL_4, r"""
e23 = birthday_exact(23)
e50 = birthday_exact(50)
n99 = smallest_group_for(0.99)
sim = birthday_simulated(23, 20_000, np.random.default_rng(0))
""")
ex.add(
    title="The birthday problem, exactly and by simulation", difficulty=2,
    prompt_md=r"""
Implement

* `birthday_exact(n, days=365)` — the exact probability that among $n$ people at least two share a birthday, via the product $1 - \prod_{i=0}^{n-1}\frac{\text{days}-i}{\text{days}}$;
* `birthday_simulated(n, n_trials, rng, days=365)` — a Monte Carlo estimate from `n_trials` random groups, drawing birthdays with `rng.integers(0, days, size=(n_trials, n))` (vectorised: sort each row and look for equal neighbours, or use any other correct method);
* `smallest_group_for(threshold, days=365)` — the smallest $n$ with `birthday_exact(n) >= threshold`.

Then compare: for $n = 23$ and 20,000 trials, how far is the simulation from the exact value relative to
the standard error $\sqrt{p(1-p)/20{,}000}$?
""",
    starter=r"""
def birthday_exact(n, days=365):
    # YOUR CODE HERE
    return None

def birthday_simulated(n, n_trials, rng, days=365):
    # YOUR CODE HERE
    return None

def smallest_group_for(threshold, days=365):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_4,
    check=rf"""
try:
    got = birthday_exact(23)
except Exception:
    got = None
check_hash('exact probability for 23 people', got, '{sha_of(ns4["e23"], decimals=4)}', decimals=4)
try:
    got = birthday_exact(2)
except Exception:
    got = None
check_close('two people: 1/365', got, 1/365)
try:
    got = birthday_exact(366)
except Exception:
    got = None
check_close('366 people: certainty', got, 1.0)
try:
    got = birthday_exact(50)
except Exception:
    got = None
check_hash('exact probability for 50 people', got, '{sha_of(ns4["e50"], decimals=4)}', decimals=4)
try:
    got = smallest_group_for(0.99)
except Exception:
    got = None
check_hash('smallest group with P >= 0.99', got, '{sha_of(ns4["n99"])}')
try:
    sim = birthday_simulated(23, 20_000, np.random.default_rng(0))
    ex_ = birthday_exact(23)
    within = abs(sim - ex_) < 4 * np.sqrt(ex_ * (1 - ex_) / 20_000)
except Exception:
    within = False
check('simulation for 23 people is within 4 standard errors of the exact value', within)
""",
    explanation_md=r"""
The complement "all different" is a chain of conditional probabilities: the second person avoids 1 day,
the third avoids 2, and so on, giving $\prod_{i<n}(365 - i)/365 \approx 0.493$ for $n = 23$. The simulation
sorts each group's birthdays and asks whether any adjacent pair is equal, which is a vectorised way of
testing "not all distinct". With 20,000 trials the standard error is about 0.0035, so a well-written
simulation lands within 0.01 of 0.507; if yours is further away than four standard errors, it is wrong,
not unlucky. The 99% threshold needs only 57 people — the quadratic growth of the number of *pairs*,
$\binom{n}{2}$, is what makes the answer so much smaller than intuition suggests.
""",
)

# ---------------------------------------------------------------- 5. PMF / expectation / variance (core)
SOL_5 = r"""
def check_pmf(values, probs):
    probs = np.asarray(probs, dtype=float)
    return bool(np.all(probs >= 0) and np.isclose(probs.sum(), 1.0))

def expectation(values, probs, g=lambda x: x):
    values = np.asarray(values, dtype=float); probs = np.asarray(probs, dtype=float)
    return float(np.sum(g(values) * probs))

def variance(values, probs):
    mu = expectation(values, probs)
    return expectation(values, probs, g=lambda x: x**2) - mu**2

def cdf(values, probs, x):
    values = np.asarray(values, dtype=float); probs = np.asarray(probs, dtype=float)
    return float(probs[values <= x].sum())
"""
ns5 = run(SOL_5, r"""
vals = [-2, 0, 1, 3, 10]
pr = [0.1, 0.3, 0.3, 0.2, 0.1]
mu = expectation(vals, pr)
var = variance(vals, pr)
e_abs = expectation(vals, pr, g=np.abs)
e_exp = expectation(vals, pr, g=lambda x: np.exp(-x))
c1 = cdf(vals, pr, 1)
""")
ex.add(
    title="PMF, expectation, LOTUS and variance from a table", difficulty=2,
    prompt_md=r"""
A discrete random variable is given as a table of `values` and `probs` (lists of equal length). Implement,
using NumPy only (no `scipy.stats`):

* `check_pmf(values, probs)` → `True` if all probabilities are non-negative and sum to 1 (use `np.isclose`), else `False`;
* `expectation(values, probs, g=lambda x: x)` → $\mathbb{E}[g(X)] = \sum_x g(x)\,p(x)$ as a float (LOTUS; `g` must accept a NumPy array);
* `variance(values, probs)` → $\operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ as a float;
* `cdf(values, probs, x)` → $F(x) = P(X \le x)$ as a float.

Test distribution: `values = [-2, 0, 1, 3, 10]`, `probs = [0.1, 0.3, 0.3, 0.2, 0.1]`.
""",
    starter=r"""
def check_pmf(values, probs):
    # YOUR CODE HERE
    return None

def expectation(values, probs, g=lambda x: x):
    # YOUR CODE HERE
    return None

def variance(values, probs):
    # YOUR CODE HERE
    return None

def cdf(values, probs, x):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_5,
    check=rf"""
vals = [-2, 0, 1, 3, 10]
pr = [0.1, 0.3, 0.3, 0.2, 0.1]
try:
    ok1 = check_pmf(vals, pr) is True and check_pmf(vals, [0.5, 0.5, 0.5, 0.5, 0.5]) is False and check_pmf([1, 2], [1.5, -0.5]) is False
except Exception:
    ok1 = False
check('check_pmf accepts a valid PMF and rejects two invalid ones', ok1)
try:
    got = expectation([1, 2, 3, 4, 5, 6], [1/6]*6)
except Exception:
    got = None
check_close('E[die] = 3.5', got, 3.5)
try:
    got = expectation(vals, pr)
except Exception:
    got = None
check_hash('E[X] for the test distribution', got, '{sha_of(ns5["mu"])}')
try:
    got = expectation(vals, pr, g=lambda x: np.exp(-x))
except Exception:
    got = None
check_hash('LOTUS: E[exp(-X)]', got, '{sha_of(ns5["e_exp"])}')
try:
    got = variance(vals, pr)
except Exception:
    got = None
check_hash('Var(X) for the test distribution', got, '{sha_of(ns5["var"])}')
try:
    got = variance([1, 2, 3, 4, 5, 6], [1/6]*6)
except Exception:
    got = None
check_close('Var(die) = 35/12', got, 35/12)
try:
    got = cdf(vals, pr, 1)
except Exception:
    got = None
check_hash('F(1) for the test distribution', got, '{sha_of(ns5["c1"])}')
try:
    got = (cdf(vals, pr, -5), cdf(vals, pr, 10), cdf(vals, pr, 2.5))
except Exception:
    got = None
check_close('F(-5) = 0, F(10) = 1, F(2.5) = F(1)', got, (0.0, 1.0, 0.7))
""",
    explanation_md=r"""
Everything is a weighted sum. LOTUS says you never need the distribution of $g(X)$: apply $g$ to the
support and reuse the same weights, so `expectation` with a `g` argument covers $\mathbb{E}[X]$,
$\mathbb{E}[X^2]$, $\mathbb{E}[e^{-X}]$ and any other moment. The variance shortcut
$\mathbb{E}[X^2] - \mu^2$ is linearity applied to $(X - \mu)^2$; for the test distribution
$\mu = 1.7$, $\mathbb{E}[X^2] = 12.5$ and $\operatorname{Var} = 9.61$. The CDF is the sum of probabilities over `values <= x`
— a boolean mask — and must be a step function that is right-continuous: $F(2.5) = F(1)$ because no mass
lies in $(1, 2.5]$. A frequent slip is to compute $\mathbb{E}[X^2]$ as `expectation(values**2, probs)`
with a Python list, which raises a `TypeError`; convert to an array first.
""",
)

# ---------------------------------------------------------------- conceptual (warm-up, placed third)
ex.add_conceptual(
    title="Independent or mutually exclusive?", difficulty=1,
    prompt_md=r"""
A colleague argues: "Our two fraud rules never fire on the same transaction — rule A catches card-not-present
fraud, rule B catches account takeovers. Since they never overlap, the two alerts are independent, so I can
multiply their probabilities to get the chance that a transaction triggers both."

1. Identify the error. Are events that never overlap independent? Prove your answer with the definition
   $P(A \cap B) = P(A)P(B)$, assuming $P(A) = 0.02$ and $P(B) = 0.01$.
2. What is $P(A \mid B)$ for mutually exclusive events with positive probability, and what does that say
   about dependence?
3. Give one example of two events that *are* independent and one of two that are mutually exclusive, both
   from the same experiment (e.g. two dice), and compute the relevant probabilities.
""",
    answer_md=r"""
1. Mutually exclusive means $A \cap B = \emptyset$, so $P(A \cap B) = 0$. Independence would require
   $P(A \cap B) = P(A)P(B) = 0.02 \times 0.01 = 0.0002 \ne 0$. The two notions are incompatible whenever both
   probabilities are positive: the colleague's "multiply the probabilities" gives $0.0002$ for an event that
   has probability exactly $0$.
2. $P(A \mid B) = P(A \cap B)/P(B) = 0$, whereas $P(A) = 0.02$. Learning that $B$ occurred changes the
   probability of $A$ (to zero), which is the definition of *dependence* — mutually exclusive events are as
   dependent as events can be.
3. Two dice: "first die shows 6" and "second die shows 6" are independent, $P = \frac{1}{6}\cdot\frac{1}{6} = \frac{1}{36}$,
   and indeed exactly one of the 36 outcomes is $(6, 6)$. "Sum is 7" and "doubles" are mutually exclusive
   (7 is odd), so $P(\text{both}) = 0$ while $P(\text{sum } 7)\,P(\text{doubles}) = \frac{1}{6}\cdot\frac{1}{6} \ne 0$,
   and $P(\text{sum } 7 \cup \text{doubles}) = \frac{6}{36} + \frac{6}{36} = \frac{1}{3}$ by additivity.
""",
)
ex.items.insert(2, ex.items.pop())          # make it Exercise 3, right after total probability

# ---------------------------------------------------------------- 6. inverse-CDF sampler (core)
SOL_6 = r"""
def quantile_discrete(values, probs, u):
    values = np.asarray(values, dtype=float); F = np.cumsum(np.asarray(probs, dtype=float))
    k = int(np.argmax(F >= u))            # first index with F(k) >= u
    return float(values[k])

def sample_discrete(values, probs, n, rng):
    values = np.asarray(values, dtype=float); F = np.cumsum(np.asarray(probs, dtype=float))
    u = rng.random(n)
    k = np.searchsorted(F, u, side="right")   # first index with u < F(k)
    k = np.minimum(k, len(values) - 1)        # guard against F[-1] slightly below 1
    return values[k]
"""
ns6 = run(SOL_6, r"""
vals = [-2, 0, 1, 3, 10]
pr = [0.1, 0.3, 0.3, 0.2, 0.1]
s = sample_discrete(vals, pr, 10_000, np.random.default_rng(123))
s_sum = int(round(float(s.sum())))
s_first = [float(x) for x in s[:5]]
q = [quantile_discrete(vals, pr, u) for u in (0.05, 0.5, 0.8, 0.99)]
""")
ex.add(
    title="Inverse-CDF sampling for a discrete distribution", difficulty=2,
    prompt_md=r"""
Any distribution can be sampled from uniforms: draw $U \sim \text{Uniform}(0, 1)$ and return
$Q(U)$, where $Q(u) = \min\{x : F(x) \ge u\}$ is the quantile function. Implement, without
`rng.choice`:

* `quantile_discrete(values, probs, u)` → $Q(u)$ as a float, for the discrete distribution given by the two lists;
* `sample_discrete(values, probs, n, rng)` → a NumPy array of `n` samples, obtained **exactly** as follows:
  `u = rng.random(n)`, then for each $u_i$ return `values[k]` for the first index `k` with $u_i < F(k)$, where
  $F$ is the cumulative sum of `probs` (use `np.searchsorted(F, u, side="right")` or an equivalent loop).

Test distribution again: `values = [-2, 0, 1, 3, 10]`, `probs = [0.1, 0.3, 0.3, 0.2, 0.1]`.
""",
    starter=r"""
def quantile_discrete(values, probs, u):
    # YOUR CODE HERE
    return None

def sample_discrete(values, probs, n, rng):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_6,
    check=rf"""
vals = [-2, 0, 1, 3, 10]
pr = [0.1, 0.3, 0.3, 0.2, 0.1]
try:
    got = [quantile_discrete(vals, pr, u) for u in (0.05, 0.5, 0.8, 0.99)]
except Exception:
    got = None
check_hash('Q(0.05), Q(0.5), Q(0.8), Q(0.99)', got, '{sha_of(ns6["q"])}')
try:
    s = np.asarray(sample_discrete(vals, pr, 10_000, np.random.default_rng(123)), dtype=float)
    ok_shape = s.shape == (10_000,)
except Exception:
    s, ok_shape = None, False
check('returns 10,000 samples', ok_shape)
check('samples are drawn from the support, not indices', ok_shape and set(np.unique(s).tolist()) <= set(float(v) for v in vals))
try:
    freqs = np.array([(s == v).mean() for v in vals])
    ok_freq = ok_shape and np.all(np.abs(freqs - np.array(pr)) < 0.02)
except Exception:
    ok_freq = False
check('empirical frequencies match the PMF to within 0.02', ok_freq)
check_hash('sum of the 10,000 seeded samples', (int(round(float(s.sum()))) if ok_shape else None), '{sha_of(ns6["s_sum"])}')
check_hash('first five seeded samples', ([float(x) for x in s[:5]] if ok_shape else None), '{sha_of(ns6["s_first"])}')
""",
    explanation_md=r"""
The cumulative sums $F = (0.1, 0.4, 0.7, 0.9, 1.0)$ cut $[0, 1)$ into intervals whose lengths are exactly
the probabilities; a uniform $U$ lands in interval $k$ with probability $p_k$, so returning `values[k]`
reproduces the PMF. `np.searchsorted(F, u, side="right")` finds the first index with $F(k) > u$, which is the
quantile function evaluated at $u$ (with the convention that handles ties). The clip `np.minimum(k, len-1)`
protects against `F[-1]` being $0.9999999$ because of floating-point addition. This is precisely what
`rng.choice(values, p=probs)` does internally, and the same idea — $X = F^{-1}(U)$ — samples continuous
distributions in Module 18. A common slip is to return the *index* `k` instead of `values[k]`, or to use
`side="left"`, which differs only when `u` equals a cumulative sum exactly (probability zero).
""",
)

# ---------------------------------------------------------------- 7. change of variables (core)
SOL_7 = r"""
def pdf_y(y):
    y = np.asarray(y, dtype=float)
    inside = (y > 0) & (y < 8)
    out = np.zeros_like(y)
    out[inside] = 0.5 * (1/3) * y[inside] ** (-2/3)     # f_X(g^-1(y)) * |d/dy g^-1(y)|,  g^-1(y) = y^(1/3)
    return out

def cdf_y(y):
    y = np.asarray(y, dtype=float)
    return np.clip(np.cbrt(np.clip(y, 0, None)) / 2, 0, 1)   # P(X <= y^(1/3)) = y^(1/3) / 2

def quantile_y(u):
    return (2 * np.asarray(u, dtype=float)) ** 3            # invert u = y^(1/3) / 2
"""
ns7 = run(SOL_7, r"""
p2 = float(pdf_y(2.0)); c2 = float(cdf_y(2.0)); q9 = float(quantile_y(0.9))
edges = np.linspace(0, 8, 200_001); mid = 0.5 * (edges[:-1] + edges[1:])
integral = float(np.sum(pdf_y(mid)) * (8 / 200_000))
assert abs(integral - 1) < 0.01, integral
""")
ex.add(
    title="Change of variables: the density of Y = X³", difficulty=2,
    prompt_md=r"""
Let $X \sim \text{Uniform}(0, 2)$, so $f_X(x) = \tfrac{1}{2}$ on $(0, 2)$, and let $Y = X^3$. Derive the density of
$Y$ with the change-of-variables formula $f_Y(y) = f_X(g^{-1}(y))\,\left|\tfrac{d}{dy} g^{-1}(y)\right|$ and
implement three vectorised functions (they must accept floats and NumPy arrays and return floats/arrays):

* `pdf_y(y)` — the density, equal to 0 outside the support $(0, 8)$;
* `cdf_y(y)` — $F_Y(y) = P(Y \le y)$, equal to 0 below the support and 1 above it;
* `quantile_y(u)` — $Q_Y(u) = F_Y^{-1}(u)$ for $u \in [0, 1]$.

Then confirm numerically that `pdf_y` integrates to 1 over $(0, 8)$ (the check uses a midpoint rule, because the density has an integrable singularity at 0) and that it exceeds 1 close to 0.
""",
    starter=r"""
def pdf_y(y):
    # YOUR CODE HERE
    return None

def cdf_y(y):
    # YOUR CODE HERE
    return None

def quantile_y(u):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_7,
    check=rf"""
try:
    got = float(pdf_y(2.0))
except Exception:
    got = None
check_hash('f_Y(2)', got, '{sha_of(ns7["p2"], decimals=4)}', decimals=4)
try:
    got = float(np.asarray(pdf_y(np.array([0.001]))).ravel()[0])
except Exception:
    got = None
check('density exceeds 1 near zero', got is not None and got > 1)
try:
    got = np.asarray(pdf_y(np.array([-1.0, 9.0])), dtype=float)
except Exception:
    got = None
check_close('density is 0 outside (0, 8)', got, np.array([0.0, 0.0]))
try:
    edges = np.linspace(0, 8, 200_001)                       # midpoint rule copes with the singularity at 0
    mid = 0.5 * (edges[:-1] + edges[1:])
    integral = float(np.sum(pdf_y(mid)) * (8 / 200_000))
except Exception:
    integral = None
check('density integrates to 1 (midpoint rule, within 0.01)', integral is not None and abs(integral - 1) < 0.01)
try:
    got = float(cdf_y(2.0))
except Exception:
    got = None
check_hash('F_Y(2)', got, '{sha_of(ns7["c2"], decimals=4)}', decimals=4)
try:
    got = np.asarray(cdf_y(np.array([-3.0, 1.0, 8.0, 50.0])), dtype=float)
except Exception:
    got = None
check_close('F_Y at -3, 1 (the median), 8 and 50', got, np.array([0.0, 0.5, 1.0, 1.0]))
try:
    got = float(quantile_y(0.9))
except Exception:
    got = None
check_hash('Q_Y(0.9)', got, '{sha_of(ns7["q9"], decimals=3)}', decimals=3)
try:
    u = np.linspace(0.01, 0.99, 50)
    got = np.asarray(cdf_y(quantile_y(u)), dtype=float)
except Exception:
    got = None
check_close('cdf_y(quantile_y(u)) == u', got, np.linspace(0.01, 0.99, 50), tol=1e-9)
""",
    explanation_md=r"""
$g(x) = x^3$ is strictly increasing with inverse $g^{-1}(y) = y^{1/3}$ and derivative $\tfrac{1}{3}y^{-2/3}$, so
$f_Y(y) = \tfrac{1}{2}\cdot\tfrac{1}{3}y^{-2/3} = \tfrac{1}{6}y^{-2/3}$ on $(0, 8)$. The density blows up at 0 —
cubing squeezes the whole interval $(0, 1)$ of $X$ into $(0, 1)$ of $Y$ *non-uniformly* and pushes mass towards 0 —
yet $\int_0^8 \tfrac{1}{6}y^{-2/3}\,dy = \tfrac{1}{6}\cdot 3\,[y^{1/3}]_0^8 = 1$. The CDF is easier to derive directly:
$P(Y \le y) = P(X \le y^{1/3}) = y^{1/3}/2$; the median of $Y$ is $1 = 1^3$, the cube of the median of $X$ (quantiles
commute with monotone increasing transformations, densities do not). The classic mistake is to write
$f_Y(y) = f_X(y^{1/3})$ and forget the Jacobian, which gives a "density" of $\tfrac12$ on $(0, 8)$ integrating to 4.
""",
)

# ---------------------------------------------------------------- 8. Monte Carlo (core)
SOL_8 = r"""
def mc_estimate(sample_fn, n, rng):
    x = np.asarray(sample_fn(n, rng), dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))

def max_of_three_uniforms(n, rng):
    return rng.random((n, 3)).max(axis=1)

def sum_of_three_uniforms_exceeds_2(n, rng):
    return (rng.random((n, 3)).sum(axis=1) > 2).astype(float)

def three_dice_sum_at_least_15(n, rng):
    return (rng.integers(1, 7, size=(n, 3)).sum(axis=1) >= 15).astype(float)

from itertools import product
p_exact_three_dice = sum(1 for w in product(range(1, 7), repeat=3) if sum(w) >= 15) / 216
"""
ns8 = run(SOL_8)
ex.add(
    title="Monte Carlo estimates and their standard errors", difficulty=2,
    prompt_md=r"""
Implement a generic estimator and three sampling functions:

* `mc_estimate(sample_fn, n, rng)` → the tuple `(estimate, standard_error)` where `sample_fn(n, rng)` returns an
  array of `n` values, `estimate` is their mean and `standard_error` is their sample standard deviation
  (`ddof=1`) divided by $\sqrt{n}$;
* `max_of_three_uniforms(n, rng)` → `n` values of $\max(U_1, U_2, U_3)$ with independent $U_i \sim \text{Uniform}(0,1)$;
* `sum_of_three_uniforms_exceeds_2(n, rng)` → `n` indicator values (0.0/1.0) of the event $U_1 + U_2 + U_3 > 2$;
* `three_dice_sum_at_least_15(n, rng)` → `n` indicator values of "the sum of three fair dice is at least 15".

Also compute `p_exact_three_dice`, the **exact** probability of the last event by enumerating all $6^3$ outcomes.
Then run each estimator with $n = 100{,}000$ and compare with the exact values $3/4$, $1/6$ and your enumeration:
every estimate should be within a few standard errors.
""",
    starter=r"""
def mc_estimate(sample_fn, n, rng):
    # YOUR CODE HERE
    return None

def max_of_three_uniforms(n, rng):
    # YOUR CODE HERE
    return None

def sum_of_three_uniforms_exceeds_2(n, rng):
    # YOUR CODE HERE
    return None

def three_dice_sum_at_least_15(n, rng):
    # YOUR CODE HERE
    return None

p_exact_three_dice = None
# YOUR CODE HERE
""",
    solution=SOL_8,
    check=rf"""
try:
    est, se = mc_estimate(lambda n, rng: np.arange(n) % 2, 1000, rng)
except Exception:
    est, se = None, None
check_close('estimate for an alternating 0/1 sequence of length 1000 is 0.5', est, 0.5)
check_close('its standard error is std(ddof=1)/sqrt(n)', se, np.std(np.arange(1000) % 2, ddof=1) / np.sqrt(1000))
def _within(fn, exact, n=100_000):
    try:
        e, s = mc_estimate(fn, n, np.random.default_rng(7))
        return s > 0 and abs(e - exact) < 4 * s and s < 0.01
    except Exception:
        return False
check('E[max of three uniforms] is within 4 SE of 3/4', _within(max_of_three_uniforms, 3/4))
check('P(U1 + U2 + U3 > 2) is within 4 SE of 1/6', _within(sum_of_three_uniforms_exceeds_2, 1/6))
check_hash('exact P(sum of three dice >= 15)', p_exact_three_dice, '{sha_of(ns8["p_exact_three_dice"], decimals=4)}', decimals=4)
check('P(three dice sum >= 15) is within 4 SE of the exact value', p_exact_three_dice is not None and _within(three_dice_sum_at_least_15, p_exact_three_dice))
try:
    out = np.asarray(sum_of_three_uniforms_exceeds_2(500, rng), dtype=float)
    ok_ind = out.shape == (500,) and set(np.unique(out).tolist()) <= {{0.0, 1.0}}
except Exception:
    ok_ind = False
check('indicator sampler returns 0/1 values of the right shape', ok_ind)
""",
    explanation_md=r"""
A Monte Carlo estimate is a sample mean, so its standard error is $\hat\sigma/\sqrt{n}$ — for an indicator,
$\sqrt{\hat p(1-\hat p)/n}$. The exact values make the point: $\mathbb{E}[\max(U_1, U_2, U_3)] = 3/4$ (the
maximum of $k$ uniforms has density $k y^{k-1}$ and mean $k/(k+1)$), $P(U_1 + U_2 + U_3 > 2) = 1/6$ (the
corner of the unit cube above the plane $x + y + z = 2$ is a tetrahedron of volume $1/6$), and the dice sum is
at least 15 in $10 + 6 + 3 + 1 = 20$ of 216 outcomes. With $n = 100{,}000$ the standard errors are about
$0.0006$, $0.0012$ and $0.0009$, so an estimate that is off by $0.01$ signals a bug, not bad luck — this is
why the estimate must always be reported *with* its standard error. The frequent error is to divide the
standard deviation by $n$ instead of $\sqrt{n}$, or to report the standard deviation of the *samples*
(which does not shrink with $n$ at all) as the uncertainty of the *estimate*.
""",
)

# ---------------------------------------------------------------- 10. naive Bayes (stretch)
SOL_10 = r"""
from collections import Counter

class NaiveBayesText:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, messages, labels):
        self.classes_ = sorted(set(labels))
        docs = {c: [m.split() for m, lab in zip(messages, labels) if lab == c] for c in self.classes_}
        self.vocab_ = sorted(set(w for c in self.classes_ for d in docs[c] for w in d))
        V = len(self.vocab_)
        self.log_prior_ = {c: np.log(len(docs[c]) / len(messages)) for c in self.classes_}
        self.log_lik_ = {}
        for c in self.classes_:
            counts = Counter(w for d in docs[c] for w in d)
            n_c = sum(counts.values())
            self.log_lik_[c] = {w: np.log((counts[w] + self.alpha) / (n_c + self.alpha * V)) for w in self.vocab_}
        return self

    def log_scores(self, message):
        words = [w for w in message.split() if w in self.log_lik_[self.classes_[0]]]
        return {c: self.log_prior_[c] + sum(self.log_lik_[c][w] for w in words) for c in self.classes_}

    def predict_proba(self, message):
        s = self.log_scores(message)
        m = max(s.values())
        z = sum(np.exp(v - m) for v in s.values())
        return {c: float(np.exp(v - m) / z) for c, v in s.items()}

    def predict(self, message):
        s = self.log_scores(message)
        return max(self.classes_, key=lambda c: (s[c], c))
"""
NB_DATA = r"""
train_msgs = [
    "free money now click here", "win a free prize now", "claim your free offer today",
    "cheap meds money back guarantee", "you win money click now", "exclusive offer limited time click",
    "urgent your account needs money", "congratulations you win a prize",
    "team meeting moved to tomorrow", "can you review the project report", "lunch today at noon",
    "notes from the meeting attached", "project deadline is friday", "call me back when you are free",
    "the budget review is on monday", "send me the report before lunch",
]
train_labels = ["spam"] * 8 + ["ham"] * 8
test_msgs = [
    "free prize money",
    "the project meeting is tomorrow",
    "click now to claim your money",
    "lunch with the team on friday",
    "you are free to review the budget",
    "urgent offer today",
    "notes from the deadline review",
    "win free lunch",
]
"""
ns10 = run(SOL_10, NB_DATA + r"""
model = NaiveBayesText(alpha=1.0).fit(train_msgs, train_labels)
preds = [model.predict(m) for m in test_msgs]
pred_str = " ".join(preds)
p_spam_first = model.predict_proba(test_msgs[0])["spam"]
p_spam_last = model.predict_proba(test_msgs[-1])["spam"]
V = len(model.vocab_)
ll_free_spam = model.log_lik_["spam"]["free"]
preds_small = " ".join(NaiveBayesText(alpha=0.1).fit(train_msgs, train_labels).predict(m) for m in test_msgs)
""")
ex.add(
    title="A naive Bayes spam filter with Laplace smoothing", difficulty=3,
    prompt_md=r"""
Implement a class `NaiveBayesText(alpha=1.0)` for bag-of-words text classification, built only from
`collections.Counter`, `np.log` and `np.exp` (no scikit-learn):

* `fit(messages, labels)` — stores `self.classes_` (sorted list of labels), `self.vocab_` (sorted list of all
  distinct words in the training messages, split on whitespace),
  `self.log_prior_[c]` $= \log(n_c / n)$ where $n_c$ is the number of training messages in class $c$, and
  `self.log_lik_[c][w]` $= \log\frac{\text{count}(w, c) + \alpha}{\text{tokens in } c + \alpha V}$
  for every class $c$ and every vocabulary word $w$ ($V$ = vocabulary size); returns `self`.
* `log_scores(message)` — a dict `{class: log_prior + sum of log_lik over the message's words}`, **ignoring**
  words not in the vocabulary.
* `predict_proba(message)` — a dict of posterior probabilities, normalised with the log-sum-exp trick
  (subtract the maximum score before exponentiating).
* `predict(message)` — the class with the highest score.

The training corpus and the eight test messages are given below. The check hashes the string of your eight
predictions joined by single spaces (e.g. `"spam ham ..."`).
""" + "\n```python\n" + textwrap.dedent(NB_DATA).strip() + "\n```\n",
    starter=NB_DATA + r"""

from collections import Counter

class NaiveBayesText:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, messages, labels):
        # YOUR CODE HERE  (set self.classes_, self.vocab_, self.log_prior_, self.log_lik_)
        return self

    def log_scores(self, message):
        # YOUR CODE HERE
        return None

    def predict_proba(self, message):
        # YOUR CODE HERE
        return None

    def predict(self, message):
        # YOUR CODE HERE
        return None
""",
    solution=NB_DATA + "\n" + SOL_10,
    check=rf"""
try:
    model = NaiveBayesText(alpha=1.0).fit(train_msgs, train_labels)
    V = len(model.vocab_)
except Exception:
    model, V = None, None
check_hash('vocabulary size', V, '{sha_of(ns10["V"])}')
try:
    got = float(model.log_lik_["spam"]["free"])
except Exception:
    got = None
check_hash('log P(free | spam) with alpha = 1', got, '{sha_of(ns10["ll_free_spam"], decimals=4)}', decimals=4)
try:
    got = float(np.exp(model.log_prior_["ham"]))
except Exception:
    got = None
check_close('P(ham) = 0.5', got, 0.5)
try:
    got = sum(np.exp(v) for v in model.log_lik_["ham"].values())
except Exception:
    got = None
check_close('smoothed word probabilities sum to 1 within each class', got, 1.0, tol=1e-9)
try:
    probs = model.predict_proba(test_msgs[0])
    got = float(probs["spam"]); ok_norm = abs(sum(probs.values()) - 1) < 1e-9
except Exception:
    got, ok_norm = None, False
check('posterior probabilities sum to 1', ok_norm)
check_hash('P(spam | "free prize money")', got, '{sha_of(ns10["p_spam_first"], decimals=3)}', decimals=3)
try:
    got = float(model.predict_proba(test_msgs[-1])["spam"])
except Exception:
    got = None
check_hash('P(spam | "win free lunch")', got, '{sha_of(ns10["p_spam_last"], decimals=3)}', decimals=3)
try:
    got = " ".join(model.predict(m) for m in test_msgs)
except Exception:
    got = None
check_hash('the eight predictions (alpha = 1)', got, '{sha_of(ns10["pred_str"])}')
try:
    got = " ".join(NaiveBayesText(alpha=0.1).fit(train_msgs, train_labels).predict(m) for m in test_msgs)
except Exception:
    got = None
check_hash('the eight predictions (alpha = 0.1)', got, '{sha_of(ns10["preds_small"])}')
""",
    explanation_md=r"""
`fit` is pure counting: class frequencies give the prior, per-class word counts plus the pseudo-count $\alpha$
give the likelihoods, and the denominator $n_c + \alpha V$ makes the smoothed probabilities of all $V$
vocabulary words sum to exactly 1 within each class (one of the checks). Scoring adds logarithms instead of
multiplying probabilities, and the log-sum-exp trick — subtract the largest score before exponentiating —
turns scores into posteriors without overflow or underflow. Words outside the vocabulary carry no
information for either class and are skipped. Two subtleties decide the ambiguous test messages: with
$\alpha = 1$ an unseen word costs $\log\frac{1}{n_c + V}$, which is *less* severe than with $\alpha = 0.1$, so
the smaller pseudo-count makes the classifier more confident and can flip borderline cases. The classic
bug is to compute the vocabulary per class instead of over the whole corpus, which makes the two
denominators inconsistent and silently biases the comparison.
""",
)

# ---------------------------------------------------------------- 11. mini-project (★★★)
PROJECT_CODE = r"""
rng = np.random.default_rng(2024)

def simulate_schedule(n_sims, rng):
    # Phase A tasks: design (uniform 8-14 days) and procurement (lognormal, median 10 days, 30% dispersion)
    design = rng.uniform(8, 14, size=n_sims)
    procurement = rng.lognormal(mean=np.log(10), sigma=0.3, size=n_sims)
    phase_a = np.maximum(design, procurement)                 # build can start only when both are done
    # Phase B: build (triangular 15-20-30 days)
    build = rng.triangular(15, 20, 30, size=n_sims)
    # dependency: a late procurement (> 12 days) means parts arrive in a rush and build takes 20% longer
    build = np.where(procurement > 12, 1.2 * build, build)
    # Phase C: testing; 1 in 4 projects needs a second test cycle (geometric number of cycles, p = 0.75)
    cycles = rng.geometric(0.75, size=n_sims)
    testing = rng.uniform(4, 6, size=(n_sims,)) * cycles
    return phase_a + build + testing, dict(design=design, procurement=procurement, build=build, testing=testing)

n_sims = 100_000
total, parts = simulate_schedule(n_sims, rng)
deadline = 45

p_on_time = (total <= deadline).mean()
se = np.sqrt(p_on_time * (1 - p_on_time) / n_sims)
q90 = np.quantile(total, 0.90)
naive_sum = 11 + 10 + (15 + 20 + 30) / 3 + 5 * (1 / 0.75)   # sum of all task means: treats parallel tasks as sequential
e_max = np.maximum(parts["design"], parts["procurement"]).mean()
print(f"mean duration        : {total.mean():6.2f} days")
print(f"naive sum of all task means: {naive_sum:.2f} days (too pessimistic: design and procurement run in parallel)")
print(f"E[max(design, procurement)] = {e_max:.2f}  >=  max of the means = 11.00  (the expected maximum is later than the later mean)")
print(f"median duration      : {np.median(total):6.2f} days")
print(f"P(finish within {deadline} d): {p_on_time:.4f} +/- {se:.4f}")
print(f"90% completion time  : {q90:6.2f} days   (promise this, not the mean)")
print(f"99% completion time  : {np.quantile(total, 0.99):6.2f} days")

plt.hist(total, bins=80, density=True, alpha=0.7)
plt.axvline(deadline, color="r", ls="--", label=f"deadline {deadline} d  (P(on time) = {p_on_time:.2f})")
plt.axvline(q90, color="k", ls=":", label=f"90% quantile = {q90:.1f} d")
plt.xlabel("total project duration (days)"); plt.ylabel("density")
plt.title("Monte Carlo distribution of the project duration"); plt.legend(); plt.show()

# sensitivity: which task drives lateness?  compare task means for late vs on-time projects
late = total > deadline
print("\nmean task durations, on-time vs late projects:")
for name, arr in parts.items():
    print(f"  {name:12s} on time {arr[~late].mean():6.2f}   late {arr[late].mean():6.2f}")
print(f"\nP(late | procurement > 12 d) = {late[parts['procurement'] > 12].mean():.3f}   vs   P(late | procurement <= 12 d) = {late[parts['procurement'] <= 12].mean():.3f}")

# convergence of the estimate with n
ns_ = [100, 1_000, 10_000, 100_000]
print("\nP(on time) by sample size:")
for m in ns_:
    t, _ = simulate_schedule(m, np.random.default_rng(1))
    p = (t <= deadline).mean()
    print(f"  n = {m:>7d}: {p:.4f}  +/- {np.sqrt(p*(1-p)/m):.4f}")
"""
ex.add_project(
    title="Mini-project: Monte Carlo risk model of a project schedule",
    prompt_md=r"""
A small engineering project has three phases. Build a Monte Carlo model of its total duration and answer two
questions a project manager actually asks: *what is the probability of finishing within 45 days?* and
*what completion time can we promise with 90% confidence?*

**The model** (use `rng = np.random.default_rng(2024)` and 100,000 simulations):

1. **Phase A** — two tasks run in parallel: *design*, Uniform(8, 14) days, and *procurement*, log-normal with
   median 10 days and $\sigma = 0.3$ on the log scale (`rng.lognormal(mean=np.log(10), sigma=0.3)`). Phase A
   ends when **both** are finished (take the maximum).
2. **Phase B** — *build*, Triangular(15, 20, 30) days. **Dependency:** if procurement took more than 12 days the
   parts arrive in a rush and build takes 20% longer.
3. **Phase C** — *testing*, Uniform(4, 6) days per test cycle; the number of cycles is Geometric with success
   probability 0.75 (`rng.geometric(0.75)`, so one cycle with probability 0.75, two with 0.1875, ...).

**Deliverables**

* `P(total <= 45)` with its Monte Carlo standard error, and the 90% (and 99%) quantile of the total duration;
* a histogram of the total duration with the deadline and the 90% quantile marked;
* a comparison of the Monte Carlo mean with the naive "sum of the expected task durations of all five
  tasks" (which treats the parallel Phase-A tasks as sequential and ignores the dependency and the random
  number of cycles) — explain the direction of the difference and which effects push which way;
* a sensitivity analysis: which task differs most between on-time and late projects, and
  $P(\text{late} \mid \text{procurement} > 12)$ versus $P(\text{late} \mid \text{procurement} \le 12)$;
* the estimate of `P(total <= 45)` for $n = 100, 1{,}000, 10{,}000, 100{,}000$ with standard errors, to show the
  $1/\sqrt{n}$ convergence.
""",
    rubric_md=r"""
* The simulation is vectorised over the 100,000 runs and seeded, so results are reproducible.
* The Phase-A maximum, the procurement → build dependency and the random number of test cycles are all
  implemented (not replaced by their expected values).
* The probability of finishing on time is reported **with** a standard error, and the 90% quantile is read off
  the simulated distribution with `np.quantile`.
* The histogram has a title, axis labels and marked deadline/quantile.
* The naive sum-of-means is computed and its error is explained: parallel tasks must be combined by a
  maximum, not a sum, and $\mathbb{E}[\max(D, P)] \ge \max(\mathbb{E}D, \mathbb{E}P)$ (the expected maximum is
  later than the later mean), while the dependency can only add delay.
* The sensitivity analysis uses conditional probabilities computed by filtering the simulated runs.
""",
    solution_sketch_md=r"""
Sample every task for all runs at once (arrays of length 100,000), combine them with `np.maximum`, `np.where`
for the dependency, and multiplication for the test cycles, and sum. The simulated mean is about 41.7 days.
The naive "sum of the means of all tasks" ($11 + 10 + 21.67 + 6.67 \approx 49.3$) is *too pessimistic* here,
because design and procurement run in **parallel**: the correct Phase-A contribution is
$\mathbb{E}[\max(D, P)] \approx 12.2$, not $\mathbb{E}D + \mathbb{E}P = 21$. Two effects push the other way and
are easy to forget: $\mathbb{E}[\max(D, P)] \ge \max(\mathbb{E}D, \mathbb{E}P) = 11$ (the maximum of random
variables is on average later than the later of the two means), and the late-procurement penalty only ever
*adds* time. $P(\text{on time})$ comes out near 0.74 with a standard error of about 0.0014; the 90% quantile
near 50 days is the number to put in the contract, five days beyond the 45-day target. Conditioning on
procurement taking more than 12 days raises the probability of lateness from about 0.12 to about 0.62, which
identifies procurement as the risk to manage. The convergence table shows the standard error shrinking from
about 0.04 at $n = 100$ to about 0.0014 at $n = 100{,}000$ — a factor $\sqrt{1000} \approx 32$.
""",
    solution_code=PROJECT_CODE,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print(f"exercises: {len(ex.items)} items — {kinds.count('code')} coding, {kinds.count('concept')} conceptual, {kinds.count('project')} project")
print("difficulties:", [it["difficulty"] for it in ex.items])
