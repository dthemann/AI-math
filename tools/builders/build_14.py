"""
build_14.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for

    part-5-learning-theory/14-information-theory

Run from the repository root:

    python3 _authoring/builders/build_14.py
    python3 tools/run_notebooks.py part-5-learning-theory/14-information-theory

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

MOD = ROOT / "part-5-learning-theory/14-information-theory"

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 14 — Information Theory")

nb.md(r"""
This notebook is the runnable companion to the concept guide: read [README.md](README.md) first, or keep
it open alongside. Every section below mirrors a numbered section of the guide. We measure the surprise
of coin flips, dice and a rare diagnosis; draw the binary entropy curve and compute the entropy of English
letters; build a Huffman coder with `heapq` and compare its expected length with the entropy; compute
cross-entropy and KL divergence and watch forward and reverse KL fit a Gaussian to two-humped data in
opposite ways; estimate mutual information from a table and from samples; draw Jensen's inequality and the
ELBO; solve a maximum-entropy problem; compute information gain for a triage table and for the iris data;
sweep the temperature of a softmax; and finish with the capacity of a noisy channel.

Everything is plain NumPy first; library calls (`scipy.stats.entropy`, `mutual_info_regression`,
`DecisionTreeClassifier`) appear only afterwards, to confirm the hand computation.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)

def entropy_bits(p):
    # H(p) = -sum p log2 p, with the convention 0 log 0 = 0 (terms with p = 0 are dropped)
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def binary_entropy(p):
    return entropy_bits([p, 1 - p])

LN2 = np.log(2)   # 1 bit = ln 2 nats;  1 nat = 1/ln 2 = 1.4427 bits
""")

# ---------------------------------------------------------------- 1. surprise
nb.md(r"""
## 1. Information as surprise

The information in an outcome of probability $p$ is $I = -\log_2 p$ bits (or $-\ln p$ nats). It is zero
for a certain event, grows without bound as $p \to 0$, and is *additive* for independent events because
their probabilities multiply. We check these on a biased coin, dice and a rare disease.
""")

nb.code(r"""
def surprise_bits(p):
    return -np.log2(p)

events = [
    ("biased coin: heads, p = 0.9",            0.9),
    ("biased coin: tails, p = 0.1",            0.1),
    ("fair die: one face, p = 1/6",            1/6),
    ("two dice: one outcome, p = 1/36",        1/36),
    ("rare disease: 'you have it', p = 1e-4",  1e-4),
    ("rare disease: 'you do not', p = 0.9999", 1 - 1e-4),
]
print(f"{'event':45s} {'bits':>8s} {'nats':>8s}")
for name, p in events:
    print(f"{name:45s} {surprise_bits(p):8.4f} {-np.log(p):8.4f}")

print(f"\nadditivity: I(1/36) = {surprise_bits(1/36):.4f}  =  I(1/6) + I(1/6) = {2 * surprise_bits(1/6):.4f}")
print(f"unit conversion: 1 nat = {1/LN2:.4f} bits,  1 bit = {LN2:.4f} nats")
print(f"twenty yes/no questions distinguish up to 2^20 = {2**20:,} objects")

p_grid = np.linspace(0.005, 1, 400)
plt.plot(p_grid, -np.log2(p_grid))
for name, p in events[:3]:
    plt.plot(p, -np.log2(p), "o", color="C3")
    plt.annotate(f"p = {p:.3g}", (p, -np.log2(p)), textcoords="offset points", xytext=(6, 4), fontsize=8)
plt.xlabel("probability p of the outcome"); plt.ylabel("information  -log2 p  (bits)")
plt.title("Rarer outcomes carry more information"); plt.show()
""")

nb.md(r"""
Tails on the 90/10 coin carries $3.32$ bits, twenty times the $0.15$ bits of heads: the logarithm
stretches small probabilities. Being told you *do not* have a one-in-ten-thousand disease carries
$0.00014$ bits — you already knew. And the two-dice outcome carries exactly twice the information of one
die, which is the additivity that forces the logarithm in the first place.
""")

# ---------------------------------------------------------------- 2. entropy
nb.md(r"""
## 2. Entropy: average surprise

$H(X) = -\sum_x p(x)\log_2 p(x)$ is the expected surprise. For a Bernoulli$(p)$ variable it is the binary
entropy $H_b(p)$, maximal (1 bit) at $p = \tfrac12$. We draw it, compute the entropy of a loaded die, and
then the entropy of English letters from their frequencies — compared with the uniform bound $\log_2 26$.
""")

nb.code(r"""
# standard English letter frequencies in percent (a widely used table; sums to ~100)
letter_freq = {'a': 8.167, 'b': 1.492, 'c': 2.782, 'd': 4.253, 'e': 12.702, 'f': 2.228, 'g': 2.015,
               'h': 6.094, 'i': 6.966, 'j': 0.153, 'k': 0.772, 'l': 4.025, 'm': 2.406, 'n': 6.749,
               'o': 7.507, 'p': 1.929, 'q': 0.095, 'r': 5.987, 's': 6.327, 't': 9.056, 'u': 2.758,
               'v': 0.978, 'w': 2.360, 'x': 0.150, 'y': 1.974, 'z': 0.074}
letters = np.array(list(letter_freq.keys()))
p_letters = np.array(list(letter_freq.values())); p_letters = p_letters / p_letters.sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ps = np.linspace(0, 1, 401)
ax1.plot(ps, [binary_entropy(p) for p in ps])
for p in (0.1, 0.5, 0.9):
    ax1.plot(p, binary_entropy(p), "o", color="C3")
ax1.set_xlabel("p = P(X = 1)"); ax1.set_ylabel("H_b(p)  (bits)"); ax1.set_title("Binary entropy function")
order = np.argsort(p_letters)[::-1]
ax2.bar(letters[order], p_letters[order])
ax2.axhline(1/26, color="k", ls="--", lw=1, label="uniform 1/26")
ax2.set_xlabel("letter"); ax2.set_ylabel("frequency")
ax2.set_title(f"English letters: H = {entropy_bits(p_letters):.3f} bits  (uniform: {np.log2(26):.3f})")
ax2.legend(); plt.tight_layout(); plt.show()

for p in (0.5, 0.9, 0.99, 0.999):
    print(f"H_b({p}) = {binary_entropy(p):.4f} bits")
loaded = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
print(f"\nfair die: H = log2 6 = {np.log2(6):.4f} bits;   loaded die {loaded}: H = {entropy_bits(loaded):.4f} bits")
H_letters = entropy_bits(p_letters)
print(f"English letters: H = {H_letters:.4f} bits/letter;  uniform bound log2 26 = {np.log2(26):.4f};  "
      f"'effective alphabet' 2^H = {2**H_letters:.1f} letters")

# the library version agrees (scipy.stats.entropy returns nats by default)
from scipy.stats import entropy as sp_entropy
print(f"scipy.stats.entropy(p_letters, base=2) = {sp_entropy(p_letters, base=2):.4f}")
""")

nb.md(r"""
Single-letter frequencies alone bring English from $4.70$ to $4.18$ bits per letter — as unpredictable as a
uniform choice among about 18 letters rather than 26. Shannon's guessing experiments, which use long
context, push the figure down to roughly 1 bit per letter; the gap between $4.18$ and $1$ is the information
carried by *spelling and grammar*, which a unigram model cannot see and a language model can.

**Try it yourself:** replace `loaded` by `[0.95, 0.01, 0.01, 0.01, 0.01, 0.01]` — how many bits does the die
produce per roll now, and how does that compare with $H_b(0.05)$?

### Joint and conditional entropy, and the chain rule

For the weather–umbrella table of the guide we compute $H(X)$, $H(Y)$, $H(X, Y)$, then $H(Y \mid X)$ two
ways: by the chain rule $H(X, Y) - H(X)$ and directly as $\sum_x p(x)\,H(Y \mid X = x)$.
""")

nb.code(r"""
# rows: X = weather (sun, rain); columns: Y = umbrella (no, yes)
joint = np.array([[0.5, 0.1],
                  [0.1, 0.3]])
p_x, p_y = joint.sum(axis=1), joint.sum(axis=0)
H_X, H_Y, H_XY = entropy_bits(p_x), entropy_bits(p_y), entropy_bits(joint)
print(f"H(X) = {H_X:.4f}   H(Y) = {H_Y:.4f}   H(X,Y) = {H_XY:.4f} bits")
print(f"chain rule:   H(Y|X) = H(X,Y) - H(X) = {H_XY - H_X:.4f}")

cond_direct = sum(p_x[i] * entropy_bits(joint[i] / p_x[i]) for i in range(2))
for i, w in enumerate(["sun", "rain"]):
    print(f"   H(Y | X = {w:4s}) = {entropy_bits(joint[i] / p_x[i]):.4f}   (weight p(x) = {p_x[i]:.1f})")
print(f"direct:       H(Y|X) = sum_x p(x) H(Y|X=x) = {cond_direct:.4f}")
print(f"\nconditioning reduces uncertainty: H(Y) - H(Y|X) = {H_Y - (H_XY - H_X):.4f} bits  (= the mutual information, Section 5)")
print(f"independent check: if X, Y were independent, H(X,Y) would be H(X) + H(Y) = {H_X + H_Y:.4f} > {H_XY:.4f}")
""")

nb.md(r"""
### Differential entropy and why it is not "information"

For a Gaussian, $h = \tfrac12\log_2(2\pi e\sigma^2)$ bits. It goes *negative* for small $\sigma$, and if you
quantise the variable into bins of width $\Delta$ the discrete entropy is about $h + \log_2(1/\Delta)$,
which diverges as the bins shrink. Only differences of differential entropies (KL, mutual information) are
meaningful.
""")

nb.code(r"""
from scipy.stats import norm

def gaussian_diff_entropy_bits(sigma):
    return 0.5 * np.log2(2 * np.pi * np.e * sigma**2)

for s in (0.1, 1.0, 10.0):
    print(f"sigma = {s:5.1f}: differential entropy h = {gaussian_diff_entropy_bits(s):7.3f} bits  ({gaussian_diff_entropy_bits(s) * LN2:6.3f} nats)")

print("\nquantised standard normal (exact bin probabilities from the CDF):")
for delta in (1.0, 0.1, 0.01):
    edges = np.arange(-12, 12 + delta, delta)
    probs = np.diff(norm.cdf(edges))
    H_disc = entropy_bits(probs)
    print(f"  bin width {delta:5.2f}: discrete entropy {H_disc:7.3f} bits   vs   h + log2(1/delta) = {gaussian_diff_entropy_bits(1.0) + np.log2(1/delta):7.3f}")
""")

# ---------------------------------------------------------------- 3. coding
nb.md(r"""
The discrete entropy grows by $\log_2 10 = 3.32$ bits each time the bins get ten times finer: there is no
finite "number of bits in a real number". The differential entropy is what remains after subtracting that
divergent part, which is why it can be negative without contradiction.

## 3. Source coding: Huffman codes and the entropy bound

Shannon: every prefix code has expected length $L \ge H$, and an optimal one has $L < H + 1$. Huffman's
algorithm — repeatedly merge the two least probable symbols — builds an optimal prefix code. We implement
it with a priority queue, check the Kraft sum $\sum_i 2^{-\ell_i}$, and compare $L$ with $H$ for the
five-symbol example of the guide and for English letters.
""")

nb.code(r"""
import heapq, itertools

def huffman_code(probs):
    # probs: dict symbol -> probability.  Returns dict symbol -> binary codeword (string).
    counter = itertools.count()                       # tie-breaker so tuples never compare nodes
    heap = [(p, next(counter), sym) for sym, p in probs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        p1, _, left = heapq.heappop(heap)             # the two least probable nodes ...
        p2, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (p1 + p2, next(counter), (left, right)))   # ... become one node
    root = heap[0][2]
    codes = {}
    def walk(node, prefix):
        if isinstance(node, tuple):
            walk(node[0], prefix + "0"); walk(node[1], prefix + "1")
        else:
            codes[node] = prefix or "0"
    walk(root, "")
    return codes

def expected_length(probs, codes):
    return sum(p * len(codes[s]) for s, p in probs.items())

def kraft_sum(codes):
    return sum(2.0 ** -len(c) for c in codes.values())

p5 = {"A": 0.4, "B": 0.2, "C": 0.2, "D": 0.1, "E": 0.1}
codes5 = huffman_code(p5)
print("five-symbol source:")
for s in p5:
    print(f"  {s}: p = {p5[s]:.1f}   code = {codes5[s]:4s}   length {len(codes5[s])}   ideal -log2 p = {-np.log2(p5[s]):.3f}")
H5 = entropy_bits(list(p5.values()))
print(f"  entropy H = {H5:.4f} bits   expected length L = {expected_length(p5, codes5):.4f} bits   Kraft sum = {kraft_sum(codes5):.3f}")
print(f"  overhead L - H = {expected_length(p5, codes5) - H5:.4f} bits ({100 * (expected_length(p5, codes5) / H5 - 1):.1f}%)   fixed-length code would need {int(np.ceil(np.log2(5)))} bits")

dyadic = {"A": 0.5, "B": 0.25, "C": 0.125, "D": 0.0625, "E": 0.0625}
cd = huffman_code(dyadic)
print(f"\ndyadic source: L = {expected_length(dyadic, cd):.4f} = H = {entropy_bits(list(dyadic.values())):.4f} bits  (powers of two -> no overhead)")

p_en = {str(s): float(p) for s, p in zip(letters, p_letters)}
codes_en = huffman_code(p_en)
L_en = expected_length(p_en, codes_en)
print(f"\nEnglish letters: H = {H_letters:.4f}   Huffman L = {L_en:.4f} bits/letter   Kraft sum = {kraft_sum(codes_en):.3f}   fixed-length: 5 bits")
print("  shortest codes:", {s: codes_en[s] for s in map(str, letters[order][:4])})
print("  longest codes: ", {s: codes_en[s] for s in map(str, letters[order][-3:])})
""")

nb.code(r"""
lengths_en = np.array([len(codes_en[s]) for s in letters[order]])
ideal_en = -np.log2(p_letters[order])
xpos = np.arange(26)
plt.bar(xpos - 0.2, lengths_en, width=0.4, label="Huffman code length")
plt.bar(xpos + 0.2, ideal_en, width=0.4, label="ideal length -log2 p")
plt.xticks(xpos, letters[order]); plt.xlabel("letter (most to least frequent)"); plt.ylabel("bits")
plt.title("Huffman lengths track -log2 p, rounded to integers"); plt.legend(); plt.show()

# arithmetic coding: a whole text costs -sum log2 p(c) + 2 bits under the model -- no rounding per symbol
text = ("information is the resolution of uncertainty and a good model of the next symbol "
        "is a good compressor of the whole text")
chars = [c for c in text if c in letter_freq]
p_model = dict(zip(letters, p_letters))
bits_arith = -sum(np.log2(p_model[c]) for c in chars) + 2
bits_huff = sum(len(codes_en[c]) for c in chars)
n = len(chars)
print(f"text of {n} letters:")
print(f"  fixed 5-bit code   : {5 * n:5d} bits  ({5:.3f} bits/letter)")
print(f"  Huffman (unigram)  : {bits_huff:5d} bits  ({bits_huff / n:.3f} bits/letter)")
print(f"  arithmetic (unigram): {bits_arith:7.1f} bits  ({bits_arith / n:.3f} bits/letter)  = total log-loss of the model + 2")
print(f"  model entropy H    : {H_letters:.3f} bits/letter (expected cost if the text really came from the unigram model)")
""")

nb.md(r"""
Huffman gives "e" and "t" three-bit codes and "q" and "z" ten-bit codes; the integer lengths hug the ideal
$-\log_2 p$ from above, and the expected length lands within a few hundredths of a bit of the entropy. The
arithmetic coder removes even that rounding: its length *is* the model's log-loss on the text, plus two bits.
That identity is the reason a language model's cross-entropy loss, in bits, is literally a compression rate.

**Try it yourself:** replace the unigram model by the uniform distribution (`p_model = {c: 1/26 ...}`) and
recompute `bits_arith` — the cost should jump to exactly $\log_2 26 = 4.70$ bits per letter.
""")

# ---------------------------------------------------------------- 4. CE and KL
nb.md(r"""
## 4. Cross-entropy and KL divergence

$$H(p, q) = -\sum_x p(x)\log q(x), \qquad D_{\mathrm{KL}}(p \,\|\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p) \ge 0 .$$

We implement both with the right conventions ($0\log 0 = 0$; $p > 0, q = 0 \Rightarrow \infty$), check
Gibbs' inequality and the asymmetry on the guide's example, and verify against `scipy.stats.entropy(p, q)`.
""")

nb.code(r"""
def kl_bits(p, q):
    p, q = np.asarray(p, dtype=float).ravel(), np.asarray(q, dtype=float).ravel()
    mask = p > 0                                   # 0 log 0 = 0: terms with p = 0 contribute nothing
    if np.any(q[mask] == 0):
        return np.inf                              # the model calls a possible event impossible
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

def cross_entropy_bits(p, q):
    return entropy_bits(p) + kl_bits(p, q)

p, q = np.array([0.9, 0.1]), np.array([0.5, 0.5])
print(f"KL(p||q) = {kl_bits(p, q):.4f} bits    KL(q||p) = {kl_bits(q, p):.4f} bits   -> asymmetric")
print(f"H(p) = {entropy_bits(p):.4f}   H(p,q) = H(p) + KL(p||q) = {cross_entropy_bits(p, q):.4f} bits  (a fair-coin code spends exactly 1 bit/symbol)")
print(f"H(q,p) = {cross_entropy_bits(q, p):.4f} bits")
print(f"Gibbs: KL(p||p) = {kl_bits(p, p):.4f};  KL for a uniform model = log2 n - H(p) = {np.log2(2) - entropy_bits(p):.4f}")
print(f"impossible event: KL([0.9, 0.1] || [1.0, 0.0]) = {kl_bits([0.9, 0.1], [1.0, 0.0])}")
print(f"\nscipy.stats.entropy(p, q, base=2) = {sp_entropy(p, q, base=2):.4f}   (KL(p||q); first argument is the 'truth')")

# the loss of a classifier is the cross-entropy between the one-hot label and its prediction
y_onehot = np.array([0, 1, 0]); q_pred = np.array([0.7, 0.2, 0.1])
print(f"\nclassifier predicts {q_pred}, label is class 2: loss = -log q[1] = {-np.log(q_pred[1]):.4f} nats = {cross_entropy_bits(y_onehot, q_pred):.4f} bits")
""")

nb.md(r"""
### Forward vs reverse KL: mode-covering vs mode-seeking

Fit a single Gaussian $q = \mathcal{N}(\mu, \sigma^2)$ to a two-humped $p = \tfrac12\mathcal{N}(-3, 1) +
\tfrac12\mathcal{N}(3, 1)$ by minimising either $D_{\mathrm{KL}}(p \,\|\, q)$ (what maximum likelihood does)
or $D_{\mathrm{KL}}(q \,\|\, p)$ (what variational inference does). Both integrals are evaluated on a grid and
minimised numerically from the same starting point.
""")

nb.code(r"""
from scipy.optimize import minimize

x = np.linspace(-12, 12, 4001)
p_mix = 0.5 * norm.pdf(x, -3, 1) + 0.5 * norm.pdf(x, 3, 1)

def kl_grid(a, b):
    # KL(a || b) for two densities tabulated on the grid x, by the trapezoid rule
    mask = a > 1e-300
    a_safe = np.where(mask, a, 1.0); b_safe = np.maximum(b, 1e-300)
    return float(np.trapezoid(np.where(mask, a * np.log(a_safe / b_safe), 0.0), x))

def q_pdf(theta):
    mu, log_sigma = theta
    return norm.pdf(x, mu, np.exp(log_sigma))

theta0 = np.array([1.0, 0.0])                 # start at mu = 1, sigma = 1 for both fits
fwd = minimize(lambda th: kl_grid(p_mix, q_pdf(th)), theta0, method="Nelder-Mead")
rev = minimize(lambda th: kl_grid(q_pdf(th), p_mix), theta0, method="Nelder-Mead")
mu_f, sig_f = fwd.x[0], np.exp(fwd.x[1]); mu_r, sig_r = rev.x[0], np.exp(rev.x[1])
print(f"forward KL(p||q) minimiser: mu = {mu_f:6.3f}, sigma = {sig_f:.3f}   KL = {fwd.fun:.4f} nats   (moment matching: mu 0, sigma sqrt(10) = {np.sqrt(10):.3f})")
print(f"reverse KL(q||p) minimiser: mu = {mu_r:6.3f}, sigma = {sig_r:.3f}   KL = {rev.fun:.4f} nats   (one hump)")
print(f"cross-check: forward KL of the reverse fit = {kl_grid(p_mix, q_pdf(rev.x)):.3f};  reverse KL of the forward fit = {kl_grid(q_pdf(fwd.x), p_mix):.3f}")
broad = minimize(lambda th: kl_grid(q_pdf(th), p_mix), np.array([0.0, np.log(3.0)]), method="Nelder-Mead")
print(f"reverse KL also has a broad *local* optimum at mu = {broad.x[0]:.3f}, sigma = {np.exp(broad.x[1]):.3f} with KL = {broad.fun:.4f} > {rev.fun:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, th, name in zip(axes, (fwd.x, rev.x), ("forward KL(p||q): mode-covering", "reverse KL(q||p): mode-seeking")):
    ax.plot(x, p_mix, "k", label="p: mixture of two Gaussians")
    ax.fill_between(x, q_pdf(th), alpha=0.4, color="C1", label="fitted Gaussian q")
    ax.set_title(name); ax.set_xlabel("x"); ax.set_xlim(-9, 9); ax.legend(fontsize=8)
axes[0].set_ylabel("density"); plt.tight_layout(); plt.show()
""")

nb.md(r"""
Forward KL forces $q$ to put mass wherever $p$ has it (otherwise $\log(p/q)$ explodes), so the fit is one
broad Gaussian over both humps with most of its mass in the empty middle. Reverse KL punishes $q$ only for
mass where $p$ is small, so it hugs a single hump and ignores half the data; the other hump is an equally
good optimum, and a wide Gaussian in the middle is a *worse* local optimum that a poorly initialised optimiser
can get stuck in. This is the quantitative content of "VI approximations are too narrow".

**Try it yourself:** change `theta0` to `[-1.0, 0.0]` — the reverse fit should jump to the *left* hump, while
the forward fit lands on the same broad Gaussian. Then try `[0.0, np.log(3.0)]`.

### KL between Gaussians: closed form vs Monte Carlo

$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1, \sigma_1^2)\,\|\,\mathcal{N}(\mu_2, \sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12 .$$
""")

nb.code(r"""
def kl_gauss(m1, s1, m2, s2):
    return np.log(s2 / s1) + (s1**2 + (m1 - m2)**2) / (2 * s2**2) - 0.5

n_mc = 200_000
for (m1, s1, m2, s2) in [(0, 1, 1, 2), (1, 2, 0, 1), (0, 1, 0, 1), (0, 1, 1, 1)]:
    xs = rng.normal(m1, s1, n_mc)
    log_ratio = norm.logpdf(xs, m1, s1) - norm.logpdf(xs, m2, s2)
    mc, se = log_ratio.mean(), log_ratio.std() / np.sqrt(n_mc)
    print(f"KL(N({m1},{s1}^2) || N({m2},{s2}^2)): closed form {kl_gauss(m1, s1, m2, s2):.4f} nats   Monte Carlo {mc:.4f} +/- {se:.4f}")
print(f"\nVAE regulariser KL(N(mu, sigma^2) || N(0,1)) for mu = 0.5, sigma = 0.8: {kl_gauss(0.5, 0.8, 0, 1):.4f} nats"
      f"  = 0.5 (mu^2 + sigma^2 - 1 - log sigma^2) = {0.5 * (0.25 + 0.64 - 1 - np.log(0.64)):.4f}")
""")

nb.md(r"""
The narrow distribution is cheap to describe with the wide one ($0.443$ nats) but not the other way round
($1.307$ nats), and the Monte Carlo estimates agree with the formula to within their standard errors. The
last line is the per-dimension KL term of every variational autoencoder.
""")

# ---------------------------------------------------------------- 5. mutual information
nb.md(r"""
## 5. Mutual information

$I(X; Y) = H(X) + H(Y) - H(X, Y) = D_{\mathrm{KL}}\big(p(x,y)\,\|\,p(x)p(y)\big)$. First from the joint
table of the diagnostic test (prevalence 1%, sensitivity 0.99, specificity 0.95), then *estimated* from
samples of a relation with zero correlation but strong dependence — including the bias of the plug-in
estimate and a shuffle null.
""")

nb.code(r"""
def mutual_information_bits(joint):
    joint = np.asarray(joint, dtype=float)
    return entropy_bits(joint.sum(axis=1)) + entropy_bits(joint.sum(axis=0)) - entropy_bits(joint)

prev, se, sp = 0.01, 0.99, 0.95
J = np.array([[se * prev,       (1 - se) * prev],        # rows: diseased, healthy
              [(1 - sp) * (1 - prev), sp * (1 - prev)]])  # cols: test +, test -
H_D = entropy_bits(J.sum(axis=1))
print(f"joint table (D/H x +/-):\n{J}")
print(f"H(D) = {H_D:.4f}   H(T) = {entropy_bits(J.sum(axis=0)):.4f}   H(D,T) = {entropy_bits(J):.4f} bits")
I_DT = mutual_information_bits(J)
print(f"I(D;T) = {I_DT:.4f} bits = {100 * I_DT / H_D:.1f}% of the uncertainty about disease status")
print(f"umbrella table: I(X;Y) = {mutual_information_bits(joint):.4f} bits  (matches Section 2)")

def test_info(se, sp, prev=0.01):
    J = np.array([[se * prev, (1 - se) * prev], [(1 - sp) * (1 - prev), sp * (1 - prev)]])
    return mutual_information_bits(J)
print("\ninformation carried by different tests at 1% prevalence:")
for se_, sp_ in [(0.99, 0.95), (0.80, 0.99), (0.99, 0.99), (0.90, 0.90)]:
    print(f"  sensitivity {se_:.2f}, specificity {sp_:.2f}: I = {test_info(se_, sp_):.4f} bits  ({100 * test_info(se_, sp_) / H_D:.0f}% of H(D))")
""")

nb.code(r"""
# estimating MI from samples: Y = X^2 + noise has zero correlation but strong dependence
n = 5_000
xs = rng.uniform(-2, 2, n)
ys = xs**2 + 0.3 * rng.normal(size=n)

def mi_binned_bits(x, y, bins=10):
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    return mutual_information_bits(counts / counts.sum())

bins = 10
mi_obs = mi_binned_bits(xs, ys, bins)
null = np.array([mi_binned_bits(xs, rng.permutation(ys), bins) for _ in range(200)])
mm_bias = (bins - 1) * (bins - 1) / (2 * n * LN2)          # Miller-Madow first-order bias of the plug-in estimate
print(f"correlation(X, Y) = {np.corrcoef(xs, ys)[0, 1]:+.4f}   (essentially zero)")
print(f"plug-in MI with {bins}x{bins} bins = {mi_obs:.4f} bits")
print(f"shuffle null (Y permuted, 200 times): mean {null.mean():.4f}, max {null.max():.4f} bits   <- what independence looks like with this estimator")
print(f"Miller-Madow bias estimate (B-1)^2 / (2 N ln 2) = {mm_bias:.4f} bits   (close to the null mean)")

from sklearn.feature_selection import mutual_info_regression
mi_knn = mutual_info_regression(xs.reshape(-1, 1), ys, random_state=0)[0] / LN2
print(f"sklearn k-NN estimator: {mi_knn:.4f} bits")

# Gaussian pair with rho = 0.9: exact MI = -0.5 log2(1 - rho^2)
rho = 0.9
g = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
print(f"\nGaussian rho = {rho}: exact I = {-0.5 * np.log2(1 - rho**2):.4f} bits;  binned ({bins} bins) {mi_binned_bits(g[:, 0], g[:, 1], bins):.4f};"
      f"  k-NN {mutual_info_regression(g[:, :1], g[:, 1], random_state=0)[0] / LN2:.4f} bits")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.scatter(xs[:1500], ys[:1500], s=4, alpha=0.5)
ax1.set_xlabel("x"); ax1.set_ylabel("y = x^2 + noise"); ax1.set_title(f"corr = {np.corrcoef(xs, ys)[0, 1]:+.3f}, but I = {mi_obs:.2f} bits")
ax2.hist(null, bins=30, alpha=0.7, label="shuffle null (Y permuted)")
ax2.axvline(mm_bias, color="k", ls="--", label=f"Miller-Madow bias {mm_bias:.4f}")
ax2.set_xlabel("plug-in mutual information under independence (bits)"); ax2.set_ylabel("count")
ax2.set_title(f"Independence null  (observed value {mi_obs:.3f} bits is far off to the right)")
ax2.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

nb.md(r"""
Correlation sees nothing ($+0.02$); mutual information sees more than a bit of dependence. But notice the
null: with $10 \times 10$ bins and 5,000 points, *independent* data still produce about $0.012$ bits of
spurious information, in line with the Miller–Madow formula, and with fewer samples or more bins that bias
grows quickly — never read a small positive estimate as evidence of dependence without a null. For the
correlated Gaussians the coarse binning *under*-estimates the exact $1.198$ bits, while the
$k$-nearest-neighbour estimator comes closer: estimator choice matters.

**Try it yourself:** set `bins = 30` and re-run — the observed value and the null both rise. Which rises more?
""")

# ---------------------------------------------------------------- 6. Jensen / ELBO
nb.md(r"""
## 6. Jensen's inequality and the ELBO

For concave $\log$, $\log\mathbb{E}[X] \ge \mathbb{E}[\log X]$: the chord lies below the curve. Applied to
$\log p(x) = \log\mathbb{E}_q[p(x, Z)/q(Z)]$ this gives the evidence lower bound, whose gap is
$D_{\mathrm{KL}}(q \,\|\, p(z \mid x))$. We draw the chord and then compute the ELBO of the guide's two-state
latent model for every $q$.
""")

nb.code(r"""
# Jensen: X is 1 or 4 with probability 1/2 each
xj = np.array([1.0, 4.0]); EX = xj.mean()
print(f"log2 E[X] = log2 {EX} = {np.log2(EX):.4f}   >=   E[log2 X] = {np.log2(xj).mean():.4f}   (gap {np.log2(EX) - np.log2(xj).mean():.4f} bits)")

# ELBO for a two-state latent variable: prior p(z) = (0.5, 0.5), likelihood p(x|z) = (0.8, 0.2)
prior = np.array([0.5, 0.5]); lik = np.array([0.8, 0.2])
evidence = float(prior @ lik); posterior = prior * lik / evidence
print(f"p(x) = {evidence:.3f}, log p(x) = {np.log(evidence):.4f} nats;  true posterior p(z|x) = {posterior}")

def kl_nats(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = a > 0
    return float(np.sum(a[m] * np.log(a[m] / b[m])))

def elbo(q):
    q = np.asarray(q, float)
    return float(np.sum(q * np.log(lik)) - kl_nats(q, prior))     # E_q[log p(x|z)] - KL(q || prior)

print(f"\n{'q(z=1)':>7s} {'E_q log p(x|z)':>15s} {'KL(q||prior)':>13s} {'ELBO':>8s} {'gap':>8s} {'KL(q||post)':>12s}")
for q1 in (0.2, 0.5, 0.8, 0.95):
    q = np.array([q1, 1 - q1])
    print(f"{q1:7.2f} {np.sum(q * np.log(lik)):15.4f} {kl_nats(q, prior):13.4f} {elbo(q):8.4f} {np.log(evidence) - elbo(q):8.4f} {kl_nats(q, posterior):12.4f}")

q_grid = np.linspace(0.001, 0.999, 500)
elbos = np.array([elbo([q1, 1 - q1]) for q1 in q_grid])
gaps = np.array([kl_nats([q1, 1 - q1], posterior) for q1 in q_grid])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
u = np.linspace(0.5, 5, 200)
ax1.plot(u, np.log2(u), label="log2 x (concave)")
ax1.plot(xj, np.log2(xj), "o-", color="C3", label="chord between x = 1 and x = 4")
ax1.plot(EX, np.log2(EX), "s", color="C2", label=f"log2 E[X] = {np.log2(EX):.3f}")
ax1.plot(EX, np.log2(xj).mean(), "v", color="C3", label=f"E[log2 X] = {np.log2(xj).mean():.3f}")
ax1.set_xlabel("x"); ax1.set_ylabel("log2 x"); ax1.set_title("Jensen: the chord lies below a concave curve"); ax1.legend(fontsize=8)
ax2.plot(q_grid, elbos, label="ELBO(q)")
ax2.axhline(np.log(evidence), color="k", ls="--", label=f"log p(x) = {np.log(evidence):.3f}")
ax2.plot(q_grid, np.log(evidence) - gaps, ":", color="C3", lw=2, label="log p(x) - KL(q || posterior)")
ax2.axvline(posterior[0], color="gray", lw=1)
ax2.set_xlabel("q(z = 1)"); ax2.set_ylabel("nats"); ax2.set_title("The ELBO touches log p(x) exactly at the posterior")
ax2.set_ylim(-2.5, 0); ax2.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

nb.md(r"""
The dotted curve $\log p(x) - D_{\mathrm{KL}}(q \,\|\, p(z \mid x))$ lies exactly on top of the ELBO: the
identity is exact, not approximate, and the bound is tight only at $q = p(z \mid x) = (0.8, 0.2)$. Moving
$q$ towards $0.95$ improves the reconstruction term but pays more KL to the prior; the posterior is where
the two balance. A VAE does this optimisation with a neural network producing $q$ and gradient steps instead
of a grid.
""")

# ---------------------------------------------------------------- 7. maximum entropy
nb.md(r"""
## 7. Maximum entropy: Jaynes' loaded die

A die averages $4.5$ instead of $3.5$ and nothing else is known. The maximum-entropy distribution with a
fixed mean has the exponential-family form $p(k) \propto e^{\lambda k}$; we solve for $\lambda$ with a root
finder and compare the entropy with other distributions that also have mean $4.5$.
""")

nb.code(r"""
from scipy.optimize import brentq

faces = np.arange(1, 7)

def maxent_die(target_mean):
    mean_at = lambda lam: float(np.sum(faces * np.exp(lam * faces)) / np.sum(np.exp(lam * faces)))
    lam = brentq(lambda lam: mean_at(lam) - target_mean, -20, 20)
    w = np.exp(lam * faces)
    return lam, w / w.sum()

for m in (3.5, 4.5, 2.5):
    lam, p_me = maxent_die(m)
    print(f"mean {m}: lambda = {lam:+.4f}   p = {np.round(p_me, 4)}   mean check {faces @ p_me:.4f}   H = {entropy_bits(p_me):.4f} bits")

lam45, p45 = maxent_die(4.5)
alternatives = {"max-ent  p ~ exp(lambda k)": p45,
                "(0, 0, 0, .5, .5, 0)": np.array([0, 0, 0, .5, .5, 0]),
                "(.1, .1, .1, .1, .1, .5)": np.array([.1, .1, .1, .1, .1, .5])}
print("\ndistributions with mean 4.5:")
for name, pa in alternatives.items():
    print(f"  {name:28s} mean = {faces @ pa:.2f}   H = {entropy_bits(pa):.4f} bits")
print(f"ratio p(k+1)/p(k) = e^lambda = {np.exp(lam45):.4f} for every k;  entropy lost to the constraint: {np.log2(6) - entropy_bits(p45):.4f} bits")

w = 0.27
for i, (name, pa) in enumerate(alternatives.items()):
    plt.bar(faces + (i - 1) * w, pa, width=w, label=name)
plt.xlabel("face k"); plt.ylabel("p(k)"); plt.title("Three PMFs with mean 4.5 — the max-ent one assumes least")
plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
The maximum-entropy solution spreads probability over all six faces in a geometric progression; the
alternatives satisfy the same constraint but assert extra facts (some faces never occur, or face 6 is
special) that nobody supplied. The same Lagrangian with a constraint on an expected *score* gives the softmax
— Section 9 explores that family through its temperature.
""")

# ---------------------------------------------------------------- 8. decision trees
nb.md(r"""
## 8. Information gain in decision trees

$\mathrm{IG}(Y; A) = H(Y) - \sum_a \frac{N_a}{N}H(Y \mid A = a)$ — the mutual information between the label
and the split, on the examples at a node. We compute it for the guide's triage table, compare with the Gini
decrease, then scan every threshold of every iris feature and compare with the root split chosen by
`DecisionTreeClassifier`.
""")

nb.code(r"""
def entropy_of_labels(y):
    _, counts = np.unique(y, return_counts=True)
    return entropy_bits(counts / counts.sum())

def gini_of_labels(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1 - np.sum(p**2))

def information_gain(feature, y, impurity=entropy_of_labels):
    feature = np.asarray(feature); y = np.asarray(y)
    cond = sum((feature == v).mean() * impurity(y[feature == v]) for v in np.unique(feature))
    return impurity(y) - cond

# ten triage patients: label = admitted
admitted = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])
fever    = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])   # 6 with fever, 4 of them admitted; 1 of 4 without
spo2_low = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])   # 4 with low saturation, all admitted; 1 of 6 others
patient_id = np.arange(10)

print(f"H(admitted) = {entropy_of_labels(admitted):.4f} bits")
for name, f in [("fever", fever), ("SpO2 < 94%", spo2_low), ("patient ID", patient_id)]:
    ig = information_gain(f, admitted); gd = information_gain(f, admitted, gini_of_labels)
    gain_ratio = ig / entropy_of_labels(f)
    print(f"  {name:12s} information gain = {ig:.4f} bits   Gini decrease = {gd:.4f}   gain ratio = {gain_ratio:.4f}")
""")

nb.code(r"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target
print(f"iris: H(species) = {entropy_of_labels(y):.4f} bits (three balanced classes = log2 3)")

best = []
plt.figure(figsize=(8, 4))
for j, fname in enumerate(iris.feature_names):
    vals = np.unique(X[:, j]); thresholds = (vals[:-1] + vals[1:]) / 2
    gains = np.array([information_gain(X[:, j] <= t, y) for t in thresholds])
    k = int(np.argmax(gains)); best.append((gains[k], j, thresholds[k]))
    plt.plot(thresholds, gains, label=fname)
plt.xlabel("threshold t  (split: feature <= t)"); plt.ylabel("information gain (bits)")
plt.title("Information gain of every candidate root split on iris"); plt.legend(); plt.show()

for g, j, t in sorted(best, reverse=True):
    print(f"  best split on {iris.feature_names[j]:18s}: <= {t:.2f}   gain {g:.4f} bits")

clf = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=0).fit(X, y)
j, t = clf.tree_.feature[0], clf.tree_.threshold[0]
print(f"\nsklearn root split: {iris.feature_names[j]} <= {t:.2f}   (criterion='entropy')")
print(f"our gain for that split: {information_gain(X[:, j] <= t, y):.4f} bits;  entropy at the root: {clf.tree_.impurity[0]:.4f}, children: {np.round(clf.tree_.impurity[1:], 4)}")
""")

nb.md(r"""
Low oxygen saturation beats fever by a factor of five, and both criteria agree. The *patient ID* feature
has the largest gain of all — one bit, the whole entropy — because it memorises each patient; its gain ratio
exposes the trick. On iris, two splits tie at $0.918$ bits (petal width $\le 0.8$ and petal length $\le 2.45$):
each isolates all 50 setosa flowers perfectly, leaving a 50/50 node with one bit of entropy, so the gain is
$\log_2 3 - \tfrac23 \cdot 1 = 0.918$. scikit-learn picks one of the two tied splits; which one depends on the
order in which it evaluates features.

**Try it yourself:** pass `gini_of_labels` as the impurity in the iris scan — does the ranking of the four
features change?
""")

# ---------------------------------------------------------------- 9. language models
nb.md(r"""
## 9. Language models: cross-entropy, perplexity and temperature

A model's per-token loss $\mathcal{L} = -\frac1T\sum_t \log p(x_t \mid x_{<t})$ (nats) has perplexity
$e^{\mathcal{L}}$, the "effective number of equally likely choices". Temperature rescales logits,
$p_T(k) \propto e^{z_k / T}$, moving along the maximum-entropy family of Section 7: $T \to 0$ is the argmax,
$T \to \infty$ is uniform. We sweep $T$ on a four-token vocabulary and demonstrate top-$k$ and top-$p$.
""")

nb.code(r"""
def softmax_T(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    e = np.exp(z - z.max())
    return e / e.sum()

def perplexity_from_probs(probs):
    probs = np.asarray(probs, dtype=float)
    return float(np.exp(-np.mean(np.log(probs))))

z = np.array([2.0, 1.0, 0.0, -1.0])
p1 = softmax_T(z)
print(f"logits {z} -> softmax {np.round(p1, 4)};  H = {entropy_bits(p1):.4f} bits = {entropy_bits(p1) * LN2:.4f} nats;  perplexity 2^H = {2**entropy_bits(p1):.4f}")
print(f"if the true next token is #2 (p = {p1[1]:.4f}): loss = {-np.log(p1[1]):.4f} nats = {-np.log2(p1[1]):.4f} bits")

print(f"\n{'T':>6s} {'p(T)':>36s} {'H bits':>8s} {'PPL':>7s}")
for T in (0.25, 0.5, 1, 2, 5, 100):
    pT = softmax_T(z, T)
    print(f"{T:6.2f} {str(np.round(pT, 4)):>36s} {entropy_bits(pT):8.4f} {2**entropy_bits(pT):7.3f}")
print(f"limits: T -> 0 gives H = 0 (greedy), T -> inf gives H = log2 V = {np.log2(4):.1f} bits, PPL = V = 4")

Ts = np.logspace(-1, 1.5, 200)
Hs = np.array([entropy_bits(softmax_T(z, T)) for T in Ts])
fig, ax1 = plt.subplots()
ax1.semilogx(Ts, Hs, label="entropy (bits)"); ax1.set_xlabel("temperature T (log scale)"); ax1.set_ylabel("entropy of softmax(z / T)  (bits)")
ax1.axhline(np.log2(4), color="k", ls="--", lw=1, label="log2 V = 2 bits (uniform)")
ax2 = ax1.twinx(); ax2.semilogx(Ts, 2**Hs, color="C3", label="perplexity 2^H"); ax2.set_ylabel("perplexity")
ax1.set_title("Temperature controls the entropy of the sampling distribution")
ax1.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8); plt.show()
""")

nb.code(r"""
# top-k and top-p (nucleus) truncation at T = 1
def top_k(p, k):
    keep = np.argsort(p)[::-1][:k]; out = np.zeros_like(p); out[keep] = p[keep]; return out / out.sum()

def top_p(p, p_mass):
    order = np.argsort(p)[::-1]; cum = np.cumsum(p[order])
    n_keep = int(np.searchsorted(cum, p_mass) + 1)
    out = np.zeros_like(p); out[order[:n_keep]] = p[order[:n_keep]]; return out / out.sum()

print(f"sorted cumulative probabilities: {np.round(np.cumsum(np.sort(p1)[::-1]), 4)}")
for name, pp in [("full softmax", p1), ("top-k, k = 2", top_k(p1, 2)), ("top-p, p = 0.9", top_p(p1, 0.9)), ("T = 0.5", softmax_T(z, 0.5))]:
    print(f"  {name:16s} {str(np.round(pp, 4)):>36s}   H = {entropy_bits(pp):.4f} bits")

# the perplexity of a sequence is the inverse geometric mean of the probabilities assigned to the actual tokens
seq_probs = np.array([0.5, 0.1, 0.25, 0.05, 0.8])
L_nats = -np.mean(np.log(seq_probs))
print(f"\nsequence probabilities {seq_probs}: mean loss {L_nats:.4f} nats = {L_nats / LN2:.4f} bits/token;  perplexity {perplexity_from_probs(seq_probs):.4f}"
      f"  = 1 / geometric mean {np.exp(np.mean(np.log(seq_probs))):.4f}")
print("\nloss (nats/token) -> bits/token -> perplexity:")
for L in (1.0, 1.5, 2.0, 2.5, 3.0):
    print(f"  {L:.1f} -> {L / LN2:.3f} -> {np.exp(L):6.2f}")
print(f"uniform over a 50,000-token vocabulary: loss = ln 50000 = {np.log(50_000):.2f} nats, PPL = 50,000")
print(f"a loss of 2.0 nats/token at 4 bytes/token = {2.0 / LN2 / 4:.3f} bits per byte  ({100 * 2.0 / LN2 / 4 / 8:.0f}% of the raw 8 bits)")

from scipy.special import softmax as sp_softmax
print(f"\nscipy.special.softmax(z / 2) agrees with softmax_T(z, 2): {np.allclose(sp_softmax(z / 2), softmax_T(z, 2))}")
""")

nb.md(r"""
Lowering $T$ sharpens *every* token's probability uniformly; top-$k$ and top-$p$ instead cut the tail and
leave the head's proportions intact — a different way to lower entropy, aimed at the region where a model's
small probabilities are least trustworthy. The perplexity of a sequence is the inverse geometric mean of the
probabilities the model gave to what actually happened; a single near-zero probability dominates it, which
is why one confidently wrong prediction can ruin an evaluation.

**Try it yourself:** append a fifth logit of $-6$ to `z` — the entropy at $T = 1$ barely moves, but the
$T \to \infty$ limit rises to $\log_2 5$. What does that say about rare tokens and high temperatures?
""")

# ---------------------------------------------------------------- 10. channel capacity
nb.md(r"""
## 10. Channel capacity: the binary symmetric channel

A channel flips each bit with probability $p$. Its capacity is $C = 1 - H_b(p)$ bits per use: below that rate
reliable communication is possible with long codes, above it impossible. We plot the capacity and simulate
repetition codes, which buy reliability only by throwing away rate.
""")

nb.code(r"""
from math import comb

def bsc_capacity(p):
    return 1 - binary_entropy(p)

def repetition_error_exact(p, r):
    # majority vote over r copies fails when more than half flip
    return sum(comb(r, k) * p**k * (1 - p)**(r - k) for k in range(r // 2 + 1, r + 1))

def repetition_error_simulated(p, r, n_bits, rng):
    flips = rng.random((n_bits, r)) < p
    return float((flips.sum(axis=1) > r // 2).mean())

p_flip = 0.1
print(f"BSC with p = {p_flip}: capacity C = 1 - H_b({p_flip}) = {bsc_capacity(p_flip):.4f} bits per channel use")
print(f"{'code':>14s} {'rate':>6s} {'bit error (exact)':>18s} {'simulated':>10s}")
print(f"{'raw bits':>14s} {1.0:6.3f} {p_flip:18.5f} {repetition_error_simulated(p_flip, 1, 200_000, rng):10.5f}")
for r in (3, 5, 7, 9):
    print(f"{'repeat x' + str(r):>14s} {1/r:6.3f} {repetition_error_exact(p_flip, r):18.5f} {repetition_error_simulated(p_flip, r, 200_000, rng):10.5f}")

p_ham = 0.01
block_raw = 1 - (1 - p_ham)**4
block_hamming = 1 - ((1 - p_ham)**7 + 7 * p_ham * (1 - p_ham)**6)
print(f"\nat p = {p_ham}: capacity {bsc_capacity(p_ham):.4f};  Hamming(7,4) rate 4/7 = {4/7:.3f}, block error {block_hamming:.4f} vs {block_raw:.4f} for 4 raw bits")

ps = np.linspace(0, 1, 401)
plt.plot(ps, [bsc_capacity(p) for p in ps])
plt.xlabel("flip probability p"); plt.ylabel("capacity  1 - H_b(p)  (bits per use)")
plt.title("Capacity of the binary symmetric channel"); plt.show()
""")

nb.md(r"""
The capacity curve is the mirror image of the binary entropy: a channel that flips half the bits carries
nothing, and a channel that flips *all* of them is perfect (just invert the output). Repetition codes
reduce the error from $0.1$ to $0.0086$ at the price of one fifth of the rate, while capacity says rates up
to $0.53$ are achievable with vanishing error. The codes that approach it — Hamming, then LDPC and polar codes
— spend their redundancy on *structure* rather than copies.
""")

nb.md(r"""
## Summary

- Information is surprise, $-\log p$; the logarithm is forced by additivity over independent events; bits and nats differ by the factor $\ln 2$.
- Entropy is expected surprise: maximal for uniform distributions, zero for certainty; $H(X, Y) = H(X) + H(Y \mid X)$; differential entropy is only meaningful in differences.
- Huffman codes come within one bit of the entropy, arithmetic coding within two bits for a whole sequence — a model's log-loss *is* a compression rate.
- Cross-entropy $= H(p) + D_{\mathrm{KL}}(p \,\|\, q)$; KL is non-negative, asymmetric and infinite for impossible events; forward KL covers modes, reverse KL seeks one.
- Mutual information detects any dependence, is estimated with an upward bias from binned data, and should be compared with a shuffle null.
- Jensen's inequality gives the ELBO, whose gap is exactly the KL divergence to the true posterior.
- Maximum entropy under moment constraints yields exponential families; the softmax is the max-ent distribution with a fixed expected score, and temperature moves along that family.
- Information gain is mutual information between label and split; the gain ratio guards against many-valued features.
- Perplexity $= e^{\text{loss}}$ is an effective branching factor; temperature, top-$k$ and top-$p$ reshape the entropy of the sampling distribution.
- Channel capacity $1 - H_b(p)$ bounds reliable communication; codes approach it with structure, not repetition.

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
import heapq, itertools
from scipy.stats import norm
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)
LN2 = np.log(2)
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
    "Module 14 — Exercises",
    intro_md=r"""
These exercises follow the order of the concept guide: entropy, cross-entropy and KL, joint and conditional
entropy, Huffman coding, KL between Gaussians, information gain, perplexity and temperature, two conceptual
questions, the ELBO, and finally a mini-project that builds a character-level language model and measures it
in bits. Unless an exercise says otherwise, work in **bits** (base-2 logarithms). Several checks compare your
answer against a *hidden* value, so compute carefully rather than guessing.
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- 1. entropy (warm-up)
SOL_1 = r"""
def entropy(p, base=2):
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]                       # 0 log 0 = 0
    return float(-np.sum(p * np.log(p)) / np.log(base))
"""
ns1 = run(SOL_1, r"""
h_a = entropy([0.1, 0.2, 0.3, 0.4])
h_a_nats = entropy([0.1, 0.2, 0.3, 0.4], base=np.e)
h_zero = entropy([0.5, 0.2, 0.2, 0.1, 0.0])
""")
ex.add(
    title="Entropy, with zeros handled correctly", difficulty=1,
    prompt_md=r"""
Implement `entropy(p, base=2)` for a discrete distribution given as a list or NumPy array of probabilities,
returning a **float** in the units of `base` (bits for 2, nats for `np.e`). Use only NumPy. Entries equal to
0 must contribute 0 (the convention $0\log 0 = 0$) — your function must not return `nan` or raise a warning
for them. Do not use `scipy.stats.entropy` inside your implementation (the check uses it as a reference).
""",
    starter=r"""
def entropy(p, base=2):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_1,
    check=rf"""
from scipy.stats import entropy as _sp_entropy
try:
    got = entropy([0.5, 0.5])
except Exception:
    got = None
check_close('fair coin has 1 bit', got, 1.0)
try:
    got = entropy(np.full(6, 1/6))
except Exception:
    got = None
check_close('fair die has log2 6 bits', got, np.log2(6))
try:
    got = entropy([0.9, 0.1], base=np.e)
except Exception:
    got = None
check_close('nats agree with scipy.stats.entropy', got, _sp_entropy([0.9, 0.1]))
try:
    got = entropy([0.1, 0.2, 0.3, 0.4])
except Exception:
    got = None
check_hash('hidden: entropy of (0.1, 0.2, 0.3, 0.4) in bits', got, '{sha_of(ns1["h_a"])}')
try:
    got = entropy([0.5, 0.2, 0.2, 0.1, 0.0])
except Exception:
    got = None
check_hash('hidden: a distribution with a zero entry', got, '{sha_of(ns1["h_zero"])}')
try:
    got = entropy([1.0, 0.0, 0.0])
except Exception:
    got = None
check_close('a certain outcome has zero entropy', got, 0.0)
""",
    explanation_md=r"""
Dropping the zero entries before taking logarithms implements $0\log 0 = 0$ without producing `nan`
(`0 * np.log(0)` is `nan` in NumPy, and `np.log(0)` emits a warning). Dividing by $\log(\text{base})$
converts from nats to any unit, since $\log_b u = \ln u / \ln b$. The common wrong turn is to use
`np.where(p > 0, p * np.log(p), 0)`, which still evaluates `np.log(0)` for the masked entries and warns;
masking first is cleaner. A certain outcome gives exactly 0 and a uniform distribution over $n$ outcomes gives
$\log_2 n$, the two extremes of the entropy.
""",
)

# ---------------------------------------------------------------- 2. CE and KL (warm-up)
SOL_2 = r"""
def kl_divergence(p, q):
    p, q = np.asarray(p, dtype=float).ravel(), np.asarray(q, dtype=float).ravel()
    mask = p > 0
    if np.any(q[mask] == 0):
        return float('inf')
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

def cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float).ravel(), np.asarray(q, dtype=float).ravel()
    mask = p > 0
    if np.any(q[mask] == 0):
        return float('inf')
    return float(-np.sum(p[mask] * np.log2(q[mask])))
"""
ns2 = run(SOL_2, r"""
p_ = np.array([0.7, 0.2, 0.1]); q_ = np.array([0.5, 0.3, 0.2])
kl_pq = kl_divergence(p_, q_); kl_qp = kl_divergence(q_, p_)
""")
ex.add(
    title="Cross-entropy and KL divergence with their edge cases", difficulty=1,
    prompt_md=r"""
Implement, in **bits**, using only NumPy:

* `kl_divergence(p, q)` $= \sum_x p(x)\log_2\frac{p(x)}{q(x)}$;
* `cross_entropy(p, q)` $= -\sum_x p(x)\log_2 q(x)$.

Both take arrays of the same length and return a **float**. Conventions: a term with $p(x) = 0$ contributes
0 whatever $q(x)$ is; if some $x$ has $p(x) > 0$ and $q(x) = 0$ the result is `float('inf')` (not `nan`,
no exception). The checks use $p = (0.7, 0.2, 0.1)$, $q = (0.5, 0.3, 0.2)$ and some edge cases.
""",
    starter=r"""
def kl_divergence(p, q):
    # YOUR CODE HERE
    return None

def cross_entropy(p, q):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_2,
    check=rf"""
from scipy.stats import entropy as _sp_entropy
p_ = np.array([0.7, 0.2, 0.1]); q_ = np.array([0.5, 0.3, 0.2])
try:
    got = kl_divergence(p_, p_)
except Exception:
    got = None
check_close('KL(p || p) = 0', got, 0.0)
try:
    got = kl_divergence([0.9, 0.1], [0.5, 0.5])
except Exception:
    got = None
check_close('KL((0.9, 0.1) || (0.5, 0.5)) matches scipy (bits)', got, _sp_entropy([0.9, 0.1], [0.5, 0.5], base=2))
try:
    got = kl_divergence(p_, q_)
except Exception:
    got = None
check_hash('hidden: KL(p || q)', got, '{sha_of(ns2["kl_pq"])}')
try:
    got = kl_divergence(q_, p_)
except Exception:
    got = None
check_hash('hidden: KL(q || p) (asymmetric!)', got, '{sha_of(ns2["kl_qp"])}')
try:
    got = cross_entropy(p_, q_)
    expected = _sp_entropy(p_, base=2) + _sp_entropy(p_, q_, base=2)
except Exception:
    got, expected = None, 1.0
check_close('H(p, q) = H(p) + KL(p || q)', got, expected)
try:
    got = cross_entropy([0.0, 1.0], [0.3, 0.7])
except Exception:
    got = None
check_close('one-hot label: cross-entropy is -log2 q[label]', got, -np.log2(0.7))
try:
    got = kl_divergence([0.0, 0.5, 0.5], [0.0, 0.5, 0.5])
except Exception:
    got = None
check_close('p = q = 0 on one outcome contributes nothing', got, 0.0)
try:
    v = kl_divergence([0.9, 0.1], [1.0, 0.0])
    got = bool(np.isinf(v)) and v > 0
except Exception:
    got = False
check('impossible event under q gives +inf', got)
""",
    explanation_md=r"""
Mask on $p > 0$ first: that implements $0\log 0 = 0$ and also avoids dividing by $q(x) = 0$ where it does not
matter. On the remaining support, any $q(x) = 0$ means the model assigns probability zero to something that
happens, and the divergence is infinite — a classifier that outputs an exact 0 for the true class has infinite
loss, which is why implementations work with `log_softmax` and clip probabilities. The identity
$H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)$ is checked numerically; with a one-hot $p$ the entropy term
vanishes and the cross-entropy collapses to $-\log_2 q(\text{label})$, the usual classification loss. The two
hidden values differ, as they must: KL is not symmetric.
""",
)

# ---------------------------------------------------------------- 3. joint table (warm-up)
SOL_3 = r"""
def H_bits(p):
    p = np.asarray(p, dtype=float).ravel(); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def joint_entropy(joint):
    return H_bits(np.asarray(joint, dtype=float))

def conditional_entropy(joint):
    # H(Y | X) with X indexing rows and Y indexing columns: H(X, Y) - H(X)
    joint = np.asarray(joint, dtype=float)
    return joint_entropy(joint) - H_bits(joint.sum(axis=1))

def mutual_information(joint):
    joint = np.asarray(joint, dtype=float)
    return H_bits(joint.sum(axis=1)) + H_bits(joint.sum(axis=0)) - joint_entropy(joint)
"""
ns3 = run(SOL_3, r"""
J3 = np.array([[0.20, 0.05, 0.00],
               [0.10, 0.20, 0.05],
               [0.00, 0.10, 0.30]])
hxy = joint_entropy(J3); hy_x = conditional_entropy(J3); hx_y = conditional_entropy(J3.T); mi = mutual_information(J3)
""")
ex.add(
    title="Joint, conditional and mutual information from a table", difficulty=1,
    prompt_md=r"""
A joint PMF is given as a 2-D NumPy array `joint` whose **rows index $X$ and columns index $Y$**. Implement,
in bits, returning floats:

* `joint_entropy(joint)` $= H(X, Y)$;
* `conditional_entropy(joint)` $= H(Y \mid X)$ (uncertainty about the *column* variable once the row is known);
* `mutual_information(joint)` $= I(X; Y) = H(X) + H(Y) - H(X, Y)$.

Handle zero cells ($0\log 0 = 0$). The checks use the table

```python
J3 = np.array([[0.20, 0.05, 0.00],
               [0.10, 0.20, 0.05],
               [0.00, 0.10, 0.30]])
```

and confirm the chain rule $H(X, Y) = H(X) + H(Y \mid X)$, symmetry of $I$, and $I = 0$ for a product table.
Hint: `conditional_entropy(joint.T)` is then $H(X \mid Y)$.
""",
    starter=r"""
def joint_entropy(joint):
    # YOUR CODE HERE
    return None

def conditional_entropy(joint):
    # H(Y | X), rows = X, columns = Y
    # YOUR CODE HERE
    return None

def mutual_information(joint):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_3,
    check=rf"""
J3 = np.array([[0.20, 0.05, 0.00],
               [0.10, 0.20, 0.05],
               [0.00, 0.10, 0.30]])
from scipy.stats import entropy as _sp_entropy
try:
    got = joint_entropy(J3)
except Exception:
    got = None
check_hash('hidden: H(X, Y)', got, '{sha_of(ns3["hxy"])}')
try:
    got = conditional_entropy(J3)
except Exception:
    got = None
check_hash('hidden: H(Y | X)', got, '{sha_of(ns3["hy_x"])}')
try:
    got = conditional_entropy(J3.T)
except Exception:
    got = None
check_hash('hidden: H(X | Y)', got, '{sha_of(ns3["hx_y"])}')
try:
    got = joint_entropy(J3) - conditional_entropy(J3)
except Exception:
    got = None
check_close('chain rule: H(X,Y) - H(Y|X) = H(X)', got, _sp_entropy(J3.sum(axis=1), base=2))
try:
    got = mutual_information(J3)
except Exception:
    got = None
check_hash('hidden: I(X; Y)', got, '{sha_of(ns3["mi"])}')
try:
    got = mutual_information(J3) - mutual_information(J3.T)
except Exception:
    got = None
check_close('I(X; Y) = I(Y; X)', got, 0.0)
try:
    px = np.array([0.3, 0.7]); py = np.array([0.2, 0.5, 0.3])
    got = mutual_information(np.outer(px, py))
except Exception:
    got = None
check_close('independent (product) table has I = 0', got, 0.0, tol=1e-9)
try:
    got = conditional_entropy(np.array([[0.5, 0.0], [0.0, 0.5]]))
except Exception:
    got = None
check_close('Y determined by X: H(Y|X) = 0', got, 0.0)
""",
    explanation_md=r"""
Everything reduces to one entropy routine applied to three objects: the full table, its row sums (the
marginal of $X$) and its column sums (the marginal of $Y$). The chain rule $H(Y \mid X) = H(X, Y) - H(X)$
avoids computing per-row conditional distributions, and transposing the table swaps the roles of $X$ and
$Y$. Mutual information is symmetric by construction, vanishes exactly for a product table, and is bounded by
the smaller marginal entropy. A frequent slip is to normalise the rows before computing $H(X, Y)$ — that gives
conditional distributions, not the joint.
""",
)

# ---------------------------------------------------------------- 4. Huffman (core)
SOL_4 = r"""
def huffman_lengths(probs):
    probs = np.asarray(probs, dtype=float)
    counter = itertools.count()
    heap = [(p, next(counter), [i]) for i, p in enumerate(probs)]
    heapq.heapify(heap)
    lengths = np.zeros(len(probs))
    while len(heap) > 1:
        p1, _, a = heapq.heappop(heap)
        p2, _, b = heapq.heappop(heap)
        for i in a + b:
            lengths[i] += 1                 # every symbol under the merged node gets one more bit
        heapq.heappush(heap, (p1 + p2, next(counter), a + b))
    return lengths

def expected_code_length(probs, lengths):
    return float(np.asarray(probs, dtype=float) @ np.asarray(lengths, dtype=float))
"""
ns4 = run(SOL_4, r"""
p8 = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03])
L8 = expected_code_length(p8, huffman_lengths(p8))
""")
ex.add(
    title="Huffman code lengths and the entropy bound", difficulty=2,
    prompt_md=r"""
Implement `huffman_lengths(probs)` returning a NumPy array with the **codeword length** (in bits) of each
symbol under a Huffman code for the probabilities `probs` (an array; use `heapq` as in the lesson — you do
not need the codewords themselves, only how many merges each symbol participates in), and
`expected_code_length(probs, lengths)` $= \sum_i p_i \ell_i$ as a float.

The checks verify on `p8 = [0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03]` that the Kraft sum
$\sum_i 2^{-\ell_i}$ equals 1, that $H \le L < H + 1$, that more probable symbols never get longer codes, and
they hash the expected length. Dyadic probabilities must give $L = H$ exactly.
""",
    starter=r"""
def huffman_lengths(probs):
    # YOUR CODE HERE  (hint: push (probability, tie_breaker, [symbol indices]) tuples onto a heap)
    return None

def expected_code_length(probs, lengths):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_4,
    check=rf"""
from scipy.stats import entropy as _sp_entropy
p8 = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03])
try:
    L_ = np.asarray(huffman_lengths(p8), dtype=float)
    ok_shape = L_.shape == (8,)
except Exception:
    L_, ok_shape = None, False
check('lengths array has shape (8,)', ok_shape)
try:
    got = float(np.sum(2.0 ** -L_)) if ok_shape else None
except Exception:
    got = None
check_close('Kraft sum is exactly 1 (a full tree)', got, 1.0)
try:
    got = bool(np.all(np.diff(L_) >= 0)) if ok_shape else False
except Exception:
    got = False
check('more probable symbols never get longer codewords', got)
try:
    L_exp = expected_code_length(p8, L_) if ok_shape else None
except Exception:
    L_exp = None
check_hash('hidden: expected length for p8', L_exp, '{sha_of(ns4["L8"])}')
try:
    H8 = _sp_entropy(p8, base=2)
    got = (H8 <= L_exp < H8 + 1) if L_exp is not None else False
except Exception:
    got = False
check('H <= L < H + 1', got)
try:
    pd_ = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625])
    got = expected_code_length(pd_, huffman_lengths(pd_))
except Exception:
    got = None
check_close('dyadic probabilities: L = H = 1.875', got, 1.875)
try:
    got = expected_code_length(np.full(4, 0.25), huffman_lengths(np.full(4, 0.25)))
except Exception:
    got = None
check_close('four equiprobable symbols: 2 bits each', got, 2.0)
""",
    explanation_md=r"""
Huffman's algorithm never needs the codewords to know their lengths: each time two nodes merge, every symbol
beneath them moves one level deeper in the tree, so its codeword grows by one bit. Carrying the list of
symbol indices in each heap entry makes that bookkeeping a single loop; the integer tie-breaker keeps
`heapq` from trying to compare lists when probabilities tie. The result is a full binary tree, hence a Kraft
sum of exactly 1, and an expected length within one bit of the entropy (2.70 versus 2.676 bits for `p8`).
With powers-of-two probabilities the ideal lengths $-\log_2 p_i$ are already integers and $L = H$. Note that
different tie-breaking can produce different *length vectors* with the same optimal expected length — the
checks only test properties that every optimal code shares.
""",
)

# ---------------------------------------------------------------- 5. KL between Gaussians (core)
SOL_5 = r"""
def kl_gaussians(mu1, sigma1, mu2, sigma2):
    return float(np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5)

def kl_gaussians_mc(mu1, sigma1, mu2, sigma2, n, rng):
    x = rng.normal(mu1, sigma1, size=n)
    return float(np.mean(norm.logpdf(x, mu1, sigma1) - norm.logpdf(x, mu2, sigma2)))
"""
ns5 = run(SOL_5, r"""
a = kl_gaussians(0, 1, 1, 2); b = kl_gaussians(1, 2, 0, 1)
mc = kl_gaussians_mc(0, 1, 1, 2, 200_000, np.random.default_rng(0))
""")
ex.add(
    title="KL divergence between two Gaussians: closed form and Monte Carlo", difficulty=2,
    prompt_md=r"""
Implement

* `kl_gaussians(mu1, sigma1, mu2, sigma2)` — $D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1, \sigma_1^2)\,\|\,\mathcal{N}(\mu_2, \sigma_2^2)\big)$ in **nats**, from the closed form in the guide (a float);
* `kl_gaussians_mc(mu1, sigma1, mu2, sigma2, n, rng)` — a Monte Carlo estimate from `n` samples $x_i \sim \mathcal{N}(\mu_1, \sigma_1^2)$ drawn with `rng.normal`, as the average of $\log p_1(x_i) - \log p_2(x_i)$ (use `scipy.stats.norm.logpdf`).

The checks hash the closed-form values for $(\mu_1, \sigma_1, \mu_2, \sigma_2) = (0, 1, 1, 2)$ and the
reverse direction, and require the Monte Carlo estimate with $n = 200{,}000$ to be within $0.01$ of the
closed form.
""",
    starter=r"""
def kl_gaussians(mu1, sigma1, mu2, sigma2):
    # YOUR CODE HERE
    return None

def kl_gaussians_mc(mu1, sigma1, mu2, sigma2, n, rng):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_5,
    check=rf"""
try:
    got = kl_gaussians(0, 1, 0, 1)
except Exception:
    got = None
check_close('identical Gaussians: KL = 0', got, 0.0)
try:
    got = kl_gaussians(0, 1, 1, 1)
except Exception:
    got = None
check_close('same variance, means one sd apart: KL = 1/2', got, 0.5)
try:
    got = kl_gaussians(0, 1, 1, 2)
except Exception:
    got = None
check_hash('hidden: KL(N(0,1) || N(1,4))', got, '{sha_of(ns5["a"])}')
try:
    got = kl_gaussians(1, 2, 0, 1)
except Exception:
    got = None
check_hash('hidden: the reverse direction KL(N(1,4) || N(0,1))', got, '{sha_of(ns5["b"])}')
try:
    got = kl_gaussians(0.5, 0.8, 0, 1) - 0.5 * (0.5**2 + 0.8**2 - 1 - np.log(0.8**2))
except Exception:
    got = None
check_close('VAE special case: 0.5 (mu^2 + sigma^2 - 1 - log sigma^2)', got, 0.0)
try:
    mc = kl_gaussians_mc(0, 1, 1, 2, 200_000, np.random.default_rng(0))
    got = abs(mc - kl_gaussians(0, 1, 1, 2)) < 0.01
except Exception:
    got = False
check('Monte Carlo estimate within 0.01 of the closed form', got)
""",
    explanation_md=r"""
The closed form comes from writing $\log\frac{p_1(x)}{p_2(x)} = \log\frac{\sigma_2}{\sigma_1} -
\frac{(x - \mu_1)^2}{2\sigma_1^2} + \frac{(x - \mu_2)^2}{2\sigma_2^2}$ and taking the expectation under $p_1$:
$\mathbb{E}[(x - \mu_1)^2] = \sigma_1^2$ and $\mathbb{E}[(x - \mu_2)^2] = \sigma_1^2 + (\mu_1 - \mu_2)^2$. The
Monte Carlo version is the definition $D_{\mathrm{KL}} = \mathbb{E}_{p_1}[\log p_1 - \log p_2]$ with the
expectation replaced by a sample mean; its standard error with $200{,}000$ samples is about $0.002$, well
inside the tolerance. Two classic mistakes: sampling from $p_2$ instead of $p_1$ (that estimates the reverse
KL, which here is $1.307$ rather than $0.443$), and forgetting the $-\tfrac12$.
""",
)

# ---------------------------------------------------------------- 6. information gain (core)
CHURN_DATA = r"""
customers = {
    "contract":      ["monthly", "monthly", "monthly", "monthly", "monthly", "monthly",
                      "yearly", "yearly", "yearly", "yearly", "yearly", "yearly"],
    "support_calls": ["high", "high", "low", "high", "low", "low",
                      "low", "low", "high", "low", "low", "high"],
    "region":        ["north", "south", "north", "north", "south", "north",
                      "south", "north", "south", "south", "north", "north"],
}
churned = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
"""
SOL_6 = r"""
def label_entropy(labels):
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))

def information_gain(feature_values, labels):
    f = np.asarray(feature_values); y = np.asarray(labels)
    cond = 0.0
    for v in np.unique(f):
        m = f == v
        cond += m.mean() * label_entropy(y[m])
    return float(label_entropy(y) - cond)

def best_feature(features, labels):
    gains = {name: information_gain(vals, labels) for name, vals in features.items()}
    return max(gains, key=gains.get)
"""
ns6 = run(SOL_6, CHURN_DATA + r"""
ig_contract = information_gain(customers["contract"], churned)
ig_calls = information_gain(customers["support_calls"], churned)
ig_region = information_gain(customers["region"], churned)
best = best_feature(customers, churned)
""")
ex.add(
    title="Information gain and the best root split for a churn tree", difficulty=2,
    prompt_md=r"""
Twelve telecom customers are described by three categorical features; `churned` is the label. Implement

* `label_entropy(labels)` — entropy in bits of the empirical label distribution of an array of labels;
* `information_gain(feature_values, labels)` — $H(Y) - \sum_a \frac{N_a}{N} H(Y \mid A = a)$ in bits, where `feature_values` is a list/array of categorical values aligned with `labels`;
* `best_feature(features, labels)` — given a dict `{name: list_of_values}`, return the **name** (a string) of the feature with the largest information gain.

The data:
""" + "\n```python\n" + textwrap.dedent(CHURN_DATA).strip() + "\n```\n",
    starter=CHURN_DATA + r"""

def label_entropy(labels):
    # YOUR CODE HERE
    return None

def information_gain(feature_values, labels):
    # YOUR CODE HERE
    return None

def best_feature(features, labels):
    # YOUR CODE HERE
    return None
""",
    solution=CHURN_DATA + "\n" + SOL_6,
    check=rf"""
try:
    got = label_entropy(churned)
except Exception:
    got = None
check_close('H(churned) = H_b(5/12)', got, -(5/12) * np.log2(5/12) - (7/12) * np.log2(7/12))
try:
    got = information_gain(churned, churned)
except Exception:
    got = None
check_close('a feature equal to the label gains the whole entropy', got, label_entropy(churned) if got is not None else 1.0)
try:
    got = information_gain(["a"] * 12, churned)
except Exception:
    got = None
check_close('a constant feature gains nothing', got, 0.0)
try:
    got = information_gain(customers["contract"], churned)
except Exception:
    got = None
check_hash('hidden: gain of contract', got, '{sha_of(ns6["ig_contract"])}')
try:
    got = information_gain(customers["support_calls"], churned)
except Exception:
    got = None
check_hash('hidden: gain of support_calls', got, '{sha_of(ns6["ig_calls"])}')
try:
    got = information_gain(customers["region"], churned)
except Exception:
    got = None
check_hash('hidden: gain of region', got, '{sha_of(ns6["ig_region"])}')
try:
    got = best_feature(customers, churned)
except Exception:
    got = None
check_hash('hidden: the best root feature', got, '{sha_of(ns6["best"])}')
""",
    explanation_md=r"""
Information gain is the mutual information between the label and the feature, computed from the node's
empirical frequencies: entropy of the labels before the split, minus the size-weighted average entropy of
the labels within each branch. Here the number of support calls is the most informative question — high-call
customers churn in 4 of 5 cases, low-call customers in 1 of 7 — contract type comes second, and region is
almost pure noise (its branches have nearly the same churn rate as the whole node, so the gain is less than
a thousandth of a bit). A feature identical to the label gains all of $H(Y)$; a constant feature gains zero.
The usual bug is weighting the branch entropies equally instead of by $N_a/N$.
""",
)

# ---------------------------------------------------------------- 7. perplexity & temperature (core)
SOL_7 = r"""
def perplexity(token_probs):
    token_probs = np.asarray(token_probs, dtype=float)
    return float(np.exp(-np.mean(np.log(token_probs))))

def softmax_temperature(logits, T):
    z = np.asarray(logits, dtype=float) / T
    e = np.exp(z - z.max())
    return e / e.sum()

def entropy_bits_of(p):
    p = np.asarray(p, dtype=float); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def temperature_for_entropy(logits, target_bits, lo=1e-3, hi=1e3, iters=200):
    # entropy of softmax(z / T) increases with T -> bisection
    for _ in range(iters):
        mid = np.sqrt(lo * hi)                        # bisection in log T
        if entropy_bits_of(softmax_temperature(logits, mid)) < target_bits:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))
"""
ns7 = run(SOL_7, r"""
ppl = perplexity([0.5, 0.1, 0.25, 0.05, 0.8])
z7 = np.array([4.0, 2.0, 1.0, 0.0])
h1 = entropy_bits_of(softmax_temperature(z7, 1.0))
T2 = temperature_for_entropy(z7, 2 * h1)
""")
assert 2.39 < ns7["T2"] < 2.42, ns7["T2"]
ex.add(
    title="Perplexity, temperature and the entropy of a softmax", difficulty=2,
    prompt_md=r"""
Implement

* `perplexity(token_probs)` — the perplexity $\exp\big(-\frac1T\sum_t \log p_t\big)$ of a sequence from the probabilities `token_probs` the model assigned to the tokens that actually occurred (a float);
* `softmax_temperature(logits, T)` — $p_k \propto \exp(z_k / T)$ as a NumPy array (subtract the maximum before exponentiating for stability);
* `entropy_bits_of(p)` — entropy of a distribution in bits;
* `temperature_for_entropy(logits, target_bits)` — the temperature $T > 0$ at which `softmax_temperature(logits, T)` has entropy `target_bits`, to within $10^{-4}$ (entropy is increasing in $T$, so bisection works; search $T \in [10^{-3}, 10^{3}]$).

Checks: the perplexity of `[0.5, 0.1, 0.25, 0.05, 0.8]` (hidden), the limits $T \to 0$ and $T \to \infty$
for logits `[4, 2, 1, 0]`, and the temperature at which the entropy of those logits **doubles** relative to
$T = 1$.
""",
    starter=r"""
def perplexity(token_probs):
    # YOUR CODE HERE
    return None

def softmax_temperature(logits, T):
    # YOUR CODE HERE
    return None

def entropy_bits_of(p):
    # YOUR CODE HERE
    return None

def temperature_for_entropy(logits, target_bits, lo=1e-3, hi=1e3):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_7,
    check=rf"""
z7 = np.array([4.0, 2.0, 1.0, 0.0])
try:
    got = perplexity([0.25, 0.25, 0.25, 0.25])
except Exception:
    got = None
check_close('uniform probabilities 1/4: perplexity 4', got, 4.0)
try:
    got = perplexity([0.5, 0.1, 0.25, 0.05, 0.8])
except Exception:
    got = None
check_hash('hidden: perplexity of the five-token sequence', got, '{sha_of(ns7["ppl"])}')
try:
    p_ = softmax_temperature(z7, 1.0)
    got = float(np.sum(p_)) if p_ is not None else None
except Exception:
    got = None
check_close('softmax sums to 1', got, 1.0)
try:
    got = entropy_bits_of(softmax_temperature(z7, 1.0))
except Exception:
    got = None
check_hash('hidden: entropy (bits) of softmax([4, 2, 1, 0]) at T = 1', got, '{sha_of(ns7["h1"])}')
try:
    got = entropy_bits_of(softmax_temperature(z7, 1e-3))
except Exception:
    got = None
check_close('T -> 0: entropy -> 0 (greedy)', got, 0.0, tol=1e-6)
try:
    got = entropy_bits_of(softmax_temperature(z7, 1e6))
except Exception:
    got = None
check_close('T -> infinity: entropy -> log2 4 = 2 bits', got, 2.0, tol=1e-6)
try:
    h1 = entropy_bits_of(softmax_temperature(z7, 1.0))
    T2 = temperature_for_entropy(z7, 2 * h1)
    ok_range = 2.39 < T2 < 2.42
    h_at = entropy_bits_of(softmax_temperature(z7, T2))
except Exception:
    ok_range, h_at, h1 = False, None, 1.0
check('temperature that doubles the entropy lies in (2.39, 2.42)', ok_range)
check_close('entropy at that temperature is exactly double', h_at, 2 * h1, tol=1e-3)
""",
    explanation_md=r"""
Perplexity is the exponential of the mean negative log-probability, equivalently the reciprocal of the
geometric mean of the assigned probabilities — $4.57$ here, dominated by the two tokens that received $0.1$
and $0.05$. Dividing the logits by $T$ before the softmax is the whole of temperature scaling; subtracting
the maximum first prevents overflow for very small $T$ without changing the result. Because the entropy of
$\text{softmax}(z/T)$ increases monotonically from 0 to $\log_2 V$, bisection (here in $\log T$) finds any
target entropy. The logits $(4, 2, 1, 0)$ have $0.859$ bits at $T = 1$ and need $T \approx 2.41$ to reach
$1.717$ bits. A common error is to compute perplexity as the mean of $1/p_t$ (arithmetic, not geometric) —
that is not the exponential of the cross-entropy.
""",
)

# ---------------------------------------------------------------- 8. conceptual: forward vs reverse KL
ex.add_conceptual(
    title="Forward or reverse KL?", difficulty=2,
    prompt_md=r"""
A colleague fits a single Gaussian $q$ to data whose true density $p$ has two well-separated modes, once by
maximum likelihood and once by minimising $D_{\mathrm{KL}}(q \,\|\, p)$ numerically. The two fits look
completely different. Explain (a) which divergence maximum likelihood minimises and why, (b) what each fit
looks like and the mechanism behind the difference, and (c) which direction variational inference uses and
what that implies for the width of its approximate posteriors.
""",
    answer_md=r"""
(a) The average log-likelihood is $\frac1n\sum_i \log q(x_i) \approx \mathbb{E}_p[\log q] = -H(p, q)$, and
$H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)$ with $H(p)$ fixed, so maximising likelihood minimises the
**forward** KL $D_{\mathrm{KL}}(p \,\|\, q)$. (b) Forward KL weights $\log(p/q)$ by $p$: wherever the data
have density, $q$ must too, or the integrand explodes. The maximum-likelihood Gaussian therefore matches the
mean and variance of the whole data set — one broad bump centred between the modes, mode-*covering*, with most
of its mass where there are no data. Reverse KL weights $\log(q/p)$ by $q$: $q$ is penalised only where it puts
mass and $p$ is small, so it shrinks onto one mode and ignores the other, mode-*seeking*; which mode it picks
depends on the initialisation. (c) Variational inference maximises the ELBO, whose gap to $\log p(x)$ is
$D_{\mathrm{KL}}(q(z) \,\|\, p(z \mid x))$ — a **reverse** KL with $q$ as the first argument. Consequently
variational posteriors tend to be too narrow and to miss secondary modes, and uncertainty estimates derived from
them are over-confident. Expectation propagation and importance-weighted bounds exist partly to counter this.
""",
)

# ---------------------------------------------------------------- 9. conceptual: diagnose MI estimate (stretch)
ex.add_conceptual(
    title="Diagnose this mutual-information claim", difficulty=3,
    prompt_md=r"""
A data scientist has 400 customers, bins two continuous features into $20 \times 20$ cells, computes the
plug-in mutual information from the resulting contingency table, obtains $I = 0.9$ bits, and concludes that
"the two features are strongly dependent, so one of them is redundant for the model". Identify everything
that is wrong or unsupported in this argument and say what you would compute instead. Use numbers where you
can.
""",
    answer_md=r"""
First, the estimate is dominated by bias. With $N = 400$ samples spread over $400$ cells, the plug-in
mutual information is positive even for *independent* features; the first-order (Miller–Madow) bias is about
$(B_x - 1)(B_y - 1)/(2N\ln 2) = 19 \times 19 / (800 \times 0.693) \approx 0.65$ bits, and the true bias at one
sample per cell is larger still. Most of the $0.9$ bits is noise. A shuffle null — permute one feature,
recompute, repeat a few hundred times — would show the independence distribution, and the observed value
should be judged against it. Second, the scale is missing: $0.9$ bits means something very different if the
features' own entropies are $1$ bit or $4.3$ bits (the maximum for 20 bins), so report $I / \min(H_X, H_Y)$ or
compare with $-\tfrac12\log_2(1 - \rho^2)$ for a Gaussian of the same correlation. Third, better estimators
exist for continuous data: fewer bins, bias correction, or the $k$-nearest-neighbour estimator in
`mutual_info_regression`. Finally, even a genuine dependence between two features does not make one
redundant for predicting a *label*: redundancy is about $I(X_2; Y \mid X_1)$, the information $X_2$ adds about
$Y$ beyond $X_1$, which can be large even when $I(X_1; X_2)$ is large (the data-processing inequality bounds
only what can be extracted from $X_1$ alone).
""",
)

# ---------------------------------------------------------------- 10. ELBO (stretch)
SOL_10 = r"""
def log_evidence(prior, lik):
    return float(np.log(np.sum(np.asarray(prior) * np.asarray(lik))))

def elbo(q, prior, lik):
    q, prior, lik = (np.asarray(a, dtype=float) for a in (q, prior, lik))
    m = q > 0
    recon = np.sum(q[m] * np.log(lik[m]))
    kl_prior = np.sum(q[m] * np.log(q[m] / prior[m]))
    return float(recon - kl_prior)

def elbo_gap(q, prior, lik):
    return float(log_evidence(prior, lik) - elbo(q, prior, lik))
"""
ns10 = run(SOL_10, r"""
prior10 = np.array([0.5, 0.3, 0.2]); lik10 = np.array([0.9, 0.3, 0.05]); q10 = np.array([0.6, 0.3, 0.1])
e_val = elbo(q10, prior10, lik10); gap_val = elbo_gap(q10, prior10, lik10)
""")
ex.add(
    title="The ELBO and its gap for a discrete latent variable", difficulty=3,
    prompt_md=r"""
A latent variable $z \in \{1, 2, 3\}$ has prior $p(z)$ = `prior`, and the observed $x$ has likelihoods
$p(x \mid z)$ = `lik` (one number per $z$). Implement, all in **nats**:

* `log_evidence(prior, lik)` $= \log p(x) = \log\sum_z p(z)\,p(x \mid z)$;
* `elbo(q, prior, lik)` $= \mathbb{E}_q[\log p(x \mid z)] - D_{\mathrm{KL}}(q(z)\,\|\,p(z))$ for a distribution `q` over $z$ (treat $q(z) = 0$ terms as 0);
* `elbo_gap(q, prior, lik)` $= \log p(x) - \mathrm{ELBO}(q)$.

The checks use `prior = [0.5, 0.3, 0.2]`, `lik = [0.9, 0.3, 0.05]`, `q = [0.6, 0.3, 0.1]`, and verify that the
gap equals $D_{\mathrm{KL}}(q \,\|\, p(z \mid x))$ with the true posterior, that the bound is tight at the
posterior, and that it is never violated for random $q$.
""",
    starter=r"""
def log_evidence(prior, lik):
    # YOUR CODE HERE
    return None

def elbo(q, prior, lik):
    # YOUR CODE HERE
    return None

def elbo_gap(q, prior, lik):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_10,
    check=rf"""
from scipy.stats import entropy as _sp_entropy
prior10 = np.array([0.5, 0.3, 0.2]); lik10 = np.array([0.9, 0.3, 0.05]); q10 = np.array([0.6, 0.3, 0.1])
post10 = prior10 * lik10 / np.sum(prior10 * lik10)
try:
    got = log_evidence(prior10, lik10)
except Exception:
    got = None
check_close('log p(x) = log 0.55', got, np.log(0.55))
try:
    got = elbo(q10, prior10, lik10)
except Exception:
    got = None
check_hash('hidden: ELBO at q = (0.6, 0.3, 0.1)', got, '{sha_of(ns10["e_val"])}')
try:
    got = elbo_gap(q10, prior10, lik10)
except Exception:
    got = None
check_close('gap = KL(q || true posterior)', got, _sp_entropy(q10, post10))
try:
    got = elbo_gap(post10, prior10, lik10)
except Exception:
    got = None
check_close('the bound is tight at the true posterior', got, 0.0, tol=1e-9)
try:
    ok = True
    for _ in range(50):
        q_rand = rng.dirichlet(np.ones(3))
        ok = ok and (elbo(q_rand, prior10, lik10) <= log_evidence(prior10, lik10) + 1e-12)
    got = ok
except Exception:
    got = False
check('ELBO <= log p(x) for 50 random q', got)
try:
    got = elbo([1.0, 0.0, 0.0], prior10, lik10)
except Exception:
    got = None
check_close('point-mass q on z = 1: log p(x|1) - log(1/p(1))', got, np.log(0.9) - np.log(1 / 0.5))
""",
    explanation_md=r"""
With a finite latent space the evidence is a plain sum, so every quantity can be computed exactly and the
ELBO identity verified to machine precision: $\log p(x) - \mathrm{ELBO}(q) = \sum_z q(z)\log\frac{q(z)}{p(z \mid x)}$,
which is zero exactly when $q$ is the posterior $(0.818, 0.164, 0.018)$ and positive otherwise — the random
$q$'s never break the bound. A point mass on one $z$ makes the KL-to-prior term $\log(1/p(z))$, the cost of
"paying for" that latent, against a reconstruction term of $\log p(x \mid z)$. VAEs compute exactly these two
terms, except that $z$ is continuous, the expectation is approximated with one reparameterised sample, and $q$
is produced by an encoder network.
""",
)

# ---------------------------------------------------------------- 11. mini-project (★★★)
TEXT_DATA = r"""
train_text = (
    "The market opens early on Saturday mornings, long before the first tram rattles across the bridge. "
    "Farmers from the hills arrive with crates of apples and pears, and the baker sets out loaves that are "
    "still warm from the oven. By eight the square is full of people carrying baskets, comparing prices, and "
    "stopping to talk with neighbours they have not seen all week. Children chase each other between the "
    "stalls while their parents argue gently about whether the household really needs another jar of honey. "
    "Near the fountain an old man sells used books from a folding table; he claims to have read every one of "
    "them, and nobody has ever caught him out. When it rains the whole scene moves under the arches of the "
    "town hall, and the voices echo until the market sounds twice as busy as it is. The cheese seller keeps a "
    "small notebook in which she records what sold and what did not, and over the years the notebook has "
    "become a better forecast of the weather than the radio: when the soft cheeses sell out, the afternoon "
    "will be warm. Towards noon the crowd thins, the farmers pack what remains into their vans, and the "
    "sweepers arrive with their wide brooms. By one o'clock the square belongs again to the pigeons, who "
    "inspect the cobblestones for crumbs with the patience of accountants. The library on the corner opens at "
    "two. Its reading room has tall windows and a clock that has been seven minutes fast for as long as anyone "
    "can remember, so that readers who trust it are always early for dinner. The librarian knows which "
    "regulars want the newspaper and which want the chess books, and she places them on the right tables "
    "before the doors open. On winter afternoons the radiators knock and hiss, and the room fills with the "
    "smell of wet coats drying. Students come here to work because the silence is kinder than the silence at "
    "home; it is a silence shared with other people, which is a different thing entirely. In the evening the "
    "cafes along the river put out their lamps, and the smell of coffee and fried onions drifts across the "
    "water. A small band plays on the terrace of the oldest cafe, mostly songs that everybody knows, so that "
    "the audience hums along without quite meaning to. The river itself moves slowly here, carrying leaves "
    "and the occasional lost football towards the sea, and couples lean on the railing to watch it go as if "
    "it were a performance arranged for them. The bridge was rebuilt after the war with stone from the old "
    "quarry, and the masons left their marks on the underside where only the boatmen can see them. Every "
    "spring the river rises to the second step of the landing stage and the children bet on whether it will "
    "reach the third; it has done so twice in living memory, and both times the band played on."
)

heldout_text = (
    "On Sunday the town is quieter. The bells of the two churches disagree about the time by a minute or so, "
    "and for that minute the whole valley rings. Families walk up the hill to the ruined castle, where the "
    "view takes in the river, the bridge, the market square and the roofs of the old quarter, and where the "
    "wind is strong enough to steal a hat. The cafe at the top sells hot chocolate and postcards that nobody "
    "writes any more. In the afternoon the football pitch below the castle fills with the sound of a match "
    "that matters enormously to the twenty people watching it. By evening the lamps along the river are lit "
    "again, the band has packed up its instruments, and the town settles into the long quiet that lasts until "
    "the first tram on Monday."
)
"""

PROJECT_CODE = TEXT_DATA + r"""

import re

ALPHABET = "abcdefghijklmnopqrstuvwxyz "
V = len(ALPHABET)
idx = {c: i for i, c in enumerate(ALPHABET)}

def preprocess(text):
    text = re.sub(r"[^a-z ]", " ", text.lower())      # keep letters and spaces only
    return re.sub(r" +", " ", text).strip()

train = preprocess(train_text); heldout = preprocess(heldout_text)
print(f"training characters: {len(train)},  held-out characters: {len(heldout)},  alphabet size V = {V}")

# --- models: uniform, unigram, bigram (add-alpha smoothing) ---------------------------------
def unigram_logprobs(text, alpha):
    counts = np.zeros(V)
    for c in text:
        counts[idx[c]] += 1
    return np.log2((counts + alpha) / (counts.sum() + alpha * V))

def bigram_logprobs(text, alpha):
    counts = np.zeros((V, V))                         # counts[prev, next]
    for a, b in zip(text[:-1], text[1:]):
        counts[idx[a], idx[b]] += 1
    probs = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * V)
    return np.log2(probs)

def cross_entropy_unigram(logp, text):
    return -np.mean([logp[idx[c]] for c in text])

def cross_entropy_bigram(logp2, logp1, text):
    # the first character is scored with the unigram model, the rest with the bigram model
    ll = [logp1[idx[text[0]]]] + [logp2[idx[a], idx[b]] for a, b in zip(text[:-1], text[1:])]
    return -np.mean(ll)

alpha = 0.5
lp1 = unigram_logprobs(train, alpha); lp2 = bigram_logprobs(train, alpha)
results = {
    "uniform": np.log2(V),
    "unigram": cross_entropy_unigram(lp1, heldout),
    "bigram":  cross_entropy_bigram(lp2, lp1, heldout),
}
print(f"\nheld-out cross-entropy (alpha = {alpha}):")
for name, ce in results.items():
    print(f"  {name:8s} {ce:.4f} bits/char   perplexity {2**ce:6.2f}   compression vs 8-bit ASCII: {100 * ce / 8:.1f}%")
print(f"\ntraining-set cross-entropy (optimistic): unigram {cross_entropy_unigram(lp1, train):.4f}, bigram {cross_entropy_bigram(lp2, lp1, train):.4f} bits/char")

# --- smoothing sweep --------------------------------------------------------------------------
alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
ce_bi = [cross_entropy_bigram(bigram_logprobs(train, a), unigram_logprobs(train, a), heldout) for a in alphas]
ce_uni = [cross_entropy_unigram(unigram_logprobs(train, a), heldout) for a in alphas]
plt.semilogx(alphas, ce_bi, "o-", label="bigram (held-out)")
plt.semilogx(alphas, ce_uni, "s-", label="unigram (held-out)")
plt.axhline(np.log2(V), color="k", ls="--", label="uniform log2 27")
plt.xlabel("smoothing pseudo-count alpha"); plt.ylabel("cross-entropy (bits/char)")
plt.title("Held-out cross-entropy of character models"); plt.legend(); plt.show()
best_a = alphas[int(np.argmin(ce_bi))]
print(f"best alpha for the bigram model on held-out text: {best_a}  ({min(ce_bi):.4f} bits/char)")

# --- what the models know: most surprising held-out characters, and a sample --------------------
surprises = sorted(((-lp2[idx[a], idx[b]], a + b) for a, b in zip(heldout[:-1], heldout[1:])), reverse=True)[:5]
print("\nmost surprising bigrams in the held-out text (bits):", [(f"{s:.1f}", pair) for s, pair in surprises])
rng = np.random.default_rng(14)
probs2 = 2 ** lp2
c = idx["t"]; sample = ["t"]
for _ in range(120):
    c = rng.choice(V, p=probs2[c]); sample.append(ALPHABET[c])
print("sampled from the bigram model:", "".join(sample))
"""
ex.add_project(
    title="Mini-project: a character-level language model measured in bits",
    prompt_md=r"""
Build the smallest possible language model and evaluate it the way large ones are evaluated — by its
cross-entropy in bits per character and its perplexity on text it has not seen.

**Data.** The two strings `train_text` and `heldout_text` are provided in the starter cell. Lower-case them,
map every character outside `a`–`z` to a space, collapse repeated spaces: the alphabet then has $V = 27$
symbols.

**Models.** (1) The *uniform* model, $p(c) = 1/27$. (2) A *unigram* model with add-$\alpha$ smoothing,
$p(c) = \frac{n_c + \alpha}{N + \alpha V}$. (3) A *bigram* model, $p(c_t \mid c_{t-1}) = \frac{n_{c_{t-1}c_t} + \alpha}{n_{c_{t-1}} + \alpha V}$,
with the first character of a text scored by the unigram model.

**Deliverables.**

* The held-out cross-entropy $-\frac1T\sum_t \log_2 p(c_t \mid c_{t-1})$ in bits per character and the
  perplexity $2^{\text{CE}}$ for all three models, plus the training-set cross-entropy of the bigram model for
  comparison (why is it lower?).
* A plot of held-out cross-entropy against $\alpha \in \{0.01, 0.03, 0.1, 0.3, 1, 3\}$ for the unigram and
  bigram models, and the best $\alpha$. Explain the shape of the bigram curve.
* The compression ratio each model would achieve with an arithmetic coder relative to 8-bit ASCII.
* The five most surprising character pairs in the held-out text, and a 120-character sample drawn from the
  bigram model (seed your generator) — does it look like English at all?
""",
    rubric_md=r"""
* Preprocessing is deterministic and the alphabet has exactly 27 symbols; the models are estimated from the
  training text **only**.
* Smoothing is applied so that no held-out character has probability 0 (an unsmoothed bigram model gives
  infinite cross-entropy on unseen pairs — you should be able to say which pairs).
* Cross-entropy is a mean of $-\log_2 p$ over held-out characters (bits/char) and perplexity is $2^{\text{CE}}$;
  the uniform model gives exactly $\log_2 27 = 4.755$ bits and perplexity 27.
* The bigram model beats the unigram model, which beats uniform, on held-out text; training cross-entropy is
  lower than held-out cross-entropy and you explain why (the model has seen those exact pairs).
* The $\alpha$ sweep shows a U-shape for the bigram model (too little smoothing over-trusts rare pairs, too much
  drags every conditional towards uniform) and you report the best value.
* The figure has a title, axis labels and a legend.
""",
    solution_sketch_md=r"""
Map characters to indices, count unigrams and the $27 \times 27$ bigram table on the training text, smooth
with $\alpha$, take $\log_2$ once, and score the held-out text by table look-ups. With $\alpha = 0.5$ the
reference implementation gets about $4.0$ bits/char for the unigram model and about $3.2$ bits/char for the
bigram model on the held-out text, against $4.755$ for uniform — the bigram model "compresses" the text to
about 40% of 8-bit ASCII, the unigram model to about 50%. The training cross-entropy of the bigram model is
lower (about $3.16$ bits) because it has seen exactly those pairs; the gap is overfitting measured in bits.
The $\alpha$ sweep is U-shaped for the bigram model: with $\alpha = 0.01$ unseen pairs cost many bits each,
with $\alpha = 3$ every conditional is pulled towards uniform; the minimum is near $\alpha \approx 0.1$ for a
training text this small. The most surprising held-out pairs are ones the training text never contained.
Sampling from the bigram model produces pronounceable nonsense with English-looking fragments ("the",
"ing") — the model knows which letter tends to follow which, nothing more. Shannon's estimate of about 1 bit
per character for English with long context shows how far a bigram model is from what spelling, grammar and
meaning allow.
""",
    solution_code=PROJECT_CODE,
    starter=TEXT_DATA + r"""

# Your mini-project workspace: preprocess, count, smooth, score the held-out text in bits/char.
""",
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print(f"exercises: {len(ex.items)} items — {kinds.count('code')} coding, {kinds.count('concept')} conceptual, {kinds.count('project')} project")
print("difficulties:", [it["difficulty"] for it in ex.items])
