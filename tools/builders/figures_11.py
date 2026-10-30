"""Figures for the Module 11 concept guide — Likelihood and Bayesian Inference.

Writes SVG files into part-4-statistics/11-likelihood-and-bayesian-inference/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, recomputed here and checked against the value the guide prints, so
picture and text agree exactly. Nothing is random: all curves are exact formulas
or the output of a deterministic optimiser.

    python tools/builders/figures_11.py                       # (re)write every figure
    python tools/builders/figures_11.py fisher-curvature      # only the named figure(s)
    python tools/builders/figures_11.py --preview DIR         # also write PNG previews to DIR
"""
import argparse
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402,F401

import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import optimize, special, stats  # noqa: E402

FIG_DIR = ROOT / "part-4-statistics/11-likelihood-and-bayesian-inference/figures"

WHITE_BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)  # lifts text off lines and fills
# Text colours: darker than the C1 / C2 / GREY fills and lines they label, so small text keeps its contrast
GREY_TEXT, DARK_ORANGE, DARK_GREEN, DARK_BLUE, DARK_RED = "0.35", "#b35806", "#1e7b1e", "#17598a", "#b01e1e"


def check(value, guide, tol):
    """Return `value` after confirming that it matches the number the guide prints (to `tol`)."""
    if abs(float(value) - guide) > tol:
        raise AssertionError(f"recomputed {float(value)!r}, but the guide prints {guide}")
    return value


def boxed(ax, xy, text, color="k", offset=(0, 0), fontsize=9, ha="center", va="center"):
    """Like figbuild.label, but on a translucent white card so it stays readable over fills and lines."""
    ax.annotate(text, xy, textcoords="offset points", xytext=offset, color=color, fontsize=fontsize,
                ha=ha, va=va, bbox=WHITE_BOX, zorder=6)


def pointer(ax, xy, xytext, text, color="k", fontsize=9, ha="center", va="center"):
    """Text at `xytext` with a thin arrow to the data point `xy`; the arrow stops short of what it points at."""
    ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=fontsize, ha=ha, va=va, zorder=6,
                bbox=WHITE_BOX,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1, shrinkA=1, shrinkB=4, mutation_scale=10))


def interval_bar(ax, lo, hi, y, color, lw=3, ls="-", capsize=0.25):
    """A horizontal interval [lo, hi] at height `y` with end caps; `capsize` is the half-height in data units."""
    ax.plot([lo, hi], [y, y], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=4)
    for x in (lo, hi):
        ax.plot([x, x], [y - capsize, y + capsize], color=color, lw=1.6, zorder=4)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_likelihood_two_readings():
    """Section 1: the table p(k | p) read along a row (a distribution) and along a column (the likelihood)."""
    k_all = np.arange(11)
    n_k = 7

    # (a) the table itself; k is discrete, so the cells are 1 wide; 51 rows of height 0.02 centred on 0, 0.02, ...
    p_grid = np.linspace(0, 1, 51)
    table = np.array([stats.binom.pmf(k_all, 10, p) for p in p_grid])

    def L(p):
        return 120 * p**7 * (1 - p)**3

    # the guide's table of L(p) and the likelihood ratio
    table_p = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    table_L = L(table_p)
    for v, g in zip(table_L, (0.117, 0.215, 0.267, 0.201, 0.057)):
        check(v, g, 0.0006)
    for v, g in zip(np.log(table_L), (-2.144, -1.537, -1.321, -1.603, -2.858)):
        check(v, g, 0.0006)
    ratio = check(L(0.7) / L(0.5), 2.28, 0.005)
    check(120 * special.beta(8, 4), 1 / 11, 1e-12)
    row = stats.binom.pmf(k_all, 10, 0.7)
    check(row.sum(), 1.0, 1e-12)
    check(row[n_k], table_L[2], 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))

    ax = axes[0]
    ax.grid(False)
    ax.pcolormesh(np.arange(-0.5, 11), np.linspace(-0.01, 1.01, 52), table, cmap="Blues", vmin=0, vmax=0.45,
                  shading="flat", edgecolors="face", linewidth=0.15)
    ax.add_patch(Rectangle((-0.5, 0.69), 11, 0.02, fill=False, ec="C1", lw=2, zorder=3))
    ax.add_patch(Rectangle((6.5, -0.01), 1, 1.02, fill=False, ec="C0", lw=2, zorder=3))
    ax.text(-0.2, 0.86, "fix $p = 0.7$, vary $k$:\na distribution", color=DARK_ORANGE, fontsize=9, ha="left", va="center")
    ax.text(10.42, 0.22, "fix $k = 7$,\nvary $p$:\nthe\nlikelihood", color=DARK_BLUE, fontsize=9, ha="right", va="center")
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([0, 2, 4, 6, 7, 8, 10])
    ax.set_title("(a) One table $p(k \\mid p)$, two readings")
    ax.set_xlabel("number of heads $k$"); ax.set_ylabel("head probability $p$")

    ax = axes[1]
    colors = ["C0" if k == n_k else "#f7c99b" for k in k_all]
    ax.bar(k_all, row, color=colors, edgecolor=["C0" if k == n_k else "C1" for k in k_all], lw=0.8, width=0.8)
    ax.set_ylim(0, 0.34)
    ax.set_xticks([0, 2, 4, 6, 7, 8, 10])
    ax.text(0.04, 0.93, "the 11 bars sum to 1", transform=ax.transAxes, fontsize=9, color=DARK_ORANGE, ha="left", va="center")
    pointer(ax, (7, row[n_k]), (3.0, 0.25), f"the same number:\n$L$(0.7) = {row[n_k]:.3f}", color=DARK_BLUE)
    ax.set_title("(b) Row $p = 0.7$: probability of each $k$")
    ax.set_xlabel("number of heads $k$"); ax.set_ylabel("$P(k \\mid p = 0.7)$")

    ax = axes[2]
    p = np.linspace(0, 1, 401)
    ax.fill_between(p, L(p), color="C0", alpha=0.18, lw=0)
    ax.plot(p, L(p), color="C0", lw=2)
    ax.plot(table_p, table_L, "o", color="C0", ms=5, zorder=4)
    ax.axvline(0.7, color=GREY, ls="--", lw=1)
    ax.plot([0.5, 0.7], [L(0.7), L(0.7)], color=GREY, ls=":", lw=1)
    ax.annotate("", xy=(0.5, L(0.7)), xytext=(0.5, L(0.5)),
                arrowprops=dict(arrowstyle="<|-|>", color="k", lw=1.2, shrinkA=2, shrinkB=0, mutation_scale=10))
    ax.text(0.47, 0.195, f"ratio\n0.267/0.117\n= {ratio:.2f}", ha="right", va="center", fontsize=9)
    ax.text(0.655, 0.055, "area = 1/11,\nnot 1", color=DARK_BLUE, fontsize=9, ha="center", va="center",
            bbox=WHITE_BOX, zorder=5)
    ax.text(0.715, 0.3, "MLE = 0.7", color=GREY_TEXT, fontsize=9, ha="left", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.34)
    ax.set_title("(c) Column $k = 7$: likelihood of each $p$")
    ax.set_xlabel("head probability $p$"); ax.set_ylabel("$L(p)$")
    fig.tight_layout()
    return fig


def fig_numerical_mle():
    """Section 3: Newton on the Gamma profile equation, a logistic fit that has a summit, separable data that has none."""
    # (a) the guide's Newton iteration
    x = np.array([1.2, 2.5, 0.8, 3.1, 1.9])
    xbar = check(x.mean(), 1.9, 1e-12)
    s = check(np.log(xbar) - np.log(x).mean(), 0.1121, 0.00006)
    f = lambda a: np.log(a) - special.digamma(a) - s
    df = lambda a: 1 / a - special.polygamma(1, a)
    a0 = check(xbar**2 / x.var(), 5.157, 0.0005)
    iterates = [a0]
    for _ in range(3):
        iterates.append(iterates[-1] - f(iterates[-1]) / df(iterates[-1]))
    for v, g in zip(iterates, (5.157, 4.556, 4.619, 4.6201)):
        check(v, g, 0.00051)
    for v, g in zip(f(np.array(iterates[:3])), (-0.01203, 0.00164, 0.00002)):
        check(v, g, 0.000006)
    a_hat = check(optimize.brentq(f, 3, 6), 4.620, 0.0005)
    check(a_hat / xbar, 2.432, 0.0005)

    # (b) dose-response fit: BFGS from (0, 1), as in the guide
    dose = np.array([1, 3, 10, 30, 100.0])
    k = np.array([1, 4, 10, 16, 19])

    def nll(theta):
        z = theta[0] + theta[1] * np.log(dose)
        return -np.sum(k * np.log(sigmoid(z)) + (20 - k) * np.log(sigmoid(-z)))

    a, b = optimize.minimize(nll, [0, 1], method="BFGS", options=dict(gtol=1e-7)).x
    check(a, -2.84, 0.005); check(b, 1.245, 0.0005)
    ec50 = check(np.exp(-a / b), 9.76, 0.005)

    # (c) separable data
    xs = np.array([-2, -1, 1, 2.0])
    ys = np.array([0, 0, 1, 1])

    def ell(w):
        z = w * xs
        return np.sum(ys * np.log(sigmoid(z)) + (1 - ys) * np.log(sigmoid(-z)))

    check(ell(0), -2.77, 0.005); check(ell(3), -0.10, 0.005); check(ell(10), -0.0001, 0.00005)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))

    ax = axes[0]
    aa = np.linspace(4.3, 5.4, 300)
    ax.axhline(0, color=GREY, lw=1)
    ax.plot(aa, f(aa), color="C0", lw=2, label="$f(\\alpha)$")
    for i in (0, 1):
        ai, an = iterates[i], iterates[i + 1]
        ax.plot([ai, an], [f(ai), 0], color="C1", ls="--", lw=1.4, zorder=3)      # tangent down to its zero
        ax.plot([an, an], [0, f(an)], color="C1", ls=":", lw=1.2, zorder=3)       # back up to the curve
    ax.plot(iterates[:2], f(np.array(iterates[:2])), "o", color="C1", ms=6, zorder=4)
    ax.plot([a_hat], [0], "*", color="C2", ms=13, mec="k", mew=0.5, zorder=5)
    pointer(ax, (iterates[0], f(iterates[0])), (5.08, -0.0165), "start: 5.157", color=DARK_ORANGE)
    pointer(ax, (iterates[1], f(iterates[1])), (4.47, -0.0065), "step 1:\n4.556", color=DARK_ORANGE)
    pointer(ax, (a_hat, 0), (4.92, 0.0075), "step 2: 4.619\nstep 3: 4.6201 = $\\hat\\alpha$", color=DARK_GREEN)
    ax.set_xlim(4.3, 5.4); ax.set_ylim(-0.02, 0.012)
    ax.set_title("(a) Newton on the Gamma profile")
    ax.set_xlabel("shape $\\alpha$"); ax.set_ylabel("$f(\\alpha) = \\log\\alpha - \\psi(\\alpha) - s$")

    ax = axes[1]
    dd = np.geomspace(0.5, 200, 300)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.plot([ec50, ec50], [0, 0.5], color=GREY, ls="--", lw=1)
    ax.plot(dd, sigmoid(a + b * np.log(dd)), color="C2", lw=2, label="fitted curve")
    ax.plot(dose, k / 20, "o", color="C0", ms=6, zorder=4, label="observed $k/20$")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(["1", "3", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_xlim(0.5, 200); ax.set_ylim(0, 1.02)
    pointer(ax, (ec50, 0.5), (40, 0.25), f"EC$_{{50}}$ = $\\exp(-\\hat a/\\hat b)$\n= {ec50:.2f} mg", color=GREY_TEXT)
    ax.legend(loc="upper left")
    ax.set_title("(b) Dose–response: a summit exists")
    ax.set_xlabel("dose (mg, log scale)"); ax.set_ylabel("fraction responding")

    ax = axes[2]
    ww = np.linspace(0, 10, 300)
    ax.axhline(0, color=GREY, ls=":", lw=1.2)
    ax.plot(ww, [ell(w) for w in ww], color="C3", lw=2)
    pts = [(0, -2.77), (3, -0.10), (10, -0.0001)]
    ax.plot([p[0] for p in pts], [ell(p[0]) for p in pts], "o", color="C3", ms=6, zorder=4, clip_on=False)
    label(ax, (0, -2.77), "$\\ell$(0) = −2.77", color=DARK_RED, offset=(9, -3), fontsize=9, va="center")
    label(ax, (3, -0.10), "$\\ell$(3) = −0.10", color=DARK_RED, offset=(4, -14), fontsize=9, va="center")
    pointer(ax, (10, -0.0001), (7.7, -0.62), "$\\ell$(10) = −0.0001", color=DARK_RED)
    ax.text(5.8, -1.45, "no maximum: $\\ell \\to 0$\nas $w \\to \\infty$, so the\nMLE does not exist", color=DARK_RED, fontsize=9,
            ha="center", va="center")
    ax.set_xlim(0, 10); ax.set_ylim(-3, 0.2)
    ax.set_title("(c) Separable: the summit is at infinity")
    ax.set_xlabel("slope $w$"); ax.set_ylabel("log-likelihood $\\ell(w)$")
    fig.tight_layout()
    return fig


def fig_fisher_curvature():
    """Section 4: curvature of the log-likelihood is precision; absolute and relative error at n = 400."""
    def drop(p, k, n):
        return k * np.log(p / (k / n)) + (n - k) * np.log((1 - p) / (1 - k / n))

    J10 = check(7 / 0.49 + 3 / 0.09, 47.6, 0.05)
    check(10 / 0.21, J10, 1e-9)
    J1000 = 1000 / 0.21
    se10 = check(1 / math.sqrt(J10), 0.145, 0.0005)
    se1000 = check(1 / math.sqrt(J1000), 0.0145, 0.00005)
    lo10, hi10 = 0.7 - 1.96 * se10, 0.7 + 1.96 * se10
    lo1000, hi1000 = check(0.7 - 1.96 * se1000, 0.672, 0.0005), check(0.7 + 1.96 * se1000, 0.728, 0.0005)
    check(lo10, 0.42, 0.005); check(hi10, 0.98, 0.005)
    level = -1.96**2 / 2
    ex_lo = optimize.brentq(lambda p: drop(p, 7, 10) - level, 0.05, 0.7)
    ex_hi = optimize.brentq(lambda p: drop(p, 7, 10) - level, 0.7, 0.999)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    p = np.linspace(0.3, 0.995, 800)
    ax.axhline(level, color=GREY, ls=":", lw=1)
    ax.plot(p, drop(p, 7, 10), color="C0", lw=2, label="7 heads in 10")
    ax.plot(p, -0.5 * J10 * (p - 0.7)**2, color="C0", ls="--", lw=1.3, label="its parabola, curvature $J$ = 47.6")
    ax.plot(p, drop(p, 700, 1000), color="C2", lw=2, label="700 heads in 1,000, curvature $J$ = 4762")
    interval_bar(ax, lo10, hi10, level, "C0", lw=2.5, capsize=0.14)
    interval_bar(ax, lo1000, hi1000, level, "C2", lw=4, capsize=0.2)
    ax.plot([ex_lo, ex_hi], [level, level], "o", mfc="white", mec="C0", mew=1.5, ms=6, zorder=5)
    ax.text(0.555, level + 0.17, f"drop of 1.96²/2 = {-level:.2f}", color=GREY_TEXT, fontsize=8.5, ha="center", va="center")
    ax.text(0.515, -3.0, f"$n$ = 10\nSE = {se10:.3f}\nWald [{lo10:.2f}, {hi10:.2f}]", color=DARK_BLUE, fontsize=8.5,
            ha="center", va="center")
    ax.text(0.843, -3.0, f"$n$ = 1000\nSE = {se1000:.4f}\nWald [{lo1000:.3f}, {hi1000:.3f}]", color=DARK_GREEN,
            fontsize=8.5, ha="center", va="center")
    ax.text(0.5, -4.25, f"open circles: the exact curve\ndrops {-level:.2f} at {ex_lo:.2f} and {ex_hi:.2f},\nnot symmetrically", color=DARK_BLUE,
            fontsize=8.5, ha="center", va="center")
    ax.set_xlim(0.3, 1); ax.set_ylim(-5, 1.75)
    ax.set_yticks([-5, -4, -3, -2, -1, 0])
    ax.legend(loc="upper center", fontsize=8.5)
    ax.set_title("(a) Same summit, 100 times the curvature")
    ax.set_xlabel("head probability $p$"); ax.set_ylabel("drop below the summit, $\\ell(p) - \\ell(0.7)$")

    ax = axes[1]
    n = 400
    pp = np.linspace(0.01, 0.99, 400)
    se = np.sqrt(pp * (1 - pp) / n)
    ax.plot(pp, se, color="C0", lw=2, label="absolute: SE")
    ax.plot(pp, se / pp, color="C3", lw=2, ls="--", label="relative: SE$/p$")
    se06 = check(math.sqrt(0.06 * 0.94 / n), 0.0119, 0.00005)
    ax.plot([0.06, 0.5], [se06, 0.025], "o", color="C0", ms=6, zorder=4)
    ax.plot([0.06], [se06 / 0.06], "o", color="C3", ms=6, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(0, 1); ax.set_ylim(0.003, 0.7)
    pointer(ax, (0.06, se06), (0.27, 0.0062), "6% ± 1.2% (the bank)", color=DARK_BLUE)
    pointer(ax, (0.5, 0.025), (0.62, 0.0115), "$p$ = 1/2: largest SE, 0.025", color=DARK_BLUE)
    pointer(ax, (0.06, se06 / 0.06), (0.3, 0.33), f"relative error {se06 / 0.06:.0%}", color=DARK_RED)
    ax.legend(loc="upper right")
    ax.set_title("(b) $n = 400$: absolute versus relative error")
    ax.set_xlabel("true rate $p$"); ax.set_ylabel("error (log scale)")
    fig.tight_layout()
    return fig


def fig_losses_as_noise_models():
    """Section 5: a noise density, the loss it implies, and the cross-entropy penalty."""
    r = np.array([0.5, -1.0, 0.2, 2.0, -0.3])
    sse = check(np.sum(r**2), 5.38, 1e-9)
    check(np.mean(r**2), 1.076, 1e-9)
    mae = check(np.mean(np.abs(r)), 0.8, 1e-9)
    check(2.5 * np.log(2 * np.pi) + sse / 2, 7.285, 0.0005)
    check(5 * np.log(2) + np.sum(np.abs(r)), 7.466, 0.0005)
    r2 = r.copy(); r2[3] = 4.0
    sse2 = check(np.sum(r2**2), 17.38, 1e-9)
    mae2 = check(np.mean(np.abs(r2)), 1.2, 1e-9)
    check(sse2 / sse, 3.2, 0.05); check(mae2 / mae, 1.5, 1e-9)
    ce7, ce1 = check(-np.log(0.7), 0.357, 0.0005), check(-np.log(0.1), 2.303, 0.0005)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))

    ax = axes[0]
    e = np.linspace(-4, 4, 801)
    gauss, lap = stats.norm.pdf(e), 0.5 * np.exp(-np.abs(e))
    ax.plot(e, gauss, color="C0", lw=2, label="Gaussian: thin tails")
    ax.plot(e, lap, color="C1", lw=2, ls="--", label="Laplace: sharp peak,\nheavy tails")
    g4, l4 = stats.norm.pdf(4), 0.5 * math.exp(-4)
    pointer(ax, (4, l4), (2.4, 0.445), f"density at 4:\nGaussian {g4:.5f}\nLaplace {l4:.4f}\n({l4 / g4:.0f} times more)", fontsize=8.5)
    ax.set_xlim(-4, 4.5); ax.set_ylim(0, 0.72)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("(a) Two noise models")
    ax.set_xlabel("residual $\\varepsilon$"); ax.set_ylabel("density")

    ax = axes[1]
    ax.plot(e, e**2 / 2, color="C0", lw=2, label="$\\varepsilon^2/2$ (Gaussian)")
    ax.plot(e, np.abs(e), color="C1", lw=2, ls="--", label="$|\\varepsilon|$ (Laplace)")
    ax.plot([2, 4, 4], [2, 8, 4], "o", color="C3", ms=6, zorder=4)
    ax.annotate("", xy=(4, 8), xytext=(2, 2), arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.2, shrinkA=4, shrinkB=4,
                                                              connectionstyle="arc3,rad=-0.15"))
    ax.annotate("", xy=(4, 4), xytext=(2, 2), arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.2, shrinkA=4, shrinkB=4,
                                                              connectionstyle="arc3,rad=0.2"))
    ax.text(0, 5.3, f"residual 2 → 4\nsquared loss 2 → 8\nSSE {sse:.2f} → {sse2:.2f} (×{sse2 / sse:.1f})", color=DARK_RED, fontsize=8.5,
            ha="center", va="center")
    ax.text(4.2, 0.05, f"absolute 2 → 4\nMAE {mae:.1f} → {mae2:.1f}\n(×{mae2 / mae:.1f})", color=DARK_RED, fontsize=8.5, ha="right",
            va="center")
    ax.set_xlim(-4, 4.25); ax.set_ylim(-1.0, 9)
    ax.set_yticks(range(0, 10))
    ax.legend(loc="upper center", fontsize=8.5)
    ax.set_title("(b) The losses they imply")
    ax.set_xlabel("residual $\\varepsilon$"); ax.set_ylabel("loss per residual, $-\\log p(\\varepsilon)$ + const")

    ax = axes[2]
    pr = np.geomspace(0.002, 1, 400)
    ax.plot(pr, -np.log(pr), color="C2", lw=2)
    for pv, v, txt, off in ((0.7, ce7, f"−log 0.7 = {ce7:.3f}", (6, 13)),
                            (0.1, ce1, f"−log 0.1 = {ce1:.3f}\n({ce1 / ce7:.1f} times more)", (9, 0)),
                            (0.01, -np.log(0.01), f"−log 0.01 = {-np.log(0.01):.3f}", (9, 0))):
        ax.plot([pv], [v], "o", color="C2", ms=6, zorder=4)
        ax.annotate(txt, (pv, v), textcoords="offset points", xytext=off, color=DARK_GREEN, fontsize=9,
                    ha="left" if pv < 0.5 else "center", va="center" if pv < 0.5 else "bottom")
    ax.text(0.06, 5.6, "loss → ∞ as $p$ → 0", color=GREY_TEXT, fontsize=9, ha="left", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 6)
    ax.set_title("(c) Cross-entropy hates confident mistakes")
    ax.set_xlabel("probability given to the true class"); ax.set_ylabel("loss (nats)")
    fig.tight_layout()
    return fig


def fig_prior_likelihood_posterior():
    """Section 6: the three-hypothesis update, and the Beta(3, 39) posterior against the Wald interval."""
    hyp = np.array([0.02, 0.05, 0.10])
    prior = np.array([0.25, 0.50, 0.25])
    lik = stats.binom.pmf(2, 40, hyp)
    for v, g in zip(lik, (0.1448, 0.2777, 0.1423)):
        check(v, g, 0.00006)
    prod = prior * lik
    for v, g in zip(prod, (0.0362, 0.1388, 0.0356)):
        check(v, g, 0.00006)
    evidence = check(prod.sum(), 0.2106, 0.00006)
    post = prod / evidence
    for v, g in zip(post, (0.172, 0.659, 0.169)):
        check(v, g, 0.0006)
    check(post @ hyp, 0.0533, 0.00006)

    beta = stats.beta(3, 39)
    mean, med, mode = check(beta.mean(), 0.0714, 0.00006), check(beta.median(), 0.0647, 0.00006), 2 / 40
    lo, hi = check(beta.ppf(0.025), 0.015, 0.0005), check(beta.ppf(0.975), 0.165, 0.0005)
    tail = check(beta.sf(0.10), 0.209, 0.0005)
    half = 1.96 * math.sqrt(0.05 * 0.95 / 40)
    w_lo, w_hi = check(0.05 - half, -0.018, 0.0005), check(0.05 + half, 0.118, 0.0005)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

    ax = axes[0]
    xpos = np.arange(3)
    w = 0.34
    ax.bar(xpos - w / 2, prior, w, color="#f7c99b", edgecolor="C1", label="prior")
    ax.bar(xpos + w / 2, post, w, color="#a8dca8", edgecolor="C2", hatch="//", label="posterior")
    ax.plot(xpos, lik, "D", color="C0", mec="white", ms=8, zorder=4, label="likelihood of 2 clicks in 40")
    for x, v in zip(xpos - w / 2, prior):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=DARK_ORANGE)
    for x, v in zip(xpos + w / 2, post):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=DARK_GREEN)
    ax.text(1, 0.96, f"likelihood = {lik[0]:.4f}, {lik[1]:.4f}, {lik[2]:.4f}\n"
                     f"prior × likelihood = {prod[0]:.4f}, {prod[1]:.4f}, {prod[2]:.4f}\n"
                     f"evidence = their sum = {evidence:.4f}; divide by it", fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY))
    ax.set_xticks(xpos); ax.set_xticklabels(["$p$ = 0.02", "$p$ = 0.05", "$p$ = 0.10"])
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.71), ncol=3, fontsize=8.5, columnspacing=1.2, handletextpad=0.5)
    ax.set_title("(a) Three hypotheses: prior × likelihood / evidence")
    ax.set_xlabel("click-through rate $p$"); ax.set_ylabel("probability")

    ax = axes[1]
    p = np.linspace(0.0005, 0.25, 800)
    pdf = beta.pdf(p)
    ax.axvspan(-0.03, 0, color="0.88", lw=0, zorder=0)
    ax.axhline(0, color="k", lw=0.8)
    inside = (p >= lo) & (p <= hi)
    ax.fill_between(p[inside], pdf[inside], color="C2", alpha=0.22, lw=0, label="95% credible [0.015, 0.165]")
    t = p >= 0.10
    ax.fill_between(p[t], pdf[t], facecolor="none", edgecolor=DARK_GREEN, hatch="////", lw=0)
    ax.plot(p, pdf, color="C2", lw=2, label="posterior Beta(3, 39)")
    ax.plot([0, 0.25], [1, 1], color="C1", lw=1.8, ls="--", label="uniform prior (height 1)")
    for v, ls, txt in ((mode, "-", "mode 0.05 = MLE"), (med, "--", "median 0.0647"), (mean, ":", "mean 0.0714")):
        ax.plot([v, v], [0, beta.pdf(v)], color="k", lw=1.2, ls=ls, label=txt, zorder=3)
    pointer(ax, (0.122, 2.05), (0.175, 4.4), f"$P(p > 0.10 \\mid \\mathcal{{D}})$ = {tail:.3f}", color=DARK_GREEN)
    y_w = -1.35
    interval_bar(ax, w_lo, w_hi, y_w, "C3", lw=3, capsize=0.35)
    ax.plot([0.05], [y_w], "o", color="C3", ms=6, zorder=5)
    ax.text(w_hi + 0.006, y_w, f"Wald [{w_lo:.3f}, {w_hi:.3f}]\nreaches negative rates".replace("-", "−"), color=DARK_RED,
            fontsize=8.5, ha="left", va="center")
    ax.text(-0.015, 6.2, "impossible: $p < 0$", color="#333333", fontsize=9, ha="center", va="center", rotation=90)
    ax.set_xlim(-0.03, 0.25); ax.set_ylim(-2.4, 13)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("(b) Uniform prior: exact posterior vs Wald")
    ax.set_xlabel("click-through rate $p$"); ax.set_ylabel("posterior density")
    fig.tight_layout()
    return fig


def fig_conjugate_shrinkage():
    """Section 7: Beta(2, 8) updated with 7/10 and then 70/100 heads; the posterior mean's path towards the data."""
    dists = [("prior Beta(2, 8)", stats.beta(2, 8), "C1", "-", (0.028, 0.482), 0.2),
             ("after 7 of 10: Beta(9, 11)", stats.beta(9, 11), "C2", "--", (0.244, 0.665), 0.45),
             ("after 70 of 100: Beta(72, 38)", stats.beta(72, 38), "C2", "-", (0.564, 0.740), 72 / 110)]
    for name, d, c, ls, (lo, hi), m in dists:
        check(d.ppf(0.025), lo, 0.0006); check(d.ppf(0.975), hi, 0.0006); check(d.mean(), m, 1e-12)
    check(10 / 20 * 0.7 + 10 / 20 * 0.2, 0.45, 1e-12)
    check(100 / 110 * 0.7 + 10 / 110 * 0.2, 0.6545, 0.00006)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    p = np.linspace(0, 1, 801)
    ax.axhline(0, color="k", lw=0.8)
    ax.plot([0.7, 0.7], [0, 10.1], color=GREY, ls=":", lw=1.3)
    ax.text(0.712, 9.6, "$k/n$ = 0.7", color=GREY_TEXT, fontsize=9, ha="left", va="center")
    text_colour = {"C1": DARK_ORANGE, "C2": DARK_GREEN}
    ys = (-0.65, -1.3, -2.3)
    for (name, d, c, ls, (lo, hi), m), y in zip(dists, ys):
        ax.plot(p, d.pdf(p), color=c, lw=2, ls=ls, label=name)
        ax.plot([lo, hi], [y, y], color=c, lw=2.6, ls=ls, solid_capstyle="butt", zorder=4)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.17, y + 0.17], color=c, lw=1.5, zorder=4)
        ax.plot([m], [y], "o", color=c, mec="k", mew=0.6, ms=6.5, zorder=5)
        txt = f"[{lo:.3f}, {hi:.3f}], mean {m:.4g}"
        if y == -1.3:   # too long for one line in the room to the right of this bar
            ax.text(hi + 0.015, y - 0.2, txt.replace(", mean", ",\nmean"), color=text_colour[c], fontsize=8.5, ha="left", va="center")
        elif hi < 0.7:
            ax.text(hi + 0.015, y, txt, color=text_colour[c], fontsize=8.5, ha="left", va="center")
        else:
            ax.text(lo - 0.015, y, txt, color=text_colour[c], fontsize=8.5, ha="right", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(-2.8, 10.4)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("(a) A prior updated, with 95% credible intervals below")
    ax.set_xlabel("head probability $p$"); ax.set_ylabel("density")

    ax = axes[1]
    n = np.geomspace(1, 1000, 300)
    ax.axhline(0.2, color=GREY, ls=":", lw=1.2)
    ax.axhline(0.7, color=GREY, ls=":", lw=1.2)
    mean_w = (2 + 0.7 * n) / (10 + n)
    mean_s = (20 + 0.7 * n) / (100 + n)
    ax.plot(n, mean_w, color="C2", lw=2, label="prior Beta(2, 8): 10 pseudo-flips")
    ax.plot(n, mean_s, color="C1", lw=2, ls="--", label="prior Beta(20, 80): 100 pseudo-flips")
    ax.plot([10, 100], [0.45, 72 / 110], "o", color="C2", ms=6.5, zorder=4)
    ax.plot([100, 1000], [0.45, 72 / 110], "s", color="C1", ms=6, zorder=4, clip_on=False)
    check((20 + 70) / 200, 0.45, 1e-12)
    check((20 + 700) / 1100, 72 / 110, 1e-12)
    ax.set_xscale("log")
    ax.set_xlim(1, 1000); ax.set_ylim(0.08, 0.8)
    ax.text(1.15, 0.725, "data: $k/n$ = 0.7", color=GREY_TEXT, fontsize=9, ha="left", va="center")
    ax.text(1.15, 0.175, "prior mean 0.2", color=GREY_TEXT, fontsize=9, ha="left", va="center")
    pointer(ax, (10, 0.45), (3.0, 0.52), "$n$ = 10: 0.45\ndata weight 50%", color=DARK_GREEN)
    pointer(ax, (100, 72 / 110), (45, 0.765), "$n$ = 100: 0.6545, data weight 91%", color=DARK_GREEN)
    pointer(ax, (100, 0.45), (235, 0.315), "$n$ = 100: the same 0.45;\nten times the prior needs\nten times the data", color=DARK_ORANGE)
    pointer(ax, (1000, 72 / 110), (365, 0.44), "$n$ = 1,000:\nthe same 0.6545", color=DARK_ORANGE)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("(b) Shrinkage fades as $n$ grows ($k/n$ fixed at 0.7)")
    ax.set_xlabel("number of flips $n$ (log scale)"); ax.set_ylabel("posterior mean")
    fig.tight_layout()
    return fig


def fig_map_ridge():
    """Section 8: the MAP objective as data parabola plus prior parabola; shrinkage against the amount of data."""
    x = np.array([1, 2, 3.0])
    y = np.array([1.0, 2.5, 2.9])
    Sxx, Sxy, Syy = check(x @ x, 14, 1e-12), check(x @ y, 14.7, 1e-9), y @ y
    mle = check(Sxy / Sxx, 1.05, 1e-9)
    map_gentle = check(Sxy / (Sxx + 0.25), 1.032, 0.0005)
    map_tight = check(Sxy / (Sxx + 25), 0.377, 0.0005)
    check(147 / 140.25, 1.048, 0.0005)
    check(1 / math.sqrt(14 / 0.25 + 1), 0.132, 0.0005)

    def sse(w):
        return Sxx * w**2 - 2 * Sxy * w + Syy

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    w = np.linspace(0, 1.5, 400)
    ax.plot(w, sse(w), color="C0", lw=2, label="data term $\\|y - wx\\|^2$")
    ax.plot(w, 25 * w**2, color="C1", lw=1.8, ls="--", label="penalty $\\lambda w^2$, $\\lambda$ = 25")
    ax.plot(w, sse(w) + 25 * w**2, color="C2", lw=2, label="their sum: MAP objective, $\\lambda$ = 25")
    ax.plot([mle], [sse(mle)], "o", mfc="white", mec="C0", mew=1.6, ms=8, zorder=4, clip_on=False)
    ax.plot([map_tight], [sse(map_tight) + 25 * map_tight**2], "o", color="C2", ms=7, zorder=4)
    ax.plot([map_gentle], [sse(map_gentle) + 0.25 * map_gentle**2], "D", color="C1", ms=4.5, zorder=5, clip_on=False)
    ax.set_xlim(0, 1.5); ax.set_ylim(0, 30)
    pointer(ax, (map_tight, sse(map_tight) + 25 * map_tight**2), (0.19, 4.9), f"MAP = {map_tight:.3f}\n($\\tau^2$ = 0.01)", color=DARK_GREEN)
    pointer(ax, (mle, sse(mle)), (1.3, 6.5), f"MLE = {mle:.2f}", color=DARK_BLUE)
    pointer(ax, (map_gentle, sse(map_gentle) + 0.25 * map_gentle**2), (0.84, 6.5), f"$\\lambda$ = 0.25: MAP = {map_gentle:.3f}", color=DARK_ORANGE)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("(a) MAP objective = data parabola + prior parabola")
    ax.set_xlabel("slope $w$"); ax.set_ylabel("sum of squares $+ \\lambda w^2$")

    ax = axes[1]
    n = np.geomspace(3, 3000, 300)
    S = Sxx * n / 3
    shrink = lambda lam, s: lam / (s + lam)   # 1 - w_MAP / w_MLE
    ax.plot(n, shrink(25, S), color="C3", lw=2, ls="--", label="tight prior, $\\lambda$ = 25")
    ax.plot(n, shrink(0.25, S), color="C1", lw=2, label="gentle prior, $\\lambda$ = 0.25")
    check(shrink(0.25, 14), 1 - map_gentle / mle, 1e-9); check(shrink(0.25, 140), 1 - (147 / 140.25) / mle, 1e-9)
    t3 = check(shrink(25, 14), 1 - map_tight / mle, 1e-9)
    ax.plot([3, 30], [shrink(0.25, 14), shrink(0.25, 140)], "o", color="C1", ms=7, zorder=4, clip_on=False)
    ax.plot([3], [t3], "s", color="C3", ms=6.5, zorder=4, clip_on=False)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(3, 3000); ax.set_ylim(1e-5, 6)
    pointer(ax, (3, t3), (20, 2), f"$n$ = 3: MAP 0.377 (down {t3:.0%})", color=DARK_RED)
    pointer(ax, (3, shrink(0.25, 14)), (7, 1.4e-3), f"$n$ = 3: 1.032\n(down {shrink(0.25, 14):.1%})", color=DARK_ORANGE)
    pointer(ax, (30, shrink(0.25, 140)), (90, 7e-3), f"$n$ = 30: 1.048\n(down {shrink(0.25, 140):.2%})", color=DARK_ORANGE)
    ax.text(3.6, 1.7e-4, "same prior, 10× the data,\n1/10 of the shrinkage", fontsize=9, ha="left", va="center", color=GREY_TEXT)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("(b) The prior's pull fades as $n$ grows")
    ax.set_xlabel("number of points $n$ (log scale, sums scaled with $n$)")
    ax.set_ylabel("shrinkage, $1 - \\hat w_{\\mathrm{MAP}}/\\hat w_{\\mathrm{MLE}}$ (log scale)")
    fig.tight_layout()
    return fig


def fig_predictive_and_occam():
    """Section 9: posterior predictive against plug-in; the Bayes factor as prior mass spread over datasets."""
    m = 100
    a, b = 3, 39
    pbar = check(a / (a + b), 0.0714, 0.00006)
    plug, pred = stats.binom(m, pbar), stats.betabinom(m, a, b)
    check(plug.mean(), 7.14, 0.005); check(plug.std(), 2.58, 0.005)
    check(pred.mean(), 7.14, 0.005); check(pred.var(), 21.9, 0.05); check(pred.std(), 4.68, 0.005)
    check(plug.var() * (a + b + m) / (a + b + 1), pred.var(), 1e-9)
    p_lo, p_hi = plug.ppf(0.025), plug.ppf(0.975)
    b_lo, b_hi = pred.ppf(0.025), pred.ppf(0.975)
    check(p_lo, 3, 0); check(p_hi, 13, 0); check(b_lo, 1, 0); check(b_hi, 18, 0)

    m1_10 = stats.binom.pmf(np.arange(11), 10, 0.5)
    ev2_10 = 1 / 11
    check(m1_10[7], 0.117, 0.0005); check(ev2_10, 0.091, 0.0005)
    bf10 = check(m1_10[7] / ev2_10, 1.29, 0.005)
    k100 = np.arange(101)
    m1_100 = stats.binom.pmf(k100, 100, 0.5)
    ev2_100 = check(1 / 101, 0.0099, 0.00005)
    check(m1_100[70], 2.3e-5, 0.05e-5)
    bf100 = check(ev2_100 / m1_100[70], 427, 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))

    ax = axes[0]
    kk = np.arange(0, 26)
    ax.bar(kk - 0.2, plug.pmf(kk), 0.4, color="#f2a6a6", edgecolor="C3", lw=0.5, label="plug-in Binomial")
    ax.bar(kk + 0.2, pred.pmf(kk), 0.4, color="#a8dca8", edgecolor="C2", lw=0.5, hatch="///", label="Beta-Binomial")
    y1, y2 = 0.185, 0.17
    interval_bar(ax, p_lo, p_hi, y1, "C3", lw=2.5, capsize=0.005)
    interval_bar(ax, b_lo, b_hi, y2, "C2", lw=2.5, capsize=0.005)
    ax.text(p_hi + 0.5, y1, f"95%: {p_lo:.0f}–{p_hi:.0f}", color=DARK_RED, fontsize=8.5, ha="left", va="center")
    ax.text(b_hi + 0.5, y2, f"95%: {b_lo:.0f}–{b_hi:.0f}", color=DARK_GREEN, fontsize=8.5, ha="left", va="center")
    ax.text(25.5, 0.098, f"same mean {plug.mean():.2f}\nSD {plug.std():.2f} vs {pred.std():.2f}", fontsize=8.5, ha="right", va="center")
    ax.set_xlim(-1, 26); ax.set_ylim(0, 0.2)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.68), fontsize=8)
    ax.set_title("(a) The next 100 impressions")
    ax.set_xlabel("clicks"); ax.set_ylabel("probability")

    ax = axes[1]
    k10 = np.arange(11)
    ax.fill_between([-0.5, 2.5], [0.25, 0.25], color="C1", alpha=0.12, lw=0)
    ax.fill_between([7.5, 10.5], [0.25, 0.25], color="C1", alpha=0.12, lw=0)
    ax.bar(k10, m1_10, 0.75, color="#a9cce3", edgecolor="C0", lw=0.8, label="$M_1$: fair coin")
    ax.bar([7], [m1_10[7]], 0.75, facecolor="#a9cce3", edgecolor="C3", lw=2.2, zorder=3)
    ax.plot([-0.5, 10.5], [ev2_10, ev2_10], color="C1", lw=2, ls="--", label="$M_2$: uniform prior, 1/11")
    ax.text(1, 0.135, "$M_2$\nwins", color=DARK_ORANGE, fontsize=8.5, ha="center", va="center")
    ax.text(9, 0.135, "$M_2$\nwins", color=DARK_ORANGE, fontsize=8.5, ha="center", va="center")
    ax.annotate(f"$k$ = 7: 0.117 vs 0.091\nBF = {bf10:.2f} for the fair coin", (7, m1_10[7]), xytext=(6.8, 0.272), fontsize=8.5,
                color=DARK_RED, ha="center", va="center", bbox=WHITE_BOX, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="C3", lw=1, shrinkB=2, mutation_scale=10))
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(0, 0.42)
    ax.set_xticks([0, 2, 4, 6, 7, 8, 10])
    ax.legend(loc="upper center", fontsize=8)
    ax.set_title("(b) 10 flips: each model sums to 1")
    ax.set_xlabel("number of heads $k$"); ax.set_ylabel("$p(k \\mid M)$")

    ax = axes[2]
    ax.plot(k100, m1_100, color="C0", lw=1.8, label="$M_1$: fair coin")
    ax.plot([0, 100], [ev2_100, ev2_100], color="C1", lw=2, ls="--", label="$M_2$: uniform prior, 1/101")
    ax.plot([70], [m1_100[70]], "o", color="C3", ms=7, zorder=5)
    ax.plot([70], [ev2_100], "o", color="C3", ms=7, zorder=5)
    ax.plot([70, 70], [m1_100[70], ev2_100], color="C3", lw=1.5, zorder=4)
    ax.annotate(f"$k$ = 70: 2.3 × 10$^{{-5}}$ vs 0.0099\nBF = {bf100:.0f} for $M_2$", (70, ev2_100), xytext=(52, 0.42), fontsize=8.5,
                color=DARK_RED, ha="center", va="center", zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="C3", lw=1, shrinkB=4, mutation_scale=10))
    ax.set_yscale("log")
    ax.set_xlim(0, 100); ax.set_ylim(1e-8, 400)
    ax.set_yticks([10.0**e for e in range(-8, 1, 2)])
    ax.legend(loc="upper center", fontsize=8)
    ax.set_title("(c) 100 flips: the flexibility pays")
    ax.set_xlabel("number of heads $k$"); ax.set_ylabel("$p(k \\mid M)$ (log scale)")
    fig.tight_layout()
    return fig


FIGURES = {
    "likelihood-two-readings": fig_likelihood_two_readings,
    "numerical-mle": fig_numerical_mle,
    "fisher-curvature": fig_fisher_curvature,
    "losses-as-noise-models": fig_losses_as_noise_models,
    "prior-likelihood-posterior": fig_prior_likelihood_posterior,
    "conjugate-shrinkage": fig_conjugate_shrinkage,
    "map-ridge": fig_map_ridge,
    "predictive-and-occam": fig_predictive_and_occam,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 11 concept-guide figures.")
    ap.add_argument("names", nargs="*", help=f"figures to draw (default: all of {', '.join(FIGURES)})")
    ap.add_argument("--preview", metavar="DIR", help="also write PNG previews to DIR")
    args = ap.parse_args()
    unknown = [n for n in args.names if n not in FIGURES]
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(unknown)}")
    for name in args.names or FIGURES:
        path = save(FIGURES[name](), FIG_DIR / f"{name}.svg", preview_dir=args.preview)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
