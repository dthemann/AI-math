"""Figures for the Module 08 concept guide — Distributions and Limit Theorems.

Writes SVG files into part-3-probability/08-distributions-and-limit-theorems/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, recomputed here and checked against the value the guide prints, so
picture and text agree exactly. Anything random (one simulated shift, coin paths,
Cauchy averages, the Monte Carlo runs) comes from a fixed np.random.default_rng seed;
the limit-theorem curves are exact, not simulated.

    python tools/builders/figures_08.py                     # (re)write every figure
    python tools/builders/figures_08.py clt-speed           # only the named figure(s)
    python tools/builders/figures_08.py --preview DIR       # also write PNG previews to DIR
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402,F401

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import integrate, stats  # noqa: E402

FIG_DIR = ROOT / "part-3-probability/08-distributions-and-limit-theorems/figures"

WHITE_BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)  # lifts text off lines and fills
# Text colours: darker than the C1 / C2 / GREY fills and lines they label, so small text keeps its contrast
GREY_TEXT, DARK_ORANGE, DARK_GREEN = "0.4", "#b35806", "#1e7b1e"


def check(value, guide, tol):
    """Return `value` after confirming that it matches the number the guide prints (to `tol`)."""
    if abs(float(value) - guide) > tol:
        raise AssertionError(f"recomputed {float(value)!r}, but the guide prints {guide}")
    return value


def boxed(ax, xy, text, color="k", offset=(0, 0), fontsize=9, ha="center", va="center"):
    """Like figbuild.label, but on a translucent white card so it stays readable over fills and lines."""
    ax.annotate(text, xy, textcoords="offset points", xytext=offset, color=color, fontsize=fontsize,
                ha=ha, va=va, bbox=WHITE_BOX, zorder=6)


def leader(ax, xy, xytext, text, color="k", fontsize=9, ha="left", va="center"):
    """A label at `xytext` joined to the data point `xy` by a thin line."""
    ax.annotate(text, xy, xytext=xytext, color=color, fontsize=fontsize, ha=ha, va=va, bbox=WHITE_BOX, zorder=6,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=0, shrinkB=1))


def rug(ax, xs, color, y=0.0, size=7, lw=1.0, zorder=5):
    """Vertical ticks along the horizontal line at height `y` (data coordinates)."""
    ax.plot(xs, np.full(len(xs), y), "|", color=color, ms=size, mew=lw, clip_on=False, zorder=zorder)


def log_indices(n_max, count=500):
    """Roughly `count` log-spaced 1-based sample sizes up to n_max: running averages stay sharp and the SVG small."""
    return np.unique(np.geomspace(1, n_max, count).round().astype(int))


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_geometric_memoryless():
    """Section 3: Geometric(0.1), memorylessness as a renormalised shift, NegBin(3, 0.1) against a Poisson of equal mean."""
    p = 0.1
    k = np.arange(1, 41)
    pmf = stats.geom.pmf(k, p)
    check(pmf[4], 0.0656, 5e-5)                                 # P(X = 5) = 0.9^4 x 0.1
    tail10 = check(stats.geom.sf(10, p), 0.349, 5e-4)           # 0.9^10
    check(1 - tail10, 0.651, 5e-4)
    mean, sd = check(stats.geom.mean(p), 10, 1e-12), check(stats.geom.std(p), 9.49, 5e-3)
    check(np.log(0.5) / np.log(0.9), 6.6, 0.05)                 # the median equation
    median = check(stats.geom.ppf(0.5, p), 7, 1e-12)
    surv5 = check(stats.geom.sf(5, p), 0.9 ** 5, 1e-12)
    cond = check(stats.geom.sf(8, p) / surv5, 0.729, 1e-12)     # P(X > 8 | X > 5)
    check(cond, stats.geom.sf(3, p), 1e-12)                     # = P(X > 3)
    r = 3
    nb = stats.nbinom(r, p)
    check(nb.mean(), 27, 1e-12); check(nb.var(), 270, 1e-9); check(nb.std(), 16.4, 0.05)
    check(nb.pmf(0), 0.001, 1e-12)
    cdf10 = check(nb.cdf(10), 0.134, 5e-4)
    po = stats.poisson(nb.mean())
    assert np.argmax(nb.pmf(np.arange(60))) in (17, 18)         # the mode sits well below the mean of 27

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.0))

    # (a) the PMF, its mean and median, and the tail beyond 10
    ax1.bar(k[k <= 10], pmf[k <= 10], width=0.8, color="C0", alpha=0.75)
    ax1.bar(k[k > 10], pmf[k > 10], width=0.8, color="C0", alpha=0.25, hatch="////", ec="C0", lw=0)
    ax1.bar(5, pmf[4], width=0.8, color="C3")
    ax1.axvline(median, color="k", ls="--", lw=1.3)
    ax1.axvline(mean, color="k", ls="-", lw=1.3)
    leader(ax1, (5, pmf[4]), (15, 0.092), f"$P(X = 5) = 0.9^4 \\times 0.1$\n$= {pmf[4]:.4f}$", color="C3")
    boxed(ax1, (40, 0.123), "dashed: median, visitor 7\n($0.9^k = 0.5$ at $k \\approx 6.6$)", ha="right", va="top", fontsize=8.5)
    boxed(ax1, (40, 0.068), f"solid: mean $1/p = {mean:g}$,\nsd ${sd:.2f}$", ha="right", va="top", fontsize=8.5)
    boxed(ax1, (40, 0.036), f"hatched: $P(X > 10)$\n$= 0.9^{{10}} = {tail10:.3f}$", ha="right", va="top", fontsize=8.5,
          color="C0")
    ax1.set_xlim(0, 41); ax1.set_ylim(0, 0.126)
    ax1.grid(axis="x", visible=False)
    ax1.set_title("(a) Geometric(0.1): the visitor\nwho makes the first purchase")
    ax1.set_xlabel("visitor number $k$"); ax1.set_ylabel("$P(X = k)$")

    # (b) five failures seen: divide the remaining tail by P(X > 5) and it is the original PMF, moved along by 5
    kk = np.arange(6, 31)
    uncond = stats.geom.pmf(kk, p)
    cond_pmf = uncond / surv5
    check(np.abs(cond_pmf - stats.geom.pmf(kk - 5, p)).max(), 0.0, 1e-12)
    ax2.bar(kk, cond_pmf, width=0.8, color="C1", alpha=0.6, label="$P(X = k \\mid X > 5)$")
    ax2.bar(kk, uncond, width=0.42, color=GREY, alpha=0.9, label=f"before dividing by $P(X > 5) = {surv5:.2f}$")
    ax2.plot(kk, stats.geom.pmf(kk - 5, p), "o", mfc="none", mec="C0", mew=1.2, ms=4.5, zorder=5,
             label="PMF of (a) moved right by 5")
    ax2.plot(np.arange(1, 6), np.full(5, 0.0), "x", color="k", ms=6, mew=1.3, clip_on=False, zorder=6,
             label="visitors 1 to 5: no purchase")
    boxed(ax2, (30.5, 0.054), f"$P(X > 8 \\mid X > 5) = 0.9^3$\n$= {cond:.3f} = P(X > 3)$", ha="right", va="center", fontsize=9)
    ax2.set_xlim(0, 31); ax2.set_ylim(0, 0.172)
    ax2.legend(loc="upper right", fontsize=8.5, handlelength=1.2, borderaxespad=0.3)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("(b) Memoryless: after five non-buyers\nthe rest is a fresh Geometric(0.1)")
    ax2.set_xlabel("visitor number $k$"); ax2.set_ylabel("conditional probability")

    # (c) over-dispersion: the same mean as a Poisson, three times the standard deviation
    y = np.arange(0, 91)
    ax3.bar(y, nb.pmf(y), width=0.85, color="C1", alpha=0.7)
    ax3.bar(y[y <= 10], nb.pmf(y[y <= 10]), width=0.85, color="C1", hatch="////", ec="k", lw=0)
    ys = np.arange(10, 46)
    ax3.vlines(ys, 0, po.pmf(ys), color="C0", lw=1.1)
    ax3.plot(ys, po.pmf(ys), "o", color="C0", ms=2.8)
    ax3.axvline(27, color="k", ls="--", lw=1.1)
    handles = [Patch(fc="C1", alpha=0.7, label=f"NegBin(3, 0.1): variance {nb.var():.0f},\nsd {nb.std():.1f}"),
               Line2D([], [], color="C0", marker="o", ms=2.8, lw=1.1, label=f"Poisson(27): variance 27,\nsd {po.std():.1f}"),
               Line2D([], [], color="k", ls="--", lw=1.1, label="shared mean 27"),
               Patch(fc="C1", hatch="////", ec="k", lw=0, label=f"$P(Y \\leq 10) = {cdf10:.3f}$")]
    ax3.set_xlim(-1, 90); ax3.set_ylim(0, 0.135)
    ax3.legend(handles=handles, loc="upper right", fontsize=8.5, handlelength=1.2, borderaxespad=0.3)
    ax3.grid(axis="x", visible=False)
    ax3.set_title("(c) Same mean 27, three times the\nspread: over-dispersed against Poisson")
    ax3.set_xlabel("non-buyers before the 3rd purchase"); ax3.set_ylabel("probability")
    fig.tight_layout()
    return fig


def fig_poisson_process():
    """Section 4: one simulated help-desk shift gives the Poisson counts of (b) and the Exponential gaps of (c)."""
    lam = 3.0
    rng = np.random.default_rng(7)
    times = np.cumsum(rng.exponential(1 / lam, 60))
    times = times[times < 8]
    gaps = np.diff(np.r_[0.0, times]) * 60                      # minutes
    hourly = np.histogram(times, np.arange(9))[0]
    assert len(times) == 24 and hourly.sum() == 24
    i_long = int(np.argmax(gaps))

    po = stats.poisson(lam)
    bi = stats.binom(180, 1 / 60)
    for j, guide in enumerate([0.0498, 0.1494, 0.2240, 0.2240]):
        check(po.pmf(j), guide, 5e-5)
    tail4 = check(po.sf(3), 0.3528, 5e-5)
    check(po.std(), 1.73, 5e-3)
    check(np.sqrt(8 * lam), 4.9, 0.05)                          # the 8-hour shift
    check(bi.pmf(0), 0.0485, 5e-5); check(bi.pmf(2), 0.2247, 5e-5)
    ks = np.arange(0, 181)
    maxdiff = check(np.abs(bi.pmf(ks) - po.pmf(ks)).max(), 0.0019, 5e-5)
    p30 = check(np.exp(-1.5), 0.223, 5e-4)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9))

    # (a) the counting process with one row of counts per hour
    ax1.step(np.r_[0.0, times, 8.0], np.r_[0, np.arange(1, len(times) + 1), len(times)], where="post", color="C0", lw=1.7)
    rug(ax1, times, "C0", size=6)
    ax1.vlines(np.arange(1, 8), 0, 29.7, color=GREY, lw=0.6, zorder=0)    # stop below the header at y = 30.8
    for h, c in enumerate(hourly):
        ax1.text(h + 0.5, 28.2, str(c), ha="center", va="center", fontsize=9.5)
    ax1.text(0.08, 30.8, "calls in each hour (seed 7):", ha="left", va="center", fontsize=8.5)
    t0, t1 = times[i_long - 1], times[i_long]
    level = i_long
    ax1.annotate("", xy=(t1, level - 1.2), xytext=(t0, level - 1.2), zorder=5,
                 arrowprops=dict(arrowstyle="<->", color="C3", lw=1.2, shrinkA=0, shrinkB=0))
    leader(ax1, (t1 - 0.2, level - 1.2), (3.55, 4.2), f"longest gap {gaps[i_long]:.0f} min;\na gap above 30 min\nhas probability {p30:.3f}",
           color="C3", fontsize=8.5)
    ax1.set_xlim(0, 8); ax1.set_ylim(0, 32)
    ax1.set_xticks(range(9))
    ax1.grid(False)
    ax1.set_title("(a) One simulated shift: rate 3 per\nhour, 24 calls ($= 8 \\times 3$) in 8 hours")
    ax1.set_xlabel("time (hours)"); ax1.set_ylabel("calls so far $N(t)$")

    # (b) counts in one hour: Poisson bars and the Binomial markers it approximates
    k = np.arange(0, 11)
    ax2.bar(k[k < 4], po.pmf(k[k < 4]), width=0.8, color="C0", alpha=0.75, label="Poisson(3)")
    ax2.bar(k[k >= 4], po.pmf(k[k >= 4]), width=0.8, color="C3", alpha=0.5, hatch="////", ec="C3", lw=0,
            label=f"hatched: $P(N \\geq 4) = {tail4:.3f}$")
    ax2.plot(k, bi.pmf(k), "o", mfc="white", mec="k", mew=1.1, ms=5, zorder=5, label="Binomial(180, 1/60)")
    ax2.axvline(lam, color="k", ls="--", lw=1.1, label=f"mean = variance = 3, sd {po.std():.2f}")
    boxed(ax2, (10.45, 0.135), f"largest difference,\nBinomial–Poisson,\nover all $k$: {maxdiff:.4f}", ha="right", va="center", fontsize=8.5)
    ax2.set_xlim(-0.7, 10.7); ax2.set_ylim(0, 0.36)
    ax2.set_xticks(k)
    ax2.legend(loc="upper right", fontsize=8.5, handlelength=1.4, borderaxespad=0.3)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("(b) Calls in one hour: Poisson(3) and the\nBinomial it approximates")
    ax2.set_xlabel("calls in the hour $k$"); ax2.set_ylabel("$P(N = k)$")

    # (c) gaps between calls, in minutes: the 24 gaps of (a) under the exact density
    rate = lam / 60                                             # per minute
    t = np.linspace(0, 120, 500)
    dens = rate * np.exp(-rate * t)
    ax3.plot(t, dens, color="C0", lw=2, label="Exponential density,\nmean 20 min")
    tail = t >= 30
    ax3.fill_between(t[tail], dens[tail], color="C3", alpha=0.35, hatch="////", lw=0, label=f"hatched: $P(\\mathrm{{gap}} > 30\\ \\mathrm{{min}})$\n$= e^{{-1.5}} = {p30:.3f}$")
    ax3.axvline(20, color="k", ls="--", lw=1.1, label="mean $60/3 = 20$ min")
    rug(ax3, gaps, DARK_ORANGE, y=0.0009, size=8, lw=1.1)
    ax3.plot([], [], "|", color=DARK_ORANGE, ms=8, mew=1.1, label="the 24 gaps of (a)")
    ax3.set_xlim(0, 120); ax3.set_ylim(0, 0.056)
    ax3.legend(loc="upper right", fontsize=8.5, handlelength=1.4, borderaxespad=0.3)
    ax3.set_title("(c) Gaps between the same calls:\nExponential, rate 0.05 per minute")
    ax3.set_xlabel("gap between consecutive calls (minutes)"); ax3.set_ylabel("density (per minute)")
    fig.tight_layout()
    return fig


def fig_exponential_hazard():
    """Section 5: the bus wait as areas under the density; inverse-CDF sampling; constant against rising and falling hazards."""
    lam = 0.2
    T = stats.expon(scale=1 / lam)
    p5 = check(T.sf(5), 0.368, 5e-4)
    check(np.exp(-0.4), 0.6703, 5e-5); check(np.exp(-0.8), 0.4493, 5e-5)
    strip = check(T.cdf(4) - T.cdf(2), 0.221, 5e-4)
    med = check(T.ppf(0.5), 3.47, 5e-3)
    p90 = check(T.ppf(0.9), 11.5, 0.05)
    check(T.sf(10) / T.sf(5), p5, 1e-12)                        # memorylessness
    check(T.std(), 5, 1e-12); check(T.mean(), 5, 1e-12)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9))

    # (a) probabilities as areas
    t = np.linspace(0, 25, 500)
    f = T.pdf(t)
    ax1.fill_between(t, f, where=(t >= 2) & (t <= 4), color="C0", alpha=0.55, lw=0)
    ax1.fill_between(t, f, where=t >= 5, color="C3", alpha=0.3, hatch="////", lw=0)
    ax1.plot(t, f, color="C0", lw=2)
    ax1.axvline(med, color="k", ls="--", lw=1.2)
    ax1.axvline(5, color="k", ls="-", lw=1.2)
    ax1.axvline(p90, color="k", ls=":", lw=1.4)
    handles = [Line2D([], [], color="k", ls="--", lw=1.2, label=f"median $\\ln 2 / 0.2 = {med:.2f}$"),
               Line2D([], [], color="k", ls="-", lw=1.2, label="mean = sd = 5"),
               Line2D([], [], color="k", ls=":", lw=1.4, label=f"90th percentile {p90:.1f}"),
               Patch(fc="C0", alpha=0.55, label=f"$P(2 < T < 4) = {strip:.3f}$"),
               Patch(fc="C3", alpha=0.3, hatch="////", label=f"$P(T > 5) = e^{{-1}} = {p5:.3f}$,\nalso $P(T > 10 \\mid T > 5)$")]
    ax1.legend(handles=handles, loc="upper right", fontsize=8.5, handlelength=1.5, borderaxespad=0.3)
    ax1.set_xlim(0, 25); ax1.set_ylim(0, 0.21)
    ax1.set_title("(a) Waiting for a bus: Exponential,\n$\\lambda = 0.2$ per minute")
    ax1.set_xlabel("wait $t$ (minutes)"); ax1.set_ylabel("density (per minute)")

    # (b) the inverse-CDF method: uniform heights fall on the curve and drop to Exponential waits
    ax2.plot(t, T.cdf(t), color="C0", lw=2)
    for u, tu, txt, xt in [(0.5, med, f"$U = 0.5 \\to {med:.2f}$", 12.1), (1 - p5, 5.0, f"$U = {1 - p5:.3f} \\to 5$", 12.1),
                           (0.9, p90, f"$U = 0.9 \\to {p90:.1f}$", 12.1)]:
        ax2.plot([0, tu], [u, u], color=GREY, lw=1)
        ax2.annotate("", xy=(tu, 0), xytext=(tu, u), arrowprops=dict(arrowstyle="-|>", color=DARK_ORANGE, lw=1.1, shrinkA=0, shrinkB=0))
        ax2.plot([tu, 12], [u, u], color=GREY, lw=0.6, ls=":")  # ties the label, stacked right of the arrows, to its height
        boxed(ax2, (xt, u), txt, offset=(0, -4), ha="left", va="top", fontsize=9)
    us = np.sort(np.random.default_rng(8).random(8))
    ts = T.ppf(us)
    check(us[0], 0.319, 5e-4); check(ts[-1], 21.8, 0.05)
    ax2.plot(np.full(8, 0.25), us, "_", color="C3", ms=9, mew=1.5, clip_on=False, zorder=6)
    ax2.plot(ts, np.full(8, 0.012), "|", color="C3", ms=9, mew=1.5, clip_on=False, zorder=6)
    boxed(ax2, (25, 0.22), "red ticks: eight\nseeded $U$ (left)\nand their waits\n(bottom)", offset=(-4, 0), ha="right", va="center",
          color="C3", fontsize=8.5)
    ax2.set_xlim(0, 25); ax2.set_ylim(0, 1.02)
    ax2.set_title("(b) Inverse-CDF sampling:\n$T = F^{-1}(U) = -\\ln(1 - U)/0.2$")
    ax2.set_xlabel("wait $t$ (minutes)"); ax2.set_ylabel("$U = F(t)$")

    # (c) hazard: constant only for the Exponential; the two Weibulls are illustrations, not worked-example numbers
    age = np.linspace(0.05, 15, 400)
    wear = stats.weibull_min(2, scale=5)
    infant = stats.weibull_min(0.5, scale=5)
    hazard = lambda d, x: d.pdf(x) / d.sf(x)                    # noqa: E731
    check(hazard(wear, 5.0), 0.4, 1e-12)
    check(hazard(infant, 5.0), 0.1, 1e-12)
    check(hazard(T, 7.0), lam, 1e-12)
    ax3.plot(age, hazard(T, age), color="C0", lw=2.2)
    ax3.plot(age, hazard(wear, age), color="C3", lw=2, ls="--")
    ax3.plot(age, hazard(infant, age), color="C2", lw=2, ls=":")
    boxed(ax3, (15, 0.2), "Exponential: constant 0.2\n(electronics in useful life)", color="C0", offset=(-4, 5), ha="right", va="bottom", fontsize=8.5)
    boxed(ax3, (15, 0.40), "rising (Weibull, shape 2):\nwear-out, tyres, bearings", color="C3", offset=(-4, 0), ha="right", va="center", fontsize=8.5)
    boxed(ax3, (15, 0.135), "falling (shape 0.5):\ninfant mortality", color=DARK_GREEN, offset=(-4, 0), ha="right", va="center", fontsize=8.5)
    ax3.set_xlim(0, 15); ax3.set_ylim(0, 0.9)
    ax3.set_title("(c) Hazard $h(t) = f(t)/P(T > t)$:\nflat only for the Exponential")
    ax3.set_xlabel("age $t$ (arbitrary time units)"); ax3.set_ylabel("hazard (per time unit)")
    fig.tight_layout()
    return fig


def fig_normal_lognormal():
    """Section 6: heights N(165, 7^2) with the 68-95-99.7 rule and one tail; log-normal income with mode < median < mean."""
    mu, sigma = 165, 7
    H = stats.norm(mu, sigma)
    for k, guide in [(1, 0.6827), (2, 0.9545), (3, 0.9973)]:
        check(H.cdf(mu + k * sigma) - H.cdf(mu - k * sigma), guide, 5e-5)
    z = check((180 - mu) / sigma, 2.14, 5e-3)
    p180 = check(H.sf(180), 0.0161, 5e-5)
    check(1 / p180, 62, 0.5)
    p99 = check(H.ppf(0.99), 181.3, 0.05)
    lo95, hi95 = check(mu - 1.96 * sigma, 151.3, 0.05), check(mu + 1.96 * sigma, 178.7, 0.05)

    s = 0.5
    Y = stats.lognorm(s, scale=50_000)
    check(Y.median(), 50_000, 1e-6)
    mean_y = check(Y.mean(), 56_657, 0.5)
    mode_y = check(50_000 * np.exp(-s ** 2), 38_940, 0.5)
    p100 = check(Y.sf(100_000), 0.083, 5e-4)
    check(np.log(2) / s, 1.386, 5e-4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.1))

    # (a) nested bands under the curve, their ranges above it; the 95% chord; the tail beyond 180
    x = np.linspace(137, 195, 800)
    f = H.pdf(x)
    for k, a in [(3, 0.14), (2, 0.2), (1, 0.28)]:
        ax1.fill_between(x, f, where=np.abs(x - mu) <= k * sigma, color="C0", alpha=a, lw=0)
    ax1.fill_between(x, f, where=x >= 180, color="C3", alpha=0.55, hatch="////", lw=0, zorder=3)
    ax1.plot(x, f, color="C0", lw=2)
    for k, y, pct in [(1, 0.0605, 68.3), (2, 0.0690, 95.4), (3, 0.0775, 99.7)]:
        a, b = mu - k * sigma, mu + k * sigma
        ax1.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="<->", color="k", lw=1, shrinkA=0, shrinkB=0))
        label(ax1, (mu, y), f"{pct}%: [{a}, {b}]", offset=(0, 2), ha="center", fontsize=8.5)
    yc = H.pdf(lo95)
    ax1.plot([lo95, hi95], [yc, yc], color="k", lw=1.2, zorder=4)
    boxed(ax1, (mu, yc), f"95%: $165 \\pm 1.96 \\times 7 = [{lo95:.1f}, {hi95:.1f}]$", offset=(0, -14), fontsize=8)
    ax1.plot([p99, p99], [0, 0.0225], color="k", ls=":", lw=1.4, zorder=5)
    leader(ax1, (p99, 0.0225), (179.5, 0.031), f"99th percentile:\n$165 + 2.326 \\times 7$\n$= {p99:.1f}$", fontsize=8.5, va="bottom")
    leader(ax1, (181, H.pdf(181) * 0.5), (184, 0.0095), f"$z = (180 - 165)/7$\n$= {z:.2f}$, $P(X > 180)$\n$= {p180:.4f}$, 1 in 62", color="C3", fontsize=8.5, va="bottom")
    ax1.set_xlim(137, 203); ax1.set_ylim(0, 0.087)
    ax1.set_title("(a) Heights $\\mathcal{N}(165, 7^2)$: the 68–95–99.7 rule and one tail")
    ax1.set_xlabel("height (cm)"); ax1.set_ylabel("density (per cm)")

    # (b) the log-normal: three different "centres", in thousands of USD
    xk = np.linspace(0.3, 200, 800)
    g = Y.pdf(xk * 1000) * 1000
    ax2.fill_between(xk, g, where=xk >= 100, color="C3", alpha=0.35, hatch="////", lw=0)
    ax2.plot(xk, g, color="C0", lw=2)
    for v, col, ls, ylab, txt, ha, dx in [
        (mode_y / 1000, "C2", ":", 0.0252, f"mode $e^{{\\mu - \\sigma^2}}$ = {mode_y:,.0f}", "left", 3),
        (50, "k", "--", 0.0232, "median $e^\\mu$ = 50,000", "left", 3),
        (mean_y / 1000, "C3", "-", 0.0205, f"mean $e^{{\\mu + \\sigma^2/2}}$ = {mean_y:,.0f}\n(not $e^\\mu$)", "left", 3)]:
        ax2.plot([v, v], [0, ylab], color=col, ls=ls, lw=1.5, zorder=4)
        col_txt = DARK_GREEN if col == "C2" else col
        boxed(ax2, (v, ylab), txt, color=col_txt, offset=(dx, 0), ha=ha, va="top", fontsize=9)
    leader(ax2, (112, Y.pdf(112_000) * 500), (125, 0.0075), f"$P(Y > 100{{,}}000) = 1 - \\Phi(1.386)$\n$= {p100:.3f}$", color="C3", fontsize=9, va="bottom")
    ax2.set_xlim(0, 200); ax2.set_ylim(0, 0.0275)
    ax2.set_title("(b) Log-normal income, $\\ln Y \\sim \\mathcal{N}(\\ln 50{,}000,\\ 0.5^2)$:\nmode < median < mean, the signature of right skew")
    ax2.set_xlabel("income (thousand USD)"); ax2.set_ylabel("density (per thousand USD)")
    fig.tight_layout()
    return fig


def fig_tails():
    """Section 8: survival functions on log-log axes with Markov's bound; Bernoulli skewness against p."""
    alpha = 1.5
    pareto = stats.pareto(alpha)                                # x_min = 1
    check(pareto.sf(10), 0.032, 5e-4); check(pareto.sf(100), 0.001, 1e-6)
    ex = stats.expon()
    p5 = check(ex.sf(5), 0.0067, 5e-5)
    markov = check(ex.mean() / 5, 0.2, 1e-12)
    ratio = check(markov / p5, 30, 0.5)
    sk_exp, kurt_exp = ex.stats(moments="sk")
    check(sk_exp, 2, 1e-9); check(kurt_exp, 6, 1e-9)
    check(1 / (2 * stats.norm.sf(4)), 16_000, 300)              # "once in 16,000"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) log-log survival
    x = np.geomspace(1, 1000, 500)
    ax1.loglog(x, pareto.sf(x), color="C3", lw=2.2, label="Pareto, $\\alpha = 1.5$: $x^{-1.5}$")
    ax1.loglog(x, 1 / x, color=GREY, ls=":", lw=1.8, label="Markov for Exponential(1): $1/x$")
    ax1.loglog(x, ex.sf(x), color="C0", lw=2, ls="--", label="Exponential(1): $e^{-x}$")
    ax1.loglog(x, 2 * stats.norm.sf(x), color="C2", lw=2, ls="-.", label="half-Normal $|Z|$: $2(1 - \\Phi(x))$")
    ax1.plot([10, 100], pareto.sf([10, 100]), "o", color="C3", ms=6, zorder=5)
    leader(ax1, (10, pareto.sf(10)), (13, 0.2), f"$10^{{-1.5}} = {pareto.sf(10):.3f}$", color="C3", va="bottom")
    label(ax1, (100, pareto.sf(100)), f"$100^{{-1.5}} = {pareto.sf(100):.3f}$", color="C3", offset=(-4, -5), ha="right", va="top", fontsize=9)
    ax1.plot([5, 5], [p5, markov], color="k", lw=1.1, marker="_", ms=7, zorder=5)
    leader(ax1, (5, 0.03), (11, 2.2e-5), f"at $x = 5$: Markov $\\leq {markov:g}$,\ntruth $e^{{-5}} = {p5:.4f}$,\n{ratio:.0f} times smaller", fontsize=8.5, va="bottom")
    ax1.set_xlim(1, 1000); ax1.set_ylim(1e-7, 1.5)
    ax1.legend(loc="lower right", fontsize=8.5, handlelength=2.2, borderaxespad=0.3)
    ax1.set_title("(a) Survival on log-log axes: a power law is a\nstraight line, light tails bend down and vanish")
    ax1.set_xlabel("$x$, in units of $x_{\\min}$ (log scale)"); ax1.set_ylabel("$P(X > x)$ (log scale)")

    # (b) the skewness of a Bernoulli as p shrinks
    skew = lambda p: (1 - 2 * p) / np.sqrt(p * (1 - p))         # noqa: E731
    check(skew(0.5), 0, 1e-12); check(skew(0.1), 2.67, 5e-3); check(skew(0.01), 9.85, 5e-3)
    sk_ln = stats.lognorm(1.0).stats(moments="s")
    check(sk_ln, 6, 0.3)                                        # pitfall 2: "skewness about 6"
    p = np.geomspace(0.003, 0.5, 400)
    ax2.semilogx(p, skew(p), color="C3", lw=2.2)
    ax2.axhline(sk_exp, color="C0", ls="--", lw=1.4)
    ax2.axhline(sk_ln, color=GREY, ls=":", lw=1.8)
    boxed(ax2, (0.0032, sk_exp), "Exponential(1): $\\gamma_1 = 2$", color="C0", offset=(0, 0), ha="left", va="center")
    boxed(ax2, (0.0032, sk_ln), f"log-normal $\\sigma = 1$: {sk_ln:.2f}", color=GREY_TEXT, offset=(0, 0), ha="left", va="center")
    for pv, txt, off, ha in [(0.5, "$p = 0.5$: 0, symmetric", (-9, -9), "right"), (0.1, f"$p = 0.1$: {skew(0.1):.2f}", (6, 5), "left"),
                             (0.01, f"$p = 0.01$: {skew(0.01):.2f}", (7, 4), "left")]:
        ax2.plot(pv, skew(pv), "o", color="C3", ms=6.5, zorder=5)
        label(ax2, (pv, skew(pv)), txt, color="C3", offset=off, ha=ha, fontsize=9.5)
    ax2.set_xlim(0.003, 0.56); ax2.set_ylim(-0.5, 12)
    ax2.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3, 0.5]); ax2.set_xticklabels(["0.003", "0.01", "0.03", "0.1", "0.3", "0.5"])
    ax2.set_title("(b) Skewness of a Bernoulli($p$): the number\nthat sets the speed of the CLT (Section 10)")
    ax2.set_xlabel("success probability $p$ (log scale)"); ax2.set_ylabel("skewness $\\gamma_1$")
    fig.tight_layout()
    return fig


def fig_lln_dilution():
    """Section 9: the proportion converges, the excess count drifts, and Cauchy averages never settle."""
    check(0.25 / (0.05 * 0.01 ** 2), 50_000, 1e-6)              # Chebyshev's n
    n_clt = check((1.96 * 0.5 / 0.01) ** 2, 9_604, 1e-6)        # "about 9,600"
    check((60 + 450) / 1000, 0.51, 1e-12)
    check(np.sqrt(100) / 2, 5, 1e-12); check(np.sqrt(10_000) / 2, 50, 1e-12)

    n_all = 50_000
    flips = np.random.default_rng(8).random((4, n_all)) < 0.5
    heads = np.cumsum(flips, axis=1)
    idx = log_indices(n_all)
    idx = idx[idx >= 10]
    n = idx.astype(float)
    prop = heads[:, idx - 1] / n
    excess = heads[:, idx - 1] - n / 2
    nn = np.geomspace(100, n_all, 200)
    path = (60 + 0.5 * (nn - 100)) / nn                         # worked example: 60 of the first 100, then exactly half
    check(path[0], 0.60, 1e-12)
    check((60 + 0.5 * 900) / 1000, 0.51, 1e-12)
    clt = 1.96 * 0.5 / np.sqrt(n)
    cheb = 0.5 / np.sqrt(0.05 * n)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.7))

    # (a) the proportion, its worked-example path and the two 95% bands
    for row in prop:
        ax1.semilogx(n, row, color="C0", lw=0.8, alpha=0.75)
    ax1.semilogx(n, 0.5 + clt, color=DARK_ORANGE, ls="--", lw=1.3); ax1.semilogx(n, 0.5 - clt, color=DARK_ORANGE, ls="--", lw=1.3)
    ax1.semilogx(n, 0.5 + cheb, color="k", ls=":", lw=1.3); ax1.semilogx(n, 0.5 - cheb, color="k", ls=":", lw=1.3)
    ax1.semilogx(nn, path, color="C3", lw=2.4, zorder=5)
    ax1.plot([100, 1000], [0.60, 0.51], "o", color="C3", ms=6, zorder=6)
    boxed(ax1, (100, 0.60), "60/100", color="C3", offset=(8, 9), ha="center", fontsize=9)
    leader(ax1, (1000, 0.51), (1300, 0.625), "510/1000", color="C3", ha="center")
    ax1.plot([n_clt, n_all], [0.51, 0.51], "o", mfc="white", mec="k", ms=5, zorder=6, clip_on=False)
    handles = [Line2D([], [], color="C0", lw=1, label="four simulated runs (seed 8)"),
               Line2D([], [], color="C3", lw=2.4, label="60 heads in 100, then exactly half"),
               Line2D([], [], color=DARK_ORANGE, ls="--", lw=1.3, label="CLT 95% band: $\\pm 0.01$ at $n = 9{,}604$"),
               Line2D([], [], color="k", ls=":", lw=1.3, label="Chebyshev 95% band: $\\pm 0.01$\nat $n = 50{,}000$")]
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.5, handlelength=1.5, frameon=False)
    ax1.set_xlim(10, n_all); ax1.set_ylim(0.25, 0.75)
    ax1.set_title("(a) The proportion converges: early\nluck is diluted, not repaid")
    ax1.set_xlabel("flips $n$ (log scale)"); ax1.set_ylabel("proportion of heads")

    # (b) the excess count does not converge
    for row in excess:
        ax2.semilogx(n, row, color="C0", lw=0.8, alpha=0.75)
    env = np.sqrt(n) / 2
    ax2.semilogx(n, env, color=DARK_ORANGE, ls="--", lw=1.3); ax2.semilogx(n, -env, color=DARK_ORANGE, ls="--", lw=1.3)
    ax2.semilogx(n, 2 * env, color="k", ls=":", lw=1.3); ax2.semilogx(n, -2 * env, color="k", ls=":", lw=1.3)
    ax2.semilogx([100, n_all], [10, 10], color="C3", lw=2.4, zorder=5)
    boxed(ax2, (170, 10), "+10", color="C3", offset=(0, 11), fontsize=9)
    handles = [Line2D([], [], color="C0", lw=1, label="the same four runs"),
               Line2D([], [], color="C3", lw=2.4, label="the 10 extra heads, never paid back"),
               Line2D([], [], color=DARK_ORANGE, ls="--", lw=1.3, label="$\\pm\\sqrt{n}/2$: 5 at 100, 50 at 10,000"),
               Line2D([], [], color="k", ls=":", lw=1.3, label="$\\pm 2$ standard deviations")]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.5, handlelength=1.5, frameon=False)
    ax2.set_xlim(10, n_all); ax2.set_ylim(-250, 250)
    ax2.set_title("(b) The count of extra heads drifts\nlike $\\sqrt{n}$: nothing pulls it back")
    ax2.set_xlabel("flips $n$ (log scale)"); ax2.set_ylabel("heads minus $n/2$")

    # (c) Cauchy running averages; the y-limits clip the largest jump, named in the panel
    idc = log_indices(10_000, 700)
    idc = idc[idc >= 10]
    runs = {}
    for seed, col, ls in [(10, "C0", "-"), (11, "C1", "--"), (12, "C2", "-.")]:
        avg = np.cumsum(np.random.default_rng(seed).standard_cauchy(10_000)) / np.arange(1, 10_001)
        runs[seed] = avg
        ax3.semilogx(idc, avg[idc - 1], color=col, ls=ls, lw=1.2, label=f"seed {seed}")
    ax3.axhline(0, color=GREY, lw=1.2)
    tops = {seed: (runs[seed].max(), int(np.argmax(runs[seed])) + 1) for seed in (10, 11)}
    assert runs[12].max() < 12
    boxed(ax3, (316, 11.7), f"off scale: seed 10 reaches {tops[10][0]:.0f} at $n = {tops[10][1]}$,\nseed 11 reaches {tops[11][0]:.0f} at $n = {tops[11][1]}$",
          ha="center", va="top", fontsize=8)
    boxed(ax3, (8_000, -5.5), "median 0, but no mean: no\nvalue for the average to settle on", ha="right", va="bottom", fontsize=8.5, color=GREY_TEXT)
    ax3.set_xlim(10, 10_000); ax3.set_ylim(-6, 12)
    ax3.legend(loc="upper left", bbox_to_anchor=(0, 0.8), fontsize=8.5, handlelength=2, borderaxespad=0.3)
    ax3.set_title("(c) Cauchy: no mean, so the running\naverage never settles")
    ax3.set_xlabel("$n$ (log scale)"); ax3.set_ylabel("running average")
    fig.tight_layout()
    return fig


def fig_clt_speed():
    """Section 10: exact standardised means of Exponentials; the comb of 30 Bernoulli(0.01); the working rule for n."""
    # the guide's other worked numbers: same formula as the figure's lines
    check(1.708 / np.sqrt(30), 0.312, 5e-4)
    check(0.5 / 0.312, 1.60, 5e-3)
    check(stats.norm.sf(1.60), 0.055, 5e-4)                     # from the guide's rounded z; unrounded it is 0.0544
    rho = integrate.quad(lambda x: abs(x - 1) ** 3 * np.exp(-x), 0, np.inf)[0]
    check(rho, 2.415, 5e-4)
    check(0.4748 * rho / np.sqrt(30), 0.21, 5e-3)               # the Berry-Esseen bound

    def density(n, z):
        return stats.gamma.pdf(1 + z / np.sqrt(n), a=n, scale=1 / n) / np.sqrt(n)    # Z = sqrt(n) (mean - 1)

    def max_gap(n):
        z = np.linspace(-12, 40, 200_001)
        z = np.sort(np.r_[z, -np.sqrt(n)])                      # includes the left end of the support
        return np.abs(stats.gamma.cdf(1 + z / np.sqrt(n), a=n, scale=1 / n) - stats.norm.cdf(z)).max()

    sk_exp = 2.0
    p01 = 0.01
    sk_b = check((1 - 2 * p01) / np.sqrt(p01 * (1 - p01)), 9.85, 5e-3)
    sk_ln = float(stats.lognorm(1.0).stats(moments="s"))
    check(sk_exp / np.sqrt(30), 0.37, 5e-3); check(sk_b / np.sqrt(30), 1.80, 5e-3)
    check(sk_exp / np.sqrt(400), 0.1, 1e-12)
    n_b01 = check((sk_b / 0.1) ** 2, 9_700, 5)
    n_b03 = check((sk_b / 0.3) ** 2, 1_000, 80)                 # "roughly 1,000" in self-assessment 7

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.0))

    # (a) exact densities: the mean of n Exponential(1) is Gamma(n, rate n)
    z = np.linspace(-4, 4, 801)
    styles = {1: ("C3", ":"), 5: ("C1", "--"), 30: ("C0", "-")}
    gaps_txt = "skewness, CDF gap"
    for n, (col, ls) in styles.items():
        ax1.plot(z, density(n, z), color=col, ls=ls, lw=2, label=f"$n = {n}$")
        gaps_txt += f"\n$n = {n}$: {sk_exp / np.sqrt(n):.2f}, {max_gap(n):.3f}"
    ax1.plot(z, stats.norm.pdf(z), color="k", lw=1.2, label="$\\mathcal{N}(0, 1)$")
    ax1.set_xlim(-4, 4); ax1.set_ylim(0, 1.08)
    ax1.legend(loc="upper right", fontsize=8.5, handlelength=1.8, borderaxespad=0.3, labelspacing=0.4)
    boxed(ax1, (4, 0.72), gaps_txt, offset=(-4, 0), ha="right", va="top", fontsize=8.5)
    ax1.set_title("(a) Standardised mean of $n$ Exponentials\n(exact), against the bell")
    ax1.set_xlabel("standardised mean $z$"); ax1.set_ylabel("density")

    # (b) the mean of 30 Bernoulli(0.01) as a count: a PMF against the CLT's bell, per unit count
    B = stats.binom(30, p01)
    k = np.arange(0, 5)
    check(B.pmf(0), 0.740, 5e-4); check(B.pmf(1), 0.224, 5e-4)
    mu, sd = 30 * p01, np.sqrt(30 * p01 * (1 - p01))
    N = stats.norm(mu, sd)
    below = N.cdf(0)
    x = np.linspace(-2, 4, 500)
    ax2.fill_between(x, N.pdf(x), where=x <= 0, color="C0", alpha=0.3, hatch="////", lw=0, zorder=3)
    ax2.bar(k, B.pmf(k), width=0.42, color="C3", alpha=0.85, zorder=2, label="Binomial(30, 0.01)")
    ax2.plot(x, N.pdf(x), color="C0", lw=2, zorder=4, label=f"CLT bell $\\mathcal{{N}}({mu:g}, {sd:.3f}^2)$")
    for j, v in zip(k[:3], B.pmf(k[:3])):
        off, ha = {0: ((0, 5), "center"), 1: ((7, 3), "left"), 2: ((0, 3), "center")}[int(j)]   # k = 1: clear of the bell
        label(ax2, (j, v), f"{v:.3f}", offset=off, ha=ha, color="C3", fontsize=9)
    ax2.annotate(f"{below:.0%} of the\nbell lies\nbelow 0, an\nimpossible\ncount", (-0.45, 0.17), xytext=(-2.65, 0.46), color="C0", fontsize=8.5, va="bottom",
                 arrowprops=dict(arrowstyle="-", color="C0", lw=0.8, shrinkA=0, shrinkB=1))
    boxed(ax2, (4.0, 0.6), f"skewness of the\nmean: $9.85/\\sqrt{{30}}$\n$= {sk_b / np.sqrt(30):.2f}$", offset=(-4, 0), ha="right", va="center", fontsize=8.5)
    ax2.set_xlim(-2.7, 4.05); ax2.set_ylim(0, 1.08)
    ax2.set_xticks(range(-2, 5))
    ax2.legend(loc="upper right", fontsize=8.5, handlelength=1.5, borderaxespad=0.3)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("(b) Mean of 30 Bernoulli(0.01): a comb\nof rare-event counts, not a bell")
    ax2.set_xlabel("number of rare events among 30"); ax2.set_ylabel("probability (bell: per unit count)")

    # (c) the working rule, read off where each line crosses the threshold
    n = np.geomspace(1, 1e5, 200)
    srcs = [(sk_b, "C3", "-", "Bernoulli(0.01), $\\gamma_1 = 9.85$"),
            (sk_ln, "C2", "-.", f"log-normal $\\sigma = 1$, $\\gamma_1 = {sk_ln:.2f}$"),
            (sk_exp, "C1", "--", "Exponential, $\\gamma_1 = 2$")]
    for g, col, ls, lab in srcs:
        ax3.loglog(n, g / np.sqrt(n), color=col, ls=ls, lw=2, label=lab)
    ax3.axhline(0.3, color="k", ls="--", lw=1)
    ax3.axhline(0.1, color="k", ls=":", lw=1)
    ax3.axvline(30, color=GREY, lw=1.1)
    boxed(ax3, (34, 5), "$n = 30$: the folk rule", color=GREY_TEXT, ha="left", va="center", fontsize=8.5)
    for g, col, off, ha, va in [(sk_exp, DARK_ORANGE, (3, 4), "left", "bottom"), (sk_ln, DARK_GREEN, (-3, -4), "right", "top"),
                                (sk_b, "C3", (3, 4), "left", "bottom")]:
        cross = (g / 0.3) ** 2
        ax3.plot(cross, 0.3, "o", color=col, ms=5.5, zorder=5)
        label(ax3, (cross, 0.3), f"{cross:,.0f}", color=col, offset=off, ha=ha, va=va, fontsize=9)
    for g, col, nv in [(sk_exp, DARK_ORANGE, 400.0), (sk_b, "C3", n_b01)]:
        ax3.plot(nv, 0.1, "s", color=col, ms=5, zorder=5)
        label(ax3, (nv, 0.1), f"{round(nv, -2):,.0f}", color=col, offset=(3, 3), ha="left", va="bottom", fontsize=9)
    boxed(ax3, (1.3, 0.3), "skew 0.3", ha="left", va="center", fontsize=8.5)
    boxed(ax3, (1.3, 0.1), "skew 0.1", ha="left", va="center", fontsize=8.5)
    ax3.set_xlim(1, 1e5); ax3.set_ylim(0.05, 30)
    ax3.legend(loc="upper right", fontsize=8, handlelength=2, borderaxespad=0.3)
    ax3.set_title("(c) The working rule $n \\gtrsim (\\gamma_1/0.3)^2$:\nwhen the mean looks symmetric")
    ax3.set_xlabel("$n$ (log scale)"); ax3.set_ylabel("skewness of the sample mean (log scale)")
    fig.tight_layout()
    return fig


def fig_monte_carlo():
    """Section 11: an integral as an average height; error against n; why antithetic pairs work."""
    g = lambda x: np.exp(-x ** 2)                               # noqa: E731
    I = check(integrate.quad(g, 0, 1)[0], 0.74682, 5e-6)
    E2 = check(integrate.quad(lambda x: g(x) ** 2, 0, 1)[0], 0.59814, 5e-6)
    var = check(E2 - I ** 2, 0.0404, 5e-5)
    sg = check(np.sqrt(var), 0.201, 5e-4)
    cov = check(integrate.quad(lambda x: g(x) * g(1 - x), 0, 1)[0] - I ** 2, -0.0388, 5e-5)
    var_pair = check(0.5 * (var + cov), 0.0008, 5e-5)
    gain = check(var / 2 / var_pair, 25, 0.5)
    check(sg / np.sqrt(10_000), 0.0020, 5e-5)
    sa = np.sqrt(2 * var_pair)                                  # per evaluation: n/2 pairs for n evaluations
    corr = cov / var

    u = np.random.default_rng(8).random(10_000)
    h = g(u)
    est12 = h[:12].mean()
    runs = {n: h[:n].mean() for n in (100, 1_000, 10_000)}
    se_run = h.std(ddof=1) / 100
    check(se_run, 0.0020, 5e-5)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.0))

    # (a) stems at twelve uniform points; the mean height estimates the area
    x = np.linspace(0, 1, 300)
    ax1.fill_between(x, g(x), color="C0", alpha=0.18, lw=0)
    ax1.plot(x, g(x), color="C0", lw=2)
    ax1.vlines(u[:12], 0, h[:12], color=DARK_ORANGE, lw=1.2)
    ax1.plot(u[:12], h[:12], "o", color=DARK_ORANGE, ms=4, zorder=5)
    ax1.axhline(est12, color=DARK_ORANGE, ls="--", lw=1.3)
    ax1.axhline(I, color="k", ls="-", lw=1.1)
    handles = [Line2D([], [], color="k", lw=1.1, label=f"shaded area as a height: {I:.3f}"),
               Line2D([], [], color=DARK_ORANGE, ls="--", lw=1.3, label=f"mean of the 12 stems: {est12:.3f}")]
    ax1.legend(handles=handles, loc="upper right", fontsize=8.5, handlelength=1.6, borderaxespad=0.3)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.3)
    ax1.set_title("(a) The integral is the average height\nof $g$ at random points")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$g(x) = e^{-x^2}$")

    # (b) closed-form standard errors and three seeded runs
    n = np.geomspace(1e2, 1e8, 100)
    ax2.loglog(n, sg / np.sqrt(n), color="C0", lw=2.2, label=f"$\\sigma_g/\\sqrt{{n}} = {sg:.3f}/\\sqrt{{n}}$")
    ax2.loglog(n, sa / np.sqrt(n), color="C1", lw=2, ls="--", label=f"antithetic: {sa:.3f}$/\\sqrt{{n}}$")
    ax2.loglog(n, sg / n, color=GREY, lw=1.6, ls=":", label=f"what pitfall 7 expects: {sg:.3f}$/n$")
    pts = [1e4, 1e6, 1e8]
    ax2.plot(pts, sg / np.sqrt(pts), "o", color="C0", ms=6, zorder=5)
    for pv, txt in zip(pts, ["0.002", "0.0002", "0.00002"]):
        if pv < 1e8:
            label(ax2, (pv, sg / np.sqrt(pv)), txt, color="C0", offset=(7, 4), fontsize=9)
        else:                                                   # up and to the left, where the line leaves room; a leader names its marker
            ax2.annotate(txt, (pv, sg / np.sqrt(pv)), textcoords="offset points", xytext=(-2, 20), color="C0", fontsize=9, ha="right", va="bottom",
                         arrowprops=dict(arrowstyle="-", color="C0", lw=0.8, relpos=(1, 0), shrinkA=0, shrinkB=3))
    ns = list(runs)
    ax2.plot(ns, [abs(runs[k] - I) for k in ns], "D", mfc="white", mec="k", ms=5, zorder=6, label="actual error of one run (seed 8)")
    ax2.set_xlim(60, 2e8); ax2.set_ylim(1e-6, 8)
    ax2.legend(loc="upper right", fontsize=8.5, handlelength=1.8, borderaxespad=0.3)
    ax2.set_title("(b) The error falls like $1/\\sqrt{n}$, not $1/n$:\none more decimal, 100 times the samples")
    ax2.set_xlabel("samples $n$ (log scale)"); ax2.set_ylabel("standard error (log scale)")

    # (c) the antithetic pair: one high, one low, and their average nearly flat
    ax3.plot(x, g(x), color="C0", lw=1.7, label="$g(u)$")
    ax3.plot(x, g(1 - x), color="C1", lw=1.7, ls="--", label="$g(1 - u)$")
    ax3.plot(x, 0.5 * (g(x) + g(1 - x)), color="C3", lw=2.6, label="pair average")
    ax3.axhline(I, color=GREY, lw=1, ls=":")
    u0 = 0.107
    ax3.plot([u0, u0], [g(u0), g(1 - u0)], color="k", lw=1)
    ax3.plot([u0, u0, u0], [g(u0), 0.5 * (g(u0) + g(1 - u0)), g(1 - u0)], "o", color="k", ms=4.5, zorder=6)
    label(ax3, (u0, g(1 - u0)), f"$u = {u0}$: heights {g(u0):.3f}\nand {g(1 - u0):.3f}, average {0.5 * (g(u0) + g(1 - u0)):.3f}", offset=(-6, -9), va="top", fontsize=8.5)
    boxed(ax3, (0.98, I), f"$I = {I:.3f}$", color=GREY_TEXT, offset=(0, 6), ha="right", va="bottom", fontsize=8.5)
    boxed(ax3, (0.5, 1.2), f"Var $g(U) = {var:.4f}$\nVar of the pair average $= {var_pair:.4f}$\ncorrelation ${corr:.2f}$", va="center", fontsize=8.5)
    ax3.set_xlim(0, 1); ax3.set_ylim(0.2, 1.5)
    ax3.legend(loc="upper center", ncol=3, fontsize=8.5, handlelength=1.4, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.3)
    ax3.set_title(f"(c) Antithetic pairs: the average is almost\nflat, {gain:.0f} times less variance")
    ax3.set_xlabel("$u$"); ax3.set_ylabel("value")
    fig.tight_layout()
    return fig


FIGURES = {
    "geometric-memoryless": fig_geometric_memoryless,
    "poisson-process": fig_poisson_process,
    "exponential-hazard": fig_exponential_hazard,
    "normal-lognormal": fig_normal_lognormal,
    "tails": fig_tails,
    "lln-dilution": fig_lln_dilution,
    "clt-speed": fig_clt_speed,
    "monte-carlo": fig_monte_carlo,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 08 concept-guide figures.")
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
