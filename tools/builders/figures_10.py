"""Figures for the Module 10 concept guide — Estimation and Sampling.

Writes SVG files into part-4-statistics/10-estimation-and-sampling/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, recomputed here and checked against the value the guide prints, so
picture and text agree exactly. The few simulated panels (the fifty intervals,
the latency bootstrap, the sample behind the bootstrap maximum) use fixed seeds,
and the latency data are the lesson notebook's own.

    python tools/builders/figures_10.py                     # (re)write every figure
    python tools/builders/figures_10.py bootstrap           # only the named figure(s)
    python tools/builders/figures_10.py --preview DIR       # also write PNG previews to DIR
"""
import argparse
import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402,F401

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy import integrate, stats  # noqa: E402

FIG_DIR = ROOT / "part-4-statistics/10-estimation-and-sampling/figures"

WHITE_BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)  # lifts text off lines and fills
# Text colours: darker than the C1 / C2 / GREY fills and lines they label, so small text keeps its contrast
GREY_TEXT, DARK_ORANGE, DARK_GREEN = "0.4", "#b35806", "#1e7b1e"
Z = stats.norm.ppf(0.975)


def check(value, guide, tol):
    """Return `value` after confirming that it matches the number the guide prints (to `tol`)."""
    if abs(float(value) - guide) > tol:
        raise AssertionError(f"recomputed {float(value)!r}, but the guide prints {guide}")
    return value


def boxed(ax, xy, text, color="k", offset=(0, 0), fontsize=9, ha="center", va="center"):
    """Like figbuild.label, but on a translucent white card so it stays readable over fills and lines."""
    ax.annotate(text, xy, textcoords="offset points", xytext=offset, color=color, fontsize=fontsize,
                ha=ha, va=va, bbox=WHITE_BOX, zorder=6)


def log_xticks(ax, ticks, labels=None):
    """Name the ticks of a log x-axis explicitly: plain numbers the guide uses, not powers of ten."""
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if labels is not None else [f"{t:,}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def wald(k, n, z=Z):
    p = k / n
    half = z * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z / den * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - half, centre + half


def exact_coverage(interval, n, p):
    """P(interval(K, n) contains p) for K ~ Bin(n, p): a finite sum, so nothing here is simulated."""
    k = np.arange(n + 1)
    lo, hi = interval(k, n)
    return stats.binom.pmf(k, n, p)[(lo <= p) & (p <= hi)].sum()


def expected_max(k):
    """E[max of k independent N(0, 1)], by numerical integration of k x phi(x) Phi(x)^(k - 1)."""
    f = lambda x: x * k * stats.norm.pdf(x) * stats.norm.cdf(x) ** (k - 1)  # noqa: E731
    return integrate.quad(f, -12, 12, limit=200)[0]


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_sampling_and_selection_bias():
    """Section 1: all ten sample means of {2, 4, 4, 6, 9}; the survey that reweights its population; bias vs noise."""
    pop = np.array([2, 4, 4, 6, 9], float)
    mu = check(pop.mean(), 5, 1e-12)
    sigma2 = check(pop.var(), 5.6, 1e-12)
    means = np.array([np.mean(c) for c in itertools.combinations(pop, 2)])
    assert sorted(means) == sorted([3, 3, 4, 5.5, 4, 5, 6.5, 5, 6.5, 7.5])
    check(means.mean(), mu, 1e-12)
    var2 = check(means.var(), 2.1, 1e-12)
    check(sigma2 / 2 * (5 - 2) / (5 - 1), var2, 1e-12)
    values, counts = np.unique(means, return_counts=True)

    share, usage, respond = np.array([0.7, 0.3]), np.array([2.0, 20.0]), np.array([0.1, 0.4])
    truth = check(share @ usage, 7.4, 1e-12)
    joint = share * respond                                      # 0.07 and 0.12
    rate = check(joint.sum(), 0.19, 1e-12)
    resp = joint / rate
    check(resp[1], 0.63, 5e-3)
    biased = check(resp @ usage, 13.4, 0.05)
    excess = check(biased / truth - 1, 0.81, 5e-3)
    sd_resp = np.sqrt(resp[0] * resp[1]) * (usage[1] - usage[0])  # SD of a two-point distribution
    sd_true = np.sqrt(share[0] * share[1]) * (usage[1] - usage[0])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.05]})

    # (a) the population as a strip, the sampling distribution as bars
    ax1.bar(values, counts / 10, width=0.42, color="C1", alpha=0.75)
    ax1.plot([2, 4, 4, 6, 9], [0.395, 0.395, 0.432, 0.395, 0.395], "o", color="C0", ms=7, ls="none")
    ax1.axvline(mu, color="k", ls="--", lw=1.2)
    ax1.text(9.6, 0.462, "the five workers", color="C0", ha="right", va="center", fontsize=9.5)
    boxed(ax1, (mu, 0.345), f"μ = {mu:g}", fontsize=9.5)
    boxed(ax1, (5.6, 0.265), f"the ten sample means:\nmean {means.mean():g} = μ,\nvariance {var2:g} = ({sigma2:g}/2) × (3/4)", fontsize=9)
    ax1.set_xlim(1.4, 9.8); ax1.set_ylim(0, 0.5)
    ax1.set_xticks([2, 3, 4, 5, 6, 7, 8, 9]); ax1.set_yticks([0, 0.1, 0.2]); ax1.grid(axis="x", visible=False)
    ax1.set_title("(a) A statistic has a distribution:\nall 10 samples of $n = 2$")
    ax1.set_xlabel("orders packed per hour"); ax1.set_ylabel("probability of each sample mean")

    # (b) unequal response rates reweight the population
    ys, h = (1.0, 0.0), 0.42
    for y, sh in zip(ys, (share, resp)):
        ax2.barh(y, sh[0], height=h, color="C0", alpha=0.7)
        ax2.barh(y, sh[1], left=sh[0], height=h, color="C3", alpha=0.75, hatch="///", ec="white", lw=0)
    for x, t in [(share[0] / 2, f"{share[0]:.0%} casual\n{usage[0]:g} h"), (share[0] + share[1] / 2, f"{share[1]:.0%}\nengaged\n{usage[1]:g} h")]:
        boxed(ax2, (x, 1.0), t, fontsize=8.5)
    for x, t in [(resp[0] / 2, f"{resp[0]:.1%}\ncasual"), (resp[0] + resp[1] / 2, f"{resp[1]:.1%} engaged")]:
        boxed(ax2, (x, 0.0), t, fontsize=8.5)
    for xa, xb, col in [(0.50, 0.26, "C0"), (0.72, 0.44, "C3")]:   # from a segment above to the same group below
        ax2.annotate("", xy=(xb, h / 2), xytext=(xa, 1.0 - h / 2),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, shrinkA=1, shrinkB=1))
    ax2.text(0.015, 0.46, f"{respond[0]:.0%} answer:\n{share[0]:.2f} × {respond[0]:.2f}\n= {joint[0]:.2f}", color="C0", ha="left", va="center", fontsize=8.5)
    ax2.text(0.99, 0.46, f"{respond[1]:.0%} answer:\n{share[1]:.2f} × {respond[1]:.2f}\n= {joint[1]:.2f}", color="C3", ha="right", va="center", fontsize=8.5)
    ax2.text(0.5, 1.0 + h / 2 + 0.03, f"true mean {truth:g} h", ha="center", va="bottom", fontsize=9.5)
    ax2.text(0.5, -h / 2 - 0.03, f"respondent mean {biased:.1f} h (+{excess:.0%})", ha="center", va="top", fontsize=9, color="C3")
    ax2.set_xlim(0, 1); ax2.set_ylim(-0.48, 1.4)
    ax2.set_yticks(list(ys)); ax2.set_yticklabels(["all\nusers", "respon-\ndents"])
    ax2.grid(False)
    ax2.set_title("(b) The mechanism: response rates\nreweight the population")
    ax2.set_xlabel("share of the group"); ax2.set_ylabel("who is counted")

    # (c) noise shrinks with the number of respondents, bias does not
    k = np.geomspace(190, 190_000, 200)
    band_b, band_t = Z * sd_resp / np.sqrt(k), Z * sd_true / np.sqrt(k)
    check(band_b[0], 1.235, 5e-3); check(band_b[-1], 0.039, 5e-4)
    ax3.fill_between(k, biased - band_b, biased + band_b, color="C3", alpha=0.25, lw=0)
    ax3.plot(k, np.full_like(k, biased), color="C3", lw=2)
    ax3.fill_between(k, truth - band_t, truth + band_t, color="C0", alpha=0.25, lw=0)
    ax3.plot(k, np.full_like(k, truth), color="C0", lw=2, ls="--")
    ax3.annotate("", xy=(4000, biased), xytext=(4000, truth),
                 arrowprops=dict(arrowstyle="<->", color="k", lw=1.2, shrinkA=0, shrinkB=0))
    boxed(ax3, (4000, 10.4), f"bias +{biased - truth:.1f} h\nat every $n$", fontsize=9.5)
    ax3.text(6000, 15.1, f"survey respondents: {biased:.1f} h", color="C3", ha="center", va="center", fontsize=9.5)
    ax3.text(6000, 5.45, f"random sample: {truth:g} h (the truth)", color="C0", ha="center", va="center", fontsize=9)
    label(ax3, (200, biased - band_b[0]), f"±{band_b[0]:.1f}", color="C3", offset=(0, -3), va="top", fontsize=9.5)
    label(ax3, (190_000, biased), f"±{band_b[-1]:.2f}", color="C3", offset=(-2, 4), ha="right", fontsize=9.5)
    ax3.set_xscale("log"); ax3.set_xlim(190, 190_000); ax3.set_ylim(4.8, 16)
    log_xticks(ax3, [190, 1900, 19_000, 190_000])
    ax3.set_title("(c) More responses shrink the noise\nband (±1.96 SE), never the bias")
    ax3.set_xlabel("number of respondents $n$ (log scale)"); ax3.set_ylabel("estimated mean usage (hours/month)")
    fig.tight_layout()
    return fig


def fig_robust_statistics():
    """Section 2: ten response times with one cold-cache outlier; moving that one value moves only the mean."""
    x = np.array([120, 135, 140, 150, 155, 160, 170, 180, 210, 2400], float)
    mean = check(x.mean(), 382, 1e-12)
    median = check(np.median(x), 157.5, 1e-12)
    trim = check(np.sort(x)[1:-1].mean(), 162.5, 1e-12)
    s = check(x.std(ddof=1), 709.5, 0.05)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    check(q1, 142.5, 1e-9); check(q3, 177.5, 1e-9)
    iqr = check(q3 - q1, 35, 1e-9)
    mad = check(np.median(np.abs(x - median)), 20, 1e-12)
    smad = check(1.4826 * mad, 29.7, 0.06)
    clean = x[:-1]
    check(clean.mean(), 157.8, 0.05); check(clean.std(ddof=1), 26.7, 0.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    # (a) dots on a strip, three location estimates, two spread brackets
    ax1.plot(clean, np.full(9, 1.0), "o", color="C0", ms=7, zorder=4)
    ax1.annotate("", xy=(398, 1.0), xytext=(350, 1.0), arrowprops=dict(arrowstyle="-|>", color="C3", lw=2, shrinkA=0, shrinkB=0))
    boxed(ax1, (398, 1.0), "tenth value: 2400\n(off scale)", color="C3", offset=(0, -7), ha="right", va="top", fontsize=9)
    for xv, col, ls, y_top in [(median, "C2", "-", 1.53), (trim, "C1", "--", 1.32), (mean, "C3", "-", 1.53)]:
        ax1.plot([xv, xv], [0.8, y_top], color=col, ls=ls, lw=1.8, zorder=3)
    boxed(ax1, (median, 1.53), f"median {median:g}", color=DARK_GREEN, ha="center", va="bottom", fontsize=9.5)
    boxed(ax1, (trim, 1.32), f"10%-trimmed mean {trim:g}", color=DARK_ORANGE, ha="left", va="center", offset=(3, 0), fontsize=9.5)
    boxed(ax1, (398, 1.53), f"mean {mean:g}", color="C3", ha="right", va="bottom", fontsize=9.5)
    for lo, hi, y, col, t in [(q1, q3, 0.52, "C2", f"IQR = {q3:g} − {q1:g} = {iqr:g}"),
                              (median - smad, median + smad, 0.24, "C1", f"median ± 1.4826 × MAD = {median:g} ± {smad:.1f}")]:
        ax1.annotate("", xy=(hi, y), xytext=(lo, y), arrowprops=dict(arrowstyle="|-|", color=col, lw=2, shrinkA=0, shrinkB=0, mutation_scale=4))
        label(ax1, (hi, y), t, color=DARK_GREEN if col == "C2" else DARK_ORANGE, offset=(6, 0), va="center", fontsize=9)
    ax1.text(398, -0.05, f"mean ± s = {mean:g} ± {s:.1f}: off the chart both ways", color="C3", ha="right", va="center", fontsize=9.5)
    ax1.set_xlim(100, 400); ax1.set_ylim(-0.2, 1.72)
    ax1.set_yticks([]); ax1.grid(axis="y", visible=False)
    ax1.set_title("(a) One cold-cache request: the mean lands\nbeyond nine of the ten values")
    ax1.set_xlabel("response time (ms)"); ax1.set_ylabel("spread (below), data (above)")

    # (b) sweep the tenth value
    xs = np.geomspace(210, 10_000, 200)
    sweep = np.column_stack([np.tile(clean, (len(xs), 1)), xs])
    m = sweep.mean(1); med = np.median(sweep, 1); tr = np.sort(sweep, 1)[:, 1:-1].mean(1)
    check(m[0], 163, 1e-9); check(m[-1], 1142, 1e-9)
    assert np.allclose(med, median) and np.allclose(tr, trim)    # flat over the whole sweep
    ax2.plot(xs, m, color="C3", lw=2, label="mean = (1420 + x)/10")
    ax2.plot(xs, tr, color="C1", lw=2, ls="--", label=f"10%-trimmed mean: stays {trim:g}")
    ax2.plot(xs, med, color="C2", lw=2, label=f"median: stays {median:g}")
    ax2.plot([2400, 2400], [0, mean], color=GREY, ls=":", lw=1.2)
    ax2.plot([2400], [mean], "o", color="C3", ms=7, zorder=5)
    label(ax2, (2400, mean), f"x = 2400: mean {mean:g}", color="C3", offset=(-7, 5), ha="right", fontsize=9.5)
    ax2.set_xscale("log"); ax2.set_xlim(210, 10_000); ax2.set_ylim(0, 1250)
    log_xticks(ax2, [210, 500, 1000, 2400, 5000, 10_000])
    ax2.legend(loc="upper left", fontsize=9)
    ax2.text(0.02, 0.55, "breakdown point: mean 0, one value is enough;\nmedian ⌊9/2⌋/10 = 0.4 at n = 10",
             transform=ax2.transAxes, ha="left", va="center", fontsize=9.5)
    ax2.set_title("(b) Move one observation: the mean follows it,\nthe median and trimmed mean do not")
    ax2.set_xlabel("value x of the tenth observation (ms, log scale)"); ax2.set_ylabel("location estimate (ms)")
    fig.tight_layout()
    return fig


def fig_bias_variance_mse():
    """Section 3: the three variance estimators; two scales as averaging proceeds; the exact James-Stein risk."""
    divisors = np.array([9.0, 10.0, 11.0])
    bias = 9 / divisors - 1
    var = 18 / divisors ** 2
    mse = var + bias ** 2
    for got, guide in zip(bias, (0, -0.100, -0.182)):
        check(got, guide, 5e-4)
    for got, guide in zip(var, (0.222, 0.180, 0.149)):
        check(got, guide, 5e-4)
    for got, guide in zip(mse, (0.222, 0.190, 0.182)):
        check(got, guide, 5e-4)

    def scale_mse(b, sd, m):
        return sd ** 2 / m + b ** 2
    check(scale_mse(0.5, 0.1, 1), 0.26, 1e-12); check(scale_mse(0, 1, 1), 1.0, 1e-12)
    check(scale_mse(0.5, 0.1, 100), 0.2501, 1e-12); check(scale_mse(0, 1, 100), 0.01, 1e-12)
    cross = 0.99 / 0.25                                          # 1/m = 0.01/m + 0.25

    p = 10
    j = np.arange(600)

    def js_risk(lam):
        """p - (p - 2)^2 E[1 / chi^2_p(lam)], the Poisson-mixture series for the non-central chi-square."""
        return p - (p - 2) ** 2 * np.sum(stats.poisson.pmf(j, lam / 2) / (p - 2 + 2 * j))
    check(js_risk(10), 6, 0.3); check(js_risk(27), 8, 0.1); check(js_risk(0), 2, 1e-9)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.0))

    # (a) stacked MSE
    xpos = np.arange(3)
    ax1.bar(xpos, var, width=0.62, color="C0", alpha=0.75, label="variance")
    ax1.bar(xpos, bias ** 2, bottom=var, width=0.62, color="C3", alpha=0.85, hatch="///", ec="white", lw=0, label="bias²")
    for i in xpos:
        label(ax1, (i, mse[i]), f"{mse[i]:.3f}", offset=(0, 2), ha="center", fontsize=10.5)
        label(ax1, (i, mse[i]), "bias 0" if i == 0 else f"bias −{abs(bias[i]):.3f}", color="C3", offset=(0, 15), ha="center", fontsize=9)
        boxed(ax1, (i, var[i] / 2), f"{var[i]:.3f}", fontsize=9.5)
    ax1.set_xticks(xpos); ax1.set_xticklabels(["n − 1 = 9\n(unbiased)", "n = 10", "n + 1 = 11"])
    ax1.set_xlim(-0.5, 2.55); ax1.set_ylim(0, 0.33); ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper center", ncol=2, fontsize=9, handlelength=1.4, columnspacing=1.0)
    ax1.set_title("(a) MSE = variance + bias²: the\nunbiased divisor is the worst")
    ax1.set_xlabel("divisor of the sum of squares\n(n = 10 draws, σ² = 1)"); ax1.set_ylabel("mean squared error")

    # (b) the two scales
    m = np.geomspace(1, 100, 200)
    ax2.plot(m, scale_mse(0.5, 0.1, m), color="C1", lw=2, ls="--", label="scale A: bias 0.5, SD 0.1")
    ax2.plot(m, scale_mse(0, 1, m), color="C0", lw=2, label="scale B: bias 0, SD 1")
    ax2.plot([cross], [1 / cross], "ko", ms=6, zorder=5)
    ax2.annotate("B wins from\nm = 4", (cross, 1 / cross), xytext=(2.5, 0.075), ha="center", va="center", fontsize=9.5,
                 arrowprops=dict(arrowstyle="-", color="k", lw=0.8, shrinkB=3))
    ax2.plot([1, 100], [scale_mse(0.5, 0.1, 1), scale_mse(0.5, 0.1, 100)], "o", color="C1", ms=6, zorder=5, clip_on=False, ls="none")
    ax2.plot([1, 100], [scale_mse(0, 1, 1), scale_mse(0, 1, 100)], "o", color="C0", ms=6, zorder=5, clip_on=False, ls="none")
    label(ax2, (1, 0.26), "0.26", color=DARK_ORANGE, offset=(6, -13), fontsize=9.5)
    label(ax2, (1, 1.0), "1.0", color="C0", offset=(7, 1), fontsize=9.5)
    label(ax2, (100, 0.2501), "0.2501", color=DARK_ORANGE, offset=(-3, 4), ha="right", fontsize=9.5)
    label(ax2, (100, 0.01), "0.01", color="C0", offset=(-7, -13), ha="right", fontsize=9.5)
    ax2.text(85, 0.85, "A's bias² = 0.25\nnever averages away", color=DARK_ORANGE, ha="right", va="center", fontsize=9)
    ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlim(1, 100); ax2.set_ylim(0.0012, 3)
    log_xticks(ax2, [1, 4, 10, 100])
    ax2.legend(loc="lower left", fontsize=9, handlelength=1.8)
    ax2.set_title("(b) Variance averages away;\nbias does not")
    ax2.set_xlabel("weighings averaged, m"); ax2.set_ylabel("MSE (g²)")

    # (c) James-Stein
    lam = np.linspace(0, 150, 151)
    risk = np.array([js_risk(v) for v in lam])
    assert risk.max() < p and np.all(np.diff(risk) > 0)
    ax3.plot(lam, np.full_like(lam, p), color="C0", lw=2, ls="--", label=f"X (unbiased): MSE = p = {p}")
    ax3.plot(lam, risk, color="C2", lw=2, label="James–Stein (exact risk)")
    for v, off in [(10, (7, -3)), (27, (6, -13))]:
        ax3.plot([v], [js_risk(v)], "o", color="C2", ms=6, zorder=5)
        label(ax3, (v, js_risk(v)), f"‖θ‖² = {v}: {js_risk(v):.1f}", color=DARK_GREEN, offset=off, fontsize=9.5)
    ax3.plot([0], [js_risk(0)], "o", color="C2", ms=6, zorder=5, clip_on=False)
    label(ax3, (0, js_risk(0)), f"θ = 0: {js_risk(0):g}", color=DARK_GREEN, offset=(8, -10), fontsize=9.5)
    boxed(ax3, (93, 3.9), "gain largest near the target\nθ = 0, shrinks with distance,\nnever becomes a loss", fontsize=9)
    ax3.set_xlim(0, 150); ax3.set_ylim(0, 14); ax3.set_yticks([0, 2, 4, 6, 8, 10])
    ax3.legend(loc="upper right", fontsize=9, handlelength=1.8)
    ax3.set_title("(c) Shrinkage with p = 10: lower\ntotal MSE for every θ")
    ax3.set_xlabel("‖θ‖²: squared distance from 0"); ax3.set_ylabel("total MSE over 10 coordinates")
    fig.tight_layout()
    return fig


def fig_standard_error():
    """Section 4: the filling machine: individual packets against means of 4 and 16; SD stays, SE falls."""
    sigma, target, xbar = 4.0, 500.0, 497.5
    se4, se16 = sigma / np.sqrt(4), sigma / np.sqrt(16)
    check(se4, 2, 1e-12); check(se16, 1, 1e-12)
    z16 = check((target - xbar) / se16, 2.5, 1e-12)
    z4 = check((target - xbar) / se4, 1.25, 1e-12)
    p16 = check(2 * stats.norm.sf(z16), 0.012, 5e-4)
    p4 = 2 * stats.norm.sf(z4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})

    # (a) three densities
    x = np.linspace(488, 512, 800)
    curves = [(sigma, "C0", ":", f"one packet: SD {sigma:g} g"), (se4, "C1", "--", f"mean of 4: SE {se4:g} g"), (se16, "C2", "-", f"mean of 16: SE {se16:g} g")]
    for sd, col, ls, lab in curves:
        ax1.plot(x, stats.norm.pdf(x, target, sd), color=col, ls=ls, lw=2, label=lab)
    for side in (x <= target - 2.5, x >= target + 2.5):
        ax1.fill_between(x[side], stats.norm.pdf(x[side], target, se16), color="C2", alpha=0.45, lw=0)
    ax1.axvline(target, color="k", ls="--", lw=1)
    label(ax1, (target, 0.43), "target 500", offset=(4, 0), va="center", fontsize=9.5)
    ax1.axvline(xbar, color="C3", lw=1.8)
    label(ax1, (xbar, 0.43), f"sample mean {xbar:g}", color="C3", offset=(-4, 0), ha="right", va="center", fontsize=9.5)
    ax1.annotate(f"n = 4: only {z4:g} SE out,\nP = {p4:.0%}: within noise", (xbar, stats.norm.pdf(xbar, target, se4)), xytext=(488.4, 0.25), color=DARK_ORANGE,
                 fontsize=9.5, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=DARK_ORANGE, lw=0.8, shrinkB=2))
    ax1.annotate(f"n = 16: {z16:g} SE out,\nshaded tails P = {p16:.1%}", (496.9, 0.012), xytext=(488.4, 0.135), color=DARK_GREEN,
                 fontsize=9.5, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=DARK_GREEN, lw=0.8, shrinkB=1))
    ax1.set_xlim(488, 512); ax1.set_ylim(0, 0.46)
    ax1.legend(loc="center right", fontsize=9, handlelength=2.2)
    ax1.set_title("(a) Packets spread with SD σ; a mean of n packets\nspreads with SE σ/√n")
    ax1.set_xlabel("weight (g)"); ax1.set_ylabel("density")

    # (b) the two curves against n
    n = np.geomspace(1, 100, 200)
    ax2.plot(n, np.full_like(n, sigma), color="C0", ls=":", lw=2, label="SD of one packet: does not shrink")
    ax2.plot(n, sigma / np.sqrt(n), color="C2", lw=2, label="SE of the mean: 4/√n")
    pts = [1, 4, 16, 64]
    ax2.plot(pts, [sigma / np.sqrt(k) for k in pts], "o", color="C2", ms=7, zorder=5, clip_on=False)
    for k, off, ha in [(1, (5, 3), "left"), (4, (5, 3), "left"), (16, (5, 3), "left"), (64, (2, 6), "center")]:
        label(ax2, (k, sigma / np.sqrt(k)), f"SE {sigma / np.sqrt(k):g}", color=DARK_GREEN, offset=off, ha=ha, fontsize=9.5)
    ax2.annotate("", xy=(16, 1), xytext=(4, 2), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2, shrinkA=6, shrinkB=6, connectionstyle="arc3,rad=0.4"))
    label(ax2, (2.0, 0.42), "n × 4 halves the SE", fontsize=9.5, offset=(0, 0))
    ax2.set_xscale("log"); ax2.set_xlim(1, 100); ax2.set_ylim(0, 5.3)
    log_xticks(ax2, pts + [100])
    ax2.legend(loc="upper right", fontsize=9, handlelength=2.2)
    ax2.set_title("(b) More data leaves the SD alone\nand shrinks the SE like 1/√n")
    ax2.set_xlabel("sample size n (log scale)"); ax2.set_ylabel("grams")
    fig.tight_layout()
    return fig


def fig_coverage_t_vs_z():
    """Section 5: fifty samples of n = 4 with the t and the z multiplier; why: the t_3 pivot has heavy tails."""
    data = np.array([9.8, 10.4, 10.1, 9.9])
    check(data.mean(), 10.05, 1e-12)
    s0 = check(data.std(ddof=1), 0.2646, 5e-5)
    t3 = check(stats.t.ppf(0.975, 3), 3.182, 5e-4)
    check(t3 * s0 / 2, 0.42, 5e-3); check(Z * s0 / 2, 0.26, 5e-3)
    narrower = check(1 - Z / t3, 0.38, 5e-3)
    cover_z = check(2 * stats.t.cdf(Z, 3) - 1, 0.855, 5e-4)
    mu = 10.0
    X = np.random.default_rng(28).normal(mu, s0, size=(50, 4))   # sigma = the worked example's s
    xb, s = X.mean(1), X.std(1, ddof=1)
    i = np.arange(1, 51)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.0))

    def intervals(ax, mult):
        half = mult * s / 2
        miss = np.abs(xb - mu) > half
        ax.vlines(i[~miss], (xb - half)[~miss], (xb + half)[~miss], color="C0", lw=1.3)
        ax.vlines(i[miss], (xb - half)[miss], (xb + half)[miss], color="C3", lw=2.6)
        ax.plot(i, xb, "o", color="k", ms=2.2, zorder=4)
        ax.plot(i[miss], np.full(miss.sum(), 11.08), "v", color="C3", ms=5, zorder=4)   # shape as well as colour
        ax.axhline(mu, color="k", ls="--", lw=1.1)
        ax.set_xlim(0, 51); ax.set_ylim(8.5, 11.2); ax.set_yticks([9, 9.5, 10, 10.5, 11])
        ax.set_xlabel("sample number")
        ax.set_ylabel("interval for μ (mg/l)")
        ax.grid(axis="x", visible=False)
        return miss.sum()

    n_t = intervals(ax1, t3)
    n_z = intervals(ax2, Z)
    assert (n_t, n_z) == (3, 7)
    check(50 * (1 - cover_z), 7.2, 0.05)
    handles = [Line2D([], [], color="C0", lw=1.5, label="contains μ = 10"),
               Line2D([], [], color="C3", lw=2.6, marker="v", ms=5, label="misses μ")]
    ax1.legend(handles=handles, loc="lower center", fontsize=8.5, ncol=2, handlelength=1.2, columnspacing=0.8)
    ax1.set_title(f"(a) Fifty 95% t-intervals (n = 4):\n{n_t} miss (2.5 expected)")
    ax2.set_title(f"(b) Same samples with z = 1.96:\n{n_z} miss (7.2 expected)")
    ax2.text(25.5, 8.68, f"{narrower:.0%} narrower; coverage {cover_z:.1%}", ha="center", va="center", fontsize=9, bbox=WHITE_BOX)

    # (c) the pivot
    x = np.linspace(-5, 5, 600)
    ft, fz = stats.t.pdf(x, 3), stats.norm.pdf(x)
    inside = np.abs(x) <= Z
    ax3.fill_between(x[inside], ft[inside], color="C0", alpha=0.2, lw=0)
    ax3.plot(x, fz, color="C1", lw=2, ls="--", label="N(0, 1)")
    ax3.plot(x, ft, color="C0", lw=2, label="t₃")
    for v in (-Z, Z):
        ax3.plot([v, v], [0, 0.285], color="C1", lw=1.3, ls="--")
    for v in (-t3, t3):
        ax3.plot([v, v], [0, 0.145], color="C0", lw=1.3)
    boxed(ax3, (0, 0.045), f"P = {cover_z:.3f}\ninside ±1.96", color="C0", fontsize=9)
    label(ax3, (Z, 0.285), "1.96", color=DARK_ORANGE, offset=(0, 2), ha="center", fontsize=9.5)
    ax3.text(3.45, 0.15, f"{t3:.2f}: t₃\nneeds this\nfor 0.95", color="C0", ha="center", va="bottom", fontsize=9)
    ax3.set_xlim(-5, 5); ax3.set_ylim(0, 0.5)
    ax3.legend(loc="upper left", fontsize=9, handlelength=2)
    ax3.set_title("(c) Why: with s for σ, the pivot\nhas heavier tails than the normal")
    ax3.set_xlabel("(x̄ − μ)/(s/√n), for n = 4"); ax3.set_ylabel("density")
    fig.tight_layout()
    return fig


def fig_wilson_vs_wald():
    """Section 5: where Wilson comes from (cut the band at p-hat); the guide's three cases; exact coverage at p = 0.05."""
    k0, n0 = 9, 200
    phat = check(k0 / n0, 0.045, 1e-12)
    wd0, wl0 = wald(k0, n0), wilson(k0, n0)
    check(wd0[0], 0.016, 5e-4); check(wd0[1], 0.074, 5e-4); check(wl0[0], 0.024, 5e-4); check(wl0[1], 0.083, 5e-4)
    check(wald(2, 150)[0], -0.005, 5e-4); check(wald(2, 150)[1], 0.032, 5e-4)
    check(wilson(2, 150)[0], 0.0037, 5e-5); check(wilson(2, 150)[1], 0.047, 5e-4)
    check(wilson(0, 20)[1], 0.161, 5e-4)
    half = lambda p, n=n0: Z * np.sqrt(p * (1 - p) / n)          # noqa: E731
    for v in wl0:                                                # the band's edges cross p-hat exactly at Wilson's limits
        check(abs(phat - v), half(v), 1e-10)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.2))

    # (a) the band and its cut
    p = np.linspace(1e-4, 0.14, 400)
    ax1.fill_between(p, p - half(p), p + half(p), color="C0", alpha=0.15, lw=0)
    ax1.plot(p, p + half(p), color="C0", lw=1.3); ax1.plot(p, p - half(p), color="C0", lw=1.3)
    ax1.plot(p, p, color=GREY, lw=0.9)
    ax1.axhline(phat, color="C3", lw=1.6)
    label(ax1, (0.14, phat), "$\\hat{p}$ = 9/200", color="C3", offset=(-2, 3), ha="right", fontsize=9.5)
    for v in wl0:
        ax1.plot([v, v], [0, phat], ":", color=DARK_GREEN, lw=1.2)
        ax1.plot([v], [phat], "o", color="C2", ms=6, zorder=5)
    ax1.hlines(0.0075, *wl0, color="C2", lw=4)
    ax1.hlines(0.0028, *wd0, color="C1", lw=4)
    ax1.plot([phat, phat], [phat - half(phat), phat + half(phat)], color="C1", lw=2.4, zorder=4)
    ax1.annotate(f"Wald borrows the\nband's height\nat $\\hat{{p}}$: ± {half(phat):.4f}", (phat, phat + half(phat)), xytext=(0.003, 0.113), color=DARK_ORANGE,
                 fontsize=9, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=DARK_ORANGE, lw=0.8, shrinkB=1))
    ax1.text(0.138, 0.0195, f"Wilson\n[{wl0[0]:.3f}, {wl0[1]:.3f}]", color=DARK_GREEN, ha="right", va="bottom", fontsize=8.5)
    ax1.text(0.138, 0.003, f"Wald\n[{wd0[0]:.3f}, {wd0[1]:.3f}]", color=DARK_ORANGE, ha="right", va="bottom", fontsize=8.5)
    ax1.set_xlim(0, 0.14); ax1.set_ylim(0, 0.14); ax1.set_aspect("equal"); ax1.set_anchor("N")
    ax1.set_xticks([0, 0.04, 0.08, 0.12]); ax1.set_yticks([0, 0.04, 0.08, 0.12])
    ax1.set_title("(a) Wilson keeps p in the SE: the\nband's width at p gives its limits")
    ax1.set_xlabel("true proportion p"); ax1.set_ylabel("observed $\\hat{p}$ (band: p ± 1.96 SE(p))")

    # (b) the three cases
    cases = [("9/200", 9, 200), ("0/20", 0, 20), ("2/150", 2, 150)]
    ys = [2, 1, 0]
    ax2.axvspan(-0.02, 0, color="0.85", lw=0)
    ax2.text(-0.0105, 2.55, "impossible", rotation=90, ha="center", va="top", fontsize=8.5, color=GREY_TEXT)
    for (name, k, n), y in zip(cases, ys):
        wd, wl = wald(k, n), wilson(k, n)
        pk = k / n
        ax2.hlines(y + 0.17, *wd, color="C1", lw=5)
        ax2.hlines(y - 0.17, *wl, color="C2", lw=5)
        ax2.plot([pk, pk], [y - 0.32, y + 0.32], color="k", lw=1.3, zorder=5)
        if k == 0:
            ax2.plot([0], [y + 0.17], "o", color="C1", ms=7, zorder=5, clip_on=False)
            label(ax2, (0, y + 0.17), "Wald [0, 0]", color=DARK_ORANGE, offset=(7, 0), va="center", fontsize=8.5)
            label(ax2, (0.01, y - 0.17), "Wilson [0, 0.161]", color=DARK_GREEN, offset=(0, -6), va="top", fontsize=8.5)
            ax2.plot([3 / n, 3 / n], [y - 0.32, y - 0.02], color="k", lw=1.1, ls=":")
            label(ax2, (3 / n, y - 0.02), "3/20", offset=(0, 1), ha="center", fontsize=9)
        else:
            label(ax2, (wd[1], y + 0.17), f"Wald [{wd[0]:.3f}, {wd[1]:.3f}]".replace("-", "−"), color=DARK_ORANGE, offset=(5, 0), va="center", fontsize=8.5)
            label(ax2, (wl[1], y - 0.17), f"Wilson [{wl[0]:.3f}, {wl[1]:.3f}]", color=DARK_GREEN, offset=(5, 0), va="center", fontsize=8.5)
    ax2.set_yticks(ys); ax2.set_yticklabels([c[0] for c in cases])
    ax2.set_xlim(-0.02, 0.26); ax2.set_ylim(-0.6, 2.6); ax2.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    ax2.grid(axis="y", visible=False)
    ax2.set_title("(b) Three cases: Wald can have\nzero width or go negative")
    ax2.set_xlabel("proportion (black tick: $\\hat{p}$)"); ax2.set_ylabel("events / trials")

    # (c) exact coverage at p = 0.05
    p0 = 0.05
    ns = np.arange(10, 1001)
    cw = np.array([exact_coverage(wald, n, p0) for n in ns])
    cs = np.array([exact_coverage(wilson, n, p0) for n in ns])
    for n, gw, gs in [(20, 0.64, 0.925), (100, 0.88, 0.97), (500, 0.93, 0.95)]:
        check(cw[n - 10], gw, 5e-3); check(cs[n - 10], gs, 5e-3)   # Wilson at n = 20 is 0.9245
    ax3.plot(ns, cw, color="C1", lw=0.9, label="Wald")
    ax3.plot(ns, cs, color="C2", lw=0.9, label="Wilson")
    ax3.axhline(0.95, color="k", ls="--", lw=1.1)
    label(ax3, (1000, 0.95), "nominal 0.95", offset=(-2, 15), ha="right", fontsize=9)
    for n in (20, 100, 500):
        ax3.plot([n], [cw[n - 10]], "s", color="C1", ms=5.5, mec="k", mew=0.5, zorder=5)
        ax3.plot([n], [cs[n - 10]], "o", color="C2", ms=5.5, mec="k", mew=0.5, zorder=5)
    label(ax3, (20, cw[10]), f"n = 20: {cw[10]:.2f}", color=DARK_ORANGE, offset=(7, -2), va="center", fontsize=9)
    ax3.annotate(f"n = 100: {cw[90]:.2f}", (100, cw[90]), xytext=(125, 0.815), color=DARK_ORANGE, fontsize=9, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=DARK_ORANGE, lw=0.8, shrinkB=3))
    ax3.annotate(f"n = 500: {cw[490]:.2f}", (500, cw[490]), xytext=(160, 0.875), color=DARK_ORANGE, fontsize=9, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=DARK_ORANGE, lw=0.8, shrinkB=3))
    ax3.set_xscale("log"); ax3.set_xlim(10, 1000); ax3.set_ylim(0.38, 1.04)
    log_xticks(ax3, [10, 20, 100, 1000])
    ax3.legend(handles=[Line2D([], [], color="C1", lw=0.9, marker="s", ms=5, mec="k", mew=0.5, label="Wald"),
                        Line2D([], [], color="C2", lw=0.9, marker="o", ms=5, mec="k", mew=0.5, label="Wilson")],
               loc="lower right", fontsize=9, handlelength=1.5)
    ax3.set_title("(c) Exact coverage at p = 0.05: Wald\nfalls far short at small n (squares)")
    ax3.set_xlabel("n (log scale)"); ax3.set_ylabel("coverage probability")
    fig.tight_layout()
    return fig


def fig_bootstrap():
    """Section 6: all 27 resamples of {1, 3, 5}; median vs p95 of the latency sample; the failure on the maximum."""
    sample = np.array([1.0, 3.0, 5.0])
    resamples = np.array(list(itertools.product(sample, repeat=3)))
    assert len(resamples) == 27
    boot = resamples.mean(1)
    check(boot.mean(), 3, 1e-12)
    bv = check(boot.var(), 8 / 9, 1e-12)
    check(sample.var() / 3, bv, 1e-12)
    vals, cnt = np.unique(np.round(boot, 9), return_counts=True)
    assert list(cnt) == [1, 3, 6, 7, 6, 3, 1]

    # the lesson's own data and resampling indices (build_10.py): the same generators, called in the same order
    latency = np.random.default_rng(7).lognormal(mean=np.log(150), sigma=0.6, size=200)
    idx = np.random.default_rng(0).integers(0, 200, size=(10_000, 200))
    med0 = check(np.median(latency), 138.7, 0.05)
    p950 = check(np.quantile(latency, 0.95), 334.3, 0.05)
    meds = np.median(latency[idx], axis=1)
    p95s = np.quantile(latency[idx], 0.95, axis=1)
    se_med = check(meds.std(ddof=1), 5.2, 0.05)
    se_p95 = check(p95s.std(ddof=1), 18.6, 0.05)
    ci_med = np.quantile(meds, [0.025, 0.975])
    ci_p95 = np.quantile(p95s, [0.025, 0.975])
    check(ci_med[0], 128.2, 0.05); check(ci_med[1], 148.4, 0.05)
    check(ci_p95[0], 290.0, 0.05); check(ci_p95[1], 356.4, 0.05)

    n = 50
    u = np.sort(np.random.default_rng(3).random(n))              # any sample shows it; the guide's fact is 1 - (1 - 1/n)^n
    umax = u[-1]
    p_same = 1 - (1 - 1 / n) ** n
    check(p_same, 0.63, 0.01)
    pmf = ((np.arange(1, n + 1) / n) ** n - (np.arange(0, n) / n) ** n)   # P(bootstrap max = u_(k))
    check(pmf[-1], p_same, 1e-12); check(pmf.sum(), 1, 1e-12)
    p_above = 1 - umax ** n

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.1))

    # (a) the exact bootstrap distribution
    ax1.bar(vals, cnt / 27, width=0.5, color="C0", alpha=0.7)
    for v, c in zip(vals, cnt):
        label(ax1, (v, c / 27), f"{c}", offset=(0, 2), ha="center", fontsize=9.5)
    ax1.plot(sample, np.full(3, 0.395), "D", color="C1", ms=7, ls="none")
    ax1.text(3, 0.42, "the sample {1, 3, 5}", color=DARK_ORANGE, ha="center", va="bottom", fontsize=9.5)
    boxed(ax1, (3, 0.335), "mean 3, variance 8/9 = (8/3)/3,\nthe plug-in formula sₙ²/n", fontsize=9)
    ax1.set_xticks(vals); ax1.set_xticklabels(["1", "5/3", "7/3", "3", "11/3", "13/3", "5"], fontsize=8.5)
    ax1.set_xlim(0.5, 5.5); ax1.set_ylim(0, 0.47); ax1.set_yticks([0, 0.1, 0.2, 0.3])
    ax1.grid(axis="x", visible=False)
    ax1.set_title("(a) All 27 resamples of {1, 3, 5}\n(numbers: how many give each mean)")
    ax1.set_xlabel("mean of the resample"); ax1.set_ylabel("bootstrap probability")

    # (b) the two statistics on one axis, each with its own bin width (narrower for the median)
    bins_med, bins_p95 = np.arange(110, 170, 2.5), np.arange(244, 420, 8)   # p95 replicates sit on few order statistics: wider bins
    h_med = ax2.hist(meds, bins=bins_med, density=True, color="C2", alpha=0.7)[0]
    h_p95 = ax2.hist(p95s, bins=bins_p95, density=True, color="C3", alpha=0.6, hatch="///", ec="white", lw=0)[0]
    yb = -0.0045
    for ci, col, tc in [(ci_med, "C2", DARK_GREEN), (ci_p95, "C3", "C3")]:
        ax2.annotate("", xy=(ci[1], yb), xytext=(ci[0], yb), arrowprops=dict(arrowstyle="|-|", color=col, lw=2, shrinkA=0, shrinkB=0, mutation_scale=3))
        ax2.text(ci.mean(), yb - 0.003, f"95% interval\n[{ci[0]:.1f}, {ci[1]:.1f}]", color=tc, ha="center", va="top", fontsize=9)
    top, top95 = h_med.max(), h_p95.max()
    for v, ytop in [(med0, top + 0.004), (p950, top95 + 0.004)]:
        ax2.plot([v, v], [0, ytop], color="k", lw=1.2, zorder=5)
    ax2.text(155, top * 0.9, f"median {med0:.1f} ms\nSE {se_med:.1f} ms", color=DARK_GREEN, ha="left", va="center", fontsize=9.5)
    ax2.text(p950, top95 + 0.006, f"p95 {p950:.1f} ms\nSE {se_p95:.1f} ms", color="C3", ha="center", va="bottom", fontsize=9.5)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlim(75, 415); ax2.set_ylim(-0.024, top + 0.012)
    ax2.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax2.grid(axis="x", visible=False)
    ax2.set_title(f"(b) 10,000 resamples of 200 latencies:\nthe p95 is {se_p95 / se_med:.1f} times as uncertain")
    ax2.set_xlabel("bootstrap replicate (ms)"); ax2.set_ylabel("density")

    # (c) the failure
    x = np.linspace(0.85, 1, 400)
    f = n * x ** (n - 1)
    ax3.plot(x, f, color="C0", lw=2)
    above = x >= umax
    ax3.fill_between(x[above], f[above], color="C0", alpha=0.2, lw=0)
    ax3.axvline(umax, color="k", ls="--", lw=1.1)
    ax3.set_xlim(0.86, 1.0); ax3.set_ylim(0, 62); ax3.grid(False)
    ax3.set_ylabel("density of the true max", color="C0")
    ax3.tick_params(axis="y", labelcolor="C0")
    ax4 = ax3.twinx()
    ax4.vlines(u[-5:], 0, pmf[-5:], color="C3", lw=2.4)
    ax4.plot(u[-5:], pmf[-5:], "o", color="C3", ms=6)
    ax4.set_ylim(0, 1.0); ax4.grid(False)
    ax4.set_ylabel("bootstrap probability of the max", color="C3")
    ax4.tick_params(axis="y", labelcolor="C3")
    label(ax4, (umax, pmf[-1]), f"{pmf[-1]:.1%} of bootstrap maxima\nequal the sample max;\nnone exceed it", color="C3", offset=(-6, 8), ha="right", fontsize=9)
    ax4.text(0.8625, 0.42, f"true max of a fresh sample:\nabove {umax:.4f} with\nprobability {p_above:.2f} (shaded)", color="C0", ha="left", va="center", fontsize=9)
    ax3.set_title("(c) Failure: the bootstrap maximum\ncollapses onto the sample maximum")
    ax3.set_xlabel(f"maximum of n = {n} uniforms")
    fig.tight_layout()
    return fig


def fig_paired_comparison():
    """Section 8: the 2 x 2 table of A and B; one gap, three standard errors; the best of k noisy candidates."""
    n = 1000
    aA, aB = 712, 701
    n10, n01 = 96, 85
    n11 = aA - n10
    n00 = n - n11 - n10 - n01
    assert n11 + n01 == aB and n11 + n00 == 819
    seA = check(np.sqrt(0.712 * 0.288 / n), 0.0143, 5e-5)
    seB = check(np.sqrt(0.701 * 0.299 / n), 0.0145, 5e-5)
    unp = check(np.hypot(seA, seB), 0.0204, 5e-5)
    d = check((n10 - n01) / n, 0.011, 1e-12)
    paired = lambda a, b: np.sqrt(((a + b) / n - ((a - b) / n) ** 2) / n)   # noqa: E731
    sep = check(paired(n10, n01), 0.0134, 5e-5)
    check(d - Z * sep, -0.015, 5e-4); check(d + Z * sep, 0.037, 5e-4)
    check(0.181 / 0.004 ** 2, 11_300, 20)
    se2 = check(paired(13, 2), 0.0039, 5e-5)                     # self-assessment question 7

    k_guide = [(3, 0.85), (5, 1.16), (10, 1.54), (50, 2.25)]
    for k, g in k_guide:
        check(expected_max(k), g, 5e-3)
    seeds = np.array([83.1, 84.0, 82.7, 83.5, 84.4])
    check(seeds.mean(), 83.54, 1e-9)
    s = check(seeds.std(ddof=1), 0.68, 5e-3)
    gain = check(seeds.max() - seeds.mean(), 0.86, 1e-9)
    check(2.25 * 0.73, 1.6, 0.05)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.1), gridspec_kw={"width_ratios": [1, 1.12, 1]})

    # (a) the mosaic
    wA = aA / n
    hBa = n11 / aA                                               # within "A right": share where B is right
    hBb = n01 / (n - aA)                                         # within "A wrong"
    cells = [(0, 0, wA, hBa, "0.87", None), (0, hBa, wA, 1 - hBa, "C0", "///"), (wA, 0, 1 - wA, hBb, "C3", "\\\\\\"), (wA, hBb, 1 - wA, 1 - hBb, "0.75", None)]
    for x0, y0, w, h, fc, hatch in cells:
        ax1.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec="white", lw=1.5, alpha=0.75 if hatch else 1, hatch=hatch))
    boxed(ax1, (wA / 2, hBa / 2), f"both right\n{n11}", fontsize=10)
    boxed(ax1, (wA / 2, hBa + (1 - hBa) / 2), f"A right, B wrong: {n10}", color="C0", fontsize=9.5)
    boxed(ax1, (wA + (1 - wA) / 2, hBb / 2), f"B right,\nA wrong:\n{n01}", color="C3", fontsize=9.5)
    boxed(ax1, (wA + (1 - wA) / 2, hBb + (1 - hBb) / 2), f"both\nwrong\n{n00}", fontsize=10)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.grid(False)
    ax1.set_xticks([wA / 2, wA + (1 - wA) / 2]); ax1.set_xticklabels([f"right ({aA})", f"wrong ({n - aA})"])
    ax1.set_yticks([])
    ax1.tick_params(length=0)
    ax1.set_title(f"(a) Only the {n10 + n01} disagreements move\nthe difference; {n11 + n00} agreements do not")
    ax1.set_xlabel("model A on the 1,000 questions"); ax1.set_ylabel("model B within each column (B right below)")

    # (b) the same gap, three SEs
    rows = [(2, unp, "C1", f"unpaired formula: SE {100 * unp:.2f}"), (1, sep, "C0", f"paired, {n10} vs {n01}: SE {100 * sep:.2f}"), (0, se2, "C2", f"if only 13 vs 2: SE {100 * se2:.2f}")]
    for y, se, col, t in rows:
        lo, hi = 100 * (d - Z * se), 100 * (d + Z * se)
        ax2.hlines(y, lo, hi, color=col, lw=6)
        ax2.plot([100 * d], [y], "o", color="k", ms=5, zorder=5)
        ax2.text(1.0, y + 0.14, t, ha="center", va="bottom", fontsize=9.5, bbox=dict(WHITE_BOX, pad=0.1), zorder=6)
        ax2.text(100 * d, y - 0.14, f"[{lo:+.1f}, {hi:+.1f}]".replace("-", "−"), ha="center", va="top", fontsize=9.5, bbox=dict(WHITE_BOX, pad=0.1), zorder=6)
    ax2.axvline(0, color="k", ls="--", lw=1.1)
    ax2.set_xlim(-4.2, 6.2); ax2.set_ylim(-0.55, 2.75)
    ax2.set_yticks([]); ax2.grid(axis="y", visible=False)
    ax2.set_title("(b) One 1.1-point gap, three 95%\nintervals: only the third excludes 0")
    ax2.set_xlabel("accuracy difference A − B (points)"); ax2.set_ylabel("analysis")

    # (c) the winner's curse
    ks = np.unique(np.geomspace(1, 100, 60).astype(int))
    em = np.array([expected_max(k) for k in ks])
    ax3.plot(ks, em, color="C0", lw=2)
    for k, g in k_guide:
        ax3.plot([k], [expected_max(k)], "o", color="C0", ms=6, zorder=5)
        label(ax3, (k, expected_max(k)), f"{expected_max(k):.2f} σ", color="C0", offset=(-6, 3), ha="right", fontsize=9.5)
    em5 = expected_max(5)
    ax3.text(0.97, 0.05, f"five seeds, s = {s:.2f} points:\nexpect +{em5 * s:.2f}, best seed +{gain:.2f}\n50 entries, SE 0.73: expect +1.6",
             transform=ax3.transAxes, ha="right", va="bottom", fontsize=8.5, bbox=WHITE_BOX)
    ax3.set_xscale("log"); ax3.set_xlim(1, 100); ax3.set_ylim(0, 2.8)
    log_xticks(ax3, [1, 3, 5, 10, 50, 100])
    ax3.set_title("(c) Winner's curse: the best of k\nequal candidates looks better")
    ax3.set_xlabel("number of candidates k (log scale)"); ax3.set_ylabel("expected excess of the max (σ units)")
    fig.tight_layout()
    return fig


FIGURES = {
    "sampling-and-selection-bias": fig_sampling_and_selection_bias,
    "robust-statistics": fig_robust_statistics,
    "bias-variance-mse": fig_bias_variance_mse,
    "standard-error": fig_standard_error,
    "coverage-t-vs-z": fig_coverage_t_vs_z,
    "wilson-vs-wald": fig_wilson_vs_wald,
    "bootstrap": fig_bootstrap,
    "paired-comparison": fig_paired_comparison,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 10 concept-guide figures.")
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
