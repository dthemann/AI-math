"""Figures for the Module 12 concept guide — Hypothesis Testing and Experiments.

Writes SVG files into part-4-statistics/12-hypothesis-testing-and-experiments/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, recomputed here and checked against the value the guide prints, so
picture and text agree exactly. Only two things are random, both with fixed
seeds: the sideways jitter of the dots in paired-vs-unpaired and the six null
walks in peeking. The false-positive rates and boundaries in peeking are not
simulated: they come from an exact recursion over the looks.

    python tools/builders/figures_12.py                     # (re)write every figure
    python tools/builders/figures_12.py peeking             # only the named figure(s)
    python tools/builders/figures_12.py --preview DIR       # also write PNG previews to DIR
"""
import argparse
import pathlib
import sys
from itertools import combinations
from math import comb

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402,F401

import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402

FIG_DIR = ROOT / "part-4-statistics/12-hypothesis-testing-and-experiments/figures"

WHITE_BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)  # lifts text off lines and fills
# Text colours: darker than the C1 / C2 / GREY fills and lines they label, so small text keeps its contrast
GREY_TEXT, DARK_ORANGE, DARK_GREEN = "0.4", "#b35806", "#1e7b1e"


def check(value, guide, tol):
    """Return `value` after confirming that it matches the number the guide prints (to `tol`)."""
    if abs(float(value) - guide) > tol:
        raise AssertionError(f"recomputed {float(value)!r}, but the guide prints {guide}")
    return value


def boxed(ax, xy, text, color="k", offset=(0, 0), fontsize=9.5, ha="center", va="center"):
    """Like figbuild.label, but on a translucent white card so it stays readable over fills and lines."""
    ax.annotate(text, xy, textcoords="offset points", xytext=offset, color=color, fontsize=fontsize,
                ha=ha, va=va, bbox=WHITE_BOX, zorder=6)


def leader(ax, xy, xytext, text, color="k", fontsize=9, ha="center", va="center", card=False):
    """Text at `xytext` with a thin line to the data point `xy`, for objects too small to hold their own label."""
    ax.annotate(text, xy, xytext=xytext, color=color, fontsize=fontsize, ha=ha, va=va, zorder=6,
                bbox=WHITE_BOX if card else None,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=1, shrinkB=1))


def minus(text):
    """Typographic minus signs in plain (non-maths) text."""
    return text.replace("-", "−")


def reject_prob(bounds, h=0.02):
    """P_H0(|Z_k| > bounds[k - 1] at some look k), for equally spaced looks at a growing sample.

    S_k = sqrt(k) Z_k is a random walk with N(0, 1) steps, so the density of S_k on the "not yet stopped"
    region follows from the previous one by convolving with the step density and cutting at the boundary.
    The cells cut by the boundary get a fractional weight, which keeps the error of the grid well below 1e-3.
    """
    edge = np.asarray(bounds, float) * np.sqrt(np.arange(1, len(bounds) + 1))
    s = np.arange(-int(edge.max() / h + 2), int(edge.max() / h + 3)) * h
    step = stats.norm.pdf(np.arange(-int(8 / h), int(8 / h) + 1) * h) * h
    inside = lambda b: np.clip((b - np.abs(s)) / h + 0.5, 0, 1)  # noqa: E731
    f = stats.norm.pdf(s) * h * inside(edge[0])                 # mass per cell of S_1, kept if not stopped
    for b in edge[1:]:
        f = fftconvolve(f, step, mode="same") * inside(b)
    return 1 - f.sum()


def pocock_constant(K, alpha=0.05):
    """The constant boundary c with overall level alpha over K equally spaced looks."""
    return brentq(lambda c: reject_prob(np.full(K, c)) - alpha, 1.9, 3.2, xtol=1e-4)


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_p_value_and_base_rate():
    """Section 1: 60 heads in 100 flips as a tail area of the exact null; the base-rate count behind p != P(H0 | data)."""
    n, x_obs = 100, 60
    mu, sd = check(n * 0.5, 50, 1e-12), check(np.sqrt(n * 0.25), 5, 1e-12)
    k = np.arange(30, 71)
    pmf = stats.binom.pmf(k, n, 0.5)
    tail = check(stats.binom.sf(x_obs - 1, n, 0.5), 0.0284, 5e-5)
    p_exact = check(2 * tail, 0.0569, 5e-5)
    check((x_obs - mu) / sd, 2.00, 1e-12); check((x_obs - 0.5 - mu) / sd, 1.90, 1e-12)
    t60, t595 = stats.norm.sf(x_obs, mu, sd), stats.norm.sf(x_obs - 0.5, mu, sd)
    p60, p595 = check(2 * t60, 0.0455, 5e-5), check(2 * t595, 0.0574, 5e-5)
    check(2 * stats.binom.sf(60, n, 0.5), 0.0352, 5e-5)         # the exact level-0.05 rule, X <= 39 or X >= 61

    total, base, power, alpha = 1000, 0.10, 0.8, 0.05
    real, null = total * base, total * (1 - base)
    det, miss = power * real, (1 - power) * real
    fa, quiet = alpha * null, (1 - alpha) * null
    sig = det + fa
    check(base * power / (base * power + (1 - base) * alpha), 0.64, 1e-12)
    frac_real = check(det / sig, 0.64, 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) the null as bars, the normal approximation as a curve, the tails at least as extreme as 60
    extreme = np.abs(k - mu) >= x_obs - mu
    ax1.bar(k[~extreme], pmf[~extreme], width=1.0, color="C0", alpha=0.8, ec="white", lw=0.4,
            label=f"Binomial({n}, 0.5): the exact null")
    ax1.bar(k[extreme], pmf[extreme], width=1.0, color="C3", alpha=0.9, ec="white", lw=0.4,
            label=f"at least as far from {mu:.0f} as {x_obs}")
    xs = np.linspace(30, 70, 401)
    ax1.plot(xs, stats.norm.pdf(xs, mu, sd), color="k", lw=1.3, label=f"normal approximation $\\mathcal{{N}}({mu:.0f}, {sd:.0f}^2)$")
    ymax = 0.14
    for v, ha, txt in [(x_obs, "left", f"observed: {x_obs}"), (2 * mu - x_obs, "right", f"as extreme: {2 * mu - x_obs:.0f}")]:
        ax1.plot([v, v], [0, 0.035], color="C3", ls="--", lw=1.2)
        ax1.text(v + (0.5 if ha == "left" else -0.5), 0.0295, txt, color="C3", ha=ha, va="center", fontsize=9)
    ax1.set_xlim(30, 70); ax1.set_ylim(0, ymax)
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.text(30.7, 0.1105, f"exact: {tail:.4f} a side, $p = {p_exact:.4f} > 0.05$\n"
             f"normal from 59.5: {t595:.4f} a side, $p = {p595:.4f} > 0.05$\n"
             f"normal from 60: {t60:.4f} a side, $p = {p60:.4f} < 0.05$",
             ha="left", va="top", fontsize=8.5, linespacing=1.4)
    ax1.set_title(f"(a) Null distribution of the number of heads:\nthe $p$-value is the area at least as extreme as {x_obs}")
    ax1.set_xlabel(f"number of heads $X$ in {n} flips"); ax1.set_ylabel("$P(X = k)$ under $H_0$")

    # the inset: the bar at 60 starts at 59.5, so a normal tail from 60 loses half of it
    ins = ax1.inset_axes([0.725, 0.345, 0.27, 0.595])
    kz = np.arange(59, 64)
    pz = stats.binom.pmf(kz, n, 0.5)
    ins.bar(kz[kz < x_obs], pz[kz < x_obs], width=1.0, color="C0", alpha=0.8, ec="white", lw=0.4)
    ins.bar(kz[kz >= x_obs], pz[kz >= x_obs], width=1.0, color="C3", alpha=0.9, ec="white", lw=0.4)
    xz = np.linspace(58.5, 63.5, 200)
    ins.plot(xz, stats.norm.pdf(xz, mu, sd), color="k", lw=1.3)
    half = np.linspace(x_obs - 0.5, x_obs, 30)
    ins.fill_between(half, stats.norm.pdf(half, mu, sd), color="C1", lw=0)
    ins.set_xlim(58.5, 63.5); ins.set_ylim(0, 0.0285)
    ins.set_xticks([59.5, 61, 62, 63]); ins.set_xticklabels(["59.5", "61", "62", "63"], fontsize=7.5)
    ins.tick_params(axis="x", pad=1.5, length=2)
    ins.set_xlabel("number of heads", fontsize=7.5, labelpad=1)
    ins.set_yticks([]); ins.grid(False)
    ins.set_title(f"zoom near {x_obs}", fontsize=8.5, pad=2.5)
    leader(ins, (59.75, 0.0085), (62.0, 0.0215), "half a bar\nlost if the\ntail starts\nat 60", fontsize=8)

    # (b) 1,000 ideas: count them through the test, then keep only the significant ones
    h = 0.44
    edge = dict(ec="0.45", lw=0.6)
    ax2.barh(0, real, height=h, color="C0", alpha=0.8, **edge)
    ax2.barh(0, null, left=real, height=h, color="0.85", **edge)
    # the order puts each significant group next to the other: real = [0, 100], null = [100, 1000]
    ax2.barh(1, miss, height=h, fc="white", hatch="/////", **dict(edge, ec="C0"))
    ax2.barh(1, det, left=miss, height=h, color="C2", alpha=0.85, **edge)
    ax2.barh(1, fa, left=real, height=h, color="C3", alpha=0.9, **edge)
    ax2.barh(1, quiet, left=real + fa, height=h, color="0.85", **edge)
    ax2.barh(2, frac_real * total, height=h, color="C2", alpha=0.85, **edge)
    ax2.barh(2, (1 - frac_real) * total, left=frac_real * total, height=h, color="C3", alpha=0.9, **edge)
    # how the 125 are stretched to the full width
    for x1, x2 in [(miss, 0), (real, frac_real * total), (real + fa, total)]:
        ax2.plot([x1, x2], [1 + h / 2, 2 - h / 2], ":", color="k", lw=1)
    ax2.plot([real, real], [h / 2, 1 - h / 2], ":", color="k", lw=1)
    ax2.text(real + null / 2, 0, f"{null:.0f} do nothing", ha="center", va="center", fontsize=9.5)
    leader(ax2, (real / 2, -h / 2), (real / 2 + 12, -0.46), f"{real:.0f} real", color="C0", fontsize=9.5, ha="left")
    ax2.text(real + fa + quiet / 2, 1, f"{quiet:.0f} correctly quiet", ha="center", va="center", fontsize=9.5)
    leader(ax2, (miss / 2, 1 - h / 2), (112, 0.53), f"{miss:.0f} missed", color="C0", fontsize=9, card=True)
    leader(ax2, (miss + det / 2, 1 - h / 2), (336, 0.53), f"{det:.0f} detected", color=DARK_GREEN, fontsize=9, card=True)
    leader(ax2, (real + fa / 2, 1 - h / 2), (610, 0.53), f"{fa:.0f} false alarms", color="C3", fontsize=9, card=True)
    ax2.text(frac_real * total / 2, 2, f"{det:.0f} real ({frac_real:.0%})", ha="center", va="center", fontsize=9.5,
             color="white", fontweight="bold")
    ax2.text((1 + frac_real) * total / 2, 2, f"{fa:.0f} false ({1 - frac_real:.0%})", ha="center", va="center",
             fontsize=9.5, color="white", fontweight="bold")
    ax2.set_xlim(0, total); ax2.set_ylim(2.5, -0.65); ax2.grid(False)
    ax2.set_yticks([0, 1, 2]); ax2.set_yticklabels(["all ideas", f"after a test,\npower {power:g},\n$\\alpha = {alpha:g}$",
                                                    f"the {sig:.0f}\nsignificant,\nstretched"])
    ax2.tick_params(axis="y", length=0)
    ax2.set_title(f"(b) If {base:.0%} of ideas are real, a significant\nresult is false {1 - frac_real:.0%} of the time")
    ax2.set_xlabel("number of technique changes (bottom bar: rescaled)"); ax2.set_ylabel("stage of the count")
    fig.tight_layout()
    return fig


def fig_power_as_area():
    """Section 2: power as the area of the alternative beyond the rejection line, for n = 50 and 100; power against n."""
    sigma, delta, alpha = 10.0, 4.0, 0.05
    Phi = stats.norm.cdf
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(0.8)
    check(z_a, 1.960, 5e-4); check(z_b, 0.842, 5e-4)
    mult = check(z_a + z_b, 2.802, 5e-4)
    check(mult ** 2, 7.85, 5e-3)
    se = lambda n: sigma * np.sqrt(2 / n)                       # noqa: E731

    def power(n, d=delta):
        return Phi(d / se(n) - z_a) + Phi(-d / se(n) - z_a)

    check(se(50), 2.0, 1e-12); check(delta / se(50), 2.0, 1e-12)
    check(Phi(0.04), 0.516, 5e-4); check(Phi(-3.96), 0.00004, 5e-6)
    check(power(50), 0.516, 5e-4)
    check(se(100), 1.414, 5e-4); check(delta / se(100), 2.83, 5e-3)
    check(power(100), 0.81, 5e-3)
    n_req = check(2 * mult ** 2 * sigma ** 2 / delta ** 2, 98.1, 0.05)
    n_arm = int(np.ceil(n_req))
    assert n_arm == 99 and power(n_arm) >= 0.8 > power(n_arm - 1)
    mde = check(mult * se(50), 5.6, 5e-3)
    check(power(50, mde), 0.80, 5e-3)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    z = np.sort(np.r_[np.linspace(-4.5, 7, 461), -z_a, z_a])    # the grid hits the rejection lines exactly

    for ax, n, tag in [(ax1, 50, "a"), (ax2, 100, "b")]:
        mu = delta / se(n)
        pw = power(n)
        h0, h1 = stats.norm.pdf(z), stats.norm.pdf(z, mu)
        ax.fill_between(z, h1, where=z >= z_a, color="C3", alpha=0.3, lw=0)
        ax.fill_between(z, h1, where=np.abs(z) <= z_a, fc="none", ec="C3", alpha=0.5, hatch="////", lw=0)
        ax.fill_between(z, h0, where=np.abs(z) >= z_a, color="C0", alpha=0.55, lw=0)
        ax.plot(z, h0, color="C0", lw=1.8)
        ax.plot(z, h1, color="C3", lw=1.8)
        for s in (-z_a, z_a):
            ax.axvline(s, color="k", ls="--", lw=1.1)
        ax.text(0, 0.42, "$H_0$: $\\mathcal{N}(0, 1)$", color="C0", fontsize=9.5, ha="center", va="bottom")
        ax.text(z_a + 0.2, 0.42, f"$H_1$: $\\mathcal{{N}}({mu:.2f}, 1)$", color="C3", fontsize=9.5, ha="left", va="bottom")
        leader(ax, (mu + 0.9, 0.1), (5.5, 0.29), f"power\n$= {pw:.2f}$", color="C3", fontsize=10)
        leader(ax, (1.2, 0.035), (-0.62, 0.135), f"$\\beta$\n$= {1 - pw:.2f}$", color="C3", fontsize=9.5)
        leader(ax, (-2.4, 0.012), (-3.25, 0.135), "$\\alpha/2$\n$= 0.025$", color="C0", fontsize=9)
        for x, txt in [((-4.5 - z_a) / 2, "reject"), (0, "do not reject"), ((7 + z_a) / 2, "reject")]:
            ax.text(x, 0.515, txt, fontsize=9, ha="center", va="center")
        ax.set_xlim(-4.5, 7); ax.set_ylim(0, 0.55)
        ax.set_xticks([-1.96, 0, 1.96, 4, 6]); ax.set_xticklabels(["−1.96", "0", "1.96", "4", "6"], fontsize=9)
        ax.set_title(f"({tag}) $n = {n}$: $\\delta/\\mathrm{{SE}} = {mu:.2f}$, power {pw:.2f}")
        ax.set_xlabel("test statistic $Z$"); ax.set_ylabel("density")

    # (c) the planning view
    ns = np.arange(4, 201)
    ax3.plot(ns, power(ns), color="C0", lw=2, label=f"$\\delta = {delta:g}$ min")
    ax3.plot(ns, power(ns, mde), color="C1", lw=2, ls="--", label=f"$\\delta = {mde:.1f}$ min")
    ax3.axhline(0.8, color=GREY, ls=":", lw=1.2)
    ax3.plot([n_arm, n_arm], [0, power(n_arm)], color=GREY, ls=":", lw=1.2)
    ax3.plot([50, 100], [power(50), power(100)], "o", color="C0", ms=6, zorder=5)
    label(ax3, (50, power(50)), f"{power(50):.2f}", color="C0", offset=(6, -5), va="top", fontsize=9.5)
    label(ax3, (100, power(100)), f"{power(100):.2f}", color="C0", offset=(7, -5), va="top", fontsize=9.5)
    ax3.plot(50, power(50, mde), "s", color="C1", ms=6, zorder=5)
    label(ax3, (50, power(50, mde)), f"{power(50, mde):.2f}", color=DARK_ORANGE, offset=(-2, 4), ha="right", fontsize=9.5)
    label(ax3, (n_arm, 0.17), f"$n = {n_arm}$", offset=(-3, 0), ha="right", va="center", fontsize=10.5)
    ax3.set_xlim(0, 200); ax3.set_ylim(0, 1)
    ax3.legend(loc="lower right", fontsize=8.5)
    ax3.set_title(f"(c) Power against $n$\n($\\sigma = {sigma:.0f}$, $\\alpha = {alpha:g}$)")
    ax3.set_xlabel("$n$ per arm"); ax3.set_ylabel("power")
    fig.tight_layout()
    return fig


def fig_paired_vs_unpaired():
    """Section 3: the same twelve numbers as six paired slopes and as two independent clouds."""
    before = np.array([31.0, 28.5, 35.2, 30.1, 27.8, 33.4])
    after = np.array([29.1, 27.9, 33.0, 29.6, 27.0, 31.5])
    d = before - after
    for got, guide in zip(d, [1.9, 0.6, 2.2, 0.5, 0.8, 1.9]):
        check(got, guide, 1e-9)
    dbar, s_d = check(d.mean(), 1.317, 5e-4), check(d.std(ddof=1), 0.763, 5e-4)
    se_d = check(s_d / np.sqrt(6), 0.311, 5e-4)
    t_d = check(dbar / se_d, 4.23, 5e-3)
    paired = stats.ttest_rel(before, after)
    check(paired.statistic, t_d, 1e-9)
    p_d = check(paired.pvalue, 0.0083, 5e-5)
    welch = stats.ttest_ind(before, after, equal_var=False)
    t_u, p_u = check(welch.statistic, 0.89, 5e-3), check(welch.pvalue, 0.40, 5e-3)
    s_b, s_a = before.std(ddof=1), after.std(ddof=1)
    check(s_b, 2.9, 0.05)                                       # "the machine-to-machine spread (about 2.9 minutes)"
    se_b, se_a = s_b / np.sqrt(6), s_a / np.sqrt(6)
    se_u = check(np.hypot(se_b, se_a), dbar / t_u, 1e-9)
    check(np.corrcoef(before, after)[0, 1], 0.98, 5e-3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)

    # (a) each machine against itself
    for b, a, ch in zip(before, after, -d):
        ax1.plot([0, 1], [b, a], "-o", color="C0", lw=1.6, ms=5.5)
        label(ax1, (1, a), minus(f"{ch:+.1f}"), offset=(8, 0), va="center", fontsize=9.5)
    label(ax1, (1, 34.9), "change\n(minutes)", offset=(8, 0), va="center", fontsize=9, color=GREY_TEXT)
    ax1.annotate("", xy=(-0.2, before.max()), xytext=(-0.2, before.min()),
                 arrowprops=dict(arrowstyle="<->", color=GREY_TEXT, lw=1.1, shrinkA=0, shrinkB=0))
    ax1.text(-0.235, (before.max() + before.min()) / 2, f"between machines: sd about {s_b:.1f}", rotation=90, ha="right",
             va="center", fontsize=9, color=GREY_TEXT)
    ax1.set_xlim(-0.4, 1.6); ax1.set_ylim(26.5, 36)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["before", "after"])
    ax1.grid(axis="x", visible=False)
    ax1.set_title(f"(a) Paired: each machine against itself\n$\\bar d = {dbar:.3f}$, $\\mathrm{{SE}} = {se_d:.3f}$, "
                  f"$t = {t_d:.2f}$, $p = {p_d:.4f}$")
    ax1.set_xlabel("measurement"); ax1.set_ylabel("cycle time (minutes)")

    # (b) the same values with the pairing thrown away
    rng = np.random.default_rng(3)
    jb, ja = rng.uniform(-0.08, 0.08, 6), rng.uniform(-0.08, 0.08, 6)
    ax2.plot(jb, before, "o", color="C0", ms=7, alpha=0.85)
    ax2.plot(1 + ja, after, "s", color="C1", ms=6.5, alpha=0.85)
    ax2.errorbar(0.25, before.mean(), yerr=se_b, fmt="s", color="k", capsize=5, ms=5, lw=1.5)
    ax2.errorbar(1.25, after.mean(), yerr=se_a, fmt="s", color="k", capsize=5, ms=5, lw=1.5)
    label(ax2, (0.25, before.mean()), f"mean ± SE\n{before.mean():.1f} ± {se_b:.2f}", offset=(9, 0), va="center", fontsize=9.5)
    label(ax2, (1.25, after.mean()), f"mean ± SE\n{after.mean():.1f} ± {se_a:.2f}", offset=(9, 0), va="center", fontsize=9.5)
    ax2.set_xlim(-0.4, 1.9)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["before", "after"])
    ax2.tick_params(axis="y", labelleft=True)
    ax2.grid(axis="x", visible=False)
    ax2.set_title(f"(b) Unpaired: two independent samples\n$\\mathrm{{SE}}$ of difference $= {se_u:.2f}$, $t = {t_u:.2f}$, $p = {p_u:.2f}$")
    ax2.set_xlabel("measurement"); ax2.set_ylabel("cycle time (minutes)")
    fig.tight_layout()
    return fig


def fig_permutation_exact():
    """Section 5: all 20 splits of six numbers as a dot plot; how low a permutation p-value can reach for each group size."""
    A, B = (12, 15, 11), (9, 8, 10)
    pool = A + B
    t_obs = check(np.mean(A) - np.mean(B), 3.67, 5e-3)
    check(np.mean(A), 12.67, 5e-3); check(np.mean(B), 9.00, 1e-12)
    diffs = []
    for g in combinations(range(6), 3):
        a = [pool[i] for i in g]
        b = [pool[i] for i in range(6) if i not in g]
        diffs.append(round(np.mean(a) - np.mean(b), 9))
    diffs = np.array(sorted(diffs))
    assert len(diffs) == 20 == comb(6, 3)
    one, two = np.mean(diffs >= t_obs - 1e-9), np.mean(np.abs(diffs) >= t_obs - 1e-9)
    check(one, 0.05, 1e-12); check(two, 0.10, 1e-12)
    assert sum(x > y for x in A for y in B) == 9                # Mann-Whitney U at its maximum
    check(stats.ttest_ind(A, B, equal_var=False).pvalue, 0.074, 5e-4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})

    # (a) stacked dots: one dot per split
    values, counts = np.unique(diffs, return_counts=True)
    assert len(values) == 12 and counts.sum() == 20 and counts.max() == 2
    for v, c in zip(values, counts):
        for y in range(1, c + 1):
            if abs(v - t_obs) < 1e-6:
                ax1.plot(v, y, "o", color="C3", ms=11, zorder=4)
            elif abs(v + t_obs) < 1e-6:
                ax1.plot(v, y, "o", mfc="white", mec="C3", mew=2, ms=10, zorder=4)
            else:
                ax1.plot(v, y, "o", color="C0", ms=10, alpha=0.8, zorder=3)
    leader(ax1, (t_obs, 1.09), (2.6, 2.78), f"observed split, {t_obs:.2f}:\n{{12, 15, 11}} vs {{9, 8, 10}}", color="C3", fontsize=9.5)
    leader(ax1, (-t_obs, 1.09), (-2.6, 2.78), minus(f"its mirror image, -{t_obs:.2f}:\n") + "{9, 8, 10} vs {12, 15, 11}", color="C3", fontsize=9.5)
    ax1.set_xlim(-4.3, 4.3); ax1.set_ylim(0.3, 3.3)
    ax1.set_yticks([1, 2]); ax1.grid(axis="y", visible=False)
    ax1.set_title(f"(a) Difference of means for all {len(diffs)} splits: only {len(values)} values occur;\n"
                  f"one-sided $p = 1/{len(diffs)}$, two-sided $2/{len(diffs)}$")
    ax1.set_xlabel("mean of group $A$ minus mean of group $B$"); ax1.set_ylabel("number of splits")

    # (b) the reach of the p-value: from 1 down to 2 / C(2n, n)
    ns = np.arange(2, 9)
    pmin = np.array([2 / comb(2 * n, n) for n in ns])
    check(pmin[1], 0.10, 1e-12)
    for n, p in zip(ns, pmin):
        reach = p <= 0.05
        ax2.bar(n, 1 - p, bottom=p, width=0.72, color="C2" if reach else "C3", alpha=0.75,
                hatch=None if reach else "///", ec="white", lw=0)
        ax2.annotate(f"{p:.3f}" if p >= 0.1 else f"{p:.3g}", (n, p), textcoords="offset points", xytext=(0, -3),
                     ha="center", va="top", fontsize=8.5)
    ax2.axhline(0.05, color="k", ls="--", lw=1.2)
    label(ax2, (1.5, 0.05), "$\\alpha = 0.05$", offset=(1, 2), ha="left", fontsize=9.5)
    leader(ax2, (3, 0.058), (2.72, 0.0065), f"worked example:\n$n = 3$, $p = 2/{len(diffs)}$", color="C3", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_xlim(1.45, 8.55); ax2.set_ylim(5e-5, 1)
    ax2.set_xticks(ns)
    ax2.grid(axis="x", visible=False)
    ax2.legend(handles=[Patch(fc="C3", alpha=0.75, hatch="///", ec="white", lw=0, label="cannot reach 0.05"),
                        Patch(fc="C2", alpha=0.75, label="can reach 0.05")], loc="lower left", fontsize=9)
    ax2.set_title("(b) How low a two-sided $p$ can reach:\n$2/\\binom{2n}{n}$, which limits every permutation test")
    ax2.set_xlabel("observations per group $n$ (equal groups)")
    ax2.set_ylabel("attainable $p$-values (log scale)")
    fig.tight_layout()
    return fig


def fig_ci_test_duality():
    """Section 6: the interval as the set of theta_0 the test does not reject; TOST as an interval inside a margin."""
    theta, se, margin = 0.5, 0.8, 2.0
    check(stats.norm.ppf(0.975), 1.96, 5e-3); check(stats.norm.ppf(0.95), 1.645, 5e-4)
    z_obs = check(theta / se, 0.625, 1e-12)
    check(2 * stats.norm.sf(z_obs), 0.53, 5e-3)
    ci = lambda z, s: (theta - z * s, theta + z * s)            # noqa: E731
    lo95, hi95 = ci(1.96, se)
    lo90, hi90 = ci(1.645, se)
    check(lo95, -1.07, 5e-3); check(hi95, 2.07, 5e-3)
    check(lo90, -0.82, 5e-3); check(hi90, 1.82, 5e-3)
    check((theta + margin) / se, 3.125, 1e-12); check((margin - theta) / se, 1.875, 1e-12)
    check(stats.norm.sf(3.125), 0.0009, 5e-5); check(stats.norm.sf(1.875), 0.030, 5e-4)
    se2 = 2.0
    check(2 * stats.norm.sf(theta / se2), 0.80, 5e-3)
    check(stats.norm.sf((margin - theta) / se2), 0.23, 5e-3)
    lo90b, hi90b = ci(1.645, se2)
    lo95b, hi95b = ci(1.96, se2)
    check(lo90b, -2.79, 5e-3); check(hi90b, 3.79, 5e-3)
    assert -margin < lo90 and hi90 < margin and not (-margin < lo90b and hi90b < margin)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    # (a) |z| against the hypothesised value: the interval is where the V stays under the threshold
    t0 = np.array([-2.5, theta, 3.5])
    ax1.axvspan(lo95, hi95, color="C0", alpha=0.10, lw=0)
    ax1.axvspan(lo90, hi90, color="C2", alpha=0.14, lw=0)
    ax1.plot(t0, np.abs(theta - t0) / se, color="k", lw=1.8)
    ax1.axhline(1.96, color="C0", ls="--", lw=1.2)
    ax1.axhline(1.645, color="C2", ls=":", lw=1.6)
    ax1.plot([lo95, hi95], [1.96, 1.96], color="C0", lw=3, solid_capstyle="butt")
    ax1.plot([lo90, hi90], [1.645, 1.645], color="C2", lw=3, solid_capstyle="butt")
    label(ax1, (3.5, 1.96), "1.96 (95%)", color="C0", offset=(-2, 3), ha="right", fontsize=9.5)
    label(ax1, (3.5, 1.645), "1.645 (90%)", color=DARK_GREEN, offset=(-2, -3), ha="right", va="top", fontsize=9.5)
    ax1.text(theta, 2.75, minus(f"95% CI [{lo95:.2f}, {hi95:.2f}]"), color="C0", ha="center", va="center", fontsize=10)
    ax1.text(theta, 2.4, minus(f"90% CI [{lo90:.2f}, {hi90:.2f}]"), color=DARK_GREEN, ha="center", va="center", fontsize=10)
    ax1.plot(0, z_obs, "o", color="C3", ms=7, zorder=5)
    boxed(ax1, (0, z_obs), f"$\\theta_0 = 0$: $z = {z_obs:.3f}$,\nnot rejected", color="C3", offset=(-9, 0), ha="right", fontsize=10)
    for x in (-1.82, 2.82):
        ax1.text(x, 0.17, "rejected\nat 5%", color=GREY_TEXT, ha="center", va="center", fontsize=9)
    ax1.set_xlim(-2.5, 3.5); ax1.set_ylim(0, 3.5)
    ax1.set_title("(a) The interval is the set of $\\theta_0$\nthe test does not reject")
    ax1.set_xlabel("hypothesised value $\\theta_0$"); ax1.set_ylabel("$|z|$ for testing $\\theta = \\theta_0$")

    # (b) TOST: the 90% interval against the margin
    ax2.axvspan(-margin, margin, color="C2", alpha=0.12, lw=0)
    ax2.axvline(0, color="k", lw=1)
    rows = [(0, lo90, hi90, lo95, hi95, "C2", DARK_GREEN, "90% CI inside the margin: equivalent"),
            (1, lo90b, hi90b, lo95b, hi95b, "C3", "C3", "90% CI sticks out: inconclusive")]
    for y, l90, h90, l95, h95, col, tcol, verdict in rows:
        ax2.plot([l95, h95], [y, y], color=col, lw=1.4, alpha=0.8, solid_capstyle="butt")
        ax2.plot([l90, h90], [y, y], color=col, lw=5.5, solid_capstyle="butt")
        ax2.plot(theta, y, "o", color="k", ms=6.5, zorder=5)
        boxed(ax2, (0.5, y - 0.26), verdict, color=tcol, fontsize=9.5)
        label(ax2, (0.5, y + 0.1), minus(f"90% [{l90:.2f}, {h90:.2f}]"), color=tcol, offset=(0, -2), ha="center", va="top", fontsize=9.5)
    boxed(ax2, (0, -0.62), f"margin $\\pm\\Delta = \\pm {margin:g}$", color=DARK_GREEN, fontsize=9.5)
    ax2.set_xlim(-4, 5); ax2.set_ylim(1.45, -0.8)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels([f"SE = {se:.1f}", f"SE = {se2:.1f}"])
    ax2.grid(axis="y", visible=False)
    ax2.set_title("(b) TOST: thick = 90% interval, thin = 95%;\nboth cover 0, and only one fits the margin")
    ax2.set_xlabel("difference $\\theta$"); ax2.set_ylabel("standard error of the study")
    fig.tight_layout()
    return fig


def fig_multiple_comparisons():
    """Section 7: ten sorted p-values against the Bonferroni, Holm and BH thresholds; FWER against the number of tests."""
    p = np.array([0.001, 0.011, 0.012, 0.019, 0.040, 0.045, 0.200, 0.350, 0.600, 0.900])
    m, q = len(p), 0.05
    i = np.arange(1, m + 1)
    bonf = q / m
    holm = q / (m + 1 - i)
    bh = q * i / m
    check(bonf, 0.005, 1e-12)
    for got, guide in zip(holm[:2], [0.0050, 0.0056]):           # the table prints Holm's thresholds up to its stop
        check(got, guide, 5e-5)
    for got, guide in zip(bh[:6], [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]):
        check(got, guide, 1e-12)
    k = i[p <= bh].max()
    assert k == 4 and (p <= bonf).sum() == 1 and (p <= 0.05).sum() == 6
    h_stop = np.argmax(p > holm)                                # 0-based position of Holm's first failure
    assert h_stop == 1
    adj = np.minimum.accumulate((m / i * p)[::-1])[::-1]
    for got, guide in zip(adj[:6], [0.010, 0.040, 0.040, 0.0475, 0.075, 0.075]):
        check(got, guide, 1e-9)
    assert p[1] > bh[1]

    ms = np.arange(1, 101)
    for mm, guide, tol in [(5, 0.23, 5e-3), (10, 0.40, 5e-3), (20, 0.64, 5e-3), (100, 0.994, 5e-4)]:
        check(1 - 0.95 ** mm, guide, tol)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) thresholds and sorted p-values; ranks 7-10 are far above the action, so they are triangles with printed values
    ymax = 0.0665
    xs = np.array([0.5, 10.5])
    ax1.plot(xs, q * xs / m, color="C2", lw=2)
    ax1.step(i, holm, where="mid", color="C1", ls="--", lw=1.8)
    ax1.axhline(bonf, color="C1", ls=":", lw=1.8)
    ax1.axhline(q, color=GREY, lw=1.2)
    ax1.plot(i[:k], p[:k], "o", color="C2", ms=7, mec="k", mew=0.7, zorder=5, label="rejected by BH")
    ax1.plot(i[k:6], p[k:6], "o", mfc="white", mec="k", mew=1.2, ms=7, zorder=5, label="not rejected by BH")
    ax1.plot(1, p[0], "o", mfc="none", mec="C1", mew=2, ms=13, zorder=5, label="rejected by all three", clip_on=False)
    ax1.plot(i[6:], np.full(4, ymax - 0.0028), "^", mfc="white", mec="k", mew=1.2, ms=7, zorder=5)
    for r, v in zip(i[6:], p[6:]):
        label(ax1, (r, ymax - 0.0028), f"{v:.3f}", offset=(0, -6), ha="center", va="top", fontsize=8.5)
    label(ax1, (6.55, ymax - 0.0028), "off the scale:", offset=(0, -6), ha="right", va="top", fontsize=8.5)
    ax1.text(8.1, 0.0405 + 0.0012, "BH: $0.05\\,i/10$", color=DARK_GREEN, rotation=np.degrees(np.arctan(q / m)), ha="center",
             va="bottom", fontsize=9.5, rotation_mode="anchor", transform_rotates_text=True)
    ax1.text(10.35, 0.0108, "Holm: $0.05/(11 - i)$", color=DARK_ORANGE, ha="right", va="center", fontsize=9.5)
    ax1.text(10.35, bonf + 0.0005, "Bonferroni: $0.05/10$", color=DARK_ORANGE, ha="right", va="bottom", fontsize=9.5)
    ax1.text(0.6, q + 0.0005, "uncorrected: $0.05$", color=GREY_TEXT, ha="left", va="bottom", fontsize=9.5)
    leader(ax1, (2, p[1] + 0.0007), (1.95, 0.0255), f"$p_{{(2)}} = {p[1]:.3f} > {bh[1]:.3f}$,\nyet swept in by BH", fontsize=9.5)
    leader(ax1, (4.08, p[3]), (5.55, 0.0172), f"$k = {k}$: largest $i$\nwith $p_{{(i)}} \\leq 0.005\\,i$", fontsize=9.5)
    ax1.set_xlim(0.5, 10.5); ax1.set_ylim(0, ymax)
    ax1.set_xticks(i)
    ax1.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0, 1), borderaxespad=0.3)
    ax1.set_title("(a) Ten sorted $p$-values against three thresholds")
    ax1.set_xlabel("rank $i$"); ax1.set_ylabel("$p_{(i)}$ and thresholds")

    # (b) the family-wise error rate
    ax2.plot(ms, 1 - 0.95 ** ms, color="C3", lw=2, label="uncorrected: $1 - 0.95^m$")
    ax2.plot(ms, 1 - (1 - 0.05 / ms) ** ms, color="C1", lw=2, ls="--", label="Bonferroni: $1 - (1 - 0.05/m)^m$")
    for mm, txt in [(5, "0.23"), (10, "0.40"), (20, "0.64"), (100, "0.994")]:
        ax2.plot(mm, 1 - 0.95 ** mm, "o", color="C3", ms=6, zorder=5)
        label(ax2, (mm, 1 - 0.95 ** mm), f"$m = {mm}$: {txt}", color="C3", offset=(7, -3) if mm < 100 else (-3, -7), ha="left" if mm < 100 else "right", va="top", fontsize=9.5)
    ax2.set_xlim(0, 103); ax2.set_ylim(0, 1.05)
    ax2.legend(loc="center right", fontsize=8.5)
    ax2.set_title("(b) Chance of at least one false alarm")
    ax2.set_xlabel("number of independent tests $m$"); ax2.set_ylabel("FWER under a global null")
    fig.tight_layout()
    return fig


def fig_peeking():
    """Section 9: six true-null walks; the false-positive rate against the number of looks; three boundaries for K = 5."""
    z_a = stats.norm.ppf(0.975)
    # exact rates at the nominal 1.96 (the guide quotes Armitage, McPherson and Rowe)
    Ks = np.array([1, 2, 5, 10, 20, 50])
    fpr = np.array([reject_prob(np.full(K, z_a)) for K in Ks])
    for got, guide in zip(fpr, [0.05, 0.08, 0.14, 0.19, 0.25, 0.32]):
        check(got, guide, 5e-3)
    poc = {K: pocock_constant(K) for K in Ks}
    check(poc[5], 2.413, 1e-3); check(poc[10], 2.555, 1e-3); check(poc[20], 2.67, 5e-3)
    fpr_poc = np.array([reject_prob(np.full(K, poc[K])) for K in Ks])
    obf_c = 2.040
    looks = np.arange(1, 6)
    obf = obf_c * np.sqrt(5 / looks)
    check(obf[0], 4.56, 5e-3); check(obf[-1], 2.04, 5e-3)
    check(reject_prob(obf), 0.05, 1e-3)
    bon5 = stats.norm.ppf(1 - 0.05 / (2 * 5))
    check(stats.norm.ppf(1 - 0.05 / (2 * 10)), 2.81, 5e-3)      # the guide's Bonferroni example, K = 10

    # six true-null experiments: each block adds 50 users per arm, so look k has 50 k per arm
    eps = np.random.default_rng(3).standard_normal((6, 20))
    k20 = np.arange(1, 21)
    Z = np.cumsum(eps, axis=1) / np.sqrt(k20)
    crossed = np.abs(Z).max(axis=1) > z_a
    assert crossed.sum() == 2 and np.abs(Z).max() < poc[20]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.2, 1, 1]})

    # (a) the walks
    for path, hit in zip(Z, crossed):
        if not hit:
            ax1.plot(k20, path, color=GREY, lw=1)
    for path in Z[crossed]:
        ax1.plot(k20, path, "-o", color="C3", lw=1.3, ms=3)
        first = np.argmax(np.abs(path) > z_a)
        ax1.plot(k20[first], path[first], "o", mfc="none", mec="k", mew=1.3, ms=10, zorder=6)
    for s in (1, -1):
        ax1.axhline(s * z_a, color="k", ls="--", lw=1.1)
        ax1.axhline(s * poc[20], color="C2", ls=":", lw=1.8)
    label(ax1, (20.3, z_a), "$\\pm 1.96$", offset=(0, 2), ha="right", fontsize=9)
    ax1.text(5.1, 2.3, "circled: naive test stops", va="center", fontsize=8.5)
    label(ax1, (20.3, poc[20]), f"Pocock $\\pm {poc[20]:.2f}$ for $K = 20$", color=DARK_GREEN, offset=(0, 2), ha="right", fontsize=9)
    ax1.set_xlim(0.5, 20.5); ax1.set_ylim(-3.6, 3.6)
    ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.set_title("(a) Six true-null tests,\na look every 50 users per arm")
    ax1.set_xlabel("look $k$ (users per arm = $50k$)"); ax1.set_ylabel("running statistic $Z_k$")

    # (b) the rate against the number of looks
    ax2.plot(Ks, fpr, "-o", color="C3", lw=1.8, ms=6, label="stop at first $|Z_k| > 1.96$")
    ax2.plot(Ks, fpr_poc, "--s", color="C2", lw=1.6, ms=5, label="Pocock boundary per $K$")
    for K, v in zip(Ks, fpr):
        label(ax2, (K, v), f"{v:.2f}", color="C3", offset=(-3, 5) if K > 1 else (4, 6), ha="right" if K > 1 else "center", fontsize=9.5)
    ax2.set_xscale("log")
    ax2.set_xticks(Ks); ax2.set_xticklabels([str(K) for K in Ks]); ax2.minorticks_off()
    ax2.set_xlim(0.8, 65); ax2.set_ylim(0, 0.42)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.set_title("(b) False-positive rate\nagainst number of looks $K$")
    ax2.set_xlabel("number of equally spaced looks $K$"); ax2.set_ylabel("$P_{H_0}$(reject at some look)")

    # (c) the three boundaries for K = 5
    ax3.plot(looks, obf, "-o", color="C3", lw=1.8, ms=6)
    ax3.plot(looks, np.full(5, bon5), "-^", color="C1", lw=1.6, ms=6)
    ax3.plot(looks, np.full(5, poc[5]), "-s", color="C2", lw=1.6, ms=5)
    ax3.axhline(z_a, color="k", ls="--", lw=1.1)
    label(ax3, (1, obf[0]), f"O'Brien–Fleming:\n${obf_c:.3f}\\sqrt{{5/k}}$, from {obf[0]:.2f}\ndown to {obf[-1]:.2f}", color="C3", offset=(18, 5), va="top", fontsize=9.5)
    ax3.text(5.25, bon5 + 0.11, f"Bonferroni: {bon5:.3f}", color=DARK_ORANGE, ha="right", va="bottom", fontsize=9)
    ax3.text(0.85, poc[5] - 0.09, f"Pocock: {poc[5]:.3f}", color=DARK_GREEN, va="top", fontsize=9.5)
    ax3.text(0.85, z_a - 0.05, "single look: 1.96", va="top", fontsize=9.5)
    ax3.set_xlim(0.7, 5.3); ax3.set_ylim(1.5, 5.0)
    ax3.set_xticks(looks)
    ax3.set_title("(c) Boundaries for $K = 5$ looks,\noverall $\\alpha = 0.05$")
    ax3.set_xlabel("look $k$"); ax3.set_ylabel("reject if $|Z_k|$ exceeds")
    fig.tight_layout()
    return fig


def fig_model_comparison():
    """Section 10: McNemar's null on the 20 discordant examples; what J resamples are worth under Nadeau-Bengio."""
    b, c = 15, 5
    nd = b + c
    k = np.arange(0, nd + 1)
    pmf = stats.binom.pmf(k, nd, 0.5)
    chi2 = check((abs(b - c) - 1) ** 2 / nd, 4.05, 1e-12)
    p_chi = check(stats.chi2.sf(chi2, 1), 0.044, 5e-4)
    p_exact = check(2 * stats.binom.cdf(c, nd, 0.5), 0.041, 5e-4)
    check(stats.chi2.sf((b - c) ** 2 / nd, 1), 0.025, 5e-4)     # without the continuity correction

    dbar, s_d, J = 0.004, 0.012, 100
    t_naive = check(dbar / (s_d / np.sqrt(J)), 3.33, 5e-3)
    check(2 * stats.t.sf(t_naive, J - 1), 0.001, 5e-4)
    t_nb = check(dbar / (s_d * np.sqrt(1 / J + 1 / 9)), 0.96, 5e-3)
    check(2 * stats.t.sf(t_nb, J - 1), 0.34, 5e-3)
    j_eff = lambda j, r: 1 / (1 / j + r)                        # noqa: E731
    worth = check(j_eff(J, 1 / 9), 8, 0.5)                      # "worth about eight"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) the discordant examples under the null
    extreme = (k <= c) | (k >= b)
    ax1.bar(k[~extreme], pmf[~extreme], width=0.85, color="C0", alpha=0.8)
    ax1.bar(k[extreme], pmf[extreme], width=0.85, color="C3", alpha=0.9)
    ax1.axvline(c, ymax=0.3, color="C3", ls="--", lw=1.2)
    ax1.axvline(b, ymax=0.3, color="C3", ls="--", lw=1.2)
    ax1.text(-0.4, 0.235, f"observed: B alone right on\n$c = {c}$, A alone right on $b = {b}$;\n"
             f"exact two-sided $p = {p_exact:.3f}$;\n$\\chi^2 = {chi2:.2f}$, $p = {p_chi:.3f}$", ha="left", va="top", fontsize=9, linespacing=1.3)
    ax1.text(20.4, 0.235, "the 980 examples both\nmodels get right or both\nget wrong carry no\ninformation about\nthe difference", ha="right", va="top", fontsize=9, linespacing=1.3, color=GREY_TEXT)
    ax1.set_xlim(-0.7, 20.7); ax1.set_ylim(0, 0.24)
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.grid(axis="x", visible=False)
    ax1.set_title(f"(a) McNemar: the {nd} discordant examples\nunder $H_0$: Binomial({nd}, 1/2)")
    ax1.set_xlabel("discordant examples won by B"); ax1.set_ylabel("probability")

    # (b) what J correlated differences are worth
    j = np.logspace(0, 3, 300)
    ax2.plot(j, j, color=GREY, ls="--", lw=1.3)
    ax2.plot(j, j_eff(j, 1 / 9), color="C0", lw=2)
    ax2.plot(j, j_eff(j, 1 / 4), color="C1", lw=2)
    ax2.plot(J, worth, "o", color="C0", ms=7, zorder=5)
    label(ax2, (J, worth), f"$J = {J}$: worth {worth:.1f}", color="C0", offset=(3, -7), ha="left", va="top", fontsize=9.5)
    label(ax2, (1000, 9), "10-fold: $n_{\\mathrm{test}}/n_{\\mathrm{train}} = 1/9$, ceiling 9", color="C0", offset=(0, 3), ha="right", fontsize=9.5)
    label(ax2, (1000, 4), "5-fold: ratio $1/4$, ceiling 4", color=DARK_ORANGE, offset=(0, 3), ha="right", fontsize=9.5)
    ax2.text(13.5, 11.3, "naive: $J$ independent differences", color=GREY_TEXT, va="center", fontsize=9.5)
    ax2.text(1000, 1.1, "$J_{\\mathrm{eff}} = 1/(1/J + n_{\\mathrm{test}}/n_{\\mathrm{train}})$", ha="right", va="center", fontsize=11)
    ax2.set_xscale("log")
    ax2.set_xticks([1, 10, 100, 1000]); ax2.set_xticklabels(["1", "10", "100", "1000"])
    ax2.set_xlim(1, 1000); ax2.set_ylim(0, 12)
    ax2.set_title("(b) Repeated CV: effective number\nof independent differences")
    ax2.set_xlabel("number of resamples $J$ (log scale)"); ax2.set_ylabel("effective number $J_{\\mathrm{eff}}$")
    fig.tight_layout()
    return fig


FIGURES = {
    "p-value-and-base-rate": fig_p_value_and_base_rate,
    "power-as-area": fig_power_as_area,
    "paired-vs-unpaired": fig_paired_vs_unpaired,
    "permutation-exact": fig_permutation_exact,
    "ci-test-duality": fig_ci_test_duality,
    "multiple-comparisons": fig_multiple_comparisons,
    "peeking": fig_peeking,
    "model-comparison": fig_model_comparison,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 12 concept-guide figures.")
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
