"""Figures for the Module 09 concept guide — Multivariate Probability.

Writes SVG files into part-3-probability/09-multivariate-probability/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, recomputed here and checked against the value the guide prints, so
picture and text agree exactly. The two scatter clouds use fixed seeds; the tail
curves of the copula figure are exact one-dimensional integrals, not simulations.

    python tools/builders/figures_09.py                     # (re)write every figure
    python tools/builders/figures_09.py cov-rectangles      # only the named figure(s)
    python tools/builders/figures_09.py --preview DIR       # also write PNG previews to DIR
"""
import argparse
import pathlib
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, GREY  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from scipy import integrate, special, stats  # noqa: E402

FIG_DIR = ROOT / "part-3-probability/09-multivariate-probability/figures"

WHITE_BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)  # lifts text off lines and fills
CARD = dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", lw=0.8)          # a framed summary card
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


def leader(ax, xy, xytext, text, color="k", fontsize=9.5, ha="left", va="center", lw=0.8):
    """A label placed at `xytext` (data coordinates), joined to the point `xy` by a thin line."""
    ax.annotate(text, xy, xytext=xytext, color=color, fontsize=fontsize, ha=ha, va=va, bbox=WHITE_BOX, zorder=6,
                arrowprops=dict(arrowstyle="-", color=color, lw=lw, shrinkA=0, shrinkB=2))


def ellipse(Sigma, c=1.0, center=(0.0, 0.0), n=361):
    """The level set (x - center)' inv(Sigma) (x - center) = c, as two coordinate arrays."""
    t = np.linspace(0, 2 * np.pi, n)
    pts = np.sqrt(c) * np.linalg.cholesky(np.asarray(Sigma, float)) @ np.vstack([np.cos(t), np.sin(t)])
    return center[0] + pts[0], center[1] + pts[1]


def gauss(x, mean, var):
    """The univariate normal density with the given mean and variance."""
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_cov_rectangles():
    """Section 2: covariance as signed rectangle areas; for Y = X^2 the two areas cancel although Y depends on X."""
    h = np.array([160., 165., 170., 175., 180.])
    w = np.array([55., 62., 66., 70., 82.])
    hb, wb = check(h.mean(), 170, 1e-12), check(w.mean(), 67, 1e-12)
    products = (h - hb) * (w - wb)
    for got, guide in zip(products, (120, 25, 0, 15, 150)):
        check(got, guide, 1e-12)
    total = check(products.sum(), 310, 1e-12)
    ssh, ssw = check(((h - hb) ** 2).sum(), 250, 1e-12), check(((w - wb) ** 2).sum(), 404, 1e-12)
    shw = check(total / (len(h) - 1), 77.5, 1e-12)
    r = check(total / np.sqrt(ssh * ssw), 0.975, 5e-4)

    xs = [Fraction(-1), Fraction(0), Fraction(1)]               # X uniform on {-1, 0, 1}, Y = X^2
    ys = [x * x for x in xs]
    mx, my = sum(xs) / 3, sum(ys) / 3
    assert mx == 0 and my == Fraction(2, 3)
    areas = [(x - mx) * (y - my) for x, y in zip(xs, ys)]
    assert areas == [Fraction(-1, 3), 0, Fraction(1, 3)] and sum(areas) == 0
    p_joint, p_x, p_y = Fraction(1, 3), Fraction(1, 3), Fraction(2, 3)
    assert p_x * p_y == Fraction(2, 9) and p_joint != p_x * p_y

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))

    # (a) five nested rectangles with a common corner at the mean point
    ax1.axvline(hb, color=GREY, ls="--", lw=1)
    ax1.axhline(wb, color=GREY, ls="--", lw=1)
    for hi, wi in zip(h, w):
        ax1.add_patch(Rectangle((hb, wb), hi - hb, wi - wb, fc="C0", alpha=0.16, ec="C0", lw=0.9, zorder=2))
    ax1.plot([hb, hb], [wb, w[2]], color="C0", lw=2.2, zorder=3)  # the third "rectangle" has width 0
    ax1.plot(h, w, "o", color="C0", ms=7, zorder=5)
    ax1.plot(hb, wb, "D", color="k", ms=5, zorder=5)
    label(ax1, (hb, wb), f"mean ({hb:.0f}, {wb:.0f})", offset=(-6, 4), ha="right", fontsize=9.5)
    for (hi, wi), area, off, ha in zip(zip(h, w), products, [(-6, -1), (-6, -1), (7, -7), (6, -1), (6, -1)],
                                       ["right", "right", "left", "left", "left"]):
        text = "0: width 0" if area == 0 else f"$+{area:.0f}$"
        label(ax1, (hi, wi), text, offset=off, ha=ha, va="center", fontsize=10)
    ax1.text(155, 86.5, f"sum of the areas $= {total:.0f}$\n$s_{{hw}} = {total:.0f}/4 = {shw:g}$ cm·kg\n"
             f"$r = {total:.0f}/\\sqrt{{{ssh:.0f} \\times {ssw:.0f}}} = {r:.3f}$", ha="left", va="top", fontsize=9.5, bbox=CARD)
    ax1.text(185.3, 51.5, "a point in this quadrant\nwould subtract its area", ha="right", va="bottom", fontsize=9, color=GREY_TEXT)
    ax1.set_xlim(154, 186); ax1.set_ylim(50, 88)
    ax1.set_title("(a) Height and weight: no area is negative, so\nnothing cancels and the covariance is positive")
    ax1.set_xlabel("height (cm)"); ax1.set_ylabel("weight (kg)")

    # (b) Y = X^2: two opposite areas and a zero-width segment
    grid = np.linspace(-1.5, 1.5, 201)
    ax2.plot(grid, grid ** 2, ":", color=GREY, lw=1.3, label="$y = x^2$")
    ymean = float(my)
    ax2.axvline(0, color=GREY, ls="--", lw=1)
    ax2.axhline(ymean, color=GREY, ls="--", lw=1)
    ax2.add_patch(Rectangle((-1, ymean), 1, 1 - ymean, fc="C1", alpha=0.35, ec="C1", lw=1, hatch="///", zorder=2))
    ax2.add_patch(Rectangle((0, ymean), 1, 1 - ymean, fc="C0", alpha=0.3, ec="C0", lw=1, zorder=2))
    ax2.plot([0, 0], [0, ymean], color="C3", lw=2.2, zorder=3)
    ax2.plot([float(x) for x in xs], [float(y) for y in ys], "o", color="C3", ms=8, zorder=5)
    boxed(ax2, (-0.5, (1 + ymean) / 2), "$(-1)(1/3) = -1/3$", color=DARK_ORANGE, fontsize=9.5)
    boxed(ax2, (0.5, (1 + ymean) / 2), "$(+1)(1/3) = +1/3$", color="C0", fontsize=9.5)
    boxed(ax2, (0, 0.27), "$(0)(-2/3) = 0$: width 0", color="C3", offset=(7, 0), ha="left", fontsize=9.5)
    boxed(ax2, (0, ymean), "mean $(0, 2/3)$", offset=(-6, -5), ha="right", va="top", fontsize=9.5)
    ax2.text(0, -0.28, "sum $= -1/3 + 0 + 1/3 = 0$, so $\\mathrm{Cov}(X, Y) = 0$;\n"
             "but $P(X = 1, Y = 1) = 1/3$, not $P(X = 1)\\,P(Y = 1) = 2/9$", ha="center", va="center", fontsize=9.5, bbox=CARD)
    ax2.set_xlim(-1.6, 1.6); ax2.set_ylim(-0.55, 1.4)
    ax2.legend(loc="upper center", fontsize=9)
    ax2.set_title("(b) $Y = X^2$ for $X$ uniform on $\\{-1, 0, 1\\}$: the areas\ncancel, yet $Y$ is a function of $X$")
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$y = x^2$")
    fig.tight_layout()
    return fig


def fig_gaussian_geometry():
    """Section 4: the ellipses of N(0, [[2,1],[1,1]]) with two points of equal length; x = Lz as a linear map."""
    Sigma = np.array([[2., 1.], [1., 1.]])
    check(np.linalg.det(Sigma), 1, 1e-12)
    Sinv = np.linalg.inv(Sigma)
    check(Sinv[0, 0], 1, 1e-12); check(Sinv[0, 1], -1, 1e-12); check(Sinv[1, 1], 2, 1e-12)
    vals, vecs = np.linalg.eigh(Sigma)
    l2, l1 = check(vals[0], 0.382, 5e-4), check(vals[1], 2.618, 5e-4)
    u1, u2 = vecs[:, 1] * np.sign(vecs[0, 1]), vecs[:, 0] * -np.sign(vecs[0, 0])   # u1 = (+, +), u2 = (-, +)
    check(u1[0], 0.851, 5e-4); check(u1[1], 0.526, 5e-4)
    angle = check(np.degrees(np.arctan2(u1[1], u1[0])), 31.7, 0.05)
    s1, s2 = check(np.sqrt(l1), 1.618, 5e-4), check(np.sqrt(l2), 0.618, 5e-4)
    c95 = check(stats.chi2.ppf(0.95, 2), 5.99, 5e-3)
    check(np.sqrt(5.99 * l1), 3.96, 5e-3); check(np.sqrt(5.99 * l2), 1.51, 5e-3)
    a, b = np.array([1., 1.]), np.array([1., -1.])
    da, db = check(a @ Sinv @ a, 1, 1e-12), check(b @ Sinv @ b, 5, 1e-12)
    fa = check(np.exp(-da / 2) / (2 * np.pi), 0.0965, 5e-5)
    fb = check(np.exp(-db / 2) / (2 * np.pi), 0.0131, 5e-5)
    check(np.linalg.norm(a) - np.linalg.norm(b), 0, 1e-12)
    L = np.linalg.cholesky(Sigma)
    check(L[0, 0], 1.414, 5e-4); check(L[1, 0], 0.707, 5e-4); check(L[1, 1], 0.707, 5e-4)
    z = np.array([0.5, 1.2])
    x = L @ z
    check(x[0], 0.707, 5e-4); check(x[1], 1.202, 5e-4)
    t = np.linspace(0, 2 * np.pi, 361)
    circle = np.vstack([np.cos(t), np.sin(t)])
    image = L @ circle
    assert np.allclose(np.sum(image * (Sinv @ image), axis=0), 1.0)  # L sends the unit circle to the ellipse

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.2))

    # (a) level sets, axes and the two named points
    ax1.plot(*ellipse(Sigma, 1), color="C0", lw=2, label="$\\Delta^2 = 1$: one-sd ellipse")
    ax1.plot(*ellipse(Sigma, c95), color="C0", ls="--", lw=1.5, label="$\\Delta^2 = 5.99$: contains 95%")
    r0 = np.sqrt(2)
    ax1.plot(r0 * circle[0], r0 * circle[1], ":", color=GREY, lw=1.4, label="circle $\\|x\\| = \\sqrt{2}$")
    arrow(ax1, s1 * u1, color="C2", lw=2)
    arrow(ax1, s2 * u2, color="C2", lw=2)
    leader(ax1, s1 * u1, (1.92, -1.25), f"$\\sqrt{{\\lambda_1}}\\,u_1$: length\n{s1:.3f}, at {angle:.1f}°\nto the $x_1$-axis",
           color=DARK_GREEN, fontsize=9.5)
    leader(ax1, s2 * u2, (-4.1, 3.55), f"$\\sqrt{{\\lambda_2}}\\,u_2$: length {s2:.3f}", color=DARK_GREEN, fontsize=9.5)
    ax1.plot(*a, "o", color="C3", ms=8, zorder=5)
    ax1.plot(*b, "s", color="C1", ms=8, zorder=5)
    leader(ax1, a, (0.2, 3.55), f"$(1, 1)$: $\\Delta^2 = {da:.0f}$, $f = {fa:.4f}$", color="C3", fontsize=9.5)
    leader(ax1, b, (0.65, -2.95), f"$(1, -1)$: $\\Delta^2 = {db:.0f}$, $f = {fb:.4f}$", color=DARK_ORANGE, fontsize=9.5)
    ax1.set_xlim(-4.3, 4.3); ax1.set_ylim(-4.3, 4.3); ax1.set_aspect("equal")
    ax1.legend(loc="lower left", fontsize=8.5, borderaxespad=0.3)
    ax1.set_title("(a) Same length $\\sqrt{2}$, very different density:\nthe density sees only the Mahalanobis distance")
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")

    # (b) one picture for both spaces: grey = before the map, colour = after it
    ax2.plot(circle[0], circle[1], "--", color=GREY, lw=1.4)
    arrow(ax2, (1, 0), color=GREY, lw=1.2)
    arrow(ax2, (0, 1), color=GREY, lw=1.2)
    ax2.plot(image[0], image[1], color="C0", lw=2)
    le1, le2 = L[:, 0], L[:, 1]
    arrow(ax2, le1, color="C2", lw=1.3)
    arrow(ax2, le2, color="C1", lw=1.3)
    part1, part2 = z[0] * le1, z[1] * le2
    ax2.plot([0, part1[0]], [0, part1[1]], color="C2", lw=3.5, solid_capstyle="butt", zorder=3)   # 0.5 L e1, tip of the first leg
    arrow(ax2, part2, start=part1, color="C1", lw=2.2)
    ax2.plot(*z, "o", mfc="white", mec=GREY_TEXT, ms=8, mew=1.5, zorder=5)
    ax2.plot(*x, "o", color="C3", ms=8, zorder=5)
    leader(ax2, z, (-1.55, 1.75), f"draw $z = ({z[0]:g}, {z[1]:g})$", color=GREY_TEXT, fontsize=9.5)
    leader(ax2, x, (1.0, 1.75), f"$x = Lz = ({x[0]:.3f}, {x[1]:.3f})$", color="C3", fontsize=9.5)
    leader(ax2, le1, (1.65, 0.3), f"$Le_1 = ({le1[0]:.3f}, {le1[1]:.3f})$", color=DARK_GREEN, fontsize=9.5)
    leader(ax2, le2, (-2.0, 0.95), f"$Le_2 = ({le2[0]:g}, {le2[1]:.3f})$", color=DARK_ORANGE, fontsize=9.5)
    leader(ax2, part1 + part2 * 0.5, (1.65, 0.85), f"$1.2\\,Le_2 = (0, {part2[1]:.3f})$", color=DARK_ORANGE, fontsize=9.5)
    leader(ax2, part1 * 0.5, (0.75, -0.8), f"$0.5\\,Le_1 = ({part1[0]:.3f}, {part1[1]:.3f})$", color=DARK_GREEN, fontsize=9.5)
    handles = [Line2D([], [], ls="--", color=GREY, lw=1.4, label="before: unit circle $\\|z\\| = 1$; grey arrows $e_1$, $e_2$"),
               Line2D([], [], color="C0", lw=2, label="after: its image, the $\\Delta^2 = 1$ ellipse")]
    ax2.legend(handles=handles, loc="lower center", fontsize=8.5, borderaxespad=0.3)
    ax2.set_xlim(-2.15, 3.25); ax2.set_ylim(-2.7, 2.7); ax2.set_aspect("equal")
    ax2.set_title("(b) Sampling $x = Lz$: the map sends the unit circle\nto the ellipse and $e_1, e_2$ to the columns of $L$")
    ax2.set_xlabel("$x_1$ (and $z_1$)"); ax2.set_ylabel("$x_2$ (and $z_2$)")
    fig.tight_layout()
    return fig


def fig_gaussian_conditioning():
    """Section 5: the two regression lines for height and weight; slices of N(0, [[2,1],[1,1]]) keep their width."""
    mu = np.array([170., 70.])
    S = np.array([[100., 72.], [72., 144.]])
    rho = check(S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]), 0.6, 1e-12)
    bw, bh = check(S[0, 1] / S[0, 0], 0.72, 1e-12), check(S[0, 1] / S[1, 1], 0.50, 1e-12)
    check(bw * bh, rho ** 2, 1e-12)
    m_w = check(mu[1] + bw * (185 - mu[0]), 80.8, 1e-12)
    sd_w = check(np.sqrt(S[1, 1] - S[0, 1] ** 2 / S[0, 0]), 9.6, 1e-12)
    m_h = check(mu[0] + bh * (90 - mu[1]), 180, 1e-12)
    sd_h = check(np.sqrt(S[0, 0] - S[0, 1] ** 2 / S[1, 1]), 8, 1e-12)
    z_h = check((185 - mu[0]) / 10, 1.5, 1e-12)
    z_w = check((m_w - mu[1]) / 12, 0.9, 1e-12)
    check(rho * z_h, z_w, 1e-12)

    T = np.array([[2., 1.], [1., 1.]])
    slope = T[0, 1] / T[1, 1]                                   # E[X1 | X2 = x2] = slope * x2
    var_c = check(T[0, 0] - T[0, 1] ** 2 / T[1, 1], 1, 1e-12)
    m_15 = check(slope * 1.5, 1.5, 1e-12)
    m_3 = slope * 3.0
    check(1 - T[0, 1] ** 2 / (T[0, 0] * T[1, 1]), 0.5, 1e-12)   # the factor 1 - rho^2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.9))

    # (a) two regression lines bracket the long axis of the ellipse
    c95 = stats.chi2.ppf(0.95, 2)
    ax1.plot(*ellipse(S, c95, mu), color=GREY, lw=1.2, label="95% ellipse")
    ax1.plot(*ellipse(S, 1, mu), color=GREY, ls="--", lw=1.1, label="one-sd ellipse")
    hh = np.array([145., 195.])
    ax1.plot(hh, mu[1] + bw * (hh - mu[0]), color="C0", lw=2.2, label=f"$\\mathbb{{E}}[W \\mid H]$: slope {bw:.2f} kg/cm")
    ww = np.array([42., 98.])
    ax1.plot(mu[0] + bh * (ww - mu[1]), ww, color="C1", lw=2.2, ls="-.", label=f"$\\mathbb{{E}}[H \\mid W]$: slope {bh:.2f} cm/kg")
    ax1.plot(*mu, "D", color="k", ms=5, zorder=5)
    ax1.errorbar(185, m_w, yerr=sd_w, fmt="o", color="C0", ms=7, capsize=4, lw=1.4, zorder=5)
    ax1.errorbar(m_h, 90, xerr=sd_h, fmt="s", color="C1", ms=7, capsize=4, lw=1.4, zorder=5)
    leader(ax1, (185, m_w - sd_w), (199.4, 50), f"$H = 185$ (1.5 sd):\n$W \\sim \\mathcal{{N}}({m_w:g}, {sd_w:g}^2)$,\nonly {z_w:.1f} sd heavy", color="C0", ha="right", fontsize=9)
    leader(ax1, (m_h + sd_h, 90), (199.4, 108), f"$W = 90$: $H \\sim \\mathcal{{N}}({m_h:.0f}, {sd_h:.0f}^2)$", color=DARK_ORANGE, ha="right", fontsize=9)
    ax1.text(141, 31.5, f"slopes ${bw:.2f} \\times {bh:.2f} = {bw * bh:.2f} = \\rho^2$, not 1", ha="left", va="bottom", fontsize=9.5, bbox=CARD)
    ax1.set_xlim(140, 200); ax1.set_ylim(30, 114)
    ax1.legend(loc="upper left", fontsize=8.5, borderaxespad=0.3)
    ax1.set_title(f"(a) Height and weight, $\\rho = {rho:g}$: two regression\nlines, neither the inverse of the other")
    ax1.set_xlabel("height $H$ (cm)"); ax1.set_ylabel("weight $W$ (kg)")

    # (b) slices of the joint density, each with its conditional bell riding on the slice line
    for c in (1, 4, c95):
        ax2.plot(*ellipse(T, c), color=GREY, lw=0.9)
    x2line = np.array([-3.2, 4.2])
    ax2.plot(slope * x2line, x2line, "--", color="C2", lw=1.4)
    ax2.text(-1.7, -2.75, "$\\mathbb{E}[X_1 \\mid X_2 = x_2] = x_2$", color=DARK_GREEN, ha="left", va="center", fontsize=9.5)
    scale = 1.5
    grid = np.linspace(-4.5, 6.5, 400)
    for x2, m, col, ls, txtcol, name in [(1.5, m_15, "C3", "-", "C3", "1.5"), (3.0, m_3, "C1", "-.", DARK_ORANGE, "3")]:
        ax2.plot(grid, np.full_like(grid, x2), color=col, lw=0.9, ls=":")
        ax2.plot(grid, x2 + scale * gauss(grid, m, var_c), color=col, ls=ls, lw=2)
        ax2.plot(m, x2, "o", color=col, ms=6, zorder=5)
        ax2.annotate("", xy=(m + np.sqrt(var_c), x2 + 0.1), xytext=(m - np.sqrt(var_c), x2 + 0.1),
                     arrowprops=dict(arrowstyle="<->", color="k", lw=1, shrinkA=0, shrinkB=0), zorder=5)
        ax2.text(-4.35, x2 + 0.08, f"$X_1 \\mid X_2 = {name} \\sim \\mathcal{{N}}({m:g}, {var_c:g})$", color=txtcol, ha="left", va="bottom", fontsize=9.5)
    ax2.text(-4.35, 5.25, "grey: contours of the joint density;\nbells: conditional densities, common scale;\n"
             "$\\leftrightarrow$: $\\pm 1$ sd, the same on both", ha="left", va="top", fontsize=8.5, color=GREY_TEXT)
    ax2.set_xlim(-4.5, 6); ax2.set_ylim(-3.2, 5.3); ax2.set_aspect("equal")
    ax2.set_title("(b) Same $\\Sigma$ as Figure 2: observing a different $x_2$\nmoves the mean, not the width")
    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    fig.tight_layout()
    return fig


def fig_total_variance():
    """Section 6: the insurer's mixture, 30 = within 6 + between 24; the within/between split in three examples."""
    pA, pB, mA, mB, vA, vB = 0.6, 0.4, 10., 20., 4., 9.
    mean = check(pA * mA + pB * mB, 14, 1e-12)
    within = check(pA * vA + pB * vB, 6, 1e-12)
    between = check(pA * (mA - mean) ** 2 + pB * (mB - mean) ** 2, 24, 1e-12)
    check(pA * (mA - mean) ** 2, 9.6, 1e-12); check(pB * (mB - mean) ** 2, 14.4, 1e-12)
    total = check(within + between, 30, 1e-12)
    check(between / total, 0.80, 1e-12)
    check(np.sqrt(total), 5.5, 0.05); check(np.sqrt(within), 2.4, 0.05)
    # Gaussian check: Var(X1 | X2) = 1, Var(E[X1 | X2]) = Var(X2) = 1, Sigma_11 = 2
    g_within, g_between = 1.0, 1.0
    check(g_within + g_between, 2, 1e-12)
    cuped_rho2 = 0.5 ** 2                                       # CUPED: rho = 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})

    # (a) two weighted humps and their sum
    x = np.linspace(0, 32, 641)
    cA, cB = pA * gauss(x, mA, vA), pB * gauss(x, mB, vB)
    ax1.fill_between(x, cA, color="C0", alpha=0.2, lw=0)
    ax1.fill_between(x, cB, color="C1", alpha=0.2, lw=0, hatch="///")
    ax1.plot(x, cA, color="C0", lw=1.6, label=f"segment A: ${pA:g} \\times \\mathcal{{N}}({mA:.0f}, {vA:.0f})$")
    ax1.plot(x, cB, color="C1", lw=1.6, ls="-.", label=f"segment B: ${pB:g} \\times \\mathcal{{N}}({mB:.0f}, {vB:.0f})$")
    ax1.plot(x, cA + cB, color="k", lw=1.5, label="mixture: all customers")
    # within: +-1 sd of each hump, drawn at the height of its inflection points
    for m, v, p, name in [(mA, vA, pA, "A"), (mB, vB, pB, "B")]:
        s = np.sqrt(v)
        y = p * gauss(m + s, m, v)
        ax1.annotate("", xy=(m + s, y), xytext=(m - s, y), arrowprops=dict(arrowstyle="<->", color="k", lw=1.1, shrinkA=0, shrinkB=0), zorder=5)
        boxed(ax1, (m, y), f"sd {s:.0f}", offset=(0, 7), fontsize=9.5)
    # between: distance of each hump's centre from the overall mean, below the baseline
    yb = -0.012
    ax1.axhline(0, color="k", lw=0.8)
    for m, peak in [(mA, pA * gauss(mA, mA, vA)), (mB, pB * gauss(mB, mB, vB))]:
        ax1.plot([m, m], [yb, peak], ":", color=GREY, lw=1)
        ax1.annotate("", xy=(m, yb), xytext=(mean, yb), arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.6, shrinkA=0, shrinkB=0), zorder=5)
        boxed(ax1, ((m + mean) / 2, yb), f"${m - mean:+.0f}$", color="C3", offset=(0, -9), fontsize=10)
    ax1.plot([mean, mean], [yb - 0.006, 0.112], "--", color="k", lw=1)
    boxed(ax1, (mean, 0.112), f"$\\mathbb{{E}}[Y] = {mean:.0f}$", offset=(0, 7), fontsize=9.5)
    ax1.text(31.6, 0.083, f"within: ${pA:g}({vA:.0f}) + {pB:g}({vB:.0f}) = {within:.0f}$\n"
             f"between: ${pA:g}(-4)^2 + {pB:g}(+6)^2 = {between:.0f}$\ntotal: $\\mathrm{{Var}}(Y) = {total:.0f}$",
             ha="right", va="center", ma="left", fontsize=9, bbox=CARD)
    ax1.set_xlim(0, 32); ax1.set_ylim(-0.027, 0.15); ax1.set_yticks([0, 0.05, 0.10, 0.15])
    ax1.legend(loc="upper right", fontsize=9, borderaxespad=0.3)
    ax1.set_title(f"(a) Insurer: $\\mathrm{{Var}}(Y) = {total:.0f}$ = within {within:.0f} (spread of each hump)\n"
                  f"+ between {between:.0f} (spread of the hump centres)")
    ax1.set_xlabel("annual claims (thousands)"); ax1.set_ylabel("density (humps scaled by their shares)")

    # (b) the split as shares of the total
    rows = [("insurer\nsegments", within / total, f"{within:.0f}", f"{between:.0f}"),
            ("Gaussian example:\n$X_1$ given $X_2$", g_within / 2, f"{g_within:.0f}", f"{g_between:.0f}"),
            ("CUPED,\n$\\rho = 0.5$", 1 - cuped_rho2, f"${1 - cuped_rho2:g}\\sigma^2$", f"${cuped_rho2:g}\\sigma^2$")]
    ypos = np.arange(len(rows))[::-1]
    for y, (_, sw, tw, tb) in zip(ypos, rows):
        ax2.barh(y, sw, height=0.6, color="C0", alpha=0.45, ec="white", lw=1.5)
        ax2.barh(y, 1 - sw, left=sw, height=0.6, color="C1", alpha=0.45, ec="white", lw=1.5, hatch="///")
        boxed(ax2, (sw / 2, y), tw, fontsize=10)
        boxed(ax2, (sw + (1 - sw) / 2, y), tb, fontsize=10)
        label(ax2, (sw, y + 0.3), f"$R^2 = {1 - sw:.0%}$".replace("%", "\\%"), offset=(0, 2), ha="left", fontsize=9)
    ax2.set_yticks(ypos); ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_xlim(0, 1); ax2.set_ylim(-0.55, 2.8)
    ax2.grid(axis="y", visible=False)
    ax2.legend(handles=[Patch(fc="C0", alpha=0.45, label="within: $\\mathbb{E}[\\mathrm{Var}(Y \\mid X)]$"),
                        Patch(fc="C1", alpha=0.45, hatch="///", label="between: $\\mathrm{Var}(\\mathbb{E}[Y \\mid X])$")],
               loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9, frameon=False, columnspacing=1.0, handlelength=1.5)
    ax2.set_title("(b) How much of the variance knowing $X$\nexplains: $R^2$ = between / total")
    ax2.set_xlabel("share of $\\mathrm{Var}(Y)$"); ax2.set_ylabel("example")
    fig.tight_layout()
    return fig


def fig_simpsons_paradox():
    """Section 7 and case study 2: an aggregate is a point on a segment, placed by the group's own mix; causal roles of S."""
    # treatments: (mild rate, severe rate, share severe)
    a0, a1, ax_, b0, b1, bx = 18 / 20, 48 / 80, 80 / 100, 68 / 80, 10 / 20, 20 / 100
    check(a0, 0.90, 1e-12); check(a1, 0.60, 1e-12); check(b0, 0.85, 1e-12); check(b1, 0.50, 1e-12)
    line = lambda r0, r1, s: r0 * (1 - s) + r1 * s              # noqa: E731
    A, B = check(line(a0, a1, ax_), 0.66, 1e-12), check(line(b0, b1, bx), 0.78, 1e-12)
    As, Bs = check(line(a0, a1, 0.5), 0.75, 1e-12), check(line(b0, b1, 0.5), 0.675, 1e-12)
    # models: (hospital 1 accuracy, hospital 2 accuracy, share from hospital 2)
    ma0, ma1, mb0, mb1 = 0.95, 0.70, 0.93, 0.65
    MA, MB = check(line(ma0, ma1, 0.8), 0.750, 1e-12), check(line(mb0, mb1, 0.2), 0.874, 1e-12)
    MA5, MB5 = check(line(ma0, ma1, 0.5), 0.825, 1e-12), check(line(mb0, mb1, 0.5), 0.790, 1e-12)
    MA3, MB3 = check(line(ma0, ma1, 0.3), 0.875, 1e-12), check(line(mb0, mb1, 0.3), 0.846, 1e-12)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.12, 1.12, 1]})
    fs = 8.5

    # (a) treatments
    xx = np.array([0, 1])
    ax1.plot(xx, [a0, a1], color="C0", lw=2.5, label="treatment A")
    ax1.plot(xx, [b0, b1], color="C1", lw=2.5, ls="-.", label="treatment B")
    ax1.plot(ax_, A, "o", color="C0", ms=9, zorder=5)
    ax1.plot(bx, B, "s", color="C1", ms=8, zorder=5)
    ax1.plot([0.5, 0.5], [0.42, As], ":", color="k", lw=1.2)
    ax1.plot([0.5, 0.5], [As, Bs], "o", mfc="white", mec="k", ms=6, zorder=5, ls="none")
    ax1.text(0.02, 0.965, f"mild (left):\nA {a0:.0%}, B {b0:.0%}", ha="left", va="center", fontsize=fs)
    ax1.text(0.98, 0.455, f"severe (right):\nA {a1:.0%}, B {b1:.0%}", ha="right", va="center", fontsize=fs)
    leader(ax1, (ax_, A), (0.98, 0.79), f"A overall: {A:.0%}\n({ax_:.0%} severe)", color="C0", ha="right", fontsize=fs)
    leader(ax1, (bx, B), (0.02, 0.62), f"B overall: {B:.0%}\n({bx:.0%} severe)", color=DARK_ORANGE, fontsize=fs)
    ax1.text(0.02, 0.455, f"dotted: 50/50 mix,\nA {As:.0%}, B {Bs:.1%}", ha="left", va="center", fontsize=fs)
    ax1.set_xlim(0, 1); ax1.set_ylim(0.4, 1.02); ax1.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax1.legend(loc="upper right", fontsize=fs, borderaxespad=0.3)
    ax1.set_title("(a) A is better at every severity,\nworse overall")
    ax1.set_xlabel("share of the treatment's patients\nwho are severe"); ax1.set_ylabel("recovery rate")

    # (b) models
    ax2.plot(xx, [ma0, ma1], color="C0", lw=2.5, label="model A")
    ax2.plot(xx, [mb0, mb1], color="C1", lw=2.5, ls="-.", label="model B")
    ax2.plot(0.8, MA, "o", color="C0", ms=9, zorder=5)
    ax2.plot(0.2, MB, "s", color="C1", ms=8, zorder=5)
    for s, ya, yb in [(0.5, MA5, MB5), (0.3, MA3, MB3)]:
        ax2.plot([s, s], [0.6, ya], ":", color="k", lw=1.2)
        ax2.plot([s, s], [ya, yb], "o", mfc="white", mec="k", ms=6, zorder=5, ls="none")
    leader(ax2, (0.8, MA), (0.99, 0.84), f"A on its test\nset: {MA:.1%}", color="C0", ha="right", fontsize=fs)
    leader(ax2, (0.2, MB), (0.01, 0.975), f"B on its test set: {MB:.1%}", color=DARK_ORANGE, fontsize=fs)
    ax2.text(0.02, 0.66, f"deployed\n70/30:\nA {MA3:.1%}\nB {MB3:.1%}", ha="left", va="center", fontsize=fs)
    ax2.text(0.525, 0.645, f"50/50 mix:\nA {MA5:.1%}\nB {MB5:.1%}", ha="left", va="center", fontsize=fs)
    ax2.set_xlim(0, 1); ax2.set_ylim(0.6, 1.02); ax2.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax2.legend(loc="upper right", fontsize=fs, borderaxespad=0.3)
    ax2.set_title("(b) Case study 2: the aggregate\nmeasured the test sets")
    ax2.set_xlabel("share of the test records from\nHospital 2 (hard cases)"); ax2.set_ylabel("accuracy")

    # (c) the three causal roles of S, one small graph per row
    def graph(y0, edges, role, verdict):
        pos = {"T": (0.8, y0), "R": (4.2, y0), "S": (2.5, y0 + 1.15)}
        for a, b in edges:
            p, q = np.array(pos[a]), np.array(pos[b])
            ax3.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2, shrinkA=9.5, shrinkB=9.5), zorder=2)
        for name, col in (("T", "C0"), ("R", "C2"), ("S", "C3")):
            ax3.add_patch(Circle(pos[name], 0.42, fc="white", ec=col, lw=1.8, zorder=3))
            ax3.text(*pos[name], f"${name}$", ha="center", va="center", fontsize=10.5, zorder=4)
        ax3.text(5.0, y0 + 0.62, f"{role}\n{verdict}", ha="left", va="center", fontsize=fs)

    graph(7.4, [("S", "T"), ("S", "R"), ("T", "R")], "confounder:", "stratify; the\nstrata are right")
    graph(3.7, [("T", "S"), ("S", "R"), ("T", "R")], "mediator:", "do not stratify; it\nremoves real effect")
    graph(0.0, [("T", "S"), ("R", "S")], "collider:", "do not stratify;\nit invents a link")
    ax3.set_xlim(0, 10); ax3.set_ylim(-0.5, 9.5); ax3.set_aspect("equal")
    ax3.grid(False); ax3.set_xticks([]); ax3.set_yticks([])
    for s in ax3.spines.values():
        s.set_visible(False)
    ax3.set_title("(c) Which table to trust\ndepends on the role of $S$")
    ax3.set_xlabel("arrow = causes;  $T$ treatment,\n$R$ outcome, $S$ stratum variable")
    fig.tight_layout()
    return fig


def fig_kalman_gp_step():
    """Section 8: one Kalman step as four bells; the GP posterior after one observation and its growing error bars."""
    mu0, P0, v, q, r, z = 10., 4., 1., 1., 3., 13.
    mu_m, P_m = check(mu0 + v, 11, 1e-12), check(P0 + q, 5, 1e-12)
    K = check(P_m / (P_m + r), 0.625, 1e-12)
    mu1, P1 = check(mu_m + K * (z - mu_m), 12.25, 1e-12), check((1 - K) * P_m, 1.875, 1e-12)
    check(1 / P_m + 1 / r, 1 / P1, 1e-12)
    sd_m, sd_r, sd_1 = check(np.sqrt(P_m), 2.24, 5e-3), check(np.sqrt(r), 1.73, 5e-3), check(np.sqrt(P1), 1.37, 5e-3)

    ell, sf2, sn2, yobs = 1., 1., 0.1, 1.
    kern = lambda x: sf2 * np.exp(-x ** 2 / (2 * ell ** 2))     # noqa: E731
    mean = lambda x: kern(x) * yobs / (sf2 + sn2)               # noqa: E731
    var = lambda x: np.clip(sf2 - kern(x) ** 2 / (sf2 + sn2), 0, None)   # noqa: E731
    check(kern(0.5), 0.8825, 5e-5); check(kern(2.0), 0.135, 5e-4)
    m05, v05 = check(mean(0.5), 0.802, 5e-4), check(var(0.5), 0.292, 5e-4)
    m2, v2 = check(mean(2.0), 0.123, 5e-4), check(var(2.0), 0.983, 5e-4)
    check(np.sqrt(v05), 0.54, 5e-3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # (a) belief, prediction, sensor, posterior
    x = np.linspace(3, 26, 691)
    ax1.plot(x, gauss(x, mu0, P0), ":", color=GREY_TEXT, lw=1.6, label=f"belief $\\mathcal{{N}}({mu0:.0f}, {P0:.0f})$")
    ax1.plot(x, gauss(x, mu_m, P_m), color="C0", lw=1.8, ls="--", label=f"predicted $\\mathcal{{N}}({mu_m:.0f}, {P_m:.0f})$")
    ax1.plot(x, gauss(x, z, r), color="C1", lw=1.8, ls="-.", label=f"sensor: $z = {z:.0f}$, $r = {r:.0f}$")
    ax1.plot(x, gauss(x, mu1, P1), color="C3", lw=2.6, label=f"posterior $\\mathcal{{N}}({mu1:g}, {P1:g})$")
    ytop = 0.325
    for xv, col in [(mu_m, "C0"), (mu1, "C3"), (z, "C1")]:
        ax1.plot([xv, xv], [0, ytop + 0.008], ":", color=col, lw=1)
    ax1.annotate("", xy=(z, ytop), xytext=(mu_m, ytop), arrowprops=dict(arrowstyle="<->", color="k", lw=1, shrinkA=0, shrinkB=0))
    ax1.annotate(f"$K = {P_m:.0f}/({P_m:.0f} + {r:.0f}) = {K:g}$ of the gap:\n$\\mu = {mu_m:.0f} + {K:g}({z:.0f} - {mu_m:.0f}) = {mu1:g}$", (z, ytop), xytext=(z + 0.4, ytop),
                 ha="left", va="center", fontsize=9, bbox=WHITE_BOX, zorder=6)
    ax1.set_xlim(3, 26); ax1.set_ylim(0, 0.36); ax1.set_xticks(range(5, 26, 5))
    ax1.legend(loc="center right", fontsize=8.5, borderaxespad=0.3, bbox_to_anchor=(1, 0.585))
    ax1.text(25.6, 0.015, f"sd {sd_m:.2f} predicted\nsd {sd_r:.2f} sensor\nsd {sd_1:.2f} posterior\nprecisions add:\n$1/{P_m:.0f} + 1/{r:.0f} = 1/{P1:g}$",
             ha="right", va="bottom", ma="left", fontsize=9, bbox=CARD)
    ax1.set_title(f"(a) One Kalman step: predict $\\mathcal{{N}}({mu0:.0f}, {P0:.0f}) \\to \\mathcal{{N}}({mu_m:.0f}, {P_m:.0f})$,\n"
                  f"then condition on $z = {z:.0f}$: narrower than either")
    ax1.set_xlabel("position"); ax1.set_ylabel("density")

    # (b) the GP posterior against its prior
    g = np.linspace(-4, 4, 401)
    ax2.fill_between(g, -2, 2, color=GREY, alpha=0.15, lw=0, label="prior: $0 \\pm 2\\sigma_f$")
    ax2.axhline(0, color=GREY, lw=0.9)
    sd = np.sqrt(var(g))
    ax2.fill_between(g, mean(g) - 2 * sd, mean(g) + 2 * sd, color="C3", alpha=0.25, lw=0, label="posterior mean $\\pm$ 2 sd")
    ax2.plot(g, mean(g), color="C3", lw=2, label="posterior mean")
    ax2.plot(0, yobs, "ko", ms=7, zorder=6)
    label(ax2, (0, yobs), "$y = 1$ at $x = 0$", offset=(-7, 9), ha="right", fontsize=9.5)
    for xs_, m, vv in [(0.5, m05, v05), (2.0, m2, v2)]:
        ax2.errorbar(xs_, m, yerr=np.sqrt(vv), fmt="s", color="#7f1d1d", ms=5, capsize=4, lw=1.4, zorder=6)
    leader(ax2, (0.5, m05 + np.sqrt(v05)), (3.95, 2.62), f"$x_* = 0.5$: mean ${m05:.3f}$, var ${v05:.3f}$, sd ${np.sqrt(v05):.2f}$", color="#7f1d1d", ha="right", fontsize=9)
    leader(ax2, (2.0, m2 - np.sqrt(v2)), (0.35, -2.85), f"$x_* = 2$: mean ${m2:.3f}$, var ${v2:.3f}$,\nalmost back to the prior", color="#7f1d1d", fontsize=9)
    ax2.set_xlim(-4, 4); ax2.set_ylim(-4.1, 3.1)
    ax2.legend(loc="lower left", fontsize=8.5, borderaxespad=0.3)
    ax2.set_title(f"(b) GP after one observation (RBF, $\\ell = {ell:g}$, $\\sigma_f = {sf2:g}$,\n"
                  f"$\\sigma_n^2 = {sn2:g}$): error bars grow away from the data")
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$f(x)$")
    fig.tight_layout()
    return fig


def lower_tail(q, rho, nu=None):
    """P(Y below its q-quantile | X below its q-quantile) for the Gaussian (nu=None) or t_nu copula, by 1-D quadrature.

    With u = F_X(x), the joint probability is the integral over u in (0, q) of P(Y < a | X = F_X^{-1}(u)),
    so the answer is that conditional probability averaged over u in (0, q), a bounded integrand on a
    finite interval. (Monte-Carlo CDFs such as scipy's multivariate_t.cdf are not accurate this far out.)
    """
    s = np.sqrt(1 - rho ** 2)
    if nu is None:
        a = special.ndtri(q)
        def inner(t):
            x = special.ndtri(q * t)
            return special.ndtr((a - rho * x) / s)
    else:
        a = special.stdtrit(nu, q)
        def inner(t):
            x = special.stdtrit(nu, q * t)                       # Y | X = x is rho x + s sqrt((nu + x^2)/(nu + 1)) t_{nu+1}
            return special.stdtr(nu + 1, (a - rho * x) / (s * np.sqrt((nu + x ** 2) / (nu + 1))))
    return integrate.quad(inner, 0, 1, epsabs=1e-11, epsrel=1e-10, limit=400)[0]


def fig_copula_tails():
    """Section 9: same marginals and correlation under two copulas; the conditional tail probability as q -> 0."""
    rho, nu = 0.5, 3
    n = 1500
    # Seed picked so that this small sample is typical: its corner counts (2 and 5) sit at the exact
    # expectations (1.9 and 4.9, printed in the panels), and many other seeds would misrepresent the tail.
    rng = np.random.default_rng(24)
    gz = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n, method="cholesky")
    w = rng.chisquare(nu, n)
    tz = gz / np.sqrt(w / nu)[:, None]                           # bivariate t_3 ...
    cz = special.ndtri(special.stdtr(nu, tz))                    # ... with its marginals mapped back to N(0, 1)
    r_g, r_t = np.corrcoef(gz.T)[0, 1], np.corrcoef(cz.T)[0, 1]
    assert abs(r_g - r_t) < 0.1                                  # "agree to within a few hundredths" on this sample

    g = {q: lower_tail(q, rho) for q in (0.05, 0.01, 0.001)}
    t = {q: lower_tail(q, rho, nu) for q in (0.05, 0.01, 0.001)}
    check(g[0.05], 0.24, 5e-3); check(g[0.01], 0.13, 5e-3); check(g[0.001], 0.05, 5e-3)
    check(t[0.05], 0.37, 5e-3); check(t[0.01], 0.33, 5e-3); check(t[0.001], 0.32, 5e-3)
    lam = check(2 * special.stdtr(nu + 1, -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))), 0.31, 5e-3)
    ratio = check(t[0.001] / g[0.001], 6, 0.5)
    check(0.01 * g[0.01], 0.0013, 5e-5); check(0.01 * t[0.01], 0.0033, 5e-5)   # self-assessment 10

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    q01 = special.ndtri(0.01)

    for ax, data, col, ttl, rr, joint in [(ax1, gz, "C0", "(a) Gaussian copula, $\\rho = 0.5$", r_g, 0.01 * g[0.01]),
                                          (ax2, cz, "C1", "(b) $t_3$-copula: same marginals,\nsame $\\rho$ parameter", r_t, 0.01 * t[0.01])]:
        ax.scatter(data[:, 0], data[:, 1], s=4, color=col, alpha=0.5, lw=0, zorder=2)
        ax.axvline(q01, color="C3", ls=":", lw=1.3, zorder=3)
        ax.axhline(q01, color="C3", ls=":", lw=1.3, zorder=3)
        inside = (data[:, 0] < q01) & (data[:, 1] < q01)
        ax.add_patch(Rectangle((-4.5, -4.5), q01 + 4.5, q01 + 4.5, fc="C3", alpha=0.1, lw=0, zorder=1))
        ax.scatter(data[inside, 0], data[inside, 1], s=16, color="C3", lw=0, zorder=4)
        ax.text(-4.3, 4.3, f"{n:,} samples, sample $r = {rr:.2f}$", ha="left", va="top", fontsize=8.5, bbox=WHITE_BOX, zorder=6)
        ax.text(4.3, -4.3, f"dotted: 1% quantiles; corner:\n{inside.sum()} points, expected {n * joint:.1f}", ha="right", va="bottom", fontsize=8.5, bbox=WHITE_BOX, zorder=6)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal")
        ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
        ax.set_title(ttl)
        ax.set_xlabel("asset 1 (standard-normal scale)"); ax.set_ylabel("asset 2 (standard-normal scale)")

    # (c) the exact curves
    qs = np.logspace(-4, np.log10(0.5), 40)
    gc = np.array([lower_tail(q, rho) for q in qs])
    tc = np.array([lower_tail(q, rho, nu) for q in qs])
    ax3.plot(qs, gc, color="C0", lw=2, label="Gaussian copula")
    ax3.plot(qs, tc, color="C1", lw=2, ls="-.", label="$t_3$-copula")
    ax3.plot(qs, qs, ":", color=GREY_TEXT, lw=1.3, label="independent: $q$")
    ax3.axhline(lam, color=GREY, ls="--", lw=1)
    label(ax3, (1.1e-4, lam), f"$t_3$ limit {lam:.2f}", color=GREY_TEXT, offset=(0, -3), va="top", fontsize=8.5)
    for q in (0.05, 0.01, 0.001):
        ax3.plot(q, g[q], "o", color="C0", ms=6, zorder=5)
        ax3.plot(q, t[q], "s", color="C1", ms=6, zorder=5)
        label(ax3, (q, g[q]), f"{g[q]:.2f}", color="C0", offset=(4, -4), va="top", fontsize=8.5)
        label(ax3, (q, t[q]), f"{t[q]:.2f}", color=DARK_ORANGE, offset=(0, 5), ha="center", fontsize=8.5)
    ax3.annotate("", xy=(0.001, t[0.001] - 0.012), xytext=(0.001, g[0.001] + 0.015), arrowprops=dict(arrowstyle="<->", color="k", lw=0.9, shrinkA=0, shrinkB=0))
    label(ax3, (0.001, (g[0.001] + t[0.001]) / 2), f"about {ratio:.0f}\ntimes", offset=(-4, 0), ha="right", va="center", fontsize=8.5)
    ax3.set_xscale("log")
    ax3.set_xticks([1e-4, 1e-3, 1e-2, 0.05, 0.25]); ax3.set_xticklabels(["0.01%", "0.1%", "1%", "5%", "25%"])
    ax3.minorticks_off()
    ax3.set_xlim(1e-4, 0.5); ax3.set_ylim(0, 0.72)
    ax3.legend(loc="upper left", fontsize=8.5, borderaxespad=0.3)
    ax3.set_title("(c) $P$(asset 2 below its $q$-quantile\n$\\mid$ asset 1 below its $q$-quantile)")
    ax3.set_xlabel("$q$ (log scale; further into the tail $\\leftarrow$)"); ax3.set_ylabel("conditional probability")
    fig.tight_layout()
    return fig


def fig_mahalanobis_anomaly():
    """Case study 1: rho = 0.9, the 99% Mahalanobis ellipse against a Euclidean circle; readings a and b."""
    rho = 0.9
    S = np.array([[1, rho], [rho, 1]])
    Sinv = np.linalg.inv(S)
    check(1 / (1 - rho ** 2), 5.26, 5e-3)
    a, b = np.array([2., 2.]), np.array([2., -2.])
    check(np.linalg.norm(a), 2.83, 5e-3); check(np.linalg.norm(b), 2.83, 5e-3)
    da, db = check(a @ Sinv @ a, 4.21, 5e-3), check(b @ Sinv @ b, 80.0, 5e-2)
    pa, pb = check(stats.chi2.sf(da, 2), 0.12, 5e-3), stats.chi2.sf(db, 2)
    check(pb * 1e18, 4, 0.5)
    limit = check(stats.chi2.ppf(0.99, 2), 9.21, 5e-3)
    rng = np.random.default_rng(1)
    cloud = rng.multivariate_normal([0, 0], S, size=300, method="cholesky")

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(cloud[:, 0], cloud[:, 1], s=6, color=GREY, alpha=0.5, lw=0, zorder=1, label="300 normal readings")
    ax.plot(*ellipse(S, limit), color="C0", lw=2.2, label=f"Mahalanobis 99%: $\\Delta^2 = {limit:.2f}$")
    t = np.linspace(0, 2 * np.pi, 361)
    ax.plot(2.8 * np.cos(t), 2.8 * np.sin(t), color="C1", ls="--", lw=1.8, label="Euclidean rule: $\\|v\\| = 2.8$")
    ax.plot(*a, "o", color="C2", ms=9, zorder=5)
    ax.plot(*b, "s", color="C3", ms=9, zorder=5)
    leader(ax, a, (-4.6, 3.85), f"$a = (2, 2)$: hot, high pressure;\n$\\Delta^2 = {da:.2f}$, tail {pa:.2f}: passes", color=DARK_GREEN, fontsize=9)
    leader(ax, b, (0.72, -3.8), f"$b = (2, -2)$: hot, low pressure;\n$\\Delta^2 = {db:.1f}$, tail $4 \\times 10^{{-18}}$:\nflagged", color="C3", fontsize=9)
    ax.text(4.68, 4.25, "both at $\\|v\\| = \\sqrt{8} = 2.83$,\njust outside the circle", ha="right", va="center", fontsize=9, color=DARK_ORANGE, bbox=WHITE_BOX, zorder=6)
    ax.set_xlim(-4.8, 4.8); ax.set_ylim(-4.8, 4.8); ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9, borderaxespad=0.3)
    ax.set_title(f"$\\rho = {rho:g}$: the same Euclidean distance,\nMahalanobis $\\Delta^2 = {da:.2f}$ versus ${db:.1f}$")
    ax.set_xlabel("temperature ($z$-score)"); ax.set_ylabel("pressure ($z$-score)")
    fig.tight_layout()
    return fig


FIGURES = {
    "cov-rectangles": fig_cov_rectangles,
    "gaussian-geometry": fig_gaussian_geometry,
    "gaussian-conditioning": fig_gaussian_conditioning,
    "total-variance": fig_total_variance,
    "simpsons-paradox": fig_simpsons_paradox,
    "kalman-gp-step": fig_kalman_gp_step,
    "copula-tails": fig_copula_tails,
    "mahalanobis-anomaly": fig_mahalanobis_anomaly,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 09 concept-guide figures.")
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
