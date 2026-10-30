"""Figures for the Module 05 concept guide — Optimisation.

Writes SVG files into part-2-calculus-and-optimization/05-optimization/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide (recomputed here, not copied), so picture and text agree exactly.

    python tools/builders/figures_05.py                     # (re)write every figure
    python tools/builders/figures_05.py gd-step-size        # only the named figure(s)
    python tools/builders/figures_05.py --preview DIR       # also write PNG previews to DIR
"""
import argparse
import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, GREY  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402

FIG_DIR = ROOT / "part-2-calculus-and-optimization/05-optimization/figures"

WHITE_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)  # lifts text off contour lines

# The ill-conditioned quadratic of Section 3: L = (1/2) theta^T A theta with A = diag(1, 10), start (10, 1)
LAM = np.array([1., 10.])
THETA0 = np.array([10., 1.])


def quad_loss(t1, t2):
    return 0.5 * (LAM[0] * t1 ** 2 + LAM[1] * t2 ** 2)


def quad_gd(eta, steps, start=THETA0):
    """GD on the Section 3 quadratic: each eigen-coordinate is multiplied by (1 - eta lambda_i) per step."""
    th = [np.asarray(start, float)]
    for _ in range(steps):
        th.append(th[-1] - eta * LAM * th[-1])
    return np.array(th)


def quad_heavy_ball(eta, beta, steps, start=THETA0):
    """Heavy-ball momentum v_t = beta v_{t-1} + g_t, theta_{t+1} = theta_t - eta v_t on the same quadratic."""
    th, v = [np.asarray(start, float)], np.zeros(2)
    for _ in range(steps):
        v = beta * v + LAM * th[-1]
        th.append(th[-1] - eta * v)
    return np.array(th)


def contours(ax, f, xlim, ylim, levels, n=400, label_levels=None, manual=None, fmt="%g", fontsize=8):
    """Grey, labelled contour lines of f(x, y) over the window; returns the grid for re-use.

    `label_levels` picks which levels carry a number (default: all, [] for none); `manual`
    gives one (x, y) point per label, near which matplotlib places it; `fontsize` sizes the labels.
    """
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    cs = ax.contour(X, Y, f(X, Y), levels=levels, colors=GREY, linewidths=0.8)
    lv = label_levels if label_levels is not None else levels
    if len(lv):
        ax.clabel(cs, levels=lv, fontsize=fontsize, fmt=fmt, manual=manual if manual is not None else False)
    return X, Y


def boxed(ax, xy, text, color="k", offset=(0, 0), fontsize=10, ha="center", va="center"):
    """Like figbuild.label, but on a translucent white card so it stays readable over lines."""
    ax.annotate(text, xy, textcoords="offset points", xytext=offset, color=color, fontsize=fontsize,
                ha=ha, va=va, bbox=WHITE_BOX, zorder=6)


def steps_to(rate, tol=1e-6):
    """Number of steps for a geometric rate to shrink an error by `tol` (131 for 0.9, 69 for 9/11)."""
    return np.log(tol) / np.log(rate)


def logistic_fit(X, y, eta, steps, ridge):
    """Full-batch GD on the mean logistic loss (+ ridge/2 ||w||^2 off the intercept) from w = 0; returns losses and w."""
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])

    def loss(w):
        z = Xb @ w
        return np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)) + 0.5 * ridge * (w[1:] @ w[1:])

    losses = [loss(w)]
    for _ in range(steps):
        g = Xb.T @ (1 / (1 + np.exp(-(Xb @ w))) - y) / len(y)
        g[1:] += ridge * w[1:]
        w = w - eta * g
        losses.append(loss(w))
    return np.array(losses), w


def breast_cancer():
    data = load_breast_cancer()
    X = data.data
    return X, (X - X.mean(0)) / X.std(0), data.target.astype(float)


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_convexity_tests():
    """Section 2: the chord test (Jensen on x^2), the first-order test (tangents below), and the non-convex x^4 - x^2."""
    f, df = (lambda x: x ** 2), (lambda x: 2 * x)              # noqa: E731
    g, d2g = (lambda x: x ** 4 - x ** 2), (lambda x: 12 * x ** 2 - 2)  # noqa: E731
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9))
    xs = np.linspace(-0.6, 3.6, 400)

    # (a) Jensen with x = 1, y = 3, alpha = 1/2: f(2) = 4 <= (1 + 9)/2 = 5
    x, y, a = 1., 3., 0.5
    xm, chord = a * x + (1 - a) * y, a * f(x) + (1 - a) * f(y)  # 2, 5
    ax1.plot(xs, f(xs), color="C0", lw=2, label="graph $x^2$")
    ax1.plot([x, y], [f(x), f(y)], color="C1", lw=2.2, label="chord")
    ax1.plot([x, y], [f(x), f(y)], "ko", ms=5, zorder=5)
    ax1.plot([xm, xm], [f(xm), chord], color="C3", ls="--", lw=1.4, zorder=4)
    ax1.plot(xm, chord, "s", color="C1", ms=6, zorder=5)
    ax1.plot(xm, f(xm), "o", color="C0", ms=6, zorder=5)
    label(ax1, (x, f(x)), "$f(1) = 1$", offset=(-6, 2), ha="right", fontsize=9)
    label(ax1, (y, f(y)), "$f(3) = 9$", offset=(-7, 0), ha="right", va="center", fontsize=9)
    ax1.annotate(f"at the midpoint $x = {xm:g}$:\ngraph $f(2) = {f(xm):g}$\n"
                 f"$\\leq$ chord $\\frac{{1}}{{2}}(1 + 9) = {chord:g}$",
                 (xm - 0.03, (f(xm) + chord) / 2), xytext=(-0.45, 10.6), ha="left", va="top", fontsize=9.5,
                 color="C3", arrowprops=dict(arrowstyle="-", color="C3", lw=0.8, shrinkB=3))
    ax1.set_xlim(-0.6, 3.6); ax1.set_ylim(-0.4, 11)
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.set_title("(a) Chord test: every chord\nlies above the graph")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$f(x) = x^2$")

    # (b) first-order test: the tangent at x = 1 stays below; the flat tangent at the stationary point x = 0
    x1 = 1.
    tangent = lambda t: f(x1) + df(x1) * (t - x1)               # noqa: E731   y = 2x - 1
    ax2.plot(xs, f(xs), color="C0", lw=2, label="graph $x^2$")
    ax2.plot(xs, tangent(xs), color="C2", lw=2, label="tangent at $x = 1$")
    ax2.plot(xs, np.zeros_like(xs), color="C2", lw=1.6, ls=":", label="flat tangent at $x = 0$")
    ax2.plot(x1, f(x1), "ko", ms=5, zorder=5)
    label(ax2, (x1, f(x1)), "$(1, 1)$", offset=(-5, 3), ha="right", fontsize=9)
    ax2.plot([3, 3], [tangent(3), f(3)], color="C3", ls="--", lw=1.4, zorder=4)
    ax2.plot(3, f(3), "o", color="C0", ms=6, zorder=5)
    ax2.plot(3, tangent(3), "s", color="C2", ms=6, zorder=5)
    ax2.annotate(f"graph $f(3) = {f(3):g}$\n$\\geq$ tangent value\n"
                 f"$f(1) + f'(1)\\cdot 2 = {tangent(3):g}$",
                 (3, tangent(3) - 0.15), xytext=(3.55, 0.3), ha="right", va="bottom", fontsize=9, color="C3",
                 arrowprops=dict(arrowstyle="-", color="C3", lw=0.8, shrinkB=2))
    ax2.set_xlim(-0.6, 3.6); ax2.set_ylim(-0.4, 11)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.set_title("(b) First-order test: every tangent\nlies below the graph")
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$f(x) = x^2$")

    # (c) x^4 - x^2: two minima at +-1/sqrt2, the chord between them passes below the graph at 0
    xg = np.linspace(-1.15, 1.15, 400)
    xmin = 1 / np.sqrt(2)
    edge = np.sqrt(2 / 12)                                       # f'' < 0 for |x| < 1/sqrt6 = 0.408
    assert d2g(0) == -2 and abs(d2g(edge)) < 1e-12
    ax3.axvspan(-edge, edge, color="C3", alpha=0.08, lw=0)
    ax3.plot(xg, g(xg), color="C0", lw=2)
    ax3.plot([-xmin, xmin], [g(xmin), g(xmin)], color="C1", lw=2.2)
    ax3.plot([-xmin, xmin], [g(-xmin), g(xmin)], "ko", ms=5, zorder=5)
    ax3.plot([0, 0], [g(xmin), g(0)], color="C3", ls="--", lw=1.4, zorder=4)
    ax3.plot(0, g(0), "o", mfc="white", mec="k", ms=6, mew=1.5, zorder=5)
    label(ax3, (0, g(0)), "local maximum", offset=(0, 7), ha="center", fontsize=9)
    boxed(ax3, (0, 0.7), f"shaded: $f''(x) = 12x^2 - 2 < 0$\nfor $|x| < 1/\\sqrt{{6}} = {edge:.2f}$", color="C3",
          fontsize=9, va="top")
    ax3.annotate(f"the chord between the two minima\npasses below the graph:\n$f(0) = {g(0):g} > {g(xmin):g}$",
                 (0, g(xmin) - 0.01), xytext=(0, -0.37), ha="center", va="top", fontsize=9, color="C3",
                 arrowprops=dict(arrowstyle="-", color="C3", lw=0.8, shrinkB=2))
    ax3.set_xticks([-xmin, 0, xmin])
    ax3.set_xticklabels(["$-1/\\sqrt{2}$", "$0$", "$1/\\sqrt{2}$"])
    ax3.set_xlim(-1.15, 1.15); ax3.set_ylim(-0.75, 0.75)
    ax3.set_title("(c) Not convex: $f(x) = x^4 - x^2$\nhas two minima")
    ax3.set_xlabel("$x$"); ax3.set_ylabel("$f(x)$")
    fig.tight_layout()
    return fig


def fig_gd_step_size():
    """Section 3: GD paths on diag(1, 10) at eta = 0.10, 0.19, 0.21, and the shrink factor |1 - eta lambda| per direction."""
    mu, L = LAM
    eta_star, rho_star = 2 / (L + mu), (L - mu) / (L + mu)      # 2/11, 9/11
    runs = [(0.10, "C2", "o", "-"), (0.19, "C0", "s", "-"), (0.21, "C3", "^", "--")]
    fig = plt.figure(figsize=(10, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1])
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) the three paths on the elongated bowl (equal aspect: the true shape of the valley)
    xlim, ylim = (-1.5, 11.5), (-2.4, 2.4)
    contours(ax1, quad_loss, xlim, ylim, [2, 8, 18, 32, quad_loss(*THETA0)], label_levels=[2, 8, 18, 32],
             manual=[(1.0, 0.55), (1.4, 1.18), (3.2, 1.6), (5.5, 1.84)], fontsize=9)
    for eta, col, mk, ls in runs:
        p = quad_gd(eta, 8)
        fac = 1 - eta * LAM
        ax1.plot(p[:, 0], p[:, 1], ls=ls, marker=mk, color=col, ms=5, lw=1.4, zorder=4,
                 label=f"$\\eta = {eta:.2f}$: factors $({round(fac[0], 2):g}, {round(fac[1], 2):g})$")
    ax1.plot(*THETA0, "ko", ms=7, zorder=5)
    boxed(ax1, THETA0, f"$\\theta_0 = (10, 1)$, $L(\\theta_0) = {quad_loss(*THETA0):g}$", offset=(4, 10), ha="center", va="bottom",
          fontsize=9.5)
    ax1.plot(0, 0, "k*", ms=12, zorder=5)
    boxed(ax1, (0, 0), "$\\theta^\\star = 0$", offset=(-10, 0), ha="right", fontsize=10)
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim); ax1.set_aspect("equal")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9.5, frameon=False,
               title="8 GD steps for each step size $\\eta$; factors $(1 - \\eta\\lambda_1, 1 - \\eta\\lambda_2)$ per step",
               title_fontsize=9.5)
    ax1.set_title("(a) GD on $L = \\frac{1}{2}(\\theta_1^2 + 10\\,\\theta_2^2)$ from $\\theta_0 = (10, 1)$", pad=44)
    ax1.set_xlabel("$\\theta_1$ (flat direction, $\\lambda = 1$)")
    ax1.set_ylabel("$\\theta_2$ (stiff, $\\lambda = 10$)")

    # (b) the shrink factor of each eigen-direction against the step size
    etas = np.linspace(0, 0.26, 1041)
    flat, stiff = np.abs(1 - etas * mu), np.abs(1 - etas * L)
    ax2.axvspan(2 / L, 0.26, color="C3", alpha=0.07, lw=0)
    ax2.plot(etas, flat, color="C1", lw=2, label="flat: $|1 - \\eta|$")
    ax2.plot(etas, stiff, color="C4", lw=2, label="stiff: $|1 - 10\\eta|$")
    ax2.plot(etas, np.maximum(flat, stiff), "k--", lw=1.3, label="rate $\\rho(\\eta)$ (the larger)")
    ax2.axhline(1, color="C3", ls=":", lw=1.3)
    label(ax2, (0.004, 1), "factor 1: no shrinkage", color="C3", offset=(0, 3), fontsize=9.5)
    ax2.axvline(eta_star, color="k", ls=":", lw=1)
    ax2.axvline(2 / L, color="C3", ls=":", lw=1)
    label(ax2, (2 / L, 0.66), "$2/L = 0.2$:\nunstable beyond", color="C3", offset=(5, 0), va="center", fontsize=9.5)
    for eta, col, mk, _ in runs:
        r = max(abs(1 - eta * mu), abs(1 - eta * L))
        ax2.plot(eta, r, marker=mk, color=col, ms=8, zorder=5, ls="none")
    ax2.plot(eta_star, rho_star, "k*", ms=13, zorder=5)
    r10, r19, r21 = (max(abs(1 - e * mu), abs(1 - e * L)) for e in (0.10, 0.19, 0.21))
    lead = dict(arrowstyle="-", lw=0.8, shrinkB=5)
    boxed(ax2, (0.10, r10), f"$\\eta = 0.10$: $\\rho = {r10:.1f}$, {steps_to(r10):.0f} steps", color="C2",
          offset=(0, -10), va="top", fontsize=9.5)
    ax2.annotate(f"$\\eta^\\star = 2/11 = {eta_star:.3f}$: $\\rho^\\star = 9/11 = {rho_star:.3f}$, "
                 f"{steps_to(rho_star):.0f} steps", (eta_star, rho_star), xytext=(0.168, 1.13), ha="right", va="center",
                 fontsize=9.5, bbox=WHITE_BOX, arrowprops=dict(color="k", **lead))
    ax2.annotate(f"$\\eta = 0.19$: $\\rho = {r19:.1f}$, {steps_to(r19):.0f} steps", (0.19, r19), xytext=(0.192, 1.36),
                 ha="right", va="center", fontsize=9.5, color="C0", bbox=WHITE_BOX, arrowprops=dict(color="C0", **lead))
    boxed(ax2, (0.21, r21), f"$\\eta = 0.21$: $\\rho = {r21:.1f}$,\nthe error grows", color="C3", offset=(8, -4),
          ha="left", va="top", fontsize=9.5)
    ax2.text(0.01, 0.98, "steps = number needed to\nshrink the error by $10^{-6}$", transform=ax2.transAxes,
             ha="left", va="top", fontsize=8.5, color="0.3")
    ax2.set_xlim(0, 0.26); ax2.set_ylim(0, 1.5)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax2.set_title("(b) Shrink factor $|1 - \\eta\\lambda|$ of each eigen-direction per step")
    ax2.set_xlabel("step size $\\eta$"); ax2.set_ylabel("$|1 - \\eta\\lambda|$ per step")
    fig.tight_layout()
    return fig


def fig_feature_scaling():
    """Section 3 climate example and Case study 1: the same problem at kappa = 140, kappa = 1, and raw vs standardised data."""
    years = np.arange(1980, 2021).astype(float)                  # 41 annual means
    n = len(years)
    hess = lambda feat: 2 / n * np.c_[np.ones(n), feat].T @ np.c_[np.ones(n), feat]  # noqa: E731
    H_raw, H_c = hess(years), hess(years - years.mean())         # eigenvalues 7e-5, 8e6; and diag(2, 280)
    H_s = hess((years - years.mean()) / years.std())             # diag(2, 2)
    ev_raw, ev_c, ev_s = (np.linalg.eigvalsh(H) for H in (H_raw, H_c, H_s))
    fig = plt.figure(figsize=(10, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.9], width_ratios=[1, 1.45])
    ax1, ax2, ax3 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    levels = [0.1, 0.3, 0.6, 1.0, 1.5]

    # (a) centred year: excess loss d^T (H/2) d = da^2 + 140 db^2; GD at eta = 1/L kills the slope error, then crawls
    excess_c = lambda da, db: 0.5 * (H_c[0, 0] * da ** 2 + H_c[1, 1] * db ** 2)  # noqa: E731
    xlim, ylim = (-1.3, 1.3), (-0.22, 0.22)
    contours(ax1, excess_c, xlim, ylim, levels, label_levels=[])
    eta = 1 / ev_c[-1]                                           # 1/280
    d = [np.array([1.0, 0.08])]                                  # an illustrative start (the guide gives no temperatures)
    for _ in range(1000):
        d.append(d[-1] - eta * H_c @ d[-1])
    d = np.array(d)
    ax1.plot(d[:2, 0], d[:2, 1], "-", color="C0", lw=1.4, zorder=4)
    ax1.plot(d[1:, 0], d[1:, 1], "-", color="C0", lw=1.4, zorder=3)
    ax1.plot(d[20::20, 0], d[20::20, 1], "|", color="C0", ms=7, mew=1, zorder=4)
    ax1.plot(*d[0], "ko", ms=6, zorder=5)
    for k, off, ha in [(1, (0, -11), "center"), (100, (0, -11), "center"), (300, (0, -11), "center"), (1000, (0, 7), "left")]:
        ax1.plot(*d[k], "o", color="C0", ms=5.5, zorder=5)
        boxed(ax1, d[k], f"step {k}: {d[k, 0]:.3g}", color="C0", offset=off, ha=ha, va="top" if off[1] < 0 else "bottom",
              fontsize=9)
    boxed(ax1, d[0], "start", offset=(0, 8), va="bottom", fontsize=9)
    ax1.plot(0, 0, "k*", ms=11, zorder=6)
    kap_c = ev_c[-1] / ev_c[0]
    boxed(ax1, (-1.27, 0.205), f"$\\kappa = {kap_c:.0f}$: contours {np.sqrt(kap_c):.1f} : 1\n"
          f"tick marks every 20 steps", ha="left", va="top", fontsize=9)
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim); ax1.set_aspect("equal")
    ax1.set_title(f"(a) Centred year: GD at $\\eta = 1/L = 1/{ev_c[-1]:.0f}$ removes the slope error in one step, "
                  "then crawls along the intercept")
    ax1.set_xlabel("intercept error $a - a^\\star$"); ax1.set_ylabel("slope error\n$b - b^\\star$")

    # (b) standardised year: circles, and one step at eta = 1/L = 1/2 lands on the optimum
    excess_s = lambda da, db: 0.5 * (H_s[0, 0] * da ** 2 + H_s[1, 1] * db ** 2)  # noqa: E731
    contours(ax2, excess_s, (-1.4, 1.4), (-1.4, 1.4), levels, label_levels=[0.1, 0.6, 1.5],
             manual=[(-0.22, -0.22), (-0.55, -0.55), (-0.87, -0.87)], fontsize=9)
    s0 = np.array([1.0, 0.8])
    s1 = s0 - (1 / ev_s[-1]) * H_s @ s0                          # (0, 0) up to rounding
    assert np.allclose(s1, 0)
    ax2.annotate("", xy=s1, xytext=s0, zorder=4,
                 arrowprops=dict(arrowstyle="-|>", color="C0", lw=2, shrinkA=4, shrinkB=7, mutation_scale=15))
    ax2.plot(*s0, "ko", ms=6, zorder=5)
    boxed(ax2, s0, "start", offset=(0, 8), va="bottom", fontsize=9)
    ax2.plot(*s1, "k*", ms=11, zorder=6)
    boxed(ax2, s1, "one step lands\non the optimum", color="C0", offset=(8, -8), ha="left", va="top", fontsize=9)
    ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4); ax2.set_aspect("equal")
    ax2.set_title(f"(b) Standardised year: $\\kappa = {ev_s[-1] / ev_s[0]:.0f}$,\n"
                  f"one step at $\\eta = 1/L = 1/{ev_s[-1]:.0f}$")
    ax2.set_xlabel("intercept error $a - a^\\star$"); ax2.set_ylabel("slope error $b - b^\\star$")

    # (c) Case study 1: logistic regression on the breast-cancer data, raw versus standardised features
    X_raw, X_std, y = breast_cancer()
    Xb = np.c_[np.ones(len(y)), X_raw]
    L_raw = np.linalg.eigvalsh(0.25 * Xb.T @ Xb / len(y))[-1]    # 4.2e5 at theta = 0 (all p_i = 1/2)
    steps, ridge = 2000, 1e-2                                    # the ridge term of the case study's Newton run, in both
    loss_raw, w_raw = logistic_fit(X_raw, y, 1 / L_raw, steps, ridge)
    loss_std, w_std = logistic_fit(X_std, y, 0.3, steps, ridge)
    acc = lambda X, w: np.mean((np.c_[np.ones(len(y)), X] @ w > 0) == (y == 1))  # noqa: E731
    k = np.arange(1, steps + 1)
    m, e = f"{1 / L_raw:.1e}".split("e")
    ax3.semilogx(k, loss_raw[1:], color="C3", lw=2, ls="--", label=f"raw features, $\\eta = 1/L = {m}\\times10^{{{int(e)}}}$")
    ax3.semilogx(k, loss_std[1:], color="C0", lw=2, label="standardised features, $\\eta = 0.3$")
    ax3.axhline(np.log(2), color=GREY, ls=":", lw=1.2)
    label(ax3, (1.1, np.log(2)), f"start, $\\theta = 0$: $\\ln 2 = {np.log(2):.3f}$", color="0.35", offset=(0, 3),
          fontsize=9)
    for kk, lv, col in [(steps, loss_raw[-1], "C3"), (200, loss_std[200], "C0"), (steps, loss_std[-1], "C0")]:
        ax3.plot(kk, lv, "o", color=col, ms=5, zorder=5)
    ax3.annotate(f"{loss_raw[-1]:.3f} after 2 000 steps\n({100 * acc(X_raw, w_raw):.1f}% correct)", (steps, loss_raw[-1]),
                 xytext=(2350, 0.53), ha="right", va="bottom", color="C3", fontsize=9,
                 arrowprops=dict(arrowstyle="-", color="C3", lw=0.8, shrinkB=4))
    label(ax3, (200, loss_std[200]), f"{loss_std[200]:.4f} after 200", color="C0", offset=(0, -7), ha="center", va="top",
          fontsize=9)
    label(ax3, (steps, loss_std[-1]), f"{loss_std[-1]:.4f} after 2 000\n({100 * acc(X_std, w_std):.1f}% correct)",
          color="C0", offset=(-4, 7), ha="right", va="bottom", fontsize=9)
    ax3.set_xlim(1, 2600); ax3.set_ylim(0, 0.8)
    ax3.legend(loc="center left", fontsize=8.5, bbox_to_anchor=(0.0, 0.6), framealpha=0.95)
    ax3.set_title("(c) Case study 1: identical data, raw versus\nstandardised features (logistic loss, GD)")
    ax3.set_xlabel("GD step (log scale)"); ax3.set_ylabel("logistic loss")
    fig.tight_layout()
    return fig


def fig_sgd_noise():
    """Section 4: batch means are unbiased with a smaller spread; the noise ball at constant eta; the cosine schedule."""
    x, y, w = np.array([1., 2., 3., 4., 5.]), np.array([1.5, 3.9, 6.2, 8.1, 9.8]), 1.5
    grad_i = lambda w: 2 * x * (w * x - y)                       # noqa: E731   per-example gradients
    g = grad_i(w)                                                # (0, -3.6, -10.2, -16.8, -23.0)
    batches = list(itertools.combinations(range(5), 2))          # all 10 size-2 batches (without replacement)
    bmeans = np.array([g[list(B)].mean() for B in batches])
    full = g.mean()                                              # -10.72 = bmeans.mean()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.1), gridspec_kw={"width_ratios": [1.25, 1.3, 0.95]})

    # (a) strip plot: per-example gradients (row 1) and all ten batch means (row 0)
    ax1.axvline(full, color="C3", lw=1.6, zorder=2)
    ax1.plot(g, np.ones(5), "o", color="C0", ms=7, zorder=4)
    for i, gi in enumerate(g):
        label(ax1, (gi, 1), f"{i + 1}", color="C0", offset=(3, 6) if i == 2 else (0, 6), ha="center", fontsize=9)
    order = np.argsort(bmeans)
    yb = np.zeros(10)
    for a_, b_ in zip(order[:-1], order[1:]):                    # lift a square that would sit on its neighbour
        if bmeans[b_] - bmeans[a_] < 1.0 and yb[a_] == 0:
            yb[b_] = 0.1
    ax1.plot(bmeans, yb, "s", color="C1", ms=6, mfc="white", mew=1.6, zorder=4)
    named = {(0, 3): (-4, 10, "bottom", "left"), (1, 2): (-4, -10, "top", "left"),  # {1,4}, {2,3}: right of the line
             (2, 4): (0, -10, "top", "center")}                                     # {3,5}
    for B, (dx, dy, va, ha_) in named.items():
        m = bmeans[batches.index(B)]
        ax1.plot(m, 0, "s", color="C1", ms=6, zorder=5)
        label(ax1, (m, 0), f"$\\{{{B[0] + 1},{B[1] + 1}\\}}$: ${m:.1f}$", color="C1", offset=(dx, dy), ha=ha_,
              va=va, fontsize=9)
    sd_i, sd_b = g.std(), bmeans.std()                           # 8.41 and 5.15 (population sd over the 5 and the 10)
    for yy, sd, txt in [(0.62, sd_i, f"mean $\\pm$ sd: sd {sd_i:.1f}"),
                        (-0.4, sd_b, f"sd ${sd_b:.2f} = {sd_i:.2f}\\sqrt{{\\frac{{1}}{{2}}\\cdot\\frac{{3}}{{4}}}}$")]:
        ax1.errorbar(full, yy, xerr=sd, color=GREY, capsize=4, lw=1.4, zorder=3)
        boxed(ax1, (full, yy), txt, color="0.3", offset=(0, -6), ha="center", va="top", fontsize=9)  # card hides the line
    boxed(ax1, (full, 1.45), f"full gradient ${full:.2f}$\n= mean of either row", color="C3", fontsize=9)
    ax1.set_yticks([0, 1], ["all 10 batch\nmeans ($b = 2$)", "per-example\n($n = 5$)"])
    ax1.set_xlim(-26, 3); ax1.set_ylim(-0.75, 1.65)
    ax1.grid(axis="y", visible=False)
    ax1.set_title("(a) Batch means are unbiased;\nthe batch size sets the spread")
    ax1.set_xlabel(f"gradient estimate at $w = {w:g}$"); ax1.set_ylabel("estimator")

    # (b) SGD with b = 2 at constant eta hovers in a noise ball around w*; cosine decay from t = 200 shrinks it
    lam = 2 * np.mean(x ** 2)                                    # L'' = 22
    w_star = (x @ y) / (x @ x)                                   # 109.3 / 55 = 1.9873
    gs = grad_i(w_star)
    sigma2 = np.var([gs[list(B)].mean() for B in batches])       # batch-2 gradient variance at w*: 0.480
    eta, T, t_dec = 0.03, 400, 200
    var_pred = eta * sigma2 / (lam * (2 - eta * lam))            # 4.9e-4
    sd_pred = np.sqrt(var_pred)                                  # 0.022
    draws = np.random.default_rng(0).integers(len(batches), size=T)  # the same batch sequence for both runs

    def sgd(decay):
        wt, path = w, [w]
        for t in range(T):
            e = eta if (not decay or t < t_dec) else eta * 0.5 * (1 + np.cos(np.pi * (t - t_dec) / (T - t_dec)))
            wt = wt - e * np.mean(grad_i(wt)[list(batches[draws[t]])])
            path.append(wt)
        return np.array(path)

    w_const, w_cos = sgd(False), sgd(True)
    t = np.arange(T + 1)
    ax2.axhspan(w_star - 2 * sd_pred, w_star + 2 * sd_pred, color="C0", alpha=0.1, lw=0)
    ax2.plot(t, w_const, color="C0", lw=1, label=f"constant $\\eta = {eta:g}$")
    ax2.plot(t[t_dec:], w_cos[t_dec:], color="C1", lw=1.6, label="cosine decay from $t = 200$")
    ax2.axhline(w_star, color="k", ls="--", lw=1, label=f"$w^\\star = {w_star:.3f}$")
    ax2.axvline(t_dec, color=GREY, ls=":", lw=1.2)
    boxed(ax2, (8, 2.066), f"band $w^\\star \\pm 2$ sd; predicted sd\n"
          f"$\\sqrt{{\\eta\\sigma^2 / (\\lambda(2 - \\eta\\lambda))}} = {sd_pred:.3f}$,\n"
          f"measured {w_const[100:].std():.3f}", color="C0", offset=(0, 0), ha="left", va="bottom", fontsize=9)
    ax2.set_xlim(0, T); ax2.set_ylim(1.86, 2.13)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax2.set_title("(b) Constant $\\eta$ hovers in a noise ball;\ndecay shrinks it ($w_0 = 1.5$, below the panel)")
    ax2.set_xlabel("SGD step $t$"); ax2.set_ylabel("$w_t$")

    # (c) the cosine schedule, with and without a linear warm-up over the first 5% of the steps
    s = np.linspace(0, 1, 501)
    cos = 0.5 * (1 + np.cos(np.pi * s))
    w_up = 0.05
    warm = np.where(s < w_up, s / w_up, 0.5 * (1 + np.cos(np.pi * (s - w_up) / (1 - w_up))))
    ax3.plot(s, cos, color="C0", lw=2, label="cosine")
    ax3.plot(s, warm, color="C1", lw=1.6, ls="--", label="warm-up + cosine")
    for q, off, ha in [(0.25, (8, 3), "left"), (0.5, (8, 3), "left"), (0.75, (-5, -5), "right")]:
        v = 0.5 * (1 + np.cos(np.pi * q))
        ax3.plot(q, v, "ko", ms=5, zorder=5)
        frac = {0.25: "T/4", 0.5: "T/2", 0.75: "3T/4"}[q]
        label(ax3, (q, v), f"${frac}$: {100 * v:.0f}%", offset=off, ha=ha, va="top" if off[1] < 0 else "bottom",
              fontsize=9.5)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.3)
    ax3.set_yticks(np.arange(0, 1.01, 0.2))
    ax3.legend(loc="upper right", fontsize=8.5)
    ax3.set_title("(c) Cosine schedule,\nwith and without warm-up")
    ax3.set_xlabel("$t / T$"); ax3.set_ylabel("$\\eta_t / \\eta_{\\max}$")
    fig.tight_layout()
    return fig


def fig_momentum_adam():
    """Section 5: tuned heavy ball versus the best constant GD step; their distance to the optimum; Adam's bias correction."""
    mu, L = LAM
    kappa = L / mu
    eta_gd, rate_gd = 2 / (L + mu), (kappa - 1) / (kappa + 1)    # 2/11, 9/11
    sk = np.sqrt(kappa)
    beta, eta_hb, rate_hb = ((sk - 1) / (sk + 1)) ** 2, (2 / (np.sqrt(L) + np.sqrt(mu))) ** 2, (sk - 1) / (sk + 1)
    fig = plt.figure(figsize=(10, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax1, ax2, ax3 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # (a) the two paths on the Section 3 bowl
    xlim, ylim = (-1.5, 11.5), (-2.4, 2.4)
    contours(ax1, quad_loss, xlim, ylim, [2, 8, 18, 32, quad_loss(*THETA0)], label_levels=[2, 8, 18, 32],
             manual=[(1.0, 0.55), (1.4, 1.18), (3.2, 1.6), (5.5, 1.84)], fontsize=9)
    p_gd, p_hb = quad_gd(eta_gd, 20), quad_heavy_ball(eta_hb, beta, 20)
    ax1.plot(p_gd[:, 0], p_gd[:, 1], "-o", color="C0", ms=4.5, lw=1.2, zorder=4,
             label=f"GD, best constant step $\\eta^\\star = 2/11$ (rate {rate_gd:.3f})")
    ax1.plot(p_hb[:, 0], p_hb[:, 1], "-s", color="C1", ms=4.5, lw=1.4, zorder=4,
             label=f"heavy ball, $\\beta = {beta:.3f}$, $\\eta = {eta_hb:.3f}$ (rate {rate_hb:.3f})")
    ax1.plot(*THETA0, "ko", ms=7, zorder=5)
    boxed(ax1, p_gd[5], "$k = 5$", color="C0", offset=(7, -3), ha="left", va="top", fontsize=9)
    boxed(ax1, p_hb[5], "$k = 5$", color="C1", offset=(0, -8), va="top", fontsize=9)
    boxed(ax1, THETA0, "$\\theta_0 = (10, 1)$", offset=(0, 10), va="bottom", fontsize=9.5)
    ax1.plot(0, 0, "k*", ms=12, zorder=6)
    boxed(ax1, (0, 0), "$\\theta^\\star = 0$", offset=(-10, 0), ha="right", fontsize=10)
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim); ax1.set_aspect("equal")
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.95, title="20 steps each", title_fontsize=9)
    ax1.set_title("(a) Tuned heavy ball versus the best constant GD step on $L = \\frac{1}{2}(\\theta_1^2 + 10\\,\\theta_2^2)$")
    ax1.set_xlabel("$\\theta_1$ (flat direction, $\\lambda = 1$)"); ax1.set_ylabel("$\\theta_2$ (stiff, $\\lambda = 10$)")

    # (b) distance to the optimum on a log axis: two straight lines of different slope
    K = 30
    n_gd = np.linalg.norm(quad_gd(eta_gd, K), axis=1)
    n_hb = np.linalg.norm(quad_heavy_ball(eta_hb, beta, K), axis=1)
    k = np.arange(K + 1)
    ax2.semilogy(k, n_gd, "o", color="C0", ms=4.5, label="GD at $\\eta^\\star$")
    ax2.semilogy(k, n_hb, "s", color="C1", ms=4.5, label="heavy ball (tuned)")
    ax2.semilogy(k, n_gd[0] * rate_gd ** k, ":", color="C0", lw=1.2, label=f"${n_gd[0]:.2f} \\times {rate_gd:.3f}^k$")
    ax2.semilogy(k, n_gd[0] * rate_hb ** k, ":", color="C1", lw=1.2, label=f"${n_gd[0]:.2f} \\times {rate_hb:.3f}^k$")
    first = lambda nrm: int(np.argmax(nrm < 1e-6 * nrm[0]))   # noqa: E731
    n_gd_long = np.linalg.norm(quad_gd(eta_gd, 100), axis=1)
    boxed(ax2, (0.6, 2e-8), f"to shrink $\\|\\theta\\|$ by $10^{{-6}}$:\nGD {first(n_gd_long)} steps; heavy ball "
          f"{steps_to(rate_hb):.0f} by its rate\n({first(n_hb)} in this run): $\\propto\\sqrt{{\\kappa}}$, not $\\kappa$",
          ha="left", va="bottom", fontsize=9)
    ax2.set_xlim(0, K); ax2.set_ylim(1e-8, 3e3)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax2.set_title("(b) Distance to the minimum")
    ax2.set_xlabel("step $k$"); ax2.set_ylabel("$\\|\\theta_k - \\theta^\\star\\|$")

    # (c) Adam under a constant gradient: the uncorrected ratio m_t / sqrt(v_t) against the corrected one
    b1, b2, T = 0.9, 0.999, 100

    def adam_ratios(gval):
        m = v = 0.
        raw, corr = [], []
        for t in range(1, T + 1):
            m = b1 * m + (1 - b1) * gval
            v = b2 * v + (1 - b2) * gval ** 2
            raw.append(m / np.sqrt(v))
            corr.append((m / (1 - b1 ** t)) / np.sqrt(v / (1 - b2 ** t)))
        return np.array(raw), np.array(corr)

    raw_s, corr_s = adam_ratios(0.1)                             # the guide's g = 0.1
    raw_l, _ = adam_ratios(100.)                                 # and g = 100: identical ratios
    t = np.arange(1, T + 1)
    ax3.plot(t, raw_s, color="C3", lw=2, label="no bias correction, $g = 0.1$")
    ax3.plot(t[::5], raw_l[::5], "o", mfc="white", mec="C3", ms=5, mew=1.2, ls="none",
             label="no bias correction, $g = 100$")
    ax3.plot(t, corr_s, color="C0", lw=2, label="with bias correction (any $g$)")
    i = int(np.argmax(raw_s))
    ax3.plot([1, t[i]], [raw_s[0], raw_s[i]], "ko", ms=5, zorder=5)
    label(ax3, (1, raw_s[0]), f"$t = 1$: {raw_s[0]:.2f}", offset=(6, -4), va="top", fontsize=9.5)
    label(ax3, (t[i], raw_s[i]), f"peak {raw_s[i]:.2f} at $t = {t[i]}$", offset=(4, 5), fontsize=9.5)
    label(ax3, (T, raw_s[-1]), f"{raw_s[-1]:.2f} at $t = {T}$", color="C3", offset=(-5, -8), ha="right", va="top",
          fontsize=9.5)
    label(ax3, (T, 1), "step equals $\\eta$ exactly", color="C0", offset=(-5, 4), ha="right", fontsize=9.5)
    ax3.set_xlim(0, T); ax3.set_ylim(0, 8.6)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax3.set_title("(c) Adam's step in units of $\\eta$\nunder a constant gradient $g$")
    ax3.set_xlabel("step $t$"); ax3.set_ylabel("$|m_t| / \\sqrt{v_t}$ (step $/\\, \\eta$)")
    fig.tight_layout()
    return fig


def fig_newton_parabola():
    """Section 6: Newton jumps to the vertex of a fitted parabola; quadratic convergence; from -1 it finds the maximum."""
    f, df, d2f = (lambda x: x ** 3 / 3 - 2 * x), (lambda x: x ** 2 - 2), (lambda x: 2 * x)  # noqa: E731
    model = lambda x0, x: f(x0) + df(x0) * (x - x0) + 0.5 * d2f(x0) * (x - x0) ** 2      # noqa: E731
    newton = lambda x0: x0 - df(x0) / d2f(x0)                                            # noqa: E731
    r2 = np.sqrt(2)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.3), gridspec_kw={"width_ratios": [0.95, 1.25, 0.95]})

    # (a) the models at x0 = 1 and x1 = 1.5 and their vertices
    xs = np.linspace(0.7, 2.0, 300)
    x0 = 1.0
    x1 = newton(x0)                                              # 1.5
    x2 = newton(x1)                                              # 1.41667
    ax1.plot(xs, f(xs), color="C0", lw=2.2, label="$f(x) = x^3/3 - 2x$")
    ax1.plot(xs, model(x0, xs), color="C1", lw=1.6, ls="--", label="parabola at $x_0 = 1$")
    ax1.plot(xs, model(x1, xs), color="C2", lw=1.1, ls="-.", alpha=0.75, label="parabola at $x_1 = 1.5$")
    ax1.plot(x0, f(x0), "ko", ms=6, zorder=5)
    label(ax1, (x0, f(x0)), "$x_0 = 1$", offset=(-6, -2), ha="right", va="top", fontsize=9.5)
    ax1.plot(x1, model(x0, x1), "o", color="C1", ms=6, zorder=5)
    ax1.plot([x1, x1], [model(x0, x1), f(x1)], color="C1", lw=1, ls=":")
    ax1.plot(x1, f(x1), "o", mfc="white", mec="C1", ms=5, mew=1.3, zorder=5)
    label(ax1, (x1, model(x0, x1)), f"vertex $x_1 = {x1:g}$", color="C1", offset=(0, -7), ha="center", va="top",
          fontsize=9.5)
    ax1.plot(r2, f(r2), "k*", ms=11, zorder=6)
    ax1.annotate(f"minimum $\\sqrt{{2}}$\n$= {r2:.5f}$;\n$x_2 = {x2:.5f}$",
                 (r2, f(r2)), xytext=(1.42, -1.585), ha="center", va="top", fontsize=8.5,
                 arrowprops=dict(arrowstyle="-", color="k", lw=0.8, shrinkB=6))
    ax1.set_xlim(0.7, 2.0); ax1.set_ylim(-2.02, -1.38)
    ax1.legend(loc="upper center", fontsize=8)
    ax1.set_title("(a) Each Newton step jumps to the\nvertex of a fitted parabola")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$f(x)$")

    # (b) error on a log axis: Newton's exponent doubles; GD's falls by a constant factor
    xk = [x0]
    for _ in range(4):
        xk.append(newton(xk[-1]))
    err = np.abs(np.array(xk) - r2)                              # 4.1e-1, 8.6e-2, 2.5e-3, 2.1e-6, 1.6e-12
    _, X_std, y = breast_cancer()                                # Case study 1: Newton on the standardised problem
    Xb, ridge = np.c_[np.ones(len(y)), X_std], 1e-2
    wv, gnorms = np.zeros(Xb.shape[1]), []
    for _ in range(9):
        p = 1 / (1 + np.exp(-(Xb @ wv)))
        g = Xb.T @ (p - y) / len(y); g[1:] += ridge * wv[1:]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb / len(y); H[1:, 1:] += ridge * np.eye(Xb.shape[1] - 1)
        gnorms.append(np.linalg.norm(g))
        wv = wv - np.linalg.solve(H, g)
    gnorms = np.array(gnorms)                                    # 1.4, 0.40, 0.15, ..., 1.5e-13
    kk = np.arange(0, 13)
    gd_err = np.linalg.norm(quad_gd(0.10, 12), axis=1)           # Section 3, eta = 1/L: 10 * 0.9^k after step 1
    ax2.semilogy(np.arange(len(err)), err, "-o", color="C0", ms=6, lw=1.6)
    ax2.semilogy(np.arange(len(gnorms)), gnorms, "-s", color="C1", ms=5, lw=1.4)
    ax2.semilogy(kk, gd_err, ":^", color="0.45", ms=4, lw=1.4)
    for k in (2, 3, 4):
        mm, ee = f"{err[k]:.1e}".split("e")
        if k == 2:                                               # beside the point it would sit on the orange curve
            ax2.annotate(f"${mm}\\times10^{{{int(ee)}}}$", (2, err[2]), xytext=(2.6, 4e-4), ha="left", va="center",
                         fontsize=8, color="C0", arrowprops=dict(arrowstyle="-", color="C0", lw=0.7, shrinkB=4))
        else:
            label(ax2, (k, err[k]), f"${mm}\\times10^{{{int(ee)}}}$", color="C0", offset=(6, 0), ha="left",
                  va="bottom", fontsize=8.5)
    label(ax2, (12, gd_err[-1]), "GD, Section 3 quadratic,\n$\\eta = 0.1$: $\\|\\theta_k\\| = 10 \\times 0.9^k$\nfor $k \\geq 1$",
          color="0.35", offset=(0, -8), ha="right", va="top", fontsize=8.5)
    label(ax2, (len(gnorms) - 1, gnorms[-1]), "Newton,\ncase study 1:\n$\\|\\nabla L(\\theta_k)\\|$", color="C1",
          offset=(7, 0), ha="left", va="bottom", fontsize=8.5)
    label(ax2, (0.0, 3e-14), "Newton on\n$x^3/3 - 2x$:\n$|x_k - \\sqrt{2}|$", color="C0", offset=(0, 0), ha="left",
          va="bottom", fontsize=8.5)
    ax2.set_xlim(-0.5, 13.2); ax2.set_ylim(1e-14, 50)
    ax2.set_title("(b) Quadratic convergence:\nthe exponent doubles per step")
    ax2.set_xlabel("iteration $k$"); ax2.set_ylabel("error measure (log scale)")

    # (c) from x0 = -1 the fitted parabola is a hill (f'' = -2 < 0) and Newton jumps to its top
    xs = np.linspace(-2.0, -0.7, 300)
    xm0 = -1.0
    xm1 = newton(xm0)                                            # -1.5
    ax3.plot(xs, f(xs), color="C0", lw=2.2, label="$f(x) = x^3/3 - 2x$")
    ax3.plot(xs, model(xm0, xs), color="C1", lw=1.6, ls="--", label="parabola fitted at $x_0 = -1$")
    ax3.text(-1.42, 1.47, f"$f''(-1) = {d2f(xm0):g} < 0$:\nthe parabola is a hill", color="C1", ha="center",
             va="center", fontsize=9.5)
    ax3.plot(xm0, f(xm0), "ko", ms=6, zorder=5)
    label(ax3, (xm0, f(xm0)), "$x_0 = -1$", offset=(-6, -6), ha="right", va="top", fontsize=9.5)
    ax3.plot(xm1, model(xm0, xm1), "o", color="C1", ms=6, zorder=5)
    ax3.annotate(f"top of the hill: $x_1 = {xm1:g}$", (xm1, model(xm0, xm1)), xytext=(-1.97, 2.065), ha="left",
                 va="center", color="C1", fontsize=9.5, arrowprops=dict(arrowstyle="-", color="C1", lw=0.8, shrinkB=4))
    ax3.plot(-r2, f(-r2), "k*", ms=11, zorder=6)
    label(ax3, (-r2, f(-r2)), "$-\\sqrt{2}$:\nmaximum of $f$", offset=(12, 4), va="bottom", fontsize=9.5)
    ax3.set_xlim(-2.0, -0.7); ax3.set_ylim(1.2, 2.12)
    ax3.legend(loc="lower center", fontsize=8.5)
    ax3.set_title("(c) From $x_0 = -1$ the same\niteration finds the maximum")
    ax3.set_xlabel("$x$"); ax3.set_ylabel("$f(x)$")
    fig.tight_layout()
    return fig


def fig_lagrange_kkt():
    """Section 7 and Case study 3: tangency at x* = (1, 2); KKT in one dimension; the active no-shorting wall."""
    a, b = np.array([1., 2.]), 5.
    x_star = b * a / (a @ a)                                     # (1, 2)
    lam = -2 * b / (a @ a)                                       # -2
    grad_f = lambda x: 2 * x                                     # noqa: E731
    assert np.allclose(grad_f(x_star) + lam * a, 0)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.7), gridspec_kw={"width_ratios": [1.2, 0.85, 1.1]})

    # (a) min ||x||^2 subject to x1 + 2 x2 = 5: the circle f = 5 touches the line at x* = (1, 2)
    th = np.linspace(0, 2 * np.pi, 400)
    for r2_, col, lw in [(1, GREY, 0.8), (5, "C0", 1.8), (10, GREY, 0.8)]:
        ax1.plot(np.sqrt(r2_) * np.cos(th), np.sqrt(r2_) * np.sin(th), color=col, lw=lw)
    for r2_, ang, col in [(1, 4.4, "0.35"), (5, 4.1, "C0"), (10, 3.9, "0.35")]:
        boxed(ax1, (np.sqrt(r2_) * np.cos(ang), np.sqrt(r2_) * np.sin(ang)), f"$f = {r2_}$", color=col, fontsize=8.5)
    xl = np.array([-3.3, 5.8])
    ax1.plot(xl, (b - xl) / a[1], color="k", lw=1.6)
    ang = np.degrees(np.arctan2(-1, 2))                          # the line's slope -1/2, drawn at equal aspect
    ax1.text(4.45, (b - 4.45) / a[1] - 0.36, "$x_1 + 2x_2 = 5$", rotation=ang, rotation_mode="anchor", ha="center",
             va="top", fontsize=9)
    ax1.plot(0, 0, "k+", ms=8)
    arrow(ax1, -0.5 * grad_f(x_star), start=x_star, color="C0", lw=1.8)  # -grad f = (-2, -4), half length: ends at 0
    arrow(ax1, a, start=x_star, color="C1", lw=1.8)
    boxed(ax1, x_star + a, "$\\nabla h = (1, 2)$", color="C1", offset=(5, 0), ha="left", va="center", fontsize=9.5)
    boxed(ax1, x_star - 0.15 * grad_f(x_star), "$-\\nabla f$", color="C0", offset=(-6, 0),
          ha="right", va="center", fontsize=9)
    ax1.plot(*x_star, "*", color="C3", ms=14, zorder=6)
    ax1.annotate(f"$x^\\star = (1, 2)$, $f^\\star = {x_star @ x_star:g}$", x_star, xytext=(-0.6, 4.1), ha="center",
                 va="center", fontsize=9.5, color="C3", bbox=WHITE_BOX,
                 arrowprops=dict(arrowstyle="-", color="C3", lw=0.8, shrinkB=6))
    xq = np.array([3., 1.])                                      # on the line, f = 10: not optimal
    ax1.plot(*xq, "ko", ms=6, zorder=5)
    arrow(ax1, -0.25 * grad_f(xq), start=xq, color="C3", lw=1.5)  # -grad f = (-6, -2), quarter length
    t_dir = np.array([-2., 1.]) / np.sqrt(5)                     # along the line, towards x*
    arrow(ax1, (-grad_f(xq) @ t_dir) * 0.25 * t_dir, start=xq, color="C2", lw=1.5)
    boxed(ax1, xq - 0.25 * grad_f(xq), "$-\\nabla f/4$", color="C3", offset=(0, -4), ha="center", va="top", fontsize=9)
    boxed(ax1, xq, f"$(3, 1)$: $f = {xq @ xq:g}$", offset=(6, 4), ha="left", va="bottom", fontsize=9.5)
    ax1.set_xlim(-3.3, 5.8); ax1.set_ylim(-3.4, 4.6); ax1.set_aspect("equal")
    ax1.set_title("(a) min $\\|x\\|^2$ on $x_1 + 2x_2 = 5$: the\nlevel set touches the line at $x^\\star$")
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")

    # (b) KKT in 1-D: (x - 2)^2 with x <= 1 (active, mu = 2) or x <= 3 (slack, mu = 0)
    f = lambda x: (x - 2) ** 2                                   # noqa: E731
    df = lambda x: 2 * (x - 2)                                   # noqa: E731
    xs = np.linspace(-0.5, 4.5, 300)
    ax2.plot(xs, f(xs), color="C0", lw=2)
    ax2.plot(2, 0, "o", mfc="white", mec="C0", ms=7, mew=1.6, zorder=5)
    label(ax2, (2, 0), "$x = 2$", color="C0", offset=(0, 7), ha="center", fontsize=9.5)
    ax2.axvline(1, color="C2", lw=2, ymax=0.5)
    ax2.axvline(3, color="C2", lw=1.6, ls="--", ymax=0.5)
    mu1 = -df(1)                                                 # stationarity 2(x - 2) + mu = 0 at x = 1: mu = 2
    ax2.plot(1, f(1), "o", color="C3", ms=7, zorder=5)
    arrow(ax2, (0.7, 0), start=(1 - 0.7, f(1)), color="C3", lw=1.6)
    label(ax2, (0.88, f(1)), "$-f'(1)$", color="C3", offset=(0, -5), ha="right", va="top", fontsize=9)
    boxed(ax2, (1.08, 5.0), f"$x \\leq 1$ (solid wall):\nactive, $x^\\star = 1$,\n$\\mu = -f'(1) = {mu1:g} > 0$",
          color="C3", ha="left", va="top", fontsize=9)
    boxed(ax2, (2.92, 3.5), "$x \\leq 3$ (dashed):\nslack, $x^\\star = 2$,\n$g(2) = -1$, $\\mu = 0$",
          color="C2", ha="right", va="top", fontsize=9)
    ax2.set_xlim(-0.5, 4.5); ax2.set_ylim(-0.3, 5.2)
    ax2.set_title("(b) KKT in one dimension:\na touched wall has $\\mu > 0$")
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$f(x) = (x - 2)^2$")

    # (c) Case study 3: portfolio variance over (w1, w2) with w3 = 1 - w1 - w2; the long-only triangle
    vol = np.array([0.20, 0.15, 0.30])
    corr = np.array([[1., 0.9, 0.1], [0.9, 1., 0.1], [0.1, 0.1, 1.]])
    S = np.outer(vol, vol) * corr
    w_unc = np.linalg.solve(S, np.ones(3)); w_unc /= w_unc.sum()            # (-0.47, 1.31, 0.16)
    w23 = np.linalg.solve(S[1:, 1:], np.ones(2)); w_lo = np.r_[0., w23 / w23.sum()]  # (0, 0.826, 0.174)
    lam_p = -2 * (S @ w_lo)[1]                                   # -0.0387
    mu_p = 2 * (S @ w_lo)[0] + lam_p                             # 0.0080 = dV/dw1 along w3 = 1 - w1 - w2
    V = lambda w1, w2: (S[0, 0] * w1 ** 2 + S[1, 1] * w2 ** 2 + S[2, 2] * (1 - w1 - w2) ** 2      # noqa: E731
                        + 2 * S[0, 1] * w1 * w2 + 2 * S[0, 2] * w1 * (1 - w1 - w2) + 2 * S[1, 2] * w2 * (1 - w1 - w2))
    xlim, ylim = (-0.8, 1.15), (-0.2, 1.6)
    ax3.axvspan(xlim[0], 0, color="C3", alpha=0.06, lw=0)
    ax3.fill([0, 1, 0], [0, 0, 1], color="C0", alpha=0.1, lw=0)
    ax3.plot([0, 1, 0, 0], [0, 0, 1, 0], color="C0", lw=1.2)
    levels = [0.018, 0.0194, 0.022, 0.026, 0.032, 0.04, 0.05]
    v_min = w_unc @ S @ w_unc                                    # 0.017488: V grows quadratically along any ray from w_unc
    ray = np.array([1.0, 0.12])
    q = V(*(w_unc[:2] + ray)) - v_min
    lab_levels = [0.026]
    contours(ax3, V, xlim, ylim, levels, label_levels=lab_levels, fmt="%.4g",
             manual=[tuple(w_unc[:2] + np.sqrt((c - v_min) / q) * ray) for c in lab_levels], fontsize=9)
    ax3.plot(*w_unc[:2], "o", color="C3", ms=7, zorder=5)
    boxed(ax3, w_unc[:2], f"unconstrained\nvol {100 * np.sqrt(w_unc @ S @ w_unc):.1f}%", color="C3", offset=(6, -14),
          ha="center", va="top", fontsize=9)
    ax3.plot(*w_lo[:2], "s", color="C1", ms=7, zorder=5)
    boxed(ax3, w_lo[:2], f"long-only\nvol {100 * np.sqrt(w_lo @ S @ w_lo):.1f}%", color="C1", offset=(12, 12),
          ha="left", va="bottom", fontsize=9)
    arrow(ax3, (-0.32, 0), start=w_lo[:2], color="C3", lw=1.5, zorder=4)
    boxed(ax3, (xlim[0] + 0.06, w_lo[1] - 0.06), f"$-\\partial V / \\partial w_1$\n$= -\\mu_1 = -{mu_p:.4f}$",
          color="C3", ha="left", va="top", fontsize=9)
    boxed(ax3, (-0.4, -0.08), "$w_1 < 0$: shorting", color="C3", fontsize=8.5)
    boxed(ax3, (0.3, 0.3), "long-only\n$w_i \\geq 0$", color="C0", fontsize=9)
    ax3.set_xlim(*xlim); ax3.set_ylim(*ylim); ax3.set_aspect("equal")
    ax3.set_title("(c) Case study 3: the no-shorting\nwall $w_1 = 0$ is active")
    ax3.set_xlabel("$w_1$"); ax3.set_ylabel("$w_2$  ($w_3 = 1 - w_1 - w_2$)")
    fig.tight_layout()
    return fig


def fig_diagnostics():
    """Section 8: the Section 3 runs as loss curves; a range test on the same quadratic; early stopping with patience 2."""
    mu, L = LAM
    runs = [(0.10, "C2", "o", "-"), (0.19, "C0", "s", "-"), (0.21, "C3", "^", "--")]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9))

    # (a) L(theta_k) for the three step sizes of Section 3
    K = 30
    k = np.arange(K + 1)
    for eta, col, mk, ls in runs:
        p = quad_gd(eta, K)
        Lk = quad_loss(p[:, 0], p[:, 1])
        ax1.semilogy(k, Lk, ls=ls, marker=mk, color=col, ms=3.5, lw=1.4, label=f"$\\eta = {eta:.2f}$")
    p21 = quad_gd(0.21, K)
    L21 = quad_loss(p21[:, 0], p21[:, 1])
    i = int(np.argmin(L21))
    ax1.annotate(f"$\\eta = 0.21$: lowest at $k = {i}$,\nthen it climbs", (k[i], L21[i]), xytext=(3, 6e3), ha="left",
                 va="center", fontsize=9, color="C3", arrowprops=dict(arrowstyle="-", color="C3", lw=0.8, shrinkB=4))
    ax1.set_xlim(0, K); ax1.set_ylim(1e-3, 1e5)
    ax1.legend(loc="lower left", fontsize=8.5)
    ax1.set_title("(a) The Section 3 runs\nas loss curves")
    ax1.set_xlabel("step $k$"); ax1.set_ylabel("$L(\\theta_k)$")

    # (b) range test (a fresh run of 20 steps at each fixed eta): the loss is lowest near 2/11, climbs past 2/L
    etas = np.logspace(-2, np.log10(0.3), 300)
    L20 = np.array([quad_loss(*quad_gd(e, 20)[-1]) for e in etas])
    L20_at = lambda e: quad_loss(*quad_gd(e, 20)[-1])           # noqa: E731
    ax2.axvspan(0.1, 0.18, color="C2", alpha=0.12, lw=0)
    ax2.loglog(etas, L20, color="C0", lw=2)
    ax2.axvline(2 / L, color="C3", ls=":", lw=1.3)
    ax2.annotate("$2/L = 0.2$", (2 / L, 2e-3), textcoords="offset points", xytext=(3, 0), rotation=90, color="C3",
                 ha="left", va="bottom", fontsize=9)
    for eta, col, mk in [(0.1, "C2", "o"), (2 / (L + mu), "k", "*"), (0.21, "C3", "^")]:
        ax2.plot(eta, L20_at(eta), marker=mk, color=col, ms=8 if mk != "*" else 13, zorder=5)
    e_best = etas[np.argmin(L20)]
    label(ax2, (0.1, L20_at(0.1)), f"$\\eta = 0.1$: {L20_at(0.1):.2f}", color="C2", offset=(-6, 0), ha="right",
          va="center", fontsize=9)
    label(ax2, (2 / (L + mu), L20_at(2 / (L + mu))), f"lowest at $\\eta \\approx {e_best:.3f}$\n($2/11 = {2 / 11:.3f}$)",
          offset=(-8, -2), ha="right", va="top", fontsize=9)
    label(ax2, (0.21, L20_at(0.21)), f"$\\eta = 0.21$: {L20_at(0.21):.0f}", color="C3", offset=(-6, 0), ha="right",
          va="center", fontsize=9)
    boxed(ax2, (0.134, 1.5e3), "start\nhere", color="C2", fontsize=9)
    ax2.set_xticks([0.01, 0.03, 0.1, 0.2], ["0.01", "0.03", "0.1", "0.2"])
    ax2.minorticks_off()
    ax2.set_xlim(0.01, 0.3); ax2.set_ylim(1e-3, 1e4)
    ax2.set_title("(b) Range test on the same quadratic\n(loss after 20 steps at each $\\eta$)")
    ax2.set_xlabel("step size $\\eta$ (log scale)"); ax2.set_ylabel("$L(\\theta_{20})$")

    # (c) early stopping with patience 2 on the guide's validation losses
    val = np.array([0.52, 0.41, 0.36, 0.35, 0.36, 0.37])
    ep = np.arange(1, len(val) + 1)
    patience, best, wait, stop = 2, 0, 0, None
    for j in range(1, len(val)):
        if val[j] < val[best]:
            best, wait = j, 0
        else:
            wait += 1
            if wait == patience:
                stop = j
                break
    ax3.plot(ep, val, "-o", color="C0", ms=6, lw=1.6, zorder=3)
    ax3.axhline(val[best], color=GREY, ls="--", lw=1)
    ax3.plot(ep[best], val[best], "*", color="C2", ms=15, zorder=5)
    label(ax3, (ep[best], val[best]), f"best: epoch {ep[best]}, {val[best]:.2f}", color="C2", offset=(0, -10),
          ha="center", va="top", fontsize=9)
    for n_bad, j in enumerate(range(best + 1, stop + 1), start=1):
        ax3.plot(ep[j], val[j], "o", color="C3", ms=7, zorder=5)
        txt, xy_txt, rp = ((f"no improvement ({n_bad})", (5.8, 0.39), (0.85, 0)) if j < stop else
                           (f"no improvement ({n_bad}):\nstop, restore epoch {ep[best]}", (5.8, 0.44), (1, 0)))
        ax3.annotate(txt, (ep[j], val[j]), xytext=xy_txt, ha="right", va="bottom", fontsize=9, color="C3",
                     arrowprops=dict(arrowstyle="-", color="C3", lw=0.8, shrinkB=5, relpos=rp))
    ax3.axvline(ep[stop], color="C3", ls=":", lw=1.2)
    ax3.set_xticks(ep)
    ax3.set_xlim(0.5, 6.5); ax3.set_ylim(0.30, 0.56)
    ax3.set_title("(c) Early stopping\nwith patience 2")
    ax3.set_xlabel("epoch"); ax3.set_ylabel("validation loss")
    fig.tight_layout()
    return fig


FIGURES = {
    "convexity-tests": fig_convexity_tests,
    "gd-step-size": fig_gd_step_size,
    "feature-scaling": fig_feature_scaling,
    "sgd-noise": fig_sgd_noise,
    "momentum-adam": fig_momentum_adam,
    "newton-parabola": fig_newton_parabola,
    "lagrange-kkt": fig_lagrange_kkt,
    "diagnostics": fig_diagnostics,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 05 concept-guide figures.")
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
