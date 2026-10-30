"""Figures for the Module 04 concept guide — Differential Calculus for ML.

Writes SVG files into part-2-calculus-and-optimization/04-differential-calculus/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, so picture and text agree exactly.

    python tools/builders/figures_04.py                     # (re)write every figure
    python tools/builders/figures_04.py chain-rule-graph    # only the named figure(s)
    python tools/builders/figures_04.py --preview DIR       # also write PNG previews to DIR
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

FIG_DIR = ROOT / "part-2-calculus-and-optimization/04-differential-calculus/figures"

WHITE_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)  # lifts text off contour lines


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def contours(ax, f, xlim, ylim, levels, n=400, label_levels=None, manual=None, upright=False):
    """Grey, labelled contour lines of f(x, y) over the window; returns the grid for re-use.

    `label_levels` picks which levels carry a number (default: all); `manual` gives one
    (x, y) point per label, near which matplotlib places it, so labels avoid the drawing.
    `upright` keeps the labels horizontal on a white card, so a minus sign cannot vanish
    into a dashed (negative-level) contour that runs along it.
    """
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    cs = ax.contour(X, Y, f(X, Y), levels=levels, colors=GREY, linewidths=0.8)
    texts = ax.clabel(cs, levels=label_levels if label_levels is not None else levels, fontsize=8, fmt="%g",
                      manual=manual if manual is not None else False)
    if upright:
        for t in texts:
            t.set_rotation(0)
            t.set_bbox(WHITE_BOX)
    return X, Y


def boxed(ax, xy, text, color="k", offset=(0, 0), fontsize=10, ha="center", va="center"):
    """Like figbuild.label, but on a translucent white card so it stays readable over contour lines."""
    ax.annotate(text, xy, textcoords="offset points", xytext=offset, color=color, fontsize=fontsize,
                ha=ha, va=va, bbox=WHITE_BOX, zorder=6)


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_tangent_and_step_size():
    """Section 1: the tangent as a marginal rate on the E_max curve; the V-shaped error of finite differences."""
    E_max, ED50 = 100., 20.
    E = lambda D: E_max * D / (ED50 + D)                        # noqa: E731
    dE = lambda D: E_max * ED50 / (ED50 + D) ** 2               # noqa: E731
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))

    # (a) dose-effect curve, tangents at 20 mg (slope 1.25) and 80 mg (slope 0.2); overshoot at 30 mg
    D = np.linspace(0, 100, 401)
    ax1.plot(D, E(D), color="C0", lw=2, label=r"$E(D) = 100D/(20 + D)$")
    for D0, span, col in [(20., (0., 48.), "C3"), (80., (45., 100.), "C2")]:
        Ds = np.array(span)
        ax1.plot(Ds, E(D0) + dE(D0) * (Ds - D0), color=col, ls="--", lw=1.6,
                 label=f"tangent at {D0:g} mg, slope $E'({D0:g}) = {dE(D0):g}$")
    ax1.plot([20, 80], [E(20), E(80)], "ko", ms=6, zorder=4)
    label(ax1, (20, E(20)), f"$E(20) = {E(20):g}$", offset=(8, -6), ha="left", va="top")
    label(ax1, (80, E(80)), f"$E(80) = {E(80):g}$", offset=(-6, 6), ha="right")
    pred = E(20) + dE(20) * 10                                   # 62.5: what the tangent predicts at 30 mg
    ax1.plot([30, 30], [E(30), pred], color="C3", lw=2.5, zorder=3)
    ax1.plot(30, E(30), "o", color="C0", ms=6, zorder=4)
    ax1.plot(30, pred, "o", color="C3", ms=6, zorder=4)
    label(ax1, (30, pred), f"tangent predicts {pred:g} (gain 12.5)\ncurve gives $E(30) = {E(30):g}$ (gain 10)",
          offset=(12, -8), va="top", fontsize=10)
    # the 2.5-point overshoot is only a few pixels tall at full size: enlarge it in an inset (empty top-left corner)
    axins = ax1.inset_axes([0.03, 0.60, 0.22, 0.30])            # top at 0.90: room for the title below the frame
    Dz = np.linspace(27.5, 32.5, 51)
    axins.plot(Dz, E(Dz), color="C0", lw=2)
    axins.plot(Dz, E(20) + dE(20) * (Dz - 20), color="C3", ls="--", lw=1.6)
    axins.plot([30, 30], [E(30), pred], color="C3", lw=2.5, zorder=3)
    axins.plot(30, E(30), "o", color="C0", ms=6, zorder=4)
    axins.plot(30, pred, "o", color="C3", ms=6, zorder=4)
    axins.set_xlim(27.5, 32.5); axins.set_ylim(57, 64)
    axins.set_xticks([30]); axins.set_yticks([60, 62.5]); axins.set_yticklabels(["60", "62.5"])
    axins.yaxis.tick_right()                                     # tick labels clear of the main y-axis labels
    axins.tick_params(labelsize=8)
    axins.set_title("zoom at 30 mg", fontsize=8, pad=3)
    ax1.indicate_inset_zoom(axins, edgecolor=GREY)
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 100)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title("(a) The derivative is a marginal rate: the tangent predicts gains")
    ax1.set_xlabel("dose $D$ (mg)"); ax1.set_ylabel("effect $E$ (% of maximum)")

    # (b) measured error of both schemes for f = x^3 at x = 2 against the truncation and round-off lines
    hs = np.logspace(-14, -1, 60)                                # deterministic sweep; h = 0.1 is the last point
    f, x0, exact, eps = (lambda x: x ** 3), 2.0, 12.0, 2.2e-16
    fwd = np.abs((f(x0 + hs) - f(x0)) / hs - exact)
    cen = np.abs((f(x0 + hs) - f(x0 - hs)) / (2 * hs) - exact)
    ax2.loglog(hs, fwd, "o-", color="C1", ms=3, lw=1, label="forward difference")
    ax2.loglog(hs, cen, "s-", color="C0", ms=3, lw=1, label="central difference")
    big, mid = hs >= 3e-9, hs >= 1e-7                            # truncation lines only where they matter (clear of the legend)
    ax2.loglog(hs[big], 6 * hs[big], "--", color="C1", lw=1.2, label=r"truncation $6h$")
    ax2.loglog(hs[mid], hs[mid] ** 2, "--", color="C0", lw=1.2, label=r"truncation $h^2$")
    ax2.loglog(hs, eps * abs(f(x0)) / hs, ":", color="k", lw=1.2, label=r"round-off $\varepsilon|f(2)|/h$")
    for err in (fwd[-1], cen[-1]):
        ax2.plot(hs[-1], err, "o", ms=8, mfc="white", mec="k", mew=1.8, zorder=5)
    label(ax2, (hs[-1], fwd[-1]), f"$h = 0.1$: error {fwd[-1]:.2f}", offset=(-8, 6), ha="right", fontsize=10)
    ax2.annotate(f"$h = 0.1$:\nerror {cen[-1]:.2f}", (hs[-1], cen[-1]), xytext=(2e-3, 1e-6), ha="left", va="top",
                 fontsize=10, arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8, shrinkB=6))
    i = int(np.argmin(cen))
    ax2.plot(hs[i], cen[i], "o", ms=8, mfc="white", mec="C0", mew=1.8, zorder=5)
    (mh, eh), (me, ee) = (f"{hs[i]:.1e}".split("e"), f"{cen[i]:.0e}".split("e"))   # 6.5e-06 and 8e-11
    ax2.annotate(f"best central difference (marked):\n$h \\approx {mh} \\times 10^{{{int(eh)}}}$, error "
                 f"$\\approx {me} \\times 10^{{{int(ee)}}}$", (hs[i], cen[i]), xytext=(1e-7, 3e-2), color="C0",
                 fontsize=9.5, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color="C0", lw=0.8, alpha=0.6, relpos=(0.74, 0), shrinkB=5))
    ax2.set_xlim(1e-14, 1); ax2.set_ylim(1e-12, 10)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    ax2.set_title("(b) Error of $f'(2)$ for $f = x^3$ versus the step $h$")
    ax2.set_xlabel("step size $h$"); ax2.set_ylabel("absolute error")
    fig.tight_layout()
    return fig


def _node(ax, xy, text, color, w=1.25, h=0.62, fontsize=9.5):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04",
                                fc="white", ec=color, lw=1.5, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=4)


def _edge(ax, p0, p1, color, ls="-", lw=1.6):
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=12))


def fig_chain_rule_graph():
    """Section 2: the one-neuron network as a graph (values forward, slopes backward); sigma' is at most 1/4."""
    x, w, b, y = 2.0, 0.5, 0.0, 1.0
    z = w * x + b                                                # 1
    p = sigmoid(z)                                               # 0.7311
    L = (p - y) ** 2                                             # 0.0723
    dL_dp = round(2 * (p - y), 4)                                # 2(p - y) = -0.53788, shown as -0.5379
    dp_dz = round(p * (1 - p), 4)                                # 0.1966
    dz_dw = x                                                    # 2
    dL_dw = dL_dp * dp_dz * dz_dw                                # -0.2115
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.9, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})  # 9.9: the tight box stays under 10 in

    # (a) the computational graph
    hw = 0.625                                                   # node half-width (boxes are 1.25 x 0.62)
    for xy, txt, col in [((0, 1.6), f"$x = {x:g}$", GREY), ((0, 0.8), f"$w = {w:g}$", GREY), ((0, 0), f"$b = {b:g}$", GREY),
                         ((2.2, 0.8), f"$z = wx + b$\n$= {z:g}$", "C0"), ((4.4, 0.8), f"$p = \\sigma(z)$\n$= {p:.4f}$", "C0"),
                         ((6.6, 0.8), f"$L = (p - y)^2$\n$= {L:.4f}$", "C0"), ((4.4, 0), f"$y = {y:g}$", GREY)]:
        _node(ax1, xy, txt, col)
    for p0, p1 in [((hw, 1.6), (2.2 - hw, 0.8 + 0.26)), ((hw, 0.9), (2.2 - hw, 0.9)), ((hw, 0.0), (2.2 - hw, 0.8 - 0.28)),
                   ((2.2 + hw, 0.9), (4.4 - hw, 0.9)), ((4.4 + hw, 0.9), (6.6 - hw, 0.9)), ((4.4 + hw, 0.0), (6.6 - hw, 0.8 - 0.28))]:
        _edge(ax1, p0, p1, "C2")                                 # forward pass
    for p0, p1 in [((6.6 - hw, 0.66), (4.4 + hw, 0.66)), ((4.4 - hw, 0.66), (2.2 + hw, 0.66)), ((2.2 - hw, 0.66), (hw, 0.66))]:
        _edge(ax1, p0, p1, "C3", ls="--")                        # backward pass
    ax1.text(3.3, 1.55, "forward pass: compute and store each value", color="C2", ha="center", va="center", fontsize=10)
    for xc, txt in [(1.1, f"$dz/dw = x = {dz_dw:g}$"),
                    (3.3, f"$dp/dz = p(1 - p)$\n$= {dp_dz:.4f}$\n(at the stored $z = 1$, not at $x$)"),
                    (5.5, f"$dL/dp = 2(p - y)$\n$= {dL_dp:.4f}$")]:
        ax1.annotate(txt, xy=(xc, 0.62), xytext=(xc, -0.5), color="C3", ha="center", va="top", fontsize=9.5,
                     arrowprops=dict(arrowstyle="-", color="C3", lw=0.6, alpha=0.5, shrinkA=2, shrinkB=2))  # faint leader up to its arrow
    ax1.text(3.3, -1.4, "backward pass: the local slopes multiply\n"
             f"$dL/dw = ({dL_dp:.4f}) \\times ({dp_dz:.4f}) \\times {dz_dw:g} = {dL_dw:.4f}$",
             color="C3", ha="center", va="center", fontsize=10)
    ax1.set_xlim(-0.75, 7.35); ax1.set_ylim(-1.8, 1.95)         # no equal aspect: the title then sits on the drawing, level with (b)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(False)
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.set_title("(a) One neuron as a graph: values forward, local slopes backward")
    ax1.set_xlabel("forward pass runs left to right; backward pass runs right to left")

    # (b) sigma and its derivative: the factor p(1 - p) never exceeds 1/4 and collapses in the tails
    zs = np.linspace(-6, 6, 601)
    ax2.plot(zs, sigmoid(zs), color="C0", lw=2, label=r"$\sigma(z)$")
    ax2.plot(zs, sigmoid(zs) * (1 - sigmoid(zs)), color="C3", lw=2, label=r"$\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$")
    # sigma'(5) ends just inside the right frame and sits low over the flat tail, so it stays clear of sigma'(2) above it
    for zk, off, ha in [(0, (6, 4), "left"), (1, (6, 4), "left"), (2, (6, 9), "left"), (5, (17, 12), "right")]:
        s = sigmoid(zk) * (1 - sigmoid(zk))
        ax2.plot(zk, s, "o", color="C3", ms=6, zorder=4)
        val = f"{s:.4f}".rstrip("0")                                 # 0.25, 0.1966, 0.105, 0.0066 as in the guide
        label(ax2, (zk, s), f"$\\sigma'({zk}) = {val}$", color="C3", offset=off, ha=ha, fontsize=10)
    ax2.set_xlim(-6, 6); ax2.set_ylim(-0.02, 1.05)              # below 0, so the sigma'(5) dot is not clipped
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title(r"(b) The factor $\sigma'(z)$ never exceeds $1/4$ and saturates")
    ax2.set_xlabel("$z$"); ax2.set_ylabel("value")
    fig.tight_layout()
    return fig


def fig_gradient_and_level_sets():
    """Section 3: level sets of f = x^2 + 3xy, the gradient (8, 3) at (1, 2), and the slope in every direction."""
    f = lambda x, y: x ** 2 + 3 * x * y                          # noqa: E731
    x0 = np.array([1., 2.])
    g = np.array([2 * x0[0] + 3 * x0[1], 3 * x0[0]])             # (8, 3)
    gn = np.linalg.norm(g)                                       # sqrt(73) = 8.544
    u_ne = np.array([1., 1.]) / np.sqrt(2)
    u_lvl = np.array([-3., 8.]) / np.sqrt(73)                    # tangent to the level set: g . u_lvl = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))

    # (a) contour map with three directions at (1, 2)
    xlim, ylim = (-0.5, 3.7), (0, 4)
    # labels in a row along y = 1, clear of the arrows; f = 7 is left unlabelled (the point label names it)
    X, Y = contours(ax1, f, xlim, ylim, [1, 4, 7, 11, 16, 22, 29], label_levels=[1, 4, 11, 16, 22, 29],
                    manual=[(0.30, 1.0), (1.0, 1.0), (2.14, 1.0), (2.77, 1.0), (3.42, 1.0), (3.16, 2.0)])
    ax1.contour(X, Y, f(X, Y), levels=[f(*x0)], colors="C0", linewidths=2)
    ax1.plot(*x0, "ko", ms=6, zorder=5)
    boxed(ax1, x0, f"$(1, 2)$: $f = {f(*x0):g}$", offset=(-8, -12), ha="right")
    Lr = 0.12 * gn                                               # one common arrow length: only direction differs
    for u, col, txt, off, ha, va in [(g / gn, "C3", f"$\\nabla f = (8, 3)$:\nslope {g @ (g / gn):.2f}", (4, -14), "left", "bottom"),
                                     (u_ne, "C1", f"north-east $(1, 1)/\\sqrt{{2}}$:\nslope {g @ u_ne:.2f}", (4, 4), "left", "bottom"),
                                     (u_lvl, "C2", f"along the\ncontour:\nslope {abs(g @ u_lvl):.0f}", (-8, 0), "right", "center")]:
        arrow(ax1, Lr * u, start=x0, color=col)
        boxed(ax1, x0 + Lr * u, txt, color=col, offset=off, ha=ha, va=va)
    right_angle(ax1, x0, g, u_lvl, size=0.15)
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim); ax1.set_aspect("equal")
    ax1.set_title("(a) Level sets of $f = x^2 + 3xy$ and three directions at $(1, 2)$")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$")

    # (b) the directional derivative as a function of the direction angle
    phi = np.linspace(0, 360, 721)
    Du = g[0] * np.cos(np.radians(phi)) + g[1] * np.sin(np.radians(phi))
    ax2.plot(phi, Du, color="C0", lw=2, label=r"$D_{\mathbf{u}} f = \nabla f^\top \mathbf{u} = 8\cos\varphi + 3\sin\varphi$")
    for v in (gn, -gn):
        ax2.axhline(v, color=GREY, ls=":", lw=1)
    ax2.axhline(0, color=GREY, lw=0.8)
    phi_g = np.degrees(np.arctan2(g[1], g[0]))                   # 20.56 degrees
    for ph, col, txt, off, ha in [(phi_g, "C3", f"${phi_g:.1f}^\\circ$ (along $\\nabla f$):\n$\\|\\nabla f\\| = {gn:.2f}$", (-6, 22), "left"),
                                  (45, "C1", f"$45^\\circ$ (north-east): {g @ u_ne:.2f}", (6, 11), "left"),  # above the dotted 8.54
                                  (phi_g + 90, "C2", f"${phi_g + 90:.1f}^\\circ$ (along the contour): 0", (6, 4), "left"),
                                  (phi_g + 180, "C3", f"${phi_g + 180:.1f}^\\circ$ (along $-\\nabla f$): ${-gn:.2f}$", (-8, -18), "right")]:
        val = g[0] * np.cos(np.radians(ph)) + g[1] * np.sin(np.radians(ph))
        ax2.plot(ph, val, "o", color=col, ms=6, zorder=4)
        boxed(ax2, (ph, val), txt, color=col, offset=off, ha=ha, va="bottom")
    ax2.set_xticks([0, 90, 180, 270, 360]); ax2.set_xlim(0, 360); ax2.set_ylim(-10.5, 13.5)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("(b) Slope in every direction: a cosine of the angle to the gradient")
    ax2.set_xlabel(r"direction angle $\varphi$ (degrees from the $x$-axis)"); ax2.set_ylabel("directional derivative")
    fig.tight_layout()
    return fig


def fig_jacobian_linearisation():
    """Section 4: the columns of the polar Jacobian at (2, pi/6), and the linearisation error for delta = (0.1, 0.05)."""
    P = lambda r, th: np.array([r * np.cos(th), r * np.sin(th)])  # noqa: E731
    r0, th0 = 2.0, np.pi / 6
    P0 = P(r0, th0)                                              # (1.732, 1.000)
    J = np.array([[np.cos(th0), -r0 * np.sin(th0)], [np.sin(th0), r0 * np.cos(th0)]])
    c1, c2 = J[:, 0], J[:, 1]                                    # (0.866, 0.5) and (-1, 1.732)
    delta = np.array([0.1, 0.05])
    lin = J @ delta                                              # (0.0366, 0.1366)
    exact = P(r0 + delta[0], th0 + delta[1]) - P0                # (0.0319, 0.1396)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))

    # (a) polar grid, the two columns of J at P0, and the parallelogram they span
    th = np.linspace(0, np.pi / 2, 100)
    for r in [1, 1.5, 2, 2.5]:
        ax1.plot(r * np.cos(th), r * np.sin(th), color=GREY, lw=0.6)
    for deg in range(0, 91, 15):
        a = np.radians(deg)
        ax1.plot([0, 2.6 * np.cos(a)], [0, 2.6 * np.sin(a)], color=GREY, lw=0.6)
    label(ax1, (2.03, 0.26), "$r = 2$", color=GREY, offset=(0, 0), ha="left", va="center", fontsize=9)  # between the arcs, below the 15-degree ray
    label(ax1, (1.0 * np.cos(th0), 1.0 * np.sin(th0)), r"$\theta = 30^\circ$", color=GREY, offset=(4, -10), fontsize=9)
    ax1.add_patch(Polygon([P0, P0 + c1, P0 + c1 + c2, P0 + c2], closed=True, fc="C0", alpha=0.08, ec="none"))
    ax1.text(1.9, 2.15, f"area $= \\det J$\n$= r = {np.linalg.det(J):g}$", ha="center", va="center", fontsize=10,
             bbox=WHITE_BOX, zorder=6)
    arrow(ax1, c1, start=P0, color="C0")
    label(ax1, P0 + c1, f"$\\partial\\mathbf{{P}}/\\partial r$\n$= ({c1[0]:.3f}, {c1[1]:.1f})$\nlength {np.linalg.norm(c1):g}",
          color="C0", offset=(6, -6), va="center", fontsize=10)
    arrow(ax1, c2, start=P0, color="C1")
    boxed(ax1, P0 + 0.3 * c2, f"$\\partial\\mathbf{{P}}/\\partial\\theta = ({c2[0]:.0f}, {c2[1]:.3f})$\nlength $r = {np.linalg.norm(c2):g}$",
          color="C1", offset=(-10, 0), ha="right", va="center", fontsize=10)
    right_angle(ax1, P0, c1, c2, size=0.12)
    ax1.plot(*P0, "ko", ms=6, zorder=5)
    boxed(ax1, P0, f"$\\mathbf{{P}}(2, \\pi/6) = ({P0[0]:.3f}, {P0[1]:g})$", offset=(-6, -14), ha="right", va="bottom", fontsize=10)
    ax1.set_xlim(-0.2, 3.7); ax1.set_ylim(-0.2, 3.4); ax1.set_aspect("equal")
    ax1.set_title(r"(a) Columns of $J$: how $(x, y)$ moves per unit $r$ and per radian $\theta$")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$")

    # (b) zoom: exact image of the small (r, theta) rectangle versus its linear image P0 + J delta
    rs, ths = np.linspace(r0, r0 + delta[0], 30), np.linspace(th0, th0 + delta[1], 30)
    img = np.concatenate([P(rs, th0).T, P(r0 + delta[0], ths).T, P(rs[::-1], th0 + delta[1]).T, P(r0, ths[::-1]).T])
    ax2.plot(img[:, 0], img[:, 1], color="C0", lw=2, label="exact image of the $\\delta$-rectangle")
    corners = np.array([[0, 0], [delta[0], 0], delta, [0, delta[1]], [0, 0]])
    par = P0 + corners @ J.T
    ax2.plot(par[:, 0], par[:, 1], color="C3", ls="--", lw=1.6, label="linear image $\\mathbf{P} + J\\boldsymbol{\\delta}$ (parallelogram)")
    ax2.plot(*P0, "ko", ms=6, zorder=5)
    label(ax2, P0, r"$\mathbf{P}(2, \pi/6)$", offset=(-8, -4), ha="right", fontsize=10)
    ax2.plot(*(P0 + exact), "o", color="C0", ms=7, zorder=5)
    label(ax2, P0 + exact, f"exact $({(P0 + exact)[0]:.4f}, {(P0 + exact)[1]:.4f})$", color="C0", offset=(-2, 8), ha="left", fontsize=10)
    ax2.plot(*(P0 + lin), "s", color="C3", ms=7, zorder=5)
    label(ax2, P0 + lin, f"predicted $({(P0 + lin)[0]:.4f}, {(P0 + lin)[1]:.4f})$", color="C3", offset=(8, -4), fontsize=10)
    label(ax2, P0 + J @ [delta[0], 0], "$\\delta = (0.1, 0)$: exact and\nlinear images coincide", color=GREY, offset=(6, -4), va="top", fontsize=8.5)
    ax2.set_xlim(1.66, 1.95); ax2.set_ylim(0.955, 1.195); ax2.set_aspect("equal")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_title(f"(b) Zoom: $\\delta = (0.1, 0.05)$, error $({exact[0] - lin[0]:.4f}, {exact[1] - lin[1]:.4f})$")
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$")
    fig.tight_layout()
    return fig


def fig_hessian_bowl_and_saddle():
    """Section 5: contours of the bowl x^2 + 3xy + 5y^2 (eigenvalues 1, 11) and the saddle x^2 + 3xy (4.16, -2.16)."""
    f = lambda x, y: x ** 2 + 3 * x * y + 5 * y ** 2             # noqa: E731
    g = lambda x, y: x ** 2 + 3 * x * y                          # noqa: E731
    Hf, Hg = np.array([[2., 3.], [3., 10.]]), np.array([[2., 3.], [3., 0.]])
    xlim, ylim = (-5.2, 5.2), (-3.4, 3.8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))

    # (a) the bowl: eigenvectors of H, scaled to the semi-axes of the ellipse f = 11
    lam, Q = np.linalg.eigh(Hf)                                  # 1, 11
    Q = Q * np.sign(Q[1])                                        # orient as in the guide: (-0.95, 0.32), (0.32, 0.95)
    c = 11.                                                      # f = (1/2) x^T H x, so f = c has semi-axes sqrt(2c / lambda)
    semi = np.sqrt(2 * c / lam)                                  # 4.69, 1.41
    # labels along the long axis on the right (the blue arrow runs left) and in a column at x = -3
    X, Y = contours(ax1, f, xlim, ylim, [1, 4.5, 22, 40],
                    manual=[(1.34, -0.45), (2.85, -0.95), (-3.0, -0.95), (-3.0, -1.75)])
    ax1.contour(X, Y, f(X, Y), levels=[c], colors="C0", linewidths=1.8)
    boxed(ax1, (3.96, -0.5), f"$f = {c:g}$", color="C0", offset=(7, 2), ha="left", fontsize=9)  # (3.96, -0.5) is on the ellipse
    ax1.plot(0, 0, "ko", ms=6, zorder=5)
    boxed(ax1, (0, 0), "minimum", offset=(8, 4), ha="left", va="bottom")
    arrow(ax1, semi[1] * Q[:, 1], color="C3")
    boxed(ax1, semi[1] * Q[:, 1], f"$\\lambda = {lam[1]:.0f}$: semi-axis {semi[1]:.2f}\nalong $({Q[0, 1]:.2f}, {Q[1, 1]:.2f})$",
          color="C3", offset=(6, 8), ha="left", va="bottom")
    arrow(ax1, semi[0] * Q[:, 0], color="C0")
    boxed(ax1, (-5.0, 3.65), f"$\\lambda = {lam[0]:.0f}$: semi-axis {semi[0]:.2f}\nalong $({Q[0, 0]:.2f}, {Q[1, 0]:.2f})$",
          color="C0", ha="left", va="top")                       # empty top-left corner: the ellipse peaks at y = 2
    boxed(ax1, (0.0, -3.05), f"axis ratio ${semi[0]:.2f} / {semi[1]:.2f} = \\sqrt{{11}} = \\sqrt{{\\lambda_{{\\max}} / \\lambda_{{\\min}}}}$")
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim); ax1.set_aspect("equal")
    ax1.set_title("(a) Bowl $f = x^2 + 3xy + 5y^2$: eigenvalues 1 and 11")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$")

    # (b) the saddle: the zero set is two lines; g rises along the x-axis and falls along (3, -2)
    mu, R = np.linalg.eigh(Hg)                                   # -2.16, 4.16
    R = R * np.sign(R[0])                                        # (0.585, -0.811), (0.811, 0.585)
    # every level meets the zero line x = -3y at the left edge: place the labels by hand, away from it
    X, Y = contours(ax2, g, xlim, ylim, [-16, -9, -4, -1, 1, 4, 9, 16], label_levels=[-16, -9, -4, 4, 9],
                    manual=[(4.0, -2.67), (-1.0, 3.33), (-1.0, 1.67), (-1.0, -1.0), (-1.85, -1.0)], upright=True)
    xs = np.array(xlim)
    ax2.plot([0, 0], ylim, color="k", lw=1.2)
    ax2.plot(xs, -xs / 3, color="k", lw=1.2)
    boxed(ax2, (-3.0, 1.0), "$g = 0$ on $x = 0$\nand on $x = -3y$", offset=(0, 14), va="bottom", fontsize=9.5)  # just above the line y = -x/3
    ax2.plot(0, 0, "ko", ms=6, zorder=5)
    boxed(ax2, (0, 0), "saddle", offset=(-8, -10), ha="right", va="top")
    arrow(ax2, (2.5, 0), color="C3")
    boxed(ax2, (3.3, 0.15), "$x$-axis: $g(t, 0) = t^2$\nrises", color="C3", va="bottom")  # below the faint eigenvector
    d = 2.5 * np.array([3., -2.]) / np.sqrt(13)
    arrow(ax2, d, color="C0")
    boxed(ax2, d, "$(3, -2)$:\n$g = -9t^2$ falls", color="C0", offset=(6, -6), ha="left", va="top")  # below the line y = -x/3
    for j, col, at, off, ha in [(1, "C3", 2.9 * R[:, 1], (6, 0), "left"), (0, "C0", (0.2, -2.95), (0, 0), "left")]:
        arrow(ax2, 2.9 * R[:, j], color=col, lw=1.2, alpha=0.6)   # fainter and longer than the two sample directions
        boxed(ax2, at, f"eigenvector\n$\\lambda = {mu[j]:.2f}$", color=col, offset=off, ha=ha, fontsize=9.5)
    ax2.set_xlim(*xlim); ax2.set_ylim(*ylim); ax2.set_aspect("equal")
    ax2.set_title(f"(b) Saddle $g = x^2 + 3xy$: eigenvalues {mu[1]:.2f} and ${mu[0]:.2f}$")
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$")
    fig.tight_layout()
    return fig


def fig_least_squares_gradient():
    """Section 7: the three-point line fit in data space and the loss ||Bx - b||^2 over (intercept, slope)."""
    t, yd = np.array([0., 1., 2.]), np.array([1., 2., 2.])
    B, b = np.column_stack([np.ones(3), t]), yd
    x0 = np.array([0., 1.])
    r0 = B @ x0 - b                                              # (-1, -1, 0), loss 2
    grad0 = 2 * B.T @ r0                                         # (-4, -2)
    xs = np.linalg.solve(B.T @ B, B.T @ b)                       # (7/6, 1/2)
    rs = B @ xs - b                                              # (1/6, -1/3, 1/6), loss 1/6
    loss = lambda x1, x2: (x1 - 1) ** 2 + (x1 + x2 - 2) ** 2 + (x1 + 2 * x2 - 2) ** 2  # noqa: E731
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))

    # (a) data space: the trial line, the least-squares line and both sets of residuals
    tt = np.array([-0.3, 2.3])
    ax1.plot(tt, x0[0] + x0[1] * tt, color="C1", ls="--", lw=1.6, label=f"trial $\\mathbf{{x}}_0 = (0, 1)$: $y = t$, loss {r0 @ r0:g}")
    ax1.plot(tt, xs[0] + xs[1] * tt, color="C0", lw=2, label="optimum $\\mathbf{x}^* = (7/6, 1/2)$: $y = 7/6 + t/2$, loss $1/6$")
    for k in range(3):
        if r0[k] != 0:
            ax1.plot([t[k], t[k]], [yd[k], yd[k] + r0[k]], color="C1", lw=3, alpha=0.6, solid_capstyle="butt",
                     label="residuals $B\\mathbf{x}_0 - \\mathbf{b} = (-1, -1, 0)$" if k == 0 else None)
        ax1.plot([t[k] + 0.04, t[k] + 0.04], [yd[k], yd[k] + rs[k]], color="C0", lw=3, alpha=0.6, solid_capstyle="butt",
                 label="residuals $B\\mathbf{x}^* - \\mathbf{b} = (1/6, -1/3, 1/6)$" if k == 0 else None)
    ax1.plot(t, yd, "ko", ms=7, zorder=5, label="data $(t, y)$")
    ax1.set_xlim(-0.4, 2.6); ax1.set_ylim(-0.3, 3.4)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.set_title("(a) Fitting $y \\approx x_1 + x_2 t$ to three points")
    ax1.set_xlabel("$t$"); ax1.set_ylabel("$y$")

    # (b) parameter space: the loss surface, the trial point with -grad, and the optimum where grad = 0
    xlim, ylim = (-1, 3), (-1, 2.5)
    contours(ax2, loss, xlim, ylim, [0.25, 0.5, 1, 2, 4, 8, 16], label_levels=[0.5, 1, 2, 4, 8, 16])  # 0.25 ring too small
    ax2.plot(*x0, "o", color="C1", ms=7, zorder=5)
    boxed(ax2, x0, f"$\\mathbf{{x}}_0 = (0, 1)$\nloss {r0 @ r0:g}", color="C1", offset=(-8, -6), ha="right", va="top")
    arrow(ax2, -0.2 * grad0, start=x0, color="C1")
    boxed(ax2, x0 - 0.2 * grad0, f"$-\\nabla L = ({-grad0[0]:g}, {-grad0[1]:g})$ (scaled)", color="C1", offset=(6, 0), ha="left")
    ax2.plot(*xs, "*", color="C0", ms=14, zorder=5)
    boxed(ax2, xs, f"$\\mathbf{{x}}^* = ({xs[0]:.3f}, {xs[1]:g})$: loss $1/6$\n$\\nabla L = 2B^\\top(B\\mathbf{{x}}^* - \\mathbf{{b}}) = \\mathbf{{0}}$",
          color="C0", offset=(6, -8), ha="left", va="top")
    ax2.set_xlim(*xlim); ax2.set_ylim(*ylim); ax2.set_aspect("equal")
    ax2.set_title("(b) Loss $\\|B\\mathbf{x} - \\mathbf{b}\\|^2$: $-\\nabla L$ does not point at $\\mathbf{x}^*$")
    ax2.set_xlabel("intercept $x_1$"); ax2.set_ylabel("slope $x_2$")
    fig.tight_layout()
    return fig


def fig_loss_gradients():
    """Section 8: logistic, squared-error-on-p and hinge losses for y = 1 against the logit, and their derivatives."""
    zs = np.linspace(-4, 4, 801)
    s = sigmoid(zs)
    y = 1.0
    x, w = np.array([1., 2.]), np.array([0.5, -0.25])
    z0 = w @ x                                                   # 0
    L0 = -np.log(sigmoid(z0))                                    # 0.6931
    w1 = w - 0.1 * (sigmoid(z0) - y) * x                         # (0.55, -0.15)
    z1 = w1 @ x                                                  # 0.25
    L1 = -np.log(sigmoid(z1))                                    # 0.5759
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) the three losses
    ax1.plot(zs, -np.log(s), color="C0", lw=2, label=r"logistic $-\ln\sigma(z)$")
    ax1.plot(zs, (s - y) ** 2, color="C1", lw=2, label=r"squared error on $p$ (Section 2): $(\sigma(z) - 1)^2$")
    ax1.plot(zs, np.maximum(0, 1 - zs), color="C2", lw=2, label=r"hinge $\max(0,\, 1 - z)$")
    for zk, Lk, txt, xy in [(z0, L0, f"$z = 0$: $L = {L0:.3f}$", (1.0, 2.4)),
                            (z1, L1, f"after one step:\n$z = {z1:g}$, $L = {L1:.3f}$", (1.0, 1.55))]:
        ax1.plot(zk, Lk, "o", color="C0", ms=6, zorder=5)
        ax1.annotate(txt, (zk, Lk), xytext=xy, color="C0", fontsize=10, va="center",   # leaders leave the left end
                     arrowprops=dict(arrowstyle="-", color="C0", lw=0.8, relpos=(0, 0.5), shrinkB=4))
    ax1.set_xlim(-4, 4); ax1.set_ylim(-0.12, 4.5)               # below 0: the hinge's flat zero stays clear of the spine
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("(a) Three losses for a positive example ($y = 1$) versus the logit $z$")
    ax1.set_xlabel("logit $z$"); ax1.set_ylabel("loss")

    # (b) their derivatives with respect to the logit
    ax2.plot(zs, s - y, color="C0", lw=2, label=r"logistic: $\sigma(z) - y$")
    ax2.plot(zs, 2 * (s - y) * s * (1 - s), color="C1", lw=2, label=r"squared error: $2(\sigma - y)\,\sigma(1 - \sigma)$")
    ax2.plot([-4, 1], [-1, -1], color="C2", lw=2, label="hinge: $-y$ for $z < 1$, else $0$")
    ax2.plot([1, 4], [0, 0], color="C2", lw=2)
    ax2.plot([1, 1], [-1, 0], color="C2", lw=1, ls=":")
    zc = -4.0
    sq_c, lg_c = 2 * (sigmoid(zc) - y) * sigmoid(zc) * (1 - sigmoid(zc)), sigmoid(zc) - y  # -0.0347, -0.982
    ax2.plot(z0, sigmoid(z0) - y, "o", color="C0", ms=6, zorder=5)
    ax2.annotate(f"$z = 0$: $p - y = {sigmoid(z0) - y:g}$", (z0, sigmoid(z0) - y), xytext=(1.2, -0.62), color="C0",
                 fontsize=10, va="center", arrowprops=dict(arrowstyle="-", color="C0", lw=0.8, shrinkB=4))
    ax2.plot(zc, sq_c, "o", color="C1", ms=6, zorder=5, clip_on=False)   # z = -4 is the left edge: keep the dots whole
    ax2.annotate(f"$z = -4$, confidently wrong:\nsquared-error slope\nonly ${sq_c:.3f}$", (zc, sq_c), xytext=(-3.9, -0.47),
                 color="C1", fontsize=9.5, va="center", arrowprops=dict(arrowstyle="-", color="C1", lw=0.8, shrinkB=4))
    ax2.plot(zc, lg_c, "o", color="C0", ms=6, zorder=5, clip_on=False)
    ax2.annotate(f"logistic slope ${lg_c:.2f}$", (zc, lg_c), xytext=(-3.9, -0.7),
                 color="C0", fontsize=9.5, va="center", arrowprops=dict(arrowstyle="-", color="C0", lw=0.8, shrinkB=4))
    ax2.set_xlim(-4, 4); ax2.set_ylim(-1.38, 0.15)
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.set_title("(b) Their derivatives $\\partial L / \\partial z$")
    ax2.set_xlabel("logit $z$"); ax2.set_ylabel(r"$\partial L / \partial z$")
    fig.tight_layout()
    return fig


def fig_gd_learning_rate():
    """Case study 2: gradient descent on the quadratic with H = [[2, 3], [3, 10]] at eta = 1/6 and eta = 0.19."""
    H = np.array([[2., 3.], [3., 10.]])
    f = lambda x, y: 0.5 * (H[0, 0] * x ** 2 + 2 * H[0, 1] * x * y + H[1, 1] * y ** 2)  # noqa: E731
    lam = np.linalg.eigvalsh(H)                                  # 1, 11
    start = np.array([3., -2.])                                  # f = 11, grad = H start = (0, -11)
    grad0 = H @ start
    newton = -np.linalg.solve(H, grad0)                          # (-3, 2)

    def path(eta, steps):
        pts = [start]
        for _ in range(steps):
            pts.append(pts[-1] - eta * H @ pts[-1])
        return np.array(pts)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1, 1]})
    xlim, ylim = (-1, 4), (-3, 2)
    for ax in (ax1, ax2):
        # labels on the empty left side: the paths stay at x > 0.3 and the Newton box is at y < -1.3
        contours(ax, f, xlim, ylim, [0.5, 2, 4.5, 11, 22], label_levels=[0.5, 2, 4.5, 11, 22],
                 manual=[(-0.7, 0.0), (-0.8, 0.81), (-0.8, 1.15), (-0.8, 1.70), (2.0, 1.39)])
        ax.plot(*start, "ko", ms=6, zorder=5)
        boxed(ax, start, f"start $(3, -2)$: $f = {f(*start):g}$\n$\\nabla f = ({grad0[0]:g}, {grad0[1]:g})$",
              offset=(-6, -10), ha="right", va="top", fontsize=9.5)
        ax.plot(0, 0, "ko", ms=6, zorder=5)
        boxed(ax, (0, 0), "minimum", offset=(0, 7), ha="center", va="bottom", fontsize=9.5)  # above: inside the frame, clear of both paths
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")

    # (a) eta = 1/6: both eigen-components shrink by 5/6 per step; Newton goes straight to the minimum
    eta_a = 1 / 6
    p = path(eta_a, 12)
    ax1.plot(p[:, 0], p[:, 1], "o-", color="C0", ms=3.5, lw=1.4, zorder=4)
    ax1.annotate("", xy=(0, 0), xytext=start, zorder=3,
                 arrowprops=dict(arrowstyle="-|>", color="C2", lw=1.6, ls="--", shrinkA=0, shrinkB=0, mutation_scale=14))
    boxed(ax1, (0.4, -1.55), f"Newton step\n$-H^{{-1}}\\nabla f = ({newton[0]:g}, {newton[1]:g})$", color="C2", fontsize=9.5)
    fac_a = np.abs(1 - eta_a * lam)                              # 5/6 and 5/6
    ax1.set_title(f"(a) $\\eta = 1/6$: both eigen-components\nshrink by $|1 - \\eta\\lambda| = {fac_a[0]:.2f}$ per step")

    # (b) eta = 0.19 > 2/11: the sharp direction flips sign and grows 1.09x per step
    eta_b = 0.19
    p = path(eta_b, 7)
    ax2.plot(p[:, 0], p[:, 1], "o-", color="C3", ms=3.5, lw=1.4, zorder=4)
    boxed(ax2, p[-1], "$k = 7$", color="C3", offset=(0, 5), va="bottom", fontsize=9)  # marks the end: (1.30, 1.39)
    fac_b = np.abs(1 - eta_b * lam)                              # 0.81 and 1.09
    ax2.set_title(f"(b) $\\eta = 0.19 > 2/11$: the sharp direction\ngrows by $|1 - \\eta\\lambda| = {fac_b[1]:.2f}$ per step")

    # (c) remaining error along each eigendirection after k steps
    k = np.arange(0, 51)
    for eta, name, col in [(eta_a, "1/6", "C0"), (eta_b, "0.19", "C3")]:
        for j, ls, which in [(0, "-", "flat"), (1, "--", "sharp")]:
            fac = abs(1 - eta * lam[j])
            solid = ls == "-"                                    # wide pale band under full-colour dashes: both stay visible
            ax3.semilogy(k, fac ** k, color=col, ls=ls, lw=3.2 if solid else 1.6, alpha=0.4 if solid else 1, zorder=2 if solid else 3,
                         label=f"$\\eta = {name}$, {which} ($\\lambda = {lam[j]:.0f}$): {fac:.2f}")
    ax3.axhline(fac_a[0] ** 50, color=GREY, ls=":", lw=1)
    m, e = f"{fac_a[0] ** 50:.1e}".split("e")                   # 1.1e-04
    label(ax3, (0, fac_a[0] ** 50), f"$(5/6)^{{50}} \\approx {m} \\times 10^{{{int(e)}}}$", color=GREY, offset=(4, 4), fontsize=9.5)
    ax3.set_xlim(0, 50); ax3.set_ylim(1e-5, 1e6)                # room between the legend and the sharp line, which ends at 74
    label(ax3, (50, fac_b[1] ** 50), f"$\\approx {fac_b[1] ** 50:.0f}$", color="C3", offset=(-2, 5), ha="right", va="bottom", fontsize=9.5)
    ax3.set_box_aspect(1)                                        # same box as (a) and (b), so the three tops line up
    ax3.legend(loc="upper left", fontsize=8.5, handlelength=1.6)
    ax3.set_title("(c) Error along each eigendirection:\nfactor $|1 - \\eta\\lambda|$ per step")
    ax3.set_xlabel("gradient step $k$"); ax3.set_ylabel("remaining error (fraction of start)")
    fig.tight_layout()
    return fig


FIGURES = {
    "tangent-and-step-size": fig_tangent_and_step_size,
    "chain-rule-graph": fig_chain_rule_graph,
    "gradient-and-level-sets": fig_gradient_and_level_sets,
    "jacobian-linearisation": fig_jacobian_linearisation,
    "hessian-bowl-and-saddle": fig_hessian_bowl_and_saddle,
    "least-squares-gradient": fig_least_squares_gradient,
    "loss-gradients": fig_loss_gradients,
    "gd-learning-rate": fig_gd_learning_rate,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 04 concept-guide figures.")
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
