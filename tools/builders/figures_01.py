"""Figures for the Module 01 concept guide — Vectors and Geometry.

Writes SVG files into part-1-linear-algebra/01-vectors-and-geometry/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, so picture and text agree exactly.

    python tools/builders/figures_01.py                  # (re)write every figure
    python tools/builders/figures_01.py projection       # only the named figure(s)
    python tools/builders/figures_01.py --preview DIR    # also write PNG previews to DIR
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402

import numpy as np  # noqa: E402

FIG_DIR = ROOT / "part-1-linear-algebra/01-vectors-and-geometry/figures"


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_linear_combinations():
    """Section 1: combinations of u, v on their skewed grid; a dependent pair spans only a line."""
    u, v, w = np.array([1., 2.]), np.array([3., -1.]), np.array([5., 0.])
    a, b = 5 / 7, 10 / 7
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.9))

    # (a) the grid of integer combinations, and w reached by a*u then b*v
    ts = np.array([-20., 20.])
    for k in range(-6, 7):
        ax1.plot(k * u[0] + ts * v[0], k * u[1] + ts * v[1], color=GREY, lw=0.6, alpha=0.5, zorder=0)
        ax1.plot(k * v[0] + ts * u[0], k * v[1] + ts * u[1], color=GREY, lw=0.6, alpha=0.5, zorder=0)
    ax1.fill([0, a * u[0], w[0]], [0, a * u[1], w[1]], color="C3", alpha=0.10, lw=0)
    arrow(ax1, u, color="C0", label=r"$\mathbf{u}$", label_offset=(-14, 2))
    arrow(ax1, v, color="C1", label=r"$\mathbf{v}$", label_offset=(-4, -16))
    arrow(ax1, a * u, color="C0", lw=1.4, alpha=0.6)
    arrow(ax1, b * v, start=a * u, color="C1", lw=1.4, alpha=0.6)
    arrow(ax1, w, color="C3", label=r"$\mathbf{w} = \frac{5}{7}\mathbf{u} + \frac{10}{7}\mathbf{v}$",
          label_offset=(-70, -22))
    label(ax1, a * u / 2, r"$\frac{5}{7}\mathbf{u}$", color="C0", offset=(-8, 0), fontsize=12, ha="right", va="center")
    label(ax1, a * u + b * v / 2, r"$\frac{10}{7}\mathbf{v}$", color="C1", offset=(4, 4), fontsize=12)
    for p, txt, off in [(u + v, r"$\mathbf{u}+\mathbf{v}$", (6, 2)), (2 * u - v, r"$2\mathbf{u}-\mathbf{v}$", (6, 2))]:
        ax1.plot(*p, "ko", ms=4)
        label(ax1, p, txt, offset=off)
    ax1.set_xlim(-2.5, 6.5); ax1.set_ylim(-2.5, 6); ax1.set_aspect("equal"); ax1.grid(False)
    ax1.set_title(r"(a) Independent: combinations of $\mathbf{u}, \mathbf{v}$ fill the plane")
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")

    # (b) u and -2u: every combination stays on one line, so w is out of reach
    xs = np.array([-3., 3.5])
    ax2.plot(xs, 2 * xs, color="C0", lw=6, alpha=0.15, solid_capstyle="butt")
    label(ax2, (6.3, -3.6), "span$\\{\\mathbf{u}, -2\\mathbf{u}\\}$ is\nonly the line $x_2 = 2x_1$",
          color="C0", offset=(0, 0), ha="right", va="center")
    arrow(ax2, -2 * u, color="C2", label=r"$-2\mathbf{u}$", label_offset=(8, -4))
    arrow(ax2, u, color="C0", label=r"$\mathbf{u}$", label_offset=(8, -6))
    ax2.plot(*w, "o", ms=8, mfc="white", mec="C3", mew=2)
    label(ax2, w, "$\\mathbf{w} = (5, 0)$ is not\na combination of them", color="C3", offset=(-4, 10), ha="center")
    ax2.set_xlim(-3.5, 6.5); ax2.set_ylim(-4.8, 6); ax2.set_aspect("equal")
    ax2.set_title(r"(b) Dependent: $2\mathbf{u} + 1\cdot(-2\mathbf{u}) = \mathbf{0}$")
    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    fig.tight_layout()
    return fig


def fig_span_in_3d():
    """Section 1: span{a1, a2} is a plane in R^3; x = a1 + 2 a2 lies in it, a3 sticks out."""
    a1, a2, a3 = np.array([1., 1., 0.]), np.array([1., 0., 1.]), np.array([0., 1., 1.])
    x = a1 + 2 * a2
    n = np.cross(a1, a2)                                  # (1, -1, -1), normal to the plane
    foot = a3 - (a3 @ n) / (n @ n) * n                    # (2/3, 1/3, 1/3), a3 dropped onto the plane
    fig = plt.figure(figsize=(6.4, 5.4))
    ax = fig.add_subplot(projection="3d")
    s, t = np.meshgrid(np.linspace(-0.6, 1.6, 2), np.linspace(-0.6, 2.4, 2))
    P = s[..., None] * a1 + t[..., None] * a2
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color="C0", alpha=0.15, shade=False, lw=0)
    for k in np.linspace(-0.6, 1.6, 12):                  # faint lines on the plane help depth perception
        L = k * a1 + np.outer([-0.6, 2.4], a2)
        ax.plot(L[:, 0], L[:, 1], L[:, 2], color="C0", lw=0.4, alpha=0.35)
    for vec, col, lab, pos in [(a1, "C0", r"$\mathbf{a}_1$", (1.15, 1.0, -0.25)),
                               (a2, "C1", r"$\mathbf{a}_2$", (1.2, -0.25, 0.95)),
                               (x, "C2", r"$\mathbf{x} = \mathbf{a}_1 + 2\mathbf{a}_2$", (2.2, 1.0, 2.35)),
                               (a3, "C3", r"$\mathbf{a}_3$", (-0.05, 1.05, 1.15))]:
        ax.quiver(0, 0, 0, *vec, color=col, lw=2.2, arrow_length_ratio=0.12)
        ax.text(*pos, lab, color=col, fontsize=13)
    path = np.array([np.zeros(3), a1, a1 + a2, x])        # x reached tip-to-tail: a1, then a2 twice
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="C1", ls="--", lw=1.2)
    ax.plot(*np.array([a3, foot]).T, color="C3", ls=":", lw=1.5)
    ax.scatter(*foot, color="C3", s=12)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    ax.set_title(r"span$\{\mathbf{a}_1, \mathbf{a}_2\}$ is a plane: $\mathbf{x}$ lies in it, $\mathbf{a}_3$ does not")
    ax.set_xlim(-1.2, 3.8); ax.set_ylim(-0.6, 1.8); ax.set_zlim(-0.6, 2.4)
    ax.set_box_aspect((5.0, 2.4, 3.0))                    # equal scale on all three axes
    ax.view_init(elev=24, azim=-58)
    return fig


def fig_norms():
    """Section 2: L1, L2 and L-infinity unit balls; crow-flies vs taxi distance to (3, 4)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    th = np.linspace(0, 2 * np.pi, 400)
    ax1.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color="C1", lw=2, label=r"$L_1$: $|x_1| + |x_2| \leq 1$ (diamond)")
    ax1.plot(np.cos(th), np.sin(th), color="C0", lw=2, label=r"$L_2$: $x_1^2 + x_2^2 \leq 1$ (disc)")
    ax1.plot([1, -1, -1, 1, 1], [1, 1, -1, -1, 1], color="C2", lw=2, label=r"$L_\infty$: $\max_i |x_i| \leq 1$ (square)")
    ax1.legend(loc="lower center", fontsize=9, framealpha=0.95)
    ax1.plot([0, 1], [1, 0], "o", color="C1", ms=6)
    label(ax1, (0, 1), "corners on the axes:\ncoordinates exactly 0", color="C1", offset=(8, 10), fontsize=9.5)
    ax1.set_xlim(-1.6, 1.6); ax1.set_ylim(-2.45, 1.6); ax1.set_aspect("equal")
    ax1.set_title("(a) Unit balls $\\{\\mathbf{x}: \\|\\mathbf{x}\\| \\leq 1\\}$ of three norms")
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")

    ax2.set_xticks(range(0, 5)); ax2.set_yticks(range(0, 6))
    stairs = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [3, 2], [3, 3], [3, 4]])
    ax2.plot(stairs[:, 0], stairs[:, 1], color="C1", lw=2.5, label=r"van on the street grid: $\|(3, 4)\|_1 = 7$")
    ax2.plot([0, 3], [0, 4], color="C0", lw=2.5, label=r"as the crow flies: $\|(3, 4)\|_2 = 5$")
    ax2.plot(0, 0, "ks", ms=8); ax2.plot(3, 4, "k*", ms=13)
    label(ax2, (0, 0), "depot", offset=(0, -16), ha="center")
    label(ax2, (3, 4), "customer", offset=(-4, 8), ha="center")
    ax2.set_xlim(-0.6, 4.6); ax2.set_ylim(-0.8, 6.2); ax2.set_aspect("equal")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax2.set_title("(b) Two rulers, two distances to one customer")
    ax2.set_xlabel("blocks east"); ax2.set_ylabel("blocks north")
    fig.tight_layout()
    return fig


def fig_cosine_vs_euclidean():
    """Section 4: d1, d2 = 4 d1 and d3 drawn true to scale in the plane they span."""
    d1, d2, d3 = np.array([2., 1., 0.]), np.array([8., 4., 0.]), np.array([0., 1., 3.])
    e = d1 / np.linalg.norm(d1)                            # first in-plane axis: along d1 (and d2)
    f = d3 - (d3 @ e) * e; f /= np.linalg.norm(f)          # second: the part of d3 orthogonal to d1
    P = {k: np.array([vec @ e, vec @ f]) for k, vec in (("d1", d1), ("d2", d2), ("d3", d3))}
    fig, ax = plt.subplots(figsize=(9, 4.1))
    arrow(ax, P["d2"], color="C0", lw=2.5)
    arrow(ax, P["d1"], color="C2", lw=2.5)
    arrow(ax, P["d3"], color="C3", lw=2.5)
    label(ax, P["d1"], "$\\mathbf{d}_1$ short ML note\n(2, 1, 0)", color="C2", offset=(0, -8), ha="center", va="top", fontsize=10)
    label(ax, P["d2"], "$\\mathbf{d}_2$ long ML article\n(8, 4, 0) = 4$\\mathbf{d}_1$", color="C0", offset=(-6, 8), ha="right", fontsize=10)
    label(ax, P["d3"], "$\\mathbf{d}_3$ football report (0, 1, 3)", color="C3", offset=(6, -4), fontsize=10)
    for q, txt, off in [("d2", r"$\|\mathbf{d}_1 - \mathbf{d}_2\| = \sqrt{45} \approx 6.71$", (0, -16)),
                        ("d3", r"$\|\mathbf{d}_1 - \mathbf{d}_3\| = \sqrt{13} \approx 3.61$", (10, 0))]:
        seg = np.array([P["d1"], P[q]])
        ax.plot(seg[:, 0], seg[:, 1], "k--", lw=1.2)
        label(ax, seg.mean(axis=0), txt, offset=off, fontsize=10, ha="center" if q == "d2" else "left")
    ang = np.degrees(np.arctan2(P["d3"][1], P["d3"][0]))
    arc = np.radians(np.linspace(0, ang, 60))
    ax.plot(0.75 * np.cos(arc), 0.75 * np.sin(arc), color="k", lw=1)
    label(ax, (0.8, 0.1), rf"$\theta \approx {ang:.1f}^\circ$" "\n" r"$\cos\theta \approx 0.141$", offset=(0, 0), fontsize=9.5, va="bottom")
    label(ax, (5.6, 0), r"angle $0^\circ$ between $\mathbf{d}_1$ and $\mathbf{d}_2$: $\cos = 1$",
          offset=(0, -34), fontsize=10, ha="center")
    ax.set_xlim(-0.7, 9.6); ax.set_ylim(-1.5, 3.9); ax.set_aspect("equal")
    ax.set_title("Euclidean distance picks the football report; the angle picks the ML article")
    ax.set_xlabel(r"coordinate along $\mathbf{d}_1$"); ax.set_ylabel("coordinate across $\\mathbf{d}_1$\n(in the plane of $\\mathbf{d}_1, \\mathbf{d}_3$)")
    fig.tight_layout()
    return fig


def fig_projection():
    """Section 5: projection of a = (2, 3) onto b = (4, 2), and the parabola of squared errors."""
    a, b = np.array([2., 3.]), np.array([4., 2.])
    t_star = a @ b / (b @ b)                               # 0.7
    p = t_star * b; r = a - p                              # (2.8, 1.4), (-0.8, 1.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax1.plot([-0.5 * b[0] / 2, 1.2 * b[0]], [-0.5 * b[1] / 2, 1.2 * b[1]], color="C1", lw=1, ls=":")
    arrow(ax1, b, color="C1", label=r"$\mathbf{b} = (4, 2)$", label_offset=(-10, -18))
    arrow(ax1, a, color="C0", label=r"$\mathbf{a} = (2, 3)$", label_offset=(-30, 6))
    arrow(ax1, p, color="C2", lw=2.6, label=r"$\mathbf{p} = 0.7\,\mathbf{b} = (2.8, 1.4)$", label_offset=(-6, -26))
    arrow(ax1, r, start=p, color="C3")
    label(ax1, p + r / 2, r"$\mathbf{r} = (-0.8, 1.6)$", color="C3", offset=(8, 0), fontsize=13, va="center")
    right_angle(ax1, p, -b, r, size=0.25)
    label(ax1, (5.1, -0.45), r"$\mathbf{r}\cdot\mathbf{b} = -3.2 + 3.2 = 0$" "\n"
          r"$\|\mathbf{a}\|^2 = \|\mathbf{p}\|^2 + \|\mathbf{r}\|^2$: $13 = 9.8 + 3.2$", offset=(0, 0), fontsize=10,
          ha="right", va="bottom")
    ax1.set_xlim(-0.6, 5.2); ax1.set_ylim(-0.6, 3.6); ax1.set_aspect("equal")
    ax1.set_title(r"(a) The shadow of $\mathbf{a}$ on the line through $\mathbf{b}$")
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")

    t = np.linspace(-0.1, 1.5, 200)
    ax2.plot(t, np.sum((a[None, :] - t[:, None] * b) ** 2, axis=1), color="C0", lw=2)
    ax2.plot(t_star, r @ r, "o", color="C3", ms=7)
    ax2.axvline(t_star, color="C3", ls="--", lw=1)
    label(ax2, (t_star, r @ r), r"minimum at $t^\star = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}} = \frac{14}{20} = 0.7$"
          "\n" r"value $\|\mathbf{r}\|^2 = 3.2$", offset=(10, 8), fontsize=10)
    ax2.plot(0, a @ a, "o", color="C0", ms=5)
    label(ax2, (0, a @ a), r"$t = 0$: $\|\mathbf{a}\|^2 = 13$", offset=(8, -2), fontsize=10)
    ax2.set_xlim(-0.15, 1.55); ax2.set_ylim(0, 16)
    ax2.set_title(r"(b) Squared error $\|\mathbf{a} - t\,\mathbf{b}\|^2 = 13 - 28t + 20t^2$")
    ax2.set_xlabel("multiple $t$ of $\\mathbf{b}$"); ax2.set_ylabel("squared distance")
    fig.tight_layout()
    return fig


def fig_hyperplane():
    """Section 7: the line 3 x1 + 4 x2 = 12, its normal w and three signed distances."""
    w, b = np.array([3., 4.]), -12.
    wn = w / np.linalg.norm(w)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    xs = np.array([-1.5, 6.5])
    line = (-b - w[0] * xs) / w[1]
    ax.fill_between(xs, line, 7, color="C3", alpha=0.08, lw=0)
    ax.fill_between(xs, -2, line, color="C0", alpha=0.08, lw=0)
    ax.plot(xs, line, "k", lw=2)
    label(ax, (5.2, 6.1), r"$\mathbf{w}\cdot\mathbf{x} + b > 0$", color="C3", offset=(0, 0), ha="center", fontsize=12)
    label(ax, (-0.6, -1.2), r"$\mathbf{w}\cdot\mathbf{x} + b < 0$", color="C0", offset=(0, 0), ha="left", fontsize=12)
    ax.text(4.2, (-b - w[0] * 4.2) / w[1] + 0.22, r"$3x_1 + 4x_2 - 12 = 0$", rotation=-36.87,
            rotation_mode="anchor", fontsize=11)
    base = np.array([0.4, 2.7])                            # a point on the line
    arrow(ax, w, start=base, color="k", lw=2, label=r"normal $\mathbf{w} = (3, 4)$", label_offset=(-120, -2))
    right_angle(ax, base, w, np.array([4., -3.]), size=0.3)
    for pt, col, txt, off in [((4., 3.), "C3", r"$(4, 3)$: $\delta = +2.4$", (8, -4)),
                              ((0., 0.), "C0", r"$(0, 0)$: $\delta = -2.4$", (6, -14)),
                              ((2., 1.5), "k", r"$(2, 1.5)$: $\delta = 0$", (10, 2))]:
        pt = np.array(pt)
        delta = (w @ pt + b) / np.linalg.norm(w)
        ax.plot(*pt, "o", color=col, ms=7, zorder=4)
        if abs(delta) > 0:
            foot = pt - delta * wn
            ax.plot([pt[0], foot[0]], [pt[1], foot[1]], color=col, ls="--", lw=1.5)
            right_angle(ax, foot, pt - foot, np.array([4., -3.]), size=0.18, color=col)
        label(ax, pt, txt, color=col, offset=off, fontsize=11)
    ax.set_xlim(-1.5, 6.5); ax.set_ylim(-2, 7); ax.set_aspect("equal")
    ax.set_title(r"$\mathbf{w}$ is perpendicular to the boundary; $\delta = (\mathbf{w}\cdot\mathbf{x} + b)/\|\mathbf{w}\|$")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    fig.tight_layout()
    return fig


def fig_high_dimensions():
    """Section 8: cosine similarity and squared distance concentrate as the dimension grows."""
    rng = np.random.default_rng(0)
    dims, n = [2, 10, 100, 1000], 20_000
    spread = 6 * np.sqrt(7 / 180)                          # std / mean of one term (x_i - y_i)^2, = 1.18
    pct = lambda x: (f"{x:.0%}" if x >= 0.1 else f"{x:.1%}").replace("%", r"\%")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for d, col in zip(dims, ["C0", "C1", "C2", "C3"]):
        U, V = rng.standard_normal((n, d)), rng.standard_normal((n, d))
        cos = np.sum(U * V, axis=1) / (np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1))
        ax1.hist(cos, bins=np.linspace(-1, 1, 161), density=True, histtype="step", lw=1.6, color=col,
                 label=rf"$d = {d}$: ${1 / np.sqrt(d):.3f}$")
        X, Y = rng.random((n, d)), rng.random((n, d))
        sq = np.sum((X - Y) ** 2, axis=1) / (d / 6)        # squared distance relative to its mean d/6
        ax2.hist(sq, bins=np.linspace(0, 3, 181), density=True, histtype="step", lw=1.6, color=col,
                 label=rf"$d = {d}$: $1.18/\sqrt{{d}} \approx {pct(spread / np.sqrt(d))}$")
    ax1.set_xlim(-1, 1); ax1.set_ylim(0, 15)
    ax1.legend(loc="upper right", fontsize=8.5, title="Gaussian; std $1/\\sqrt{d}$", title_fontsize=8.5)
    ax1.set_title(r"(a) Cosine of two random vectors: std $= 1/\sqrt{d}$")
    ax1.set_xlabel("cosine similarity"); ax1.set_ylabel("density")
    ax2.set_xlim(0, 3); ax2.set_ylim(0, 12)
    ax2.legend(loc="upper right", fontsize=8.5, title="uniform in the unit cube", title_fontsize=8.5)
    ax2.set_title(r"(b) $\|\mathbf{x} - \mathbf{y}\|^2$ / mean: relative spread $\approx 1.18/\sqrt{d}$")
    ax2.set_xlabel("squared distance divided by its mean $d/6$"); ax2.set_ylabel("density")
    fig.tight_layout()
    return fig


def fig_feature_scaling():
    """Case study 1: contribution of each feature to the squared distance, raw and standardised."""
    q = np.array([55., 140., 220., 7.0])
    cands = {r"$\mathbf{p}_1$": np.array([56., 142., 225., 9.5]),
             r"$\mathbf{p}_2$": np.array([62., 150., 200., 6.9]),
             r"$\mathbf{p}_3$": np.array([40., 120., 260., 7.1])}
    sd = np.array([15., 20., 40., 1.0])
    feats = ["age", "systolic BP", "cholesterol", "HbA1c"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, scale, title in [(axes[0], np.ones(4), r"(a) Raw units: cholesterol dominates, $\mathbf{p}_1$ nearest"),
                             (axes[1], sd, r"(b) Each feature / typical spread: $\mathbf{p}_1$ farthest")]:
        names = list(cands)
        for i, name in enumerate(names):
            parts = ((q - cands[name]) / scale) ** 2
            left = 0.0
            for j, val in enumerate(parts):
                ax.barh(i, val, left=left, color=f"C{j}", height=0.6, label=feats[j] if i == 0 else None)
                left += val
            ax.annotate(rf"$\|\cdot\| = {np.sqrt(left):.2f}$", (left, i), textcoords="offset points",
                        xytext=(5, 0), va="center", fontsize=10)
        ax.set_yticks(range(len(names)), names); ax.invert_yaxis()
        ax.set_xlim(0, 1.3 * max(np.sum(((q - p) / scale) ** 2) for p in cands.values()))
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel(r"squared distance to $\mathbf{q}$, split by feature"); ax.set_ylabel("candidate")
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    return fig


FIGURES = {
    "linear-combinations": fig_linear_combinations,
    "span-in-3d": fig_span_in_3d,
    "norms": fig_norms,
    "cosine-vs-euclidean": fig_cosine_vs_euclidean,
    "projection": fig_projection,
    "hyperplane": fig_hyperplane,
    "high-dimensions": fig_high_dimensions,
    "feature-scaling": fig_feature_scaling,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 01 concept-guide figures.")
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
