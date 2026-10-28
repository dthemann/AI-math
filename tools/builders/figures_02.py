"""Figures for the Module 02 concept guide — Matrices and Linear Maps.

Writes SVG files into part-1-linear-algebra/02-matrices-and-linear-maps/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, so picture and text agree exactly.

    python tools/builders/figures_02.py                      # (re)write every figure
    python tools/builders/figures_02.py ill-conditioning     # only the named figure(s)
    python tools/builders/figures_02.py --preview DIR        # also write PNG previews to DIR
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

FIG_DIR = ROOT / "part-1-linear-algebra/02-matrices-and-linear-maps/figures"


def _line_label(ax, text, x, slope, intercept, dy=0.08, color="k", fontsize=10, ha="left"):
    """Text written along the line y = slope * x + intercept (axes must have equal aspect)."""
    ax.text(x, slope * x + intercept + dy, text, rotation=np.degrees(np.arctan(slope)),
            rotation_mode="anchor", color=color, fontsize=fontsize, ha=ha, va="bottom")


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_column_vs_row_picture():
    """Section 1: Ax = (0, 2) as a walk along the columns (output space) and as two crossing lines (input space)."""
    A, x = np.array([[1., 2.], [3., 4.]]), np.array([2., -1.])
    a1, a2, b = A[:, 0], A[:, 1], A @ x                       # b = (0, 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.0), gridspec_kw={"width_ratios": [1.2, 1.45]})

    # (a) column picture: walk 2 a1, then -1 a2, and land on b
    arrow(ax1, x[0] * a1, color="C0", lw=1.4, alpha=0.6)
    arrow(ax1, x[1] * a2, start=x[0] * a1, color="C1", lw=1.4, alpha=0.6)
    arrow(ax1, a1, color="C0", label=r"$\mathbf{a}_1$", label_offset=(-6, -4))
    ax1.texts[-1].set_ha("right"); ax1.texts[-1].set_va("top")
    arrow(ax1, a2, color="C1", label=r"$\mathbf{a}_2$", label_offset=(7, -4))
    arrow(ax1, b, color="C3", lw=2.6, label="$A\\mathbf{x} = 2\\mathbf{a}_1 - 1\\mathbf{a}_2$\n$= (0, 2)$",
          label_offset=(-8, 0), fontsize=12)
    ax1.texts[-1].set_ha("right"); ax1.texts[-1].set_va("center")
    label(ax1, 0.75 * x[0] * a1, r"$2\mathbf{a}_1$", color="C0", offset=(7, 0), fontsize=12, va="center")
    label(ax1, x[0] * a1 + 0.5 * x[1] * a2, r"$-1\,\mathbf{a}_2$", color="C1", offset=(-7, 0), fontsize=12,
          ha="right", va="center")
    label(ax1, (3.7, -0.55), "$\\mathbf{a}_1 = (1, 3)$\n$\\mathbf{a}_2 = (2, 4)$\n$\\mathbf{x} = (2, -1)$",
          offset=(0, 0), fontsize=10, ha="right", va="bottom")
    ax1.set_xlim(-2.6, 3.9); ax1.set_ylim(-0.7, 6.8); ax1.set_aspect("equal")
    ax1.set_title("(a) Column picture (output space):\n$A\\mathbf{x}$ is a walk along the columns")
    ax1.set_xlabel("output coordinate $b_1$"); ax1.set_ylabel("output coordinate $b_2$")

    # (b) row picture: each row is a line in the input plane; x is their crossing
    xs = np.array([-2.5, 5.0])
    for i, (col, txt, lx) in enumerate([("C0", r"row 1: $x_1 + 2x_2 = 0$", -2.3),
                                        ("C1", r"row 2: $3x_1 + 4x_2 = 2$", -2.3)]):
        slope, icpt = -A[i, 0] / A[i, 1], b[i] / A[i, 1]
        ax2.plot(xs, slope * xs + icpt, color=col, lw=2)
        _line_label(ax2, txt, lx, slope, icpt, color=col, fontsize=11)
    sol = np.linalg.solve(A, b)                              # (2, -1)
    ax2.plot(*sol, "o", color="C3", ms=8, zorder=4)
    label(ax2, sol, "$\\mathbf{x} = (2, -1)$:\nboth equations hold", color="C3", offset=(10, 8), fontsize=11)
    ax2.set_xlim(-2.6, 5.1); ax2.set_ylim(-3.1, 2.9); ax2.set_aspect("equal")   # room for the label at the crossing
    ax2.set_title("(b) Row picture (input space):\neach row is a line, $\\mathbf{x}$ is where they cross")
    ax2.set_xlabel("input coordinate $x_1$"); ax2.set_ylabel("input coordinate $x_2$")
    fig.tight_layout()
    fig.align_titles()                                        # the equal-aspect panels differ in height: keep the titles level
    return fig


def fig_unit_square_determinant():
    """Section 3: the unit square under the shear H_2, the scaling diag(2, 3) and the projection onto e1."""
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]).T          # corners as columns
    H, S, P = np.array([[1., 2.], [0., 1.]]), np.diag([2., 3.]), np.array([[1., 0.], [0., 0.]])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 0.95, 1.2]})

    def base(ax, M, lab1, lab2, off1, off2, ha1="left"):
        img = M @ square
        ax.fill(img[0], img[1], color="C0", alpha=0.25, lw=0)
        ax.plot(np.r_[img[0], img[0, 0]], np.r_[img[1], img[1, 0]], color="C0", lw=1)
        ax.plot(np.r_[square[0], 0], np.r_[square[1], 0], color=GREY, ls="--", lw=1.2, label="unit square")
        arrow(ax, M[:, 0], color="C0", label=lab1, label_offset=off1, fontsize=11)
        ax.texts[-1].set_ha(ha1)
        arrow(ax, M[:, 1], color="C1", label=lab2, label_offset=off2, fontsize=11)
        ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        return img

    # (a) shear: slanted, but base 1 and height 1
    ax = axes[0]
    base(ax, H, r"$H\mathbf{e}_1 = (1, 0)$", r"$H\mathbf{e}_2 = (2, 1)$", (4, -16), (-4, 7))   # under the baseline, clear of the slanted edge
    ax.plot([2, 2], [0, 1], color="k", ls=":", lw=1.2)
    label(ax, (2, 0.5), "height 1", offset=(5, 0), fontsize=10, va="center")
    label(ax, (0.5, 0), "base 1", offset=(0, -13), fontsize=10, ha="center")
    label(ax, (-0.3, 2.0), r"$\det H_2 = 1\cdot 1 - 2\cdot 0 = 1$: area still 1", offset=(0, 0), fontsize=10, va="top")
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.6, 2.1)
    ax.set_title("(a) Shear $H_2$: corners\n$(0,0), (1,0), (3,1), (2,1)$")

    # (b) scaling: a 2 x 3 rectangle
    ax = axes[1]
    base(ax, S, r"$S\mathbf{e}_1 = (2, 0)$", r"$S\mathbf{e}_2 = (0, 3)$", (-4, -14), (6, 2), ha1="right")   # under the bottom edge
    label(ax, (1, 1.8), r"$\det S = 2\cdot 3 = 6$" "\n" r"area $\times\, 6$", offset=(0, 0), fontsize=10, ha="center", va="center")
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.8, 3.6)
    ax.set_title(r"(b) Scaling $\mathrm{diag}(2, 3)$")

    # (c) projection onto e1: the square collapses to a segment, two inputs share one output
    ax = axes[2]
    base(ax, P, r"$P\mathbf{e}_1 = (1, 0)$", None, (6, -4), (0, 0))
    ax.plot([0, 1], [0, 0], color="C0", lw=5, solid_capstyle="butt", zorder=2)
    ax.plot(0, 0, "o", color="C1", ms=7, zorder=5)
    label(ax, (0, 0), r"$P\mathbf{e}_2 = \mathbf{0}$", color="C1", offset=(-4, 7), fontsize=11, ha="right")
    for pt in [(0.5, 0.2), (0.5, 0.8)]:
        ax.plot(*pt, "o", color="C3", ms=6, zorder=5)
        ax.plot([pt[0], pt[0]], [pt[1], 0], color="C3", ls=":", lw=1.2)
        ax.annotate(f"$({pt[0]}, {pt[1]})$", pt, textcoords="offset points", xytext=(7, 0), color="C3",
                    fontsize=10, va="center", bbox=dict(fc="white", ec="none", pad=1), zorder=6)
    img_pt = P @ np.array([0.5, 0.8])                        # (0.5, 0): the same for both inputs
    ax.plot(*img_pt, "o", ms=8, mfc="white", mec="C3", mew=2, zorder=6)
    label(ax, img_pt, "$(0.5, 0)$: two inputs,\none output", color="C3", offset=(0, -12), fontsize=10, ha="center", va="top")
    label(ax, (-0.4, 1.45), r"$\det P = 0$: area 0, $y$ is lost", offset=(0, 0), fontsize=10, va="top")
    ax.set_xlim(-0.75, 2.15); ax.set_ylim(-0.75, 1.55)
    ax.set_title("(c) Projection onto $\\mathbf{e}_1$:\na dimension is destroyed")
    fig.tight_layout()
    fig.align_titles()                                        # the equal-aspect panels differ in height: keep the titles level
    return fig


def fig_quadratic_forms():
    """Section 4: a rotation keeps lengths; x^T A x is a bowl for a positive definite A and a saddle otherwise."""
    Q = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
    x = np.array([3., 4.]); Qx = Q @ x                       # (-1, 7)/sqrt(2), length 5
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # (a) orthogonal Q: both arrows end on the circle of radius 5
    th = np.linspace(0, np.pi, 200)
    ax1.plot(5 * np.cos(th), 5 * np.sin(th), color=GREY, ls="--", lw=1)
    arrow(ax1, x, color="C0", label=r"$\mathbf{x} = (3, 4)$", label_offset=(6, -2), fontsize=12)
    arrow(ax1, Qx, color="C1", label=r"$Q\mathbf{x} = (-1, 7)/\sqrt{2}$", label_offset=(6, 8), fontsize=12)
    t0, t1 = np.arctan2(x[1], x[0]), np.arctan2(Qx[1], Qx[0])
    arc = np.linspace(t0, t1, 50)
    ax1.plot(1.6 * np.cos(arc), 1.6 * np.sin(arc), color="k", lw=1)
    label(ax1, (1.9 * np.cos((t0 + t1) / 2), 1.9 * np.sin((t0 + t1) / 2)), r"$45^\circ$", offset=(0, 0), fontsize=11, ha="center", va="center")
    label(ax1, (4.6, 0.3), r"$\|Q\mathbf{x}\| = \|\mathbf{x}\| = 5$" "\n" r"$\|Q\mathbf{x}\|^2 = \frac{1}{2}(1 + 49) = 25$", offset=(0, 0), fontsize=10, ha="right", va="bottom")
    ax1.set_xlim(-2.6, 5.6); ax1.set_ylim(-0.5, 6.0); ax1.set_aspect("equal")
    ax1.set_title("(a) Orthogonal $Q$ (rotation by $45^\\circ$):\nlengths are kept")
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")

    # (b) and (c): level sets of the two quadratic forms
    g1, g2 = np.meshgrid(np.linspace(-2.4, 2.4, 401), np.linspace(-2.4, 2.4, 401))   # = the axes limits, so contour labels stay in view
    for ax, M, name in [(ax2, np.array([[2., 1.], [1., 2.]]), "f"), (ax3, np.array([[1., 2.], [2., 1.]]), "g")]:
        val = M[0, 0] * g1 ** 2 + 2 * M[0, 1] * g1 * g2 + M[1, 1] * g2 ** 2
        if name == "f":
            cs = ax.contour(g1, g2, val, levels=[1, 2, 4, 6], colors="C0", linewidths=1.2)
            ax.clabel(cs, fmt="%g", fontsize=10, inline=True,   # by hand, on the arrow-free diagonals: f = 1, 2, 4, 6
                      manual=[(-0.71, 0.71), (-0.58, -0.58), (-1.41, 1.41), (-1.0, -1.0)])
        else:
            ax.contourf(g1, g2, val, levels=[-50, 0], colors=["C3"], alpha=0.08)
            cs = ax.contour(g1, g2, val, levels=[-2, -1, 0, 1, 2, 6],
                            colors=["C3", "C3", "k", "C0", "C0", "C0"], linewidths=1.2,
                            linestyles=["dashed", "dashed", "solid", "solid", "solid", "solid"])
            ax.clabel(cs, levels=[6, -2], fmt="%g", fontsize=10, inline=True,   # only the two levels the caption discusses:
                      manual=[(2.0, 0.24), (-2.0, -0.24), (-1.25, 0.86)])   # g = 6 twice; g = -2 once, on the open upper-left arm near its
            # vertex, where the g = -1 and g = -2 arms are furthest apart (the lower-right arm carries the g(1, -1) = -2 box instead)
        for p in [np.array([1., 1.]), np.array([1., -1.])]:
            arrow(ax, p, color=GREY, lw=1.2, zorder=2)
            ax.plot(*p, "o", color="C3", ms=6, zorder=5)
            up = p[1] > 0                                     # two-line value label on a white box, so it ends inside the frame
            ax.annotate(rf"${name}({p[0]:g}, {p[1]:g})$" "\n" rf"$= {p @ M @ p:g}$", p, textcoords="offset points",
                        xytext=(7, 3) if up else (7, -3), color="C3", fontsize=10, ha="left", va="bottom" if up else "top",
                        bbox=dict(fc="white", ec="none", pad=1), zorder=6)
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    label(ax2, (-2.3, -2.3), "short axis $(1, 1)$,\nlong axis $(1, -1)$", offset=(0, 0), fontsize=9, va="bottom")   # short lines: the f = 6 ellipse dips to (0, -1.73)
    ax2.set_title("(b) Positive definite $A = [[2, 1], [1, 2]]$:\n$f = \\mathbf{x}^\\top\\! A\\mathbf{x} = 2x_1^2 + 2x_1x_2 + 2x_2^2$ is a bowl")
    label(ax3, (-2.3, -2.3), "shaded: $g < 0$", offset=(0, 0), fontsize=9, va="bottom", color="C3")
    ax3.set_title("(c) Symmetric but not PSD:\n$g = x_1^2 + 4x_1x_2 + x_2^2$ is a saddle")
    fig.tight_layout()
    return fig


def fig_column_space_and_null_space():
    """Section 5: the columns of M lie in one plane of R^3 (b sticks out); the null line and the parallel solution line."""
    M = np.array([[1., 2., 3.], [2., 4., 6.], [1., 0., 1.]])
    c1, c2, c3 = M.T
    b = np.array([2., 1., 1.])                                # b2 = 1 is not 2 b1 = 4, so b is off the plane
    n = np.cross(c1, c2)                                      # (-4, 2, 0): the plane C(M) is x2 = 2 x1
    foot = b - (b @ n) / (n @ n) * n                          # (0.8, 1.6, 1.0), the point of C(M) nearest b
    z = np.array([1., 1., -1.])                               # M z = 0: the null-space direction
    xp = np.array([0., 0., 1.])                               # M xp = c3
    assert np.allclose(M @ z, 0) and np.allclose(M @ xp, c3) and np.allclose(M.T @ n, 0)
    assert np.allclose(foot, [0.8, 1.6, 1.0]) and np.isclose(abs(b @ n) / np.linalg.norm(n), 6 / np.sqrt(20))
    fig = plt.figure(figsize=(10, 4.9))

    # (a) output space: the plane x2 = 2 x1 holds all three columns; b = (2, 1, 1) is off it
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u, v = np.meshgrid(np.linspace(-0.3, 3.4, 2), np.linspace(-0.6, 1.8, 2))
    ax.plot_surface(u, 2 * u, v, color="C0", alpha=0.15, shade=False, lw=0)
    for k in np.linspace(-0.3, 3.4, 12):                      # faint in-plane lines help depth perception
        ax.plot([k, k], [2 * k, 2 * k], [-0.6, 1.8], color="C0", lw=0.4, alpha=0.35)
    for vec, col, lab, pos, ha, va in [(c1, "C0", r"$\mathbf{c}_1 = (1, 2, 1)$", (0.95, 2.0, 1.25), "right", "bottom"),
                                       (c2, "C1", r"$\mathbf{c}_2 = (2, 4, 0)$", (2.1, 4.1, -0.4), "left", "top"),
                                       (c3, "C2", r"$\mathbf{c}_3 = \mathbf{c}_1 + \mathbf{c}_2 = (3, 6, 1)$", (2.95, 6.0, 1.25), "right", "bottom"),
                                       (b, "C3", r"$\mathbf{b} = (2, 1, 1)$", (2.3, 0.85, 0.95), "left", "top")]:
        ax.quiver(0, 0, 0, *vec, color=col, lw=2.2, arrow_length_ratio=0.1)
        ax.text(*pos, lab, color=col, fontsize=10.5, ha=ha, va=va)
    path = np.array([np.zeros(3), c1, c3])                    # c3 reached tip-to-tail: c1, then c2
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="C1", ls="--", lw=1.2)
    ax.plot(*np.array([b, foot]).T, color="C3", ls=":", lw=1.5)    # dotted: the perpendicular from b to the plane
    ax.text(3.4, 6.8, 1.85, "$\\mathcal{C}(M)$: the plane $x_2 = 2x_1$", color="C0", fontsize=10.5, ha="right", va="bottom")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 6.5); ax.set_zlim(-0.6, 2.0)
    ax.set_xticks([0, 1, 2, 3]); ax.set_zticks([0, 1, 2])   # fewer tick labels in the crowded bottom-left corner
    ax.set_box_aspect((4.0, 7.0, 2.6))                        # equal scale on all three axes
    ax.view_init(elev=40, azim=-62)                           # high enough that the tip of b is seen clear of c3
    ax.set_title("(a) Output space: all three columns lie in one plane, so $\\operatorname{rank} M = 2$;\n"
                 "$\\mathbf{b} = (2, 1, 1)$ is off that plane, so $M\\mathbf{x} = \\mathbf{b}$ has no solution")

    # (b) input space: the null line t(1, 1, -1) and the parallel line of solutions of M x = c3
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    L0 = np.outer([-1.3, 1.5], z)
    ax.plot(L0[:, 0], L0[:, 1], L0[:, 2], color="C3", lw=2.5)
    ax.quiver(0, 0, 0, *z, color="C3", lw=2.2, arrow_length_ratio=0.12)
    L1 = xp + np.outer([-0.8, 1.4], z)
    ax.plot(L1[:, 0], L1[:, 1], L1[:, 2], color="C2", ls="--", lw=2)
    for s in [xp, xp + z, xp + 0.5 * z]:                      # (0,0,1), (1,1,0), (0.5,0.5,0.5): all map to c3
        assert np.allclose(M @ s, c3)
        ax.scatter(*s, color="C2", s=30, zorder=5)
        ax.text(s[0] + 0.08, s[1] - 0.05, s[2] + 0.12, f"$({s[0]:g}, {s[1]:g}, {s[2]:g})$", color="C2", fontsize=10)
    ax.scatter(0, 0, 0, color="k", s=15)
    ax.text2D(0.02, 0.98, "$\\mathcal{N}(M) = t\\,(1, 1, -1)$:\nall of it maps to $\\mathbf{0}$",
              color="C3", fontsize=10.5, transform=ax.transAxes, ha="left", va="top")
    ax.text2D(0.98, 0.98, "dashed: $(0, 0, 1) + t\\,(1, 1, -1)$;\nevery point of it solves $M\\mathbf{x} = \\mathbf{c}_3$",
              color="C2", fontsize=10.5, transform=ax.transAxes, ha="right", va="top")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    ax.set_xlim(-1.5, 1.8); ax.set_ylim(-1.5, 1.8); ax.set_zlim(-1.5, 1.8)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-60)
    ax.set_title("(b) Input space: the null-space line, and the parallel\nline of solutions of $M\\mathbf{x} = \\mathbf{c}_3$")
    fig.tight_layout()
    return fig


def fig_ill_conditioning():
    """Section 6: row pictures of a well-conditioned and an ill-conditioned 2x2 system under a nudge of b."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.7), gridspec_kw={"width_ratios": [1, 0.95]})

    # (a) well conditioned: rows cross at 45 degrees, a visible nudge moves x a little
    A, b0, b1 = np.array([[2., 1.], [1., 3.]]), np.array([5., 10.]), np.array([5., 10.5])
    x0, x1 = np.linalg.solve(A, b0), np.linalg.solve(A, b1)   # (1, 3) and (0.9, 3.2)
    kappa = np.linalg.cond(A)
    rel_b, rel_x = abs(b1[1] - b0[1]) / b0[1], np.linalg.norm(x1 - x0) / np.linalg.norm(x0)   # 5 % and 7 %
    xs1, xs2 = np.array([-0.3, 2.7]), np.array([-1., 4.])
    ax1.plot(xs1, (b0[0] - A[0, 0] * xs1) / A[0, 1], color="C0", lw=2, label=r"$2x_1 + x_2 = 5$")
    ax1.plot(xs2, (b0[1] - A[1, 0] * xs2) / A[1, 1], color="C1", lw=2, label=r"$x_1 + 3x_2 = 10$")
    ax1.plot(xs2, (b1[1] - A[1, 0] * xs2) / A[1, 1], color="C1", lw=1.4, ls="--", label=r"$x_1 + 3x_2 = 10.5$: $b_2$ nudged by $0.5$")
    ax1.plot(*x0, "o", color="C3", ms=8, zorder=4)
    ax1.plot(*x1, "o", ms=8, mfc="white", mec="C3", mew=2, zorder=4)
    label(ax1, x0, r"$\mathbf{x} = (1, 3)$", color="C3", offset=(-8, -14), fontsize=11, ha="right")
    label(ax1, x1, r"$\mathbf{x}' = (0.9, 3.2)$", color="C3", offset=(8, 6), fontsize=11)
    label(ax1, (-0.9, 0.15), f"the rows cross at $45^\\circ$,\n$\\kappa = {kappa:.1f}$; $b_2$ moved {100 * rel_b:.0f} %,\n"
          f"$\\mathbf{{x}}$ moved {100 * rel_x:.0f} % (by {np.linalg.norm(x1 - x0):.2f})",
          offset=(0, 0), fontsize=10, ha="left", va="bottom")   # bottom-left corner, in lines short enough to end left of the blue line
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax1.set_xlim(-1, 4); ax1.set_ylim(0, 5.6); ax1.set_aspect("equal")
    ax1.set_title("(a) Well conditioned: a visible nudge, a small move")
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")

    # (b) ill conditioned: the rows are 0.003 degrees apart, an invisible nudge moves x by 1.41
    A, b0, b1 = np.array([[1., 1.], [1., 1.0001]]), np.array([2., 2.0001]), np.array([2., 2.0002])
    x0, x1 = np.linalg.solve(A, b0), np.linalg.solve(A, b1)   # (1, 1) and (0, 2)
    assert np.allclose(x0, [1, 1]) and np.allclose(x1, [0, 2])
    kappa = np.linalg.cond(A)
    rel_b, rel_x = abs(b1[1] - b0[1]) / b0[1], np.linalg.norm(x1 - x0) / np.linalg.norm(x0)   # 5e-5 and 100 %
    deg = np.degrees(np.arccos(A[0] @ A[1] / (np.linalg.norm(A[0]) * np.linalg.norm(A[1]))))
    xs = np.array([-0.7, 2.7])
    ax2.plot(xs, (b0[1] - A[1, 0] * xs) / A[1, 1], color="C1", lw=5, alpha=0.5, solid_capstyle="butt",
             label=r"$x_1 + 1.0001\,x_2 = 2.0001$")
    ax2.plot(xs, (b0[0] - A[0, 0] * xs) / A[0, 1], color="C0", lw=1.5, label=r"$x_1 + x_2 = 2$")
    ax2.plot(xs, (b1[1] - A[1, 0] * xs) / A[1, 1], color="C1", lw=1.2, ls="--", label=r"$x_1 + 1.0001\,x_2 = 2.0002$: $b_2$ nudged")
    ax2.plot(*x0, "o", color="C3", ms=8, zorder=4)
    ax2.plot(*x1, "o", ms=8, mfc="white", mec="C3", mew=2, zorder=4)
    label(ax2, x0, "$\\mathbf{x} = (1, 1)$\nfor $\\mathbf{b} = (2, 2.0001)$", color="C3", offset=(-9, -9), fontsize=11,
          ha="right", va="top")
    label(ax2, x1, "$\\mathbf{x}' = (0, 2)$\nfor $\\mathbf{b} = (2, 2.0002)$", color="C3", offset=(10, 2), fontsize=11)
    shift = np.array([0.12, 0.12]) + np.array([0.25, -0.25])  # the move, drawn beside the line and slid down it, clear of the x' label
    arrow(ax2, x1 - x0, start=x0 + shift, color="C3", lw=1.8)
    label(ax2, (x0 + x1) / 2 + shift, f"$\\mathbf{{x}}$ moves by {np.linalg.norm(x1 - x0):.2f}", color="C3", offset=(6, 4), fontsize=10)
    label(ax2, (2.7, 3.0), f"the rows are only ${deg:.3f}^\\circ$ apart; $\\kappa = {kappa / 1e4:.0f} \\times 10^4$\n"
          f"$b_2$ moved {100 * rel_b:.3f} %, $\\mathbf{{x}}$ moved {100 * rel_x:.0f} %",
          offset=(0, 0), fontsize=10, ha="right", va="top")
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.95, title="three lines, one on top of another", title_fontsize=9)
    ax2.set_xlim(-0.8, 2.8); ax2.set_ylim(-0.8, 3.05); ax2.set_aspect("equal")   # head-room for the note above the x' label
    ax2.set_title("(b) Ill conditioned: an invisible nudge, a huge move")
    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    fig.tight_layout()
    return fig


def fig_least_squares_projection():
    """Section 7: the fitted line with its four residuals, and the same numbers as a projection in R^4."""
    t, y = np.arange(4.), np.array([1., 2., 2., 4.])
    A = np.c_[np.ones(4), t]
    xh = np.linalg.solve(A.T @ A, A.T @ y)                    # (0.9, 0.9)
    yh, r = A @ xh, y - A @ xh                                # fitted (0.9, 1.8, 2.7, 3.6), residuals (0.1, 0.2, -0.7, 0.4)
    assert np.allclose(xh, 0.9) and np.allclose(A.T @ r, 0)
    comp = 1 + t                                              # the competing line y = 1 + t, fitted vector (1, 2, 3, 4)
    rss, rss_c = r @ r, np.sum((y - comp) ** 2)               # 0.70 and 1.00
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.9]})

    # (a) data space: the line and its vertical residuals
    tt = np.array([-0.3, 3.3])
    ax1.plot(tt, xh[0] + xh[1] * tt, color="C0", lw=2, label=rf"$\hat y = 0.9 + 0.9\,t$: RSS $= {rss:.2f}$")
    ax1.plot(tt, 1 + tt, color=GREY, lw=1.4, ls="--", label=rf"$y = 1 + t$: RSS $= {rss_c:.2f}$")
    for ti, yi, fi, ri in zip(t, y, yh, r):
        ax1.plot([ti, ti], [fi, yi], color="C3", lw=2.2, zorder=2)
        if ri > 0:                                            # above-left of the data point, where neither line passes
            label(ax1, (ti, yi), f"${ri:+.1f}$", color="C3", offset=(-4, 5), fontsize=10, ha="right", va="bottom")
        else:
            label(ax1, (ti, (fi + yi) / 2), f"${ri:+.1f}$", color="C3", offset=(8, 0), fontsize=10, va="center")
    ax1.plot(t, yh, "o", color="C0", ms=4, zorder=3)
    ax1.plot(t, y, "ko", ms=7, zorder=4, label="data $(t, y)$")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)
    label(ax1, (3.4, 0.15), "RSS $= 0.01 + 0.04 + 0.49 + 0.16 = 0.70$\nresiduals sum to 0", offset=(0, 0), fontsize=9.5,
          ha="right", va="bottom")
    ax1.set_xlim(-0.65, 3.8); ax1.set_ylim(0, 5)               # room on the left for the '+0.1' label
    ax1.set_title("(a) Data space: the line and its four residuals")
    ax1.set_xlabel("$t$ (dose, units of 10 mg)"); ax1.set_ylabel("$y$")

    # (b) R^4, drawn true to scale in the plane spanned by y-hat and r (the column space is seen edge-on)
    ny, nyh, nr = np.linalg.norm(y), np.linalg.norm(yh), np.linalg.norm(r)      # 5, 4.93, 0.84
    assert np.isclose(ny ** 2, nyh ** 2 + nr ** 2)
    e1 = yh / nyh
    cx = comp @ e1                                            # 5.48: (1, 2, 3, 4) is a multiple of y-hat, so it lies on this axis
    assert np.isclose(np.linalg.norm(comp - cx * e1), 0)
    Yh, Y, C = np.array([nyh, 0.]), np.array([nyh, nr]), np.array([cx, 0.])
    ax2.fill_between([-0.3, 6.4], -0.14, 0.14, color="C0", alpha=0.12, lw=0)
    label(ax2, (-0.3, -0.22), "$\\mathcal{C}(A)$\nseen edge-on", color="C0", offset=(0, 0), fontsize=10, va="top")
    arrow(ax2, Yh, color="C0", lw=2.6)
    label(ax2, (3.05, -0.22), r"$\hat{\mathbf{y}} = H\mathbf{y}$, length $\sqrt{24.3} = 4.93$", color="C0", offset=(0, 0), fontsize=11,
          ha="center", va="top")
    arrow(ax2, Y, color="k", lw=2)
    label(ax2, Y / 2, r"$\mathbf{y}$, length 5", color="k", offset=(-4, 10), fontsize=11, ha="center")
    arrow(ax2, Y - Yh, start=Yh, color="C3", lw=2.2)
    label(ax2, (Yh[0] + 0.55, Y[1] / 2), "$\\mathbf{r} = \\mathbf{y} - \\hat{\\mathbf{y}}$\nlength $\\sqrt{0.7} = 0.84$", color="C3",
          offset=(0, 0), fontsize=11, va="center")
    right_angle(ax2, Yh, (-1, 0), (0, 1), size=0.2)
    ax2.plot(*C, "o", color=GREY, ms=6, zorder=4)
    ax2.plot([C[0], Y[0]], [C[1], Y[1]], color=GREY, ls="--", lw=1.2)
    label(ax2, (7.35, -0.22), "$(1, 2, 3, 4)$, the fit $y = 1 + t$:\ndistance to $\\mathbf{y}$ is $\\sqrt{1.00} = 1.00 > 0.84$", color="0.4",
          offset=(0, 0), fontsize=9, ha="right", va="top")
    label(ax2, (0.0, 1.65), r"$\|\mathbf{y}\|^2 = \|\hat{\mathbf{y}}\|^2 + \|\mathbf{r}\|^2$: $25 = 24.3 + 0.7$;"
          rf" angle between $\mathbf{{y}}$ and $\hat{{\mathbf{{y}}}}$: ${np.degrees(np.arccos(nyh / ny)):.1f}^\circ$",
          offset=(0, 0), fontsize=10, va="top")
    ax2.set_xlim(-0.4, 7.4); ax2.set_ylim(-0.9, 1.75); ax2.set_aspect("equal")
    ax2.set_title(r"(b) In $\mathbb{R}^4$: $\hat{\mathbf{y}}$ is the shadow of $\mathbf{y}$ on $\mathcal{C}(A)$, $\mathbf{r}$ is perpendicular to it")
    ax2.set_xlabel(r"coordinate along $\hat{\mathbf{y}}$"); ax2.set_ylabel(r"coordinate along $\mathbf{r}$")
    fig.tight_layout()
    fig.align_titles()                                        # the equal-aspect panel (b) is shorter than (a): keep the titles level
    return fig


def fig_change_of_basis():
    """Section 8: x = (3, 1) and A x = (7, 5) read on the p-grid: coordinates (2, 1) -> (6, 1) = (3*2, 1*1)."""
    P = np.array([[1., 1.], [1., -1.]]); p1, p2 = P.T
    A = np.array([[2., 1.], [1., 2.]])
    x = np.array([3., 1.]); c = np.linalg.solve(P, x)         # (2, 1)
    Ax = A @ x; c2 = np.linalg.solve(P, Ax)                   # (7, 5) and (6, 1)
    D = np.linalg.solve(P, A @ P)                             # diag(3, 1)
    assert np.allclose(c, [2, 1]) and np.allclose(c2, [6, 1]) and np.allclose(D, np.diag([3., 1.]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.1))
    ts = np.array([-20., 20.])
    for ax in (ax1, ax2):
        for k in range(-6, 9):                                # the p-grid: integer multiples of p1 and p2
            ax.plot(k * p1[0] + ts * p2[0], k * p1[1] + ts * p2[1], color=GREY, lw=0.6, alpha=0.5, zorder=0)
            ax.plot(k * p2[0] + ts * p1[0], k * p2[1] + ts * p1[1], color=GREY, lw=0.6, alpha=0.5, zorder=0)
        ax.set_xlim(-1.6, 9.6); ax.set_ylim(-2.6, 7.1); ax.set_aspect("equal")
        ax.grid(False)                                        # the p-grid is the only grid
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

    # (a) one arrow, two coordinate systems
    arrow(ax1, c[0] * p1, color="C0", lw=1.4, alpha=0.6)
    arrow(ax1, c[1] * p2, start=c[0] * p1, color="C1", lw=1.4, alpha=0.6)
    arrow(ax1, p1, color="C0", label=r"$\mathbf{p}_1 = (1, 1)$", label_offset=(-8, 4), fontsize=11)
    ax1.texts[-1].set_ha("right")
    arrow(ax1, p2, color="C1", label=r"$\mathbf{p}_2 = (1, -1)$", label_offset=(6, -6), fontsize=11)
    arrow(ax1, x, color="C3", lw=2.6, label=r"$\mathbf{x}$", label_offset=(6, -10))
    label(ax1, (-1.2, 6.8), "$\\mathbf{x} = (3, 1)$ in standard coordinates\n$= (2, 1)$ in the $\\mathbf{p}$-basis",
          color="C3", offset=(0, 0), fontsize=11, va="top")    # explained in the empty upper left, as in (b)
    label(ax1, c[0] * p1 * 0.75, r"$2\mathbf{p}_1$", color="C0", offset=(-7, 2), fontsize=12, ha="right")
    label(ax1, c[0] * p1 + 0.5 * c[1] * p2, r"$1\mathbf{p}_2$", color="C1", offset=(-5, -4), fontsize=12, ha="right", va="top")
    ax1.set_title("(a) One arrow, two coordinate systems: $\\mathbf{x} = 2\\mathbf{p}_1 + 1\\mathbf{p}_2$\n(grey: the grid of the $\\mathbf{p}$-basis)")

    # (b) the same map A, read on the p-grid, is diag(3, 1)
    arrow(ax2, c2[0] * p1, color="C0", lw=1.4, alpha=0.6)
    arrow(ax2, c2[1] * p2, start=c2[0] * p1, color="C1", lw=1.4, alpha=0.6)
    arrow(ax2, x, color="C3", lw=1.4, alpha=0.35, label=r"$\mathbf{x}$", label_offset=(6, -8))
    ax2.texts[-1].set_alpha(0.6)
    arrow(ax2, A @ p1, color="C0", label=r"$A\mathbf{p}_1 = (3, 3) = 3\mathbf{p}_1$", label_offset=(-8, 4), fontsize=11)
    ax2.texts[-1].set_ha("right")
    arrow(ax2, A @ p2, color="C1", label=r"$A\mathbf{p}_2 = (1, -1) = 1\mathbf{p}_2$", label_offset=(6, -6), fontsize=11)
    arrow(ax2, Ax, color="C3", lw=2.6)
    label(ax2, (9.4, 2.4), "$A\\mathbf{x} = (7, 5)$\n$= (6, 1)$ in the $\\mathbf{p}$-basis\n$= (3\\cdot 2,\\ 1\\cdot 1)$",
          color="C3", offset=(0, 0), fontsize=12, ha="right", va="top")
    label(ax2, c2[0] * p1 * 0.75, r"$6\mathbf{p}_1$", color="C0", offset=(-7, 2), fontsize=12, ha="right")
    label(ax2, c2[0] * p1 + 0.5 * c2[1] * p2, r"$1\mathbf{p}_2$", color="C1", offset=(6, 2), fontsize=12)
    ax2.set_title("(b) The same map $A$ read in $\\mathbf{p}$-coordinates\nis $P^{-1}AP = \\mathrm{diag}(3, 1)$")
    fig.tight_layout()
    return fig


def fig_markov_chain():
    """Section 9: the two-state weather chain, convergence to (5/6, 1/6) from both starts, and the geometric rate 0.4."""
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    pi_inf = np.array([5., 1.]) / 6
    assert np.allclose(pi_inf @ P, pi_inf)
    lam2 = np.sort(np.linalg.eigvals(P))[0].real             # 0.4
    T = 8
    traj = {}
    for name, start in [("Rainy", (0., 1.)), ("Sunny", (1., 0.))]:
        pi, seq = np.array(start), [start[0]]
        for _ in range(T):
            pi = pi @ P; seq.append(pi[0])
        traj[name] = np.array(seq)
    assert np.allclose(traj["Rainy"][:4], [0, 0.5, 0.70, 0.78])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})

    # (a) the chain as a diagram: the arrows out of each state carry its row of P
    R = 0.6
    centres = {"Sunny": np.array([0., 0.]), "Rainy": np.array([3., 0.])}
    for (name, cen), col in zip(centres.items(), ["C0", "C1"]):
        ax1.add_patch(Circle(cen, R, facecolor=col, alpha=0.2, edgecolor=col, lw=1.5))
        ax1.text(*cen, name, ha="center", va="center", fontsize=11)
    pt = lambda name, deg: centres[name] + R * np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    style = dict(arrowstyle="-|>", mutation_scale=14, lw=1.6, color="k")
    ax1.add_patch(FancyArrowPatch(pt("Sunny", 30), pt("Rainy", 150), connectionstyle="arc3,rad=-0.35", **style))
    ax1.add_patch(FancyArrowPatch(pt("Rainy", 210), pt("Sunny", -30), connectionstyle="arc3,rad=-0.35", **style))
    ax1.add_patch(FancyArrowPatch(pt("Sunny", 60), pt("Sunny", 120), connectionstyle="arc3,rad=1.8", **style))
    ax1.add_patch(FancyArrowPatch(pt("Rainy", 60), pt("Rainy", 120), connectionstyle="arc3,rad=1.8", **style))
    for xy, txt in [((1.5, 0.88), "0.1"), ((1.5, -0.9), "0.5"), ((0, 1.22), "0.9"), ((3, 1.22), "0.5")]:
        ax1.text(*xy, txt, ha="center", va="center", fontsize=11)
    ax1.set_xlim(-1.2, 4.2); ax1.set_ylim(-1.5, 1.8); ax1.set_aspect("equal")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(False)
    ax1.set_xlabel("$P = [[0.9, 0.1], [0.5, 0.5]]$:\nthe arrows out of each state sum to 1")
    ax1.set_ylabel("states")
    ax1.set_title("(a) The chain as a diagram")

    # (b) P(Sunny) over time from the two starts
    days = np.arange(T + 1)
    ax2.axhline(pi_inf[0], color="C3", ls="--", lw=1.2, label=r"stationary $\pi_S = 5/6$")
    ax2.plot(days, traj["Sunny"], "s-", color="C0", ms=5, lw=1.5, label="start Sunny, $(1, 0)$")
    ax2.plot(days, traj["Rainy"], "o-", color="C1", ms=5, lw=1.5, label="start Rainy, $(0, 1)$")
    for d in (1, 2, 3):
        label(ax2, (d, traj["Rainy"][d]), f"{traj['Rainy'][d]:.2f}", color="C1", offset=(4, -12), fontsize=9)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax2.set_xlim(-0.3, T + 0.3); ax2.set_ylim(-0.04, 1.04)      # room for the markers on the limits
    ax2.set_title("(b) Two starts, one limit")
    ax2.set_xlabel("day $t$"); ax2.set_ylabel("$\\pi_t(\\mathrm{Sunny})$")

    # (c) the gap to 5/6 shrinks by the factor 0.4 every day
    gap_r, gap_s = np.abs(traj["Rainy"] - pi_inf[0]), np.abs(traj["Sunny"] - pi_inf[0])
    assert np.allclose(gap_r[1:] / gap_r[:-1], lam2)
    ax3.semilogy(days, gap_r[0] * lam2 ** days, color="C3", ls="--", lw=1.2, label=r"$(5/6)\cdot 0.4^{\,t}$")
    ax3.semilogy(days, gap_r, "o", color="C1", ms=5, label="start Rainy")
    ax3.semilogy(days, gap_s, "s", color="C0", ms=5, label="start Sunny")
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax3.set_xlim(-0.3, T + 0.3); ax3.set_ylim(3e-5, 2)
    ax3.set_title("(c) Each day the gap shrinks\n" rf"by the factor $\lambda_2 = {lam2:.1f}$")
    ax3.set_xlabel("day $t$"); ax3.set_ylabel(r"$|\pi_t(\mathrm{Sunny}) - 5/6|$ (log scale)")
    fig.tight_layout()
    fig.align_titles()                                        # the equal-aspect diagram (a) is shorter than (b), (c): keep the titles level
    return fig


FIGURES = {
    "column-vs-row-picture": fig_column_vs_row_picture,
    "unit-square-determinant": fig_unit_square_determinant,
    "quadratic-forms": fig_quadratic_forms,
    "column-space-and-null-space": fig_column_space_and_null_space,
    "ill-conditioning": fig_ill_conditioning,
    "least-squares-projection": fig_least_squares_projection,
    "change-of-basis": fig_change_of_basis,
    "markov-chain": fig_markov_chain,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 02 concept-guide figures.")
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
