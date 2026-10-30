"""Figures for the Module 03 concept guide — Eigenvalues, SVD and PCA.

Writes SVG files into part-1-linear-algebra/03-eigenvalues-svd-and-pca/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, so picture and text agree exactly: each function recomputes its
numbers from the guide's matrices and checks them against the printed values.

    python tools/builders/figures_03.py                      # (re)write every figure
    python tools/builders/figures_03.py svd-rotate-stretch   # only the named figure(s)
    python tools/builders/figures_03.py --preview DIR        # also write PNG previews to DIR
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FIG_DIR = ROOT / "part-1-linear-algebra/03-eigenvalues-svd-and-pca/figures"

TH = np.linspace(0, 2 * np.pi, 400)
CIRCLE = np.array([np.cos(TH), np.sin(TH)])            # the unit circle, 2 x 400


def close(got, expected, atol=5e-4):
    """Assert that a recomputed number agrees with the value printed in the guide."""
    np.testing.assert_allclose(got, expected, atol=atol, rtol=0)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def deg(v):
    """Direction of the vector v in degrees."""
    return np.degrees(np.arctan2(v[1], v[0]))


def line(ax, d, through=(0, 0), span=(-10, 10), **kw):
    """The line through the point `through` along the direction d (parameter range `span`)."""
    d, p = unit(d), np.asarray(through, float)
    pts = np.array([p + span[0] * d, p + span[1] * d])
    ax.plot(pts[:, 0], pts[:, 1], **kw)


def axes_2d(ax, title, xlim, ylim, xlabel="$x_1$", ylabel="$x_2$"):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def tight_layout(fig, max_width=10.0):
    """fig.tight_layout(), narrowed if needed so that the saved SVG is at most `max_width` inches wide.

    tight_layout() ignores the width of axes titles, so a two-line title over a narrow
    outer panel can stick out past the figure edge; save() crops to everything drawn
    (bbox_inches="tight") plus 0.1 in of padding per side, and the SVG grows beyond 10 in.
    """
    fig.tight_layout()
    renderer, margin = fig.canvas.get_renderer(), 0.0
    limit = max_width - 2 * plt.rcParams["savefig.pad_inches"] - 0.02   # the SVG backend sets text a hair wider than Agg
    for _ in range(4):
        fig.canvas.draw()
        excess = fig.get_tightbbox(renderer).width - limit          # inches
        if excess <= 0:
            return
        margin += excess / 2 / fig.get_figwidth() + 0.001
        fig.tight_layout(rect=(margin, 0, 1 - margin, 1))


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_eigen_directions():
    """Section 1: A = [[4,1],[2,3]] only stretches two directions; a rotation keeps none, a shear one."""
    A = np.array([[4., 1.], [2., 3.]])
    close(np.sort(np.linalg.eigvals(A)), [2, 5])
    v1, v2 = unit([1, 1]), unit([1, -2])                  # the guide's (1, 1) and (1, -2), unit length
    close(A @ v1, 5 * v1); close(A @ v2, 2 * v2)
    angle = np.degrees(np.arccos(abs(v1 @ v2))); close(angle, 71.57, atol=0.01)   # between the eigen-lines
    e1, e2 = np.array([1., 0.]), np.array([0., 1.])
    Ae1 = A @ e1; close(Ae1, [4, 2]); close(deg(Ae1), 26.57, atol=0.01)
    R = np.array([[0., -1.], [1., 0.]])                   # rotation by 90 degrees
    close(np.sort_complex(np.linalg.eigvals(R)), [-1j, 1j])
    Sh = np.array([[1., 1.], [0., 1.]])                   # shear
    close(np.linalg.eigvals(Sh), [1, 1])
    assert np.linalg.matrix_rank(Sh - np.eye(2)) == 1     # eigenvalue 1 is double, eigenspace is a line
    close(Sh @ e1, e1); close(Sh @ e2, [1, 1]); close(Sh @ -e2, [-1, -1])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.45, 1, 1.05]})

    # (a) the two eigen-lines, with the unit eigenvectors v^_i = v_i/|v_i| and A v^_i on them; e1 is turned off its line
    line(ax1, v1, color="C0", lw=0.8, alpha=0.6, zorder=1)
    line(ax1, v2, color="C1", lw=0.8, alpha=0.6, zorder=1)
    arrow(ax1, 5 * v1, color="C0", lw=1.6, alpha=0.6, label=r"$A\hat v_1 = 5\hat v_1$", label_offset=(6, -4), fontsize=11)
    arrow(ax1, v1, color="C0", lw=2.6, label=r"$\hat v_1$", label_offset=(-19, -1), fontsize=11)
    arrow(ax1, 2 * v2, color="C1", lw=1.6, alpha=0.6, label=r"$A\hat v_2 = 2\hat v_2$", label_offset=(6, -6), fontsize=11)
    arrow(ax1, v2, color="C1", lw=2.6, label=r"$\hat v_2$", label_offset=(-16, -8), fontsize=11)
    arrow(ax1, Ae1, color="C2", lw=1.6, alpha=0.7)
    arrow(ax1, e1, color="C2", lw=2.6, label="$e_1$", label_offset=(3, -14), fontsize=11)
    label(ax1, Ae1, "$e_1 = (1, 0)$ is turned:\n$Ae_1 = (4, 2)$ at $26.6^\\circ$", color="C2", offset=(10, -20),
          fontsize=9.5, ha="center", va="top")
    arc = np.radians(np.linspace(45, deg(-v2), 60))       # from v1 to the upper half of the v2 line
    ax1.plot(1.9 * np.cos(arc), 1.9 * np.sin(arc), color="k", lw=0.9)
    label(ax1, (0.45, 2.25), f"${angle:.1f}^\\circ$ between\nthe eigen-lines,\nnot $90^\\circ$", offset=(0, 0), fontsize=9.5,
          ha="center", va="bottom")
    axes_2d(ax1, "(a) $A =$[[4, 1], [2, 3]]: two directions are\nonly stretched, every other one turns",
            (-3.2, 6.6), (-3.1, 4.7))
    ax1.grid(False)

    # (b) rotation by 90 degrees: three unit vectors (grey) and their images (red)
    ax2.plot(*CIRCLE, color=GREY, lw=1, zorder=0)
    for d in (0, 60, 120):
        u = np.array([np.cos(np.radians(d)), np.sin(np.radians(d))])
        arrow(ax2, u, color=GREY, lw=1.8)
        arrow(ax2, R @ u, color="C3", lw=1.8)
    t = np.radians(np.linspace(0, 90, 60)); r = 1.22
    ax2.plot(r * np.cos(t[:-3]), r * np.sin(t[:-3]), color="C3", lw=1, ls="--")
    ax2.annotate("", xy=(r * np.cos(t[-1]), r * np.sin(t[-1])), xytext=(r * np.cos(t[-4]), r * np.sin(t[-4])),
                 arrowprops=dict(arrowstyle="-|>", color="C3", lw=1, mutation_scale=10))
    label(ax2, (r * np.cos(t[30]), r * np.sin(t[30])), "$+90^\\circ$", color="C3", offset=(3, 3), fontsize=10)
    label(ax2, e1, "$e_1$", color=GREY, offset=(3, -13), fontsize=11)
    label(ax2, R @ e1, "$Re_1 = (0, 1)$", color="C3", offset=(-5, 3), fontsize=10, ha="right")
    ax2.text(0.05, -1.38, "$\\lambda = \\pm i$: no real direction\nkeeps its line", ha="center", va="center", fontsize=9.5)
    axes_2d(ax2, "(b) Rotation $R$ by $90^\\circ$: every\ndirection turns, no real eigenvector", (-1.6, 1.6), (-1.8, 1.6))

    # (c) shear: unit square -> parallelogram; e1 fixed; (0, 1) and (0, -1) slide sideways
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)
    par = sq @ Sh.T
    ax3.plot(sq[:, 0], sq[:, 1], color=GREY, ls="--", lw=1)
    ax3.fill(par[:, 0], par[:, 1], color=GREY, alpha=0.12, lw=0)
    ax3.plot(par[:, 0], par[:, 1], color=GREY, lw=1)
    ax3.axhline(0, color="C0", lw=0.9, alpha=0.8, zorder=1)
    label(ax3, (-1.6, 0), "invariant\nline", color="C0", offset=(2, 3), fontsize=9)
    for x in (e2, -e2):
        y = Sh @ x
        arrow(ax3, x, color="C3", lw=1.2, alpha=0.45)
        ax3.plot([x[0], y[0]], [x[1], y[1]], color="C3", ls=":", lw=1)
        arrow(ax3, y, color="C3", lw=2)
    label(ax3, (1, 1), "$(0, 1) \\to (1, 1)$", color="C3", offset=(-2, 8), fontsize=10)
    label(ax3, (-1, -1), "$(0, -1) \\to (-1, -1)$", color="C3", offset=(4, -9), fontsize=10, va="top")
    arrow(ax3, e1, color="C0", lw=2.6)
    label(ax3, (0.17, -0.1), "$e_1 \\to e_1$: the only\neigen-direction\n($\\lambda = 1$, double)", color="C0", offset=(0, 0),
          fontsize=9, va="top")
    axes_2d(ax3, "(c) Shear [[1, 1], [0, 1]]: a double\neigenvalue but one eigen-direction", (-1.7, 2.8), (-1.65, 1.6))
    tight_layout(fig)
    return fig


def fig_markov_forgetting():
    """Section 2: the two-brand chain walks along x1 + x2 = 1 towards pi; the offset shrinks by 0.4 a month."""
    T = np.array([[0.9, 0.5], [0.1, 0.5]])
    close(np.sort(np.linalg.eigvals(T)), [0.4, 1])
    pi = np.array([5 / 6, 1 / 6]); close(T @ pi, pi); close(pi, [0.833, 0.167], atol=1e-3)
    v2 = np.array([1., -1.]); close(T @ v2, 0.4 * v2)
    X = [np.array([0., 1.])]
    for _ in range(12):
        X.append(T @ X[-1])
    X = np.array(X)
    close(X[1:4], [[0.5, 0.5], [0.7, 0.3], [0.78, 0.22]])
    close(X[3], pi - 5 / 6 * 0.4 ** 3 * v2)
    k = np.arange(13)
    dist = np.linalg.norm(X - pi, axis=1); close(dist, 5 / 6 * np.sqrt(2) * 0.4 ** k)
    close(dist[0], 1.1785, atol=1e-4); close(0.4 ** 10, 1.05e-4, atol=1e-6); close(dist[10], 1.2e-4, atol=1e-5)
    k85 = next(j for j in range(200) if 0.85 ** j < 1e-4); assert k85 == 57
    k04 = next(j for j in range(200) if 0.4 ** j < 1e-4); assert k04 == 11
    close(0.85 ** 57, 9.5e-5, atol=1e-6); close(dist[0], 1.18, atol=5e-3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) the share vectors on the segment x1 + x2 = 1
    ax1.plot([1, 0], [0, 1], color=GREY, lw=7, alpha=0.22, solid_capstyle="butt", zorder=0)   # the title names it
    line(ax1, pi, span=(0, 1.05), color="C3", ls="--", lw=1, zorder=1)
    ax1.text(0.1, 0.035, "eigenvalue-1 direction", rotation=deg(pi), rotation_mode="anchor", color="C3",
             fontsize=9, va="bottom")
    for j in range(3):
        arrow(ax1, X[j + 1] - X[j], start=X[j], color="C0", lw=1.4, alpha=0.7)
    ax1.plot(X[:4, 0], X[:4, 1], "o", color="C0", ms=5, zorder=4)
    ax1.plot(*pi, "*", color="C3", ms=14, zorder=5)
    label(ax1, X[0], "$x_0 = (0, 1)$", color="C0", offset=(7, 0), fontsize=10, va="center")
    label(ax1, X[1], "$x_1 = (0.5, 0.5)$", color="C0", offset=(7, 2), fontsize=10)
    label(ax1, X[2], "$x_2 = (0.7, 0.3)$", color="C0", offset=(7, 2), fontsize=10)
    label(ax1, X[3], "$x_3 = (0.78, 0.22)$", color="C0", offset=(7, 3), fontsize=10)
    label(ax1, pi, "$\\pi = (5/6, 1/6)$\n$T\\pi = \\pi$", color="C3", offset=(38, -10), fontsize=10, va="top")
    off = np.array([0.075, 0.075])
    ax1.annotate("", xy=X[1] + off, xytext=X[0] + off + 0.13 * np.array([1, -1]),   # starts below the x0 label
                 arrowprops=dict(arrowstyle="<|-|>", color="C1", lw=1.2, mutation_scale=10))
    label(ax1, (X[0] + X[1]) / 2 + off, "offset from $\\pi$ is along $v_2 = (1, -1)$\nand shrinks $\\times 0.4$ per month",
          color="C1", offset=(8, 2), fontsize=9)
    axes_2d(ax1, "(a) Market shares move along $x_1 + x_2 = 1$ towards $\\pi$", (-0.05, 1.4), (-0.08, 1.12),
            "share of brand A", "share of brand B")

    # (b) distance to pi on a log axis
    kk = np.linspace(0, 12, 200)
    ax2.semilogy(kk, dist[0] * 0.4 ** kk, color="C0", ls="--", lw=1.4,
                 label="two-brand chain: $1.18 \\cdot 0.4^k$")
    ax2.semilogy(k[:11], dist[:11], "o", color="C0", ms=5, label="$\\|x_k - \\pi\\|$ from direct multiplication")
    ax2.semilogy(kk, dist[0] * 0.85 ** kk, color="C1", ls="--", lw=1.4,
                 label="Google-matrix rate: $1.18 \\cdot 0.85^k$ (a different, much\nlarger chain; $0.85^k$ drops below $10^{-4}$ after 57 steps)")
    label(ax2, (10, dist[10]), "$0.4^{10} \\approx 10^{-4}$", color="C0", offset=(8, 2), fontsize=10)
    ax2.set_xlim(-0.4, 13.2); ax2.set_ylim(1e-6, 2); ax2.set_xticks(range(0, 13, 2))   # room for the legend below the line
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    ax2.set_title("(b) Distance to $\\pi$ shrinks by $\\lambda_2 = 0.4$ every month")
    ax2.set_xlabel("month $k$"); ax2.set_ylabel("$\\|x_k - \\pi\\|_2$ (log scale)")
    tight_layout(fig)
    return fig


def fig_quadratic_form():
    """Section 3: S = [[2,1],[1,2]] as a stretch along q1, q2; the Rayleigh quotient; the level set x'Sx = 1."""
    S = np.array([[2., 1.], [1., 2.]])
    close(np.linalg.eigvalsh(S), [1, 3])
    q1, q2 = unit([1, 1]), unit([1, -1])
    close(S @ q1, 3 * q1); close(S @ q2, q2); close(q1 @ q2, 0)
    close(3 * np.outer(q1, q1) + np.outer(q2, q2), S)
    close(np.linalg.svd(S, compute_uv=False), [3, 1])          # image-ellipse semi-axes
    th = np.linspace(-90, 180, 400)
    xs = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
    ray = np.einsum("ij,ij->j", xs, S @ xs); close(ray, 2 + np.sin(2 * np.radians(th)))
    close(np.array([1, 0]) @ S @ np.array([1, 0]), 2)
    close(ray.max(), 3); close(ray.min(), 1); close(th[ray.argmax()], 45, atol=0.5)
    Q = np.column_stack([q1, q2])
    level = Q @ (np.diag([1 / np.sqrt(3), 1]) @ CIRCLE)         # the ellipse x'Sx = 1
    close(np.einsum("ij,ij->j", level, S @ level), 1)
    a1 = q1 / np.sqrt(3); close(np.linalg.norm(a1), 0.577, atol=1e-3)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1.1, 1.25, 1]})

    # (a) image of the unit circle: semi-axes 3 along q1, 1 along q2
    ax1.plot(*CIRCLE, color=GREY, lw=1, zorder=0)
    label(ax1, (-0.71, -0.71), "unit\ncircle", color=GREY, offset=(-3, -3), fontsize=9, ha="right", va="top")
    img = S @ CIRCLE
    ax1.plot(img[0], img[1], color="C0", lw=1.6)
    arrow(ax1, 3 * q1, color="C0", lw=1.6, alpha=0.6, label="$Sq_1 = 3q_1$", label_offset=(-4, 10), fontsize=11)
    ax1.texts[-1].set_ha("right")
    arrow(ax1, q1, color="C0", lw=2.4, label="$q_1$", label_offset=(-20, -6), fontsize=11)
    arrow(ax1, q2, color="C1", lw=2.4, label="$Sq_2 = 1q_2$\n(unchanged)", label_offset=(3, -10), fontsize=10)
    ax1.texts[-1].set_va("top")
    right_angle(ax1, (0, 0), q1, q2, size=0.22)
    axes_2d(ax1, "(a) Image of the unit circle under $S$:\n$\\times 3$ along $q_1$, $\\times 1$ along $q_2$",
            (-3.3, 3.5), (-3.3, 3.3))

    # (b) the Rayleigh quotient around the circle
    ax2.plot(th, ray, color="C0", lw=2)
    ax2.axhline(3, color="C0", ls="--", lw=1); ax2.axhline(1, color="C1", ls="--", lw=1)
    label(ax2, (-85, 3), "$\\lambda_{\\max} = 3$", color="C0", offset=(0, 3), fontsize=10)
    label(ax2, (45, 1), "$\\lambda_{\\min} = 1$", color="C1", offset=(0, -4), fontsize=10, ha="center", va="top")
    ax2.plot(45, 3, "o", color="C0", ms=7); ax2.plot([135, -45], [1, 1], "o", color="C1", ms=7)
    ax2.plot(0, 2, "o", color="C2", ms=7)
    # q2 = (1, -1)/sqrt2 points at -45 degrees, so the minimum at 135 degrees is -q2
    close(deg(q2), -45); close(deg(-q2), 135)
    label(ax2, (45, 3), "$q_1$ at $45^\\circ$", color="C0", offset=(6, 2), fontsize=10)
    label(ax2, (135, 1), "$-q_2$ at $135^\\circ$", color="C1", offset=(0, -9), fontsize=10, ha="center", va="top")
    label(ax2, (-45, 1), "$q_2$", color="C1", offset=(0, 8), fontsize=10, ha="center")
    label(ax2, (0, 2), "$x = (1, 0)$:\n$x^{\\!\\!\\top}\\!S x = 2$", color="C2", offset=(6, -6), fontsize=9.5, va="top")
    ax2.set_xlim(-90, 180); ax2.set_ylim(0, 3.6); ax2.set_xticks([-90, -45, 0, 45, 90, 135, 180])
    ax2.set_xticklabels([f"${t}^\\circ$" for t in [-90, -45, 0, 45, 90, 135, 180]])
    ax2.set_title("(b) Rayleigh quotient on the unit circle:\n$x^{\\!\\!\\top}\\!S x = 2 + \\sin 2\\theta$")
    ax2.set_xlabel("angle $\\theta$ of the unit vector $x$"); ax2.set_ylabel("$x^{\\!\\!\\top}\\!S x$")

    # (c) the level set x'Sx = 1: semi-axes 1/sqrt(3) along q1, 1 along q2
    ax3.plot(*CIRCLE, color=GREY, lw=1, ls=":", zorder=0)
    label(ax3, (0, 1), "unit circle", color=GREY, offset=(0, 3), fontsize=9, ha="center")
    ax3.plot(level[0], level[1], color="C0", lw=1.6)
    arrow(ax3, a1, color="C0", lw=2.4, label="$1/\\sqrt{3} \\approx 0.577$\nalong $q_1$", label_offset=(22, 12), fontsize=10)
    arrow(ax3, q2, color="C1", lw=2.4, label="$1$ along $q_2$", label_offset=(2, -10), fontsize=10)
    ax3.texts[-1].set_va("top")
    axes_2d(ax3, "(c) Level set $x^{\\!\\!\\top}\\!S x = 1$: tightest\nwhere $S$ is stiffest", (-1.3, 2.35), (-1.3, 1.45))
    tight_layout(fig)
    return fig


def fig_svd_rotate_stretch():
    """Section 4: A = [[3,0],[4,5]] on the unit circle in three stages: V', then Sigma, then U."""
    A = np.array([[3., 0.], [4., 5.]])
    close(A.T @ A, [[25, 20], [20, 25]]); close(np.linalg.eigvalsh(A.T @ A), [5, 45])
    s1, s2 = np.sqrt(45), np.sqrt(5)
    close([s1, s2], [6.708, 2.236], atol=1e-3); close(np.linalg.svd(A, compute_uv=False), [s1, s2])
    v1, v2 = unit([1, 1]), unit([1, -1])                   # the guide's signs (det V = -1: a reflection)
    close(A @ v1, np.array([3, 9]) / np.sqrt(2)); close(A @ v2, np.array([3, -1]) / np.sqrt(2))
    u1, u2 = A @ v1 / s1, A @ v2 / s2
    close(u1, unit([1, 3])); close(u2, unit([3, -1])); close(u1 @ u2, 0)
    close(deg(u1), 71.57, atol=0.01); close(deg(v1), 45)
    Vt, Sig, U = np.array([v1, v2]), np.diag([s1, s2]), np.column_stack([u1, u2])
    close(U @ Sig @ Vt, A); close(s1 * s2, 15); close(np.linalg.det(A), 15)
    close(np.sort(np.linalg.eigvals(A).real), [3, 5])      # eigenvalues are not the singular values
    e2 = np.array([0., 1.]); close(A @ e2, [0, 5])         # eigenvector of A for lambda = 5
    mid = Sig @ Vt @ e2; close(mid, [4.743, -1.581], atol=1e-3)
    stage_b, stage_c = Sig @ Vt @ CIRCLE, A @ CIRCLE
    close(stage_b, np.array([s1 * np.cos(TH - np.pi / 4), -s2 * np.sin(TH - np.pi / 4)]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.7))

    # (a) input: unit circle and right singular vectors
    ax1.plot(*CIRCLE, color=GREY, lw=1.2, zorder=0)
    arrow(ax1, v1, color="C0", lw=2.4, label="$v_1$ at $45^\\circ$", label_offset=(4, 2), fontsize=11)
    arrow(ax1, v2, color="C1", lw=2.4, label="$v_2$ at $-45^\\circ$", label_offset=(7, -11), fontsize=11)
    ax1.plot(*e2, "o", color="C2", ms=7, zorder=4)
    label(ax1, e2, "$(0, 1)$: eigenvector\nof $A$, $\\lambda = 5$", color="C2", offset=(8, 0), fontsize=9.5)
    right_angle(ax1, (0, 0), v1, v2, size=0.15)
    axes_2d(ax1, "(a) Input: the unit circle and the\nright singular vectors $v_1$, $v_2$", (-1.4, 2.1), (-1.6, 1.6))

    # (b) after V' and Sigma: axis-aligned ellipse
    ax2.plot(*CIRCLE, color=GREY, lw=1, ls="--", zorder=0)
    ax2.plot(stage_b[0], stage_b[1], color="C0", lw=1.6)
    arrow(ax2, (s1, 0), color="C0", lw=2.4)
    label(ax2, (3.6, 0), "$\\sigma_1 = 6.71$", color="C0", offset=(0, 5), fontsize=10, ha="center")
    arrow(ax2, (0, s2), color="C1", lw=2.4, label="$\\sigma_2 = 2.24$", label_offset=(5, 6), fontsize=10)
    ax2.plot(*mid, "o", color="C2", ms=7, zorder=4)
    label(ax2, mid, "$\\Sigma V^\\top (0, 1)$\n$= (4.74, -1.58)$", color="C2", offset=(-4, -6), fontsize=9, ha="center", va="top")
    label(ax2, (-7.2, -6.9), "$V^\\top$ sends $v_1 \\to e_1$, $v_2 \\to e_2$;\n$\\Sigma$ stretches the axes", offset=(0, 0),
          fontsize=9, va="bottom")
    axes_2d(ax2, "(b) $\\Sigma V^\\top$ applied to the circle: $v_i \\to e_i$,\nthen stretch by $\\sigma_1$, $\\sigma_2$ (dashed: scale)",
            (-7.5, 7.5), (-7.5, 8.3))

    # (c) after U: the ellipse A maps the circle to
    ax3.plot(stage_c[0], stage_c[1], color="C0", lw=1.6)
    arrow(ax3, s1 * u1, color="C0", lw=2.4, label="$\\sigma_1 u_1 = Av_1$", label_offset=(-6, 4), fontsize=10)
    ax3.texts[-1].set_ha("right")
    arrow(ax3, s2 * u2, color="C1", lw=2.4, label="$\\sigma_2 u_2 = Av_2$", label_offset=(3, -12), fontsize=10)
    ax3.plot([0, 0], [0, 5], color=GREY, ls=":", lw=1.2, zorder=1)
    ax3.plot(*(A @ e2), "o", color="C2", ms=7, zorder=4)
    label(ax3, A @ e2, "$A(0, 1) = (0, 5)$:\non the ellipse,\nnot on an axis", color="C2", offset=(-12, 0), fontsize=9,
          ha="right", va="center")
    axes_2d(ax3, "(c) then $U$: $e_i \\to u_i$ ($u_1$ at $71.6^\\circ$);\nthe ellipse $A$ maps the circle to",
            (-7.5, 7.5), (-7.5, 8.3))
    tight_layout(fig)
    return fig


def fig_low_rank_projection():
    """Section 5: A_1 drops each column of A onto the line along u_1; the truncated SVD of a 512 x 512 image at rank 20."""
    A = np.array([[3., 0.], [4., 5.]])
    s1, s2 = np.sqrt(45), np.sqrt(5)
    u1, v1 = unit([1, 3]), unit([1, 1])
    A1 = s1 * np.outer(u1, v1); close(A1, [[1.5, 1.5], [4.5, 4.5]])
    close(np.outer(u1, u1) @ A, A1)                       # A_1 = projection of the columns onto u_1
    close(np.linalg.norm(A - A1), s2); close(s2, 2.236, atol=1e-3); close(45 / 50, 0.9)
    cols = A.T                                            # (3, 4) and (0, 5)
    proj = np.array([(c @ u1) * u1 for c in cols]); close(proj, [[1.5, 4.5], [1.5, 4.5]])
    res = cols - proj; close(res, [[1.5, -0.5], [-1.5, 0.5]]); close(np.sum(res ** 2, axis=1), [2.5, 2.5])
    close(np.sum(res ** 2), 5)

    def error_onto(d):
        d = unit(d)
        return sum(np.sum((c - (c @ d) * d) ** 2) for c in cols)
    close(error_onto([0, 1]), 9); close(error_onto([1, 1]), 13)
    stored = 20 * (512 + 512 + 1); assert stored == 20_500 and 512 * 512 == 262_144
    close(stored / 262_144, 0.078, atol=1e-3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})

    # (a) both columns dropped perpendicularly onto the line along u1
    line(ax1, u1, span=(-0.9, 5.25), color="C0", lw=1, zorder=1)
    label(ax1, 5.25 * u1, "line along $u_1$", color="C0", offset=(5, 0), fontsize=10, va="center")
    for c, col, name, off, ha in [(cols[0], "C2", "$(3, 4)$: column 1 of $A$", (0, -10), "center"),
                                 (cols[1], "C3", "$(0, 5)$: column 2 of $A$", (-8, 8), "left")]:
        p = (c @ u1) * u1
        ax1.plot([c[0], p[0]], [c[1], p[1]], color=col, ls="--", lw=1.4)
        right_angle(ax1, p, c - p, u1 if c[0] > p[0] else -u1, size=0.22, color=col)
        ax1.plot(*c, "o", color=col, ms=7, zorder=4)
        label(ax1, c, name, color=col, offset=off, fontsize=10, ha=ha, va="top" if off[1] < 0 else "bottom")
    ax1.plot(*proj[0], "o", color="C0", ms=7, zorder=5)
    label(ax1, proj[0], "$(1.5, 4.5)$: both\ncolumns of $A_1$", color="C0", offset=(-16, -10), fontsize=10, ha="right", va="top")
    label(ax1, (5.6, -0.65), "$\\|A - A_1\\|_F^2 = 2.5 + 2.5 = 5 = \\sigma_2^2$\nonto the $x_2$-axis: 9; onto $(1, 1)$: 13",
          offset=(0, 0), fontsize=9, ha="right", va="bottom")
    axes_2d(ax1, "(a) Rank one: each column of $A$ dropped\nperpendicularly onto the line along $u_1$", (-2.6, 5.7), (-0.9, 6.4),
            "first entry $x_1$", "second entry $x_2$")

    # (b) block diagram of the truncated SVD, one data unit = one matrix entry
    def block(x0, w, h, color, name, count, count_y):
        ax2.add_patch(Rectangle((x0, 0), w, h, facecolor="none" if color == GREY else color, edgecolor=color,
                                lw=1.2, alpha=1 if color == GREY else 0.7))
        ax2.text(x0 + w / 2, -22, name, ha="center", va="bottom", fontsize=11, color=color)
        ax2.text(x0 + w / 2, count_y, count, ha="center", va="top", fontsize=9)
    block(0, 512, 512, GREY, "$A$", "$512 \\times 512 = 262{,}144$", 530)
    block(575, 20, 512, "C0", "$U_k$", "$512 \\times 20 = 10{,}240$", 530)
    block(690, 20, 20, "C1", "$\\Sigma_k$", "$20 \\times 20$\n(diagonal)", 38)
    block(780, 512, 20, "C2", "$V_k^\\top$", "$20 \\times 512 = 10{,}240$ (20 rows)", 38)
    ax2.text(543, 256, "$\\approx$", ha="center", va="center", fontsize=16)
    ax2.text(1036, 330, "$A_k = U_k \\Sigma_k V_k^\\top$ stores\n$10{,}240 + 20 + 10{,}240 = 20{,}500$\nnumbers: 7.8 % of $262{,}144$",
             ha="center", va="center", fontsize=10)
    ax2.set_xlim(-30, 1320); ax2.set_ylim(600, -85); ax2.set_aspect("equal")
    ax2.set_xticks([0, 512]); ax2.set_yticks([0, 512]); ax2.grid(False)
    ax2.set_title("(b) Truncated SVD of a $512 \\times 512$ image at rank $k = 20$,\ndrawn to scale: 20,500 numbers instead of 262,144")
    ax2.set_xlabel("column index"); ax2.set_ylabel("row index")
    tight_layout(fig)
    return fig


def fig_pca_centring_scaling():
    """Section 6: PCA of the five points; without centring and with x2 in finer units the answer changes."""
    X = np.array([[3, 1], [4, 3], [5, 2], [6, 4], [7, 5]], float)
    mu = X.mean(axis=0); close(mu, [5, 3])
    Xc = X - mu; close(Xc, [[-2, -2], [-1, 0], [0, -1], [1, 1], [2, 2]])
    C = Xc.T @ Xc / 4; close(C, [[2.5, 2.25], [2.25, 2.5]])
    close(np.linalg.eigvalsh(C), [0.25, 4.75])
    w1, w2 = unit([1, 1]), unit([1, -1]); close(C @ w1, 4.75 * w1); close(C @ w2, 0.25 * w2)
    close(4.75 / 5, 0.95)
    z = Xc @ w1; close(z, [-2.83, -0.71, -0.71, 1.41, 2.83], atol=5e-3); close(z.var(ddof=1), 4.75)
    rec = mu + np.outer(z, w1); close(rec, [[3, 1], [4.5, 2.5], [4.5, 2.5], [6, 4], [7, 5]])
    close(X[1] - rec[1], [-0.5, 0.5]); close(X[1] - rec[1], -0.707 * w2, atol=1e-3)
    close(np.sum((X - rec) ** 2), 1.0); close(np.linalg.svd(Xc, compute_uv=False) ** 2 / 4, [4.75, 0.25])
    # pitfall 3: no centring
    G = X.T @ X; close(G, [[135, 84], [84, 55]])
    gl, GV = np.linalg.eigh(G)
    raw = GV[:, -1] * np.sign(GV[0, -1]); close(raw, [0.8456, 0.5339], atol=1e-4)
    close(deg(raw), 32.27, atol=0.01); close(np.sqrt(gl[::-1]), [13.71, 1.40], atol=0.01)
    close(gl[-1] / gl.sum(), 0.9897, atol=1e-4); close(deg(mu), 30.96, atol=0.01)
    # pitfall 4: x2 in units ten times finer
    Xs = X * [1, 10]; mus = Xs.mean(axis=0); close(mus, [5, 30])
    Cs = np.cov(Xs.T); close(Cs, [[2.5, 22.5], [22.5, 250]])
    sl, SV = np.linalg.eigh(Cs); close(sl, [0.47, 252.03], atol=0.01)
    ws = SV[:, -1] * np.sign(SV[0, -1]); close(ws, [0.0898, 0.996], atol=5e-4)
    close(deg(ws), 84.85, atol=0.01); close(sl[-1] / sl.sum(), 0.9981, atol=1e-4)
    close(np.corrcoef(X.T), [[1, 0.9], [0.9, 1]]); close(np.linalg.eigvalsh([[1, 0.9], [0.9, 1]]), [0.1, 1.9])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.05, 1.2, 0.95]})

    # (a) the correct PCA: line through the mean along w1, reconstructions, residuals
    line(ax1, w1, through=mu, span=(-3.6, 3.6), color="C0", lw=1.2, zorder=1)
    label(ax1, mu - 3.4 * w1, "$w_1$ line: $\\lambda_1 = 4.75$\n(95 % of the variance)", color="C0", offset=(8, 2),
          fontsize=9.5, va="top")
    for i in (1, 2):   # solid: only about 7 pt of each residual shows between the two markers
        ax1.plot([X[i, 0], rec[i, 0]], [X[i, 1], rec[i, 1]], color="C3", ls="-", lw=1.6, zorder=2)
    ax1.plot(X[:, 0], X[:, 1], "o", color="C0", ms=7, zorder=4)
    ax1.plot(*rec[1], "o", ms=9, mfc="white", mec="C0", mew=1.3, zorder=3)   # (4.5, 2.5): shared by both off-line points
    ax1.plot(*mu, "D", color="C3", ms=6, zorder=5)
    label(ax1, mu, "$\\mu = (5, 3)$", color="C3", offset=(9, 0), fontsize=10, va="center")
    label(ax1, X[1], "residuals\n$\\mp(0.5, -0.5)$\nalong $w_2$", color="C3", offset=(-5, 6), fontsize=9, ha="right", va="bottom")
    arrow(ax1, w2, start=mu, color="C1", lw=2, label="$w_2$: $\\lambda_2 = 0.25$", label_offset=(-3, -8), fontsize=10)
    axes_2d(ax1, "(a) PCA: the line through $\\mu$ that\nkeeps the most variance", (0.3, 8.5), (-0.5, 6.5))

    # (b) pitfall 3: the top singular vector of the raw X points at the mean
    ax2.plot(X[:, 0], X[:, 1], "o", color="C0", ms=7, zorder=4)
    line(ax2, raw, span=(0, 10.2), color="C1", lw=1.6, zorder=2)
    arrow(ax2, mu, color=GREY, lw=1.4)
    ax2.text(0.15, 0.37, "mean direction $31.0^\\circ$", rotation=deg(raw), rotation_mode="anchor",
             color=GREY, fontsize=9, va="bottom")
    line(ax2, w1, through=mu, span=(-2.5, 2.5), color="C0", lw=1.2, ls="--", zorder=1)
    label(ax2, mu + 2.5 * w1, "true PC1 from (a):\n$45^\\circ$ through $\\mu$", color="C0", offset=(-4, 4), fontsize=9.5, ha="right")
    ax2.plot(*mu, "D", color="C3", ms=6, zorder=5)
    label(ax2, (8.45, -1.22), "top right singular vector of raw $X$:\n$32.3^\\circ$ through the origin; it \"explains\"\n99.0 % of $\\sum x^2$, not of the variance",
          color="C1", offset=(0, 0), fontsize=9, ha="right", va="bottom")
    axes_2d(ax2, "(b) Pitfall 3, no centring: the top singular\nvector of $X$ points at the mean", (-0.5, 8.5), (-1.3, 6.5))

    # (c) pitfall 4: x2 in units ten times finer: a vertical streak, PC1 follows x2
    ax3.plot(Xs[:, 0], Xs[:, 1], "o", color="C0", ms=7, zorder=4)
    line(ax3, ws, through=mus, span=(-31, 29), color="C1", lw=1.6, zorder=2)
    ax3.plot(*mus, "D", color="C3", ms=5, zorder=3)
    label(ax3, mus + 18 * ws, "PC1 at $84.9^\\circ$,\nloadings $(0.09, 1.00)$:\nexplains 99.8 %", color="C1", offset=(8, 0),
          fontsize=9.5, va="center")
    label(ax3, mus, "$\\mu = (5, 30)$", color="C3", offset=(7, -4), fontsize=10, va="top")
    label(ax3, (13, -0.5), "standardise both\ncolumns to get (a)\nback: $45^\\circ$, 95 %",
          offset=(0, 0), fontsize=9, va="bottom")
    axes_2d(ax3, "(c) Pitfall 4, $x_2$ in ten-times-finer units:\nPC1 follows the big numbers", (-8, 46), (-2, 58))
    tight_layout(fig)
    return fig


def fig_pseudo_inverse():
    """Section 7: A = [[1,1],[1,1]]; least squares in the output plane, minimum norm in the input plane."""
    A = np.array([[1., 1.], [1., 1.]])
    U, s, Vt = np.linalg.svd(A); close(s, [2, 0])
    u1, v1 = unit([1, 1]), unit([1, 1]); close(np.abs(U[:, 0] @ u1), 1); close(np.abs(Vt[0] @ v1), 1)
    Ap = np.outer(v1, u1) / 2; close(Ap, np.full((2, 2), 0.25)); close(np.linalg.pinv(A), Ap)
    b_in = np.array([1., 1.]); xp_in = Ap @ b_in; close(xp_in, [0.5, 0.5]); close(A @ xp_in, b_in)
    close(np.linalg.norm(xp_in), 0.7071, atol=1e-4)
    for x, n in [((1, 0), 1), ((1.5, -0.5), 1.5811)]:
        close(A @ np.array(x, float), b_in); close(np.linalg.norm(x), n, atol=1e-4)
    b_out = np.array([1., 0.]); xp_out = Ap @ b_out; close(xp_out, [0.25, 0.25])
    fit = A @ xp_out; close(fit, [0.5, 0.5]); r = b_out - fit; close(r, [0.5, -0.5]); close(r @ u1, 0)
    close(np.linalg.norm(r), 0.7071, atol=1e-4)
    close(A @ unit([1, -1]), [0, 0])                       # null(A) is the line along (1, -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1.2]})

    # (a) output space: Ax+ is the foot of the perpendicular from b = (1, 0) onto col(A)
    line(ax1, (1, 1), span=(-0.6, 2.1), color="C0", lw=1.2, zorder=1)
    label(ax1, (1.3, 1.3), "col($A$) = span$\\{(1, 1)\\}$", color="C0", offset=(-6, 0), fontsize=10, ha="right", va="center")
    line(ax1, (1, -1), span=(-1, 1), color=GREY, lw=1, ls=":", zorder=1)
    label(ax1, (0.6, -0.6), "null($A^\\top$)", color=GREY, offset=(8, 0), fontsize=9.5, va="center")
    arrow(ax1, b_out, color="C3", lw=2.4, label="$b = (1, 0)$", label_offset=(2, -14), fontsize=11)
    arrow(ax1, fit, color="C0", lw=2.4, label="$Ax^+ = (0.5, 0.5)$", label_offset=(-8, 4), fontsize=11)
    ax1.texts[-1].set_ha("right")
    ax1.plot([fit[0], b_out[0]], [fit[1], b_out[1]], color="C3", ls="--", lw=1.4)
    right_angle(ax1, fit, r, -u1, size=0.1, color="C3")
    label(ax1, (fit + b_out) / 2, "residual\n$(0.5, -0.5)$", color="C3", offset=(6, 2), fontsize=10)
    ax1.plot(*b_in, "o", color="C2", ms=7, zorder=4)
    label(ax1, b_in, "$b = (1, 1)$ lies\nin col($A$):\nsolved exactly", color="C2", offset=(7, 2), fontsize=9.5, va="top")
    axes_2d(ax1, "(a) Output space: $Ax^+$ is the projection\nof $b$ onto col($A$)", (-0.45, 1.75), (-0.75, 1.5), "$b_1$", "$b_2$")

    # (b) input space: the solutions of Ax = (1, 1) form a line; x+ is its point nearest the origin
    line(ax2, (1, -1), through=(0.5, 0.5), span=(-1.6, 1.6), color="C2", lw=1.6, zorder=2)
    label(ax2, (-0.2, 1.68), "solutions of $Ax = (1, 1)$:\nthe line $x_1 + x_2 = 1$", color="C2", offset=(0, 0), fontsize=10, va="top")
    line(ax2, (1, -1), span=(-1.3, 0.7), color=GREY, lw=1, ls="--", zorder=1)   # stops above the (1.5, -0.5) label
    label(ax2, 1.3 * unit([-1, 1]), "null($A$)", color=GREY, offset=(4, 3), fontsize=9.5)
    line(ax2, (1, 1), span=(-0.9, 1.6), color=GREY, lw=1, ls=":", zorder=1)
    label(ax2, 1.6 * u1, "row($A$)", color=GREY, offset=(4, 0), fontsize=9.5, va="center")
    arrow(ax2, xp_in, color="C0", lw=2.6, label="$x^+ = (0.5, 0.5)$, norm $0.71$", label_offset=(22, 2), fontsize=11)
    right_angle(ax2, xp_in, -u1, unit([1, -1]), size=0.1, color="C0")
    for x, txt, off, ha, va in [((1, 0), "$(1, 0)$: norm $1$", (7, 1), "left", "bottom"),
                                ((1.5, -0.5), "$(1.5, -0.5)$: norm $1.58$", (-5, -7), "right", "top")]:
        ax2.plot(*x, "o", color="C2", ms=7, zorder=4)
        label(ax2, x, txt, color="C2", offset=off, fontsize=10, ha=ha, va=va)
    ax2.plot(*xp_out, "o", color="C3", ms=5, zorder=4)
    label(ax2, xp_out, "$x^+$ for\n$b = (1, 0)$", color="C3", offset=(3, -7), fontsize=9, va="top")
    # the Pythagoras argument (every other solution is x+ + z with Az = 0) is in the caption
    axes_2d(ax2, "(b) Input space: the solutions of $Ax = (1, 1)$\nform a line; $x^+$ is the shortest", (-1.0, 2.5), (-1.2, 1.75))
    tight_layout(fig)
    return fig


def fig_condition_number():
    """Section 8: the kappa = 19 Gram matrix squashes the circle; gradient descent zig-zags with error x 0.9 per step."""
    G = np.array([[1., 0.9], [0.9, 1.]])
    lam = np.linalg.eigvalsh(G); close(lam, [0.1, 1.9]); kappa = lam[1] / lam[0]; close(kappa, 19)
    close(np.linalg.cond(G), 19)
    q1, q2 = unit([1, 1]), unit([1, -1])
    close(G @ q1, 1.9 * q1); close(G @ q2, 0.1 * q2)
    close(G @ q1, [1.3435, 1.3435], atol=1e-4); close(G @ q2, [0.0707, -0.0707], atol=1e-4)
    close([1 / 0.1, 1 / 1.9], [10, 0.526], atol=1e-3)
    eta = 2 / (lam[1] + lam[0]); close(eta, 1); rate = (kappa - 1) / (kappa + 1); close(rate, 0.9)
    x = [np.array([1.5, -0.5])]
    for _ in range(12):
        x.append(x[-1] - eta * G @ x[-1])
    x = np.array(x)
    close(x[1:3], [[0.45, -1.35], [1.215, -0.405]]); close(x[12], [0.4236, -0.1412], atol=1e-4)
    close(np.linalg.norm(x, axis=1), np.linalg.norm(x[0]) * rate ** np.arange(13))
    close(np.log(1e-3) / np.log(0.9), 65.6, atol=0.1); assert 0.98 ** 345 < 1e-3 < 0.98 ** 340
    # the section's headline matrix, for the record (not drawn: its ellipse is a segment)
    B = np.array([[1., 1.], [1., 1.0001]]); close(np.linalg.svd(B, compute_uv=False), [2.00005, 0.00005], atol=1e-5)
    close(np.linalg.cond(B), 40002, atol=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1.05]})

    # (a) the image of the unit circle: a 1.9 x 0.1 ellipse
    ax1.plot(*CIRCLE, color=GREY, lw=1, zorder=0)
    label(ax1, (-0.72, 0.72), "unit circle", color=GREY, offset=(-2, 2), fontsize=9, ha="right")
    img = G @ CIRCLE
    ax1.fill(img[0], img[1], color="C0", alpha=0.15, lw=0)
    ax1.plot(img[0], img[1], color="C0", lw=1.6)
    arrow(ax1, G @ q1, color="C0", lw=1.6, alpha=0.6, label="$Gq_1 = 1.9\\,q_1$", label_offset=(0, 9), fontsize=11)
    ax1.texts[-1].set_ha("center")
    arrow(ax1, q1, color="C0", lw=2.4, label="$q_1$", label_offset=(16, -8), fontsize=11)
    arrow(ax1, q2, color="C1", lw=2.4, label="$q_2$", label_offset=(4, -4), fontsize=11)
    # G q2 = 0.1 q2 is shorter than an arrowhead: mark its tip with a dot and lead the label to it
    ax1.plot(*(G @ q2), "o", ms=5, mfc="white", mec="C1", mew=1.5, zorder=6)
    ax1.annotate("$Gq_2 = 0.1\\,q_2$\n(barely visible)", xy=G @ q2, xytext=(-18, 6), textcoords="offset points", ha="right",
                 color="C1", fontsize=10, arrowprops=dict(arrowstyle="-", color="C1", lw=0.8, shrinkA=0, shrinkB=3),
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))   # hides the circle under the text
    label(ax1, (-2.0, -2.0), "solving $Gx = b$ reverses this: errors in $b$ along\n$(1, -1)$ are amplified $\\times 10$, along $(1, 1)$\ndamped $\\times 0.53$; the ratio $19$ is $\\kappa$",
          offset=(0, 0), fontsize=9, va="bottom")
    axes_2d(ax1, "(a) $G =$[[1, 0.9], [0.9, 1]], $\\kappa = 19$: the unit\ncircle becomes a $1.9 \\times 0.1$ ellipse", (-2.1, 2.3), (-2.1, 2.1))

    # (b) gradient descent on the quadratic with the best fixed step
    xx, yy = np.meshgrid(np.linspace(-0.3, 1.9, 300), np.linspace(-1.7, 0.5, 300))
    f = 0.5 * (xx ** 2 + 2 * G[0, 1] * xx * yy + yy ** 2)
    ax2.contour(xx, yy, f, levels=[0.05, 0.2, 0.5, 1.0, 1.5], colors=GREY, linewidths=0.8, alpha=0.8)
    line(ax2, q2, span=(-2.8, 2.15), color=GREY, lw=1, ls="--", zorder=1)   # ends above the bottom note
    label(ax2, (-0.15, 0.15), "valley floor along $(1, -1)$: $\\lambda_{\\min} = 0.1$", color="0.35", offset=(6, 6), fontsize=9)
    ax2.texts[-1].set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))   # lift it off the contours
    ax2.plot(x[:, 0], x[:, 1], "-o", color="C3", lw=1.3, ms=4.5, zorder=4)
    ax2.plot(0, 0, "*", color="C0", ms=14, zorder=5)
    label(ax2, (0, 0), "minimum", color="C0", offset=(7, 2), fontsize=10)
    ax2.texts[-1].set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    label(ax2, x[0], "$x_0 = (1.5, -0.5)$", color="C3", offset=(5, -2), fontsize=10, va="top")
    label(ax2, x[1], "$x_1$", color="C3", offset=(-6, -4), fontsize=10, ha="right", va="top")
    label(ax2, x[12], "$x_{12}$", color="C3", offset=(-4, 4), fontsize=10, ha="right")
    label(ax2, (-0.25, -1.82), "$\\eta = 2/(\\lambda_{\\max} + \\lambda_{\\min}) = 1$, $\\|x_k\\| = 0.9^k\\|x_0\\|$: 66 steps\nfor three digits ($\\kappa = 100$: $0.98$ per step, about 345)",
          offset=(0, 0), fontsize=9, va="bottom")
    ax2.texts[-1].set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))   # contours end under the note
    axes_2d(ax2, "(b) Gradient descent on $\\frac{1}{2}x^{\\!\\!\\top}\\!G x$ with the best\nfixed step: the error shrinks by $0.9$ per step",
            (-0.3, 2.3), (-1.88, 0.5))
    tight_layout(fig)
    return fig


FIGURES = {
    "eigen-directions": fig_eigen_directions,
    "markov-forgetting": fig_markov_forgetting,
    "quadratic-form": fig_quadratic_form,
    "svd-rotate-stretch": fig_svd_rotate_stretch,
    "low-rank-projection": fig_low_rank_projection,
    "pca-centring-scaling": fig_pca_centring_scaling,
    "pseudo-inverse": fig_pseudo_inverse,
    "condition-number": fig_condition_number,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 03 concept-guide figures.")
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
