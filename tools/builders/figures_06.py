"""Figures for the Module 06 concept guide — Automatic Differentiation and Backpropagation.

Writes SVG files into part-2-calculus-and-optimization/06-autodiff-and-backpropagation/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, recomputed here (and asserted), so picture and text agree exactly.

    python tools/builders/figures_06.py                        # (re)write every figure
    python tools/builders/figures_06.py computational-graph    # only the named figure(s)
    python tools/builders/figures_06.py --preview DIR          # also write PNG previews to DIR
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402,F401

import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

FIG_DIR = ROOT / "part-2-calculus-and-optimization/06-autodiff-and-backpropagation/figures"

HIGHLIGHT = "#ffe8c2"   # the lesson notebook's operation-node tint, here for highlighted matrix entries
FWD, BWD = "C2", "C3"   # forward-pass arrows are green, backward-pass arrows and adjoints red


# ============================================================================
# Small helpers for the graph diagrams
# ============================================================================
def num(v):
    """A worked-example number as compact text: no '-0', no trailing zeros."""
    v = float(v)
    return f"{0.0 if v == 0 else v:g}"


def sci(v, digits=1):
    """Mathtext for a number in scientific notation, e.g. 6.7 x 10^-5."""
    e = int(np.floor(np.log10(abs(v))))
    return rf"{v / 10 ** e:.{digits}f} \times 10^{{{e}}}"


def rel(v):
    """A relative error as mathtext: scientific below 0.1, plain above."""
    return f"{v:.2g}" if v >= 0.1 else sci(v)


def diagram(ax, xlim, ylim, title, xlabel):
    """A diagram panel: equal aspect, no ticks, no frame, no grid; title and xlabel carry the words."""
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)


def node(ax, xy, text, color, size, fontsize=9.5, ls="-", text_color="k"):
    """A graph node: `text` centred in a rounded box of `size` = (width, height) with a coloured edge."""
    x, y = xy
    w, h = size
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", fc="white",
                                ec=color, lw=1.5, ls=ls, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=text_color, zorder=4)


def clip(p0, p1, size0, size1, pad=0.04):
    """The segment p0 -> p1 shortened to start and end on the (padded) box boundaries; also its direction."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; d /= np.linalg.norm(d)

    def t(size):
        hw, hh = size[0] / 2 + pad, size[1] / 2 + pad
        return hw / abs(d[0]) if abs(d[0]) * hh >= abs(d[1]) * hw else hh / abs(d[1])
    return p0 + t(size0) * d, p1 - t(size1) * d, d


def edge(ax, p0, p1, color, reverse=False, lw=1.5):
    """Arrow from p0 to p1; with `reverse` it runs from p1 back to p0, dashed (the backward pass)."""
    a, b = (p1, p0) if reverse else (p0, p1)
    ax.annotate("", xy=b, xytext=a, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls="--" if reverse else "-",
                                shrinkA=0, shrinkB=0, mutation_scale=12))


def edge_label(ax, p0, p1, d, text, color, frac=0.5, offset=0.2, fontsize=9):
    """`text` written along the edge p0 -> p1 at the fraction `frac` of its length, `offset` data units aside."""
    m = p0 + frac * (p1 - p0) + offset * np.array([-d[1], d[0]])
    ax.text(*m, text, color=color, fontsize=fontsize, ha="center", va="center",
            rotation=np.degrees(np.arctan2(d[1], d[0])), rotation_mode="anchor", zorder=4)


def draw_graph(ax, pos, size, edges, texts, colors, arrow_color, reverse, labels=None, label_color=None,
               frac=0.5, offsets=None, fontsize=9.5, label_size=9, dead=(), dead_color=GREY):
    """Nodes and arrows of one graph panel. `dead` nodes are drawn dotted grey, with grey edges.

    `frac` (a number or a per-edge dict) says where along an edge its label sits; `offsets` (per-edge dict)
    overrides the perpendicular offset of 0.2 data units, a negative value putting the label on the other side.
    """
    for a, c in edges:
        s, e, d = clip(pos[a], pos[c], size[a], size[c])
        col = dead_color if (a in dead or c in dead) else arrow_color
        edge(ax, s, e, col, reverse=reverse)
        if labels and labels.get((a, c)):
            f = frac.get((a, c), 0.5) if isinstance(frac, dict) else frac
            off = (offsets or {}).get((a, c), 0.2)
            edge_label(ax, s, e, d, labels[(a, c)], label_color or col, frac=f, offset=off, fontsize=label_size)
    for k, xy in pos.items():
        if k in dead:
            node(ax, xy, texts[k], dead_color, size[k], fontsize=fontsize, ls=":", text_color=dead_color)
        else:
            node(ax, xy, texts[k], colors[k], size[k], fontsize=fontsize)


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_computational_graph():
    """Section 2 (and 5, 7): the graph of L = (wx + b - y)^2; forward values, then adjoints flowing back."""
    w, x, b, y = 2.0, 3.0, 1.0, 5.0
    v1 = w * x; v2 = v1 + b; v3 = v2 - y; L = v3 ** 2                          # 6, 7, 2, 4
    loc = {("w", "v1"): x, ("x", "v1"): w, ("v1", "v2"): 1.0, ("b", "v2"): 1.0,   # local derivative on each edge
           ("v2", "v3"): 1.0, ("y", "v3"): -1.0, ("v3", "L"): 2 * v3}
    Lb = 1.0                                                                    # the seed
    v3b = Lb * loc[("v3", "L")]; v2b = v3b * loc[("v2", "v3")]; yb = v3b * loc[("y", "v3")]
    v1b = v2b * loc[("v1", "v2")]; bb = v2b * loc[("b", "v2")]
    wb = v1b * loc[("w", "v1")]; xb = v1b * loc[("x", "v1")]
    assert (v1, v2, v3, L) == (6, 7, 2, 4) and (v3b, v2b, yb, v1b, bb, wb, xb) == (4, 4, -4, 4, 4, 12, 8)
    assert wb == 2 * (w * x + b - y) * x and bb == 2 * (w * x + b - y)          # closed form, Section 5
    eta = 0.05                                                                  # the GD step of Section 5
    assert np.allclose([w - eta * wb, b - eta * bb, (w - eta * wb) * x + (b - eta * bb)], [1.4, 0.8, 5.0])
    assert np.isclose(eta, 1 / (2 * (x ** 2 + 1)))

    pos = {"w": (0, 3), "x": (0, 2), "b": (0, 1), "y": (0, 0), "v1": (2.8, 2.5), "v2": (5.6, 1.75),
           "v3": (8.4, 0.9), "L": (11.0, 0.9)}
    size = {k: (1.4, 0.7) if len(k) == 1 else (1.6, 0.7) for k in pos}
    colors = {k: GREY if len(k) == 1 else "C0" for k in pos}
    fwd = {"w": f"$w = {num(w)}$", "x": f"$x = {num(x)}$", "b": f"$b = {num(b)}$", "y": f"$y = {num(y)}$",
           "v1": f"$v_1 = wx = {num(v1)}$", "v2": f"$v_2 = v_1 + b = {num(v2)}$",
           "v3": f"$v_3 = v_2 - y = {num(v3)}$", "L": f"$L = v_3^2 = {num(L)}$"}
    bwd = {"w": rf"$\bar w = {num(wb)}$", "x": rf"$\bar x = {num(xb)}$", "b": rf"$\bar b = {num(bb)}$",
           "y": rf"$\bar y = {num(yb)}$", "v1": rf"$\bar v_1 = {num(v1b)}$", "v2": rf"$\bar v_2 = {num(v2b)}$",
           "v3": rf"$\bar v_3 = {num(v3b)}$", "L": rf"$\bar L = {num(Lb)}$"}
    along = {("w", "v1"): rf"$\times\, x = {num(x)}$", ("x", "v1"): rf"$\times\, w = {num(w)}$",
             ("v1", "v2"): r"$\times\, 1$", ("b", "v2"): r"$\times\, 1$", ("v2", "v3"): r"$\times\, 1$",
             ("y", "v3"): r"$\times\, (-1)$", ("v3", "L"): rf"$\times\, 2v_3 = {num(2 * v3)}$"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.3))
    draw_graph(ax1, pos, size, list(loc), fwd, colors, FWD, reverse=False)
    diagram(ax1, (-0.8, 11.8), (-0.6, 3.6),
            "(a) Forward pass: evaluate the nodes in topological order and keep every value",
            "four inputs, four operations, one output; values flow left to right")
    draw_graph(ax2, pos, size, list(loc), bwd, colors, BWD, reverse=True, labels=along)
    diagram(ax2, (-0.8, 11.8), (-0.6, 3.6),
            r"(b) Backward pass: seed $\bar L = 1$, multiply by the local derivative on each edge, "
            "add where a node is shared",
            rf"$\bar v_3 = 1 \cdot 2 v_3 = {num(v3b)}$,  $\bar v_2 = \bar b = {num(v2b)}$,  $\bar y = {num(yb)}$,  "
            rf"$\bar w = \bar v_1 \cdot x = {num(wb)}$,  $\bar x = \bar v_1 \cdot w = {num(xb)}$")
    fig.tight_layout()
    return fig


def fig_forward_vs_reverse():
    """Section 3: f = (x1 x2 + x1)^2 at (2, 3): two forward sweeps (one per input) against one reverse sweep."""
    x1, x2 = 2.0, 3.0
    v1 = x1 * x2; v2 = v1 + x1; v3 = v2 ** 2                                    # 6, 8, 64
    loc = {("x1", "v1"): x2, ("x2", "v1"): x1, ("v1", "v2"): 1.0, ("x1", "v2"): 1.0, ("v2", "v3"): 2 * v2}

    def tangents(x1d, x2d):
        v1d = loc[("x1", "v1")] * x1d + loc[("x2", "v1")] * x2d
        v2d = loc[("v1", "v2")] * v1d + loc[("x1", "v2")] * x1d
        return v1d, v2d, loc[("v2", "v3")] * v2d
    t1, t2 = tangents(1, 0), tangents(0, 1)
    v3b = 1.0; v2b = v3b * loc[("v2", "v3")]; v1b = v2b * loc[("v1", "v2")]
    x1b_a, x1b_b = v1b * loc[("x1", "v1")], v2b * loc[("x1", "v2")]            # x1 has two children
    x1b = x1b_a + x1b_b; x2b = v1b * loc[("x2", "v1")]
    assert (v1, v2, v3) == (6, 8, 64) and t1 == (3, 4, 64) and t2 == (2, 2, 32)
    assert (v2b, v1b, x1b_a, x1b_b, x1b, x2b) == (16, 16, 48, 16, 64, 32)
    assert x1b == 2 * x1 * (x2 + 1) ** 2 and x2b == 2 * x1 ** 2 * (x2 + 1)      # closed form
    assert (t1[2], t2[2]) == (x1b, x2b)                                         # two sweeps = one sweep

    pos = {"x1": (0, 1.6), "x2": (0, 0), "v1": (2.6, 0), "v2": (5.2, 0.8), "v3": (7.8, 0.8)}
    size = {k: (1.6, 0.76) for k in pos}
    colors = {k: GREY if k.startswith("x") else "C0" for k in pos}
    along = {("x1", "v1"): rf"$\times\, x_2 = {num(x2)}$", ("x2", "v1"): rf"$\times\, x_1 = {num(x1)}$",
             ("v1", "v2"): r"$\times\, 1$", ("x1", "v2"): r"$\times\, 1$",
             ("v2", "v3"): rf"$\times\, 2 v_2 = {num(2 * v2)}$"}
    panels = [
        (rf"(a) Forward mode, seed $\dot x_1 = 1$: tangents flow left to right, $\dot v_3 = {num(t1[2])}$",
         {"x1": f"$x_1 = {num(x1)}$\n$\\dot x_1 = 1$", "x2": f"$x_2 = {num(x2)}$\n$\\dot x_2 = 0$",
          "v1": rf"$v_1 = x_1 x_2 = {num(v1)}$" "\n" rf"$\dot v_1 = {num(x2)} \cdot 1 + {num(x1)} \cdot 0 = {num(t1[0])}$",
          "v2": rf"$v_2 = v_1 + x_1 = {num(v2)}$" "\n" rf"$\dot v_2 = {num(t1[0])} + 1 = {num(t1[1])}$",
          "v3": rf"$v_3 = v_2^2 = {num(v3)}$" "\n" rf"$\dot v_3 = {num(2 * v2)} \cdot {num(t1[1])} = {num(t1[2])}$"},
         FWD, False),
        (rf"(b) Forward mode again, seed $\dot x_2 = 1$: a second sweep for the second input, $\dot v_3 = {num(t2[2])}$",
         {"x1": f"$x_1 = {num(x1)}$\n$\\dot x_1 = 0$", "x2": f"$x_2 = {num(x2)}$\n$\\dot x_2 = 1$",
          "v1": rf"$v_1 = {num(v1)}$" "\n" rf"$\dot v_1 = {num(x2)} \cdot 0 + {num(x1)} \cdot 1 = {num(t2[0])}$",
          "v2": rf"$v_2 = {num(v2)}$" "\n" rf"$\dot v_2 = {num(t2[0])} + 0 = {num(t2[1])}$",
          "v3": rf"$v_3 = {num(v3)}$" "\n" rf"$\dot v_3 = {num(2 * v2)} \cdot {num(t2[1])} = {num(t2[2])}$"},
         FWD, False),
        (r"(c) Reverse mode, seed $\bar v_3 = 1$: one sweep, both partial derivatives",
         {"x1": rf"$x_1 = {num(x1)}$" "\n" rf"$\bar x_1 = {num(x1b_a)} + {num(x1b_b)} = {num(x1b)}$",
          "x2": rf"$x_2 = {num(x2)}$" "\n" rf"$\bar x_2 = {num(v1b)} \cdot {num(x1)} = {num(x2b)}$",
          "v1": rf"$v_1 = {num(v1)}$" "\n" rf"$\bar v_1 = {num(v2b)} \cdot 1 = {num(v1b)}$",
          "v2": rf"$v_2 = {num(v2)}$" "\n" rf"$\bar v_2 = 1 \cdot {num(2 * v2)} = {num(v2b)}$",
          "v3": rf"$v_3 = {num(v3)}$" "\n" r"$\bar v_3 = 1$"},
         BWD, True),
    ]
    notes = [r"$x_1$ feeds two nodes, so $\dot x_1$ is used twice",
             "the same sweep again, with the other seed",
             r"$x_1$ has two children, so $\bar x_1$ adds two contributions"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.6))
    for ax, (title, texts, col, rev), note in zip(axes, panels, notes):
        draw_graph(ax, pos, size, list(loc), texts, colors, col, reverse=rev, labels=along, label_color=GREY,
                   fontsize=8.5, label_size=8.5)
        diagram(ax, (-0.9, 8.7), (-0.5, 2.1), title,
                r"grey edge labels: the local derivatives $\partial\,\mathrm{child} / \partial\,\mathrm{parent}$; " + note)
    fig.tight_layout()
    return fig


def fig_jacobian_rows_columns():
    """Section 3 (Cost): a forward sweep fills one Jacobian column, a reverse sweep one row; gradient cost vs n."""
    m, n = 3, 8                      # schematic shape; the guide's own example is 3 x 1000 (self-assessment 3)
    col, row = 3, 2                  # the column one forward sweep fills, the row one reverse sweep fills
    n_gpt, n_bank = 124_439_808, 120                                            # Section 3; Case study 2
    rev_lo, rev_hi, rev_typ = 2, 4, 3                                           # reverse mode: "two to four", about three
    assert round(2 * n_gpt / 1e8, 1) == 2.5 and round(2 * n_gpt / rev_typ / 1e7) == 8     # 2.5e8 passes; ~1e8 times cheaper
    assert 2 * n_bank == 240 and round(2 * n_bank / rev_hi) == 60                          # 240 pricings vs 3-4: 60-fold

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    for i in range(m):
        for j in range(n):
            ax1.add_patch(Rectangle((j, m - 1 - i), 1, 1, fc="white", ec=GREY, lw=0.8))
    ax1.add_patch(Rectangle((col - 1, 0), 1, m, fc=FWD, ec="none", alpha=0.25))
    ax1.add_patch(Rectangle((0, m - row), n, 1, fc=BWD, ec="none", alpha=0.25))
    for j in range(n):
        ax1.text(j + 0.5, -0.1, f"$x_{j + 1}$", ha="center", va="top", fontsize=8.5, color=GREY)
    for i in range(m):
        ax1.text(-0.12, m - 0.5 - i, f"$f_{i + 1}$", ha="right", va="center", fontsize=8.5, color=GREY)
    ax1.annotate("", xy=(col - 0.5, m + 0.12), xytext=(col - 0.5, m + 1.0), zorder=3,
                 arrowprops=dict(arrowstyle="-|>", color=FWD, lw=1.6, mutation_scale=12))
    ax1.text(col - 0.5, m + 1.1, rf"forward sweep seeded at $x_{col}$" "\n" rf"fills one column $J e_{col}$",
             color=FWD, ha="center", va="bottom", fontsize=9)
    ax1.annotate("", xy=(n + 0.12, m - row + 0.5), xytext=(n + 1.2, m - row + 0.5), zorder=3,
                 arrowprops=dict(arrowstyle="-|>", color=BWD, lw=1.6, mutation_scale=12))
    ax1.text(n + 1.3, m - row + 0.5, rf"reverse sweep" "\n" rf"seeded at $f_{row}$ fills" "\n" rf"one row $e_{row}^\top J$",
             color=BWD, ha="left", va="center", fontsize=9)
    ax1.text(n / 2, -0.7, r"$n$ inputs $\rightarrow$ $n$ forward sweeps for all of $J$",
             ha="center", va="top", fontsize=9, color=FWD)
    ax1.text(n / 2, -1.3, r"$m$ outputs $\rightarrow$ $m$ reverse sweeps", ha="center", va="top", fontsize=9, color=BWD)
    ax1.text(n / 2, -2.0, "a training loss has $m = 1$:\nits single row is the whole gradient",
             ha="center", va="top", fontsize=9.5)
    diagram(ax1, (-0.8, 13.3), (-3.2, 5.5),
            rf"(a) The Jacobian $J$ of $f: \mathbb{{R}}^{n} \to \mathbb{{R}}^{m}$, filled column by column or row by row",
            "each sweep costs a small constant times one evaluation of $f$")

    ns = np.logspace(0, 9.3, 200)
    ax2.loglog(ns, 2 * ns, color="C1", lw=2, label="central differences, $2n$")
    ax2.loglog(ns, ns, color=FWD, lw=2, label="forward mode, $n$ sweeps")
    ax2.fill_between(ns, rev_lo, rev_hi, color=BWD, alpha=0.15, lw=0)
    ax2.loglog(ns, np.full_like(ns, float(rev_typ)), color=BWD, lw=2,
               label=f"reverse mode, {rev_lo}–{rev_hi} for any $n$")
    for nn, mk in [(n_gpt, "o"), (n_bank, "s")]:
        ax2.axvline(nn, color=GREY, ls=":", lw=1)
        ax2.plot([nn, nn], [2 * nn, rev_typ], mk, color="k", ms=5, zorder=4)
    label(ax2, (n_gpt, 2 * n_gpt), rf"GPT-2 small" "\n" rf"$n = {sci(n_gpt, 2)}$" "\n" rf"${sci(2 * n_gpt)}$ passes",
          offset=(-6, 4), ha="right", fontsize=9)
    label(ax2, (n_gpt, rev_typ), rf"about {rev_typ}", offset=(-6, 5), ha="right", fontsize=9)
    label(ax2, (n_bank, 2 * n_bank), f"Case study 2\n$n = {n_bank}$:\n{2 * n_bank} pricings",
          offset=(-6, 4), ha="right", fontsize=9)
    label(ax2, (n_bank, rev_typ), f"{rev_typ}–{rev_hi}", offset=(6, 5), fontsize=9)
    ax2.set_xlim(1, 2e9); ax2.set_ylim(1, 1e11)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax2.set_title("(b) Cost of one full gradient, in forward-pass units", fontsize=10)
    ax2.set_xlabel("number of inputs $n$"); ax2.set_ylabel("forward-pass equivalents")
    fig.tight_layout()
    return fig


def fig_backprop_2_2_1():
    """Section 4: the 2-2-1 ReLU network; weights and activations forwards, deltas and weight gradients backwards."""
    W1 = np.array([[1., -1.], [0., 1.]]); b1 = np.zeros(2); W2 = np.array([1., 0.5]); b2 = 0.0
    x = np.array([1., 2.]); y = 3.0
    z1 = W1 @ x + b1; a1 = np.maximum(z1, 0); yhat = W2 @ a1 + b2; L = 0.5 * (yhat - y) ** 2
    d2 = yhat - y; dW2 = d2 * a1; db2 = d2                                      # output layer
    back = W2 * d2; mask = (z1 > 0).astype(float); d1 = back * mask              # W2^T delta2, masked
    dW1 = np.outer(d1, x); db1 = d1
    assert np.allclose(z1, [-1, 2]) and np.allclose(a1, [0, 2]) and yhat == 1 and L == 2 and d2 == -2
    assert np.allclose(dW2, [0, -4]) and db2 == -2 and np.allclose(back, [-2, -1]) and np.allclose(mask, [0, 1])
    assert np.allclose(d1, [0, -1]) and np.allclose(dW1, [[0, 0], [-1, -2]]) and np.allclose(db1, [0, -1])

    pos = {"x1": (0, 1.7), "x2": (0, 0.3), "h1": (3.3, 1.7), "h2": (3.3, 0.3), "o": (6.6, 1.0), "L": (9.4, 1.0)}
    size = {k: (1.8, 0.84) for k in pos}
    colors = {k: GREY if k.startswith("x") else "C0" for k in pos}
    edges = [("x1", "h1"), ("x2", "h1"), ("x1", "h2"), ("x2", "h2"), ("h1", "o"), ("h2", "o"), ("o", "L")]
    fwd = {"x1": f"$x_1 = {num(x[0])}$", "x2": f"$x_2 = {num(x[1])}$",
           "h1": rf"unit 1: $z = {num(z1[0])}$" "\n" rf"$a = \mathrm{{ReLU}}({num(z1[0])}) = {num(a1[0])}$",
           "h2": rf"unit 2: $z = {num(z1[1])}$" "\n" rf"$a = \mathrm{{ReLU}}({num(z1[1])}) = {num(a1[1])}$",
           "o": rf"$\hat y = {num(a1[0])} \cdot {num(W2[0])} + {num(a1[1])} \cdot {num(W2[1])} = {num(yhat)}$",
           "L": rf"$\mathcal{{L}} = \frac{{1}}{{2}}({num(yhat)} - {num(y)})^2 = {num(L)}$"}
    weights = {("x1", "h1"): f"${num(W1[0, 0])}$", ("x2", "h1"): f"${num(W1[0, 1])}$",
               ("x1", "h2"): f"${num(W1[1, 0])}$", ("x2", "h2"): f"${num(W1[1, 1])}$",
               ("h1", "o"): f"${num(W2[0])}$", ("h2", "o"): f"${num(W2[1])}$", ("o", "L"): f"$y = {num(y)}$"}
    bwd = {"x1": f"$x_1 = {num(x[0])}$", "x2": f"$x_2 = {num(x[1])}$",
           "h1": rf"${num(d2)} \cdot {num(W2[0])} = {num(back[0])}$, mask ${num(mask[0])}$" "\n" rf"$(\delta_1)_1 = {num(d1[0])}$",
           "h2": rf"${num(d2)} \cdot {num(W2[1])} = {num(back[1])}$, mask ${num(mask[1])}$" "\n" rf"$(\delta_1)_2 = {num(d1[1])}$",
           "o": rf"$\delta_2 = \hat y - y = {num(d2)}$", "L": r"$\bar{\mathcal{L}} = 1$"}
    grads = {("x1", "h1"): f"${num(dW1[0, 0])}$", ("x2", "h1"): f"${num(dW1[0, 1])}$",
             ("x1", "h2"): rf"${num(d1[1])} \cdot {num(x[0])} = {num(dW1[1, 0])}$",
             ("x2", "h2"): rf"${num(d1[1])} \cdot {num(x[1])} = {num(dW1[1, 1])}$",
             ("h1", "o"): rf"${num(d2)} \cdot {num(a1[0])} = {num(dW2[0])}$",
             ("h2", "o"): rf"${num(d2)} \cdot {num(a1[1])} = {num(dW2[1])}$", ("o", "L"): ""}

    # label positions along the edges (fraction from the parent): keep the two crossing W1 edges apart
    frac_fwd = {("x1", "h1"): 0.6, ("x2", "h2"): 0.6, ("x1", "h2"): 0.2, ("x2", "h1"): 0.2,
                ("h1", "o"): 0.35, ("h2", "o"): 0.35, ("o", "L"): 0.5}
    frac_bwd = {("x1", "h1"): 0.4, ("x2", "h2"): 0.4, ("x1", "h2"): 0.25, ("x2", "h1"): 0.25,
                ("h1", "o"): 0.6, ("h2", "o"): 0.6}
    side_fwd = {("x2", "h1"): -0.2}                        # the rising diagonal is labelled below itself
    side_bwd = {("x2", "h1"): -0.2, ("x2", "h2"): -0.2}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.8))
    draw_graph(ax1, pos, size, edges, fwd, colors, FWD, reverse=False, labels=weights, frac=frac_fwd,
               offsets=side_fwd, fontsize=9)
    diagram(ax1, (-1.0, 10.4), (-0.35, 2.4),
            rf"(a) Forward pass of the 2–2–1 network: $z_1 = ({num(z1[0])}, {num(z1[1])})$, "
            rf"$a_1 = ({num(a1[0])}, {num(a1[1])})$, $\hat y = {num(yhat)}$, loss ${num(L)}$",
            "edge labels: the weights $W_1$ (left) and $W_2$ (right); the first hidden unit is off")
    draw_graph(ax2, pos, size, edges, bwd, colors, BWD, reverse=True, labels=grads, frac=frac_bwd,
               offsets=side_bwd, fontsize=9, dead=("h1",))
    diagram(ax2, (-1.0, 10.4), (-0.35, 2.4),
            rf"(b) Backward pass: $\delta_2 = {num(d2)}$ comes back through $W_2^\top$, is masked by "
            rf"$f'(z_1) = ({num(mask[0])}, {num(mask[1])})$, gives $\delta_1 = ({num(d1[0])}, {num(d1[1])})$",
            rf"edge labels: the weight gradients $\delta_l a_{{l-1}}^\top$ — $\partial\mathcal{{L}}/\partial W_2 = "
            rf"({num(dW2[0])}, {num(dW2[1])})$, $\partial\mathcal{{L}}/\partial W_1$ has rows "
            rf"$({num(dW1[0, 0])}, {num(dW1[0, 1])})$ and $({num(dW1[1, 0])}, {num(dW1[1, 1])})$; "
            r"bias gradients equal $\delta$")
    fig.tight_layout()
    return fig


def fig_matmul_vjp_shapes():
    """Section 6: the 2 x 2 worked example entry by entry, the shape test with unequal sides, broadcast = sum."""
    A = np.array([[1., 2.], [3., 4.]]); B = np.array([[1., 0.], [1., 1.]])
    Y = A @ B; G = np.ones_like(Y)                                              # L = sum of Y, so G is all ones
    Abar, Bbar, wrong = G @ B.T, A.T @ G, B.T @ G
    assert np.allclose(Y, [[3, 2], [7, 4]]) and Y.sum() == 16
    assert np.allclose(Abar, [[1, 2], [1, 2]]) and np.allclose(Bbar, [[4, 4], [6, 6]])
    spot = G[0, 0] * B[1, 0] + G[0, 1] * B[1, 1]                                # A_12 enters Y_11 and Y_12
    assert spot == Abar[0, 1] == 2 and np.allclose(wrong, [[2, 2], [1, 1]])      # the wrong guess has the right shape
    m, k, n = 2, 3, 4                                                           # unequal sides make the shape test bite

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.15, 1.2, 1]})

    # (a) the worked example, with the uses of A_12 and the sources of A-bar_12 highlighted
    def table(ax, x0, top, M, name, hl=(), cell=0.5):
        r, c = M.shape
        for i in range(r):
            for j in range(c):
                ax.add_patch(Rectangle((x0 + j * cell, top - (i + 1) * cell), cell, cell,
                                       fc=HIGHLIGHT if (i, j) in hl else "white", ec=GREY, lw=0.8))
                ax.text(x0 + (j + 0.5) * cell, top - (i + 0.5) * cell, num(M[i, j]), ha="center", va="center", fontsize=9)
        ax.text(x0 + c * cell / 2, top + 0.1, name, ha="center", va="bottom", fontsize=10)
    table(ax1, 0, 1.5, A, "$A$", hl=[(0, 1)]); table(ax1, 1.5, 1.5, B, "$B$", hl=[(1, 0), (1, 1)])
    table(ax1, 3.0, 1.5, Y, "$Y$", hl=[(0, 0), (0, 1)])
    table(ax1, 0, -0.5, G, "$G$"); table(ax1, 1.5, -0.5, B.T, r"$B^\top$", hl=[(0, 1), (1, 1)])
    table(ax1, 3.0, -0.5, Abar, r"$\bar A = G B^\top$", hl=[(0, 1)])
    for yy in (1.0, -1.0):
        ax1.text(1.25, yy, r"$\times$", ha="center", va="center"); ax1.text(2.75, yy, "$=$", ha="center", va="center")
    diagram(ax1, (-0.3, 4.3), (-1.8, 2.0), r"(a) $Y = AB$, $L = \sum_{ij} Y_{ij}$, so $G = \mathbf{1}\mathbf{1}^\top$",
            rf"$A_{{12}}$ feeds $Y_{{11}}$ (via $B_{{21}} = {num(B[1, 0])}$) and $Y_{{12}}$ (via $B_{{22}} = {num(B[1, 1])}$)," "\n"
            rf"so $\bar A_{{12}} = G_{{11}} B_{{21}} + G_{{12}} B_{{22}} = {num(spot)}$")

    # (b) Strang-style blocks: only G B^T lands on A's shape
    def block(ax, x0, top, rows, cols, name, shape, color, scale=0.3):
        ax.add_patch(Rectangle((x0, top - rows * scale), cols * scale, rows * scale, fc=color, ec="k", lw=1, alpha=0.3))
        ax.text(x0 + cols * scale / 2, top - rows * scale / 2, name, ha="center", va="center", fontsize=9.5)
        ax.text(x0 + cols * scale / 2, top + 0.04, shape, ha="center", va="bottom", fontsize=8.5)
    block(ax2, 0, 2.0, m, k, "$A$", r"$m \times k$", "C0"); block(ax2, 1.2, 2.0, k, n, "$B$", r"$k \times n$", "C1")
    block(ax2, 2.7, 2.0, m, n, "$Y$", r"$m \times n$", FWD)
    ax2.text(1.05, 1.7, r"$\times$", ha="center", va="center"); ax2.text(2.55, 1.7, "$=$", ha="center", va="center")
    block(ax2, 0, 0.5, m, n, "$G$", r"$m \times n$", FWD); block(ax2, 1.5, 0.5, n, k, r"$B^\top$", r"$n \times k$", "C1")
    block(ax2, 2.7, 0.5, m, k, r"$\bar A$", r"$m \times k$", "C0")
    ax2.text(1.35, 0.2, r"$\times$", ha="center", va="center"); ax2.text(2.55, 0.2, "$=$", ha="center", va="center")
    ax2.text(1.95, -1.05, r"$G B^\top$: $(m \times n)(n \times k) = m \times k$, the shape of $A$" "\n"
             r"$B^\top G$: $(n \times k)(m \times n)$, inner sizes $k \neq m$: no product",
             ha="center", va="center", fontsize=8.5)
    diagram(ax2, (-0.3, 4.1), (-1.5, 2.4), r"(b) The shape test: only $G B^\top$ has the shape of $A$",
            f"blocks drawn with $m = {m}$, $k = {k}$, $n = {n}$")

    # (c) a bias broadcast to every row forwards is summed over the rows backwards
    sc, rows = 0.42, 3
    for i in range(rows):
        for j in range(2):
            ax3.add_patch(Rectangle((j * sc, 2.0 - (i + 1) * sc), sc, sc, fc="C1", alpha=0.25, ec="k", lw=0.8))
            ax3.text((j + 0.5) * sc, 2.0 - (i + 0.5) * sc, f"$b_{j + 1}$", ha="center", va="center", fontsize=9)
            ax3.add_patch(Rectangle((2.0 + j * sc, 2.0 - (i + 1) * sc), sc, sc, fc=FWD, alpha=0.25, ec="k", lw=0.8))
            ax3.text(2.0 + (j + 0.5) * sc, 2.0 - (i + 0.5) * sc, f"$G_{{{i + 1}{j + 1}}}$", ha="center", va="center", fontsize=8.5)
    for j in range(2):
        ax3.add_patch(Rectangle((j * sc, 0.0), sc, sc, fc="C1", alpha=0.5, ec="k", lw=0.8))
        ax3.text((j + 0.5) * sc, sc / 2, f"$b_{j + 1}$", ha="center", va="center", fontsize=9)
        ax3.add_patch(Rectangle((2.0 + j * sc, 0.0), sc, sc, fc=BWD, alpha=0.3, ec="k", lw=0.8))
        ax3.text(2.0 + (j + 0.5) * sc, sc / 2, rf"$\bar b_{j + 1}$", ha="center", va="center", fontsize=9)
        ax3.annotate("", xy=(2.0 + (j + 0.5) * sc, 0.48), xytext=(2.0 + (j + 0.5) * sc, 0.7),
                     arrowprops=dict(arrowstyle="-|>", color=BWD))
    ax3.annotate("", xy=(sc, 0.7), xytext=(sc, 0.48), arrowprops=dict(arrowstyle="-|>", color=FWD))
    ax3.text(sc, 2.08, "copy up", ha="center", va="bottom", fontsize=9, color=FWD)
    ax3.text(2.0 + sc, 2.08, "sum down", ha="center", va="bottom", fontsize=9, color=BWD)
    ax3.text(sc, -0.3, r"$\mathbf{1} b^\top$", ha="center", va="top", fontsize=9)
    ax3.text(2.0 + sc, -0.3, r"$\bar b = G^\top \mathbf{1}$", ha="center", va="top", fontsize=9)
    diagram(ax3, (-0.25, 3.1), (-0.8, 2.5), "(c) Broadcast forwards, sum backwards",
            r"$b$ is used once per row, so $\bar b_j$ collects one" "\n" r"contribution per row: $\bar b_j = G_{1j} + G_{2j} + G_{3j}$")
    fig.tight_layout()
    return fig


def fig_gradient_check_pitfalls():
    """Section 7: the planted tanh bug, the ReLU false alarm, and float32 against float64 on the same check."""
    h = 1e-5
    # (a) the tanh rule: correct (output) versus bug (input), compared at three inputs
    checks = {}
    for a0 in (0.1, 0.5, 1.0):
        good, bad = 1 - np.tanh(a0) ** 2, 1 - a0 ** 2
        witness = (np.tanh(a0 + h) - np.tanh(a0 - h)) / (2 * h)
        checks[a0] = (good, bad, abs(good - bad) / max(abs(good), abs(bad), 1e-12))
        assert abs(witness - good) < 1e-9                                       # the central difference sides with the output rule
    assert abs(checks[0.5][0] - 0.7864) < 5e-5 and abs(checks[0.5][2] - 0.046) < 1e-3     # Section 7, Case study 3
    assert abs(checks[0.1][0] - 0.9901) < 5e-5 and abs(checks[0.1][2] - 7e-5) < 1e-5
    assert abs(checks[1.0][0] - 0.42) < 5e-4 and checks[1.0][1] == 0 and checks[1.0][2] == 1
    # (b) ReLU within h of its kink (Section 8's worked example, pitfall 1)
    z0 = 1e-6
    relu = lambda z: max(z, 0.0)                                                # noqa: E731
    ghat_relu = (relu(z0 + h) - relu(z0 - h)) / (2 * h)
    assert abs(ghat_relu - 0.55) < 1e-12
    # (c) Section 7's check of L = (wx + b - y)^2 at w = 2, in float64 and entirely in float32
    g = 12.0
    hs = np.logspace(-8, -1, 141)
    L64 = lambda w: (w * 3.0 + 1.0 - 5.0) ** 2                                  # noqa: E731
    r64 = np.array([abs((L64(2.0 + hh) - L64(2.0 - hh)) / (2 * hh) - g) / g for hh in hs])

    def L32(w):
        return (w * np.float32(3) + np.float32(1) - np.float32(5)) ** 2
    w32 = np.float32(2)
    r32 = np.array([abs(float((L32(w32 + np.float32(hh)) - L32(w32 - np.float32(hh))) / (np.float32(2) * np.float32(hh))) - g) / g
                    for hh in hs])
    i5 = int(np.argmin(np.abs(hs - 1e-5)))
    assert np.isclose(hs[i5], 1e-5) and 1e-11 < r64[i5] < 5e-11 and 1e-3 < r32[i5] < 2e-3
    assert abs((L64(2 + h) - L64(2 - h)) - 0.00024) < 1e-12                     # the guide's difference 0.00024

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1.2, 0.95, 1.1]})
    a = np.linspace(-1.6, 1.6, 400)
    ax1.plot(a, 1 - np.tanh(a) ** 2, color="C0", lw=2, label=r"$1 - \tanh^2 a$ (correct: uses the output)")
    ax1.plot(a, 1 - a ** 2, color="C3", lw=2, ls="--", label=r"$1 - a^2$ (bug: uses the input)")
    lines = []
    for a0, off, ha in [(0.1, (-5, 5), "right"), (0.5, (6, 4), "left"), (1.0, (6, 2), "left")]:
        good, bad, r = checks[a0]
        ax1.plot([a0, a0], [bad, good], color="k", lw=1)
        ax1.plot(a0, good, "o", color="C0", ms=5, zorder=4); ax1.plot(a0, bad, "o", color="C3", ms=5, zorder=4)
        label(ax1, (a0, good), f"$a = {num(a0)}$", offset=off, ha=ha, fontsize=8.5)
        lines.append(rf"$a = {num(a0)}$: ${good:.4g}$ vs ${num(round(bad, 4))}$, $r = {rel(r)}$")
    ax1.text(1.52, 2.25, "\n".join(lines), ha="right", va="top", fontsize=8, linespacing=1.4)
    ax1.set_xlim(-1.6, 1.6); ax1.set_ylim(-1.6, 2.3)
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.95)
    ax1.set_title("(a) A bug that hides near $a = 0$", fontsize=10)
    ax1.set_xlabel("$a$ (input of tanh)"); ax1.set_ylabel(r"claimed derivative $\tanh'(a)$")

    u = 1e-5                                                                    # axis unit of panel (b)
    zz = np.array([-2e-5, 0.0, 2e-5])
    ax2.plot(zz / u, np.maximum(zz, 0) / u, color="C0", lw=2)
    ax2.plot([(z0 - h) / u, (z0 + h) / u], [relu(z0 - h) / u, relu(z0 + h) / u], "o--", color="C3", lw=1.6, ms=6)
    ax2.axvline(z0 / u, color=GREY, ls=":", lw=1)
    ax2.text(1.9, 1.62, "ReLU$(z)$", color="C0", ha="right", va="top", fontsize=9)
    ax2.text(1.9, -0.52, f"central difference:\nslope $({relu(z0 + h) / u:.1f} - 0) / 2 = {ghat_relu:.2f}$",
             color="C3", ha="right", va="bottom", fontsize=8.5)
    ax2.text(z0 / u - 0.1, 1.9, "$z_0 = 10^{-6}$\nanalytic slope $1$", ha="right", va="top", fontsize=8.5)
    label(ax2, ((z0 - h) / u, 0), "$z_0 - h$", offset=(0, -13), ha="center", fontsize=8.5)
    label(ax2, ((z0 + h) / u, (z0 + h) / u), "$z_0 + h$", offset=(-7, 5), ha="right", fontsize=8.5)
    ax2.set_xlim(-2, 2); ax2.set_ylim(-0.6, 2.0)
    ax2.set_title("(b) A false alarm at a kink", fontsize=10)
    ax2.set_xlabel("$z$ in units of $10^{-5}$"); ax2.set_ylabel("ReLU$(z)$ in units of $10^{-5}$")

    ax3.loglog(hs, r64, color="C0", lw=2, label="float64")
    ax3.loglog(hs, r32, color="C1", lw=2, label="float32")
    ax3.axhspan(1e-2, 1, color="C3", alpha=0.08, lw=0); ax3.axhspan(1e-4, 1e-2, color="C1", alpha=0.08, lw=0)
    ax3.axhspan(1e-7, 1e-4, color="C2", alpha=0.06, lw=0)
    for xx, yy, txt, col, ha in [(7e-2, 1e-1, "bug", "C3", "right"), (7e-2, 1e-3, "suspicious", "C1", "right"),
                                 (1.4e-8, 3e-6, "acceptable with kinks", "C2", "left"), (1.4e-8, 1e-11, "correct", "C0", "left")]:
        ax3.text(xx, yy, txt, color=col, fontsize=8, va="center", ha=ha)
    ax3.axvline(1e-5, color=GREY, ls=":", lw=1)
    ax3.plot([1e-5, 1e-5], [r64[i5], r32[i5]], "o", color="k", ms=5, zorder=4)
    label(ax3, (1e-5, r64[i5]), rf"$h = 10^{{-5}}$, float64:" "\n" rf"$r = {sci(r64[i5])}$", offset=(6, 4), fontsize=8.5)
    label(ax3, (1e-5, r32[i5]), rf"float32: $r = {sci(r32[i5])}$", offset=(6, 3), fontsize=8.5)
    ax3.set_xlim(1e-8, 1e-1); ax3.set_ylim(1e-12, 1)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax3.set_title("(c) Precision sets the floor of the same check", fontsize=10)
    ax3.set_xlabel("step $h$"); ax3.set_ylabel(rf"relative error $r$ against the analytic ${num(g)}$")
    fig.tight_layout()
    return fig


def fig_activation_memory():
    """Case study 1: activations kept for the backward pass against the weights, and their growth with s."""
    s, b, h, a, layers = 1024, 8, 768, 12, 12                                   # GPT-2 small, batch 8, context 1,024
    n_params, bytes_per = 124_439_808, 2                                        # fp16

    def act_gb(s, per_layer=False):                                             # Korthikanti et al. (2022), per layer
        return (1 if per_layer else layers) * s * b * h * (34 + 5 * a * s / h) / 1e9
    sbh = s * b * h
    linear, attention = layers * 34 * sbh / 1e9, layers * 5 * a * s ** 2 * b / 1e9
    total, weights, ckpt = act_gb(s), n_params * bytes_per / 1e9, layers * 2 * sbh / 1e9
    assert sbh == 6_291_456 and 34 + 5 * a * s / h == 114 and round(act_gb(s, per_layer=True), 2) == 0.72
    assert round(total, 1) == 8.6 and abs(linear + attention - total) < 1e-9 and round(weights, 2) == 0.25
    assert round(total / weights) == 35 and round(total / ckpt) == 57 and round(2 * sbh / 1e6, 1) == 12.6
    assert round(attention / total, 2) == 0.70

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    cats = ["activations kept for\nthe backward pass, 12 layers", "weights (fp16)",
            "activations with checkpointing\n($2 s b h$ per layer)"]
    ax1.barh(0, linear, height=0.55, color=BWD, alpha=0.45, label="linear terms, $34\\,s b h$")
    ax1.barh(0, attention, left=linear, height=0.55, color=BWD, label="attention terms, $5\\,a s^2 b$")
    ax1.text(linear / 2, 0, f"{linear:.2f} GB", ha="center", va="center", fontsize=8.5)
    ax1.text(linear + attention / 2, 0, f"{attention:.2f} GB", ha="center", va="center", fontsize=8.5, color="white")
    ax1.barh(1, weights, height=0.55, color="C0")
    ax1.barh(2, ckpt, height=0.55, color=FWD)
    for yy, val, txt in [(0, total, f"{total:.1f} GB, {total / weights:.0f} times the weights"), (1, weights, f"{weights:.2f} GB"),
                         (2, ckpt, f"{ckpt:.2f} GB, {total / ckpt:.0f} times less")]:
        ax1.text(val + 0.15, yy, txt, va="center", ha="left", fontsize=9)
    ax1.set_yticks(range(3), cats); ax1.invert_yaxis(); ax1.set_xlim(0, 13.5)
    ax1.grid(axis="y", visible=False)
    ax1.legend(loc="center right", fontsize=8.5, framealpha=0.95)
    ax1.set_title("(a) What the backward pass must keep: GPT-2 small", fontsize=10)
    ax1.set_xlabel("gigabytes (batch 8, context 1,024, 16-bit)"); ax1.set_ylabel("")

    ss = 2 ** np.arange(7, 14)                                                  # 128 ... 8192
    assert np.allclose(act_gb(ss), [0.415, 1.019, 2.793, 8.607, 29.29, 106.90, 407.08], rtol=2e-3)
    ax2.loglog(ss, act_gb(ss), "o-", color=BWD, lw=2, ms=4, label="activations, 12 layers")
    ax2.loglog(ss, layers * 34 * ss * b * h / 1e9, color="C1", ls="--", lw=1.4, label="linear terms $\\propto s$")
    ax2.loglog(ss, layers * 5 * a * ss ** 2 * b / 1e9, color=FWD, ls=":", lw=1.6, label="attention terms $\\propto s^2$")
    ax2.loglog(ss, layers * 2 * ss * b * h / 1e9, color=FWD, lw=1.6, label="with checkpointing")
    ax2.axhline(weights, color="C0", lw=1.6, label=f"weights: {weights:.2f} GB")
    ax2.plot(s, total, "o", color="k", ms=6, zorder=4)
    label(ax2, (s, total), f"the case study:\n$s = {s:,}$, {total:.1f} GB", offset=(8, -4), va="top", fontsize=9)
    ax2.set_xlim(100, 10000); ax2.set_ylim(0.01, 1000)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax2.set_title("(b) Activations grow with $s$ and $s^2$; weights do not", fontsize=10)
    ax2.set_xlabel("sequence length $s$ (batch 8)"); ax2.set_ylabel("GB (per-layer formula, fp16)")
    fig.tight_layout()
    return fig


FIGURES = {
    "computational-graph": fig_computational_graph,
    "forward-vs-reverse": fig_forward_vs_reverse,
    "jacobian-rows-columns": fig_jacobian_rows_columns,
    "backprop-2-2-1": fig_backprop_2_2_1,
    "matmul-vjp-shapes": fig_matmul_vjp_shapes,
    "gradient-check-pitfalls": fig_gradient_check_pitfalls,
    "activation-memory": fig_activation_memory,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 06 concept-guide figures.")
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
