"""
figbuild.py — tiny helper library for the *figures* in the concept guides.

The concept guides (README.md) are plain Markdown, so their figures are static
SVG files committed next to the guide, in `<module>/figures/`. A module's
figures are drawn by a plain script, `tools/builders/figures_XX.py`, using the
helpers below. They fix one visual style and make the output reproducible:
regenerating a module's figures changes no file unless a figure changed.

Usage sketch
------------
    import sys; sys.path.insert(0, "tools")
    from figbuild import plt, arrow, save, label, right_angle

    fig, ax = plt.subplots(figsize=(5, 4.5))
    arrow(ax, (3, 4), color="C0", label=r"$\\mathbf{u}$")      # from the origin
    arrow(ax, (4, 3), start=(3, 4), color="C1")                 # tip-to-tail
    ax.set_title("Tip-to-tail addition"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    save(fig, "part-1-linear-algebra/01-vectors-and-geometry/figures/example.svg")

Conventions (see CONTRIBUTING.md): a white background, so the figure also reads
on GitHub's dark theme; every panel has a title and axis labels; the numbers
drawn are the guide's own worked-example numbers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

STYLE = {
    "figure.facecolor": "white",   # a white card stays readable on GitHub's dark theme
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "mathtext.fontset": "cm",      # maths in the figures matches the LaTeX in the guides
    "svg.fonttype": "path",        # glyphs stored as paths: identical rendering everywhere
    "svg.hashsalt": "ai-math",     # stable element ids, so regenerated SVGs are byte-identical
}
plt.rcParams.update(STYLE)


def arrow(ax, vec, start=(0, 0), color="C0", lw=2.0, label=None, label_offset=(6, 4),
          alpha=1.0, fontsize=13, zorder=3):
    """Draw `vec` as an arrow from `start` (data coordinates), optionally labelled at its tip."""
    x0, y0 = start
    x1, y1 = x0 + vec[0], y0 + vec[1]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                                shrinkA=0, shrinkB=0, mutation_scale=15))
    if label:
        ax.annotate(label, (x1, y1), textcoords="offset points", xytext=label_offset,
                    color=color, fontsize=fontsize, zorder=zorder)
    ax.update_datalim([(x0, y0), (x1, y1)])   # annotate() alone does not autoscale the axes


GREY = "0.55"   # for construction lines and grids that must not compete with the data


def label(ax, xy, text, color="k", offset=(6, 4), fontsize=11, ha="left", va="bottom"):
    """Place `text` next to the data point `xy`, offset in points so it never scales with the axes."""
    ax.annotate(text, xy, textcoords="offset points", xytext=offset, color=color,
                fontsize=fontsize, ha=ha, va=va)


def right_angle(ax, corner, d1, d2, size=0.18, color="k"):
    """Small square marking the right angle at `corner` between the directions d1 and d2."""
    import numpy as np
    d1, d2 = np.asarray(d1, float), np.asarray(d2, float)
    d1, d2 = size * d1 / np.linalg.norm(d1), size * d2 / np.linalg.norm(d2)
    c = np.asarray(corner, float)
    pts = np.array([c + d1, c + d1 + d2, c + d2])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1)


def save(fig, path, preview_dir=None):
    """Write `fig` as a reproducible SVG (no timestamp or version stamp) and close it.

    Relative paths are taken from the repository root. With `preview_dir`, a PNG
    copy is written there too, for viewers that cannot display SVG.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    # matplotlib's tight bbox leaves out the axis labels of 3-D axes; include them explicitly
    extra = [lab for ax in fig.axes if hasattr(ax, "zaxis") for lab in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label)]
    fig.savefig(path, format="svg", bbox_inches="tight", bbox_extra_artists=extra or None,
                metadata={"Date": None, "Creator": None})
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / (path.stem + ".png"), dpi=110, bbox_inches="tight", bbox_extra_artists=extra or None)
    plt.close(fig)
    return path
