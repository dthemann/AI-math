"""Figures for the Module 07 concept guide — Probability Foundations.

Writes SVG files into part-3-probability/07-probability-foundations/figures/,
where README.md embeds them. Every number drawn is a worked-example number from
the guide, recomputed here and checked against the value the guide prints, so
picture and text agree exactly. Nothing is random: the dice averages are exact
convolutions, not simulations.

    python tools/builders/figures_07.py                     # (re)write every figure
    python tools/builders/figures_07.py conditioning-tree   # only the named figure(s)
    python tools/builders/figures_07.py --preview DIR       # also write PNG previews to DIR
"""
import argparse
import pathlib
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from figbuild import plt, arrow, save, label, right_angle, GREY  # noqa: E402,F401

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

FIG_DIR = ROOT / "part-3-probability/07-probability-foundations/figures"

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


def node(ax, xy, text, ec, fc="white", fontsize=9.5, ha="center", lw=1.5):
    """A rounded tree node; it is opaque and drawn above the branches, so they appear to end at its border."""
    ax.text(*xy, text, ha=ha, va="center", fontsize=fontsize, zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=lw))


def branch(ax, p0, p1, color, text=None, lw=1.6, where=0.5, fontsize=9):
    """A straight tree branch from p0 to p1, optionally labelled with its probability at the point `where`."""
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=1, solid_capstyle="butt")
    if text:
        xm, ym = p0[0] + where * (p1[0] - p0[0]), p0[1] + where * (p1[1] - p0[1])
        ax.text(xm, ym, text, ha="center", va="center", fontsize=fontsize, color=color, zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


def bare(ax, xlabel, ylabel, xticks=(), xticklabels=()):
    """A canvas for a diagram: no frame, no grid, no y ticks; the x ticks name the stages."""
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticklabels))
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


# ============================================================================
# FIGURES (one function per file; each returns the matplotlib Figure)
# ============================================================================
def fig_counting_grid():
    """Section 2: two dice as 36 equally likely cells (events A, B, C); inclusion-exclusion on a 52-card deck."""
    faces = range(1, 7)
    omega = [(i, j) for i in faces for j in faces]
    A = [w for w in omega if w[0] + w[1] == 7]                  # sum is 7
    B = [w for w in omega if w[0] == w[1]]                      # doubles
    C = [w for w in omega if 6 in w]                            # at least one six
    no_six = [w for w in omega if 6 not in w]
    P = lambda E: Fraction(len(E), len(omega))                  # noqa: E731
    assert len(omega) == 36 and len(A) == len(B) == 6 and not set(A) & set(B)
    assert len(no_six) == 25 and P(C) == 1 - P(no_six) == Fraction(11, 36)
    pAB = P(A) + P(B)                                           # disjoint, so the probabilities simply add
    assert pAB == P(set(A) | set(B)) == Fraction(1, 3)
    check(P(C), 0.306, 5e-4)

    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    suits = ["spades", "clubs", "diamonds", "hearts"]           # bottom row to top row
    deck = [(r, s) for s in suits for r in ranks]
    aces = [c for c in deck if c[0] == "A"]
    hearts = [c for c in deck if c[1] == "hearts"]
    both = [c for c in aces if c in hearts]
    union = [c for c in deck if c in aces or c in hearts]
    p_union = Fraction(len(aces), 52) + Fraction(len(hearts), 52) - Fraction(len(both), 52)
    assert len(union) == 16 and p_union == Fraction(len(union), 52) == Fraction(4, 13)
    check(p_union, 0.308, 5e-4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [6, 13]})

    # (a) the 6 x 6 grid: fills for A and B, a hatched outline for C (it overlaps both)
    for cells, col in [(A, "C0"), (B, "C1")]:
        for i, j in cells:
            ax1.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fc=col, alpha=0.6, lw=0))
    ax1.add_patch(Polygon([(5.5, 0.5), (6.5, 0.5), (6.5, 6.5), (0.5, 6.5), (0.5, 5.5), (5.5, 5.5)], closed=True,
                          fc="none", ec="C2", lw=2.5, hatch="///", zorder=3))
    for k in np.arange(0.5, 7):
        ax1.plot([k, k], [0.5, 6.5], color=GREY, lw=0.7, zorder=2)
        ax1.plot([0.5, 6.5], [k, k], color=GREY, lw=0.7, zorder=2)
    ax1.set_xlim(0.5, 6.5); ax1.set_ylim(0.5, 6.5); ax1.set_aspect("equal"); ax1.grid(False)
    ax1.set_xticks(list(faces)); ax1.set_yticks(list(faces)); ax1.tick_params(length=0)
    handles = [Patch(fc="C0", alpha=0.6, label=f"$A$: sum is 7, {len(A)} cells, $P(A) = {len(A)}/36$"),
               Patch(fc="C1", alpha=0.6, label=f"$B$: doubles, {len(B)} cells, $P(B) = {len(B)}/36$"),
               Patch(fc="none", ec="C2", lw=2, hatch="///",
                     label=f"$C$: at least one six, $36 - 5 \\times 5 = {len(C)}$ cells,\n"
                           f"$P(C) = 11/36 \\approx {float(P(C)):.3f}$"),
               Line2D([], [], ls="none", label="$A$ and $B$ share no cell: $P(A \\cup B) = 12/36 = 1/3$")]
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, handlelength=1.6,
               frameon=False)
    ax1.set_title("(a) Two dice: 36 equally likely cells,\nso $P(\\text{event}) = \\text{cells} / 36$")
    ax1.set_xlabel("first die"); ax1.set_ylabel("second die")
    ax1.set_anchor("N")

    # (b) the deck as a 4 x 13 grid: aces column, hearts row, the shared cell outlined
    col_of = {r: k for k, r in enumerate(ranks)}
    row_of = {s: k for k, s in enumerate(suits)}
    for cells, col in [(aces, "C0"), (hearts, "C1")]:
        for r, s in cells:
            ax2.add_patch(Rectangle((col_of[r] - 0.5, row_of[s] - 0.5), 1, 1, fc=col, alpha=0.6, lw=0))
    (rb, sb), = both
    ax2.add_patch(Rectangle((col_of[rb] - 0.5, row_of[sb] - 0.5), 1, 1, fc="none", ec="C3", lw=2.8, zorder=4, clip_on=False))
    ax2.plot(col_of[rb], row_of[sb], "o", color="C3", ms=7, zorder=5)
    for k in np.arange(-0.5, 13):
        ax2.plot([k, k], [-0.5, 3.5], color=GREY, lw=0.7, zorder=2)
    for k in np.arange(-0.5, 4):
        ax2.plot([-0.5, 12.5], [k, k], color=GREY, lw=0.7, zorder=2)
    ax2.annotate("ace of hearts: in both events,\nso subtract it once", (col_of[rb] + 0.2, row_of[sb] - 0.25),
                 xytext=(2.0, 1.0), ha="left", va="center", fontsize=10, color="C3", zorder=6,
                 bbox=dict(WHITE_BOX, alpha=1), arrowprops=dict(arrowstyle="-|>", color="C3", lw=1, shrinkA=2, shrinkB=2))
    ax2.text(0.5, -0.62, f"$P(\\mathrm{{ace}} \\cup \\mathrm{{heart}}) = 4/52 + 13/52 - 1/52 = 16/52 = 4/13 "
             f"\\approx {float(p_union):.3f}$", transform=ax2.transAxes, ha="center", va="top", fontsize=11.5)
    ax2.set_xlim(-0.5, 12.5); ax2.set_ylim(-0.5, 3.5); ax2.set_aspect("equal"); ax2.grid(False)
    ax2.set_xticks(range(13)); ax2.set_xticklabels(ranks)
    ax2.set_yticks(range(4)); ax2.set_yticklabels(suits); ax2.tick_params(length=0)
    handles = [Patch(fc="C0", alpha=0.6, label=f"aces: {len(aces)} cells"),
               Patch(fc="C1", alpha=0.6, label=f"hearts: {len(hearts)} cells"),
               Patch(fc="none", ec="C3", lw=2, label="in both: counted twice in 4 + 13")]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=9, frameon=False)
    ax2.set_anchor("N")                                         # both panels hang from the top; the formula fills the space below
    ax2.set_title("(b) Inclusion–exclusion on a 52-card deck:\nadd both events, subtract the overlap once")
    ax2.set_xlabel("rank"); ax2.set_ylabel("suit")
    fig.tight_layout()
    return fig


def fig_conditioning_tree():
    """Section 3: the two-stage manufacturing tree; conditioning on D2 keeps two leaves and rescales them."""
    p1, p2_d1, p2_ok = 0.04, 0.10, 0.02                         # P(D1), P(D2 | D1), P(D2 | not D1)
    leaf = {("D1", "D2"): p1 * p2_d1, ("D1", "ok"): p1 * (1 - p2_d1),
            ("ok", "D2"): (1 - p1) * p2_ok, ("ok", "ok"): (1 - p1) * (1 - p2_ok)}
    for key, guide in [(("D1", "D2"), 0.0040), (("D1", "ok"), 0.0360), (("ok", "D2"), 0.0192), (("ok", "ok"), 0.9408)]:
        check(leaf[key], guide, 1e-12)
    check(sum(leaf.values()), 1.0, 1e-12)
    pD2 = check(leaf["D1", "D2"] + leaf["ok", "D2"], 0.0232, 1e-12)
    check(1 - leaf["ok", "ok"], 0.0592, 1e-12)                  # defective at either stage
    check(p1 * pD2, 0.00093, 5e-6)                              # what independence would require
    post = check(leaf["D1", "D2"] / pD2, 0.172, 5e-4)           # P(D1 | D2)
    check(post / p1, 4.3, 0.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1.55, 1]})

    # (a) the tree: multiply along a path, add the D2 leaves
    root, n1, n0 = (0.0, 0.5), (1.0, 0.76), (1.0, 0.24)
    lx = 1.78                                                   # leaves are left-aligned here
    ys = {("D1", "D2"): 0.93, ("D1", "ok"): 0.63, ("ok", "D2"): 0.37, ("ok", "ok"): 0.07}
    col = {("D1", "D2"): "C3", ("D1", "ok"): GREY, ("ok", "D2"): "C0", ("ok", "ok"): GREY}
    branch(ax1, root, n1, "C3", f"{p1:.2f}")
    branch(ax1, root, n0, "C0", f"{1 - p1:.2f}")
    for (a, b), probs in [(("D1", "D2"), p2_d1), (("D1", "ok"), 1 - p2_d1), (("ok", "D2"), p2_ok), (("ok", "ok"), 1 - p2_ok)]:
        branch(ax1, n1 if a == "D1" else n0, (lx, ys[a, b]), col[a, b], f"{probs:.2f}", where=0.68)
    node(ax1, root, "all\nparts", GREY)
    node(ax1, n1, "$D_1$: damaged\nin moulding", "C3")
    node(ax1, n0, "$D_1^c$: not\ndamaged", "C0")
    names = {("D1", "D2"): "$D_1 \\cap D_2$", ("D1", "ok"): "$D_1 \\cap D_2^c$",
             ("ok", "D2"): "$D_1^c \\cap D_2$", ("ok", "ok"): "$D_1^c \\cap D_2^c$"}
    factors = {("D1", "D2"): (p1, p2_d1), ("D1", "ok"): (p1, 1 - p2_d1),
               ("ok", "D2"): (1 - p1, p2_ok), ("ok", "ok"): (1 - p1, 1 - p2_ok)}
    for key in ys:
        f1, f2 = factors[key]
        kept = key[1] == "D2"
        node(ax1, (lx, ys[key]), f"{names[key]}\n${f1:.2f} \\times {f2:.2f} = {leaf[key]:.4f}$",
             col[key], ha="left", lw=2.2 if kept else 1.0, fontsize=9.5)
    sx, sy = 3.3, 0.65                                          # the sum, between the two kept leaves
    for key, dy in [(("D1", "D2"), 0.1), (("ok", "D2"), -0.1)]:
        ax1.annotate("", xy=(sx - 0.42, sy + dy), xytext=(2.66, ys[key]),
                     arrowprops=dict(arrowstyle="-|>", color=col[key], lw=1.2, shrinkA=0, shrinkB=0))
    ax1.text(sx, sy, f"add the $D_2$\nleaves:\n$P(D_2) = {leaf['D1', 'D2']:.4f}$\n$+\\ {leaf['ok', 'D2']:.4f} = {pD2:.4f}$",
             ha="center", va="center", fontsize=9.5, zorder=4,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", lw=1))
    ax1.set_xlim(-0.3, 3.82); ax1.set_ylim(-0.03, 1.03)
    bare(ax1, "stage (a branch carries a conditional probability;\na leaf multiplies the branches on its path)", "branches",
         [0, 1, 2.2], ["start", "stage 1: moulding", "stage 2: finishing"])
    ax1.set_title("(a) Multiply along a path, add across leaves")

    # (b) conditioning as zooming in: the D2 sliver of all parts, rescaled by 1 / P(D2)
    w, xl, xr = 0.45, 0.0, 1.15
    ax2.bar(xl, 1 - pD2, bottom=pD2, width=w, color=GREY, alpha=0.25, hatch="..", ec=GREY, lw=0.8)
    ax2.bar(xl, leaf["D1", "D2"], width=w, color="C3", alpha=0.8, lw=0)
    ax2.bar(xl, leaf["ok", "D2"], bottom=leaf["D1", "D2"], width=w, color="C0", alpha=0.6, lw=0)
    ax2.bar(xr, post, width=w, color="C3", alpha=0.8, lw=0)
    ax2.bar(xr, 1 - post, bottom=post, width=w, color="C0", alpha=0.6, lw=0)
    for yl, yr in [(0, 0), (leaf["D1", "D2"], post), (pD2, 1)]:
        ax2.plot([xl + w / 2, xr - w / 2], [yl, yr], ":", color="k", lw=1)
    boxed(ax2, (xl, 0.55), f"$D_2^c$:\n{1 - pD2:.4f}\ndiscarded", fontsize=9.5)
    ax2.annotate(f"$D_2$:\n{pD2:.4f}", (xl, pD2 / 2), xytext=(xl, 0.12), ha="center", va="bottom", fontsize=9.5,
                 bbox=WHITE_BOX, zorder=6,                      # a leader down to the hairline D_2 sliver
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1, shrinkA=0, shrinkB=1))
    xt = xr + w / 2 + 0.07                                      # labels to the right of the rescaled bar
    ax2.text(xt, 0.6, f"$D_1^c$: {leaf['ok', 'D2']:.4f} / {pD2:.4f}\n≈ {1 - post:.3f}", ha="left", va="center",
             fontsize=9.5, color="C0")
    ax2.text(xt, post, f"$D_1$: {leaf['D1', 'D2']:.4f} / {pD2:.4f}\n≈ {post:.3f}, which is\n{post / p1:.1f} times the dashed\n"
             f"$P(D_1) = {p1:.2f}$, the\nvalue if independent", ha="left", va="center", fontsize=9.5,
             color="C3")
    ax2.plot([xr - w / 2, xr + w / 2], [p1, p1], "--", color="k", lw=1.6, zorder=5)
    ax2.set_xlim(-0.45, 3.3); ax2.set_ylim(0, 1.02)
    ax2.set_xticks([xl, xr]); ax2.set_xticklabels(["all parts", f"given $D_2$\n(divide by {pD2:.4f})"])
    ax2.grid(axis="x", visible=False)
    ax2.set_title("(b) Conditioning on $D_2$: keep its leaves,\nrescale them to sum to 1")
    ax2.set_xlabel("before and after learning that $D_2$ happened"); ax2.set_ylabel("probability")
    fig.tight_layout()
    return fig


def fig_independent_vs_exclusive():
    """Section 3: independence as a rectangle crossing (self-assessment 1) against two disjoint dice events."""
    pA, pB, pU = 0.5, 0.4, 0.7                                  # self-assessment question 1
    pAB = check(pA + pB - pU, 0.2, 1e-12)                       # inclusion-exclusion
    check(pAB - pA * pB, 0.0, 1e-12)                            # = product: independent
    check(pAB / pB, pA, 1e-12)                                  # P(A | B) = P(A)
    qA = qB = Fraction(6, 36)                                   # Section 2: sum is 7, doubles
    check(qA * qB, 0.028, 5e-4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))

    # (a) A = left half, B = bottom strip; their overlap is exactly the product
    ax1.add_patch(Rectangle((0, 0), pA, 1, fc="C0", alpha=0.3, lw=0))
    ax1.add_patch(Rectangle((0, 0), 1, pB, fc="C1", alpha=0.3, lw=0))
    ax1.add_patch(Rectangle((0, 0), pA, pB, fc="none", ec="C3", lw=2.5, zorder=3, clip_on=False))
    boxed(ax1, (pA / 2, 0.82), f"$A$: the left half,\n$P(A) = {pA:g}$", color="C0")
    boxed(ax1, ((1 + pA) / 2, pB / 2), f"$B$: the bottom strip,\n$P(B) = {pB:g}$; inside $B$,\n$A$ still fills half:\n"
          f"$P(A \\mid B) = {pAB:g}/{pB:g}$\n$= {pAB / pB:g} = P(A)$", color=DARK_ORANGE, fontsize=9.5)
    boxed(ax1, (pA / 2, pB / 2), f"$A \\cap B$:\n${pA:g} + {pB:g} - {pU:g} = {pAB:g}$\n$= {pA:g} \\times {pB:g}$",
          color="C3", fontsize=9.5)
    label(ax1, ((1 + pA) / 2, (1 + pB) / 2), f"neither: {1 - pU:.1f}", color=GREY_TEXT, offset=(0, 0), ha="center",
          va="center", fontsize=10)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect("equal"); ax1.grid(False)
    ax1.set_title(f"(a) Independent: $P(A) = {pA:g}$, $P(B) = {pB:g}$,\noverlap ${pAB:g}$ = the product")
    ax1.set_xlabel("sample space as a unit square: area = probability"); ax1.set_ylabel("sample space (height)")

    # (b) two disjoint strips: sum is 7 and doubles
    ax2.add_patch(Rectangle((0, 0), float(qA), 1, fc="C0", alpha=0.45, lw=0))
    ax2.add_patch(Rectangle((float(qA), 0), float(qB), 1, fc="C1", alpha=0.45, lw=0))
    ax2.text(float(qA) / 2, 0.5, "$A$: sum is 7, $P(A) = 1/6$", rotation=90, ha="center", va="center", fontsize=10)
    ax2.text(float(qA) + float(qB) / 2, 0.5, "$B$: doubles, $P(B) = 1/6$", rotation=90, ha="center", va="center",
             fontsize=10)
    x0 = float(qA + qB) + 0.05
    ax2.text(x0, 0.86, "$A \\cap B = \\emptyset$: $P(A \\cap B) = 0$,\n"
             f"but $P(A)\\,P(B) = 1/36 \\approx {float(qA * qB):.3f}$", ha="left", va="center", fontsize=9.5,
             color="C3")
    ax2.text(x0, 0.5, "inside $B$ there is no $A$:\n$P(A \\mid B) = 0$, not $P(A) = 1/6$;\n"
             "knowing $B$ rules $A$ out,\nso $A$ and $B$ are dependent", ha="left", va="center",
             fontsize=9.5)
    ax2.text(x0, 0.14, "the rest of the square:\nneither event", ha="left", va="center", fontsize=9.5, color=GREY_TEXT)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect("equal"); ax2.grid(False)
    ax2.set_title("(b) Mutually exclusive: sum is 7 and doubles,\neach $1/6$, never together")
    ax2.set_xlabel("sample space as a unit square: area = probability"); ax2.set_ylabel("sample space (height)")
    fig.tight_layout()
    return fig


def fig_bayes_frequency_tree():
    """Section 4: 10,000 people through the diagnostic test; the odds form as equal steps of log-odds."""
    se, sp, prev, N = 0.99, 0.95, 0.01, 10_000
    ill = prev * N
    healthy = N - ill
    tp, fn = se * ill, (1 - se) * ill
    fp, tn = (1 - sp) * healthy, sp * healthy
    for got, guide in [(ill, 100), (healthy, 9900), (tp, 99), (fn, 1), (fp, 495), (tn, 9405), (tp + fp, 594)]:
        check(got, guide, 1e-9)
    p_pos = check(se * prev + (1 - sp) * (1 - prev), 0.0594, 1e-12)
    check(tp / (tp + fp), 1 / 6, 1e-12)                      # PPV from the counts
    check(se * prev / p_pos, 1 / 6, 1e-12)
    check(tn / (tn + fn), 0.9999, 5e-5)
    LR = check(se / (1 - sp), 19.8, 1e-12)
    odds = [prev / (1 - prev)]
    odds += [odds[-1] * LR, odds[-1] * LR * LR]
    check(odds[1], 0.2, 1e-12); check(odds[2], 3.96, 1e-12)
    probs = [o / (1 + o) for o in odds]
    check(probs[2], 0.80, 5e-3)
    prev50 = check((1 - sp) / (se + 1 - sp), 0.048, 5e-4)       # PPV = 0.5 here
    fmt = lambda v: f"{v:,.0f}"                                 # noqa: E731

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) the natural-frequency tree, then the positives collected into one bar
    root, nD, nH = (0.0, 0.5), (1.0, 0.77), (1.0, 0.23)
    lv = {"tp": (2.0, 0.93), "fn": (2.0, 0.65), "fp": (2.0, 0.37), "tn": (2.0, 0.07)}
    branch(ax1, root, nD, "C3", f"{prev:.0%}")
    branch(ax1, root, nH, "C0", f"{1 - prev:.0%}")
    branch(ax1, nD, lv["tp"], "C3", f"{se:.2f}")
    branch(ax1, nD, lv["fn"], GREY, f"{1 - se:.2f}")
    branch(ax1, nH, lv["fp"], "C0", f"{1 - sp:.2f}")
    branch(ax1, nH, lv["tn"], GREY, f"{sp:.2f}")
    node(ax1, root, f"{fmt(N)}\npeople", GREY)
    node(ax1, nD, f"{fmt(ill)}\ndiseased", "C3", lw=2)
    node(ax1, nH, f"{fmt(healthy)}\nhealthy", "C0", lw=2)
    node(ax1, lv["tp"], f"{fmt(tp)} test +", "C3", lw=2)
    node(ax1, lv["fn"], f"{fmt(fn)} test −", GREY, lw=1)
    node(ax1, lv["fp"], f"{fmt(fp)} test +", "C0", lw=2)
    node(ax1, lv["tn"], f"{fmt(tn)} test −", GREY, lw=1)
    boxed(ax1, (1.62, 1.0), f"sensitivity $P(+ \\mid D) = {fmt(tp)}/{fmt(ill)} = {se:.2f}$", color="C3",
          ha="right", fontsize=9.5)
    boxed(ax1, (1.62, -0.03), f"specificity $P(- \\mid D^c) = {sp:.2f}$", color=GREY_TEXT, ha="right", fontsize=9.5)
    # the stacked bar of all positives, heights in proportion 99 : 495
    xb, wb, y0, y1 = 3.05, 0.34, 0.12, 0.88
    ytp = y1 - (y1 - y0) * tp / (tp + fp)
    ax1.add_patch(Rectangle((xb - wb / 2, ytp), wb, y1 - ytp, fc="C3", alpha=0.75, lw=0))
    ax1.add_patch(Rectangle((xb - wb / 2, y0), wb, ytp - y0, fc="C0", alpha=0.55, lw=0))
    ax1.text(xb, (y1 + ytp) / 2, fmt(tp), ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
    ax1.text(xb, (ytp + y0) / 2, fmt(fp), ha="center", va="center", fontsize=9.5)
    ax1.annotate("", xy=(xb - wb / 2, (y1 + ytp) / 2), xytext=(2.42, lv["tp"][1]),
                 arrowprops=dict(arrowstyle="-|>", color="C3", lw=1, ls="--", shrinkA=2, shrinkB=2))
    ax1.annotate("", xy=(xb - wb / 2, lv["fp"][1]), xytext=(2.42, lv["fp"][1]),
                 arrowprops=dict(arrowstyle="-|>", color="C0", lw=1, ls="--", shrinkA=2, shrinkB=2))
    ax1.text(xb, y1 + 0.03, f"all {fmt(tp + fp)}\npositives", ha="center", va="bottom", fontsize=9.5)
    ax1.text(xb, y0 - 0.03, f"$P(D \\mid {{+}}) =$\n${fmt(tp)}/{fmt(tp + fp)} = 1/6$", ha="center", va="top",
             fontsize=9.5, color="C3")
    ax1.set_xlim(-0.35, 3.45); ax1.set_ylim(-0.09, 1.08)
    bare(ax1, "stage of the count (expected numbers of people)", "branches",
         [0, 1, 2, xb], ["everyone", "disease", "test result", "positives"])
    ax1.set_title(f"(a) {fmt(N)} people, prevalence {prev:.0%}:\nexpected counts on each branch")

    # (b) log-odds ladder: every positive test adds ln(19.8)
    z = [logit(p) for p in probs]
    step = np.log(LR)
    check(step, 2.99, 5e-3)
    ax2.axhline(0, color="k", ls="--", lw=1)
    boxed(ax2, (-0.35, 0), "$p = 0.5$", ha="left", va="center", fontsize=9)
    ax2.axhline(-step, color="C3", ls=":", lw=1.4)
    label(ax2, (2.55, -step), f"prior needed for one\npositive to reach 50%:\nprevalence {prev50:.1%}",
          color="C3", offset=(0, 4), ha="right", va="bottom", fontsize=9)
    for k, off, ha, va in [(0, (14, -8), "left", "top"), (1, (-12, 12), "right", "bottom")]:  # into the empty corners
        ax2.annotate("", xy=(k + 1, z[k + 1]), xytext=(k, z[k]),
                     arrowprops=dict(arrowstyle="-|>", color="C1", lw=1.8, shrinkA=5, shrinkB=5))
        boxed(ax2, (k + 0.5, (z[k] + z[k + 1]) / 2), f"$\\times {LR:g}$ in odds,\n$+{step:.2f}$ in log-odds",
              color=DARK_ORANGE, offset=off, ha=ha, va=va, fontsize=10)   # text darker than the C1 arrows
    ax2.plot(range(3), z, "o", color="k", ms=7, zorder=5)
    for k, txt, off, ha in [(0, f"prior $p = {probs[0]:g}$\nodds 1 : 99", (8, 0), "left"),
                            (1, f"$p = 1/6$\nodds {odds[1]:.1f}", (-9, 4), "right"),
                            (2, f"$p = {probs[2]:.2f}$\nodds {odds[2]:.2f}", (-9, 4), "right")]:
        label(ax2, (k, z[k]), txt, offset=off, ha=ha, va="center" if k == 0 else "bottom", fontsize=9.5)
    sec = ax2.secondary_yaxis("right", functions=(expit, logit))
    sec.set_yticks([0.01, prev50, 1 / 6, 0.5, 0.8, 0.95])
    sec.set_yticklabels(["0.01", f"{prev50:.3f}", "0.167", "0.5", "0.8", "0.95"])
    sec.set_ylabel("probability of disease")
    ax2.set_xlim(-0.4, 2.6); ax2.set_ylim(-5.3, 3.3)
    ax2.set_xticks([0, 1, 2])
    ax2.set_title(f"(b) Odds form: each positive test\nadds $\\ln {LR:g} \\approx {step:.2f}$ to the log-odds")
    ax2.set_xlabel("number of positive, conditionally independent tests"); ax2.set_ylabel("log-odds of disease")
    fig.tight_layout()
    return fig


def fig_change_of_variables():
    """Section 5: Y = X^2 for uniform X — squeezed intervals, the density 1/(2 sqrt y), the CDF sqrt(y)."""
    xe = np.linspace(0, 1, 11)                                  # ten x-intervals, each of probability 0.1
    ye = xe ** 2                                                # their images under g(x) = x^2
    wy = np.diff(ye)
    hy = 0.1 / wy                                               # average density of Y over each image
    fY = lambda y: 1 / (2 * np.sqrt(y))                         # noqa: E731
    FY = np.sqrt
    check(fY(0.01), 5.0, 1e-12)                                 # a density above 1
    check(FY(1.0) - FY(0.0), 1.0, 1e-12)                        # integrates to 1
    med = check(0.5 ** 2, 0.25, 1e-12)                          # Q_Y(0.5)
    check(FY(med), 0.5, 1e-12)
    check(wy[1], 0.03, 1e-12); check(wy[-1], 0.19, 1e-12)
    HI, LO = 1, 9                                               # the squeezed and the stretched interval

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.0))

    # (a) the map, with each x-interval carried to its y-interval
    rug = (-0.085, -0.035)                                      # strips just outside the unit square
    for k in range(10):
        fc = "C1" if k % 2 == 0 else "white"
        ax1.add_patch(Rectangle((xe[k], rug[0]), 0.1, rug[1] - rug[0], fc=fc, alpha=0.5, ec=GREY, lw=0.4))
        ax1.add_patch(Rectangle((rug[0], ye[k]), rug[1] - rug[0], wy[k], fc=fc, alpha=0.5, ec=GREY, lw=0.4))
    for k, col, al in [(HI, "C3", 0.4), (LO, "C2", 0.18)]:      # the wide ribbon is paler, so it does not dominate
        a, b = xe[k], xe[k + 1]
        ax1.add_patch(Polygon([(a, rug[0]), (b, rug[0]), (b, b * b), (rug[0], b * b), (rug[0], a * a), (a, a * a)],
                              closed=True, fc=col, alpha=al, lw=0))
    for x in xe:
        ax1.plot([x, x], [rug[1], x * x], color=GREY, lw=0.6)
        ax1.plot([rug[1], x], [x * x, x * x], color=GREY, lw=0.6)
    xs = np.linspace(0, 1, 201)
    ax1.plot(xs, xs ** 2, color="C0", lw=2, zorder=7)            # above the label cards, so none can hide it
    ax1.annotate(f"$[0.1, 0.2] \\to [0.01, 0.04]$:\nwidth {wy[HI]:.2f}, squeezed", (0.15, 0.03), xytext=(0.02, 0.47),
                 color="C3", fontsize=9, ha="left", va="center", bbox=WHITE_BOX, zorder=6,
                 arrowprops=dict(arrowstyle="-", color="C3", lw=0.8, shrinkB=2))
    boxed(ax1, (0.02, 0.70), f"$[0.9, 1] \\to [0.81, 1]$:\nwidth {wy[LO]:.2f}, stretched", color=DARK_GREEN,
          ha="left", fontsize=9)
    ax1.set_xlim(-0.1, 1.02); ax1.set_ylim(-0.1, 1.02); ax1.set_aspect("equal"); ax1.grid(False)
    ax1.set_xticks([0, 0.5, 1]); ax1.set_yticks([0, 0.5, 1])
    ax1.set_title("(a) $Y = X^2$: ten $x$-strips of\nprobability 0.1 each, mapped")
    ax1.set_xlabel("$x$ (uniform on $[0, 1]$)"); ax1.set_ylabel("$y = x^2$")

    # (b) equal areas over unequal widths: the density of Y
    for k in range(10):
        fc, al = {HI: ("C3", 0.55), LO: ("C2", 0.55)}.get(k, ("C1", 0.3))
        ax2.bar(ye[k], hy[k], width=wy[k], align="edge", fc=fc, alpha=al, ec="C1", lw=0.6)
    yy = np.linspace(0.004, 1, 400)
    ax2.plot(yy, fY(yy), color="C0", lw=2, label="$f_Y(y) = 1/(2\\sqrt{y})$")
    ax2.plot(0.01, fY(0.01), "ko", ms=6, zorder=5)
    label(ax2, (0.01, fY(0.01)), "$f_Y(0.01) = 5$:\na density, not\na probability", offset=(10, -2),
          va="top", fontsize=10)
    ax2.annotate(f"first bar, $[0, 0.01]$:\nheight {hy[0]:.0f}, off scale", (0.005, 5.9), xytext=(0.07, 5.85), ha="left", va="top",
                 fontsize=9, arrowprops=dict(arrowstyle="-", color="k", lw=0.8, shrinkB=0))
    ax2.annotate(f"$0.1/{wy[HI]:.2f} = {hy[HI]:.2f}$", (ye[HI] + wy[HI] / 2, hy[HI]), xytext=(0.3, 3.0), color="C3",
                 fontsize=10.5, ha="left", va="center", arrowprops=dict(arrowstyle="-", color="C3", lw=0.8, shrinkB=2))
    label(ax2, (ye[LO] + wy[LO] / 2, hy[LO]), f"$0.1/{wy[LO]:.2f}$\n$= {hy[LO]:.2f}$", color=DARK_GREEN, offset=(-4, 6),
          ha="center", fontsize=10.5)
    boxed(ax2, (0.55, 1.9), "every bar has area 0.1", fontsize=9)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 6)
    ax2.legend(loc="upper right", bbox_to_anchor=(1, 0.86), fontsize=9)
    ax2.set_box_aspect(1)
    ax2.set_title("(b) Density of $Y$: equal areas,\nso squeezed bins are tall")
    ax2.set_xlabel("$y$"); ax2.set_ylabel("density of $Y$ (bars: bin averages)")

    # (c) the CDF and reading a quantile off it
    ax3.plot(xs, FY(xs), color="C0", lw=2, label="$F_Y(y) = \\sqrt{y}$")
    ax3.plot(xs, xs, color=GREY, ls="--", lw=1.4, label="uniform: $F_X(x) = x$")
    ax3.plot([0, med, med], [0.5, 0.5, 0], ":", color="C3", lw=1.4)
    ax3.plot(med, 0.5, "o", color="C3", ms=6, zorder=5)
    label(ax3, (med, 0), f"$Q_Y(0.5) = 0.5^2 = {med:g}$:\nthe median of $Y$", color="C3", offset=(5, 5), fontsize=10)
    ax3.plot(0.5, 0.5, "o", color=GREY, ms=5, zorder=5)
    label(ax3, (0.5, 0.5), "uniform's\nmedian 0.5", color=GREY_TEXT, offset=(6, -3), va="top", fontsize=9)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.02); ax3.set_aspect("equal")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_title("(c) CDF $F_Y(y) = \\sqrt{y}$: invert it\nto read off quantiles")
    ax3.set_xlabel("$y$"); ax3.set_ylabel("$F(y) = P(Y \\leq y)$")
    fig.tight_layout()
    return fig


def fig_expectation_balance():
    """Section 6: E[X] = 3.5 as the balance point of a die's PMF; E[X^2] = 15.17 against (E[X])^2 = 12.25."""
    x = np.arange(1, 7)
    p = np.full(6, 1 / 6)
    EX = check(x @ p, 3.5, 1e-12)
    EX2 = check(x ** 2 @ p, 91 / 6, 1e-12)
    check(EX2, 15.17, 5e-3)
    check(EX ** 2, 12.25, 1e-12)
    var = check(EX2 - EX ** 2, 35 / 12, 1e-12)
    check((x - EX) @ p, 0.0, 1e-12)                             # deviations balance

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))

    # (a) six weights of 1/6 on a lever; the fulcrum at the mean
    ax1.bar(x, p, width=0.55, color="C0", alpha=0.75)
    ax1.plot([0.5, 6.5], [0, 0], color="k", lw=3, solid_capstyle="butt", zorder=3)
    ax1.add_patch(Polygon([(EX, 0), (EX - 0.28, -0.04), (EX + 0.28, -0.04)], closed=True, fc="C3", ec="C3", zorder=4))
    for xi in x:
        label(ax1, (xi, 1 / 6), f"${xi - EX:+.1f}$".replace("+", "+\\,").replace("-", "-\\,"), offset=(0, 3),
              ha="center", fontsize=9.5)
    boxed(ax1, (3.5, 0.215), "deviations from 3.5, each with weight 1/6: they sum to 0", fontsize=9)
    label(ax1, (EX, -0.04), f"balance point $\\mathbb{{E}}[X] = {EX:g}$ (no face shows it)", color="C3",
          offset=(0, -3), ha="center", va="top", fontsize=9.5)
    ax1.set_xlim(0.4, 6.6); ax1.set_ylim(-0.085, 0.24)
    ax1.set_xticks(x); ax1.set_yticks([0, 1 / 6]); ax1.set_yticklabels(["0", "1/6"])
    ax1.grid(axis="x", visible=False)
    ax1.set_title("(a) $\\mathbb{E}[X]$ is the centre of mass of the PMF")
    ax1.set_xlabel("face $x$"); ax1.set_ylabel("$P(X = x)$")

    # (b) the six heights x^2, their mean, and the height at the mean
    xs = np.linspace(0.5, 6.5, 300)
    ax2.plot(xs, xs ** 2, color="C0", lw=1.6, label="$g(x) = x^2$")
    ax2.plot(x, x ** 2, "o", color="C0", ms=7, zorder=5, label="the six equally likely heights $x^2$")
    ax2.axhline(EX2, color="C0", ls="--", lw=1.4)
    boxed(ax2, (0.6, EX2), f"mean of the six heights:\n$\\mathbb{{E}}[X^2] = 91/6 \\approx {EX2:.2f}$", color="C0",
          offset=(0, 4), ha="left", va="bottom", fontsize=9.5)
    ax2.plot([EX, EX], [0, EX ** 2], ":", color=GREY, lw=1.2)
    ax2.plot(EX, EX ** 2, "o", color="C3", ms=8, zorder=6)
    label(ax2, (EX, EX ** 2), f"height at the mean:\n$(\\mathbb{{E}}[X])^2 = 3.5^2 = {EX ** 2:g}$", color="C3",
          offset=(10, -6), va="top", fontsize=9.5)
    ax2.annotate("", xy=(EX - 0.12, EX2), xytext=(EX - 0.12, EX ** 2),
                 arrowprops=dict(arrowstyle="<->", color="k", lw=1.1, shrinkA=0, shrinkB=0))
    boxed(ax2, (EX - 0.12, (EX ** 2 + EX2) / 2), f"gap $35/12 \\approx {var:.2f} = \\mathrm{{Var}}(X)$",
          offset=(-6, 0), ha="right", va="center", fontsize=9.5)
    ax2.set_xlim(0.5, 6.5); ax2.set_ylim(0, 39)
    ax2.set_xticks(x)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title("(b) $\\mathbb{E}[X^2] \\neq (\\mathbb{E}[X])^2$: the average height lies\n"
                  "above the convex curve's height at the average")
    ax2.set_xlabel("face $x$"); ax2.set_ylabel("$g(x) = x^2$")
    fig.tight_layout()
    return fig


def fig_variance_and_averages():
    """Section 7: same mean, different spread (independent dice vs Y = 7 - X); averages narrow like sigma/sqrt(n)."""
    die = np.full(6, 1 / 6)
    faces = np.arange(1, 7)
    var1 = check(faces ** 2 @ die - (faces @ die) ** 2, 35 / 12, 1e-12)
    sd1 = check(np.sqrt(var1), 1.708, 5e-4)
    pS = np.convolve(die, die)                                  # sums 2..12
    s = np.arange(2, 13)
    for k in (2, 7, 12):
        check(pS[k - 2], (6 - abs(k - 7)) / 36, 1e-12)
    varS = check(s ** 2 @ pS - (s @ pS) ** 2, 35 / 6, 1e-12)
    check(np.sqrt(varS), 2.415, 5e-4)
    cov = check((faces - 3.5) @ ((7 - faces - 3.5) * die), -2.917, 5e-4)   # Cov(X, 7 - X)
    check(var1 + var1 + 2 * cov, 0.0, 1e-12)

    def average_pmf(n):
        p = np.array([1.0])
        for _ in range(n):
            p = np.convolve(p, die)
        return np.arange(n, 6 * n + 1) / n, p                   # values of the average, exact probabilities

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))

    # (a) the two sums
    ax1.bar(s, pS, width=0.8, color="C0", alpha=0.7,
            label=f"independent: sd {np.sqrt(varS):.3f}\n(Var$= {var1:.3f} + {var1:.3f}$)")
    ax1.bar(7, 1.0, width=0.28, color="C3", alpha=0.9, hatch="//", ec="white", lw=0,
            label=f"$Y = 7 - X$: sd 0\n(Cov$= {cov:.3f}$ cancels)")
    label(ax1, (7, 0.35), "$P(X + Y = 7) = 1$\n(off scale)", color="C3", offset=(8, -4), va="top", fontsize=9.5)
    ax1.plot(7, 0, "^", color="k", ms=11, clip_on=False, zorder=6,
             label="both means $3.5 + 3.5 = 7$\n(linearity)")
    ax1.set_xlim(1.3, 12.7); ax1.set_ylim(0, 0.35)
    ax1.set_xticks(s)
    h, lab = ax1.get_legend_handles_labels()                    # the two series first, then their common mean
    ax1.legend(h[1:] + h[:1], lab[1:] + lab[:1], loc="upper left", fontsize=9, labelspacing=0.8, handlelength=1.4)
    ax1.grid(axis="x", visible=False)
    ax1.set_title("(a) Same mean by linearity; the spread\ndepends on the dependence")
    ax1.set_xlabel("sum of the two dice"); ax1.set_ylabel("probability")

    # (b) exact distributions of the average of n rolls, as densities (probability / spacing 1/n)
    sds = {}
    for n, col in [(1, "C0"), (10, "C1"), (100, "C2")]:
        v, p = average_pmf(n)
        dens = p * n
        sds[n] = check(np.sqrt(((v - 3.5) ** 2) @ p), sd1 / np.sqrt(n), 1e-9)
        lab = f"$n = {n}$: sd {sds[n]:.3f}"
        if n == 1:
            ax2.bar(v, dens, width=1.0, color=col, alpha=0.45, ec="white", lw=1.2, label=lab)  # tiles [0.5, 6.5]: area 1
        elif n == 10:
            ax2.step(v, dens, where="mid", color=col, lw=1.6, label=lab)
        else:
            ax2.plot(v, dens, color=col, lw=2, label=lab)
    check(average_pmf(100)[1].max() * 100, 2.33, 5e-3)
    v, p = average_pmf(100)
    exact = p[np.abs(v - 3.5) >= 0.5 - 1e-9].sum()              # P(|mean - 3.5| >= 0.5)
    cheb = check((sds[100] / 0.5) ** 2, 0.117, 5e-4)
    for xv in (3.0, 4.0):
        ax2.axvline(xv, color="k", ls="--", lw=1)
    boxed(ax2, (0.6, 2.6), f"$n = 100$: on or outside\nthe dashed lines $3.5 \\pm 0.5$\nwith exact probability\n"
          f"{exact:.4f}; Chebyshev's\nbound is only $\\leq {cheb:.3f}$", offset=(0, 0), ha="left", va="top", fontsize=9)
    ax2.set_xlim(0.5, 6.5); ax2.set_ylim(0, 2.75)
    h, lab = ax2.get_legend_handles_labels()                    # lines come before bars: put n = 1 first again
    ax2.legend(h[-1:] + h[:-1], lab[-1:] + lab[:-1], loc="upper right", fontsize=9)
    ax2.set_title(f"(b) The average of $n$ rolls narrows\nlike $\\sigma/\\sqrt{{n}} = {sd1:.3f}/\\sqrt{{n}}$")
    ax2.set_xlabel("average of $n$ rolls"); ax2.set_ylabel("probability per unit of the average")
    fig.tight_layout()
    return fig


def fig_naive_bayes_evidence():
    """Section 9: per-word log-likelihood ratios of the six-message corpus; 'free meeting now' as a sum of evidence."""
    spam = ["free money now", "free offer now", "win money"]
    ham = ["team meeting now", "project meeting", "lunch meeting"]
    vocab = ["free", "money", "now", "offer", "win", "team", "meeting", "project", "lunch"]
    count = lambda docs: {w: sum(d.split().count(w) for d in docs) for w in vocab}  # noqa: E731
    cs, ch = count(spam), count(ham)
    ns, nh, V = sum(cs.values()), sum(ch.values()), len(vocab)
    assert (ns, nh, V) == (8, 7, 9)
    Ps = {w: Fraction(cs[w] + 1, ns + V) for w in vocab}        # add-one smoothing
    Ph = {w: Fraction(ch[w] + 1, nh + V) for w in vocab}
    assert Ps["free"] == Fraction(3, 17) and Ps["meeting"] == Fraction(1, 17) and Ph["free"] == Fraction(1, 16)
    assert Ph["now"] == Fraction(2, 16) and Ph["meeting"] == Fraction(4, 16)
    llr = {w: np.log(float(Ps[w])) - np.log(float(Ph[w])) for w in vocab}
    msg = ["free", "meeting", "now"]
    prior = Fraction(len(spam), 6)
    score_s = prior * Ps["free"] * Ps["meeting"] * Ps["now"]
    score_h = (1 - prior) * Ph["free"] * Ph["meeting"] * Ph["now"]
    assert score_s == Fraction(9, 9826) and score_h == Fraction(8, 8192)
    check(np.log(float(score_s)), -6.996, 5e-4); check(np.log(float(score_h)), -6.931, 5e-4)
    p_ham = check(score_h / (score_s + score_h), 0.52, 5e-3)
    prior_llr = np.log(float(prior)) - np.log(float(1 - prior))  # 0: equal priors
    total = prior_llr + sum(llr[w] for w in msg)
    check(total, np.log(float(score_s)) - np.log(float(score_h)), 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) evidence per word; hatched bars exist only thanks to the add-one count
    order = sorted(vocab, key=lambda w: (-llr[w], vocab.index(w)))
    ypos = np.arange(len(order))[::-1]
    for y, w in zip(ypos, order):
        zero = cs[w] == 0 or ch[w] == 0
        ax1.barh(y, llr[w], height=0.65, color="C3" if llr[w] > 0 else "C0", alpha=0.7,
                 hatch="///" if zero else None, ec="white", lw=0)
        pos = llr[w] > 0                                         # value at the bar's end, counts across the axis
        label(ax1, (llr[w], y), f"{llr[w]:+.2f}".replace("-", "−"), offset=(4 if pos else -4, 0),
              ha="left" if pos else "right", va="center", fontsize=9)
        label(ax1, (0, y), f"spam {cs[w]}, ham {ch[w]}", color="0.35", offset=(-5 if pos else 5, 0),
              ha="right" if pos else "left", va="center", fontsize=9)
    ax1.axvline(0, color="k", lw=1)
    ax1.set_yticks(ypos); ax1.set_yticklabels(order)
    ax1.set_xlim(-2.15, 1.75); ax1.set_ylim(-1.9, len(order) - 0.4)
    ax1.legend(handles=[Patch(fc="0.6", hatch="///", ec="white", lw=0,
                              label="count 0 in one class: without add-one\nsmoothing this bar would be infinite")],
               loc="lower left", fontsize=9, handlelength=2.2)
    ax1.grid(axis="y", visible=False)
    ax1.set_title("(a) Evidence per word: $\\ln P(w \\mid \\mathrm{spam}) - \\ln P(w \\mid \\mathrm{ham})$\n"
                  f"(add-one smoothed, denominators {ns + V} and {nh + V})")
    ax1.set_xlabel("log-likelihood ratio (positive = towards spam)"); ax1.set_ylabel("word")

    # (b) the waterfall for "free meeting now"
    steps = [("prior", prior_llr)] + [(w, llr[w]) for w in msg]
    names = [n for n, _ in steps] + ["total"]
    yv = np.arange(len(names))[::-1]
    run = 0.0
    for k, (name, d) in enumerate(steps):
        new = run + d
        if d != 0:
            ax2.barh(yv[k], d, left=run, height=0.55, color="C3" if d > 0 else "C0", alpha=0.7)
        else:
            ax2.plot(run, yv[k], "D", color="k", ms=6, zorder=5)
            label(ax2, (run, yv[k]), "equal priors:\n$\\ln(1/2) - \\ln(1/2) = 0$", offset=(8, 0), va="center",
                  fontsize=9)
        ax2.plot([new, new], [yv[k] - 0.28, yv[k + 1] + 0.28], "--", color=GREY, lw=1)
        run = new
    ax2.barh(yv[-1], total, height=0.55, color="C0", alpha=0.9)
    label(ax2, (total, yv[-1]), f"total ${total:.3f} < 0$:\nham wins,\n$P(\\mathrm{{ham}} \\mid w) = {float(p_ham):.3f}$",
          color="C0", offset=(-6, 0), ha="right", va="center", fontsize=9.5)
    ax2.axvline(0, color="k", lw=1.2)
    label(ax2, (0, yv[0] + 0.45), "decision boundary", offset=(-4, 0), ha="right", va="center", fontsize=9)
    ticks = ["prior: 0"] + [f'"{w}": {llr[w]:+.3f}'.replace("-", "−") for w in msg] + [f"total: {total:.3f}".replace("-", "−")]
    ax2.set_yticks(yv); ax2.set_yticklabels(ticks)
    ax2.set_xlim(-1.9, 1.6); ax2.set_ylim(-0.7, len(names) - 0.2)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("(b) \"free meeting now\": add the evidence,\nthe sign decides")
    ax2.set_xlabel("cumulative log-odds for spam"); ax2.set_ylabel("term added")
    fig.tight_layout()
    return fig


FIGURES = {
    "counting-grid": fig_counting_grid,
    "independent-vs-exclusive": fig_independent_vs_exclusive,
    "conditioning-tree": fig_conditioning_tree,
    "bayes-frequency-tree": fig_bayes_frequency_tree,
    "change-of-variables": fig_change_of_variables,
    "expectation-balance": fig_expectation_balance,
    "variance-and-averages": fig_variance_and_averages,
    "naive-bayes-evidence": fig_naive_bayes_evidence,
}


def main():
    ap = argparse.ArgumentParser(description="Draw the Module 07 concept-guide figures.")
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
