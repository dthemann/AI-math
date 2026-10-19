#!/usr/bin/env python3
"""Builder for Module 17 — The Mathematics of Transformers.

Generates lesson.ipynb, exercises.ipynb and solutions.ipynb into
part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers. Hidden
answers for the exercise self-checks are computed here by *executing the
solution code*. Everything is NumPy (+ scipy.special.erf, sklearn probe).
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import Notebook, ExerciseSet, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers"


def lit(x, decimals=6) -> str:
    """Python literal of a rounded array, for embedding expected values in check cells."""
    return "np.array(" + repr(np.round(np.asarray(x, dtype=float), decimals).tolist()) + ")"


# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 17 — The Mathematics of Transformers")

nb.md(r"""
This notebook builds a transformer piece by piece in NumPy and checks every claim of the
concept guide numerically: embeddings as one-hot products, scaled dot-product attention and
its heat-map, the variance of $QK^\top$ with and without $\sqrt{d_k}$, the softmax Jacobian
against finite differences, multi-head attention with a parameter counter that reproduces
GPT-2 small's 124M, the causal mask, sinusoidal and rotary positions, LayerNorm and a full
pre-norm block, decoding strategies and the KV cache, LoRA as a low-rank approximation,
and the geometry of embeddings.

Read the [concept guide](README.md) first, or keep it open alongside — every section here
mirrors a numbered section there. No framework is used anywhere: each object is a few
lines of NumPy, so you can see exactly what the maths does.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True, linewidth=120)
""")

# ---------------------------------------------------------------- Section 1
nb.md(r"""
## 1. Tokens, embeddings and the residual stream

A token index $t$ becomes a vector by looking up row $t$ of the embedding matrix
$E \in \mathbb{R}^{V\times d}$ — which is the same as multiplying the one-hot vector $e_t$ by $E$:

$$x = e_t^\top E = E_{t,:}.$$

Layers then *add* to this vector (the residual stream), and at the end the tied unembedding
turns the final vector $h$ into logits $z = Eh$ — one dot product per vocabulary entry — and
a softmax turns the logits into next-token probabilities. We use the five-token vocabulary
of the guide.
""")

nb.code(r"""
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)       # shift for numerical safety (softmax is shift-invariant)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

vocab = ["the", "cat", "sat", "on", "mat"]
E = np.array([[ 1.0,  0.0, 0.5],
              [ 0.0,  1.0, 0.5],
              [-1.0,  0.0, 0.5],
              [ 0.0, -1.0, 0.5],
              [ 0.5,  0.5, 0.0]])
V, d = E.shape

tokens = np.array([0, 1, 2])                          # "the cat sat"
one_hot = np.eye(V)[tokens]                            # (n, V)
X = one_hot @ E                                        # (n, d): the initial residual stream
print("one-hot rows:\n", one_hot)
print("one_hot @ E equals E[tokens]:", np.allclose(X, E[tokens]))
print("residual stream X (one row per position):\n", X)

# the gradient of a lookup: only the rows that were used receive anything
dX = np.ones_like(X)                                   # pretend every output coordinate wants to grow
dE = one_hot.T @ dX
print("rows of E that receive a gradient:", np.flatnonzero(np.abs(dE).sum(axis=1)), "(tokens used:", tokens, ")")

# tied unembedding: logits are dot products with every row of E
h = np.array([0.8, 0.2, 0.5])                          # a final residual-stream vector
z = E @ h
p = softmax(z)
for tok, zz, pp in zip(vocab, z, p):
    print(f"  {tok:4s}  logit {zz:6.2f}   prob {pp:.3f}")
print("probabilities sum to", p.sum(), "; predicted next token:", vocab[int(np.argmax(p))])
""")

nb.md(r"""
The lookup *is* a matrix product, which is why its gradient is so sparse: `dE` is zero in
the rows of tokens that did not occur. The logits reproduce the guide's hand calculation
$(1.05, 0.45, -0.55, 0.05, 0.5)$ and the model predicts "the" with probability 0.371, because
$h$ points most nearly along the first row of $E$. For GPT-2 small the same two objects are
$E \in \mathbb{R}^{50257\times768}$ and $h \in \mathbb{R}^{768}$.
""")

# ---------------------------------------------------------------- Section 2
nb.md(r"""
## 2. Attention as a soft lookup

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V ,$$

with the softmax taken **row-wise**, so each row of $P$ is a probability distribution over
the key positions and each output row is a convex combination of value rows. We first
reproduce the guide's three-token example by hand, then run a four-token example with
learned-looking projections and draw the attention weights.
""")

nb.code(r"""
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)                      # scores (n_q, n_k)
    if mask is not None:
        S = np.where(mask, -np.inf, S)              # True = forbidden
    P = softmax(S, axis=-1)                         # one distribution per query
    return P @ V, P

# --- the guide's example: 3 tokens, d_k = 2
Q = K = np.array([[1., 0.], [0., 1.], [1., 1.]])
Vv = np.array([[2., 0.], [0., 2.], [1., 1.]])
O, P = attention(Q, K, Vv)
print("scaled scores:\n", Q @ K.T / np.sqrt(2))
print("attention weights P (rows sum to one):\n", P, "\nrow sums:", P.sum(axis=1))
print("output O = P V:\n", O)

# --- uniform scores give the plain mean of the values
O_uni, P_uni = attention(np.zeros((3, 2)), K, Vv)
print("\nzero queries -> uniform weights:", P_uni[0], " output = mean of V:", np.allclose(O_uni, Vv.mean(axis=0)))

# --- a 4-token example with projections
n, d_model, d_k = 4, 8, 4
X4 = rng.normal(size=(n, d_model))
W_Q, W_K, W_V = (rng.normal(size=(d_model, d_k)) / np.sqrt(d_model) for _ in range(3))
Q4, K4, V4 = X4 @ W_Q, X4 @ W_K, X4 @ W_V
O4, P4 = attention(Q4, K4, V4)
print("\nshapes: X", X4.shape, " Q,K,V", Q4.shape, " P", P4.shape, " O", O4.shape)
print("P4:\n", P4)

fig, ax = plt.subplots(figsize=(4.8, 4))
im = ax.imshow(P4, cmap="Blues", vmin=0, vmax=1)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{P4[i, j]:.2f}", ha="center", va="center", color="black" if P4[i, j] < 0.6 else "white")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xlabel("key position j"); ax.set_ylabel("query position i")
ax.set_title("Attention weights P (each row sums to 1)"); plt.colorbar(im, ax=ax, fraction=0.046); plt.show()
""")

nb.md(r"""
Row 1 of the hand example is $(0.401, 0.198, 0.401)$ and its output $(1.203, 0.797)$ — a
blend of the first and third values. Row 3 attends to everything about equally and returns
the centroid $(1, 1)$. In the heat-map each row is one query's distribution over keys; the
mixing weights change with the input, which is what makes attention a *data-dependent*
linear combination rather than a fixed layer.

**Try it yourself:** multiply `Q4` by 5 before calling `attention` (as if the scores were not
scaled). Watch the rows of `P4` turn into near one-hot vectors.
""")

# ---------------------------------------------------------------- Section 3
nb.md(r"""
## 3. Why divide by $\sqrt{d_k}$

If the components of $q$ and $k$ are independent with mean 0 and variance 1, then
$\operatorname{Var}(q^\top k) = \sum_{i=1}^{d_k}\operatorname{Var}(q_ik_i) = d_k$, so the raw
scores have standard deviation $\sqrt{d_k}$ and $q^\top k/\sqrt{d_k}$ has variance exactly 1.
We measure this, and we measure what it does to the softmax: the mean entropy of attention
rows for 16 keys, with and without the scaling.
""")

nb.code(r"""
def entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=axis)

ds = [4, 16, 64, 256]
n_keys, n_trials, n_samp = 16, 200, 20000
var_raw, var_scaled, H_raw, H_scaled = [], [], [], []
for dk in ds:
    q = rng.normal(size=(n_samp, dk)); k = rng.normal(size=(n_samp, dk))
    s = (q * k).sum(axis=1)
    var_raw.append(s.var()); var_scaled.append((s / np.sqrt(dk)).var())
    Hr, Hs = [], []
    for _ in range(n_trials):
        Qt = rng.normal(size=(n_keys, dk)); Kt = rng.normal(size=(n_keys, dk))
        S = Qt @ Kt.T
        Hr.append(entropy(softmax(S)).mean()); Hs.append(entropy(softmax(S / np.sqrt(dk))).mean())
    H_raw.append(np.mean(Hr)); H_scaled.append(np.mean(Hs))
    print(f"d_k = {dk:3d}: Var(q.k) = {var_raw[-1]:7.2f} (theory {dk}),  Var(q.k/sqrt(d_k)) = {var_scaled[-1]:.3f}"
          f"   | mean attention entropy: raw {H_raw[-1]:.2f}, scaled {H_scaled[-1]:.2f} nats (uniform = {np.log(n_keys):.2f})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].loglog(ds, var_raw, "o-", label="raw scores q.k"); axes[0].loglog(ds, var_scaled, "s-", label="scaled q.k / sqrt(d_k)")
axes[0].loglog(ds, ds, "k--", lw=0.8, label="variance = d_k")
axes[0].set_xlabel("head width d_k"); axes[0].set_ylabel("variance of a score"); axes[0].set_title("Score variance grows like d_k"); axes[0].legend()
axes[1].semilogx(ds, H_raw, "o-", label="raw scores"); axes[1].semilogx(ds, H_scaled, "s-", label="scaled scores")
axes[1].axhline(np.log(n_keys), color="gray", ls=":", label="uniform (ln 16)")
axes[1].set_xlabel("head width d_k"); axes[1].set_ylabel("mean entropy of attention rows (nats)")
axes[1].set_title("Without scaling the softmax collapses"); axes[1].legend(); plt.tight_layout(); plt.show()

print("\nsoftmax of a score 1 above its competitor:", softmax(np.array([1., 0.])), "  8 above:", softmax(np.array([8., 0.])))
""")

nb.md(r"""
The measured variance of the raw scores tracks $d_k$ to within sampling error while the scaled
variance stays at $1.0$. The consequence is on the right: with $d_k = 256$ the raw-score
attention rows have an entropy near zero — essentially one-hot — while the scaled rows keep
most of the $\ln 16 = 2.77$ nats of a uniform distribution. A one-hot softmax has a Jacobian
of almost zero (next section), so without the $\sqrt{d_k}$ the queries and keys of a wide
head would barely learn.
""")

# ---------------------------------------------------------------- Section 4
nb.md(r"""
## 4. The softmax Jacobian and attention gradients

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j), \qquad J = \operatorname{diag}(p) - pp^\top,
\qquad \frac{\partial L}{\partial z} = J^\top g = p\odot\big(g - (p^\top g)\mathbf 1\big).$$

We verify $J$ against central finite differences, watch it shrink as the softmax saturates
(and as the temperature drops), then check the full attention backward pass
$\partial V = P^\top G$, $\partial S = P\odot(GV^\top - \operatorname{rowsum}(GV^\top\odot P))$,
$\partial Q = \partial S\,K/\sqrt{d_k}$, $\partial K = \partial S^\top Q/\sqrt{d_k}$ the same way.
""")

nb.code(r"""
def softmax_jacobian(z):
    p = softmax(z)
    return np.diag(p) - np.outer(p, p)

def finite_difference_jacobian(f, z, eps=1e-6):
    J = np.zeros((f(z).size, z.size))
    for j in range(z.size):
        e = np.zeros_like(z); e[j] = eps
        J[:, j] = (f(z + e) - f(z - e)) / (2 * eps)
    return J

z = np.log(np.array([0.7, 0.2, 0.1]))               # logits whose softmax is exactly (0.7, 0.2, 0.1)
J = softmax_jacobian(z)
print("J for p = (0.7, 0.2, 0.1):\n", J)
print("row sums:", J.sum(axis=1), "  max |J - finite differences| = %.2e" % np.abs(J - finite_difference_jacobian(softmax, z)).max())
g = np.array([1., 0., 0.])
print("upstream g = (1,0,0)  ->  dL/dz = J^T g =", J.T @ g, " = p*(g - p.g) =", softmax(z) * (g - softmax(z) @ g))

# saturation: scale the logits up (equivalently, lower the temperature)
print("\nscale of logits (1/T)   p_max     largest |J| entry")
for scale in [0.5, 1, 2, 4, 8]:
    Js = softmax_jacobian(scale * z)
    print(f"{scale:16.1f}    {softmax(scale * z).max():.4f}    {np.abs(Js).max():.2e}")

# --- the attention backward pass, checked against finite differences
def attention_backward(Q, K, V, G):
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k); P = softmax(S)
    dV = P.T @ G
    dP = G @ V.T
    dS = P * (dP - (dP * P).sum(axis=1, keepdims=True))
    return dS @ K / np.sqrt(d_k), dS.T @ Q / np.sqrt(d_k), dV

Qa, Ka, Va = (rng.normal(size=(5, 3)) for _ in range(3))
G = rng.normal(size=(5, 3))                          # dL/dO for the scalar loss L = sum(G * O)
loss = lambda Q, K, V: (G * attention(Q, K, V)[0]).sum()
dQ, dK, dV = attention_backward(Qa, Ka, Va, G)
def fd_grad(f, M, eps=1e-6):
    out = np.zeros_like(M)
    for idx in np.ndindex(M.shape):
        Mp = M.copy(); Mp[idx] += eps; Mm = M.copy(); Mm[idx] -= eps
        out[idx] = (f(Mp) - f(Mm)) / (2 * eps)
    return out
print("\nattention backward vs finite differences:")
print("  dQ max error %.2e" % np.abs(dQ - fd_grad(lambda M: loss(M, Ka, Va), Qa)).max())
print("  dK max error %.2e" % np.abs(dK - fd_grad(lambda M: loss(Qa, M, Va), Ka)).max())
print("  dV max error %.2e" % np.abs(dV - fd_grad(lambda M: loss(Qa, Ka, M), Va)).max())
""")

nb.md(r"""
Every entry of $J$ matches the finite-difference estimate to about $10^{-10}$, and every row
sums to zero: adding a constant to all logits changes nothing, so the loss has no gradient
in that direction. Scaling the logits by 8 (temperature $1/8$) drives $p_{\max}$ to $0.9998$
and the largest Jacobian entry to about $4\times10^{-5}$ — five thousand times smaller than at
scale 1. The attention backward formulas also agree with finite differences; notice that
$\partial V = P^\top G$ needs no Jacobian at all, while $\partial Q$ and $\partial K$ pass
through $P\odot(\cdot)$ and vanish wherever a row of $P$ is saturated.

**Try it yourself:** set `z = np.log(np.array([0.98, 0.01, 0.01]))` and recompute `J`. What
is the largest entry now, and what does that mean for a classifier that is already confident?
""")

# ---------------------------------------------------------------- Section 5
nb.md(r"""
## 5. Multi-head attention, parameters and FLOPs

$h$ heads of width $d_{\text{head}} = d/h$ each run attention in their own subspace; the
outputs are concatenated and mixed by $W_O$:

$$\operatorname{MHA}(X) = [H_1\,|\,\cdots\,|\,H_h]\,W_O, \qquad H_i = \operatorname{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}).$$

The attention sub-layer has $4d^2$ weights whatever $h$ is; the feed-forward sub-layer with
$d_{\text{ff}} = 4d$ adds $8d^2$; so a block has $12d^2$ weights. Per layer and sequence of $n$
tokens the FLOPs are about $24nd^2$ (weights) $+\ 4n^2d$ (attention) — the attention term wins
beyond $n = 6d$. We print shapes, count GPT-2 small, and plot the crossover.
""")

nb.code(r"""
def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=None, verbose=False):
    n, d = X.shape
    d_head = d // n_heads
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V                            # (n, d) each
    split = lambda M: M.reshape(n, n_heads, d_head).transpose(1, 0, 2)   # -> (h, n, d_head)
    Qh, Kh, Vh = split(Q), split(K), split(V)
    S = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(d_head)              # (h, n, n)
    if mask is not None:
        S = np.where(mask[None], -np.inf, S)
    P = softmax(S, axis=-1)
    Hh = P @ Vh                                                   # (h, n, d_head)
    concat = Hh.transpose(1, 0, 2).reshape(n, d)                  # (n, d)
    if verbose:
        print(f"X {X.shape} -> Q,K,V {Q.shape} -> per head {Qh.shape} -> scores {S.shape} -> heads out {Hh.shape} -> concat {concat.shape} -> W_O -> {(concat @ W_O).shape}")
    return concat @ W_O, P

n, d, h = 6, 16, 4
Xm = rng.normal(size=(n, d))
W_Q, W_K, W_V, W_O = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
out, P_heads = multi_head_attention(Xm, W_Q, W_K, W_V, W_O, h, verbose=True)
print("each head's rows sum to one:", np.allclose(P_heads.sum(axis=-1), 1))
print("rank of one head's score matrix W_Q^(i) W_K^(i)^T:", np.linalg.matrix_rank(W_Q[:, :d // h] @ W_K[:, :d // h].T), "(= d_head =", d // h, ")")

def count_params(d, n_layer, d_ff, vocab, n_ctx, tied=True):
    attn = 4 * d * d + 4 * d                   # W_Q, W_K, W_V, W_O and their biases
    mlp = d * d_ff + d_ff + d_ff * d + d       # W_1, b_1, W_2, b_2
    ln = 2 * (2 * d)                           # two LayerNorms (gain + bias each)
    block = attn + mlp + ln
    emb = vocab * d + n_ctx * d                # token table + learned position table
    unemb = 0 if tied else vocab * d
    total = n_layer * block + emb + unemb + 2 * d          # + final LayerNorm
    return dict(attention_weights=4 * d * d, mlp_weights=2 * d * d_ff, block=block,
                all_blocks=n_layer * block, embeddings=emb, unembedding=unemb, total=total)

gpt2 = count_params(d=768, n_layer=12, d_ff=3072, vocab=50257, n_ctx=1024)
for k_, v_ in gpt2.items():
    print(f"{k_:18s} {v_:>12,d}")
print("12 d^2 =", f"{12 * 768 ** 2:,d}", "  -> GPT-2 small has", f"{gpt2['total'] / 1e6:.1f}M parameters")

def flops_per_layer(n, d):
    return 24 * n * d ** 2, 4 * n ** 2 * d       # (weight FLOPs, attention FLOPs)

print("\nGPT-2 small at n = 1024: weights %.2e FLOPs, attention %.2e FLOPs (%.0f%% of the layer); crossover at n = 6d = %d"
      % (*flops_per_layer(1024, 768), 100 * flops_per_layer(1024, 768)[1] / sum(flops_per_layer(1024, 768)), 6 * 768))

ns = np.logspace(np.log10(128), np.log10(262144), 60)
for dd in [768, 4096, 12288]:
    w_, a_ = flops_per_layer(ns, dd)
    plt.semilogx(ns, a_ / (w_ + a_), label=f"d = {dd}")
    plt.axvline(6 * dd, color="gray", ls=":", lw=0.8)
plt.axhline(0.5, color="k", ls="--", lw=0.8)
plt.title("Share of layer FLOPs spent in attention (4 n^2 d) vs context length")
plt.xlabel("context length n"); plt.ylabel("attention share of FLOPs"); plt.legend(); plt.show()
""")

nb.md(r"""
The shapes tell the story: $Q$, $K$, $V$ are split into $h = 4$ heads of width $4$, each
producing its own $6\times6$ score matrix, and the concatenation restores width 16 before
$W_O$. Each head's score matrix has rank at most $d_{\text{head}}$ — a head is a low-rank
bilinear form on the residual stream. The counter gives $124{,}439{,}808$ parameters for GPT-2
small, with the blocks at $12d^2$ weights each and the token table alone 31 % of the model.
The dotted lines mark $n = 6d$: GPT-2 small (context 1024) never reaches it, while a
$d = 4096$ model at a 128k context spends 84 % of its layer FLOPs on attention.
""")

# ---------------------------------------------------------------- Section 6
nb.md(r"""
## 6. Masking and causality

Adding $-\infty$ above the diagonal of the scores makes $e^{-\infty} = 0$, so position $i$
attends only to positions $\le i$ and the attention matrix is lower-triangular. The reward
is parallel training: the output at every position is a legitimate next-token prediction given
the true prefix (teacher forcing), and one forward pass yields $n$ loss terms,
$\mathcal L = -\tfrac1n\sum_i\log p(t_{i+1}\mid t_{\le i})$, perplexity $e^{\mathcal L}$.
""")

nb.code(r"""
def causal_mask(n):
    return np.triu(np.ones((n, n), dtype=bool), k=1)      # True above the diagonal = forbidden

n = 6
mask = causal_mask(n)
print("mask (1 = forbidden):\n", mask.astype(int))

# the classic bug: 0 * -inf is NaN
with np.errstate(invalid="ignore"):
    bad = np.triu(np.ones((n, n)), 1) * -np.inf
print("entries of np.triu(ones) * -inf that are NaN:", int(np.isnan(bad).sum()), "-> use np.where(mask, -inf, S) instead")

Xc = rng.normal(size=(n, d))
out_c, P_c = multi_head_attention(Xc, W_Q, W_K, W_V, W_O, h, mask=mask)
print("\nhead 0 attention weights with the causal mask:\n", P_c[0])
print("lower-triangular:", np.allclose(np.triu(P_c, 1), 0), " rows sum to one:", np.allclose(P_c.sum(-1), 1), " any NaN:", np.isnan(out_c).any())

# leakage test: change the LAST token and see which outputs move
Xc2 = Xc.copy(); Xc2[-1] += 10.0
out_c2, _ = multi_head_attention(Xc2, W_Q, W_K, W_V, W_O, h, mask=mask)
print("positions whose output changed after editing token 5:", np.flatnonzero(np.abs(out_c2 - out_c).max(axis=1) > 1e-12))
out_u2, _ = multi_head_attention(Xc2, W_Q, W_K, W_V, W_O, h)
out_u, _ = multi_head_attention(Xc, W_Q, W_K, W_V, W_O, h)
print("same edit WITHOUT the mask changes positions:", np.flatnonzero(np.abs(out_u2 - out_u).max(axis=1) > 1e-12))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, Pm, title in [(axes[0], multi_head_attention(Xc, W_Q, W_K, W_V, W_O, h)[1][0], "no mask"), (axes[1], P_c[0], "causal mask")]:
    ax.imshow(Pm, cmap="Blues", vmin=0, vmax=1); ax.set_title(f"head 0 attention, {title}")
    ax.set_xlabel("key position"); ax.set_ylabel("query position")
plt.tight_layout(); plt.show()

# teacher forcing: the loss over all positions in one pass
true_next = np.array([3, 1, 4, 0, 2])                     # targets for positions 0..4 of a 6-token sequence
logits = rng.normal(size=(5, V)) * 2
probs = softmax(logits)
p_correct = probs[np.arange(5), true_next]
loss = -np.log(p_correct).mean()
print("\nprobability assigned to each true next token:", p_correct)
print(f"mean cross-entropy {loss:.3f} nats, perplexity {np.exp(loss):.2f}; uniform over 50257 tokens would be {np.log(50257):.2f} nats")
""")

nb.md(r"""
With the mask, editing the last token changes only the last output; without it, every position
sees the edit — including positions that are supposed to be *predicting* that token. That is
the whole justification for training on all $n$ positions at once. The NaN line is the most
common attention bug in practice: build masks with `np.where`, never by multiplying by
$-\infty$.

**Try it yourself:** replace `-np.inf` in `multi_head_attention` by `-1e9`. The results are
identical — a large finite negative number is what most frameworks actually use.
""")

# ---------------------------------------------------------------- Section 7
nb.md(r"""
## 7. Positional information: equivariance, sinusoids and RoPE

Attention without positions is **permutation-equivariant**: permuting the rows of $X$ permutes
the output rows identically, $\operatorname{Attn}(\Pi Q, \Pi K, \Pi V) = \Pi\operatorname{Attn}(Q, K, V)$.
Sinusoidal encodings fix this with a multi-resolution clock,

$$PE(pos, 2i) = \sin(pos\,\omega_i), \quad PE(pos, 2i+1) = \cos(pos\,\omega_i), \quad \omega_i = 10000^{-2i/d},$$

whose shift by $k$ is a fixed rotation $PE(pos+k) = R_k\,PE(pos)$. RoPE rotates the query and
key pairs themselves by $m\theta_i$, so that $q_m^\top k_n = q^\top R((n-m)\theta)\,k$ depends
only on the offset $n - m$.
""")

nb.code(r"""
# --- permutation equivariance, numerically
perm = rng.permutation(n)
Pi = np.eye(n)[perm]                                   # permutation matrix: (Pi @ X)[i] = X[perm[i]]
out_perm, _ = multi_head_attention(Pi @ Xc, W_Q, W_K, W_V, W_O, h)
print("Attn(Pi X) == Pi Attn(X):", np.allclose(out_perm, Pi @ out_u), " (no mask, no positions)")

# --- sinusoidal encodings
def sinusoidal_pe(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d // 2)[None, :]
    omega = 10000.0 ** (-2 * i / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(pos * omega)
    pe[:, 1::2] = np.cos(pos * omega)
    return pe

PE = sinusoidal_pe(64, 32)
print("\nPE(0..2) for d = 4:\n", sinusoidal_pe(3, 4))

def shift_matrix(k, d):
    # block-diagonal rotation R_k with PE(pos + k) = R_k PE(pos)
    omega = 10000.0 ** (-2 * np.arange(d // 2) / d)
    R = np.zeros((d, d))
    for i, w in enumerate(omega):
        c, s = np.cos(k * w), np.sin(k * w)
        R[2*i:2*i+2, 2*i:2*i+2] = [[c, s], [-s, c]]
    return R

k_shift = 7
R7 = shift_matrix(k_shift, 32)
print(f"max |PE(pos+{k_shift}) - R_{k_shift} PE(pos)| over pos = 0..56:", np.abs(PE[k_shift:] - PE[:-k_shift] @ R7.T).max())
print("R_k is orthogonal:", np.allclose(R7.T @ R7, np.eye(32)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
im = axes[0].imshow(PE.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
axes[0].set_xlabel("position"); axes[0].set_ylabel("dimension"); axes[0].set_title("Sinusoidal PE (d = 32): fast rows at the top, slow at the bottom")
plt.colorbar(im, ax=axes[0], fraction=0.03)
G = PE @ PE.T
axes[1].imshow(G, cmap="viridis"); axes[1].set_title("PE(i) . PE(j): depends on |i - j| only")
axes[1].set_xlabel("position j"); axes[1].set_ylabel("position i"); plt.tight_layout(); plt.show()
""")

nb.md(r"""
The permutation test passes to machine precision — shuffle the tokens and the outputs
shuffle with them, so a bare transformer cannot know word order. The sinusoidal table is a
set of clocks: the top rows cycle every few positions, the bottom rows barely move over 64
positions. The shift property holds exactly ($PE(pos+7) = R_7\,PE(pos)$ for every $pos$), and
the Gram matrix $PE(i)\cdot PE(j)$ is constant along diagonals — the encoding already "knows"
relative distance. RoPE makes that the *only* thing the scores know:
""")

nb.code(r"""
def rope(x, positions, base=10000.0):
    # x: (n, d) with d even; rotate pair i at position m by angle m * theta_i
    n_pos, d = x.shape
    theta = base ** (-2 * np.arange(d // 2) / d)                 # (d/2,)
    ang = positions[:, None] * theta[None, :]                    # (n, d/2)
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * c - x2 * s
    out[:, 1::2] = x1 * s + x2 * c
    return out

d_r = 8
q = rng.normal(size=(1, d_r)); k = rng.normal(size=(1, d_r))
def rope_score(m, n_):
    return (rope(q, np.array([m])) @ rope(k, np.array([n_])).T).item()

print("same content q, k at different absolute positions, same offset n - m = -2:")
for m in [3, 10, 57, 1000]:
    print(f"   positions (m, n) = ({m}, {m - 2}):  q_m . k_n = {rope_score(m, m - 2): .6f}")
print("offset changed to -3:  ", f"{rope_score(10, 7): .6f}", "   offset 0:", f"{rope_score(10, 10): .6f}", "= raw q.k =", f"{(q @ k.T).item(): .6f}")

# the score as a function of the offset only: q_m^T k_n = q^T R((n-m) theta) k
offsets = np.arange(-30, 31)
scores = np.array([rope_score(100, 100 + o) for o in offsets])
scores_other_base = np.array([rope_score(5, 5 + o) for o in offsets])
print("score curve identical when computed around position 100 and around position 5:", np.allclose(scores, scores_other_base))
plt.plot(offsets, scores, "o-", ms=3)
plt.title("RoPE: the score q_m . k_n depends only on the offset n - m"); plt.xlabel("offset n - m"); plt.ylabel("score"); plt.show()
""")

nb.md(r"""
Moving the pair $(q, k)$ from positions $(3, 1)$ to $(1000, 998)$ leaves the score unchanged
to the last digit, while changing the offset by one changes it — exactly the statement
$q_m^\top k_n = q^\top R((n-m)\theta)\,k$. The curve is a sum of $d/2$ sinusoids in the offset
with frequencies $\theta_i$; the slowest ones only complete a cycle after thousands of
positions, which is why a model trained on short contexts has never seen the angles that
long contexts produce.

**Try it yourself:** change `base` to `500000.0` (Llama 3's choice) and re-plot. The curve
becomes smoother in the offset — the slow pairs rotate even more slowly.
""")

# ---------------------------------------------------------------- Section 8
nb.md(r"""
## 8. LayerNorm, residuals and a full pre-norm block

LayerNorm acts on *one token's* vector, over its $d$ features:

$$\operatorname{LN}(x) = \gamma\odot\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad
\mu = \tfrac1d\textstyle\sum_j x_j, \quad \sigma^2 = \tfrac1d\sum_j (x_j - \mu)^2 .$$

A pre-norm block is $X \leftarrow X + \operatorname{MHA}(\operatorname{LN}_1 X)$,
$X \leftarrow X + W_2\operatorname{GELU}(W_1\operatorname{LN}_2 X + b_1) + b_2$, with
$\operatorname{GELU}(u) = u\,\Phi(u)$. We implement all of it and run a two-layer, 4-head toy model.
""")

nb.code(r"""
from scipy.special import erf

def layer_norm(X, gamma=None, beta=None, eps=1e-5):
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    Xn = (X - mu) / np.sqrt(var + eps)
    if gamma is not None:
        Xn = Xn * gamma + beta
    return Xn

def rms_norm(X, gamma=None, eps=1e-5):
    rms = np.sqrt((X ** 2).mean(axis=-1, keepdims=True) + eps)
    return X / rms if gamma is None else gamma * X / rms

def gelu(u):
    return 0.5 * u * (1.0 + erf(u / np.sqrt(2.0)))      # u * Phi(u), exact form

x = np.array([[2., 4., 6., 8.]])
print("LN(2,4,6,8) =", layer_norm(x, eps=0.0), "   mean %.1f, std %.1f" % (layer_norm(x, eps=0.0).mean(), layer_norm(x, eps=0.0).std()))
print("LN(10 x + 100) =", layer_norm(10 * x + 100, eps=0.0), " (scale and shift removed)")
print("RMSNorm(2,4,6,8) =", rms_norm(x, eps=0.0))
print("GELU(-1), GELU(1), GELU(2) =", gelu(np.array([-1., 1., 2.])))

def init_block(d, d_ff, rng):
    s = 1 / np.sqrt(d)
    return dict(W_Q=rng.normal(size=(d, d)) * s, W_K=rng.normal(size=(d, d)) * s, W_V=rng.normal(size=(d, d)) * s,
                W_O=rng.normal(size=(d, d)) * s, W_1=rng.normal(size=(d, d_ff)) * s, b_1=np.zeros(d_ff),
                W_2=rng.normal(size=(d_ff, d)) / np.sqrt(d_ff), b_2=np.zeros(d),
                g1=np.ones(d), be1=np.zeros(d), g2=np.ones(d), be2=np.zeros(d))

def transformer_block(X, p, n_heads, mask):
    a, _ = multi_head_attention(layer_norm(X, p["g1"], p["be1"]), p["W_Q"], p["W_K"], p["W_V"], p["W_O"], n_heads, mask)
    X = X + a                                                  # residual 1
    f = gelu(layer_norm(X, p["g2"], p["be2"]) @ p["W_1"] + p["b_1"]) @ p["W_2"] + p["b_2"]
    return X + f                                               # residual 2

n_tok, d_m, n_heads, d_ff, n_layers = 8, 32, 4, 128, 2
V_toy = 12
E_toy = rng.normal(size=(V_toy, d_m)) * 0.5
toks = rng.integers(0, V_toy, size=n_tok)
X = E_toy[toks] + sinusoidal_pe(n_tok, d_m)
blocks = [init_block(d_m, d_ff, rng) for _ in range(n_layers)]
norms = [np.linalg.norm(X, axis=1).mean()]
for p_ in blocks:
    X = transformer_block(X, p_, n_heads, causal_mask(n_tok))
    norms.append(np.linalg.norm(X, axis=1).mean())
logits = layer_norm(X) @ E_toy.T                               # final LN, tied unembedding
print("\ntokens:", toks)
print("mean residual-stream norm after embedding, block 1, block 2:", np.round(norms, 3))
print("logits shape", logits.shape, " next-token probabilities at the last position:", softmax(logits[-1]))
print("parameters in this toy model:", sum(v.size for p_ in blocks for v in p_.values()) + E_toy.size, " (12 d^2 per block =", 12 * d_m ** 2, ")")
""")

nb.md(r"""
LayerNorm removes exactly two degrees of freedom (overall offset and scale) and keeps the
pattern; RMSNorm removes only the scale. In the pre-norm block the residual stream's norm
*grows* with depth — each block adds something to an untouched identity path — which is why
a final LayerNorm sits before the unembedding. The toy model has $12d^2 = 12{,}288$ weights
per block (plus biases and norms), exactly the accounting of Section 5.
""")

# ---------------------------------------------------------------- Section 9
nb.md(r"""
## 9. Decoding strategies and the KV cache

Given logits $z$: greedy takes $\arg\max$; temperature samples from $\operatorname{softmax}(z/T)$;
top-$k$ keeps the $k$ largest; top-$p$ keeps the smallest set with cumulative mass $\ge p$;
beam search keeps the $B$ best partial sequences by total log-probability. The KV cache stores
$K$ and $V$ for every past token in every layer — $2\,L\,t\,h_{kv}d_{\text{head}}$ numbers.
""")

nb.code(r"""
z = np.array([2.0, 1.0, 0.5, -1.0])
labels = ["A", "B", "C", "D"]

def temperature_probs(z, T):
    return softmax(z / T)

def top_k_probs(p, k):
    keep = np.argsort(-p)[:k]
    q = np.zeros_like(p); q[keep] = p[keep]
    return q / q.sum()

def top_p_probs(p, p_nucleus):
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    n_keep = int(np.searchsorted(cum, p_nucleus) + 1)          # smallest prefix with mass >= p
    q = np.zeros_like(p); q[order[:n_keep]] = p[order[:n_keep]]
    return q / q.sum()

p1 = temperature_probs(z, 1.0)
print("greedy picks:", labels[int(np.argmax(z))])
for T in [0.5, 1.0, 2.0]:
    pT = temperature_probs(z, T)
    print(f"T = {T}: p = {pT}, entropy {entropy(pT):.3f} nats")
print("top-k, k = 2:", top_k_probs(p1, 2))
print("top-p, p = 0.9: cumulative", np.cumsum(np.sort(p1)[::-1]), "->", top_p_probs(p1, 0.9))

Ts = np.linspace(0.1, 5, 100)
plt.plot(Ts, [entropy(temperature_probs(z, T)) for T in Ts])
plt.axhline(np.log(4), color="gray", ls=":", label="uniform (ln 4)")
plt.title("Entropy of softmax(z / T) grows with temperature"); plt.xlabel("temperature T"); plt.ylabel("entropy (nats)"); plt.legend(); plt.show()

# beam search on a toy model: next-token log-probs depend on the previous token only (a Markov chain)
logP = np.log(softmax(rng.normal(size=(4, 4)) * 2, axis=1))     # logP[prev, next]
def beam_search(logP, start, length, B):
    beams = [([start], 0.0)]
    for _ in range(length):
        cands = [(seq + [nxt], lp + logP[seq[-1], nxt]) for seq, lp in beams for nxt in range(4)]
        beams = sorted(cands, key=lambda c: -c[1])[:B]
    return beams
greedy_seq, greedy_lp = [0], 0.0
for _ in range(4):
    nxt = int(np.argmax(logP[greedy_seq[-1]])); greedy_lp += logP[greedy_seq[-1], nxt]; greedy_seq.append(nxt)
best = beam_search(logP, 0, 4, B=3)[0]
print("\ngreedy sequence", greedy_seq, "log-prob %.3f" % greedy_lp)
print("beam (B=3)     ", best[0], "log-prob %.3f  (beam >= greedy always)" % best[1])

def kv_cache_bytes(n_layer, n_tokens, n_kv_heads, d_head, bytes_per_number=2):
    return 2 * n_layer * n_tokens * n_kv_heads * d_head * bytes_per_number

print("\nKV cache:")
for name, cfg in [("GPT-2 small, 1024 ctx", dict(n_layer=12, n_tokens=1024, n_kv_heads=12, d_head=64)),
                  ("7B-class MHA, 8k ctx", dict(n_layer=32, n_tokens=8192, n_kv_heads=32, d_head=128)),
                  ("70B-class GQA, 8k ctx", dict(n_layer=80, n_tokens=8192, n_kv_heads=8, d_head=128)),
                  ("70B-class if MHA, 8k ctx", dict(n_layer=80, n_tokens=8192, n_kv_heads=64, d_head=128))]:
    b = kv_cache_bytes(**cfg)
    print(f"  {name:26s} {b / 2**30:7.2f} GiB  ({b / cfg['n_tokens'] / 1024:6.0f} KiB per token)")
""")

nb.md(r"""
Lowering the temperature to 0.5 squares the odds (0.61 → 0.84 for the top token) and halves
the entropy; raising it to 2 flattens the distribution. Top-$k$ and top-$p$ then cut the tail
and renormalise. Beam search never scores below greedy because greedy is one of the paths it
considers. The cache arithmetic reproduces the guide: 4 GiB for a 7B model at 8k context in
fp16, and a 70B model with grouped-query attention needs *less* cache (2.5 GiB) than the 7B
model, because only $h_{kv}d_{\text{head}} = 1024$ numbers per layer per token are stored.

**Try it yourself:** change `bytes_per_number` to 1 (an 8-bit cache) or `n_tokens` to
131072 (a 128k context) and see which models still fit in 80 GB next to their weights.
""")

# ---------------------------------------------------------------- Section 10
nb.md(r"""
## 10. LoRA as low-rank approximation; quantisation error

LoRA writes the fine-tuned weight as $W = W_0 + \tfrac{\alpha}{r}BA$ with $B \in \mathbb{R}^{d\times r}$,
$A \in \mathbb{R}^{r\times d}$: $r(d_{\text{in}} + d_{\text{out}})$ trainable numbers instead of
$d_{\text{in}}d_{\text{out}}$. If the ideal update $\Delta W^\ast$ has singular values $\sigma_i$,
the best rank-$r$ adapter leaves error $\sqrt{\sum_{i>r}\sigma_i^2}$ (Eckart–Young, Module 03).
We build a true rank-4 update plus noise and watch the error curve, then measure absmax
quantisation error against its $s/\sqrt{12}$ prediction.
""")

nb.code(r"""
d_lora = 256
W0 = rng.normal(size=(d_lora, d_lora)) / np.sqrt(d_lora)
B_true = rng.normal(size=(d_lora, 4)); A_true = rng.normal(size=(4, d_lora))
dW_true = B_true @ A_true / np.sqrt(4 * d_lora) + 0.002 * rng.normal(size=(d_lora, d_lora))   # rank 4 + small noise

U, s, Vt = np.linalg.svd(dW_true)
print("leading singular values of the update:", s[:6], "...")
ranks = np.arange(1, 33)
err = [np.linalg.norm(dW_true - (U[:, :r] * s[:r]) @ Vt[:r]) for r in ranks]
ey = [np.sqrt((s[r:] ** 2).sum()) for r in ranks]
print("max |measured error - Eckart-Young formula| = %.2e" % np.abs(np.array(err) - np.array(ey)).max())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].semilogy(ranks, np.array(err) / np.linalg.norm(dW_true), "o-", label="||dW - B A||_F / ||dW||_F")
axes[0].axvline(4, color="gray", ls=":", label="true rank 4"); axes[0].set_xlabel("adapter rank r"); axes[0].set_ylabel("relative error")
axes[0].set_title("Best rank-r approximation of a fine-tuning update"); axes[0].legend()
params_full = d_lora ** 2
axes[1].plot(ranks, ranks * 2 * d_lora / params_full * 100, "s-")
axes[1].set_xlabel("adapter rank r"); axes[1].set_ylabel("trainable parameters (% of d^2)"); axes[1].set_title(f"LoRA parameter cost, d = {d_lora}")
plt.tight_layout(); plt.show()

# merging: the adapted layer and the merged weight give identical outputs
r, alpha = 8, 16
B_ad, A_ad = U[:, :r] * np.sqrt(s[:r]), (np.sqrt(s[:r])[:, None] * Vt[:r]) * r / alpha
x = rng.normal(size=(5, d_lora))
y_adapter = x @ W0.T + (x @ A_ad.T) @ B_ad.T * (alpha / r)
W_merged = W0 + (alpha / r) * B_ad @ A_ad
print("adapter path == merged weight:", np.allclose(y_adapter, x @ W_merged.T))
print(f"rank-{r} adapter: {r * 2 * d_lora:,d} parameters = {r * 2 * d_lora / d_lora**2:.2%} of the {d_lora**2:,d}-entry matrix;"
      f"  for d = 4096: {8 * 2 * 4096:,d} vs {4096**2:,d} ({8 * 2 * 4096 / 4096**2:.2%})")

# absmax quantisation
def quantize_absmax(w, bits):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(w).max() / qmax
    q = np.round(w / s)
    return q, s

w = rng.normal(size=100_000) * 0.02                              # realistic weight magnitudes
print("\nabsmax quantisation of 100k weights (std 0.02, absmax %.4f):" % np.abs(w).max())
for bits in [8, 4]:
    q, s_ = quantize_absmax(w, bits)
    e = w - q * s_
    print(f"  int{bits}: scale s = {s_:.2e}, max |error| = {np.abs(e).max():.2e} (bound s/2 = {s_/2:.2e}), RMS error = {e.std():.2e} (prediction s/sqrt(12) = {s_/np.sqrt(12):.2e}), relative to weight std: {e.std()/w.std():.1%}")
blocks_ = w.reshape(-1, 32)                                      # block-wise scales: 32 weights per scale
e_blk = np.concatenate([(b - quantize_absmax(b, 4)[0] * quantize_absmax(b, 4)[1]) for b in blocks_])
print(f"  int4 with per-32-weight blocks: RMS error = {e_blk.std():.2e} ({e_blk.std()/w.std():.1%} of weight std)")
""")

nb.md(r"""
The error curve drops to the noise floor at $r = 4$ — the true rank — and sits exactly on the
Eckart–Young formula; beyond that, extra rank buys nothing but parameters. A rank-8 adapter on
a $4096\times4096$ matrix is 0.39 % of its size, and the merged weight reproduces the adapter
path to machine precision, so inference costs nothing extra. Quantisation error behaves as
uniform rounding noise: RMS $s/\sqrt{12}$, so int8 is at 1 % of the weights' typical size
while a single int4 scale for the whole tensor is at 17 %; small blocks with their own scales
cut the int4 error roughly in half (about 10 %), which — together with outlier handling — is the trick behind 4-bit inference.
""")

# ---------------------------------------------------------------- Section 11
nb.md(r"""
## 11. Embedding geometry: anisotropy and linear probes

Hidden states of real transformers share a large common component, so the cosine similarity
between *unrelated* tokens is far above zero (anisotropy); centring fixes it. A linear probe
— logistic regression from frozen hidden states to a label — tests whether a concept is a
*direction* in the representation. We simulate both effects.
""")

nb.code(r"""
from sklearn.linear_model import LogisticRegression

def cosine_matrix(H):
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    return Hn @ Hn.T

n_tok, d_h = 300, 64
labels_c = rng.integers(0, 2, size=n_tok)
concept_dir = rng.normal(size=d_h); concept_dir /= np.linalg.norm(concept_dir)
common = 3.0 * rng.normal(size=d_h)                                # the shared offset every token carries
H = rng.normal(size=(n_tok, d_h)) + common + 4.0 * (labels_c[:, None] - 0.5) * concept_dir   # isotropic noise + anisotropy + a concept direction

C_raw = cosine_matrix(H); C_cen = cosine_matrix(H - H.mean(axis=0))
off = ~np.eye(n_tok, dtype=bool)
print(f"mean cosine between unrelated tokens: raw {C_raw[off].mean():.3f}, after centring {C_cen[off].mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].hist(C_raw[off], bins=60, alpha=0.7, label="raw"); axes[0].hist(C_cen[off], bins=60, alpha=0.7, label="centred")
axes[0].set_title("Pairwise cosine similarities of hidden states"); axes[0].set_xlabel("cosine"); axes[0].set_ylabel("count"); axes[0].legend()

# linear probe: can logistic regression read the concept off the hidden state?
idx = rng.permutation(n_tok); tr, te = idx[:200], idx[200:]
probe = LogisticRegression(max_iter=2000).fit(H[tr], labels_c[tr])
acc = probe.score(H[te], labels_c[te])
w_hat = probe.coef_[0] / np.linalg.norm(probe.coef_)
print(f"linear probe held-out accuracy: {acc:.2f};  cosine(probe direction, true concept direction) = {abs(w_hat @ concept_dir):.3f}")
proj = H @ concept_dir
axes[1].hist(proj[labels_c == 0], bins=30, alpha=0.7, label="label 0"); axes[1].hist(proj[labels_c == 1], bins=30, alpha=0.7, label="label 1")
axes[1].set_title("Projection of hidden states onto the concept direction"); axes[1].set_xlabel("h . w"); axes[1].set_ylabel("count"); axes[1].legend()
plt.tight_layout(); plt.show()

# the guide's 3-D example
u, v = np.array([1., 0, 0]), np.array([0, 1., 0])
print("\ncos(u, v) = %.3f;  after adding (3,3,3) to both: %.3f;  after centring again: %.3f"
      % (u @ v, (u + 3) @ (v + 3) / np.linalg.norm(u + 3) / np.linalg.norm(v + 3),
         float(cosine_matrix(np.stack([u + 3, v + 3]) - np.stack([u + 3, v + 3]).mean(axis=0))[0, 1])))
""")

nb.md(r"""
The shared offset makes every pair of tokens look alike (mean cosine around 0.9); subtracting
the mean vector restores a distribution centred on zero, and *then* cosine similarity means
something. The probe recovers the planted concept direction almost exactly and classifies
held-out tokens with high accuracy — the operational meaning of "the concept is linearly
represented". Both diagnostics are routine when building retrieval indexes or interpreting
hidden states.
""")

# ---------------------------------------------------------------- Summary
nb.md(r"""
## Summary

- Embedding lookup is a one-hot matrix product; with tied weights, next-token logits are dot
  products of the residual stream with every token's embedding.
- Attention is $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$: a data-dependent convex
  combination of values; uniform scores give the plain mean.
- Scores have variance $d_k$ at initialisation; dividing by $\sqrt{d_k}$ keeps the softmax
  soft and the gradients alive.
- The softmax Jacobian is $\operatorname{diag}(p) - pp^\top$; its rows sum to zero and it
  vanishes under saturation. Attention's backward pass follows from it in four lines.
- A block has $12d^2$ weights and costs $24nd^2 + 4n^2d$ FLOPs per layer; GPT-2 small's
  $124{,}439{,}808$ parameters follow from its config; attention dominates beyond $n = 6d$.
- The causal mask ($-\infty$ above the diagonal, built with `np.where`) makes every position a
  valid next-token prediction, so one pass gives $n$ loss terms.
- Attention is permutation-equivariant; sinusoidal encodings shift by rotations, RoPE makes
  the score a function of the relative offset only.
- LayerNorm normalises each token over its features; pre-norm residual blocks let the stream
  grow while the sub-layers see unit-scale inputs.
- Temperature, top-$k$, top-$p$ and beam search trade accuracy for diversity; the KV cache
  costs $2Lt\,h_{kv}d_{\text{head}}$ numbers — 4 GiB for a 7B model at 8k context in fp16.
- LoRA is a learned truncated SVD of the weight update; merged, it costs nothing at inference.
  Quantisation error is uniform rounding noise with RMS $s/\sqrt{12}$.
- Hidden states are anisotropic — centre before comparing cosines — and linear probes test
  whether a concept is a direction.

## Next: open `exercises.ipynb`

Twelve exercises with instant self-checks, from embedding lookup to a single-head causal
transformer trained with hand-derived gradients.
""")

nb.save(MOD / "lesson.ipynb")
print("lesson cells:", len(nb.cells))

# =============================================================================
# EXERCISES — solution code is executed here to compute the hidden answers
# =============================================================================
SETUP = r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
"""

ex = ExerciseSet(
    "Module 17 — Exercises",
    intro_md=r"""
Exercises for **Module 17 — The Mathematics of Transformers**. They follow the order of the
[concept guide](README.md): embeddings and tied logits, scaled dot-product attention,
LayerNorm, the softmax Jacobian, the causal mask, parameter counting, sinusoidal and rotary
positions, the KV cache, a full pre-norm block, a conceptual diagnosis of the "temperature
replaces $\sqrt{d_k}$" fallacy, and a mini-project that trains a single-head causal
transformer with hand-derived gradients. Use only NumPy (plus `scipy.special.erf` for the GELU) and keep everything seeded.
A `softmax(z, axis=-1)` helper is defined in the setup cell.
""",
    setup_code=SETUP,
)

ns: dict = {}
exec(SETUP, ns)


def run_solution(src: str) -> None:
    """Execute solution code inside the shared namespace (hashes come from *running* it)."""
    exec(src, ns)


# ---------------------------------------------------------------- Ex 1
SOL1 = r"""
def embed(tokens, E):
    one_hot = np.eye(E.shape[0])[tokens]        # (n, V)
    return one_hot @ E                          # (n, d)

def tied_logits(h, E):
    return E @ h                                # one dot product per vocabulary row
"""
run_solution(SOL1)
_E1 = np.array([[1., 0., 0.5], [0., 1., 0.5], [-1., 0., 0.5], [0., -1., 0.5], [0.5, 0.5, 0.]])
_h1 = np.array([0.8, 0.2, 0.5])
h1 = sha_of(float(ns["softmax"](ns["tied_logits"](_h1, _E1)).max()))

ex.add(
    title="Embedding lookup and tied logits",
    difficulty=1,
    prompt_md=r"""
Implement

* `embed(tokens, E)` — for an integer array `tokens` of shape `(n,)` and an embedding matrix
  `E` of shape `(V, d)`, return the `(n, d)` matrix of embeddings **as a one-hot matrix
  product** (`np.eye(V)[tokens] @ E`), not by indexing;
* `tied_logits(h, E)` — for a residual-stream vector `h` of shape `(d,)`, return the `(V,)`
  vector of logits $z = Eh$ under weight tying.

The hidden check asks for the largest next-token probability, $\max_v \operatorname{softmax}(Eh)_v$,
for the guide's five-token matrix and $h = (0.8, 0.2, 0.5)$.
""",
    starter=r"""
def embed(tokens, E):
    # YOUR CODE HERE  (one-hot rows times E)
    return None

def tied_logits(h, E):
    # YOUR CODE HERE
    return None
""",
    solution=SOL1,
    check=r"""
_E = np.array([[1., 0., 0.5], [0., 1., 0.5], [-1., 0., 0.5], [0., -1., 0.5], [0.5, 0.5, 0.]])
try:
    got = embed(np.array([0, 1, 2]), _E)
except Exception:
    got = None
check_close('embed returns the rows of E for tokens 0,1,2', got, _E[[0, 1, 2]])
try:
    got_rep = embed(np.array([4, 4, 1]), _E)
except Exception:
    got_rep = None
check_close('repeated tokens give repeated rows', got_rep, _E[[4, 4, 1]])
try:
    z = tied_logits(np.array([0.8, 0.2, 0.5]), _E)
except Exception:
    z = None
check_close('logits for h = (0.8, 0.2, 0.5)', z, np.array([1.05, 0.45, -0.55, 0.05, 0.5]))
try:
    p_max = float(softmax(tied_logits(np.array([0.8, 0.2, 0.5]), _E)).max())
except Exception:
    p_max = None
check_hash('hidden: largest next-token probability (3 decimals)', p_max, 'HASH1')
""".replace("HASH1", h1),
    explanation_md=r"""
A one-hot row $e_t^\top$ times $E$ picks out row $t$, so `np.eye(V)[tokens] @ E` equals
`E[tokens]` — but writing it as a product makes the gradient obvious: $\partial L/\partial E =
\text{one\_hot}^\top\,\partial L/\partial X$, non-zero only in the rows that occurred. Tied
logits are the dot products of $h$ with every embedding row, $z = Eh$, giving $(1.05, 0.45,
-0.55, 0.05, 0.5)$ and a top probability of $0.371$ for "the". A common slip is to write
`h @ E` (shape mismatch) or `E.T @ h`.
""",
)

# ---------------------------------------------------------------- Ex 2
SOL2 = r"""
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)
    P = softmax(S, axis=-1)
    return P @ V, P
"""
run_solution(SOL2)

ex.add(
    title="Scaled dot-product attention",
    difficulty=1,
    prompt_md=r"""
Implement `scaled_dot_product_attention(Q, K, V)` for `Q` of shape `(n_q, d_k)`, `K` of shape
`(n_k, d_k)` and `V` of shape `(n_k, d_v)`. Return the tuple `(output, P)` where
`P = softmax(Q K^T / sqrt(d_k))` is taken **row-wise** (one distribution per query) and
`output = P @ V`. Use only NumPy and the provided `softmax`.
""",
    starter=r"""
def scaled_dot_product_attention(Q, K, V):
    # YOUR CODE HERE  (scores, divide by sqrt(d_k), row-wise softmax, weighted sum of V)
    return None, None
""",
    solution=SOL2,
    check=r"""
_Q = np.array([[1., 0.], [0., 1.], [1., 1.]]); _V = np.array([[2., 0.], [0., 2.], [1., 1.]])
try:
    out, P = scaled_dot_product_attention(_Q, _Q, _V)
except Exception:
    out, P = None, None
check_close('guide example: attention weights', P, np.array([[0.401112, 0.197776, 0.401112], [0.197776, 0.401112, 0.401112], [0.248255, 0.248255, 0.50349]]), tol=1e-5)
check_close('guide example: output rows', out, np.array([[1.203336, 0.796664], [0.796664, 1.203336], [1.0, 1.0]]), tol=1e-5)
_g = np.random.default_rng(3)
_K2 = _g.normal(size=(6, 4)); _V2 = _g.normal(size=(6, 3))
try:
    out_u, P_u = scaled_dot_product_attention(np.zeros((2, 4)), _K2, _V2)
except Exception:
    out_u, P_u = None, None
check('zero queries give uniform weights 1/6', P_u is not None and np.allclose(P_u, 1 / 6))
check_close('uniform weights give the mean of V', out_u, np.tile(_V2.mean(axis=0), (2, 1)))
try:
    _, P_s = scaled_dot_product_attention(_g.normal(size=(5, 4)), _K2, _V2)
    row_sums = P_s.sum(axis=1)
except Exception:
    row_sums = None
check_close('rows of P sum to one for random inputs', row_sums, np.ones(5))
""",
    explanation_md=r"""
The score matrix $S = QK^\top/\sqrt{d_k}$ has one row per query and one column per key; the
softmax must run along `axis=-1` so that each query's weights form a distribution. The output
row $o_i = \sum_j P_{ij}v_j$ is then a convex combination of value rows, which is why constant
scores give exactly the mean of $V$. The two classic bugs are the wrong softmax axis (columns
sum to one instead of rows) and dividing *after* the softmax, which does nothing useful.
""",
)

# ---------------------------------------------------------------- Ex 3
SOL4 = r"""
def layer_norm(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta
"""
run_solution(SOL4)

ex.add(
    title="LayerNorm from scratch",
    difficulty=1,
    prompt_md=r"""
Implement `layer_norm(X, gamma, beta, eps=1e-5)` for `X` of shape `(n, d)` (one token per
row), `gamma` and `beta` of shape `(d,)`:

$$\operatorname{LN}(x) = \gamma\odot\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,$$

with $\mu$ and $\sigma^2$ (the **biased** variance, `ddof=0`) computed **per row over the $d$
features**. Use only NumPy.
""",
    starter=r"""
def layer_norm(X, gamma, beta, eps=1e-5):
    # YOUR CODE HERE  (statistics along axis=-1 with keepdims=True)
    return None
""",
    solution=SOL4,
    check=r"""
_x = np.array([[2., 4., 6., 8.]])
try:
    got = layer_norm(_x, np.ones(4), np.zeros(4), eps=0.0)
except Exception:
    got = None
check_close('LN(2,4,6,8) with gamma=1, beta=0', got, np.array([[-1.341641, -0.447214, 0.447214, 1.341641]]), tol=1e-5)
try:
    got_inv = layer_norm(10 * _x + 100, np.ones(4), np.zeros(4), eps=0.0)
except Exception:
    got_inv = None
check_close('invariant to scaling and shifting the input', got_inv, got if got is not None else np.zeros(1))
_g = np.random.default_rng(5)
_X = _g.normal(size=(7, 16)) * 3 + 2; _gam = _g.normal(size=16); _bet = _g.normal(size=16)
try:
    out = layer_norm(_X, _gam, _bet)
    ref = _gam * (_X - _X.mean(1, keepdims=True)) / np.sqrt(_X.var(1, keepdims=True) + 1e-5) + _bet
except Exception:
    out, ref = None, np.zeros(1)
check_close('matches the formula with learned gamma, beta on a 7 x 16 batch', out, ref)
try:
    row_means = layer_norm(_X, np.ones(16), np.zeros(16)).mean(axis=1)
except Exception:
    row_means = None
check_close('every normalised row has mean 0', row_means, np.zeros(7), tol=1e-8)
""",
    explanation_md=r"""
The statistics belong to one token: `axis=-1` with `keepdims=True` so that broadcasting
subtracts each row's own mean and divides by its own standard deviation. The output is
invariant to any affine rescaling of the input row, and $\gamma$, $\beta$ then put back a learned
scale and offset. Normalising along `axis=0` instead would mix tokens — batch-norm behaviour —
which breaks causality at inference and depends on which other sequences are in the batch.
""",
)

# ---------------------------------------------------------------- Ex 4
SOL5 = r"""
def softmax_jacobian(z):
    p = softmax(z)
    return np.diag(p) - np.outer(p, p)

def finite_difference_jacobian(f, z, eps=1e-6):
    n_out, n_in = f(z).size, z.size
    J = np.zeros((n_out, n_in))
    for j in range(n_in):
        e = np.zeros_like(z); e[j] = eps
        J[:, j] = (f(z + e) - f(z - e)) / (2 * eps)
    return J
"""
run_solution(SOL5)
_Jsat = ns["softmax_jacobian"](np.array([6.0, 0.0, 0.0, 0.0]))
h5 = sha_of(float(np.abs(_Jsat).max()), decimals=4)

ex.add(
    title="The softmax Jacobian, verified by finite differences",
    difficulty=2,
    prompt_md=r"""
Implement

* `softmax_jacobian(z)` — for logits `z` of shape `(n,)`, return the $n\times n$ matrix
  $J_{ij} = \partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$, i.e. $\operatorname{diag}(p) - pp^\top$;
* `finite_difference_jacobian(f, z, eps=1e-6)` — a generic central-difference Jacobian of a
  vector function `f` at `z`: column $j$ is $(f(z + \epsilon e_j) - f(z - \epsilon e_j))/(2\epsilon)$.

The hidden check asks for the largest absolute entry of the Jacobian at the saturated logits
$z = (6, 0, 0, 0)$, rounded to 4 decimals.
""",
    starter=r"""
def softmax_jacobian(z):
    # YOUR CODE HERE
    return None

def finite_difference_jacobian(f, z, eps=1e-6):
    # YOUR CODE HERE
    return None
""",
    solution=SOL5,
    check=r"""
_z = np.log(np.array([0.7, 0.2, 0.1]))
try:
    J = softmax_jacobian(_z)
except Exception:
    J = None
check_close('J for p = (0.7, 0.2, 0.1)', J, np.array([[0.21, -0.14, -0.07], [-0.14, 0.16, -0.02], [-0.07, -0.02, 0.09]]), tol=1e-9)
check('rows of J sum to zero', J is not None and np.allclose(J.sum(axis=1), 0))
_g = np.random.default_rng(7); _z2 = _g.normal(size=6)
try:
    J_an = softmax_jacobian(_z2); J_fd = finite_difference_jacobian(softmax, _z2)
except Exception:
    J_an, J_fd = None, np.zeros(1)
check_close('analytic Jacobian matches finite differences on random logits', J_an, J_fd, tol=1e-6)
try:
    J_lin = finite_difference_jacobian(lambda v: np.array([[1., 2.], [3., 4.]]) @ v, np.array([0.3, -0.7]))
except Exception:
    J_lin = None
check_close('finite differences recover the matrix of a linear map', J_lin, np.array([[1., 2.], [3., 4.]]), tol=1e-6)
try:
    big = float(np.abs(softmax_jacobian(np.array([6.0, 0.0, 0.0, 0.0]))).max())
except Exception:
    big = None
check_hash('hidden: largest |J| entry at z = (6,0,0,0), 4 decimals', big, 'HASH5', decimals=4)
""".replace("HASH5", h5),
    explanation_md=r"""
For $i = j$ the quotient rule gives $p_i - p_i^2$, for $i \ne j$ it gives $-p_ip_j$; in matrix
form $\operatorname{diag}(p) - pp^\top$, which is symmetric with zero row sums because
$\sum_j p_j = 1$. Central differences have error $O(\epsilon^2)$, so with $\epsilon = 10^{-6}$ the
agreement is to about $10^{-9}$. At $z = (6, 0, 0, 0)$ the top probability is $0.993$ and the
largest Jacobian entry is $p_1(1-p_1) \approx 0.0074$ — the gradient wall that saturation builds.
""",
)

# ---------------------------------------------------------------- Ex 5
SOL6 = r"""
def causal_mask(n):
    return np.triu(np.ones((n, n), dtype=bool), k=1)      # True above the diagonal = forbidden

def causal_attention(Q, K, V):
    n = Q.shape[0]
    S = Q @ K.T / np.sqrt(Q.shape[-1])
    S = np.where(causal_mask(n), -np.inf, S)
    P = softmax(S, axis=-1)
    return P @ V, P
"""
run_solution(SOL6)

ex.add(
    title="The causal mask and a no-leakage test",
    difficulty=2,
    prompt_md=r"""
Implement

* `causal_mask(n)` — an `(n, n)` boolean array that is `True` strictly **above** the diagonal
  (the forbidden future positions) and `False` elsewhere;
* `causal_attention(Q, K, V)` — scaled dot-product attention where the masked scores are set
  to `-np.inf` **before** the softmax (use `np.where`, never `mask * -np.inf`), returning
  `(output, P)`.

The checks verify that `P` is lower-triangular with rows summing to one, that no entry is NaN,
and — the real test — that changing the last token leaves every earlier output unchanged.
""",
    starter=r"""
def causal_mask(n):
    # YOUR CODE HERE
    return None

def causal_attention(Q, K, V):
    # YOUR CODE HERE
    return None, None
""",
    solution=SOL6,
    check=r"""
try:
    m = causal_mask(4)
except Exception:
    m = None
check('causal_mask(4) is the strict upper triangle', m is not None and m.dtype == bool and np.array_equal(m, np.array([[0,1,1,1],[0,0,1,1],[0,0,0,1],[0,0,0,0]], dtype=bool)))
_g = np.random.default_rng(9)
_X = _g.normal(size=(6, 8)); _Wq, _Wk, _Wv = (_g.normal(size=(8, 4)) for _ in range(3))
try:
    out, P = causal_attention(_X @ _Wq, _X @ _Wk, _X @ _Wv)
except Exception:
    out, P = None, None
check('P is lower-triangular', P is not None and np.allclose(np.triu(P, 1), 0))
check('rows of P sum to one and contain no NaN', P is not None and not np.isnan(P).any() and np.allclose(P.sum(axis=1), 1))
check('position 0 attends only to itself', P is not None and np.isclose(P[0, 0], 1.0))
try:
    _X2 = _X.copy(); _X2[-1] += 5.0
    out2, _ = causal_attention(_X2 @ _Wq, _X2 @ _Wk, _X2 @ _Wv)
    changed = np.flatnonzero(np.abs(out2 - out).max(axis=1) > 1e-12)
except Exception:
    changed = None
check('editing the last token changes only the last output (no leakage)', changed is not None and np.array_equal(changed, np.array([5])))
""",
    explanation_md=r"""
`np.triu(np.ones((n, n), dtype=bool), k=1)` marks the strict upper triangle; `np.where(mask,
-np.inf, S)` replaces those scores so that $e^{-\infty} = 0$ after the (shift-invariant)
softmax. Multiplying a 0/1 mask by $-\infty$ instead produces `0 * -inf = nan` in the
*allowed* positions and poisons every row. The leakage test is the operational definition of
causality: because row $i$ of $P$ is zero beyond column $i$, output $i$ is a function of
tokens $0..i$ only, and perturbing token 5 can move only output 5.
""",
)

# ---------------------------------------------------------------- Ex 6
SOL7 = r"""
def transformer_params(d, n_heads, d_ff, vocab, n_ctx, n_layer, tied=True):
    attn = 4 * d * d + 4 * d                    # W_Q, W_K, W_V, W_O + four bias vectors
    mlp = d * d_ff + d_ff + d_ff * d + d        # W_1, b_1, W_2, b_2
    ln = 2 * (2 * d)                            # two LayerNorms (gamma, beta)
    block = attn + mlp + ln
    total = n_layer * block + vocab * d + n_ctx * d + 2 * d   # + token table, position table, final LN
    if not tied:
        total += vocab * d
    return block, total

block_gpt2, total_gpt2 = transformer_params(d=768, n_heads=12, d_ff=3072, vocab=50257, n_ctx=1024, n_layer=12)
print(f"GPT-2 small: {block_gpt2:,d} per block, {total_gpt2:,d} in total")
"""
run_solution(SOL7)
assert ns["total_gpt2"] == 124_439_808, ns["total_gpt2"]
h7a = sha_of(int(ns["block_gpt2"]))
h7b = sha_of(int(ns["total_gpt2"]))
_bx, _tx = ns["transformer_params"](1600, 25, 6400, 50257, 1024, 48)
h7c = sha_of(int(_tx))

ex.add(
    title="Counting the parameters of a transformer",
    difficulty=2,
    prompt_md=r"""
Implement `transformer_params(d, n_heads, d_ff, vocab, n_ctx, n_layer, tied=True)` returning
the tuple `(per_block, total)` of **integer** parameter counts for a GPT-2-style pre-norm
decoder with learned absolute positions:

* per block: $W_Q, W_K, W_V, W_O$ (each $d\times d$) with four bias vectors of length $d$;
  $W_1 \in \mathbb{R}^{d\times d_{\text{ff}}}$, $b_1$, $W_2 \in \mathbb{R}^{d_{\text{ff}}\times d}$, $b_2$;
  two LayerNorms with $\gamma, \beta \in \mathbb{R}^d$ each;
* total: `n_layer` blocks + token embedding ($V\times d$) + position embedding ($n_{\text{ctx}}\times d$)
  + a final LayerNorm ($2d$); if `tied=False`, add a separate unembedding matrix ($V\times d$).

Set `block_gpt2, total_gpt2 = transformer_params(768, 12, 3072, 50257, 1024, 12)`. The total
should be the famous "124M". Note that `n_heads` does not change the count — explain to
yourself why.
""",
    starter=r"""
def transformer_params(d, n_heads, d_ff, vocab, n_ctx, n_layer, tied=True):
    # YOUR CODE HERE
    return None, None

block_gpt2, total_gpt2 = None, None   # YOUR CODE HERE
""",
    solution=SOL7,
    check=r"""
check_hash('hidden: parameters per GPT-2-small block', block_gpt2 if isinstance(block_gpt2, (int, np.integer)) else None, 'HASH7A')
check_hash('hidden: total parameters of GPT-2 small', total_gpt2 if isinstance(total_gpt2, (int, np.integer)) else None, 'HASH7B')
try:
    b_toy, t_toy = transformer_params(d=8, n_heads=2, d_ff=32, vocab=10, n_ctx=4, n_layer=1)
except Exception:
    b_toy, t_toy = None, None
check('toy config d=8, d_ff=32: block has 12*64 weights + 8*4 + 32 + 8 biases + 32 LN = 872', b_toy == 872)
check('toy config total = 872 + 80 + 32 + 16 = 1000', t_toy == 1000)
try:
    _, t_untied = transformer_params(d=8, n_heads=2, d_ff=32, vocab=10, n_ctx=4, n_layer=1, tied=False)
except Exception:
    t_untied = None
check('untied unembedding adds vocab * d', t_untied == 1080)
try:
    _, t_xl = transformer_params(d=1600, n_heads=25, d_ff=6400, vocab=50257, n_ctx=1024, n_layer=48)
except Exception:
    t_xl = None
check_hash('hidden: GPT-2 XL total (d=1600, 48 layers)', t_xl if isinstance(t_xl, (int, np.integer)) else None, 'HASH7C')
""".replace("HASH7A", h7a).replace("HASH7B", h7b).replace("HASH7C", h7c),
    explanation_md=r"""
The weights per block are $4d^2$ (attention) $+ 2d\,d_{\text{ff}} = 8d^2$ (feed-forward with
$d_{\text{ff}} = 4d$), i.e. $12d^2 = 7{,}077{,}888$ for $d = 768$; biases ($4d + d_{\text{ff}} + d$)
and two LayerNorms ($4d$) add $9{,}984$, giving $7{,}087{,}872$ per block. Twelve blocks plus the
$50{,}257\times768$ token table, the $1024\times768$ position table and the final LayerNorm give
$124{,}439{,}808$. The head count never appears: splitting $W_Q$ into $h$ slices of width $d/h$
does not change its size. GPT-2 XL by the same recipe comes to about $1.56$ billion.
""",
)

# ---------------------------------------------------------------- Ex 7
SOL8 = r"""
def sinusoidal_pe(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d // 2)[None, :]
    omega = 10000.0 ** (-2 * i / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(pos * omega)
    pe[:, 1::2] = np.cos(pos * omega)
    return pe

def shift_matrix(k, d):
    omega = 10000.0 ** (-2 * np.arange(d // 2) / d)
    R = np.zeros((d, d))
    for i, w in enumerate(omega):
        c, s = np.cos(k * w), np.sin(k * w)
        R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[c, s], [-s, c]]
    return R
"""
run_solution(SOL8)

ex.add(
    title="Sinusoidal positional encodings and the relative-shift matrix",
    difficulty=2,
    prompt_md=r"""
Implement

* `sinusoidal_pe(n_pos, d)` — the `(n_pos, d)` table with $PE(pos, 2i) = \sin(pos\,\omega_i)$,
  $PE(pos, 2i+1) = \cos(pos\,\omega_i)$ and $\omega_i = 10000^{-2i/d}$ for $i = 0, \dots, d/2-1$
  (`d` is even; sines in the even columns, cosines in the odd columns);
* `shift_matrix(k, d)` — the `(d, d)` block-diagonal matrix $R_k$ with $2\times2$ blocks
  $\begin{pmatrix}\cos k\omega_i & \sin k\omega_i\\ -\sin k\omega_i & \cos k\omega_i\end{pmatrix}$,
  so that $PE(pos + k) = R_k\,PE(pos)$ for **every** `pos`.

The checks verify the guide's $d = 4$ values and the shift identity for $k = 5$ on a $d = 16$ table.
""",
    starter=r"""
def sinusoidal_pe(n_pos, d):
    # YOUR CODE HERE
    return None

def shift_matrix(k, d):
    # YOUR CODE HERE
    return None
""",
    solution=SOL8,
    check=r"""
try:
    pe4 = sinusoidal_pe(3, 4)
except Exception:
    pe4 = None
check_close('PE for d = 4, positions 0..2', pe4, np.array([[0., 1., 0., 1.], [0.841471, 0.540302, 0.01, 0.99995], [0.909297, -0.416147, 0.019999, 0.9998]]), tol=1e-5)
try:
    pe = sinusoidal_pe(40, 16); R5 = shift_matrix(5, 16)
    shifted = pe[:-5] @ R5.T          # row pos -> R_5 PE(pos)
except Exception:
    pe, R5, shifted = None, None, None
check('shapes (40, 16) and (16, 16)', pe is not None and pe.shape == (40, 16) and R5.shape == (16, 16))
check_close('PE(pos + 5) == R_5 PE(pos) for all pos', shifted, pe[5:] if pe is not None else np.zeros(1), tol=1e-9)
check('R_5 is orthogonal', R5 is not None and np.allclose(R5.T @ R5, np.eye(16)))
try:
    row_norms = np.linalg.norm(sinusoidal_pe(10, 8), axis=1)
except Exception:
    row_norms = None
check_close('every row of PE has norm sqrt(d/2)', row_norms, np.full(10, 2.0))
""",
    explanation_md=r"""
Each pair $(\sin(pos\,\omega_i), \cos(pos\,\omega_i))$ is a point on the unit circle at angle
$pos\,\omega_i$, so advancing by $k$ positions is a rotation by $k\omega_i$ — the angle-addition
formulas written as a $2\times2$ matrix. Because the rotation depends on $k$ but not on $pos$,
a learned $W_Q$, $W_K$ can match "the token $k$ steps back" with a position-independent linear
map. Every row has norm $\sqrt{d/2}$ since each pair contributes $\sin^2 + \cos^2 = 1$. A common
error is to use $\omega_i = 10000^{-i/d}$ (forgetting the factor 2) or to interleave the pairs
differently from the matrix — either breaks the shift identity.
""",
)

# ---------------------------------------------------------------- Ex 8
SOL9 = r"""
def rope(x, positions, base=10000.0):
    n_pos, d = x.shape
    theta = base ** (-2 * np.arange(d // 2) / d)
    ang = positions[:, None] * theta[None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * c - x2 * s
    out[:, 1::2] = x1 * s + x2 * c
    return out

def rope_scores(Q, K, positions):
    return rope(Q, positions) @ rope(K, positions).T / np.sqrt(Q.shape[-1])
"""
run_solution(SOL9)

ex.add(
    title="Rotary position embeddings (RoPE)",
    difficulty=2,
    prompt_md=r"""
Implement

* `rope(x, positions, base=10000.0)` — for `x` of shape `(n, d)` (`d` even) and integer
  `positions` of shape `(n,)`, rotate each coordinate pair $(x_{2i}, x_{2i+1})$ of row $m$ by
  the angle $m\,\theta_i$ with $\theta_i = \text{base}^{-2i/d}$:
  $(x_{2i}\cos - x_{2i+1}\sin,\ x_{2i}\sin + x_{2i+1}\cos)$;
* `rope_scores(Q, K, positions)` — the score matrix $\operatorname{rope}(Q)\operatorname{rope}(K)^\top/\sqrt{d}$.

The checks verify the $2$-D rotation by hand, that norms are preserved, and the defining
property: shifting **all** positions by the same amount leaves every score unchanged, while a
different offset changes them.
""",
    starter=r"""
def rope(x, positions, base=10000.0):
    # YOUR CODE HERE
    return None

def rope_scores(Q, K, positions):
    # YOUR CODE HERE
    return None
""",
    solution=SOL9,
    check=r"""
try:
    r = rope(np.array([[1., 0.]]), np.array([3]))        # d = 2: theta_0 = 1, angle = 3 rad
except Exception:
    r = None
check_close('d = 2, position 3 rotates (1,0) to (cos 3, sin 3)', r, np.array([[np.cos(3.0), np.sin(3.0)]]))
_g = np.random.default_rng(11)
_Q = _g.normal(size=(5, 8)); _K = _g.normal(size=(5, 8)); _pos = np.arange(5)
try:
    norms_ok = np.allclose(np.linalg.norm(rope(_Q, _pos), axis=1), np.linalg.norm(_Q, axis=1))
except Exception:
    norms_ok = False
check('rotation preserves the norm of every row', norms_ok)
try:
    S0 = rope_scores(_Q, _K, _pos); S_shift = rope_scores(_Q, _K, _pos + 1000)
except Exception:
    S0, S_shift = None, np.zeros(1)
check_close('shifting all positions by 1000 leaves the scores unchanged', S0, S_shift, tol=1e-8)
try:
    S_other = rope_scores(_Q, _K, np.array([0, 2, 4, 6, 8]))
    differs = not np.allclose(S0, S_other)
except Exception:
    differs = False
check('different relative offsets give different scores', differs)
try:
    same_pos = rope_scores(_Q, _K, np.zeros(5, dtype=int))
except Exception:
    same_pos = None
check_close('all positions equal -> plain scaled dot products', same_pos, _Q @ _K.T / np.sqrt(8))
""",
    explanation_md=r"""
Rotating $q$ by $m\theta$ and $k$ by $n\theta$ gives $q_m^\top k_n = q^\top R(m\theta)^\top R(n\theta)k
= q^\top R((n-m)\theta)k$, so only the offset $n - m$ survives — adding $1000$ to every position
changes nothing, while spacing the positions differently does. Rotations are orthogonal, so
norms are preserved and at position $0$ (angle $0$) RoPE is the identity. Implementations
usually write the rotation as `x * cos + rotate_half(x) * sin`; the pairing convention differs
between codebases (adjacent pairs here, first-half/second-half in Llama) but the mathematics is
the same.
""",
)

# ---------------------------------------------------------------- Ex 9
SOL10 = r"""
def kv_cache_bytes(n_layer, n_tokens, n_kv_heads, d_head, bytes_per_number=2):
    return 2 * n_layer * n_tokens * n_kv_heads * d_head * bytes_per_number

def max_sequences(gpu_bytes, weight_bytes, n_layer, n_tokens, n_kv_heads, d_head, bytes_per_number=2):
    per_seq = kv_cache_bytes(n_layer, n_tokens, n_kv_heads, d_head, bytes_per_number)
    return int((gpu_bytes - weight_bytes) // per_seq)

cache_7b = kv_cache_bytes(n_layer=32, n_tokens=8192, n_kv_heads=32, d_head=128)
cache_gqa = kv_cache_bytes(n_layer=48, n_tokens=32768, n_kv_heads=8, d_head=128)
print(f"7B-class, 8k context, fp16: {cache_7b / 2**30:.2f} GiB;  48-layer GQA model at 32k: {cache_gqa / 2**30:.2f} GiB")
"""
run_solution(SOL10)
assert ns["cache_7b"] == 2**32
h10a = sha_of(int(ns["cache_gqa"]))
h10b = sha_of(int(ns["max_sequences"](80 * 10**9, 2 * 6_740_000_000, 32, 4096, 8, 128)))

ex.add(
    title="KV-cache memory arithmetic",
    difficulty=2,
    prompt_md=r"""
Implement

* `kv_cache_bytes(n_layer, n_tokens, n_kv_heads, d_head, bytes_per_number=2)` — the bytes
  needed to cache keys **and** values for one sequence: $2 \cdot L \cdot t \cdot h_{kv} \cdot d_{\text{head}} \cdot \text{bytes}$
  (return an `int`);
* `max_sequences(gpu_bytes, weight_bytes, n_layer, n_tokens, n_kv_heads, d_head, bytes_per_number=2)` —
  how many full-length sequences fit in the memory left after the weights (floor division, `int`).

Then set `cache_7b = kv_cache_bytes(32, 8192, 32, 128)` (should be exactly 4 GiB) and
`cache_gqa = kv_cache_bytes(48, 32768, 8, 128)` (hidden). The second hidden check calls
`max_sequences` with an 80 GB GPU ($80\times10^9$ bytes), a 6.74B-parameter model in fp16, and
a 4096-token context with 8 KV heads of width 128.
""",
    starter=r"""
def kv_cache_bytes(n_layer, n_tokens, n_kv_heads, d_head, bytes_per_number=2):
    # YOUR CODE HERE
    return None

def max_sequences(gpu_bytes, weight_bytes, n_layer, n_tokens, n_kv_heads, d_head, bytes_per_number=2):
    # YOUR CODE HERE
    return None

cache_7b = None    # YOUR CODE HERE
cache_gqa = None   # YOUR CODE HERE
""",
    solution=SOL10,
    check=r"""
check('cache_7b is exactly 4 GiB = 2**32 bytes', cache_7b == 2 ** 32)
try:
    gpt2 = kv_cache_bytes(12, 1024, 12, 64)
except Exception:
    gpt2 = None
check('GPT-2 small at 1024 tokens: 36 MiB', gpt2 == 36 * 2 ** 20)
try:
    int8 = kv_cache_bytes(32, 8192, 32, 128, bytes_per_number=1)
except Exception:
    int8 = None
check('an 8-bit cache halves the bytes', int8 == 2 ** 31)
check_hash('hidden: cache_gqa (48 layers, 32k tokens, 8 KV heads of 128, fp16)', cache_gqa if isinstance(cache_gqa, (int, np.integer)) else None, 'HASH10A')
try:
    n_seq = max_sequences(80 * 10**9, 2 * 6_740_000_000, 32, 4096, 8, 128)
except Exception:
    n_seq = None
check_hash('hidden: sequences that fit next to a 6.74B fp16 model on 80 GB', n_seq if isinstance(n_seq, (int, np.integer)) else None, 'HASH10B')
""".replace("HASH10A", h10a).replace("HASH10B", h10b),
    explanation_md=r"""
Each layer stores one key vector and one value vector per token, each of length
$h_{kv}d_{\text{head}}$ ($= d$ for ordinary multi-head attention), hence the factor $2$. For
$L = 32$, $t = 8192$, $d = 4096$ in fp16: $2\cdot32\cdot8192\cdot4096\cdot2 = 2^{32}$ bytes, i.e.
exactly $4$ GiB. Grouped-query attention shrinks $h_{kv}$ and therefore the cache
proportionally, which is why a 48-layer GQA model at 32k context needs only $6$ GiB. For the
serving question, $80\times10^9 - 13.48\times10^9 = 66.5$ GB remain, and each 4096-token sequence
with 8 KV heads costs $2\cdot32\cdot4096\cdot1024\cdot2 = 512$ MiB, so $123$ sequences fit —
the cache, not the weights, sets the batch size.
""",
)

# ---------------------------------------------------------------- Ex 10 (stretch)
SOL11 = r"""
def gelu(u):
    return 0.5 * u * (1.0 + erf(u / np.sqrt(2.0)))

def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    n, d = X.shape
    d_head = d // n_heads
    split = lambda M: M.reshape(n, n_heads, d_head).transpose(1, 0, 2)
    Qh, Kh, Vh = split(X @ W_Q), split(X @ W_K), split(X @ W_V)
    S = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(d_head)
    if mask is not None:
        S = np.where(mask[None], -np.inf, S)
    P = softmax(S, axis=-1)
    return (P @ Vh).transpose(1, 0, 2).reshape(n, d) @ W_O

def transformer_block(X, p, n_heads, mask=None):
    h1 = layer_norm(X, p["g1"], p["b1"])
    X = X + multi_head_attention(h1, p["W_Q"], p["W_K"], p["W_V"], p["W_O"], n_heads, mask)
    h2 = layer_norm(X, p["g2"], p["b2"])
    return X + gelu(h2 @ p["W_1"] + p["bf1"]) @ p["W_2"] + p["bf2"]
"""
run_solution(SOL11)

ex.add(
    title="A complete pre-norm transformer block",
    difficulty=3,
    prompt_md=r"""
Using your `layer_norm` from Exercise 3 and the provided `softmax`, implement

* `gelu(u)` $= u\,\Phi(u) = \tfrac12 u\,(1 + \operatorname{erf}(u/\sqrt2))$ (use `scipy.special.erf`, imported in the setup);
* `multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=None)` — split the `(n, d)`
  projections into `n_heads` heads of width `d // n_heads`, run scaled dot-product attention
  per head (scale by $\sqrt{d_{\text{head}}}$, apply the boolean `mask` with `-np.inf` if given),
  concatenate, and multiply by `W_O`; return the `(n, d)` output;
* `transformer_block(X, p, n_heads, mask=None)` — the pre-norm block
  $X \leftarrow X + \operatorname{MHA}(\operatorname{LN}_1 X)$, then
  $X \leftarrow X + \operatorname{GELU}(\operatorname{LN}_2(X)\,W_1 + b_{f1})\,W_2 + b_{f2}$,
  with parameters in the dict `p` under the keys
  `W_Q, W_K, W_V, W_O, W_1, bf1, W_2, bf2, g1, b1, g2, b2` (`g`/`b` are the LayerNorm gains and biases).

The checks compare against a reference on random weights, verify that one head equals plain
attention, that the block is causal when masked, and that zero attention/FFN output weights
make the block the identity.
""",
    starter=r"""
def gelu(u):
    # YOUR CODE HERE
    return None

def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    # YOUR CODE HERE
    return None

def transformer_block(X, p, n_heads, mask=None):
    # YOUR CODE HERE
    return None
""",
    solution=SOL11,
    check=r"""
try:
    g_vals = gelu(np.array([-1., 0., 1., 2.]))
except Exception:
    g_vals = None
check_close('GELU(-1, 0, 1, 2)', g_vals, np.array([-0.158655, 0.0, 0.841345, 1.9545]), tol=1e-5)
_g = np.random.default_rng(17)
_n, _d, _h, _dff = 6, 16, 4, 64
_X = _g.normal(size=(_n, _d))
_p = dict(W_Q=_g.normal(size=(_d, _d)) / 4, W_K=_g.normal(size=(_d, _d)) / 4, W_V=_g.normal(size=(_d, _d)) / 4, W_O=_g.normal(size=(_d, _d)) / 4,
          W_1=_g.normal(size=(_d, _dff)) / 4, bf1=_g.normal(size=_dff) * 0.1, W_2=_g.normal(size=(_dff, _d)) / 8, bf2=_g.normal(size=_d) * 0.1,
          g1=1 + 0.1 * _g.normal(size=_d), b1=0.1 * _g.normal(size=_d), g2=1 + 0.1 * _g.normal(size=_d), b2=0.1 * _g.normal(size=_d))
def _ref_mha(X, p, n_heads, mask):
    n, d = X.shape; dh = d // n_heads; outs = []
    for i in range(n_heads):
        sl = slice(i * dh, (i + 1) * dh)
        S = (X @ p["W_Q"][:, sl]) @ (X @ p["W_K"][:, sl]).T / np.sqrt(dh)
        if mask is not None:
            S = np.where(mask, -np.inf, S)
        outs.append(softmax(S, axis=-1) @ (X @ p["W_V"][:, sl]))
    return np.concatenate(outs, axis=1) @ p["W_O"]
def _ref_block(X, p, n_heads, mask):
    ln = lambda Z, g, b: g * (Z - Z.mean(-1, keepdims=True)) / np.sqrt(Z.var(-1, keepdims=True) + 1e-5) + b
    X = X + _ref_mha(ln(X, p["g1"], p["b1"]), p, n_heads, mask)
    H = ln(X, p["g2"], p["b2"]) @ p["W_1"] + p["bf1"]
    return X + 0.5 * H * (1 + erf(H / np.sqrt(2))) @ p["W_2"] + p["bf2"]
try:
    mha = multi_head_attention(_X, _p["W_Q"], _p["W_K"], _p["W_V"], _p["W_O"], _h)
except Exception:
    mha = None
check_close('multi-head attention matches a per-head loop reference (4 heads)', mha, _ref_mha(_X, _p, _h, None), tol=1e-8)
try:
    one_head = multi_head_attention(_X, _p["W_Q"], _p["W_K"], _p["W_V"], _p["W_O"], 1)
    plain = softmax((_X @ _p["W_Q"]) @ (_X @ _p["W_K"]).T / np.sqrt(_d), axis=-1) @ (_X @ _p["W_V"]) @ _p["W_O"]
except Exception:
    one_head, plain = None, np.zeros(1)
check_close('one head equals plain attention times W_O', one_head, plain, tol=1e-8)
_mask = np.triu(np.ones((_n, _n), dtype=bool), 1)
try:
    blk = transformer_block(_X, _p, _h, _mask)
except Exception:
    blk = None
check_close('causal pre-norm block matches the reference', blk, _ref_block(_X, _p, _h, _mask), tol=1e-8)
try:
    _X2 = _X.copy(); _X2[-1] += 3.0
    blk2 = transformer_block(_X2, _p, _h, _mask)
    causal_ok = np.allclose(blk2[:-1], blk[:-1]) and not np.allclose(blk2[-1], blk[-1])
except Exception:
    causal_ok = False
check('masked block is causal: editing the last token changes only the last row', causal_ok)
try:
    _p0 = dict(_p); _p0["W_O"] = np.zeros((_d, _d)); _p0["W_2"] = np.zeros((_dff, _d)); _p0["bf2"] = np.zeros(_d)
    ident = transformer_block(_X, _p0, _h)
except Exception:
    ident = None
check_close('zero output weights make the block the identity (residual path)', ident, _X)
""",
    explanation_md=r"""
Reshaping the `(n, d)` projections to `(n, h, d_head)` and moving the head axis first lets one
batched matmul compute all heads' scores at once; the heads are concatenated by reversing the
reshape before $W_O$. The pre-norm block normalises *before* each sub-layer and adds the result
to the untouched residual stream — which is why zeroing $W_O$, $W_2$ and $b_{f2}$ reduces the
block to the identity, and why gradients always have a clean path through deep stacks.
Causality is inherited from the mask alone: LayerNorm and the FFN act on each row separately.
Frequent errors: scaling by $\sqrt{d}$ instead of $\sqrt{d_{\text{head}}}$, normalising after the
residual add (post-norm), and transposing the wrong axes when re-merging heads.
""",
)

# ---------------------------------------------------------------- Ex 11 (conceptual)
ex.add_conceptual(
    title="Diagnose: 'temperature makes the sqrt(d_k) unnecessary'",
    difficulty=2,
    prompt_md=r"""
A colleague argues: "Dividing attention scores by $\sqrt{d_k}$ is just a fixed temperature, and
the model can learn any temperature it likes by scaling $W_Q$ — so the constant is cosmetic and I
removed it from my implementation with $d_k = 128$." Training is now far slower and sometimes
diverges.

Explain what is right and what is wrong in the argument. Use the variance of the scores at
initialisation, the softmax Jacobian, and the size of the gradient that reaches $W_Q$ and $W_K$
to say *why* the learning dynamics changed even though the function class did not.
""",
    answer_md=r"""
The colleague is right that the function class is unchanged: $\operatorname{softmax}(QK^\top/\sqrt{d_k})$
with $W_Q$ equals $\operatorname{softmax}(QK^\top)$ with $W_Q/\sqrt{d_k}$, so any attention pattern
reachable with the constant is reachable without it. The argument fails on the *dynamics*. At
initialisation the components of $q$ and $k$ have roughly unit variance, so a raw score
$q^\top k$ has variance $d_k = 128$ and standard deviation $\approx 11$; the softmax of a row
whose entries differ by about $11$ is essentially one-hot. The softmax Jacobian
$\operatorname{diag}(p) - pp^\top$ then has entries of order $p_{\max}(1 - p_{\max}) \approx e^{-11}$,
and the gradient reaching the scores — and through them $W_Q$ and $W_K$ — is multiplied by that
factor. The weights that would have to shrink by $\sqrt{128}$ to fix the problem receive
almost no gradient to do so, so training stalls; where some rows are not saturated the gradients
are large and uneven, which produces the instability. The $\sqrt{d_k}$ places the model, *at the
start*, in the regime where gradients flow; a "learned temperature" can only be learned if the
gradient signal exists. The same reasoning is why initialisation scales (Module 16) matter even
though they, too, are "just a reparametrisation".
""",
)

# ---------------------------------------------------------------- Ex 12 (project)
PROJECT_DATA = r"""
# ---- toy task: sequences that repeat a random 5-token pattern; the model must learn "copy the token 4 steps back" ----
V_TOK, PERIOD, N_POS, D_MODEL = 8, 5, 20, 16
g = np.random.default_rng(0)

def make_batch(batch_size, g):
    pats = np.stack([g.permutation(V_TOK)[:PERIOD] for _ in range(batch_size)])     # a fresh pattern per sequence
    seq = np.concatenate([pats] * (N_POS // PERIOD + 2), axis=1)[:, :N_POS + 1]
    return seq[:, :-1], seq[:, 1:]                                                    # inputs (B, n), targets (B, n)

SCORED = np.arange(N_POS) >= PERIOD - 1       # from position 4 on, the next token is determined by the past
MASK = np.triu(np.ones((N_POS, N_POS), dtype=bool), 1)
tok, tgt = make_batch(2, g)
print("example input :", tok[0]); print("example target:", tgt[0]); print("positions scored:", np.flatnonzero(SCORED))
"""

ex.add_project(
    title="Mini-project: train a single-head causal transformer with hand-derived gradients",
    difficulty=3,
    prompt_md=r"""
Build the smallest possible language model — token embedding $E$ + learned position table $P$,
one causal single-head attention layer ($W_Q, W_K, W_V$, no $W_O$), and an unembedding $W_U$ —
and train it with **gradients you derive yourself** (no autograd) on sequences that repeat a
random 5-token pattern. The model can only solve the task by learning to attend exactly 4
positions back and copy, so the attention pattern it learns is checkable.

1. Forward pass for a batch `tok` of shape `(B, n)`: $X = E[\text{tok}] + P$;
   $Q, K, V = XW_Q, XW_K, XW_V$; $S = QK^\top/\sqrt{d}$ with the causal mask; $P_{\text{att}} = \operatorname{softmax}(S)$;
   $O = P_{\text{att}}V$; logits $= OW_U$; loss = mean cross-entropy over the **scored** positions
   (`SCORED`, where the next token is predictable).
2. Backward pass by hand: $\partial\text{logits} = (p - y)/m$ on scored positions (Module 16);
   $\partial W_U = O^\top\partial\text{logits}$; $\partial O = \partial\text{logits}\,W_U^\top$;
   $\partial V = P_{\text{att}}^\top\partial O$; $\partial P_{\text{att}} = \partial O\,V^\top$;
   $\partial S = P_{\text{att}}\odot(\partial P_{\text{att}} - \operatorname{rowsum}(\partial P_{\text{att}}\odot P_{\text{att}}))$;
   $\partial Q = \partial S\,K/\sqrt d$; $\partial K = \partial S^\top Q/\sqrt d$; then
   $\partial W_Q = X^\top\partial Q$ etc., $\partial X = \partial Q\,W_Q^\top + \partial K\,W_K^\top + \partial V\,W_V^\top$,
   $\partial P = \sum_{\text{batch}}\partial X$, and $\partial E$ by scattering $\partial X$ into the rows of the tokens used (`np.add.at`).
3. **Gradient-check** every parameter against central finite differences before training.
4. Train with Adam (Module 05) for about 1000 steps, batch 16, learning rate $10^{-2}$; plot
   the loss; report next-token accuracy on fresh sequences.
5. Plot the learned attention matrix for one sequence and verify that each scored query puts
   most of its weight on a position `PERIOD - 1 = 4` steps back — or 9 or 14 steps back, which
   carry the same token in a period-5 sequence. Measure the total weight on those positions.

Keep the run under 60 seconds (the reference takes a few seconds). The cell below defines the
task and the data generator.
""",
    rubric_md=r"""
- Every analytic gradient matches central finite differences to a relative error below $10^{-5}$.
- The training loss falls from about $\ln 8 \approx 2.08$ to below $0.05$ and the next-token
  accuracy on fresh sequences is above 0.98.
- The learned attention of scored positions concentrates (total weight $> 0.8$) on the positions
  $4, 9, 14, \dots$ steps back — all congruent to $i - 4 \pmod 5$ and all carrying the right
  token; the plot has a title, axis labels and a colour bar. You can explain why the model is
  free to pick any of them and why it may pick different ones at different positions.
- You can explain in two sentences why the model needs the position table to solve this task
  (a bare attention layer is permutation-equivariant and cannot say "4 steps back"), and
  predict what happens if you remove `P` — then test it.
- You can point to the line in your backward pass that implements the softmax Jacobian and
  explain why it is $P\odot(\cdot - \text{rowsum})$ rather than a full $n\times n\times n$ tensor.
- (Bonus) Replace the learned position table by RoPE on $Q$ and $K$ (Exercise 8) and confirm the
  model still learns the offset-4 pattern.
""",
    starter=PROJECT_DATA + r"""

# YOUR CODE HERE
# def init_params(g): ...                       E (V, d), P (n, d), W_Q, W_K, W_V (d, d), W_U (d, V)
# def forward(params, tok): ...                 return a dict of intermediates (X, Q, K, V, P_att, O, logits)
# def loss_and_grads(params, tok, tgt): ...     hand-derived gradients for every parameter
# gradient check, Adam training loop, loss plot, accuracy on fresh data, attention heat-map
""",
    solution_sketch_md=r"""
Store intermediates in the forward pass and walk them backwards. The only non-trivial step is the
softmax Jacobian applied to each attention row, `dS = P_att * (dP - (dP * P_att).sum(-1, keepdims=True))`,
and the scatter-add into $E$ with `np.add.at(dE, tok.ravel(), dX.reshape(-1, d))`. The gradient
check agrees to about $10^{-8}$ relative error; Adam at learning rate $10^{-2}$ drives the loss
below $10^{-3}$ within a few hundred steps and the accuracy on fresh sequences to $1.0$. The
attention heat-map shows diagonal bands at offsets 4, 9 and 14: query position $i$ attends to a
key $i - 4$, $i - 9$ or $i - 14$, whose *next* token is in every case the target — the task does
not distinguish between these copy sources, so the model settles on whichever the optimiser
reached first for each query position, and the total weight on the valid set is close to 1.
Without the position table the loss plateaus near $\ln 8$ — the model can see which tokens are
present but not where.
""",
    solution_code=PROJECT_DATA + r"""

def init_params(g, scale=0.1):
    return dict(E=scale * g.normal(size=(V_TOK, D_MODEL)), P=scale * g.normal(size=(N_POS, D_MODEL)),
                W_Q=scale * g.normal(size=(D_MODEL, D_MODEL)), W_K=scale * g.normal(size=(D_MODEL, D_MODEL)),
                W_V=scale * g.normal(size=(D_MODEL, D_MODEL)), W_U=scale * g.normal(size=(D_MODEL, V_TOK)))

def forward(p, tok):
    X = p["E"][tok] + p["P"][None, :, :]                       # (B, n, d)
    Q, K, Vv = X @ p["W_Q"], X @ p["W_K"], X @ p["W_V"]
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(D_MODEL)
    S = np.where(MASK[None], -np.inf, S)
    P_att = softmax(S, axis=-1)
    O = P_att @ Vv
    return dict(X=X, Q=Q, K=K, V=Vv, P_att=P_att, O=O, logits=O @ p["W_U"])

def loss_and_grads(p, tok, tgt):
    c = forward(p, tok)
    B = tok.shape[0]
    probs = softmax(c["logits"], axis=-1)                     # (B, n, V)
    Y = np.zeros_like(probs); Y[np.arange(B)[:, None], np.arange(N_POS)[None, :], tgt] = 1.0
    w = SCORED[None, :, None].astype(float); m = B * SCORED.sum()
    loss = -(w * Y * np.log(probs + 1e-12)).sum() / m
    dlogits = w * (probs - Y) / m                             # softmax + cross-entropy: (p - y) / m
    gr = {}
    gr["W_U"] = np.einsum("bnd,bnv->dv", c["O"], dlogits)
    dO = dlogits @ p["W_U"].T
    dV = c["P_att"].transpose(0, 2, 1) @ dO
    dP = dO @ c["V"].transpose(0, 2, 1)
    dS = c["P_att"] * (dP - (dP * c["P_att"]).sum(axis=-1, keepdims=True))   # softmax Jacobian, row by row
    dQ = dS @ c["K"] / np.sqrt(D_MODEL)
    dK = dS.transpose(0, 2, 1) @ c["Q"] / np.sqrt(D_MODEL)
    X = c["X"]
    gr["W_Q"] = np.einsum("bnd,bne->de", X, dQ); gr["W_K"] = np.einsum("bnd,bne->de", X, dK); gr["W_V"] = np.einsum("bnd,bne->de", X, dV)
    dX = dQ @ p["W_Q"].T + dK @ p["W_K"].T + dV @ p["W_V"].T
    gr["P"] = dX.sum(axis=0)
    gr["E"] = np.zeros_like(p["E"]); np.add.at(gr["E"], tok.ravel(), dX.reshape(-1, D_MODEL))
    acc = float((probs.argmax(-1) == tgt)[:, SCORED].mean())
    return loss, gr, acc

# ---- gradient check
params = init_params(g); tok, tgt = make_batch(2, g)
loss0, grads, _ = loss_and_grads(params, tok, tgt)
print(f"initial loss {loss0:.4f} (ln 8 = {np.log(8):.4f})")
worst = 0.0
for name in params:
    for _ in range(3):
        idx = tuple(g.integers(0, s) for s in params[name].shape)
        old = params[name][idx]; eps = 1e-5
        params[name][idx] = old + eps; lp = loss_and_grads(params, tok, tgt)[0]
        params[name][idx] = old - eps; lm = loss_and_grads(params, tok, tgt)[0]
        params[name][idx] = old
        fd = (lp - lm) / (2 * eps)
        worst = max(worst, abs(grads[name][idx] - fd) / max(1e-8, abs(fd)))
print(f"gradient check: worst relative error over 18 random entries = {worst:.2e}")

# ---- Adam
params = init_params(g)
m1 = {k: np.zeros_like(v) for k, v in params.items()}; m2 = {k: np.zeros_like(v) for k, v in params.items()}
lr, b1, b2, eps_adam = 1e-2, 0.9, 0.999, 1e-8
history = []
for step in range(1, 1001):
    tok, tgt = make_batch(16, g)
    loss, grads, acc = loss_and_grads(params, tok, tgt)
    for k in params:
        m1[k] = b1 * m1[k] + (1 - b1) * grads[k]; m2[k] = b2 * m2[k] + (1 - b2) * grads[k] ** 2
        params[k] -= lr * (m1[k] / (1 - b1 ** step)) / (np.sqrt(m2[k] / (1 - b2 ** step)) + eps_adam)
    history.append(loss)
    if step % 200 == 0:
        print(f"step {step:4d}: loss {loss:.4f}, batch accuracy {acc:.3f}")

tok_test, tgt_test = make_batch(200, np.random.default_rng(123))
_, _, acc_test = loss_and_grads(params, tok_test, tgt_test)
print(f"next-token accuracy on 200 fresh sequences: {acc_test:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].semilogy(history); axes[0].set_title("Training loss (next-token cross-entropy)"); axes[0].set_xlabel("step"); axes[0].set_ylabel("loss (nats)")
c = forward(params, tok_test[:1])
im = axes[1].imshow(c["P_att"][0], cmap="Blues", vmin=0, vmax=1)
axes[1].set_title("Learned causal attention: query i -> key i - 4 (mod 5)"); axes[1].set_xlabel("key position"); axes[1].set_ylabel("query position")
plt.colorbar(im, ax=axes[1], fraction=0.046); plt.tight_layout(); plt.show()

P_learned = c["P_att"][0]
offsets = np.arange(N_POS)[SCORED] - P_learned[SCORED].argmax(axis=1)
qpos, kpos = np.meshgrid(np.arange(N_POS), np.arange(N_POS), indexing="ij")
valid = (kpos < qpos) & ((qpos - kpos) % PERIOD == PERIOD - 1)        # offsets 4, 9, 14, ... all carry the right token
weight_valid = (P_learned * valid).sum(axis=1)[SCORED]
print("argmax offset (query - key) for scored positions:", offsets, "-> all congruent to 4 mod 5:", bool(np.all(offsets % PERIOD == PERIOD - 1)))
print(f"total attention weight on valid copy sources (offsets 4, 9, 14): mean {weight_valid.mean():.3f}, min {weight_valid.min():.3f}")
""",
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print("exercises:", len(ex.items), "| code:", kinds.count("code"), "| concept:", kinds.count("concept"),
      "| project:", kinds.count("project"))
print("difficulties:", [it["difficulty"] for it in ex.items])
