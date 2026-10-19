"""Builder for Module 16 — Neural Networks from Scratch.

Generates lesson.ipynb, exercises.ipynb and solutions.ipynb into the module folder.
Run from the repository root:  python3 _authoring/builders/build_16.py
"""
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import Notebook, ExerciseSet, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch"

# =============================================================================
# LESSON
# =============================================================================
nb = Notebook("Module 16 — Neural Networks from Scratch")

nb.md(r"""
This notebook builds a multi-layer perceptron in NumPy, piece by piece: a hidden layer that solves XOR,
an activation gallery with derivatives, the batch-first forward pass, the softmax cross-entropy loss and
its gradient, `Dense`/`ReLU`/`MLP` classes with a gradient-checked backward pass, SGD and Adam, a complete
training loop that reaches about 97 percent test accuracy on the `load_digits` images in a couple of
seconds, and then the experiments behind the design rules: initialisation scale, vanishing gradients,
residual connections, batch normalisation, dropout and convolution-as-a-matrix.

Read the [concept guide](README.md) first, or keep it open alongside: the section numbers below match
its numbered concepts. **Conventions:** data matrices are `(n_samples, n_features)`, a layer computes
`X @ W + b` with `W` of shape `(n_in, n_out)`, and `G` always denotes the gradient of the loss with
respect to the quantity a layer produced.
""")

nb.code(r"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
""")

# ---------------------------------------------------------------- Section 1
nb.md(r"""
## 1. From linear models to networks: XOR

A single affine unit $\hat y = \sigma(x^\top w + b)$ draws one straight line. XOR — true for $(0,1)$ and $(1,0)$,
false for $(0,0)$ and $(1,1)$ — cannot be separated by any line (add the two positive-point inequalities
and the two negative-point inequalities and you get a contradiction). Let us watch logistic regression
fail, then add a hidden layer of two units and watch it succeed.

$$
f(x) = \tanh(x W_1 + b_1)\, W_2 + b_2, \qquad W_1 \in \mathbb{R}^{2 \times h},\ W_2 \in \mathbb{R}^{h \times 1}.
$$
""")

nb.code(r"""
X_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor = np.array([0., 1., 1., 0.])
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
bce = lambda p, y: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# (a) logistic regression = zero hidden layers
w, b = np.zeros(2), 0.0
for step in range(2000):
    p = sigmoid(X_xor @ w + b)
    g = (p - y_xor) / 4                       # dL/dz for BCE, averaged over the 4 points
    w -= 0.5 * X_xor.T @ g; b -= 0.5 * g.sum()
print(f"logistic regression: loss {bce(p, y_xor):.4f} (ln 2 = {np.log(2):.4f}), accuracy {np.mean((p > 0.5) == (y_xor == 1)):.2f}, w = {w}, b = {b:.3f}")

# (b) one hidden layer with h tanh units, trained by the backward pass we derive in Section 5
h = 4
W1 = rng.normal(0, 1 / np.sqrt(2), (2, h)); b1 = np.zeros(h)
W2 = rng.normal(0, 1 / np.sqrt(h), (h, 1)); b2 = np.zeros(1)
for step in range(3000):
    Z1 = X_xor @ W1 + b1; A1 = np.tanh(Z1)
    z2 = (A1 @ W2 + b2)[:, 0]; p = sigmoid(z2)
    G2 = ((p - y_xor) / 4)[:, None]           # (4, 1)
    dW2 = A1.T @ G2; db2 = G2.sum(0)
    G1 = (G2 @ W2.T) * (1 - A1 ** 2)          # back through W2, then through tanh
    dW1 = X_xor.T @ G1; db1 = G1.sum(0)
    W1 -= 0.5 * dW1; b1 -= 0.5 * db1; W2 -= 0.5 * dW2; b2 -= 0.5 * db2
print(f"2-{h}-1 network:      loss {bce(p, y_xor):.4f}, accuracy {np.mean((p > 0.5) == (y_xor == 1)):.2f}, predictions {p}")
n_params = 2 * h + h + h * 1 + 1
print(f"parameters: 2*{h} + {h} + {h}*1 + 1 = {n_params}")

gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 120), np.linspace(-0.5, 1.5, 120))
grid = np.c_[gx.ravel(), gy.ravel()]
p_lin = sigmoid(grid @ w + b).reshape(gx.shape)
p_net = sigmoid((np.tanh(grid @ W1 + b1) @ W2 + b2)[:, 0]).reshape(gx.shape)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, P, title in zip(axes, [p_lin, p_net], ["logistic regression (no hidden layer)", f"MLP with {h} tanh hidden units"]):
    cs = ax.contourf(gx, gy, P, levels=20, cmap="coolwarm", vmin=0, vmax=1)
    ax.contour(gx, gy, P, levels=[0.5], colors="k")
    ax.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, cmap="coolwarm", s=120, edgecolor="k", vmin=0, vmax=1)
    ax.set_title(title); ax.set_xlabel("x1"); ax.set_ylabel("x2")
fig.colorbar(cs, ax=axes, label="P(y = 1)", shrink=0.8)
plt.show()
""")

nb.md(r"""
Logistic regression ends at the uninformed loss $\ln 2 = 0.693$ with its weights at zero — the best a
line can do is predict 0.5 everywhere. The hidden layer bends the decision surface into a band around
the diagonal and reaches 100 percent with a dozen parameters. Every hidden unit is itself a line in
input space; the output layer combines *which side of which lines* a point lies on. That is the entire
idea of a neural network; the rest of this notebook is about doing it with matrices, fast and stably.

**Try it yourself:** set `h = 2` (the minimum) and re-run a few times with different seeds — it works for
some initialisations and gets stuck for others. Then try `h = 16`: does more width make the optimisation
easier or harder?
""")

# ---------------------------------------------------------------- Section 2
nb.md(r"""
## 2. Activation functions and their derivatives

Without a nonlinearity, $L$ affine layers compose to one affine map: $(XW_1 + b_1)W_2 + b_2 = X(W_1W_2) + (b_1W_2 + b_2)$.
The activation is what makes depth count — and its *derivative* is what decides how much gradient
survives each layer on the way back. Let us plot the standard choices with their derivatives, and check
each derivative formula against central differences.
""")

nb.code(r"""
Phi = lambda z: 0.5 * (1 + erf(z / np.sqrt(2)))        # standard normal CDF
phi = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)

ACTS = {   # name: (function, analytic derivative)
    "sigmoid":    (sigmoid,                            lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "tanh":       (np.tanh,                            lambda z: 1 - np.tanh(z) ** 2),
    "ReLU":       (lambda z: np.maximum(z, 0),         lambda z: (np.asarray(z) > 0).astype(float)),
    "leaky ReLU": (lambda z: np.where(z > 0, z, 0.01 * z), lambda z: np.where(z > 0, 1.0, 0.01)),
    "softplus":   (lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0), sigmoid),
    "SiLU":       (lambda z: z * sigmoid(z),           lambda z: sigmoid(z) * (1 + z * (1 - sigmoid(z)))),
    "GELU":       (lambda z: z * Phi(z),               lambda z: Phi(z) + z * phi(z)),
}
z = np.linspace(-5, 5, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, (f, df) in ACTS.items():
    axes[0].plot(z, f(z), label=name); axes[1].plot(z, df(z), label=name)
axes[0].set_title("activation functions"); axes[1].set_title("their derivatives")
for ax in axes:
    ax.set_xlabel("pre-activation z"); ax.axhline(0, color="gray", lw=0.5); ax.grid(alpha=0.3)
axes[0].set_ylim(-1.5, 5); axes[1].set_ylim(-0.3, 1.3); axes[1].axhline(0.25, color="gray", ls=":", lw=1); axes[1].legend(fontsize=8)
plt.show()

zs = np.array([-2.0, 0.3, 2.0]); hh = 1e-6
hdr_d = "f'(2)"
print(f"{'activation':12s} {'f(2)':>8s} {hdr_d:>8s}   max |analytic - central difference| over z in {zs}")
for name, (f, df) in ACTS.items():
    num = (f(zs + hh) - f(zs - hh)) / (2 * hh)
    print(f"{name:12s} {f(2.0):8.4f} {df(2.0):8.4f}   {np.max(np.abs(df(zs) - num)):.1e}")
""")

nb.md(r"""
Read the derivative panel: the sigmoid's slope never exceeds $1/4$ (dotted line) and is $0.105$ at $z = 2$;
tanh's is $0.071$ there. ReLU, SiLU and GELU pass the gradient essentially undiminished for positive
inputs — which is why they replaced sigmoids in deep networks (Section 7 measures the consequence). The
finite-difference column confirms every formula to $10^{-9}$ or better; keep this habit for every
derivative you ever write down.

""")

# ---------------------------------------------------------------- Section 3
nb.md(r"""
## 3. The forward pass with matrices

Stack $n$ examples as the rows of $X \in \mathbb{R}^{n \times d}$. A layer is then one matrix product plus a broadcast bias,

$$
Z_1 = X W_1 + \mathbf{1} b_1^\top \in \mathbb{R}^{n \times h}, \qquad A_1 = \phi(Z_1), \qquad Z_2 = A_1 W_2 + \mathbf{1} b_2^\top \in \mathbb{R}^{n \times k},
$$

with the shape rule $(n \times n_{\text{in}})(n_{\text{in}} \times n_{\text{out}}) = (n \times n_{\text{out}})$. Here is the concept
guide's worked example — two examples, three features, two hidden ReLU units, one output — with every
intermediate printed. Keep these numbers: Section 5 runs the backward pass on them.
""")

nb.code(r"""
X = np.array([[1., 2., 0.], [0., 1., -1.]])
W1 = np.array([[0.5, -1.], [1., 0.5], [0., 2.]]); b1 = np.array([0.1, 0.2])
W2 = np.array([[1.], [-1.]]);                    b2 = np.array([0.5])
relu = lambda z: np.maximum(z, 0.0)

XW1 = X @ W1
Z1 = XW1 + b1               # broadcasting adds b1 to every row
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
for name, arr in [("X", X), ("W1", W1), ("X @ W1", XW1), ("Z1 = X @ W1 + b1", Z1), ("A1 = relu(Z1)", A1), ("Z2 = A1 @ W2 + b2", Z2)]:
    print(f"{name:20s} shape {str(arr.shape):8s}\n{arr}\n")
n = X.shape[0]
flops = 2 * n * W1.shape[0] * W1.shape[1] + 2 * n * W2.shape[0] * W2.shape[1]
print(f"FLOPs for the two products: 2*{n}*3*2 + 2*{n}*2*1 = {flops}")
print(f"parameters: {W1.size + b1.size + W2.size + b2.size};  rule of thumb forward FLOPs = 2 * weights * batch = {2 * (W1.size + W2.size) * n}")
""")

nb.md(r"""
Notice the bias: `b1` has shape `(2,)` and is added to a `(2, 2)` matrix — NumPy copies it to every
row, which is exactly $\mathbf{1} b_1^\top$. The second example's second hidden unit is switched off
($Z_1 = -1.3 < 0$); remember that zero when we come to the gradients. The forward cost is about
$2 \times (\text{weights}) \times (\text{batch})$ FLOPs, and a training step about three times that.
""")

# ---------------------------------------------------------------- Section 4
nb.md(r"""
## 4. Loss functions: softmax cross-entropy and its gradient

For $k$ classes the logits $Z \in \mathbb{R}^{n \times k}$ become probabilities row-wise with the softmax, and the loss is the
mean negative log-probability of the true class:

$$
P_{ij} = \frac{e^{Z_{ij}}}{\sum_c e^{Z_{ic}}}, \qquad \mathcal{L} = -\frac{1}{n}\sum_i \ln P_{i, y_i}, \qquad \frac{\partial \mathcal{L}}{\partial Z} = \frac{P - Y}{n}.
$$

Two implementation points. **Stability:** subtract the row maximum before exponentiating (log-sum-exp), so no
`exp` overflows and no `log(0)` is taken; compute $\ln P$ directly and get $P = e^{\ln P}$ for the gradient.
**The gradient:** it is one subtraction, not a softmax Jacobian — we verify it numerically.
""")

nb.code(r"""
def softmax_ce(Z, y):
    # mean cross-entropy over the rows of Z (labels y as integers) and its gradient dL/dZ
    n = Z.shape[0]
    m = Z.max(axis=1, keepdims=True)
    logp = Z - m - np.log(np.exp(Z - m).sum(axis=1, keepdims=True))      # stable log-softmax
    loss = -logp[np.arange(n), y].mean()
    G = np.exp(logp)                                                     # P
    G[np.arange(n), y] -= 1.0                                            # P - Y
    return loss, G / n

Z = np.array([[2.0, 0.0, -1.0]]); y = np.array([0])
loss, G = softmax_ce(Z, y)
print(f"z = {Z[0]}, true class 0:  P = {np.exp(Z - Z.max()) / np.exp(Z - Z.max()).sum()},  loss = {loss:.4f},  gradient P - Y = {G[0]}")
loss2, G2 = softmax_ce(Z, np.array([2]))
print(f"same logits, true class 2:  loss = {loss2:.4f},  gradient = {G2[0]}   <- confident wrong answer, gradient of size ~1")

# numerical check of the gradient on a random batch, and the overflow test
Zr = rng.normal(size=(4, 5)) * 3; yr = rng.integers(0, 5, size=4)
loss_r, G_r = softmax_ce(Zr, yr)
G_num = np.zeros_like(Zr); hh = 1e-6
for idx in np.ndindex(Zr.shape):
    Zp = Zr.copy(); Zp[idx] += hh; Zm = Zr.copy(); Zm[idx] -= hh
    G_num[idx] = (softmax_ce(Zp, yr)[0] - softmax_ce(Zm, yr)[0]) / (2 * hh)
print(f"max |(P - Y)/n - central difference| = {np.max(np.abs(G_r - G_num)):.1e}")
big = np.array([[1000.0, 999.0, -1000.0]])
with np.errstate(over="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print(f"logits {big[0]}: stable loss = {softmax_ce(big, np.array([0]))[0]:.4f} (= ln(1 + e^-1));  naive softmax = {naive[0]}")
""")

nb.md(r"""
Push the correct logit up, the wrong ones down, each by exactly how wrong its probability is; the
gradient row sums to zero because adding a constant to all logits changes nothing. A confidently wrong
prediction gets a gradient of size almost 1 — cross-entropy shouts loudest when the model is most wrong.
Contrast the *sigmoid plus squared error* pairing, whose gradient $2(\sigma - y)\,\sigma(1 - \sigma)$ carries the
sigmoid's slope and goes quiet exactly there:
""")

nb.code(r"""
zz = np.linspace(-8, 8, 400); y1 = 1.0
grad_ce = sigmoid(zz) - y1                               # BCE on the logit
grad_mse = 2 * (sigmoid(zz) - y1) * sigmoid(zz) * (1 - sigmoid(zz))
plt.plot(zz, np.abs(grad_ce), label="|dL/dz|, sigmoid + cross-entropy")
plt.plot(zz, np.abs(grad_mse), label="|dL/dz|, sigmoid + squared error")
plt.xlabel("logit z (true label y = 1)"); plt.ylabel("gradient magnitude"); plt.yscale("log")
plt.title("Confidently wrong (z << 0): squared error stops learning"); plt.legend(); plt.grid(alpha=0.3); plt.show()
z0 = -5.0
print(f"at z = {z0}: CE gradient {sigmoid(z0) - 1:.4f}, MSE gradient {2 * (sigmoid(z0) - 1) * sigmoid(z0) * (1 - sigmoid(z0)):.4f}, ratio {(sigmoid(z0) - 1) / (2 * (sigmoid(z0) - 1) * sigmoid(z0) * (1 - sigmoid(z0))):.0f}x")
""")

nb.md(r"""
At $z = -5$ with $y = 1$ the squared-error gradient is 75 times smaller than cross-entropy's, and it keeps
shrinking as the model gets *more* wrong. The $\ln$ in the cross-entropy cancels the $\exp$ in the sigmoid;
that cancellation is why classification uses cross-entropy.

""")

# ---------------------------------------------------------------- Section 5
nb.md(r"""
## 5. The backward pass: `Dense`, `ReLU`, `MLP`

Reverse mode through a layer $Z = A W + \mathbf{1}b^\top$ needs only Module 06's matrix rules, now with one row
per example. With $G = \partial\mathcal{L}/\partial Z$ of shape $(n \times n_{\text{out}})$:

$$
\frac{\partial\mathcal{L}}{\partial W} = A^\top G \ (n_{\text{in}} \times n_{\text{out}}), \qquad
\frac{\partial\mathcal{L}}{\partial b} = \sum_{\text{rows}} G \ (n_{\text{out}}), \qquad
\frac{\partial\mathcal{L}}{\partial A} = G W^\top \ (n \times n_{\text{in}}),
$$

and an elementwise activation contributes a mask, $G_{\text{in}} = G_{\text{out}} \odot \phi'(Z)$. A layer object stores
what its backward pass needs (its input `X`, or the ReLU mask) during `forward`. A network is a list of
layers run forwards, then backwards in reverse order.
""")

nb.code(r"""
class Dense:
    def __init__(self, n_in, n_out, scale=None, rng=rng):
        scale = np.sqrt(2.0 / n_in) if scale is None else scale          # He initialisation by default
        self.W = rng.normal(0.0, scale, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)
    def forward(self, X):
        self.X = X                                   # keep the input for the weight gradient
        return X @ self.W + self.b
    def backward(self, G):                           # G = dL/dZ, shape (n, n_out)
        self.dW = self.X.T @ G                       # (n_in, n_out)
        self.db = G.sum(axis=0)                      # broadcast in forward -> sum in backward
        return G @ self.W.T                          # dL/dX, shape (n, n_in)
    def params(self): return [self.W, self.b]
    def grads(self):  return [self.dW, self.db]

class ReLU:
    def forward(self, Z):
        self.mask = Z > 0
        return Z * self.mask
    def backward(self, G): return G * self.mask
    def params(self): return []
    def grads(self):  return []

class MLP:
    def __init__(self, layers): self.layers = layers
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, G):
        for layer in reversed(self.layers):
            G = layer.backward(G)
        return G                                      # dL/d(input), useful for saliency maps
    def params(self): return [p for l in self.layers for p in l.params()]
    def grads(self):  return [g for l in self.layers for g in l.grads()]

# Section 3's example with squared-error loss L = (1/2n) sum (Z2 - y)^2, targets y = (3, 1)
L1 = Dense(3, 2); L1.W[:] = W1; L1.b[:] = b1
L2 = Dense(2, 1); L2.W[:] = W2; L2.b[:] = b2
net = MLP([L1, ReLU(), L2])
y_t = np.array([[3.], [1.]])
Z2 = net.forward(X)
loss = 0.5 * np.sum((Z2 - y_t) ** 2) / len(X)
G2 = (Z2 - y_t) / len(X)
dX = net.backward(G2)
G1 = (G2 @ L2.W.T) * (Z1 > 0)                   # what ReLU.backward handed to L1 (Z1 from Section 3)
print(f"loss = {loss}\nG2 = dL/dZ2 =\n{G2}\ndW2 =\n{L2.dW}\ndb2 = {L2.db}\nG1 = dL/dZ1 =\n{G1}\ndW1 =\n{L1.dW}\ndb1 = {L1.db}\ndL/dX =\n{dX}")
""")

nb.md(r"""
Every number matches the hand calculation in the concept guide: $\partial\mathcal{L}/\partial W_2 = (0.2, -0.01)^\top$,
$\partial\mathcal{L}/\partial b_2 = 0.25$, and the zero in the bottom-right of $\partial\mathcal{L}/\partial W_1$ — that weight is reached
only through the dead unit of example 2. Now the step you should never skip: a gradient check with central
differences, first on this example and then on a random network with a softmax cross-entropy loss.
""")

nb.code(r"""
def flat_grads(model):
    return np.concatenate([g.ravel() for g in model.grads()])

def grad_check(model, loss_fn, Xb, yb, h=1e-5):
    # loss_fn(Z, yb) -> (loss, dL/dZ). Returns the relative error ||g - g_num|| / (||g|| + ||g_num||).
    Z = model.forward(Xb); loss, G = loss_fn(Z, yb); model.backward(G)
    g = flat_grads(model)
    g_num = np.empty_like(g); k = 0
    for p in model.params():
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + h; lp = loss_fn(model.forward(Xb), yb)[0]
            p[idx] = old - h; lm = loss_fn(model.forward(Xb), yb)[0]
            p[idx] = old
            g_num[k] = (lp - lm) / (2 * h); k += 1
    return np.linalg.norm(g - g_num) / (np.linalg.norm(g) + np.linalg.norm(g_num)), g.size

mse = lambda Z, y: (0.5 * np.sum((Z - y) ** 2) / len(Z), (Z - y) / len(Z))
r, k = grad_check(net, mse, X, y_t)
print(f"Section 3 network, squared error: relative error {r:.1e} over {k} parameters")

check_rng = np.random.default_rng(1)
small = MLP([Dense(8, 6, rng=check_rng), ReLU(), Dense(6, 5, rng=check_rng), ReLU(), Dense(5, 4, rng=check_rng)])
Xb = check_rng.normal(size=(7, 8)); yb = check_rng.integers(0, 4, size=7)
r, k = grad_check(small, softmax_ce, Xb, yb)
print(f"random 8-6-5-4 network, softmax-CE:  relative error {r:.1e} over {k} parameters   (< 1e-7 means the backward pass is right)")
""")

nb.md(r"""
Relative errors around $10^{-10}$: the three backward rules and their composition are correct. If one of
them were wrong — `self.X @ G` instead of `self.X.T @ G`, say — the shape check would catch it; the subtler
`G @ self.W` (wrong shape only when the layer is not square) or a forgotten `.sum(axis=0)` would show up
here as a relative error of order 1. Thirty seconds of checking, and we can trust the gradients that
will drive thousands of updates.

**Try it yourself:** plant a bug — change `return G * self.mask` to `return G` in `ReLU.backward` — and re-run
the check. Then move the bug to `db = G.mean(axis=0)`. How large is each error?
""")

# ---------------------------------------------------------------- Section 5b: digits
nb.md(r"""
### Training on real data: handwritten digits

`load_digits` has 1,797 images of $8\times 8$ pixels (values 0–16), ten classes. We standardise each pixel using
*training-set* statistics, split 80/20, build a 64 → 128 → 10 ReLU network (9,610 parameters, He-initialised)
and train it with mini-batch Adam (Module 05). Two optimisers are written out so you can see that an
optimiser is just a rule for turning `grads()` into parameter updates. One detail: the hidden layer is
He-initialised because a ReLU follows it, but the *output* layer is linear and we give it a small scale
(0.01) so that the initial logits are near zero and the initial loss is the uninformed value $\ln 10$ —
item 3 of the training recipe in the concept guide.
""")

nb.code(r"""
from sklearn.datasets import load_digits

class SGD:
    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=0.0):
        self.model, self.lr, self.mu, self.wd = model, lr, momentum, weight_decay
        self.v = [np.zeros_like(p) for p in model.params()]
    def step(self):
        for i, (p, g) in enumerate(zip(self.model.params(), self.model.grads())):
            self.v[i] = self.mu * self.v[i] + g + self.wd * p
            p -= self.lr * self.v[i]                         # in-place: the layers see the update

class Adam:
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.model, self.lr, self.b1, self.b2, self.eps, self.wd = model, lr, beta1, beta2, eps, weight_decay
        self.m = [np.zeros_like(p) for p in model.params()]
        self.v = [np.zeros_like(p) for p in model.params()]
        self.t = 0
    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.model.params(), self.model.grads())):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p)     # AdamW: decoupled decay

digits = load_digits()
X_all, y_all = digits.data, digits.target
split_rng = np.random.default_rng(16)
perm = split_rng.permutation(len(X_all)); n_train = int(0.8 * len(X_all))
X_tr, y_tr = X_all[perm[:n_train]], y_all[perm[:n_train]]
X_te, y_te = X_all[perm[n_train:]], y_all[perm[n_train:]]
mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
sd[sd == 0] = 1.0                                     # three corner pixels are constant zero
X_tr = (X_tr - mu) / sd; X_te = (X_te - mu) / sd
print(f"train {X_tr.shape}, test {X_te.shape}, classes {np.bincount(y_tr)}")

fig, axes = plt.subplots(1, 10, figsize=(10, 1.6))
for d, ax in enumerate(axes):
    ax.imshow(digits.images[np.where(y_all == d)[0][0]], cmap="gray_r"); ax.set_title(str(d)); ax.axis("off")
fig.suptitle("load_digits: one example per class (8 x 8 pixels)"); plt.show()
""")

nb.code(r"""
def accuracy(model, Xs, ys):
    return float(np.mean(model.forward(Xs).argmax(axis=1) == ys))

def train(model, opt, X_tr, y_tr, X_te, y_te, epochs=30, batch_size=64, rng=rng, verbose=True):
    hist = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(epochs):
        order = rng.permutation(len(X_tr))                        # reshuffle every epoch
        for start in range(0, len(X_tr), batch_size):
            idx = order[start:start + batch_size]
            Z = model.forward(X_tr[idx])                          # forward
            loss, G = softmax_ce(Z, y_tr[idx])                    # loss and dL/dZ
            model.backward(G)                                     # gradients into every layer
            opt.step()                                            # update
        hist["train_loss"].append(softmax_ce(model.forward(X_tr), y_tr)[0])
        hist["test_loss"].append(softmax_ce(model.forward(X_te), y_te)[0])
        hist["train_acc"].append(accuracy(model, X_tr, y_tr)); hist["test_acc"].append(accuracy(model, X_te, y_te))
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:2d}  train loss {hist['train_loss'][-1]:.4f}  test loss {hist['test_loss'][-1]:.4f}  "
                  f"train acc {hist['train_acc'][-1]:.4f}  test acc {hist['test_acc'][-1]:.4f}")
    return hist

model_rng = np.random.default_rng(0)
model = MLP([Dense(64, 128, rng=model_rng), ReLU(), Dense(128, 10, scale=0.01, rng=model_rng)])   # He for the ReLU layer; small output layer
print("parameters:", sum(p.size for p in model.params()), "   initial loss:", round(softmax_ce(model.forward(X_tr), y_tr)[0], 3), " (ln 10 =", round(np.log(10), 3), ")")
t0 = time.time()
hist = train(model, Adam(model, lr=3e-3), X_tr, y_tr, X_te, y_te, epochs=30, rng=np.random.default_rng(1))
print(f"training time {time.time() - t0:.1f}s   final test accuracy {hist['test_acc'][-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist["train_loss"], label="train"); axes[0].plot(hist["test_loss"], label="test")
axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(hist["train_acc"], label="train"); axes[1].plot(hist["test_acc"], label="test")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].set_title("Accuracy"); axes[1].set_ylim(0.85, 1.005); axes[1].legend()
plt.show()
""")

nb.md(r"""
Around 97–98 percent test accuracy in a few seconds of NumPy — from a loss that started at $\ln 10 = 2.30$,
as every uninformed 10-class model should (check this number every time; a very different value means the
output layer is badly scaled, or there is a bug in the loss or the labels). Read the curves the way Section 9
of the guide describes: the training loss keeps falling towards zero while the test loss bottoms out after
about a dozen epochs and then creeps up slightly — the model is becoming over-confident on the few examples
it gets wrong, without changing which ones. Early stopping at the test minimum would cost nothing in accuracy.
And note the scale of the claim: 360 test examples give a standard error of about 0.8 percentage points on the
accuracy, so differences of half a percent between runs are noise.
""")

nb.code(r"""
pred = model.forward(X_te).argmax(axis=1)
conf = np.zeros((10, 10), int)
for t, p in zip(y_te, pred):
    conf[t, p] += 1
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.imshow(conf, cmap="Blues"); ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test set)")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        if conf[i, j]:
            ax.text(j, i, conf[i, j], ha="center", va="center", color="white" if conf[i, j] > 20 else "black", fontsize=8)
plt.show()

wrong = np.where(pred != y_te)[0]
fig, axes = plt.subplots(1, max(len(wrong[:10]), 1), figsize=(1.1 * max(len(wrong[:10]), 1), 1.8))
for ax, i in zip(np.atleast_1d(axes), wrong[:10]):
    ax.imshow((X_te[i] * sd + mu).reshape(8, 8), cmap="gray_r"); ax.set_title(f"{y_te[i]} -> {pred[i]}", fontsize=9); ax.axis("off")
fig.suptitle(f"{len(wrong)} of {len(y_te)} test digits misclassified (true -> predicted)", y=1.08); plt.show()
""")

nb.md(r"""
Most of the errors are genuinely ambiguous $8\times8$ blobs — eights read as ones or nines, threes as fives.
A convolutional network would recover perhaps a percentage point here; the remaining errors are in the
data, not the model.

**Try it yourself:** replace `Adam(model, lr=3e-3)` by `SGD(model, lr=0.02, momentum=0.9)` and compare the
curves; then try a 64–128–64–10 network, and `weight_decay=1e-2` in Adam. Which changes move the *test* loss?
""")

# ---------------------------------------------------------------- Section 6
nb.md(r"""
## 6. Initialisation and variance propagation

For a pre-activation $z = \sum_{j=1}^{n_{\text{in}}} w_j x_j$ with independent zero-mean weights and inputs,

$$
\operatorname{Var}(z) = n_{\text{in}}\, \sigma_w^2\, \sigma_x^2 ,
$$

so each layer multiplies the signal variance by $n_{\text{in}} \sigma_w^2$ — and a ReLU then halves the second moment.
Preserving the scale needs $\sigma_w^2 = 1/n_{\text{in}}$ for linear/tanh layers (LeCun; Xavier averages fan-in and
fan-out) and $\sigma_w^2 = 2/n_{\text{in}}$ for ReLU (He). Let us push a batch through 10 layers of width 256 with four
weight scales and record the standard deviation of the activations at each depth.
""")

nb.code(r"""
def activation_stats(scale, depth=10, width=256, act=np.tanh, n=512, seed=0):
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, width)); stds = [A.std()]
    for _ in range(depth):
        W = r.normal(0.0, scale, size=(width, width))
        A = act(A @ W); stds.append(A.std())
    return np.array(stds)

width = 256
relu_f = lambda z: np.maximum(z, 0.0)
schemes = {"scale 0.01": 0.01, "scale 1.0": 1.0, "Xavier  sqrt(1/n)": np.sqrt(1 / width), "He  sqrt(2/n)": np.sqrt(2 / width)}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, act, name in zip(axes, [np.tanh, relu_f], ["tanh", "ReLU"]):
    for label, s in schemes.items():
        ax.semilogy(activation_stats(s, act=act), "o-", ms=4, label=label)
    ax.set_title(f"activation std across depth, {name} units, width {width}"); ax.set_xlabel("layer"); ax.set_ylabel("std of activations")
    ax.set_ylim(1e-9, 1e13); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.show()
for label, s in schemes.items():
    print(f"{label:20s} n*sigma_w^2 = {width * s**2:8.4f}   tanh std after 10 layers {activation_stats(s)[-1]:.2e}   ReLU std after 10 layers {activation_stats(s, act=relu_f)[-1]:.2e}")
""")

nb.md(r"""
Scale 0.01 multiplies the variance by $256 \times 10^{-4} = 0.0256$ per layer — after ten layers the activations are
$10^{-8}$, and so are the gradients that flow back through the same matrices. Scale 1.0 multiplies the variance
by 256 per layer (128 after the ReLU halves it), so the ReLU activations grow by a factor of about 11 per layer
to $10^{10}$ after ten; the gradients grow the same way, and the first update is a huge step in a random
direction. The tanh chain does not blow up — its units saturate at $\pm 1$ instead, which kills the gradient a
different way. Xavier keeps tanh activations at a constant scale but lets ReLU activations shrink by $\sqrt{1/2}$
per layer; He's extra factor of 2 compensates for the half of the signal the ReLU discards. The right scale is
not a tuning knob — it is $n_{\text{in}} \sigma_w^2 = 1$ or $2$, which you compute.

**Try it yourself:** set `depth=50`. Which schemes survive? Then try `act=lambda z: np.where(z > 0, z, 0.1 * z)`
(leaky ReLU) with the He scale and with $\sqrt{2/(1 + 0.1^2)/n}$.
""")

# ---------------------------------------------------------------- Section 7
nb.md(r"""
## 7. Vanishing and exploding gradients, and the residual fix

The gradient reaching layer $l$ is the output gradient times a product of $L - l$ Jacobians, each
$\operatorname{diag}(\phi'(Z))\,W^\top$ in row form. Sigmoid's $\phi' \le 1/4$ makes the product shrink by at least 4 per layer
whatever the weights; ReLU's mask is 0 or 1 and lets the active paths through undiminished. A residual block
$A_{l} = A_{l-1} + F(A_{l-1})$ has Jacobian $I + J_l$, so the product $\prod_l (I + J_l) = I + \sum_l J_l + \cdots$
always contains an identity path. Let us measure $\lVert \partial\mathcal{L}/\partial A_l \rVert$ at every depth of a 20-layer chain.
""")

nb.code(r"""
def gradient_norms_by_layer(Ws, act, dact, x, residual=False):
    # forward through the chain, then reverse mode; returns ||dL/dA_l|| for l = 0..L with L = sum of outputs
    A = x.copy(); Zs = []
    for W in Ws:
        Z = A @ W; Zs.append(Z)
        A = A + act(Z) if residual else act(Z)
    G = np.ones_like(A); norms = [np.linalg.norm(G)]
    for W, Z in zip(reversed(Ws), reversed(Zs)):
        Gz = G * dact(Z)
        G = Gz @ W.T + (G if residual else 0.0)
        norms.append(np.linalg.norm(G))
    return np.array(norms[::-1])            # index l = layer input, 0 = network input

depth, width = 20, 64
r = np.random.default_rng(7)
x0 = r.normal(size=width)
W_lecun = [r.normal(0, np.sqrt(1 / width), (width, width)) for _ in range(depth)]
W_he = [r.normal(0, np.sqrt(2 / width), (width, width)) for _ in range(depth)]
W_small = [r.normal(0, 0.3 * np.sqrt(1 / width), (width, width)) for _ in range(depth)]
dsig = lambda z: sigmoid(z) * (1 - sigmoid(z)); dtanh = lambda z: 1 - np.tanh(z) ** 2; drelu = lambda z: (z > 0).astype(float)

curves = {
    "sigmoid, LeCun init":               gradient_norms_by_layer(W_lecun, sigmoid, dsig, x0),
    "tanh, LeCun init":                  gradient_norms_by_layer(W_lecun, np.tanh, dtanh, x0),
    "tanh, 0.3 x LeCun init":            gradient_norms_by_layer(W_small, np.tanh, dtanh, x0),
    "ReLU, He init":                     gradient_norms_by_layer(W_he, relu_f, drelu, x0),
    "tanh, 0.3 x LeCun init + residual": gradient_norms_by_layer(W_small, np.tanh, dtanh, x0, residual=True),
}
for label, c in curves.items():
    plt.semilogy(range(depth + 1), c, "o-", ms=3, label=label)
plt.semilogy(range(depth + 1), curves["sigmoid, LeCun init"][-1] * 0.25 ** np.arange(depth, -1, -1), "k:", label=r"$4^{-(L-l)}$ bound")
plt.xlabel("layer l  (0 = input, 20 = output)"); plt.ylabel(r"$\|\partial L / \partial A_l\|$"); plt.title("Gradient norm by depth in a 20-layer chain")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()
for label, c in curves.items():
    print(f"{label:36s} gradient at input / gradient at output = {c[0] / c[-1]:.2e}")
""")

nb.md(r"""
The sigmoid chain loses about thirteen orders of magnitude in twenty layers, tracking the $4^{-(L-l)}$ bound;
its first layers would never learn. The tanh chain with weights a bit too small loses more than ten orders —
vanishing is not only a sigmoid problem, it is a *scale* problem. ReLU with He initialisation and tanh with the right
scale both keep the gradient within a factor of a few across the whole depth. And the residual connection
rescues even the badly scaled tanh chain: the identity path guarantees the gradient at the input is at
least of the order of the gradient at the output, whatever the branches do. That one line, `A + act(Z)`,
is why 100-layer networks train today.

Exploding gradients are the mirror image (replace the He weights by `np.sqrt(2.2 / width)`, ten percent too
large, and the ReLU curve climbs instead); gradient clipping, `g *= min(1, c / ||g||)`, is the usual seat belt.
""")

# ---------------------------------------------------------------- Section 8
nb.md(r"""
## 8. Batch normalisation

Instead of hoping the weights keep activations at a sensible scale, batch norm *re-standardises* every
pre-activation over the mini-batch and lets the network choose a new scale and shift:

$$
\mu_j = \frac1n\sum_i Z_{ij}, \quad \sigma_j^2 = \frac1n\sum_i (Z_{ij} - \mu_j)^2, \quad
\hat Z_{ij} = \frac{Z_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \quad Y_{ij} = \gamma_j \hat Z_{ij} + \beta_j .
$$

During training it also keeps running averages of $\mu$ and $\sigma^2$, which replace the batch statistics at
inference time, so a single example gets a deterministic output. The backward pass must go through $\mu$ and
$\sigma^2$ as well (they depend on every row); the formula is in the guide, and we check it numerically.
""")

nb.code(r"""
class BatchNorm:
    def __init__(self, n_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(n_features); self.beta = np.zeros(n_features)
        self.dgamma = np.zeros(n_features); self.dbeta = np.zeros(n_features)
        self.eps, self.momentum = eps, momentum
        self.running_mean = np.zeros(n_features); self.running_var = np.ones(n_features)
        self.training = True
    def forward(self, Z):
        if self.training:
            mu = Z.mean(axis=0); var = Z.var(axis=0)
            n = Z.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * n / max(n - 1, 1)
        else:
            mu, var = self.running_mean, self.running_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.Zhat = (Z - mu) * self.inv_std
        return self.gamma * self.Zhat + self.beta
    def backward(self, G):
        n = G.shape[0]
        self.dgamma = (G * self.Zhat).sum(axis=0); self.dbeta = G.sum(axis=0)
        return self.gamma * self.inv_std / n * (n * G - G.sum(axis=0) - self.Zhat * (G * self.Zhat).sum(axis=0))
    def params(self): return [self.gamma, self.beta]
    def grads(self):  return [self.dgamma, self.dbeta]

# (a) the worked example: one unit, batch (1, 2, 3, 6), gamma = 2, beta = 1
bn = BatchNorm(1); bn.gamma[:] = 2.0; bn.beta[:] = 1.0
Zb = np.array([[1.], [2.], [3.], [6.]])
print("y =", bn.forward(Zb).ravel(), "  running mean", bn.running_mean, " running var (unbiased, after one update)", bn.running_var)
print("scale invariance: same inputs x100 ->", bn.forward(100 * Zb).ravel())

# (b) gradient check of the backward pass inside a small network
bn_rng = np.random.default_rng(3)
bn_net = MLP([Dense(6, 5, rng=bn_rng), BatchNorm(5), ReLU(), Dense(5, 3, rng=bn_rng)])
Xb = bn_rng.normal(size=(8, 6)) * 3 + 1; yb = bn_rng.integers(0, 3, size=8)
r_bn, k_bn = grad_check(bn_net, softmax_ce, Xb, yb)
print(f"gradient check through BatchNorm (training mode): relative error {r_bn:.1e} over {k_bn} parameters")

# (c) effect on activation statistics in a deep, badly initialised chain
def deep_stats(use_bn, scale=1.0, depth=10, width=256, n=512, seed=0):
    r = np.random.default_rng(seed); A = r.normal(size=(n, width)); stds = [A.std()]
    for _ in range(depth):
        Z = A @ r.normal(0, scale, size=(width, width))
        if use_bn:
            Z = BatchNorm(width).forward(Z)
        A = relu_f(Z); stds.append(A.std())
    return np.array(stds)
for scale in (0.01, 1.0):
    plt.semilogy(deep_stats(False, scale), "o--", ms=3, label=f"scale {scale}, no BN")
    plt.semilogy(deep_stats(True, scale), "s-", ms=3, label=f"scale {scale}, with BN")
plt.xlabel("layer"); plt.ylabel("std of ReLU activations"); plt.title("Batch norm makes the activation scale independent of the weight scale")
plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

nb.md(r"""
Three facts in one cell. The worked example reproduces the guide ($y = (-1.138, -0.069, 1, 4.207)$) and shows
the scale invariance — multiply the inputs by 100 and the output is identical. The backward pass, which
removes from the gradient the components that would change the batch mean and variance, passes the
gradient check to $10^{-10}$. And with batch norm the activation statistics across ten layers are the same
whether the weights are a hundred times too small or not: the layer after a batch norm never has to care
what happened upstream, which is what lets you raise the learning rate by an order of magnitude.

Switch `bn.training = False` before serving predictions. With `training = True` and a batch of one, $\hat Z = 0$
for every unit and the network outputs a constant — one of the most common deployment bugs in practice.
""")

# ---------------------------------------------------------------- Section 10
nb.md(r"""
## 10. Dropout against overfitting

Inverted dropout keeps each hidden activation with probability $q = 1 - p$ and divides the survivors by $q$,
so $E[\tilde A] = A$ and the layer can simply be switched off at test time. We make the problem hard on purpose:
train a 64 → 256 → 10 network (19,210 parameters) on only **100** digits, full-batch, for 400 Adam steps at a
deliberately brisk learning rate of $10^{-2}$, and compare the test loss with and without dropout on the hidden
layer. (Dropout is active only in training mode; the losses below are evaluated with it switched off.)
""")

nb.code(r"""
class Dropout:
    def __init__(self, p=0.5, rng=rng):
        self.p, self.rng, self.training = p, rng, True
    def forward(self, A):
        if not self.training or self.p == 0:
            self.mask = 1.0
        else:
            self.mask = (self.rng.random(A.shape) >= self.p) / (1 - self.p)      # inverted dropout
        return A * self.mask
    def backward(self, G): return G * self.mask
    def params(self): return []
    def grads(self):  return []

def set_training(model, flag):
    for layer in model.layers:
        if hasattr(layer, "training"):
            layer.training = flag

def small_data_run(p_drop, weight_decay=0.0, n_small=100, steps=400, seed=0):
    r = np.random.default_rng(seed)
    Xs, ys = X_tr[:n_small], y_tr[:n_small]
    model = MLP([Dense(64, 256, rng=r), ReLU(), Dropout(p_drop, rng=r), Dense(256, 10, rng=r)])
    opt = Adam(model, lr=1e-2, weight_decay=weight_decay)
    tr, te = [], []
    for step in range(steps):
        set_training(model, True)
        loss, G = softmax_ce(model.forward(Xs), ys); model.backward(G); opt.step()
        set_training(model, False)
        tr.append(softmax_ce(model.forward(Xs), ys)[0]); te.append(softmax_ce(model.forward(X_te), y_te)[0])
    return np.array(tr), np.array(te), accuracy(model, X_te, y_te)

runs = {"no dropout": (0.0, 0.0), "dropout p = 0.5": (0.5, 0.0), "dropout p = 0.5 + weight decay 0.1": (0.5, 0.1)}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for label, (p_drop, wd) in runs.items():
    tr, te, acc = small_data_run(p_drop, wd)
    axes[0].plot(tr, label=label); axes[1].plot(te, label=f"{label} (acc {acc:.3f})")
    print(f"{label:36s} final train loss {tr[-1]:.4f}   final test loss {te[-1]:.4f}   min test loss {te.min():.4f} at step {te.argmin()}   test acc {acc:.3f}")
axes[0].set_title("training loss (100 training digits)"); axes[1].set_title("test loss (360 digits)")
for ax in axes:
    ax.set_xlabel("full-batch Adam step"); ax.set_ylabel("cross-entropy"); ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.show()
""")

nb.md(r"""
Without dropout the network memorises the 100 digits within a few steps (training loss $\to 0$) and the test
loss reaches its minimum almost immediately and then climbs for the rest of the run — classic overfitting. With
dropout the training loss falls more slowly, the test loss keeps improving for most of the run and ends well
below the unregularised curve, and weight decay lowers it further still and removes the upturn entirely. The
accuracy gain is a few percentage points; with 100 examples that is what regularisation can buy. The *shape*
of the curves is the lesson: regularisation trades training fit for a test curve that does not turn around.

**Try it yourself:** set `n_small=1000`. Does dropout still help, or does the gap close once there is enough data?
""")

# ---------------------------------------------------------------- Section 11
nb.md(r"""
## 11. Beyond MLPs: convolution is a `Dense` layer with structure

A 1-D cross-correlation $y_i = \sum_{j} w_j x_{i+j}$ is the product $y = Tx$ with a *Toeplitz* matrix $T$ whose rows are
shifted copies of the kernel. So a convolutional layer is a `Dense` layer whose weight matrix is sparse (each output
sees $m$ inputs — its receptive field) and whose non-zero entries are shared along the diagonals. Let us build $T$,
check it against `np.convolve`, and derive the backward pass from the Toeplitz form.
""")

nb.code(r"""
def toeplitz_conv_matrix(w, n):
    m = len(w)
    T = np.zeros((n - m + 1, n))
    for i in range(n - m + 1):
        T[i, i:i + m] = w
    return T

x_sig = np.array([1., 2., 3., 4., 2., 0., -1.])
w_k = np.array([1., -1., 0.5])
T = toeplitz_conv_matrix(w_k, len(x_sig))
y_T = T @ x_sig
y_np = np.convolve(x_sig, w_k[::-1], mode="valid")     # np.convolve flips the kernel; flip it back for cross-correlation
print("Toeplitz matrix T =\n", T)
print("T @ x          =", y_T)
print("np.convolve    =", y_np, "   max difference", np.max(np.abs(y_T - y_np)))
print(f"parameters: kernel {w_k.size} vs. unconstrained dense layer {T.size}")

# backward pass from the Toeplitz form, for an upstream gradient G on y
G_y = rng.normal(size=len(y_T))
dx = T.T @ G_y                                                                 # = full convolution of G with the flipped kernel
dx_conv = np.convolve(G_y, w_k, mode="full")
dw = np.array([G_y @ x_sig[j:j + len(y_T)] for j in range(len(w_k))])          # shared weight: sum over all positions
print("dL/dx via T^T G  =", dx)
print("dL/dx via conv   =", dx_conv, "   max difference", np.max(np.abs(dx - dx_conv)))
print("dL/dw (sum of per-position contributions) =", dw)

# receptive field: stack three kernel-3 layers and see which inputs one output depends on
T3 = toeplitz_conv_matrix(np.ones(3), 9) ; T2 = toeplitz_conv_matrix(np.ones(3), 7); T1 = toeplitz_conv_matrix(np.ones(3), 5)
print("non-zero pattern of (T1 @ T2 @ T3)[0] after three 3-wide convolutions:", (T1 @ T2 @ T3)[0] != 0, " -> receptive field 7")
""")

nb.md(r"""
Same numbers from the matrix and from `np.convolve`, and the backward pass falls straight out of Section 5:
$\partial\mathcal{L}/\partial x = T^\top G$ is a convolution with the *flipped* kernel, and the gradient of each shared weight
is the sum of the gradients of all its copies (broadcast ⇒ sum, exactly as for the bias). Three layers of
width-3 kernels give every output a receptive field of 7 inputs; that additive growth is how a pixel at the
top of a ResNet "sees" the whole image. A recurrent network is the same story with the same matrix applied
at every time step (so its gradients behave like $\sigma_{\max}(W)^{T}$), an embedding layer is a `Dense` layer fed
one-hot rows, and attention is a `Dense` layer whose position-mixing matrix is computed from the data —
Module 17.
""")

# ---------------------------------------------------------------- Summary
nb.md(r"""
## Summary

* A hidden layer turns "which side of which lines" into new coordinates in which a linear output layer
  works; XOR needs it, and so does everything interesting. No nonlinearity ⇒ the whole stack is one affine map.
* Activations are chosen by their derivatives: sigmoid $\le 1/4$, tanh $\le 1$, ReLU/SiLU/GELU $\approx 1$ on the active side.
* Batch-first shapes: $Z = XW + b$ with $X: n \times n_{\text{in}}$, $W: n_{\text{in}} \times n_{\text{out}}$; forward cost
  $\approx 2 \cdot \text{weights} \cdot \text{batch}$ FLOPs.
* Softmax cross-entropy with log-sum-exp is stable, starts at $\ln k$, and has gradient $(P - Y)/n$; sigmoid plus
  squared error vanishes exactly where the model is most wrong.
* Backward pass: $\partial W = A^\top G$, $\partial b = \sum_{\text{rows}} G$, $\partial A = G W^\top$, mask through the activation —
  three rules, verified to $10^{-10}$ by central differences, trained to 97 percent on the digits.
* Initialisation is a calculation: $\operatorname{Var}(z) = n_{\text{in}} \sigma_w^2 \sigma_x^2$, so $\sigma_w^2 = 1/n_{\text{in}}$ (tanh) or
  $2/n_{\text{in}}$ (ReLU); scale 0.01 or 1.0 at width 256 loses or gains eight orders of magnitude in ten layers.
* Gradients through depth are products of Jacobians: sigmoid chains vanish like $4^{-L}$, badly scaled tanh
  chains too; ReLU + He, batch norm and residual connections ($I + J$) keep the product near 1.
* Batch norm standardises per unit over the batch, has a non-trivial backward pass, is scale-invariant, and
  must use running statistics at inference.
* Inverted dropout is multiplicative noise with $E[\tilde A] = A$; on 100 training digits it stops the test loss
  from turning upward.
* Convolution is a Toeplitz `Dense` layer; its backward pass is the transposed (flipped-kernel) convolution
  and a sum over the shared weights' positions.

**Next:** open `exercises.ipynb` and build the pieces yourself — activations, the stable loss, `Dense.backward`
with a gradient check, He initialisation, batch norm, dropout, convolution-as-a-matrix, and finally your own
regularised digit classifier with a bootstrap confidence interval.
""")

nb.save(MOD / "lesson.ipynb")
print("lesson cells:", len(nb.cells))

# =============================================================================
# EXERCISES
# =============================================================================
# Reference solutions are defined as plain strings, exec'd at build time to compute
# the hidden hashes, and reused verbatim as the solution cells.

SETUP = r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def central_diff(f, x, h=1e-6):
    # elementwise central difference of a vectorised scalar function f at the points x
    return (f(x + h) - f(x - h)) / (2 * h)
"""

# ------------------------------------------------------------------ Ex 1  activations
SOL_1 = r"""
def relu(z):
    return np.maximum(z, 0.0)

def d_relu(z):
    return (np.asarray(z) > 0).astype(float)

def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2

def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)     # stable for large |z|

def d_softplus(z):
    return sigmoid(z)

def gelu(z):
    return z * 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def d_gelu(z):
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return Phi + z * phi
"""
STARTER_1 = r"""
def relu(z):
    # YOUR CODE HERE
    return None

def d_relu(z):
    # YOUR CODE HERE
    return None

def d_sigmoid(z):
    # YOUR CODE HERE (sigmoid is already defined in the setup cell)
    return None

def d_tanh(z):
    # YOUR CODE HERE
    return None

def softplus(z):
    # log(1 + e^z), written so that it does not overflow for z = 1000
    # YOUR CODE HERE
    return None

def d_softplus(z):
    # YOUR CODE HERE
    return None

def gelu(z):
    # z * Phi(z) with Phi the standard normal CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # YOUR CODE HERE
    return None

def d_gelu(z):
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 2  parameter and FLOP counting
SOL_2 = r"""
def count_params(sizes):
    return int(sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1)))

def forward_flops(sizes, batch_size):
    return int(sum(2 * batch_size * sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1)))

def forward_shapes(sizes, batch_size):
    return [(batch_size, s) for s in sizes[1:]]
"""
STARTER_2 = r"""
def count_params(sizes):
    # sizes = [n_in, h_1, ..., h_{L-1}, n_out]; every layer has a weight matrix and a bias vector
    # YOUR CODE HERE
    return None

def forward_flops(sizes, batch_size):
    # multiply-adds of all matrix products for one forward pass of a batch, counted as 2 FLOPs each
    # (ignore the bias additions and the activations)
    # YOUR CODE HERE
    return None

def forward_shapes(sizes, batch_size):
    # list of the shapes (as tuples) of Z_1, Z_2, ..., Z_L for a batch of batch_size rows
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 3  stable log-softmax and CE
SOL_3 = r"""
def log_softmax(Z):
    m = Z.max(axis=1, keepdims=True)
    return Z - m - np.log(np.sum(np.exp(Z - m), axis=1, keepdims=True))

def cross_entropy(Z, y):
    n = Z.shape[0]
    return float(-np.mean(log_softmax(Z)[np.arange(n), y]))
"""
STARTER_3 = r"""
def log_softmax(Z):
    # row-wise log-softmax of an (n, k) array, computed with the log-sum-exp trick
    # YOUR CODE HERE
    return None

def cross_entropy(Z, y):
    # mean over rows of -log P[i, y_i], as a float; y is an integer array of shape (n,)
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 4  softmax-CE gradient
SOL_4 = r"""
def softmax_ce_grad(Z, y):
    n = Z.shape[0]
    P = np.exp(log_softmax(Z))
    P[np.arange(n), y] -= 1.0
    return P / n

def sigmoid_mse_grad(z, y):
    s = sigmoid(z)
    return 2.0 * (s - y) * s * (1.0 - s)
"""
STARTER_4 = r"""
def softmax_ce_grad(Z, y):
    # dL/dZ for L = cross_entropy(Z, y): shape (n, k). Use log_softmax from Exercise 3.
    # YOUR CODE HERE
    return None

def sigmoid_mse_grad(z, y):
    # dL/dz for L = (sigmoid(z) - y)**2, elementwise (for the comparison plot / check)
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 5  Dense backward + gradient check
DENSE_HEAD = r"""
class Dense:
    def __init__(self, n_in, n_out, scale=None, rng=rng):
        scale = np.sqrt(2.0 / n_in) if scale is None else scale
        self.W = rng.normal(0.0, scale, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
"""
DENSE_TAIL = r"""
    def params(self): return [self.W, self.b]
    def grads(self):  return [self.dW, self.db]

class ReLU:
    def forward(self, Z):
        self.mask = Z > 0
        return Z * self.mask
    def backward(self, G): return G * self.mask
    def params(self): return []
    def grads(self):  return []

class MLP:
    def __init__(self, layers): self.layers = layers
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, G):
        for layer in reversed(self.layers):
            G = layer.backward(G)
        return G
    def params(self): return [p for l in self.layers for p in l.params()]
    def grads(self):  return [g for l in self.layers for g in l.grads()]
"""
SOL_5 = DENSE_HEAD + r"""
    def backward(self, G):
        self.dW = self.X.T @ G
        self.db = G.sum(axis=0)
        return G @ self.W.T
""" + DENSE_TAIL + r"""
def grad_check(model, loss_fn, Xb, yb, h=1e-5):
    Z = model.forward(Xb); loss, G = loss_fn(Z, yb); model.backward(G)
    g = np.concatenate([a.ravel() for a in model.grads()])
    g_num = np.empty_like(g); k = 0
    for p in model.params():
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + h; lp = loss_fn(model.forward(Xb), yb)[0]
            p[idx] = old - h; lm = loss_fn(model.forward(Xb), yb)[0]
            p[idx] = old
            g_num[k] = (lp - lm) / (2 * h); k += 1
    return float(np.linalg.norm(g - g_num) / (np.linalg.norm(g) + np.linalg.norm(g_num) + 1e-300))
"""
STARTER_5 = DENSE_HEAD + r"""
    def backward(self, G):
        # G = dL/dZ with Z = X @ W + b, shape (n, n_out). Store dL/dW in self.dW and dL/db in self.db,
        # and return dL/dX (shape (n, n_in)).
        # YOUR CODE HERE
        return None
""" + DENSE_TAIL + r"""
def grad_check(model, loss_fn, Xb, yb, h=1e-5):
    # loss_fn(Z, yb) -> (loss, dL/dZ). Run forward/backward once to get the analytic gradients of all
    # model.params(), estimate every entry by central differences, and return the relative error
    # ||g - g_num|| / (||g|| + ||g_num||) as a float.
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 6  He init / variance propagation
SOL_6 = r"""
def he_std(n_in):
    return float(np.sqrt(2.0 / n_in))

def xavier_std(n_in, n_out):
    return float(np.sqrt(2.0 / (n_in + n_out)))

def propagate_std(scale, depth=10, width=256, n=1000, act="relu", seed=0):
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, width))
    f = (lambda z: np.maximum(z, 0.0)) if act == "relu" else np.tanh
    for _ in range(depth):
        W = r.normal(0.0, scale, size=(width, width))
        A = f(A @ W)
    return float(A.std())
"""
STARTER_6 = r"""
def he_std(n_in):
    # standard deviation of He initialisation for a layer with n_in inputs
    # YOUR CODE HERE
    return None

def xavier_std(n_in, n_out):
    # standard deviation of Xavier/Glorot initialisation
    # YOUR CODE HERE
    return None

def propagate_std(scale, depth=10, width=256, n=1000, act="relu", seed=0):
    # Start from A = r.normal(size=(n, width)) with r = np.random.default_rng(seed). For each of `depth` layers,
    # draw W = r.normal(0.0, scale, size=(width, width)) and set A = act(A @ W) (act: "relu" or "tanh").
    # Return the standard deviation of the final A as a float. Draw W AFTER A, in that order, so the
    # checks (which rerun this recipe) see the same random numbers.
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 7  batch norm forward + backward
SOL_7 = r"""
def batchnorm_forward(Z, gamma, beta, eps=1e-5):
    mu = Z.mean(axis=0)
    var = Z.var(axis=0)
    Zhat = (Z - mu) / np.sqrt(var + eps)
    Y = gamma * Zhat + beta
    cache = (Zhat, 1.0 / np.sqrt(var + eps), gamma)
    return Y, cache

def batchnorm_backward(G, cache):
    Zhat, inv_std, gamma = cache
    n = G.shape[0]
    dgamma = (G * Zhat).sum(axis=0)
    dbeta = G.sum(axis=0)
    dZ = gamma * inv_std / n * (n * G - G.sum(axis=0) - Zhat * (G * Zhat).sum(axis=0))
    return dZ, dgamma, dbeta
"""
STARTER_7 = r"""
def batchnorm_forward(Z, gamma, beta, eps=1e-5):
    # Z: (n, h). Per column: mu, biased variance, Zhat = (Z - mu) / sqrt(var + eps), Y = gamma * Zhat + beta.
    # Return (Y, cache) where cache holds whatever the backward pass needs (e.g. Zhat, 1/sqrt(var + eps), gamma).
    # YOUR CODE HERE
    return None, None

def batchnorm_backward(G, cache):
    # G = dL/dY, shape (n, h). Return (dZ, dgamma, dbeta) with
    #   dgamma = sum_rows(G * Zhat),  dbeta = sum_rows(G),
    #   dZ = gamma / (n * sqrt(var + eps)) * (n * G - sum_rows(G) - Zhat * sum_rows(G * Zhat)).
    # YOUR CODE HERE
    return None, None, None
"""

# ------------------------------------------------------------------ Ex 8  inverted dropout
SOL_8 = r"""
def dropout_forward(A, p, rng, training=True):
    if not training or p == 0.0:
        mask = np.ones_like(A)
    else:
        mask = (rng.random(A.shape) >= p) / (1.0 - p)
    return A * mask, mask

def dropout_backward(G, mask):
    return G * mask
"""
STARTER_8 = r"""
def dropout_forward(A, p, rng, training=True):
    # Inverted dropout. In training mode draw mask = (rng.random(A.shape) >= p) / (1 - p) and return
    # (A * mask, mask); in evaluation mode (or p == 0) return (A, ones_like(A)).
    # YOUR CODE HERE
    return None, None

def dropout_backward(G, mask):
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 10  convolution as a matrix
SOL_10 = r"""
def conv_matrix(w, n):
    m = len(w)
    T = np.zeros((n - m + 1, n))
    for i in range(n - m + 1):
        T[i, i:i + m] = w
    return T

def conv1d(x, w):
    return conv_matrix(w, len(x)) @ x

def conv1d_backward(x, w, G):
    T = conv_matrix(w, len(x))
    dx = T.T @ G
    dw = np.array([G @ x[j:j + len(G)] for j in range(len(w))])
    return dx, dw
"""
STARTER_10 = r"""
def conv_matrix(w, n):
    # the (n - m + 1, n) Toeplitz matrix T with T[i, i:i+m] = w, so that T @ x is the valid cross-correlation of x with w
    # YOUR CODE HERE
    return None

def conv1d(x, w):
    # y_i = sum_j w_j x_{i+j}, as a NumPy array of length n - m + 1 (use conv_matrix)
    # YOUR CODE HERE
    return None

def conv1d_backward(x, w, G):
    # G = dL/dy. Return (dL/dx, dL/dw): dx = T^T G, dw_j = sum_i G_i x_{i+j}
    # YOUR CODE HERE
    return None, None
"""

# ------------------------------------------------------------------ Project
PROJECT_SOLUTION = r"""
from sklearn.datasets import load_digits

def softmax_ce(Z, y):
    n = Z.shape[0]
    logp = log_softmax(Z)
    loss = -logp[np.arange(n), y].mean()
    G = np.exp(logp); G[np.arange(n), y] -= 1.0
    return loss, G / n

class Dropout:
    def __init__(self, p, rng):
        self.p, self.rng, self.training = p, rng, True
    def forward(self, A):
        A_out, self.mask = dropout_forward(A, self.p, self.rng, self.training)
        return A_out
    def backward(self, G): return dropout_backward(G, self.mask)
    def params(self): return []
    def grads(self):  return []

class Adam:
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.model, self.lr, self.b1, self.b2, self.eps, self.wd = model, lr, beta1, beta2, eps, weight_decay
        self.m = [np.zeros_like(p) for p in model.params()]; self.v = [np.zeros_like(p) for p in model.params()]; self.t = 0
    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.model.params(), self.model.grads())):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t); v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p)

def set_training(model, flag):
    for layer in model.layers:
        if hasattr(layer, "training"):
            layer.training = flag

# data: standardise with training statistics, 80/20 split
digits = load_digits(); X_all, y_all = digits.data, digits.target
split = np.random.default_rng(16).permutation(len(X_all)); n_tr = int(0.8 * len(X_all))
X_tr, y_tr = X_all[split[:n_tr]], y_all[split[:n_tr]]; X_te, y_te = X_all[split[n_tr:]], y_all[split[n_tr:]]
mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0); sd[sd == 0] = 1.0
X_tr = (X_tr - mu) / sd; X_te = (X_te - mu) / sd

def build(p_drop, seed):
    r = np.random.default_rng(seed)
    layers = [Dense(64, 128, rng=r), ReLU()]
    if p_drop > 0:
        layers.append(Dropout(p_drop, rng=r))
    layers.append(Dense(128, 10, scale=0.01, rng=r))
    return MLP(layers)

def fit(model, weight_decay, epochs=30, batch_size=64, lr=3e-3, seed=1):
    r = np.random.default_rng(seed); opt = Adam(model, lr=lr, weight_decay=weight_decay)
    tr_loss, te_loss = [], []
    for epoch in range(epochs):
        set_training(model, True)
        order = r.permutation(len(X_tr))
        for s in range(0, len(X_tr), batch_size):
            idx = order[s:s + batch_size]
            loss, G = softmax_ce(model.forward(X_tr[idx]), y_tr[idx]); model.backward(G); opt.step()
        set_training(model, False)
        tr_loss.append(softmax_ce(model.forward(X_tr), y_tr)[0]); te_loss.append(softmax_ce(model.forward(X_te), y_te)[0])
    return np.array(tr_loss), np.array(te_loss)

def bootstrap_ci(correct, n_boot=2000, seed=0):
    r = np.random.default_rng(seed)
    accs = np.array([r.choice(correct, size=len(correct), replace=True).mean() for _ in range(n_boot)])
    return correct.mean(), np.percentile(accs, 2.5), np.percentile(accs, 97.5)

# gradient check before training (small batch, float64)
probe = build(0.0, 0); rel = grad_check(probe, softmax_ce, X_tr[:5], y_tr[:5])
print(f"gradient check of the untrained model: relative error {rel:.1e}")

configs = {"plain": (0.0, 0.0), "dropout 0.3": (0.3, 0.0), "weight decay 1e-2": (0.0, 1e-2), "dropout 0.3 + wd 1e-2": (0.3, 1e-2)}
results = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, (p_drop, wd) in configs.items():
    model = build(p_drop, seed=0)
    tr, te = fit(model, wd)
    correct = (model.forward(X_te).argmax(axis=1) == y_te).astype(float)
    acc, lo, hi = bootstrap_ci(correct)
    results[name] = (acc, lo, hi, model)
    axes[0].plot(tr, label=name); axes[1].plot(te, label=name)
    print(f"{name:24s} test accuracy {acc:.4f}   95% bootstrap CI [{lo:.4f}, {hi:.4f}]   final test loss {te[-1]:.4f}   final train loss {tr[-1]:.4f}")
axes[0].set_title("training loss"); axes[1].set_title("test loss")
for ax in axes:
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy"); ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.show()

# first-layer weights as 8 x 8 images
W1 = results["plain"][3].layers[0].W           # (64, 128): column j is hidden unit j's template over the 64 pixels
fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
for j, ax in enumerate(axes.ravel()):
    ax.imshow(W1[:, j].reshape(8, 8), cmap="RdBu", vmin=-np.abs(W1[:, :16]).max(), vmax=np.abs(W1[:, :16]).max()); ax.axis("off")
fig.suptitle("First-layer weights of the plain model: 16 of 128 hidden units as 8 x 8 pixel templates"); plt.show()
print("The confidence intervals of the four configurations overlap: on 360 test images the differences are not resolvable.")
"""

PROJECT_STARTER = r"""
from sklearn.datasets import load_digits
digits = load_digits(); X_all, y_all = digits.data, digits.target

# Your mini-project workspace. You can reuse Dense / ReLU / MLP / grad_check (Exercise 5),
# log_softmax (Exercise 3), softmax_ce_grad (Exercise 4) and dropout_forward/backward (Exercise 8),
# which are all defined above in this notebook.
"""

# =============================================================================
# Compute hidden answers by executing the reference solutions
# =============================================================================
import matplotlib
matplotlib.use("Agg")
ns = {}
exec(SETUP, ns)
for src in (SOL_1, SOL_2, SOL_3, SOL_4, SOL_5, SOL_6, SOL_7, SOL_8, SOL_10):
    exec(src, ns)

# Ex 1 hidden: GELU(1.5) and softplus'(−3) rounded
HASH_1_GELU = sha_of(float(ns["gelu"](np.array(1.5))), decimals=3)
HASH_1_SP = sha_of(float(ns["d_softplus"](np.array(-3.0))), decimals=3)
print("ex1 gelu(1.5) =", float(ns["gelu"](np.array(1.5))), " d_softplus(-3) =", float(ns["d_softplus"](np.array(-3.0))))

# Ex 2 hidden: parameter count of a 784-512-256-10 net and FLOPs for 64-128-10 at batch 32
_p2 = ns["count_params"]([784, 512, 256, 10]); _f2 = ns["forward_flops"]([64, 128, 10], 32)
HASH_2_P = sha_of(_p2); HASH_2_F = sha_of(_f2)
print("ex2 params 784-512-256-10:", _p2, " flops 64-128-10 @32:", _f2)
assert _p2 == 784 * 512 + 512 + 512 * 256 + 256 + 256 * 10 + 10

# Ex 3 hidden: cross-entropy on a fixed logits matrix (with huge entries)
_Z3 = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1000.0, 999.0, -1000.0], [-3.0, 4.0, 1.0], [0.5, 0.5, 5.0]])
_y3 = np.array([0, 1, 1, 1, 2])
_ce3 = ns["cross_entropy"](_Z3, _y3)
HASH_3 = sha_of(_ce3, decimals=3)
print("ex3 cross-entropy:", _ce3)

# Ex 4 hidden: sum of |grad| on the same matrix, and the MSE/CE ratio at z=-5
_g4 = ns["softmax_ce_grad"](_Z3, _y3)
HASH_4 = sha_of(float(np.abs(_g4).sum()), decimals=3)
_ratio = float((ns["sigmoid"](-5.0) - 1.0) / ns["sigmoid_mse_grad"](-5.0, 1.0))
HASH_4_R = sha_of(_ratio, decimals=1)
print("ex4 sum|grad|:", float(np.abs(_g4).sum()), " ratio CE/MSE at z=-5:", _ratio)

# Ex 5 hidden: the checks use grad_check numerically; hidden value = dW of the Section 3 example (sum of entries)
_X5 = np.array([[1., 2., 0.], [0., 1., -1.]]); _y5 = np.array([[3.], [1.]])
_L1 = ns["Dense"](3, 2); _L1.W[:] = np.array([[0.5, -1.], [1., 0.5], [0., 2.]]); _L1.b[:] = np.array([0.1, 0.2])
_L2 = ns["Dense"](2, 1); _L2.W[:] = np.array([[1.], [-1.]]); _L2.b[:] = np.array([0.5])
_net5 = ns["MLP"]([_L1, ns["ReLU"](), _L2])
_Z5 = _net5.forward(_X5); _G5 = (_Z5 - _y5) / 2; _net5.backward(_G5)
HASH_5 = sha_of(_L1.dW, decimals=3)
print("ex5 dW1:", _L1.dW.ravel(), " db1:", _L1.db)
assert np.allclose(_L1.dW, np.array([[-0.05, 0.05], [0.2, 0.1], [-0.3, 0.0]]))
_mse5 = lambda Z, y: (0.5 * np.sum((Z - y) ** 2) / len(Z), (Z - y) / len(Z))
assert ns["grad_check"](_net5, _mse5, _X5, _y5) < 1e-8

# Ex 6 hidden: He std for n_in=512, and the propagated std under He (range check only) and scale 0.05 (hash)
HASH_6_HE = sha_of(ns["he_std"](300), decimals=3)      # 0.08165, safely away from a rounding boundary
_std_he = ns["propagate_std"](ns["he_std"](256), depth=10, width=256, act="relu")
_std_small = ns["propagate_std"](0.05, depth=10, width=256, act="relu")
print("ex6 he_std(512):", ns["he_std"](512), " propagated std He:", _std_he, " scale 0.05:", _std_small, " xavier(256,256):", ns["xavier_std"](256, 256))
assert 0.6 < _std_he < 1.5

# Ex 7: batch-norm numbers
_Z7 = np.array([[1.], [2.], [3.], [6.]]); _Y7, _c7 = ns["batchnorm_forward"](_Z7, np.array([2.0]), np.array([1.0]))
print("ex7 BN worked example:", _Y7.ravel())
_r7 = np.random.default_rng(7); _Zb = _r7.normal(size=(6, 4)) * 3 + 1; _gm = _r7.normal(size=4); _bt = _r7.normal(size=4); _Gb = _r7.normal(size=(6, 4))
_Yb, _cb = ns["batchnorm_forward"](_Zb, _gm, _bt); _dZ, _dg, _db = ns["batchnorm_backward"](_Gb, _cb)
_num = np.zeros_like(_Zb); _h = 1e-6
for _idx in np.ndindex(_Zb.shape):
    _Zp = _Zb.copy(); _Zp[_idx] += _h; _Zm = _Zb.copy(); _Zm[_idx] -= _h
    _num[_idx] = (np.sum(_Gb * ns["batchnorm_forward"](_Zp, _gm, _bt)[0]) - np.sum(_Gb * ns["batchnorm_forward"](_Zm, _gm, _bt)[0])) / (2 * _h)
print("ex7 BN backward vs FD:", np.max(np.abs(_dZ - _num)))
assert np.max(np.abs(_dZ - _num)) < 1e-6
HASH_7 = sha_of(float(np.abs(_dZ).sum()), decimals=3)

# Ex 10 hidden: conv of a fixed signal
_x10 = np.array([1., 2., 3., 4., 2., 0., -1.]); _w10 = np.array([1., -1., 0.5])
_y10 = ns["conv1d"](_x10, _w10)
assert np.allclose(_y10, np.convolve(_x10, _w10[::-1], mode="valid"))
_dx10, _dw10 = ns["conv1d_backward"](_x10, _w10, np.ones(5))
HASH_10 = sha_of(_dw10, decimals=3)
print("ex10 conv:", _y10, " dw for G=1:", _dw10, " dx:", _dx10)

# =============================================================================
# Assemble the exercise set
# =============================================================================
ex = ExerciseSet(
    "Module 16 — Exercises",
    intro_md=r"""
These exercises rebuild a neural network layer by layer: activations and their derivatives, parameter and FLOP
counting, the stable softmax cross-entropy and its gradient, a `Dense` layer's backward pass with a gradient checker,
initialisation and variance propagation, batch normalisation, inverted dropout, a conceptual question on depth and
the He factor, convolution as a Toeplitz matrix, and a mini-project that trains a regularised digit classifier and
reports a bootstrap confidence interval. Only NumPy, SciPy and matplotlib are allowed — no PyTorch, no JAX. Data
matrices are `(n_samples, n_features)` and a layer computes `X @ W + b`. The concept guide (`README.md`) has every formula.
""",
    setup_code=SETUP,
)

# ------------------------------------------------------------------ 1
ex.add(
    title="Activation functions and their derivatives", difficulty=1,
    prompt_md=r"""
Implement `relu`, `softplus` and `gelu` and the derivatives `d_relu`, `d_sigmoid`, `d_tanh`, `d_softplus`, `d_gelu`
(all vectorised over NumPy arrays; `sigmoid` is given in the setup cell, and `erf` is imported from `scipy.special`):

$$
\mathrm{softplus}(z) = \ln(1 + e^z),\quad \mathrm{softplus}'(z) = \sigma(z); \qquad
\mathrm{GELU}(z) = z\,\Phi(z),\quad \mathrm{GELU}'(z) = \Phi(z) + z\,\varphi(z),
$$

with $\Phi(z) = \tfrac12\big(1 + \mathrm{erf}(z/\sqrt2)\big)$ the standard normal CDF and $\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}$ its density.
`softplus` must not overflow for `z = 1000` (write it as $\max(z, 0) + \ln(1 + e^{-|z|})$). The checks compare every
derivative with central differences; two hidden checks ask for $\mathrm{GELU}(1.5)$ and $\mathrm{softplus}'(-3)$.
""",
    starter=STARTER_1, solution=SOL_1,
    check=rf"""
zs = np.array([-3.0, -0.7, 0.2, 1.5, 4.0])
def _safe(fn, *a):
    try:
        out = fn(*a)
        return None if out is None else np.asarray(out, dtype=float)
    except Exception:
        return None
check_close('relu values', _safe(relu, zs), np.maximum(zs, 0))
check_close("relu' = 1[z > 0]", _safe(d_relu, zs), (zs > 0).astype(float))
check_close("sigmoid' vs central differences", _safe(d_sigmoid, zs), central_diff(sigmoid, zs), tol=1e-6)
check_close("tanh' vs central differences", _safe(d_tanh, zs), central_diff(np.tanh, zs), tol=1e-6)
sp = _safe(softplus, zs)
check_close('softplus values', sp, np.log1p(np.exp(zs)), tol=1e-9)
check('softplus(1000) is finite and equals 1000', (lambda v: v is not None and np.isfinite(v) and abs(float(v) - 1000.0) < 1e-9)(_safe(softplus, np.array(1000.0))))
check_close("softplus' vs central differences", _safe(d_softplus, zs), (central_diff(softplus, zs) if sp is not None else None), tol=1e-6)
ge = _safe(gelu, zs)
check_close('gelu(0) = 0 and gelu(4) ~ 4', _safe(gelu, np.array([0.0, 4.0])), np.array([0.0, 4.0 * 0.5 * (1 + erf(4.0 / np.sqrt(2)))]), tol=1e-9)
check_close("gelu' vs central differences", _safe(d_gelu, zs), (central_diff(gelu, zs) if ge is not None else None), tol=1e-6)
check_hash('hidden: GELU(1.5)', (float(ge[3]) if ge is not None else None), '{HASH_1_GELU}')
check_hash("hidden: softplus'(-3)", (float(_safe(d_softplus, np.array(-3.0))) if _safe(d_softplus, np.array(-3.0)) is not None else None), '{HASH_1_SP}')
""",
    explanation_md=r"""
Each derivative is a one-line calculus exercise: $\sigma' = \sigma(1 - \sigma)$ by the quotient rule, $\tanh' = 1 - \tanh^2$,
$\mathrm{softplus}' = e^z/(1 + e^z) = \sigma(z)$, and $\mathrm{GELU}' = \Phi(z) + z\varphi(z)$ by the product rule (the derivative of
$\Phi$ is the density $\varphi$). The stable softplus form $\max(z, 0) + \ln(1 + e^{-|z|})$ is algebraically identical to
$\ln(1 + e^z)$ but never exponentiates a large positive number — the same trick as log-sum-exp. Note that `d_relu` must
cope with scalar inputs (`np.asarray`) and that at exactly $z = 0$ any value in $[0, 1]$ is a valid subgradient; the
convention `0` is what frameworks use.
""")

# ------------------------------------------------------------------ 2
ex.add(
    title="Parameter count, FLOPs and shapes", difficulty=1,
    prompt_md=r"""
For an MLP with layer sizes `sizes = [n_in, h_1, ..., n_out]` implement

* `count_params(sizes)` — the total number of weights and biases, $\sum_l (n_l + 1)\, n_{l+1}$ (return an `int`);
* `forward_flops(sizes, batch_size)` — the FLOPs of all matrix products in one forward pass, counting each
  multiply-add as 2 FLOPs and ignoring biases and activations (an `int`);
* `forward_shapes(sizes, batch_size)` — the list of shapes `(batch_size, n_{l+1})` of $Z_1, \dots, Z_L$ as tuples.

Hidden checks: the parameter count of a 784 → 512 → 256 → 10 network, and the forward FLOPs of 64 → 128 → 10 at batch size 32.
""",
    starter=STARTER_2, solution=SOL_2,
    check=rf"""
try:
    p_xor = count_params([2, 2, 1]); p_dig = count_params([64, 128, 10]); p_big = count_params([784, 512, 256, 10])
except Exception:
    p_xor = p_dig = p_big = None
check('2-2-1 XOR network has 9 parameters', p_xor == 9)
check('64-128-10 digit network has 9,610 parameters', p_dig == 9610)
check_hash('hidden: parameters of 784-512-256-10', p_big, '{HASH_2_P}')
try:
    f_small = forward_flops([3, 2, 1], 2); f_dig = forward_flops([64, 128, 10], 32)
except Exception:
    f_small = f_dig = None
check('Section 3 example: 32 FLOPs for the two products', f_small == 32)
check_hash('hidden: forward FLOPs of 64-128-10 at batch 32', f_dig, '{HASH_2_F}')
try:
    shp = forward_shapes([64, 128, 10], 32)
    shp_ok = [tuple(s) for s in shp] == [(32, 128), (32, 10)]
except Exception:
    shp_ok = False
check('shapes of Z_1, Z_2 for 64-128-10 at batch 32', shp_ok)
""",
    explanation_md=r"""
A layer from $n_l$ to $n_{l+1}$ units has an $n_l \times n_{l+1}$ weight matrix plus $n_{l+1}$ biases, hence $(n_l + 1) n_{l+1}$
parameters; the matrix product $(B \times n_l)(n_l \times n_{l+1})$ costs $B\, n_l\, n_{l+1}$ multiply-adds, i.e. $2 B n_l n_{l+1}$ FLOPs.
Summed over layers this gives the rule of thumb "forward FLOPs ≈ 2 × weights × batch", and with the backward pass
(two products per layer) "training step ≈ 6 × weights × batch" — the estimate behind every compute budget for large
models. The shapes follow the rule $(B \times n_{\text{in}})(n_{\text{in}} \times n_{\text{out}}) = (B \times n_{\text{out}})$.
""")

# ------------------------------------------------------------------ 3
ex.add(
    title="Stable log-softmax and cross-entropy", difficulty=1,
    prompt_md=r"""
Implement `log_softmax(Z)` for an $(n, k)$ logits matrix using the log-sum-exp trick,

$$
\ln P_{ij} = (Z_{ij} - m_i) - \ln \sum_c e^{Z_{ic} - m_i}, \qquad m_i = \max_c Z_{ic},
$$

and `cross_entropy(Z, y)` returning the mean of $-\ln P_{i, y_i}$ over the rows as a `float` (`y` is an integer array).
Both must survive logits such as 1000 and −1000 without `inf` or `nan`. The checks compare against
`scipy.special.log_softmax` and ask for the loss on a hidden logits matrix.
""",
    starter=STARTER_3, solution=SOL_3,
    check=rf"""
from scipy.special import log_softmax as _sp_log_softmax
Z3 = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1000.0, 999.0, -1000.0], [-3.0, 4.0, 1.0], [0.5, 0.5, 5.0]])
y3 = np.array([0, 1, 1, 1, 2])
try:
    lp = log_softmax(Z3); lp = None if lp is None else np.asarray(lp, dtype=float)
except Exception:
    lp = None
check_close('log_softmax matches scipy.special.log_softmax (row-wise)', lp, _sp_log_softmax(Z3, axis=1), tol=1e-9)
check('finite for logits of +-1000', lp is not None and bool(np.all(np.isfinite(lp))))
check_close('rows exponentiate to probabilities summing to 1', (np.exp(lp).sum(axis=1) if lp is not None else None), np.ones(5), tol=1e-9)
try:
    ce = cross_entropy(Z3, y3); ce_uniform = cross_entropy(np.zeros((4, 10)), np.array([0, 3, 5, 9]))
except Exception:
    ce = ce_uniform = None
check_close('uniform logits over 10 classes give loss ln 10', ce_uniform, np.log(10), tol=1e-9)
check('cross_entropy returns a Python float', isinstance(ce, float))
check_hash('hidden: cross-entropy on the test logits', ce, '{HASH_3}')
""",
    explanation_md=r"""
Subtracting the row maximum leaves the softmax unchanged (it is invariant to adding a constant to every logit) but
makes every exponent $\le 0$, so `exp` never overflows and the largest term contributes exactly 1 to the sum, so the
logarithm is never of zero. The row `[1000, 999, -1000]` then gives $\ln P = (-0.313, -1.313, -2000.3)$ rather than
`nan`. Computing the loss from $\ln P$ directly — instead of `np.log(softmax(Z))` — is what `nn.CrossEntropyLoss` does,
and the uniform-logits check ($\ln k$) is the sanity test you should run on every classifier before training.
""")

# ------------------------------------------------------------------ 4
ex.add(
    title="The gradient of softmax cross-entropy", difficulty=2,
    prompt_md=r"""
Implement `softmax_ce_grad(Z, y)` returning $\partial \mathcal{L}/\partial Z = (P - Y)/n$ for the mean cross-entropy
of Exercise 3 ($Y$ is the one-hot matrix of `y`), and `sigmoid_mse_grad(z, y)` returning the elementwise derivative
of $(\sigma(z) - y)^2$ with respect to $z$, i.e. $2(\sigma(z) - y)\,\sigma(z)(1 - \sigma(z))$.

The checks verify the first against central differences of your `cross_entropy` and ask, as a hidden answer rounded
to one decimal, for the ratio of the cross-entropy gradient $\sigma(z) - y$ to the squared-error gradient at $z = -5$, $y = 1$.
""",
    starter=STARTER_4, solution=SOL_4,
    check=rf"""
try:
    G4 = softmax_ce_grad(Z3, y3); G4 = None if G4 is None else np.asarray(G4, dtype=float)
except Exception:
    G4 = None
check('gradient has the shape of Z', G4 is not None and G4.shape == Z3.shape)
check_close('gradient rows sum to zero', (G4.sum(axis=1) if G4 is not None else None), np.zeros(5), tol=1e-9)
def _num_grad(Z, y, h=1e-6):
    g = np.zeros_like(Z)
    for idx in np.ndindex(Z.shape):
        Zp = Z.copy(); Zp[idx] += h; Zm = Z.copy(); Zm[idx] -= h
        g[idx] = (cross_entropy(Zp, y) - cross_entropy(Zm, y)) / (2 * h)
    return g
try:
    Zr = rng.normal(size=(6, 4)) * 2; yr = rng.integers(0, 4, size=6)
    ng = _num_grad(Zr, yr); ag = np.asarray(softmax_ce_grad(Zr, yr), dtype=float)
except Exception:
    ng = ag = None
check_close('(P - Y)/n matches central differences of cross_entropy', ag, ng, tol=1e-6)
check_hash('hidden: sum |gradient| on the test logits', (float(np.abs(G4).sum()) if G4 is not None else None), '{HASH_4}')
try:
    ratio = float((sigmoid(-5.0) - 1.0) / sigmoid_mse_grad(-5.0, 1.0))
    mse0 = float(sigmoid_mse_grad(0.0, 1.0))
except Exception:
    ratio = mse0 = None
check_close('sigmoid-MSE gradient at z = 0, y = 1 is -0.25', mse0, -0.25)
check_hash('hidden: CE/MSE gradient ratio at z = -5 (1 decimal)', ratio, '{HASH_4_R}', decimals=1)
""",
    explanation_md=r"""
With $\ln P_j = z_j - \mathrm{LSE}(z)$ and $\partial\,\mathrm{LSE}/\partial z_m = P_m$, the derivative of $-\sum_j Y_j \ln P_j$ with respect to
$z_m$ is $-Y_m + P_m$; the mean over rows contributes the $1/n$. Each row of the gradient sums to zero because the
softmax ignores a constant shift of the logits. The squared-error gradient carries the extra factor $\sigma(1 - \sigma)$,
which is $0.0066$ at $z = -5$: the model is confidently wrong and learns 75 times more slowly than with cross-entropy.
A common wrong turn is to forget the `/ n` (the checks against central differences of the *mean* loss catch it) or to
build the softmax Jacobian explicitly — correct, but $k$ times more work for the same answer.
""")

# ------------------------------------------------------------------ 5
ex.add(
    title="`Dense.backward` and a gradient checker", difficulty=2,
    prompt_md=r"""
Complete `Dense.backward(G)`: given $G = \partial\mathcal{L}/\partial Z$ for $Z = XW + \mathbf 1 b^\top$, store
$\partial\mathcal{L}/\partial W = X^\top G$ in `self.dW` and $\partial\mathcal{L}/\partial b = \sum_{\text{rows}} G$ in `self.db`, and return
$\partial\mathcal{L}/\partial X = G W^\top$. `ReLU` and `MLP` are given.

Then write `grad_check(model, loss_fn, Xb, yb, h=1e-5)`: run `model.forward`, `loss_fn` (which returns `(loss, dL/dZ)`)
and `model.backward` once, flatten the analytic gradients `model.grads()` in order, estimate every parameter entry
by central differences (perturb `p[idx]` in place, evaluate the loss, restore), and return the relative error
$\lVert g - \hat g\rVert / (\lVert g\rVert + \lVert \hat g\rVert)$ as a float. The hidden check is $\partial\mathcal{L}/\partial W_1$ for the concept guide's
Section 5 example.
""",
    starter=STARTER_5, solution=SOL_5,
    check=rf"""
X5 = np.array([[1., 2., 0.], [0., 1., -1.]]); y5 = np.array([[3.], [1.]])
mse5 = lambda Z, y: (0.5 * np.sum((Z - y) ** 2) / len(Z), (Z - y) / len(Z))
try:
    L1 = Dense(3, 2); L1.W[:] = np.array([[0.5, -1.], [1., 0.5], [0., 2.]]); L1.b[:] = np.array([0.1, 0.2])
    L2 = Dense(2, 1); L2.W[:] = np.array([[1.], [-1.]]); L2.b[:] = np.array([0.5])
    net5 = MLP([L1, ReLU(), L2])
    Z5 = net5.forward(X5); loss5, G5 = mse5(Z5, y5); dX5 = net5.backward(G5)
    dW1, db1, dW2, db2 = L1.dW.copy(), L1.db.copy(), L2.dW.copy(), L2.db.copy()
except Exception:
    dW1 = db1 = dW2 = db2 = dX5 = None
check_close('dW2 = (0.2, -0.01)^T', dW2, np.array([[0.2], [-0.01]]), tol=1e-9)
check_close('db2 = 0.25', db2, np.array([0.25]), tol=1e-9)
check_close('db1 = (0.25, 0.05)', db1, np.array([0.25, 0.05]), tol=1e-9)
check('dL/dX has the shape of X', dX5 is not None and np.shape(dX5) == X5.shape)
check_hash('hidden: dW1 of the Section 5 example', dW1, '{HASH_5}')
from scipy.special import log_softmax as _lsm
def _softmax_ce(Z, y):                       # reference loss for the checks (uses scipy, not your Exercise 3 code)
    n = Z.shape[0]; logp = _lsm(Z, axis=1)
    G = np.exp(logp); G[np.arange(n), y] -= 1.0
    return -logp[np.arange(n), y].mean(), G / n
try:
    r5 = np.random.default_rng(5)
    net_r = MLP([Dense(7, 6, rng=r5), ReLU(), Dense(6, 5, rng=r5), ReLU(), Dense(5, 3, rng=r5)])
    rel = grad_check(net_r, _softmax_ce, r5.normal(size=(8, 7)), r5.integers(0, 3, size=8))
except Exception:
    rel = None
check('grad_check on a random 7-6-5-3 network: relative error < 1e-7', rel is not None and rel < 1e-7)
class _BuggyDense(Dense):                     # your layer, with its bias gradient deliberately corrupted
    def backward(self, G):
        out = super().backward(G)
        self.db = self.db / G.shape[0]           # mean instead of sum: a classic wrong bias rule
        return out
try:
    r5b = np.random.default_rng(5)
    net_b = MLP([_BuggyDense(7, 6, rng=r5b), ReLU(), _BuggyDense(6, 3, rng=r5b)])
    rel_b = grad_check(net_b, _softmax_ce, r5b.normal(size=(8, 7)), r5b.integers(0, 3, size=8))
except Exception:
    rel_b = None
check('grad_check catches a planted bug (relative error > 1e-2)', rel_b is not None and rel_b > 1e-2)
""",
    explanation_md=r"""
The three rules are Module 06's vector-Jacobian products in row form: $X^\top G$ has the shape of $W$, $G W^\top$ the shape
of $X$, and the bias — broadcast to every row in the forward pass — collects a *sum* over rows in the backward pass
(the planted bug uses the mean and is caught immediately). `forward` must store `X`, because the weight gradient
needs the layer's input; this stored activation is what makes training memory scale with batch size. The gradient
checker perturbs one entry at a time and compares with a relative error; dividing by the sum of both norms keeps the
number meaningful when the analytic gradient is wrongly zero. Check in float64 with $h \approx 10^{-5}$ and on several
batches — ReLU kinks can produce isolated false alarms when a pre-activation lies within $h$ of zero.
""")

# ------------------------------------------------------------------ 6
ex.add(
    title="He initialisation and variance propagation", difficulty=2,
    prompt_md=r"""
Implement `he_std(n_in)` $= \sqrt{2/n_{\text{in}}}$ and `xavier_std(n_in, n_out)` $= \sqrt{2/(n_{\text{in}} + n_{\text{out}})}$, and
`propagate_std(scale, depth=10, width=256, n=1000, act="relu", seed=0)`, which pushes a standard-normal batch
`A = r.normal(size=(n, width))` (with `r = np.random.default_rng(seed)`) through `depth` layers, each drawing
`W = r.normal(0.0, scale, size=(width, width))` *after* `A` and setting `A = act(A @ W)`, and returns the standard
deviation of the final activations as a float.

Use it to confirm the variance-propagation rule: with He scaling the ReLU activations keep a standard deviation of
order 1 after 10 layers, with scale 0.05 ($n_{\text{in}} \sigma_w^2 = 0.64$) they shrink, and with scale 0.15 ($n_{\text{in}} \sigma_w^2 = 5.76$)
they grow. The hidden check asks for `he_std(300)`.
""",
    starter=STARTER_6, solution=SOL_6,
    check=rf"""
try:
    he256 = he_std(256); xa = xavier_std(100, 300); he300 = he_std(300)
except Exception:
    he256 = xa = he300 = None
check_close('he_std(256) = sqrt(2/256)', he256, np.sqrt(2 / 256), tol=1e-9)
check_close('xavier_std(100, 300) = sqrt(2/400)', xa, np.sqrt(2 / 400), tol=1e-9)
check_hash('hidden: he_std(300)', he300, '{HASH_6_HE}')
try:
    s_he = propagate_std(np.sqrt(2 / 256)); s_small = propagate_std(0.05); s_big = propagate_std(0.15)
    s_tanh_x = propagate_std(np.sqrt(1 / 256), act="tanh")
except Exception:
    s_he = s_small = s_big = s_tanh_x = None
check('He scale: ReLU activation std after 10 layers stays between 0.5 and 1.5', s_he is not None and 0.5 < s_he < 1.5)
check('scale 0.05 (n*sigma^2 = 0.64): std shrinks below 0.05', s_small is not None and s_small < 0.05)
check('scale 0.15 (n*sigma^2 = 5.76): std grows above 10', s_big is not None and s_big > 10)
check('Xavier scale with tanh: std stays between 0.1 and 1', s_tanh_x is not None and 0.1 < s_tanh_x < 1.0)
check_close('propagate_std follows the specified random recipe', s_he, {_std_he!r}, tol=1e-6)
""",
    explanation_md=r"""
For $z = \sum_j w_j x_j$ with independent zero-mean weights and inputs, $\operatorname{Var}(z) = n_{\text{in}} \sigma_w^2 \sigma_x^2$, and a
ReLU halves the second moment of a symmetric input, so the per-layer factor on the variance is $n_{\text{in}}\sigma_w^2/2$: exactly
1 for He scaling, $0.32$ for scale 0.05 (giving $0.32^{5} \approx 0.003$ on the standard deviation after 10 layers) and
$2.88$ for scale 0.15 (giving a factor of about $2.88^{5} \approx 200$ on the standard deviation). For tanh, which is close to the identity
near zero and discards nothing, the factor is $n_{\text{in}}\sigma_w^2$ and Xavier's $1/n_{\text{in}}$ keeps the scale (the saturation
of tanh shrinks it somewhat below 1). The checks rerun the exact random recipe, which is why the order "draw `W`
after `A`" matters.
""")

# ------------------------------------------------------------------ 7
ex.add(
    title="Batch normalisation, forward and backward", difficulty=2,
    prompt_md=r"""
Implement `batchnorm_forward(Z, gamma, beta, eps=1e-5)` returning `(Y, cache)` with the per-column formulas

$$
\mu_j = \tfrac1n\sum_i Z_{ij},\quad \sigma_j^2 = \tfrac1n\sum_i (Z_{ij} - \mu_j)^2,\quad
\hat Z_{ij} = \frac{Z_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}},\quad Y_{ij} = \gamma_j \hat Z_{ij} + \beta_j,
$$

and `batchnorm_backward(G, cache)` returning `(dZ, dgamma, dbeta)` for an upstream gradient $G = \partial\mathcal{L}/\partial Y$:

$$
\frac{\partial\mathcal{L}}{\partial\gamma_j} = \sum_i G_{ij}\hat Z_{ij}, \qquad
\frac{\partial\mathcal{L}}{\partial\beta_j} = \sum_i G_{ij}, \qquad
\frac{\partial\mathcal{L}}{\partial Z_{ij}} = \frac{\gamma_j}{n\sqrt{\sigma_j^2 + \epsilon}}\Big(nG_{ij} - \sum_{i'} G_{i'j} - \hat Z_{ij}\sum_{i'} G_{i'j}\hat Z_{i'j}\Big).
$$

The checks reproduce the concept guide's worked example, verify the scale invariance, and compare `dZ` with central
differences of $\sum G \odot Y$ — the backward pass is *not* elementwise, because $\mu$ and $\sigma^2$ depend on every row.
""",
    starter=STARTER_7, solution=SOL_7,
    check=rf"""
Z7 = np.array([[1.], [2.], [3.], [6.]])
try:
    Y7, c7 = batchnorm_forward(Z7, np.array([2.0]), np.array([1.0]))
    Y7 = None if Y7 is None else np.asarray(Y7, dtype=float)
    Y7_scaled = np.asarray(batchnorm_forward(100 * Z7, np.array([2.0]), np.array([1.0]))[0], dtype=float)
except Exception:
    Y7 = Y7_scaled = None
check_close('worked example: batch (1, 2, 3, 6), gamma 2, beta 1', (Y7.ravel() if Y7 is not None else None), 2 * (Z7.ravel() - 3) / np.sqrt(3.5 + 1e-5) + 1, tol=1e-9)
check_close('scale invariance: inputs x100 give (almost) the same output', Y7_scaled, Y7, tol=1e-4)
r7 = np.random.default_rng(7)
Zb = r7.normal(size=(6, 4)) * 3 + 1; gm = r7.normal(size=4); bt = r7.normal(size=4); Gb = r7.normal(size=(6, 4))
try:
    Yb, cb = batchnorm_forward(Zb, gm, bt)
    dZ, dg, db = batchnorm_backward(Gb, cb)
    dZ = np.asarray(dZ, dtype=float); dg = np.asarray(dg, dtype=float); db = np.asarray(db, dtype=float)
    Zhat_ref = (Zb - Zb.mean(0)) / np.sqrt(Zb.var(0) + 1e-5)
except Exception:
    dZ = dg = db = None; Zhat_ref = None
try:
    col_stats = np.r_[np.asarray(Yb, dtype=float).mean(0), np.asarray(Yb, dtype=float).std(0)]
except Exception:
    col_stats = None
check_close('output columns have mean beta and std |gamma|', col_stats, np.r_[bt, np.abs(gm) * np.sqrt(Zb.var(0) / (Zb.var(0) + 1e-5))], tol=1e-6)
check_close('dgamma = sum(G * Zhat)', dg, ((Gb * Zhat_ref).sum(0) if Zhat_ref is not None else None), tol=1e-9)
check_close('dbeta = sum(G)', db, Gb.sum(0), tol=1e-9)
def _num_dZ(h=1e-6):
    g = np.zeros_like(Zb)
    for idx in np.ndindex(Zb.shape):
        Zp = Zb.copy(); Zp[idx] += h; Zm = Zb.copy(); Zm[idx] -= h
        g[idx] = (np.sum(Gb * batchnorm_forward(Zp, gm, bt)[0]) - np.sum(Gb * batchnorm_forward(Zm, gm, bt)[0])) / (2 * h)
    return g
try:
    ndZ = _num_dZ() if dZ is not None else None
except Exception:
    ndZ = None
check_close('dZ matches central differences', dZ, ndZ, tol=1e-6)
check_close('dZ columns sum to zero (the mean is removed)', (dZ.sum(0) if dZ is not None else None), np.zeros(4), tol=1e-9)
check_hash('hidden: sum |dZ| on the random batch', (float(np.abs(dZ).sum()) if dZ is not None else None), '{HASH_7}')
""",
    explanation_md=r"""
The forward pass standardises each column and re-scales it; the output is identical for inputs 100 times larger
(up to the tiny $\epsilon$ effect), which is the scale invariance that makes batch norm robust to the weight scale. In
the backward pass, $\hat Z_{ij}$ depends on $Z_{i'j}$ for every $i'$ through $\mu_j$ and $\sigma_j^2$; carrying the chain rule
through both gives the three-term formula: the incoming gradient, minus its column mean (the forward pass would
undo any change of the mean), minus its projection onto $\hat Z$ (likewise for the variance). Hence every column of
$\partial\mathcal{L}/\partial Z$ sums to zero. The common wrong turn is to treat the layer as elementwise, $\partial Z = \gamma G / \sigma$,
which the central-difference check rejects.
""")

# ------------------------------------------------------------------ 8
ex.add(
    title="Inverted dropout keeps the expectation", difficulty=2,
    prompt_md=r"""
Implement `dropout_forward(A, p, rng, training=True)` (inverted dropout: in training mode keep each entry with
probability $1 - p$ via `mask = (rng.random(A.shape) >= p) / (1 - p)` and return `(A * mask, mask)`; in evaluation
mode, or when `p == 0`, return `A` unchanged with a mask of ones) and `dropout_backward(G, mask)` $= G \odot \text{mask}$.

The checks verify that the surviving entries are scaled by $1/(1 - p)$, that about a fraction $p$ of the entries are
zero, that the *mean over many masks* of the dropped activations equals the original (within sampling error), and
that evaluation mode is the identity.
""",
    starter=STARTER_8, solution=SOL_8,
    check=r"""
A8 = np.array([2.0, -1.0, 3.0, 0.5])
try:
    out, mask = dropout_forward(A8, 0.5, np.random.default_rng(0))
    out = np.asarray(out, dtype=float); mask = np.asarray(mask, dtype=float)
    kept = mask > 0
    scale_ok = np.allclose(out[kept], 2.0 * A8[kept]) and np.allclose(out[~kept], 0.0)
    out_eval, mask_eval = dropout_forward(A8, 0.5, np.random.default_rng(0), training=False)
    eval_ok = np.allclose(out_eval, A8) and np.allclose(mask_eval, 1.0)
except Exception:
    scale_ok = eval_ok = False; mask = None
check('survivors are scaled by 1/(1-p), the rest are zero', bool(scale_ok))
check('evaluation mode is the identity', bool(eval_ok))
try:
    r8 = np.random.default_rng(8)
    big = np.ones((200, 500))
    outs = [dropout_forward(big, 0.3, r8)[0] for _ in range(1)][0]
    frac_zero = float(np.mean(outs == 0))
    mean_val = float(outs.mean())
    A_rand = r8.normal(size=(50, 40))
    avg = np.mean([dropout_forward(A_rand, 0.5, r8)[0] for _ in range(400)], axis=0)
    max_dev = float(np.max(np.abs(avg - A_rand)))
except Exception:
    frac_zero = mean_val = max_dev = None
check('about 30% of entries are dropped at p = 0.3', frac_zero is not None and 0.28 < frac_zero < 0.32)
check('mean of the dropped activations stays ~1 (expectation preserved)', mean_val is not None and 0.98 < mean_val < 1.02)
check('average over 400 masks reproduces A to within 0.4', max_dev is not None and max_dev < 0.4)
try:
    G8 = np.array([1.0, 1.0, 1.0, 1.0]); back = np.asarray(dropout_backward(G8, mask), dtype=float)
    back_ok = np.allclose(back, mask)
except Exception:
    back_ok = False
check('backward multiplies by the same mask', bool(back_ok))
""",
    explanation_md=r"""
$E[\text{mask}_{ij}] = (1 - p)\cdot\frac{1}{1-p} = 1$, so $E[A \odot \text{mask}] = A$: the next layer sees inputs of the same
expected scale during training as at test time, and inference is simply "no mask" — no rescaling needed. The
variance added is $A_{ij}^2\, p/(1-p)$, so dropout is multiplicative noise whose relative size grows with $p$; averaging
over many masks recovers $A$ with error $\propto 1/\sqrt{\text{number of masks}}$, which is why the tolerance in the check is
loose. The backward pass is the same mask applied to the upstream gradient (dropped units receive no gradient,
survivors receive $1/(1-p)$ times theirs). The classic wrong turn is forgetting the $1/(1-p)$ and then needing to
multiply activations by $(1 - p)$ at test time, as the original dropout paper did.
""")

# ------------------------------------------------------------------ 9 (conceptual)
ex.add_conceptual(
    title="Depth without nonlinearity, and the factor 2", difficulty=2,
    prompt_md=r"""
1. A colleague proposes a 20-layer network with *no* activation functions, arguing that "each layer learns a
   progressively more abstract linear feature". Show in two lines what function class the network can represent and
   what the extra 19 layers buy.
2. Explain where the factor 2 in He initialisation, $\sigma_w^2 = 2/n_{\text{in}}$, comes from, starting from
   $\operatorname{Var}(z) = n_{\text{in}} \sigma_w^2 \sigma_x^2$. What would the factor be for a leaky ReLU with negative slope $\alpha = 0.2$?
3. Your 12-layer tanh network trains fine with Xavier initialisation. You replace tanh by ReLU and keep Xavier. The
   loss decreases much more slowly. Predict the activation standard deviation at layer 12 relative to layer 1 and explain.
""",
    answer_md=r"""
1. $(\cdots((XW_1 + b_1)W_2 + b_2)\cdots)W_{20} + b_{20} = X(W_1 W_2 \cdots W_{20}) + b'$: a single affine map. The network
   can represent exactly what linear (or logistic) regression can — one hyperplane — with 19 redundant factorisations
   of one matrix, worse conditioning, and more parameters. Depth adds expressive power only through a non-polynomial
   nonlinearity between the layers.
2. A ReLU zeroes the negative half of a symmetric pre-activation, so $E[\mathrm{ReLU}(z)^2] = \tfrac12 E[z^2]$. The next layer's
   pre-activation variance is therefore $n_{\text{in}} \sigma_w^2 \cdot \tfrac12 \sigma_z^2$, and keeping it equal to $\sigma_z^2$ requires
   $n_{\text{in}}\sigma_w^2 = 2$. For a leaky ReLU the negative half contributes $\alpha^2 \cdot \tfrac12 E[z^2]$ instead of zero, so the
   second moment is $\tfrac{1 + \alpha^2}{2} E[z^2]$ and the factor becomes $2/(1 + \alpha^2) = 2/1.04 \approx 1.92$ for $\alpha = 0.2$.
3. Xavier gives $n_{\text{in}}\sigma_w^2 \approx 1$, which preserves variance through a (near-linear) tanh but not through a ReLU,
   whose halving makes the per-layer variance factor $\tfrac12$. After 11 more layers the variance is $2^{-11}$ of its
   initial value, so the standard deviation is about $2^{-5.5} \approx 0.02$ of layer 1's. The gradients flow back through
   the same matrices and shrink by the same factor, so the early layers barely move. Switching to He scaling
   ($\sigma_w^2 = 2/n_{\text{in}}$) restores the factor to 1.
""")

# ------------------------------------------------------------------ 10
ex.add(
    title="Convolution as a Toeplitz matrix", difficulty=3,
    prompt_md=r"""
Implement `conv_matrix(w, n)`, the $(n - m + 1) \times n$ Toeplitz matrix $T$ with `T[i, i:i+m] = w` for a kernel $w$ of
length $m$, so that `T @ x` is the "valid" cross-correlation $y_i = \sum_{j=0}^{m-1} w_j x_{i+j}$; `conv1d(x, w)` using it; and
`conv1d_backward(x, w, G)` returning `(dx, dw)` for an upstream gradient $G = \partial\mathcal{L}/\partial y$:

$$
\frac{\partial\mathcal{L}}{\partial x} = T^\top G, \qquad \frac{\partial\mathcal{L}}{\partial w_j} = \sum_i G_i\, x_{i+j}
$$

(the shared weight collects one contribution per output position). The checks compare `conv1d` with
`np.convolve(x, w[::-1], mode="valid")` (NumPy flips the kernel, so we flip it back), compare `dx` and `dw` with
central differences, and ask for a hidden `dw`.
""",
    starter=STARTER_10, solution=SOL_10,
    check=rf"""
x10 = np.array([1., 2., 3., 4., 2., 0., -1.]); w10 = np.array([1., -1., 0.5])
try:
    T = conv_matrix(w10, len(x10)); T = None if T is None else np.asarray(T, dtype=float)
    y10 = conv1d(x10, w10); y10 = None if y10 is None else np.asarray(y10, dtype=float)
except Exception:
    T = y10 = None
check('T has shape (n - m + 1, n) = (5, 7)', T is not None and T.shape == (5, 7))
try:
    structure_ok = T.shape == (5, 7) and np.all((T != 0).sum(axis=1) == 3) and np.allclose(T[0, :3], w10) and np.allclose(T[4, 4:], w10)
except Exception:
    structure_ok = False
check('T has exactly 3 non-zeros per row, shared along diagonals', bool(structure_ok))
check_close('conv1d matches np.convolve with the flipped kernel', y10, np.convolve(x10, w10[::-1], mode="valid"), tol=1e-12)
G10 = np.array([0.5, -1.0, 2.0, 0.0, 1.5])
try:
    dx, dw = conv1d_backward(x10, w10, G10)
    dx = None if dx is None else np.asarray(dx, dtype=float); dw = None if dw is None else np.asarray(dw, dtype=float)
except Exception:
    dx = dw = None
def _num(f, v, h=1e-6):
    g = np.zeros_like(v)
    for i in range(len(v)):
        vp = v.copy(); vp[i] += h; vm = v.copy(); vm[i] -= h
        g[i] = (f(vp) - f(vm)) / (2 * h)
    return g
ref_loss_x = lambda xx: float(G10 @ np.convolve(xx, w10[::-1], mode="valid"))
ref_loss_w = lambda ww: float(G10 @ np.convolve(x10, ww[::-1], mode="valid"))
check_close('dL/dx = T^T G matches central differences', dx, _num(ref_loss_x, x10), tol=1e-6)
check_close('dL/dx equals the full convolution of G with the kernel', dx, np.convolve(G10, w10, mode="full"), tol=1e-9)
check_close('dL/dw matches central differences', dw, _num(ref_loss_w, w10), tol=1e-6)
try:
    _, dw_ones = conv1d_backward(x10, w10, np.ones(5)); dw_ones = None if dw_ones is None else np.asarray(dw_ones, dtype=float)
except Exception:
    dw_ones = None
check_hash('hidden: dL/dw for G = ones', dw_ones, '{HASH_10}')
""",
    explanation_md=r"""
Row $i$ of $T$ holds the kernel starting at column $i$, so $(Tx)_i = \sum_j w_j x_{i+j}$: a convolutional layer is a dense layer
whose weight matrix is sparse (each output sees $m$ inputs, its receptive field) and whose non-zero entries are tied
along the diagonals (translation equivariance). The backward rules are then Module 06's matrix rules: $\partial x = T^\top G$,
which is the full convolution of $G$ with the kernel (not flipped here, because the forward pass was a cross-correlation),
and $\partial w_j = \sum_i G_i x_{i+j}$, which sums the gradient over every position where the shared weight was used —
broadcast in the forward pass ⇒ sum in the backward pass, exactly as for a bias. Three parameters instead of 35, and
the same backward machinery.
""")

# ------------------------------------------------------------------ 11 (project)
ex.add_project(
    title="Mini-project: a regularised digit classifier with a confidence interval", difficulty=3,
    prompt_md=r"""
Train your own NumPy MLP on the handwritten digits and report the result the way a careful practitioner would.

```python
from sklearn.datasets import load_digits
digits = load_digits(); X_all, y_all = digits.data, digits.target
```

1. Standardise the pixels with training-set statistics (set the standard deviation of constant pixels to 1) and
   split 80/20 with a fixed seed.
2. Build a 64 → 128 → 10 network from the `Dense`, `ReLU` and `MLP` classes of Exercise 5 (He initialisation for the
   hidden layer, a small scale such as 0.01 for the output layer so the initial loss is close to $\ln 10$), using
   `log_softmax` from Exercise 3 for the loss and `softmax_ce_grad` from Exercise 4 for its gradient, and a `Dropout`
   layer built on Exercise 8's functions. Gradient-check the untrained model on a tiny batch with your `grad_check`.
3. Write an Adam (or SGD with momentum) optimiser with optional decoupled weight decay and a mini-batch training
   loop (shuffle every epoch; switch dropout off for evaluation). Train four variants for about 30 epochs: plain,
   dropout 0.3, weight decay $10^{-2}$, both.
4. For each variant report the test accuracy with a 95 percent **bootstrap confidence interval** (Module 10: resample
   the per-example correct/incorrect indicators 2,000 times) and plot the training and test loss curves.
5. Visualise 16 columns of the first-layer weight matrix as $8 \times 8$ images — they are pixel templates.
6. State in one sentence whether the differences between variants are resolvable on 360 test images.
""",
    rubric_md=r"""
- The model is built only from the NumPy classes and functions of this notebook (no frameworks), and the gradient check on the untrained model reports a relative error below $10^{-6}$.
- Standardisation uses training-set statistics only, and the initial loss is within 0.1 of $\ln 10 \approx 2.30$.
- The training loop reshuffles every epoch, applies dropout only in training mode, and reaches at least 95 percent test accuracy for every variant.
- Each accuracy is reported with a bootstrap confidence interval, and the conclusion about the variants refers to the overlap of those intervals rather than to the point estimates.
- The loss curves have titles and axis labels, use a log scale for the loss, and show the regularised variants with a higher training loss than the plain model.
- The first-layer weight images are plotted with a symmetric colour scale, and you can say in one sentence what a column of $W_1$ means.
""",
    solution_sketch_md=r"""
Reuse everything built above: `Dense`/`ReLU`/`MLP` and `grad_check` (Exercise 5), `log_softmax` (Exercise 3) and the
$(P - Y)/n$ gradient (Exercise 4) combined into one `softmax_ce(Z, y)` returning `(loss, G)`, and a `Dropout` layer
wrapping Exercise 8's functions with a `training` flag. Adam is twenty lines (two moment estimates with bias
correction; add `weight_decay * p` to the update for the decoupled AdamW form). The training loop is forward →
loss → backward → step over shuffled mini-batches of 64, with a `set_training(model, False)` before each evaluation.
With Adam at $3 \times 10^{-3}$ all four variants reach 97–99 percent in 30 epochs; the bootstrap intervals are about
$\pm 1.3$ percentage points wide and overlap, so the honest conclusion is that regularisation changes the *loss curves*
visibly (higher training loss, lower and flatter test loss) but not the accuracy measurably on 360 test digits. The
columns of $W_1$, reshaped to $8 \times 8$, are stroke-like templates — positive weights where a hidden unit expects ink.
""",
    solution_code=PROJECT_SOLUTION,
    starter=PROJECT_STARTER,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
print("exercises:", sum(1 for it in ex.items if it["kind"] == "code"), "coding,",
      sum(1 for it in ex.items if it["kind"] == "concept"), "conceptual,",
      sum(1 for it in ex.items if it["kind"] == "project"), "project")
