"""Builder for Module 06 — Automatic Differentiation and Backpropagation.

Generates lesson.ipynb, exercises.ipynb and solutions.ipynb into the module folder.
Run from the repository root:  python3 _authoring/builders/build_06.py
"""
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import Notebook, ExerciseSet, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-2-calculus-and-optimization/06-autodiff-and-backpropagation"

# =============================================================================
# LESSON
# =============================================================================
nb = Notebook("Module 06 — Automatic Differentiation and Backpropagation")

nb.md(r"""
This notebook builds the machinery behind `loss.backward()` from nothing but Python and NumPy.
You will measure the accuracy of finite differences, draw a computational graph, run forward-mode
differentiation with dual numbers, write a scalar autograd engine, train a neural network with it,
re-derive the same gradients with matrix rules, and learn to *check* any gradient numerically.

Read the [concept guide](README.md) first, or keep it open alongside: the section numbers below
match its numbered concepts.
""")

nb.code(r"""
import math
import time
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
""")

# ---------------------------------------------------------------- Section 1
nb.md(r"""
## 1. Three ways to get a derivative

Symbolic differentiation returns a formula, numerical differentiation nudges the input, and
automatic differentiation applies the chain rule to the program's own arithmetic. The numerical
route is the one with a tuning knob. For a step $h$,

$$
D^{+}_h f(x) = \frac{f(x+h) - f(x)}{h} = f'(x) + O(h), \qquad
D^{c}_h f(x) = \frac{f(x+h) - f(x-h)}{2h} = f'(x) + O(h^2),
$$

but both also carry a round-off error of order $\varepsilon |f(x)| / h$ with $\varepsilon \approx 2.2\times10^{-16}$.
Truncation error falls with $h$, round-off error rises, so there is a best $h$ and an accuracy floor.
Let us measure both for $f(x) = x^3$ at $x = 2$, where $f'(2) = 12$ exactly.
""")

nb.code(r"""
f = lambda x: x**3
x0, exact = 2.0, 12.0
hs = np.logspace(-12, -1, 60)
err_fwd = np.array([abs((f(x0 + h) - f(x0)) / h - exact) for h in hs])
err_ctr = np.array([abs((f(x0 + h) - f(x0 - h)) / (2 * h) - exact) for h in hs])

plt.loglog(hs, err_fwd, label="forward difference")
plt.loglog(hs, err_ctr, label="central difference")
plt.loglog(hs, 2.2e-16 * abs(f(x0)) / hs, "k:", label=r"round-off $\varepsilon|f|/h$")
plt.xlabel("step h"); plt.ylabel("absolute error of f'(2)")
plt.title("Finite differences: truncation vs round-off"); plt.legend(); plt.show()

for h in (1e-1, 1e-5, 1e-8, 1e-12):
    fwd = (f(x0 + h) - f(x0)) / h
    ctr = (f(x0 + h) - f(x0 - h)) / (2 * h)
    print(f"h={h:7.0e}  forward={fwd:.10f}  central={ctr:.10f}")
print("best achievable: forward ~1e-8 around h=1e-8, central ~1e-10 around h=1e-5")
print("automatic differentiation of x**3 at 2 gives 3*x**2 =", 3 * x0**2, "with no h at all")
""")

nb.md(r"""
Notice the V shape. Left of the minimum the two evaluations of $f$ agree to almost all their
digits and the subtraction destroys the rest; right of it the Taylor remainder dominates. Central
differences buy two orders of magnitude at the price of one extra evaluation per input — and a
*full gradient* costs $2n$ evaluations for $n$ inputs. Keep this plot in mind: it is why finite
differences are a *test*, not a training method, and why gradient checks (Section 7) use
$h \approx 10^{-5}$ in float64.

**Try it yourself:** repeat the experiment with `np.float32(x0)` and `np.float32(h)`. Where does
the floor move to, and what does that imply for gradient checks in single precision?
""")

# ---------------------------------------------------------------- Section 2
nb.md(r"""
## 2. Computational graphs

An expression becomes a directed acyclic graph: inputs are leaves, every primitive operation is a
node, and an edge means "this value is used by that operation". Evaluating the nodes in a
*topological order* (every node after its parents) is the forward pass. Here is the squared error
of a one-feature linear model, $L = (wx + b - y)^2$, with $w=2,\ x=3,\ b=1,\ y=5$.
""")

nb.code(r"""
w, x, b, y = 2.0, 3.0, 1.0, 5.0
v1 = w * x          # 6
v2 = v1 + b         # 7
v3 = v2 - y         # 2
L = v3 ** 2         # 4
vals = {"w": w, "x": x, "b": b, "y": y, "v1 = w*x": v1, "v2 = v1+b": v2, "v3 = v2-y": v3, "L = v3**2": L}
pos = {"w": (0, 3), "x": (0, 2), "b": (0, 1), "y": (0, 0),
       "v1 = w*x": (2, 2.5), "v2 = v1+b": (4, 1.75), "v3 = v2-y": (6, 0.9), "L = v3**2": (8, 0.9)}
edges = [("w", "v1 = w*x"), ("x", "v1 = w*x"), ("v1 = w*x", "v2 = v1+b"), ("b", "v2 = v1+b"),
         ("v2 = v1+b", "v3 = v2-y"), ("y", "v3 = v2-y"), ("v3 = v2-y", "L = v3**2")]

fig, ax = plt.subplots(figsize=(8.5, 3.4))
for a, c in edges:
    ax.annotate("", xy=pos[c], xytext=pos[a],
                arrowprops=dict(arrowstyle="->", lw=1.3, shrinkA=22, shrinkB=22, color="gray"))
for name, (px, py) in pos.items():
    leaf = len(name) == 1
    ax.text(px, py, f"{name}\n= {vals[name]:g}", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#dbe9ff" if leaf else "#ffe8c2", ec="gray"))
ax.set_xlim(-0.8, 8.9); ax.set_ylim(-0.6, 3.6); ax.axis("off")
ax.set_title("Computational graph of L = (w*x + b - y)**2 with forward values")
plt.show()

print("topological order:", [k for k in vals])
print("forward values   :", [vals[k] for k in vals])
""")

nb.md(r"""
Blue leaves are inputs, orange nodes are operations. The graph does not depend on the numbers:
feed it a different data point and the *structure* is identical, which is why a framework can run
one generic backward routine for every example. A spreadsheet's dependency graph, the dataflow
of a cost model, and the `grad_fn` chain PyTorch attaches to every tensor are all this picture.
""")

# ---------------------------------------------------------------- Section 3
nb.md(r"""
## 3. The chain rule on graphs: forward mode and reverse mode

Each edge carries a *local* derivative. **Forward mode** pushes tangents $\dot v_i = \partial v_i / \partial x_j$
from one chosen input towards the output; **reverse mode** pulls adjoints $\bar v_k = \partial L / \partial v_k$
from the output back to every input:

$$
\dot v_i = \sum_{k \in \mathrm{pa}(i)} \frac{\partial \phi_i}{\partial v_k}\,\dot v_k, \qquad
\bar v_k = \sum_{i \in \mathrm{ch}(k)} \bar v_i\,\frac{\partial \phi_i}{\partial v_k}.
$$

Forward mode has a beautiful implementation: **dual numbers**. Carry the pair $(v, \dot v)$ through
every operation with the product rule built in, $(a,\dot a)(b,\dot b) = (ab,\ a\dot b + \dot a b)$, and the
derivative pops out of the computation with no graph stored at all.
""")

nb.code(r"""
class Dual:
    # a value together with its tangent (derivative w.r.t. one chosen input)
    def __init__(self, val, dot=0.0):
        self.val, self.dot = float(val), float(dot)
    def _lift(o):
        return o if isinstance(o, Dual) else Dual(o)
    def __add__(self, o):
        o = Dual._lift(o); return Dual(self.val + o.val, self.dot + o.dot)
    def __sub__(self, o):
        o = Dual._lift(o); return Dual(self.val - o.val, self.dot - o.dot)
    def __mul__(self, o):
        o = Dual._lift(o); return Dual(self.val * o.val, self.val * o.dot + self.dot * o.val)
    def __pow__(self, k):
        return Dual(self.val ** k, k * self.val ** (k - 1) * self.dot)
    def exp(self):
        e = math.exp(self.val); return Dual(e, e * self.dot)
    def sin(self):
        return Dual(math.sin(self.val), math.cos(self.val) * self.dot)
    __radd__, __rmul__ = __add__, __mul__
    def __repr__(self):
        return f"Dual(val={self.val:.6g}, dot={self.dot:.6g})"

# f(x1, x2) = (x1*x2 + x1)**2 at (2, 3): closed form df/dx1 = 2*x1*(x2+1)**2 = 64, df/dx2 = 2*x1**2*(x2+1) = 32
def f(x1, x2):
    return (x1 * x2 + x1) ** 2

sweep1 = f(Dual(2.0, 1.0), Dual(3.0, 0.0))   # seed x1
sweep2 = f(Dual(2.0, 0.0), Dual(3.0, 1.0))   # seed x2
print("sweep seeded at x1:", sweep1, " -> df/dx1 =", sweep1.dot)
print("sweep seeded at x2:", sweep2, " -> df/dx2 =", sweep2.dot)

g = lambda x: x * x.exp() + x.sin()           # g'(x) = e^x (1 + x) + cos x
d = g(Dual(1.0, 1.0))
print(f"g'(1) by dual numbers = {d.dot:.12f}, closed form = {math.e * 2 + math.cos(1.0):.12f}")
""")

nb.md(r"""
Two sweeps were needed for two inputs: forward mode costs one pass **per input**. Reverse mode
runs the arrows backwards, starting from $\bar L = 1$, and costs one pass **per output**. By hand on
the same graph ($v_1 = x_1 x_2 = 6$, $v_2 = v_1 + x_1 = 8$, $v_3 = v_2^2 = 64$):

$\bar v_3 = 1$, $\ \bar v_2 = 2 v_2 = 16$, $\ \bar v_1 = 16$, $\ \bar x_1 = \bar v_1 x_2 + \bar v_2 \cdot 1 = 48 + 16 = 64$, $\ \bar x_2 = \bar v_1 x_1 = 32$.

One sweep, both derivatives — and $x_1$, which is used twice, correctly collects two contributions.
For a loss with millions of parameters and one output this is the whole game: the full gradient
costs a small constant times one forward pass, not millions of them. We will *measure* this once
we have a reverse-mode engine (Section 5).

**Try it yourself:** add `__truediv__` to `Dual` using the quotient rule and check $\frac{d}{dx}\,\frac{\sin x}{x}$ at $x = 2$ against the closed form $\frac{x\cos x - \sin x}{x^2}$.
""")

# ---------------------------------------------------------------- Section 4
nb.md(r"""
## 4. Backpropagation: reverse mode on a neural network

For layers $z_l = W_l a_{l-1} + b_l$, $a_l = f(z_l)$, reverse mode has a regular structure. With
$\delta_l := \partial\mathcal{L}/\partial z_l$:

$$
\delta_L = \nabla_{a_L}\mathcal{L} \odot f'(z_L), \qquad
\delta_l = (W_{l+1}^\top \delta_{l+1}) \odot f'(z_l), \qquad
\frac{\partial \mathcal{L}}{\partial W_l} = \delta_l a_{l-1}^\top, \qquad
\frac{\partial \mathcal{L}}{\partial b_l} = \delta_l .
$$

The concept guide works a 2–2–1 ReLU network by hand (input $x=(1,2)$, target $y=3$, squared loss).
Let us reproduce every number.
""")

nb.code(r"""
relu = lambda z: np.maximum(z, 0.0)
drelu = lambda z: (z > 0).astype(float)

W1 = np.array([[1.0, -1.0], [0.0, 1.0]]); b1 = np.zeros(2)
W2 = np.array([[1.0, 0.5]]);              b2 = np.zeros(1)
x = np.array([1.0, 2.0]); y_target = 3.0

# forward
z1 = W1 @ x + b1; a1 = relu(z1)
z2 = W2 @ a1 + b2; y_hat = z2[0]          # linear output
loss = 0.5 * (y_hat - y_target) ** 2
print(f"z1={z1}, a1={a1}, y_hat={y_hat}, loss={loss}")

# backward
delta2 = np.array([y_hat - y_target])          # dL/dz2 for a linear output with squared loss
dW2 = np.outer(delta2, a1); db2 = delta2
delta1 = (W2.T @ delta2) * drelu(z1)
dW1 = np.outer(delta1, x);  db1 = delta1
print("delta2 =", delta2, " dW2 =", dW2, " db2 =", db2)
print("delta1 =", delta1)
print("dW1 =\n", dW1, "\ndb1 =", db1)
""")

nb.md(r"""
The first hidden unit had $z_1 = -1 < 0$, so its ReLU derivative is 0 and *every* gradient that
flows through it is 0: its weights and bias will not move. That is the "dying ReLU" problem in
two lines of arithmetic. Also note what had to be kept from the forward pass: $a_1$ (for $\partial\mathcal{L}/\partial W_2$)
and $z_1$ (for the mask). In a deep network this stored-activation memory, not the weights,
dominates training memory. (A leaky ReLU, with derivative `np.where(z > 0, 1.0, 0.01)`, would let the dead
unit receive a small gradient and recover.)
""")

# ---------------------------------------------------------------- Section 5
nb.md(r"""
## 5. A scalar autograd engine from scratch

Now the real thing. A `Value` remembers its `data`, its `grad`, the `Value`s it was made from, and a
tiny closure `_backward` that passes its gradient to its parents using the local derivative rule.
Operator overloading builds the graph silently as you write ordinary arithmetic; `backward()`
sorts the graph topologically, seeds the output with gradient 1 and calls every closure in
reverse order. This is the design of Karpathy's `micrograd` and, in spirit, of PyTorch's autograd.

| op | local rule in the closure (`g` is the gradient arriving at `out`) |
|---|---|
| `a + b` | `a.grad += g; b.grad += g` |
| `a * b` | `a.grad += b.data*g; b.grad += a.data*g` |
| `a ** k` | `a.grad += k*a.data**(k-1)*g` |
| `exp`, `log` | `a.grad += out.data*g`, `a.grad += g/a.data` |
| `tanh`, `relu` | `a.grad += (1-out.data**2)*g`, `a.grad += (a.data>0)*g` |

Gradients **accumulate** (`+=`): a node used twice must receive both contributions. The flip side
is that you must zero all `grad`s before every new `backward()`.
""")

nb.code(r"""
class Value:
    # a scalar that remembers how it was computed, and how to pass a gradient back
    def __init__(self, data, _parents=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = _parents
        self._op = _op
        self._backward = lambda g: None     # receives dL/d(out) and adds to the parents' grads

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward(g):
            self.grad += g
            other.grad += g
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward(g):
            self.grad += other.data * g
            other.grad += self.data * g
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward(g):
            self.grad += k * self.data ** (k - 1) * g
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _backward(g):
            self.grad += e * g                       # d/da e^a = e^a = the output
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _backward(g):
            self.grad += (1.0 / self.data) * g       # needs the *input*
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward(g):
            self.grad += (1.0 - t ** 2) * g          # needs the *output*
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")
        def _backward(g):
            self.grad += (1.0 if self.data > 0 else 0.0) * g
        out._backward = _backward
        return out

    # convenience operators, all expressed through the primitives above
    def __neg__(self):            return self * -1.0
    def __radd__(self, other):    return self + other
    def __sub__(self, other):     return self + (-other)
    def __rsub__(self, other):    return other + (-self)
    def __rmul__(self, other):    return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other):return other * self ** -1

    def topo_order(self):
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)          # a node is appended only after all its parents
        visit(self)
        return order

    def backward(self):
        order = self.topo_order()
        self.grad = 1.0
        for v in reversed(order):
            v._backward(v.grad)          # hand each node's complete gradient to its parents

print("Value engine ready:", Value(2.0) * 3 + 1)
""")

nb.code(r"""
# Section 2's graph, now differentiated automatically
w, x, b, y = Value(2.0), Value(3.0), Value(1.0), Value(5.0)
L = (w * x + b - y) ** 2
L.backward()
print(f"L = {L.data},  dL/dw = {w.grad},  dL/dx = {x.grad},  dL/db = {b.grad},  dL/dy = {y.grad}")
print("closed form: dL/dw = 2*(wx+b-y)*x =", 2 * (2 * 3 + 1 - 5) * 3, ", dL/db = 2*(wx+b-y) =", 2 * (2 * 3 + 1 - 5))

print("\ntopological order (op: data):")
for v in L.topo_order():
    print(f"   {v._op or 'leaf':6s} data={v.data:6.1f}  grad={v.grad:6.1f}")
""")

nb.md(r"""
The order lists leaves first and the loss last; `backward()` walks it in reverse so that every
node's gradient is complete before it is passed on. The `-1.0` leaf and the `*` node with value $-5$
are what `__sub__` creates (`a - b` is implemented as `a + b * (-1)`) — a reminder that the engine
differentiates the program *as written*, constants and all.

Now the step you should never skip: compare with central differences. We write a helper that
evaluates an expression on plain floats, perturbs one input at a time, and reports the relative error.
""")

nb.code(r"""
def expression(a, b, c):
    # works on floats and on Values alike (that is the point of operator overloading)
    return (a * b + c.exp()).tanh() * (a ** 2 / (b + 3.0)) + (b * c).log()

def numerical_grad(fn, args, h=1e-5):
    grads = []
    for i in range(len(args)):
        plus = list(args); plus[i] += h
        minus = list(args); minus[i] -= h
        grads.append((fn(*plus) - fn(*minus)) / (2 * h))
    return np.array(grads)

def relative_error(g, g_num):
    return np.linalg.norm(g - g_num) / (np.linalg.norm(g) + np.linalg.norm(g_num) + 1e-300)

class F:   # float wrapper so that .exp()/.tanh()/.log() exist on plain numbers too
    def __init__(self, v): self.v = float(v)
    def exp(self): return F(math.exp(self.v))
    def log(self): return F(math.log(self.v))
    def tanh(self): return F(math.tanh(self.v))
    def _l(o): return o.v if isinstance(o, F) else float(o)
    def __add__(s, o): return F(s.v + F._l(o))
    def __mul__(s, o): return F(s.v * F._l(o))
    def __pow__(s, k): return F(s.v ** k)
    def __truediv__(s, o): return F(s.v / F._l(o))
    __radd__, __rmul__ = __add__, __mul__

a, b, c = Value(0.7), Value(1.3), Value(0.4)
out = expression(a, b, c); out.backward()
g_auto = np.array([a.grad, b.grad, c.grad])
g_num = numerical_grad(lambda *t: expression(*[F(v) for v in t]).v, [0.7, 1.3, 0.4])
print("autograd   :", g_auto)
print("numerical  :", g_num)
print(f"relative error = {relative_error(g_auto, g_num):.2e}   (< 1e-7 means the engine is right)")
""")

nb.md(r"""
A relative error around $10^{-10}$ is the signature of a correct engine; a wrong closure would
show up as $10^{-2}$ or worse. We will systematise this in Section 7.

Now the promised measurement of **cost**. Take a function of $n$ inputs. Forward mode (our `Dual`)
needs $n$ sweeps, each costing $O(n)$, so $O(n^2)$ total; reverse mode (our `Value`) needs one
forward pass plus one backward sweep, $O(n)$ total.
""")

nb.code(r"""
def chain(xs):
    # a function of n inputs whose graph has about 2n nodes
    acc = xs[0]
    for xi in xs[1:]:
        acc = acc * xi + xi
    return acc ** 2

ns = [25, 50, 100, 200, 400]
t_fwd, t_rev = [], []
for n in ns:
    xs = rng.uniform(0.9, 1.1, size=n)
    t0 = time.perf_counter()
    grad_fwd = np.array([chain([Dual(v, 1.0 if j == i else 0.0) for j, v in enumerate(xs)]).dot
                         for i in range(n)])                      # n sweeps
    t_fwd.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    vs = [Value(v) for v in xs]; out = chain(vs); out.backward()   # one sweep
    grad_rev = np.array([v.grad for v in vs])
    t_rev.append(time.perf_counter() - t0)
    assert relative_error(grad_fwd, grad_rev) < 1e-10

plt.loglog(ns, t_fwd, "o-", label="forward mode (n Dual sweeps)")
plt.loglog(ns, t_rev, "s-", label="reverse mode (1 Value backward)")
plt.xlabel("number of inputs n"); plt.ylabel("seconds for the full gradient")
plt.title("Cost of a full gradient: forward vs reverse mode"); plt.legend(); plt.show()
print("ratio forward/reverse at n=400: %.0fx" % (t_fwd[-1] / t_rev[-1]))
""")

nb.md(r"""
The two curves have different slopes on the log-log plot: forward mode's time grows roughly like
$n^2$, reverse mode's like $n$, and the gap widens without bound. For a 124-million-parameter
model the ratio is not 100 but $10^8$. This is the "cheap gradient principle", and it is the
reason gradient-based deep learning is possible at all.

### Training a neural network with the engine

A neuron is a dot product plus a bias followed by a nonlinearity, all made of `Value`s; a layer is a
list of neurons; an MLP is a list of layers. We train on a synthetic two-moons dataset. For the loss we
add one *fused* operation, binary cross-entropy on the logit $s$, computed in the stable
log-sum-exp form $\max(s,0) + \log(1 + e^{-|s|}) - ys$ with the compact gradient $\sigma(s) - y$
(Sections 6 and 8 explain why frameworks fuse exactly this kind of op).
""")

nb.code(r"""
from sklearn.datasets import make_moons

def bce_with_logits(s, y):
    # fused, numerically stable binary cross-entropy of logit s against label y in {0, 1}
    sd = s.data
    out = Value(max(sd, 0.0) + math.log1p(math.exp(-abs(sd))) - y * sd, (s,), "bce")
    p = 1.0 / (1.0 + math.exp(-sd))
    def _backward(g):
        s.grad += (p - y) * g                   # the famous p - y
    out._backward = _backward
    return out

class Neuron:
    def __init__(self, n_in, nonlin=True):
        scale = math.sqrt(2.0 / n_in)                              # He initialisation
        self.w = [Value(rng.normal(0.0, scale)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin
    def __call__(self, xs):
        act = self.b
        for wi, xi in zip(self.w, xs):
            act = act + wi * xi
        return act.relu() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out, nonlin=True):
        self.neurons = [Neuron(n_in, nonlin) for _ in range(n_out)]
    def __call__(self, xs):
        return [n(xs) for n in self.neurons]
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i + 1], nonlin=(i < len(sizes) - 2)) for i in range(len(sizes) - 1)]
    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs[0] if len(xs) == 1 else xs
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

X, y = make_moons(n_samples=200, noise=0.15, random_state=0)
model = MLP([2, 16, 16, 1])
params = model.parameters()
print("parameters:", len(params), " (2*16+16) + (16*16+16) + (16+1) =", 2 * 16 + 16 + 16 * 16 + 16 + 16 + 1)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=18, edgecolor="k", linewidth=0.3)
plt.xlabel("x1"); plt.ylabel("x2"); plt.title("Two moons: 200 points, two classes"); plt.show()
""")

nb.code(r"""
def batch_loss(model, idx):
    total = Value(0.0)
    for i in idx:
        total = total + bce_with_logits(model([X[i, 0], X[i, 1]]), float(y[i]))
    return total * (1.0 / len(idx))

def accuracy(model):
    scores = np.array([model([xi[0], xi[1]]).data for xi in X])
    return np.mean((scores > 0) == (y == 1))

lr, batch_size, steps = 0.2, 16, 300
loss_hist = []
t0 = time.time()
for step in range(steps):
    idx = rng.choice(len(X), size=batch_size, replace=False)
    loss = batch_loss(model, idx)
    for p in params:
        p.grad = 0.0                 # zero the accumulated gradients!
    loss.backward()
    for p in params:
        p.data -= lr * p.grad        # plain SGD
    loss_hist.append(loss.data)
    if step % 50 == 0 or step == steps - 1:
        print(f"step {step:3d}  minibatch loss {loss.data:.3f}  train accuracy {accuracy(model):.3f}")
print(f"training took {time.time() - t0:.1f}s with {len(params)} parameters of pure Python")

plt.plot(loss_hist, lw=0.8, alpha=0.5, label="minibatch loss")
plt.plot(np.convolve(loss_hist, np.ones(20) / 20, mode="valid"), lw=2, label="20-step moving average")
plt.xlabel("SGD step"); plt.ylabel("binary cross-entropy"); plt.title("Training loss of the Value-MLP")
plt.legend(); plt.show()
""")

nb.code(r"""
gx, gy = np.meshgrid(np.linspace(-1.8, 2.8, 60), np.linspace(-1.3, 1.8, 50))
scores = np.array([model([a, b_]).data for a, b_ in zip(gx.ravel(), gy.ravel())]).reshape(gx.shape)
plt.contourf(gx, gy, 1 / (1 + np.exp(-scores)), levels=20, cmap="coolwarm", alpha=0.6)
plt.colorbar(label="P(class 1)")
plt.contour(gx, gy, scores, levels=[0.0], colors="k", linewidths=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=18, edgecolor="k", linewidth=0.3)
plt.xlabel("x1"); plt.ylabel("x2"); plt.title(f"Decision boundary, train accuracy {accuracy(model):.3f}"); plt.show()
""")

nb.md(r"""
A few hundred SGD steps carve a curved boundary between the moons. Every gradient came from the
closures you read above — nothing else. The whole loop is the template of every training script:
forward, zero the gradients, backward, update.

**Try it yourself:** swap `relu` for `tanh` in `Neuron.__call__` (keep the output linear) or change
`batch_size` to 64 — how do the loss curve and the runtime change?
""")

# ---------------------------------------------------------------- Section 6
nb.md(r"""
## 6. Vector and matrix autograd: vector-Jacobian products

A scalar engine spends most of its time in Python bookkeeping (the training loop above ran about
a thousand times slower than NumPy would). Frameworks make each *array* operation a single node
with a backward rule that maps the upstream gradient $G = \partial L / \partial Y$ to gradients of the
inputs, without ever forming a Jacobian. For the operations in our network:

| op | forward | backward |
|---|---|---|
| matmul | $Y = AB$ | $\bar A = G B^\top,\quad \bar B = A^\top G$ |
| bias broadcast | $Y = Z + \mathbf 1 b^\top$ | $\bar Z = G,\quad \bar b = G^\top \mathbf 1$ (sum over rows) |
| elementwise | $Y = f(X)$ | $\bar X = G \odot f'(X)$ |
| mean of losses | $L = \tfrac1B\sum_i \ell_i$ | $\bar\ell_i = 1/B$ |

Let us re-implement the *same* trained network with NumPy arrays, run one forward/backward pass
on a minibatch with these rules, and compare with the scalar engine's gradients, parameter by parameter.
""")

nb.code(r"""
# copy the trained parameters into matrices: W_l has shape (n_out, n_in)
def layer_matrices(layer):
    W = np.array([[wi.data for wi in n.w] for n in layer.neurons])
    b = np.array([n.b.data for n in layer.neurons])
    return W, b

(W1, b1), (W2, b2), (W3, b3) = [layer_matrices(l) for l in model.layers]
print("shapes:", W1.shape, b1.shape, W2.shape, b2.shape, W3.shape, b3.shape)

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def forward_backward_numpy(Xb, yb):
    B = len(Xb)
    # forward (rows are examples, so the layer map is Z = A @ W.T + b)
    Z1 = Xb @ W1.T + b1; A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ W2.T + b2; A2 = np.maximum(Z2, 0.0)
    S = (A2 @ W3.T + b3)[:, 0]                                    # logits
    loss = np.mean(np.maximum(S, 0) + np.log1p(np.exp(-np.abs(S))) - yb * S)
    # backward: upstream gradient of the mean is 1/B per example
    dS = (sigmoid(S) - yb) / B                                    # fused BCE rule, shape (B,)
    G3 = dS[:, None]                                              # dL/d(A2 @ W3.T + b3), shape (B, 1)
    dW3 = G3.T @ A2;            db3 = G3.sum(axis=0)              # matmul rule + broadcast rule
    G2 = (G3 @ W3) * (Z2 > 0)                                     # back through matmul, then ReLU mask
    dW2 = G2.T @ A1;            db2 = G2.sum(axis=0)
    G1 = (G2 @ W2) * (Z1 > 0)
    dW1 = G1.T @ Xb;            db1 = G1.sum(axis=0)
    return loss, [dW1, db1, dW2, db2, dW3, db3]

idx = rng.choice(len(X), size=32, replace=False)
loss_np, grads_np = forward_backward_numpy(X[idx], y[idx].astype(float))

# the same minibatch through the scalar engine
loss_v = batch_loss(model, idx)
for p in params:
    p.grad = 0.0
loss_v.backward()
grads_v = []
for layer in model.layers:
    grads_v.append(np.array([[wi.grad for wi in n.w] for n in layer.neurons]))
    grads_v.append(np.array([n.b.grad for n in layer.neurons]))

print(f"loss: numpy {loss_np:.10f}  scalar engine {loss_v.data:.10f}")
for name, gn, gv in zip(["dW1", "db1", "dW2", "db2", "dW3", "db3"], grads_np, grads_v):
    print(f"{name}: shape {gn.shape}, max |numpy - scalar| = {np.max(np.abs(gn - gv)):.2e}")
""")

nb.md(r"""
Identical to about $10^{-16}$: the matrix rules are the scalar closures, summed over a batch and
written as matrix products. Watch the shape logic in `forward_backward_numpy`: $\bar W$ must have
$W$'s shape, so `G.T @ A` is the only arrangement that can be right; and every bias — broadcast
across the batch in the forward pass — becomes a `sum(axis=0)` in the backward pass.

### The fused softmax-cross-entropy gradient

For $K$ classes, $p = \mathrm{softmax}(z)$ and one-hot $y$, the loss $-\sum_k y_k \ln p_k$ has the gradient
$$\frac{\partial \mathcal L}{\partial z} = p - y .$$
Frameworks fuse log-softmax and the negative log-likelihood into one op for stability (no $\ln 0$),
speed (one subtraction instead of a Jacobian product) and memory (one pass over the logits, which in a
language model are the largest tensor around). Let us verify the formula against the slow route.
""")

nb.code(r"""
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def softmax_jacobian(z):
    p = softmax(z)
    return np.diag(p) - np.outer(p, p)         # dp_i/dz_j = p_i (delta_ij - p_j)

z = np.array([2.0, 0.0, 0.0]); target = 1
y_onehot = np.eye(3)[target]
p = softmax(z)
fused = p - y_onehot
chained = softmax_jacobian(z).T @ (-y_onehot / p)   # dL/dp = -y/p, then through the softmax Jacobian
num = numerical_grad(lambda *zz: -np.log(softmax(np.array(zz))[target]), list(z))
print("p          =", p)
print("fused p - y=", fused)
print("via Jacobian", chained)
print("numerical   ", num)
print(f"relative errors: fused vs numerical {relative_error(fused, num):.1e}, Jacobian vs fused {relative_error(chained, fused):.1e}")
""")

nb.md(r"""
Three routes, one answer: raise the logit of the correct class, lower the others, each by exactly
how wrong its probability is. The fused form needed no Jacobian and no division by $p$ — the two
things that go wrong numerically when a prediction is confidently wrong.

**Try it yourself:** set `z = np.array([30.0, 0.0, 0.0])` with `target = 1`. The fused gradient is fine;
what happens to `-y_onehot / p` on the chained route?
""")

# ---------------------------------------------------------------- Section 7
nb.md(r"""
## 7. Gradient checking

A wrong backward rule does not crash: it trains a slightly wrong model, slowly. The defence is a
*gradient checker*: central differences on a few parameters, compared through a relative error

$$
r = \frac{\|g - \hat g\|_2}{\|g\|_2 + \|\hat g\|_2}, \qquad
r < 10^{-7} \text{ correct},\quad 10^{-7}\!-\!10^{-4} \text{ kinks},\quad > 10^{-2} \text{ bug}.
$$

We check the scalar engine on the trained network, then plant two bugs and watch the checker catch them.
""")

nb.code(r"""
def grad_check(model, idx, n_params=20, h=1e-5, seed=0):
    # analytic gradient of the minibatch loss
    params = model.parameters()
    for p in params:
        p.grad = 0.0
    batch_loss(model, idx).backward()
    chosen = np.random.default_rng(seed).choice(len(params), size=n_params, replace=False)
    g, g_num = [], []
    for k in chosen:
        p = params[k]
        g.append(p.grad)
        old = p.data
        p.data = old + h; lp = batch_loss(model, idx).data
        p.data = old - h; lm = batch_loss(model, idx).data
        p.data = old
        g_num.append((lp - lm) / (2 * h))
    g, g_num = np.array(g), np.array(g_num)
    return relative_error(g, g_num), g, g_num

def smallest_relu_input(model, idx):
    # how close does any hidden pre-activation in this minibatch come to ReLU's kink at 0?
    closest = np.inf
    for i in idx:
        acts = [X[i, 0], X[i, 1]]
        for layer in model.layers[:-1]:
            pre = np.array([n.b.data + sum(wi.data * a for wi, a in zip(n.w, acts)) for n in layer.neurons])
            closest = min(closest, np.min(np.abs(pre)))
            acts = list(np.maximum(pre, 0.0))
    return closest

for trial in range(5):
    idx = rng.choice(len(X), size=8, replace=False)
    r, g, g_num = grad_check(model, idx)
    print(f"minibatch {trial}: relative error {r:.1e}, closest hidden pre-activation to 0: {smallest_relu_input(model, idx):.1e}")
""")

nb.md(r"""
Relative errors around $10^{-10}$ say the engine is right. If one minibatch shows a much larger error, look at
the last column: a hidden pre-activation within about $h = 10^{-5}$ of zero means the central difference
straddled a ReLU kink, and the "error" is the kink, not the code (Section 8 shows this in isolation). That is
why a gradient check is run on several minibatches, and why a single outlier is interpreted before it is believed.

Now let us plant two bugs and watch the checker catch them.
""")

nb.code(r"""
# bug 1: the tanh closure uses the *input* instead of the output   (1 - a^2 instead of 1 - tanh(a)^2)
def buggy_tanh(self):
    out = Value(math.tanh(self.data), (self,), "tanh")
    def _backward(g):
        self.grad += (1.0 - self.data ** 2) * g
    out._backward = _backward
    return out

# bug 2: the multiplication closure forgets to swap the operands
def buggy_mul(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")
    def _backward(g):
        self.grad += self.data * g
        other.grad += other.data * g
    out._backward = _backward
    return out

def check_expression(fn, point, label):
    vs = [Value(v) for v in point]
    out = fn(*vs); out.backward()
    g = np.array([v.grad for v in vs])
    g_num = numerical_grad(lambda *t: fn(*[F(v) for v in t]).v, list(point))
    print(f"{label:38s} relative error = {relative_error(g, g_num):.1e}")

expr = lambda a, b: (a * b).tanh() + a * a
good_tanh, good_mul = Value.tanh, Value.__mul__
check_expression(expr, [0.3, 0.2], "correct engine")
Value.tanh = buggy_tanh
check_expression(expr, [0.3, 0.2], "buggy tanh, small inputs")
check_expression(expr, [1.5, 1.2], "buggy tanh, larger inputs")
Value.tanh = good_tanh
Value.__mul__ = buggy_mul
check_expression(expr, [0.3, 0.2], "buggy mul")
Value.__mul__ = good_mul
check_expression(expr, [0.3, 0.2], "repaired engine")
""")

nb.md(r"""
The multiplication bug is loud at any point. The `tanh` bug is nearly invisible near zero (where
$1 - a^2 \approx 1 - \tanh^2 a$) and glaring for larger inputs — which is why you check at several
points and parameter scales, not just at a small random initialisation.

Three practical rules: check in **float64** (in float32 the floor is about $10^{-3}$, indistinguishable
from a mild bug); freeze **randomness** (dropout masks, data order) so both loss evaluations see the
same function; and expect **kinks** (ReLU, max) to produce isolated large errors when a pre-activation
sits within $h$ of zero — Section 8 shows one.
""")

# ---------------------------------------------------------------- Section 8
nb.md(r"""
## 8. Where autodiff goes wrong

Autodiff differentiates the program you wrote, not the function you meant. Three demonstrations:
a false alarm at a ReLU kink, overflow in a naive log-sum-exp, and gradients vanishing through a
long chain of sigmoids.

**Log-sum-exp.** $\ln\sum_i e^{z_i}$ overflows once any $z_i > 709$. Subtracting $m = \max_i z_i$ fixes it
exactly: $\ln\sum_i e^{z_i} = m + \ln\sum_i e^{z_i - m}$, and every exponent is now $\le 0$.
""")

nb.code(r"""
# (a) kink: central differences straddling ReLU's corner
z0, h = 1e-6, 1e-5
num = (max(z0 + h, 0) - max(z0 - h, 0)) / (2 * h)
print(f"ReLU at z={z0:g}: analytic derivative 1, central difference {num:.3f}  -> a false alarm, not a bug")

# (b) overflow and the log-sum-exp trick
z = np.array([1000.0, 1000.0, 999.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.log(np.sum(np.exp(z)))
    naive_softmax = np.exp(z) / np.sum(np.exp(z))
m = z.max()
stable = m + np.log(np.sum(np.exp(z - m)))
stable_softmax = np.exp(z - m) / np.sum(np.exp(z - m))
print(f"naive  log-sum-exp = {naive},  softmax = {naive_softmax}")
print(f"stable log-sum-exp = {stable:.6f},  softmax = {stable_softmax}")
print(f"log-softmax of the first entry = {z[0] - stable:.6f}  (computed without ever forming p)")

# (c) vanishing gradients through a chain of sigmoid units
def sigmoid_chain_grad(depth, weight):
    a = 0.5
    grad = 1.0
    for _ in range(depth):
        zz = weight * a
        s = 1 / (1 + np.exp(-zz))
        grad *= s * (1 - s) * weight     # local derivative of sigmoid(w*a) w.r.t. a
        a = s
    return grad

depths = np.arange(1, 31)
for wgt in (1.0, 4.0, 8.0):
    plt.semilogy(depths, [abs(sigmoid_chain_grad(d, wgt)) for d in depths], "o-", ms=3, label=f"weight {wgt:g}")
plt.semilogy(depths, 0.25 ** depths, "k--", label=r"bound $4^{-L}$ for weight 1")
plt.xlabel("depth L"); plt.ylabel("|d output / d input|"); plt.title("Gradient through a chain of sigmoid units")
plt.legend(); plt.show()
print("sigma' <= 1/4, so with unit weights the gradient shrinks at least like 4^-L; large weights explode instead")
""")

nb.md(r"""
Three different failure modes, three different remedies: interpret kinks correctly when checking
(or check away from them), always compute logarithms of sums of exponentials in the shifted form
(this is what `log_softmax` and `BCEWithLogitsLoss` do, and what our fused `bce_with_logits` did), and
control the scale of the per-layer Jacobians through initialisation, normalisation and residual
connections — the topic of Module 16.

A related operation that autodiff *cannot* help with is anything piecewise constant: rounding,
`argmax`, sampling a token. Their derivative is zero almost everywhere, so upstream parameters receive
nothing; the *straight-through estimator* substitutes the identity in the backward pass. And because
the backward pass is itself made of differentiable operations, differentiating it again gives
Hessian-vector products at the cost of about two backward passes — no $n \times n$ Hessian needed.

**Try it yourself:** in part (c) replace the sigmoid by `tanh` (derivative $1 - t^2 \le 1$) and see how
the decay changes; then by ReLU with weight 1, where the chain is exactly 1 for positive inputs.
""")

# ---------------------------------------------------------------- Summary
nb.md(r"""
## Summary

* Finite differences have an accuracy floor set by round-off (about $10^{-8}$ forward, $10^{-10}$ central
  in float64) and cost one or two extra evaluations **per input**; they are for testing, not training.
* A computational graph records every primitive operation; the forward pass evaluates it in topological order.
* Forward mode (dual numbers) pushes tangents, one sweep per input; reverse mode pulls adjoints, one sweep
  per output. For a scalar loss, reverse mode's cost is independent of the number of parameters — we measured it.
* Backpropagation is reverse mode on a layered network: $\delta_l = (W_{l+1}^\top\delta_{l+1})\odot f'(z_l)$,
  $\partial\mathcal L/\partial W_l = \delta_l a_{l-1}^\top$; it needs the stored activations.
* A `Value` class with `data`, `grad`, parents and a `_backward` closure, plus a topological sort, is a complete
  autograd engine. Gradients accumulate with `+=`, so zero them before every backward pass.
* Array autograd uses vector-Jacobian products: $\bar A = G B^\top$, $\bar B = A^\top G$, broadcast ⇒ sum,
  elementwise ⇒ mask; the fused softmax-cross-entropy gradient is $p - y$. Our NumPy rules matched the scalar
  engine to $10^{-16}$.
* Gradient checking with central differences and a relative error catches wrong rules in seconds — check in
  float64, at several points, away from kinks.
* Kinks, overflow (use log-sum-exp) and long chains (vanishing/exploding gradients) are where "the derivative
  of the program" stops being "the derivative you wanted".

**Next:** open `exercises.ipynb` and build the pieces yourself — a `Dual` class, the missing closures of a
`Value` engine, the matmul backward rule, a gradient checker, and finally your own trained network.
""")

nb.save(MOD / "lesson.ipynb")
print("lesson cells:", len(nb.cells))

# =============================================================================
# EXERCISES
# =============================================================================
# Reference solutions are defined here as plain strings, exec'd at build time to
# compute the hidden hashes, and reused verbatim as the solution cells.

SETUP = r"""
import math
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)
"""

# A complete reference Value engine, shared by several exercises (given to the learner).
VALUE_FULL = r"""
class Value:
    def __init__(self, data, _parents=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = _parents
        self._op = _op
        self._backward = lambda g: None

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward(g):
            self.grad += g
            other.grad += g
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward(g):
            self.grad += other.data * g
            other.grad += self.data * g
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward(g):
            self.grad += k * self.data ** (k - 1) * g
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _backward(g):
            self.grad += e * g
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _backward(g):
            self.grad += g / self.data
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward(g):
            self.grad += (1.0 - t * t) * g
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")
        def _backward(g):
            self.grad += (1.0 if self.data > 0 else 0.0) * g
        out._backward = _backward
        return out

    def __neg__(self):             return self * -1.0
    def __radd__(self, other):     return self + other
    def __sub__(self, other):      return self + (-other)
    def __rsub__(self, other):     return other + (-self)
    def __rmul__(self, other):     return self * other
    def __truediv__(self, other):  return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def topo_order(self):
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        return order

    def backward(self):
        self.grad = 1.0
        for v in reversed(self.topo_order()):
            v._backward(v.grad)
"""

# ------------------------------------------------------------------ Ex 1
SOL_1 = r"""
def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def best_h(f, x, exact, hs, method):
    errs = [abs(method(f, x, h) - exact) for h in hs]
    return float(hs[int(np.argmin(errs))])
"""
STARTER_1 = r"""
def forward_diff(f, x, h):
    # YOUR CODE HERE
    return None

def central_diff(f, x, h):
    # YOUR CODE HERE
    return None

def best_h(f, x, exact, hs, method):
    # return the h in hs (a NumPy array) whose estimate method(f, x, h) is closest to exact
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 2
SOL_2 = r"""
class Dual:
    def __init__(self, val, dot=0.0):
        self.val, self.dot = float(val), float(dot)
    @staticmethod
    def lift(o):
        return o if isinstance(o, Dual) else Dual(o)
    def __add__(self, o):
        o = Dual.lift(o); return Dual(self.val + o.val, self.dot + o.dot)
    def __mul__(self, o):
        o = Dual.lift(o); return Dual(self.val * o.val, self.val * o.dot + self.dot * o.val)
    def __truediv__(self, o):
        o = Dual.lift(o); return Dual(self.val / o.val, (self.dot * o.val - self.val * o.dot) / o.val ** 2)
    def exp(self):
        e = math.exp(self.val); return Dual(e, e * self.dot)
    def sin(self):
        return Dual(math.sin(self.val), math.cos(self.val) * self.dot)
    def cos(self):
        return Dual(math.cos(self.val), -math.sin(self.val) * self.dot)
    __radd__, __rmul__ = __add__, __mul__

def derivative(f, x):
    return f(Dual(x, 1.0)).dot
"""
STARTER_2 = r"""
class Dual:
    def __init__(self, val, dot=0.0):
        self.val, self.dot = float(val), float(dot)
    @staticmethod
    def lift(o):
        return o if isinstance(o, Dual) else Dual(o)
    def __add__(self, o):
        o = Dual.lift(o); return Dual(self.val + o.val, self.dot + o.dot)
    def __mul__(self, o):
        o = Dual.lift(o)
        # YOUR CODE HERE (product rule)
        return Dual(self.val * o.val, 0.0)
    def __truediv__(self, o):
        o = Dual.lift(o)
        # YOUR CODE HERE (quotient rule)
        return Dual(self.val / o.val, 0.0)
    def exp(self):
        # YOUR CODE HERE
        return Dual(math.exp(self.val), 0.0)
    def sin(self):
        # YOUR CODE HERE
        return Dual(math.sin(self.val), 0.0)
    def cos(self):
        # YOUR CODE HERE
        return Dual(math.cos(self.val), 0.0)
    __radd__, __rmul__ = __add__, __mul__

def derivative(f, x):
    # f takes and returns a Dual; return f'(x) using one forward-mode sweep
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 3  (reverse sweep by hand)
SOL_3 = r"""
# forward values
v1 = 2.0 * 3.0          # x1 * x2 = 6
v2 = math.exp(v1 * 0.5)      # exp(v1/2) = e^3
v3 = v1 + v2 * 2.0 ** 2      # v1 + v2 * x1^2
# adjoints
bar_v3 = 1.0
bar_v2 = bar_v3 * 2.0 ** 2                 # 4
bar_v1 = bar_v3 * 1.0 + bar_v2 * v2 * 0.5  # direct + through exp
bar_x1 = bar_v1 * 3.0 + bar_v3 * v2 * 2 * 2.0   # x1 feeds v1 and the x1^2 factor
bar_x2 = bar_v1 * 2.0
adjoints = {"v3": bar_v3, "v2": bar_v2, "v1": bar_v1, "x1": bar_x1, "x2": bar_x2}
"""
STARTER_3 = r"""
# forward values (given)
v1 = 2.0 * 3.0
v2 = math.exp(v1 * 0.5)
v3 = v1 + v2 * 2.0 ** 2
# YOUR CODE HERE: fill in the adjoints (floats) in reverse topological order
bar_v3 = None
bar_v2 = None
bar_v1 = None
bar_x1 = None
bar_x2 = None
adjoints = {"v3": bar_v3, "v2": bar_v2, "v1": bar_v1, "x1": bar_x1, "x2": bar_x2}
"""

# ------------------------------------------------------------------ Ex 4  (topological sort)
SOL_4 = r"""
def topo_sort(nodes, parents):
    order, seen = [], set()
    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for p in parents.get(n, ()):
            visit(p)
        order.append(n)
    for n in nodes:
        visit(n)
    return order

def is_valid_topo(order, nodes, parents):
    if sorted(order) != sorted(nodes) or len(order) != len(set(order)):
        return False
    position = {n: i for i, n in enumerate(order)}
    return all(position[p] < position[n] for n in nodes for p in parents.get(n, ()))
"""
STARTER_4 = r"""
def topo_sort(nodes, parents):
    # nodes: list of hashable node ids; parents: dict node -> tuple of parent ids (leaves may be absent)
    # return a list containing every node exactly once, each after all of its parents
    # YOUR CODE HERE
    return None

def is_valid_topo(order, nodes, parents):
    # True iff order is a permutation of nodes in which every parent precedes its child
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 5  (complete the closures)
_VALUE_CORE_HEAD = r"""
class Value:
    def __init__(self, data, _parents=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = _parents
        self._op = _op
        self._backward = lambda g: None

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward(g):
            self.grad += g
            other.grad += g
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward(g):
            self.grad += k * self.data ** (k - 1) * g
        out._backward = _backward
        return out
"""
_VALUE_CORE_TAIL = r"""
    def __neg__(self):             return self * -1.0
    def __radd__(self, other):     return self + other
    def __sub__(self, other):      return self + (-other)
    def __rsub__(self, other):     return other + (-self)
    def __rmul__(self, other):     return self * other
    def __truediv__(self, other):  return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def topo_order(self):
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        return order

    def backward(self):
        self.grad = 1.0
        for v in reversed(self.topo_order()):
            v._backward(v.grad)
"""
SOL_5 = _VALUE_CORE_HEAD + r"""
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward(g):
            self.grad += other.data * g
            other.grad += self.data * g
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _backward(g):
            self.grad += e * g
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _backward(g):
            self.grad += g / self.data
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward(g):
            self.grad += (1.0 - t * t) * g
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")
        def _backward(g):
            self.grad += (1.0 if self.data > 0 else 0.0) * g
        out._backward = _backward
        return out
""" + _VALUE_CORE_TAIL
STARTER_5 = _VALUE_CORE_HEAD + r"""
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward(g):
            # YOUR CODE HERE: add the contributions to self.grad and other.grad
            pass
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _backward(g):
            # YOUR CODE HERE
            pass
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _backward(g):
            # YOUR CODE HERE
            pass
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward(g):
            # YOUR CODE HERE
            pass
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")
        def _backward(g):
            # YOUR CODE HERE
            pass
        out._backward = _backward
        return out
""" + _VALUE_CORE_TAIL

# ------------------------------------------------------------------ Ex 6  (two-layer backprop in NumPy)
SOL_6 = r"""
def mlp_grads(W1, b1, W2, b2, X, Y):
    N = X.shape[0]
    Z1 = X @ W1.T + b1                 # (N, H)
    A1 = np.tanh(Z1)
    Yhat = A1 @ W2.T + b2              # (N, K)
    loss = 0.5 * np.sum((Yhat - Y) ** 2) / N
    delta2 = (Yhat - Y) / N            # (N, K)
    dW2 = delta2.T @ A1                # (K, H)
    db2 = delta2.sum(axis=0)
    delta1 = (delta2 @ W2) * (1.0 - A1 ** 2)   # (N, H)
    dW1 = delta1.T @ X                 # (H, D)
    db1 = delta1.sum(axis=0)
    return loss, dW1, db1, dW2, db2
"""
STARTER_6 = r"""
def mlp_grads(W1, b1, W2, b2, X, Y):
    # forward: Z1 = X W1^T + b1, A1 = tanh(Z1), Yhat = A1 W2^T + b2
    # loss = (1/(2N)) * sum((Yhat - Y)**2)
    # return loss, dW1, db1, dW2, db2 with the shapes of W1, b1, W2, b2
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 7  (matmul backward)
SOL_7 = r"""
def matmul_backward(A, B, G):
    return G @ B.T, A.T @ G

def numerical_matmul_grads(A, B, loss_fn, h=1e-6):
    dA = np.zeros_like(A); dB = np.zeros_like(B)
    for idx in np.ndindex(A.shape):
        Ap = A.copy(); Am = A.copy(); Ap[idx] += h; Am[idx] -= h
        dA[idx] = (loss_fn(Ap @ B) - loss_fn(Am @ B)) / (2 * h)
    for idx in np.ndindex(B.shape):
        Bp = B.copy(); Bm = B.copy(); Bp[idx] += h; Bm[idx] -= h
        dB[idx] = (loss_fn(A @ Bp) - loss_fn(A @ Bm)) / (2 * h)
    return dA, dB
"""
STARTER_7 = r"""
def matmul_backward(A, B, G):
    # Y = A @ B, G = dL/dY (same shape as Y). Return (dL/dA, dL/dB).
    # YOUR CODE HERE
    return None

def numerical_matmul_grads(A, B, loss_fn, h=1e-6):
    # central differences on every entry of A and of B for the scalar loss_fn(A @ B)
    # return (dA, dB) with the shapes of A and B
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 8  (fused softmax-CE)
SOL_8 = r"""
def log_softmax(Z):
    m = Z.max(axis=1, keepdims=True)
    return Z - m - np.log(np.sum(np.exp(Z - m), axis=1, keepdims=True))

def softmax_ce(Z, labels):
    N = Z.shape[0]
    logp = log_softmax(Z)
    loss = -np.mean(logp[np.arange(N), labels])
    P = np.exp(logp)
    grad = P.copy()
    grad[np.arange(N), labels] -= 1.0
    return loss, grad / N
"""
STARTER_8 = r"""
def log_softmax(Z):
    # row-wise log-softmax of a (N, K) array, stable for large entries
    # YOUR CODE HERE
    return None

def softmax_ce(Z, labels):
    # mean cross-entropy over the N rows (labels: integer array of shape (N,)) and its gradient dL/dZ, shape (N, K)
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 9  (gradient checker)
SOL_9 = r"""
def grad_check(loss_fn, grad_fn, theta, n_check=None, h=1e-5, rng=None):
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(grad_fn(theta), dtype=float)
    idx = np.arange(theta.size)
    if n_check is not None and n_check < theta.size:
        idx = (rng if rng is not None else np.random.default_rng(0)).choice(theta.size, size=n_check, replace=False)
    g_sel, g_num = np.empty(len(idx)), np.empty(len(idx))
    for j, k in enumerate(idx):
        e = np.zeros_like(theta); e.flat[k] = h
        g_num[j] = (loss_fn(theta + e) - loss_fn(theta - e)) / (2 * h)
        g_sel[j] = g.flat[k]
    rel = np.linalg.norm(g_sel - g_num) / (np.linalg.norm(g_sel) + np.linalg.norm(g_num) + 1e-300)
    worst = int(idx[np.argmax(np.abs(g_sel - g_num))])
    return float(rel), worst
"""
STARTER_9 = r"""
def grad_check(loss_fn, grad_fn, theta, n_check=None, h=1e-5, rng=None):
    # loss_fn(theta) -> float, grad_fn(theta) -> array like theta.
    # Check all entries (or n_check randomly chosen ones) with central differences and return
    # (relative_error, index_of_worst_entry), where relative_error = ||g - g_num|| / (||g|| + ||g_num||).
    # YOUR CODE HERE
    return None
"""

# ------------------------------------------------------------------ Ex 11  (planted bug)
BUGGY_LAYER = r"""
def buggy_forward_backward(W, b, X, Y):
    # a tanh regression layer with a planted bug in the backward pass
    Z = X @ W.T + b
    A = np.tanh(Z)
    loss = 0.5 * np.mean((A - Y) ** 2)
    G = (A - Y) / A.size
    dZ = G * (1.0 - Z ** 2)            # <-- derivative of tanh written in terms of the input
    dW = dZ.T @ X
    db = dZ.sum(axis=0)
    return loss, dW, db
"""
SOL_11 = r"""
def fixed_forward_backward(W, b, X, Y):
    Z = X @ W.T + b
    A = np.tanh(Z)
    loss = 0.5 * np.mean((A - Y) ** 2)
    G = (A - Y) / A.size
    dZ = G * (1.0 - A ** 2)            # tanh' = 1 - tanh(z)^2, i.e. 1 - A**2
    dW = dZ.T @ X
    db = dZ.sum(axis=0)
    return loss, dW, db

bug_location = "tanh derivative"
"""
STARTER_11 = r"""
def fixed_forward_backward(W, b, X, Y):
    # copy buggy_forward_backward here and repair the one wrong line
    # YOUR CODE HERE
    return None

# one of: "forward pass", "loss", "upstream gradient G", "tanh derivative", "dW", "db"
bug_location = None
"""

# ------------------------------------------------------------------ Project
PROJECT_SOLUTION = r"""
from sklearn.datasets import make_moons

def bce_with_logits(s, y):
    sd = s.data
    out = Value(max(sd, 0.0) + math.log1p(math.exp(-abs(sd))) - y * sd, (s,), "bce")
    p = 1.0 / (1.0 + math.exp(-sd))
    def _backward(g):
        s.grad += (p - y) * g
    out._backward = _backward
    return out

class Neuron:
    def __init__(self, n_in, nonlin=True):
        self.w = [Value(rng.normal(0.0, math.sqrt(2.0 / n_in))) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin
    def __call__(self, xs):
        act = self.b
        for wi, xi in zip(self.w, xs):
            act = act + wi * xi
        return act.relu() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out, nonlin=True):
        self.neurons = [Neuron(n_in, nonlin) for _ in range(n_out)]
    def __call__(self, xs):
        return [n(xs) for n in self.neurons]
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i + 1], nonlin=(i < len(sizes) - 2)) for i in range(len(sizes) - 1)]
    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs[0]
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

X, y = make_moons(n_samples=200, noise=0.15, random_state=0)
model = MLP([2, 16, 1])
params = model.parameters()

def minibatch_loss(idx):
    total = Value(0.0)
    for i in idx:
        total = total + bce_with_logits(model([X[i, 0], X[i, 1]]), float(y[i]))
    return total * (1.0 / len(idx))

def train_accuracy():
    s = np.array([model([a, b]).data for a, b in X])
    return float(np.mean((s > 0) == (y == 1)))

# gradient check before training (on 10 random parameters, one small minibatch)
idx0 = rng.choice(len(X), size=8, replace=False)
for p in params:
    p.grad = 0.0
minibatch_loss(idx0).backward()
g_ana, g_num, h = [], [], 1e-5
for k in rng.choice(len(params), size=10, replace=False):
    p = params[k]; g_ana.append(p.grad); old = p.data
    p.data = old + h; lp = minibatch_loss(idx0).data
    p.data = old - h; lm = minibatch_loss(idx0).data
    p.data = old; g_num.append((lp - lm) / (2 * h))
g_ana, g_num = np.array(g_ana), np.array(g_num)
print(f"gradient check before training: relative error {np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num)):.1e}")

# Adam, because plain SGD needs more steps than the scalar engine can afford
lr, beta1, beta2, eps = 0.05, 0.9, 0.999, 1e-8
m = np.zeros(len(params)); v = np.zeros(len(params))
losses = []
for step in range(1, 301):
    idx = rng.choice(len(X), size=32, replace=False)
    loss = minibatch_loss(idx)
    for p in params:
        p.grad = 0.0
    loss.backward()
    g = np.array([p.grad for p in params])
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    update = lr * (m / (1 - beta1 ** step)) / (np.sqrt(v / (1 - beta2 ** step)) + eps)
    for p, u in zip(params, update):
        p.data -= u
    losses.append(loss.data)
    if step % 100 == 0:
        print(f"step {step}: minibatch loss {loss.data:.3f}, train accuracy {train_accuracy():.3f}")

final_acc = train_accuracy()
print(f"final train accuracy: {final_acc:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(losses, lw=0.8, alpha=0.5); axes[0].plot(np.convolve(losses, np.ones(15) / 15, mode="valid"), lw=2)
axes[0].set_xlabel("Adam step"); axes[0].set_ylabel("minibatch BCE"); axes[0].set_title("Training loss")
gx, gy = np.meshgrid(np.linspace(-1.8, 2.8, 50), np.linspace(-1.3, 1.8, 40))
S = np.array([model([a, b]).data for a, b in zip(gx.ravel(), gy.ravel())]).reshape(gx.shape)
axes[1].contourf(gx, gy, 1 / (1 + np.exp(-S)), levels=20, cmap="coolwarm", alpha=0.6)
axes[1].contour(gx, gy, S, levels=[0.0], colors="k")
axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15, edgecolor="k", linewidth=0.3)
axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2"); axes[1].set_title(f"Decision boundary (accuracy {final_acc:.2f})")
plt.tight_layout(); plt.show()
"""

# =============================================================================
# Compute hidden answers by executing the reference solutions
# =============================================================================
import matplotlib
matplotlib.use("Agg")
ns = {}
exec(SETUP, ns)
exec(VALUE_FULL, ns)
exec(SOL_1, ns); exec(SOL_2, ns); exec(SOL_3, ns); exec(SOL_4, ns)
exec(SOL_6, ns); exec(SOL_7, ns); exec(SOL_8, ns); exec(SOL_9, ns); exec(BUGGY_LAYER, ns); exec(SOL_11, ns)

# Ex 1 hidden: forward and central estimates of (x^3)' at 2 with h = 0.1 (closed form: 12.61 and 12.01)
_cube = lambda t: t ** 3
HASH_1_F = sha_of(ns["forward_diff"](_cube, 2.0, 0.1), decimals=3)
HASH_1_C = sha_of(ns["central_diff"](_cube, 2.0, 0.1), decimals=3)
_hs = np.logspace(-12, -1, 23)
print("ex1 estimates:", ns["forward_diff"](_cube, 2.0, 0.1), ns["central_diff"](_cube, 2.0, 0.1),
      "best h:", ns["best_h"](np.exp, 1.0, np.e, _hs, ns["forward_diff"]), ns["best_h"](np.exp, 1.0, np.e, _hs, ns["central_diff"]))

# Ex 2 hidden: derivative of x*exp(x)*sin(x)/(1+x) at 1.3
_f2 = lambda x: x * x.exp() * x.sin() / (x + 1.0)
_d2 = ns["derivative"](_f2, 2.1)
_d2_exact = (lambda x: (math.exp(x) * math.sin(x) * (1 + x) + x * math.exp(x) * math.sin(x) * (1 + x) + x * math.exp(x) * math.cos(x) * (1 + x) - x * math.exp(x) * math.sin(x)) / (1 + x) ** 2)(2.1)
assert abs(_d2 - _d2_exact) < 1e-12, (_d2, _d2_exact)
HASH_2 = sha_of(_d2, decimals=3)
print("ex2 derivative:", _d2)

# Ex 3 hidden adjoints
_adj = ns["adjoints"]
HASH_3_X1 = sha_of(_adj["x1"], decimals=3)
HASH_3_X2 = sha_of(_adj["x2"], decimals=3)
HASH_3_V1 = sha_of(_adj["v1"], decimals=3)
# cross-check adjoints with the Value engine
_V = ns["Value"]
_x1, _x2 = _V(2.0), _V(3.0)
_v1 = _x1 * _x2; _v2 = (_v1 * 0.5).exp(); _v3 = _v1 + _v2 * _x1 ** 2
_v3.backward()
assert abs(_x1.grad - _adj["x1"]) < 1e-9 and abs(_x2.grad - _adj["x2"]) < 1e-9, (_x1.grad, _adj)
print("ex3 adjoints:", _adj)

# Ex 5 hidden: gradient of a fixed expression through the completed engine
exec(SOL_5, ns)
_a, _b, _c = ns["Value"](0.8), ns["Value"](-0.3), ns["Value"](1.7)
_out = ((_a * _b).exp() + _c.log() * _a.tanh()).relu() * (_a * 2.0 - _b) ** 2
_out.backward()
HASH_5 = sha_of([_a.grad, _b.grad, _c.grad], decimals=3)
print("ex5 grads:", _a.grad, _b.grad, _c.grad)

# Ex 6 hidden: sum of |dW1| on a seeded problem
_r6 = np.random.default_rng(6)
_X6 = _r6.normal(size=(5, 3)); _Y6 = _r6.normal(size=(5, 2))
_W1 = _r6.normal(size=(4, 3)) * 0.5; _b1 = _r6.normal(size=4) * 0.1
_W2 = _r6.normal(size=(2, 4)) * 0.5; _b2 = _r6.normal(size=2) * 0.1
_loss6, _dW1, _db1, _dW2, _db2 = ns["mlp_grads"](_W1, _b1, _W2, _b2, _X6, _Y6)
HASH_6 = sha_of(float(np.sum(np.abs(_dW1))), decimals=3)
print("ex6 loss, sum|dW1|:", _loss6, float(np.sum(np.abs(_dW1))))

# Ex 8 hidden: loss on a fixed logits matrix
_Z8 = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 0.5], [100.0, 0.0, -100.0], [-3.0, 4.0, 1.0]])
_lab8 = np.array([0, 2, 2, 1])
_loss8, _grad8 = ns["softmax_ce"](_Z8, _lab8)
HASH_8 = sha_of(_loss8, decimals=3)
print("ex8 loss:", _loss8)

# Ex 11 hidden: location string hash + check that the fix matches finite differences
HASH_11 = sha_of("tanh derivative")
_r11 = np.random.default_rng(11)
_X11 = _r11.normal(size=(6, 3)); _Y11 = _r11.uniform(-0.9, 0.9, size=(6, 2))
_W11 = _r11.normal(size=(2, 3)); _b11 = _r11.normal(size=2) * 0.1
_lf = lambda th: ns["fixed_forward_backward"](th[:6].reshape(2, 3), th[6:], _X11, _Y11)[0]
_gf = lambda th: np.concatenate([ns["fixed_forward_backward"](th[:6].reshape(2, 3), th[6:], _X11, _Y11)[1].ravel(),
                                 ns["fixed_forward_backward"](th[:6].reshape(2, 3), th[6:], _X11, _Y11)[2]])
_rel_fixed, _ = ns["grad_check"](_lf, _gf, np.concatenate([_W11.ravel(), _b11]))
_gb = lambda th: np.concatenate([ns["buggy_forward_backward"](th[:6].reshape(2, 3), th[6:], _X11, _Y11)[1].ravel(),
                                 ns["buggy_forward_backward"](th[:6].reshape(2, 3), th[6:], _X11, _Y11)[2]])
_rel_buggy, _ = ns["grad_check"](_lf, _gb, np.concatenate([_W11.ravel(), _b11]))
print("ex11 relative error fixed / buggy:", _rel_fixed, _rel_buggy)
assert _rel_fixed < 1e-7 and _rel_buggy > 1e-2

# =============================================================================
# Assemble the exercise set
# =============================================================================
ex = ExerciseSet(
    "Module 06 — Exercises",
    intro_md=r"""
These exercises rebuild the pieces of an automatic-differentiation system: finite differences,
forward mode with dual numbers, a reverse sweep by hand, topological sorting, the closures of a scalar
autograd engine, the matrix backward rules, a fused classification loss, a gradient checker, and a
planted bug to find. The mini-project trains a network with *your* engine. Only NumPy, `math` and
matplotlib are allowed — no PyTorch, no JAX. The concept guide (`README.md`) has every formula you need.
""",
    setup_code=SETUP,
)

# ------------------------------------------------------------------ 1
ex.add(
    title="Finite differences and the best step size", difficulty=1,
    prompt_md=r"""
Implement the two finite-difference estimators of $f'(x)$,

$$D^{+}_h f(x) = \frac{f(x+h) - f(x)}{h}, \qquad D^{c}_h f(x) = \frac{f(x+h) - f(x-h)}{2h},$$

as `forward_diff(f, x, h)` and `central_diff(f, x, h)` (both return a float), and `best_h(f, x, exact, hs, method)`,
which returns the entry of the NumPy array `hs` for which `method(f, x, h)` is closest to the known value `exact`.

Then answer (the checks hash your numbers): for $f(x) = x^3$ at $x = 2$ with $h = 0.1$, what do the two estimators return?
""",
    starter=STARTER_1, solution=SOL_1,
    check=rf"""
cube = lambda t: t ** 3
try:
    f_est = forward_diff(cube, 2.0, 0.1)
    c_est = central_diff(cube, 2.0, 0.1)
except Exception:
    f_est = c_est = None
check_hash('forward estimate of (x^3)\' at 2 with h=0.1', f_est, '{HASH_1_F}')
check_hash('central estimate of (x^3)\' at 2 with h=0.1', c_est, '{HASH_1_C}')
try:
    hs = np.logspace(-12, -1, 23)
    bh_f = best_h(np.exp, 1.0, np.e, hs, forward_diff)
    bh_c = best_h(np.exp, 1.0, np.e, hs, central_diff)
except Exception:
    bh_f = bh_c = None
check('best forward-difference h for exp at 1 lies in [1e-9, 1e-7]', bh_f is not None and 1e-9 <= bh_f <= 1e-7)
check('best central-difference h for exp at 1 lies in [1e-7, 1e-4]', bh_c is not None and 1e-7 <= bh_c <= 1e-4)
check('central beats forward at h=1e-3 for sin at 0.5',
      (abs(central_diff(np.sin, 0.5, 1e-3) - np.cos(0.5)) < abs(forward_diff(np.sin, 0.5, 1e-3) - np.cos(0.5))) if c_est is not None else False)
""",
    explanation_md=r"""
The forward estimate carries an $O(h)$ error ($3xh + h^2 = 0.61$ here, giving 12.61) while the central one is
$O(h^2)$ ($h^2 = 0.01$, giving 12.01). The best step is where truncation error and round-off error ($\approx \varepsilon|f|/h$)
balance: about $\sqrt{\varepsilon} \approx 10^{-8}$ for forward differences and $\varepsilon^{1/3} \approx 10^{-5}$ for central
differences. Going smaller than that makes the estimate *worse*, which is the mistake people make when they
"improve" a gradient check by choosing $h = 10^{-12}$.
""")

# ------------------------------------------------------------------ 2
ex.add(
    title="Forward mode with dual numbers", difficulty=1,
    prompt_md=r"""
Complete the `Dual` class: a value `val` carrying a tangent `dot`. Addition is given. Implement the product rule in
`__mul__`, the quotient rule in `__truediv__`, and `exp`, `sin`, `cos` via $g(a, \dot a) = (g(a), g'(a)\,\dot a)$.
Then implement `derivative(f, x)` which returns $f'(x)$ from **one** forward sweep (seed the input with tangent 1).

The hidden check asks for $\frac{d}{dx}\Big[\frac{x\,e^{x}\sin x}{x + 1}\Big]$ at $x = 2.1$.
""",
    starter=STARTER_2, solution=SOL_2,
    check=rf"""
try:
    d_prod = derivative(lambda x: x * x * x, 2.0)                      # 3x^2 = 12
    d_quot = derivative(lambda x: x.sin() / x, 2.0)                    # (x cos x - sin x)/x^2
    d_exp = derivative(lambda x: (x * 3.0).exp(), 0.5)                 # 3 e^{{1.5}}
    d_hidden = derivative(lambda x: x * x.exp() * x.sin() / (x + 1.0), 2.1)
except Exception:
    d_prod = d_quot = d_exp = d_hidden = None
check_close('product rule: d/dx x^3 at 2', d_prod, 12.0)
check_close('quotient rule: d/dx sin(x)/x at 2', d_quot, (2 * np.cos(2) - np.sin(2)) / 4)
check_close('chain through exp: d/dx e^(3x) at 0.5', d_exp, 3 * np.exp(1.5))
check_hash('hidden: derivative of x e^x sin x / (x+1) at 2.1', d_hidden, '{HASH_2}')
""",
    explanation_md=r"""
A dual number $(a, \dot a)$ behaves like $a + \dot a\,\epsilon$ with $\epsilon^2 = 0$, so multiplying two of them gives
$ab + (a\dot b + \dot a b)\epsilon$: the product rule falls out of the algebra. Every elementary function is
lifted by the chain rule, $g(a + \dot a\epsilon) = g(a) + g'(a)\dot a\,\epsilon$. Seeding the input with $\dot x = 1$ makes
every tangent equal to $\partial(\cdot)/\partial x$, so the output's tangent *is* $f'(x)$ — exact to rounding, with no step $h$.
The common wrong turn is forgetting to lift plain floats (`x + 1.0`) to duals with tangent 0.
""")

# ------------------------------------------------------------------ 3
ex.add(
    title="A reverse sweep by hand", difficulty=1,
    prompt_md=r"""
Consider the graph with inputs $x_1 = 2$, $x_2 = 3$ and nodes

$$v_1 = x_1 x_2, \qquad v_2 = \exp(v_1 / 2), \qquad v_3 = v_1 + v_2\, x_1^2 .$$

Note that $x_1$ feeds two nodes ($v_1$ and the factor $x_1^2$) and $v_1$ feeds two nodes ($v_2$ and $v_3$ directly).
Fill in the adjoints $\bar v_3, \bar v_2, \bar v_1, \bar x_1, \bar x_2$ (as floats) by one reverse sweep,
summing contributions where a node has several children. Use the forward values already computed in the cell.
""",
    starter=STARTER_3, solution=SOL_3,
    check=rf"""
def _get(k):
    try:
        v = adjoints[k]
        return None if v is None else float(v)
    except Exception:
        return None
check_close('adjoint of the output is 1', _get('v3'), 1.0)
check_close('adjoint of v2 is x1^2 = 4', _get('v2'), 4.0)
check_hash('hidden: adjoint of v1 (two contributions)', _get('v1'), '{HASH_3_V1}')
check_hash('hidden: adjoint of x1 (two contributions)', _get('x1'), '{HASH_3_X1}')
check_hash('hidden: adjoint of x2', _get('x2'), '{HASH_3_X2}')
""",
    explanation_md=r"""
Reverse topological order: $\bar v_3 = 1$. Then $\bar v_2 = \bar v_3 \cdot x_1^2 = 4$. The node $v_1$ has two children, so
$\bar v_1 = \bar v_3 \cdot 1 + \bar v_2 \cdot \tfrac12 e^{v_1/2} = 1 + 2 e^{3} \approx 41.17$. Likewise $x_1$ feeds $v_1$ and
$x_1^2$: $\bar x_1 = \bar v_1 x_2 + \bar v_3 \cdot 2 x_1 v_2 = 3\bar v_1 + 4 e^3 \approx 203.9$, and $\bar x_2 = \bar v_1 x_1 \approx 82.3$.
Forgetting one of the two contributions — the `=` instead of `+=` bug — is the single most common error in hand-written
backward passes, and it is exactly what the `+=` in an autograd engine protects against.
""")

# ------------------------------------------------------------------ 4
ex.add(
    title="Topological sort of a computational graph", difficulty=2,
    prompt_md=r"""
A graph is given as a list `nodes` and a dict `parents` mapping each node to a tuple of its parents (leaves have
no entry or an empty tuple). Implement `topo_sort(nodes, parents)` returning a list that contains every node exactly
once with each node *after* all of its parents (depth-first search with a visited set, as in the `Value.backward`
method), and `is_valid_topo(order, nodes, parents)` returning `True` iff `order` is such a list. Your function must
handle nodes that are reachable from several children (diamonds) and must not rely on the input order of `nodes`.
""",
    starter=STARTER_4, solution=SOL_4,
    check=r"""
chain_nodes = ["x", "a", "b", "L"]
chain_par = {"a": ("x",), "b": ("a",), "L": ("b",)}
diamond_nodes = ["L", "v3", "v2", "v1", "x1", "x2"]          # deliberately scrambled
diamond_par = {"v1": ("x1", "x2"), "v2": ("v1",), "v3": ("v1", "x1"), "L": ("v2", "v3")}
try:
    o1 = topo_sort(chain_nodes, chain_par)
    o2 = topo_sort(diamond_nodes, diamond_par)
except Exception:
    o1 = o2 = None
check('chain graph sorted', o1 == ["x", "a", "b", "L"])
def _valid(order, nodes, parents):
    if order is None or sorted(order) != sorted(nodes):
        return False
    pos = {n: i for i, n in enumerate(order)}
    return all(pos[p] < pos[n] for n in nodes for p in parents.get(n, ()))
check('diamond graph: every node once, parents first', _valid(o2, diamond_nodes, diamond_par))
try:
    ok_valid = is_valid_topo(["x1", "x2", "v1", "v2", "v3", "L"], diamond_nodes, diamond_par)
    ok_invalid = is_valid_topo(["x1", "v1", "x2", "v2", "v3", "L"], diamond_nodes, diamond_par)
    ok_dup = is_valid_topo(["x1", "x2", "v1", "v1", "v2", "L"], diamond_nodes, diamond_par)
except Exception:
    ok_valid = ok_invalid = ok_dup = None
check('is_valid_topo accepts a valid order', ok_valid is True)
check('is_valid_topo rejects a parent after its child', ok_invalid is False)
check('is_valid_topo rejects duplicates / missing nodes', ok_dup is False)
# random DAGs: generated so that parents always have smaller ids
ok_random = True
for trial in range(5):
    n = 12
    par = {i: tuple(int(p) for p in rng.choice(i, size=min(i, 2), replace=False)) for i in range(1, n)}
    nodes = list(rng.permutation(n))
    try:
        ok_random &= _valid(topo_sort(nodes, par), nodes, par)
    except Exception:
        ok_random = False
check('5 random DAGs sorted correctly', bool(ok_random))
""",
    explanation_md=r"""
Depth-first search appends a node *after* recursing into its parents, so by the time a node is appended all its
ancestors are already in the list; the `seen` set guarantees that a node reachable along several paths (the diamond)
is appended once. Starting the search from every node in turn makes the result independent of the input order and
covers disconnected pieces. The validity test is the definition itself: a permutation of the nodes in which every
parent's position is smaller than its child's. `backward()` walks this order *in reverse* so that a node's gradient
is complete before it is handed to its parents.
""")

# ------------------------------------------------------------------ 5
ex.add(
    title="Complete the autograd engine", difficulty=2,
    prompt_md=r"""
The `Value` class below is complete except for the backward closures of `*`, `exp`, `log`, `tanh` and `relu`.
Each closure receives `g`, the gradient of the loss with respect to the node's output, and must **add** the correct
contribution to each parent's `.grad` (use the local derivative rules in the concept guide's table). Note that the
forward values `e` and `t` are already available to the closures.

The checks compare your gradients with closed forms and with central differences; the hidden check differentiates
a composite expression through all five operations.
""",
    starter=STARTER_5, solution=SOL_5,
    check=rf"""
def _grads(fn, *vals):
    try:
        vs = [Value(v) for v in vals]
        out = fn(*vs)
        out.backward()
        return [v.grad for v in vs]
    except Exception:
        return None
def _numgrad(fn, *vals, h=1e-6):
    out = []
    for i in range(len(vals)):
        p = list(vals); m = list(vals); p[i] += h; m[i] -= h
        out.append((fn(*p) - fn(*m)) / (2 * h))
    return out
g_mul = _grads(lambda a, b: a * b, 3.0, -2.0)
check_close('mul: d(ab)/da = b and d(ab)/db = a', g_mul, [-2.0, 3.0])
check_close('mul: a*a gives 2a (two contributions)', _grads(lambda a: a * a, 1.5), [3.0])
check_close('exp: derivative is the output', _grads(lambda a: a.exp(), 0.7), [np.exp(0.7)])
check_close('log: derivative is 1/a', _grads(lambda a: a.log(), 4.0), [0.25])
check_close('tanh: derivative is 1 - tanh^2', _grads(lambda a: a.tanh(), 0.6), [1 - np.tanh(0.6) ** 2])
check_close('relu: 1 for positive, 0 for negative input', [_grads(lambda a: a.relu(), 2.0), _grads(lambda a: a.relu(), -2.0)], [[1.0], [0.0]])
composite = lambda a, b, c: ((a * b).exp() + c.log() * a.tanh()).relu() * (a * 2.0 - b) ** 2
num = _numgrad(lambda a, b, c: max(math.exp(a * b) + math.log(c) * math.tanh(a), 0.0) * (a * 2.0 - b) ** 2, 0.8, -0.3, 1.7)
g_comp = _grads(composite, 0.8, -0.3, 1.7)
check_close('composite expression matches central differences', g_comp, num, tol=1e-5)
check_hash('hidden: composite gradient', g_comp, '{HASH_5}')
""",
    explanation_md=r"""
Each closure is one line of calculus: $\partial(ab)/\partial a = b$ (so `self.grad += other.data * g`, and symmetrically),
$(e^a)' = e^a$ (the stored output `e`), $(\ln a)' = 1/a$ (needs the input), $\tanh' = 1 - t^2$ (the stored output `t`),
and $\mathrm{relu}'(a) = \mathbb 1[a > 0]$. Two things to get right: *accumulate* with `+=` so that `a * a` yields $2a$ rather than
$a$, and multiply by the incoming `g` — the chain rule — rather than returning the bare local derivative. A gradient
check is the fastest way to find out which of the five you got wrong.
""")

# ------------------------------------------------------------------ 6
ex.add(
    title="Backpropagation through a two-layer network in NumPy", difficulty=2,
    prompt_md=r"""
Implement `mlp_grads(W1, b1, W2, b2, X, Y)` for a network with one $\tanh$ hidden layer and a linear output, trained
with the mean squared error over $N$ examples stored as **rows**:

$$Z_1 = X W_1^\top + b_1,\quad A_1 = \tanh(Z_1),\quad \hat Y = A_1 W_2^\top + b_2,\quad
\mathcal L = \frac{1}{2N}\sum_{n,k} (\hat Y_{nk} - Y_{nk})^2 .$$

Shapes: `X` is $(N, D)$, `W1` is $(H, D)$, `b1` is $(H,)$, `W2` is $(K, H)$, `b2` is $(K,)$, `Y` is $(N, K)$.
Return `loss, dW1, db1, dW2, db2` with the gradients shaped like the corresponding parameters. Use the recurrence
$\delta_2 = (\hat Y - Y)/N$, $\delta_1 = (\delta_2 W_2) \odot (1 - A_1^2)$, $\partial\mathcal L/\partial W_l = \delta_l^\top A_{l-1}$,
$\partial\mathcal L/\partial b_l = \sum_n \delta_l$.
""",
    starter=STARTER_6, solution=SOL_6,
    check=rf"""
r6 = np.random.default_rng(6)
X6 = r6.normal(size=(5, 3)); Y6 = r6.normal(size=(5, 2))
W1 = r6.normal(size=(4, 3)) * 0.5; b1 = r6.normal(size=4) * 0.1
W2 = r6.normal(size=(2, 4)) * 0.5; b2 = r6.normal(size=2) * 0.1
try:
    loss, dW1, db1, dW2, db2 = mlp_grads(W1, b1, W2, b2, X6, Y6)
except Exception:
    loss = dW1 = db1 = dW2 = db2 = None
def _loss_only(W1_, b1_, W2_, b2_):
    A1 = np.tanh(X6 @ W1_.T + b1_)
    return 0.5 * np.sum((A1 @ W2_.T + b2_ - Y6) ** 2) / len(X6)
check_close('loss value', loss, _loss_only(W1, b1, W2, b2))
check('gradient shapes match the parameters',
      dW1 is not None and np.shape(dW1) == W1.shape and np.shape(db1) == b1.shape and np.shape(dW2) == W2.shape and np.shape(db2) == b2.shape)
def _num(param, setter, h=1e-6):
    g = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        p = param.copy(); m = param.copy(); p[idx] += h; m[idx] -= h
        g[idx] = (setter(p) - setter(m)) / (2 * h)
    return g
check_close('dW2 matches central differences', dW2, _num(W2, lambda P: _loss_only(W1, b1, P, b2)), tol=1e-6)
check_close('db2 matches central differences', db2, _num(b2, lambda P: _loss_only(W1, b1, W2, P)), tol=1e-6)
check_close('dW1 matches central differences', dW1, _num(W1, lambda P: _loss_only(P, b1, W2, b2)), tol=1e-6)
check_close('db1 matches central differences', db1, _num(b1, lambda P: _loss_only(W1, P, W2, b2)), tol=1e-6)
check_hash('hidden: sum of |dW1|', (float(np.sum(np.abs(dW1))) if dW1 is not None else None), '{HASH_6}')
""",
    explanation_md=r"""
With examples as rows, the layer map is $A W^\top + b$ and every rule of the concept guide is transposed accordingly:
$\partial\mathcal L/\partial W_2 = \delta_2^\top A_1$ has shape $(K, N)(N, H) = (K, H)$ — the shape of $W_2$, which is the sanity check.
The bias was broadcast across rows in the forward pass, so its gradient sums over rows. The mask $1 - A_1^2$ is
$\tanh'(Z_1)$ written through the output, and the $1/N$ of the mean enters once, in $\delta_2$, and is carried through
everything downstream by the chain rule. Writing `delta2.T @ A1` as `A1.T @ delta2` gives a $(H, K)$ array — wrong shape
— which is why the shape test catches most mistakes before the numerical one does.
""")

# ------------------------------------------------------------------ 7
ex.add(
    title="The matmul backward rule", difficulty=2,
    prompt_md=r"""
For $Y = AB$ with $A \in \mathbb R^{m\times k}$, $B \in \mathbb R^{k \times n}$ and upstream gradient $G = \partial L/\partial Y \in \mathbb R^{m \times n}$,
implement `matmul_backward(A, B, G)` returning $(\bar A, \bar B) = (G B^\top,\ A^\top G)$.

Then write `numerical_matmul_grads(A, B, loss_fn, h=1e-6)` that estimates $\partial L/\partial A$ and $\partial L/\partial B$ entry by entry with
central differences for an arbitrary scalar `loss_fn(Y)` — this is the independent witness that lets you check the rule
for *any* downstream loss, not only $L = \sum Y$. Use `np.ndindex` to loop over entries.
""",
    starter=STARTER_7, solution=SOL_7,
    check=r"""
A = np.array([[1.0, 2.0], [3.0, 4.0]]); B = np.array([[1.0, 0.0], [1.0, 1.0]])
try:
    dA, dB = matmul_backward(A, B, np.ones((2, 2)))
except Exception:
    dA = dB = None
check_close('dA for L = sum(AB)', dA, np.array([[1.0, 2.0], [1.0, 2.0]]))
check_close('dB for L = sum(AB)', dB, np.array([[4.0, 4.0], [6.0, 6.0]]))
A2 = rng.normal(size=(3, 4)); B2 = rng.normal(size=(4, 2)); C2 = rng.normal(size=(3, 2))
loss_fn = lambda Y: float(np.sum((Y - C2) ** 2) + np.sum(np.sin(Y)))
try:
    Y2 = A2 @ B2
    G2 = 2 * (Y2 - C2) + np.cos(Y2)                 # dL/dY for this loss
    dA2, dB2 = matmul_backward(A2, B2, G2)
    nA, nB = numerical_matmul_grads(A2, B2, loss_fn)
except Exception:
    dA2 = dB2 = nA = nB = None
check('shapes: dA like A, dB like B', dA2 is not None and np.shape(dA2) == A2.shape and np.shape(dB2) == B2.shape)
check_close('rectangular case: dA matches central differences', dA2, nA, tol=1e-5)
check_close('rectangular case: dB matches central differences', dB2, nB, tol=1e-5)
""",
    explanation_md=r"""
From $Y_{ij} = \sum_k A_{ik} B_{kj}$: $A_{ik}$ appears in row $i$ of $Y$ with coefficients $B_{kj}$, so
$\bar A_{ik} = \sum_j G_{ij} B_{kj} = (G B^\top)_{ik}$; and $B_{kj}$ appears in column $j$ with coefficients $A_{ik}$, so
$\bar B_{kj} = \sum_i A_{ik} G_{ij} = (A^\top G)_{kj}$. The rectangular test matters: with square matrices several wrong
arrangements (such as $B^\top G$) still have the right shape and only fail numerically, whereas with $3\times4$ and $4\times2$
matrices only the correct formula even multiplies. Note that the numerical witness never needs to know the rule — it
perturbs entries and re-runs the loss, which is why it is trustworthy.
""")

# ------------------------------------------------------------------ 8
ex.add(
    title="Fused, stable softmax cross-entropy", difficulty=2,
    prompt_md=r"""
Implement `log_softmax(Z)` for an $(N, K)$ array of logits using the log-sum-exp trick (subtract the row maximum before
exponentiating), and `softmax_ce(Z, labels)` returning the **mean** cross-entropy over rows,
$\mathcal L = -\frac1N\sum_n \log p_{n, y_n}$, together with its gradient $\partial\mathcal L/\partial Z = (P - Y)/N$ where $Y$ is the one-hot
matrix of `labels`. Your function must survive logits like 100 and -100 without `inf` or `nan`.
""",
    starter=STARTER_8, solution=SOL_8,
    check=rf"""
Z8 = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 0.5], [100.0, 0.0, -100.0], [-3.0, 4.0, 1.0]])
lab8 = np.array([0, 2, 2, 1])
try:
    lp = log_softmax(Z8)
    loss8, grad8 = softmax_ce(Z8, lab8)
except Exception:
    lp = loss8 = grad8 = None
check('log-softmax rows exponentiate to probabilities summing to 1', lp is not None and np.allclose(np.exp(lp).sum(axis=1), 1.0))
check('no inf/nan for logits of +-100', lp is not None and np.all(np.isfinite(lp)) and loss8 is not None and np.isfinite(loss8))
check_close('uniform row has log-prob -log 3', (lp[1] if lp is not None else None), np.full(3, -np.log(3)))
def _num_grad(Z, h=1e-6):
    g = np.zeros_like(Z)
    for idx in np.ndindex(Z.shape):
        p = Z.copy(); m = Z.copy(); p[idx] += h; m[idx] -= h
        g[idx] = (softmax_ce(p, lab8)[0] - softmax_ce(m, lab8)[0]) / (2 * h)
    return g
try:
    ng = _num_grad(Z8)
except Exception:
    ng = None
check_close('gradient (P - Y)/N matches central differences', grad8, ng, tol=1e-6)
check('gradient rows sum to zero', grad8 is not None and np.allclose(grad8.sum(axis=1), 0.0))
check_hash('hidden: mean loss on the test logits', loss8, '{HASH_8}')
""",
    explanation_md=r"""
Subtracting the row maximum $m$ leaves $\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}$ unchanged mathematically but makes
every exponent $\le 0$, so nothing overflows and the largest term contributes exactly $e^0 = 1$ to the sum — no
`log(0)` either. The fused gradient $P - Y$ follows from $\partial \log p_k/\partial z_j = \mathbb 1[k=j] - p_j$ and needs no
softmax Jacobian; each row sums to zero because the softmax is invariant to adding a constant to all logits. The mean
divides both the loss and its gradient by $N$. Frameworks (`nn.CrossEntropyLoss`) implement exactly this fused form for
stability, speed and memory.
""")

# ------------------------------------------------------------------ 9
ex.add(
    title="A reusable gradient checker", difficulty=2,
    prompt_md=r"""
Write `grad_check(loss_fn, grad_fn, theta, n_check=None, h=1e-5, rng=None)` where `loss_fn(theta)` returns a float and
`grad_fn(theta)` returns an array shaped like `theta`. Estimate the gradient entries by central differences (all entries,
or `n_check` randomly chosen ones using `rng.choice(theta.size, size=n_check, replace=False)`), and return the tuple
`(relative_error, worst_index)` with

$$\text{relative error} = \frac{\|g - \hat g\|_2}{\|g\|_2 + \|\hat g\|_2}, \qquad \text{worst index} = \arg\max_i |g_i - \hat g_i|$$

(the index into the flattened `theta`). The checker is used in the next two exercises, so make it robust to `theta` of
any shape.
""",
    starter=STARTER_9, solution=SOL_9,
    check=r"""
quad = lambda th: float(th @ th + np.sin(th[0]))
quad_grad = lambda th: 2 * th + np.array([np.cos(th[0]), 0.0, 0.0])
quad_grad_bad = lambda th: 2 * th + np.array([np.cos(th[0]), 0.0, 0.5])
th0 = np.array([0.3, -1.2, 2.0])
try:
    r_good, w_good = grad_check(quad, quad_grad, th0)
    r_bad, w_bad = grad_check(quad, quad_grad_bad, th0)
except Exception:
    r_good = r_bad = w_good = w_bad = None
check('correct gradient: relative error < 1e-7', r_good is not None and r_good < 1e-7)
check('wrong gradient: relative error > 1e-2', r_bad is not None and r_bad > 1e-2)
check('worst index points at the wrong entry (2)', w_bad == 2)
M = rng.normal(size=(3, 3)); M = M + M.T
quad2 = lambda th: float(0.5 * th.ravel() @ M @ th.ravel())
quad2_grad = lambda th: (M @ th.ravel()).reshape(th.shape)
try:
    r_sub, _ = grad_check(quad2, quad2_grad, np.ones((3, 1)), n_check=2, rng=np.random.default_rng(3))
except Exception:
    r_sub = None
check('2-D theta and a random subset of entries', r_sub is not None and r_sub < 1e-7)
""",
    explanation_md=r"""
Perturb one entry at a time with $\pm h$, re-evaluate the loss, divide by $2h$. The *relative* error is what makes the
number interpretable: gradients of a mean-reduced loss are tiny, and an absolute difference of $10^{-6}$ may be
either perfect or catastrophically wrong depending on scale. Dividing by the sum of the two norms (not just one) avoids
blow-ups when the analytic gradient is wrongly zero. Checking a random subset keeps the cost at $2\,n_{\text{check}}$ loss
evaluations even for large models; the worst index tells you *which* parameter — hence which layer or rule — to look at.
""")

# ------------------------------------------------------------------ 10 (conceptual)
ex.add_conceptual(
    title="Forward or reverse mode?", difficulty=3,
    prompt_md=r"""
For each situation, say whether you would use forward mode or reverse mode, roughly how many sweeps it costs, and why:

1. The gradient of a scalar training loss for a network with 10 million parameters.
2. The full Jacobian of a function $f: \mathbb R^{3} \to \mathbb R^{5000}$ (3 design parameters, 5000 simulated sensor readings).
3. A Hessian-vector product $Hv$ for the loss in (1).
4. A colleague proposes estimating the gradient in (1) with central differences on a GPU "because it parallelises well". What is wrong with that?
""",
    answer_md=r"""
1. Reverse mode: one output, so a single backward sweep (a small constant times one forward pass) gives all $10^7$ partial
   derivatives. Forward mode would need $10^7$ sweeps.
2. Forward mode: three inputs, so three sweeps (one per input, each seeding a different tangent) give the entire
   $5000\times3$ Jacobian. Reverse mode would need 5000 sweeps, one per output. Dual numbers or `jax.jvp` are the tools.
3. Forward-over-reverse: compute the gradient $\nabla L$ with one reverse sweep, then take a forward-mode (directional)
   derivative of that gradient in direction $v$. Cost is about two to four forward-pass equivalents, independent of the
   parameter count, and the $10^7\times10^7$ Hessian is never formed (Pearlmutter's trick).
4. Parallelism does not change the operation count: $2\times10^7$ forward passes per gradient versus roughly three for
   backpropagation, a factor of about $10^7$ more work per step. The estimate is also only approximate — truncation
   and round-off errors, severe in float32 — while reverse mode is exact to rounding. Finite differences are for
   *checking* a few entries, not for training.
""")

# ------------------------------------------------------------------ 11 (planted bug)
ex.add(
    title="Diagnose a planted bug", difficulty=3,
    prompt_md=r"""
The function below computes the loss and gradients of a one-layer $\tanh$ regression model,
$A = \tanh(X W^\top + b)$, $\mathcal L = \tfrac12\,\mathrm{mean}\big((A - Y)^2\big)$. One line of its backward pass is wrong.

```python
def buggy_forward_backward(W, b, X, Y):
    Z = X @ W.T + b
    A = np.tanh(Z)
    loss = 0.5 * np.mean((A - Y) ** 2)
    G = (A - Y) / A.size
    dZ = G * (1.0 - Z ** 2)
    dW = dZ.T @ X
    db = dZ.sum(axis=0)
    return loss, dW, db
```

1. Use your `grad_check` from Exercise 9 on `buggy_forward_backward` (flatten `W` and `b` into one parameter vector) at
   two scales of `W` — entries of size about 0.1 and about 2 — and note how the relative error depends on the scale.
2. Set `bug_location` to one of `"forward pass"`, `"loss"`, `"upstream gradient G"`, `"tanh derivative"`, `"dW"`, `"db"`.
3. Write `fixed_forward_backward` with the repair. The checks run your gradient checker on it.
""",
    starter=BUGGY_LAYER + STARTER_11, solution=BUGGY_LAYER + SOL_11,
    check=rf"""
r11 = np.random.default_rng(11)
X11 = r11.normal(size=(6, 3)); Y11 = r11.uniform(-0.9, 0.9, size=(6, 2))
W11 = r11.normal(size=(2, 3)); b11 = r11.normal(size=2) * 0.1
theta11 = np.concatenate([W11.ravel(), b11])
def _unpack(th):
    return th[:6].reshape(2, 3), th[6:]
try:
    loss_fixed = lambda th: fixed_forward_backward(*_unpack(th), X11, Y11)[0]
    grad_fixed = lambda th: np.concatenate([fixed_forward_backward(*_unpack(th), X11, Y11)[1].ravel(),
                                            fixed_forward_backward(*_unpack(th), X11, Y11)[2]])
    r_fixed, _ = grad_check(loss_fixed, grad_fixed, theta11)
    same_loss = abs(loss_fixed(theta11) - buggy_forward_backward(W11, b11, X11, Y11)[0]) < 1e-12
    grad_buggy = lambda th: np.concatenate([buggy_forward_backward(*_unpack(th), X11, Y11)[1].ravel(),
                                            buggy_forward_backward(*_unpack(th), X11, Y11)[2]])
    r_buggy, _ = grad_check(loss_fixed, grad_buggy, theta11)
except Exception:
    r_fixed = r_buggy = None; same_loss = False
check('the buggy version fails the gradient check (relative error > 1e-2)', r_buggy is not None and r_buggy > 1e-2)
check('the fixed version passes the gradient check (relative error < 1e-7)', r_fixed is not None and r_fixed < 1e-7)
check('the fix leaves the forward pass / loss unchanged', bool(same_loss))
check_hash('bug location named correctly', bug_location, '{HASH_11}')
""",
    explanation_md=r"""
The line `dZ = G * (1.0 - Z ** 2)` uses the *input* of $\tanh$ where the derivative needs the *output*:
$\tanh'(z) = 1 - \tanh(z)^2 = 1 - A^2$. Near $z = 0$ the two agree to second order, so with small weights the gradient
check reports a mild error of order $10^{-3}$–$10^{-2}$; with weights of size 2 the pre-activations are large,
$1 - z^2$ even turns negative, and the relative error approaches 1. This scale dependence is the diagnostic
signature of a wrong activation derivative, and it is why a gradient check at one tiny initialisation is not enough.
The forward pass and the loss are untouched by the fix, which the third check confirms.
""")

# ------------------------------------------------------------------ 12 (project)
ex.add_project(
    title="Mini-project: train a network with your own engine", difficulty=3,
    prompt_md=r"""
Using the `Value` class you completed in Exercise 5 (it is still defined in this notebook), build and train a small
neural network on the two-moons dataset:

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.15, random_state=0)
```

1. Write `Neuron`, `Layer` and `MLP` classes made of `Value`s (ReLU or tanh hidden units, a linear output logit), with a
   `parameters()` method. A 2–16–1 architecture (65 parameters) is enough.
2. Implement a numerically stable binary cross-entropy on the logit, either as a fused `Value` operation with gradient
   $\sigma(s) - y$ or by composing `Value` operations in the log-sum-exp form.
3. Before training, gradient-check a handful of parameters with central differences (relative error below $10^{-6}$).
4. Train with minibatch SGD or Adam for a few hundred steps, remembering to zero the gradients before each backward
   pass. Keep the run under about 30 seconds: a 2–16–1 network, minibatches of 32 and 300 steps are plenty.
5. Plot the training loss curve and the decision boundary (evaluate the model on a grid and use `plt.contourf`), and
   report the final training accuracy. Aim for **above 95 percent**.
""",
    rubric_md=r"""
- Your network is built only from `Value` objects and your own code (no NumPy autodiff tricks, no frameworks).
- A gradient check on at least 5 parameters passes with relative error below $10^{-6}$ before training starts.
- Gradients are zeroed before every backward pass, and the loss is averaged over the minibatch.
- Training accuracy exceeds 95 percent and the loss curve decreases (noisy minibatch loss is fine).
- The decision boundary plot shows a curved boundary separating the two moons, with axis labels and a title.
- You can explain why the fused `bce_with_logits` is preferable to `-(y*log(p) + (1-y)*log(1-p))` built from `log` and `exp` nodes.
""",
    solution_sketch_md=r"""
Reuse the lesson's `Neuron`/`Layer`/`MLP` (a neuron is a bias plus a sum of `w_i * x_i`, followed by `relu`), add a fused
`bce_with_logits(s, y)` whose forward value is $\max(s,0) + \log(1 + e^{-|s|}) - ys$ and whose closure adds $(\sigma(s) - y)\,g$ to
`s.grad`. The training loop is forward → zero grads → `backward()` → update. Adam converges in fewer steps than SGD, which
matters when every step costs several thousand Python objects; with Adam (learning rate 0.05, minibatch 32) a 2–16–1
network reaches about 97 percent training accuracy in 300 steps. The gradient check at initialisation costs 20 extra
minibatch evaluations and takes a fraction of a second.
""",
    solution_code=PROJECT_SOLUTION,
    starter=r"""
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.15, random_state=0)

# Your mini-project workspace: Neuron / Layer / MLP, a stable BCE, a gradient check, a training loop, two plots.
""",
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
print("exercises:", sum(1 for it in ex.items if it["kind"] == "code"), "coding,",
      sum(1 for it in ex.items if it["kind"] == "concept"), "conceptual,",
      sum(1 for it in ex.items if it["kind"] == "project"), "project")

