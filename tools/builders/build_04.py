"""
build_04.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for
Module 04 — Differential Calculus for ML.

Run from the repository root:
    python3 _authoring/builders/build_04.py
    python3 tools/run_notebooks.py part-2-calculus-and-optimization/04-differential-calculus
"""
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import Notebook, ExerciseSet, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-2-calculus-and-optimization/04-differential-calculus"

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 04 — Differential Calculus for ML")

nb.md(r"""
This notebook is the runnable companion to the concept guide. It builds every object of the
module in plain NumPy — finite differences, activation derivatives, gradients, Jacobians,
Hessians, Taylor models, the matrix-calculus identities and the gradients of ML losses — and
checks each analytic formula against a numerical one.

Read the [concept guide](README.md) first, or keep it open alongside: each section below
mirrors a numbered section there. Run the cells top to bottom; the "Try it yourself"
prompts are invitations to break things and see what happens.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=5, suppress=True)
""")

# ----------------------------------------------------------------------------- 1
nb.md(r"""
## 1. The derivative and its numerical approximation

The derivative is the limit of a difference quotient,

$$f'(x) = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h},$$

and a computer approximates it with a *finite* $h$. Two schemes:

| scheme | formula | truncation error |
|---|---|---|
| forward | $\dfrac{f(x+h)-f(x)}{h}$ | $O(h)$ |
| central | $\dfrac{f(x+h)-f(x-h)}{2h}$ | $O(h^2)$ |

Truncation error shrinks with $h$, but *round-off* error grows like $\varepsilon|f|/h$
(with machine epsilon $\varepsilon \approx 2.2\times 10^{-16}$), because two nearly equal
numbers are subtracted and the difference is divided by something tiny. Let us see both effects.
""")

nb.code(r"""
def fwd_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Hand example from the guide: f(x) = x^3 at x = 2, exact derivative 12
f = lambda x: x**3
print("forward, h=0.1 :", fwd_diff(f, 2.0, 0.1), "   error", fwd_diff(f, 2.0, 0.1) - 12)
print("central, h=0.1 :", central_diff(f, 2.0, 0.1), "   error", central_diff(f, 2.0, 0.1) - 12)

# Now sweep h over 14 decades for f(x) = exp(x) at x = 1 (exact derivative e)
g, x0, exact = np.exp, 1.0, np.e
hs = np.logspace(-14, -1, 60)
err_f = np.abs(fwd_diff(g, x0, hs) - exact)
err_c = np.abs(central_diff(g, x0, hs) - exact)

plt.loglog(hs, err_f, "o-", ms=3, label="forward difference")
plt.loglog(hs, err_c, "s-", ms=3, label="central difference")
plt.loglog(hs, hs * np.e / 2, "--", lw=1, color="gray", label=r"$h\,e/2$ (truncation, forward)")
plt.loglog(hs, 2.2e-16 * np.e / hs, ":", lw=1, color="black", label=r"$\varepsilon e / h$ (round-off)")
plt.xlabel("step size h"); plt.ylabel("absolute error of derivative estimate")
plt.title("Finite-difference error of d/dx exp(x) at x = 1")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()

i_f, i_c = np.argmin(err_f), np.argmin(err_c)
print(f"best forward : h = {hs[i_f]:.1e}, error {err_f[i_f]:.1e}")
print(f"best central : h = {hs[i_c]:.1e}, error {err_c[i_c]:.1e}")
""")

nb.md(r"""
Notice the V shape. On the right, both curves are straight lines: slope 1 for the forward
scheme (error $\propto h$) and slope 2 for the central scheme (error $\propto h^2$), exactly
the truncation orders. On the left, both ride the dotted round-off line $\varepsilon e/h$, which
*rises* as $h$ shrinks. The best step is where the two meet: about $10^{-8}$ for forward
(error $\sim 10^{-8}$) and about $10^{-5}$ to $10^{-6}$ for central (error $\sim 10^{-10}$).
Below $h \approx 10^{-12}$ the estimate is mostly noise — "smaller is safer" is false.

For the hand example, the forward error $0.61$ matches $\tfrac{h}{2}f''(2) = 0.05\cdot 12 = 0.6$,
and the central error $0.01$ matches $\tfrac{h^2}{6}f'''(2) = \tfrac{0.01}{6}\cdot 6 = 0.01$.

**Try it yourself:** replace `np.exp` by `np.sin` (exact derivative `np.cos(1.0)`), or cast
inputs to `np.float32` — where does the V's bottom move, and why?
""")

# ----------------------------------------------------------------------------- 2
nb.md(r"""
## 2. The rules you actually use — activation functions and their derivatives

The chain rule says that composed functions multiply their slopes. Applied to the sigmoid
$\sigma(z) = 1/(1+e^{-z})$ it gives $\sigma'(z) = \sigma(z)(1-\sigma(z))$; for
$\tanh$ it gives $1 - \tanh^2 z$; $\mathrm{softplus}(z) = \ln(1+e^z)$ has derivative
$\sigma(z)$; and $\mathrm{ReLU}(z) = \max(0,z)$ has derivative $\mathbb{1}[z>0]$ (with any
value in $[0,1]$ acceptable as a subgradient at $z = 0$).

Below we implement each derivative *from the formula* and confirm it against a central
difference — the habit this module wants you to form.
""")

nb.code(r"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

activations = {
    "sigmoid":  (sigmoid,                         lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "tanh":     (np.tanh,                         lambda z: 1 - np.tanh(z)**2),
    "softplus": (lambda z: np.logaddexp(0.0, z),  sigmoid),
    "ReLU":     (lambda z: np.maximum(0.0, z),    lambda z: (z > 0).astype(float)),
}

z = np.linspace(-6, 6, 601)
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, (name, (act, dact)) in zip(axes.ravel(), activations.items()):
    ax.plot(z, act(z), label=name)
    ax.plot(z, dact(z), "--", label=name + "'")
    ax.set_title(name + " and its derivative"); ax.set_xlabel("z"); ax.set_ylabel("value")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); plt.show()

# analytic vs numerical derivative, away from the ReLU kink
zz = np.linspace(-4, 4, 17) + 0.013
for name, (act, dact) in activations.items():
    num = central_diff(act, zz, 1e-6)
    print(f"{name:9s} max |analytic - numerical| = {np.max(np.abs(dact(zz) - num)):.2e}")

print("\nmax of sigma'(z) =", np.max(activations['sigmoid'][1](z)), "  (at z = 0: exactly 1/4)")
print("sigma'(2) =", activations['sigmoid'][1](2.0), "  sigma'(5) =", activations['sigmoid'][1](5.0))
""")

nb.md(r"""
Every analytic derivative agrees with the numerical one to about $10^{-10}$. Two things to
notice in the plots: the sigmoid's derivative never exceeds $1/4$ and is already $0.105$ at
$z = 2$ — this *saturation* is the origin of vanishing gradients; and ReLU's derivative is
exactly $1$ wherever the unit is active, so gradients pass through unchanged.

A one-neuron network makes the chain rule concrete: $z = wx + b$, $p = \sigma(z)$,
$L = (p - y)^2$. Then $\dfrac{dL}{dw} = \dfrac{dL}{dp}\dfrac{dp}{dz}\dfrac{dz}{dw} = 2(p-y)\cdot p(1-p)\cdot x$.
""")

nb.code(r"""
w, b, x, y = 0.5, 0.0, 2.0, 1.0
z = w * x + b
p = sigmoid(z)
L = (p - y)**2
dL_dw = 2 * (p - y) * p * (1 - p) * x
print(f"forward : z = {z}, p = {p:.4f}, L = {L:.4f}")
print(f"backward: dL/dw = 2(p-y) * p(1-p) * x = {dL_dw:.4f}")
print("numerical check:", central_diff(lambda w_: (sigmoid(w_ * x + b) - y)**2, w, 1e-6))

# why sigmoid networks vanish: ten layers, each contributing at most 1/4
print("\nten sigmoid layers, best case gradient factor:", 0.25**10)
""")

# ----------------------------------------------------------------------------- 3
nb.md(r"""
## 3. Partial derivatives and the gradient

For $f:\mathbb{R}^n\to\mathbb{R}$, the gradient $\nabla f$ collects the partial derivatives.
The directional derivative along a unit vector $\mathbf{u}$ is $D_{\mathbf{u}} f = \nabla f^\top \mathbf{u}$,
and by Cauchy–Schwarz it is largest when $\mathbf{u}$ points along $\nabla f$ — the gradient is
the **direction of steepest ascent**, with slope $\lVert\nabla f\rVert$, and it is **perpendicular
to level sets**.

We use the guide's example $f(x,y) = x^2 + 3xy$ with $\nabla f = (2x + 3y,\; 3x)$.
""")

nb.code(r"""
def f2(v):
    x, y = v
    return x**2 + 3 * x * y

def grad_f2(v):
    x, y = v
    return np.array([2 * x + 3 * y, 3 * x])

def grad_fd(f, x, h=1e-6):
    # central-difference gradient for any f: R^n -> R  (built from scratch; reused below)
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g

p0 = np.array([1.0, 2.0])
print("analytic gradient at (1,2):", grad_f2(p0), "  numerical:", grad_fd(f2, p0))
print("norm of gradient:", np.linalg.norm(grad_f2(p0)))

# contour plot with gradient arrows
xs = np.linspace(-3, 3, 200); ys = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xs, ys)
Z = X**2 + 3 * X * Y
plt.figure(figsize=(6.5, 5.5))
cs = plt.contour(X, Y, Z, levels=20, cmap="viridis")
plt.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
qx, qy = np.meshgrid(np.linspace(-2.5, 2.5, 7), np.linspace(-2.5, 2.5, 7))
gx, gy = 2 * qx + 3 * qy, 3 * qx
plt.quiver(qx, qy, gx, gy, color="crimson", angles="xy", scale=120, width=0.004)
plt.plot(*p0, "ko"); plt.annotate("(1, 2)", p0 + 0.1)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.title(r"Level sets of $f = x^2 + 3xy$ with gradient arrows")
plt.show()
""")

nb.code(r"""
# the directional derivative as a function of direction, at (1, 2)
angles = np.linspace(0, 2 * np.pi, 3601)
U = np.stack([np.cos(angles), np.sin(angles)])           # unit vectors
g = grad_f2(p0)
dir_deriv = g @ U                                        # grad^T u for every direction
numeric = np.array([(f2(p0 + 1e-6 * u) - f2(p0 - 1e-6 * u)) / 2e-6 for u in U.T])

best = angles[np.argmax(dir_deriv)]
grad_angle = np.arctan2(g[1], g[0])
plt.plot(np.degrees(angles), dir_deriv, label=r"$\nabla f^\top u$ (analytic)")
plt.plot(np.degrees(angles), numeric, "--", label="finite difference along u")
plt.axhline(np.linalg.norm(g), color="gray", ls=":", label=r"$\|\nabla f\|$")
plt.axhline(-np.linalg.norm(g), color="gray", ls=":")
plt.axvline(np.degrees(grad_angle), color="crimson", lw=1, label="direction of gradient")
plt.xlabel("direction angle (degrees)"); plt.ylabel("directional derivative")
plt.title("Slope in every direction at (1, 2)"); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()

print(f"steepest ascent at {np.degrees(best):.1f} deg; gradient points at {np.degrees(grad_angle):.1f} deg")
print(f"max slope {dir_deriv.max():.4f} vs ||grad|| = {np.linalg.norm(g):.4f}")
perp = np.array([-g[1], g[0]]) / np.linalg.norm(g)
print("slope perpendicular to gradient:", round(float(g @ perp), 12))
""")

nb.md(r"""
In the contour plot every arrow crosses the level curve through its base at a right angle;
where the contours are dense (steep terrain) the arrows are long. The sinusoid in the second
plot *is* the Cauchy–Schwarz inequality: $\nabla f^\top\mathbf{u} = \lVert\nabla f\rVert\cos\phi$,
where $\phi$ is the angle between $\mathbf{u}$ and the gradient. The maximum
$\lVert\nabla f\rVert = \sqrt{73} \approx 8.544$ occurs at $\phi = 0$; perpendicular to the gradient
the slope is $0$ — moving along the level set.

**Try it yourself:** change `f2` to the saddle $x^2 - y^2$ or to the Rosenbrock function
$(1-x)^2 + 100(y-x^2)^2$. Where do the arrows become tiny?
""")

# ----------------------------------------------------------------------------- 4
nb.md(r"""
## 4. Vector-valued functions and the Jacobian

For $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$ the Jacobian is the $m\times n$ matrix
$J_{ij} = \partial f_i/\partial x_j$, and
$\mathbf{f}(\mathbf{x}+\boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + J\boldsymbol{\delta}$
with error $O(\lVert\boldsymbol{\delta}\rVert^2)$. Composition multiplies Jacobians:
$J_{\mathbf{g}\circ\mathbf{f}} = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}))\,J_{\mathbf{f}}(\mathbf{x})$.

Example: polar to Cartesian, $\mathbf{P}(r,\theta) = (r\cos\theta, r\sin\theta)$ with

$$J_{\mathbf{P}} = \begin{pmatrix}\cos\theta & -r\sin\theta\\ \sin\theta & r\cos\theta\end{pmatrix}, \qquad \det J_{\mathbf{P}} = r.$$
""")

nb.code(r"""
def polar(v):
    r, th = v
    return np.array([r * np.cos(th), r * np.sin(th)])

def jac_polar(v):
    r, th = v
    return np.array([[np.cos(th), -r * np.sin(th)],
                     [np.sin(th),  r * np.cos(th)]])

def jacobian_fd(F, x, h=1e-6):
    # numerical Jacobian: one central difference per input coordinate = one column each
    x = np.asarray(x, dtype=float)
    cols = []
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        cols.append((F(x + e) - F(x - e)) / (2 * h))
    return np.stack(cols, axis=1)

v0 = np.array([2.0, np.pi / 6])
print("analytic Jacobian:\n", jac_polar(v0))
print("numerical Jacobian:\n", jacobian_fd(polar, v0))
print("determinant:", np.linalg.det(jac_polar(v0)), " (should equal r = 2)")

# linearisation: predicted vs exact change for a perturbation delta, then shrink delta
delta = np.array([0.1, 0.05])
print("\nJ delta        :", jac_polar(v0) @ delta)
print("exact change   :", polar(v0 + delta) - polar(v0))

scales = np.logspace(-4, 0, 25)
lin_err = [np.linalg.norm(polar(v0 + s * delta) - polar(v0) - jac_polar(v0) @ (s * delta)) for s in scales]
plt.loglog(scales * np.linalg.norm(delta), lin_err, "o-", ms=4, label="linearisation error")
plt.loglog(scales * np.linalg.norm(delta), (scales * np.linalg.norm(delta))**2, "--", color="gray", label=r"$\|\delta\|^2$ reference")
plt.xlabel(r"$\|\delta\|$"); plt.ylabel(r"$\|P(x+\delta) - P(x) - J\delta\|$")
plt.title("Error of the linear approximation is quadratic in the step"); plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

nb.code(r"""
# chain rule in matrix form: compose polar with a linear map g(p) = A p + c
A = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])     # g: R^2 -> R^3
c = np.array([0.5, 0.0, -1.0])
g_lin = lambda p: A @ p + c
h_comp = lambda v: g_lin(polar(v))                      # h = g o P : R^2 -> R^3

J_chain = A @ jac_polar(v0)                             # J_g (constant A) times J_P
J_num = jacobian_fd(h_comp, v0)
print("J_g J_P (3x2):\n", J_chain)
print("max |chain rule - numerical| =", np.max(np.abs(J_chain - J_num)))

# backprop preview: a vector-Jacobian product never forms J explicitly
gbar = np.array([1.0, -2.0, 0.5])                       # dL/dh, say
print("VJP  J^T gbar          :", J_chain.T @ gbar)
print("same via two VJPs      :", jac_polar(v0).T @ (A.T @ gbar))
""")

nb.md(r"""
The analytic and numerical Jacobians agree to $10^{-9}$, and the linearisation error falls on
a line of slope 2 in the log-log plot — halve the step, quarter the error. The composed map's
Jacobian is the *product* $J_{\mathbf{g}}J_{\mathbf{P}}$ (shapes $3\times 2 = (3\times 2)(2\times 2)$),
and the backward pass of a network computes $J^\top\bar{\mathbf{g}}$ one factor at a time,
right to left, which is exactly what the last two lines show. Module 06 builds this machinery.

**Try it yourself:** evaluate `jac_polar` at `r = 0`. What happens to the determinant, and what does
that say about the inverse map (Cartesian to polar) at the origin?
""")

# ----------------------------------------------------------------------------- 5
nb.md(r"""
## 5. The Hessian and second-order information

The Hessian $H_{ij} = \partial^2 f/\partial x_i\partial x_j$ is symmetric (Schwarz) and its
eigenvalues are the curvatures along its eigenvectors. At a critical point ($\nabla f = 0$):
all eigenvalues $> 0$ ⇒ minimum, all $< 0$ ⇒ maximum, mixed signs ⇒ saddle.

We use $f(x,y) = x^4 - 2x^2 + y^2$ (a "double well"): critical points are $(\pm 1, 0)$ and
$(0, 0)$, and we also look at $g(x,y) = -(x^2+y^2)$ and the guide's $x^2 + 3xy + 5y^2$.
""")

nb.code(r"""
def hessian_fd(f, x, h=1e-4):
    # numerical Hessian via second central differences (h larger than for first derivatives:
    # the round-off error now scales like eps/h^2)
    x = np.asarray(x, dtype=float); n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H

def classify(H, tol=1e-8):
    lam = np.linalg.eigvalsh(H)
    if np.all(lam > tol):  return "local minimum", lam
    if np.all(lam < -tol): return "local maximum", lam
    if np.any(np.abs(lam) <= tol): return "degenerate (inconclusive)", lam
    return "saddle point", lam

dwell = lambda v: v[0]**4 - 2 * v[0]**2 + v[1]**2
hess_dwell = lambda v: np.array([[12 * v[0]**2 - 4, 0.0], [0.0, 2.0]])

for cp in [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 0.0])]:
    H_an, H_num = hess_dwell(cp), hessian_fd(dwell, cp)
    kind, lam = classify(H_an)
    print(f"critical point {cp}: gradient {grad_fd(dwell, cp).round(6)}, eigenvalues {lam}, -> {kind}"
          f"   (numerical Hessian error {np.max(np.abs(H_an - H_num)):.1e})")

print()
for name, H in {"x^2+3xy+5y^2": np.array([[2., 3.], [3., 10.]]),
                "x^2+3xy":      np.array([[2., 3.], [3., 0.]]),
                "-(x^2+y^2)":   np.array([[-2., 0.], [0., -2.]])}.items():
    kind, lam = classify(H)
    print(f"{name:14s} eigenvalues {lam.round(4)} -> {kind}")
""")

nb.code(r"""
xs = np.linspace(-1.8, 1.8, 200); ys = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(xs, ys)
Z = X**4 - 2 * X**2 + Y**2

fig = plt.figure(figsize=(11, 4.2))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f"); ax.set_title("Double well: two minima and a saddle")
ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contour(X, Y, Z, levels=25, cmap="viridis"); ax2.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
ax2.plot([1, -1], [0, 0], "go", label="minima (eigenvalues 8, 2)")
ax2.plot([0], [0], "r^", label="saddle (eigenvalues -4, 2)")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_title("Level sets and critical points"); ax2.legend(fontsize=8)
fig.tight_layout(); plt.show()

# saddles dominate in high dimensions: random symmetric Hessians at "critical points"
trials = 20000
for n in (1, 2, 3, 4, 6, 8):
    M = rng.standard_normal((trials, n, n)); M = (M + np.swapaxes(M, 1, 2)) / 2
    lam = np.linalg.eigvalsh(M)
    frac_min = np.mean(np.all(lam > 0, axis=1))
    print(f"n = {n}: fraction of random symmetric matrices that are positive definite = {frac_min:.4f}"
          f"   (coin-flip estimate 2^-n = {2.0**-n:.4f})")
""")

nb.md(r"""
At $(\pm 1, 0)$ both eigenvalues are positive (bowl); at the origin the Hessian has
eigenvalues $-4$ and $2$ — curving down along $x$, up along $y$ — a saddle, which is why the
level sets there look like a cross. The last experiment shows how quickly "all curvatures
positive" becomes rare as dimension grows — for Gaussian random matrices even faster than the
coin-flip estimate $2^{-n}$, because eigenvalues of a random symmetric matrix repel each other
and spread symmetrically about zero. In a high-dimensional landscape, a point with zero gradient
is almost never a minimum. (A side note on `hessian_fd`: it uses `h = 1e-4`, much larger than for
first derivatives, because the second-difference formula divides by $h^2$ and its round-off term
is $\varepsilon/h^2$.)
""")

# ----------------------------------------------------------------------------- 6
nb.md(r"""
## 6. Taylor expansion

$$f(a+h) = f(a) + f'(a)h + \frac{f''(a)}{2}h^2 + \dots + \frac{f^{(k)}(a)}{k!}h^k + O(h^{k+1}).$$

For $e^x$ about $0$ every derivative is $1$, so $T_k(x) = \sum_{j=0}^{k} x^j/j!$. We overlay the
polynomials on the function and then measure how the error scales with $|x|$.
""")

nb.code(r"""
from math import factorial

def taylor_exp(x, k):
    return sum(x**j / factorial(j) for j in range(k + 1))

xs = np.linspace(-2.5, 2.5, 400)
plt.plot(xs, np.exp(xs), "k", lw=2.5, label="exp(x)")
for k in (0, 1, 2, 3, 5):
    plt.plot(xs, taylor_exp(xs, k), "--", label=f"degree {k}")
plt.ylim(-1, 10); plt.xlabel("x"); plt.ylabel("value"); plt.title("Taylor polynomials of exp(x) about 0")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()

print("exp(0.5) =", np.exp(0.5))
for k in (1, 2, 3, 4):
    print(f"  degree {k}: T_k = {taylor_exp(0.5, k):.6f}, error = {np.exp(0.5) - taylor_exp(0.5, k):.2e}")

# error scaling: for degree k, error ~ C h^(k+1)
hs = np.logspace(-2, 0, 30)
for k in (1, 2, 3):
    err = np.abs(np.exp(hs) - taylor_exp(hs, k))
    slope = np.polyfit(np.log(hs), np.log(err), 1)[0]
    plt.loglog(hs, err, "o-", ms=3, label=f"degree {k}: fitted slope {slope:.2f}")
plt.xlabel("h"); plt.ylabel("|exp(h) - T_k(h)|"); plt.title("Taylor error scales like $h^{k+1}$")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()
""")

nb.md(r"""
Each extra term buys roughly two more correct digits at $x = 0.5$, and the log-log slopes
come out as $2, 3, 4$ for degrees $1, 2, 3$ — the remainder is $O(h^{k+1})$ as promised.
The polynomials are excellent near $0$ and useless far away: a Taylor model is *local*.

The multivariate version, $f(\mathbf{x}+\boldsymbol{\delta}) \approx f + \nabla f^\top\boldsymbol{\delta} + \tfrac12\boldsymbol{\delta}^\top H\boldsymbol{\delta}$,
is the model gradient descent and Newton's method minimise. One consequence: on a quadratic
with Hessian eigenvalues $\lambda_i$, a gradient step multiplies the error along each
eigendirection by $(1 - \eta\lambda_i)$, so the method is stable only for $\eta < 2/\lambda_{\max}$.
""")

nb.code(r"""
# gradient descent on the quadratic with Hessian [[2,3],[3,10]] (eigenvalues 1 and 11)
H = np.array([[2.0, 3.0], [3.0, 10.0]])
quad = lambda v: 0.5 * v @ H @ v
x_start = np.array([3.0, -2.0])

def run_gd(eta, steps=40):
    x = x_start.copy(); hist = [quad(x)]
    for _ in range(steps):
        x = x - eta * (H @ x)             # gradient of 0.5 x^T H x is H x (H symmetric)
        hist.append(quad(x))
    return np.array(hist)

for eta in (0.05, 1/6, 0.17, 0.19):
    hist = run_gd(eta)
    plt.semilogy(hist, label=f"eta = {eta:.3f}")
plt.axhline(1e-12, color="gray", lw=0.5)
plt.xlabel("gradient step"); plt.ylabel("loss (log scale)"); plt.title(r"Stable iff $\eta < 2/\lambda_{\max} = 2/11 \approx 0.182$")
plt.legend(); plt.grid(alpha=0.3); plt.show()
print("predicted contraction per step at eta = 1/6:", 1 - (1/6) * 1, "(flat dir) and", abs(1 - (1/6) * 11), "(sharp dir)")
""")

nb.md(r"""
With $\eta = 0.19 > 2/11$ the loss explodes (the sharp direction is multiplied by $-1.09$ each
step); with $\eta = 1/6$ both directions contract by $5/6$ per step. The learning rate is
dictated by the *largest* curvature while progress along the *smallest* curvature is what
takes long — the conditioning story of Module 05.

**Try it yourself:** replace `H` by `[[1, 0], [0, 100]]` (condition number 100). How many steps
does the flat direction now need at the largest stable learning rate?
""")

# ----------------------------------------------------------------------------- 7
nb.md(r"""
## 7. Matrix calculus identities, verified numerically

| expression | gradient |
|---|---|
| $\mathbf{a}^\top\mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top A\mathbf{x}$ | $(A + A^\top)\mathbf{x}$ |
| $\lVert B\mathbf{x}-\mathbf{b}\rVert^2$ | $2B^\top(B\mathbf{x}-\mathbf{b})$ |
| $\operatorname{tr}(XA)$ | $A^\top$ |
| $\log\det X$ | $X^{-\top}$ |

Random matrices plus `grad_fd` from Section 3 give an independent check of each row. Note
that `A` below is deliberately *not* symmetric, so $2A\mathbf{x}$ would be wrong.
""")

nb.code(r"""
n, m = 4, 6
a = rng.standard_normal(n)
A = rng.standard_normal((n, n))                 # NOT symmetric on purpose
B = rng.standard_normal((m, n))
b = rng.standard_normal(m)
x = rng.standard_normal(n)

checks = {
    "grad a^T x = a":                     (grad_fd(lambda v: a @ v, x),                     a),
    "grad x^T A x = (A + A^T) x":         (grad_fd(lambda v: v @ A @ v, x),                 (A + A.T) @ x),
    "grad x^T A x != 2 A x (asym. A)":    (grad_fd(lambda v: v @ A @ v, x),                 2 * A @ x),
    "grad ||Bx - b||^2 = 2 B^T (Bx - b)": (grad_fd(lambda v: np.sum((B @ v - b)**2), x),    2 * B.T @ (B @ x - b)),
}
for name, (num, ana) in checks.items():
    print(f"{name:36s} max |numerical - formula| = {np.max(np.abs(num - ana)):.2e}")

# matrix-valued inputs: flatten X to use the same finite-difference helper
X = rng.standard_normal((3, 3)) + 4 * np.eye(3)          # well-conditioned, positive determinant
C = rng.standard_normal((3, 3))
g_tr = grad_fd(lambda v: np.trace(v.reshape(3, 3) @ C), X.ravel()).reshape(3, 3)
g_ld = grad_fd(lambda v: np.log(np.linalg.det(v.reshape(3, 3))), X.ravel()).reshape(3, 3)
print(f"{'grad tr(XC) = C^T':36s} max error = {np.max(np.abs(g_tr - C.T)):.2e}")
print(f"{'grad log det X = X^-T':36s} max error = {np.max(np.abs(g_ld - np.linalg.inv(X).T)):.2e}")
""")

nb.code(r"""
# least squares: gradient = 0  <=>  normal equations. The guide's 3-point line fit:
B3 = np.array([[1., 0.], [1., 1.], [1., 2.]]); b3 = np.array([1., 2., 2.])
loss = lambda v: np.sum((B3 @ v - b3)**2)
grad_ls = lambda v: 2 * B3.T @ (B3 @ v - b3)

x0 = np.array([0.0, 1.0])
print("loss at x0 =", loss(x0), "  gradient =", grad_ls(x0), "  numerical:", grad_fd(loss, x0))

x_star = np.linalg.solve(B3.T @ B3, B3.T @ b3)
print("normal-equation solution x* =", x_star, "  loss* =", loss(x_star))
print("gradient at x* =", grad_ls(x_star).round(12), "  (zero: first-order optimality)")
print("residual at x* =", (B3 @ x_star - b3).round(6), "  B^T residual =", (B3.T @ (B3 @ x_star - b3)).round(12))
print("np.linalg.lstsq agrees:", np.allclose(np.linalg.lstsq(B3, b3, rcond=None)[0], x_star))
""")

nb.md(r"""
Every identity matches finite differences to $10^{-9}$ or better, and the row labelled
`!= 2 A x` shows a large error — the $A^\top$ is not optional for a non-symmetric $A$. The
least-squares block is Module 02's normal equations rediscovered as "set the gradient to
zero": at $\mathbf{x}^*$ the residual is orthogonal to the columns of $B$ (that is what
$B^\top\mathbf{r}^* = \mathbf{0}$ says) and `np.linalg.lstsq` returns the same point.
""")

# ----------------------------------------------------------------------------- 8
nb.md(r"""
## 8. Gradients of ML losses: logistic regression by hand

With $z_i = \mathbf{w}^\top\mathbf{x}_i$, $p_i = \sigma(z_i)$ and the binary cross-entropy
$L = -\frac1n\sum_i [y_i\ln p_i + (1-y_i)\ln(1-p_i)]$, the chain rule gives the famously
clean result

$$\frac{\partial L}{\partial z_i} = \frac{p_i - y_i}{n}, \qquad \nabla_{\mathbf{w}} L = \frac1n X^\top(\mathbf{p} - \mathbf{y}).$$

We check it on a synthetic dataset, take a few gradient steps and watch the loss fall and
the predictions become calibrated in aggregate.
""")

nb.code(r"""
# synthetic 2-feature dataset with an intercept column
n = 200
X_raw = rng.standard_normal((n, 2))
w_true = np.array([0.5, 2.0, -1.5])                     # intercept, w1, w2
X = np.column_stack([np.ones(n), X_raw])
p_true = sigmoid(X @ w_true)
y = (rng.random(n) < p_true).astype(float)

def bce_loss(w):
    z = X @ w
    # numerically stable: log(1+exp(-z)) = softplus(-z), and -log(1-p) = softplus(z)
    return np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))

def bce_grad(w):
    p = sigmoid(X @ w)
    return X.T @ (p - y) / n

w0 = np.zeros(3)
print("analytic gradient :", bce_grad(w0))
print("numerical gradient:", grad_fd(bce_loss, w0))
print("relative error    :", np.linalg.norm(bce_grad(w0) - grad_fd(bce_loss, w0)) / np.linalg.norm(bce_grad(w0)))
""")

nb.code(r"""
w, eta = np.zeros(3), 0.5
hist = [bce_loss(w)]
for step in range(300):
    w = w - eta * bce_grad(w)
    hist.append(bce_loss(w))

plt.plot(hist); plt.xlabel("gradient step"); plt.ylabel("binary cross-entropy")
plt.title("Logistic regression trained with the analytic gradient"); plt.grid(alpha=0.3); plt.show()

p_hat = sigmoid(X @ w)
print("learned w        :", w)
print("true  w          :", w_true)
print("mean prediction  :", p_hat.mean(), "   positive rate:", y.mean(), "  (calibration in aggregate)")
print("||gradient|| at end:", np.linalg.norm(bce_grad(w)))
print("accuracy         :", np.mean((p_hat > 0.5) == (y == 1)))
""")

nb.md(r"""
The analytic gradient $X^\top(\mathbf{p}-\mathbf{y})/n$ matches finite differences to about
$10^{-10}$ in relative error, the loss decreases monotonically with a reasonable step size
(logistic loss is convex), and at the end the mean predicted probability equals the empirical
positive rate — the intercept component of "$\nabla L = 0$" reads $\sum_i p_i = \sum_i y_i$.
The learned weights sit near the true ones, with the discrepancy you would expect from 200 noisy
labels.

**Try it yourself:** set `eta = 5.0`. Does the convex loss still decrease at every step? Then
replace `bce_loss` by the squared error `np.mean((sigmoid(X @ w) - y)**2)` and its gradient
`X.T @ ((p - y) * p * (1 - p)) / n` — compare the speed of training and the final calibration.
""")

# ----------------------------------------------------------------------------- summary
nb.md(r"""
## Summary

- A derivative is a sensitivity and the slope of the best linear approximation; a central
  difference estimates it with error $O(h^2)$ until round-off ($\varepsilon/h$) takes over around
  $h \approx 10^{-6}$ — the V-shaped error curve.
- The chain rule multiplies slopes; the activation derivatives $\sigma(1-\sigma)$,
  $1-\tanh^2$ and the ReLU mask are what backprop multiplies by, and $\sigma' \le 1/4$ is why
  deep sigmoid networks vanish.
- The gradient is the direction of steepest ascent (Cauchy–Schwarz) and is perpendicular to
  level sets; directional derivatives are $\nabla f^\top\mathbf{u}$.
- The Jacobian linearises a vector map; Jacobians of composed maps multiply; backprop
  computes $J^\top\bar{\mathbf{g}}$ one layer at a time.
- The Hessian's eigenvalues classify critical points; in high dimensions almost all critical
  points are saddles; $\eta < 2/\lambda_{\max}$ keeps gradient descent stable.
- Taylor models are local: the degree-$k$ error scales as $h^{k+1}$; gradient descent and
  Newton's method minimise first- and second-order Taylor models.
- $\nabla \mathbf{x}^\top A\mathbf{x} = (A+A^\top)\mathbf{x}$ and
  $\nabla\lVert B\mathbf{x}-\mathbf{b}\rVert^2 = 2B^\top(B\mathbf{x}-\mathbf{b})$; setting the latter
  to zero gives the normal equations.
- Logistic loss has gradient $X^\top(\mathbf{p}-\mathbf{y})/n$, which makes the fitted model
  calibrated in aggregate — and every one of these formulas was confirmed with finite differences.

**Next:** open `exercises.ipynb` and implement these ideas yourself, ending with a reusable
gradient checker.
""")

nb.save(MOD / "lesson.ipynb")
print("lesson cells:", len(nb.cells))

# =============================================================================
# EXERCISES / SOLUTIONS
# =============================================================================
SETUP = r"""
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def reference_grad_fd(f, x, h=1e-6):
    # Reference central-difference gradient used by the self-checks below.
    # Exercise 1 asks you to write your own version — try it without peeking.
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g
"""

# namespace in which solution code is executed to compute hidden answers
NS: dict = {}
exec(textwrap.dedent(SETUP), NS)


def run_solution(src: str) -> None:
    exec(textwrap.dedent(src), NS)


def safe_sha(value, decimals=3) -> str:
    """sha_of, but assert the value is not within 1e-5 of a rounding boundary."""
    arr = np.asarray(value, dtype=float).ravel()
    scaled = arr * 10**decimals
    dist = 0.5 - np.abs(scaled - np.round(scaled))        # distance to the x.5 boundary (in last digit)
    assert np.all(dist > 1e-2), f"value too close to rounding boundary: {arr}, {dist}"
    return sha_of(value, decimals)


def fill(template: str, **hashes: str) -> str:
    out = template
    for k, v in hashes.items():
        out = out.replace("__" + k + "__", v)
    return out


ex = ExerciseSet(
    "Module 04 — Exercises",
    intro_md=r"""
These exercises mirror the sections of the [concept guide](README.md): numerical derivatives,
activation derivatives, gradients, Jacobians, Hessians, Taylor models, matrix calculus and the
gradients of ML losses, ending with a reusable gradient checker. Use only NumPy (no autodiff
libraries, no `scipy.misc.derivative`).
""",
    setup_code=SETUP,
)

# ----------------------------------------------------------------------------- Ex 1
SOL1 = r"""
def central_diff(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

def grad_fd(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g
"""
run_solution(SOL1)
ex.add(
    title="Central differences in one and many dimensions", difficulty=1,
    prompt_md=r"""
Implement two numerical-derivative helpers using **central differences**:

1. `central_diff(f, x, h=1e-6)` for a scalar function $f:\mathbb{R}\to\mathbb{R}$, returning
   $\dfrac{f(x+h)-f(x-h)}{2h}$ as a float.
2. `grad_fd(f, x, h=1e-6)` for $f:\mathbb{R}^n\to\mathbb{R}$ taking a 1-D NumPy array `x`,
   returning the gradient as an array of the same shape: perturb one coordinate at a time by
   $\pm h$ (use the standard basis vectors $\mathbf{e}_i$).

Use only NumPy. Do not look at `reference_grad_fd` in the setup cell until your checks pass.
""",
    starter=r"""
def central_diff(f, x, h=1e-6):
    # YOUR CODE HERE
    return None

def grad_fd(f, x, h=1e-6):
    # YOUR CODE HERE
    return None
""",
    solution=SOL1,
    check=r"""
try:
    got = central_diff(lambda t: t**3, 2.0, 0.1)
except Exception:
    got = None
check_close('central difference of x^3 at 2 with h = 0.1 equals 12.01', got, 12.01, tol=1e-9)

try:
    got = central_diff(np.exp, 1.0)
except Exception:
    got = None
check_close('central difference of exp at 1 is e (to 1e-8)', got, np.e, tol=1e-8)

try:
    got = grad_fd(lambda v: v @ v, np.array([1.0, 2.0, 3.0]))
except Exception:
    got = None
check_close('gradient of x.x at (1,2,3) is (2,4,6)', got, np.array([2.0, 4.0, 6.0]), tol=1e-7)

try:
    got = grad_fd(lambda v: np.sum(np.sin(v)) * v[0], np.array([0.3, -1.2, 2.0, 0.7]))
    exp_ = reference_grad_fd(lambda v: np.sum(np.sin(v)) * v[0], np.array([0.3, -1.2, 2.0, 0.7]))
except Exception:
    got, exp_ = None, None
check_close('gradient of a 4-D function matches the reference', got, exp_, tol=1e-7)
check('grad_fd returns an array with the shape of x', got is not None and np.shape(got) == (4,))
""",
    explanation_md=r"""
The central difference cancels the even-order Taylor terms, so its error is $O(h^2)$ rather
than the forward scheme's $O(h)$; for $x^3$ with $h = 0.1$ the error is exactly
$\tfrac{h^2}{6}f'''(2) = 0.01$, giving $12.01$. The gradient version simply applies the scalar
formula coordinate by coordinate along the basis vectors $\mathbf{e}_i$, which costs $2n$
function evaluations. A common wrong turn is to perturb `x` in place (`x[i] += h`) and forget
to undo it, which corrupts later coordinates — building a fresh perturbation vector avoids this.
""")

# ----------------------------------------------------------------------------- Ex 2
SOL2 = r"""
def fd_errors(f, x, exact, hs):
    hs = np.asarray(hs, dtype=float)
    est = (f(x + hs) - f(x - hs)) / (2 * hs)
    return np.abs(est - exact)

def best_step(f, x, exact, hs):
    err = fd_errors(f, x, exact, hs)
    return float(np.asarray(hs)[np.argmin(err)])
"""
run_solution(SOL2)
ex.add(
    title="The step-size trade-off", difficulty=1,
    prompt_md=r"""
Write `fd_errors(f, x, exact, hs)` that returns, for each step size in the 1-D array `hs`, the
absolute error $\left|\dfrac{f(x+h)-f(x-h)}{2h} - \texttt{exact}\right|$ of the central-difference
derivative at the scalar point `x` (vectorise over `hs`; `f` is a NumPy ufunc such as `np.exp`).
Then write `best_step(f, x, exact, hs)` returning the step size (a float) with the smallest error.

Use `hs = np.logspace(-14, -1, 53)` and $f = \exp$ at $x = 1$ (exact derivative $e$) to see the
V-shaped curve from the lesson. Where is the minimum, and how small is the best error?
""",
    starter=r"""
def fd_errors(f, x, exact, hs):
    # YOUR CODE HERE
    return None

def best_step(f, x, exact, hs):
    # YOUR CODE HERE
    return None

hs = np.logspace(-14, -1, 53)
# optional: plt.loglog(hs, fd_errors(np.exp, 1.0, np.e, hs)); plt.show()
""",
    solution=SOL2 + r"""
hs = np.logspace(-14, -1, 53)
errs = fd_errors(np.exp, 1.0, np.e, hs)
plt.loglog(hs, errs, "o-", ms=3)
plt.xlabel("h"); plt.ylabel("absolute error"); plt.title("Central-difference error vs step size (exp at x = 1)")
plt.grid(alpha=0.3); plt.show()
print("best h:", best_step(np.exp, 1.0, np.e, hs), " best error:", errs.min())
""",
    check=r"""
hs = np.logspace(-14, -1, 53)
try:
    errs = np.asarray(fd_errors(np.exp, 1.0, np.e, hs), dtype=float)
    if errs.shape != hs.shape:
        errs = None
except Exception:
    errs = None
check('fd_errors returns one error per step size', errs is not None)
check_close('error at h = 0.1 matches the closed form', None if errs is None else errs[-1],
            abs((np.exp(1.1) - np.exp(0.9)) / 0.2 - np.e), tol=1e-12)
try:
    hb = float(best_step(np.exp, 1.0, np.e, hs))
except Exception:
    hb = None
check('best step size lies between 1e-7 and 1e-4 (round-off vs truncation balance)', hb is not None and 1e-7 <= hb <= 1e-4,
      hint='think about where the two error sources are equal')
check('best error is below 1e-8', errs is not None and float(np.min(errs)) < 1e-8)
check('error at h = 1e-13 is worse than at h = 1e-5 (round-off dominates)',
      errs is not None and errs[np.argmin(np.abs(hs - 1e-13))] > errs[np.argmin(np.abs(hs - 1e-5))])
""",
    explanation_md=r"""
The error has two parts: truncation, $\approx \tfrac{h^2}{6}|f'''(x)|$, which falls as $h$
shrinks, and round-off, $\approx \varepsilon|f(x)|/h$ with $\varepsilon \approx 2.2\times10^{-16}$,
which grows. They balance near $h \approx \varepsilon^{1/3} \approx 6\times 10^{-6}$, where the
error bottoms out around $10^{-10}$. Choosing a "tiny" $h = 10^{-13}$ gives an estimate that is
wrong in the third or fourth significant digit — the most common finite-difference mistake.
""")

# ----------------------------------------------------------------------------- Ex 3
SOL3 = r"""
def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

def d_tanh(z):
    return 1 - np.tanh(z)**2

def d_softplus(z):
    return sigmoid(z)

def d_relu(z):
    return (np.asarray(z) > 0).astype(float)
"""
run_solution(SOL3)
ex.add(
    title="Derivatives of the activation functions", difficulty=1,
    prompt_md=r"""
Implement the derivatives of the four standard activations as NumPy functions of an array `z`:
`d_sigmoid(z)`, `d_tanh(z)`, `d_softplus(z)` (where $\mathrm{softplus}(z) = \ln(1+e^z)$) and
`d_relu(z)` (return `0.0` at $z = 0$, the usual subgradient choice). Derive each formula with the
chain rule first — you should only need `sigmoid` (defined in the setup cell) and `np.tanh`, not
`np.exp` directly.
""",
    starter=r"""
def d_sigmoid(z):
    # YOUR CODE HERE
    return None

def d_tanh(z):
    # YOUR CODE HERE
    return None

def d_softplus(z):
    # YOUR CODE HERE
    return None

def d_relu(z):
    # YOUR CODE HERE
    return None
""",
    solution=SOL3,
    check=r"""
zz = np.array([-3.0, -0.7, 0.2, 1.5, 4.0])
h = 1e-6
pairs = [('sigmoid', d_sigmoid, sigmoid), ('tanh', d_tanh, np.tanh),
         ('softplus', d_softplus, lambda t: np.logaddexp(0.0, t)), ('ReLU', d_relu, lambda t: np.maximum(0.0, t))]
for name, dact, act in pairs:
    try:
        got = dact(zz)
    except Exception:
        got = None
    check_close(f'd_{name} matches a central difference at 5 points', got, (act(zz + h) - act(zz - h)) / (2 * h), tol=1e-6)
try:
    got = d_sigmoid(0.0)
except Exception:
    got = None
check_close("sigma'(0) = 1/4", got, 0.25)
try:
    got = d_relu(np.array([-1.0, 0.0, 2.0]))
except Exception:
    got = None
check_close("ReLU' at (-1, 0, 2) is (0, 0, 1)", got, np.array([0.0, 0.0, 1.0]))
""",
    explanation_md=r"""
Writing $\sigma = (1+e^{-z})^{-1}$ and applying the chain rule gives
$\sigma' = e^{-z}/(1+e^{-z})^2 = \sigma(1-\sigma)$; for $\tanh = (e^z - e^{-z})/(e^z + e^{-z})$ the
quotient rule gives $1 - \tanh^2$; and $\frac{d}{dz}\ln(1+e^z) = e^z/(1+e^z) = \sigma(z)$. The
practical point is that every derivative is a cheap function of the activation *output* that
the forward pass already computed, which is exactly how frameworks implement the backward pass.
ReLU is not differentiable at $0$, but any value in $[0,1]$ is a valid subgradient and the
kink is hit with probability zero.
""")

# ----------------------------------------------------------------------------- Ex 4 (conceptual)
ex.add_conceptual(
    title="Why the gradient is the direction of steepest ascent", difficulty=1,
    prompt_md=r"""
A colleague says: *"The gradient points uphill because each component $\partial f/\partial x_i$ is
positive when $f$ increases in $x_i$."* That is true but does not explain why the *combined*
direction is the steepest one. Give the real argument, and state (a) what the maximal slope is,
(b) in which direction the slope is zero, and (c) what this implies about the angle between the
gradient and the level set through the point.
""",
    answer_md=r"""
For a unit vector $\mathbf{u}$ the slope in direction $\mathbf{u}$ is the directional derivative
$D_{\mathbf{u}}f = \nabla f^\top\mathbf{u} = \lVert\nabla f\rVert\,\lVert\mathbf{u}\rVert\cos\phi = \lVert\nabla f\rVert\cos\phi$,
where $\phi$ is the angle between $\mathbf{u}$ and $\nabla f$. By Cauchy–Schwarz this is at most
$\lVert\nabla f\rVert$, with equality exactly when $\phi = 0$, i.e. $\mathbf{u} = \nabla f/\lVert\nabla f\rVert$.
(a) The maximal slope is therefore $\lVert\nabla f\rVert$, and the most negative slope,
$-\lVert\nabla f\rVert$, is attained along $-\nabla f$ — the direction gradient descent takes.
(b) The slope is zero when $\cos\phi = 0$, i.e. for every $\mathbf{u}$ orthogonal to the gradient.
(c) Moving along the level set $\{f = c\}$ leaves $f$ unchanged, so every tangent direction of
the level set has zero directional derivative and is orthogonal to $\nabla f$: the gradient is
perpendicular to the level set. The colleague's component-wise statement misses that the
*ratio* of the components matters — walking along $(1, 0)$ when $\nabla f = (1, 100)$ is uphill
but almost flat compared with walking along $(1, 100)/\sqrt{10001}$.
""")

# ----------------------------------------------------------------------------- Ex 5
SOL4 = r"""
def f4(v):
    x, y = v
    return x**2 * y + np.sin(x * y) + np.exp(y)

def grad_f4(v):
    x, y = v
    return np.array([2 * x * y + y * np.cos(x * y),
                     x**2 + x * np.cos(x * y) + np.exp(y)])
"""
run_solution(SOL4)
g4a = NS["grad_f4"](np.array([1.0, 2.0]))
g4b = NS["grad_f4"](np.array([1.5, -1.0]))
H4A, H4B = safe_sha(g4a), safe_sha(g4b)
ex.add(
    title="Analytic gradient of a two-variable function", difficulty=2,
    prompt_md=r"""
Let $f(x, y) = x^2 y + \sin(xy) + e^{y}$. The function is given as `f4(v)` below (with `v = (x, y)`).
Work out $\partial f/\partial x$ and $\partial f/\partial y$ by hand (product rule for $x^2 y$,
chain rule for $\sin(xy)$) and implement `grad_f4(v)` returning the gradient as a length-2 array.

The checks compare your gradient at $(1, 2)$ and at $(1.5, -1)$ against hidden values, so a
finite-difference shortcut would pass — but write the analytic formula; that is the skill.
""",
    starter=r"""
def f4(v):
    x, y = v
    return x**2 * y + np.sin(x * y) + np.exp(y)

def grad_f4(v):
    x, y = v
    # YOUR CODE HERE
    return None
""",
    solution=SOL4,
    check=fill(r"""
try:
    got_a = grad_f4(np.array([1.0, 2.0]))
    got_b = grad_f4(np.array([1.5, -1.0]))
except Exception:
    got_a, got_b = None, None
check('gradient has shape (2,)', got_a is not None and np.shape(got_a) == (2,))
check_hash('hidden: gradient at (1, 2)', got_a, '__A__')
check_hash('hidden: gradient at (1.5, -1)', got_b, '__B__')
try:
    got_c = grad_f4(np.array([-0.4, 0.9]))
except Exception:
    got_c = None
check_close('matches a finite-difference gradient at (-0.4, 0.9)', got_c, reference_grad_fd(f4, np.array([-0.4, 0.9])), tol=1e-6)
""", A=H4A, B=H4B),
    explanation_md=r"""
Treat $y$ as a constant when differentiating in $x$: $\partial_x(x^2 y) = 2xy$,
$\partial_x \sin(xy) = y\cos(xy)$ (chain rule, inner derivative $y$), $\partial_x e^y = 0$.
Then treat $x$ as a constant: $\partial_y(x^2 y) = x^2$, $\partial_y\sin(xy) = x\cos(xy)$,
$\partial_y e^y = e^y$. At $(1,2)$ this gives $(4 + 2\cos 2,\; 1 + \cos 2 + e^2) \approx (3.168, 7.973)$.
The classic slip is writing $\cos(xy)$ without the inner factor — the finite-difference check
catches it immediately.
""")

# ----------------------------------------------------------------------------- Ex 5
SOL5 = r"""
def F5(v):
    x, y = v
    return np.array([x * y, x**2 + y**2, np.exp(x) * np.sin(y)])

def jac_F5(v):
    x, y = v
    return np.array([[y,                    x],
                     [2 * x,                2 * y],
                     [np.exp(x) * np.sin(y), np.exp(x) * np.cos(y)]])
"""
run_solution(SOL5)
J5 = NS["jac_F5"](np.array([0.5, 1.0]))
H5 = safe_sha(J5)
ex.add(
    title="Jacobian of a map from R^2 to R^3", difficulty=2,
    prompt_md=r"""
The map $\mathbf{F}(x, y) = \bigl(xy,\; x^2 + y^2,\; e^{x}\sin y\bigr)$ is given as `F5(v)`.
Implement `jac_F5(v)` returning its Jacobian as a $3\times 2$ array with
$J_{ij} = \partial F_i/\partial x_j$ (rows = outputs, columns = inputs).

Then convince yourself of the linearisation $\mathbf{F}(\mathbf{v}+\boldsymbol{\delta}) \approx \mathbf{F}(\mathbf{v}) + J\boldsymbol{\delta}$:
the last check verifies that the error is tiny for a small $\boldsymbol{\delta}$.
""",
    starter=r"""
def F5(v):
    x, y = v
    return np.array([x * y, x**2 + y**2, np.exp(x) * np.sin(y)])

def jac_F5(v):
    x, y = v
    # YOUR CODE HERE
    return None
""",
    solution=SOL5,
    check=fill(r"""
try:
    J = np.asarray(jac_F5(np.array([0.5, 1.0])), dtype=float)
except Exception:
    J = None
check('Jacobian has shape (3, 2): 3 outputs, 2 inputs', J is not None and J.shape == (3, 2))
check_hash('hidden: Jacobian at (0.5, 1.0)', J, '__J__')
try:
    J0 = jac_F5(np.array([0.0, 0.0]))
except Exception:
    J0 = None
check_close('Jacobian at the origin is [[0,0],[0,0],[0,1]]', J0, np.array([[0., 0.], [0., 0.], [0., 1.]]))
try:
    v0, d = np.array([0.5, 1.0]), np.array([1e-3, -2e-3])
    lin_err = np.linalg.norm(F5(v0 + d) - F5(v0) - jac_F5(v0) @ d)
except Exception:
    lin_err = None
check('linearisation error for |delta| ~ 2e-3 is below 1e-5 (quadratic in delta)', lin_err is not None and lin_err < 1e-5)
""", J=H5),
    explanation_md=r"""
Each row is the gradient of one output component, written as a row: $\nabla(xy) = (y, x)$,
$\nabla(x^2+y^2) = (2x, 2y)$, $\nabla(e^x\sin y) = (e^x\sin y,\; e^x\cos y)$. The shape rule
"outputs × inputs" is the quickest sanity check — a $2\times 3$ answer means the layout was
transposed. The linearisation error scales with $\lVert\boldsymbol{\delta}\rVert^2$, so a step of
size $2\times10^{-3}$ gives an error of order $10^{-6}$.
""")

# ----------------------------------------------------------------------------- Ex 6
SOL6 = r"""
def g6(v):
    x, y = v
    return x**4 + y**4 - 4 * x * y

def grad_g6(v):
    x, y = v
    return np.array([4 * x**3 - 4 * y, 4 * y**3 - 4 * x])

def hess_g6(v):
    x, y = v
    return np.array([[12 * x**2, -4.0],
                     [-4.0,      12 * y**2]])

def classify(H, tol=1e-8):
    lam = np.linalg.eigvalsh(H)
    if np.all(lam > tol):
        return "minimum"
    if np.all(lam < -tol):
        return "maximum"
    if np.any(np.abs(lam) <= tol):
        return "degenerate"
    return "saddle"
"""
run_solution(SOL6)
eig11 = np.linalg.eigvalsh(NS["hess_g6"](np.array([1.0, 1.0])))
eig00 = np.linalg.eigvalsh(NS["hess_g6"](np.array([0.0, 0.0])))
H6E, H6C0, H6C1 = safe_sha(eig11), sha_of(NS["classify"](NS["hess_g6"](np.array([0.0, 0.0])))), sha_of(NS["classify"](NS["hess_g6"](np.array([1.0, 1.0]))))
ex.add(
    title="Hessian and classification of critical points", difficulty=2,
    prompt_md=r"""
Let $g(x, y) = x^4 + y^4 - 4xy$ (given as `g6`, with gradient `grad_g6` provided). Its critical
points are $(0,0)$, $(1,1)$ and $(-1,-1)$ — verify this from $\nabla g = \mathbf{0}$.

1. Implement `hess_g6(v)` returning the $2\times2$ Hessian (an array).
2. Implement `classify(H, tol=1e-8)` that returns one of the strings `"minimum"`, `"maximum"`,
   `"saddle"` or `"degenerate"` from the eigenvalues of a symmetric matrix `H`
   (use `np.linalg.eigvalsh`; treat eigenvalues with $|\lambda| \le$ `tol` as zero).

The hidden checks look at the eigenvalues at $(1, 1)$ and the classification of the two kinds of
critical point.
""",
    starter=r"""
def g6(v):
    x, y = v
    return x**4 + y**4 - 4 * x * y

def grad_g6(v):
    x, y = v
    return np.array([4 * x**3 - 4 * y, 4 * y**3 - 4 * x])

def hess_g6(v):
    x, y = v
    # YOUR CODE HERE
    return None

def classify(H, tol=1e-8):
    # YOUR CODE HERE: return "minimum", "maximum", "saddle" or "degenerate"
    return None
""",
    solution=SOL6,
    check=fill(r"""
try:
    H11 = np.asarray(hess_g6(np.array([1.0, 1.0])), dtype=float)
    lam11 = np.linalg.eigvalsh(H11)
except Exception:
    H11, lam11 = None, None
check('Hessian has shape (2, 2) and is symmetric', H11 is not None and H11.shape == (2, 2) and np.allclose(H11, H11.T))
check_hash('hidden: eigenvalues of the Hessian at (1, 1)', lam11, '__E__')
try:
    c0 = classify(hess_g6(np.array([0.0, 0.0])))
    c1 = classify(hess_g6(np.array([1.0, 1.0])))
    cm = classify(hess_g6(np.array([-1.0, -1.0])))
except Exception:
    c0, c1, cm = None, None, None
check_hash('hidden: classification of the origin', c0, '__C0__')
check_hash('hidden: classification of (1, 1)', c1, '__C1__')
check('(1, 1) and (-1, -1) are classified the same way', c1 is not None and c1 == cm)
try:
    got = classify(np.array([[-2.0, 0.0], [0.0, -1.0]]))
except Exception:
    got = None
check('negative definite Hessian is a maximum', got == 'maximum')
try:
    got = classify(np.array([[1.0, 0.0], [0.0, 0.0]]))
except Exception:
    got = None
check('a zero eigenvalue is degenerate', got == 'degenerate')
""", E=H6E, C0=H6C0, C1=H6C1),
    explanation_md=r"""
Differentiating the gradient once more gives $g_{xx} = 12x^2$, $g_{yy} = 12y^2$ and
$g_{xy} = g_{yx} = -4$ (Schwarz: the mixed partials agree). At $(1,1)$ the Hessian is
$\begin{pmatrix}12 & -4\\ -4 & 12\end{pmatrix}$ with eigenvalues $8$ and $16$ — both positive, a
minimum. At the origin it is $\begin{pmatrix}0 & -4\\ -4 & 0\end{pmatrix}$ with eigenvalues
$\pm 4$ — a saddle: $g$ decreases along $(1,1)$ and increases along $(1,-1)$. Working with
eigenvalues rather than the 2-D determinant shortcut makes `classify` work in any dimension.
""")

# ----------------------------------------------------------------------------- Ex 7
SOL7 = r"""
from math import factorial

def taylor_coeffs(derivs):
    return np.array([d / factorial(j) for j, d in enumerate(derivs)], dtype=float)

def taylor_eval(coeffs, a, x):
    x = np.asarray(x, dtype=float)
    return sum(c * (x - a)**j for j, c in enumerate(coeffs))
"""
run_solution(SOL7)
c_ln = NS["taylor_coeffs"]([0.0, 1.0, -1.0, 2.0, -6.0])
H7 = safe_sha(c_ln)
ex.add(
    title="Taylor polynomials from derivative values", difficulty=2,
    prompt_md=r"""
1. Implement `taylor_coeffs(derivs)`: given the list $[f(a), f'(a), f''(a), \dots, f^{(k)}(a)]$ of
   derivative values at a point $a$, return the array of Taylor coefficients
   $c_j = f^{(j)}(a)/j!$ for $j = 0,\dots,k$.
2. Implement `taylor_eval(coeffs, a, x)` returning $T_k(x) = \sum_j c_j (x-a)^j$ (vectorised over
   a NumPy array `x`).

For $\ln(1+x)$ about $a = 0$ the derivatives are $0, 1, -1, 2, -6, \dots$ (that is,
$f^{(j)}(0) = (-1)^{j+1}(j-1)!$ for $j\ge1$). For $e^x$ about $0$ they are all $1$.
""",
    starter=r"""
from math import factorial

def taylor_coeffs(derivs):
    # YOUR CODE HERE
    return None

def taylor_eval(coeffs, a, x):
    # YOUR CODE HERE
    return None
""",
    solution=SOL7,
    check=fill(r"""
try:
    c = np.asarray(taylor_coeffs([0.0, 1.0, -1.0, 2.0, -6.0]), dtype=float)
except Exception:
    c = None
check_hash('hidden: degree-4 Taylor coefficients of ln(1+x) about 0', c, '__C__')
try:
    got = taylor_eval(taylor_coeffs([1.0] * 4), 0.0, 0.5)
except Exception:
    got = None
check_close('degree-3 Taylor polynomial of exp at 0.5 is 1.645833...', got, 1 + 0.5 + 0.125 + 0.125 / 6, tol=1e-9)
try:
    xs = np.array([0.05, 0.1, 0.2])
    got = taylor_eval(c, 0.0, xs)
    err = np.abs(got - np.log1p(xs))
except Exception:
    got, err = None, None
check('taylor_eval is vectorised over x', got is not None and np.shape(got) == (3,))
check('degree-4 model of ln(1+x) is within 1e-4 at x = 0.2 but the error grows with x',
      err is not None and err[2] < 1e-4 and err[0] < err[1] < err[2])
try:
    got = taylor_eval(taylor_coeffs([np.e, np.e, np.e]), 1.0, 1.1)
except Exception:
    got = None
check_close('expansion of exp about a = 1 evaluated at 1.1', got, np.e * (1 + 0.1 + 0.005), tol=1e-9)
""", C=H7),
    explanation_md=r"""
Taylor's theorem matches the polynomial's derivatives to the function's at the base point;
since $\frac{d^j}{dx^j}(x-a)^j = j!$, the coefficient must be $f^{(j)}(a)/j!$. For $\ln(1+x)$
this gives $x - x^2/2 + x^3/3 - x^4/4$, and for $e^x$ the familiar $1 + x + x^2/2 + x^3/6$, which
at $0.5$ is $1.645833$ against the true $1.648721$. The error of the degree-$k$ model grows like
$|x-a|^{k+1}$, which is why the check insists the error increase with $x$. Forgetting the
factorial is the usual mistake; forgetting to expand in $(x - a)$ rather than $x$ is the second.
""")

# ----------------------------------------------------------------------------- Ex 8
SOL8 = r"""
def ls_loss(x, A, b):
    r = A @ x - b
    return float(r @ r)

def ls_grad(x, A, b):
    return 2 * A.T @ (A @ x - b)

def normal_eq_solve(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)
"""
run_solution(SOL8)
_rng8 = np.random.default_rng(0)
A8 = _rng8.standard_normal((8, 3)); b8 = _rng8.standard_normal(8); x8 = np.array([0.5, -1.0, 2.0])
g8 = NS["ls_grad"](x8, A8, b8); xs8 = NS["normal_eq_solve"](A8, b8)
H8G, H8X = safe_sha(g8), safe_sha(xs8)
ex.add(
    title="Gradient of least squares and the normal equations", difficulty=2,
    prompt_md=r"""
For $L(\mathbf{x}) = \lVert A\mathbf{x} - \mathbf{b}\rVert^2$ implement

- `ls_loss(x, A, b)` returning the scalar loss,
- `ls_grad(x, A, b)` returning the analytic gradient $2A^\top(A\mathbf{x}-\mathbf{b})$ (derive it
  from $\nabla \mathbf{x}^\top M\mathbf{x} = 2M\mathbf{x}$ and $\nabla \mathbf{c}^\top\mathbf{x} = \mathbf{c}$,
  or via the chain rule with the Jacobian $A$ of the residual), and
- `normal_eq_solve(A, b)` that returns the minimiser by solving $A^\top A\mathbf{x} = A^\top\mathbf{b}$
  with `np.linalg.solve` — this is "set the gradient to zero".

The checks use `A = rng.standard_normal((8, 3))`, `b = rng.standard_normal(8)` drawn from a
fresh `np.random.default_rng(0)` and the point `x0 = (0.5, -1, 2)`.
""",
    starter=r"""
def ls_loss(x, A, b):
    # YOUR CODE HERE
    return None

def ls_grad(x, A, b):
    # YOUR CODE HERE
    return None

def normal_eq_solve(A, b):
    # YOUR CODE HERE
    return None
""",
    solution=SOL8,
    check=fill(r"""
_r = np.random.default_rng(0)
A, b, x0 = _r.standard_normal((8, 3)), _r.standard_normal(8), np.array([0.5, -1.0, 2.0])
try:
    got = ls_loss(x0, A, b)
except Exception:
    got = None
check_close('loss at x0 equals the squared residual norm', got, float(np.sum((A @ x0 - b)**2)), tol=1e-9)
try:
    g = np.asarray(ls_grad(x0, A, b), dtype=float)
except Exception:
    g = None
check('gradient has shape (3,)', g is not None and g.shape == (3,))
check_hash('hidden: gradient at x0', g, '__G__')
try:
    g_ref = reference_grad_fd(lambda v: ls_loss(v, A, b), x0)
except Exception:
    g_ref = None
check_close('gradient matches finite differences', g, g_ref, tol=1e-5)
try:
    xs = np.asarray(normal_eq_solve(A, b), dtype=float)
    g_star = np.asarray(ls_grad(xs, A, b), dtype=float)
except Exception:
    xs, g_star = None, None
check_hash('hidden: least-squares solution', xs, '__X__')
check('gradient vanishes at the normal-equation solution', g_star is not None and np.linalg.norm(g_star) < 1e-9)
check('normal-equation solution agrees with np.linalg.lstsq', xs is not None and np.allclose(xs, np.linalg.lstsq(A, b, rcond=None)[0]))
""", G=H8G, X=H8X),
    explanation_md=r"""
Expanding $\lVert A\mathbf{x}-\mathbf{b}\rVert^2 = \mathbf{x}^\top A^\top A\mathbf{x} - 2\mathbf{b}^\top A\mathbf{x} + \mathbf{b}^\top\mathbf{b}$
and differentiating term by term gives $2A^\top A\mathbf{x} - 2A^\top\mathbf{b} = 2A^\top(A\mathbf{x}-\mathbf{b})$.
Setting this to zero is the normal equations, so the least-squares solution of Module 02 is the
unique critical point of a convex quadratic (Hessian $2A^\top A \succeq 0$). The common bug is
$2A(A\mathbf{x}-\mathbf{b})$ — wrong shape whenever $A$ is not square, and wrong values even when
it is; the finite-difference check exposes it.
""")

# ----------------------------------------------------------------------------- Ex 9
SOL9 = r"""
def bce_loss(w, X, y):
    z = X @ w
    # stable form: -log(sigmoid(z)) = softplus(-z), -log(1 - sigmoid(z)) = softplus(z)
    return float(np.mean(y * np.logaddexp(0.0, -z) + (1 - y) * np.logaddexp(0.0, z)))

def bce_grad(w, X, y):
    p = sigmoid(X @ w)
    return X.T @ (p - y) / len(y)
"""
run_solution(SOL9)
X9 = np.array([[1.0, 0.5, -1.0], [1.0, -1.5, 0.5], [1.0, 2.0, 1.0], [1.0, 0.0, -0.5]])
y9 = np.array([1.0, 0.0, 1.0, 0.0]); w9 = np.array([0.1, -0.2, 0.3])
L9 = NS["bce_loss"](w9, X9, y9); g9 = NS["bce_grad"](w9, X9, y9)
H9L, H9G = safe_sha(L9), safe_sha(g9)
ex.add(
    title="Gradient of the logistic loss", difficulty=2,
    prompt_md=r"""
For logistic regression with $p_i = \sigma(\mathbf{w}^\top\mathbf{x}_i)$ and the mean binary
cross-entropy $L(\mathbf{w}) = -\frac1n\sum_i\bigl[y_i\ln p_i + (1-y_i)\ln(1-p_i)\bigr]$, implement

- `bce_loss(w, X, y)` (a float; `X` is $n\times d$, `y` is a 0/1 array of length $n$), and
- `bce_grad(w, X, y)` returning $\nabla_{\mathbf{w}}L = \frac1n X^\top(\mathbf{p}-\mathbf{y})$ — derive
  $\partial L_i/\partial z_i = p_i - y_i$ first (Section 8 of the guide).

Tiny dataset for the checks:
```python
X = np.array([[1.0, 0.5, -1.0], [1.0, -1.5, 0.5], [1.0, 2.0, 1.0], [1.0, 0.0, -0.5]])
y = np.array([1.0, 0.0, 1.0, 0.0]);   w = np.array([0.1, -0.2, 0.3])
```
""",
    starter=r"""
def bce_loss(w, X, y):
    # YOUR CODE HERE
    return None

def bce_grad(w, X, y):
    # YOUR CODE HERE
    return None
""",
    solution=SOL9,
    check=fill(r"""
X = np.array([[1.0, 0.5, -1.0], [1.0, -1.5, 0.5], [1.0, 2.0, 1.0], [1.0, 0.0, -0.5]])
y = np.array([1.0, 0.0, 1.0, 0.0]); w = np.array([0.1, -0.2, 0.3])
try:
    L = float(bce_loss(w, X, y))
except Exception:
    L = None
check_hash('hidden: loss at w', L, '__L__')
try:
    L0 = float(bce_loss(np.zeros(3), X, y))
except Exception:
    L0 = None
check_close('loss at w = 0 is ln 2', L0, np.log(2.0), tol=1e-9)
try:
    g = np.asarray(bce_grad(w, X, y), dtype=float)
except Exception:
    g = None
check('gradient has shape (3,)', g is not None and g.shape == (3,))
check_hash('hidden: gradient at w', g, '__G__')
try:
    g_ref = reference_grad_fd(lambda v: bce_loss(v, X, y), w)
except Exception:
    g_ref = None
check_close('gradient matches finite differences', g, g_ref, tol=1e-6)
try:
    L_after = float(bce_loss(w - 0.5 * g, X, y))
except Exception:
    L_after = None
check('one gradient step (eta = 0.5) lowers the loss', L is not None and L_after is not None and L_after < L)
""", L=H9L, G=H9G),
    explanation_md=r"""
With $p = \sigma(z)$, $\partial L_i/\partial p = (p-y)/(p(1-p))$ and $dp/dz = p(1-p)$; the
product is simply $p - y$, and the chain rule through $z = \mathbf{w}^\top\mathbf{x}$ multiplies by
$\mathbf{x}$. Stacking the $n$ examples gives $X^\top(\mathbf{p}-\mathbf{y})/n$. Computing the loss
as $\mathrm{softplus}(-z)$ and $\mathrm{softplus}(z)$ via `np.logaddexp` avoids $\ln 0$ for
confident predictions — the reason frameworks offer "with logits" losses. At $\mathbf{w} = 0$
every $p_i = 1/2$, so the loss is $\ln 2$ regardless of the data.
""")

# ----------------------------------------------------------------------------- Ex 11 (stretch)
SOL11 = r"""
def hvp_fd(grad_f, x, v, h=1e-5):
    x = np.asarray(x, dtype=float); v = np.asarray(v, dtype=float)
    return (grad_f(x + h * v) - grad_f(x - h * v)) / (2 * h)

def top_curvature(grad_f, x, n_iter=200, seed=0):
    r = np.random.default_rng(seed)
    v = r.standard_normal(len(x))
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(n_iter):
        Hv = hvp_fd(grad_f, x, v)
        lam = float(v @ Hv)
        nrm = np.linalg.norm(Hv)
        if nrm == 0:
            break
        v = Hv / nrm
    return lam, v
"""
run_solution(SOL11)
H11 = np.array([[2.0, 3.0], [3.0, 10.0]])
lam11_est, _ = NS["top_curvature"](lambda x: H11 @ x, np.zeros(2))
assert abs(lam11_est - 11.0) < 1e-6
# a non-quadratic test function f(x) = sum(x^4) + x^T M x  with M symmetric
M11 = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
x11 = np.array([0.5, -1.0, 0.25])
H_exact_11 = np.diag(12 * x11**2) + 2 * M11
lam_exact_11 = float(np.max(np.linalg.eigvalsh(H_exact_11)))
lam_est_11, _ = NS["top_curvature"](lambda x: 4 * x**3 + 2 * M11 @ x, x11)
assert abs(lam_est_11 - lam_exact_11) < 1e-4, (lam_est_11, lam_exact_11)
H11L = safe_sha(lam_exact_11, decimals=2)
ex.add(
    title="Hessian-vector products and the sharpest direction", difficulty=3,
    prompt_md=r"""
For a billion parameters the Hessian cannot be stored, but the *product* $H\mathbf{v}$ can be
computed from gradients alone, because $H$ is the Jacobian of $\nabla f$:

$$H\mathbf{v} \approx \frac{\nabla f(\mathbf{x} + h\mathbf{v}) - \nabla f(\mathbf{x} - h\mathbf{v})}{2h}.$$

1. Implement `hvp_fd(grad_f, x, v, h=1e-5)` using this formula (`grad_f` returns the gradient
   array at a point).
2. Implement `top_curvature(grad_f, x, n_iter=200, seed=0)` that runs **power iteration** with
   `hvp_fd` to estimate the largest eigenvalue $\lambda_{\max}$ of the Hessian at `x` (start from a
   random unit vector drawn with `np.random.default_rng(seed)`, repeatedly set
   $\mathbf{v} \leftarrow H\mathbf{v}/\lVert H\mathbf{v}\rVert$, and return the tuple
   `(lam, v)` with `lam = v @ (H v)` as a float). Assume the Hessian's largest-magnitude eigenvalue
   is positive, as it is for the test functions.

Test functions: the quadratic $\tfrac12\mathbf{x}^\top H\mathbf{x}$ with
$H = \begin{pmatrix}2&3\\3&10\end{pmatrix}$ (gradient $H\mathbf{x}$, eigenvalues $1$ and $11$) and
$f(\mathbf{x}) = \sum_i x_i^4 + \mathbf{x}^\top M\mathbf{x}$ with gradient $4\mathbf{x}^3 + 2M\mathbf{x}$
for the symmetric $M$ given in the check cell, at $\mathbf{x} = (0.5, -1, 0.25)$. Recall from the
lesson that $2/\lambda_{\max}$ is the largest stable learning rate for gradient descent.
""",
    starter=r"""
def hvp_fd(grad_f, x, v, h=1e-5):
    # YOUR CODE HERE
    return None

def top_curvature(grad_f, x, n_iter=200, seed=0):
    # YOUR CODE HERE: power iteration using hvp_fd; return (lam, v)
    return None
""",
    solution=SOL11,
    check=fill(r"""
H = np.array([[2.0, 3.0], [3.0, 10.0]])
grad_quad = lambda x: H @ x
try:
    got = hvp_fd(grad_quad, np.array([1.0, -1.0]), np.array([1.0, 2.0]))
except Exception:
    got = None
check_close('H v for the quadratic equals the matrix-vector product (8, 23)', got, H @ np.array([1.0, 2.0]), tol=1e-6)
try:
    lam, v = top_curvature(grad_quad, np.zeros(2))
    lam = float(lam)
except Exception:
    lam, v = None, None
check_close('largest curvature of the quadratic is 11', lam, 11.0, tol=1e-5)
check('returned eigenvector is a unit vector aligned with (0.316, 0.949)',
      v is not None and abs(np.linalg.norm(v) - 1) < 1e-6 and abs(abs(np.dot(v, [0.31622777, 0.9486833])) - 1) < 1e-4)

M = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
grad_quartic = lambda x: 4 * x**3 + 2 * M @ x
x0 = np.array([0.5, -1.0, 0.25])
try:
    lam2 = float(top_curvature(grad_quartic, x0)[0])
except Exception:
    lam2 = None
check_hash('hidden: largest curvature of the quartic at x0 (2 decimals)', lam2, '__LAM__', decimals=2)
check('stability limit 2/lambda_max for the quartic lies between 0.11 and 0.13', lam2 is not None and 0.11 < 2 / lam2 < 0.13)
""", LAM=H11L),
    explanation_md=r"""
Since $\nabla f(\mathbf{x} + h\mathbf{v}) \approx \nabla f(\mathbf{x}) + hH\mathbf{v}$ (the first-order
Taylor expansion of the *gradient*), the central difference of two gradients along $\mathbf{v}$
isolates $H\mathbf{v}$ with $O(h^2)$ error — two gradient evaluations, no Hessian. Power iteration
(Module 03) then converges to the eigenvector of the largest-magnitude eigenvalue, because each
multiplication by $H$ amplifies that component the most; the Rayleigh quotient
$\mathbf{v}^\top H\mathbf{v}$ returns the eigenvalue. For the quartic, the exact Hessian is
$\mathrm{diag}(12x_i^2) + 2M$, whose largest eigenvalue at $\mathbf{x}_0$ is about $16.6$, so gradient
descent there would need $\eta < 0.12$. Autodiff frameworks compute the same $H\mathbf{v}$ exactly
with a forward-over-reverse pass (`torch.autograd.functional.hvp`), and this is how the
"sharpness" of a trained network is measured in practice.
""")

# ----------------------------------------------------------------------------- Ex 12 (project)
PROJECT_STARTER = r"""
# --- data and a two-layer *linear* model:  y_hat = X @ W1 @ W2,  loss = mean((y_hat - y)^2) ---
rng_p = np.random.default_rng(1)
n, d, k = 50, 4, 3
X = rng_p.standard_normal((n, d))
y = X @ rng_p.standard_normal(d) + 0.1 * rng_p.standard_normal(n)
W1 = 0.5 * rng_p.standard_normal((d, k))
W2 = 0.5 * rng_p.standard_normal(k)

def pack(W1, W2):
    return np.concatenate([W1.ravel(), W2.ravel()])

def unpack(theta):
    return theta[:d * k].reshape(d, k), theta[d * k:]

def loss(theta):
    W1, W2 = unpack(theta)
    r = X @ W1 @ W2 - y
    return float(np.mean(r**2))

def grad_loss_buggy(theta):
    # A colleague's hand-derived gradient. It contains at least one mistake. Find it with grad_check.
    W1, W2 = unpack(theta)
    hdn = X @ W1                      # n x k
    r = hdn @ W2 - y                  # n
    gW2 = hdn.T @ r / n               # dL/dW2
    gW1 = X.T @ np.outer(r, W2)       # dL/dW1
    return pack(gW1, gW2)

def grad_check(f, grad_f, x, h=1e-6):
    # YOUR CODE HERE: return (relative_error, numerical_gradient, analytic_gradient)
    return None

# Your mini-project workspace: write grad_check, run it on grad_loss_buggy, fix the gradient,
# then plot the relative error versus h to see the round-off regime.
"""

PROJECT_SOLUTION = r"""
rng_p = np.random.default_rng(1)
n, d, k = 50, 4, 3
X = rng_p.standard_normal((n, d))
y = X @ rng_p.standard_normal(d) + 0.1 * rng_p.standard_normal(n)
W1 = 0.5 * rng_p.standard_normal((d, k))
W2 = 0.5 * rng_p.standard_normal(k)

def pack(W1, W2):
    return np.concatenate([W1.ravel(), W2.ravel()])

def unpack(theta):
    return theta[:d * k].reshape(d, k), theta[d * k:]

def loss(theta):
    W1, W2 = unpack(theta)
    r = X @ W1 @ W2 - y
    return float(np.mean(r**2))

def grad_loss_buggy(theta):
    W1, W2 = unpack(theta)
    hdn = X @ W1
    r = hdn @ W2 - y
    gW2 = hdn.T @ r / n               # missing factor 2
    gW1 = X.T @ np.outer(r, W2)       # missing factor 2/n
    return pack(gW1, gW2)

def grad_loss(theta):
    # L = (1/n) sum_i r_i^2 with r = X W1 W2 - y.  dL/dr = 2r/n.
    # r = H W2 with H = X W1  =>  dL/dW2 = H^T (2r/n);  dL/dH = (2r/n) W2^T  =>  dL/dW1 = X^T dL/dH
    W1, W2 = unpack(theta)
    hdn = X @ W1
    r = hdn @ W2 - y
    dr = 2 * r / n
    gW2 = hdn.T @ dr
    gW1 = X.T @ np.outer(dr, W2)
    return pack(gW1, gW2)

def grad_check(f, grad_f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    g_num = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g_num[i] = (f(x + e) - f(x - e)) / (2 * h)
    g_an = np.asarray(grad_f(x), dtype=float)
    rel = np.linalg.norm(g_an - g_num) / (np.linalg.norm(g_an) + np.linalg.norm(g_num) + 1e-300)
    return float(rel), g_num, g_an

theta0 = pack(W1, W2)
rel_bug, g_num, g_bug = grad_check(loss, grad_loss_buggy, theta0)
rel_ok, _, g_ok = grad_check(loss, grad_loss, theta0)
print(f"relative error, buggy gradient : {rel_bug:.3e}   -> FAIL (should be < 1e-6)")
print(f"relative error, fixed gradient : {rel_ok:.3e}   -> PASS")

# which coordinates are wrong, and by what factor?  (the ratio reveals the missing constants)
ratio = g_num / g_bug
print("numerical / buggy ratio for W1 block:", np.unique(ratio[:d * k].round(6)))
print("numerical / buggy ratio for W2 block:", np.unique(ratio[d * k:].round(6)))

# relative error versus step size. Surprise: for the *linear* model there is no V shape, because
# the loss is an exact quadratic along every coordinate direction, so the central difference has
# zero truncation error and only round-off (which shrinks as h grows) remains.
hs = np.logspace(-12, -1, 45)
rels_lin = [grad_check(loss, grad_loss, theta0, h=h)[0] for h in hs]

# a non-linear hidden layer brings the truncation term back:  y_hat = tanh(X W1) W2
def loss_tanh(theta):
    W1, W2 = unpack(theta)
    r = np.tanh(X @ W1) @ W2 - y
    return float(np.mean(r**2))

def grad_loss_tanh(theta):
    W1, W2 = unpack(theta)
    a = X @ W1
    hdn = np.tanh(a)
    r = hdn @ W2 - y
    dr = 2 * r / n
    gW2 = hdn.T @ dr
    dh = np.outer(dr, W2)                 # dL/dh, n x k
    da = dh * (1 - hdn**2)                # chain rule through tanh
    gW1 = X.T @ da
    return pack(gW1, gW2)

rel_tanh, _, _ = grad_check(loss_tanh, grad_loss_tanh, theta0)
print(f"relative error, tanh model gradient : {rel_tanh:.3e}   -> PASS")
rels_tanh = [grad_check(loss_tanh, grad_loss_tanh, theta0, h=h)[0] for h in hs]

plt.loglog(hs, rels_lin, "o-", ms=3, label="linear model (quadratic loss: no truncation error)")
plt.loglog(hs, rels_tanh, "s-", ms=3, label="tanh hidden layer (truncation + round-off: V shape)")
plt.xlabel("finite-difference step h"); plt.ylabel("relative error of gradient check")
plt.title("Gradient-check error vs step size"); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()
print("tanh model: best h =", hs[int(np.argmin(rels_tanh))], " best relative error =", min(rels_tanh))

# sanity: a few gradient steps with the fixed gradient reduce the loss
theta = theta0.copy()
for _ in range(200):
    theta -= 0.05 * grad_loss(theta)
print(f"linear model loss before {loss(theta0):.4f}  ->  after 200 steps {loss(theta):.4f}")
"""
# make sure the solution code runs and behaves as described
_pns: dict = {}
exec("import numpy as np\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n" + textwrap.dedent(PROJECT_SOLUTION), _pns)
assert _pns["rel_bug"] > 1e-2 and _pns["rel_ok"] < 1e-7, (_pns["rel_bug"], _pns["rel_ok"])
assert _pns["rel_tanh"] < 1e-7 and 1e-8 < _pns["hs"][int(np.argmin(_pns["rels_tanh"]))] < 1e-3

ex.add_project(
    title="Mini-project: a reusable gradient checker", difficulty=3,
    prompt_md=r"""
Analytic gradients are where most bugs in hand-written ML code live. Build the tool that catches
them and use it to debug a real (small) model.

1. Write `grad_check(f, grad_f, x, h=1e-6)` for any scalar function `f` of a 1-D parameter vector
   `x` and a candidate gradient function `grad_f`. It should return the **relative error**

   $$\frac{\lVert \mathbf{g}_{\text{analytic}} - \mathbf{g}_{\text{numeric}}\rVert}{\lVert \mathbf{g}_{\text{analytic}}\rVert + \lVert \mathbf{g}_{\text{numeric}}\rVert}$$

   together with both gradients, using central differences for $\mathbf{g}_{\text{numeric}}$.
2. The starter cell defines a two-layer *linear* model $\hat{\mathbf{y}} = XW_1W_2$ with MSE loss
   $L = \frac1n\lVert XW_1W_2 - \mathbf{y}\rVert^2$, and a colleague's gradient `grad_loss_buggy`.
   Run your checker on it. It should fail.
3. Derive the correct gradient with the chain rule (hint: with $\mathbf{r} = XW_1W_2 - \mathbf{y}$ and
   $\mathbf{h} = XW_1$, $\partial L/\partial\mathbf{r} = 2\mathbf{r}/n$; then
   $\nabla_{W_2}L = H^\top(\partial L/\partial\mathbf{r})$ and $\nabla_{W_1}L = X^\top(\partial L/\partial\mathbf{r})W_2^\top$).
   Use the *ratio* of numerical to analytic entries to locate the bug, write `grad_loss`, and show
   the checker passes (relative error below $10^{-6}$).
4. Plot the relative error of the correct gradient against $h$ from $10^{-12}$ to $10^{-1}$. You
   will *not* see the V shape from the lesson — explain why (what is the third derivative of the
   loss along a single coordinate of $W_1$ or $W_2$?). Then replace the hidden layer by
   $\tanh(XW_1)$, derive the gradient of the new model (one extra factor $1 - \tanh^2$ in the
   chain), check it, and plot its error curve: the truncation arm should reappear.
5. Take a few gradient steps with your corrected gradient and confirm the loss decreases.
""",
    rubric_md=r"""
- `grad_check` works for any `f`/`grad_f` pair and any dimension, never modifies `x` in place, and
  returns a relative error that is scale-free (dividing by the sum of the norms, with a guard
  against division by zero).
- The buggy gradient is flagged with a relative error far above $10^{-6}$, and you can say *which
  constants* are missing from each block (a factor $2$ in $W_2$, a factor $2/n$ in $W_1$).
- The corrected gradient is derived with the chain rule, written in matrix form with the right
  shapes ($d\times k$ and $k$), and passes with relative error around $10^{-9}$ or better.
- You can explain why the linear model's error curve has no truncation arm (the loss is exactly
  quadratic along each coordinate, so the central difference is exact and only round-off remains)
  and why the tanh model's curve shows the V shape with a minimum near $h \approx 10^{-5}$.
- The tanh model's gradient is derived with one more chain-rule factor, $1 - \tanh^2$, applied
  elementwise to the hidden pre-activations, and passes the checker.
- A few gradient-descent steps with the corrected gradient reduce the loss.
""",
    solution_sketch_md=r"""
The checker is Exercise 1's `grad_fd` plus a normalised comparison. For the model, write the
forward pass as $\mathbf{h} = XW_1$, $\mathbf{r} = \mathbf{h}W_2 - \mathbf{y}$, $L = \frac1n\mathbf{r}^\top\mathbf{r}$.
Backwards: $\partial L/\partial\mathbf{r} = \frac{2}{n}\mathbf{r}$; since $\mathbf{r}$ is linear in
$W_2$ with Jacobian $H = XW_1$, $\nabla_{W_2}L = H^\top\frac{2}{n}\mathbf{r}$ (shape $k$); since
$\mathbf{r} = XW_1W_2$ is linear in $W_1$, $\nabla_{W_1}L = X^\top\bigl(\frac{2}{n}\mathbf{r}\bigr)W_2^\top$,
the outer product giving shape $d\times k$. The colleague forgot the factor $2$ in the $W_2$ block
and the factor $2/n$ in the $W_1$ block — the ratio of numerical to analytic entries is exactly
$2$ and $2/n = 0.04$ in the two blocks, which pinpoints the bug without any further algebra.
The error-versus-$h$ curve for the linear model only *falls* as $h$ grows: along any single
coordinate the loss is a quadratic, its third derivative is zero, and the central difference is
exact up to round-off ($\propto \varepsilon/h$). With a $\tanh$ hidden layer,
$\hat{\mathbf{y}} = \tanh(XW_1)W_2$, the backward pass gains the factor $1 - \tanh^2(XW_1)$ on the
way to $W_1$, the third derivative is no longer zero, and the familiar V appears with its minimum
near $h \approx 10^{-5}$.
""",
    solution_code=PROJECT_SOLUTION,
    starter=PROJECT_STARTER,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print("exercises:", len(ex.items), "coding:", kinds.count("code"), "conceptual:", kinds.count("concept"), "project:", kinds.count("project"))
print("difficulties:", [it["difficulty"] for it in ex.items])
print("wrote", MOD / "lesson.ipynb", MOD / "exercises.ipynb", MOD / "solutions.ipynb")
