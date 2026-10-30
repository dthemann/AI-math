"""Builder for Module 05 — Optimisation: writes lesson.ipynb, exercises.ipynb, solutions.ipynb."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import Notebook, ExerciseSet, sha_of
import numpy as np
from sklearn.datasets import load_breast_cancer

MOD = ROOT / "part-2-calculus-and-optimization/05-optimization"

# =============================================================================
# LESSON
# =============================================================================
nb = Notebook("Module 05 — Optimisation")

nb.md(r"""
This notebook is the runnable companion to the [concept guide](README.md) — read the guide first, or
keep it open alongside. Everything here is built from scratch in NumPy: gradient descent on an
ill-conditioned bowl (including a run that diverges), the convergence rate as a function of the
condition number, mini-batch SGD and its noise floor, momentum / RMSProp / Adam on the Rosenbrock
function and on a real logistic-regression problem, Newton's method, Lagrange multipliers as a
picture, and the diagnostics (schedules, range test, clipping, early stopping) you use in practice.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
""")

# ---------------------------------------------------------------- 1. landscapes
nb.md(r"""
## 1. The optimisation problem: landscapes, minima and saddles

Training minimises an empirical risk $L(\theta) = \frac1n\sum_i \ell(f_\theta(x_i), y_i)$. The
simplest possible instance: fit $y \approx w x$ to three points by mean squared error. In one
dimension you can see the whole landscape, so we plot it and check that the derivative changes
sign exactly at the minimiser $w^\star = \sum x_i y_i / \sum x_i^2$.

A *stationary* point has $\nabla L = 0$, but that is not enough to be a minimum: on
$f(x, y) = x^2 - y^2$ the origin is stationary and is a **saddle** — downhill along $y$, uphill
along $x$. Gradient descent started exactly on a saddle never moves; started anywhere nearby it
slides off along the negative-curvature direction.
""")

nb.code(r"""
x = np.array([1., 2., 3.])
y = np.array([2.1, 3.9, 6.2])

def loss_1d(w):
    return np.mean((w * x - y) ** 2)

def dloss_1d(w):
    return np.mean(2 * x * (w * x - y))

w_star = (x @ y) / (x @ x)
for w in [1.5, 2.0, 2.1]:
    print(f"w = {w:4.2f}   L(w) = {loss_1d(w):.4f}   dL/dw = {dloss_1d(w):+.3f}")
print(f"closed form: w* = {w_star:.4f},  L(w*) = {loss_1d(w_star):.4f},  dL/dw(w*) = {dloss_1d(w_star):+.1e}")

ws = np.linspace(0.5, 3.5, 300)
w_tab = np.array([1.5, 2.0, 2.1])                      # the three tabulated points
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ws, [loss_1d(w) for w in ws], label="L(w)")
ax[0].plot(ws, [dloss_1d(w) for w in ws], label="dL/dw", alpha=0.7)
ax[0].plot(w_tab, [dloss_1d(w) for w in w_tab], "ko", ms=4, label="dL/dw at w = 1.5, 2.0, 2.1")
ax[0].axhline(0, color="gray", lw=0.8)
ax[0].axvline(w_star, color="gray", ls="--", lw=0.8, label=f"w* = {w_star:.3f}")
ax[0].set_title("A 1-D loss landscape: y ~ w x on three points")
ax[0].set_xlabel("w"); ax[0].set_ylabel("value"); ax[0].legend(fontsize=8, loc="upper left")
# zoom on the sign change of dL/dw between w = 2.0 and w = 2.1
inset = ax[0].inset_axes([0.62, 0.14, 0.35, 0.33])
wz = np.linspace(1.9, 2.2, 50)
inset.plot(wz, [dloss_1d(w) for w in wz], color="C1")
inset.plot(w_tab[1:], [dloss_1d(w) for w in w_tab[1:]], "ko", ms=4)
for w in w_tab[1:]:
    inset.annotate(f"{dloss_1d(w):+.3f}", (w, dloss_1d(w)), textcoords="offset points",
                   xytext=(-34, 12) if w < w_star else (5, -11), fontsize=7)
inset.axhline(0, color="gray", lw=0.8); inset.axvline(w_star, color="gray", ls="--", lw=0.8)
inset.set_title("zoom: dL/dw near w*", fontsize=7); inset.set_xlabel("w", fontsize=7); inset.set_ylabel("dL/dw", fontsize=7)
inset.tick_params(labelsize=6)

X, Y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101))
cs = ax[1].contour(X, Y, X ** 2 - Y ** 2, levels=np.linspace(-4, 4, 17), cmap="coolwarm")
ax[1].clabel(cs, fmt="%.0f", fontsize=7)
ax[1].plot(0, 0, "ko"); ax[1].annotate("saddle", (0.1, 0.1))
ax[1].set_title("f(x, y) = x^2 - y^2: stationary, but not a minimum")
ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Notice that the derivative is negative at $w = 2.0$ and positive at $w = 2.1$: the minimiser
$w^\star = 2.036$ sits between them, exactly where $\mathrm{d}L/\mathrm{d}w$ crosses zero. On the
right, the contours of $x^2 - y^2$ form a cross at the origin: the Hessian $\mathrm{diag}(2, -2)$
has one positive and one negative eigenvalue. In high dimensions saddles are far more common than
local minima, and most "the loss has plateaued" moments in deep learning are slow passages near
saddles, not convergence.

What does gradient descent actually do near that saddle? The gradient is $(2x, -2y)$, so one step with
step size $\eta$ multiplies $x$ by $1 - 2\eta$ and $y$ by $1 + 2\eta$. We start four runs at $x_0 = 1.5$,
at different small distances $y_0$ from the $x$-axis, and follow both the path and the loss.
""")

nb.code(r"""
eta_sad = 0.1                                   # x shrinks by 1 - 2 eta = 0.8 per step, y grows by 1 + 2 eta = 1.2
Xs, Ys = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].contour(Xs, Ys, Xs ** 2 - Ys ** 2, levels=np.linspace(-4, 4, 17), cmap="coolwarm", alpha=0.5)
for y0, mk in [(0.3, "o"), (1e-2, "s"), (-1e-4, "^"), (0.0, "x")]:
    p = np.array([1.5, y0]); path_sad = [p]
    for _ in range(90):
        p = p - eta_sad * np.array([2 * p[0], -2 * p[1]]); path_sad.append(p)
    path_sad = np.array(path_sad); keep = np.abs(path_sad[:, 1]) <= 2        # stop drawing once the run has left the box
    lab = f"start y0 = {y0:g}" + ("  (exactly on the x-axis)" if y0 == 0 else "")
    ax[0].plot(path_sad[keep, 0], path_sad[keep, 1], marker=mk, ms=3, lw=1, label=lab)
    ax[1].plot(np.flatnonzero(keep), (path_sad[:, 0] ** 2 - path_sad[:, 1] ** 2)[keep], marker=mk, ms=2, lw=1, label=lab)
    if y0 != 0:
        print(f"y0 = {y0:+g}: |y| reaches 1 after about ln(1/|y0|)/ln(1.2) = {np.log(1 / abs(y0)) / np.log(1 + 2 * eta_sad):.0f} steps")
    else:
        print(f"y0 = 0    : after 90 steps the iterate is at {path_sad[-1]} -- stuck on the saddle")
ax[0].plot(1.5, 0, "k^", ms=8, label="common start (1.5, y0)"); ax[0].plot(0, 0, "ko", label="saddle")
ax[0].set_title("GD near a saddle: slide in along x, escape along y"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[0].legend(fontsize=7, loc="lower left")
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_title("f along each path: escape after ~ ln(1/|y0|)/ln 1.2 steps"); ax[1].set_xlabel("GD step"); ax[1].set_ylabel("f(x, y)")
ax[1].set_ylim(-4.5, 2.5); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** the start exactly on the $x$-axis slides into the saddle and stays there for ever,
while every other start first slides in ($x$ shrinks by $1 - 2\eta = 0.8$ per step) and then escapes
along $y$, which grows by $1 + 2\eta = 1.2$ per step. The loss falls to near $f = 0$ and stays there
until the escape, when $\lvert y\rvert$ reaches 1 (and $f \approx -1$) after about
$\ln(1/\lvert y_0\rvert)/\ln 1.2$ steps: 25 and 51 for the two small $y_0$, but only 7 for $y_0 = 0.3$, so
early that there is hardly a plateau. The closer a run passes a saddle, the longer it looks converged.
The sign of $y_0$ only decides which way it leaves.
""")

# ---------------------------------------------------------------- 2. convexity
nb.md(r"""
## 2. Convexity: the Hessian test and Jensen's inequality

A twice-differentiable function is convex iff its Hessian is positive semi-definite everywhere,
$\nabla^2 f \succeq 0$. We test this numerically by estimating the Hessian with finite differences
at many random points and recording the *smallest* eigenvalue:

* the logistic loss of a linear classifier, $\frac1n\sum_i \log(1 + e^{-\tilde y_i (w_0 + w_1 x_i)})$,
  has Hessian $\frac1n X^\top S X$ with $S_{ii} = p_i(1-p_i) \ge 0$ — convex;
* a one-neuron "network" $a \tanh(w x)$ with squared loss is **not** convex.

We also check Jensen's inequality $f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$ for $f(x) = x^2$ with a random
discrete distribution.
""")

nb.code(r"""
n = 40
x1 = rng.normal(size=n)
y_cls = (x1 + 0.5 * rng.normal(size=n) > 0).astype(float)      # labels in {0, 1}
y_pm = 2 * y_cls - 1                                            # the same labels in {-1, +1}

def logistic_loss(w):
    z = w[0] + w[1] * x1
    return np.mean(np.logaddexp(0.0, -y_pm * z))

def tiny_net_loss(theta):
    a, w = theta
    return np.mean((a * np.tanh(w * x1) - y_pm) ** 2)

def numerical_hessian(f, p, h=1e-4):
    p = np.asarray(p, dtype=float); d = len(p); H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei = np.zeros(d); ei[i] = h
            ej = np.zeros(d); ej[j] = h
            H[i, j] = (f(p + ei + ej) - f(p + ei - ej) - f(p - ei + ej) + f(p - ei - ej)) / (4 * h * h)
    return H

points = rng.normal(size=(60, 2)) * 2.0
min_eig_logistic = [np.linalg.eigvalsh(numerical_hessian(logistic_loss, p))[0] for p in points]
min_eig_net = [np.linalg.eigvalsh(numerical_hessian(tiny_net_loss, p))[0] for p in points]
print(f"logistic loss : smallest Hessian eigenvalue over 60 points = {min(min_eig_logistic):+.4f}  (never negative)")
print(f"tiny tanh net : smallest Hessian eigenvalue over 60 points = {min(min_eig_net):+.4f}  "
      f"(negative at {sum(m < -1e-8 for m in min_eig_net)} of 60 points)")

# Jensen's inequality for f(x) = x^2 and a random discrete distribution
vals = rng.normal(size=6); probs = rng.dirichlet(np.ones(6))
lhs = (probs @ vals) ** 2; rhs = probs @ vals ** 2
print(f"Jensen for x^2:  f(E[X]) = {lhs:.4f}  <=  E[f(X)] = {rhs:.4f}   gap = Var(X) = {rhs - lhs:.4f}")

# the two minima of x^4 - x^2 and the chord between them
xx = np.linspace(-1.3, 1.3, 300); f4 = xx ** 4 - xx ** 2
plt.plot(xx, f4, label="x^4 - x^2")
xa, xb = -1 / np.sqrt(2), 1 / np.sqrt(2)
plt.plot([xa, xb], [xa ** 4 - xa ** 2, xb ** 4 - xb ** 2], "o--", label="chord between the two minima")
plt.title("A non-convex function: the chord passes below the graph")
plt.xlabel("x"); plt.ylabel("f(x)"); plt.legend(); plt.show()
""")

nb.md(r"""
Notice that the logistic loss never shows a negative curvature direction, while the one-neuron
network does at most of the random points. For $x^2$ the Jensen gap is exactly the variance
of $X$ — a fact you will meet again as the bias–variance decomposition (Module 13) and in the ELBO
(Module 18). The chord picture is the definition of convexity made visible: for a convex function
the dashed line would lie *above* the curve.

Where does the tiny network's negative curvature come from? It has only two parameters, so we can
draw its whole loss surface over $(a, w)$, mark the 60 test points by the sign of their smallest
Hessian eigenvalue, and follow the loss along the straight line between its two lowest points.
""")

nb.code(r"""
# the tiny network's loss over a grid of (a, w); no new random numbers are drawn here
Ag, Wg = np.meshgrid(np.linspace(-6, 6, 121), np.linspace(-6, 6, 121))
Zt = np.mean((Ag[..., None] * np.tanh(Wg[..., None] * x1) - y_pm) ** 2, axis=-1)
i_min = np.unravel_index(np.argmin(Zt), Zt.shape)
theta_min = np.array([Ag[i_min], Wg[i_min]])            # one minimum; its mirror image is -theta_min
neg = np.array(min_eig_net) < -1e-8                     # test points with a negative-curvature direction
print(f"grid minima at (a, w) = ({theta_min[0]:+.1f}, {theta_min[1]:+.1f}) and ({-theta_min[0]:+.1f}, {-theta_min[1]:+.1f}), "
      f"loss {tiny_net_loss(theta_min):.3f} at both")
print(f"loss at the midpoint (0, 0) = {tiny_net_loss(np.zeros(2)):.3f};  negative curvature at {neg.sum()} of 60 test points")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
lev = [0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.5, 5.0]          # darker grey = higher loss
ax[0].contourf(Ag, Wg, Zt, levels=[Zt.min()] + lev + [Zt.max()], cmap="Greys", alpha=0.45)
cs = ax[0].contour(Ag, Wg, Zt, levels=lev, colors="k", linewidths=0.6, alpha=0.6)
ax[0].clabel(cs, [0.3, 0.6, 1.0, 2.5], fmt="%.1f", fontsize=7)
ax[0].plot(points[neg, 0], points[neg, 1], "o", color="crimson", ms=5, label="negative curvature (filled)")
ax[0].plot(points[~neg, 0], points[~neg, 1], "o", mfc="white", mec="k", ms=5, label="no negative curvature (hollow)")
ax[0].plot([theta_min[0], -theta_min[0]], [theta_min[1], -theta_min[1]], "k*--", ms=12, lw=1, label="the two minima and their chord")
ax[0].set_title("Tiny tanh net: two mirror-image minima"); ax[0].set_xlabel("a (output weight)"); ax[0].set_ylabel("w (input weight)")
ax[0].legend(fontsize=7, loc="lower right"); ax[0].set_aspect("equal")

s = np.linspace(-0.5, 1.5, 201)
line_pts = np.outer(1 - s, theta_min) + np.outer(s, -theta_min)  # theta(s) = (1 - s) theta_min + s (-theta_min)
ax[1].plot(s, [tiny_net_loss(p) for p in line_pts], label="tiny-net loss along the line")
ax[1].plot([0, 1], [tiny_net_loss(theta_min), tiny_net_loss(-theta_min)], "ko--", lw=1, label="chord between the two minima")
ax[1].annotate(f"midpoint: loss {tiny_net_loss(np.zeros(2)):.3f}", (0.5, tiny_net_loss(np.zeros(2))),
               textcoords="offset points", xytext=(12, 8), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax[1].set_title("Along the straight line between the minima")
ax[1].set_xlabel("position s along the line (0 and 1 = the two minima)"); ax[1].set_ylabel("loss"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** the tiny network has two mirror-image minima, $(a, w) \approx (1.0, 2.6)$ and
$(-1.0, -2.6)$, because flipping both signs gives identical predictions ($-a\tanh(-wx) = a\tanh(wx)$,
the one-neuron version of swapping hidden units). Halfway between them the loss is $1.000$, against
$0.262$ at the ends, so the graph rises *above* the chord, which convexity forbids. The filled points,
43 of 60, have a negative Hessian eigenvalue: for this model non-convexity is the rule, not the exception.

**Try it yourself:** replace `tiny_net_loss` by the squared loss of a *linear* model
`np.mean((theta[0] + theta[1] * x1 - y_pm) ** 2)` — the smallest eigenvalue should become
non-negative everywhere.
""")

# ---------------------------------------------------------------- 3. gradient descent
nb.md(r"""
## 3. Gradient descent on an ill-conditioned bowl

Take $L(\theta) = \tfrac12 \theta^\top A \theta$ with eigenvalues $\mu = 1$ and $L = 10$, rotated by
30° so that the axes are not aligned with the coordinates. In the eigenbasis each coordinate is
multiplied by $(1 - \eta \lambda_i)$ per step, so

* every step size $\eta < 2/L = 0.2$ converges,
* $\eta^\star = 2/(L + \mu) = 2/11 \approx 0.18$ is the fastest constant step, with rate
  $(\kappa - 1)/(\kappa + 1) = 9/11$,
* $\eta > 0.2$ makes the stiff coordinate *grow* by a factor $\lvert 1 - \eta L\rvert > 1$ per step.

Watch the four paths below: a cautious step crawls along the valley floor, a near-optimal step
zig-zags but converges, and a step just above $2/L$ zig-zags *outwards*.
""")

nb.code(r"""
phi = np.deg2rad(30.0)
Q = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
A = Q @ np.diag([1.0, 10.0]) @ Q.T                      # eigenvalues 1 and 10, rotated axes
mu, L = np.linalg.eigvalsh(A)
print(f"eigenvalues: mu = {mu:.1f}, L = {L:.1f}, kappa = {L / mu:.1f}")
print(f"largest stable step 2/L = {2 / L:.3f}, optimal constant step 2/(L+mu) = {2 / (L + mu):.3f}, "
      f"optimal rate (kappa-1)/(kappa+1) = {(L - mu) / (L + mu):.3f}")

def quad_loss(th):
    return 0.5 * th @ A @ th

def gradient_descent(grad, x0, eta, steps):
    path = [np.asarray(x0, dtype=float)]
    for _ in range(steps):
        path.append(path[-1] - eta * grad(path[-1]))
    return np.array(path)

x0 = Q @ np.array([8.0, 2.0])        # 8 along the flat eigen-direction, 2 along the stiff one
G1, G2 = np.meshgrid(np.linspace(-10, 10, 201), np.linspace(-10, 10, 201))
Z = 0.5 * (A[0, 0] * G1 ** 2 + 2 * A[0, 1] * G1 * G2 + A[1, 1] * G2 ** 2)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, eta in zip(axes.ravel(), [0.05, 0.15, 0.19, 0.21]):
    path = gradient_descent(lambda th: A @ th, x0, eta, 12)
    ax.contour(G1, G2, Z, levels=np.logspace(-1, 2.6, 14), cmap="viridis", alpha=0.7)
    ax.plot(path[:, 0], path[:, 1], "o-", color="crimson", ms=3, lw=1)
    ax.plot(*x0, "k^", ms=8); ax.plot(0, 0, "k*", ms=10)
    ax.set_title(f"eta = {eta}: L(theta_12) = {quad_loss(path[-1]):.2f}  (L(theta_0) = {quad_loss(x0):.0f})"
                 + ("\nDIVERGES: eta > 2/L" if eta > 2 / L else ""))
    ax.set_xlabel("theta_1"); ax.set_ylabel("theta_2"); ax.set_aspect("equal")
    print(f"eta = {eta:.2f}: factors per step in the eigenbasis = {1 - eta * mu:+.2f} (flat), {1 - eta * L:+.2f} (stiff);"
          f"   loss after 12 steps = {quad_loss(path[-1]):.3f}")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Notice that for $\eta = 0.05$ the stiff direction is dealt with immediately but the flat direction
shrinks by only $0.95$ per step; at $\eta = 0.19$ the stiff coordinate flips sign each step (factor
$-0.9$) while the flat one shrinks by $0.81$ — the zig-zag; and at $\eta = 0.21$ the factor is
$-1.1$: the oscillation grows without bound (the loss climbs from 52 to 197 in 12 steps, even though
the flat coordinate is still shrinking — the black triangle marks the start). **No constant step size can be good for both
directions at once** — that is what the condition number measures, and it is why feature scaling,
momentum and adaptive methods exist.
""")

nb.md(r"""
### Convergence rate versus condition number

The theory predicts that GD with the optimal step needs about
$\ln(1/\varepsilon)\,/\,\ln\!\big(\tfrac{\kappa + 1}{\kappa - 1}\big) \approx \tfrac{\kappa}{2}\ln(1/\varepsilon)$
steps to shrink the error by a factor $\varepsilon$. We build random $6\times6$ quadratics with a
prescribed $\kappa$, run GD to $\varepsilon = 10^{-3}$, and measure the per-step contraction from the
last ten iterations.
""")

nb.code(r"""
def random_quadratic(kappa, d=6):
    Qd, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Qd @ np.diag(np.linspace(1.0, kappa, d)) @ Qd.T

kappas = np.array([5, 10, 20, 50, 100, 200, 500])
steps_measured, rate_measured, rate_theory, steps_theory = [], [], [], []
for kappa in kappas:
    Ak = random_quadratic(kappa)
    ev = np.linalg.eigvalsh(Ak); eta = 2 / (ev[-1] + ev[0])
    th = rng.normal(size=6); tol = 1e-3 * np.linalg.norm(th); errs = []
    for k in range(20000):
        th = th - eta * (Ak @ th); errs.append(np.linalg.norm(th))
        if errs[-1] < tol:
            break
    steps_measured.append(len(errs))
    tail = min(10, len(errs) - 1)
    rate_measured.append((errs[-1] / errs[-1 - tail]) ** (1 / tail))
    rate_theory.append((kappa - 1) / (kappa + 1))
    steps_theory.append(np.log(1e-3) / np.log((kappa - 1) / (kappa + 1)))
    print(f"kappa = {kappa:4d}: measured rate {rate_measured[-1]:.4f} vs theory {rate_theory[-1]:.4f};"
          f"  steps to 1e-3: {steps_measured[-1]:5d} (bound {steps_theory[-1]:7.1f})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogx(kappas, rate_measured, "o", label="measured from last 10 steps")
ax[0].semilogx(kappas, rate_theory, "-", label="(kappa-1)/(kappa+1)")
ax[0].set_title("Per-step contraction of GD with the optimal step"); ax[0].set_xlabel("condition number kappa"); ax[0].set_ylabel("rate"); ax[0].legend()
ax[1].loglog(kappas, steps_measured, "o", label="measured")
ax[1].loglog(kappas, steps_theory, "-", label="theory bound")
ax[1].set_title("Steps to reduce the error 1000x"); ax[1].set_xlabel("condition number kappa"); ax[1].set_ylabel("steps"); ax[1].legend()
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Notice that the measured contraction matches $(\kappa-1)/(\kappa+1)$ to three or four decimals once
the run is long enough to settle (for $\kappa = 5$ it ends after 14 steps), and the step count grows
*linearly* in $\kappa$ (slope one on the log–log plot). The measured counts sit below the
bound (by 3 to 28%) because under $\eta^\star$ only the two extreme eigen-directions ($\mu$ and $L$)
contract at the worst-case rate, and a random start puts only part of its length along them.

### Where the condition number comes from: feature scaling

The Hessian of the mean squared error of a linear model is $\tfrac{2}{n}X^\top X$, so $\kappa$ is set
by the *scale* of the columns of $X$. Fitting a temperature trend against the raw calendar year is a
famous way to get an unusable problem.
""")

nb.code(r"""
years = np.arange(1980, 2021).astype(float)
for name, feature in [("raw year", years),
                      ("centred year", years - years.mean()),
                      ("standardised year", (years - years.mean()) / years.std())]:
    Xf = np.c_[np.ones_like(years), feature]
    ev = np.linalg.eigvalsh(2 * Xf.T @ Xf / len(years))
    print(f"{name:18s}: Hessian eigenvalues {ev[0]:10.3e}, {ev[1]:10.3e}   kappa = {ev[1] / ev[0]:9.3g}   2/L = {2 / ev[1]:.3g}")
""")

nb.md(r"""
Same data, same model, same optimum — but the raw parametrisation needs a step below
$2.5\times10^{-7}$ and roughly $\kappa/2 \approx 6\times10^{10}$ of them for every factor $e$ of error
reduction, while the standardised one
is solved in a single step. **Standardise your inputs** is a theorem about $\kappa$, not a
superstition.

**Try it yourself:** in the contour experiment set the eigenvalues to `[1.0, 100.0]` (so $\kappa = 100$)
and re-run with `eta = 2 / (L + mu)`. How many steps does the flat direction now need?
""")

# ---------------------------------------------------------------- 4. SGD
nb.md(r"""
## 4. Stochastic gradient descent: unbiased, noisy, cheap

For linear regression on $n = 2000$ points with 5 correlated features, one full gradient costs
$2000 \times 5$ multiply-adds. A mini-batch of 8 costs $8\times5$ but is only an *estimate*:
$\mathbb{E}[g_B] = \nabla L$ (unbiased), with covariance $\tfrac{\Sigma}{b}\cdot\tfrac{n-b}{n-1} \approx \Sigma/b$. We first check both facts
directly, then compare full-batch GD with SGD at batch sizes 8, 32 and 128 using the **same number
of epochs** (passes over the data), so that every run sees the same amount of data.
""")

nb.code(r"""
n, d = 2000, 5
C = np.full((d, d), 0.6); np.fill_diagonal(C, 1.0)         # correlated features -> kappa ~ 9
X = rng.normal(size=(n, d)) @ np.linalg.cholesky(C).T
w_true = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
y = X @ w_true + 0.5 * rng.normal(size=n)

def mse(w):
    return np.mean((X @ w - y) ** 2)

def grad_batch(w, idx):
    return 2.0 * X[idx].T @ (X[idx] @ w - y[idx]) / len(idx)

w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
H = 2 * X.T @ X / n; ev = np.linalg.eigvalsh(H); L_lin = ev[-1]
print(f"Hessian eigenvalues {ev}  ->  kappa = {ev[-1] / ev[0]:.1f},  2/L = {2 / L_lin:.3f}")

# (a) unbiasedness and variance of the mini-batch gradient at a fixed point
w0 = np.zeros(d)
full_grad = grad_batch(w0, np.arange(n))
per_example = 2.0 * X * (X @ w0 - y)[:, None]                     # n x d matrix of per-example gradients
Sig = np.cov(per_example.T, bias=True)                            # population covariance Sigma of the per-example gradients
print(f"full gradient at w0 = 0: {full_grad},  norm {np.linalg.norm(full_grad):.2f}")
for b in [1, 8, 64]:
    est = np.array([grad_batch(w0, rng.choice(n, size=b, replace=False)) for _ in range(2000)])
    print(f"batch {b:3d}: mean of 2000 estimates - full gradient = {np.abs(est.mean(0) - full_grad).max():.3f} (max abs),"
          f"   trace of covariance = {np.trace(np.cov(est.T)):8.2f}   prediction tr(Sigma)/b * (n-b)/(n-1) = {np.trace(Sig) / b * (n - b) / (n - 1):8.2f}")
""")

nb.md(r"""
The same facts as a picture: below, 300 fresh estimates per batch size are drawn in the plane of two
gradient coordinates, each cloud with the two-standard-deviation ellipse of the predicted covariance
(it would hold about 86% of a Gaussian cloud), and on the right the total variance is measured over a
much wider range of batch sizes. A separate random generator is used so that the rest of the lesson
sees the same random numbers as before.
""")

nb.code(r"""
r_cloud = np.random.default_rng(11)                      # local generator: the global rng stream is left untouched
j, k = 0, 3                                              # the two gradient coordinates to draw
tt = np.linspace(0, 2 * np.pi, 200); circle = np.c_[np.cos(tt), np.sin(tt)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for b, colour in [(1, "tab:blue"), (8, "tab:orange"), (64, "tab:green")]:
    est = np.array([grad_batch(w0, r_cloud.choice(n, size=b, replace=False)) for _ in range(300)])
    ax[0].plot(est[:, j], est[:, k], ".", color=colour, ms=2, alpha=0.4)
    C2 = Sig[np.ix_([j, k], [j, k])] / b * (n - b) / (n - 1)
    ell = full_grad[[j, k]] + 2 * circle @ np.linalg.cholesky(C2).T         # the 2-standard-deviation ellipse
    ax[0].plot(ell[:, 0], ell[:, 1], color=colour, lw=1.5, zorder=4, label=f"batch {b}: 300 estimates, 2-sd ellipse")
    print(f"batch {b:2d}: largest distance of an estimate from the full gradient = {np.linalg.norm(est - full_grad, axis=1).max():5.1f}"
          f"  (in the plotted plane: {np.linalg.norm(est[:, [j, k]] - full_grad[[j, k]], axis=1).max():4.1f})")
ax[0].plot(*full_grad[[j, k]], "k*", ms=6, zorder=5, label="full gradient")
ax[0].set_title("Mini-batch gradients scatter around the full gradient")
ax[0].set_xlabel(f"gradient component {j + 1}"); ax[0].set_ylabel(f"gradient component {k + 1}")
ax[0].legend(fontsize=7, loc="upper left"); ax[0].set_aspect("equal")

bs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
tr_meas = [np.trace(np.cov(np.array([grad_batch(w0, r_cloud.choice(n, size=b, replace=False)) for _ in range(300)]).T)) for b in bs]
ax[1].loglog(bs, tr_meas, "o", label="measured (300 draws each)")
ax[1].loglog(bs, np.trace(Sig) / bs * (n - bs) / (n - 1), "-", label="tr(Sigma)/b * (n-b)/(n-1)")
ax[1].set_title("Total variance of the estimate falls like 1/b")
ax[1].set_xlabel("batch size b"); ax[1].set_ylabel("trace of covariance"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print("measured trace of covariance:", np.round(tr_meas, 2))
""")

nb.md(r"""
**Notice that** every cloud is centred on the star (the estimate is unbiased) and each ellipse is
$\sqrt8 \approx 2.8$ times smaller than the previous one. The printed means agree with the full
gradient (norm 5.6) to within the Monte-Carlo error of 2000 draws: the largest deviations, 0.07, 0.12
and 0.01, are about two standard errors or less. On the right the measured variance follows
$\operatorname{tr}\Sigma/b\cdot(n-b)/(n-1)$ over nearly three decades of $b$. The batch-1 cloud is also
heavy-tailed: single-example gradients land up to about 44 units from the star in this plane, and up
to 81 units away in all five coordinates, against a full gradient of norm 5.6 — which is why
per-example SGD needs small steps.

Now the training runs. We use $\eta = 1/L$ for every batch size; each run gets 20 epochs.
""")

nb.code(r"""
def sgd(batch_size, eta, epochs, seed, schedule=None):
    r = np.random.default_rng(seed)
    w = np.zeros(d); hist = []; t = 0
    steps_per_epoch = n // batch_size
    for _ in range(epochs):
        perm = r.permutation(n)
        for s in range(steps_per_epoch):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            lr = eta if schedule is None else schedule(t)
            w = w - lr * grad_batch(w, idx); t += 1
            hist.append(mse(w))
    return w, np.array(hist)

epochs, eta = 20, 1 / L_lin
runs = {}
for b in [8, 32, 128, n]:
    w_b, hist = runs[b] = sgd(b, eta, epochs, seed=1)
    spe = n // b
    print(f"batch {b:5d} ({spe:4d} steps/epoch): excess loss after epoch 1 = {hist[spe - 1] - mse(w_ols):.4f},"
          f"   mean excess over last epoch = {hist[-spe:].mean() - mse(w_ols):.2e}")

excess0 = mse(np.zeros(d)) - mse(w_ols)                 # common starting point at epoch 0
plt.figure(figsize=(8, 4))
for i_run, (b, (w_b, hist)) in enumerate(runs.items()):
    spe = n // b
    ep = np.arange(0, len(hist) + 1) / spe
    plt.semilogy(ep, np.r_[excess0, hist - mse(w_ols)], color=f"C{i_run}", lw=0.5, alpha=0.3)
    epoch_means = np.r_[excess0, hist.reshape(epochs, spe).mean(1) - mse(w_ols)]   # mean excess over each epoch
    plt.semilogy(np.arange(epochs + 1), epoch_means, "o-", color=f"C{i_run}", ms=3, lw=2,
                 label=f"batch {b}" + (" (full GD)" if b == n else ""))
plt.title("Excess loss L(w) - L(w_ols): per-step (faint) and per-epoch mean (bold), eta = 1/L")
plt.xlabel("epochs (passes over the data)"); plt.ylabel("excess loss"); plt.legend(loc="upper right"); plt.show()
""")

nb.md(r"""
Notice three things. (1) Per epoch, SGD wins by a landslide: full GD takes one step per pass and
after 20 passes is only where batch-8 SGD was after a fraction of *one* pass. (By epoch 20 full GD has
caught up with batch 8's noise floor, about $5\times10^{-2}$; batches 32 and 128 end about 6 and 28 times
lower.) (2) Smaller batches
plateau at a *higher noise floor*: the bold per-epoch means level off at about $5\times10^{-2}$
(batch 8), $8\times10^{-3}$ (batch 32) and $2\times10^{-3}$ (batch 128), because with a constant step
the iterates hover in a "noise ball" whose variance is $\propto \eta\sigma^2/b$. (3) Batch 128 is the
sweet spot here — low noise, 15 steps per epoch.

The cure for the noise floor is a **decaying step size**. Below we re-run batch 8 with a
$1/t$-type decay and a cosine schedule, and also verify the noise-ball formula
$\mathrm{Var}(\theta_\infty) = \eta\sigma^2/(\lambda(2 - \eta\lambda))$ on a 1-D quadratic with artificial
gradient noise.
""")

nb.code(r"""
spe = n // 8
_, hist_const = runs[8]
_, hist_inv = sgd(8, eta, epochs, seed=1, schedule=lambda t: eta / (1 + t / spe))
_, hist_cos = sgd(8, eta, epochs, seed=1, schedule=lambda t: eta * 0.5 * (1 + np.cos(np.pi * t / (epochs * spe))))
for name, h in [("constant", hist_const), ("1/t decay", hist_inv), ("cosine decay", hist_cos)]:
    print(f"batch 8, {name:13s}: excess loss at the last step = {h[-1] - mse(w_ols):.2e},"
          f"   mean excess over the last epoch = {h[-spe:].mean() - mse(w_ols):.2e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ep = np.arange(1, len(hist_const) + 1) / spe
for i_run, (name, h) in enumerate([("constant", hist_const), ("1/t decay", hist_inv), ("cosine decay", hist_cos)]):
    ax[0].semilogy(ep, h - mse(w_ols), color=f"C{i_run}", lw=0.5, alpha=0.3)
    ax[0].semilogy(np.arange(1, epochs + 1), h.reshape(epochs, spe).mean(1) - mse(w_ols), "o-", color=f"C{i_run}",
                   ms=3, lw=2, label=f"{name} (bold: per-epoch mean)")
ax[0].set_title("SGD (batch 8): the schedule shrinks the noise floor"); ax[0].set_xlabel("epochs"); ax[0].set_ylabel("excess loss"); ax[0].legend(fontsize=8)

# noise-ball check on L(theta) = lambda/2 theta^2 with gradient noise of variance sigma^2
lam, sigma = 1.0, 1.0
etas = np.array([0.05, 0.1, 0.2, 0.5, 1.0]); emp = []
for e in etas:
    th = 0.0; r = np.random.default_rng(3); samples = []
    for t in range(20000):
        th = th - e * (lam * th + sigma * r.normal())
        if t > 2000:
            samples.append(th)
    emp.append(np.var(samples))
ax[1].plot(etas, emp, "o", label="empirical variance of theta")
ax[1].plot(etas, etas * sigma ** 2 / (lam * (2 - etas * lam)), "-", label="eta sigma^2 / (lambda (2 - eta lambda))")
ax[1].plot(etas, etas * sigma ** 2 / (2 * lam), "--", label="small-eta approximation eta sigma^2 / (2 lambda)")
ax[1].set_title("The SGD noise ball grows with the step size"); ax[1].set_xlabel("eta"); ax[1].set_ylabel("stationary variance"); ax[1].legend()
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Notice that the noise-ball formula is exact (the dots sit on the solid curve), and that cosine decay
brings batch-8 SGD to below $10^{-4}$ of the optimum (a last-epoch mean of $7.7\times10^{-5}$, against
$5.1\times10^{-4}$ for the $1/t$ decay and $5.2\times10^{-2}$ for the constant step — nearly three orders
of magnitude better) using exactly the same gradients. Large-model training recipes (warm-up, then cosine
decay to 10% of the peak) are this experiment at scale.

**Try it yourself:** change the batch-8 constant step to `eta / 4`. The noise floor drops by about
4x, but how many more epochs does the first phase take?
""")

# ---------------------------------------------------------------- 5. momentum & adaptive
nb.md(r"""
## 5. Momentum, RMSProp and Adam from their update equations

All five optimisers below are a few lines each. We write them as functions of the gradient
$g_t$ with an explicit state so that the equations are visible:

| method | state update | step |
|---|---|---|
| GD | – | $\theta \leftarrow \theta - \eta g$ |
| heavy ball | $v \leftarrow \beta v + g$ | $\theta \leftarrow \theta - \eta v$ |
| Nesterov | $v \leftarrow \beta v + \nabla L(\theta - \eta\beta v)$ | $\theta \leftarrow \theta - \eta v$ |
| RMSProp | $s \leftarrow \rho s + (1-\rho) g^2$ | $\theta \leftarrow \theta - \eta\, g / (\sqrt{s} + \epsilon)$ |
| Adam | $m \leftarrow \beta_1 m + (1-\beta_1) g$, $v \leftarrow \beta_2 v + (1-\beta_2) g^2$ | $\theta \leftarrow \theta - \eta\, \hat m / (\sqrt{\hat v} + \epsilon)$ with $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$ |

The test function is Rosenbrock's banana $f(x, y) = (1 - x)^2 + 100 (y - x^2)^2$: a curved, narrow
valley whose floor has very low curvature along the valley and very high curvature across it — the
ill-conditioned bowl of Section 3, bent.
""")

nb.code(r"""
def rosenbrock(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])

def optimise(grad, x0, method, eta, steps, beta=0.9, rho=0.9, beta1=0.9, beta2=0.999, eps=1e-8):
    theta = np.asarray(x0, dtype=float)
    v = np.zeros_like(theta); s = np.zeros_like(theta); m = np.zeros_like(theta)
    path = [theta.copy()]
    for t in range(1, steps + 1):
        if method == "gd":
            theta = theta - eta * grad(theta)
        elif method == "momentum":
            v = beta * v + grad(theta)
            theta = theta - eta * v
        elif method == "nesterov":
            v = beta * v + grad(theta - eta * beta * v)
            theta = theta - eta * v
        elif method == "rmsprop":
            g = grad(theta); s = rho * s + (1 - rho) * g * g
            theta = theta - eta * g / (np.sqrt(s) + eps)
        elif method == "adam":
            g = grad(theta)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** t); v_hat = v / (1 - beta2 ** t)
            theta = theta - eta * m_hat / (np.sqrt(v_hat) + eps)
        path.append(theta.copy())
    return np.array(path)

x0 = (-1.5, 1.5)
settings = {"gd": 2e-3, "momentum": 2e-3, "nesterov": 1e-3, "rmsprop": 1e-2, "adam": 1e-1}
paths = {mth: optimise(rosenbrock_grad, x0, mth, eta, 2000) for mth, eta in settings.items()}

G1, G2 = np.meshgrid(np.linspace(-2, 2, 301), np.linspace(-1, 3, 301))
Z = (1 - G1) ** 2 + 100 * (G2 - G1 ** 2) ** 2
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].contour(G1, G2, Z, levels=np.logspace(-1, 3.5, 20), cmap="gray", alpha=0.5)
for mth, path in paths.items():
    ax[0].plot(path[:, 0], path[:, 1], lw=1, label=f"{mth} (eta={settings[mth]:g})")
    fvals = np.array([rosenbrock(p) for p in path])
    ax[1].semilogy(fvals, lw=1, label=mth)
    print(f"{mth:9s} eta = {settings[mth]:5g}: f after 500 steps = {fvals[500]:9.2e},  after 2000 = {fvals[-1]:9.2e},  final point = {path[-1]}")
ax[0].plot(*x0, "k^", ms=8, label="start (-1.5, 1.5)"); ax[0].plot(1, 1, "k*", ms=10, label="minimum (1, 1)")
ax[0].set_title("Trajectories on the Rosenbrock valley"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[0].legend(fontsize=7, loc="upper right", ncol=2)          # the band y > 2 holds no trajectory
ax[1].set_title("f along the trajectory"); ax[1].set_xlabel("step"); ax[1].set_ylabel("f(x, y)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Notice how plain GD creeps along the valley floor: its step is limited by the *across-valley*
curvature ($\approx 2/L$) and the *along-valley* progress per step is tiny. Momentum accumulates
along-valley velocity (effective step $\eta/(1-\beta) = 10\eta$) and reaches $10^{-16}$. Adam and
RMSProp normalise each coordinate by its own gradient scale, so they take steps of size $\approx\eta$
regardless of the curvature; Adam with $\eta = 0.1$ is fast here, with $\eta = 0.01$ it would be the
slowest of all — the learning rate still matters.

**Try it yourself:** set `momentum` to `eta = 3e-3`. It diverges within a handful of steps — the
stability limit of heavy ball is about $2(1+\beta)/L$, roughly twice GD's $2/L$. Momentum buys speed
along the valley, not a much larger stable step.

### Bias correction, and why Adam's first steps have size ≈ η

Adam's running averages start at zero, so for small $t$ the raw $m_t$ and $v_t$ are biased towards
zero — but not by the same factor. Without correction the first step would be
$\eta\,(1-\beta_1) g / \sqrt{(1-\beta_2) g^2} = \eta \cdot 0.1/\sqrt{0.001} \approx 3.16\,\eta$. With correction it is
exactly $\eta\,\mathrm{sign}(g)$ (ignoring $\epsilon$), whatever the magnitude of $g$.
""")

nb.code(r"""
beta1, beta2 = 0.9, 0.999
for g in [np.array([0.1, -100.0]), np.array([1e-4, 1e4])]:
    m1 = (1 - beta1) * g; v1 = (1 - beta2) * g * g
    print(f"g = {g}:  step/eta without bias correction = {m1 / np.sqrt(v1)},  with correction = {(m1 / (1 - beta1)) / np.sqrt(v1 / (1 - beta2))}")

# heavy-ball velocity under a constant gradient converges to g/(1-beta)
v = 0.0; vs = []
for t in range(40):
    v = 0.9 * v + 1.0; vs.append(v)
plt.plot(range(1, 41), vs, "o-", ms=3); plt.axhline(10, color="gray", ls="--")
plt.title("Heavy-ball velocity under a constant unit gradient (beta = 0.9) -> 1/(1-beta) = 10")
plt.xlabel("step"); plt.ylabel("v_t"); plt.show()
""")

nb.md(r"""
The printout covers only step 1. Below we follow $\lvert\text{step}\rvert/\eta$ for 3000 steps, with
and without the bias correction, once for a constant gradient $g = 0.1$ and once for a noisy gradient
whose mean is still $0.1$ but whose standard deviation is $0.3$.
""")

nb.code(r"""
t_ad = np.arange(1, 3001)
r_ad = np.random.default_rng(4)                          # local generator for the noisy gradient
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i_ax, (label, gs) in enumerate([("constant gradient g = 0.1", np.full(len(t_ad), 0.1)),
                                    ("noisy gradient 0.1 + N(0, 0.3^2)", 0.1 + 0.3 * r_ad.normal(size=len(t_ad)))]):
    m_a = v_a = 0.0; raw, corr = [], []
    for t_k, g_k in zip(t_ad, gs):
        m_a = beta1 * m_a + (1 - beta1) * g_k; v_a = beta2 * v_a + (1 - beta2) * g_k * g_k
        raw.append(abs(m_a) / np.sqrt(v_a))
        corr.append(abs(m_a / (1 - beta1 ** t_k)) / np.sqrt(v_a / (1 - beta2 ** t_k)))
    ax[i_ax].semilogx(t_ad, raw, label="without bias correction")
    ax[i_ax].semilogx(t_ad, corr, label="with bias correction")
    ax[i_ax].axhline(1, color="gray", ls="--", lw=0.8)
    ax[i_ax].set_title(f"|Adam step| / eta, {label}", fontsize=10)
    ax[i_ax].set_xlabel("step t (log scale)"); ax[i_ax].set_ylabel("|step| / eta"); ax[i_ax].legend(fontsize=8)
    if i_ax == 0:
        raw = np.array(raw)
        print(f"constant gradient, no correction: step 1 = {raw[0]:.2f} eta, peak {raw.max():.2f} eta at t = {int(np.argmax(raw)) + 1},"
              f" t = 100: {raw[99]:.2f} eta, t = 1000: {raw[999]:.2f} eta;  with correction: between {min(corr):.4f} and {max(corr):.4f} eta")
    else:
        print(f"noisy gradient, with correction: mean |step|/eta over the last 1000 steps = {np.mean(corr[-1000:]):.2f}"
              f"   (|mean g| / rms(g) = {abs(gs.mean()) / np.sqrt(np.mean(gs ** 2)):.2f})")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** without bias correction the first step is $3.16\eta$ and the steps keep *growing*, to
about $6.6\eta$ at $t = 12$, and are still $3.2\eta$ at $t = 100$ — an anti-warm-up exactly when the
iterate is least trustworthy. With the correction the step is exactly $\eta$ for a consistent gradient.
Under noise (right) even the corrected step is only about $\eta\,\lvert\text{mean } g\rvert/\text{rms}(g)
\approx 0.3\eta$: Adam slows down by itself when the gradient direction is unreliable.

### Why momentum helps: the rotated bowl again, with κ = 100

The velocity plot above showed that under a *constant* gradient the heavy-ball velocity builds up to
$g/(1-\beta)$. On the bowl of Section 3, made harder with eigenvalues $1$ and $100$, we now compare GD at
its optimal step $2/(L+\mu)$, the default heavy ball ($\beta = 0.9$, $\eta = 1/L$) and the tuned heavy
ball ($\beta = \big(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^2 = (9/11)^2$, $\eta = \big(\tfrac{2}{\sqrt L+\sqrt\mu}\big)^2$),
and look at the default heavy ball's velocity in the eigenbasis.
""")

nb.code(r"""
A100 = Q @ np.diag([1.0, 100.0]) @ Q.T                    # same rotation as Section 3, kappa = 100
mu_b, L_b = 1.0, 100.0
th0 = Q @ np.array([8.0, 2.0]); steps_m = 120
beta_t = ((np.sqrt(L_b) - np.sqrt(mu_b)) / (np.sqrt(L_b) + np.sqrt(mu_b))) ** 2
eta_t = (2 / (np.sqrt(L_b) + np.sqrt(mu_b))) ** 2
gd_p = optimise(lambda th: A100 @ th, th0, "gd", 2 / (L_b + mu_b), steps_m)
hb_p = optimise(lambda th: A100 @ th, th0, "momentum", 1 / L_b, steps_m, beta=0.9)
hbt_p = optimise(lambda th: A100 @ th, th0, "momentum", eta_t, steps_m, beta=beta_t)

# the heavy ball's velocity v_t = beta v_(t-1) + g_t, recomputed from its path and rotated into the eigenbasis (flat, stiff)
v_hb = np.zeros(2); v_eig, g_eig = [], []
for th in hb_p[:-1]:
    g_hb = A100 @ th; v_hb = 0.9 * v_hb + g_hb
    v_eig.append(Q.T @ v_hb); g_eig.append(Q.T @ g_hb)
v_eig, g_eig = np.array(v_eig), np.array(g_eig)
for name, pth in [("GD, eta = 2/(L+mu)", gd_p), ("heavy ball, beta = 0.9", hb_p), (f"tuned heavy ball, beta = {beta_t:.3f}", hbt_p)]:
    print(f"{name:32s}: ||theta_60|| = {np.linalg.norm(pth[60]):.2e},  ||theta_120|| = {np.linalg.norm(pth[120]):.2e}")
print(f"flat component:  gradient never exceeds {np.abs(g_eig[:, 0]).max():.1f}, velocity peaks at {v_eig[:, 0].max():.1f} (step {int(np.argmax(v_eig[:, 0])) + 1})")
print(f"stiff component: gradient peaks at {np.abs(g_eig[:, 1]).max():.0f}, velocity peaks at {np.abs(v_eig[:, 1]).max():.0f}")

Gx, Gy = np.meshgrid(np.linspace(-9, 9, 181), np.linspace(-9, 9, 181))
Zb = 0.5 * (A100[0, 0] * Gx ** 2 + 2 * A100[0, 1] * Gx * Gy + A100[1, 1] * Gy ** 2)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].contour(Gx, Gy, Zb, levels=np.logspace(-1, 3.5, 14), cmap="viridis", alpha=0.6)
ax[0].plot(gd_p[:61, 0], gd_p[:61, 1], "o-", ms=2, lw=1, label="GD, eta = 2/(L+mu)")
ax[0].plot(hb_p[:61, 0], hb_p[:61, 1], "s-", ms=2, lw=1, label="heavy ball, beta = 0.9, eta = 1/L")
ax[0].plot(*th0, "k^", ms=8, label="start"); ax[0].plot(0, 0, "k*", ms=10, label="minimum")
ax[0].set_title("kappa = 100: first 60 steps"); ax[0].set_xlabel("theta_1"); ax[0].set_ylabel("theta_2")
ax[0].legend(fontsize=7, loc="lower right"); ax[0].set_aspect("equal")
k_m = np.arange(steps_m + 1)
ax[1].semilogy(k_m, np.linalg.norm(gd_p, axis=1), label="GD: rate 0.980")
ax[1].semilogy(k_m, np.linalg.norm(hb_p, axis=1), label="heavy ball beta 0.9: rate sqrt(beta) = 0.949")
ax[1].semilogy(k_m, np.linalg.norm(hbt_p, axis=1), label=f"tuned heavy ball (beta {beta_t:.2f}): rate 0.818")
ax[1].set_title("Distance to the minimum"); ax[1].set_xlabel("step"); ax[1].set_ylabel("||theta_k||"); ax[1].legend(fontsize=7)
t_m = np.arange(1, 41)
ax[2].plot(t_m, g_eig[:40, 0], "o-", ms=3, lw=1, color="tab:blue", label="gradient, flat component")
ax[2].plot(t_m, v_eig[:40, 0], "s-", ms=3, lw=1.5, color="tab:orange", label="velocity, flat component")
ax[2].plot(t_m, g_eig[:40, 1] / 10, "o:", ms=2, lw=1, color="tab:blue", label="gradient, stiff component / 10")
ax[2].plot(t_m, v_eig[:40, 1] / 10, "s:", ms=2, lw=1, color="tab:orange", label="velocity, stiff component / 10")
ax[2].axhline(0, color="gray", lw=0.8)
ax[2].set_title("Heavy ball: flat pushes add up, stiff kicks do not"); ax[2].set_xlabel("step")
ax[2].set_ylabel("component in the eigenbasis"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** after 60 steps GD has only reached $\lVert\theta\rVert = 2.5$, the default heavy ball
($\beta = 0.9$, $\eta = 1/L$) $0.07$ and the tuned heavy ball $1.4\times10^{-3}$. In the middle panel GD
falls along a straight line with slope $0.980$ per step; the default heavy ball oscillates under an
envelope that shrinks by $\sqrt\beta = 0.949$ per step (step 60 happens to fall in one of its dips:
between steps 50 and 70 it ranges from 0.04 to 0.48); and the tuned heavy ball falls at $0.818$ per step,
times the slowly growing factor $k$ of its critically damped modes. The right panel shows why: along the flat direction successive gradients
share a sign, so the velocity piles up to 44, more than five times the largest flat gradient (8.0); along the stiff
direction the gradient keeps changing sign and the velocity never exceeds a single gradient (both
peak at 200). Momentum is a large step along the valley and a small one across it — the $\sqrt\kappa$
instead of $\kappa$ of the guide.
""")

nb.md(r"""
### The same optimisers on a real problem: logistic regression

The breast-cancer data (569 tumours, 30 features) with standardised features and a small ridge
term $\lambda = 10^{-2}$. The loss is convex, so every method converges to the *same* optimum; what
differs is how fast. We log the loss for 300 full-batch steps of each method.
""")

nb.code(r"""
data = load_breast_cancer()
X_raw, y_bc = data.data, data.target.astype(float)
X_std = (X_raw - X_raw.mean(0)) / X_raw.std(0)
Xb = np.c_[np.ones(len(X_std)), X_std]                 # 569 x 31 with an intercept column
lam_ridge = 1e-2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(w, X=Xb, y=y_bc):
    z = X @ w
    return np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)) + 0.5 * lam_ridge * (w[1:] @ w[1:])

def log_grad(w, X=Xb, y=y_bc):
    g = X.T @ (sigmoid(X @ w) - y) / len(y)
    g[1:] += lam_ridge * w[1:]
    return g

def log_hess(w, X=Xb, y=y_bc):
    p = sigmoid(X @ w); s = p * (1 - p)
    Hm = (X * s[:, None]).T @ X / len(y)
    Hm[1:, 1:] += lam_ridge * np.eye(X.shape[1] - 1)
    return Hm

ev0 = np.linalg.eigvalsh(log_hess(np.zeros(Xb.shape[1])))
print(f"Hessian at w = 0 (standardised): L = {ev0[-1]:.3f}, mu = {ev0[0]:.2e}, kappa = {ev0[-1] / ev0[0]:.3g}, 2/L = {2 / ev0[-1]:.3f}")
ev_raw = np.linalg.eigvalsh(0.25 * np.c_[np.ones(len(X_raw)), X_raw].T @ np.c_[np.ones(len(X_raw)), X_raw] / len(X_raw))
print(f"Hessian at w = 0 (raw features) : L = {ev_raw[-1]:.3g},  2/L = {2 / ev_raw[-1]:.2e}   <- why unscaled GD seems 'broken'")

# the optimum L(w*) itself, from 10 Newton steps (Section 6 explains why this converges so fast)
w_opt = np.zeros(Xb.shape[1])
for _ in range(10):
    w_opt = w_opt - np.linalg.solve(log_hess(w_opt), log_grad(w_opt))
L_star = log_loss(w_opt)
ev_opt = np.linalg.eigvalsh(log_hess(w_opt))
print(f"optimum: L(w*) = {L_star:.10f};  Hessian at w*: L = {ev_opt[-1]:.3f}, mu = {ev_opt[0]:.2e}, kappa = {ev_opt[-1] / ev_opt[0]:.3g}")

lr_settings = {"gd": 0.5, "momentum": 0.3, "nesterov": 0.3, "rmsprop": 0.01, "adam": 0.05}
w0 = np.zeros(Xb.shape[1])
plt.figure(figsize=(8, 4))
for mth, eta in lr_settings.items():
    path = optimise(log_grad, w0, mth, eta, 300)
    losses = np.array([log_loss(w) for w in path])
    acc = np.mean((Xb @ path[-1] > 0) == (y_bc == 1))
    print(f"{mth:9s} eta = {eta:5g}: loss at step 50 = {losses[50]:.4f}, 150 = {losses[150]:.4f};  excess at step 300 = {losses[300] - L_star:.1e};  train accuracy = {acc:.4f}")
    plt.semilogy(np.maximum(losses - L_star, 1e-16), lw=1, label=f"{mth} (eta={eta:g})")
plt.title("Logistic regression (breast cancer, standardised): excess loss above the optimum")
plt.xlabel("step"); plt.ylabel("L(w) - L(w*)  (floored at 1e-16)"); plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
Notice that the condition number at $w = 0$ is still $\approx 330$ after standardisation — and it is
set by the ridge term: without it $\mu \approx 3\times10^{-5}$ and $\kappa \approx 10^5$, because the 30
features are strongly correlated. That $\kappa$ is measured at $w = 0$, where the logistic curvature
$p(1-p) = 1/4$ is largest; at the optimum the Hessian is about 15 times flatter ($L = 0.22$,
$\kappa \approx 23$), so $\eta = 0.5$, a step that is safe everywhere, is very conservative there and plain
GD is slow (excess loss $3\times10^{-5}$ after 300 steps). Momentum and Nesterov converge linearly,
reaching $2\times10^{-14}$ and $3\times10^{-16}$ at step 300 (Nesterov reaches the floating-point floor of
about $10^{-17}$ some 20 steps later, momentum some 60–80 steps later), and Adam reaches $4\times10^{-9}$.
RMSProp with $\rho = 0.9$ and a constant $\eta$ never settles: its $\sqrt{s}$ forgets within about ten
steps, so $\sqrt{s} \approx \lvert g\rvert$ and the step $\eta\,g/\sqrt{s}$ stays $\approx\eta$ even as
$g \to 0$, and it hovers about $10^{-4}$ above the optimum. Adam's $\beta_2 = 0.999$ remembers the larger
early gradients, so its steps shrink as $g$ does (RMSProp with $\rho = 0.999$ does the same over these
300 steps, but only until its memory of about 1000 steps runs out); a decaying schedule is the general cure. On the raw features the largest stable
step would be $5\times10^{-6}$ — the usual experience of "GD does not work" on unscaled data.
""")

# ---------------------------------------------------------------- 6. Newton
nb.md(r"""
## 6. Newton's method: curvature-aware steps and quadratic convergence

Minimising the second-order Taylor model $L(\theta + p) \approx L + g^\top p + \tfrac12 p^\top H p$ over $p$
gives the Newton step $p = -H^{-1} g$. Near a minimum with a positive-definite, Lipschitz Hessian the
error is *squared* at every iteration. We show this on the logistic regression above (where
$H = \tfrac1n X^\top S X + \lambda I$ is available in closed form) and verify against
`sklearn.linear_model.LogisticRegression`, which solves the same problem with L-BFGS
(`C = 1/(n\lambda)` converts our penalty into scikit-learn's convention).
""")

nb.code(r"""
w = np.zeros(Xb.shape[1]); newton_losses = [log_loss(w)]; grad_norms = [np.linalg.norm(log_grad(w))]
for k in range(8):
    w = w - np.linalg.solve(log_hess(w), log_grad(w))
    newton_losses.append(log_loss(w)); grad_norms.append(np.linalg.norm(log_grad(w)))
w_newton = w.copy()
print("Newton: gradient norm per iteration")
for k, gnorm in enumerate(grad_norms):
    print(f"  iteration {k}: ||grad|| = {gnorm:.2e}   loss = {newton_losses[k]:.6f}")

clf = LogisticRegression(C=1 / (lam_ridge * len(y_bc)), tol=1e-10, max_iter=10000).fit(X_std, y_bc)
w_sklearn = np.r_[clf.intercept_, clf.coef_.ravel()]
print(f"\nmax |w_newton - w_sklearn(lbfgs)| = {np.abs(w_newton - w_sklearn).max():.2e}   (sklearn needed {clf.n_iter_[0]} L-BFGS iterations)")

gd_path = optimise(log_grad, np.zeros(Xb.shape[1]), "gd", 0.5, 300)
gd_losses = np.array([log_loss(v) for v in gd_path])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(gd_losses - newton_losses[-1], label="GD, eta = 0.5")
ax[0].semilogy(np.array(newton_losses[:-1]) - newton_losses[-1], "o-", label="Newton, iterations 0-7 (iteration 8 is L*)")
ax[0].set_title("Excess loss: Newton vs GD on logistic regression"); ax[0].set_xlabel("iteration"); ax[0].set_ylabel("L - L*  (L* = Newton iteration 8)"); ax[0].set_xlim(-1, 60); ax[0].legend()
ax[1].semilogy(grad_norms, "o-"); ax[1].set_title("Newton: ||grad|| per iteration (digits double)"); ax[1].set_xlabel("iteration"); ax[1].set_ylabel("||grad||")
plt.tight_layout(); plt.show()

# Newton on f(x) = x^3/3 - 2x  (f' = x^2 - 2, f'' = 2x): minimum at +sqrt(2), maximum at -sqrt(2)
for x_start in [1.0, -1.0]:
    xk = x_start; print(f"start x0 = {x_start:+.1f}:", end="")
    for k in range(5):
        xk = xk - (xk ** 2 - 2) / (2 * xk)
        print(f"  {xk:+.10f}", end="")
    print(f"   -> f''(x) = {2 * xk:+.2f} ({'minimum' if xk > 0 else 'MAXIMUM'})")
print(f"errors from x0 = 1 are squared each step: ratio e_(k+1)/e_k^2 -> 1/(2 sqrt 2) = {1 / (2 * np.sqrt(2)):.4f}")
""")

nb.md(r"""
Notice the signature of quadratic convergence: the exponent of the gradient norm goes
$0, -1, -1, -2, -2, -3, -5, -7, -13$ — once the iterate is close, the number of correct digits doubles
per step. Eight $31\times31$ solves beat 300 gradient steps. The catch is cost: forming $H$ is
$O(nd^2)$ and solving is $O(d^3)$, impossible for $d = 10^8$; and far from the optimum $H$ may be
indefinite. The 1-D run at the end of the cell shows the other caveat: from $x_0 = -1$ Newton
converges, just as fast, to the *maximum* of $f$ — it solves $f' = 0$ and does not know the difference.

**Try it yourself:** replace `np.linalg.solve(log_hess(w), log_grad(w))` by a *damped* step
`np.linalg.solve(log_hess(w) + 1.0 * np.eye(31), log_grad(w))`. Convergence becomes linear instead of
quadratic — this is the Levenberg–Marquardt idea, trading speed for robustness far from the optimum.
""")


# ---------------------------------------------------------------- 7. Lagrange
nb.md(r"""
## 7. Constrained optimisation: the tangency picture

Minimise $f(x, y) = x^2 + 2y^2$ subject to $h(x, y) = x + y - 3 = 0$. At the constrained optimum the
level set of $f$ is *tangent* to the constraint line, so $\nabla f = -\lambda \nabla h$. Solving the
linear KKT system

$$
\begin{pmatrix} 2 & 0 & 1 \\ 0 & 4 & 1 \\ 1 & 1 & 0 \end{pmatrix}
\begin{pmatrix} x \\ y \\ \lambda \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 3 \end{pmatrix}
$$

gives $(x^\star, y^\star, \lambda) = (2, 1, -4)$ and $f^\star = 6$. We verify the shadow-price
interpretation $\mathrm{d}f^\star/\mathrm{d}b = -\lambda$ by nudging the constraint level, and
we solve the same problem by **projected gradient descent** (step, then project back onto the line).
""")

nb.code(r"""
K = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 1.0], [1.0, 1.0, 0.0]])
x_s, y_s, lam_s = np.linalg.solve(K, np.array([0.0, 0.0, 3.0]))
f_con = lambda p: p[0] ** 2 + 2 * p[1] ** 2
print(f"KKT solution: x* = {x_s:.3f}, y* = {y_s:.3f}, lambda = {lam_s:.3f}, f* = {f_con((x_s, y_s)):.3f}")
print(f"grad f = {np.array([2 * x_s, 4 * y_s])},  grad h = [1. 1.]  -> parallel (ratio {2 * x_s:.1f})")

def f_star(b):          # optimal value as a function of the constraint level  x + y = b
    sol = np.linalg.solve(K, np.array([0.0, 0.0, b])); return f_con(sol[:2])
print(f"shadow price: (f*(3.01) - f*(2.99)) / 0.02 = {(f_star(3.01) - f_star(2.99)) / 0.02:.3f}  vs  -lambda = {-lam_s:.3f}")

# projected gradient descent: step, then project onto the line x + y = 3
a_vec = np.array([1.0, 1.0])
def project_onto_line(p, a=a_vec, b=3.0):
    return p - (a @ p - b) / (a @ a) * a
p = np.array([-1.0, 4.0]); pgd_path = [p.copy()]
for _ in range(40):
    p = project_onto_line(p - 0.2 * np.array([2 * p[0], 4 * p[1]])); pgd_path.append(p.copy())
pgd_path = np.array(pgd_path)
print(f"projected GD after 40 steps: {pgd_path[-1]}")

G1, G2 = np.meshgrid(np.linspace(-1, 4, 301), np.linspace(-1, 4, 301))
plt.figure(figsize=(6, 5.5))
cs = plt.contour(G1, G2, G1 ** 2 + 2 * G2 ** 2, levels=[1, 3, 10, 15, 22, 30], cmap="viridis")
plt.clabel(cs, fmt="%.0f", fontsize=7)
plt.contour(G1, G2, G1 ** 2 + 2 * G2 ** 2, levels=[6], colors="k", linestyles="--", linewidths=1.5)
plt.plot([], [], "k--", lw=1.5, label="level set f = 6 (tangent)")          # legend entry for the contour above
xx = np.linspace(-1, 4, 50); plt.plot(xx, 3 - xx, "k-", lw=1.5, label="constraint x + y = 3")
plt.plot(pgd_path[:, 0], pgd_path[:, 1], "o", color="crimson", ms=5, lw=0, zorder=5, label="projected GD iterates")
# grad f = (4, 4) drawn at half length, grad h = (1, 1) at full length on top: same direction, different lengths
plt.quiver(x_s, y_s, 2 * x_s, 4 * y_s, angles="xy", scale_units="xy", scale=2, width=0.012, color="tab:blue",
           label="grad f at x* = (4, 4), drawn at half length")
plt.quiver(x_s, y_s, 1, 1, angles="xy", scale_units="xy", scale=1, width=0.006, color="k", zorder=4, label="grad h = (1, 1)")
plt.plot(x_s, y_s, "r*", ms=12, zorder=6)
plt.title("At the optimum the level set of f is tangent to the constraint"); plt.xlabel("x"); plt.ylabel("y")
plt.legend(fontsize=8, loc="lower left"); plt.gca().set_aspect("equal"); plt.show()
""")

nb.md(r"""
Notice that the dashed level set $f = 6$ touches the constraint line at exactly one point, and the two
gradient arrows there point the same way (the blue one is drawn at half its length, so it is still twice
as long as the black one): $\nabla f = (4, 4) = 4\,\nabla h$, so $\lambda = -4$ in the
convention $\nabla f + \lambda\nabla h = 0$. The shadow price checks out: moving the constraint level
from 3 to 3.01 raises the minimal cost by about $0.04 = -\lambda \times 0.01$. Projected GD finds the same point
without ever forming the Lagrangian — the approach used when the constraint set is simple (a
box, a sphere, the probability simplex).

**Try it yourself:** change the constraint to an *inequality* $x + y \ge 3$ by projecting only when
$x + y < 3$. Start at $(3, 3)$ — the constraint is inactive there and projected GD walks to the
unconstrained minimum $(0, 0)$ until it hits the line, then slides along it to $(2, 1)$.
""")

# ---------------------------------------------------------------- 8. diagnostics
nb.md(r"""
## 8. Practical diagnostics: schedules, the range test, clipping, early stopping

Four tools, one figure each.

1. **Learning-rate schedules** — step decay, cosine, and warm-up + cosine, as functions of the step.
2. **The learning-rate range test** — run GD while increasing $\eta$ geometrically and watch the loss;
   the knee where it starts climbing is $2/L$, so the recommended $\eta$ is a few times less.
3. **Gradient clipping by global norm** — $g \leftarrow g\,\min(1, c/\lVert g\rVert)$ keeps the direction
   and bounds the step.
4. **Early stopping** — train on 120 of the 569 tumours and validate on the rest: the training loss
   keeps falling, the validation loss turns around.
""")

nb.code(r"""
T, T_warm, eta_max = 1000, 100, 0.1
t = np.arange(T + 1)
step_decay = eta_max * 0.5 ** np.minimum(t // 250, 3)       # four levels; t = T stays on the last one
cosine = 0.5 * eta_max * (1 + np.cos(np.pi * t / T))
warm_cos = np.where(t < T_warm, eta_max * t / T_warm, 0.5 * eta_max * (1 + np.cos(np.pi * np.clip(t - T_warm, 0, None) / (T - T_warm))))
plt.plot(t, step_decay, label="step decay (halve every 250)"); plt.plot(t, cosine, label="cosine"); plt.plot(t, warm_cos, label="linear warm-up (100) + cosine")
plt.title("Learning-rate schedules"); plt.xlabel("step t"); plt.ylabel("eta_t"); plt.legend(); plt.show()
print(f"cosine schedule: eta at T/4 = {0.5 * (1 + np.cos(np.pi / 4)):.3f} eta_max,  at T/2 = {0.5:.3f} eta_max,  at 3T/4 = {0.5 * (1 + np.cos(3 * np.pi / 4)):.3f} eta_max")

# learning-rate range test on the linear-regression problem of Section 4 (true 2/L is known),
# run the realistic way: with mini-batches of 32, one step per candidate eta, eta growing geometrically
etas_test = np.logspace(-3, 0, 300)
r = np.random.default_rng(0); w = np.zeros(d); range_losses = []
for e in etas_test:
    idx = r.choice(n, size=32, replace=False)
    w = w - e * grad_batch(w, idx); range_losses.append(mse(w))
range_losses = np.array(range_losses)
i_min = int(np.argmin(range_losses))
knee = etas_test[i_min + int(np.argmax(range_losses[i_min:] > 2 * range_losses.min()))]
print(f"range test: lowest loss at eta = {etas_test[i_min]:.3f}; loss has doubled by eta = {knee:.3f};  theory 2/L = {2 / L_lin:.3f};  a safe starting point is 1/L = {1 / L_lin:.3f}")
plt.loglog(etas_test, range_losses, label="loss after each step")
plt.axvline(1 / L_lin, color="tab:green", ls="-.", label="1/L (suggested start)")
plt.axvline(knee, color="crimson", ls=":", label="knee (loss doubled)")
plt.axvline(2 / L_lin, color="gray", ls="--", label="2/L")
plt.ylim(0.1, 100); plt.title("Learning-rate range test: increase eta geometrically, watch the loss")
plt.xlabel("eta (increasing over the run, log scale)"); plt.ylabel("full loss after the step (log scale)"); plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
Notice that the loss falls from its starting value, is flat and minimal over a wide band, and shoots
up by orders of magnitude close to $2/L$ — the range test *measures* the curvature constant $L$ you would
otherwise have to guess. The green line, $1/L$, sits inside the flat band, a safe distance below the knee. The mini-batch noise is
what makes the knee visible: with full-batch GD on an already-converged iterate, an unstable step
needs several iterations before the error in the stiff direction has grown enough to see, so the
knee appears late. On a non-quadratic loss the knee is less sharp, but the recipe is the same:
pick a value a few times below it.
""")

nb.code(r"""
def clip_by_global_norm(g, c):
    norm = np.linalg.norm(g)
    return g * min(1.0, c / norm) if norm > 0 else g

g = np.array([3.0, 4.0])
print(f"g = {g}, ||g|| = {np.linalg.norm(g):.1f}  ->  clipped at c = 1: {clip_by_global_norm(g, 1.0)}  (norm {np.linalg.norm(clip_by_global_norm(g, 1.0)):.1f}, same direction)")

# clipping rescues a too-large step on the rotated quadratic of Section 3
for eta in [0.5, 1.0]:
    for c in [None, 2.0]:
        th = Q @ np.array([8.0, 2.0]); diverged = False
        for _ in range(200):
            gq = A @ th
            th = th - eta * (gq if c is None else clip_by_global_norm(gq, c))
            if not np.isfinite(th).all() or np.linalg.norm(th) > 1e6:
                diverged = True; break
        print(f"eta = {eta}, clip = {c}: {'DIVERGED' if diverged else f'||theta_200|| = {np.linalg.norm(th):.3f}'}")
""")

nb.md(r"""
Four printed lines do not show *where* the clipped iterate goes. We draw the run with $\eta = 0.5$
and $c = 2$ next to the unclipped one, and follow both eigen-coordinates (flat, stiff) of the clipped
runs for $\eta = 0.5$ and $\eta = 1$.
""")

nb.code(r"""
def run_clipped(eta, c, steps):
    th = Q @ np.array([8.0, 2.0]); path_c = [th]
    for _ in range(steps):
        gq = A @ th
        th = th - eta * (gq if c is None else clip_by_global_norm(gq, c)); path_c.append(th)
        if np.linalg.norm(th) > 1e3:                    # unclipped runs: stop once they have clearly left
            break
    return np.array(path_c)

pc, pn = run_clipped(0.5, 2.0, 40), run_clipped(0.5, None, 40)
print("unclipped, eta = 0.5: ||theta_k|| for k = 0..5 =", np.round(np.linalg.norm(pn[:6], axis=1), 1))
for eta in [0.5, 1.0]:
    z = run_clipped(eta, 2.0, 200) @ Q                  # rows are (flat, stiff) eigen-coordinates
    print(f"clipped, eta = {eta}: flat coordinate -> {z[-1, 0]:+.1e};  stiff coordinate bounces between "
          f"{z[-2, 1]:+.3f} and {z[-1, 1]:+.3f}, a jump of {abs(z[-1, 1] - z[-2, 1]):.3f} = eta * c")

Gx, Gy = np.meshgrid(np.linspace(-9, 9, 181), np.linspace(-9, 9, 181))
Zq = 0.5 * (A[0, 0] * Gx ** 2 + 2 * A[0, 1] * Gx * Gy + A[1, 1] * Gy ** 2)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].contour(Gx, Gy, Zq, levels=np.logspace(-1, 2.6, 14), cmap="viridis", alpha=0.6)
ax[0].plot(pn[:4, 0], pn[:4, 1], "x--", color="gray", lw=1, label="no clipping (leaves the plot)")
ax[0].plot(pc[:, 0], pc[:, 1], "o-", color="crimson", ms=3, lw=1, label="clipped at c = 2 (40 steps)")
ax[0].plot(*pc[0], "k^", ms=8, label="start"); ax[0].plot(0, 0, "k*", ms=10, label="minimum")
ax[0].set_xlim(-9, 9); ax[0].set_ylim(-9, 9); ax[0].set_aspect("equal")
ax[0].set_title("eta = 0.5 > 2/L: clipping turns divergence into a bounce", fontsize=10)
ax[0].set_xlabel("theta_1"); ax[0].set_ylabel("theta_2"); ax[0].legend(fontsize=7, loc="upper left")
for eta, ls in [(0.5, "-"), (1.0, "--")]:
    z = run_clipped(eta, 2.0, 60) @ Q
    ax[1].plot(z[:, 0], ls, color="tab:blue", lw=1, label=f"flat coordinate, eta = {eta}")
    ax[1].plot(z[:, 1], ls, color="tab:orange", marker=".", ms=3, lw=1, label=f"stiff coordinate, eta = {eta}")
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_title("Clipped GD in the eigenbasis: a 2-cycle across the valley", fontsize=10)
ax[1].set_xlabel("step"); ax[1].set_ylabel("coordinate"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Notice that** clipping caps the instability rather than removing it. While $\lVert g\rVert > c$ every
step has length exactly $\eta c = 1$, so the iterate walks steadily down the valley; near the minimum
the stiff coordinate still wants to grow by $\lvert 1 - 0.5\cdot10\rvert = 4$ per step, is clipped back,
and ends in a two-step bounce between $-0.54$ and $+0.46$ (between $+0.88$ and $-1.12$ for $\eta = 1$),
a jump of exactly $\eta c$, while the flat coordinate converges. The printed $\lVert\theta_{200}\rVert$
values above are just one end of each bounce. Without clipping the same run passes
$\lVert\theta\rVert = 2000$ after five steps. Turning an explosion into a bounded bounce is exactly what
you want when a single bad mini-batch would otherwise destroy a week of training — but clipping cannot
make a step $\eta > 2/L$ converge.
Clipping handles spikes; the learning rate handles curvature.
""")

nb.code(r"""
perm = rng.permutation(len(y_bc)); train_idx, val_idx = perm[:120], perm[120:]
w = np.zeros(Xb.shape[1]); train_curve, val_curve = [], []
for k in range(3000):
    w = w - 0.5 * (Xb[train_idx].T @ (sigmoid(Xb[train_idx] @ w) - y_bc[train_idx]) / len(train_idx))   # no ridge term this time
    train_curve.append(log_loss(w, Xb[train_idx], y_bc[train_idx]) - 0.5 * lam_ridge * (w[1:] @ w[1:]))
    val_curve.append(log_loss(w, Xb[val_idx], y_bc[val_idx]) - 0.5 * lam_ridge * (w[1:] @ w[1:]))
val_curve = np.array(val_curve); best = int(np.argmin(val_curve))
patience, stop_at = 100, None
for k in range(len(val_curve)):
    if k - int(np.argmin(val_curve[:k + 1])) >= patience:
        stop_at = k; break
print(f"best validation loss {val_curve[best]:.4f} at step {best}; final validation loss {val_curve[-1]:.4f}; training loss keeps falling to {train_curve[-1]:.4f}")
print(f"early stopping with patience {patience} would stop at step {stop_at} and restore step {int(np.argmin(val_curve[:stop_at + 1]))}")
plt.semilogy(train_curve, label="training loss (120 samples)"); plt.semilogy(val_curve, label="validation loss (449 samples)")
plt.axvline(best, color="gray", ls="--", label="best validation"); plt.axvline(stop_at, color="crimson", ls=":", label=f"stop (patience {patience})")
plt.title("Early stopping: validation loss turns around while training loss keeps falling"); plt.xlabel("GD step"); plt.ylabel("logistic loss"); plt.legend(); plt.show()
""")

nb.md(r"""
Notice that the unregularised model fitted on 120 samples drives its training loss towards zero
while the validation loss bottoms out early and then rises — overfitting, visible as a curve.
Stopping at the validation minimum is both a stopping rule and a regulariser; Module 13 makes the
link to ridge regression precise (for a quadratic loss, GD stopped at step $k$ behaves like ridge
with $\lambda \approx 1/(\eta k)$).

**Try it yourself:** put the ridge term back into the gradient (`+ lam_ridge * w[1:]`). The
validation curve should flatten instead of turning up.
""")

# ---------------------------------------------------------------- summary
nb.md(r"""
## Summary

* A learning problem is $\min_\theta L(\theta)$; stationary points include saddles, and in high
  dimensions saddles, not local minima, are what slow training down.
* Convexity ($\nabla^2 L \succeq 0$) makes every stationary point a global minimum; linear and
  logistic regression are convex, neural networks are not.
* On a quadratic, GD converges iff $\eta < 2/L$, fastest at $\eta = 2/(L+\mu)$ with rate
  $(\kappa-1)/(\kappa+1)$; the step count is linear in $\kappa$, and $\kappa$ is set by feature scaling.
* Mini-batch gradients are unbiased with variance $\propto 1/b$; a constant step leaves a noise
  ball of variance $\approx \eta\sigma^2/(2\lambda)$, which schedules shrink.
* Momentum accumulates along-valley velocity ($\sqrt\kappa$ instead of $\kappa$); RMSProp and Adam give
  each parameter its own step; Adam's bias correction makes its first step $\eta\,\mathrm{sign}(g)$ (up to $\epsilon$)
  and keeps the early steps $\approx\eta$.
* Newton's method squares the error each iteration but costs $O(d^3)$; L-BFGS is the practical
  version, and it is what `LogisticRegression` runs.
* At a constrained optimum the level set is tangent to the constraint: $\nabla f + \lambda\nabla h = 0$;
  the multiplier is the shadow price, and KKT adds $\mu \ge 0$ and complementary slackness.
* Read the loss and gradient-norm curves: range test for $\eta$, warm-up and decay, clipping for
  spikes, early stopping for overfitting.

**Next: open `exercises.ipynb`** and implement these pieces yourself.
""")

nb.save(MOD / "lesson.ipynb")
print("lesson cells:", len(nb.cells))

# =============================================================================
# EXERCISES — hidden answers are computed here by running the reference solutions
# =============================================================================
SETUP = r"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)
"""

ex = ExerciseSet(
    "Module 05 — Exercises",
    intro_md=r"""
These exercises make you implement the pieces of Module 05 yourself: the stability limit of gradient
descent, mini-batch SGD, a convexity test, learning-rate schedules, Adam from its equations, Newton's
method, a Lagrange problem, gradient clipping and a learning-rate sweep, two conceptual questions and
a mini-project comparing optimisers on a classification task. Use only NumPy (and the imports in the
setup cell) unless an exercise says otherwise.
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- Ex 1: 2/L
A1 = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
ev1 = np.linalg.eigvalsh(A1)
max_step_A1 = 2.0 / ev1[-1]                           # 0.42265
opt_step_A1 = 2.0 / (ev1[-1] + ev1[0])                # 0.33333
ex.add(
    title="Gradient descent on a quadratic and the largest stable step",
    difficulty=1,
    prompt_md=r"""
For $L(\theta) = \tfrac12\theta^\top A\theta$ with symmetric positive-definite $A$, gradient descent is
$\theta_{k+1} = \theta_k - \eta A\theta_k$. Implement

* `gd_quadratic(A, theta0, eta, steps)` — returns the NumPy array of iterates of shape
  `(steps + 1, d)`, starting with `theta0`;
* `max_stable_step(A)` — returns the largest step size for which GD converges from every start,
  as a `float` ($2/L$ with $L$ the largest eigenvalue; use `np.linalg.eigvalsh`);
* `optimal_step(A)` — returns the step $2/(L + \mu)$ that minimises the worst-case contraction factor.

Test matrix: `A = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])`.
""",
    starter=r"""
def gd_quadratic(A, theta0, eta, steps):
    # YOUR CODE HERE
    return None

def max_stable_step(A):
    # YOUR CODE HERE
    return None

def optimal_step(A):
    # YOUR CODE HERE
    return None
""",
    solution=r"""
def gd_quadratic(A, theta0, eta, steps):
    path = [np.asarray(theta0, dtype=float)]
    for _ in range(steps):
        path.append(path[-1] - eta * (A @ path[-1]))
    return np.array(path)

def max_stable_step(A):
    L = np.linalg.eigvalsh(A)[-1]
    return float(2.0 / L)

def optimal_step(A):
    ev = np.linalg.eigvalsh(A)
    return float(2.0 / (ev[-1] + ev[0]))
""",
    check=rf"""
A = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
try:
    path = gd_quadratic(A, np.array([1., 1., 1.]), 0.1, 50)
    shape_ok = path.shape == (51, 3)
    converged = np.linalg.norm(path[-1]) < 1e-2 * np.linalg.norm(path[0])
except Exception:
    shape_ok, converged = False, False
check('gd_quadratic returns 51 iterates of dimension 3', shape_ok)
check('with eta = 0.1 the iterates shrink by 100x in 50 steps', converged)
try:
    eta_max = max_stable_step(A)
    diverges = np.linalg.norm(gd_quadratic(A, np.array([1., 1., 1.]), 1.1 * eta_max, 60)[-1]) > 10
    stays = np.linalg.norm(gd_quadratic(A, np.array([1., 1., 1.]), 0.9 * eta_max, 60)[-1]) < 1
except Exception:
    eta_max, diverges, stays = None, False, False
check_hash('hidden: largest stable step 2/L for the test matrix', eta_max, '{sha_of(max_step_A1)}')
check('GD at 1.1 * max_stable_step diverges, at 0.9 * max_stable_step it does not', diverges and stays)
try:
    eta_opt = optimal_step(A)
except Exception:
    eta_opt = None
check_hash('hidden: optimal constant step 2/(L+mu)', eta_opt, '{sha_of(opt_step_A1)}')
""",
    explanation_md=r"""
In the eigenbasis of $A$ every coordinate is multiplied by $(1 - \eta\lambda_i)$ per step, so GD
converges from every start iff $\lvert 1 - \eta\lambda_i\rvert < 1$ for all $i$, i.e. $\eta < 2/\lambda_{\max}$.
The worst-case factor $\max(\lvert 1-\eta\mu\rvert, \lvert 1-\eta L\rvert)$ is smallest when the two are
equal, giving $\eta^\star = 2/(L+\mu)$ and a rate of $(\kappa-1)/(\kappa+1)$. For the test matrix
the eigenvalues are $3 - \sqrt3 \approx 1.27$, $3$ and $3 + \sqrt3 \approx 4.73$, so $L = 3 + \sqrt3$,
$\mu = 3 - \sqrt3$, $\eta_{\max} = 2/(3 + \sqrt3) \approx 0.42$ and $\eta^\star = 2/6 = 1/3$. A common slip is to use the largest *diagonal entry* instead of the largest
eigenvalue — for a non-diagonal matrix that overestimates the safe step.
""",
)

# ---------------------------------------------------------------- Ex 2: convexity via Hessian
def _hess_f(x, y):
    return np.array([[12 * x * x - 4.0, 1.0], [1.0, 2.0]])
def _hess_g(x, y):
    return np.array([[2.0, 1.0], [1.0, 2.0]])
_grid = [(x, y) for x in np.linspace(-2, 2, 41) for y in np.linspace(-2, 2, 41)]
min_curv_f = float(min(np.linalg.eigvalsh(_hess_f(x, y))[0] for x, y in _grid))   # -4.162...
min_curv_g = float(min(np.linalg.eigvalsh(_hess_g(x, y))[0] for x, y in _grid))   # 1.0
ex.add(
    title="Convexity test via Hessian eigenvalues",
    difficulty=1,
    prompt_md=r"""
A twice-differentiable function is convex on a region iff its Hessian is positive semi-definite at
every point of that region. Implement `min_curvature(hess_fn, grid)` which takes a function
`hess_fn(x, y)` returning the $2\times2$ Hessian at a point and an iterable of `(x, y)` points, and
returns the **smallest Hessian eigenvalue over the whole grid** as a `float` (use `np.linalg.eigvalsh`).
A negative result proves the function is not convex on that region.

Then write the Hessians by hand for

* $f(x, y) = x^4 - 2x^2 + y^2 + xy$ as `hess_f(x, y)`,
* $g(x, y) = x^2 + xy + y^2$ as `hess_g(x, y)`,

and evaluate `min_curvature` for both on the grid `[(x, y) for x in np.linspace(-2, 2, 41) for y in np.linspace(-2, 2, 41)]`.
""",
    starter=r"""
def min_curvature(hess_fn, grid):
    # YOUR CODE HERE
    return None

def hess_f(x, y):
    # YOUR CODE HERE  (Hessian of x^4 - 2x^2 + y^2 + xy)
    return None

def hess_g(x, y):
    # YOUR CODE HERE  (Hessian of x^2 + xy + y^2)
    return None
""",
    solution=r"""
def min_curvature(hess_fn, grid):
    return float(min(np.linalg.eigvalsh(np.asarray(hess_fn(x, y), dtype=float))[0] for x, y in grid))

def hess_f(x, y):
    return np.array([[12 * x ** 2 - 4.0, 1.0], [1.0, 2.0]])

def hess_g(x, y):
    return np.array([[2.0, 1.0], [1.0, 2.0]])
""",
    check=rf"""
grid = [(x, y) for x in np.linspace(-2, 2, 41) for y in np.linspace(-2, 2, 41)]
try:
    hf0 = np.asarray(hess_f(0.0, 0.0), dtype=float); hg0 = np.asarray(hess_g(0.5, -1.0), dtype=float)
except Exception:
    hf0, hg0 = None, None
check_close('hess_f at the origin is [[-4, 1], [1, 2]]', hf0, np.array([[-4., 1.], [1., 2.]]))
check_close('hess_g is the constant matrix [[2, 1], [1, 2]]', hg0, np.array([[2., 1.], [1., 2.]]))
try:
    mf = min_curvature(hess_f, grid); mg = min_curvature(hess_g, grid)
except Exception:
    mf, mg = None, None
check_hash('hidden: smallest curvature of f on the grid', mf, '{sha_of(min_curv_f)}')
check_hash('hidden: smallest curvature of g on the grid', mg, '{sha_of(min_curv_g)}')
check('f is NOT convex on the grid, g is', (mf is not None) and (mg is not None) and mf < 0 < mg)
""",
    explanation_md=r"""
$\partial^2 f/\partial x^2 = 12x^2 - 4$ is negative for $\lvert x\rvert < 1/\sqrt3$, so near the $y$-axis
the Hessian has a negative eigenvalue and $f$ is not convex (it has two minima at $x \approx \pm1$ and a
saddle in between). $g$ has the constant Hessian $\begin{pmatrix}2&1\\1&2\end{pmatrix}$ with eigenvalues
$1$ and $3$, so it is $1$-strongly convex and $3$-smooth everywhere. Checking only the diagonal
entries is a common mistake: a matrix can have positive diagonal entries and still have a negative
eigenvalue (e.g. $\begin{pmatrix}1&2\\2&1\end{pmatrix}$).
""",
)

# ---------------------------------------------------------------- Ex 3: schedules
def _cosine_lr(t, T, eta_max, eta_min=0.0):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))
def _warmup_cosine(t, T_warm, T, eta_max):
    if t < T_warm:
        return eta_max * t / T_warm
    return _cosine_lr(t - T_warm, T - T_warm, eta_max, 0.0)
sum_warm_cos = float(sum(_warmup_cosine(t, 100, 1000, 0.1) for t in range(1001)))   # 50.0
ex.add(
    title="Learning-rate schedules",
    difficulty=1,
    prompt_md=r"""
Implement two schedules as plain functions of the step counter $t$ (integers starting at 0):

* `cosine_lr(t, T, eta_max, eta_min=0.0)` $= \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})\big(1 + \cos(\pi t/T)\big)$ for $0 \le t \le T$;
* `warmup_cosine(t, T_warm, T, eta_max)` — linear warm-up $\eta_{\max}\, t / T_{\text{warm}}$ for
  $t < T_{\text{warm}}$, then cosine decay from $\eta_{\max}$ to $0$ over the remaining $T - T_{\text{warm}}$
  steps (i.e. `cosine_lr(t - T_warm, T - T_warm, eta_max, 0.0)`).

Both return a `float`.
""",
    starter=r"""
def cosine_lr(t, T, eta_max, eta_min=0.0):
    # YOUR CODE HERE
    return None

def warmup_cosine(t, T_warm, T, eta_max):
    # YOUR CODE HERE
    return None
""",
    solution=r"""
def cosine_lr(t, T, eta_max, eta_min=0.0):
    return float(eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T)))

def warmup_cosine(t, T_warm, T, eta_max):
    if t < T_warm:
        return float(eta_max * t / T_warm)
    return cosine_lr(t - T_warm, T - T_warm, eta_max, 0.0)
""",
    check=rf"""
try:
    c = [cosine_lr(t, 1000, 1.0) for t in (0, 250, 500, 750, 1000)]
except Exception:
    c = None
check_close('cosine: eta at t = 0, T/4, T/2, 3T/4, T', c, [1.0, 0.8535533906, 0.5, 0.1464466094, 0.0])
try:
    wc = [warmup_cosine(t, 100, 1000, 0.1) for t in (0, 50, 100, 1000)]
    total = float(sum(warmup_cosine(t, 100, 1000, 0.1) for t in range(1001)))
except Exception:
    wc, total = None, None
check_close('warm-up + cosine: eta at t = 0, 50, 100, 1000', wc, [0.0, 0.05, 0.1, 0.0])
check_hash('hidden: total learning rate summed over t = 0..1000', total, '{sha_of(sum_warm_cos)}')
""",
    explanation_md=r"""
Cosine decay spends a long time near $\eta_{\max}$ (85% of it at $T/4$), passes $\eta_{\max}/2$ at $T/2$
and approaches zero smoothly, which shrinks the SGD noise ball at the end of training. Warm-up
protects the first steps, when adaptive optimisers' running averages are unreliable and the
gradient at a random initialisation can be large. The summed learning rate is a useful sanity
check: the linear ramp contributes $\approx \eta_{\max}T_{\text{warm}}/2$ and the cosine half contributes
$\eta_{\max}(T - T_{\text{warm}})/2$ — here $5 + 45 = 50$.
""",
)

# ---------------------------------------------------------------- Ex 4: SGD linreg
def _sgd_linreg(X, y, eta, batch_size, epochs, seed):
    r = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        perm = r.permutation(n)
        for s in range(n // batch_size):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            Xb, yb = X[idx], y[idx]
            w = w - eta * (2.0 / batch_size) * Xb.T @ (Xb @ w - yb)
    return w
_r = np.random.default_rng(7)
X4 = _r.normal(size=(500, 4)) * np.array([1.0, 2.0, 0.5, 1.5])
w4_true = np.array([2.0, -1.0, 0.5, 3.0])
y4 = X4 @ w4_true + 0.3 * _r.normal(size=500)
w4_sgd = _sgd_linreg(X4, y4, 0.05, 25, 5, 0)
w4_ols = np.linalg.lstsq(X4, y4, rcond=None)[0]
assert np.abs(w4_sgd - w4_ols).max() < 0.1
ex.add(
    title="Mini-batch SGD for linear regression",
    difficulty=2,
    prompt_md=r"""
Implement `sgd_linreg(X, y, eta, batch_size, epochs, seed)` for the mean-squared-error loss
$L(w) = \tfrac1n\lVert Xw - y\rVert^2$ (no intercept, $w$ starts at zeros). Follow this protocol exactly
so that your result matches the reference run:

1. `r = np.random.default_rng(seed)`;
2. for each epoch: `perm = r.permutation(n)`, then for `s` in `range(n // batch_size)` take the
   rows `perm[s*batch_size:(s+1)*batch_size]` as the batch (drop the remainder);
3. update `w = w - eta * g` with the mini-batch gradient $g = \tfrac{2}{b}X_B^\top(X_B w - y_B)$.

Return the final `w` (shape `(d,)`). Data for the check:

```python
r = np.random.default_rng(7)
X = r.normal(size=(500, 4)) * np.array([1.0, 2.0, 0.5, 1.5])
w_true = np.array([2.0, -1.0, 0.5, 3.0])
y = X @ w_true + 0.3 * r.normal(size=500)
```
""",
    starter=r"""
def sgd_linreg(X, y, eta, batch_size, epochs, seed):
    r = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    # YOUR CODE HERE
    return None
""",
    solution=r"""
def sgd_linreg(X, y, eta, batch_size, epochs, seed):
    r = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        perm = r.permutation(n)
        for s in range(n // batch_size):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            Xb, yb = X[idx], y[idx]
            w = w - eta * (2.0 / batch_size) * Xb.T @ (Xb @ w - yb)
    return w
""",
    check=rf"""
r = np.random.default_rng(7)
X = r.normal(size=(500, 4)) * np.array([1.0, 2.0, 0.5, 1.5])
w_true = np.array([2.0, -1.0, 0.5, 3.0])
y = X @ w_true + 0.3 * r.normal(size=500)
w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
try:
    w_hat = sgd_linreg(X, y, 0.05, 25, 5, 0)
    close_to_ols = np.abs(w_hat - w_ols).max() < 0.1
except Exception:
    w_hat, close_to_ols = None, False
check('after 5 epochs the SGD solution is within 0.1 of the least-squares solution', close_to_ols)
check_close('exact match with the reference run (same seed, same protocol)', w_hat, np.array({np.array2string(w4_sgd, precision=10, separator=', ', floatmode='fixed')}), tol=1e-6)
try:
    w_full = sgd_linreg(X, y, 0.2, 500, 300, 0)           # batch = n is plain gradient descent (eta = 0.2 < 2/L = 0.255)
except Exception:
    w_full = None
check_close('with batch_size = n it is gradient descent and converges to OLS', w_full, w_ols, tol=1e-3)
""",
    explanation_md=r"""
Each mini-batch gradient is an unbiased estimate of the full gradient, so SGD follows the right
direction on average while costing only $b$ rows per step; one epoch here is 20 steps instead of one.
With $b = n$ the loop degenerates to full-batch GD and converges to the least-squares solution.
The exact-match check works because the protocol pins down the random permutation. If yours does
not match, you probably sampled with replacement, drew a new permutation per step instead of per
epoch, or used $1/b$ instead of $2/b$ in the gradient (with other sizes, keeping the remainder batch
also breaks the match; here $500/25 = 20$ leaves none).
""",
)

# ---------------------------------------------------------------- Ex 5: Adam
def _adam_step(theta, g, m, v, t, eta=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return theta - eta * m_hat / (np.sqrt(v_hat) + eps), m, v
A5 = np.array([[3.0, 1.0], [1.0, 1.0]]); b5 = np.array([1.0, -2.0])
_th = np.zeros(2); _m = np.zeros(2); _v = np.zeros(2)
for _t in range(1, 201):
    _th, _m, _v = _adam_step(_th, A5 @ _th - b5, _m, _v, _t, eta=0.1)
adam_ref_theta = _th.copy()
assert np.abs(adam_ref_theta - np.linalg.solve(A5, b5)).max() < 1e-3
ex.add(
    title="Adam from the update equations",
    difficulty=2,
    prompt_md=r"""
Implement one Adam step as a pure function

```python
adam_step(theta, g, m, v, t, eta=1e-3, beta1=0.9, beta2=0.999, eps=1e-8) -> (theta_new, m_new, v_new)
```

following the equations $m \leftarrow \beta_1 m + (1-\beta_1) g$, $v \leftarrow \beta_2 v + (1-\beta_2) g^2$
(element-wise), $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$,
$\theta \leftarrow \theta - \eta\,\hat m/(\sqrt{\hat v} + \epsilon)$. The step counter `t` starts at 1.

Then use it to minimise the quadratic $L(\theta) = \tfrac12\theta^\top A\theta - b^\top\theta$ with
`A = np.array([[3., 1.], [1., 1.]])`, `b = np.array([1., -2.])`, gradient $A\theta - b$, from
$\theta_0 = 0$, with `eta = 0.1` for exactly 200 steps; store the result in `theta_adam`.
""",
    starter=r"""
def adam_step(theta, g, m, v, t, eta=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # YOUR CODE HERE
    return None, None, None

A = np.array([[3., 1.], [1., 1.]]); b = np.array([1., -2.])
theta_adam = None
# YOUR CODE HERE: run 200 Adam steps with eta = 0.1 from theta = zeros(2)
""",
    solution=r"""
def adam_step(theta, g, m, v, t, eta=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return theta - eta * m_hat / (np.sqrt(v_hat) + eps), m, v

A = np.array([[3., 1.], [1., 1.]]); b = np.array([1., -2.])
theta = np.zeros(2); m = np.zeros(2); v = np.zeros(2)
for t in range(1, 201):
    theta, m, v = adam_step(theta, A @ theta - b, m, v, t, eta=0.1)
theta_adam = theta
""",
    check=rf"""
try:
    th1, m1, v1 = adam_step(np.zeros(2), np.array([0.1, -100.0]), np.zeros(2), np.zeros(2), 1, eta=0.01)
except Exception:
    th1, m1, v1 = None, None, None
check_close('first step has size eta in every coordinate, whatever the gradient scale', th1, np.array([-0.01, 0.01]), tol=1e-6)
check_close('first-moment state after step 1 is (1 - beta1) * g', m1, np.array([0.01, -10.0]), tol=1e-9)
check_close('second-moment state after step 1 is (1 - beta2) * g^2', v1, np.array([1e-5, 10.0]), tol=1e-9)
try:
    got = np.asarray(theta_adam, dtype=float)
except Exception:
    got = None
check_close('200 Adam steps on the quadratic match the reference run', got, np.array({np.array2string(adam_ref_theta, precision=10, separator=', ', floatmode='fixed')}), tol=1e-6)
check_close('... and land near the exact minimiser A^-1 b = (1.5, -3.5)', got, np.linalg.solve(np.array([[3., 1.], [1., 1.]]), np.array([1., -2.])), tol=1e-3)
""",
    explanation_md=r"""
The two running averages start at zero and are therefore biased towards zero early on — by the
factor $(1-\beta_1^t)$ for $m$ and $(1-\beta_2^t)$ for $v$. Dividing them out makes the very first
step equal to $\eta\,\mathrm{sign}(g)$ (up to $\epsilon$), which is why Adam is insensitive to the overall scale
of the gradient (and why its learning rate is typically 10–100 times smaller than SGD's). Typical
mistakes: forgetting the bias correction (first step $\approx 3.16\eta$), applying `np.sqrt` to the
raw $v$ instead of $\hat v$, or using $t$ starting at 0 (division by zero in the correction).
""",
)

# ---------------------------------------------------------------- Ex 6: Newton for logistic regression
_bc = load_breast_cancer()
X6_raw, y6 = _bc.data, _bc.target.astype(float)
X6 = np.c_[np.ones(len(X6_raw)), (X6_raw - X6_raw.mean(0)) / X6_raw.std(0)]
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def _newton_logreg(X, y, lam, iters):
    n, d = X.shape
    w = np.zeros(d)
    reg = lam * np.eye(d); reg[0, 0] = 0.0
    for _ in range(iters):
        p = _sigmoid(X @ w)
        g = X.T @ (p - y) / n + reg @ w
        H = (X * (p * (1 - p))[:, None]).T @ X / n + reg
        w = w - np.linalg.solve(H, g)
    return w
w6 = _newton_logreg(X6, y6, 1e-2, 10)
loss6 = float(np.mean(y6 * np.logaddexp(0, -X6 @ w6) + (1 - y6) * np.logaddexp(0, X6 @ w6)) + 0.5 * 1e-2 * (w6[1:] @ w6[1:]))
acc6 = float(np.mean((X6 @ w6 > 0) == (y6 == 1)))
g6 = X6.T @ (_sigmoid(X6 @ w6) - y6) / len(y6); g6[1:] += 1e-2 * w6[1:]
assert np.linalg.norm(g6) < 1e-8
ex.add(
    title="Newton's method for logistic regression",
    difficulty=2,
    prompt_md=r"""
For the $\ell_2$-regularised logistic loss (labels $y_i \in \{0, 1\}$, first column of $X$ is the intercept,
which is **not** penalised)

$$
L(w) = \frac1n\sum_{i=1}^n \Big[-y_i \log p_i - (1-y_i)\log(1-p_i)\Big] + \frac{\lambda}{2}\sum_{j \ge 1} w_j^2,
\qquad p_i = \sigma(x_i^\top w),
$$

the gradient is $\nabla L = \tfrac1n X^\top(p - y) + \lambda \tilde w$ and the Hessian is
$\nabla^2 L = \tfrac1n X^\top S X + \lambda \tilde I$, where $S = \mathrm{diag}\big(p_i(1-p_i)\big)$ and
$\tilde w$, $\tilde I$ have a zero in the intercept position. Implement

* `logreg_grad_hess(w, X, y, lam)` returning the tuple `(grad, hess)`;
* `newton_logreg(X, y, lam, iters)` which starts at `w = 0` and performs `iters` Newton steps
  `w = w - solve(H, g)`, returning `w`.

Use `np.linalg.solve`, not an explicit inverse. The check runs 10 iterations with $\lambda = 10^{-2}$ on the
standardised breast-cancer data (with an intercept column) and verifies that the gradient has
vanished to $10^{-8}$ — the quadratic-convergence signature.
""",
    starter=r"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logreg_grad_hess(w, X, y, lam):
    # YOUR CODE HERE
    return None, None

def newton_logreg(X, y, lam, iters):
    n, d = X.shape
    w = np.zeros(d)
    # YOUR CODE HERE
    return None
""",
    solution=r"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logreg_grad_hess(w, X, y, lam):
    n, d = X.shape
    reg = lam * np.eye(d); reg[0, 0] = 0.0            # do not penalise the intercept
    p = sigmoid(X @ w)
    grad = X.T @ (p - y) / n + reg @ w
    hess = (X * (p * (1 - p))[:, None]).T @ X / n + reg
    return grad, hess

def newton_logreg(X, y, lam, iters):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(iters):
        g, H = logreg_grad_hess(w, X, y, lam)
        w = w - np.linalg.solve(H, g)
    return w
""",
    check=rf"""
data = load_breast_cancer()
Xr, yb = data.data, data.target.astype(float)
Xs = np.c_[np.ones(len(Xr)), (Xr - Xr.mean(0)) / Xr.std(0)]
try:
    g0, H0 = logreg_grad_hess(np.zeros(Xs.shape[1]), Xs, yb, 1e-2)
    g0 = np.asarray(g0, dtype=float); H0 = np.asarray(H0, dtype=float)
    h_ok = H0.shape == (31, 31) and np.allclose(H0, H0.T) and np.allclose(H0[1:, 1:] - 1e-2 * np.eye(30), 0.25 * Xs[:, 1:].T @ Xs[:, 1:] / len(yb))
    h_intercept = abs(H0[0, 0] - 0.25) < 1e-9
except Exception:
    g0, h_ok, h_intercept = None, False, False
check_close('gradient at w = 0 is X^T (0.5 - y) / n', g0, Xs.T @ (0.5 - yb) / len(yb), tol=1e-9)
check('Hessian at w = 0 is X^T X / (4n) plus lambda on the penalised block', h_ok)
check('the intercept is not penalised (H[0, 0] = 0.25 at w = 0)', h_intercept)
try:
    w_hat = newton_logreg(Xs, yb, 1e-2, 10)
    g_fin, _ = logreg_grad_hess(w_hat, Xs, yb, 1e-2)
    gnorm = float(np.linalg.norm(g_fin))
    acc = float(np.mean((Xs @ w_hat > 0) == (yb == 1)))
    loss_fin = float(np.mean(yb * np.logaddexp(0, -Xs @ w_hat) + (1 - yb) * np.logaddexp(0, Xs @ w_hat)) + 0.5 * 1e-2 * (w_hat[1:] @ w_hat[1:]))
except Exception:
    gnorm, acc, loss_fin = None, None, None
check('after 10 Newton steps the gradient norm is below 1e-8', gnorm is not None and gnorm < 1e-8)
check_hash('hidden: training accuracy of the Newton solution', acc, '{sha_of(acc6)}')
check_hash('hidden: regularised loss at the optimum (3 decimals)', loss_fin, '{sha_of(loss6)}')
""",
    explanation_md=r"""
Newton's step solves the second-order Taylor model exactly, so once the iterate is near the optimum
the error is squared at every iteration — the gradient norm goes $1.4, 0.4, 0.15, 0.06, 0.02, 3\times10^{-3},
10^{-4}, 10^{-7}, 10^{-13}$. The Hessian $\tfrac1n X^\top S X$ is the Gauss–Newton-like matrix of the
logistic model: $S_{ii} = p_i(1-p_i) \le 1/4$, so at $w = 0$ it equals $X^\top X/(4n)$. Leaving the
intercept unpenalised is standard (its scale has nothing to do with overfitting); forgetting the
ridge term makes the Hessian nearly singular here because the 30 features are strongly correlated.
This is exactly what `LogisticRegression` computes — via L-BFGS rather than explicit Hessians.
""",
)

# ---------------------------------------------------------------- Ex 7: Lagrange analytic
# min x^2 + y^2 + z^2  s.t.  x - 2y + 3z = 14  ->  x* = 14*(1,-2,3)/14 = (1,-2,3), f* = 14, lambda = -2
a7 = np.array([1.0, -2.0, 3.0]); b7 = 14.0
x7 = b7 * a7 / (a7 @ a7); f7 = float(x7 @ x7); lam7 = float(-2 * b7 / (a7 @ a7))
# KKT: min (x-3)^2 + (y-1)^2  s.t. x + y <= 2  ->  active, x* = (2, 0) (projection of (3,1) onto the line), mu = 2
x8 = np.array([2.0, 0.0]); mu8 = 2.0
ex.add(
    title="Lagrange multipliers and KKT by hand",
    difficulty=2,
    prompt_md=r"""
Solve the following two problems **analytically** (pen and paper, then type in the numbers):

**(a)** Minimise $f(x, y, z) = x^2 + y^2 + z^2$ subject to $x - 2y + 3z = 14$. Report the minimiser
as `x_star_a` (NumPy array of length 3), the optimal value `f_star_a`, and the multiplier
`lambda_a` in the convention $\nabla f + \lambda\nabla h = 0$ with $h = x - 2y + 3z - 14$.

**(b)** Minimise $g(x, y) = (x - 3)^2 + (y - 1)^2$ subject to $x + y \le 2$. Report the minimiser
`x_star_b` (array of length 2) and the KKT multiplier `mu_b` $\ge 0$ of the constraint
$x + y - 2 \le 0$ (stationarity: $\nabla g + \mu\nabla c = 0$).

Hint for (a): stationarity gives $x \propto \nabla h$; for (b) first decide whether the constraint
is active by checking the unconstrained minimiser.
""",
    starter=r"""
x_star_a = None      # np.array([...])
f_star_a = None
lambda_a = None

x_star_b = None      # np.array([...])
mu_b = None
""",
    solution=r"""
a = np.array([1.0, -2.0, 3.0]); b = 14.0
x_star_a = b * a / (a @ a)          # (1, -2, 3): stationarity 2x + lambda a = 0 -> x = -lambda a / 2; feasibility fixes lambda
f_star_a = float(x_star_a @ x_star_a)   # 14
lambda_a = -2.0 * b / (a @ a)       # -2

x_star_b = np.array([2.0, 0.0])     # unconstrained minimiser (3, 1) violates x + y <= 2, so the constraint is active
mu_b = 2.0                          # 2(x-3) + mu = 0 and 2(y-1) + mu = 0 at (2, 0) give mu = 2
""",
    check=rf"""
try:
    xa = np.asarray(x_star_a, dtype=float)
    feasible_a = abs(xa @ np.array([1., -2., 3.]) - 14.0) < 1e-6
    parallel_a = np.allclose(np.cross(2 * xa, np.array([1., -2., 3.])), 0.0, atol=1e-6)
except Exception:
    xa, feasible_a, parallel_a = None, False, False
check('(a) your point satisfies the constraint', feasible_a)
check('(a) grad f is parallel to grad h at your point', parallel_a)
check_hash('(a) hidden: minimiser', (xa if xa is not None else None), '{sha_of(x7)}')
check_hash('(a) hidden: optimal value', (float(f_star_a) if f_star_a is not None else None), '{sha_of(f7)}')
check_hash('(a) hidden: multiplier lambda', (float(lambda_a) if lambda_a is not None else None), '{sha_of(lam7)}')
try:
    xb = np.asarray(x_star_b, dtype=float)
    active_b = abs(xb.sum() - 2.0) < 1e-6
except Exception:
    xb, active_b = None, False
check('(b) the constraint is active at your point', active_b)
check_hash('(b) hidden: minimiser', (xb if xb is not None else None), '{sha_of(x8)}')
check_hash('(b) hidden: multiplier mu', (float(mu_b) if mu_b is not None else None), '{sha_of(mu8)}')
check('(b) dual feasibility: mu >= 0', mu_b is not None and float(mu_b) >= 0)
""",
    explanation_md=r"""
(a) Stationarity $2x + \lambda a = 0$ says the minimiser is a multiple of $a = (1, -2, 3)$; feasibility
$a^\top x = 14$ fixes the multiple: $x^\star = 14\,a/\lVert a\rVert^2 = (1, -2, 3)$, $f^\star = 14$ and
$\lambda = -2\cdot14/14 = -2$. This is the projection of the origin onto the plane — the general
formula $x^\star = b\,a/\lVert a\rVert^2$ from the concept guide. (b) The unconstrained minimiser $(3, 1)$
has $x + y = 4 > 2$, so the constraint must be active; on the line $x + y = 2$ the closest point to
$(3, 1)$ is its orthogonal projection $(3, 1) - \tfrac{4 - 2}{2}(1, 1) = (2, 0)$, and stationarity
$2(x - 3) + \mu = 0$, $2(y - 1) + \mu = 0$ gives $\mu = 2 > 0$ from both equations, confirming the
active set. Had the constraint been $x + y \le 5$, the unconstrained point would be feasible,
$\mu = 0$ and complementary slackness would hold with a slack constraint.
""",
)

# ---------------------------------------------------------------- Ex 8: clipping + LR sweep
def _clip_by_global_norm(g, c):
    norm = np.linalg.norm(g)
    return g * min(1.0, c / norm) if norm > 0 else g
def _lr_sweep(A, x0, etas, steps=30):
    out = []
    for eta in etas:
        x = np.asarray(x0, dtype=float).copy()
        for _ in range(steps):
            x = x - eta * (A @ x)
        out.append(0.5 * x @ A @ x)
    return np.array(out)
A8 = np.diag([1.0, 10.0]); x08 = np.array([10.0, 1.0]); etas8 = np.logspace(-2, 0, 41)
losses8 = _lr_sweep(A8, x08, etas8)
best_eta8 = float(etas8[int(np.argmin(losses8))])              # 0.1778...
f0_8 = 0.5 * x08 @ A8 @ x08
first_bad8 = float(etas8[int(np.argmax(losses8 > f0_8))])      # 0.2239 (first eta above 2/L = 0.2)
assert first_bad8 > 0.2 and etas8[int(np.argmax(losses8 > f0_8)) - 1] < 0.2
ex.add(
    title="Gradient clipping and a learning-rate sweep",
    difficulty=2,
    prompt_md=r"""
Two diagnostics from Section 8 of the guide.

1. `clip_by_global_norm(g, c)` — return $g \cdot \min(1, c/\lVert g\rVert_2)$ (and `g` unchanged if its norm is 0).
2. `lr_sweep(A, x0, etas, steps=30)` — for each $\eta$ in `etas`, run `steps` iterations of GD on
   $\tfrac12 x^\top A x$ from `x0` and return the array of final loss values $\tfrac12 x^\top A x$, one per $\eta$.

Then, with `A = np.diag([1., 10.])`, `x0 = np.array([10., 1.])` and `etas = np.logspace(-2, 0, 41)`,
set `best_eta` to the $\eta$ with the smallest final loss and `first_bad_eta` to the smallest $\eta$ at which the
final loss **exceeds the initial loss** $\tfrac12 x_0^\top A x_0$ (divergence). Compare both with $2/L$ and $2/(L+\mu)$.
""",
    starter=r"""
def clip_by_global_norm(g, c):
    # YOUR CODE HERE
    return None

def lr_sweep(A, x0, etas, steps=30):
    # YOUR CODE HERE
    return None

A = np.diag([1., 10.]); x0 = np.array([10., 1.]); etas = np.logspace(-2, 0, 41)
best_eta = None
first_bad_eta = None
# YOUR CODE HERE
""",
    solution=r"""
def clip_by_global_norm(g, c):
    norm = np.linalg.norm(g)
    return g * min(1.0, c / norm) if norm > 0 else g

def lr_sweep(A, x0, etas, steps=30):
    out = []
    for eta in etas:
        x = np.asarray(x0, dtype=float).copy()
        for _ in range(steps):
            x = x - eta * (A @ x)
        out.append(0.5 * x @ A @ x)
    return np.array(out)

A = np.diag([1., 10.]); x0 = np.array([10., 1.]); etas = np.logspace(-2, 0, 41)
losses = lr_sweep(A, x0, etas)
best_eta = float(etas[np.argmin(losses)])                         # closest grid point to 2/(L+mu) = 0.182
first_bad_eta = float(etas[np.argmax(losses > 0.5 * x0 @ A @ x0)])  # first grid point above 2/L = 0.2
print(f"best eta = {best_eta:.4f} (theory 2/(L+mu) = {2/11:.4f});  first diverging eta = {first_bad_eta:.4f} (theory 2/L = 0.2)")
""",
    check=rf"""
try:
    c1 = clip_by_global_norm(np.array([3.0, 4.0]), 1.0)
    c2 = clip_by_global_norm(np.array([0.3, 0.4]), 1.0)
    c3 = clip_by_global_norm(np.zeros(2), 1.0)
except Exception:
    c1, c2, c3 = None, None, None
check_close('clipping (3, 4) at c = 1 gives (0.6, 0.8)', c1, np.array([0.6, 0.8]))
check_close('a gradient already inside the ball is unchanged', c2, np.array([0.3, 0.4]))
check_close('the zero gradient stays zero', c3, np.zeros(2))
try:
    sw = np.asarray(lr_sweep(np.diag([1., 10.]), np.array([10., 1.]), np.array([0.05, 0.1, 0.3]), steps=5), dtype=float)
except Exception:
    sw = None
check_close('lr_sweep: 5 steps at eta = 0.05, 0.1, 0.3', sw, np.array({np.array2string(_lr_sweep(A8, x08, np.array([0.05, 0.1, 0.3]), 5), precision=10, separator=', ', floatmode='fixed')}), tol=1e-6)
check_hash('hidden: best eta on the grid', (float(best_eta) if best_eta is not None else None), '{sha_of(best_eta8)}')
check_hash('hidden: first diverging eta on the grid', (float(first_bad_eta) if first_bad_eta is not None else None), '{sha_of(first_bad8)}')
check('best eta is below 2/L = 0.2 and first diverging eta is above it', best_eta is not None and first_bad_eta is not None and best_eta < 0.2 < first_bad_eta)
""",
    explanation_md=r"""
Clipping rescales the whole gradient vector so that its norm is at most $c$; it changes the length,
never the direction, and bounds every step by $\eta c$. The sweep is the learning-rate range test in
its simplest form: the final loss is smallest at the grid point nearest $2/(L+\mu) = 0.182$ and jumps
above the initial loss at the first grid point beyond $2/L = 0.2$, because the stiff coordinate is
multiplied by $\lvert 1 - 10\eta\rvert > 1$ thirty times. Two frequent mistakes: clipping each
coordinate separately (that is a different operation, "clip by value", and it does change the
direction), and running the sweep on a single shared `x` across learning rates instead of restarting
from `x0`.
""",
)

# ---------------------------------------------------------------- Ex 9: heavy ball (stretch)
def _steps_to_tol_gd(A, x0, eta, tol=1e-6, max_steps=100000):
    x = np.asarray(x0, dtype=float).copy()
    for k in range(1, max_steps + 1):
        x = x - eta * (A @ x)
        if np.linalg.norm(x) < tol:
            return k
    return None
def _heavy_ball(A, x0, eta, beta, tol=1e-6, max_steps=100000):
    x = np.asarray(x0, dtype=float).copy(); v = np.zeros_like(x)
    for k in range(1, max_steps + 1):
        v = beta * v + A @ x
        x = x - eta * v
        if np.linalg.norm(x) < tol:
            return k
    return None
kappa9 = 100.0; A9 = np.diag([1.0, kappa9]); x09 = np.array([1.0, 1.0])
beta9 = ((np.sqrt(kappa9) - 1) / (np.sqrt(kappa9) + 1)) ** 2
eta9 = (2 / (np.sqrt(kappa9) + 1.0)) ** 2
k_gd9 = _steps_to_tol_gd(A9, x09, 2 / (kappa9 + 1))      # 709
k_hb9 = _heavy_ball(A9, x09, eta9, beta9)                # 95
assert k_gd9 == 709 and k_hb9 == 95
ex.add(
    title="Heavy-ball momentum beats gradient descent by a factor of about sqrt(kappa)",
    difficulty=3,
    prompt_md=r"""
On the quadratic $\tfrac12 x^\top A x$ with $A = \mathrm{diag}(1, \kappa)$, $\kappa = 100$, compare

* `steps_to_tol_gd(A, x0, eta, tol=1e-6, max_steps=100000)` — plain GD; return the number of steps
  $k$ (counting from 1) after which $\lVert x_k\rVert < $ `tol` for the first time (or `None`);
* `heavy_ball(A, x0, eta, beta, tol=1e-6, max_steps=100000)` — Polyak momentum
  $v_k = \beta v_{k-1} + A x_{k-1}$, $x_k = x_{k-1} - \eta v_k$ with $v_0 = 0$; same return convention.

Run both from `x0 = np.array([1., 1.])`: GD with its optimal step $\eta = 2/(L+\mu)$, and heavy ball with
the tuned constants $\beta = \big(\tfrac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\big)^2$,
$\eta = \big(\tfrac{2}{\sqrt L + \sqrt\mu}\big)^2$. Store the two step counts in `k_gd` and `k_hb`. The
theory predicts rates $(\kappa-1)/(\kappa+1) = 0.980$ versus $(\sqrt\kappa-1)/(\sqrt\kappa+1) = 0.818$.
""",
    starter=r"""
def steps_to_tol_gd(A, x0, eta, tol=1e-6, max_steps=100000):
    # YOUR CODE HERE
    return None

def heavy_ball(A, x0, eta, beta, tol=1e-6, max_steps=100000):
    # YOUR CODE HERE
    return None

kappa = 100.0; A = np.diag([1.0, kappa]); x0 = np.array([1.0, 1.0])
k_gd = None
k_hb = None
# YOUR CODE HERE
""",
    solution=r"""
def steps_to_tol_gd(A, x0, eta, tol=1e-6, max_steps=100000):
    x = np.asarray(x0, dtype=float).copy()
    for k in range(1, max_steps + 1):
        x = x - eta * (A @ x)
        if np.linalg.norm(x) < tol:
            return k
    return None

def heavy_ball(A, x0, eta, beta, tol=1e-6, max_steps=100000):
    x = np.asarray(x0, dtype=float).copy(); v = np.zeros_like(x)
    for k in range(1, max_steps + 1):
        v = beta * v + A @ x
        x = x - eta * v
        if np.linalg.norm(x) < tol:
            return k
    return None

kappa = 100.0; A = np.diag([1.0, kappa]); x0 = np.array([1.0, 1.0])
L, mu = kappa, 1.0
k_gd = steps_to_tol_gd(A, x0, 2 / (L + mu))
beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
eta_hb = (2 / (np.sqrt(L) + np.sqrt(mu))) ** 2
k_hb = heavy_ball(A, x0, eta_hb, beta)
print(f"GD: {k_gd} steps   heavy ball: {k_hb} steps   ratio {k_gd / k_hb:.1f}   (theory: ln-rate ratio {np.log(0.818) / np.log(0.980):.1f})")
""",
    check=rf"""
try:
    k1 = steps_to_tol_gd(np.diag([1., 1.]), np.array([1., 1.]), 0.5, tol=1e-6)      # identity: factor 0.5 per step
    k2 = heavy_ball(np.diag([1., 1.]), np.array([1., 1.]), 0.5, 0.0, tol=1e-6)      # beta = 0 is plain GD
except Exception:
    k1, k2 = None, None
check('GD on the identity with eta = 0.5 needs 21 steps to shrink below 1e-6', k1 == 21)
check('heavy ball with beta = 0 reproduces GD', k2 == 21)
check_hash('hidden: GD steps for kappa = 100', (int(k_gd) if k_gd is not None else None), '{sha_of(int(k_gd9))}')
check_hash('hidden: heavy-ball steps for kappa = 100', (int(k_hb) if k_hb is not None else None), '{sha_of(int(k_hb9))}')
check('heavy ball is at least 5x faster', k_gd is not None and k_hb is not None and k_gd / k_hb > 5)
""",
    explanation_md=r"""
GD with the optimal step contracts by $0.980$ per step, so it needs about $\ln(10^{-6}\!/\!\sqrt2)/\ln 0.98 \approx 700$
steps; the tuned heavy ball contracts by $0.818$ and needs about 95. The ratio, $\approx 7.5$, is
close to $\sqrt\kappa\,/\,1.3$ — the promised $\sqrt\kappa$ speed-up up to a constant. Momentum achieves this
because the sideways (stiff-direction) kicks alternate in sign and cancel in $v$, while the
along-valley pushes accumulate. Note that the optimal $\beta \approx 0.67$ is *smaller* than the
usual 0.9: with $\beta = 0.9$ and $\eta = 1/L$ the heavy ball needs 264 steps here — still 2.7x
better than GD, but the constants matter.
""",
)

# ---------------------------------------------------------------- Ex 10: conceptual — momentum in valleys
ex.add_conceptual(
    title="Why momentum helps in narrow valleys",
    difficulty=2,
    prompt_md=r"""
A colleague says: *"Momentum just makes the effective learning rate $\eta/(1-\beta)$ larger. So I could
get the same effect by using plain gradient descent with a 10x larger step."* Explain, using the
ill-conditioned bowl with eigenvalues $\mu = 1$ and $L = 100$, (i) what goes wrong with the 10x
larger plain step, (ii) why momentum does not suffer from the same problem, and (iii) roughly how
many steps each method needs to reduce the error by $10^{6}$.
""",
    answer_md=r"""
(i) The largest stable step for GD is $2/L = 0.02$. A step of $10/L = 0.1$ multiplies the stiff
coordinate by $1 - 0.1\cdot100 = -9$ per iteration: it diverges immediately. The effective step of
momentum is only $\eta/(1-\beta)$ for a *constant* gradient; on an oscillating one it is much smaller.

(ii) In the stiff direction the gradient flips sign at every step, so the velocity
$v_t = \beta v_{t-1} + g_t$ keeps *cancelling* and stays small; in the flat direction the gradient
always points the same way, so $v_t$ builds up to $g/(1-\beta)$. Momentum therefore behaves like a
large step along the valley and a small step across it — something no single scalar $\eta$ can do.

(iii) GD with its best constant step contracts by $(\kappa-1)/(\kappa+1) = 0.980$ per step, so
$10^{6}$ needs about $\ln(10^{6})/\ln(1/0.98) \approx 690$ steps. Tuned heavy ball contracts by
$(\sqrt\kappa-1)/(\sqrt\kappa+1) = 0.818$ and needs about $70$ — roughly a factor $\sqrt\kappa$ fewer.
(The count you measure in Exercise 9 comes out somewhat higher: with the tuned $\beta$ the extreme modes
are critically damped, which adds a factor $k$ to $0.818^k$; the speed-up is still of order $\sqrt\kappa$, only a constant factor smaller.)
""",
)

# ---------------------------------------------------------------- Ex 11: conceptual — KKT diagnosis
ex.add_conceptual(
    title="Diagnose a KKT argument",
    difficulty=2,
    prompt_md=r"""
A data scientist solves $\min_w \tfrac12\lVert w\rVert^2$ subject to $y_i(w^\top x_i + b) \ge 1$ for all $i$ (the
hard-margin SVM) and reports: *"All 10 000 training points have a positive Lagrange multiplier, so
all of them are support vectors, and the model is therefore very robust."* Using the KKT conditions,
explain why this report is almost certainly wrong, what the multipliers of most points must be, and
what a positive multiplier tells you about a point's position relative to the margin.
""",
    answer_md=r"""
Write the constraints as $g_i(w, b) = 1 - y_i(w^\top x_i + b) \le 0$ with multipliers $\alpha_i \ge 0$.
Complementary slackness, $\alpha_i\, g_i = 0$, says that a point with $\alpha_i > 0$ must have
$g_i = 0$, i.e. it lies **exactly on the margin** $y_i(w^\top x_i + b) = 1$. A point strictly outside
the margin has $g_i < 0$ and hence $\alpha_i = 0$. For 10 000 points in general position it is
impossible for all of them to lie exactly on two parallel hyperplanes, so the vast majority must
have $\alpha_i = 0$ — they are not support vectors and could be deleted without changing the
solution $w = \sum_i \alpha_i y_i x_i$. If the solver really returned $\alpha_i > 0$ for every
point, either the multipliers are tiny numerical noise (check them against a tolerance) or a
soft-margin formulation was used, where $\alpha_i = C$ for every *misclassified* or
inside-margin point — which would indicate a badly fitting model, the opposite of "robust".
A positive multiplier means the constraint is active and "pushing": removing that point would
move the decision boundary.
""",
)

# ---------------------------------------------------------------- Ex 12: mini-project
ex.add_project(
    title="Mini-project: GD vs momentum vs Adam on logistic regression",
    prompt_md=r"""
Generate a two-class dataset with 400 points in 10 dimensions where the feature scales differ a lot
(so that the problem is badly conditioned), for example

```python
r = np.random.default_rng(1)
scales = np.logspace(0, 1.5, 10)                 # feature scales from 1 to ~32
X = r.normal(size=(400, 10)) * scales
w_true = r.normal(size=10) / scales             # so every feature matters about equally
y = (X @ w_true + 0.5 * r.normal(size=400) > 0).astype(float)
```

Train an $\ell_2$-regularised logistic regression (intercept included, $\lambda = 10^{-3}$) with
**plain GD, heavy-ball momentum ($\beta = 0.9$) and Adam**, each implemented by you, for 500
full-batch steps, and

1. plot the training loss curves of the three methods on a log scale, for at least three learning
   rates per method (e.g. a factor 3 apart);
2. report the final loss and training accuracy of the best run of each method;
3. repeat the experiment with standardised features and compare;
4. write 3–5 sentences on which method is most sensitive to the learning rate, how standardisation
   changes the picture, and what you would recommend to a colleague fitting such a model.
""",
    rubric_md=r"""
* implement the three optimisers correctly from their update equations (no library optimisers);
* choose learning-rate grids that include at least one clearly too-large (oscillating or diverging) and one too-slow run per method, and recognise them in the plot;
* identify the raw problem's bad conditioning as the reason plain GD is slow and relate it to the feature scales;
* show that standardisation narrows the gap between the methods and permits a larger stable step;
* report accuracy honestly (training accuracy on the same data is optimistic) and say so;
* give a recommendation that mentions both feature scaling and a learning-rate test.
""",
    solution_sketch_md=r"""
With the raw features, the Hessian eigenvalues span the squared feature scales, so $\kappa$ is of
the order $10^3$ and plain GD needs a tiny step ($\eta \lesssim 2/L$) that leaves the flat directions
almost untouched after 500 steps; momentum helps by about $\sqrt\kappa$; Adam's per-coordinate
scaling makes it nearly indifferent to the feature scales, but it still diverges or stalls if $\eta$
is far off. After standardising, $L$ drops by orders of magnitude, GD and momentum can use a step
around $1/L$ (here several units) and all three methods converge in a few hundred steps. The two optima
differ noticeably (penalised loss about $0.098$ raw versus $0.131$ standardised): standardising divides
feature $j$ by $\sigma_j$, so its weight is multiplied by $\sigma_j$ and the ridge penalty on it by
$\sigma_j^2$ (up to $\approx 1000\times$ here) — the standardised problem is much more strongly regularised.
Compare the two set-ups by accuracy or by the unpenalised loss, not by the penalised objective. Recommendation:
standardise, run a quick learning-rate sweep, use momentum or Adam with the largest stable step,
and validate on held-out data.
""",
    solution_code=r"""
r = np.random.default_rng(1)
scales = np.logspace(0, 1.5, 10)
X = r.normal(size=(400, 10)) * scales
w_true = r.normal(size=10) / scales
y = (X @ w_true + 0.5 * r.normal(size=400) > 0).astype(float)
lam = 1e-3

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def loss_and_grad(w, Xb, y):
    z = Xb @ w; p = sigmoid(z)
    loss = np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)) + 0.5 * lam * (w[1:] @ w[1:])
    g = Xb.T @ (p - y) / len(y); g[1:] += lam * w[1:]
    return loss, g

def train(Xb, y, method, eta, steps=500, beta=0.9, beta1=0.9, beta2=0.999, eps=1e-8):
    w = np.zeros(Xb.shape[1]); v = np.zeros_like(w); m = np.zeros_like(w); hist = []
    for t in range(1, steps + 1):
        loss, g = loss_and_grad(w, Xb, y); hist.append(loss)
        if not np.isfinite(loss):
            break
        if method == "gd":
            w = w - eta * g
        elif method == "momentum":
            v = beta * v + g; w = w - eta * v
        elif method == "adam":
            m = beta1 * m + (1 - beta1) * g; v = beta2 * v + (1 - beta2) * g * g
            w = w - eta * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)
    return w, np.array(hist)

grids = {"gd": [1e-3, 1e-2, 3e-2, 1e-1], "momentum": [3e-4, 3e-3, 1e-2, 1e-1, 3e-1], "adam": [3e-3, 3e-2, 1e-1, 1.0]}
grids_std = {"gd": [0.3, 1.0, 3.0, 30.0], "momentum": [0.1, 0.3, 1.0, 100.0], "adam": [3e-3, 3e-2, 1e-1, 3.0]}
for title, Xf, grid in [("raw features", X, grids), ("standardised features", (X - X.mean(0)) / X.std(0), grids_std)]:
    Xb = np.c_[np.ones(len(Xf)), Xf]
    H0 = 0.25 * Xb.T @ Xb / len(y); ev = np.linalg.eigvalsh(H0)
    print(f"\n=== {title}: Hessian at 0 has L = {ev[-1]:.3g}, kappa = {ev[-1] / max(ev[0], 1e-12):.3g}, 2/L = {2 / ev[-1]:.3g}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, method in zip(axes, grid):
        best = None
        for eta in grid[method]:
            w, hist = train(Xb, y, method, eta)
            # unstable = diverged, ended above the start, or still oscillating: the loss rose in more than 5 of the last 50 steps
            unstable = len(hist) < 500 or not np.isfinite(hist[-1]) or hist[-1] > hist[0] or np.sum(np.diff(hist[-50:]) > 1e-9) > 5
            ax.semilogy(hist, lw=1, label=f"eta={eta:g}" + ("  (unstable)" if unstable else ""))
            if not unstable and (best is None or hist[-1] < best[1]):
                z = Xb @ w
                best = (eta, hist[-1], np.mean((z > 0) == (y == 1)), np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)))
        ax.set_title(f"{method} ({title})"); ax.set_xlabel("step"); ax.legend(fontsize=7)
        if best:
            print(f"  {method:9s}: best eta = {best[0]:g}, final loss = {best[1]:.4f} (unpenalised {best[3]:.4f}), training accuracy = {best[2]:.3f}")
    axes[0].set_ylabel("training loss"); plt.tight_layout(); plt.show()
print("\nDiscussion: on raw features the Hessian's condition number is about a thousand, so plain GD must use a tiny step and")
print("barely moves in the flat directions; momentum improves by roughly sqrt(kappa); Adam is nearly indifferent to the scales")
print("but has its own learning-rate window. Standardising removes most of the conditioning problem and lets all three methods")
print("converge in a few hundred steps. The two optima differ noticeably (penalised loss ~0.098 raw vs ~0.131 standardised) because")
print("the ridge penalty is not scale-invariant: standardising multiplies the penalty on feature j by its variance (up to ~1000x here),")
print("so compare the set-ups by accuracy or unpenalised loss, not by the penalised objective. Recommendation: standardise, sweep the")
print("learning rate, prefer momentum/Adam, and validate on held-out data -- the training accuracy reported above is optimistic.")
""",
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
print("exercises:", len(ex.items), "items;", sum(1 for i in ex.items if i["kind"] == "code"), "coding,",
      sum(1 for i in ex.items if i["kind"] == "concept"), "conceptual,", sum(1 for i in ex.items if i["kind"] == "project"), "project")
