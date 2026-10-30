# Module 05 — Optimisation

> After this module you can look at a training-loss curve and say *why* it looks the way it does — and whether to change the step size, switch optimiser, rescale the features, add a constraint, or simply stop.

## Why this matters for ML and AI

Every model you have ever trained was the output of an optimisation algorithm. `LinearRegression` solves a convex least-squares problem in closed form; `LogisticRegression` runs L-BFGS, a quasi-Newton method; a support-vector machine is a constrained quadratic programme whose Lagrange multipliers *are* the support vectors; GPT-style language models are trained with AdamW, a momentum method with per-parameter step sizes and decoupled weight decay; reinforcement-learning policies are improved by stochastic gradient ascent under a KL constraint. The learning rate is, by a wide margin, the hyper-parameter that most often decides whether a run succeeds, and almost all of its behaviour is explained by two numbers you met in Module 03: the largest and smallest curvature of the loss.

The mathematics here is also where the folklore of deep learning becomes checkable. "Standardise your inputs", "use warm-up", "momentum helps in narrow valleys", "Adam needs a smaller learning rate than SGD", "clip the gradient norm at 1.0" — each of these is a theorem, or a corollary of one, about quadratics, condition numbers, and noisy gradient estimates. Once you can derive the stable step-size range $\eta < 2/L$ on a quadratic you will never again guess learning rates at random.

Finally, constrained optimisation is the language in which many "principled" ML objects are stated: maximum-entropy distributions (hence softmax), the SVM margin, the KL penalty in RLHF, projected gradient steps that keep weights non-negative. Lagrange multipliers and the KKT conditions tell you what the solution of such a problem must look like before you compute anything.

## Learning objectives

- Formulate a learning task as minimisation of an empirical risk and classify its stationary points (minima, maxima, saddles).
- Test convexity with the chord, first-order and Hessian criteria, and explain why convex problems have no spurious local minima.
- Derive the stability condition $\eta < 2/L$, the optimal step $2/(L+\mu)$ and the rate $(\kappa-1)/(\kappa+1)$ of gradient descent on a quadratic, and connect it to feature scaling.
- Explain why mini-batch gradients are unbiased, derive how their variance scales with batch size and how large the resulting SGD noise ball is, and say what a learning-rate schedule does to it.
- Write the update equations of heavy-ball momentum, Nesterov, RMSProp, Adam and AdamW from memory, implement them in NumPy, and explain what Adam's bias correction and AdamW's decoupled weight decay fix.
- Derive Newton's method from a second-order Taylor expansion, state its quadratic convergence and its $O(d^3)$ cost, and describe the L-BFGS and Gauss–Newton work-arounds.
- Solve small equality-constrained problems with Lagrange multipliers, read the KKT conditions for inequality constraints (why their multipliers are non-negative, and complementary slackness), and compute the dual of a small problem.
- Diagnose a training run from its loss curve and gradient norms and choose the right fix (step size, schedule, clipping, early stopping).

## Prerequisites

- [Module 04 — Differential calculus](../04-differential-calculus/README.md): gradients, Hessians, the chain rule and second-order Taylor expansion. Recall that $\nabla f(\theta)$ points in the direction of steepest ascent and that $f(\theta + p) \approx f(\theta) + \nabla f(\theta)^\top p + \tfrac12 p^\top \nabla^2 f(\theta)\, p$.
- [Module 03 — Eigenvalues, SVD and PCA](../../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md): eigendecomposition of symmetric matrices, positive (semi-)definiteness and the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$.
- [Module 02 — Matrices and linear maps](../../part-1-linear-algebra/02-matrices-and-linear-maps/README.md): the least-squares problem and the normal equations, which are the closed-form answer we compare iterative methods against.

## 1. The optimisation problem in machine learning

**Intuition.** Training a model is hiking in fog. Your position is the parameter vector $\theta$, the altitude is the loss, and all you can measure locally is the slope under your feet. You want the lowest point in the valley system. The questions of this module are: which direction to step, how far, how to afford it when the map is enormous, and how you know when you have arrived.

**Definition.** Given training pairs $(x_i, y_i)$, a model $f_\theta$ and a per-example loss $\ell$, the *empirical risk* is

$$
L(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell\big(f_\theta(x_i), y_i\big) + \lambda R(\theta),
$$

where $R$ is an optional regulariser with weight $\lambda \ge 0$. Training means solving $\min_{\theta \in \mathbb{R}^d} L(\theta)$. A point is *stationary* if $\nabla L(\theta) = 0$. It is a *local minimum* if $L(\theta) \le L(\theta')$ for all $\theta'$ in a neighbourhood, a *global minimum* if that holds everywhere, and a *saddle point* if it is stationary but the Hessian $\nabla^2 L(\theta)$ has both positive and negative eigenvalues (downhill in some directions, uphill in others). "Converged" is always relative to a tolerance: $\lVert \nabla L(\theta_k)\rVert \le \varepsilon$, or $L(\theta_k) - L^\star \le \varepsilon$, or $\lVert\theta_k - \theta^\star\rVert \le \varepsilon$. These are not equivalent; a flat valley floor has a tiny gradient while $\theta$ is still far from $\theta^\star$.

**Worked example.** Fit $y \approx w x$ (no intercept) to three points $x = (1, 2, 3)$, $y = (2.1, 3.9, 6.2)$ with the mean squared error $L(w) = \frac13\sum_i (w x_i - y_i)^2$.

| $w$ | residuals $w x_i - y_i$ | $L(w)$ | $L'(w) = \frac{2}{3}\sum_i x_i (w x_i - y_i)$ |
|---|---|---|---|
| 1.5 | $(-0.6, -0.9, -1.7)$ | $1.353$ | $-5.00$ |
| 2.0 | $(-0.1, 0.1, -0.2)$ | $0.020$ | $-0.33$ |
| 2.1 | $(0.0, 0.3, 0.1)$ | $0.033$ | $+0.60$ |

The derivative changes sign between $2.0$ and $2.1$, so the minimum lies there. Setting $L'(w) = 0$ gives $w^\star = \sum x_i y_i / \sum x_i^2 = 28.5/14 = 2.036$ with $L^\star = 0.0140$. In one dimension you can solve this by hand; the rest of the module is about the case $d = 10^9$ where you cannot.

**Real-life example (healthcare).** In a phase-I dose-finding study the "parameter" is the dose, and the objective is efficacy minus a penalty for toxicity. The landscape is one-dimensional, the two terms pull in opposite directions, and the minimiser is the therapeutic window. Trials move the dose step by step based on observed responses: exactly the structure of iterative optimisation, with the difference that each "gradient evaluation" costs a cohort of patients, so the step-size rule matters enormously.

**In ML/AI this is …** the loop inside every `model.fit(...)`: `loss.backward(); optimizer.step()` in PyTorch, `model.compile(optimizer=..., loss=...)` in Keras. The `loss` is $L$, the `optimizer` is the update rule, and the stopping criterion is usually a fixed number of epochs plus early stopping on a validation metric.

## 2. Convexity

**Intuition.** A convex function is bowl-shaped: the straight chord between any two points on its graph lies *above* the graph. Such a landscape has no false valleys, so a hiker who only ever walks downhill cannot get stuck anywhere except the bottom. Most of classical statistics lives in this world; deep learning does not, which is why initialisation and random seeds matter there.

**Definition.** A set $C$ is convex if $\alpha x + (1-\alpha) y \in C$ for all $x, y \in C$ and $\alpha \in [0, 1]$. A function $f: C \to \mathbb{R}$ is convex if for all such $x, y, \alpha$

$$
f\big(\alpha x + (1-\alpha) y\big) \le \alpha f(x) + (1-\alpha) f(y).
$$

Applied to a weighted average of many points this becomes *Jensen's inequality*, $f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$. Two equivalent tests, each for $f$ smooth enough to state it:

- *First-order* (for differentiable $f$): $f(y) \ge f(x) + \nabla f(x)^\top (y - x)$ for all $x, y$ — the tangent plane lies below the graph.
- *Second-order* (for twice-differentiable $f$): the Hessian is positive semi-definite everywhere, $\nabla^2 f(x) \succeq 0$ (all eigenvalues $\ge 0$).

$f$ is *$\mu$-strongly convex* if $\nabla^2 f \succeq \mu I$ with $\mu > 0$ (curvature at least $\mu$ in every direction), and *$L$-smooth* if $\nabla^2 f \preceq L I$ (curvature at most $L$). The ratio $\kappa = L/\mu$ is the condition number of the problem.

*Why convex implies "local = global"*: let $x^\star$ be a stationary point, $\nabla f(x^\star) = 0$. The first-order condition gives $f(y) \ge f(x^\star) + 0^\top(y - x^\star) = f(x^\star)$ for every $y$: the tangent at a stationary point is flat, and the whole graph lies above it (the dotted line in Figure 1(b)). So any stationary point is a global minimum, and with strict convexity it is unique.

**Worked example.** Jensen on $f(x) = x^2$ with $x = 1$, $y = 3$, $\alpha = \tfrac12$: $f(2) = 4 \le \tfrac12(1 + 9) = 5$ — the gap at the midpoint of the chord in Figure 1(a). The first-order test on the same function, with $x = 1$ and $y = 3$: the tangent at $1$ is the line $t \mapsto 1 + 2(t - 1) = 2t - 1$, which at $t = 3$ predicts $f(1) + f'(1)(3 - 1) = 5$, below $f(3) = 9$ (Figure 1(b)). Hessian test on $f(x, y) = x^2 + xy + y^2$: $\nabla^2 f = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$ with eigenvalues $1$ and $3$, so $f$ is $1$-strongly convex and $3$-smooth, $\kappa = 3$. Non-example: $f(x) = x^4 - x^2$ has $f''(x) = 12x^2 - 2$, negative for $\lvert x\rvert < 1/\sqrt6 \approx 0.41$, e.g. $f''(0) = -2 < 0$; it has two minima at $x = \pm 1/\sqrt 2$, where $f = -\tfrac14$, and a local maximum $f(0) = 0$ between them. The chord from $-1/\sqrt2$ to $1/\sqrt2$ is flat at height $-\tfrac14$ and passes *below* the graph at $x = 0$ (Figure 1(c)). For ML losses: the MSE of a linear model has Hessian $\tfrac{2}{n} X^\top X \succeq 0$, and the logistic loss has Hessian $\tfrac1n X^\top S X$ with $S = \mathrm{diag}\big(p_i(1 - p_i)\big) \succeq 0$ — both convex. A neural network with one hidden layer is *not*: swapping two hidden units gives a different $\theta$ with identical loss, and the straight line between these two minima passes through worse parameters, which a convex function forbids.

![Three panels. Left: the parabola x squared with its chord from (1, 1) to (3, 9); at x = 2 the chord is at 5 and the graph at 4. Middle: the same parabola with its tangent line at x = 1, which reaches 5 at x = 3 where the graph is at 9, and a dotted flat tangent at the minimum x = 0. Right: the double-well curve x to the fourth minus x squared, with a shaded band around the origin where it curves downwards, two minima at minus and plus one over root two joined by a flat chord at minus 0.25, and the local maximum at the origin above that chord.](figures/convexity-tests.svg)

**Figure 1.** The convexity tests on the worked example. (a) Chord test on $f(x) = x^2$: the chord from $(1, 1)$ to $(3, 9)$ lies above the graph; at the midpoint $x = 2$ the chord is at $\tfrac12(1 + 9) = 5$ (square) and the graph at $f(2) = 4$ (dot) — Jensen's inequality. (b) First-order test on the same graph: the tangent at $x = 1$, $y = 2x - 1$, lies below the graph everywhere; at $x = 3$ it gives $f(1) + f'(1)\cdot 2 = 5$ against $f(3) = 9$. The dotted flat tangent at the stationary point $x = 0$ lies below the whole graph, which is the "local = global" argument in one picture. (c) The non-example $f(x) = x^4 - x^2$: in the shaded band $\lvert x\rvert < 1/\sqrt6 = 0.41$ the second derivative $12x^2 - 2$ is negative; the two minima at $\pm 1/\sqrt2$ have value $-0.25$, and the chord between them passes below the local maximum $f(0) = 0$ (open circle). The Hessian example $x^2 + xy + y^2$ (eigenvalues $1$ and $3$) is not drawn.

**Real-life example (manufacturing).** A plant's weekly cost of producing $q$ units is $C(q) = 500 + 20 q + 0.5 q^2$ (the quadratic term is overtime and wear). Revenue is $100 q$. Profit $P(q) = 80 q - 0.5 q^2 - 500$ is concave ($P'' = -1$), so its single stationary point $q = 80$ (profit 2 700) is the global optimum; the planner can stop as soon as the marginal profit $P'(q) = 80 - q$ hits zero. Diminishing returns are concavity.

**In ML/AI this is …** the reason `sklearn.linear_model.LogisticRegression`, `Ridge` and `svm.SVC` return the same answer regardless of initialisation: their objectives are convex, so every solver lands at the one global optimum. In deep learning the loss is non-convex, which is why papers report means over seeds and why "loss landscape" is a research topic.

## 3. Gradient descent and the step size

**Intuition.** Take a step against the gradient; repeat. The only free choice is the step length $\eta$. Too small and you crawl; too large and you overshoot the valley floor, land higher on the other side, and the oscillation grows until the loss is `nan`. The right $\eta$ is set by the *curvature* of the loss, which is why the eigenvalues of the Hessian appear immediately.

**Definition.** Gradient descent (GD) with step size (learning rate) $\eta > 0$ is

$$
\theta_{k+1} = \theta_k - \eta\, \nabla L(\theta_k).
$$

To analyse it, take the quadratic $L(\theta) = \tfrac12 \theta^\top A \theta$ with $A$ symmetric positive definite, eigenvalues $0 < \mu = \lambda_1 \le \dots \le \lambda_d = L$, and minimiser $\theta^\star = 0$. In the eigenbasis of $A$ the coordinates $z_i$ of $\theta$ decouple:

$$
z_i^{(k+1)} = z_i^{(k)} - \eta \lambda_i z_i^{(k)} = (1 - \eta\lambda_i)\, z_i^{(k)}, \qquad\text{so}\qquad z_i^{(k)} = (1 - \eta \lambda_i)^k z_i^{(0)}.
$$

Every coordinate shrinks geometrically iff $\lvert 1 - \eta\lambda_i\rvert < 1$ for all $i$, i.e. iff $\eta < 2/L$. The slowest coordinate sets the rate $\rho(\eta) = \max\big(\lvert 1 - \eta\mu\rvert, \lvert 1 - \eta L\rvert\big)$, which is minimised when the two terms are equal: $\eta^\star = 2/(L + \mu)$, giving

$$
\rho^\star = \frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1}, \qquad \kappa = \frac{L}{\mu}.
$$

With the safer choice $\eta = 1/L$ the rate is $1 - 1/\kappa$, so reaching accuracy $\varepsilon$ costs about $\kappa \ln(1/\varepsilon)$ steps: the cost of GD is *linear in the condition number*. Exact line search (choose $\eta_k$ to minimise $L$ along the ray; for a quadratic $\eta_k = g^\top g / g^\top A g$) helps but cannot beat the same $\kappa$ dependence. For general smooth, strongly convex $L$ the same statements hold with $L$ and $\mu$ the extreme Hessian eigenvalues over the region visited.

**Worked example.** Let $A = \mathrm{diag}(1, 10)$, so $\mu = 1$, $L = 10$, $\kappa = 10$, start $\theta_0 = (10, 1)$.

| $\eta$ | factors $(1 - \eta\lambda_i)$ | $\theta_1$ | $\theta_2$ | $\theta_3$ | behaviour |
|---|---|---|---|---|---|
| $0.10$ | $(0.9,\ 0)$ | $(9,\ 0)$ | $(8.1,\ 0)$ | $(7.29,\ 0)$ | stiff direction solved in one step, flat direction crawls (131 steps to $10^{-6}$) |
| $0.19$ | $(0.81,\ -0.9)$ | $(8.1,\ -0.9)$ | $(6.56,\ 0.81)$ | $(5.31,\ -0.73)$ | zig-zag: the stiff coordinate flips sign every step |
| $0.21$ | $(0.79,\ -1.1)$ | $(7.9,\ -1.1)$ | $(6.24,\ 1.21)$ | $(4.93,\ -1.33)$ | $0.21 > 2/L = 0.2$: the stiff coordinate *grows* — divergence |

The optimum $\eta^\star = 2/11 = 0.182$ gives rate $9/11 = 0.818$, i.e. about 69 steps to $10^{-6}$ instead of 131. No constant step can do better, because any $\eta$ that is good for the flat direction is bad for the stiff one — this is the zig-zag picture on an elongated bowl. Figure 2(a) draws the three rows of the table on that bowl, and Figure 2(b) shows the two shrink factors as functions of $\eta$: the best constant step sits where the slowly falling line of the flat direction meets the rising arm of the stiff direction's V.

![Two rows. Top: elliptical contours of a bowl ten times steeper vertically than horizontally, with eight gradient-descent steps from (10, 1) for each of three step sizes: one drops onto the horizontal axis in one step and crawls along it, one zig-zags inward, and one zig-zags with growing amplitude. Bottom: the per-step shrink factor of the flat and of the steep direction against the step size, a slowly falling line and a V with its zero at 0.1, crossing at the best step 2/11; beyond 0.2 the larger factor exceeds one.](figures/gd-step-size.svg)

**Figure 2.** The worked example, $L(\theta) = \tfrac12(\theta_1^2 + 10\theta_2^2)$ from $\theta_0 = (10, 1)$, where the loss is $L(\theta_0) = 55$. (a) Eight GD steps for each step size of the table. $\eta = 0.10$ (circles, solid) solves the stiff coordinate in one step, landing on $(9, 0)$, and then crawls along the flat axis with factor $0.9$. $\eta = 0.19$ (squares, solid) flips the sign of $\theta_2$ every step but shrinks it (factor $-0.9$). $\eta = 0.21$ (triangles, dashed) flips it and grows it (factor $-1.1$), reaching $\theta_2 = 2.14$ at step 8 while $\theta_1$ is still shrinking. (b) The shrink factor of each eigen-direction against $\eta$: flat $\lvert 1 - \eta\rvert$, stiff $\lvert 1 - 10\eta\rvert$ (the V with its zero at $\eta = 0.1$) and the rate $\rho(\eta)$, the larger of the two (dashed). The best step is where the two lines cross, $\eta^\star = 2/11 = 0.182$ with $\rho^\star = 9/11 = 0.818$ (69 steps per factor $10^{-6}$); beyond $2/L = 0.2$ (shaded) the stiff factor exceeds 1 and the error grows. $\eta = 0.10$ and $\eta = 0.19$ have the same rate $0.9$ (131 steps) for opposite reasons: the first is held back by the flat direction, the second by the stiff one.

**Real-life example (climate).** Fit a linear warming trend, temperature $\approx a + b\cdot\text{year}$, to 41 annual means for 1980–2020 by GD on the MSE. With the raw year as the feature, the Hessian $\tfrac{2}{n}X^\top X$ has eigenvalues $\approx 7\times10^{-5}$ and $8\times 10^{6}$: $\kappa \approx 10^{11}$, so a stable step must be below $2.5\times10^{-7}$ and the intercept direction would take on the order of $10^{11}$ steps. Centring the year ($\text{year} - 2000$) brings $\kappa$ down to $140$: the Hessian becomes $\mathrm{diag}(2, 280)$, because the centred years have mean $0$ and variance $140$, and GD at $\eta = 1/L$ removes the slope error in one step and then crawls along the intercept with factor $1 - 2/280 = 0.993$ per step (Figure 3(a)). Standardising the year to unit variance gives the Hessian $2I$, $\kappa = 1$, and GD with $\eta = 1/L$ converges in a single step (Figure 3(b)). Nothing about the data changed — only its parametrisation.

![Three panels. Top: very elongated elliptical contours, about 12 to 1, on which gradient descent removes the slope error in its first step and then crawls along the intercept axis, with tick marks every 20 steps bunching up near the optimum. Bottom left: circular contours of the standardised problem, where a single arrow goes from the start straight to the optimum. Bottom right: logistic loss against the gradient-descent step on a log axis for the breast-cancer data, the raw-feature curve still near 0.3 after 2000 steps and the standardised curve down at 0.1.](figures/feature-scaling.svg)

**Figure 3.** Feature scaling changes the shape of the bowl, not the data. (a) The climate fit with the centred year: the Hessian $\mathrm{diag}(2, 280)$ has $\kappa = 140$, so the contours of the excess loss are ellipses $\sqrt{140} \approx 11.8$ times longer than wide. GD at $\eta = 1/L = 1/280$ removes the slope error in its first step and then crawls along the intercept axis with factor $0.993$ per step (tick marks every 20 steps): the intercept error is $0.993$ of its starting value after step 1, $0.488$ after 100 steps, $0.116$ after 300 and $0.00077$ after 1 000. (b) The standardised year: Hessian $2I$, $\kappa = 1$, circular contours, and one step at $\eta = 1/L = 1/2$ lands on the optimum. The raw-year problem cannot be drawn at all: its ellipse would be $\sqrt{1.14\times10^{11}} \approx 338\,000$ times longer than wide. The start points in (a) and (b) are illustrative — the text gives no temperatures, and the shape of the bowls depends only on the years. (c) Case study 1 on scikit-learn's bundled breast-cancer data: logistic loss against GD step (log scale) from $\theta = 0$, where the loss is $\ln 2 = 0.693$ (dotted). Raw features at $\eta = 1/L = 2.4\times10^{-6}$ (dashed) are still at $0.304$ after 2 000 steps ($90.9\%$ correct); standardised features at $\eta = 0.3$ (solid) reach $0.1003$ after 200 steps and $0.0996$ after 2 000 ($98.6\%$). Both runs include the case study's ridge term $\lambda = 10^{-2}$, which leaves the raw run's numbers unchanged.

**In ML/AI this is …** the `lr` argument of `torch.optim.SGD`, and the reason `StandardScaler` precedes `SGDRegressor` in every scikit-learn pipeline. Batch normalisation and layer normalisation are, in part, in-network feature scaling that keeps the effective $\kappa$ of each layer's problem moderate. The condition number here is the one from Module 03, applied to the Hessian rather than to the data matrix.

## 4. Stochastic gradient descent

**Intuition.** The full gradient is an average over all $n$ examples. When $n$ is $10^9$ you cannot afford one such average per step — but you do not need it. A random mini-batch gives an estimate that is right on average and wrong by an amount that shrinks with the batch size. SGD trades gradient accuracy for many more, cheaper steps, and the noise it introduces turns out to be partly a feature.

**Definition.** With a mini-batch $B \subset \{1, \dots, n\}$ of size $b$ drawn uniformly at random,

$$
g_B(\theta) = \frac1b \sum_{i \in B} \nabla \ell_i(\theta), \qquad \mathbb{E}[g_B(\theta)] = \nabla L(\theta),
$$

so the mini-batch gradient is an *unbiased* estimate of the full one. Its covariance is about $\Sigma/b$ — exactly $\tfrac{\Sigma}{b}\cdot\tfrac{n-b}{n-1}$ when sampling without replacement — where $\Sigma$ is the covariance of the per-example gradients. The SGD step is $\theta_{t+1} = \theta_t - \eta_t\, g_{B_t}(\theta_t)$, and an *epoch* is one pass over the data, $n/b$ steps. Noise has a price: for the 1-D quadratic $L = \tfrac{\lambda}{2}\theta^2$ with gradient noise of variance $\sigma^2$, the iterates do not converge to $0$ but hover with stationary variance

$$
\mathrm{Var}(\theta_\infty) = \frac{\eta\,\sigma^2}{\lambda(2 - \eta\lambda)} \approx \frac{\eta\, \sigma^2}{2\lambda},
$$

a "noise ball" whose size is proportional to $\eta$. Hence the learning-rate *schedule*: a large $\eta$ early for fast progress, then decay so the ball shrinks. The classical conditions $\sum_t \eta_t = \infty$, $\sum_t \eta_t^2 < \infty$ (e.g. $\eta_t \propto 1/t$) guarantee convergence; in practice step decay, cosine decay $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})\big(1 + \cos(\pi t/T)\big)$ and a linear *warm-up* from $0$ over the first few hundred steps (which protects against early large, poorly-estimated steps) are the common choices.

*Why the mini-batch gradient is unbiased.* A uniformly random batch of size $b$ contains any given example $i$ with probability $\binom{n-1}{b-1}\big/\binom{n}{b} = b/n$. Hence

$$
\mathbb{E}[g_B] = \frac1b\sum_{i=1}^{n} P(i \in B)\,\nabla\ell_i = \frac1b\cdot\frac{b}{n}\sum_{i=1}^{n}\nabla\ell_i = \nabla L
$$

(a regulariser $\lambda R$, if present, is added unchanged to every batch gradient). Which examples a particular batch leaves out does not matter; what matters is that every example is equally likely to be in it. A sampler that favours some examples is biased unless it reweights them, which is the idea of importance sampling ([Module 18](../../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md), Section 4).

*Why the variance shrinks like $1/b$.* With replacement, the $b$ terms are independent draws, each with covariance $\Sigma$ (the covariance of the $n$ per-example gradients, divisor $n$), and the covariance of their mean is $\Sigma/b$ — the variance-of-a-mean rule of [Module 07](../../part-3-probability/07-probability-foundations/README.md) (Section 7). Without replacement, two different draws are slightly negatively correlated, with covariance $-\Sigma/(n-1)$, because an example drawn once cannot be drawn again; then

$$
\mathrm{Cov}(g_B) = \frac{1}{b^2}\Big(b\,\Sigma - b(b-1)\,\frac{\Sigma}{n-1}\Big) = \frac{\Sigma}{b}\cdot\frac{n-b}{n-1},
$$

the finite-population correction of [Module 10](../../part-4-statistics/10-estimation-and-sampling/README.md) (Section 1), negligible when $b \ll n$. The standard deviation therefore falls only like $1/\sqrt b$: doubling the batch doubles the cost of a step but shrinks the noise by a factor of only $\sqrt2$.

*Where the noise ball comes from.* Model the stochastic gradient on $L = \tfrac\lambda2\theta^2$ as $g_t = \lambda\theta_t + \xi_t$, with independent noise $\xi_t$ of mean $0$ and variance $\sigma^2$. Then $\theta_{t+1} = (1 - \eta\lambda)\,\theta_t - \eta\,\xi_t$, and because $\xi_t$ is independent of $\theta_t$ the variances add:

$$
v_{t+1} = (1 - \eta\lambda)^2\, v_t + \eta^2\sigma^2, \qquad v_t = \mathrm{Var}(\theta_t).
$$

For $\eta < 2/\lambda$ this recursion contracts to its fixed point $v = \eta^2\sigma^2/\big(1 - (1 - \eta\lambda)^2\big) = \eta\sigma^2/\big(\lambda(2 - \eta\lambda)\big)$, the formula above; the approximation $\eta\sigma^2/(2\lambda)$ needs $\eta\lambda \ll 1$. Here $\sigma^2$ is the variance of the *batch* gradient at the optimum, about $\Sigma/b$, so the ball shrinks with $\eta$ and with $1/b$ alike: halving the batch at constant $\eta$ doubles its variance, and halving $\eta$ as well restores it (approximately; exactly only in the small-$\eta$ approximation, since the factor $2 - \eta\lambda$ changes too). This is the reasoning behind the "linear scaling rule" of large-batch training, which raises $\eta$ in proportion to $b$. The expected excess loss inside the ball is $\tfrac\lambda2\,\mathrm{Var}(\theta_\infty) \approx \eta\sigma^2/4$: a floor proportional to $\eta$.

*What a schedule does.* The classical conditions $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ (Robbins and Monro) are sufficient for convergence, and each has a reading. With gradients bounded by $G$, the iterates can never move further than $G\sum_t\eta_t$ from the start, so the first sum must diverge for a distant minimum to be reachable. Each step injects $\eta_t^2\sigma^2$ of fresh noise variance, and the second condition keeps the total injected noise finite. $\eta_t \propto 1/t$ satisfies both. A constant step satisfies only the first, which is exactly why it ends in a noise ball; an exponentially decaying step $\eta_t = \eta_0\gamma^t$ satisfies only the second, and can travel at most $\eta_0 G/(1 - \gamma)$ in total, possibly stopping short of the minimum. Cosine decay over a fixed budget of $T$ steps is the finite-horizon version of "large early, small late": $82\%$ of the summed step sizes fall in the first half, and the last quarter, below $15\%$ of $\eta_{\max}$, shrinks the ball. Warm-up addresses the other end of the run: the first steps are taken where the curvature of a freshly initialised network is often large and, for Adam, where the running averages rest on very few gradients (Section 5).

*In practice.* At constant $\eta$, a training loss that settles on a noisy floor and drops when $\eta$ is halved is sitting in a noise ball, not at a converged point (the fourth row of the table in Section 8). The batch-gradient variance can be measured directly: compute $g_B$ on a few dozen mini-batches at the same $\theta$ and take their sample variance. The noise ball is also more than a nuisance: Langevin dynamics ([Module 18](../../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md), Section 7) adds Gaussian noise to gradient steps on purpose, so that the cloud of iterates becomes a sample from a distribution.

**Worked example.** Five points $x = (1, 2, 3, 4, 5)$, $y = (1.5, 3.9, 6.2, 8.1, 9.8)$, model $y \approx w x$, current $w = 1.5$. Per-example gradients $2 x_i (w x_i - y_i)$ are $(0, -3.6, -10.2, -16.8, -23.0)$, whose mean, $-10.72$, is the full gradient (Figure 4(a)). Mini-batches of size 2: $\{1, 4\}$ gives $-8.4$, $\{2, 3\}$ gives $-6.9$, $\{3, 5\}$ gives $-16.6$. Every estimate has the right sign (increase $w$) but the magnitudes differ by a factor of two or more; averaged over all ten possible batches they give exactly $-10.72$, because each example appears in $4$ of the $10$ batches, a fraction $2/5 = b/n$. The per-example variance is $\sigma^2 = 70.7$; the variance of a size-2 batch mean is $\tfrac{\sigma^2}{b}\cdot\tfrac{n-b}{n-1} = \tfrac{70.7}{2}\cdot\tfrac{3}{4} = 26.5$, which you can confirm by averaging over all ten batches. In standard deviations that is $8.4 \to 5.2$, not the $8.4/\sqrt2 = 5.9$ of sampling with replacement. Under a cosine schedule with $\eta_{\min} = 0$ and $T$ total steps, the step is $85\%$ of $\eta_{\max}$ at $t = T/4$, $50\%$ at $T/2$ and $15\%$ at $3T/4$ (Figure 4(c)).

The noise ball of the same problem: the curvature is $\lambda = L''(w) = \tfrac25\sum_i x_i^2 = 22$ and the optimum $w^\star = \sum_i x_i y_i/\sum_i x_i^2 = 109.3/55 = 1.987$. At $w^\star$ the per-example gradients are $(0.97, 0.30, -1.43, -1.21, 1.36)$, with variance $1.28$, so the $\sigma^2$ of the noise-ball formula — the variance of a size-2 batch gradient at the optimum, not the per-example $70.7$ at $w = 1.5$ above — is $\tfrac{1.28}{2}\cdot\tfrac34 = 0.48$. With $\eta = 0.03$, so that $\eta\lambda = 0.66$ is not small, the ball has $\mathrm{Var}(w_\infty) = \tfrac{0.03 \cdot 0.48}{22\,(2 - 0.66)} = 4.9\times10^{-4}$, a standard deviation of $0.022$ (the small-$\eta$ approximation would give $0.018$). A simulation measures $0.023$ (Figure 4(b)): close, but not exact, because each batch also has its own curvature, $\tfrac2b\sum_{i \in B} x_i^2$ between $5$ and $41$, which the additive-noise model ignores.

![Three panels. Left: five per-example gradients spread between minus 23 and 0, and below them the ten size-2 batch means, clustered more tightly, both rows centred on the full gradient minus 10.72. Middle: an SGD trace at constant step size hovering in a narrow band around the optimum 1.987, and a second trace, started at step 200 with a decaying step size, that settles close to it. Right: the cosine learning-rate schedule falling from 1 to 0, at 85, 50 and 15 percent at a quarter, a half and three quarters of the run, and the same schedule after a short linear warm-up.](figures/sgd-noise.svg)

**Figure 4.** (a) The worked example's gradient estimates at $w = 1.5$. Top row: the five per-example gradients $0, -3.6, -10.2, -16.8, -23.0$ (labelled by example number 1–5), standard deviation $8.41$. Bottom row: the means of all ten batches of size 2, with the three named in the text filled; they cluster more tightly, standard deviation $5.15 = 8.41\sqrt{\tfrac12\cdot\tfrac34}$, and both rows average exactly to the full gradient $-10.72$ (vertical line). (b) SGD with $b = 2$ from $w_0 = 1.5$ (below the panel); each step picks one of the ten batches at random. At constant $\eta = 0.03$ the iterate hovers around $w^\star = 1.987$ (dashed), mostly inside the band $w^\star \pm 2$ standard deviations (96% of steps 100–400), with the predicted standard deviation $\sqrt{\eta\sigma^2/(\lambda(2 - \eta\lambda))} = 0.022$ for $\lambda = 22$, $\sigma^2 = 0.48$; over steps 100–400 the measured one is $0.023$. From the dotted line at $t = 200$ the step size follows a cosine decay to zero instead (same batch sequence), and that trace settles close to $w^\star$ ($1.991$ at $t = 400$, within a fifth of a noise-ball standard deviation). The 400-step horizon, the switch to decay at $t = 200$ and the random seed are illustrative choices, not values from the text; the step size $0.03$ is the one of the noise-ball example above. (c) The cosine schedule (solid) at $85\%$, $50\%$ and $15\%$ of $\eta_{\max}$ at $T/4$, $T/2$ and $3T/4$, and the same schedule after a $5\%$ linear warm-up (dashed).

**Real-life example (e-commerce).** A recommender scores (user, item) pairs with a dot product of embeddings. Clicks arrive as a stream of millions of events per day; the "dataset" is never complete. The system applies one SGD step per event (or per small batch): update only the two embedding rows involved, cost independent of history size. Yesterday's log is one "epoch", the step size is decayed overnight, and new users start with a larger step so that their embedding moves quickly. Full-batch GD is not merely slower here — it is undefined, because the data never stops arriving.

**In ML/AI this is …** `DataLoader(batch_size=..., shuffle=True)` plus `CosineAnnealingLR` / `OneCycleLR` in PyTorch, `SGDClassifier` in scikit-learn, and the warm-up-then-decay schedule of the original Transformer. Module 15 returns to the observation that SGD's noise acts as an implicit regulariser that favours flat minima.

## 5. Momentum and adaptive methods

**Intuition.** A heavy ball rolling down the ill-conditioned bowl of Section 3 does not zig-zag: the sideways kicks alternate in sign and cancel, while the downhill pushes all point the same way and accumulate. That is momentum. Adaptive methods attack the same problem differently: give every parameter its own step size, dividing by a running estimate of that parameter's typical gradient magnitude, so stiff and flat directions receive comparable updates.

**Definition.** With $g_t = \nabla L(\theta_t)$ (or a mini-batch estimate):

- *Heavy-ball momentum* (Polyak): $v_t = \beta v_{t-1} + g_t$, $\theta_{t+1} = \theta_t - \eta v_t$. Under a constant gradient $v_t \to g/(1 - \beta)$, so the effective step is $\eta/(1-\beta)$: ten times $\eta$ for $\beta = 0.9$. On a quadratic with condition number $\kappa$, the tuned heavy ball ($\beta = \big(\tfrac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\big)^2$, $\eta = \big(\tfrac{2}{\sqrt L + \sqrt\mu}\big)^2$) converges at rate $\tfrac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$: the cost grows like $\sqrt\kappa$ instead of $\kappa$. For the bowl of Section 3 ($\kappa = 10$) that is $\beta = 0.270$, $\eta = 0.231$ and rate $0.519$, against $0.818$ for GD at its best constant step (Figure 5(a), (b)).
- *Nesterov momentum*: the same, but the gradient is evaluated at the look-ahead point $\theta_t - \eta\beta v_{t-1}$; it has the same $\sqrt\kappa$ rate with guarantees for general convex functions.
- *RMSProp*: $s_t = \rho s_{t-1} + (1 - \rho) g_t^2$ (element-wise), $\theta_{t+1} = \theta_t - \eta\, g_t / (\sqrt{s_t} + \epsilon)$.
- *Adam*: momentum on the gradient and on its square, both bias-corrected:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad \hat m_t = \frac{m_t}{1 - \beta_1^t},\quad \hat v_t = \frac{v_t}{1 - \beta_2^t},\quad \theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon},
$$

with defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.
- *AdamW*: $\theta_{t+1} = \theta_t - \eta\big(\hat m_t / (\sqrt{\hat v_t} + \epsilon) + \lambda \theta_t\big)$. Weight decay is applied *outside* the adaptive scaling. If you instead add $\lambda\theta$ to $g_t$ (L2 regularisation), the decay term gets divided by $\sqrt{\hat v_t}$ and parameters with large gradients are barely regularised.

Adam is not guaranteed to converge even on simple convex problems: with a short memory $\beta_2$, a rare large gradient can shrink the effective step just when it is needed and then let it grow again; AMSGrad (keep the running *maximum* of $v_t$) is the standard fix. Large models nevertheless use a *shorter* second-moment memory, $\beta_2 = 0.95$–$0.99$, for a stability reason of their own, explained under *Units* below.

*Exponential moving averages.* Unrolling Adam's first moment gives

$$
m_t = (1 - \beta_1)\sum_{s=1}^{t} \beta_1^{\,t-s}\, g_s ,
$$

a weighted average of past gradients whose weights decay geometrically: a gradient $k$ steps old counts $\beta_1^k$ times as much as the newest one. The memory is about $1/(1 - \beta)$ steps — 10 for $\beta_1 = 0.9$, 1 000 for $\beta_2 = 0.999$, 20 for $\beta_2 = 0.95$. The heavy-ball velocity (not to be confused with Adam's $v_t$) is the same sum without the factor $(1 - \beta)$, $v_t = \sum_{s \le t} \beta^{t-s} g_s$, which is why under a constant gradient it grows to $g/(1 - \beta)$ rather than to $g$.

*Where bias correction comes from.* The weights $(1 - \beta)\beta^{t-s}$, $s = 1, \dots, t$, sum to $1 - \beta^t$, not to $1$, because the averages start from $m_0 = v_0 = 0$. Under a constant gradient $g$ this gives $m_t = (1 - \beta_1^t)\,g$ and $v_t = (1 - \beta_2^t)\,g^2$, so dividing by $1 - \beta_1^t$ and $1 - \beta_2^t$ restores $g$ and $g^2$ exactly. Without the correction the step would be $\eta\,(1 - \beta_1^t)/\sqrt{1 - \beta_2^t}$: $3.16\,\eta$ at $t = 1$, a peak of $6.57\,\eta$ at $t = 12$ and still $3.24\,\eta$ at $t = 100$ (Figure 5(c)), and $1.26\,\eta$ at $t = 1\,000$ — several times too long, exactly when the estimates rest on the fewest gradients.

*Units, and the size of $\eta$.* $g_t$ and $\sqrt{v_t}$ have the same units (loss per unit of parameter), so $\hat m_t/\sqrt{\hat v_t}$ is a pure number of order 1, and each coordinate moves by roughly $\eta$ per step whatever the scale of the loss: multiply the loss by 1 000 and plain GD's steps grow 1 000-fold, while Adam's do not change (up to $\epsilon$). Adam's $\eta$ is therefore a distance in parameter space — $10^{-3}$ means "move each weight by about a thousandth per step" — which is why it transfers between problems. The same normalisation is a risk. After a long quiet stretch in which $m_t$ and $v_t$ have decayed to almost nothing, a sudden gradient $G$ gives $m_t \approx (1 - \beta_1)G$ and $v_t \approx (1 - \beta_2)G^2$, a step of $(1 - \beta_1)/\sqrt{1 - \beta_2}$ times $\eta$ in that coordinate whatever the size of $G$: $3.16\,\eta$ with $\beta_2 = 0.999$, but $0.45\,\eta$ with $\beta_2 = 0.95$, whose memory of about 20 steps is close to the 10 of $m_t$, so $v_t$ keeps pace with $m_t$ after a change of gradient scale and the step never rises much above $\eta$ (at most $1.10\,\eta$, against $6.57\,\eta$ for $\beta_2 = 0.999$).

*$\epsilon$.* It keeps the division safe for a coordinate whose gradients have been exactly zero, and it sets a floor: where $\sqrt{\hat v_t} \ll \epsilon$ the update is $\eta\,\hat m_t/\epsilon$, momentum SGD with learning rate $\eta/\epsilon$ on gradients that are themselves far below $\epsilon$. The default $10^{-8}$ is tiny on purpose; some training recipes raise it to temper the normalisation of very small gradients.

*L2 regularisation versus decoupled weight decay, worked.* Take two weights, both equal to $1$, with steady gradients $g_1 = 10$ and $g_2 = 0.01$, and $\lambda = 0.1$. L2 regularisation adds $\lambda\theta = 0.1$ to each gradient before Adam normalises it; with a steady gradient $\sqrt{\hat v} = \lvert g + 0.1\rvert$, so the decay's share of the step is $0.1/10.1 \approx 1\%$ for the first weight but $0.1/0.11 \approx 91\%$ for the second. Per step the large-gradient weight is pulled towards zero by $0.0099\,\eta$ and the small-gradient one by $0.91\,\eta$, about 92 times more. AdamW shrinks both by the same $\eta\lambda\theta = 0.1\,\eta$, as plain SGD with weight decay would. In code, `torch.optim.Adam(weight_decay=...)` is the L2 form and `torch.optim.AdamW` the decoupled one; ridge regression is the same penalty on a linear model ([Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md), Section 4), and [Module 16](../../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md) (Section 9) puts these pieces into a complete training loop.

**Worked example.** Momentum with $\beta = 0.9$ and constant gradient $g = 1$: $v = 1,\ 1.9,\ 2.71,\ 3.44,\ 4.10, \dots \to 10$. Adam's first step with $g_1 = 0.1$: $m_1 = 0.01$, $v_1 = 10^{-5}$. *Without* bias correction the update ratio is $m_1 / \sqrt{v_1} = 0.01 / 0.00316 = 3.16$, a step three times larger than $\eta$ in the first iteration, and the ratio keeps growing for a dozen steps (Figure 5(c)). *With* correction, $\hat m_1 = 0.1$, $\hat v_1 = 0.01$, ratio $= 1.0$ exactly. The same calculation with $g_1 = 100$ also gives ratio $1.0$: Adam's early steps have size $\approx \eta$ in every coordinate *regardless of the gradient's scale*. That is the per-parameter normalisation at work, and it is why Adam's $\eta$ is typically 10–100 times smaller than SGD's.

![Three panels. Top: the elongated bowl again, with 20 steps of gradient descent at its best step size zig-zagging slowly and 20 steps of tuned heavy-ball momentum reaching the minimum much sooner. Bottom left: the distance to the minimum on a log scale against the step, two nearly straight lines, the momentum one much steeper. Bottom right: Adam's step in units of the learning rate under a constant gradient, rising from 3.16 to a peak of 6.57 at step 12 and then slowly falling without bias correction, and flat at exactly 1 with it.](figures/momentum-adam.svg)

**Figure 5.** (a) The bowl of Figure 2(a), with 20 steps of each method from $\theta_0 = (10, 1)$. GD at its best constant step $\eta^\star = 2/11$ (circles) still zig-zags, with rate $0.818$. The tuned heavy ball for $\kappa = 10$, $\beta = \big(\tfrac{\sqrt{10} - 1}{\sqrt{10} + 1}\big)^2 = 0.270$ and $\eta = \big(\tfrac{2}{\sqrt{10} + 1}\big)^2 = 0.231$ (squares), passes $(7.69, -1.31)$, $(5.29, 1.09)$, $(3.42, -0.78)$ and reaches the minimum far sooner; the tags mark step 5, $(3.67, -0.37)$ for GD and $(1.29, -0.33)$ for the heavy ball. (b) The distance $\lVert\theta_k - \theta^\star\rVert$ on a log scale. GD lies exactly on the dotted line $10.05\times0.818^k$; the heavy ball follows the steeper dotted line $10.05\times0.519^k$ but drifts above it, by a factor of about 16 at $k = 30$, because the tuned heavy ball is critically damped and its error behaves like $k\rho^k$. To shrink the distance by $10^{-6}$ GD needs 69 steps and the heavy ball 21 by its rate (26 in this run): the cost grows like $\sqrt\kappa$, not $\kappa$. The common heavy-ball setting $\beta = 0.9$ with $\eta = 0.1$ (not drawn) also converges on this bowl, but at rate $\sqrt{0.9} = 0.949$, slower here than GD, so $\beta = 0.27$ is the tuned value for $\kappa = 10$, not a general recommendation. (c) Adam's step in units of $\eta$ under a constant gradient $g$. Without bias correction the ratio $m_t/\sqrt{v_t}$ is $3.16$ at $t = 1$, peaks at $6.57$ at $t = 12$ and is still $3.24$ at $t = 100$, identical for $g = 0.1$ (line) and $g = 100$ (open circles); with bias correction the step is exactly $\eta$ for any $g$.

**Real-life example (computer vision).** You train an image classifier with plain SGD and the loss drops quickly, then sits at $1.9$ for twenty epochs before falling again. The plateau is a region where the gradient is tiny in most directions (a saddle or a flat shelf), so each step moves almost nothing. Momentum keeps the velocity built up before the plateau and carries you across; Adam rescales the tiny gradient to a step of size $\approx\eta$ and marches through. Both convert a twenty-epoch plateau into a few hundred steps — the standard remedy when a loss curve "stalls but has not converged".

**In ML/AI this is …** `torch.optim.SGD(momentum=0.9, nesterov=True)` for convolutional networks, and `torch.optim.AdamW` for essentially every transformer. GPT-3 was trained with $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $0.1$. Adam keeps two extra float32 numbers per parameter; for a 175-billion-parameter model that is $175\times10^9 \times 8$ bytes $\approx 1.4$ TB of optimiser state alone, which is why sharded (ZeRO) and 8-bit optimiser states exist.

## 6. Second-order methods

**Intuition.** Gradient descent sees only the slope and has to guess how far to go. Newton's method also measures the curvature, fits a parabola to the loss, and jumps straight to that parabola's minimum. On a quadratic it finishes in one step; near the minimum of any smooth function the number of correct digits roughly doubles per iteration. The price is forming and solving with a $d \times d$ Hessian.

**Derivation and statement.** Expand to second order around $\theta_k$ with $g = \nabla L(\theta_k)$ and $H = \nabla^2 L(\theta_k)$:

$$
L(\theta_k + p) \approx L(\theta_k) + g^\top p + \tfrac12 p^\top H p .
$$

Minimising the right-hand side over $p$ (gradient $g + Hp = 0$) gives the Newton step

$$
p = -H^{-1} g, \qquad \theta_{k+1} = \theta_k - H^{-1}\nabla L(\theta_k).
$$

If $H$ is positive definite and Lipschitz continuous near a minimiser $\theta^\star$, then for $\theta_k$ close enough $\lVert\theta_{k+1} - \theta^\star\rVert \le C\lVert\theta_k - \theta^\star\rVert^2$ — *quadratic convergence*. The caveats: (i) the Hessian costs $O(n d^2)$ to form and $O(d^3)$ to solve against, hopeless for $d = 10^8$; (ii) far from a minimum $H$ may be indefinite and the step may go uphill, so one *damps* it by solving $(H + \tau I)p = -g$ and/or adds a line search; (iii) Newton finds *stationary* points, not minima. Quasi-Newton methods avoid the Hessian: BFGS maintains an approximation $B_k^{-1}$ updated from the pairs $s_k = \theta_{k+1} - \theta_k$, $y_k = g_{k+1} - g_k$ so that the secant condition $B_{k+1} s_k = y_k$ holds; *L-BFGS* keeps only the last $m \approx 10$ pairs, costing $O(md)$ per step. For least-squares objectives $L = \tfrac12\lVert r(\theta)\rVert^2$, *Gauss–Newton* uses $H \approx J^\top J$ with $J$ the Jacobian of the residuals, and *Levenberg–Marquardt* solves $(J^\top J + \tau I)p = -J^\top r$, interpolating between Gauss–Newton ($\tau \to 0$) and small gradient steps ($\tau$ large).

**Worked example.** Minimise $f(x) = x^3/3 - 2x$, so $f'(x) = x^2 - 2$ and $f''(x) = 2x$; the minimum is at $\sqrt2$. From $x_0 = 1$ the Newton iteration $x \leftarrow x - (x^2 - 2)/(2x)$ gives

| $k$ | $x_k$ | error $\lvert x_k - \sqrt2\rvert$ |
|---|---|---|
| 0 | $1$ | $4.1\times10^{-1}$ |
| 1 | $1.5$ | $8.6\times10^{-2}$ |
| 2 | $1.41667$ | $2.5\times10^{-3}$ |
| 3 | $1.4142157$ | $2.1\times10^{-6}$ |
| 4 | $1.41421356237$ | $1.6\times10^{-12}$ |

The error is squared (times $\approx 0.35$) at every step; on a log scale the exponent doubles (Figure 6(b)). Geometrically, each step jumps to the vertex of the parabola that matches $f$, $f'$ and $f''$ at the current point: at $x_0 = 1$ that parabola is $-\tfrac53 - (x - 1) + (x - 1)^2$, with its vertex at $x_1 = 1.5$ (Figure 6(a)). From $x_0 = -1$ the same iteration converges just as fast to $-\sqrt2$ — which is the local *maximum* of $f$, since $f''(-\sqrt2) < 0$. At the start, $f''(-1) = -2$: the fitted parabola opens downwards, and its vertex is the top of a hill (Figure 6(c)). Newton solves $f' = 0$; it does not know the difference.

![Three panels. Left: the cubic x cubed over three minus 2x near its minimum, with the parabola fitted at x = 1, whose vertex at 1.5 is the next Newton iterate, and a second parabola fitted there, whose vertex is practically the minimum at root two. Middle: errors on a log scale against the iteration, two Newton curves that plunge faster and faster to about ten to the minus 12 and ten to the minus 13, and a gradient-descent line that falls slowly and straight. Right: the same cubic near minus root two, where the parabola fitted at minus 1 opens downwards and Newton jumps to its top, towards the local maximum.](figures/newton-parabola.svg)

**Figure 6.** (a) $f(x) = x^3/3 - 2x$ and the parabola Newton fits at $x_0 = 1$, $q_0(x) = -\tfrac53 - (x - 1) + (x - 1)^2$ (dashed). Its vertex is $x_1 = 1.5$, where the parabola is at $-1.917$ and $f$ itself at $-1.875$ (open circle). The parabola fitted at $x_1 = 1.5$ (dash-dot) has its vertex at $x_2 = 1.41667$, practically on the minimum $\sqrt2 = 1.41421$ (star). (b) Error against iteration on a log scale. Newton on $x^3/3 - 2x$, $\lvert x_k - \sqrt2\rvert$ (circles): $4.1\times10^{-1}$, $8.6\times10^{-2}$, $2.5\times10^{-3}$, $2.1\times10^{-6}$, $1.6\times10^{-12}$. Newton on Case study 1, the gradient norm $\lVert\nabla L(\theta_k)\rVert$ (squares): a few slower steps, then the same plunge, from $1.4$ to $1.5\times10^{-13}$ in eight iterations. For contrast, GD on the Section 3 quadratic at $\eta = 0.1$, $\lVert\theta_k\rVert = 10\times0.9^k$ for $k \ge 1$ (dotted, triangles), is a straight line. The panel mixes two error measures and three problems; compare the shapes, not the heights. (c) The same iteration from $x_0 = -1$, where $f''(-1) = -2 < 0$: the fitted parabola $\tfrac53 - (x + 1) - (x + 1)^2$ opens downwards, Newton jumps to its top, $x_1 = -1.5$ (value $1.917$), and converges to $-\sqrt2$, the local maximum of $f$ (value $1.886$).

**Real-life example (physics/engineering).** Predicting a satellite's position requires solving Kepler's equation $E - e\sin E = M$ for the eccentric anomaly $E$, thousands of times per second in a GPS receiver. Newton on $F(E) = E - e\sin E - M$ with $F'(E) = 1 - e\cos E$, $e = 0.5$, $M = 1$, starting at $E_0 = 1$: $1 \to 1.5765 \to 1.50021 \to 1.4987017 \to 1.4987011335$, with $\lvert F\rvert = 9\times10^{-14}$ after four steps. Calibrating a physical model (reaction-rate constants, lens distortion, a battery model) against measurements is a nonlinear least-squares problem, and Levenberg–Marquardt is the workhorse.

**In ML/AI this is …** `LogisticRegression(solver='lbfgs')`, the scikit-learn default, and `scipy.optimize.minimize(method='L-BFGS-B')`; the IRLS algorithm that fits generalised linear models is exactly Newton's method for canonical links such as the logit (in general it is Fisher scoring, Newton with the expected Hessian); `scipy.optimize.curve_fit` is Levenberg–Marquardt. For deep networks the full Hessian is out of reach, but Hessian-vector products come free from autodiff (Module 06) and K-FAC and Shampoo are practical approximate second-order optimisers; the Fisher-information view of curvature returns in Module 11.

## 7. Constrained optimisation: Lagrange multipliers and KKT

**Intuition.** You walk along a fence (the constraint) looking for the lowest point of a hillside. At the best point along the fence, the fence must be tangent to the contour line of the hill: if it crossed the contour, walking a little further along the fence would still take you downhill. Tangent contour and fence means their normals are parallel, i.e. the gradient of the objective is a multiple of the gradient of the constraint. That multiple is the Lagrange multiplier, and it measures how much the optimum would improve if the fence were moved.

**Statement.** For $\min f(x)$ subject to $h(x) = 0$ define the Lagrangian $\mathcal{L}(x, \lambda) = f(x) + \lambda h(x)$. At a constrained optimum (with $\nabla h \ne 0$)

$$
\nabla f(x^\star) + \lambda \nabla h(x^\star) = 0, \qquad h(x^\star) = 0 .
$$

With this sign convention $-\lambda$ is the rate at which the optimal value changes when the constraint level is relaxed (the *shadow price*). For inequality constraints $g_j(x) \le 0$ and equality constraints $h_i(x) = 0$, the *Karush–Kuhn–Tucker (KKT)* conditions at $x^\star$ are

1. stationarity: $\nabla f + \sum_i \lambda_i \nabla h_i + \sum_j \mu_j \nabla g_j = 0$;
2. primal feasibility: $h_i(x^\star) = 0$, $g_j(x^\star) \le 0$;
3. dual feasibility: $\mu_j \ge 0$ — an inequality can only push you *away* from the forbidden side;
4. complementary slackness: $\mu_j\, g_j(x^\star) = 0$ — a constraint that is not active ($g_j < 0$) has zero multiplier, and a constraint with positive multiplier is active ($g_j = 0$).

*Why $\mu_j \ge 0$.* Take a single inequality $g(x) \le 0$ that is active at $x^\star$: $x^\star$ lies on the boundary, and $g$ increases towards the forbidden side, so $\nabla g$ points out of the feasible set. Stationarity reads $-\nabla f(x^\star) = \mu\,\nabla g(x^\star)$: the direction of steepest descent must point straight out of the feasible set, into the wall. If $\mu$ were negative, $-\nabla f$ would point into the feasible set; a small step along it would keep $g \le 0$ and lower $f$, so $x^\star$ would not be optimal. In the one-dimensional worked example below, $(x - 2)^2$ with $g(x) = x - 1 \le 0$, stationarity is $-f'(1) = 2 = \mu\,g'(1)$, so $\mu = 2$: the push of the negative gradient against the wall in Figure 7(b).

*Complementary slackness is a case split.* For each inequality, either $g_j(x^\star) < 0$ — the constraint is slack, could be deleted without disturbing $x^\star$ as a local optimum (for a convex problem, as the global one too; Pitfall 7 says why not in general), and has $\mu_j = 0$ — or $g_j(x^\star) = 0$, and it behaves like an equality whose multiplier must come out non-negative. With $m$ inequalities there are up to $2^m$ cases, and the *active-set method* searches them sensibly: guess which constraints are active, solve the resulting equality-constrained problem, then check that the other constraints hold and that the active multipliers are $\ge 0$; if a check fails, release a constraint with a negative multiplier or add a violated one, and repeat. Case study 3 runs exactly this with the guess $w_1 = 0$, and Figure 7(c) shows why it is the natural first guess: the unconstrained optimum lies beyond the wall $w_1 = 0$; the KKT check in the case study then confirms it.

For convex $f, g_j$ and affine $h_i$ these conditions are necessary *and* sufficient (under a mild regularity condition). The *dual problem* $\max_{\lambda,\, \mu \ge 0} \min_x \mathcal{L}(x, \lambda, \mu)$ always gives a lower bound on the optimal value (weak duality) and equals it for such convex problems (strong duality). When KKT is awkward to solve, two practical alternatives are the *penalty method* — minimise $f(x) + \rho\lVert h(x)\rVert^2$ with growing $\rho$ — and *projected gradient descent* $x \leftarrow \Pi_C\big(x - \eta \nabla f(x)\big)$, where $\Pi_C$ snaps the iterate back onto the feasible set (clip to $[0, \infty)$, normalise to the unit sphere, …).

*Weak duality in one line.* For any feasible $x$, any $\lambda$ and any $\mu \ge 0$, $\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_i \lambda_i \cdot 0 + \sum_j \mu_j\, g_j(x) \le f(x)$, because every $\mu_j g_j(x) \le 0$. Minimising the left-hand side over *all* $x$ can only lower it, so the *dual function* $d(\lambda, \mu) = \min_x \mathcal{L}(x, \lambda, \mu)$ satisfies $d(\lambda, \mu) \le f^\star$: every choice of multipliers certifies a lower bound, and the dual problem looks for the best one. ML cares because the dual can be the easier problem. The SVM dual has one variable per training point, simple box constraints $0 \le \alpha_i \le C$ plus a single linear equality, and sees the data only through inner products $x_i^\top x_j$, which is what makes kernels possible; the multipliers of the maximum-entropy problem become the weights of the softmax/Gibbs form ([Module 14](../../part-5-learning-theory/14-information-theory/README.md), Section 7); and the KL penalty $\beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ of RLHF ([Module 19](../../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md), Section 9) is the Lagrangian term of a KL *constraint*, such as TRPO's, with the multiplier fixed at $\beta$.

*Penalty and projection, compared.* The stationarity condition of $f + \rho\lVert h\rVert^2$ is $\nabla f + 2\rho\,h(x)\,\nabla h = 0$, the Lagrange condition with $\lambda = 2\rho\,h(x)$. Since the true multiplier $\lambda^\star$ is usually not zero, the penalty solution must violate the constraint by $h(x) \approx \lambda^\star/(2\rho)$, which vanishes only as $\rho \to \infty$ — the reason $\rho$ has to grow (worked below). Projection instead enforces feasibility exactly at every step and is the method of choice when $\Pi_C$ is cheap: clipping to a box, rescaling into a norm ball, normalising onto a sphere.

*In practice.* `scipy.optimize.minimize(method='trust-constr')` returns the multipliers in `res.v`; on the worked examples below, with the line passed as `LinearConstraint([[1, 2]], 5, 5)` and the wall $x \le 1$ as `LinearConstraint([[1]], -np.inf, 1)`, it reports $-2$ and $2$ (to solver tolerance), the $\lambda$ and $\mu$ of this guide. The older dictionary form `{'type': 'ineq', 'fun': lambda x: 1 - x[0]}`, which writes the wall as $1 - x \ge 0$, reports $-2$ for it instead, and other solvers and textbooks also use the opposite sign, so check the convention on a problem whose answer you know, then verify $\mu_j \ge 0$ and $\mu_j g_j = 0$.

**Worked example.** Minimise $\lVert x\rVert^2$ subject to $a^\top x = b$ with $a = (1, 2)$, $b = 5$. Stationarity: $2x + \lambda a = 0$, so $x = -\tfrac{\lambda}{2} a$. Feasibility: $a^\top x = -\tfrac{\lambda}{2}\lVert a\rVert^2 = b$, so $\lambda = -2b/\lVert a\rVert^2 = -2$ and $x^\star = b\,a/\lVert a\rVert^2 = (1, 2)$ with $f^\star = 5$. Check the shadow price: $f^\star(b) = b^2/\lVert a\rVert^2$, so $\mathrm{d}f^\star/\mathrm{d}b = 2b/\lVert a\rVert^2 = 2 = -\lambda$. The geometry: $x^\star$ is the point of the line closest to the origin, and $\nabla f = 2x^\star = (2, 4)$ is parallel to $\nabla h = a = (1, 2)$ (Figure 7(a)). For KKT, minimise $(x - 2)^2$ subject to $x \le 1$: the unconstrained minimiser $x = 2$ is infeasible, so the constraint is active, $x^\star = 1$, and stationarity $2(x - 2) + \mu = 0$ gives $\mu = 2 \ge 0$. Change the constraint to $x \le 3$: now $x = 2$ is feasible, $g(2) = -1 < 0$, and complementary slackness forces $\mu = 0$ (Figure 7(b)).

The dual of the first problem: minimising $\mathcal{L}(x, \lambda) = \lVert x\rVert^2 + \lambda(a^\top x - b)$ over $x$ gives $x = -\tfrac\lambda2 a$ and the dual function $d(\lambda) = -\tfrac{\lambda^2}{4}\lVert a\rVert^2 - \lambda b = -\tfrac54\lambda^2 - 5\lambda$. Every $\lambda$ gives a lower bound on $f^\star = 5$ — $d(0) = 0$, $d(-1) = 3.75$ — and the best, where $d'(\lambda) = -\tfrac52\lambda - 5 = 0$, is $\lambda = -2$ with $d(-2) = 5 = f^\star$: strong duality, by hand. The penalty method on the same problem, minimising $\lVert x\rVert^2 + \rho(a^\top x - 5)^2$, gives $x = \tfrac{5\rho}{1 + 5\rho}\,a$ and the violation $a^\top x - 5 = -5/(1 + 5\rho)$: $-0.83$ for $\rho = 1$, $-0.098$ for $\rho = 10$, $-0.010$ for $\rho = 100$, close to $\lambda^\star/(2\rho) = -1/\rho$, while $2\rho\,h(x) = -1.67,\ -1.96,\ -2.00$ approaches $\lambda^\star = -2$.

![Three panels. Left: circles of constant squared length around the origin and the line x1 plus 2 x2 equals 5, which the circle of value 5 touches at the point (1, 2); there the negative gradient points to the origin and the constraint gradient points the opposite way along the same line. Middle: the parabola (x minus 2) squared with a solid wall at x = 1, against which the constrained minimum is pushed, and a dashed wall at x = 3 that does not bind. Right: contours of portfolio variance over two of the weights, the shaded long-only triangle, the unconstrained optimum outside it in the shorting region and the constrained optimum on its edge w1 = 0, with an arrow pushing into the wall.](figures/lagrange-kkt.svg)

**Figure 7.** (a) The worked example: circles of constant $f = \lVert x\rVert^2$ ($f = 1$, $5$, $10$) and the constraint line $x_1 + 2x_2 = 5$. The circle $f = 5$ touches the line at $x^\star = (1, 2)$ (star). There $-\nabla f = (-2, -4)$, drawn at half length, ends exactly at the unconstrained minimum $0$, and $\nabla h = (1, 2)$ points the opposite way along the same line: $\nabla f + \lambda\nabla h = 0$ with $\lambda = -2$. At the non-optimal point $(3, 1)$ of the line, where $f = 10$, $-\nabla f = (-6, -2)$ (quarter length) still has a component along the line towards $x^\star$ (the short arrow lying on the line), so sliding along the fence lowers $f$. (b) KKT in one dimension, $f(x) = (x - 2)^2$. The wall $x \le 1$ (solid) is active: $x^\star = 1$, and the negative gradient pushes into the wall with $\mu = -f'(1) = 2 > 0$. The wall $x \le 3$ (dashed) is slack: $x^\star = 2$ (open circle), $g(2) = -1 < 0$, $\mu = 0$. (c) Case study 3 in the plane of $(w_1, w_2)$, with $w_3 = 1 - w_1 - w_2$: contours of the portfolio variance, the long-only triangle $w_i \ge 0$, and the shorting region $w_1 < 0$ (shaded). The unconstrained optimum $w = (-0.47, 1.31, 0.16)$, volatility $13.2\%$ (dot), lies outside the triangle; the long-only optimum $w = (0, 0.83, 0.17)$, volatility $13.9\%$ (square), sits on the wall $w_1 = 0$. The arrow is $-\partial V/\partial w_1 = -\mu_1 = -0.0080$: the negative gradient pushing against the wall, its size the multiplier (the arrow shows the direction only and is not to scale).

**Real-life example (sport).** A coach splits $30$ weekly hours between two athletes whose expected gains have diminishing returns, $4\sqrt{h_1} + 2\sqrt{h_2}$. Maximising subject to $h_1 + h_2 = 30$ means minimising $-f$, so in the convention of the Statement stationarity reads $-\nabla f + \lambda\nabla h = 0$, i.e. $\nabla f = \lambda\nabla h$: $2/\sqrt{h_1} = 1/\sqrt{h_2} = \lambda$, so $h_1 = 4 h_2$, hence $(h_1, h_2) = (24, 6)$, total gain $24.5$, and $\lambda = 2/\sqrt{24} = 0.41$. The shadow price $-\lambda$ of the minimisation says that the optimal $-f$ falls by $0.41$ per extra hour: one extra hour in the week is worth $0.41$ units of gain, whoever gets it. Note that the stronger athlete gets four times the hours, not all of them — concavity is what prevents the corner solution.

**In ML/AI this is …** the hard-margin SVM, $\min \tfrac12\lVert w\rVert^2$ s.t. $y_i(w^\top x_i + b) \ge 1$, whose multipliers $\alpha_i$ are non-zero only for the support vectors (complementary slackness) and whose dual is what `sklearn.svm.SVC` solves; the maximum-entropy distribution under moment constraints, whose solution $p_i \propto \exp(-\lambda^\top \phi(x_i))$ is the softmax/Gibbs form (Module 14); the KL penalty in RLHF and PPO as the Lagrangian relaxation of TRPO's KL *constraint*; and projected gradient steps that keep weights non-negative (NMF) or inside a norm ball.

## 8. Practical diagnostics

**Intuition.** You cannot see the loss landscape, but you can see three time series: training loss, validation loss and gradient norm. Reading them is like reading a patient's vital signs — each pathology has a signature, and each signature has a treatment.

**Signals and fixes.**

| What you see | Likely cause | Fix |
|---|---|---|
| loss rises, oscillates, or becomes `nan` within a few steps | $\eta > 2/L$ in some direction | lower $\eta$, add warm-up, clip gradients |
| loss decreases slowly and almost linearly | $\eta$ far below $2/L$, or $\kappa$ huge | raise $\eta$, standardise inputs, momentum/Adam |
| loss stalls at a shelf, gradient norm tiny | saddle or flat region | momentum, Adam, better initialisation |
| loss plateaus at a noisy floor | SGD noise ball, $\eta$ constant | decay $\eta$, larger batch |
| train loss falls, validation loss rises | overfitting | early stopping, regularisation, more data |
| sudden isolated spikes in loss and $\lVert g\rVert$ | rare large gradients (long sequences, outliers) | gradient clipping, smaller $\beta_2$ |

Three tools make these decisions quantitative. The *learning-rate range test* increases $\eta$ geometrically over a few hundred steps, from $10^{-6}$ to $1$, while logging the loss; the loss falls fastest just below the $\eta$ at which it starts climbing, and that value (or a few times less) is your starting point. *Gradient clipping by global norm* rescales $g \leftarrow g\cdot\min\big(1, c/\lVert g\rVert\big)$, which leaves direction unchanged and, for a plain SGD step, bounds the step to $\eta c$; for Adam, whose step is about $\eta$ per coordinate whatever the gradient's size (Section 5), it instead keeps a single spike from swamping the moment estimates $m_t$ and $v_t$. *Early stopping with patience $p$* stops when the validation loss has not improved for $p$ consecutive evaluations and restores the best checkpoint; it is both a stopping rule and a regulariser (Module 13).

**Worked example.** Clipping $g = (3, 4)$ at $c = 1$: $\lVert g\rVert = 5$, factor $\min(1, 1/5) = 0.2$, clipped gradient $(0.6, 0.8)$, norm $1$. Early stopping with patience 2 on validation losses $0.52, 0.41, 0.36, 0.35, 0.36, 0.37$: epoch 5 is no improvement (1), epoch 6 is no improvement (2) — stop, restore epoch 4 (Figure 8(c)). Divergence check: on the quadratic of Section 3 with $L = 10$, a range test that runs each $\eta$ afresh for 20 steps shows the loss climbing at $\eta = 0.2$ (Figure 8(b)); the recommended starting value $\eta \approx 0.1$–$0.18$ is exactly what the theory prescribes. The three runs of the Section 3 table, drawn as loss curves in Figure 8(a), show the signature of a step just above $2/L$: at $\eta = 0.21$ the loss falls from $55$ to $17.7$ in five steps, because the flat coordinate still shrinks, and then climbs, reaching about $1\,500$ at step 30, because the stiff coordinate grows by $1.1$ per step.

![Three panels. Left: loss on a log scale against the step for three step sizes; two fall steadily, the third falls for five steps and then climbs. Middle: the loss after 20 steps against the step size, both on log axes, falling to its lowest value near 0.18 and shooting up past 0.2, with a shaded recommended band from 0.1 to 0.18. Right: six validation losses by epoch, with the best at epoch 4 and the stopping point after two epochs without improvement.](figures/diagnostics.svg)

**Figure 8.** (a) The three runs of the Section 3 worked example as loss curves, $L(\theta_k)$ on a log scale from $L(\theta_0) = 55$. $\eta = 0.10$ (circles) falls along an almost straight line, reaching $0.09$ at step 30; $\eta = 0.19$ (squares) drops faster at first and then runs parallel to it, ten times lower, ending at $0.009$ (the same rate $0.9$, Figure 2(b)); $\eta = 0.21$ (triangles, dashed) is lowest at step 5 ($17.7$) and then climbs to $1.5\times10^{3}$. A curve that falls and then turns up is the signature of $\eta$ just above $2/L$. (b) A range test on the same quadratic: each $\eta$ is run fresh for 20 steps, and $L(\theta_{20})$ is plotted against $\eta$. The loss falls from $34$ at $\eta = 0.01$ to $0.74$ at $0.1$, is lowest, $0.018$, at $\eta^\star = 2/11 = 0.182$ (star), and shoots up past $2/L = 0.2$ (dotted): $5.0$ at $0.2$, $226$ at $0.21$. The shaded band $0.1$–$0.18$ is the recommended start. This is the fresh-start variant of the test. The single-run geometric sweep described above turns up only near $\eta \approx 0.76$, far beyond $2/L$: by the time $\eta$ passes $0.2$ the stiff coordinate has been shrunk almost to zero and needs several unstable steps to grow back (the lesson shows how mini-batch noise makes the knee appear near $2/L$). (c) The early-stopping example: validation losses $0.52, 0.41, 0.36, 0.35, 0.36, 0.37$. The best epoch is 4 (star); epochs 5 and 6 bring no improvement, so with patience 2 training stops after epoch 6 and restores epoch 4. The clipping example, $(3, 4) \to (0.6, 0.8)$, is not drawn.

**Real-life example (NLP).** You fine-tune a sentiment classifier on product reviews. Training is smooth until step 1 200, when the loss jumps from $0.3$ to $2.1$ and the model never recovers. The gradient-norm log shows a spike of $47$ at that step against a typical $0.8$: a batch of unusually long, contradictory reviews produced a huge gradient and Adam's running averages were pushed out of the good basin. Clipping the global norm at $1.0$ and adding a 500-step warm-up removes the spike — the same two settings that every transformer training recipe ships with.

**In ML/AI this is …** `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, `ReduceLROnPlateau`, the `EarlyStopping` callback in Keras, `early_stopping=True` in scikit-learn's `MLPClassifier` and `SGDClassifier`, and the gradient-norm panel in TensorBoard or Weights & Biases that experienced practitioners open before the loss panel.

### A checklist before you blame the model

1. Standardise inputs (and targets for regression). Check the Hessian's condition number on a subsample if the model is linear.
2. Run a learning-rate range test; start at a value a few times below the knee.
3. Use warm-up for adaptive methods and for any model with attention or recurrence.
4. Log the gradient norm; clip at a value near its typical magnitude.
5. Decay the learning rate (cosine or step) once the loss sits on a noisy floor.
6. Monitor validation loss; stop with patience and keep the best checkpoint.
7. If the loss is `nan`: lower $\eta$ by $10\times$ first, then look for numerical issues (log of zero, division by $\sqrt{\hat v}$ with $\epsilon = 0$).

## Common pitfalls and misconceptions

1. **"A smaller learning rate is always safer."** Below $2/L$ you cannot diverge, but with $\eta \ll 1/L$ the flat directions take $\sim 1/(\eta\mu)$ steps and you may stop at a point that merely *looks* converged. The fix is to pick $\eta$ from the curvature (range test, $\approx 1/L$), not from fear.
2. **"The gradient is zero, so we are at a minimum."** Zero gradient is necessary, not sufficient; saddles and maxima are also stationary. Check the Hessian's smallest eigenvalue (or the loss along a few random directions), and remember that Newton converges to *any* stationary point.
3. **"Adam does not need a learning-rate schedule or warm-up."** Adam normalises the step but its early estimates of $\hat v$ are based on very few gradients, and its steps are all $\approx\eta$ in size; large-model training uses warm-up plus cosine decay with Adam precisely because of this.
4. **"Mini-batch gradients are biased because they ignore most of the data."** They are exactly unbiased: $\mathbb{E}[g_B] = \nabla L$. What a small batch adds is *variance*, which the step size and schedule control.
5. **"L2 regularisation and weight decay are the same thing."** They coincide for plain SGD and differ for Adam, where the L2 gradient is divided by $\sqrt{\hat v}$. AdamW's decoupled decay is the version that behaves as intended.
6. **"Feature scaling is just a convenience."** For GD it changes the condition number and hence the step count — from $10^{11}$ to $1$ in the climate example. For Newton-type methods and closed-form least squares, scaling is indeed a convenience, which tells you that the issue belongs to the optimiser, not to the model.
7. **"A Lagrange multiplier of zero means the constraint is irrelevant."** It means the constraint is not *active at the optimum* — it may still be what keeps a different, infeasible region out of play, and it may become active when the data or the objective change slightly.

## Real-world case studies

### Case study 1 — Logistic regression on the breast-cancer data: scaling, GD and Newton

The scikit-learn breast-cancer set has $n = 569$ tumours with $d = 30$ measured features and a benign/malignant label. The features live on wildly different scales: "mean area" has standard deviation $352$, "mean smoothness" $0.014$. The logistic loss is convex with Hessian $\tfrac1n X^\top S X$, $S_{ii} = p_i(1-p_i)$. At $\theta = 0$ (all $p_i = 0.5$) the Hessian of the raw problem has largest eigenvalue $L \approx 4.2\times10^{5}$ and smallest $\approx 1.8\times10^{-7}$: $\kappa \approx 2\times10^{12}$ and the largest stable step is $2/L \approx 4.8\times10^{-6}$. Running GD at $\eta = 1/L$ for 2 000 steps reaches loss $0.304$ and accuracy $90.9\%$ — still far from the optimum. After standardising every feature the Hessian has $L \approx 3.3$ and $2/L \approx 0.60$. From here on the loss also carries a small ridge term $\tfrac{\lambda}{2}\lVert w\rVert^2$ with $\lambda = 10^{-2}$ on the non-intercept weights, which keeps the Hessian safely positive definite (the raw run above gives the same loss and accuracy with or without it). GD with $\eta = 0.3$ then reaches a loss, ridge term included, of $0.0996$ and accuracy $98.6\%$ in the same 2 000 steps (already $0.1003$ after 200); Figure 3(c) shows both runs. The data are identical; only $\kappa$ changed. Newton's method on the same standardised, ridge-regularised problem takes the gradient norm from $1.4$ to $4\times10^{-1}$, $1.5\times10^{-1}$, $5.8\times10^{-2}$, $1.8\times10^{-2}$, $2.8\times10^{-3}$, $9.6\times10^{-5}$, $1.1\times10^{-7}$, $1.5\times10^{-13}$ in eight iterations — the digits-doubling signature of quadratic convergence (Figure 6(b)) — at a cost of one $31\times31$ solve per step. This is what `LogisticRegression` does internally via L-BFGS, and it is why that solver converges in a few dozen iterations while a naive GD loop needs thousands.

### Case study 2 — The training budget of a large language model

Published training details for GPT-3 (175 billion parameters) read like a worked example of Sections 4 and 5. Batches of $3.2$ million tokens; linear warm-up over the first $375$ million tokens, i.e. about $117$ optimiser steps; cosine decay to $10\%$ of the peak learning rate ($6\times10^{-5}$) over $260$ billion tokens, about $81\,000$ steps; Adam with $\beta_1 = 0.9$, $\beta_2 = 0.95$ (a shorter memory for the second moment than the default $0.999$, for stability against rare large gradients); global gradient-norm clipping at $1.0$; decoupled weight decay $0.1$. The optimiser state is two float32 moments per parameter: $175\times10^9\times 8\ \text{bytes} \approx 1.4$ TB, more than the weights themselves, which forces the state to be sharded across hundreds of accelerators. Every one of these choices maps to a concept here: warm-up guards Adam's early, poorly estimated $\hat v_t$; cosine decay shrinks the noise ball; $\beta_2 = 0.95$ lets $v_t$ keep pace with $m_t$ after a sudden change of gradient scale, so a spike after a quiet stretch moves a weight by about $0.45\,\eta$ rather than $3.16\,\eta$ (Section 5); clipping caps the gradient that enters Adam's moment estimates, so a single spike cannot swamp $m_t$ and $v_t$ for the steps that follow (for plain SGD it would bound the step itself to $\eta c$). The learning rate itself was chosen smaller for larger models ($6\times10^{-4}$ for the 125-million-parameter version, $6\times10^{-5}$ for the largest), consistent with the larger curvature (larger $L$) that wider and deeper networks exhibit; the stable range $\eta < 2/L$ shrinks as $L$ grows. The batch size was *increased* with model size, which by the $\Sigma/b$ rule lowers gradient variance and makes a given learning rate safer. Since one evaluation of the full objective would need the entire corpus of roughly $500$ billion tokens, every one of the roughly $94\,000$ updates ($300$ billion training tokens at $3.2$ million per batch) was a stochastic step: the whole run is SGD, in the Adam flavour, over about $0.6$ epochs of the corpus in aggregate — the large web-crawl portion for under half an epoch, while small, up-weighted sources such as Wikipedia were seen up to $3.4$ times.

### Case study 3 — A minimum-variance portfolio with a long-only constraint

Three assets have volatilities $20\%$, $15\%$ and $30\%$; the first two are strongly correlated ($0.9$), the third is nearly independent of both ($0.1$). The covariance matrix is $\Sigma = \begin{pmatrix} 0.04 & 0.027 & 0.006 \\ 0.027 & 0.0225 & 0.0045 \\ 0.006 & 0.0045 & 0.09 \end{pmatrix}$. Minimising the variance $w^\top\Sigma w$ subject to $\mathbf 1^\top w = 1$ is an equality-constrained quadratic: stationarity gives $2\Sigma w = -\lambda\mathbf 1$, so $w \propto \Sigma^{-1}\mathbf 1$, normalised to sum to one: $w = (-0.47, 1.31, 0.16)$, with volatility $13.2\%$ — lower than the safest single asset, because the third asset diversifies, and achieved partly by *shorting* the first asset against the second. Add the inequality constraints $w_i \ge 0$ (no shorting). KKT says: try the active set in which $w_1 = 0$ and the other two are free — the natural guess, since the unconstrained optimum lies beyond that wall (Figure 7(c)). Solving the two-asset problem gives $w = (0, 0.826, 0.174)$ with volatility $13.9\%$. Checking the conditions: stationarity on the free assets gives $\lambda = -2(\Sigma w)_2 = -0.0387$, and the multiplier of the active constraint is $\mu_1 = 2(\Sigma w)_1 + \lambda = 0.0080 > 0$ (dual feasibility holds), while $\mu_2 = \mu_3 = 0$ because those constraints are slack (complementary slackness). All four conditions hold, and since the problem is convex this is the global optimum. The multiplier has a meaning: relaxing the rule to allow $w_1 \ge -0.01$ would lower the variance by about $0.008 \times 0.01 = 8\times10^{-5}$, from $0.01937$ to $0.01929$ — the price of the no-shorting rule, in variance units.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Empirical risk minimisation | every `fit`, `loss.backward(); optimizer.step()` | defines what "training" means |
| Convexity | linear/logistic regression, ridge, lasso, SVM | unique optimum; solver choice does not change the answer |
| Non-convexity, saddles | deep networks | seeds matter; plateaus are saddles, not convergence |
| $\eta < 2/L$, rate $(\kappa-1)/(\kappa+1)$ | learning-rate choice, `StandardScaler`, batch/layer norm | explains divergence and slow training |
| Unbiased mini-batch gradients | `DataLoader`, `SGDClassifier`, online recommenders | scales to data that does not fit in memory |
| LR schedules, warm-up | `CosineAnnealingLR`, `OneCycleLR`, transformer recipes | shrinks the noise ball; protects early steps |
| Noise ball $\propto \eta/b$ | large-batch training, the linear scaling rule $\eta \propto b$ | why the learning rate is raised with the batch size |
| Momentum / Nesterov | `SGD(momentum=0.9)` for CNNs | $\sqrt\kappa$ instead of $\kappa$ steps |
| Adam / AdamW | `AdamW` for transformers, diffusion models | per-parameter scale invariance; decoupled decay |
| Exponential moving averages, bias correction | Adam's $m_t$ and $v_t$, batch-norm running statistics, weight averaging | memory $\approx 1/(1-\beta)$ steps; averages started at zero need correcting |
| Newton, L-BFGS | `LogisticRegression`, IRLS for GLMs, `scipy.optimize` | fast, accurate for small–medium $d$ |
| Gauss–Newton / LM | `curve_fit`, bundle adjustment in computer vision | nonlinear least squares |
| Lagrange / KKT | SVM dual, max-entropy ⇒ softmax, TRPO/PPO KL constraint | structure of the solution; support vectors |
| Projection / penalty | NMF, weight constraints, RLHF KL penalty | enforce constraints inside gradient loops |
| Clipping, early stopping | `clip_grad_norm_`, `EarlyStopping` | stability; regularisation |

## Cheat sheet

| Item | Formula / fact |
|---|---|
| Empirical risk | $L(\theta) = \frac1n \sum_i \ell(f_\theta(x_i), y_i) + \lambda R(\theta)$ |
| Convex function | $f(\alpha x + (1-\alpha) y) \le \alpha f(x) + (1-\alpha) f(y)$; for twice-differentiable $f$ equivalently $\nabla^2 f \succeq 0$ |
| Strong convexity / smoothness | $\mu I \preceq \nabla^2 f \preceq L I$, $\kappa = L/\mu$ |
| GD | $\theta_{k+1} = \theta_k - \eta\nabla L(\theta_k)$ |
| Stable step (quadratic) | $\eta < 2/L$; optimal $\eta^\star = 2/(L+\mu)$; rate $(\kappa-1)/(\kappa+1)$ |
| Steps to accuracy $\varepsilon$ | $\approx \kappa\ln(1/\varepsilon)$ with $\eta = 1/L$ |
| Mini-batch gradient | $\mathbb{E}[g_B] = \nabla L$; $\mathrm{Cov}(g_B) = \frac{\Sigma}{b}\cdot\frac{n-b}{n-1}$ without replacement, $\Sigma/b$ with replacement; $\approx \Sigma / b$ for $b \ll n$ |
| SGD noise ball (1-D) | $\mathrm{Var}(\theta_\infty) = \eta\sigma^2/(\lambda(2 - \eta\lambda)) \approx \eta\sigma^2/(2\lambda)$, $\sigma^2$ = batch-gradient variance |
| Step-size conditions | $\sum_t \eta_t = \infty$, $\sum_t \eta_t^2 < \infty$ (e.g. $\eta_t \propto 1/t$) |
| Cosine schedule | $\eta_t = \eta_{\min} + \frac12(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$ |
| Heavy ball | $v_t = \beta v_{t-1} + g_t$, $\theta_{t+1} = \theta_t - \eta v_t$; effective step $\eta/(1-\beta)$; rate $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ |
| Exponential moving average | $m_t = (1-\beta)\sum_{s \le t}\beta^{t-s} g_s$; weights sum to $1 - \beta^t$; memory $\approx 1/(1-\beta)$ steps |
| RMSProp | $s_t = \rho s_{t-1} + (1-\rho) g_t^2$, step $\eta g_t/(\sqrt{s_t}+\epsilon)$ |
| Adam | $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$, step $\eta \hat m_t/(\sqrt{\hat v_t}+\epsilon)$ |
| AdamW | add $\eta\lambda\theta_t$ to the step, outside the adaptive scaling |
| Newton | $\theta_{k+1} = \theta_k - H^{-1}\nabla L$; quadratic convergence; $O(d^3)$ per step |
| Gauss–Newton / LM | $(J^\top J + \tau I)\, p = -J^\top r$ |
| Lagrange (equality) | $\nabla f + \lambda\nabla h = 0$, $h = 0$; $-\lambda$ = shadow price |
| KKT | stationarity, $g_j \le 0$, $\mu_j \ge 0$, $\mu_j g_j = 0$ |
| Lagrange dual | $d(\lambda, \mu) = \min_x \mathcal{L}(x, \lambda, \mu) \le f^\star$ for $\mu \ge 0$; for convex problems $\max_{\lambda,\, \mu \ge 0} d(\lambda, \mu) = f^\star$ (under a mild regularity condition) |
| Penalty method | $f + \rho\lVert h\rVert^2$: Lagrange with $\lambda = 2\rho\,h(x)$; violation $\approx \lambda^\star/(2\rho)$ |
| Min $\lVert x\rVert^2$ s.t. $a^\top x = b$ | $x^\star = b\,a/\lVert a\rVert^2$ |
| Gradient clipping | $g \leftarrow g\cdot\min(1, c/\lVert g\rVert)$ |

## Self-assessment

1. **GD on a quadratic with Hessian eigenvalues 0.5 and 50 diverges at $\eta = 0.05$. Why, and what is the best constant step?**
<details><summary>Answer</summary>

$2/L = 2/50 = 0.04 < 0.05$, so the factor for the stiff direction is $1 - 0.05\cdot50 = -1.5$, whose powers grow. The best constant step is $2/(L+\mu) = 2/50.5 \approx 0.0396$, with rate $(\kappa-1)/(\kappa+1) = 99/101 \approx 0.98$: about 700 steps per factor $10^{-6}$. Standardising the features would be the real fix.

</details>

2. **Is $f(x, y) = x^2 - y^2$ convex? What kind of stationary point is the origin?**
<details><summary>Answer</summary>

No: the Hessian $\mathrm{diag}(2, -2)$ has a negative eigenvalue. The origin has zero gradient and an indefinite Hessian, so it is a saddle point; GD started exactly there stays forever, started anywhere off the $x$-axis it escapes along $y$.

</details>

3. **Why is the mini-batch gradient unbiased even though it uses only $b$ of $n$ examples? What does the batch size control?**
<details><summary>Answer</summary>

Each example is included with the same probability, so the expectation of the batch average equals the average over all examples, $\mathbb{E}[g_B] = \nabla L$. The batch size controls the variance, roughly $\Sigma/b$: larger batches give a smaller noise ball for the same step size, at proportionally higher cost per step.

</details>

4. **Explain in one sentence each why momentum and Adam help on a plateau.**
<details><summary>Answer</summary>

Momentum carries the velocity accumulated before the plateau across the flat region where the gradient alone would move nothing; Adam divides the tiny gradient by its own tiny running magnitude, so the step is still about $\eta$ in size.

</details>

5. **Without bias correction, what is the size of Adam's first step relative to $\eta$ (take $\beta_1 = 0.9$, $\beta_2 = 0.999$)?**
<details><summary>Answer</summary>

$m_1 = 0.1 g$, $v_1 = 0.001 g^2$, so $m_1/\sqrt{v_1} = 0.1/\sqrt{0.001} \approx 3.16$: the first step is more than three times $\eta$, in the direction of the sign of $g$. Bias correction divides by $0.1$ and $\sqrt{0.001}$ respectively and restores a step of exactly $\eta$.

</details>

6. **Why do Adam and L2 regularisation interact badly, and what does AdamW change?**
<details><summary>Answer</summary>

With L2 the regularisation gradient $\lambda\theta$ is added to $g_t$ and then divided by $\sqrt{\hat v_t}$, so parameters with historically large gradients are regularised much less than those with small gradients. AdamW applies the decay $\eta\lambda\theta_t$ directly to the parameters, outside the adaptive scaling, so every weight decays at the same relative rate.

</details>

7. **Newton's method converges in one step on a quadratic. Why is it nevertheless rarely used for neural networks?**
<details><summary>Answer</summary>

The Hessian has $d^2$ entries and the solve costs $O(d^3)$; for $d = 10^8$ neither storing nor inverting it is possible. Away from a minimum it may be indefinite and send the step uphill, and with mini-batch noise the curvature estimate is unreliable. Quasi-Newton (L-BFGS), Hessian-vector products and K-FAC-style approximations are the practical compromises.

</details>

8. **Minimise $x^2 + 2y^2$ subject to $x + y = 3$.**
<details><summary>Answer</summary>

Stationarity: $2x + \lambda = 0$, $4y + \lambda = 0$, so $x = 2y$. Feasibility: $3y = 3$, so $y = 1$, $x = 2$, $f^\star = 4 + 2 = 6$, and $\lambda = -4$. Geometrically the contour ellipse of $f$ through $(2, 1)$ is tangent to the line $x + y = 3$: $\nabla f = (4, 4)$ is parallel to $\nabla h = (1, 1)$.

</details>

9. **In an SVM, which training points have non-zero Lagrange multipliers, and which KKT condition says so?**
<details><summary>Answer</summary>

Complementary slackness $\alpha_i\big(y_i(w^\top x_i + b) - 1\big) = 0$: a point strictly outside the margin has slack $> 0$, hence $\alpha_i = 0$; only points on the margin (and, in the soft-margin case, inside it) can have $\alpha_i > 0$. Those are the support vectors, and the weight vector $w = \sum_i \alpha_i y_i x_i$ depends only on them.

</details>

10. **Your loss goes to `nan` at step 3 with Adam at $\eta = 10^{-3}$. List the first three things you check.**
<details><summary>Answer</summary>

(i) Input scaling and the loss implementation (log of zero, overflow in `exp`); (ii) the gradient norm at steps 1–3 — if it is huge, clip and add warm-up; (iii) the learning rate — drop it by $10\times$ and run a range test. Only then suspect the architecture.

</details>

11. **You halve the batch size from 256 to 128 and keep the learning rate. What happens to the SGD noise ball, and which single change restores it?**
<details><summary>Answer</summary>

The batch-gradient variance is about $\Sigma/b$ (the finite-population factor is close to 1 when $b \ll n$), so it doubles. The noise-ball variance $\eta\sigma^2/\big(\lambda(2 - \eta\lambda)\big) \approx \eta\sigma^2/(2\lambda)$ doubles with it, and so does the excess-loss floor $\approx \eta\sigma^2/4$. Halving $\eta$ restores the ball (exactly in the small-$\eta$ approximation) — the linear scaling rule $\eta \propto b$ read backwards. The price is slower progress per step, partly made up for by each step costing half as much.

</details>

12. **Write the dual function of Question 8's problem, minimise $x^2 + 2y^2$ subject to $x + y = 3$, and check strong duality.**
<details><summary>Answer</summary>

$\mathcal{L} = x^2 + 2y^2 + \lambda(x + y - 3)$. Minimising over $x$ and $y$ gives $2x + \lambda = 0$ and $4y + \lambda = 0$, so $x = -\lambda/2$, $y = -\lambda/4$ and $d(\lambda) = \tfrac{\lambda^2}{4} + \tfrac{\lambda^2}{8} + \lambda\big(-\tfrac{3\lambda}{4} - 3\big) = -\tfrac38\lambda^2 - 3\lambda$. Every $\lambda$ gives a lower bound on $f^\star = 6$, e.g. $d(0) = 0$ and $d(-2) = 4.5$. The best, where $d'(\lambda) = -\tfrac34\lambda - 3 = 0$, is $\lambda = -4$ — the multiplier of Question 8 — with $d(-4) = -6 + 12 = 6 = f^\star$.

</details>

13. **With $\beta_1 = 0.9$ and $\beta_2 = 0.95$, roughly how many past steps do Adam's $m_t$ and $v_t$ remember? After a long stretch of near-zero gradients a single huge gradient $G$ arrives: how large is the step in that coordinate, and does clipping the gradient change it?**
<details><summary>Answer</summary>

About $1/(1 - \beta)$ steps: 10 for $m_t$, 20 for $v_t$. After the quiet stretch $m_t \approx 0.1\,G$ and $v_t \approx 0.05\,G^2$, so the step is $\eta \cdot 0.1\,G/\sqrt{0.05\,G^2} = 0.1/\sqrt{0.05}\;\eta \approx 0.45\,\eta$ whatever $G$ is ($3.16\,\eta$ with $\beta_2 = 0.999$). Clipping rescales $G$ but leaves this ratio unchanged. What it does change is how far the spike inflates $m_t$ and $v_t$, and hence how much it distorts the steps that follow.

</details>

## Further reading

- S. Boyd and L. Vandenberghe, *Convex Optimization* (Cambridge University Press, 2004) — free PDF from the authors; chapters 2–5 cover convex sets, functions, duality and KKT.
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed. (Springer, 2006) — line search, Newton, quasi-Newton and L-BFGS, nonlinear least squares, constrained methods.
- I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning* (MIT Press, 2016), chapter 8 "Optimization for Training Deep Models" — free online.
- L. Bottou, F. E. Curtis and J. Nocedal, "Optimization Methods for Large-Scale Machine Learning", *SIAM Review* 60(2), 2018 — the authoritative survey of SGD theory and practice.
- D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
- I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 — AdamW.
- S. J. Reddi, S. Kale and S. Kumar, "On the Convergence of Adam and Beyond", ICLR 2018 — the Adam counter-example and AMSGrad.
- I. Sutskever, J. Martens, G. Dahl and G. Hinton, "On the importance of initialization and momentum in deep learning", ICML 2013.
- G. Goh, "Why Momentum Really Works", *Distill*, 2017 — interactive derivation of the $\sqrt\kappa$ rate.
- L. N. Smith, "Cyclical Learning Rates for Training Neural Networks", WACV 2017 — the learning-rate range test.

## What's in the notebooks

**`lesson.ipynb`** builds everything in NumPy: the 1-D loss landscape of Section 1 with its derivative, and gradient descent sliding into, and escaping from, the saddle of $x^2 - y^2$, after about $\ln(1/\lvert y_0\rvert)/\ln 1.2$ steps; a numerical convexity test via Hessian eigenvalues, the chord test failing for $x^4 - x^2$, and the loss surface of a one-neuron tanh network with its two mirror-image minima; contour plots of an ill-conditioned quadratic with GD paths for several step sizes, including one that diverges; an experiment measuring GD's convergence rate against the condition number and checking it against $(\kappa-1)/(\kappa+1)$; clouds of mini-batch gradients around the full gradient, whose total variance falls like $1/b$, and GD versus mini-batch SGD on a linear-regression problem, with the noise floor, decaying schedules and a check of the noise-ball formula; momentum, RMSProp and Adam implemented from their update equations and compared on the Rosenbrock function and on logistic regression for the standardised breast-cancer data, together with Adam's step size with and without bias correction and the heavy ball against GD on a $\kappa = 100$ bowl, where its velocity piles up along the valley; Newton versus GD on that same logistic regression; a visualisation of the Lagrange "tangent level set" condition; and the practical diagnostics: learning-rate schedules side by side, a mini-batch learning-rate range test, gradient clipping turning a divergent run into a bounded bounce, and early stopping on a training set of 120 tumours.

**`exercises.ipynb`** (with `solutions.ipynb`) tests the same ideas: the largest stable step on a given quadratic, mini-batch SGD for linear regression, a convexity check via Hessian eigenvalues, cosine and warm-up schedules, Adam from the equations matched against a reference run, Newton's method for logistic regression, a Lagrange problem solved by hand, gradient clipping and a learning-rate sweep, heavy-ball momentum against GD on a $\kappa = 100$ quadratic, two conceptual questions on momentum and on KKT, and a mini-project that trains logistic regression with GD, momentum and Adam on a synthetic dataset and studies the sensitivity to the learning rate.

Next: [Module 06 — Automatic differentiation and backpropagation](../06-autodiff-and-backpropagation/README.md), where the gradients this module consumes are produced automatically.
