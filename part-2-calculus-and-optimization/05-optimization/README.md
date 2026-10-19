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
- Explain why mini-batch gradients are unbiased, how their variance scales with batch size, and what a learning-rate schedule does to the noise floor.
- Write the update equations of heavy-ball momentum, Nesterov, RMSProp, Adam and AdamW from memory and implement them in NumPy.
- Derive Newton's method from a second-order Taylor expansion, state its quadratic convergence and its $O(d^3)$ cost, and describe the L-BFGS and Gauss–Newton work-arounds.
- Solve small equality-constrained problems with Lagrange multipliers and read the KKT conditions, including complementary slackness, for inequality constraints.
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

Applied to a weighted average of many points this becomes *Jensen's inequality*, $f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$. Two equivalent tests for differentiable $f$:

- *First-order*: $f(y) \ge f(x) + \nabla f(x)^\top (y - x)$ for all $x, y$ — the tangent plane lies below the graph.
- *Second-order*: the Hessian is positive semi-definite everywhere, $\nabla^2 f(x) \succeq 0$ (all eigenvalues $\ge 0$).

$f$ is *$\mu$-strongly convex* if $\nabla^2 f \succeq \mu I$ with $\mu > 0$ (curvature at least $\mu$ in every direction), and *$L$-smooth* if $\nabla^2 f \preceq L I$ (curvature at most $L$). The ratio $\kappa = L/\mu$ is the condition number of the problem.

*Why convex implies "local = global"*: let $x^\star$ be a stationary point, $\nabla f(x^\star) = 0$. The first-order condition gives $f(y) \ge f(x^\star) + 0^\top(y - x^\star) = f(x^\star)$ for every $y$. So any stationary point is a global minimum, and with strict convexity it is unique.

**Worked example.** Jensen on $f(x) = x^2$ with $x = 1$, $y = 3$, $\alpha = \tfrac12$: $f(2) = 4 \le \tfrac12(1 + 9) = 5$. Hessian test on $f(x, y) = x^2 + xy + y^2$: $\nabla^2 f = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$ with eigenvalues $1$ and $3$, so $f$ is $1$-strongly convex and $3$-smooth, $\kappa = 3$. Non-example: $f(x) = x^4 - x^2$ has $f''(0) = -2 < 0$; it has two minima at $x = \pm 1/\sqrt 2$ and a local maximum between them. The chord from $-1/\sqrt2$ to $1/\sqrt2$ passes *below* the graph at $x = 0$. For ML losses: the MSE of a linear model has Hessian $\tfrac{2}{n} X^\top X \succeq 0$, and the logistic loss has Hessian $\tfrac1n X^\top S X$ with $S = \mathrm{diag}\big(p_i(1 - p_i)\big) \succeq 0$ — both convex. A neural network with one hidden layer is *not*: swapping two hidden units gives a different $\theta$ with identical loss, and the straight line between these two minima passes through worse parameters, which a convex function forbids.

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
| $0.21$ | $(0.79,\ -1.1)$ | $(7.89,\ -1.11)$ | $(6.23,\ 1.22)$ | $(4.92,\ -1.35)$ | $0.21 > 2/L = 0.2$: the stiff coordinate *grows* — divergence |

The optimum $\eta^\star = 2/11 = 0.182$ gives rate $9/11 = 0.818$, i.e. about 69 steps to $10^{-6}$ instead of 131. No constant step can do better, because any $\eta$ that is good for the flat direction is bad for the stiff one — this is the zig-zag picture on an elongated bowl.

**Real-life example (climate).** Fit a linear warming trend, temperature $\approx a + b\cdot\text{year}$, to 41 annual means for 1980–2020 by GD on the MSE. With the raw year as the feature, the Hessian $\tfrac{2}{n}X^\top X$ has eigenvalues $\approx 10^{-4}$ and $8\times 10^{6}$: $\kappa \approx 10^{11}$, so a stable step must be below $2.5\times10^{-7}$ and the intercept direction would take on the order of $10^{11}$ steps. Centring the year ($\text{year} - 2000$) brings $\kappa$ down to $140$; standardising it to unit variance gives $\kappa = 1$, and GD with $\eta = 1/L$ converges in a single step. Nothing about the data changed — only its parametrisation.

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

**Worked example.** Five points $x = (1, 2, 3, 4, 5)$, $y = (1.5, 3.9, 6.2, 8.1, 9.8)$, model $y \approx w x$, current $w = 1.5$. Per-example gradients $2 x_i (w x_i - y_i)$ are $(0, -3.6, -10.2, -16.8, -23.0)$, whose mean, $-10.72$, is the full gradient. Mini-batches of size 2: $\{1, 4\}$ gives $-8.4$, $\{2, 3\}$ gives $-6.9$, $\{3, 5\}$ gives $-16.6$. Every estimate has the right sign (increase $w$) but the magnitudes differ by a factor of two or more; averaged over all ten possible batches they give exactly $-10.72$. The per-example variance is $\sigma^2 = 70.7$; the variance of a size-2 batch mean is $\tfrac{\sigma^2}{b}\cdot\tfrac{n-b}{n-1} = \tfrac{70.7}{2}\cdot\tfrac{3}{4} = 26.5$, which you can confirm by averaging over all ten batches. Under a cosine schedule with $T$ total steps, the step is $85\%$ of $\eta_{\max}$ at $t = T/4$, $50\%$ at $T/2$ and $15\%$ at $3T/4$.

**Real-life example (e-commerce).** A recommender scores (user, item) pairs with a dot product of embeddings. Clicks arrive as a stream of millions of events per day; the "dataset" is never complete. The system applies one SGD step per event (or per small batch): update only the two embedding rows involved, cost independent of history size. Yesterday's log is one "epoch", the step size is decayed overnight, and new users start with a larger step so that their embedding moves quickly. Full-batch GD is not merely slower here — it is undefined, because the data never stops arriving.

**In ML/AI this is …** `DataLoader(batch_size=..., shuffle=True)` plus `CosineAnnealingLR` / `OneCycleLR` in PyTorch, `SGDClassifier` in scikit-learn, and the warm-up-then-decay schedule of the original Transformer. Module 15 returns to the observation that SGD's noise acts as an implicit regulariser that favours flat minima.

## 5. Momentum and adaptive methods

**Intuition.** A heavy ball rolling down the ill-conditioned bowl of Section 3 does not zig-zag: the sideways kicks alternate in sign and cancel, while the downhill pushes all point the same way and accumulate. That is momentum. Adaptive methods attack the same problem differently: give every parameter its own step size, dividing by a running estimate of that parameter's typical gradient magnitude, so stiff and flat directions receive comparable updates.

**Definition.** With $g_t = \nabla L(\theta_t)$ (or a mini-batch estimate):

- *Heavy-ball momentum* (Polyak): $v_t = \beta v_{t-1} + g_t$, $\theta_{t+1} = \theta_t - \eta v_t$. Under a constant gradient $v_t \to g/(1 - \beta)$, so the effective step is $\eta/(1-\beta)$: ten times $\eta$ for $\beta = 0.9$. On a quadratic with condition number $\kappa$, the tuned heavy ball ($\beta = \big(\tfrac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\big)^2$, $\eta = \big(\tfrac{2}{\sqrt L + \sqrt\mu}\big)^2$) converges at rate $\tfrac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$: the cost grows like $\sqrt\kappa$ instead of $\kappa$.
- *Nesterov momentum*: the same, but the gradient is evaluated at the look-ahead point $\theta_t - \eta\beta v_{t-1}$; it has the same $\sqrt\kappa$ rate with guarantees for general convex functions.
- *RMSProp*: $s_t = \rho s_{t-1} + (1 - \rho) g_t^2$ (element-wise), $\theta_{t+1} = \theta_t - \eta\, g_t / (\sqrt{s_t} + \epsilon)$.
- *Adam*: momentum on the gradient and on its square, both bias-corrected:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad \hat m_t = \frac{m_t}{1 - \beta_1^t},\quad \hat v_t = \frac{v_t}{1 - \beta_2^t},\quad \theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon},
$$

with defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.
- *AdamW*: $\theta_{t+1} = \theta_t - \eta\big(\hat m_t / (\sqrt{\hat v_t} + \epsilon) + \lambda \theta_t\big)$. Weight decay is applied *outside* the adaptive scaling. If you instead add $\lambda\theta$ to $g_t$ (L2 regularisation), the decay term gets divided by $\sqrt{\hat v_t}$ and parameters with large gradients are barely regularised.

Adam is not guaranteed to converge even on simple convex problems: with a short memory $\beta_2$, a rare large gradient can shrink the effective step just when it is needed and then let it grow again; AMSGrad (keep the running *maximum* of $v_t$) is the standard fix, and in practice $\beta_2 = 0.95$–$0.99$ is used for large models for related stability reasons.

**Worked example.** Momentum with $\beta = 0.9$ and constant gradient $g = 1$: $v = 1,\ 1.9,\ 2.71,\ 3.44,\ 4.10, \dots \to 10$. Adam's first step with $g_1 = 0.1$: $m_1 = 0.01$, $v_1 = 10^{-5}$. *Without* bias correction the update ratio is $m_1 / \sqrt{v_1} = 0.01 / 0.00316 = 3.16$, a step three times larger than $\eta$ in the first iteration. *With* correction, $\hat m_1 = 0.1$, $\hat v_1 = 0.01$, ratio $= 1.0$ exactly. The same calculation with $g_1 = 100$ also gives ratio $1.0$: Adam's early steps have size $\approx \eta$ in every coordinate *regardless of the gradient's scale*. That is the per-parameter normalisation at work, and it is why Adam's $\eta$ is typically 10–100 times smaller than SGD's.

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

The error is squared (times $\approx 0.35$) at every step. From $x_0 = -1$ the same iteration converges just as fast to $-\sqrt2$ — which is the local *maximum* of $f$, since $f''(-\sqrt2) < 0$. Newton solves $f' = 0$; it does not know the difference.

**Real-life example (physics/engineering).** Predicting a satellite's position requires solving Kepler's equation $E - e\sin E = M$ for the eccentric anomaly $E$, thousands of times per second in a GPS receiver. Newton on $F(E) = E - e\sin E - M$ with $F'(E) = 1 - e\cos E$, $e = 0.5$, $M = 1$, starting at $E_0 = 1$: $1 \to 1.5765 \to 1.50021 \to 1.4987017 \to 1.4987011335$, with $\lvert F\rvert = 9\times10^{-14}$ after four steps. Calibrating a physical model (reaction-rate constants, lens distortion, a battery model) against measurements is a nonlinear least-squares problem, and Levenberg–Marquardt is the workhorse.

**In ML/AI this is …** `LogisticRegression(solver='lbfgs')`, the scikit-learn default, and `scipy.optimize.minimize(method='L-BFGS-B')`; the IRLS algorithm that fits generalised linear models is exactly Newton's method; `scipy.optimize.curve_fit` is Levenberg–Marquardt. For deep networks the full Hessian is out of reach, but Hessian-vector products come free from autodiff (Module 06) and K-FAC and Shampoo are practical approximate second-order optimisers; the Fisher-information view of curvature returns in Module 11.

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

For convex $f, g_j$ and affine $h_i$ these conditions are necessary *and* sufficient (under a mild regularity condition). The *dual problem* $\max_{\lambda,\, \mu \ge 0} \min_x \mathcal{L}(x, \lambda, \mu)$ always gives a lower bound on the optimal value (weak duality) and equals it for such convex problems (strong duality). When KKT is awkward to solve, two practical alternatives are the *penalty method* — minimise $f(x) + \rho\lVert h(x)\rVert^2$ with growing $\rho$ — and *projected gradient descent* $x \leftarrow \Pi_C\big(x - \eta \nabla f(x)\big)$, where $\Pi_C$ snaps the iterate back onto the feasible set (clip to $[0, \infty)$, normalise to the unit sphere, …).

**Worked example.** Minimise $\lVert x\rVert^2$ subject to $a^\top x = b$ with $a = (1, 2)$, $b = 5$. Stationarity: $2x + \lambda a = 0$, so $x = -\tfrac{\lambda}{2} a$. Feasibility: $a^\top x = -\tfrac{\lambda}{2}\lVert a\rVert^2 = b$, so $\lambda = -2b/\lVert a\rVert^2 = -2$ and $x^\star = b\,a/\lVert a\rVert^2 = (1, 2)$ with $f^\star = 5$. Check the shadow price: $f^\star(b) = b^2/\lVert a\rVert^2$, so $\mathrm{d}f^\star/\mathrm{d}b = 2b/\lVert a\rVert^2 = 2 = -\lambda$. The geometry: $x^\star$ is the point of the line closest to the origin, and $\nabla f = 2x^\star = (2, 4)$ is parallel to $\nabla h = a = (1, 2)$. For KKT, minimise $(x - 2)^2$ subject to $x \le 1$: the unconstrained minimiser $x = 2$ is infeasible, so the constraint is active, $x^\star = 1$, and stationarity $2(x - 2) + \mu = 0$ gives $\mu = 2 \ge 0$. Change the constraint to $x \le 3$: now $x = 2$ is feasible, $g(2) = -1 < 0$, and complementary slackness forces $\mu = 0$.

**Real-life example (sport).** A coach splits $30$ weekly hours between two athletes whose expected gains have diminishing returns, $4\sqrt{h_1} + 2\sqrt{h_2}$. Maximising subject to $h_1 + h_2 = 30$: stationarity gives $2/\sqrt{h_1} = 1/\sqrt{h_2} = \lambda$, so $h_1 = 4 h_2$, hence $(h_1, h_2) = (24, 6)$, total gain $24.5$, and $\lambda = 2/\sqrt{24} = 0.41$: one extra hour in the week is worth $0.41$ units of gain, whoever gets it. Note that the stronger athlete gets four times the hours, not all of them — concavity is what prevents the corner solution.

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

Two tools make these decisions quantitative. The *learning-rate range test* increases $\eta$ geometrically over a few hundred steps, from $10^{-6}$ to $1$, while logging the loss; the loss falls fastest just below the $\eta$ at which it starts climbing, and that value (or a few times less) is your starting point. *Gradient clipping by global norm* rescales $g \leftarrow g\cdot\min\big(1, c/\lVert g\rVert\big)$, which leaves direction unchanged and bounds the step to $\eta c$. *Early stopping with patience $p$* stops when the validation loss has not improved for $p$ consecutive evaluations and restores the best checkpoint; it is both a stopping rule and a regulariser (Module 13).

**Worked example.** Clipping $g = (3, 4)$ at $c = 1$: $\lVert g\rVert = 5$, factor $\min(1, 1/5) = 0.2$, clipped gradient $(0.6, 0.8)$, norm $1$. Early stopping with patience 2 on validation losses $0.52, 0.41, 0.36, 0.35, 0.36, 0.37$: epoch 5 is no improvement (1), epoch 6 is no improvement (2) — stop, restore epoch 4. Divergence check: on the quadratic of Section 3 with $L = 10$, a range test shows the loss climbing at $\eta = 0.2$; the recommended starting value $\eta \approx 0.1$–$0.18$ is exactly what the theory prescribes.

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

The scikit-learn breast-cancer set has $n = 569$ tumours with $d = 30$ measured features and a benign/malignant label. The features live on wildly different scales: "mean area" has standard deviation $352$, "mean smoothness" $0.014$. The logistic loss is convex with Hessian $\tfrac1n X^\top S X$, $S_{ii} = p_i(1-p_i)$. At $\theta = 0$ (all $p_i = 0.5$) the Hessian of the raw problem has largest eigenvalue $L \approx 4.2\times10^{5}$ and smallest $\approx 1.8\times10^{-7}$: $\kappa \approx 2\times10^{12}$ and the largest stable step is $2/L \approx 4.8\times10^{-6}$. Running GD at $\eta = 1/L$ for 2 000 steps reaches loss $0.304$ and accuracy $90.9\%$ — still far from the optimum. After standardising every feature the Hessian has $L \approx 3.3$, $2/L \approx 0.60$, and GD with $\eta = 0.3$ reaches loss $0.0996$ and accuracy $98.6\%$ in the same 2 000 steps (already $0.1003$ after 200). The data are identical; only $\kappa$ changed. Newton's method on the standardised problem (with a small ridge term $\lambda = 10^{-2}$ so the Hessian is safely positive definite) takes the gradient norm from $1.4$ to $4\times10^{-1}$, $1.5\times10^{-1}$, $5.8\times10^{-2}$, $1.8\times10^{-2}$, $2.8\times10^{-3}$, $9.6\times10^{-5}$, $1.1\times10^{-7}$, $1.5\times10^{-13}$ in eight iterations — the digits-doubling signature of quadratic convergence — at a cost of one $31\times31$ solve per step. This is what `LogisticRegression` does internally via L-BFGS, and it is why that solver converges in a few dozen iterations while a naive GD loop needs thousands.

### Case study 2 — The training budget of a large language model

Published training details for GPT-3 (175 billion parameters) read like a worked example of Sections 4 and 5. Batches of $3.2$ million tokens; linear warm-up over the first $375$ million tokens, i.e. about $117$ optimiser steps; cosine decay to $10\%$ of the peak learning rate ($6\times10^{-5}$) over $260$ billion tokens, about $81\,000$ steps; Adam with $\beta_1 = 0.9$, $\beta_2 = 0.95$ (a shorter memory for the second moment than the default $0.999$, for stability against rare large gradients); global gradient-norm clipping at $1.0$; decoupled weight decay $0.1$. The optimiser state is two float32 moments per parameter: $175\times10^9\times 8\ \text{bytes} \approx 1.4$ TB, more than the weights themselves, which forces the state to be sharded across hundreds of accelerators. Every one of these choices maps to a concept here: warm-up guards Adam's early, poorly estimated $\hat v_t$; cosine decay shrinks the noise ball; $\beta_2 = 0.95$ addresses the Adam convergence issue; clipping bounds the step to $\eta c$ regardless of a spike's size. The learning rate itself was chosen smaller for larger models ($6\times10^{-4}$ for the 125-million-parameter version, $6\times10^{-5}$ for the largest), consistent with the larger curvature (larger $L$) that wider and deeper networks exhibit; the stable range $\eta < 2/L$ shrinks as $L$ grows. The batch size was *increased* with model size, which by the $\Sigma/b$ rule lowers gradient variance and makes a given learning rate safer. Since one evaluation of the full objective would need the entire $300$-billion-token corpus, every one of the roughly $93\,000$ updates was a stochastic step: the whole training run is a single pass of SGD, in the Adam flavour, with fewer than one epoch over the data.

### Case study 3 — A minimum-variance portfolio with a long-only constraint

Three assets have volatilities $20\%$, $15\%$ and $30\%$; the first two are strongly correlated ($0.9$), the third is nearly independent of both ($0.1$). The covariance matrix is $\Sigma = \begin{pmatrix} 0.04 & 0.027 & 0.006 \\ 0.027 & 0.0225 & 0.0045 \\ 0.006 & 0.0045 & 0.09 \end{pmatrix}$. Minimising the variance $w^\top\Sigma w$ subject to $\mathbf 1^\top w = 1$ is an equality-constrained quadratic: stationarity gives $2\Sigma w = -\lambda\mathbf 1$, so $w \propto \Sigma^{-1}\mathbf 1$, normalised to sum to one: $w = (-0.47, 1.31, 0.16)$, with volatility $13.2\%$ — lower than the safest single asset, because the third asset diversifies, and achieved partly by *shorting* the first asset against the second. Add the inequality constraints $w_i \ge 0$ (no shorting). KKT says: try the active set in which $w_1 = 0$ and the other two are free. Solving the two-asset problem gives $w = (0, 0.826, 0.174)$ with volatility $13.9\%$. Checking the conditions: stationarity on the free assets gives $\lambda = -2(\Sigma w)_2 = -0.0387$, and the multiplier of the active constraint is $\mu_1 = 2(\Sigma w)_1 + \lambda = 0.0080 > 0$ (dual feasibility holds), while $\mu_2 = \mu_3 = 0$ because those constraints are slack (complementary slackness). All four conditions hold, and since the problem is convex this is the global optimum. The multiplier has a meaning: relaxing the rule to allow $w_1 \ge -0.01$ would lower the variance by about $0.008 \times 0.01 = 8\times10^{-5}$, from $0.01937$ to $0.01929$ — the price of the no-shorting rule, in variance units.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Empirical risk minimisation | every `fit`, `loss.backward(); optimizer.step()` | defines what "training" means |
| Convexity | linear/logistic regression, ridge, lasso, SVM | unique optimum; solver choice does not change the answer |
| Non-convexity, saddles | deep networks | seeds matter; plateaus are saddles, not convergence |
| $\eta < 2/L$, rate $(\kappa-1)/(\kappa+1)$ | learning-rate choice, `StandardScaler`, batch/layer norm | explains divergence and slow training |
| Unbiased mini-batch gradients | `DataLoader`, `SGDClassifier`, online recommenders | scales to data that does not fit in memory |
| LR schedules, warm-up | `CosineAnnealingLR`, `OneCycleLR`, transformer recipes | shrinks the noise ball; protects early steps |
| Momentum / Nesterov | `SGD(momentum=0.9)` for CNNs | $\sqrt\kappa$ instead of $\kappa$ steps |
| Adam / AdamW | `AdamW` for transformers, diffusion models | per-parameter scale invariance; decoupled decay |
| Newton, L-BFGS | `LogisticRegression`, IRLS for GLMs, `scipy.optimize` | fast, accurate for small–medium $d$ |
| Gauss–Newton / LM | `curve_fit`, bundle adjustment in computer vision | nonlinear least squares |
| Lagrange / KKT | SVM dual, max-entropy ⇒ softmax, TRPO/PPO KL constraint | structure of the solution; support vectors |
| Projection / penalty | NMF, weight constraints, RLHF KL penalty | enforce constraints inside gradient loops |
| Clipping, early stopping | `clip_grad_norm_`, `EarlyStopping` | stability; regularisation |

## Cheat sheet

| Item | Formula / fact |
|---|---|
| Empirical risk | $L(\theta) = \frac1n \sum_i \ell(f_\theta(x_i), y_i) + \lambda R(\theta)$ |
| Convex function | $f(\alpha x + (1-\alpha) y) \le \alpha f(x) + (1-\alpha) f(y)$; equivalently $\nabla^2 f \succeq 0$ |
| Strong convexity / smoothness | $\mu I \preceq \nabla^2 f \preceq L I$, $\kappa = L/\mu$ |
| GD | $\theta_{k+1} = \theta_k - \eta\nabla L(\theta_k)$ |
| Stable step (quadratic) | $\eta < 2/L$; optimal $\eta^\star = 2/(L+\mu)$; rate $(\kappa-1)/(\kappa+1)$ |
| Steps to accuracy $\varepsilon$ | $\approx \kappa\ln(1/\varepsilon)$ with $\eta = 1/L$ |
| Mini-batch gradient | $\mathbb{E}[g_B] = \nabla L$, $\mathrm{Cov}(g_B) \approx \Sigma / b$ |
| SGD noise ball (1-D) | $\mathrm{Var}(\theta_\infty) \approx \eta\sigma^2/(2\lambda)$ |
| Cosine schedule | $\eta_t = \eta_{\min} + \frac12(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$ |
| Heavy ball | $v_t = \beta v_{t-1} + g_t$, $\theta_{t+1} = \theta_t - \eta v_t$; effective step $\eta/(1-\beta)$; rate $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ |
| RMSProp | $s_t = \rho s_{t-1} + (1-\rho) g_t^2$, step $\eta g_t/(\sqrt{s_t}+\epsilon)$ |
| Adam | $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$, step $\eta \hat m_t/(\sqrt{\hat v_t}+\epsilon)$ |
| AdamW | add $\eta\lambda\theta_t$ to the step, outside the adaptive scaling |
| Newton | $\theta_{k+1} = \theta_k - H^{-1}\nabla L$; quadratic convergence; $O(d^3)$ per step |
| Gauss–Newton / LM | $(J^\top J + \tau I)\, p = -J^\top r$ |
| Lagrange (equality) | $\nabla f + \lambda\nabla h = 0$, $h = 0$; $-\lambda$ = shadow price |
| KKT | stationarity, $g_j \le 0$, $\mu_j \ge 0$, $\mu_j g_j = 0$ |
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

**`lesson.ipynb`** builds everything in NumPy: contour plots of an ill-conditioned quadratic with GD paths for several step sizes, including one that diverges; an experiment measuring GD's convergence rate against the condition number and checking it against $(\kappa-1)/(\kappa+1)$; GD versus mini-batch SGD on a linear-regression problem, with the noise floor and a decaying schedule; momentum, RMSProp and Adam implemented from their update equations and compared on the Rosenbrock function and on logistic regression for the standardised breast-cancer data; Newton versus GD on that same logistic regression; a visualisation of the Lagrange "tangent level set" condition; and the common learning-rate schedules plotted side by side.

**`exercises.ipynb`** (with `solutions.ipynb`) tests the same ideas: the largest stable step on a given quadratic, mini-batch SGD for linear regression, a convexity check via Hessian eigenvalues, cosine and warm-up schedules, Adam from the equations matched against a reference run, Newton's method for logistic regression, a Lagrange problem solved by hand, gradient clipping and a learning-rate range test, two conceptual questions on momentum and on KKT, and a mini-project that trains logistic regression with GD, momentum and Adam on a synthetic dataset and studies the sensitivity to the learning rate.

Next: [Module 06 — Automatic differentiation and backpropagation](../06-autodiff-and-backpropagation/README.md), where the gradients this module consumes are produced automatically.
