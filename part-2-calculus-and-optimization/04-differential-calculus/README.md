# Module 04 — Differential Calculus for ML

> After this module you can take any loss function a model computes, write down its gradient, Jacobian and Hessian by hand, check them numerically, and explain exactly what "training" is doing to the parameters.

## Why this matters for ML and AI

Every learning algorithm you will meet in this course — linear regression, logistic regression, neural networks, transformers, diffusion models, policy-gradient reinforcement learning — is trained the same way: write a scalar *loss* that measures how wrong the model is, compute how that loss changes when each parameter moves a little, and nudge the parameters in the direction that reduces it. "How the loss changes when a parameter moves a little" is a derivative. For a model with a billion parameters it is a gradient with a billion entries, and the machinery that computes it (backpropagation, automatic differentiation in PyTorch and JAX) is nothing more than the chain rule applied systematically. When a GPT-style model is trained, the `loss.backward()` call multiplies Jacobians along a computational graph; the `optimizer.step()` call uses the resulting gradient (and, for Adam, a running estimate of curvature) to move the weights.

Calculus also tells you *where* a model can get stuck and *how fast* it can move. Second derivatives — the Hessian — describe curvature. Curvature decides whether a flat spot is a minimum or a saddle, how large a learning rate you can afford, why some directions in parameter space train a thousand times slower than others, and why "sharp" minima generalise worse than "flat" ones. Taylor expansion turns a complicated loss into a local linear or quadratic model, and essentially every optimiser in Module 05 is a rule for minimising such a local model.

Finally, matrix calculus is the language of the derivations you will read in papers: the gradient of a least-squares loss gives the normal equations, the gradient of logistic loss is the famously clean $\sigma(z) - y$, the gradient of softmax-cross-entropy is $p - y$, and the Jacobian of a layer is what flows backward through a network. This module gives you those results with every step shown, so that in later modules you can read a gradient formula and know where it came from.

## Learning objectives

- Define the derivative as a limit and as the slope of the best linear approximation, and compute one by hand and numerically.
- Choose a finite-difference step size by balancing truncation error against round-off error, and explain the resulting log-log error curve.
- Apply the sum, product, quotient and chain rules fluently, including to the sigmoid, tanh, softplus and ReLU activations.
- Compute partial derivatives and gradients of multivariate functions, and prove that the gradient is the direction of steepest ascent.
- Build the Jacobian of a vector-valued map, use it to linearise the map and propagate measurement errors, and multiply Jacobians along a composition.
- Compute a Hessian, classify critical points by its eigenvalues, and explain why saddle points dominate high-dimensional landscapes.
- Write first- and second-order Taylor expansions in one and many variables and bound the error.
- Derive the key matrix-calculus identities, including $\nabla_x \lVert Ax - b \rVert^2 = 2A^\top(Ax - b)$, and recover the normal equations from them.
- Derive the gradients of the MSE, logistic (binary cross-entropy), softmax-cross-entropy and hinge losses with respect to model outputs and weights.
- Validate any analytic gradient with a finite-difference gradient checker and interpret its relative error.

## Prerequisites

- [Module 01 — Vectors and geometry](../../part-1-linear-algebra/01-vectors-and-geometry/README.md): dot products, norms, the Cauchy–Schwarz inequality, projections.
- [Module 02 — Matrices and linear maps](../../part-1-linear-algebra/02-matrices-and-linear-maps/README.md): matrix–vector products, transposes, the normal equations $A^\top A x = A^\top b$.
- From [Module 03 — Eigenvalues, SVD and PCA](../../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md) you will need only one fact, which we restate when it is used: a symmetric matrix has real eigenvalues and orthogonal eigenvectors, and it is positive definite exactly when all its eigenvalues are positive.
- High-school calculus: what a function, a limit and a slope are. Everything else is rebuilt here.

Notation used throughout: scalars are lower-case ($x$, $h$), vectors are bold lower-case ($\mathbf{x} \in \mathbb{R}^n$), matrices are upper-case ($A \in \mathbb{R}^{m \times n}$). $\mathbf{x}^\top \mathbf{y}$ is the dot product. All vectors are column vectors.

## 1. The derivative: sensitivity and best linear approximation

**Intuition.** A derivative answers one question: *if I change the input a little, how much does the output change, per unit of input?* If you drive and your odometer reads $s(t)$ kilometres at time $t$ hours, the derivative $s'(t)$ is your speed — the ratio of a small distance to the small time it took. The same number has a second meaning: it is the slope of the straight line that best approximates the function near $t$. Zoom in far enough on any smooth curve and it looks like a line; the derivative is the slope of that line. Both views matter in ML: *sensitivity* ("how much does the loss care about this weight?") and *linearisation* ("replace the complicated loss by a line and reason about the line").

**Definition.** For $f: \mathbb{R} \to \mathbb{R}$ the derivative at $x$ is

$$
f'(x) = \frac{df}{dx}(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h},
$$

provided the limit exists. Here $h$ is a small change in the input, $f(x+h) - f(x)$ is the resulting change in the output, and the quotient is the average rate of change over the interval; the derivative is the limiting *instantaneous* rate. Equivalently, $f'(x)$ is the unique number $a$ such that

$$
f(x+h) = f(x) + a\,h + o(h),
$$

where $o(h)$ denotes a remainder that shrinks faster than $h$ (so $o(h)/h \to 0$). The line $\ell(x+h) = f(x) + f'(x)h$ is the **tangent line**, and $a\,h$ is the **differential**. A function that has a derivative at every point is *differentiable*; a differentiable function is continuous, but not conversely ($|x|$ is continuous at 0 and has no derivative there).

**Numerical derivatives and their error.** You cannot take a limit on a computer, so you pick a small $h$ and use a *finite difference*:

| scheme | formula | truncation error |
|---|---|---|
| forward | $\dfrac{f(x+h) - f(x)}{h}$ | $\tfrac{h}{2} f''(x) + \dots = O(h)$ |
| central | $\dfrac{f(x+h) - f(x-h)}{2h}$ | $\tfrac{h^2}{6} f'''(x) + \dots = O(h^2)$ |

The truncation error comes from the Taylor expansion (Section 6). But there is a second error source: $f(x+h)$ and $f(x)$ are stored with about 16 significant digits (double precision, machine epsilon $\varepsilon \approx 2.2 \times 10^{-16}$), so their difference carries an absolute error of order $\varepsilon |f(x)|$, and dividing by $h$ amplifies it to $\varepsilon |f| / h$. Total error for the central scheme is roughly

$$
E(h) \approx \frac{h^2}{6}|f'''| + \frac{\varepsilon |f|}{h},
$$

which is minimised at $h \approx \varepsilon^{1/3} \approx 6 \times 10^{-6}$, with best achievable error around $10^{-10}$ to $10^{-11}$ (for the forward scheme: $h \approx \varepsilon^{1/2} \approx 10^{-8}$, error $\approx 10^{-8}$). On a log-log plot of error versus $h$ this produces a V shape: a straight line of slope 2 (truncation) on the right, a line of slope $-1$ (round-off) on the left. You will draw this plot in the lesson notebook. The practical lesson: *never* use $h = 10^{-12}$ "to be safe"; the result would be mostly noise.

**Worked example.** Let $f(x) = x^3$ at $x = 2$, where the exact derivative is $f'(2) = 3 \cdot 2^2 = 12$. With $h = 0.1$:

- forward: $\dfrac{2.1^3 - 2^3}{0.1} = \dfrac{9.261 - 8}{0.1} = 12.61$ (error $0.61$);
- central: $\dfrac{2.1^3 - 1.9^3}{0.2} = \dfrac{9.261 - 6.859}{0.2} = 12.01$ (error $0.01$).

The forward error $0.61$ matches the prediction $\tfrac{h}{2}f''(2) = 0.05 \cdot 12 = 0.6$; the central error $0.01$ matches $\tfrac{h^2}{6}f'''(2) = \tfrac{0.01}{6}\cdot 6 = 0.01$ exactly, because $x^3$ has no fourth derivative to spoil it. Halving $h$ would halve the forward error and quarter the central error.

**Real-life example (pharmacology).** A drug's effect often follows an $E_{\max}$ curve, $E(D) = \dfrac{E_{\max} D}{ED_{50} + D}$, with dose $D$ in mg. Take $E_{\max} = 100$ (percent of maximal effect) and $ED_{50} = 20$ mg. The derivative is $E'(D) = \dfrac{E_{\max} ED_{50}}{(ED_{50} + D)^2}$. At $D = 20$ mg, $E = 50$ and $E'(20) = \dfrac{100 \cdot 20}{40^2} = 1.25$ percent per mg: adding 10 mg should buy about $12.5$ points (the exact gain is $E(30) - E(20) = 60 - 50 = 10$; the tangent overshoots because the curve bends down). At $D = 80$ mg, $E = 80$ but $E'(80) = \dfrac{2000}{100^2} = 0.2$ percent per mg — the same 10 mg now buys under 2 points. The derivative is the *marginal benefit* of dose, and its collapse is the quantitative reason dose-escalation trials stop where they do. An economist would call the same quantity *marginal cost* or *marginal revenue*.

**In ML/AI this is …** $\partial L / \partial w_i$, the sensitivity of the loss to weight $w_i$ that `loss.backward()` computes for every parameter. The finite-difference formula is the ground truth used by `torch.autograd.gradcheck` and every hand-written gradient checker, and the step-size analysis is why those tools use $h \approx 10^{-6}$ in double precision and why gradient checks in `float32` are unreliable.

## 2. The rules you actually use

**Intuition.** You rarely differentiate from the limit definition. Instead you memorise the derivatives of a handful of building blocks and four rules for combining them. The most important rule is the chain rule: *when functions are composed, their slopes multiply.* If $y$ changes 3 times as fast as $u$, and $u$ changes 2 times as fast as $x$, then $y$ changes $3 \times 2 = 6$ times as fast as $x$. A deep network is a long composition, and backpropagation is this multiplication carried out layer by layer.

**Definitions and statements.** For differentiable $f, g$ and constants $a, b$:

| rule | statement |
|---|---|
| linearity | $(a f + b g)' = a f' + b g'$ |
| product | $(fg)' = f'g + fg'$ |
| quotient | $\left(\dfrac{f}{g}\right)' = \dfrac{f'g - fg'}{g^2}$ |
| chain | $\dfrac{d}{dx} f(g(x)) = f'(g(x))\, g'(x)$ |

The building-block derivatives you will use constantly:

| function | derivative | note |
|---|---|---|
| $x^n$ | $n x^{n-1}$ | any real $n$, including $\sqrt{x} = x^{1/2}$ and $1/x = x^{-1}$ |
| $e^x$ | $e^x$ | the only function (up to scale) equal to its own derivative |
| $\ln x$ | $1/x$ | hence $(\ln f)' = f'/f$, the "log-derivative trick" |
| $\sin x$, $\cos x$ | $\cos x$, $-\sin x$ | |
| $\sigma(x) = \dfrac{1}{1+e^{-x}}$ | $\sigma(x)\bigl(1 - \sigma(x)\bigr)$ | maximum $1/4$ at $x = 0$ |
| $\tanh x$ | $1 - \tanh^2 x$ | maximum $1$ at $x = 0$ |
| $\mathrm{softplus}(x) = \ln(1 + e^x)$ | $\sigma(x)$ | smooth ReLU |
| $\mathrm{ReLU}(x) = \max(0, x)$ | $1$ if $x > 0$, $0$ if $x < 0$ | undefined at $0$; see below |

*Deriving the sigmoid rule* (worth doing once): write $\sigma = (1 + e^{-x})^{-1}$ and apply the chain rule with outer function $u^{-1}$ and inner $u = 1 + e^{-x}$:

$$
\sigma'(x) = -(1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x)\,(1 - \sigma(x)),
$$

using $\dfrac{e^{-x}}{1+e^{-x}} = 1 - \dfrac{1}{1+e^{-x}}$. The payoff: once you have computed $\sigma(x)$ in the forward pass, its derivative costs one subtraction and one multiplication — no extra exponentials.

*ReLU at zero.* $\max(0, x)$ has slope 0 on the left and 1 on the right, so no single tangent line exists at $x = 0$. Any slope in $[0, 1]$ is a valid **subgradient** (a line through the point that stays below the function). Frameworks simply pick 0 (PyTorch) and move on; since a weight sum lands exactly at $0.0$ with probability zero in practice, the choice almost never matters.

**Worked examples.**

1. Chain rule: $h(x) = e^{-x^2/2}$ (the Gaussian bell). Outer $e^u$, inner $u = -x^2/2$, $u' = -x$. So $h'(x) = e^{-x^2/2} \cdot (-x) = -x\,e^{-x^2/2}$. At $x = 1$: $h'(1) = -e^{-1/2} \approx -0.607$.
2. Sigmoid at $x = 2$: $\sigma(2) = 1/(1 + e^{-2}) = 1/(1 + 0.1353) \approx 0.8808$, so $\sigma'(2) = 0.8808 \times 0.1192 \approx 0.1050$. Already at $x = 2$ the slope is only $0.105$; at $x = 5$ it is $0.0066$. This is *saturation*.
3. A one-neuron network: $z = wx + b$, $p = \sigma(z)$, $L = (p - y)^2$, with $w = 0.5$, $b = 0$, $x = 2$, $y = 1$. Forward: $z = 1$, $p = \sigma(1) \approx 0.7311$, $L \approx 0.0723$. Backward, by the chain rule three times:

$$
\frac{dL}{dw} = \frac{dL}{dp}\cdot\frac{dp}{dz}\cdot\frac{dz}{dw} = 2(p - y)\cdot p(1-p)\cdot x = 2(-0.2689)(0.1966)(2) \approx -0.2115.
$$

Increasing $w$ slightly decreases the loss (the derivative is negative), which is correct: $p$ is below the target $y = 1$ and a larger $w$ pushes it up.

**Real-life example (physics / engineering).** A spherical weather balloon is inflated so that its radius grows at $dr/dt = 0.5$ cm/s. Its volume is $V = \tfrac{4}{3}\pi r^3$, a function of $r$, which is a function of $t$. By the chain rule, $\dfrac{dV}{dt} = \dfrac{dV}{dr}\cdot\dfrac{dr}{dt} = 4\pi r^2 \cdot 0.5$. When $r = 10$ cm this is $4\pi \cdot 100 \cdot 0.5 \approx 628$ cm³/s. The "related rates" problems of a first calculus course are the chain rule in disguise, and so is backpropagation.

**In ML/AI this is …** the backward pass of every activation layer. `torch.sigmoid` stores its output $\sigma(z)$ and multiplies the incoming gradient by $\sigma(1 - \sigma)$; `torch.tanh` multiplies by $1 - \tanh^2$; ReLU multiplies by a 0/1 mask. The fact that $\sigma' \le 1/4$ is the mathematical origin of the *vanishing-gradient problem*: a 10-layer sigmoid network multiplies ten factors each at most $0.25$, so the gradient reaching the first layer is at most $0.25^{10} \approx 10^{-6}$ times the gradient at the output. ReLU's slope of exactly 1 on its active side is why it replaced sigmoid in deep networks (Module 16).

## 3. Partial derivatives and the gradient

**Intuition.** A loss depends on many parameters at once. The honest way to ask "how sensitive is the loss to weight number 7?" is to wiggle weight 7 while holding every other weight fixed. That is a partial derivative. Collect all of them into a vector and you have the gradient. Picture a hilly landscape with altitude $f(x, y)$: the partial derivative in $x$ is the slope you feel walking due east, the partial in $y$ is the slope walking due north, and the gradient vector points in the direction of the steepest climb, with length equal to that steepest slope. Water flows along $-\nabla f$. So does gradient descent.

**Definition.** For $f: \mathbb{R}^n \to \mathbb{R}$, the partial derivative with respect to the $i$-th coordinate at $\mathbf{x}$ is

$$
\frac{\partial f}{\partial x_i}(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\,\mathbf{e}_i) - f(\mathbf{x})}{h},
$$

where $\mathbf{e}_i$ is the $i$-th standard basis vector (a 1 in position $i$, zeros elsewhere). The **gradient** is the column vector of all partials,

$$
\nabla f(\mathbf{x}) = \begin{pmatrix} \partial f / \partial x_1 \\ \vdots \\ \partial f / \partial x_n \end{pmatrix} \in \mathbb{R}^n .
$$

The **directional derivative** of $f$ along a unit vector $\mathbf{u}$ (so $\lVert \mathbf{u} \rVert = 1$) is the slope you feel walking in direction $\mathbf{u}$:

$$
D_{\mathbf{u}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h} = \nabla f(\mathbf{x})^\top \mathbf{u},
$$

where the last equality holds for differentiable $f$ (it follows from the linear approximation $f(\mathbf{x} + h\mathbf{u}) \approx f(\mathbf{x}) + h\,\nabla f^\top \mathbf{u}$, Section 6). Partial derivatives are the special case $\mathbf{u} = \mathbf{e}_i$.

**Theorem (steepest ascent).** Among all unit directions $\mathbf{u}$, the directional derivative is largest when $\mathbf{u}$ points along $\nabla f$, and that largest value is $\lVert \nabla f \rVert$. *Proof:* by Cauchy–Schwarz (Module 01), $\nabla f^\top \mathbf{u} \le \lVert \nabla f \rVert \lVert \mathbf{u} \rVert = \lVert \nabla f \rVert$, with equality exactly when $\mathbf{u}$ is a positive multiple of $\nabla f$, i.e. $\mathbf{u} = \nabla f / \lVert \nabla f \rVert$. Likewise the steepest *descent* direction is $-\nabla f$, with slope $-\lVert \nabla f \rVert$. Any $\mathbf{u}$ orthogonal to the gradient gives slope $0$: moving along it leaves $f$ unchanged to first order, so **the gradient is perpendicular to the level set** $\{\mathbf{x} : f(\mathbf{x}) = c\}$ through the point. On a contour plot the gradient arrows cross the contour lines at right angles — the lesson notebook draws this.

**Worked example.** $f(x, y) = x^2 + 3xy$. Treating $y$ as a constant, $\partial f/\partial x = 2x + 3y$; treating $x$ as a constant, $\partial f/\partial y = 3x$. At $(1, 2)$:

$$
\nabla f(1, 2) = \begin{pmatrix} 2 + 6 \\ 3 \end{pmatrix} = \begin{pmatrix} 8 \\ 3 \end{pmatrix}, \qquad \lVert \nabla f \rVert = \sqrt{64 + 9} = \sqrt{73} \approx 8.544.
$$

Walking north-east, $\mathbf{u} = (1, 1)/\sqrt{2}$, the slope is $(8 + 3)/\sqrt{2} \approx 7.78$: steep, but less than the maximum $8.544$ available along $(8, 3)/\sqrt{73}$. Walking along $(-3, 8)/\sqrt{73}$, perpendicular to the gradient, the slope is $(-24 + 24)/\sqrt{73} = 0$. Check the linear approximation: moving by $(0.01, 0.01)$ predicts a change of $8(0.01) + 3(0.01) = 0.11$, so $f \approx 7.11$; the exact value is $f(1.01, 2.01) = 1.0201 + 6.0903 = 7.1104$. The error $0.0004$ is second order, as it should be.

**Real-life example (sport / hiking).** A trail-running app stores altitude as a function $a(x, y)$ of map coordinates in metres. At your position the app estimates $\nabla a = (-0.10, -0.08)$ — altitude drops 10 cm per metre eastward and 8 cm per metre northward. The steepest descent is then towards $(0.10, 0.08)/0.128$, roughly east-north-east, with a grade of $\lVert \nabla a \rVert = \sqrt{0.01 + 0.0064} \approx 0.128$, i.e. 12.8 %. If you prefer a gentle route, follow the contour: head along $(-0.08, 0.10)$ or its negative, where the grade is zero. The same calculation underlies how a baker tunes two knobs of a recipe (oven temperature and baking time) when a judge's score is a function of both: the gradient says which knob matters more and in which direction to turn it.

**In ML/AI this is …** `param.grad`, the gradient of the loss with respect to each weight tensor, which `optimizer.step()` subtracts scaled by the learning rate. The loss landscape $L(\boldsymbol{\theta})$ is the altitude map, the steepest-ascent theorem is why "descend along $-\nabla L$" is the default move, and "gradient ⟂ level sets" is why gradient descent zig-zags across narrow valleys (Module 05). Saliency maps in computer vision are gradients too: $\nabla_{\text{pixels}}\,\text{score}$ shows which pixels the classifier is most sensitive to.

## 4. Vector-valued functions and the Jacobian

**Intuition.** A layer of a neural network takes a vector in and produces a vector out; so does a change of coordinates, a robot arm (joint angles in, hand position out), or a sensor model. For such a map every output can depend on every input, so its "derivative" is a *table* of partials: output $i$ with respect to input $j$. That table is the Jacobian matrix, and it is the matrix of the best *linear* approximation of the map. Composition of maps turns into multiplication of Jacobians — the chain rule, now in matrix form.

**Definition.** For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ with components $f_1, \dots, f_m$, the **Jacobian** at $\mathbf{x}$ is the $m \times n$ matrix

$$
J_{\mathbf{f}}(\mathbf{x}) = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} =
\begin{pmatrix}
\partial f_1 / \partial x_1 & \cdots & \partial f_1 / \partial x_n \\
\vdots & \ddots & \vdots \\
\partial f_m / \partial x_1 & \cdots & \partial f_m / \partial x_n
\end{pmatrix},
\qquad (J)_{ij} = \frac{\partial f_i}{\partial x_j}.
$$

Row $i$ is the gradient of the $i$-th output, transposed. When $m = 1$, $J = \nabla f^\top$ (a row). The Jacobian defines the **linearisation**

$$
\mathbf{f}(\mathbf{x} + \boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + J_{\mathbf{f}}(\mathbf{x})\,\boldsymbol{\delta},
$$

with an error of order $\lVert \boldsymbol{\delta} \rVert^2$. **Chain rule in matrix form:** if $\mathbf{h}(\mathbf{x}) = \mathbf{g}(\mathbf{f}(\mathbf{x}))$ with $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ and $\mathbf{g}: \mathbb{R}^m \to \mathbb{R}^p$, then

$$
J_{\mathbf{h}}(\mathbf{x}) = J_{\mathbf{g}}\bigl(\mathbf{f}(\mathbf{x})\bigr)\; J_{\mathbf{f}}(\mathbf{x}), \qquad (p \times n) = (p \times m)(m \times n).
$$

The shapes force the order: the outer function's Jacobian stands on the left. A useful sanity check on any Jacobian you write down is that its shape must be (number of outputs) × (number of inputs). When $m = n$, $|\det J|$ is the local volume-scaling factor of the map; it is the reason a $r$ appears in polar-coordinate integrals and the reason normalising flows (Module 18) need log-determinants of Jacobians.

**Worked example (polar to Cartesian).** $\mathbf{P}(r, \theta) = (r\cos\theta,\; r\sin\theta)$. The Jacobian is

$$
J_{\mathbf{P}} = \begin{pmatrix} \partial x/\partial r & \partial x/\partial \theta \\ \partial y/\partial r & \partial y/\partial \theta \end{pmatrix}
= \begin{pmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{pmatrix}, \qquad \det J_{\mathbf{P}} = r\cos^2\theta + r\sin^2\theta = r .
$$

At $r = 2$, $\theta = \pi/6$: $\cos\theta = 0.866$, $\sin\theta = 0.5$, so

$$
J = \begin{pmatrix} 0.866 & -1.000 \\ 0.500 & 1.732 \end{pmatrix}, \qquad \mathbf{P}(2, \pi/6) = (1.732,\; 1.000).
$$

Perturb by $\boldsymbol{\delta} = (0.1, 0.05)$. Linear prediction: $J\boldsymbol{\delta} = (0.0866 - 0.05,\; 0.05 + 0.0866) = (0.0366,\; 0.1366)$. Exact change: $\mathbf{P}(2.1, \pi/6 + 0.05) - \mathbf{P}(2, \pi/6) = (0.0319,\; 0.1396)$. The prediction is off by $(-0.0048, 0.0030)$; halving $\boldsymbol{\delta}$ shrinks the error to about $(-0.0012, 0.0008)$ — a factor of 4, confirming the quadratic error order.

**Real-life example (engineering: error propagation).** A total station measures a point by distance $r$ and bearing $\theta$, with independent errors of standard deviation $\sigma_r$ and $\sigma_\theta$. What is the uncertainty of the Cartesian position? Linearise: $\boldsymbol{\delta}_{xy} \approx J\,\boldsymbol{\delta}_{r\theta}$, so the covariance transforms as $\Sigma_{xy} \approx J\,\Sigma_{r\theta}\,J^\top$. The entry $-r\sin\theta$ in $J$ says that a small angular error is multiplied by the distance $r$ before it becomes a position error — at $r = 2$ km, a bearing error of $0.5°$ ($0.0087$ rad) becomes a $17$ m sideways error (Case study 1 carries the full computation through). The same $J \Sigma J^\top$ rule is the delta method in statistics (Module 10) and the covariance-propagation step of the Kalman filter (Module 09).

**In ML/AI this is …** the object backpropagation moves around. For a layer $\mathbf{y} = \mathbf{f}(\mathbf{x})$ the backward pass receives $\mathbf{g} = \partial L / \partial \mathbf{y}$ and returns $\partial L / \partial \mathbf{x} = J_{\mathbf{f}}^\top \mathbf{g}$, a *vector–Jacobian product* (VJP). Reverse-mode autodiff never forms the full Jacobian; it applies $J_1^\top J_2^\top \cdots J_k^\top$ to the loss gradient one factor at a time (Module 06 builds this from scratch). The softmax Jacobian $\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$ of Module 17 is a Jacobian in exactly this sense.

## 5. The Hessian and second-order information

**Intuition.** The gradient tells you the slope; the Hessian tells you how the slope *changes* — the curvature. Standing at a point where the ground is level (gradient zero), the Hessian tells you whether you are at the bottom of a bowl, the top of a dome, or on a saddle, where the ground curves up in one direction and down in another. In two dimensions a saddle is a special event; in a million dimensions, where a critical point is a minimum only if the ground curves up in *every* one of a million directions, it is the norm.

**Definition.** For twice-differentiable $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** is the $n \times n$ matrix of second partial derivatives,

$$
H_f(\mathbf{x}) = \nabla^2 f(\mathbf{x}), \qquad (H)_{ij} = \frac{\partial^2 f}{\partial x_i\,\partial x_j} .
$$

It is the Jacobian of the gradient map $\mathbf{x} \mapsto \nabla f(\mathbf{x})$. **Schwarz's theorem:** when the second partials are continuous, $\partial^2 f / \partial x_i \partial x_j = \partial^2 f / \partial x_j \partial x_i$, so $H$ is symmetric and (Module 03) has real eigenvalues $\lambda_1, \dots, \lambda_n$ with orthonormal eigenvectors. The curvature of $f$ along a unit direction $\mathbf{u}$ is $\mathbf{u}^\top H \mathbf{u}$; the eigenvectors are the directions of extreme curvature and the eigenvalues are those curvatures.

**Second-derivative test.** At a **critical point** $\mathbf{x}^*$ (where $\nabla f(\mathbf{x}^*) = \mathbf{0}$):

| eigenvalues of $H(\mathbf{x}^*)$ | $H$ is | $\mathbf{x}^*$ is |
|---|---|---|
| all $> 0$ | positive definite | strict local minimum |
| all $< 0$ | negative definite | strict local maximum |
| mixed signs | indefinite | saddle point |
| some $= 0$, rest one sign | semidefinite | inconclusive (need higher order) |

In two dimensions there is a shortcut: with $D = \det H = f_{xx}f_{yy} - f_{xy}^2$, a saddle has $D < 0$; $D > 0$ gives a minimum if $f_{xx} > 0$ and a maximum if $f_{xx} < 0$ (because $\det H = \lambda_1 \lambda_2$ and $\operatorname{tr} H = \lambda_1 + \lambda_2$).

**Worked example.** $f(x, y) = x^2 + 3xy + 5y^2$. Gradient: $(2x + 3y,\; 3x + 10y)$, zero only at the origin. Hessian:

$$
H = \begin{pmatrix} 2 & 3 \\ 3 & 10 \end{pmatrix}, \qquad \det H = 20 - 9 = 11 > 0,\quad f_{xx} = 2 > 0 \;\Rightarrow\; \text{minimum}.
$$

The eigenvalues solve $\lambda^2 - 12\lambda + 11 = 0$, giving $\lambda = 1$ and $\lambda = 11$: the bowl is 11 times steeper (in curvature) along the eigenvector for $\lambda = 11$, roughly $(0.32, 0.95)$, than along $(-0.95, 0.32)$. Now change one coefficient: $g(x, y) = x^2 + 3xy$ (the function from Section 3). Its Hessian is $\begin{pmatrix} 2 & 3 \\ 3 & 0 \end{pmatrix}$ with $\det = -9 < 0$; eigenvalues $1 \pm \sqrt{10}$, i.e. $4.16$ and $-2.16$. The origin is a saddle: $g$ increases along the $x$-axis ($g(t, 0) = t^2$) and decreases along $(3, -2)$, since $g(3t, -2t) = 9t^2 - 18t^2 = -9t^2$.

**Why saddles dominate in high dimensions.** At a random critical point of a generic function in $\mathbb{R}^n$, think of each eigenvalue's sign as a coin flip. A minimum needs all $n$ flips positive, probability $2^{-n}$; for $n = 10^6$ parameters that is effectively zero, and almost every critical point with a loss well above the global minimum is a saddle. Theory for random landscapes and experiments on real networks agree: the practical obstacle in deep learning is not getting stuck in a bad local minimum but *slowing down* near saddles and plateaus, where the gradient is tiny. Momentum and the noise in SGD help precisely because they keep moving along the negative-curvature directions.

**Real-life example (physics: stability of an equilibrium).** A marble rests where the potential energy $U(x, y)$ has zero gradient. The Hessian of $U$ decides its fate: positive definite means every small push raises the energy and the marble rolls back (stable equilibrium, like the bottom of a bowl); indefinite means some push lowers the energy and the marble leaves (unstable, like a saddle-shaped mountain pass). A pencil balanced on its tip is a maximum of $U$. Engineers checking whether a bridge configuration or a power-grid operating point is stable perform exactly this eigenvalue test on the Hessian of an energy function.

**In ML/AI this is …** the matrix behind Newton's method $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - H^{-1}\nabla L$, behind the learning-rate limit $\eta < 2/\lambda_{\max}$, and behind the condition number $\lambda_{\max}/\lambda_{\min}$ that controls how slowly gradient descent converges (all Module 05). "Sharpness" of a minimum means a large $\lambda_{\max}(H)$. Forming $H$ for a billion parameters is impossible, but a *Hessian–vector product* $H\mathbf{v}$ costs about one extra backward pass — it is what `torch.autograd.functional.hvp` computes and what power iteration (Module 03) uses to estimate $\lambda_{\max}$.

## 6. Taylor expansion: local linear and quadratic models

**Intuition.** Near a point, a smooth function is well described by a polynomial built from its derivatives at that point: the value gives the height, the first derivative the tilt, the second the bend. Keep one term and you have the tangent line; keep two and you have the best-fitting parabola. Taylor's theorem says how far you can trust the approximation. Almost every optimiser is a rule for minimising such a local model and then moving, so this section is the bridge to Module 05.

**Statement.** For $f: \mathbb{R} \to \mathbb{R}$ with $k + 1$ continuous derivatives, the degree-$k$ Taylor polynomial about $a$ is

$$
T_k(a + h) = f(a) + f'(a)\,h + \frac{f''(a)}{2!}h^2 + \dots + \frac{f^{(k)}(a)}{k!}h^k,
$$

and the error (Lagrange form of the remainder) is

$$
f(a + h) - T_k(a + h) = \frac{f^{(k+1)}(\xi)}{(k+1)!}\,h^{k+1} \quad \text{for some } \xi \text{ between } a \text{ and } a + h .
$$

So the error of the degree-$k$ model is $O(h^{k+1})$: the linear model has quadratic error, the quadratic model has cubic error. In $n$ dimensions, for $f: \mathbb{R}^n \to \mathbb{R}$,

$$
f(\mathbf{x} + \boldsymbol{\delta}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \boldsymbol{\delta} + \tfrac{1}{2}\,\boldsymbol{\delta}^\top H_f(\mathbf{x})\,\boldsymbol{\delta} + O(\lVert \boldsymbol{\delta} \rVert^3).
$$

The first two terms are the **first-order (linear) model**; all three are the **second-order (quadratic) model**. The latter is exactly where the Hessian's role as curvature comes from: along a unit direction $\mathbf{u}$, $f(\mathbf{x} + t\mathbf{u}) \approx f + t\,\nabla f^\top\mathbf{u} + \tfrac{1}{2}t^2\,\mathbf{u}^\top H\mathbf{u}$.

**Worked example.** Expand $e^x$ about $a = 0$: all derivatives equal $1$ at $0$, so $T_3(x) = 1 + x + x^2/2 + x^3/6$. At $x = 0.5$: $T_3 = 1 + 0.5 + 0.125 + 0.020833 = 1.645833$, versus $e^{0.5} = 1.648721$. The error $0.002888$ is below the Lagrange bound $\dfrac{e^{\xi}}{4!}(0.5)^4 \le \dfrac{1.6487 \cdot 0.0625}{24} = 0.00429$. Two more: $\ln(1 + x) = x - x^2/2 + x^3/3 - \dots$ gives $\ln 1.1 \approx 0.1 - 0.005 + 0.000333 = 0.095333$ against the true $0.095310$; and $\sin x = x - x^3/6 + \dots$ gives $\sin 0.5 \approx 0.5 - 0.020833 = 0.479167$ against $0.479426$. Each extra term buys roughly two more correct digits when $|x| = 0.5$.

**Real-life example (engineering: linearising a sensor).** The period of a pendulum of length $L$ is $T = 2\pi\sqrt{L/g}$ only under the **small-angle approximation** $\sin\theta \approx \theta$, the first Taylor term. At $10°$ ($0.1745$ rad), $\sin\theta = 0.1736$ and the relative error is $0.5\%$; at $30°$ it is $4.7\%$, and the clock drifts noticeably. In the same spirit, a thermistor's resistance $R(T)$ is a steep exponential-like curve, but a controller working within a few degrees of a set point $T_0$ uses the linearised model $R \approx R(T_0) + R'(T_0)(T - T_0)$ — a single slope constant — and recalibrates the slope if the set point moves. Linearisation is how nonlinear hardware gets controlled by linear control theory.

**In ML/AI this is …** the justification for gradient descent. Minimising the first-order model plus a proximity penalty,

$$
\min_{\boldsymbol{\delta}} \; L(\boldsymbol{\theta}) + \nabla L^\top \boldsymbol{\delta} + \frac{1}{2\eta}\lVert \boldsymbol{\delta} \rVert^2 \quad\Longrightarrow\quad \boldsymbol{\delta}^* = -\eta\,\nabla L,
$$

is exactly one gradient step with learning rate $\eta$ (set the gradient of the objective in $\boldsymbol{\delta}$ to zero: $\nabla L + \boldsymbol{\delta}/\eta = 0$). Replacing the penalty $\tfrac{1}{2\eta} I$ by the true curvature $\tfrac{1}{2} H$ gives Newton's step $-H^{-1}\nabla L$; replacing it by a diagonal estimate of curvature from squared gradients gives Adam and RMSProp's per-coordinate step sizes. The second-order model also explains the sharpness story: plugging a step $\boldsymbol{\delta} = -\eta\nabla L$ into the quadratic model shows the loss *increases* along an eigendirection with curvature $\lambda$ whenever $\eta\lambda > 2$, so the largest stable learning rate is $2/\lambda_{\max}(H)$. The trust-region methods and proximal methods of Module 05 are all variations on "minimise a Taylor model, but don't trust it too far".

## 7. Matrix calculus essentials

**Intuition.** Differentiating expressions like $\mathbf{x}^\top A\mathbf{x}$ or $\lVert A\mathbf{x} - \mathbf{b} \rVert^2$ entry by entry is tedious and error-prone. A small table of identities, each derived once, lets you differentiate whole vector and matrix expressions in one line — and lets you read the derivations in papers. The two reliable habits are: (1) when in doubt, write out the sum and differentiate one coordinate; (2) check the *shape* — the gradient of a scalar with respect to a vector in $\mathbb{R}^n$ must again be in $\mathbb{R}^n$, and the gradient with respect to a matrix in $\mathbb{R}^{m \times n}$ must be $m \times n$.

**Layout convention.** For a scalar $f(\mathbf{x})$ we write $\nabla_{\mathbf{x}} f$ as a column vector of the same shape as $\mathbf{x}$ (the "denominator layout" used by every deep-learning framework: `w.grad` has the shape of `w`). For vector-valued $\mathbf{f}$ we use the Jacobian with rows indexed by outputs, as in Section 4 ("numerator layout"). The two conventions are transposes of each other; mixing them is the most common source of transposes that will not line up. Whatever you read, decide which layout the author uses before trusting a formula.

**The identities.** For $\mathbf{a}, \mathbf{x} \in \mathbb{R}^n$, $A \in \mathbb{R}^{n \times n}$ (not necessarily symmetric), $B \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$, and square invertible $X$:

| expression | gradient | remark |
|---|---|---|
| $\mathbf{a}^\top \mathbf{x}$ | $\nabla_{\mathbf{x}} = \mathbf{a}$ | linear ⇒ constant gradient |
| $\mathbf{x}^\top A \mathbf{x}$ | $\nabla_{\mathbf{x}} = (A + A^\top)\mathbf{x}$ | $= 2A\mathbf{x}$ if $A$ symmetric |
| $\lVert B\mathbf{x} - \mathbf{b} \rVert^2$ | $\nabla_{\mathbf{x}} = 2B^\top(B\mathbf{x} - \mathbf{b})$ | derived below |
| $B\mathbf{x}$ | $J = B$ | Jacobian of a linear map is the matrix |
| $\operatorname{tr}(X A)$ | $\nabla_X = A^\top$ | trace is the matrix inner product |
| $\log\det X$ | $\nabla_X = X^{-\top}$ | appears in Gaussian log-likelihoods |
| $\lVert X \rVert_F^2 = \operatorname{tr}(X^\top X)$ | $\nabla_X = 2X$ | weight decay / L2 regularisation |

*Derivation of the first two.* $\mathbf{a}^\top \mathbf{x} = \sum_j a_j x_j$, so $\partial/\partial x_i = a_i$: the gradient is $\mathbf{a}$. For the quadratic form, $\mathbf{x}^\top A\mathbf{x} = \sum_{j,k} x_j A_{jk} x_k$. Differentiating with respect to $x_i$ hits every term where $j = i$ and every term where $k = i$ (both, when $j = k = i$, by the product rule): $\partial/\partial x_i = \sum_k A_{ik}x_k + \sum_j x_j A_{ji} = (A\mathbf{x})_i + (A^\top\mathbf{x})_i$. Hence $\nabla_{\mathbf{x}} \mathbf{x}^\top A\mathbf{x} = (A + A^\top)\mathbf{x}$.

*Derivation of the least-squares gradient.* Expand: $\lVert B\mathbf{x} - \mathbf{b} \rVert^2 = (B\mathbf{x} - \mathbf{b})^\top(B\mathbf{x} - \mathbf{b}) = \mathbf{x}^\top B^\top B\mathbf{x} - 2\mathbf{b}^\top B\mathbf{x} + \mathbf{b}^\top\mathbf{b}$. Apply the identities term by term: the quadratic term has symmetric matrix $B^\top B$, giving $2B^\top B\mathbf{x}$; the linear term is $\mathbf{a}^\top\mathbf{x}$ with $\mathbf{a} = -2B^\top\mathbf{b}$, giving $-2B^\top\mathbf{b}$; the constant gives $\mathbf{0}$. Sum: $2B^\top B\mathbf{x} - 2B^\top\mathbf{b} = 2B^\top(B\mathbf{x} - \mathbf{b})$. Alternatively, by the chain rule: the residual $\mathbf{r} = B\mathbf{x} - \mathbf{b}$ has Jacobian $B$, the outer function $\lVert\mathbf{r}\rVert^2$ has gradient $2\mathbf{r}$, and the VJP rule of Section 4 gives $J^\top(2\mathbf{r}) = 2B^\top\mathbf{r}$. Setting the gradient to zero yields $B^\top B\mathbf{x} = B^\top \mathbf{b}$ — the **normal equations** of Module 02, now revealed as the first-order optimality condition of least squares. The Hessian is $2B^\top B$, positive semidefinite, so this critical point is a minimum (unique when $B$ has full column rank).

*Trace and log-det.* $\operatorname{tr}(XA) = \sum_{i,j} X_{ij}A_{ji}$, so $\partial/\partial X_{ij} = A_{ji}$, i.e. $\nabla_X = A^\top$. For $\log\det X$, the proof hinges on Jacobi's formula $\frac{d}{dt}\det X(t) = \det X \cdot \operatorname{tr}\bigl(X^{-1}\dot X\bigr)$; dividing by $\det X$ gives $d\log\det X = \operatorname{tr}(X^{-1}dX)$, and reading off the coefficient of $dX_{ij}$ via the trace identity gives $\nabla_X \log\det X = (X^{-1})^\top$.

**Worked example.** Fit a line $y \approx x_1 + x_2 t$ to the points $(t, y) = (0, 1), (1, 2), (2, 2)$. Then

$$
B = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 1 \\ 2 \\ 2 \end{pmatrix}, \quad \text{try } \mathbf{x}_0 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}: \; B\mathbf{x}_0 - \mathbf{b} = \begin{pmatrix} -1 \\ -1 \\ 0 \end{pmatrix}, \; \text{loss} = 2 .
$$

Gradient at $\mathbf{x}_0$: $2B^\top \mathbf{r} = 2\begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{pmatrix}\begin{pmatrix} -1 \\ -1 \\ 0 \end{pmatrix} = 2\begin{pmatrix} -2 \\ -1 \end{pmatrix} = \begin{pmatrix} -4 \\ -2 \end{pmatrix}$. Both components are negative, so increasing the intercept and the slope lowers the loss. Solving the normal equations, $B^\top B = \begin{pmatrix} 3 & 3 \\ 3 & 5 \end{pmatrix}$, $B^\top\mathbf{b} = \begin{pmatrix} 5 \\ 6 \end{pmatrix}$, gives $\mathbf{x}^* = (7/6,\; 1/2) \approx (1.167, 0.5)$ with loss $1/6$. You can verify the gradient at $\mathbf{x}^*$ is zero: $\mathbf{r}^* = (1/6, -1/3, 1/6)$ and $B^\top\mathbf{r}^* = (0, 0)$ — the residual is orthogonal to the columns of $B$, the projection picture from Module 02.

**Real-life example (finance: portfolio risk).** A portfolio with weights $\mathbf{w}$ in assets with covariance matrix $\Sigma$ has variance $\mathbf{w}^\top\Sigma\mathbf{w}$. With two assets of volatility 20 % and 30 % and correlation $1/6$, $\Sigma = \begin{pmatrix} 0.04 & 0.01 \\ 0.01 & 0.09 \end{pmatrix}$, and for $\mathbf{w} = (0.6, 0.4)$ the variance is $0.0336$ (volatility $18.3\%$). The gradient $\nabla_{\mathbf{w}}(\mathbf{w}^\top\Sigma\mathbf{w}) = 2\Sigma\mathbf{w} = (0.056, 0.084)$ is the *marginal risk* of each asset: shifting a unit of weight into asset 2 raises the variance 1.5 times faster than into asset 1. Risk managers report $w_i(\Sigma\mathbf{w})_i$ as each asset's *risk contribution*; here both are $0.0168$, so this happens to be a risk-parity portfolio. Markowitz optimisation is nothing but setting the gradient of $\mathbf{w}^\top\Sigma\mathbf{w} - \lambda\,\boldsymbol{\mu}^\top\mathbf{w}$ to zero under constraints (Module 05).

**In ML/AI this is …** the normal equations inside `sklearn.linear_model.LinearRegression` (and, with $2\lambda\mathbf{x}$ added to the gradient, inside `Ridge`); the gradient $2X^\top(X\mathbf{w} - \mathbf{y})/n$ that gradient descent on a linear layer with MSE loss computes; the $\nabla_X \log\det X = X^{-\top}$ that appears whenever a Gaussian's covariance is learned (Gaussian processes, Module 09; VAEs, Module 18; the log-determinant term of normalising flows); and the trace identity that makes derivations of the ELBO and of KL divergences between Gaussians (Module 14) go through in two lines.

## 8. Gradients of the common ML losses

**Intuition.** A loss function compares a prediction with a target. Its gradient with respect to the prediction says "in which direction, and how hard, should the prediction move"; the chain rule then pushes that signal back to the weights. For the losses that matter most, that first gradient is astonishingly simple — essentially *prediction minus target* — and this simplicity is not an accident but a consequence of pairing each output nonlinearity with its matching loss.

**Statements.** Throughout, $z$ is the model's raw output (the *logit*) and the weights enter through $z = \mathbf{w}^\top\mathbf{x}$, so by the chain rule $\nabla_{\mathbf{w}} L = \dfrac{\partial L}{\partial z}\,\mathbf{x}$ for any loss.

*Mean squared error.* For a single example, $L = \tfrac{1}{2}(z - y)^2$ and $\partial L/\partial z = z - y$. Over $n$ examples with $L = \tfrac{1}{n}\sum_i (\mathbf{w}^\top\mathbf{x}_i - y_i)^2$, the gradient is $\nabla_{\mathbf{w}} L = \tfrac{2}{n}X^\top(X\mathbf{w} - \mathbf{y})$, Section 7 again.

*Logistic loss (binary cross-entropy).* The model predicts $p = \sigma(z)$ for the probability that $y = 1$, and the loss is the negative log-likelihood

$$
L = -\bigl[y\ln p + (1 - y)\ln(1 - p)\bigr].
$$

Derivation of $\partial L / \partial z$. By the chain rule, $\dfrac{\partial L}{\partial z} = \dfrac{\partial L}{\partial p}\cdot\dfrac{dp}{dz}$. The first factor is $-\dfrac{y}{p} + \dfrac{1-y}{1-p} = \dfrac{-y(1-p) + (1-y)p}{p(1-p)} = \dfrac{p - y}{p(1-p)}$. The second is $\sigma'(z) = p(1 - p)$ from Section 2. The denominators cancel:

$$
\frac{\partial L}{\partial z} = \sigma(z) - y, \qquad \nabla_{\mathbf{w}} L = (\sigma(\mathbf{w}^\top\mathbf{x}) - y)\,\mathbf{x}.
$$

The gradient is the prediction error times the input: the Hessian is $\sigma(1-\sigma)\,\mathbf{x}\mathbf{x}^\top$, positive semidefinite, so logistic loss in $\mathbf{w}$ is convex (Module 05) and the second-order Newton step for it is the IRLS algorithm (Module 13).

*Softmax cross-entropy (preview).* For $K$ classes with logits $\mathbf{z}$, probabilities $p_k = e^{z_k}/\sum_j e^{z_j}$ and one-hot target $\mathbf{y}$, $L = -\sum_k y_k \ln p_k$ and $\partial L/\partial \mathbf{z} = \mathbf{p} - \mathbf{y}$. The proof pairs the softmax Jacobian $\partial p_k/\partial z_j = p_k(\delta_{kj} - p_j)$ with the loss gradient $-y_k/p_k$ and uses $\sum_k y_k = 1$; Module 16 does it in full. Note how the binary case is the special case $K = 2$.

*Hinge loss.* $L = \max(0,\, 1 - y z)$ with $y \in \{-1, +1\}$. For $yz < 1$ (margin violated) $\partial L/\partial z = -y$; for $yz > 1$ it is $0$; at $yz = 1$ any value in $[-y, 0]$ (or $[0, -y]$) is a subgradient, exactly as for ReLU — indeed the hinge loss *is* a ReLU applied to $1 - yz$. Gradient descent on hinge loss updates only on examples that are misclassified or inside the margin; those are the support vectors.

**Worked example.** One example $\mathbf{x} = (1, 2)$ with label $y = 1$, weights $\mathbf{w} = (0.5, -0.25)$. Then $z = 0.5 - 0.5 = 0$, $p = \sigma(0) = 0.5$, and $L = -\ln 0.5 = 0.6931$. The gradient is $(p - y)\,\mathbf{x} = (-0.5)(1, 2) = (-0.5, -1)$. A gradient step with $\eta = 0.1$ gives $\mathbf{w}' = (0.55, -0.15)$, $z' = 0.55 - 0.30 = 0.25$, $p' = \sigma(0.25) = 0.5622$, $L' = 0.5759$: the loss fell, as it must for a small enough step in the direction $-\nabla L$. Finite differences confirm the gradient: perturbing $w_1$ by $10^{-6}$ changes $L$ by $-0.5 \times 10^{-6}$. The lesson notebook repeats this check on a 50-point dataset.

**Real-life example (healthcare: calibrating a risk model).** A hospital's readmission model outputs $p = \sigma(z)$ for the probability that a discharged patient returns within 30 days. Training it with logistic loss, each patient contributes the gradient $(p - y)\mathbf{x}$: a patient predicted at $p = 0.9$ who was *not* readmitted ($y = 0$) pushes the weights down with force $0.9$ along their feature vector, while a correctly predicted low-risk patient ($p = 0.05$, $y = 0$) barely moves them. At the optimum the gradient sums to zero, $\sum_i (p_i - y_i)\mathbf{x}_i = \mathbf{0}$; since the first feature is the constant 1 (the intercept), this says $\sum_i p_i = \sum_i y_i$ — the predicted probabilities are *calibrated in aggregate*, averaging to the observed readmission rate. That property, a direct consequence of the $p - y$ gradient, is why clinicians can read logistic-regression outputs as probabilities, and why models trained with squared error on 0/1 labels or with hinge loss need a separate calibration step (Platt scaling, isotonic regression) before their scores can be used as risks.

**In ML/AI this is …** `nn.BCEWithLogitsLoss` and `nn.CrossEntropyLoss`, whose backward passes implement $\sigma(z) - y$ and $\mathbf{p} - \mathbf{y}$ directly rather than chaining separate sigmoid/softmax and log gradients — both for speed and for numerical stability, since computing $\ln\sigma(z)$ as $-\mathrm{softplus}(-z)$ avoids overflow (the "logits" versions exist for exactly this reason). The same $p - y$ structure appears in the policy-gradient update of Module 19 and in the score-matching objective of diffusion models in Module 18.

## Common pitfalls and misconceptions

1. **"A smaller finite-difference step is always more accurate."** Below $h \approx 10^{-6}$ (central) the result is dominated by round-off: the two function values agree in their leading digits, cancellation destroys the rest, and dividing by a tiny $h$ magnifies the garbage. *Fix:* use $h \approx 10^{-6}$ to $10^{-5}$ in double precision, scale $h$ with $|x|$, and look at the V-shaped error curve once so you believe it.

2. **"The gradient points toward the minimum."** It points in the direction of steepest *local increase*, perpendicular to the level set. In an elongated valley that direction is almost perpendicular to the valley floor, so gradient descent zig-zags; the minimum lies in a direction weighted by the inverse curvature, $-H^{-1}\nabla f$, which is what Newton's method follows. *Fix:* remember that $-\nabla f$ is the best direction for an infinitesimal step, not for a finite one.

3. **"Zero gradient means I have found a minimum."** A zero gradient is necessary, not sufficient: the point may be a maximum or, overwhelmingly often in high dimensions, a saddle. *Fix:* check the Hessian's eigenvalues (or, in practice, whether perturbing the parameters increases the loss in every direction tried).

4. **Treating $\dfrac{d}{dx}f(g(x))$ as $f'(x)\,g'(x)$.** The outer derivative must be evaluated at the inner value, $f'(g(x))$. With $f = e^u$ and $g = -x^2/2$ the correct answer is $-x\,e^{-x^2/2}$, not $-x\,e^{x}$. *Fix:* name the intermediate variable ($u = g(x)$), differentiate with respect to it, then substitute back — this is exactly what autodiff does when it stores forward-pass values.

5. **Forgetting the $A^\top$ in $\nabla \mathbf{x}^\top A\mathbf{x} = (A + A^\top)\mathbf{x}$.** Writing $2A\mathbf{x}$ is correct only for symmetric $A$. The error is invisible in most textbook examples (covariance matrices, $B^\top B$) and bites when $A$ is a generic weight matrix. *Fix:* when unsure, differentiate one coordinate of the double sum.

6. **Mixing gradient layouts and "fixing" shape errors with a transpose.** If a formula produces a $(1 \times n)$ row where you need an $(n \times 1)$ column, the cause is usually a numerator-versus-denominator layout mismatch, not a typo. Transposing until shapes agree gives correct shapes and sometimes a wrong answer (e.g. $B^\top \mathbf{r}$ versus $B\mathbf{r}$ when $B$ is square). *Fix:* fix the layout convention first; verify with a finite-difference check.

7. **"ReLU is not differentiable, so gradient descent should not work."** Non-differentiability on a measure-zero set (the single point $0$) is harmless: a subgradient exists everywhere, and the probability of landing exactly on the kink is zero. The same holds for the hinge loss and the $\ell_1$ penalty. *Fix:* use a subgradient; worry instead about *saturation* (sigmoid tails), which kills gradients on a large set.

8. **Trusting a Taylor model far from its base point.** The tangent to $E(D)$ at 20 mg predicted a gain of 12.5 for +10 mg; the true gain was 10. Linear models are local, and the error grows like the square of the distance. *Fix:* keep steps small relative to the curvature ($\eta < 2/\lambda_{\max}$), or use a trust region that shrinks when the model's prediction disagrees with the actual loss.

## Real-world case studies

### Case study 1 — Propagating radar measurement errors (Jacobians)

An air-traffic radar reports an aircraft at range $r = 2000$ m and bearing $\theta = 30°$ with independent measurement errors $\sigma_r = 10$ m and $\sigma_\theta = 0.5° = 0.008727$ rad. A collision-avoidance system needs the position uncertainty in Cartesian coordinates. The map is $\mathbf{P}(r, \theta) = (r\cos\theta, r\sin\theta)$, with Jacobian

$$
J = \begin{pmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{pmatrix}
= \begin{pmatrix} 0.866 & -1000 \\ 0.500 & 1732 \end{pmatrix}.
$$

Linearising, $\boldsymbol{\delta}_{xy} \approx J\boldsymbol{\delta}_{r\theta}$, so the covariance is $\Sigma_{xy} \approx J\,\mathrm{diag}(\sigma_r^2, \sigma_\theta^2)\,J^\top$. Working it through: $\sigma_\theta^2 = 7.615 \times 10^{-5}$, and

$$
\Sigma_{xy} = \begin{pmatrix} 0.866^2\cdot 100 + 1000^2 \cdot 7.615\times 10^{-5} & \cdots \\ \cdots & 0.5^2 \cdot 100 + 1732^2 \cdot 7.615\times 10^{-5} \end{pmatrix}
= \begin{pmatrix} 151.2 & -88.6 \\ -88.6 & 253.5 \end{pmatrix} \text{ m}^2.
$$

So $\sigma_x = \sqrt{151.2} = 12.3$ m and $\sigma_y = \sqrt{253.5} = 15.9$ m, with a strong negative correlation. More informative are the eigenvalues of $\Sigma_{xy}$: $100$ and $304.6$, whose square roots are $10$ m and $17.45$ m — the error ellipse has semi-axes exactly $\sigma_r$ (along the line of sight) and $r\sigma_\theta = 2000 \times 0.008727 = 17.45$ m (perpendicular to it). The Jacobian has rotated the radar's range–bearing ellipse into map coordinates without changing its shape, because the second column of $J$ has length $r$ and is orthogonal to the first. The design conclusion is quantitative: at this range, bearing error dominates by a factor of $1.75$; halving $\sigma_\theta$ would help far more than halving $\sigma_r$, and the benefit grows linearly with range. This $J\Sigma J^\top$ step is run at every tick of every Kalman filter (Module 09), and the same linearisation is why extended Kalman filters fail when the Jacobian changes a lot over one standard deviation of the state — i.e. when the second-order Taylor term is not negligible.

### Case study 2 — Why one learning rate cannot fit all directions (Hessians and Taylor)

Consider training a two-parameter model whose loss near the optimum is well approximated by its quadratic Taylor model with Hessian $H = \begin{pmatrix} 2 & 3 \\ 3 & 10 \end{pmatrix}$, the matrix from Section 5, eigenvalues $\lambda_1 = 1$ and $\lambda_2 = 11$. In the eigenbasis the loss is $\tfrac{1}{2}(\lambda_1 u_1^2 + \lambda_2 u_2^2)$, and a gradient-descent step with learning rate $\eta$ multiplies each coordinate by $(1 - \eta\lambda_i)$. Stability requires $|1 - \eta\lambda_i| < 1$ for both, i.e. $\eta < 2/\lambda_{\max} = 2/11 \approx 0.18$. Try $\eta = 0.19$: the sharp direction is multiplied by $1 - 2.09 = -1.09$ each step, so the error along it grows by 9 % per step while flipping sign — the loss *diverges* with oscillation, the classic symptom of a learning rate that is too high. Now take a safe $\eta = 1/6 \approx 0.167$, the value that balances the two directions ($|1 - \eta\lambda_1| = |1 - \eta\lambda_2| = 5/6$). The error shrinks by a factor $5/6$ per step, so after 50 steps only $(5/6)^{50} \approx 1.1 \times 10^{-4}$ of the initial error remains in each direction. Good — but only because the condition number $\kappa = \lambda_2/\lambda_1 = 11$ is modest. For $\kappa = 1000$, common in practice, the same analysis gives a best-case rate of $1 - 2/(\kappa + 1) \approx 0.998$ per step: 3,000 steps to reduce the error by a factor of 400. The flat direction, with curvature 1, makes essentially no progress at the small learning rate the sharp direction forces on you. This is the whole case for per-coordinate step sizes (Adagrad, RMSProp, Adam), for momentum, and for Newton's method, which rescales every direction by its own curvature so that $\eta_{\text{eff}} = 1/\lambda_i$ in each — and it is why batch normalisation and careful initialisation, which improve the conditioning of $H$, speed up training so dramatically (Modules 05 and 16).

### Case study 3 — Delta-gamma hedging of an option book (Taylor in finance)

An options desk holds a position whose value $V$ depends on the price $S$ of the underlying stock. The Greeks are Taylor coefficients: $\Delta = \partial V/\partial S$ and $\Gamma = \partial^2 V/\partial S^2$. The second-order Taylor model is $V(S + \delta) \approx V(S) + \Delta\,\delta + \tfrac{1}{2}\Gamma\,\delta^2$. Suppose a call option on a stock trading at $S = 100$ has $\Delta = 0.6$ and $\Gamma = 0.05$ (per share, per unit price). If the stock jumps by $\delta = 4$, the first-order (delta) model predicts a gain of $0.6 \times 4 = 2.4$ per share; adding the curvature term gives $2.4 + 0.5 \times 0.05 \times 16 = 2.8$, a 17 % correction. A desk that is "delta-hedged" holds $-\Delta$ shares against each option so that the first-order term cancels: the hedged position's P&L is then $\approx \tfrac{1}{2}\Gamma\delta^2 = 0.4$ per share for this move, positive in *both* directions, which is why long-option positions are said to "profit from volatility". A desk that also wants $\Gamma = 0$ must add other options, because a stock has zero curvature in its own price. The Taylor remainder here is not academic: in a 20 % crash ($\delta = -20$) the quadratic model itself is off, since $\Gamma$ changes with $S$ (the third derivative, "speed", is non-zero) — one of the ways static hedges failed in 1987 and 2008. The point for an ML practitioner: the gradient tells you the first-order effect of a parameter change, the Hessian the second, and when a step is large relative to how fast the Hessian itself changes, neither is to be trusted. Trust-region methods formalise exactly that caution.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Derivative as sensitivity | `loss.backward()`, `param.grad` | Training is "move each parameter against its sensitivity" |
| Finite differences | `gradcheck`, unit tests for custom layers | Independent ground truth for analytic gradients; the step-size trade-off explains why `float32` gradient checks fail |
| Chain rule | Backpropagation, autodiff (Module 06) | Every deep network is a composition; gradients are products of local derivatives |
| Activation derivatives $\sigma(1-\sigma)$, $1-\tanh^2$, ReLU mask | Backward pass of activation layers | Explains vanishing gradients ($\sigma' \le 1/4$) and the move to ReLU |
| Gradient and steepest ascent | Gradient descent, SGD, saliency maps | $-\nabla L$ is the locally best direction; gradient ⟂ level sets causes zig-zagging |
| Jacobian and VJPs | Reverse-mode autodiff, normalising flows, softmax Jacobian in attention | Backprop multiplies Jacobians; $\log\lvert\det J\rvert$ gives flow likelihoods |
| Hessian and its spectrum | Newton's method, learning-rate limit $2/\lambda_{\max}$, sharp vs flat minima, saddle escape | Curvature sets step sizes and convergence speed; saddles dominate high dimensions |
| Taylor expansion | Derivation of GD, Newton, Adam; trust regions; RoPE and small-angle tricks | Optimisers minimise a local linear or quadratic model |
| $\nabla \lVert A\mathbf{x} - \mathbf{b} \rVert^2 = 2A^\top(A\mathbf{x} - \mathbf{b})$ | Linear regression, ridge, least-squares layers | The normal equations are "gradient = 0" |
| $\nabla_X \log\det X$, trace identities | Gaussian likelihoods, GPs, VAEs, KL between Gaussians | Lets covariance parameters be learned by gradient methods |
| $\sigma(z) - y$ and $\mathbf{p} - \mathbf{y}$ | `BCEWithLogitsLoss`, `CrossEntropyLoss`, policy gradients | Cleanest possible training signal; gives aggregate calibration |
| Subgradients | ReLU, hinge loss, $\ell_1$/lasso | Non-smooth losses still train; kinks are a measure-zero problem |

## Cheat sheet

| Item | Formula / fact |
|---|---|
| Derivative | $f'(x) = \lim_{h\to 0}\dfrac{f(x+h)-f(x)}{h}$; $f(x+h) = f(x) + f'(x)h + o(h)$ |
| Central difference | $f'(x) \approx \dfrac{f(x+h)-f(x-h)}{2h}$, error $O(h^2)$; best $h \approx 10^{-6}$ to $10^{-5}$ (double precision) |
| Forward difference | $f'(x) \approx \dfrac{f(x+h)-f(x)}{h}$, error $O(h)$; best $h \approx 10^{-8}$ |
| Product / quotient | $(fg)' = f'g + fg'$; $(f/g)' = (f'g - fg')/g^2$ |
| Chain rule | $(f\circ g)' = f'(g(x))\,g'(x)$; matrix form $J_{g\circ f} = J_g J_f$ |
| Key derivatives | $(e^x)' = e^x$; $(\ln x)' = 1/x$; $(x^n)' = nx^{n-1}$; $\sigma' = \sigma(1-\sigma)$; $\tanh' = 1-\tanh^2$; $\mathrm{softplus}' = \sigma$; $\mathrm{ReLU}' = \mathbb{1}[x>0]$ |
| Gradient | $\nabla f = (\partial f/\partial x_1, \dots, \partial f/\partial x_n)^\top$; directional derivative $D_{\mathbf{u}}f = \nabla f^\top\mathbf{u}$ |
| Steepest ascent | $\max_{\lVert\mathbf{u}\rVert = 1} \nabla f^\top\mathbf{u} = \lVert\nabla f\rVert$ at $\mathbf{u} = \nabla f/\lVert\nabla f\rVert$; $\nabla f \perp$ level sets |
| Jacobian | $(J)_{ij} = \partial f_i/\partial x_j$, shape $m\times n$; $\mathbf{f}(\mathbf{x}+\boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + J\boldsymbol{\delta}$; VJP $= J^\top\mathbf{g}$ |
| Error propagation | $\Sigma_{\text{out}} \approx J\,\Sigma_{\text{in}}\,J^\top$ |
| Hessian | $(H)_{ij} = \partial^2 f/\partial x_i\partial x_j$, symmetric; curvature along $\mathbf{u}$ is $\mathbf{u}^\top H\mathbf{u}$ |
| Critical-point test | $\nabla f = 0$ and $H \succ 0$ ⇒ min; $H \prec 0$ ⇒ max; mixed eigenvalues ⇒ saddle; 2-D: $\det H < 0$ ⇒ saddle |
| Taylor (1-D) | $f(a+h) = \sum_{k=0}^{n} \dfrac{f^{(k)}(a)}{k!}h^k + \dfrac{f^{(n+1)}(\xi)}{(n+1)!}h^{n+1}$ |
| Taylor ($n$-D) | $f(\mathbf{x}+\boldsymbol{\delta}) \approx f + \nabla f^\top\boldsymbol{\delta} + \tfrac12\boldsymbol{\delta}^\top H\boldsymbol{\delta}$ |
| GD from Taylor | $\arg\min_{\boldsymbol{\delta}}\, \nabla f^\top\boldsymbol{\delta} + \tfrac{1}{2\eta}\lVert\boldsymbol{\delta}\rVert^2 = -\eta\nabla f$; stable iff $\eta < 2/\lambda_{\max}(H)$ |
| Matrix calculus | $\nabla_{\mathbf{x}}\mathbf{a}^\top\mathbf{x} = \mathbf{a}$; $\nabla_{\mathbf{x}}\mathbf{x}^\top A\mathbf{x} = (A+A^\top)\mathbf{x}$; $\nabla_{\mathbf{x}}\lVert B\mathbf{x}-\mathbf{b}\rVert^2 = 2B^\top(B\mathbf{x}-\mathbf{b})$ |
| Matrix calculus (matrices) | $\nabla_X \operatorname{tr}(XA) = A^\top$; $\nabla_X \log\det X = X^{-\top}$; $\nabla_X \lVert X\rVert_F^2 = 2X$ |
| Normal equations | $\nabla_{\mathbf{x}} \lVert B\mathbf{x}-\mathbf{b}\rVert^2 = 0 \iff B^\top B\mathbf{x} = B^\top\mathbf{b}$ |
| Loss gradients w.r.t. logit | MSE: $z - y$; logistic: $\sigma(z) - y$; softmax-CE: $\mathbf{p} - \mathbf{y}$; hinge: $-y\,\mathbb{1}[yz<1]$ |
| Weights via chain rule | $z = \mathbf{w}^\top\mathbf{x} \Rightarrow \nabla_{\mathbf{w}}L = (\partial L/\partial z)\,\mathbf{x}$ |
| Gradient check | relative error $\dfrac{\lVert \mathbf{g}_{\text{analytic}} - \mathbf{g}_{\text{numeric}}\rVert}{\lVert \mathbf{g}_{\text{analytic}}\rVert + \lVert \mathbf{g}_{\text{numeric}}\rVert}$; expect $< 10^{-6}$ with central differences in double precision |

## Self-assessment

1. **Why does the error of a central-difference derivative first fall and then rise as $h$ shrinks? Roughly where is the minimum?**
<details><summary>Answer</summary>

For large $h$ the dominant error is truncation, $\tfrac{h^2}{6}|f'''|$, which falls as $h^2$. For small $h$ the dominant error is round-off: $f(x+h)$ and $f(x-h)$ each carry an absolute error of about $\varepsilon |f|$ with $\varepsilon \approx 2.2 \times 10^{-16}$, and dividing their difference by $2h$ gives an error of order $\varepsilon|f|/h$, which grows as $h$ shrinks. Balancing $h^2 \sim \varepsilon/h$ gives $h \approx \varepsilon^{1/3} \approx 6 \times 10^{-6}$ and a best error around $10^{-10}$ to $10^{-11}$. For the forward difference the balance is $h \sim \varepsilon/h$, so $h \approx 10^{-8}$ and the best error is only about $10^{-8}$.
</details>

2. **State the chain rule for $h(x) = f(g(x))$ and explain why "slopes multiply" is the right picture.**
<details><summary>Answer</summary>

$h'(x) = f'(g(x))\,g'(x)$. A small change $dx$ produces a change $du = g'(x)\,dx$ in the intermediate variable $u = g(x)$, which in turn produces $dh = f'(u)\,du = f'(g(x))\,g'(x)\,dx$. Each stage scales the perturbation by its local slope, so the overall scaling is the product. In matrix form, for vector-valued maps, the Jacobians multiply, $J_{f\circ g} = J_f(g(x))\,J_g(x)$, with the outer function on the left. Backpropagation is this product evaluated from the loss backwards.
</details>

3. **Prove that the gradient is the direction of steepest ascent and that it is perpendicular to level sets.**
<details><summary>Answer</summary>

The directional derivative along a unit vector $\mathbf{u}$ is $D_{\mathbf{u}}f = \nabla f^\top\mathbf{u}$. By Cauchy–Schwarz, $\nabla f^\top\mathbf{u} \le \lVert\nabla f\rVert\lVert\mathbf{u}\rVert = \lVert\nabla f\rVert$, with equality iff $\mathbf{u}$ is a positive multiple of $\nabla f$. So the maximum slope $\lVert\nabla f\rVert$ is attained along $\nabla f/\lVert\nabla f\rVert$. Moving along a level set keeps $f$ constant, so the directional derivative along any tangent $\mathbf{t}$ to the level set is $0 = \nabla f^\top\mathbf{t}$ — the gradient is orthogonal to every tangent direction, hence perpendicular to the level set.
</details>

4. **A map $\mathbf{f}: \mathbb{R}^3 \to \mathbb{R}^5$ is followed by $\mathbf{g}: \mathbb{R}^5 \to \mathbb{R}^2$. What are the shapes of $J_{\mathbf{f}}$, $J_{\mathbf{g}}$ and $J_{\mathbf{g}\circ\mathbf{f}}$, and in which order do they multiply? If a loss $L$ depends on the output, what does the backward pass compute?**
<details><summary>Answer</summary>

$J_{\mathbf{f}}$ is $5 \times 3$, $J_{\mathbf{g}}$ is $2 \times 5$, and $J_{\mathbf{g}\circ\mathbf{f}} = J_{\mathbf{g}}\,J_{\mathbf{f}}$ is $2 \times 3$ (outer on the left; the inner dimension 5 must match). Given $\partial L/\partial \mathbf{y} \in \mathbb{R}^2$, the backward pass computes $\partial L/\partial\mathbf{x} = J_{\mathbf{f}}^\top J_{\mathbf{g}}^\top\,(\partial L/\partial\mathbf{y})$, a $3$-vector, as two successive vector–Jacobian products — never materialising the $2 \times 3$ Jacobian itself. The cost is roughly that of the forward pass, independent of the number of parameters, which is why reverse mode wins for scalar losses.
</details>

5. **The Hessian of a loss at a critical point has eigenvalues $(3, 0.01, -0.5)$. What kind of point is it, and what would gradient descent do nearby?**
<details><summary>Answer</summary>

Mixed signs make it a saddle point. Near it, gradient descent moves away along the eigenvector with eigenvalue $-0.5$ (the loss decreases in that direction) but extremely slowly, because the gradient component there is proportional to $0.5 \times$ the displacement, which starts near zero. Along the $0.01$ direction progress is also slow. Along the $3$ direction it converges quickly, and a learning rate above $2/3$ would make it diverge. The practical picture is a long plateau: the loss stagnates until the small negative-curvature component has grown enough to escape. Momentum and gradient noise speed this up.
</details>

6. **Write the second-order Taylor model of $L$ at $\boldsymbol{\theta}$ and derive the largest learning rate for which gradient descent is stable on a quadratic.**
<details><summary>Answer</summary>

$L(\boldsymbol{\theta} + \boldsymbol{\delta}) \approx L + \nabla L^\top\boldsymbol{\delta} + \tfrac12\boldsymbol{\delta}^\top H\boldsymbol{\delta}$. For an exactly quadratic loss with minimiser $\boldsymbol{\theta}^*$, $\nabla L = H(\boldsymbol{\theta} - \boldsymbol{\theta}^*)$, so a gradient step gives $\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* = (I - \eta H)(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)$. In the eigenbasis each component is multiplied by $1 - \eta\lambda_i$; convergence requires $|1 - \eta\lambda_i| < 1$ for all $i$, i.e. $0 < \eta < 2/\lambda_{\max}$. The slowest direction then contracts by $1 - \eta\lambda_{\min}$, so the convergence rate is governed by the condition number $\lambda_{\max}/\lambda_{\min}$.
</details>

7. **Derive $\nabla_{\mathbf{x}} \lVert A\mathbf{x} - \mathbf{b}\rVert^2$ and show that setting it to zero gives the normal equations.**
<details><summary>Answer</summary>

Expand: $\lVert A\mathbf{x} - \mathbf{b}\rVert^2 = \mathbf{x}^\top A^\top A\mathbf{x} - 2\mathbf{b}^\top A\mathbf{x} + \mathbf{b}^\top\mathbf{b}$. Using $\nabla\,\mathbf{x}^\top M\mathbf{x} = 2M\mathbf{x}$ for symmetric $M = A^\top A$ and $\nabla\,\mathbf{c}^\top\mathbf{x} = \mathbf{c}$ with $\mathbf{c} = -2A^\top\mathbf{b}$, the gradient is $2A^\top A\mathbf{x} - 2A^\top\mathbf{b} = 2A^\top(A\mathbf{x} - \mathbf{b})$. Setting it to zero: $A^\top A\mathbf{x} = A^\top\mathbf{b}$. Equivalently, the residual $A\mathbf{x} - \mathbf{b}$ must be orthogonal to the column space of $A$, which is the projection characterisation of least squares. The Hessian $2A^\top A$ is positive semidefinite, so this critical point is a global minimum.
</details>

8. **Show that the gradient of the binary cross-entropy with respect to the logit is $\sigma(z) - y$, and explain what this implies about the fitted model's average prediction.**
<details><summary>Answer</summary>

With $p = \sigma(z)$ and $L = -[y\ln p + (1-y)\ln(1-p)]$, $\partial L/\partial p = (p - y)/(p(1-p))$ and $dp/dz = p(1-p)$; the product is $p - y$. Hence $\nabla_{\mathbf{w}} L = (p - y)\mathbf{x}$ when $z = \mathbf{w}^\top\mathbf{x}$. At the optimum the total gradient over the training set vanishes, $\sum_i (p_i - y_i)\mathbf{x}_i = 0$. If one feature is the constant intercept, that component reads $\sum_i p_i = \sum_i y_i$: the mean predicted probability equals the empirical positive rate, so a logistic model fitted by maximum likelihood is calibrated in aggregate.
</details>

9. **A colleague's gradient checker reports a relative error of $3 \times 10^{-3}$ for a layer using `float32`, and $2 \times 10^{-9}$ in `float64`. Is the analytic gradient wrong?**
<details><summary>Answer</summary>

Almost certainly not. In single precision $\varepsilon \approx 1.2 \times 10^{-7}$, so the best achievable central-difference error is around $\varepsilon^{2/3} \approx 2 \times 10^{-5}$ in ideal conditions and often $10^{-3}$ or worse for real layers — the finite-difference reference itself is inaccurate, not the analytic gradient. The $2 \times 10^{-9}$ result in double precision is well within the expected $10^{-6}$ threshold and confirms the implementation. Gradient checks should always be run in `float64` with $h \approx 10^{-6}$; a genuine bug typically shows relative errors of order $10^{-1}$ to $1$.
</details>

10. **Why is $\nabla_{\mathbf{x}}\,\mathbf{x}^\top A\mathbf{x} = (A + A^\top)\mathbf{x}$ rather than $2A\mathbf{x}$, and when does the difference matter?**
<details><summary>Answer</summary>

Writing the quadratic form as $\sum_{j,k} x_j A_{jk}x_k$, the variable $x_i$ appears in the terms with $j = i$ (contributing $\sum_k A_{ik}x_k = (A\mathbf{x})_i$) and in the terms with $k = i$ (contributing $\sum_j A_{ji}x_j = (A^\top\mathbf{x})_i$). The sum is $((A + A^\top)\mathbf{x})_i$. When $A$ is symmetric the two contributions coincide and the gradient is $2A\mathbf{x}$; for a non-symmetric $A$ (a generic weight matrix, a non-symmetric bilinear attention score $\mathbf{q}^\top W\mathbf{k}$ differentiated with respect to a shared vector) only the symmetric part $\tfrac12(A + A^\top)$ influences the quadratic form, and $2A\mathbf{x}$ would be wrong.
</details>

## Further reading

- Gilbert Strang, *Calculus* (Wellesley-Cambridge Press; free online at MIT OpenCourseWare) — Chapters 2–4 and 13 for derivatives, the chain rule and partial derivatives, written with an eye to applications.
- Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020; free PDF at mml-book.github.io) — Chapter 5 "Vector Calculus" covers gradients, Jacobians, backpropagation and higher-order derivatives in exactly the notation used by ML papers.
- Kaare Brandt Petersen and Michael Syskind Pedersen, *The Matrix Cookbook* (2012, free PDF) — the standard reference table of matrix derivatives, including the trace and log-determinant identities stated here.
- Jonathan Richard Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain* (1994, free technical report, CMU) — the clearest treatment of quadratic forms, their gradients, and why conditioning governs convergence; perfect preparation for Module 05.
- Stephen Boyd and Lieven Vandenberghe, *Convex Optimization* (Cambridge University Press, 2004; free PDF) — Appendix A.4 for matrix calculus conventions and Chapter 9 for how gradient and Newton methods follow from Taylor models.
- Jorge Nocedal and Stephen J. Wright, *Numerical Optimization* (2nd edition, Springer, 2006) — Chapter 2 for Taylor's theorem and the second-order conditions, Chapter 8 for finite-difference derivative approximation and the step-size analysis.
- Andreas Griewank and Andrea Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd edition, SIAM, 2008) — the authoritative source on forward and reverse mode, Jacobian–vector products, and why reverse mode costs a constant factor of the forward pass.
- Yann N. Dauphin et al., "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization" (NeurIPS 2014) — the paper behind the "saddles, not minima" story for deep-learning loss landscapes.
- Grant Sanderson, *Essence of Calculus* (3Blue1Brown, free video series) — twelve short visual lectures; the chain-rule and Taylor-series episodes are the best intuition primers available.
- Christopher Olah, "Calculus on Computational Graphs: Backpropagation" (colah.github.io, 2015) — a short blog post that turns the chain rule into the forward/reverse-mode picture used in Module 06.

## What's in the notebooks

**`lesson.ipynb`** demonstrates each numbered concept in NumPy: the finite-difference error curve on a log-log plot (truncation versus round-off, optimal step for forward and central schemes); sigmoid, tanh, softplus and ReLU with their derivatives checked numerically; a contour plot of $f(x, y) = x^2 + 3xy$ with gradient arrows perpendicular to the level sets and a numerical test of the steepest-ascent theorem; the polar-to-Cartesian Jacobian versus finite differences, its linearisation error versus step size, and the Jacobian chain rule on a composed map; Hessian eigenvalues classifying the critical points of a 2-D function with a minimum, a maximum and saddles; Taylor polynomials of $e^x$ overlaid on the function with errors growing as $h^{k+1}$; the matrix-calculus identities verified on random matrices, ending in the normal equations; and the logistic-loss gradient on a synthetic dataset checked against finite differences, followed by a few gradient steps. "Try it yourself" prompts invite you to break things.

**`exercises.ipynb`** (with `solutions.ipynb`) tests the same ideas: a general central-difference gradient; the step-size study; activation derivatives; the analytic gradient and Hessian of 2-D functions with hidden-answer checks; the Jacobian of an $\mathbb{R}^2 \to \mathbb{R}^3$ map; Taylor coefficients; the least-squares gradient against finite differences and the normal equations; the logistic-loss gradient on a tiny dataset; conceptual questions on steepest ascent and on reading a gradient checker; and a mini-project in which you build a reusable `grad_check(f, grad_f, x)` and use it to validate — and debug — the gradient of a two-layer linear model's MSE loss.

Next module: [Module 05 — Optimization](../05-optimization/README.md), where the gradient, Hessian and Taylor model become algorithms. Then [Module 06 — Autodiff and backpropagation](../06-autodiff-and-backpropagation/README.md) turns the chain rule into code.
