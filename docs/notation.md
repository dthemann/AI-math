# Notation

The course uses one consistent notation throughout. When a module needs something extra, it defines it on first use.

## Scalars, vectors, matrices

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $a, b, \lambda, \alpha$ | scalars | lower-case italic |
| $\mathbf{x}, \mathbf{w}, \mathbf{v}$ | vectors (column vectors) | bold lower-case; $\mathbf{x} \in \mathbb{R}^d$ |
| $x_i$ | the $i$-th component of $\mathbf{x}$ | indices start at 1 in text, at 0 in code |
| $\mathbf{x}^{(i)}$ | the $i$-th data point (a vector) | superscript in parentheses |
| $\mathbf{A}, \mathbf{W}, \mathbf{\Sigma}$ | matrices | bold upper-case; $\mathbf{A} \in \mathbb{R}^{m \times n}$ has $m$ rows, $n$ columns |
| $A_{ij}$ | entry in row $i$, column $j$ | |
| $\mathbf{X}$ | the data (design) matrix | $n \times d$: one **row per sample**, one column per feature |
| $\mathbf{y}$ | the vector of targets | length $n$ |
| $\mathbf{I}, \mathbf{I}_n$ | identity matrix | |
| $\mathbf{0}, \mathbf{1}$ | all-zeros / all-ones vector | |
| $\mathbf{e}_i$ | $i$-th standard basis vector | |
| $\mathbf{A}^\top$ | transpose | $(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top$ |
| $\mathbf{A}^{-1}, \mathbf{A}^{+}$ | inverse, Moore–Penrose pseudo-inverse | |
| $\det \mathbf{A}$, $\operatorname{tr} \mathbf{A}$ | determinant, trace | |
| $\operatorname{rank} \mathbf{A}$ | rank | |
| $\operatorname{diag}(\mathbf{v})$ | diagonal matrix with $\mathbf{v}$ on the diagonal | |
| $\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^\top \mathbf{v} = \langle \mathbf{u}, \mathbf{v} \rangle$ | inner (dot) product | in code: `u @ v` |
| $\mathbf{u} \odot \mathbf{v}$ | element-wise (Hadamard) product | in code: `u * v` |
| $\Vert\mathbf{x}\Vert_2$, $\Vert\mathbf{x}\Vert_1$, $\Vert\mathbf{x}\Vert_\infty$ | Euclidean, Manhattan and max norms | $\Vert\mathbf{x}\Vert$ without subscript means $\Vert\mathbf{x}\Vert_2$ |
| $\Vert\mathbf{A}\Vert_F$ | Frobenius norm | $\sqrt{\sum_{ij} A_{ij}^2}$ |
| $\lambda_i$, $\sigma_i$ | eigenvalues, singular values | ordered $\sigma_1 \ge \sigma_2 \ge \dots$ |
| $\mathbf{A} \succeq 0$, $\mathbf{A} \succ 0$ | positive semi-definite, positive definite | |
| $\kappa(\mathbf{A})$ | condition number $\sigma_{\max}/\sigma_{\min}$ | |

## Calculus

| Symbol | Meaning |
|--------|---------|
| $f'(x)$, $\frac{df}{dx}$ | derivative of a scalar function |
| $\frac{\partial f}{\partial x_i}$ | partial derivative |
| $\nabla f(\mathbf{x})$ | gradient, a column vector in $\mathbb{R}^d$ |
| $\nabla_{\mathbf{w}} L$ | gradient of $L$ with respect to the parameters $\mathbf{w}$ |
| $\mathbf{J}_f(\mathbf{x})$ or $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$ | Jacobian of $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, an $m \times n$ matrix |
| $\mathbf{H}_f(\mathbf{x})$ or $\nabla^2 f$ | Hessian, a $d \times d$ symmetric matrix |
| $\sigma(z) = 1/(1+e^{-z})$ | logistic sigmoid |
| $\operatorname{softmax}(\mathbf{z})_i = e^{z_i}/\sum_j e^{z_j}$ | softmax |
| $\eta$ | learning rate / step size |
| $\theta$, $\boldsymbol{\theta}$ | generic (vector of) model parameters |
| $\arg\min_{\mathbf{x}} f(\mathbf{x})$ | the minimiser of $f$ |

## Probability

| Symbol | Meaning |
|--------|---------|
| $\Omega$, $A, B$ | sample space, events |
| $P(A)$, $P(A \mid B)$ | probability, conditional probability |
| $X, Y, Z$ | random variables (upper-case); realisations $x, y, z$ |
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | "$X$ is distributed as" — here normal with mean $\mu$, **variance** $\sigma^2$ |
| $\mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$ | multivariate normal with mean vector and covariance matrix |
| $\mathrm{Bernoulli}(p)$, $\mathrm{Bin}(n, p)$, $\mathrm{Poisson}(\lambda)$, $\mathrm{Exp}(\lambda)$, $\mathrm{Beta}(a, b)$, $\mathrm{Gamma}(\alpha, \beta)$, $\mathrm{Unif}(a, b)$ | named distributions; Exponential and Gamma use **rates** ($\mathrm{Exp}(\lambda)$ has mean $1/\lambda$; $\mathrm{Gamma}(\alpha, \beta)$ has shape $\alpha$, rate $\beta$, mean $\alpha/\beta$); full details in Module 08 |
| $p_X(x)$, $p(x)$ | probability mass function or density |
| $F_X(x) = P(X \le x)$ | cumulative distribution function |
| $\mathbb{E}[X]$, $\mu$ | expectation, mean |
| $\operatorname{Var}(X)$, $\sigma^2$ | variance |
| $\operatorname{Cov}(X, Y)$, $\rho$ | covariance, correlation |
| $\mathbf{\Sigma}$ | covariance matrix |
| $X \perp Y$ | independence |
| $\hat{\theta}$ | an estimator / estimate of $\theta$ |
| $\mathcal{L}(\theta)$, $\ell(\theta)$ | likelihood, log-likelihood |
| $p(\theta)$, $p(\theta \mid \mathcal{D})$ | prior, posterior |
| $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ | the data set |
| $H_0$, $H_1$ | null and alternative hypotheses |
| $\alpha$, $\beta$, $1-\beta$ | significance level, type-II error rate, power |
| $H(X)$, $H(p, q)$, $D_{\mathrm{KL}}(p \,\Vert\, q)$, $I(X; Y)$ | entropy, cross-entropy, KL divergence, mutual information |
| $\mathrm{i.i.d.}$ | independent and identically distributed |
| $\xrightarrow{p}$, $\xrightarrow{d}$ | convergence in probability, in distribution |

## Learning and reinforcement learning

| Symbol | Meaning |
|--------|---------|
| $h \in \mathcal{H}$ | a hypothesis (model) in a hypothesis class |
| $L(y, \hat{y})$ | loss function |
| $R(h)$, $\hat{R}(h)$ | true (expected) risk, empirical risk |
| $\mathrm{VC}(\mathcal{H})$ | VC dimension |
| $\lambda$ | regularisation strength |
| $s, a, r, s'$ | state, action, reward, next state |
| $\pi(a \mid s)$ | policy |
| $\gamma$ | discount factor |
| $V^\pi(s)$, $Q^\pi(s, a)$, $V^*$, $Q^*$ | state/action value functions, optimal values |
| $G_t = \sum_k \gamma^k r_{t+k+1}$ | return |

## Code conventions

* `rng = np.random.default_rng(seed)` is the only source of randomness; samples are drawn with `rng.normal`, `rng.uniform`, `rng.choice`, …
* Data matrices are `X.shape == (n_samples, n_features)`; weights are `w.shape == (n_features,)` or `W.shape == (n_in, n_out)`, so that a layer is `X @ W + b`.
* `@` is matrix multiplication, `*` is element-wise multiplication.
* Logarithms are natural (`np.log`) unless a formula explicitly uses $\log_2$ (bits).
