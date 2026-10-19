# Glossary

This glossary collects the terms the nineteen concept guides define, in alphabetical order. Each entry gives a short definition in the course's notation (bold lower-case for vectors, bold upper-case for matrices, a data matrix $\mathbf{X}$ of size $n \times d$ with one row per sample), one formula where it helps, and an arrow to the module whose concept guide introduces the term; when a term is central to several modules, the others are named in the text. Use it to look something up mid-module rather than to learn from scratch: the definitions assume the surrounding section. Symbols and spellings follow [notation.md](notation.md). Acronyms (ELBO, KL, MLE, SVD, …) are listed under their first letter; Greek-letter terms such as ε-greedy are spelled out.

## A

**A/A test** — An experiment in which both arms receive the identical treatment, run to check that randomisation, logging and analysis produce a false-positive rate equal to the nominal $\alpha$ before a real A/B test is trusted. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**A/B test** — A randomised controlled experiment comparing a control variant A with a treatment variant B on a pre-registered primary metric, analysed with a two-sample test (frequentist) or a posterior over the lift (Bayesian, Module 11). The sample size per arm for a two-sample z-test is $n = 2(z_{1-\alpha/2} + z_{1-\beta})^2\sigma^2/\delta^2$. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Activation function** — The element-wise non-linearity $\phi$ applied after a layer's affine map, $\mathbf{A} = \phi(\mathbf{Z})$; without one, stacked layers collapse into a single affine map. The course uses sigmoid, tanh, ReLU, leaky ReLU, softplus, SiLU and GELU; their derivatives are what backpropagation multiplies by (Module 04 derives them). → [Module 16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md)

**Adam and AdamW** — Adaptive optimisers that keep bias-corrected running means of the gradient ($\hat{\mathbf{m}}_t$) and of its square ($\hat{\mathbf{v}}_t$) and step $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\,\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$. AdamW applies the weight decay $\eta\lambda\boldsymbol{\theta}$ outside the adaptive scaling. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Adjacency matrix** — The square matrix of a graph with $A_{ij} = 1$ when node $i$ links to node $j$; $(\mathbf{A}^k)_{ij}$ counts $k$-step walks from $i$ to $j$. Normalising its rows gives the transition matrix of a random walk. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Advantage function** — $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$: how much better action $a$ is than the policy's average in state $s$. Its one-step estimate $r + \gamma\hat V(s') - \hat V(s)$ is the TD error, and it replaces the raw return in actor–critic and PPO updates to reduce variance. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**AIC and BIC** — Model-selection criteria that penalise the fit by the parameter count $p$: for Gaussian regression $\mathrm{AIC} = n\log(\mathrm{RSS}/n) + 2p$ and $\mathrm{BIC} = n\log(\mathrm{RSS}/n) + p\log n$. BIC approximates $-2\log p(\mathcal{D} \mid M)$, the marginal likelihood of Module 11. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Anisotropy (of embeddings)** — The empirical fact that learned embedding vectors share a common direction, so the mean cosine similarity of unrelated vectors is well above zero; centre the vectors before comparing them. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**Antithetic variates** — A variance-reduction trick for Monte Carlo: evaluate the integrand at $U$ and $1 - U$ so that negatively correlated pairs cancel part of each other's noise; the pair variance is $\tfrac12[\sigma_g^2 + \operatorname{Cov}(g(U), g(1-U))]$. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Arithmetic coding** — A compression scheme whose code length for a sequence is, within two bits, $-\log_2\prod_t p(x_t \mid x_{<t})$: the total log-loss of the probabilistic model that drives it. It is the precise sense in which a language model is a compressor. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Attention (scaled dot-product)** — $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\,\mathbf{V}$ with a row-wise softmax: every output is a data-dependent weighted average of the value vectors, with weights set by query–key similarity. The $\sqrt{d_k}$ keeps the score variance at 1 so the softmax does not saturate. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**Autocorrelation (of a Markov chain)** — The correlation $\rho_k = \operatorname{Corr}(x_t, x_{t+k})$ between MCMC draws $k$ steps apart. The integrated autocorrelation time $\tau = 1 + 2\sum_{k \ge 1}\rho_k$ says how many correlated draws are worth one independent sample. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Automatic differentiation (autodiff)** — Computing exact derivatives of a program by applying the chain rule to its computational graph one primitive operation at a time, at a cost of a small constant times the function itself; contrasted with symbolic and numerical differentiation. → [Module 06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md)

**Autoregressive model** — A generative model that factorises the joint by the chain rule, $p(\mathbf{x}) = \prod_i p(x_i \mid x_{<i})$, and learns each conditional; language models (Module 17) are the main example. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

## B

**Backpropagation** — Reverse-mode autodiff applied to a neural network: with $\boldsymbol{\delta}_l = \partial\mathcal{L}/\partial\mathbf{z}_l$, the recurrence $\boldsymbol{\delta}_l = (\mathbf{W}_{l+1}^\top\boldsymbol{\delta}_{l+1}) \odot f'(\mathbf{z}_l)$ delivers every weight gradient in one backward sweep. Module 16 writes it for a whole mini-batch at once. → [Module 06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md)

**Bandit (multi-armed)** — A sequential decision problem with $K$ arms of unknown mean reward and no state: each round you pull one arm and observe its reward. It isolates the exploration–exploitation trade-off; ε-greedy, UCB1 and Thompson sampling are the standard strategies. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Base-rate fallacy** — Confusing $P(\text{positive test} \mid \text{disease})$ with $P(\text{disease} \mid \text{positive test})$; when prevalence is low, even a sensitive and specific test produces mostly false positives. The fix is Bayes' theorem or a natural-frequency table. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Basis** — A set of linearly independent vectors whose span is the whole space, so that every vector has unique coordinates in it. In an orthonormal basis the coordinates are dot products, $c_i = \mathbf{x}\cdot\mathbf{q}_i$. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Batch normalisation** — A layer that standardises each feature over the mini-batch, $\hat{\mathbf{Z}} = (\mathbf{Z} - \boldsymbol{\mu})/\sqrt{\boldsymbol{\sigma}^2 + \epsilon}$, then rescales with learned $\gamma$ and $\beta$; at inference it uses running statistics. It keeps activations in a well-conditioned range and so counters vanishing and exploding gradients. → [Module 16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md)

**Bayes factor** — The ratio of marginal likelihoods of two models, $\mathrm{BF}_{12} = p(\mathcal{D} \mid M_1)/p(\mathcal{D} \mid M_2)$: how much the data shift the odds between them. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Bayes risk** — $R^* = \inf_f R(f)$, the lowest true risk achievable by any function; the irreducible noise floor that no learner can beat. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Bayes' theorem** — $P(H \mid E) = P(E \mid H)\,P(H)/P(E)$: posterior equals likelihood times prior over evidence. In odds form, posterior odds $=$ likelihood ratio $\times$ prior odds, which makes sequential updating a multiplication. Module 11 applies it to parameters. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Bellman equations** — The recursive consistency conditions for value functions: $V^\pi(s) = \sum_a \pi(a \mid s)\,[R(s,a) + \gamma\sum_{s'}P(s' \mid s, a)V^\pi(s')]$ (expectation) and $V^*(s) = \max_a[R(s,a) + \gamma\sum_{s'}P(s' \mid s,a)V^*(s')]$ (optimality). The expectation equation is linear: $\mathbf{V}^\pi = (\mathbf{I} - \gamma\mathbf{P}^\pi)^{-1}\mathbf{R}^\pi$. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Benjamini–Hochberg procedure** — Controls the false discovery rate at level $q$: sort the $m$ p-values, find the largest $k$ with $p_{(k)} \le kq/m$ and reject the $k$ smallest. Less conservative than Bonferroni when many effects are real. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Bernoulli distribution** — A single yes/no trial with success probability $p$: $P(X = 1) = p$, mean $p$, variance $p(1-p)$. Its negative log-likelihood is the binary cross-entropy loss. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Beta distribution** — A distribution over a probability, $f(\theta) \propto \theta^{a-1}(1-\theta)^{b-1}$ on $(0, 1)$, with mean $a/(a+b)$. It is the conjugate prior for Bernoulli and Binomial data: $a$ and $b$ act as prior successes and failures (Module 11). → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Bias (of an estimator)** — $\operatorname{Bias}(\hat\theta) = \mathbb{E}[\hat\theta] - \theta$, the systematic error averaged over samples; an estimator is unbiased when the bias is zero for every $\theta$. Unbiasedness is not the goal in itself: a biased estimator can have lower MSE. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Bias–variance decomposition** — For squared loss, $\mathbb{E}[(y - \hat f)^2] = \sigma^2 + (f - \bar f)^2 + \mathbb{E}[(\hat f - \bar f)^2]$: noise, squared bias and variance of the fitted model. Regularisation trades a little bias for a lot of variance; Module 15 adds capacity and double descent, Module 19 uses the vocabulary to compare Monte Carlo with TD learning. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Binomial distribution** — The number of successes in $n$ independent Bernoulli($p$) trials: $P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$, mean $np$, variance $np(1-p)$. For large $n$ and small $p$ it is approximately Poisson($np$). → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Bits and nats** — Units of information: $-\log_2 p$ is measured in bits, $-\ln p$ in nats, and $1\ \text{nat} = 1.4427$ bits. Every information-theoretic formula holds in either base; only the unit changes. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Bonferroni correction** — Test each of $m$ hypotheses at level $\alpha/m$ so that the family-wise error rate is at most $\alpha$ (by the union bound). Valid under any dependence but conservative; Holm's step-down version is uniformly more powerful. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Bootstrap** — Estimate the sampling distribution of any statistic by resampling the data with replacement $B$ times and recomputing the statistic: the standard deviation of the resampled values is the bootstrap standard error, and their $\alpha/2$ and $1 - \alpha/2$ quantiles give the percentile interval. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Bootstrapping (reinforcement learning)** — Updating a value estimate toward a target that itself contains an estimate, as TD(0) does with $r + \gamma V(s')$. It makes online learning possible and is one leg of the deadly triad. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Box–Muller transform** — Turns two independent $\mathrm{Unif}(0,1)$ draws into two independent standard normals: $Z_1 = \sqrt{-2\ln U_1}\cos 2\pi U_2$, $Z_2 = \sqrt{-2\ln U_1}\sin 2\pi U_2$. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Burn-in** — The initial segment of an MCMC chain, before it reaches the typical set of the target, that is discarded. Thinning (keeping every $k$-th draw) saves storage but never improves an estimate. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

## C

**Calibration** — A probabilistic classifier is calibrated when $P(y = 1 \mid \hat p = p) = p$: among all cases given probability 0.7, seventy per cent are positive. The Brier score $\frac1n\sum(\hat p_i - y_i)^2$ decomposes into calibration and refinement; temperature scaling is the usual fix. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Categorical and Multinomial distributions** — Categorical($\boldsymbol{\pi}$) is one draw from $K$ classes with $P(\text{class } k) = \pi_k$, written with a one-hot $\mathbf{y}$ as $\prod_k \pi_k^{y_k}$; Multinomial($n, \boldsymbol{\pi}$) counts the classes in $n$ draws. The categorical negative log-likelihood is the cross-entropy loss. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Cauchy–Schwarz inequality** — $\lvert\mathbf{u}\cdot\mathbf{v}\rvert \le \lVert\mathbf{u}\rVert\,\lVert\mathbf{v}\rVert$, with equality iff the vectors are parallel. It is why cosine similarity lies in $[-1, 1]$, and it implies the triangle inequality. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Causal mask** — Setting attention scores to $-\infty$ for positions $j > i$ so that each token attends only to itself and earlier tokens; the attention matrix becomes lower-triangular and one parallel pass yields $n$ next-token predictions (teacher forcing). → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**Central limit theorem (CLT)** — For i.i.d. $X_i$ with mean $\mu$ and finite variance $\sigma^2$, $\sqrt n(\bar X_n - \mu)/\sigma \xrightarrow{d} \mathcal{N}(0, 1)$: averages are approximately normal whatever the summands' distribution. The Berry–Esseen bound says the error shrinks like the skewness over $\sqrt n$. Module 10 builds confidence intervals on it. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Chain rule** — $(f \circ g)'(x) = f'(g(x))\,g'(x)$; for compositions of vector maps the Jacobians multiply, $\mathbf{J}_{g\circ f} = \mathbf{J}_g\mathbf{J}_f$. It is the single rule behind backpropagation (Module 06). → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**Change of basis** — Writing the same vector or map in new coordinates: $\mathbf{x} = \mathbf{P}\mathbf{c}$ and $\mathbf{A}' = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$ (similarity), which preserves determinant, trace, rank and eigenvalues. Diagonalisation and PCA (Module 03) are changes of basis that make a map or a dataset simple. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Change of variables** — If $Y = g(X)$ with $g$ monotone, $f_Y(y) = f_X(g^{-1}(y))\,\lvert (g^{-1})'(y)\rvert$; in several dimensions the derivative becomes $\lvert\det\mathbf{J}\rvert$. Normalising flows (Module 18) are built on it. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Chebyshev's inequality** — $P(\lvert X - \mu\rvert \ge k\sigma) \le 1/k^2$ for any distribution with finite variance. It proves the weak law of large numbers (Module 08) and is the loosest of the concentration bounds of Module 15. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Chi-square distribution** — The distribution of a sum of $k$ squared standard normals, $\chi^2_k = \mathrm{Gamma}(k/2, 1/2)$, with mean $k$ and variance $2k$. The squared Mahalanobis distance of a $d$-dimensional Gaussian vector is $\chi^2_d$ (Module 09). → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Chi-square test** — Pearson's test of independence in an $r \times c$ contingency table: $\chi^2 = \sum(O - E)^2/E$ with expected counts $E_{ij} = R_iC_j/N$ and $(r-1)(c-1)$ degrees of freedom. For a $2 \times 2$ table it equals the squared two-proportion z-statistic; use Fisher's exact test when expected counts are small. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Cholesky factorisation** — Writing a symmetric positive definite matrix as $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ with $\mathbf{L}$ lower-triangular. It is how you sample $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$: $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$ with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Column space** — $\mathcal{C}(\mathbf{A})$, the set of all outputs $\mathbf{A}\mathbf{x}$, i.e. the span of the columns; its dimension is the rank. $\mathbf{A}\mathbf{x} = \mathbf{b}$ is solvable iff $\mathbf{b} \in \mathcal{C}(\mathbf{A})$, and least squares projects $\mathbf{b}$ onto it. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Computational graph** — A directed acyclic graph whose nodes are the intermediate values of a program and whose edges are the primitive operations that produced them. Forward evaluation follows a topological order; autodiff propagates derivatives along the same edges. → [Module 06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md)

**Concentration inequality** — A bound on the probability that a random quantity deviates from its mean, such as Markov's, Chebyshev's or Hoeffding's inequality. They replace CLT approximations with finite-sample guarantees and are the raw material of generalisation bounds. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Condition number** — $\kappa(\mathbf{A}) = \sigma_{\max}/\sigma_{\min}$, the factor by which a linear solve can amplify relative error in the data. For a quadratic loss, gradient descent needs about $\kappa$ iterations per digit of accuracy (Module 05), which is why feature scaling matters; $\kappa(\mathbf{A}^\top\mathbf{A}) = \kappa(\mathbf{A})^2$. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**Conditional expectation** — $\mathbb{E}[Y \mid X]$, the mean of $Y$ given $X$, itself a random variable. It is the best mean-squared-error predictor of $Y$ from $X$ and hence the target of regression (Module 13); the tower property says $\mathbb{E}[\mathbb{E}[Y \mid X]] = \mathbb{E}[Y]$. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Conditional probability** — $P(A \mid B) = P(A \cap B)/P(B)$, the probability of $A$ once $B$ is known. It gives the multiplication rule $P(A \cap B) = P(A \mid B)P(B)$ and, chained, the factorisation of any joint distribution. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Confidence interval** — An interval $[L, U]$ computed from the data such that $P_\theta(L \le \theta \le U) \ge 1 - \alpha$ for every $\theta$. Coverage is a property of the procedure, not of one realised interval; for a mean, $\bar x \pm t_{n-1,\alpha/2}\,s/\sqrt n$. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Confounder** — A variable that influences both the treatment and the outcome, so that an observed association is not a causal effect; Simpson's paradox is the extreme case. Randomisation (Module 12) breaks the link between treatment and every confounder at once. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Conjugate prior** — A prior family that the likelihood maps back into itself, so the posterior has a closed form: Beta–Binomial, Gamma–Poisson and Normal–Normal. The prior parameters behave like pseudo-observations. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Consistency** — An estimator is consistent when $\hat\theta_n \xrightarrow{p} \theta$ as $n \to \infty$; a sufficient condition is that both its bias and its variance go to zero. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Contraction mapping** — An operator $T$ with $\lVert TU - TV\rVert \le \gamma\lVert U - V\rVert$ for some $\gamma < 1$. The Bellman operator is a $\gamma$-contraction in the max norm, so it has a unique fixed point and value iteration converges geometrically. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Convexity** — A function is convex when every chord lies above its graph, $f(\alpha\mathbf{x} + (1-\alpha)\mathbf{y}) \le \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})$, equivalently when $\nabla^2 f \succeq 0$. Convex problems have no spurious local minima, so any stationary point is a global minimiser. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Convolution** — A linear map whose matrix is Toeplitz: the same small kernel is applied at every position, so a layer has $m$ parameters instead of $n(n-m+1)$, and its backward pass is a convolution with the flipped kernel. → [Module 16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md)

**Coordinate descent** — Minimising over one coordinate at a time while holding the others fixed. For the lasso each update is a closed-form soft threshold, $\beta_j \leftarrow S_\lambda(\mathbf{x}_j^\top\mathbf{r}^{(j)})/\lVert\mathbf{x}_j\rVert^2$. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Copula** — A joint CDF on the unit square that couples arbitrary marginals, $F(x, y) = C(F_X(x), F_Y(y))$; it separates dependence from the marginal shapes. The Gaussian copula has zero tail dependence, which is why it understates joint extremes. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Correlation** — $\rho = \operatorname{Cov}(X, Y)/(\sigma_X\sigma_Y) \in [-1, 1]$, the covariance of the standardised variables. It measures linear dependence only: $\rho = 0$ does not imply independence, and $\lvert\rho\rvert = 1$ iff $Y = aX + b$. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Cosine similarity** — $\cos\theta = \mathbf{u}\cdot\mathbf{v}/(\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert)$, the dot product of the unit vectors. It is scale-invariant and the standard similarity for embeddings; for random directions in $\mathbb{R}^d$ it has mean 0 and standard deviation $1/\sqrt d$. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Covariance and covariance matrix** — $\operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$ measures how two variables move together; for a random vector, $\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top]$ is symmetric positive semidefinite and $\operatorname{Var}(\mathbf{a}^\top\mathbf{X}) = \mathbf{a}^\top\boldsymbol{\Sigma}\mathbf{a}$. The sample version $\mathbf{X}_c^\top\mathbf{X}_c/(n-1)$ is the matrix PCA diagonalises (Module 03). → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Cramér–Rao bound** — No unbiased estimator has variance below $1/I_n(\theta)$, the inverse Fisher information; the MLE attains this bound asymptotically. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Credible interval** — An interval holding $1 - \alpha$ of the posterior probability mass, $P(\theta \in [L, U] \mid \mathcal{D}) = 1 - \alpha$. Unlike a confidence interval it is a statement about $\theta$ given this particular data set. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Cross-entropy** — $H(p, q) = -\sum_x p(x)\log q(x) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)$: the expected code length when data from $p$ are encoded with a model $q$. As a loss it is the negative log-likelihood of a categorical model (Module 11); its gradient with respect to the logits is $\mathbf{p} - \mathbf{y}$ (Modules 04 and 16). → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Cross-validation** — Estimating out-of-sample error by splitting the data into $k$ folds and averaging the loss of each fold's model, fitted on the other $k - 1$ folds. Leave-one-out has the shortcut $\frac1n\sum_i\big((y_i - \hat y_i)/(1 - h_{ii})\big)^2$ for any linear smoother; choosing hyper-parameters inside the loop requires nested cross-validation (Module 15). → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Curse of dimensionality** — In high dimension random vectors are nearly orthogonal, pairwise distances concentrate (the nearest point is almost as far as the farthest), and almost all of a cube's volume sits in its corners. It is why nearest-neighbour methods degrade, why attention scores are rescaled by $\sqrt{d_k}$ (Module 17), and why rejection sampling fails in high dimension (Module 18). → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

## D

**Data-generating process** — The unknown mechanism that produces the observations; what "the population" means when there is no finite list of all possible images or sentences. Statistics asks which process produced the data you have. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Deadly triad** — Bootstrapping, function approximation and off-policy data: any two are safe, all three together can make value estimates diverge. Target networks, replay buffers and small learning rates are the engineering response in deep RL. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Design matrix** — The $n \times d$ data matrix $\mathbf{X}$ with one row per sample and one column per feature; a layer computes $\mathbf{X}\mathbf{W} + \mathbf{b}$ and least squares solves $\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Detailed balance** — The condition $\pi_i P_{ij} = \pi_j P_{ji}$ for all pairs of states. It implies that $\boldsymbol{\pi}$ is stationary for the chain (the converse is false) and is how Metropolis–Hastings is proved correct. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Determinant** — $\det\mathbf{A}$, the signed factor by which the linear map $\mathbf{A}$ scales volume; $\det\mathbf{A} = 0$ exactly when the map squashes some direction to zero (singular). For $2 \times 2$: $ad - bc$; $\det(\mathbf{A}\mathbf{B}) = \det\mathbf{A}\det\mathbf{B}$ and $\det\mathbf{A} = \prod_i\lambda_i$. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Deviance** — Twice the log-likelihood gap between the saturated and the fitted GLM; for Poisson regression $D = 2\sum_i[y_i\log(y_i/\hat\mu_i) - (y_i - \hat\mu_i)]$. It generalises the residual sum of squares to non-Gaussian models. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Diagonalisation** — Writing $\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}$ with the eigenvectors in $\mathbf{P}$ and eigenvalues in the diagonal $\mathbf{D}$, so that $\mathbf{A}^k = \mathbf{P}\mathbf{D}^k\mathbf{P}^{-1}$ and long-run behaviour is read off the eigenvalues. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**Differential entropy** — $h(X) = -\int f\log f$, the continuous analogue of entropy: $\tfrac12\log(2\pi e\sigma^2)$ for a Gaussian. Unlike discrete entropy it can be negative and depends on the units of $X$. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Diffusion model** — A generative model that destroys data with Gaussian noise, $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0, (1 - \bar\alpha_t)\mathbf{I})$, trains a network to predict the noise, $\mathbb{E}\lVert\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)\rVert^2$, and samples by reversing the process step by step. The noise prediction is a rescaled score, which connects sampling to Langevin dynamics. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Directional derivative** — $D_{\mathbf{u}}f = \nabla f^\top\mathbf{u}$, the rate of change of $f$ along the unit direction $\mathbf{u}$. It is largest when $\mathbf{u}$ points along the gradient, which is why the gradient is the direction of steepest ascent. → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**Discount factor** — $\gamma \in [0, 1)$ in the return $G_t = \sum_k\gamma^k r_{t+k+1}$; it keeps returns finite and sets an effective horizon of $1/(1 - \gamma)$ steps. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Distribution shift** — The deployment distribution differs from the training one: covariate shift changes $P(\mathbf{x})$, label shift changes $P(y)$, concept shift changes $P(y \mid \mathbf{x})$. Every generalisation guarantee assumes the two coincide. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Dot product** — See *Inner product*.

**Double descent** — Test error that falls, rises to a peak near the interpolation threshold $p \approx n$, and then falls again as capacity grows further; it contradicts the classical U-curve and is reproduced in the course with random features and minimum-norm least squares. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**DPO (direct preference optimisation)** — Fits a policy to preference pairs $(y_w, y_l)$ with the loss $-\log\sigma\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}\big)$, which follows in closed form from the KL-regularised RLHF objective without training a separate reward model. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Dropout** — Regularisation that zeroes each activation independently with probability $1 - q$ during training; inverted dropout divides the survivors by $q$ so that the expected activation is unchanged and nothing is rescaled at test time. → [Module 16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md)

**Dual numbers** — Pairs $(a, \dot a)$ with the rule $(a, \dot a)(b, \dot b) = (ab, a\dot b + \dot a b)$ and $g(a, \dot a) = (g(a), g'(a)\dot a)$; arithmetic on them is forward-mode autodiff. → [Module 06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md)

## E

**Early stopping** — Halting training when the validation loss stops improving. For least squares, $t$ steps of gradient descent act like ridge regression with $\lambda \approx 1/(\eta t)$, so the stopping time is a capacity control (Module 15). → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Eckart–Young theorem** — The best rank-$k$ approximation of $\mathbf{A}$ in the Frobenius or spectral norm is the truncated SVD $\mathbf{A}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^\top$, with error $\sqrt{\sum_{i>k}\sigma_i^2}$. It underlies PCA and the low-rank view of LoRA (Module 17). → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**Effective degrees of freedom** — For a linear smoother $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$, $\operatorname{tr}\mathbf{H}$; for ridge regression $\operatorname{df}(\lambda) = \sum_j d_j^2/(d_j^2 + \lambda)$, which falls from $p$ to 0 as $\lambda$ grows. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Effective sample size (ESS)** — For importance sampling, $(\sum w_i)^2/\sum w_i^2$: how many equally weighted samples the weighted set is worth. For an MCMC chain, $n/\tau$ with $\tau$ the integrated autocorrelation time. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Eigenvalue and eigenvector** — A non-zero $\mathbf{v}$ with $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$: a direction the map only stretches, by the factor $\lambda$. The eigenvalues solve $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$; their sum is the trace and their product the determinant. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**Elastic net** — The penalty $\lambda_1\lVert\boldsymbol{\beta}\rVert_1 + \lambda_2\lVert\boldsymbol{\beta}\rVert_2^2$, combining the sparsity of the lasso with the stability of ridge under correlated features. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**ELBO (evidence lower bound)** — $\log p(\mathbf{x}) \ge \mathbb{E}_q[\log p(\mathbf{x} \mid \mathbf{z})] - D_{\mathrm{KL}}(q(\mathbf{z}) \,\|\, p(\mathbf{z}))$, a reconstruction term minus a KL term; the gap is exactly $D_{\mathrm{KL}}(q(\mathbf{z}) \,\|\, p(\mathbf{z} \mid \mathbf{x}))$, so maximising the bound over $q$ approximates the posterior. Derived from Jensen's inequality here, it becomes the VAE objective in Module 18. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Embedding** — A learned vector representation of a discrete item (token, user, product), stored as a row of a matrix $\mathbf{E}$ and looked up as a one-hot product $\mathbf{e}_t^\top\mathbf{E}$; similarity between items is then a dot product or cosine. Module 17 ties the output "unembedding" to the same matrix. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Empirical Bayes** — Estimating the prior's hyper-parameters from the data themselves (for example by matching the moments of many Beta–Binomial units) and then shrinking each unit's estimate toward the fitted prior mean. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Empirical risk minimisation (ERM)** — Choosing the hypothesis that minimises the average loss on the training sample, $\hat R_S(h) = \frac1n\sum_i\ell(h(\mathbf{x}_i), y_i)$, as a proxy for the true risk. Module 05 states it as the optimisation problem of ML; Module 15 asks when the proxy can be trusted. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Energy-based model (EBM)** — $p(\mathbf{x}) = e^{-E(\mathbf{x})}/Z$ with an unnormalised energy $E$; the gradient of the log-likelihood is $-\nabla_\theta E(\mathbf{x}) + \mathbb{E}_p[\nabla_\theta E]$, whose second term needs samples from the model. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Entropy** — $H(X) = -\sum_x p(x)\log p(x)$, the average surprise of a distribution and the minimum average code length per symbol. It is between 0 and $\log n$, maximal for the uniform distribution and concave in $p$. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**ε-greedy** — With probability $1 - \varepsilon$ pull the arm (or action) with the highest estimated value, otherwise a uniformly random one. With a constant $\varepsilon$ it keeps exploring bad arms forever, so its regret grows linearly. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Equivalence test (TOST)** — Two one-sided tests that establish a difference lies within a pre-specified margin $\pm\Delta$: equivalence at level $\alpha$ holds iff the $(1 - 2\alpha)$ confidence interval lies inside $(-\Delta, \Delta)$; non-inferiority uses one side only. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Estimator** — A rule $\hat\theta = T(X_1, \dots, X_n)$ that turns a sample into a guess for a parameter; it is a random variable with its own distribution, bias and variance, and its realised value is the estimate. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Evidence (marginal likelihood)** — $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\,p(\theta)\,d\theta$, the normaliser in Bayes' theorem for parameters and the quantity Bayes factors compare. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Excess risk decomposition** — $R(\tilde h) - R^*$ splits into approximation error (best in class versus Bayes), estimation error (ERM solution versus best in class) and optimisation error (what the optimiser returned versus the ERM solution). → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Expectation** — $\mathbb{E}[X] = \sum_x x\,p(x)$ or $\int x\,f(x)\,dx$, the probability-weighted average. It is linear, $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$, even for dependent variables. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Exponential distribution** — The waiting time at constant rate $\lambda$: density $\lambda e^{-\lambda t}$, mean $1/\lambda$, survival $P(T > t) = e^{-\lambda t}$. It is memoryless and gives the gaps of a Poisson process. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Exponential family** — Distributions of the form $p(x) \propto \exp(\sum_k\lambda_k f_k(x))$ with sufficient statistics $f_k$ and natural parameters $\lambda_k$; they are the maximum-entropy distributions under moment constraints and the response distributions of GLMs (Module 13). → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

## F

**False discovery rate (FDR)** — The expected fraction of rejected hypotheses that are false rejections; controlled at level $q$ by the Benjamini–Hochberg procedure. Contrast with the family-wise error rate. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Family-wise error rate (FWER)** — The probability of at least one false rejection among $m$ tests, $1 - (1 - \alpha)^m$ when they are independent; Bonferroni and Holm hold it below $\alpha$. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**FID (Fréchet inception distance)** — The Fréchet distance between Gaussians fitted to real and generated feature vectors, $\lVert\boldsymbol{\mu}_r - \boldsymbol{\mu}_g\rVert^2 + \operatorname{tr}\big(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r\boldsymbol{\Sigma}_g)^{1/2}\big)$; it measures sample quality and diversity, not likelihood. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Finite differences** — Numerical derivatives: forward $\frac{f(x+h) - f(x)}{h}$ with error $O(h)$ and central $\frac{f(x+h) - f(x-h)}{2h}$ with error $O(h^2)$. The best $h$ balances truncation against round-off, about $10^{-5}$ for central differences in double precision. → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**Fisher information** — $i(\theta) = \operatorname{Var}[\partial_\theta\log p] = -\mathbb{E}[\partial^2_\theta\log p]$, the expected curvature of the log-likelihood; for $n$ observations $I_n = n\,i$. The MLE's standard error is approximately $1/\sqrt{I_n(\hat\theta)}$. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Fisher's exact test** — A test of association in a $2 \times 2$ table that conditions on the margins and sums the hypergeometric probabilities of tables at least as extreme; use it when expected counts are too small for the chi-square test. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Forward mode and reverse mode** — The two sweeps of autodiff: forward mode propagates tangents $\dot v_i = \partial v_i/\partial x_j$ from one input, reverse mode propagates adjoints $\bar v_k = \partial L/\partial v_k$ from one output. A scalar loss with millions of parameters needs one reverse sweep, which is why backpropagation is reverse mode. → [Module 06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md)

**Frobenius norm** — $\lVert\mathbf{A}\rVert_F = \sqrt{\sum_{ij}A_{ij}^2} = \sqrt{\sum_i\sigma_i^2}$, the Euclidean norm of a matrix read as one long vector. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

## G

**Gamma distribution** — $f(x) \propto x^{\alpha-1}e^{-\beta x}$ on $x > 0$ with mean $\alpha/\beta$: the sum of $\alpha$ Exponential($\beta$) waiting times, and the conjugate prior for a Poisson rate (Module 11). The chi-square distribution is a special case. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**GAN (generative adversarial network)** — A generator trained against a discriminator; the optimal discriminator is $D^* = p_d/(p_d + p_g)$ and the generator then minimises $2\,\mathrm{JSD}(p_d \,\|\, p_g) - \log 4$, the Jensen–Shannon divergence to the data distribution. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Gauss–Markov theorem** — Under uncorrelated, equal-variance errors, OLS is the best linear unbiased estimator (BLUE): no other linear unbiased estimator has smaller variance. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Gauss–Newton and Levenberg–Marquardt** — Second-order methods for least-squares losses that replace the Hessian by $\mathbf{J}^\top\mathbf{J}$ and solve $(\mathbf{J}^\top\mathbf{J} + \tau\mathbf{I})\,\mathbf{p} = -\mathbf{J}^\top\mathbf{r}$; the damping $\tau$ interpolates between Newton and gradient descent. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Gaussian conditioning** — For a jointly Gaussian vector, the conditional of $\mathbf{x}_1$ given $\mathbf{x}_2$ is Gaussian with mean $\boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ and covariance $\boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$ (the Schur complement). The Kalman filter and Gaussian-process regression are two instances of this one formula. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Gaussian process (GP) regression** — Treats the unknown function as a Gaussian vector with covariance given by a kernel such as the RBF $k(x, x') = \sigma_f^2\exp(-(x - x')^2/2\ell^2)$ and predicts by Gaussian conditioning: mean $\mathbf{k}_*^\top(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{y}$, variance $k_{**} - \mathbf{k}_*^\top(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{k}_*$. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Generalisation gap** — $R(h) - \hat R_S(h)$, the difference between a model's true risk and its training error. Bounding it uniformly over the hypothesis class is the subject of learning theory. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Generalised linear model (GLM)** — A model in which a link function of the conditional mean is linear in the features, $g(\mu) = \boldsymbol{\beta}^\top\mathbf{x}$, with an exponential-family response: identity link for Gaussian, logit for Bernoulli, log for Poisson. Fitted by Newton's method / IRLS. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Geometric distribution** — The number of Bernoulli($p$) trials up to and including the first success, $P(X = k) = (1-p)^{k-1}p$, mean $1/p$; the discrete memoryless distribution. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Gibbs sampling** — An MCMC method that updates one coordinate at a time by sampling it from its full conditional $p(x_j \mid \mathbf{x}_{-j})$; every proposal is accepted, but strongly correlated coordinates make it mix slowly. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Gradient** — $\nabla f = (\partial f/\partial x_1, \dots, \partial f/\partial x_d)^\top$, the vector of partial derivatives. It points in the direction of steepest ascent, is perpendicular to the level sets, and has length equal to the maximal rate of change. → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**Gradient checking** — Validating an analytic gradient against central differences with the relative error $\lVert\mathbf{g} - \hat{\mathbf{g}}\rVert/(\lVert\mathbf{g}\rVert + \lVert\hat{\mathbf{g}}\rVert)$; below $10^{-7}$ in double precision is correct, above $10^{-2}$ is a bug. Kinks (ReLU) and float32 give false alarms. → [Module 06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md)

**Gradient clipping** — Rescaling the gradient when its norm exceeds a threshold, $\mathbf{g} \leftarrow \mathbf{g}\cdot\min(1, c/\lVert\mathbf{g}\rVert)$, to stop a single exploding step from wrecking training. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Gradient descent** — The iteration $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta\nabla L(\boldsymbol{\theta}_k)$. On a quadratic it is stable iff $\eta < 2/L$, converges fastest with $\eta^\star = 2/(L + \mu)$, and shrinks the error by $(\kappa - 1)/(\kappa + 1)$ per step. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Gram–Schmidt process** — Builds an orthonormal basis from independent vectors by subtracting from each vector its projections onto the previous basis vectors and normalising; `np.linalg.qr` does the same up to signs. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Growth function** — $\Pi_{\mathcal{H}}(n)$, the maximum number of distinct labellings a hypothesis class can realise on $n$ points; at most $2^n$, and polynomial in $n$ once $n$ exceeds the VC dimension (Sauer's lemma). → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

## H

**Hamiltonian Monte Carlo (HMC)** — MCMC that augments the state with a momentum $\mathbf{v}$, simulates the Hamiltonian $H = -\log\tilde p(\mathbf{x}) + \tfrac12\lVert\mathbf{v}\rVert^2$ with leapfrog steps, and accepts with probability $\min\{1, e^{-\Delta H}\}$; gradients let it move far in one proposal. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Hat matrix** — $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$, the orthogonal projector onto the column space that maps $\mathbf{y}$ to the fitted values; $\mathbf{H}^2 = \mathbf{H} = \mathbf{H}^\top$ and $\operatorname{tr}\mathbf{H} = \operatorname{rank}\mathbf{X}$. Its diagonal entries are the leverages (Module 13). → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Hessian** — The $d \times d$ symmetric matrix of second partial derivatives, $H_{ij} = \partial^2 f/\partial x_i\partial x_j$. Its eigenvalues classify critical points (all positive: minimum; mixed: saddle) and set the curvature, and hence the step size, of gradient descent. → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**Hoeffding's inequality** — For $n$ independent variables in $[0, 1]$, $P(\lvert\bar X_n - \mu\rvert \ge t) \le 2e^{-2nt^2}$: a distribution-free, exponential tail bound. It sizes test sets, $n \ge \ln(2/\delta)/(2\varepsilon^2)$, and yields the UCB1 index (Module 19). → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Holm procedure** — A step-down multiple-comparison method: sort the p-values and compare $p_{(i)}$ with $\alpha/(m - i + 1)$, stopping at the first failure. It controls the FWER like Bonferroni but rejects more. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Huffman code** — The optimal prefix code when each symbol is coded separately; its expected length lies within one bit of the entropy, in line with the source-coding theorem. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Hyperplane** — The set $\{\mathbf{x} : \mathbf{w}\cdot\mathbf{x} + b = 0\}$ with normal vector $\mathbf{w}$; a linear classifier's decision boundary. The signed distance of a point is $(\mathbf{w}\cdot\mathbf{x} + b)/\lVert\mathbf{w}\rVert$ and its margin is that distance times the label. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Hypothesis class** — The set $\mathcal{H}$ of functions a learner is allowed to return; its size or capacity (finite cardinality, VC dimension, norm bound) governs how far training error can be trusted. Module 13 states the supervised-learning setup. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

## I

**i.i.d. (independent and identically distributed)** — The assumption that the observations are drawn independently from one fixed distribution. Likelihoods factorise under it, the CLT and Hoeffding's inequality need it, and every generalisation bound assumes it. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Implicit regularisation** — The preference an optimiser shows among the many interpolating solutions: gradient descent from zero on least squares stays in the row space of $\mathbf{X}$ and converges to the minimum-norm solution $\mathbf{X}^+\mathbf{y}$ without any explicit penalty. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Importance sampling** — Estimating $\mathbb{E}_p[f]$ with draws from a proposal $q$ by reweighting, $\mathbb{E}_p[f] = \mathbb{E}_q[f\,p/q]$; the self-normalised version $\sum\tilde w_if_i/\sum\tilde w_i$ works with unnormalised densities. A light-tailed proposal gives a few huge weights and a tiny effective sample size. Module 19 uses it for off-policy evaluation. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Independence** — $P(A \cap B) = P(A)P(B)$, or for random variables $p(x, y) = p_X(x)p_Y(y)$: knowing one tells you nothing about the other. Not the same as mutual exclusivity, and strictly stronger than zero correlation (Module 09). → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Indicator variable** — $\mathbf{1}_A$ equals 1 when event $A$ occurs and 0 otherwise, so $\mathbb{E}[\mathbf{1}_A] = P(A)$; with linearity of expectation it turns counting problems into sums of probabilities. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Inductive bias** — The assumptions a learner brings beyond the data: the architecture, the regulariser, the prior. The no-free-lunch theorem says some such bias is unavoidable; the choice of CNN, transformer or GNN is the main one in deep learning. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Information content (surprise)** — $I(x) = -\log p(x)$: the information gained from observing an outcome of probability $p(x)$, in bits or nats. The logarithm is forced by requiring independent surprises to add. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Information gain** — The reduction in label entropy achieved by splitting on attribute $A$, $\mathrm{IG} = H(Y) - \sum_a\frac{N_a}{N}H(Y \mid A = a) = I(Y; A)$; the splitting criterion of decision trees, alongside the Gini decrease. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Inner (dot) product** — $\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \sum_i u_iv_i = \lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert\cos\theta$: a weighted sum and a measure of alignment at once. Orthogonality means $\mathbf{u}\cdot\mathbf{v} = 0$; attention scores (Module 17) are dot products. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Interpolation (of training data)** — A model interpolates when its training error is zero. Over-parameterised networks reach it even on random labels; whether an interpolating model generalises depends on which interpolant the optimiser chooses. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Inverse (of a matrix)** — $\mathbf{A}^{-1}$ with $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$, which exists iff $\det\mathbf{A} \ne 0$; $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$. Solve systems with `np.linalg.solve` rather than forming the inverse, for both speed and accuracy. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Inverse-CDF sampling** — If $U \sim \mathrm{Unif}(0, 1)$ then $F^{-1}(U)$ has CDF $F$; for the Exponential, $X = -\ln(1 - U)/\lambda$. The quantile function is introduced in Module 07 and the method catalogued in Module 08. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**IRLS (iteratively reweighted least squares)** — Newton's method for a GLM written as a sequence of weighted least-squares fits: for logistic regression $\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} + (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top(\mathbf{y} - \mathbf{p})$ with $W_{ii} = p_i(1 - p_i)$. It diverges on perfectly separable data. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

## J

**Jacobian** — For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the $m \times n$ matrix $J_{ij} = \partial f_i/\partial x_j$ that gives the best linear approximation $\mathbf{f}(\mathbf{x} + \boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}\boldsymbol{\delta}$ and propagates covariance as $\mathbf{J}\boldsymbol{\Sigma}\mathbf{J}^\top$. Products of Jacobians are what vanish or explode through deep networks (Module 16). → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**James–Stein estimator** — $\hat{\boldsymbol{\theta}}_{\mathrm{JS}} = \big(1 - \frac{p-2}{\lVert\mathbf{X}\rVert^2}\big)\mathbf{X}$, which shrinks $p \ge 3$ independent normal means toward zero and beats the unbiased estimator in total MSE for every $\boldsymbol{\theta}$. The canonical proof that bias can pay. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Jensen's inequality** — For convex $\varphi$, $\varphi(\mathbb{E}X) \le \mathbb{E}\varphi(X)$; with $\varphi = -\log$ it gives $\log\mathbb{E}X \ge \mathbb{E}\log X$. It proves $D_{\mathrm{KL}} \ge 0$ and derives the ELBO. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Jensen–Shannon divergence** — The symmetrised KL to the mixture $m = (p + q)/2$: $\tfrac12 D(p \,\|\, m) + \tfrac12 D(q \,\|\, m)$, bounded by $\log 2$; its square root is a metric. The original GAN objective minimises it (Module 18). → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Joint, marginal and conditional distributions** — The joint $p(x, y)$ describes two variables together; the marginal $p_X(x) = \sum_y p(x, y)$ sums one out; the conditional $p(y \mid x) = p(x, y)/p_X(x)$ renormalises a slice. Independence is the factorisation $p(x, y) = p_X(x)p_Y(y)$. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

## K

**Kalman filter** — Recursive Gaussian conditioning for a moving quantity: predict $\mu^- = \mu + v$, $P^- = P + q$, then correct with gain $K = P^-/(P^- + r)$, $\mu \leftarrow \mu^- + K(z - \mu^-)$, $P \leftarrow (1 - K)P^-$. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Kernel** — A function $k(\mathbf{x}, \mathbf{x}')$ that acts as an inner product in some feature space, so that the Gram matrix $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is symmetric positive semidefinite; the RBF kernel is the course's running example, as the covariance of a Gaussian process. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**KKT conditions** — Necessary conditions for a constrained optimum with inequality constraints $g_j \le 0$: stationarity of the Lagrangian, feasibility, non-negative multipliers $\mu_j \ge 0$ and complementary slackness $\mu_jg_j = 0$ (a constraint is either active or its multiplier is zero). → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**KL divergence (Kullback–Leibler)** — $D_{\mathrm{KL}}(p \,\|\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \ge 0$, zero iff $p = q$: the extra code length from using model $q$ when the truth is $p$. It is asymmetric (forward KL covers modes, reverse KL seeks them) and not a metric. Minimising cross-entropy minimises it (Module 11); it regularises RLHF (Module 19). → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**KV cache** — At generation step $t$ the keys and values of all earlier positions are stored rather than recomputed, costing $2\,L\,t\,h_{kv}d_{\mathrm{head}}$ numbers (4 GiB in fp16 for a 32-layer, $d = 4096$ model at 8192 tokens). Grouped-query attention shrinks $h_{kv}$ to reduce it. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

## L

**Lagrange multipliers** — For an equality-constrained minimum of $f$ subject to $h = 0$, the gradients are parallel: $\nabla f + \lambda\nabla h = 0$. The multiplier $-\lambda$ is the shadow price, the rate at which the optimum changes with the constraint level. Module 14 uses the Lagrangian to derive maximum-entropy distributions. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Langevin dynamics** — The update $\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\epsilon}{2}\nabla\log p(\mathbf{x}_t) + \sqrt{\epsilon}\,\boldsymbol{\xi}_t$: gradient ascent on the log-density plus Gaussian noise, which samples from $p$ as $\epsilon \to 0$. Only the score is needed, so the normaliser never appears. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Laplace smoothing** — Adding one pseudo-count to every category, $\hat P(w \mid c) = (n_{wc} + 1)/(n_c + V)$, so that unseen words do not zero out a naive Bayes product. It is the posterior mean under a uniform Beta or Dirichlet prior (Module 11). → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Lasso** — Least squares with an $L_1$ penalty $\lambda\lVert\boldsymbol{\beta}\rVert_1$. The diamond-shaped constraint set has corners on the axes, so solutions have exact zeros; the one-dimensional update is the soft threshold $S_\lambda(z) = \operatorname{sign}(z)\max(\lvert z\rvert - \lambda, 0)$. Equivalent to MAP with a Laplace prior (Module 11). → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Law of large numbers (LLN)** — The sample mean converges to the expectation: in the weak form $P(\lvert\bar X_n - \mu\rvert \ge \varepsilon) \le \sigma^2/(n\varepsilon^2) \to 0$ by Chebyshev. It fails for heavy-tailed distributions without a mean, such as the Cauchy. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Law of total probability** — $P(A) = \sum_i P(A \mid B_i)P(B_i)$ for a partition $\{B_i\}$ of the sample space; it computes the evidence in Bayes' theorem. Its expectation version is the tower property (Module 09). → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Law of total variance** — $\operatorname{Var}Y = \mathbb{E}[\operatorname{Var}(Y \mid X)] + \operatorname{Var}(\mathbb{E}[Y \mid X])$: within-group variance plus between-group variance. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Layer normalisation** — Standardising each token's feature vector over its $d$ components, $\gamma\odot(\mathbf{x} - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$, independently of the batch; RMSNorm drops the mean subtraction. Preferred to batch norm in sequence models because it does not depend on the other sequences in the batch. Module 16 introduces it next to batch norm. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**Leakage** — Information from the test or validation data reaching the training process (fitting a scaler on all rows, duplicated records across splits, features computed with future data), which makes validation scores optimistic. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Learning rate (step size)** — $\eta$, the multiplier on the gradient in each update; too small crawls, too large diverges. Schedules such as cosine decay $\eta_t = \eta_{\min} + \frac12(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$ lower the SGD noise floor over time. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Least squares** — Choosing $\hat{\mathbf{x}}$ to minimise $\lVert\mathbf{A}\mathbf{x} - \mathbf{b}\rVert^2$ when $\mathbf{A}\mathbf{x} = \mathbf{b}$ has no exact solution; geometrically, projecting $\mathbf{b}$ onto the column space, algebraically the normal equations. Module 03 solves it via the SVD, Module 13 turns it into regression. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Leverage** — $h_{ii}$, the $i$-th diagonal entry of the hat matrix: how strongly observation $i$ pulls its own fitted value. The average is $p/n$, and $h_{ii} > 2p/n$ flags an influential input; Cook's distance combines leverage with the residual. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Likelihood** — $\mathcal{L}(\theta) = \prod_i p(x_i \mid \theta)$, the probability of the observed data as a function of the parameter; its logarithm $\ell(\theta)$ is a sum. It is not a probability distribution over $\theta$. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Linear combination, span and independence** — $c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k$ is a linear combination; the span is the set of all of them; the vectors are linearly independent when only the trivial combination gives $\mathbf{0}$. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Linear map** — A function with $f(a\mathbf{x} + b\mathbf{y}) = af(\mathbf{x}) + bf(\mathbf{y})$; every such map between coordinate spaces is multiplication by a matrix, and $\mathbf{A}\mathbf{x}$ is a combination of the columns of $\mathbf{A}$ weighted by the entries of $\mathbf{x}$. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Log-derivative trick** — $\nabla_\theta\mathbb{E}_{p_\theta}[f] = \mathbb{E}_{p_\theta}[f\,\nabla_\theta\log p_\theta]$: a gradient of an expectation over a parameterised distribution becomes an expectation you can estimate by sampling. It yields REINFORCE and the score-function estimator (Module 18). → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Logistic regression** — The GLM $p = \sigma(\boldsymbol{\beta}^\top\mathbf{x})$ for a binary target, fitted by maximising the Bernoulli log-likelihood; gradient $\mathbf{X}^\top(\mathbf{p} - \mathbf{y})$, Hessian $\mathbf{X}^\top\mathbf{W}\mathbf{X}$, and $e^{\beta_j}$ is an odds ratio. It is the zero-hidden-layer special case of Module 16 and a maximum-entropy model in Module 14. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Logit** — The pre-softmax (or pre-sigmoid) score $z$ of a model; $\log\frac{p}{1-p}$ for a probability $p$. Loss gradients with respect to logits are especially simple: $\sigma(z) - y$ and $\mathbf{p} - \mathbf{y}$. → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**Log-sum-exp trick** — $\ln\sum_i e^{z_i} = m + \ln\sum_i e^{z_i - m}$ with $m = \max_i z_i$, which avoids overflow in softmax and cross-entropy; $\ln\mathbf{p} = \mathbf{z} - \mathrm{LSE}(\mathbf{z})$. → [Module 06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md)

**LoRA (low-rank adaptation)** — Fine-tuning a frozen weight matrix through a low-rank update, $\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$, with $r(d_{\mathrm{in}} + d_{\mathrm{out}})$ trainable parameters that can be merged into $\mathbf{W}_0$ for inference. It is the low-rank approximation of Module 03 applied to the weight change. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**LOTUS (law of the unconscious statistician)** — $\mathbb{E}[g(X)] = \sum_x g(x)p(x)$ or $\int g(x)f(x)\,dx$: the expectation of a transformed variable needs only the distribution of the original one. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Low-rank approximation** — Replacing $\mathbf{A}$ by the sum of its top $k$ singular components, $\mathbf{A}_k = \sum_{i \le k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$; optimal by Eckart–Young and the basis of compression, PCA and LoRA. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

## M

**Mahalanobis distance** — $\Delta(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}$, the distance that accounts for the data's spread and correlations; $\Delta^2 \sim \chi^2_d$ for a Gaussian, which gives anomaly-detection thresholds. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Mann–Whitney U test** — A rank-based two-sample test: $U$ counts the pairs $(a_i, b_j)$ with $a_i > b_j$ and is compared with its null mean $n_1n_2/2$. It needs no normality and resists outliers. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**MAP (maximum a posteriori) estimation** — $\hat\theta_{\mathrm{MAP}} = \arg\max[\ell(\theta) + \log p(\theta)]$, the mode of the posterior. A Gaussian prior makes it ridge regression with $\lambda = \sigma^2/\tau^2$, a Laplace prior makes it the lasso; weight decay is MAP in disguise. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Margin** — For a labelled point, $y\,(\mathbf{w}\cdot\mathbf{x} + b)/\lVert\mathbf{w}\rVert$: the signed distance to the decision hyperplane, positive when correctly classified. Margin bounds (Module 15) make generalisation depend on $(r/\rho)^2$ rather than on the dimension. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Markov chain** — A sequence of states in which the next state depends only on the current one, $P(s_{t+1} \mid s_t, s_{t-1}, \dots) = P(s_{t+1} \mid s_t)$, described by a transition matrix $\mathbf{P}$ with rows summing to 1; distributions evolve as $\boldsymbol{\pi}_{t+1} = \boldsymbol{\pi}_t\mathbf{P}$. Module 18 builds samplers from them, Module 19 adds actions and rewards. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Markov decision process (MDP)** — A tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ of states, actions, transition probabilities $P(s' \mid s, a)$, expected rewards $R(s, a)$ and a discount factor, with the Markov property that the future depends on the past only through the current state. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Markov's inequality** — $P(X \ge a) \le \mathbb{E}[X]/a$ for a non-negative $X$; the weakest concentration bound and the one from which Chebyshev's and Hoeffding's inequalities are derived. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Matrix multiplication** — $(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik}B_{kj}$, defined when the inner dimensions agree, and equal to composing the two linear maps; not commutative, $O(mpn)$ for an $(m \times p)(p \times n)$ product. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Maximum-entropy principle** — Among all distributions satisfying known constraints $\mathbb{E}[f_k] = c_k$, choose the one with the largest entropy; the Lagrangian solution is the exponential family $p(x) \propto \exp(-\sum_k\lambda_kf_k(x))$ (Gaussian for fixed mean and variance, softmax for a fixed expected score). → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**MCMC (Markov chain Monte Carlo)** — Sampling from a target by running a Markov chain whose stationary distribution is the target and averaging along the trajectory; Metropolis–Hastings, Gibbs, Langevin and HMC are the instances in the course. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Mean squared error (MSE)** — Of an estimator: $\mathbb{E}[(\hat\theta - \theta)^2] = \operatorname{Var}(\hat\theta) + \operatorname{Bias}(\hat\theta)^2$ (Module 10). As a loss: the average squared residual, which is the Gaussian negative log-likelihood (Module 11) with logit gradient $z - y$ (Module 04). → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Memorylessness** — $P(T > s + t \mid T > s) = P(T > t)$: having waited $s$ tells you nothing about the remaining wait. Only the Exponential (continuous) and Geometric (discrete) distributions have it. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Metropolis–Hastings** — Propose $x' \sim q(x' \mid x)$ and accept with probability $\min\{1, \frac{\tilde p(x')\,q(x \mid x')}{\tilde p(x)\,q(x' \mid x)}\}$, otherwise stay; the chain satisfies detailed balance with respect to $p$ and needs only the unnormalised $\tilde p$. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Minimum detectable effect (MDE)** — The smallest true difference an experiment of given size detects with the chosen power, $\delta_{\min} = (z_{1-\alpha/2} + z_{1-\beta})\,\sigma\sqrt{2/n}$ for two arms. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Mixing time** — How long a Markov chain takes to forget its start; the error decays like $\lvert\lambda_2\rvert^t$, so the mixing time scales as $1/(1 - \lvert\lambda_2\rvert)$ with $\lambda_2$ the second-largest eigenvalue modulus of $\mathbf{P}$. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Mixture of experts (MoE)** — A layer with $N_e$ parallel expert networks and a router $\mathbf{g} = \operatorname{softmax}(\mathbf{W}_g\mathbf{x})$ that keeps only the top-$k$ experts per token, so parameters scale with $N_e$ while compute scales with $k$. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**MLE (maximum likelihood estimation)** — $\hat\theta = \arg\max_\theta\ell(\theta)$, the parameter that makes the data most probable; found where the score $\ell'(\theta)$ vanishes. It is invariant under reparameterisation, asymptotically normal with variance $1/I_n$, and the reason MSE and cross-entropy are the standard losses. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Moment generating function (MGF)** — $M_X(t) = \mathbb{E}[e^{tX}]$; its $k$-th derivative at 0 is the $k$-th moment and the MGF of an independent sum is the product of the MGFs, which identifies the distribution of sums. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Momentum (heavy ball and Nesterov)** — Accumulating a velocity $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \mathbf{g}_t$ and stepping $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\mathbf{v}_t$, which damps oscillation across narrow valleys and improves the rate on a quadratic from $(\kappa - 1)/(\kappa + 1)$ to $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$; Nesterov evaluates the gradient at the look-ahead point. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Monte Carlo estimation** — Approximating $\mathbb{E}[g(X)]$ by the average $\frac1n\sum_i g(X_i)$ of simulated draws; its standard error $\sigma_g/\sqrt n$ does not depend on the dimension. Module 08 adds variance reduction, Module 18 the samplers, Module 19 the Monte Carlo value update. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Multi-head attention** — Running $h$ attention heads in parallel on $d/h$-dimensional projections and concatenating the results through an output projection $\mathbf{W}_O$; the $4d^2$ attention parameters per block are independent of $h$. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**Multicollinearity** — Features that are nearly linear combinations of one another, making $\mathbf{X}^\top\mathbf{X}$ ill-conditioned and coefficients unstable; measured by the variance inflation factor $\mathrm{VIF}_j = 1/(1 - R_j^2)$. Exact collinearity is a rank deficiency (Module 02). → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Multivariate Gaussian** — $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with density $(2\pi)^{-d/2}\lvert\boldsymbol{\Sigma}\rvert^{-1/2}\exp(-\tfrac12\Delta^2)$; its level sets are ellipsoids along the eigenvectors of $\boldsymbol{\Sigma}$, affine maps keep it Gaussian ($\mathbf{A}\mathbf{X} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$), and marginals and conditionals are Gaussian. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Mutual information** — $I(X; Y) = D_{\mathrm{KL}}(p(x, y) \,\|\, p(x)p(y)) = H(X) - H(X \mid Y) \ge 0$: how many bits one variable carries about another, zero iff they are independent. Unlike correlation it captures any dependence; for a bivariate Gaussian it is $-\tfrac12\log(1 - \rho^2)$. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

## N

**Naive Bayes** — A classifier that applies Bayes' theorem under the assumption that features are conditionally independent given the class, $\hat c = \arg\max_c[\log P(c) + \sum_j\log P(w_j \mid c)]$, computed in log space with Laplace smoothing. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Negative Binomial distribution** — The number of failures before the $r$-th success, $P(X = k) = \binom{k+r-1}{k}p^r(1-p)^k$; its variance exceeds its mean, which makes it the standard model for over-dispersed counts that the Poisson underfits. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Newton's method** — $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \mathbf{H}^{-1}\nabla L$, the minimiser of the local quadratic Taylor model. It converges quadratically near a minimum but costs $O(d^3)$ per step; L-BFGS and Gauss–Newton are the work-arounds, IRLS (Module 13) its GLM form. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**No-free-lunch theorem** — Averaged uniformly over all possible target functions, every learning algorithm has the same expected error on unseen inputs. Learning therefore requires an inductive bias that matches the problem. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Norm** — A length function: $\lVert\mathbf{x}\rVert_2 = \sqrt{\sum x_i^2}$ (Euclidean), $\lVert\mathbf{x}\rVert_1 = \sum\lvert x_i\rvert$ (Manhattan), $\lVert\mathbf{x}\rVert_\infty = \max\lvert x_i\rvert$, with unit balls shaped like a sphere, a diamond and a cube. Ridge and lasso (Module 13) penalise the $L_2$ and $L_1$ norms. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Normal (Gaussian) distribution** — $\mathcal{N}(\mu, \sigma^2)$ with density $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$; the limit of sums of many small effects (CLT), closed under independent sums, and the reason Gaussian noise leads to the squared-error loss. Standardising gives $Z = (X - \mu)/\sigma \sim \mathcal{N}(0, 1)$ and the 68–95–99.7 rule. → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Normal equations** — $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^\top\mathbf{b}$, the condition that the least-squares residual is orthogonal to the column space; derived geometrically here and by matrix calculus in Module 04. Solve them via QR or SVD rather than by inverting $\mathbf{A}^\top\mathbf{A}$. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Normalising flow** — A generative model built from an invertible map $f$ with a tractable Jacobian, so that $\log p(\mathbf{x}) = \log p_Z(f(\mathbf{x})) + \log\lvert\det\mathbf{J}_f(\mathbf{x})\rvert$ gives an exact likelihood. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Null hypothesis and alternative** — $H_0$ is the default claim a test tries to reject (no effect, equal means); $H_1$ is what you conclude when the data are too surprising under $H_0$. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Null space** — $\mathcal{N}(\mathbf{A}) = \{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{0}\}$, the inputs the map destroys. A non-trivial null space means a system has infinitely many solutions and a regression has non-identifiable coefficients. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

## O

**Odds and odds ratio** — Odds are $p/(1 - p)$; in logistic regression a one-unit increase in $x_j$ multiplies the odds by $e^{\beta_j}$, the odds ratio. Bayes' theorem in odds form (Module 07) multiplies prior odds by a likelihood ratio. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Off-policy and on-policy learning** — On-policy methods (SARSA) evaluate and improve the policy that generates the data; off-policy methods (Q-learning) learn about the greedy policy from data collected by an exploratory one, and need importance weights or a max operator to correct for the mismatch. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**OLS (ordinary least squares)** — $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$, derived as a projection, by calculus and as the Gaussian MLE; $\operatorname{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$ and $R^2 = 1 - \mathrm{RSS}/\mathrm{TSS}$. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**One-hot encoding** — Representing category $k$ of $K$ as the standard basis vector $\mathbf{e}_k$. Multiplying a one-hot vector by a matrix selects a row (embedding lookup), and including all $K$ columns together with an intercept is a rank deficiency (the dummy-variable trap). → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Orthogonal matrix** — $\mathbf{Q}$ with $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$: orthonormal columns, $\mathbf{Q}^{-1} = \mathbf{Q}^\top$, and $\lVert\mathbf{Q}\mathbf{x}\rVert = \lVert\mathbf{x}\rVert$, so lengths and angles survive. Rotations and permutations are examples; the $\mathbf{U}$ and $\mathbf{V}$ of the SVD are orthogonal. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Orthonormal basis** — Basis vectors with $\mathbf{q}_i\cdot\mathbf{q}_j = \delta_{ij}$; coordinates are then dot products, $\mathbf{x} = \sum_i(\mathbf{x}\cdot\mathbf{q}_i)\mathbf{q}_i$, and Parseval's identity gives $\lVert\mathbf{x}\rVert^2 = \sum_i c_i^2$. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

## P

**p-value** — The probability, computed under $H_0$, of a test statistic at least as extreme as the observed one, $p = P_{H_0}(T \ge t_{\mathrm{obs}})$. It is not the probability that $H_0$ is true, and is $\mathrm{Unif}(0, 1)$ under $H_0$ for a continuous statistic. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**PAC learning** — A class is probably approximately correct learnable when some algorithm, given $n \ge n_{\mathcal{H}}(\varepsilon, \delta)$ i.i.d. examples from any distribution, returns a hypothesis with risk at most $\varepsilon$ (above the best in class) with probability at least $1 - \delta$. The fundamental theorem: learnable iff the VC dimension is finite. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**PageRank** — The stationary distribution of the Google matrix $\mathbf{G} = d\mathbf{M} + \frac{1-d}{n}\mathbf{1}\mathbf{1}^\top$, a random surfer who follows links with probability $d$ and jumps uniformly otherwise; computed by power iteration, converging at rate $\lvert\lambda_2\rvert \le d$. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**Paired test** — Comparing two conditions on the same units (patients before and after, two models on the same test items) and testing the differences, which removes between-unit variance and raises power; the paired t-test and McNemar's test are the two cases in the course. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Partial derivative** — $\partial f/\partial x_i$, the derivative with respect to one variable with the others held fixed; the components of the gradient. → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**PCA (principal component analysis)** — Centre the data, form $\mathbf{C} = \mathbf{X}_c^\top\mathbf{X}_c/(n-1)$, and take its eigenvectors (the right singular vectors of $\mathbf{X}_c$) as the directions of maximal variance; scores $\mathbf{Z} = \mathbf{X}_c\mathbf{V}_k$, explained variance $\lambda_i = \sigma_i^2/(n-1)$, reconstruction error $(n-1)\sum_{i>k}\lambda_i$. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**Peeking (optional stopping)** — Checking a running experiment repeatedly and stopping when it looks significant; with $K = 10$ equally spaced looks at nominal $\alpha = 0.05$ the false-positive rate is about $0.19$. Group-sequential boundaries (Pocock) or Bayesian analyses are the principled alternatives. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Permutation equivariance** — Attention applied to permuted rows gives permuted outputs, $\operatorname{Attn}(\boldsymbol{\Pi}\mathbf{Q}, \boldsymbol{\Pi}\mathbf{K}, \boldsymbol{\Pi}\mathbf{V}) = \boldsymbol{\Pi}\operatorname{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$; a transformer therefore knows nothing about order unless positional information is injected. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**Permutation test** — A test that builds the null distribution by shuffling the group labels and recomputing the statistic, $\hat p = (1 + \#\{T^{(b)} \ge t_{\mathrm{obs}}\})/(B + 1)$; exact when all splits are enumerated and valid for any statistic. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Perplexity** — $\exp(\mathcal{L}_{\mathrm{nats}}) = 2^{\mathcal{L}_{\mathrm{bits}}}$, the exponential of the average cross-entropy per token: the inverse geometric mean of the probabilities a language model assigned to the true tokens, equal to $V$ for a uniform model over $V$ tokens. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**PMF, PDF and CDF** — A probability mass function $p(x) = P(X = x)$ (discrete), a probability density $f$ with $P(a \le X \le b) = \int_a^b f$ (continuous), and the cumulative distribution function $F(x) = P(X \le x)$ with $f = F'$. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Poisson distribution and process** — Poisson($\lambda$) has $P(X = k) = e^{-\lambda}\lambda^k/k!$ with mean and variance $\lambda$, the limit of Binomial($n, \lambda/n$). In a Poisson process events arrive at constant rate: counts in a window of length $t$ are Poisson($\lambda t$) and gaps are Exponential($\lambda$). → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Policy** — $\pi(a \mid s)$, the agent's rule for choosing actions, deterministic or stochastic. Policy improvement makes it greedy with respect to $Q^\pi$; policy gradient methods parameterise it directly. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Policy gradient theorem and REINFORCE** — $\nabla_\theta J = \mathbb{E}\big[\sum_t\nabla_\theta\log\pi_\theta(a_t \mid s_t)\,(G_t - b(s_t))\big]$ for any baseline $b$, obtained from the log-derivative trick; REINFORCE is its Monte Carlo estimate, and a state-dependent baseline reduces variance without adding bias. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Policy iteration and value iteration** — Dynamic programming for a known MDP: policy iteration alternates exact policy evaluation with greedy improvement; value iteration applies the Bellman optimality operator repeatedly, $V_{k+1}(s) = \max_a[R(s,a) + \gamma\sum_{s'}P(s' \mid s,a)V_k(s')]$, converging geometrically at rate $\gamma$. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Positional encoding (sinusoidal and RoPE)** — Position information added to or rotated into token vectors. Sinusoidal encodings use $\sin(pos\,\omega_i), \cos(pos\,\omega_i)$ with $\omega_i = 10000^{-2i/d}$; rotary encodings rotate queries and keys so that $\mathbf{q}_m^\top\mathbf{k}_n = \mathbf{q}^\top\mathbf{R}((n - m)\theta)\mathbf{k}$ depends only on the relative offset. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**Positive semidefinite (PSD) and positive definite** — A symmetric $\mathbf{A}$ with $\mathbf{x}^\top\mathbf{A}\mathbf{x} \ge 0$ for all $\mathbf{x}$ ($\mathbf{A} \succeq 0$), equivalently all eigenvalues $\ge 0$; positive definite ($\succ 0$) when strict for $\mathbf{x} \ne \mathbf{0}$. Covariance matrices, kernels, $\mathbf{A}^\top\mathbf{A}$ and the Hessian at a minimum are PSD. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Posterior** — $p(\theta \mid \mathcal{D}) = p(\mathcal{D} \mid \theta)\,p(\theta)/p(\mathcal{D})$, the distribution over parameters after seeing the data; the object Bayesian inference computes and MCMC (Module 18) samples from. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Posterior predictive distribution** — $p(x_{\mathrm{new}} \mid \mathcal{D}) = \int p(x_{\mathrm{new}} \mid \theta)\,p(\theta \mid \mathcal{D})\,d\theta$, which averages predictions over parameter uncertainty and is therefore wider than any plug-in prediction. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Power** — $1 - \beta$, the probability of rejecting $H_0$ when the alternative is true; for a two-sample z-test $1 - \beta \approx \Phi\big(\delta/(\sigma\sqrt{2/n}) - z_{1-\alpha/2}\big)$. Sample-size planning solves this for $n$. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Power iteration** — $\mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k/\lVert\mathbf{A}\mathbf{x}_k\rVert$ converges to the dominant eigenvector at rate $\lvert\lambda_2/\lambda_1\rvert^k$; PageRank is its best-known use. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**PPO (proximal policy optimisation)** — A policy-gradient method that limits each update through the clipped objective $\min\big(\rho\hat A, \operatorname{clip}(\rho, 1 - \epsilon, 1 + \epsilon)\hat A\big)$ with $\rho = \pi_\theta/\pi_{\mathrm{old}}$, so the new policy cannot move far from the one that collected the data. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Prior** — $p(\theta)$, the distribution over parameters before the data; conjugate priors give closed-form posteriors, and a Gaussian or Laplace prior is a ridge or lasso penalty once you take logs. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Projection** — The component of $\mathbf{a}$ along $\mathbf{b}$, $\mathbf{p} = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\mathbf{b}$, leaving a residual $\mathbf{a} - \mathbf{p}$ orthogonal to $\mathbf{b}$; the closest point on the line. Module 02 generalises it to a subspace (the hat matrix), which is least squares. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Pseudo-inverse (Moore–Penrose)** — $\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$, inverting the non-zero singular values; $\mathbf{A}^+\mathbf{b}$ is the minimum-norm least-squares solution, which is what `np.linalg.lstsq` returns when there are more unknowns than equations. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

## Q

**Q-learning** — Off-policy TD control: $Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma\max_{a'}Q(s', a') - Q(s, a)]$, which learns the optimal action values while following an exploratory policy; it converges with Robbins–Monro step sizes if every pair is visited infinitely often. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Quadratic form** — $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ for symmetric $\mathbf{A}$; in the eigenbasis it is $\sum_i\lambda_iy_i^2$, so its sign pattern is read off the eigenvalues and the Rayleigh quotient $\mathbf{x}^\top\mathbf{A}\mathbf{x}/\mathbf{x}^\top\mathbf{x}$ lies between $\lambda_{\min}$ and $\lambda_{\max}$. The Gaussian exponent (Module 09) and the second-order Taylor term (Module 04) are quadratic forms. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**Quantile function** — $Q(u) = \min\{x : F(x) \ge u\}$, the inverse of the CDF; `ppf` in `scipy.stats`. Quantiles of the sampling distribution give confidence intervals, and $Q(U)$ with uniform $U$ is inverse-CDF sampling. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Quantisation** — Storing weights in $b$ bits with a per-block scale $s = \max\lvert w\rvert/(2^{b-1} - 1)$ and $\hat w = s\operatorname{round}(w/s)$; the rounding error is roughly uniform on $[-s/2, s/2]$ with RMS $s/\sqrt{12}$. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

## R

**Rademacher complexity** — A data-dependent capacity measure: how well the class can correlate with random $\pm 1$ labels on the sample. The bound $R(h) \le \hat R_S(h) + 2\mathfrak{R}_n(\mathcal{H}) + \sqrt{\ln(1/\delta)/(2n)}$ gives $\mathfrak{R}_n \le rB/\sqrt n$ for norm-bounded linear classifiers. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Random variable** — A function from the sample space to the real numbers, so that events become statements like $X \le x$; discrete ones have a PMF, continuous ones a PDF. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Randomised controlled experiment** — Assigning units to treatment or control by a random mechanism independent of everything about them, so that the two groups differ only by chance and by the treatment; control, blinding and blocking make the comparison precise. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Rank** — $\operatorname{rank}\mathbf{A} = \dim\mathcal{C}(\mathbf{A})$, the number of linearly independent columns (equal to the number of independent rows and to the number of non-zero singular values). Rank–nullity: $\operatorname{rank}\mathbf{A} + \dim\mathcal{N}(\mathbf{A}) = n$. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Regret** — $R_T = \sum_a\Delta_aN_a(T)$ with gap $\Delta_a = \mu^* - \mu_a$: the reward lost by not pulling the best arm from the start. Good bandit algorithms have sublinear regret, the best achievable rate being $O(\log T)$. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Regularisation** — Adding a penalty $\lambda R(\boldsymbol{\theta})$ to the empirical risk (or otherwise restricting the hypothesis class) to trade training fit for lower variance. Ridge, lasso, weight decay, early stopping, dropout and data augmentation are all instances; Module 11 reads penalties as priors, Module 15 as capacity control. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Rejection sampling** — Draw $x \sim q$ and accept it with probability $\tilde p(x)/(Mq(x))$ for an envelope constant $M \ge \sup\tilde p/q$; the acceptance rate is $Z/M$ and collapses exponentially with dimension. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Reparameterisation trick** — Writing a sample as a deterministic function of the parameters and of parameter-free noise, $\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\mathbf{z}$ with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, so that gradients flow through the sampler. The resulting pathwise gradient has lower variance than the score-function estimator; it trains the VAE. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Residual connection** — $\mathbf{A}_l = \mathbf{A}_{l-1} + F(\mathbf{A}_{l-1})$, whose Jacobian $\mathbf{I} + \mathbf{J}_l$ keeps an identity path through the product of layer Jacobians and so prevents vanishing gradients. In a transformer the rows carried along this path form the residual stream (Module 17). → [Module 16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md)

**Return** — $G_t = \sum_{k \ge 0}\gamma^kr_{t+k+1} = r_{t+1} + \gamma G_{t+1}$, the discounted sum of future rewards; value functions are its expectations and $\lvert G_t\rvert \le R_{\max}/(1 - \gamma)$. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Ridge regression** — Least squares plus $\lambda\lVert\boldsymbol{\beta}\rVert_2^2$: $\hat{\boldsymbol{\beta}}_\lambda = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$, which shrinks each singular direction by the filter factor $d_j^2/(d_j^2 + \lambda)$ (previewed in Module 03) and is MAP estimation with a Gaussian prior, $\lambda = \sigma^2/\tau^2$ (Module 11). → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Risk (true and empirical)** — The true risk $R(h) = \mathbb{E}[\ell(h(\mathbf{x}), y)]$ is the expected loss on a fresh draw; the empirical risk $\hat R_S(h) = \frac1n\sum_i\ell(h(\mathbf{x}_i), y_i)$ is its sample average, unbiased for a fixed $h$ but optimistic for the $h$ chosen by training. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**RLHF (reinforcement learning from human feedback)** — Fine-tuning a language model to maximise $\mathbb{E}_{\pi_\theta}[r] - \beta\operatorname{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$, a learned reward minus a KL penalty to a reference policy; the optimum is $\pi^*(y) \propto \pi_{\mathrm{ref}}(y)e^{r(y)/\beta}$, which DPO exploits. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**RMSProp** — An adaptive optimiser that divides each gradient component by the root of a running mean of its squares, $\mathbf{s}_t = \rho\mathbf{s}_{t-1} + (1 - \rho)\mathbf{g}_t^2$, step $\eta\mathbf{g}_t/(\sqrt{\mathbf{s}_t} + \epsilon)$, so that all coordinates move at a comparable rate. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Robbins–Monro conditions** — Step sizes with $\sum_t\alpha_t = \infty$ and $\sum_t\alpha_t^2 < \infty$ (for example $1/t$), under which stochastic approximation schemes such as TD and Q-learning converge. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Rotation matrix** — $\mathbf{R}(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$, orthogonal with determinant 1 and $\mathbf{R}(\alpha)^\top\mathbf{R}(\beta) = \mathbf{R}(\beta - \alpha)$, the identity behind rotary positional encodings (Module 17). → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

## S

**Saddle point** — A stationary point where the Hessian has both positive and negative eigenvalues: a minimum along some directions and a maximum along others. In high dimension almost all critical points of a loss are saddles rather than local minima. → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**Sample space and events** — $\Omega$ is the set of all possible outcomes; an event is a subset of it. The axioms $P(A) \ge 0$, $P(\Omega) = 1$ and additivity over disjoint events generate every rule of probability. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Sample variance** — $s^2 = \frac{1}{n-1}\sum(x_i - \bar x)^2$; the divisor $n - 1$ makes it unbiased because the deviations from $\bar x$ are slightly smaller than those from $\mu$, $\mathbb{E}\sum(X_i - \bar X)^2 = (n-1)\sigma^2$. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Sampling distribution** — The distribution a statistic would have over repeated samples from the same process; its standard deviation is the standard error. The CLT gives its shape for means, the bootstrap approximates it for anything. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**SARSA** — On-policy TD control that updates $Q(s, a)$ toward $r + \gamma Q(s', a')$ using the action $a'$ actually taken next; it learns the value of the exploratory policy it follows. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Sauer's lemma** — For a class of VC dimension $d$, $\Pi_{\mathcal{H}}(n) \le \sum_{i=0}^d\binom{n}{i} \le (en/d)^d$: beyond $d$ points the growth function is polynomial, which is what makes the VC bound possible. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Scaling law** — An empirical fit of language-model loss to model size $N$ and data size $D$, $L(N, D) = E + A/N^\alpha + B/D^\beta$ (Chinchilla: $\alpha = 0.34$, $\beta = 0.28$); it describes how test loss falls with compute but is not a theorem. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Score (statistics)** — $s(\theta) = \partial_\theta\ell(\theta)$, the gradient of the log-likelihood; it vanishes at the MLE and its variance is the Fisher information. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Score function (of a density)** — $\nabla_{\mathbf{x}}\log p(\mathbf{x})$, the gradient of the log-density with respect to the data, independent of the normaliser $Z$. Langevin dynamics follows it, and a diffusion model's noise predictor estimates it: $\nabla\log q_t(\mathbf{x}_t) \approx -\boldsymbol{\varepsilon}_\theta/\sqrt{1 - \bar\alpha_t}$. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Selection bias** — Systematic error from a sample whose inclusion depends on the quantity of interest (approved loan applicants, volunteers, reachable households); unlike sampling noise it does not shrink with more data. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Sensitivity, specificity, PPV and NPV** — Sensitivity is $P(\text{positive} \mid \text{disease})$, specificity $P(\text{negative} \mid \text{healthy})$; the positive and negative predictive values invert them with Bayes' theorem and depend on prevalence: $\mathrm{PPV} = \frac{\mathrm{se}\cdot\pi}{\mathrm{se}\cdot\pi + (1 - \mathrm{sp})(1 - \pi)}$. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**SGD (stochastic gradient descent)** — Gradient descent on a mini-batch estimate of the gradient, which is unbiased with covariance about $\boldsymbol{\Sigma}/b$ for batch size $b$; the noise leaves the iterates in a ball around the minimum whose size scales with $\eta$, which decaying schedules shrink. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Shattering** — A hypothesis class shatters a set of $n$ points when it realises all $2^n$ labellings of them; the VC dimension is the largest such $n$. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Shrinkage** — Pulling estimates toward a common value (zero, a grand mean, a prior mean), accepting bias to cut variance: James–Stein (Module 10), ridge regression (Module 13), empirical Bayes and the posterior mean $\frac{a + k}{a + b + n}$. → [Module 11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md)

**Sigmoid (logistic function)** — $\sigma(z) = 1/(1 + e^{-z})$, mapping a logit to a probability, with derivative $\sigma(1 - \sigma) \le 1/4$. The bound is why deep sigmoid networks suffer vanishing gradients (Modules 06 and 16). → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**Simpson's paradox** — An association that holds in every stratum of a confounder reverses in the pooled data, because $P(R \mid T) = \sum_sP(R \mid T, s)P(s \mid T)$ mixes the strata differently for each treatment. Standardised rates or randomisation remove it. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Singular value decomposition (SVD)** — $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \sum_i\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ with orthogonal $\mathbf{U}, \mathbf{V}$ and singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$: every matrix is a rotation, a stretch and a rotation. The $\sigma_i^2$ are the eigenvalues of $\mathbf{A}^\top\mathbf{A}$, the rank is the number of non-zero $\sigma_i$, and it delivers PCA, the pseudo-inverse, the condition number and low-rank approximation. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**Smoothness and strong convexity** — $f$ is $L$-smooth when $\nabla^2 f \preceq L\mathbf{I}$ (curvature at most $L$) and $\mu$-strongly convex when $\nabla^2 f \succeq \mu\mathbf{I}$ with $\mu > 0$; the ratio $\kappa = L/\mu$ is the condition number that sets the speed of gradient descent. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Softmax** — $\operatorname{softmax}(\mathbf{z})_i = e^{z_i}/\sum_je^{z_j}$, turning a score vector into a probability vector; the maximum-entropy distribution with a fixed expected score (Module 14). Its Jacobian is $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ (Module 17), and combined with cross-entropy its logit gradient is $\mathbf{p} - \mathbf{y}$. → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**Soft thresholding** — $S_\lambda(z) = \operatorname{sign}(z)\max(\lvert z\rvert - \lambda, 0)$, the proximal operator of the $L_1$ norm: it shrinks a coefficient toward zero and sets it exactly to zero below $\lambda$. The lasso solution for orthonormal features is $S_\lambda$ applied to the OLS coefficients. → [Module 13](../part-5-learning-theory/13-linear-models-and-regularization/README.md)

**Source-coding theorem** — The entropy is the limit of lossless compression: the optimal expected code length per symbol satisfies $H \le L^* < H + 1$ bits, and prefix codes obey the Kraft inequality $\sum_i 2^{-\ell_i} \le 1$. → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Spectral theorem** — A symmetric matrix has real eigenvalues and an orthonormal eigenbasis: $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ with $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$. It is why covariance matrices and Hessians can always be diagonalised by a rotation. → [Module 03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md)

**Standard error** — The standard deviation of an estimator's sampling distribution: $\sigma/\sqrt n$ for a mean (estimated by $s/\sqrt n$) and $\sqrt{p(1 - p)/n} \le 1/(2\sqrt n)$ for a proportion. It is the honest uncertainty to attach to any reported metric, including a test accuracy. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Standardisation** — $Z = (X - \mu)/\sigma$ gives mean 0 and variance 1; for features, $z_j = (x_j - \bar x_j)/s_j$, required before any penalised regression and before comparing distances across features of different scales. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Stationary distribution** — A $\boldsymbol{\pi}$ with $\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}$, the left eigenvector of the transition matrix for eigenvalue 1; it is unique, and the chain converges to it from any start, when the chain is irreducible and aperiodic. Module 03 computes it by power iteration, Module 18 designs chains whose stationary distribution is a target posterior. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Student-t distribution** — The distribution of a standardised mean when $\sigma$ is estimated from the sample, with $\nu$ degrees of freedom: heavier tails than the normal, approaching it as $\nu \to \infty$, Cauchy at $\nu = 1$. It gives the $t$-interval (Module 10) and the $t$-tests (Module 12). → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

## T

**t-test** — A test for a mean using $t = (\bar x - \mu_0)/(s/\sqrt n)$ with $n - 1$ degrees of freedom; the Welch two-sample version allows unequal variances, and the paired version tests the mean of within-pair differences. Report Cohen's $d$ and a confidence interval alongside the p-value. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

**Taylor expansion** — The local polynomial model $f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f^\top\boldsymbol{\delta} + \tfrac12\boldsymbol{\delta}^\top\mathbf{H}\boldsymbol{\delta}$, with a remainder bounded by the next derivative. Gradient descent minimises the first-order model with a step penalty, Newton's method the second-order one. → [Module 04](../part-2-calculus-and-optimization/04-differential-calculus/README.md)

**TD(0) (temporal-difference learning)** — $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$: move the value toward a one-step bootstrapped target. It has lower variance but more bias than the Monte Carlo update $V(s) \leftarrow V(s) + \alpha(G_t - V(s))$, and works online. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Temperature** — Dividing logits by $T$ before the softmax, $p_T(k) \propto \exp(z_k/T)$: $T \to 0$ gives the argmax, $T \to \infty$ the uniform distribution, and the entropy of the sampling distribution increases with $T$. Also used for calibration (temperature scaling, Module 15) and decoding (Module 17). → [Module 14](../part-5-learning-theory/14-information-theory/README.md)

**Thompson sampling** — Keep a posterior over each arm's mean, draw one sample from each posterior per round and pull the arm with the largest sample; with Bernoulli rewards and a Beta(1, 1) prior the posterior is Beta$(1 + s_a, 1 + f_a)$. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Top-k and top-p (nucleus) sampling** — Decoding rules that truncate the next-token distribution to the $k$ most probable tokens, or to the smallest set whose cumulative probability reaches $p$, and renormalise before sampling. → [Module 17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)

**Trace** — $\operatorname{tr}\mathbf{A} = \sum_iA_{ii} = \sum_i\lambda_i$, invariant under change of basis; the trace of a projector equals the dimension of the space it projects onto, which is why $\operatorname{tr}\mathbf{H} = p$ counts degrees of freedom. → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Transition matrix** — A row-stochastic matrix $\mathbf{P}$ with $P_{ij} = P(s_{t+1} = j \mid s_t = i)$ and rows summing to 1, which propagates a distribution over states as $\boldsymbol{\pi}_{t+1} = \boldsymbol{\pi}_t\mathbf{P}$. Its eigenvalues have modulus at most 1 (Module 03). → [Module 02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md)

**Type I and type II errors** — Rejecting a true $H_0$ (false positive, probability $\alpha$) and failing to reject a false $H_0$ (false negative, probability $\beta$); power is $1 - \beta$. → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

## U

**UCB1 (upper confidence bound)** — Pull $\arg\max_a\big[\hat\mu_a + \sqrt{2\ln t/N_a}\big]$: the empirical mean plus a Hoeffding-derived bonus that shrinks as an arm is sampled, so under-explored arms are tried optimistically. It achieves logarithmic regret. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Uniform distribution** — $\mathrm{Unif}(a, b)$ with density $1/(b - a)$ on $[a, b]$, mean $(a + b)/2$, variance $(b - a)^2/12$; the "no preference" distribution and the raw material every sampler transforms (Module 18). → [Module 08](../part-3-probability/08-distributions-and-limit-theorems/README.md)

**Union bound** — $P(\bigcup_iA_i) \le \sum_iP(A_i)$. It justifies the Bonferroni correction (Module 12) and turns a single-hypothesis concentration bound into one that holds for every hypothesis in a finite class (Module 15). → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Universal approximation theorem** — For every continuous function on a compact set and every $\varepsilon > 0$ there is a one-hidden-layer network with finitely many units within $\varepsilon$ of it everywhere. It says nothing about how many units are needed or whether gradient descent will find them. → [Module 16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md)

## V

**Value functions** — $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ and $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$, the expected return from a state (or state–action pair) under policy $\pi$; $V^*$ and $Q^*$ are their optima and $\pi^* = \arg\max_aQ^*$. → [Module 19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md)

**Vanishing and exploding gradients** — Gradients that shrink or grow geometrically with depth because the backward pass multiplies Jacobians, $\lVert\mathbf{G}_l\rVert \lesssim \lVert\mathbf{G}_L\rVert\prod_m\lVert\mathbf{W}_m\rVert_2\max\lvert\phi'\rvert$ (at most $4^{-L}$ for sigmoids). ReLU, careful initialisation, normalisation layers, residual connections and clipping each attack the product. → [Module 16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md)

**Variance** — $\operatorname{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - \mu^2$, with $\operatorname{Var}(aX + b) = a^2\operatorname{Var}(X)$ and $\operatorname{Var}(X + Y) = \operatorname{Var}X + \operatorname{Var}Y + 2\operatorname{Cov}(X, Y)$; the standard deviation is its square root, in the units of $X$. → [Module 07](../part-3-probability/07-probability-foundations/README.md)

**Variance propagation and initialisation** — For a layer with $n_{\mathrm{in}}$ independent inputs, $\operatorname{Var}(z) = n_{\mathrm{in}}\sigma_w^2\sigma_x^2$; keeping it constant through depth gives the LeCun ($\sigma_w^2 = 1/n_{\mathrm{in}}$), Xavier ($2/(n_{\mathrm{in}} + n_{\mathrm{out}})$) and He ($2/n_{\mathrm{in}}$, because ReLU halves the second moment) initialisations. → [Module 16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md)

**Variational autoencoder (VAE)** — A generative model trained by maximising the ELBO with an encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ and a decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$: loss $-\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] + \operatorname{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I}))$, with the Gaussian KL in closed form and one reparameterised sample for the expectation. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**Variational inference** — Approximating an intractable posterior $p(\mathbf{z} \mid \mathbf{x})$ by the member of a tractable family $q$ that maximises the ELBO, which is the same as minimising $\operatorname{KL}(q \,\|\, p(\mathbf{z} \mid \mathbf{x}))$; optimisation replaces sampling. → [Module 18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)

**VC dimension** — The largest number of points a hypothesis class can shatter: 1 for thresholds, 2 for intervals, 4 for axis-aligned rectangles, $d + 1$ for affine classifiers in $\mathbb{R}^d$, infinite for $\operatorname{sign}\sin(\omega x)$. Finite VC dimension is equivalent to PAC learnability, and the VC bound reads $R(h) \le \hat R_S(h) + \sqrt{2d\ln(en/d)/n} + \sqrt{\ln(1/\delta)/(2n)}$. → [Module 15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md)

**Vector** — An ordered list of numbers read as an arrow, a point or a data record in $\mathbb{R}^d$, combined by addition and scaling. Bold lower-case in this course; rows of the design matrix and embeddings are vectors. → [Module 01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)

**Vector–Jacobian product (VJP)** — $\mathbf{J}^\top\mathbf{g}$, what one reverse-mode step computes for each primitive without ever forming $\mathbf{J}$: for $\mathbf{Y} = \mathbf{A}\mathbf{B}$, $\bar{\mathbf{A}} = \mathbf{G}\mathbf{B}^\top$ and $\bar{\mathbf{B}} = \mathbf{A}^\top\mathbf{G}$; for element-wise $f$, $\bar{\mathbf{X}} = \mathbf{G}\odot f'(\mathbf{X})$; a forward broadcast becomes a backward sum. → [Module 06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md)

## W

**Wald and Wilson intervals** — Confidence intervals for a proportion: Wald, $\hat p \pm z\sqrt{\hat p(1 - \hat p)/n}$, under-covers for small $n$ or $p$ near 0 or 1; Wilson, which inverts the score test, keeps its nominal coverage and should be the default. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)

**Weight decay** — Adding $\lambda\boldsymbol{\theta}$ to the gradient (equivalent to an $L_2$ penalty under SGD) or subtracting $\eta\lambda\boldsymbol{\theta}$ directly from the parameters (decoupled, AdamW); the MAP view is a Gaussian prior with variance $1/\lambda$. → [Module 05](../part-2-calculus-and-optimization/05-optimization/README.md)

**Whitening** — Transforming a random vector to unit covariance, $\mathbf{Z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{X} - \boldsymbol{\mu})$; the correlation matrix $\mathbf{R} = \mathbf{D}^{-1/2}\boldsymbol{\Sigma}\mathbf{D}^{-1/2}$ is the partial step that only rescales. → [Module 09](../part-3-probability/09-multivariate-probability/README.md)

**Winner's curse** — Selecting the best of $m$ noisy scores inflates the winner: the expected maximum of $m$ standard normals is about $1.16$ for $m = 5$ and $2.25$ for $m = 50$, in units of the score's standard error. It is why the validation score of the chosen model is optimistic (Module 15) and why the best of several seeds overstates a model (Module 10). → [Module 12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md)

## Z

**z-test and z-interval** — Inference for a mean or proportion when the standard error is known or $n$ is large, using $z = (\bar x - \mu_0)/(\sigma/\sqrt n)$ and $\bar x \pm z_{\alpha/2}\sigma/\sqrt n$; the two-proportion z-test compares conversion rates in an A/B test. → [Module 10](../part-4-statistics/10-estimation-and-sampling/README.md)
