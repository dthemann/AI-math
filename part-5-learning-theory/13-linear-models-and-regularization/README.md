# Module 13 — Linear Models and Regularisation

> After this module you can fit, regularise, interpret and validate linear and generalised linear models from scratch, explain exactly why ridge shrinks and lasso selects, split a model's error into bias, variance and noise, and choose hyperparameters honestly with cross-validation.

## Why this matters for ML and AI

Linear models are the oldest tool in the box and still the most used. Every credit score that decides a loan, most insurance premiums, the dose-proportionality analysis in a drug's regulatory file and the baseline in almost every Kaggle competition are linear or generalised linear models. They are fast, they have closed forms you can audit, their coefficients mean something, and — most importantly for this course — every idea that makes modern deep learning work appears here first in a form small enough to prove: regularisation, the bias–variance trade-off, cross-validation, convex optimisation of a likelihood, and the gradient `p − y` that drives every classifier.

They are also *inside* the big models. The last layer of a neural network is a linear model (softmax regression) on learned features. Weight decay in SGD and AdamW is ridge regression. A "linear probe" that reads a concept out of a language model's hidden state is logistic regression. L1 penalties prune networks and train sparse autoencoders for interpretability. The Gauss–Markov theorem, the SVD view of ridge and the soft-thresholding operator are the clearest places to understand what all of that machinery actually does to parameters.

The module runs in a straight line: the supervised learning setup and empirical risk → ordinary least squares derived three ways → basis expansion and why it overfits → ridge (shrinkage) → lasso (sparsity) → logistic regression and generalised linear models (likelihood beyond the Gaussian) → the bias–variance decomposition (why regularisation helps) → cross-validation (how to choose how much) → diagnostics (how to tell when a fit is lying to you).

## Learning objectives

By the end of this module you will be able to:

- State the supervised learning problem precisely (hypothesis class, loss, true vs empirical risk) and explain why $E[Y\mid X]$ is the target of regression.
- Derive the OLS solution by geometry, calculus and maximum likelihood; compute the hat matrix, residuals, $R^2$ and coefficient standard errors by hand on a small dataset.
- Build polynomial, interaction and one-hot features, standardise correctly, and explain why a degree-9 polynomial through 10 points is useless.
- Derive ridge regression's closed form, its SVD filter factors $d_j^2/(d_j^2+\lambda)$ and its Bayesian (Gaussian-prior) interpretation; compute effective degrees of freedom.
- Derive the soft-thresholding operator, implement lasso by coordinate descent, and explain geometrically why L1 produces exact zeros while L2 does not.
- Fit logistic and Poisson regression by Newton's method / IRLS from scratch, interpret odds ratios and rate ratios, and explain why separable data makes the unpenalised MLE diverge.
- Prove the bias–variance decomposition for squared loss and measure its terms by simulation.
- Implement k-fold and leave-one-out cross-validation (including the hat-matrix shortcut), explain why nested cross-validation is needed, and read a learning curve.
- Diagnose heteroscedasticity, high leverage, multicollinearity (VIF), extrapolation and leakage in a fitted model.

## Prerequisites

- [Module 02 — Matrices and linear maps](../../part-1-linear-algebra/02-matrices-and-linear-maps/README.md): least squares and the normal equations, projections, column space, rank.
- [Module 03 — Eigenvalues, SVD and PCA](../../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md): the SVD, the pseudo-inverse, condition number, ridge filter factors (previewed there).
- [Module 04 — Differential calculus](../../part-2-calculus-and-optimization/04-differential-calculus/README.md) and [Module 05 — Optimisation](../../part-2-calculus-and-optimization/05-optimization/README.md): gradients, Hessians, convexity, gradient descent, Newton's method.
- [Module 09 — Multivariate probability](../../part-3-probability/09-multivariate-probability/README.md): conditional expectation, covariance, the multivariate Gaussian.
- [Module 10 — Estimation and sampling](../../part-4-statistics/10-estimation-and-sampling/README.md): bias, variance and MSE of an estimator, standard errors, the bootstrap.
- [Module 11 — Likelihood and Bayesian inference](../../part-4-statistics/11-likelihood-and-bayesian-inference/README.md): MLE, MAP, Gaussian likelihood ⇒ squared loss, Gaussian prior ⇒ L2 penalty.
- [Module 12 — Hypothesis testing and experiments](../../part-4-statistics/12-hypothesis-testing-and-experiments/README.md): t-tests, comparing models properly.

Recall in particular that the projection of $y$ onto the column space of $X$ is $X(X^\top X)^{-1}X^\top y$, that a convex function has a unique minimum where its gradient vanishes, and that maximising a Gaussian log-likelihood is the same as minimising a sum of squares.

## 1. The supervised learning setup

**Intuition.** You are shown pairs (input, answer) drawn from some unknown process and must produce a rule that will give good answers on *new* inputs from the same process. Three choices define the game: which rules you are allowed to pick from (the hypothesis class), how you score a wrong answer (the loss), and how you estimate your score before the new inputs arrive (empirical risk on the data you have — a stand-in for the true risk you cannot see). Almost everything that can go wrong in machine learning is a gap between those two risks.

**Definition.** Data $(x_i, y_i)$, $i = 1,\dots,n$, are drawn independently from a joint distribution $P$ over inputs $x \in \mathbb{R}^d$ and targets $y$. A **hypothesis class** $\mathcal{H}$ is a set of functions $f$; a **loss** $\ell(\hat y, y) \ge 0$ scores a prediction. The **true risk** and the **empirical risk** of $f$ are

$$R(f) = \mathbb{E}_{(x,y)\sim P}\,[\ell(f(x), y)], \qquad \hat R_n(f) = \frac1n\sum_{i=1}^n \ell(f(x_i), y_i).$$

**Empirical risk minimisation** (ERM) picks $\hat f = \arg\min_{f\in\mathcal{H}} \hat R_n(f)$. For squared loss the best possible predictor over *all* functions is the **regression function** $f^*(x) = \mathbb{E}[Y \mid X = x]$ (Module 09): for any $f$,

$$\mathbb{E}[(Y - f(X))^2] = \mathbb{E}[(Y - f^*(X))^2] + \mathbb{E}[(f^*(X) - f(X))^2],$$

because the cross term $\mathbb{E}[(Y - f^*(X))(f^*(X) - f(X))]$ vanishes once you condition on $X$. The first term, $\sigma^2 = \mathbb{E}[\operatorname{Var}(Y\mid X)]$, is **irreducible noise**; the second is how far your $f$ is from the truth. The **linear hypothesis class** is $f_\beta(x) = \beta^\top x$ with a constant feature $x_0 = 1$ carrying the intercept. Because $\hat R_n(\hat f)$ is minimised over the same data it is evaluated on, it is optimistic; you therefore split the data into a **training set** (fit), a **validation set** (choose between models) and a **test set** (one final honest estimate of $R$).

**Worked example.** Five pairs: $x = (1, 2, 3, 4, 5)$, $y = (2.1, 3.9, 6.2, 7.8, 10.1)$. Consider the hypothesis $f(x) = 2x$. Residuals $y_i - f(x_i)$: $0.1, -0.1, 0.2, -0.2, 0.1$; squared: $0.01, 0.01, 0.04, 0.04, 0.01$; empirical risk under squared loss $\hat R_5 = 0.11/5 = 0.022$. The line $f(x) = 2x + 0.02$ does marginally better ($0.0216$); ERM over all lines finds the exact minimiser (next section). The *true* risk of either line is unknowable from these five points — only fresh data can estimate it.

**Real-life example (e-commerce).** A retailer predicts daily units sold of a product from price, a promotion flag, day of week and a trend. Hypothesis class: linear in those features. Loss: squared error in units (or absolute error, if a few blockbuster days should not dominate). Training set: January–September; validation: October, used to pick between feature sets; test: November, touched once. A linear model is the right first model here not because demand is linear but because it fits in seconds, its coefficients ("a 10 % price cut adds 38 units/day") can be sanity-checked by the category manager, and it sets the bar a gradient-boosted model must beat.

**In ML/AI this is …** the training loop itself: `loss.backward()` on a mini-batch is a stochastic estimate of the gradient of $\hat R_n$; the validation loss watched for early stopping is an estimate of $R$. Every neural network ends in a linear model — the output layer $\hat y = W h + b$ is linear regression (or logistic regression) on the learned features $h$ — and "linear probes" in interpretability research are literally logistic regression on a model's hidden states. In scikit-learn the class is `sklearn.linear_model`.

## 2. Ordinary least squares, three ways

**Intuition.** With $n$ observations and $p$ coefficients, the vector of predictions $X\beta$ can only reach the $p$-dimensional column space of $X$ inside $\mathbb{R}^n$. The target $y$ usually lies outside it. Least squares picks the point of that subspace closest to $y$ — the orthogonal projection — and the residual is the perpendicular that is left over. The same solution falls out of calculus and of a Gaussian likelihood; seeing all three removes any mystery about why squared error is the default.

**Definition.** With $X \in \mathbb{R}^{n\times p}$ (full column rank) and $y \in \mathbb{R}^n$, **ordinary least squares** (OLS) minimises $\operatorname{RSS}(\beta) = \|y - X\beta\|^2$.

*Geometry.* The residual $e = y - X\hat\beta$ must be orthogonal to every column: $X^\top e = 0$, i.e. the **normal equations** $X^\top X\hat\beta = X^\top y$, so $\hat\beta = (X^\top X)^{-1}X^\top y$ (Module 02).

*Calculus.* $\nabla_\beta \operatorname{RSS} = -2X^\top(y - X\beta)$; setting it to zero gives the same equations, and the Hessian $2X^\top X$ is positive definite, so this is the unique minimum (Module 04).

*Probability.* If $y_i = \beta^\top x_i + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d., the log-likelihood is $-\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|y - X\beta\|^2$; maximising over $\beta$ is minimising RSS, and maximising over $\sigma^2$ gives $\hat\sigma^2_{\text{MLE}} = \operatorname{RSS}/n$ (Module 11).

The **hat matrix** $H = X(X^\top X)^{-1}X^\top$ maps $y$ to fitted values $\hat y = Hy$; it is a projection ($H^2 = H = H^\top$) with $\operatorname{tr}(H) = p$, and $e = (I - H)y$. Its diagonal entries $h_{ii}$ are the **leverages**. Goodness of fit: $R^2 = 1 - \operatorname{RSS}/\operatorname{TSS}$ with $\operatorname{TSS} = \sum_i(y_i - \bar y)^2$, the fraction of variance explained. **Gauss–Markov theorem:** if $\mathbb{E}[\varepsilon] = 0$ and $\operatorname{Var}(\varepsilon) = \sigma^2 I$ (uncorrelated, equal variance — no normality needed), then among all *linear unbiased* estimators OLS has the smallest variance (it is **BLUE**), with $\operatorname{Var}(\hat\beta) = \sigma^2(X^\top X)^{-1}$. The unbiased noise estimate is $\hat\sigma^2 = \operatorname{RSS}/(n - p)$. Adding normality of $\varepsilon$ makes $\hat\beta_j/\operatorname{SE}(\hat\beta_j) \sim t_{n-p}$ under $\beta_j = 0$, which is the t-test printed by every regression package (Module 12).

**Worked example.** $x = (1,2,3,4,5)$, $y = (2,3,5,4,6)$, model $y = \beta_0 + \beta_1 x$. Then

$$X^\top X = \begin{pmatrix} 5 & 15 \\ 15 & 55\end{pmatrix}, \qquad X^\top y = \begin{pmatrix} 20 \\ 69 \end{pmatrix}, \qquad \hat\beta = \frac{1}{5\cdot 55 - 15^2}\begin{pmatrix} 55 & -15 \\ -15 & 5\end{pmatrix}\begin{pmatrix} 20 \\ 69\end{pmatrix} = \begin{pmatrix} 1.3 \\ 0.9 \end{pmatrix}.$$

Fitted values $2.2, 3.1, 4.0, 4.9, 5.8$; residuals $-0.2, -0.1, 1.0, -0.9, 0.2$. Check the geometry: $\sum e_i = 0$ and $\sum x_i e_i = -0.2 - 0.2 + 3.0 - 3.6 + 1.0 = 0$ — the residual is orthogonal to both columns. $\operatorname{RSS} = 0.04 + 0.01 + 1 + 0.81 + 0.04 = 1.9$, $\operatorname{TSS} = 10$, so $R^2 = 0.81$. Leverages $h_{ii} = \frac15 + \frac{(x_i - 3)^2}{10}$: $0.6, 0.3, 0.2, 0.3, 0.6$, summing to $2 = p$. Noise $\hat\sigma^2 = 1.9/3 = 0.633$; $\operatorname{SE}(\hat\beta_1) = \sqrt{0.633/10} = 0.252$; $t = 0.9/0.252 = 3.58$ on 3 degrees of freedom, two-sided $p \approx 0.037$.

**Real-life example (clinical pharmacology).** A phase-I study gives five dose levels of a new drug and measures exposure (the area under the plasma-concentration curve, AUC). "Dose proportionality" — AUC linear in dose through the origin — is a regulatory question answered by exactly this regression: the slope estimates exposure per milligram, the intercept should be indistinguishable from zero, and the confidence interval of the slope is what the submission reports. An $R^2$ of 0.98 across 40 subjects is typical when kinetics are linear; a curved residual pattern at high doses is the fingerprint of saturated clearance and sends the team to a non-linear model.

**In ML/AI this is …** `sklearn.linear_model.LinearRegression`, which calls `scipy.linalg.lstsq` (an SVD solve, Module 03) rather than inverting $X^\top X$; the mean-squared-error loss, which is the Gaussian negative log-likelihood; and gradient descent on MSE (Module 05), which converges to the same $\hat\beta$ — the normal equations are a single Newton step on a quadratic. The hat matrix returns in Section 9 as the trick behind fast leave-one-out cross-validation.

## 3. Feature engineering and basis expansion

**Intuition.** "Linear model" means linear in the *coefficients*, not in the raw inputs. Replace $x$ by any fixed set of transformed features $\phi(x)$ — powers, products, indicator variables — and the fitting machinery is unchanged while the fitted curve can bend. The price is that the more features you add, the more ways the model has to fit the noise, and some features (powers of a large number) are numerically poisonous unless you rescale.

**Definition.** A **basis expansion** fits $f(x) = \beta^\top\phi(x)$ with $\phi:\mathbb{R}^d \to \mathbb{R}^p$. Common choices: **polynomial features** $\phi(x) = (1, x, x^2, \dots, x^k)$ (the design matrix is a Vandermonde matrix); **interactions** $x_1 x_2$, so that the effect of $x_1$, $\partial f/\partial x_1 = \beta_1 + \beta_3 x_2$, depends on $x_2$; **one-hot encoding** of a categorical variable with $K$ levels into $K - 1$ indicator columns when an intercept is present (the omitted level is the baseline, and including all $K$ makes the columns sum to the intercept column — exact collinearity, the "dummy variable trap"); **splines**, piecewise polynomials joined smoothly at knots, which bend locally without the global wiggle of high-degree polynomials. **Standardisation** replaces each column by $z_j = (x_j - \bar x_j)/s_j$. It matters for two reasons: numerically, powers of raw values give wildly different column scales and a huge condition number (Module 03); statistically, a penalty $\lambda\|\beta\|^2$ treats all coefficients alike, but a coefficient on height-in-millimetres is 1,000 times smaller than the same effect in metres and so is penalised a million times less — regularisation without standardisation penalises *units*, not effects. **Overfitting by expansion:** a polynomial of degree $n - 1$ through $n$ distinct points interpolates them exactly (the $n\times n$ Vandermonde matrix is invertible), so a degree-9 polynomial on 10 points has zero training error and wild oscillations between the points.

**Worked example.** Standardise $x = (50, 70, 90, 110, 130)$: mean $90$, population standard deviation $\sqrt{(1600+400+0+400+1600)/5} = 28.28$, so $z = (-1.41, -0.71, 0, 0.71, 1.41)$. One-hot: a machine variable with levels {A, B, C} and an intercept becomes columns $(1, [\text{B}], [\text{C}])$; the coefficient on $[\text{B}]$ is the mean difference between B and the baseline A. Conditioning: the cubic design matrix $(1, x, x^2, x^3)$ on $x = 0, 1, \dots, 10$ has condition number $1{,}617$; on the standardised $z$ it is $6.1$. Same model, same fitted curve, 260 times less error amplification.

**Real-life example (manufacturing).** An injection-moulding line wants to predict part yield from melt temperature $T$, holding pressure $P$ and the machine (one of four). Yield peaks at an intermediate temperature, so include $T^2$; the best pressure depends on the temperature, so include $T\times P$; one-hot the machine with three indicators. Standardising $T$ and $P$ first turns "$T^2$" from numbers around $50{,}000$ into numbers around $1$, and makes the coefficient on $T\times P$ readable as "the pressure effect changes by this much per one-standard-deviation change in temperature". The fitted quadratic directly yields the optimal set-point, which is the whole point of the exercise.

**In ML/AI this is …** `PolynomialFeatures`, `OneHotEncoder`, `SplineTransformer` and `StandardScaler`, chained in a `Pipeline` so that the scaling statistics are learned on training folds only (Section 10). Kernel methods are basis expansion taken to infinite dimension; a neural network *learns* $\phi$ instead of having it specified; an embedding layer is a learned replacement for one-hot encoding; batch normalisation is standardisation applied inside the network (Module 16).

## 4. Ridge regression

**Intuition.** When two columns of $X$ are nearly parallel, the data cannot tell their coefficients apart: a large positive on one and a large negative on the other fit almost as well as two moderate values. OLS, being unbiased, reports whichever extreme the noise favours — enormous coefficients with opposite signs. Ridge adds a small cost for coefficient size. It gives up a little bias in exchange for a large drop in variance, and the SVD shows precisely which directions it tames: the ones the data barely constrain.

**Definition.** With $X$ and $y$ centred (so the intercept is $\bar y$ and is not penalised),

$$\hat\beta_\lambda = \arg\min_\beta\ \|y - X\beta\|^2 + \lambda\|\beta\|^2 = (X^\top X + \lambda I)^{-1}X^\top y, \qquad \lambda > 0 .$$

*SVD view.* Write the thin SVD $X = UDV^\top$ with singular values $d_1 \ge \dots \ge d_p$. Then $X^\top X + \lambda I = V(D^2 + \lambda I)V^\top$ and $X^\top y = VDU^\top y$, so

$$\hat\beta_\lambda = V(D^2 + \lambda I)^{-1}DU^\top y = \sum_{j=1}^{p}\frac{d_j}{d_j^2 + \lambda}\,\langle u_j, y\rangle\, v_j, \qquad \hat\beta_{\text{OLS}} = \sum_{j=1}^{p}\frac{1}{d_j}\langle u_j, y\rangle\, v_j .$$

Along each principal direction $v_j$ the OLS coefficient is multiplied by the **filter factor** $d_j^2/(d_j^2 + \lambda)$: directions with $d_j^2 \gg \lambda$ pass almost untouched, directions with $d_j^2 \ll \lambda$ — the low-variance, noise-amplifying ones — are shrunk towards zero. Ridge is a smooth version of the pseudo-inverse's hard cut-off (Module 03). The fitted values are $\hat y = H_\lambda y$ with $H_\lambda = X(X^\top X + \lambda I)^{-1}X^\top$, and the **effective degrees of freedom** are $\operatorname{df}(\lambda) = \operatorname{tr}(H_\lambda) = \sum_j d_j^2/(d_j^2 + \lambda)$, falling from $p$ at $\lambda = 0$ to $0$ as $\lambda \to \infty$. *Bayesian view* (Module 11): with prior $\beta \sim \mathcal{N}(0, \tau^2 I)$ and noise $\mathcal{N}(0, \sigma^2)$, the log-posterior is $-\frac{1}{2\sigma^2}\|y - X\beta\|^2 - \frac{1}{2\tau^2}\|\beta\|^2 + \text{const}$, so the MAP estimate is ridge with $\lambda = \sigma^2/\tau^2$: a strong penalty is a narrow prior. $\lambda$ is chosen by cross-validation (Section 9).

**Worked example.** Two standardised features with correlation $0.9$, scaled so that $X^\top X = \begin{pmatrix} 1 & 0.9 \\ 0.9 & 1\end{pmatrix}$, and $X^\top y = (1, 0.8)$. OLS: $\hat\beta = \frac{1}{0.19}\begin{pmatrix} 1 & -0.9 \\ -0.9 & 1\end{pmatrix}\begin{pmatrix} 1 \\ 0.8\end{pmatrix} = (1.474, -0.526)$ — opposite signs although both features correlate *positively* with $y$, the classic collinearity symptom. The eigenvalues of $X^\top X$ are $1.9$ along $(1,1)/\sqrt2$ and $0.1$ along $(1,-1)/\sqrt2$. With $\lambda = 0.5$ the filter factors are $1.9/2.4 = 0.79$ and $0.1/0.6 = 0.17$: the "shared" direction is kept, the "difference" direction is mostly removed, and $\hat\beta_{0.5} = (0.542, 0.208)$, both positive, with norm $0.58$ against $1.56$ for OLS and $\operatorname{df}(0.5) = 0.96$ instead of $2$. Even $\lambda = 0.1$ gives $(0.95, -0.05)$. In the Bayesian reading, $\lambda = 0.5$ corresponds to $\sigma = 1$ and a prior standard deviation $\tau = \sqrt2$ on each coefficient.

**Real-life example (climate).** Statistical downscaling predicts the temperature at one weather station from the output of a coarse climate model on a few hundred surrounding grid cells. Neighbouring cells are correlated at 0.95 or more, so OLS produces coefficient maps that look like noise — large positive and negative weights on adjacent cells that cancel. Ridge with $\lambda$ chosen by cross-validation yields a smooth, physically plausible weight map concentrated on the cells upwind of the station and typically cuts out-of-sample error by a third. The same situation — many correlated predictors, few observations — occurs in gene-expression studies (20,000 genes, 100 patients, where OLS is not even defined because $p > n$) and in marketing-mix models where TV, radio and search spend rise and fall together with campaigns.

**In ML/AI this is …** **weight decay**: the SGD update $w \leftarrow w - \eta(\nabla L(w) + \lambda w)$ is gradient descent on the ridge objective, and AdamW's "decoupled" weight decay is the same idea applied after the adaptive step. It is `sklearn.linear_model.Ridge(alpha=λ)`, Tikhonov regularisation in inverse problems, the $\lambda I$ that makes Gaussian-process and kernel ridge regression solvable, and the Gaussian-prior-equals-L2 correspondence of Module 11.

## 5. Lasso and sparsity

**Intuition.** Picture the contours of the residual sum of squares — ellipses centred on the OLS solution — expanding until they first touch the constraint region $\{\|\beta\|_1 \le t\}$. That region is a diamond (in 2-D) with corners on the axes, and an expanding ellipse hits a corner far more often than a flat edge. At a corner, some coefficients are *exactly* zero. The L2 ball is round and has no corners, so ridge shrinks every coefficient but almost never zeroes one. The lasso therefore fits and selects variables in one step.

**Definition.** The **lasso** minimises

$$\tfrac12\|y - X\beta\|^2 + \lambda\|\beta\|_1, \qquad \|\beta\|_1 = \sum_j|\beta_j| .$$

The objective is convex but not differentiable at zero, so there is no closed form in general. In the **orthonormal case** $X^\top X = I$ the problem separates: with $\hat\beta^{\text{ols}}_j = x_j^\top y$, each coordinate minimises $\frac12(\hat\beta^{\text{ols}}_j - \beta_j)^2 + \lambda|\beta_j|$. For $\beta_j > 0$ the derivative $\beta_j - \hat\beta^{\text{ols}}_j + \lambda = 0$ gives $\beta_j = \hat\beta^{\text{ols}}_j - \lambda$, valid only if $\hat\beta^{\text{ols}}_j > \lambda$; symmetrically for $\beta_j < 0$; otherwise the minimum sits at the kink. The result is the **soft-thresholding operator**

$$\hat\beta_j = S_\lambda(\hat\beta^{\text{ols}}_j), \qquad S_\lambda(z) = \operatorname{sign}(z)\max(|z| - \lambda, 0),$$

compared with ridge's proportional shrinkage $\hat\beta^{\text{ols}}_j/(1 + \lambda)$ in the same setting. For general $X$, **coordinate descent** cycles through $j = 1,\dots,p$, each time solving the one-dimensional problem exactly with the partial residual $r^{(j)} = y - \sum_{k\neq j}x_k\beta_k$:

$$\beta_j \leftarrow \frac{S_\lambda\!\left(x_j^\top r^{(j)}\right)}{\|x_j\|^2}.$$

Each update is cheap, the objective never increases, and because the non-smooth part is separable the method converges to the global minimum. The **elastic net** adds a ridge term, $\lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2$, which keeps groups of correlated predictors together instead of picking one. Limits of the lasso: among strongly correlated predictors it selects one essentially at random and the choice flips between bootstrap samples; with $p > n$ it selects at most $n$ variables; and because it shrinks the survivors as well as zeroing the rest, the non-zero coefficients are biased towards zero (refitting OLS on the selected set, the "relaxed lasso", fixes this). Convention: scikit-learn's `Lasso(alpha)` minimises $\frac{1}{2n}\|y - X\beta\|^2 + \alpha\|\beta\|_1$, so $\lambda = n\alpha$.

**Worked example.** Orthonormal design, OLS coefficients $(3, -0.5, 1.2)$, $\lambda = 1$: soft-thresholding gives $(2, 0, 0.2)$ — one coefficient killed, the others moved towards zero by exactly $1$. Ridge with $\lambda = 1$ gives $(1.5, -0.25, 0.6)$: everything halved, nothing zero. Now the correlated design of Section 4, $X^\top X = \begin{pmatrix} 1 & 0.9 \\ 0.9 & 1\end{pmatrix}$, $X^\top y = (1, 0.8)$, with $\lambda = 0.3$ and start $\beta = 0$. Sweep 1: $j = 1$: $x_1^\top r^{(1)} = 1 - 0.9\cdot 0 = 1$, so $\beta_1 = S_{0.3}(1) = 0.7$; $j = 2$: $x_2^\top r^{(2)} = 0.8 - 0.9\cdot 0.7 = 0.17$, and $S_{0.3}(0.17) = 0$. Sweep 2 changes nothing: converged at $(0.7, 0)$. Optimality check: the gradient of the smooth part is $X^\top X\beta - X^\top y = (-0.3, -0.17)$; for the active coordinate it equals $-\lambda\operatorname{sign}(\beta_1)$ ✓, and for the inactive one $|{-0.17}| \le 0.3$ ✓. The lasso has kept the stronger of two near-duplicates and discarded the other.

**Real-life example (NLP).** A spam filter represents each e-mail by counts of 200,000 distinct words and word pairs, with 50,000 labelled messages — more features than examples. L1-penalised logistic regression selects a few thousand informative terms ("unsubscribe", "wire transfer", the sender's domain) and assigns exact zeros to the rest, which makes the classifier tiny, fast and auditable: you can print the 50 most positive and most negative words. Correlated groups are visible here too — "viagra" and "v1agra" share the weight arbitrarily — which is one reason the elastic net is popular for text.

**In ML/AI this is …** `Lasso`, `ElasticNet`, `LogisticRegression(l1_ratio=1)`; the proximal-gradient methods ISTA/FISTA, where a gradient step is followed by soft-thresholding (the proximal operator of the L1 norm); L1 pruning of neural-network weights; compressed sensing; and the L1 penalty on latent activations in sparse autoencoders used to read interpretable features out of language models.

## 6. Logistic regression as a GLM

**Intuition.** For a yes/no target, fitting a line to the 0/1 labels predicts probabilities below zero and above one. Instead, let a linear score $\beta^\top x$ set the *log-odds*, squash it through the sigmoid to get a probability, and fit by maximum likelihood. The decision boundary — where the probability is one half — is the hyperplane $\beta^\top x = 0$, so the classifier is still linear; only the loss has changed.

**Definition.** Model $Y\mid x \sim \text{Bernoulli}(p(x))$ with

$$p(x) = \sigma(\beta^\top x) = \frac{1}{1 + e^{-\beta^\top x}}, \qquad \log\frac{p(x)}{1 - p(x)} = \beta^\top x .$$

The negative log-likelihood (the **cross-entropy** or log-loss) is $\operatorname{NLL}(\beta) = \sum_{i=1}^n\left[\log\!\left(1 + e^{\beta^\top x_i}\right) - y_i\,\beta^\top x_i\right]$, with

$$\nabla\operatorname{NLL} = X^\top(p - y), \qquad \nabla^2\operatorname{NLL} = X^\top W X, \quad W = \operatorname{diag}\big(p_i(1 - p_i)\big).$$

$W \succeq 0$ makes the Hessian positive semidefinite, so the problem is convex, but $\nabla\operatorname{NLL} = 0$ has no closed-form solution. **Newton's method** iterates $\beta \leftarrow \beta + (X^\top W X)^{-1}X^\top(y - p)$; rewriting it as $\beta \leftarrow (X^\top W X)^{-1}X^\top W z$ with the working response $z = X\beta + W^{-1}(y - p)$ shows each step is a weighted least-squares fit, hence the name **iteratively reweighted least squares** (IRLS). Gradient descent on $\ell$ also works (slower, but it scales). **Separable data:** if some $\beta$ classifies every training point correctly, then scaling it by $c \to \infty$ drives every $p_i$ to its label and $\ell \to 0$ without ever reaching it — the MLE does not exist and the coefficients diverge. The cure is a penalty, $\ell(\beta) + \frac{\lambda}{2}\|\beta\|^2$, which also stabilises nearly separable problems; scikit-learn regularises by default. **Interpretation:** $e^{\beta_j}$ is the **odds ratio** — the factor by which the odds multiply per unit increase in $x_j$. **Calibration:** among all cases predicted at $0.3$, about 30 % should be positive; logistic regression fitted by maximum likelihood is calibrated on its training distribution because the score equations $X^\top(y - p) = 0$ with an intercept force $\sum p_i = \sum y_i$. **Multinomial** logistic regression uses the softmax $p_k(x) = e^{\beta_k^\top x}/\sum_l e^{\beta_l^\top x}$; its gradient is again $X^\top(P - Y)$ with one-hot $Y$.

**Worked example.** $x = (-2, -1, 0, 1, 2)$, $y = (0, 1, 0, 1, 1)$, model $\sigma(\beta_0 + \beta_1 x)$. Start at $\beta = (0, 0)$, so every $p_i = 0.5$. Gradient of the *log*-likelihood: $X^\top(y - p) = \big(\sum(y_i - 0.5),\ \sum x_i(y_i - 0.5)\big) = (0.5, 2.0)$. $W = 0.25\,I$, so $X^\top W X = 0.25\begin{pmatrix} 5 & 0 \\ 0 & 10\end{pmatrix} = \begin{pmatrix} 1.25 & 0 \\ 0 & 2.5\end{pmatrix}$, and the Newton step is $(0.5/1.25,\ 2.0/2.5) = (0.4, 0.8)$. Five more steps converge to $\hat\beta = (0.623, 1.090)$, raising the log-likelihood from $5\ln 0.5 = -3.47$ to $-2.42$. The decision boundary is at $x = -0.623/1.090 = -0.57$; the odds ratio per unit of $x$ is $e^{1.090} = 2.98$; at $x = 0$ the predicted probability is $\sigma(0.623) = 0.65$. Had the labels been $(0, 0, 0, 1, 1)$ the data would be separable and the iterations would never settle.

**Real-life example (healthcare).** A ten-year cardiovascular risk score is a logistic regression on age, sex, systolic blood pressure, cholesterol, smoking and diabetes. Suppose the fitted log-odds for a given patient profile is $-2.0$ (a risk of $\sigma(-2.0) = 11.9\,\%$) and the smoking coefficient is $0.7$: the same patient as a smoker has log-odds $-1.3$ and risk $21.4\,\%$, an odds ratio of $e^{0.7} = 2.0$. Clinicians use the model precisely because each coefficient is a published, checkable quantity and because treatment thresholds ("statin above 10 %") need calibrated probabilities, not just a ranking.

**In ML/AI this is …** the output layer of every binary classifier (`BCEWithLogitsLoss` in PyTorch) and, via softmax, every multi-class one; the gradient $p - y$ with respect to the logits is the first step of back-propagation in Module 16. `sklearn.linear_model.LogisticRegression` minimises the L2-penalised loss with L-BFGS, with $C = 1/\lambda$. Separability is the reason weight decay or early stopping is needed even for a linear classifier on linearly separable embeddings, and the calibration property is what temperature scaling (Module 14) restores in over-confident deep networks.

## 7. Generalised linear models

**Intuition.** Linear regression, logistic regression and Poisson regression are one model with three dials: a distribution for the target given the inputs, a linear predictor $\eta = \beta^\top x$, and a **link** function that connects the distribution's mean to $\eta$. Choosing the distribution to match the data type (continuous, binary, count) and the link to keep the mean in its legal range (positive counts need an exponential) gives a family of models that all share the same fitting algorithm and the same gradient.

**Definition.** $Y\mid x$ follows a one-parameter **exponential family** $p(y\mid\theta) = h(y)\exp\big(\theta y - A(\theta)\big)$, whose mean and variance are $\mu = A'(\theta)$ and $\operatorname{Var}(Y) = A''(\theta)$. A **GLM** sets $g(\mu) = \eta = \beta^\top x$ for a monotone link $g$. The **canonical link** is $g = (A')^{-1}$, which makes $\theta = \eta$:

| Distribution | $A(\theta)$ | Mean $\mu$ | Canonical link | Variance |
|---|---|---|---|---|
| Gaussian | $\theta^2/2$ | $\theta$ | identity, $\mu = \eta$ | constant $\sigma^2$ |
| Bernoulli | $\log(1 + e^\theta)$ | $\sigma(\theta)$ | logit, $\log\frac{\mu}{1-\mu} = \eta$ | $\mu(1-\mu)$ |
| Poisson | $e^\theta$ | $e^\theta$ | log, $\log\mu = \eta$ | $\mu$ |

With the canonical link the log-likelihood is $\sum_i[y_i\eta_i - A(\eta_i)]$, its gradient is $X^\top(y - \mu)$ for *every* family, and the Hessian is $-X^\top W X$ with $W_{ii} = \operatorname{Var}(y_i)$; IRLS is the same algorithm with a different $W$. The **deviance** $D = 2[\ell_{\text{sat}} - \ell(\hat\beta)]$ compares the fit to the saturated model that fits every point exactly: for the Gaussian it is RSS$/\sigma^2$; for the Poisson

$$D = 2\sum_i\left[y_i\log\frac{y_i}{\hat\mu_i} - (y_i - \hat\mu_i)\right]$$

(with $0\log 0 = 0$), and for a well-specified model it is roughly $\chi^2_{n-p}$. Counts observed over different **exposures** $t_i$ use an **offset**: $\log\mu_i = \log t_i + \beta^\top x_i$, modelling the rate. When counts are more variable than Poisson allows (**overdispersion**, $\operatorname{Var} > \mu$), switch to a negative-binomial or quasi-Poisson model; the point estimates barely change but the standard errors do.

**Worked example.** Poisson regression with $\log\mu = -2 + 0.5x$. At $x = 2$, $\mu = e^{-1} = 0.368$; at $x = 4$, $\mu = e^0 = 1$. Each unit of $x$ multiplies the expected count by $e^{0.5} = 1.65$ — the **rate ratio**. Deviance for observed counts $y = (2, 0, 3)$ with fitted means $\hat\mu = (1.5, 0.5, 2.5)$: the terms are $2[2\ln(2/1.5) - 0.5] = 0.151$, $2[0 - (0 - 0.5)] = 1.000$ and $2[3\ln(3/2.5) - 0.5] = 0.094$, so $D = 1.245$ — small relative to the $n - p$ you would compare it with, i.e. the three points are fitted about as well as Poisson noise allows.

**Real-life example (healthcare operations).** A hospital models daily emergency admissions with a Poisson GLM: log link, day of week and public holidays one-hot, temperature and a flu-season indicator, and $\log(\text{catchment population})$ as an offset so the coefficients describe rates per inhabitant. A coefficient of $0.18$ on "Monday" is a rate ratio of $e^{0.18} = 1.20$ — 20 % more admissions than the baseline day; a flu-season coefficient of $0.35$ means a factor $1.42$. The staffing plan is built from the predicted mean, and the deviance over degrees of freedom (often 1.5–2 for admissions, signalling overdispersion) tells the planner to widen the uncertainty bands.

**In ML/AI this is …** `PoissonRegressor`, `GammaRegressor` and `TweedieRegressor` in scikit-learn and `GLM` in statsmodels. Deeper: a neural network's output activation is the inverse canonical link (identity, sigmoid, exp, softmax) and its loss is the matching negative log-likelihood, which is why the gradient at the pre-activation is always $\hat\mu - y$ — for MSE, for binary cross-entropy, for Poisson NLL and for softmax cross-entropy alike. A GLM is a neural network with no hidden layer.

## 8. Bias–variance decomposition

**Intuition.** Imagine fitting your model not once but on a thousand independent training sets of the same size and looking at the thousand predictions at a fixed input. Their average may miss the truth (bias — the model is too simple to represent it) and they may scatter widely around that average (variance — the model is so flexible it follows each training set's noise). Expected squared error is bias squared plus variance plus the noise nobody can remove. Simple models have high bias and low variance; flexible ones the reverse; regularisation is a knob between them.

**Definition and derivation.** Let $y = f(x) + \varepsilon$ with $\mathbb{E}[\varepsilon] = 0$, $\operatorname{Var}(\varepsilon) = \sigma^2$, and let $\hat f$ be fitted on a random training set $D$ independent of the test noise. Write $\bar f(x) = \mathbb{E}_D[\hat f(x)]$. At a fixed $x$,

$$\mathbb{E}_{D,\varepsilon}\big[(y - \hat f(x))^2\big] = \mathbb{E}\big[(\varepsilon + (f - \bar f) + (\bar f - \hat f))^2\big] = \sigma^2 + (f(x) - \bar f(x))^2 + \mathbb{E}_D\big[(\hat f(x) - \bar f(x))^2\big],$$

because all three cross terms vanish: $\varepsilon$ is independent of everything else and has mean zero, and $\mathbb{E}_D[\hat f - \bar f] = 0$. In words,

$$\text{expected squared error} = \text{noise} + \text{bias}^2 + \text{variance}.$$

*How complexity moves the terms.* For OLS with $p$ features and the design fixed, $\operatorname{Var}(\hat y_i) = \sigma^2 h_{ii}$, so the variance averaged over the training inputs is $\sigma^2\operatorname{tr}(H)/n = \sigma^2 p/n$: it grows linearly with the number of parameters and shrinks with the sample size. Bias falls as $p$ grows (a richer basis approximates $f$ better). Their sum is the classical **U-curve** of test error against complexity. For ridge, increasing $\lambda$ raises bias and lowers variance (the effective $\operatorname{df}(\lambda)$ replaces $p$); for $k$-nearest neighbours the variance is $\sigma^2/k$ while bias grows with $k$ as the neighbourhood widens. The U-curve is the small-model story; for very over-parameterised models the test error can fall *again* past the interpolation threshold — **double descent**, taken up in Module 15.

**Worked example.** At one input, estimator A has bias $0.3$ and variance $0.04$; estimator B is unbiased with variance $0.2$; the noise is $\sigma^2 = 0.25$. Expected squared errors: A: $0.09 + 0.04 + 0.25 = 0.38$; B: $0 + 0.2 + 0.25 = 0.45$. The biased estimator wins. Now a concrete model: $f(x) = \sin(2\pi x)$ on $[0, 1]$, $n = 30$ training points, $\sigma = 0.3$, polynomials fitted by least squares; the lesson notebook measures the terms over 200 independent training sets. Straight line: bias$^2 \approx 0.20$ (the best line cannot follow a sine), variance $\approx 0.02$. Cubic: bias$^2 \approx 0.005$, variance $\approx 0.016$. Degree 5: bias$^2 \approx 0.001$, variance $\approx 0.05$. Degree 9: variance above $5$ — the polynomial swings wildly between and beyond the sampled points. Totals with $\sigma^2 = 0.09$: about $0.31$, $0.11$, $0.14$ and $> 5$: the cubic sits at the bottom of the U. The in-sample formula $\sigma^2 p/n$ would give $0.006$, $0.012$, $0.018$; the measured variance is larger because test points near the edges of $[0,1]$ have high leverage and because, for a misspecified model, *which* $x$-values were sampled also changes the best-fitting curve — the training inputs are part of the randomness in $D$.

**Real-life example (sport).** A football analyst predicts how many goals a striker will score next season. Using last season's tally alone (20 goals) is unbiased but has high variance — a season is a small sample of chances and luck. Using the positional average (10 goals) has zero variance and a large bias for genuinely good players. A shrinkage estimate, $0.6\times 20 + 0.4\times 10 = 16$, accepts some bias to cut the variance and beats both on next season's actual numbers; the shrinkage weight is set by how much of the between-player spread is skill rather than noise. This is the same calculation that makes ridge regression and hierarchical models work, and the reason "regression to the mean" is a prediction, not a mystery.

**In ML/AI this is …** the reason bagging and random forests reduce error (averaging over bootstrap training sets attacks the variance term), the reason dropout, early stopping, data augmentation and weight decay help (variance), and the reason a model that underfits is not rescued by more data (bias). Learning curves (Section 9) make the terms visible; the noise term is the "Bayes error" or "noise ceiling" that no model can beat.

## 9. Model selection and validation

**Intuition.** Training error flatters a model because the same data chose its parameters. The honest way to estimate performance is on data the model never saw, and cross-validation recycles a limited dataset so that every point gets to be "unseen" exactly once. One subtlety is easy to miss: if you also use the held-out score to *choose* the model (a value of $\lambda$, a feature set), that score is no longer honest — you have optimised it. Then you need a second, outer loop.

**Definition.** **k-fold cross-validation** splits the indices into $k$ roughly equal folds $F_1, \dots, F_k$; for each fold, fit on the other $k - 1$ and evaluate on it:

$$\operatorname{CV}_k = \frac1k\sum_{f=1}^{k}\frac{1}{|F_f|}\sum_{i\in F_f}\ell\big(y_i, \hat f^{(-f)}(x_i)\big).$$

$k = 5$ or $10$ is standard; $k = n$ is **leave-one-out** (LOO). For OLS, ridge and smoothing splines — penalised least-squares fits $\hat y = Hy$ — LOO needs no refitting (this is not true of every linear smoother: for $k$-nearest-neighbour regression, deleting a point changes its neighbourhood and the shortcut fails):

$$\operatorname{CV}_{\text{LOO}} = \frac1n\sum_{i=1}^{n}\left(\frac{y_i - \hat y_i}{1 - h_{ii}}\right)^2,$$

an exact identity whose proof hinges on the fact that deleting observation $i$ changes the fit at $x_i$ by $e_i h_{ii}/(1 - h_{ii})$. **Generalised cross-validation** (GCV) replaces each $h_{ii}$ by the average $\operatorname{tr}(H)/n$. To choose a hyperparameter, compute the CV error on a grid and take the minimiser — or the largest $\lambda$ within one standard error of the minimum (the **one-standard-error rule**), which prefers simpler models. **Nested CV:** an inner loop selects $\lambda$ on the training part of each outer fold, and the outer fold's error is reported; the CV score of the winning $\lambda$ from a single loop is optimistically biased because it is a minimum over many noisy estimates. **Information criteria** avoid refitting by penalising fit with parameter count: for a Gaussian model, $\operatorname{AIC} = n\log(\operatorname{RSS}/n) + 2p$ and $\operatorname{BIC} = n\log(\operatorname{RSS}/n) + p\log n$ (up to constants); AIC is asymptotically equivalent to LOO, BIC penalises harder and favours smaller models as $n$ grows. **Learning curves** plot training and validation error against the training-set size $n$: a high-bias model's two curves converge quickly to the same high plateau (more data will not help); a high-variance model shows a persistent gap that closes slowly (more data *will* help).

**Worked example.** Return to the five-point OLS fit of Section 2: residuals $(-0.2, -0.1, 1.0, -0.9, 0.2)$, leverages $(0.6, 0.3, 0.2, 0.3, 0.6)$. LOO residuals $e_i/(1 - h_{ii})$: $(-0.5, -0.143, 1.25, -1.286, 0.5)$; $\operatorname{CV}_{\text{LOO}} = (0.25 + 0.0204 + 1.5625 + 1.6531 + 0.25)/5 = 0.747$, almost double the training MSE of $0.38$ — refitting five times by hand gives identical numbers. The end points, with leverage $0.6$, have their residuals inflated by $2.5\times$: high-leverage points are the ones the model will miss most badly when they are new. AIC for the line: $5\ln(1.9/5) + 4 = -0.84$; for the intercept-only model: $5\ln(10/5) + 2 = 5.47$. The line wins comfortably.

**Real-life example (computer vision).** A team trains a linear classifier on image embeddings to flag defective products from photographs, with several photos of each product taken in one session. Random 10-fold CV gives 97 % accuracy; in production it is 84 %. The folds leaked: photos of the same product sat on both sides of the split, so the model was partly recognising the product and the lighting, not the defect. Grouping the folds by product (`GroupKFold`) gives an honest 85 %. The same mistake happens with multiple scans per patient, repeated measurements per machine, and time series split at random instead of chronologically.

**In ML/AI this is …** `KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, `cross_val_score` and `GridSearchCV`; `RidgeCV`, which uses the LOO shortcut (`gcv_mode`) to try every $\lambda$ at the cost of one fit; the validation set used for early stopping and learning-rate selection; the public/private leaderboard split of competitions (the private board is the outer loop); and scaling laws, which are learning curves on a log–log scale.

## 10. Diagnostics and pitfalls

**Intuition.** A coefficient table and an $R^2$ are summaries, and summaries hide things. The residuals know whether the model is missing a curve, whether the noise grows with the signal, whether one point is steering the whole fit, and whether two features are fighting over the same explanatory power. Ten minutes of plots after fitting catch most of the mistakes that make a model fail in production.

**Definition.** *Residual plots:* residuals against fitted values should be a structureless band; a funnel means heteroscedasticity (non-constant variance — standard errors are wrong, consider a log transform of $y$ or weighted least squares), a curve means a missing non-linearity. *Leverage and influence:* $h_{ii}$ averages $p/n$ and a point with $h_{ii} > 2p/n$ is far from the others in feature space; its influence on the fit is measured by **Cook's distance** $D_i = \frac{e_i^2}{p\,\hat\sigma^2}\cdot\frac{h_{ii}}{(1 - h_{ii})^2}$, roughly the squared change in fitted values when point $i$ is dropped (values near 1 deserve a look). *Multicollinearity:* the **variance inflation factor** $\operatorname{VIF}_j = 1/(1 - R_j^2)$, where $R_j^2$ is from regressing feature $j$ on the other features, satisfies

$$\operatorname{Var}(\hat\beta_j) = \frac{\sigma^2\,\operatorname{VIF}_j}{\sum_i (x_{ij} - \bar x_j)^2};$$

VIF above 5–10 means that coefficient is poorly determined (predictions may still be fine). *Extrapolation:* a linear fit is a local approximation; outside the convex hull of the training inputs there is no data to contradict it. *Leakage:* any information in the training features that will not exist at prediction time — the target in disguise, future values, or preprocessing statistics (scaler means, feature selection) computed on the full dataset before splitting. *Target scaling:* a right-skewed target (prices, counts, durations) is often better modelled as $\log y$; remember that $\exp$ of a predicted mean of $\log y$ is a median, not a mean. *Standardised coefficients* ("effect per standard deviation of $x_j$") are comparable across features; raw coefficients are not. *Correlation is not causation:* in observational data a coefficient estimates association conditional on the other included features. **Omitted-variable bias:** if the truth is $y = \beta_1x_1 + \beta_2x_2 + \varepsilon$ and $x_2 = \delta x_1 + u$, regressing $y$ on $x_1$ alone gives $\mathbb{E}[\hat\beta_1] = \beta_1 + \beta_2\delta$ (Module 09, Simpson's paradox).

**Worked example.** VIF for two standardised features with correlation $r$: $1/(1 - r^2)$ gives $1.33$ at $r = 0.5$, $5.26$ at $0.9$, $10.3$ at $0.95$ and $50.3$ at $0.99$ — at $r = 0.99$ the standard errors are inflated by $\sqrt{50.3} \approx 7$, exactly the instability ridge was built for. Leverage: append the point $x = 10$ to the five-point data of Section 2; its leverage is $0.836$, above $2p/n = 0.67$, while the original points drop to $0.17$–$0.36$: the new point alone now largely decides the slope. Omitted variable: with $\beta_1 = 2$, $\beta_2 = 1$, $\delta = 0.5$, the one-feature regression reports a slope of $2.5$, not $2$. In the diabetes dataset used in the notebooks, the serum measurements `s1` and `s2` have correlation $0.90$ and VIFs of $59$ and $39$, which is why their OLS coefficients are large with opposite signs and why the lasso drops one of them.

**Real-life example (engineering).** A structural-health model predicts the load on a bridge girder from twelve strain gauges. Three gauges sit within a metre of each other and read almost identically (VIF around 80), so their OLS coefficients are $+310$, $-290$ and $+15$ with enormous standard errors; ridge turns them into three similar moderate weights without changing the predictions. The training data cover loads up to 40 tonnes; the first 60-tonne truck is an extrapolation, and the model's confident prediction there is unverified. And a residual-versus-fitted plot shows a funnel — the gauges' noise grows with strain — so the reported prediction intervals, computed assuming constant variance, are too narrow at high loads and too wide at low ones.

**In ML/AI this is …** `OLSInfluence` and `variance_inflation_factor` in statsmodels; wrapping preprocessing in a scikit-learn `Pipeline` inside `cross_val_score` so scalers and selectors see only training folds; `TimeSeriesSplit` for leak-free backtesting; distribution shift, which is extrapolation in disguise; test-set contamination in language-model benchmarks, which is leakage at scale; and the "Clever Hans" shortcuts (a ruler in the photo of a melanoma) that grouped cross-validation and residual analysis are designed to catch.

## Common pitfalls and misconceptions

1. **Penalising the intercept.** Ridge or lasso with the constant column included shrinks the mean of $y$ towards zero, which is meaningless. → Centre $X$ and $y$ (or exclude the intercept from the penalty, as `fit_intercept=True` does).
2. **Regularising unstandardised features.** The penalty then depends on units: a coefficient on a feature measured in millimetres is 1,000 times smaller than in metres and is barely shrunk. → Standardise first (inside the CV pipeline), then fit; un-scale the coefficients for reporting if needed.
3. **"A lasso zero means no effect."** The lasso drops one of two near-duplicate predictors arbitrarily; the dropped one may matter just as much. → Check the correlation structure, use the elastic net for groups, and examine selection stability across bootstrap samples.
4. **"High $R^2$ means a good model."** $R^2$ rises with every added feature and says nothing about out-of-sample error or whether the residuals have structure. → Use cross-validated error or adjusted criteria, and look at the residual plots.
5. **Trusting the CV score of the chosen hyperparameter.** The minimum over a grid of noisy estimates is optimistically biased. → Nested CV, or a truly untouched test set evaluated once.
6. **Fitting preprocessing on the whole dataset before splitting.** Scaler means, imputations and feature selection then carry test information into training — leakage that inflates every score. → `Pipeline` + `cross_val_score`.
7. **Diagnosing a diverging logistic regression as a bug.** On separable data the unpenalised MLE does not exist; coefficients growing without bound is the correct behaviour. → Add an L2 penalty (or use the default regularisation), and read the warning as "your classes are separable".
8. **Interpreting a coefficient as a causal effect.** In observational data $\hat\beta_j$ is an association conditional on the other included features and is biased by anything omitted. → Say "associated with", include confounders deliberately, and treat causal claims as a different problem.

## Real-world case studies

### Case study 1 — A credit scorecard (finance)

A lender has 50,000 past loan applications with a 4 % default rate and wants a score for new applicants. The model is logistic regression on about 15 features: age, income, months at current address, number of delinquencies in the last two years, credit utilisation, loan-to-income ratio and a few one-hot categories. Continuous features are first **binned** into 5–10 groups and each bin replaced by its weight of evidence, $\log\frac{P(\text{bin}\mid\text{good})}{P(\text{bin}\mid\text{bad})}$ — a monotone, outlier-proof transformation that keeps the model linear in parameters while letting each feature's effect bend. L2 regularisation with $\lambda$ chosen by 5-fold CV handles the correlated pairs (income and loan-to-income); the lasso path is used as a feature-selection screen so that the final scorecard has 10–12 features the credit committee can read.

The intercept corresponds to the base log-odds $\ln(0.04/0.96) = -3.18$; a coefficient of $0.8$ on "each delinquency" is an odds ratio of $e^{0.8} = 2.23$. The log-odds are converted to points so that 20 points double the odds: $\text{score} = 600 + \frac{20}{\ln 2}\ln\frac{\text{odds of repayment}}{50}$, which anchors 600 points at odds of 50:1 (a 2 % default probability). An applicant with predicted default probability 4 % has repayment odds 24:1 and scores $600 + 28.85\ln(24/50) \approx 579$. Three properties of logistic regression make it the regulatory standard here: the score is monotone in each feature, so every decline can be explained by the features that cost the most points (regulators in many jurisdictions require stated reasons for adverse decisions); maximum-likelihood fitting makes the probabilities calibrated, so the pricing team can translate score bands directly into expected losses; and the whole model fits in a second, so it can be refitted monthly and checked for population drift by comparing the distribution of scores — the diagnostics of Section 10 in production.

### Case study 2 — Selecting a gene signature with the lasso (healthcare)

An oncology group has expression levels of 20,000 genes for 300 patients and wants a small panel predicting recurrence within five years. With $p \gg n$ the OLS and unpenalised logistic fits interpolate the labels perfectly and are worthless. L1-penalised logistic regression on standardised log-expression values traces a path: at large $\lambda$ no gene is selected, at $\lambda$ chosen by 10-fold CV about 25 genes are non-zero, and at smaller $\lambda$ the count climbs past 300 and the CV error rises again.

The trap is in the validation. If the 25 genes are selected on all 300 patients and *then* a logistic model on those genes is cross-validated, the apparent AUC is around 0.95; a nested procedure, which repeats the selection inside each training fold, gives 0.72 — the honest number, and the one that will be reproduced in an external cohort. The difference is leakage of the labels through the selection step (Section 9), and it is the most common reason published signatures fail to replicate. Two further Section 5 lessons apply. Genes in the same pathway are highly correlated, so the lasso's choice among them is unstable: running the fit on 200 bootstrap samples shows that only about 8 genes appear in more than 70 % of the panels, and those form the reported signature; an elastic-net penalty with $\lambda_2 > 0$ makes the selected set noticeably more stable. And because the lasso shrinks the survivors as well as zeroing the rest, the final risk model is refitted by L2-penalised logistic regression on the 8 chosen genes, whose odds ratios are then calibrated against the observed recurrence rates.

### Case study 3 — Pricing motor insurance with Poisson and Gamma GLMs (insurance)

An insurer sets premiums from 400,000 policy-years with 28,000 claims. Pure premium is modelled as frequency × severity. Frequency: a Poisson GLM with log link, offset $\log(\text{exposure})$ so that a policy covering half a year contributes half the expected count, and features driver age band, vehicle group, region, no-claims years and annual mileage (binned, one-hot). Severity: a Gamma GLM with log link on the claims. Because both use a log link, every fitted coefficient becomes a multiplicative **relativity** $e^{\beta_j}$, and the premium is the base rate times a product of relativities — the tariff table the regulator inspects and the sales system applies.

Suppose the base frequency is 0.12 claims per policy-year, the relativity for drivers aged 18–21 is $e^{0.35} = 1.42$ and that for a rural region is $e^{-0.25} = 0.78$. A young rural driver has expected frequency $0.12\times1.42\times0.78 = 0.133$ per year, or $0.066$ over a six-month policy; multiplied by a fitted mean severity of 2,400 the pure premium is about 160 for the half year, before expenses and profit loading. Section 7's deviance diagnostics do real work: the frequency model's deviance is 1.6 times its degrees of freedom, so the actuaries report quasi-Poisson standard errors inflated by $\sqrt{1.6}$ and are candid that the 2-level regional effect is not significant. The 300 vehicle groups are a classic ridge problem — some have five policies — so the group effects are shrunk towards their class mean with an L2 penalty tuned by cross-validation on a year held out in time, not at random, because the goal is to price next year's policies.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Empirical risk minimisation | Every training loop (`loss.backward()` on a mini-batch) | The training loss estimates $\hat R_n$, the validation loss estimates $R$ |
| OLS / normal equations | `LinearRegression`, `np.linalg.lstsq`, the closed-form last layer of some models | The one model you can solve exactly; MSE = Gaussian likelihood |
| Basis expansion | Kernels, learned features, embeddings, positional encodings (Module 17) | A network learns $\phi(x)$; the head on top is a linear model |
| Ridge / L2 penalty | Weight decay in SGD and AdamW, Tikhonov, kernel ridge, GP noise term | Shrinks poorly determined directions; Gaussian prior (Module 11) |
| Lasso / L1 penalty | Pruning, sparse coding, sparse autoencoders, compressed sensing | Exact zeros via soft-thresholding; the proximal operator of $\lVert\cdot\rVert_1$ |
| Logistic regression | Output layer of classifiers, linear probes, `BCEWithLogitsLoss` | Gradient $p - y$; separability explains why regularisation is needed |
| Softmax regression | Final layer of every LLM and image classifier | Multinomial GLM; gradient $P - Y$ (Module 16) |
| GLMs and canonical links | Output activations and losses (identity/MSE, sigmoid/BCE, exp/Poisson NLL) | Pre-activation gradient is always $\hat\mu - y$ |
| Bias–variance | Bagging, dropout, early stopping, double descent (Module 15) | Explains which interventions attack which error term |
| Cross-validation | `GridSearchCV`, early stopping, leaderboards, hyperparameter search | Honest estimates; nested CV for honest selection |
| Leverage, leakage, VIF | Data pipelines, `Pipeline`, `GroupKFold`, benchmark contamination | Why models fail in production despite good CV scores |

## Cheat sheet

| Fact | Formula / statement |
|---|---|
| Risks | $R(f) = \mathbb{E}[\ell(f(X),Y)]$; $\hat R_n(f) = \frac1n\sum_i\ell(f(x_i),y_i)$; squared-loss optimum $f^*(x) = \mathbb{E}[Y\mid X=x]$ |
| OLS | $\hat\beta = (X^\top X)^{-1}X^\top y$; $H = X(X^\top X)^{-1}X^\top$; $\operatorname{tr}H = p$; $e = (I-H)y$ |
| Fit and inference | $R^2 = 1 - \operatorname{RSS}/\operatorname{TSS}$; $\hat\sigma^2 = \operatorname{RSS}/(n-p)$; $\operatorname{Var}(\hat\beta) = \sigma^2(X^\top X)^{-1}$; Gauss–Markov: OLS is BLUE |
| Standardisation | $z_j = (x_j - \bar x_j)/s_j$; required before any penalty |
| Ridge | $\hat\beta_\lambda = (X^\top X + \lambda I)^{-1}X^\top y = \sum_j \frac{d_j}{d_j^2+\lambda}\langle u_j,y\rangle v_j$; $\operatorname{df}(\lambda) = \sum_j\frac{d_j^2}{d_j^2+\lambda}$; MAP with $\lambda = \sigma^2/\tau^2$ |
| Soft threshold | $S_\lambda(z) = \operatorname{sign}(z)\max(\lvert z\rvert - \lambda, 0)$; lasso = $S_\lambda(\hat\beta^{\text{ols}})$ for orthonormal $X$ |
| Coordinate descent | $\beta_j \leftarrow S_\lambda(x_j^\top r^{(j)})/\lVert x_j\rVert^2$, $r^{(j)} = y - \sum_{k\ne j}x_k\beta_k$; sklearn $\lambda = n\alpha$ |
| Elastic net | $\lambda_1\lVert\beta\rVert_1 + \lambda_2\lVert\beta\rVert_2^2$ |
| Logistic regression | $p = \sigma(\beta^\top x)$; $\nabla\operatorname{NLL} = X^\top(p - y)$; $\nabla^2\operatorname{NLL} = X^\top WX$, $W_{ii} = p_i(1-p_i)$; odds ratio $e^{\beta_j}$ |
| Newton / IRLS | $\beta \leftarrow \beta + (X^\top WX)^{-1}X^\top(y - p)$; no MLE on separable data |
| GLM | $g(\mu) = \beta^\top x$; canonical links identity / logit / log; score $X^\top(y - \mu)$ |
| Poisson deviance | $D = 2\sum_i[y_i\log(y_i/\hat\mu_i) - (y_i - \hat\mu_i)]$; offset $\log t_i$ for exposure |
| Bias–variance | $\mathbb{E}[(y - \hat f)^2] = \sigma^2 + (f - \bar f)^2 + \mathbb{E}[(\hat f - \bar f)^2]$; OLS average variance $\sigma^2 p/n$ |
| k-fold CV | $\frac1k\sum_f$ (mean loss on fold $f$ of the model fitted without fold $f$) |
| LOO shortcut | $\operatorname{CV}_{\text{LOO}} = \frac1n\sum_i\big(\frac{y_i - \hat y_i}{1 - h_{ii}}\big)^2$ for OLS, ridge and smoothing splines (penalised least squares; not every linear smoother) |
| AIC / BIC | $n\log(\operatorname{RSS}/n) + 2p$; $n\log(\operatorname{RSS}/n) + p\log n$ |
| Diagnostics | leverage flag $h_{ii} > 2p/n$; Cook's $D_i = \frac{e_i^2}{p\hat\sigma^2}\frac{h_{ii}}{(1-h_{ii})^2}$; $\operatorname{VIF}_j = 1/(1 - R_j^2)$ |

## Self-assessment

1. **Why is $\mathbb{E}[Y\mid X]$ the best predictor under squared loss, and what is the risk of that predictor?**
   <details><summary>Answer</summary>

   For any $f$, $\mathbb{E}[(Y - f(X))^2] = \mathbb{E}[(Y - f^*(X))^2] + \mathbb{E}[(f^*(X) - f(X))^2]$ with $f^* = \mathbb{E}[Y\mid X]$, because the cross term has zero conditional mean. The second term is non-negative and zero only for $f = f^*$. The risk of $f^*$ is $\mathbb{E}[\operatorname{Var}(Y\mid X)]$, the irreducible noise.

   </details>

2. **Give the one-line geometric argument for the normal equations.**
   <details><summary>Answer</summary>

   The fitted vector $X\hat\beta$ is the orthogonal projection of $y$ onto the column space of $X$, so the residual $y - X\hat\beta$ is orthogonal to every column: $X^\top(y - X\hat\beta) = 0$, i.e. $X^\top X\hat\beta = X^\top y$.

   </details>

3. **What does the Gauss–Markov theorem assume, and what does it *not* say?**
   <details><summary>Answer</summary>

   It assumes a correctly specified linear mean, errors with zero mean, constant variance and no correlation. It says OLS has the smallest variance among *linear unbiased* estimators. It does not require normality, and it does not say OLS has the smallest mean squared error overall — biased estimators such as ridge can beat it.

   </details>

4. **A ridge fit uses a design matrix whose singular values are $10, 3, 0.1$. With $\lambda = 1$, by what factor is the OLS coefficient along each singular direction multiplied, and what is $\operatorname{df}(\lambda)$?**
   <details><summary>Answer</summary>

   Filter factors $d_j^2/(d_j^2 + \lambda)$: $100/101 = 0.990$, $9/10 = 0.900$, $0.01/1.01 = 0.0099$. The weak direction is essentially removed. $\operatorname{df}(1) = 0.990 + 0.900 + 0.010 = 1.90$ instead of $3$.

   </details>

5. **For an orthonormal design, OLS gives coefficients $(2.5, -0.8, 0.3)$. What do lasso and ridge return with $\lambda = 0.5$?**
   <details><summary>Answer</summary>

   Lasso (soft-thresholding): $(2.0, -0.3, 0)$. Ridge ($\div(1 + \lambda)$): $(1.667, -0.533, 0.2)$. Lasso zeroes the small coefficient and shifts the others by a constant; ridge scales all of them by the same factor and zeroes none.

   </details>

6. **Why does the unpenalised logistic-regression MLE fail to exist on linearly separable data, and what does an L2 penalty change?**
   <details><summary>Answer</summary>

   If $\beta$ separates the classes, $c\beta$ with $c \to \infty$ pushes every $p_i$ towards its label and the negative log-likelihood towards its infimum $0$ without attaining it, so no finite maximiser exists. Adding $\frac{\lambda}{2}\lVert\beta\rVert^2$ makes the objective strictly convex and coercive (it grows without bound as $\lVert\beta\rVert \to \infty$), so a unique finite minimiser exists.

   </details>

7. **In a Poisson GLM with log link, the coefficient on "weekend" is $-0.4$. Interpret it, and explain what an offset does.**
   <details><summary>Answer</summary>

   The expected count on weekends is $e^{-0.4} = 0.67$ times the weekday count — a rate ratio of 0.67, i.e. 33 % fewer events, all else equal. An offset $\log t_i$ added to the linear predictor with a fixed coefficient of 1 makes the model describe the *rate* per unit exposure, so observations with different exposures (policy durations, populations, time windows) are comparable.

   </details>

8. **A model has training MSE 1.0 and 10-fold CV MSE 1.1 at $n = 200$; doubling the data gives 1.05 and 1.08. Bias or variance problem? What would you try?**
   <details><summary>Answer</summary>

   The train and validation errors are close and barely improve with more data: the model is bias-limited (underfitting). More data will not help; add features, interactions or a richer basis, or reduce the regularisation. A variance problem would show a large train–validation gap that narrows as $n$ grows.

   </details>

9. **You pick $\lambda$ by 5-fold CV from a grid of 50 values and report the CV error of the winner. Why is that number too optimistic, and what fixes it?**
   <details><summary>Answer</summary>

   Each CV error is a noisy estimate; the minimum of 50 noisy estimates is biased downward (you have partly selected on noise). Nested CV — select $\lambda$ with an inner loop on each outer training fold and evaluate on the outer test fold — or a single untouched test set gives an unbiased estimate of the selected model's performance.

   </details>

10. **Two standardised features have correlation 0.98. What is their VIF, what does it imply for the OLS coefficients, and does it hurt predictions?**
   <details><summary>Answer</summary>

   $\operatorname{VIF} = 1/(1 - 0.98^2) = 25.3$: coefficient standard errors are $\sqrt{25.3} \approx 5$ times larger than if the features were uncorrelated, and the two estimates will be large with opposite signs and highly unstable across samples. Predictions within the range of the data are largely unaffected, because only the *sum* of the two effects is well determined; ridge or dropping one feature stabilises the coefficients.

   </details>

## Further reading

- Trevor Hastie, Robert Tibshirani and Jerome Friedman, *The Elements of Statistical Learning* (2nd ed., Springer, 2009; free PDF from the authors' site), Chapters 3, 4 and 7 — linear methods for regression and classification, and model assessment.
- Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani, *An Introduction to Statistical Learning* (2nd ed., Springer, 2021; free PDF at statlearning.com), Chapters 3–6 — the gentler treatment, with labs.
- Trevor Hastie, Robert Tibshirani and Martin Wainwright, *Statistical Learning with Sparsity: The Lasso and Generalizations* (CRC Press, 2015; free PDF from the authors) — everything about L1.
- Arthur E. Hoerl and Robert W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems", *Technometrics*, 1970 — the original ridge paper.
- Robert Tibshirani, "Regression Shrinkage and Selection via the Lasso", *Journal of the Royal Statistical Society B*, 1996.
- Hui Zou and Trevor Hastie, "Regularization and Variable Selection via the Elastic Net", *Journal of the Royal Statistical Society B*, 2005.
- Jerome Friedman, Trevor Hastie and Robert Tibshirani, "Regularization Paths for Generalized Linear Models via Coordinate Descent", *Journal of Statistical Software*, 2010 — the algorithm behind `glmnet` and scikit-learn's lasso.
- Peter McCullagh and John A. Nelder, *Generalized Linear Models* (2nd ed., Chapman & Hall, 1989) — the standard reference; the original idea is Nelder and Wedderburn, *JRSS A*, 1972.
- Stuart Geman, Elie Bienenstock and René Doursat, "Neural Networks and the Bias/Variance Dilemma", *Neural Computation*, 1992 — the decomposition applied to learning machines.
- Gavin C. Cawley and Nicola L. C. Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation", *Journal of Machine Learning Research*, 2010 — why nested cross-validation is necessary.
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022; free PDF at probml.ai), Chapters 10–12 — logistic regression, linear regression and GLMs with modern notation.

## What's in the notebooks

**`lesson.ipynb`** walks through every section with code you can edit: empirical versus true risk on synthetic data; OLS solved three ways (normal equations, gradient descent on RSS, numerical Gaussian maximum likelihood) and shown to agree, with the hat matrix, $R^2$ and standard errors; the polynomial-overfitting panel at degrees 1, 3 and 9 on ten points and the condition number before and after standardisation; ridge from the closed form, its coefficient path against $\lambda$, the SVD filter factors and effective degrees of freedom, checked against `sklearn.linear_model.Ridge`; the soft-thresholding operator, a lasso built from your own coordinate descent matched to `Lasso`, and the lasso path on the diabetes data; logistic regression by Newton/IRLS from scratch on the standardised breast-cancer data compared with `LogisticRegression`, followed by the separable-data divergence demonstration; Poisson regression by gradient ascent and Newton on synthetic counts with the deviance; the bias–variance decomposition measured over hundreds of training sets for polynomial degrees 0–10; k-fold cross-validation for $\lambda$, the exact leave-one-out shortcut verified against brute force, and learning curves for under- and over-parameterised models; and residual, leverage and VIF diagnostics.

**`exercises.ipynb`** has twelve exercises with instant self-checks: OLS via the normal equations with $R^2$; polynomial features; the soft-thresholding operator; ridge in closed form against scikit-learn; lasso by coordinate descent; the logistic-regression gradient, Hessian and Newton step; bias², variance and noise by simulation; k-fold cross-validation that selects $\lambda$ from a grid (hidden answer); a conceptual question on standardising before regularisation; Poisson regression with an exposure offset and its deviance; the leave-one-out shortcut for ridge; and a mini-project — an end-to-end regularised regression on `load_diabetes` with standardisation, nested cross-validation for ridge and lasso, a comparison of which coefficients survive, and a test RMSE with a bootstrap confidence interval. `solutions.ipynb` contains reference implementations and explanations.
