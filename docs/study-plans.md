# Study plans

A module takes roughly **6–10 hours** done properly: about half of that on the concept guide and lesson notebook, half on the exercises and mini-project. The whole course is about 120–180 hours. Pick the plan that fits your week; all links point to each module's concept guide.

## Plan A — Steady pace at about 6 hours a week (≈ 7 months)

One module every week and a half, with a short review week after each part. This is the plan for a working week with a fixed learning budget.

| Weeks | Modules | Milestone |
|-------|---------|-----------|
| 1–5 | [01](../part-1-linear-algebra/01-vectors-and-geometry/README.md), [02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md), [03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md) + review | You can read any ML paper's linear algebra: projections, least squares, SVD, PCA. |
| 6–10 | [04](../part-2-calculus-and-optimization/04-differential-calculus/README.md), [05](../part-2-calculus-and-optimization/05-optimization/README.md), [06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md) + review | You have built and trained a network with your own autograd engine. |
| 11–15 | [07](../part-3-probability/07-probability-foundations/README.md), [08](../part-3-probability/08-distributions-and-limit-theorems/README.md), [09](../part-3-probability/09-multivariate-probability/README.md) + review | Bayes, distributions, LLN/CLT and the multivariate Gaussian are second nature. |
| 16–20 | [10](../part-4-statistics/10-estimation-and-sampling/README.md), [11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md), [12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md) + review | You can put honest error bars on a metric and design an A/B test. |
| 21–25 | [13](../part-5-learning-theory/13-linear-models-and-regularization/README.md), [14](../part-5-learning-theory/14-information-theory/README.md), [15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md) + review | You know why regularisation, cross-entropy and validation sets work. |
| 26–31 | [16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md), [17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md), [18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md), [19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md) | You can derive attention, the VAE loss, diffusion and policy gradients yourself. |

**How to spend the six hours.** Four sessions of about 90 minutes work better than one long block. Over a module's week and a half (six sessions): sessions 1–2 read the concept guide and work its examples by hand; session 3 runs the lesson notebook; sessions 4–5 do the exercises; session 6 is for the mini-project and answering the self-assessment questions from memory. In a review week, redo the self-assessment questions of the part's three modules and rewrite their cheat sheets in your own words. If a week is short, skip the mini-project and come back to it in the review week — never skip the exercises.

If six hours a week is your ceiling and you have a specific goal, one of the topic tracks below (Plans D–F) gets you there in roughly 10–12 weeks.

## Plan B — The full course in 12 weeks (≈ 10–15 h/week)

| Week | Modules | Focus |
|------|---------|-------|
| 1 | [01](../part-1-linear-algebra/01-vectors-and-geometry/README.md), [02](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md) | Vectors, dot products, projections; matrices as maps, least squares |
| 2 | [03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md) | Eigenvalues, SVD, PCA — take your time, everything later uses it |
| 3 | [04](../part-2-calculus-and-optimization/04-differential-calculus/README.md), [05](../part-2-calculus-and-optimization/05-optimization/README.md) | Gradients, Jacobians, Hessians; gradient descent, Adam, constraints |
| 4 | [06](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md) | Build an autograd engine and train a network with it |
| 5 | [07](../part-3-probability/07-probability-foundations/README.md), [08](../part-3-probability/08-distributions-and-limit-theorems/README.md) | Probability, Bayes, distributions, LLN/CLT |
| 6 | [09](../part-3-probability/09-multivariate-probability/README.md), [10](../part-4-statistics/10-estimation-and-sampling/README.md) | Covariance, multivariate Gaussian; estimators, CIs, bootstrap |
| 7 | [11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md), [12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md) | MLE ⇒ loss functions, Bayesian updating; tests, power, A/B testing |
| 8 | [13](../part-5-learning-theory/13-linear-models-and-regularization/README.md) | Linear models, ridge/lasso, logistic regression, bias–variance |
| 9 | [14](../part-5-learning-theory/14-information-theory/README.md), [15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md) | Entropy, KL, mutual information; generalisation bounds, double descent |
| 10 | [16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md) | MLP from scratch: initialisation, normalisation, training loop |
| 11 | [17](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md) | Attention, positional encodings, layer norm, KV cache, LoRA |
| 12 | [18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md), [19](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md) | MCMC, VAEs, diffusion; bandits, Bellman equations, policy gradients |

## Plan C — Intensive 6 weeks (≈ 25–30 h/week)

One part per week: Week 1 = Part I (Modules 01–03), Week 2 = Part II (04–06), Week 3 = Part III (07–09), Week 4 = Part IV (10–12), Week 5 = Part V (13–15), Week 6 = Part VI (16–19). Do every exercise; postpone mini-projects you cannot fit into the week to a catch-up weekend at the end.

## Plan D — "I need to understand transformers and LLMs" (≈ 65–75 hours)

1. [01 Vectors and geometry](../part-1-linear-algebra/01-vectors-and-geometry/README.md) — embeddings, cosine similarity, high-dimensional geometry
2. [02 Matrices and linear maps](../part-1-linear-algebra/02-matrices-and-linear-maps/README.md) — sections on matrix multiplication, rotations and orthogonal matrices
3. [03 Eigenvalues, SVD and PCA](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md) — low-rank approximation (for LoRA)
4. [04 Differential calculus](../part-2-calculus-and-optimization/04-differential-calculus/README.md) — chain rule, Jacobians, softmax gradient
5. [05 Optimisation](../part-2-calculus-and-optimization/05-optimization/README.md) — Adam, learning-rate schedules
6. [06 Autodiff and backpropagation](../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md)
7. [11 Likelihood and Bayesian inference](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md) — section "MLE gives you your loss functions"
8. [14 Information theory](../part-5-learning-theory/14-information-theory/README.md) — cross-entropy, KL, perplexity, temperature
9. [16 Neural networks from scratch](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md)
10. [17 The mathematics of transformers](../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)
11. [19 MDPs and reinforcement learning](../part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md) — section on KL-regularised RL, RLHF and DPO

## Plan E — "I work with data and experiments" (statistics track, ≈ 60–75 hours)

[07](../part-3-probability/07-probability-foundations/README.md) → [08](../part-3-probability/08-distributions-and-limit-theorems/README.md) → [09](../part-3-probability/09-multivariate-probability/README.md) → [10](../part-4-statistics/10-estimation-and-sampling/README.md) → [11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md) → [12](../part-4-statistics/12-hypothesis-testing-and-experiments/README.md) → [13](../part-5-learning-theory/13-linear-models-and-regularization/README.md) → [15](../part-5-learning-theory/15-generalization-and-learning-theory/README.md), with [01](../part-1-linear-algebra/01-vectors-and-geometry/README.md)–[03](../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md) as reference when covariance matrices and PCA appear.

## Plan F — "Generative models and probabilistic AI" (≈ 60–70 hours)

[07](../part-3-probability/07-probability-foundations/README.md) → [08](../part-3-probability/08-distributions-and-limit-theorems/README.md) → [09](../part-3-probability/09-multivariate-probability/README.md) → [11](../part-4-statistics/11-likelihood-and-bayesian-inference/README.md) → [14](../part-5-learning-theory/14-information-theory/README.md) → [04](../part-2-calculus-and-optimization/04-differential-calculus/README.md) (chain rule, Jacobians, change of variables) → [16](../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md) → [18](../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md).

## Keeping the material alive

* After each part, redo the self-assessment questions of its modules from memory.
* Keep a personal formula sheet — rewrite each module's cheat sheet in your own words.
* Revisit the mini-projects after a month and extend them (another dataset, another dimension, another algorithm).
* When you read a new paper, find its central equation and identify which modules it rests on; the "Where this shows up in ML, DL and AI" table at the end of each guide is a good index for that.
