# Mathematics & Statistics for Machine Learning, Deep Learning and AI

[![Execute all notebooks](https://github.com/dthemann/AI-math/actions/workflows/notebooks.yml/badge.svg)](https://github.com/dthemann/AI-math/actions/workflows/notebooks.yml)

> A complete, self-paced course that builds the mathematical foundation behind modern machine learning — from vectors and gradients to transformers, diffusion models and reinforcement learning — with every concept explained in depth, grounded in real-world examples, implemented from scratch in NumPy, and tested with self-checking exercises.

**19 modules · 6 parts · 57 notebooks · ~175,000 words of concept guides · 220 exercises · 1,100+ automatic self-checks · ~200 figures**

---

## Who this is for

You can write Python, you remember *some* mathematics from school or a first-year course, and you want to **actually understand** the maths that machine-learning libraries hide from you: why attention divides by √d, why cross-entropy is the "right" loss, what a p-value really says, why PCA is an eigenvalue problem, what a diffusion model computes. The course assumes no prior knowledge beyond that; it never assumes you remember a formula it has not derived or stated itself.

If you already know some of the material, each module is self-contained enough to be taken on its own — the **Prerequisites** section of every concept guide tells you exactly which earlier modules it relies on.

## How the course is organised

Every module lives in its own folder and consists of exactly four files:

| File | What it is | How to use it |
|------|------------|---------------|
| `README.md` | The **concept guide** — a detailed, written explanation of every idea in the module: intuition, precise definitions, hand-worked numerical examples, real-life scenarios, where the concept appears in ML/AI, common pitfalls, case studies, a cheat sheet and self-assessment questions. | Read it first, or alongside the lesson. It renders on GitHub, including the mathematics. |
| `lesson.ipynb` | The **narrated walk-through** — the same ideas built from scratch in NumPy, with visualisations and "try it yourself" prompts. | Run it cell by cell; change the numbers; break things. |
| `exercises.ipynb` | **10–12 exercises** with instant ✅/❌ self-checks, graded ★☆☆ → ★★★, ending in a mini-project. Some checks compare against hidden answers so you cannot peek. | Fill in the `# YOUR CODE HERE` cells, run the check cells, watch your score. |
| `solutions.ipynb` | Reference solutions with explanations of *why* they work and the common wrong turns. | Open only after a serious attempt. |

A module takes roughly **6–10 hours** to work through properly (guide + lesson + exercises). The whole course is about 120–180 hours of study — see [docs/study-plans.md](docs/study-plans.md) for paths ranging from an intensive 6-week sprint to a steady 6-hours-a-week plan, plus shorter topic tracks (transformers and LLMs, statistics and experiments, generative models).

## Curriculum

### Part I — Linear algebra: the language of data
| # | Module | You will be able to… |
|---|--------|----------------------|
| 01 | [Vectors and geometry](part-1-linear-algebra/01-vectors-and-geometry/README.md) | Treat data as vectors; use norms, dot products, cosine similarity, projections, orthonormal bases and hyperplanes; understand why high-dimensional geometry is strange — and why embeddings work. |
| 02 | [Matrices and linear maps](part-1-linear-algebra/02-matrices-and-linear-maps/README.md) | See matrices as functions; multiply, invert and decompose them; understand rank and null spaces; derive and solve least squares two ways; change basis; read Markov transition matrices. |
| 03 | [Eigenvalues, SVD and PCA](part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md) | Find the directions a map only stretches; diagonalise; use the spectral theorem and the SVD; compress and denoise with low-rank approximations; derive PCA; run PageRank; reason about condition numbers. |

### Part II — Calculus and optimisation: how models learn
| # | Module | You will be able to… |
|---|--------|----------------------|
| 04 | [Differential calculus for ML](part-2-calculus-and-optimization/04-differential-calculus/README.md) | Compute and interpret gradients, Jacobians and Hessians; use the chain rule and Taylor expansions; apply the matrix-calculus identities behind every ML loss function. |
| 05 | [Optimisation](part-2-calculus-and-optimization/05-optimization/README.md) | Recognise convexity; analyse gradient descent and its step size; implement SGD, momentum, RMSProp, Adam and Newton's method; solve constrained problems with Lagrange multipliers and KKT conditions. |
| 06 | [Automatic differentiation and backpropagation](part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md) | Explain forward- vs reverse-mode AD; build a working autograd engine from scratch; derive backpropagation; gradient-check anything; train a small neural network with your own engine. |

### Part III — Probability: reasoning under uncertainty
| # | Module | You will be able to… |
|---|--------|----------------------|
| 07 | [Probability foundations](part-3-probability/07-probability-foundations/README.md) | Use the axioms, conditional probability and Bayes' theorem correctly (base rates!); work with random variables, expectation and variance; simulate; build a naive-Bayes classifier by hand. |
| 08 | [Distributions and limit theorems](part-3-probability/08-distributions-and-limit-theorems/README.md) | Choose and use the key discrete and continuous distributions; understand moments and tails; state and demonstrate the law of large numbers and the central limit theorem; estimate anything by Monte Carlo. |
| 09 | [Multivariate probability](part-3-probability/09-multivariate-probability/README.md) | Handle joint, marginal and conditional distributions; covariance matrices and the multivariate Gaussian; Gaussian conditioning (and from it a Kalman filter and Gaussian-process regression); spot Simpson's paradox. |

### Part IV — Statistics: learning from data
| # | Module | You will be able to… |
|---|--------|----------------------|
| 10 | [Estimation and sampling](part-4-statistics/10-estimation-and-sampling/README.md) | Reason about estimators (bias, variance, MSE, consistency), standard errors, sampling distributions, confidence intervals and the bootstrap; plan sample sizes; put honest error bars on model metrics. |
| 11 | [Likelihood and Bayesian inference](part-4-statistics/11-likelihood-and-bayesian-inference/README.md) | Derive maximum-likelihood estimators and Fisher information; see how MLE produces every standard loss function and MAP produces regularisation; update conjugate posteriors; run a Bayesian A/B test. |
| 12 | [Hypothesis testing and experiments](part-4-statistics/12-hypothesis-testing-and-experiments/README.md) | Design and analyse experiments: p-values, power, t-, chi-square and permutation tests, multiple-comparison corrections, A/B testing and clinical-trial logic; compare ML models without fooling yourself. |

### Part V — From statistics to learning: why generalisation works
| # | Module | You will be able to… |
|---|--------|----------------------|
| 13 | [Linear models and regularisation](part-5-learning-theory/13-linear-models-and-regularization/README.md) | Derive OLS three ways; use ridge and lasso (and understand *why* they shrink or sparsify); fit logistic regression and GLMs; derive and simulate the bias–variance decomposition; select models with cross-validation. |
| 14 | [Information theory](part-5-learning-theory/14-information-theory/README.md) | Compute entropy, cross-entropy, KL divergence and mutual information; connect them to compression, loss functions, decision trees, the ELBO and language-model perplexity and temperature. |
| 15 | [Generalisation and learning theory](part-5-learning-theory/15-generalization-and-learning-theory/README.md) | Use concentration inequalities and union bounds to bound test error; understand VC dimension, PAC learning, capacity control, double descent, implicit regularisation, no-free-lunch and distribution shift. |

### Part VI — Bridges to deep learning and modern AI
| # | Module | You will be able to… |
|---|--------|----------------------|
| 16 | [Neural networks from scratch](part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md) | Build, train and gradient-check a multi-layer perceptron in NumPy; derive the softmax–cross-entropy gradient, Xavier/He initialisation and batch normalisation; diagnose vanishing/exploding gradients. |
| 17 | [The mathematics of transformers](part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md) | Implement scaled dot-product and multi-head attention; derive the √d scaling and the softmax Jacobian; understand positional encodings and RoPE as rotations, layer norm, the KV cache, decoding strategies and LoRA. |
| 18 | [Sampling, MCMC and generative models](part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md) | Sample by inverse-CDF, rejection and importance sampling; run Metropolis–Hastings and Gibbs; derive the ELBO and the VAE loss; derive the forward and reverse processes of diffusion models. |
| 19 | [Markov decision processes and reinforcement learning](part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl/README.md) | Solve bandits with UCB and Thompson sampling; write and solve Bellman equations; implement value iteration, Q-learning and REINFORCE; understand the maths behind PPO, RLHF and DPO. |

Reference material: [notation](docs/notation.md) · [glossary](docs/glossary.md) · [further reading](docs/further-reading.md) · [how to use this course](docs/how-to-use-this-course.md) · [study plans](docs/study-plans.md)

## Getting started

```bash
git clone https://github.com/dthemann/AI-math.git
cd AI-math
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                                            # or: jupyter notebook
```

Then open `part-1-linear-algebra/01-vectors-and-geometry/README.md`, read, and start `lesson.ipynb`.

Everything runs on a laptop CPU with **NumPy, SciPy, Matplotlib, pandas and scikit-learn** only. There is deliberately **no PyTorch/TensorFlow/JAX**: the point of the course is to build the maths yourself, and every neural network, attention block, sampler and policy-gradient method here is written in plain NumPy. No notebook downloads data; all datasets are either generated synthetically or bundled with scikit-learn. Each notebook executes in well under two minutes.

### Verifying the notebooks

```bash
python tools/run_notebooks.py                 # executes all 57 notebooks and checks every solution self-check
python tools/run_notebooks.py part-3-probability   # just one part
```

The GitHub Actions workflow in `.github/workflows/notebooks.yml` does the same on every push.

## Learning principles behind the design

* **Derive, don't decree.** Short derivations are included wherever they fit in ten lines; otherwise the guide states the result and says what the proof hinges on.
* **Numbers first.** Every concept is worked through with small, concrete numbers you can follow by hand before it is generalised.
* **From scratch, then the library.** The notebooks build each object in NumPy and then confirm that `np.linalg`, `scipy.stats` or `scikit-learn` give the same answer.
* **Real problems.** Examples come from diagnostics and clinical trials, finance, e-commerce, manufacturing, language, vision, physics and sport — because the same maths runs all of them.
* **Feedback loops.** The exercises tell you immediately whether you are right, including on questions whose answers are hidden, so you can test your understanding honestly.
* **Everything connects.** Each guide ends with a table mapping its concepts to the exact places they appear in ML, DL and modern AI systems, and links forward and backward across the course.

## Contributing

Corrections, better examples and new exercises are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the conventions (notebook structure, self-check helpers, how to test).

## License

Code and text are released under the [MIT License](LICENSE). Datasets used are those bundled with scikit-learn (BSD-licensed) or generated synthetically.
