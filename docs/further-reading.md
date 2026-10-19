# Further reading

A curated shelf. Everything here is a real, widely used reference; items marked **(free)** are legally available online from the authors or publishers. Each module's concept guide ends with its own, more specific list.

## Books that cover most of the course

| Reference | Why |
|-----------|-----|
| Deisenroth, Faisal & Ong — *Mathematics for Machine Learning* (CUP, 2020) **(free: mml-book.github.io)** | The closest single book to this course: linear algebra, analytic geometry, matrix decompositions, vector calculus, probability, optimisation, then PCA, GMMs, SVMs. |
| Boyd & Vandenberghe — *Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares* (CUP, 2018) **(free: stanford.edu/~boyd/vmls)** | Gentle, example-rich linear algebra with least squares at its centre. |
| Murphy — *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022) and *Advanced Topics* (2023) **(free: probml.github.io/pml-book)** | Encyclopaedic, modern, probabilistic view of ML; the "Advanced Topics" volume covers VI, MCMC, generative models and RL. |
| Bishop — *Pattern Recognition and Machine Learning* (Springer, 2006) **(free PDF from Microsoft Research)** | The classic probabilistic ML text; superb on Bayesian linear models, EM, variational inference. |
| Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning* (2nd ed., Springer, 2009) **(free: hastie.su.domains/ElemStatLearn)** | Statistical learning from the statisticians' side: linear models, regularisation, model selection, boosting. |
| James, Witten, Hastie, Tibshirani (& Taylor) — *An Introduction to Statistical Learning* (2nd ed. 2021, Python ed. 2023) **(free: statlearning.com)** | Same ideas as ESL at a gentler level, with labs. |

## Part I — Linear algebra

* Strang — *Introduction to Linear Algebra* (6th ed., Wellesley-Cambridge, 2023) and the MIT 18.06 lectures **(free on MIT OpenCourseWare)**.
* 3Blue1Brown — *Essence of Linear Algebra* video series **(free)** — the best geometric intuition for eigenvectors and determinants anywhere.
* Axler — *Linear Algebra Done Right* (4th ed., Springer, 2024) **(free PDF, Open Access)** — the coordinate-free, proof-based view.
* Trefethen & Bau — *Numerical Linear Algebra* (SIAM, 1997) — SVD, QR, conditioning and stability for people who compute.
* Shlens — *A Tutorial on Principal Component Analysis* (arXiv:1404.1100, 2014) **(free)**.

## Part II — Calculus and optimisation

* Boyd & Vandenberghe — *Convex Optimization* (CUP, 2004) **(free: stanford.edu/~boyd/cvxbook)** — convexity, duality, KKT, interior-point methods.
* Nocedal & Wright — *Numerical Optimization* (2nd ed., Springer, 2006) — line search, trust regions, quasi-Newton, constrained methods.
* Ruder — *An overview of gradient descent optimization algorithms* (arXiv:1609.04747, 2016) **(free)** — momentum, Adagrad, RMSProp, Adam in one place.
* Kingma & Ba — *Adam: A Method for Stochastic Optimization* (arXiv:1412.6980, 2014) **(free)**.
* Baydin, Pearlmutter, Radul & Siskind — *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018; arXiv:1502.05767) **(free)**.
* Karpathy — *micrograd* (GitHub) and the lecture *The spelled-out intro to neural networks and backpropagation* **(free)** — a scalar autograd engine built live.
* Petersen & Pedersen — *The Matrix Cookbook* **(free PDF)** — every matrix-calculus identity you will ever need.

## Part III — Probability

* Blitzstein & Hwang — *Introduction to Probability* (2nd ed., CRC, 2019) **(free online; Harvard Stat 110 lectures on YouTube)** — the stories behind the distributions.
* Ross — *A First Course in Probability* (10th ed., Pearson, 2018).
* Grimmett & Stirzaker — *Probability and Random Processes* (4th ed., OUP, 2020) — for Markov chains and limit theorems with full proofs.
* Wasserman — *All of Statistics* (Springer, 2004) — a brisk, complete bridge from probability to statistics to learning.

## Part IV — Statistics

* Efron & Hastie — *Computer Age Statistical Inference* (CUP, 2016) **(free PDF from the authors)** — frequentist, Bayesian and Fisherian inference, the bootstrap, and modern large-scale testing.
* Gelman, Carlin, Stern, Dunson, Vehtari & Rubin — *Bayesian Data Analysis* (3rd ed., CRC, 2013) **(free PDF from the authors)**.
* McElreath — *Statistical Rethinking* (2nd ed., CRC, 2020; lectures **free** on YouTube) — the most intuitive Bayesian course in existence.
* Casella & Berger — *Statistical Inference* (2nd ed., 2002) — MLE, sufficiency, Cramér–Rao, hypothesis testing, rigorously.
* Kohavi, Tang & Xu — *Trustworthy Online Controlled Experiments* (CUP, 2020) — A/B testing as practised at scale, including peeking, novelty effects and metric design.
* Gelman, Hill & Vehtari — *Regression and Other Stories* (CUP, 2020) — regression as a tool for thinking, with all the pitfalls.
* Benjamini & Hochberg — *Controlling the False Discovery Rate* (JRSS-B, 1995).

## Part V — Learning theory and information theory

* Shalev-Shwartz & Ben-David — *Understanding Machine Learning: From Theory to Algorithms* (CUP, 2014) **(free PDF from the authors)** — PAC learning, VC dimension, Rademacher complexity, SRM.
* Mohri, Rostamizadeh & Talwalkar — *Foundations of Machine Learning* (2nd ed., MIT Press, 2018) **(free PDF)**.
* Cover & Thomas — *Elements of Information Theory* (2nd ed., Wiley, 2006).
* MacKay — *Information Theory, Inference, and Learning Algorithms* (CUP, 2003) **(free: inference.org.uk/mackay/itila)** — compression, codes, Bayesian inference and neural networks in one brilliant book.
* Zhang, Bengio, Hardt, Recht & Vinyals — *Understanding deep learning requires rethinking generalization* (ICLR 2017; arXiv:1611.03530).
* Belkin, Hsu, Ma & Mandal — *Reconciling modern machine-learning practice and the classical bias–variance trade-off* (PNAS 2019; arXiv:1812.11118) — double descent.
* Kaplan et al. — *Scaling Laws for Neural Language Models* (arXiv:2001.08361, 2020); Hoffmann et al. — *Training Compute-Optimal Large Language Models* (arXiv:2203.15556, 2022).

## Part VI — Deep learning and modern AI

* Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016) **(free: deeplearningbook.org)** — Part I is an excellent compact review of exactly the maths in this course.
* Prince — *Understanding Deep Learning* (MIT Press, 2023) **(free: udlbook.github.io/udlbook)** — modern, visual, covers transformers, diffusion and RL.
* Zhang, Lipton, Li & Smola — *Dive into Deep Learning* **(free: d2l.ai)** — runnable, framework-based companion once you have built things in NumPy.
* Nielsen — *Neural Networks and Deep Learning* **(free: neuralnetworksanddeeplearning.com)** — backpropagation explained from first principles.
* Glorot & Bengio — *Understanding the difficulty of training deep feedforward neural networks* (AISTATS 2010); He et al. — *Delving Deep into Rectifiers* (ICCV 2015) — the initialisation papers.
* Ioffe & Szegedy — *Batch Normalization* (ICML 2015); Ba, Kiros & Hinton — *Layer Normalization* (arXiv:1607.06450, 2016).
* Vaswani et al. — *Attention Is All You Need* (NeurIPS 2017; arXiv:1706.03762).
* Phuong & Hutter — *Formal Algorithms for Transformers* (arXiv:2207.09238, 2022) — pseudocode-level precision for every transformer component.
* Su et al. — *RoFormer: Enhanced Transformer with Rotary Position Embedding* (arXiv:2104.09864, 2021).
* Hu et al. — *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv:2106.09685, 2021).
* Alammar — *The Illustrated Transformer* **(free blog)**; Karpathy — *Let's build GPT: from scratch, in code, spelled out* and *nanoGPT* **(free)**.
* Kingma & Welling — *Auto-Encoding Variational Bayes* (arXiv:1312.6114, 2013); Blei, Kucukelbir & McAuliffe — *Variational Inference: A Review for Statisticians* (JASA 2017; arXiv:1601.00670).
* Ho, Jain & Abbeel — *Denoising Diffusion Probabilistic Models* (NeurIPS 2020; arXiv:2006.11239); Song et al. — *Score-Based Generative Modeling through Stochastic Differential Equations* (ICLR 2021; arXiv:2011.13456); Luo — *Understanding Diffusion Models: A Unified Perspective* (arXiv:2208.11970, 2022).
* Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018) **(free: incompleteideas.net/book/the-book-2nd.html)**.
* Lattimore & Szepesvári — *Bandit Algorithms* (CUP, 2020) **(free PDF from the authors)**.
* Schulman et al. — *Proximal Policy Optimization Algorithms* (arXiv:1707.06347, 2017); Ouyang et al. — *Training language models to follow instructions with human feedback* (arXiv:2203.02155, 2022); Rafailov et al. — *Direct Preference Optimization* (arXiv:2305.18290, 2023).
* OpenAI — *Spinning Up in Deep RL* **(free)** — clear derivations of policy-gradient methods.

## How to read a paper from this list

Read the abstract, the figures and the conclusion first. Then find the one equation the paper is about and re-derive it with the tools from this course — most of the "modern AI" papers above rest on a handful of ideas from Modules 03, 04, 11, 14 and 18. If a derivation step is unclear, that is the place to slow down, not skip.
