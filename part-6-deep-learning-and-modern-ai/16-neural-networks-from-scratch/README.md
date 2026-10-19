# Module 16 — Neural Networks from Scratch

> After this module you can write a multi-layer perceptron in NumPy — forward pass, loss, backward pass, initialisation, normalisation, optimiser and training loop — train it to 98 percent accuracy on handwritten digits, and explain every design choice in it with a formula rather than a slogan.

## Why this matters for ML and AI

Every deep learning system you have heard of is built from the same two ingredients: affine maps $XW + b$ and elementwise nonlinearities, stacked. A transformer block in GPT-2 is an attention layer followed by exactly the two-layer network of this module (768 → 3,072 → 768 units with a GELU in between), and that MLP holds 46 percent of the model's 124 million parameters. A convolutional network is an MLP with most of its weights forced to be zero or equal. A recurrent network is an MLP applied to its own output. If you understand the vectorised forward and backward pass of a plain MLP, the rest of modern AI is a set of variations on where the weights are allowed to go.

The mathematics also explains the things that go wrong. Why does a 30-layer network with sigmoid units refuse to train, while a 152-layer ResNet trains easily? Because the gradient is a product of 30 Jacobians each of norm at most $1/4$, and a residual connection adds an identity to every factor. Why does initialising weights with standard deviation 0.01 work for a 3-layer network and silently kill a 10-layer one? Because the variance of each layer's output is multiplied by $n_{\text{in}} \sigma_w^2$ per layer, and $0.0256^{10} \approx 10^{-16}$. Why is cross-entropy used for classification and not squared error? Because the gradient of a sigmoid-plus-squared-error output at a confident wrong answer is 75 times smaller than that of cross-entropy. Each of these is a two-line calculation once you have the vocabulary, and this module gives you that vocabulary.

Finally, building the network yourself is the fastest way to stop treating frameworks as magic. After this module `nn.Linear`, `nn.CrossEntropyLoss`, `kaiming_normal_`, `nn.BatchNorm1d`, `nn.Dropout` and `model.eval()` are names for six functions you have written and tested with finite differences, and the shape error or `NaN` loss that used to cost an afternoon becomes a thirty-second diagnosis.

## Learning objectives

- Explain why a single affine layer cannot represent XOR, construct a two-layer network that does, and count the parameters of any MLP from its layer sizes.
- State the universal approximation theorem precisely, build the bump-function intuition behind it, and say what it does *not* promise about learning.
- Write down the formulas and derivatives of sigmoid, tanh, ReLU, leaky ReLU, softplus, SiLU and GELU, and prove that depth without nonlinearity collapses to a single affine map.
- Run the forward pass of an MLP on a mini-batch with explicit shapes at every step, using broadcasting for the bias, and estimate its cost in FLOPs.
- Derive the gradient $(P - Y)/n$ of softmax cross-entropy, implement it with the log-sum-exp trick, and explain why squared error is the wrong loss for classification.
- Derive the backward pass of a two-layer network in matrix form, implement it as `Dense.backward`, and verify it with a finite-difference gradient check.
- Derive the variance-propagation rule $\operatorname{Var}(z) = n_{\text{in}} \operatorname{Var}(w) \operatorname{Var}(x)$ and from it the Xavier and He initialisations, including where the factor 2 comes from.
- Diagnose vanishing and exploding gradients as products of Jacobians and explain how ReLU, careful initialisation, batch normalisation, residual connections and gradient clipping each attack the problem.
- Implement batch normalisation (training and inference modes) and inverted dropout, and explain with an expectation argument why each is scaled the way it is.
- Write a complete mini-batch training loop with Adam, a learning-rate schedule, weight decay and early stopping, and read its loss curves correctly.

## Prerequisites

- [Module 02 — Matrices and linear maps](../../part-1-linear-algebra/02-matrices-and-linear-maps/README.md): matrix products, transposes, shape bookkeeping; a matrix product $XW$ as "every row of $X$ mapped by the same linear map".
- [Module 03 — Eigenvalues, SVD and PCA](../../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md): powers of a matrix grow or shrink according to its largest singular value — the reason recurrent gradients explode or vanish (Section 11).
- [Module 04 — Differential calculus](../../part-2-calculus-and-optimization/04-differential-calculus/README.md): the chain rule, Jacobians, and the matrix-calculus facts $\partial (XW) / \partial W$ and $\partial (XW) / \partial X$.
- [Module 05 — Optimization](../../part-2-calculus-and-optimization/05-optimization/README.md): SGD, momentum, Adam and AdamW, learning-rate schedules; this module *uses* those optimisers and supplies the gradients they consume.
- [Module 06 — Autodiff and backpropagation](../../part-2-calculus-and-optimization/06-autodiff-and-backpropagation/README.md): vector-Jacobian products ($\bar A = G B^\top$, $\bar B = A^\top G$, broadcast ⇒ sum), the backprop recurrence for a single example, and gradient checking with central differences. Section 5 of this module is that recurrence written for a whole mini-batch at once.
- [Module 11 — Likelihood and Bayesian inference](../../part-4-statistics/11-likelihood-and-bayesian-inference/README.md): squared error as Gaussian negative log-likelihood, cross-entropy as categorical negative log-likelihood, and weight decay as a Gaussian prior.
- [Module 13 — Linear models and regularisation](../../part-5-learning-theory/13-linear-models-and-regularization/README.md): logistic regression — the zero-hidden-layer special case of everything here — and the bias–variance trade-off.
- [Module 14 — Information theory](../../part-5-learning-theory/14-information-theory/README.md): cross-entropy and KL divergence, which give the classification loss its meaning.

**A note on conventions.** Throughout this module, and in all three notebooks, data matrices are *rows-are-examples*: $X \in \mathbb{R}^{n \times d}$ holds $n$ examples with $d$ features each, and a layer computes $Z = XW + b$ with $W \in \mathbb{R}^{n_{\text{in}} \times n_{\text{out}}}$ and $b$ broadcast across rows. Module 06 wrote the single-example version as a column vector, $z = Wx + b$ with $W$ of shape $(n_{\text{out}}, n_{\text{in}})$; the two are transposes of each other, and every formula below is the row form. Keras and TensorFlow store the kernel exactly as we do; PyTorch's `nn.Linear` stores the transpose, shape `(out_features, in_features)`, and computes `x @ W.T + b`. Same mathematics, different storage.

## 1. From linear models to networks

**Intuition.** Logistic regression (Module 13) draws a single straight line — a hyperplane in $d$ dimensions — through feature space and calls one side "class 1". Many real decision rules are not like that. "Approve the loan if income is high *or* savings are high, but not if both are low *or* both are suspiciously high" is a rule a single line cannot express. The fix is not a cleverer line but a *change of coordinates learned from data*: first compute a handful of new features, each itself a thresholded linear function of the inputs, then draw the line in the new coordinates. A neural network is that idea, repeated.

**Definition.** A *perceptron* or single affine unit computes $\hat y = \phi(x^\top w + b)$ for a weight vector $w \in \mathbb{R}^d$, bias $b \in \mathbb{R}$ and a nonlinearity $\phi$ (a step function in Rosenblatt's 1958 original, a sigmoid in logistic regression). A *multi-layer perceptron* (MLP) with one hidden layer of width $h$ computes, for a single example $x \in \mathbb{R}^d$ written as a row vector,

$$
f(x) = \phi\!\big(x W_1 + b_1\big)\, W_2 + b_2, \qquad W_1 \in \mathbb{R}^{d \times h},\ b_1 \in \mathbb{R}^{h},\ W_2 \in \mathbb{R}^{h \times k},\ b_2 \in \mathbb{R}^{k},
$$

with $\phi$ applied elementwise. With $L$ layers the pattern is $A_0 = x$, $A_l = \phi(A_{l-1} W_l + b_l)$ for $l < L$, and a final affine layer $A_{L-1} W_L + b_L$ without nonlinearity whose output feeds the loss. The *parameter count* of a layer is $n_{\text{in}} n_{\text{out}} + n_{\text{out}}$, so a network with layer sizes $d, h_1, \dots, h_{L-1}, k$ has $\sum_l (n_l + 1)\, n_{l+1}$ parameters.

**Why XOR needs a hidden layer.** The XOR function on $\{0,1\}^2$ is $1$ for $(0,1)$ and $(1,0)$ and $0$ for $(0,0)$ and $(1,1)$. Suppose a single affine unit separated them: $w_1 x_1 + w_2 x_2 + b > 0$ on the two positive points and $\le 0$ on the two negative ones. Adding the two positive inequalities gives $w_1 + w_2 + 2b > 0$. The negative point $(0,0)$ gives $b \le 0$ and $(1,1)$ gives $w_1 + w_2 + b \le 0$; adding these, $w_1 + w_2 + 2b \le 0$. Contradiction — no line works, in any number of training steps, with any optimiser. (Minsky and Papert made this observation in 1969, and it stalled the field for a decade.)

**Worked example.** Two hidden ReLU units solve XOR. Take $W_1 = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$, $b_1 = (0, -1)$, $W_2 = \begin{pmatrix} 1 \\ -2 \end{pmatrix}$, $b_2 = 0$. The hidden pre-activations are $x_1 + x_2$ and $x_1 + x_2 - 1$:

| $x$ | $x W_1 + b_1$ | $A_1 = \mathrm{ReLU}(\cdot)$ | $A_1 W_2$ |
|---|---|---|---|
| $(0,0)$ | $(0, -1)$ | $(0, 0)$ | $0$ |
| $(0,1)$ | $(1, 0)$ | $(1, 0)$ | $1$ |
| $(1,0)$ | $(1, 0)$ | $(1, 0)$ | $1$ |
| $(1,1)$ | $(2, 1)$ | $(2, 1)$ | $2 - 2 = 0$ |

The first hidden unit counts how many inputs are on; the second fires only when both are. In the hidden coordinates $(A_{11}, A_{12})$ the four points sit at $(0,0), (1,0), (1,0), (2,1)$, and the line $a_1 - 2a_2 = 0.5$ separates them. The network has $2 \cdot 2 + 2 + 2 \cdot 1 + 1 = 9$ parameters. For the digit classifier in the notebooks, 64 inputs → 128 hidden → 10 outputs gives $65 \cdot 128 + 129 \cdot 10 = 8{,}320 + 1{,}290 = 9{,}610$ parameters.

**Universal approximation.** The theorem (Cybenko 1989 for sigmoids; Hornik 1991; Leshno et al. 1993 for any non-polynomial activation) says: for every continuous $g$ on a compact set $K \subset \mathbb{R}^d$ and every $\varepsilon > 0$ there is a one-hidden-layer network $f$ with finitely many units such that $\sup_{x \in K} |f(x) - g(x)| < \varepsilon$. The intuition in one dimension: the difference of two ReLUs, $\mathrm{ReLU}(x - a) - \mathrm{ReLU}(x - a - \delta)$, is a ramp that rises from 0 to $\delta$ between $a$ and $a + \delta$ and stays flat afterwards; the difference of two such ramps is a bump; a weighted sum of bumps is a histogram, and histograms approximate any continuous function. Concretely, a ReLU network with $m$ hidden units interpolates $g(x) = x^2$ on $[0,1]$ at $m+1$ equally spaced knots with maximum error $1/(4m^2)$: two hidden units ($0.5\,\mathrm{ReLU}(x) + \mathrm{ReLU}(x - 0.5)$) give error $0.0625$, eight give $0.0039$, fifty give $10^{-4}$.

What the theorem does **not** say: how many units are needed (for most functions the number grows exponentially in $d$ for one hidden layer, which is the argument for depth), that gradient descent will *find* those weights, or that the network will generalise from finitely many samples. Approximation, optimisation and generalisation are three separate questions, and the theorem answers only the first.

**Real-life example (finance).** A bank predicts credit default from 30 tabular features — income, debt ratio, number of recent enquiries, months since last delinquency. Logistic regression gives each feature one coefficient, so "high debt ratio is risky" must hold regardless of income. A 30 → 64 → 64 → 1 MLP ($31 \cdot 64 + 65 \cdot 64 + 65 = 6{,}209$ parameters) learns hidden units such as "debt ratio above 0.4 *and* income below 30k" and weighs them in the final layer, which is exactly the XOR mechanism at scale. On tabular data the gain over a well-tuned gradient-boosted tree is often small, but the same architecture is what makes the model trainable end to end with other components (text, embeddings) in a single loss.

**In ML/AI this is …** `nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, k))` in PyTorch, `keras.Sequential([Dense(h, activation="relu"), Dense(k)])`, and the feed-forward block `Linear(d, 4d) → GELU → Linear(4d, d)` inside every transformer layer, which is a one-hidden-layer MLP applied to every token position independently.

## 2. Activation functions

**Intuition.** The nonlinearity is what makes depth count. Composing affine maps gives an affine map: $(xW_1 + b_1) W_2 + b_2 = x (W_1 W_2) + (b_1 W_2 + b_2)$, so a 100-layer network with no activation function is a single matrix and can still only represent linear decision boundaries — the XOR argument of Section 1 applies to it unchanged. The activation bends the space between layers. Which activation you choose decides two things: how the forward signal is squashed (saturation) and how much of the gradient survives the backward pass (the derivative).

**Definitions.** With $\sigma(z) = 1/(1 + e^{-z})$ the logistic sigmoid and $\Phi$ the standard normal CDF:

| name | $\phi(z)$ | $\phi'(z)$ | range | notes |
|---|---|---|---|---|
| sigmoid | $\sigma(z)$ | $\sigma(z)(1 - \sigma(z))$ | $(0, 1)$ | $\phi' \le 1/4$; saturates both sides; output not zero-centred |
| tanh | $\tanh z = 2\sigma(2z) - 1$ | $1 - \tanh^2 z$ | $(-1, 1)$ | $\phi' \le 1$; zero-centred; still saturates |
| ReLU | $\max(z, 0)$ | $\mathbb{1}[z > 0]$ | $[0, \infty)$ | no saturation for $z > 0$; exactly zero gradient for $z < 0$ |
| leaky ReLU | $\max(z, \alpha z)$, $\alpha \approx 0.01$ | $1$ if $z > 0$ else $\alpha$ | $\mathbb{R}$ | gradient never exactly zero |
| softplus | $\ln(1 + e^z)$ | $\sigma(z)$ | $(0, \infty)$ | smooth ReLU; used for positive outputs (variances, rates) |
| SiLU / Swish | $z\,\sigma(z)$ | $\sigma(z)\big(1 + z(1 - \sigma(z))\big)$ | $\ge -0.278$ | smooth, non-monotonic; derivative can exceed 1 |
| GELU | $z\,\Phi(z)$ | $\Phi(z) + z\,\varphi(z)$ | $\ge -0.17$ | $\varphi$ = normal PDF; the transformer default |

Each derivative is a one-line calculation: sigmoid from $\sigma' = \sigma(1 - \sigma)$ (quotient rule on $1/(1 + e^{-z})$), tanh from $\tanh = 2\sigma(2z) - 1$, SiLU and GELU by the product rule.

**Worked example.** At $z = 2$: $\sigma(2) = 0.8808$ with slope $0.8808 \cdot 0.1192 = 0.1050$; $\tanh 2 = 0.9640$ with slope $1 - 0.9293 = 0.0707$; ReLU gives $2$ with slope $1$; softplus gives $\ln(1 + e^2) = 2.1269$ with slope $\sigma(2) = 0.8808$; SiLU gives $2 \cdot 0.8808 = 1.7616$ with slope $0.8808\,(1 + 2 \cdot 0.1192) = 1.0908$; GELU gives $2 \cdot \Phi(2) = 2 \cdot 0.9772 = 1.9545$ with slope $0.9772 + 2 \cdot 0.0540 = 1.0852$. At $z = -2$: sigmoid slope is again $0.1050$, tanh slope $0.0707$, ReLU is $0$ with slope $0$, leaky ReLU is $-0.02$ with slope $0.01$, GELU is $-0.0455$ with slope $-0.0852$. Two facts to remember from these numbers: at a modest pre-activation of 2 the sigmoid already passes only 10 percent of the gradient and tanh only 7 percent, while ReLU, SiLU and GELU pass all of it or slightly more; and at $z = 5$ the sigmoid slope is $0.0066$ — the unit is *saturated*, and whatever is upstream of it learns 150 times more slowly than it would through a ReLU.

**Dead ReLUs.** A ReLU unit whose pre-activation is negative for every input in the data has zero output and zero gradient for its weights (Module 06's worked example showed this for one unit). Nothing can pull it back, so it is dead for the rest of training. Large learning rates and large negative bias updates are the usual cause; leaky ReLU, careful initialisation (Section 6) and batch normalisation (Section 8) are the usual cures. In a healthy ReLU network roughly half the hidden units are inactive for any given input — that is sparsity, not death; death is a unit inactive for *all* inputs.

**Real-life example (computer vision).** AlexNet, the network that won ImageNet in 2012, replaced the tanh units of earlier convolutional networks with ReLUs and reported that training to 25 percent error on CIFAR-10 became about six times faster — the gradient simply stopped being multiplied by numbers like $0.07$ at every layer. Today's vision backbones (EfficientNet, YOLO variants) use SiLU, and transformer language models use GELU: smooth near zero (which helps optimisation) but ReLU-like for large $|z|$ (which keeps gradients alive).

**In ML/AI this is …** `torch.nn.ReLU`, `nn.GELU` (GPT-2, BERT and most transformers; `approximate="tanh"` uses the fast $0.5 z (1 + \tanh[\sqrt{2/\pi}(z + 0.044715 z^3)])$ form), `nn.SiLU` (the gate inside Llama's SwiGLU feed-forward block), `nn.Softplus` as the output activation that keeps a predicted variance positive, and `F.leaky_relu(negative_slope=0.2)` in GAN discriminators.

## 3. The forward pass with matrices

**Intuition.** Processing one example at a time would mean a Python loop over the data and a tiny matrix-vector product per step, which wastes almost all of a CPU's or GPU's arithmetic capacity. Stack $n$ examples as the rows of a matrix $X$ and the hidden layer becomes a single matrix product $XW_1$: every row of $X$ is pushed through the same linear map in one call. The bias is a single vector added to every row — NumPy's broadcasting does the copying implicitly. This "batch-first" convention is what every framework uses, and getting the shapes right at every step is most of the work of implementing a layer.

**Definition.** For a mini-batch $X \in \mathbb{R}^{n \times d}$ and a network with hidden widths $h_1, \dots, h_{L-1}$ and output width $k$:

$$
Z_1 = X W_1 + \mathbf{1} b_1^\top \in \mathbb{R}^{n \times h_1}, \qquad A_1 = \phi(Z_1), \qquad
Z_l = A_{l-1} W_l + \mathbf{1} b_l^\top, \qquad A_l = \phi(Z_l), \qquad Z_L = A_{L-1} W_L + \mathbf{1} b_L^\top \in \mathbb{R}^{n \times k}.
$$

Here $\mathbf{1} \in \mathbb{R}^n$ is the all-ones column, so $\mathbf{1} b^\top$ is the matrix with $b$ in every row — in code, simply `X @ W + b`, where NumPy broadcasts the shape-`(h,)` bias against the shape-`(n, h)` product. The shape rule is mechanical: $(n \times n_{\text{in}}) \cdot (n_{\text{in}} \times n_{\text{out}}) = (n \times n_{\text{out}})$, and the bias has length $n_{\text{out}}$. The final $Z_L$ holds the *logits*; the loss function (Section 4) turns them into a number.

**Cost.** A matrix product of shapes $(n \times a)(a \times c)$ takes $nac$ multiplications and about as many additions, so $\approx 2nac$ floating-point operations (FLOPs). A layer therefore costs $2 n\, n_{\text{in}} n_{\text{out}}$ FLOPs for the product plus $n\, n_{\text{out}}$ for the bias and $n\, n_{\text{out}}$ for the activation — the product dominates. A useful summary: a forward pass costs about $2 \times (\text{number of weights}) \times (\text{batch size})$ FLOPs. The backward pass costs about twice the forward pass (one product for $\partial W$, one for $\partial A$), so a training step is roughly $6 \times \text{weights} \times \text{batch}$ FLOPs — the "6N" rule used to budget large-model training.

**Worked example.** Two examples, three features, two hidden ReLU units, one output:

$$
X = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 1 & -1 \end{pmatrix},\quad
W_1 = \begin{pmatrix} 0.5 & -1 \\ 1 & 0.5 \\ 0 & 2 \end{pmatrix},\quad
b_1 = (0.1,\ 0.2),\quad
W_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix},\quad b_2 = 0.5 .
$$

First the product, row by row: row 1 of $X$ times column 1 of $W_1$ is $1 \cdot 0.5 + 2 \cdot 1 + 0 \cdot 0 = 2.5$; times column 2 is $1 \cdot (-1) + 2 \cdot 0.5 + 0 \cdot 2 = 0$. Row 2: $0 \cdot 0.5 + 1 \cdot 1 + (-1) \cdot 0 = 1$ and $0 \cdot (-1) + 1 \cdot 0.5 + (-1) \cdot 2 = -1.5$. So

$$
X W_1 = \begin{pmatrix} 2.5 & 0 \\ 1 & -1.5 \end{pmatrix}, \qquad
Z_1 = X W_1 + \mathbf{1} b_1^\top = \begin{pmatrix} 2.6 & 0.2 \\ 1.1 & -1.3 \end{pmatrix}, \qquad
A_1 = \mathrm{ReLU}(Z_1) = \begin{pmatrix} 2.6 & 0.2 \\ 1.1 & 0 \end{pmatrix}.
$$

The second example's second hidden unit is switched off. Then $Z_2 = A_1 W_2 + b_2 = (2.6 - 0.2 + 0.5,\ 1.1 - 0 + 0.5)^\top = (2.9,\ 1.6)^\top$: two logits, one per example, shape $(2 \times 1)$. Shapes at each step: $(2 \times 3)(3 \times 2) \to (2 \times 2) \to (2 \times 2)(2 \times 1) \to (2 \times 1)$. FLOP count: $2 \cdot 2 \cdot 3 \cdot 2 + 2 \cdot 2 \cdot 2 \cdot 1 = 24 + 8 = 32$ for the two products. Keep these numbers — Section 5 runs the backward pass on exactly this example.

**Real-life example (manufacturing and quality control).** A semiconductor fab records 64 sensor readings (temperatures, pressures, gas flows) for every wafer and wants to flag likely defects before the expensive test stage. A mini-batch of 256 wafers is a $256 \times 64$ matrix; a 64 → 128 → 32 → 1 network pushes it through products of shapes $(256 \times 64)(64 \times 128)$, $(256 \times 128)(128 \times 32)$ and $(256 \times 32)(32 \times 1)$, about $2 \cdot 256 \cdot (8{,}192 + 4{,}096 + 32) \approx 6.3$ million FLOPs — a few microseconds on a GPU, and a few milliseconds even on the fab's edge CPU. The same code path, with the same shape discipline, scores one wafer ($n = 1$) in real time and re-trains overnight on a year of wafers ($n = 10^6$, in mini-batches).

**In ML/AI this is …** the `forward` method of `nn.Linear` (`input @ weight.T + bias`, since PyTorch stores the transpose), the batch dimension that is always first in `(batch, features)` and `(batch, seq_len, d_model)` tensors, the `2 · params · tokens` FLOP estimate for a forward pass and the `6 · params · tokens` estimate for training that appear in every scaling-law paper, and the "shape mismatch" `RuntimeError` that is the most common deep learning bug.

## 4. Loss functions and output layers

**Intuition.** The last layer's job is to produce something the loss can compare with the target, and the right pairing of output activation and loss comes from maximum likelihood (Module 11). Predicting a real number? Assume Gaussian noise, and the negative log-likelihood is squared error with an identity output. Predicting a yes/no? Assume a Bernoulli outcome, and the negative log-likelihood is binary cross-entropy with a sigmoid output. Predicting one of $K$ classes? Assume a categorical outcome, and the negative log-likelihood is cross-entropy with a softmax output. In each case the pairing has a gorgeous consequence: the gradient of the loss with respect to the pre-activation is simply *prediction minus target*.

**Definitions.** For a batch of $n$ examples with logits $Z \in \mathbb{R}^{n \times k}$ (one row $z_i$ per example):

- *Regression* ($k$ outputs, targets $Y \in \mathbb{R}^{n \times k}$): $\mathcal{L} = \frac{1}{2n} \lVert Z - Y \rVert_F^2$, with $\partial \mathcal{L} / \partial Z = (Z - Y)/n$. This is the Gaussian negative log-likelihood with unit variance, up to a constant.
- *Binary classification* ($k = 1$, targets $y_i \in \{0, 1\}$): $p_i = \sigma(z_i)$ and $\mathcal{L} = -\frac{1}{n} \sum_i \big[y_i \ln p_i + (1 - y_i) \ln(1 - p_i)\big]$, with $\partial \mathcal{L} / \partial z_i = (p_i - y_i)/n$. The stable form is $\ln(1 + e^{-|z|}) + \max(z, 0) - yz$ (Module 06).
- *Multi-class classification* (targets $y_i \in \{1, \dots, k\}$, one-hot matrix $Y$): $P = \mathrm{softmax}(Z)$ row-wise, $P_{ij} = e^{Z_{ij}} / \sum_{c} e^{Z_{ic}}$, and

$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \ln P_{i, y_i}, \qquad \frac{\partial \mathcal{L}}{\partial Z} = \frac{1}{n}\,(P - Y).
$$

*Derivation of the softmax-CE gradient* (one row, drop the $1/n$). $\ln P_j = z_j - \mathrm{LSE}(z)$ where $\mathrm{LSE}(z) = \ln \sum_c e^{z_c}$ and $\partial\, \mathrm{LSE} / \partial z_m = e^{z_m} / \sum_c e^{z_c} = P_m$. So $\partial \ln P_j / \partial z_m = \mathbb{1}[j = m] - P_m$. The loss is $-\sum_j Y_j \ln P_j$ with exactly one $Y_j = 1$, hence $\partial \mathcal{L} / \partial z_m = -\sum_j Y_j (\mathbb{1}[j = m] - P_m) = -Y_m + P_m$. Averaging over the batch gives the $1/n$.

*The log-sum-exp trick.* $e^{z}$ overflows float64 at $z > 709$ and float32 at $z > 88$. Since softmax is invariant to adding a constant to every logit, subtract the row maximum $m_i = \max_j Z_{ij}$ first: $\ln P_{ij} = (Z_{ij} - m_i) - \ln \sum_c e^{Z_{ic} - m_i}$. Every exponent is now $\le 0$, the largest term is exactly $e^0 = 1$, and no logarithm of zero is ever taken. Compute the loss from $\ln P$ directly and get $P$ as $e^{\ln P}$ for the gradient.

**Worked example.** Logits $z = (2, 0, -1)$, true class 1 (the first entry). Row maximum $m = 2$; shifted logits $(0, -2, -3)$; exponentials $(1,\ 0.1353,\ 0.0498)$ summing to $1.1851$; $\mathrm{LSE} = 2 + \ln 1.1851 = 2.1698$. Probabilities $P = (0.8438,\ 0.1142,\ 0.0420)$. Loss $= -\ln 0.8438 = 0.1698$ (equivalently $\mathrm{LSE} - z_1 = 2.1698 - 2$). Gradient $P - Y = (-0.1562,\ 0.1142,\ 0.0420)$: push the correct logit *up* by 0.156, the wrong ones *down* in proportion to their probability, and the entries sum to zero because adding a constant to all logits changes nothing. If the true class were instead class 3, the loss would be $-\ln 0.0420 = 3.1698$ and the gradient $(0.8438,\ 0.1142,\ -0.9580)$: a confident wrong prediction gets a gradient of magnitude almost 1 on the correct logit — cross-entropy shouts loudest when the model is most wrong.

**Why not squared error for classification?** Pair a sigmoid output with $\mathcal{L} = (\sigma(z) - y)^2$. Then $\partial \mathcal{L} / \partial z = 2 (\sigma(z) - y)\, \sigma(z)(1 - \sigma(z))$ — the error *times the sigmoid's slope*. At $z = -5$ with $y = 1$ the model is confidently wrong ($\sigma = 0.0067$); the cross-entropy gradient is $\sigma - y = -0.993$, but the squared-error gradient is $2 \cdot (-0.993) \cdot 0.0067 = -0.0132$, 75 times smaller. The saturated sigmoid throttles the very correction that is most needed, and training stalls on exactly the examples it gets wrong. Cross-entropy's $\ln$ cancels the sigmoid's $\exp$, so the slope factor disappears. (Squared error on logits is also not a proper likelihood for a Bernoulli outcome, and its minimiser is not the calibrated probability.)

**Label smoothing.** Replacing the one-hot target by $Y^{\text{ls}} = (1 - \varepsilon) Y + \varepsilon / k$ (typically $\varepsilon = 0.1$) keeps the gradient $P - Y^{\text{ls}}$ but makes the optimum finite: the model is asked to put probability $1 - \varepsilon + \varepsilon/k$ on the correct class rather than 1, which caps the logit gap at $\ln\big(((1-\varepsilon)k + \varepsilon)/\varepsilon\big)$ — about $4.5$ for $k = 10$ — instead of letting it grow without bound. It improves calibration and often accuracy by a few tenths of a percent (Szegedy et al., 2016).

**Real-life example (healthcare).** A dermatology classifier assigns a skin lesion image to one of seven diagnoses. The softmax output gives seven probabilities that sum to one, and the cross-entropy loss is the negative log-probability of the dermatologist's label; a lesion the model labels "benign nevus" with probability 0.95 that turns out to be melanoma incurs a loss of $-\ln 0.02 \approx 3.9$ (if melanoma got 2 percent), ten times the loss of a borderline case, so the rare dangerous errors dominate the gradient. Clinicians read the probabilities as calibrated risks, which is only justified because cross-entropy is a proper scoring rule: its expected value is minimised by reporting the true conditional probabilities (Module 11).

**In ML/AI this is …** `nn.CrossEntropyLoss` (log-softmax fused with the negative log-likelihood, exactly the log-sum-exp form above, with a `label_smoothing=` argument), `nn.BCEWithLogitsLoss` for the binary case, `nn.MSELoss` for regression, and the `P - Y` that is the first thing any language model's backward pass computes over its vocabulary-sized logits.

## 5. The backward pass, derived

**Intuition.** Module 06 showed that reverse-mode differentiation of a matrix product needs only two more matrix products, and that a broadcast in the forward pass becomes a sum in the backward pass. An MLP is a chain of exactly those operations, so its backward pass is a chain of exactly those rules, applied from the loss back to the first layer. The quantity that travels backwards is $G_l = \partial \mathcal{L} / \partial Z_l$, an $n \times h_l$ matrix with one row per example — the batched version of Module 06's $\delta_l$.

**Definition (the recurrence).** With the forward quantities of Section 3 and $G_L = \partial \mathcal{L} / \partial Z_L$ from Section 4 (for softmax-CE, $(P - Y)/n$; for squared error, $(Z_L - Y)/n$), for $l = L, L-1, \dots, 1$:

$$
\frac{\partial \mathcal{L}}{\partial W_l} = A_{l-1}^\top G_l \ \ (n_{\text{in}} \times n_{\text{out}}), \qquad
\frac{\partial \mathcal{L}}{\partial b_l} = G_l^\top \mathbf{1} = \sum_{i=1}^n (G_l)_{i,:} \ \ (n_{\text{out}}), \qquad
\frac{\partial \mathcal{L}}{\partial A_{l-1}} = G_l W_l^\top \ \ (n \times n_{\text{in}}),
$$

and then through the activation, $G_{l-1} = \big(G_l W_l^\top\big) \odot \phi'(Z_{l-1})$. Each formula is a Module 06 rule: $Z = AW$ gives $\bar W = A^\top G$ and $\bar A = G W^\top$; the bias was broadcast to every row, so its gradient sums over rows; the elementwise activation contributes an elementwise mask. The shape check is automatic — $A_{l-1}^\top G_l$ is $(n_{\text{in}} \times n)(n \times n_{\text{out}})$, the shape of $W_l$, and $G_l W_l^\top$ is $(n \times n_{\text{out}})(n_{\text{out}} \times n_{\text{in}})$, the shape of $A_{l-1}$. Any other arrangement either fails to multiply or has the wrong shape. Note what must be stored from the forward pass: every layer's input $A_{l-1}$ (for $\partial W_l$) and its pre-activation $Z_l$ or mask $\phi'(Z_l)$. That is why training memory scales with batch size × depth × width.

**Worked example.** Continue Section 3 with targets $y = (3, 1)^\top$ and squared error $\mathcal{L} = \frac{1}{2n} \sum_i (Z_{2,i} - y_i)^2$, $n = 2$. Residuals $Z_2 - y = (-0.1,\ 0.6)^\top$, so $\mathcal{L} = \tfrac{1}{4}(0.01 + 0.36) = 0.0925$ and

$$
G_2 = \frac{Z_2 - y}{n} = \begin{pmatrix} -0.05 \\ 0.3 \end{pmatrix}.
$$

Output layer: $\partial \mathcal{L} / \partial W_2 = A_1^\top G_2 = \begin{pmatrix} 2.6 & 1.1 \\ 0.2 & 0 \end{pmatrix}\begin{pmatrix} -0.05 \\ 0.3 \end{pmatrix} = \begin{pmatrix} -0.13 + 0.33 \\ -0.01 + 0 \end{pmatrix} = \begin{pmatrix} 0.2 \\ -0.01 \end{pmatrix}$ and $\partial \mathcal{L} / \partial b_2 = -0.05 + 0.3 = 0.25$.

Back through $W_2$: $\partial \mathcal{L} / \partial A_1 = G_2 W_2^\top = \begin{pmatrix} -0.05 \\ 0.3 \end{pmatrix}\begin{pmatrix} 1 & -1 \end{pmatrix} = \begin{pmatrix} -0.05 & 0.05 \\ 0.3 & -0.3 \end{pmatrix}$. The ReLU mask is $\mathbb{1}[Z_1 > 0] = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$, so

$$
G_1 = \begin{pmatrix} -0.05 & 0.05 \\ 0.3 & 0 \end{pmatrix}, \qquad
\frac{\partial \mathcal{L}}{\partial W_1} = X^\top G_1 = \begin{pmatrix} 1 & 0 \\ 2 & 1 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} -0.05 & 0.05 \\ 0.3 & 0 \end{pmatrix} = \begin{pmatrix} -0.05 & 0.05 \\ 0.2 & 0.1 \\ -0.3 & 0 \end{pmatrix}, \qquad
\frac{\partial \mathcal{L}}{\partial b_1} = (0.25,\ 0.05).
$$

Check one entry by hand: $W_{1,[3,1]}$ (third feature → first hidden unit) only matters through example 2 (example 1 has $x_3 = 0$), where $\partial Z_{1,[2,1]} / \partial W_{1,[3,1]} = x_{2,3} = -1$, the unit is active, and $\partial \mathcal{L} / \partial Z_{1,[2,1]} = 0.3$; product $-0.3$. ✓ The dead unit (example 2, hidden unit 2) contributes zero to its row of $G_1$, which is why $\partial \mathcal{L} / \partial W_{1,[3,2]} = 0$: that weight is only reached through an inactive unit. Central differences on this example reproduce every entry to $10^{-10}$ — the lesson notebook runs the check.

**Implementation.** The clean way to code this is a `Dense` layer object that stores its input in `forward` and uses it in `backward`:

```python
class Dense:
    def forward(self, X):           # X: (n, n_in)
        self.X = X
        return X @ self.W + self.b  # (n, n_out)
    def backward(self, G):          # G = dL/dZ: (n, n_out)
        self.dW = self.X.T @ G      # (n_in, n_out)
        self.db = G.sum(axis=0)     # (n_out,)
        return G @ self.W.T         # dL/dX: (n, n_in)
```

and a `ReLU` layer whose `backward` returns `G * (Z > 0)`. A network is a list of such layers; `forward` runs them in order, `backward` runs them in reverse, handing each layer's returned gradient to the one before. This is literally how every framework's autograd works, with the bookkeeping automated. Gradient-check it with central differences at relative error below $10^{-7}$ before you train anything (Module 06, Section 7); the lesson notebook does so on a tiny network and gets $7 \times 10^{-11}$.

**Real-life example (NLP).** The feed-forward block of a transformer layer — `Linear(768, 3072) → GELU → Linear(3072, 768)` in GPT-2 — processes a batch of $n = 8 \times 1{,}024 = 8{,}192$ token positions. Its backward pass is the recurrence above with $A_0$ of shape $(8{,}192 \times 768)$: two products of about $2 \cdot 8{,}192 \cdot 768 \cdot 3{,}072 \approx 3.9 \times 10^{10}$ FLOPs each for $\partial W$ and $\partial A$ per linear layer, and a stored $(8{,}192 \times 3{,}072)$ pre-activation (100 MB in float32) for the GELU mask. Multiply by 12 layers and the stored activations of the MLP blocks alone are over a gigabyte — the memory wall of Module 06's case study, made of nothing but `Dense.backward`.

**In ML/AI this is …** the backward kernels `AddmmBackward0` and `ReluBackward0` in a PyTorch autograd graph, `grad_fn` chains in exactly the reverse order of the `nn.Sequential`, the `retain_graph`/activation-memory trade-offs of gradient checkpointing, and `torch.autograd.gradcheck` for the verification step.

## 6. Initialisation and variance propagation

**Intuition.** Before the first gradient step the weights are random numbers, and their *scale* decides whether the network is trainable at all. Each layer multiplies its input by a random matrix; the output's typical size is the input's typical size times a factor that depends on the weight variance and the fan-in. Too small, and after ten layers the activations are $10^{-8}$ and the gradients (which flow through the same matrices) are just as tiny. Too large, and they are $10^{12}$ and the activations saturate or overflow. Exactly right, and the signal keeps its size at every depth — which is a condition you can solve for in closed form.

**Definition.** An *initialisation scheme* draws each weight of a layer independently from a zero-mean distribution with variance $\sigma_w^2$ chosen from the layer's fan-in $n_{\text{in}}$ and fan-out $n_{\text{out}}$, and sets biases to zero. The three standard choices are

$$
\text{LeCun: } \sigma_w^2 = \frac{1}{n_{\text{in}}}, \qquad
\text{Xavier/Glorot: } \sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad
\text{He/Kaiming: } \sigma_w^2 = \frac{2}{n_{\text{in}}},
$$

either as a normal $\mathcal{N}(0, \sigma_w^2)$ or as a uniform $U(-a, a)$ with $a = \sigma_w \sqrt{3}$ (same variance). All three follow from one rule — the variance of a pre-activation is $\operatorname{Var}(z) = n_{\text{in}}\, \sigma_w^2\, \sigma_x^2$ — plus a requirement that the signal's scale be preserved from layer to layer; they differ only in which activation they assume. The derivation follows.

**Why zeros fail.** If $W_1 = 0$ then every hidden unit computes the same thing (the bias), receives the same gradient ($\partial \mathcal{L}/\partial W_1 = X^\top G_1$ with identical columns of $G_1$), and stays identical forever: the layer behaves as a single unit no matter how wide it is. This is the *symmetry* problem, and any constant initialisation has it. Random weights break the symmetry; biases can safely start at zero.

**Derivation.** Consider one pre-activation $z = \sum_{j=1}^{n_{\text{in}}} w_j x_j$ with weights $w_j$ i.i.d. with mean 0 and variance $\sigma_w^2$, inputs $x_j$ i.i.d. with mean 0 and variance $\sigma_x^2$, weights independent of inputs. Each term has mean $E[w_j]E[x_j] = 0$ and variance $E[w_j^2 x_j^2] = \sigma_w^2 \sigma_x^2$ (independence), and the terms are uncorrelated, so

$$
\operatorname{Var}(z) = n_{\text{in}}\, \sigma_w^2\, \sigma_x^2 .
$$

For the variance to be preserved, $\operatorname{Var}(z) = \sigma_x^2$, we need $n_{\text{in}} \sigma_w^2 = 1$, i.e. $\sigma_w = 1/\sqrt{n_{\text{in}}}$. This is the *LeCun* initialisation, appropriate for linear or tanh layers near zero (where $\tanh z \approx z$). Glorot and Bengio (2010) asked that the *backward* signal also keep its variance; the backward pass multiplies by $W^\top$, which has fan-in $n_{\text{out}}$, giving the condition $n_{\text{out}} \sigma_w^2 = 1$. The compromise is **Xavier/Glorot** initialisation, $\sigma_w^2 = 2 / (n_{\text{in}} + n_{\text{out}})$, or the uniform version $U(\pm \sqrt{6/(n_{\text{in}} + n_{\text{out}})})$ which has the same variance.

**The factor 2 for ReLU.** A ReLU sets the negative half of a symmetric input to zero. If $z$ is symmetric about zero with variance $\sigma_z^2$, then $E[\mathrm{ReLU}(z)^2] = \tfrac12 E[z^2] = \tfrac12 \sigma_z^2$ (half the mass is unchanged, half becomes zero). So the next layer's inputs have second moment $\tfrac12 \sigma_z^2$, and the variance of its pre-activations is $n_{\text{in}} \sigma_w^2 \cdot \tfrac12 \sigma_z^2$. Preservation requires $n_{\text{in}} \sigma_w^2 / 2 = 1$, hence

$$
\sigma_w = \sqrt{\frac{2}{n_{\text{in}}}} \qquad \text{(He et al., 2015)}.
$$

The 2 compensates for the half the ReLU throws away. (ReLU outputs have a non-zero mean, $E[\mathrm{ReLU}(z)] = \sigma_z / \sqrt{2\pi}$, so the argument uses second moments rather than variances; the conclusion is the same.) For leaky ReLU with slope $\alpha$ the factor becomes $2/(1 + \alpha^2)$; for SiLU and GELU, which pass roughly half the signal, He initialisation is the standard choice.

**Worked example.** A layer with $n_{\text{in}} = 4$ inputs of unit variance and weights with $\sigma_w = 0.5$: $\operatorname{Var}(z) = 4 \cdot 0.25 \cdot 1 = 1$ — fine for a linear layer. Put a ReLU after it and feed a second identical layer: its inputs have second moment $0.5$, so its pre-activations have variance $4 \cdot 0.25 \cdot 0.5 = 0.5$; after ten such layers the variance is $0.5^{10} \approx 10^{-3}$ and the standard deviation about $0.03$. With He scaling $\sigma_w = \sqrt{2/4} = 0.707$: $\operatorname{Var}(z) = 4 \cdot 0.5 \cdot 0.5 = 1$ at every layer. Now the extreme cases the lesson notebook demonstrates on a width-256 network: $\sigma_w = 0.01$ gives $n_{\text{in}} \sigma_w^2 = 0.0256$ per layer, so after 10 layers the variance is $0.0256^{10} \approx 1.2 \times 10^{-16}$ and the activation standard deviation is $10^{-8}$ — all gradients are effectively zero. $\sigma_w = 1$ gives a factor 256 per layer and $256^{10} \approx 10^{24}$ — the activations overflow float32 by layer 5.

**Real-life example (physics and engineering).** A surrogate model replaces an expensive fluid-dynamics simulation: 40 geometry parameters in, pressure at 2,000 surface points out, with an 8-layer, 512-wide ReLU network trained on 50,000 simulations. The team's first attempt, weights from $\mathcal{N}(0, 0.01^2)$ copied from a tutorial for a 2-layer network, produces a loss that does not move for 20 epochs: $n_{\text{in}} \sigma_w^2 = 512 \cdot 10^{-4} = 0.05$ per layer, so the output of layer 8 is $0.05^{4} \approx 6 \times 10^{-6}$ times the input scale and every gradient is below float32 resolution relative to the Adam $\epsilon$. Switching to He initialisation ($\sigma_w = \sqrt{2/512} = 0.0625$) — a one-line change — brings the loss down by a factor of 10 in the first epoch. The lesson here is not "use He" but "compute $n_{\text{in}} \sigma_w^2$".

**In ML/AI this is …** `torch.nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")` (He) and `xavier_uniform_` (Glorot); PyTorch's `nn.Linear` default, which is Kaiming-uniform with $a = \sqrt{5}$ and gives variance $1/(3 n_{\text{in}})$ — smaller than He, which is why explicit initialisation still matters; GPT-2's $\mathcal{N}(0, 0.02^2)$ with residual projections scaled by $1/\sqrt{2 L}$ so that the sum of $2L$ residual branches keeps unit variance; and the "μP" parameterisation that makes learning rates transfer across widths by fixing exactly these variance rules.

## 7. Vanishing and exploding gradients

**Intuition.** The backward pass through $L$ layers multiplies the output gradient by $L$ Jacobians. A product of many numbers each a little below 1 is almost zero; a product of many numbers each a little above 1 is enormous. Nothing in between is stable unless each factor is held close to exactly 1. Depth is a geometric series, and this section is about controlling its ratio.

**Definition.** From the recurrence of Section 5, the gradient reaching layer $l$ is

$$
G_l = G_L \prod_{m = L}^{l+1} \Big( W_m^\top \odot\text{-masked by } \phi'(Z_{m-1}) \Big)
\quad\text{so}\quad
\lVert G_l \rVert \;\le\; \lVert G_L \rVert \prod_{m=l+1}^{L} \lVert W_m \rVert_2\, \max |\phi'(Z_{m-1})| .
$$

(The product is of the Jacobians $\partial Z_m / \partial Z_{m-1} = \operatorname{diag}(\phi'(Z_{m-1}))\, W_m$ written in row form.) For sigmoid units, $\max \sigma' = 1/4$, so with weights of spectral norm around 1 the bound shrinks by at least $4$ per layer: $4^{-10} \approx 10^{-6}$ for ten layers, $4^{-30} \approx 10^{-18}$ for thirty. With tanh, $\max \phi' = 1$ but the typical value for pre-activations of unit size is around $0.4$–$0.6$, so the decay is slower but still geometric. With ReLU the mask is exactly 0 or 1, so the activation contributes no decay on the active paths, and the product reduces to a product of weight matrices — whose growth or decay is controlled by initialisation (Section 6). Exploding gradients are the mirror image: spectral norms above 1 compound just as fast, and a single step of size $10^{6}$ destroys the model.

**Worked example.** A chain of ten sigmoid units with unit weights, input $0.5$: the activations converge quickly to the fixed point $a^\star = \sigma(a^\star) \approx 0.659$, where $\sigma' = 0.659 \cdot 0.341 = 0.2247$. The gradient product is $0.235 \cdot 0.227 \cdot 0.225 \cdot 0.2248^7 \approx 3.5 \times 10^{-7}$: a loss gradient of 1 at the output arrives at the input as $3.5 \times 10^{-7}$, and the first layer's weights move $10^{6}$ times more slowly than the last's. Raising the weight to 4 does not help — the fixed point moves to $0.981$, where $\sigma' \cdot w = 0.0761$, and the product is smaller still. There is no weight scale at which a deep sigmoid chain passes gradients: saturation wins either way. The same chain with ReLU units and unit weights passes the gradient *exactly* unchanged along the active path.

**Remedies, and what each does to the product.**

| remedy | effect on the Jacobian product |
|---|---|
| ReLU-family activations | mask entries are 0 or 1; no $1/4$ factor; active paths carry the gradient undiminished |
| Xavier/He initialisation | keeps $\lVert W_m \rVert$ near 1 at the start so the product starts near 1 (Section 6) |
| Batch / layer normalisation | re-standardises each layer's pre-activations every step, so the forward signal and hence the backward Jacobians stay $O(1)$ regardless of the weights (Section 8) |
| Residual connections | each block computes $A_{l} = A_{l-1} + F(A_{l-1})$, so the Jacobian is $I + \partial F / \partial A_{l-1}$; the product of such factors contains an identity path that no depth can shrink |
| Gradient clipping | if $\lVert g \rVert > c$, scale $g \leftarrow c\, g / \lVert g \rVert$; does nothing to vanishing but caps the damage of a single exploding step |
| Shorter paths (LSTM gates, attention) | let information skip many multiplications (Section 11) |

*The residual derivation.* With $A_l = A_{l-1} + F(A_{l-1})$, the chain rule gives $\partial \mathcal{L} / \partial A_{l-1} = \partial \mathcal{L} / \partial A_l \cdot \big(I + \partial F / \partial A_{l-1}\big)$. Expanding the product over $L$ blocks, $\prod_l (I + J_l) = I + \sum_l J_l + \sum_{l < m} J_m J_l + \cdots$: the leading term is the identity, so even if every $J_l$ is tiny the gradient at the input is at least the gradient at the output. In Section 7's chain, replacing each unit by $a \mapsto a + 0.1 \tanh(a)$ turns the ten-layer product into $1.79$ instead of $10^{-7}$. He et al. (2016) trained a 152-layer network this way after showing that a *plain* 56-layer network had higher training error than a 20-layer one — not overfitting but failed optimisation.

**Real-life example (computer vision — why early deep nets failed and ResNets worked).** Between 1998 (LeNet-5, 5 layers) and 2012 (AlexNet, 8 layers) networks barely got deeper, because deeper sigmoid or tanh networks simply did not train — the $4^{-L}$ factor above. ReLU and He initialisation took the field to 19 layers (VGG, 2014) and 22 (GoogLeNet), but VGG's authors reported that initialisation was so delicate they trained shallow versions first and used them to initialise the deeper ones. Batch normalisation (2015) and residual connections (2015) removed the ceiling: ResNet-152 trained in one go, and its 1,202-layer variant was trainable too. The depth of today's models — 96 transformer layers in GPT-3, each a residual block with normalisation — is a direct consequence of these two ideas.

**In ML/AI this is …** `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` in essentially every LLM training script, the residual stream `x = x + attn(ln(x)); x = x + mlp(ln(x))` of a pre-norm transformer block, the `nn.ReLU`/`nn.GELU` choice itself, and the `grad_norm` curve that practitioners watch on their training dashboards for spikes.

## 8. Normalisation layers

**Intuition.** Sections 6 and 7 fought to keep activations at a sensible scale by choosing the weights carefully at initialisation — but the weights change during training, and the scale drifts. Batch normalisation (Ioffe and Szegedy, 2015) stops fighting and simply *re-standardises* each pre-activation to mean 0 and variance 1 over the current mini-batch, then lets the network choose a new scale and shift with two learnable parameters per unit. The layer after it always sees inputs of a known scale, whatever happened upstream, and the learning rate can be much larger.

**Definition (batch norm, forward).** For a pre-activation matrix $Z \in \mathbb{R}^{n \times h}$ and each column (unit) $j$:

$$
\mu_j = \frac{1}{n} \sum_i Z_{ij}, \qquad
\sigma_j^2 = \frac{1}{n} \sum_i (Z_{ij} - \mu_j)^2, \qquad
\hat Z_{ij} = \frac{Z_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \qquad
Y_{ij} = \gamma_j \hat Z_{ij} + \beta_j ,
$$

with learnable $\gamma, \beta \in \mathbb{R}^h$ (initialised to 1 and 0) and $\epsilon \approx 10^{-5}$ for numerical safety. Since the layer subtracts the mean, the preceding `Dense` layer's bias is redundant and is usually dropped.

**Training vs inference.** The statistics $\mu, \sigma^2$ are *batch* statistics, which makes the output of one example depend on the others in its batch — fine for training, unacceptable for serving a single request. During training the layer keeps exponential moving averages, $\mu_{\text{run}} \leftarrow (1 - \alpha)\mu_{\text{run}} + \alpha \mu$ (and likewise for $\sigma^2$, with PyTorch's $\alpha = 0.1$ and the unbiased $n/(n-1)$ variance), and at inference it uses these fixed running values. Forgetting to switch modes — `model.eval()` — is one of the most common deep learning bugs: a model that was 95 percent accurate in validation returns garbage in production because single-example "batch statistics" are $\mu = z$, $\sigma^2 = 0$.

**The backward pass.** $\hat Z$ depends on every example through $\mu$ and $\sigma^2$, so the backward pass is not elementwise. With $G = \partial \mathcal{L} / \partial Y$ (shape $n \times h$) and everything column-wise:

$$
\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_i G_{ij} \hat Z_{ij}, \qquad
\frac{\partial \mathcal{L}}{\partial \beta} = \sum_i G_{ij}, \qquad
\frac{\partial \mathcal{L}}{\partial Z_{ij}} = \frac{\gamma_j}{n \sqrt{\sigma_j^2 + \epsilon}} \Big( n\, G_{ij} - \sum_{i'} G_{i'j} - \hat Z_{ij} \sum_{i'} G_{i'j} \hat Z_{i'j} \Big).
$$

In words: the incoming gradient, minus its batch mean, minus its projection onto $\hat Z$ — batch norm's backward pass removes from the gradient the components that would change the batch mean or variance, because the forward pass would undo them anyway. The derivation is the chain rule through $\mu$ and $\sigma^2$ (about 15 lines; see the exercises). Verify it with finite differences before trusting it.

**Why it helps.** The original paper attributed the benefit to reducing "internal covariate shift". Santurkar et al. (2018) showed experimentally that the shift is not actually reduced much and that the mechanism is *smoother optimisation*: batch norm makes the loss less sensitive to the learning rate and the Jacobians better conditioned (the gradient with respect to $Z$ is bounded by $\gamma / \sigma$, so a layer whose pre-activations have grown large automatically gets a smaller effective gradient), which allows learning rates 10 times larger and makes initialisation nearly irrelevant.

**Layer norm** (Ba, Kiros and Hinton, 2016) normalises *across the features of one example* instead of across the batch: $\mu_i = \frac{1}{h}\sum_j Z_{ij}$, $\sigma_i^2 = \frac{1}{h}\sum_j (Z_{ij} - \mu_i)^2$, same $\gamma, \beta$. There are no batch statistics, so training and inference are identical and batch size 1 works. It is the normalisation used in every transformer, and Module 17 treats its geometry (a projection onto the hyperplane $\sum_j z_j = 0$ followed by scaling to a sphere).

**When batch norm hurts.** Small batches (statistics from 2 or 4 examples are too noisy; the running averages drift), recurrent networks (a different statistic per time step, or statistics that leak across steps), and any setting where examples in a batch are not exchangeable — contrastive learning with positive pairs in the same batch, or federated training on non-i.i.d. shards. Group norm and layer norm are the alternatives.

**Worked example.** One unit, batch of four pre-activations $z = (1, 2, 3, 6)$. Mean $\mu = 3$, biased variance $\sigma^2 = \frac{1}{4}(4 + 1 + 0 + 9) = 3.5$, standard deviation $1.871$. Normalised: $\hat z = (-1.069,\ -0.535,\ 0,\ 1.604)$, which has mean 0 and variance 1 by construction. With $\gamma = 2$, $\beta = 1$: $y = (-1.138,\ -0.069,\ 1,\ 4.207)$. For the running statistics PyTorch would record the unbiased variance $\frac{4}{3} \cdot 3.5 = 4.667$. If this unit had been fed pre-activations a hundred times larger, $z = (100, 200, 300, 600)$, the output $y$ would be *identical* — that scale invariance is the whole point.

**Real-life example (climate).** A weather-emulation model takes gridded atmospheric fields — temperature in kelvin around 280, pressure in pascals around $10^5$, humidity in kg/kg around $10^{-3}$ — as input channels. Without normalisation the pressure channel dominates every dot product by a factor of $10^8$ over humidity, the first layer's effective learning rate differs by that much between channels, and training diverges or ignores humidity entirely. Standardising the inputs once (per-channel mean and standard deviation from the training set) fixes the first layer; batch or layer normalisation inside the network does the same job for every subsequent layer, automatically, as the weights evolve. Every published weather model (GraphCast, Pangu-Weather, FourCastNet) has both.

**In ML/AI this is …** `nn.BatchNorm1d`/`nn.BatchNorm2d` (with `running_mean`, `running_var`, `momentum=0.1`, and the `model.train()`/`model.eval()` switch), `nn.LayerNorm` in every transformer block, `RMSNorm` (layer norm without the mean subtraction, used in Llama), `nn.GroupNorm` in diffusion U-Nets, and the `BatchNorm → fold into Conv weights` step that deployment compilers perform because at inference batch norm is just an affine map.

## 9. Training loop mechanics

**Intuition.** You now have a function that maps parameters to a loss and a function that returns its gradient. Training is "step downhill, repeat", but the details decide whether it takes ten minutes or never finishes: how many examples per gradient estimate, how large a step, how the step size changes over time, when to stop. Each of these has a reason, and the reasons come from Modules 05 and 10.

**Definitions.**

- *Mini-batch.* The full-data gradient $\nabla \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \nabla \ell_i$ is replaced by the average over a random subset of $B$ examples. The estimate is unbiased, with variance proportional to $1/B$ (Module 10), and costs $B/N$ of a full pass. Typical $B$ is 32–512; larger batches give smoother gradients but fewer updates per epoch, and in practice the learning rate is scaled up roughly linearly with $B$ to compensate.
- *Epoch.* One pass through the training set: $\lceil N / B \rceil$ steps. Shuffle the data at the start of every epoch so consecutive mini-batches are independent — presenting the same order each epoch, or sorted data, introduces correlated gradients and visible loss oscillations.
- *Learning rate and schedule.* Module 05's analysis gives the stable range $\eta < 2/\lambda_{\max}$ for the local curvature; in practice you find the largest $\eta$ that does not diverge and then decay it. Common schedules: step decay (divide by 10 every $k$ epochs), cosine $\eta_t = \frac{\eta_0}{2}\big(1 + \cos(\pi t / T)\big)$, and linear warm-up over the first few hundred steps (start at $\approx 0$, ramp to $\eta_0$) which protects the early steps when Adam's variance estimates are still based on a handful of gradients.
- *Weight decay.* Add $\frac{\lambda}{2} \lVert \theta \rVert^2$ to the loss, equivalently $\lambda \theta$ to the gradient; this is the MAP estimate under a Gaussian prior $\theta \sim \mathcal{N}\big(0, 1/(N\lambda)\big)$ when, as here, the data term is the *mean* loss over $N$ examples (the prior variance is $1/\lambda$ if the loss is the summed negative log-likelihood; Module 11). With Adam the decoupled form $\theta \leftarrow \theta - \eta(\hat m / (\sqrt{\hat v} + \epsilon) + \lambda \theta)$ (AdamW) is preferred, because adding $\lambda \theta$ to the gradient would have it divided by $\sqrt{\hat v}$ and regularise large-gradient weights hardly at all (Module 05).
- *Early stopping.* Evaluate the loss on a held-out validation set every epoch; keep the parameters with the lowest validation loss; stop when it has not improved for $p$ epochs ("patience"). It is a regulariser: the number of steps limits how far the weights move from their initialisation.
- *Monitoring.* Plot training and validation loss per epoch. Both falling: keep going. Training falling, validation rising: overfitting — stop, regularise, or get more data. Both high and flat: underfitting or a too-small learning rate — increase capacity or $\eta$. Training loss `NaN`: learning rate too high, missing log-sum-exp, or an exploding gradient — lower $\eta$, clip, check the loss code.

**The template.** Every training script in existence is a decoration of this skeleton:

```python
for epoch in range(n_epochs):
    perm = rng.permutation(n_train)
    for start in range(0, n_train, batch_size):
        idx = perm[start:start + batch_size]
        logits = model.forward(X_train[idx])             # forward
        loss, G = softmax_ce(logits, y_train[idx])       # loss and dL/dlogits
        model.backward(G)                                # gradients into every layer
        optimizer.step(lr=schedule(step))                # Adam / SGD update, weight decay
    val_loss = evaluate(model, X_val, y_val)             # eval mode: no dropout, BN running stats
    if val_loss < best: best, best_params = val_loss, copy(params)
    elif epochs_since_best > patience: break
```

**Worked example.** The digits task has 1,797 images; an 80/20 split gives 1,437 training and 360 test examples. With $B = 64$ an epoch is $\lceil 1437/64 \rceil = 23$ steps, so 30 epochs are 690 steps. A cosine schedule from $\eta_0 = 3 \times 10^{-3}$ over $T = 690$ steps is at $1.5 \times 10^{-3}$ halfway (step 345), $1.2 \times 10^{-4}$ at step 600, and 0 at the end. Weight decay $\lambda = 10^{-2}$ with $\eta = 10^{-3}$ shrinks a weight that receives no gradient by $(1 - 10^{-5})^{690} = 0.993$ over the whole run — a tiny effect at this scale; $\lambda = 0.1$ shrinks it to $0.50$. (In the lesson notebook a 64 → 128 → 10 network with Adam at $\eta = 3 \times 10^{-3}$ and no schedule reaches 87 percent test accuracy after one epoch, 98 percent after five, and about 98 percent after thirty, in well under ten seconds; the training loss keeps falling to $0.002$ while the test loss bottoms out near $0.065$ after about twenty epochs and then stays flat or creeps up slightly — early stopping would halt there with no loss of accuracy.)

**Real-life example (e-commerce).** A recommendation model is retrained nightly on the day's 50 million click events. At $B = 4{,}096$ that is 12,200 steps per epoch, three epochs, warm-up over the first 1,000 steps and cosine decay to zero; validation is the *next* day's first hour of traffic (a time-based split, because a random split would leak tomorrow's items into training). The team watches two curves: validation log-loss, which must beat yesterday's model before deployment, and the gradient norm, whose spikes flag a batch of corrupted events. When the validation loss rises after epoch 2 while training loss falls, they do not train longer — they add weight decay and keep the epoch-2 checkpoint.

**In ML/AI this is …** `torch.utils.data.DataLoader(shuffle=True, batch_size=B)`, `torch.optim.AdamW(lr, weight_decay)`, `torch.optim.lr_scheduler.CosineAnnealingLR` and `LinearLR` for warm-up, `model.train()`/`model.eval()`, the `EarlyStopping` callback in Keras and Lightning, and the training/validation loss panels in TensorBoard and Weights & Biases.

## 10. Regularisation in networks

**Intuition.** A network with 10,000 parameters trained on 1,400 digits can memorise the training set — the training loss goes to zero while the test loss rises. Regularisation makes memorisation harder than generalisation. The methods look different (penalties, random noise, synthetic data) but each either restricts the effective number of parameters or forces the network to be robust to perturbations that do not change the label.

**Definitions.**

*L2 / weight decay* (Section 9) penalises $\lVert \theta \rVert^2$; Module 13 showed that for linear models it shrinks the solution along low-variance directions and is the MAP estimate under a Gaussian prior. For networks it also keeps pre-activations small, which keeps units in the near-linear part of smooth activations and bounds the Lipschitz constant of each layer.

*Dropout* (Srivastava et al., 2014). During training, each hidden activation is kept with probability $q = 1 - p$ and set to zero otherwise, independently per unit and per example. *Inverted* dropout divides the kept activations by $q$:

$$
\tilde A = \frac{M \odot A}{q}, \qquad M_{ij} \sim \mathrm{Bernoulli}(q), \qquad E[\tilde A_{ij}] = \frac{q\, A_{ij}}{q} = A_{ij}.
$$

The scaling keeps the *expected* input to the next layer unchanged, so at test time dropout is simply switched off (no mask, no scaling) and the network sees inputs of the same scale it trained on. The backward pass multiplies the incoming gradient by the same $M / q$. Why it regularises: a unit cannot rely on any particular other unit being present, so co-adapted detectors are penalised; equivalently, training approximates an ensemble of $2^h$ thinned networks whose predictions are averaged at test time. Typical $p$ is 0.1–0.5 for hidden layers, and 0 for the input layer of small networks.

*Data augmentation.* Apply label-preserving transformations — shifts, rotations, crops, flips for images; synonym replacement or back-translation for text; random time shifts for sensor data — to each training example every epoch. It tells the model which directions in input space do *not* matter and is the most effective regulariser available when such transformations are known. *Noise injection* (Gaussian noise on inputs or weights) is the continuous version; for linear models, adding noise of variance $\sigma^2$ to the inputs is exactly L2 regularisation with $\lambda = n\sigma^2$ (Bishop, 1995).

*Early stopping* (Section 9) limits the distance travelled from the initialisation, which for the linearised network is equivalent to L2 regularisation with $\lambda \propto 1/(\eta T)$.

**Why bigger-plus-regularised beats smaller.** Module 15's double descent shows that test error as a function of model size need not be U-shaped: past the point where the network can interpolate the training set, more parameters often *reduce* test error, because among the many zero-training-loss solutions SGD with weight decay and small initialisation finds smoother ones. In practice a 4-times-wider network with dropout 0.5 and weight decay usually beats a narrow network with no regularisation, and this is why the standard advice is "make it big enough to overfit, then regularise".

**Worked example.** Inverted dropout with $p = 0.5$ on a hidden vector $a = (2, -1, 3, 0.5)$: a mask draw $M = (1, 0, 1, 1)$ gives $\tilde a = (2, -1, 3, 0.5) \odot (1, 0, 1, 1) / 0.5 = (4, 0, 6, 1)$. Over many draws each coordinate is kept half the time at twice its value, so the mean is $(2, -1, 3, 0.5)$ — the original. The variance added per coordinate is $a_j^2 (1 - q)/q = a_j^2$, so dropout is multiplicative noise with coefficient of variation 1 at $p = 0.5$. In the lesson notebook, a 64 → 256 → 10 network trained for 400 full-batch Adam steps on only 100 digits drives its training loss to zero within a few steps; without dropout the test loss reaches its minimum of $0.47$ at step 7 and then climbs to $0.63$ by the end (88 percent accuracy), with $p = 0.5$ it ends at $0.58$ (89 percent), and with dropout plus weight decay $0.1$ it falls steadily to $0.39$ with no upturn at all (89 percent). Accuracy barely moves because 100 examples is a very hard regime, but the *shape* of the test-loss curve is exactly what the theory predicts: the unregularised model becomes ever more confidently wrong, the regularised one does not.

**The recipe.** Karpathy's checklist, lightly compressed: (1) get the data pipeline right and look at the examples; (2) overfit a tiny subset to zero loss to prove the model and gradients work; (3) check the initial loss equals the uninformed value ($\ln k$ for $k$ classes — $\ln 10 = 2.303$ for digits); (4) make the model big enough to overfit the real training set; (5) regularise — data augmentation, weight decay, dropout, early stopping, in that order of effectiveness; (6) tune the learning rate and schedule; (7) ensemble or distil if the last percent matters.

**Real-life example (sport).** A football analytics team predicts the probability that a shot becomes a goal (the "expected goals" model) from 20 features — distance, angle, body part, defender positions — using a 20 → 64 → 64 → 1 network trained on 40,000 shots. The unregularised network reaches log-loss 0.21 on training shots and 0.29 on the held-out season: it has memorised specific shot configurations. With dropout 0.3 and weight decay $10^{-4}$ the training log-loss rises to 0.26 and the held-out log-loss falls to 0.26 — a worse fit to the past and a better forecast of the future, which is the only thing the model is for. Mirroring the shots left-right (a label-preserving augmentation that doubles the data) closes the gap further.

**In ML/AI this is …** `nn.Dropout(p)` (inverted, active only in `model.train()`), `weight_decay=0.1` in the AdamW call of every LLM recipe, `torchvision.transforms.RandomResizedCrop`/`RandomHorizontalFlip`, mixup and CutMix in vision, *dropout on attention weights* and *stochastic depth* (randomly skipping whole residual blocks) in transformers, and the `ReduceLROnPlateau` and `EarlyStopping` callbacks.

## 11. Beyond MLPs: the same mathematics, with constraints

**Intuition.** The architectures that dominate practice — convolutional, recurrent, attention-based, embedding-based — are not new kinds of mathematics. Each is a `Dense` layer with a *constraint* on its weight matrix (most entries forced to zero, or forced to be equal, or computed from the data, or indexed by a one-hot input), and each inherits the forward pass of Section 3, the backward pass of Section 5 and the gradient-flow analysis of Section 7 unchanged. Knowing which constraint each one imposes tells you its parameter count, its backward rule and its failure mode.

**Definitions, one paragraph each.**

*Convolution as a linear map with weight sharing.* A 1-D convolution of a signal $x \in \mathbb{R}^{n}$ with a kernel $w \in \mathbb{R}^{m}$ computes $y_i = \sum_{j=0}^{m-1} w_j x_{i+j}$ (cross-correlation, the deep learning convention). This is $y = T x$ for the $(n - m + 1) \times n$ *Toeplitz* matrix $T$ whose row $i$ holds $w$ shifted to start at column $i$ — e.g. for $x = (1, 2, 3, 4)$ and $w = (1, -1)$, $T = \begin{pmatrix} 1 & -1 & 0 & 0 \\ 0 & 1 & -1 & 0 \\ 0 & 0 & 1 & -1 \end{pmatrix}$ and $y = (-1, -1, -1)$. So a convolutional layer *is* a `Dense` layer whose weight matrix has two constraints: it is sparse (each output sees only $m$ inputs, its *receptive field*) and its non-zero entries are shared along the diagonals (the same $w$ everywhere, hence translation equivariance). The parameter count is $m$ instead of $n(n - m + 1)$; a $3 \times 3$ convolution from 64 to 128 channels has $3 \cdot 3 \cdot 64 \cdot 128 + 128 = 73{,}856$ parameters, where a dense layer between the same $32 \times 32 \times 64$ and $32 \times 32 \times 128$ feature maps would have $8.6 \times 10^9$. The backward pass follows from the Toeplitz form: $\partial \mathcal{L} / \partial x = T^\top G$ is a convolution with the flipped kernel, and $\partial \mathcal{L} / \partial w$ sums the per-position products — the gradient of a shared weight is the sum of the gradients of all its copies (Module 06: broadcast ⇒ sum). *Pooling* (max or mean over a window) is a fixed, parameter-free down-sampling; stacking layers of kernel $m$ and stride $s$ grows the receptive field additively, which is how a pixel at the top of a ResNet "sees" the whole image.

*Recurrence as repeated application of one map.* A recurrent network processes a sequence $x_1, \dots, x_T$ by $h_t = \phi(h_{t-1} W_h + x_t W_x + b)$: the *same* $W_h$ at every step, so the unrolled network is an MLP of depth $T$ with tied weights. Backpropagation through time is Section 5's recurrence along $t$, and the gradient from step $T$ to step $t$ contains the product $\prod_{s = t+1}^{T} \operatorname{diag}(\phi'(z_s))\, W_h^\top$. Because the *same* matrix appears $T - t$ times, Module 03 tells you exactly what happens: the product behaves like $\sigma_{\max}(W_h)^{T - t}$ — it explodes if the largest singular value exceeds 1 and vanishes if it is below 1, and for tanh units the $\phi' < 1$ factors only make the vanishing worse. Over 100 time steps there is no margin at all, which is why plain RNNs cannot learn dependencies longer than a few dozen steps and why LSTM/GRU gates (an additive, residual-like path for the cell state) and gradient clipping were invented.

*Attention as data-dependent weighting.* A `Dense` layer mixes *features* with a fixed matrix $W$. Attention mixes *positions* with a matrix that is computed from the data: for a sequence $X \in \mathbb{R}^{T \times d}$, $\mathrm{softmax}(QK^\top / \sqrt{d_k})\, V$ where $Q = X W_Q$, $K = X W_K$, $V = X W_V$. The $T \times T$ softmax matrix is a row-stochastic weighting that depends on $X$ itself, so the "weight matrix" that mixes positions is different for every input, while the learnable parameters ($W_Q, W_K, W_V$) are shared across positions exactly as a convolution kernel is shared across pixels. The path from any position to any other is one step long — no product of $T$ Jacobians — which is the structural reason transformers handle long range dependencies that RNNs cannot. Module 17 derives the $\sqrt{d_k}$, the softmax Jacobian and the rest.

*Embedding layers as lookup = one-hot × matrix.* A categorical input with $V$ possible values (a word, a user ID, a product) is encoded as a one-hot row vector $e_v \in \mathbb{R}^V$, and a `Dense` layer without bias computes $e_v E = E_{v,:}$: the $v$-th row of $E \in \mathbb{R}^{V \times d}$. So an embedding layer is a linear layer whose input is one-hot, implemented as a table lookup because multiplying by a one-hot vector is wasteful. Its gradient follows from Section 5: $\partial \mathcal{L} / \partial E = X_{\text{one-hot}}^\top G$, which is non-zero only in the rows of the tokens present in the batch — sparse updates, which is why embedding tables of $50{,}257 \times 768 = 38.6$ million parameters (GPT-2's, 31 percent of the model) train efficiently.

**Worked example.** Convolve $x = (1, 2, 3, 4)$ with $w = (1, -1)$ and suppose the upstream gradient is $G = (1, 1, 1)$ (loss $= \sum_i y_i$). Forward: $y = Tx = (1 - 2,\ 2 - 3,\ 3 - 4) = (-1, -1, -1)$, which `np.convolve(x, w[::-1], mode="valid")` reproduces (NumPy's convolution flips the kernel, so we flip it back to get cross-correlation). Backward through the Toeplitz form: $\partial \mathcal{L} / \partial x = T^\top G = (1,\ -1 + 1,\ -1 + 1,\ -1) = (1, 0, 0, -1)$ — the full convolution of $G$ with the flipped kernel — and the shared-weight gradient collects one term per output position, $\partial \mathcal{L} / \partial w_0 = \sum_i G_i x_i = 1 + 2 + 3 = 6$, $\partial \mathcal{L} / \partial w_1 = \sum_i G_i x_{i+1} = 2 + 3 + 4 = 9$. Two parameters, where the unconstrained $3 \times 4$ dense layer would have twelve. For the recurrence: with $W_h$ having largest singular value $1.1$ and tanh units, the gradient 50 steps back is at most $1.1^{50} \approx 117$ times the output gradient before the $\phi' < 1$ factors, and with singular value $0.9$ at most $0.9^{50} \approx 0.005$ — the same two-sided cliff as Section 7, with the single matrix making it a clean geometric series.

**Real-life example (NLP and computer vision).** A product-search system at an online retailer embeds each of 2 million product IDs into 64 dimensions (a $2 \times 10^6 \times 64$ table, 128 million parameters, updated sparsely — only the few thousand products in each mini-batch receive gradient), runs a 1-D convolution over the characters of the query (kernel width 5, 256 channels: $5 \cdot 256 \cdot 256 = 328{,}000$ parameters regardless of query length, because the weights are shared along the text), and scores products against the query with a dot product. Replacing the convolution with a small attention layer let the model relate the first and last words of a long query in one step rather than through ten stacked convolutions — the receptive-field arithmetic of this section turned into a product decision.

**In ML/AI this is …** `nn.Conv2d` (Toeplitz-structured `Dense` with `kernel_size`, `stride`, `padding`), `nn.MaxPool2d`, `nn.LSTM`/`nn.GRU`, `nn.MultiheadAttention`, and `nn.Embedding` — five modules, one backward rule.

## Common pitfalls and misconceptions

1. **"More layers always means a more powerful model."** → Without a nonlinearity between them, $L$ affine layers compose to one affine map (Section 2), and with saturating nonlinearities the extra layers may be untrainable because the gradient shrinks by a factor per layer (Section 7). → Depth helps only with a non-polynomial activation and with gradients kept alive: ReLU-family units, He initialisation, normalisation, residual connections.
2. **Initialising all weights to zero (or any constant).** → Every unit in a layer computes the same output and receives the same gradient, so the layer never becomes more expressive than a single unit. → Draw weights randomly with the variance of Section 6; zero is fine for biases only.
3. **Copying an initialisation scale from a different architecture.** → $\mathcal{N}(0, 0.01^2)$ is harmless for a 2-layer, 50-wide network and fatal for a 10-layer, 512-wide one: the activation variance is multiplied by $n_{\text{in}} \sigma_w^2$ per layer. → Compute $n_{\text{in}} \sigma_w^2$; use $2/n_{\text{in}}$ for ReLU and $1/n_{\text{in}}$ (or $2/(n_{\text{in}} + n_{\text{out}})$) for tanh.
4. **Computing softmax then `log`, or `exp` without subtracting the maximum.** → Overflow at logits above 88 (float32) and `log(0)` for confident wrong predictions produce `inf` and `NaN`. → Use the log-sum-exp form, compute `log_softmax` directly, and get the probabilities as `exp(log_softmax)`.
5. **Using squared error for classification "because it's simpler".** → The gradient $2(\sigma - y)\sigma(1 - \sigma)$ vanishes exactly when the model is confidently wrong — 75 times smaller than cross-entropy's at $z = -5$ — and the minimiser is not a calibrated probability. → Pair sigmoid with binary cross-entropy and softmax with cross-entropy; both give gradient $p - y$.
6. **Forgetting `model.eval()` (or the inverse-dropout scaling).** → Batch norm with batch statistics on a single example gives $\hat z = 0$ for everything; dropout left on at test time injects noise and, without the $1/q$ scaling, changes the input scale of every layer. → Switch off dropout and use running statistics at inference; use inverted dropout so no test-time rescaling is needed.
7. **Judging a model by its training loss.** → A 10,000-parameter network memorises 1,400 examples; training loss 0.003 says nothing about the 360 you have not seen. → Hold out a validation set, plot both curves, stop at the validation minimum, and report test performance with a confidence interval (Module 10).
8. **Treating a "wrong" gradient check at a ReLU kink as a bug, or checking in float32.** → Central differences straddling $z = 0$ give $\approx 0.5$ against an analytic 0 or 1, and float32 limits agreement to about $10^{-3}$ regardless of correctness. → Check in float64 with $h \approx 10^{-5}$, on several random batches, and interpret isolated kink errors as Module 06 describes.

## Real-world case studies

### Case study 1: the digit classifier, by the numbers

The `load_digits` dataset has 1,797 greyscale images of $8 \times 8$ pixels with values 0–16, ten classes, roughly 180 per class. The notebooks standardise each pixel using training-set statistics (three corner pixels are constant zero and get a standard deviation of 1 to avoid division by zero), split 1,437/360, and train a 64 → 128 → 10 ReLU network: $65 \cdot 128 + 129 \cdot 10 = 9{,}610$ parameters, He-initialised, softmax cross-entropy, Adam at $\eta = 3 \times 10^{-3}$, mini-batches of 64, 30 epochs = 690 steps. The initial loss is $2.30 \approx \ln 10$, as it must be for an uninformed 10-way classifier — a check worth doing every time. The output layer is given a small initial scale (0.01) so that the initial logits are near zero and the initial loss is 2.33. After one epoch (23 steps) test accuracy is 87 percent; after five, 97.8 percent; after thirty, 98.3 percent, with the training loss at 0.002 and 100 percent training accuracy from epoch 10 on. The wall-clock time is under a second of pure NumPy per run.

Now read those numbers critically. The test set has 360 examples, so 98.3 percent accuracy is $354/360$ and its standard error is $\sqrt{0.983 \cdot 0.017 / 360} = 0.0068$; a 95 percent interval is roughly $[97.0, 99.6]$ percent, and five different random seeds land between 97.8 and 98.9 percent — within the noise. Claiming that one architecture "beats" another by half a percent on this test set is not supportable, which is what Module 12 warned about. The test loss tells a second story: it falls to about 0.065 by epoch 20 and then stays flat or creeps up by a thousandth or two while accuracy stays fixed — the model is becoming over-confident on the examples it gets wrong (its logit gaps keep growing because nothing stops them; label smoothing or weight decay would), without changing which examples those are. The six misclassified test digits, plotted in the notebook, are mostly genuinely ambiguous 8×8 blobs — a 3 that looks like a 5, a 9 whose loop is closed like an 8. A convolutional network might gain a fraction of a percentage point here; the rest is in the data.

### Case study 2: GPT-2's MLP blocks are this module

GPT-2 small has 12 transformer layers with hidden size $d = 768$. Each layer contains an attention sub-block with four $768 \times 768$ projections plus biases ($4 \cdot 768^2 + 4 \cdot 768 = 2{,}362{,}368$ parameters) and a feed-forward sub-block that is a one-hidden-layer MLP, $768 \to 3{,}072 \to 768$ with a GELU in between: $768 \cdot 3072 + 3072 + 3072 \cdot 768 + 768 = 4{,}722{,}432$ parameters. Twelve layers of MLP blocks hold $56.7$ million parameters, 45.5 percent of the model's 124.4 million; the token embedding table ($50{,}257 \times 768 = 38.6$ million, 31 percent) and the attention projections (28.3 million, 23 percent) hold most of the rest.

Everything in this module applies to those blocks directly. The forward pass is `X @ W1 + b1`, GELU, `@ W2 + b2`, on $X$ of shape (tokens × 768), with the residual connection $X + \mathrm{MLP}(\mathrm{LN}(X))$ of Section 7 wrapped around it and the layer norm of Section 8 in front. The backward pass is Section 5's recurrence with the GELU derivative $\Phi(z) + z \varphi(z)$ as the mask. The initialisation is $\mathcal{N}(0, 0.02^2)$ for the first matrix ($n_{\text{in}} \sigma_w^2 = 768 \cdot 0.0004 = 0.31$, deliberately below 1 because the layer norm upstream fixes the input scale and the residual stream accumulates $2 \times 12 = 24$ branch outputs), with the second matrix additionally scaled by $1/\sqrt{24}$ so that the sum over branches keeps unit variance. Training uses AdamW with decoupled weight decay 0.1 (Section 9), cosine decay with warm-up, gradient clipping at norm 1.0 (Section 7), and dropout 0.1 on the residual branches (Section 10). Per token, one MLP block costs about $2 \times 4.7$ million $= 9.4$ million FLOPs forward and twice that backward, and its stored GELU pre-activation is 3,072 floats per token — at batch 8 × 1,024 tokens, 100 MB per layer, 1.2 GB for the model, the dominant activation memory after attention. When you read a transformer paper's "FFN" or "MLP" ablations — SwiGLU variants, 4× vs 8× expansion, removing biases — you are reading about Sections 1–6 of this module.

### Case study 3: a clinical risk score that collapsed in deployment

A hospital team trains a 40 → 128 → 128 → 1 network to predict 30-day readmission from electronic health records, with batch normalisation after each hidden layer (training-set validation AUC 0.81). The model is deployed as a service that scores patients one at a time at discharge. Within a week the scores are nonsensical: nearly every patient receives a probability near 0.5. The post-mortem finds the two Section 8 errors in one system. First, the inference code called the network in training mode, so batch norm computed statistics over a batch of one — every normalised pre-activation was exactly zero and the network output was the constant $\sigma(\beta$-terms$)$ regardless of the patient. Second, when that was fixed by switching to running statistics, scores were still shifted: the running averages had been accumulated with momentum 0.1 over the *last* training batches only, which happened to be drawn from a single ward because the training loader had not been reshuffled per epoch (Section 9), so the running mean of "age" was 71 rather than the population's 58.

The repaired pipeline shuffles every epoch, re-estimates the running statistics with a full pass over the training set in eval mode after training (a common final step), and includes a unit test that scores one patient alone and the same patient inside a batch of 32 and asserts the outputs match to $10^{-6}$. Validation AUC in deployment: 0.80. Neither bug touched the mathematics of the model; both were failures to respect the difference between batch statistics and population statistics that the formulas of Section 8 make explicit.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Affine layer $XW + b$ | `nn.Linear`, every projection in attention, the FFN of each transformer block | The single primitive that everything else constrains or composes |
| XOR / hidden layer | Tabular MLPs, the FFN "key–value memory" view of transformers | Hidden units compute conjunctions the output layer can linearly combine |
| Universal approximation | Justifies MLP heads on any backbone; does not justify any specific width | Separates approximation from optimisation and generalisation |
| Activation derivatives | `ReluBackward0`, `GeluBackward0`; dying-ReLU diagnostics | The mask that decides how much gradient each unit passes |
| Batch-first shapes, broadcasting | `(batch, features)`, `(batch, seq, d_model)`; the "6N FLOPs" training-cost rule | Shape discipline is most of implementation; FLOP counts drive budgets |
| Softmax-CE and $P - Y$ | `nn.CrossEntropyLoss`, LM heads over the vocabulary, `label_smoothing=` | Stable, one-pass loss and gradient; the right likelihood for categories |
| Backprop recurrence $A^\top G$, $G W^\top$, column sums | Autograd backward kernels, gradient checkpointing, activation memory | Explains training-memory scaling and how custom layers get gradients |
| Variance propagation, Xavier/He | `kaiming_normal_`, `xavier_uniform_`, GPT-2's $1/\sqrt{2L}$ residual scaling, μP | Trainability at depth is decided before the first step |
| Vanishing/exploding gradients | Residual streams, LSTM gates, `clip_grad_norm_`, grad-norm dashboards | Depth is a product of Jacobians; every modern architecture controls it |
| Batch norm / layer norm | `BatchNorm2d` in CNNs, `LayerNorm`/`RMSNorm` in transformers, `model.eval()` | Scale invariance and smoother optimisation; the train/inference distinction |
| Mini-batch loop, schedules, weight decay | `DataLoader`, `AdamW`, `CosineAnnealingLR`, warm-up | Noise, step size and prior in one loop |
| Dropout (inverted) | `nn.Dropout`, attention dropout, stochastic depth | Multiplicative noise ≈ ensemble of thinned networks |
| Convolution = Toeplitz `Dense` | `nn.Conv2d`, receptive fields, weight-sharing gradients | 10⁵ times fewer parameters via sparsity and sharing |
| Recurrence = tied-weight depth | `nn.LSTM`, BPTT, truncated BPTT | Same-matrix products explode or vanish geometrically |
| Embedding = one-hot × matrix | `nn.Embedding`, sparse gradients, 31 percent of GPT-2 | Lookup is a linear layer in disguise |

## Cheat sheet

| Item | Formula or fact |
|---|---|
| Layer (batch-first) | $Z = XW + \mathbf{1} b^\top$, $X: n \times n_{\text{in}}$, $W: n_{\text{in}} \times n_{\text{out}}$, $b: n_{\text{out}}$; $A = \phi(Z)$ |
| Parameter count | $\sum_l (n_l + 1)\, n_{l+1}$; e.g. 64–128–10 → 9,610 |
| FLOPs | forward $\approx 2 \cdot \text{weights} \cdot \text{batch}$; training step $\approx 6 \cdot \text{weights} \cdot \text{batch}$ |
| No nonlinearity | $(XW_1 + b_1)W_2 + b_2 = X(W_1 W_2) + (b_1 W_2 + b_2)$: one affine map |
| Sigmoid, tanh | $\sigma' = \sigma(1 - \sigma) \le 1/4$; $\tanh' = 1 - \tanh^2 \le 1$; $\tanh z = 2\sigma(2z) - 1$ |
| ReLU family | $\mathrm{ReLU}' = \mathbb{1}[z > 0]$; leaky: $\alpha$ for $z < 0$; softplus$' = \sigma$; SiLU$' = \sigma(1 + z(1 - \sigma))$; GELU$' = \Phi(z) + z\varphi(z)$ |
| Softmax-CE | $\mathcal{L} = -\frac1n \sum_i \ln P_{i y_i}$, $\ \partial \mathcal{L}/\partial Z = (P - Y)/n$; initial loss $\ln k$ |
| Log-sum-exp | $\ln \sum_j e^{z_j} = m + \ln \sum_j e^{z_j - m}$, $m = \max_j z_j$; $\ln P = Z - \mathrm{LSE}(Z)$ |
| Sigmoid + MSE gradient | $2(\sigma - y)\,\sigma(1 - \sigma)$ — vanishes when confidently wrong; use CE |
| Label smoothing | $Y^{\text{ls}} = (1 - \varepsilon) Y + \varepsilon / k$ |
| Backward (layer $l$) | $\partial W_l = A_{l-1}^\top G_l$, $\ \partial b_l = \sum_{\text{rows}} G_l$, $\ \partial A_{l-1} = G_l W_l^\top$, $\ G_{l-1} = \partial A_{l-1} \odot \phi'(Z_{l-1})$ |
| Variance propagation | $\operatorname{Var}(z) = n_{\text{in}} \sigma_w^2 \sigma_x^2$; ReLU halves the second moment |
| Initialisation | LeCun $\sigma_w^2 = 1/n_{\text{in}}$; Xavier $2/(n_{\text{in}} + n_{\text{out}})$; He $2/n_{\text{in}}$; zeros ⇒ symmetry, never |
| Gradient through depth | $\lVert G_l \rVert \lesssim \lVert G_L \rVert \prod_m \lVert W_m \rVert_2 \max\lvert\phi'\rvert$; sigmoid: $\le 4^{-L}$ |
| Residual block | $A_l = A_{l-1} + F(A_{l-1})$ ⇒ Jacobian $I + J_l$; product contains the identity path |
| Gradient clipping | $g \leftarrow c\, g / \lVert g \rVert$ if $\lVert g \rVert > c$ |
| Batch norm | $\hat Z = (Z - \mu) / \sqrt{\sigma^2 + \epsilon}$ per column, $Y = \gamma \hat Z + \beta$; running stats at inference |
| BN backward | $\partial Z = \frac{\gamma}{n\sqrt{\sigma^2 + \epsilon}}\big(nG - \sum G - \hat Z \sum G \hat Z\big)$ (column-wise sums) |
| Layer norm | same formulas per *row*; no batch dependence; transformers |
| Inverted dropout | $\tilde A = M \odot A / q$, $M \sim \mathrm{Bern}(q)$, $E[\tilde A] = A$; off at test time |
| Weight decay | $+\lambda \theta$ in the gradient (SGD) or decoupled $-\eta\lambda\theta$ (AdamW); Gaussian prior with variance $1/(N\lambda)$ for a mean loss over $N$ examples ($1/\lambda$ for a summed NLL) |
| Cosine schedule | $\eta_t = \tfrac{\eta_0}{2}(1 + \cos(\pi t / T))$; warm-up: linear ramp over the first steps |
| Convolution | $y = Tx$, Toeplitz $T$; params $m$ not $n(n-m+1)$; $\partial x = T^\top G$ = flipped-kernel convolution |
| Recurrence | $h_t = \phi(h_{t-1} W_h + x_t W_x + b)$; gradient $\sim \sigma_{\max}(W_h)^{T - t}$ |
| Embedding | $e_v E = E_{v,:}$; gradient non-zero only in rows present in the batch |

## Self-assessment

1. **Why can no single affine unit compute XOR, and what is the smallest change that fixes it?**
   <details><summary>Answer</summary>Adding the two positive-point inequalities gives $w_1 + w_2 + 2b > 0$, adding the two negative-point inequalities gives $w_1 + w_2 + 2b \le 0$ — a contradiction, so no hyperplane separates the classes. One hidden layer with two ReLU units (computing $x_1 + x_2$ and $x_1 + x_2 - 1$) maps the four points to new coordinates in which a line does separate them.</details>
2. **A colleague stacks 20 `Linear` layers with no activation functions and reports that the model is "very deep". What can it represent?**
   <details><summary>Answer</summary>Only affine maps: $X W_1 W_2 \cdots W_{20} + (\text{bias terms})$ is a single matrix and vector. It has the expressive power of linear regression, with worse conditioning and more parameters. Depth counts only with a nonlinearity between layers.</details>
3. **What are the shapes of $W_1$, $b_1$, $Z_1$ and $\partial \mathcal{L} / \partial W_1$ for a batch of 32 examples with 20 features and 50 hidden units, and which product gives $\partial \mathcal{L} / \partial W_1$?**
   <details><summary>Answer</summary>$W_1: 20 \times 50$, $b_1: 50$, $Z_1: 32 \times 50$, $\partial \mathcal{L} / \partial W_1: 20 \times 50 = X^\top G_1$ with $X: 32 \times 20$ and $G_1: 32 \times 50$. The bias gradient sums $G_1$ over its 32 rows.</details>
4. **Derive $\partial \mathcal{L} / \partial z = p - y$ for softmax cross-entropy in three lines.**
   <details><summary>Answer</summary>$\ln p_j = z_j - \ln \sum_c e^{z_c}$, so $\partial \ln p_j / \partial z_m = \mathbb{1}[j = m] - p_m$. The loss is $-\sum_j y_j \ln p_j$ with $\sum_j y_j = 1$, so $\partial \mathcal{L} / \partial z_m = -\sum_j y_j(\mathbb{1}[j=m] - p_m) = -y_m + p_m$.</details>
5. **Why does a sigmoid output with squared-error loss learn slowly on confidently wrong examples, and by how much at $z = -5$, $y = 1$?**
   <details><summary>Answer</summary>Its gradient $2(\sigma - y)\sigma(1 - \sigma)$ contains the sigmoid's slope, which is $0.0066$ at $z = -5$; the gradient is $-0.013$ versus $-0.993$ for cross-entropy, 75 times smaller. The $\ln$ in cross-entropy cancels the $\exp$ in the sigmoid and removes the slope factor.</details>
6. **Where does the factor 2 in He initialisation come from, and what would it be for a leaky ReLU with slope $\alpha$?**
   <details><summary>Answer</summary>A ReLU zeroes the negative half of a symmetric input, so $E[\mathrm{ReLU}(z)^2] = \tfrac12 E[z^2]$; preserving the second moment through $n_{\text{in}}$ inputs needs $n_{\text{in}} \sigma_w^2 / 2 = 1$. For leaky ReLU the negative half contributes $\alpha^2 \cdot \tfrac12 E[z^2]$, so the factor is $2 / (1 + \alpha^2)$.</details>
7. **Ten sigmoid layers with weights of spectral norm 1: give an upper bound on the gradient reaching the first layer, and name three changes that each remove the problem.**
   <details><summary>Answer</summary>Each layer multiplies the gradient by at most $\max \sigma' = 1/4$, so the bound is $4^{-10} \approx 10^{-6}$ relative to the output gradient. Remedies: ReLU-family activations (mask of 0/1, no $1/4$); residual connections (Jacobian $I + J$, identity path); batch or layer normalisation (keeps pre-activations $O(1)$ so Jacobians stay $O(1)$). Careful initialisation helps ReLU networks but cannot rescue sigmoid chains, since saturation makes the product smaller for larger weights too.</details>
8. **A batch-norm network gives perfect validation accuracy but outputs the same score for every patient in production. What happened?**
   <details><summary>Answer</summary>It is running in training mode and scoring one example at a time, so the batch statistics are $\mu = z$ and $\sigma^2 = 0$: every normalised pre-activation is 0 and the output is a constant determined by the $\beta$ parameters. Inference must use the running mean and variance (`model.eval()`).</details>
9. **Why is the kept activation divided by $q$ in inverted dropout, and what would go wrong without it?**
   <details><summary>Answer</summary>$E[M \odot A / q] = qA/q = A$, so the next layer's expected input is unchanged between training (mask on) and inference (mask off). Without the division the training-time input would be $q$ times smaller than at test time, and the network would see inputs of the wrong scale when deployed — or need a compensating rescaling at test time, as the original dropout did.</details>
10. **The training loss of your digit classifier reaches 0.002 and the test loss creeps up slightly after epoch 20 while test accuracy stays at 98 percent. Is the model overfitting, and what would you do?**
    <details><summary>Answer</summary>Mildly: the model is becoming over-confident (logit gaps grow without bound) rather than changing its decisions, which is why accuracy is flat while the log-loss rises. Early stopping at the test-loss minimum, label smoothing or weight decay all address it; comparing architectures by half-percent differences on 360 test examples does not, since the standard error of the accuracy is about 0.7 percentage points.</details>

## Further reading

- I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*, MIT Press, 2016 — Chapter 6 (deep feedforward networks), Chapter 7 (regularisation) and Chapter 8 (optimisation); free online.
- M. Nielsen, *Neural Networks and Deep Learning*, 2015 — a free online book that builds a NumPy digit classifier and derives backpropagation and cross-entropy with great care.
- X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks", AISTATS 2010 — Xavier initialisation and the variance-propagation argument.
- K. He, X. Zhang, S. Ren and J. Sun, "Delving Deep into Rectifiers", ICCV 2015 — He initialisation and the factor 2; and "Deep Residual Learning for Image Recognition", CVPR 2016 — residual connections.
- S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015; and S. Santurkar et al., "How Does Batch Normalization Help Optimization?", NeurIPS 2018 — the method and the corrected explanation.
- N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", *JMLR* 15, 2014.
- G. Cybenko, "Approximation by superpositions of a sigmoidal function", *Mathematics of Control, Signals and Systems* 2, 1989; and M. Leshno et al., "Multilayer feedforward networks with a nonpolynomial activation function can approximate any function", *Neural Networks* 6, 1993.
- A. Karpathy, "A Recipe for Training Neural Networks", blog post, 2019 — the checklist of Section 10, with the war stories behind each item.
- Stanford CS231n course notes, "Neural Networks Part 1–3" — activations, initialisation, batch norm, dropout and the training loop, with code.
- D. Hendrycks and K. Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415, 2016 — the transformer activation and its derivative.

## What's in the notebooks

**`lesson.ipynb`** builds the module in code, in the order of the sections above: a logistic-regression model failing on XOR and a two-unit hidden layer solving it, with both decision surfaces drawn; an activation gallery with analytic derivatives checked against central differences; the Section 3 forward pass with shapes printed at every step; the stable softmax cross-entropy with its $P - Y$ gradient verified numerically, and the sigmoid-plus-MSE gradient collapse plotted; `Dense`, `ReLU` and `MLP` classes with `forward` and `backward`, gradient-checked to $10^{-10}$ on the Section 5 example and on a random network; `SGD` and `Adam` optimisers and a complete mini-batch training loop on the 1,797 `load_digits` images (standardise, 80/20 split, about 98 percent test accuracy in under a second) with loss and accuracy curves, a confusion matrix and the misclassified digits; activation standard deviations across 10 layers for weight scales 0.01, 1.0, Xavier and He; gradient magnitudes per layer for sigmoid versus ReLU chains and the residual fix; a batch-norm layer in training and inference mode with its effect on activation statistics; inverted dropout and its effect on train/test loss when only 100 training digits are available; and a 1-D convolution written as a Toeplitz matrix product and checked against `np.convolve`.

**`exercises.ipynb`** (with `solutions.ipynb`) tests the same ideas with instant checks: activation functions and their derivatives against finite differences; parameter and FLOP counting with hidden answers; a stable log-softmax and cross-entropy compared with `scipy.special`; the $P - Y$ gradient; a `Dense` layer's backward pass verified by a gradient checker; He initialisation and the variance-propagation rule measured on a 10-layer network; batch-norm forward and backward against the formulas; inverted dropout's expectation; a conceptual question on depth without nonlinearity and the He factor 2; convolution as a matrix; and a mini-project that trains your NumPy MLP on the digits with and without dropout and weight decay, reports test accuracy with a bootstrap confidence interval, and visualises the first-layer weights as 8 × 8 images.
