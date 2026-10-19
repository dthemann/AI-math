# Module 17 — The Mathematics of Transformers

> After this module you can write a complete transformer block in NumPy from memory, explain every constant in it (the $\sqrt{d_k}$, the $10000$ in the positional encoding, the $12d^2$ parameters per block), budget the memory and compute of a language model from its configuration, and read papers on attention, RoPE, LoRA and KV caches as applied linear algebra rather than folklore.

## Why this matters for ML and AI

Every large language model in production today — GPT-4, Claude, Gemini, Llama, Mistral — is a stack of nearly identical blocks, and each block is built from four operations you have already met: a matrix multiplication (Module 02), a softmax (Module 14), a normalisation (Module 16) and an addition. The same architecture, with minor changes, powers vision models (ViT), protein-structure prediction (AlphaFold's Evoformer), speech recognition (Whisper) and code assistants. There is no separate mathematics for each of these; there is one set of equations whose consequences you can derive by hand.

What makes transformers worth a module of their own is not that the equations are hard but that their *consequences* are quantitative and surprisingly far-reaching. Why is attention divided by $\sqrt{d_k}$? Because a dot product of $d_k$ independent unit-variance numbers has variance $d_k$, and a softmax fed inputs with standard deviation $8$ produces probabilities of $0.9997$ and gradients of order $10^{-4}$ — the model stops learning. Why does doubling the context length of a chat model quadruple part of its cost? Because the attention score matrix is $n\times n$. Why does a 7-billion-parameter model that fits in 14 GB of GPU memory suddenly need another 4 GB per conversation? Because the key–value cache stores $2 \cdot \text{layers} \cdot d_{\text{model}}$ numbers per token, and $2 \cdot 32 \cdot 4096 \cdot 8192 \cdot 2$ bytes is exactly 4 GiB. Why can you fine-tune that model on one consumer GPU by training $0.1\,\%$ of its weights? Because weight *updates* are approximately low-rank, and Module 03 told you exactly how cheap a rank-8 matrix is.

This module works through the architecture in the order the data flows: tokens become vectors (embeddings), vectors talk to each other (attention), the result is stabilised (layer norm, residuals) and transformed (feed-forward), and the final vector is turned back into a distribution over tokens (unembedding, cross-entropy, sampling). Along the way you will prove that attention is permutation-equivariant (so positions must be injected), that sinusoidal and rotary encodings turn relative position into rotations, that the softmax Jacobian is $\operatorname{diag}(p) - pp^\top$, and that a transformer block has $12d^2$ weights. Everything is implemented in NumPy in the notebooks — no framework hides the maths.

## Learning objectives

By the end of this module you will be able to:

- Implement token embedding as a one-hot matrix product, tie the unembedding to it, and compute next-token logits and probabilities from a residual-stream vector.
- Implement scaled dot-product attention from scratch, work a three-token example by hand, and explain attention as a data-dependent weighted average of values.
- Derive why the scores are divided by $\sqrt{d_k}$ from the variance of a sum of products, and quantify how softmax saturation kills gradients.
- Write down the softmax Jacobian $\operatorname{diag}(p) - pp^\top$, verify it numerically, and propagate gradients through attention to queries, keys and values.
- Count the parameters ($\approx 12d^2$ per block) and the FLOPs ($24nd^2 + 4n^2d$ per layer) of a transformer, and compute the context length at which attention cost overtakes the weight cost.
- Build a causal mask, explain why it makes next-token training a single parallel pass, and compute the cross-entropy loss and perplexity.
- Prove the permutation equivariance of attention, implement sinusoidal and rotary positional encodings, and show that both turn relative offsets into rotations.
- Implement LayerNorm and a full pre-norm transformer block, and explain why layer norm (not batch norm) is used in sequence models.
- Implement greedy, temperature, top-$k$, top-$p$ and beam decoding on a logit vector, and compute the KV-cache memory of a model from its configuration.
- Explain LoRA as a low-rank update $W_0 + BA$, compute its parameter savings, merge it into the base weights, and describe quantisation error and mixture-of-experts routing quantitatively.

## Prerequisites

- [Module 01 — Vectors and geometry](../../part-1-linear-algebra/01-vectors-and-geometry/README.md): dot products as similarity, norms, cosine similarity, one-hot vectors, embeddings.
- [Module 02 — Matrices and linear maps](../../part-1-linear-algebra/02-matrices-and-linear-maps/README.md): matrix multiplication as composition, rotation matrices, orthogonal matrices, rank.
- [Module 03 — Eigenvalues, SVD and PCA](../../part-1-linear-algebra/03-eigenvalues-svd-and-pca/README.md): truncated SVD, Eckart–Young, low-rank approximation.
- [Module 04 — Differential calculus](../../part-2-calculus-and-optimization/04-differential-calculus/README.md): Jacobians, the chain rule, finite-difference gradient checks.
- [Module 14 — Information theory](../../part-5-learning-theory/14-information-theory/README.md): softmax, cross-entropy, entropy, perplexity, temperature.
- [Module 16 — Neural networks from scratch](../16-neural-networks-from-scratch/README.md): the softmax–cross-entropy gradient $p - y$, batch normalisation, residual connections, variance propagation through layers.

Recall in particular that a rotation matrix $R(\theta)$ satisfies $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$, that the variance of a sum of independent terms is the sum of their variances, and that for a softmax followed by cross-entropy the gradient with respect to the logits is simply $p - y$.

## 1. Tokens, embeddings and the residual stream

**Intuition.** A language model never sees letters or words; it sees integers. A *tokeniser* chops text into pieces from a fixed vocabulary — common words are single tokens, rare words are split into several — and each piece gets an index. The model's first job is to turn that integer into a vector it can do arithmetic with. The simplest such map is a lookup table: row $t$ of a matrix $E$ is "the vector for token $t$". That vector is the starting content of a $d$-dimensional *residual stream*, one per position, which every later layer reads from and adds to. At the end, the same table is read backwards: the stream's final vector is compared with every row of $E$ by a dot product, and the best-matching rows get the highest next-token scores.

**Definition.** The most common tokeniser is *byte-pair encoding* (BPE): start with single bytes as tokens, repeatedly merge the most frequent adjacent pair in a training corpus into a new token, and stop after a fixed number of merges. GPT-2's vocabulary has $50{,}257$ tokens: $256$ byte tokens, $50{,}000$ merges and one end-of-text token. Let $V$ be the vocabulary size and $d$ (often written $d_{\text{model}}$) the width of the model. The **embedding matrix** is $E \in \mathbb{R}^{V\times d}$. For a token index $t$ with one-hot vector $e_t \in \{0,1\}^V$,

$$x = e_t^\top E = E_{t,:} \in \mathbb{R}^{d},$$

so "lookup" is a matrix product with a one-hot vector — which is exactly how its gradient works: only row $t$ of $E$ receives a gradient. For a sequence of $n$ tokens the input is $X \in \mathbb{R}^{n\times d}$, one row per position, plus positional information (Section 7). Each layer $\ell$ computes an update and *adds* it: $X^{(\ell+1)} = X^{(\ell)} + f_\ell(X^{(\ell)})$. The rows of $X$ are the **residual stream**; the layers never overwrite it, they only write into it. At the top, with **tied unembedding**, the logits for the next token at a position with final vector $h \in \mathbb{R}^d$ are

$$z = E h \in \mathbb{R}^{V}, \qquad p = \operatorname{softmax}(z), \qquad p_v = \frac{e^{z_v}}{\sum_{u=1}^{V} e^{z_u}} .$$

Untied models use a separate matrix $W_U \in \mathbb{R}^{d\times V}$ instead; tying saves $Vd$ parameters and makes the geometric statement explicit: the logit for token $v$ is the dot product of the stream with $v$'s own embedding.

**Worked example.** Vocabulary $\{\text{the}, \text{cat}, \text{sat}, \text{on}, \text{mat}\}$, $d = 3$,

$$E = \begin{pmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \\ -1 & 0 & 0.5 \\ 0 & -1 & 0.5 \\ 0.5 & 0.5 & 0 \end{pmatrix}.$$

Token "sat" has index $2$, so $e_2 = (0,0,1,0,0)$ and $e_2^\top E = (-1, 0, 0.5)$ — the third row. Suppose after all the layers the residual stream at the last position holds $h = (0.8, 0.2, 0.5)$. Logits: $z = Eh = (0.8 + 0.25,\ 0.2 + 0.25,\ -0.8 + 0.25,\ -0.2 + 0.25,\ 0.4 + 0.1) = (1.05, 0.45, -0.55, 0.05, 0.5)$. Exponentials $(2.858, 1.568, 0.577, 1.051, 1.649)$ sum to $7.703$, so

$$p = (0.371,\ 0.204,\ 0.075,\ 0.136,\ 0.214).$$

The model predicts "the" with probability $0.37$ because $h$ points most nearly along the first row of $E$. For GPT-2 small the same objects are $E \in \mathbb{R}^{50257\times768}$ — $38.6$ million parameters, about $31\,\%$ of the whole model — and $h \in \mathbb{R}^{768}$.

**Real-life example (e-commerce).** Session-based recommenders treat a shopper's click history as a sentence and products as tokens: a catalogue of $200{,}000$ items is a "vocabulary", each item gets a learned embedding row, and the model predicts the next item exactly as a language model predicts the next word. The tied unembedding then means that the score of item $v$ for this shopper is the dot product of the session vector with $v$'s embedding — so items that are often bought in the same sessions end up with nearby embedding rows, and a nearest-neighbour search over $E$ is a cheap "similar items" feature. Rare items are the equivalent of rare words: with few sessions to learn from, their rows barely move from initialisation, which is why practitioners group them into coarser categories, the recommender's version of BPE.

**In ML/AI this is …** `nn.Embedding(vocab, d)` in PyTorch (a lookup, with the gradient of a one-hot matmul); `tie_word_embeddings=True` in Hugging Face configs; `model.lm_head.weight` being the same tensor as `model.wte.weight` in GPT-2; and the residual-stream picture that mechanistic-interpretability work uses to talk about "what a layer writes".

## 2. Attention as a soft lookup

**Intuition.** A dictionary lookup takes a query, finds the one key that matches exactly, and returns its value. Attention does the same thing softly: every position in the sequence publishes a *key* (what I contain) and a *value* (what I will hand over), each position asks a *query* (what I am looking for), and the answer is a weighted average of all the values, with weights given by how well the query matches each key. No key matches exactly; instead every key contributes in proportion to its match. The crucial difference from an ordinary neural-network layer is that the mixing weights are not parameters: they are computed from the data, so the same layer can route information from word 3 to word 7 in one sentence and from word 1 to word 7 in the next.

**Definition.** For an input $X \in \mathbb{R}^{n\times d}$ with learned projections $W_Q, W_K \in \mathbb{R}^{d\times d_k}$ and $W_V \in \mathbb{R}^{d\times d_v}$,

$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V,$$

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V .$$

The **score matrix** $S = QK^\top/\sqrt{d_k} \in \mathbb{R}^{n\times n}$ holds $S_{ij} = q_i^\top k_j / \sqrt{d_k}$, the similarity between the query at position $i$ and the key at position $j$. The softmax is applied **row-wise**, so each row of $P = \operatorname{softmax}(S)$ is a probability distribution over the $n$ positions (Module 14): $P_{ij} \ge 0$, $\sum_j P_{ij} = 1$. The output is $O = PV$, whose row $i$ is $o_i = \sum_j P_{ij} v_j$ — a convex combination of the value vectors, weighted by how much position $i$ attends to position $j$. Two facts follow immediately. If all scores in a row are equal, $P_{ij} = 1/n$ and $o_i$ is the plain mean of the values. And $O = PV$ is linear in $V$ for fixed $P$, but $P$ itself depends on $X$ — attention is a *data-dependent linear combination*, where an MLP's mixing matrix is fixed after training.

**Worked example.** Three tokens, $d_k = d_v = 2$. Take the projected matrices directly:

$$Q = K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \qquad V = \begin{pmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}.$$

Raw scores $QK^\top = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}$; dividing by $\sqrt{2} = 1.414$ gives rows $(0.707, 0, 0.707)$, $(0, 0.707, 0.707)$, $(0.707, 0.707, 1.414)$. Row 1 of the softmax: $e^{0.707} = 2.028$, $e^0 = 1$, $e^{0.707} = 2.028$, sum $5.056$, so $P_{1,:} = (0.401, 0.198, 0.401)$. Row 3: $(2.028, 2.028, 4.113)$, sum $8.169$, $P_{3,:} = (0.248, 0.248, 0.504)$. Outputs:

$$o_1 = 0.401\,(2,0) + 0.198\,(0,2) + 0.401\,(1,1) = (1.203,\ 0.797), \qquad o_3 = 0.248\,(2,0) + 0.248\,(0,2) + 0.504\,(1,1) = (1.0,\ 1.0).$$

Token 1's query matches its own key and token 3's key equally well, so it blends their values and mostly ignores token 2. Token 3's query $(1,1)$ matches everything, and its output is the exact centroid of the values. By symmetry $o_2 = (0.797, 1.203)$.

**Real-life example (healthcare).** A model that predicts 30-day hospital readmission reads a patient's record as a sequence of coded events — diagnoses, procedures, lab results, prescriptions — each embedded as a vector. At the final position the query is, in effect, "what in this history matters for readmission?"; the keys describe each past event; the values carry the event's clinical content. A heart-failure diagnosis three admissions ago and a recent potassium result can both receive high weights while routine events receive almost none, and the output vector is a readmission-relevant summary weighted exactly by those scores. Because the weights are data-dependent, the same model summarises a diabetic patient's record by attending to entirely different events — a fixed-weight layer could not do that.

**In ML/AI this is …** `torch.nn.functional.scaled_dot_product_attention(q, k, v)`, the `softmax(QK^T/√d_k)V` of the "Attention Is All You Need" paper (Vaswani et al., 2017), *cross*-attention when $K$ and $V$ come from a different sequence than $Q$ (an encoder's output, an image's patches, retrieved documents), and the attention maps people plot as heat-maps: those are the rows of $P$.

## 3. Why divide by $\sqrt{d_k}$

**Intuition.** Add up $d_k$ random numbers and the sum's typical size grows like $\sqrt{d_k}$, not $d_k$ — the random-walk law. Dot products are sums of $d_k$ products, so raw attention scores get larger as the head gets wider, for no reason that has anything to do with the data. A softmax of large numbers is a near-one-hot vector, and a near-one-hot softmax has almost no gradient. Dividing by $\sqrt{d_k}$ puts the scores back on a scale of order one, whatever the width.

**Definition / derivation.** Let the components $q_i$ and $k_i$ ($i = 1, \dots, d_k$) be independent with mean $0$ and variance $1$ — a reasonable description of projected activations at initialisation. Then

$$\mathbb{E}[q^\top k] = \sum_i \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0, \qquad \operatorname{Var}(q^\top k) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = d_k ,$$

using independence for the first equality (variances of independent terms add) and $\operatorname{Var}(q_ik_i) = \mathbb{E}[q_i^2k_i^2] - (\mathbb{E}[q_ik_i])^2 = 1\cdot1 - 0$ for the second. So the raw scores have standard deviation $\sqrt{d_k}$, and $q^\top k/\sqrt{d_k}$ has variance exactly $1$. Why that matters: softmax is translation-invariant but not scale-invariant. For two scores differing by $\Delta$, the larger receives probability $1/(1 + e^{-\Delta})$; with $\Delta = 1$ that is $0.731$, with $\Delta = 8$ it is $0.9997$. The gradient of the softmax (Section 4) is proportional to $p_i(1 - p_i)$ on the diagonal, which for $p_i = 0.9997$ is $3\times10^{-4}$: every gradient flowing back through that row is multiplied by a number that small, and the keys and queries stop learning (vanishing gradients, Module 04 and Module 16). With $d_k = 64$ and no scaling, score differences of order $8$ are the typical case, not the exception.

**Worked example.** $d_k = 64$, components standard normal. Raw scores have standard deviation $\sqrt{64} = 8$; scaled scores have standard deviation $1$. Take a row of three raw scores $(8, 0, 0)$ (one standard deviation above the others): softmax gives $(0.9993, 0.0003, 0.0003)$ — the head attends to one token almost exclusively, and $p_1(1-p_1) = 0.0007$. After scaling, the same row is $(1, 0, 0)$, softmax $(0.576, 0.212, 0.212)$, and $p_1(1 - p_1) = 0.244$: three hundred times more gradient. The lesson notebook simulates this for $d_k \in \{4, 16, 64, 256\}$: the variance of raw scores tracks $d_k$ to within sampling error, the scaled variance stays at $1.0$, and the mean entropy of the attention rows collapses towards $0$ without scaling but stays near $\ln n$ with it.

**Real-life example (physics and engineering).** A sensor that adds up $d$ independent noisy readings — a radio receiver integrating over $d$ samples, a telescope summing $d$ photon counts — sees its noise amplitude grow as $\sqrt{d}$ while a steady signal grows as $d$, which is why averaging improves the signal-to-noise ratio by $\sqrt{d}$. Engineers normalise accordingly: an averaging filter divides by $d$ to keep the signal level fixed, and a matched filter normalises by $\sqrt{d}$ to keep the *noise* level fixed. Attention is in the second situation at initialisation — there is no signal yet, only noise — so the $\sqrt{d_k}$ normalisation is the one that keeps the softmax in its responsive range.

**In ML/AI this is …** the `/ math.sqrt(d_k)` (or `scale=` argument) in every attention implementation; the reason "attention logit growth" is monitored in large-model training and why some models add a normalisation of $Q$ and $K$ before the dot product (QK-norm); and the same variance-propagation logic as Xavier/He initialisation in Module 16 — keep every signal at unit variance so no nonlinearity saturates.

## 4. The softmax Jacobian and attention gradients

**Intuition.** Push one logit up and its probability rises — but the other probabilities must fall to keep the total at one. So the derivative of a softmax is not a diagonal matrix: every output depends on every input. Two consequences matter for training. First, the gradient that reaches the logits is the upstream gradient *re-centred* by its probability-weighted mean, so adding a constant to all logits changes nothing. Second, when one probability is near one, every entry of the Jacobian is near zero — a saturated softmax is a wall for gradients.

**Definition.** For $p = \operatorname{softmax}(z)$ with $p_i = e^{z_i}/\sum_u e^{z_u}$,

$$\frac{\partial p_i}{\partial z_j} = p_i\,(\delta_{ij} - p_j), \qquad J = \frac{\partial p}{\partial z} = \operatorname{diag}(p) - pp^\top .$$

Derivation (two lines): for $i = j$, the quotient rule gives $\partial p_i/\partial z_i = (e^{z_i}\sum_u e^{z_u} - e^{z_i}e^{z_i})/(\sum_u e^{z_u})^2 = p_i - p_i^2$; for $i \ne j$ the numerator is $-e^{z_i}e^{z_j}$, giving $-p_ip_j$. $J$ is symmetric, every row and column sums to zero ($J\mathbf{1} = p - p(p^\top\mathbf{1}) = 0$), and for an upstream gradient $g = \partial L/\partial p$ the chain rule gives

$$\frac{\partial L}{\partial z} = J^\top g = p \odot \big(g - (p^\top g)\mathbf{1}\big),$$

the upstream gradient minus its $p$-weighted mean, scaled by $p$. With cross-entropy $L = -\log p_y$, $g = -e_y/p_y$ and this collapses to the familiar $p - e_y$ (Module 16). **Temperature** $T$ replaces $z$ by $z/T$; the Jacobian becomes $(1/T)(\operatorname{diag}(p) - pp^\top)$ with $p = \operatorname{softmax}(z/T)$, so a low temperature both sharpens $p$ and shrinks the gradient. In attention the same algebra applies to each row of $P = \operatorname{softmax}(S)$ with $S = QK^\top/\sqrt{d_k}$ and $O = PV$. Given $\partial L/\partial O = G$:

$$\frac{\partial L}{\partial V} = P^\top G, \qquad \frac{\partial L}{\partial P} = GV^\top, \qquad \frac{\partial L}{\partial S} = P \odot\Big(GV^\top - \operatorname{rowsum}\big(GV^\top \odot P\big)\mathbf{1}^\top\Big),$$

$$\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}}\,\frac{\partial L}{\partial S}\,K, \qquad \frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}}\,\frac{\partial L}{\partial S}^{\!\top} Q .$$

The value gradient is the attention pattern transposed — a value receives gradient from every position that attended to it. The query and key gradients pass through $P \odot (\cdot)$: in a saturated row every entry of $P$ but one is $\approx 0$ and the one is $\approx 1$ with $GV^\top - \text{mean} \approx 0$, so the row's contribution to $\partial L/\partial Q$ and $\partial L/\partial K$ vanishes. That is the mechanism behind Section 3.

**Worked example.** $p = (0.7, 0.2, 0.1)$:

$$J = \operatorname{diag}(p) - pp^\top = \begin{pmatrix} 0.7 - 0.49 & -0.14 & -0.07 \\ -0.14 & 0.2 - 0.04 & -0.02 \\ -0.07 & -0.02 & 0.1 - 0.01 \end{pmatrix} = \begin{pmatrix} 0.21 & -0.14 & -0.07 \\ -0.14 & 0.16 & -0.02 \\ -0.07 & -0.02 & 0.09 \end{pmatrix}.$$

Each row sums to zero. An upstream gradient $g = (1, 0, 0)$ ("make $p_1$ larger") becomes $\partial L/\partial z = Jg = (0.21, -0.14, -0.07)$: raise $z_1$, lower the others, in proportion to their probabilities. Now saturate: $p = (0.999, 0.0005, 0.0005)$ gives $J_{11} = 0.999 \times 0.001 \approx 0.001$ and every other entry below $5\times10^{-4}$. The same $g$ produces a gradient two hundred times smaller than before. A finite-difference check — perturb $z_j$ by $10^{-5}$ and divide the change in $p_i$ by $10^{-5}$ — reproduces every entry of $J$ to about six digits; the lesson notebook does it.

**Real-life example (manufacturing and quality).** A visual-inspection classifier on a production line outputs a softmax over $\{\text{good}, \text{scratch}, \text{dent}\}$. After a few weeks of training on clean data it reports $0.999$ "good" for almost every part. A new scratch type appears, is mislabelled "good" by the model, and the engineers add the examples to the training set — but the loss barely moves. The reason is the Jacobian: with $p_{\text{good}} = 0.999$ the gradient reaching the logits is scaled by $p_i(1 - p_i) \approx 10^{-3}$, and the layers below receive almost nothing. The standard fixes all act on the same quantity: label smoothing keeps targets away from exact one-hots, a temperature or logit clipping keeps $p$ away from saturation, and in large models learning-rate warm-up keeps the attention and output softmaxes soft during the early, fragile steps.

**In ML/AI this is …** the backward pass of `softmax` and of `scaled_dot_product_attention` in autograd (Module 06); the `p - y` gradient of `CrossEntropyLoss`; label smoothing; and the reason FlashAttention recomputes $P$ from saved row statistics during the backward pass rather than storing the $n\times n$ matrix — the gradient formulas above need only $P$, $G$, $V$, $Q$ and $K$.

## 5. Multi-head attention and projections

**Intuition.** One attention pattern per layer is too little: a token may need to look at the previous word for grammar, at a noun several words back for reference, and at the start of the sentence for context, all at once. Multi-head attention runs several small attentions in parallel, each in its own low-dimensional subspace of the residual stream, and then stitches the results together with one more linear map. Splitting the width $d$ into $h$ heads of width $d/h$ costs no extra parameters compared with a single wide head — it only changes *where* the parameters are used.

**Definition.** With $h$ heads of width $d_{\text{head}} = d/h$, head $i$ has $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d\times d_{\text{head}}}$ and computes $H_i = \operatorname{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}) \in \mathbb{R}^{n\times d_{\text{head}}}$. The heads are concatenated and projected back:

$$\operatorname{MHA}(X) = \big[H_1 \,|\, H_2 \,|\, \cdots \,|\, H_h\big]\, W_O, \qquad W_O \in \mathbb{R}^{d\times d}.$$

Stacking the per-head matrices gives $W_Q, W_K, W_V \in \mathbb{R}^{d\times d}$, so the attention sub-layer has $4d^2$ weights regardless of $h$. The **feed-forward** sub-layer (Section 8) has $W_1 \in \mathbb{R}^{d\times d_{\text{ff}}}$ and $W_2 \in \mathbb{R}^{d_{\text{ff}}\times d}$; with the conventional $d_{\text{ff}} = 4d$ that is $8d^2$ weights. Hence a block has

$$4d^2 + 8d^2 = 12d^2 \ \text{ weights}, \quad \text{plus } O(d) \text{ biases and layer-norm parameters}.$$

Compute, in floating-point operations (one multiply-add counts as $2$), per layer for a sequence of $n$ tokens: multiplying $X \in \mathbb{R}^{n\times d}$ by all $12d^2$ weights costs $2 \cdot 12d^2 \cdot n = 24nd^2$; forming $QK^\top$ for all heads costs $2n^2 d$ and $PV$ another $2n^2d$, so attention adds $4n^2d$:

$$\text{FLOPs per layer} \approx 24nd^2 + 4n^2d, \qquad \frac{4n^2d}{24nd^2} = \frac{n}{6d}.$$

Attention's quadratic term overtakes the weight term when $n > 6d$. The score matrices themselves need $h \cdot n^2$ numbers of memory per layer if materialised — the $O(n^2)$ memory that long contexts fight against. Each head's score matrix $XW_Q^{(i)}W_K^{(i)\top}X^\top$ has rank at most $d_{\text{head}}$ (Module 03): a head is a low-rank bilinear form on the residual stream.

**Worked example (a GPT-2-small-sized layer).** $d = 768$, $h = 12$, $d_{\text{head}} = 64$, $d_{\text{ff}} = 3072$, $n = 1024$. Attention weights: $4 \times 768^2 = 2{,}359{,}296$. Feed-forward weights: $2 \times 768 \times 3072 = 4{,}718{,}592$. Total $7{,}077{,}888 = 12 \times 768^2$ exactly. Biases ($3d + d + d_{\text{ff}} + d = 6{,}912$) and two layer norms ($4d = 3{,}072$) bring the block to $7{,}087{,}872$; twelve blocks give $85{,}054{,}464$. Add the token embedding $50{,}257 \times 768 = 38{,}597{,}376$, the position embedding $1{,}024 \times 768 = 786{,}432$ and the final layer norm ($1{,}536$): $124{,}439{,}808$ parameters — the "124M" of GPT-2 small. FLOPs per layer at full context: $24 \times 1024 \times 768^2 = 1.45\times10^{10}$ for the weights and $4 \times 1024^2 \times 768 = 3.2\times10^{9}$ for attention — $22\,\%$ of the weight cost, since $n/6d = 1024/4608$, or $18\,\%$ of the layer total. At $n = 4608$ the two are equal; at $n = 32{,}768$ attention would be $7\times$ the weight cost.

**Real-life example (computer vision).** A Vision Transformer splits a $224\times224$ image into $16\times16$ patches, giving $n = 196$ tokens; ViT-Base uses the same $d = 768$, $12$ heads, $12$ layers as GPT-2 small and has about $86$ million parameters. At $n = 196$ attention is a negligible $196/4608 = 4\,\%$ of the FLOPs. Move to a $1024\times1024$ medical scan with the same patch size and $n = 4096$: the weight cost grows $21\times$ (linearly in $n$) but the attention cost grows $437\times$ (quadratically), and the $12$ score matrices per layer now hold $12 \times 4096^2 \approx 2\times10^8$ numbers. This is why high-resolution vision models use windowed attention, pooling between stages, or larger patches — the $n^2$ is unavoidable in full attention, so they shrink $n$.

**In ML/AI this is …** `nn.MultiheadAttention(embed_dim=d, num_heads=h)`; the `n_head`, `n_embd` and `n_layer` fields of a GPT config; the parameter-count rule of thumb $N \approx 12\,L\,d^2$ for a decoder with $L$ layers (excluding embeddings), which is where the scaling-law papers' "non-embedding parameters" come from; and the $6N$ FLOPs-per-token training estimate ($2N$ forward, $4N$ backward) that follows from "two FLOPs per weight per token".

## 6. Masking and causality

**Intuition.** A language model is trained to predict the next token, so when it is processing position $i$ it must not be able to see positions $i+1, i+2, \dots$ — otherwise it would simply copy the answer. The causal mask enforces this by making every future position's attention weight exactly zero. The payoff is enormous: because position $i$ is blind to the future, you can feed a whole sentence in at once and read off a prediction at *every* position simultaneously, each one a legitimate next-token prediction. One forward pass yields $n$ training examples.

**Definition.** Define the mask $M \in \mathbb{R}^{n\times n}$ by $M_{ij} = 0$ if $j \le i$ and $M_{ij} = -\infty$ if $j > i$ (strictly above the diagonal). Causal attention is

$$P = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right), \qquad e^{-\infty} = 0 \ \Rightarrow\ P_{ij} = 0 \text{ for } j > i,$$

so $P$ is lower-triangular and each row is a distribution over positions $1, \dots, i$ only. Row $1$ can attend only to itself. Since every later layer is also causal (or acts position-wise), the output at position $i$ depends only on tokens $1..i$ — by induction over layers. **Teacher forcing**: during training the input is the true sequence $t_1, \dots, t_n$ and the target at position $i$ is $t_{i+1}$; the model is scored on its predictions given the *true* prefix rather than its own earlier guesses. The loss is the mean next-token cross-entropy (Module 14),

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n} \log p_\theta\!\left(t_{i+1} \mid t_1, \dots, t_i\right), \qquad \text{perplexity} = e^{\mathcal{L}},$$

so a perplexity of $20$ means the model is, on average, as uncertain as a uniform choice among $20$ tokens. **Padding masks** handle batches of unequal length: padded positions get $-\infty$ as keys so no real token attends to them, and their own loss terms are dropped. Encoder-style models (BERT) use no causal mask — every token sees every other — and are trained with a different objective.

**Worked example.** Three tokens whose scaled scores against positions $1, 2, 3$ happen to be $(1, 2, 3)$ for every query. Masked rows: row 1 is $(1, -\infty, -\infty) \to (1, 0, 0)$; row 2 is $(1, 2, -\infty) \to \operatorname{softmax}(1, 2) = (0.269, 0.731, 0)$; row 3 is $(1, 2, 3) \to (2.718, 7.389, 20.086)/30.193 = (0.090, 0.245, 0.665)$. Without the mask every row would be $(0.090, 0.245, 0.665)$ and position 1 would be reading token 3 — the token it is supposed to predict two steps later. For the loss: if the model assigns probabilities $0.5$, $0.2$ and $0.05$ to the three true next tokens, $\mathcal{L} = -\tfrac13(\ln 0.5 + \ln 0.2 + \ln 0.05) = \tfrac13(0.693 + 1.609 + 2.996) = 1.766$ nats and the perplexity is $e^{1.766} = 5.85$. A uniform model over GPT-2's vocabulary would have $\mathcal{L} = \ln 50257 = 10.82$.

**Real-life example (finance).** A transformer forecasting next-day returns from a history of prices is only valid if the prediction for day $i$ uses prices up to day $i$. Without the causal mask, a sequence model trained on a whole year of data sees day $i+1$'s price while "predicting" it, posts a spectacular backtest, and collapses on live data — the classic look-ahead bias, implemented as a missing $-\infty$. The mask is the mathematical statement of "no information from the future": the output at position $i$ is a function of $x_1, \dots, x_i$ only, and the same forward pass then provides a legitimate out-of-sample test at every day of the year.

**In ML/AI this is …** `is_causal=True` in `scaled_dot_product_attention`, the `torch.tril` buffer named `bias` or `mask` in GPT-2 implementations, `attention_mask` in Hugging Face tokenisers (the padding mask), the `labels = input_ids[1:]` shift in language-model training code, and the `-inf`/`finfo.min` fill value — Module 14's cross-entropy averaged over positions is `nn.CrossEntropyLoss` on the flattened $(n, V)$ logits.

## 7. Positional information

**Intuition.** Attention treats the sequence as a *set*: it compares every query with every key, and nothing in Section 2 mentions where a token sits. Shuffle the tokens and the outputs shuffle identically — "dog bites man" and "man bites dog" would produce the same bag of vectors. Position must therefore be injected explicitly. The elegant way is to encode position as an *angle*: rotating by $\theta$ and then by $\phi$ is the same as rotating by $\theta + \phi$, so shifting a position by $k$ becomes a fixed rotation, and the relationship between two positions becomes a rotation by their *difference*.

**Definition and the two-line proof.** Let $\Pi$ be any $n\times n$ permutation matrix ($\Pi^\top\Pi = I$). Permuting the rows of the input permutes $Q, K, V$: they become $\Pi Q, \Pi K, \Pi V$. Then

$$\operatorname{softmax}\!\big((\Pi Q)(\Pi K)^\top\big)\,\Pi V = \operatorname{softmax}\!\big(\Pi\, QK^\top \Pi^\top\big)\,\Pi V = \Pi\,\operatorname{softmax}(QK^\top)\,\Pi^\top\Pi V = \Pi\,\operatorname{softmax}(QK^\top)\,V ,$$

because a row-wise softmax commutes with permuting rows and columns. So attention (without a causal mask) is **permutation-equivariant**, and so is every position-wise layer; without positional information the model cannot distinguish orders. Three remedies:

*Learned absolute embeddings* (GPT-2): a table $P \in \mathbb{R}^{n_{\max}\times d}$, with $X_{i,:} = E_{t_i,:} + P_{i,:}$. Simple, but undefined beyond $n_{\max}$.

*Sinusoidal encodings* (Vaswani et al., 2017): for position $pos$ and pair index $i = 0, \dots, d/2 - 1$, with frequencies $\omega_i = 10000^{-2i/d}$,

$$PE(pos, 2i) = \sin(pos\,\omega_i), \qquad PE(pos, 2i+1) = \cos(pos\,\omega_i).$$

Each pair $(\sin(pos\,\omega_i), \cos(pos\,\omega_i))$ is a point on the unit circle at angle $pos\,\omega_i$; the first pairs rotate fast (one radian per position), the last pairs rotate slowly (one radian per $10{,}000$ positions), a multi-resolution clock. The key property: shifting by $k$ positions is a *linear* map that does not depend on $pos$,

$$\begin{pmatrix} \sin((pos+k)\omega_i) \\ \cos((pos+k)\omega_i)\end{pmatrix} = \begin{pmatrix} \cos k\omega_i & \sin k\omega_i \\ -\sin k\omega_i & \cos k\omega_i \end{pmatrix}\begin{pmatrix} \sin(pos\,\omega_i) \\ \cos(pos\,\omega_i)\end{pmatrix},$$

by the angle-addition formulas — so $PE(pos + k) = R_k\,PE(pos)$ with a block-diagonal rotation $R_k$, and a learned $W_Q$, $W_K$ can in principle implement "attend $k$ positions back" by matching $q$ against $R_k$ of the key's encoding.

*Rotary position embeddings* (RoPE; Su et al., 2021), used by Llama, Mistral and most current open models: instead of adding positions to $X$, rotate the queries and keys themselves. Pair up the $d_{\text{head}}$ coordinates and rotate pair $i$ at position $m$ by angle $m\theta_i$ with $\theta_i = 10000^{-2i/d_{\text{head}}}$. For one pair, with $R(\alpha)$ the $2\times2$ rotation matrix,

$$q_m = R(m\theta)\,q, \qquad k_n = R(n\theta)\,k, \qquad q_m^\top k_n = q^\top R(m\theta)^\top R(n\theta)\,k = q^\top R\big((n - m)\theta\big)\,k ,$$

using $R(\alpha)^\top = R(-\alpha)$ and $R(-\alpha)R(\beta) = R(\beta - \alpha)$ (Module 02). The score depends on the *relative* offset $n - m$ and on the content $q, k$ — never on the absolute positions. Long-context extrapolation is the weak point of all three schemes: a model trained at $n_{\max} = 4096$ has never seen the angle $m\theta_i$ for $m > 4096$ in its fast-rotating pairs, and attention quality degrades beyond the training length. Practical fixes rescale the angles (position interpolation: use $m\theta_i \cdot n_{\max}/n_{\text{new}}$ so the seen range is reused), or enlarge the base $10000$ so the slow pairs rotate even more slowly, followed by a short fine-tune.

**Worked example.** Sinusoidal, $d = 4$: $\omega_0 = 1$, $\omega_1 = 10000^{-2/4} = 0.01$. $PE(0) = (0, 1, 0, 1)$; $PE(1) = (\sin 1, \cos 1, \sin 0.01, \cos 0.01) = (0.841, 0.540, 0.010, 1.000)$; $PE(2) = (0.909, -0.416, 0.020, 1.000)$. Check the shift property on the fast pair with $k = 1$: $R_1 = \begin{pmatrix} \cos 1 & \sin 1 \\ -\sin 1 & \cos 1\end{pmatrix} = \begin{pmatrix} 0.540 & 0.841 \\ -0.841 & 0.540\end{pmatrix}$, and $R_1 (0.841, 0.540)^\top = (0.454 + 0.454,\ -0.708 + 0.292) = (0.909, -0.416)$ — exactly the fast pair of $PE(2)$. RoPE, one pair, $\theta = 0.5$ rad per position, $q = k = (1, 0)$: at positions $m = 3$, $n = 1$: $q_3 = (\cos 1.5, \sin 1.5) = (0.071, 0.997)$, $k_1 = (\cos 0.5, \sin 0.5) = (0.878, 0.479)$, $q_3^\top k_1 = 0.062 + 0.478 = 0.540 = \cos(1.0) = \cos((1 - 3)\,0.5)$. At positions $8$ and $6$: $q_8 = (-0.654, -0.757)$, $k_6 = (-0.990, 0.141)$, dot product $0.647 - 0.107 = 0.540$ again. Same offset, same score, different absolute positions.

**Real-life example (NLP).** "The lawyer that the client sued was late" and "The client that the lawyer sued was late" contain the same multiset of tokens; who sued whom is carried entirely by order. A permutation-equivariant model assigns both sentences identical token representations, hence identical answers to "who sued?". With positional information the query from "sued" can match keys with a *relative* pattern ("the noun two tokens before me inside the relative clause"), and RoPE makes that pattern a fixed rotation independent of where the clause starts in the document — which is precisely why relative schemes generalise better than absolute tables to sentences in unfamiliar positions. The same mathematics appears in protein language models (positions along the amino-acid chain) and, with two angles instead of one, in vision transformers that rotate by row and column.

**In ML/AI this is …** `wpe` (word position embedding) in GPT-2, the `rotary_emb` / `apply_rotary_pos_emb` functions in Llama-family code (which implement $R(m\theta)$ on pairs as `x * cos + rotate_half(x) * sin`), the `rope_theta` config field (the base $10000$, raised to $500{,}000$ in Llama 3 for longer contexts), and the "position interpolation" and "YaRN" tricks for extending context after training.

## 8. Layer normalisation and residual connections

**Intuition.** A stack of 32 or 96 blocks only trains if the signal flowing through it keeps a stable scale: too small and gradients vanish, too large and softmaxes saturate. Two devices do this. The *residual connection* makes every block compute a small correction added to an identity path, so the stream is never farther than one addition from its input and gradients always have a clean route back (Module 16). *Layer normalisation* rescales each token's vector to zero mean and unit variance across its features before it enters a sub-layer, so the sub-layer always sees inputs of a standard size no matter how much the residual stream has grown.

**Definition.** For one token's vector $x \in \mathbb{R}^d$ (one row of the residual stream), with learned gain $\gamma \in \mathbb{R}^d$ and bias $\beta \in \mathbb{R}^d$,

$$\mu = \frac{1}{d}\sum_{j=1}^d x_j, \qquad \sigma^2 = \frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2, \qquad \operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta ,$$

with $\epsilon \approx 10^{-5}$ for numerical safety. The statistics are over the $d$ features of a *single* token, so the operation is identical for a batch of one and a batch of a thousand, for a sequence of $3$ tokens and of $3{,}000$. Batch normalisation (Module 16) instead averages over the batch and sequence positions, which fails here: sequences have different lengths, padding corrupts the statistics, autoregressive generation runs with one token at a time, and the train-time and inference-time statistics differ. **RMSNorm** (used by Llama and most recent models) drops the mean and the bias: $\operatorname{RMS}(x) = \sqrt{\tfrac1d\sum_j x_j^2 + \epsilon}$, $\operatorname{RMSNorm}(x) = \gamma \odot x/\operatorname{RMS}(x)$ — cheaper, and empirically as good. The **pre-norm** block, standard since GPT-2, is

$$X \leftarrow X + \operatorname{MHA}\big(\operatorname{LN}_1(X)\big), \qquad X \leftarrow X + \operatorname{FFN}\big(\operatorname{LN}_2(X)\big), \qquad \operatorname{FFN}(x) = \phi(x W_1 + b_1)\,W_2 + b_2 ,$$

with $x$ one row (token) of $X$, $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, and $\phi$ the GELU, $\operatorname{GELU}(u) = u\,\Phi(u)$ where $\Phi$ is the standard normal CDF (Module 08) — a smooth ReLU that is $\approx u$ for large positive $u$, $\approx 0$ for large negative $u$, and $-0.159$ at $u = -1$. The original transformer used **post-norm**, $X \leftarrow \operatorname{LN}(X + \operatorname{MHA}(X))$, which normalises the residual stream itself; it needs learning-rate warm-up to train at all because the gradient through the final layers is large at initialisation, whereas pre-norm keeps the identity path untouched and trains stably at depth. The price is that the residual stream's norm grows with depth under pre-norm, which is why a final LayerNorm sits before the unembedding. The feed-forward block has a useful reading as a **key–value memory**: the rows of $W_1$ are $d_{\text{ff}}$ keys, the activation $\phi(W_1 x)$ says how strongly the current token matches each key, and the columns of $W_2$ are the values added to the stream — a lookup table with $4d$ entries per layer, which is where much of a model's factual recall appears to live.

**Worked example.** $x = (2, 4, 6, 8)$: $\mu = 5$, $\sigma^2 = \tfrac14(9 + 1 + 1 + 9) = 5$, $\sqrt{5} = 2.236$, so with $\gamma = \mathbf{1}$, $\beta = 0$, $\epsilon \to 0$: $\operatorname{LN}(x) = (-1.342, -0.447, 0.447, 1.342)$ — mean $0$, variance $1$. Scale the input by $10$ and the output is unchanged; add $100$ to every component and it is unchanged: LayerNorm removes exactly the two degrees of freedom (overall offset and overall scale) and keeps the *pattern*. RMSNorm: $\operatorname{RMS} = \sqrt{(4 + 16 + 36 + 64)/4} = \sqrt{30} = 5.477$, output $(0.365, 0.730, 1.095, 1.461)$ — scale-invariant but not shift-invariant. GELU on the values $(-1, 1, 2)$: $-1 \times \Phi(-1) = -0.159$, $1\times\Phi(1) = 0.841$, $2\times\Phi(2) = 1.954$.

**Real-life example (engineering).** A radio receiver's automatic gain control measures the incoming signal's power over a short window and rescales it to a fixed level before demodulation, so that the demodulator — a nonlinear circuit with a narrow working range — behaves the same whether the transmitter is a kilometre away or a hundred. LayerNorm is automatic gain control for a neural network: the "window" is one token's $d$ features, the "fixed level" is unit variance, and the nonlinear circuit is the softmax and GELU that follow. The learned $\gamma$ and $\beta$ then let the network choose the operating point it prefers, exactly as a receiver has a manual volume knob after the AGC.

**In ML/AI this is …** `nn.LayerNorm(d)` and the `ln_1`, `ln_2`, `ln_f` modules of GPT-2; `RMSNorm` in Llama and Mistral; `nn.GELU` (GPT-2 uses the tanh approximation) and the gated variant SwiGLU, $\big(\operatorname{SiLU}(x W_1) \odot x W_3\big) W_2$, in Llama; the `x = x + attn(ln_1(x))` idiom that marks a pre-norm block; and the residual-stream view of Module 16's skip connections — the gradient of the loss with respect to the stream at layer $\ell$ is the gradient at layer $\ell+1$ plus a correction, never a product of $L$ Jacobians.

## 9. Training objectives and inference

**Intuition.** Training is one big parallel pass per batch: every position predicts its successor, and the mean cross-entropy is minimised by gradient descent. Generation is the opposite: one token at a time, each new token appended to the context and fed back in. The decoding rule — how to turn the probability vector into a token — trades off accuracy against diversity, and the engineering trick that makes generation affordable is to *cache* the keys and values of all earlier tokens so that each new step costs one token's worth of compute rather than the whole context's.

**Definition.** Given the final distribution $p = \operatorname{softmax}(z)$ over $V$ tokens at the current step:

- **Greedy**: emit $\arg\max_v z_v$. Deterministic, often repetitive — a high-probability loop ("I think that I think that …") has no mechanism to escape.
- **Temperature** $T$: sample from $\operatorname{softmax}(z/T)$. $T \to 0$ recovers greedy; $T = 1$ is the model's own distribution; $T > 1$ flattens it and raises its entropy (Module 14). Since $\log p_v^{(T)} = z_v/T - \log Z_T$, temperature rescales *log*-odds: a token $10\times$ less likely at $T = 1$ is $\sqrt{10} \approx 3.2\times$ less likely at $T = 2$.
- **Top-$k$**: keep the $k$ largest probabilities, set the rest to zero, renormalise. Removes the long tail of thousands of individually unlikely but collectively probable junk tokens.
- **Top-$p$ (nucleus)**: sort $p$ descending, keep the smallest prefix whose cumulative probability is $\ge p$, renormalise. Adapts $k$ to the step: one token when the model is sure, many when it is not.
- **Beam search** (width $B$): keep the $B$ partial sequences with the highest total log-probability $\sum_i \log p(t_i \mid t_{<i})$, extend each by every token, and keep the best $B$ again. Finds higher-likelihood sequences than greedy, favours short outputs unless the score is length-normalised, and produces bland text for open-ended generation — it is used for translation and speech recognition, where one correct answer exists.

Repetition is a low-entropy trap: once a phrase has appeared, the attention heads that copy from context raise its probability, greedy decoding selects it, and the loop reinforces itself; a repetition penalty divides the logits of already-seen tokens by a factor $> 1$. The **KV cache**: at generation step $t$ the new token's query must attend to the keys and values of all $t$ positions, but the earlier $K$ and $V$ rows do not change (they depend only on their own tokens, by causality), so they are stored. Per layer that is $K, V \in \mathbb{R}^{t\times d}$, i.e. $2\,t\,d$ numbers, or with $h_{kv}$ key–value heads of width $d_{\text{head}}$,

$$\text{cache bytes} = 2 \times L \times t \times h_{kv}\,d_{\text{head}} \times (\text{bytes per number}), \qquad h_{kv} d_{\text{head}} = d \ \text{ for standard MHA}.$$

Each decoding step then costs $O(d^2 L)$ for the weights plus $O(t\,d\,L)$ for attention against the cache, instead of $O(t d^2 L)$ for recomputing everything. The consequence for serving: a decoding step reads *every* weight once to produce one token, so with batch size $1$ a GPU is memory-bandwidth-bound and mostly idle; batching $B$ sequences reads the weights once for $B$ tokens, raising throughput roughly $B$-fold until the cache or compute saturates. **Grouped-query attention** (GQA) shares each $K, V$ head among several query heads, shrinking $h_{kv}$ and the cache by that factor.

**Worked example.** Logits $z = (2, 1, 0.5, -1)$. At $T = 1$: $p = (0.610, 0.224, 0.136, 0.030)$, entropy $1.014$ nats. At $T = 0.5$ (logits doubled): $p = (0.842, 0.114, 0.042, 0.002)$, entropy $0.538$. At $T = 2$: $p = (0.434, 0.264, 0.205, 0.097)$, entropy $1.265$. Top-$k$ with $k = 2$ at $T = 1$: $(0.610, 0.224)/0.834 = (0.731, 0.269, 0, 0)$. Top-$p$ with $p = 0.9$: cumulative sums $(0.610, 0.834, 0.970, 1.0)$, so three tokens are kept: $(0.629, 0.231, 0.140, 0)$. KV cache for a 7B-class model — $L = 32$ layers, $d = 4096$ ($32$ heads of $128$), context $t = 8192$, fp16 ($2$ bytes): $2 \times 32 \times 8192 \times 4096 \times 2 = 4{,}294{,}967{,}296$ bytes $= 4$ GiB, or $512$ KiB per token. GPT-2 small at its full $1024$-token context: $2 \times 12 \times 1024 \times 768 \times 2 = 36$ MiB. A 70B-class model with $80$ layers and GQA ($8$ KV heads of width $128$): $2 \times 80 \times 8192 \times 1024 \times 2 = 2.5$ GiB — smaller than the 7B model's cache, because $h_{kv}d_{\text{head}} = 1024$ instead of $4096$; with full MHA ($64$ heads) it would be $20$ GiB.

**Real-life example (e-commerce customer service).** A retailer serves a chat assistant from a 7B model on a GPU with $80$ GB of memory and about $2$ TB/s of memory bandwidth. Weights in fp16 take $13.5$ GB; the remaining $\approx 66$ GB is cache. At $4$ GiB per $8192$-token conversation, that is roughly $16$ concurrent conversations at full length; with GQA ($8$ KV heads) the cache drops to $1$ GiB per conversation and the same GPU holds about $60$. Speed: each decoding step must stream the $13.5$ GB of weights through the processor, which at $2$ TB/s takes about $7$ ms — so a *single* conversation cannot exceed roughly $150$ tokens per second however fast the arithmetic is, and that same $7$ ms can serve all $60$ conversations' next tokens at once. This is why serving systems batch aggressively, why "tokens per second" numbers quoted for one user and for a loaded server differ by two orders of magnitude, and why 4-bit weight quantisation (Section 10) speeds up single-user decoding nearly $4\times$: it is bandwidth, not FLOPs, that is being saved.

**In ML/AI this is …** `model.generate(temperature=, top_k=, top_p=, num_beams=, repetition_penalty=)` in Hugging Face; `past_key_values` / `use_cache=True` (the KV cache); `num_key_value_heads` in Llama configs (GQA; $8$ for the 70B model); PagedAttention in vLLM, which manages cache memory in blocks like an operating system manages virtual memory; and Module 14's perplexity, reported on held-out text as the standard measure of a language model's quality.

## 10. Low-rank adaptation and the linear algebra of fine-tuning

**Intuition.** Fine-tuning a 7B-parameter model the ordinary way means storing a gradient and two Adam moments for every parameter — about $16$ bytes per parameter in mixed precision, over $100$ GB. LoRA's bet is that the *change* a fine-tune makes to each weight matrix is approximately low-rank: it moves the model along a few directions, not all $d^2$ of them. If so, you can freeze $W_0$ and learn only a thin pair of factors $B A$ — Module 03's truncated SVD, but learned rather than computed. The adapters are tiny, swappable, and can be folded into $W_0$ afterwards at zero inference cost.

**Definition.** For a frozen weight $W_0 \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, LoRA (Hu et al., 2021) parametrises the fine-tuned weight as

$$W = W_0 + \Delta W, \qquad \Delta W = \frac{\alpha}{r}\, B A, \qquad B \in \mathbb{R}^{d_{\text{out}}\times r},\ A \in \mathbb{R}^{r\times d_{\text{in}}},\ r \ll \min(d_{\text{in}}, d_{\text{out}}),$$

with $A$ initialised Gaussian, $B = 0$ (so training starts exactly at $W_0$), and $\alpha$ a fixed scale. The update has rank at most $r$ and costs $r(d_{\text{in}} + d_{\text{out}})$ parameters instead of $d_{\text{in}}d_{\text{out}}$ — a fraction $r(d_{\text{in}} + d_{\text{out}})/(d_{\text{in}}d_{\text{out}})$, which for square matrices is $2r/d$. The forward pass computes $W_0x + \tfrac{\alpha}{r}B(Ax)$ — two thin products — and after training the adapter is **merged**, $W \leftarrow W_0 + \tfrac{\alpha}{r}BA$, so inference runs the original architecture. Why low rank is plausible: the intrinsic dimension of fine-tuning — the smallest random subspace of parameter space in which a task can be learned to within $90\,\%$ of full fine-tuning — was measured at a few hundred to a few thousand dimensions for models with hundreds of millions of parameters (Aghajanyan et al., 2020), and the SVD of actual full-fine-tune updates shows fast-decaying singular values. If the true update had rank $r^\ast$, Eckart–Young says the rank-$r$ adapter can match it exactly for $r \ge r^\ast$ and makes error $\sqrt{\sum_{i>r}\sigma_i^2}$ otherwise.

**Quantisation** stores weights in $b$ bits instead of $16$. Absmax (symmetric) integer quantisation of a block of weights $w$ with scale $s$ and $b$ bits:

$$s = \frac{\max_j |w_j|}{2^{b-1} - 1}, \qquad q_j = \operatorname{round}(w_j/s), \qquad \hat w_j = s\,q_j, \qquad |w_j - \hat w_j| \le \tfrac{s}{2}, \qquad \operatorname{RMS}(w - \hat w) \approx \frac{s}{\sqrt{12}},$$

the rounding error being approximately uniform on $[-s/2, s/2]$ (variance $s^2/12$, Module 08). An asymmetric scheme adds a zero-point $z$ so that $\hat w = s(q - z)$ and uses the full range for skewed blocks. Because one outlier sets $s$ for the whole block, quantisation is done in small blocks (32–128 weights, each with its own scale), and in QLoRA the frozen base weights are 4-bit while the LoRA factors stay in 16-bit.

**Mixture of experts** replaces one feed-forward block by $N_e$ experts $\operatorname{FFN}_1, \dots, \operatorname{FFN}_{N_e}$ and a router $g = \operatorname{softmax}(W_g x) \in \mathbb{R}^{N_e}$; only the top-$k$ experts (typically $k = 2$) are evaluated, their gates renormalised, and the output is $\sum_{e\in\text{top-}k} \tilde g_e\,\operatorname{FFN}_e(x)$. The parameter count grows $N_e$-fold while the FLOPs per token grow only $k$-fold — a *sparse softmax* in which the Jacobian of Section 4 flows only through the selected gates — at the cost of all those parameters having to sit in memory, and of an auxiliary load-balancing loss to stop the router from collapsing onto two favourite experts.

**Worked example.** A $4096\times4096$ projection has $16{,}777{,}216$ weights. A rank-$8$ LoRA adds $8\times(4096 + 4096) = 65{,}536$ — $0.39\,\%$. Applying it to the query and value projections of all $32$ layers of a 7B model trains $32 \times 2 \times 65{,}536 = 4.19$ million parameters, $0.06\,\%$ of $6.7$ billion; Adam's states for those cost $4.19\text{M}\times 12$ bytes $= 50$ MB instead of $81$ GB. Quantisation: weights in a block range over $[-2, 2]$; int8 gives $s = 2/127 = 0.0157$, maximum error $0.0079$, RMS error $0.0045$; int4 ($2^3 - 1 = 7$ levels each side) gives $s = 2/7 = 0.286$ and RMS error $0.083$ — eighteen times worse, which is why 4-bit schemes rely on small blocks and on the fact that most blocks of trained weights have ranges far below their outliers. With realistic weights of standard deviation $0.02$ and an absmax of $0.088$, int8 gives $s = 6.9\times10^{-4}$ and RMS error $2.0\times10^{-4}$ — $1\,\%$ of the weights' typical size. Mixture of experts: a model with $8$ experts of which $2$ are active (Mixtral's configuration) has roughly $47$ billion parameters but uses about $13$ billion per token.

**Real-life example (healthcare).** A hospital wants a 7B-parameter assistant that writes discharge summaries in its own house style from clinical notes that cannot leave the premises. Full fine-tuning needs about $108$ GB for weights, gradients and optimiser state — several data-centre GPUs. With LoRA rank $16$ on all attention projections, the trainable parameters are a few tens of millions; the base weights are loaded once in 4-bit ($3.4$ GB) and the whole job fits a single $24$ GB workstation card, training overnight on the hospital's own machine. Afterwards the adapter — a file of some tens of megabytes — is merged into the base weights for deployment, and a second adapter trained on radiology reports can be swapped in without touching the first. Every number in that story is one of the formulas above.

**In ML/AI this is …** `peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])` and `model.merge_and_unload()`; `bitsandbytes` 8-bit and NF4 quantisation and the `load_in_4bit=True` flag (QLoRA); GGUF/GPTQ/AWQ weight formats for local inference; and the `num_local_experts` / `num_experts_per_tok` fields of Mixtral-style configs. The SVD view of $\Delta W$ is Module 03 verbatim.

## 11. What the embedding geometry does

**Intuition.** Everything a transformer knows about a token at a given moment is a point in $\mathbb{R}^d$. Nearby points mean "similar in the ways the model cares about"; and because the model's own readout is linear (logits are dot products with $E$, Section 1), properties the model uses tend to be encoded as *directions*: "is a verb", "refers to a female person", "sentiment positive". That is a linear-algebra statement with a precise test — can a linear classifier read the property off the vector? — and some precise failure modes.

**Definition.** Cosine similarity (Module 01) of two vectors is $\cos\angle(u, v) = u^\top v/(\|u\|\|v\|)$. For contextual hidden states one also measures **anisotropy**: the mean cosine similarity between unrelated tokens' vectors. An isotropic representation has mean cosine near $0$; real transformer hidden states are strongly anisotropic — all vectors share a large common component — with mean cosine between random words often above $0.5$ in upper layers (Ethayarajh, 2019). The fix is to centre (subtract the mean vector) and often to whiten (Module 03) before comparing. A **linear probe** fits logistic regression (Module 13) $\hat y = \sigma(w^\top h + b)$ from frozen hidden states $h$ at some layer to a label (part of speech, sentiment, truth of a statement); high probe accuracy means the property is *linearly* accessible there. The **linear representation hypothesis** — concepts are directions, and $h + \lambda w$ moves the model's output along the concept — is exactly what the word-analogy arithmetic $\text{king} - \text{man} + \text{woman} \approx \text{queen}$ expressed for static embeddings, and what steering vectors do for transformers.

**Worked example.** $u = (1, 0, 0)$ and $v = (0, 1, 0)$ are orthogonal, cosine $0$. Add a common component $c = (3, 3, 3)$ to both, as anisotropy does: $u' = (4, 3, 3)$, $v' = (3, 4, 3)$, $u'^\top v' = 12 + 12 + 9 = 33$, $\|u'\|^2 = \|v'\|^2 = 34$, cosine $33/34 = 0.971$. Two unrelated vectors now look nearly identical; subtracting the mean $(3.5, 3.5, 3)$ gives $(0.5, -0.5, 0)$ and $(-0.5, 0.5, 0)$ with cosine $-1$ — the shared offset was hiding the only information there was. Linear probe: hidden states from a two-class problem with means $\mu_0$ and $\mu_1$ and identity covariance have the Bayes-optimal linear direction $w \propto \mu_1 - \mu_0$; logistic regression recovers it, and its accuracy is the measure of how linearly separable the concept is at that layer.

**Real-life example (computer vision).** CLIP (Radford et al., 2021) trains an image encoder and a text encoder so that matching image–caption pairs have high cosine similarity and mismatched pairs low. Zero-shot classification is then pure geometry: embed the prompts "a photo of a cat", "a photo of a dog", …, embed the image, compute the cosines, multiply by a learned temperature (around $100$ at the end of training) and take a softmax. The anisotropy pitfall appears here too — raw cosines between any image and any caption cluster in a narrow band — which is why the temperature is large and why practitioners sometimes centre the embeddings when building a retrieval index. The same cosine-plus-temperature is the retrieval step of every RAG system: the "relevance score" of a document chunk to a question is $\cos\angle(e_q, e_d)$ in a sentence-embedding space.

**In ML/AI this is …** `cosine_similarity` in sentence-embedding and vector-database APIs; `output_hidden_states=True` followed by `sklearn.linear_model.LogisticRegression` — a probe; the `logit_scale` of CLIP; activation steering and "representation engineering" in interpretability research; and the reason the unembedding works at all — the final LayerNorm plus the tied $E$ make the next-token logits a cosine-like comparison between the residual stream and each token's direction.

## Common pitfalls and misconceptions

1. **Building the causal mask as `np.triu(np.ones((n, n)), 1) * -np.inf`.** The entries that should stay unmasked become $0 \times (-\infty) = \text{NaN}$ and every attention row is NaN. → Use `np.where(mask, -np.inf, 0.0)` or fill with a large negative finite number such as `np.finfo(dtype).min`; and subtract the row maximum before exponentiating so the softmax never overflows.
2. **Taking the softmax over the wrong axis.** `softmax(S, axis=0)` normalises columns, so the weights of one query no longer sum to one and the output is not a weighted average. → Rows are queries, columns are keys; normalise over the last axis, and check `P.sum(axis=-1) == 1`.
3. **Forgetting the $\sqrt{d_k}$ or scaling after the softmax.** The scores have variance $d_k$, the softmax saturates, the attention gradients (Section 4) vanish, and training stalls or needs an absurdly small learning rate. → Scale the scores *before* the softmax; the simulation in the lesson shows the entropy collapse directly.
4. **Normalising over the sequence in LayerNorm.** Averaging over positions mixes tokens and breaks causality at inference (the statistics of position $i$ would depend on later tokens). → LayerNorm statistics are over the $d$ features of one token: `axis=-1`, one mean and one variance per row.
5. **Reading attention weights as explanations.** A weight of $0.9$ on token $j$ says the output of this head at this layer was mostly $v_j$; it does not say token $j$ caused the prediction — the value may be near zero, later layers may discard it, and other heads may dominate. → Treat attention maps as one diagnostic among several; use ablations or probes for causal claims.
6. **Thinking the parameter count sets the memory of inference.** For long contexts and large batches the KV cache dominates: $4$ GiB per $8192$-token sequence for a 7B model whose weights take $13.5$ GB. → Budget $2Lt\,h_{kv}d_{\text{head}}$ bytes-per-number for the cache, and remember that decoding is bandwidth-bound, not FLOP-bound.
7. **Believing LoRA slows inference.** The adapter is merged, $W \leftarrow W_0 + \tfrac{\alpha}{r}BA$, so the deployed model has exactly the original shapes and cost. → Merge before deployment; keep adapters separate only when you need to swap them.
8. **Applying temperature after top-$k$, or to probabilities instead of logits.** Temperature acts on logits ($z/T$) before the softmax; filtering then renormalising is a separate step, and dividing probabilities by $T$ does nothing meaningful. → Order: logits → divide by $T$ → softmax → top-$k$/top-$p$ filter → renormalise → sample.

## Real-world case studies

### Case study 1 — Anatomy of GPT-2 small: where the 124 million parameters live

GPT-2 small is the smallest model whose configuration everybody knows, which makes it the ideal object for a complete budget. Configuration: $V = 50{,}257$, $n_{\max} = 1024$, $d = 768$, $h = 12$ (so $d_{\text{head}} = 64$), $d_{\text{ff}} = 3072$, $L = 12$, pre-norm, learned absolute positions, tied unembedding, GELU. Per block: $W_Q, W_K, W_V, W_O$ hold $4 \times 768^2 = 2{,}359{,}296$ weights; $W_1, W_2$ hold $2 \times 768 \times 3072 = 4{,}718{,}592$; the sum $7{,}077{,}888$ is $12d^2$ on the nose, and the biases and two LayerNorms add $9{,}984$ more. Twelve blocks: $85{,}054{,}464$ — $68\,\%$ of the model. The token embedding is the single largest tensor, $38{,}597{,}376$ ($31\,\%$); the position table is $786{,}432$; the final LayerNorm $1{,}536$. Total $124{,}439{,}808$.

Now the compute. For one full-context forward pass, $n = 1024$: weights cost $24nd^2 = 1.45\times10^{10}$ FLOPs per layer, attention $4n^2d = 3.2\times10^9$, i.e. $1.77\times10^{10}$ per layer and $2.1\times10^{11}$ for twelve — plus the unembedding, $2 \times 1024 \times 768 \times 50{,}257 = 7.9\times10^{10}$, which is not small: the vocabulary projection is $37\,\%$ of the attention-plus-FFN cost for this model, a consequence of its tiny $d$. In fp16 the weights occupy $249$ MB; the KV cache at full context is $36$ MiB per sequence; the $12$ score matrices per layer, if materialised in fp32, take $12 \times 1024^2 \times 4 = 48$ MiB per layer. The crossover $n = 6d = 4608$ is beyond the model's context, so GPT-2 small is weight-dominated everywhere it can run. Scale the same recipe to GPT-2 XL ($d = 1600$, $L = 48$): $12 \times 48 \times 1600^2 = 1.47$ billion non-embedding parameters, which is where its "1.5B" comes from. Every one of these numbers was derived from Sections 1, 5, 6 and 9, and the lesson notebook's `count_params` function reproduces them.

### Case study 2 — Serving a 7B model: cache, bandwidth and batching

Take a Llama-2-7B-class decoder: $L = 32$, $d = 4096$, $32$ heads of $128$, $d_{\text{ff}} = 11008$ (SwiGLU uses three matrices), $V = 32{,}000$, context $4096$, about $6.7$ billion parameters. Weights in fp16: $13.5$ GB. Per token per layer the cache holds $2d = 8192$ fp16 numbers; across $32$ layers that is $512$ KiB per token, $2$ GiB per $4096$-token conversation and $4$ GiB if the context is extended to $8192$. A GPU with $80$ GB therefore holds the weights plus about $16$ full-length $8$k conversations, or $32$ at $4$k.

Time per decoding step is set by memory traffic: every step reads all $13.5$ GB of weights plus the cache of every active sequence. At $2$ TB/s the weights alone take $6.7$ ms, so one user sees at most $\approx 150$ tokens/s — and the arithmetic units are nearly idle, because a single token's $2 \times 6.7\times10^9 = 1.3\times10^{10}$ FLOPs would take a few tens of microseconds at hundreds of TFLOP/s. Batching $B = 32$ sequences reuses each weight read for $32$ tokens; the cache reads grow with $B$ (at $8$k context, $32 \times 4$ GiB $= 128$ GiB per step — more than the weights, and more than the GPU holds). This is where grouped-query attention earns its place: with $8$ KV heads instead of $32$, the cache is $128$ KiB per token, $1$ GiB per $8$k conversation, and $B = 48$ fits alongside the weights. Prefill — processing the prompt — is the opposite regime: all $n$ prompt tokens go through in one pass, $24nd^2 + 4n^2d$ FLOPs per layer, compute-bound, which is why the "time to first token" scales with prompt length while the per-token speed afterwards does not. The numbers also explain the popularity of 4-bit weights for local inference: $13.5$ GB becomes $3.4$ GB, the per-step weight read drops $4\times$, and the single-user speed rises almost $4\times$ with no change in FLOPs.

### Case study 3 — LoRA fine-tuning on a single GPU

A team wants to adapt the same 7B model to summarise technical support tickets. Full fine-tuning in mixed precision keeps fp16 weights ($2$ bytes), fp16 gradients ($2$), an fp32 master copy ($4$) and two fp32 Adam moments ($8$): $16$ bytes per parameter, $108$ GB before a single activation is stored — four to eight data-centre GPUs. LoRA with rank $r = 16$ on all four attention projections of all $32$ layers trains $32 \times 4 \times 16 \times (4096 + 4096) = 16.8$ million parameters, $0.25\,\%$ of the model; their gradients and Adam states total $16.8\text{M} \times 16 = 268$ MB. The frozen base is loaded in 4-bit ($3.4$ GB) with the LoRA factors in bf16, and activations for a batch of $8$ sequences of $2048$ tokens with gradient checkpointing fit in the rest of a $24$ GB card.

The linear algebra says what can and cannot be learned. Each adapted matrix moves within a rank-$16$ subspace: the change $\Delta W = \tfrac{\alpha}{r}BA$ has at most $16$ non-zero singular values, so if the "true" full-fine-tune update for this task has a singular-value spectrum that decays quickly — as measured updates typically do — the adapter captures most of its energy, and Eckart–Young bounds what it misses by $\sqrt{\sum_{i>16}\sigma_i^2}$. Doubling $r$ doubles the adapter's parameters and halves nothing in inference cost, because the merged weight $W_0 + \tfrac{\alpha}{r}BA$ is a dense $4096\times4096$ matrix either way. The team ends up with a $34$ MB adapter file ($16.8$M parameters in bf16), keeps three of them for three ticket categories, and swaps them at request time — the base model's $3.4$ GB is shared. In the lesson notebook you will see this trade-off on a synthetic update of true rank $4$: the rank-$r$ reconstruction error is exactly zero for $r \ge 4$ and follows $\sqrt{\sum_{i>r}\sigma_i^2}$ below.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Embedding as one-hot matmul, tied unembedding | `nn.Embedding`, `lm_head`, item embeddings in recommenders | Logits are dot products with token directions; gradients hit only the rows that occur |
| Scaled dot-product attention | Every transformer layer, cross-attention in encoder–decoders and diffusion models | Data-dependent linear combination of values; the attention map is the row-stochastic $P$ |
| $\sqrt{d_k}$ scaling | `scaled_dot_product_attention`, QK-norm | Keeps score variance at $1$ so the softmax does not saturate and gradients survive |
| Softmax Jacobian $\operatorname{diag}(p) - pp^\top$ | Backprop through attention and the output layer, label smoothing, temperature | Explains vanishing gradients at saturation; cross-entropy gradient $p - y$ |
| $12d^2$ parameters, $24nd^2 + 4n^2d$ FLOPs | Scaling laws, model cards, hardware sizing | Parameter and compute budgets from the config; attention dominates beyond $n = 6d$ |
| Causal mask, teacher forcing | GPT-style pre-training, time-series transformers | One parallel pass gives $n$ next-token examples; no look-ahead |
| Permutation equivariance | Why positional encodings exist; set transformers, point clouds | Attention sees a set; order must be injected |
| Sinusoidal / RoPE rotations | `wpe`, `rotary_emb`, `rope_theta`, position interpolation | Relative offsets become rotations; extrapolation limits of long context |
| LayerNorm, RMSNorm, pre-norm residuals | `nn.LayerNorm`, Llama's `RMSNorm`, `x + f(ln(x))` | Stable training at depth; per-token statistics suit variable-length sequences |
| Decoding rules | `generate(temperature, top_k, top_p, num_beams)` | Accuracy–diversity trade-off; entropy of the sampling distribution |
| KV cache arithmetic | `past_key_values`, GQA, PagedAttention | Serving memory and throughput; decoding is bandwidth-bound |
| LoRA $W_0 + BA$, quantisation, MoE | `peft`, QLoRA, GGUF, Mixtral | Fine-tuning on one GPU; weight compression; sparse compute |
| Cosine similarity, probes, anisotropy | CLIP, RAG retrieval, interpretability | Concepts as directions; centring before comparing |

## Cheat sheet

| Fact | Formula / statement |
|---|---|
| Embedding and logits | $x = e_t^\top E = E_{t,:}$; tied: $z = Eh$, $p = \operatorname{softmax}(z)$ |
| Attention | $\operatorname{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V$, row-wise softmax; $o_i = \sum_j P_{ij}v_j$ |
| Score variance | $\operatorname{Var}(q^\top k) = d_k$ for i.i.d. zero-mean unit-variance components; scaled variance $1$ |
| Softmax Jacobian | $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$; $J = \operatorname{diag}(p) - pp^\top$; $J^\top g = p\odot(g - p^\top g)$ |
| Attention backward | $\partial V = P^\top G$; $\partial S = P\odot(GV^\top - \operatorname{rowsum}(GV^\top\odot P))$; $\partial Q = \partial S\,K/\sqrt{d_k}$; $\partial K = \partial S^\top Q/\sqrt{d_k}$ |
| Multi-head | $d_{\text{head}} = d/h$; $[H_1\,\vert\cdots\vert\,H_h]W_O$; $4d^2$ attention weights independent of $h$ |
| Parameters per block | $12d^2$ weights with $d_{\text{ff}} = 4d$ (+ $O(d)$); model $\approx 12Ld^2 + Vd$ |
| FLOPs per layer | $24nd^2 + 4n^2d$; ratio $n/(6d)$; training $\approx 6N$ FLOPs per token |
| Causal mask | $M_{ij} = -\infty$ for $j > i$; $P$ lower-triangular; loss $= -\tfrac1n\sum_i \log p(t_{i+1}\mid t_{\le i})$; perplexity $e^{\mathcal L}$ |
| Permutation equivariance | $\operatorname{Attn}(\Pi Q, \Pi K, \Pi V) = \Pi\operatorname{Attn}(Q, K, V)$ |
| Sinusoidal PE | $PE(pos, 2i) = \sin(pos\,\omega_i)$, $PE(pos, 2i+1) = \cos(pos\,\omega_i)$, $\omega_i = 10000^{-2i/d}$; $PE(pos+k) = R_k PE(pos)$ |
| RoPE | $q_m = R(m\theta)q$, $k_n = R(n\theta)k$ $\Rightarrow$ $q_m^\top k_n = q^\top R((n-m)\theta)k$ |
| LayerNorm | $\gamma\odot(x-\mu)/\sqrt{\sigma^2+\epsilon} + \beta$, statistics over the $d$ features of one token; RMSNorm $\gamma\odot x/\operatorname{RMS}(x)$ |
| Pre-norm block | $X \leftarrow X + \operatorname{MHA}(\operatorname{LN}_1 X)$; $X \leftarrow X + \operatorname{GELU}(\operatorname{LN}_2(X)\,W_1)\,W_2$ |
| Temperature | $\operatorname{softmax}(z/T)$; $T\to0$ greedy; entropy increases with $T$ |
| Top-$k$ / top-$p$ | Keep $k$ largest / smallest prefix with cumulative mass $\ge p$; renormalise |
| KV cache | $2\,L\,t\,h_{kv}d_{\text{head}}\times\text{bytes}$; 32 layers, $d = 4096$, $t = 8192$, fp16 $\Rightarrow 4$ GiB |
| LoRA | $W = W_0 + \tfrac{\alpha}{r}BA$; $r(d_{\text{in}}+d_{\text{out}})$ parameters ($2r/d$ of a square matrix); merge for inference |
| Absmax quantisation | $s = \max\lvert w\rvert/(2^{b-1}-1)$; $\hat w = s\operatorname{round}(w/s)$; error $\le s/2$, RMS $s/\sqrt{12}$ |
| MoE routing | $g = \operatorname{softmax}(W_gx)$, keep top-$k$, renormalise; params $\times N_e$, FLOPs $\times k$ |
| Anisotropy | Mean cosine of unrelated vectors $\gg 0$; centre before comparing |

## Self-assessment

1. **Why is "lookup row $t$ of $E$" the same as a matrix product, and what does that imply for the gradient of $E$?**
   <details><summary>Answer</summary>
   $e_t^\top E$ with $e_t$ one-hot selects row $t$. By the chain rule, $\partial L/\partial E = e_t\,(\partial L/\partial x)^\top$, which is zero everywhere except row $t$: only the embeddings of tokens that actually appear in the batch receive a gradient. Rare tokens are therefore updated rarely, which is one reason their embeddings are poorly trained.
   </details>

2. **All scores in one row of $QK^\top$ are equal. What is the attention output for that query?**
   <details><summary>Answer</summary>
   The softmax of a constant row is uniform, $P_{ij} = 1/n$, so $o_i = \tfrac1n\sum_j v_j$ — the plain mean of the value vectors. This is also what a freshly initialised, heavily scaled-down attention head computes, and why early in training attention outputs look like averages.
   </details>

3. **Head width $d_k = 256$, no scaling. What is the standard deviation of the raw scores at initialisation, and roughly what probability does a score one standard deviation above its two competitors receive?**
   <details><summary>Answer</summary>
   $\sqrt{256} = 16$. A score $16$ above two others gives $\operatorname{softmax}(16, 0, 0) \approx (1 - 2e^{-16}, e^{-16}, e^{-16})$, i.e. $p_1 \approx 1 - 2\times10^{-7}$. The diagonal Jacobian entry $p_1(1 - p_1) \approx 2\times10^{-7}$: effectively no gradient reaches the queries and keys of that row.
   </details>

4. **Show that the rows of the softmax Jacobian sum to zero, and explain what this means for the gradient with respect to the logits.**
   <details><summary>Answer</summary>
   $J\mathbf 1 = \operatorname{diag}(p)\mathbf 1 - pp^\top\mathbf 1 = p - p\,(p^\top\mathbf 1) = p - p = 0$ since $\sum_i p_i = 1$. Because $J$ is symmetric, also $\mathbf 1^\top J = 0$, so $\mathbf 1^\top(\partial L/\partial z) = \mathbf 1^\top J^\top g = 0$: the logit gradient always sums to zero. Adding a constant to all logits does not change the softmax, so the loss has no gradient in that direction.
   </details>

5. **A decoder has $d = 2048$, $L = 24$, $d_{\text{ff}} = 8192$, $V = 32{,}000$, untied unembedding. Estimate its parameter count, and the context length at which attention FLOPs equal weight FLOPs.**
   <details><summary>Answer</summary>
   Blocks: $12 \times 24 \times 2048^2 = 1.21\times10^9$. Embedding and unembedding: $2 \times 32{,}000 \times 2048 = 1.31\times10^8$. Total about $1.34$ billion. Crossover at $n = 6d = 12{,}288$ tokens.
   </details>

6. **Why can a causally masked transformer be trained on all $n$ next-token predictions of a sequence in a single forward pass, while an RNN must process tokens sequentially?**
   <details><summary>Answer</summary>
   With the causal mask, the output at position $i$ depends only on tokens $1..i$, so the prediction at every position is a valid next-token prediction given the true prefix (teacher forcing), and all $n$ of them can be computed as one matrix product $PV$ with a lower-triangular $P$. An RNN's state at position $i$ is a function of the state at $i-1$, so the computation is inherently serial; a transformer's positions are coupled only through attention, which is one parallel operation.
   </details>

7. **Prove in two lines that attention without positional information is permutation-equivariant, and state why RoPE still allows the model to use order.**
   <details><summary>Answer</summary>
   With $\Pi$ a permutation matrix, $\operatorname{softmax}(\Pi QK^\top\Pi^\top)\Pi V = \Pi\operatorname{softmax}(QK^\top)\Pi^\top\Pi V = \Pi\operatorname{softmax}(QK^\top)V$, using that row-wise softmax commutes with permuting rows and columns and $\Pi^\top\Pi = I$. RoPE breaks the symmetry by rotating $q_m$ and $k_n$ by position-dependent angles so that $q_m^\top k_n = q^\top R((n-m)\theta)k$ depends on the relative offset $n - m$; permuting the tokens changes the offsets and hence the scores.
   </details>

8. **Why do transformers use layer normalisation rather than batch normalisation?**
   <details><summary>Answer</summary>
   LayerNorm normalises over the features of one token, so it needs no batch statistics: it behaves identically for a batch of one, for sequences of different lengths with padding, and during autoregressive generation one token at a time. BatchNorm's statistics depend on which other sequences and positions are in the batch, differ between training and inference, and would leak information across positions and examples.
   </details>

9. **A model has $L = 48$, $h_{kv} = 8$, $d_{\text{head}} = 128$, fp16 cache. How many bytes of KV cache does one token need, and how much for a batch of $16$ sequences of $32{,}768$ tokens?**
   <details><summary>Answer</summary>
   Per token: $2 \times 48 \times 8 \times 128 \times 2 = 196{,}608$ bytes $= 192$ KiB. Batch: $16 \times 32{,}768 \times 192$ KiB $= 96$ GiB — more than an $80$ GB GPU holds even before the weights. Either shorten the context, shrink the batch, quantise the cache to $8$ bits, or shard across GPUs.
   </details>

10. **A rank-$16$ LoRA on a $4096\times4096$ matrix is trained, and a colleague proposes rank $32$ "because inference is cheap anyway". What does rank change, and what does it not?**
    <details><summary>Answer</summary>
    Rank $32$ doubles the adapter's parameters ($32 \times 8192 = 262{,}144$ vs $131{,}072$), its optimiser state and its file size, and enlarges the subspace the update can live in (Eckart–Young: the achievable error drops from $\sqrt{\sum_{i>16}\sigma_i^2}$ to $\sqrt{\sum_{i>32}\sigma_i^2}$ of the ideal update). It changes nothing about inference once merged: $W_0 + \tfrac{\alpha}{r}BA$ is a dense $4096\times4096$ matrix at either rank. Whether rank $32$ helps depends on how fast the singular values of the true update decay.
    </details>

## Further reading

- Ashish Vaswani et al., "Attention Is All You Need" (NeurIPS 2017, arXiv:1706.03762) — the original paper: scaled dot-product attention, multi-head attention, sinusoidal encodings, the $d_{\text{ff}} = 4d$ convention.
- Alec Radford et al., "Language Models are Unsupervised Multitask Learners" (OpenAI, 2019) — GPT-2: pre-norm blocks, byte-level BPE, the 124M/355M/774M/1.5B configurations.
- Jianlin Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021, arXiv:2104.09864) — RoPE and the relative-position derivation of Section 7.
- Jimmy Lei Ba, Jamie Ryan Kiros and Geoffrey Hinton, "Layer Normalization" (2016, arXiv:1607.06450); Ruibin Xiong et al., "On Layer Normalization in the Transformer Architecture" (ICML 2020) — why pre-norm trains without warm-up.
- Edward J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021, arXiv:2106.09685); Tim Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023, arXiv:2305.14314).
- Ari Holtzman et al., "The Curious Case of Neural Text Degeneration" (ICLR 2020, arXiv:1904.09751) — nucleus sampling and why greedy/beam decoding repeats.
- Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022) — the $O(n^2)$ memory problem and its solution; Joshua Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023) — the KV-cache reduction of Section 9.
- Jared Kaplan et al., "Scaling Laws for Neural Language Models" (2020, arXiv:2001.08361) — the $N \approx 12Ld^2$ and $C \approx 6ND$ accounting.
- Mary Phuong and Marcus Hutter, "Formal Algorithms for Transformers" (2022, arXiv:2207.09238) — every transformer variant as precise pseudocode with shapes; Nelson Elhage et al., "A Mathematical Framework for Transformer Circuits" (Anthropic, 2021) — the residual-stream view and attention heads as low-rank bilinear maps.
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (free video lecture) and the accompanying nanoGPT repository; Jay Alammar, "The Illustrated Transformer" (free blog post) — the two best informal companions to this module.

## What's in the notebooks

**`lesson.ipynb`** mirrors the eleven sections with NumPy code you can edit: embedding lookup as a one-hot matrix product and tied-unembedding logits on the five-token example; scaled dot-product attention on a four-token sequence with a heat-map of $P$ and a check that uniform scores give the mean of the values; a simulation of the variance of $QK^\top$ with and without $\sqrt{d_k}$ scaling for $d_k \in \{4, 16, 64, 256\}$ and the collapse of attention entropy; the softmax Jacobian verified against finite differences and its shrinkage under saturation and temperature; a multi-head attention forward pass with shape printing, a `count_params` function that reproduces GPT-2 small's $124{,}439{,}808$, and the $24nd^2$ versus $4n^2d$ crossover plot; the causal mask with a leakage test (changing a future token leaves earlier outputs unchanged) and the next-token loss under teacher forcing; a sinusoidal positional-encoding heat-map with numerical verification of $PE(pos+k) = R_kPE(pos)$, and RoPE on 2-D pairs showing that $q_m^\top k_n$ depends only on $n - m$; LayerNorm and RMSNorm from scratch and a complete pre-norm transformer block; greedy, temperature, top-$k$, top-$p$ and beam decoding on a toy logit vector with an entropy-versus-temperature sweep, and a KV-cache calculator; LoRA as rank-$r$ approximation of a rank-4 update with the Eckart–Young error curve, parameter savings and absmax quantisation error; and embedding geometry — anisotropy, centring, and a linear probe on synthetic hidden states.

**`exercises.ipynb`** has twelve exercises with instant self-checks: embedding lookup and tied logits (hidden answer); scaled dot-product attention against the guide's worked example; LayerNorm; the softmax Jacobian against finite differences (hidden answer at saturated logits); a causal mask with a no-leakage test; a parameter-count function (hidden answers for GPT-2 small and GPT-2 XL); sinusoidal encodings and the relative-shift matrix; RoPE and its relative-position property; KV-cache bytes and the number of sequences that fit on a GPU (hidden answers); a complete pre-norm transformer block checked against a reference; a conceptual diagnosis of the argument "temperature makes the $\sqrt{d_k}$ unnecessary"; and a mini-project: train a single-head causal self-attention model with hand-derived gradients to predict the next token of a periodic sequence, gradient-check it, and plot the attention pattern it learns. `solutions.ipynb` contains reference implementations and explanations.
