# Module 01 — Vectors and Geometry

> After this module you can treat any data record — a patient, a pixel grid, a sentence — as a point in $\mathbb{R}^n$, measure how far apart two such points are, how aligned they are, split them with a hyperplane, and explain why all of this behaves strangely once $n$ is in the hundreds.

## Why this matters for ML and AI

Every machine-learning system you will meet in this course begins with the same move: it turns the thing it cares about into a list of numbers. A hospital record becomes $(54, 130, 210, 27.5)$ — age, systolic blood pressure, cholesterol, BMI. A $28 \times 28$ grey-scale digit becomes 784 numbers between 0 and 1. The token "insulin" entering a transformer becomes a vector with 768 entries (GPT-2 small) or 12,288 entries (GPT-3 175B). Once that move is made, "learning" is geometry: nearby points should get similar predictions, a classifier is a flat sheet that separates two clouds of points, and "this document is relevant to that query" means two arrows point in roughly the same direction.

The vocabulary of this module is therefore not a warm-up — it *is* the working language of the field. Retrieval-augmented generation (RAG) ranks passages by the cosine similarity of their embedding vectors. A recommender such as the ones behind Netflix or Spotify places users and items in one shared space and scores a pair by their dot product. Logistic regression, linear support vector machines and the final layer of every neural classifier are hyperplanes. The $\sqrt{d_k}$ that divides attention scores in a transformer exists only because of a fact about dot products of random high-dimensional vectors that you will simulate in the lesson notebook.

There is also a trap waiting. Our intuition is built in three dimensions, and almost everything about distance, angle and volume changes character when the dimension is 100 or 1,000. Section 8 is about that: why nearest-neighbour search degrades, why random embeddings are nearly orthogonal, and why this is sometimes a curse and sometimes a gift.

## Learning objectives

By the end of this module you will be able to:

- Represent data records, images and tokens as vectors in $\mathbb{R}^n$ and manipulate them with addition, scaling and linear combinations.
- Decide whether a set of vectors is linearly independent — by hand, from a picture and with `np.linalg.matrix_rank` — and describe its span as a line, plane or subspace through the origin.
- Compute the $L_1$, $L_2$ and $L_\infty$ norms of a vector by hand and in NumPy, and choose between them for a given task.
- State and use the algebraic and geometric definitions of the dot product, and derive one from the other.
- Compute angles, test orthogonality, and apply the Cauchy–Schwarz and triangle inequalities.
- Explain when cosine similarity is preferable to Euclidean distance and implement top-$k$ nearest-neighbour retrieval over an embedding table.
- Project a vector onto a direction, split it into "explained" and "residual" parts, and verify that the residual is orthogonal.
- Build an orthonormal basis with Gram–Schmidt and read off coordinates with dot products.
- Describe a linear classifier as a hyperplane and compute signed distances and margins.
- Predict how pairwise distances and cosine similarities of random vectors behave as the dimension grows, and connect that to attention scaling and nearest-neighbour methods.

## Prerequisites

This is the first module, so nothing from the course is assumed. From school mathematics you should recall Pythagoras' theorem, the definition of $\cos\theta$ in a right triangle, how to solve two linear equations in two unknowns, and basic Python with NumPy arrays (`np.array`, slicing, `@` for matrix products). Everything else is built here.

Forward links you will see repeatedly: [Module 02 — Matrices and linear maps](../02-matrices-and-linear-maps/README.md) turns projections into least squares; [Module 03 — Eigenvalues, SVD and PCA](../03-eigenvalues-svd-and-pca/README.md) finds the *best* orthonormal basis for a dataset; [Module 13 — Linear models and regularisation](../../part-5-learning-theory/13-linear-models-and-regularization/README.md) explains why the $L_1$ norm produces sparsity; [Module 17 — Mathematics of transformers](../../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md) is dot products, norms and rotations applied at scale.

## 1. Vectors: arrows, points and data records

**Intuition.** A vector is one object with three faces. As an *arrow* it is a displacement: "go 3 east and 4 north". As a *point* it is a location: the tip of that arrow when it starts at the origin. As a *data record* it is simply an ordered list of measurements. The power of linear algebra is that the same rules — add component-wise, scale every component — are meaningful in all three readings. Adding two patient records and halving the result gives the "average patient"; subtracting last year's record from this year's gives the "change" vector; both are the same operation as placing arrows tip-to-tail.

**Definition.** A vector $\mathbf{v} \in \mathbb{R}^n$ is an ordered tuple of $n$ real numbers, $\mathbf{v} = (v_1, v_2, \dots, v_n)$; $n$ is the *dimension*, and we write vectors in bold lowercase and their components in italic with an index. Two operations define the whole subject:

$$\mathbf{u} + \mathbf{v} = (u_1 + v_1, \dots, u_n + v_n), \qquad c\,\mathbf{v} = (c v_1, \dots, c v_n) \quad (c \in \mathbb{R}).$$

*Linear combinations.* A *linear combination* of $\mathbf{v}_1, \dots, \mathbf{v}_k$ is any vector

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k, \qquad c_1, \dots, c_k \in \mathbb{R}.$$

Read it as a recipe: scale each ingredient vector by its coefficient, then add. Geometrically it is a walk: go $c_1$ times along $\mathbf{v}_1$, then $c_2$ times along $\mathbf{v}_2$, and so on, placing the scaled arrows tip-to-tail; where the walk ends is the combination. The coefficients may be any real numbers — a negative one means walking backwards, a fraction means walking part of the way — and the order of the steps does not matter. Familiar operations are special cases: the sum (all $c_i = 1$), the difference ($1$ and $-1$), the mean of $m$ vectors (all $c_i = 1/m$) and, more generally, a weighted average with non-negative weights summing to 1 (a *convex combination*, which always lands "between" the vectors). Even coordinates are coefficients: $(x_1, x_2) = x_1\mathbf{e}_1 + x_2\mathbf{e}_2$ with $\mathbf{e}_1 = (1, 0)$ and $\mathbf{e}_2 = (0, 1)$.

*Span.* The *span* of $\mathbf{v}_1, \dots, \mathbf{v}_k$, written $\operatorname{span}\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$, is the set of all their linear combinations — every point the walks can reach. It always contains $\mathbf{0}$ (take every $c_i = 0$), and adding or scaling vectors of the span never leads out of it, so the span is a flat object through the origin: the origin alone (if all the vectors are zero), a line, a plane, or a higher-dimensional flat called a *subspace*. In $\mathbb{R}^2$ the span of $\mathbf{u} = (1, 2)$ is the line $x_2 = 2x_1$, while the span of $\mathbf{u}$ and $\mathbf{v} = (3, -1)$ is the whole plane: in Figure 1(a) their integer combinations already form a skewed grid over the plane, and fractional coefficients fill in everything between the grid lines. Asking "is $\mathbf{b}$ in the span?" is asking "do the equations $c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k = \mathbf{b}$ have a solution?" — a linear system, the subject of [Module 02](../02-matrices-and-linear-maps/README.md).

*Linear independence.* The vectors are *linearly independent* if the only way to get

$$c_1 \mathbf{v}_1 + \dots + c_k \mathbf{v}_k = \mathbf{0}$$

is $c_1 = \dots = c_k = 0$ (the *trivial* combination), and *linearly dependent* otherwise. The test looks odd at first — why the zero vector? — but it is equivalent to two statements that are easier to feel:

- *No vector is redundant.* If a combination with some $c_j \ne 0$ gives $\mathbf{0}$, divide by $c_j$ and rearrange: $\mathbf{v}_j = -\sum_{i \ne j} (c_i/c_j)\,\mathbf{v}_i$, so $\mathbf{v}_j$ is a combination of the others and removing it does not shrink the span. Conversely, if $\mathbf{v}_j = \sum_{i \ne j} a_i\mathbf{v}_i$, moving $\mathbf{v}_j$ to the other side gives a combination equal to $\mathbf{0}$ whose coefficient on $\mathbf{v}_j$ is $-1 \ne 0$.
- *Coefficients are unique.* If $\mathbf{x} = \sum_i a_i\mathbf{v}_i = \sum_i b_i\mathbf{v}_i$, subtracting gives $\sum_i (a_i - b_i)\,\mathbf{v}_i = \mathbf{0}$, and independence forces every $a_i = b_i$. For independent vectors each point of the span has exactly one recipe; for dependent vectors each point of the span has infinitely many, because any multiple of a combination equal to $\mathbf{0}$ can be added to a recipe without changing the result.

Geometrically, independence means that each vector adds a genuinely new direction: $k$ independent vectors span a $k$-dimensional flat, while a dependent set spans less than its head count suggests. Dependence has a picture of its own — a non-trivial combination equal to $\mathbf{0}$ is a walk along scaled copies of the vectors that returns to its starting point, a closed loop. Quick consequences: a single vector is independent exactly when it is not $\mathbf{0}$; two vectors are dependent exactly when one is a multiple of the other, i.e. both lie on one line through the origin (Figure 1(b)); any set containing $\mathbf{0}$ is dependent, since $1\cdot\mathbf{0} = \mathbf{0}$; and more than $n$ vectors in $\mathbb{R}^n$ are always dependent — any three vectors in $\mathbb{R}^2$ are dependent, because there is no room for a third independent direction. In $\mathbb{R}^3$ (Figure 2), two independent vectors span a plane through the origin, and a third vector is independent of them exactly when it sticks out of that plane.

*Basis and dimension.* A set that is both independent and spanning — enough vectors to reach everything, none of them redundant — is a *basis*. Every basis of $\mathbb{R}^n$ has exactly $n$ vectors (that number is the *dimension*), and the unique coefficients of a vector in a basis are its *coordinates* in that basis; the usual coordinates are the coefficients in the standard basis $\mathbf{e}_1, \dots, \mathbf{e}_n$. Section 6 shows that coordinates are especially easy to read off in an orthonormal basis, and [Module 02](../02-matrices-and-linear-maps/README.md) (Section 8) shows how to switch between bases.

*Testing independence in practice.* By hand, set the combination equal to $\mathbf{0}$ and solve for the coefficients, as in the worked example below. Two vectors in $\mathbb{R}^2$ have a one-line test: $\mathbf{u}$ and $\mathbf{v}$ are independent exactly when $u_1 v_2 - u_2 v_1 \ne 0$ (the determinant of the matrix with columns $\mathbf{u}$ and $\mathbf{v}$). In code, stack the vectors as the columns of a matrix and compare its *rank* — the number of independent columns — with the number of vectors: `np.linalg.matrix_rank(np.column_stack([v1, v2, v3])) == 3`. Rank, determinants and the *null space*, which collects all the closed loops, are the subject of [Module 02](../02-matrices-and-linear-maps/README.md), Section 5. With measured data exact dependence is rare, but *near* dependence — one feature almost a combination of others — is common, and it makes regression coefficients unstable (multicollinearity, [Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md)).

![Two panels. Left: the vectors u = (1, 2) and v = (3, -1) over the skewed grid of their integer combinations; the path five-sevenths of u followed by ten-sevenths of v ends at w = (5, 0), and the triangle it encloses with w is shaded. Right: u and -2u lie on the single line x2 = 2 x1, and the point w = (5, 0) off that line cannot be reached.](figures/linear-combinations.svg)

**Figure 1.** (a) Independent vectors: every point of the plane is a combination $a\mathbf{u} + b\mathbf{v}$; the grey lines mark integer values of $a$ and $b$, so $\mathbf{u} + \mathbf{v}$ and $2\mathbf{u} - \mathbf{v}$ are grid points, and $\mathbf{w} = (5, 0)$ is reached by walking $\tfrac{5}{7}\mathbf{u}$ and then $\tfrac{10}{7}\mathbf{v}$. (b) A dependent pair: all combinations of $\mathbf{u}$ and $-2\mathbf{u}$ stay on one line, so their span is a line and $\mathbf{w}$ is out of reach.

![A three-dimensional plot. The shaded plane spanned by a1 = (1, 1, 0) and a2 = (1, 0, 1) contains the vector x = a1 + 2 a2 = (3, 1, 2), reached by the dashed path a1, a2, a2. The vector a3 = (0, 1, 1) points out of the plane; a dotted line drops from its tip to the plane.](figures/span-in-3d.svg)

**Figure 2.** In $\mathbb{R}^3$ two independent vectors $\mathbf{a}_1, \mathbf{a}_2$ span a plane. $\mathbf{x} = \mathbf{a}_1 + 2\mathbf{a}_2$ lies in it (dashed: the walk $\mathbf{a}_1$, $\mathbf{a}_2$, $\mathbf{a}_2$), so $\{\mathbf{a}_1, \mathbf{a}_2, \mathbf{x}\}$ is dependent. $\mathbf{a}_3$ sticks out of the plane (dotted: its distance to the plane), so $\{\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3\}$ is independent and spans all of $\mathbb{R}^3$.

**Worked example.** Let $\mathbf{u} = (1, 2)$ and $\mathbf{v} = (3, -1)$.

- $\mathbf{u} + \mathbf{v} = (4, 1)$, $\;2\mathbf{u} = (2, 4)$, $\;2\mathbf{u} - \mathbf{v} = (2 - 3, 4 + 1) = (-1, 5)$ — in Figure 1(a), the grid points with coefficients $(a, b) = (1, 1)$ and $(2, -1)$.
- Independence: suppose $a(1, 2) + b(3, -1) = (0, 0)$. The two components give $a + 3b = 0$ and $2a - b = 0$, so $b = 2a$ and then $a + 6a = 0$, hence $a = 0$ and $b = 0$. Independent — and since there are two of them in $\mathbb{R}^2$, they span the whole plane. The shortcut agrees: $u_1 v_2 - u_2 v_1 = 1\cdot(-1) - 2\cdot 3 = -7 \ne 0$.
- Expressing a third vector: to write $\mathbf{w} = (5, 0)$ as $a\mathbf{u} + b\mathbf{v}$, solve $a + 3b = 5$, $2a - b = 0$: again $b = 2a$, so $7a = 5$, giving $a = 5/7$, $b = 10/7$. Check: $\tfrac{5}{7}(1, 2) + \tfrac{10}{7}(3, -1) = (\tfrac{5 + 30}{7}, \tfrac{10 - 10}{7}) = (5, 0)$. Because $\mathbf{u}$ and $\mathbf{v}$ are independent, this is the *only* recipe for $\mathbf{w}$.
- So $\{\mathbf{u}, \mathbf{v}, \mathbf{w}\}$ is dependent, as it must be. Multiplying $\tfrac{5}{7}\mathbf{u} + \tfrac{10}{7}\mathbf{v} - \mathbf{w} = \mathbf{0}$ by 7 gives the non-trivial relation $5\mathbf{u} + 10\mathbf{v} - 7\mathbf{w} = \mathbf{0}$; check: $(5 + 30 - 35,\ 10 - 10 - 0) = (0, 0)$. In Figure 1(a) the relation is the shaded triangle: walk $\tfrac{5}{7}\mathbf{u}$, then $\tfrac{10}{7}\mathbf{v}$, then $-\mathbf{w}$, and you are back at the origin.
- A dependent pair: $\mathbf{u}$ and $-2\mathbf{u} = (-2, -4)$ satisfy $2\mathbf{u} + 1\cdot(-2\mathbf{u}) = \mathbf{0}$. Every combination $a\mathbf{u} + b(-2\mathbf{u}) = (a - 2b)\,\mathbf{u}$ lies on the line $x_2 = 2x_1$ (Figure 1(b)), so $\mathbf{w} = (5, 0)$ is out of reach: the components would need $a - 2b = 5$ and $2a - 4b = 0$, i.e. $2 \cdot 5 = 0$. The points they do reach have infinitely many recipes: $\mathbf{u} = 1\cdot\mathbf{u} + 0\cdot(-2\mathbf{u}) = 3\mathbf{u} + 1\cdot(-2\mathbf{u}) = \dots$
- In $\mathbb{R}^3$ (Figure 2): $\mathbf{a}_1 = (1, 1, 0)$ and $\mathbf{a}_2 = (1, 0, 1)$ are independent (neither is a multiple of the other) and span a plane. The vector $\mathbf{x} = (3, 1, 2) = \mathbf{a}_1 + 2\mathbf{a}_2$ lies in that plane, so $\{\mathbf{a}_1, \mathbf{a}_2, \mathbf{x}\}$ is dependent. $\mathbf{a}_3 = (0, 1, 1)$ does not: $s\mathbf{a}_1 + t\mathbf{a}_2 = (s + t, s, t) = (0, 1, 1)$ forces $s = t = 1$ from the last two components, and then the first is $2 \ne 0$. So $\{\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3\}$ is independent — three independent vectors in $\mathbb{R}^3$, hence a basis. Section 6 turns exactly these three vectors into an orthonormal basis.

**Real-life example (healthcare).** A clinic stores each patient visit as $\mathbf{p} = (\text{age}, \text{systolic BP}, \text{cholesterol}, \text{BMI})$. For one patient, $\mathbf{p}_{2024} = (54, 130, 210, 27.5)$ and $\mathbf{p}_{2025} = (55, 142, 198, 28.1)$. The change vector $\mathbf{p}_{2025} - \mathbf{p}_{2024} = (1, 12, -12, 0.6)$ says at a glance: blood pressure up, cholesterol down. The mean record of a cohort of $m$ patients, $\bar{\mathbf{p}} = \tfrac{1}{m}(\mathbf{p}_1 + \dots + \mathbf{p}_m)$, is a linear combination with all coefficients equal to $1/m$; a risk model that compares a patient to "the typical patient" is really looking at $\mathbf{p} - \bar{\mathbf{p}}$.

Dependence turns up in clinical data whenever one column is computed from others. Many laboratories do not measure LDL cholesterol directly but calculate it with the Friedewald equation $\text{LDL} = \text{TC} - \text{HDL} - \text{TG}/5$ (all in mg/dL; TC is total cholesterol, TG triglycerides): a patient with TC $= 220$, HDL $= 45$ and TG $= 175$ is reported with LDL $= 220 - 45 - 35 = 140$. Collect each feature over all patients into one column vector, and the LDL column is *exactly* the combination $1\cdot\text{TC} - 1\cdot\text{HDL} - 0.2\cdot\text{TG}$ of the other three — the four columns are linearly dependent. For a linear risk score $w_1\,\text{TC} + w_2\,\text{HDL} + w_3\,\text{TG} + w_4\,\text{LDL}$ the weights are then not unique: since $\text{LDL} - \text{TC} + \text{HDL} + 0.2\,\text{TG} = 0$ for every patient, adding $t$ times it to the score shows that $(w_1 - t,\ w_2 + t,\ w_3 + 0.2t,\ w_4 + t)$ gives identical scores for every $t$. "The effect of LDL with the other lipids held fixed" has no answer in such data, because LDL cannot change while the other three stay fixed. [Module 02](../02-matrices-and-linear-maps/README.md), Section 5 meets the same problem as a rank-deficient design matrix.

**In ML/AI this is …** the design matrix `X` of shape `(n_samples, n_features)` whose rows are the feature vectors; the lookup table behind `nn.Embedding`, which maps a token id to a row vector; the word-vector arithmetic $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$ of word2vec; and the *residual stream* of a transformer, where every layer literally adds its output vector to the running representation. Linear combinations are everywhere downstream: a linear layer's output $W\mathbf{x} = x_1\mathbf{w}_1 + \dots + x_n\mathbf{w}_n$ is a combination of the columns $\mathbf{w}_j$ of $W$ with the input entries as coefficients, so it can only produce vectors in their span ([Module 02](../02-matrices-and-linear-maps/README.md)), and an attention head outputs a convex combination of value vectors with the softmax weights as coefficients ([Module 17](../../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)). Independence is about *identifiability*: when feature columns are dependent — the LDL example above, or the one-hot "dummy-variable trap" of Module 02 — a linear model's weights cannot be pinned down, and when they are nearly dependent the weights become unstable. It also caps capacity: $\mathbb{R}^d$ holds at most $d$ independent directions, although (Section 8) it holds vastly more *nearly orthogonal* ones, which is how a 768-dimensional model can represent far more than 768 features.

## 2. Norms and distance

**Intuition.** A norm is a ruler: it assigns a length to a vector. There is more than one sensible ruler. "As the crow flies" is the Euclidean ($L_2$) norm. "Number of city blocks a taxi drives" is the $L_1$ norm. "The single largest component in absolute value" is the $L_\infty$ norm — the right one when the worst coordinate is what matters, e.g. the maximum deviation of any sensor. Once you have a length you have a distance: the length of the difference.

**Definition.** A norm is a function $\|\cdot\| : \mathbb{R}^n \to [0, \infty)$ with (i) $\|\mathbf{x}\| = 0$ only for $\mathbf{x} = \mathbf{0}$, (ii) $\|c\mathbf{x}\| = |c|\,\|\mathbf{x}\|$, and (iii) the triangle inequality $\|\mathbf{x} + \mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|$. The family you will use:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}, \qquad \|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|, \qquad \|\mathbf{x}\|_\infty = \max_i |x_i|, \qquad \|\mathbf{x}\|_p = \Big(\sum_{i=1}^n |x_i|^p\Big)^{1/p} \;(p \ge 1).$$

$L_2$ and $L_\infty$ are the cases $p = 2$ and $p \to \infty$ of the general $L_p$ norm. A *unit vector* has norm 1; *normalising* $\mathbf{x} \ne \mathbf{0}$ means forming $\hat{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|$, which keeps the direction and discards the length. The *distance* induced by a norm is $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$. The norms are never far apart — for every $\mathbf{x} \in \mathbb{R}^n$,

$$\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt{n}\,\|\mathbf{x}\|_2 \le n\,\|\mathbf{x}\|_\infty .$$

The *unit ball* $\{\mathbf{x} : \|\mathbf{x}\| \le 1\}$ shows the personality of each norm: in two dimensions the $L_2$ ball is a disc, the $L_1$ ball is a diamond with corners on the axes, and the $L_\infty$ ball is a square. Those corners are why an $L_1$ penalty pushes coefficients to *exactly* zero (the lasso, [Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md)), while the round $L_2$ ball merely shrinks them.

![Two panels. Left: the unit balls of three norms drawn together: a diamond for L1 with corners on the axes, a circle for L2 and a square for L-infinity. Right: a street grid with a depot at the origin and a customer at (3, 4); the straight line has length 5 and a staircase route along the streets has length 7.](figures/norms.svg)

**Figure 3.** (a) The three unit balls. The $L_1$ diamond's corners lie on the axes, at points where one coordinate is exactly zero — the geometric source of the lasso's sparsity. (b) The courier of the real-life example below: the straight line to the customer at $(3, 4)$ has $L_2$ length 5, while every route along the street grid with 3 blocks east and 4 north has $L_1$ length 7.

**Worked example.** Take $\mathbf{x} = (3, -4, 12)$.

- $\|\mathbf{x}\|_2 = \sqrt{9 + 16 + 144} = \sqrt{169} = 13$; $\;\|\mathbf{x}\|_1 = 3 + 4 + 12 = 19$; $\;\|\mathbf{x}\|_\infty = 12$. Check the chain: $12 \le 13 \le 19 \le \sqrt{3}\cdot 13 \approx 22.5 \le 36$.
- Unit vector: $\hat{\mathbf{x}} = (3/13, -4/13, 12/13)$, and indeed $(9 + 16 + 144)/169 = 1$.
- Distance between $\mathbf{a} = (1, 2, 3)$ and $\mathbf{b} = (4, 6, 3)$: the difference is $(3, 4, 0)$, so the Euclidean distance is $5$, the $L_1$ distance is $7$ and the $L_\infty$ distance is $4$.

**Real-life example (logistics, then nutrition).** A courier depot is at the origin of a city grid and a customer is 3 blocks east and 4 blocks north. The straight-line distance is $\|(3, 4)\|_2 = 5$ blocks, but the van drives $\|(3, 4)\|_1 = 7$ blocks; a routing model that uses $L_2$ for a grid city systematically underestimates travel time by up to a factor of $\sqrt{2}$. A different ruler problem appears in a nutrition app that represents foods per 100 g as (protein, fat, carbohydrate): chicken breast $(31, 3.6, 0)$, salmon $(20, 13, 0)$, cooked lentils $(9, 0.4, 20)$. Euclidean distances are $14.5$ (chicken–salmon), $29.9$ (chicken–lentils) and $26.1$ (salmon–lentils), so the app correctly groups the two animal proteins — but note that these distances depend on the units; had fat been measured in kilojoules instead of grams it would have dominated every comparison. Scaling features before measuring distance is covered in the pitfalls section.

**In ML/AI this is …** the mean-squared-error loss, which is $\tfrac{1}{n}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$; weight decay, which pulls $\mathbf{w}$ towards zero at every step — for plain gradient descent the same as adding $\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$ to the loss (the `weight_decay` argument of PyTorch's `SGD`), while `AdamW` applies it as a separate, "decoupled" shrinkage; the lasso penalty $\lambda\|\mathbf{w}\|_1$; gradient clipping (`clip_grad_norm_`), which rescales the gradient when its $L_2$ norm exceeds a threshold; $k$-nearest neighbours and $k$-means, which live on Euclidean distance; and LayerNorm/RMSNorm, which divide an activation vector by (a version of) its norm.

## 3. The inner (dot) product

**Intuition.** Multiply matching components and add up. That single number carries two readings at once. Geometrically it measures *alignment*: large and positive when two arrows point the same way, zero when they are perpendicular, negative when they oppose. Algebraically it is a *weighted sum*: if one vector holds quantities and the other holds prices, or features and weights, the dot product is a total. The whole of linear modelling rests on the fact that these two readings are the same number.

**Definition.** For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$,

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^\top \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\|_2 \,\|\mathbf{v}\|_2 \cos\theta,$$

where $\theta \in [0, \pi]$ is the angle between the two vectors. The dot product is symmetric, linear in each argument, and $\mathbf{u} \cdot \mathbf{u} = \|\mathbf{u}\|_2^2$. Two vectors are *orthogonal* when $\mathbf{u} \cdot \mathbf{v} = 0$; the zero vector is orthogonal to everything.

*Why the two definitions agree* (law of cosines). The arrows $\mathbf{u}$, $\mathbf{v}$ and $\mathbf{u} - \mathbf{v}$ form a triangle whose angle between the sides $\mathbf{u}$ and $\mathbf{v}$ is $\theta$, so $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$. Expanding the left side algebraically, $\|\mathbf{u} - \mathbf{v}\|^2 = (\mathbf{u} - \mathbf{v})\cdot(\mathbf{u} - \mathbf{v}) = \|\mathbf{u}\|^2 - 2\,\mathbf{u}\cdot\mathbf{v} + \|\mathbf{v}\|^2$. Comparing the two gives $\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$. The lesson notebook verifies this identity numerically on random vectors.

Two inequalities follow immediately and are used everywhere:

- **Cauchy–Schwarz:** $|\mathbf{u} \cdot \mathbf{v}| \le \|\mathbf{u}\|\,\|\mathbf{v}\|$, with equality only when one vector is a multiple of the other. One-line reason: $|\cos\theta| \le 1$. (A proof that does not assume the geometry: $\|\mathbf{u} - t\mathbf{v}\|^2 \ge 0$ for all $t$ is a quadratic in $t$ with no two real roots, so its discriminant $4(\mathbf{u}\cdot\mathbf{v})^2 - 4\|\mathbf{u}\|^2\|\mathbf{v}\|^2$ is $\le 0$.)
- **Triangle inequality:** $\|\mathbf{u} + \mathbf{v}\| \le \|\mathbf{u}\| + \|\mathbf{v}\|$. Square both sides: the left is $\|\mathbf{u}\|^2 + 2\,\mathbf{u}\cdot\mathbf{v} + \|\mathbf{v}\|^2$, and Cauchy–Schwarz bounds the middle term by $2\|\mathbf{u}\|\|\mathbf{v}\|$.

**Worked example.** $\mathbf{u} = (3, 4)$, $\mathbf{v} = (4, 3)$: $\mathbf{u}\cdot\mathbf{v} = 12 + 12 = 24$, $\|\mathbf{u}\| = \|\mathbf{v}\| = 5$, so $\cos\theta = 24/25 = 0.96$ and $\theta = \arccos 0.96 \approx 16.3^\circ$. Law-of-cosines check: $\mathbf{u} - \mathbf{v} = (-1, 1)$ has squared length $2$, and $25 + 25 - 2\cdot 25 \cdot 0.96 = 50 - 48 = 2$. Orthogonality: $(3, 4)\cdot(-4, 3) = -12 + 12 = 0$, so $(-4, 3)$ is perpendicular to $(3, 4)$ — rotating any 2-D vector $(a, b)$ by $90^\circ$ anticlockwise gives $(-b, a)$. Cauchy–Schwarz: $|24| \le 25$.

**Real-life example (finance).** A portfolio holds $\mathbf{h} = (10, 5, 20)$ shares of three stocks priced at $\mathbf{p} = (120, 300, 45)$ EUR. Its value is the dot product $\mathbf{h}\cdot\mathbf{p} = 1200 + 1500 + 900 = 3600$ EUR. Today's returns are $\mathbf{r} = (+1\%, -2\%, +0.5\%)$ and the value weights are $\mathbf{w} = (1200, 1500, 900)/3600 = (0.333, 0.417, 0.250)$; the portfolio return is again a dot product, $\mathbf{w}\cdot\mathbf{r} = 0.333 - 0.833 + 0.125 = -0.375\%$. Insurers compute risk scores the same way: a weight per risk factor, a dot product with the applicant's factor vector, a threshold.

**In ML/AI this is …** the pre-activation of every neuron, $z = \mathbf{w}\cdot\mathbf{x} + b$; the `decision_function` of `sklearn.linear_model.LogisticRegression`; the attention score $\mathbf{q}_i \cdot \mathbf{k}_j$ between a query and a key (divided by $\sqrt{d_k}$, see Section 8); and matrix multiplication itself, where entry $(i, j)$ of $AB$ is the dot product of row $i$ of $A$ with column $j$ of $B$ — which is why a whole layer's worth of dot products is one `X @ W`.

## 4. Cosine similarity and embeddings

**Intuition.** If two documents are about the same topic, their word-count vectors point in the same direction — but a 3,000-word article has a much longer vector than a 300-word summary of it. The dot product rewards the long document for being long; Euclidean distance punishes it. Cosine similarity looks only at the angle, so it compares *what the documents are about* rather than *how much they say*.

**Definition.**

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|_2\,\|\mathbf{v}\|_2} \in [-1, 1].$$

It is the dot product of the normalised vectors $\hat{\mathbf{u}}\cdot\hat{\mathbf{v}}$. It is *scale-invariant*: $\cos(a\mathbf{u}, b\mathbf{v}) = \cos(\mathbf{u}, \mathbf{v})$ for any $a, b > 0$. For non-negative vectors such as counts it lies in $[0, 1]$. *Cosine distance* is $1 - \cos(\mathbf{u}, \mathbf{v})$; it is related to Euclidean distance between the normalised vectors by $\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|^2 = 2 - 2\cos(\mathbf{u}, \mathbf{v})$, so after normalisation the two rankings coincide. For mean-centred vectors, cosine similarity is exactly the Pearson correlation coefficient.

*Nearest-neighbour search.* Stack $N$ embedding vectors as rows of a matrix $E \in \mathbb{R}^{N \times d}$ and normalise each row. For a normalised query $\hat{\mathbf{q}}$, the vector of all $N$ cosine similarities is the single matrix–vector product $E\hat{\mathbf{q}}$; the top-$k$ results are its $k$ largest entries. This is what a "vector database" computes, exactly for small $N$ and approximately for billions.

**Worked example.** Vocabulary $(\text{model}, \text{data}, \text{goal})$ and three documents: $\mathbf{d}_1 = (2, 1, 0)$ (a short note about ML), $\mathbf{d}_2 = (8, 4, 0)$ (a long article about ML), $\mathbf{d}_3 = (0, 1, 3)$ (a football report).

- $\cos(\mathbf{d}_1, \mathbf{d}_2) = \dfrac{16 + 4}{\sqrt{5}\sqrt{80}} = \dfrac{20}{\sqrt{400}} = 1$ — identical direction; $\mathbf{d}_2 = 4\mathbf{d}_1$.
- $\cos(\mathbf{d}_1, \mathbf{d}_3) = \dfrac{0 + 1 + 0}{\sqrt{5}\sqrt{10}} = \dfrac{1}{\sqrt{50}} \approx 0.141$.
- Euclidean distances: $\|\mathbf{d}_1 - \mathbf{d}_2\| = \|(-6, -3, 0)\| = \sqrt{45} \approx 6.71$ but $\|\mathbf{d}_1 - \mathbf{d}_3\| = \|(2, 0, -3)\| = \sqrt{13} \approx 3.61$.

Euclidean distance declares the football report the *closer* neighbour of the ML note, purely because the long ML article is long. Cosine similarity gets it right.

![The three document vectors drawn to scale in the plane that contains them. d1 and d2 = 4 d1 point along the same horizontal direction, d2 four times as long; d3 points almost straight up at about 82 degrees. Dashed segments show the Euclidean distances: about 6.71 from d1 to d2 and 3.61 from d1 to d3.](figures/cosine-vs-euclidean.svg)

**Figure 4.** The three documents drawn true to scale in the plane that contains them (the origin, $\mathbf{d}_1$ and $\mathbf{d}_2 = 4\mathbf{d}_1$ lie on one line, so with $\mathbf{d}_3$ the four points lie in a single plane). The football report is the nearer neighbour of $\mathbf{d}_1$ by Euclidean distance, yet it points almost $82^\circ$ away; the long article points in exactly the same direction.

**Real-life example (e-commerce support).** An online retailer embeds every incoming support ticket and compares it with the last 90 days of tickets. A query "parcel not delivered, tracking shows delivered" with cosine similarity above 0.95 to an existing ticket is flagged as a probable duplicate and merged; similarities between 0.80 and 0.95 are shown to the agent as "related tickets" with their resolutions. The same geometry runs the recommender: users and products are embedded in one space, and "products for you" is the top-$k$ list by dot product or cosine with the user's vector. Ticket length no longer matters because direction, not magnitude, is compared (spelling variants are handled by the embedding model itself, which maps them to nearby vectors).

**In ML/AI this is …** the scoring function of semantic search and RAG retrieval (FAISS `IndexFlatIP` on normalised vectors, the `<=>` cosine-distance operator in pgvector); the image–text match in CLIP, which is a cosine similarity multiplied by a learned temperature; the similarity inside contrastive losses such as InfoNCE; `sklearn.metrics.pairwise.cosine_similarity`; and the word-analogy test for word2vec and GloVe embeddings.

## 5. Projections

**Intuition.** Hold an arrow $\mathbf{a}$ above a line in the direction $\mathbf{b}$ and shine a light straight down onto the line. The shadow is the projection: the part of $\mathbf{a}$ that lies *along* $\mathbf{b}$. What is left over — $\mathbf{a}$ minus its shadow — sticks out perpendicularly. Every "how much of this is explained by that direction?" question is a projection, and the perpendicular left-over is the "unexplained" residual.

**Definition.** For $\mathbf{b} \ne \mathbf{0}$ with unit vector $\hat{\mathbf{b}} = \mathbf{b}/\|\mathbf{b}\|$:

$$\text{scalar projection: } \; \mathbf{a}\cdot\hat{\mathbf{b}} = \|\mathbf{a}\|\cos\theta, \qquad \text{vector projection: } \; \mathbf{p} = \operatorname{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b} = (\mathbf{a}\cdot\hat{\mathbf{b}})\,\hat{\mathbf{b}}.$$

The *residual* $\mathbf{r} = \mathbf{a} - \mathbf{p}$ is orthogonal to $\mathbf{b}$: $\mathbf{r}\cdot\mathbf{b} = \mathbf{a}\cdot\mathbf{b} - \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}(\mathbf{b}\cdot\mathbf{b}) = 0$. Consequently Pythagoras holds, $\|\mathbf{a}\|^2 = \|\mathbf{p}\|^2 + \|\mathbf{r}\|^2$, and the fraction of $\mathbf{a}$'s squared length "explained" by direction $\mathbf{b}$ is $\|\mathbf{p}\|^2/\|\mathbf{a}\|^2 = \cos^2\theta$.

*Projection is the best approximation.* Among all multiples $t\mathbf{b}$, the one closest to $\mathbf{a}$ is the projection. The squared error $f(t) = \|\mathbf{a} - t\mathbf{b}\|^2 = \|\mathbf{a}\|^2 - 2t\,\mathbf{a}\cdot\mathbf{b} + t^2\|\mathbf{b}\|^2$ is a parabola in $t$ with minimum where $f'(t) = -2\,\mathbf{a}\cdot\mathbf{b} + 2t\|\mathbf{b}\|^2 = 0$, i.e. $t^\star = \mathbf{a}\cdot\mathbf{b}/\mathbf{b}\cdot\mathbf{b}$. Replace the line by a subspace spanned by several columns and the same argument gives least squares ([Module 02](../02-matrices-and-linear-maps/README.md)). In matrix form the projection onto $\hat{\mathbf{b}}$ is $\mathbf{p} = (\hat{\mathbf{b}}\hat{\mathbf{b}}^\top)\mathbf{a}$ — a rank-one matrix applied to $\mathbf{a}$.

**Worked example.** $\mathbf{a} = (2, 3)$, $\mathbf{b} = (4, 2)$. Then $\mathbf{a}\cdot\mathbf{b} = 8 + 6 = 14$ and $\mathbf{b}\cdot\mathbf{b} = 20$, so $t^\star = 0.7$ and $\mathbf{p} = 0.7\,(4, 2) = (2.8, 1.4)$. Residual $\mathbf{r} = (2 - 2.8, 3 - 1.4) = (-0.8, 1.6)$; check $\mathbf{r}\cdot\mathbf{b} = -3.2 + 3.2 = 0$. Pythagoras: $\|\mathbf{p}\|^2 = 7.84 + 1.96 = 9.8$, $\|\mathbf{r}\|^2 = 0.64 + 2.56 = 3.2$, sum $13 = 4 + 9 = \|\mathbf{a}\|^2$. The scalar projection is $14/\sqrt{20} \approx 3.13$, and direction $\mathbf{b}$ explains $9.8/13 \approx 75\%$ of $\mathbf{a}$'s squared length.

![Two panels. Left: vectors a = (2, 3) and b = (4, 2) from the origin; the projection p = (2.8, 1.4) lies along b, and the residual r = (-0.8, 1.6) runs from p to the tip of a at a right angle to b. Right: the squared distance 13 - 28t + 20t squared as a function of t, a parabola with its minimum 3.2 at t = 0.7.](figures/projection.svg)

**Figure 5.** (a) The worked example: the projection $\mathbf{p} = 0.7\,\mathbf{b}$ is the foot of the perpendicular from $\mathbf{a}$ to the line through $\mathbf{b}$ (dotted), and the residual $\mathbf{r}$ meets that line at a right angle. (b) The squared distance from $\mathbf{a}$ to the multiple $t\mathbf{b}$ is a parabola in $t$ with its minimum at $t^\star = 0.7$, where it equals $\|\mathbf{r}\|^2 = 3.2$: the projection is the best approximation of $\mathbf{a}$ on the line.

**Real-life example (physics/engineering).** A 10 kg sled rests on a $30^\circ$ slope. Gravity is $\mathbf{F} = (0, -98.1)$ N. The downhill direction is the unit vector $\mathbf{d} = (\cos 30^\circ, -\sin 30^\circ) = (0.866, -0.5)$. The component of gravity pulling the sled down the slope is the scalar projection $\mathbf{F}\cdot\mathbf{d} = 0 + 49.05 = 49.05$ N (this is $mg\sin\theta$), and the residual $\mathbf{F} - 49.05\,\mathbf{d} = (-42.5, -73.6)$ N, of magnitude $84.96$ N $= mg\cos\theta$, presses the sled into the slope and is balanced by the normal force. The same decomposition answers a marketing question — project a customer's spending vector onto the "budget-shopper" direction to see how much of their behaviour that single trait explains, and look at the residual to see what it does not — and a data-cleaning one: subtracting the projection onto a known nuisance direction (a seasonal trend, a batch effect) removes exactly that component and nothing else.

**In ML/AI this is …** ordinary least squares, which projects the target vector onto the span of the feature columns ([Module 02](../02-matrices-and-linear-maps/README.md)); principal-component scores, which are projections onto the leading eigenvectors ([Module 03](../03-eigenvalues-svd-and-pca/README.md)); the "debiasing" of word embeddings by projecting out a gender direction (Bolukbasi et al., 2016); and the linear probes and steering vectors of LLM interpretability, which read a concept out of a hidden state by a dot product with a learned direction and write it in by adding a multiple of that direction.

## 6. Orthonormal bases and coordinates

**Intuition.** A coordinate system is a choice of reference directions; the same arrow gets different numbers in different systems. If the reference directions are mutually perpendicular and of unit length — *orthonormal* — then finding coordinates needs no equation solving: each coordinate is one dot product, and lengths and angles come out identical in the new system. Gram–Schmidt is the procedure that straightens any set of independent directions into such a system, by repeatedly subtracting projections.

**Definition.** Vectors $\mathbf{q}_1, \dots, \mathbf{q}_n \in \mathbb{R}^n$ form an *orthonormal basis* if $\mathbf{q}_i\cdot\mathbf{q}_j = 1$ when $i = j$ and $0$ otherwise. The standard basis $\mathbf{e}_1 = (1, 0, \dots, 0), \dots, \mathbf{e}_n$ is one example. For any $\mathbf{x}$,

$$\mathbf{x} = \sum_{i=1}^n c_i\,\mathbf{q}_i \quad\text{with}\quad c_i = \mathbf{x}\cdot\mathbf{q}_i, \qquad\text{and}\qquad \|\mathbf{x}\|^2 = \sum_{i=1}^n c_i^2 \;\;\text{(Parseval)}.$$

The coordinate formula follows by dotting both sides of the expansion with $\mathbf{q}_i$: every other term dies. Parseval follows by expanding $\mathbf{x}\cdot\mathbf{x}$ the same way. Collecting the $\mathbf{q}_i$ as columns of a matrix $Q$, orthonormality reads $Q^\top Q = I$, the coordinates are $Q^\top\mathbf{x}$, and $\|Q^\top\mathbf{x}\| = \|\mathbf{x}\|$: an orthonormal change of coordinates is a rigid rotation (possibly with a reflection) and changes no length and no angle.

*Gram–Schmidt.* Given independent $\mathbf{a}_1, \dots, \mathbf{a}_k$: set $\mathbf{q}_1 = \mathbf{a}_1/\|\mathbf{a}_1\|$; then for each $j = 2, \dots, k$ subtract from $\mathbf{a}_j$ its projections onto all previous $\mathbf{q}_i$, $\mathbf{v}_j = \mathbf{a}_j - \sum_{i<j}(\mathbf{a}_j\cdot\mathbf{q}_i)\,\mathbf{q}_i$, and normalise, $\mathbf{q}_j = \mathbf{v}_j/\|\mathbf{v}_j\|$. Each $\mathbf{v}_j$ is a residual in the sense of Section 5, hence orthogonal to everything before it. (`np.linalg.qr` does the same job with a numerically more stable algorithm; its columns can differ from Gram–Schmidt's by a sign.)

**Worked example.** $\mathbf{a}_1 = (1, 1, 0)$, $\mathbf{a}_2 = (1, 0, 1)$, $\mathbf{a}_3 = (0, 1, 1)$.

- $\mathbf{q}_1 = (1, 1, 0)/\sqrt{2}$.
- $\mathbf{a}_2\cdot\mathbf{q}_1 = 1/\sqrt{2}$, so $\mathbf{v}_2 = (1, 0, 1) - \tfrac{1}{\sqrt 2}\cdot\tfrac{(1, 1, 0)}{\sqrt 2} = (1, 0, 1) - (\tfrac12, \tfrac12, 0) = (\tfrac12, -\tfrac12, 1)$, with $\|\mathbf{v}_2\| = \sqrt{\tfrac14 + \tfrac14 + 1} = \sqrt{3/2}$, giving $\mathbf{q}_2 = (1, -1, 2)/\sqrt{6}$.
- $\mathbf{a}_3\cdot\mathbf{q}_1 = 1/\sqrt 2$ and $\mathbf{a}_3\cdot\mathbf{q}_2 = (0 - 1 + 2)/\sqrt 6 = 1/\sqrt 6$, so $\mathbf{v}_3 = (0, 1, 1) - (\tfrac12, \tfrac12, 0) - (\tfrac16, -\tfrac16, \tfrac26) = (-\tfrac23, \tfrac23, \tfrac23)$, with $\|\mathbf{v}_3\| = 2/\sqrt 3$, giving $\mathbf{q}_3 = (-1, 1, 1)/\sqrt 3$.
- Checks: $\mathbf{q}_1\cdot\mathbf{q}_2 = (1 - 1 + 0)/\sqrt{12} = 0$, $\mathbf{q}_2\cdot\mathbf{q}_3 = (-1 - 1 + 2)/\sqrt{18} = 0$, $\mathbf{q}_1\cdot\mathbf{q}_3 = 0$.
- Coordinates of $\mathbf{x} = (3, 1, 2)$: $c_1 = 4/\sqrt 2 \approx 2.828$, $c_2 = (3 - 1 + 4)/\sqrt 6 \approx 2.449$, $c_3 = (-3 + 1 + 2)/\sqrt 3 = 0$. Parseval: $8 + 6 + 0 = 14 = 9 + 1 + 4$. The zero third coordinate tells you $\mathbf{x}$ lies in the plane spanned by $\mathbf{a}_1$ and $\mathbf{a}_2$ — indeed $\mathbf{x} = \mathbf{a}_1 + 2\mathbf{a}_2$.

**Real-life example (computer vision / colour).** A camera pixel is a vector $(R, G, B)$, say $(200, 150, 100)$. In the orthonormal basis $\mathbf{q}_1 = (1, 1, 1)/\sqrt 3$ (brightness), $\mathbf{q}_2 = (1, 0, -1)/\sqrt 2$ (red-versus-blue) and $\mathbf{q}_3 = (1, -2, 1)/\sqrt 6$ (green-versus-magenta), the coordinates are $c_1 = 450/\sqrt 3 \approx 259.8$, $c_2 = 100/\sqrt 2 \approx 70.7$ and $c_3 = (200 - 300 + 100)/\sqrt 6 = 0$. Parseval: $67{,}500 + 5{,}000 + 0 = 72{,}500 = 200^2 + 150^2 + 100^2$. Almost all of this pixel's energy sits in the brightness coordinate, as it does on average for natural images, whose R, G and B channels are strongly correlated. That, together with the eye's much lower acuity for colour detail, is why JPEG converts to a luma/chroma space and then compresses the two chroma channels far more aggressively than luma. Its YCbCr transform is a cousin of this basis rather than a copy: the two chroma rows span exactly the plane of $\mathbf{q}_2$ and $\mathbf{q}_3$, but the luma row $(0.299, 0.587, 0.114)$ is a perceptually weighted brightness, and the rows are neither unit length nor mutually orthogonal. Audio engineers do the same with sines and cosines: the discrete Fourier transform is a change to an orthogonal basis, and each frequency coefficient is a dot product of the signal with one basis waveform.

**In ML/AI this is …** PCA, which is nothing but a data-chosen orthonormal basis ([Module 03](../03-eigenvalues-svd-and-pca/README.md)); the matrices $U$ and $V$ of the singular value decomposition; orthogonal weight initialisation (`torch.nn.init.orthogonal_`), which preserves the norm of signals passing through a layer; rotary positional embeddings, which rotate pairs of coordinates in orthonormal 2-D planes ([Module 17](../../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)); and whitening transforms in preprocessing.

## 7. Hyperplanes and half-spaces

**Intuition.** In two dimensions a linear classifier draws a line; in three it places a sheet of paper; in $n$ dimensions it uses a flat $(n - 1)$-dimensional object called a hyperplane. The hyperplane is completely described by one perpendicular arrow $\mathbf{w}$ (the *normal*) and one offset $b$. Everything on one side gets one label, everything on the other side the other label, and how far a point sits from the sheet is a natural measure of how confident the classifier should be.

**Definition.** For $\mathbf{w} \in \mathbb{R}^n$, $\mathbf{w} \ne \mathbf{0}$, and $b \in \mathbb{R}$, the hyperplane is

$$H = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{w}\cdot\mathbf{x} + b = 0\}.$$

$\mathbf{w}$ is normal to $H$: if $\mathbf{x}, \mathbf{y} \in H$ then $\mathbf{w}\cdot(\mathbf{x} - \mathbf{y}) = -b + b = 0$, so $\mathbf{w}$ is perpendicular to every direction lying in $H$. The two *half-spaces* are $\{\mathbf{w}\cdot\mathbf{x} + b > 0\}$ (the side $\mathbf{w}$ points to) and $\{\mathbf{w}\cdot\mathbf{x} + b < 0\}$. The *signed distance* from a point to $H$ is

$$\delta(\mathbf{x}) = \frac{\mathbf{w}\cdot\mathbf{x} + b}{\|\mathbf{w}\|},$$

positive on the $\mathbf{w}$ side; in particular the distance from the origin is $|b|/\|\mathbf{w}\|$. Derivation: write $\mathbf{x} = \mathbf{x}_0 + t\,\hat{\mathbf{w}}$ with $\mathbf{x}_0 \in H$ the foot of the perpendicular; then $\mathbf{w}\cdot\mathbf{x} + b = (\mathbf{w}\cdot\mathbf{x}_0 + b) + t\|\mathbf{w}\| = t\|\mathbf{w}\|$, so $t = \delta(\mathbf{x})$. Note that $(c\mathbf{w}, cb)$ for $c > 0$ describes the *same* hyperplane with raw scores $c$ times larger — the signed distance is invariant, the raw score is not. A linear classifier predicts $\hat{y} = \operatorname{sign}(\mathbf{w}\cdot\mathbf{x} + b)$, and the *margin* of a labelled point $(\mathbf{x}, y)$ with $y \in \{-1, +1\}$ is $y\,\delta(\mathbf{x})$: positive if correctly classified, and larger the farther it is from the boundary. A support vector machine chooses $\mathbf{w}, b$ to maximise the smallest margin over the training set.

**Worked example.** $\mathbf{w} = (3, 4)$, $b = -12$: the line $3x + 4y = 12$ through $(4, 0)$ and $(0, 3)$, with $\|\mathbf{w}\| = 5$.

- Point $(4, 3)$: $\mathbf{w}\cdot\mathbf{x} + b = 12 + 12 - 12 = 12$, signed distance $12/5 = 2.4$, on the positive side.
- Origin: $-12/5 = -2.4$, on the negative side.
- Point $(2, 1.5)$: $6 + 6 - 12 = 0$ — on the line.
- Doubling to $\mathbf{w} = (6, 8)$, $b = -24$ gives raw score $24$ at $(4, 3)$ but the same distance $24/10 = 2.4$.

![The line 3 x1 + 4 x2 = 12 divides the plane into a shaded positive side, upper right, and a shaded negative side, lower left. The normal vector w = (3, 4) starts on the line and points into the positive side at a right angle. Dashed perpendiculars run from the point (4, 3) to the line, signed distance +2.4, and from the origin to the line, signed distance -2.4; the point (2, 1.5) lies on the line.](figures/hyperplane.svg)

**Figure 6.** The worked example. The normal $\mathbf{w}$ is perpendicular to the boundary, not along it (pitfall 4), and points into the half-space where $\mathbf{w}\cdot\mathbf{x} + b > 0$. Each dashed segment is a signed distance $\delta(\mathbf{x})$, measured along $\mathbf{w}$: positive on the side $\mathbf{w}$ points to, negative on the other.

**Real-life example (finance: credit approval).** A lender scores applicants on $\mathbf{x} = (\text{income in thousand EUR},\ \text{debt-to-income ratio in percent})$ with the rule "approve if $0.05\,\text{income} - 0.1\,\text{DTI} - 1 \ge 0$", i.e. $\mathbf{w} = (0.05, -0.1)$, $b = -1$, $\|\mathbf{w}\| = \sqrt{0.0025 + 0.01} \approx 0.1118$. Applicant A $(80, 20)$: score $4 - 2 - 1 = 1$, distance $1/0.1118 \approx 8.9$ — a clear approval. Applicant B $(40, 30)$: score $2 - 3 - 1 = -2$, distance $-17.9$ — a clear decline. Applicant C $(55, 22)$: score $2.75 - 2.2 - 1 = -0.45$, distance $-4.0$ — a decline, but close enough to the boundary that a human review or a request for more documents is sensible. The boundary itself passes through $(60, 20)$: an applicant with 60k income and 20% DTI is exactly on the fence. One caveat the geometry makes visible: those distances mix "thousand EUR" and "percent", so they are only meaningful after the features have been put on comparable scales.

**In ML/AI this is …** the decision boundary of logistic regression, the perceptron and the linear SVM (`sklearn.svm.LinearSVC`); the final linear layer of a neural classifier, whose $K$ rows are $K$ hyperplanes in the penultimate feature space with softmax picking the largest signed score; every ReLU unit, which asks "which side of the hyperplane $\mathbf{w}\cdot\mathbf{x} + b = 0$ am I on?" and thereby cuts the input space into polytopes; and linear probes, which are hyperplanes fitted in a network's activation space.

## 8. High-dimensional geometry

**Intuition.** Everything above is true in any dimension, but the *typical* behaviour of points changes dramatically as $d$ grows. Four surprises: (1) pairwise distances between random points become nearly all the same; (2) random directions are nearly orthogonal; (3) almost all of a cube's volume is in its corners, far from the inscribed ball; (4) almost all of a ball's volume is in a thin shell near its surface. None of these can be drawn in three dimensions, which is why they feel paradoxical. All of them follow from one principle: a sum of $d$ independent random terms has a mean that grows like $d$ and a standard deviation that grows only like $\sqrt{d}$, so relative fluctuations shrink like $1/\sqrt{d}$ (the law of large numbers, [Module 08](../../part-3-probability/08-distributions-and-limit-theorems/README.md)).

**Statements.**

- *Distance concentration.* For $\mathbf{x}, \mathbf{y}$ with independent coordinates uniform on $[0, 1]$, $\|\mathbf{x} - \mathbf{y}\|^2 = \sum_i (x_i - y_i)^2$ is a sum of $d$ independent terms with mean $1/6$ and variance $7/180$. So $\mathbb{E}\|\mathbf{x} - \mathbf{y}\|^2 = d/6$, its standard deviation is $\sqrt{7d/180}$, and the *relative* spread is $\sqrt{7/180}\,/\,(1/6)\,/\sqrt{d} \approx 1.18/\sqrt{d}$: about 84% at $d = 2$, 12% at $d = 100$, 3.7% at $d = 1000$. Nearest and farthest neighbours of a random point end up at almost the same distance.
- *Near-orthogonality.* For $\mathbf{u}, \mathbf{v}$ with independent standard normal coordinates, $\cos(\mathbf{u}, \mathbf{v})$ has mean $0$ and variance exactly $1/d$, so a typical cosine is about $\pm 1/\sqrt d$: $0.71$ at $d = 2$, $0.32$ at $d = 10$, $0.10$ at $d = 100$, $0.032$ at $d = 1000$. Equivalently, $\mathbf{u}\cdot\mathbf{v}$ has standard deviation $\sqrt d$ while each $\|\mathbf{u}\| \approx \sqrt d$.
- *Ball versus cube.* The unit ball in $\mathbb{R}^d$ has volume $V_d = \pi^{d/2}/\Gamma(d/2 + 1)$, while the cube $[-1, 1]^d$ containing it has volume $2^d$. The ratio is $0.785$ at $d = 2$, $0.524$ at $d = 3$, $0.0025$ at $d = 10$ and $2.5\times 10^{-8}$ at $d = 20$. Volume lives in the corners.
- *Thin shells.* The fraction of a ball's volume within radius $0.9$ of the centre is $0.9^d$: $0.35$ at $d = 10$ and $2.7\times 10^{-5}$ at $d = 100$. Likewise a standard Gaussian sample in $\mathbb{R}^d$ has $\|\mathbf{x}\|^2 \sim \chi^2_d$ with mean $d$ and standard deviation $\sqrt{2d}$, so $\|\mathbf{x}\|$ itself is about $\sqrt d \pm 0.7$ for large $d$: the cloud is a shell of radius $\sqrt d$ whose absolute thickness stays fixed while its relative thickness shrinks like $1/\sqrt{2d}$.

![Two histogram panels for dimensions 2, 10, 100 and 1000. Left: the cosine similarity of random Gaussian vector pairs; for d = 2 it spreads over the whole range from -1 to 1, and as d grows it narrows to a tall spike at 0. Right: the squared distance between random points in the unit cube divided by its mean; for d = 2 it is spread from 0 to beyond 3, and as d grows it narrows to a spike at 1.](figures/high-dimensions.svg)

**Figure 7.** The first two statements, simulated with 20,000 random pairs per dimension. (a) The cosine between two random Gaussian vectors covers the whole range $[-1, 1]$ for $d = 2$ but piles up ever more tightly around 0 as $d$ grows, with standard deviation $1/\sqrt d$. (b) The squared distance between two uniform random points of the unit cube, divided by its mean $d/6$: at $d = 1000$ almost every pair lies within a few per cent of the typical value, so nearest and farthest neighbours barely differ.

**Worked example.** At $d = 10$: $V_{10} = \pi^5/\Gamma(6) = 306.02/120 = 2.55$, and $2.55/2^{10} = 2.55/1024 \approx 0.0025$ — a random point in the cube has a 0.25% chance of landing in the inscribed ball. The typical cosine between two random 10-dimensional Gaussian vectors is $1/\sqrt{10} \approx 0.32$, so a cosine of $0.32$ between two 10-dimensional embeddings is *not* evidence of relatedness, whereas a cosine of $0.32$ between two 1000-dimensional embeddings is ten standard deviations above chance. At $d = 768$ — the width of BERT-base — chance-level cosine similarity is about $0.036$.

**Real-life example (healthcare / genomics).** A hospital wants to find "patients like this one" using gene-expression profiles with $d = 20{,}000$ genes. If most genes are irrelevant noise for the condition in question, the squared distances between any two patients are dominated by $20{,}000$ noise terms whose sum fluctuates by only around 1% ($1.18/\sqrt{20{,}000} \approx 0.8\%$ for the uniform noise above; the constant depends on the noise distribution): every patient is roughly equally far from every other, and the nearest neighbour is nearest by accident. The two standard remedies are both geometric: select or weight the few hundred genes that matter (shrinking the effective $d$), or learn a low-dimensional embedding in which distance reflects the clinical signal. Real data is usually far from uniform — it lies near a low-dimensional surface inside the ambient space, which is why nearest-neighbour methods work at all in practice.

**In ML/AI this is …** the $\sqrt{d_k}$ in scaled dot-product attention: $\mathbf{q}\cdot\mathbf{k}$ for roughly independent zero-mean, unit-variance coordinates has standard deviation $\sqrt{d_k}$, and dividing by it keeps the softmax inputs of order one so the attention weights do not saturate ([Module 17](../../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md)); the reason a $d = 768$ model can store far more than 768 nearly-orthogonal features ("superposition"); the degradation of $k$-NN and kernel density estimation on raw high-dimensional inputs, and the success of learned embeddings that lower the effective dimension; and the shell geometry of Gaussian latent spaces in VAEs and diffusion models ([Module 18](../../part-6-deep-learning-and-modern-ai/18-sampling-mcmc-and-generative-models/README.md)).

## Common pitfalls and misconceptions

1. **Measuring Euclidean distance on unscaled features.** Mistake: computing $\|\mathbf{x} - \mathbf{y}\|$ on raw (age in years, income in EUR, cholesterol in mg/dL). Why wrong: the feature with the largest numbers dominates — income differences of thousands swamp age differences of tens. Fix: standardise each feature (subtract its mean, divide by its standard deviation) before any distance-based method; the case study below shows the nearest neighbour becoming the farthest.
2. **Confusing the dot product with cosine similarity.** Mistake: ranking documents by $\mathbf{q}\cdot\mathbf{d}$ and calling it "similarity". Why wrong: the dot product grows with $\|\mathbf{d}\|$, so long or frequent items win regardless of topic. Fix: normalise when length is noise (text, counts); keep the raw dot product when length carries meaning (attention scores, embeddings trained to encode confidence in their norm).
3. **"Orthogonal means unrelated."** Mistake: concluding two feature vectors have nothing to do with each other because their dot product is zero. Why wrong: orthogonality is a *linear* statement; $x$ and $x^2$ sampled symmetrically around zero are orthogonal yet perfectly dependent. Fix: read $\mathbf{u}\cdot\mathbf{v} = 0$ as "no linear alignment" — for mean-centred vectors it means uncorrelated, nothing more ([Module 09](../../part-3-probability/09-multivariate-probability/README.md)).
4. **Thinking $\mathbf{w}$ lies along the decision boundary.** Mistake: drawing $\mathbf{w}$ as the direction of the separating line. Why wrong: $\mathbf{w}$ is *perpendicular* to the hyperplane; the boundary consists of all directions orthogonal to $\mathbf{w}$. Fix: remember $\mathbf{w}\cdot(\mathbf{x} - \mathbf{y}) = 0$ for any two boundary points, and that rescaling $(\mathbf{w}, b)$ by a positive constant leaves the boundary unchanged.
5. **Projecting with $(\mathbf{a}\cdot\mathbf{b})\,\mathbf{b}$.** Mistake: dropping the denominator $\mathbf{b}\cdot\mathbf{b}$. Why wrong: the formula is only correct when $\mathbf{b}$ is a unit vector; otherwise the "projection" is too long by a factor $\|\mathbf{b}\|^2$ and the residual is no longer orthogonal. Fix: normalise $\mathbf{b}$ first, or always divide by $\mathbf{b}\cdot\mathbf{b}$, and check $\mathbf{r}\cdot\mathbf{b} = 0$ in code.
6. **Treating cosine similarity as a probability or as linear in meaning.** Mistake: "0.8 is twice as related as 0.4", or "0.9 means 90% match". Why wrong: cosine is a cosine of an angle, its baseline depends on the dimension (about $1/\sqrt d$ for random vectors), and its calibration depends on how the embedding was trained. Fix: compare similarities *relative* to each other (rankings) or calibrate thresholds on labelled pairs.
7. **Trusting three-dimensional intuition in $\mathbb{R}^{1000}$.** Mistake: assuming "the nearest neighbour is much closer than the rest" or "random vectors are often similar". Why wrong: distances concentrate and random cosines are $\approx 1/\sqrt d$. Fix: simulate before you trust an intuition about high dimensions (the lesson notebook shows how), and reduce dimension or learn embeddings before nearest-neighbour methods.
8. **Using the $L_2$ norm by reflex.** Mistake: always choosing Euclidean distance or an $L_2$ penalty. Why wrong: $L_2$ squares deviations, so one outlier coordinate dominates; $L_1$ is more robust to outliers and its penalty produces exact zeros. Fix: match the norm to the goal — robustness or sparsity ($L_1$), smoothness and closed-form solutions ($L_2$), worst-case guarantees ($L_\infty$).

## Real-world case studies

### Case study 1 — Patient similarity search and the scaling trap (healthcare)

A clinical decision-support tool finds the three most similar past patients for a new case using Euclidean distance on $\mathbf{x} = (\text{age}, \text{systolic BP}, \text{total cholesterol}, \text{HbA1c})$. The query patient is $\mathbf{q} = (55, 140, 220, 7.0)$. Three candidates: $\mathbf{p}_1 = (56, 142, 225, 9.5)$ (near-identical demographics, but poorly controlled diabetes), $\mathbf{p}_2 = (62, 150, 200, 6.9)$ and $\mathbf{p}_3 = (40, 120, 260, 7.1)$.

Raw distances: $\mathbf{q} - \mathbf{p}_1 = (-1, -2, -5, -2.5)$, so $\|\cdot\| = \sqrt{1 + 4 + 25 + 6.25} = \sqrt{36.25} \approx 6.0$; $\mathbf{q} - \mathbf{p}_2 = (-7, -10, 20, 0.1)$ gives $\sqrt{549.01} \approx 23.4$; $\mathbf{q} - \mathbf{p}_3 = (15, 20, -40, -0.1)$ gives $\sqrt{2225.01} \approx 47.2$. The tool ranks $\mathbf{p}_1$ as by far the most similar patient. Clinically this is the *worst* answer: an HbA1c of 9.5% versus 7.0% is the single most important difference in the table, yet it contributes only $6.25$ to the squared distance while a cholesterol gap of 40 mg/dL contributes $1600$.

Now divide each feature by a typical population standard deviation — 15 years, 20 mmHg, 40 mg/dL and 1.0 percentage point. The scaled differences become $(-0.07, -0.10, -0.13, -2.5)$ with norm $\approx 2.51$; $(-0.47, -0.50, 0.50, 0.10)$ with norm $\approx 0.85$; and $(1.0, 1.0, -1.0, -0.1)$ with norm $\approx 1.73$. The ranking is now $\mathbf{p}_2$, then $\mathbf{p}_3$, then $\mathbf{p}_1$ — the patient who looked most similar is now the least similar (Figure 8). Nothing about the data changed; only the ruler did. Every distance-based method in this course ($k$-NN, $k$-means, kernel methods, retrieval over hand-crafted features) inherits this sensitivity, which is why `StandardScaler` precedes them in almost every scikit-learn pipeline. Learned embeddings sidestep the problem only because the training objective decides the scale of each direction — and even then, embeddings from two different models live on incompatible rulers and must never be compared directly.

![Two panels of horizontal stacked bars, one bar per candidate patient p1, p2, p3, each split into the contributions of age, systolic BP, cholesterol and HbA1c to the squared distance from the query patient. Left, raw units: cholesterol makes up most of every bar and p1 has by far the shortest bar, distance 6.02, against 23.43 and 47.17. Right, after dividing each feature by its typical spread: the HbA1c part makes p1 the longest bar, distance 2.51, against 0.85 for p2 and 1.73 for p3.](figures/feature-scaling.svg)

**Figure 8.** Each bar is a candidate's squared distance to the query patient $\mathbf{q}$, split into the contribution of each feature. (a) In raw units, cholesterol differences of tens of mg/dL supply most of every bar, so $\mathbf{p}_1$ looks nearest. (b) After each feature is divided by its typical spread, the 2.5-point HbA1c gap dominates and $\mathbf{p}_1$ becomes the farthest candidate.

### Case study 2 — A credit rule as a hyperplane, and what the margin buys you (finance)

Return to the lender of Section 7, $\mathbf{w} = (0.05, -0.1)$, $b = -1$, on (income in thousand EUR, DTI in %). A compliance team asks two questions. First: *which applicants would flip if the model were slightly different?* The signed distance answers it directly. Applicant C at $(55, 22)$ has distance $-4.0$; shifting the boundary towards C by 4 units of distance — raising $b$ from $-1$ to about $-0.55$, which lowers the income needed at 20% DTI from 60k to about 51k — flips C to approve, whereas applicant B at distance $-17.9$ stays declined under any plausible perturbation. Ranking the declined applicants by $|\delta(\mathbf{x})|$ gives the review queue for free.

Second: *how confident is the rule?* Suppose the historical approvals and declines are separable and the closest approved applicant sits at distance $+3.1$ while the closest declined one sits at $-2.6$. Then the rule has a margin of $2.6$: every historical decision would survive a shift of the boundary by up to $2.6$ units. A support vector machine formalises exactly this: among all separating hyperplanes it picks the one whose smallest margin is largest, which turns out to depend only on the handful of applicants nearest the boundary (the support vectors). If the team then rescales $\mathbf{w}$ and $b$ by 100 to express the score in "points", nothing geometric changes: raw scores become $100$, $-200$ and $-45$, but the distances stay $8.9$, $-17.9$ and $-4.0$ because $\|\mathbf{w}\|$ also grows by 100. Logistic regression is the instructive exception: its probability $\sigma(\mathbf{w}\cdot\mathbf{x} + b) = \sigma(\|\mathbf{w}\|\,\delta(\mathbf{x}))$ depends on $\|\mathbf{w}\|$ as well as on the signed distance, so the same rescaling would move applicant A from $\sigma(1) \approx 0.73$ to $\sigma(100) \approx 1$ and applicant C from $\sigma(-0.45) \approx 0.39$ to essentially 0 — the boundary stays put, the confidence does not (Self-assessment 4). That scaling freedom is also why an $L_2$ penalty on $\mathbf{w}$ in a linear SVM is equivalent to asking for a wide margin ([Module 13](../../part-5-learning-theory/13-linear-models-and-regularization/README.md)).

### Case study 3 — Users and items in one space (e-commerce)

A streaming service has four users rating three films on a 1–5 scale: Alice $(5, 4, 1)$, Bob $(4, 5, 1)$, Carol $(1, 1, 5)$, Dan $(5, 3, 3)$. A fourth film has been rated by Bob (4), Carol (2) and Dan (5), but not by Alice. How should the service predict Alice's rating? User-based collaborative filtering treats each user as a vector in "film space" and measures taste by cosine similarity: $\cos(\text{Alice}, \text{Bob}) = (20 + 20 + 1)/(\sqrt{42}\sqrt{42}) = 41/42 \approx 0.976$; $\cos(\text{Alice}, \text{Carol}) = (5 + 4 + 5)/(\sqrt{42}\sqrt{27}) = 14/33.67 \approx 0.416$; $\cos(\text{Alice}, \text{Dan}) = (25 + 12 + 3)/(\sqrt{42}\sqrt{43}) = 40/42.50 \approx 0.941$.

The prediction is the similarity-weighted average of the neighbours' ratings for the fourth film:

$$\hat{r}_{\text{Alice}} = \frac{0.976\cdot 4 + 0.416\cdot 2 + 0.941\cdot 5}{0.976 + 0.416 + 0.941} = \frac{9.44}{2.33} \approx 4.05.$$

Bob and Dan, who share Alice's taste, pull the estimate towards 4–5; Carol, who likes what Alice dislikes, has a small weight and pulls it down a little. Two refinements both come from this module. First, cosine ignores a user's overall generosity only partially: a user who rates everything one star higher has a slightly different direction. Subtracting each user's mean rating before computing cosine (which turns it into Pearson correlation) removes that. Second, the same geometry supports *item–item* filtering — films as vectors in "user space" — and, with learned rather than raw vectors, matrix factorisation: fit a vector per user and per film so that their dot product reproduces the known ratings. The mini-project in `exercises.ipynb` builds the user-based version from scratch on synthetic data and evaluates it with RMSE on held-out ratings.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Vectors as data records | Design matrix rows, `nn.Embedding` rows, residual stream of a transformer | Everything downstream is geometry on these points |
| Linear combinations, span | Hidden units as weighted sums; word-vector analogies; mixture-of-experts outputs | A layer can only produce vectors in the span of what it combines |
| Linear independence | Identifiable regression weights, dummy-variable trap, multicollinearity, rank of a design matrix | Dependent features make weights non-unique; nearly dependent ones make them unstable |
| $L_2$ norm | MSE loss, weight decay, gradient clipping, LayerNorm/RMSNorm | Controls scale of weights, errors and activations |
| $L_1$ norm | Lasso, MAE loss, sparse coding | Robustness to outliers; exact zeros via the diamond-shaped ball |
| Dot product | Neuron pre-activation $\mathbf{w}\cdot\mathbf{x} + b$, attention scores $\mathbf{q}\cdot\mathbf{k}$, matrix multiplication | The basic unit of computation in every linear layer |
| Cosine similarity | Semantic search, RAG, deduplication, CLIP, contrastive learning | Compares direction (meaning) independent of magnitude |
| Projection and residual | Least squares, PCA scores, debiasing word embeddings, steering vectors | Splits a vector into "explained by a direction" and "the rest" |
| Orthonormal bases | PCA, SVD, QR, orthogonal initialisation, rotary position embeddings, DCT in JPEG | Coordinates by dot products; lengths and angles preserved |
| Hyperplanes and margins | Logistic regression, SVMs, perceptron, softmax output layers, ReLU units, linear probes | The decision boundary of every linear classifier |
| Distance concentration | Failure of $k$-NN on raw high-dimensional inputs; need for learned embeddings | Distances stop discriminating as $d$ grows |
| Near-orthogonality of random vectors | $\sqrt{d_k}$ scaling in attention, superposition of features, random projections | Chance-level similarity is $\approx 1/\sqrt d$ |

## Cheat sheet

| Object | Formula | Key fact |
|---|---|---|
| Linear combination | $c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k$ | Span = all such combinations: $\{\mathbf{0}\}$, a line, a plane, … through the origin |
| Linear independence | $\sum_i c_i\mathbf{v}_i = \mathbf{0}$ only for all $c_i = 0$ | $\iff$ no vector is redundant $\iff$ coefficients are unique; test: $\operatorname{rank}[\mathbf{v}_1 \cdots \mathbf{v}_k] = k$; at most $n$ in $\mathbb{R}^n$ |
| $L_2$ norm | $\lVert\mathbf{x}\rVert_2 = \sqrt{\sum x_i^2} = \sqrt{\mathbf{x}\cdot\mathbf{x}}$ | Euclidean length; unit ball is round |
| $L_1$, $L_\infty$ norms | $\sum\lvert x_i\rvert$, $\;\max_i\lvert x_i\rvert$ | Diamond and square unit balls; $\lVert\mathbf{x}\rVert_\infty \le \lVert\mathbf{x}\rVert_2 \le \lVert\mathbf{x}\rVert_1$ |
| Unit vector | $\hat{\mathbf{x}} = \mathbf{x}/\lVert\mathbf{x}\rVert$ | Direction without length |
| Distance | $d(\mathbf{x}, \mathbf{y}) = \lVert\mathbf{x} - \mathbf{y}\rVert$ | Depends on the norm *and* on feature scaling |
| Dot product | $\mathbf{u}\cdot\mathbf{v} = \sum u_i v_i = \lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert\cos\theta$ | Alignment and weighted sum at once; $\mathbf{u}\cdot\mathbf{u} = \lVert\mathbf{u}\rVert^2$ |
| Angle | $\theta = \arccos\dfrac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert}$ | Orthogonal $\iff$ $\mathbf{u}\cdot\mathbf{v} = 0$ |
| Cauchy–Schwarz | $\lvert\mathbf{u}\cdot\mathbf{v}\rvert \le \lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert$ | Equality iff parallel; implies the triangle inequality |
| Cosine similarity | $\dfrac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert} \in [-1, 1]$ | Scale-invariant; $\lVert\hat{\mathbf{u}} - \hat{\mathbf{v}}\rVert^2 = 2 - 2\cos$ |
| Projection onto $\mathbf{b}$ | $\mathbf{p} = \dfrac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b}$, $\;\mathbf{r} = \mathbf{a} - \mathbf{p}$ | $\mathbf{r}\perp\mathbf{b}$; $\lVert\mathbf{a}\rVert^2 = \lVert\mathbf{p}\rVert^2 + \lVert\mathbf{r}\rVert^2$; best approximation on the line |
| Orthonormal basis | $\mathbf{q}_i\cdot\mathbf{q}_j = \delta_{ij}$, $\;c_i = \mathbf{x}\cdot\mathbf{q}_i$ | $\mathbf{x} = \sum c_i\mathbf{q}_i$; Parseval $\lVert\mathbf{x}\rVert^2 = \sum c_i^2$; $Q^\top Q = I$ |
| Gram–Schmidt | $\mathbf{v}_j = \mathbf{a}_j - \sum_{i<j}(\mathbf{a}_j\cdot\mathbf{q}_i)\mathbf{q}_i$, $\;\mathbf{q}_j = \mathbf{v}_j/\lVert\mathbf{v}_j\rVert$ | Subtract projections, normalise; `np.linalg.qr` up to signs |
| Hyperplane | $\mathbf{w}\cdot\mathbf{x} + b = 0$ | $\mathbf{w}$ is the normal; $(c\mathbf{w}, cb)$ with $c > 0$ is the same plane |
| Signed distance | $\delta(\mathbf{x}) = \dfrac{\mathbf{w}\cdot\mathbf{x} + b}{\lVert\mathbf{w}\rVert}$ | Margin of $(\mathbf{x}, y)$ is $y\,\delta(\mathbf{x})$ |
| Random cosine in $\mathbb{R}^d$ | mean $0$, std $1/\sqrt d$ | Chance-level similarity; motivates $\sqrt{d_k}$ in attention |
| Distance concentration | relative spread of $\lVert\mathbf{x} - \mathbf{y}\rVert^2 \approx 1.18/\sqrt d$ (uniform cube) | Nearest $\approx$ farthest for large $d$ |
| Ball in cube | $V_d/2^d = \pi^{d/2}/(\Gamma(d/2 + 1)\,2^d)$ | $0.0025$ at $d = 10$; volume sits in the corners |

## Self-assessment

1. **Why is a linear model's prediction "just a dot product", and what does the geometric reading tell you about which inputs produce large predictions?**
   <details><summary>Answer</summary>

   The prediction is $\mathbf{w}\cdot\mathbf{x} + b = \|\mathbf{w}\|\|\mathbf{x}\|\cos\theta + b$. For inputs of fixed length it is largest when $\mathbf{x}$ points along $\mathbf{w}$, smallest when it points the opposite way, and equal to $b$ for every input orthogonal to $\mathbf{w}$. So $\mathbf{w}$ is the direction the model "looks in", and everything perpendicular to it is invisible to the model.

   </details>

2. **Two document vectors have cosine similarity 1 but Euclidean distance 20. What can you say about them?**
   <details><summary>Answer</summary>

   They point in exactly the same direction — one is a positive multiple of the other, e.g. the same word proportions at different lengths. The Euclidean distance of 20 measures only the difference in length. For topic matching the documents are identical; for "how much text", they differ.

   </details>

3. **Prove that the residual of a projection is orthogonal to the direction projected onto.**
   <details><summary>Answer</summary>

   With $\mathbf{p} = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\mathbf{b}$ and $\mathbf{r} = \mathbf{a} - \mathbf{p}$: $\mathbf{r}\cdot\mathbf{b} = \mathbf{a}\cdot\mathbf{b} - \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}(\mathbf{b}\cdot\mathbf{b}) = \mathbf{a}\cdot\mathbf{b} - \mathbf{a}\cdot\mathbf{b} = 0$.

   </details>

4. **A colleague fits a classifier and reports $\mathbf{w} = (2, -1)$, $b = 3$. Another colleague fits the same data and reports $\mathbf{w} = (4, -2)$, $b = 6$. Who is right?**
   <details><summary>Answer</summary>

   Both describe the same hyperplane — the second is the first scaled by 2 — so every point gets the same sign and the same signed distance (the raw score doubles, but so does $\|\mathbf{w}\|$). They differ only in raw score, which matters once the score is pushed through a sigmoid: the second model is "more confident" about every point. Regularisation on $\|\mathbf{w}\|$ is what pins down the scale.

   </details>

5. **Why does Cauchy–Schwarz guarantee that cosine similarity is always between $-1$ and $1$?**
   <details><summary>Answer</summary>

   Cauchy–Schwarz says $|\mathbf{u}\cdot\mathbf{v}| \le \|\mathbf{u}\|\|\mathbf{v}\|$. Dividing both sides by $\|\mathbf{u}\|\|\mathbf{v}\| > 0$ gives $|\cos(\mathbf{u}, \mathbf{v})| \le 1$. Equality holds exactly when the vectors are parallel ($+1$) or anti-parallel ($-1$).

   </details>

6. **You compute coordinates of $\mathbf{x}$ in an orthonormal basis and find $(3, 0, 4, 0, 0)$. What is $\|\mathbf{x}\|$, and in which subspace does $\mathbf{x}$ lie?**
   <details><summary>Answer</summary>

   By Parseval, $\|\mathbf{x}\|^2 = 9 + 0 + 16 + 0 + 0 = 25$, so $\|\mathbf{x}\| = 5$. The zero coordinates say $\mathbf{x}$ has no component along $\mathbf{q}_2, \mathbf{q}_4, \mathbf{q}_5$, so it lies in the two-dimensional plane spanned by $\mathbf{q}_1$ and $\mathbf{q}_3$.

   </details>

7. **Why do the first three steps of Gram–Schmidt produce vectors that are orthogonal to each other, not just the last one to the previous?**
   <details><summary>Answer</summary>

   Each new $\mathbf{v}_j$ is the residual after subtracting the projections onto *all* previous $\mathbf{q}_i$. Because those $\mathbf{q}_i$ are already mutually orthogonal, removing the component along one of them does not re-introduce a component along another, so $\mathbf{v}_j\cdot\mathbf{q}_i = \mathbf{a}_j\cdot\mathbf{q}_i - (\mathbf{a}_j\cdot\mathbf{q}_i)(\mathbf{q}_i\cdot\mathbf{q}_i) = 0$ for every $i < j$.

   </details>

8. **Two 512-dimensional embeddings of unrelated sentences have cosine similarity 0.05. Is that surprisingly high, surprisingly low, or about what you would expect?**
   <details><summary>Answer</summary>

   For random directions in $\mathbb{R}^{512}$ the typical cosine is about $\pm 1/\sqrt{512} \approx 0.044$, so 0.05 is one standard deviation from zero — chance level. Real embedding models are often *anisotropic* (all sentences share a common direction), which raises the baseline; estimate it from unrelated pairs before interpreting any similarity.

   </details>

9. **In $k$-nearest-neighbour regression on 2,000 raw sensor features, the nearest neighbour is only 3% closer than the 100th nearest. Diagnose the problem and name two fixes.**
   <details><summary>Answer</summary>

   Distance concentration: with thousands of mostly uninformative coordinates, all pairwise distances fluctuate by only about $1/\sqrt d$ relative, so "nearest" carries almost no information. Fixes: reduce the effective dimension (drop or down-weight irrelevant features, project to leading principal components as in Module 03, or learn a task-specific embedding), and use a distance or learned metric that reflects what matters for the prediction.

   </details>

10. **Why does the unit ball of the $L_1$ norm "encourage sparsity" while the $L_2$ ball does not?**
    <details><summary>Answer</summary>

    The $L_1$ ball is a diamond whose corners lie on the coordinate axes, where some coordinates are exactly zero. When a smooth loss is minimised subject to staying inside the ball, the optimum tends to land on a corner, giving exact zeros. The $L_2$ ball is round with no corners, so the optimum lands at a generic point where all coordinates are small but non-zero. Module 13 makes this precise for ridge and lasso.

    </details>

11. **A dataset has three feature columns, and you add a fourth that is the sum of the first three. What happens to the span of the columns, and to the weights of a linear model fitted on all four?**
    <details><summary>Answer</summary>

    The span does not change: the new column is a combination of the others, so it adds no new direction, and the four columns are linearly dependent ($\mathbf{c}_1 + \mathbf{c}_2 + \mathbf{c}_3 - \mathbf{c}_4 = \mathbf{0}$). The model can make exactly the same predictions as before, but its weights are no longer unique: $(w_1 + t, w_2 + t, w_3 + t, w_4 - t)$ gives identical predictions for every $t$, because the change in the prediction is $t$ times a combination that equals $\mathbf{0}$. Drop the redundant column, or add a penalty such as ridge that singles out one solution (Modules 02 and 13).

    </details>

## Further reading

- Gilbert Strang, *Introduction to Linear Algebra* (any recent edition), Chapters 1 and 4 — vectors, dot products, orthogonality and projections, written with unusual clarity. His MIT 18.06 lectures are free on MIT OpenCourseWare.
- Stephen Boyd and Lieven Vandenberghe, *Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares* (Cambridge University Press, 2018; free PDF from the authors' Stanford page), Chapters 1–5 — vectors, norms, distance, clustering and linear independence with many applied examples.
- Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020; free PDF at mml-book.github.io), Chapters 2 and 3 — linear algebra and "analytic geometry": norms, inner products, angles, projections, Gram–Schmidt.
- Ian Goodfellow, Yoshua Bengio and Aaron Courville, *Deep Learning* (MIT Press, 2016; free online), Chapter 2 — the linear algebra a deep-learning practitioner uses daily.
- Avrim Blum, John Hopcroft and Ravindran Kannan, *Foundations of Data Science* (Cambridge University Press, 2020; free PDF), Chapter 2 "High-Dimensional Space" — the rigorous version of Section 8: concentration, volume of the ball, near-orthogonality.
- 3Blue1Brown, *Essence of Linear Algebra* (free video series) — the best visual intuition for vectors, span, linear independence and the dot product.
- Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan and Uri Shaft, "When Is 'Nearest Neighbor' Meaningful?" (ICDT 1999), and Charu Aggarwal, Alexander Hinneburg and Daniel Keim, "On the Surprising Behavior of Distance Metrics in High Dimensional Space" (ICDT 2001) — the two classic papers on distance concentration and the choice of $L_p$ norm.
- Tomas Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013) — word2vec and the arithmetic of word vectors.
- Tolga Bolukbasi et al., "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings" (NeurIPS 2016) — projections used to remove a direction from embeddings.
- Ashish Vaswani et al., "Attention Is All You Need" (NeurIPS 2017), Section 3.2.1 — the one paragraph that justifies the $\sqrt{d_k}$ scaling with exactly the argument of Section 8.

## What's in the notebooks

**`lesson.ipynb`** walks through the eight concepts with code and figures: drawing vectors, sums and scalar multiples with `quiver`; computing norms three ways (by hand, via $\sqrt{\mathbf{x}\cdot\mathbf{x}}$, via `np.linalg.norm`) and drawing the $L_1$, $L_2$ and $L_\infty$ unit balls; verifying the law-of-cosines identity and Cauchy–Schwarz numerically on random vectors; a toy semantic search over eight hand-written bag-of-words documents that shows cosine and Euclidean rankings disagreeing; a projection plot with its residual and the parabola of squared errors; a Gram–Schmidt implementation checked against `np.linalg.qr`; a nearest-centroid classifier drawn as a hyperplane with its normal vector and signed distances; and a simulation of pairwise cosine similarity and distance concentration for $d = 2, 10, 100, 1000$, plus the ball-in-cube volume ratio.

**`exercises.ipynb`** (with `solutions.ipynb`) has twelve exercises with instant self-checks: norms without `np.linalg.norm`, angles between vectors, normalisation with a zero-vector guard, top-$k$ cosine retrieval over an embedding table, projection with an orthogonality check, Gram–Schmidt, vectorised signed distances to a hyperplane, two conceptual questions (cosine versus Euclidean for text; the $\sqrt{d_k}$ in attention), a stretch exercise on distance concentration, a stretch exercise on reflecting points across a hyperplane, and a mini-project that builds a user-based collaborative-filtering recommender on a synthetic ratings matrix and evaluates it with RMSE against held-out ratings.
