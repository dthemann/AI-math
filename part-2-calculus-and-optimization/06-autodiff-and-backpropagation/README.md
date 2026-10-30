# Module 06 — Automatic Differentiation and Backpropagation

> After this module you can take any program that computes a loss, explain exactly how a framework gets its gradient, build that machinery yourself in a few dozen lines of Python, and prove with a numerical check that it is correct.

## Why this matters for ML and AI

Every modern learning system is trained the same way: write down a loss, compute its gradient with respect to millions or billions of parameters, step downhill, repeat. [Module 05](../05-optimization/README.md) covered the stepping. This module covers the gradient, and the gradient is where the engineering miracle hides. A model such as GPT-2 small has 124 million parameters. If you estimated its gradient by nudging each parameter and re-running the model, one gradient would cost 124 million forward passes. Reverse-mode automatic differentiation (AD) — backpropagation is its name when the program is a neural network — delivers the exact gradient for the price of roughly two or three forward passes, regardless of how many parameters there are. Without that single algorithmic fact, deep learning would not exist.

Autodiff is also the piece of the stack that practitioners most often treat as magic. `loss.backward()` in PyTorch, `jax.grad` in JAX, `tf.GradientTape` in TensorFlow: all of them record the operations your code performs, then run the chain rule backwards through that recording. Understanding the mechanism pays off constantly: you will know why gradients must be zeroed between steps, why training memory is dominated by activations rather than weights, why `log(softmax(z))` produces NaNs while `log_softmax(z)` does not, how custom layers get their backward rules, and how to verify a hand-written gradient in thirty seconds instead of debugging a silently wrong model for a week.

The same mathematics runs far outside machine learning. Adjoint methods compute aerodynamic shape sensitivities in aircraft design, risk sensitivities ("Greeks") for banks' derivative books, and the gradients that weather centres use to fit forecast models to observations. These are all reverse-mode AD under other names, and you will meet them in the examples below.

## Learning objectives

- Compare symbolic, numerical and automatic differentiation by cost and accuracy, and choose the right tool for a given task.
- Draw the computational graph of an expression, order it topologically and run a forward pass by hand.
- Execute forward-mode (tangent) and reverse-mode (adjoint) sweeps by hand on a small graph, recognise them as Jacobian-vector and vector-Jacobian products, and explain, with a counting argument, why reverse mode wins for scalar losses.
- Derive the backpropagation recurrence $\delta_l = (W_{l+1}^\top \delta_{l+1}) \odot f'(z_l)$ and use it to compute every gradient of a two-layer network by hand, for one example and for a minibatch.
- Implement a scalar autograd engine (a `Value` class with operator overloading, closures and a topological sort) and train a neural network with it.
- Write vector-Jacobian-product rules for matrix multiplication, broadcasting, elementwise functions and fused softmax-cross-entropy, derive new ones from the scalar definition and the shape test, and explain why frameworks fuse operations.
- Verify any gradient implementation with central differences and a relative-error criterion, and recognise the false alarms caused by kinks and float32.
- Diagnose non-differentiable points, overflow in `exp`/`log`, and vanishing or exploding gradients, and name the standard fixes.

## Prerequisites

- [Module 04 — Differential calculus](../04-differential-calculus/README.md): partial derivatives, gradients, Jacobians and above all the multivariate chain rule, $\frac{\partial L}{\partial x} = \sum_i \frac{\partial L}{\partial v_i}\frac{\partial v_i}{\partial x}$.
- [Module 05 — Optimization](../05-optimization/README.md): gradient descent, step sizes, SGD and Adam; this module supplies the gradients those algorithms consume.
- [Module 02 — Matrices and linear maps](../../part-1-linear-algebra/02-matrices-and-linear-maps/README.md): matrix products, transposes and shape bookkeeping, needed for the vector-Jacobian rules in Section 6.
- Python: classes, operator overloading (`__add__`, `__mul__`), closures and recursion.

## 1. Three ways to get a derivative

**Intuition.** You have a program that maps inputs to a number and want to know how that number changes when an input moves. *Symbolic* differentiation manipulates formulas and hands back a new formula; it is exact, but the formula can grow explosively and it cannot see through `if` statements or loops. *Numerical* differentiation nudges the input and measures the change; it works on any black box but is approximate and costs one extra program run per input. *Automatic* differentiation applies the chain rule to the program's actual arithmetic, one primitive operation at a time; it is exact up to rounding, handles control flow (you differentiate the path actually taken), and in reverse mode costs a small constant times one program run for *all* inputs at once.

**Definition.** The forward difference and central difference estimates of $f'(x)$ with step $h$ are

$$
D_h^{+} f(x) = \frac{f(x+h) - f(x)}{h} = f'(x) + \tfrac{1}{2} f''(x)\, h + O(h^2), \qquad
D_h^{c} f(x) = \frac{f(x+h) - f(x-h)}{2h} = f'(x) + \tfrac{1}{6} f'''(x)\, h^2 + O(h^4).
$$

Both also suffer round-off error of order $\varepsilon |f(x)| / h$, where $\varepsilon \approx 2.2 \times 10^{-16}$ is the machine epsilon of float64 (about $1.2 \times 10^{-7}$ in float32). Truncation error shrinks with $h$ while round-off grows, so there is an optimal step: roughly $h \approx \sqrt{\varepsilon} \approx 10^{-8}$ for forward differences (achievable error $\sim 10^{-8}$) and $h \approx \varepsilon^{1/3} \approx 10^{-5}$ for central differences (error $\sim 10^{-10}$). For a function of $n$ inputs, a full gradient needs $n+1$ (forward) or $2n$ (central) evaluations.

**Worked example.** Take $f(x) = x^3$ at $x = 2$, so $f'(2) = 12$, with $h = 0.1$:

- Forward: $\dfrac{2.1^3 - 2^3}{0.1} = \dfrac{9.261 - 8}{0.1} = 12.61$. Error $0.61 = 3xh + h^2 = 0.6 + 0.01$: first order in $h$.
- Central: $\dfrac{2.1^3 - 1.9^3}{0.2} = \dfrac{9.261 - 6.859}{0.2} = 12.01$. Error $0.01 = h^2$: second order in $h$.

Shrinking $h$ to $10^{-5}$ makes the central estimate $12.0000000002$. Shrinking it to $10^{-12}$ makes things *worse*: $2.000000000001^3 - 1.999999999999^3$ is computed with only a few significant digits left, and the quotient ($12.0011$) is accurate to about three decimals. [Module 04](../04-differential-calculus/README.md), Figure 1(b), plots the whole error curve for this example: a V whose central-difference minimum lies near $h \approx 6.5 \times 10^{-6}$. Automatic differentiation has no $h$ and no such trade-off: the derivative of $x^3$ is evaluated as $3x^2 = 12$ exactly.

**Real-life example (engineering and finance).** An aerodynamics team parameterises a wing with 2,000 shape variables and wants the sensitivity of drag to each one. One flow simulation takes an hour. Finite differences would need 2,001 simulations — three months of compute for a single gradient. The *adjoint method*, which is reverse-mode AD applied to the discretised flow equations, delivers all 2,000 sensitivities for roughly the cost of two simulations. The same trick, known in finance as adjoint algorithmic differentiation (AAD), lets a bank compute the sensitivities of a derivatives book to hundreds of market inputs — the Greeks — with one pricing plus one backward sweep instead of hundreds of "bump and revalue" repricings (Giles and Glasserman, 2006).

**In ML/AI this is …** `loss.backward()` and `jax.grad` (automatic, reverse mode); `torch.autograd.gradcheck` and `scipy.optimize.approx_fprime` (numerical, used to *test* the automatic kind); SymPy or Mathematica (symbolic, used to derive closed forms such as the softmax Jacobian, rarely used at training time).

## 2. Computational graphs

**Intuition.** A program is a long composition of tiny functions: add, multiply, exponentiate. If you write each intermediate result as a named variable and draw an arrow from every operand to every result that uses it, you get a directed acyclic graph. The graph is a complete record of *how* the output was computed, which is exactly what the chain rule needs. Frameworks build this graph for you at run time — every tensor remembers the operation that produced it.

**Definition.** A computational graph for a function $f$ consists of nodes $v_1, \dots, v_N$. The first $n$ nodes are the inputs. Every other node is produced by a primitive operation $\phi_i$ applied to its parents,

$$
v_i = \phi_i\big(v_j : j \in \mathrm{pa}(i)\big), \qquad i = n+1, \dots, N,
$$

and the last node $v_N$ is the output. A *topological order* lists the nodes so that every node appears after all of its parents; the *forward pass* evaluates the nodes in that order. Because the graph is acyclic, such an order always exists, and depth-first search produces one in $O(\text{nodes} + \text{edges})$ time.

**Worked example.** The squared error of a one-feature linear model, $L = (wx + b - y)^2$, with $w = 2,\ x = 3,\ b = 1,\ y = 5$:

```
  w ──┐
      ├─[×]─ v1 ──┐
  x ──┘           ├─[+]─ v2 ──┐
  b ──────────────┘           ├─[−]─ v3 ──[²]─ L
  y ──────────────────────────┘
```

Topological order and forward values: $v_1 = wx = 6$, $v_2 = v_1 + b = 7$, $v_3 = v_2 - y = 2$, $L = v_3^2 = 4$. Four inputs, four operation nodes, one output. Nothing about the graph is specific to the numbers; the same graph with different inputs gives a different forward pass but identical structure, which is why a framework can reuse one backward routine for every data point. Figure 1(a) draws this forward pass; Figure 1(b) runs the same graph backwards, the subject of the next section.

![Two stacked diagram panels of the graph of the squared error of a one-feature linear model. Top: four grey input boxes w = 2, x = 3, b = 1 and y = 5 on the left feed blue operation boxes v1 = wx = 6, v2 = v1 + b = 7, v3 = v2 - y = 2 and the output L = v3 squared = 4 along solid green arrows that run left to right. Bottom: the same graph with dashed red arrows running right to left; the boxes now hold the adjoints, 1 at L, 4 at v3, v2 and v1, 4 at b, -4 at y, 12 at w and 8 at x, and each arrow is labelled with the local derivative it multiplies by: 2 v3 = 4, 1, -1, 1, 1, x = 3 and w = 2.](figures/computational-graph.svg)

**Figure 1.** The worked example as a computational graph. (a) The forward pass evaluates the nodes in topological order — $v_1 = 6$, $v_2 = 7$, $v_3 = 2$, $L = 4$ — and keeps every value. (b) The backward pass on the same graph: seed $\bar L = 1$, then multiply by the local derivative on each edge, right to left. The edge into $v_3$ multiplies by $2v_3 = 4$, which can be evaluated only because the forward value $v_3 = 2$ was kept; the edge into $w$ multiplies by $x = 3$ and gives $\bar w = 12$, the edge into $x$ multiplies by $w = 2$ and gives $\bar x = 8$, and the edge into $y$ multiplies by $-1$ and gives $\bar y = -4$. Every node here has a single child, so nothing is summed; Figure 3 shows a node with two children. Panel (b) is exactly the four closure calls of Section 5's worked example, and $\bar w = 12$ is the value that Section 7's gradient check confirms.

**Real-life example (manufacturing).** A plant's cost model for a product is a graph: raw-material prices feed component costs, component costs and labour rates feed assembly cost, assembly cost and scrap rate feed unit cost, unit cost and volume feed margin. A spreadsheet of such formulas *is* a computational graph, and the spreadsheet engine recalculates cells in topological order after an edit. "What is the sensitivity of margin to the scrap rate?" is a derivative on this graph, and Section 3 shows how to get it for every input at once.

**In ML/AI this is …** the dynamic graph PyTorch records during a forward pass (inspect `tensor.grad_fn` and its `next_functions`), the static graphs of TensorFlow 1 and ONNX, and the `jaxpr` that `jax.make_jaxpr` prints. "Define-by-run" means the graph is rebuilt from the operations actually executed on each call.

## 3. The chain rule on graphs: forward mode and reverse mode

**Intuition.** Each edge of the graph carries one *local* derivative, $\partial v_i / \partial v_j$, which depends only on the primitive $\phi_i$ and its operands. The derivative of the output with respect to an input is the sum over all paths from input to output of the products of the edge derivatives along the path. Enumerating paths is exponential, but the chain rule lets you accumulate them in one sweep, and there are two directions to sweep. *Forward mode* starts at one input and pushes "how fast does each node change when this input moves" (a *tangent*) towards the output. *Reverse mode* starts at the output and pulls "how much does the output care about this node" (an *adjoint*) back towards the inputs.

**Definition.** Fix an input $x_j$ and define tangents $\dot v_i = \partial v_i / \partial x_j$. Forward mode seeds $\dot x_j = 1$, $\dot x_k = 0$ for $k \ne j$, and propagates in topological order:

$$
\dot v_i = \sum_{k \in \mathrm{pa}(i)} \frac{\partial \phi_i}{\partial v_k}\, \dot v_k .
$$

Fix the output $L$ and define adjoints $\bar v_k = \partial L / \partial v_k$. Reverse mode seeds $\bar L = 1$ and propagates in *reverse* topological order:

$$
\bar v_k = \sum_{i \in \mathrm{ch}(k)} \bar v_i\, \frac{\partial \phi_i}{\partial v_k},
$$

summing over the children of $v_k$. The sum is why a node used twice receives two contributions, and why every autograd engine *accumulates* gradients with `+=`.

*Local derivatives.* Each edge's factor $\partial \phi_i / \partial v_k$ depends only on the primitive $\phi_i$ and the forward values of its operands. The worked example below, $f = (x_1 x_2 + x_1)^2$ at $(2, 3)$, has five edges and five local derivatives: $\partial v_1/\partial x_1 = x_2 = 3$, $\partial v_1/\partial x_2 = x_1 = 2$, $\partial v_2/\partial v_1 = 1$, $\partial v_2/\partial x_1 = 1$ and $\partial v_3/\partial v_2 = 2v_2 = 16$. The first needs the value $x_2 = 3$ and the last needs $v_2 = 8$: a derivative sweep reads forward values. Forward mode has them at hand, because it runs alongside the forward pass; reverse mode runs afterwards, so the forward pass must store them — the memory cost of Section 4 and Case study 1.

*Paths.* Applied repeatedly, the multivariate chain rule of [Module 04](../04-differential-calculus/README.md) becomes a sum over paths: $\partial f/\partial x_1$ is the sum, over every path from $x_1$ to the output, of the product of the local derivatives along it. In the worked example $x_1$ has two paths: $x_1 \to v_1 \to v_2 \to v_3$ contributes $3 \cdot 1 \cdot 16 = 48$ and $x_1 \to v_2 \to v_3$ contributes $1 \cdot 16 = 16$, a total of $64$; $x_2$ has the single path $x_2 \to v_1 \to v_2 \to v_3$, contributing $2 \cdot 1 \cdot 16 = 32$. Path counts explode — between one input and the output of a network with ten fully connected hidden layers of width 100 and a scalar output there are $100^{10} = 10^{20}$ paths — but neither sweep enumerates them. The tangent $\dot v_i$ is the sum over all paths from the seeded input to $v_i$, the adjoint $\bar v_k$ the sum over all paths from $v_k$ to the output, and each sweep extends these partial sums by one edge at a time, so it visits every edge once.

*Forward mode is a Jacobian-vector product.* The seed need not be a coordinate vector. Seeding the inputs with any direction $\dot{\mathbf{x}}$ makes the sweep compute $\dot{\mathbf{y}} = J\dot{\mathbf{x}}$, the directional derivative of every output along $\dot{\mathbf{x}}$; the seed $\mathbf{e}_j$ picks out column $j$ of $J$. In the worked example $J = (64 \ \ 32)$, and the seed $\dot{\mathbf{x}} = (1, 1)$ gives $\dot v_1 = 3 + 2 = 5$, $\dot v_2 = 5 + 1 = 6$, $\dot v_3 = 16 \cdot 6 = 96 = 64 + 32$: one number, the rate of change of $f$ when both inputs increase at unit speed (the slope per unit distance along the diagonal, Module 04's directional derivative for the unit vector $(1, 1)/\sqrt{2}$, is $96/\sqrt{2} \approx 67.9$), not the two partial derivatives. Tangents ride along with the forward pass and can be discarded as soon as their children have used them, so forward mode stores nothing extra; dual numbers (below) implement it. A full gradient needs one sweep per input (Figure 3(a) and (b)).

*Reverse mode is a vector-Jacobian product.* Seeding the outputs with any vector $\bar{\mathbf{y}}$ makes the sweep compute $\bar{\mathbf{x}} = J^\top \bar{\mathbf{y}}$; the seed $\mathbf{e}_i$ gives row $i$ of $J$, and for a scalar loss the seed $\bar L = 1$ gives its only row, the entire gradient. Two rules make the sweep correct. A node's adjoint is complete only once every child has passed its contribution back ($\bar x_1$ needs both $\bar v_1$ and $\bar v_2$), which is why the sweep runs in *reverse* topological order and why contributions are added with `+=`, never assigned (common pitfall 3). And a contribution $\bar v_i\, \partial \phi_i / \partial v_k$ reads stored forward values whenever the local derivative depends on them: the factors $x_2 = 3$, $x_1 = 2$ and $2v_2 = 16$ in Figure 3(c); only the additions' factors of 1 need nothing stored. Every backward pass in [Module 16](../../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md) and [Module 17](../../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md) is this sweep, with array-valued nodes (Section 6).

**Cost.** One forward sweep gives the derivative with respect to one input, for every node and output. One reverse sweep gives the derivative of one output with respect to every node and input. Each sweep costs a small constant multiple (in practice two to four) of the forward evaluation. For $f : \mathbb{R}^n \to \mathbb{R}^m$, the full Jacobian therefore costs about $n$ forward sweeps or $m$ reverse sweeps (Figure 2(a)). A training loss has $m = 1$ and $n$ in the millions: reverse mode wins by a factor of $n$. For GPT-2 small, $n = 124{,}439{,}808$; the gradient costs about three forward passes instead of 124 million with forward differences, or $2.5 \times 10^8$ with central differences (Figure 2(b)). This is Griewank's "cheap gradient principle", and it is the single reason gradient-based deep learning is computationally feasible.

![Two panels. Left: a schematic Jacobian of a map from 8 inputs to 3 outputs, drawn as a grid of 3 rows by 8 columns; column 3 is shaded green, with an arrow labelled forward sweep seeded at x3 fills one column, and row 2 is shaded red, with an arrow labelled reverse sweep seeded at f2 fills one row; notes below say that n inputs need n forward sweeps, m outputs need m reverse sweeps, and a training loss has a single row, the whole gradient. Right: a log-log plot of the cost of one full gradient in forward-pass units against the number of inputs n, from below 1 to 2 billion; the orange central-difference line 2n and the dashed green forward-mode line, about 3n inside a band from 2n to 4n, rise together, while the red reverse-mode line stays flat at about 3 inside a band from 2 to 4; square markers at n = 120 compare 240 pricings with 3 to 4, and round markers at n = 124 million compare 250 million passes with about 3.](figures/jacobian-rows-columns.svg)

**Figure 2.** (a) What one sweep computes, on a schematic Jacobian of $f : \mathbb{R}^8 \to \mathbb{R}^3$ (the shape is illustrative; self-assessment 3 asks about a $3 \times 1000$ one). A forward sweep seeded at input $x_3$ fills column 3, $J\mathbf{e}_3$; a reverse sweep seeded at output $f_2$ fills row 2, $\mathbf{e}_2^\top J$. The whole Jacobian takes $n = 8$ forward or $m = 3$ reverse sweeps, and a training loss, with $m = 1$, has a single row: the whole gradient. (b) The cost of one full gradient in forward-pass units: central differences need $2n$ evaluations and forward mode $n$ sweeps of two to four passes each (exact, but no cheaper), while reverse mode needs two to four whatever $n$ is. At GPT-2 small's $n = 1.24 \times 10^8$ (round markers) that is $2.5 \times 10^8$ passes against about 3, the factor of $10^8$ of Case study 1; at Case study 2's $n = 120$ (square markers) it is 240 pricings against three or four.

*Why "two to four".* The backward rule of a primitive costs a small multiple of the primitive itself, typically about twice. $c = ab$ is one multiplication forward and two backward ($\bar a \mathrel{+}= b\,\bar c$, $\bar b \mathrel{+}= a\,\bar c$). A matrix product $Y = AB$ with $A$ of shape $m \times k$ and $B$ of shape $k \times n$ costs about $2mkn$ floating-point operations forward, and its backward rule is two products of exactly that size, $\bar A = G B^\top$ and $\bar B = A^\top G$ (Section 6). A network dominated by matrix products therefore spends about twice its forward cost in the reverse sweep, so the whole gradient, forward pass included, costs about three forward passes; reading back the stored values and the bookkeeping push the factor towards four.

*Which mode when.* The sweep counts decide: for $f : \mathbb{R}^n \to \mathbb{R}^m$ use reverse mode when $m < n$ and forward mode when $n < m$. A training loss ($m = 1$) always goes to reverse mode; the sensitivity of a long simulated trajectory (thousands of outputs) to one or two design parameters goes to forward mode. When $n \approx m$ the counts tie, and forward mode wins on memory because it stores nothing. Hessian-vector products apply one mode to the output of another: forward over reverse, or reverse over reverse as in Section 8. Both modes return exact derivatives of the program, so both must agree with central differences (Section 7).

**Worked example.** Let $f(x_1, x_2) = (x_1 x_2 + x_1)^2$ at $(x_1, x_2) = (2, 3)$. Nodes: $v_1 = x_1 x_2 = 6$, $v_2 = v_1 + x_1 = 8$, $v_3 = v_2^2 = 64$. Note that $x_1$ has two children.

Reverse sweep: $\bar v_3 = 1$; $\bar v_2 = \bar v_3 \cdot 2 v_2 = 16$; $\bar v_1 = \bar v_2 \cdot 1 = 16$; $\bar x_1 = \bar v_1 \cdot x_2 + \bar v_2 \cdot 1 = 48 + 16 = 64$; $\bar x_2 = \bar v_1 \cdot x_1 = 32$. One sweep, both partial derivatives (Figure 3(c)). The two terms of $\bar x_1$ are the two paths of the *Paths* paragraph.

Forward sweep seeded at $x_1$ ($\dot x_1 = 1, \dot x_2 = 0$): $\dot v_1 = x_2 \dot x_1 + x_1 \dot x_2 = 3$; $\dot v_2 = \dot v_1 + \dot x_1 = 4$; $\dot v_3 = 2 v_2 \dot v_2 = 64$. A second sweep seeded at $x_2$: $\dot v_1 = 2$, $\dot v_2 = 2$, $\dot v_3 = 32$. Two sweeps for the same information (Figure 3(a) and (b)). Check against the closed form $f = x_1^2 (x_2 + 1)^2$: $\partial f / \partial x_1 = 2 x_1 (x_2+1)^2 = 64$ and $\partial f / \partial x_2 = 2 x_1^2 (x_2 + 1) = 32$.

![Three stacked diagram panels of the same graph: grey input boxes x1 = 2 and x2 = 3 on the left, blue boxes v1 = x1 x2 = 6, v2 = v1 + x1 = 8 and v3 = v2 squared = 64, and a long edge from x1 straight to v2, so that x1 feeds two nodes. Every edge carries the same grey local derivative in all three panels: x2 = 3 and x1 = 2 into v1, 1 and 1 into v2, and 2 v2 = 16 into v3. Top: a forward sweep along green arrows with tangent 1 at x1 and 0 at x2, giving the tangents 3, 4 and 64. Middle: a second forward sweep with tangent 0 at x1 and 1 at x2, giving 2, 2 and 32. Bottom: one reverse sweep along dashed red arrows from the adjoint 1 at v3, giving 16 at v2, 16 at v1, 32 at x2, and 16 times 3 plus 16 times 1 = 64 at x1.](figures/forward-vs-reverse.svg)

**Figure 3.** The worked example $f = (x_1 x_2 + x_1)^2$ at $(2, 3)$, swept three times; the grey edge labels are the same five local derivatives in every panel. (a) Forward mode seeded at $x_1$: the tangents $3$, $4$, $64$ flow left to right and end at $\partial f/\partial x_1 = 64$. Because $x_1$ feeds two nodes, $\dot x_1$ is used twice. (b) A second forward sweep, seeded at $x_2$, is needed for $\partial f/\partial x_2 = 32$. (c) One reverse sweep from $\bar v_3 = 1$ delivers both. $x_1$ has two children, so its adjoint adds two contributions, $16 \cdot 3 + 16 \cdot 1 = 48 + 16 = 64$, one per path; an engine that assigned instead of adding would keep only one of them, 16 or 48 (common pitfall 3).

Forward mode has a neat implementation through *dual numbers*: carry pairs $(v, \dot v)$ with $(a, \dot a)(b, \dot b) = (ab,\ a \dot b + \dot a b)$ and $\exp(a, \dot a) = (e^a, e^a \dot a)$. The tangent rides along with the value and no graph is stored; the lesson notebook implements this `Dual` class.

**Real-life example (climate and weather).** Numerical weather prediction centres fit the initial state of the atmosphere — tens of millions of grid values — to the past hours of observations by minimising a misfit function. The gradient of that scalar misfit with respect to every initial-state variable is computed by an *adjoint model*: the forecast code run backwards, i.e. reverse-mode AD of a physics simulation. This technique (4D-Var) has been operational at the European Centre for Medium-Range Weather Forecasts since 1997; forward mode would require one forecast per initial-state variable and is out of the question.

**In ML/AI this is …** `loss.backward()` and `jax.vjp` (reverse mode, adjoints); `jax.jvp` and `torch.func.jvp` (forward mode, tangents). Forward mode is the right tool when outputs outnumber inputs, for Jacobians of small functions (`jax.jacfwd`), and for Hessian-vector products via forward-over-reverse (the alternative to the reverse-over-reverse form of Section 8).

## 4. Backpropagation: reverse mode on a neural network

**Intuition.** A feed-forward network is a chain of layers, each an affine map followed by an elementwise nonlinearity. Reverse mode on that chain has a regular structure: the adjoint of a layer's pre-activation, usually called $\delta_l$, is obtained from the next layer's $\delta_{l+1}$ by multiplying with the transposed weight matrix and masking with the activation derivative. Once you have $\delta_l$, the weight gradient is an outer product with the layer's input. "Backpropagation" (Rumelhart, Hinton and Williams, 1986) is this recurrence; reverse-mode AD had been described earlier by Linnainmaa (1970) and Werbos (1974).

**Definition.** For layers $l = 1, \dots, L$ with input $a_0 = x$:

$$
z_l = W_l a_{l-1} + b_l, \qquad a_l = f(z_l), \qquad \text{loss } \mathcal{L}(a_L, y).
$$

Define $\delta_l := \partial \mathcal{L} / \partial z_l$ (a vector the size of layer $l$). Then

$$
\delta_L = \nabla_{a_L} \mathcal{L} \odot f'(z_L), \qquad
\delta_l = \big(W_{l+1}^\top \delta_{l+1}\big) \odot f'(z_l), \qquad
\frac{\partial \mathcal{L}}{\partial W_l} = \delta_l\, a_{l-1}^\top, \qquad
\frac{\partial \mathcal{L}}{\partial b_l} = \delta_l ,
$$

where $\odot$ is elementwise multiplication. Derivation of the middle formula in three lines: $z_{l+1} = W_{l+1} f(z_l) + b_{l+1}$, so by the chain rule $\partial \mathcal{L} / \partial z_l = (\partial z_{l+1} / \partial z_l)^\top \delta_{l+1}$; the Jacobian $\partial z_{l+1}/\partial z_l = W_{l+1}\, \mathrm{diag}(f'(z_l))$; transposing gives $\mathrm{diag}(f'(z_l)) W_{l+1}^\top \delta_{l+1}$, which is the stated elementwise product. For a linear output layer, $f'(z_L) = 1$.

*Where $\delta$ comes from.* $\delta_l$ is the adjoint $\bar z_l$ of Section 3, for the pre-activation node $z_l$. It is the one adjoint carried from layer to layer because $z_l$ is a bottleneck: the parameters of layer $l$, and everything below it, reach the loss only through $z_l$.

*The weight gradient is an outer product.* Entry $i$ of $z_l$ is $z_{l,i} = \sum_j W_{l,ij}\, a_{l-1,j} + b_{l,i}$. The weight $W_{l,ij}$ appears in this one entry and nowhere else, with coefficient $a_{l-1,j}$, so $\partial \mathcal{L}/\partial W_{l,ij} = \delta_{l,i}\, a_{l-1,j}$; collected into a matrix, that is the outer product $\delta_l a_{l-1}^\top$. The bias enters with coefficient 1, so $\partial \mathcal{L}/\partial b_{l,i} = \delta_{l,i}$. Two consequences can be read off without computing anything: row $i$ of the weight gradient is zero whenever $\delta_{l,i} = 0$ (a unit that passes no signal back, such as a dead ReLU), and column $j$ is zero whenever $a_{l-1,j} = 0$ (a weight leaving a silent unit or a zero feature cannot change the loss on this example). In the worked example below, $\delta_1 = (0, -1)^\top$ and $x = (1, 2)^\top$ give $\partial\mathcal{L}/\partial W_1$ the rows $(0, 0)$ and $(-1, -2)$, and $a_1 = (0, 2)^\top$ makes the first entry of $\partial\mathcal{L}/\partial W_2 = (0, -4)$ zero (Figure 4(b)).

*The output layer.* Every backward pass starts from $\delta_L$, and for the standard pairings of output and loss it is "prediction minus target". A linear output with $\mathcal{L} = \tfrac12\lVert\hat y - y\rVert^2$ gives $\delta_L = \hat y - y$ (the $\tfrac12$ cancels the 2 of the square); softmax with cross-entropy gives $\delta_L = p - y$ (Section 6); a sigmoid output with binary cross-entropy gives $\delta_L = \sigma(z_L) - y$. In each pairing the derivative of the output nonlinearity cancels against the derivative of the loss. Pair a sigmoid with squared error instead and the factor $\sigma'(z_L) \le 1/4$ survives, weakening the signal most when the prediction is confidently wrong ([Module 04](../04-differential-calculus/README.md), Figure 7).

*Three lines become a loop.* With every $z_l$ and $a_l$ stored by the forward pass:

```text
delta = dLoss/dz_L                                  # prediction minus target
for l = L, L-1, ..., 1:
    gW[l] = outer(delta, a[l-1])                    # reads the stored a_{l-1}
    gb[l] = delta
    if l > 1:
        delta = (W[l].T @ delta) * fprime(z[l-1])   # reads W_l and the stored z_{l-1}
```

The weights are updated only after the loop: the line that computes the next $\delta$ reads $W_l$, so stepping $W_l$ before that line has run would send the error back through the wrong weights. Frameworks enforce this order by separating `loss.backward()` from `optimizer.step()`.

*Minibatches.* Real code stores a batch of $B$ examples as the rows of a matrix: $A_{l-1}$ is $B \times n_{l-1}$, the layer computes $Z_l = A_{l-1} W_l^\top + \mathbf{1} b_l^\top$, and $\Delta_l = \partial \mathcal{L}/\partial Z_l$ is $B \times n_l$, with one row $\delta_l^\top$ per example. The rules of Section 6 turn the recurrence into

$$
\bar W_l = \Delta_l^\top A_{l-1} = \sum_{\text{examples}} \delta_l\, a_{l-1}^\top, \qquad
\bar b_l = \Delta_l^\top \mathbf{1}, \qquad
\Delta_{l-1} = (\Delta_l W_l) \odot f'(Z_{l-1}),
$$

so the weight gradient is the sum of the per-example outer products and the bias gradient the sum of the per-example $\delta$s. A loss averaged over the batch puts a factor $1/B$ into every row of $\Delta_L$, and from there into every gradient (Section 7, pitfall (5)). [Module 16](../../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md), Section 5, writes the same recurrence with each weight matrix stored the other way round, as (inputs × outputs).

*How to test.* Assert that every gradient has the shape of its parameter ($\bar W_l$ is $n_l \times n_{l-1}$, like $W_l$), and compare a few entries with central differences (Section 7), or the whole recurrence with a scalar engine (Section 5), on a tiny network; they must agree to rounding.

**Worked example.** A 2–2–1 network with ReLU hidden units, linear output and loss $\mathcal{L} = \tfrac12 (\hat y - y)^2$. Input $x = (1, 2)^\top$, target $y = 3$,

$$
W_1 = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix},\ b_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix},\qquad
W_2 = \begin{pmatrix} 1 & 0.5 \end{pmatrix},\ b_2 = 0 .
$$

Forward: $z_1 = W_1 x = (1 - 2,\ 0 + 2)^\top = (-1, 2)^\top$; $a_1 = \mathrm{ReLU}(z_1) = (0, 2)^\top$; $z_2 = W_2 a_1 = 0 \cdot 1 + 2 \cdot 0.5 = 1 = \hat y$; $\mathcal{L} = \tfrac12 (1 - 3)^2 = 2$.

Backward: $\delta_2 = \hat y - y = -2$. Then $\partial \mathcal{L} / \partial W_2 = \delta_2 a_1^\top = (0,\ -4)$ and $\partial \mathcal{L} / \partial b_2 = -2$. Next, $W_2^\top \delta_2 = (-2, -1)^\top$ and $f'(z_1) = (0, 1)^\top$ (the first unit is off), so $\delta_1 = (0, -1)^\top$. Finally

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \delta_1 x^\top = \begin{pmatrix} 0 & 0 \\ -1 & -2 \end{pmatrix}, \qquad
\frac{\partial \mathcal{L}}{\partial b_1} = \begin{pmatrix} 0 \\ -1 \end{pmatrix}.
$$

The dead ReLU unit receives zero gradient in both its weights and its bias: the "dying ReLU" problem in miniature. Figure 4 draws both passes.

![Two stacked diagram panels of a network with two inputs, two ReLU hidden units and one linear output. Top: the forward pass along green arrows labelled with the weights; inputs x1 = 1 and x2 = 2, hidden unit 1 with pre-activation -1 and activation 0, hidden unit 2 with pre-activation 2 and activation 2, output y-hat = 0 times 1 plus 2 times 0.5 = 1, and loss one half of (1 - 3) squared = 2. Bottom: the backward pass along dashed arrows from the loss box with adjoint 1; the output error -2 goes back through the second-layer weights as -2 and -1; the first hidden unit, drawn dotted and grey, is masked to 0, the second passes -1; the arrows carry the weight gradients 0 and -4 for the second layer, 0 and 0 for the weights into the dead unit, and -1 and -2 for the weights into the live one.](figures/backprop-2-2-1.svg)

**Figure 4.** The 2–2–1 worked example. (a) The forward pass, with the entries of $W_1$ and $W_2$ on the edges: $z_1 = (-1, 2)$, $a_1 = (0, 2)$, $\hat y = 1$, $\mathcal{L} = 2$; the first hidden unit is off. (b) The backward pass: $\delta_2 = \hat y - y = -2$ comes back through $W_2^\top$ as $(-2, -1)$, is masked by $f'(z_1) = (0, 1)$ and becomes $\delta_1 = (0, -1)$. Each edge label is a weight gradient, the $\delta$ of the edge's child times the stored activation of its parent: $\partial\mathcal{L}/\partial W_2 = (0, -4)$, zero for the weight leaving the dead unit because $a_{1,1} = 0$; and $\partial\mathcal{L}/\partial W_1$ with rows $(0, 0)$ and $(-1, -2)$, zero for both weights into the dead unit because $\delta_{1,1} = 0$. The bias gradients equal the $\delta$s.

**Memory.** The loop above reads $a_{l-1}$ (for the weight gradient) and $z_l$ or its mask $f'(z_l)$ of every layer — the values written in the boxes of Figure 4(a) — so all activations of the forward pass must be kept until the backward pass reaches them. Per layer that is $B \times n_{l-1}$ inputs and $B \times n_l$ masks, against $n_l \times n_{l-1}$ weights shared by all examples. Training memory is therefore proportional to depth × batch size × width, and in large models it dwarfs the weights (Figure 8 and the first case study). *Checkpointing* (Chen et al., 2016) stores only every $k$-th layer's activations and recomputes the rest during the backward pass. With $k = \sqrt{L}$ it keeps $\sqrt{L}$ checkpoints, and the backward pass recomputes one segment of $\sqrt{L}$ layers at a time from its checkpoint, so about $2\sqrt{L}$ layers' worth of activations are alive at once — 20 instead of 100 for $L = 100$. The trade is about one extra forward pass for memory that grows like $O(\sqrt{L})$ instead of $O(L)$.

**Real-life example (healthcare).** A chest X-ray classifier is trained by exactly this recurrence, with convolutions in place of dense matrices. Run the backward sweep one step further than usual — past the first layer, to the input pixels — and you get $\partial \mathcal{L} / \partial x$, a *saliency map* showing which pixels most influence the "pneumonia" score. Radiology teams use such maps to check that a model attends to the lung field rather than to a hospital's text label in the image corner, a shortcut that has derailed real deployments.

**In ML/AI this is …** the backward kernels of `torch.nn.Linear` and `torch.nn.Conv2d`; `torch.utils.checkpoint` and the "activation recomputation" flags of Megatron-LM and DeepSpeed; `.requires_grad_()` on an input image to get saliency maps.

## 5. A scalar autograd engine from scratch

**Intuition.** Make every number an object that remembers (a) its value, (b) the objects it was computed from, and (c) a tiny function that knows how to pass a gradient to those parents. Overload `+`, `*`, `**` and a few elementary functions so that ordinary-looking arithmetic silently builds the graph. Then `backward()` is: topologically sort the graph, set the output's gradient to 1, and call every node's tiny function in reverse order. This is the design of Karpathy's `micrograd`, and in essence of PyTorch's autograd.

**Definition (the engine).** A `Value` holds `data` (float), `grad` (float, initially 0), `_parents` (tuple of `Value`s) and `_backward` (a closure). An operation creates the output node and defines its closure using the *local derivative rule*:

| operation | forward | closure (`g` = `out.grad`) |
|---|---|---|
| `a + b` | $a + b$ | `a.grad += g; b.grad += g` |
| `a * b` | $ab$ | `a.grad += b.data * g; b.grad += a.data * g` |
| `a ** k` | $a^k$ | `a.grad += k * a.data**(k-1) * g` |
| `a.exp()` | $e^a$ | `a.grad += out.data * g` |
| `a.log()` | $\ln a$ | `a.grad += g / a.data` |
| `a.tanh()` | $\tanh a$ | `a.grad += (1 - out.data**2) * g` |
| `a.relu()` | $\max(a, 0)$ | `a.grad += (a.data > 0) * g` |

`backward()` builds the topological order by depth-first search (append a node after visiting its parents), sets `self.grad = 1.0`, and calls `_backward` for each node in reversed order. Keeping everything scalar sidesteps broadcasting and shape bookkeeping; the cost is speed, which is why Section 6 moves to arrays.

**Worked example.** Run Section 2's graph through the engine: $w = 2, x = 3, b = 1, y = 5$, `L = (w*x + b - y)**2`. Reverse topological order and closure calls:

1. $L = v_3^2$: $\bar v_3 \mathrel{+}= 2 v_3 \cdot \bar L = 2 \cdot 2 \cdot 1 = 4$.
2. $v_3 = v_2 - y$ (implemented as $v_2 + (-1)\,y$): $\bar v_2 \mathrel{+}= 4$, $\bar y \mathrel{+}= -4$.
3. $v_2 = v_1 + b$: $\bar v_1 \mathrel{+}= 4$, $\bar b \mathrel{+}= 4$.
4. $v_1 = w x$: $\bar w \mathrel{+}= x \cdot 4 = 12$, $\bar x \mathrel{+}= w \cdot 4 = 8$.

These four calls are Figure 1(b) read from right to left. Check: $\partial L / \partial w = 2(wx + b - y)\, x = 2 \cdot 2 \cdot 3 = 12$ and $\partial L / \partial b = 2 \cdot 2 = 4$. A gradient-descent step with learning rate $0.05$ moves $w$ to $2 - 0.6 = 1.4$ and $b$ to $0.8$; the new prediction $1.4 \cdot 3 + 0.8 = 5.0$ hits the target exactly, because for a single data point the step $\eta = 1/(2(x^2 + 1))$ is the one that zeroes the residual.

Two details matter in practice. Gradients accumulate with `+=`, so you must reset every parameter's `grad` to zero before each `backward()`, otherwise step two's gradient includes step one's. And each closure must use the right quantity: the derivative of $\tanh$ is cheapest through the *output* $\tanh(a)$, while the derivative of $\ln$ needs the *input* $a$.

**Real-life example (sport).** The Elo rating system predicts that player A beats player B with probability $E_A = 1 / (1 + 10^{(R_B - R_A)/400})$ and updates $R_A \leftarrow R_A + K (S_A - E_A)$ after a match with outcome $S_A \in \{0, 1\}$. Build the log-loss $-[S_A \ln E_A + (1 - S_A)\ln(1 - E_A)]$ out of `Value`s with the ratings as parameters and call `backward()`: the gradient with respect to $R_A$ is $-(S_A - E_A)\ln(10)/400$, so one SGD step with learning rate $\eta$ is exactly Elo's update with $K = \eta \ln(10)/400$. With $R_A = 1600$, $R_B = 1400$ we get $E_A = 0.760$; if A loses, $K = 32$ moves A down by $32 \times 0.760 = 24.3$ points. Elo's system, adopted by the US Chess Federation in 1960, is SGD on a logistic model, and autograd rediscovers its update rule mechanically.

**In ML/AI this is …** `torch.Tensor` with `requires_grad=True`, `grad_fn` (the closure), `optimizer.zero_grad()` (the reset), and `torch.autograd.Function` for custom operations with hand-written `forward` and `backward` static methods.

## 6. Vector and matrix autograd: vector-Jacobian products

**Intuition.** Real networks operate on arrays, and a scalar engine would create a node per matrix element — a 1,000 × 1,000 matmul would spawn a billion nodes. Instead, treat each array operation as one node whose backward rule maps the *upstream gradient* $G = \partial L / \partial Y$ (same shape as the output $Y$) to the gradient with respect to each input (same shape as that input). Mathematically this is a vector-Jacobian product, $\bar x = J^\top \bar y$, but you never form the Jacobian: for a matmul the Jacobian has $(mn)(mk)$ entries while the rule needs only one more matmul.

**Definition (the four rules you use every day).** With $G = \partial L / \partial Y$:

| operation | forward | backward rule |
|---|---|---|
| matrix product | $Y = AB$, $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$ | $\bar A = G B^\top$, $\quad \bar B = A^\top G$ |
| elementwise function | $Y = f(X)$ | $\bar X = G \odot f'(X)$ |
| broadcast bias | $Y = Z + \mathbf{1} b^\top$ (add $b$ to every row) | $\bar Z = G$, $\quad \bar b = G^\top \mathbf{1}$ (sum over the broadcast axis) |
| sum | $s = \sum_{ij} X_{ij}$ | $\bar X = \bar s\, \mathbf{1}\mathbf{1}^\top$ |

*From scalar adjoints to arrays.* An array operation is many scalar operations at once, and its backward rule is Section 3's adjoint rule applied to every entry: for each input entry, add up $G$ times the local derivative over every output entry it feeds. For $Y = AB$, $Y_{ij} = \sum_k A_{ik} B_{kj}$. The entry $A_{ik}$ appears in $Y_{ij}$ for every column $j$, with coefficient $B_{kj}$, so

$$
\bar A_{ik} = \sum_j G_{ij}\, B_{kj} = (G B^\top)_{ik}.
$$

The sum over $j$ is the `+=` of a node with $n$ children, written as a matrix product. The same argument gives $\bar B = A^\top G$ (self-assessment 6), and the worked example below checks one entry by hand (Figure 5(a)).

*The shape test.* Shape bookkeeping is the sanity check, and often the derivation itself: $\bar A$ must have $A$'s shape $m \times k$, and $G B^\top$ is $(m \times n)(n \times k)$. When $m$, $k$ and $n$ differ, every other arrangement of $G$ and $B$ either fails to multiply or has the wrong shape; the tempting guess $B^\top G$ cannot even be formed unless $k = m$ (Figure 5(b)). Three small rules follow from the same entry-by-entry argument: $Y = X^\top$ gives $\bar X = G^\top$; $Y = cX$ gives $\bar X = cG$; and the mean $s = \frac{1}{N}\sum_{ij} X_{ij}$ of $N$ entries gives $\bar X = (\bar s / N)\, \mathbf{1}\mathbf{1}^\top$.

*Broadcasting becomes summation.* Broadcasting in the forward pass always becomes a *sum* in the backward pass, because the broadcast value was used once per copy and each use contributes. A bias row $b$ added to every row of a matrix is used once per row, so $\bar b_j = \sum_i G_{ij}$, i.e. $\bar b = G^\top \mathbf{1}$ — `G.sum(axis=0)` in NumPy (Figure 5(c)). In general, sum $G$ over every axis along which the forward pass broadcast, keeping any axis of size 1 that was stretched (`keepdims=True`), so that the result has the shape of the original. When the rows are the examples of a batch, the bias gradient is therefore the sum of the per-example gradients (Section 4, *Minibatches*).

*Elementwise functions have diagonal Jacobians.* $Y = f(X)$ applies one scalar function to each entry, so $Y_{ij}$ depends on $X_{ij}$ alone and the Jacobian is diagonal, with entries $f'(X_{ij})$. Multiplying by a diagonal matrix is an elementwise product, so $\bar X = G \odot f'(X)$ and no matrix is ever formed: the ReLU mask $\mathbb{1}[X > 0]$, the sigmoid's $p(1 - p)$ and the $\tanh$'s $1 - t^2$ are all evaluated from stored forward values.

*Rows are examples.* Code stores a batch as a matrix $X$ with one example per row, so a linear layer computes $Z = X W^\top + \mathbf{1} b^\top$ (this is `nn.Linear`). The matmul, transpose and broadcast rules give, with $G = \partial \mathcal{L}/\partial Z$,

$$
\bar W = G^\top X, \qquad \bar X = G W, \qquad \bar b = G^\top \mathbf{1},
$$

which are Section 4's $\delta_l a_{l-1}^\top$ and $\delta_l$ summed over the batch, and $W_l^\top \delta_l$ written as one row per example. For Section 4's 2–2–1 network as a batch of one, $X = (1 \ \ 2)$ and $G = (0 \ \ {-1})$ give $\bar W_1 = G^\top X$ with rows $(0, 0)$ and $(-1, -2)$, as before, and $\bar X = G W_1 = (0, -1)$: the gradient with respect to the input, the saliency of Section 4's healthcare example.

*A new rule, derived.* The method handles operations the table does not list. Attention scores $S = QK^\top$ (before Module 17's scaling by $1/\sqrt{d_k}$, which only rescales $G$) follow from the matmul and transpose rules: $\bar Q = G K$ and $\bar K = G^\top Q$. When the same matrix plays both roles, $S = XX^\top$, $X$ is used twice and its gradient is the sum of both contributions, $\bar X = GX + G^\top X = (G + G^\top)X$. With $X = A$ from the worked example below and $G$ all ones, $L = \sum_{ij}(XX^\top)_{ij} = 52$ and $\bar X$ has rows $(8, 12)$ and $(8, 12)$, which central differences confirm. [Module 17](../../part-6-deep-learning-and-modern-ai/17-mathematics-of-transformers/README.md), Section 4, derives the full attention backward pass this way.

*How to test.* Assert in every backward rule that each gradient has its input's shape, and compare a few entries with central differences (Section 7); `torch.autograd.gradcheck` does the comparison for a custom `autograd.Function`.

**Fused softmax-cross-entropy.** For logits $z \in \mathbb{R}^K$, $p = \mathrm{softmax}(z)$ and one-hot target $y$, the loss $\mathcal{L} = -\sum_k y_k \ln p_k$ has gradient

$$
\frac{\partial \mathcal{L}}{\partial z} = p - y .
$$

Derivation: $\ln p_k = z_k - \ln \sum_i e^{z_i}$, so $\partial \ln p_k / \partial z_j = \mathbb{1}[k = j] - p_j$; multiply by $-y_k$, sum over $k$ and use $\sum_k y_k = 1$ to get $-y_j + p_j$. Frameworks fuse the two operations for three reasons: the chained path through $\ln(\mathrm{softmax}(z))$ can produce $\ln 0 = -\infty$ while the fused form is finite; the fused gradient is one subtraction instead of a softmax Jacobian times a vector; and a fused kernel reads and writes the (often huge) logits tensor once instead of several times.

**Worked example.** $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$, $Y = AB = \begin{pmatrix} 3 & 2 \\ 7 & 4 \end{pmatrix}$, $L = \sum_{ij} Y_{ij} = 16$, so $G$ is the all-ones matrix. Then

$$
\bar A = G B^\top = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 1 & 2 \end{pmatrix}, \qquad
\bar B = A^\top G = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 4 & 4 \\ 6 & 6 \end{pmatrix}.
$$

Spot check by hand: $A_{12}$ appears in $Y_{11} = A_{11} B_{11} + A_{12} B_{21}$ and $Y_{12} = A_{11} B_{12} + A_{12} B_{22}$ with coefficients $B_{21} + B_{22} = 2$, matching $\bar A_{12} = 2$ (Figure 5(a)). For the softmax rule, $z = (2, 0, 0)$ gives $p = (e^2, 1, 1)/(e^2 + 2) = (0.787, 0.107, 0.107)$; with target class 2, the gradient is $p - y = (0.787, -0.893, 0.107)$: push the correct logit up, the others down, in proportion to how wrong the probabilities are.

![Three panels. Left: the products A times B equals Y and G times B transpose equals A-bar written out as 2 by 2 tables, with A = 1, 2, 3, 4, B = 1, 0, 1, 1, Y = 3, 2, 7, 4, G all ones and A-bar = 1, 2, 1, 2; the entry A12, the second row of B, the entries Y11 and Y12, the first row of G, the second column of B transpose and the entry A-bar12 = 2 are highlighted. Middle: rectangles with sides proportional to the matrix shapes, A (m by k) times B (k by n) equals Y (m by n), and G (m by n) times B transpose (n by k) equals A-bar (m by k), with a note that B transpose times G cannot be multiplied because its inner sizes differ. Right: a single bias row b copied upwards into three identical rows in the forward pass, and a three-row gradient G summed downwards into a single row b-bar in the backward pass.](figures/matmul-vjp-shapes.svg)

**Figure 5.** (a) The worked example: $Y = AB$ and $L = \sum_{ij} Y_{ij} = 16$, so $G$ is all ones and $\bar A = GB^\top$ has rows $(1, 2)$ and $(1, 2)$. The highlighted cells are the spot check: $A_{12}$ feeds $Y_{11}$ through $B_{21} = 1$ and $Y_{12}$ through $B_{22} = 1$, so $\bar A_{12} = G_{11}B_{21} + G_{12}B_{22} = 2$. (b) The shape test, with blocks drawn for $m = 2$, $k = 3$, $n = 4$ (distinct sizes chosen for the drawing, not the worked example's): only $GB^\top$, $(m \times n)(n \times k)$, has the shape $m \times k$ of $A$, and $B^\top G$ cannot be formed because its inner sizes $k$ and $m$ differ. In the $2 \times 2$ example $B^\top G$ does exist, with rows $(2, 2)$ and $(1, 1)$, but it is the wrong value. (c) Broadcasting forwards is summing backwards: $b$ is copied into every row of $\mathbf{1}b^\top$, so $\bar b_j$ collects one contribution per row, $\bar b_j = G_{1j} + G_{2j} + G_{3j}$, i.e. $\bar b = G^\top \mathbf{1}$.

**Real-life example (NLP).** A language model predicts the next token with a softmax over a vocabulary of 50,000 to 150,000 entries. The logits tensor has shape (batch × sequence × vocabulary) and is routinely the single largest activation in the model — at batch 8, sequence 1,024 and vocabulary 50,257 it holds 412 million floats, 1.6 GB in float32. The fused softmax-cross-entropy kernel computes the loss and the gradient $p - y$ in one pass over that tensor; an unfused `log(softmax(z))` followed by a separate backward would read and write it several times and risk `log(0)` for confidently wrong predictions.

**In ML/AI this is …** `torch.nn.CrossEntropyLoss` (log-softmax and negative log-likelihood fused), the `MmBackward0` node of `torch.matmul`, `np.sum` appearing in every bias gradient, and the fused attention kernels (FlashAttention) that apply the same "read the big tensor once" logic to softmax inside attention.

## 7. Gradient checking

**Intuition.** Automatic differentiation is only as correct as its local rules, and a wrong rule does not crash — it trains a slightly (or badly) wrong model. The defence is to compare the analytic gradient with a numerical one on a few parameters. Central differences are the independent witness: slow, approximate, but almost impossible to get wrong.

**Definition.** For a loss $L(\theta)$ and analytic gradient $g$, estimate component $i$ by

$$
\hat g_i = \frac{L(\theta + h e_i) - L(\theta - h e_i)}{2h}, \qquad h \approx 10^{-5} \text{ in float64},
$$

and compute the relative error

$$
r_i = \frac{|g_i - \hat g_i|}{\max(|g_i|, |\hat g_i|, 10^{-12})} \qquad \text{or, for the whole vector,} \qquad r = \frac{\|g - \hat g\|_2}{\|g\|_2 + \|\hat g\|_2}.
$$

Rules of thumb: $r < 10^{-7}$ is a correct gradient; $10^{-7}$ to $10^{-4}$ is acceptable for functions with kinks (ReLU, max, absolute value); $10^{-4}$ to $10^{-2}$ is suspicious; $r > 10^{-2}$ is a bug. Checking all $n$ parameters costs $2n$ forward passes, so check a random subset of a few dozen, on a small network, with a small batch, in float64, at *several* parameter settings, before any training.

**Worked example.** Section 5's engine on $L = (wx + b - y)^2$ at $w = 2$: analytic $\bar w = 12$ (Figure 1(b)). Numerically, $L(2 + 10^{-5}) - L(2 - 10^{-5}) = (2.00003)^2 - (1.99997)^2 = 0.00024$, so $\hat g = 0.00024 / (2 \times 10^{-5}) = 12.0000$ and $r \approx 10^{-11}$. Now plant the classic bug: the `*` closure uses `self.data` for both parents, giving $\bar w = w \cdot 4 = 8$. Then $r = |8 - 12| / 12 = 0.33$: unmistakable. A subtler bug, the $\tanh$ closure using $1 - a^2$ (input) instead of $1 - \tanh(a)^2$ (output), gives $r = 0.046$ at $a = 0.5$ but only $7 \times 10^{-5}$ at $a = 0.1$ — which is why you check at several points and not only near zero. Figure 6(b) follows this bug across inputs: between $a \approx 0.02$ and $a \approx 0.11$ it falls in the band that is acceptable only for functions with kinks, and below $a \approx 0.02$ it even scores under $10^{-7}$, in the \"correct\" band ($6.6 \times 10^{-9}$ at $a = 0.01$, the left end of the plot), which makes it easy to wave through although $\tanh$ is smooth and a correct rule scores about $10^{-11}$; only above $a \approx 0.35$ does it land in the "bug" band.

**Pitfalls.** (1) Kinks: if a ReLU input $z$ lies within $h$ of zero, the difference quotient straddles the kink and reports $(z + h)/(2h)$ — somewhere between 0 and 1, about $0.5$ close to the kink — against an analytic 0 or 1: a false alarm (Figure 7, in Section 8). (2) float32: round-off grows like $\varepsilon \lvert f \rvert / h$ with $\varepsilon \approx 1.2 \times 10^{-7}$, so with the usual $h = 10^{-5}$ even a correct rule agrees only to a few parts in a thousand (Figure 6(c): $r = 4.2 \times 10^{-3}$, in the "suspicious" band). The best float32 steps, near $\varepsilon^{1/3} \approx 5 \times 10^{-3}$, do better but still leave errors between about $3 \times 10^{-7}$ and $10^{-5}$, short of the "correct" band, so check in float64. (3) Randomness: dropout masks and data shuffling must be frozen (fix the seed or disable them), otherwise the two loss evaluations are not the same function. (4) Batch normalisation in training mode changes its statistics with every perturbation; check in evaluation mode. (5) A loss averaged over a batch has a gradient $1/B$ times that of the summed loss, so an absolute error threshold depends on the loss convention; use the relative, not the absolute, error.

![Three panels about checking the derivative of tanh. Left: the correct rule, 1 minus tanh squared of a, as a solid blue curve and the planted bug, 1 minus a squared, as a dashed red curve, for a from 0 to 1.6; they nearly coincide at a = 0.1 (0.9901 against 0.99), differ at a = 0.5 (0.7864 against 0.75) and are far apart at a = 1 (0.42 against 0). Middle: on a logarithmic scale, the relative error of each rule against the central difference, over shaded bands labelled correct, acceptable with kinks, suspicious and bug; the dashed red curve of the bug climbs through all four bands, with dots at 6.7e-5, 4.6e-2 and 1, while the thin blue curve of the correct rule stays near 1e-11. Right: the relative error of the correct rule at a = 0.5 against the step h on log-log axes, over the same bands; the blue float64 curve is a V with a noisy floor that dips to about 1e-12 (lowest 3e-13) for steps of a few times 1e-6 and reads 1.2e-11 at h = 1e-5, while the orange float32 curve stays near 1 for tiny steps, falls noisily to between about 3e-7 and 1e-5 for steps near 5e-3, never reaching the correct band, and reads 4.2e-3 at h = 1e-5.](figures/gradient-check-pitfalls.svg)

**Figure 6.** The planted $\tanh$ bug of the worked example, checked against the central difference with $h = 10^{-5}$. (a) The correct rule $1 - \tanh^2 a$ and the bug $1 - a^2$: $0.9901$ against $0.99$ at $a = 0.1$, $0.7864$ against $0.75$ at $a = 0.5$, $0.42$ against $0$ at $a = 1$. (b) The bug's relative error, on the bands of the rules of thumb above: $6.7 \times 10^{-5}$ at $a = 0.1$, $4.6 \times 10^{-2}$ at $a = 0.5$ and $1$ at $a = 1$. It crosses $10^{-4}$ only at $a \approx 0.11$ and $10^{-2}$ at $a \approx 0.35$, so a check at small inputs alone, read with the lenient $10^{-4}$ threshold, would let it through, and below $a \approx 0.02$ even the strict $10^{-7}$ threshold would; the correct rule stays near $10^{-11}$ everywhere. (c) Pitfall (2): the correct rule at $a = 0.5$ checked with every step $h$. In float64 the error is a V — round-off on the left, truncation with slope 2 on the right — and $h = 10^{-5}$ gives $r = 1.2 \times 10^{-11}$. In float32 the same correct rule scores $r = 4.2 \times 10^{-3}$ at $h = 10^{-5}$, in the "suspicious" band, and even its best steps stay above the "correct" band; at $h = 0.1$ both precisions show the same truncation error, $1.2 \times 10^{-3}$.

**Real-life example (finance).** A quant desk replaces bump-and-revalue sensitivities with hand-coded adjoint (AAD) code to cut the nightly risk run from hours to minutes. Regulators and model validators require evidence that the fast code agrees with the slow one, so the validation suite bumps a sample of risk factors by a basis point, reprices, and reports the relative difference against the adjoint sensitivities — exactly the gradient check above, on a code base worth billions in exposure. Discrepancies of $10^{-2}$ in a delta would be found before they reached a hedging decision.

**In ML/AI this is …** `torch.autograd.gradcheck` (requires float64 inputs, default `eps=1e-6`, central differences) and `gradgradcheck` for second derivatives; the gradient-check recipe in Stanford's CS231n notes; the unit tests every custom CUDA kernel or `autograd.Function` ships with.

## 8. Where autodiff goes wrong, and some advanced notes

**Intuition.** Autodiff differentiates the program you wrote, not the function you meant. Wherever the program has a kink, a jump, an overflow or a very long chain of multiplications, the gradient it returns is formally defined but may be useless. Knowing the four failure modes and their fixes is most of what separates a smooth training run from a mysterious one.

**Definitions and facts.**

*Non-differentiable points.* ReLU, $|x|$, $\max$ and clipping have kinks; at a kink any value in the subgradient interval is defensible, and frameworks pick a convention (PyTorch uses $\mathrm{relu}'(0) = 0$). Since a kink is hit with probability zero for continuous inputs, training is unaffected. Piecewise-*constant* operations — rounding, $\arg\max$, sampling a discrete token — have derivative zero almost everywhere, so no gradient reaches anything upstream. The *straight-through estimator* (Bengio, Léonard and Courville, 2013) uses the hard operation in the forward pass and pretends it was the identity (or a smooth surrogate) in the backward pass. It is biased but works remarkably well.

*Numerical stability.* $\ln \sum_i e^{z_i}$ overflows when any $z_i > 709$ in float64 (about $88$ in float32). The log-sum-exp trick subtracts the maximum: with $m = \max_i z_i$,

$$
\ln \sum_i e^{z_i} = m + \ln \sum_i e^{z_i - m},
$$

and every exponent is now $\le 0$. Log-softmax is $z - \mathrm{LSE}(z)$, computed without ever forming $p$.

*Exploding and vanishing gradients.* Through $L$ layers the adjoint is a product $\delta_1 = \big(\prod_{l=1}^{L-1} \mathrm{diag}(f'(z_l))\, W_{l+1}^\top\big)\, \delta_L$, with the factors ordered by increasing $l$ from left to right. With sigmoid activations $f' \le 1/4$, so whenever every weight matrix has spectral norm at most 1 the gradient is multiplied by at most $4^{-(L-1)}$ on its way from $\delta_L$ to $\delta_1$ (counted from $\partial\mathcal{L}/\partial a_L$ instead, a sigmoid output unit adds one more factor $\sigma'(z_L) \le 1/4$, as in worked example (3) below; with binary cross-entropy that factor cancels, Section 4); weights of norm larger than 4 can make it explode instead. Remedies — careful initialisation, normalisation layers, residual connections and gradient clipping — are the subject of [Module 16](../../part-6-deep-learning-and-modern-ai/16-neural-networks-from-scratch/README.md).

*Higher-order derivatives.* The backward pass is itself made of differentiable operations, so you can differentiate the gradient ("double backward"). The Hessian-vector product $Hv = \nabla_\theta\big(\nabla_\theta L \cdot v\big)$ costs about two backward passes and never forms the $n \times n$ Hessian (Pearlmutter, 1994); it powers Newton-CG optimisers, curvature estimates and influence functions.

**Worked example.** Three quick numbers. (1) Kink: gradient-checking ReLU at $z = 10^{-6}$ with $h = 10^{-5}$ gives $\hat g = (1.1 \times 10^{-5} - 0) / (2 \times 10^{-5}) = 0.55$ against the analytic $1$ — a false alarm, not a bug (Figure 7). (2) Overflow: for $z = (1000, 1000)$ the naive computation gives `exp(1000) = inf` and a NaN softmax, while the stable form gives $1000 + \ln 2 = 1000.693$. (3) Vanishing: a chain of ten sigmoid units with unit weights contributes ten factors $\sigma' \le 1/4$, one per unit, so it scales the gradient by at most $4^{-10} \approx 9.5 \times 10^{-7}$; a loss gradient of $1$ at the output arrives at the input as less than $10^{-6}$.

![Two panels. Left: the ReLU function near zero, with both axes in units of 1e-5: a blue line flat at 0 for negative z and rising with slope 1 for positive z; a shaded window from -0.9 to 1.1 around the check point z0 = 1e-6, which is marked by a dotted vertical line; red dots at the two evaluation points, z0 - h with value 0 and z0 + h with value 1.1, joined by a dashed red secant of slope 0.55. Right: the derivative of ReLU against the check point z in units of h: the blue analytic step, 0 up to and including z = 0 and 1 beyond, and the dashed red central-difference estimate, 0 below -h, a straight ramp from 0 to 1 across the shaded window from -h to h, and 1 above h; an open circle marks the top of the jump; dots mark the estimate 0.5 against the analytic 0 at z = 0, and at z0 = 0.1 h a thin black segment joins the estimate 0.55 to the analytic 1.](figures/relu-kink.svg)

**Figure 7.** Worked example (1): a correct ReLU rule "fails" a gradient check. (a) With $z_0 = 10^{-6}$ and $h = 10^{-5}$ the two evaluations $z_0 \pm h$ fall on opposite sides of the kink, at $-0.9 \times 10^{-5}$ (where ReLU is 0) and $1.1 \times 10^{-5}$, so the central difference measures the secant slope $(1.1 - 0)/2 = 0.55$ instead of the true slope 1. (b) The same estimate for every check point $z$. Outside the window $|z| < h$ it is exact; inside, it is the ramp $(z + h)/(2h)$. Just above zero it reads about $0.5$ against the analytic 1, a relative error close to $0.5$ (self-assessment 7); at $z_0 = 0.1h$ it reads $0.55$, relative error $0.45$; and anywhere in $-h < z \le 0$, including $z = 0$ with PyTorch's convention $\mathrm{relu}'(0) = 0$, it is compared with an analytic 0, so the relative error is 1.

**Real-life example (computer vision).** An image classifier shipped to phones must run with 8-bit integer weights. Quantisation-aware training inserts a rounding operation into the forward pass so that the network learns to live with the quantisation error. Rounding has zero derivative everywhere, so the straight-through estimator lets the gradient flow as if rounding were the identity; the full-precision shadow weights are updated and re-rounded each step. This is how models such as MobileNet are quantised with accuracy loss of about a percentage point rather than tens of points.

**In ML/AI this is …** `torch.logsumexp`, `F.log_softmax` and the stable `BCEWithLogitsLoss`; the `.detach()` idiom `x_hard + x_soft - x_soft.detach()` that implements straight-through in VQ-VAE and Gumbel-softmax code; `torch.nn.utils.clip_grad_norm_`; and `torch.autograd.grad(..., create_graph=True)` or `jax.hessian` for second-order quantities.

## Common pitfalls and misconceptions

1. **"Backprop is just finite differences."** → Finite differences cost $n$ evaluations and are approximate; backprop costs a constant multiple of one evaluation and is exact to rounding (Figure 2(b)). → Use finite differences only to *check* backprop.
2. **Forgetting to zero gradients.** → Every autograd engine accumulates with `+=` so that shared nodes receive all their contributions; stale gradients from the previous step then leak into the current one. → Reset `grad` on every parameter before each `backward()` (`optimizer.zero_grad()`).
3. **Writing `=` instead of `+=` in a backward closure.** → A node with two children receives only the last contribution; Section 3's $\bar x_1$ would come out as 16 or 48 instead of 64 (Figure 3(c)). → Always accumulate; gradient-check expressions where a variable is reused.
4. **Using the input where the output belongs, or vice versa.** → $\tanh'$ and $\exp'$ are cheapest through the *output* ($1 - t^2$, $e^a$), $\ln'$ and power rules need the *input*. → Derive each closure from the formula, then check it numerically at several points (the $\tanh$ bug of Figure 6 scores below $10^{-4}$ at small inputs, where a lenient check lets it through).
5. **Trusting a gradient check done in float32 or at a kink.** → With the usual $h = 10^{-5}$, round-off limits float32 agreement to a few parts in a thousand ($r = 4.2 \times 10^{-3}$, Figure 6(c)), and within $h$ of a kink correct code can produce relative errors of $0.5$ or even $1$ (Figure 7). → Check in float64, away from kinks, and interpret relative error with the thresholds in Section 7.
6. **Guessing matrix backward rules by pattern matching.** → $\bar A = B^\top G$ has the wrong shape or the wrong value; only $G B^\top$ is correct (Figure 5(b)). → Derive from a scalar entry or trust the shape test: the result must have the shape of the input it is the gradient for.
7. **Computing `log(softmax(z))` and wondering about NaNs.** → `exp` overflows above 709 and `log(0)` is $-\infty$. → Use log-sum-exp with the max subtracted; call the fused loss.
8. **Expecting a gradient through `argmax`, rounding or sampling.** → These are piecewise constant; their derivative is zero almost everywhere, so upstream parameters never move. → Use a soft relaxation (softmax with temperature, Gumbel-softmax) or a straight-through estimator.

## Real-world case studies

### Case study 1: why training a transformer is feasible, and why memory, not compute, is the first wall

GPT-2 small has $N = 124$ million parameters, 12 layers, hidden size $h = 768$, $a = 12$ attention heads and a context of $s = 1{,}024$ tokens. A forward pass costs about $2N$ floating-point operations per token, so a batch of $b = 8$ sequences costs $2 \times 1.24 \times 10^8 \times 8{,}192 \approx 2.0 \times 10^{12}$ FLOPs. Estimating the gradient with central differences would need $2N \approx 2.5 \times 10^8$ such passes — about $5 \times 10^{20}$ FLOPs per optimisation step, roughly an hour on a thousand-GPU cluster, for a single step of a model that needs hundreds of thousands of steps. Reverse-mode AD does it in about three forward-pass equivalents, $6 \times 10^{12}$ FLOPs: a factor of $10^8$ cheaper. That is the counting argument of Section 3 with real numbers (the round markers of Figure 2(b)).

The price is memory. Every activation needed by a backward rule must be stored: the inputs to each matmul (for $\bar W = G A^\top$), the softmax probabilities inside attention, the masks of the nonlinearities. A careful count for a standard transformer layer in 16-bit precision (Korthikanti et al., 2022) gives about $s b h \,(34 + 5 a s / h)$ bytes of activations per layer. For the numbers above, $a s / h = 16$, so the bracket is $114$, and $s b h = 6.3 \times 10^6$: about $0.72$ GB per layer, $8.6$ GB for all 12 layers. The weights themselves occupy $124 \times 10^6 \times 2$ bytes $= 0.25$ GB. Activations outweigh weights 35 to 1, and they scale with batch size and with the *square* of the sequence length through the attention term (Figure 8).

![Two panels. Left: horizontal bars in gigabytes for GPT-2 small at batch 8, context 1,024 and 16-bit precision: the activations kept for the backward pass, 8.6 GB, stacked from 2.57 GB of linear terms in orange and 6.04 GB of attention terms in purple; the weights, 0.25 GB, in blue; and the activations kept with checkpointing, 0.15 GB, in green, with a hatched extension to the 0.87 GB peak while one layer is recomputed. Right: a log-log plot of gigabytes for 12 layers against the sequence length from 100 to 10,000 at batch 8: the red total activation curve rises from 0.42 GB at length 128 to about 400 GB at 8,192; it is the sum of an orange dashed part that grows in proportion to the length and a purple dotted part that grows with its square and takes over at long contexts; the green checkpointing line stays far lower, and the blue weights line is flat at 0.25 GB; a black dot marks the case study, length 1,024 and 8.6 GB.](figures/activation-memory.svg)

**Figure 8.** Activation memory of GPT-2 small from the per-layer estimate $sbh\,(34 + 5as/h)$ bytes in 16-bit precision. (a) At batch $b = 8$ and context $s = 1{,}024$: $8.6$ GB of activations for 12 layers — $2.57$ GB from the terms linear in $s$ and $6.04$ GB (70 %) from the attention term — against $0.25$ GB of weights, a ratio of 35. Keeping only each layer's input ($2sbh$ bytes per layer) stores $0.15$ GB, 57 times less, at the cost of one extra forward pass; while the backward pass recomputes a layer, that layer's own $0.72$ GB is alive as well, so the peak is about $0.87$ GB, still ten times less. (b) The same estimate as the context grows, at batch 8: the linear terms grow like $s$ and the attention terms like $s^2$, overtaking them beyond $s \approx 435$. The total runs from $0.42$ GB at $s = 128$ to about $400$ GB at $s = 8{,}192$ — the formula extrapolated beyond GPT-2's context of 1,024 — while the weights stay at $0.25$ GB.

Checkpointing changes the trade. Storing only each layer's input ($2 s b h \approx 12.6$ MB per layer, $0.15$ GB in total) and recomputing the inside of the layer during the backward pass cuts the stored activations by a factor of about 57 (the peak, which also holds the one layer being recomputed, by about 10) at the cost of one extra forward pass, roughly a third more compute. Practitioners tune this knob daily under names such as "activation recomputation" and "gradient checkpointing"; the underlying fact is simply that reverse mode must remember the forward pass.

### Case study 2: adjoint Greeks on a derivatives book

A bank's exotic-rates desk prices its book with a Monte Carlo model driven by $n = 120$ market inputs (forward rates, volatilities, correlations). One pricing of a trade takes 2 seconds. Risk management needs the sensitivity of every trade's price to every input — the deltas and vegas used to hedge — every night, for 5,000 trades.

Bump-and-revalue (finite differences) costs $n + 1 = 121$ pricings per trade with forward differences, $2n = 240$ with the more accurate central differences: $5{,}000 \times 240 \times 2\,\mathrm{s} = 2.4 \times 10^6$ seconds, 28 CPU-days per night. The bumps also interact badly with Monte Carlo noise: the difference of two noisy prices divided by a small bump is itself noisy unless the same random numbers are reused, and discontinuous payoffs (digital options) bias the bumped estimate.

Adjoint algorithmic differentiation (Giles and Glasserman, 2006) applies reverse mode to the pricing code: one forward simulation records the path-wise computations, one backward sweep propagates $\bar{\text{price}} = 1$ to all 120 inputs. The cost is a fixed multiple of one pricing, typically three to four times, independent of $n$: $5{,}000 \times 8\,\mathrm{s} = 11$ CPU-hours instead of 28 CPU-days, a 60-fold speed-up at $n = 120$ (the square markers of Figure 2(b)) that grows linearly with the number of risk factors. The sensitivities are exact derivatives of the simulated price, so they inherit no bump noise. The two caveats are the ones from Sections 4 and 8: the backward sweep must store or recompute every path's intermediate values (memory), and payoff discontinuities need smoothing because the derivative of a step function is zero almost everywhere (kinks). Speed-up and caveats alike are the mathematics of this module with "parameters" renamed "risk factors".

### Case study 3: finding a planted gradient bug in thirty seconds

A team writes its own `tanh` activation with a backward rule for a tiny recurrent model and trains it for two days; the loss plateaus at a mediocre value. Someone finally runs a gradient check at the initial parameters: relative errors of $4.6 \times 10^{-2}$ on some weights, $10^{-11}$ on others. Weights feeding the `tanh` are the suspicious ones. Testing the unit on its own at input $a = 0.5$: analytic rule says $0.75$; central difference says $(\tanh(0.50001) - \tanh(0.49999)) / (2 \times 10^{-5}) = 0.7864$. The rule had been coded as $1 - a^2$ — the input squared — instead of $1 - \tanh(a)^2$. The two agree near zero ($0.99$ vs $0.9901$ at $a = 0.1$) and differ wildly for larger inputs ($0$ vs $0.42$ at $a = 1$) — Figure 6(a) and (b) — which explains both why the bug was invisible in a quick check at small weights and why training stalled once activations grew: hidden units with larger pre-activations received almost no gradient. The fix is one character; the lesson is to gradient-check every hand-written rule at several input scales before training, and to make the check part of the test suite.

## Where this shows up in ML, DL and AI

| Concept | Where it appears | Why it matters |
|---|---|---|
| Finite differences | `torch.autograd.gradcheck`, unit tests for custom ops | Independent witness for every hand-written gradient |
| Computational graph | `grad_fn` chains in PyTorch, `jaxpr`, ONNX graphs, TF graphs | The data structure that makes reverse mode possible; also used for compilation and fusion |
| Forward mode / JVP | `jax.jvp`, `torch.func.jvp`, forward-over-reverse Hessian-vector products | Cheap when inputs are few or outputs many; second-order methods |
| Reverse mode / VJP | `loss.backward()`, `jax.grad`, `jax.vjp` | One sweep gives all parameter gradients: the cheap gradient principle |
| Backprop recurrence $\delta_l$ | Backward kernels of linear, convolution and attention layers | Explains dead ReLUs, vanishing gradients, saliency maps |
| Activation memory and checkpointing | `torch.utils.checkpoint`, activation recomputation in LLM training | Memory, not FLOPs, is the first limit on batch size and context length |
| Gradient accumulation (`+=`) | `optimizer.zero_grad()`, gradient accumulation over micro-batches | Shared nodes need sums; forgetting the reset silently corrupts training |
| Matmul and broadcast rules | `MmBackward0`, every bias gradient being a `sum` | Lets frameworks never form a Jacobian |
| Batched linear-layer rules, $\bar W = G^\top X$, $\bar b = G^\top \mathbf{1}$ | The backward of `nn.Linear` and of the attention projections | A weight gradient is a sum of per-example outer products, so a mean loss scales it by $1/B$ |
| Fused softmax-cross-entropy, $p - y$ | `nn.CrossEntropyLoss`, language-model output layers | Stability and a single pass over the largest tensor in the model |
| Log-sum-exp | `torch.logsumexp`, `log_softmax`, `BCEWithLogitsLoss` | Prevents overflow and `log(0)` in every classification loss |
| Straight-through estimator | VQ-VAE codebooks, quantisation-aware training, Gumbel-softmax | Gradients through discrete choices |
| Double backward | `create_graph=True`, `jax.hessian`, meta-learning (MAML), influence functions | Curvature and second-order information without an $n \times n$ Hessian |

## Cheat sheet

| Item | Formula or fact |
|---|---|
| Forward difference | $\frac{f(x+h) - f(x)}{h}$, error $O(h)$, best $h \approx \sqrt{\varepsilon} \approx 10^{-8}$ |
| Central difference | $\frac{f(x+h) - f(x-h)}{2h}$, error $O(h^2)$, best $h \approx \varepsilon^{1/3} \approx 10^{-5}$ |
| Chain rule on a graph | $\partial f / \partial x$ = sum over all paths from $x$ to $f$ of the product of the local derivatives along the path |
| Forward-mode tangent | $\dot v_i = \sum_{k \in \mathrm{pa}(i)} \frac{\partial \phi_i}{\partial v_k} \dot v_k$, one sweep per input |
| Reverse-mode adjoint | $\bar v_k = \sum_{i \in \mathrm{ch}(k)} \bar v_i \frac{\partial \phi_i}{\partial v_k}$, one sweep per output |
| JVP and VJP | forward sweep: $\dot{\mathbf y} = J\dot{\mathbf x}$ (seed $\mathbf e_j$ gives column $j$); reverse sweep: $\bar{\mathbf x} = J^\top \bar{\mathbf y}$ (seed $\mathbf e_i$ gives row $i$) |
| Jacobian cost, $f: \mathbb{R}^n \to \mathbb{R}^m$ | $\approx n$ forward sweeps or $m$ reverse sweeps, each a small constant times $f$ |
| Dual numbers | $(a, \dot a)(b, \dot b) = (ab, a\dot b + \dot a b)$, $\ g(a, \dot a) = (g(a), g'(a)\dot a)$ |
| Backprop recurrence | $\delta_l = (W_{l+1}^\top \delta_{l+1}) \odot f'(z_l)$, $\ \partial \mathcal{L}/\partial W_l = \delta_l a_{l-1}^\top$, $\ \partial \mathcal{L}/\partial b_l = \delta_l$ |
| Output layer $\delta_L$ | prediction minus target: $\hat y - y$ (linear output, $\mathcal{L} = \tfrac12 \lVert \hat y - y \rVert^2$), $p - y$ (softmax, cross-entropy), $\sigma(z_L) - y$ (sigmoid, binary cross-entropy) |
| Matmul VJP | $Y = AB$: $\bar A = G B^\top$, $\bar B = A^\top G$ |
| Batched linear layer (rows are examples) | $Z = XW^\top + \mathbf{1}b^\top$: $\bar W = G^\top X$, $\bar X = GW$, $\bar b = G^\top \mathbf{1}$ |
| Elementwise VJP | $Y = f(X)$: $\bar X = G \odot f'(X)$ |
| Broadcast VJP | forward broadcast ⇒ backward sum over the broadcast axis |
| Softmax-CE gradient | $\partial \mathcal{L} / \partial z = p - y$ |
| Local derivatives | $\exp' = e^a$ (output), $\ln' = 1/a$, $\tanh' = 1 - t^2$ (output), $\mathrm{relu}' = \mathbb{1}[a>0]$, $(a^k)' = k a^{k-1}$ |
| Gradient check | $r = \lVert g - \hat g \rVert / (\lVert g \rVert + \lVert \hat g \rVert)$; $< 10^{-7}$ good, $> 10^{-2}$ bug; float64, $h = 10^{-5}$ |
| Log-sum-exp | $\ln \sum e^{z_i} = m + \ln \sum e^{z_i - m}$, $m = \max_i z_i$ |
| Sigmoid chain bound | $\sigma' \le 1/4$, so $\lVert\delta_1\rVert \le 4^{-(L-1)}\prod_{l=2}^{L} \lVert W_l\rVert_2\,\lVert\delta_L\rVert$: guaranteed geometric shrinkage when every $\lVert W_l\rVert_2 < 4$; large weights can make gradients explode instead |
| Hessian-vector product | $Hv = \nabla_\theta (\nabla_\theta L \cdot v)$, about two backward passes |
| Activation memory (transformer layer, fp16) | $\approx s b h (34 + 5 a s/h)$ bytes; checkpointing keeps $\approx 2 s b h$ |

## Self-assessment

1. **Why does reverse mode cost about the same for a model with 10 parameters and one with 10 million, while finite differences do not?**
   <details><summary>Answer</summary>

   Reverse mode performs one backward sweep per *output*. A loss has one output, so one sweep — costing a constant multiple of the forward pass — yields all $n$ partial derivatives, whatever $n$ is. Finite differences need one or two extra forward passes per *input*, so their cost grows linearly in $n$ (Figure 2(b): the flat red line against the rising orange one).

   </details>
2. **A node is used by three later operations. What does its adjoint look like, and what bug does this expose in a hand-written engine?**
   <details><summary>Answer</summary>

   Its adjoint is the sum of three contributions, $\bar v = \sum_{i \in \mathrm{ch}(v)} \bar v_i \,\partial \phi_i / \partial v$. An engine that assigns with <code>=</code> instead of accumulating with <code>+=</code> keeps only the last child's contribution and produces a wrong gradient exactly when variables are reused — in Figure 3(c), $\bar x_1$ would be 16 or 48 instead of 64.

   </details>
3. **You want the full $3 \times 1000$ Jacobian of $f: \mathbb{R}^{1000} \to \mathbb{R}^3$. Which mode, and roughly how many sweeps?**
   <details><summary>Answer</summary>

   Reverse mode: three sweeps (one per output, each filling one row as in Figure 2(a)), each costing a small multiple of $f$. Forward mode would need 1,000 sweeps, one per column. For $f: \mathbb{R}^3 \to \mathbb{R}^{1000}$ the answer flips: three forward sweeps.

   </details>
4. **In the 2–2–1 worked example of Section 4, why are all gradients of the first hidden unit zero, and what would change with a leaky ReLU?**
   <details><summary>Answer</summary>

   Its pre-activation $z = -1$ is negative, so $\mathrm{relu}'(z) = 0$ and $\delta_1$ has a zero entry, which multiplies into $\partial \mathcal{L}/\partial W_1$ and $\partial \mathcal{L}/\partial b_1$ (the grey, masked unit of Figure 4(b)). With leaky ReLU, $f'(z) = \alpha$ (say $0.01$) instead of $0$, so the unit receives a small but non-zero gradient and can recover.

   </details>
5. **Why does training memory grow with batch size even though the number of parameters does not?**
   <details><summary>Answer</summary>

   The backward rules need the forward activations ($a_{l-1}$ for $\bar W_l = \delta_l a_{l-1}^\top$, $z_l$ or $f'(z_l)$ for the mask) of every example in the batch, so stored activations scale with batch × depth × width. Weights are shared across the batch. For GPT-2 small at batch 8 and context 1,024 that is $8.6$ GB of activations against $0.25$ GB of weights (Figure 8(a)). Checkpointing trades recomputation for storing fewer activations.

   </details>
6. **Derive the backward rule for $Y = AB$ with respect to $B$ from the scalar definition.**
   <details><summary>Answer</summary>

   $Y_{ij} = \sum_k A_{ik} B_{kj}$, so $\partial Y_{ij}/\partial B_{kj} = A_{ik}$ and $B_{kj}$ affects only column $j$ of $Y$. Hence $\bar B_{kj} = \sum_i G_{ij} A_{ik} = (A^\top G)_{kj}$, i.e. $\bar B = A^\top G$, of shape $k \times n$ as required. It is the mirror image of the derivation of $\bar A$ in Section 6, and passes the shape test of Figure 5(b): $(k \times m)(m \times n) = k \times n$.

   </details>
7. **Your gradient check reports relative error $0.5$ on one ReLU weight and $10^{-10}$ on all others. Bug or not?**
   <details><summary>Answer</summary>

   Almost certainly not a bug: that pattern is the signature of a pre-activation within $h$ of zero, where the central difference straddles the kink and returns about half the one-sided slope. In Figure 7(b), a pre-activation just above zero gives an estimate near $0.5$ against the analytic 1, a relative error near $0.5$. Re-check at a different input or parameter setting; a real bug in the ReLU rule would affect every ReLU weight.

   </details>
8. **Why can `log(softmax(z))` give NaN while `log_softmax(z)` does not, even in float64?**
   <details><summary>Answer</summary>

   If some $z_i > 709$, <code>exp</code> overflows to infinity and the softmax becomes inf/inf = NaN; if a class is confidently wrong, its probability underflows to 0 and <code>log(0)</code> is $-\infty$. <code>log_softmax</code> computes $z - m - \ln \sum e^{z_i - m}$ with $m = \max z$: every exponent is at most 0 and no logarithm of zero is taken.

   </details>
9. **What does the straight-through estimator change, and why is it not "cheating"?**
   <details><summary>Answer</summary>

   It keeps the hard, non-differentiable operation (rounding, argmax) in the forward pass but replaces its zero-almost-everywhere derivative with the identity (or a smooth surrogate) in the backward pass. The resulting update is a biased estimate of a gradient that does not exist, but it moves the upstream parameters in a direction that reduces the loss of the relaxed problem, and empirically it trains quantised networks and discrete latent models well.

   </details>
10. **How can you compute a Hessian-vector product for a model with $10^8$ parameters without $10^{16}$ memory?**
    <details><summary>Answer</summary>

    Form the scalar $s = \nabla_\theta L \cdot v$ with a backward pass that keeps its own graph (<code>create_graph=True</code>), then differentiate $s$ with respect to $\theta$: $\nabla_\theta s = H v$. Two backward passes, $O(n)$ memory, and no Hessian is ever materialised.

    </details>
11. **A bias $b \in \mathbb{R}^n$ is added to every row of a batch of pre-activations of shape $B \times n$. Your backward pass returns $\bar b$ with shape $B \times n$. What is missing, and what is the correct gradient?**
    <details><summary>Answer</summary>

    The forward pass broadcast $b$ to every row, so $b$ was used $B$ times and its gradient must collect all $B$ contributions: $\bar b = G^\top \mathbf{1}$, the upstream gradient summed over the batch axis (<code>G.sum(axis=0)</code>), with the shape $n$ of $b$ (Figure 5(c)). Returning $G$ itself skips that sum. The shape test catches the mistake, because a gradient always has the shape of the quantity it belongs to.

    </details>

## Further reading

- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed., SIAM, 2008 — the definitive reference, including the cheap gradient principle and checkpointing schedules.
- A. G. Baydin, B. A. Pearlmutter, A. A. Radul and J. M. Siskind, "Automatic Differentiation in Machine Learning: a Survey", *Journal of Machine Learning Research* 18, 2018 — free, readable, and the best bridge between the AD and ML communities.
- D. E. Rumelhart, G. E. Hinton and R. J. Williams, "Learning representations by back-propagating errors", *Nature* 323, 1986 — the paper that made backpropagation the standard training algorithm.
- A. Karpathy, *micrograd* (GitHub) and the accompanying lecture "The spelled-out intro to neural networks and backpropagation" (2022) — the design this module's `Value` engine follows.
- I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*, MIT Press, 2016, Section 6.5 "Back-Propagation and Other Differentiation Algorithms" — free online; covers symbol-to-number vs symbol-to-symbol differentiation.
- Stanford CS231n course notes, "Backpropagation, Intuitions" and "Neural Networks Part 3: Learning and Evaluation" (gradient checks) — the source of the relative-error thresholds used here.
- M. Giles and P. Glasserman, "Smoking adjoints: fast Monte Carlo Greeks", *Risk*, January 2006 — reverse mode in quantitative finance.
- T. Chen, B. Xu, C. Zhang and C. Guestrin, "Training Deep Nets with Sublinear Memory Cost", arXiv:1604.06174, 2016 — gradient checkpointing.
- B. A. Pearlmutter, "Fast Exact Multiplication by the Hessian", *Neural Computation* 6(1), 1994 — Hessian-vector products from a double backward pass.
- Y. Bengio, N. Léonard and A. Courville, "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation", arXiv:1308.3432, 2013 — the straight-through estimator.

## What's in the notebooks

**`lesson.ipynb`** walks through the module in code: the forward-vs-central finite-difference error curve as a function of $h$ (the V-shaped trade-off), with its round-off and truncation guide lines; a drawn computational graph with its forward pass; a `Dual` class that performs forward-mode AD and reproduces Section 3's numbers in two sweeps, drawn side by side with the single reverse sweep; the backprop recurrence applied to Section 4's 2–2–1 network, with slices of the loss showing that the dead ReLU unit's zero gradient is the true slope; the `Value` autograd engine built step by step, its topological order printed, the graph it actually builds drawn with every node's value and gradient, and its gradients verified against central differences; a timing of the full gradient showing forward mode's cost growing like $n^2$ and reverse mode's like $n$; a 2–16–16–1 network of `Value`s trained on a two-moons dataset with a loss curve and decision boundary; the same network re-implemented with NumPy arrays and the matrix backward rules, producing gradients identical to the scalar engine's to about $10^{-16}$; the fused softmax-cross-entropy gradient $p - y$ checked numerically, and the unfused route's division by $p$ followed until it overflows; gradient checks of the trained network, including a false alarm at a ReLU kink that vanishes when $h$ shrinks, and two planted bugs that each hide at one end of a sweep over input scales; and the log-sum-exp, ReLU-kink and vanishing- and exploding-gradient demonstrations from Section 8.

**`exercises.ipynb`** (with `solutions.ipynb`) tests the same ideas: forward and central differences; completing a `Dual` class; a reverse-mode sweep by hand with hidden answers; a topological sort; completing the `*`, `exp`, `log`, `tanh` and `relu` closures of a `Value` class and verifying them numerically; the two-layer backprop recurrence in NumPy; the matmul backward rule; a stable, fused softmax-cross-entropy loss-and-gradient; a reusable gradient checker; a conceptual question on choosing forward or reverse mode; a planted bug to diagnose with your checker; and a mini-project that trains a two-layer network on two moons with your own engine to above 95 percent training accuracy.
