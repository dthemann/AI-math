"""
build_19.py — generates lesson.ipynb, exercises.ipynb and solutions.ipynb for

    part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl

Run from the repository root:

    python3 _authoring/builders/build_19.py
    python3 tools/run_notebooks.py part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl

Everything is plain NumPy: the bandits, the gridworld environments, dynamic programming, TD
learning, linear function approximation, REINFORCE and the DPO check are all implemented here
(no gymnasium, no deep-learning framework). All hidden answers are computed by *executing* the
reference solutions and hashing the result with `sha_of`, so the exercises notebook never
contains the answers in clear text.
"""
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (tools/builders/ -> repo)
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import ExerciseSet, Notebook, sha_of  # noqa: E402

import numpy as np  # noqa: E402

MOD = ROOT / "part-6-deep-learning-and-modern-ai/19-markov-decision-processes-and-rl"

# =============================================================================
# LESSON NOTEBOOK
# =============================================================================
nb = Notebook("Module 19 — Markov Decision Processes and Reinforcement Learning")

nb.md(r"""
This notebook is the runnable companion to the concept guide: read [README.md](README.md) first, or
keep it open alongside. Every section mirrors a numbered section of the guide. We compute discounted
returns, race ε-greedy against UCB1 and Thompson sampling on a bandit, solve a three-state MDP exactly
and by iteration, watch the Bellman operator contract, plan in a gridworld with value and policy
iteration, learn without a model (Monte Carlo, TD(0), SARSA, Q-learning), add linear function
approximation, derive and test the REINFORCE estimator with and without a baseline, and finish with a
numerical check of the closed-form optimal policy behind DPO and a picture of PPO's clipped objective.

Everything is plain NumPy — the environments are implemented here in a few dozen lines each.
""")

nb.code(r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)
""")

# ---------------------------------------------------------------- 1. returns
nb.md(r"""
## 1. Returns, discounting and the effective horizon

The return from time $t$ is $G_t = \sum_{k \ge 0} \gamma^k r_{t+k+1}$, and it obeys the recursion
$G_t = r_{t+1} + \gamma G_{t+1}$. The weights $\gamma^k$ sum to $1/(1-\gamma)$, the **effective horizon**;
a reward $k$ steps away counts for half its face value after $k = \ln 2 / \ln(1/\gamma)$ steps.
We compute the return of the reward sequence from the guide for several $\gamma$ and draw the weights.
""")

nb.code(r"""
def discounted_return(rewards, gamma):
    # G_0 = sum_k gamma^k r_{k+1}
    return float(sum(gamma**k * r for k, r in enumerate(rewards)))

def returns_backward(rewards, gamma):
    # all G_t at once via the recursion G_t = r_{t+1} + gamma G_{t+1}, run from the end
    G = np.zeros(len(rewards)); running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G

rewards = [1, 0, 0, 4, 2]
for gamma in (0.5, 0.9, 0.99):
    G = returns_backward(rewards, gamma)
    print(f"gamma = {gamma:4.2f}: G_0 = {discounted_return(rewards, gamma):6.3f}   all G_t = {G}   "
          f"horizon 1/(1-gamma) = {1/(1-gamma):6.1f}   half-life = {np.log(2)/np.log(1/gamma):5.1f} steps")

print("\nweight gamma^k of a reward k steps ahead:")
print(f"{'k':>6s}" + "".join(f"{g:>10.2f}" for g in (0.5, 0.9, 0.99)))
for k in (1, 5, 10, 20, 50, 100, 200):
    print(f"{k:6d}" + "".join(f"{g**k:10.5f}" for g in (0.5, 0.9, 0.99)))
""")

nb.md(r"""
The reward of 4 at step four is worth $0.5$ at $\gamma = 0.5$ and $2.916$ at $\gamma = 0.9$: the two agents
are solving different problems. In the table, read down each column to the row $k = 1/(1-\gamma)$: the weight
has fallen to $\gamma^{1/(1-\gamma)} \approx 1/e \approx 0.37$ for every $\gamma$ close to 1, which is why
"$1/(1-\gamma)$ steps" is the right mental picture of how far an agent looks ahead.
""")

# ---------------------------------------------------------------- 2. bandits
nb.md(r"""
## 2. Multi-armed bandits: ε-greedy, UCB1 and Thompson sampling

A 10-armed Bernoulli bandit with success probabilities drawn from $\text{Uniform}(0.1, 0.9)$ in each run.
Each algorithm is a *selection rule* that sees the pull counts $N_a$, the reward sums $S_a$ and the round
$t$:

* **ε-greedy**: random arm with probability $\varepsilon$, otherwise the best empirical mean;
* **optimistic initialisation**: greedy, with unseen arms valued at 1;
* **UCB1**: $\arg\max_a\, S_a/N_a + c\sqrt{\ln t / N_a}$ with $c = \sqrt 2$ (every arm is pulled once first);
* **Thompson sampling**: draw $\tilde\mu_a \sim \text{Beta}(1 + S_a,\, 1 + N_a - S_a)$ and pull the largest draw.

We record the per-step regret $\mu^* - \mu_{a_t}$ and whether the optimal arm was pulled, averaged over 20 runs.
""")

nb.code(r"""
def argmax_random_tie(values, rng):
    return int(rng.choice(np.flatnonzero(values == values.max())))

def select_eps_greedy(N, S, t, rng, eps=0.1):
    if rng.random() < eps:
        return int(rng.integers(len(N)))
    means = np.where(N > 0, S / np.maximum(N, 1), 0.0)
    return argmax_random_tie(means, rng)

def select_optimistic(N, S, t, rng, init=1.0):
    means = np.where(N > 0, S / np.maximum(N, 1), init)      # unseen arms look as good as possible
    return argmax_random_tie(means, rng)

def select_ucb1(N, S, t, rng, c=np.sqrt(2)):
    if (N == 0).any():                                        # pull every arm once first
        return int(np.argmin(N))
    return int(np.argmax(S / N + c * np.sqrt(np.log(t) / N)))

def select_thompson(N, S, t, rng):
    return int(np.argmax(rng.beta(S + 1, N - S + 1)))         # Beta(1,1) prior, one posterior draw per arm

def run_bandit(select, probs, T, rng, **kw):
    n_arms = len(probs); N = np.zeros(n_arms); S = np.zeros(n_arms)
    regret = np.zeros(T); optimal = np.zeros(T); best = int(np.argmax(probs))
    for t in range(1, T + 1):
        a = select(N, S, t, rng, **kw)
        r = float(rng.random() < probs[a])
        N[a] += 1; S[a] += r
        regret[t - 1] = probs[best] - probs[a]; optimal[t - 1] = (a == best)
    return regret, optimal

def average_runs(select, n_runs=20, T=2000, n_arms=10, seed=0, **kw):
    rng_b = np.random.default_rng(seed)
    cum_regret = np.zeros((n_runs, T)); opt = np.zeros((n_runs, T))
    for run in range(n_runs):
        probs = rng_b.uniform(0.1, 0.9, size=n_arms)
        regret, optimal = run_bandit(select, probs, T, rng_b, **kw)
        cum_regret[run] = np.cumsum(regret); opt[run] = optimal
    return cum_regret.mean(axis=0), opt.mean(axis=0)

algorithms = {
    "eps-greedy (eps = 0.1)":  (select_eps_greedy, {}),
    "optimistic init":         (select_optimistic, {}),
    "UCB1 (c = sqrt 2)":       (select_ucb1, {}),
    "UCB1 (c = 0.5)":          (select_ucb1, {"c": 0.5}),
    "Thompson sampling":       (select_thompson, {}),
}
results = {name: average_runs(sel, **kw) for name, (sel, kw) in algorithms.items()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for name, (cum_regret, frac_opt) in results.items():
    ax1.plot(cum_regret, label=name)
    ax2.plot(np.convolve(frac_opt, np.ones(50) / 50, mode="valid"), label=name)
ax1.set_xlabel("round t"); ax1.set_ylabel("cumulative regret (mean of 20 runs)"); ax1.set_title("Cumulative regret, 10 arms")
ax2.set_xlabel("round t"); ax2.set_ylabel("fraction of optimal pulls (50-round average)"); ax2.set_title("How often the best arm is pulled")
ax1.legend(fontsize=8); plt.tight_layout(); plt.show()

print(f"{'algorithm':26s} {'regret @ 400':>13s} {'regret @ 2000':>14s} {'optimal (last 200)':>19s}")
for name, (cum_regret, frac_opt) in results.items():
    print(f"{name:26s} {cum_regret[399]:13.1f} {cum_regret[-1]:14.1f} {frac_opt[-200:].mean():19.3f}")
""")

nb.md(r"""
ε-greedy's regret grows along a straight line: it keeps spending 10% of its pulls on random arms forever.
UCB1 with the textbook constant $\sqrt 2$ explores *too much* on this easy problem (its bonuses stay large
for a long time), while $c = 0.5$ and Thompson sampling bend over towards the logarithmic shape the theory
promises. Optimistic initialisation is excellent here but has no mechanism for revisiting an arm whose
early draws were unlucky — on a harder problem it can lock onto the wrong arm.

The guide's worked example: three arms, $t = 10$, $N = (5, 3, 2)$, successes $(3, 2, 1)$.
""")

nb.code(r"""
N = np.array([5., 3., 2.]); S = np.array([3., 2., 1.]); t = 10
means = S / N; bonus = np.sqrt(2 * np.log(t) / N)
print("empirical means :", means)
print("UCB1 bonuses    :", bonus)
print("UCB1 indices    :", means + bonus, " -> pull arm", int(np.argmax(means + bonus)) + 1, "(the least-tried one)")

# Thompson sampling pulls each arm with the posterior probability that it is the best
draws = rng.beta(S + 1, N - S + 1, size=(200_000, 3))
print("Thompson pull probabilities:", np.bincount(draws.argmax(axis=1), minlength=3) / len(draws))
print("eps-greedy (eps = 0.1) pull probabilities:", np.array([0.1/3, 0.9 + 0.1/3, 0.1/3]))

# the Beta posteriors Thompson sampling maintains, early and late in a 3-arm run
probs3 = np.array([0.3, 0.5, 0.65])
N3 = np.zeros(3); S3 = np.zeros(3); snapshots = {}
for step in range(1, 301):
    a = select_thompson(N3, S3, step, rng); r = float(rng.random() < probs3[a]); N3[a] += 1; S3[a] += r
    if step in (20, 300): snapshots[step] = (N3.copy(), S3.copy())
x = np.linspace(0, 1, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for ax, (step, (N3s, S3s)) in zip(axes, snapshots.items()):
    for a in range(3):
        ax.plot(x, stats.beta.pdf(x, S3s[a] + 1, N3s[a] - S3s[a] + 1), label=f"arm {a+1}: p = {probs3[a]}, pulls = {int(N3s[a])}")
        ax.axvline(probs3[a], color=f"C{a}", ls=":", lw=1)
    ax.set_title(f"Beta posteriors after {step} rounds"); ax.set_xlabel("success probability"); ax.legend(fontsize=8)
axes[0].set_ylabel("posterior density"); plt.tight_layout(); plt.show()
""")

nb.md(r"""
At $t = 10$ UCB1 pulls the arm with the *lowest* empirical mean, because it has been tried least and its
upper confidence bound is the highest; Thompson sampling still gives that arm about a quarter of the
pulls. After 300 rounds the posterior of the best arm is sharp and centred on $0.65$, while the posteriors
of the arms it stopped pulling stay wide — Thompson sampling only sharpens what it needs to know.

**Try it yourself:** set `eps=0.01` and `eps=0.3` in `average_runs(select_eps_greedy, eps=...)` and compare the
regret at 400 and at 2000 rounds — a small $\varepsilon$ is slow at first and good later, a large one the reverse.

### An adaptive clinical trial (Case study 1 of the guide)

Two treatments with true response rates $0.4$ and $0.6$, 200 patients. A fixed 1:1 design sends 100 patients to
each arm; Thompson sampling assigns each patient by posterior draws. We simulate 200 trials of each design and
measure two things: how many patients received the inferior treatment, and how often the trial ends with
$P(p_B > p_A \mid \text{data}) > 0.95$ — a proxy for its power to declare a winner.
""")

nb.code(r"""
def run_trial(p_true, n_patients, method, rng):
    succ = np.zeros(2); fail = np.zeros(2); n_on_A = 0
    for i in range(n_patients):
        arm = int(np.argmax(rng.beta(succ + 1, fail + 1))) if method == "thompson" else i % 2
        response = rng.random() < p_true[arm]
        succ[arm] += response; fail[arm] += 1 - response; n_on_A += (arm == 0)
    return n_on_A, succ, fail

def posterior_prob_B_better(succ, fail, rng, n=20_000):
    return float(np.mean(rng.beta(succ[1] + 1, fail[1] + 1, n) > rng.beta(succ[0] + 1, fail[0] + 1, n)))

rng_t = np.random.default_rng(7)
summary = {}
for method in ("fixed 1:1", "thompson"):
    on_A, responders, declared = [], [], []
    for trial in range(200):
        n_on_A, succ, fail = run_trial((0.4, 0.6), 200, method, rng_t)
        on_A.append(n_on_A); responders.append(succ.sum()); declared.append(posterior_prob_B_better(succ, fail, rng_t) > 0.95)
    summary[method] = (np.array(on_A), np.array(responders), np.array(declared))
    print(f"{method:10s}: patients on the inferior arm A: mean {np.mean(on_A):6.1f} (sd {np.std(on_A):5.1f}, > 100 in {np.mean(np.array(on_A) > 100):.0%} of trials)   "
          f"responders: mean {np.mean(responders):6.1f}   P(B declared better at 95%): {np.mean(declared):.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
ax1.hist(summary["thompson"][0], bins=30, alpha=0.7, label="Thompson sampling")
ax1.axvline(100, color="k", ls="--", label="fixed 1:1 design (always 100)")
ax1.set_xlabel("patients (of 200) assigned to the inferior treatment"); ax1.set_ylabel("number of simulated trials")
ax1.set_title("Adaptive allocation spares patients ..."); ax1.legend(fontsize=8)
ax2.bar(["fixed 1:1", "thompson"], [summary[m][2].mean() for m in ("fixed 1:1", "thompson")], color=["gray", "C0"])
ax2.set_ylim(0, 1); ax2.set_ylabel("fraction of trials declaring B better (95%)"); ax2.set_title("... but loses power to declare a winner")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Thompson sampling sends on average only about 28 of the 200 patients (instead of 100) to the inferior arm and
produces about fourteen more responders per trial — but the probability of *declaring* B better at the 95% level
drops from about 0.89 to about 0.74, because the inferior arm's response rate is estimated from a few dozen
patients and its posterior stays wide. (The fixed design's 0.89 is what the usual power calculation predicts: a
difference of 0.2 with standard error $\sqrt{2 \cdot 0.24/100} \approx 0.069$ is 2.9 standard errors, and
$\Phi(2.9 - 1.645) \approx 0.89$.) Regret minimisation and hypothesis testing are different objectives; real adaptive designs cap the imbalance
or use a burn-in to buy back power.
""")

# ---------------------------------------------------------------- 3. MDP
nb.md(r"""
## 3. A three-state MDP and exact policy evaluation

The guide's machine: states **G**ood, **W**orn, **B**roken; actions **run** (0) and **service** (1); $\gamma = 0.8$.
We store the model as arrays `P[s, a, s']` and `R[s, a]`. For a deterministic policy $\pi$ the Bellman expectation
equation is linear,

$$V^\pi = R^\pi + \gamma P^\pi V^\pi \quad\Longrightarrow\quad V^\pi = (I - \gamma P^\pi)^{-1} R^\pi,$$

and $Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')$ tells us where a different action would do better.
""")

nb.code(r"""
STATES = ["Good", "Worn", "Broken"]; ACTS = ["run", "service"]
P_M = np.zeros((3, 2, 3))                       # P_M[s, a, s']
P_M[0, 0] = [0.75, 0.25, 0.0]                   # run in Good:   stays good or wears
P_M[1, 0] = [0.0, 0.5, 0.5]                     # run in Worn:   stays worn or breaks
P_M[2, 0] = [0.0, 0.0, 1.0]                     # run in Broken: stays broken
P_M[:, 1] = [1.0, 0.0, 0.0]                     # service from anywhere -> Good
R_M = np.array([[10., -2.], [6., -2.], [0., -12.]])   # R_M[s, a]
GAMMA_M = 0.8
assert np.allclose(P_M.sum(axis=2), 1.0)

def policy_evaluation_exact(P, R, policy, gamma):
    # policy: array of action indices, one per state
    S = P.shape[0]; idx = np.arange(S)
    P_pi = P[idx, policy]                       # rows P(. | s, pi(s))
    R_pi = R[idx, policy]
    return np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)

def q_from_v(P, R, V, gamma):
    return R + gamma * np.einsum("sat,t->sa", P, V)

pi_run = np.array([0, 0, 0])
V_run = policy_evaluation_exact(P_M, R_M, pi_run, GAMMA_M)
print("I - gamma P^pi for 'always run':\n", np.eye(3) - GAMMA_M * P_M[np.arange(3), pi_run])
print("V^pi (always run) =", V_run)
Q_run = q_from_v(P_M, R_M, V_run, GAMMA_M)
print("\nQ^pi(s, a) under 'always run':")
for s in range(3):
    print(f"  {STATES[s]:7s} run {Q_run[s, 0]:6.2f}   service {Q_run[s, 1]:6.2f}   -> greedy: {ACTS[int(Q_run[s].argmax())]}")
""")

nb.md(r"""
Exactly the hand computation of the guide: $V_B = 0$, $V_W = 10$, $V_G = 30$. Servicing from any state leads
to Good (worth 30), so it beats running in **Worn** ($22 > 10$) and in **Broken** ($12 > 0$) but not in **Good**
($22 < 30$). The greedy policy with respect to $Q^{\pi}$ is therefore (run, service, service) — Section 5 shows
that this one improvement step already reaches the optimum.
""")

# ---------------------------------------------------------------- 4. Bellman / contraction
nb.md(r"""
## 4. Bellman operators are $\gamma$-contractions

Iterating $V \leftarrow R^\pi + \gamma P^\pi V$ converges to $V^\pi$, and iterating the optimality operator
$(TV)(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V(s')]$ converges to $V^*$, in both cases because
$\lVert TU - TV\rVert_\infty \le \gamma\, \lVert U - V\rVert_\infty$. We check the contraction on random pairs of value
vectors and watch the error shrink by a factor $\gamma$ per sweep.
""")

nb.code(r"""
def bellman_optimality_operator(V, P, R, gamma):
    return q_from_v(P, R, V, gamma).max(axis=1)

def value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000):
    V = np.zeros(P.shape[0]); history = [V.copy()]
    for k in range(max_iter):
        V_new = bellman_optimality_operator(V, P, R, gamma)
        history.append(V_new.copy())
        if np.abs(V_new - V).max() < tol:
            return V_new, q_from_v(P, R, V_new, gamma).argmax(axis=1), history
        V = V_new
    return V, q_from_v(P, R, V, gamma).argmax(axis=1), history

# (a) contraction check on random value vectors
ratios = []
for _ in range(1000):
    U, V = rng.normal(size=3) * 50, rng.normal(size=3) * 50
    TU, TV = bellman_optimality_operator(U, P_M, R_M, GAMMA_M), bellman_optimality_operator(V, P_M, R_M, GAMMA_M)
    ratios.append(np.abs(TU - TV).max() / np.abs(U - V).max())
print(f"max over 1000 random pairs of ||TU - TV|| / ||U - V|| = {max(ratios):.4f}  (gamma = {GAMMA_M})")

# (b) value iteration and iterative policy evaluation, error per sweep
V_star, pi_star, hist = value_iteration(P_M, R_M, GAMMA_M)
print(f"V* = {V_star}, optimal policy = {[ACTS[a] for a in pi_star]}, sweeps to 1e-10: {len(hist) - 1}")
err_vi = np.array([np.abs(h - V_star).max() for h in hist])
for k in range(4):
    print(f"  sweep {k}: V_k = {hist[k]}, error {err_vi[k]:.4f}")

V_pi1 = policy_evaluation_exact(P_M, R_M, pi_star, GAMMA_M)
V = np.zeros(3); err_pe = [np.abs(V - V_pi1).max()]
for k in range(60):
    V = R_M[np.arange(3), pi_star] + GAMMA_M * P_M[np.arange(3), pi_star] @ V
    err_pe.append(np.abs(V - V_pi1).max())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.semilogy(err_pe, "o-", ms=3, label="iterative policy evaluation of pi*")
ax1.semilogy(err_pe[0] * GAMMA_M ** np.arange(len(err_pe)), "k--", label="gamma^k bound")
ax1.set_xlabel("sweep k"); ax1.set_ylabel("||V_k - V^pi||_inf"); ax1.set_title("Policy evaluation converges geometrically"); ax1.legend()
ax2.semilogy(err_vi, "o-", ms=3, label="value iteration")
ax2.semilogy(err_vi[0] * GAMMA_M ** np.arange(len(err_vi)), "k--", label="gamma^k bound")
ax2.set_xlabel("sweep k"); ax2.set_ylabel("||V_k - V*||_inf"); ax2.set_title("Value iteration: error shrinks by gamma per sweep"); ax2.legend()
plt.tight_layout(); plt.show()

eps_stop = 1e-3; k_stop = next(k for k in range(1, len(hist)) if np.abs(hist[k] - hist[k-1]).max() < eps_stop)
print(f"stopping when successive sweeps differ by < {eps_stop}: sweep {k_stop}, true error {err_vi[k_stop]:.2e}, "
      f"guaranteed bound eps*gamma/(1-gamma) = {eps_stop*GAMMA_M/(1-GAMMA_M):.2e}")
""")

nb.md(r"""
No pair of value vectors is ever pushed apart by more than the factor $\gamma = 0.8$, and both error curves
run parallel to the dashed $\gamma^k$ line on the log scale — a straight line of slope $\log\gamma$. The
stopping rule "successive sweeps differ by less than $\epsilon$" certifies an error of at most
$\epsilon\gamma/(1-\gamma) = 4\epsilon$ here; with $\gamma = 0.99$ the same rule would only certify $99\epsilon$.

**Try it yourself:** set `GAMMA_M = 0.95` in Section 3 and re-run — how many sweeps does value iteration need,
and does the optimal policy change? (At $\gamma = 0.3$ it does: the myopic operator never services at all.)
""")

# ---------------------------------------------------------------- 5. gridworld DP
nb.md(r"""
## 5. Planning in a gridworld: value iteration and policy iteration

A $5 \times 5$ gridworld with walls, one pit and *slippery* moves: with probability 0.2 the agent moves
perpendicular to the intended direction (0.1 each way). Entering the goal gives $+10$, the pit $-10$, every
other move $-1$; goal and pit are terminal. The class below builds the full model `P[s, a, s']`, `R[s, a]`
so that we can plan exactly, and also offers a sampling `step` for the model-free methods later.
""")

nb.code(r"""
class GridWorld:
    ACTIONS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])      # 0 up, 1 right, 2 down, 3 left
    ARROWS = ["^", ">", "v", "<"]

    def __init__(self, n_rows, n_cols, start, goal, walls=(), pits=(), slip=0.0,
                 step_reward=-1.0, goal_reward=10.0, pit_reward=-10.0, gamma=0.9):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.n_states, self.n_actions = n_rows * n_cols, 4
        self.start, self.goal = self.idx(*start), self.idx(*goal)
        self.walls = {self.idx(*w) for w in walls}
        self.pits = {self.idx(*p) for p in pits}
        self.terminal = {self.goal} | self.pits
        self.slip, self.step_reward, self.goal_reward, self.pit_reward, self.gamma = slip, step_reward, goal_reward, pit_reward, gamma
        self.P, self.R = self._build_model()
        self.cumP = self.P.cumsum(axis=2)                       # for fast sampling

    def idx(self, r, c): return r * self.n_cols + c
    def rc(self, s): return divmod(s, self.n_cols)

    def _move(self, s, a):
        r, c = self.rc(s); dr, dc = self.ACTIONS[a]; r2, c2 = r + dr, c + dc
        if not (0 <= r2 < self.n_rows and 0 <= c2 < self.n_cols) or self.idx(r2, c2) in self.walls:
            return s                                             # bump: stay in place
        return self.idx(r2, c2)

    def _reward(self, s2):
        if s2 == self.goal: return self.goal_reward
        if s2 in self.pits: return self.pit_reward
        return self.step_reward

    def _build_model(self):
        S, A = self.n_states, self.n_actions
        P = np.zeros((S, A, S)); R = np.zeros((S, A))
        for s in range(S):
            if s in self.terminal or s in self.walls:            # absorbing, zero reward
                P[s, :, s] = 1.0; continue
            for a in range(A):
                outcomes = {a: 1.0 - self.slip, (a - 1) % 4: self.slip / 2, (a + 1) % 4: self.slip / 2}
                for a2, prob in outcomes.items():
                    if prob == 0: continue
                    s2 = self._move(s, a2)
                    P[s, a, s2] += prob; R[s, a] += prob * self._reward(s2)
        return P, R

    def step(self, s, a, rng):
        s2 = int(np.searchsorted(self.cumP[s, a], rng.random()))
        return s2, self._reward(s2), s2 in self.terminal

    def show_values(self, V, policy=None, ax=None, title="V*"):
        ax = ax or plt.gca()
        grid = V.reshape(self.n_rows, self.n_cols).copy()
        for w in self.walls: grid[self.rc(w)] = np.nan
        im = ax.imshow(grid, cmap="viridis"); plt.colorbar(im, ax=ax, fraction=0.046)
        for s in range(self.n_states):
            r, c = self.rc(s)
            if s in self.walls: ax.text(c, r, "wall", ha="center", va="center", color="w", fontsize=7); continue
            label = "GOAL" if s == self.goal else ("PIT" if s in self.pits else f"{V[s]:.1f}")
            ax.text(c, r - 0.28, label, ha="center", va="center", color="w", fontsize=7)
            if policy is not None and s not in self.terminal:
                dr, dc = self.ACTIONS[policy[s]]
                ax.arrow(c, r + 0.1, 0.3 * dc, 0.3 * dr, head_width=0.12, color="w", length_includes_head=True)
        sr, sc = self.rc(self.start); ax.text(sc, sr + 0.38, "start", ha="center", color="orange", fontsize=7)
        ax.set_title(title); ax.set_xlabel("column"); ax.set_ylabel("row")
        ax.set_xticks(range(self.n_cols)); ax.set_yticks(range(self.n_rows))

gw = GridWorld(5, 5, start=(4, 0), goal=(0, 4), walls=[(1, 1), (2, 1), (1, 3), (3, 3)], pits=[(2, 3)], slip=0.2, gamma=0.9)
assert np.allclose(gw.P.sum(axis=2), 1.0)
V_star_gw, pi_star_gw, hist_gw = value_iteration(gw.P, gw.R, gw.gamma)
print(f"value iteration: {len(hist_gw) - 1} sweeps to tolerance 1e-10")
print("V* on the grid:\n", V_star_gw.reshape(5, 5))
print("greedy policy:\n", np.array([gw.ARROWS[a] for a in pi_star_gw]).reshape(5, 5))
plt.figure(figsize=(5.5, 5)); gw.show_values(V_star_gw, pi_star_gw, title="Gridworld: V* and the greedy policy (slip = 0.2, gamma = 0.9)"); plt.show()

# how risky is the greedy policy? Probability of ending in the pit, by an absorption-probability solve (Module 18):
# h = P_TT h + P_T,pit over the non-terminal states T, using the transition matrix of pi*
T_states = [s for s in range(gw.n_states) if s not in gw.terminal | gw.walls]
P_pi = gw.P[np.arange(gw.n_states), pi_star_gw]
pit = next(iter(gw.pits))
h_pit = np.linalg.solve(np.eye(len(T_states)) - P_pi[np.ix_(T_states, T_states)], P_pi[T_states, pit])
p_pit = dict(zip(T_states, h_pit))
for cell in [(4, 0), (3, 2), (4, 3), (2, 4)]:
    print(f"P(end in the pit | start at {cell}, follow pi*) = {p_pit[gw.idx(*cell)]:.3f}")
""")

nb.md(r"""
The values fall off with distance from the goal; the start is worth about $-2.3$ because the $+10$ is at least
eight slippery steps away. Three corridors lead to the goal: up the left column and along the top row, which never
passes the pit, and the middle and right-hand columns, each of which passes one cell beside it. The greedy policy
splits the grid between them — from the start it takes the safe left/top route, while cells nearer the middle or
the right accept one risky step because the detour would cost more. The absorption probabilities price that risk:
about $0.11$ of agents starting at $(3, 2)$, $(4, 3)$ or $(2, 4)$ end in the pit (each attempt to pass it slips
sideways into it with probability $0.1$, and a slip into a wall just means trying again: $0.1/0.9$), against about
$0.02$ from the start. The policy is read off as $\arg\max_a Q^*(s, a)$: no search, no simulation, just the model.

Now **policy iteration**: evaluate exactly (a linear solve), improve greedily, repeat until the policy is stable.
We compare the number of iterations of the two methods as $\gamma$ varies.
""")

nb.code(r"""
def policy_iteration(P, R, gamma, max_iter=100):
    S = P.shape[0]; policy = np.zeros(S, dtype=int)
    for n in range(1, max_iter + 1):
        V = policy_evaluation_exact(P, R, policy, gamma)
        Q = q_from_v(P, R, V, gamma)
        greedy = Q.argmax(axis=1)
        # keep the current action where it is already (within rounding) a maximiser -> guaranteed termination
        stable = np.isclose(Q[np.arange(S), policy], Q.max(axis=1))
        new_policy = np.where(stable, policy, greedy)
        if np.array_equal(new_policy, policy):
            return V, policy, n
        policy = new_policy
    return V, policy, max_iter

V_pi_gw, pi_pi_gw, n_pi = policy_iteration(gw.P, gw.R, gw.gamma)
print(f"policy iteration: {n_pi} iterations; agrees with value iteration: "
      f"values {np.allclose(V_pi_gw, V_star_gw)}, policy on non-terminal states "
      f"{np.array_equal(pi_pi_gw[[s for s in range(25) if s not in gw.terminal | gw.walls]], pi_star_gw[[s for s in range(25) if s not in gw.terminal | gw.walls]])}")

def policy_string(env, policy):
    return "".join("." if s in env.terminal | env.walls else env.ARROWS[int(policy[s])] for s in range(env.n_states))

print(f"\n{'gamma':>6s} {'VI sweeps':>10s} {'PI iters':>9s} {'worst-case VI bound':>20s} {'rho':>6s} {'gamma*rho':>10s}   policy (row by row)")
for g in [0.5, 0.8, 0.9, 0.95, 0.99]:
    V_g, pi_g, h = value_iteration(gw.P, gw.R, g, tol=1e-8)
    _, _, n = policy_iteration(gw.P, gw.R, g)
    worst = np.log(np.abs(V_g).max() / 1e-8) / np.log(1 / g)               # sweeps a factor-gamma contraction would need
    P_TT = gw.P[np.arange(gw.n_states), pi_g][np.ix_(T_states, T_states)]  # pi_g's transitions among non-terminal states
    rho = np.abs(np.linalg.eigvals(P_TT)).max()                            # spectral radius < 1: the agent terminates
    print(f"{g:6.2f} {len(h) - 1:10d} {n:9d} {worst:20.0f} {rho:6.3f} {g * rho:10.3f}   {policy_string(gw, pi_g)}")
""")

nb.md(r"""
Policy iteration finishes in a handful of iterations whatever $\gamma$ is, each costing a $25 \times 25$ linear
solve, and agrees with value iteration to machine precision. Value iteration's sweep count rises only gently (from
22 to 36) and stays far below the worst-case bound $\ln(\lVert V^*\rVert_\infty/\epsilon)/\ln(1/\gamma)$ — about 2,000 at
$\gamma = 0.99$. The reason is that this task is *episodic*: the goal and the pit are absorbing with value 0, so an
error is carried forward only along paths that have not yet terminated, and the asymptotic factor per sweep is
$\gamma\rho$, where $\rho$ is the spectral radius of the optimal policy's transition matrix among the non-terminal
states — roughly, the per-step probability of not having finished yet. Here $\rho \approx 0.44$, so even at
$\gamma = 0.99$ errors shrink by a factor of about $0.43$ per sweep. (At $\gamma = 0.5$ the policy changes: the
myopic agent at $(2, 2)$ prefers to bump into the wall rather than walk past the pit, a near-self-loop that pushes
$\rho$ up to $0.91$.) In a *continuing* task such as the machine of Section 4, $\rho = 1$ and the factor really is
$\gamma$. For a few thousand states policy iteration's $O(|\mathcal S|^3)$ solve is still fine; beyond that, value
iteration (or a few evaluation sweeps per improvement — *modified* policy iteration) takes over.

**Try it yourself:** set `pit_reward=-50` (keeping `slip=0.2`) and re-run: $(3, 1)$, $(3, 2)$ and $(4, 1)$ switch to the
safe left-column route at a cost of two to four extra steps, while $(4, 2)$ and $(4, 3)$ keep the shorter route past
the pit — for them the risky step lies several moves ahead and is discounted (at $\gamma = 0.99$, $(4, 2)$ detours
too). Then set `slip=0.0`: the start's value rises from $-2.3$ to about $-0.4$, the price of slippery floors.
""")

# ---------------------------------------------------------------- 6. model-free prediction and control
nb.md(r"""
## 6. Learning without a model: Monte Carlo, TD(0), SARSA and Q-learning

First **prediction**. We simulate 5,000 episodes of the optimal machine policy (vectorised) and estimate
$V^{\pi^*}(\text{Good})$ by averaging returns — the Monte Carlo estimate — against the exact $40$. Then the
classic five-state random walk: an agent starts in the middle, moves left or right at random, and receives $+1$
on exiting to the right. The true values are $1/6, \dots, 5/6$. We compare first-visit Monte Carlo
($V(s) \leftarrow V(s) + \alpha(G - V(s))$) with TD(0) ($V(s) \leftarrow V(s) + \alpha(r + V(s') - V(s))$).
""")

nb.code(r"""
# Monte Carlo estimate of V^pi*(Good) in the machine MDP, 5000 episodes truncated at 100 steps
n_ep, T_max = 5000, 100
cumP_M = P_M.cumsum(axis=2)
s = np.zeros(n_ep, dtype=int); G = np.zeros(n_ep); discount = 1.0
for t in range(T_max):
    a = pi_star[s]
    G += discount * R_M[s, a]                                   # expected reward of (s, a); fine for estimating V
    u = rng.random(n_ep)
    s = (u[:, None] > cumP_M[s, a]).sum(axis=1)                 # sample s' from P(. | s, a)
    discount *= GAMMA_M
print(f"Monte Carlo: V(Good) ~ {G.mean():.3f} +/- {G.std(ddof=1)/np.sqrt(n_ep):.3f}   exact 40.000   "
      f"(sd of a single return {G.std(ddof=1):.2f}; truncation bias < {GAMMA_M**T_max * 10/(1-GAMMA_M):.1e})")

# random walk: states 0..4, exits at -1 (reward 0) and 5 (reward 1); start at 2
def random_walk_episode(rng):
    s = 2; traj = []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        r = 1.0 if s2 == 5 else 0.0
        done = s2 < 0 or s2 == 5
        traj.append((s, r, s2, done)); s = s2
        if done: return traj

true_V = np.arange(1, 6) / 6

def mc_vs_td(n_runs=100, n_episodes=200, alpha_mc=0.02, alpha_td=0.1, seed=1):
    rng_w = np.random.default_rng(seed)
    err_mc = np.zeros((n_runs, n_episodes)); err_td = np.zeros((n_runs, n_episodes))
    for run in range(n_runs):
        V_mc = np.full(5, 0.5); V_td = np.full(5, 0.5)
        for ep in range(n_episodes):
            traj = random_walk_episode(rng_w)
            for (s, r, s2, done) in traj:                       # TD(0): one update per step
                target = r + (0.0 if done else V_td[s2])
                V_td[s] += alpha_td * (target - V_td[s])
            G = 0.0; first_visit_return = {}
            for (s, r, s2, done) in reversed(traj):             # MC: returns from the end; earliest visit wins
                G = r + G; first_visit_return[s] = G
            for s, G in first_visit_return.items():
                V_mc[s] += alpha_mc * (G - V_mc[s])
            err_mc[run, ep] = np.sqrt(np.mean((V_mc - true_V) ** 2))
            err_td[run, ep] = np.sqrt(np.mean((V_td - true_V) ** 2))
    return err_mc.mean(axis=0), err_td.mean(axis=0)

for a_mc, a_td, ls in ((0.02, 0.05, "-"), (0.04, 0.15, "--")):
    e_mc, e_td = mc_vs_td(alpha_mc=a_mc, alpha_td=a_td)
    plt.plot(e_mc, "C0", ls=ls, label=f"Monte Carlo, alpha = {a_mc}")
    plt.plot(e_td, "C3", ls=ls, label=f"TD(0), alpha = {a_td}")
    print(f"RMS error after 200 episodes: MC(alpha={a_mc}) {e_mc[-1]:.3f}   TD(alpha={a_td}) {e_td[-1]:.3f}")
plt.xlabel("episodes"); plt.ylabel("RMS error of V (mean over 100 runs)")
plt.title("Random walk: TD(0) learns faster than Monte Carlo"); plt.legend(); plt.show()
""")

nb.md(r"""
In both pairs TD(0) drops faster: its target $r + V(s')$ has the variance of one transition, while a Monte Carlo
return carries the noise of the whole walk. The price is bias — early on, $V(s')$ is wrong — and with a constant
$\alpha$ both methods plateau at a noise floor set by $\alpha$: the dashed, larger-step curves fall fastest and settle
highest, and TD's floor at $\alpha = 0.15$ even ends above Monte Carlo's at $\alpha = 0.04$. Comparing methods fairly
means comparing each at its best step size. Robbins–Monro step sizes ($\sum\alpha_t = \infty$, $\sum\alpha_t^2 < \infty$)
would send both errors to zero.

Now **control** on the cliff-walking task of Sutton and Barto: a $4 \times 12$ grid, start bottom-left, goal
bottom-right, the bottom row in between is a cliff ($-100$ and back to the start); every other move costs $-1$,
$\gamma = 1$. Both SARSA and Q-learning act ε-greedily with $\varepsilon = 0.1$ and differ only in the target:
$Q(s', a')$ for the action actually taken next (SARSA) versus $\max_{a'} Q(s', a')$ (Q-learning).
""")

nb.code(r"""
class CliffWalk:
    ACTIONS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
    def __init__(self, n_rows=4, n_cols=12):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.n_states, self.n_actions = n_rows * n_cols, 4
        self.start, self.goal = self.idx(n_rows - 1, 0), self.idx(n_rows - 1, n_cols - 1)
        self.cliff = {self.idx(n_rows - 1, c) for c in range(1, n_cols - 1)}
    def idx(self, r, c): return r * self.n_cols + c
    def rc(self, s): return divmod(s, self.n_cols)
    def step(self, s, a):
        r, c = self.rc(s); dr, dc = self.ACTIONS[a]
        s2 = self.idx(min(max(r + dr, 0), self.n_rows - 1), min(max(c + dc, 0), self.n_cols - 1))
        if s2 in self.cliff: return self.start, -100.0, False
        return s2, -1.0, s2 == self.goal

def td_control(env, method, n_episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0, max_steps=1000):
    rng_c = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions)); episode_return = np.zeros(n_episodes)
    def eps_greedy(s):
        if rng_c.random() < eps: return int(rng_c.integers(env.n_actions))
        return argmax_random_tie(Q[s], rng_c)
    for ep in range(n_episodes):
        s = env.start; a = eps_greedy(s); total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(s, a); total += r
            a2 = eps_greedy(s2)
            if method == "q-learning":
                target = r + (0.0 if done else gamma * Q[s2].max())        # off-policy: the greedy continuation
            else:
                target = r + (0.0 if done else gamma * Q[s2, a2])          # on-policy: the continuation actually taken
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            if done: break
        episode_return[ep] = total
    return Q, episode_return

def greedy_path(env, Q, max_steps=60):
    s = env.start; path = [s]
    for _ in range(max_steps):
        s, _, done = env.step(s, int(Q[s].argmax())); path.append(s)
        if done: break
    return path

cliff = CliffWalk(); curves = {}; Qs = {}
for method in ("sarsa", "q-learning"):
    runs = np.array([td_control(cliff, method, seed=seed)[1] for seed in range(5)])
    curves[method] = runs.mean(axis=0); Qs[method] = td_control(cliff, method, seed=0)[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
for method, c in curves.items():
    ax1.plot(np.convolve(c, np.ones(10) / 10, mode="valid"), label=method)
    print(f"{method:11s}: mean return per episode over the last 100 episodes = {c[-100:].mean():7.2f}   "
          f"greedy path length = {len(greedy_path(cliff, Qs[method])) - 1}")
ax1.set_ylim(-150, 0); ax1.set_xlabel("episode"); ax1.set_ylabel("return per episode (10-episode average, 5 seeds)")
ax1.set_title("Cliff walking: SARSA earns more while learning"); ax1.legend()
grid = np.zeros((4, 12)); grid[3, 1:11] = -1
ax2.imshow(grid, cmap="Greys", vmin=-1.5, vmax=0.5)
for method, c, m in (("sarsa", "C0", "o"), ("q-learning", "C1", "s")):
    rc = np.array([cliff.rc(s) for s in greedy_path(cliff, Qs[method])])
    ax2.plot(rc[:, 1] + (0.1 if method == "sarsa" else -0.1), rc[:, 0], marker=m, color=c, label=f"{method} greedy path")
ax2.text(5.5, 3, "C L I F F", ha="center", va="center", color="w"); ax2.text(0, 3.4, "S", ha="center"); ax2.text(11, 3.4, "G", ha="center")
ax2.set_xlabel("column"); ax2.set_ylabel("row"); ax2.set_title("Learned greedy paths"); ax2.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Q-learning learns the optimal (shortest) path along the cliff edge — but while it is still exploring with
$\varepsilon = 0.1$ it keeps falling off, so its return *during training* is far worse. SARSA evaluates the
ε-greedy policy it actually follows, learns that the edge is dangerous for a slipping agent, and takes the
safe upper route: longer, but with higher training returns. Same data, same ε, different target — this is the
on-/off-policy distinction made visible.

**Try it yourself:** make the exploration decay inside `td_control` (e.g. `eps_t = eps / (1 + ep / 20)`) — does SARSA's
greedy path move toward the cliff edge as its own exploration fades? (SARSA converges to the optimal policy only if
exploration vanishes in the limit while every action is still tried infinitely often.) Or set `alpha=0.1` and watch
the learning curves smooth out and slow down.
""")

# ---------------------------------------------------------------- 7. function approximation
nb.md(r"""
## 7. Linear function approximation with semi-gradient TD(0)

A 20-state random walk with $\gamma = 1$: exit left gives $-1$, exit right $+1$, so $V(s)$ rises linearly from
$-0.90$ to $+0.90$. We approximate $\hat V(s; w) = w^\top\phi(s)$ with three feature maps — a cubic polynomial in
the normalised position (4 weights), state aggregation into 4 blocks of 5 states (4 weights), and one-hot
features (20 weights, which recovers tabular TD) — using the semi-gradient update

$$w \leftarrow w + \alpha\,[r + \gamma\, w^\top\phi(s') - w^\top\phi(s)]\,\phi(s).$$
""")

nb.code(r"""
n_chain = 20
P_chain = np.zeros((n_chain, n_chain)); R_chain = np.zeros(n_chain)
for s in range(n_chain):
    for d, reward in ((-1, -1.0), (1, 1.0)):
        if 0 <= s + d < n_chain: P_chain[s, s + d] += 0.5
        else: R_chain[s] += 0.5 * reward
V_true_chain = np.linalg.solve(np.eye(n_chain) - P_chain, R_chain)        # exact values (gamma = 1, absorbing exits)

def chain_episode(rng):
    s = n_chain // 2; traj = []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        if s2 < 0:         traj.append((s, -1.0, None)); return traj
        if s2 >= n_chain:  traj.append((s, +1.0, None)); return traj
        traj.append((s, 0.0, s2)); s = s2

def features(s, kind):
    x = s / (n_chain - 1)
    if kind == "polynomial (cubic)": return np.array([1.0, x, x**2, x**3])
    if kind == "aggregation (4 blocks)": f = np.zeros(4); f[s // 5] = 1.0; return f
    if kind == "one-hot (tabular)": f = np.zeros(n_chain); f[s] = 1.0; return f

def semi_gradient_td0(kind, n_episodes=2000, alpha=0.01, gamma=1.0, seed=0):
    rng_f = np.random.default_rng(seed); w = np.zeros(len(features(0, kind)))
    for _ in range(n_episodes):
        for (s, r, s2) in chain_episode(rng_f):
            phi = features(s, kind)
            target = r + (0.0 if s2 is None else gamma * w @ features(s2, kind))
            w += alpha * (target - w @ phi) * phi                        # gradient of (target - w.phi)^2 wrt w, target held fixed
    return w

plt.plot(V_true_chain, "k-", lw=2, label="true V")
for kind, alpha in (("polynomial (cubic)", 0.01), ("aggregation (4 blocks)", 0.02), ("one-hot (tabular)", 0.05)):
    w = semi_gradient_td0(kind, alpha=alpha)
    V_hat = np.array([w @ features(s, kind) for s in range(n_chain)])
    plt.step(range(n_chain), V_hat, where="mid", label=f"{kind}, {len(w)} weights, RMSE {np.sqrt(np.mean((V_hat - V_true_chain)**2)):.3f}")
    print(f"{kind:24s} weights: {w}")
plt.xlabel("state"); plt.ylabel("value"); plt.title("Semi-gradient TD(0) on a 20-state chain, 2000 episodes"); plt.legend(fontsize=8); plt.show()

# where does semi-gradient TD converge? The TD fixed point solves Phi^T D (R + P Phi w - Phi w) = 0 (gamma = 1 here),
# with D the on-policy state distribution; compare it with the best approximation, the D-weighted projection of V.
start_dist = np.zeros(n_chain); start_dist[n_chain // 2] = 1.0
visits = np.linalg.solve(np.eye(n_chain) - P_chain.T, start_dist)          # expected visits per episode to each state
D = np.diag(visits / visits.sum())
for kind in ("polynomial (cubic)", "aggregation (4 blocks)"):
    Phi = np.array([features(s, kind) for s in range(n_chain)])
    w_td = np.linalg.solve(Phi.T @ D @ (np.eye(n_chain) - P_chain) @ Phi, Phi.T @ D @ R_chain)
    w_best = np.linalg.solve(Phi.T @ D @ Phi, Phi.T @ D @ V_true_chain)
    rmse = lambda w: np.sqrt(np.mean((Phi @ w - V_true_chain) ** 2))
    print(f"{kind:24s} TD fixed point: RMSE {rmse(w_td):.3f}, w = {np.round(w_td, 3) + 0.0}   "
          f"best approximation: RMSE {rmse(w_best):.3f}, w = {np.round(w_best, 3) + 0.0}")
print(f"aggregated chain: a walker in the leftmost block exits left with probability "
      f"{0.5 * D[0, 0] / D.diagonal()[:5].sum():.3f} per step (true chain: 0.5 from state 0)")
""")

nb.md(r"""
The cubic polynomial has the true (linear) value function inside its span, so its TD fixed point is exact; after
2,000 episodes it is within an RMSE of about $0.08$, the rest being slow progress along the nearly collinear
features $x^2$ and $x^3$. Aggregation cannot represent a slope within a block — and TD does not even find the best
staircase: the best approximation (the $D$-weighted projection of $V$) has RMSE $0.14$, while the TD fixed point,
where the learned weights stall, has $0.41$. For state aggregation the TD fixed point is the exact value function of
an *aggregated* chain whose block-to-block transitions are averaged over the states inside each block; in that chain
a walker in the leftmost block exits with probability $0.03$ per step instead of $0.5$ from state 0, so its values
are pulled toward the middle. (The textbook guarantee — TD's error is within a factor $1/(1-\gamma)$ of the best
approximation's — says nothing at $\gamma = 1$; here TD's error is about three times the best.) The one-hot features
are exact but need five times as many weights and more data. Representation *and* bootstrapping set the floor on the
error before learning starts — and each update of the polynomial weights moved *every* state's estimate, which is
both the point of generalisation and the opening for the deadly triad.
""")

# ---------------------------------------------------------------- 8. policy gradients
nb.md(r"""
## 8. Policy gradients: the score-function estimator, baselines and REINFORCE

For a softmax policy over three arms with deterministic rewards $\mu = (1, 2, 3)$ and scores $\theta = 0$, the
exact gradient is $\nabla_\theta J = \pi \odot (\mu - J) = (-\tfrac13, 0, \tfrac13)$. The REINFORCE estimate from a
single pull $a$ is $(\mu_a - b)(\mathbf e_a - \pi)$: unbiased for **every** baseline $b$, because
$\mathbb E_{a\sim\pi}[\nabla\log\pi(a)] = \nabla\sum_a\pi(a) = 0$, but with a variance that depends strongly on $b$.
Then the baseline at work: REINFORCE on a two-context, three-arm contextual bandit whose rewards are offset by
$+10$, with and without a running-mean baseline.
""")

nb.code(r"""
def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

mu = np.array([1.0, 2.0, 3.0]); theta = np.zeros(3); pi = softmax(theta); J = pi @ mu
grad_exact = pi * (mu - J)
print("exact gradient:", grad_exact)

n = 100_000
a = rng.choice(3, size=n, p=pi); score = np.eye(3)[a] - pi            # grad log pi(a) = e_a - pi
baselines = np.linspace(-1, 5, 61); var_mc = []
for b in (0.0, 1.0, 2.0, 3.0):
    est = (mu[a] - b)[:, None] * score
    print(f"baseline b = {b}: mean of estimates {est.mean(axis=0)}   total variance {est.var(axis=0).sum():.3f}")
for b in baselines:
    var_mc.append(((mu[a] - b)[:, None] * score).var(axis=0).sum())
var_exact = [(2/9) * np.sum((mu - b) ** 2) - 2/9 for b in baselines]     # closed form for this bandit

# contextual bandit: 2 contexts x 3 arms, Gaussian rewards around context-dependent means (+ offset)
def contextual_reinforce(n_steps=300, lr=0.05, baseline=False, offset=10.0, seed=0):
    rng_r = np.random.default_rng(seed)
    means = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]]) + offset
    theta = np.zeros((2, 3)); b = np.zeros(2); p_opt = np.zeros(n_steps)
    for t in range(n_steps):
        x = int(rng_r.integers(2)); p = softmax(theta[x]); a = int(rng_r.choice(3, p=p))
        r = means[x, a] + 0.5 * rng_r.standard_normal()
        advantage = r - (b[x] if baseline else 0.0)
        theta[x] += lr * advantage * (np.eye(3)[a] - p)                    # REINFORCE step
        if baseline: b[x] += 0.1 * (r - b[x])                              # running mean of rewards per context
        p_opt[t] = np.mean([softmax(theta[c])[means[c].argmax()] for c in range(2)])
    return p_opt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
ax1.plot(baselines, var_mc, label="Monte Carlo (100k pulls)"); ax1.plot(baselines, var_exact, "k--", label="closed form")
ax1.axvline(J, color="gray", ls=":", label="b = J = 2 (expected reward)")
ax1.set_xlabel("baseline b"); ax1.set_ylabel("total variance of the gradient estimate"); ax1.set_title("Variance of REINFORCE vs baseline (bias is 0 for all b)"); ax1.legend(fontsize=8)
for offset, ls in ((0.0, "--"), (10.0, "-")):
    for use_b, c in ((False, "C3"), (True, "C0")):
        runs = np.array([contextual_reinforce(baseline=use_b, offset=offset, seed=s) for s in range(20)])
        ax2.plot(runs.mean(axis=0), color=c, ls=ls, label=f"{'with' if use_b else 'no'} baseline, offset {offset:+.0f}")
        print(f"offset {offset:+5.1f}, {'with' if use_b else 'no  '} baseline: P(optimal arm) after 300 steps = "
              f"{runs[:, -1].mean():.3f} (sd over 20 seeds {runs[:, -1].std():.3f})")
ax2.set_xlabel("step"); ax2.set_ylabel("probability of the optimal arm (mean of 20 seeds)"); ax2.set_title("Contextual bandit: a baseline rescues REINFORCE from an offset"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
The variance curve is a parabola with its minimum near the expected reward — $0.22$ at $b = 2$ against $2.89$ at
$b = 0$, thirteen times smaller for the same expectation. In the contextual bandit, adding $10$ to every reward
changes nothing about which arm is best, yet REINFORCE without a baseline stalls: every action, good or bad,
receives a large positive push, and the policy's progress depends on the lucky difference of two big numbers.
A running-mean baseline removes the offset and restores the signal.

Finally REINFORCE on a deterministic $4 \times 4$ gridworld (start bottom-left, goal top-right, $-1$ per step,
$+10$ at the goal, episodes cut off after 15 steps) with a tabular softmax policy, comparing no baseline with a
learned state-value baseline $\hat V(s)$ (which turns $G_t$ into an advantage estimate). Like most practical
implementations, the code uses the common $\gamma^t$-free variant of the policy gradient: step $t$'s score is weighted by
$G_t - \hat V(s_t)$ rather than by $\gamma^t\,(G_t - \hat V(s_t))$ as the exact gradient of $J = \mathbb{E}[G_0]$ requires. With
$\gamma = 0.99$ and at most 15 steps the two weightings differ by at most 13%; the simplification is exact only for $\gamma = 1$.
""")

nb.code(r"""
small = GridWorld(4, 4, start=(3, 0), goal=(0, 3), slip=0.0, gamma=0.99)

def reinforce_gridworld(env, n_episodes=300, lr=0.05, gamma=0.99, horizon=15, baseline=None, seed=0):
    rng_g = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions)); V_hat = np.zeros(env.n_states)
    totals = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s = env.start; traj = []
        for t in range(horizon):
            p = softmax(theta[s]); a = int(rng_g.choice(env.n_actions, p=p))
            s2, r, done = env.step(s, a, rng_g); traj.append((s, a, r, p)); s = s2
            if done: break
        G = 0.0; returns = []
        for (_, _, r, _) in reversed(traj):
            G = r + gamma * G; returns.append(G)
        returns = returns[::-1]
        grad = np.zeros_like(theta)
        for (s, a, r, p), G in zip(traj, returns):
            advantage = G - (V_hat[s] if baseline == "state value" else 0.0)
            grad[s] += advantage * (np.eye(env.n_actions)[a] - p)          # gamma^t-free variant (exact: times gamma**t)
        theta += lr * grad                                                 # one gradient step per episode
        for (s, a, r, p), G in zip(traj, returns):
            V_hat[s] += 0.1 * (G - V_hat[s])                               # Monte Carlo critic for the baseline
        totals[ep] = sum(r for (_, _, r, _) in traj)
    return totals, theta

for baseline, c in ((None, "C3"), ("state value", "C0")):
    runs = np.array([reinforce_gridworld(small, baseline=baseline, seed=s)[0] for s in range(10)])
    m = runs.mean(axis=0); finals = runs[:, -50:].mean(axis=1)
    plt.plot(np.convolve(m, np.ones(10) / 10, mode="valid"), color=c, label=f"REINFORCE, {'no baseline' if baseline is None else 'learned V(s) baseline'} (mean of 10 seeds)")
    print(f"{'no baseline        ' if baseline is None else 'state-value baseline'}: mean return over episodes 251-300, per seed: {np.round(finals, 1)}")
    print(f"{'':21s} mean {finals.mean():6.2f}   median {np.median(finals):6.2f}   seeds stuck below 0: {(finals < 0).sum()}")
plt.axhline(5.0, color="k", ls="--", label="optimal return (5 steps of -1, then +10)")
plt.xlabel("episode"); plt.ylabel("undiscounted return (10-episode average, 10 seeds)")
plt.title("REINFORCE on a 4x4 gridworld"); plt.legend(fontsize=8); plt.show()
""")

nb.md(r"""
Nine of the ten no-baseline seeds reach the optimal return of 5 within a couple of hundred episodes — and one
never does: its early returns were all negative, every action it tried was pushed *down*, and the policy
collapsed onto a loop that never reaches the goal. The learned state-value baseline turns returns into
advantages ("better or worse than usual from here"), and all ten seeds converge. Two lessons from Section 10 of
the guide in one plot: the mean curve hides a bimodal outcome, so always look at the per-seed results; and a
baseline matters most when the rewards are all of one sign. The update rule is identical in both cases:
`theta += lr * advantage * (onehot(a) - pi)`, the softmax cross-entropy gradient with the sampled action as the
label and the advantage as the sample weight.

**Try it yourself:** set `lr=0.2` and count the seeds that get stuck. Then add an entropy bonus $c\,H(\pi(\cdot \mid s))$ to
the objective: its gradient with respect to the logits is $-c\,\pi \odot (\log\pi + H)$, so inside the loop over the
trajectory add `H = -(p * np.log(p)).sum(); grad[s] -= 0.01 * p * (np.log(p) + H)` and see whether the stuck seeds recover.
""")

# ---------------------------------------------------------------- 9. DPO / PPO
nb.md(r"""
## 9. The optimal KL-regularised policy, DPO and PPO's clip

For a prompt with four candidate responses, a reference policy $\pi_{\text{ref}}$ and rewards $r$, the policy maximising
$\mathbb E_\pi[r] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ is $\pi^*(y) = \pi_{\text{ref}}(y)\,e^{r(y)/\beta}/Z$ with optimal value
$\beta \log Z$. We verify this by maximising the objective directly over the probability simplex (gradient ascent
on softmax logits), trace the reward–KL frontier as $\beta$ varies, and draw PPO's clipped surrogate.
""")

nb.code(r"""
pi_ref = np.array([0.4, 0.3, 0.2, 0.1]); r_model = np.array([1.0, 2.0, 0.0, 3.0])

def kl(p, q): return float(np.sum(p * np.log(p / q)))
def objective(p, beta): return float(p @ r_model - beta * kl(p, pi_ref))

def optimal_policy(beta):
    w = pi_ref * np.exp(r_model / beta); return w / w.sum(), w.sum()

def maximise_numerically(beta, n_steps=8000, lr=1.0):
    z = np.log(pi_ref).copy()                                       # start from the reference
    for _ in range(n_steps):
        q = softmax(z)
        g_q = r_model - beta * (np.log(q / pi_ref) + 1.0)           # d objective / d q
        z += lr * q * (g_q - q @ g_q)                                # chain rule through the softmax
    return softmax(z)

for beta in (2.0, 1.0, 0.5):
    p_star, Z = optimal_policy(beta); p_num = maximise_numerically(beta)
    print(f"beta = {beta}: closed form {p_star}  numeric {p_num}  max|diff| = {np.abs(p_star - p_num).max():.1e}")
    print(f"           objective at pi*: {objective(p_star, beta):.4f} = beta log Z = {beta*np.log(Z):.4f};  "
          f"at pi_ref: {objective(pi_ref, beta):.4f};  E[r] = {p_star @ r_model:.3f}, KL = {kl(p_star, pi_ref):.3f}")

# DPO's identity: reward differences equal beta * differences of log-ratios (log Z cancels)
p_star, _ = optimal_policy(1.0)
print(f"\nr(4) - r(1) = {r_model[3] - r_model[0]:.3f};  beta[log pi*(4)/ref(4) - log pi*(1)/ref(1)] = "
      f"{np.log(p_star[3]/pi_ref[3]) - np.log(p_star[0]/pi_ref[0]):.3f}")

betas = np.logspace(-1, 1.3, 40)
frontier = np.array([(kl(optimal_policy(b)[0], pi_ref), optimal_policy(b)[0] @ r_model) for b in betas])
true_quality = np.array([1.0, 2.0, 0.0, 1.5])                        # suppose the reward model over-rates response 4
true_frontier = np.array([optimal_policy(b)[0] @ true_quality for b in betas])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
ax1.plot(frontier[:, 0], frontier[:, 1], "o-", ms=3, label="proxy reward E_pi*[r]")
ax1.plot(frontier[:, 0], true_frontier, "s--", ms=3, label="true quality if r(4) is over-rated (1.5, not 3)")
for b in (2.0, 1.0, 0.5, 0.2):
    p_b, _ = optimal_policy(b); ax1.annotate(f"beta={b}", (kl(p_b, pi_ref), p_b @ r_model), textcoords="offset points", xytext=(5, -10), fontsize=8)
ax1.set_xlabel("KL(pi* || pi_ref) in nats"); ax1.set_ylabel("expected reward"); ax1.set_title("Reward-KL frontier of the optimal policy"); ax1.legend(fontsize=8)
rho = np.linspace(0.5, 1.5, 300); eps_clip = 0.2
for A, c in ((1.0, "C2"), (-1.0, "C3")):
    ax2.plot(rho, np.minimum(rho * A, np.clip(rho, 1 - eps_clip, 1 + eps_clip) * A), color=c, label=f"clipped surrogate, advantage A = {A:+.0f}")
    ax2.plot(rho, rho * A, color=c, ls=":", lw=1)
ax2.axvline(1 - eps_clip, color="gray", ls=":"); ax2.axvline(1 + eps_clip, color="gray", ls=":")
ax2.set_xlabel("importance ratio rho = pi_theta / pi_old"); ax2.set_ylabel("surrogate objective"); ax2.set_title("PPO clip (eps = 0.2): flat where the ratio has moved far enough in the advantage's direction"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
Gradient ascent over the simplex lands on the Boltzmann re-weighting $\pi_{\text{ref}}\, e^{r/\beta}/Z$ to six decimals,
and the maximal objective equals $\beta\log Z$ — the two facts DPO rests on. The frontier is concave: the first
tenth of a nat of KL buys a lot of reward, the last nat very little. If the reward model over-rates one
response, the *true* quality peaks at a moderate $\beta$ and then falls while the proxy keeps rising — reward
over-optimisation in miniature, and the reason RLHF runs watch the KL to the reference. PPO's clipped surrogate
is flat beyond $1 \pm \epsilon$ *only* on the side the advantage points to; a bad action whose probability has grown
($A < 0$, $\rho > 1.2$) still receives a full gradient pushing it back.
""")

nb.md(r"""
## Summary

- The return $G_t = \sum_k \gamma^k r_{t+k+1}$ defines the objective; $1/(1-\gamma)$ is the horizon the agent effectively cares about.
- Bandits isolate exploration: ε-greedy has linear regret, UCB1 (Hoeffding bonus) and Thompson sampling (posterior draws) have logarithmic regret and need no tuning of an exploration schedule.
- For a fixed policy the Bellman equation is linear and $V^\pi = (I - \gamma P^\pi)^{-1}R^\pi$; the optimality operator is a $\gamma$-contraction, so value iteration converges geometrically and a stopping rule yields an error bound.
- Policy iteration converges in a few expensive iterations, value iteration in many cheap sweeps whose number grows like $1/(1-\gamma)$.
- Monte Carlo is unbiased and noisy; TD(0) bootstraps, trading bias for variance and winning on finite data. SARSA (on-policy) and Q-learning (off-policy) differ only in the target — and therefore in what they learn.
- Semi-gradient TD with linear features converges on-policy to a fixed point whose quality the representation sets; bootstrapping + off-policy + approximation is the deadly triad.
- The score-function trick gives $\nabla J = \mathbb E[\sum_t \gamma^t\,\nabla\log\pi(a_t \mid s_t)(G_t - b(s_t))]$ for any baseline $b$ (practical code, including ours, often drops the $\gamma^t$); a good baseline cuts variance by an order of magnitude at zero bias.
- The optimal KL-regularised policy is $\pi_{\text{ref}}\,e^{r/\beta}/Z$ (DPO's closed form); PPO's clip removes the incentive to move the policy too far in one update.

**Next:** open `exercises.ipynb` and work through the self-checked problems; the solutions notebook has reference implementations.
""")

nb.save(MOD / "lesson.ipynb")
n_fig = sum(c.source.count("plt.show()") for c in nb.cells if c.cell_type == "code")
print(f"lesson.ipynb: {len(nb.cells)} cells, {n_fig} figures")

# =============================================================================
# EXERCISES — hidden answers are computed by executing the reference solutions
# =============================================================================
SETUP = r"""
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (7, 4)
np.set_printoptions(precision=4, suppress=True)


class GridWorld:
    # A small gridworld with an explicit model P[s, a, s'], R[s, a] and a sampling step().
    # Actions: 0 up, 1 right, 2 down, 3 left. With probability `slip` the move goes sideways
    # (half each way). Bumping into a wall or the border leaves the agent where it is.
    # Entering the goal gives goal_reward, a pit pit_reward, any other move step_reward; goal and pits are terminal.
    ACTIONS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
    ARROWS = ["^", ">", "v", "<"]

    def __init__(self, n_rows, n_cols, start, goal, walls=(), pits=(), slip=0.0,
                 step_reward=-1.0, goal_reward=10.0, pit_reward=-10.0, gamma=0.9):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.n_states, self.n_actions = n_rows * n_cols, 4
        self.start, self.goal = self.idx(*start), self.idx(*goal)
        self.walls = {self.idx(*w) for w in walls}
        self.pits = {self.idx(*p) for p in pits}
        self.terminal = {self.goal} | self.pits
        self.slip, self.step_reward, self.goal_reward, self.pit_reward, self.gamma = slip, step_reward, goal_reward, pit_reward, gamma
        self.P, self.R = self._build_model()
        self.cumP = self.P.cumsum(axis=2)

    def idx(self, r, c): return r * self.n_cols + c
    def rc(self, s): return divmod(s, self.n_cols)

    def _move(self, s, a):
        r, c = self.rc(s); dr, dc = self.ACTIONS[a]; r2, c2 = r + dr, c + dc
        if not (0 <= r2 < self.n_rows and 0 <= c2 < self.n_cols) or self.idx(r2, c2) in self.walls:
            return s
        return self.idx(r2, c2)

    def _reward(self, s2):
        if s2 == self.goal: return self.goal_reward
        if s2 in self.pits: return self.pit_reward
        return self.step_reward

    def _build_model(self):
        S, A = self.n_states, self.n_actions
        P = np.zeros((S, A, S)); R = np.zeros((S, A))
        for s in range(S):
            if s in self.terminal or s in self.walls:
                P[s, :, s] = 1.0; continue
            for a in range(A):
                outcomes = {a: 1.0 - self.slip, (a - 1) % 4: self.slip / 2, (a + 1) % 4: self.slip / 2}
                for a2, prob in outcomes.items():
                    if prob == 0: continue
                    s2 = self._move(s, a2)
                    P[s, a, s2] += prob; R[s, a] += prob * self._reward(s2)
        return P, R

    def step(self, s, a, rng):
        # sample s' ~ P(. | s, a); returns (s', reward, done)
        s2 = int(np.searchsorted(self.cumP[s, a], rng.random()))
        return s2, self._reward(s2), s2 in self.terminal

    def policy_string(self, policy):
        # arrows for non-terminal, non-wall states, '.' elsewhere; row by row
        return "".join("." if (s in self.terminal or s in self.walls) else self.ARROWS[int(policy[s])] for s in range(self.n_states))


# the gridworld used in Exercises 6, 8 and the mini-project
grid = GridWorld(4, 4, start=(3, 0), goal=(0, 3), walls=[(1, 1), (2, 2)], pits=[(0, 2)], slip=0.1, gamma=0.95)

# a 4-state, 2-action MDP used in Exercises 4 and 5 (P4[s, a, s'], R4[s, a])
P4 = np.zeros((4, 2, 4))
P4[0, 0] = [0.5, 0.5, 0.0, 0.0];  P4[0, 1] = [0.0, 0.0, 1.0, 0.0]
P4[1, 0] = [0.0, 0.2, 0.8, 0.0];  P4[1, 1] = [0.6, 0.0, 0.0, 0.4]
P4[2, 0] = [0.0, 0.0, 0.0, 1.0];  P4[2, 1] = [0.3, 0.3, 0.4, 0.0]
P4[3, 0] = [1.0, 0.0, 0.0, 0.0];  P4[3, 1] = [0.0, 0.0, 0.0, 1.0]
R4 = np.array([[1.0, 0.0], [2.0, -1.0], [0.0, 3.0], [-2.0, 0.5]])
"""


def run(solution_src: str, extra_src: str = "") -> dict:
    """Execute the setup, a solution and optional extra code in a fresh namespace; return it."""
    ns: dict = {}
    exec(textwrap.dedent(SETUP), ns)
    exec(textwrap.dedent(solution_src), ns)
    if extra_src:
        exec(textwrap.dedent(extra_src), ns)
    return ns


ex = ExerciseSet(
    "Module 19 — Exercises",
    intro_md=r"""
These exercises follow the order of the concept guide: returns and discounting, bandit algorithms, exact
policy evaluation, Bellman operators and value iteration, model-free learning (TD, Monte Carlo, SARSA,
Q-learning), the score-function gradient, and finally a mini-project that pits dynamic programming,
Q-learning and REINFORCE against each other on the same gridworld. The setup cell defines a small
`GridWorld` class (with an explicit model `P[s, a, s']`, `R[s, a]` and a sampling `step`) and a 4-state MDP
`P4, R4` that several exercises share. Several checks compare against *hidden* values, so compute rather
than guess.
""",
    setup_code=SETUP,
)

# ---------------------------------------------------------------- 1. discounted return (warm-up)
SOL_1 = r"""
def discounted_return(rewards, gamma):
    return float(sum(gamma**k * r for k, r in enumerate(rewards)))

def all_returns(rewards, gamma):
    G = np.zeros(len(rewards)); running = 0.0
    for t in reversed(range(len(rewards))):          # G_t = r_{t+1} + gamma G_{t+1}, from the end
        running = rewards[t] + gamma * running
        G[t] = running
    return G
"""
ns1 = run(SOL_1, r"""
rw = [2, -1, 0, 3, 5, -2, 1]
g09 = discounted_return(rw, 0.9)
g05 = discounted_return(rw, 0.5)
allg = all_returns(rw, 0.9)
g_far = discounted_return([0, 0, 0, 0, 10], 0.8)
""")
ex.add(
    title="Discounted returns", difficulty=1,
    prompt_md=r"""
Implement

* `discounted_return(rewards, gamma)` — the return $G_0 = \sum_{k \ge 0} \gamma^k r_{k+1}$ of a finite reward sequence (a Python list or array, `rewards[k]` is $r_{k+1}$), as a float;
* `all_returns(rewards, gamma)` — a NumPy array with $G_t$ for every $t$, computed **backwards** with the recursion $G_t = r_{t+1} + \gamma G_{t+1}$ (one pass, no nested loops).

Test sequence: `rw = [2, -1, 0, 3, 5, -2, 1]`.
""",
    starter=r"""
def discounted_return(rewards, gamma):
    # YOUR CODE HERE
    return None

def all_returns(rewards, gamma):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_1,
    check=rf"""
rw = [2, -1, 0, 3, 5, -2, 1]
try:
    got = discounted_return([1, 0, 0, 4, 2], 0.5)
except Exception:
    got = None
check_close('guide example: G_0 = 1.625 at gamma = 0.5', got, 1.625)
try:
    got = discounted_return(rw, 0.9)
except Exception:
    got = None
check_hash('G_0 of rw at gamma = 0.9', got, '{sha_of(ns1["g09"])}')
try:
    got = discounted_return(rw, 0.5)
except Exception:
    got = None
check_hash('G_0 of rw at gamma = 0.5', got, '{sha_of(ns1["g05"])}')
try:
    G = np.asarray(all_returns(rw, 0.9), dtype=float)
    ok_shape = G.shape == (7,)
except Exception:
    G, ok_shape = None, False
check('all_returns has one entry per step', ok_shape)
try:
    g0 = discounted_return(rw, 0.9)
except Exception:
    g0 = None
check_close('all_returns[0] equals discounted_return', (G[0] if ok_shape else None), g0)
check_hash('all_returns at gamma = 0.9 (whole array)', (G if ok_shape else None), '{sha_of(ns1["allg"])}')
check_close('recursion holds: G_2 = r_3 + gamma G_3', (G[2] if ok_shape else None), ((rw[2] + 0.9 * G[3]) if ok_shape else None))
try:
    got = discounted_return([0, 0, 0, 0, 10], 0.8)
except Exception:
    got = None
check_hash('a reward of 10 four steps away at gamma = 0.8', got, '{sha_of(ns1["g_far"])}')
""",
    explanation_md=r"""
The forward sum weights the $k$-th reward by $\gamma^k$; the backward pass is the same number computed through
the recursion $G_t = r_{t+1} + \gamma G_{t+1}$, which is how every RL implementation computes returns for a whole
episode in one sweep (and the seed of the Bellman equation). A reward of 10 that arrives four steps late is worth
$10 \cdot 0.8^4 = 4.096$ now — less than half. The common slip is an off-by-one in the exponent: $r_1$ (the first
reward, index 0 in the list) carries weight $\gamma^0 = 1$.
""",
)

# ---------------------------------------------------------------- 2. UCB1 (warm-up)
SOL_2 = r"""
def ucb1_indices(counts, sums, t, c=np.sqrt(2)):
    counts = np.asarray(counts, dtype=float); sums = np.asarray(sums, dtype=float)
    safe = np.maximum(counts, 1.0)
    idx = sums / safe + c * np.sqrt(np.log(t) / safe)
    return np.where(counts > 0, idx, np.inf)          # never-pulled arms have an infinite index

def ucb1_select(counts, sums, t, c=np.sqrt(2)):
    return int(np.argmax(ucb1_indices(counts, sums, t, c)))
"""
ns2 = run(SOL_2, r"""
idx_guide = ucb1_indices([5, 3, 2], [3, 2, 1], 10)
idx_hidden = ucb1_indices([12, 30, 8, 50], [7, 20, 3, 31], 100)
sel_hidden = ucb1_select([12, 30, 8, 50], [7, 20, 3, 31], 100)
sel_c = ucb1_select([12, 30, 8, 50], [7, 20, 3, 31], 100, c=0.2)
""")
ex.add(
    title="The UCB1 index", difficulty=1,
    prompt_md=r"""
Implement

* `ucb1_indices(counts, sums, t, c=np.sqrt(2))` — a NumPy array with $\hat\mu_a + c\sqrt{\ln t / N_a}$ for every arm, where `counts[a]` $= N_a$ and `sums[a]` is the total reward of arm $a$; an arm with $N_a = 0$ must get the index `np.inf` (so it is pulled first) and must **not** raise a division warning or error;
* `ucb1_select(counts, sums, t, c=np.sqrt(2))` — the integer index of the arm with the largest UCB1 index (`np.argmax` breaks ties by the first maximum, which is fine).

Test case from the guide: `counts = [5, 3, 2]`, `sums = [3, 2, 1]`, `t = 10`.
""",
    starter=r"""
def ucb1_indices(counts, sums, t, c=np.sqrt(2)):
    # YOUR CODE HERE
    return None

def ucb1_select(counts, sums, t, c=np.sqrt(2)):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_2,
    check=rf"""
try:
    got = np.asarray(ucb1_indices([5, 3, 2], [3, 2, 1], 10), dtype=float)
except Exception:
    got = None
check_close('guide example: indices (1.560, 1.906, 2.017)', got, {ns2["idx_guide"].tolist()!r}, tol=1e-3)
try:
    got = ucb1_select([5, 3, 2], [3, 2, 1], 10)
except Exception:
    got = None
check('guide example: UCB1 pulls arm 3 (index 2)', got == 2)
try:
    got = np.asarray(ucb1_indices([4, 0, 3], [2, 0, 3], 7), dtype=float)
    ok_inf = np.isinf(got[1]) and np.isfinite(got[0]) and np.isfinite(got[2])
except Exception:
    ok_inf = False
check('an unpulled arm gets an infinite index (and nothing raises)', ok_inf)
try:
    got = ucb1_select([4, 0, 3], [2, 0, 3], 7)
except Exception:
    got = None
check('an unpulled arm is selected first', got == 1)
try:
    got = np.asarray(ucb1_indices([12, 30, 8, 50], [7, 20, 3, 31], 100), dtype=float)
except Exception:
    got = None
check_hash('hidden: indices for counts (12, 30, 8, 50), sums (7, 20, 3, 31), t = 100', got, '{sha_of(ns2["idx_hidden"])}')
try:
    got = ucb1_select([12, 30, 8, 50], [7, 20, 3, 31], 100)
except Exception:
    got = None
check_hash('hidden: the selected arm', got, '{sha_of(ns2["sel_hidden"])}')
try:
    got = ucb1_select([12, 30, 8, 50], [7, 20, 3, 31], 100, c=0.2)
except Exception:
    got = None
check_hash('hidden: the selected arm with a small exploration constant c = 0.2', got, '{sha_of(ns2["sel_c"])}')
""",
    explanation_md=r"""
The index is "empirical mean plus a bonus that shrinks like $1/\sqrt{N_a}$ and grows like $\sqrt{\ln t}$": Hoeffding's
inequality says the true mean exceeds it with probability at most $t^{-4}$ when $c = \sqrt 2$ and rewards lie in
$[0, 1]$. Dividing by `np.maximum(counts, 1)` and then overwriting the unpulled arms with `np.inf` avoids the
division-by-zero warning and guarantees every arm is tried once. In the hidden case the arm with the best
empirical mean (index 1, $0.667$ from 30 pulls) wins only when $c$ is small, e.g. $c = 0.2$; with $c = \sqrt 2$ the bonus
dominates and index 0 (12 pulls, mean $0.583$, UCB $1.459$) narrowly beats the least-pulled arm, index 2 (8 pulls), whose
larger bonus cannot make up for its low mean of $0.375$ (UCB $1.448$). The exploration constant is a genuine tuning
parameter: the theory's $\sqrt 2$ is conservative.
""",
)

# ---------------------------------------------------------------- 3. Thompson sampling (warm-up)
SOL_3 = r"""
def thompson_select(successes, failures, rng):
    draws = rng.beta(np.asarray(successes) + 1.0, np.asarray(failures) + 1.0)   # one draw per arm, in arm order
    return int(np.argmax(draws))

def run_thompson(probs, T, rng):
    k = len(probs); successes = np.zeros(k); failures = np.zeros(k); pulls = np.zeros(k, dtype=int)
    for t in range(T):
        a = thompson_select(successes, failures, rng)
        reward = rng.random() < probs[a]
        successes[a] += reward; failures[a] += 1 - reward; pulls[a] += 1
    return pulls
"""
ns3 = run(SOL_3, r"""
pulls = run_thompson([0.2, 0.5, 0.75, 0.6], 1000, np.random.default_rng(3))
sel7 = thompson_select([10, 1, 5], [2, 1, 5], np.random.default_rng(7))
""")
ex.add(
    title="A Thompson sampling step", difficulty=1,
    prompt_md=r"""
Implement Bernoulli Thompson sampling with a $\text{Beta}(1, 1)$ prior on every arm:

* `thompson_select(successes, failures, rng)` — draw **one** sample per arm from $\text{Beta}(1 + s_a,\, 1 + f_a)$ with a single call `rng.beta(alpha_vector, beta_vector)` (arms in order, so the result is reproducible), and return the integer index of the largest draw;
* `run_thompson(probs, T, rng)` — simulate `T` rounds on a Bernoulli bandit with success probabilities `probs` (reward `rng.random() < probs[a]`, drawn **after** the selection in each round), updating the counts, and return an integer array with the number of pulls of each arm.

The checks run `run_thompson([0.2, 0.5, 0.75, 0.6], 1000, np.random.default_rng(3))` and look at how the pulls are distributed.
""",
    starter=r"""
def thompson_select(successes, failures, rng):
    # YOUR CODE HERE
    return None

def run_thompson(probs, T, rng):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_3,
    check=rf"""
try:
    sel = [thompson_select([50, 0], [0, 50], np.random.default_rng(i)) for i in range(20)]
    ok = all(s == 0 for s in sel)
except Exception:
    ok = False
check('an arm with 50/50 successes always beats one with 0/50', ok)
try:
    got = thompson_select([10, 1, 5], [2, 1, 5], np.random.default_rng(7))
except Exception:
    got = None
check_hash('hidden: the arm chosen with rng seed 7 for successes (10, 1, 5), failures (2, 1, 5)', got, '{sha_of(ns3["sel7"])}')
try:
    pulls = np.asarray(run_thompson([0.2, 0.5, 0.75, 0.6], 1000, np.random.default_rng(3)))
    ok_shape = pulls.shape == (4,) and int(pulls.sum()) == 1000
except Exception:
    pulls, ok_shape = None, False
check('pulls has one entry per arm and sums to T', ok_shape)
check('the best arm (p = 0.75) gets more than 85% of the pulls', ok_shape and pulls[2] > 850)
check('the worst arm (p = 0.2) gets fewer than 40 pulls', ok_shape and pulls[0] < 40)
check('every arm is pulled at least once', ok_shape and bool((pulls >= 1).all()))
check_hash('hidden: the exact pull counts for seed 3', (pulls if ok_shape else None), '{sha_of(ns3["pulls"])}')
""",
    explanation_md=r"""
With a $\text{Beta}(1,1)$ prior the posterior of arm $a$ after $s_a$ successes and $f_a$ failures is
$\text{Beta}(1 + s_a, 1 + f_a)$ — the conjugate update of Module 11 — and Thompson sampling pulls the arm whose
*sample* is largest, which happens with exactly the posterior probability that the arm is best. Over 1,000
rounds the best arm collects well over 90% of the pulls while the obviously bad arm is abandoned after a
handful of tries; the second-best arm ($0.6$) keeps a few dozen pulls because its posterior overlaps the best
arm's for a while. The exact counts depend on consuming the generator in the specified order (one `beta` call,
then one `random` call per round); that is why the prompt pins the order down.
""",
)

# ---------------------------------------------------------------- 4. exact policy evaluation (core)
SOL_4 = r"""
def policy_evaluation_exact(P, R, policy, gamma):
    S = P.shape[0]; idx = np.arange(S)
    P_pi = P[idx, policy]                       # P_pi[s, s'] = P(s' | s, pi(s))
    R_pi = R[idx, policy]
    return np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)

def q_from_v(P, R, V, gamma):
    return R + gamma * np.einsum("sat,t->sa", P, V)

def greedy_policy(Q):
    return Q.argmax(axis=1)
"""
ns4 = run(SOL_4, r"""
pol = np.array([0, 1, 1, 0])
V4 = policy_evaluation_exact(P4, R4, pol, 0.9)
Q4 = q_from_v(P4, R4, V4, 0.9)
g4 = greedy_policy(Q4)
V_machine = policy_evaluation_exact(
    np.array([[[0.75, 0.25, 0.0], [1, 0, 0]], [[0.0, 0.5, 0.5], [1, 0, 0]], [[0, 0, 1.0], [1, 0, 0]]]),
    np.array([[10., -2.], [6., -2.], [0., -12.]]), np.array([0, 0, 0]), 0.8)
""")
ex.add(
    title="Exact policy evaluation and the action values", difficulty=2,
    prompt_md=r"""
For a finite MDP stored as `P[s, a, s']` and `R[s, a]` and a **deterministic** policy given as an integer array
`policy[s]`, implement

* `policy_evaluation_exact(P, R, policy, gamma)` — $V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$ via `np.linalg.solve` (build $P^\pi$ and $R^\pi$ by picking, for each state, the row/entry of the chosen action; no loops over states needed: use fancy indexing `P[np.arange(S), policy]`);
* `q_from_v(P, R, V, gamma)` — the array $Q(s, a) = R(s, a) + \gamma\sum_{s'} P(s' \mid s, a)\, V(s')$;
* `greedy_policy(Q)` — `Q.argmax(axis=1)`.

Apply them to the 4-state MDP `P4, R4` from the setup with `policy = [0, 1, 1, 0]` and $\gamma = 0.9$.
""",
    starter=r"""
def policy_evaluation_exact(P, R, policy, gamma):
    # YOUR CODE HERE
    return None

def q_from_v(P, R, V, gamma):
    # YOUR CODE HERE
    return None

def greedy_policy(Q):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_4,
    check=rf"""
P_m = np.array([[[0.75, 0.25, 0.0], [1, 0, 0]], [[0.0, 0.5, 0.5], [1, 0, 0]], [[0, 0, 1.0], [1, 0, 0]]])
R_m = np.array([[10., -2.], [6., -2.], [0., -12.]])
try:
    got = np.asarray(policy_evaluation_exact(P_m, R_m, np.array([0, 0, 0]), 0.8), dtype=float)
except Exception:
    got = None
check_close('the guide machine, always run: V = (30, 10, 0)', got, [30.0, 10.0, 0.0], tol=1e-8)
pol = np.array([0, 1, 1, 0])
try:
    V = np.asarray(policy_evaluation_exact(P4, R4, pol, 0.9), dtype=float)
    ok_shape = V.shape == (4,)
except Exception:
    V, ok_shape = None, False
check('V has shape (4,)', ok_shape)
check_hash('hidden: V^pi for policy (0, 1, 1, 0) on P4, R4', (V if ok_shape else None), '{sha_of(ns4["V4"])}')
try:
    resid = V - (R4[np.arange(4), pol] + 0.9 * P4[np.arange(4), pol] @ V)
    ok_bellman = ok_shape and np.abs(resid).max() < 1e-8
except Exception:
    ok_bellman = False
check('V satisfies the Bellman expectation equation (residual < 1e-8)', ok_bellman)
try:
    Q = np.asarray(q_from_v(P4, R4, V, 0.9), dtype=float)
    ok_q = Q.shape == (4, 2)
except Exception:
    Q, ok_q = None, False
check('Q has shape (4, 2)', ok_q)
check_close('Q(s, pi(s)) equals V(s)', (Q[np.arange(4), pol] if ok_q else None), (V if ok_shape else None))
check_hash('hidden: Q^pi (whole table)', (Q if ok_q else None), '{sha_of(ns4["Q4"])}')
try:
    got = np.asarray(greedy_policy(Q))
except Exception:
    got = None
check_hash('hidden: the greedy policy with respect to Q^pi', got, '{sha_of(ns4["g4"])}')
""",
    explanation_md=r"""
Fixing a deterministic policy picks one row of `P[s]` and one entry of `R[s]` per state, which fancy indexing
`P[np.arange(S), policy]` does in one line; the Bellman expectation equation is then the linear system
$(I - \gamma P^\pi)V = R^\pi$, always solvable because $\gamma P^\pi$ has spectral radius at most $\gamma < 1$.
$Q^\pi$ needs one more backup through the model, and the greedy policy differs from $\pi$ in several states — in the
4-state MDP, acting greedily with respect to $Q^\pi$ switches the action in states 0, 1 and 3, the first step of
policy iteration. The classic mistake is to invert the matrix explicitly or to loop over states and actions in
Python; `solve` plus `einsum` is both faster and clearer.
""",
)

# ---------------------------------------------------------------- 5. Bellman optimality + contraction (core)
SOL_5 = r"""
def bellman_optimality_operator(V, P, R, gamma):
    return (R + gamma * np.einsum("sat,t->sa", P, V)).max(axis=1)

def contraction_ratio(U, V, P, R, gamma):
    TU = bellman_optimality_operator(U, P, R, gamma); TV = bellman_optimality_operator(V, P, R, gamma)
    return float(np.abs(TU - TV).max() / np.abs(U - V).max())

def value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000):
    V = np.zeros(P.shape[0])
    for n in range(1, max_iter + 1):
        V_new = bellman_optimality_operator(V, P, R, gamma)
        if np.abs(V_new - V).max() < tol:
            V = V_new; break
        V = V_new
    Q = R + gamma * np.einsum("sat,t->sa", P, V)
    return V, Q.argmax(axis=1), n
"""
ns5 = run(SOL_5, r"""
T0 = bellman_optimality_operator(np.zeros(4), P4, R4, 0.9)
T2 = bellman_optimality_operator(T0, P4, R4, 0.9)
Vs, pis, n_it = value_iteration(P4, R4, 0.9)
rng_c = np.random.default_rng(5)
ratios = [contraction_ratio(rng_c.normal(size=4) * 10, rng_c.normal(size=4) * 10, P4, R4, 0.9) for _ in range(200)]
max_ratio = max(ratios)
""")
ex.add(
    title="The Bellman optimality operator, its contraction and value iteration", difficulty=2,
    prompt_md=r"""
Implement

* `bellman_optimality_operator(V, P, R, gamma)` — $(TV)(s) = \max_a \big[R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V(s')\big]$ as an array over states;
* `contraction_ratio(U, V, P, R, gamma)` — the number $\lVert TU - TV\rVert_\infty / \lVert U - V\rVert_\infty$ for two value vectors $U \ne V$;
* `value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000)` — start from $V = 0$, repeat $V \leftarrow TV$ until the largest change is below `tol`, and return the triple `(V_star, greedy_policy, n_sweeps)` where `n_sweeps` counts the applications of $T$ (including the last one) and `greedy_policy` is the argmax of $Q^*$ computed from the final $V$.

Use the 4-state MDP `P4, R4` with $\gamma = 0.9$.
""",
    starter=r"""
def bellman_optimality_operator(V, P, R, gamma):
    # YOUR CODE HERE
    return None

def contraction_ratio(U, V, P, R, gamma):
    # YOUR CODE HERE
    return None

def value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000):
    # YOUR CODE HERE
    return None, None, None
""",
    solution=SOL_5,
    check=rf"""
try:
    got = np.asarray(bellman_optimality_operator(np.zeros(4), P4, R4, 0.9), dtype=float)
except Exception:
    got = None
check_close('T applied to V = 0 is the best immediate reward per state', got, R4.max(axis=1))
try:
    got = np.asarray(bellman_optimality_operator(bellman_optimality_operator(np.zeros(4), P4, R4, 0.9), P4, R4, 0.9), dtype=float)
except Exception:
    got = None
check_hash('hidden: T^2 applied to 0', got, '{sha_of(ns5["T2"])}')
try:
    rng_c = np.random.default_rng(5)
    ratios = [contraction_ratio(rng_c.normal(size=4) * 10, rng_c.normal(size=4) * 10, P4, R4, 0.9) for _ in range(200)]
    ok = len(ratios) == 200 and all(np.isfinite(ratios)) and max(ratios) <= 0.9 + 1e-9
except Exception:
    ok = False
check('no pair of value vectors is pushed apart by more than gamma = 0.9', ok)
try:
    V_star, pi_star, n_sweeps = value_iteration(P4, R4, 0.9)
    V_star = np.asarray(V_star, dtype=float); pi_star = np.asarray(pi_star)
    ok_shape = V_star.shape == (4,) and pi_star.shape == (4,)
except Exception:
    V_star, pi_star, n_sweeps, ok_shape = None, None, None, False
check('value iteration returns V* and a policy of the right shapes', ok_shape)
try:
    fixed = ok_shape and np.abs(bellman_optimality_operator(V_star, P4, R4, 0.9) - V_star).max() < 1e-8
except Exception:
    fixed = False
check('V* is a fixed point of T (|TV* - V*| < 1e-8)', fixed)
check_hash('hidden: V* (rounded to 3 decimals)', (V_star if ok_shape else None), '{sha_of(ns5["Vs"])}')
check_hash('hidden: the optimal policy', (pi_star if ok_shape else None), '{sha_of(ns5["pis"])}')
check('number of sweeps is in the range the contraction predicts (150-300 for tol 1e-10)', (n_sweeps is not None) and 150 <= int(n_sweeps) <= 300)
""",
    explanation_md=r"""
$T$ is the right-hand side of the Bellman optimality equation; it is a $\gamma$-contraction in the max norm
because the maximum of differences is bounded by the difference of maxima and each row of $P$ averages (never
amplifies) a vector. Hence the random-pair ratios never exceed $0.9$, iterating $T$ from any start converges to
the unique fixed point $V^*$, and the error after $k$ sweeps is at most $\gamma^k$ times the initial error — about
$\ln(10^{10} \cdot 23)/\ln(1/0.9) \approx 250$ sweeps for a tolerance of $10^{-10}$ from values of order 20, matching
the sweep count the check accepts. The greedy policy with respect to $V^*$ is optimal; note that it is
$(1, 0, 1, 0)$, not the greedy policy of Exercise 4, which was only one improvement step away from
$(0, 1, 1, 0)$.
""",
)

# ---------------------------------------------------------------- 6. value iteration on the gridworld (core)
SOL_6 = r"""
def solve_gridworld(env, tol=1e-10):
    V = np.zeros(env.n_states)
    for n in range(1, 100_000):
        Q = env.R + env.gamma * np.einsum("sat,t->sa", env.P, V)
        V_new = Q.max(axis=1)
        if np.abs(V_new - V).max() < tol:
            V = V_new; break
        V = V_new
    Q = env.R + env.gamma * np.einsum("sat,t->sa", env.P, V)
    return V, Q.argmax(axis=1), n
"""
ns6 = run(SOL_6, r"""
V_g, pi_g, n_g = solve_gridworld(grid)
pol_str = grid.policy_string(pi_g)
v_start = float(V_g[grid.start])
grid_slow = GridWorld(4, 4, start=(3, 0), goal=(0, 3), walls=[(1, 1), (2, 2)], pits=[(0, 2)], slip=0.1, gamma=0.5)
v_start_slow = float(solve_gridworld(grid_slow)[0][grid_slow.start])
""")
ex.add(
    title="Value iteration on the gridworld", difficulty=2,
    prompt_md=r"""
Implement `solve_gridworld(env, tol=1e-10)` that runs value iteration on a `GridWorld` instance using its model
`env.P`, `env.R`, `env.gamma` and returns `(V_star, policy, n_sweeps)` — the optimal values (array of length
`env.n_states`), the greedy policy (integer array of actions) and the number of sweeps. You may reuse your
`value_iteration` from Exercise 5 or write it afresh.

Apply it to the setup's `grid` ($4 \times 4$, walls at (1,1) and (2,2), a pit at (0,2), goal at (0,3), 10% slip,
$\gamma = 0.95$) and look at the result:

```python
V_star, policy, n = solve_gridworld(grid)
print(V_star.reshape(4, 4)); print(np.array(list(grid.policy_string(policy))).reshape(4, 4))
```

Terminal states and walls have value 0 (they are absorbing with zero reward in this model). The check hashes
the policy string over non-terminal states, so make sure your greedy policy is the argmax of
$R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^*(s')$ — no ties occur in this gridworld.
""",
    starter=r"""
def solve_gridworld(env, tol=1e-10):
    # YOUR CODE HERE
    return None, None, None
""",
    solution=SOL_6,
    check=rf"""
try:
    V_star, policy, n = solve_gridworld(grid)
    V_star = np.asarray(V_star, dtype=float); policy = np.asarray(policy)
    ok_shape = V_star.shape == (grid.n_states,) and policy.shape == (grid.n_states,)
except Exception:
    V_star, policy, n, ok_shape = None, None, None, False
check('shapes are right', ok_shape)
check('terminal states and walls have value 0', ok_shape and all(abs(V_star[s]) < 1e-12 for s in grid.terminal | grid.walls))
try:
    Q = grid.R + grid.gamma * np.einsum("sat,t->sa", grid.P, V_star)
    ok_fixed = ok_shape and np.abs(Q.max(axis=1) - V_star).max() < 1e-8
except Exception:
    ok_fixed = False
check('V* is a fixed point of the Bellman optimality operator', ok_fixed)
check_hash('hidden: V*(start) to 3 decimals', (float(V_star[grid.start]) if ok_shape else None), '{sha_of(ns6["v_start"])}')
check_hash('hidden: the whole V* table', (V_star if ok_shape else None), '{sha_of(ns6["V_g"])}')
try:
    got = grid.policy_string(policy) if ok_shape else None
except Exception:
    got = None
check_hash('hidden: the greedy policy string', got, '{sha_of(ns6["pol_str"])}')
check('the greedy policy enters the goal from the state below it', ok_shape and int(policy[grid.idx(1, 3)]) == 0)
try:
    grid_slow = GridWorld(4, 4, start=(3, 0), goal=(0, 3), walls=[(1, 1), (2, 2)], pits=[(0, 2)], slip=0.1, gamma=0.5)
    got = float(np.asarray(solve_gridworld(grid_slow)[0], dtype=float)[grid_slow.start])
except Exception:
    got = None
check_hash('hidden: V*(start) when gamma = 0.5', got, '{sha_of(ns6["v_start_slow"])}')
""",
    explanation_md=r"""
Nothing new is needed beyond Exercise 5: the gridworld's model is just a larger `P` and `R`. The optimal values
drop from about $9.7$ next to the goal to about $2.2$ at the start, and the arrows route the agent right along
the bottom row and up the last column — away from the pit, which with 10% slip is dangerous to pass. With
$\gamma = 0.5$ the same path is optimal but worth less than zero from the start: the $+10$ arrives on the sixth move,
weighted by $0.5^5 \approx 0.03$, so it is worth only about $0.31$ against $-1.94$ for the five $-1$ steps before it.
Value iteration needs only about 36 sweeps — not the roughly 500 that a contraction factor of $0.95$ would suggest for a
tolerance of $10^{-10}$ — because the task is episodic: errors propagate only along paths that have not yet reached
the goal or the pit, so the effective factor per sweep is $\gamma\rho \approx 0.34$, where $\rho \approx 0.35$ is the
spectral radius of the optimal policy's transition matrix among the non-terminal states.
""",
)

# ---------------------------------------------------------------- 7. TD(0) and first-visit MC updates (core)
SOL_7 = r"""
def td0_update(V, s, r, s_next, done, alpha, gamma):
    V = V.copy()
    target = r + (0.0 if done else gamma * V[s_next])
    V[s] += alpha * (target - V[s])
    return V

def first_visit_mc_update(V, episode, alpha, gamma):
    V = V.copy(); G = 0.0; first_return = {}
    for (s, r) in reversed(episode):          # walk backwards; the earliest visit overwrites later ones
        G = r + gamma * G
        first_return[s] = G
    for s, G in first_return.items():
        V[s] += alpha * (G - V[s])
    return V

def random_walk_episode(rng):
    # states 0..4, start at 2; exit left (-> -1) gives 0, exit right (-> 5) gives 1
    s = 2; traj = []
    while True:
        s_next = s + (1 if rng.random() < 0.5 else -1)
        r = 1.0 if s_next == 5 else 0.0
        done = s_next < 0 or s_next == 5
        traj.append((s, r, s_next, done)); s = s_next
        if done: return traj

def td0_on_random_walk(n_episodes, alpha, rng):
    V = np.full(5, 0.5)
    for _ in range(n_episodes):
        for (s, r, s_next, done) in random_walk_episode(rng):
            V = td0_update(V, s, r, s_next, done, alpha, 1.0)
    return V
"""
ns7 = run(SOL_7, r"""
V_td = td0_on_random_walk(100, 0.1, np.random.default_rng(5))
V_mc_ex = first_visit_mc_update(np.zeros(3), [(0, 0.0), (1, 0.0), (0, 1.0), (2, 2.0)], 0.5, 0.9)
""")
ex.add(
    title="TD(0) and first-visit Monte Carlo updates", difficulty=2,
    prompt_md=r"""
Implement the two prediction updates and apply TD(0) to the five-state random walk:

* `td0_update(V, s, r, s_next, done, alpha, gamma)` — return a **copy** of `V` with $V(s) \leftarrow V(s) + \alpha\,[r + \gamma V(s') - V(s)]$; if `done` is `True` the bootstrap term $\gamma V(s')$ is **zero** (the episode ended);
* `first_visit_mc_update(V, episode, alpha, gamma)` — `episode` is a list of `(state, reward)` pairs in time order, the reward being the one received *after* leaving that state; compute the return $G_t$ following each step (backwards, $G_t = r_{t+1} + \gamma G_{t+1}$), and for every state update $V(s) \leftarrow V(s) + \alpha\,(G - V(s))$ **once**, using the return of its *first* visit; return the updated copy;
* `random_walk_episode(rng)` — states $0, \dots, 4$, start at 2, move left or right with probability $\tfrac12$ (`rng.random() < 0.5` means right), reward 1 on exiting to the right (to position 5), 0 otherwise; return a list of `(s, r, s_next, done)` tuples;
* `td0_on_random_walk(n_episodes, alpha, rng)` — start from $V = 0.5$ everywhere, run `n_episodes` episodes applying `td0_update` after every step with $\gamma = 1$, and return $V$.

The true values are $V(s) = (s+1)/6$.
""",
    starter=r"""
def td0_update(V, s, r, s_next, done, alpha, gamma):
    # YOUR CODE HERE
    return None

def first_visit_mc_update(V, episode, alpha, gamma):
    # YOUR CODE HERE
    return None

def random_walk_episode(rng):
    # YOUR CODE HERE
    return None

def td0_on_random_walk(n_episodes, alpha, rng):
    # YOUR CODE HERE
    return None
""",
    solution=SOL_7,
    check=rf"""
V0 = np.array([2.0, 1.0, 3.0])
try:
    got = np.asarray(td0_update(V0, 0, 1.0, 2, False, 0.1, 0.9), dtype=float)
except Exception:
    got = None
check_close('TD(0): 2 + 0.1 (1 + 0.9 * 3 - 2) = 2.17, other entries untouched', got, [2.17, 1.0, 3.0])
try:
    got = np.asarray(td0_update(V0, 0, 1.0, 2, True, 0.1, 0.9), dtype=float)
except Exception:
    got = None
check_close('TD(0) at a terminal transition: no bootstrap', got, [1.9, 1.0, 3.0])
check('td0_update does not modify its input in place', bool(np.allclose(V0, [2.0, 1.0, 3.0])))
try:
    got = np.asarray(first_visit_mc_update(np.zeros(3), [(0, 0.0), (1, 0.0), (0, 1.0), (2, 2.0)], 0.5, 0.9), dtype=float)
except Exception:
    got = None
check_close('first-visit MC on a 4-step episode with a revisited state', got, {ns7["V_mc_ex"].tolist()!r}, tol=1e-9)
try:
    traj = random_walk_episode(np.random.default_rng(1))
    ok_traj = traj[0][0] == 2 and bool(traj[-1][3]) and all(not t[3] for t in traj[:-1]) \
              and all(abs(t[2] - t[0]) == 1 for t in traj) and traj[-1][1] == (1.0 if traj[-1][2] == 5 else 0.0)
except Exception:
    ok_traj = False
check('random_walk_episode starts at 2, moves by 1 and ends exactly once', bool(ok_traj))
try:
    V_td = np.asarray(td0_on_random_walk(100, 0.1, np.random.default_rng(5)), dtype=float)
    ok_shape = V_td.shape == (5,)
except Exception:
    V_td, ok_shape = None, False
check('TD(0) estimate has 5 entries', ok_shape)
check('TD(0) after 100 episodes is within 0.15 of the truth everywhere', ok_shape and bool(np.abs(V_td - np.arange(1, 6) / 6).max() < 0.15))
check('TD(0) values increase from left to right', ok_shape and bool((np.diff(V_td) > 0).all()))
check_hash('hidden: the exact TD(0) estimate for seed 5', (V_td if ok_shape else None), '{sha_of(ns7["V_td"])}')
""",
    explanation_md=r"""
TD(0) moves $V(s)$ a fraction $\alpha$ toward the one-step target $r + \gamma V(s')$; at a terminal transition there
is no next value, so the target is just $r$ — forgetting the `done` flag is the most common TD bug and it makes
the agent "see" value beyond the end of the episode. First-visit Monte Carlo walks the episode backwards
accumulating $G$; writing `first_return[s] = G` on every visit and letting later (earlier-in-time) visits
overwrite leaves exactly the return of the first visit. In the test episode state 0 is visited at $t = 0$ and
$t = 2$; its first-visit return is $0 + 0.9\,(0 + 0.9\,(1 + 0.9 \cdot 2)) = 2.268$, so $V(0) = 0.5 \cdot 2.268 = 1.134$.
On the random walk 100 episodes of TD(0) with $\alpha = 0.1$ land within about $0.1$ of $1/6, \dots, 5/6$.
""",
)

# ---------------------------------------------------------------- 8. Q-learning and SARSA (core)
SOL_8 = r"""
def q_learning_update(Q, s, a, r, s_next, done, alpha, gamma):
    Q = Q.copy()
    target = r + (0.0 if done else gamma * Q[s_next].max())
    Q[s, a] += alpha * (target - Q[s, a])
    return Q

def sarsa_update(Q, s, a, r, s_next, a_next, done, alpha, gamma):
    Q = Q.copy()
    target = r + (0.0 if done else gamma * Q[s_next, a_next])
    Q[s, a] += alpha * (target - Q[s, a])
    return Q

def q_learning_run(env, n_episodes, alpha, gamma, eps, rng, max_steps=200):
    Q = np.zeros((env.n_states, env.n_actions)); returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s = env.start; total = 0.0
        for _ in range(max_steps):
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                a = int(np.argmax(Q[s]))
            s_next, r, done = env.step(s, a, rng); total += r
            Q = q_learning_update(Q, s, a, r, s_next, done, alpha, gamma)
            s = s_next
            if done: break
        returns[ep] = total
    return Q, returns
"""
ns8 = run(SOL_8 + "\n" + SOL_4, r"""
Q0 = np.array([[0.5, 0.0], [0.2, 0.8], [-0.3, 0.1]])
q_new = q_learning_update(Q0, 0, 0, -1.0, 1, False, 0.5, 0.9)[0, 0]
s_new = sarsa_update(Q0, 0, 0, -1.0, 1, 0, False, 0.5, 0.9)[0, 0]
Q_run, rets = q_learning_run(grid, 400, 0.2, grid.gamma, 0.1, np.random.default_rng(11))
V_greedy = policy_evaluation_exact(grid.P, grid.R, Q_run.argmax(axis=1), grid.gamma)
v_start_greedy = float(V_greedy[grid.start])
""")
assert ns8["v_start_greedy"] > 2.10, ns8["v_start_greedy"]
ex.add(
    title="Q-learning and SARSA", difficulty=2,
    prompt_md=r"""
Implement the two control updates and a complete tabular Q-learning agent for the gridworld:

* `q_learning_update(Q, s, a, r, s_next, done, alpha, gamma)` — returns a copy of `Q` with $Q(s,a) \leftarrow Q(s,a) + \alpha\,[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]$ (no bootstrap when `done`);
* `sarsa_update(Q, s, a, r, s_next, a_next, done, alpha, gamma)` — the same with the target $r + \gamma Q(s', a')$;
* `q_learning_run(env, n_episodes, alpha, gamma, eps, rng, max_steps=200)` — start from $Q = 0$; in each episode start at `env.start`, act ε-greedily (**first** draw `rng.random() < eps` to decide whether to explore; if exploring draw `rng.integers(env.n_actions)`, otherwise take `np.argmax(Q[s])`), step the environment with `env.step(s, a, rng)`, apply `q_learning_update`, stop at a terminal state or after `max_steps` steps; return `(Q, returns)` where `returns[ep]` is the undiscounted sum of rewards of episode `ep`.

Run `q_learning_run(grid, 400, alpha=0.2, gamma=grid.gamma, eps=0.1, rng=np.random.default_rng(11))` and compare the greedy
policy you obtain with the optimal one from Exercise 6 — evaluate it *exactly* with the linear solve of Exercise 4
(the check does this for you).
""",
    starter=r"""
def q_learning_update(Q, s, a, r, s_next, done, alpha, gamma):
    # YOUR CODE HERE
    return None

def sarsa_update(Q, s, a, r, s_next, a_next, done, alpha, gamma):
    # YOUR CODE HERE
    return None

def q_learning_run(env, n_episodes, alpha, gamma, eps, rng, max_steps=200):
    # YOUR CODE HERE
    return None, None
""",
    solution=SOL_8,
    check=rf"""
Q0 = np.array([[0.5, 0.0], [0.2, 0.8], [-0.3, 0.1]])
try:
    got = float(q_learning_update(Q0, 0, 0, -1.0, 1, False, 0.5, 0.9)[0, 0])
except Exception:
    got = None
check_close('Q-learning: 0.5 + 0.5 (-1 + 0.9 * 0.8 - 0.5) = 0.11', got, 0.11)
try:
    got = float(sarsa_update(Q0, 0, 0, -1.0, 1, 0, False, 0.5, 0.9)[0, 0])
except Exception:
    got = None
check_close('SARSA with a_next = 0: 0.5 + 0.5 (-1 + 0.9 * 0.2 - 0.5) = -0.16', got, -0.16)
try:
    got = float(q_learning_update(Q0, 2, 1, 4.0, 0, True, 0.5, 0.9)[2, 1])
except Exception:
    got = None
check_close('Q-learning at a terminal transition: no bootstrap', got, 0.1 + 0.5 * (4.0 - 0.1))
check('updates leave the input table unchanged', bool(np.allclose(Q0, [[0.5, 0.0], [0.2, 0.8], [-0.3, 0.1]])))
try:
    Q_run, rets = q_learning_run(grid, 400, 0.2, grid.gamma, 0.1, np.random.default_rng(11))
    Q_run = np.asarray(Q_run, dtype=float); rets = np.asarray(rets, dtype=float)
    ok_shape = Q_run.shape == (grid.n_states, grid.n_actions) and rets.shape == (400,)
except Exception:
    Q_run, rets, ok_shape = None, None, False
check('Q table and returns have the right shapes', ok_shape)
check('the agent improves: mean return of the last 100 episodes exceeds that of the first 100', ok_shape and rets[-100:].mean() > rets[:100].mean() + 2)
try:
    S = grid.n_states; idx = np.arange(S); pol = Q_run.argmax(axis=1)
    V_greedy = np.linalg.solve(np.eye(S) - grid.gamma * grid.P[idx, pol], grid.R[idx, pol])
    v_start = float(V_greedy[grid.start])
except Exception:
    v_start = None
check('the learned greedy policy is near-optimal from the start state (exact value > 2.10; optimum 2.162)', v_start is not None and v_start > 2.10)
check_hash('hidden: exact value of the learned greedy policy at the start (seed 11), 3 decimals', v_start, '{sha_of(ns8["v_start_greedy"])}')
""",
    explanation_md=r"""
The two updates differ in one symbol: Q-learning bootstraps from $\max_{a'} Q(s', a')$ — the value of the greedy
continuation, whatever the agent does next — while SARSA bootstraps from the action actually taken, so it learns
the value of its own exploratory behaviour. In the hand example the same transition raises $Q(0, 0)$ to $0.11$
under Q-learning and lowers it to $-0.16$ under SARSA, because the exploratory next action was the worse one. The
full run reaches a near-optimal greedy policy in 400 episodes; its *exact* value from the start, computed by the
linear solve, is a far better yardstick than the noisy training returns, which include the cost of ε-exploration.
The hidden hash requires the exact random-number order described in the prompt (explore? → which arm → step),
since any other order changes the trajectory; that fragility is itself a lesson about seeds in RL.
""",
)

# ---------------------------------------------------------------- 9. score-function estimator (stretch)
SOL_9 = r"""
def softmax(z):
    z = z - np.max(z); e = np.exp(z); return e / e.sum()

def exact_policy_gradient(theta, mu):
    p = softmax(theta)
    return p * (mu - p @ mu)                       # d/dtheta_j sum_a pi_a mu_a = pi_j (mu_j - J)

def score_function_estimate(theta, mu, n_samples, rng, baseline=0.0):
    p = softmax(theta); k = len(mu)
    a = rng.choice(k, size=n_samples, p=p)
    scores = np.eye(k)[a] - p                       # grad log pi(a) = e_a - pi, one row per sample
    per_sample = (mu[a] - baseline)[:, None] * scores
    return per_sample.mean(axis=0), float(per_sample.var(axis=0).sum())
"""
ns9 = run(SOL_9, r"""
theta = np.array([0.5, -0.5, 0.0, 1.0]); mu = np.array([1.0, 3.0, 2.0, 0.5])
g_exact = exact_policy_gradient(theta, mu)
J = float(softmax(theta) @ mu)
m0, v0 = score_function_estimate(theta, mu, 50_000, np.random.default_rng(1), 0.0)
mJ, vJ = score_function_estimate(theta, mu, 50_000, np.random.default_rng(1), J)
""")
ex.add(
    title="The log-derivative trick: unbiased for every baseline, variance depends on it", difficulty=3,
    prompt_md=r"""
A softmax policy over $K$ arms has $\pi_\theta(a) = e^{\theta_a}/\sum_b e^{\theta_b}$ and deterministic rewards $\mu_a$,
so $J(\theta) = \sum_a \pi_\theta(a)\mu_a$. Implement

* `softmax(z)` (numerically stable);
* `exact_policy_gradient(theta, mu)` — $\nabla_\theta J$ in closed form: $\partial J/\partial\theta_j = \pi_j(\mu_j - J)$;
* `score_function_estimate(theta, mu, n_samples, rng, baseline=0.0)` — draw `n_samples` actions with `rng.choice(K, size=n_samples, p=pi)`, form the per-sample estimates $(\mu_a - b)\,\nabla_\theta\log\pi_\theta(a)$ with $\nabla_\theta\log\pi_\theta(a) = \mathbf e_a - \pi$, and return the pair `(mean_estimate, total_variance)` where `total_variance` is the sum over the $K$ coordinates of the per-sample variances.

Use $\theta = (0.5, -0.5, 0, 1)$, $\mu = (1, 3, 2, 0.5)$, 50,000 samples, and compare baselines $b = 0$ and $b = J(\theta)$.
""",
    starter=r"""
def softmax(z):
    # YOUR CODE HERE
    return None

def exact_policy_gradient(theta, mu):
    # YOUR CODE HERE
    return None

def score_function_estimate(theta, mu, n_samples, rng, baseline=0.0):
    # YOUR CODE HERE
    return None, None
""",
    solution=SOL_9,
    check=rf"""
theta = np.array([0.5, -0.5, 0.0, 1.0]); mu = np.array([1.0, 3.0, 2.0, 0.5])
try:
    g = np.asarray(exact_policy_gradient(theta, mu), dtype=float)
    # central finite differences of J as an independent reference
    def J_of(th):
        e = np.exp(th - th.max()); return float(e / e.sum() @ mu)
    fd = np.array([(J_of(theta + 1e-6 * np.eye(4)[i]) - J_of(theta - 1e-6 * np.eye(4)[i])) / 2e-6 for i in range(4)])
except Exception:
    g, fd = None, None
check_close('exact gradient matches finite differences', g, fd, tol=1e-6)
check_close('the gradient coordinates sum to zero (softmax is shift-invariant)', (float(g.sum()) if g is not None else None), 0.0, tol=1e-9)
try:
    m0, v0 = score_function_estimate(theta, mu, 50_000, np.random.default_rng(1), 0.0)
    m0 = np.asarray(m0, dtype=float); v0 = float(v0)
except Exception:
    m0, v0 = None, None
check('with b = 0 the estimate is within 0.02 of the exact gradient in every coordinate', m0 is not None and g is not None and bool(np.abs(m0 - g).max() < 0.02))
try:
    J = J_of(theta)
    mJ, vJ = score_function_estimate(theta, mu, 50_000, np.random.default_rng(1), J)
    mJ = np.asarray(mJ, dtype=float); vJ = float(vJ)
except Exception:
    mJ, vJ = None, None
check('with b = J the estimate is still within 0.02 of the exact gradient (unbiased)', mJ is not None and g is not None and bool(np.abs(mJ - g).max() < 0.02))
check('the baseline b = J cuts the total variance to between 20% and 30% of b = 0', (v0 is not None and vJ is not None) and 0.2 < vJ / v0 < 0.3)
check_hash('hidden: total variance with b = 0 (seed 1), 1 decimal', (round(v0, 1) if v0 is not None else None), '{sha_of(round(ns9["v0"], 1), decimals=1)}', decimals=1)
""",
    explanation_md=r"""
For a softmax policy $\nabla_\theta \log\pi(a) = \mathbf e_a - \pi$, so one sample's estimate is $(\mu_a - b)(\mathbf e_a - \pi)$.
Its expectation is $\sum_a \pi_a(\mu_a - b)(\mathbf e_a - \pi) = \sum_a \pi_a\mu_a\mathbf e_a - J\pi - b(\pi - \pi) = \pi\odot\mu - J\pi$,
which is the exact gradient $\pi\odot(\mu - J)$ *for every* $b$ — the $b$ terms cancel because
$\sum_a \pi_a(\mathbf e_a - \pi) = 0$. The variance does depend on $b$: subtracting the expected reward removes the
component of the signal that carries no information about *which* action was better, here cutting the total
variance to a quarter. The finite-difference check is a useful habit: it validates the closed-form gradient
independently of the sampling estimator, so a failure in one check can be attributed to the right function.
""",
)

# ---------------------------------------------------------------- 10. conceptual: discounting (core)
ex.add_conceptual(
    title="Why discount, and what changes when you do", difficulty=1,
    prompt_md=r"""
A colleague trains an agent to manage a hospital's bed allocation with $\gamma = 0.9$ because "the textbook uses 0.9",
on a time step of one hour. Patients admitted today typically occupy beds for three to five days. Explain (a) the two
mathematical roles the discount factor plays, (b) what effective horizon $\gamma = 0.9$ implies on an hourly step and why
that is a problem here, (c) what value of $\gamma$ would match a horizon of about four days, and (d) one computational
consequence of that choice for value iteration and for the variance of Monte Carlo returns.
""",
    answer_md=r"""
(a) Discounting makes the infinite sum of rewards converge ($|G_t| \le R_{\max}/(1-\gamma)$) and encodes a preference
for sooner rewards: a reward $k$ steps away is worth $\gamma^k$ of the same reward now. Equivalently $\gamma$ is a
per-step continuation probability, so the agent plans as if the episode ended after a geometric number of steps with
mean $1/(1-\gamma)$. (b) With $\gamma = 0.9$ the effective horizon is $1/(1-0.9) = 10$ steps — ten *hours*. A bed freed
or blocked four days later (96 steps) is weighted by $0.9^{96} \approx 4 \times 10^{-5}$, so the agent is effectively
blind to the consequences that matter for this problem; it would, for example, admit freely now and never "see" the
congestion it causes on day three. (c) A four-day horizon is about 96 hourly steps, so $1/(1-\gamma) \approx 96$ gives
$\gamma \approx 0.99$ (half-life $\ln 2/\ln(1/0.99) \approx 69$ hours). (d) Value iteration needs about
$\ln(1/\epsilon)/\ln(1/\gamma)$ sweeps, which grows roughly like $1/(1-\gamma)$: ten times more sweeps at $0.99$ than at
$0.9$, and the stopping-rule error bound $\epsilon\gamma/(1-\gamma)$ loosens by the same factor. Monte Carlo returns
now sum about a hundred noisy rewards instead of ten, so their variance — and the variance of REINFORCE
gradients — grows accordingly, which is why long-horizon problems push practitioners toward TD-style
bootstrapping and learned baselines.
""",
)

# ---------------------------------------------------------------- 11. conceptual: on- vs off-policy (core)
ex.add_conceptual(
    title="On-policy versus off-policy: diagnose the argument", difficulty=2,
    prompt_md=r"""
A learner writes: "Q-learning and SARSA both act ε-greedily and both update from the transitions the agent actually
experiences, so they are both on-policy methods and must learn the same $Q$ table in the limit. The cliff-walking
experiment where they learn different paths must be a bug." Explain what is wrong with this argument, state precisely
what each algorithm converges to, and explain why Q-learning's training-time return can be lower than SARSA's even
though its final greedy policy is better. Finally, name one situation in practice where the off-policy property is
indispensable.
""",
    answer_md=r"""
On- versus off-policy is about the **target**, not about where the data come from. SARSA's target
$r + \gamma Q(s', a')$ uses the action the ε-greedy *behaviour* policy actually takes next, so its fixed point is
$Q^{\pi_\varepsilon}$, the action-value function of the exploratory policy itself (and it reaches $Q^*$ only if
$\varepsilon \to 0$, "greedy in the limit"). Q-learning's target $r + \gamma\max_{a'}Q(s', a')$ evaluates the *greedy*
continuation regardless of what the agent then does, so its fixed point is $Q^*$ for any behaviour policy that keeps
visiting every state–action pair — the behaviour and target policies differ, which is what "off-policy" means. The two
tables therefore differ wherever exploration is costly: next to the cliff, $Q^{\pi_\varepsilon}$ is low (a slip falls
off) while $Q^*$ is high (the greedy agent never slips), so SARSA's greedy path keeps a safe distance and
Q-learning's hugs the edge. During training both agents still explore with probability $\varepsilon$; the one whose
policy walks along the cliff falls off regularly and collects less reward, even though the policy it is converging
to is the optimal one once exploration stops. The different paths are the expected behaviour, not a bug.
Off-policy learning is indispensable whenever the data were generated by a policy you cannot or will not re-run —
experience replay in DQN (old transitions from older policies), learning from human demonstrations, and offline RL
from hospital, logistics or recommender logs where trying a new policy on live users or patients is not an option.
""",
)

# ---------------------------------------------------------------- 12. mini-project (★★★)
PROJECT_CODE = r"""
def softmax(z):
    z = z - np.max(z); e = np.exp(z); return e / e.sum()

def value_iteration_env(env, tol=1e-10):
    V = np.zeros(env.n_states)
    for n in range(1, 100_000):
        Q = env.R + env.gamma * np.einsum("sat,t->sa", env.P, V); V_new = Q.max(axis=1)
        if np.abs(V_new - V).max() < tol:
            V = V_new; break
        V = V_new
    Q = env.R + env.gamma * np.einsum("sat,t->sa", env.P, V)
    return V, Q.argmax(axis=1), n

def exact_value_of(env, policy):
    # exact V^pi of a deterministic policy by the linear solve (the honest yardstick)
    S = env.n_states; idx = np.arange(S)
    return np.linalg.solve(np.eye(S) - env.gamma * env.P[idx, policy], env.R[idx, policy])

def q_learning(env, n_episodes, alpha, eps, rng, max_steps=200):
    Q = np.zeros((env.n_states, env.n_actions)); rets = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s = env.start; total = 0.0
        for _ in range(max_steps):
            a = int(rng.integers(env.n_actions)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = env.step(s, a, rng); total += r
            Q[s, a] += alpha * (r + (0.0 if done else env.gamma * Q[s2].max()) - Q[s, a]); s = s2
            if done: break
        rets[ep] = total
    return Q, rets

def reinforce(env, n_episodes, lr, rng, horizon=25, batch=4, baseline=True):
    # tabular softmax policy; gradient accumulated over `batch` episodes, learned V(s) baseline;
    # common gamma^t-free variant: step t's score is weighted by G_t - V(s_t) (the exact gradient would multiply by gamma**t)
    theta = np.zeros((env.n_states, env.n_actions)); V_b = np.zeros(env.n_states); rets = np.zeros(n_episodes)
    grad = np.zeros_like(theta)
    for ep in range(n_episodes):
        s = env.start; traj = []
        for _ in range(horizon):
            p = softmax(theta[s]); a = int(rng.choice(env.n_actions, p=p))
            s2, r, done = env.step(s, a, rng); traj.append((s, a, r, p)); s = s2
            if done: break
        G = 0.0; returns = []
        for (_, _, r, _) in reversed(traj):
            G = r + env.gamma * G; returns.append(G)
        returns = returns[::-1]
        for (s, a, r, p), G in zip(traj, returns):
            adv = G - (V_b[s] if baseline else 0.0)
            grad[s] += adv * (np.eye(env.n_actions)[a] - p)
        if (ep + 1) % batch == 0:
            theta += lr * grad / batch; grad = np.zeros_like(theta)
        for (s, a, r, p), G in zip(traj, returns):
            V_b[s] += 0.1 * (G - V_b[s])
        rets[ep] = sum(r for (_, _, r, _) in traj)
    return theta, rets

def smooth(x, w=20): return np.convolve(x, np.ones(w) / w, mode="valid")

# 1. the exact answer
V_star, pi_star, n_sweeps = value_iteration_env(grid)
print(f"value iteration: {n_sweeps} sweeps, V*(start) = {V_star[grid.start]:.3f}  (undiscounted return of the optimal path: 4)")
print(np.array(list(grid.policy_string(pi_star))).reshape(4, 4))

# 2. Q-learning and REINFORCE over several seeds, judged by the EXACT value of their greedy policies
n_seeds = 5
q_curves, r_curves, q_vals, r_vals = [], [], [], []
for seed in range(n_seeds):
    Q, rets = q_learning(grid, 400, alpha=0.2, eps=0.1, rng=np.random.default_rng(seed))
    q_curves.append(rets); q_vals.append(exact_value_of(grid, Q.argmax(axis=1))[grid.start])
    theta, rets = reinforce(grid, 800, lr=0.1, rng=np.random.default_rng(seed))
    r_curves.append(rets); r_vals.append(exact_value_of(grid, theta.argmax(axis=1))[grid.start])
print(f"\nexact value of the learned greedy policy at the start (optimum {V_star[grid.start]:.3f}):")
print(f"  Q-learning (400 episodes): {np.round(q_vals, 3)}  mean {np.mean(q_vals):.3f}")
print(f"  REINFORCE  (800 episodes): {np.round(r_vals, 3)}  mean {np.mean(r_vals):.3f}")
print(f"  training return, last 100 episodes: Q-learning {np.mean([c[-100:] for c in q_curves]):.2f}, REINFORCE {np.mean([c[-100:] for c in r_curves]):.2f}")

# 2b. what makes REINFORCE work here? two ablations on the same five seeds
abl_a = [exact_value_of(grid, reinforce(grid, 400, lr=0.05, rng=np.random.default_rng(seed), horizon=40, batch=1)[0].argmax(axis=1))[grid.start]
         for seed in range(n_seeds)]
abl_b = [exact_value_of(grid, reinforce(grid, 800, lr=0.1, rng=np.random.default_rng(seed), baseline=False)[0].argmax(axis=1))[grid.start]
         for seed in range(n_seeds)]
print(f"  REINFORCE ablation, one update per episode, lr 0.05, horizon 40: {np.round(abl_a, 3)}")
print(f"  REINFORCE ablation, batches of 4, horizon 25, NO baseline:       {np.round(abl_b, 3)}")
print(f"  (a policy that never reaches the goal is worth -1/(1 - gamma) = {-1 / (1 - grid.gamma):.0f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(smooth(np.mean(q_curves, axis=0)), label="Q-learning (eps = 0.1, alpha = 0.2)")
ax1.plot(smooth(np.mean(r_curves, axis=0)), label="REINFORCE (lr = 0.1, batches of 4, V(s) baseline)")
ax1.axhline(4.0, color="k", ls="--", label="undiscounted return of the optimal path")
ax1.set_xlabel("episode"); ax1.set_ylabel("undiscounted training return (20-episode average, 5 seeds)")
ax1.set_title("Learning curves on the 4x4 gridworld"); ax1.legend(fontsize=8)

# 3. sensitivity: gamma (exact) and epsilon (Q-learning with a short budget, exact value of the greedy policy)
print("\nsensitivity to gamma (value iteration):")
for g in [0.5, 0.8, 0.9, 0.95, 0.99]:
    env_g = GridWorld(4, 4, start=(3, 0), goal=(0, 3), walls=[(1, 1), (2, 2)], pits=[(0, 2)], slip=0.1, gamma=g)
    V_g, pi_g, n_g = value_iteration_env(env_g)
    print(f"  gamma = {g:4.2f}: V*(start) = {V_g[grid.start]:7.3f}, {n_g:3d} sweeps, policy {env_g.policy_string(pi_g)}")
epsilons = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]
print("\nsensitivity to epsilon (Q-learning, 60 episodes, 10 seeds):")
mean_train, mean_exact, min_exact = [], [], []
for eps in epsilons:
    tr, ex_ = [], []
    for seed in range(10):
        Q, rets = q_learning(grid, 60, alpha=0.2, eps=eps, rng=np.random.default_rng(seed))
        tr.append(rets[-30:].mean()); ex_.append(exact_value_of(grid, Q.argmax(axis=1))[grid.start])
    mean_train.append(np.mean(tr)); mean_exact.append(np.mean(ex_)); min_exact.append(np.min(ex_))
    long_ = [exact_value_of(grid, q_learning(grid, 400, alpha=0.2, eps=eps, rng=np.random.default_rng(seed))[0].argmax(axis=1))[grid.start]
             for seed in range(5)]
    print(f"  eps = {eps:4.2f}: training return (last 30) {np.mean(tr):6.2f}   exact value of greedy policy: mean {np.mean(ex_):6.3f}, "
          f"worst seed {np.min(ex_):6.3f}, seeds within 0.05 of optimum {sum(v > V_star[grid.start] - 0.05 for v in ex_)}/10"
          f"   | with 400 episodes (5 seeds): worst seed {np.min(long_):6.3f}")
ax2.plot(epsilons, mean_train, "o-", label="training return, last 30 of 60 episodes")
ax2.plot(epsilons, mean_exact, "s-", label="exact value of the greedy policy (mean of 10 seeds)")
ax2.plot(epsilons, min_exact, "s:", label="exact value of the greedy policy (worst seed)")
ax2.axhline(V_star[grid.start], color="k", ls="--", label="V*(start)")
ax2.set_xlabel("epsilon"); ax2.set_ylabel("value"); ax2.set_title("Q-learning with a 60-episode budget: exploration vs training return"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""
ex.add_project(
    title="Mini-project: value iteration vs Q-learning vs REINFORCE on one gridworld",
    prompt_md=r"""
Solve the setup's `grid` three ways and compare them honestly.

1. **Dynamic programming.** Run value iteration (your Exercise 5/6 code) to get $V^*$, the optimal policy and the
   number of sweeps. This is the oracle.
2. **Q-learning.** Train a tabular ε-greedy Q-learning agent ($\alpha = 0.2$, $\varepsilon = 0.1$, 400 episodes) with
   **5 different seeds**. For each seed record the training returns and, at the end, the **exact** value of the learned
   greedy policy from the start state, computed with the linear solve of Exercise 4 (not with sampled returns).
3. **REINFORCE.** Train a tabular softmax policy with REINFORCE and a learned state-value baseline (the common
   $\gamma^t$-free form of the update is fine, but say which form you use; learning rate 0.1, gradients accumulated over
   batches of 4 episodes, 800 episodes, episodes cut at 25 steps) with the same 5
   seeds; record the same two quantities. If some seeds fail, do not hide them — report them and try to explain.
4. **Report.** Plot the smoothed learning curves of both learners (mean over seeds) against the undiscounted return
   of the optimal path, and print the per-seed exact values of the two learned policies next to $V^*(\text{start})$.
5. **Sensitivity.** (a) Re-run value iteration for $\gamma \in \{0.5, 0.8, 0.9, 0.95, 0.99\}$: how do $V^*(\text{start})$,
   the sweep count and the policy string change? (b) Give Q-learning a *small* budget of 60 episodes and re-run it
   for $\varepsilon \in \{0, 0.01, 0.05, 0.1, 0.3, 0.5\}$ with 10 seeds; plot the training return of the last 30 episodes
   and the exact value of the final greedy policy (mean and worst seed) against $\varepsilon$ — they move in opposite
   directions; explain why, and explain why the effect fades when the budget is 400 episodes.

Use `rng = np.random.default_rng(seed)` for seed in `range(5)` (or `range(10)` for the $\varepsilon$ sweep), and keep the
whole thing under a minute.
""",
    rubric_md=r"""
* Value iteration is implemented on the model, and the number of sweeps is reported together with $V^*$.
* Q-learning and REINFORCE are run with several seeds, and the **exact** value of each learned greedy policy is computed
  with the linear solve — not estimated from noisy training returns.
* The learning-curve plot has a title, axis labels, a legend and a reference line, and it averages over seeds (or shows
  every seed) rather than showing a single lucky run.
* The $\gamma$ sweep reports the sweep counts, notices that they grow far more slowly than the worst-case $1/(1-\gamma)$
  scaling (and explains why, for an episodic task), shows that $V^*(\text{start})$ changes sign, and comments on whether
  the optimal *policy* changes.
* The $\varepsilon$ sweep separates "return while learning" from "quality of the learned policy" and explains the trade-off
  and why it depends on the training budget.
* A short written comparison: which method needed the model, which needed the most episodes, which was most
  seed-sensitive, and which hyper-parameters mattered for REINFORCE.
""",
    solution_sketch_md=r"""
Value iteration converges in 36 sweeps to $V^*(\text{start}) \approx 2.16$ and the policy that walks right along the
bottom row and up the right-hand column (undiscounted return 4). Q-learning with $\varepsilon = 0.1$ reaches greedy
policies within $0.003$ of the optimum on all five seeds after 400 episodes, while its *training* return is only
about 3 because one move in ten is random. REINFORCE is far more delicate. With one update per episode, a learning
rate of 0.05 and episodes of up to 40 steps, two of the five seeds collapse onto policies that rarely reach the goal
(exact values about $-12$ and $-10$; never reaching it would be worth $-20$) and two more stall around $1.3$–$1.5$ —
failures that a mean learning curve hides and the per-seed exact values expose. Averaging the gradient over four
episodes and cutting episodes at 25 steps, with the state-value baseline, makes all five seeds converge (exact values
$2.159$–$2.162$); keeping the batching but dropping the baseline leaves two seeds short (about $1.02$ and $2.05$). In
the $\gamma$ sweep the optimal *policy* does not change at all on this grid, but $V^*(\text{start})$ goes from $-1.74$ at
$\gamma = 0.5$ (the $+10$, weighted by $0.5^5$, cannot pay for the five $-1$ steps before it) to $+3.61$ at
$\gamma = 0.99$. The sweep count rises only from 23 to 37 — far below the worst-case
$\ln(\lVert V^*\rVert_\infty/\epsilon)/\ln(1/\gamma)$, which exceeds 2,000 at $\gamma = 0.99$ — because the task is episodic:
errors are carried forward only along paths that have not yet ended, so the effective factor per sweep is $\gamma$
times the spectral radius of the transition matrix among non-terminal states (about $0.35$ here). With only 60
episodes of Q-learning the exploration trade-off is plain: the greedy agent ($\varepsilon = 0$) earns the most during
training but only half the seeds end within $0.05$ of the optimum (worst seed about $1.15$, and at $\varepsilon = 0.05$
one seed is as low as $0.38$), because state–action pairs off the first lucky path are rarely tried; $\varepsilon = 0.3$
earns the least while training and delivers the optimal policy on every seed. With 400 episodes the worst of five seeds
is within $0.003$ of the optimum for every $\varepsilon$ in the sweep, $0$ included — the slip in the dynamics supplies
enough exploration on its own, given time — and only the training return still depends on $\varepsilon$. Value iteration alone
needed the model; REINFORCE needed the most episodes and tuning (batch size, horizon, baseline); Q-learning was the
least seed-sensitive learner here, and being off-policy it could afford heavy exploration.
""",
    solution_code=PROJECT_CODE,
)

ex.write(MOD / "exercises.ipynb", MOD / "solutions.ipynb")
kinds = [it["kind"] for it in ex.items]
print(f"exercises: {len(ex.items)} items — {kinds.count('code')} coding, {kinds.count('concept')} conceptual, {kinds.count('project')} project")
print("difficulties:", [it["difficulty"] for it in ex.items])
