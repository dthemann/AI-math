# How to use this course

## The four files of a module

Each module folder has the same four files. Use them in this order:

1. **`README.md` — read.** This is the concept guide. It is long on purpose. Each numbered section follows the same rhythm — *intuition → definition → worked numbers → real-life example → where it appears in ML/AI* — so you always know where you are. Work the numerical examples with pen and paper (or a Python prompt) rather than just nodding along. The guide ends with a cheat sheet and self-assessment questions whose answers are hidden behind *Show answer* toggles.

2. **`lesson.ipynb` — run.** The notebook builds the same ideas in code. Run cells one at a time. When you meet a *Try it yourself* prompt, do it — change a parameter, re-run, predict the output before you look. Everything is implemented from scratch in NumPy before any library function is called, so you can read the implementation of every object you use.

3. **`exercises.ipynb` — do.** Ten to twelve exercises graded ★☆☆ (warm-up), ★★☆ (core) and ★★★ (stretch), followed by a mini-project. Replace every `# YOUR CODE HERE` with your own implementation and run the check cell beneath it.

4. **`solutions.ipynb` — compare.** Only after a serious attempt. Every solution has a short *Why this works* note that also names the most common wrong turn. If your approach differed and still passed the checks, that is fine — compare for style and insight, not conformity.

## How the self-checks work

Each check cell prints a line per test:

```
✅ norm of (3, 4) is 5
❌ hidden: norm of (1, 2, 2)   — hint: remember the square root
```

* Checks **never stop the notebook**. You can run the whole notebook top-to-bottom at any time; the final `summary()` cell prints your score, e.g. `Score: 7/11 checks passed`, and lists what is still failing.
* Some checks compare against a **hidden answer**. Only a hash of the correct value is stored, so you cannot read the expected number off the notebook. Hidden checks round your answer before hashing (usually to 3 decimals), so a result of `0.16667` and `0.1666667` are treated the same — but an answer that is off in the third decimal will fail. If you are convinced your reasoning is right and the check disagrees, check your rounding and units first, then look at the solutions.
* Other checks compare to a **reference you can see** (for example your implementation against `np.linalg.norm`) or test a **property** (`shape == (3, 3)`, a probability between 0.45 and 0.55). These are the same tests the solutions notebook passes.
* Conceptual questions have no automatic check; they hide a model answer behind a *Show answer* toggle. Write your answer down *before* opening it — the act of writing is where the learning happens.
* The mini-project at the end of each exercise set is open-ended and comes with a **rubric**. The solutions notebook contains a sketch and a runnable reference implementation, but there is no single right answer.

## Suggested rhythm

A module takes roughly 6–10 hours; the concept guides are long (typically 7,000–12,000 words) because every idea is worked through with numbers. A workable split:

| Session | Length | What |
|---------|--------|------|
| 1 | 90–150 min | Read the guide's numbered concept sections; work the examples by hand. |
| 2 | 60–120 min | Finish the guide (pitfalls, case studies), then run the lesson notebook and do the *Try it yourself* prompts. |
| 3 | 60–120 min | Exercises ★☆☆ and ★★☆. |
| 4 | 90–150 min | Stretch exercises, the mini-project, then compare with the solutions; re-read the cheat sheet and answer the self-assessment questions from memory. |

Spacing the sessions over several days beats doing them back-to-back: the self-assessment questions at the end of each guide are also good material for spaced repetition a week or a month later.

## Prerequisites and order

The parts build on each other, but the dependencies are explicit: every guide has a **Prerequisites** section listing the earlier modules it relies on. If you already know linear algebra, start at Module 04; if you only need probability and statistics, Modules 07–12 stand on their own with light references back to Modules 01–03. See [study-plans.md](study-plans.md) for several paths through the material.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Python 3.10 or newer. No GPU, no internet connection, no data downloads are needed; all datasets are generated in the notebooks or bundled with scikit-learn.

## When you are stuck

* Re-read the *intuition* paragraph of the relevant section; most confusion is about what the symbols *mean*, not about algebra.
* Make the problem smaller: 2 dimensions instead of n, 3 data points instead of 1,000, one step of the algorithm by hand.
* Print shapes. In the notebooks, nine out of ten bugs are shape bugs.
* Use the hints printed by failing checks.
* Look at the solution for *that one exercise only*, understand it, then re-implement it from a blank cell without looking.
