"""
nbbuild.py — tiny helper library for *authoring* the course notebooks.

The notebooks in this repository are the canonical learning material; this
module exists so that notebooks (and the paired exercises / solutions
notebooks with their self-checks) can be generated from plain Python scripts
instead of being hand-edited as JSON.

Usage sketch
------------
    import sys; sys.path.insert(0, "tools")
    from nbbuild import Notebook, ExerciseSet, sha_of

    nb = Notebook("Module 01 — Vectors and Geometry")
    nb.md("## 1. Vectors as arrows and as data")
    nb.code("import numpy as np\\nv = np.array([3., 4.])\\nprint(np.linalg.norm(v))")
    nb.save("part-1-linear-algebra/01-vectors-and-geometry/lesson.ipynb")

    ex = ExerciseSet("Module 01 — Exercises", intro_md="...", setup_code="import numpy as np")
    ex.add(title="Cosine similarity", difficulty=1,
           prompt_md="Implement `cosine(u, v)` ...",
           starter="def cosine(u, v):\\n    # YOUR CODE HERE\\n    return None\\n",
           solution="def cosine(u, v):\\n    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))\\n",
           check="check_close('cosine of (1,0),(1,1)', cosine(np.array([1.,0.]), np.array([1.,1.])), 1/np.sqrt(2))",
           explanation_md="The dot product divided by the two norms is the cosine of the angle ...")
    ex.add_conceptual(title="Why normalise?", prompt_md="...", answer_md="...")
    ex.add_project(title="Mini project", prompt_md="...", rubric_md="...", solution_sketch_md="...")
    ex.write("…/exercises.ipynb", "…/solutions.ipynb")

Self-check helpers available inside every exercises/solutions notebook
---------------------------------------------------------------------
    check(name, condition, hint="")                 -> prints ✅ / ❌, never raises
    check_close(name, got, expected, tol=1e-6)      -> numeric comparison with tolerance
    check_hash(name, value, expected_sha, decimals=3) -> compares a *hidden* answer
    summary()                                        -> prints "Score: k/n checks passed"

`sha_of(value, decimals=3)` (exported here) produces the hash that
`check_hash` expects; call it at build time on the *correct* value so the
expected answer never appears in clear text in the exercises notebook.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# ---------------------------------------------------------------------------
# Self-check helper code. This exact source is injected as the first code
# cell of every exercises/solutions notebook AND exec'd here so that build
# scripts use the identical canonicalisation when computing hashes.
# ---------------------------------------------------------------------------
HELPER_SOURCE = '''
import hashlib as _hashlib
import numpy as _np

_results = {}

def _canon(value, decimals=3):
    """Canonical string form of an answer, so hashes are stable."""
    if isinstance(value, (bool, _np.bool_)):
        return "true" if value else "false"
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    if isinstance(value, (int, _np.integer)):
        return str(int(value))
    if isinstance(value, (float, _np.floating)):
        s = f"{float(value):.{decimals}f}"
        return "0." + "0" * decimals if s in ("-0." + "0" * decimals, "-0") else s
    if isinstance(value, dict):
        return "{" + ",".join(f"{_canon(k, decimals)}:{_canon(v, decimals)}" for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))) + "}"
    try:
        arr = _np.asarray(value)
        if arr.ndim == 0:
            return _canon(arr.item(), decimals)
        return "[" + ",".join(_canon(x, decimals) for x in arr.ravel().tolist()) + "]"
    except Exception:
        return str(value)

def sha_of(value, decimals=3):
    return _hashlib.sha256(_canon(value, decimals).encode()).hexdigest()[:16]

def check(name, condition, hint=""):
    """Record and print a pass/fail mark. Never raises, so the notebook keeps running."""
    try:
        ok = bool(condition)
    except Exception:
        ok = False
    _results[name] = ok
    print(("✅ " if ok else "❌ ") + name + ("" if ok or not hint else f"   — hint: {hint}"))
    return ok

def check_close(name, got, expected, tol=1e-6, hint=""):
    try:
        g = _np.asarray(got, dtype=float)
        e = _np.asarray(expected, dtype=float)
        ok = g.shape == e.shape and bool(_np.allclose(g, e, atol=tol, rtol=tol))
    except Exception:
        ok = False
    return check(name, ok, hint)

def check_hash(name, value, expected_sha, decimals=3, hint=""):
    """Compare an answer against a hidden expected value (only its hash is stored)."""
    try:
        ok = value is not None and sha_of(value, decimals) == expected_sha
    except Exception:
        ok = False
    return check(name, ok, hint)

def summary():
    n_ok = sum(1 for v in _results.values() if v)
    n = len(_results)
    print(f"\\nScore: {n_ok}/{n} checks passed" + ("  🎉" if n and n_ok == n else ""))
    if n and n_ok < n:
        print("Not yet passing: " + ", ".join(k for k, v in _results.items() if not v))
'''

# make sha_of / _canon available to build scripts (identical semantics)
_ns: dict = {}
exec(HELPER_SOURCE, _ns)
sha_of = _ns["sha_of"]
_canon = _ns["_canon"]

STARS = {1: "★☆☆", 2: "★★☆", 3: "★★★"}


def _dedent(s: str) -> str:
    return textwrap.dedent(s).strip("\n")


class Notebook:
    """Minimal builder for a Jupyter notebook."""

    def __init__(self, title: str | None = None):
        self.cells: list = []
        if title:
            self.md(f"# {title}")

    def md(self, text: str) -> "Notebook":
        self.cells.append(new_markdown_cell(_dedent(text)))
        return self

    def code(self, src: str) -> "Notebook":
        self.cells.append(new_code_cell(_dedent(src)))
        return self

    def to_nbformat(self):
        nb = new_notebook(cells=self.cells)
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
        return nb

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(self.to_nbformat(), str(path))
        return path


class ExerciseSet:
    """Builds a paired exercises.ipynb / solutions.ipynb with self-checks."""

    def __init__(self, title: str, intro_md: str = "", setup_code: str = "import numpy as np\nimport matplotlib.pyplot as plt"):
        self.title = title
        self.intro_md = intro_md
        self.setup_code = setup_code
        self.items: list[dict] = []

    # -- coding exercise with automatic check --------------------------------
    def add(self, title: str, prompt_md: str, starter: str, solution: str, check: str,
            difficulty: int = 1, explanation_md: str = ""):
        self.items.append(dict(kind="code", title=title, prompt_md=prompt_md, starter=starter,
                               solution=solution, check=check, difficulty=difficulty,
                               explanation_md=explanation_md))
        return self

    # -- conceptual question, answer hidden behind <details> -----------------
    def add_conceptual(self, title: str, prompt_md: str, answer_md: str, difficulty: int = 1):
        self.items.append(dict(kind="concept", title=title, prompt_md=prompt_md,
                               answer_md=answer_md, difficulty=difficulty))
        return self

    # -- open-ended mini project with rubric ---------------------------------
    def add_project(self, title: str, prompt_md: str, rubric_md: str, solution_sketch_md: str = "",
                    solution_code: str = "", difficulty: int = 3, starter: str = ""):
        self.items.append(dict(kind="project", title=title, prompt_md=prompt_md, rubric_md=rubric_md,
                               solution_sketch_md=solution_sketch_md, solution_code=solution_code,
                               difficulty=difficulty, starter=starter))
        return self

    # -------------------------------------------------------------------------
    def _build(self, solutions: bool) -> Notebook:
        nb = Notebook(self.title + (" — Solutions" if solutions else ""))
        how_to = (
            "**How these exercises work.** Each coding exercise has a *starter* cell "
            "(replace `# YOUR CODE HERE` with your implementation) followed by a *check* cell. "
            "Run the check cell to get an instant ✅/❌. Checks never stop the notebook, so you can "
            "run everything top-to-bottom at any time and see your score at the end. "
            "Some checks compare against a *hidden* answer (only a hash is stored), so you cannot "
            "peek at the number — think first, then verify. Conceptual questions hide their answer "
            "behind a *Show answer* toggle. Mini-projects are open-ended and come with a rubric. "
            "Difficulty is marked ★☆☆ (warm-up), ★★☆ (core), ★★★ (stretch)."
        )
        if solutions:
            how_to = ("This notebook contains reference solutions and explanations for every exercise. "
                      "Try the exercises yourself first — the learning happens in the struggle.")
        nb.md((self.intro_md.strip() + "\n\n" if self.intro_md else "") + how_to)
        nb.md("### Setup and self-check helpers\nRun the next two cells first.")
        nb.code(self.setup_code)
        nb.code(HELPER_SOURCE.strip("\n"))

        for i, it in enumerate(self.items, 1):
            stars = STARS.get(it.get("difficulty", 1), "★☆☆")
            head = f"## Exercise {i} — {it['title']}  `{stars}`\n\n{_dedent(it['prompt_md'])}"
            if it["kind"] == "code":
                nb.md(head)
                nb.code(it["solution"] if solutions else it["starter"])
                nb.md("**Check your answer:**")
                nb.code(it["check"])
                if solutions and it.get("explanation_md"):
                    nb.md("**Why this works**\n\n" + _dedent(it["explanation_md"]))
            elif it["kind"] == "concept":
                if solutions:
                    nb.md(head + "\n\n**Answer**\n\n" + _dedent(it["answer_md"]))
                else:
                    nb.md(head + "\n\n<details><summary><b>Show answer</b></summary>\n\n"
                          + _dedent(it["answer_md"]) + "\n\n</details>")
            elif it["kind"] == "project":
                nb.md(head + "\n\n**Rubric — you have mastered this when you can:**\n\n" + _dedent(it["rubric_md"]))
                if solutions:
                    if it.get("solution_sketch_md"):
                        nb.md("**Solution sketch**\n\n" + _dedent(it["solution_sketch_md"]))
                    if it.get("solution_code"):
                        nb.code(it["solution_code"])
                else:
                    nb.code(it.get("starter") or "# Your mini-project workspace\n")
        nb.md("## Score")
        nb.code("summary()")
        return nb

    def write(self, exercises_path: str | Path, solutions_path: str | Path):
        self._build(solutions=False).save(exercises_path)
        self._build(solutions=True).save(solutions_path)
        return Path(exercises_path), Path(solutions_path)
