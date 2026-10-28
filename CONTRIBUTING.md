# Contributing

Corrections, clearer explanations, better real-world examples and new exercises are all welcome.

## Conventions

### Module layout

Every module folder contains these four files, plus an optional `figures/` folder of SVG images used by the guide:

```
README.md         concept guide (the long-form explanation)
lesson.ipynb      narrated, runnable walk-through
exercises.ipynb   8–12 self-checking exercises + one mini-project (outputs stripped)
solutions.ipynb   reference solutions with explanations (executed, outputs kept)
figures/          (optional) SVG figures embedded in README.md
```

### Concept guides (`README.md`)

Required sections, in order: title and hook · *Why this matters for ML and AI* · *Learning objectives* · *Prerequisites* · numbered concept sections · *Common pitfalls and misconceptions* · *Real-world case studies* · *Where this shows up in ML, DL and AI* · *Cheat sheet* · *Self-assessment* · *Further reading* · *What's in the notebooks*.

Each numbered concept section gives, in order: intuition, a precise definition with LaTeX (`$...$`, `$$...$$` — GitHub renders both), a worked example with actual numbers, a real-life example, and a short "In ML/AI this is …" paragraph. Cross-link other modules with relative paths.

### Figures in concept guides

Add a figure where a picture removes a misunderstanding the text alone leaves open, not to every section. Figures are generated, never hand-drawn: a module's figures come from `tools/builders/figures_XX.py`, built on the helpers in `tools/figbuild.py`, and are written as SVG into the module's `figures/` folder.

```bash
python tools/builders/figures_01.py                      # (re)write all figures of Module 01
python tools/builders/figures_01.py projection --preview /tmp/fig   # one figure, plus a PNG preview
```

* Draw the guide's own worked-example numbers, so that picture and text agree exactly.
* Every panel has a title and axis labels; keep the white background so the figure also reads on GitHub's dark theme.
* Embed with a relative Markdown image link whose alt text describes the figure in words (no `$` maths in alt text), followed by a bold caption; `tools/check_repo.py` verifies that the link resolves:

  ```markdown
  ![Vectors a and b, the projection p of a onto b, and the residual r at a right angle to b.](figures/projection.svg)

  **Figure 5.** (a) The projection ... (b) ...
  ```
* Output is reproducible: regenerating unchanged figures produces byte-identical files.

### Self-assessment answers

Answers are hidden behind `<details>`. Leave a blank line after `<details><summary>Answer</summary>` and another before `</details>`; otherwise CommonMark treats the whole block as raw HTML and the maths inside the answer is not rendered — on GitHub or anywhere else.

### Rendering the guides as PDF

The guides also render as dark PDFs — black pages, white text, figures inverted to dark, the self-assessment answers printed:

```bash
python tools/md2pdf.py                                   # every guide and doc -> build/pdf/
python tools/md2pdf.py part-1-linear-algebra/01-vectors-and-geometry/README.md docs/glossary.md
```

This needs `pandoc` and a Chromium/Chrome binary (found automatically from the VS Code "Markdown PDF" extension's download, or given with `--chrome`); maths is rendered with KaTeX fetched from a CDN, and `--mathml` works offline. In VS Code, the "Markdown PDF" extension's right-click export produces the same look: `.vscode/settings.json` points it at the shared stylesheet `tools/pdf/dark.css`. Generated PDFs are ignored by git.

### Notebooks

* Allowed libraries: the standard library, NumPy, SciPy, Matplotlib, pandas, scikit-learn. No deep-learning frameworks — the course builds the maths in NumPy on purpose.
* No downloads: synthetic data (seeded with `np.random.default_rng(...)`) or scikit-learn's **bundled** datasets only (`load_digits`, `load_iris`, `load_wine`, `load_breast_cancer`, `load_diabetes`).
* Every notebook must execute top-to-bottom in under 90 seconds on two CPU cores.
* Every figure has a title and axis labels.

### Exercises and self-checks

The exercises and solutions notebooks share one set of helper functions, injected as their first code cell:

| Helper | Use |
|--------|-----|
| `check(name, condition, hint="")` | Property or range checks. Prints ✅/❌, never raises. |
| `check_close(name, got, expected, tol=1e-6)` | Numeric comparison against a reference the learner may see. |
| `check_hash(name, value, expected_sha, decimals=3)` | Comparison against a **hidden** answer — only a short SHA-256 of the canonicalised, rounded value is stored. |
| `summary()` | Prints the score `k/n`. |

Rules of thumb:

* Starter cells return `None` (or assign `None`) and must never raise, so the exercises notebook runs top-to-bottom before anything is filled in.
* Check cells wrap calls in `try/except` and pass `None` on failure, so a wrong implementation gives ❌ rather than a traceback.
* Use hidden hashes only for deterministic answers (closed-form numbers, seeded computations you have actually run) and round so that floating-point noise cannot flip the hash.

### Generating notebooks

Every module's three notebooks are generated by a script in `tools/builders/` (`build_01.py` … `build_19.py`), using the small helper library `tools/nbbuild.py` (see its docstring). The builders are deterministic: hidden-answer hashes are computed by running the reference solutions, so regenerating a module reproduces the same notebooks. To change a notebook, edit its builder and regenerate:

```bash
python tools/builders/build_07.py                          # writes the module's three notebooks (no outputs yet)
python tools/run_notebooks.py part-3-probability/07-probability-foundations   # executes and verifies them
```

You may also edit a notebook directly in Jupyter for a small fix — but then port the change to the builder, or the next regeneration will overwrite it. In either case run `tools/run_notebooks.py` on the module before opening a pull request. It executes all three notebooks, stores outputs in `lesson.ipynb` and `solutions.ipynb`, strips outputs from `exercises.ipynb`, and fails if any solution self-check prints ❌. `python tools/check_repo.py` lints the repository structure (required guide sections, links, notebook hygiene).

## Pull requests

* One module (or one cross-cutting fix) per pull request.
* State what you changed and why; for mathematical corrections, include the derivation or a reference.
* CI executes every notebook; make sure it is green.
