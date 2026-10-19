#!/usr/bin/env python3
"""
run_notebooks.py — execute course notebooks end-to-end and verify them.

    python tools/run_notebooks.py                      # every module in the repo
    python tools/run_notebooks.py part-3-probability   # one part
    python tools/run_notebooks.py part-1-linear-algebra/01-vectors-and-geometry   # one module
    python tools/run_notebooks.py --no-write ...       # execute only, do not touch the files

What it does for each module folder
-----------------------------------
* lesson.ipynb     — executed; outputs (plots, prints) are saved into the file so the
                     notebook reads well on GitHub.
* solutions.ipynb  — executed; outputs saved; FAILS if any self-check prints ❌ or the
                     final score is not n/n.
* exercises.ipynb  — executed to prove it runs top-to-bottom with the starter code
                     (self-checks print ❌ but must never raise); outputs are then
                     stripped so learners start from a clean notebook.

Exit status is non-zero if any notebook raised, timed out, or any solution check failed.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NB_NAMES = ("lesson.ipynb", "exercises.ipynb", "solutions.ipynb")


def module_dirs(targets: list[str]) -> list[Path]:
    paths = [ROOT / t for t in targets] if targets else [ROOT]
    found: set[Path] = set()
    for p in paths:
        if (p / "lesson.ipynb").exists():
            found.add(p)
        for nb in p.rglob("lesson.ipynb"):
            if "_authoring" not in nb.parts and ".ipynb_checkpoints" not in nb.parts:
                found.add(nb.parent)
    return sorted(found)


def cell_text(cell) -> str:
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            out.append("".join(o.get("text", "")))
        elif o.get("output_type") in ("execute_result", "display_data"):
            out.append("".join(o.get("data", {}).get("text/plain", "")))
        elif o.get("output_type") == "error":
            out.append("\n".join(o.get("traceback", [])))
    return "\n".join(out)


def execute(path: Path, timeout: int) -> tuple[nbformat.NotebookNode, str | None, float]:
    nb = nbformat.read(str(path), as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}},
                            extra_arguments=["--log-level=40"])  # silence kernel log noise
    t0 = time.time()
    err = None
    try:
        client.execute()
    except CellExecutionError as e:
        err = str(e)
    except Exception as e:  # timeouts, kernel death
        err = f"{type(e).__name__}: {e}"
    return nb, err, time.time() - t0


def strip_outputs(nb):
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
    return nb


def check_solution_outputs(nb) -> list[str]:
    problems = []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        txt = cell_text(c)
        if "❌" in txt:
            problems.extend(l.strip() for l in txt.splitlines() if "❌" in l)
        m = re.search(r"Score: (\d+)/(\d+) checks passed", txt)
        if m and m.group(1) != m.group(2):
            problems.append(f"score {m.group(1)}/{m.group(2)}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="module or part folders (default: whole repo)")
    ap.add_argument("--timeout", type=int, default=600, help="per-cell timeout in seconds")
    ap.add_argument("--no-write", action="store_true", help="execute but leave files untouched")
    ap.add_argument("--only", choices=NB_NAMES, help="run just one notebook type")
    args = ap.parse_args()

    mods = module_dirs(args.targets)
    if not mods:
        print("no modules found for", args.targets)
        return 1

    failures = 0
    for mod in mods:
        rel = mod.relative_to(ROOT)
        for name in NB_NAMES:
            if args.only and name != args.only:
                continue
            path = mod / name
            if not path.exists():
                print(f"MISSING   {rel}/{name}")
                failures += 1
                continue
            nb, err, dt = execute(path, args.timeout)
            status = "ok"
            if err:
                status = "ERROR"
                failures += 1
            elif name == "solutions.ipynb":
                probs = check_solution_outputs(nb)
                if probs:
                    status = "CHECKS FAILED"
                    failures += 1
                    err = "\n".join(probs)
            print(f"{status:14s}{rel}/{name}  ({dt:5.1f}s)")
            if err:
                print(textwrap_indent(err))
            if not args.no_write and not err:
                if name == "exercises.ipynb":
                    strip_outputs(nb)
                nbformat.write(nb, str(path))
    print("\nALL GOOD" if failures == 0 else f"\n{failures} problem(s) found")
    return 0 if failures == 0 else 1


def textwrap_indent(s: str, n: int = 2000) -> str:
    s = s if len(s) <= n else s[:n] + "\n    ... (truncated)"
    return "\n".join("    " + l for l in s.splitlines())


if __name__ == "__main__":
    sys.exit(main())
