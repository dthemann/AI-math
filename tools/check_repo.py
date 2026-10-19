#!/usr/bin/env python3
"""
check_repo.py — structural lint for the course repository.

    python tools/check_repo.py          # report problems, exit 1 if any
    python tools/check_repo.py --stats  # also print per-module statistics

Checks: every module has its four files; concept guides contain the required
sections; relative Markdown links resolve; notebooks are valid, carry a kernelspec,
use no forbidden libraries or downloads; exercises.ipynb has stripped outputs and
no leaked solutions; solutions.ipynb has no placeholders.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SECTIONS = [
    "Why this matters", "Learning objectives", "Prerequisites",
    "Common pitfalls", "Real-world case studies", "Where this shows up",
    "Cheat sheet", "Self-assessment", "Further reading", "What's in the notebooks",
]
FORBIDDEN_CODE = [r"fetch_\w+\(", r"!pip", r"^\s*import torch", r"^\s*import tensorflow", r"^\s*import jax",
                  r"urllib\.request", r"requests\.get", r"wget ", r"curl "]
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def module_dirs() -> list[Path]:
    return sorted(p.parent for p in ROOT.rglob("README.md")
                  if p.parent != ROOT and p.parent.name[:2].isdigit() and "_authoring" not in p.parts)


def check_links(md: Path, problems: list[str]):
    text = md.read_text(encoding="utf-8")
    # drop fenced code blocks and inline math before scanning for links
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    text = re.sub(r"\$[^$\n]*\$", "", text)
    for target in LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#")[0]
        if not path:
            continue
        if not (md.parent / path).exists():
            problems.append(f"{md.relative_to(ROOT)}: broken link -> {target}")


def load_nb(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_cells(nb):
    return [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"]


def md_cells(nb):
    return [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"] if c["cell_type"] == "markdown"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    problems: list[str] = []
    stats = []
    for mod in module_dirs():
        rel = mod.relative_to(ROOT)
        files = {n: mod / n for n in ("README.md", "lesson.ipynb", "exercises.ipynb", "solutions.ipynb")}
        for n, p in files.items():
            if not p.exists():
                problems.append(f"{rel}: missing {n}")
        readme = files["README.md"]
        words = 0
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            words = len(re.findall(r"\b\w+\b", text))
            for sec in REQUIRED_SECTIONS:
                if not re.search(r"^##+\s+.*" + re.escape(sec), text, flags=re.M | re.I):
                    problems.append(f"{rel}/README.md: missing section '{sec}'")
            if words < 2500:
                problems.append(f"{rel}/README.md: only {words} words (expected ≥ 2500)")
            check_links(readme, problems)
        n_lesson = n_ex = 0
        for n in ("lesson.ipynb", "exercises.ipynb", "solutions.ipynb"):
            p = files[n]
            if not p.exists():
                continue
            try:
                nb = load_nb(p)
            except Exception as e:
                problems.append(f"{rel}/{n}: invalid notebook ({e})")
                continue
            if nb.get("metadata", {}).get("kernelspec", {}).get("name") != "python3":
                problems.append(f"{rel}/{n}: kernelspec is not python3")
            codes = code_cells(nb)
            for src in codes:
                for pat in FORBIDDEN_CODE:
                    if re.search(pat, src, flags=re.M):
                        problems.append(f"{rel}/{n}: forbidden pattern {pat!r}")
            if n == "lesson.ipynb":
                n_lesson = len(nb["cells"])
                if n_lesson < 20:
                    problems.append(f"{rel}/{n}: only {n_lesson} cells")
            if n == "exercises.ipynb":
                n_ex = sum(1 for m in md_cells(nb) if m.lstrip().startswith("## Exercise"))
                if n_ex < 8:
                    problems.append(f"{rel}/{n}: only {n_ex} exercises")
                if any(c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"):
                    problems.append(f"{rel}/{n}: has outputs (should be stripped)")
                if not any("YOUR CODE HERE" in s for s in codes):
                    problems.append(f"{rel}/{n}: no 'YOUR CODE HERE' placeholders found")
            if n == "solutions.ipynb":
                if any("YOUR CODE HERE" in s for s in codes):
                    problems.append(f"{rel}/{n}: contains 'YOUR CODE HERE'")
                if not any(c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"):
                    problems.append(f"{rel}/{n}: has no outputs (should be executed)")
        stats.append((str(rel), words, n_lesson, n_ex))

    for md in list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md")):
        check_links(md, problems)

    if args.stats:
        print(f"{'module':70s}{'words':>7s}{'cells':>7s}{'exerc.':>7s}")
        for rel, w, c, e in stats:
            print(f"{rel:70s}{w:7d}{c:7d}{e:7d}")
        print(f"{'TOTAL':70s}{sum(s[1] for s in stats):7d}{sum(s[2] for s in stats):7d}{sum(s[3] for s in stats):7d}")
        print()
    if problems:
        print("\n".join(problems))
        print(f"\n{len(problems)} problem(s)")
        return 1
    print("repository structure OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
