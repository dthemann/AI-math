#!/usr/bin/env python3
"""
md2pdf.py — render the Markdown guides as dark-themed PDFs (black pages, white text).

    python tools/md2pdf.py                                   # every guide and doc -> build/pdf/
    python tools/md2pdf.py part-1-linear-algebra/01-vectors-and-geometry/README.md docs/glossary.md
    python tools/md2pdf.py --out ~/pdfs --chrome /path/to/chrome
    python tools/md2pdf.py --mathml                          # offline: MathML instead of KaTeX

Pipeline: pandoc turns the GitHub-flavoured Markdown (with $-maths) into HTML, and a
headless Chromium prints that HTML to PDF. The look comes from tools/pdf/dark.css — the
same stylesheet the VS Code "Markdown PDF" extension uses (see .vscode/settings.json), so
both routes give the same result: black pages, white text, figures inverted to dark, and
the self-assessment answers printed instead of hidden.

Requirements: pandoc on PATH, and a Chromium/Chrome binary — found automatically from the
Markdown PDF extension's download or from PATH, or given with --chrome / $CHROME. KaTeX is
loaded from a CDN while rendering; use --mathml when there is no network.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "tools/pdf/dark.css"
KATEX_URL = "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/"

# pandoc's standalone template adds a light stylesheet and a narrow column; undo that and
# set the paper. Everything else (colours, figures, answers) is dark.css.
PRINT_CSS = """
@page {
  size: A4; margin: 15mm 12mm 14mm; background: #000;
  @bottom-center { content: counter(page) " / " counter(pages); color: #999; font-size: 9pt; }
}
html { background-color: #000 !important; }
body { max-width: none !important; margin: 0 !important; padding: 0 !important;
       font-size: 10.5pt; line-height: 1.45; }
"""

CHROME_CANDIDATES = [
    str(Path.home() / ".vscode-server/data/User/globalStorage/yzane.markdown-pdf/chrome/*/chrome-linux64/chrome"),
    str(Path.home() / ".vscode/extensions/yzane.markdown-pdf-*/node_modules/puppeteer-core/.local-chromium/*/chrome-linux/chrome"),
    "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/opt/google/chrome/chrome",
]


def find_chrome(explicit: str | None) -> str:
    """The Chromium binary: --chrome, then $CHROME, then known download locations, then PATH."""
    for cand in [explicit, os.environ.get("CHROME")]:
        if cand:
            if Path(cand).is_file():
                return cand
            sys.exit(f"chrome binary not found: {cand}")
    for pattern in CHROME_CANDIDATES:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit("no Chromium/Chrome found — export a guide once with the Markdown PDF extension "
             "(it downloads one), or pass --chrome /path/to/chrome")


def default_inputs() -> list[Path]:
    mods = sorted(p for p in ROOT.glob("part-*/*/README.md"))
    docs = sorted(ROOT.glob("docs/*.md"))
    return [ROOT / "README.md", ROOT / "CONTRIBUTING.md"] + docs + mods


def output_name(md: Path) -> str:
    rel = md.resolve().relative_to(ROOT)
    if rel.name == "README.md" and len(rel.parts) > 1:
        return rel.parent.name + ".pdf"          # 01-vectors-and-geometry.pdf
    if rel.parts[0] == "docs":
        return "docs-" + rel.stem + ".pdf"       # docs-glossary.pdf
    return rel.stem + ".pdf"                      # README.pdf, CONTRIBUTING.pdf


def title_of(text: str, md: Path) -> str:
    m = re.search(r"^#\s+(.+?)\s*$", text, flags=re.M)
    return m.group(1) if m else md.stem


def open_details_blocks(text: str) -> str:
    """Make sure every <details> block is surrounded by blank lines.

    Without them CommonMark treats the whole block as raw HTML, so the maths inside the
    self-assessment answers would be printed as $-source. The guides are written this way
    already; this keeps the PDFs right if an answer ever is not.
    """
    one_liner = re.compile(r"^(\s*)(<details[^>]*><summary>.*?</summary>)\s*(.+?)\s*(</details>)\s*$")
    lines: list[str] = []
    for line in text.split("\n"):
        m = one_liner.match(line)                 # "<details><summary>Answer</summary>text</details>" on one line
        if m:
            indent, opening, body, closing = m.groups()
            lines += [indent + opening, "", indent + body, "", indent + closing]
        else:
            lines.append(line)
    out: list[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("</details>") and out and out[-1].strip():
            out.append("")
        out.append(line)
        if s.startswith("<details") and s.endswith("</summary>") and i + 1 < len(lines) and lines[i + 1].strip():
            out.append("")
    return "\n".join(out)


# Most Linux installs have no colour-emoji font, so the few emoji in the guides would print as boxes.
EMOJI = {"✅": "✓", "❌": "✗", "🎉": ""}


def highlight_flag() -> str:
    """pandoc 3.8 renamed --highlight-style to --syntax-highlighting; use whichever this pandoc has."""
    out = subprocess.run(["pandoc", "--version"], capture_output=True, text=True).stdout
    m = re.match(r"pandoc(?:\.exe)?\s+(\d+)\.(\d+)", out)
    new = m and (int(m.group(1)), int(m.group(2))) >= (3, 8)
    return "--syntax-highlighting" if new else "--highlight-style"


def build_html(md: Path, html: Path, mathml: bool, workdir: Path) -> None:
    text = md.read_text(encoding="utf-8")
    src = workdir / md.name
    src.write_text(open_details_blocks(text).translate(str.maketrans(EMOJI)), encoding="utf-8")
    cmd = ["pandoc", str(src), "-f", "gfm+tex_math_dollars", "-t", "html5", "--standalone",
           "--css", str(CSS), "--metadata", f"pagetitle={title_of(text, md)}",
           highlight_flag(), "breezedark", "-o", str(html)]
    cmd.append("--mathml" if mathml else f"--katex={KATEX_URL}")
    subprocess.run(cmd, check=True)
    page = html.read_text(encoding="utf-8")
    base = md.resolve().parent.as_uri() + "/"              # relative figure paths resolve next to the guide
    page = page.replace("<head>", f'<head>\n<base href="{base}">\n<style>{PRINT_CSS}</style>', 1)
    page = page.replace("<details>", "<details open>")       # belt and braces beside the CSS rule
    html.write_text(page, encoding="utf-8")


def print_pdf(chrome: str, html: Path, pdf: Path) -> None:
    cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
           "--no-pdf-header-footer", "--virtual-time-budget=30000",   # let KaTeX finish before printing
           f"--print-to-pdf={pdf}", html.resolve().as_uri()]
    res = subprocess.run(cmd, capture_output=True, text=True)
    noise = ("dbus", "Failed to connect to the bus", "DevTools listening")
    err = "\n".join(l for l in res.stderr.splitlines() if l and not any(n in l for n in noise))
    if res.returncode != 0 or not pdf.exists():
        sys.exit(f"chromium failed on {html.name}:\n{err}")
    if err:
        print(err, file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render Markdown guides as dark PDFs.")
    ap.add_argument("inputs", nargs="*", help="Markdown files (default: every guide and doc)")
    ap.add_argument("--out", default=str(ROOT / "build/pdf"), help="output directory (default: build/pdf)")
    ap.add_argument("--chrome", help="Chromium/Chrome binary (default: auto-detect, or $CHROME)")
    ap.add_argument("--mathml", action="store_true", help="render maths as MathML (no network needed)")
    ap.add_argument("--keep-html", action="store_true", help="keep the intermediate HTML next to the PDFs")
    ap.add_argument("--jobs", type=int, default=4, help="parallel Chromium processes")
    args = ap.parse_args()

    if not shutil.which("pandoc"):
        sys.exit("pandoc not found on PATH")
    chrome = find_chrome(args.chrome)
    inputs = [Path(p).resolve() for p in args.inputs] or default_inputs()
    missing = [p for p in inputs if not p.is_file()]
    if missing:
        sys.exit("not found: " + ", ".join(map(str, missing)))
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        def render(md: Path) -> tuple[Path, Path]:
            sub = work / output_name(md)[:-4]
            sub.mkdir()
            html = (out if args.keep_html else sub) / (output_name(md)[:-4] + ".html")
            pdf = out / output_name(md)
            build_html(md, html, args.mathml, sub)
            print_pdf(chrome, html, pdf)
            return md, pdf

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            for md, pdf in pool.map(render, inputs):
                print(f"{md.relative_to(ROOT) if md.is_relative_to(ROOT) else md}  ->  "
                      f"{pdf.relative_to(ROOT) if pdf.is_relative_to(ROOT) else pdf}  ({pdf.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
