#!/usr/bin/env python3
"""Catch the build-breaking markdown traps without running Jekyll.

`check_links.py` is the better tool but it needs a built site, and Ruby is not
always where the measurements are. This reads the sources instead and looks for
the four things that break this book specifically:

  1. Liquid tags inside fenced code blocks. `{{` is parsed everywhere, including
     inside ```` ``` ````, so an HLO snippet with `replica_groups={{0,1}}` is a
     build failure whose error names a line in a different file. Wrap in raw.
  2. A `toc` name that will not resolve to a heading id. Kramdown deletes
     apostrophes and slashes when generating ids while Liquid's `slugify` turns
     them into hyphens, so the two disagree and the sidebar link 404s silently.
  3. A prev/next chain that is not symmetric. Nothing infers it.
  4. A `figure.liquid` include pointing at an asset that does not exist.

    python tools/check_source.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = [p for p in sorted(ROOT.glob("pages/*.md")) + [ROOT / "index.md"] if p.exists()]

FENCE = re.compile(r"^(\s*)(```|~~~)")
LIQUID = re.compile(r"\{\{|\{%")
RAW_OPEN = re.compile(r"\{%-?\s*raw\s*-?%\}")
RAW_CLOSE = re.compile(r"\{%-?\s*endraw\s*-?%\}")
HEADING = re.compile(r"^(#{2,6})\s+(.*?)\s*$")
FIGURE = re.compile(r'figure\.liquid[^%]*?path=["\']([^"\']+)["\']')
RELATIVE_URL = re.compile(r"\{\{\s*['\"](/[^'\"]*)['\"]\s*\|\s*relative_url\s*\}\}")
COMMENT_OPEN = re.compile(r"<!--")
COMMENT_CLOSE = re.compile(r"-->")


def liquid_slugify(text: str) -> str:
    """Jekyll's default `slugify`: every non-alphanumeric run becomes a hyphen."""
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", text.lower()))


def kramdown_id(text: str) -> str:
    """Kramdown's heading id: punctuation is deleted, then spaces become hyphens."""
    s = re.sub(r"^[^a-zA-Z]+", "", text)
    s = re.sub(r"[^a-zA-Z0-9 \-]", "", s)
    return re.sub(r"-+", "-", s.replace(" ", "-")).lower()


def front_matter(text: str) -> tuple[dict, int]:
    """Parse just enough YAML for toc names and the nav chain."""
    if not text.startswith("---"):
        return {}, 0
    end = text.find("\n---", 3)
    if end == -1:
        return {}, 0
    block = text[3:end]
    data: dict = {"toc": []}
    in_toc = False
    for line in block.splitlines():
        if line.startswith("toc:"):
            in_toc = True
            continue
        if in_toc:
            if line and not line[0].isspace():
                in_toc = False
            else:
                m = re.match(r"\s*(?:-\s*)?name:\s*(.+?)\s*$", line)
                if m:
                    data["toc"].append(m.group(1).strip().strip("\"'"))
                continue
        m = re.match(r"(\w+):\s*(.*?)\s*$", line)
        if m:
            data[m.group(1)] = m.group(2).strip().strip("\"'")
    return data, text[:end].count("\n") + 2


def scan(path: Path) -> list[str]:
    text = path.read_text()
    lines = text.splitlines()
    rel = path.relative_to(ROOT)
    problems: list[str] = []

    fm, body_start = front_matter(text)

    in_fence = False
    fence_start = 0
    in_raw = False
    in_comment = False
    headings: list[str] = []

    for i, line in enumerate(lines, 1):
        if RAW_OPEN.search(line):
            in_raw = True
        if RAW_CLOSE.search(line):
            in_raw = False

        # HTML comments hold the BLOCKED notes and are not rendered, but Liquid
        # does expand includes inside them, so track them for the figure check.
        if COMMENT_OPEN.search(line) and not COMMENT_CLOSE.search(line):
            in_comment = True
        elif COMMENT_CLOSE.search(line):
            in_comment = False

        m = FENCE.match(line)
        if m:
            if in_fence:
                in_fence = False
            else:
                in_fence = True
                fence_start = i
            continue

        if in_fence and not in_raw and LIQUID.search(line):
            problems.append(
                f"{rel}:{i}: Liquid tag inside a code fence opened at line "
                f"{fence_start}; wrap the block in raw/endraw"
            )

        if not in_fence and i > body_start:
            h = HEADING.match(line)
            if h:
                headings.append(h.group(2))

        for asset in FIGURE.findall(line):
            if not (ROOT / asset).exists():
                where = " (inside an HTML comment, which Liquid still expands)" if in_comment else ""
                problems.append(f"{rel}:{i}: figure path {asset} does not exist{where}")

        # Cross-chapter links are hand-written and a typo in one renders as a
        # working link to a 404.
        for target in RELATIVE_URL.findall(line):
            path = target.strip("/").split("#")[0]
            if not path or "." in Path(path).name:
                continue
            if not (ROOT / f"{path}.md").exists() and not (ROOT / path).exists():
                problems.append(f"{rel}:{i}: relative_url target /{path} is not a page")

    if in_fence:
        problems.append(f"{rel}: unclosed code fence opened at line {fence_start}")

    # toc names must slugify to an id kramdown will actually emit.
    ids = {kramdown_id(h) for h in headings}
    for name in fm.get("toc", []):
        want = liquid_slugify(name)
        if want not in ids:
            near = kramdown_id(name)
            hint = f"; kramdown would emit '{near}'" if near in ids else ""
            problems.append(f"{rel}: toc entry '{name}' -> #{want} matches no heading{hint}")

    return problems


def check_chain() -> list[str]:
    """Verify prev/next symmetry across the book."""
    problems: list[str] = []
    urls: dict[str, Path] = {}
    meta: dict[str, dict] = {}
    for path in PAGES:
        fm, _ = front_matter(path.read_text())
        url = "/" + path.relative_to(ROOT).with_suffix("").as_posix()
        if path.name == "index.md":
            url = "/"
        urls[url] = path
        meta[url] = fm

    for url, fm in meta.items():
        nxt = fm.get("next_section_url", "")
        if not nxt:
            continue
        if nxt not in meta:
            problems.append(f"{urls[url].name}: next_section_url {nxt} is not a page")
        elif meta[nxt].get("previous_section_url", "") != url:
            problems.append(
                f"chain broken: {urls[url].name} -> {nxt}, but that page's "
                f"previous_section_url is '{meta[nxt].get('previous_section_url', '')}'"
            )
    return problems


def main() -> int:
    problems: list[str] = []
    for path in PAGES:
        problems.extend(scan(path))
    problems.extend(check_chain())

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"OK: {len(PAGES)} pages, no Liquid-in-fence, toc anchors resolve, chain symmetric")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
