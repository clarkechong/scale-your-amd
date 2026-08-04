#!/usr/bin/env python3
"""Check the built site for broken internal links, anchors and navigation.

Run `bundle exec jekyll build` first, then `python3 tools/check_links.py`.

Three classes of breakage this catches, none of which makes Jekyll fail:
  1. An internal href pointing at a page that does not exist.
  2. A `toc` entry whose slugified name does not match any heading id, which is
     what happens when a heading contains an apostrophe or a slash.
  3. A prev/next chain that is not symmetric, because nothing infers it.
"""

import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "_site"
BASEURL = "/scale-your-amd"

HREF = re.compile(r'href="([^"]*)"')
ID = re.compile(r'\sid="([^"]+)"')


def built_pages():
    return {p.relative_to(SITE).as_posix(): p.read_text() for p in SITE.rglob("*.html")}


def resolve(href):
    """Map a site path to the file Jekyll built for it.

    Extensionless paths get `.html` appended, which is how both GitHub Pages and
    `jekyll serve` resolve them. Anything that already has an extension, such as a
    stylesheet or a font, is taken literally.
    """
    path = href[len(BASEURL):].lstrip("/")
    if path in ("", "/"):
        return "index.html"
    if "." not in path.rsplit("/", 1)[-1]:
        path += ".html"
    return path


def main():
    pages = built_pages()
    if not pages:
        sys.exit("no built pages found: run `bundle exec jekyll build` first")

    problems = []

    for name, html in sorted(pages.items()):
        ids = set(ID.findall(html))
        for href in HREF.findall(html):
            if href.startswith("#"):
                anchor = href[1:]
                if anchor and anchor not in ids:
                    problems.append(f"{name}: anchor #{anchor} has no matching id")
            elif href.startswith(BASEURL):
                target = resolve(href.split("#")[0].split("?")[0])
                if target and not (SITE / target).exists():
                    problems.append(f"{name}: link {href} -> missing {target}")
            elif href.startswith("/"):
                problems.append(
                    f"{name}: link {href} is missing the baseurl "
                    f"(use relative_url)"
                )

    # The prev/next chain is hand-maintained, so verify it is symmetric.
    nav = {}
    for name, html in pages.items():
        prev = next_ = None
        for cls, href in re.findall(
            r'class="(left-button|right-button) section-button"><a href="([^"]*)"', html
        ):
            if cls == "left-button":
                prev = resolve(href)
            else:
                next_ = resolve(href)
        nav[name] = (prev, next_)

    for name, (_, next_) in sorted(nav.items()):
        if next_ is None:
            continue
        if next_ not in nav:
            problems.append(f"{name}: next -> {next_} which does not exist")
        elif nav[next_][0] != name:
            problems.append(
                f"chain broken: {name} -> {next_}, but {next_} points back to "
                f"{nav[next_][0]}"
            )

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)

    anchors = sum(len(HREF.findall(h)) for h in pages.values())
    print(f"OK: {len(pages)} pages, {anchors} links, chain symmetric")


if __name__ == "__main__":
    main()
