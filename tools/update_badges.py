#!/usr/bin/env python3
"""Sync the hard-coded badge numbers on the homepages with res/scholar.json
and res/stars.json, so the values baked into the HTML (the no-JS fallback)
stay as fresh as the client-side refresh makes them at runtime.

- citation badges: "<a href=...citation_for_view=...:CLUSTER>N+ citations</a>"
  gets N replaced with the live count from scholar.json, floored to 100
- star badges: "<a href=https://github.com/OWNER/REPO><strong>N stars</strong>"
  gets N replaced with the live count from stars.json, formatted like
  res/site.js does ("1.5k")

Only the two homepages carry such badges. Files keep their original line
endings; a file is rewritten only when a number actually changed.
"""
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAGES = ["index.htm", "index_cn.htm"]


def load(name):
    with open(os.path.join(ROOT, "res", name), encoding="utf-8") as f:
        return json.load(f)


def fmt_stars(n):
    if n < 1000:
        return str(n)
    k = "%.1f" % (n / 1000)
    return (k[:-2] if k.endswith(".0") else k) + "k"


CITE_RE = re.compile(r'(citation_for_view=[^"]*:([\w-]+)"[^>]*>)[\d,]+\+ citations')
STAR_RE = re.compile(r'(<a href="https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[^"]*"><strong>)[\d.,]+k? stars</strong>')


def sync(path, papers, stars):
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()

    def cite_sub(m):
        n = papers.get(m.group(2))
        if not n:
            return m.group(0)
        return m.group(1) + format(n // 100 * 100, ",") + "+ citations"

    def star_sub(m):
        n = stars.get(m.group(2))
        if n is None:
            return m.group(0)
        return m.group(1) + fmt_stars(n) + " stars</strong>"

    new = CITE_RE.sub(cite_sub, text)
    new = STAR_RE.sub(star_sub, new)
    if new != text:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new)
        return True
    return False


def main():
    papers = load("scholar.json").get("papers", {})
    stars = load("stars.json").get("stars", {})
    for page in PAGES:
        path = os.path.join(ROOT, page)
        changed = sync(path, papers, stars)
        print("%s: %s" % (page, "updated" if changed else "already in sync"))


if __name__ == "__main__":
    main()
