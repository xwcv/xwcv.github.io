#!/usr/bin/env python3
"""Regenerate res/og-card.png (the 1200x630 social share card) with the live
citation / h-index numbers from res/scholar.json.

Renders tools/ogcard_template.html through headless Chrome (preinstalled on
GitHub Actions runners; on macOS the standard Chrome path is used). Rewrites
the PNG only when the numbers actually changed, so the workflow commits stay
quiet on no-change weeks.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEMPLATE = os.path.join(ROOT, "tools", "ogcard_template.html")
OUT = os.path.join(ROOT, "res", "og-card.png")
TMP_HTML = os.path.join(ROOT, "_ogcard_render.html")

CHROME_CANDIDATES = [
    os.environ.get("CHROME", ""),
    shutil.which("google-chrome") or "",
    shutil.which("chrome") or "",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def fmt_k(n):
    return "%d,000+" % (n // 1000) if n >= 1000 else str(n)


def main():
    with open(os.path.join(ROOT, "res", "scholar.json"), encoding="utf-8") as f:
        d = json.load(f)
    citations = int(d.get("citations", 0))
    hindex = int(d.get("hindex", 0))
    if not citations or not hindex:
        print("scholar.json missing citation numbers, skipping")
        sys.exit(1)

    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("{{CITATIONS}}", fmt_k(citations)).replace("{{HINDEX}}", str(hindex))

    # the card was authored for these numbers; bail if the PNG already matches
    marker = "%s|%s" % (fmt_k(citations), hindex)
    state = os.path.join(ROOT, "res", ".og-card-state")
    if os.path.exists(OUT) and os.path.exists(state):
        with open(state, encoding="utf-8") as f:
            if f.read().strip() == marker:
                print("og-card.png already up to date (%s)" % marker)
                return

    chrome = next((c for c in CHROME_CANDIDATES if c and os.path.exists(c)), None)
    if not chrome:
        print("no Chrome binary found, cannot render the card")
        sys.exit(1)

    with open(TMP_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    try:
        subprocess.run([
            chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--window-size=1200,630", "--virtual-time-budget=3000",
            "--screenshot=" + OUT, "file://" + TMP_HTML,
        ], check=True, capture_output=True)
    finally:
        os.remove(TMP_HTML)

    with open(state, "w", encoding="utf-8") as f:
        f.write(marker)
    print("wrote res/og-card.png (%s)" % marker)


if __name__ == "__main__":
    main()
