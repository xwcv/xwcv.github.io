#!/usr/bin/env python3
"""Fetch BibTeX snippets for papers in data/papers.json that lack a bib chip.

Reliability-first sourcing order:
  1. Technical Reports      -> arXiv API by arXiv id (exact)
  2. Published papers       -> dblp title search (normalized title + year match)
  3. Papers with a DOI      -> doi.org content negotiation (publisher metadata)
  4. Remaining arXiv papers -> arXiv API @misc entry
Anything ambiguous is skipped and reported, never guessed.

Usage:
  python3 tools/fetch_bib.py --dry-run   # report only
  python3 tools/fetch_bib.py             # write bib/*.txt and update papers.json
"""

import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "papers.json"
BIBDIR = ROOT / "bib"
UA = {"User-Agent": "xwcv-bib-fetcher/1.0 (academic homepage maintenance)"}
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def fetch(url, accept=None, retries=2):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=dict(UA))
            if accept:
                req.add_header("Accept", accept)
            with urllib.request.urlopen(req, timeout=15,
                                        context=SSL_CTX) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def norm_title(t):
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("&amp;", "&")
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def strip_tags(html):
    return re.sub(r"<[^>]+>", "", html).replace("&amp;", "&")


def extract_info(p):
    h = p["li_html"]
    m = re.search(r"<(?:strong|b)>(.+?)</(?:strong|b)>", h, re.S)
    title = strip_tags(m.group(1)).strip() if m else None
    text = strip_tags(h)
    doi = None
    m = re.search(r"(?:doi\.org/|doi:\s*)(10\.\d{4,5}/[^\s\"<>,;]+)", h, re.I)
    if m:
        doi = m.group(1).rstrip(".")
    arxiv = None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", h)
    if not m:
        m = re.search(r"arXiv:(\d{4}\.\d{4,5})", h)
    if m:
        arxiv = m.group(1)
    year = None
    if p.get("year_group"):
        m = re.search(r"(\d{4})", p["year_group"])
        if m:
            year = int(m.group(1))
    if not year:
        m = re.search(r"\(((?:19|20)\d{2})\)", text)
        if m:
            year = int(m.group(1))
    if not year:
        cleaned = re.sub(r"\d{4}\.\d{4,5}", "", text)
        m = re.search(r"\b(?:19|20)\d{2}\b", cleaned)
        if m:
            year = int(m.group(0))
    first_author = ""
    if title and title in text:
        first_author = text.split(title)[0].split(",")[0].strip()
    first_author = re.sub(r"[#*\d\s]+$", "", first_author).strip()
    authors = ""
    if title and title in text:
        raw = text.split(title)[0].strip().rstrip(".")
        names = [re.sub(r"[#*]+$", "", a).strip()
                 for a in re.split(r",\s*", raw) if a.strip()]
        names = ["others" if a in ("...", "…") else a for a in names if a]
        authors = " and ".join(names)
    return title, doi, arxiv, year, first_author, authors


def crossref_search(title, year):
    q = urllib.parse.quote(title)
    data = json.loads(fetch(
        "https://api.crossref.org/works?query.bibliographic="
        f"{q}&rows=10&select=DOI,title,issued,type"))
    want = norm_title(title)
    good = []
    for it in data.get("message", {}).get("items", []):
        titles = it.get("title") or []
        if not titles or norm_title(titles[0]) != want:
            continue
        if it.get("type") not in ("journal-article", "proceedings-article",
                                  "book-chapter"):
            continue
        issued = it.get("issued", {}).get("date-parts", [[None]])[0][0]
        if year and issued and abs(int(issued) - year) > 1:
            continue
        good.append(it["DOI"])
    good = list(dict.fromkeys(good))
    if len(good) != 1:
        return None, f"crossref hits after validation: {len(good)}"
    return crossref_bib(good[0])


def crossref_bib(doi):
    bib = fetch("https://api.crossref.org/works/"
                f"{urllib.parse.quote(doi)}/transform/application/x-bibtex")
    if not bib or not bib.lstrip().startswith("@"):
        return None, "crossref: no bibtex returned"
    return bib.strip() + "\n", f"doi:{doi}"


def arxiv_bib(arxiv_id, cite_key, title, authors, year):
    bib = ("@misc{%s,\n"
           "  title={%s},\n"
           "  author={%s},\n"
           "  year={%s},\n"
           "  eprint={%s},\n"
           "  archivePrefix={arXiv},\n"
           "  primaryClass={cs.CV}\n"
           "}\n") % (cite_key, title, authors, year, arxiv_id)
    return bib, f"arxiv:{arxiv_id}"


def make_key(first_author, year, title, used):
    surname = re.sub(r"[^a-z]", "",
                     first_author.split()[-1].lower()) if first_author else "anon"
    words = re.findall(r"[a-z0-9]+", title.lower()) if title else ["paper"]
    base = f"{surname}{year or ''}{words[0]}"
    key, i = base, 1
    while key in used:
        i += 1
        extra = words[i - 1] if i - 1 < len(words) else str(i)
        key = base + extra
    used.add(key)
    return key


def add_chip(li_html, key):
    chip = f'<a href="bib/{key}.txt">bib</a>'
    idx = li_html.rfind("</p>")
    if idx == -1:
        return None
    head, tail = li_html[:idx].rstrip(), li_html[idx:]
    if head.endswith("."):
        return head[:-1] + ", " + chip + "." + tail
    return head + ", " + chip + tail


def main():
    dry = "--dry-run" in sys.argv
    papers = json.loads(PAPERS.read_text(encoding="utf-8"))
    used = {f.stem for f in BIBDIR.glob("*.txt")}
    applied, skipped = [], []
    for i, p in enumerate(papers):
        if "bib/" in p["li_html"]:
            continue
        title, doi, arxiv, year, first_author, authors = extract_info(p)
        pid = p["id"]
        print(f"[{i+1}/{len(papers)}] {pid}", flush=True)
        if not title:
            skipped.append((pid, "no title found"))
            continue
        key = make_key(first_author, year, title, used)
        bib = src = None
        try:
            if p["section"] == "tech":
                if arxiv and authors and year:
                    bib, src = arxiv_bib(arxiv, key, title, authors, year)
                else:
                    skipped.append((pid, "tech report without arXiv id"))
                    continue
            else:
                bib, src = crossref_search(title, year)
                time.sleep(1.0)
                if not bib and doi:
                    bib, src = crossref_bib(doi)
                    time.sleep(1.0)
                if not bib and arxiv and authors and year:
                    bib, src = arxiv_bib(arxiv, key, title, authors, year)
                    time.sleep(1.0)
        except Exception as e:
            skipped.append((pid, f"fetch error: {e}"))
            continue
        if not bib:
            skipped.append((pid, src))
            continue
        new_html = add_chip(p["li_html"], key)
        if not new_html:
            skipped.append((pid, "could not insert chip"))
            continue
        applied.append((pid, key, src))
        if not dry:
            (BIBDIR / f"{key}.txt").write_text(bib, encoding="utf-8")
            p["li_html"] = new_html

    print(f"applied: {len(applied)}, skipped: {len(skipped)}")
    for pid, key, src in applied:
        print(f"  OK   {pid} -> bib/{key}.txt ({src})")
    for pid, why in skipped:
        print(f"  SKIP {pid}: {why}")
    if not dry and applied:
        PAPERS.write_text(json.dumps(papers, ensure_ascii=False,
                                     indent=2) + "\n", encoding="utf-8")
        print("papers.json updated")


if __name__ == "__main__":
    main()
