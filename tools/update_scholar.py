#!/usr/bin/env python3
"""Fetch citation stats from a public Google Scholar profile and update res/scholar.json.

Fetches both the profile-level stats (total citations, h-index) and the
per-publication citation counts, keyed by each paper's cluster id — the id in
the "citation_for_view" link that every "N+ citations" badge on the site uses,
so res/site.js can refresh those badges from the same JSON file.

Google Scholar offers no official API, so this scrapes the public profile pages.
The fetch can be blocked (HTTP 429 / CAPTCHA), especially from CI machines —
in that case the existing JSON is left untouched and the script exits 0.
"""
import datetime
import json
import os
import re
import urllib.request

USER_ID = "qNCTLV0AAAAJ"
URL = "https://scholar.google.com/citations?hl=en&user=" + USER_ID
LIST_URL = URL + "&view_op=list_works&pagesize=100&cstart=%d"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "res", "scholar.json")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", "ignore")


def fetch_papers():
    """Return {cluster_id: citation_count} from the publication list pages."""
    papers = {}
    cstart = 0
    while True:
        html = fetch(LIST_URL % cstart)
        # One title link and one citation-count cell per row, in the same order.
        clusters = re.findall(
            r'<a href="[^"]*citation_for_view=' + USER_ID + r':([\w-]+)[^"]*" class="gsc_a_at"',
            html)
        counts = re.findall(r'<a [^>]*class="gsc_a_ac[^"]*"[^>]*>(\d*)<', html)
        if not clusters or len(clusters) != len(counts):
            break
        for c, n in zip(clusters, counts):
            papers[c] = int(n) if n else 0
        if len(clusters) < 100:
            break
        cstart += 100
    return papers


def main():
    try:
        html = fetch(URL)
    except Exception as e:
        print("fetch failed, keeping existing data: %s" % e)
        return

    # Profile stats table: rows are Citations / h-index / i10-index,
    # each with "all" and "since 20xx" cells, in document order.
    vals = re.findall(r'gsc_rsb_std">(\d+)<', html)
    if len(vals) < 3:
        print("could not parse stats table, keeping existing data")
        return
    citations, hindex = int(vals[0]), int(vals[2])

    old = {}
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            pass

    # Sanity check: stats should never drop; a smaller number means a bad fetch.
    if citations < int(old.get("citations", 0)) or hindex < int(old.get("hindex", 0)):
        print("parsed values (%d, %d) lower than existing, skipping" % (citations, hindex))
        return

    # Per-paper counts are best-effort: keep the old ones if the list pages
    # could not be fetched or parsed.
    try:
        papers = fetch_papers()
    except Exception as e:
        print("paper list fetch failed, keeping existing: %s" % e)
        papers = {}
    if not papers:
        papers = old.get("papers", {})

    if (citations == old.get("citations") and hindex == old.get("hindex")
            and papers == old.get("papers", {})):
        print("no change: citations=%d h-index=%d" % (citations, hindex))
        return

    data = {
        "citations": citations,
        "hindex": hindex,
        "updated": datetime.date.today().isoformat(),
        "papers": papers,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print("updated: %s" % json.dumps(data))


if __name__ == "__main__":
    main()
