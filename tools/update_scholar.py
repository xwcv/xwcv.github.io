#!/usr/bin/env python3
"""Fetch citation stats from a public Google Scholar profile and update res/scholar.json.

Google Scholar offers no official API, so this scrapes the public profile page.
The fetch can be blocked (HTTP 429 / CAPTCHA), especially from CI machines —
in that case the existing JSON is left untouched and the script exits 0.
"""
import datetime
import json
import os
import re
import sys
import urllib.request

USER_ID = "qNCTLV0AAAAJ"
URL = "https://scholar.google.com/citations?hl=en&user=" + USER_ID
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "res", "scholar.json")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", "ignore")
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

    if citations == old.get("citations") and hindex == old.get("hindex"):
        print("no change: citations=%d h-index=%d" % (citations, hindex))
        return

    data = {
        "citations": citations,
        "hindex": hindex,
        "updated": datetime.date.today().isoformat(),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print("updated: %s" % json.dumps(data))


if __name__ == "__main__":
    main()
