#!/usr/bin/env python3
"""Fetch citation stats via SerpAPI's Google Scholar Author endpoint and
update res/scholar.json.

Fetches both the profile-level stats (total citations, h-index) and the
per-publication citation counts, keyed by each paper's cluster id — the id in
the "citation_for_view" link that every "N+ citations" badge on the site uses,
so res/site.js can refresh those badges from the same JSON file.

Direct scraping of Google Scholar is blocked (HTTP 403) from CI machines, so
this goes through SerpAPI (https://serpapi.com). The API key is read from the
SERPAPI_API_KEY environment variable (stored as a GitHub Actions secret).

Note on quota: SerpAPI's free plan allows 100 searches/month. The profile
stats come with the first page of articles; the remaining article pages cost
one request per 20 papers (~15 extra requests for ~300 papers), so the
workflow runs weekly to stay within the free quota.

On any failure the existing JSON is left untouched and the script exits 1 so
the workflow run is visibly marked as failed.
"""
import datetime
import json
import os
import sys
import urllib.parse
import urllib.request

USER_ID = "qNCTLV0AAAAJ"
API_URL = "https://serpapi.com/search.json"
PAGE_SIZE = 20  # SerpAPI returns 20 articles per page for author profiles
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "res", "scholar.json")

API_KEY = os.environ.get("SERPAPI_API_KEY", "")


def fetch(params):
    params = dict(params)
    params.update({
        "engine": "google_scholar_author",
        "author_id": USER_ID,
        "hl": "en",
        "api_key": API_KEY,
    })
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", "ignore"))
    if "error" in data:
        raise RuntimeError("SerpAPI error: %s" % data["error"])
    return data


def parse_stats(data):
    """Extract (citations_all, h_index_all) from the cited_by table.

    The table rows are Citations / h-index / i10-index in document order;
    row keys are localized, so match positionally but defensively.
    """
    table = data.get("cited_by", {}).get("table", [])
    if len(table) < 2:
        raise RuntimeError("cited_by table missing or too short")

    def row_all(row):
        for cell in row.values():
            if isinstance(cell, dict) and "all" in cell:
                return int(cell["all"])
        raise RuntimeError("could not find 'all' value in cited_by row")

    return row_all(table[0]), row_all(table[1])


def parse_articles(data):
    """Return {cluster_id: citation_count} from one page of articles."""
    papers = {}
    for art in data.get("articles", []):
        cid = art.get("citation_id", "")
        if ":" not in cid:
            continue
        cluster = cid.split(":", 1)[1]
        papers[cluster] = int(art.get("cited_by", {}).get("value", 0))
    return papers


def fetch_all_papers(first_page):
    """Paginate through the article list; first_page is already fetched."""
    papers = parse_articles(first_page)
    cstart = len(papers)
    while cstart and cstart % PAGE_SIZE == 0:
        page = fetch({"cstart": cstart})
        chunk = parse_articles(page)
        if not chunk:
            break
        papers.update(chunk)
        if len(chunk) < PAGE_SIZE:
            break
        cstart += len(chunk)
    return papers


def main():
    if not API_KEY:
        print("SERPAPI_API_KEY is not set")
        sys.exit(1)

    try:
        first_page = fetch({})
        citations, hindex = parse_stats(first_page)
    except Exception as e:
        print("fetch failed, keeping existing data: %s" % e)
        sys.exit(1)

    old = {}
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            pass

    # Sanity check: citations should never drop; h-index may legitimately
    # fluctuate by 1 (Scholar merges/recounts), so only flag a larger drop —
    # a much smaller number means a bad fetch.
    if (citations < int(old.get("citations", 0))
            or hindex < int(old.get("hindex", 0)) - 1):
        print("parsed values (%d, %d) lower than existing, skipping" % (citations, hindex))
        sys.exit(1)

    # Per-paper counts are best-effort: keep the old ones if pagination fails.
    try:
        papers = fetch_all_papers(first_page)
    except Exception as e:
        print("paper list pagination failed, keeping existing: %s" % e)
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
    print("updated: citations=%d h-index=%d papers=%d" % (citations, hindex, len(papers)))


if __name__ == "__main__":
    main()
