#!/usr/bin/env python3
"""Fetch GitHub star counts for the code repos linked from pubs.htm and
write res/stars.json, which res/site.js uses to render a "star" badge next
to each code link.

Repos are extracted by taking the first two path segments of every
https://github.com/<owner>/<repo> link in pubs.htm (a few links point into
subdirectories of a repo). Each unique repo is queried once via the GitHub
REST API.

The GITHUB_TOKEN env var is used when present (the scheduled workflow passes
the built-in Actions token). Without a token the API allows only 60
requests/hour, below the number of linked repos, so the script refuses to
run unauthenticated.

A repo that errors (e.g. 404 for a not-yet-public repo) is skipped and keeps
its previous count; if nothing could be fetched at all the existing JSON is
left untouched and the script exits 1 so the workflow run is visibly marked
as failed.
"""
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PUBS = os.path.join(ROOT, "pubs.htm")
OUT = os.path.join(ROOT, "res", "stars.json")
API_URL = "https://api.github.com/repos/"

TOKEN = os.environ.get("GITHUB_TOKEN", "")

REPO_RE = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


def find_repos():
    with open(PUBS, encoding="utf-8") as f:
        html = f.read()
    repos = []
    for owner, repo in REPO_RE.findall(html):
        full = owner + "/" + repo
        if full not in repos:
            repos.append(full)
    return repos


def fetch_stars(full):
    req = urllib.request.Request(API_URL + full, headers={
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer " + TOKEN,
        "User-Agent": "xwcv-stars-updater",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", "ignore"))
    return int(data["stargazers_count"])


def main():
    if not TOKEN:
        print("GITHUB_TOKEN is not set (unauthenticated API quota is too small)")
        sys.exit(1)

    repos = find_repos()
    print("found %d unique repos in pubs.htm" % len(repos))

    old = {}
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                old = json.load(f).get("stars", {})
        except Exception:
            pass

    stars = dict(old)
    fetched = 0
    for full in repos:
        for attempt in range(3):
            try:
                stars[full] = fetch_stars(full)
                fetched += 1
                break
            except urllib.error.HTTPError as e:
                # e.g. 404 for a repo that is not public yet: keep the old value
                print("skipped %s: HTTP %s" % (full, e.code))
                break
            except Exception as e:
                if attempt == 2:
                    print("skipped %s: %s" % (full, e))
                time.sleep(1 + attempt)
        time.sleep(0.1)

    if fetched == 0:
        print("no repo could be fetched, keeping existing data")
        sys.exit(1)

    if stars == old:
        print("no change: %d repos, %d fetched" % (len(stars), fetched))
        return

    data = {
        "updated": datetime.date.today().isoformat(),
        "stars": stars,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    print("updated: %d repos, %d fetched" % (len(stars), fetched))


if __name__ == "__main__":
    main()
