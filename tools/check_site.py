#!/usr/bin/env python3
"""Consistency checks for the xwcv site. Run locally or in CI (both update
workflows run it as a non-blocking warning step).

Checks:
1. EN/CN pubs pages carry the same titled entries in the same order
2. no duplicate paper titles inside one pubs page (normalized), except the
   whitelisted conference+journal-extension pairs
3. every local link/src (bib/, pubs/, res/) resolves to an existing file
4. projs EN/CN card counts match; a card whose paper is in the major pubs
   list must not carry an arXiv venue tag (venue sync rule)

Exit code 1 if any check fails; findings are printed as a report.
"""
import html
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PUBS = ["pubs.html", "pubs_cn.html"]
PROJS = ["projs.html", "projs_cn.html"]
PAGES = PUBS + PROJS + ["index.html", "index_cn.html",
                        "group.html", "group_cn.html", "404.html"]

# conference paper + journal extension are intentionally separate entries
DUP_WHITELIST = {
    "ccnetcrisscrossattentionforsemanticsegmentation",
    "deeplearningrepresentationusingautoencoderfor3dshaperetrieval",
}

failures = []


def report(ok, msg):
    print(("PASS" if ok else "FAIL") + "  " + msg)
    if not ok:
        failures.append(msg)


def titled_entries(path):
    s = open(os.path.join(ROOT, path), encoding="utf-8").read()
    out = []
    for m in re.finditer(r"<li[^>]*><p>(.*?)</p></li>", s, re.S):
        t = re.search(r"<strong>(.*?)</strong>", m.group(1), re.S)
        if t and len(t.group(1)) > 8:
            out.append(html.unescape(re.sub(r"\s+", " ", t.group(1))).strip())
    return out


def norm(t):
    return re.sub(r"[^a-z0-9]+", "", t.lower())


# 1. EN/CN pubs identical entry lists
en, cn = (titled_entries(p) for p in PUBS)
report(en == cn, "pubs EN/CN entries identical (%d vs %d)" % (len(en), len(cn)))

# 2. duplicate titles within one pubs page
for p in PUBS:
    seen = {}
    dups = []
    for t in titled_entries(p):
        n = norm(t)
        if n in seen and n not in DUP_WHITELIST:
            dups.append(t[:60])
        seen[n] = True
    report(not dups, "%s: no unexpected duplicate titles%s"
           % (p, (" -> " + "; ".join(dups)) if dups else ""))

# 3. local links resolve
missing = []
for p in PAGES:
    s = open(os.path.join(ROOT, p), encoding="utf-8").read()
    refs = re.findall(r'(?:href|src|poster|data-src)="(\.?/?(?:bib|pubs|res)/[^"#?]+?)(?:\?[^"#]*)?"', s)
    refs += re.findall(r'photo: "(\./res/[^"]+)"', s)
    for r in refs:
        fs = os.path.join(ROOT, r.lstrip("./"))
        if not os.path.exists(fs):
            missing.append("%s -> %s" % (p, r))
report(not missing, "all local links resolve%s"
       % ("" if not missing else " -> " + "; ".join(missing[:5])))

# 4a. projs EN/CN card counts match
counts = []
for p in PROJS:
    s = open(os.path.join(ROOT, p), encoding="utf-8").read()
    counts.append(len(re.findall(r'<li class="proj-card">', s)))
report(counts[0] == counts[1], "projs EN/CN card counts equal (%d vs %d)"
       % tuple(counts))

# 4b. venue sync: a projs card whose paper is in the major pubs list must not
# be tagged arXiv
major = open(os.path.join(ROOT, "pubs.html"), encoding="utf-8").read()
mi = major.find('id="major-papers-heading"')
major = norm(major[mi:major.find("</ol>", mi)])
stale = []
s = open(os.path.join(ROOT, "projs.html"), encoding="utf-8").read()
for m in re.finditer(r'<li class="proj-card">(.*?)</li>', s, re.S):
    b = m.group(1)
    if 'venue-tag v-journal' not in b:
        continue
    title = re.search(r'proj-title">(.*?)</h3>', b, re.S).group(1)
    t = norm(html.unescape(title))
    # pubs titles may drop the "Project:" prefix — match the longest tail
    for cut in (0, t.find(":") + 1 if ":" in t else 0):
        tail = t[cut:]
        if len(tail) > 15 and tail in major:
            stale.append(title[:50])
            break
report(not stale, "no arXiv-tagged projs card already in the major pubs list%s"
       % ("" if not stale else " -> " + "; ".join(stale)))

print()
if failures:
    print("%d check(s) failed" % len(failures))
    sys.exit(1)
print("all checks passed")
