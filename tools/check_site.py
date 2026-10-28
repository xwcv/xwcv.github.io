#!/usr/bin/env python3
"""Validate the generated xwcv site against data/papers.json and projects.json."""
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pagegen import NAV_ITEMS, canonical, counterpart, page_context
from sitegen import (
    RESOURCE_LABELS, ROOT, TOPICS, anchors, arxiv_id, load, normalize, paper_index,
    project_venue, title_info,
)

PAGES = [
    "index.html", "index_cn.html", "pubs.html", "pubs_cn.html",
    "projs.html", "projs_cn.html", "group.html", "group_cn.html", "404.html",
]
PUBS = ["pubs.html", "pubs_cn.html"]
PROJS = ["projs.html", "projs_cn.html"]
DUP_EXTENSIONS = {
    "ccnetcrisscrossattentionforsemanticsegmentation",
    "deeplearningrepresentationusingautoencoderfor3dshaperetrieval",
    "scalespacesiftflow",
}
failures = []


def report(ok, message):
    print(("PASS" if ok else "FAIL") + "  " + message)
    if not ok:
        failures.append(message)


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.main = 0
        self.h1 = 0
        self.ids = []
        self.refs = []
        self.canonical = None
        self.alternates = {}
        self.meta = {}
        self.assets = []
        self.videos = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = a.get("class", "").split()
        if tag == "main" and "container" in classes:
            self.main += 1
        if tag == "h1":
            self.h1 += 1
        if a.get("id"):
            self.ids.append(a["id"])
        for attr in ("href", "src", "poster", "data-src"):
            if attr in a:
                self.refs.append((tag, attr, a[attr]))
        if tag == "video":
            self.videos.append(a)
        if tag == "link" and a.get("rel") == "canonical":
            self.canonical = a.get("href")
        if tag == "link" and a.get("rel") == "alternate" and a.get("hreflang"):
            self.alternates[a["hreflang"]] = a.get("href")
        if tag == "meta" and (a.get("property") or a.get("name")):
            self.meta[a.get("property") or a.get("name")] = a.get("content", "")
        if tag == "link" and "xwcv.css" in a.get("href", ""):
            self.assets.append(a["href"])
        if tag == "script" and "site.js" in a.get("src", ""):
            self.assets.append(a["src"])


parsed = {}


def parse(path):
    if path not in parsed:
        parser = PageParser()
        parser.feed((ROOT / path).read_text(encoding="utf-8"))
        parsed[path] = parser
    return parsed[path]


def validate_refs(path, parser):
    bad = []
    for tag, attr, value in parser.refs:
        if not value:
            bad.append(path + ": empty " + tag + "[" + attr + "]")
            continue
        try:
            url = urlsplit(value)
        except ValueError:
            bad.append(path + ": malformed URL " + value)
            continue
        if url.scheme or url.netloc or not url.path:
            continue
        target_path = unquote(url.path)
        target = ROOT / target_path.lstrip("/") if target_path.startswith("/") else (ROOT / path).parent / target_path
        if target_path.endswith("/"):
            target = target / "index.html"
        if not target.exists():
            bad.append(path + " -> " + value)
            continue
        if url.fragment and target.suffix == ".html" and url.fragment not in parse(str(target.relative_to(ROOT))).ids:
            bad.append(path + " -> missing anchor " + value)
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("//"):
            continue
        for photo in re.findall(r'photo:\s*"([^"]+)"', line):
            target = (ROOT / path).parent / photo
            if not target.exists():
                bad.append(path + " -> " + photo)
    report(not bad, path + ": local resources and anchors resolve" + (" -> " + "; ".join(bad[:4]) if bad else ""))


def paper_lis(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    return [match.group(0) for match in re.finditer(r"<li\b[^>]*>.*?</li>", source, re.S)
            if 'class="year-heading"' not in match.group(0)]


def paper_titles(path):
    return [normalize(title_info(item)["text"]) for item in paper_lis(path)]


def expected_paper_titles(papers):
    by_section = defaultdict(list)
    for paper in papers:
        by_section[paper["section"]].append(paper)
    if by_section["tech"]:
        by_section["tech"].sort(key=lambda p: tuple(int(x) for x in (arxiv_id(p["li_html"]) or "0000.0000").split(".")), reverse=True)
    expected = []
    for section in ("major", "other", "book", "misc", "tech"):
        expected.extend(normalize(title_info(paper["li_html"])["text"]) for paper in by_section[section])
    return expected


def validate_data(papers, projects, config):
    paper_ids = [paper.get("id", "") for paper in papers]
    report(paper_ids and len(paper_ids) == len(set(paper_ids)), "paper IDs are present and unique")
    project_ids = [project.get("id", "") for project in projects]
    report(project_ids and len(project_ids) == len(set(project_ids)), "project IDs are present and unique")
    report(all(project.get("pub_id") in paper_ids for project in projects), "every project pub_id references an existing paper")
    valid_sections = {"major", "other", "book", "misc", "tech"}
    bad = [paper["id"] for paper in papers if paper.get("section") not in valid_sections or (paper["section"] == "major" and not paper.get("year_group"))]
    report(not bad, "paper sections and Major year groups are valid" + (" -> " + ", ".join(bad[:3]) if bad else ""))
    bad = [paper["id"] for paper in papers if any(topic not in TOPICS for topic in paper.get("topics", []))]
    report(not bad, "paper topics use the controlled vocabulary" + (" -> " + ", ".join(bad[:3]) if bad else ""))
    bad = []
    for paper in papers:
        try:
            paper_index(paper)
        except Exception as exc:
            bad.append(paper.get("id", "<missing>") + ": " + str(exc))
        for link in anchors(paper["li_html"]):
            if link["label"].lower() in RESOURCE_LABELS and not link["href"]:
                bad.append(paper["id"] + ": empty " + link["label"])
    report(not bad, "paper rows expose valid derived metadata and links" + (" -> " + "; ".join(bad[:3]) if bad else ""))
    bad = []
    for project in projects:
        if not project.get("description_en") or not project.get("description_cn"):
            bad.append(project["id"] + ": description")
        media = project.get("media", {})
        if not media.get("src") or not (ROOT / media.get("src", "")).exists():
            bad.append(project["id"] + ": media")
        if media.get("video") and (not media.get("poster") or not (ROOT / media["poster"]).exists()):
            bad.append(project["id"] + ": poster")
        for link in project.get("links", []):
            if not link.get("href") or link.get("label", "").lower() not in RESOURCE_LABELS:
                bad.append(project["id"] + ": link " + link.get("label", "<missing>"))
    report(not bad, "project descriptions, media and links are valid" + (" -> " + "; ".join(bad[:4]) if bad else ""))
    versions_ok = re.fullmatch(r"\d{8}-\d+", config.get("css_version", "")) and re.fullmatch(r"\d{8}-\d+", config.get("js_version", ""))
    report(bool(versions_ok), "shared asset versions use the expected YYYYMMDD-N format")


def validate_removed_features():
    stale = []
    for path in PUBS:
        source = (ROOT / path).read_text(encoding="utf-8")
        for marker in ("paper-abs", "paper-abs-toggle", "pub-filter", "data-paper-id"):
            if marker in source:
                stale.append(path + ":" + marker)
    for path in PAGES:
        source = (ROOT / path).read_text(encoding="utf-8")
        for marker in ('href="research.html', 'href="research_cn.html', 'href="projects/', 'href="projects_cn/'):
            if marker in source:
                stale.append(path + ":" + marker)
    report(not stale, "removed research/detail/abstract features stay removed" + (" -> " + "; ".join(stale[:4]) if stale else ""))


def main():
    papers = load("papers.json")
    projects = load("projects.json")
    config = load("site.json")
    paper_map = {paper["id"]: paper for paper in papers}
    validate_data(papers, projects, config)
    validate_removed_features()

    for path in PAGES:
        parser = parse(path)
        report(parser.main == 1 and parser.h1 == 1, path + ": one main.container and one h1")
        report(len(parser.ids) == len(set(parser.ids)), path + ": element IDs are unique")
        lang, prefix, _, _ = page_context(path)
        nav = re.search(r'<nav class="topnav".*?</nav>', (ROOT / path).read_text(encoding="utf-8"), re.S)
        missing = []
        for _, _, href in NAV_ITEMS[lang]:
            target = prefix + href
            if not nav or 'href="' + target + '"' not in nav.group(0):
                missing.append(target)
        report(not missing, path + ": localized navigation targets are valid" + (" -> " + ", ".join(missing) if missing else ""))
        validate_refs(path, parser)
        for asset in parser.assets:
            expected = config["css_version"] if "xwcv.css" in asset else config["js_version"]
            report("v=" + expected in asset, path + ": current asset version " + expected)
        for video in parser.videos:
            report(bool(video.get("data-src")) and bool(video.get("poster")) and "autoplay" not in video and "src" not in video,
                   path + ": deferred video has data-src/poster and no autoplay/src")
        if path == "404.html":
            continue
        expected = canonical(path)
        report(parser.canonical == expected, path + ": canonical is correct")
        is_cn = path.endswith("_cn.html")
        en_path = counterpart(path) if is_cn else path
        cn_path = path if is_cn else counterpart(path)
        expected_alternates = {"en": canonical(en_path), "zh-CN": canonical(cn_path), "x-default": canonical(en_path)}
        report(parser.alternates == expected_alternates, path + ": hreflang pair is complete and reciprocal")
        for key in ("og:url", "twitter:url"):
            if key in parser.meta:
                report(parser.meta[key] == expected, path + ": " + key + " matches canonical")

    pubs_en = paper_titles("pubs.html")
    pubs_cn = paper_titles("pubs_cn.html")
    report(len(pubs_en) == len(papers) and pubs_en == pubs_cn,
           "pubs EN/CN contain all %d paper entries in the same order" % len(papers))
    report(pubs_en == expected_paper_titles(papers), "pubs content order matches data/papers.json exactly")
    duplicates = defaultdict(int)
    arxiv_duplicates = defaultdict(list)
    for item in paper_lis("pubs.html"):
        title = normalize(title_info(item)["text"])
        duplicates[title] += 1
        aid = arxiv_id(item)
        if aid:
            arxiv_duplicates[aid].append(title)
    unexpected = [key for key, count in duplicates.items() if count > 1 and key not in DUP_EXTENSIONS]
    report(not unexpected, "no unexpected duplicate paper titles" + (" -> " + ", ".join(unexpected[:3]) if unexpected else ""))
    unexpected_arxiv = [key for key, titles in arxiv_duplicates.items() if len(titles) > 1 and any(title not in DUP_EXTENSIONS for title in titles)]
    report(not unexpected_arxiv, "no unexpected duplicate arXiv IDs")

    expected_values = []
    number = 1
    groups = []
    for paper in papers:
        if paper["section"] != "major":
            continue
        if not groups or groups[-1][0] != paper["year_group"]:
            groups.append((paper["year_group"], []))
        groups[-1][1].append(paper)
    for _, group in groups:
        expected_values.append(number)
        number += len(group)
    for path in PUBS:
        values = [int(value) for value in re.findall(r'<li\b[^>]*value="(\d+)"', (ROOT / path).read_text(encoding="utf-8"))]
        report(values == expected_values, path + ": major year numbering anchors are continuous")

    tech = re.search(r'<section\b[^>]*aria-labelledby="tech-reports-heading"[^>]*>(.*?)</section>',
                     (ROOT / "pubs.html").read_text(encoding="utf-8"), re.S).group(1)
    tech_keys = [tuple(int(x) for x in (arxiv_id(item) or "0000.0000").split(".")) for item in re.findall(r"<li\b[^>]*>.*?</li>", tech, re.S)]
    report(tech_keys == sorted(tech_keys, reverse=True), "technical reports are strictly descending by arXiv ID")

    for path in PROJS:
        source = (ROOT / path).read_text(encoding="utf-8")
        cards = re.findall(r'<li class="proj-card">(.*?)</li>', source, re.S)
        report(len(cards) == len(projects), path + ": all %d project cards render" % len(projects))
        bad_venues = []
        for card, project in zip(cards, projects):
            match = re.search(r'<span class="venue-tag ([^"]+)">(.*?)</span>', card, re.S)
            expected = project_venue(project, paper_map)
            if not match or match.group(1) != expected["class"] or normalize(match.group(2)) != normalize(expected["label"]):
                bad_venues.append(project["id"])
        report(not bad_venues, path + ": project venues stay synchronized with papers" + (" -> " + ", ".join(bad_venues[:3]) if bad_venues else ""))

    for path in ("index.html", "index_cn.html"):
        source = (ROOT / path).read_text(encoding="utf-8")
        hrefs = re.findall(r'<a class="gal-card" href="([^"]+)"', source)
        report(hrefs == [project["href"] for project in projects],
               path + ": homepage retains all %d external project links" % len(projects))

    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    tree = ET.parse(ROOT / "sitemap.xml")
    sitemap_urls = {node.findtext("s:loc", namespaces=ns) for node in tree.getroot().findall("s:url", ns)}
    expected_urls = {canonical(path) for path in PAGES if path != "404.html"}
    report(sitemap_urls == expected_urls, "sitemap covers exactly all %d indexable static pages" % len(expected_urls))

    print()
    if failures:
        print("%d check(s) failed" % len(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
