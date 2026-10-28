#!/usr/bin/env python3
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TOPICS = {
    "autonomous-driving": ("Autonomous Driving", "自动驾驶"),
    "embodied-ai": ("Embodied AI", "具身智能"),
    "world-model": ("World Model", "世界模型"),
    "generative-model": ("Generative Model", "生成模型"),
    "multimodal-llm": ("Multimodal LLM", "多模态大模型"),
    "efficient-llm": ("Efficient LLM", "高效 LLM"),
    "3d-4d-vision": ("3D / 4D Vision", "3D / 4D 视觉"),
    "segmentation": ("Segmentation", "图像分割"),
    "detection-tracking": ("Detection & Tracking", "检测与跟踪"),
    "medical-imaging": ("Medical Imaging", "医学影像"),
    "visual-representation": ("Visual Representation", "视觉表征"),
}
TOPIC_BY_LABEL = {en.lower(): slug for slug, (en, zh) in TOPICS.items()}
SECTION_BY_HEADING = {
    "major-papers-heading": "major",
    "other-papers-heading": "other",
    "book-chapters-heading": "book",
    "misc-heading": "misc",
    "tech-reports-heading": "tech",
}
SECTION_HEADING = {value: key for key, value in SECTION_BY_HEADING.items()}
RESOURCE_LABELS = {
    "pdf", "code", "arxiv", "bib", "project page", "project", "video",
    "website", "dataset", "demo", "supplementary", "slides",
}
VENUE_ALIASES = {
    "cvpr": "CVPR", "iccv": "ICCV", "iccvw": "ICCVW", "eccv": "ECCV",
    "neurips": "NeurIPS", "nips": "NeurIPS", "icml": "ICML", "iclr": "ICLR",
    "aaai": "AAAI", "corl": "CoRL", "acm mm": "ACM MM", "acmmm": "ACM MM",
    "icip": "ICIP", "icpr": "ICPR", "prcv": "PRCV", "accv": "ACCV",
    "wacv": "WACV", "ecai": "ECAI", "ccpr": "CCPR", "3dv": "3DV",
    "ieee tpami": "TPAMI", "tpami": "TPAMI", "ijcv": "IJCV",
    "ieee tmi": "TMI", "ieee tip": "TIP", "ieee tcsvt": "TCSVT",
    "ieee tnnls": "TNNLS", "tits": "TITS", "thms": "THMS",
    "ieee tase": "TASE", "ra-l": "RA-L", "scis": "SCIS", "prl": "PRL",
    "jcst": "JCST", "j intell manuf": "J Intell Manuf", "infsci": "InfSci",
    "imavis": "IMAVIS", "cviu": "CVIU", "cvmj": "CVMJ", "apl": "APL",
    "nat. med.": "Nat. Med.", "nature medicine": "Nat. Med.",
    "nat. commun.": "Nat. Commun.", "nature communications": "Nat. Commun.",
    "npj digit. med.": "npj Digit. Med.", "npj digital medicine": "npj Digit. Med.",
    "cell rep. med.": "Cell Rep. Med.", "med": "Med",
    "machine intelligence research": "MIR", "image and vision computing": "IMAVIS",
}
VENUE_CLASSES = {
    "CVPR": "v-cvpr", "ICCV": "v-iccv", "ICCVW": "v-iccv", "ECCV": "v-eccv",
    "NeurIPS": "v-neurips", "ICML": "v-icml", "ICLR": "v-iclr",
    "AAAI": "v-aaai", "CoRL": "v-corl",
}
TOP_VENUES = {"TPAMI", "IJCV", "Nat. Med.", "Nat. Commun.", "npj Digit. Med.", "Cell Rep. Med.", "Med"}


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def dump(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plain(value):
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "", plain(value).lower())


def slugify(value):
    value = plain(value).lower().replace("²", "2").replace("³", "3")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value)).strip("-")


def escape(value):
    return html.escape(str(value), quote=True)


def title_info(li_html):
    for match in re.finditer(r"<(strong|b)\b[^>]*>(.*?)</\1>", li_html, re.S | re.I):
        if len(plain(match.group(2))) > 8:
            return {
                "html": match.group(2),
                "text": plain(match.group(2)),
                "start": match.start(),
                "end": match.end(),
            }
    raise ValueError("Paper title not found in " + li_html[:140])


def anchors(value):
    pattern = r"<a\b[^>]*href=\"([^\"]*)\"[^>]*>(.*?)</a>"
    return [{"href": match.group(1), "html": match.group(2), "label": plain(match.group(2))}
            for match in re.finditer(pattern, value, re.S | re.I)]


def arxiv_id(value):
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", value, re.I)
    return match.group(1) if match else None


def authors_html(li_html, info=None):
    info = info or title_info(li_html)
    p_start = re.search(r"<p\b[^>]*>", li_html[:info["start"]], re.I)
    if not p_start:
        raise ValueError("Paper paragraph not found")
    value = li_html[p_start.end():info["start"]].strip()
    return value[:-1].rstrip() if value.endswith(".") else value


def paper_year(paper, info=None):
    info = info or title_info(paper["li_html"])
    group = paper.get("year_group", "")
    if paper["section"] == "major" and re.fullmatch(r"\d{4}", group):
        return int(group)
    suffix = plain(paper["li_html"][info["end"]:])
    if paper["section"] == "tech":
        match = re.search(r"arXiv:\d{4}\.\d{4,5}\s*\(((?:19|20)\d{2})\)", suffix, re.I)
        if not match:
            match = re.search(r"\(((?:19|20)\d{2})\)", suffix)
        return int(match.group(1)) if match else None
    match = re.search(r"\b((?:19|20)\d{2})\b", suffix)
    return int(match.group(1)) if match else None


def venue_info(paper):
    info = title_info(paper["li_html"])
    year = paper_year(paper, info)
    suffix = paper["li_html"][info["end"]:]
    if paper["section"] == "tech" or "arxiv preprint" in plain(suffix).lower():
        return {"label": ("arXiv " + str(year)) if year else "arXiv", "slug": "arxiv", "class": "v-journal", "year": year}
    canonical = None
    for match in re.finditer(r"<(?:strong|b)\b[^>]*>(.*?)</(?:strong|b)>", suffix, re.S | re.I):
        raw = plain(match.group(1))
        canonical = VENUE_ALIASES.get(raw.lower())
        if canonical:
            break
    if not canonical:
        text = plain(suffix)
        for alias, candidate in VENUE_ALIASES.items():
            if len(alias) > 5 and re.search(r"\b" + re.escape(alias) + r"\b", text.lower()):
                canonical = candidate
                break
    canonical = canonical or "Other"
    if canonical == "Other":
        label = "Other"
        venue_class = "v-journal"
    else:
        label = canonical + (" " + str(year) if year else "")
        venue_class = VENUE_CLASSES.get(canonical, "v-top" if canonical in TOP_VENUES else "v-journal")
    return {"label": label, "slug": slugify(canonical), "class": venue_class, "year": year}


def paper_resources(paper):
    resources = []
    for link in anchors(paper["li_html"]):
        label = link["label"].lower()
        resource = None
        if label == "abstract" or label == "arxiv" or label.startswith("arxiv:") or label == "pdf":
            resource = "pdf"
        elif label in RESOURCE_LABELS:
            resource = label
        if resource:
            resource = resource.replace(" ", "-")
            if resource not in resources:
                resources.append(resource)
    return resources


def paper_index(paper):
    info = title_info(paper["li_html"])
    venue = venue_info(paper)
    return {
        "title_html": info["html"],
        "title": info["text"],
        "authors_html": authors_html(paper["li_html"], info),
        "arxiv": arxiv_id(paper["li_html"]),
        "year": venue["year"],
        "venue": venue,
        "resources": paper_resources(paper),
    }


def project_by_paper(projects):
    return {project["pub_id"]: project for project in projects}


def paper_lookup(papers):
    return papers if isinstance(papers, dict) else {paper["id"]: paper for paper in papers}


def project_title(project, papers):
    papers = paper_lookup(papers)
    return project.get("title_html") or paper_index(papers[project["pub_id"]])["title_html"]


def project_authors(project, papers):
    papers = paper_lookup(papers)
    return project.get("authors_html") or paper_index(papers[project["pub_id"]])["authors_html"]


def project_venue(project, papers):
    papers = paper_lookup(papers)
    venue = paper_index(papers[project["pub_id"]])["venue"]
    return {
        "label": project.get("venue") or venue["label"],
        "class": project.get("venue_class") or venue["class"],
    }


def project_topics(project, papers):
    papers = paper_lookup(papers)
    return papers[project["pub_id"]].get("topics", [])


def topic_label(slug, lang):
    pair = TOPICS[slug]
    return pair[1] if lang == "cn" else pair[0]


def detail_url(project, lang, prefix=""):
    return project["href"]


def project_links(project, lang, prefix=""):
    return project["links"]


def render_media(project, lang, prefix="", gallery=False, detail=False):
    media = project["media"]
    src = prefix + media["src"]
    alt = media.get("gallery_alt_" + lang) if gallery else media.get("alt_" + lang)
    alt = alt or media.get("alt_en") or project["id"]
    if media.get("video"):
        video = prefix + media["video"]
        poster = prefix + media["poster"]
        aria = media.get("aria_" + lang) or alt
        return '<video data-src="' + escape(video) + '" poster="' + escape(poster) + '" muted loop playsinline aria-label="' + escape(aria) + '"></video>'
    loading = "eager" if detail else "lazy"
    return '<img src="' + escape(src) + '" alt="' + escape(alt) + '" loading="' + loading + '" decoding="async">'


def render_project_card(project, papers, lang, prefix=""):
    papers = paper_lookup(papers)
    venue = project_venue(project, papers)
    title = project_title(project, papers)
    authors = project_authors(project, papers)
    topics = project_topics(project, papers)
    url = detail_url(project, lang, prefix)
    meta = '<span class="venue-tag ' + venue["class"] + '">' + escape(venue["label"]) + '</span>'
    meta += "".join('<span class="proj-topic">' + escape(topic_label(slug, lang)) + '</span>' for slug in topics)
    links = " ".join('<a href="' + escape(link["href"]) + '">' + escape(link["label"]) + '</a>' for link in project_links(project, lang, prefix))
    return (
        '<li class="proj-card">\n'
        '  <a class="proj-media" href="' + escape(url) + '" title="' + escape(plain(title)) + '">\n'
        '    ' + render_media(project, lang, prefix=prefix) + '\n'
        '  </a>\n'
        '  <div class="proj-body">\n'
        '    <div class="proj-meta">' + meta + '</div>\n'
        '    <h3 class="proj-title">' + title + '</h3>\n'
        '    <p class="proj-authors">' + authors + '</p>\n'
        '    <p class="proj-desc">' + escape(project["description_" + lang]) + '</p>\n'
        '    <p class="proj-links">' + links + '</p>\n'
        '  </div>\n'
        '</li>'
    )


def gallery_name(project, papers):
    if project.get("short_name"):
        return project["short_name"]
    title = plain(project_title(project, papers))
    return title.split(":", 1)[0] if ":" in title else title


def render_gallery_card(project, papers, lang):
    venue = project_venue(project, papers)
    topics = project_topics(project, papers)
    topic = topic_label(topics[0], lang) if topics else ""
    return (
        '<a class="gal-card" href="' + escape(detail_url(project, lang)) + '">\n'
        '  <span class="gal-media">' + render_media(project, lang, prefix="./", gallery=True) + '</span>\n'
        '  <span class="gal-body"><span class="gal-name">' + escape(gallery_name(project, papers)) + '</span>'
        '<span class="gal-meta"><span class="venue-tag ' + venue["class"] + '">' + escape(venue["label"]) + '</span>'
        '<span class="gal-topic">' + escape(topic) + '</span></span></span>\n'
        '</a>'
    )


def replace_inner(source, opening_pattern, closing, content):
    match = re.search(opening_pattern, source, re.S)
    if not match:
        raise ValueError("Block not found: " + opening_pattern)
    start = match.end()
    end = source.find(closing, start)
    if end < 0:
        raise ValueError("Closing block not found: " + closing)
    return source[:start] + "\n" + content + "\n" + source[end:]


def render_paper(paper, value=None):
    raw = paper["li_html"]
    opening = re.match(r"<li\b[^>]*>", raw, re.S | re.I)
    if not opening:
        raise ValueError("Invalid paper li for " + paper["id"])
    start = opening.group(0)
    inner = raw[opening.end():raw.rfind("</li>")]
    if value is not None:
        start = re.sub(r'\s+value="\d+"', "", start)
        start = start[:-1] + ' value="' + str(value) + '">'
    return start + inner + "</li>"


def render_pubs(source, lang, papers):
    by_section = defaultdict(list)
    for paper in papers:
        by_section[paper["section"]].append(paper)
    if by_section["tech"]:
        by_section["tech"].sort(key=lambda p: tuple(int(x) for x in (paper_index(p)["arxiv"] or "0000.0000").split(".")), reverse=True)
    for section, heading in SECTION_HEADING.items():
        entries = by_section[section]
        if section == "major":
            groups = []
            for paper in entries:
                if not groups or groups[-1][0] != paper["year_group"]:
                    groups.append((paper["year_group"], []))
                groups[-1][1].append(paper)
            content = []
            number = 1
            for group, group_papers in groups:
                content.append('<li class="year-heading">Year ' + group + '</li>')
                for i, paper in enumerate(group_papers):
                    content.append(render_paper(paper, number if i == 0 else None))
                number += len(group_papers)
            content = "\n\n".join(content)
        else:
            content = "\n\n".join(render_paper(paper) for paper in entries)
        section_match = re.search(r'<section\b[^>]*aria-labelledby="' + heading + r'"[^>]*>.*?</section>', source, re.S)
        if not section_match:
            raise ValueError("Publication section not found: " + heading)
        block = section_match.group(0)
        updated = replace_inner(block, r"<ol\b[^>]*>", "</ol>", content)
        source = source[:section_match.start()] + updated + source[section_match.end():]
    return source


def render_home(source, lang, papers, projects):
    cards = "\n".join(render_gallery_card(project, papers, lang) for project in projects)
    source = replace_inner(source, r'<div class="gal-track"[^>]*>', "</div>", cards)
    return re.sub(r"<!-- DIRECTIONS:START -->.*?<!-- DIRECTIONS:END -->\s*", "", source, flags=re.S)


def sync_assets(source, config):
    source = re.sub(r"(res/xwcv\.css\?v=)[0-9-]+", r"\g<1>" + config["css_version"], source)
    source = re.sub(r"(res/site\.js\?v=)[0-9-]+", r"\g<1>" + config["js_version"], source)
    return source


def build_outputs():
    papers = load("papers.json")
    projects = load("projects.json")
    config = load("site.json")
    paper_map = {paper["id"]: paper for paper in papers}
    outputs = {}
    for lang, suffix in (("en", ""), ("cn", "_cn")):
        outputs["pubs" + suffix + ".html"] = sync_assets(render_pubs(read("pubs" + suffix + ".html"), lang, papers), config)
        grid = "\n".join(render_project_card(project, paper_map, lang) for project in projects)
        outputs["projs" + suffix + ".html"] = sync_assets(replace_inner(read("projs" + suffix + ".html"), r'<ol class="proj-grid"[^>]*>', "</ol>", grid), config)
        outputs["index" + suffix + ".html"] = sync_assets(render_home(read("index" + suffix + ".html"), lang, papers, projects), config)
    from pagegen import sync_chrome_and_sitemap
    return sync_chrome_and_sitemap(outputs, config, papers, projects)


def write_outputs(outputs, check=False):
    changed = []
    for path, content in outputs.items():
        target = ROOT / path
        if check:
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                changed.append(path)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                target.write_text(content, encoding="utf-8")
                changed.append(path)
    return changed
