#!/usr/bin/env python3
import datetime
import re
import xml.etree.ElementTree as ET
from sitegen import ROOT, escape, gallery_name, read, sync_assets

BASE_URL = "https://xwcv.github.io/"
NAV_ITEMS = {
    "en": [("home", "HOME", "index.html"), ("pubs", "PUBLICATIONS", "pubs.html"), ("projects", "PROJECTS", "projs.html"), ("group", "GROUP", "group.html")],
    "cn": [("home", "主  页", "index_cn.html"), ("pubs", "论  文", "pubs_cn.html"), ("projects", "项  目", "projs_cn.html"), ("group", "团  队", "group_cn.html")],
}
THEME_ICONS = '<svg class="icon-moon" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg><svg class="icon-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>'


def language(path):
    return "cn" if path.endswith("_cn.html") else "en"


def counterpart(path):
    if path == "index.html":
        return "index_cn.html"
    if path == "index_cn.html":
        return "index.html"
    if path.endswith("_cn.html"):
        return path[:-8] + ".html"
    if path.endswith(".html"):
        return path[:-5] + "_cn.html"
    raise ValueError(path)


def canonical(path):
    return BASE_URL if path == "index.html" else BASE_URL + path


def page_context(path):
    lang = language(path)
    prefix = "/" if path == "404.html" else ""
    active = "home" if path.startswith("index") else "pubs" if path.startswith("pubs") else "projects" if path.startswith("projs") else "group" if path.startswith("group") else None
    other = counterpart(path) if path != "404.html" else "index_cn.html"
    return lang, prefix, active, prefix + other


def render_nav(lang, prefix, active, lang_href):
    items = []
    for key, label, href in NAV_ITEMS[lang]:
        current = key == active
        attrs = ' class="active" aria-current="page"' if current else ""
        items.append('<a' + attrs + ' href="' + prefix + href + '">' + label + '</a>')
    other_lang = "en" if lang == "cn" else "zh"
    label = "English" if lang == "cn" else "中文"
    items.append('<a class="lang" data-lang-switch href="' + lang_href + '" lang="' + other_lang + '">' + label + '</a>')
    toggle_label = "切换深色模式" if lang == "cn" else "Toggle dark mode"
    items.append('<button class="theme-toggle" type="button" aria-label="' + toggle_label + '" title="' + toggle_label + '">' + THEME_ICONS + '</button>')
    aria = "主导航" if lang == "cn" else "Main navigation"
    return '<nav class="topnav" aria-label="' + aria + '"><div class="container">' + "".join(items) + '</div></nav>'


def update_project_meta(source, papers, projects, lang):
    names = [gallery_name(project, papers) for project in projects]
    joined = ", ".join(names[:-1]) + ", and " + names[-1]
    if lang == "cn":
        description = "王兴刚教授团队的开源科研项目，涵盖计算机视觉、机器人、自动驾驶与人工智能，包括" + "、".join(names) + "。"
        keywords = "王兴刚 项目, 计算机视觉, 机器人, 自动驾驶, " + ", ".join(names)
    else:
        description = "Open-source research projects by Professor Xinggang Wang's group in computer vision, robotics, autonomous driving, and artificial intelligence, including " + joined + "."
        keywords = "Xinggang Wang projects, computer vision projects, robotics, autonomous driving, world action model, vision-language model, " + ", ".join(names)
    source = re.sub(r'(<meta name="description" content=")[^"]*(")', r'\g<1>' + escape(description) + r'\g<2>', source, count=1)
    source = re.sub(r'(<meta name="keywords" content=")[^"]*(")', r'\g<1>' + escape(keywords) + r'\g<2>', source, count=1)
    return source


def build_sitemap(outputs):
    lastmods = {}
    if (ROOT / "sitemap.xml").exists():
        tree = ET.parse(ROOT / "sitemap.xml")
        ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for node in tree.getroot().findall("s:url", ns):
            loc = node.findtext("s:loc", namespaces=ns)
            lastmods[loc] = node.findtext("s:lastmod", namespaces=ns) or datetime.date.today().isoformat()
    paths = set(outputs) | {"group.html", "group_cn.html"}
    urls = []
    for path in sorted(paths):
        if not path.endswith(".html") or path == "404.html":
            continue
        urls.append((canonical(path), path))
    today = datetime.date.today().isoformat()
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for loc, path in urls:
        frequency = "weekly" if path.startswith(("pubs", "projs")) else "monthly"
        priority = "1.0" if path == "index.html" else "0.9" if path == "index_cn.html" else "0.8" if frequency == "weekly" else "0.7"
        lines += ['  <url>', '    <loc>' + loc + '</loc>', '    <lastmod>' + lastmods.get(loc, today) + '</lastmod>', '    <changefreq>' + frequency + '</changefreq>', '    <priority>' + priority + '</priority>', '  </url>']
    lines.append('</urlset>')
    return "\n".join(lines) + "\n"


def sync_chrome_and_sitemap(outputs, config, papers, projects):
    outputs.setdefault("group.html", read("group.html"))
    outputs.setdefault("group_cn.html", read("group_cn.html"))
    outputs.setdefault("404.html", read("404.html"))
    for path in list(outputs):
        if not path.endswith(".html"):
            continue
        lang, prefix, active, lang_href = page_context(path)
        source = sync_assets(outputs[path], config)
        source = re.sub(r'<nav class="topnav".*?</nav>', render_nav(lang, prefix, active, lang_href), source, count=1, flags=re.S)
        if path in ("projs.html", "projs_cn.html"):
            source = update_project_meta(source, papers, projects, lang)
        outputs[path] = source
    outputs["sitemap.xml"] = build_sitemap(outputs)
    return outputs
