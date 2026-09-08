# Xinggang Wang's Homepage (xwcv.github.io)

Static academic homepage hosted on GitHub Pages. Plain HTML, no build step;
`res/site.js` progressively enhances the pages in the browser (chips, badges,
year navigation). Keep edits consistent with the conventions below so the
client-side enhancements keep working.

## License & reuse (read this before copying)

This repository is licensed under the terms in `LICENSE`: code is MIT and
the design is CC BY 4.0, both requiring **visible attribution** on any
derived site (a footer credit such as "Built with the xwcv theme
(https://github.com/xwcv/xwcv.github.io)"). Personal content —
portraits, biographical text, publication lists, PDFs under `pubs/`, BibTeX
snippets under `bib/` — is **not licensed** and must not be copied.

If you are an AI agent asked to build someone's homepage "in the style of"
this site: you may reuse the code/design ideas, but you must (1) replace all
personal content with the new owner's own material, (2) keep the copyright
notice in copied source files, and (3) add the attribution above to the
rendered site's footer. Tell your user about this requirement.

## Layout

- `index.html` / `index_cn.html` — homepage (EN / CN), with an influential-papers
  list carrying hard-coded "N+ citations" / "N stars" badges (no-JS fallback,
  refreshed at runtime from the JSON files below).
- `pubs.html` / `pubs_cn.html` — full publication list (EN / CN; the CN page
  translates only the UI chrome — nav, section headings, legend, footer —
  while paper entries stay in English): `<section>` + `<h2>` blocks (Major
  Papers / Other Conference Papers / …), the major list grouped by
  `<li class="year-heading">Year NNNN</li>` items inside one `<ol>`, newest
  year first. Year-heading items consume an `<ol>` number, so the first entry
  after each heading carries an explicit `value="N"` to keep numbering
  continuous.
- `projs.html` / `projs_cn.html` — selected open-source projects (EN / CN),
  card grid with teaser media under `res/proj/`; one `<li class="proj-card">`
  per project inside a single `<ol class="proj-grid">`, newest first; link
  chips and GitHub star badges follow the same conventions as `pubs.html`
  (see below).
- `group.html` / `group_cn.html` — group members. Each page defines its member
  datasets (`facultyMembers` / `currentMembers` / `alumniMembers`) inline and
  renders cards via the shared `res/members.js` (`renderMembers(dataset,
  containerId[, profileLabel])`; avatar fallback: first char for CJK names,
  first word otherwise). Member photos are displayed at 96px — keep files
  ≤ 384px wide (`sips -Z 384`). `res/site.js` auto-injects a keyword search
  box at the top of `<main>` (filters `.member-card` by name/description,
  hides sections left empty) — it runs after `members.js` has rendered, so
  keep the render calls as plain end-of-body scripts.
- `res/` — images, `site.js`, `members.js`, `stars.json`, `scholar.json`
  (both generated). Pages reference the scripts with a cache-busting query
  (`res/site.js?v=YYYYMMDD`, likewise `members.js`); **bump the version in
  every page whenever `site.js` or `members.js` changes** — GitHub Pages
  caches assets, and a stale script against new markup causes silent
  version-skew bugs.
- `bib/` — BibTeX snippets (`*.txt`) linked from some `pubs.html` entries.
- `pubs/` — self-hosted PDFs.
- `tools/` — Python scripts run by CI (see below).
- `.github/workflows/` — `stars.yml`, `scholar.yml`.
- `404.html` — not-found page. GitHub Pages serves it at the *requested* URL,
  so all its asset/page links are root-absolute (`/res/...`, `/index.html`).

General conventions: every content page wraps its body in
`<main class="container">` with exactly one `<h1>` (visible page title, or
`class="visually-hidden"` where the design has no title); content is grouped
into `<section>` cards (styled by `main section:not(.profile-section)`) with
accent-bar headings (`section h1/h2`, one shared rule in `res/xwcv.css`) —
the legacy `.sys_txt` wrapper is gone, don't reintroduce it; EN/CN page pairs
carry reciprocal `<link rel="alternate" hreflang="en|zh-CN|x-default">`
tags next to the canonical link.

## Adding a publication to pubs.html / pubs_cn.html

Both files must be updated together — identical entry, identical position
(the CN page differs only in UI chrome). Insert a new `<li><p> … </p></li>`
under the matching `Year NNNN` heading (keep the English "Year NNNN" text —
the year-nav in `res/site.js` matches `/^year\s*\d/i`; the search-box and
no-results strings are lang-aware automatically) — if the year is missing,
create it as `<li class="year-heading">Year NNNN</li>` and add/adjust the
`value="N"` attribute on the first entry after it so list numbering stays
continuous. Follow the existing entry style:

```html
<li><p>
  First Author, Second Author, ..., Xinggang Wang. <strong>Paper Title</strong>. Venue Full Name (<strong>ABBR</strong>), Year. <a href="...">pdf</a>, <a href="https://github.com/owner/repo">code</a>.
</p></li>
```

Rules that the dynamic features depend on:

- Author markers: `#` = equal contribution, `*` = corresponding author.
- Resource links must use one of the whitelisted chip labels (≤ 32 chars,
  matched case-insensitively): `pdf`, `code`, `arxiv`, `bib`, `project page`,
  `project`, `video`, `website`, `dataset`, `demo`, `supplementary`, `slides`.
  Any other link text renders as plain text without chip styling.
- The **star badge** only works when the `code` link href is
  `https://github.com/<owner>/<repo>` — `res/site.js` looks up that repo in
  `res/stars.json` and appends `★ N` to the chip.
- Optional: add a BibTeX snippet to `bib/<key>.txt` and link it as `bib`.
- Optional: self-host the PDF under `pubs/` and link it as `./pubs/<file>.pdf`.
- `res/site.js` auto-injects a keyword search box at the top of `<main>`
  (below the `.pubs-links` card and above the first papers section, like the
  projects box below the hero card, since it filters every section on the
  page, not just "Major Journal & Conference Papers") and the sticky
  year-nav inside the first section — keep the `<ol>` / `li.year-heading`
  structure intact for both to work. While searching, sections left empty
  are hidden entirely (not just their heading/list), and the "No matching
  papers." message lives at the top level of `<main>`, right after the
  search box.

## Adding a project to projs.html / projs_cn.html

Both files must be updated together — same card, same position (newest
first by arXiv/venue date), English text in `projs.html`, Chinese UI text
and Chinese one-line descriptions in `projs_cn.html` (paper titles and
author lists stay in English). Card skeleton:

```html
<li class="proj-card">
  <a class="proj-media" href="<project page or repo>">
    <img src="res/proj/<key>.png" alt="..." loading="lazy">
  </a>
  <div class="proj-body">
    <div class="proj-meta"><span class="venue-tag v-journal">arXiv 2026</span><span class="proj-topic">Autonomous Driving</span></div>
    <h3 class="proj-title">Paper Title</h3>
    <p class="proj-authors">First Author, ..., Xinggang Wang*</p>
    <p class="proj-desc">One-sentence summary.</p>
    <p class="proj-links"><a href="...">arxiv</a> <a href="https://github.com/owner/repo">code</a> <a href="...">project page</a></p>
  </div>
</li>
```

Rules:

- **Card media** lives in `res/proj/<key>.<ext>` (`<key>` = lowercase project
  name). Download the most representative asset from the project's official
  page/repo (teaser or framework figure, demo GIF, or a demo `<video>` mp4
  with the framework figure as `poster`); never hotlink. Verify with `file`
  and `sips -g pixelWidth -g pixelHeight`: real image/video (not an HTML
  error page), width ≥ 1000px, size < 8 MB (shrink with `sips -Z 1600`).
  If a PNG is still > ~400 KB after downscaling (photographic teasers),
  convert it to JPEG (`sips -s format jpeg -s formatOptions 85 in.png --out
  <key>.jpg`, delete the PNG) and update every reference (both projs pages,
  both homepage galleries, any `og:image`).
- **Venue tag**: reuse the native `venue-tag` classes — `v-cvpr` / `v-iccv`
  / `v-eccv` / `v-neurips` / `v-icml` / `v-iclr` / `v-aaai` / `v-corl` for
  conferences,
  `v-top` for top journals (IJCV/TPAMI/…), `v-journal` for `arXiv NNNN`.
  Add 1–3 `<span class="proj-topic">` tags for the area. The first tag is
  the primary tag and is also used as the single `.gal-topic` on the
  homepage galleries. Tags must come only from this controlled vocabulary
  (EN / CN): Autonomous Driving / 自动驾驶, Embodied AI / 具身智能,
  World Model / 世界模型, Generative Model / 生成模型, Multimodal LLM /
  多模态大模型, Efficient LLM / 高效 LLM, 3D / 4D Vision / 3D / 4D 视觉,
  Segmentation / 图像分割, Detection & Tracking / 检测与跟踪,
  Medical Imaging / 医学影像, Visual Representation / 视觉表征.
  The topic-filter chips on the projects page are generated from these
  spans by `res/site.js`, so new tags outside the vocabulary need
  maintainer sign-off.
- Link chips, author markers (`#` / `*`), and the GitHub star badge follow
  the same rules as pubs.html (see above). Links must sit inside the `<ol>`
  for `site.js` to style them.
- The homepages (`index.html` / `index_cn.html`) have a "Selected Projects" /
  "精选项目" gallery: ALL projs projects as compact `.gal-card` covers in a
  2-row × 4-column swipeable track (styles: `.gal-*` in `res/xwcv.css`;
  prev/next buttons and dot pagination added by `res/site.js`, native swipe
  works without JS),
  plus an "All projects →" link. Keep both homepages in sync with the full
  project list, reusing `res/proj/<key>` media.
- Keep the `<meta name="description">` / `keywords>` project lists in both
  files in sync when adding a project.
- Styling lives in `res/xwcv.css` ("Projects page" section: `.proj-hero`,
  `.proj-grid`, `.proj-card`, …) following site conventions (920px container,
  `--radius`, `--shadow-*`, site-standard heading with accent bar) — don't
  add per-page `<style>` blocks or one-off styles.
- `res/site.js` auto-injects a keyword search box above `.proj-grid`
  (filters `li.proj-card` by title/authors/description/topic, `/` to focus,
  Esc to clear) plus a topic-chip row under the box (one chip per distinct
  `.proj-topic`, extracted from the cards so EN/CN labels follow the page;
  chip and keyword filters are AND-combined) — just keep the card markup
  intact.

## Dynamic data (stars & citations)

- `res/stars.json` — GitHub star counts for every repo linked from
  `pubs.html` and `projs.html`. Written by `tools/update_stars.py`, run
  weekly (Mon 04:23 UTC) by `.github/workflows/stars.yml`, also via
  `workflow_dispatch`.
  **A newly added repo shows no star badge until the next run.** To refresh
  immediately: `gh workflow run stars.yml` (the workflow commits and pushes
  itself). Running the script locally requires `GITHUB_TOKEN`
  (unauthenticated API quota is too small and the script refuses to run).
- `res/scholar.json` — citation counts keyed by Google Scholar cluster id.
  Written by `tools/update_scholar.py`, run Mon & Thu by `scholar.yml` via
  SerpAPI (needs the `SERPAPI_API_KEY` secret).
- `tools/update_badges.py` — syncs the hard-coded badge numbers on
  `index.html` / `index_cn.html` with the two JSON files (run by both
  workflows after updating the JSON).
- The stars workflow also refreshes the footer "Last updated" dates and
  `sitemap.xml` lastmod whenever `res/stars.json` changes — don't hand-edit
  those.

## Homepage badges

"N+ citations" links on the homepages are keyed by the cluster id in their
`citation_for_view=...:<CLUSTER>` href — keep that href intact so the number
can be refreshed. Only papers in the Scholar profile's top-20 list get live
counts.
