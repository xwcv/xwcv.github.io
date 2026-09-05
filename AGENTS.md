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

- `index.htm` / `index_cn.htm` — homepage (EN / CN), with an influential-papers
  list carrying hard-coded "N+ citations" / "N stars" badges (no-JS fallback,
  refreshed at runtime from the JSON files below).
- `pubs.htm` — full publication list, grouped by `Year NNNN` paragraphs inside
  one `<ol>`, newest year first.
- `group.htm` / `group_cn.htm` — group members.
- `res/` — images, `site.js`, `stars.json`, `scholar.json` (both generated).
- `bib/` — BibTeX snippets (`*.txt`) linked from some `pubs.htm` entries.
- `pubs/` — self-hosted PDFs.
- `tools/` — Python scripts run by CI (see below).
- `.github/workflows/` — `stars.yml`, `scholar.yml`.

## Adding a publication to pubs.htm

Insert a new `<li><p> … </p></li>` under the matching `Year NNNN` heading
(create the heading if missing). Follow the existing entry style:

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

## Dynamic data (stars & citations)

- `res/stars.json` — GitHub star counts for every repo linked from
  `pubs.htm`. Written by `tools/update_stars.py`, run weekly (Mon 04:23 UTC)
  by `.github/workflows/stars.yml`, also via `workflow_dispatch`.
  **A newly added repo shows no star badge until the next run.** To refresh
  immediately: `gh workflow run stars.yml` (the workflow commits and pushes
  itself). Running the script locally requires `GITHUB_TOKEN`
  (unauthenticated API quota is too small and the script refuses to run).
- `res/scholar.json` — citation counts keyed by Google Scholar cluster id.
  Written by `tools/update_scholar.py`, run Mon & Thu by `scholar.yml` via
  SerpAPI (needs the `SERPAPI_API_KEY` secret).
- `tools/update_badges.py` — syncs the hard-coded badge numbers on
  `index.htm` / `index_cn.htm` with the two JSON files (run by both
  workflows after updating the JSON).
- The stars workflow also refreshes the footer "Last updated" dates and
  `sitemap.xml` lastmod whenever `res/stars.json` changes — don't hand-edit
  those.

## Homepage badges

"N+ citations" links on the homepages are keyed by the cluster id in their
`citation_for_view=...:<CLUSTER>` href — keep that href intact so the number
can be refreshed. Only papers in the Scholar profile's top-20 list get live
counts.
