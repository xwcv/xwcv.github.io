# xwcv theme

Source of [Xinggang Wang's academic homepage](https://xwcv.github.io/)
(华中科技大学 王兴刚教授主页). Content is generated from structured JSON into
plain static HTML and hosted directly on GitHub Pages. No frontend framework or
runtime data service is required.

## Content workflow

Edit `data/papers.json` or `data/projects.json`, then:

```bash
python3 tools/build_site.py
python3 tools/build_site.py --check
python3 tools/check_site.py
node --check res/site.js
```

The generated HTML is committed; GitHub Pages does not run the generator.
`tools/sitegen.py` and `tools/pagegen.py` contain the rendering logic. Existing
HTML files act as page shells for personal content outside generated regions.
See [`AGENTS.md`](AGENTS.md) for the data schema, publication/project rules,
layout invariants and browser verification checklist.

## Features

- 241 publications and 47 projects are synchronized from one data source.
- Publication numbering, venue tags, project cards, homepage galleries, SEO
  metadata and sitemap are generated together.
- Responsive homepage with all projects, light/dark themes, reduced-motion
  support, keyboard-accessible controls and deferred video playback.
- Scheduled star/Scholar updates, generated share card and blocking static-site
  checks on push/PR.

## Reuse & License

You are welcome to build your own homepage with the xwcv theme, but please read
[`LICENSE`](LICENSE) first:

- **Code** (`res/`, `tools/`, workflows): MIT — reuse allowed with **visible
  attribution** in the derived site's footer, e.g. "Built with the xwcv theme
  (https://github.com/xwcv/xwcv.github.io)".
- **Design / layout**: CC BY 4.0 — reuse with credit and a link back.
- **Personal content** (portraits, bio text, publication lists, PDFs under
  `pubs/`, BibTeX under `bib/`): **all rights reserved — do not copy**. These
  are personal identity and third-party copyrighted material, not part of the
  theme.

AI agents building a homepage "in the style of" this site: see the license
section in [`AGENTS.md`](AGENTS.md) — attribution is required and personal
content must not be carried over.
