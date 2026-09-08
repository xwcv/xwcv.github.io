/* Progressive UI enhancements for the xwcv theme (xwcv.github.io).
   Copyright (c) Xinggang Wang — licensed under the terms in LICENSE (MIT with
   attribution; derived sites must credit the xwcv theme visibly, linking to
   https://github.com/xwcv/xwcv.github.io).
   Every feature degrades gracefully: without JS the pages render identically. */
document.addEventListener('DOMContentLoaded', function () {
  'use strict';

  /* 0. Theme toggle: persists choice to localStorage (read by the inline head script) */
  var root = document.documentElement;
  document.querySelectorAll('.theme-toggle').forEach(function (btn) {
    btn.setAttribute('aria-pressed', root.getAttribute('data-theme') === 'dark' ? 'true' : 'false');
    btn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      btn.setAttribute('aria-pressed', next === 'dark' ? 'true' : 'false');
      try { localStorage.setItem('theme', next); } catch (e) {}
    });
  });

  /* 1. Citation badges: "4000+ citations" -> pill; "3.9k stars" -> star pill
        (the star glyph comes from CSS, the number is refreshed in step 5) */
  document.querySelectorAll('a').forEach(function (a) {
    var t = a.textContent.trim();
    if (/^[\d,]+\+?\s*citations?$/i.test(t)) {
      a.classList.add('cite-badge');
      a.innerHTML = a.innerHTML.replace(/^([\d,]+\+?)/, '<strong>$1</strong>');
    } else if (/^[\d.,]+k?\s+stars?$/i.test(t)) {
      a.classList.add('star-badge');
    }
  });

  /* 2. Resource chips inside ordered lists: pdf / code / arXiv ... -> chip */
  document.querySelectorAll('ol a').forEach(function (a) {
    var t = a.textContent.trim();
    if (t.length <= 32 && /^(pdf|code|arxiv|bib|project page|project|video|website|dataset|demo|supplementary|slides)$/i.test(t)) {
      a.classList.add('res-chip');
    }
  });

  /* Wrap every occurrence of q in <mark> inside the element's text nodes
     (used by the publication and project search boxes). */
  var markMatches = function (el, q) {
    Array.prototype.forEach.call(el.querySelectorAll('mark'), function (m) {
      m.replaceWith(m.textContent);
    });
    el.normalize();
    if (!q) return;
    var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    var nodes = [];
    while (walker.nextNode()) {
      var n = walker.currentNode;
      // skip chip links: painting marks inside the blue res-chips looks wrong
      if (n.parentElement && n.parentElement.closest('.res-chip')) continue;
      nodes.push(n);
    }
    nodes.forEach(function (node) {
      var text = node.nodeValue;
      var lower = text.toLowerCase();
      var idx = lower.indexOf(q);
      if (idx === -1) return;
      var frag = document.createDocumentFragment();
      var pos = 0;
      while (idx !== -1) {
        frag.appendChild(document.createTextNode(text.slice(pos, idx)));
        var mark = document.createElement('mark');
        mark.textContent = text.slice(idx, idx + q.length);
        frag.appendChild(mark);
        pos = idx + q.length;
        idx = lower.indexOf(q, pos);
      }
      frag.appendChild(document.createTextNode(text.slice(pos)));
      node.parentNode.replaceChild(frag, node);
    });
  };

  /* 3. Year quick navigation on the publications page.
        Turns "Year 2026" paragraphs into anchor targets and builds a sticky jump bar. */
  var yearPs = Array.prototype.filter.call(
    document.querySelectorAll('ol > li.year-heading'),
    function (p) { return /^year\s*\d/i.test(p.textContent.trim()); }
  );
  if (yearPs.length > 3) {
    var pubsZh = /^zh/i.test(document.documentElement.lang || '');
    var nav = document.createElement('nav');
    nav.className = 'year-nav';
    nav.setAttribute('aria-label', pubsZh ? '跳转到年份' : 'Jump to year');
    yearPs.forEach(function (p) {
      var label = p.textContent.trim().replace(/^year\s*/i, '');
      var id = 'year-' + label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
      p.id = id;
      var a = document.createElement('a');
      a.href = '#' + id;
      a.textContent = label;
      nav.appendChild(a);
    });
    var firstOl = yearPs[0].parentNode;
    firstOl.parentNode.insertBefore(nav, firstOl);

    // scrollspy: highlight the pill of the year currently in view
    if ('IntersectionObserver' in window) {
      var navLinks = {};
      Array.prototype.forEach.call(nav.querySelectorAll('a'), function (a) {
        navLinks[a.getAttribute('href').slice(1)] = a;
      });
      var spy = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (!en.isIntersecting) return;
          for (var id in navLinks) navLinks[id].classList.remove('active');
          var link = navLinks[en.target.id];
          if (link) link.classList.add('active');
        });
      }, { rootMargin: '-80px 0px -70% 0px' });
      yearPs.forEach(function (p) { spy.observe(p); });
    }

    /* 3b. Paper search: instant keyword filter over every paper entry
           (matches title / authors / venue text), with a match counter.
           While searching, the year headers and any
           section left empty ("Other Conference Papers", ...) are hidden.
           "/" focuses the box, Esc clears it. */
    var items = Array.prototype.slice.call(document.querySelectorAll('ol li:not(.year-heading)'));
    // map each list to its section header, e.g. <h2>Book Chapters</h2><ol>
    var lists = Array.prototype.map.call(document.querySelectorAll('ol'), function (ol) {
      var prev = ol.previousElementSibling;
      while (prev && prev.nodeName !== 'H2') prev = prev.previousElementSibling;
      return { ol: ol, header: prev };
    });
    var box = document.createElement('div');
    box.className = 'pubs-search';
    box.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>';
    var input = document.createElement('input');
    input.type = 'search';
    input.placeholder = pubsZh ? '搜索论文：标题、作者、期刊/会议 ...  ( / )' : 'Search papers by title, author, venue ...  ( / )';
    input.setAttribute('aria-label', pubsZh ? '搜索论文' : 'Search papers');
    var count = document.createElement('span');
    count.className = 'pubs-search-count';
    box.appendChild(input);
    box.appendChild(count);
    // top-level placement (like the projects page): the search scope is the
    // whole page, not just the first section, so the box lives in <main>
    // between the .pubs-links card and the first papers section, matching
    // the projects page where the box sits below the hero card
    var firstSection = firstOl.closest('section') || firstOl.parentNode;
    firstSection.parentNode.insertBefore(box, firstSection);
    var empty = document.createElement('p');
    empty.className = 'pubs-no-results';
    empty.textContent = pubsZh ? '没有匹配的论文。' : 'No matching papers.';
    empty.style.display = 'none';
    // top level too, right after the search box: hiding the first section
    // while searching must not hide the no-results message with it
    firstSection.parentNode.insertBefore(empty, firstSection);
    input.addEventListener('input', function () {
      var q = input.value.trim().toLowerCase();
      var shown = 0;
      items.forEach(function (li) {
        var hit = !q || li.textContent.toLowerCase().indexOf(q) !== -1;
        li.style.display = hit ? '' : 'none';
        markMatches(li, hit ? q : '');
        if (hit) shown++;
      });
      yearPs.forEach(function (p) {
        p.style.display = q ? 'none' : '';
      });
      nav.style.display = q ? 'none' : '';
      lists.forEach(function (s) {
        var any = Array.prototype.some.call(s.ol.querySelectorAll('li:not(.year-heading)'), function (li) {
          return li.style.display !== 'none';
        });
        var hide = q && !any;
        var sec = s.ol.closest('section');
        if (sec) {
          // hide the whole section card, not just its heading and list
          sec.style.display = hide ? 'none' : '';
          // a still-hidden reveal-section can enter the viewport when the
          // sections above it collapse; reveal it immediately instead of
          // relying on the IntersectionObserver to notice the layout shift
          if (!hide && q) sec.classList.add('in');
        } else {
          if (s.header) s.header.style.display = hide ? 'none' : '';
          if (s.ol !== firstOl) s.ol.style.display = hide ? 'none' : '';
        }
      });
      empty.style.display = q && !shown ? '' : 'none';
      count.textContent = q ? shown + ' / ' + items.length : '';
    });
    document.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return;
      var tag = document.activeElement && document.activeElement.tagName;
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
        e.preventDefault();
        input.focus();
      } else if (e.key === 'Escape' && document.activeElement === input) {
        input.value = '';
        input.dispatchEvent(new Event('input'));
        input.blur();
      }
    });
  }

  /* 3c. Project search on the projects page: instant keyword filter over the
         project cards (matches title / authors / description / topic), with a
         match counter, plus a row of topic chips under the box (extracted from
         the cards' .proj-topic, so EN/CN labels follow the page language);
         chip and keyword filters are AND-combined. "/" focuses the box,
         Esc clears it. */
  var projGrid = document.querySelector('ol.proj-grid');
  if (projGrid) {
    var cards = Array.prototype.slice.call(projGrid.querySelectorAll('li.proj-card'));
    var zh = /^zh/i.test(document.documentElement.lang || '');
    var pbox = document.createElement('div');
    pbox.className = 'pubs-search';
    pbox.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>';
    var pinput = document.createElement('input');
    pinput.type = 'search';
    pinput.placeholder = zh ? '搜索项目：名称、作者、关键词 ...  ( / )' : 'Search projects by name, author, keyword ...  ( / )';
    pinput.setAttribute('aria-label', zh ? '搜索项目' : 'Search projects');
    var pcount = document.createElement('span');
    pcount.className = 'pubs-search-count';
    pbox.appendChild(pinput);
    pbox.appendChild(pcount);
    projGrid.parentNode.insertBefore(pbox, projGrid);
    var pempty = document.createElement('p');
    pempty.className = 'pubs-no-results';
    pempty.textContent = zh ? '没有匹配的项目。' : 'No matching projects.';
    pempty.style.display = 'none';
    projGrid.parentNode.insertBefore(pempty, projGrid);

    // topic chips: one per distinct .proj-topic (a card may carry up to 3),
    // most frequent first
    var topicsOf = cards.map(function (card) {
      return Array.prototype.map.call(card.querySelectorAll('.proj-topic'), function (t) {
        return t.textContent.trim();
      });
    });
    var topics = [];
    topicsOf.forEach(function (ts) {
      ts.forEach(function (t) {
        if (!t) return;
        for (var i = 0; i < topics.length; i++) {
          if (topics[i].name === t) { topics[i].n++; return; }
        }
        topics.push({ name: t, n: 1 });
      });
    });
    topics.sort(function (a, b) { return b.n - a.n; });
    var activeTag = null;
    var chips = [];
    var apply = function () {
      var q = pinput.value.trim().toLowerCase();
      var shown = 0;
      cards.forEach(function (card, i) {
        var hit = (!activeTag || topicsOf[i].indexOf(activeTag) !== -1) &&
                  (!q || card.textContent.toLowerCase().indexOf(q) !== -1);
        card.style.display = hit ? '' : 'none';
        markMatches(card, hit ? q : '');
        if (hit) shown++;
      });
      chips.forEach(function (chip) {
        var on = chip._tag === activeTag;
        chip.classList.toggle('proj-tag-on', on);
        chip.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      pempty.style.display = (q || activeTag) && !shown ? '' : 'none';
      pcount.textContent = (q || activeTag) ? shown + ' / ' + cards.length : '';
    };
    if (topics.length > 1) {
      var tagRow = document.createElement('div');
      tagRow.className = 'proj-tags';
      tagRow.setAttribute('role', 'group');
      tagRow.setAttribute('aria-label', zh ? '按主题筛选项目' : 'Filter projects by topic');
      var mkChip = function (label, countN, value) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'proj-tag';
        b._tag = value;
        b.setAttribute('aria-pressed', 'false');
        b.textContent = label;
        if (countN) {
          var n = document.createElement('span');
          n.className = 'proj-tag-n';
          n.textContent = countN;
          b.appendChild(n);
        }
        b.addEventListener('click', function () {
          activeTag = activeTag === value ? null : value;
          apply();
        });
        tagRow.appendChild(b);
        chips.push(b);
      };
      mkChip(zh ? '全部' : 'All', 0, null);
      topics.forEach(function (t) { mkChip(t.name, t.n, t.name); });
      projGrid.parentNode.insertBefore(tagRow, pempty);
      chips[0].classList.add('proj-tag-on');
      chips[0].setAttribute('aria-pressed', 'true');
    }
    pinput.addEventListener('input', apply);
    document.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return;
      var tag = document.activeElement && document.activeElement.tagName;
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
        e.preventDefault();
        pinput.focus();
      } else if (e.key === 'Escape' && document.activeElement === pinput) {
        pinput.value = '';
        pinput.dispatchEvent(new Event('input'));
        pinput.blur();
      }
    });
  }

  /* 3d. Member search on the group pages: instant keyword filter over the
         member cards (matches name / description text), with a match counter.
         Sections left empty while searching are hidden. Same placement as the
         pubs/projs boxes: top of <main>, above the first section.
         "/" focuses the box, Esc clears it. Runs on DOMContentLoaded, after
         members.js has rendered the cards. */
  var memberCards = Array.prototype.slice.call(document.querySelectorAll('.member-card'));
  if (memberCards.length) {
    var mzh = /^zh/i.test(document.documentElement.lang || '');
    var mSections = [];
    memberCards.forEach(function (card) {
      var sec = card.closest('section');
      if (sec && mSections.indexOf(sec) === -1) mSections.push(sec);
    });
    var mbox = document.createElement('div');
    mbox.className = 'pubs-search';
    mbox.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>';
    var minput = document.createElement('input');
    minput.type = 'search';
    minput.placeholder = mzh ? '搜索成员：姓名、研究方向 ...  ( / )' : 'Search members by name, research interest ...  ( / )';
    minput.setAttribute('aria-label', mzh ? '搜索成员' : 'Search members');
    var mcount = document.createElement('span');
    mcount.className = 'pubs-search-count';
    mbox.appendChild(minput);
    mbox.appendChild(mcount);
    var mFirstSection = mSections[0];
    mFirstSection.parentNode.insertBefore(mbox, mFirstSection);
    var mempty = document.createElement('p');
    mempty.className = 'pubs-no-results';
    mempty.textContent = mzh ? '没有匹配的成员。' : 'No matching members.';
    mempty.style.display = 'none';
    mFirstSection.parentNode.insertBefore(mempty, mFirstSection);
    minput.addEventListener('input', function () {
      var q = minput.value.trim().toLowerCase();
      var shown = 0;
      memberCards.forEach(function (card) {
        var hit = !q || card.textContent.toLowerCase().indexOf(q) !== -1;
        card.style.display = hit ? '' : 'none';
        markMatches(card, hit ? q : '');
        if (hit) shown++;
      });
      mSections.forEach(function (sec) {
        var any = Array.prototype.some.call(sec.querySelectorAll('.member-card'), function (card) {
          return card.style.display !== 'none';
        });
        var hide = q && !any;
        sec.style.display = hide ? 'none' : '';
        // same reveal safeguard as the pubs search: a section that jumps
        // into the viewport during filtering must not stay at opacity 0
        if (!hide && q) sec.classList.add('in');
      });
      mempty.style.display = q && !shown ? '' : 'none';
      mcount.textContent = q ? shown + ' / ' + memberCards.length : '';
    });
    document.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return;
      var tag = document.activeElement && document.activeElement.tagName;
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
        e.preventDefault();
        minput.focus();
      } else if (e.key === 'Escape' && document.activeElement === minput) {
        minput.value = '';
        minput.dispatchEvent(new Event('input'));
        minput.blur();
      }
    });
  }

  /* 4. Google Scholar stats: refresh the hard-coded numbers from res/scholar.json
        (written by the scheduled GitHub Action). Fails silently, keeping the
        hard-coded values, when the file is missing or unreachable.
        d.papers maps each paper's citation_for_view cluster id to its citation
        count, so every "N+ citations" badge linking to a citation page gets
        refreshed too (rounded down to the nearest 100, matching the badges). */
  var gsCit = document.getElementById('gs-citations');
  var gsH = document.getElementById('gs-hindex');
  if (gsCit || gsH) {
    fetch('res/scholar.json', { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        // count-up animation for the two headline numbers
        var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        var countUp = function (el, target) {
          if (reduce || !('requestAnimationFrame' in window)) {
            el.textContent = Number(target).toLocaleString('en-US');
            return;
          }
          var t0 = null, dur = 900;
          var tick = function (t) {
            if (!t0) t0 = t;
            var p = Math.min((t - t0) / dur, 1);
            var eased = 1 - Math.pow(1 - p, 3);
            el.textContent = Math.round(target * eased).toLocaleString('en-US');
            if (p < 1) requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        };
        if (gsCit && d.citations) countUp(gsCit, d.citations);
        if (gsH && d.hindex) countUp(gsH, d.hindex);
        if (d.years) {
          // yearly-citation bar chart next to the totals
          var stats = document.querySelector('.scholar-stats');
          if (stats && !stats.querySelector('.gs-graph')) {
            var years = Object.keys(d.years).sort();
            var max = Math.max.apply(null, years.map(function (y) { return d.years[y]; }));
            var H = 30;
            var svg = '<svg class="gs-graph" width="' + (years.length * 8 - 2) + '" height="' + H
              + '" role="img" aria-label="Citations per year">';
            years.forEach(function (y, i) {
              var h = Math.max(2, Math.round(d.years[y] / max * (H - 4)));
              svg += '<rect x="' + i * 8 + '" y="' + (H - h) + '" width="6" height="' + h + '" rx="1.5">'
                + '<title>' + y + ': ' + Number(d.years[y]).toLocaleString('en-US') + ' citations</title></rect>';
            });
            stats.insertAdjacentHTML('beforeend', svg + '</svg>');
          }
        }
        if (d.papers) {
          document.querySelectorAll('a[href*="citation_for_view="]').forEach(function (a) {
            var m = /citation_for_view=[^&:]+:([\w-]+)/.exec(a.href);
            var n = m && d.papers[m[1]];
            if (n) {
              a.innerHTML = '<strong>' + (Math.floor(n / 100) * 100).toLocaleString('en-US') + '+</strong> citations';
            }
          });
        }
      })
      .catch(function () {});
  }

  /* 5. GitHub star counts, using res/stars.json (written weekly by the
        scheduled GitHub Action): each code chip on pubs.html becomes a
        GitHub-style "code | ★ N" button, and hard-coded "N stars" links
        on the homepage get their number refreshed. Fails silently when
        the file is missing or a repo has no count. Skipped entirely on
        pages without any GitHub link (e.g. 404.html). */
  if (document.querySelector('a[href^="https://github.com/"]')) {
    fetch('res/stars.json', { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || !d.stars) return;
        var fmt = function (n) {
          if (n < 1000) return String(n);
          var k = (n / 1000).toFixed(1);
          return (k.slice(-2) === '.0' ? k.slice(0, -2) : k) + 'k';
        };
        document.querySelectorAll('a[href^="https://github.com/"]').forEach(function (a) {
          var m = /^https:\/\/github\.com\/([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)/.exec(a.href);
          var n = m && d.stars[m[1]];
          if (n == null) return;
          if (a.classList.contains('res-chip')) {
            var s = document.createElement('span');
            s.className = 'chip-stars';
            s.textContent = '★ ' + fmt(n);
            s.title = n.toLocaleString('en-US') + ' GitHub stars';
            a.appendChild(s);
          } else if (/^\s*[\d.,]+k?\s+stars?\s*$/i.test(a.textContent)) {
            // Homepage-style hard-coded "3.9k stars" links: refresh the number
            var t = a.querySelector('strong') || a;
            t.textContent = fmt(n) + ' stars';
          }
        });
      })
      .catch(function () {});
  }

  /* 6. Venue tags. Homepage influential-papers list: wrap "CVPR 2024"-style
        venue mentions in colored tags (text inside links is left alone).
        Publications list: parenthesized venue abbreviations like
        "(<strong>CVPR</strong>)" get the same treatment — only whitelisted
        abbreviations are touched, everything else stays as it is. */
  var papersSec = document.querySelector('[aria-labelledby="papers-heading"]');
  if (papersSec) {
    var VENUES = /\b(IEEE TPAMI|IEEE TMI|IJCV|TPAMI|CVPR|ICCV|ECCV|NeurIPS|ICML|ICLR|AAAI)(\s+\d{4})?/g;
    var vclass = function (v) {
      if (/TPAMI|IJCV/.test(v)) return 'v-top';
      return /TMI/.test(v) ? 'v-journal' : 'v-' + v.toLowerCase();
    };
    Array.prototype.forEach.call(papersSec.querySelectorAll('li'), function (li) {
      var walker = document.createTreeWalker(li, NodeFilter.SHOW_TEXT);
      var nodes = [];
      while (walker.nextNode()) {
        var n = walker.currentNode;
        VENUES.lastIndex = 0;
        if (!n.parentNode.closest('a') && VENUES.test(n.nodeValue)) nodes.push(n);
      }
      nodes.forEach(function (node) {
        var text = node.nodeValue;
        var frag = document.createDocumentFragment();
        var pos = 0, m;
        VENUES.lastIndex = 0;
        while ((m = VENUES.exec(text))) {
          frag.appendChild(document.createTextNode(text.slice(pos, m.index)));
          var s = document.createElement('span');
          s.className = 'venue-tag ' + vclass(m[1]);
          s.textContent = m[0];
          frag.appendChild(s);
          pos = m.index + m[0].length;
        }
        frag.appendChild(document.createTextNode(text.slice(pos)));
        node.parentNode.replaceChild(frag, node);
      });
    });
  }

  var VMAP = {
    'CVPR': 'v-cvpr',
    'ICCV': 'v-iccv', 'ICCVW': 'v-iccv',
    'ECCV': 'v-eccv',
    'NeurIPS': 'v-neurips', 'NIPS': 'v-neurips',
    'ICML': 'v-icml',
    'ICLR': 'v-iclr',
    'AAAI': 'v-aaai',
    // top journals (Nature/Cell portfolio, TPAMI, IJCV) get the gold tag
    'IEEE TPAMI': 'v-top', 'IJCV': 'v-top',
    'Nat. Med.': 'v-top', 'Nat. Commun.': 'v-top', 'NPJ Digit. Med.': 'v-top',
    'Cell Rep. Med.': 'v-top', 'Med': 'v-top'
  };
  ['IEEE TIP', 'IEEE TMI', 'IEEE TCSVT', 'IEEE TNNLS',
   'TITS', 'THMS', 'IEEE TASE', 'RA-L', 'SCIS', 'PRL', 'JCST', 'J Intell Manuf',
   'InfSci', 'IMAVIS', 'CVIU', 'CVMJ', 'APL', 'ACM MM', 'ICIP', 'ICPR', 'PRCV',
   'ACCV', 'WACV', 'ECAI', 'CoRL', 'CCPR', '3DV'
  ].forEach(function (v) { VMAP[v] = 'v-journal'; });
  Array.prototype.forEach.call(document.querySelectorAll('ol li strong'), function (st) {
    var cls = VMAP[st.textContent.trim()];
    if (!cls) return;
    if (cls !== 'v-top') {
      // non-top venues must be wrapped in literal parentheses: (CVPR)
      var prev = st.previousSibling, next = st.nextSibling;
      if (!prev || !next || prev.nodeType !== 3 || next.nodeType !== 3) return;
      if (!/\(\s*$/.test(prev.nodeValue) || !/^\s*\)/.test(next.nodeValue)) return;
    }
    st.classList.add('venue-tag', cls);
  });

  /* 7. Subtle reveal-on-scroll for page sections. Skipped entirely when the
        user prefers reduced motion, and never applied without JS support. */
  if ('IntersectionObserver' in window
      && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          en.target.classList.add('in');
          io.unobserve(en.target);
        }
      });
    }, { rootMargin: '0px 0px -8% 0px' });
    Array.prototype.forEach.call(document.querySelectorAll('main section'), function (sec, i) {
      sec.classList.add('reveal');
      sec.style.transitionDelay = Math.min(i * 60, 240) + 'ms';
      io.observe(sec);
    });
  }

  /* 8. Back-to-top button, only on long pages */
  if (document.documentElement.scrollHeight > window.innerHeight * 3) {
    var top = document.createElement('button');
    top.className = 'back-to-top';
    top.type = 'button';
    top.title = 'Back to top';
    top.setAttribute('aria-label', 'Back to top');
    top.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
    document.body.appendChild(top);
    var onScroll = function () {
      top.classList.toggle('show', window.scrollY > 600);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    top.addEventListener('click', function () {
      var smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      window.scrollTo({ top: 0, behavior: smooth ? 'smooth' : 'auto' });
    });
  }

  /* 9. Email: the address is split across data attributes on the
        contact-icons row so the static HTML never contains it whole. JS
        prepends an envelope mailto icon; without JS no email is shown. */
  document.querySelectorAll('.contact-icons[data-u][data-d]').forEach(function (icons) {
    var addr = icons.getAttribute('data-u') + '@' + icons.getAttribute('data-d');
    var a = document.createElement('a');
    a.href = 'mailto:' + addr;
    a.title = 'Email';
    a.setAttribute('aria-label', 'Email');
    a.innerHTML = '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3.5 6.5 12 13l8.5-6.5"/></svg>';
    icons.insertBefore(a, icons.firstChild);
  });

  /* 10. Project gallery (homepage): the 2-row card track scrolls natively
         (touch swipe / trackpad / drag); JS only adds prev/next buttons and
         the dot pagination below the track. Both stay hidden when the track
         does not overflow. */
  document.querySelectorAll('.gal').forEach(function (gal) {
    var track = gal.querySelector('.gal-track');
    if (!track) return;
    var mk = function (cls, label, path) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'gal-btn ' + cls;
      b.setAttribute('aria-label', label);
      b.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="' + path + '"/></svg>';
      gal.appendChild(b);
      return b;
    };
    var prev = mk('gal-prev', 'Previous projects', 'M15 18l-6-6 6-6');
    var next = mk('gal-next', 'Next projects', 'M9 6l6 6-6 6');
    var dots = document.createElement('div');
    dots.className = 'gal-dots';
    gal.appendChild(dots);
    var pageCount = 0;
    var update = function () {
      var over = track.scrollWidth - track.clientWidth;
      var has = over > 8;
      prev.style.display = has ? '' : 'none';
      next.style.display = has ? '' : 'none';
      dots.style.display = has ? '' : 'none';
      gal.classList.toggle('gal-fit', !has);
      if (!has) return;
      var atStart = track.scrollLeft <= 4;
      var atEnd = track.scrollLeft >= over - 4;
      prev.disabled = atStart;
      next.disabled = atEnd;
      gal.classList.toggle('gal-mid', !atStart && !atEnd);
      gal.classList.toggle('gal-end', atEnd);
      var pages = Math.round(over / track.clientWidth) + 1;
      if (pages !== pageCount) {
        pageCount = pages;
        dots.textContent = '';
        for (var i = 0; i < pages; i++) {
          var d = document.createElement('button');
          d.type = 'button';
          d.className = 'gal-dot';
          d.setAttribute('aria-label', 'Go to projects page ' + (i + 1));
          (function (n) {
            d.addEventListener('click', function () {
              track.scrollTo({ left: n * track.clientWidth, behavior: smooth ? 'smooth' : 'auto' });
            });
          })(i);
          dots.appendChild(d);
        }
      }
      var active = atEnd ? pageCount - 1 : Math.round(track.scrollLeft / track.clientWidth);
      Array.prototype.forEach.call(dots.children, function (d, i) {
        d.classList.toggle('gal-dot-on', i === active);
      });
    };
    var smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    update();
    prev.addEventListener('click', function () {
      track.scrollBy({ left: -track.clientWidth, behavior: smooth ? 'smooth' : 'auto' });
    });
    next.addEventListener('click', function () {
      track.scrollBy({ left: track.clientWidth, behavior: smooth ? 'smooth' : 'auto' });
    });
    track.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
  });

  /* 10b. Respect reduced-motion: keep autoplay demo videos paused */
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.querySelectorAll('video[autoplay]').forEach(function (v) {
      v.removeAttribute('autoplay');
      v.pause();
    });
  }
});
