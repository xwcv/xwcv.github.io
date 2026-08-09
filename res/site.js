/* Progressive UI enhancements for xwcv.github.io.
   Every feature degrades gracefully: without JS the pages render identically. */
document.addEventListener('DOMContentLoaded', function () {
  'use strict';

  /* 0. Theme toggle: persists choice to localStorage (read by the inline head script) */
  var root = document.documentElement;
  document.querySelectorAll('.theme-toggle').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
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
    if (t.length <= 32 && /^(pdf|code|arxiv|project page|project|video|website|dataset|demo|supplementary|slides)$/i.test(t)) {
      a.classList.add('res-chip');
    }
  });

  /* 3. Year quick navigation on the publications page.
        Turns "Year 2026" paragraphs into anchor targets and builds a sticky jump bar. */
  var yearPs = Array.prototype.filter.call(
    document.querySelectorAll('ol > p'),
    function (p) { return /^year\s*\d/i.test(p.textContent.trim()); }
  );
  if (yearPs.length > 3) {
    var nav = document.createElement('nav');
    nav.className = 'year-nav';
    nav.setAttribute('aria-label', 'Jump to year');
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
           While searching, the year headers (and their <br>) and any
           section left empty ("Other Conference Papers", ...) are hidden.
           "/" focuses the box, Esc clears it. */
    var items = Array.prototype.slice.call(document.querySelectorAll('ol li'));
    // map each list to its section header, e.g. <br><p><b>Book Chapters</b></p><ol>
    var lists = Array.prototype.map.call(document.querySelectorAll('ol'), function (ol) {
      var prev = ol.previousElementSibling;
      while (prev && prev.nodeName === 'BR') prev = prev.previousElementSibling;
      return { ol: ol, header: prev && prev.nodeName === 'P' && prev.querySelector('b') ? prev : null };
    });
    var box = document.createElement('div');
    box.className = 'pubs-search';
    box.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>';
    var input = document.createElement('input');
    input.type = 'search';
    input.placeholder = 'Search papers by title, author, venue ...  ( / )';
    input.setAttribute('aria-label', 'Search papers');
    var count = document.createElement('span');
    count.className = 'pubs-search-count';
    box.appendChild(input);
    box.appendChild(count);
    firstOl.parentNode.insertBefore(box, nav);
    var empty = document.createElement('p');
    empty.className = 'pubs-no-results';
    empty.textContent = 'No matching papers.';
    empty.style.display = 'none';
    firstOl.parentNode.insertBefore(empty, firstOl);
    // wrap every occurrence of q in <mark> inside the entry's text nodes
    var highlight = function (li, q) {
      Array.prototype.forEach.call(li.querySelectorAll('mark'), function (m) {
        m.replaceWith(m.textContent);
      });
      li.normalize();
      if (!q) return;
      var walker = document.createTreeWalker(li, NodeFilter.SHOW_TEXT);
      var nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
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
    input.addEventListener('input', function () {
      var q = input.value.trim().toLowerCase();
      var shown = 0;
      items.forEach(function (li) {
        var hit = !q || li.textContent.toLowerCase().indexOf(q) !== -1;
        li.style.display = hit ? '' : 'none';
        highlight(li, hit ? q : '');
        if (hit) shown++;
      });
      yearPs.forEach(function (p) {
        var br = p.previousSibling;
        p.style.display = q ? 'none' : '';
        if (br && br.nodeName === 'BR') br.style.display = q ? 'none' : '';
      });
      lists.forEach(function (s) {
        var any = Array.prototype.some.call(s.ol.querySelectorAll('li'), function (li) {
          return li.style.display !== 'none';
        });
        var hide = q && !any;
        if (s.header) s.header.style.display = hide ? 'none' : '';
        if (s.ol !== firstOl) s.ol.style.display = hide ? 'none' : '';
      });
      empty.style.display = q && !shown ? '' : 'none';
      count.textContent = q ? shown + ' / ' + items.length : '';
    });
    document.addEventListener('keydown', function (e) {
      var tag = document.activeElement && document.activeElement.tagName;
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault();
        input.focus();
      } else if (e.key === 'Escape' && document.activeElement === input) {
        input.value = '';
        input.dispatchEvent(new Event('input'));
        input.blur();
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
        scheduled GitHub Action): each code chip on pubs.htm becomes a
        GitHub-style "code | ★ N" button, and hard-coded "N stars" links
        on the homepage get their number refreshed. Fails silently when
        the file is missing or a repo has no count. */
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
});
