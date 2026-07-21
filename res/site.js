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

  /* 1. Citation badges: "4000+ citations" -> pill */
  document.querySelectorAll('a').forEach(function (a) {
    if (/^[\d,]+\+?\s*citations?$/i.test(a.textContent.trim())) {
      a.classList.add('cite-badge');
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
        if (gsCit && d.citations) gsCit.textContent = Number(d.citations).toLocaleString('en-US');
        if (gsH && d.hindex) gsH.textContent = d.hindex;
        if (d.papers) {
          document.querySelectorAll('a[href*="citation_for_view="]').forEach(function (a) {
            var m = /citation_for_view=[^&:]+:([\w-]+)/.exec(a.href);
            var n = m && d.papers[m[1]];
            if (n) {
              a.textContent = (Math.floor(n / 100) * 100).toLocaleString('en-US') + '+ citations';
            }
          });
        }
      })
      .catch(function () {});
  }
});
