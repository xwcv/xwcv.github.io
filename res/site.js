/* Progressive UI enhancements for xwcv.github.io.
   Every feature degrades gracefully: without JS the pages render identically. */
document.addEventListener('DOMContentLoaded', function () {
  'use strict';

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
});
