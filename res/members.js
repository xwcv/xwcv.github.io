/* Shared member-card rendering for group.html / group_cn.html.
   Pages define facultyMembers/currentMembers/alumniMembers datasets,
   then call renderMembers(dataset, containerId[, profileLabel]). */

// HTML-escape dataset strings before they go into innerHTML
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// 用字符串生成 hash
function randomColor(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  // 映射到色环，保证色彩明亮且分布均匀
  const hue = Math.abs(hash) % 360;
  const saturation = 65; // %
  const lightness = 60;  // %
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

// SVG 占位头像：中文名取第一个字，英文名取第一个单词
function createAvatar(name) {
  const compact = (name || '').trim();
  const cjk = /^[一-鿿]/.test(compact);
  const first = cjk ? compact[0] : compact.split(/\s+/)[0];
  const bg = randomColor(first);
  const fontSize = cjk ? '2.2em' : '1.2em';
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120' preserveAspectRatio='xMidYMid meet'><rect width='100%' height='100%' rx='60' fill='${bg}'/><text x='50%' y='50%' text-anchor='middle' dominant-baseline='middle' dy='.1em' font-size='${fontSize}' font-family='Arial' fill='white' font-weight='bold'>${first}</text></svg>`;
  // percent-encode: a raw '#', '&' or '<' in the name would corrupt the URI
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function renderMembers(members, containerId, profileLabel) {
  const label = profileLabel || 'Profile';
  const container = document.getElementById(containerId);
  // a typo'd container id must not abort the remaining renderMembers calls
  if (!container) return;
  container.innerHTML = '';
  members.forEach(member => {
    // 真实图片接口：如有真实图片，将 src 换成图片路径
    const imgSrc = member.photo ? member.photo : createAvatar(member.name);

    // 将 desc 中的 Markdown 链接 [text](url) 转换为 HTML <a> 标签
    const descWithLinks = escapeHtml(member.desc || '').replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener" style="font-weight: 600;">$1</a>'
    );

    const card = document.createElement('div');
    card.className = 'member-card';
    card.innerHTML = `
      <img class="member-photo" src="${imgSrc}" alt="${escapeHtml(member.name)}" loading="lazy" decoding="async">
      <div class="member-name">${escapeHtml(member.name)}</div>
      <div class="member-desc">${descWithLinks}</div>
      ${member.link ? `<div class="member-link"><a href="${escapeHtml(member.link)}" target="_blank" rel="noopener">${label}</a></div>` : ''}
    `;
    container.appendChild(card);
  });
}
