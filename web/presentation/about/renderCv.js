// Render About page content from cv.json (RU/EN)
(async function () {
  const locale = (document.documentElement.getAttribute('lang') || 'ru').toLowerCase();
  const isEn = locale === 'en';
  const base = isEn ? '../web' : 'web';
  const t = (obj) => (obj && typeof obj === 'object') ? (isEn ? (obj.en ?? obj.ru) : (obj.ru ?? obj.en)) : obj;

  async function load() {
    const url = `${base}/assets/cv.json`;
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`cv.json load failed: ${res.status}`);
    return await res.json();
  }

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }

  function renderTags(tags) {
    const wrap = el('div', 'pixel-flex pixel-flex-wrap pixel-gap-1 pixel-mt-2');
    (tags || []).forEach(tag => wrap.appendChild(el('span', 'pixel-tag', tag)));
    return wrap;
  }

  function renderExperience(list) {
    const host = document.querySelector('.exp-grid');
    if (!host) return;
    host.innerHTML = '';
    (list || []).forEach(item => {
      const card = el('div', 'pixel-card exp-card');
      card.appendChild(el('h3', '', t(item.role)));
      const meta = el('div', 'exp-meta');
      meta.innerHTML = `<span>${t(item.company)}</span><span class="dot"></span><span>${t(item.period)}</span>`;
      card.appendChild(meta);
      if (item.summary) card.appendChild(el('p', 'exp-summary', t(item.summary)));
      const ul = el('ul', 'exp-bullets');
      (t(item.bullets) || []).forEach(b => ul.appendChild(el('li', '', b)));
      card.appendChild(ul);
      card.appendChild(renderTags(item.tags));
      host.appendChild(card);
    });
  }

  function renderEducation(list) {
    const host = document.querySelector('.edu-grid');
    if (!host) return;
    host.innerHTML = '';
    (list || []).forEach(item => {
      const card = el('div', 'pixel-card edu-card');
      card.appendChild(el('h3', '', t(item.degree)));
      card.appendChild(el('div', 'edu-meta', t(item.meta)));
      const ul = el('ul', 'edu-highlights');
      (t(item.highlights) || []).forEach(p => ul.appendChild(el('li', '', p)));
      card.appendChild(ul);
      card.appendChild(renderTags(item.tags));
      host.appendChild(card);
    });
  }

  function renderCourses(list) {
    const host = document.querySelector('.course-grid');
    if (!host) return;
    host.innerHTML = '';
    (list || []).forEach(item => {
      const card = el('div', 'pixel-card course-card');
      card.appendChild(el('h3', '', t(item.title)));
      card.appendChild(el('div', 'course-meta', t(item.meta)));
      const ul = el('ul', 'course-points');
      (t(item.points) || []).forEach(p => ul.appendChild(el('li', '', p)));
      card.appendChild(ul);
      card.appendChild(renderTags(item.tags));
      host.appendChild(card);
    });
  }

  function renderInterests(list) {
    const host = document.querySelector('.interest-grid');
    if (!host) return;
    host.innerHTML = '';
    (list || []).forEach(item => {
      const card = el('div', 'pixel-card interest-card');
      card.appendChild(el('h3', '', t(item.title)));
      card.appendChild(el('p', '', t(item.desc)));
      host.appendChild(card);
    });
  }

  try {
    const data = await load();
    // Skills: if present, update tags block under About skills section
    const skillsWrap = document.querySelector('.pixel-card h2.section-heading[data-i18n="about.skills"]')?.parentElement?.querySelector('.pixel-flex.pixel-flex-wrap');
    if (skillsWrap && Array.isArray(data.skills)) {
      skillsWrap.innerHTML = '';
      data.skills.forEach(s => skillsWrap.appendChild(el('span', 'pixel-tag', s)));
    }
    renderExperience(data.experience);
    renderEducation(data.education);
    renderCourses(data.courses);
    renderInterests(data.interests);
  } catch (e) {
    console.warn('About cv render failed:', e);
  }
})();

