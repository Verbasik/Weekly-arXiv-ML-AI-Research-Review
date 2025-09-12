/* Lightweight cross-page search suggestions for TWRB
 * - Works on any page containing .nav-search (input + button)
 * - Uses web/infrastructure/data/index.json from GitHub raw
 * - On main page (index.html) it defers to ResearchController (window.ResearchApp)
 */
(function () {
  // If main app controller is present, it already manages search
  if (window.ResearchApp && window.ResearchApp.config) {
    return; // index.html path — handled by ResearchController
  }

  const holder = document.querySelector('.nav-search');
  if (!holder) return;

  const input = holder.querySelector('input');
  const button = holder.querySelector('button');
  if (!input || !button) return;

  // Build suggestions container
  holder.style.position = 'relative';
  const box = document.createElement('div');
  box.className = 'search-suggestions pixel-card';
  box.setAttribute('role', 'listbox');
  box.style.display = 'none';
  holder.appendChild(box);

  // Data cache
  let DATA = null;
  let FLAT = null;
  let timer = null;

  function getConfig() {
    // Fallbacks for repo/branch
    const repo = 'Verbasik/Weekly-arXiv-ML-AI-Research-Review';
    const branch = 'develop';
    return { repo, branch };
  }

  async function fetchIndex() {
    if (DATA) return DATA;
    const { repo, branch } = getConfig();
    const lang = (document.documentElement.getAttribute('lang') || 'ru').toLowerCase();
    const file = lang === 'en' ? 'index.en.json' : 'index.json';
    const url = `https://raw.githubusercontent.com/${repo}/${branch}/web/infrastructure/data/${file}`;
    let resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) {
      // Fallback to RU index if EN missing
      const fallback = `https://raw.githubusercontent.com/${repo}/${branch}/web/infrastructure/data/index.json`;
      resp = await fetch(fallback, { cache: 'no-store' });
      if (!resp.ok) throw new Error(`Index fetch failed: ${resp.status}`);
    }
    DATA = await resp.json();
    FLAT = flatten(DATA);
    return DATA;
  }

  function flatten(data) {
    const items = [];
    if (!data || !Array.isArray(data.years)) return items;
    for (const y of data.years) {
      const year = y.year;
      for (const w of (y.weeks || [])) {
        items.push({
          year,
          id: w.week,
          title: w.title || '',
          description: w.description || '',
          tags: Array.isArray(w.tags) ? w.tags : []
        });
      }
    }
    return items;
  }

  function score(item, q) {
    let s = 0;
    const t = item.title.toLowerCase();
    const d = item.description.toLowerCase();
    if (t.includes(q)) s += 10;
    if (d.includes(q)) s += 5;
    for (const tag of item.tags) {
      if ((tag || '').toLowerCase().includes(q)) s += 7;
    }
    return s;
  }

  function searchLocal(query) {
    const q = (query || '').trim().toLowerCase();
    if (q.length < 2) return [];
    return FLAT
      .map(item => ({ item, relevance: score(item, q) }))
      .filter(r => r.relevance > 0)
      .sort((a, b) => b.relevance - a.relevance)
      .map(r => r.item);
  }

  function renderSuggestions(list) {
    if (!list || list.length === 0) {
      const msg = (window.i18n && typeof window.i18n.t === 'function') ? (window.i18n.t('search.empty') || 'Ничего не найдено') : 'Ничего не найдено';
      box.innerHTML = `<div class="search-suggestion empty">${msg}</div>`;
      box.style.display = 'block';
      return;
    }
    const max = 8;
    box.innerHTML = list.slice(0, max).map(it => (
      `<button class="search-suggestion" role="option" data-year="${it.year}" data-id="${it.id}">` +
      `${it.title}` +
      `</button>`
    )).join('');
    box.style.display = 'block';

    box.querySelectorAll('.search-suggestion').forEach(btn => {
      btn.addEventListener('click', () => {
        const year = btn.getAttribute('data-year');
        const id = btn.getAttribute('data-id');
        openOnMain(year, id);
      });
    });
  }

  function openOnMain(year, id) {
    // If already on main page
    const isIndex = /\/index\.html?$/.test(location.pathname) || location.pathname === '/' || location.pathname === '';
    const isEn = /(^|\/)en(\/|$)/.test(location.pathname) || (document.documentElement.getAttribute('lang') || '').toLowerCase() === 'en';
    if (isIndex) {
      location.hash = `#${year}/${id}`;
    } else {
      const target = isEn ? `../en/index.html#${year}/${id}` : `../index.html#${year}/${id}`;
      location.href = target;
    }
    box.style.display = 'none';
  }

  async function handleSuggest(value) {
    const q = (value || '').trim();
    if (q.length < 2) {
      box.style.display = 'none';
      box.innerHTML = '';
      return;
    }
    try {
      await fetchIndex();
      renderSuggestions(searchLocal(q));
    } catch (e) {
      console.error(e);
      const msg = (window.i18n && typeof window.i18n.t === 'function') ? (window.i18n.t('search.error') || 'Ошибка поиска') : 'Ошибка поиска';
      box.innerHTML = `<div class=\"search-suggestion empty\">${msg}</div>`;
      box.style.display = 'block';
    }
  }

  // Events
  input.addEventListener('input', (e) => {
    const v = e.target.value || '';
    clearTimeout(timer);
    timer = setTimeout(() => handleSuggest(v), 200);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      box.style.display = 'none';
      box.innerHTML = '';
    }
    if (e.key === 'Enter') {
      // open first result if exists
      const first = box.querySelector('.search-suggestion');
      if (first) first.click();
    }
  });

  button.addEventListener('click', () => {
    const first = box.querySelector('.search-suggestion');
    if (first) {
      first.click();
    } else {
      handleSuggest(input.value || '');
    }
  });

  document.addEventListener('click', (e) => {
    if (!holder.contains(e.target)) {
      box.style.display = 'none';
    }
  });
})();
