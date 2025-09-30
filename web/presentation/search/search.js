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

  function isEnglishPage() {
    try {
      const path = (typeof window !== 'undefined' && window.location && window.location.pathname) ? window.location.pathname : '';
      return /(?:-|_)en\.html$/i.test(path);
    } catch (e) { return false; }
  }

  async function fetchIndex() {
    if (DATA) return DATA;
    const { repo, branch } = getConfig();
    const en = isEnglishPage();
    const primary = `https://raw.githubusercontent.com/${repo}/${branch}/web/infrastructure/data/${en ? 'index-en.json' : 'index.json'}`;
    const altEn = `https://raw.githubusercontent.com/${repo}/${branch}/web/infrastructure/data/index_en.json`;
    const fallback = `https://raw.githubusercontent.com/${repo}/${branch}/web/infrastructure/data/index.json`;

    let resp = await fetch(primary, { cache: 'no-store' });
    if (!resp.ok && en) {
      // Try alternative EN name, then fallback to RU
      resp = await fetch(altEn, { cache: 'no-store' });
      if (!resp.ok) {
        resp = await fetch(fallback, { cache: 'no-store' });
      }
    }
    if (!resp.ok) throw new Error(`Index fetch failed: ${resp.status}`);
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
      box.innerHTML = '<div class="search-suggestion empty">Ничего не найдено</div>';
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
    const isIndex = (/\/index(?:-en|_en)?\.html?$/.test(location.pathname)) || location.pathname === '/' || location.pathname === '';
    const en = isEnglishPage();
    if (isIndex) {
      location.hash = `#${year}/${id}`;
    } else {
      // From nested pages like web/about.html → go to ../index.html
      const target = `../index${en ? '-en' : ''}.html#${year}/${id}`;
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
      box.innerHTML = '<div class="search-suggestion empty">Ошибка поиска</div>';
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
