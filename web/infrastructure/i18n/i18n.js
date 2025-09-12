// Lightweight i18n helper (no framework). Attaches to window.i18n
;(function(){
  const STORE = {
    locale: 'ru',
    supported: ['ru','en'],
    dicts: {},
    listeners: new Set(),
  };

  function detectLocaleFromPath() {
    const path = location.pathname;
    // Heuristic: '/en/' segment near start indicates EN
    return /(^|\/)en(\/|$)/.test(path) ? 'en' : 'ru';
  }

  function setHtmlLang(locale) {
    document.documentElement.setAttribute('lang', locale);
  }

  async function loadDict(locale) {
    if (STORE.dicts[locale]) return STORE.dicts[locale];
    const base = (detectLocaleFromPath()==='en') ? '..' : '.'; // when inside /en, json lives one level up
    const url = `${base}/web/infrastructure/i18n/${locale}.json`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`i18n: failed to load ${url}`);
    const json = await res.json();
    STORE.dicts[locale] = json;
    return json;
  }

  function t(key, params) {
    const dict = STORE.dicts[STORE.locale] || {};
    const raw = key.split('.').reduce((acc,k)=>acc&&acc[k], dict);
    if (!raw) return key;
    if (!params) return raw;
    return raw.replace(/\{(.*?)\}/g, (_,p)=> (p in params ? params[p] : `{${p}}`));
  }

  function bindDom(root=document) {
    // Text nodes
    root.querySelectorAll('[data-i18n]').forEach(el=>{
      const key = el.getAttribute('data-i18n');
      el.textContent = t(key);
    });
    // Attributes
    root.querySelectorAll('[data-i18n-attr]').forEach(el=>{
      const attrs = el.getAttribute('data-i18n-attr').split(',').map(s=>s.trim());
      attrs.forEach(attr=>{
        const key = el.getAttribute(`data-i18n-${attr}`);
        if (key) el.setAttribute(attr, t(key));
      });
    });
  }

  async function setLocale(locale) {
    if (!STORE.supported.includes(locale)) return;
    STORE.locale = locale;
    localStorage.setItem('twrb.locale', locale);
    await loadDict(locale);
    setHtmlLang(locale);
    bindDom();
    STORE.listeners.forEach(cb=>{ try{ cb(locale); }catch(e){} });
  }

  function onLocaleChange(cb){ STORE.listeners.add(cb); return ()=>STORE.listeners.delete(cb); }

  async function init() {
    const stored = localStorage.getItem('twrb.locale');
    const detected = detectLocaleFromPath();
    const locale = stored || detected || 'ru';
    await setLocale(locale);
  }

  window.i18n = { init, setLocale, t, bindDom, onLocaleChange };
})();

