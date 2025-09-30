// Ensure language switch stays pinned to viewport even on buggy browsers
(function() {
  function px(varName, fallback) {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
      if (v && v.endsWith('px')) return parseInt(v, 10);
    } catch (_) {}
    return fallback;
  }

  function ensurePinned() {
    var el = document.querySelector('.lang-switch');
    if (!el) return;

    // Create a viewport-fixed overlay root and portal the switch into it
    var overlay = document.getElementById('twrb-lang-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'twrb-lang-overlay';
      // Fixed to viewport, extreme z-index to win all stacking contexts
      overlay.style.setProperty('position', 'fixed', 'important');
      overlay.style.setProperty('top', '0', 'important');
      overlay.style.setProperty('left', '0', 'important');
      overlay.style.setProperty('width', '100vw', 'important');
      overlay.style.setProperty('height', '0', 'important');
      overlay.style.setProperty('z-index', '2147483647', 'important');
      overlay.style.setProperty('pointer-events', 'none', 'important');
      document.body.appendChild(overlay);
    }
    if (el.parentElement !== overlay) {
      overlay.appendChild(el);
    }

    // Forcefully pin using absolute positioning synced with scroll
    const topGap = px('--pixel-space-3', 24);
    const rightGap = px('--pixel-space-3', 24);

    function place() {
      // Position inside overlay root using absolute coordinates
      el.style.setProperty('position', 'absolute', 'important');
      el.style.setProperty('top', topGap + 'px', 'important');
      el.style.setProperty('right', rightGap + 'px', 'important');
      el.style.setProperty('left', 'auto', 'important');
      el.style.setProperty('z-index', '2147483647', 'important');
      el.style.setProperty('pointer-events', 'auto', 'important');
    }

    let raf = null;
    function onScrollResize() {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = null;
        place();
      });
    }

    place();
    // Overlay is fixed to viewport, no need to track scroll; keep resize for safety
    window.addEventListener('resize', onScrollResize);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensurePinned);
  } else {
    ensurePinned();
  }
})();
