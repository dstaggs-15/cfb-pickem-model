/* docs/fpi.js
 * Adds ESPN FPI overlay to each game card without touching your model/pipeline.
 * - Reads docs/data/fpi.json
 * - Finds .card elements rendered by app.js
 * - Displays FPI rank/value for home & away + who FPI favors
 * - Shows a small AGREE/DISAGREE badge vs model pick
 */

(function () {
  const FPI_URL = 'data/fpi.json';

  const normalize = (s) =>
    String(s || '')
      .toLowerCase()
      .replace(/&/g, 'and')
      .replace(/[^a-z0-9 ]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

  function waitForCards(cb, tries = 40) {
    const cards = document.querySelectorAll('.card');
    if (cards.length) return cb();
    if (tries <= 0) return;
    setTimeout(() => waitForCards(cb, tries - 1), 150);
  }

  function favoredByFPI(home, away, map) {
    const h = map.get(normalize(home));
    const a = map.get(normalize(away));
    if (!h && !a) return null;
    if (h && a) {
      // Favor higher FPI value; tie -> null
      if (typeof h.fpi === 'number' && typeof a.fpi === 'number') {
        if (h.fpi > a.fpi) return home;
        if (a.fpi > h.fpi) return away;
      }
      // fallback to lower rank if values are missing
      if (typeof h.rank === 'number' && typeof a.rank === 'number') {
        if (h.rank < a.rank) return home;
        if (a.rank < h.rank) return away;
      }
    } else if (h) {
      return home;
    } else {
      return away;
    }
    return null;
  }

  function buildTeamMap(json) {
    const map = new Map();
    if (json && json.teams) {
      for (const [name, obj] of Object.entries(json.teams)) {
        map.set(normalize(name), {
          name,
          rank: (obj && typeof obj.rank === 'number') ? obj.rank : null,
          fpi:  (obj && typeof obj.fpi  === 'number') ? obj.fpi  : null
        });
      }
    }
    return map;
  }

  function renderFPIBlock(card, home, away, pick, map) {
    const h = map.get(normalize(home));
    const a = map.get(normalize(away));

    const fpiFav = favoredByFPI(home, away, map);
    const agree = fpiFav && pick ? (normalize(fpiFav) === normalize(pick)) : null;

    const wrap = document.createElement('div');
    wrap.className = 'fpi-wrap';
    wrap.innerHTML = `
      <div class="fpi-row">
        <div class="fpi-side">
          <span class="fpi-team">${home}</span>
          <span class="fpi-meta">${h ? `FPI ${h.fpi ?? '—'}  ·  #${h?.rank ?? '—'}` : 'FPI —'}</span>
        </div>
        <div class="fpi-vs">FPI</div>
        <div class="fpi-side right">
          <span class="fpi-team">${away}</span>
          <span class="fpi-meta">${a ? `FPI ${a.fpi ?? '—'}  ·  #${a?.rank ?? '—'}` : 'FPI —'}</span>
        </div>
      </div>
      <div class="fpi-fav">
        ${fpiFav ? `FPI favors: <strong>${fpiFav}</strong>` : 'FPI favors: <strong>—</strong>'}
        ${agree === true  ? `<span class="fpi-badge agree">AGREE</span>` : ''}
        ${agree === false ? `<span class="fpi-badge disagree">DISAGREE</span>` : ''}
      </div>
    `;
    card.appendChild(wrap);
  }

  function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .fpi-wrap{
        margin-top:10px;padding-top:10px;border-top:1px dashed rgba(255,255,255,.08);
      }
      .fpi-row{display:flex;align-items:center;justify-content:space-between;gap:10px;}
      .fpi-side{display:flex;flex-direction:column;gap:2px;color:#d6d6d6;}
      .fpi-side.right{ text-align:right; align-items:flex-end; }
      .fpi-team{font-weight:800;}
      .fpi-meta{font-size:.9rem;color:#a9b0b6;}
      .fpi-vs{color:#9aa0a6;font-weight:800;}
      .fpi-fav{margin-top:6px;color:#cfd2d6;font-weight:700;}
      .fpi-badge{
        margin-left:8px;padding:2px 8px;border-radius:999px;font-size:.75rem;font-weight:900;
      }
      .fpi-badge.agree{ background:#1f6f3f; color:#eaf7ee; }
      .fpi-badge.disagree{ background:#6f1f1f; color:#fde8e8; }
      @media (max-width:640px){
        .fpi-meta{font-size:.85rem;}
      }
    `;
    document.head.appendChild(style);
  }

  async function run() {
    injectStyles();
    let map = new Map();
    try {
      const resp = await fetch(FPI_URL, { cache: 'no-store' });
      if (resp.ok) {
        const json = await resp.json();
        map = buildTeamMap(json);
      }
    } catch (_) { /* ignore */ }

    // decorate existing cards
    document.querySelectorAll('.card').forEach(card => {
      const names = card.querySelectorAll('.row.hdr .team');
      if (names.length < 2) return;
      const home = names[0].textContent.trim();
      const away = names[1].textContent.trim();

      const pickEl = card.querySelector('.pick .value');
      const pick = pickEl ? pickEl.textContent.trim() : null;

      renderFPIBlock(card, home, away, pick, map);
    });
  }

  window.addEventListener('load', () => waitForCards(run));
})();
