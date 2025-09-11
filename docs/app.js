/* docs/app.js
 * Front-end only:
 *  - De-duplicate by (home_team, away_team) keeping the last row
 *  - Render in your exact order (list below)
 *  - Team colors baked in here; readable text color computed automatically
 */

(async function () {
  const PRED_URL = 'data/predictions.json';

  // ===== 1) YOUR DESIRED DISPLAY ORDER (AWAY @ HOME) =====
  // Edit this list any time. Matching is case/spacing/&-robust.
  const DESIRED_ORDER = [
    ["Clemson", "Georgia Tech"],
    ["Memphis", "Troy"],
    ["Georgia", "Tennessee"],
    ["Washington State", "North Texas"],
    ["Pittsburgh", "West Virginia"],
    ["App State", "Southern Miss"],
    ["Texas A&M", "Notre Dame"],
    ["Vanderbilt", "South Carolina"],
    ["Duke", "Tulane"],
    ["Minnesota", "California"],
    ["Wisconsin", "Alabama"],
  ];
  // =======================================================

  // ===== 2) BAKED-IN TEAM COLORS (hex). Add as needed. =====
  // Primary only; text color is picked for contrast. Fallback colors are generated.
  const TEAM_COLORS = {
    "Clemson": "#F56600",
    "Georgia Tech": "#B3A369",
    "Memphis": "#003087",
    "Troy": "#7C0025",
    "Georgia": "#BA0C2F",
    "Tennessee": "#FF8200",
    "Washington State": "#981E32",
    "North Texas": "#00853E",
    "Pittsburgh": "#003594",
    "West Virginia": "#002855",
    "Appalachian State": "#111111",
    "App State": "#111111",
    "Southern Miss": "#000000",
    "Southern Mississippi": "#000000",
    "Texas A&M": "#500000",
    "Notre Dame": "#0C2340",
    "Vanderbilt": "#866D3B",
    "South Carolina": "#73000A",
    "Duke": "#001A57",
    "Tulane": "#006747",
    "Minnesota": "#7A0019",
    "California": "#003262",
    "Cal": "#003262",
    "Alabama": "#9E1B32",
    "Wisconsin": "#C5050C"
  };
  // ========================================================

  // ---------- helpers ----------
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const normalize = (s) =>
    String(s || '')
      .toLowerCase()
      .replace(/&/g, 'and')
      .replace(/[^a-z0-9 ]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

  function pickTextColor(hex) {
    if (!hex || typeof hex !== 'string') return '#ffffff';
    const h = hex.replace('#', '');
    if (h.length !== 6) return '#ffffff';
    const r = parseInt(h.slice(0, 2), 16) / 255;
    const g = parseInt(h.slice(2, 4), 16) / 255;
    const b = parseInt(h.slice(4, 6), 16) / 255;
    const lin = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
    const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
    return L >= 0.6 ? '#000000' : '#ffffff';
  }

  function colorFromName(name) {
    const str = String(name || '');
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
    const hue = hash % 360;
    const sat = 62; // %
    const lig = 38; // %
    return hslToHex(hue, sat, lig);
  }

  function hslToHex(h, s, l) {
    s /= 100; l /= 100;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c/2;
    let r=0, g=0, b=0;
    if      (0 <= h && h < 60)  { r=c; g=x; b=0; }
    else if (60 <= h && h <120) { r=x; g=c; b=0; }
    else if (120<= h && h<180)  { r=0; g=c; b=x; }
    else if (180<= h && h<240)  { r=0; g=x; b=c; }
    else if (240<= h && h<300)  { r=x; g=0; b=c; }
    else                        { r=0; g=0; b=x; }
    const toHex = (v) => ('0' + Math.round((v + m) * 255).toString(16)).slice(-2);
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  }

  function normHex(hex) {
    if (!hex) return null;
    let h = String(hex).trim();
    if (!h) return null;
    if (!h.startsWith('#')) h = `#${h}`;
    if (h.length === 4) h = `#${h[1]}${h[1]}${h[2]}${h[2]}${h[3]}${h[3]}`;
    return h.length === 7 ? h.toLowerCase() : null;
  }

  function getTeamColor(team) {
    const exact = TEAM_COLORS[team];
    let primary = normHex(exact);
    if (!primary) primary = colorFromName(team);
    const text = pickTextColor(primary);
    return { primary, text };
  }

  // Deduplicate by (home, away) keeping last occurrence
  function dedupeGames(games) {
    const byKey = {};
    games.forEach((g, i) => {
      const key = `${g.home_team}__${g.away_team}`;
      byKey[key] = { ...g, _order: i };
    });
    return Object.values(byKey).sort((a, b) => a._order - b._order);
  }

  // Build quick lookup for desired order using AWAY@HOME key
  function makeDesiredIndex(list) {
    const idx = new Map();
    list.forEach((pair, i) => {
      const [away, home] = pair;
      idx.set(`${normalize(away)}@${normalize(home)}`, i);
    });
    return idx;
  }

  function sortToDesiredOrder(games, desiredIndex) {
    const BIG = 1e9;
    const withRank = games.map((g, pos) => {
      const keyAwayAtHome = `${normalize(g.away_team)}@${normalize(g.home_team)}`;
      const keyHomeAtAway = `${normalize(g.home_team)}@${normalize(g.away_team)}`;
      let rank = desiredIndex.has(keyAwayAtHome)
        ? desiredIndex.get(keyAwayAtHome)
        : (desiredIndex.has(keyHomeAtAway) ? desiredIndex.get(keyHomeAtAway) : BIG + pos);
      return { g, rank, pos };
    });
    withRank.sort((a, b) => (a.rank - b.rank) || (a.pos - b.pos));
    return withRank.map(x => x.g);
  }

  function percent(x) {
    const v = Math.max(0, Math.min(1, Number(x)));
    return `${(v * 100).toFixed(1)}%`;
  }

  // Try to find existing container/search input regardless of id names
  function findSearchBox() {
    return (
      $('#team-filter') ||
      $('#teamFilter') ||
      $$('input').find(el => (el.placeholder || '').toLowerCase().includes('filter')) ||
      null
    );
  }
  function findCardsContainer() {
    return $('#cards') || $('#predictions') || $('#list') || $('#root') || document.body;
  }

  // ---------- load predictions ----------
  let predictions = [];
  try {
    const res = await fetch(PRED_URL, { cache: 'no-store' });
    const json = await res.json();
    predictions = Array.isArray(json) ? json : (json.games || []);
  } catch (e) {
    console.error('Failed to load predictions.json', e);
    predictions = [];
  }

  const deduped = dedupeGames(predictions);
  const desiredIndex = makeDesiredIndex(DESIRED_ORDER);
  const ordered = sortToDesiredOrder(deduped, desiredIndex);

  // ---------- render ----------
  const container = findCardsContainer();
  container.innerHTML = '';

  function renderOne(g) {
    const home = String(g.home_team || '');
    const away = String(g.away_team || '');
    const pHome = Number(g.model_prob_home || 0.5);
    const pAway = 1 - pHome;

    const cHome = getTeamColor(home);
    const cAway = getTeamColor(away);

    const card = document.createElement('div');
    card.className = 'card';

    const hdr = document.createElement('div');
    hdr.className = 'row hdr';
    hdr.innerHTML = `
      <div class="team left">${home}</div>
      <div class="at">@</div>
      <div class="team right">${away}</div>
    `;

    const barWrap = document.createElement('div');
    barWrap.className = 'bar-wrap';

    const left = document.createElement('div');
    left.className = 'bar left';
    left.style.width = `${(pHome * 100).toFixed(1)}%`;
    left.style.background = cHome.primary;
    left.style.color = cHome.text;
    left.textContent = percent(pHome);

    const right = document.createElement('div');
    right.className = 'bar right';
    right.style.width = `${(pAway * 100).toFixed(1)}%`;
    right.style.background = cAway.primary;
    right.style.color = cAway.text;
    right.textContent = percent(pAway);

    barWrap.append(left, right);

    const pickLine = document.createElement('div');
    pickLine.className = 'pick';
    const pickTeam = pHome >= 0.5 ? home : away;
    pickLine.innerHTML = `<span class="label">PICK:</span> <span class="value">${pickTeam}</span>`;

    card.append(hdr, barWrap, pickLine);
    return card;
  }

  function render(list) {
    container.innerHTML = '';
    list.forEach(g => container.appendChild(renderOne(g)));
  }

  // initial render in your order
  render(ordered);

  // search filter (keeps your order within filtered subset)
  const search = findSearchBox();
  if (search) {
    search.addEventListener('input', () => {
      const q = (search.value || '').toLowerCase().trim();
      if (!q) return render(ordered);
      const filtered = ordered.filter(g =>
        String(g.home_team).toLowerCase().includes(q) ||
        String(g.away_team).toLowerCase().includes(q)
      );
      render(filtered);
    });
  }

  // minimal styles (safe if you already have CSS)
  const style = document.createElement('style');
  style.textContent = `
    .card {
      background:#1f1f1f; border-radius:14px; padding:16px 18px; margin:18px auto;
      max-width:860px; box-shadow:0 6px 18px rgba(0,0,0,0.25); border:1px solid rgba(255,255,255,0.06);
    }
    .row.hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
    .row.hdr .team{font-weight:700;font-size:1.15rem;color:#eaeaea}
    .row.hdr .at{color:#bdbdbd;font-weight:700}
    .bar-wrap{display:flex;width:100%;height:28px;background:#3a3a3a;border-radius:8px;overflow:hidden}
    .bar{display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.9rem}
    .bar.left{border-right:1px solid rgba(0,0,0,0.2)}
    .pick{margin-top:10px;color:#bdbdbd;font-weight:700}
    .pick .label{color:#9aa0a6;margin-right:8px}
    .pick .value{color:#eaeaea}
    @media (max-width:640px){
      .card{margin:14px 12px}
      .row.hdr .team{font-size:1rem}
      .bar-wrap{height:24px}
      .bar{font-size:.85rem}
    }
  `;
  document.head.appendChild(style);
})();
