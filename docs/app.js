/*
 * Front-end script for the CFB Pick'em site.
 * This version fixes the bug where the page would go blank because the script
 * cleared the entire <body>. It now targets the correct container element
 * (`predictions-container`) and binds the search box to the `filterInput` ID.
 */

(async function () {
  // URLs for data
  const PRED_URL = "data/predictions.json";
  const FPI_URL = "data/fpi.json";
  const cacheBust = () => `?v=${Date.now()}`;

  // ==== 1) Custom order (Away @ Home) — adjust as needed ====
  const DESIRED_ORDER = [
    ["Texas Tech", "Utah"],
    ["Arkansas", "Memphis"],
    ["SMU", "TCU"],
    ["Michigan", "Nebraska"],
    ["North Carolina", "UCF"],
    ["NC State", "Duke"],
    ["Florida", "Miami"],
    ["Illinois", "Indiana"],
    ["Arizona State", "Baylor"],
    ["BYU", "East Carolina"],
  ];
  // Render only the games specified in DESIRED_ORDER
  const ONLY_USE_DESIRED = true;

  // ==== 2) Baked-in team colors (primary) — extend as needed ====
  const TEAM_COLORS = {
    "Texas Tech": "#CC0000",
    Utah: "#CC0000",
    Arkansas: "#9D2235",
    Memphis: "#003DA5",
    SMU: "#C41230",
    TCU: "#4D1979",
    Michigan: "#00274C",
    Nebraska: "#E41C38",
    "North Carolina": "#7BAFD4",
    UCF: "#BA9B37",
    "NC State": "#CC0000",
    Duke: "#003087",
    Florida: "#0021A5",
    Miami: "#F47321",
    Illinois: "#13294B",
    Indiana: "#990000",
    "Arizona State": "#8C1D40",
    Baylor: "#004834",
    BYU: "#002E5D",
    "East Carolina": "#592A8A",
  };

  // ---------- helpers ----------
  const normalize = (s) =>
    String(s || "")
      .toLowerCase()
      .replace(/&/g, "and")
      .replace(/[^a-z0-9 ]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();

  // Build rank map for ordering (both orientations)
  const RANK = new Map();
  DESIRED_ORDER.forEach((pair, idx) => {
    const [away, home] = pair;
    RANK.set(`${normalize(home)}__${normalize(away)}`, idx);
    RANK.set(`${normalize(away)}__${normalize(home)}`, idx);
  });

  function pickTextColor(hex) {
    if (!hex || typeof hex !== "string") return "#fff";
    const h = hex.replace("#", "");
    if (h.length !== 6) return "#fff";
    const r = parseInt(h.slice(0, 2), 16) / 255;
    const g = parseInt(h.slice(2, 4), 16) / 255;
    const b = parseInt(h.slice(4, 6), 16) / 255;
    const lin = (v) =>
      v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
    return L >= 0.6 ? "#000" : "#fff";
  }

  function colorFromName(name) {
    const str = String(name || "");
    let hash = 0;
    for (let i = 0; i < str.length; i++)
      hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
    const hue = hash % 360,
      s = 62,
      l = 38;
    return hslToHex(hue, s, l);
  }

  function hslToHex(h, s, l) {
    s /= 100;
    l /= 100;
    const c = (1 - Math.abs(2 * l - 1)) * s,
      x = c * (1 - Math.abs(((h / 60) % 2) - 1)),
      m = l - c / 2;
    let r = 0,
      g = 0,
      b = 0;
    if (0 <= h && h < 60) {
      r = c;
      g = x;
      b = 0;
    } else if (60 <= h && h < 120) {
      r = x;
      g = c;
      b = 0;
    } else if (120 <= h && h < 180) {
      r = 0;
      g = c;
      b = x;
    } else if (180 <= h && h < 240) {
      r = 0;
      g = x;
      b = c;
    } else if (240 <= h && h < 300) {
      r = x;
      g = 0;
      b = c;
    } else {
      r = 0;
      g = 0;
      b = x;
    }
    const toHex = (v) =>
      ("0" + Math.round((v + m) * 255).toString(16)).slice(-2);
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  }

  function teamColor(team) {
    const exact = TEAM_COLORS[team];
    if (exact) return exact;
    // Normalize match
    const n = normalize(team);
    for (const k of Object.keys(TEAM_COLORS)) {
      if (normalize(k) === n) return TEAM_COLORS[k];
    }
    return colorFromName(team);
  }

  function dedupe(games) {
    const byKey = {};
    games.forEach((g, i) => {
      const key = `${g.home_team}__${g.away_team}`;
      byKey[key] = { ...g, _order: i };
    });
    return Object.values(byKey).sort((a, b) => a._order - b._order);
  }

  function orderGames(games) {
    const withRank = games.map((g) => ({
      ...g,
      _rank:
        RANK.get(`${normalize(g.home_team)}__${normalize(g.away_team)}`) ??
        Number.POSITIVE_INFINITY,
    }));
    let list = withRank;
    if (ONLY_USE_DESIRED) {
      list = withRank.filter(
        (g) => g._rank !== Number.POSITIVE_INFINITY
      );
    }
    return list.sort(
      (a, b) => a._rank - b._rank || a._order - b._order
    );
  }

  // ---- load data (with cache-busting) ----
  let preds = [];
  try {
    const res = await fetch(PRED_URL + cacheBust(), { cache: "no-store" });
    const json = await res.json();
    preds = Array.isArray(json) ? json : json.games || [];
  } catch (e) {
    console.error("Failed to load predictions.json", e);
    preds = [];
  }

  // FPI map
  let fpiMap = new Map();
  let fpiMeta = null;
  try {
    const res = await fetch(FPI_URL + cacheBust(), { cache: "no-store" });
    const json = await res.json();
    if (json && Array.isArray(json.data)) {
      fpiMeta = json.meta || null;
      json.data.forEach((item) => {
        const n = normalize(item.name);
        fpiMap.set(n, item);
      });
    }
  } catch (e) {
    console.warn("No FPI data available", e);
    fpiMap = new Map();
  }

  // Prepare predictions
  const deduped = dedupe(preds);
  const ordered = orderGames(deduped);

  // ---- render ----
  // FIX: target the correct container instead of document.body
  const container =
    document.getElementById("predictions-container") || document.body;
  container.innerHTML = "";

  const percent = (x) =>
    `${(Math.max(0, Math.min(1, Number(x))) * 100).toFixed(1)}%`;

  function fpiEntry(team) {
    return fpiMap.get(normalize(team)) || null;
  }

  function favoredByFPI(home, away) {
    const H = fpiEntry(home),
      A = fpiEntry(away);
    if (!H && !A) return null;
    if (H && A) {
      if (typeof H.fpi === "number" && typeof A.fpi === "number") {
        if (H.fpi > A.fpi) return home;
        if (A.fpi > H.fpi) return away;
      }
      if (typeof H.rank === "number" && typeof A.rank === "number") {
        if (H.rank < A.rank) return home;
        if (A.rank < H.rank) return away;
      }
      return null;
    }
    return H ? home : away;
  }

  function renderFPI(card, home, away, modelPick) {
    const H = fpiEntry(home),
      A = fpiEntry(away);
    if (!H && !A) return;

    const fav = favoredByFPI(home, away);
    const agree =
      fav && modelPick
        ? normalize(fav) === normalize(modelPick)
        : null;

    const wrap = document.createElement("div");
    wrap.className = "fpi-wrap";
    wrap.innerHTML = `
      <div class="fpi-row">
        <div class="fpi-side">
          <span class="fpi-team">${home}</span>
          <span class="fpi-meta">${
            H ? `FPI ${H.fpi ?? "—"} · #${H.rank ?? "—"}` : "FPI —"
          }</span>
        </div>
        <div class="fpi-vs">FPI</div>
        <div class="fpi-side right">
          <span class="fpi-team">${away}</span>
          <span class="fpi-meta">${
            A ? `FPI ${A.fpi ?? "—"} · #${A.rank ?? "—"}` : "FPI —"
          }</span>
        </div>
      </div>
      <div class="fpi-fav">
        ${
          fav
            ? `FPI favors: <strong>${fav}</strong>`
            : "FPI favors: <strong>—</strong>"
        }
        ${
          agree === true
            ? '<span class="fpi-badge agree">AGREE</span>'
            : ""
        }
        ${
          agree === false
            ? '<span class="fpi-badge disagree">DISAGREE</span>'
            : ""
        }
        ${
          fpiMeta && fpiMeta.updated
            ? `<span class="fpi-stamp">(${fpiMeta.source || "FPI"} · ${
                fpiMeta.updated
              })</span>`
            : ""
        }
      </div>
    `;
    card.appendChild(wrap);
  }

  function renderOne(g) {
    const home = String(g.home_team || "");
    const away = String(g.away_team || "");
    const pHome = Number(
      g.model_prob_home || g.model_prob_home_raw || 0.5
    );
    const pAway = 1 - pHome;

    const homeColor = teamColor(home);
    const awayColor = teamColor(away);
    const homeText = pickTextColor(homeColor);
    const awayText = pickTextColor(awayColor);

    const card = document.createElement("div");
    card.className = "card";

    const hdr = document.createElement("div");
    hdr.className = "row hdr";
    hdr.innerHTML = `
      <div class="team left">${home}</div>
      <div class="at">@</div>
      <div class="team right">${away}</div>
    `;

    const barWrap = document.createElement("div");
    barWrap.className = "bar-wrap";
    const left = document.createElement("div");
    left.className = "bar left";
    left.style.width = `${(pHome * 100).toFixed(1)}%`;
    left.style.background = homeColor;
    left.style.color = homeText;
    left.textContent = percent(pHome);

    const right = document.createElement("div");
    right.className = "bar right";
    right.style.width = `${(pAway * 100).toFixed(1)}%`;
    right.style.background = awayColor;
    right.style.color = awayText;
    right.textContent = percent(pAway);

    barWrap.append(left, right);

    const pickLine = document.createElement("div");
    pickLine.className = "pick";
    const pickTeam = pHome >= 0.5 ? home : away;
    pickLine.innerHTML = `<span class="label">PICK:</span> <span class="value">${pickTeam}</span>`;

    card.append(hdr, barWrap, pickLine);

    // Attach FPI overlay (if available)
    renderFPI(card, home, away, pickTeam);

    return card;
  }

  function render(list) {
    container.innerHTML = "";
    list.forEach((g) => container.appendChild(renderOne(g)));
  }

  // Initial render
  render(ordered);

  // Inject CSS rules for cards and FPI block (mirrored from original script)
  const style = document.createElement("style");
  style.textContent = `
    .card {
      background:#171717; border-radius:14px; padding:16px 18px; margin:18px auto;
      max-width:860px; box-shadow:0 6px 18px rgba(0,0,0,.25); border:1px solid rgba(255,255,255,.06);
    }
    .row.hdr { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
    .row.hdr .team { font-weight:700; font-size:1.15rem; color:#eaeaea; }
    .row.hdr .at { color:#bdbdbd; font-weight:700; }
    .bar-wrap { display:flex; width:100%; height:28px; background:#2a2a2a; border-radius:8px; overflow:hidden; }
    .bar { display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.9rem; }
    .bar.left { border-right:1px solid rgba(0,0,0,.25); }
    .pick { margin-top:10px; color:#bdbdbd; font-weight:700; }
    .pick .label { color:#9aa0a6; margin-right:8px; }
    .pick .value { color:#eaeaea; }
    .fpi-wrap{ margin-top:10px; padding-top:10px; border-top:1px dashed rgba(255,255,255,.08); }
    .fpi-row{ display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .fpi-side{ display:flex; flex-direction:column; gap:2px; color:#d6d6d6; }
    .fpi-side.right{ text-align:right; align-items:flex-end; }
    .fpi-team{ font-weight:800; }
    .fpi-meta{ font-size:.9rem; color:#a9b0b6; }
    .fpi-vs{ color:#9aa0a6; font-weight:800; }
    .fpi-fav{ margin-top:6px; color:#cfd2d6; font-weight:700; }
    .fpi-badge{ margin-left:8px; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:900; }
    .fpi-badge.agree{ background:#1f6f3f; color:#eaf7ee; }
    .fpi-badge.disagree{ background:#6f1f1f; color:#fde8e8; }
    .fpi-stamp{ margin-left:8px; color:#8d94a1; font-weight:600; font-size:.8rem; }
    @media (max-width:640px){
      .card{ margin:14px 12px; }
      .row.hdr .team{ font-size:1rem; }
      .bar-wrap{ height:24px; }
      .bar{ font-size:.85rem; }
      .fpi-meta{ font-size:.85rem; }
    }
  `;
  document.head.appendChild(style);

  // Bind search to the correct input field
  const search = document.getElementById("filterInput");
  if (search) {
    search.addEventListener("input", () => {
      const q = normalize(search.value || "");
      if (!q) {
        return render(ordered);
      }
      const filtered = ordered.filter(
        (g) =>
          normalize(g.home_team).includes(q) ||
          normalize(g.away_team).includes(q)
      );
      render(filtered);
    });
  }
})();
