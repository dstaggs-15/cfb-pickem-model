/* eslint-disable */
async function fetchJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${path}`);
  return res.json();
}

const paths = {
  picks: "./data/predictions.json",
  spreads: "./data/spreads.json",
  teams: "./data/team_summaries.json",
  colors: "./data/team_colors.json",
};

function byKey(arr, key) {
  const m = new Map();
  for (const x of arr) m.set(x[key], x);
  return m;
}

function keyHomeAway(home, away) {
  return `${home}@@${away}`;
}

function colorFor(team, colors) {
  const c = colors[team] || null;
  if (!c) return { primary: "#222", text: "#fff" };
  const primary = c.primary || c[0] || "#222";
  const text = c.text || "#fff";
  return { primary, text };
}

function pct(p) {
  return `${Math.round(p * 100)}%`;
}

function fmtSpreadRow(s) {
  if (!s) return null;
  const parts = [];
  if (s.market_line) parts.push(s.market_line);
  if (s.over_under != null && !Number.isNaN(s.over_under)) parts.push(`Total ${s.over_under}`);
  if (parts.length === 0) return null;
  return parts.join(" • ");
}

function rankBadge(p) {
  if (p == null) return "";
  const r = Math.round((1 - p) * 100) + 1; // convert percentile to rank-ish feel
  return `~P${Math.max(1, Math.min(100, r))}`;
}

function makeDetailRow(label, left, right) {
  return `
    <div class="detail-row">
      <div class="detail-label">${label}</div>
      <div class="detail-val">${left || "-"}</div>
      <div class="detail-val">${right || "-"}</div>
    </div>
  `;
}

function teamPanel(homeTeam, awayTeam, teamIndex) {
  const h = teamIndex[homeTeam] || {};
  const a = teamIndex[awayTeam] || {};

  const rows = [
    makeDetailRow("Games", h.games, a.games),
    makeDetailRow("PPG", h.ppg, a.ppg),
    makeDetailRow("PAPG (def)", h.papg, a.papg),
    makeDetailRow("Yds/G (off)", h.off_ypg, a.off_ypg),
    makeDetailRow("Yds Allowed/G", h.def_yapg, a.def_yapg),
    makeDetailRow("3rd% (off)", h.off_3rd_pct, a.off_3rd_pct),
    makeDetailRow("3rd% (def, lower better)", h.def_3rd_pct, a.def_3rd_pct),
  ];

  const ranks = [
    makeDetailRow("Off PPG rank", rankBadge(h?.ranks?.off_ppg_pct), rankBadge(a?.ranks?.off_ppg_pct)),
    makeDetailRow("Off YPG rank", rankBadge(h?.ranks?.off_ypg_pct), rankBadge(a?.ranks?.off_ypg_pct)),
    makeDetailRow("Def PAPG rank", rankBadge(h?.ranks?.def_papg_pct), rankBadge(a?.ranks?.def_papg_pct)),
    makeDetailRow("Def YAPG rank", rankBadge(h?.ranks?.def_yapg_pct), rankBadge(a?.ranks?.def_yapg_pct)),
  ];

  return `
    <div class="panel">
      <div class="panel-grid">
        <div class="panel-team">${homeTeam}</div>
        <div class="panel-team">${awayTeam}</div>
      </div>
      <div class="panel-rows">${rows.join("")}</div>
      <div class="panel-rows">${ranks.join("")}</div>
    </div>
  `;
}

function bubbleHTML(game, spreadRow, colors, teamIndex) {
  const pick = game.pick;
  const home = game.home;
  const away = game.away;
  const { primary, text } = colorFor(pick, colors);

  const market = spreadRow ? `<div class="market">${spreadRow}</div>` : "";

  return `
    <article class="card">
      <button class="bubble" style="--bubble-bg:${primary};--bubble-fg:${text};" data-key="${keyHomeAway(home, away)}">
        <div class="matchup">
          <span class="team team-home">${home}</span>
          <span class="vs">vs</span>
          <span class="team team-away">${away}</span>
        </div>
        <div class="pickline">
          <span class="pick">Pick: ${pick}</span>
          <span class="prob">(${pct(game.home === pick ? game.home_prob : game.away_prob)})</span>
        </div>
        ${market}
        <div class="hint">Click for details</div>
      </button>
      <div class="collapse" id="detail-${keyHomeAway(home, away)}">
        ${teamPanel(home, away, teamIndex)}
      </div>
    </article>
  `;
}

async function main() {
  const [picks, spreads, teams, colors] = await Promise.all([
    fetchJSON(paths.picks).catch(() => ({ games: [] })),
    fetchJSON(paths.spreads).catch(() => ({ games: [] })),
    fetchJSON(paths.teams).catch(() => ({ teams: [] })),
    fetchJSON(paths.colors).catch(() => ({})),
  ]);

  // Top meta
  const meta = document.getElementById("meta");
  const m = picks.metric || {};
  const bits = [];
  if (m.season_ahead_acc != null) bits.push(`ACC ${m.season_ahead_acc}`);
  if (m.season_ahead_auc != null) bits.push(`AUC ${m.season_ahead_auc}`);
  if (m.season_ahead_brier != null) bits.push(`Brier ${m.season_ahead_brier}`);
  meta.textContent = bits.length ? `Model: ${picks.model} | ${bits.join(" • ")}` : `Model: ${picks.model || ""}`;

  // Index data
  const spreadMap = new Map();
  (spreads.games || []).forEach(g => {
    spreadMap.set(keyHomeAway(g.home, g.away), g);
  });
  const teamIndex = {};
  (teams.teams || []).forEach(t => { teamIndex[t.team] = t; });

  // Render cards
  const cards = document.getElementById("cards");
  const chunks = [];
  for (const g of (picks.games || [])) {
    const srec = spreadMap.get(keyHomeAway(g.home, g.away)) || null;
    const spreadRow = fmtSpreadRow(srec);
    chunks.push(bubbleHTML(g, spreadRow, colors, teamIndex));
  }
  cards.innerHTML = chunks.join("");

  // Interactions
  cards.addEventListener("click", (e) => {
    const btn = e.target.closest(".bubble");
    if (!btn) return;
    const key = btn.dataset.key;
    const detail = document.getElementById(`detail-${key}`);
    if (!detail) return;
    detail.classList.toggle("open");
  });

  // Filter
  const filterBox = document.getElementById("filterBox");
  filterBox.addEventListener("input", (e) => {
    const q = e.target.value.trim().toLowerCase();
    for (const card of cards.querySelectorAll(".card")) {
      const txt = card.textContent.toLowerCase();
      card.style.display = txt.includes(q) ? "" : "none";
    }
  });
}

main().catch(err => {
  console.error(err);
  document.getElementById("cards").innerHTML = `<p class="error">Failed to load data. See console.</p>`;
});
