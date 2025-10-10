/* globals Chart */
const DATA_URL = 'docs/data/team_stats.json';
const RANK_URL = 'docs/data/cfbrank.json';

// The metrics we show + how to pull them + whether higher is better
const METRICS = [
  { key: 'ppg',          label: 'Points / Game (PPG)',       pathOff: ['offense.ppg','off_ppg','ppg_off'], pathDef: ['defense.ppg','def_ppg','ppg_def'], higherIsBetter: true,  decimals: 1 },
  { key: 'yards_pg',     label: 'Yards / Game',              pathOff: ['offense.yards_per_game','off_yards_per_game','yards_off_g'], pathDef: ['defense.yards_per_game','def_yards_per_game','yards_def_g'], higherIsBetter: true,  decimals: 1 },
  { key: 'success_rate', label: 'Success Rate',              pathOff: ['offense.success_rate','off_success_rate'], pathDef: ['defense.success_rate','def_success_rate'], higherIsBetter: true,  decimals: 3 },
  { key: 'explosive',    label: 'Explosiveness',             pathOff: ['offense.explosiveness','off_explosiveness'], pathDef: ['defense.explosiveness','def_explosiveness'], higherIsBetter: true,  decimals: 3 },
  { key: 'ppa',          label: 'PPA (Pred. Points Added)',  pathOff: ['offense.ppa','off_ppa'], pathDef: ['defense.ppa','def_ppa'], higherIsBetter: true,  decimals: 3 },
];
const EXTRA = [
  { key: 'fpi', label: 'ESPN FPI', path: ['fpi','espn_fpi'], decimals: 2, higherIsBetter: true }
];

let rawTeams = [];
let rankMap = new Map(); // Team -> rank #
let teamsByName = new Map(); // Team -> object
let radarChart, effBarChart;

init().catch(console.error);

async function init(){
  const [teamRes, rankRes] = await Promise.all([
    fetch(DATA_URL, {cache:'no-store'}),
    fetch(RANK_URL, {cache:'no-store'})
  ]);
  rawTeams = await teamRes.json();
  const rankJson = await rankRes.json().catch(()=>({teams:{}}));
  rankMap = new Map(Object.entries(rankJson.teams || {}));

  // Make lookup by normalized name
  teamsByName = new Map(rawTeams.map(t => [norm(t.team || t.name), t]));

  // Compute per-stat ranks across all teams
  computeAllRanks();

  // Build selects
  buildTeamSelects();

  // Render default team (first in ranked list if present)
  const first = document.querySelector('#teamSelect option')?.value;
  if (first){
    document.getElementById('teamSelect').value = first;
    renderTeam(first);
  }

  // Compare handler
  document.getElementById('compareBtn').addEventListener('click', () => {
    const a = document.getElementById('teamA').value;
    const b = document.getElementById('teamB').value;
    renderCompare(a, b);
  });

  // Change team
  document.getElementById('teamSelect').addEventListener('change', (e)=>{
    renderTeam(e.target.value);
  });
}

function norm(s){ return (s||'').trim(); }

function buildTeamSelects(){
  // Sort by rank first, then A->Z
  const sorted = [...teamsByName.values()].sort((a,b)=>{
    const ra = rankMap.get(norm(a.team || a.name)) || 9999;
    const rb = rankMap.get(norm(b.team || b.name)) || 9999;
    if (ra !== rb) return ra - rb;
    return norm(a.team||a.name).localeCompare(norm(b.team||b.name));
  });

  const mainSel = document.getElementById('teamSelect');
  const aSel    = document.getElementById('teamA');
  const bSel    = document.getElementById('teamB');
  [mainSel,aSel,bSel].forEach(sel => sel.innerHTML='');

  for (const t of sorted){
    const name = norm(t.team || t.name);
    const rank = rankMap.get(name);
    const fpi  = getVal(t, ['fpi','espn_fpi']);
    const label = `${rank ? '#'+rank+' ' : ''}${name}${Number.isFinite(fpi) ? ` — FPI: ${fpi.toFixed(2)}`:''}`;
    const opt1 = new Option(label, name);
    const opt2 = new Option(label, name);
    const opt3 = new Option(label, name);
    mainSel.add(opt1);
    aSel.add(opt2);
    bSel.add(opt3);
  }
}

function getVal(obj, paths){
  for (const p of paths){
    const v = drill(obj, p);
    if (v !== undefined && v !== null) return (typeof v === 'string') ? Number(v) : v;
  }
  return undefined;
}
function drill(o, path){ // 'a.b.c'
  const parts = path.split('.');
  let cur = o;
  for (const k of parts){
    if (cur && Object.prototype.hasOwnProperty.call(cur, k)){
      cur = cur[k];
    } else return undefined;
  }
  return cur;
}

function computeAllRanks(){
  // Create arrays of values for each metric (offensive-side for ranks generally)
  METRICS.forEach(m => {
    const vals = [];
    for (const t of teamsByName.values()){
      const v = getVal(t, m.pathOff);
      if (Number.isFinite(v)) vals.push(v);
    }
    // Rank: higher is better unless specified
    const sorted = [...vals].sort((a,b) => m.higherIsBetter ? (b-a) : (a-b));
    for (const t of teamsByName.values()){
      const v = getVal(t, m.pathOff);
      if (!Number.isFinite(v)) continue;
      const r = sorted.findIndex(x => x === v) + 1;
      t.__ranks = t.__ranks || {};
      t.__ranks[m.key] = r || null;
    }
  });
}

function renderTeam(teamName){
  const team = teamsByName.get(norm(teamName));
  if (!team) return;

  // header badge
  const badge = document.getElementById('teamBadge');
  const rank  = rankMap.get(norm(team.team || team.name));
  const fpi   = getVal(team, EXTRA[0].path);
  badge.textContent = `${rank ? '#'+rank+' ' : ''}${team.team || team.name}${Number.isFinite(fpi) ? ` — FPI: ${fpi.toFixed(EXTRA[0].decimals)}`:''}`;

  // summary cards
  const grid = document.getElementById('summaryGrid');
  grid.innerHTML = '';
  const cards = [];

  // Offense cards
  for (const m of METRICS){
    const vOff = getVal(team, m.pathOff);
    const vDef = getVal(team, m.pathDef);
    cards.push(cardEl(`${m.label} — Offense`, formatVal(vOff,m), rankText(team.__ranks?.[m.key])));
    cards.push(cardEl(`${m.label} — Defense`, formatVal(vDef,m)));
  }
  // Extra
  cards.push(cardEl(EXTRA[0].label, Number.isFinite(fpi) ? fpi.toFixed(EXTRA[0].decimals) : '—'));

  cards.forEach(c => grid.appendChild(c));

  // charts
  drawRadar(team);
  drawEffBar(team);
}

function cardEl(label, value, sub){
  const d = document.createElement('div');
  d.className = 'card stat-card';
  d.innerHTML = `
    <div class="stat-label">${label}</div>
    <div class="stat-value">${value ?? '—'}</div>
    ${sub ? `<div class="stat-sub">${sub}</div>`:''}
  `;
  return d;
}
function rankText(rk){
  return rk ? `Rank: ${rk}` : '';
}
function formatVal(v, meta){
  if (!Number.isFinite(v)) return '—';
  const d = Number.isFinite(meta.decimals) ? meta.decimals : 2;
  return v.toFixed(d);
}

// Normalize to 0..1 for the radar
function normalize(vals){
  const finite = vals.filter(v => Number.isFinite(v));
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  return vals.map(v=>{
    if (!Number.isFinite(v)) return 0;
    if (max === min) return 0.5;
    return (v - min) / (max - min);
  });
}

function drawRadar(team){
  const labels = METRICS.map(m => m.label.replace(/ \(.+?\)/,''));
  const raw = METRICS.map(m => getVal(team, m.pathOff));
  const data = normalize(raw);

  const ctx = document.getElementById('radarChart');
  if (radarChart) radarChart.destroy();
  radarChart = new Chart(ctx, {
    type:'radar',
    data:{
      labels,
      datasets:[{
        label:'Offense (norm.)',
        data,
        fill:true
      }]
    },
    options:{
      responsive:true,
      scales:{ r:{ grid:{color:'#222'}, pointLabels:{color:'#b7bec6'} } },
      plugins:{ legend:{ labels:{ color:'#b7bec6'} } }
    }
  });
}

function drawEffBar(team){
  // Pick a few “efficiency-ish” stats: PPA, Success Rate, Explosiveness
  const trio = ['ppa','success_rate','explosive'];
  const labels = METRICS.filter(m => trio.includes(m.key)).map(m=>m.label);
  const off = METRICS.filter(m => trio.includes(m.key)).map(m => getVal(team, m.pathOff) ?? 0);
  const def = METRICS.filter(m => trio.includes(m.key)).map(m => getVal(team, m.pathDef) ?? 0);

  const ctx = document.getElementById('effBarChart');
  if (effBarChart) effBarChart.destroy();
  effBarChart = new Chart(ctx, {
    type:'bar',
    data:{
      labels,
      datasets:[
        { label:'Offense', data: off },
        { label:'Defense', data: def }
      ]
    },
    options:{
      responsive:true,
      plugins:{ legend:{ labels:{ color:'#b7bec6'} } },
      scales:{
        x:{ ticks:{ color:'#b7bec6' }, grid:{ color:'#222' } },
        y:{ ticks:{ color:'#b7bec6' }, grid:{ color:'#222' } }
      }
    }
  });
}

// ===== Compare =====
function renderCompare(nameA, nameB){
  const tA = teamsByName.get(norm(nameA));
  const tB = teamsByName.get(norm(nameB));
  if (!tA || !tB) return;

  document.getElementById('thA').textContent = displayName(tA);
  document.getElementById('thB').textContent = displayName(tB);

  const body = document.getElementById('compareBody');
  body.innerHTML = '';

  // For each stat: compute rank (already done for offense) and decide advantage
  for (const m of METRICS){
    const rkA = tA.__ranks?.[m.key] ?? null;
    const rkB = tB.__ranks?.[m.key] ?? null;

    const adv = decideAdvantage(rkA, rkB);
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${m.label} (Off.)</td>
      <td>${rankCell(rkA)}</td>
      <td>${rankCell(rkB)}</td>
      <td>${advHtml(adv)}</td>
    `;
    body.appendChild(row);
  }

  // Extra: FPI head-to-head
  const fpiA = getVal(tA, EXTRA[0].path);
  const fpiB = getVal(tB, EXTRA[0].path);
  const advF = Number.isFinite(fpiA) && Number.isFinite(fpiB)
    ? (fpiA === fpiB ? 'tie' : (fpiA > fpiB ? 'A' : 'B')) : 'na';

  const frow = document.createElement('tr');
  frow.innerHTML = `
    <td>${EXTRA[0].label}</td>
    <td>${Number.isFinite(fpiA) ? fpiA.toFixed(EXTRA[0].decimals) : '—'}</td>
    <td>${Number.isFinite(fpiB) ? fpiB.toFixed(EXTRA[0].decimals) : '—'}</td>
    <td>${advHtml(advF)}</td>
  `;
  body.appendChild(frow);
}

function displayName(t){
  const name = norm(t.team || t.name);
  const rank = rankMap.get(name);
  return rank ? `#${rank} ${name}` : name;
}
function rankCell(r){ return r ? `#${r}` : '—'; }
function decideAdvantage(rA, rB){
  if (!rA || !rB) return 'na';
  if (rA === rB) return 'tie';
  return (rA < rB) ? 'A' : 'B'; // smaller (closer to #1) is better
}
function advHtml(who){
  if (who === 'na') return '<span class="badge-bad">n/a</span>';
  if (who === 'tie') return '<span class="badge-good">Tie</span>';
  return who === 'A'
    ? '<span class="badge-good">Team A</span>'
    : '<span class="badge-good">Team B</span>';
}
