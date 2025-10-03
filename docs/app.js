/* docs/app.js
 * Keep the model UI (bars, @ header, % inside bars) and add an ESPN FPI row under it.
 * Inputs:
 *   docs/data/predictions.json -> { "games":[...] } or [ ... ]
 *   docs/data/fpi.json         -> { "data":[{name,rank,fpi},...] }  OR  { "teams":{ name:{rank,fpi} } }
 */

(async function () {
  const PRED_URL = 'data/predictions.json';
  const FPI_URL  = 'data/fpi.json';
  const cacheBust = () => `?v=${Date.now()}`;

  // === This week’s 10 (Away, Home) ===
  const DESIRED_ORDER = [
    ["Iowa State", "Cincinnati"],
    ["Kansas State", "Baylor"],
    ["Vanderbilt", "Alabama"],
    ["Texas", "Florida"],
    ["Virginia", "Louisville"],
    ["Washington", "Maryland"],
    ["UNLV", "Wyoming"],
    ["Miami", "Florida State"],
    ["Kansas", "UCF"],
    ["Duke", "California"]
  ];
  const ONLY_USE_DESIRED = true; // set false to show all

  // Minimal colors (unknowns get hashed colors that still look good on dark)
  const TEAM_COLORS = {
    "Alabama":"#9E1B32","Baylor":"#004834","California":"#003262","Cincinnati":"#E00122",
    "Duke":"#003087","Florida":"#0021A5","Florida State":"#782F40","Iowa State":"#A71930",
    "Kansas":"#0051BA","Kansas State":"#512888","Louisville":"#AD0000","Maryland":"#E03A3E",
    "Miami":"#F47321","Texas":"#BF5700","UCF":"#BA9B37","UNLV":"#CC0000","Vanderbilt":"#866D4B",
    "Washington":"#4B2E83","Wyoming":"#492F24"
  };

  // ---------- helpers ----------
  const $  = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const normalize = (s) =>
    String(s || '')
      .toLowerCase()
      .replace(/&/g, 'and')
      .replace(/[^a-z0-9 ]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

  // ESPN name normalization (handles “Cal”, “Miami (FL)”, etc.)
  const ALIASES = new Map([
    ["usc","southern california"],["ole miss","mississippi"],["ucf","central florida"],
    ["byu","brigham young"],["lsu","louisiana state"],["miami fl","miami"],
    ["cal","california"],["kansas st","kansas state"],["iowa st","iowa state"]
  ]);
  const stripParens = (s) => String(s||'').replace(/\s*\(.*?\)\s*/g,' ').trim();
  const normAlias = (s) => {
    let n = normalize(stripParens(s));
    if (ALIASES.has(n)) return ALIASES.get(n);
    for (const [k,v] of ALIASES.entries()) {
      if (n===k || n.startsWith(k+' ') || n.endsWith(' '+k) || n.includes(' '+k+' ')) return v;
    }
    return n;
  };

  // Order mapping
  const RANK = new Map();
  DESIRED_ORDER.forEach((pair, i) => {
    const [away, home] = pair;
    RANK.set(`${normalize(home)}__${normalize(away)}`, i);
    RANK.set(`${normalize(away)}__${normalize(home)}`, i);
  });

  function pickTextColor(hex) {
    if (!hex) return '#fff';
    const h = hex.replace('#',''); if (h.length!==6) return '#fff';
    const r=parseInt(h.slice(0,2),16)/255, g=parseInt(h.slice(2,4),16)/255, b=parseInt(h.slice(4,6),16)/255;
    const lin = (v)=> v<=0.04045? v/12.92 : Math.pow((v+0.055)/1.055,2.4);
    const L = 0.2126*lin(r)+0.7152*lin(g)+0.0722*lin(b);
    return L >= 0.6 ? '#000' : '#fff';
  }
  function colorFromName(name) {
    let hash=0; for (let i=0;i<String(name).length;i++) hash=(hash*31+String(name).charCodeAt(i))>>>0;
    const hue=hash%360, s=62, l=38;
    return hslToHex(hue,s,l);
  }
  function hslToHex(h,s,l){s/=100;l/=100;const c=(1-Math.abs(2*l-1))*s,x=c*(1-Math.abs(((h/60)%2)-1)),m=l-c/2;
    let r=0,g=0,b=0;
    if(h<60){r=c;g=x;}else if(h<120){r=x;g=c;}else if(h<180){g=c;b=x;}else if(h<240){g=x;b=c;}
    else if(h<300){r=x;b=c;}else{b=x;}
    const to=(v)=>('0'+Math.round((v+m)*255).toString(16)).slice(-2);
    return `#${to(r)}${to(g)}${to(b)}`;}
  function teamColor(t){return TEAM_COLORS[t] || (()=>{const n=Object.keys(TEAM_COLORS).find(k=>normalize(k)===normalize(t));return n?TEAM_COLORS[n]:colorFromName(t);})();}

  function dedupe(games){
    const seen={};
    games.forEach((g,i)=>{ seen[`${g.home_team}__${g.away_team}`]={...g,_order:i}; });
    return Object.values(seen).sort((a,b)=>a._order-b._order);
  }
  function orderGames(games){
    const withRank=games.map(g=>({...g,_rank:RANK.get(`${normalize(g.home_team)}__${normalize(g.away_team)}`)??Infinity}));
    let list=withRank;
    if(ONLY_USE_DESIRED) list=withRank.filter(g=>g._rank!==Infinity);
    return list.sort((a,b)=>(a._rank-b._rank)||(a._order-b._order));
  }

  // FPI styles (tiny, won’t change your existing look)
  (function injectFPIStyles(){
    const style=document.createElement('style');
    style.textContent=`
      .fpi-wrap{margin-top:10px;padding-top:10px;border-top:1px dashed rgba(255,255,255,.12);}
      .fpi-row{display:flex;align-items:center;justify-content:space-between;gap:10px;}
      .fpi-side{display:flex;flex-direction:column;gap:2px;color:#d6d6d6;}
      .fpi-side.right{text-align:right;align-items:flex-end;}
      .fpi-team{font-weight:800;}
      .fpi-meta{font-size:.9rem;color:#a9b0b6;}
      .fpi-vs{color:#9aa0a6;font-weight:800;}
      .fpi-fav{margin-top:6px;color:#cfd2d6;font-weight:700;}
      .fpi-badge{margin-left:8px;padding:2px 8px;border-radius:999px;font-size:.75rem;font-weight:900;}
      .fpi-badge.agree{background:#1f6f3f;color:#eaf7ee;}
      .fpi-badge.disagree{background:#6f1f1f;color:#fde8e8;}
      @media (max-width:640px){ .fpi-meta{font-size:.85rem;} }
    `;
    document.head.appendChild(style);
  })();

  // ---------- load model preds ----------
  let preds=[];
  try{
    const r=await fetch(PRED_URL+cacheBust(),{cache:'no-store'});
    const j=await r.json();
    preds=Array.isArray(j)? j : (j.games||[]);
  }catch(e){ console.error('predictions.json load failed',e); preds=[]; }

  // ---------- load FPI ----------
  let fpiMap=new Map();
  try{
    const r=await fetch(FPI_URL+cacheBust(),{cache:'no-store'});
    const j=await r.json();
    const put=(name,obj)=>{
      fpiMap.set(normAlias(name),{
        name, rank:(obj&&Number.isFinite(Number(obj.rank)))?Number(obj.rank):null,
              fpi: (obj&&Number.isFinite(Number(obj.fpi))) ?Number(obj.fpi) :null
      });
    };
    if (Array.isArray(j.data)) j.data.forEach(it=>put(it.name,it));
    else if (j.teams && typeof j.teams==='object') Object.entries(j.teams).forEach(([n,o])=>put(n,o));
  }catch(e){ console.warn('fpi.json load failed',e); fpiMap=new Map(); }

  // ---------- build UI ----------
  const container = $('#predictions-container');
  container.innerHTML = '';

  const base = dedupe(preds);
  const games = orderGames(base);
  if (!games.length){
    container.innerHTML = `<div class="status-message">No predictions found.</div>`;
    return;
  }

  games.forEach(g=>{
    const home = g.home_team;
    const away = g.away_team;

    const homeColor = teamColor(home);
    const awayColor = teamColor(away);
    const homeText  = pickTextColor(homeColor);
    const awayText  = pickTextColor(awayColor);

    const pHome = Number(g.model_prob_home ?? g.prob_home ?? 0.5);
    const pAway = 1 - pHome;
    const pick  = g.pick || (pHome >= 0.5 ? home : away);

    // ---- CARD ----
    const card = document.createElement('div');
    card.className = 'card';

    // Header: team boxes + "@"
    const hdr = document.createElement('div');
    hdr.className = 'row hdr';
    hdr.innerHTML = `
      <div class="team" style="background:${homeColor};color:${homeText};">${home}</div>
      <div class="vs">@</div>
      <div class="team" style="background:${awayColor};color:${awayText};">${away}</div>
    `;
    card.appendChild(hdr);

    // Bars + % inside (left/right) + PICK line
    const body = document.createElement('div');
    body.className = 'row body';
    body.innerHTML = `
      <div class="bar-wrap">
        <div class="bar left"  style="width:${(pHome*100).toFixed(1)}%; background:${homeColor};">
          <span class="pct">${(pHome*100).toFixed(1)}%</span>
        </div>
        <div class="bar right" style="width:${(pAway*100).toFixed(1)}%; background:${awayColor};">
          <span class="pct">${(pAway*100).toFixed(1)}%</span>
        </div>
      </div>
      <div class="meta-line"><span class="label">PICK:</span> <strong>${pick}</strong></div>
    `;
    card.appendChild(body);

    // -------- FPI row UNDER the bars ----------
    const fHome = fpiMap.get(normAlias(home));
    const fAway = fpiMap.get(normAlias(away));

    // Who FPI favors?
    let fpiFav = null;
    if (fHome && fAway) {
      if (Number.isFinite(fHome.fpi) && Number.isFinite(fAway.fpi)) {
        fpiFav = fHome.fpi > fAway.fpi ? home : (fAway.fpi > fHome.fpi ? away : null);
      }
      if (!fpiFav && Number.isFinite(fHome.rank) && Number.isFinite(fAway.rank)) {
        fpiFav = fHome.rank < fAway.rank ? home : (fAway.rank < fHome.rank ? away : null);
      }
    } else if (fHome && !fAway) fpiFav = home;
    else if (!fHome && fAway)   fpiFav = away;

    const agree = fpiFav && pick ? (normalize(fpiFav) === normalize(pick)) : null;

    const fpiWrap = document.createElement('div');
    fpiWrap.className = 'fpi-wrap';
    fpiWrap.innerHTML = `
      <div class="fpi-row">
        <div class="fpi-side">
          <span class="fpi-team">${home}</span>
          <span class="fpi-meta">${fHome ? `FPI ${fHome.fpi ?? '—'}  ·  #${fHome.rank ?? '—'}` : 'FPI —'}</span>
        </div>
        <div class="fpi-vs">FPI</div>
        <div class="fpi-side right">
          <span class="fpi-team">${away}</span>
          <span class="fpi-meta">${fAway ? `FPI ${fAway.fpi ?? '—'}  ·  #${fAway.rank ?? '—'}` : 'FPI —'}</span>
        </div>
      </div>
      <div class="fpi-fav">
        ${fpiFav ? `FPI favors: <strong>${fpiFav}</strong>` : 'FPI favors: <strong>—</strong>'}
        ${agree === true  ? `<span class="fpi-badge agree">AGREE</span>` : ''}
        ${agree === false ? `<span class="fpi-badge disagree">DISAGREE</span>` : ''}
      </div>
    `;
    card.appendChild(fpiWrap);

    container.appendChild(card);
  });

  // existing filter box support
  const filterInput = $('#filterInput');
  if (filterInput) {
    filterInput.addEventListener('input', () => {
      const q = normalize(filterInput.value || '');
      $$('.card').forEach(card => {
        const txt = normalize(card.textContent || '');
        card.style.display = q && !txt.includes(q) ? 'none' : '';
      });
    });
  }
})();
