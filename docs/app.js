/* docs/app.js
 * Bubble cards + pill bars (with % inside) + FPI row (rank, FPI, favors, AGREE/DISAGREE) + CFB Ranking badges (#N TeamName).
 * Inputs:
 *   docs/data/predictions.json -> { "games":[...] }  (array also ok)
 *   docs/data/fpi.json         -> { "data":[{name,rank,fpi},...] }  OR  { "teams":{ name:{rank,fpi} } }
 *   docs/data/cfbrank.json     -> { "ranks":[{team,rank},...] } OR { "teams": { team: rank } } (Top-25 okay)
 */

(async function () {
  const PRED_URL   = 'data/predictions.json';
  const FPI_URL    = 'data/fpi.json';
  const RANKS_URL  = 'data/cfbrank.json';
  const cacheBust  = () => `?v=${Date.now()}`;

  // === This week’s 10 (Away, Home) — EXACT ORDER YOU ASKED ===
  const DESIRED_ORDER = [
    ["LSU", "Vanderbilt"],
    ["Georgia Tech", "Duke"],
    ["Arizona", "Houston"],
    ["Baylor", "TCU"],
    ["Oklahoma", "South Carolina"],
    ["Ole Miss", "Georgia"],
    ["Old Dominion", "James Madison"],
    ["Penn State", "Iowa"],
    ["Missouri", "Auburn"],
    ["Utah", "BYU"]
    ["Alabama", "Tennessee"]
  ];
  const ONLY_USE_DESIRED = true;

  // Team colors (fallback → hashed color).
  const TEAM_COLORS = {
    // New/this week
    "LSU": "#461D7C",
    "Vanderbilt": "#866D4B",
    "Georgia Tech": "#B3A369",
    "Duke": "#003087",
    "Arizona": "#AB0520",
    "Houston": "#C8102E",
    "Baylor": "#004834",
    "TCU": "#4D1979",
    "Oklahoma": "#841617",
    "South Carolina": "#73000A",
    // Canonical for Ole Miss (aliases map "Ole Miss" -> "Mississippi")
    "Mississippi": "#082C5C",
    "Georgia": "#BA0C2F",
    "Old Dominion": "#003A70",
    "James Madison": "#450084",
    "Penn State": "#1E407C",
    "Iowa": "#FFCD00",
    "Missouri": "#F1B82D",
    "Auburn": "#0C2340",
    "Utah": "#CC0000",
    "BYU": "#0D254C",

    // Existing palette retained
    "Alabama":"#9E1B32","Arizona State":"#8C1D40","California":"#003262","Colorado":"#CFB87C",
    "Cincinnati":"#E00122","Florida":"#0021A5","Florida State":"#782F40","Indiana":"#990000",
    "Iowa State":"#A71930","Kansas":"#0051BA","Kansas State":"#512888","Louisville":"#AD0000",
    "Maryland":"#E03A3E","Michigan":"#00274C","Nebraska":"#E41C38","Oregon":"#004F27",
    "Texas":"#BF5700","UCF":"#BA9B37","UNLV":"#CC0000","USC":"#990000",
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

  const stripParens = (s) => String(s||'').replace(/\s*\(.*?\)\s*/g,' ').trim();

  // Expanded aliases for robustness
  const ALIASES = new Map([
    ["usc","southern california"],
    ["ole miss","mississippi"],
    ["ucf","central florida"],
    ["byu","brigham young"],
    ["lsu","louisiana state"],
    ["miami fl","miami"],
    ["cal","california"],
    ["kansas st","kansas state"],
    ["iowa st","iowa state"],
    ["asu","arizona state"],
    ["ariz st","arizona state"],
    ["okla","oklahoma"],
    ["ou","oklahoma"],
    ["uga","georgia"],
    ["bama","alabama"],
    ["mizzou","missouri"],
    ["kansas st.","kansas state"],
    // New helpful ones this week:
    ["psu","penn state"],
    ["penn st","penn state"],
    ["jmu","james madison"],
    ["old dominon","old dominion"],      // (typo safety)
    ["gt","georgia tech"],
    ["hou","houston"],
    ["sc","south carolina"]
  ]);

  const normAlias = (s) => {
    let n = normalize(stripParens(s));
    if (ALIASES.has(n)) return ALIASES.get(n);
    for (const [k,v] of ALIASES.entries()) {
      if (n===k || n.startsWith(k+' ') || n.endsWith(' '+k) || n.includes(' '+k+' ')) return v;
    }
    return n;
  };

  // For ordering by the list you provided (works either orientation)
  const RANK = new Map();
  DESIRED_ORDER.forEach((pair, i) => {
    const [away, home] = pair;
    RANK.set(`${normalize(home)}__${normalize(away)}`, i);
    RANK.set(`${normalize(away)}__${normalize(home)}`, i);
  });

  function pickTextColor(hex){
    if(!hex) return '#fff';
    const h=hex.replace('#',''); if(h.length!==6) return '#fff';
    const r=parseInt(h.slice(0,2),16)/255, g=parseInt(h.slice(2,4),16)/255, b=parseInt(h.slice(4,6),16)/255;
    const lin = (v)=> v<=0.04045? v/12.92 : Math.pow((v+0.055)/1.055,2.4);
    const L = 0.2126*lin(r)+0.7152*lin(g)+0.0722*lin(b);
    return L >= 0.6 ? '#000' : '#fff';
  }
  function hslToHex(h,s,l){s/=100;l/=100;const c=(1-Math.abs(2*l-1))*s,x=c*(1-Math.abs(((h/60)%2)-1)),m=l-c/2;
    let r=0,g=0,b=0;
    if(h<60){r=c;g=x;}else if(h<120){r=x;g=c;}else if(h<180){g=c;b=x;}else if(h<240){g=x;b=c;}
    else if(h<300){r=x;b=c;}else{b=x;}
    const to=(v)=>('0'+Math.round((v+m)*255).toString(16)).slice(-2);
    return `#${to(r)}${to(g)}${to(b)}`;}
  function colorFromName(name){
    let hash=0; for(let i=0;i<String(name).length;i++) hash=(hash*31+String(name).charCodeAt(i))>>>0;
    const hue=hash%360, s=62, l=38;
    return hslToHex(hue,s,l);
  }
  function teamColor(t){
    return TEAM_COLORS[t] || (()=>{const n=Object.keys(TEAM_COLORS).find(k=>normalize(k)===normalize(t));return n?TEAM_COLORS[n]:colorFromName(t);})();
  }

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

  // ---- Orientation safety: force games to match DESIRED_ORDER away/home ----
  const desiredPairsNorm = DESIRED_ORDER.map(([a,h]) => [normAlias(a), normAlias(h)]);
  function orientToDesired(g){
    const a = normAlias(g.away_team);
    const h = normAlias(g.home_team);
    for (const [ad, hd] of desiredPairsNorm){
      const matches = (a===ad && h===hd);
      const swapped = (a===hd && h===ad);
      if (matches) return g; // already correct
      if (swapped){
        const out = {...g};
        // swap teams
        out.home_team = g.away_team;
        out.away_team = g.home_team;
        // flip home probability if present
        const ph = Number(g.model_prob_home ?? g.prob_home);
        if (Number.isFinite(ph)){
          if (out.model_prob_home != null) out.model_prob_home = 1 - ph;
          else out.prob_home = 1 - ph;
        }
        // let PICK be recomputed later
        delete out.pick;
        return out;
      }
    }
    return g; // not in desired list or already fine
  }

  // ---------- load predictions ----------
  let preds=[];
  try{
    const r=await fetch(PRED_URL+cacheBust(),{cache:'no-store'});
    const j=await r.json();
    preds=Array.isArray(j)? j : (j.games||[]);
  }catch(e){
    console.error('predictions.json load failed',e);
    preds=[];
  }

  // ---------- load FPI ----------
  let fpiMap=new Map();
  try{
    const r=await fetch(FPI_URL+cacheBust(),{cache:'no-store'});
    const j=await r.json();
    const put=(name,obj)=>{
      fpiMap.set(normAlias(name),{
        name,
        rank:(obj && Number.isFinite(Number(obj.rank))) ? Number(obj.rank) : null,
        fpi: (obj && Number.isFinite(Number(obj.fpi)))  ? Number(obj.fpi)  : null
      });
    };
    if (Array.isArray(j.data)) j.data.forEach(it=>put(it.name,it));
    else if (j.teams && typeof j.teams==='object') Object.entries(j.teams).forEach(([n,o])=>put(n,o));
  }catch(e){
    console.warn('fpi.json load failed',e);
    fpiMap=new Map();
  }

  // ---------- load CFB Ranking (Top-25) ----------
  let rankMap = new Map();
  try{
    const r=await fetch(RANKS_URL+cacheBust(),{cache:'no-store'});
    if (r.ok) {
      const j=await r.json();
      if (Array.isArray(j?.ranks)) {
        j.ranks.forEach(({team,rank})=>{
          if (team && Number.isFinite(Number(rank))) rankMap.set(normAlias(team), Number(rank));
        });
      } else if (j?.teams && typeof j.teams === 'object') {
        Object.entries(j.teams).forEach(([team,rank])=>{
          if (team && Number.isFinite(Number(rank))) rankMap.set(normAlias(team), Number(rank));
        });
      }
    }
  }catch(e){
    console.warn('cfbrank.json load failed', e);
    rankMap = new Map();
  }

  // ---------- build UI ----------
  const container = document.getElementById('predictions-container') || document.body;
  container.innerHTML = '';

  // Orientation fix applied here
  const base = dedupe(preds.map(orientToDesired));
  const games = orderGames(base);

  if (!games.length){
    container.innerHTML = `<div class="status-message">No predictions found.</div>`;
    return;
  }

  const pct = (x) => (Number(x)*100).toFixed(1) + '%';
  const showName = (team) => {
    const r = rankMap.get(normAlias(team));
    return (Number.isFinite(r) ? `#${r} ` : '') + team;
  };

  games.forEach(g=>{
    const away = g.away_team;         // <-- AWAY first
    const home = g.home_team;

    const awayColor = teamColor(away);
    const homeColor = teamColor(home);
    const awayText  = pickTextColor(awayColor);
    const homeText  = pickTextColor(homeColor);

    const pHome = Number(g.model_prob_home ?? g.prob_home ?? 0.5);
    const pAway = 1 - pHome;
    const pick  = g.pick || (pHome >= 0.5 ? home : away);

    // ---- CARD ----
    const card = document.createElement('div');
    card.className = 'card';

    // Header chips and "@"
    // *** FIXED: render AWAY @ HOME (was HOME @ AWAY) ***
    const hdr = document.createElement('div');
    hdr.className = 'row hdr';
    hdr.innerHTML = `
      <div class="team" style="background:${awayColor};color:${awayText};">${showName(away)}</div>
      <div class="vs">@</div>
      <div class="team" style="background:${homeColor};color:${homeText};">${showName(home)}</div>
    `;
    card.appendChild(hdr);

    // Bars (pill) with % inside + PICK line
    const body = document.createElement('div');
    body.className = 'row body';
    body.innerHTML = `
      <div class="bar-wrap">
        <div class="bar left"  style="width:${(pAway*100).toFixed(1)}%; background:${awayColor};">
          <span class="pct">${pct(pAway)}</span>
        </div>
        <div class="bar right" style="width:${(pHome*100).toFixed(1)}%; background:${homeColor};">
          <span class="pct">${pct(pHome)}</span>
        </div>
      </div>
      <div class="meta-line"><span class="label">PICK:</span> <strong>${showName(pick)}</strong></div>
    `;
    card.appendChild(body);

    // ----- FPI (underneath) -----
    const fAway = fpiMap.get(normAlias(away));
    const fHome = fpiMap.get(normAlias(home));

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
          <span class="fpi-team">${showName(away)}</span>
          <span class="fpi-meta">${fAway ? `FPI ${fAway.fpi ?? '—'}  ·  #${fAway.rank ?? '—'}` : 'FPI —'}</span>
        </div>
        <div class="fpi-vs">FPI</div>
        <div class="fpi-side right">
          <span class="fpi-team">${showName(home)}</span>
          <span class="fpi-meta">${fHome ? `FPI ${fHome.fpi ?? '—'}  ·  #${fHome.rank ?? '—'}` : 'FPI —'}</span>
        </div>
      </div>
      <div class="fpi-fav">
        ${fpiFav ? `FPI favors: <strong>${showName(fpiFav)}</strong>` : 'FPI favors: <strong>—</strong>'}
        ${agree === true  ? `<span class="fpi-badge agree">AGREE</span>` : ''}
        ${agree === false ? `<span class="fpi-badge disagree">DISAGREE</span>` : ''}
      </div>
    `;
    card.appendChild(fpiWrap);

    container.appendChild(card);
  });

  // Filter box support (optional element with id="filterInput")
  const filterInput = document.getElementById('filterInput');
  if (filterInput) {
    const normalizeAll = (s)=>normalize(String(s||''));
    filterInput.addEventListener('input', () => {
      const q = normalizeAll(filterInput.value || '');
      $$('.card').forEach(card => {
        const txt = normalizeAll(card.textContent || '');
        card.style.display = q && !txt.includes(q) ? 'none' : '';
      });
    });
  }
})();
