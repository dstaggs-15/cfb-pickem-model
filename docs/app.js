/* ======= Team Colors + UI rendering (bubble cards) =======
   - Uses built-in TEAM_COLORS (primary/secondary).
   - Optionally merges docs/assets/team_colors.json if present.
   - Falls back to a deterministic nice color via hashing.
*/

// Minimal, robust aliasing so names like "UTEP" / "Texas-El Paso" match
const TEAM_ALIASES = {
  "texas-el paso": "UTEP",
  "utep miners": "UTEP",
  "utsa": "UTSA",
  "texas-san antonio": "UTSA",
  "texas a & m": "Texas A&M",
  "texas a&m": "Texas A&M",
  "ole miss": "Ole Miss",
  "mississippi": "Ole Miss",
  "louisiana state": "LSU",
  "ohio st": "Ohio State",
  "oklahoma st": "Oklahoma State",
  "washington st": "Washington State",
  "arizona st": "Arizona State",
  "colorado st": "Colorado State",
  "fresno st": "Fresno State",
  "boise st": "Boise State",
  "georgia st": "Georgia State",
  "georgia so": "Georgia Southern",
  "southern miss": "Southern Miss",
  "utsa roadrunners": "UTSA",
  "ucf knights": "UCF",
  "miami (fl)": "Miami (FL)",
  "miami fl": "Miami (FL)",
  "miami (oh)": "Miami (OH)",
  "miami oh": "Miami (OH)",
  "sj state": "San Jose State",
  "sjsu": "San Jose State",
  "hawaii": "Hawai'i",
  "hawai'i": "Hawai'i",
};

// Built-in colors (primary, secondary). Additions welcome.
// Covers Power teams + many G5. Unknowns use fallback hashing.
// If you want 100% coverage with official hexes, drop a JSON at docs/assets/team_colors.json:
// { "Texas": {"primary":"#BF5700","secondary":"#333333"}, ... }
const TEAM_COLORS = {
  "Alabama":        { primary: "#9E1B32", secondary: "#828A8F" },
  "Auburn":         { primary: "#0C2340", secondary: "#E87722" },
  "Arkansas":       { primary: "#9D2235", secondary: "#000000" },
  "LSU":            { primary: "#461D7C", secondary: "#FDD023" },
  "Texas A&M":      { primary: "#500000", secondary: "#FFFFFF" },
  "Ole Miss":       { primary: "#CF102D", secondary: "#14213D" },
  "Mississippi State": { primary: "#660000", secondary: "#FFFFFF" },
  "Georgia":        { primary: "#CC0000", secondary: "#000000" },
  "Florida":        { primary: "#0021A5", secondary: "#FA4616" },
  "Tennessee":      { primary: "#FF8200", secondary: "#58595B" },
  "Kentucky":       { primary: "#0033A0", secondary: "#FFFFFF" },
  "South Carolina": { primary: "#73000A", secondary: "#000000" },
  "Missouri":       { primary: "#F1B82D", secondary: "#000000" },
  "Vanderbilt":     { primary: "#866D4B", secondary: "#000000" },

  "Texas":          { primary: "#BF5700", secondary: "#333333" },
  "Oklahoma":       { primary: "#841617", secondary: "#FFFFFF" },
  "Oklahoma State": { primary: "#FF7300", secondary: "#000000" },
  "Kansas":         { primary: "#0051BA", secondary: "#E8000D" },
  "Kansas State":   { primary: "#512888", secondary: "#7F7F7F" },
  "Baylor":         { primary: "#154734", secondary: "#FFC72C" },
  "TCU":            { primary: "#4D1979", secondary: "#A2AAAD" },
  "Texas Tech":     { primary: "#CC0000", secondary: "#000000" },
  "Iowa State":     { primary: "#C8102E", secondary: "#F1BE48" },
  "West Virginia":  { primary: "#002855", secondary: "#EAAA00" },
  "UCF":            { primary: "#BA9B37", secondary: "#000000" },
  "Cincinnati":     { primary: "#E00122", secondary: "#000000" },
  "Houston":        { primary: "#C8102E", secondary: "#000000" },
  "BYU":            { primary: "#002255", secondary: "#A9B0B7" },

  "Ohio State":     { primary: "#CE0F3D", secondary: "#666666" },
  "Michigan":       { primary: "#00274C", secondary: "#FFCB05" },
  "Penn State":     { primary: "#041E42", secondary: "#FFFFFF" },
  "Michigan State": { primary: "#18453B", secondary: "#FFFFFF" },
  "Maryland":       { primary: "#E03A3E", secondary: "#FFD520" },
  "Rutgers":        { primary: "#CC0033", secondary: "#000000" },
  "Indiana":        { primary: "#990000", secondary: "#FFFFFF" },
  "Illinois":       { primary: "#13294B", secondary: "#E84A27" },
  "Iowa":           { primary: "#000000", secondary: "#F1BE48" },
  "Minnesota":      { primary: "#7A0019", secondary: "#FFCC33" },
  "Nebraska":       { primary: "#E41C38", secondary: "#000000" },
  "Northwestern":   { primary: "#4E2A84", secondary: "#FFFFFF" },
  "Purdue":         { primary: "#CFB991", secondary: "#000000" },
  "Wisconsin":      { primary: "#C5050C", secondary: "#FFFFFF" },
  "Oregon":         { primary: "#154733", secondary: "#FEE123" },
  "Washington":     { primary: "#4B2E83", secondary: "#B7A57A" },
  "USC":            { primary: "#990000", secondary: "#FFC72C" },
  "UCLA":           { primary: "#3284BF", secondary: "#FFB300" },

  "Notre Dame":     { primary: "#0C2340", secondary: "#AE9142" },
  "Clemson":        { primary: "#F56600", secondary: "#522D80" },
  "Florida State":  { primary: "#782F40", secondary: "#CEB888" },
  "Miami (FL)":     { primary: "#005030", secondary: "#F47321" },
  "Duke":           { primary: "#003087", secondary: "#FFFFFF" },
  "North Carolina": { primary: "#7BAFD4", secondary: "#13294B" },
  "NC State":       { primary: "#CC0000", secondary: "#000000" },
  "Wake Forest":    { primary: "#9E7E38", secondary: "#000000" },
  "Virginia":       { primary: "#232D4B", secondary: "#F84C1E" },
  "Virginia Tech":  { primary: "#630031", secondary: "#CF4420" },
  "Boston College": { primary: "#98002E", secondary: "#A39161" },
  "Syracuse":       { primary: "#F76900", secondary: "#002D62" },
  "Georgia Tech":   { primary: "#B3A369", secondary: "#003057" },
  "Pitt":           { primary: "#003594", secondary: "#FFB81C" },
  "Louisville":     { primary: "#AD0000", secondary: "#000000" },
  "SMU":            { primary: "#E41C38", secondary: "#00539B" },

  "Utah":           { primary: "#CC0000", secondary: "#000000" },
  "Colorado":       { primary: "#CFB87C", secondary: "#000000" },
  "Arizona":        { primary: "#AB0520", secondary: "#0C234B" },
  "Arizona State":  { primary: "#8C1D40", secondary: "#FFC627" },
  "Stanford":       { primary: "#8C1515", secondary: "#FFFFFF" },
  "California":     { primary: "#003262", secondary: "#FDB515" },
  "Oregon State":   { primary: "#DC4405", secondary: "#000000" },
  "Washington State": { primary: "#981E32", secondary: "#5E6A71" },

  "Boise State":    { primary: "#0033A0", secondary: "#D64309" },
  "Fresno State":   { primary: "#C41230", secondary: "#00285E" },
  "San Diego State":{ primary: "#A6192E", secondary: "#000000" },
  "San Jose State": { primary: "#0055A2", secondary: "#FFB81C" },
  "Hawai'i":        { primary: "#024731", secondary: "#A2AAAD" },
  "Air Force":      { primary: "#003087", secondary: "#A2AAAD" },
  "Colorado State": { primary: "#1E4D2B", secondary: "#C8C372" },
  "Utah State":     { primary: "#0F2439", secondary: "#C8C8C8" },
  "Wyoming":        { primary: "#582C1F", secondary: "#FFC425" },
  "UNLV":           { primary: "#C41E3A", secondary: "#000000" },
  "Nevada":         { primary: "#003366", secondary: "#9EA2A2" },
  "New Mexico":     { primary: "#BA0C2F", secondary: "#A7A8AA" },

  "Memphis":        { primary: "#0C1D8C", secondary: "#A5ACAF" },
  "Tulane":         { primary: "#006747", secondary: "#66D0A4" },
  "UTSA":           { primary: "#0C2340", secondary: "#F15A22" },
  "North Texas":    { primary: "#00853E", secondary: "#000000" },
  "Rice":           { primary: "#00205B", secondary: "#A2AAAD" },
  "SMU":            { primary: "#E41C38", secondary: "#00539B" },
  "Navy":           { primary: "#00205B", secondary: "#C5B783" },
  "Army":           { primary: "#2E2D29", secondary: "#C5B783" },
  "Temple":         { primary: "#9D2235", secondary: "#000000" },
  "UAB":            { primary: "#115740", secondary: "#C69214" },
  "Charlotte":      { primary: "#0B5138", secondary: "#B9975B" },
  "FAU":            { primary: "#00335F", secondary: "#CC0033" },
  "FIU":            { primary: "#081E3F", secondary: "#B6862C" },
  "WKU":            { primary: "#D50032", secondary: "#000000" },
  "Middle Tennessee": { primary: "#0066CC", secondary: "#A2AAAD" },
  "Louisiana Tech": { primary: "#003087", secondary: "#D50032" },

  "Appalachian State": { primary: "#222222", secondary: "#FFCC00" },
  "Coastal Carolina":  { primary: "#007377", secondary: "#231F20" },
  "Georgia Southern":  { primary: "#041E42", secondary: "#A4A9AD" },
  "Georgia State":     { primary: "#0039A6", secondary: "#C8102E" },
  "James Madison":     { primary: "#450084", secondary: "#CBB677" },
  "Marshall":          { primary: "#00B140", secondary: "#000000" },
  "Old Dominion":      { primary: "#003057", secondary: "#9EA2A2" },
  "South Alabama":     { primary: "#BF0D3E", secondary: "#00205B" },
  "Southern Miss":     { primary: "#B9975B", secondary: "#000000" },
  "Troy":              { primary: "#8A2432", secondary: "#A7A8AA" },
  "Louisiana":         { primary: "#C8102E", secondary: "#000000" },
  "ULM":               { primary: "#7C2529", secondary: "#B9975B" },
  "Texas State":       { primary: "#7C2529", secondary: "#9D968D" },
  "Arkansas State":    { primary: "#CC0000", secondary: "#000000" },

  "Tulsa":            { primary: "#002D72", secondary: "#CFB53B" },
  "East Carolina":    { primary: "#592A8A", secondary: "#F3BD1B" },
  "USF":              { primary: "#006747", secondary: "#CFC493" },

  "Liberty":          { primary: "#0A2240", secondary: "#C8102E" },
  "New Mexico State": { primary: "#7C2529", secondary: "#000000" },
  "UMass":            { primary: "#881C1C", secondary: "#FFFFFF" },
  "Notre Dame":       { primary: "#0C2340", secondary: "#AE9142" },

  "Fresno State":     { primary: "#C41230", secondary: "#00285E" },
  "UTEP":             { primary: "#041E42", secondary: "#FF8200" },
};

// ---- Utility: normalize team name, look up colors, fallback to hash color ----
function normalizeTeamName(name) {
  if (!name) return "";
  const plain = name.replace(/\s+/g, " ").trim();
  const key = plain.toLowerCase();
  if (TEAM_ALIASES[key]) return TEAM_ALIASES[key];
  return plain;
}

function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!m) return null;
  return { r: parseInt(m[1],16), g: parseInt(m[2],16), b: parseInt(m[3],16) };
}

function getLuminance(hex) {
  const rgb = hexToRgb(hex);
  if (!rgb) return 0.5;
  // relative luminance (sRGB)
  const srgb = ["r","g","b"].map(k => {
    let v = rgb[k]/255;
    return v <= 0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4);
  });
  return 0.2126*srgb[0] + 0.7152*srgb[1] + 0.0722*srgb[2];
}

function readableTextColor(bgHex) {
  return getLuminance(bgHex) > 0.45 ? "#111111" : "#FFFFFF";
}

// Deterministic pleasant color for unknown teams (HSL hash)
function hashColor(name) {
  const s = (name || "X").toLowerCase();
  let h = 0;
  for (let i=0;i<s.length;i++) h = ((h<<5)-h) + s.charCodeAt(i);
  const hue = Math.abs(h) % 360;
  const sat = 65, light = 45;
  // Convert HSL to HEX
  function hslToHex(h, s, l) {
    s/=100; l/=100;
    const k = n => (n + h/30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = n => l - a * Math.max(-1, Math.min(k(n)-3, Math.min(9-k(n), 1)));
    const toHex = x => Math.round(255*x).toString(16).padStart(2,"0");
    return `#${toHex(f(0))}${toHex(f(8))}${toHex(f(4))}`;
  }
  return hslToHex(hue, sat, light);
}

function getTeamColors(team) {
  const t = normalizeTeamName(team);
  const entry = TEAM_COLORS[t];
  if (entry) return entry;
  const primary = hashColor(t);
  const secondary = readableTextColor(primary) === "#FFFFFF" ? "#1f2937" : "#FFFFFF";
  return { primary, secondary };
}

// Try to load external JSON mapping if present and merge (non-blocking)
(async function mergeExternalColors(){
  try {
    const res = await fetch("assets/team_colors.json", { cache: "no-store" });
    if (res.ok) {
      const extra = await res.json();
      for (const [k,v] of Object.entries(extra)) {
        TEAM_COLORS[k] = v;
      }
      console.log("Loaded external team_colors.json");
    }
  } catch (_) { /* optional file, ignore errors */ }
})();

// ===== Rendering =====
async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}
const pct = x => (x*100).toFixed(1) + "%";

function makeCard(game) {
  const home = normalizeTeamName(game.home);
  const away = normalizeTeamName(game.away);
  const pick = game.pick;
  // Pick colors from the picked team's palette for emphasis
  const picked = normalizeTeamName(pick);
  const { primary, secondary } = getTeamColors(picked);
  const loseTeam = picked === home ? away : home;
  const loseColors = getTeamColors(loseTeam);

  const card = document.createElement("article");
  card.className = "pick-card";
  card.style.setProperty("--card-bg", primary);
  card.style.setProperty("--card-border", secondary);
  card.style.setProperty("--card-text", readableTextColor(primary));
  card.style.setProperty("--chip-lose-bg", loseColors.primary);
  card.style.setProperty("--chip-lose-text", readableTextColor(loseColors.primary));

  card.innerHTML = `
    <div class="card-top">
      <div class="chip chip-pick" title="Predicted winner">
        <span class="chip-dot"></span>
        <span class="team-name">${picked}</span>
      </div>
      <div class="prob">${picked === home ? pct(game.home_prob) : pct(game.away_prob)}</div>
    </div>
    <div class="card-mid">
      <div class="vs">vs</div>
    </div>
    <div class="card-bottom">
      <div class="chip chip-lose" title="Opponent">
        <span class="team-name">${loseTeam}</span>
      </div>
      <div class="prob alt">${picked === home ? pct(game.away_prob) : pct(game.home_prob)}</div>
    </div>
  `;
  return card;
}

function render(pred) {
  const meta = document.getElementById("meta");
  meta.textContent =
    `Generated: ${pred.generated_at} • Model ${pred.model} • Test acc: ${pred.metric?.test_accuracy ?? "n/a"}` +
    (pred.unknown_teams?.length ? ` • Unknown: ${pred.unknown_teams.join(", ")}` : "");

  const container = document.getElementById("cards");
  container.innerHTML = "";
  pred.games.forEach(g => container.appendChild(makeCard(g)));
}

async function init() {
  try {
    const pred = await loadJSON("data/predictions.json");
    render(pred);
  } catch (e) {
    console.error(e);
    const container = document.getElementById("cards");
    container.innerHTML = `<p class="error">Could not load predictions.json (has the workflow run yet?).</p>`;
  }
}
document.getElementById("refresh").addEventListener("click", init);
init();
