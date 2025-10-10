// docs/analytics.js
(async function () {
  const $ = (sel) => document.querySelector(sel);
  const teamSelect = $("#teamSelect");
  const teamMeta   = $("#teamMeta");

  // Data sources
  const TEAM_STATS_URL = "data/team_stats.json";
  const CFRANK_URL     = "data/cfbrank.json";  // optional
  const FPI_URL        = "data/fpi.json";      // optional fallback / context if you keep it updated elsewhere

  // Load JSON (ignore missing optional files)
  async function safeFetchJSON(url) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch {
      return null;
    }
  }

  const teamStats = await safeFetchJSON(TEAM_STATS_URL) || [];
  const cfRankRaw = await safeFetchJSON(CFRANK_URL);
  const fpiRaw    = await safeFetchJSON(FPI_URL);

  // Build reverse rank lookup if present
  let modelRanks = {};
  if (cfRankRaw) {
    const latestYear = Object.keys(cfRankRaw).sort((a,b)=>(+a)-(+b)).pop();
    if (latestYear) {
      for (const [rank, team] of Object.entries(cfRankRaw[latestYear])) {
        modelRanks[team] = +rank;
      }
    }
  }

  // Helper: get rank/FPI from sources or from pre-attached fields
  function labelForTeam(rec) {
    const team = rec.team;
    const rank = rec.model_rank ?? modelRanks[team];
    const fpi  = rec.fpi?.fpi;

    let label = team;
    if (rank) label = `#${rank} ${label}`;
    if (typeof fpi === "number") label = `${label} — FPI: ${fpi.toFixed(2)}`;
    return label;
  }

  // Populate team select (sorted by model rank first, fallback alpha)
  const sorted = [...teamStats].sort((a, b) => {
    const ra = a.model_rank ?? modelRanks[a.team] ?? 9999;
    const rb = b.model_rank ?? modelRanks[b.team] ?? 9999;
    if (ra !== rb) return ra - rb;
    return a.team.localeCompare(b.team);
  });

  for (const rec of sorted) {
    const opt = document.createElement("option");
    opt.value = rec.team;
    opt.textContent = labelForTeam(rec);
    teamSelect.appendChild(opt);
  }

  // Charts
  let radarChart = null;
  let barChart   = null;

  function destroyCharts() {
    if (radarChart) { radarChart.destroy(); radarChart = null; }
    if (barChart)   { barChart.destroy();   barChart   = null; }
  }

  function z(v) {
    return (typeof v === "number" && isFinite(v)) ? v : 0;
  }

  // Normalize into [0, 1] for radar
  function normalize(val, min, max) {
    if (!isFinite(val) || !isFinite(min) || !isFinite(max) || max <= min) return 0;
    return (val - min) / (max - min);
  }

  // Collect offense metrics for radar
  function offenseMetrics(rec) {
    const s = rec.simple || {};
    const a = rec.advanced || {};

    // Common keys (best effort—CFBD naming to_dict)
    const ppg   = s["offense__points_per_game"] ?? s["offense__points"] ?? null;
    const ypg   = s["offense__yards_per_game"] ?? s["offense__total_yards"] ?? null;
    const ypp   = s["offense__yards_per_play"] ?? null;

    const sr    = a["offense__success_rate"] ?? null;
    const expl  = a["offense__explosiveness"] ?? null;
    const ppa   = a["offense__ppa"] ?? null;

    return { ppg:z(ppg), ypg:z(ypg), ypp:z(ypp), sr:z(sr), expl:z(expl), ppa:z(ppa) };
  }

  // Build global mins/maxes for normalization across all teams
  const mins = { ppg:Infinity, ypg:Infinity, ypp:Infinity, sr:Infinity, expl:Infinity, ppa:Infinity };
  const maxs = { ppg:-Infinity, ypg:-Infinity, ypp:-Infinity, sr:-Infinity, expl:-Infinity, ppa:-Infinity };
  for (const rec of teamStats) {
    const m = offenseMetrics(rec);
    for (const k of Object.keys(mins)) {
      mins[k] = Math.min(mins[k], m[k]);
      maxs[k] = Math.max(maxs[k], m[k]);
    }
  }

  function updateCharts(teamName) {
    const rec = teamStats.find(r => r.team === teamName);
    if (!rec) return;

    // Header meta
    teamMeta.textContent = labelForTeam(rec);

    // Radar (offense profile)
    const m = offenseMetrics(rec);
    const labels = ["PPG","Yards/G","Yards/Play","Success Rate","Explosiveness","PPA"];
    const rawVals = [m.ppg, m.ypg, m.ypp, m.sr, m.expl, m.ppa];
    const normVals = [
      normalize(m.ppg,  mins.ppg,  maxs.ppg),
      normalize(m.ypg,  mins.ypg,  maxs.ypg),
      normalize(m.ypp,  mins.ypp,  maxs.ypp),
      normalize(m.sr,   mins.sr,   maxs.sr),
      normalize(m.expl, mins.expl, maxs.expl),
      normalize(m.ppa,  mins.ppa,  maxs.ppa),
    ];

    destroyCharts();

    const ctx1 = document.getElementById("radarOffense").getContext("2d");
    radarChart = new Chart(ctx1, {
      type: "radar",
      data: {
        labels,
        datasets: [{
          label: "Offense (normalized)",
          data: normVals
        }]
      },
      options: {
        responsive: true,
        scales: { r: { beginAtZero: true, max: 1 } },
        plugins: { legend: { display: false } }
      }
    });

    // Bar (off vs def efficiency)
    const a = rec.advanced || {};
    const off_ppa  = z(a["offense__ppa"]);
    const def_ppa  = z(a["defense__ppa"]);
    const off_sr   = z(a["offense__success_rate"]);
    const def_sr   = z(a["defense__success_rate"]);
    const off_expl = z(a["offense__explosiveness"]);
    const def_expl = z(a["defense__explosiveness"]);

    const ctx2 = document.getElementById("barEff").getContext("2d");
    barChart = new Chart(ctx2, {
      type: "bar",
      data: {
        labels: ["PPA", "Success Rate", "Explosiveness"],
        datasets: [
          { label: "Offense", data: [off_ppa, off_sr, off_expl] },
          { label: "Defense", data: [def_ppa, def_sr, def_expl] }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { position: "top" } },
        scales: { y: { beginAtZero: true } }
      }
    });
  }

  // Initialize dropdown + charts
  if (sorted.length > 0) {
    updateCharts(sorted[0].team);
  }
  teamSelect.addEventListener("change", (e) => updateCharts(e.target.value));
})();
