async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}
function pct(x) { return (x*100).toFixed(1) + "%"; }

function render(pred) {
  const meta = document.getElementById("meta");
  meta.textContent =
    `Generated: ${pred.generated_at} • Model ${pred.model} • Test acc: ${pred.metric?.test_accuracy ?? "n/a"}` +
    (pred.unknown_teams?.length ? ` • Unknown: ${pred.unknown_teams.join(", ")}` : "");

  const tbody = document.querySelector("#predTable tbody");
  tbody.innerHTML = "";
  for (const g of pred.games) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${g.home}</td>
      <td>${g.away}</td>
      <td>${pct(g.home_prob)}</td>
      <td>${pct(g.away_prob)}</td>
      <td><strong>${g.pick}</strong></td>`;
    tbody.appendChild(tr);
  }
}

async function init() {
  try {
    const pred = await loadJSON("data/predictions.json");
    render(pred);
  } catch (e) {
    console.error(e);
    alert("Could not load predictions.json (workflow run not finished yet?)");
  }
}
document.getElementById("refresh").addEventListener("click", init);
init();
