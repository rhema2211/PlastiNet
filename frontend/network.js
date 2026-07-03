/* network.js
 * Renders the live synaptic graph and talks to the Flask API.
 * Every train click -> POST /api/train -> re-render with grow/prune animations.
 */

const API = "http://127.0.0.1:5000/api";

const REGION_COLOR = {
  attention: "#4CE0D2",
  memory: "#8B7CF6",
  focus: "#F5A623",
};

let svg = d3.select("#graph");
const width = 900, height = 620;

let simulation = null;
let regionsByNode = {};   // "N3" -> "attention"
let currentMode = "child";

function nodeRegion(id) {
  return regionsByNode[id] || "attention";
}

// ------------------------------------------------------------------
// API calls
// ------------------------------------------------------------------
async function startSession(mode) {
  const res = await fetch(`${API}/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  const data = await res.json();

  regionsByNode = {};
  for (const [region, nodes] of Object.entries(data.regions)) {
    nodes.forEach(n => (regionsByNode[n] = region));
  }

  renderGraph(data.snapshot, [], []);
  updateReport(data.report);
  logLine(`session started — mode: ${mode}`, "log-muted");
}

async function trainRegion(region) {
  setButtonsEnabled(false);
  const res = await fetch(`${API}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ region }),
  });
  const data = await res.json();

  renderGraph(data.snapshot, data.grown, data.pruned);
  updateReport(data.report);

  if (data.grown.length) {
    logLine(`+ grew ${data.grown.length} synapse(s) in ${region}`, "log-grow");
  }
  if (data.pruned.length) {
    logLine(`− pruned ${data.pruned.length} synapse(s)`, "log-prune");
  }
  if (!data.grown.length && !data.pruned.length) {
    logLine(`trained ${region} — weights updated`, "log-muted");
  }
  setButtonsEnabled(true);
}

async function setMode(mode) {
  await fetch(`${API}/mode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  currentMode = mode;
  logLine(`switched to ${mode.toUpperCase()} plasticity mode`, "log-muted");
}

// ------------------------------------------------------------------
// Rendering
// ------------------------------------------------------------------
function renderGraph(snapshot, grownPairs, prunedPairs) {
  const grownSet = new Set(grownPairs.map(([a, b]) => `${a}->${b}`));
  const prunedSet = new Set(prunedPairs.map(([a, b]) => `${a}->${b}`));

  const nodes = snapshot.nodes.map(id => ({ id, region: nodeRegion(id) }));
  const links = snapshot.edges.map(e => ({
    source: e.source,
    target: e.target,
    weight: e.weight,
    key: `${e.source}->${e.target}`,
  }));

  svg.selectAll("*").remove();

  if (simulation) simulation.stop();
  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(110).strength(0.4))
    .force("charge", d3.forceManyBody().strength(-220))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide(28));

  const link = svg.append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", d => REGION_COLOR[nodeRegion(d.source.id || d.source)] || "#3A4356")
    .attr("stroke-width", d => 1 + d.weight * 5)
    .attr("stroke-opacity", d => 0.25 + d.weight * 0.6)
    .attr("class", d => (grownSet.has(d.key) ? "synapse-new" : ""))
    .attr("stroke-dasharray", d => (grownSet.has(d.key) ? "60 60" : null));

  const node = svg.append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("class", "neuron")
    .attr("r", 12)
    .attr("fill", d => REGION_COLOR[d.region])
    .call(drag(simulation));

  const label = svg.append("g")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .text(d => d.id)
    .attr("font-family", "IBM Plex Mono, monospace")
    .attr("font-size", 9)
    .attr("fill", "#0B0E14")
    .attr("font-weight", 600)
    .attr("text-anchor", "middle")
    .attr("dy", 3)
    .attr("pointer-events", "none");

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y);
  });
}

function drag(sim) {
  function started(event, d) {
    if (!event.active) sim.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }
  function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
  function ended(event, d) {
    if (!event.active) sim.alphaTarget(0);
    d.fx = null; d.fy = null;
  }
  return d3.drag().on("start", started).on("drag", dragged).on("end", ended);
}

// ------------------------------------------------------------------
// UI helpers
// ------------------------------------------------------------------
function updateReport(report) {
  for (const region of ["attention", "memory", "focus"]) {
    const val = report[region] ?? 0;
    document.getElementById(`bar-${region}`).style.width = `${Math.min(100, val * 100)}%`;
    document.getElementById(`val-${region}`).textContent = val.toFixed(3);
  }
}

function logLine(text, cls) {
  const log = document.getElementById("log");
  const line = document.createElement("div");
  line.className = `log-line ${cls || ""}`;
  line.textContent = `> ${text}`;
  log.prepend(line);
}

function setButtonsEnabled(enabled) {
  document.querySelectorAll(".train-btn").forEach(b => (b.disabled = !enabled));
}

// ------------------------------------------------------------------
// Wire up controls
// ------------------------------------------------------------------
document.querySelectorAll(".train-btn").forEach(btn => {
  btn.addEventListener("click", () => trainRegion(btn.dataset.region));
});

document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    setMode(btn.dataset.mode);
  });
});

document.getElementById("reset-btn").addEventListener("click", () => {
  startSession(currentMode);
});

// initial load
startSession("child").catch(() => {
  logLine("could not reach backend — is app.py running on :5000?", "log-prune");
});
