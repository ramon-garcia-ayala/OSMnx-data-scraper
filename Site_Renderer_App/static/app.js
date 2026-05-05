/* ── Site Renderer ──────────────────────────────────────
   Interactive map for configuring & running the pipeline.
   ──────────────────────────────────────────────────────── */

// ── state ───────────────────────────────────────────────
let locations = [];
let nextId = 1;
let activeId = null;
let notebookDefs = [];
let enabledNotebooks = new Set(["01", "02", "03", "04", "05"]);
let csvStatus = {};
let pipelinePolling = false;

// ── theme ───────────────────────────────────────────────
function currentTheme() {
  return document.documentElement.getAttribute("data-theme") || "dark";
}

const TILE_URLS = {
  dark:  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  light: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
};
const MINI_TILE_URLS = {
  dark:  "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
  light: "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
};

let mainTileLayer, miniTileLayer;

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
  swapTiles(next);
}

function swapTiles(theme) {
  if (mainTileLayer) mainTileLayer.setUrl(TILE_URLS[theme]);
  if (miniTileLayer) miniTileLayer.setUrl(MINI_TILE_URLS[theme]);
}

// ── main map ────────────────────────────────────────────
const map = L.map("map", { zoomControl: false }).setView([40.7589, -73.9851], 13);
L.control.zoom({ position: "topright" }).addTo(map);

const theme = currentTheme();
mainTileLayer = L.tileLayer(TILE_URLS[theme], {
  attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
  maxZoom: 19,
}).addTo(map);

// ── mini-map ────────────────────────────────────────────
const minimap = L.map("minimap", {
  zoomControl: false, attributionControl: false,
  dragging: false, scrollWheelZoom: false,
  doubleClickZoom: false, boxZoom: false,
  keyboard: false, tap: false,
}).setView([20, 0], 1);

miniTileLayer = L.tileLayer(MINI_TILE_URLS[theme], { maxZoom: 6 }).addTo(minimap);

let minimapMarkers = L.layerGroup().addTo(minimap);

function updateMinimap() {
  minimapMarkers.clearLayers();
  locations.forEach((loc, i) => {
    const color = getColor(i);
    const m = L.circleMarker([loc.lat, loc.lon], {
      radius: 5, color, fillColor: color, fillOpacity: 1, weight: 1,
    });
    m.on("click", () => { map.setView([loc.lat, loc.lon], 14); setActive(loc.id); });
    m.bindTooltip(loc.name, { direction: "top", className: "minimap-tooltip" });
    minimapMarkers.addLayer(m);
  });
  if (locations.length > 0) {
    const group = L.featureGroup(locations.map((l) => L.marker([l.lat, l.lon])));
    minimap.fitBounds(group.getBounds().pad(1.5));
  }
}

// ── colours (softer Apple palette) ──────────────────────
const COLORS = ["#007AFF", "#34C759", "#FF9500", "#AF52DE", "#FF3B30", "#5AC8FA", "#FF2D55", "#64D2FF"];
function getColor(i) { return COLORS[i % COLORS.length]; }

// ── context popup ────────────────────────────────────────
let contextPopupLocId = null;

function showContextPopup(locId, screenX, screenY) {
  closeContextPopup();
  contextPopupLocId = locId;

  const loc = locations.find((l) => l.id === locId);
  if (!loc) return;
  const index = locations.indexOf(loc);
  const color = getColor(index);
  const radius = computeRadius(loc);

  const popup = document.createElement("div");
  popup.id = "ctx-popup";
  popup.innerHTML = `
    <div class="ctx-header">
      <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:10px;height:10px;border-radius:50%;background:${color};flex-shrink:0;"></div>
        <span class="ctx-name">${loc.name}</span>
      </div>
      <button class="ctx-close" onclick="closeContextPopup()">&#x2715;</button>
    </div>
    <div class="ctx-body">
      <div class="ctx-row"><span class="ctx-label">Coordinates</span><span class="ctx-value mono">${loc.lat.toFixed(6)}, ${loc.lon.toFixed(6)}</span></div>
      <div class="ctx-row"><span class="ctx-label">Walk radius</span><span class="ctx-value">${radius.toFixed(0)} m</span></div>
      <div class="ctx-row"><span class="ctx-label">Walk time</span><span class="ctx-value">${loc.walk_minutes} min</span></div>
      <div class="ctx-row"><span class="ctx-label">Walk speed</span><span class="ctx-value">${loc.walk_speed_m_min} m/min</span></div>
      <div class="ctx-row" id="ctx-geocode-row">
        <span class="ctx-label">Location</span>
        <span class="ctx-value" id="ctx-geocode-val"><span class="ctx-loading">Looking up...</span></span>
      </div>
    </div>
    <button class="ctx-delete-btn" onclick="removeLocationFromPopup(${locId})">Remove location</button>
  `;

  document.body.appendChild(popup);

  // position: keep inside viewport
  const pw = 260, ph = 220;
  const vw = window.innerWidth, vh = window.innerHeight;
  let x = screenX + 12, y = screenY - 10;
  if (x + pw > vw - 8) x = screenX - pw - 12;
  if (y + ph > vh - 8) y = vh - ph - 8;
  if (y < 8) y = 8;
  popup.style.left = x + "px";
  popup.style.top  = y + "px";

  // reverse geocode asynchronously
  reverseGeocode(loc.lat, loc.lon).then((place) => {
    const el = document.getElementById("ctx-geocode-val");
    if (el) el.textContent = place || "—";
  });
}

function closeContextPopup() {
  const el = document.getElementById("ctx-popup");
  if (el) el.remove();
  contextPopupLocId = null;
}

function removeLocationFromPopup(id) {
  closeContextPopup();
  removeLocation(id);
}

async function reverseGeocode(lat, lon) {
  try {
    // reuse the backend proxy but with a reverse query
    const res = await fetch(`/api/geocode?q=${lat},${lon}`);
    const results = await res.json();
    if (!results.length) return null;
    // extract city + country from display_name (last two meaningful parts)
    const parts = results[0].display_name.split(",").map((s) => s.trim());
    // take country (last) + one before it, avoiding duplicates
    const country = parts[parts.length - 1];
    const region  = parts.length >= 3 ? parts[parts.length - 3] : parts[0];
    return region !== country ? `${region}, ${country}` : country;
  } catch { return null; }
}

// ── helpers ─────────────────────────────────────────────
function computeRadius(loc) {
  return (loc.walk_minutes || 15) * (loc.walk_speed_m_min || 80);
}

function toast(msg, type = "success") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "show " + type;
  setTimeout(() => { el.className = ""; }, 3000);
}

// ── tabs ────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name)
  );
  document.querySelectorAll(".tab-content").forEach((c) =>
    c.classList.toggle("active", c.id === `tab-${name}`)
  );
  if (name === "pipeline") renderPipelineTab();
}

// ── city search ─────────────────────────────────────────
let searchTimeout = null;

document.getElementById("search-input").addEventListener("input", (e) => {
  clearTimeout(searchTimeout);
  const q = e.target.value.trim();
  if (q.length < 3) { document.getElementById("search-results").style.display = "none"; return; }
  searchTimeout = setTimeout(() => searchCity(q), 400);
});

document.getElementById("search-input").addEventListener("keydown", (e) => {
  if (e.key === "Escape") { document.getElementById("search-results").style.display = "none"; e.target.blur(); }
});

async function searchCity(query) {
  const container = document.getElementById("search-results");
  try {
    const res = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`);
    const results = await res.json();
    if (!results.length) {
      container.innerHTML = '<div class="search-result-item">No results found</div>';
      container.style.display = "block";
      return;
    }
    container.innerHTML = results.map((r) =>
      `<div class="search-result-item" onclick="flyToResult(${r.lat}, ${r.lon}, '${r.display_name.replace(/'/g, "\\'")}')">${r.display_name}</div>`
    ).join("");
    container.style.display = "block";
  } catch { container.style.display = "none"; }
}

function flyToResult(lat, lon, name) {
  map.flyTo([lat, lon], 14, { duration: 1.2 });
  document.getElementById("search-results").style.display = "none";
  document.getElementById("search-input").value = "";
  toast(`Navigated to ${name.substring(0, 40)}`);
}

document.addEventListener("click", (e) => {
  if (!e.target.closest("#search-box")) document.getElementById("search-results").style.display = "none";
  if (!e.target.closest("#ctx-popup")) closeContextPopup();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeContextPopup();
});

// ── marker icon builder ─────────────────────────────────
function buildMarkerIcon(index, color) {
  return L.divIcon({
    className: "custom-marker",
    html: `<div style="
      background:${color}; width:26px; height:26px; border-radius:50%;
      border:2.5px solid white; display:flex; align-items:center; justify-content:center;
      color:white; font-weight:700; font-size:11px; font-family:'Helvetica Neue',Helvetica,sans-serif;
      box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    ">${index + 1}</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
  });
}

// ── add location ────────────────────────────────────────
function addLocation(lat, lon, name, walk_minutes, walk_speed_m_min) {
  const id = nextId++;
  const index = locations.length;
  const color = getColor(index);

  const loc = {
    id,
    name: name || `location_${id}`,
    lat, lon,
    walk_minutes: walk_minutes ?? 15.0,
    walk_speed_m_min: walk_speed_m_min ?? 80.0,
  };

  const marker = L.marker([lat, lon], { icon: buildMarkerIcon(index, color), draggable: true }).addTo(map);
  const radius = computeRadius(loc);
  const circle = L.circle([lat, lon], {
    radius, color, fillColor: color, fillOpacity: 0.06, weight: 1.5, dashArray: "6 4",
  }).addTo(map);

  marker.on("dragend", () => {
    const pos = marker.getLatLng();
    loc.lat = Math.round(pos.lat * 1e6) / 1e6;
    loc.lon = Math.round(pos.lng * 1e6) / 1e6;
    circle.setLatLng(pos);
    closeContextPopup();
    renderSidebar();
    setActive(id);
    updateMinimap();
  });
  marker.on("click", () => { closeContextPopup(); setActive(id); });
  marker.on("contextmenu", (e) => {
    L.DomEvent.preventDefault(e);
    const containerPoint = map.latLngToContainerPoint(e.latlng);
    const mapRect = document.getElementById("map").getBoundingClientRect();
    showContextPopup(id, mapRect.left + containerPoint.x, mapRect.top + containerPoint.y);
  });

  loc.marker = marker;
  loc.circle = circle;
  locations.push(loc);

  renderSidebar();
  setActive(id);
  updateMinimap();
  return loc;
}

// ── remove / rebuild ────────────────────────────────────
function removeLocation(id) {
  const idx = locations.findIndex((l) => l.id === id);
  if (idx === -1) return;
  map.removeLayer(locations[idx].marker);
  map.removeLayer(locations[idx].circle);
  locations.splice(idx, 1);
  if (activeId === id) activeId = locations.length ? locations[0].id : null;
  rebuildMarkerIcons();
  renderSidebar();
  updateMinimap();
}

function rebuildMarkerIcons() {
  locations.forEach((loc, i) => {
    const color = getColor(i);
    loc.marker.setIcon(buildMarkerIcon(i, color));
    loc.circle.setStyle({ color, fillColor: color });
  });
}

// ── active ──────────────────────────────────────────────
function setActive(id) {
  activeId = id;
  document.querySelectorAll(".location-card").forEach((card) =>
    card.classList.toggle("active", card.dataset.id == id)
  );
  // highlight the active circle, dim the rest
  locations.forEach((loc, i) => {
    const isActive = loc.id === id;
    loc.circle.setStyle({
      fillOpacity: isActive ? 0.22 : 0.06,
      weight:      isActive ? 2.5  : 1.5,
      opacity:     isActive ? 1    : 0.55,
    });
    if (isActive) loc.circle.bringToFront();
  });
}

// ── update field ────────────────────────────────────────
function updateField(id, field, value) {
  const loc = locations.find((l) => l.id === id);
  if (!loc) return;

  if (field === "name") {
    loc[field] = value.replace(/\s+/g, "_").toLowerCase();
  } else {
    const num = parseFloat(value);
    if (isNaN(num) || num <= 0) return;
    loc[field] = num;
  }

  if (field === "lat" || field === "lon") {
    loc.marker.setLatLng([loc.lat, loc.lon]);
    loc.circle.setLatLng([loc.lat, loc.lon]);
    updateMinimap();
  }
  if (field === "walk_minutes" || field === "walk_speed_m_min") {
    loc.circle.setRadius(computeRadius(loc));
  }

  const radiusEl = document.querySelector(`.location-card[data-id="${id}"] .radius-display`);
  if (radiusEl) radiusEl.textContent = `${computeRadius(loc).toFixed(0)} m radius`;

  if (field === "lat" || field === "lon") {
    const coordEl = document.querySelector(`.location-card[data-id="${id}"] .coord-display`);
    if (coordEl) coordEl.textContent = `${loc.lat.toFixed(6)}, ${loc.lon.toFixed(6)}`;
  }
}

// ── render sidebar ──────────────────────────────────────
function renderSidebar() {
  const list = document.getElementById("location-list");

  if (locations.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="plus">+</div>
        <p>Click on the map to add a location</p>
        <p class="hint">Search for a city above, then click to place markers.</p>
      </div>`;
    return;
  }

  list.innerHTML = locations.map((loc, i) => {
    const color = getColor(i);
    const radius = computeRadius(loc);
    return `
    <div class="location-card ${loc.id === activeId ? "active" : ""}" data-id="${loc.id}" onclick="setActive(${loc.id})">
      <div class="location-card-header">
        <div style="display:flex; align-items:center; gap:10px;">
          <div class="location-index" style="background:${color}">${i + 1}</div>
          <input class="name-input" value="${loc.name}"
            onchange="updateField(${loc.id}, 'name', this.value)"
            onclick="event.stopPropagation()" />
        </div>
        <button class="btn-delete" onclick="event.stopPropagation(); removeLocation(${loc.id})" title="Remove">&times;</button>
      </div>
      <div class="coord-display">${loc.lat.toFixed(6)}, ${loc.lon.toFixed(6)}</div>
      <div class="field-row">
        <div class="field-group">
          <label>Walk minutes</label>
          <input type="number" min="1" step="1" value="${loc.walk_minutes}"
            onchange="updateField(${loc.id}, 'walk_minutes', this.value)"
            onclick="event.stopPropagation()" />
        </div>
        <div class="field-group">
          <label>Speed (m/min)</label>
          <input type="number" min="1" step="5" value="${loc.walk_speed_m_min}"
            onchange="updateField(${loc.id}, 'walk_speed_m_min', this.value)"
            onclick="event.stopPropagation()" />
        </div>
      </div>
      <div class="radius-display">${radius.toFixed(0)} m radius</div>
    </div>`;
  }).join("");

  updateCombineButton();
}

// ── pipeline tab ────────────────────────────────────────
function renderPipelineTab() {
  renderNotebookCheckboxes();
  renderPipelineSites();
  renderPipelineLog();
}

function renderNotebookCheckboxes() {
  const container = document.getElementById("notebook-checkboxes");
  if (!notebookDefs.length) { container.innerHTML = '<div style="color:var(--text-tertiary); font-size:12px">Loading...</div>'; return; }
  container.innerHTML = notebookDefs.map((nb) => {
    const checked = enabledNotebooks.has(nb.id) ? "checked" : "";
    const cls = nb.id === "06" ? " joker" : "";
    return `
    <label class="nb-checkbox${cls}">
      <input type="checkbox" ${checked} onchange="toggleNotebook('${nb.id}', this.checked)" />
      <span class="nb-id">${nb.id}</span>
      <span>${nb.label}</span>
    </label>`;
  }).join("");
}

function toggleNotebook(id, checked) {
  if (checked) enabledNotebooks.add(id); else enabledNotebooks.delete(id);
}

function renderPipelineSites(statusData) {
  const container = document.getElementById("pipeline-sites");
  if (!locations.length) {
    container.innerHTML = '<div class="empty-pipeline">Add locations on the map first.</div>';
    return;
  }

  const sites = statusData?.sites || {};
  container.innerHTML = locations.map((loc, i) => {
    const color = getColor(i);
    const siteStatus = sites[loc.name] || {};
    const hasCsv = !!csvStatus[loc.name];

    const steps = notebookDefs.map((nb) => {
      const st = siteStatus[nb.id] || "pending";
      return `<div class="nb-step ${st}" title="${nb.label}: ${st}">${nb.id}</div>`;
    }).join("");

    const isRunning = Object.values(siteStatus).some((s) => s === "running");
    const btnDisabled = statusData?.running ? "disabled" : "";

    return `
    <div class="pipeline-site-card">
      <div class="pipeline-site-header">
        <div class="pipeline-site-name" style="color:${color}">
          ${loc.name}
          <span class="csv-badge ${hasCsv ? "has-csv" : "no-csv"}">${hasCsv ? "CSV" : "no CSV"}</span>
        </div>
        <button class="btn-run-site" ${btnDisabled} onclick="runSite('${loc.name}')">
          ${isRunning ? "Running..." : "Run"}
        </button>
      </div>
      <div class="nb-steps">${steps}</div>
    </div>`;
  }).join("");
}

function renderPipelineLog(logLines) {
  const container = document.getElementById("pipeline-log");
  const lines = logLines || [];
  if (!lines.length) { container.innerHTML = '<span style="color:var(--text-tertiary)">No log output yet</span>'; return; }
  container.innerHTML = lines.map((l) => `<div class="log-line">${escapeHtml(l)}</div>`).join("");
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── map click ───────────────────────────────────────────
map.on("click", (e) => {
  if (contextPopupLocId !== null) { closeContextPopup(); return; }
  addLocation(Math.round(e.latlng.lat * 1e6) / 1e6, Math.round(e.latlng.lng * 1e6) / 1e6);
  toast(`Location #${locations.length} added`);
});

map.on("contextmenu", () => { /* suppress browser menu on map */ });

// ── API ─────────────────────────────────────────────────
async function loadLocations() {
  try {
    const res = await fetch("/api/locations");
    const data = await res.json();
    if (data.length) {
      data.forEach((loc) => addLocation(loc.lat, loc.lon, loc.name, loc.walk_minutes, loc.walk_speed_m_min));
      const group = L.featureGroup(locations.map((l) => l.marker));
      map.fitBounds(group.getBounds().pad(0.2));
      toast(`Loaded ${data.length} locations`);
    }
  } catch (err) { console.error("Failed to load locations:", err); }
}

async function loadNotebooks() {
  try { const res = await fetch("/api/notebooks"); notebookDefs = await res.json(); }
  catch (err) { console.error("Failed to load notebooks:", err); }
}

async function loadCsvStatus() {
  try { const res = await fetch("/api/csv-status"); csvStatus = await res.json(); updateCombineButton(); }
  catch (err) { console.error("Failed to load CSV status:", err); }
}

async function saveLocations() {
  const payload = locations.map((loc) => ({
    name: loc.name, lat: loc.lat, lon: loc.lon,
    walk_minutes: loc.walk_minutes, walk_speed_m_min: loc.walk_speed_m_min,
  }));
  try {
    const res = await fetch("/api/locations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await res.json();
    if (data.ok) toast(`Saved ${data.count} locations`); else toast(data.error || "Save failed", "error");
  } catch { toast("Network error", "error"); }
}

// ── pipeline actions ────────────────────────────────────
async function runSite(siteName) {
  await saveLocations();
  const notebooks = Array.from(enabledNotebooks);
  if (!notebooks.length) { toast("Select at least one notebook", "error"); return; }
  try {
    const res = await fetch("/api/run-site", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sites: [siteName], notebooks }) });
    const data = await res.json();
    if (data.error) { toast(data.error, "error"); return; }
    toast(`Pipeline started for ${siteName}`);
    showPipelineRunning(true); switchTab("pipeline"); startPolling();
  } catch { toast("Failed to start pipeline", "error"); }
}

async function runAll() {
  await saveLocations();
  if (!locations.length) { toast("No locations to run", "error"); return; }
  const notebooks = Array.from(enabledNotebooks);
  if (!notebooks.length) { toast("Select at least one notebook", "error"); return; }
  try {
    const res = await fetch("/api/run-all", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ notebooks }) });
    const data = await res.json();
    if (data.error) { toast(data.error, "error"); return; }
    toast(data.message); showPipelineRunning(true); switchTab("pipeline"); startPolling();
  } catch { toast("Failed to start pipeline", "error"); }
}

async function cancelPipeline() {
  try {
    const res = await fetch("/api/cancel", { method: "POST" });
    const data = await res.json();
    if (data.ok) toast("Cancel requested"); else toast(data.error || "Cancel failed", "error");
  } catch { toast("Failed to cancel", "error"); }
}

async function combineCSVs() {
  try {
    const res = await fetch("/api/combine", { method: "POST" });
    const data = await res.json();
    if (data.ok) toast(`Combined ${data.rows} rows from ${data.sites.length} sites`);
    else toast(data.error || "Combine failed", "error");
  } catch { toast("Failed to combine CSVs", "error"); }
}

function showPipelineRunning(running) {
  document.getElementById("btn-run-all").disabled = running;
  document.getElementById("btn-cancel").style.display = running ? "" : "none";
  document.querySelectorAll(".btn-run-site").forEach((b) => (b.disabled = running));
}

// ── polling ─────────────────────────────────────────────
function startPolling() { if (pipelinePolling) return; pipelinePolling = true; pollPipeline(); }

async function pollPipeline() {
  try {
    const res = await fetch("/api/pipeline-status");
    const data = await res.json();
    renderPipelineSites(data);
    renderPipelineLog(data.log);

    if (data.running) {
      showPipelineRunning(true);
      setTimeout(pollPipeline, 1500);
    } else {
      showPipelineRunning(false);
      pipelinePolling = false;
      const allSites = Object.values(data.sites || {});
      if (allSites.length) {
        const anyFailed = allSites.some((s) => Object.values(s).some((v) => v === "failed"));
        if (data.cancelled) toast("Pipeline cancelled", "error");
        else if (anyFailed) toast("Pipeline completed with errors", "error");
        else toast("Pipeline completed successfully!");
      }
      loadCsvStatus();
    }
  } catch { pipelinePolling = false; setTimeout(pollPipeline, 5000); }
}

function updateCombineButton() {
  const btn = document.getElementById("btn-combine");
  if (!locations.length) { btn.disabled = true; return; }
  btn.disabled = !locations.every((loc) => csvStatus[loc.name]);
}

// ── clear all ───────────────────────────────────────────
function clearAll() {
  if (!locations.length) return;
  if (!confirm("Remove all locations?")) return;
  [...locations].forEach((l) => { map.removeLayer(l.marker); map.removeLayer(l.circle); });
  locations = []; activeId = null; nextId = 1;
  renderSidebar(); updateMinimap();
  toast("All locations cleared");
}

// ── init ────────────────────────────────────────────────
async function init() {
  await loadNotebooks();
  await loadLocations();
  await loadCsvStatus();
  renderPipelineTab();

  try {
    const res = await fetch("/api/pipeline-status");
    const data = await res.json();
    if (data.running) { showPipelineRunning(true); startPolling(); switchTab("pipeline"); }
  } catch { /* ignore */ }
}

init();
