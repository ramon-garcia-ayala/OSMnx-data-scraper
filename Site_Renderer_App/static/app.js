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
let _pipelineStartTime = null;
let osmTags = null;
let osmPresets = {};
let visibleColumns = null; // null = all columns visible
let amenityClusterGroup = null;
let currentRouteLayer = null;
let csvSortCol = null;
let csvSortDir = "asc";

// ── label colors (matching CSS) ─────────────────────────
const LABEL_COLORS = {
  food_drink:            "#FF9F0A",
  retail:                "#0A84FF",
  personal_care:         "#FF375F",
  health:                "#FF453A",
  finance:               "#30D158",
  hospitality:           "#BF5AF2",
  office_professional:   "#64D2FF",
  leisure_entertainment: "#FFD60A",
};

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

// Auto-detect OS theme changes
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  if (!localStorage.getItem("theme")) {
    const t = e.matches ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
    swapTiles(t);
  }
});

function swapTiles(theme) {
  if (mainTileLayer) mainTileLayer.setUrl(TILE_URLS[theme]);
  if (miniTileLayer) miniTileLayer.setUrl(MINI_TILE_URLS[theme]);
}

// ── main map ────────────────────────────────────────────
const map = L.map("map", { zoomControl: false }).setView([40.7589, -73.9851], 13);
L.control.zoom({ position: "topright" }).addTo(map);

// ── fit-all button (above zoom controls) ────────────────
const FitAllControl = L.Control.extend({
  options: { position: "topright" },
  onAdd() {
    const btn = L.DomUtil.create("div", "leaflet-bar leaflet-control fit-all-control");
    btn.innerHTML = `<a href="#" title="Fit all sites" role="button" aria-label="Fit all sites">⊡</a>`;
    L.DomEvent.disableClickPropagation(btn);
    btn.querySelector("a").addEventListener("click", (e) => {
      e.preventDefault();
      fitAllSites();
    });
    return btn;
  },
});
new FitAllControl().addTo(map);

function fitAllSites() {
  if (!locations.length) { toast("No locations to fit", "error"); return; }
  const group = L.featureGroup(locations.map((l) => l.circle));
  map.fitBounds(group.getBounds().pad(0.1));
}

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

// ── panel collapse/expand ───────────────────────────────
function togglePanel(side) {
  const panel = document.getElementById(`${side}-panel`);
  const btn   = document.getElementById(`${side}-panel-open`);
  const collapsed = panel.classList.toggle("collapsed");
  btn.style.display = collapsed ? "" : "none";
}

// ── context popup (now LEFT-click) ──────────────────────
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

  const pw = 260, ph = 220;
  const vw = window.innerWidth, vh = window.innerHeight;
  let x = screenX + 12, y = screenY - 10;
  if (x + pw > vw - 8) x = screenX - pw - 12;
  if (y + ph > vh - 8) y = vh - ph - 8;
  if (y < 8) y = 8;
  popup.style.left = x + "px";
  popup.style.top  = y + "px";

  reverseGeocode(loc.lat, loc.lon).then((place) => {
    const el = document.getElementById("ctx-geocode-val");
    if (el) el.textContent = place || "\u2014";
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
    const res = await fetch(`/api/geocode?q=${lat},${lon}`);
    const results = await res.json();
    if (!results.length) return null;
    const parts = results[0].display_name.split(",").map((s) => s.trim());
    // return neighborhood/locality + city (first 2-3 meaningful parts)
    if (parts.length >= 4) return `${parts[1]}, ${parts[2]}, ${parts[parts.length - 1]}`;
    if (parts.length >= 2) return `${parts[0]}, ${parts[parts.length - 1]}`;
    return parts[0];
  } catch { return null; }
}

// ── helpers ─────────────────────────────────────────────
function computeRadius(loc) {
  return (loc.walk_minutes || 15) * (loc.walk_speed_m_min || 80);
}

function csvParamsMatch(loc, params) {
  if (!params) return false;
  return (
    params.lat              === loc.lat &&
    params.lon              === loc.lon &&
    params.walk_minutes     === loc.walk_minutes &&
    params.walk_speed_m_min === loc.walk_speed_m_min
  );
}

let _toastTimer = null;

function toast(msg, type = "success") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "show " + type;

  // Error toasts are clickable — copy text and confirm
  if (type === "error") {
    el.classList.add("copyable");
    el.onclick = () => {
      navigator.clipboard.writeText(msg).catch(() => {
        const ta = document.createElement("textarea");
        ta.value = msg; ta.style.cssText = "position:fixed;opacity:0";
        document.body.appendChild(ta); ta.select();
        document.execCommand("copy"); ta.remove();
      });
      el.textContent = "Copied!";
      el.classList.remove("copyable");
      el.onclick = null;
      clearTimeout(_toastTimer);
      _toastTimer = setTimeout(() => { el.className = ""; }, 1200);
    };
  } else {
    el.classList.remove("copyable");
    el.onclick = null;
  }

  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.className = ""; el.onclick = null; }, 3000);
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── left tabs ───────────────────────────────────────────
function switchLeftTab(name) {
  document.querySelectorAll("#left-tabs .tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name)
  );
  ["locations", "osm-tags", "settings"].forEach((id) => {
    const el = document.getElementById(`tab-${id}`);
    if (el) el.classList.toggle("active", id === name);
  });
  if (name === "osm-tags") renderOsmTagsEditor();
  if (name === "settings") renderColumnSelector();
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
  if (!e.target.closest("#ctx-popup") && !e.target.closest(".custom-marker")) closeContextPopup();
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
    reverseGeocode(loc.lat, loc.lon).then((place) => {
      loc.neighborhood = place;
      renderSidebar();
    });
  });

  // LEFT-click for context popup (item #6)
  marker.on("click", (e) => {
    L.DomEvent.stopPropagation(e);
    const containerPoint = map.latLngToContainerPoint(e.latlng);
    const mapRect = document.getElementById("map").getBoundingClientRect();
    showContextPopup(id, mapRect.left + containerPoint.x, mapRect.top + containerPoint.y);
    setActive(id);
  });

  loc.marker = marker;
  loc.circle = circle;
  loc.neighborhood = null;
  locations.push(loc);

  renderSidebar();
  setActive(id);
  updateMinimap();

  // fetch neighborhood info
  reverseGeocode(lat, lon).then((place) => {
    loc.neighborhood = place;
    renderSidebar();
  });

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
  loadAmenityMarkers();
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
  locations.forEach((loc) => {
    const isActive = loc.id === id;
    loc.circle.setStyle({
      fillOpacity: isActive ? 0.22 : 0.06,
      weight:      isActive ? 2.5  : 1.5,
      opacity:     isActive ? 1    : 0.55,
    });
    if (isActive) loc.circle.bringToFront();
  });
  // show amenity markers for active site only
  loadAmenityMarkers();
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

// ── bulk apply to all (item #3) ─────────────────────────
function applyToAll(field) {
  const active = locations.find((l) => l.id === activeId);
  if (!active || locations.length < 2) return;
  const val = active[field];
  locations.forEach((loc) => {
    loc[field] = val;
    if (field === "walk_minutes" || field === "walk_speed_m_min") {
      loc.circle.setRadius(computeRadius(loc));
    }
  });
  renderSidebar();
  toast(`Applied ${field.replace(/_/g, " ")} = ${val} to all sites`);
}

// ── zoom to site perimeter ──────────────────────────────
function zoomToSite(id) {
  const loc = locations.find((l) => l.id === id);
  if (!loc) return;
  setActive(id);
  map.fitBounds(loc.circle.getBounds().pad(0.1));
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
    const isActive = loc.id === activeId;
    return `
    <div class="location-card ${isActive ? "active" : ""}" data-id="${loc.id}" onclick="setActive(${loc.id})">
      <div class="location-card-header">
        <div style="display:flex; align-items:center; gap:10px;">
          <div class="location-index" style="background:${color}" onclick="event.stopPropagation(); zoomToSite(${loc.id})">${i + 1}</div>
          <input class="name-input" value="${loc.name}"
            onchange="updateField(${loc.id}, 'name', this.value)"
            onclick="event.stopPropagation()" />
        </div>
        <button class="btn-delete" onclick="event.stopPropagation(); removeLocation(${loc.id})" title="Remove">&times;</button>
      </div>
      <div class="coord-display">${loc.lat.toFixed(6)}, ${loc.lon.toFixed(6)}</div>
      ${loc.neighborhood ? `<div class="neighborhood-display">${escapeHtml(loc.neighborhood)}</div>` : ""}
      <div class="field-row">
        <div class="field-group">
          <label>Walk minutes ${isActive && locations.length > 1 ? `<a class="apply-all-link" onclick="event.stopPropagation(); applyToAll('walk_minutes')">Apply to all</a>` : ""}</label>
          <input type="number" min="1" step="1" value="${loc.walk_minutes}"
            onchange="updateField(${loc.id}, 'walk_minutes', this.value)"
            onclick="event.stopPropagation()" />
        </div>
        <div class="field-group">
          <label>Speed (m/min) ${isActive && locations.length > 1 ? `<a class="apply-all-link" onclick="event.stopPropagation(); applyToAll('walk_speed_m_min')">Apply to all</a>` : ""}</label>
          <input type="number" min="1" step="5" value="${loc.walk_speed_m_min}"
            onchange="updateField(${loc.id}, 'walk_speed_m_min', this.value)"
            onclick="event.stopPropagation()" />
        </div>
      </div>
      <div class="radius-display">${radius.toFixed(0)} m radius</div>
    </div>`;
  }).join("");

  renderPipelineSites();
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
    const plutoTag = nb.needs_pluto
      ? `<span class="pluto-tag" title="${nb.pluto_available ? 'PLUTO data found' : 'PLUTO not found — will be skipped'}">${nb.pluto_available ? 'NYC' : 'NYC only'}</span>`
      : "";
    return `
    <label class="nb-checkbox${cls}">
      <input type="checkbox" ${checked} onchange="toggleNotebook('${nb.id}', this.checked)" />
      <span class="nb-id">${nb.id}</span>
      <span>${nb.label}</span>
      ${plutoTag}
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
    const csvInfo = csvStatus[loc.name];
    const hasCsv  = !!csvInfo;
    const paramsMatch = hasCsv && csvParamsMatch(loc, csvInfo.params);

    const effectiveStatus = (nb) => {
      if (siteStatus[nb.id]) return siteStatus[nb.id];
      if (paramsMatch) return "done";
      return "pending";
    };

    const steps = notebookDefs.map((nb) => {
      const st = effectiveStatus(nb);
      return `<div class="nb-step ${st}" title="${nb.label}: ${st}">${nb.id}</div>`;
    }).join("");

    const isRunning = Object.values(siteStatus).some((s) => s === "running");
    const btnDisabled = statusData?.running ? "disabled" : "";
    const csvBadge = hasCsv
      ? `<span class="csv-badge has-csv" title="${paramsMatch ? "Up to date" : "Params changed"}">${paramsMatch ? "\u2713 CSV" : "\u26A0 CSV"}</span>`
      : `<span class="csv-badge no-csv">no CSV</span>`;

    return `
    <div class="pipeline-site-card">
      <div class="pipeline-site-header">
        <div class="pipeline-site-name" style="color:${color}">
          ${loc.name}
          ${csvBadge}
        </div>
        <div style="display:flex;gap:5px;align-items:center;">
          ${hasCsv ? `<button class="btn-delete-csv" onclick="deleteSiteCsv('${loc.name}')" title="Delete CSV">&times;</button>` : ""}
          ${hasCsv ? `<button class="btn-view-csv" onclick="openCsvModal('${loc.name}')" title="Open CSV">&#8599;</button>` : ""}
          <button class="btn-run-site" ${btnDisabled} onclick="runSite('${loc.name}')">
            ${isRunning ? "Running..." : "Run"}
          </button>
        </div>
      </div>
      <div class="nb-steps">${steps}</div>
    </div>`;
  }).join("");
}

function renderPipelineLog(logLines) {
  const container = document.getElementById("pipeline-log");
  const copyBtn   = document.getElementById("btn-copy-log");
  const lines = logLines || [];

  if (!lines.length) {
    container.innerHTML = '<span style="color:var(--text-tertiary)">No log output yet</span>';
    if (copyBtn) copyBtn.style.display = "none";
    return;
  }

  const hasError = lines.some((l) => /FAILED|ERROR|Traceback|Exception/i.test(l));
  if (copyBtn) copyBtn.style.display = hasError ? "" : "none";

  container.innerHTML = lines.map((l) => {
    let cls = "log-line";
    if (/FAILED|ERROR|Traceback|Exception/i.test(l)) cls += " log-error";
    else if (/ OK$| done$/i.test(l) || /successfully|COMPLETE/i.test(l)) cls += " log-ok";
    else if (/starting\.\.\.|running/i.test(l)) cls += " log-running";
    return `<div class="${cls}">${escapeHtml(l)}</div>`;
  }).join("");

  container.scrollTop = container.scrollHeight;
}

function copyLogError() {
  const lines = Array.from(document.querySelectorAll("#pipeline-log .log-line"))
    .map((el) => el.textContent)
    .join("\n");
  navigator.clipboard.writeText(lines).then(
    () => toast("Log copied to clipboard"),
    () => {
      // fallback for browsers without clipboard API
      const ta = document.createElement("textarea");
      ta.value = lines;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      toast("Log copied to clipboard");
    }
  );
}

// ── edit mode ───────────────────────────────────────────
let editMode = false;

function toggleEditMode() {
  editMode = !editMode;
  const btn = document.getElementById("btn-edit-mode");
  btn.classList.toggle("active", editMode);
  document.getElementById("map").style.cursor = editMode ? "crosshair" : "";
}

// ── map click ───────────────────────────────────────────
map.on("click", (e) => {
  if (contextPopupLocId !== null) { closeContextPopup(); return; }
  if (!editMode) return;
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
  try { const res = await fetch("/api/csv-status"); csvStatus = await res.json(); }
  catch (err) { console.error("Failed to load CSV status:", err); }
}

async function loadOsmTags() {
  try { const res = await fetch("/api/osm-tags"); osmTags = await res.json(); }
  catch (err) { console.error("Failed to load OSM tags:", err); }
}

async function loadOsmPresets() {
  try { const res = await fetch("/api/osm-tag-presets"); osmPresets = await res.json(); }
  catch (err) { console.error("Failed to load OSM presets:", err); }
}

async function deleteSiteCsv(siteName) {
  if (!confirm(`Delete CSV for "${siteName}"?`)) return;
  try {
    const res = await fetch("/api/csv-delete-site", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ site: siteName }),
    });
    const data = await res.json();
    if (data.ok) {
      toast(`CSV deleted for ${siteName}`);
      await loadCsvStatus();
      renderPipelineSites();
      loadAmenityMarkers();
    } else { toast(data.error || "Delete failed", "error"); }
  } catch { toast("Failed to delete CSV", "error"); }
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
    showPipelineRunning(true); startPolling();
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
    toast(data.message); showPipelineRunning(true); startPolling();
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
    if (data.ok) {
      toast(`Combined ${data.rows} rows from ${data.sites.length} sites`);
      await loadCombinedCsvInfo();
    } else {
      toast(data.error || "Combine failed", "error");
    }
  } catch { toast("Failed to combine CSVs", "error"); }
}

function showPipelineRunning(running) {
  document.getElementById("btn-run-all").disabled = running;
  document.getElementById("btn-cancel").style.display = running ? "" : "none";
  document.querySelectorAll(".btn-run-site").forEach((b) => (b.disabled = running));
}

// ── time helpers ─────────────────────────────────────────
function _fmtSecs(s) {
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60), r = s % 60;
  return r ? `${m}m ${r}s` : `${m}m`;
}

function _etaLabel(startTime, done, remaining) {
  if (!startTime) return "";
  const elapsed = (Date.now() - startTime) / 1000;
  const elapsedStr = `${_fmtSecs(elapsed)} elapsed`;
  if (done > 0 && remaining > 0) {
    const eta = (elapsed / done) * remaining;
    return `${elapsedStr} · ~${_fmtSecs(eta)} remaining`;
  }
  return elapsedStr;
}

// ── pipeline progress bar ────────────────────────────────
function renderPipelineProgress(statusData) {
  const section = document.getElementById("pipeline-progress-section");
  if (!section) return;

  const sites = statusData?.sites || {};
  if (!Object.keys(sites).length) { section.style.display = "none"; return; }

  let done = 0, running = 0, failed = 0, pending = 0;
  Object.values(sites).forEach((site) => {
    Object.values(site).forEach((st) => {
      if      (st === "done")    done++;
      else if (st === "running") running++;
      else if (st === "failed")  failed++;
      else if (st === "pending") pending++;
    });
  });

  const total = done + running + failed + pending;
  if (!total) { section.style.display = "none"; return; }

  section.style.display = "";
  const donePct    = (done    / total * 100).toFixed(1);
  const runningPct = (running / total * 100).toFixed(1);
  const failedPct  = (failed  / total * 100).toFixed(1);

  document.getElementById("prog-done").style.width    = donePct    + "%";
  document.getElementById("prog-running").style.width = runningPct + "%";
  document.getElementById("prog-failed").style.width  = failedPct  + "%";

  const parts = [`${done}/${total} steps`];
  if (running) parts.push(`<span class="prog-lbl-running">${running} running</span>`);
  if (failed)  parts.push(`<span class="prog-lbl-failed">${failed} failed</span>`);
  if (pending) parts.push(`<span class="prog-lbl-pending">${pending} pending</span>`);
  const eta = statusData.running ? _etaLabel(_pipelineStartTime, done, pending) : "";
  const etaLine = eta ? `<div class="prog-lbl-eta">${eta}</div>` : "";
  document.getElementById("pipeline-progress-label").innerHTML = parts.join(" · ") + etaLine;
}

// ── polling ─────────────────────────────────────────────
function startPolling() { if (pipelinePolling) return; pipelinePolling = true; _pipelineStartTime = Date.now(); pollPipeline(); }

async function pollPipeline() {
  try {
    const res = await fetch("/api/pipeline-status");
    const data = await res.json();
    renderPipelineSites(data);
    renderPipelineLog(data.log);
    renderPipelineProgress(data);

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
        else {
          toast("Pipeline completed successfully!");
          const chk = document.getElementById("chk-auto-combine");
          if (chk && chk.checked) combineCSVs();
        }
      }
      await loadCsvStatus();
      renderPipelineSites(data);
    }
  } catch { pipelinePolling = false; setTimeout(pollPipeline, 5000); }
}


// ── CSV modal (server-side pagination + sorting + label colors) ──
let csvModalMeta = null;
let csvCurrentPage = 0;

async function openCsvModal(siteName) {
  csvCurrentPage = 0;
  csvModalMeta = null;
  csvSortCol = null;
  csvSortDir = "asc";
  _showCsvOverlay(siteName, null, null);
  await _fetchAndRenderCsvPage(siteName, 0);
}

async function _fetchAndRenderCsvPage(siteName, page) {
  try {
    let url = `/api/csv-site/${encodeURIComponent(siteName)}?page=${page}&size=50`;
    if (csvSortCol) url += `&sort=${encodeURIComponent(csvSortCol)}&dir=${csvSortDir}`;
    const res = await fetch(url);
    if (!res.ok) { toast("Failed to load CSV", "error"); closeCsvModal(); return; }
    const data = await res.json();
    if (data.error) { toast(data.error, "error"); closeCsvModal(); return; }
    csvModalMeta = data;
    csvCurrentPage = data.page;
    _showCsvOverlay(siteName, data, data.data);
  } catch (err) {
    console.error(err);
    toast("Failed to load CSV", "error");
    closeCsvModal();
  }
}

function csvSort(col) {
  if (csvSortCol === col) {
    csvSortDir = csvSortDir === "asc" ? "desc" : "asc";
  } else {
    csvSortCol = col;
    csvSortDir = "asc";
  }
  csvCurrentPage = 0;
  if (csvModalMeta) {
    _showCsvOverlay(csvModalMeta.site, null, null);
    _fetchAndRenderCsvPage(csvModalMeta.site, 0);
  }
}

function _showCsvOverlay(siteName, meta, rows) {
  const old = document.getElementById("csv-modal-overlay");
  if (old) old.remove();

  const overlay = document.createElement("div");
  overlay.id = "csv-modal-overlay";
  overlay.addEventListener("click", (e) => { if (e.target === overlay) closeCsvModal(); });

  if (!meta) {
    overlay.innerHTML = `
      <div class="csv-modal">
        <div class="csv-modal-header">
          <span class="csv-modal-title">${escapeHtml(siteName)}</span>
          <button class="ctx-close" onclick="closeCsvModal()">&#x2715;</button>
        </div>
        <div class="csv-modal-loading">Loading\u2026</div>
      </div>`;
    document.body.appendChild(overlay);
    return;
  }

  const { file, rows: totalRows, total_pages, page, columns } = meta;
  const rowStart = page * 50 + 1;
  const rowEnd   = Math.min((page + 1) * 50, totalRows);

  const headerCells = columns.map((c) => {
    const isSort = csvSortCol === c;
    const arrow = isSort
      ? `<span class="sort-arrow active">${csvSortDir === "asc" ? "\u25B2" : "\u25BC"}</span>`
      : `<span class="sort-arrow">\u25B2</span>`;
    return `<th onclick="csvSort('${escapeHtml(c)}')">${escapeHtml(String(c))} ${arrow}</th>`;
  }).join("");

  const labelIdx = columns.indexOf("label");
  const latIdx = columns.indexOf("lat");
  const lonIdx = columns.indexOf("lon");
  const bodyRows = rows.map((row) => {
    const hasCoords = row["lat"] != null && row["lon"] != null;
    const clickAttr = hasCoords
      ? ` class="csv-row-clickable" onclick="csvRowNavigate(${row["lat"]}, ${row["lon"]}, '${escapeHtml(String(row["name"] || ""))}')"`
      : "";
    return `<tr${clickAttr}>${columns.map((c, ci) => {
      const v = row[c];
      if (v === null || v === undefined) return `<td><span class="null-val">\u2014</span></td>`;
      const str = escapeHtml(String(v));
      if (ci === labelIdx && v) {
        return `<td class="label-${v}">${str}</td>`;
      }
      return `<td>${str}</td>`;
    }).join("")}</tr>`;
  }).join("");

  overlay.innerHTML = `
    <div class="csv-modal">
      <div class="csv-modal-header">
        <div>
          <span class="csv-modal-title">${escapeHtml(siteName)}</span>
          <span class="csv-modal-meta">${totalRows.toLocaleString()} rows &middot; ${columns.length} columns &middot; ${escapeHtml(file)}</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <a class="btn-csv-download" href="/api/csv-site/${encodeURIComponent(siteName)}/download" download="${escapeHtml(file)}">Download CSV</a>
          <button class="ctx-close" onclick="closeCsvModal()">&#x2715;</button>
        </div>
      </div>
      <div class="csv-table-wrap">
        <table class="csv-table">
          <thead><tr>${headerCells}</tr></thead>
          <tbody>${bodyRows}</tbody>
        </table>
      </div>
      <div class="csv-modal-footer">
        <div style="display:flex;align-items:center;gap:6px;">
          <span class="csv-page-info">Page</span>
          <input type="number" class="csv-page-input" id="csv-page-input" value="${page + 1}" min="1" max="${total_pages}" onkeydown="if(event.key==='Enter')csvGoToPage()" />
          <span class="csv-page-info">of ${total_pages} &nbsp;&middot;&nbsp; rows ${rowStart}\u2013${rowEnd}</span>
        </div>
        <div style="display:flex;gap:6px;">
          <button class="btn-page" onclick="csvPageNav(-1)" ${page === 0 ? "disabled" : ""}>&larr; Prev</button>
          <button class="btn-page" onclick="csvPageNav(1)"  ${page >= total_pages - 1 ? "disabled" : ""}>Next &rarr;</button>
        </div>
      </div>
    </div>`;

  document.body.appendChild(overlay);
}

async function csvPageNav(dir) {
  if (!csvModalMeta) return;
  const next = csvCurrentPage + dir;
  if (next < 0 || next >= csvModalMeta.total_pages) return;
  _showCsvOverlay(csvModalMeta.site, null, null);
  await _fetchAndRenderCsvPage(csvModalMeta.site, next);
}

function csvRowNavigate(lat, lon, name) {
  // show inline confirmation bar at bottom of modal
  const existing = document.querySelector(".csv-nav-bar");
  if (existing) existing.remove();

  const bar = document.createElement("div");
  bar.className = "csv-nav-bar";
  bar.innerHTML = `
    <span>Go to <strong>${escapeHtml(name || "this location")}</strong>?</span>
    <div style="display:flex;gap:6px;">
      <button class="btn-nav-go" onclick="this.closest('.csv-nav-bar').remove(); _doCsvNavigate(${lat}, ${lon}, '${escapeHtml(name || "")}')">Go</button>
      <button class="btn-nav-cancel" onclick="this.closest('.csv-nav-bar').remove()">Cancel</button>
    </div>
  `;
  document.querySelector(".csv-modal").appendChild(bar);
  return;
}

function _doCsvNavigate(lat, lon, name) {
  // capture site before closing modal
  const modalSite = csvModalMeta ? csvModalMeta.site : null;
  closeCsvModal();

  // enable all markers so the target is visible
  const chk = document.getElementById("chk-show-all-markers");
  if (chk && !chk.checked) {
    chk.checked = true;
    toggleAllMarkers(true);
  }

  map.flyTo([lat, lon], 19, { duration: 1 });
  if (name) toast(`Navigated to ${name.substring(0, 40)}`);

  // find the site origin for this marker to compute route
  let originSite = null;
  if (modalSite && modalSite !== "combined") {
    originSite = locations.find((l) => l.name === modalSite);
  }
  if (!originSite) {
    // find closest site
    let minDist = Infinity;
    locations.forEach((l) => {
      const d = Math.abs(l.lat - lat) + Math.abs(l.lon - lon);
      if (d < minDist) { minDist = d; originSite = l; }
    });
  }

  // highlight with pulsing circle and compute route
  setTimeout(() => {
    const pulse = L.circleMarker([lat, lon], {
      radius: 18,
      fillColor: "#0A84FF",
      fillOpacity: 0.3,
      weight: 2,
      color: "#0A84FF",
      className: "csv-marker-pulse",
    }).addTo(map);
    setTimeout(() => map.removeLayer(pulse), 3000);

  }, 1200);
}

function csvGoToPage() {
  if (!csvModalMeta) return;
  const input = document.getElementById("csv-page-input");
  const target = parseInt(input.value, 10) - 1;
  if (isNaN(target) || target < 0 || target >= csvModalMeta.total_pages) return;
  _showCsvOverlay(csvModalMeta.site, null, null);
  _fetchAndRenderCsvPage(csvModalMeta.site, target);
}

function closeCsvModal() {
  const el = document.getElementById("csv-modal-overlay");
  if (el) el.remove();
  csvModalMeta = null;
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeCsvModal(); closeContextPopup(); }
  if (document.getElementById("csv-modal-overlay")) {
    if (e.key === "ArrowRight") csvPageNav(1);
    if (e.key === "ArrowLeft")  csvPageNav(-1);
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "s") {
    e.preventDefault();
    saveLocations();
  }
});


// ── amenity markers on map (item #5) ────────────────────
let showAllMarkers = false;

function toggleAllMarkers(checked) {
  showAllMarkers = checked;
  loadAmenityMarkers();
}

async function loadAmenityMarkers() {
  // clear existing
  if (amenityClusterGroup) {
    map.removeLayer(amenityClusterGroup);
    amenityClusterGroup = null;
  }
  if (currentRouteLayer) {
    map.removeLayer(currentRouteLayer);
    currentRouteLayer = null;
  }

  // determine which sites to show (only when toggle is enabled)
  if (!showAllMarkers) return;
  const sitesToLoad = locations.filter((l) => csvStatus[l.name]);

  if (!sitesToLoad.length) return;

  amenityClusterGroup = L.markerClusterGroup({
    maxClusterRadius: 50,
    spiderfyOnMaxZoom: true,
    showCoverageOnHover: false,
    zoomToBoundsOnClick: true,
  });

  await Promise.all(sitesToLoad.map(async (site) => {
    try {
      const res = await fetch(`/api/csv-markers/${encodeURIComponent(site.name)}`);
      if (!res.ok) return;
      const data = await res.json();
      if (!data.markers || !data.markers.length) return;

      data.markers.forEach((m) => {
        if (m.lat == null || m.lon == null) return;
        const color = LABEL_COLORS[m.label] || "#888";
        const icon = L.divIcon({
          className: "amenity-marker-wrap",
          html: `<div class="amenity-marker" style="background:${color};width:12px;height:12px;"></div>`,
          iconSize: [12, 12],
          iconAnchor: [6, 6],
        });

        const marker = L.marker([m.lat, m.lon], { icon });

        const popupLines = [];
        if (m.name) popupLines.push(`<div class="popup-name">${escapeHtml(m.name)}</div>`);
        popupLines.push(`<div class="popup-row"><span>Site</span><span>${escapeHtml(site.name)}</span></div>`);
        if (m.label) popupLines.push(`<div class="popup-row"><span>Label</span><span class="label-${m.label}">${m.label}</span></div>`);
        if (m.distance_m != null) popupLines.push(`<div class="popup-row"><span>Distance</span><span>${Math.round(m.distance_m)} m</span></div>`);
        if (m.osm_id) popupLines.push(`<div class="popup-row"><span>OSM ID</span><span>${m.osm_id}</span></div>`);
        popupLines.push(`<button class="popup-route-btn" onclick="computeRoute(${site.lat}, ${site.lon}, ${m.lat}, ${m.lon})">Show walking route</button>`);

        marker.bindPopup(`<div class="amenity-popup">${popupLines.join("")}</div>`, { maxWidth: 250 });
        amenityClusterGroup.addLayer(marker);
      });
    } catch (err) {
      console.error(`Failed to load markers for ${site.name}:`, err);
    }
  }));

  map.addLayer(amenityClusterGroup);
}

// ── walking route computation (item #5) ─────────────────
async function computeRoute(originLat, originLon, destLat, destLon) {
  // remove previous route
  if (currentRouteLayer) {
    map.removeLayer(currentRouteLayer);
    currentRouteLayer = null;
  }

  // show persistent loading toast with pulse
  const el = document.getElementById("toast");
  el.textContent = "Computing walking route...";
  el.className = "show loading-pulse";

  try {
    const res = await fetch("/api/walking-route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin_lat: originLat, origin_lon: originLon, dest_lat: destLat, dest_lon: destLon }),
    });
    const data = await res.json();
    el.className = "";
    if (data.error) { toast(data.error, "error"); return; }

    currentRouteLayer = L.polyline(data.route, {
      color: getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#0A84FF",
      weight: 4,
      opacity: 0.8,
      dashArray: "8 6",
      lineCap: "round",
      pane: "overlayPane",
    }).addTo(map);
    currentRouteLayer.bringToFront();
    map.fitBounds(currentRouteLayer.getBounds().pad(0.15));

    toast(`Route: ${Math.round(data.length_m)} m`);
  } catch (err) {
    el.className = "";
    console.error(err);
    toast("Failed to compute route", "error");
  }
}

// ── OSM tag editor (item #9) ────────────────────────────
function renderOsmTagsEditor() {
  if (!osmTags) return;
  const container = document.getElementById("osm-tags-editor");

  const sections = [
    { key: "commercial_tags", title: "Commercial Tags" },
    { key: "amenity_exclude", title: "Amenity Exclude" },
    { key: "leisure_exclude", title: "Leisure Exclude" },
    { key: "tourism_exclude", title: "Tourism Exclude" },
    { key: "shop_exclude", title: "Shop Exclude" },
  ];

  let html = "";
  sections.forEach((s) => {
    const tags = osmTags[s.key] || [];
    const chips = tags.map((t) =>
      `<span class="tag-chip">${escapeHtml(t)}<span class="tag-remove" onclick="removeTag('${s.key}', '${escapeHtml(t)}')">&times;</span></span>`
    ).join("");
    html += `
    <div class="tag-section">
      <div class="tag-section-title">${s.title}</div>
      <div class="tag-list">${chips || '<span style="color:var(--text-tertiary);font-size:11px">None</span>'}</div>
      <div class="tag-add-row">
        <input class="tag-add-input" id="tag-input-${s.key}" placeholder="Add tag..." onkeydown="if(event.key==='Enter')addTag('${s.key}')" />
        <button class="tag-add-btn" onclick="addTag('${s.key}')">+</button>
      </div>
    </div>`;
  });

  // preset controls
  const presetOptions = Object.keys(osmPresets).map((n) =>
    `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`
  ).join("");

  html += `
  <div class="tag-section">
    <div class="tag-section-title">Presets</div>
    <div class="tag-preset-row">
      <select class="tag-preset-select" id="tag-preset-select">
        <option value="">Load preset...</option>
        ${presetOptions}
      </select>
      <button class="tag-add-btn" onclick="loadTagPreset()">Load</button>
    </div>
    <div class="tag-preset-row">
      <input class="tag-add-input" id="tag-preset-name" placeholder="Preset name..." />
      <button class="tag-add-btn" onclick="saveTagPreset()">Save</button>
    </div>
  </div>
  <div style="padding:4px 0;">
    <button class="btn btn-primary" onclick="saveOsmTags()" style="width:100%">Save Tag Configuration</button>
  </div>`;

  container.innerHTML = html;
}

function addTag(key) {
  const input = document.getElementById(`tag-input-${key}`);
  const val = input.value.trim().toLowerCase();
  if (!val || !osmTags[key]) return;
  if (!osmTags[key].includes(val)) {
    osmTags[key].push(val);
    renderOsmTagsEditor();
  }
  input.value = "";
}

function removeTag(key, val) {
  if (!osmTags[key]) return;
  osmTags[key] = osmTags[key].filter((t) => t !== val);
  renderOsmTagsEditor();
}

async function saveOsmTags() {
  try {
    const res = await fetch("/api/osm-tags", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(osmTags) });
    const data = await res.json();
    if (data.ok) toast("OSM tag configuration saved"); else toast("Failed to save", "error");
  } catch { toast("Network error", "error"); }
}

async function loadTagPreset() {
  const name = document.getElementById("tag-preset-select").value;
  if (!name || !osmPresets[name]) return;
  osmTags = JSON.parse(JSON.stringify(osmPresets[name]));
  renderOsmTagsEditor();
  toast(`Loaded preset: ${name}`);
}

async function saveTagPreset() {
  const name = document.getElementById("tag-preset-name").value.trim();
  if (!name) { toast("Enter a preset name", "error"); return; }
  try {
    const res = await fetch("/api/osm-tag-presets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, config: osmTags }),
    });
    const data = await res.json();
    if (data.ok) {
      osmPresets[name] = JSON.parse(JSON.stringify(osmTags));
      renderOsmTagsEditor();
      toast(`Preset "${name}" saved`);
    }
  } catch { toast("Failed to save preset", "error"); }
}

// ── column selector (item #13) ──────────────────────────
const ALL_COLUMNS = [
  "osm_id", "name", "lat", "lon", "label", "distance_m",
  "lot_area_sqft", "highway_type", "lanes",
  "dist_bus_stop_m", "dist_hospital_m", "dist_school_m", "dist_park_m",
  "res_ratio", "com_ratio", "median_income",
];

function renderColumnSelector() {
  const container = document.getElementById("column-selector-list");
  if (!container) return;
  const active = visibleColumns || ALL_COLUMNS;
  container.innerHTML = ALL_COLUMNS.map((col) => {
    const on = active.includes(col);
    return `<span class="col-toggle ${on ? "active" : ""}" onclick="toggleColumn('${col}')">${col}</span>`;
  }).join("");
}

function toggleColumn(col) {
  if (!visibleColumns) visibleColumns = [...ALL_COLUMNS];
  const idx = visibleColumns.indexOf(col);
  if (idx >= 0) {
    if (visibleColumns.length <= 1) return; // keep at least one
    visibleColumns.splice(idx, 1);
  } else {
    visibleColumns.push(col);
  }
  renderColumnSelector();
  // refresh markers if active
  loadAmenityMarkers();
}

// ── session export/import (item #10) ────────────────────
async function exportSession() {
  try {
    const res = await fetch("/api/session/export");
    const data = await res.json();
    // add client-side state
    data.enabled_notebooks = Array.from(enabledNotebooks);
    data.visible_columns = visibleColumns;
    data.theme = currentTheme();

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `site_renderer_session_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast("Session exported");
  } catch { toast("Export failed", "error"); }
}

function importSession() {
  document.getElementById("session-file-input").click();
}

async function handleSessionFile(input) {
  const file = input.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const data = JSON.parse(text);

    // send to server
    await fetch("/api/session/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    // apply client-side state
    if (data.enabled_notebooks) {
      enabledNotebooks = new Set(data.enabled_notebooks);
    }
    if (data.visible_columns) {
      visibleColumns = data.visible_columns;
    }
    if (data.theme) {
      document.documentElement.setAttribute("data-theme", data.theme);
      localStorage.setItem("theme", data.theme);
      swapTiles(data.theme);
    }
    if (data.osm_tags) {
      osmTags = data.osm_tags;
    }

    // reload locations
    locations.forEach((l) => { map.removeLayer(l.marker); map.removeLayer(l.circle); });
    locations = [];
    activeId = null;
    nextId = 1;
    await loadLocations();
    await loadCsvStatus();
    renderPipelineTab();
    renderColumnSelector();

    toast("Session imported successfully");
  } catch (err) {
    console.error(err);
    toast("Failed to import session", "error");
  }
  input.value = "";
}

// ── clear all ───────────────────────────────────────────
function clearAll() {
  if (!locations.length) return;
  if (!confirm("Remove all locations?")) return;
  [...locations].forEach((l) => { map.removeLayer(l.marker); map.removeLayer(l.circle); });
  locations = []; activeId = null; nextId = 1;
  if (amenityClusterGroup) { map.removeLayer(amenityClusterGroup); amenityClusterGroup = null; }
  if (currentRouteLayer) { map.removeLayer(currentRouteLayer); currentRouteLayer = null; }
  renderSidebar(); updateMinimap();
  toast("All locations cleared");
}

// ── ML Plots ─────────────────────────────────────────────
let combinedCsvInfo  = null;
let plotPolling      = false;
let _plotStartTime   = null;
let plotExcludeCols  = new Set(["median_income", "lanes"]);
let plotSkipSet      = new Set(); // plot ids to skip (none skipped by default)

const AVAILABLE_PLOTS = [
  { id: "01", label: "Count by Location & Label" },
  { id: "02", label: "Boxplot: Commercial Ratio" },
  { id: "03", label: "Boxplot: Dist. to Hospital" },
  { id: "04", label: "Boxplot: Residential Ratio" },
  { id: "05", label: "Scatter: Residential Ratio" },
  { id: "06", label: "Pairplot per Location" },
  { id: "07", label: "Pairplot Combined" },
  { id: "08", label: "Correlation Heatmap" },
  { id: "09", label: "Confusion Matrix (LR)" },
  { id: "10", label: "Confusion Matrix (XGB)" },
];

const PLOT_FEATURE_COLS = [
  "lot_area_sqft", "dist_bus_stop_m", "dist_hospital_m", "dist_school_m",
  "dist_park_m", "com_ratio", "res_ratio", "median_income", "lanes", "highway_type",
];

async function loadCombinedCsvInfo() {
  try {
    const res = await fetch("/api/combined-csv-info");
    combinedCsvInfo = await res.json();
  } catch { combinedCsvInfo = null; }
  renderCombinedSection();
}

function renderCombinedSection() {
  const section = document.getElementById("combined-plots-section");
  const info    = document.getElementById("combined-plots-info");
  if (!combinedCsvInfo || !combinedCsvInfo.file) {
    section.style.display = "none";
    return;
  }
  section.style.display = "";
  const { file, run_id, modified, plots } = combinedCsvInfo;
  const date = new Date(modified * 1000).toLocaleString();
  const hasPlots = plots && plots.length > 0;

  info.innerHTML = `
    <div class="combined-csv-card">
      <div class="combined-csv-name" title="${escapeHtml(file)}">${escapeHtml(file)}</div>
      <div class="combined-csv-date">${date}</div>
      <div class="combined-csv-actions">
        <button class="btn btn-secondary" style="flex:1" onclick="openCsvModal('combined')">Preview</button>
        ${hasPlots
          ? `<button class="btn btn-secondary" onclick="openPlotsGalleryLatest()">View Plots (${plots.length})</button>
             <button class="btn-delete-csv" onclick="deletePlots()" title="Delete all plots">&times;</button>`
          : ""}
        <button class="btn btn-run" onclick="openPlotsColumnSelector('${run_id}', '${file}')">
          ${hasPlots ? "Re-run Plots" : "Generate Plots"}
        </button>
      </div>
      <div id="plot-progress-wrap" style="display:none;"></div>
    </div>`;
}

function plot_state_running() { return false; } // updated by polling

function openPlotsColumnSelector(runId, csvFile) {
  const old = document.getElementById("plots-col-selector-overlay");
  if (old) old.remove();

  const overlay = document.createElement("div");
  overlay.id = "plots-col-selector-overlay";
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

  // Column chips
  const colChips = PLOT_FEATURE_COLS.map((col) => {
    const included = !plotExcludeCols.has(col);
    return `<span class="col-toggle ${included ? "active" : ""}"
      onclick="_clickPlotCol('${col}', this)">${col}</span>`;
  }).join("");

  // Plot chips — all active by default (none skipped)
  const plotChips = AVAILABLE_PLOTS.map(({ id, label }) => {
    const included = !plotSkipSet.has(id);
    return `<span class="col-toggle ${included ? "active" : ""}"
      onclick="_clickPlotToggle('${id}', this)">${label}</span>`;
  }).join("");

  overlay.innerHTML = `
    <div class="plots-col-modal">
      <div class="csv-modal-header">
        <span class="csv-modal-title">Plot Configuration</span>
        <button class="ctx-close" onclick="document.getElementById('plots-col-selector-overlay').remove()">&#x2715;</button>
      </div>
      <p class="plots-col-hint" style="margin-top:10px;font-weight:600;">Plots to generate</p>
      <p class="plots-col-hint" style="margin-top:2px;">Highlighted = will be generated. Click to toggle.</p>
      <div class="plots-col-grid">${plotChips}</div>
      <p class="plots-col-hint" style="margin-top:12px;font-weight:600;">ML feature columns</p>
      <p class="plots-col-hint" style="margin-top:2px;">Identifiers (osm_id, lat, lon, name…) are always excluded from ML.</p>
      <div class="plots-col-grid">${colChips}</div>
      <div class="csv-modal-footer" style="justify-content:flex-end;gap:8px;">
        <button class="btn btn-secondary" onclick="document.getElementById('plots-col-selector-overlay').remove()">Cancel</button>
        <button class="btn btn-run" onclick="_confirmRunPlots('${runId}', '${csvFile}')">Run Plots &#9654;</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);
}

function _clickPlotCol(col, el) {
  const nowIncluded = el.classList.toggle("active");
  if (nowIncluded) plotExcludeCols.delete(col);
  else             plotExcludeCols.add(col);
}

function _clickPlotToggle(id, el) {
  const nowIncluded = el.classList.toggle("active");
  if (nowIncluded) plotSkipSet.delete(id);
  else             plotSkipSet.add(id);
}

async function _confirmRunPlots(runId, csvFile) {
  document.getElementById("plots-col-selector-overlay")?.remove();
  const excludeCols = Array.from(plotExcludeCols).join(",");
  const skipPlots   = Array.from(plotSkipSet).join(",");
  await runPlots(csvFile, excludeCols, skipPlots);
}

async function runPlots(csvFile, excludeCols, skipPlots = "") {
  try {
    const res = await fetch("/api/run-plots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ csv_file: csvFile, exclude_cols: excludeCols, skip_plots: skipPlots }),
    });
    const data = await res.json();
    if (data.error) { toast(data.error, "error"); return; }
    toast("Plot generation started...");
    startPlotPolling();
  } catch { toast("Failed to start plot generation", "error"); }
}

function startPlotPolling() {
  if (plotPolling) return;
  plotPolling = true;
  _plotStartTime = Date.now();
  _pollPlots();
}

async function _pollPlots() {
  try {
    const res  = await fetch("/api/plot-status");
    const data = await res.json();

    _renderPlotProgress(data);
    renderPipelineLog(data.log);   // show plot log in the same log panel

    if (data.running) {
      setTimeout(_pollPlots, 2000);
    } else {
      plotPolling = false;
      if (data.plots && data.plots.length > 0) {
        toast(`${data.plots.length} plots generated!`);
        await loadCombinedCsvInfo();
        openPlotsGallery(data.run_id, data.plots);
      } else {
        toast("Plot generation failed — see Log for details", "error");
        _renderPlotProgress(null);
      }
    }
  } catch { plotPolling = false; }
}

function _renderPlotProgress(data) {
  const el = document.getElementById("plot-progress-wrap");
  if (!el) return;

  if (!data || (!data.running && (!data.plots || !data.plots.length))) {
    el.style.display = "none";
    return;
  }

  const plots   = data.plots || [];
  const running = data.running;

  // Build chip list from existing files
  const chips = plots.map((fname) => {
    const short = fname.replace(/^\d+_/, "").replace(/_/g, " ").replace(".png", "");
    return `<span class="plot-prog-chip done" title="${escapeHtml(fname)}">${escapeHtml(short)}</span>`;
  }).join("");

  const spinner = running
    ? `<span class="plot-prog-chip spinning">generating…</span>`
    : "";

  const totalExpected = AVAILABLE_PLOTS.length - plotSkipSet.size;
  const fillPct = running ? Math.round(plots.length / Math.max(totalExpected, 1) * 100) : 100;
  const etaStr  = running ? _etaLabel(_plotStartTime, plots.length, Math.max(totalExpected - plots.length, 0)) : "";

  el.style.display = "";
  el.innerHTML = `
    <div class="plot-progress-track">
      <div class="plot-progress-fill" style="width:${fillPct}%"></div>
    </div>
    <div class="plot-progress-label">
      ${plots.length}${totalExpected ? `/${totalExpected}` : ""} plot${plots.length !== 1 ? "s" : ""} saved${running ? "…" : " ✓"}
      ${etaStr ? `<div class="prog-lbl-eta">${etaStr}</div>` : ""}
    </div>
    <div class="plot-prog-chips">${chips}${spinner}</div>`;
}

async function deletePlots() {
  if (!confirm("Delete all generated plots?")) return;
  try {
    const res = await fetch("/api/delete-plots", { method: "POST" });
    const data = await res.json();
    if (data.ok) {
      toast(`Deleted ${data.deleted} plot${data.deleted !== 1 ? "s" : ""}`);
      await loadCombinedCsvInfo();
    } else {
      toast("Failed to delete plots", "error");
    }
  } catch { toast("Failed to delete plots", "error"); }
}

async function openPlotsGalleryLatest() {
  await loadCombinedCsvInfo();
  const plots = combinedCsvInfo?.plots;
  if (plots && plots.length > 0) openPlotsGallery("latest", plots);
  else toast("No plots found", "error");
}

function openPlotsGallery(runId, plots) {
  const old = document.getElementById("plots-gallery-overlay");
  if (old) old.remove();

  const overlay = document.createElement("div");
  overlay.id = "plots-gallery-overlay";
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

  const cards = plots.map((fname) => {
    const url = `/api/plots/${encodeURIComponent(fname)}`;
    const label = fname.replace(/^\d+_/, "").replace(/_/g, " ").replace(".png", "");
    return `
      <div class="plot-card">
        <div class="plot-card-label">${escapeHtml(label)}</div>
        <a href="${url}" target="_blank">
          <img class="plot-thumb" src="${url}" alt="${escapeHtml(fname)}" loading="lazy" />
        </a>
        <a class="btn-csv-download" href="${url}" download="${escapeHtml(fname)}" style="margin-top:6px;display:block;text-align:center;">
          Download
        </a>
      </div>`;
  }).join("");

  overlay.innerHTML = `
    <div class="plots-gallery-modal">
      <div class="csv-modal-header">
        <span class="csv-modal-title">Plots — ${escapeHtml(runId)}</span>
        <button class="ctx-close" onclick="document.getElementById('plots-gallery-overlay').remove()">&#x2715;</button>
      </div>
      <div class="plots-gallery-grid">${cards}</div>
    </div>`;
  document.body.appendChild(overlay);
}

// ── init ────────────────────────────────────────────────
async function init() {
  await Promise.all([loadNotebooks(), loadOsmTags(), loadOsmPresets()]);
  await loadLocations();
  await loadCsvStatus();
  await loadCombinedCsvInfo();
  renderPipelineTab();
  renderColumnSelector();

  try {
    const res = await fetch("/api/pipeline-status");
    const data = await res.json();
    if (data.running) { showPipelineRunning(true); startPolling(); }
  } catch { /* ignore */ }

  try {
    const res = await fetch("/api/plot-status");
    const data = await res.json();
    if (data.running) startPlotPolling();
  } catch { /* ignore */ }
}

init();
