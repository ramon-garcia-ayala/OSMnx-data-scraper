"""Build the GitHub-Pages map (docs/index.html) with a Ground-truth / Prediction toggle.

The map stores, per grid cell, BOTH the real `zone_type` (ground truth) and the
model's `predicted_zone`, plus the predicted-class confidence. A small control in
the top-right switches the cell colors between the two layers; the popup always
shows both. Rendering uses a single Leaflet Canvas, so 20k+ cells stay light.

Unified compact data format (one source of truth, shared with the notebook):

    DATA = {
      "stepLat", "stepLon",            # constant grid step (deg)
      "center": [lat, lon],
      "classes": ["Residential", ...], # class index order
      "colors":  ["#2166AC", ...],     # one hex per class
      "cells":   [[latSW, lonSW, actualIdx, predIdx, confPct, cellId], ...],
    }

`make_html(data, title)` returns the standalone HTML string. Running this file as
a script migrates the existing docs/index.html (old Folium-derived format) into the
new toggle format in place.
"""
import json
import os
import re

# Categorical colors per zone (matches the notebook's PREFERRED palette).
PREFERRED = {
    "Commercial":  "#B2182B",   # red
    "Residential": "#2166AC",   # blue
    "Mixed-Use":   "#7B3294",   # purple
    "Other":       "#1B7837",   # green
    "Industrial":  "#F1A340",   # orange
    "Institutional": "#999999",
    "Open Space":  "#4DAC26",
}

TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>__TITLE__</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body { width:100%; height:100%; margin:0; padding:0; }
        #map { position:absolute; top:0; left:0; right:0; bottom:0; }
        .panel { position:fixed; z-index:1000; background:white; padding:10px 12px;
                 border-radius:6px; border:1px solid #888; font-size:13px; line-height:1.5;
                 font-family:Helvetica, Arial, sans-serif; box-shadow:0 1px 6px rgba(0,0,0,.2); }
        .toggle { top:10px; right:10px; }
        .toggle .row { margin-top:6px; }
        .toggle button { font:inherit; cursor:pointer; border:1px solid #888; background:#f3f3f3;
                         padding:5px 11px; border-radius:4px; margin-right:4px; }
        .toggle button.active { background:#333; color:#fff; border-color:#333; }
        .legend { bottom:30px; left:10px; }
        .loading { top:50%; left:50%; transform:translate(-50%,-50%); }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel toggle">
        <b>Map layer</b>
        <div class="row">
            <button id="btnPred" class="active" onclick="setMode('pred')">Prediction</button>
            <button id="btnGt" onclick="setMode('gt')">Ground truth</button>
        </div>
    </div>
    <div class="panel legend" id="legend"></div>
    <div id="loading" class="panel loading">Loading grid&hellip;</div>
    <script>
        const DATA = __DATA__;
        const COLORS = DATA.colors, CLASSES = DATA.classes;
        const dLat = DATA.stepLat, dLon = DATA.stepLon;
        let mode = "pred";   // "pred" | "gt"

        const map = L.map("map", { preferCanvas: true }).setView(DATA.center, 11);
        L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
            maxZoom: 20,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        }).addTo(map);

        const renderer = L.canvas({ padding: 0.5 });
        function classIdx(c) { return mode === "gt" ? c[2] : c[3]; }
        function popupFor(c) {
            return "<b>" + c[5] + "</b><br>Ground truth: " + CLASSES[c[2]] +
                   "<br>Predicted: <b>" + CLASSES[c[3]] + "</b><br>Confidence: " + c[4] + "%";
        }

        const rects = [];
        const group = L.featureGroup();
        for (const c of DATA.cells) {
            const lat = c[0], lon = c[1];
            const rect = L.rectangle(
                [[lat, lon], [lat + dLat, lon + dLon]],
                { renderer: renderer, color: "gray", weight: 0.3,
                  fill: true, fillColor: COLORS[c[3]], fillOpacity: 0.7 }
            );
            rect.bindPopup(() => popupFor(c), { maxWidth: 220 });
            rect._c = c;
            rects.push(rect);
            rect.addTo(group);
        }
        group.addTo(map);

        function renderLegend() {
            const title = mode === "gt" ? "Ground-truth zone" : "Predicted zone";
            let rows = "";
            for (let i = 0; i < CLASSES.length; i++)
                rows += '<span style="color:' + COLORS[i] + ';">&#9632;</span> ' + CLASSES[i] + '<br>';
            document.getElementById("legend").innerHTML = "<b>" + title + "</b><br>" + rows;
        }
        function setMode(m) {
            if (m === mode) return;
            mode = m;
            document.getElementById("btnPred").classList.toggle("active", m === "pred");
            document.getElementById("btnGt").classList.toggle("active", m === "gt");
            for (const r of rects) r.setStyle({ fillColor: COLORS[classIdx(r._c)] });
            renderLegend();
        }

        renderLegend();
        document.getElementById("loading").remove();
    </script>
</body>
</html>
"""


def make_html(data, title="Zone classification map"):
    """Return the standalone toggle-map HTML for a unified `data` dict."""
    data_js = json.dumps(data, separators=(",", ":"))
    return TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_js)


def _migrate_existing(src):
    """Read the old Folium-derived docs/index.html and return a unified data dict."""
    doc = open(src, encoding="utf-8").read()
    old = json.loads(re.search(r"const DATA = (\{.*?\});", doc, re.S).group(1))
    classes = old["classes"]                          # ["Residential","Commercial"]
    colors = [PREFERRED.get(c, "#999999") for c in classes]
    cells = []
    for c in old["cells"]:
        # old: [latSW, lonSW, colorIdx, actualIdx, predIdx, pComm, cellId]
        lat, lon, _cidx, a, p, pcomm, cell_id = c
        conf = pcomm if p == 1 else 100 - pcomm        # predicted-class confidence %
        cells.append([lat, lon, a, p, int(conf), cell_id])
    return {
        "stepLat": old["stepLat"], "stepLon": old["stepLon"],
        "center": old["center"], "classes": classes, "colors": colors, "cells": cells,
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "index.html")
    data = _migrate_existing(src)
    html = make_html(data, title="NYC Zoning — Ground truth vs Prediction")
    open(src, "w", encoding="utf-8").write(html)
    print(f"wrote {src}: {len(html)/1e6:.2f} MB | {len(data['cells'])} cells | "
          f"classes={data['classes']}")
