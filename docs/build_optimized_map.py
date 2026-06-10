"""Rebuild docs/index.html as a lightweight Leaflet map.

The original 28 MB file (exported by Folium) encoded all 23,768 grid cells as
individual SVG rectangles, each with verbose JSON styling and a pre-rendered
popup. This script parses that file once and regenerates an equivalent map that:

  * stores each cell as a compact array (SW corner + palette index + bits),
    deriving the NE corner from the constant grid step,
  * renders on a single Canvas renderer instead of 23k SVG DOM nodes,
  * builds popup HTML lazily, only on click.

Result: same map, ~1 MB instead of 28 MB, opens almost instantly.

Run from repo root:  python docs/build_optimized_map.py
The original is backed up to docs/index_full.html.bak on first run.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "index.html")
BAK = os.path.join(HERE, "index_full.html.bak")

doc = open(SRC, encoding="utf-8").read()

# Back up the original heavy file once (so this is re-runnable / reversible).
if not os.path.exists(BAK):
    open(BAK, "w", encoding="utf-8").write(doc)

# --- Parse the Folium output ------------------------------------------------
chunks = doc.split("var rectangle_")[1:]
rb = re.compile(r"L\.rectangle\(\s*\[\[([-\d.]+), ([-\d.]+)\], \[([-\d.]+), ([-\d.]+)\]\]")
fc = re.compile(r'"fillColor": "([^"]+)"')
hp = re.compile(
    r"<b>([^<]+)</b><br>Actual: ([^<]+)<br>Predicted: <b>([^<]+)</b><br>(.*?)</div>",
    re.S,
)

CLASSES = ["Residential", "Commercial"]  # index 0 / 1
cidx = {c: i for i, c in enumerate(CLASSES)}

palette = []          # unique fill colors (hex without '#')
pal_idx = {}
cells = []            # [latSW6, lonSW6, colorIdx, aBit, pBit, pComm, cellId]
min_lat = min_lon = max_lat = max_lon = None

for ch in chunks:
    g = rb.search(ch)
    lat1, lon1 = float(g.group(1)), float(g.group(2))   # SW corner
    color = fc.search(ch).group(1).lstrip("#")
    m = hp.search(ch)
    cell_id, actual, pred, prob_part = m.group(1), m.group(2), m.group(3), m.group(4)
    pcomm = int(re.search(r"P\(Commercial\): (\d+)%", prob_part).group(1))

    if color not in pal_idx:
        pal_idx[color] = len(palette)
        palette.append(color)

    cells.append([
        round(lat1, 6), round(lon1, 6),
        pal_idx[color], cidx[actual], cidx[pred], pcomm, cell_id,
    ])

    if min_lat is None:
        min_lat = max_lat = lat1
        min_lon = max_lon = lon1
    min_lat, max_lat = min(min_lat, lat1), max(max_lat, lat1)
    min_lon, max_lon = min(min_lon, lon1), max(max_lon, lon1)

# Constant grid step (verified: every cell is identical in size).
g0 = rb.search(chunks[0])
STEP_LAT = round(float(g0.group(3)) - float(g0.group(1)), 7)
STEP_LON = round(float(g0.group(4)) - float(g0.group(2)), 7)

center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

print(f"cells={len(cells)}  colors={len(palette)}  "
      f"step=({STEP_LAT},{STEP_LON})  center=({center_lat:.5f},{center_lon:.5f})")

# --- Emit the lightweight HTML ---------------------------------------------
data = {
    "stepLat": STEP_LAT,
    "stepLon": STEP_LON,
    "center": [round(center_lat, 5), round(center_lon, 5)],
    "palette": palette,
    "classes": CLASSES,
    "cells": cells,
}
# separators without spaces -> smallest JSON
data_js = json.dumps(data, separators=(",", ":"))

html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NYC Predicted Zoning</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body { width:100%; height:100%; margin:0; padding:0; }
        #map { position:absolute; top:0; left:0; right:0; bottom:0; }
        .legend { position:fixed; bottom:30px; left:10px; z-index:1000;
                  background:white; padding:10px; border-radius:5px;
                  border:1px solid gray; font-size:13px; line-height:1.4;
                  font-family:Helvetica, Arial, sans-serif; }
        .loading { position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
                   z-index:2000; background:white; padding:14px 22px; border-radius:6px;
                   border:1px solid #888; font-family:Helvetica, Arial, sans-serif;
                   font-size:14px; box-shadow:0 2px 8px rgba(0,0,0,.2); }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="legend">
        <b>Predicted zone</b><br>
        <span style="color:#B2182B;">&#9632;</span> Commercial<br>
        <span style="color:#2166AC;">&#9632;</span> Residential
    </div>
    <div id="loading" class="loading">Loading grid&hellip;</div>
    <script>
        const DATA = __DATA__;
        const map = L.map("map", { preferCanvas: true }).setView(DATA.center, 11);
        L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
            maxZoom: 20,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        }).addTo(map);

        const renderer = L.canvas({ padding: 0.5 });
        const dLat = DATA.stepLat, dLon = DATA.stepLon;

        function popupFor(c) {
            const cellId = c[6], actual = DATA.classes[c[3]],
                  pred = DATA.classes[c[4]], pc = c[5];
            return "<b>" + cellId + "</b><br>Actual: " + actual +
                   "<br>Predicted: <b>" + pred + "</b><br>" +
                   "P(Commercial): " + pc + "%<br>P(Residential): " + (100 - pc) + "%";
        }

        const group = L.featureGroup();
        for (const c of DATA.cells) {
            const lat = c[0], lon = c[1];
            const rect = L.rectangle(
                [[lat, lon], [lat + dLat, lon + dLon]],
                { renderer: renderer, color: "gray", weight: 0.3,
                  fill: true, fillColor: "#" + DATA.palette[c[2]],
                  fillOpacity: 0.7 }
            );
            rect.bindPopup(() => popupFor(c), { maxWidth: 220 });
            rect.addTo(group);
        }
        group.addTo(map);
        document.getElementById("loading").remove();
    </script>
</body>
</html>
"""
html = html.replace("__DATA__", data_js)
open(SRC, "w", encoding="utf-8").write(html)
print(f"wrote {SRC}: {len(html):,} chars ({len(html)/1e6:.2f} MB)")
