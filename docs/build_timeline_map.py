"""Build docs/index.html as a TIMELINE map: a year slider (2012 · 2018 · 2026)
that shows the model's predicted zoning from each year's OSM snapshot, plus a
Ground-truth / Prediction toggle.

How it works
------------
The 150 m NYC grid is identical across years (same PLUTO grid + labels), so the
cells align 1:1. For each year we load that year's `all_boroughs_combined.csv`,
scale the 9 features with the exported StandardScaler, and run the exported
XGBoost classifier -> a predicted class (Commercial / Residential) per cell.
As OSM coverage grew over the years, the predicted urban character changes — the
slider animates that. The ground-truth layer is the (constant) PLUTO zone_type.

Run from repo root:  python docs/build_timeline_map.py
"""
import json
import math
import os

import h5py
import numpy as np
import pandas as pd
import xgboost as xgb

BASE = r"H:\My Drive\03_AIA_TERM\Data Encoding\TEAM_Notebooks\02_csv-datasets"
MODEL_DIR = (r"H:\My Drive\03_AIA_TERM\Data Encoding\TEAM_Notebooks\06_Models"
             r"\SL_NYC_classification_150m-grid_Distributed-Mixed-Use_acc0.8901977282288599")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "index.html")

# year label -> source folder (all share the identical 150 m grid)
YEARS = {
    "2012": "NYC_2012_150m-grid",
    "2018": "NYC_2018_150m-grid",
    "2026": "NYC_150m-grid",          # most recent (present-day OSM)
}
CELL_SIZE_M = 150

# Feature order is the CSV-native order; confirmed to match scaler.mean_.
FEATURES = ["amenity_density", "amenity_ratio_food_drink", "shop_density_km2",
            "road_density_primary", "transit_stop_density", "intersection_density",
            "cell_lot_count", "avg_floors", "landuse_entropy"]

PRED_COLORS = {"Commercial": "#B2182B", "Residential": "#2166AC"}
GT_CLASSES = ["Residential", "Commercial", "Mixed-Use", "Other"]
GT_COLORS = ["#2166AC", "#B2182B", "#7B3294", "#1B7837"]


def load_model():
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(MODEL_DIR, "model.json"))
    with h5py.File(os.path.join(MODEL_DIR, "scaler.h5")) as f:
        mean, scale = f["mean"][()], f["scale"][()]
    with h5py.File(os.path.join(MODEL_DIR, "encoder.h5")) as f:
        classes = [c.decode() for c in f["classes"][()]]   # index order of predict()
    return clf, mean, scale, classes


def main():
    clf, mean, scale, pred_classes = load_model()
    pred_colors = [PRED_COLORS[c] for c in pred_classes]

    frames = {y: pd.read_csv(os.path.join(BASE, folder, "all_boroughs_combined.csv"))
              for y, folder in YEARS.items()}
    for y, df in frames.items():
        df["key"] = df["borough"] + "/" + df["cell_id"]

    # Canonical cell order = the most-recent layer; align all years to it.
    base_year = "2026"
    canon = frames[base_year].set_index("key")
    keys = list(canon.index)
    for y, df in frames.items():
        frames[y] = df.set_index("key").reindex(keys)

    gt_idx = {c: i for i, c in enumerate(GT_CLASSES)}
    gtcol = canon["zone_type"].map(gt_idx).fillna(len(GT_CLASSES) - 1).astype(int).tolist()

    preds = {}
    for y in YEARS:
        X = frames[y][FEATURES].astype(float).values
        Xs = (X - mean) / scale
        preds[y] = [int(v) for v in clf.predict(Xs)]   # idx into pred_classes
        share_com = 100 * np.mean([pred_classes[v] == "Commercial" for v in preds[y]])
        print(f"  {y}: predicted Commercial {share_com:.1f}% of {len(preds[y])} cells")

    ref_lat = float(canon["cell_lat"].mean())
    lat_step = CELL_SIZE_M / 111_000
    lon_step = CELL_SIZE_M / (111_000 * math.cos(math.radians(ref_lat)))
    half_lat, half_lon = lat_step / 2, lon_step / 2

    cells = [[round(float(la) - half_lat, 6), round(float(lo) - half_lon, 6), g, cid]
             for la, lo, g, cid in zip(canon["cell_lat"], canon["cell_lon"],
                                       gtcol, canon["cell_id"])]

    data = {
        "stepLat": round(lat_step, 7), "stepLon": round(lon_step, 7),
        "center": [round(ref_lat, 5), round(float(canon["cell_lon"].mean()), 5)],
        "years": list(YEARS.keys()),
        "predClasses": pred_classes, "predColors": pred_colors,
        "gtClasses": GT_CLASSES, "gtColors": GT_COLORS,
        "cells": cells, "preds": preds,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"wrote {OUT}: {len(html)/1e6:.2f} MB | {len(cells)} cells | years={list(YEARS)}")


TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NYC Zoning Timeline — 2012 / 2018 / 2026</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body { width:100%; height:100%; margin:0; padding:0; }
        #map { position:absolute; top:0; left:0; right:0; bottom:0; }
        .panel { position:fixed; z-index:1000; background:white; padding:10px 12px;
                 border-radius:6px; border:1px solid #888; font-size:13px; line-height:1.5;
                 font-family:Helvetica, Arial, sans-serif; box-shadow:0 1px 6px rgba(0,0,0,.2); }
        .ctrl { top:10px; right:10px; width:210px; }
        .ctrl .row { margin-top:8px; }
        .ctrl button { font:inherit; cursor:pointer; border:1px solid #888; background:#f3f3f3;
                       padding:5px 11px; border-radius:4px; margin-right:4px; }
        .ctrl button.active { background:#333; color:#fff; border-color:#333; }
        .ctrl input[type=range] { width:100%; margin:6px 0 2px; }
        .ticks { display:flex; justify-content:space-between; font-size:11px; color:#444; }
        .yr { font-size:22px; font-weight:bold; }
        .muted { color:#999; }
        .legend { bottom:30px; left:10px; }
        .loading { top:50%; left:50%; transform:translate(-50%,-50%); }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel ctrl">
        <b>Map layer</b>
        <div class="row">
            <button id="btnPred" class="active" onclick="setMode('pred')">Prediction</button>
            <button id="btnGt" onclick="setMode('gt')">Ground truth</button>
        </div>
        <div class="row" id="timeWrap">
            <div>Year: <span class="yr" id="yrLabel"></span></div>
            <input type="range" id="slider" min="0" step="1" />
            <div class="ticks" id="ticks"></div>
        </div>
    </div>
    <div class="panel legend" id="legend"></div>
    <div id="loading" class="panel loading">Loading grid&hellip;</div>
    <script>
        const DATA = __DATA__;
        const dLat = DATA.stepLat, dLon = DATA.stepLon;
        const YEARS = DATA.years;
        let mode = "pred";           // "pred" | "gt"
        let yi = YEARS.length - 1;   // year index -> most recent by default

        const map = L.map("map", { preferCanvas: true }).setView(DATA.center, 11);
        L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
            maxZoom: 20,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        }).addTo(map);

        const renderer = L.canvas({ padding: 0.5 });
        function colorOf(i) {
            if (mode === "gt") return DATA.gtColors[DATA.cells[i][2]];
            return DATA.predColors[DATA.preds[YEARS[yi]][i]];
        }
        function popupFor(i) {
            const c = DATA.cells[i];
            let rows = "";
            for (const y of YEARS)
                rows += "Predicted " + y + ": <b>" + DATA.predClasses[DATA.preds[y][i]] + "</b><br>";
            return "<b>" + c[3] + "</b><br>Ground truth: " + DATA.gtClasses[c[2]] + "<br>" + rows;
        }

        const rects = [];
        const group = L.featureGroup();
        for (let i = 0; i < DATA.cells.length; i++) {
            const c = DATA.cells[i];
            const rect = L.rectangle(
                [[c[0], c[1]], [c[0] + dLat, c[1] + dLon]],
                { renderer: renderer, color: "gray", weight: 0.3,
                  fill: true, fillColor: colorOf(i), fillOpacity: 0.7 }
            );
            rect.bindPopup(((k) => () => popupFor(k))(i), { maxWidth: 240 });
            rects.push(rect);
            rect.addTo(group);
        }
        group.addTo(map);

        function recolor() { for (let i = 0; i < rects.length; i++) rects[i].setStyle({ fillColor: colorOf(i) }); }

        function renderLegend() {
            let title, classes, colors;
            if (mode === "gt") { title = "Ground-truth zone (PLUTO)"; classes = DATA.gtClasses; colors = DATA.gtColors; }
            else { title = "Predicted zone — " + YEARS[yi]; classes = DATA.predClasses; colors = DATA.predColors; }
            let rows = "";
            for (let i = 0; i < classes.length; i++)
                rows += '<span style="color:' + colors[i] + ';">&#9632;</span> ' + classes[i] + '<br>';
            document.getElementById("legend").innerHTML = "<b>" + title + "</b><br>" + rows;
        }

        function refreshTime() {
            document.getElementById("yrLabel").textContent = YEARS[yi];
            document.getElementById("timeWrap").style.opacity = (mode === "gt") ? 0.4 : 1;
            document.getElementById("slider").disabled = (mode === "gt");
        }
        function setMode(m) {
            mode = m;
            document.getElementById("btnPred").classList.toggle("active", m === "pred");
            document.getElementById("btnGt").classList.toggle("active", m === "gt");
            refreshTime(); recolor(); renderLegend();
        }

        // Slider setup
        const slider = document.getElementById("slider");
        slider.max = YEARS.length - 1;
        slider.value = yi;
        document.getElementById("ticks").innerHTML = YEARS.map(y => "<span>" + y + "</span>").join("");
        slider.addEventListener("input", () => {
            yi = parseInt(slider.value, 10);
            if (mode !== "pred") setMode("pred");
            refreshTime(); recolor(); renderLegend();
        });

        refreshTime(); renderLegend();
        document.getElementById("loading").remove();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
