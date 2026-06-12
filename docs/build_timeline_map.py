"""Build docs/index.html: a multi-CITY, multi-LAYER zoning map.

City buttons   : NYC · Chicago · Detroit · San Francisco
Layer slider   : Ground truth + prediction layer(s)
  - NYC has a year timeline (Ground truth · 2012 · 2018 · 2026) because we have
    the historical OSM snapshots; each year is the model's prediction on that
    year's OSM.
  - Chicago / Detroit / San Francisco have one present-day prediction (the NYC
    XGBoost model applied zero-shot to that city's OSM features — the project's
    cross-city transfer design) plus their PLUTO/assessor ground truth.

For the three added cities `avg_floors` is not a real measurement (Chicago has
none; Detroit/SF carry a constant placeholder), so it is neutralized to the
scaler mean before prediction.

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

CELL_SIZE_M = 150
FEATURES = ["amenity_density", "amenity_ratio_food_drink", "shop_density_km2",
            "road_density_primary", "transit_stop_density", "intersection_density",
            "cell_lot_count", "avg_floors", "landuse_entropy"]

PRED_COLORS = {"Commercial": "#B2182B", "Residential": "#2166AC"}
GT_PALETTE = {
    "Residential": "#2166AC", "Commercial": "#B2182B", "Mixed-Use": "#7B3294",
    "Industrial": "#F1A340", "Institutional": "#888888", "Open Space": "#1B7837",
    "Other": "#444444", "Unknown": "#cccccc",
}
GT_ORDER = ["Residential", "Commercial", "Mixed-Use", "Industrial",
            "Institutional", "Open Space", "Other", "Unknown"]

# NYC = multi-year (all_boroughs_combined). Others = single present-day combined_grid.
NYC_YEARS = {"2012": "NYC_2012_150m-grid", "2016": "NYC_2016_150m-grid", "2026": "NYC_150m-grid"}
OTHER_CITIES = {
    "Chicago": "CHI_150m-grid",
    "Detroit": "DET_150m-grid",
    "San Francisco": "SF_150m-grid",
}


def load_model():
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(MODEL_DIR, "model.json"))
    with h5py.File(os.path.join(MODEL_DIR, "scaler.h5")) as f:
        mean, scale = f["mean"][()], f["scale"][()]
    with h5py.File(os.path.join(MODEL_DIR, "encoder.h5")) as f:
        classes = [c.decode() for c in f["classes"][()]]
    return clf, np.asarray(mean), np.asarray(scale), classes


def predict(clf, mean, scale, df, neutralize_floors=False):
    X = df[FEATURES].astype(float).copy()
    X["avg_floors"] = X["avg_floors"].fillna(mean[FEATURES.index("avg_floors")])
    if neutralize_floors:
        X["avg_floors"] = mean[FEATURES.index("avg_floors")]
    Xs = (X.values - mean) / scale
    return [int(v) for v in clf.predict(Xs)]


def grid_steps(ref_lat):
    lat_step = CELL_SIZE_M / 111_000
    lon_step = CELL_SIZE_M / (111_000 * math.cos(math.radians(ref_lat)))
    return lat_step, lon_step


def city_block(df, gt_col="zone_type"):
    """Common per-city payload (cells + geometry + gt classes), preds added by caller."""
    present = [c for c in GT_ORDER if c in set(df[gt_col])]
    gt_idx = {c: i for i, c in enumerate(present)}
    ref_lat = float(df["cell_lat"].mean())
    lat_step, lon_step = grid_steps(ref_lat)
    half_lat, half_lon = lat_step / 2, lon_step / 2
    cells = [[round(float(la) - half_lat, 6), round(float(lo) - half_lon, 6),
              int(gt_idx.get(z, len(present) - 1)), cid]
             for la, lo, z, cid in zip(df["cell_lat"], df["cell_lon"], df[gt_col], df["cell_id"])]
    return {
        "stepLat": round(lat_step, 7), "stepLon": round(lon_step, 7),
        "center": [round(ref_lat, 5), round(float(df["cell_lon"].mean()), 5)],
        "bounds": [[round(float(df["cell_lat"].min()) - half_lat, 6),
                    round(float(df["cell_lon"].min()) - half_lon, 6)],
                   [round(float(df["cell_lat"].max()) + half_lat, 6),
                    round(float(df["cell_lon"].max()) + half_lon, 6)]],
        "gtClasses": present,
        "cells": cells,
    }


def main():
    clf, mean, scale, pred_classes = load_model()
    pred_colors = [PRED_COLORS[c] for c in pred_classes]
    by_city = {}

    # --- NYC: timeline (years share one identical grid) -----------------------
    nyc_frames = {y: pd.read_csv(os.path.join(BASE, f, "all_boroughs_combined.csv"))
                  for y, f in NYC_YEARS.items()}
    for y, df in nyc_frames.items():
        df["key"] = df["borough"] + "/" + df["cell_id"]
    canon = nyc_frames["2026"].set_index("key")
    keys = list(canon.index)
    nyc = city_block(canon.reset_index())
    preds = {}
    for y, df in nyc_frames.items():
        preds[y] = predict(clf, mean, scale, df.set_index("key").reindex(keys))
        com = 100 * np.mean([pred_classes[v] == "Commercial" for v in preds[y]])
        print(f"  NYC {y}: Commercial {com:.1f}%")
    # Layer order: 2012 -> 2016 -> 2026 -> Ground truth (GT last).
    nyc["layers"] = ([{"t": "pred", "key": y, "label": y} for y in NYC_YEARS]
                     + [{"t": "gt", "label": "Ground truth"}])
    nyc["preds"] = preds
    by_city["NYC"] = nyc

    # --- Chicago / Detroit / San Francisco: GT + one transfer prediction ------
    for name, folder in OTHER_CITIES.items():
        df = pd.read_csv(os.path.join(BASE, folder, "combined_grid.csv"))
        blk = city_block(df)
        blk["preds"] = {"pred": predict(clf, mean, scale, df, neutralize_floors=True)}
        blk["layers"] = [{"t": "pred", "key": "pred", "label": "Prediction"},
                         {"t": "gt", "label": "Ground truth"}]
        com = 100 * np.mean([pred_classes[v] == "Commercial" for v in blk["preds"]["pred"]])
        print(f"  {name}: Commercial {com:.1f}% of {len(df)} cells")
        by_city[name] = blk

    data = {
        "cities": ["NYC", "Chicago", "Detroit", "San Francisco"],
        "predClasses": pred_classes, "predColors": pred_colors,
        "gtPalette": GT_PALETTE,
        "byCity": by_city,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(OUT, "w", encoding="utf-8").write(html)
    total = sum(len(c["cells"]) for c in by_city.values())
    print(f"wrote {OUT}: {len(html)/1e6:.2f} MB | {total} cells across {len(by_city)} cities")


TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Urban Zoning — cities &amp; timeline</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body { width:100%; height:100%; margin:0; padding:0; }
        #map { position:absolute; top:0; left:0; right:0; bottom:0; }
        .panel { position:fixed; z-index:1000; background:white; padding:10px 12px;
                 border-radius:6px; border:1px solid #888; font-size:13px; line-height:1.5;
                 font-family:Helvetica, Arial, sans-serif; box-shadow:0 1px 6px rgba(0,0,0,.2); }
        .ctrl { top:10px; right:10px; width:230px; }
        .ctrl .row { margin:6px 0; display:flex; flex-wrap:wrap; gap:4px; }
        .ctrl button { font:inherit; cursor:pointer; border:1px solid #888; background:#f3f3f3;
                       padding:4px 9px; border-radius:4px; }
        .ctrl button.active { background:#333; color:#fff; border-color:#333; }
        .ctrl hr { border:none; border-top:1px solid #ddd; margin:8px 0 4px; }
        .ctrl input[type=range] { width:100%; margin:6px 0 2px; }
        .ticks { display:flex; justify-content:space-between; font-size:11px; color:#444; }
        .yr { font-size:20px; font-weight:bold; }
        .legend { bottom:30px; left:10px; }
        .loading { top:50%; left:50%; transform:translate(-50%,-50%); }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel ctrl">
        <b>City</b>
        <div class="row" id="cityBtns"></div>
        <hr>
        <b>Map layer</b>
        <div class="row"><span class="yr" id="layerLabel"></span></div>
        <input type="range" id="slider" min="0" step="1" />
        <div class="ticks" id="ticks"></div>
    </div>
    <div class="panel legend" id="legend"></div>
    <div id="loading" class="panel loading">Loading grid&hellip;</div>
    <script>
        const DATA = __DATA__;
        let city = DATA.cities[0];
        let li = 0;
        let rects = [], group = null;

        const map = L.map("map", { preferCanvas: true }).setView(DATA.byCity[city].center, 11);
        L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
            maxZoom: 20,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        }).addTo(map);
        const renderer = L.canvas({ padding: 0.5 });

        function C() { return DATA.byCity[city]; }
        function curLayer() { return C().layers[li]; }
        function gtColors() { return C().gtClasses.map(n => DATA.gtPalette[n] || "#999"); }
        function colorOf(i) {
            const L = curLayer(), cell = C().cells[i];
            if (L.t === "gt") return gtColors()[cell[2]];
            return DATA.predColors[C().preds[L.key][i]];
        }
        function popupFor(i) {
            const Cc = C(), cell = Cc.cells[i];
            let rows = "";
            for (const L of Cc.layers)
                if (L.t === "pred")
                    rows += "Predicted " + L.label + ": <b>" + DATA.predClasses[Cc.preds[L.key][i]] + "</b><br>";
            return "<b>" + cell[3] + "</b><br>Ground truth: " + Cc.gtClasses[cell[2]] + "<br>" + rows;
        }

        function buildCity() {
            if (group) group.remove();
            rects = []; group = L.featureGroup();
            const Cc = C(), dLat = Cc.stepLat, dLon = Cc.stepLon;
            for (let i = 0; i < Cc.cells.length; i++) {
                const c = Cc.cells[i];
                const r = L.rectangle([[c[0], c[1]], [c[0] + dLat, c[1] + dLon]],
                    { renderer: renderer, color: "gray", weight: 0.3,
                      fill: true, fillColor: colorOf(i), fillOpacity: 0.7 });
                r.bindPopup(((k) => () => popupFor(k))(i), { maxWidth: 240 });
                rects.push(r); r.addTo(group);
            }
            group.addTo(map);
            map.fitBounds(Cc.bounds);
        }
        function recolor() { for (let i = 0; i < rects.length; i++) rects[i].setStyle({ fillColor: colorOf(i) }); }

        function renderLegend() {
            const L = curLayer();
            let title, classes, colors;
            if (L.t === "gt") { title = city + " — ground truth"; classes = C().gtClasses; colors = gtColors(); }
            else { title = city + " — predicted (" + L.label + ")"; classes = DATA.predClasses; colors = DATA.predColors; }
            let rows = "";
            for (let i = 0; i < classes.length; i++)
                rows += '<span style="color:' + colors[i] + ';">&#9632;</span> ' + classes[i] + '<br>';
            document.getElementById("legend").innerHTML = "<b>" + title + "</b><br>" + rows;
        }
        function refreshLabel() { document.getElementById("layerLabel").textContent = curLayer().label; }

        const slider = document.getElementById("slider");
        function setupSlider() {
            const Cc = C();
            slider.max = Cc.layers.length - 1;
            li = 0;                              // default -> most recent prediction
            for (let k = 0; k < Cc.layers.length; k++) if (Cc.layers[k].t === "pred") li = k;
            slider.value = li;
            document.getElementById("ticks").innerHTML =
                Cc.layers.map(L => "<span>" + L.label + "</span>").join("");
        }
        slider.addEventListener("input", () => {
            li = parseInt(slider.value, 10);
            recolor(); renderLegend(); refreshLabel();
        });

        function makeCityButtons() {
            const wrap = document.getElementById("cityBtns");
            wrap.innerHTML = "";
            for (const name of DATA.cities) {
                const b = document.createElement("button");
                b.textContent = name;
                b.className = (name === city) ? "active" : "";
                b.onclick = () => setCity(name);
                wrap.appendChild(b);
            }
        }
        function setCity(name) {
            city = name;
            makeCityButtons();
            setupSlider();
            buildCity();
            renderLegend(); refreshLabel();
        }

        makeCityButtons();
        setupSlider();
        buildCity();
        renderLegend(); refreshLabel();
        document.getElementById("loading").remove();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
