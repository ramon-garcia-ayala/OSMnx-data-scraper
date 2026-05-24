"""
Site Scraper – Flask backend
Serves the interactive Leaflet map and exposes API endpoints for:
  - GET/POST /api/locations        → read/write locations.json
  - POST     /api/run-site         → run pipeline for one site
  - POST     /api/run-all          → run pipeline for all sites
  - POST     /api/cancel           → cancel running pipeline
  - GET      /api/pipeline-status  → per-site, per-notebook status
  - GET      /api/csv-status       → which site CSVs exist
  - GET      /api/csv-history/<n>  → all CSV versions for a site
  - GET      /api/csv-site/<n>     → paginated CSV viewer
  - GET      /api/csv-site/<n>/download
  - POST     /api/csv-delete       → delete specific CSV files
  - POST     /api/csv-cleanup      → remove orphan intermediates
  - GET      /api/csv-markers/<n>  → lightweight marker data
  - POST     /api/combine          → merge all site CSVs
  - GET      /api/csv-data         → combined CSV as JSON
  - GET/POST /api/osm-tags         → OSM tag configuration
  - GET/POST /api/osm-tag-presets  → tag presets
  - GET      /api/session/export   → export all settings as JSON
  - POST     /api/session/import   → import settings from JSON
  - POST     /api/walking-route    → compute OSMnx walking route
  - GET      /api/geocode          → proxy to Nominatim
  - GET      /api/notebooks        → notebook definitions
"""

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time as _time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

# ── paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
SCRAPER_DIR = BASE_DIR.parent / "General-OSM-Scraper"
LOCATIONS_FILE = SCRAPER_DIR / "locations.json"
CSV_DIR = SCRAPER_DIR / "csv"
OSM_TAGS_FILE = SCRAPER_DIR / "osm_tags_config.json"
OSM_PRESETS_FILE = SCRAPER_DIR / "osm_tag_presets.json"
PLUTO_FILE = SCRAPER_DIR.parent / "ramy" / "NYC_pluto_25v4_csv" / "pluto_25v4.csv"
PEDESTRIAN_FILE = SCRAPER_DIR.parent / "ramy" / "pedestrian_mobility"

app = Flask(__name__)

# ── notebook definitions ─────────────────────────────────
NOTEBOOKS = [
    {"id": "01", "file": "01_identifiers.ipynb",           "label": "Identifiers"},
    {"id": "02", "file": "02_y_target.ipynb",              "label": "Y-Target"},
    {"id": "03", "file": "03_morphological.ipynb",         "label": "Morphological",         "needs_pluto": True},
    {"id": "04", "file": "04_synergistic_proximity.ipynb", "label": "Synergistic Proximity"},
    {"id": "05", "file": "05_socioeconomic.ipynb",         "label": "Socioeconomic",          "needs_pluto": True},
    {"id": "06", "file": "06_joker.ipynb",                 "label": "Joker (median income)"},
    # ── enrichment notebooks ──────────────────────────────
    {"id": "07", "file": "07_rent_proxy.ipynb",            "label": "Rent Proxy",             "group": "enrichment", "needs_pluto": True},
    {"id": "08", "file": "08_building_age.ipynb",          "label": "Building Age",           "group": "enrichment", "needs_pluto": True},
    {"id": "09", "file": "09_foot_traffic.ipynb",          "label": "Foot Traffic",           "group": "enrichment", "needs_pedestrian": True},
    {"id": "10", "file": "10_subway_distance.ipynb",       "label": "Subway Distance",        "group": "enrichment"},
    {"id": "11", "file": "11_shop_density.ipynb",          "label": "Shop Density",           "group": "enrichment"},
    {"id": "12", "file": "12_population_density.ipynb",    "label": "Population Density",     "group": "enrichment"},
    {"id": "13", "file": "13_mix_shop_types.ipynb",        "label": "Shop Type Mix",          "group": "enrichment"},
]

# ── default OSM tags ─────────────────────────────────────
DEFAULT_OSM_TAGS = {
    "commercial_tags": ["shop", "amenity", "office", "craft", "tourism", "leisure"],
    "amenity_exclude": [
        "parking", "bench", "waste_basket", "recycling", "post_box",
        "bicycle_parking", "drinking_water", "telephone", "toilets",
        "street_lamp", "fire_station", "school", "place_of_worship",
        "clock", "bicycle_rental", "parking_entrance", "vending_machine",
        "fountain", "charging_station", "device_charging_station",
        "bus_station", "social_facility", "community_centre", "police",
        "university", "college", "kindergarten",
    ],
    "leisure_exclude": [
        "garden", "park", "pitch", "playground", "outdoor_seating",
        "swimming_pool", "dog_park", "nature_reserve", "picnic_site",
        "shelter", "watering_place", "common",
    ],
    "tourism_exclude": [
        "artwork", "viewpoint", "information", "attraction",
        "picnic_site", "camp_site", "caravan_site",
    ],
    "shop_exclude": ["vacant", "yes"],
}

# ── pipeline state ───────────────────────────────────────
pipeline_state = {
    "running": False,
    "cancelled": False,
    "current_site": None,
    "current_notebook": None,
    "sites": {},
    "log": [],
}
_pipeline_process = None
_pipeline_lock = threading.Lock()

# ── ML plot paths & state ────────────────────────────────
ML_NOTEBOOK    = BASE_DIR.parent / "ML_Plot" / "NYC_classification.ipynb"
PLOTS_BASE_DIR = BASE_DIR.parent / "ML_Plot" / "outputs"

plot_state = {
    "running": False,
    "run_id": None,
    "log": [],
}
_plot_process = None
_plot_lock    = threading.Lock()

# ── graph cache for walking routes ───────────────────────
_graph_cache = {}
_graph_cache_lock = threading.Lock()


# ── pages ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── API: locations ───────────────────────────────────────
@app.route("/api/locations", methods=["GET"])
def get_locations():
    if LOCATIONS_FILE.exists():
        data = json.loads(LOCATIONS_FILE.read_text(encoding="utf-8"))
    else:
        data = []
    return jsonify(data)


@app.route("/api/locations", methods=["POST"])
def save_locations():
    locations = request.get_json(force=True)
    if not isinstance(locations, list):
        return jsonify({"error": "Expected a JSON array"}), 400
    LOCATIONS_FILE.write_text(
        json.dumps(locations, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return jsonify({"ok": True, "count": len(locations)})


# ── API: geocode (proxy to Nominatim) ───────────────────
@app.route("/api/geocode", methods=["GET"])
def geocode():
    import urllib.request
    import urllib.parse

    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    params = urllib.parse.urlencode({
        "q": q, "format": "json", "limit": 5, "addressdetails": 1,
    })
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "SiteRenderer/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    results = [
        {
            "display_name": r.get("display_name", ""),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "type": r.get("type", ""),
        }
        for r in data
    ]
    return jsonify(results)


# ── API: CSV status ──────────────────────────────────────
@app.route("/api/csv-status", methods=["GET"])
def csv_status():
    existing = {}
    if not CSV_DIR.exists():
        return jsonify(existing)

    candidates = {}
    for f in CSV_DIR.glob("*_20*.csv"):
        stem = f.stem
        parts = stem.rsplit("_", 2)
        if len(parts) < 3:
            continue
        site_name = stem[: stem.rfind("_" + parts[-2])]
        if site_name == "combined":
            continue
        mtime = f.stat().st_mtime
        if site_name not in candidates or mtime > candidates[site_name]["mtime"]:
            candidates[site_name] = {"file": f.name, "mtime": mtime, "path": f}

    for site_name, info in candidates.items():
        entry = {"file": info["file"], "modified": info["mtime"], "params": None}
        sidecar = CSV_DIR / f"{site_name}.params.json"
        if sidecar.exists():
            try:
                entry["params"] = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing[site_name] = entry

    return jsonify(existing)


# ── API: CSV history (all versions for a site) ───────────
@app.route("/api/csv-history/<site_name>", methods=["GET"])
def csv_history(site_name):
    if not CSV_DIR.exists():
        return jsonify([])
    files = sorted(
        [f for f in CSV_DIR.glob(f"{site_name}_20*.csv")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    result = []
    for f in files:
        stat = f.stat()
        result.append({
            "file": f.name,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
            "modified_str": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return jsonify(result)


# ── API: CSV data for a single site (paginated) ─────────
@app.route("/api/csv-site/<site_name>", methods=["GET"])
def get_site_csv(site_name):
    import pandas as pd

    page      = int(request.args.get("page", 0))
    page_size = int(request.args.get("size", 50))
    page_size = max(1, min(page_size, 200))
    sort_col  = request.args.get("sort", None)
    sort_dir  = request.args.get("dir", "asc")

    matches = sorted(
        [f for f in CSV_DIR.glob(f"{site_name}_20*.csv")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return jsonify({"error": f"No CSV found for site '{site_name}'"}), 404

    latest = matches[0]
    df = pd.read_csv(latest)

    if sort_col and sort_col in df.columns:
        ascending = sort_dir != "desc"
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")

    total_rows = len(df)
    total_pages = max(1, -(-total_rows // page_size))
    page = max(0, min(page, total_pages - 1))

    page_df = df.iloc[page * page_size : (page + 1) * page_size]

    # Convert to records and replace NaN/NaT with None for JSON serialization
    records = page_df.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[k] = None

    return jsonify({
        "site": site_name,
        "file": latest.name,
        "rows": total_rows,
        "total_pages": total_pages,
        "page": page,
        "page_size": page_size,
        "columns": list(df.columns),
        "data": records,
    })


# ── API: download CSV for a single site ─────────────────
@app.route("/api/csv-site/<site_name>/download", methods=["GET"])
def download_site_csv(site_name):
    matches = sorted(
        [f for f in CSV_DIR.glob(f"{site_name}_20*.csv")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return jsonify({"error": f"No CSV found for site '{site_name}'"}), 404
    return send_file(matches[0], as_attachment=True, download_name=matches[0].name)


# ── API: CSV markers (lightweight for map) ───────────────
@app.route("/api/csv-markers/<site_name>", methods=["GET"])
def csv_markers(site_name):
    import pandas as pd

    matches = sorted(
        [f for f in CSV_DIR.glob(f"{site_name}_20*.csv")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return jsonify({"error": f"No CSV found for site '{site_name}'"}), 404

    df = pd.read_csv(matches[0])
    cols = ["osm_id", "name", "lat", "lon", "label", "distance_m"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    # drop rows without coordinates
    df = df.dropna(subset=["lat", "lon"])
    records = df.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[k] = None
    return jsonify({"site": site_name, "markers": records})


# ── API: CSV delete ──────────────────────────────────────
@app.route("/api/csv-delete", methods=["POST"])
def csv_delete():
    body = request.get_json(force=True)
    files = body.get("files", [])
    deleted = []
    for fname in files:
        fp = CSV_DIR / fname
        if fp.exists() and fp.parent == CSV_DIR and fp.suffix == ".csv":
            fp.unlink()
            deleted.append(fname)
            # also remove sidecar if it's the latest
            site_name = fname.rsplit("_", 2)[0] if fname.count("_") >= 2 else None
            if site_name:
                sidecar = CSV_DIR / f"{site_name}.params.json"
                if sidecar.exists():
                    sidecar.unlink()
    return jsonify({"ok": True, "deleted": deleted})


@app.route("/api/csv-delete-site", methods=["POST"])
def csv_delete_site():
    body = request.get_json(force=True)
    site_name = body.get("site", "")
    if not site_name:
        return jsonify({"error": "No site specified"}), 400
    deleted = []
    for f in CSV_DIR.glob(f"{site_name}_20*.csv"):
        f.unlink()
        deleted.append(f.name)
    sidecar = CSV_DIR / f"{site_name}.params.json"
    if sidecar.exists():
        sidecar.unlink()
    return jsonify({"ok": True, "deleted": deleted})


# ── API: CSV cleanup (remove intermediates) ──────────────
INTERMEDIATES = [
    "csv/00_base_data.csv", "csv/01_identifiers.csv", "csv/02_y_target.csv",
    "csv/03_morphological.csv", "csv/04_synergistic_proximity.csv",
    "csv/05_socioeconomic.csv", "csv/06_joker.csv",
    "csv/07_rent_proxy.csv", "csv/08_building_age.csv",
    "csv/09_foot_traffic.csv", "csv/10_subway_distance.csv",
    "csv/11_shop_density.csv", "csv/12_population_density.csv",
    "csv/13_mix_shop_types.csv",
]

@app.route("/api/csv-cleanup", methods=["POST"])
def csv_cleanup():
    removed = []
    for f in INTERMEDIATES:
        p = SCRAPER_DIR / f
        if p.exists():
            p.unlink()
            removed.append(f)
    return jsonify({"ok": True, "removed": removed})


# ── API: CSV data (combined) ────────────────────────────
@app.route("/api/csv-data", methods=["GET"])
def get_csv_data():
    import pandas as pd
    csv_files = sorted(CSV_DIR.glob("combined_*.csv"))
    if not csv_files:
        return jsonify([])
    df = pd.read_csv(csv_files[-1])
    df = df.where(df.notna(), None)
    return jsonify(df.to_dict(orient="records"))


# ── API: OSM tag configuration ──────────────────────────
@app.route("/api/osm-tags", methods=["GET"])
def get_osm_tags():
    if OSM_TAGS_FILE.exists():
        try:
            data = json.loads(OSM_TAGS_FILE.read_text(encoding="utf-8"))
            return jsonify(data)
        except Exception:
            pass
    return jsonify(DEFAULT_OSM_TAGS)


@app.route("/api/osm-tags", methods=["POST"])
def save_osm_tags():
    data = request.get_json(force=True)
    OSM_TAGS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return jsonify({"ok": True})


# ── API: OSM tag presets ─────────────────────────────────
@app.route("/api/osm-tag-presets", methods=["GET"])
def get_osm_tag_presets():
    if OSM_PRESETS_FILE.exists():
        try:
            return jsonify(json.loads(OSM_PRESETS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return jsonify({"default": DEFAULT_OSM_TAGS})


@app.route("/api/osm-tag-presets", methods=["POST"])
def save_osm_tag_preset():
    body = request.get_json(force=True)
    name = body.get("name", "").strip()
    config = body.get("config")
    if not name or not config:
        return jsonify({"error": "name and config required"}), 400

    presets = {}
    if OSM_PRESETS_FILE.exists():
        try:
            presets = json.loads(OSM_PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    presets[name] = config
    OSM_PRESETS_FILE.write_text(
        json.dumps(presets, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return jsonify({"ok": True, "presets": list(presets.keys())})


# ── API: session export/import ──────────────────────────
@app.route("/api/session/export", methods=["GET"])
def session_export():
    locations = []
    if LOCATIONS_FILE.exists():
        try:
            locations = json.loads(LOCATIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    osm_tags = DEFAULT_OSM_TAGS
    if OSM_TAGS_FILE.exists():
        try:
            osm_tags = json.loads(OSM_TAGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    return jsonify({
        "version": 1,
        "exported_at": datetime.now().isoformat(),
        "locations": locations,
        "osm_tags": osm_tags,
    })


@app.route("/api/session/import", methods=["POST"])
def session_import():
    data = request.get_json(force=True)
    imported = []

    if "locations" in data and isinstance(data["locations"], list):
        LOCATIONS_FILE.write_text(
            json.dumps(data["locations"], indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        imported.append("locations")

    if "osm_tags" in data and isinstance(data["osm_tags"], dict):
        OSM_TAGS_FILE.write_text(
            json.dumps(data["osm_tags"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        imported.append("osm_tags")

    return jsonify({"ok": True, "imported": imported})


# ── API: walking route via OSMnx ─────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


@app.route("/api/walking-route", methods=["POST"])
def walking_route():
    try:
        import osmnx as ox
        import networkx as nx
    except ImportError:
        return jsonify({"error": "osmnx not installed"}), 500

    body = request.get_json(force=True)
    o_lat, o_lon = body["origin_lat"], body["origin_lon"]
    d_lat, d_lon = body["dest_lat"], body["dest_lon"]

    try:
        # Cache key: origin rounded to 2 decimals (~1km grid) — one graph per site
        cache_key = (round(o_lat, 2), round(o_lon, 2))
        with _graph_cache_lock:
            G = _graph_cache.get(cache_key)

        if G is None:
            # Load a graph covering the site's walk radius (use origin as center)
            dist = max(1500, _haversine(o_lat, o_lon, d_lat, d_lon) * 1.8)
            G = ox.graph_from_point((o_lat, o_lon), dist=dist, network_type="walk")
            with _graph_cache_lock:
                if len(_graph_cache) > 10:
                    _graph_cache.pop(next(iter(_graph_cache)))
                _graph_cache[cache_key] = G

        orig_node = ox.nearest_nodes(G, o_lon, o_lat)
        dest_node = ox.nearest_nodes(G, d_lon, d_lat)
        route = nx.shortest_path(G, orig_node, dest_node, weight="length")
        route_coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in route]
        length = nx.shortest_path_length(G, orig_node, dest_node, weight="length")

        return jsonify({"route": route_coords, "length_m": round(length, 1)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: PLUTO status ───────────────────────────────────
@app.route("/api/pluto-status", methods=["GET"])
def pluto_status():
    return jsonify({"available": PLUTO_FILE.exists(), "path": str(PLUTO_FILE)})


# ── pipeline runner ──────────────────────────────────────
def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    pipeline_state["log"].append(line)
    if len(pipeline_state["log"]) > 1000:
        pipeline_state["log"] = pipeline_state["log"][-1000:]
    print(line)


def _run_notebook(nb_file, params, site_name, nb_id):
    global _pipeline_process

    pipeline_state["current_notebook"] = nb_id
    pipeline_state["sites"][site_name][nb_id] = "running"
    _log(f"{site_name} | {nb_id} starting...")

    nb_path = SCRAPER_DIR / nb_file
    if not nb_path.exists():
        pipeline_state["sites"][site_name][nb_id] = "failed"
        _log(f"{site_name} | {nb_id} FAILED – notebook not found")
        return False

    tmp_out = Path(tempfile.mktemp(suffix=".ipynb"))

    cmd = [
        sys.executable, "-m", "papermill",
        str(nb_path), str(tmp_out),
        "--kernel", "python3",
    ]
    for key, val in params.items():
        cmd.extend(["-p", key, str(val)])

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(SCRAPER_DIR),
        )
        with _pipeline_lock:
            _pipeline_process = proc

        stdout, stderr = proc.communicate(timeout=1800)

        if pipeline_state["cancelled"]:
            pipeline_state["sites"][site_name][nb_id] = "failed"
            _log(f"{site_name} | {nb_id} CANCELLED")
            return False

        if proc.returncode == 0:
            pipeline_state["sites"][site_name][nb_id] = "done"
            _log(f"{site_name} | {nb_id} OK")
            return True
        else:
            pipeline_state["sites"][site_name][nb_id] = "failed"
            _log(f"{site_name} | {nb_id} FAILED (exit {proc.returncode})")
            for line in (stderr or "").splitlines():
                if line.strip():
                    _log(line)
            return False

    except subprocess.TimeoutExpired:
        proc.kill()
        pipeline_state["sites"][site_name][nb_id] = "failed"
        _log(f"{site_name} | {nb_id} TIMEOUT")
        return False
    except Exception as e:
        pipeline_state["sites"][site_name][nb_id] = "failed"
        _log(f"{site_name} | {nb_id} ERROR: {e}")
        return False
    finally:
        with _pipeline_lock:
            _pipeline_process = None
        if tmp_out.exists():
            tmp_out.unlink()


def _get_nb_params(loc, nb_id):
    walk_min = loc.get("walk_minutes", 15.0)
    walk_spd = loc.get("walk_speed_m_min", 80.0)
    radius_m = walk_min * walk_spd
    net_dist = int(radius_m + 300)

    if nb_id == "01":
        p = {
            "LAT": loc["lat"], "LON": loc["lon"],
            "WALK_MINUTES": walk_min, "WALK_SPEED_M_MIN": walk_spd,
            "RADIUS_M": radius_m,
        }
        if OSM_TAGS_FILE.exists():
            p["OSM_TAGS_CONFIG"] = str(OSM_TAGS_FILE)
        return p
    elif nb_id == "03":
        return {
            "ORIGIN_LAT": loc["lat"], "ORIGIN_LON": loc["lon"],
            "NETWORK_DIST": net_dist,
        }
    return {}


FINAL_COLUMNS = [
    "osm_id", "name", "lat", "lon", "label", "distance_m",
    "lot_area_sqft", "highway_type", "lanes",
    "dist_bus_stop_m", "dist_hospital_m", "dist_school_m", "dist_park_m",
    "res_ratio", "com_ratio",
    "median_income",
    "assess_per_sqft",
    "year_built",
    "pedestrian_rank", "pedestrian_category",
    "dist_subway_m",
    "shops_within_100m",
    "population_density",
    "shop_mix_entropy",
]

CSV_GROUPS = [
    ("csv/01_identifiers.csv",           ["osm_id", "name", "lat", "lon", "distance_m", "label"]),
    ("csv/03_morphological.csv",         ["highway_type", "lanes", "lot_area_sqft"]),
    ("csv/04_synergistic_proximity.csv", ["dist_bus_stop_m", "dist_hospital_m", "dist_school_m", "dist_park_m"]),
    ("csv/05_socioeconomic.csv",         ["com_ratio", "res_ratio"]),
    ("csv/06_joker.csv",                 ["median_income"]),
    ("csv/07_rent_proxy.csv",            ["assess_per_sqft"]),
    ("csv/08_building_age.csv",          ["year_built"]),
    ("csv/09_foot_traffic.csv",          ["pedestrian_rank", "pedestrian_category"]),
    ("csv/10_subway_distance.csv",       ["dist_subway_m"]),
    ("csv/11_shop_density.csv",          ["shops_within_100m"]),
    ("csv/12_population_density.csv",    ["population_density"]),
    ("csv/13_mix_shop_types.csv",        ["shop_mix_entropy"]),
]


def _merge_site_csvs(site_name, run_id, loc_params=None):
    import pandas as pd

    id_csv = SCRAPER_DIR / "csv" / "01_identifiers.csv"
    if not id_csv.exists():
        _log(f"{site_name} | merge FAILED – 01_identifiers.csv not found")
        return None

    df = pd.read_csv(id_csv)
    for csv_path, cols in CSV_GROUPS[1:]:
        full_path = SCRAPER_DIR / csv_path
        if full_path.exists():
            df_other = pd.read_csv(full_path)
            df = df.merge(df_other, on="osm_id", how="left")
        else:
            for col in cols:
                df[col] = None

    df = df[[c for c in FINAL_COLUMNS if c in df.columns]]
    df.insert(0, "location", site_name)

    # Remove previous versions of this site's CSV
    for old in CSV_DIR.glob(f"{site_name}_20*.csv"):
        old.unlink()

    out_path = CSV_DIR / f"{site_name}_{run_id}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    _log(f"{site_name} | merged → {out_path.name} ({len(df)} rows)")

    if loc_params:
        sidecar = CSV_DIR / f"{site_name}.params.json"
        sidecar.write_text(json.dumps(loc_params, ensure_ascii=False), encoding="utf-8")

    return out_path


def _cleanup_intermediates():
    for f in INTERMEDIATES:
        p = SCRAPER_DIR / f
        if p.exists():
            p.unlink()


def _run_site(loc, enabled_notebooks, run_id):
    site_name = loc["name"]
    pipeline_state["current_site"] = site_name
    _log(f"{'='*40}")
    _log(f"SITE: {site_name} ({loc['lat']}, {loc['lon']})")
    _log(f"{'='*40}")

    all_ok = True
    for nb in NOTEBOOKS:
        if pipeline_state["cancelled"]:
            for nb2 in NOTEBOOKS:
                if pipeline_state["sites"][site_name].get(nb2["id"]) == "pending":
                    pipeline_state["sites"][site_name][nb2["id"]] = "skipped"
            return False

        if nb["id"] not in enabled_notebooks:
            pipeline_state["sites"][site_name][nb["id"]] = "skipped"
            _log(f"{site_name} | {nb['id']} SKIPPED")
            continue

        # auto-skip PLUTO-dependent notebooks if PLUTO not available
        if nb.get("needs_pluto") and not PLUTO_FILE.exists():
            pipeline_state["sites"][site_name][nb["id"]] = "skipped"
            _log(f"{site_name} | {nb['id']} SKIPPED (PLUTO data not available)")
            continue

        # auto-skip pedestrian-dependent notebooks if data not available
        if nb.get("needs_pedestrian") and not PEDESTRIAN_FILE.exists():
            pipeline_state["sites"][site_name][nb["id"]] = "skipped"
            _log(f"{site_name} | {nb['id']} SKIPPED (pedestrian data not available)")
            continue

        params = _get_nb_params(loc, nb["id"])
        ok = _run_notebook(nb["file"], params, site_name, nb["id"])
        if not ok:
            all_ok = False
            for nb2 in NOTEBOOKS:
                if pipeline_state["sites"][site_name].get(nb2["id"]) == "pending":
                    pipeline_state["sites"][site_name][nb2["id"]] = "skipped"
            break

    if all_ok:
        loc_params = {
            "lat":              loc.get("lat"),
            "lon":              loc.get("lon"),
            "walk_minutes":     loc.get("walk_minutes"),
            "walk_speed_m_min": loc.get("walk_speed_m_min"),
            "notebooks":        sorted(enabled_notebooks),
        }
        _merge_site_csvs(site_name, run_id, loc_params)

    _cleanup_intermediates()
    return all_ok


def _pipeline_thread(site_names, locations_data, enabled_notebooks):
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for loc in locations_data:
        if loc["name"] not in site_names:
            continue
        if pipeline_state["cancelled"]:
            break
        _run_site(loc, enabled_notebooks, run_id)

    pipeline_state["running"] = False
    pipeline_state["current_site"] = None
    pipeline_state["current_notebook"] = None
    if pipeline_state["cancelled"]:
        _log("Pipeline CANCELLED by user")
    else:
        _log("Pipeline COMPLETE")


# ── API: run pipeline ───────────────────────────────────
@app.route("/api/run-site", methods=["POST"])
def run_site():
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 409

    body = request.get_json(force=True)
    site_names = body.get("sites", [])
    enabled_notebooks = body.get("notebooks", [])

    if not site_names:
        return jsonify({"error": "No sites specified"}), 400
    if not enabled_notebooks:
        return jsonify({"error": "No notebooks selected"}), 400

    if not LOCATIONS_FILE.exists():
        return jsonify({"error": "locations.json not found"}), 404
    locations_data = json.loads(LOCATIONS_FILE.read_text(encoding="utf-8"))

    pipeline_state["running"] = True
    pipeline_state["cancelled"] = False
    pipeline_state["log"] = []
    pipeline_state["sites"] = {}
    for name in site_names:
        pipeline_state["sites"][name] = {}
        for nb in NOTEBOOKS:
            if nb["id"] in enabled_notebooks:
                pipeline_state["sites"][name][nb["id"]] = "pending"
            else:
                pipeline_state["sites"][name][nb["id"]] = "skipped"

    thread = threading.Thread(
        target=_pipeline_thread,
        args=(site_names, locations_data, enabled_notebooks),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True, "message": f"Pipeline started for {len(site_names)} site(s)"})


@app.route("/api/run-all", methods=["POST"])
def run_all():
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 409

    body = request.get_json(force=True)
    enabled_notebooks = body.get("notebooks", ["01", "02", "03", "04", "05"])

    if not LOCATIONS_FILE.exists():
        return jsonify({"error": "locations.json not found"}), 404
    locations_data = json.loads(LOCATIONS_FILE.read_text(encoding="utf-8"))
    site_names = [loc["name"] for loc in locations_data]

    pipeline_state["running"] = True
    pipeline_state["cancelled"] = False
    pipeline_state["log"] = []
    pipeline_state["sites"] = {}
    for name in site_names:
        pipeline_state["sites"][name] = {}
        for nb in NOTEBOOKS:
            if nb["id"] in enabled_notebooks:
                pipeline_state["sites"][name][nb["id"]] = "pending"
            else:
                pipeline_state["sites"][name][nb["id"]] = "skipped"

    thread = threading.Thread(
        target=_pipeline_thread,
        args=(site_names, locations_data, enabled_notebooks),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True, "message": f"Pipeline started for {len(site_names)} site(s)"})


# ── API: cancel pipeline ────────────────────────────────
@app.route("/api/cancel", methods=["POST"])
def cancel_pipeline():
    if not pipeline_state["running"]:
        return jsonify({"error": "No pipeline running"}), 400

    pipeline_state["cancelled"] = True

    with _pipeline_lock:
        if _pipeline_process and _pipeline_process.poll() is None:
            try:
                _pipeline_process.terminate()
                _log("Sent TERMINATE to running notebook")
            except OSError:
                pass

    return jsonify({"ok": True, "message": "Cancel requested"})


# ── API: pipeline status ────────────────────────────────
@app.route("/api/pipeline-status", methods=["GET"])
def get_pipeline_status():
    return jsonify(pipeline_state)


# ── API: combine CSVs ───────────────────────────────────
@app.route("/api/combine", methods=["POST"])
def combine_csvs():
    import pandas as pd

    if not CSV_DIR.exists():
        return jsonify({"error": "csv/ directory not found"}), 404

    site_csvs = {}
    for f in CSV_DIR.glob("*_20*.csv"):
        if f.stem.startswith("combined"):
            continue
        stem = f.stem
        if len(stem) > 20:
            site_name = stem[:-20]
            if site_name not in site_csvs or f.stat().st_mtime > site_csvs[site_name][1]:
                site_csvs[site_name] = (f, f.stat().st_mtime)

    if not site_csvs:
        return jsonify({"error": "No site CSVs found"}), 404

    dfs = []
    for site_name, (csv_path, _) in site_csvs.items():
        dfs.append(pd.read_csv(csv_path))

    df_combined = pd.concat(dfs, ignore_index=True)

    # Remove previous combined CSVs
    for old in CSV_DIR.glob("combined_20*.csv"):
        old.unlink()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = CSV_DIR / f"combined_{run_id}.csv"
    df_combined.to_csv(out_path, index=False, encoding="utf-8")

    return jsonify({
        "ok": True,
        "file": out_path.name,
        "rows": len(df_combined),
        "sites": list(site_csvs.keys()),
    })


# ── API: notebook definitions ────────────────────────────
@app.route("/api/notebooks", methods=["GET"])
def get_notebooks():
    result = []
    for nb in NOTEBOOKS:
        entry = dict(nb)
        if nb.get("needs_pluto"):
            entry["pluto_available"] = PLUTO_FILE.exists()
        if nb.get("needs_pedestrian"):
            entry["pedestrian_available"] = PEDESTRIAN_FILE.exists()
        result.append(entry)
    return jsonify(result)


# ── ML plots runner ─────────────────────────────────────
def _extract_notebook_traceback(nb_path):
    """Read the executed notebook and extract traceback from failed cells."""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        lines = []
        for cell in nb.get("cells", []):
            for output in cell.get("outputs", []):
                if output.get("output_type") in ("error", "stream"):
                    tb = output.get("traceback") or []
                    text = output.get("text", "")
                    # strip ANSI color codes
                    import re
                    ansi = re.compile(r"\x1b\[[0-9;]*m")
                    for t in tb:
                        for l in ansi.sub("", t).splitlines():
                            if l.strip():
                                lines.append(l)
                    if isinstance(text, list):
                        text = "".join(text)
                    for l in text.splitlines():
                        if l.strip():
                            lines.append(l)
        return lines
    except Exception as e:
        return [f"(could not read notebook output: {e})"]


def _run_plots_thread(csv_path, plots_dir, run_id, exclude_cols, skip_plots=""):
    global _plot_process
    plot_state["running"] = True
    plot_state["run_id"]  = run_id
    plot_state["log"]     = [f"[{datetime.now().strftime('%H:%M:%S')}] Starting plots..."]

    tmp_out = Path(tempfile.mktemp(suffix=".ipynb"))
    cmd = [
        sys.executable, "-m", "papermill",
        str(ML_NOTEBOOK), str(tmp_out),
        "--kernel", "python3",
        "--log-output",          # stream cell output to stderr
        "-p", "CSV_PATH",     csv_path,
        "-p", "PLOTS_DIR",    plots_dir,
        "-p", "EXCLUDE_COLS", exclude_cols,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # merge both streams
            text=True, cwd=str(ML_NOTEBOOK.parent),
        )
        with _plot_lock:
            _plot_process = proc
        stdout, _ = proc.communicate(timeout=3600)
        ts = datetime.now().strftime("%H:%M:%S")

        if proc.returncode == 0:
            plot_state["log"].append(f"[{ts}] Plots generated successfully")
            # Delete plots the user chose to skip
            if skip_plots:
                plots_dir_path = Path(plots_dir)
                for skip_id in skip_plots.split(","):
                    sid = skip_id.strip().zfill(2)
                    if sid:
                        for f in plots_dir_path.glob(f"{sid}_*.png"):
                            f.unlink()
                            plot_state["log"].append(f"Skipped (removed): {f.name}")
        else:
            plot_state["log"].append(f"[{ts}] FAILED (exit {proc.returncode})")
            # 1. Log merged stdout (papermill progress + cell output)
            for line in (stdout or "").splitlines():
                if line.strip():
                    plot_state["log"].append(line)
            # 2. Also extract traceback directly from the output notebook
            if tmp_out.exists():
                tb_lines = _extract_notebook_traceback(tmp_out)
                if tb_lines:
                    plot_state["log"].append("── Cell traceback ──")
                    plot_state["log"].extend(tb_lines)

    except subprocess.TimeoutExpired:
        proc.kill()
        plot_state["log"].append("TIMEOUT — plot generation took too long")
    except Exception as e:
        plot_state["log"].append(f"ERROR: {e}")
    finally:
        plot_state["running"] = False
        with _plot_lock:
            _plot_process = None
        if tmp_out.exists():
            tmp_out.unlink()


@app.route("/api/run-plots", methods=["POST"])
def run_plots():
    if plot_state["running"]:
        return jsonify({"error": "Plot generation already running"}), 409

    body         = request.get_json(force=True)
    csv_filename = body.get("csv_file", "")
    exclude_cols = body.get("exclude_cols", "")

    csv_path = CSV_DIR / csv_filename
    if not csv_path.exists():
        return jsonify({"error": f"CSV not found: {csv_filename}"}), 404

    skip_plots = body.get("skip_plots", "")

    # Always write to a fixed "latest" folder — overwrites previous plots
    plots_dir_path = PLOTS_BASE_DIR / "latest"
    plots_dir_path.mkdir(parents=True, exist_ok=True)
    for old in plots_dir_path.glob("*.png"):
        old.unlink()

    plots_dir    = plots_dir_path.as_posix()
    csv_path_str = csv_path.as_posix()

    threading.Thread(
        target=_run_plots_thread,
        args=(csv_path_str, plots_dir, "latest", exclude_cols, skip_plots),
        daemon=True,
    ).start()
    return jsonify({"ok": True, "run_id": "latest"})


@app.route("/api/plot-status", methods=["GET"])
def get_plot_status():
    plots_dir = PLOTS_BASE_DIR / "latest"
    plots     = sorted([f.name for f in plots_dir.glob("*.png")]) if plots_dir.exists() else []
    return jsonify({**plot_state, "plots": plots})


@app.route("/api/plots/<filename>")
def serve_plot(filename):
    if ".." in filename:
        return jsonify({"error": "Invalid path"}), 400
    return send_file(PLOTS_BASE_DIR / "latest" / filename)


@app.route("/api/delete-plots", methods=["POST"])
def delete_plots():
    plots_dir = PLOTS_BASE_DIR / "latest"
    deleted = []
    if plots_dir.exists():
        for f in plots_dir.glob("*.png"):
            f.unlink()
            deleted.append(f.name)
    return jsonify({"ok": True, "deleted": len(deleted)})


@app.route("/api/combined-csv-info", methods=["GET"])
def combined_csv_info():
    if not CSV_DIR.exists():
        return jsonify({"file": None})
    files = sorted(
        CSV_DIR.glob("combined_20*.csv"),
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    if not files:
        return jsonify({"file": None})
    f         = files[0]
    plots_dir = PLOTS_BASE_DIR / "latest"
    plots     = sorted([p.name for p in plots_dir.glob("*.png")]) if plots_dir.exists() else []
    return jsonify({
        "file":     f.name,
        "run_id":   "latest",
        "modified": f.stat().st_mtime,
        "plots":    plots,
    })


# ── main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Locations file : {LOCATIONS_FILE}")
    print(f"CSV directory  : {CSV_DIR}")
    print(f"PLUTO data     : {'FOUND' if PLUTO_FILE.exists() else 'NOT FOUND (NYC-only notebooks will be skipped)'}")
    print(f"Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
