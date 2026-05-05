"""
Site Renderer – Flask backend
Serves the interactive Leaflet map and exposes API endpoints for:
  - GET/POST /api/locations        → read/write locations.json
  - POST     /api/run-site         → run pipeline for one site
  - POST     /api/run-all          → run pipeline for all sites
  - POST     /api/cancel           → cancel running pipeline
  - GET      /api/pipeline-status  → per-site, per-notebook status
  - GET      /api/csv-status       → which site CSVs exist
  - POST     /api/combine          → merge all site CSVs
  - GET      /api/csv-data         → combined CSV as JSON
  - GET      /api/geocode          → proxy to Nominatim
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time as _time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# ── paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
SCRAPER_DIR = BASE_DIR.parent / "General-OSM-Scraper"
LOCATIONS_FILE = SCRAPER_DIR / "locations.json"
CSV_DIR = SCRAPER_DIR / "csv"

app = Flask(__name__)

# ── notebook definitions ─────────────────────────────────
NOTEBOOKS = [
    {"id": "01", "file": "01_identifiers.ipynb",           "label": "Identifiers"},
    {"id": "02", "file": "02_y_target.ipynb",              "label": "Y-Target"},
    {"id": "03", "file": "03_morphological.ipynb",         "label": "Morphological"},
    {"id": "04", "file": "04_synergistic_proximity.ipynb", "label": "Synergistic Proximity"},
    {"id": "05", "file": "05_socioeconomic.ipynb",         "label": "Socioeconomic"},
    {"id": "06", "file": "06_joker.ipynb",                 "label": "Joker (median income)"},
]

# ── pipeline state ───────────────────────────────────────
# status per site: { "site_name": { "01": "pending"|"running"|"done"|"failed"|"skipped", ... } }
pipeline_state = {
    "running": False,
    "cancelled": False,
    "current_site": None,
    "current_notebook": None,
    "sites": {},        # per-site, per-notebook status
    "log": [],          # recent log lines
}
_pipeline_process = None   # subprocess.Popen reference
_pipeline_lock = threading.Lock()


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


# ── API: CSV status (which sites have CSVs) ─────────────
@app.route("/api/csv-status", methods=["GET"])
def csv_status():
    """Return which site CSVs exist in the csv/ directory."""
    existing = {}
    if CSV_DIR.exists():
        for f in CSV_DIR.glob("*_20*.csv"):
            name = f.stem
            # strip the timestamp suffix: everything after the last date pattern
            parts = name.rsplit("_", 2)
            if len(parts) >= 3:
                site_name = name[: name.rfind("_" + parts[-2])]
                if site_name != "combined":
                    existing[site_name] = {
                        "file": f.name,
                        "modified": f.stat().st_mtime,
                    }
    return jsonify(existing)


# ── API: CSV data ────────────────────────────────────────
@app.route("/api/csv-data", methods=["GET"])
def get_csv_data():
    """Return the most recent combined CSV as JSON."""
    import pandas as pd

    csv_files = sorted(CSV_DIR.glob("combined_*.csv"))
    if not csv_files:
        return jsonify([])
    df = pd.read_csv(csv_files[-1])
    df = df.where(df.notna(), None)
    return jsonify(df.to_dict(orient="records"))


# ── pipeline runner ──────────────────────────────────────
def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    pipeline_state["log"].append(line)
    # keep last 200 lines
    if len(pipeline_state["log"]) > 200:
        pipeline_state["log"] = pipeline_state["log"][-200:]
    print(line)


def _run_notebook(nb_file, params, site_name, nb_id):
    """Run a single notebook via papermill. Returns True on success."""
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
    # add parameters
    for key, val in params.items():
        cmd.extend(["-p", key, str(val)])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRAPER_DIR),
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
            err_short = (stderr or "")[-500:]
            _log(f"{site_name} | {nb_id} FAILED (exit {proc.returncode}): {err_short}")
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
    """Return papermill parameters for a given notebook and location."""
    walk_min = loc.get("walk_minutes", 15.0)
    walk_spd = loc.get("walk_speed_m_min", 80.0)
    radius_m = walk_min * walk_spd
    net_dist = int(radius_m + 300)

    if nb_id == "01":
        return {
            "LAT": loc["lat"], "LON": loc["lon"],
            "WALK_MINUTES": walk_min, "WALK_SPEED_M_MIN": walk_spd,
            "RADIUS_M": radius_m,
        }
    elif nb_id == "03":
        return {
            "ORIGIN_LAT": loc["lat"], "ORIGIN_LON": loc["lon"],
            "NETWORK_DIST": net_dist,
        }
    return {}


INTERMEDIATES = [
    "csv/00_base_data.csv", "csv/01_identifiers.csv", "csv/02_y_target.csv",
    "csv/03_morphological.csv", "csv/04_synergistic_proximity.csv",
    "csv/05_socioeconomic.csv", "csv/06_joker.csv",
]

FINAL_COLUMNS = [
    "osm_id", "name", "lat", "lon", "label", "distance_m",
    "lot_area_sqft", "highway_type", "lanes",
    "dist_bus_stop_m", "dist_hospital_m", "dist_school_m", "dist_park_m",
    "res_ratio", "com_ratio",
    "median_income",
]

CSV_GROUPS = [
    ("csv/01_identifiers.csv",           ["osm_id", "name", "lat", "lon", "distance_m", "label"]),
    ("csv/03_morphological.csv",         ["highway_type", "lanes", "lot_area_sqft"]),
    ("csv/04_synergistic_proximity.csv", ["dist_bus_stop_m", "dist_hospital_m", "dist_school_m", "dist_park_m"]),
    ("csv/05_socioeconomic.csv",         ["com_ratio", "res_ratio"]),
    ("csv/06_joker.csv",                 ["median_income"]),
]


def _merge_site_csvs(site_name, run_id):
    """Merge intermediate CSVs for one site into a final per-site CSV."""
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

    out_path = CSV_DIR / f"{site_name}_{run_id}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    _log(f"{site_name} | merged → {out_path.name} ({len(df)} rows)")
    return out_path


def _cleanup_intermediates():
    for f in INTERMEDIATES:
        p = SCRAPER_DIR / f
        if p.exists():
            p.unlink()


def _run_site(loc, enabled_notebooks, run_id):
    """Run pipeline for a single location. Returns True if all notebooks succeeded."""
    site_name = loc["name"]
    pipeline_state["current_site"] = site_name
    _log(f"{'='*40}")
    _log(f"SITE: {site_name} ({loc['lat']}, {loc['lon']})")
    _log(f"{'='*40}")

    all_ok = True
    for nb in NOTEBOOKS:
        if pipeline_state["cancelled"]:
            # mark remaining as skipped
            for nb2 in NOTEBOOKS:
                if pipeline_state["sites"][site_name].get(nb2["id"]) == "pending":
                    pipeline_state["sites"][site_name][nb2["id"]] = "skipped"
            return False

        if nb["id"] not in enabled_notebooks:
            pipeline_state["sites"][site_name][nb["id"]] = "skipped"
            _log(f"{site_name} | {nb['id']} SKIPPED")
            continue

        params = _get_nb_params(loc, nb["id"])
        ok = _run_notebook(nb["file"], params, site_name, nb["id"])
        if not ok:
            all_ok = False
            # Mark remaining as skipped on failure (except already run ones)
            for nb2 in NOTEBOOKS:
                if pipeline_state["sites"][site_name].get(nb2["id"]) == "pending":
                    pipeline_state["sites"][site_name][nb2["id"]] = "skipped"
            break

    if all_ok:
        _merge_site_csvs(site_name, run_id)

    _cleanup_intermediates()
    return all_ok


def _pipeline_thread(site_names, locations_data, enabled_notebooks):
    """Background thread that runs the pipeline for given sites."""
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
    """Run pipeline for one or more sites."""
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 409

    body = request.get_json(force=True)
    site_names = body.get("sites", [])               # list of site names to run
    enabled_notebooks = body.get("notebooks", [])     # list of notebook IDs to run

    if not site_names:
        return jsonify({"error": "No sites specified"}), 400
    if not enabled_notebooks:
        return jsonify({"error": "No notebooks selected"}), 400

    # Load locations
    if not LOCATIONS_FILE.exists():
        return jsonify({"error": "locations.json not found"}), 404
    locations_data = json.loads(LOCATIONS_FILE.read_text(encoding="utf-8"))

    # Init status
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
    """Run pipeline for all locations."""
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 409

    body = request.get_json(force=True)
    enabled_notebooks = body.get("notebooks", ["01", "02", "03", "04", "05"])

    if not LOCATIONS_FILE.exists():
        return jsonify({"error": "locations.json not found"}), 404
    locations_data = json.loads(LOCATIONS_FILE.read_text(encoding="utf-8"))
    site_names = [loc["name"] for loc in locations_data]

    # Init status
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
    """Merge all per-site CSVs into a combined CSV."""
    import pandas as pd

    if not CSV_DIR.exists():
        return jsonify({"error": "csv/ directory not found"}), 404

    # Find the most recent CSV for each site
    site_csvs = {}
    for f in CSV_DIR.glob("*_20*.csv"):
        if f.stem.startswith("combined"):
            continue
        # Extract site name (everything before the timestamp)
        stem = f.stem
        # timestamp format: YYYY-MM-DD_HH-MM-SS (19 chars)
        if len(stem) > 20:
            site_name = stem[:-20]  # remove _YYYY-MM-DD_HH-MM-SS
            if site_name not in site_csvs or f.stat().st_mtime > site_csvs[site_name][1]:
                site_csvs[site_name] = (f, f.stat().st_mtime)

    if not site_csvs:
        return jsonify({"error": "No site CSVs found"}), 404

    dfs = []
    for site_name, (csv_path, _) in site_csvs.items():
        dfs.append(pd.read_csv(csv_path))

    df_combined = pd.concat(dfs, ignore_index=True)
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
    return jsonify(NOTEBOOKS)


# ── main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Locations file : {LOCATIONS_FILE}")
    print(f"CSV directory  : {CSV_DIR}")
    print(f"Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
