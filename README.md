# OSMnx Data Scraper — Site Scraper

Extracts commercial establishment data from OpenStreetMap for NYC neighborhoods and assembles it into ML-ready CSVs. Two interfaces: an interactive Flask web app and a Jupyter notebook pipeline.

---

## Quick start

**Requirements:** Python 3.11, Windows

```powershell
# 1. Allow script execution (once per session)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 2. Run setup — creates .venv, installs all dependencies, registers Jupyter kernel
.\setup.ps1
```

> If you get a `.venv` error on first run, delete the `.venv` folder and re-run `setup.ps1`. This can happen if the project was cloned to a different drive than where the venv was originally created.

---

## Option A — Flask web UI (recommended)

Double-click **`launch_site_scraper.bat`** or use the VS Code build task (`Ctrl+Shift+B`).

Then open **http://localhost:5000** in your browser.

Features:
- Add locations by clicking the map or entering coordinates
- Select which notebooks to run (core steps 01-06, plus enrichment steps 07-13)
- Real-time pipeline progress per site
- CSV browser with pagination and column completeness view
- Walking route visualization via OSMnx
- Session export/import
- OSM tag configuration

---

## Option B — Notebook pipeline directly

Open `General-OSM-Scraper/00_orchestrator.ipynb` and run cells in order using the **OSMnx Scraper (Python 3.11)** kernel.

The orchestrator loops over locations in `locations.json`, runs notebooks 01-13 per location via papermill, and merges all outputs into `csv/combined_<RUN_ID>.csv`.

---

## Adding locations

Edit `General-OSM-Scraper/locations.json`:

```json
[
    {
        "name": "times_square",
        "lat": 40.7589,
        "lon": -73.9851,
        "walk_minutes": 15.0,
        "walk_speed_m_min": 80.0
    }
]
```

**Current locations:** Times Square, Upper East Side, Harlem, Lower East Side

---

## Notebooks

```
General-OSM-Scraper/
├── 00_orchestrator.ipynb       <- start here; loops over all locations
├── 01_identifiers.ipynb        <- fetches shops from OSM (Overpass API)
├── 02_y_target.ipynb           <- amenity label classification
├── 03_morphological.ipynb      <- street width + plot area (needs PLUTO)
├── 04_synergistic_proximity.ipynb  <- distances to bus, hospital, school, park
├── 05_socioeconomic.ipynb      <- residential/commercial building ratios (needs PLUTO)
├── 06_census_income.ipynb      <- median income via TigerWeb + Census API
├── 07_rent_proxy.ipynb         <- assessed value per sqft from PLUTO (needs PLUTO)
├── 08_building_age.ipynb       <- year built from PLUTO (needs PLUTO)
├── 09_foot_traffic.ipynb       <- pedestrian demand rank (needs pedestrian data)
├── 10_subway_distance.ipynb    <- haversine distance to nearest subway entrance
├── 11_shop_density.ipynb       <- shop count within 100m radius (BallTree)
├── 12_population_density.ipynb <- Census tract population via ACS-5 + Tiger API
├── 13_shop_type_mix.ipynb      <- Shannon entropy of label distribution within 200m
├── locations.json              <- neighborhoods to scrape
└── csv/                        <- per-location intermediates + combined output
```

Notebooks 03, 05, 07, 08 require NYC PLUTO tax-lot data (`ramy/NYC_pluto_25v4_csv/`) — auto-skipped if absent.
Notebook 09 requires NYC Pedestrian Mobility Plan data (`ramy/pedestrian_mobility/`) — auto-skipped if absent.

---

## Output columns

| Group | Columns |
|-------|---------|
| Location | `location` |
| Identifiers | `osm_id`, `lat`, `lon`, `distance_m` |
| Y-target | `label` |
| Morphological | `highway_type`, `lanes`, `lot_area_sqft` |
| Proximity | `dist_bus_stop_m`, `dist_hospital_m`, `dist_school_m`, `dist_park_m` |
| Socioeconomic | `com_ratio`, `res_ratio` |
| Census | `median_income`, `population_density` |
| PLUTO-derived | `rent_proxy_per_sqft`, `year_built` |
| Foot traffic | `pedestrian_demand_rank` |
| Subway | `dist_subway_m` |
| Density | `shop_density_100m`, `shop_type_entropy_200m` |

---

## ML Visualization

`ML_Plot/NYC_classification.ipynb` runs logistic regression and XGBoost on the combined CSV, generating plots to `ML_Plot/outputs/latest/`. Auto-detects the latest `combined_*.csv` — no hardcoded path needed.

---

## Project structure

```
OSMnx-data-scraper/
├── General-OSM-Scraper/        <- notebook pipeline
├── Site_Scraper_App/           <- Flask web UI
│   ├── Site_Scraper_App.py     <- backend (~1100 lines, 40+ endpoints)
│   ├── static/app.js           <- frontend (vanilla JS + Leaflet)
│   └── templates/index.html
├── ML_Plot/                    <- ML classification + plots
├── launch_site_scraper.bat     <- launches the web UI
├── setup.ps1                   <- one-time environment setup
└── requirements.txt            <- all Python dependencies
```

---

## Changelog

### 2026-05-10 — Fix setup and missing dependencies
- Rebuilt `.venv` after project moved from E: to C: drive (old venv was corrupted)
- Added `flask`, `papermill`, `xgboost`, `matplotlib` to `requirements.txt` (were missing)
- Fixed `setup.ps1` encoding issue (smart quotes broke PowerShell parser)
- Updated `setup.ps1` next-steps instructions to reference current app entry points

### 2026-05-07 — Notebooks 07-13 + enrichment timeline UI
- Added notebooks 07-13 (rent proxy, building age, foot traffic, subway distance, shop density, population density, shop type mix)
- Flask UI groups notebooks 07-13 under a single "Enrichment" toggle with dot-based mini-timeline
- Cached TigerWeb census tract centroids (`cache/tract_centroids.json`)

### 2026-05-04 — Multi-location pipeline + column cleanup
- Added `locations.json` for multi-neighborhood scraping
- Orchestrator iterates all locations, merges into `combined_<RUN_ID>.csv` with `location` column
- Removed unused/redundant columns from final CSV

### 2026-05-01 — Pipeline V01
- Initial end-to-end pipeline via `00_orchestrator.ipynb`
- Versioned CSV output
- `.gitignore` excludes intermediate and final CSV files
