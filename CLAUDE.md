# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OSMnx Data Scraper extracts commercial establishment data from OpenStreetMap for NYC neighborhoods and assembles it into ML-ready CSVs. It has two interfaces: a Jupyter notebook pipeline and an interactive Flask web UI.

## Setup

Requires Python 3.11 on Windows. Run `setup.ps1` (needs `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first). This creates the `.venv`, installs dependencies from `requirements.txt`, and registers the `osmnx-scraper` Jupyter kernel.

If `.venv` exists but is missing `Scripts/` and `Lib/`, it was created on a different drive — delete it and re-run `setup.ps1`. `requirements.txt` includes all dependencies: `flask`, `papermill`, `xgboost`, `matplotlib`, `osmnx`, `scikit-learn`, and all Jupyter/notebook packages.

## Running

- **Flask web UI:** `launch_site_scraper.bat` or VS Code build task (Ctrl+Shift+B). Runs on http://localhost:5000.
- **Notebook pipeline directly:** Run `General-OSM-Scraper/00_orchestrator.ipynb` cells in order using the "OSMnx Scraper (Python 3.11)" kernel.

There are no tests, linters, or CI configured.

## Architecture

### Notebook Pipeline (`General-OSM-Scraper/`)

The orchestrator (`00_orchestrator.ipynb`) loops over locations defined in `locations.json` and runs numbered feature-extraction notebooks (01–13) via papermill. Each notebook receives parameters (location name, lat/lon, walk radius) and writes a per-location CSV. The orchestrator merges all per-location CSVs into `csv/combined_<RUN_ID>.csv`.

Notebook responsibilities:
- **01** — Fetches shops from OSM via Overpass API
- **02** — Classifies amenity labels (y-target)
- **03** — Street width + plot area (requires PLUTO data, auto-skipped if absent)
- **04** — Proximity distances (bus, hospital, school, park)
- **05** — Residential/commercial building ratios (requires PLUTO data)
- **06** — Census median income (uses cached TigerWeb centroids + parallel fetching)
- **07** — Rent proxy: assessed value per sqft from PLUTO as commercial rent estimate (requires PLUTO data)
- **08** — Building age: year built from PLUTO (requires PLUTO data)
- **09** — Foot traffic: pedestrian demand rank from NYC Pedestrian Mobility Plan GeoJSON
- **10** — Subway distance: haversine distance to nearest subway entrance (via Overpass API)
- **11** — Shop density: count of other shops within 100m radius (BallTree)
- **12** — Population density: Census tract population via ACS-5 2022 + Tiger API
- **13** — Shop type mix: Shannon entropy of label distribution within 200m radius

### Flask Web UI (`Site_Scraper_App/`)

Single-file Flask backend (`Site_Scraper_App.py`, ~1100 lines) with 40+ REST endpoints. Runs notebooks via papermill in a background thread. Frontend is vanilla JS (`static/app.js`) with Leaflet.js maps, no build step.

Key backend features: pipeline orchestration, CSV management with pagination, OSMnx walking route computation (with in-memory graph caching), session export/import, OSM tag configuration.

The Flask UI registers all notebooks 01–13. Notebooks 01–06 are "core" and shown as individual checkboxes and large step indicators. Notebooks 07–13 are tagged `group: "enrichment"` and appear in the UI as a single toggle ("Enrichment") with a subtle dot-based mini-timeline in site cards. Dependency auto-skip applies to both: `needs_pluto` (03, 05, 07, 08) and `needs_pedestrian` (09).

### ML Visualization (`ML_Plot/`)

`NYC_classification.ipynb` runs logistic regression and XGBoost on the combined CSV, generating plots to `outputs/latest/`. It auto-detects the latest `combined_*.csv` — no hardcoded path needed.

## Key Data Flow

```
locations.json → 00_orchestrator → papermill(01..13) → per-site CSVs → combined CSV → ML_Plot
                                                                              ↑
                                                    Site_Scraper_App (web UI triggers 01–13)
```

## Important Conventions

- Notebooks 03, 05, 07, and 08 depend on NYC PLUTO tax-lot data (`ramy/NYC_pluto_25v4_csv/`). They auto-skip if the data is not present.
- Notebook 09 depends on NYC Pedestrian Mobility Plan data (`ramy/pedestrian_mobility/`).
- Notebook 12 calls the US Census ACS-5 API and Tiger API at runtime (requires internet).
- OSM tag configuration is stored in `General-OSM-Scraper/osm_tags_config.json` and passed to notebook 01 as the `OSM_TAGS_CONFIG` parameter.
- CSV outputs go to `General-OSM-Scraper/csv/` and are gitignored. Each notebook writes its own intermediate CSV (`csv/07_rent_proxy.csv`, etc.) which the orchestrator merges.
- The `cache/` directories store Overpass API responses to avoid redundant network calls.
- `cache/tract_centroids.json` is a shared cache of TigerWeb census tract centroids used by notebooks 06 and 12. First run fetches ~2000 centroids in parallel; subsequent runs are instant.
- Notebooks are JSON — when editing f-strings, always use explicit `\n` escapes, never literal newlines inside strings (they produce `SyntaxError: unterminated string literal`).
