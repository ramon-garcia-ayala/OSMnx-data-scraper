# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Urban zone classification pipeline that predicts land use (Commercial vs Residential) from OpenStreetMap data across 6 US cities. The core research question: can observable urban characteristics extracted from OSM predict official zoning without access to proprietary property datasets?

## Repository Structure

- **`Final-Search-ML-Pipeline/`** — Multi-city pipeline (5 Python scripts + 1 Jupyter notebook), 6 US cities.
- **`Grid-Finding/`** — Active NYC-only, per-borough workflow. Consolidated config-driven scraper `data_scraping.ipynb` plus a Papermill orchestrator (`00`–`10`). Feature catalogs in `feature_reference/`.
- **`2026.05.25_ShallowLearning-Dimensionality_ramón/`** — Shallow-learning + dimensionality-reduction notebooks driven by `config.py`, run on `all_cities_combined.csv`.
- **`previous_workflows/`** — Archived earlier iterations (General-OSM-Scraper, Zone-Finding, ML_Plot, Site_Scraper_App, etc.). Not actively used.
- **`assets/`** — Course materials and presentations
- **`datasets/`** — Property datasets (gitignored). PLUTO, OPA, Cook County CSVs.
- **`docs/`** — GitHub Pages web map (`index.html`) + its generator `build_timeline_map.py`. Interactive multi-city zoning viewer with a ground-truth/prediction layer slider and an NYC year timeline.

## Setup

Requires Python 3.11 on Windows. `requirements.txt` includes all dependencies: `xgboost`, `matplotlib`, `osmnx`, `scikit-learn`, `tensorflow`, `minisom`, `papermill`, and Jupyter packages.

There are no tests, linters, or CI configured.

## Running

```bash
cd Final-Search-ML-Pipeline
python run_pipeline.py              # all 6 cities
python run_pipeline.py NYC Chicago  # specific cities
# Then open ml_analysis.ipynb for ML + visualization
```

First run takes ~15 min (Overpass API queries); subsequent runs ~2-3 min (cached). LA may take longer due to Overpass memory limits on building queries.

## Architecture (`Final-Search-ML-Pipeline/`)

**File structure:**
- `config.py` — City registry (6 cities), column mappings, constants, `load_property_data()` adapter
- `overpass.py` — Overpass API with SHA1 cache, endpoint rotation, BallTree spatial assignment
- `grid.py` — Grid generation + zone_type derivation (property/hybrid/osm modes)
- `features.py` — 9 feature extractors (amenity, building, landuse, commercial, office, road, transit, intersection, nightlife) + merge
- `run_pipeline.py` — CLI entry point, runs all cities, merges into `csv/all_cities_combined.csv`
- `ml_analysis.ipynb` — Single notebook: EDA, LR/XGB/RF/SVC/ANN classification, K-Means, PCA, t-SNE, SOM, heatmaps, cross-city comparison, Binary vs 3-Class comparison, Transfer Learning
- `experiment_log.md` — Experiment documentation log (in Spanish) tracking iterations, decisions, and results

**City registry:**
| City | Mode | Group | Dataset |
|------|------|-------|---------|
| NYC | property | ground_truth | `datasets/pluto_25v4.csv` (PLUTO) |
| Philadelphia | property | ground_truth | `datasets/philadelphia_opa.csv` (OPA) |
| Chicago | hybrid | ground_truth | `datasets/chicago_cook_county.csv` (Cook County Assessor) |
| DC | osm | osm_only | OSM-only (no local dataset) |
| SF | osm | osm_only | OSM-only (no local dataset) |
| LA | osm | osm_only | OSM-only (no local dataset) |

**Three data modes:**
- `property` — Full local dataset (PLUTO-like): grid from convex hull, zone_type from landuse codes, building features from property data
- `hybrid` — Local dataset for grid/zone_type, OSM fallback for missing building features
- `osm` — OSM-only: `osmnx.geocode_to_gdf()` for boundary, Overpass landuse polygons + building tags

**Experimental design:** Train on Ground Truth cities (property data for zone_type labels), predict on OSM-Only cities (zero-shot transfer). Delta = 2.2% (GT 88.9% vs OSM 86.7%).

**Data flow:**
```
config.py → run_pipeline.py → grid.py + features.py → per-city CSVs → all_cities_combined.csv → ml_analysis.ipynb
```

**Output:** `csv/all_cities_combined.csv` with 60,278 cells × 19 columns. ML notebook generates 22 plots to `outputs/`.

## Architecture (`Grid-Finding/` — NYC per-borough)

Self-contained NYC workflow: grids each borough, derives `zone_type` from PLUTO `landuse`, and extracts PLUTO + OSM features per grid cell (`CELL_SIZE_M`, default 150m). Two ways to run:

- **`data_scraping.ipynb`** — Single consolidated, **config-driven** scraper (preferred). Every CSV column is declared in ONE "COLUMN CONFIG" cell; nothing about features is hardcoded in the engine cells. Edit the config → Restart kernel → Run All.
  - `PLUTO_FEATURES` — `name: (pluto_column, agg[, decimals])`, auto-loaded + auto-aggregated (integer-valued cols become `Int64`)
  - `OSM_POINT_FEATURES` — `name: {tag, values, exclude?, brand_ratio_as?}` (POIs/km²)
  - `OSM_MULTIKEY_FEATURES` — multi-key density + optional value ratio (e.g. `amenity_density` + food/drink ratio)
  - `OSM_ROAD_FEATURE` / `OSM_TRANSIT_FEATURE` / `OSM_INTERSECTION_FEATURE` — special features (set `name=None` to disable)
  - `LANDUSE_TO_ZONE`, `PLURALITY_THRESHOLD` — label rules
  - `COLUMN_ORDER` auto-derives (identity → OSM → PLUTO); a guard raises on duplicate column names
  - Engine: PLUTO built in one groupby pass; OSM uses one cached Overpass query per feature, a single reused `BallTree`, and `np.add.at` accumulation
  - Outputs `csv/<Borough>/combined_grid.csv` + `csv/all_boroughs_combined.csv` (latter prepends a `borough` column)
- **`00_orchestrator.ipynb`** — Papermill orchestrator running notebooks `01`–`10` per borough (grid, features, ML, heatmaps, comparison). Older multi-notebook flow.

`feature_reference/FEATURES_OSM.md` and `FEATURES_PLUTO.md` are copy-paste-ready catalogs of every available OSM tag / PLUTO column.

`zone_type` is always PLUTO-derived (grid geometry + landuse label) — there is no toggle to fully remove PLUTO. Setting `PLUTO_FEATURES={}` + the include toggles to `False` yields an OSM-features-only set while PLUTO still provides grid + label.

## Web map (`docs/` — GitHub Pages)

`docs/index.html` is a self-contained Leaflet + Canvas map served via GitHub Pages. It is **generated** — do not hand-edit it; edit `docs/build_timeline_map.py` and re-run `python docs/build_timeline_map.py`, then commit the new `index.html`.

- **City switcher** (NYC · Chicago · Detroit · San Francisco) — buttons swap the rendered 150 m grid (only the active city's cells are drawn).
- **Layer slider** — per city: prediction layer(s) followed by **Ground truth** (PLUTO/assessor `zone_type`) last. NYC has a year timeline **2012 · 2016 · 2026** (the model's prediction on each year's *historical* OSM, queried via Overpass attic `[date:…]`); Chicago/Detroit/SF have one present-day **Prediction** layer. The slider defaults to the most-recent prediction.
- **Predictions** use the exported NYC XGBoost model in `06_Models/SL_NYC_classification_150m-grid_Distributed-Mixed-Use_acc0.89.../` (`model.json` + `scaler.h5` + `encoder.h5`). The **9 features must be in CSV-native order** (verified against `scaler.mean_`). Chicago/Detroit/SF are scored **zero-shot** with the NYC model (the cross-city transfer design); `avg_floors` is neutralized to the scaler mean for those three because it is not a real measurement there (Chicago has none; Detroit/SF carry a constant placeholder).
- Source CSVs are read from the team Google Drive (`H:\…\TEAM_Notebooks\02_csv-datasets\<grid>\...`), not from this repo. Historical NYC grids come from `01_data-scrapers/NYC_DataScraping_{2012,2016,2018}.ipynb` — copies of `NYC_DataScrapíng.ipynb` that inject `OSM_DATE` (attic). Detroit/SF ground truth is auto-downloaded by `Detroit_DataScraping.ipynb` / `SanFrancisco_DataScraping.ipynb` into `…/CITIES_ground-truth_datasets/`.
- `docs/build_optimized_map.py` and `docs/build_toggle_map.py` are earlier single-purpose builders superseded by `build_timeline_map.py`; `docs/index_full.html.bak` is the original heavy Folium export (gitignored).

## Important Conventions

- Notebooks are JSON — edit `.ipynb` with the NotebookEdit tool, not text Edit. When editing f-strings, always use explicit `\n` escapes, never literal newlines inside strings (they produce `SyntaxError: unterminated string literal`).
- In `Grid-Finding/data_scraping.ipynb`, add/remove/rename any feature by editing ONLY the "COLUMN CONFIG" cell — the engine cells and `COLUMN_ORDER` are derived from it.
- OSM `*_density` columns are **POIs per km²** (count ÷ cell area), so magnitudes scale with `CELL_SIZE_M`; `*_ratio` columns are 0–1 proportions. Scale features only at modeling time (StandardScaler), not during EDA.
- Using PLUTO `comarea`/`resarea` to *define* the `zone_type` label is fine; using them as model *features* is leakage. Same caution applies to OSM `landuse=*` density features vs the PLUTO landuse label.
- Overpass helper rotates mirrors (de / kumi / private.coffee — `maps.mail.ru` dropped for frequent 403s) with exponential backoff and ≥1.5s request spacing to survive 429/403 rate limits on heavy feature sets.
- CSV outputs go to per-city folders under `Final-Search-ML-Pipeline/csv/`. All CSVs, outputs, and cache are gitignored.
- The `cache/` directory stores Overpass API responses as JSON files keyed by SHA1 of the query string. Safe to delete to force re-fetch.
- Property datasets stored in `datasets/` at repo root (gitignored). Download manually — see dataset URLs in config.py comments.
- ML notebook uses subsampling for expensive operations: SOM (8K), t-SNE (8K), SVC training (10K), SVC tuning (5K), ANN (10K).
- Overpass API queries for very large cities (LA) may hit memory limits. The pipeline handles this gracefully with retry logic and fallback to landuse-only when building queries fail.
