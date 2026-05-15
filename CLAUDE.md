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
- **Zone Finding pipeline:** Run `Zone-Finding/00_orchestrator.ipynb` cells in order. First run takes ~5–10 min (batch Overpass API queries); subsequent runs ~2–3 min (cached). All OSM notebooks (02, 05, 07, 09) use single batch bbox queries + BallTree matching instead of per-tract queries.
- **Grid Finding pipeline:** Run `Grid-Finding/00_orchestrator.ipynb` cells in order. Change `BOROUGH`, `CELL_SIZE_M`, `INCLUDE_OTHER`, and `COMBINED_SHEETS` in the first code cell to configure the run. Same batch Overpass + BallTree pattern as Zone-Finding but with a regular grid. Produces heatmap visualization. First run ~5–10 min; subsequent runs ~2–3 min (cached).

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

### Zone Finding Pipeline (`Zone-Finding/`)

Predicts urban zone types for Manhattan census tracts (~310) using a classification model trained on PLUTO + OSM + Census features. The raw PLUTO landuse produces 5 zone types (Residential, Commercial, Mixed-Use, Institutional, Open Space); notebook 10 merges to 2 classes for binary classification: **Commercial** (~35) and **Residential** (~238, includes Mixed-Use). Other tracts (Institutional + Open Space) are dropped — too noisy with insufficient OSM signal to distinguish from Residential.

The orchestrator (`00_orchestrator.ipynb`) runs notebooks 01–09 via papermill, merges CSVs on `tract_id`, then runs the ML notebook (10). Configuration lives in `zones.json`.

Notebook responsibilities:
- **01** — Zone definition: derives Y variable (`zone_type`) from PLUTO `landuse` via area-weighted aggregation per census tract (`bct2020`)
- **02** — Amenity composition: counts/ratios of OSM amenity categories per tract (single batch Overpass query + BallTree)
- **03** — Building characteristics: avg floors, year built, lot area from PLUTO (`needs_pluto`)
- **04** — Land use mix: Shannon entropy + HHI of landuse distribution (`needs_pluto`). No raw area ratios (prevents Y leakage)
- **05** — Accessibility: subway/bus distances, transit stop count, intersection density (3 batch Overpass queries + BallTree, no OSMnx dependency)
- **06** — Socioeconomic: median income, population density, poverty rate (Census ACS-5 API, direct tract FIPS join)
- **07** — Tourism intensity: hotel count, tourism POI count/density/ratio (single batch Overpass query + BallTree)
- **08** — Pedestrian activity: pedestrian rank from NYC Pedestrian Mobility Plan (`needs_pedestrian`). Vectorized regex extraction.
- **09** — Commercial density: shop count, density, type entropy, brand ratio (single batch Overpass query + BallTree)
- **10** — ML classification: Logistic Regression, Random Forest, XGBoost with stratified train/test split. Binary classification: Commercial vs Residential (Other tracts dropped). Feature selection reduces ~40 columns to 10 features with proven signal: amenity_density, shop_density_km2, brand_ratio, tourism_density, landuse_entropy, amenity_ratio_food_drink, avg_floors, avg_yearbuilt, building_count, total_bldg_area. Drops non-discriminating features (subway/bus distances, pedestrian rank, intersection density) and redundant features (correlated pairs).

Feature strategy (no data overlap): PLUTO for building/land features, OSM for amenities/transit/tourism, Census for socioeconomic. PLUTO area ratios excluded to prevent Y variable leakage.

Dropped features (no discriminating power in Manhattan — identical distributions across classes):
- `dist_subway_mean`, `dist_bus_mean` — subway/bus coverage too uniform across Manhattan
- `pedestrian_rank_mean`, `pedestrian_rank_max`, `pct_high_traffic_segments` — high foot traffic in both commercial and residential areas
- `intersection_density`, `transit_stop_count` — flat across all zone types

Dropped features (redundant — high correlation with kept features):
- `amenity_count_total` (r=1.00 with amenity_density), `amenity_count_food_drink`, `amenity_count_retail`, `amenity_count_services`, `amenity_count_office`, `amenity_count_hotel`, `amenity_count_entertainment`, `amenity_count_education`, `amenity_count_healthcare` — raw counts redundant with density/ratio
- `amenity_ratio_retail`, `amenity_ratio_services`, `amenity_ratio_office`, `amenity_ratio_hotel`, `amenity_ratio_entertainment`, `amenity_ratio_education`, `amenity_ratio_healthcare` — only food_drink ratio kept (strongest signal)
- `shop_count` (r=0.91 with shop_density_km2), `hotel_count` (r=0.96 with tourism_density), `tourism_poi_count`, `tourism_ratio` — density versions kept
- `lot_area_mean`, `landuse_hhi` (r=-0.92 with landuse_entropy) — redundant with kept features

Data flow:
```
zones.json → 00_orchestrator → papermill(01..09) → per-notebook CSVs → combined_zones CSV → 10_ml_classification
```

### Grid Finding Pipeline (`Grid-Finding/`)

Grid-based version of Zone-Finding. Replaces irregular census tracts with a **regular grid** (default 150m x 150m). Portable to any NYC borough — all parameters are configured in a single cell at the top of the orchestrator:

- **`BOROUGH`**: `["MN"]` (Manhattan), `["BK"]` (Brooklyn), `["QN"]` (Queens), `["BX"]` (Bronx), `["SI"]` (Staten Island), or combinations like `["MN", "BK"]`
- **`CELL_SIZE_M`**: grid cell size in meters (default 150)
- **`MIN_LOTS_PER_CELL`**: minimum PLUTO lots for a cell to be valid (default 3)
- **`INCLUDE_OTHER`**: `False` = binary (Commercial vs Residential), `True` = 3-class (+ Other)
- **`COMBINED_SHEETS`**: `True` = generate combined ML sheets (all boroughs + combined panel) in comparison, `False` = skip (faster)

Cell counts vary by borough: Manhattan ~1,810, Brooklyn ~4,500+, Queens ~6,000+, Bronx ~2,500+, Staten Island ~2,000+ (at 150m).

Leaner pipeline: only 9 notebooks, only the 10 proven features are computed (no redundant/noise columns). Skips accessibility (05), socioeconomic (06), and pedestrian (08) notebooks entirely since all their features were noise.

The orchestrator (`00_orchestrator.ipynb`) **loops over each borough independently**: for each one it writes `grid.json`, runs notebooks 01–06, merges CSVs, then runs ML (07) and heatmap (08). Each borough gets its own folder (`csv/Manhattan/`, `outputs/Manhattan/`) that is overwritten on every run. After all boroughs finish, comparison (09) runs if 2+ boroughs exist.

Notebook responsibilities:
- **01** — Grid definition: generates 150m grid, clips to PLUTO convex hull, assigns lots to cells, derives zone_type via area-weighted landuse aggregation
- **02** — Amenity composition: amenity_density + amenity_ratio_food_drink (single batch Overpass query + BallTree)
- **03** — Building characteristics: avg_floors, avg_yearbuilt, building_count, total_bldg_area from PLUTO
- **04** — Land use mix: landuse_entropy only (HHI dropped as redundant)
- **05** — Tourism intensity: tourism_density only (single batch Overpass query + BallTree)
- **06** — Commercial density: shop_density_km2, brand_ratio (single batch Overpass query + BallTree)
- **07** — ML classification: LR, XGBoost, RF. Classification mode controlled by `include_other` toggle in `grid.json`: `false` (default) = binary Commercial vs Residential, `true` = 3-class with Other. Uses `class_weight="balanced"`. Exports predictions CSV for heatmap
- **08** — Heatmap visualization: adapts to `include_other` toggle. Binary mode: diverging colormap (blue→red by P(Commercial)). 3-class mode: categorical colors (red/blue/green) with confidence alpha. Both include contextily basemap + folium interactive map + summary dashboard
- **09** — Borough comparison: auto-runs when 2+ borough folders have `07_predictions.csv`. Generates: comparison bar charts (cell counts, accuracy, class distribution), combined multi-borough heatmap (static + folium), side-by-side per-borough heatmaps and dashboards, normalized feature means comparison. When `COMBINED_SHEETS` is enabled, also trains ML on merged data from all boroughs and generates 9 combined sheets (one per plot type) with per-borough thumbnails + combined panel, saved to `outputs/Comparison/Combined Plots/`. Skipped for single-borough runs

Data flow:
```
For each borough in BOROUGH:
  grid.json → papermill(01..06) → per-notebook CSVs → combined_grid.csv → 07_ml → 08_heatmap
After all boroughs:
  09_comparison (if 2+ unique boroughs)
```

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
- Zone-Finding notebooks 03, 04 depend on PLUTO (`needs_pluto` flag in `zones.json`). Notebook 08 depends on Pedestrian Mobility data (`needs_pedestrian`). Both auto-skip with NaN-filled CSVs if data is absent.
- Zone-Finding CSV outputs go to `Zone-Finding/csv/` and are gitignored. The `cache/` directory stores Overpass API responses as JSON files keyed by SHA1 of the query string. Cache persists across kernel restarts and machine reboots — only invalidated if query parameters change (e.g., different bbox or coordinates). Safe to delete `Zone-Finding/cache/` to force re-fetch from Overpass. Batch queries produce fewer, larger cache files than per-tract queries.
- Zone-Finding is fully independent from General-OSM-Scraper — no shared code or utilities.
- Zone-Finding combined CSV: ~310 rows (Manhattan census tracts) x ~35 feature columns. Unit of analysis is the census tract (`bct2020` from PLUTO), not individual POIs.
- Grid-Finding is fully independent from both Zone-Finding and General-OSM-Scraper — no shared code.
- Grid-Finding CSV outputs go to per-borough folders under `Grid-Finding/csv/` (e.g. `csv/Manhattan/`), cache to `Grid-Finding/cache/`. Same Overpass cache strategy as Zone-Finding. Intermediate CSVs (01–06, 07_predictions) are gitignored; combined CSVs and outputs are tracked.
- Grid-Finding combined CSV: `combined_grid.csv` inside the borough folder. ~1,810 rows (150m grid cells) x 15 columns (5 base + 10 features). Unit of analysis is a 150m x 150m grid cell. Portable to other boroughs by changing `borough_filter` in `grid.json`.
- Grid-Finding notebooks 03, 04 depend on PLUTO (`needs_pluto` flag). OSM notebooks (02, 05, 06) work anywhere with Overpass API.
- Grid-Finding heatmap (notebook 08) requires `folium` for interactive map. Static matplotlib heatmap always works.
- Grid-Finding comparison (notebook 09) scans borough folders under `csv/` for `07_predictions.csv`. Only generates output when 2+ boroughs exist. Single-borough runs update only their own `outputs/{borough}/` folder.
