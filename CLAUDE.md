# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Urban zone classification pipeline that predicts land use (Commercial vs Residential) from OpenStreetMap data across 6 US cities. The core research question: can observable urban characteristics extracted from OSM predict official zoning without access to proprietary property datasets?

## Repository Structure

- **`Final-Search-ML-Pipeline/`** — Active pipeline (5 Python scripts + 1 Jupyter notebook)
- **`previous_workflows/`** — Archived earlier iterations (General-OSM-Scraper, Grid-Finding, Zone-Finding, ML_Plot, Site_Scraper_App, etc.). Not actively used.
- **`assets/`** — Course materials and presentations
- **`datasets/`** — Property datasets (gitignored). PLUTO, OPA, Cook County CSVs.

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

## Important Conventions

- Notebooks are JSON — when editing f-strings, always use explicit `\n` escapes, never literal newlines inside strings (they produce `SyntaxError: unterminated string literal`).
- CSV outputs go to per-city folders under `Final-Search-ML-Pipeline/csv/`. All CSVs, outputs, and cache are gitignored.
- The `cache/` directory stores Overpass API responses as JSON files keyed by SHA1 of the query string. Safe to delete to force re-fetch.
- Property datasets stored in `datasets/` at repo root (gitignored). Download manually — see dataset URLs in config.py comments.
- ML notebook uses subsampling for expensive operations: SOM (8K), t-SNE (8K), SVC training (10K), SVC tuning (5K), ANN (10K).
- Overpass API queries for very large cities (LA) may hit memory limits. The pipeline handles this gracefully with retry logic and fallback to landuse-only when building queries fail.
