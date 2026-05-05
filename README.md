# OSMnx Data Scraper

Extracts commercial establishment data from OpenStreetMap and assembles it into a single ML-ready CSV. Supports scraping multiple city neighborhoods in a single pipeline run.

---

## Setup

**Requirements:** Python 3.11, Windows

```powershell
# 1. Allow script execution (once per session)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 2. Run setup — creates .venv, installs dependencies, registers Jupyter kernel
.\setup.ps1
```

Then in VS Code, select the kernel **OSMnx Scraper (Python 3.11)** for all notebooks.

---

## Running the pipeline

Open `General-OSM-Scraper/00_orchestrator.ipynb` and run the cells in order:

| Cell | Action |
|------|--------|
| 1 | Imports |
| 2 | Pipeline config — set `RUN_MODULE[nb] = False` to skip a step; loads `locations.json` |
| 3 | Runs notebooks 01–05 for **each location** defined in `locations.json` |
| 4 | Merges all per-location CSVs → `csv/combined_<RUN_ID>.csv` |
| 5 | Preview of combined dataset (dtypes + first 10 rows) |
| 6 | Column completeness report |

> **Joker notebook (06):** Disabled by default (`INCLUDE_JOKER = False`). Set to `True` to include median income data from the US Census Bureau API — slow, skip if not needed.

---

## Multi-location scraping

Locations are defined in `General-OSM-Scraper/locations.json`. Add, remove, or modify entries to control which neighborhoods are scraped in each run.

```json
[
    {
        "name": "times_square",
        "lat": 40.7589,
        "lon": -73.9851,
        "walk_minutes": 15.0,
        "walk_speed_m_min": 80.0
    },
    {
        "name": "upper__east_side",
        "lat": 40.7735,
        "lon": -73.9565,
        "walk_minutes": 20.0,
        "walk_speed_m_min": 80.0
    }
]
```

Each location generates its own intermediate CSV. All locations are combined at the end into a single `combined_<RUN_ID>.csv` with a `location` column identifying the source neighborhood.

**Current locations:** Times Square, Upper East Side, Harlem, Lower East Side

---

## Output columns

| Group | Columns |
|-------|---------|
| Location | `location` |
| Identifiers | `osm_id`, `lat`, `lon`, `distance_m` |
| Y-target | `label` |
| Morphological | `highway_type`, `lanes`, `lot_area_sqft` |
| Synergistic proximity | `dist_bus_stop_m`, `dist_hospital_m`, `dist_school_m`, `dist_park_m` |
| Socioeconomic | `com_ratio`, `res_ratio` |
| Joker *(optional)* | `median_income` |

---

## Notebooks

```
General-OSM-Scraper/
├── 00_orchestrator.ipynb           ← start here; loops over all locations
├── 01_identifiers.ipynb            ← fetches shops from OSM (Overpass API)
├── 02_y_target.ipynb               ← amenity label classification
├── 03_morphological.ipynb          ← street width (highway type, lanes) + plot area
├── 04_synergistic_proximity.ipynb  ← distances to bus stops, hospitals, schools, parks
├── 05_socioeconomic.ipynb          ← residential / commercial building ratios
├── 06_joker.ipynb                  ← median income (US Census Bureau API) — optional
├── locations.json                  ← list of neighborhoods to scrape
└── csv/                            ← per-location intermediates + combined output
```

---

## Changelog

### 2026-05-04 — Multi-location pipeline + column cleanup
- Added `locations.json` to define multiple neighborhoods as scrape targets
- Orchestrator now iterates over all locations, running the full pipeline per location
- All per-location outputs are merged into a single `combined_<RUN_ID>.csv` with a `location` column
- Removed unused / redundant columns from the final CSV output
- Updated morphological notebook (03) and socioeconomic notebook (05) for consistency with new schema
- `INCLUDE_JOKER` flag added to orchestrator to skip the slow median-income step (default: `False`)
- Added column completeness report cell to orchestrator

### 2026-05-01 — Pipeline V01
- Initial end-to-end pipeline wiring via `00_orchestrator.ipynb`
- Versioned CSV output: `final_dataset_v001_<YYYY-MM-DD_HH-MM-SS>.csv`
- `.gitignore` updated to exclude intermediate and final CSV files
- PLUTO data removed from pipeline scope
