# OSMnx Data Scraper

Extracts commercial establishment data from OpenStreetMap and assembles it into a single ML-ready CSV.

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
| 2 | Pipeline config — set `RUN_MODULE[nb] = False` to skip a step |
| 3 | Runs notebooks 01–05 automatically |
| 4 | **[Optional]** Runs notebook 06 (median income) — slow, skip if not needed |
| 5 | Merges all outputs → `General-OSM-Scraper/csv/final_dataset_vXXX_YYYY-MM-DD.csv` and deletes intermediates |

---

## Output columns

| Group | Columns |
|-------|---------|
| Identifiers | `osm_id`, `lat`, `lon` |
| Y-target | `amenity_label` |
| Morphological | `street_width_m`, `plot_area_m2` |
| Synergistic proximity | `dist_bus_stop_m`, `dist_hospital_m`, `dist_school_m`, `dist_park_m` |
| Socioeconomic | `residential_ratio`, `commercial_ratio` |
| Joker *(optional)* | `median_income` |

---

## Notebooks

```
General-OSM-Scraper/
├── 00_orchestrator.ipynb       ← start here
├── 01_identifiers.ipynb        ← fetches shops from OSM (Overpass API)
├── 02_y_target.ipynb           ← amenity label classification
├── 03_morphological.ipynb      ← street width + plot area (Overpass API)
├── 04_synergistic_proximity.ipynb  ← distances to bus stops, hospitals, schools, parks
├── 05_socioeconomic.ipynb      ← residential / commercial building ratios
├── 06_joker.ipynb              ← median income (US Census Bureau API)
└── csv/                        ← intermediate + final CSVs
```
