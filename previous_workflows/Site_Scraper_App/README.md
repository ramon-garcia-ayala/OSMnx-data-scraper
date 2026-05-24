# Site Renderer App

Interactive web UI for the OSMnx commercial establishment scraping pipeline. Built with Flask + Leaflet.js.

## Setup

```bash
pip install flask pandas osmnx networkx papermill
```

## Run

```bash
cd Site_Renderer_App
python Site_Renderer_App.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Features

- **Interactive map** with draggable location markers and walk radius circles
- **City search** via Nominatim geocoding proxy
- **Mini-map** with global overview of all locations
- **Pipeline execution** via papermill with per-site, per-notebook status tracking
- **CSV viewer** with server-side pagination, column sorting, and label color-coding
- **Amenity markers** from CSV data, clustered per-site on the map
- **Walking route computation** via OSMnx (on-demand, click any amenity marker)
- **OSM tag editor** for configuring COMMERCIAL_TAGS and exclude lists, with preset save/load
- **Bulk parameter editing** ("Apply to all" links for walk_minutes and walk_speed)
- **Dual collapsible panels** (Locations on left, Pipeline on right)
- **Column selector** to toggle which CSV columns appear on map popups
- **Session export/import** (download/upload all settings as JSON)
- **Auto dark/light mode** from OS `prefers-color-scheme`, with manual override via localStorage
- **PLUTO auto-detection** (notebooks 03/05 auto-skipped if PLUTO data not found)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/api/locations` | Read/write locations.json |
| GET | `/api/geocode?q=...` | Nominatim geocoding proxy |
| GET | `/api/csv-status` | CSV existence + params per site |
| GET | `/api/csv-history/<site>` | All CSV versions for a site |
| GET | `/api/csv-site/<site>?page=0&size=50&sort=col&dir=asc` | Paginated CSV data |
| GET | `/api/csv-site/<site>/download` | Download raw CSV |
| GET | `/api/csv-markers/<site>` | Lightweight marker data (lat, lon, label, name) |
| POST | `/api/csv-delete` | Delete specific CSV files |
| POST | `/api/csv-cleanup` | Remove intermediate CSVs |
| GET/POST | `/api/osm-tags` | OSM tag configuration |
| GET/POST | `/api/osm-tag-presets` | Tag preset management |
| GET | `/api/session/export` | Export all settings as JSON |
| POST | `/api/session/import` | Import settings from JSON |
| POST | `/api/walking-route` | Compute OSMnx walking route |
| POST | `/api/run-site` | Run pipeline for specific sites |
| POST | `/api/run-all` | Run pipeline for all sites |
| POST | `/api/cancel` | Cancel running pipeline |
| GET | `/api/pipeline-status` | Pipeline execution status |
| POST | `/api/combine` | Merge all site CSVs |
| GET | `/api/notebooks` | Notebook definitions |
| GET | `/api/pluto-status` | PLUTO data availability |

## Project Structure

```
Site_Renderer_App/
  Site_Renderer_App.py    # Flask backend
  templates/
    index.html            # HTML template
  static/
    app.js                # Frontend JavaScript
    style.css             # Apple-style CSS (dark/light)
  README.md               # This file
```

## Notes

- Walking route computation requires `osmnx` and `networkx`. Graph data is cached in memory per area to speed up subsequent route queries.
- Notebooks 03 (Morphological) and 05 (Socioeconomic) depend on NYC PLUTO tax-lot data. If the PLUTO CSV is not found at the expected path, these notebooks are automatically skipped and marked as "NYC only" in the UI.
- OSM tag configuration is saved to `General-OSM-Scraper/osm_tags_config.json`. The pipeline passes this path to notebook 01 as a parameter (`OSM_TAGS_CONFIG`). To use custom tags, the notebook should be updated to read from this file.
