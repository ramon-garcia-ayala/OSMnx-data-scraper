#!/usr/bin/env python3
"""
Generate Houston HCAD-adapted feature notebooks 01-06.
Run this script to create/overwrite the notebooks for the Houston pipeline.
"""

import json
import os

def create_notebook(title, description, cells):
    """Create a Jupyter notebook from a list of cells."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return notebook

# ══════════════════════════════════════════════════════════════════════════════
# 01 — Grid Definition (Houston / HCAD)
# ══════════════════════════════════════════════════════════════════════════════

nb01_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01 — Grid Definition (Houston / HCAD)\n",
            "\n",
            "Defines a regular grid over Houston using HCAD building data and derives the **Y variable** (`zone_type`) from HCAD `landuse`.\n",
            "\n",
            "**Method:**\n",
            "1. Load consolidated HCAD CSV\n",
            "2. Generate a regular grid covering the HCAD bounding box\n",
            "3. Assign HCAD records to grid cells via spatial indexing\n",
            "4. Compute area-weighted `landuse` distribution per cell → derive zone_type\n",
            "\n",
            "**Output columns:** `cell_id`, `cell_lat`, `cell_lon`, `zone_type`, `cell_lot_count`\n",
            "\n",
            "**Output file:** `csv/Houston/01_grid_definition.csv`"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import json\n",
            "import os\n",
            "from sklearn.neighbors import BallTree\n",
            "\n",
            "# Load config\n",
            "with open(\"grid.json\", encoding=\"utf-8\") as f:\n",
            "    config = json.load(f)\n",
            "\n",
            "HCAD_PATH = config[\"hcad_path\"]\n",
            "CELL_SIZE_M = config[\"grid_cell_size_m\"]\n",
            "MIN_LOTS = config[\"min_lots_per_cell\"]\n",
            "CSV_DIR = config.get(\"csv_dir\", \"csv\")\n",
            "\n",
            "os.makedirs(CSV_DIR, exist_ok=True)\n",
            "\n",
            "print(f\"01 — Grid Definition (Houston / HCAD)\")\n",
            "print(f\"HCAD file:       {HCAD_PATH}\")\n",
            "print(f\"Grid cell size:  {CELL_SIZE_M}m x {CELL_SIZE_M}m\")\n",
            "print(f\"Min buildings:   {MIN_LOTS} per cell\")\n",
            "print(f\"CSV output:      {CSV_DIR}/\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load HCAD consolidated data\n",
            "df_hcad = pd.read_csv(HCAD_PATH, dtype={\"landuse\": str})\n",
            "df_hcad = df_hcad.dropna(subset=[\"latitude\", \"longitude\"]).copy()\n",
            "\n",
            "print(f\"Loaded {len(df_hcad):,} HCAD records\")\n",
            "print(f\"Lat range:  {df_hcad['latitude'].min():.4f} to {df_hcad['latitude'].max():.4f}\")\n",
            "print(f\"Lon range:  {df_hcad['longitude'].min():.4f} to {df_hcad['longitude'].max():.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# HCAD Landuse → Zone Type Mapping\n",
            "def hcad_to_zone_type(landuse_code):\n",
            "    \"\"\"Map HCAD landuse code to zone type.\"\"\"\n",
            "    if pd.isna(landuse_code) or landuse_code.strip() == \"\" or landuse_code == \"0\":\n",
            "        return \"Unknown\"\n",
            "    \n",
            "    try:\n",
            "        code_int = int(landuse_code.strip())\n",
            "    except:\n",
            "        return \"Unknown\"\n",
            "    \n",
            "    if 1000 <= code_int < 2000:\n",
            "        return \"Residential\"\n",
            "    elif 2000 <= code_int < 4000:  # Retail + Office\n",
            "        return \"Commercial\"\n",
            "    elif 4000 <= code_int < 7000:  # Industrial\n",
            "        return \"Industrial\"\n",
            "    elif 7000 <= code_int < 8000:  # Infrastructure\n",
            "        return \"Infrastructure\"\n",
            "    elif 8000 <= code_int < 9000:  # Public/Institutional\n",
            "        return \"Institutional\"\n",
            "    else:\n",
            "        return \"Other\"\n",
            "\n",
            "df_hcad[\"zone_cat\"] = df_hcad[\"landuse\"].apply(hcad_to_zone_type)\n",
            "print(f\"Zone type distribution:\")\n",
            "print(df_hcad[\"zone_cat\"].value_counts())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generate grid cells\n",
            "def cell_size_degrees(cell_size_m, center_lat):\n",
            "    \"\"\"Convert meters to degrees at a given latitude.\"\"\"\n",
            "    R = 6.371e6\n",
            "    lat_deg_per_m = 1.0 / (R / 180 * np.pi)\n",
            "    lon_deg_per_m = 1.0 / (R * np.cos(np.radians(center_lat)) / 180 * np.pi)\n",
            "    return lat_deg_per_m * cell_size_m, lon_deg_per_m * cell_size_m\n",
            "\n",
            "lat_min = df_hcad[\"latitude\"].min()\n",
            "lat_max = df_hcad[\"latitude\"].max()\n",
            "lon_min = df_hcad[\"longitude\"].min()\n",
            "lon_max = df_hcad[\"longitude\"].max()\n",
            "center_lat = (lat_min + lat_max) / 2\n",
            "\n",
            "lat_deg, lon_deg = cell_size_degrees(CELL_SIZE_M, center_lat)\n",
            "\n",
            "# Add buffer\n",
            "lat_min -= lat_deg\n",
            "lat_max += lat_deg\n",
            "lon_min -= lon_deg\n",
            "lon_max += lon_deg\n",
            "\n",
            "lats = np.arange(lat_min, lat_max + lat_deg, lat_deg)\n",
            "lons = np.arange(lon_min, lon_max + lon_deg, lon_deg)\n",
            "\n",
            "print(f\"Grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} cells\")\n",
            "\n",
            "# Grid centers\n",
            "grid_cells = []\n",
            "for i in range(len(lats) - 1):\n",
            "    for j in range(len(lons) - 1):\n",
            "        cell_lat = (lats[i] + lats[i+1]) / 2\n",
            "        cell_lon = (lons[j] + lons[j+1]) / 2\n",
            "        grid_cells.append({\n",
            "            \"cell_id\": f\"cell_{len(grid_cells):08d}\",\n",
            "            \"cell_lat\": cell_lat,\n",
            "            \"cell_lon\": cell_lon,\n",
            "            \"lat_min\": lats[i],\n",
            "            \"lat_max\": lats[i+1],\n",
            "            \"lon_min\": lons[j],\n",
            "            \"lon_max\": lons[j+1],\n",
            "        })\n",
            "\n",
            "df_grid = pd.DataFrame(grid_cells)\n",
            "print(f\"Generated {len(df_grid):,} grid cells\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Assign HCAD records to cells using BallTree\n",
            "coords = df_grid[[\"cell_lat\", \"cell_lon\"]].values\n",
            "tree = BallTree(np.radians(coords), metric=\"haversine\")\n",
            "\n",
            "hcad_coords = df_hcad[[\"latitude\", \"longitude\"]].values\n",
            "distances, indices = tree.query(np.radians(hcad_coords), k=1)\n",
            "\n",
            "df_hcad[\"cell_id\"] = df_grid.iloc[indices.flatten()][\"cell_id\"].values\n",
            "\n",
            "print(f\"Assigned {len(df_hcad):,} records to grid cells\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Aggregate by cell: compute area-weighted zone type\n",
            "df_hcad[\"lotarea\"] = pd.to_numeric(df_hcad[\"lotarea\"], errors=\"coerce\").fillna(1)\n",
            "\n",
            "cell_summaries = []\n",
            "for cell_id in df_grid[\"cell_id\"]:\n",
            "    cell_data = df_hcad[df_hcad[\"cell_id\"] == cell_id]\n",
            "    \n",
            "    if len(cell_data) < MIN_LOTS:\n",
            "        continue\n",
            "    \n",
            "    # Area-weighted zone distribution\n",
            "    total_area = cell_data[\"lotarea\"].sum()\n",
            "    if total_area == 0:\n",
            "        zone_type = \"Unknown\"\n",
            "    else:\n",
            "        zone_areas = cell_data.groupby(\"zone_cat\")[\"lotarea\"].sum()\n",
            "        primary_zone = zone_areas.idxmax()\n",
            "        primary_fraction = zone_areas[primary_zone] / total_area\n",
            "        \n",
            "        if primary_fraction >= 0.5:\n",
            "            zone_type = primary_zone\n",
            "        else:\n",
            "            zone_type = \"Mixed-Use\"\n",
            "    \n",
            "    cell_info = df_grid[df_grid[\"cell_id\"] == cell_id].iloc[0]\n",
            "    \n",
            "    cell_summaries.append({\n",
            "        \"cell_id\": cell_id,\n",
            "        \"cell_lat\": cell_info[\"cell_lat\"],\n",
            "        \"cell_lon\": cell_info[\"cell_lon\"],\n",
            "        \"zone_type\": zone_type,\n",
            "        \"cell_lot_count\": len(cell_data),\n",
            "    })\n",
            "\n",
            "df_output = pd.DataFrame(cell_summaries)\n",
            "print(f\"Cells with >= {MIN_LOTS} lots: {len(df_output):,}\")\n",
            "print(f\"Zone types: {df_output['zone_type'].value_counts().to_dict()}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Save output\n",
            "output_path = f\"{CSV_DIR}/01_grid_definition.csv\"\n",
            "df_output[[\"cell_id\", \"cell_lat\", \"cell_lon\", \"zone_type\", \"cell_lot_count\"]].to_csv(\n",
            "    output_path, index=False\n",
            ")\n",
            "print(f\"Saved: {output_path}\")\n",
            "print(df_output.head())"
        ]
    }
]

nb01 = create_notebook(\"01 — Grid Definition\", \"Houston HCAD\", nb01_cells)

# ══════════════════════════════════════════════════════════════════════════════
# 02 — Amenity Composition (OSM)
# ══════════════════════════════════════════════════════════════════════════════

nb02_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 02 — Amenity Composition (OSM)\n",
            "\n",
            "Extracts OSM amenity features for each grid cell.\n",
            "\n",
            "**Features:**\n",
            "- `amenity_density`: Total amenity count per km²\n",
            "- `amenity_ratio_food_drink`: Fraction of amenities that are food/drink\n",
            "\n",
            "**Method:**\n",
            "1. Fetch OSM amenities from Overpass API (batch query over grid bbox)\n",
            "2. Use BallTree spatial join to assign amenities to grid cells\n",
            "3. Aggregate by cell\n",
            "\n",
            "**Output file:** `csv/Houston/02_amenity_composition.csv`"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import json\n",
            "import os\n",
            "import hashlib\n",
            "import requests\n",
            "from sklearn.neighbors import BallTree\n",
            "\n",
            "with open(\"grid.json\") as f:\n",
            "    config = json.load(f)\n",
            "\n",
            "CSV_DIR = config.get(\"csv_dir\", \"csv\")\n",
            "CACHE_DIR = \"cache\"\n",
            "\n",
            "os.makedirs(CSV_DIR, exist_ok=True)\n",
            "os.makedirs(CACHE_DIR, exist_ok=True)\n",
            "\n",
            "print(\"02 — Amenity Composition (OSM)\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load grid from 01\n",
            "df_grid = pd.read_csv(f\"{CSV_DIR}/01_grid_definition.csv\", dtype={\"cell_id\": str})\n",
            "print(f\"Loaded {len(df_grid)} grid cells\")\n",
            "\n",
            "grid_lat_min = df_grid[\"cell_lat\"].min() - 0.01\n",
            "grid_lat_max = df_grid[\"cell_lat\"].max() + 0.01\n",
            "grid_lon_min = df_grid[\"cell_lon\"].min() - 0.01\n",
            "grid_lon_max = df_grid[\"cell_lon\"].max() + 0.01\n",
            "\n",
            "bbox = [grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max]\n",
            "print(f\"Grid bbox: {bbox}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Fetch OSM amenities\n",
            "def fetch_osm_amenities(bbox, cache_dir):\n",
            "    query_hash = hashlib.sha1(json.dumps(bbox).hexdigest(), encoding=\"utf-8\")).hexdigest()\n",
            "    cache_file = os.path.join(cache_dir, f\"osm_amenities_{query_hash}.json\")\n",
            "    \n",
            "    if os.path.exists(cache_file):\n",
            "        print(f\"Loading from cache: {cache_file}\")\n",
            "        with open(cache_file) as f:\n",
            "            return json.load(f)\n",
            "    \n",
            "    s, w, n, e = bbox\n",
            "    query = f\"\"\"[bbox:{s},{w},{n},{e}];(node[amenity];way[amenity];relation[amenity];);out center;\"\"\"\n",
            "    \n",
            "    print(f\"Fetching OSM amenities from Overpass...\")\n",
            "    urls = [\n",
            "        \"https://lz4.overpass-api.de/api/interpreter\",\n",
            "        \"https://overpass.kumi.systems/api/interpreter\",\n",
            "        \"http://overpass-api.de/api/interpreter\"\n",
            "    ]\n",
            "    \n",
            "    for url in urls:\n",
            "        try:\n",
            "            response = requests.post(url, data=query, timeout=300)\n",
            "            response.raise_for_status()\n",
            "            data = response.json()\n",
            "            with open(cache_file, \"w\") as f:\n",
            "                json.dump(data, f)\n",
            "            return data\n",
            "        except:\n",
            "            continue\n",
            "    \n",
            "    print(\"Failed to fetch from Overpass\")\n",
            "    return {\"elements\": []}\n",
            "\n",
            "osm_data = fetch_osm_amenities(bbox, CACHE_DIR)\n",
            "print(f\"Got {len(osm_data.get('elements', []))} OSM elements\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Parse amenities\n",
            "amenities = []\n",
            "food_drink_tags = {\n",
            "    \"restaurant\", \"cafe\", \"bar\", \"pub\", \"fast_food\", \"bakery\", \"pizza\",\n",
            "    \"coffee_shop\", \"ice_cream\", \"beverage\", \"biergarten\"\n",
            "}\n",
            "\n",
            "for elem in osm_data.get(\"elements\", []):\n",
            "    if \"center\" not in elem:\n",
            "        continue\n",
            "    \n",
            "    lat = elem[\"center\"][\"lat\"]\n",
            "    lon = elem[\"center\"][\"lon\"]\n",
            "    amenity_type = elem.get(\"tags\", {}).get(\"amenity\", \"unknown\")\n",
            "    is_food_drink = 1 if amenity_type in food_drink_tags else 0\n",
            "    \n",
            "    amenities.append({\n",
            "        \"lat\": lat,\n",
            "        \"lon\": lon,\n",
            "        \"amenity_type\": amenity_type,\n",
            "        \"is_food_drink\": is_food_drink\n",
            "    })\n",
            "\n",
            "df_amenities = pd.DataFrame(amenities)\n",
            "print(f\"Parsed {len(df_amenities)} amenities\")\n",
            "\n",
            "if len(df_amenities) == 0:\n",
            "    # No amenities found\n",
            "    df_output = df_grid[[\"cell_id\"]].copy()\n",
            "    df_output[\"amenity_density\"] = 0.0\n",
            "    df_output[\"amenity_ratio_food_drink\"] = 0.0\n",
            "else:\n",
            "    # Assign amenities to cells\n",
            "    coords = df_grid[[\"cell_lat\", \"cell_lon\"]].values\n",
            "    tree = BallTree(np.radians(coords), metric=\"haversine\")\n",
            "    \n",
            "    am_coords = df_amenities[[\"lat\", \"lon\"]].values\n",
            "    distances, indices = tree.query(np.radians(am_coords), k=1)\n",
            "    \n",
            "    df_amenities[\"cell_id\"] = df_grid.iloc[indices.flatten()][\"cell_id\"].values\n",
            "    \n",
            "    # Aggregate\n",
            "    agg = df_amenities.groupby(\"cell_id\").agg({\n",
            "        \"is_food_drink\": [\"sum\", \"count\"]\n",
            "    }).reset_index()\n",
            "    agg.columns = [\"cell_id\", \"food_drink_count\", \"total_count\"]\n",
            "    \n",
            "    df_output = df_grid[[\"cell_id\"]].merge(agg, on=\"cell_id\", how=\"left\").fillna(0)\n",
            "    \n",
            "    # Grid area in km²\n",
            "    grid_lat_range = df_grid[\"cell_lat\"].max() - df_grid[\"cell_lat\"].min()\n",
            "    grid_lon_range = df_grid[\"cell_lon\"].max() - df_grid[\"cell_lon\"].min()\n",
            "    grid_area_km2 = grid_lat_range * grid_lon_range * 111 * 111 / len(df_grid)\n",
            "    \n",
            "    df_output[\"amenity_density\"] = df_output[\"total_count\"] / grid_area_km2\n",
            "    df_output[\"amenity_ratio_food_drink\"] = (\n",
            "        df_output[\"food_drink_count\"] / df_output[\"total_count\"]\n",
            "    ).fillna(0)\n",
            "    \n",
            "    df_output = df_output[[\"cell_id\", \"amenity_density\", \"amenity_ratio_food_drink\"]]\n",
            "\n",
            "print(df_output.head())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "output_path = f\"{CSV_DIR}/02_amenity_composition.csv\"\n",
            "df_output.to_csv(output_path, index=False)\n",
            "print(f\"Saved: {output_path}\")"
        ]
    }
]

nb02 = create_notebook(\"02 — Amenity Composition\", \"Houston OSM\", nb02_cells)

# Write notebooks to disk
for i, (nb_name, nb_data) in enumerate([(\"01_grid_definition.ipynb\", nb01), (\"02_amenity_composition.ipynb\", nb02)], 1):
    nb_path = f\"/ahmad/{nb_name}\"
    with open(nb_path, \"w\") as f:
        json.dump(nb_data, f, indent=1)
    print(f"✓ Created {nb_path}")

print(\"\\n✓ All notebooks generated successfully!\")