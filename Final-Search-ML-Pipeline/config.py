"""
Central configuration for the Urban ML Pipeline.
City registry, column mappings, feature constants, and data adapters.
"""
import os
import pandas as pd
import numpy as np

# ── Pipeline Constants ────────────────────────────────

GRID_CELL_SIZE_M = 150
MIN_FEATURES_PER_CELL = 3
INCLUDE_OTHER = False

FEATURE_COLS = [
    "amenity_density", "amenity_ratio_food_drink",
    "avg_floors", "avg_yearbuilt", "building_count", "total_bldg_area",
    "landuse_entropy", "shop_density_km2",
    "office_density", "road_density_primary", "transit_stop_density",
    "intersection_density", "nightlife_density",
]

CLASS_MAP = {
    "Commercial": "Commercial",
    "Mixed-Use": "Residential",
    "Residential": "Residential",
}
CLASS_MAP_BINARY = {
    "Commercial": "Commercial",
    "Residential": "Residential",
}
CLASS_MAP_3CLASS = {
    "Commercial": "Commercial",
    "Mixed-Use": "Mixed-Use",
    "Residential": "Residential",
}
CLASS_MAP_OTHER = {
    "Institutional": "Other",
    "Open Space": "Other",
    "Industrial": "Other",
    "Infrastructure": "Other",
}

# ── OSM Zone Type Rules (for OSM-only mode) ──────────

OSM_ZONE_RULES = {
    "Residential": {"osm_landuse": ["residential"], "threshold": 0.40},
    "Commercial": {"osm_landuse": ["commercial", "retail"], "threshold": 0.40},
    "Industrial": {"osm_landuse": ["industrial"], "threshold": 0.30},
    "Institutional": {"osm_landuse": ["institutional", "education", "religious"], "threshold": 0.30},
    "Open Space": {"osm_landuse": ["recreation_ground", "grass", "park", "forest", "meadow"], "threshold": 0.30},
    "Mixed-Use": {"description": "No single category dominates"},
}

OSM_BUILDING_CATEGORIES = {
    "residential": ["apartments", "house", "detached", "semidetached_house",
                     "terrace", "dormitory", "residential"],
    "commercial": ["commercial", "office", "retail", "warehouse",
                    "supermarket", "kiosk"],
    "industrial": ["industrial", "manufacture"],
    "institutional": ["school", "university", "hospital", "church",
                      "public", "government", "civic"],
}

# ── Landuse category groups for Shannon entropy (OSM) ─

OSM_LANDUSE_GROUPS = {
    "residential": "residential",
    "commercial": "commercial", "retail": "commercial",
    "industrial": "industrial",
    "institutional": "institutional", "education": "institutional",
    "religious": "institutional",
    "recreation_ground": "recreation", "grass": "recreation",
    "park": "recreation", "forest": "recreation", "meadow": "recreation",
    "farmland": "agriculture", "farmyard": "agriculture",
    "railway": "infrastructure", "highway": "infrastructure",
    "military": "infrastructure", "cemetery": "infrastructure",
}

PLURALITY_THRESHOLD = 0.40

# ── ANN / ML Config ──────────────────────────────────

ANN_CONFIG = {
    "hidden_layers": [64, 32],
    "learning_rate": 0.01,
    "epochs": 500,
    "validation_split": 0.1,
}
KMEANS_MAX_K = 10
TSNE_PERPLEXITY = 30

# ── Datasets base path ───────────────────────────────

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

# ── City Registry ────────────────────────────────────

CITY_REGISTRY = {
    "NYC": {
        "city_query": "Manhattan, New York, USA",
        "data_source": "property",
        "group": "ground_truth",
        "property_file": os.path.join(DATASETS_DIR, "pluto_25v4.csv"),
        "column_map": {
            "latitude": "latitude",
            "longitude": "longitude",
            "landuse": "landuse",
            "lot_area": "lotarea",
            "num_floors": "numfloors",
            "year_built": "yearbuilt",
            "num_buildings": "numbldgs",
            "building_area": "bldgarea",
        },
        "filter_column": "borocode",
        "filter_values": ["1"],  # Manhattan
        "landuse_map": {
            "01": "Residential", "02": "Residential", "03": "Residential",
            "04": "Mixed-Use", "05": "Commercial", "06": "Industrial",
            "07": "Infrastructure", "08": "Institutional",
            "09": "Open Space", "10": "Infrastructure", "11": "Infrastructure",
        },
    },
    "Philadelphia": {
        "city_query": "Philadelphia, Pennsylvania, USA",
        "data_source": "property",
        "group": "ground_truth",
        "property_file": os.path.join(DATASETS_DIR, "philadelphia_opa.csv"),
        "column_map": {
            "latitude": "lat",
            "longitude": "lng",
            "landuse": "zoning",
            "lot_area": "total_area",
            "num_floors": "number_stories",
            "year_built": "year_built",
            "num_buildings": None,
            "building_area": "total_livable_area",
        },
        "landuse_map": {
            # Philadelphia zoning codes
            "RSA1": "Residential", "RSA2": "Residential", "RSA3": "Residential",
            "RSA4": "Residential", "RSA5": "Residential", "RSA6": "Residential",
            "RSD1": "Residential", "RSD2": "Residential", "RSD3": "Residential",
            "RTA1": "Residential", "RM1": "Residential", "RM2": "Residential",
            "RM3": "Residential", "RM4": "Residential",
            "CMX1": "Commercial", "CMX2": "Commercial", "CMX2.5": "Commercial",
            "CMX3": "Commercial", "CMX4": "Commercial", "CMX5": "Commercial",
            "CA1": "Commercial", "CA2": "Commercial",
            "I1": "Industrial", "I2": "Industrial", "I3": "Industrial",
            "ICMX": "Industrial", "IP": "Industrial",
            "SP-INS": "Institutional", "SP-STA": "Institutional",
            "SP-AIR": "Infrastructure", "SP-PO-A": "Open Space",
            "SP-PO-P": "Open Space",
        },
    },
    "DC": {
        "city_query": "Washington, District of Columbia, USA",
        "data_source": "osm",
        "group": "osm_only",
        "property_file": None,
        "column_map": {
            "latitude": "Y",
            "longitude": "X",
            "landuse": "USECODE",
            "lot_area": "LANDAREA",
            "num_floors": "STORIES",
            "year_built": "AYB",
            "num_buildings": None,
            "building_area": "GBA",
        },
        "landuse_map": {
            # DC use codes (selected common ones)
            "011": "Residential", "012": "Residential", "013": "Residential",
            "014": "Residential", "015": "Residential", "016": "Residential",
            "017": "Residential", "019": "Residential",
            "021": "Residential", "022": "Residential", "023": "Residential",
            "024": "Residential", "025": "Residential", "029": "Residential",
            "031": "Residential", "032": "Residential", "039": "Residential",
            "041": "Commercial", "042": "Commercial", "043": "Commercial",
            "044": "Commercial", "045": "Commercial", "046": "Commercial",
            "047": "Commercial", "048": "Commercial", "049": "Commercial",
            "051": "Industrial", "052": "Industrial", "053": "Industrial",
            "061": "Institutional", "062": "Institutional", "063": "Institutional",
            "071": "Open Space", "072": "Open Space", "073": "Open Space",
        },
    },
    "Chicago": {
        "city_query": "Chicago, Illinois, USA",
        "data_source": "hybrid",
        "group": "ground_truth",
        "property_file": os.path.join(DATASETS_DIR, "chicago_cook_county.csv"),
        "column_map": {
            "latitude": "latitude",
            "longitude": "longitude",
            "landuse": "class",
            "lot_area": None,
            "num_floors": None,
            "year_built": None,
            "num_buildings": None,
            "building_area": None,
        },
        "filter_column": "municipality_name",
        "filter_values": ["CITY OF CHICAGO"],
        "landuse_map": {
            # Cook County property class codes (first digit)
            "2": "Residential",
            "3": "Commercial", "5": "Commercial",
            "4": "Industrial",
            "6": "Institutional", "7": "Institutional",
            "1": "Open Space",
        },
        "landuse_map_prefix": True,  # match by first digit
    },
    "SF": {
        "city_query": "San Francisco, California, USA",
        "data_source": "osm",
        "group": "osm_only",
        "property_file": None,
        "column_map": {
            "latitude": None,  # geometry-based, needs geocoding
            "longitude": None,
            "landuse": "LANDUSE",
            "lot_area": None,
            "num_floors": None,
            "year_built": None,
            "num_buildings": None,
            "building_area": None,
        },
        "landuse_map": {
            "CIE": "Institutional", "MED": "Institutional",
            "MIPS": "Commercial", "RETAIL": "Commercial", "VISITOR": "Commercial",
            "PDR": "Industrial",
            "RESIDENT": "Residential", "MIXED": "Mixed-Use",
            "OpenSpace": "Open Space",
        },
    },
    "LA": {
        "city_query": "Los Angeles, California, USA",
        "data_source": "osm",
        "group": "osm_only",
        "property_file": None,
        "column_map": {
            "latitude": "CENTER_LAT",
            "longitude": "CENTER_LON",
            "landuse": "UseType",
            "lot_area": "SQFTmain",
            "num_floors": None,
            "year_built": "YearBuilt",
            "num_buildings": "Units",
            "building_area": "SQFTmain",
        },
        "landuse_map": {
            "Residential": "Residential",
            "Commercial": "Commercial",
            "Industrial": "Industrial",
            "Institutional": "Institutional",
            "Recreational": "Open Space",
        },
        "landuse_map_contains": True,  # partial string match
    },
}


# ── Helper: load and normalize property data ─────────

def load_property_data(city_key):
    """
    Load a city's property dataset and normalize columns to canonical names.
    Returns DataFrame with columns: latitude, longitude, landuse, lot_area,
    num_floors, year_built, num_buildings, building_area.
    Returns None if property file doesn't exist.
    """
    city = CITY_REGISTRY[city_key]
    prop_file = city.get("property_file")

    if not prop_file or not os.path.exists(prop_file):
        print(f"  Property file not found: {prop_file}")
        return None

    col_map = city["column_map"]

    # Determine which columns to read
    usecols = []
    rename_map = {}
    for canonical, actual in col_map.items():
        if actual is not None:
            usecols.append(actual)
            rename_map[actual] = canonical

    # Add filter column if present
    filter_col = city.get("filter_column")
    if filter_col and filter_col not in usecols:
        usecols.append(filter_col)

    print(f"  Loading {os.path.basename(prop_file)}...")
    df = pd.read_csv(prop_file, usecols=usecols, dtype={col_map.get("landuse", "landuse"): str},
                     low_memory=False)

    # Apply filter
    if filter_col and "filter_values" in city:
        df = df[df[filter_col].astype(str).isin(city["filter_values"])].copy()
        print(f"  Filtered by {filter_col}: {len(df)} rows")

    # Rename columns
    df = df.rename(columns=rename_map)

    # Convert numeric columns
    for col in ["latitude", "longitude", "lot_area", "num_floors",
                "year_built", "num_buildings", "building_area"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without coordinates
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude", "longitude"]).copy()

    # Clean values
    if "year_built" in df.columns:
        df.loc[df["year_built"] < 1700, "year_built"] = np.nan
    if "num_floors" in df.columns:
        df.loc[df["num_floors"] <= 0, "num_floors"] = np.nan

    # Map landuse to zone categories
    if "landuse" in df.columns:
        lu_map = city.get("landuse_map", {})
        if city.get("landuse_map_prefix"):
            # Match by first digit (e.g., Chicago class codes)
            df["zone_cat"] = df["landuse"].astype(str).str[0].map(lu_map).fillna("Unknown")
        elif city.get("landuse_map_contains"):
            # Partial string match (e.g., LA use types)
            def _match_contains(val):
                val_str = str(val)
                for key, zone in lu_map.items():
                    if key.lower() in val_str.lower():
                        return zone
                return "Unknown"
            df["zone_cat"] = df["landuse"].apply(_match_contains)
        else:
            df["zone_cat"] = df["landuse"].map(lu_map).fillna("Unknown")

    print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def get_class_map(mode=None):
    """Return the active class mapping based on mode or INCLUDE_OTHER.

    Parameters
    ----------
    mode : str or None
        "binary"  -> CLASS_MAP_BINARY (Commercial / Residential only)
        "3class"  -> CLASS_MAP_3CLASS (Commercial / Mixed-Use / Residential)
        None      -> default behaviour (CLASS_MAP, optionally merged with OTHER)
    """
    if mode == "binary":
        return dict(CLASS_MAP_BINARY)
    if mode == "3class":
        cm = dict(CLASS_MAP_3CLASS)
        if INCLUDE_OTHER:
            cm.update(CLASS_MAP_OTHER)
        return cm
    cm = dict(CLASS_MAP)
    if INCLUDE_OTHER:
        cm.update(CLASS_MAP_OTHER)
    return cm
