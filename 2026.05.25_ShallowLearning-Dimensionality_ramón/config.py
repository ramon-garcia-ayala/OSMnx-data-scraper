"""
Central configuration for the Dimensionality_ramón bundle.

Edit ONLY this file to change parameters for Tutorial 8 and Tutorial 9.
Both notebooks import from here with `from config import *`.
"""

# ============================================================
#  EDITABLE PARAMETERS — change here, both notebooks react
# ============================================================

# --- Data source ---
# 'property' -> 5 features from official datasets (PLUTO/OPA/Cook County). Forces ground_truth cities.
# 'osm'      -> 9 features from OpenStreetMap. Respects CITIES / CITY_GROUP.
# 'mixed'    -> 14 combined features. Forces ground_truth. Exports two separate CSVs.
SOURCE = 'osm'

# --- City filter ---
# None | ['NYC'] | ['NYC','Chicago','Philadelphia'] | ['DC','SF','LA']
# Ignored when SOURCE in ('property','mixed') -> forced to ground_truth.
CITIES = None

# Group shortcut when CITIES = None:  'ground_truth' | 'osm_only' | None
CITY_GROUP = None

# --- Class balancing ---
# 'undersample' | 'none'
BALANCE_METHOD = 'none'

# --- Manual feature selection (overrides auto-by-SOURCE) ---
# None -> respect SOURCE; or explicit list (4-10 recommended for DL)
FEATURES = None

# Columns to drop after FEATURES/SOURCE has been applied
EXCLUDE_FEATURES = []

# --- Reproducibility & split ---
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
#  CONSTANTS (rarely change)
# ============================================================

GROUP_MAP = {
    'ground_truth': ['NYC', 'Philadelphia', 'Chicago'],
    'osm_only':     ['DC', 'SF', 'LA'],
}

PROPERTY_FEATURES = [
    'cell_lot_count',
    'avg_floors',
    'avg_yearbuilt',
    'total_bldg_area',
    'building_count',
]

OSM_FEATURES = [
    'amenity_density',
    'amenity_ratio_food_drink',
    'landuse_entropy',
    'shop_density_km2',
    'office_density',
    'road_density_primary',
    'transit_stop_density',
    'intersection_density',
    'nightlife_density',
]

ALL_FEATURES = PROPERTY_FEATURES + OSM_FEATURES   # 14 columns


# ============================================================
#  PATHS
# ============================================================

CSV_PATH = r"E:\IAAC Local GIT Repositories\OSMnx-data-scraper\Final-Search-ML-Pipeline\csv\all_cities_combined.csv"
EXPORT_DIR = r"E:\IAAC Local GIT Repositories\OSMnx-data-scraper\ramon\csv"
_BUNDLE_DIR = r"E:\IAAC Local GIT Repositories\OSMnx-data-scraper\2026.05.25_ShallowLearning-Dimensionality_ramón"
PLOTS_DIR = _BUNDLE_DIR + r"\outputs"                           # legacy / fallback
PLOTS_DIR_DIMRED   = _BUNDLE_DIR + r"\outputs\DimReduction"     # used by DimReduction_OSM.ipynb
PLOTS_DIR_SHALLOW  = _BUNDLE_DIR + r"\outputs\ShallowLearning"  # used by ShallowLearning_OSM.ipynb


# ============================================================
#  HELPER FUNCTIONS — used by both notebooks
# ============================================================

def resolve_cities(data_raw_cities):
    """Return the list of cities to use given SOURCE/CITIES/CITY_GROUP.

    `data_raw_cities`: pd.Series or list of cities available in the dataset.
    """
    if SOURCE in ('property', 'mixed'):
        return list(GROUP_MAP['ground_truth'])
    if SOURCE == 'osm':
        if CITIES is not None:
            return [CITIES] if isinstance(CITIES, str) else list(CITIES)
        if CITY_GROUP is not None:
            return list(GROUP_MAP[CITY_GROUP])
        return sorted(set(data_raw_cities))
    raise ValueError(f"Unknown SOURCE: {SOURCE!r}. Use 'property', 'osm' or 'mixed'.")


def resolve_features():
    """Return the final feature list based on SOURCE / FEATURES / EXCLUDE_FEATURES."""
    if FEATURES is not None:
        base = list(FEATURES)
    elif SOURCE == 'property':
        base = list(PROPERTY_FEATURES)
    elif SOURCE == 'osm':
        base = list(OSM_FEATURES)
    elif SOURCE == 'mixed':
        base = list(ALL_FEATURES)
    else:
        raise ValueError(f"Unknown SOURCE: {SOURCE!r}")

    unknown = [c for c in base if c not in ALL_FEATURES]
    if unknown:
        raise ValueError(f"FEATURES contains unknown columns: {unknown}")

    cols = [c for c in base if c not in EXCLUDE_FEATURES]
    if not cols:
        raise ValueError("Feature selection ended up empty.")
    return cols
