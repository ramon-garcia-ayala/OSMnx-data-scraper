import geopandas as gpd
import pandas as pd
import osmnx as ox
import numpy as np
from scipy.spatial import cKDTree
import os

# --- CONFIG ---
NUTS_CODE = "FR10"
OUTPUT_DIR = "output"
OUTPUT_FILE = f"{OUTPUT_DIR}/buildings_{NUTS_CODE}_ml.xlsx"
SAMPLE_PER_TYPE = 10000

# --- SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

storage_opts = {
    "anon": True,
    "client_kwargs": {"endpoint_url": "https://s3.eubucco.com"}
}

# --- FETCH EUBUCCO ---
print(f"Fetching EUBUCCO data for {NUTS_CODE}...")
path = f"s3://eubucco/v0.2/buildings/parquet/nuts_id={NUTS_CODE}/{NUTS_CODE}.parquet"
gdf = gpd.read_parquet(path, storage_options=storage_opts)
print(f"Loaded {len(gdf)} buildings")

# --- REPROJECT & LAT/LON ---
gdf = gdf.to_crs("EPSG:4326")
gdf["latitude"] = gdf.geometry.centroid.y
gdf["longitude"] = gdf.geometry.centroid.x

# --- CHECK AVAILABLE TYPES ---
print("\nAvailable types:")
print(gdf["type"].value_counts())
print("\nAvailable subtypes:")
print(gdf["subtype"].value_counts())

# --- FILTER: use 'type' for residential, 'subtype' for commercial/industrial ---
conditions = (
    ((gdf["type"] == "residential")) |
    ((gdf["subtype"] == "commercial")) |
    ((gdf["subtype"] == "industrial"))
)
gdf_filtered = gdf[conditions].copy()

# assign label
gdf_filtered["label"] = gdf_filtered.apply(
    lambda row: "residential" if row["type"] == "residential" else row["subtype"],
    axis=1
)

print(f"\nAfter filtering: {len(gdf_filtered)} buildings")
print(gdf_filtered["label"].value_counts())

# --- SAMPLE BALANCED ---
chunks = []
for label, group in gdf_filtered.groupby("label"):
    sample = group.sample(min(len(group), SAMPLE_PER_TYPE), random_state=42)
    chunks.append(sample)
sampled = pd.concat(chunks).reset_index(drop=True)

print(f"\nSampled {len(sampled)} buildings total")
print(sampled["label"].value_counts())

# --- FETCH OSM FEATURES FOR PARIS ---
print("\nFetching OSM points of interest for Paris...")
place = "Paris, France"

def get_osm_coords(tags):
    try:
        gdf_poi = ox.features_from_place(place, tags=tags)
        gdf_poi = gdf_poi.to_crs("EPSG:4326")
        coords = np.array([
            [geom.centroid.y, geom.centroid.x]
            for geom in gdf_poi.geometry
        ])
        return coords
    except Exception as e:
        print(f"  Warning: could not fetch {tags} — {e}")
        return np.empty((0, 2))

print("  Fetching metro stations...")
metro_coords = get_osm_coords({"railway": "station"})

print("  Fetching schools...")
school_coords = get_osm_coords({"amenity": "school"})

print("  Fetching hospitals...")
hospital_coords = get_osm_coords({"amenity": "hospital"})

print("  Fetching parks...")
park_coords = get_osm_coords({"leisure": "park"})

print("  Fetching supermarkets...")
supermarket_coords = get_osm_coords({"shop": "supermarket"})

# --- COMPUTE DISTANCES ---
print("\nComputing distances...")
building_coords = sampled[["latitude", "longitude"]].values

def nearest_distance(building_coords, poi_coords):
    if len(poi_coords) == 0:
        return np.full(len(building_coords), np.nan)
    tree = cKDTree(poi_coords)
    dist, _ = tree.query(building_coords, k=1)
    return dist * 111000  # degrees to meters

sampled["dist_metro_m"]       = nearest_distance(building_coords, metro_coords)
sampled["dist_school_m"]      = nearest_distance(building_coords, school_coords)
sampled["dist_hospital_m"]    = nearest_distance(building_coords, hospital_coords)
sampled["dist_park_m"]        = nearest_distance(building_coords, park_coords)
sampled["dist_supermarket_m"] = nearest_distance(building_coords, supermarket_coords)

# --- EXPORT ---
df = sampled[[
    "label",
    "latitude", "longitude",
    "height", "floors", "construction_year",
    "dist_metro_m", "dist_school_m", "dist_hospital_m",
    "dist_park_m", "dist_supermarket_m"
]].copy()

print(f"\nSaving to {OUTPUT_FILE}...")
df.to_excel(OUTPUT_FILE, index=False)
print(f"Done! {len(df)} rows saved.")
print(df["label"].value_counts())