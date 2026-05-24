"""
Shared Overpass API utilities: cached queries, BallTree spatial matching, BBOX computation.
"""
import hashlib
import json
import os
import time

import numpy as np
import requests
from sklearn.neighbors import BallTree

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
HEADERS = {"User-Agent": "urban-ml-pipeline/1.0 (research project)"}
EARTH_RADIUS_M = 6_371_000


def _cache_path(query, cache_dir="cache"):
    h = hashlib.sha1(query.encode()).hexdigest()
    return os.path.join(cache_dir, f"{h}.json")


def _is_error_response(data):
    """Check if an Overpass response contains a server-side error (OOM, timeout)."""
    remark = data.get("remark", "")
    return "error" in remark.lower() or "out of memory" in remark.lower()


def query_overpass_cached(query, cache_dir="cache", max_retries=6):
    """Execute an Overpass query with SHA1-based file caching and endpoint rotation."""
    os.makedirs(cache_dir, exist_ok=True)
    cp = _cache_path(query, cache_dir)
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        # Invalidate cached error responses
        if _is_error_response(data):
            print(f"    Cache invalidated (server error): {data.get('remark', '')[:80]}")
            os.remove(cp)
        else:
            return data

    last_error = None
    for attempt in range(max_retries):
        ep = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            r = requests.post(ep, data={"data": query}, headers=HEADERS, timeout=300)
            r.raise_for_status()
            data = r.json()
            # Don't cache server-side errors (OOM, timeout)
            if _is_error_response(data):
                last_error = RuntimeError(f"Server error: {data.get('remark', '')}")
                wait = 10 + attempt * 5
                print(f"    Overpass attempt {attempt+1}/{max_retries}: {data.get('remark', '')[:80]}, "
                      f"retrying in {wait}s...")
                time.sleep(wait)
                continue
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except Exception as e:
            last_error = e
            wait = 10 + attempt * 5
            print(f"    Overpass attempt {attempt+1}/{max_retries} failed ({ep.split('/')[2]}), "
                  f"retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Overpass query failed after {max_retries} attempts: {last_error}")


def compute_bbox(df_grid, buffer=0.015):
    """Compute a bounding box string from grid cell coordinates."""
    lat_min = df_grid["cell_lat"].min() - buffer
    lat_max = df_grid["cell_lat"].max() + buffer
    lon_min = df_grid["cell_lon"].min() - buffer
    lon_max = df_grid["cell_lon"].max() + buffer
    return f"{lat_min},{lon_min},{lat_max},{lon_max}"


def balltree_assign(poi_records, df_grid, max_dist_m=500):
    """
    Assign POI records to nearest grid cells using BallTree (haversine).

    Args:
        poi_records: list of dicts, each must have 'lat' and 'lon' keys
        df_grid: DataFrame with 'cell_id', 'cell_lat', 'cell_lon'
        max_dist_m: maximum distance in meters for assignment

    Returns:
        dict mapping cell_id -> list of indices into poi_records
    """
    cell_ids = df_grid["cell_id"].tolist()
    result = {cid: [] for cid in cell_ids}

    if not poi_records:
        return result

    cell_coords_rad = np.radians(df_grid[["cell_lat", "cell_lon"]].values)
    poi_coords = np.radians([[p["lat"], p["lon"]] for p in poi_records])

    tree = BallTree(cell_coords_rad, metric="haversine")
    distances, indices = tree.query(poi_coords, k=1)

    for j, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist * EARTH_RADIUS_M <= max_dist_m:
            result[cell_ids[idx]].append(j)

    return result
