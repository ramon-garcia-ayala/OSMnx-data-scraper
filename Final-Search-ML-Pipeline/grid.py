"""
Grid generation and zone_type assignment.
Supports property data (PLUTO-like), hybrid, and OSM-only modes.
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon, box
from shapely import STRtree

from config import (CITY_REGISTRY, GRID_CELL_SIZE_M, MIN_FEATURES_PER_CELL,
                    OSM_BUILDING_CATEGORIES, OSM_ZONE_RULES, PLURALITY_THRESHOLD,
                    load_property_data)
from overpass import query_overpass_cached


def generate_grid(city_key):
    """
    Generate a grid of cells with zone_type classification for a city.

    Returns:
        DataFrame with columns: cell_id, cell_lat, cell_lon, zone_type, cell_lot_count
    """
    city = CITY_REGISTRY[city_key]
    data_source = city["data_source"]
    csv_dir = f"csv/{city_key}"
    os.makedirs(csv_dir, exist_ok=True)

    print(f"\n  01 · Grid Definition ({data_source} mode)")

    if data_source == "property":
        df_grid = _grid_from_property(city_key, city)
    elif data_source == "hybrid":
        df_grid = _grid_from_hybrid(city_key, city)
    else:
        df_grid = _grid_from_osm(city_key, city)

    # Save
    output_path = f"{csv_dir}/01_grid_definition.csv"
    df_grid.to_csv(output_path, index=False, encoding="utf-8")
    print(f"    Saved: {output_path} ({len(df_grid)} cells)")
    print(f"    Zone breakdown:")
    for zt in sorted(df_grid["zone_type"].unique()):
        n = (df_grid["zone_type"] == zt).sum()
        print(f"      {zt:<20s} {n:>5d} cells")

    return df_grid


def _grid_from_property(city_key, city):
    """Generate grid from property/parcel data (PLUTO-like)."""
    df_prop = load_property_data(city_key)
    if df_prop is None or len(df_prop) == 0:
        raise RuntimeError(f"No property data available for {city_key}")

    cell_size = GRID_CELL_SIZE_M
    ref_lat = df_prop["latitude"].mean()
    lat_step = cell_size / 111_000
    lon_step = cell_size / (111_000 * math.cos(math.radians(ref_lat)))
    buffer = lat_step

    lat_min = df_prop["latitude"].min() - buffer
    lon_min = df_prop["longitude"].min() - buffer

    # Convex hull for clipping
    coords = df_prop[["longitude", "latitude"]].values
    hull = ConvexHull(coords)
    hull_polygon = Polygon(coords[hull.vertices])

    # Assign lots to grid cells
    df_prop["grid_row"] = ((df_prop["latitude"] - lat_min) / lat_step).astype(int)
    df_prop["grid_col"] = ((df_prop["longitude"] - lon_min) / lon_step).astype(int)
    df_prop["cell_id"] = ("r" + df_prop["grid_row"].astype(str).str.zfill(4) +
                          "_c" + df_prop["grid_col"].astype(str).str.zfill(4))

    cell_records = []
    for (row, col), group in df_prop.groupby(["grid_row", "grid_col"]):
        lot_count = len(group)
        cell_lat = lat_min + (row + 0.5) * lat_step
        cell_lon = lon_min + (col + 0.5) * lon_step

        if not hull_polygon.contains(Point(cell_lon, cell_lat)):
            continue
        if lot_count < MIN_FEATURES_PER_CELL:
            continue

        cell_id = f"r{row:04d}_c{col:04d}"

        # Zone type via area-weighted plurality
        if "lot_area" in group.columns and group["lot_area"].sum() > 0:
            total_area = group["lot_area"].sum()
            zone_area = group.groupby("zone_cat")["lot_area"].sum()
        else:
            total_area = lot_count
            zone_area = group["zone_cat"].value_counts()

        dominant = zone_area.idxmax()
        dominant_ratio = zone_area[dominant] / total_area

        if dominant in ("Infrastructure", "Unknown"):
            remaining = zone_area.drop(["Infrastructure", "Unknown"], errors="ignore")
            if len(remaining) > 0:
                dominant = remaining.idxmax()
                dominant_ratio = remaining[dominant] / total_area
            else:
                dominant = "Mixed-Use"
                dominant_ratio = 0

        zone_type = dominant if dominant_ratio >= PLURALITY_THRESHOLD else "Mixed-Use"

        cell_records.append({
            "cell_id": cell_id,
            "cell_lat": round(cell_lat, 7),
            "cell_lon": round(cell_lon, 7),
            "zone_type": zone_type,
            "cell_lot_count": lot_count,
        })

    print(f"    Property mode: {len(cell_records)} cells from {len(df_prop)} parcels")
    return pd.DataFrame(cell_records)


def _grid_from_hybrid(city_key, city):
    """Generate grid using property data for landuse + OSM boundary."""
    df_prop = load_property_data(city_key)

    if df_prop is not None and "latitude" in df_prop.columns and len(df_prop) > 0:
        # Use property data coordinates for grid generation
        return _grid_from_property(city_key, city)
    else:
        # Fall back to full OSM mode
        print(f"    Hybrid mode: property data insufficient, falling back to OSM")
        return _grid_from_osm(city_key, city)


def _grid_from_osm(city_key, city):
    """Generate grid from OSM data (city boundary + landuse polygons)."""
    import osmnx as ox

    city_query = city["city_query"]
    print(f"    Geocoding: {city_query}")
    gdf = ox.geocode_to_gdf(city_query)
    boundary = gdf.geometry.iloc[0]

    cell_size = GRID_CELL_SIZE_M
    bounds = boundary.bounds  # (minx, miny, maxx, maxy)
    lon_min_b, lat_min_b, lon_max_b, lat_max_b = bounds

    ref_lat = (lat_min_b + lat_max_b) / 2
    lat_step = cell_size / 111_000
    lon_step = cell_size / (111_000 * math.cos(math.radians(ref_lat)))
    buffer = lat_step

    lat_min = lat_min_b - buffer
    lat_max = lat_max_b + buffer
    lon_min = lon_min_b - buffer
    lon_max = lon_max_b + buffer

    n_rows = int(math.ceil((lat_max - lat_min) / lat_step))
    n_cols = int(math.ceil((lon_max - lon_min) / lon_step))
    print(f"    Grid: {n_rows} x {n_cols} = {n_rows * n_cols:,} candidates")

    # Generate cells inside boundary
    candidates = []
    for r in range(n_rows):
        cell_lat = lat_min + (r + 0.5) * lat_step
        for c in range(n_cols):
            cell_lon = lon_min + (c + 0.5) * lon_step
            if boundary.contains(Point(cell_lon, cell_lat)):
                candidates.append({
                    "grid_row": r, "grid_col": c,
                    "cell_lat": round(cell_lat, 7),
                    "cell_lon": round(cell_lon, 7),
                    "cell_id": f"r{r:04d}_c{c:04d}",
                })
    print(f"    Inside boundary: {len(candidates):,} cells")

    # Query landuse polygons
    bbox = f"{lat_min_b},{lon_min_b},{lat_max_b},{lon_max_b}"
    query_lu = (f'[out:json][timeout:180];\n'
                f'(\n'
                f'  way["landuse"]({bbox});\n'
                f'  relation["landuse"]({bbox});\n'
                f');\n'
                f'out body geom;')

    print("    Querying OSM landuse polygons...")
    data_lu = query_overpass_cached(query_lu)

    # Build reverse map
    osm_val_to_zone = {}
    for zone_name, rule in OSM_ZONE_RULES.items():
        if "osm_landuse" in rule:
            for val in rule["osm_landuse"]:
                osm_val_to_zone[val] = zone_name

    landuse_records = []
    for el in data_lu.get("elements", []):
        tags = el.get("tags", {})
        lu_val = tags.get("landuse", "")
        zone_cat = osm_val_to_zone.get(lu_val, "Other")
        geom = el.get("geometry", [])
        if geom and len(geom) >= 3:
            try:
                coords = [(p["lon"], p["lat"]) for p in geom]
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    landuse_records.append({"polygon": poly, "zone_cat": zone_cat})
            except Exception:
                pass
    print(f"    Parsed {len(landuse_records)} landuse polygons")

    # Query buildings for fallback (may OOM for very large cities)
    query_bldg = (f'[out:json][timeout:120];\n'
                  f'(\n'
                  f'  way["building"]({bbox});\n'
                  f');\n'
                  f'out center tags;')
    print("    Querying OSM buildings for fallback...")
    try:
        data_bldg = query_overpass_cached(query_bldg)
    except RuntimeError as e:
        print(f"    WARNING: Building query failed ({e}), proceeding with landuse only")
        data_bldg = {"elements": []}

    bldg_val_to_zone = {}
    for zone_key, vals in OSM_BUILDING_CATEGORIES.items():
        for v in vals:
            bldg_val_to_zone[v] = zone_key.capitalize()

    bldg_records = []
    for el in data_bldg.get("elements", []):
        tags = el.get("tags", {})
        bldg_val = tags.get("building", "yes")
        lat = (el.get("center", {}) or {}).get("lat")
        lon = (el.get("center", {}) or {}).get("lon")
        if lat and lon:
            bldg_records.append({
                "lat": float(lat), "lon": float(lon),
                "zone": bldg_val_to_zone.get(bldg_val),
            })
    print(f"    Found {len(bldg_records)} buildings")

    # Build spatial index for landuse polygons
    half_lat = lat_step / 2
    half_lon = lon_step / 2
    cell_records = []

    lu_polygons = [lr["polygon"] for lr in landuse_records]
    lu_tree = STRtree(lu_polygons) if lu_polygons else None

    # Pre-index buildings into grid cells for O(1) lookup
    bldg_by_cell = {}
    for b in bldg_records:
        br = int((b["lat"] - lat_min) / lat_step)
        bc = int((b["lon"] - lon_min) / lon_step)
        bkey = (br, bc)
        if bkey not in bldg_by_cell:
            bldg_by_cell[bkey] = []
        bldg_by_cell[bkey].append(b)

    for cell in candidates:
        cell_box = box(cell["cell_lon"] - half_lon, cell["cell_lat"] - half_lat,
                       cell["cell_lon"] + half_lon, cell["cell_lat"] + half_lat)

        # Method 1: landuse polygon intersection (spatial index accelerated)
        zone_areas = {}
        total_inter = 0
        if lu_tree is not None:
            nearby_idx = lu_tree.query(cell_box)
            for idx in nearby_idx:
                lr = landuse_records[idx]
                try:
                    inter = cell_box.intersection(lr["polygon"])
                    if inter.area > 0:
                        zone_areas[lr["zone_cat"]] = zone_areas.get(lr["zone_cat"], 0) + inter.area
                        total_inter += inter.area
                except Exception:
                    pass

        zone_type = None
        feature_count = 0

        if total_inter > 0:
            dominant = max(zone_areas, key=zone_areas.get)
            dominant_ratio = zone_areas[dominant] / total_inter
            if dominant == "Other":
                remaining = {k: v for k, v in zone_areas.items() if k != "Other"}
                if remaining:
                    dominant = max(remaining, key=remaining.get)
                    dominant_ratio = remaining[dominant] / total_inter
            zone_type = dominant if dominant_ratio >= PLURALITY_THRESHOLD else "Mixed-Use"
            feature_count = int(total_inter / (cell_box.area + 1e-12) * 10) + 1

        # Method 2: building tag fallback (pre-indexed grid lookup)
        if zone_type is None:
            bkey = (cell["grid_row"], cell["grid_col"])
            bldg_in_cell = bldg_by_cell.get(bkey, [])
            feature_count = len(bldg_in_cell)
            if feature_count >= MIN_FEATURES_PER_CELL:
                zone_counts = {}
                for b in bldg_in_cell:
                    if b["zone"]:
                        zone_counts[b["zone"]] = zone_counts.get(b["zone"], 0) + 1
                if zone_counts:
                    dominant = max(zone_counts, key=zone_counts.get)
                    dominant_ratio = zone_counts[dominant] / sum(zone_counts.values())
                    zone_type = dominant if dominant_ratio >= PLURALITY_THRESHOLD else "Mixed-Use"
                else:
                    zone_type = "Mixed-Use"

        if zone_type and feature_count >= MIN_FEATURES_PER_CELL:
            cell_records.append({
                "cell_id": cell["cell_id"],
                "cell_lat": cell["cell_lat"],
                "cell_lon": cell["cell_lon"],
                "zone_type": zone_type,
                "cell_lot_count": feature_count,
            })

    coverage = len(cell_records) / len(candidates) * 100 if candidates else 0
    print(f"    OSM mode: {len(cell_records)} cells ({coverage:.1f}% coverage)")
    return pd.DataFrame(cell_records)
