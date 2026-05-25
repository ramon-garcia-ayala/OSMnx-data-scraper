"""
Feature extraction: amenity composition, building characteristics,
land use mix, tourism intensity, commercial density.
"""
import math
import os
import re

import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import Polygon, box

from config import (CITY_REGISTRY, GRID_CELL_SIZE_M, OSM_LANDUSE_GROUPS,
                    load_property_data)
from overpass import balltree_assign, compute_bbox, query_overpass_cached


CELL_AREA_KM2 = (GRID_CELL_SIZE_M / 1000) ** 2
QUERY_RADIUS = 500  # meters


def shannon_entropy(proportions):
    """Compute Shannon entropy from an array of proportions."""
    proportions = proportions[proportions > 0]
    if len(proportions) == 0:
        return 0.0
    return float(-np.sum(proportions * np.log2(proportions)))


# ═══════════════════════════════════════════════════════
#  02 — Amenity Composition (always OSM)
# ═══════════════════════════════════════════════════════

QUERY_TAGS = ["amenity", "shop", "office", "craft", "tourism", "leisure"]
FOOD_DRINK_VALUES = {
    "restaurant", "fast_food", "cafe", "bar", "pub", "food_court", "bakery",
    "ice_cream", "deli", "confectionery", "coffee", "tea", "beverages",
    "butcher", "greengrocer", "pastry", "brewery", "grocery", "seafood",
}


def extract_amenity_composition(df_grid, city_config, csv_dir):
    """Extract amenity_density and amenity_ratio_food_drink per cell."""
    print("  02 · Amenity Composition")
    bbox = compute_bbox(df_grid)

    tag_filters = "\n ".join(
        [f'node["{tag}"]({bbox});' for tag in QUERY_TAGS]
        + [f'way["{tag}"]({bbox});' for tag in QUERY_TAGS]
    )
    query = f"[out:json][timeout:120];\n(\n {tag_filters}\n);\nout center tags;"
    data = query_overpass_cached(query)

    poi_records = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        lat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        lon = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        if not (lat and lon):
            continue
        is_food = False
        has_tag = False
        for tag_key in QUERY_TAGS:
            if tag_key in tags:
                has_tag = True
                if tags[tag_key] in FOOD_DRINK_VALUES:
                    is_food = True
                break
        if has_tag:
            poi_records.append({"lat": float(lat), "lon": float(lon), "is_food": is_food})

    print(f"    Found {len(poi_records)} POIs ({sum(p['is_food'] for p in poi_records)} food/drink)")

    assignments = balltree_assign(poi_records, df_grid, max_dist_m=QUERY_RADIUS)
    cell_ids = df_grid["cell_id"].tolist()

    records = []
    for cid in cell_ids:
        indices = assignments[cid]
        total = len(indices)
        food_count = sum(1 for j in indices if poi_records[j]["is_food"])
        records.append({
            "cell_id": cid,
            "amenity_density": round(total / CELL_AREA_KM2, 2),
            "amenity_ratio_food_drink": round(food_count / total, 4) if total > 0 else 0.0,
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{csv_dir}/02_amenity_composition.csv", index=False, encoding="utf-8")
    print(f"    OK ({len(df)} cells, mean density: {df['amenity_density'].mean():.1f})")
    return df


# ═══════════════════════════════════════════════════════
#  03 — Building Characteristics (property or OSM)
# ═══════════════════════════════════════════════════════

def extract_building_characteristics(df_grid, city_config, csv_dir):
    """Extract avg_floors, avg_yearbuilt, building_count, total_bldg_area per cell."""
    print("  03 · Building Characteristics")
    city_key = city_config["_city_key"]
    city = CITY_REGISTRY[city_key]
    data_source = city["data_source"]

    # Try property data first (for "property" and "hybrid" modes)
    if data_source in ("property", "hybrid"):
        df_prop = load_property_data(city_key)
        col_map = city["column_map"]
        has_building_cols = (
            df_prop is not None and
            col_map.get("num_floors") is not None and
            "num_floors" in df_prop.columns
        )
        if has_building_cols:
            return _building_from_property(df_grid, df_prop, csv_dir)

    # Fall back to OSM
    return _building_from_osm(df_grid, csv_dir)


def _building_from_property(df_grid, df_prop, csv_dir):
    """Aggregate building features from property data."""
    cell_size = GRID_CELL_SIZE_M
    ref_lat = df_prop["latitude"].mean()
    lat_step = cell_size / 111_000
    lon_step = cell_size / (111_000 * math.cos(math.radians(ref_lat)))
    lat_min = df_prop["latitude"].min() - lat_step
    lon_min = df_prop["longitude"].min() - lon_step

    df_prop = df_prop.copy()
    df_prop["grid_row"] = ((df_prop["latitude"] - lat_min) / lat_step).astype(int)
    df_prop["grid_col"] = ((df_prop["longitude"] - lon_min) / lon_step).astype(int)
    df_prop["cell_id"] = ("r" + df_prop["grid_row"].astype(str).str.zfill(4) +
                          "_c" + df_prop["grid_col"].astype(str).str.zfill(4))

    valid_cells = set(df_grid["cell_id"])
    df_prop = df_prop[df_prop["cell_id"].isin(valid_cells)]

    agg_dict = {}
    if "num_floors" in df_prop.columns:
        agg_dict["avg_floors"] = ("num_floors", "mean")
    if "year_built" in df_prop.columns:
        agg_dict["avg_yearbuilt"] = ("year_built", "mean")
    if "building_area" in df_prop.columns:
        agg_dict["total_bldg_area"] = ("building_area", "sum")
    if "num_buildings" in df_prop.columns:
        agg_dict["building_count"] = ("num_buildings", "sum")
    else:
        agg_dict["building_count"] = ("cell_id", "count")

    agg = df_prop.groupby("cell_id").agg(**agg_dict).reset_index()

    if "avg_floors" in agg.columns:
        agg["avg_floors"] = agg["avg_floors"].round(1)
    if "avg_yearbuilt" in agg.columns:
        agg["avg_yearbuilt"] = agg["avg_yearbuilt"].round(0).astype("Int64")
    if "total_bldg_area" in agg.columns:
        agg["total_bldg_area"] = agg["total_bldg_area"].round(0)

    df_result = df_grid[["cell_id"]].merge(agg, on="cell_id", how="left")

    # Ensure all expected columns exist
    for col in ["avg_floors", "avg_yearbuilt", "building_count", "total_bldg_area"]:
        if col not in df_result.columns:
            df_result[col] = np.nan

    df_result.to_csv(f"{csv_dir}/03_building_characteristics.csv", index=False, encoding="utf-8")
    print(f"    Property mode: {len(df_result)} cells")
    return df_result


def _building_from_osm(df_grid, csv_dir):
    """Extract building features from OSM building tags and geometry."""
    bbox = compute_bbox(df_grid)
    query = (f'[out:json][timeout:180];\n'
             f'(\n'
             f'  way["building"]({bbox});\n'
             f');\n'
             f'out body geom tags;')

    try:
        data = query_overpass_cached(query)
    except RuntimeError as e:
        print(f"    WARNING: Building query failed ({e}), returning NaN features")
        data = {"elements": []}

    ref_lat = df_grid["cell_lat"].mean()
    cell_size = GRID_CELL_SIZE_M
    lat_step = cell_size / 111_000
    lon_step = cell_size / (111_000 * math.cos(math.radians(ref_lat)))
    lat_min = df_grid["cell_lat"].min() - lat_step
    lon_min = df_grid["cell_lon"].min() - lon_step

    valid_cells = set(df_grid["cell_id"])
    cos_ref = math.cos(math.radians(ref_lat))
    SQ_FT_PER_SQ_M = 10.7639

    bldg_records = []
    n_levels = 0
    n_date = 0

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        geom = el.get("geometry", [])
        if not geom or len(geom) < 3:
            continue

        lats = [p["lat"] for p in geom]
        lons = [p["lon"] for p in geom]
        clat = sum(lats) / len(lats)
        clon = sum(lons) / len(lons)

        grid_row = int((clat - lat_min) / lat_step)
        grid_col = int((clon - lon_min) / lon_step)
        cell_id = f"r{grid_row:04d}_c{grid_col:04d}"
        if cell_id not in valid_cells:
            continue

        # Parse building:levels
        levels = None
        levels_str = tags.get("building:levels", "")
        if levels_str:
            try:
                levels = float(levels_str)
                if levels > 0:
                    n_levels += 1
                else:
                    levels = None
            except (ValueError, TypeError):
                pass

        # Parse start_date -> year
        year_built = None
        date_str = tags.get("start_date", "")
        if date_str:
            m = re.search(r"(1[6-9]\d{2}|20[0-2]\d)", date_str)
            if m:
                year_built = int(m.group(1))
                n_date += 1

        # Footprint area
        area_sqft = 0
        try:
            coords = [(p["lon"], p["lat"]) for p in geom]
            poly = Polygon(coords)
            if poly.is_valid and poly.area > 0:
                area_m2 = poly.area * (111_000 * cos_ref) * 111_000
                area_sqft = area_m2 * SQ_FT_PER_SQ_M
        except Exception:
            pass

        bldg_records.append({
            "cell_id": cell_id,
            "levels": levels,
            "year_built": year_built,
            "area_sqft": area_sqft,
        })

    n_total = len(bldg_records)
    print(f"    OSM: {n_total} buildings "
          f"(levels: {100*n_levels/max(n_total,1):.0f}%, "
          f"year: {100*n_date/max(n_total,1):.0f}%)")

    if n_total > 0:
        df_bldg = pd.DataFrame(bldg_records)
        agg = df_bldg.groupby("cell_id").agg(
            avg_floors=("levels", "mean"),
            avg_yearbuilt=("year_built", "mean"),
            total_bldg_area=("area_sqft", "sum"),
            building_count=("cell_id", "count"),
        ).reset_index()
        agg["avg_floors"] = agg["avg_floors"].round(1)
        agg["avg_yearbuilt"] = agg["avg_yearbuilt"].round(0).astype("Int64")
        agg["total_bldg_area"] = agg["total_bldg_area"].round(0)
    else:
        agg = pd.DataFrame(columns=["cell_id", "avg_floors", "avg_yearbuilt",
                                     "total_bldg_area", "building_count"])

    df_result = df_grid[["cell_id"]].merge(agg, on="cell_id", how="left")
    df_result.to_csv(f"{csv_dir}/03_building_characteristics.csv", index=False, encoding="utf-8")
    print(f"    OK ({len(df_result)} cells)")
    return df_result


# ═══════════════════════════════════════════════════════
#  04 — Land Use Mix (property or OSM)
# ═══════════════════════════════════════════════════════

def extract_land_use_mix(df_grid, city_config, csv_dir):
    """Extract landuse_entropy per cell."""
    print("  04 · Land Use Mix")
    city_key = city_config["_city_key"]
    city = CITY_REGISTRY[city_key]
    data_source = city["data_source"]

    if data_source in ("property", "hybrid"):
        df_prop = load_property_data(city_key)
        if df_prop is not None and "landuse" in df_prop.columns:
            return _landuse_from_property(df_grid, df_prop, csv_dir)

    return _landuse_from_osm(df_grid, csv_dir)


def _landuse_from_property(df_grid, df_prop, csv_dir):
    """Shannon entropy from property landuse codes."""
    cell_size = GRID_CELL_SIZE_M
    ref_lat = df_prop["latitude"].mean()
    lat_step = cell_size / 111_000
    lon_step = cell_size / (111_000 * math.cos(math.radians(ref_lat)))
    lat_min = df_prop["latitude"].min() - lat_step
    lon_min = df_prop["longitude"].min() - lon_step

    df_prop = df_prop.copy()
    df_prop["grid_row"] = ((df_prop["latitude"] - lat_min) / lat_step).astype(int)
    df_prop["grid_col"] = ((df_prop["longitude"] - lon_min) / lon_step).astype(int)
    df_prop["cell_id"] = ("r" + df_prop["grid_row"].astype(str).str.zfill(4) +
                          "_c" + df_prop["grid_col"].astype(str).str.zfill(4))

    valid_cells = set(df_grid["cell_id"])
    df_prop = df_prop[df_prop["cell_id"].isin(valid_cells)]

    records = []
    for cell_id, group in df_prop.groupby("cell_id"):
        if "lot_area" in group.columns and group["lot_area"].sum() > 0:
            total_area = group["lot_area"].sum()
            lu_area = group.groupby("landuse")["lot_area"].sum()
            proportions = (lu_area / total_area).values
        else:
            counts = group["landuse"].value_counts()
            proportions = (counts / counts.sum()).values
        records.append({"cell_id": cell_id, "landuse_entropy": round(shannon_entropy(proportions), 4)})

    df_mix = pd.DataFrame(records)
    df_result = df_grid[["cell_id"]].merge(df_mix, on="cell_id", how="left")
    df_result["landuse_entropy"] = df_result["landuse_entropy"].fillna(0.0)
    df_result.to_csv(f"{csv_dir}/04_land_use_mix.csv", index=False, encoding="utf-8")
    print(f"    Property mode: {len(df_result)} cells (mean entropy: {df_result['landuse_entropy'].mean():.3f})")
    return df_result


def _landuse_from_osm(df_grid, csv_dir):
    """Shannon entropy from OSM landuse polygon intersections."""
    bbox = compute_bbox(df_grid)
    query_lu = (f'[out:json][timeout:180];\n'
                f'(\n'
                f'  way["landuse"]({bbox});\n'
                f'  relation["landuse"]({bbox});\n'
                f');\n'
                f'out body geom;')

    data_lu = query_overpass_cached(query_lu)

    landuse_polys = []
    for el in data_lu.get("elements", []):
        tags = el.get("tags", {})
        lu_val = tags.get("landuse", "")
        category = OSM_LANDUSE_GROUPS.get(lu_val, "other")
        geom = el.get("geometry", [])
        if geom and len(geom) >= 3:
            try:
                coords = [(p["lon"], p["lat"]) for p in geom]
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    landuse_polys.append({"polygon": poly, "category": category})
            except Exception:
                pass

    print(f"    OSM: {len(landuse_polys)} landuse polygons")

    ref_lat = df_grid["cell_lat"].mean()
    lat_step = GRID_CELL_SIZE_M / 111_000
    lon_step = GRID_CELL_SIZE_M / (111_000 * math.cos(math.radians(ref_lat)))

    # Build spatial index for landuse polygons
    lu_geoms = [lp["polygon"] for lp in landuse_polys]
    lu_tree = STRtree(lu_geoms) if lu_geoms else None

    half_lat = lat_step / 2
    half_lon = lon_step / 2
    records = []
    for _, row in df_grid.iterrows():
        cell_box = box(row["cell_lon"] - half_lon, row["cell_lat"] - half_lat,
                       row["cell_lon"] + half_lon, row["cell_lat"] + half_lat)

        cat_areas = {}
        if lu_tree is not None:
            nearby_idx = lu_tree.query(cell_box)
            for idx in nearby_idx:
                lp = landuse_polys[idx]
                try:
                    inter = cell_box.intersection(lp["polygon"])
                    if inter.area > 0:
                        cat_areas[lp["category"]] = cat_areas.get(lp["category"], 0) + inter.area
                except Exception:
                    pass

        total = sum(cat_areas.values())
        if total > 0:
            proportions = np.array(list(cat_areas.values())) / total
            entropy = round(shannon_entropy(proportions), 4)
        else:
            entropy = 0.0

        records.append({"cell_id": row["cell_id"], "landuse_entropy": entropy})

    df_result = pd.DataFrame(records)
    df_result.to_csv(f"{csv_dir}/04_land_use_mix.csv", index=False, encoding="utf-8")
    print(f"    OK ({len(df_result)} cells, mean entropy: {df_result['landuse_entropy'].mean():.3f})")
    return df_result


# ═══════════════════════════════════════════════════════
#  05 — Tourism Intensity (always OSM)
# ═══════════════════════════════════════════════════════

TOURISM_VALUES = {"hotel", "hostel", "motel", "guest_house", "museum",
                  "attraction", "viewpoint", "gallery", "artwork",
                  "information", "theme_park", "zoo", "aquarium"}


def extract_tourism_intensity(df_grid, city_config, csv_dir):
    """Extract tourism_density per cell from OSM tourism POIs."""
    print("  05 · Tourism Intensity")
    bbox = compute_bbox(df_grid)

    query = (f'[out:json][timeout:90];\n'
             f'(node["tourism"]({bbox});\n'
             f' way["tourism"]({bbox}););\n'
             f'out center tags;')

    data = query_overpass_cached(query)

    poi_records = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        tourism_val = tags.get("tourism", "")
        lat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        lon = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        if lat and lon and tourism_val in TOURISM_VALUES:
            poi_records.append({"lat": float(lat), "lon": float(lon)})

    print(f"    Found {len(poi_records)} tourism POIs")

    assignments = balltree_assign(poi_records, df_grid, max_dist_m=QUERY_RADIUS)

    records = []
    for cid in df_grid["cell_id"]:
        tc = len(assignments[cid])
        records.append({"cell_id": cid, "tourism_density": round(tc / CELL_AREA_KM2, 2)})

    df = pd.DataFrame(records)
    df.to_csv(f"{csv_dir}/05_tourism_intensity.csv", index=False, encoding="utf-8")
    print(f"    OK ({(df['tourism_density'] > 0).sum()} cells with tourism)")
    return df


# ═══════════════════════════════════════════════════════
#  06 — Commercial Density (always OSM)
# ═══════════════════════════════════════════════════════

def extract_commercial_density(df_grid, city_config, csv_dir):
    """Extract shop_density_km2 per cell."""
    print("  06 · Commercial Density")
    bbox = compute_bbox(df_grid)

    query = (f'[out:json][timeout:90];\n'
             f'(node["shop"]({bbox});\n'
             f' way["shop"]({bbox}););\n'
             f'out center tags;')

    data = query_overpass_cached(query)

    poi_records = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        shop_val = tags.get("shop", "")
        if shop_val and shop_val not in {"vacant", "yes"}:
            lat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
            lon = el.get("lon") or (el.get("center", {}) or {}).get("lon")
            if lat and lon:
                poi_records.append({"lat": float(lat), "lon": float(lon)})

    print(f"    Found {len(poi_records)} shops")

    assignments = balltree_assign(poi_records, df_grid, max_dist_m=QUERY_RADIUS)

    records = []
    for cid in df_grid["cell_id"]:
        indices = assignments[cid]
        total = len(indices)
        records.append({
            "cell_id": cid,
            "shop_density_km2": round(total / CELL_AREA_KM2, 2),
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{csv_dir}/06_commercial_density.csv", index=False, encoding="utf-8")
    print(f"    OK (mean shops/km²: {df['shop_density_km2'].mean():.1f})")
    return df


# ═══════════════════════════════════════════════════════
#  07 — Office Density (always OSM)
# ═══════════════════════════════════════════════════════

def extract_office_density(df_grid, city_config, csv_dir):
    """Extract office_density per cell from OSM office POIs."""
    print("  07 · Office Density")
    bbox = compute_bbox(df_grid)

    query = (f'[out:json][timeout:90];\n'
             f'(node["office"]({bbox});\n'
             f' way["office"]({bbox}););\n'
             f'out center tags;')

    data = query_overpass_cached(query)

    poi_records = []
    for el in data.get("elements", []):
        lat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        lon = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        if lat and lon:
            poi_records.append({"lat": float(lat), "lon": float(lon)})

    print(f"    Found {len(poi_records)} offices")

    assignments = balltree_assign(poi_records, df_grid, max_dist_m=QUERY_RADIUS)

    records = []
    for cid in df_grid["cell_id"]:
        total = len(assignments[cid])
        records.append({
            "cell_id": cid,
            "office_density": round(total / CELL_AREA_KM2, 2),
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{csv_dir}/07_office_density.csv", index=False, encoding="utf-8")
    print(f"    OK (mean offices/km²: {df['office_density'].mean():.1f})")
    return df


# ═══════════════════════════════════════════════════════
#  08 — Road Density (always OSM)
# ═══════════════════════════════════════════════════════

def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance between two points in metres."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_road_density(df_grid, city_config, csv_dir):
    """Extract road_density_primary (km of major roads per km²) per cell."""
    print("  08 · Road Density")
    bbox = compute_bbox(df_grid)

    query = (f'[out:json][timeout:180];\n'
             f'way["highway"~"^(primary|secondary|tertiary|'
             f'primary_link|secondary_link|tertiary_link)$"]({bbox});\n'
             f'out geom;')

    data = query_overpass_cached(query)

    ref_lat = df_grid["cell_lat"].mean()
    cell_size = GRID_CELL_SIZE_M
    lat_step = cell_size / 111_000
    lon_step = cell_size / (111_000 * math.cos(math.radians(ref_lat)))
    lat_min = df_grid["cell_lat"].min() - lat_step
    lon_min = df_grid["cell_lon"].min() - lon_step
    valid_cells = set(df_grid["cell_id"])

    cell_road_km = {}
    n_ways = 0
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue

        # Compute way length
        length_m = 0.0
        for i in range(len(geom) - 1):
            length_m += _haversine_m(
                geom[i]["lat"], geom[i]["lon"],
                geom[i + 1]["lat"], geom[i + 1]["lon"],
            )

        # Centroid for cell assignment
        clat = sum(p["lat"] for p in geom) / len(geom)
        clon = sum(p["lon"] for p in geom) / len(geom)
        grid_row = int((clat - lat_min) / lat_step)
        grid_col = int((clon - lon_min) / lon_step)
        cell_id = f"r{grid_row:04d}_c{grid_col:04d}"
        if cell_id not in valid_cells:
            continue

        cell_road_km[cell_id] = cell_road_km.get(cell_id, 0.0) + length_m / 1000.0
        n_ways += 1

    print(f"    Found {n_ways} road ways")

    records = []
    for cid in df_grid["cell_id"]:
        records.append({
            "cell_id": cid,
            "road_density_primary": round(cell_road_km.get(cid, 0.0) / CELL_AREA_KM2, 2),
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{csv_dir}/08_road_density.csv", index=False, encoding="utf-8")
    print(f"    OK (mean road km/km²: {df['road_density_primary'].mean():.1f})")
    return df


# ═══════════════════════════════════════════════════════
#  09 — Transit Stop Density (always OSM)
# ═══════════════════════════════════════════════════════

def extract_transit_density(df_grid, city_config, csv_dir):
    """Extract transit_stop_density per cell from OSM transit nodes."""
    print("  09 · Transit Stop Density")
    bbox = compute_bbox(df_grid)

    query = (f'[out:json][timeout:120];\n'
             f'(\n'
             f'  node["highway"="bus_stop"]({bbox});\n'
             f'  node["railway"~"^(station|halt|subway_entrance|tram_stop)$"]({bbox});\n'
             f'  node["public_transport"="stop_position"]({bbox});\n'
             f');\n'
             f'out center tags;')

    data = query_overpass_cached(query)

    seen = set()
    poi_records = []
    for el in data.get("elements", []):
        lat = el.get("lat")
        lon = el.get("lon")
        if lat and lon:
            key = (round(float(lat), 6), round(float(lon), 6))
            if key not in seen:
                seen.add(key)
                poi_records.append({"lat": float(lat), "lon": float(lon)})

    print(f"    Found {len(poi_records)} unique transit stops")

    assignments = balltree_assign(poi_records, df_grid, max_dist_m=QUERY_RADIUS)

    records = []
    for cid in df_grid["cell_id"]:
        total = len(assignments[cid])
        records.append({
            "cell_id": cid,
            "transit_stop_density": round(total / CELL_AREA_KM2, 2),
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{csv_dir}/09_transit_density.csv", index=False, encoding="utf-8")
    print(f"    OK (mean stops/km²: {df['transit_stop_density'].mean():.1f})")
    return df


# ═══════════════════════════════════════════════════════
#  10 — Intersection Density (always OSM)
# ═══════════════════════════════════════════════════════

def extract_intersection_density(df_grid, city_config, csv_dir):
    """Extract intersection_density per cell (nodes shared by 3+ highway ways)."""
    print("  10 · Intersection Density")
    bbox = compute_bbox(df_grid)

    query = (f'[out:json][timeout:180];\n'
             f'way["highway"~"^(primary|secondary|tertiary|residential|'
             f'living_street|unclassified|service)$"]({bbox});\n'
             f'out body;\n'
             f'>;\n'
             f'out skel;')

    data = query_overpass_cached(query)

    # Pass 1: count how many ways reference each node
    node_way_count = {}
    for el in data.get("elements", []):
        if el.get("type") == "way":
            for nid in el.get("nodes", []):
                node_way_count[nid] = node_way_count.get(nid, 0) + 1

    # Pass 2: get coordinates for intersection nodes (count >= 3)
    intersection_nodes = {nid for nid, cnt in node_way_count.items() if cnt >= 3}

    poi_records = []
    for el in data.get("elements", []):
        if el.get("type") == "node" and el.get("id") in intersection_nodes:
            lat = el.get("lat")
            lon = el.get("lon")
            if lat and lon:
                poi_records.append({"lat": float(lat), "lon": float(lon)})

    print(f"    Found {len(poi_records)} intersections (nodes in 3+ ways)")

    assignments = balltree_assign(poi_records, df_grid, max_dist_m=QUERY_RADIUS)

    records = []
    for cid in df_grid["cell_id"]:
        total = len(assignments[cid])
        records.append({
            "cell_id": cid,
            "intersection_density": round(total / CELL_AREA_KM2, 2),
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{csv_dir}/10_intersection_density.csv", index=False, encoding="utf-8")
    print(f"    OK (mean intersections/km²: {df['intersection_density'].mean():.1f})")
    return df


# ═══════════════════════════════════════════════════════
#  11 — Nightlife Density (always OSM)
# ═══════════════════════════════════════════════════════

NIGHTLIFE_VALUES = {
    "bar", "pub", "nightclub", "cinema", "theatre", "casino",
    "gambling", "stripclub",
}


def extract_nightlife_density(df_grid, city_config, csv_dir):
    """Extract nightlife_density per cell from OSM nightlife/entertainment POIs."""
    print("  11 · Nightlife Density")
    bbox = compute_bbox(df_grid)

    query = (f'[out:json][timeout:90];\n'
             f'(node["amenity"~"^(bar|pub|nightclub|cinema|theatre|casino|gambling|stripclub)$"]({bbox});\n'
             f' way["amenity"~"^(bar|pub|nightclub|cinema|theatre|casino)$"]({bbox}););\n'
             f'out center tags;')

    data = query_overpass_cached(query)

    poi_records = []
    for el in data.get("elements", []):
        lat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        lon = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        if lat and lon:
            poi_records.append({"lat": float(lat), "lon": float(lon)})

    print(f"    Found {len(poi_records)} nightlife POIs")

    assignments = balltree_assign(poi_records, df_grid, max_dist_m=QUERY_RADIUS)

    records = []
    for cid in df_grid["cell_id"]:
        total = len(assignments[cid])
        records.append({
            "cell_id": cid,
            "nightlife_density": round(total / CELL_AREA_KM2, 2),
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{csv_dir}/11_nightlife_density.csv", index=False, encoding="utf-8")
    print(f"    OK (mean nightlife/km²: {df['nightlife_density'].mean():.1f})")
    return df


# ═══════════════════════════════════════════════════════
#  Master: extract all features and merge
# ═══════════════════════════════════════════════════════

def extract_all_features(df_grid, city_key, csv_dir):
    """
    Run all 9 feature extractors, merge on cell_id, save combined_grid.csv.
    Returns the merged DataFrame.
    """
    city_config = {"_city_key": city_key}

    df_amenity = extract_amenity_composition(df_grid, city_config, csv_dir)
    df_building = extract_building_characteristics(df_grid, city_config, csv_dir)
    df_landuse = extract_land_use_mix(df_grid, city_config, csv_dir)
    df_commercial = extract_commercial_density(df_grid, city_config, csv_dir)
    df_office = extract_office_density(df_grid, city_config, csv_dir)
    df_road = extract_road_density(df_grid, city_config, csv_dir)
    df_transit = extract_transit_density(df_grid, city_config, csv_dir)
    df_intersection = extract_intersection_density(df_grid, city_config, csv_dir)
    df_nightlife = extract_nightlife_density(df_grid, city_config, csv_dir)

    # Merge all on cell_id
    df_combined = df_grid.copy()
    for df_feat in [df_amenity, df_building, df_landuse, df_commercial,
                    df_office, df_road, df_transit, df_intersection, df_nightlife]:
        merge_cols = [c for c in df_feat.columns if c != "cell_id"]
        df_combined = df_combined.merge(
            df_feat[["cell_id"] + merge_cols], on="cell_id", how="left"
        )

    combined_path = f"{csv_dir}/combined_grid.csv"
    df_combined.to_csv(combined_path, index=False, encoding="utf-8")
    print(f"\n  Combined: {df_combined.shape[0]} cells x {df_combined.shape[1]} cols")
    print(f"  Saved: {combined_path}")

    return df_combined
