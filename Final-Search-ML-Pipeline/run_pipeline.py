#!/usr/bin/env python
"""
Urban ML Pipeline — Feature extraction for multiple US cities.

Usage:
    python run_pipeline.py                  # Run all cities in CITY_REGISTRY
    python run_pipeline.py NYC Chicago DC   # Run specific cities

Output:
    csv/{city}/combined_grid.csv per city
    csv/all_cities_combined.csv  (all cities merged, ready for ML)

After running, open ml_analysis.ipynb for ML models + visualization.
"""
import os
import sys
import time

import pandas as pd

from config import CITY_REGISTRY
from grid import generate_grid
from features import extract_all_features


def run_city(city_key):
    """Run full feature extraction pipeline for one city."""
    if city_key not in CITY_REGISTRY:
        print(f"ERROR: Unknown city '{city_key}'. Available: {list(CITY_REGISTRY.keys())}")
        return None

    city = CITY_REGISTRY[city_key]
    csv_dir = f"csv/{city_key}"
    os.makedirs(csv_dir, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  CITY: {city_key}")
    print(f"  Query: {city['city_query']}")
    print(f"  Mode: {city['data_source']}")
    print(f"  CSV: {csv_dir}/")
    print(f"{'#' * 60}")

    t0 = time.time()

    try:
        # Step 1: Grid generation + zone_type
        df_grid = generate_grid(city_key)

        if len(df_grid) == 0:
            print(f"\n  WARNING: No grid cells generated for {city_key}")
            return None

        # Step 2: Feature extraction (all 5 steps) + merge
        df_combined = extract_all_features(df_grid, city_key, csv_dir)

        elapsed = time.time() - t0
        print(f"\n  {city_key} DONE — {len(df_combined)} cells in {elapsed:.1f}s")
        return df_combined

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  {city_key} FAILED ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_all_cities(city_results):
    """Merge all per-city combined CSVs into a single all_cities_combined.csv."""
    all_dfs = []
    for city_key, df in city_results.items():
        if df is not None:
            df_copy = df.copy()
            df_copy.insert(0, "city", city_key)
            all_dfs.append(df_copy)

    if not all_dfs:
        print("\nNo cities completed successfully — skipping merge.")
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    output_path = "csv/all_cities_combined.csv"
    os.makedirs("csv", exist_ok=True)
    df_all.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  ALL CITIES COMBINED")
    print(f"{'=' * 60}")
    print(f"  Total: {len(df_all)} cells from {len(all_dfs)} cities")
    for city_key in city_results:
        if city_results[city_key] is not None:
            n = len(city_results[city_key])
            print(f"    {city_key:<20s} {n:>6d} cells")
    print(f"\n  Saved: {output_path}")
    print(f"  Columns: {list(df_all.columns)}")
    return df_all


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITY_REGISTRY.keys())

    print(f"Urban ML Pipeline")
    print(f"Cities to process: {cities}")
    print(f"Available cities: {list(CITY_REGISTRY.keys())}")

    t_start = time.time()
    results = {}

    for city_key in cities:
        results[city_key] = run_city(city_key)

    # Merge all cities into single CSV
    merge_all_cities(results)

    t_total = time.time() - t_start
    n_ok = sum(1 for v in results.values() if v is not None)
    n_fail = sum(1 for v in results.values() if v is None)

    print(f"\n{'#' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  {n_ok} succeeded, {n_fail} failed, {t_total:.0f}s total")
    print(f"{'#' * 60}")
    print(f"\n  Next: open ml_analysis.ipynb to run ML models + visualization")


if __name__ == "__main__":
    main()
