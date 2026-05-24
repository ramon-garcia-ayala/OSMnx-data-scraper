#!/usr/bin/env python
"""Generate a pairplot from the existing all_cities_combined.csv.

Usage:
    python generate_pairplot.py          # uses csv/all_cities_combined.csv
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config import FEATURE_COLS, CLASS_MAP, INCLUDE_OTHER

CSV_PATH = "csv/all_cities_combined.csv"
OUTPUT_PATH = "outputs/22_pairplot.png"
SAMPLE_N = 3000  # subsample for speed — 60K would take ages

print(f"Reading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} rows, {len(df.columns)} columns")

# ── Apply class mapping (same logic as ml_analysis.ipynb) ─────────
df["label"] = df["zone_type"].map(CLASS_MAP)
if not INCLUDE_OTHER:
    df = df.dropna(subset=["label"])
print(f"  {len(df)} rows after class mapping")

# ── Select top features by ablation importance ────────────────────
# Use the 6 most discriminating features (full 13 would be unreadable)
TOP_FEATURES = [
    "landuse_entropy",
    "intersection_density",
    "amenity_density",
    "shop_density_km2",
    "building_count",
    "transit_stop_density",
]

# ── Fill NaN and subsample ────────────────────────────────────────
for col in TOP_FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna(0)

if len(df) > SAMPLE_N:
    rng = np.random.RandomState(42)
    df = df.sample(n=SAMPLE_N, random_state=rng)
    print(f"  Subsampled to {SAMPLE_N} rows")

# ── Generate pairplot ─────────────────────────────────────────────
print("Generating pairplot (this may take a minute) ...")
sns.set_theme(style="ticks", font_scale=0.8)

g = sns.pairplot(
    df[TOP_FEATURES + ["label"]],
    hue="label",
    palette={"Commercial": "#e74c3c", "Residential": "#3498db"},
    diag_kind="kde",
    plot_kws={"alpha": 0.3, "s": 8, "edgecolor": "none"},
    diag_kws={"linewidth": 1.5},
    corner=True,
)
g.figure.suptitle(
    "Pairplot — Top 6 Features by Ablation Importance (3K sample)",
    y=1.01, fontsize=14, fontweight="bold",
)

os.makedirs("outputs", exist_ok=True)
g.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"  Saved: {OUTPUT_PATH}")
plt.close()
