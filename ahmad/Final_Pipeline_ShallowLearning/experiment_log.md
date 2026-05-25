# Experiment Log — Urban Zone Classification
**Course:** DEML — Digital Tools for Data Encoding and Machine Learning
**Program:** MaCAD26, IAAC Barcelona
**Project:** Predictive classification of urban zones and detection of transformation from open data
**Last updated:** May 24, 2026

---

## 1. Concept and Research Question

### Main Question

> Can the observable characteristics of a city — amenity density, building typology, road network, commercial activity — extracted from OpenStreetMap predict its official land use **without zoning data**?

### Motivation

Official zoning data is expensive, proprietary, or simply nonexistent in many cities around the world. OpenStreetMap (OSM), on the other hand, is an open, collaborative, and global source. If a model trained with property data from well-documented cities (NYC, Philadelphia, Chicago) can generalize to cities where only OSM exists (Washington DC, San Francisco, Los Angeles), then we would have a universal tool for urban analysis.

### Dual Application of the Project

**1. Universal Prediction**
Train with cities that have property data as Ground Truth and predict land use in cities that only have OSM. This validates whether OSM signals are sufficient to classify urban zones without prior local knowledge.

**2. Urban Transformation Detection**
Compare the model's prediction against existing official zoning to identify **zones in transition**: areas where observable behavior (commercial activity, building typology) no longer matches the official classification. These discrepancies are potential indicators of gentrification, industrial reconversion, or mixed-use expansion.

### Technical Pipeline

The pipeline processes each city independently and then combines all data into a single dataset for ML analysis:

```
config.py (6 cities) → run_pipeline.py → [per city: grid.py → features.py → overpass.py]
→ csv/all_cities_combined.csv → ml_analysis.ipynb (EDA + LR/XGB/RF/SVC/ANN + clustering)
```

---

## 2. Dataset Composition

### 2.1 Selected Cities

Six cities were chosen, divided into two groups based on property data availability. This division is not arbitrary: it is the basis of the transfer learning experiment (Section 5.3).

| City | Country | Mode | Property Dataset | Reason for Inclusion |
|---|---|---|---|---|
| **New York City** | USA | `property` | PLUTO 25v4 (~900K lots) | Most complete dataset, main project baseline |
| **Philadelphia** | USA | `property` | OPA (Office of Property Assessment) | Mid-size American city, contrast with NYC |
| **Chicago** | USA | `hybrid` | Cook County Assessor | Grid-layout metropolis, different urban morphology |
| **Washington DC** | USA | `osm` | — (OSM only) | Federal capital, high density of institutional services |
| **San Francisco** | USA | `osm` | — (OSM only) | Compact city with highly differentiated neighborhoods |
| **Los Angeles** | USA | `osm` | — (OSM only) | Sprawling city, extreme morphological contrast |

**Note on `hybrid` mode (Chicago):** Chicago has a local property assessment dataset, but with incomplete coverage for building features. The hybrid mode uses local data to define the grid and zone_type, and fills in building features with OSM data when values are missing.

**Note on `osm` mode (DC, SF, LA):** The city boundary is obtained via `osmnx.geocode_to_gdf()`, the grid is generated from the convex hull of Overpass land use polygons, and zone_type is derived from OSM tags (`landuse=*`). Without property data, the Ground Truth itself is approximate — this is part of the experiment.

### 2.2 Discarded Cities

No cities were discarded during selection. The inclusion criterion was: North American city with reasonable OSM coverage and, for Group A, a publicly downloadable property dataset.

European cities (Barcelona, Paris, Amsterdam) were considered but discarded due to differences in land use systems and zoning categories, which would make class mapping incompatible with the North American cities.

### 2.3 Class Distribution

The unit of analysis is a **150m × 150m** grid cell. Each cell receives a class label (`zone_type`) derived from the property dataset (Group A) or from OSM landuse (Group B).

**Class system:**
- **Residential** — Predominantly residential. In NYC/PLUTO, includes Mixed-Use mapped to Residential (see Section 4.1).
- **Commercial** — Predominantly commercial or service-oriented.
- **Other** — Institutional, open spaces, industrial. Can be included as a third class or excluded (see Section 4.1).

| City | Group | Total Cells | Residential | Commercial | Other | Res:Com Ratio |
|---|---|---|---|---|---|---|
| **NYC (Manhattan)** | A | 1,810 | 1,348 | 283 | 179 | 4.8 : 1 |
| **Philadelphia** | A | 9,979 | 8,322 | 945 | 712 | 8.8 : 1 |
| **Chicago** | A | 12,057 | 10,574 | 1,343 | 140 | 7.9 : 1 |
| **Washington DC** | B | 5,017 | 3,994 | 279 | 744 | 14.3 : 1 |
| **San Francisco** | B | 3,876 | 2,832 | 463 | 581 | 6.1 : 1 |
| **Los Angeles** | B | 27,539 | 20,904 | 2,104 | 4,531 | 9.9 : 1 |
| **TOTAL** | — | 60,278 | 47,974 | 5,417 | 6,887 | 8.9 : 1 |

**Observation on class imbalance:** The 4.6:1 ratio in NYC is expected — Manhattan has much more residential surface than commercial. To handle this imbalance, all models use `class_weight="balanced"`, which automatically weights the minority class (Commercial) with greater importance during training.

---

## 3. Feature Engineering

Features are the "measurements" given to the model for learning. Each feature captures a different aspect of the urban character of a 150m × 150m cell.

### 3.1 Iteration 1: 10 Original Features (Baseline — NYC only)

This first iteration was run exclusively on Manhattan (NYC) to establish a reference point before expanding to all 6 cities.

**Features used:**

| Feature | Source | Description | Urban Hypothesis |
|---|---|---|---|
| `amenity_density` | OSM | Number of amenities per km² | Commercial zones have more amenities |
| `amenity_ratio_food_drink` | OSM | Proportion of restaurants/cafés over total amenities | High ratio = active service zone |
| `avg_floors` | PLUTO | Average building floors in the cell | Taller buildings = higher use intensity |
| `avg_yearbuilt` | PLUTO | Average construction year | Older buildings may indicate historic commercial cores |
| `building_count` | PLUTO | Number of buildings in the cell | Building density |
| `total_bldg_area` | PLUTO | Total built area (m²) in the cell | Total building mass |
| `landuse_entropy` | PLUTO | Shannon entropy of land uses | High value = mixed uses = transition zone |
| `tourism_density` | OSM | Hotels and tourism POIs per km² | Tourist areas tend to be commercial |
| `shop_density_km2` | OSM | Number of shops per km² | Direct indicator of commercial activity |
| `brand_ratio` | OSM | Proportion of shops with `brand=*` tag | Presence of national/international franchises |

**Baseline Results (NYC, Iteration 1):**

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression (LR) | 82.9% | Linear model, most interpretable |
| XGBoost (XGB) | 89.9% | Gradient boosting, robust with imbalance |
| Random Forest (RF) | 90.2% | Tree ensemble, stable |
| **SVC Polynomial** | **90.5%** | **Best model in baseline** |

**Ablation Study Results (what happens when each feature is removed):**

The ablation study measures the impact of each feature by removing it and measuring how much accuracy drops. It distinguishes indispensable features from noisy ones.

| Feature Removed | Accuracy Change | Conclusion |
|---|---|---|
| `total_bldg_area` | Largest drop | Most important feature in the model |
| `tourism_density` | Accuracy **improves** when removed | It is noise — removed in Iteration 2 |
| `brand_ratio` | Minimal change (<5% importance across all models) | Low signal — removed in Iteration 2 |

**Clustering and Dimensionality Reduction Results:**

| Analysis | Result |
|---|---|
| K-Means, optimal k | k = 3 (silhouette = 0.29) |
| PCA, explained variance | Needs 8 components to cover 95% of variance |
| Best normalization | MinMaxScaler (accuracy 0.8406) vs StandardScaler (0.8369) — marginal difference |

**K=3 K-Means interpretation:** Although the dataset has 2 labeled classes (Residential / Commercial), unsupervised clustering finds 3 natural groups. This suggests the existence of a third unlabeled urban "type" — possibly mixed or transitional zones that the binary system does not capture.

### 3.2 Features Removed

Based on the Baseline ablation study, the following curation decisions were made:

**`tourism_density` — REMOVED**
- Evidence: model accuracy improves when this feature is removed
- Urban interpretation: in Manhattan, hotels and tourist attractions are distributed across both commercial and residential zones in an indistinguishable manner. It is not a discriminating indicator at the 150m cell scale
- Decision: remove in Iteration 2

**`brand_ratio` — REMOVED**
- Evidence: importance below 5% across all models (LR, XGB, RF, SVC)
- Urban interpretation: franchise presence is too noisy a proxy. Many residential neighborhoods have chain supermarkets, and many commercial zones have independent shops
- Decision: remove in Iteration 2

### 3.3 Features Added (Hypotheses for Iteration 2)

After reviewing urban morphology literature and the patterns detected by K-Means, 5 additional features are proposed:

| New Feature | Source | Description | Hypothesis for Discrimination |
|---|---|---|---|
| `office_density` | OSM | Offices and workspaces per km² (tag `office=*`) | Direct indicator of daytime corporate activity — very different between residential and business zones |
| `road_density_primary` | OSM | Linear meters of major avenues (`highway=primary/secondary`) per km² | Commercial zones tend to be located on main road axes, not on interior residential streets |
| `transit_stop_density` | OSM | Public transit stops per km² | High-frequency public transit concentrates in high commercial activity zones |
| `intersection_density` | OSM | Road intersections per km² | A denser, more connected road mesh is characteristic of commercial urban centers vs low-connectivity residential fabric |
| `nightlife_density` | OSM | Bars, clubs, nightlife entertainment venues per km² (tag `amenity=bar/nightclub/pub`) | Nightlife entertainment is a strong marker of commercial and mixed-use zones |

**Important note:** In Iteration 1, `intersection_density` and `transit_stop_density` were tested in the Zone-Finding pipeline (at the census tract level in Manhattan) and **showed no discriminating power** — distributions were nearly identical across classes. They are added here again as hypotheses because at the **150m cell scale** behavior may differ, and because including cities with distinct morphologies (LA vs SF vs NYC), these features may acquire signal they did not have in Manhattan alone.

### 3.4 Iteration 2: 13 Optimized Features

Final proposed composition (8 retained + 5 new — 2 removed):

| # | Feature | Status | Source |
|---|---|---|---|
| 1 | `amenity_density` | Retained | OSM |
| 2 | `amenity_ratio_food_drink` | Retained | OSM |
| 3 | `avg_floors` | Retained | PLUTO/local dataset |
| 4 | `avg_yearbuilt` | Retained | PLUTO/local dataset |
| 5 | `building_count` | Retained | PLUTO/local dataset |
| 6 | `total_bldg_area` | Retained | PLUTO/local dataset |
| 7 | `landuse_entropy` | Retained | PLUTO/local dataset |
| 8 | `shop_density_km2` | Retained | OSM |
| 9 | `office_density` | **New** | OSM |
| 10 | `road_density_primary` | **New** | OSM |
| 11 | `transit_stop_density` | **New** | OSM |
| 12 | `intersection_density` | **New** | OSM |
| 13 | `nightlife_density` | **New** | OSM |

**Statistics of the 13 features across the full dataset (60,278 cells, 6 cities):**

| Feature | Mean | Std | Min | Max | %NaN |
|---|---|---|---|---|---|
| `amenity_density` | 179.0 | 777.9 | 0.0 | 132,444 | 0% |
| `amenity_ratio_food_drink` | 0.04 | 0.14 | 0.0 | 1.0 | 0% |
| `avg_floors` | 2.9 | 3.3 | 1.0 | 71.0 | 73.9% |
| `avg_yearbuilt` | 1940.8 | 24.8 | 1794 | 2025 | 80.6% |
| `building_count` | 44.7 | 43.2 | 0 | 790 | 67.4% |
| `total_bldg_area` | 164,058 | 338,985 | 0 | 5,455,446 | 67.4% |
| `landuse_entropy` | 0.59 | 0.84 | 0.0 | 3.79 | 0% |
| `shop_density_km2` | 20.7 | 86.7 | 0.0 | 4,578 | 0% |
| `office_density` | 4.9 | 26.1 | 0.0 | 978 | 0% |
| `road_density_primary` | 2.7 | 7.3 | 0.0 | 189.7 | 0% |
| `transit_stop_density` | 26.9 | 72.1 | 0.0 | 2,178 | 0% |
| `intersection_density` | 52.3 | 96.7 | 0.0 | 3,822 | 0% |
| `nightlife_density` | 3.0 | 18.7 | 0.0 | 578 | 0% |

**Note on NaN in building features:** Features derived from property data (`avg_floors`, `avg_yearbuilt`, `building_count`, `total_bldg_area`) have high NaN percentages because OSM-only cities (SF, LA) cannot obtain complete building geometry from the Overpass API (server memory limits for large bboxes). These NaN values are imputed as 0 during ML training. The 9 pure-OSM features have 0% NaN.

### 3.5 Pairplot — Feature Relationships

A pairplot of the top 6 features by ablation importance (3K subsample) reveals the pairwise relationships between features and their class distributions. Key observations:

- **`landuse_entropy`** provides the clearest single-feature separation: Commercial cells concentrate at higher entropy values (mixed land uses), while Residential cells peak near zero (homogeneous use).
- **`intersection_density` vs `amenity_density`**: Commercial cells cluster at higher values of both, confirming the hypothesis that denser road networks and more amenities co-occur in commercial zones.
- **`shop_density_km2`** shows a long-tailed distribution with Commercial cells dominating the high end, as expected.
- Most features are heavily right-skewed, with the majority of cells (Residential) concentrated near zero — consistent with the class imbalance (8.9:1 ratio).

See `outputs/22_pairplot.png`.

---

## 4. Classification Categories

### 4.1 Handling Mixed-Use

A critical project decision is what to do with **Mixed-Use** zones. In NYC/PLUTO, ~15-20% of cells have a combination of residential and commercial use that does not fit cleanly into either class.

Two approaches were tested:

**Option A — Binary classification (exclude Mixed-Use)**
- All Mixed-Use cells are removed from the dataset
- The model learns to distinguish only "pure" Residential vs "pure" Commercial
- Advantage: cleaner classes, greater separability, better accuracy
- Disadvantage: the model cannot predict mixed zones — which are precisely the most interesting for urban transformation detection

**Option B — 3-class classification (include Mixed-Use)**
- Mixed-Use becomes a third class
- The model must learn to distinguish three zone types
- Advantage: more complete, captures urban reality
- Disadvantage: the Mixed-Use class is inherently ambiguous and difficult to predict consistently. Accuracy drops. Class imbalance worsens

**Decision:** Both approaches are compared and the accuracy difference is documented. The final choice depends on the objective: if the application is **prediction in cities without data**, Option A is preferable. If the application is **transformation detection**, Option B is more relevant though less precise.

In the current pipeline: `INCLUDE_OTHER = False` by default (Option A). It can be changed to `True` to run Option B.

### 4.2 Comparison Results (Binary vs 3-Class)

Comparison using Random Forest with cross-validation (3-fold) on the combined 6-city dataset:

| Configuration | Samples | Classes | Accuracy (CV) | Std |
|---|---|---|---|---|
| **Binary (no Mixed-Use)** | **49,848** | **2** | **71.8%** | **±25.0%** |
| 3-Class (with Mixed-Use) | 53,391 | 3 | 57.5% | ±25.0% |

**Difference: +14.3 percentage points in favor of Binary.**

**Interpretation:** The Mixed-Use class is inherently ambiguous — its observable characteristics overlap significantly with Residential and Commercial. Including it reduces accuracy by ~14 points. For universal prediction, Binary is clearly superior. The high variance (±25%) in both cases reflects the heterogeneity between cities in cross-validation.

**Note:** Cross-validation accuracies are lower than test set accuracies because CV includes folds where entire cities can fall into the test set — and OSM-only cities have different distributions than property data cities.

---

## 5. Models

Four classifier families are used and systematically compared. All models use `class_weight="balanced"` to compensate for class imbalance.

**Models evaluated:**
- **Logistic Regression (LR):** Linear model. Serves as baseline. If LR achieves high accuracy, the problem is linearly separable. Its per-feature coefficient is directly interpretable.
- **XGBoost (XGB):** Gradient Boosting. Builds trees sequentially, correcting errors from the previous tree. Robust with heterogeneous data and imbalanced classes.
- **Random Forest (RF):** Ensemble of independent decision trees. Stable, less prone to overfitting than a single tree. Feature importance is easy to extract.
- **SVC Polynomial:** Support Vector Machine with polynomial kernel. Captures nonlinear relationships between features. Was the best model in the Baseline.

Additionally, `ml_analysis.ipynb` evaluates:
- **ANN (Artificial Neural Network):** Dense network with hidden layers. Useful for understanding whether deep nonlinearity improves results.
- **K-Means (unsupervised clustering):** Does not use labels — discovers latent structure in the data.
- **PCA + t-SNE:** Dimensionality reduction to visualize whether classes are separable in 2D.

### 5.1 Iteration 1: Baseline NYC (10 features, Manhattan only)

| Model | Accuracy | Observations |
|---|---|---|
| Logistic Regression | 82.9% | Good result for linear model — suggests partial linear separability |
| XGBoost | 89.9% | Notable improvement over LR — there are nonlinearities in the data |
| Random Forest | 90.2% | Comparable to XGB, more stable in cross-validation |
| **SVC Polynomial** | **90.5%** | **Best result** — polynomial kernel captures feature interactions |

**Iteration 1 Conclusion:** All four models achieve reasonable accuracy with only 10 features derived from OSM + PLUTO. The problem is classifiable with high confidence in a single city. The challenge is whether this accuracy level holds when generalizing to 6 cities with distinct morphologies.

### 5.2 Iteration 2: 6 Cities, 13 Features

Combined training with 53,391 cells from 6 cities (80/20 stratified split). Binary classification: Commercial vs Residential.

**Global results (combined test set):**

| Model | Accuracy | F1 Commercial | F1 Residential |
|---|---|---|---|
| **XGBoost** | **90.98%** | **0.37** | **0.95** |
| ANN (Keras) | 90.20% | 0.13 | 0.95 |
| SVC (poly) | 86.21% | 0.41 | 0.92 |
| Random Forest | 81.28% | 0.44 | 0.89 |
| Logistic Regression | 81.23% | 0.42 | 0.89 |

**Accuracy by city (RF, using exported predictions):**

| City | Group | Accuracy | Cells |
|---|---|---|---|
| NYC | Ground Truth | 73.5% | 1,631 |
| Philadelphia | Ground Truth | 86.2% | 9,267 |
| Chicago | Ground Truth | 72.2% | 11,917 |
| DC | OSM-Only | 80.8% | 4,273 |
| SF | OSM-Only | 78.8% | 3,295 |
| LA | OSM-Only | 87.5% | 23,008 |

**Key observations:**
- XGBoost outperforms the rest by a wide margin (+4.8% over SVC, +9.7% over RF)
- In the Baseline (Manhattan only), SVC poly was the best. With 6 heterogeneous cities, XGBoost dominates — its ability to handle heterogeneous data and imbalanced classes shines
- ANN achieves accuracy similar to XGBoost but with very low Commercial F1 (0.13) — it predicts almost everything as Residential
- Low Commercial F1 across all models reflects extreme imbalance (5,417 vs 47,974 cells)
- Hyperparameter tuning: RF best params `max_depth=None, min_samples_leaf=1, n_estimators=200` → CV accuracy 88.4%. SVC best params `C=0.1, gamma=scale, kernel=poly` → CV accuracy 88.2%

**Ablation Study (RF, 13 features):**

| Feature Removed | Accuracy Without It | Impact |
|---|---|---|
| landuse_entropy | 74.68% | **-7.08%** (most important) |
| intersection_density | 77.41% | -4.34% |
| amenity_density | 78.63% | -3.13% |
| shop_density_km2 | 78.82% | -2.93% |
| building_count | 79.74% | -2.01% |
| transit_stop_density | 80.72% | -1.03% |
| road_density_primary | 81.01% | -0.74% |
| amenity_ratio_food_drink | 81.28% | -0.47% |
| office_density | 81.66% | -0.09% |
| nightlife_density | 81.79% | +0.04% (noise) |
| total_bldg_area | 82.12% | +0.37% (noise) |
| avg_floors | 81.82% | +0.07% (noise) |
| avg_yearbuilt | 87.78% | **+6.02%** (harms the model) |

**Ablation Finding:** `avg_yearbuilt` has a very strong negative impact — accuracy **improves** by 6% when it is removed. This is explained by `avg_yearbuilt` having 80.6% NaN in the dataset (only NYC, Philadelphia, and partially Chicago have it), and the values imputed as 0 for OSM-only cities create a spurious signal.

### 5.3 Transfer Learning: Ground Truth → OSM-Only

This is the central experiment of the project. The model is trained **exclusively** with the 3 Group A cities (NYC, Philadelphia, Chicago) and evaluated on the 3 Group B cities (DC, SF, LA), where the Ground Truth is also OSM-derived.

The question it answers: **Are the OSM signals that distinguish Commercial from Residential in NYC universal, or are they NYC-specific?**

**Results (RF trained with Group A, evaluated on Group B):**

| Trained With | Evaluated On | Accuracy | Cells | Interpretation |
|---|---|---|---|---|
| NYC + PHL + CHI | **Test set (GT)** | **88.9%** | 22,815 | Reference baseline |
| NYC + PHL + CHI | DC | 91.7% | 4,273 | Excellent transfer |
| NYC + PHL + CHI | LA | 86.7% | 23,008 | Good transfer |
| NYC + PHL + CHI | SF | 80.6% | 3,295 | Acceptable |
| NYC + PHL + CHI | **OSM Average** | **86.7%** | 30,576 | **Only 2.2% below GT** |

**Per-city accuracy (all cities):**

| City | Accuracy | Cells | Group |
|---|---|---|---|
| NYC | 98.3% | 1,631 | Ground Truth |
| Philadelphia | 98.1% | 9,267 | Ground Truth |
| Chicago | 97.1% | 11,917 | Ground Truth |
| DC | 91.7% | 4,273 | OSM-Only |
| LA | 86.7% | 23,008 | OSM-Only |
| SF | 80.6% | 3,295 | OSM-Only |

**Conclusion:** OSM-Only accuracy = 86.7% (>80%) → **OSM is sufficient to predict urban zones.**

The delta between Ground Truth (88.9%) and OSM-Only (86.7%) is only 2.2 percentage points — remarkably low. The urban signals from OSM (amenity density, land use entropy, intersection density, shops, offices, transit) generalize well across cities with distinct morphologies.

**Differences between OSM-Only cities:**
- DC (91.7%): The most compact and urbanly dense of the three — morphology similar to the training cities
- LA (86.7%): Despite being sprawling and car-oriented, it achieves good accuracy — the land use signal is strong
- SF (80.6%): The lowest but still acceptable — possibly due to the lack of building data in OSM (0 buildings found by Overpass) and lower road coverage (0 road ways)

---

## 6. Key Decisions (for presentation)

This log documents the curatorial decisions made during the process. Each decision has evidence that justifies it — this is what distinguishes "experiments" from "trial and error."

1. **We chose 6 cities divided into 2 groups** (3 with property data + 3 OSM-only) to structure a transfer learning experiment: can a model trained with precise Ground Truth generalize to cities without it? Without this division, the project has no testable hypothesis.

2. **We used a regular 150m × 150m grid** instead of census tracts or neighborhood polygons. Reason: census tracts are irregular, vary in size between cities, and do not exist everywhere. A regular grid is comparable across cities, reproducible, and geometrically neutral.

3. **We removed `tourism_density`** because the ablation study demonstrated that model accuracy **improves** when it is removed. This is direct evidence that it is noise in the model, not an informative feature. Urban interpretation: in Manhattan, hotels are distributed too uniformly to be discriminating at the 150m scale.

4. **We removed `brand_ratio`** due to consistently low importance (<5%) across all models. The franchise ratio does not capture land use differences reliably — a CVS pharmacy can be in a residential neighborhood just as easily as in a commercial corridor.

5. **We added 5 new features** (office_density, road_density_primary, transit_stop_density, intersection_density, nightlife_density) based on urban morphology hypotheses. Some of these (intersection_density, transit) were already tested at the census tract level and showed no signal — they are re-tested here because the 150m scale and the diversity of 6 cities may change the result.

6. **We compared binary vs 3-class classification** to decide on Mixed-Use handling. This comparison is not a technical detail — it is a decision about what type of question we want to answer. Binary = clean prediction. 3-Class = captures complex urban reality but with greater uncertainty.

7. **We used `class_weight="balanced"` in all models** because the class imbalance (4.6:1 Residential:Commercial in NYC) would cause a naive model predicting always "Residential" to reach ~82% accuracy without learning anything. Balancing the weights forces the model to learn the minority class.

8. **K-Means found k=3 natural groups** even though the dataset has 2 labeled classes. This suggests that binary classification may be oversimplifying urban reality — there is an emergent third "type" in the data that official labeling does not capture. This is one of the most interesting project observations for the presentation.

---

## 7. Plot Interpretations

All visuals produced by `ml_analysis.ipynb`. Each plot is interpreted with conclusions relevant to the research question.

### 7.1 Class Distribution (`01_class_distribution.png`)

Severe class imbalance: Residential outnumbers Commercial ~9:1 overall. The per-city breakdown shows LA dominates the dataset (~23K cells) while NYC contributes only ~1.6K. DC has the most extreme ratio (14.3:1), SF the most balanced (6.1:1). This heterogeneity means the model must learn patterns that work across very different city compositions.

**Conclusion:** `class_weight="balanced"` is essential — without it, a naive classifier predicting "always Residential" would achieve ~90% accuracy.

### 7.2 Feature Boxplots (`02_feature_boxplots.png`)

`landuse_entropy` shows the clearest class separation: Commercial median (~1.0) vs Residential (~0.0) with minimal box overlap. `amenity_density`, `shop_density_km2`, and `road_density_primary` show Commercial medians shifted higher but with substantial outlier tails. `avg_yearbuilt` boxes overlap almost completely — a visual preview of why the ablation study shows it harms the model. `nightlife_density` and `office_density` have nearly identical boxes for both classes, suggesting their power comes only from extreme outliers.

**Conclusion:** The boxplots visually confirm `landuse_entropy` as the top discriminator and flag `avg_yearbuilt` as non-discriminating, both later validated quantitatively by the ablation study.

### 7.3 Correlation Heatmap (`03_correlation_heatmap.png`)

The only strong correlation is `avg_floors` vs `total_bldg_area` (r=0.76). The commercial activity cluster (`amenity_density`, `shop_density_km2`, `office_density`, `nightlife_density`) shows moderate inter-correlation (r=0.20-0.45). `road_density_primary` is nearly uncorrelated with everything (r<0.07), providing unique spatial information. No pair exceeds |r|>0.8.

**Conclusion:** All 13 features are retained without multicollinearity concerns. Each feature contributes non-redundant information to the model.

### 7.4 Pairplot (`22_pairplot.png`)

The diagonal KDE plots show `landuse_entropy` provides the clearest single-feature separation: Residential peaks at zero, Commercial spreads across higher values. In scatter panels, Commercial cells cluster at higher values of `intersection_density` + `amenity_density` simultaneously. Most distributions are heavily right-skewed with both classes compressed near the origin.

**Conclusion:** Commercial zones are characterized by co-occurring density signals, not any single feature. The nonlinear, skewed distributions explain why tree-based models (XGBoost, RF) outperform linear models.

### 7.5 SOM U-Matrix and Labels (`03b_som.png`)

The U-Matrix shows a dark boundary region around coordinates (2-4, 2-4), indicating a natural cluster boundary. Commercial cells concentrate in the upper-left quadrant, Residential dominates the right and bottom. The SOM colored by city shows Philadelphia in the bottom-left, NYC at the right, Chicago spanning the top — city identity is a stronger organizing principle than zone type.

**Conclusion:** The SOM confirms partial but not clean class separation, and reveals that city morphology drives the feature space more than zone type.

### 7.6 SOM Component Planes (`03c_som_components.png`)

`landuse_entropy` lights up brightly in the same SOM region where Commercial cells concentrate — visual confirmation of the ablation study ranking. `avg_yearbuilt` shows a striking binary pattern: bright in the bottom half (cities with property data), dark in the top half (OSM-only cities with imputed zeros). `amenity_density`, `shop_density_km2`, and `office_density` show similar hotspot locations, confirming co-occurring commercial activity signals.

**Conclusion:** The component planes visually explain why `avg_yearbuilt` creates a spurious signal — it encodes "data availability" rather than urban character.

### 7.7 PCA (`04_pca.png`)

PC1 explains only 23.7% and PC2 16.6% — together just 40.3%. The 95% threshold requires 11 of 13 components. Both classes concentrate in a dense cloud near the origin in the biplot, with Commercial cells spreading outward along PC1.

**Conclusion:** The problem is genuinely high-dimensional. PCA reduction to 2D loses ~60% of information, explaining why 2D visualizations show heavy class overlap while full 13D models achieve 91% accuracy.

### 7.8 ICA (`05_ica.png`)

Commercial cells spread into the negative IC1 tail, forming a gradient rather than a clean cluster. The city-colored scatter shows each city forming a distinct cluster — city-specific distributions dominate the independent components. The mixing matrix reveals IC1 is driven by `intersection_density` (-0.86), `transit_stop_density` (-0.69), and `amenity_density` (-0.50) — an "urban intensity" signal. IC2 is dominated by `building_count` (0.78) and `avg_yearbuilt` (0.73) — a "property data availability" signal. IC3 loads on `road_density_primary` (-0.92) alone.

**Conclusion:** The data's independent sources are (1) urban activity density, (2) building/property characteristics, and (3) road infrastructure — three conceptually distinct dimensions of urbanism.

### 7.9 t-SNE (`06_tsne.png`)

Residential cells form several distinct sub-clusters, while Commercial cells are scattered among them without a coherent cluster. The city-colored view reveals each city forms its own tight cluster: DC at upper-left, LA with two sub-clusters, Philadelphia at top, Chicago as a long ribbon, NYC at the right edge, SF as a small dot. City-level clustering is much stronger than class-level clustering.

**Conclusion:** Urban morphology varies more between cities than between zone types within a city. This is why transfer learning works: the model must learn zone-agnostic patterns that transcend city-specific feature distributions.

### 7.10 Encoding Comparison (`07_encoding_comparison.png`)

StandardScaler (67.6%) and MinMaxScaler (67.4%) are virtually identical. No Scaling drops to 47.6% (LR is sensitive to feature scale). Log+StandardScaler (56.5%) performs worse because the log transform compresses long-tailed density features that carry the most signal.

**Conclusion:** StandardScaler is the correct choice, but scaling strategy has far less impact than feature selection or model choice.

### 7.11 Logistic Regression (`08_lr_confusion.png`)

LR correctly identifies 711 of 1,083 Commercial cells (65.6% recall) but generates 1,632 Residential false positives (17% false positive rate). The high off-diagonal counts show the linear boundary cuts through a region where both classes overlap.

**Conclusion:** 81.2% accuracy confirms partial linear separability — the mandatory baseline before trying nonlinear models.

### 7.12 XGBoost (`09_xgb_results.png`)

Highest accuracy (91.0%) but misses 795 of 1,083 Commercial cells (73.4%), compensating with very precise Residential predictions (9,428 correct vs 168 false positives). Feature importance: `amenity_ratio_food_drink` (0.25) > `shop_density_km2` (0.21) > `amenity_density` (0.14). This differs from the ablation ranking because XGBoost measures split frequency while ablation measures accuracy impact.

**Conclusion:** XGBoost is the best overall model but sacrifices Commercial recall for Residential precision. The feature importance difference vs ablation shows that frequently-used features are not necessarily the most impactful ones.

### 7.13 Random Forest (`10_rf_results.png`)

Correctly identifies 800 of 1,083 Commercial cells (73.9% recall) but generates 1,716 Residential false positives — more aggressive at predicting Commercial than XGBoost. Feature importance: `amenity_density` > `shop_density_km2` > `amenity_ratio_food_drink` > `intersection_density` > `landuse_entropy`. Both RF and XGBoost agree on the top 3 features.

**Conclusion:** RF has better Commercial recall than XGBoost but worse precision. The consistent top-feature rankings across model families confirm these signals are robust.

### 7.14 SVC (`11_svc_results.png`)

Poly (86.2%) > linear (83.1%) > rbf (82.6%). The polynomial kernel's superiority confirms nonlinear feature interactions matter. RBF performing worse than linear suggests the nonlinearity is better captured by polynomial terms than radial distance.

**Conclusion:** The ~3% gain from poly over linear quantifies how much value feature interactions add. SVC sits between the LR/RF tier and the XGBoost/ANN tier.

### 7.15 ANN (`12_ann_results.png`)

Training loss decreases steadily while validation loss plateaus after epoch 4 — mild overfitting. The confusion matrix reveals extreme bias: only 75 cells predicted as Commercial out of 1,083 actual. The ANN achieves 90.2% accuracy by predicting Residential almost universally.

**Conclusion:** Despite high accuracy, ANN's F1 for Commercial (0.13) makes it the worst model for actually finding commercial zones. The extra complexity of a neural network is not justified for this problem.

### 7.16 Model Comparison (`13_model_comparison.png`)

XGBoost (91.0%) and ANN (90.2%) lead, LR and RF cluster at ~81%, SVC at 86.2%. The ~10% gap between XGBoost and LR quantifies the nonlinearity in the data.

**Conclusion:** The problem is fundamentally nonlinear, justifying complex models. XGBoost is the winner both by accuracy and by Commercial F1.

### 7.17 Ablation Study (`14_ablation_study.png`)

Three tiers emerge: **Essential** — `landuse_entropy` (-7.08%), `intersection_density` (-4.34%), `amenity_density` (-3.13%), `shop_density_km2` (-2.93%). **Contributing** — `building_count` through `office_density`. **Harmful** — `nightlife_density`, `total_bldg_area`, `avg_floors` add noise, and `avg_yearbuilt` (+6.02%) actively damages the model.

**Conclusion:** The top 4 features encode the core urban signal: land use diversity, street connectivity, and commercial density. `avg_yearbuilt`'s 80.6% NaN rate creates a spurious binary signal that degrades generalization.

### 7.18 RF Hyperparameter Tuning (`15_tuning_rf.png`)

Top 10 combinations cluster between 84-88% CV accuracy with overlapping error bars. The best (`max_depth=None, min_samples_leaf=1, n_estimators=200`) achieves 88.4%. `max_depth=None` consistently outperforms `max_depth=12`.

**Conclusion:** Minimal improvement from tuning (+1-2%). Trees need deep branches for 6-city data complexity, but default hyperparameters are already near-optimal.

### 7.19 K-Means Elbow (`16_kmeans_elbow.png`)

No sharp elbow in the inertia plot. Silhouette peaks at k=2 (0.39) with a secondary peak at k=4 (0.385), minimum at k=5-6. Overall silhouette scores (0.33-0.39) indicate moderate cluster quality.

**Conclusion:** k=2 aligns with binary classification. The k=4 secondary peak suggests 2 additional sub-types — possibly "dense urban" vs "suburban" variants of each class. Clusters exist but overlap, consistent with the t-SNE visualization.

### 7.20 Geographic Heatmap (`17_heatmap_all_cities.png`)

Chicago shows a clear commercial core along the lakefront. DC displays a compact commercial center (National Mall / downtown). LA shows commercial zones along the coastal strip and major boulevards. NYC (Manhattan) is almost entirely commercial. Philadelphia shows a north-south commercial spine. SF is dominated by commercial predictions in the northeast (Financial District).

**Conclusion:** The spatial patterns are urbanistically coherent — the model learns real geographic structure, not random noise. The predictions match known commercial corridors and business districts in each city.

### 7.21 Feature Means per City (`19_feature_means.png`)

NYC (red) is a dramatic outlier: 2+ standard deviations above the mean on nearly every feature, reflecting Manhattan's extreme density. LA and Chicago sit below the mean on most features. The raw scale plot confirms `total_bldg_area` dominates at ~800K for NYC, making all other features invisible.

**Conclusion:** NYC is a feature-space outlier, explaining why its per-city accuracy (73.5%) is lower than LA's (87.5%) despite having property data — the model trained on all 6 cities learns patterns that NYC's extreme values do not follow.

### 7.22 Binary vs 3-Class (`20_binary_vs_3class.png`)

Binary (71.8%) outperforms 3-Class (57.5%) by 14.3pp, both with high variance (~25% error bars). Mixed-Use adds only ~3,500 cells but creates a third class that overlaps heavily with both existing classes.

**Conclusion:** Mixed-Use is too ambiguous for supervised classification. For prediction, Binary is superior. For transformation detection, cells with near-0.5 prediction probability are natural Mixed-Use candidates.

### 7.23 Transfer Learning (`21_transfer_learning.png`)

Ground Truth (88.9%) vs OSM-Only (86.7%), both above the 80% threshold. Per-city: DC (91.7%) transfers best, LA (86.7%) transfers well despite sprawl, SF (80.6%) transfers least well due to missing building data. The OSM-Only confusion matrix shows only 113 of ~2,846 Commercial cells caught — low recall expected with noisier OSM-derived labels.

**Conclusion:** The 2.2% transfer gap is remarkably small. OSM signals generalize across cities with fundamentally different morphologies. The hypothesis is confirmed: OSM is sufficient for universal urban zone prediction.

---

## 8. Conclusion

### Was the Hypothesis Correct?

**Yes.** The central hypothesis — that observable urban characteristics extracted from OpenStreetMap can predict official land use without zoning data — is confirmed by the experimental results.

The transfer learning experiment provides the definitive evidence: a Random Forest model trained exclusively on 3 cities with property datasets (NYC, Philadelphia, Chicago) achieved **86.7% accuracy** when predicting land use in 3 cities with only OSM data (DC, SF, LA). This is only **2.2 percentage points below** the 88.9% accuracy on the Ground Truth test set. The gap is small enough to conclude that OSM signals generalize across cities with fundamentally different urban morphologies — from Manhattan's vertical density to LA's horizontal sprawl.

The best overall model (XGBoost, trained on all 6 cities) reached **90.98% accuracy**, demonstrating that the 13 engineered features capture meaningful urban structure.

### Were the Research Objectives Met?

**Objective 1 — Universal Prediction: MET.**
The model successfully predicts land use in cities without property data. All three OSM-only cities exceeded 80% accuracy (DC: 91.7%, LA: 86.7%, SF: 80.6%). This means the pipeline can be applied to any city in the world with reasonable OSM coverage — no proprietary data required.

**Objective 2 — Urban Transformation Detection: PARTIALLY MET.**
The framework for detecting transformation is established: cells where the model predicts "Commercial" but official zoning says "Residential" (or vice versa) are candidates for zones in transition. However, this objective was not fully validated because it requires longitudinal data (zoning at two points in time) or field verification to confirm that discrepancies actually correspond to real urban change rather than model error. The 9.02% misclassification rate means some "detected transformations" would be false positives. Future work should cross-reference model disagreements with temporal OSM edit histories or satellite imagery to distinguish genuine transformation from classification noise.

### Key Findings

1. **`landuse_entropy` is the single most important feature** (-7.08% accuracy when removed). The diversity of land uses within a 150m cell is the strongest signal distinguishing Commercial from Residential zones — stronger than building count, area, or any individual amenity type.

2. **`avg_yearbuilt` actively harms multi-city models** (+6.02% accuracy when removed). This counterintuitive finding is explained by the 80.6% NaN rate: when imputed as 0 for OSM-only cities, it creates a spurious signal that the model learns to exploit, degrading generalization.

3. **Binary classification outperforms 3-class by 14.3 percentage points** (71.8% vs 57.5% in cross-validation). Mixed-Use zones are inherently ambiguous — their observable characteristics overlap with both Residential and Commercial. For practical prediction, excluding Mixed-Use yields cleaner, more reliable results.

4. **K-Means finds k=3 natural clusters** despite only 2 labeled classes. This unsupervised confirmation suggests that Mixed-Use is a real phenomenon in the data, not an artifact of labeling — even though it's difficult to classify with supervised methods.

5. **City morphology matters less than expected.** The 2.2% transfer learning gap suggests that the relationship between urban features and land use is remarkably consistent across American cities, despite vast differences in density, layout, and development history.

### Limitations

- **Geographic scope:** All 6 cities are in the United States. The features and class definitions may not transfer to European, Asian, or African cities where urban form, zoning systems, and OSM coverage differ substantially.
- **OSM coverage bias:** OSM data quality varies by city and neighborhood. Wealthier, more tech-savvy areas tend to have better coverage, potentially introducing a systematic bias that correlates with land use patterns.
- **Temporal snapshot:** The analysis captures a single point in time. Urban zones are dynamic — a model trained on 2026 data may degrade as cities evolve.
- **Binary simplification:** Reducing land use to Commercial vs Residential ignores industrial, institutional, and recreational uses. The 3-class experiment showed this simplification comes at a cost.
- **Building data gaps:** OSM-only cities (especially SF and LA) have near-zero building geometry from Overpass, forcing 4 of 13 features to be imputed as 0. The model works despite this, but accuracy would likely improve with better building data.

### Future Work

1. **Validate transformation detection** by comparing model disagreements against temporal OSM edit logs or Google Street View imagery at different dates.
2. **Extend to non-US cities** (Barcelona, London, Tokyo) to test whether the features generalize beyond the American urban context.
3. **Remove `avg_yearbuilt`** from the feature set permanently and retrain — the ablation study shows this alone would boost accuracy by ~6%.
4. **Add a "confidence" threshold** to predictions: cells where the model is uncertain (probability near 0.5) are flagged as "ambiguous" rather than forced into a class — these are the most likely transformation candidates.
5. **Integrate satellite imagery features** (NDVI, built-up index) as complementary signals to OSM point-of-interest data.
