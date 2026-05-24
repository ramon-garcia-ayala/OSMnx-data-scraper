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

## 7. Key Plots for Presentation

The following visuals are produced by `ml_analysis.ipynb` and are the most relevant for communicating the project's findings:

### Context Plots (establish the problem)

| Plot | Expected File | What It Demonstrates |
|---|---|---|
| City grid map | `heatmap_[city].png` | Scale and geographic distribution of the dataset — the problem has a spatial dimension |
| Class distribution by city | (bar chart in EDA) | Imbalance varies by city — NYC is not representative of all |

### Baseline Plots (Iteration 1)

| Plot | Expected File | What It Demonstrates |
|---|---|---|
| Feature importance (RF + XGB) | (in ml_analysis) | `total_bldg_area` dominates — building mass is the strongest signal |
| Ablation study | (accuracy bars per removed feature) | Evidence for why `tourism_density` and `brand_ratio` were removed |
| Confusion matrix (SVC, Iteration 1) | (in ml_analysis) | Where the best model makes mistakes — what type of cells are hard to classify |

### Multi-city Comparison Plots (Iteration 2)

| Plot | Expected File | What It Demonstrates |
|---|---|---|
| Accuracy by city and model | `comparison_accuracy.png` | Are some cities more predictable than others? Why? |
| Normalized feature importance by city | (feature heatmap) | Does the same feature matter equally in NYC and LA? |
| Combined heatmap (6 cities) | (folium or matplotlib) | The research question visualized in space |

### Clustering and Dimensionality Plots

| Plot | Expected File | What It Demonstrates |
|---|---|---|
| t-SNE colored by class | (in ml_analysis) | Are classes visually separable in 2D? Is there overlap? |
| t-SNE colored by city | (in ml_analysis) | Does each city form its own cluster or do they mix? If NYC and SF are mixed, patterns are similar |
| K-Means k=3 on t-SNE | (in ml_analysis) | The emergent third cluster — is it consistent with Mixed-Use? |
| PCA scree plot | (in ml_analysis) | Requires 8 components for 95% variance — the problem is genuinely multidimensional |

### Central Project Plot (Transfer Learning)

| Plot | Expected File | What It Demonstrates |
|---|---|---|
| Accuracy: trained on Group A, evaluated on Group B | (table + bars) | The answer to the research question: does OSM generalize? |
| Prediction maps for DC/SF/LA | (folium per city) | Predictions in cities without Ground Truth — this is what the project enables |

---

*Document completed. Pipeline executed with 6 cities (60,278 cells), ML notebook run with all models and visualizations. Transfer Learning results confirm the main hypothesis: OSM is sufficient for universal urban zone prediction (86.7% accuracy, only 2.2% below Ground Truth).*
