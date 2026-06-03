# Session Log — Ramón

Document that tracks the progress of the Dimensionality / Shallow Learning analysis on the OSM dataset. Updated at the end of every conversation turn.

---

## Session 2 — 2026-05-26

### Bundle folder rename
- Folder renamed from `2026.05.25_Dimensionality_ramón` to `2026.05.25_ShallowLearning-Dimensionality_ramón` (third rename of this bundle: Deep-Learning → Dimensionality → ShallowLearning-Dimensionality).
- Updated the absolute path in 3 places:
  - `config.py` → `_BUNDLE_DIR`
  - `02_DimReduction_OSM.ipynb` → cell `b7e09131` (`sys.path.insert`)
  - `01_ShallowLearning_OSM.ipynb` → `cell-4` (`sys.path.insert`)
- README.md tree updated.
- Memory rule (`session_log_practice.md`) updated to point to the new path.
- Had to kill Python processes before renaming (Jupyter kernels were holding a lock on the folder).

---

## Session 1 — 2026-05-25

### Session goal
Create a dimensionality reduction notebook (PCA + t-SNE + SOM) to analyze whether OpenStreetMap data alone is enough to distinguish Commercial vs Residential zones across 6 US cities.

### Artifacts created / modified
- **`Tutorial8_DimReduction_OSM.ipynb`** — notebook adapted from `Copy_of_Tutorial8_DimReduction.ipynb` (Iris) to the `all_cities_combined.csv` dataset.
- **`SESSION_LOG.md`** — this document.
- **`.gitignore`** — added rules `.venv/`, `venv/`, `env/` (commit `5bd4743`).
- **`.venv/pyvenv.cfg`** — removed from git tracking (it was pointing to another contributor's machine).

### Functional changes to the notebook
1. **Dataset loading** via absolute path to `all_cities_combined.csv` (60,278 rows × 19 columns).
2. **Configurable city filter** via `CITIES` and `CITY_GROUP`:
   - `CITIES = None` + `CITY_GROUP = 'ground_truth'` → NYC + Philadelphia + Chicago
   - `CITY_GROUP = 'osm_only'` → DC + SF + LA
   - `CITIES = ['NYC']` → single city
3. **Class balancing** via `BALANCE_METHOD`:
   - `'undersample'`: random subsample of Residential to N of Commercial (no synthetic data)
   - `'none'`: no balancing
4. **Feature selection** via `FEATURES` and `EXCLUDE_FEATURES`:
   - `FEATURES = None` → use all 14
   - Explicit list to pick between 4 and 10 (DL recommendation)
   - `EXCLUDE_FEATURES` to drop columns with many NaNs (e.g. `avg_yearbuilt`, `avg_floors`)
5. **Reproducibility** centralized in `RANDOM_STATE = 42` (propagated to `sample`, `TSNE`, `MiniSom`).
6. **PCA**: scree plot, biplot with loadings, components heatmap.
7. **t-SNE**: subsample up to 8,000 points, adaptive perplexity.
8. **SOM**: 10×10 grid, 2,000 iterations, U-matrix.

### Problems encountered and solved
| Problem | Cause | Fix |
|---|---|---|
| `pip install minisom` failed with `No Python at 'C:\Users\Hani\miniconda3\python.exe'` | The inherited `.venv` was created on "Hani"'s machine and carried absolute paths to his disk. | Removed from git tracking, recreated with `py -3.11 -m venv .venv` pointing to `C:\Users\gramo\AppData\Local\Programs\Python\Python311`. |
| `KeyError: 'zone_type'` when printing `data['zone_type'].value_counts()` after balancing. | Pandas 3.0 changed the behavior of `groupby().apply()`: it no longer includes the grouping column in the result. | Replaced with a list comprehension: `[data[data['zone_type']==cls].sample(...) for cls in ...]` + `pd.concat`. |

### Environment
- Python **3.11.0** (`C:\Users\gramo\AppData\Local\Programs\Python\Python311`)
- Venv at `E:\IAAC Local GIT Repositories\OSMnx-data-scraper\.venv`
- Installed dependencies: `pandas 3.0.3`, `numpy 2.4.6`, `scikit-learn 1.8.0`, `seaborn`, `matplotlib`, `minisom 2.3.6`, `ipykernel`, `jupyter`, `papermill 2.7.0`, `geopandas 1.1.3`, `folium 0.20.0`, `shapely 2.1.2`, `osmnx 2.1.0`, `mlxtend 0.24.0`, `xgboost 3.2.0`, `plotly 6.7.0`, `contextily`, `rasterio`, `mercantile`, `geopy`.
- Kernel `python3` registered at `E:\...\.venv\share\jupyter\kernels\python3` (papermill finds it by name).
- **Pending** (if needed later): `tensorflow` for `ml_analysis.ipynb` (neural networks).

### Data distribution (for reference)
After global filter to Commercial/Residential:

| City | Commercial | Residential | R:C ratio |
|---|---|---|---|
| NYC | 283 | 737 | 2.6 |
| SF | 463 | 1,805 | 3.9 |
| Chicago | 1,343 | 9,702 | 7.2 |
| Philadelphia | 945 | 7,956 | 8.4 |
| LA | 2,104 | 20,596 | 9.8 |
| DC | 279 | 3,635 | 13.0 |
| **Total** | **5,417** | **44,431** | **8.2** |

### State at end of session
- Notebook structurally complete and runnable.
- Current config: `CITY_GROUP = 'ground_truth'`, `BALANCE_METHOD = 'undersample'`, `FEATURES = None`, `EXCLUDE_FEATURES = []`.
- After balancing: 5,142 rows (2,571 Commercial + 2,571 Residential).
- Balancing bug fixed; user pending end-to-end re-run.

### Suggested next steps
1. Do **Restart Kernel + Run All** to validate the full pipeline after the fix.
2. Inspect how many NaNs each feature has (`data[ALL_FEATURES].isna().sum()`) and decide which columns to move to `EXCLUDE_FEATURES`.
3. Define a `FEATURES` list of 6–8 "clean" columns for the main DL experiment.
4. Repeat the analysis with `CITY_GROUP = 'osm_only'` and compare visual separability vs the `ground_truth` group.

### Accumulated notes and clarifications
- **Normalization**: the notebook does normalize inside the pipeline, although `data_num` keeps raw values on purpose (for inspection and interpretable plots).
  - `data_scaled` = StandardScaler (μ=0, σ=1) → used by **PCA and t-SNE**.
  - `X_scaled` = MinMaxScaler (0–1) → used by **SOM**.
  - StandardScaler is recommended for PCA/t-SNE (statistical standard), MinMaxScaler only for SOM.

### Orchestrator (Grid-Finding)
- Installed dependencies to run `previous_workflows/Grid-Finding/00_orchestrator.ipynb`: papermill, geopandas, folium, shapely, osmnx, mlxtend, xgboost.
- Kernel `python3` already registered and pointing to the venv → papermill finds it automatically.
- The orchestrator uses papermill to run notebooks 01-09 per borough (Manhattan, Brooklyn, etc.) and merge the CSVs.
- **Bug found**: orchestrator was running with a `cwd` other than `Grid-Finding/`, so `PLUTO_PATH = "../ramy/..."` would not resolve and every notebook crashed with `FileNotFoundError`.
- **First fix attempt (didn't work)**: `os.chdir(ORCHESTRATOR_DIR)` in the imports cell. Reason it failed: papermill spawns the child kernel in a separate process that **does not inherit** the parent's modified cwd.
- **Final fix**: pass `cwd=str(ORCHESTRATOR_DIR)` directly to `pm.execute_notebook(...)` in the `run_notebook` helper (cell `bd804c46`). Also: use absolute path for the input notebook (`ORCHESTRATOR_DIR / nb_path`) and save a copy of the failed notebook as `_failed_<n>.ipynb` for debugging.
- **Extra fix**: `PLUTO_PATH` changed to absolute path (`E:\IAAC...\ramy\NYC_pluto_25v4_csv\pluto_25v4.csv`) in the orchestrator parameters cell. Reason: if the user opens one of notebooks 01/03/04 directly from VS Code, cwd ends up at the workspace root and the relative path `../ramy/...` does not resolve. Absolute path works from any cwd.
- **Convention**: notebooks 01-06 must be run ONLY via the orchestrator, never directly, because they depend on `grid.json` in cwd and on the chain of CSVs produced by the orchestrator.

### First successful orchestrator run (Manhattan)
- Notebooks 01-07 all OK. Times: 01 (5.7s), 02 (3.8s), 03 (4.6s), 04 (4.9s), 05 (3.1s), 06 (3.0s), 07 (7.1s).
- Output: `csv/Manhattan/combined_grid.csv` with **1,810 cells × 15 columns**.
- 08 (Heatmap) failed because `contextily` was missing → installed afterwards.
- Extra dependencies added to the venv: `contextily`, `rasterio`, `mercantile`, `geopy`, `affine`, `cligj`, `click-plugins`, `geographiclib`.

### `SOURCE` parameter in Tutorial 8
- Added a new parameter `SOURCE = 'property' | 'osm' | 'mixed'` to split the analysis by data source:
  - `'property'` (default) → forces ground_truth cities (NYC/Phila/Chicago) and uses 5 features from official datasets (PLUTO/OPA/Cook County): `cell_lot_count, avg_floors, avg_yearbuilt, total_bldg_area, building_count`.
  - `'osm'` → respects `CITIES`/`CITY_GROUP` and uses 9 OpenStreetMap features.
  - `'mixed'` → forces ground_truth, uses the combined 14 features for PCA/t-SNE/SOM, **and exports two separate CSVs** (`dataset_property.csv` and `dataset_osm.csv`) for later comparison.
- Constants `PROPERTY_FEATURES` (5) and `OSM_FEATURES` (9) replaced the old `ALL_FEATURES`; `ALL_FEATURES = PROPERTY_FEATURES + OSM_FEATURES`.
- CSVs saved to `ramon/csv/dataset_<subset>.csv` with columns `city, cell_id, zone_type, <features>`.
- Naming `property` (not `pluto`) chosen because it matches `load_property_data()` and `feature_flags.needs_pluto` in the repo — generic name covering PLUTO/OPA/Cook County.

### Reorganization into a dated bundle folder
- Created folder `2026.05.25_Dimensionality_ramón/` at the **repo root** (first lived inside `ramon/`, later moved out at user's request so it can be treated as a standalone deliverable independent of the personal workspace).
- Moved into that folder: `Tutorial8_DimReduction_OSM.ipynb` and this `SESSION_LOG.md`.
- Added `README.md` with data flow and notebook parameters.
- **Important clarification**: Tutorial 8 **does not extract data** directly — it consumes a pre-built CSV. The extraction is done by `Final-Search-ML-Pipeline/run_pipeline.py` and lives in `Final-Search-ML-Pipeline/csv/all_cities_combined.csv` (not included in this bundle due to size / gitignored).
- Updated the memory rule (`session_log_practice.md`) to point to the new SESSION_LOG location.

### Centralizing config in `config.py`
- Created `2026.05.25_Dimensionality_ramón/config.py` with:
  - Editable parameters (SOURCE, CITIES, CITY_GROUP, BALANCE_METHOD, FEATURES, EXCLUDE_FEATURES, RANDOM_STATE, TEST_SIZE)
  - Constants (GROUP_MAP, PROPERTY_FEATURES, OSM_FEATURES, ALL_FEATURES)
  - Paths (CSV_PATH, EXPORT_DIR, PLOTS_DIR)
  - Helpers `resolve_cities(data_raw_cities)` and `resolve_features()` encapsulating the filtering logic
- Tutorial 8 (`b7e09131`, `423e339b`, `62ba2a98`) and Tutorial 9 (`cell-4`, `cell-7`, `cell-10`) now read from `config.py` via `from config import *` + `importlib.reload(config)`.
- Benefit: the user edits a single file and both notebooks react. Zero duplicated filtering logic.
- Pattern: `selected_cities = resolve_cities(data_raw['city'].unique())` and `feature_cols = resolve_features()`.

### Tutorial 9 — Consolidated Shallow Learning
- Created `Tutorial9_ShallowLearning_OSM.ipynb` (31 cells) in the Dimensionality folder.
- Consolidates the logic of notebooks 07-10 from `Grid-Finding/` adapted to the multi-city `all_cities_combined.csv` dataset (instead of Manhattan-only `combined_grid.csv`).
- Same config block as Tutorial 8: `SOURCE`, `CITIES`, `CITY_GROUP`, `BALANCE_METHOD`, `FEATURES`, `EXCLUDE_FEATURES`, `RANDOM_STATE`.
- Sections: config + filters + balancing (same as Tutorial 8) → EDA (countplot, boxplots, correlation from 07) → pairplot (from 10) → train/test + 3 shallow models (LogReg/RF/XGBoost from 07) → feature importance → geospatial heatmap with subplot per city (adapted from 08) → model comparison + accuracy per city + cross-tab (adapted from 09).
- PNG outputs in `2026.05.25_Dimensionality_ramón/outputs/` with `_<SOURCE>` suffix. Predictions in `ramon/csv/predictions_<SOURCE>.csv`.
- Suggested pattern: run twice (SOURCE='property' and 'osm') for comparative evidence.

### Bug: hardcoded labels in `myplot`
- The `myplot` function (cell `8bd36927`) had `'PC1'` and `'PC2'` hardcoded as axis labels, column names and title. Consequence: when called with PC1+PC3 or PC3+PC4, the plots still showed "PC1" and "PC2" — confusing and false.
- Fix: added parameters `x_name='PC1'` and `y_name='PC2'` (with defaults for backwards compatibility), propagated to `pd.DataFrame(columns=...)`, `xlabel`, `ylabel` and `title`.
- Updated the 3 cells that call `myplot` (`17d59beb`, `72ad5037`, `46f1bb96`) to pass the correct names depending on the components being plotted.

### Interactive 3D visualizations with Plotly
- Installed `plotly 6.7.0` in the venv.
- Added to the notebook (after the PCs pairplot):
  - **PCA 3D**: rotatable scatter of PC1/PC2/PC3 with `color=zone_type` and `size=cell_lot_count` (bubble effect, 4th visual dimension). Captures ~52% of variance vs 39% in 2D.
  - **t-SNE 3D**: same subsample as the 2D t-SNE but with n_components=3. Better local cluster separation than PCA.
- Hover shows `city` + `cell_id` to identify points. Clicking the legend hides/isolates classes.
- Reason: user wanted to see multidimensional "bubbles and clusters". 2D PCA was leaving 61% of variance unvisualized.

### Extra biplots + top-4 PCs pairplot
- User reported that with 8 PCs the variance is heavily spread (PC1 24%, PC2 15%, PC3 13%, PC4 12%, ..., cumsum PC1-2 = only 39%).
- Added three extra PCA visualizations to the notebook (after the loadings):
  - Biplot PC1 vs PC3 (user manual).
  - Biplot PC3 vs PC4 (user manual).
  - Pairplot of the top 4 PCs with `sns.pairplot` (inserted by Claude). Covers the 6 unique combinations (4 choose 2) in a single figure with KDE on the diagonal.
- Reason: with PC1+PC2 capturing only 39%, the traditional biplot misses 61% of the structure. The pairplot exposes class separation in secondary PCs.

### Clarification on property vs osm comparison
- User asked whether the two CSVs (`dataset_property.csv` and `dataset_osm.csv`) could have the same features.
- Answer: features are disjoint by design (property=building stats, osm=urban context). To compare "which source predicts zoning better" we do not need the same columns — train the same kind of model on each CSV and compare accuracy. That is the scientifically honest comparison.
- User decision: "Same predictive power" — keep the current design and compare model results, not individual features.

### Fork of Grid-Finding to the repo root
- Copied notebooks 00-09 from `previous_workflows/Grid-Finding/` to `Grid-Finding/` (excluding outputs: `cache/`, `csv/`, `outputs/`, `grid.json`, `_failed_*.ipynb`).
- `.gitignore` already present in the new folder (blocks `cache/`, `csv/`, `outputs/`).
- `00_orchestrator.ipynb` updated: `ORCHESTRATOR_DIR` now points to `E:\...\Grid-Finding` (not `previous_workflows/...`).

### New notebook 10_pairplot.ipynb
- Created at `Grid-Finding/10_pairplot.ipynb`. Replicates the pairplot logic from `Tutorial8_DimReduction_OSM.ipynb`:
  - Reads `combined_grid.csv` (`CSV_PATH` parameter via papermill).
  - Filters to Commercial vs Residential.
  - Auto-selects numeric features (excludes ids and label).
  - Imputes NaN with the median.
  - Generates the pairplot with subsample of 2,000 points, `hue='zone_type'`, alpha=0.4, s=12.
  - Saves PNG to `PLOTS_DIR/pairplot_features.png` (dpi 120).
- Has the parameters cell tagged `parameters` (unlike 01-06), so papermill can inject `CSV_PATH` and `PLOTS_DIR` without a warning.
- The orchestrator calls it after notebook 08 (Heatmap), inside the per-borough loop.

### Zombie Jupyter processes
- After several failed orchestrator attempts, 10 Python processes remained hanging (4 from the project venv, 6 from other installations). That saturated VS Code and left `01_grid_definition.ipynb` stuck on an infinite "Loading..." spinner.
- **Solution applied**: `Get-Process python | Stop-Process -Force` + `Developer: Reload Window` in VS Code.
- **Lesson**: if VS Code stalls loading notebooks, first check zombie Python processes with `Get-Process python | Select-Object Id, Path`.
- **Harmless warning**: papermill prints `Passed unknown parameter: GRID_CONFIG / Input notebook does not contain a cell with tag 'parameters'`. Not fatal because notebooks have `GRID_CONFIG = "grid.json"` as a normal variable and that default matches what the orchestrator writes in cwd.

### Quick 3-model accuracy bar chart
- Added a cell to Tutorial 9 plotting LogReg, RandomForest and XGBoost test accuracy side by side with percentages on top of each bar and a red dashed line at 50% (random baseline).
- Saved to `outputs/10_accuracy_3models_<SOURCE>.png`.

### ROC curves of the 3 models
- Added another cell to Tutorial 9 plotting the ROC curves of the 3 models on the same axes, with AUC in the legend. Diagonal grey line indicates random.
- Saved to `outputs/11_roc_3models_<SOURCE>.png`.

### Notebook renaming + separate output folders
- Renamed the two notebooks dropping the `Tutorial8_` / `Tutorial9_` prefixes (user later prefixed them with `02_` / `01_`):
  - `Tutorial8_DimReduction_OSM.ipynb` → `02_DimReduction_OSM.ipynb`
  - `Tutorial9_ShallowLearning_OSM.ipynb` → `01_ShallowLearning_OSM.ipynb`
- `config.py` extended with two output paths:
  - `PLOTS_DIR_DIMRED`  → `outputs/DimReduction/`
  - `PLOTS_DIR_SHALLOW` → `outputs/ShallowLearning/`
  - Old `PLOTS_DIR` kept as fallback.
- Each notebook's config cell now assigns `PLOTS_DIR = PLOTS_DIR_<corresponding>` so all existing `savefig(f"{PLOTS_DIR}/...")` calls land in the right subfolder.
- Added `plt.savefig()` calls to the DimReduction notebook (it didn't save plots before): scree, biplots (one per call to `myplot`), loadings heatmap, PCs pairplot, t-SNE 2D, SOM mapping, U-matrix.
- Removed duplicated Spanish cells (cell-31 through cell-34) that were leftover from the earlier translation pass in the Shallow Learning notebook.
- README.md tree section updated with the new file names and dual-folder output structure.

### Full translation to English
- User requested all text inside `2026.05.25_Dimensionality_ramón/` be in English.
- Translated: `config.py` (comments and docstrings), `README.md`, this `SESSION_LOG.md`, `Tutorial8_DimReduction_OSM.ipynb` (markdown cells + code comments), `Tutorial9_ShallowLearning_OSM.ipynb` (markdown cells + code comments).
- Memory rule (`session_log_practice.md`) updated so future sessions know the active log lives in this English file at the new path.
