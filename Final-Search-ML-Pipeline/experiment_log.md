# Experiment Log — Clasificación de Zonas Urbanas
**Curso:** DEML — Digital Tools for Data Encoding and Machine Learning
**Programa:** MaCAD26, IAAC Barcelona
**Proyecto:** Clasificación predictiva de zonas urbanas y detección de transformación desde datos abiertos
**Última actualización:** 24 de mayo de 2026

---

## 1. Concepto y Pregunta de Investigación

### Pregunta principal

> ¿Pueden las características observables de una ciudad — densidad de amenidades, tipología edificatoria, red vial, actividad comercial — extraídas de OpenStreetMap predecir su uso de suelo oficial **sin datos de zonificación**?

### Motivación

Los datos de zonificación oficial son costosos, propietarios o simplemente inexistentes en muchas ciudades del mundo. OpenStreetMap (OSM), en cambio, es una fuente abierta, colaborativa y global. Si un modelo entrenado con datos de propiedad de ciudades bien documentadas (NYC, Philadelphia, Chicago) puede generalizar a ciudades donde sólo existe OSM (Washington DC, San Francisco, Los Ángeles), entonces tendríamos una herramienta universal de análisis urbano.

### Aplicación dual del proyecto

**1. Predicción universal**
Entrenar con ciudades que tienen datos de propiedad como Ground Truth y predecir el uso de suelo en ciudades que sólo disponen de OSM. Esto valida si las señales de OSM son suficientes para clasificar zonas urbanas sin conocimiento previo local.

**2. Detección de transformación urbana**
Comparar la predicción del modelo contra el zoning oficial existente para identificar **zonas en transición**: áreas donde el comportamiento observable (actividad comercial, tipología edilicia) ya no coincide con la clasificación oficial. Estas discrepancias son indicadores potenciales de gentrificación, reconversión industrial o expansión de usos mixtos.

### Pipeline técnico

El pipeline procesa cada ciudad de forma independiente y luego combina todos los datos en un único dataset para el análisis ML:

```
config.py (6 ciudades) → run_pipeline.py → [por ciudad: grid.py → features.py → overpass.py]
→ csv/all_cities_combined.csv → ml_analysis.ipynb (EDA + LR/XGB/RF/SVC/ANN + clustering)
```

---

## 2. Composición del Dataset

### 2.1 Ciudades Seleccionadas

Se eligieron 6 ciudades divididas en dos grupos según la disponibilidad de datos de propiedad. Esta división no es arbitraria: es la base del experimento de transfer learning (Sección 5.3).

| Ciudad | País | Modo | Dataset de propiedad | Razón de inclusión |
|---|---|---|---|---|
| **New York City** | EE.UU. | `property` | PLUTO 25v4 (~900K lotes) | Dataset más completo, baseline principal del proyecto |
| **Philadelphia** | EE.UU. | `property` | OPA (Office of Property Assessment) | Ciudad media norteamericana, contraste con NYC |
| **Chicago** | EE.UU. | `hybrid` | Cook County Assessor | Metrópoli de cuadrícula, diferente morfología urbana |
| **Washington DC** | EE.UU. | `osm` | — (solo OSM) | Capital federal, alta densidad de servicios institucionales |
| **San Francisco** | EE.UU. | `osm` | — (solo OSM) | Ciudad compacta con barrios muy diferenciados |
| **Los Ángeles** | EE.UU. | `osm` | — (solo OSM) | Ciudad dispersa, contraste morfológico extremo |

**Nota sobre el modo `hybrid` (Chicago):** Chicago tiene dataset local de avalúo de propiedad, pero con cobertura incompleta para features de edificación. El modo híbrido usa datos locales para definir la grilla y el zone_type, y completa las features de edificios con datos de OSM cuando faltan valores.

**Nota sobre el modo `osm` (DC, SF, LA):** La frontera de la ciudad se obtiene mediante `osmnx.geocode_to_gdf()`, la grilla se genera por la envolvente convexa de polígonos de uso de suelo de Overpass, y el zone_type se deriva de tags de OSM (`landuse=*`). Sin datos de propiedad, el Ground Truth mismo es aproximado — esto es parte del experimento.

### 2.2 Ciudades Descartadas

No se descartaron ciudades durante la selección. El criterio de inclusión fue: ciudad norteamericana con cobertura razonable de OSM y, para el Grupo A, con un dataset de propiedad descargable públicamente.

Se consideró agregar ciudades europeas (Barcelona, París, Ámsterdam) pero se descartaron por las diferencias en el sistema de uso de suelo y las categorías de zonificación, que harían incompatible el mapping de clases con las ciudades norteamericanas.

### 2.3 Distribución de Clases

La unidad de análisis es una celda de grilla de **150m × 150m**. Cada celda recibe una etiqueta de clase (`zone_type`) derivada del dataset de propiedad (Grupo A) o de OSM landuse (Grupo B).

**Sistema de clases:**
- **Residential** — Predominantemente residencial. En NYC/PLUTO incluye Mixed-Use mapeado a Residential (ver Sección 4.1).
- **Commercial** — Predominantemente comercial o de servicios.
- **Other** — Institucional, espacios abiertos, industrial. Se puede incluir como tercera clase o excluir (ver Sección 4.1).

| Ciudad | Grupo | Celdas totales | Residential | Commercial | Other | Ratio Res:Com |
|---|---|---|---|---|---|---|
| **NYC (Manhattan)** | A | 1,810 | 1,348 | 283 | 179 | 4.8 : 1 |
| **Philadelphia** | A | 9,979 | 8,322 | 945 | 712 | 8.8 : 1 |
| **Chicago** | A | 12,057 | 10,574 | 1,343 | 140 | 7.9 : 1 |
| **Washington DC** | B | 5,017 | 3,994 | 279 | 744 | 14.3 : 1 |
| **San Francisco** | B | 3,876 | 2,832 | 463 | 581 | 6.1 : 1 |
| **Los Ángeles** | B | 27,539 | 20,904 | 2,104 | 4,531 | 9.9 : 1 |
| **TOTAL** | — | 60,278 | 47,974 | 5,417 | 6,887 | 8.9 : 1 |

**Observación sobre el desbalance de clases:** El ratio 4.6:1 en NYC es esperado — Manhattan tiene mucha más superficie residencial que comercial. Para manejar este desbalance, todos los modelos usan `class_weight="balanced"`, que pondera automáticamente la clase minoritaria (Commercial) con mayor importancia durante el entrenamiento.

---

## 3. Feature Engineering

Las features son las "mediciones" que le damos al modelo para que aprenda. Cada feature captura un aspecto diferente del carácter urbano de una celda de 150m × 150m.

### 3.1 Iteración 1: 10 Features Originales (Baseline — NYC solo)

Esta primera iteración se ejecutó únicamente sobre Manhattan (NYC) para establecer un punto de referencia antes de expandir a las 6 ciudades.

**Features utilizadas:**

| Feature | Fuente | Descripción | Hipótesis urbana |
|---|---|---|---|
| `amenity_density` | OSM | Número de amenidades por km² | Zonas comerciales tienen más amenidades |
| `amenity_ratio_food_drink` | OSM | Proporción de restaurantes/cafés sobre total de amenidades | Alta proporción = zona de servicios activa |
| `avg_floors` | PLUTO | Promedio de pisos de edificios en la celda | Edificios altos = mayor intensidad de uso |
| `avg_yearbuilt` | PLUTO | Año de construcción promedio | Edificios más viejos pueden indicar núcleos históricos comerciales |
| `building_count` | PLUTO | Número de edificios en la celda | Densidad edificatoria |
| `total_bldg_area` | PLUTO | Área total construida (m²) en la celda | Masa edilicia total |
| `landuse_entropy` | PLUTO | Entropía de Shannon de los usos de suelo | Valor alto = mezcla de usos = zona de transición |
| `tourism_density` | OSM | Hoteles y POIs turísticos por km² | Zonas turísticas tienden a ser comerciales |
| `shop_density_km2` | OSM | Número de comercios por km² | Indicador directo de actividad comercial |
| `brand_ratio` | OSM | Proporción de comercios con tag `brand=*` | Presencia de franquicias nacionales/internacionales |

**Resultados del Baseline (NYC, Iteración 1):**

| Modelo | Accuracy | Notas |
|---|---|---|
| Logistic Regression (LR) | 82.9% | Modelo lineal, más interpretable |
| XGBoost (XGB) | 89.9% | Gradient boosting, robusto con desbalance |
| Random Forest (RF) | 90.2% | Ensemble de árboles, estable |
| **SVC Polynomial** | **90.5%** | **Mejor modelo del baseline** |

**Resultados del Ablation Study (qué pasa cuando se elimina cada feature):**

El ablation study mide el impacto de cada feature eliminándola y midiendo cuánto cae la accuracy. Permite distinguir features indispensables de features ruidosas.

| Feature eliminada | Cambio en accuracy | Conclusión |
|---|---|---|
| `total_bldg_area` | Mayor caída | Feature más importante del modelo |
| `tourism_density` | Accuracy **mejora** al quitarla | Es ruido — se elimina en Iteración 2 |
| `brand_ratio` | Cambio mínimo (<5% importancia en todos los modelos) | Baja señal — se elimina en Iteración 2 |

**Resultados de Clustering y Reducción Dimensional:**

| Análisis | Resultado |
|---|---|
| K-Means, k óptimo | k = 3 (silhouette = 0.29) |
| PCA, varianza explicada | Necesita 8 componentes para cubrir el 95% de la varianza |
| Mejor normalización | MinMaxScaler (accuracy 0.8406) vs StandardScaler (0.8369) — diferencia marginal |

**Interpretación del k=3 en K-Means:** Aunque el dataset tiene 2 clases etiquetadas (Residential / Commercial), el clustering no supervisado encuentra 3 grupos naturales. Esto sugiere la existencia de un tercer "tipo" urbano sin etiquetar — posiblemente zonas mixtas o de transición que el sistema binario no captura.

### 3.2 Features Eliminadas

Basado en el ablation study del Baseline, se tomaron las siguientes decisiones de curación:

**`tourism_density` — ELIMINADA**
- Evidencia: la accuracy del modelo mejora cuando se elimina esta feature
- Interpretación urbana: en Manhattan, los hoteles y atracciones turísticas están distribuidos tanto en zonas comerciales como residenciales de forma poco diferenciable. No es un indicador discriminante a escala de celda 150m
- Decisión: eliminar en Iteración 2

**`brand_ratio` — ELIMINADA**
- Evidencia: importancia menor al 5% en todos los modelos (LR, XGB, RF, SVC)
- Interpretación urbana: la presencia de franquicias es un proxy demasiado ruidoso. Muchos barrios residenciales tienen supermercados de cadena, y muchas zonas comerciales tienen locales independientes
- Decisión: eliminar en Iteración 2

### 3.3 Features Añadidas (Hipótesis para Iteración 2)

Tras revisar la literatura de morfología urbana y los patrones detectados por K-Means, se proponen 5 features adicionales:

| Feature nueva | Fuente | Descripción | Hipótesis de por qué discrimina |
|---|---|---|---|
| `office_density` | OSM | Oficinas y espacios de trabajo por km² (tag `office=*`) | Indicador directo de actividad corporativa diurna — muy diferente entre zonas residenciales y de negocios |
| `road_density_primary` | OSM | Metros lineales de avenidas principales (`highway=primary/secondary`) por km² | Las zonas comerciales suelen ubicarse sobre ejes viales principales, no en calles residenciales interiores |
| `transit_stop_density` | OSM | Paradas de transporte público por km² | El transporte público de alta frecuencia se concentra en zonas de alta actividad comercial |
| `intersection_density` | OSM | Intersecciones viales por km² | Una malla vial más densa y conectada es característica de centros urbanos comerciales vs tejido residencial de baja conectividad |
| `nightlife_density` | OSM | Bares, discotecas, lugares de entretenimiento nocturno por km² (tag `amenity=bar/nightclub/pub`) | El entretenimiento nocturno es un marcador fuerte de zonas comerciales y mixtas |

**Nota importante:** En la Iteración 1, `intersection_density` y `transit_stop_density` fueron testadas en el pipeline Zone-Finding (a nivel de census tract en Manhattan) y **no mostraron poder discriminante** — las distribuciones eran casi idénticas entre clases. Se añaden aquí de nuevo como hipótesis porque a **escala de celda 150m** el comportamiento puede ser diferente, y porque al incluir ciudades con morfologías distintas (LA vs SF vs NYC), estas features podrían adquirir señal que no tenían en Manhattan solo.

### 3.4 Iteración 2: 13 Features Optimizadas

Composición final propuesta (8 features retenidas + 5 nuevas — 2 eliminadas):

| # | Feature | Estado | Fuente |
|---|---|---|---|
| 1 | `amenity_density` | Retenida | OSM |
| 2 | `amenity_ratio_food_drink` | Retenida | OSM |
| 3 | `avg_floors` | Retenida | PLUTO/dataset local |
| 4 | `avg_yearbuilt` | Retenida | PLUTO/dataset local |
| 5 | `building_count` | Retenida | PLUTO/dataset local |
| 6 | `total_bldg_area` | Retenida | PLUTO/dataset local |
| 7 | `landuse_entropy` | Retenida | PLUTO/dataset local |
| 8 | `shop_density_km2` | Retenida | OSM |
| 9 | `office_density` | **Nueva** | OSM |
| 10 | `road_density_primary` | **Nueva** | OSM |
| 11 | `transit_stop_density` | **Nueva** | OSM |
| 12 | `intersection_density` | **Nueva** | OSM |
| 13 | `nightlife_density` | **Nueva** | OSM |

**Estadísticas de las 13 features sobre el dataset completo (60,278 celdas, 6 ciudades):**

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

**Nota sobre NaN en features de edificación:** Las features derivadas de datos de propiedad (`avg_floors`, `avg_yearbuilt`, `building_count`, `total_bldg_area`) tienen altos porcentajes de NaN porque las ciudades OSM-only (SF, LA) no pueden obtener geometría de edificios completa del Overpass API (límites de memoria del servidor para bboxes grandes). Estos NaN se imputan como 0 durante el entrenamiento ML. Las 9 features OSM-puras tienen 0% NaN.

---

## 4. Categorías de Clasificación

### 4.1 Manejo de Mixed-Use

Una decisión crítica del proyecto es qué hacer con las zonas **Mixed-Use** (uso mixto). En NYC/PLUTO, ~15-20% de las celdas tienen una combinación de uso residencial y comercial que no encaja limpiamente en ninguna clase.

Se probaron dos enfoques:

**Opción A — Clasificación binaria (excluir Mixed-Use)**
- Se eliminan todas las celdas Mixed-Use del dataset
- El modelo aprende a distinguir únicamente Residential "puro" vs Commercial "puro"
- Ventaja: clases más limpias, mayor separabilidad, mejor accuracy
- Desventaja: el modelo no puede predecir zonas mixtas — que son precisamente las más interesantes para detección de transformación urbana

**Opción B — Clasificación 3 clases (incluir Mixed-Use)**
- Mixed-Use se convierte en una tercera clase
- El modelo tiene que aprender a distinguir tres tipos de zona
- Ventaja: más completo, captura la realidad urbana
- Desventaja: la clase Mixed-Use es inherentemente ambigua y difícil de predecir con consistencia. El accuracy baja. El desbalance de clases empeora

**Decisión:** Se comparan ambos enfoques y se documenta la diferencia de accuracy. La elección final depende del objetivo: si la aplicación es **predicción en ciudades sin datos**, Opción A es preferible. Si la aplicación es **detección de transformación**, Opción B es más relevante aunque menos precisa.

En el pipeline actual: `INCLUDE_OTHER = False` por defecto (Opción A). Se puede cambiar a `True` para ejecutar Opción B.

### 4.2 Resultados de la Comparación (Binary vs 3-Class)

Comparación usando Random Forest con cross-validation (3-fold) sobre el dataset combinado de 6 ciudades:

| Configuración | Muestras | Clases | Accuracy (CV) | Std |
|---|---|---|---|---|
| **Binary (sin Mixed-Use)** | **49,848** | **2** | **71.8%** | **±25.0%** |
| 3-Class (con Mixed-Use) | 53,391 | 3 | 57.5% | ±25.0% |

**Diferencia: +14.3 puntos porcentuales a favor de Binary.**

**Interpretación:** La clase Mixed-Use es inherentemente ambigua — sus características observables se solapan significativamente con Residential y Commercial. Incluirla reduce la accuracy en ~14 puntos. Para la aplicación de predicción universal, Binary es claramente superior. La alta varianza (±25%) en ambos casos refleja la heterogeneidad entre ciudades en cross-validation.

**Nota:** Los accuracies de cross-validation son más bajos que los de test set porque el CV incluye folds donde ciudades enteras pueden caer en test — y ciudades OSM-only tienen distribuciones distintas a las de property data.

---

## 5. Modelos

Se utilizan cuatro familias de clasificadores y se comparan sistemáticamente. Todos los modelos usan `class_weight="balanced"` para compensar el desbalance entre clases.

**Modelos evaluados:**
- **Logistic Regression (LR):** Modelo lineal. Sirve como línea base. Si LR alcanza alta accuracy, el problema es linealmente separable. Su coeficiente por feature es directamente interpretable.
- **XGBoost (XGB):** Gradient Boosting. Construye árboles secuencialmente corrigiendo errores del árbol anterior. Robusto con datos heterogéneos y clases desbalanceadas.
- **Random Forest (RF):** Ensemble de árboles de decisión independientes. Estable, menos propenso a overfitting que un árbol único. La importancia de features es fácil de extraer.
- **SVC Polynomial:** Support Vector Machine con kernel polinomial. Captura relaciones no lineales entre features. Fue el mejor modelo en el Baseline.

Adicionalmente, `ml_analysis.ipynb` evalúa:
- **ANN (Red Neuronal Artificial):** Red densa con capas ocultas. Útil para entender si la no-linealidad profunda mejora resultados.
- **K-Means (clustering no supervisado):** No usa las etiquetas — descubre estructura latente en los datos.
- **PCA + t-SNE:** Reducción dimensional para visualizar si las clases son separables en 2D.

### 5.1 Iteración 1: Baseline NYC (10 features, solo Manhattan)

| Modelo | Accuracy | Observaciones |
|---|---|---|
| Logistic Regression | 82.9% | Buen resultado para modelo lineal — sugiere separabilidad parcial lineal |
| XGBoost | 89.9% | Mejora notable sobre LR — hay no-linealidades en los datos |
| Random Forest | 90.2% | Comparable a XGB, más estable en validación cruzada |
| **SVC Polynomial** | **90.5%** | **Mejor resultado** — el kernel polinomial captura interacciones entre features |

**Conclusión de la Iteración 1:** Los cuatro modelos alcanzan accuracy razonable con solo 10 features derivadas de OSM + PLUTO. El problema es clasificable con alta confianza en una sola ciudad. El reto es si este nivel de accuracy se mantiene al generalizar a 6 ciudades con morfologías distintas.

### 5.2 Iteración 2: 6 Ciudades, 13 Features

Entrenamiento combinado con 53,391 celdas de 6 ciudades (80/20 split estratificado). Clasificación binaria: Commercial vs Residential.

**Resultados globales (test set combinado):**

| Modelo | Accuracy | F1 Commercial | F1 Residential |
|---|---|---|---|
| **XGBoost** | **90.98%** | **0.37** | **0.95** |
| ANN (Keras) | 90.20% | 0.13 | 0.95 |
| SVC (poly) | 86.21% | 0.41 | 0.92 |
| Random Forest | 81.28% | 0.44 | 0.89 |
| Logistic Regression | 81.23% | 0.42 | 0.89 |

**Accuracy por ciudad (RF, usando predicciones exportadas):**

| Ciudad | Grupo | Accuracy | Celdas |
|---|---|---|---|
| NYC | Ground Truth | 73.5% | 1,631 |
| Philadelphia | Ground Truth | 86.2% | 9,267 |
| Chicago | Ground Truth | 72.2% | 11,917 |
| DC | OSM-Only | 80.8% | 4,273 |
| SF | OSM-Only | 78.8% | 3,295 |
| LA | OSM-Only | 87.5% | 23,008 |

**Observaciones clave:**
- XGBoost supera al resto por un margen amplio (+4.8% sobre SVC, +9.7% sobre RF)
- En el Baseline (solo Manhattan), SVC poly era el mejor. Con 6 ciudades heterogéneas, XGBoost domina — su capacidad de manejar datos heterogéneos y clases desbalanceadas brilla
- ANN alcanza accuracy similar a XGBoost pero con F1 Commercial muy bajo (0.13) — predice casi todo como Residential
- F1 Commercial bajo en todos los modelos refleja el desbalance extremo (5,417 vs 47,974 celdas)
- Hyperparameter tuning: RF best params `max_depth=None, min_samples_leaf=1, n_estimators=200` → CV accuracy 88.4%. SVC best params `C=0.1, gamma=scale, kernel=poly` → CV accuracy 88.2%

**Ablation Study (RF, 13 features):**

| Feature eliminada | Accuracy sin ella | Impacto |
|---|---|---|
| landuse_entropy | 74.68% | **-7.08%** (más importante) |
| intersection_density | 77.41% | -4.34% |
| amenity_density | 78.63% | -3.13% |
| shop_density_km2 | 78.82% | -2.93% |
| building_count | 79.74% | -2.01% |
| transit_stop_density | 80.72% | -1.03% |
| road_density_primary | 81.01% | -0.74% |
| amenity_ratio_food_drink | 81.28% | -0.47% |
| office_density | 81.66% | -0.09% |
| nightlife_density | 81.79% | +0.04% (ruido) |
| total_bldg_area | 82.12% | +0.37% (ruido) |
| avg_floors | 81.82% | +0.07% (ruido) |
| avg_yearbuilt | 87.78% | **+6.02%** (daña el modelo) |

**Hallazgo del Ablation:** `avg_yearbuilt` tiene un impacto negativo muy fuerte — la accuracy **mejora** un 6% al eliminarlo. Esto se explica porque `avg_yearbuilt` tiene 80.6% de NaN en el dataset (solo NYC, Philadelphia y Chicago parcialmente lo tienen), y los valores imputados como 0 para ciudades OSM-only crean una señal espuria.

### 5.3 Transfer Learning: Ground Truth → OSM-Only

Este es el experimento central del proyecto. Se entrena el modelo **únicamente** con las 3 ciudades del Grupo A (NYC, Philadelphia, Chicago) y se evalúa su accuracy en las 3 ciudades del Grupo B (DC, SF, LA), donde el Ground Truth es también OSM-derivado.

La pregunta que responde: **¿Las señales de OSM que distinguen Commercial de Residential en NYC son universales o son específicas de Nueva York?**

**Resultados (RF entrenado con Grupo A, evaluado en Grupo B):**

| Entrenado con | Evaluado en | Accuracy | Celdas | Interpretación |
|---|---|---|---|---|
| NYC + PHL + CHI | **Test set (GT)** | **88.9%** | 22,815 | Baseline de referencia |
| NYC + PHL + CHI | DC | 91.7% | 4,273 | Excelente transferencia |
| NYC + PHL + CHI | LA | 86.7% | 23,008 | Buena transferencia |
| NYC + PHL + CHI | SF | 80.6% | 3,295 | Aceptable |
| NYC + PHL + CHI | **Promedio OSM** | **86.7%** | 30,576 | **Solo 2.2% debajo de GT** |

**Per-city accuracy (todas las ciudades):**

| Ciudad | Accuracy | Celdas | Grupo |
|---|---|---|---|
| NYC | 98.3% | 1,631 | Ground Truth |
| Philadelphia | 98.1% | 9,267 | Ground Truth |
| Chicago | 97.1% | 11,917 | Ground Truth |
| DC | 91.7% | 4,273 | OSM-Only |
| LA | 86.7% | 23,008 | OSM-Only |
| SF | 80.6% | 3,295 | OSM-Only |

**Conclusión:** OSM-Only accuracy = 86.7% (>80%) → **OSM es suficiente para predecir zonas urbanas.**

El delta entre Ground Truth (88.9%) y OSM-Only (86.7%) es solo 2.2 puntos porcentuales — remarkablemente bajo. Las señales urbanas de OSM (densidad de amenidades, entropía de uso de suelo, densidad de intersecciones, comercios, oficinas, transporte) generalizan bien entre ciudades con morfologías distintas.

**Diferencias entre ciudades OSM-Only:**
- DC (91.7%): La más compacta y urbanamente densa de las tres — morfología similar a las ciudades de training
- LA (86.7%): A pesar de ser dispersa y sprawl-heavy, alcanza buena accuracy — la señal de uso de suelo es fuerte
- SF (80.6%): La más baja pero aún aceptable — posiblemente por la falta de datos de buildings en OSM (0 buildings encontrados por Overpass) y menor cobertura de carreteras (0 road ways)

---

## 6. Decisiones Clave (para la presentación)

Este log documenta las decisiones curatoriales tomadas durante el proceso. Cada decisión tiene una evidencia que la justifica — esto es lo que diferencia "experimentos" de "prueba y error".

1. **Elegimos 6 ciudades divididas en 2 grupos** (3 con datos de propiedad + 3 solo OSM) para estructurar un experimento de transfer learning: ¿puede un modelo entrenado con Ground Truth preciso generalizar a ciudades sin él? Sin esta división, el proyecto no tiene hipótesis testable.

2. **Usamos una grilla regular de 150m × 150m** en lugar de census tracts o polígonos de barrio. Razón: los census tracts son irregulares, varían de tamaño entre ciudades y no existen en todas partes. Una grilla regular es comparable entre ciudades, reproducible, y geométricamente neutral.

3. **Eliminamos `tourism_density`** porque el ablation study demostró que la accuracy del modelo **mejora** al quitarla. Esto es evidencia directa de que es ruido en el modelo, no una feature informativa. Interpretación urbana: en Manhattan, los hoteles se distribuyen de forma demasiado uniforme para ser discriminantes a escala de 150m.

4. **Eliminamos `brand_ratio`** por importancia consistentemente baja (<5%) en todos los modelos. El ratio de franquicias no captura diferencias de uso de suelo de forma confiable — una farmacia CVS puede estar en un barrio residencial igual que en un corredor comercial.

5. **Añadimos 5 features nuevas** (office_density, road_density_primary, transit_stop_density, intersection_density, nightlife_density) basadas en hipótesis de morfología urbana. Algunas de estas (intersection_density, transit) ya fueron testadas a nivel de census tract y no mostraron señal — se re-testean aquí porque la escala 150m y la diversidad de 6 ciudades pueden cambiar el resultado.

6. **Comparamos clasificación binaria vs 3 clases** para decidir el manejo de Mixed-Use. Esta comparación no es un detalle técnico — es una decisión sobre qué tipo de pregunta queremos responder. Binary = predicción limpia. 3-Class = captura realidad urbana compleja pero con mayor incertidumbre.

7. **Usamos `class_weight="balanced"` en todos los modelos** porque el desbalance de clases (4.6:1 Residential:Commercial en NYC) haría que un modelo naive que predijera siempre "Residential" alcanzara ~82% de accuracy sin aprender nada. Balancear los pesos fuerza al modelo a aprender la clase minoritaria.

8. **K-Means encontró k=3 grupos naturales** aunque el dataset tiene 2 clases etiquetadas. Esto sugiere que la clasificación binaria puede estar simplificando demasiado la realidad urbana — hay un tercer "tipo" emergente en los datos que el etiquetado oficial no captura. Esta es una de las observaciones más interesantes del proyecto para la presentación.

---

## 7. Plots Clave para la Presentación

Los siguientes visuales están producidos por `ml_analysis.ipynb` y son los más relevantes para comunicar los hallazgos del proyecto:

### Plots de contexto (establecen el problema)

| Plot | Archivo esperado | Qué demuestra |
|---|---|---|
| Mapa de grilla de ciudades | `heatmap_[ciudad].png` | Escala y distribución geográfica del dataset — el problema tiene dimensión espacial |
| Distribución de clases por ciudad | (bar chart en EDA) | El desbalance varía por ciudad — NYC no es representativa de todas |

### Plots del Baseline (Iteración 1)

| Plot | Archivo esperado | Qué demuestra |
|---|---|---|
| Feature importance (RF + XGB) | (en ml_analysis) | `total_bldg_area` domina — la masa edilicia es la señal más fuerte |
| Ablation study | (barras de accuracy por feature removida) | Evidencia de por qué se eliminó `tourism_density` y `brand_ratio` |
| Matriz de confusión (SVC, Iteración 1) | (en ml_analysis) | Dónde se equivoca el mejor modelo — qué tipo de celdas son difíciles de clasificar |

### Plots de comparación multi-ciudad (Iteración 2)

| Plot | Archivo esperado | Qué demuestra |
|---|---|---|
| Accuracy por ciudad y por modelo | `comparison_accuracy.png` | ¿Algunas ciudades son más predecibles que otras? ¿Por qué? |
| Feature importance normalizada por ciudad | (heatmap de features) | ¿La misma feature importa igual en NYC que en LA? |
| Mapa heatmap combinado (6 ciudades) | (folium o matplotlib) | La pregunta de investigación visualizada en el espacio |

### Plots de clustering y dimensionalidad

| Plot | Archivo esperado | Qué demuestra |
|---|---|---|
| t-SNE coloreado por clase | (en ml_analysis) | ¿Son visualmente separables las clases en 2D? ¿Hay solapamiento? |
| t-SNE coloreado por ciudad | (en ml_analysis) | ¿Cada ciudad forma su propio cluster o se mezclan? Si NYC y SF están mezcladas, los patrones son similares |
| K-Means k=3 sobre t-SNE | (en ml_analysis) | El tercer cluster emergente — ¿es consistente con Mixed-Use? |
| Scree plot PCA | (en ml_analysis) | Requiere 8 componentes para 95% varianza — el problema es genuinamente multidimensional |

### Plot central del proyecto (Transfer Learning)

| Plot | Archivo esperado | Qué demuestra |
|---|---|---|
| Accuracy: entrenado en Grupo A, evaluado en Grupo B | (tabla + barras) | La respuesta a la pregunta de investigación: ¿generaliza OSM? |
| Mapa de predicciones en DC/SF/LA | (folium por ciudad) | Las predicciones en ciudades sin Ground Truth — esto es lo que el proyecto habilita |

---

*Documento completado. Pipeline ejecutado con 6 ciudades (60,278 celdas), ML notebook corrido con todos los modelos y visualizaciones. Resultados de Transfer Learning confirman la hipótesis principal: OSM es suficiente para predicción universal de zonas urbanas (86.7% accuracy, solo 2.2% debajo del Ground Truth).*
