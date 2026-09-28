# PySpatialStats Tutorial

A progressive guide from first fit to advanced spatial modelling workflows.

**Audience:** GIS analysts, spatial econometricians, and ML practitioners new to geographically weighted (GW) methods.

**Prerequisites:** Python 3.9+, basic NumPy/pandas, and familiarity with regression concepts.

---

## Table of Contents

1. [What is PySpatialStats?](#1-what-is-pyspatialstats)
2. [Installation](#2-installation)
3. [Core Concepts](#3-core-concepts)
4. [Level 1 — Getting Started (Basic)](#4-level-1--getting-started-basic)
5. [Level 2 — Classical GW Models (Intermediate)](#5-level-2--classical-gw-models-intermediate)
6. [Level 3 — Machine Learning GW Models (Intermediate+)](#6-level-3--machine-learning-gw-models-intermediate)
7. [Level 4 — Spatiotemporal & Multivariate (Advanced)](#7-level-4--spatiotemporal--multivariate-advanced)
8. [Level 5 — Deep Learning & Network Models (Advanced)](#8-level-5--deep-learning--network-models-advanced)
9. [Diagnostics & Model Checking](#9-diagnostics--model-checking)
10. [Visualization](#10-visualization)
11. [Performance: Multi-core CPU & GPU](#11-performance-multi-core-cpu--gpu)
12. [End-to-End Workflows](#12-end-to-end-workflows)
13. [Choosing the Right Model](#13-choosing-the-right-model)
14. [Troubleshooting](#14-troubleshooting)
15. [Further Reading](#15-further-reading)

---

## 1. What is PySpatialStats?

PySpatialStats is a Python package for **geographically weighted modelling** — a family of methods where statistical relationships are allowed to vary across space (and time).

| Paradigm | PySpatialStats classes | When to use |
|---|---|---|
| Classical GW statistics | `GWR`, `MGWR`, `GWSS`, `GWGLM` | Interpretable local coefficients, inference |
| GW machine learning | `GWRandomForest`, `GWXGBoost`, `GWLightGBM` | Non-linear relationships, prediction |
| GW deep learning | `GNNWR`, `GTNNWR` | Complex non-stationarity, learned weights |
| Multivariate / network | `GWPCA`, `NetworkGWR` | Dimension reduction, network-constrained space |

**Package layout:**

```
spatialstats/gwmodel/
├── core/           # Kernels, bandwidth selection, base classes
├── linear/         # GWR, MGWR, GWGLM, GWSS, MixedGWR
├── spatiotemporal/ # GTWR
├── ensemble/       # GW Random Forest, XGBoost, LightGBM
├── deep/           # GNNWR, GTNNWR (PyTorch)
├── multivariate/   # GWPCA
├── network/        # NetworkGWR
├── diagnostics/    # Moran's I, F3 test, collinearity
├── visualization/  # Coefficient & importance maps
├── datasets/       # Example data loaders
└── utils/          # Distance, stats, parallel, compute config
```

---

## 2. Installation

### Basic (core only)

```bash
pip install pyspatialstats
# or from the local wheel:
pip install pyspatialstats-0.1.0-py3-none-any.whl
```

Core dependencies: `numpy`, `scipy`, `pandas`, `geopandas`, `scikit-learn`, `shapely`, `joblib`.

### Optional extras

```bash
pip install pyspatialstats[deep]       # PyTorch → GNNWR, GTNNWR
pip install pyspatialstats[boosting]   # XGBoost, LightGBM
pip install pyspatialstats[viz]        # matplotlib, folium
pip install pyspatialstats[network]    # networkx
pip install pyspatialstats[all]      # everything
```

### Verify installation

```python
import spatialstats
import spatialstats.gwmodel as gwmodel
print(spatialstats.__version__)   # 0.1.0

from spatialstats.gwmodel.linear import GWR
from spatialstats.gwmodel.datasets import load_georgia
print("PySpatialStats ready.")
```

---

## 3. Core Concepts

Before fitting models, understand three ideas that appear in every GW class.

### 3.1 Spatial coordinates (`geometry`)

All models accept coordinates as:

- A NumPy array of shape `(n, 2)` — columns are `x` and `y` (or lon/lat)
- A GeoPandas `GeoSeries` or `GeoDataFrame`

```python
import numpy as np

coords = np.column_stack([gdf["x"], gdf["y"]])   # ndarray
# or pass gdf directly — geometry is parsed automatically
```

### 3.2 Kernels and bandwidth

A **kernel** assigns a weight to each observation based on its distance from a focal point. A **bandwidth** controls how far influence extends.

| Setting | Meaning |
|---|---|
| `fixed=True`, `bandwidth=2.0` | Fixed distance threshold (e.g. 2 km) |
| `fixed=False`, `bandwidth=50` | Adaptive: use 50 nearest neighbours |
| `kernel="bisquare"` | Compact support — weights go to zero beyond bandwidth |
| `kernel="gaussian"` | Smooth decay, no hard cutoff |
| `bandwidth="auto"` | Selected by AICc, AIC, BIC, or CV |

```python
from spatialstats.gwmodel.core.kernels import SpatialKernel

kern = SpatialKernel("bisquare", fixed=True, bandwidth=10.0)
distances = np.array([0, 3, 7, 12])
weights = kern(distances)   # [1.0, 0.64, 0.16, 0.0]
```

### 3.3 The fit → predict pattern

Every estimator follows scikit-learn conventions:

```python
model = SomeGWModel(...)
model.fit(X, y, geometry)       # train
yhat = model.predict(X, geometry) # predict at same or new locations
score = model.score(X, y, geometry)  # R² (regression)
```

---

## 4. Level 1 — Getting Started (Basic)

**Goal:** Run your first GWR model in under 10 lines.

### 4.1 Load example data

```python
from spatialstats.gwmodel.datasets import load_georgia

gdf, features, target = load_georgia()
coords = gdf[["x", "y"]].values
X = gdf[features].values
y = gdf[target].values

print(X.shape, y.shape, coords.shape)   # (159, 6) (159,) (159, 2)
```

Other bundled datasets:

| Loader | Domain | Target |
|---|---|---|
| `load_georgia()` | US counties | `TotPop90` |
| `load_londonhouse()` | London housing | `Price` |
| `load_beijing_pm25()` | Air quality | `PM25` |
| `load_ncovid()` | COVID cases | `Cases` |

### 4.2 Your first GWR model

```python
from spatialstats.gwmodel.linear import GWR

gwr = GWR(
    bandwidth=2.0,      # fixed distance (degrees, for this dataset)
    fixed=True,
    kernel="bisquare",
)
gwr.fit(X, y, coords)

print(gwr.summary())
print("R²:", gwr.score(X, y, coords))
```

### 4.3 Read the output

After `fit()`, key attributes are:

| Attribute | Shape | Description |
|---|---|---|
| `coef_` | `(n, p+1)` | Local regression coefficients (incl. intercept) |
| `std_err_` | `(n, p+1)` | Standard errors |
| `t_values_` | `(n, p+1)` | t-statistics |
| `p_values_` | `(n, p+1)` | p-values (Benjamini–Hochberg corrected) |
| `fitted_` | `(n,)` | In-sample fitted values |
| `residuals_` | `(n,)` | Residuals |
| `aicc_` | scalar | Corrected AIC |
| `bandwidth_` | scalar | Bandwidth used (or selected) |

### 4.4 Predict at new locations

```python
# Hold out the last 20 observations
X_train, X_test = X[:-20], X[-20:]
y_train, y_test = y[:-20], y[-20:]
coords_train, coords_test = coords[:-20], coords[-20:]

gwr.fit(X_train, y_train, coords_train)
y_pred = gwr.predict(X_test, coords_test)
```

### 4.5 Use your own data

```python
import pandas as pd
import geopandas as gpd

# From a CSV with x, y columns
df = pd.read_csv("my_data.csv")
coords = df[["lon", "lat"]].values
X = df[["income", "age", "density"]].values
y = df["price"].values

gwr = GWR(bandwidth="auto", fixed=False, kernel="bisquare")
gwr.fit(X, y, coords)
```

---

## 5. Level 2 — Classical GW Models (Intermediate)

**Goal:** Choose bandwidths wisely, model multiple scales, and handle non-Gaussian outcomes.

### 5.1 Automatic bandwidth selection

```python
gwr = GWR(
    bandwidth="auto",   # golden-section search
    fixed=False,        # adaptive k-NN
    kernel="bisquare",
    criterion="AICc",   # or "AIC", "BIC", "CV"
    n_jobs=-1,          # use all CPU cores
)
gwr.fit(X, y, coords)
print("Selected bandwidth:", gwr.bandwidth_)
```

### 5.2 Kernel comparison

```python
for kernel in ["bisquare", "gaussian", "exponential", "tricube"]:
    m = GWR(bandwidth="auto", fixed=False, kernel=kernel, n_jobs=-1)
    m.fit(X, y, coords)
    print(f"{kernel:12s}  AICc={m.aicc_:.1f}  R²={m.score(X, y, coords):.3f}")
```

### 5.3 Multiscale GWR (MGWR)

When different predictors operate at different spatial scales, use **MGWR** — each covariate gets its own bandwidth via back-fitting.

```python
from spatialstats.gwmodel.linear import MGWR

mgwr = MGWR(
    kernel="bisquare",
    fixed=False,
    max_iter=20,
    tol=1.0,
    n_jobs=-1,
)
mgwr.fit(X, y, coords)

print("Per-covariate bandwidths:", mgwr.bandwidths_)
print(mgwr.summary())
```

**Interpretation:** A large bandwidth for covariate `j` means that predictor's effect is relatively stationary; a small bandwidth means it varies sharply over space.

### 5.4 Geographically Weighted Summary Statistics (GWSS)

Explore local descriptive statistics before modelling.

```python
from spatialstats.gwmodel.linear import GWSS

gwss = GWSS(bandwidth=25, fixed=True, quantile=True)
gwss.fit(X, coords, var_names=features)

print("Local means:\n", gwss.local_mean_[:3])
print("Local std:\n",  gwss.local_std_[:3])

# Export to DataFrame
df_stats = gwss.to_dataframe()
```

### 5.5 Generalised Linear Models (GWGLM)

For count, rate, or binary outcomes:

```python
from spatialstats.gwmodel.linear import GWGLM

# Poisson regression (counts)
gwglm = GWGLM(family="poisson", bandwidth=30, fixed=True)
gwglm.fit(X_counts, y_counts, coords)

# Binomial regression (0/1 outcomes)
gwglm_bin = GWGLM(family="binomial", bandwidth=30, fixed=True)
gwglm_bin.fit(X_bin, y_binary, coords)

# Gaussian (equivalent to GWR)
gwglm_g = GWGLM(family="gaussian", bandwidth="auto", fixed=False)
gwglm_g.fit(X, y, coords)
```

### 5.6 Mixed (semiparametric) GWR

Fix some coefficients globally while letting others vary locally.

```python
from spatialstats.gwmodel.linear import MixedGWR

# Covariate index 0 is global; indices 1, 2 vary spatially
mixed = MixedGWR(
    fixed_vars=[0],
    varying_vars=[1, 2],
    bandwidth=20,
    fixed=True,
)
mixed.fit(X, y, coords)

print("Global coefficients:", mixed.global_coef_)
print("Local coefficients shape:", mixed.coef_.shape)
```

---

## 6. Level 3 — Machine Learning GW Models (Intermediate+)

**Goal:** Capture non-linear spatial relationships with ensemble methods.

### 6.1 GW Random Forest

Combines a **global** Random Forest with **local** forests weighted by spatial kernel.

```python
from spatialstats.gwmodel.ensemble import GWRandomForest

gwrf = GWRandomForest(
    n_estimators=100,
    bandwidth=2.0,
    fixed=True,
    kernel="bisquare",
    train_weighted=True,    # weight training samples by kernel
    predict_weighted=True,  # blend global + local predictions
    n_jobs=-1,
    random_state=42,
)
gwrf.fit(X, y, coords)

preds = gwrf.predict(X, coords)
importance_map = gwrf.feature_importance_map()   # shape (n, p)
```

**Auto bandwidth via Moran's I:**

```python
gwrf_auto = GWRandomForest(
    n_estimators=50,
    bandwidth="auto",   # theory-informed Moran's I selection
    fixed=True,
)
gwrf_auto.fit(X, y, coords)
print("Auto bandwidth:", gwrf_auto.bandwidth_)
```

### 6.2 GW XGBoost

```python
from spatialstats.gwmodel.ensemble import GWXGBoost

gwxgb = GWXGBoost(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    bandwidth=25,
    fixed=True,
    n_jobs=-1,
)
gwxgb.fit(X, y, coords)
print(gwxgb.predict(X, coords).shape)
```

### 6.3 GW LightGBM

```python
from spatialstats.gwmodel.ensemble import GWLightGBM

gwlgb = GWLightGBM(
    n_estimators=100,
    num_leaves=31,
    bandwidth=25,
    fixed=True,
    n_jobs=-1,
)
gwlgb.fit(X, y, coords)
```

### 6.4 When to prefer ML over GWR

| Situation | Prefer |
|---|---|
| Need interpretable local coefficients | `GWR` / `MGWR` |
| Strong non-linearity, interactions | `GWRandomForest`, `GWXGBoost` |
| Large `n`, prediction-focused | `GWXGBoost` or `GWLightGBM` |
| Formal inference (p-values) | `GWR` / `MGWR` |

---

## 7. Level 4 — Spatiotemporal & Multivariate (Advanced)

**Goal:** Model processes that vary in both space and time, or reduce multivariate dimensionality locally.

### 7.1 Geographically and Temporally Weighted Regression (GTWR)

```python
from spatialstats.gwmodel.spatiotemporal import GTWR
import numpy as np

n = len(y)
times = np.arange(n)   # or datetime ordinals, epoch seconds, etc.

gtwr = GTWR(
    bandwidth_s=20,     # spatial bandwidth
    lambda_=1.0,        # space-time trade-off (higher → more temporal weight)
    fixed=False,
    kernel="bisquare",
    n_jobs=-1,
)
gtwr.fit(X, y, coords, times)
print("AICc:", gtwr.aicc_)
preds = gtwr.predict(X, coords, times)
```

**Choosing `lambda_`:** Start with `lambda_=1.0` (equal weighting of normalised space and time distances). Increase to emphasise temporal proximity; decrease to emphasise spatial proximity.

### 7.2 Geographically Weighted PCA (GWPCA)

Discover locally varying principal components.

```python
from spatialstats.gwmodel.multivariate import GWPCA

gwpca = GWPCA(
    n_components=2,
    bandwidth=20,
    fixed=True,
    robust=False,   # set True for outlier-resistant local covariances
)
gwpca.fit(X, coords)

print("Local loadings:", gwpca.local_loadings_.shape)    # (n, p, k)
print("Local scores:",   gwpca.local_scores_.shape)      # (n, k)
print("Variance explained:", gwpca.local_pve_[:3])
```

**Robust mode** uses median-based local covariances — useful when outliers contaminate local neighbourhoods.

---

## 8. Level 5 — Deep Learning & Network Models (Advanced)

**Goal:** Learn spatial weight functions with neural networks, or constrain distance along a network.

### 8.1 GNNWR — Neural Network Weighted Regression

GNNWR replaces fixed kernel functions with a **Spatially Weighted Neural Network (SWNN)** that learns weights from proximity features.

```python
from spatialstats.gwmodel.deep import GNNWR

gnnwr = GNNWR(
    hidden_layers=[64, 32, 16],
    n_epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    patience=20,        # early stopping
    device="auto",      # cuda → mps → cpu
    random_state=42,
)
gnnwr.fit(X, y, coords)

preds = gnnwr.predict(X, coords)
print("Training loss history:", gnnwr.train_loss_[:5], "...")
print("Local coefficients:", gnnwr.coef_.shape)
```

**Hyperparameter tips:**

| Parameter | Guidance |
|---|---|
| `hidden_layers` | Start `[32, 16]`; increase for complex surfaces |
| `n_epochs` | 50–200; watch `val_loss_` for overfitting |
| `batch_size` | 32–128; smaller for small `n` |
| `device` | `"cuda"` on NVIDIA GPU; `"auto"` recommended |

### 8.2 GTNNWR — Spatiotemporal Neural Network WR

```python
from spatialstats.gwmodel.deep import GTNNWR

gtnnwr = GTNNWR(
    spatial_layers=[32, 16],
    temporal_layers=[16, 8],
    fusion_layers=[32, 16],
    n_epochs=50,
    device="auto",
    random_state=42,
)
gtnnwr.fit(X, y, coords, times)
preds = gtnnwr.predict(X, coords, times)
```

### 8.3 Network GWR

Use shortest-path distance on a graph instead of Euclidean distance.

```python
import networkx as nx
from spatialstats.gwmodel.network import NetworkGWR

# Build a grid network
G = nx.grid_2d_graph(10, 10)
for u, v in G.edges():
    G[u][v]["weight"] = 1.0

net_gwr = NetworkGWR(
    network=G,
    bandwidth=5,
    fixed=False,
    kernel="bisquare",
    n_jobs=-1,
)
net_gwr.fit(X, y, coords)
```

Requires: `pip install pyspatialstats[network]`

---

## 9. Diagnostics & Model Checking

Always validate GW models — local fitting can mask global misspecification.

### 9.1 Residual spatial autocorrelation (Moran's I)

```python
from spatialstats.gwmodel.diagnostics import moran_test

result = moran_test(gwr.residuals_, coords, k=8)
print("Moran's I:", result["I"])
print("p-value:",   result["p_value"])
# p < 0.05 → residuals still spatially autocorrelated; reconsider bandwidth/model
```

### 9.2 F3 test for coefficient non-stationarity

```python
from spatialstats.gwmodel.diagnostics import f3_test
import numpy as np

hat_diag = np.full(len(y), gwr.effective_df_ / len(y))
f3 = f3_test(gwr.coef_, gwr.residuals_, hat_diag)
for j, (stat, p) in enumerate(zip(f3["statistic"], f3["p_value"])):
    print(f"Covariate {j}: F3={stat:.3f}, p={p:.4f}")
```

### 9.3 Monte Carlo variability test

```python
from spatialstats.gwmodel.diagnostics import monte_carlo_variability

mc = monte_carlo_variability(gwr.coef_, n_sim=499)
print("p-values:", mc["p_value"])
```

### 9.4 Local collinearity (VIF & condition number)

```python
from spatialstats.gwmodel.diagnostics import local_vif, local_condition_number
from spatialstats.gwmodel.core.kernels import SpatialKernel
from spatialstats.gwmodel.utils.distance import pairwise_distances

X_int = np.column_stack([np.ones(len(y)), X])
D = pairwise_distances(coords)
kernel = SpatialKernel("bisquare", fixed=True, bandwidth=gwr.bandwidth_)

vif = local_vif(X_int, D, kernel)          # (n, p) — VIF > 10 is concerning
cn  = local_condition_number(gwr.coef_, X_int, D, kernel)  # (n,)
```

### 9.5 Local significance masking

```python
sig = gwr.local_significance(alpha=0.05)
# Boolean array (n, p) — True where coefficient is significant
```

---

## 10. Visualization

Requires: `pip install pyspatialstats[viz]` (matplotlib).

### 10.1 Local coefficient maps

```python
from spatialstats.gwmodel.visualization import plot_local_coefficients

fig = plot_local_coefficients(
    gwr, gdf,
    covariate=None,       # None = all covariates
    significance=True,
    alpha=0.05,
    cmap="RdBu_r",
    save_path="coef_maps.png",
)
```

### 10.2 Feature importance maps (ensemble models)

```python
from spatialstats.gwmodel.visualization import plot_local_importance

fig = plot_local_importance(
    gwrf, gdf,
    feature=None,          # None = all features
    importance_type="gain",
)
```

### 10.3 Residual maps

```python
from spatialstats.gwmodel.visualization import plot_residuals

fig = plot_residuals(gwr, gdf, save_path="residuals.png")
```

---

## 11. Performance: Multi-core CPU & GPU

Heavy GW models fit one local model per observation — parallelism matters.

### 11.1 Global compute configuration

```python
from spatialstats import set_compute_options, GWR, GNNWR, GWXGBoost

# All CPU cores + auto GPU detection
set_compute_options(n_jobs=-1, device="auto")

# Process-based parallelism (good for large GWR on many cores)
set_compute_options(n_jobs=8, parallel_backend="processes")

# Full GPU stack (NVIDIA CUDA)
set_compute_options(
    n_jobs=-1,
    device="cuda",
    xgboost_device="cuda",
    lightgbm_device="gpu",
)
```

### 11.2 Per-model overrides

```python
gwr = GWR(bandwidth="auto", n_jobs=4)          # 4 cores for this fit only
gnnwr = GNNWR(hidden_layers=[64, 32], device="cuda")
```

### 11.3 Verify your environment

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from spatialstats.gwmodel.utils.compute import resolve_torch_device
print("PySpatialStats will use:", resolve_torch_device("auto"))
```

> **Note:** Use the same Python environment for PyTorch and PySpatialStats. A CPU-only PyTorch build will not use your GPU even if CUDA drivers are installed.

### 11.4 Scaling guidelines

| `n` (observations) | Recommended settings |
|---|---|
| < 200 | Default settings, `n_jobs=2` sufficient |
| 200–2 000 | `n_jobs=-1`; consider `bandwidth="auto"` overnight for MGWR |
| 2 000–10 000 | `parallel_backend="processes"`; subsample for bandwidth search |
| > 10 000 | Use `GWRandomForest` / `GWXGBoost`; GWR may be impractical |

---

## 12. End-to-End Workflows

### 12.1 Exploratory spatial analysis → GWR → diagnostics → maps

```python
from spatialstats.gwmodel.datasets import load_londonhouse
from spatialstats.gwmodel.linear import GWSS, GWR
from spatialstats.gwmodel.diagnostics import moran_test
from spatialstats.gwmodel.visualization import plot_local_coefficients, plot_residuals

# 1. Load
gdf, features, target = load_londonhouse()
coords = gdf[["x", "y"]].values
X, y = gdf[features].values, gdf[target].values

# 2. Explore
gwss = GWSS(bandwidth=0.05, fixed=True, quantile=True)
gwss.fit(X, coords, var_names=features)

# 3. Model
gwr = GWR(bandwidth="auto", fixed=False, kernel="bisquare", n_jobs=-1)
gwr.fit(X, y, coords)

# 4. Diagnose
moran = moran_test(gwr.residuals_, coords)
print(f"Moran's I p-value: {moran['p_value']:.4f}")

# 5. Visualise
plot_local_coefficients(gwr, gdf, save_path="london_coefs.png")
plot_residuals(gwr, gdf, save_path="london_residuals.png")
```

### 12.2 Compare model families

```python
from spatialstats.gwmodel.linear import GWR, MGWR
from spatialstats.gwmodel.ensemble import GWRandomForest, GWXGBoost
from spatialstats.gwmodel.deep import GNNWR

models = {
    "GWR":        GWR(bandwidth="auto", fixed=False, n_jobs=-1),
    "MGWR":       MGWR(fixed=False, max_iter=10, n_jobs=-1),
    "GW-RF":      GWRandomForest(n_estimators=50, bandwidth="auto", n_jobs=-1),
    "GW-XGB":     GWXGBoost(n_estimators=50, bandwidth=25, fixed=True, n_jobs=-1),
    "GNNWR":      GNNWR(hidden_layers=[32, 16], n_epochs=30, device="auto"),
}

for name, model in models.items():
    model.fit(X, y, coords)
    r2 = model.score(X, y, coords)
    print(f"{name:10s}  R² = {r2:.4f}")
```

### 12.3 Train / validation split with spatial awareness

```python
# Simple random split (for illustration — consider spatial CV in practice)
idx = np.random.permutation(len(y))
train_idx, val_idx = idx[:400], idx[400:]

gwr = GWR(bandwidth="auto", fixed=False, n_jobs=-1)
gwr.fit(X[train_idx], y[train_idx], coords[train_idx])

y_val_pred = gwr.predict(X[val_idx], coords[val_idx])
val_r2 = 1 - np.sum((y[val_idx] - y_val_pred)**2) / np.sum((y[val_idx] - y[val_idx].mean())**2)
print(f"Validation R²: {val_r2:.4f}")
```

---

## 13. Choosing the Right Model

```
                    ┌─────────────────────────────────────┐
                    │     Is the outcome continuous?      │
                    └──────────────┬──────────────────────┘
                          yes      │      no
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         Need local coefficients?          Use GWGLM
         (inference / policy)              (poisson / binomial)
                    │
          yes ──────┴────── no
           │                │
           ▼                ▼
    Linear relationship?   Use GWRandomForest
           │               or GWXGBoost
     yes ──┴── no
      │        │
      ▼        ▼
    GWR/     GNNWR
    MGWR     (deep, GPU)
```

| Research question | Recommended model |
|---|---|
| How does the effect of X on Y vary across space? | `GWR`, `MGWR` |
| Do predictors operate at different scales? | `MGWR` |
| Is the relationship non-linear? | `GWRandomForest`, `GWXGBoost` |
| Space + time interaction? | `GTWR`, `GTNNWR` |
| Local multivariate structure? | `GWPCA` |
| Movement constrained by a network? | `NetworkGWR` |
| Some covariates global, others local? | `MixedGWR` |

---

## 14. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `LinAlgError` during fit | Local collinearity | Reduce `p`; increase bandwidth; check VIF |
| All p-values = 1 | Bandwidth too large | Decrease bandwidth; use adaptive kernel |
| Very slow fit | Large `n`, GWR | Use `n_jobs=-1`; try `GWRandomForest` instead |
| `ImportError: PyTorch required` | Missing deep extra | `pip install pyspatialstats[deep]` |
| GPU not used | CPU-only PyTorch | Install CUDA PyTorch matching your driver |
| `ImportError: xgboost` | Missing boosting extra | `pip install pyspatialstats[boosting]` |
| Negative local R² | Small local sample / bad bandwidth | Increase adaptive bandwidth |
| MGWR won't converge | `tol` too strict | Increase `tol`; reduce `max_iter`; set `bw_init` |

### Run the test suite

```bash
python tests_spatialstats.gwmodel.py
# Expected: 52 passed, 0 failed
```

---

## 15. Further Reading

### Key references

- Fotheringham, Brunsdon & Charlton (2002). *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships.* Wiley.
- Oshan et al. (2019). mgwr: A Python Implementation of Multiscale GWR. *ISPRS IJGI*, 8(6), 269.
- Sun & Hu (2024). PyGRF: An Improved Python Package for Geographically Weighted Random Forest. *Transactions in GIS*, 28(7), 2476–2491.
- Du et al. (2020). Geographically Neural Network Weighted Regression. *IJGIS*, 34(7), 1353–1377.
- Wu et al. (2021). Geographically and Temporally Neural Network Weighted Regression. *IJGIS*, 35(3), 582–608.

### Suggested learning path

| Week | Focus | Exercises |
|---|---|---|
| 1 | Sections 1–4 | Fit GWR on `load_georgia()`; plot coefficients |
| 2 | Section 5 | Compare kernels; run MGWR; try GWGLM Poisson |
| 3 | Section 6 | GW-RF on `load_londonhouse()`; compare R² with GWR |
| 4 | Sections 7–8 | GTWR with synthetic times; GNNWR with GPU |
| 5 | Sections 9–12 | Full diagnostic workflow; write a model comparison notebook |

---

*PySpatialStats v0.1.0 — BSD-3-Clause License*
