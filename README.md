<p align="left">
  <img src="docs/images/package_logo.png" alt="PySpatialStats logo" width="110"/>
</p>

# PySpatialStats

**A comprehensive Python package for spatial statistics and geographically weighted modeling**

PySpatialStats provides a unified framework for:

- **Core** spatial data containers, CRS helpers, and distance utilities
- **Spatial weights** (contiguity, distance, graph) with libpysal interoperability
- **ESDA** — Moran's I, Geary's C, Getis-Ord G, LISA, join counts
- **Point pattern analysis** — intensity / KDE, Clark–Evans, G/F/J, Ripley's K/L and pair correlation, CSR envelopes
- **Spatial regression** — OLS diagnostics, Lagrange Multiplier tests, spatial lag / error / Durbin models, impacts
- **Spatial clustering** — Gi\* hot spots, LISA clusters, Kulldorff scan statistic, DBSCAN, SKATER regionalisation
- **Disease mapping** — SMRs, empirical Bayes smoothing, BYM (built-in MCMC sampler or PyMC)
- **Interpolation** — polynomial trend surfaces, Thiessen polygons, nearest neighbour, IDW, TIN, thin-plate splines, variograms and ordinary kriging, areal / dasymetric transfer
- **Spatial sampling** — random, systematic, stratified, two-stage, GRTS, cLHS designs and estimators
- **Geographically Weighted Modeling** (`spatialstats.gwmodel`) — GWR, MGWR, GW ML, GNNWR/GTNNWR

## Installation

```bash
pip install pyspatialstats                  # core
pip install pyspatialstats[viz]             # + matplotlib / folium
pip install pyspatialstats[deep]            # + PyTorch (GNNWR, GTNNWR)
pip install pyspatialstats[boosting]        # + XGBoost, LightGBM
pip install pyspatialstats[libpysal]        # + libpysal weight converters
pip install pyspatialstats[bayes]           # + PyMC / ArviZ (optional NUTS backend for BYM)
pip install pyspatialstats[all]             # everything
```

Or install from source:

```bash
git clone https://github.com/zia207/PySpatialStats.git
cd PySpatialStats
pip install -e ".[all]"
```

## Quick Start

### Exploratory spatial data analysis

```python
from spatialstats import Queen, Moran, Moran_Local
from spatialstats.datasets import load_lattice

gdf, y_col = load_lattice(n=8)
w = Queen(gdf)
mi = Moran(gdf[y_col], w, permutations=199)
print(mi.summary())

lisa = Moran_Local(gdf[y_col], w, permutations=199)
print(lisa.to_frame().head())
```

### Geographically weighted regression

```python
from spatialstats import GWR
from spatialstats.datasets import load_georgia

gdf, features, target = load_georgia()
coords = gdf[['x', 'y']].values
X = gdf[features].values
y = gdf[target].values

gwr = GWR(bandwidth=2.0, fixed=True, kernel='bisquare')
gwr.fit(X, y, coords)
print(gwr.summary())
```

### Spatial statistics tour

Every snippet below runs on the bundled `data/` files.

```python
import geopandas as gpd, numpy as np
from spatialstats import Queen

# Spatial regression: does the outcome need a spatial model?
from spatialstats.regression import OLS, SpatialLag, SpatialError, compare_models

gdf = gpd.read_file("data/diabetes_atlantic.shp")
gdf["Urban"] = (gdf["UrbRural"] == "Urban").astype(float)
X, w = gdf[["Obesity", "PhysInact", "SVI", "Urban"]], Queen(gdf)

ols = OLS(gdf["Dia_pct"], X, w=w)
print(ols.lm_tests().recommendation())                    # Lagrange Multiplier decision rule
slm, sem = SpatialLag(gdf["Dia_pct"], X, w), SpatialError(gdf["Dia_pct"], X, w)
print(compare_models({"OLS": ols, "SLM": slm, "SEM": sem})[["aic", "resid_moran_I"]])
print(slm.impacts(seed=1))                                # direct / indirect / total effects

# Clusters: hot spots, a population-adjusted scan statistic, contiguous regions
from spatialstats.cluster import getis_ord_gi, spatial_scan, skater

ne = gpd.read_file("data/diabetes_northeast.shp")
wne = Queen(ne)
hot = getis_ord_gi(ne["Dia_pct"], wne)                    # FDR-corrected Gi*
xy = np.column_stack([ne.geometry.centroid.x, ne.geometry.centroid.y])
scan = spatial_scan(xy, ne["Dia_count"], population=ne["POP_Total"], max_pop_fraction=0.1, seed=1)
regions = skater(ne[["Dia_pct", "Obesity", "SVI"]], wne, n_clusters=6)

# Disease mapping: shrink unstable small-area rates
from spatialstats.bayes import expected_counts, eb_gamma_poisson, BYM

O = ne["Dia_count"].to_numpy()
E = expected_counts(ne["POP_Total"], observed=O)
eb = eb_gamma_poisson(O, E)                               # empirical Bayes
fit = BYM(O, E, wne, seed=1)                              # BYM; backend="pymc" for NUTS
fit.rr_mean, fit.exceedance(1.1)                          # posterior relative risk, P(RR > 1.1)

# Point patterns
from spatialstats.pointpattern import PointPattern, Window, envelope

# these files declare UTM zone 11N but Buffalo lies in zone 17N: go through lon/lat to the right zone
# (Chapter 0 notebook, section 0.5, shows how to detect this)
to_utm17 = lambda g: g.to_crs(4326).to_crs(26917)
crimes = to_utm17(gpd.read_file("data/Crime_location_2018.shp"))
city = to_utm17(gpd.read_file("data/buffalo_city.shp"))
pp = PointPattern.from_gdf(crimes, window=Window.from_gdf(city), clip=True).deduplicate()
env = envelope(pp.subset(np.arange(1500)), "L", nsim=39, seed=1, r=np.linspace(0, 1500, 30))
env.test("mad")                                           # global Monte Carlo p-value against CSR

# Interpolation with cross-validation
import pandas as pd
from spatialstats.interpolate import IDW, OrdinaryKriging, compare_methods

p = pd.read_csv("data/CA_pm25_2025.csv")
s = p.groupby("Site ID").agg(lat=("Lat", "mean"), lon=("Long", "mean"), pm=("Daily_PM2.5_ug_m3", "mean"))
pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon, s.lat), crs=4326).to_crs(3310)
xy = np.column_stack([pts.geometry.x, pts.geometry.y])
compare_methods({"IDW": IDW(2), "kriging": OrdinaryKriging("auto")}, xy, pts["pm"].to_numpy())

# Sampling design, scored against a fully known population
from spatialstats.sampling import simple_random, grts, compare_designs

us = pd.read_csv("data/diabetes_usa.csv")
cxy, y = us[["X", "Y"]].to_numpy(), us["Diabetes_per"].to_numpy()
compare_designs(y, {"SRS": lambda rng: simple_random(cxy, 100, rng),
                    "GRTS": lambda rng: grts(cxy, 100, seed=rng)}, n_reps=300, seed=1)
```

See **[TUTORIAL.md](TUTORIAL.md)** for a full guide covering GW models, and
**[TUTORIAL_SPATIAL_STATISTICS_OUTLINE.md](TUTORIAL_SPATIAL_STATISTICS_OUTLINE.md)** for the spatial-statistics tutorial plan.

## Package layout

| Module | Status | Description |
|--------|--------|-------------|
| `spatialstats.core` | ready | SpatialData, SpatialResult, CRS, distance |
| `spatialstats.weights` | ready | Queen/Rook, KNN, kernel, Delaunay, Gabriel, RNG |
| `spatialstats.esda` | ready | Global/local autocorrelation, join counts |
| `spatialstats.viz` | ready | Moran scatter, LISA map, choropleth |
| `spatialstats.datasets` | ready | Teaching loaders |
| `spatialstats.gwmodel` | ready | GWR / MGWR / GW ML / deep GW models |
| `spatialstats.pointpattern` | ready | Windows, KDE, quadrat / Clark–Evans, G/F/J, Ripley K/L/g, cross-K, K-inhom, envelopes |
| `spatialstats.regression` | ready | OLS + diagnostics, LM tests, SLM / SEM / SDM / SLX by ML, impacts |
| `spatialstats.cluster` | ready | Gi\*, LISA clusters, scan statistic, DBSCAN/HDBSCAN, SKATER, constrained clustering |
| `spatialstats.bayes` | ready | SMR, empirical Bayes, BYM/ICAR (built-in sampler; PyMC optional, BYM2) |
| `spatialstats.interpolate` | ready | Trend surface, Thiessen polygons, nearest neighbour, IDW, TIN, thin-plate spline, variograms, ordinary kriging, areal/dasymetric, CV |
| `spatialstats.sampling` | ready | SRS, systematic, stratified, two-stage, GRTS, space-filling, cLHS, estimators |
| `spatialstats.spacetime` | planned | Stub package |

## Performance: Multi-core CPU & GPU

```python
from spatialstats import set_compute_options, GWR

set_compute_options(n_jobs=-1, device="auto")
```

## License

BSD-3-Clause
