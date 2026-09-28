# Spatial Statistics with PySpatialStats — Detailed Tutorial Outline

A part-by-part plan for a hands-on tutorial that covers every module of PySpatialStats
(`core`, `weights`, `esda`, `viz`, `datasets`, `gwmodel`, `pointpattern`, `regression`, `cluster`, `bayes`,
`interpolate`, `sampling`, and the still-planned `spacetime`).

*Status as of 2026-09-21: Part 0 is written, executed and rendered; Parts 1–8 have working package APIs but no notebooks yet;
Part 9 is blocked on the `spacetime` module.*

---

## How to use this outline

### Status at a glance

Legend — ✅ package API exists and has been validated · 🟡 partly covered (a workaround inside the notebook, or a stated limitation) · 🔲 blocked (module missing).

| # | Part | Chapters | Module(s) | Notebook(s) — `examples/notebooks/` | Est. time | Status |
|---|---|---|---|---|---|---|
| 0 | Setup and Foundations | 0.1–0.4 | `core`, `datasets`, `viz` | `ch00_setup_and_foundations.ipynb` (full) · `ch00_1`–`ch00_4` sub-chapters · HTML: `docs/notebooks/` | 60 min | ✅ **written and rendered** |
| 1 | Point Pattern Analysis | 1.1–1.5 | `pointpattern` | `ch01a_point_patterns_intensity`, `ch01b_point_patterns_interaction` | 2 × 60 min | ✅ API ready |
| 2 | Spatial Autocorrelation | 2.1–2.6 | `weights`, `esda`, `viz` | `ch02a_spatial_weights`, `ch02b_spatial_autocorrelation` | 2 × 60 min | 🟡 `esda` inference caveats (C1–C3) |
| 3 | Spatial Regression | 3.1–3.7 | `regression` (+ `esda`) | `ch03_spatial_regression` | 75 min | ✅ API ready |
| 4 | Spatial Cluster Analysis | 4.1–4.8 | `cluster` (+ `esda`, `weights`) | `ch04_spatial_clustering` | 75 min | ✅ API ready |
| 5 | Disease Mapping & Bayesian Smoothing | 5.1–5.11 | `bayes` (+ `esda`, `viz`) | `ch05a_disease_mapping_empirical_bayes`, `ch05b_bayesian_disease_mapping_bym` | 2 × 60 min | ✅ API ready (PyMC optional) |
| 6 | Geographically Weighted Models | 6.1–6.14 | `gwmodel` | `ch06a_gwr_core`, `ch06b_gw_extensions` | 2 × 75 min | ✅ (the deep-learning chapter needs PyTorch) |
| 7 | Spatial Interpolation | 7.1–7.8 | `interpolate` | `ch07_spatial_interpolation` | 75 min | ✅ API ready |
| 8 | Spatial Sampling and Design | 8.1–8.5 | `sampling` | `ch08_spatial_sampling` | 60 min | ✅ API ready |
| 9 | Spatio-Temporal Analysis | 9.1–9.7 | `spacetime` (+ `gwmodel.GTWR`, `esda`) | `ch09_spatiotemporal` | 60 min | 🔲 `spacetime` is a stub |
| 10 | Capstone: End-to-End Workflow | — | all | `ch10_capstone` | 90 min | 🟡 all but the space–time step |

Parts 1–6 follow the requested structure. Part 0 is a prerequisite; Parts 7–9 exist so that *every* module of the package appears in the tutorial.
Each chapter (1.1, 2.2, …) is a standalone chapter nested inside its part; the notebook column says which chapters are grouped into one notebook file (≈ 40–90 cells each, as in Part 0).

### Reading paths and dependencies

| Part | Needs | Feeds |
|---|---|---|
| 0 Foundations | — | everything |
| 1 Point patterns | 0 | 4.3 (density clusters), 7 (intensity ≈ surface), 9.4 |
| 2 Autocorrelation | 0 | 3, 4, 5, 6, 9 |
| 3 Regression | 0, 2 | 6 (comparison workflow), 10 |
| 4 Clustering | 0, 2 (weights, LISA); 1 for 4.3 | 5.8 (hot spots of raw vs smoothed), 10 |
| 5 Disease mapping | 0, 2 (binary weights, islands) | 6 (GWGLM), 9, 10 |
| 6 GW models | 0, 3 | 7.5, 9.6 |
| 7 Interpolation | 0 | 8.4 (monitoring networks) |
| 8 Sampling | 0; 2 (effective sample size); 7 (8.4) | 10 |
| 9 Space–time | 2, 6 | — |

Suggested paths: **health geography** 0 → 2 → 3 → 4 → 5 → 6 → 10 · **events and environment** 0 → 1 → 7 → 8 · **minimum viable** 0 → 2.

### Conventions for every notebook

These come from building Part 0, where reading the executed outputs caught six mismatches between prose and results.

**Structure.** Title table (prerequisites, modules, data, time) → learning objectives → roadmap → theory with mathematics → walkthrough → pitfalls → exercises **with executed solutions** → summary → where next → references.

**Truthfulness.**
- Numbers in prose are computed and inserted (`display(Markdown(f"…"))`), or checked against the output after execution. Never type an expected value.
- "Method A beats method B" needs a table, plot or `assert` in the notebook.
- Interpretation is written *after* the run. Timings vary between runs and are never quoted in prose.
- Reference numbers in this outline come from the **module validation runs**, not from notebooks; the notebook must recompute them.

**Data first.**
- Start each notebook by auditing its layers with the `crs_audit` / `load_study_area` recipe from Part 0 §0.9.
- Use **EPSG:26917** for the Buffalo files (their declared UTM 11N is wrong for the location; see C9).
- Prefer geometry to stored coordinate columns after any reprojection (C10).

**Reproducibility.** Explicit `seed=` everywhere; state permutation counts (`999` for reported results, `199` for drafts); relative paths only in outputs; no stderr noise (catch intentional warnings).

**Environment.** Python 3.11; extras `viz` (needs `mapclassify` for natural breaks), `bayes` and `libpysal`. PyTorch only for Part 6's deep-learning chapter (6.11), guarded by an availability check.

**Figures.** One style (see Part 0); every figure gets alt text; check figures by eye and against theory when a curve is expected (Part 0's Mercator plot lies on the sec²φ curve).

**Build pipeline.**
1. Execute in a clean kernel from **both** the repository root and `examples/notebooks/`; commit the executed `.ipynb`.
2. Render HTML to `docs/notebooks/` with nbconvert's `lab` template. That template ignores output-metadata alt text, so substitute the alt texts into the HTML afterwards.
3. Verify the LaTeX by typesetting a copy against a local MathJax and checking for error boxes.
4. The shipped HTML loads MathJax from a CDN (nbconvert default): fine online, raw LaTeX offline. Vendoring MathJax (≈ 2 MB) is an open decision (see "Decisions needed").

### Implemented API and remaining gaps

What each chapter can call now, and what this outline mentions that the package still does **not** do (say so in the tutorial).

| Part | Public API (import from the subpackage) | Not implemented / limits |
|---|---|---|
| 0 `core`, `datasets`, `viz` | `SpatialData` (`from_frame`, `set_roles`, `coords_array`, `to_crs`, `ensure_projected`), `SpatialResult`, `validate_crs`, `to_crs`, `is_geographic`, `is_projected`, `ensure_projected`, `coords_from_geometry`, `euclidean_distance_matrix`, `euclidean_distances`, `haversine_distance_matrix`, `pairwise_distances`, `normalize_coords`; `load_lattice`, `load_points_csr` (+ four simulated GW loaders); `choropleth` | No CRS audit helper (the notebook defines one); stale `coords` role after reprojection (C10); `ensure_projected` defaults to Web Mercator |
| 1 `pointpattern` | `Window`, `PointPattern` (`from_gdf`, `deduplicate`, `jitter`, `subset`, `by_mark`); `quadrat_test`, `kde`, `bandwidth_scott`, `bandwidth_cv`, `intensity_at_points`, `relative_risk`; `clark_evans`, `g_function`, `f_function`, `j_function`; `ripley_k`, `ripley_l`, `pair_correlation`, `cross_k`, `k_inhom`; `simulate_csr`, `simulate_thomas`, `simulate_matern_inhibition`; `envelope` (+ `.test('mad'/'dclf')`) | Isotropic (Ripley) edge correction (translation, border, none are provided); mark-correlation functions beyond `cross_k`; space–time (Knox) tests |
| 2 `weights`, `esda`, `viz` | `W` (`lag`, `transform`, `summary`, `islands`, `to_libpysal`), `Queen`, `Rook`, `higher_order`, `KNN`, `DistanceBand`, `Kernel`, `Delaunay`, `Gabriel`, `RelativeNeighborhood`; `Moran`, `Geary`, `GeneralG`, `MoranBV`, `Moran_Diff`, `Moran_Local`, `Geary_Local`, `G_Local`, `Join_Counts`, `fdr_bh`; `moran_scatter`, `lisa_cluster_map` | `esda` permutation inference is unreliable (C1–C3); Moran-scatter axis label (C4); loops are slow at n ≈ 3,000 (C5) |
| 3 `regression` | `OLS` (`lm_tests`, `residual_moran`, `diagnostics`, `jarque_bera`, `breusch_pagan`), `SLX`, `SpatialLag`, `SpatialError`, `SpatialDurbin` (`slx=True` on either estimator), `impacts`, `lr_test`, `compare_models`, `LMResult.recommendation` | 2SLS / GMM, SARAR, robust standard errors, prediction at new locations, spatial panels; standard errors need a dense n×n inverse (n ≤ 6,000) |
| 4 `cluster` | `getis_ord_gi` (Gi / Gi\*), `local_moran_clusters`, `spatial_scan`, `dbscan_clusters`, `hdbscan_clusters`, `k_distance`, `skater`, `agglomerative`, `evaluate_regions`, `jaccard`, `agreement_matrix`, `partition_agreement`, `flag_stability` | max-p regions, multivariate / bivariate local Moran, Bernoulli or space–time scan models, elliptical windows |
| 5 `bayes` | `expected_counts`, `smr`, `smr_confidence_interval`, `excess_pvalue`, `funnel_limits`; `eb_gamma_poisson` (MLE / moments), `eb_james_stein`, `eb_local`; `BYM` (`model='bym'/'icar'/'iid'`, `backend='gibbs'/'pymc'/'auto'`, covariates, `exceedance`, `shrinkage`, DIC, R̂/ESS); BYM2 via `backend='pymc'`; `simulate_relative_risk`, `simulate_counts`, `evaluate_smoothing`, `shrinkage_summary` | Leroux / space–time BYM, Assunção–Reis EB, age standardisation beyond `expected_counts(pop_by_stratum, reference_rate=…)`; PyMC backend needs a connected graph (no islands) |
| 6 `gwmodel` | `GWR`, `MGWR`, `MixedGWR`, `GWSS`, `GWGLM`, `GTWR`, `GWRandomForest`, `GWXGBoost`, `GWLightGBM`, `GNNWR`, `GTNNWR`, `GWPCA`, `NetworkGWR`, `BandwidthSelector`, diagnostics, visualization, `set_compute_options` | **No `offset` in `GWGLM`**; deep models need PyTorch; boosting needs XGBoost / LightGBM |
| 7 `interpolate` | `NearestNeighbor`, `IDW`, `TIN`, `RBF`, `Spline`; `empirical_variogram`, `fit_variogram`, `fit_variogram_auto`, `VariogramModel`, `OrdinaryKriging` (prediction variance, local kriging); `areal_weighting`, `dasymetric`; `make_grid`, `cross_validate` (`loo` / `kfold` / `spatial_block`), `grid_search`, `compare_methods` | Sibson natural neighbour (`TIN` is the linear analogue), universal / co-kriging, anisotropy, raster ancillary data |
| 8 `sampling` | `simple_random`, `systematic` (Hilbert / x / y order), `grid_points`, `stratified` (proportional / equal / Neyman; `within='grts'`), `two_stage`, `grts` (equal or PPS), `space_filling`, `clhs`; `estimate_mean` (SRS, stratified, local-mean, two-stage variance), `sample_size_mean`, `effective_sample_size`, `design_effect`; `spatial_balance`, `coverage_metrics`, `evaluate_design`, `compare_designs` | Local pivotal / balanced acceptance sampling, ratio and domain estimators, model-based design optimisation |
| 9 `spacetime` | *stub*; usable today: `gwmodel.GTWR`, per-year `esda`/`cluster` calls | Everything else in Part 9 |

**Validation evidence** (for "can I trust this?" boxes): regression matches `spreg` (OLS, all five LM statistics; ML lag to ~1e-8, ML error to ~1e-6);
kriging matches `pykrige` to 1e-14; Gi\* z-scores match `esda` to 1e-14; the BYM sampler and PyMC/NUTS agree on posterior relative risks (correlation 1.000);
simulation studies check CSR test size and power, edge-correction bias, scan-statistic size and power, GRTS inclusion probabilities and confidence-interval coverage.

---

## Part 0 — Setup and Foundations  ✅ written

**Notebooks:**
- Full chapter: [`ch00_setup_and_foundations.ipynb`](examples/notebooks/ch00_setup_and_foundations.ipynb) · **HTML:** [`docs/notebooks/ch00_setup_and_foundations.html`](docs/notebooks/ch00_setup_and_foundations.html) (86 cells, 143 formulas, 11 figures).
- Sub-chapters (Chapter 1 style; same content split for stepwise reading):
  - [`ch00_1_installation_and_package_map.ipynb`](examples/notebooks/ch00_1_installation_and_package_map.ipynb) — §§0.1–0.2
  - [`ch00_2_spatial_data_types_and_containers.ipynb`](examples/notebooks/ch00_2_spatial_data_types_and_containers.ipynb) — §§0.3–0.4
  - [`ch00_3_crs_and_distances.ipynb`](examples/notebooks/ch00_3_crs_and_distances.ipynb) — §§0.5–0.6
  - [`ch00_4_datasets_maps_and_workflow.ipynb`](examples/notebooks/ch00_4_datasets_maps_and_workflow.ipynb) — §§0.7–0.12

This section summarises what was built; later notebooks should link back to Part 0 instead of repeating it.

**Sections.**
- **0.1 Installation and environment** — extras read from `pyproject.toml` and checked against what is installed; the optional-dependency pattern (a lazy `ImportError` that names the extra).
- **0.2 Package map** — import graph derived from the source (eager vs lazy edges); the shared `SpatialResult` interface.
- **0.3 What is spatial data?** — geostatistical field $Z(\mathbf s)$, lattice $\{Z_i\}$ with $\mathbf W$, point pattern with intensity $\lambda(\mathbf s)$; support and MAUP.
- **0.4 `SpatialData`** — roles, `Mixed` geometry type, centroid vs representative point, reprojection helpers.
- **0.5 Coordinate reference systems** — theory (scale factors, Mercator sec²φ, grid convergence), measured area/distance distortion, the Buffalo case study, CRS mismatch and missing CRS.
- **0.6 Distances** — metric axioms, haversine, distance-matrix cost, per-axis normalisation trap.
- **0.7 Datasets** — a catalogue built by inspecting every file; synthetic data with known truth; data-hygiene traps.
- **0.8 First maps** — quantiles / equal interval / Fisher–Jenks, goodness of variance fit, counts vs rates.
- **0.9 Loading recipe** — `crs_audit` and `load_study_area`; **0.10** pitfalls; **0.11** exercises with solutions; **0.12** summary and references.

**Findings that later chapters depend on.**
- The Buffalo files are in the wrong UTM zone: scale 1.121, areas × 1.257, rotation 26.7° (C9). Reproject via lon/lat to EPSG:26917.
- `SpatialData` keeps a stale `coords` role after `to_crs` (C10), and `coords_array()` returns representative points, not centroids (C11).
- `natural_breaks` silently becomes quantiles without `mapclassify` (C12).
- Numeric FIPS merges work, but string keys built from float FIPS do not match and leading zeros are already lost.
- 36% of crime events (5,068 of 14,001) share a location with an earlier event.
- Web Mercator inflates the Northeast's area 1.86×; an equal-area CRS is within 0.0001%.

---

## Part 1 — Point Pattern Analysis  ✅ (`spatialstats.pointpattern`)

**Objectives.** Describe where events occur; estimate and map intensity; measure interaction between events; test complete spatial randomness (CSR) by simulation; connect point patterns to area-based analysis.
**Prerequisites.** Part 0 (CRS, distances, data audit). **Notebooks.** `ch01a` (chapters 1.1–1.2), `ch01b` (chapters 1.3–1.5).

**Data.**

| Dataset | Use |
|---|---|
| `load_points_csr(n=100)` | Calibration: what "random" looks like; envelope sanity check |
| `Crime_location_2018.shp` (14,001 pts) + `crime_data_2018.csv` (8 incident types; Theft 6,652, Assault 2,770, Breaking & Entering 2,377, Theft of Vehicle 1,014, …; hour, day) | Buffalo crime events; marks = `incident_type`. **Reproject to EPSG:26917 through lon/lat** (C9) |
| `buffalo_city.shp` (35 neighbourhoods) | Study window (`Window.from_gdf`) and aggregation units; the layer's `sqmiles` gives true areas (105.8 km² in total) |

`cancer_points.csv` is *not* used here: it covers the 217 northeastern counties, not Buffalo.

**Reference results** (module validation on synthetic data — the notebook must recompute them).

| Check | Result |
|---|---|
| K(r) bias, unit square, n = 80, 200 simulations | translation correction: mean ratio to πr² within 0.5% up to r = 0.25; uncorrected: about 20% too low |
| CSR test size (global MAD test, 5% level, 60 replicates) | ≈ 5% |
| Power vs a Thomas cluster process / a Matérn inhibition process | ≈ 100% / ≈ 100% |
| Buffalo events inside the city polygon | 13,992 of 14,001 (`clip=True` drops the rest) |

### Chapter 1.1 — Fundamentals of Point Patterns

- **Concepts.** Events vs points; study window; marks; first-order (intensity) vs second-order (interaction) properties; the homogeneous Poisson process as the CSR model, with clustered (Neyman–Scott, Cox) and regular (inhibition) alternatives; why the window and the population at risk matter.
- **Preparing point data.**
  - Audit and reproject (Part 0 §0.9); build the window with `Window.from_gdf(city)`; `PointPattern.from_gdf(pts, window=…, clip=True)`.
  - **Duplicates:** 36% of events share a location. Use `n_duplicates` to detect and `deduplicate()` or `jitter(scale)` to handle them, and say which and why — every distance-based statistic below is affected.
  - Window as bounding box vs polygon; edge effects; `Window.convex_hull` as an alternative window.

### Chapter 1.2 — First-order Analysis: Spatial Intensity

- **Intensity.** $\hat\lambda = n/|W|$ (`pp.intensity`); a non-constant λ(s) is the rule for crime.
- **Quadrat test** (`quadrat_test`): χ² and index of dispersion on cells weighted by their area inside the window; sensitivity to cell size; Monte Carlo p-value (`permutations=`) when expected counts are small.
- **Kernel density** (`kde`): bandwidth rules (`bandwidth_scott`, likelihood cross-validation `bandwidth_cv`), Gaussian vs Epanechnikov product kernels, uniform edge correction on any window shape; the integral of the surface equals *n* (check it); `KDEResult.at()` for evaluation at events; map with the Part 0 conventions.
- **Comparing surfaces.** Thefts vs assaults; relative-risk surface with `relative_risk(cases, controls)` — e.g. assaults (cases) against all other crime (controls) — to separate "more crime" from "more assault-prone".

### Chapter 1.3 — Second-order and Distance-based Summary Functions

- **Nearest-neighbour metrics.** `nearest_neighbor_distances`; `clark_evans` with the Donnelly boundary correction (compare `correction='none'`); `g_function` and `f_function` with Kaplan–Meier / border / no correction; `j_function` = (1−G)/(1−F).
- **K, L and g.** `ripley_k`, `ripley_l` (plot L(r) − r), `pair_correlation` (kernel-smoothed; its bandwidth rule). Edge corrections available: **translation** (exact set covariance for rectangles, raster approximation for polygons), **border**, none — why the uncorrected estimator is biased low at large r.
- **Reading the plots.** Above the CSR curve = clustering at that scale; below = regularity; multi-scale structure; do not read r beyond about a quarter of the window's short side.

### Chapter 1.4 — Hypothesis Testing and Model Extensions

- **Monte Carlo envelopes** (`envelope`): conditional CSR simulation in the same window; pointwise band vs global tests (`.test('mad')`, `.test('dclf')`); why a pointwise band over-rejects when scanned over many r. Teach calibration with `simulate_csr`, `simulate_thomas`, `simulate_matern_inhibition` (size ≈ 5%, power ≈ 100% in validation).
- **Beyond CSR.**
  - **Inhomogeneous K** (`k_inhom`): intensity from the pattern itself (leave-one-out kernel estimate, `intensity_at_points`) or supplied; contrast with the homogeneous K that mistakes a gradient for clustering.
  - **Marked patterns:** `cross_k` between incident types (are thefts clustered around assaults?).
  - **Time marks:** hour and weekday as a bridge to Part 9.
- Connection to permutation ideas in Part 2.

### Chapter 1.5 — Integration with Areal Analysis, Pitfalls and Exercises

- **Points to areas.** `geopandas.sjoin` counts per neighbourhood; **crimes per km² using true areas** (from `sqmiles` or the corrected CRS), not raw counts; small-number instability as a prelude to Part 5; Queen weights and Moran's I on the 35 polygons (Part 2).
- **Pitfalls.** MAUP, window choice, duplicate points, wrong CRS (a 12% distance error changes every r), "clustering" that is only population.
- **Exercises.**
  1. Does the CSR envelope for thefts change if the window is the convex hull instead of the city boundary?
  2. Repeat the K analysis in the declared CRS and in EPSG:26917 — by how much do the scales shift, and why?
  3. Deduplicate vs jitter: how do G(r) and K(r) at small r change?
  4. Is there evidence that assaults and vehicle thefts attract each other (`cross_k` with an envelope)?

---

## Part 2 — Spatial Autocorrelation  🟡 (`weights`, `esda`, `viz`)

**Objectives.** Understand and measure spatial dependence; build and compare spatial weights; test global and local autocorrelation; map clusters; interpret Moran scatterplots for health outcomes.
**Prerequisites.** Part 0. **Notebooks.** `ch02a` (2.1–2.3: concepts, weights, exploration), `ch02b` (2.4–2.6: global, local, robustness).

**Data.**

| Dataset | Use |
|---|---|
| `diabetes_northeast.shp` (217 counties; `Dia_pct`, `Obesity`, `PhysInact`, `Acc_Exer`, `PCP_rate`, `FoodEnvIdx`, `SVI`, `UrbRural`; EPSG:5070) | Primary example. Queen weights: mean 5.19 neighbours, max 10, **1 island, 2 components** |
| `diabetes_atlantic.shp` (666 counties) | Larger scale; Queen: no islands, 1–11 neighbours |
| `diabetes_usa.csv` + `US_COUNTY.shp` (3,107) | National scale (mind the slow loops, C5) |
| `load_lattice(n=8)` | Known-truth sandbox: verify autocorrelation by hand |
| `data_1998_2010_long_lbc.csv` (3,107 counties × 15 years, **1998–2012**) | Change over time (`Moran_Diff`) |

**Reference results** (validation run, `Dia_pct` on the Northeast, Queen).

| Check | Result |
|---|---|
| Global Moran's I | 0.240 (expected −0.0046), z = 5.61, permutation p = 0.005 with 199 permutations (the floor is 1/(m+1)) |
| `esda.Moran_Local`, default `fdr=True` | **0 significant counties** |
| `esda.Moran_Local`, `fdr=False`, 999 permutations | 29 significant: 16 HH, 10 LL, 2 LH, 1 HL |
| `cluster.local_moran_clusters`, no correction | 30 significant at p ≤ 0.05 (two-sided); FDR with 9,999 permutations: 1 HH |

### Chapter 2.1 — Concepts, Data and Foundations

- **Tobler's first law**; positive, negative and no autocorrelation; spatial dependence vs spatial heterogeneity and why the difference matters for inference.
- Why care: violated independence inflates significance and biases inference (motivation for Part 3).
- Scale, MAUP and the choice of areal unit; data overview as in the table above.

### Chapter 2.2 — Spatial Weights: Building and Understanding (`weights`)

- **Anatomy of `W`:** `neighbors`, `weights`, `id_order`, `sparse`, `cardinalities`, `islands`, `n_components`, `is_symmetric`, `summary()`.
- **Types.**
  - Contiguity: `Queen` (edge or vertex) vs `Rook` (edge only) — different on lattices, similar on irregular counties.
  - Distance: `DistanceBand(threshold, binary, alpha)` — the smallest threshold with no island.
  - `KNN(k)` (k = 4, 6, 8; asymmetric).
  - `Kernel(bandwidth | k, kernel, fixed)` — link to Part 6.
  - `Delaunay`, `Gabriel`, `RelativeNeighborhood` — nested sparsity.
  - `higher_order(w, k)` for correlograms.
- **Transformations** (`W.transform('row' | 'binary' | …)`): why row-standardisation is the default and where it hurts; ICAR in Part 5 needs binary.
- **Islands and components:** the Northeast has one — options are a KNN fallback, dropping it, or a manual link; every downstream test is affected.
- **Coordinates for distance-based weights:** projected CRS; representative point vs centroid (Part 0 §0.4).
- **Spatial lag** `W.lag(y)`: hand computation on the 8×8 lattice. **Interoperability:** `to_libpysal()` / `from_libpysal()` (`[libpysal]` extra).
- **Sensitivity table (preview):** one variable under six weight definitions → Moran's I (completed in 2.6).

### Chapter 2.3 — Visualisation and Exploratory Analysis

- `viz.choropleth` of `Dia_pct` under three classification schemes (Part 0 §0.8); the "eyeball test" before measuring.
- **Moran scatterplot** (`viz.moran_scatter(result)` or `MoranResult.plot()`): axes z vs Wz; the slope is Moran's I for row-standardised W; four quadrants HH, LL, HL, LH; high-leverage points.
- **Health reading:** HH = high prevalence in a high region (regional intervention); HL = a local outlier (local factor or data issue); LH = a possibly protective pocket; LL = a low-prevalence region.
- Note that the axis label says "standardized" although the code only centres (C4).

### Chapter 2.4 — Measuring Global Spatial Autocorrelation (`esda`)

- **Moran's I:** definition, E[I] = −1/(n−1), variance, z; `Moran(y, w, permutations, seed)` → `.I`, `.expected_I`, `.z_score`, `.p_norm` (analytic), `.p_sim`, `.p_value`. Permutation mechanics, the pseudo p-value (1 + #extreme)/(1 + m), the minimum attainable p = 1/(m+1), choosing m.
- **Other global measures.**
  - `Geary` (C < 1 ⇒ positive autocorrelation). **Do not report its permutation p-value until C1 is fixed** (C = 0.73 returned p = 1.0).
  - `GeneralG` — high-value vs low-value clustering.
  - `Join_Counts` — binary "above median" (BB / WW / BW).
  - `MoranBV` — obesity at *i* vs diabetes at neighbours.
  - `Moran_Diff` — 1998 vs 2012 county rates.
  - Spatial correlogram: Moran's I over `higher_order(w, k)`, k = 1…8.
- **Output handling:** `.summary()`, `.stats`, `.to_frame()` (Part 0 §0.2).

### Chapter 2.5 — Local Spatial Autocorrelation (LISA)

- **Local Moran's I.** `Moran_Local(y, w, permutations, alpha, fdr)` gives `Ii`, `q` (1 = HH, 2 = LH, 3 = LL, 4 = HL, 0 = NS), `label`, `p_sim`, `p_adjusted` via `.to_frame()`.
- **Inference — teach both, and be explicit about the risk (C2, C3).**
  - *Conditional permutation:* hold y_i fixed, resample its neighbours from the other n−1 values. `cluster.local_moran_clusters` implements this; `esda.Moran_Local` permutes all of y.
  - *Multiple testing:* Benjamini–Hochberg (`fdr_bh`) and its resolution limit — the smallest two-sided p is 2/(m+1), so FDR needs m in the thousands for n ≈ 200; show raw and adjusted maps side by side.
  - Recommended workflow: `esda` for Iᵢ and quadrants, `cluster.local_moran_clusters` for significance; cross-check with PySAL (Appendix F).
- **Mapping:** `viz.lisa_cluster_map(result, gdf)` or `LocalMoranResult.plot()`; read the cluster map against the choropleth and the Moran scatterplot (three-panel figure).
- **Companions:** `Geary_Local`, `G_Local` (full treatment in Part 4).

### Chapter 2.6 — Sensitivity, Pitfalls and Exercises

- **Sensitivity.** Repeat under Queen, KNN(6), DistanceBand; which clusters persist (`cluster.flag_stability`); permutation count and seed; row-standardised vs binary; island handling.
- **Pitfalls.** Small-population rates create spurious clusters (Part 5); global stationarity assumed (Part 6); edge effects, MAUP, projection; clusters are not causes (Part 3).
- **Exercises.**
  1. Compute the spatial lag on the lattice by hand and with `W.lag`.
  2. Rank six weight definitions by Moran's I and explain the differences.
  3. Compare LISA clusters for `Obesity` and `Dia_pct`.
  4. How many permutations are needed before FDR flags anything, and why?
  5. Reproduce the global Moran's I with `libpysal`/`esda` and report the agreement.

> ⚠ **Before writing this chapter:** resolve or explicitly disclose C1–C3 (Geary p-value, permutation scheme, `fdr=True` default). In the smoke test the default `Moran_Local` labelled all 217 counties "not significant" although global I = 0.24 is strongly significant.

---

## Part 3 — Spatial Regression  ✅ (`spatialstats.regression`)

**Objectives.** Detect spatial dependence in OLS residuals; fit and interpret spatial lag and spatial error models; choose between models using formal tests and information criteria.
**Prerequisites.** Parts 0, 2. **Notebook.** `ch03_spatial_regression`.

**Data.** `diabetes_atlantic.shp` (666 counties): `Dia_pct ~ Obesity + PhysInact + Acc_Exer + PCP_rate + FoodEnvIdx + SVI + Urban`, with `Urban = (UrbRural == "Urban")` (a string column must be coded); Queen weights, row-standardised, no islands.

**Reference results** (validation run, exactly this specification).

| Model | log-lik | AIC | BIC | Spatial parameter | Moran's I of residuals |
|---|---|---|---|---|---|
| OLS | −784.2 | 1584.4 | 1620.4 | — | 0.181 (z = 7.6) |
| SLM (spatial lag) | −767.8 | 1553.6 | 1594.1 | ρ = 0.178 (s.e. 0.031) | 0.075 |
| SEM (spatial error) | −761.8 | 1541.7 | 1582.2 | λ = 0.352 (s.e. 0.051) | −0.022 |
| SDM (Durbin) | −753.1 | 1538.2 | 1610.2 | ρ = 0.321 | −0.021 |

- LM statistics: lag 36.1, robust lag 5.5, error 53.9, robust error 23.3, SARMA 59.4 → `recommendation()` says *spatial error model (both robust tests significant; larger statistic chosen)*. LR test OLS vs SLM: 32.8, p ≈ 1e-8.
- **A teaching point:** AIC prefers the SDM (1538.2) but BIC prefers the SEM (1582.2 vs 1610.2) — the SDM adds 7 spatially lagged covariates. Discuss what the criteria trade off.
- For a row-standardised W the total impact of a regressor in the SLM is exactly β/(1−ρ) (`impacts()` reproduces it).

### Chapter 3.1 — Foundations of Spatial Regression

- Why OLS fails on spatial data: residual dependence biases standard errors (error case) or coefficients (omitted-lag case).
- **Two mechanisms:** *substantive* dependence (spillovers → lag model) vs *nuisance* dependence (unmodelled spatial structure → error model).
- Taxonomy: OLS → SLX → SAR (SLM) → SEM → SARAR → SDM → SDEM (which are available: see the API table).
- Data preparation (above); standardise covariates for interpretability; check islands before anything else.

### Chapter 3.2 — OLS Baseline and Diagnostics

- `OLS(y, X, w=w)`: coefficients, R², adjusted R², Gaussian log-likelihood, AIC, BIC. **Information criteria count the β's plus the spatial parameter (σ² is not counted, for every model alike)**, so they are comparable across OLS, SLM, SEM, SDM on the same data and W.
- Classical diagnostics: condition number, `jarque_bera`, `breusch_pagan`.
- **Residual autocorrelation:** `residual_moran()` uses the Cliff–Ord moments for regression residuals; `viz.moran_scatter` and `viz.choropleth` of the residuals — the bridge from Part 2.

### Chapter 3.3 — Model Selection: Spatial Dependence Diagnostics

- `lm_tests()`: LM-lag, LM-error, robust versions, SARMA (χ², 1 d.f.; 2 for SARMA); dependence on the trace terms of W.
- **Anselin's decision rule** (`LMResult.recommendation()`): neither significant → OLS; one → that model; both → compare robust versions.
- How to read the diagnostics table; why the answer can change with W.

### Chapter 3.4 — Spatial Regression Model Families

- **SLM / SAR:** y = ρWy + Xβ + ε, reduced form y = (I − ρW)⁻¹(Xβ + ε). ML via the concentrated log-likelihood with a sparse log-determinant; ρ bounds from the extreme eigenvalues of W; asymptotic standard errors from the information matrix.
- **Interpreting coefficients:** β is *not* a marginal effect. `impacts()` gives direct, indirect (spillover) and total effects with simulation intervals (LeSage & Pace). Example: a change in obesity in county *i* changes diabetes in *i* and, through ρ, in its neighbours.
- **SEM:** y = Xβ + u, u = λWu + ε; coefficients keep their ordinary meaning; what λ ≠ 0 implies (omitted spatial structure or mismatch). Filtered vs raw residuals.
- **Extensions:** `SLX` (OLS with WX), `SpatialDurbin` (SAR + WX), Durbin error (`SpatialError(slx=True)`); the common-factor idea (SDM ⇒ SEM). Not available: 2SLS/GMM, SARAR.

### Chapter 3.5 — Comparison, Reporting and Diagnostics

- **Comparing models:** `compare_models` (log-lik, AIC, BIC, pseudo-R², ρ/λ, residual Moran); `lr_test` for nested pairs (OLS vs SLM, SLM vs SDM). Likelihood comparisons are valid only for the same data and W.
- **Pseudo-R²** = squared correlation of y with ρWy + Xβ (PySAL's definition; also `predy_reduced` for the reduced form) — not comparable with OLS R².
- **Acceptance test:** residual Moran's I must be ≈ 0 after fitting (0.181 → −0.022 for the SEM).
- Sensitivity to W (Queen vs KNN); summary table OLS | SLX | SLM | SEM | SDM; reporting standards: state W, estimator, impacts, diagnostics.
- Prediction: in-sample only for now (no new-location prediction).

### Chapter 3.6 — Limits and Connections

- OLS, SAR and SEM are **global**: one β for the whole region. Mapping residuals motivates GWR (Part 6); the joint comparison workflow is Chapter 6.8.

### Chapter 3.7 — Code Skeleton and Exercises

```python
from spatialstats import Queen
from spatialstats.regression import OLS, SpatialLag, SpatialError, SpatialDurbin, compare_models, lr_test

w = Queen(gdf)                                   # row-standardised
ols = OLS(gdf["Dia_pct"], X, w=w)                # X: DataFrame with the 7 covariates
ols.lm_tests().recommendation()                  # decision rule
slm, sem = SpatialLag(y, X, w), SpatialError(y, X, w)
compare_models({"OLS": ols, "SLM": slm, "SEM": sem, "SDM": SpatialDurbin(y, X, w)})
slm.impacts(seed=1)                              # direct / indirect / total, with intervals
lr_test(ols, slm)
```

**Exercises.**
1. Show that SEM coefficients are close to OLS while SLM coefficients differ; explain.
2. Choose between SLM and SEM with the LM tests, then confirm with AIC — do they agree?
3. Verify total impact = β/(1−ρ) for the SLM; what changes for a non-row-standardised W?
4. Refit with a KNN(6) W: do the conclusions and the residual Moran's I change?
5. Explain why AIC and BIC disagree about the SDM.

---

## Part 4 — Spatial Cluster Analysis  ✅ (`spatialstats.cluster`)

**Objectives.** Distinguish the several meanings of "cluster"; detect hot spots, cluster–outlier patterns, density clusters, scan-statistic clusters and contiguous regions; choose and validate the method that matches the question.
**Prerequisites.** Parts 0, 2 (weights, LISA); Part 1 for Chapter 4.3. **Notebook.** `ch04_spatial_clustering`.

**Data.** `diabetes_northeast.shp` (hot spots, scan statistic with `Dia_count` and `POP_Total`, regionalisation); `Crime_location_2018.shp` reprojected to EPSG:26917 (density clusters); `cancer_areas.shp` + `cancer_points.csv` (optional second scan example — the counts must be *derived*, see 4.4).

**Reference results** (validation runs).

| Check | Result |
|---|---|
| Gi\* on `Dia_pct`, Queen, analytic p with FDR (default) | 7 hot spots, 3 cold spots |
| Gi\* with permutation p and FDR, 999 permutations | 0 flagged — the p-value floor (2/(m+1)) blocks FDR (a warning is issued) |
| Null calibration of Gi\* (random data) | 4.7% raw rejections at 5%; FDR flags anything in ≈ 7% of null datasets |
| Scan statistic under H0 (150 replicates, 99 permutations) | false-positive rate 4.7% |
| Scan power, RR = 1.6 planted in 15 areas (30 replicates) | 100%; recovered 13 of 15 planted areas with no spurious members |
| SKATER, 6 regions on `Dia_pct`, `Obesity`, `SVI` | R² = 0.27; the island forms its own region (warning) |

### Chapter 4.1 — Concepts and Overview

| Question | Family | Tools | Status |
|---|---|---|---|
| Are high values near each other? Where? | Statistical hot spots | `getis_ord_gi`, `local_moran_clusters` (compare `esda.G_Local`, `Moran_Local`) | ✅ |
| Do *cases* exceed what population implies? | Scan statistic | `spatial_scan` (Poisson, circular) | ✅ |
| Where are *events* densely packed? | Density-based | `dbscan_clusters`, `hdbscan_clusters`, `k_distance`; KDE (Part 1) | ✅ |
| Which areas form similar *contiguous regions*? | Regionalisation | `skater`, `agglomerative`, `evaluate_regions` | ✅ (max-p missing) |

### Chapter 4.2 — Hot-Spot Detection and Local Clustering

- **Global tests (review):** `GeneralG`, `Join_Counts` — *is there* clustering vs *where*.
- **Getis–Ord Gi and Gi\*** (`getis_ord_gi(y, w, star, inference, permutations, correction, alpha)`): Gi excludes the area, Gi\* includes it; analytic z (Ord & Getis) and conditional-permutation p are both returned; tiers 99/95/90% from adjusted p; `.hot`, `.cold`, `.labels`, `.plot(gdf)`.
- **Multiple testing:** FDR (default), Bonferroni or none; dependence between the n tests; the permutation resolution limit above.
- **Gi\* vs Local Moran:** Gi\* finds clusters of high or low *values*; LISA also finds spatial outliers — side-by-side maps and a confusion table (`agreement_matrix`).
- **Local Geary and multivariate clustering:** `esda.Geary_Local`; multivariate LISA is a 🟡 workaround (standardise → first principal component → local Moran).
- **Robustness:** weights, permutations, α, FDR on/off; a stability index with `flag_stability` (share of settings in which a county is flagged).

### Chapter 4.3 — Density-Based Point Clustering

- `k_distance` (knee for `eps`), `dbscan_clusters(coords, eps, min_samples)`, `hdbscan_clusters(min_cluster_size)`; noise, cluster hulls and densities (`summary_frame`).
- **Duplicates count as separate points**, so a single address with many reports can become a "cluster" — deduplicate or reason about marks first (Part 0: 36% of events repeat a location).
- Compare with a KDE surface from Part 1 on the same CRS and window; `eps` must be in true metres, hence the corrected CRS.

### Chapter 4.4 — Spatial Scan Statistic for Clusters of Cases

- Kulldorff scan with a Poisson model: circular windows grown around each location, log-likelihood ratio, Monte Carlo p-value over the *maximum* ratio (so searching everywhere is already accounted for); most likely and secondary non-overlapping clusters (`n_clusters`); `max_pop_fraction` (0.5 vs 0.1–0.25), `direction`.
- **Data:** primary example `Dia_count` with `POP_Total` on the Northeast (`expected` or `population`).
- **Secondary example (with a stated assumption):** `cancer_areas.shp` provides only `rate` (98–192). Cases are *derived* as rate × population / unit, with population from `cancer_points.POP10` summed by `FIPS` — confirm what the rate means (per what, age-adjusted?).
- **Key hazard:** `cancer_areas.FIPS` is int and `cancer_points.FIPS` is float. A numeric `merge` works (pandas casts), but string keys built from them do not match (`'42049'` vs `'42049.0'`); build a canonical 5-digit string key (Part 0 §0.7).
- Contrast with Gi\*: the scan adjusts for population at risk; Gi\* uses a value field.

### Chapter 4.5 — Regionalisation: Spatially Constrained Clustering

- **Problem:** group counties into *k* contiguous regions with similar profiles.
- `skater(data, w, n_clusters, min_size, floor, floor_value)`: minimum spanning tree with attribute-distance costs, then cuts that maximise the reduction in within-region sum of squares; every region is contiguous by construction. `agglomerative` uses **W as the connectivity graph** of `sklearn`'s Ward clustering (regions are contiguous only within connected components).
- **Islands:** a disconnected component (the Northeast island) always forms its own region and violates `min_size` / `floor_value`; a warning says so. With the island removed the constraints are honoured (minimum size 10, minimum regional population 1.4 M in the validation run).
- Validation: `evaluate_regions` (within SS, R², silhouette), `RegionResult.is_contiguous(w)`, choosing *k*. Missing: max-p regions.

### Chapter 4.6 — Comparing and Validating Cluster Results

- One panel: Gi\* | LISA | scan | DBSCAN | regions on a shared basemap; `agreement_matrix` (Jaccard), `partition_agreement` (adjusted Rand, NMI) between two regionalisations.
- What several methods agree on is credible; what appears in only one needs an explanation. Reporting: state W, α, correction, permutations, seed.

### Chapter 4.7 — Bridges

- Space–time clusters and emerging hot spots → Part 9; population-adjusted risk instead of raw rates → Part 5.

### Chapter 4.8 — Exercises

1. Do Gi\* and Local Moran flag the same counties for `Dia_pct`? Quantify with `agreement_matrix`.
2. Re-run the scan with maximum window sizes of 10% and 50% of the population; what changes?
3. Show why 999 permutations plus FDR flags nothing, and find the number of permutations at which it starts to.
4. Compare `agglomerative` and `skater` with `partition_agreement`; which is contiguous?
5. Add a minimum-population floor to `skater` and explain what happens to the island.

---

## Part 5 — Disease Mapping and Bayesian Smoothing  ✅ (`spatialstats.bayes`)

**Objectives.** Understand why small-area rates are unstable; implement empirical Bayes smoothing; understand and fit the BYM model (built-in sampler, PyMC when installed); compare raw and smoothed estimates visually and statistically; interpret shrinkage.
**Prerequisites.** Parts 0, 2 (binary weights, islands). **Notebooks.** `ch05a` (5.1–5.5: the problem, raw estimates, empirical Bayes), `ch05b` (5.6–5.11: BYM, MCMC, comparison, reporting).

**Data.** `diabetes_northeast.shp` (`Dia_count`, `POP_Total`); a **simulation with known truth** on the same county graph; `cancer_areas.shp` + `cancer_points.csv` for a real rate example.

**Reference results** (validation study: true log-risk smooth with σ = 0.35 on the 217-county graph; expected counts scaled to average 40; mean of 6 replicates).

| Estimator | RMSE vs truth | Spearman with truth |
|---|---|---|
| Raw SMR | 0.364 | 0.75 |
| EB gamma–Poisson (MLE) | 0.210 | 0.78 |
| EB gamma–Poisson (moments) | 0.219 | 0.77 |
| Normal EB on log SMR | 0.214 | 0.77 |
| Local EB (neighbourhood) | 0.167 | 0.89 |
| **BYM (built-in sampler)** | **0.142** | **0.91** |

BYM 95% credible intervals covered the truth 96.7% of the time (mean width 0.59).
**Sampler settings matter:** with `thin=1` (1,500 draws, 1,000 tuning) the worst R̂ was 1.32 and the worst effective sample size 14; with the default `thin=10`, `tune=2000` it was 1.016 and 345 (about 10 s for 4 chains). PyMC/NUTS agreed with the built-in sampler (posterior-mean correlation 1.000, τ_u 2.10 vs 2.12, DIC 639.0 vs 638.3).

### Chapter 5.1 — Introduction and Objectives

- Objectives as above. The package offers `expected_counts`, `smr`, `eb_*`, `BYM`; the MCMC step uses the built-in sampler by default and PyMC (`pyspatialstats[bayes]`) on request, so the notebook runs with or without PyMC.

### Chapter 5.2 — Motivation: the Small-Area Estimation Problem

- Poisson variability: Var(rate) ∝ 1/population, so the smallest areas produce the most extreme rates.
- The **funnel**: raw rates against population (`funnel_limits`), extremes belonging to small areas.
- Consequences: false hot spots, unstable rankings, small-cell privacy issues, misleading Part 2/4 clusters.
- Goal: **borrow strength** — across all areas (empirical Bayes) and across neighbours (spatial models).

### Chapter 5.3 — Data and Making the Problem Visible

- **Observed / expected:** `E = expected_counts(POP_Total, observed=Dia_count)` — indirect standardisation *without* age strata (state the limitation).
- **These counties are not sparse** (minimum 368 cases, minimum population ≈ 4,400), so shrinkage would be mild. Remedies: (1) **simulation** — draw a spatially smooth true risk (`simulate_relative_risk`), then `simulate_counts` with heterogeneous E; (2) thinning to emulate a rare outcome.
- **Real rate example:** `cancer_areas.shp` has only `rate`; derive counts as rate × population (`cancer_points.POP10` summed by `FIPS`) and document the assumption. The 217 areas are the same northeastern counties as `diabetes_northeast`.
- **Weights:** Queen, binary (ICAR needs 0/1 adjacency); the island has no spatial information and receives only the unstructured effect (PyMC needs the island linked).

### Chapter 5.4 — Raw Estimates and Naïve Inference

- Crude rate, **SMR = O/E**, its Poisson variance, exact intervals (`smr_confidence_interval`), `excess_pvalue`; map with `viz.choropleth`.
- How much apparent autocorrelation in the SMR is due to unequal populations? (`esda.Moran`.)

### Chapter 5.5 — Empirical Bayes Smoothing

- **Gamma–Poisson:** θ ~ Gamma(a, b), O | θ ~ Poisson(Eθ), posterior mean (a + O)/(b + E) = w·SMR + (1 − w)·(a/b) with w = E/(E + b), so **small E ⇒ more shrinkage**. `eb_gamma_poisson(O, E, method='mle' | 'mom')`. If there is no overdispersion the moment prior variance is ≤ 0 and every area is shrunk to the overall mean — a legitimate result, not an error.
- **Normal / James–Stein shrinkage** on log rates: `eb_james_stein(x, var, method='normal' | 'james-stein')`; positive-part rule; needs k ≥ 4 for the classical version.
- **Spatial EB:** `eb_local(O, E, w)` shrinks toward the neighbourhood mean (Marshall); an island falls back to the global mean.
- Output table: raw SMR, EB-global, EB-local, weight on the data, expected count.

### Chapter 5.6 — Spatial Bayesian Models: BYM and BYM2

- **BYM:** O ~ Poisson(Eθ), log θ = α + xβ + u + v; **u** ICAR (conditional on neighbours, precision τ_u(D − W), sum-to-zero); **v** iid (area-specific heterogeneity); `model='bym' | 'icar' | 'iid'`. Covariates: residual smoothing vs explanation.
- **BYM2** (PyMC only): b = σ(√(φ/s)·u\* + √(1−φ)·v\*), φ the spatial fraction, s the Sørbye–Rue scaling (`icar_scaling_factor`); draw the DAG and map every symbol to code.
- What the posterior provides: relative risks, credible intervals, **exceedance probabilities** P(θ > 1) (`exceedance`), `spatial_fraction`, DIC.

### Chapter 5.7 — MCMC Implementation

- **Built-in sampler** (default): blocked Gibbs / Metropolis in the convolution parametrisation, graph-coloured updates; handles islands; multiple chains with dispersed starts; `max_rhat`, `min_ess`, `diagnostics`.
- **PyMC backend** (`backend='pymc'`, optional): NUTS; needs a connected graph. **Gotcha worth teaching:** PyMC's `ICAR` is *unnormalised* in its scale (its density lacks the σ^−(n−c) term), so putting a prior on `sigma` gives a wrong posterior — use a fixed-scale ICAR multiplied by the scale (what the package does).
- Diagnostics: R̂ (< 1.05), effective sample size, divergences; why the built-in sampler needs thinning; posterior predictive checks; DIC (and WAIC/LOO with ArviZ when using PyMC).
- If PyMC is absent the notebook still runs; PyMC-only cells (BYM2) are guarded.

### Chapter 5.8 — Comparing and Evaluating Smoothing Methods

- **Visual:** maps of SMR | EB | BYM with **identical class breaks**; raw-vs-smoothed scatter with the 1:1 line; funnel plot with smoothed values; shrinkage against E; maps of credible-interval width and P(θ > 1).
- **Statistical (simulated truth):** `evaluate_smoothing` — RMSE, MAE, bias, Spearman, interval coverage and width (results above); E-weighted RMSE; how many counties change decile.
- **Spatial structure:** smoothing raises autocorrelation by construction — do not read it as evidence of a process. Compare hot spots from raw SMR and smoothed risk (Part 4): how many "clusters" were noise?
- **Over-smoothing:** genuine local excesses can be erased; prior sensitivity (`priors={'tau_u': …}`).

### Chapter 5.9 — Shrinkage Assessment and Interpretation

- Shrinkage toward the global mean (EB) vs the neighbourhood mean (spatial models); which areas move most (small populations, extreme SMRs, isolated areas); `BYM.shrinkage()` and `shrinkage_summary`.

### Chapter 5.10 — Reporting, Interpretation and Ethics

- Report smoothed maps with uncertainty; small-cell suppression; ecological fallacy; communicating risk.

### Chapter 5.11 — Hand-offs and Exercises

- To Part 6: `GWGLM` Poisson for counts — **no offset support**, so it cannot replace E-adjusted models. To Part 9: space–time BYM (not available).
- **Exercises.**
  1. Simulate a rare outcome and show EB beats the SMR in RMSE.
  2. Show the weight on the data tends to 1 as E → ∞.
  3. Rerun BYM with `thin=1` and read the diagnostics: what goes wrong?
  4. Add `Obesity` as a covariate; how do the smoothed risks and the spatial fraction change?
  5. Compare `model='bym'`, `'icar'` and `'iid'` by DIC.

---

## Part 6 — Geographically Weighted Models  ✅ (`spatialstats.gwmodel`)

**Objectives.** Model spatially varying relationships; select kernel and bandwidth; diagnose and interpret local coefficients; compare global, spatial and local models.
**Prerequisites.** Parts 0, 3. **Notebooks.** `ch06a` (6.1–6.8: core GWR), `ch06b` (6.9–6.14: extensions and practice).
The API is documented in `TUTORIAL.md` (§3.3 the fit → predict pattern, §4.2 first GWR, §5.1 bandwidth selection, §9 diagnostics); this chapter reframes it around a health workflow and does not repeat it.

### Chapter 6.1 — Introduction and Motivation

- The stationarity assumption of global models (Part 3); evidence of non-stationarity (residual maps, local statistics, regional fits); local regression, kernel weights, bandwidth as a spatial scale.

### Chapter 6.2 — Data

- **Real:** `diabetes_atlantic.shp` (666 counties; centroids `x`, `y` in EPSG:5070 metres) — `Dia_pct ~ Obesity + PhysInact + Acc_Exer + PCP_rate + FoodEnvIdx + SVI`.
- **Panel:** `data_1998_2010_long_lbc.csv` (FIPS, X, Y, Year, SMOKING, POVERTY, PM25, NO2, SO2, RATE; **1998–2012**) for GTWR.
- **Simulated (API demos only):** `load_georgia`, `load_londonhouse`, `load_beijing_pm25`, `load_ncovid` — never report results from them as real.

### Chapter 6.3 — Kernels, Bandwidths and the Fitting Pattern

- Kernels `gaussian`, `bisquare`, `tricube`, `exponential`, `boxcar`, `triangular`; `SpatialKernel`, `AnisotropicKernel`; fixed (distance) vs adaptive (k-nearest) bandwidth.
- `BandwidthSelector` (golden section, grid search; AICc, AIC, BIC, CV); `visualization.plot_bandwidth_search`.
- Geometry input: an `(n, 2)` array or GeoDataFrame in a **projected CRS** (bandwidths are in coordinate units).

### Chapter 6.4 — Exploratory: GWSS

- Local mean, standard deviation, skewness and correlation; `GWSS.to_dataframe()`; map local means before any regression.

### Chapter 6.5 — GWR and Inference

- `GWR(bandwidth, fixed, kernel, criterion).fit(X, y, geometry)` (`bandwidth='auto'`); `coef_`, `std_err_`, `t_values_`, `p_values_`, `fitted_`, `residuals_`, `r2_local_`, `bandwidth_`, `aicc_`, `effective_df_`; `summary()`.
- **Local significance and multiplicity:** `local_significance(alpha)`, `utils.bh_correction`, `t_to_p` — hundreds of local tests need adjustment.
- Coefficient maps `plot_local_coefficients(model, gdf, covariate, significance=True)`; prediction at new locations; **spatial** train/test splits, not row order.

### Chapter 6.6 — Advanced Variants

- **MGWR:** a bandwidth per covariate by back-fitting; read each bandwidth as the range of that process.
- **MixedGWR:** `fixed_vars` (global) vs `varying_vars` (local).
- **GWGLM:** gaussian, poisson (counts), binomial (prevalence). **No `offset` argument**, so exposure-adjusted rates (log E) cannot be modelled directly — say so, and model the rate with another family or list it as a package gap.

### Chapter 6.7 — Diagnostics and Model Checking

- Residual autocorrelation: `diagnostics.moran_test` vs `esda.Moran` on the same residuals (they should agree); non-stationarity: `f3_test`, `monte_carlo_variability`, `GWR.monte_carlo_test`; local collinearity: `local_vif`, `local_condition_number`; `visualization.plot_residuals`, local R².

### Chapter 6.8 — Comparing Global, Spatial and Local Models

- Table OLS | SLM | SEM (Part 3) | GWR | MGWR: AICc, log-likelihood-equivalent, residual Moran's I, range of local R². Lag models capture spillover; GWR captures heterogeneity — use both when appropriate; compare information criteria across estimators with caution.

### Chapter 6.9 — Machine-Learning and Ensemble GW Models

- `GWRandomForest` (bandwidth by a Moran's I criterion), `GWXGBoost`, `GWLightGBM`; `feature_importance_map`, `visualization.plot_local_importance`; optional SHAP.
- Validate with **spatial block CV** (`interpolate.cross_validate(method='spatial_block')` shows the idea), not random k-fold — autocorrelation leaks information between folds. GWR vs ML: interpretability against predictive accuracy.

### Chapter 6.10 — Spatio-Temporal and Multivariate

- **GTWR:** `fit(X, y, geometry, times)`, spatial bandwidth and time scaling λ, prediction at new (place, time). **GWPCA:** local loadings, scores, variance explained, robust mode.

### Chapter 6.11 — Deep and Network Models

- **GNNWR** (`hidden_layers`, `n_epochs`, `batch_size`, `learning_rate`, `patience`, `device`), **GTNNWR**, **NetworkGWR** (shortest-path distance via `networkx`). Guard every cell with an availability check: without PyTorch, constructing `GNNWR()` works but `fit()` raises an `ImportError` naming the package (Part 0 §0.1).

### Chapter 6.12 — Performance and Scaling

- `set_compute_options(n_jobs, device)`, `get_compute_config`, `ComputeConfig`. One local fit per location plus an n × n distance matrix: **time real fits at n = 217, 666, 3,107** instead of quoting big-O. GPU for deep models.

### Chapter 6.13 — Model Selection and Troubleshooting

- Table: research question → model (inference, prediction, scale of processes, space–time); singular matrices, bandwidth at bounds, slow fits, missing optional dependencies.

### Chapter 6.14 — Exercises

1. Fit GWR on `diabetes_atlantic`, map the `PhysInact` coefficient, and find where it is not significant after FDR correction.
2. Compare residual Moran's I for OLS and GWR.
3. Compare the bandwidth and coefficient maps for `kernel='bisquare'` and `'gaussian'`.
4. Time GWR at three sample sizes and extrapolate.

---

## Part 7 — Spatial Interpolation  ✅ (`spatialstats.interpolate`)

**Objectives.** Predict a continuous surface from point measurements; compare deterministic and geostatistical interpolators with honest validation; quantify uncertainty; transfer data between incompatible zone systems.
**Prerequisites.** Part 0. **Notebook.** `ch07_spatial_interpolation`.
Kriging is included although the module's original stub did not list it: without it there is no prediction variance.

**Data.**
- `CA_pm25_2025.csv` (59,566 daily records from 163 monitors): aggregate to site means (keep sites with ≥ 100 observations → 159), project lon/lat to a metric CRS (EPSG:3310 California Albers or 5070).
- `CA_pm25_grid_predictions.csv` (196,764 rows: `x`, `y`, `month`, `pm25_pred`, `pm25_sd`): a pre-computed surface with uncertainty, useful for comparison; its provenance is undocumented, so do not call it "truth".
- **Areal transfer uses the Northeast, not California:** `diabetes_northeast.shp` counties, with `cancer_points.csv` (population-weighted points for those same 217 counties) as the ancillary layer.

**Reference results** (validation runs).

| Check | Result |
|---|---|
| Leave-one-out RMSE (159 sites) | ordinary kriging 1.71 < thin-plate spline 1.82 ≈ IDW (p = 2) 1.83 < TIN 1.90 < nearest 2.13 |
| Spatial-block CV (120 km blocks, 8 folds) | every method 2.7–2.9, R² ≈ 0 — with a ~130 km range, holding out whole neighbourhoods leaves almost no skill |
| Auto-fitted variogram | spherical, nugget ≈ 1.21, partial sill ≈ 6.74, range ≈ 133 km |
| Kriging-variance calibration (standardised LOO errors) | mean −0.03, s.d. 0.90 (about 97% within ±1.96) |
| Kriging vs `pykrige` | agree to 1e-14 (mind that `pykrige` takes the *total* sill) |
| Areal transfer, 60 coarse zones → 217 counties, extensive `Dia_count` | area weighting RMSE 24,427 (r = 0.64) vs dasymetric with real `cancer_points` 10,464 (r = 0.93); totals conserved |

### Chapter 7.1 — Introduction and Concepts

- Continuous field vs sampled points; smoothness and stationarity; exact vs smoothing interpolators; deterministic methods give no uncertainty. The variogram of Part 0 §0.3.

### Chapter 7.2 — Data and Preparation

- Aggregate and project as above; **a projected CRS is required** for every distance-based method (Part 0); make prediction grids with `make_grid(region, resolution)`; the pre-computed surface (provenance caveat).

### Chapter 7.3 — Deterministic Interpolation

- `NearestNeighbor` (Voronoi), `TIN` (Delaunay linear — the linear analogue of natural neighbour; Sibson is not implemented), `IDW(power, k, radius, smoothing)`, `RBF` / `Spline` (thin-plate; smoothing for noisy data).
- One interface for all: `fit(coords, values).predict(new)` and `predict_grid`; `clone()` / `get_params()`.

### Chapter 7.4 — Geostatistical Interpolation (new)

- **Theory:** the semivariogram γ(h) = ½ Var[Z(s + h) − Z(s)] = C(0) − C(h); nugget, partial sill, range; spherical, exponential and Gaussian models with a *practical* range.
- `empirical_variogram` (Matheron), `fit_variogram` (Cressie-weighted least squares), `fit_variogram_auto`; plot with `VariogramModel.plot(empirical)`.
- **Ordinary kriging** (`OrdinaryKriging(variogram, n_neighbors)`): the BLUP under an unknown constant mean, weights from the variogram (clustered stations are down-weighted), **prediction variance** available and independent of the observed values; local kriging for larger n; exact at data points.
- **Calibration check:** standardised leave-one-out errors should have mean 0 and standard deviation 1.

### Chapter 7.5 — Model Validation

- `cross_validate` with `loo`, `kfold`, `spatial_block(block_size=…)`; RMSE, MAE, mean error, R², residual maps; `grid_search` (e.g. IDW power and k) and `compare_methods` on the same folds.
- **Lesson:** leave-one-out is optimistic for clustered networks (every held-out site has close neighbours); spatial blocks mimic prediction into unsampled areas (links to Part 8).

### Chapter 7.6 — Covariate-Assisted Interpolation

- GWR / GW ensemble `predict` at grid points as an interpolator with covariates (Part 6); when covariates beat IDW; regression kriging is not implemented (residual kriging can be done by hand).

### Chapter 7.7 — Areal Interpolation

- `areal_weighting(source, target, extensive=…, intensive=…)`: uniform density inside each zone; extensive values are split by area share, intensive ones averaged with area weights; the CRS must be projected and identical.
- `dasymetric(source, target, ancillary_points, weight=…)`: allocate by ancillary weight; zones without points fall back to area weighting so totals are conserved. Show the RMSE comparison above and the conservation check.
- Conservation of totals (pycnophylactic property) vs smoothness.

### Chapter 7.8 — Exercises

1. Tune the IDW power and neighbour count by spatial CV; compare with kriging and with the pre-computed surface.
2. Refit the variogram with each model; which fits best and does the LOO error change?
3. Map the kriging standard deviation; where is it largest and why?
4. Aggregate the Northeast to states and back to counties with `areal_weighting`: conservation, then error.
5. Why is leave-one-out CV over-optimistic here? Show it.

---

## Part 8 — Spatial Sampling and Design  ✅ (`spatialstats.sampling`)

**Objectives.** Design spatial samples; understand how spatial autocorrelation changes the effective sample size; compare designs and estimators against a known truth.
**Prerequisites.** Part 0; Part 2 (effective sample size); Part 7 for 8.4. **Notebook.** `ch08_spatial_sampling`.

**The "known population" laboratory.** All 3,107 US counties in `diabetes_usa.csv` (columns `X`, `Y`, `Diabetes_per`, `Urban_Rural`, `State`, `POP_Total`) form a census: the true mean is known, so every design can be scored by repeated sampling.

**Reference results** (n = 100, `Diabetes_per`, 800 replicates per design).

| Design | Monte Carlo s.d. | Design effect vs SRS | Spatial balance B (lower is better) | CI coverage |
|---|---|---|---|---|
| Simple random | 0.157 | 1.00 | 0.33 | ≈ 94% |
| Systematic, Hilbert order | 0.115 | 0.54 | 0.08 | ≈ 98% (conservative) |
| GRTS | 0.130 | 0.69 | 0.11 | ≈ 94% |
| Stratified (urban/rural) + GRTS within | 0.127 | 0.65 | 0.15 | ≈ 98% |
| Two-stage, 12 states × 8 counties | 0.319 | 4.1 | — | ≈ 93% |
| Two-stage, 25 states × 4 counties | 0.213 | 1.8 | — | ≈ 97% |

- On a spatially **unstructured** variable GRTS gives no gain (design effect ≈ 1.02) with correct coverage — balance only helps when the variable is spatially structured.
- PPS-GRTS (`size=POP_Total`) has exact inclusion probabilities; the Horvitz–Thompson total was unbiased to 0.3%.
- The local-mean standard error tracked the true spread for GRTS (0.127 vs 0.130).

### Chapter 8.1 — Foundations

- Design-based vs model-based inference; inclusion probabilities; spatial balance.
- **Effective sample size:** why neighbouring samples are partly redundant — `effective_sample_size(n, ρ, kind='ar1' | 'equicorrelated')` (ρ = 0.5, n = 100 → about 33); connect to Part 2. Design effect `design_effect`.

### Chapter 8.2 — Sampling Designs

- **Classical:** `simple_random`; `systematic` (Hilbert order gives an evenly spread sample, x-order gives transects); `grid_points` for a continuous region (random-origin square or hexagonal lattice).
- **Stratified and cluster:** `stratified` by `Urban_Rural` or state, allocation `proportional` / `equal` / `neyman` (needs stratum s.d.), `within='grts'`; `two_stage`.
- **Spatially balanced:** `grts` (random hierarchical address ordering + systematic PPS; exact first-order probabilities), `size=` for unequal probabilities.
- **Model-based (not probability samples):** `space_filling` (`kmeans` or `maximin`) and `clhs` (conditioned Latin hypercube on `Obesity`, `SVI`, `PhysInact`) — covariate-space coverage, no inclusion probabilities, so no design-based estimation.
- `sample_size_mean(sd, margin, N, conf, deff)`.

### Chapter 8.3 — Estimation and Variance

- `estimate_mean`: Horvitz–Thompson / Hájek mean and total; variance chosen by design — SRS, stratified, **local-mean** for balanced/systematic samples (neighbours in the sample are similar when y is spatially structured), ultimate-cluster with ratio linearisation for two-stage.
- Estimating the design effect and CI coverage by simulation with `evaluate_design` / `compare_designs`.

### Chapter 8.4 — Monitoring-Network Design

- Placing new PM2.5 monitors in California (Part 7 data): gaps in the 163 sites; `coverage_metrics` (mean and maximum distance to the nearest site, minimum spacing, balance) for `space_filling` vs current sites vs random.
- **Sequential selection** by maximum kriging variance (`OrdinaryKriging.predict(..., return_std=True)`) or maximin distance — a loop in the notebook; not a package function.

### Chapter 8.5 — Exercises

1. Repeat 1,000 draws for SRS, stratified and balanced designs; plot RMSE against n.
2. Show that GRTS gains nothing for a spatially random variable.
3. Compare Neyman with proportional allocation using a proxy for stratum s.d.
4. How large must n be for a ±0.25-point margin under SRS and under GRTS?
5. Why can two-stage sampling have a design effect above 1, and when does it pay off?

---

## Part 9 — Spatio-Temporal Analysis  🔲 (`spatialstats.spacetime` is a stub)

**Objectives.** Extend autocorrelation, clustering and regression to data indexed by place and time.
**Prerequisites.** Parts 2, 6. **Notebook.** `ch09_spatiotemporal` — write the chapters marked ✅ first; the rest waits on the module.
The stub's docstring lists the intended scope: space–time weights and autocorrelation, emerging hot spots, space–time KDE, ST kriging, spatial panel regression.

**What can be done now.** `esda.Moran` / `Moran_Diff` and `cluster.getis_ord_gi` year by year; `gwmodel.GTWR`; Mann–Kendall by `scipy` on the per-county z-score series; time marks from `crime_data_2018.csv`.

### Chapter 9.1 — Concepts and Data

- Space–time dependence; separable vs non-separable structure; time as a third axis or its own; the locations × time "cube".
- **Data:** `data_1998_2010_long_lbc.csv` — 3,107 counties × 15 years (**1998–2012**; the file name says 2010): `RATE`, `SMOKING`, `POVERTY`, `PM25`, `NO2`, `SO2`; `crime_data_2018.csv` — event timestamps (`date_incid`, `hour_of_day`, `day_of_week`, `incident_type`), coordinates in the mis-zoned CRS (C9), so use `long`/`lat`.

### Chapter 9.2 — Space–Time Weights and Global Analysis

- Space–time neighbours: spatial neighbours × temporal lags (Kronecker constructions) 🔲. Space–time Moran's I and the **Knox test** for event interaction 🔲.
- ✅ Baseline: year-by-year Moran's I and `Moran_Diff`.

### Chapter 9.3 — Local and Emerging Hot Spots

- ✅ Per-year Gi\* (`getis_ord_gi`); Mann–Kendall trend of each county's z-score series; categories new, persistent, intensifying, diminishing, sporadic, oscillating 🟡 (assemble by hand); multiple testing over space × time.

### Chapter 9.4 — Space–Time Point Patterns and Density

- Space–time KDE for events (hour of day × location), separable vs product kernels, animation frames 🔲 (space-only KDE from Part 1 per time slice is available).

### Chapter 9.5 — Space–Time Interpolation

- Extending Part 7 with a time dimension; the space–time variogram 🔲.

### Chapter 9.6 — Regression with Space–Time Data

- **Spatial panel regression** (fixed/random effects with spatial lag or error) 🔲.
- ✅ **GTWR** (`gwmodel.GTWR`): time scaling λ, bandwidths, coefficient surfaces over time; compare pooled OLS, panel and GTWR.

### Chapter 9.7 — Exercises

- Which counties show *intensifying* hot spots of `RATE` between 1998 and 2012?
- How does the Moran's I of `RATE` evolve, and is the change significant (`Moran_Diff`)?

---

## Part 10 — Capstone: End-to-End Workflow  🟡

**Goal.** One coherent health-geography study using the whole package on `diabetes_atlantic`. **Notebook.** `ch10_capstone`.

1. **Data and CRS** (Part 0): audit with `load_study_area`, reproject, map.
2. **Weights** (Part 2): Queen + KNN sensitivity, island handling.
3. **Autocorrelation** (Part 2): global and local; Moran scatterplot.
4. **Clusters** (Part 4): Gi\* and LISA agreement; regionalisation.
5. **Explanation** (Part 3): OLS → LM tests → SLM / SEM.
6. **Heterogeneity** (Part 6): GWR / MGWR; local coefficient maps; diagnostics.
7. **Rates and uncertainty** (Part 5): SMR → EB → BYM; probability maps.
8. **Prediction** (Parts 6 and 7): GW ensemble or interpolation for new places; spatial CV.
9. **Reporting:** a decision log (weights, α, correction, permutations, seeds), the reproducible notebook, a one-page summary figure. (Space–time step deferred until `spacetime` exists.)

### Choosing a method — decision map

| If you want to… | Use |
|---|---|
| Set up, audit and map spatial data | Part 0 |
| Describe event locations | Part 1 |
| Test whether values are spatially dependent | Part 2 |
| Explain an outcome accounting for spatial dependence | Part 3 |
| Locate clusters or hot spots | Part 4 |
| Map rates for small areas reliably | Part 5 |
| Let relationships vary over space | Part 6 |
| Predict a surface / move data across zones | Part 7 |
| Design a survey or monitoring network | Part 8 |
| Work with time | Part 9 |

---

## Appendices

### A. Module and API map

| Module | Status | Tutorial parts | API |
|---|---|---|---|
| `core` | ✅ | 0 | see "Implemented API" above |
| `weights` | ✅ | 2–5 | " |
| `esda` | 🟡 (C1–C3) | 2–5, 9 | " |
| `viz` | ✅ | all | " |
| `datasets` | ✅ | 0, 2, 5, 6 | " |
| `pointpattern` | ✅ | 1 | " |
| `regression` | ✅ | 3 | " |
| `cluster` | ✅ | 4 | " |
| `bayes` | ✅ | 5 | " |
| `gwmodel` | ✅ | 6, 9 | " |
| `interpolate` | ✅ | 7 | " |
| `sampling` | ✅ | 8 | " |
| `spacetime` | 🔲 stub | 9 | — |

### B. Dataset catalogue

Geography matters when pairing datasets: files in the same row of "Shares geography with" can be combined; others cannot.

| File | Rows | Declared CRS | Key fields | Shares geography with | Used in |
|---|---|---|---|---|---|
| `diabetes_northeast.shp/.csv` | 217 counties | EPSG:5070 | `Dia_pct`, `Dia_count`, `POP_Total`, `Obesity`, `PhysInact`, `Acc_Exer`, `PCP_rate`, `FoodEnvIdx`, `SVI`, `UrbRural` (str) | `COUNTY_NORTHEAST`, `cancer_areas`, `cancer_points` | 0, 2, 4, 5, 7.7 |
| `diabetes_atlantic.shp/.csv` | 666 counties | EPSG:5070 | same | `COUNTY_ATLANTIC` | 3, 6, 10 |
| `diabetes_usa.csv` + `US_COUNTY.shp` | 3,107 counties | EPSG:5070 (shp) | same (CSV names differ: `Diabetes_per`, `X`, `Y`, `Urban_Rural`) | — | 2, 8 |
| `COUNTY_NORTHEAST/ATLANTIC.shp` | 217 / 666 | EPSG:5070 | `FIPS`, `STATE`, `COUNTY` | as above | boundaries |
| `cancer_areas.shp` / `cancer_data.gpkg` (layers `areas`, `points`) | 217 areas | custom Lambert conformal conic (no EPSG) | `FIPS` (int), `rate` (98–192) | Northeast | 4, 5 |
| `cancer_points.csv` | 5,419 points | same Lambert CC | `FIPS` (float), `POP10` | Northeast (91.5% of points fall inside their own county polygon) | 4.4, 5, 7.7 |
| `Crime_location_2018.shp` | 14,001 pts | **EPSG:3718 (UTM 11N) — wrong for Buffalo; use EPSG:26917** | `ID` | Buffalo | 1, 4.3 |
| `Crime_Incidents_2018.shp` | 14,036 pts | **none** (lon/lat degrees; set 4326 after checking) | `incident_type`, timestamps | Buffalo | 1, 9 |
| `crime_data_2018.csv` | 14,001 | lon/lat + `x_m`, `y_m` in the same mis-zoned UTM 11N | 8 incident types, hour, day | Buffalo | 1, 9 |
| `buffalo_city.shp` | 35 polygons | **EPSG:3718 — wrong zone**; `sqmiles` gives true areas | `nbhdname`, `sqmiles` | Buffalo | 1 |
| `CA_pm25_2025.csv` | 59,566 obs, 163 sites | lon/lat | `Daily_PM2.5_ug_m3`, `Daily_AQI`, `Date` | California | 7, 8.4 |
| `CA_pm25_grid_predictions.csv` | 196,764 | x/y | `month`, `pm25_pred`, `pm25_sd` | California | 7 |
| `data_1998_2010_long_lbc.csv` | 46,605 (3,107 × 15 yr, 1998–2012) | X/Y | `RATE`, `SMOKING`, `POVERTY`, `PM25`, `NO2`, `SO2` | US counties | 2, 6, 9 |
| `load_lattice`, `load_points_csr` | synthetic | — | — | — | 0, 1, 2, 5 |
| GW loaders (`load_georgia`, …) | **simulated** | — | — | — | 6 (API demos) |

**Data-hygiene notes for the tutorial.** Several CRS conventions, one file with none and one valid-but-wrong; FIPS stored as int / float (numeric merges work, string keys do not, leading zeros lost — use a canonical 5-digit string key);
`UrbRural` is a string; CSV and shapefile versions of the diabetes data name columns differently; 36% of the crime events repeat an earlier location.

### C. Implementation caveats to resolve or disclose

Each affects what the tutorial can truthfully say. **Status:** *open* = unresolved in the package; *mitigated* = the tutorial or the new modules work around it.

| # | Location | Observation | Status / effect on the tutorial |
|---|---|---|---|
| C1 | `esda/inference.py:conditional_permutation` | Two-tailed test compares `abs(sim) >= abs(obs)`. Geary's C has null 1, not 0, so C = 0.734 (positive autocorrelation) returns **p = 1.0**; Moran's I is also compared on the absolute value rather than |I − E[I]| | *Open.* Do not report Geary permutation p-values (Chapter 2.4) |
| C2 | same | Permutes **all** of `y` even for local statistics — unconditional randomisation, not the conditional permutation the name implies | *Mitigated in Part 4* by `cluster` (correct scheme). Independent check: `esda`'s own conditional permutation flagged 46 counties at one-sided p ≤ 0.05; the same test in `cluster.local_moran_clusters` gave 46 at two-sided p ≤ 0.10; the package's `Moran_Local` gave 32 |
| C3 | `esda/local_.py:Moran_Local` | Default `fdr=True` with a permutation p-value floor of 1/(m+1) gives **0 significant counties** (`fdr=False`, 999 permutations: 29 — 16 HH, 10 LL, 2 LH, 1 HL) | *Open.* Teach both settings and the resolution limit (Chapter 2.5) |
| C4 | `viz/moran.py` | Axis label says "standardized attribute" but the code only mean-centres | *Open;* state it in the text |
| C5 | `esda/global_.py:_geary_C`, `_general_G` | Python loops over non-zero weights — slow at n ≈ 3,000 | *Open;* note runtime in Part 2 |
| C6 | `datasets` GW loaders | Simulated, not the real datasets their names suggest; the wrappers in `datasets` have no docstring (the originals in `gwmodel.datasets` say "Simulated …") | *Mitigated:* label every use as synthetic |
| C7 | `pyproject.toml` | `bayes` extra added; PyMC imported lazily | *Resolved.* Guard optional cells |
| C8 | environment | `.venv` lacks geopandas; system Python lacks some extras | *Open:* document one reproducible environment (Python 3.11, extras `viz`, `bayes`, `libpysal`) |
| C9 | `Crime_location_2018.shp`, `buffalo_city.shp`, `crime_data_2018.csv` | Declared UTM 11N (114°–120°W) but Buffalo is at 78.9°W: distances × 1.121 (formula 1.1209), areas × 1.257, grid rotation 26.7° (formula 26.7°); geodesic area 105.8 km² equals the layer's own `sqmiles` | *Mitigated:* reproject via lon/lat to EPSG:26917 (Part 0 §0.5); README example updated. Any number computed in the declared CRS is wrong |
| C10 | `core/data.py:SpatialData.to_crs / ensure_projected` | A `coords` role that names columns (set by the user or by `from_frame`) is copied unchanged, so `coords_array()` returns stale coordinates | *Open;* demonstrated in Part 0 §0.4. Suggested fix: drop the role on reprojection |
| C11 | `core/data.py:SpatialData.coords_array` | Docstring says centroids; the code returns representative points (mean 3.6 km, max 32.8 km from the centroid on the Northeast counties) | *Open;* documented in Part 0 §0.4 |
| C12 | `viz/choropleth.py:_classify` | Without `mapclassify`, `natural_breaks` silently falls back to quantiles; the colour bar shows the class index, not data values | *Open;* demonstrated in Part 0 §0.8 |
| C13 | `ensure_projected` default | Targets Web Mercator (EPSG:3857), which inflates distances by sec φ and areas by sec²φ | *Mitigated:* always pass `target_crs=` (Part 0 §0.5) |
| C14 | `bayes` PyMC backend | Needs a connected adjacency (no islands); PyMC's `ICAR` is unnormalised in its scale | *Mitigated* in the package (fixed-scale ICAR × scale); islands → use `backend='gibbs'` |

### D. Build order and next steps

1. **Part 0** — done. Reuse its audit recipe at the top of every notebook.
2. **Part 2** (weights, then autocorrelation) — after deciding how to handle C1–C3 (see "Decisions needed"). Part 4 can start in parallel: `cluster` already uses the correct inference.
3. **Part 3** and **Part 5** (start with the empirical-Bayes simulation study, then BYM) — the validation studies above are the notebook skeletons.
4. **Part 1** (needs the corrected Buffalo CRS from step 1), then **Parts 7 and 8** (8 reuses the county census; 8.4 reuses Part 7's California data).
5. **Part 6** (reuse `TUTORIAL.md`; time real fits) and **Part 10**.
6. **Part 9** once `spacetime` lands; write the ✅ chapters earlier if wanted.

**Decisions needed.**
- Fix the `esda` inference (C1–C3) before Part 2, or write Part 2 around them and point to `cluster`?
- Promote the Part 0 audit recipe (`crs_audit`, `load_study_area`) into `spatialstats.core`, and fix `SpatialData` (C10, C11)?
- Vendor MathJax (≈ 2 MB) into `docs/notebooks/` so the HTML renders offline, or keep the CDN?
- Add a repository script for the execute → render → alt-text → verify pipeline so every chapter is built the same way?

### E. Key references

- Tobler (1970) A computer movie simulating urban growth in the Detroit region
- Snyder (1987) *Map Projections — A Working Manual*; Karney (2013) Algorithms for geodesics; Sinnott (1984) Virtues of the haversine; Openshaw (1984) *The Modifiable Areal Unit Problem*; Fisher (1958); Jenks (1967)
- Anselin (1988) *Spatial Econometrics*; Anselin (1995) Local indicators of spatial association; Anselin, Syabri & Kho (2006) GeoDa
- Getis & Ord (1992); Ord & Getis (1995); Cliff & Ord (1981) *Spatial Processes*; Moran (1950); Geary (1954)
- LeSage & Pace (2009) *Introduction to Spatial Econometrics*
- Baddeley, Rubak & Turner (2015) *Spatial Point Patterns*; Diggle (2013) *Statistical Analysis of Spatial and Spatio-Temporal Point Patterns*; Ripley (1976); Donnelly (1978)
- Kulldorff (1997) A spatial scan statistic; Assunção et al. (2006) SKATER
- Marshall (1991) Mapping disease and mortality rates using empirical Bayes estimators; Clayton & Kaldor (1987); Besag, York & Mollié (1991); Riebler et al. (2016) BYM2; Sørbye & Rue (2014) Scaling intrinsic GMRF priors
- Brunsdon, Fotheringham & Charlton (1996) GWR; Fotheringham, Yang & Kang (2017) MGWR; Huang, Wu & Barry (2010) GTWR; Du et al. (2020) GNNWR
- Cressie (1993) *Statistics for Spatial Data*; Brus (2022) *Spatial Sampling with R*; Stevens & Olsen (2004) Spatially balanced sampling of natural resources; Minasny & McBratney (2006) cLHS
- Bivand, Pebesma & Gómez-Rubio (2013) *Applied Spatial Data Analysis with R*; Rey & Anselin (2007) PySAL

### F. Cross-checking against PySAL

For every ✅ result in Parts 2 and 4, run the equivalent `libpysal` / `esda` call (via `W.to_libpysal()`) and report the agreement in a validation notebook or box. It is also the fastest way to confirm or refute the caveats in Appendix C (the C2 comparison above was done this way).
