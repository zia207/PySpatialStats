"""
Tests for the spatial-statistics modules: pointpattern, regression, cluster, bayes,
sampling, interpolate.

Run with pytest, or standalone: ``PYTHONPATH=. python tests/test_spatial_modules.py``.

Most tests use synthetic data with a known truth (fixed seeds). Cross-checks against
reference libraries (spreg, pykrige, pointpats, esda, PyMC) run only if those are installed.
"""

import importlib
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def skip(msg):
    raise unittest.SkipTest(msg)


def need(module):
    try:
        return importlib.import_module(module)
    except Exception:
        skip(f"{module} not installed")


def lattice(n=12):
    """n×n queen-contiguity lattice: (GeoDataFrame, W, coords)."""
    from spatialstats import Queen
    from spatialstats.datasets import load_lattice
    gdf, _ = load_lattice(n=n, seed=0)
    return gdf, Queen(gdf), gdf[["x", "y"]].to_numpy()


# =============================================================================
# pointpattern
# =============================================================================

def test_window_and_pattern():
    from shapely.geometry import Polygon
    from spatialstats.pointpattern import Window, PointPattern
    w = Window.from_bounds(0, 0, 2, 1)
    assert w.is_rectangle and np.isclose(w.area, 2.0)
    assert list(w.contains(np.array([[1, .5], [3, .5]]))) == [True, False]
    assert np.isclose(w.set_covariance(np.array([[0.0, 0.0]]))[0], 2.0)
    assert np.isclose(w.set_covariance(np.array([[0.5, 0.25]]))[0], 1.5 * 0.75)
    L = Window(Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]))
    assert not L.is_rectangle and np.isclose(L.area, 3.0)
    assert np.isclose(L.set_covariance(np.array([[0.0, 0.0]]))[0], 3.0, rtol=0.02)
    pts = np.random.default_rng(0).uniform(0, 1, (50, 2))
    pp = PointPattern(pts, window=Window.from_bounds(0, 0, 1, 1))
    assert pp.n == 50 and np.isclose(pp.intensity, 50)
    try:
        PointPattern(np.array([[5.0, 5.0]]), window=Window.from_bounds(0, 0, 1, 1))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    dup = PointPattern(np.vstack([pts, pts[:5]]), window=pp.window)
    assert dup.n_duplicates == 5 and dup.deduplicate().n == 50


def test_k_function_edge_correction_is_unbiased():
    from spatialstats.pointpattern import Window, simulate_csr, ripley_k
    win = Window.from_bounds(0, 0, 1, 1)
    r = np.linspace(0, 0.25, 11)
    trans = np.mean([ripley_k(simulate_csr(n=80, window=win, seed=s), r).estimate for s in range(120)], axis=0)
    none = np.mean([ripley_k(simulate_csr(n=80, window=win, seed=s), r, "none").estimate for s in range(120)], axis=0)
    theo = np.pi * r ** 2
    assert abs(trans[-1] / theo[-1] - 1) < 0.04, trans[-1] / theo[-1]
    assert none[-1] / theo[-1] < 0.9              # uncorrected estimator is biased low


def test_polygon_window_k_matches_rectangle_approach():
    from shapely.geometry import Polygon
    from spatialstats.pointpattern import Window, simulate_csr, ripley_k
    L = Window(Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]))
    r = np.linspace(0, 0.3, 9)
    K = np.mean([ripley_k(simulate_csr(n=150, window=L, seed=s), r).estimate for s in range(60)], axis=0)
    assert abs(K[-1] / (np.pi * r[-1] ** 2) - 1) < 0.08


def test_csr_tests_size_and_power():
    from spatialstats.pointpattern import Window, simulate_csr, simulate_thomas, simulate_matern_inhibition, \
        envelope, clark_evans, quadrat_test
    win = Window.from_bounds(0, 0, 1, 1)
    rej = lambda gen: np.mean([envelope(gen(s), "L", nsim=39, seed=100 + s).test("mad") <= 0.1 for s in range(20)])  # noqa: E731
    assert rej(lambda s: simulate_csr(n=80, window=win, seed=s)) <= 0.3
    assert rej(lambda s: simulate_thomas(12, 0.03, 7, win, seed=s)) >= 0.9
    assert rej(lambda s: simulate_matern_inhibition(300, 0.05, win, seed=s)) >= 0.9
    clustered = simulate_thomas(12, 0.03, 7, win, seed=3)
    ce = clark_evans(clustered)
    assert ce["R"] < 0.7 and ce["p_clustered"] < 0.01
    q = quadrat_test(clustered, 4, 4, permutations=99, seed=1)
    assert q["p_clustered"] < 0.01 and q["index_of_dispersion"] > 1.5
    q0 = quadrat_test(simulate_csr(n=200, window=win, seed=9), 4, 4)
    assert q0["p_value"] > 0.02


def test_nn_functions_direction():
    from spatialstats.pointpattern import Window, simulate_thomas, g_function, f_function, j_function
    pp = simulate_thomas(12, 0.03, 7, Window.from_bounds(0, 0, 1, 1), seed=3)
    g, f, j = g_function(pp), f_function(pp), j_function(pp)
    sl = slice(10, 50)
    assert np.mean(g.estimate[sl] > g.theoretical[sl]) > 0.9       # clustering: G above CSR
    assert np.mean(f.estimate[sl] < f.theoretical[sl]) > 0.9       # F below CSR
    assert np.nanmean(j.estimate[sl] < 1) > 0.9                    # J < 1


def test_kde_integrates_to_n_and_pair_correlation():
    from shapely.geometry import Polygon
    from spatialstats.pointpattern import Window, simulate_csr, simulate_thomas, kde, pair_correlation, \
        k_inhom, PointPattern, intensity_at_points
    L = Window(Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]))
    pp = simulate_csr(n=300, window=L, seed=2)
    k = kde(pp, bandwidth=0.15, grid_size=150)
    assert abs(k.integral / pp.n - 1) < 0.06
    assert np.isfinite(k.at(pp.coords)).all()
    th = simulate_thomas(12, 0.03, 7, Window.from_bounds(0, 0, 1, 1), seed=3)
    g = pair_correlation(th)
    assert g.estimate[:5].mean() > 3                               # strong short-range excess
    # inhomogeneous K of an inhomogeneous Poisson process ≈ pi r^2 (homogeneous K would be inflated)
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 1, (4000, 2))
    keep = rng.uniform(size=4000) < pts[:, 0]                      # intensity increases with x
    pi_ = PointPattern(pts[keep], window=Window.from_bounds(0, 0, 1, 1))
    r = np.linspace(0, 0.1, 6)
    ki = k_inhom(pi_, intensity=pts[keep][:, 0] * 4000, r=r)
    assert abs(ki.estimate[-1] / ki.theoretical[-1] - 1) < 0.1


def test_pointpats_crosscheck():
    pointpats = need("pointpats")
    from spatialstats.pointpattern import Window, simulate_thomas, ripley_k
    pp = simulate_thomas(12, 0.03, 7, Window.from_bounds(0, 0, 1, 1), seed=3)
    r = np.linspace(0.01, 0.2, 10)
    # pointpats' K (no edge correction) uses the same estimator, but the bounding box of the points as the window
    hull_win = Window.from_points(pp.coords)
    from spatialstats.pointpattern import PointPattern
    same = PointPattern(pp.coords, window=hull_win)
    support, k_ref = pointpats.k(pp.coords, support=r, edge_correction=None)
    k_me = ripley_k(same, r, correction="none").estimate
    assert np.allclose(np.asarray(support), r) and np.allclose(np.asarray(k_ref), k_me, rtol=0.03)


# =============================================================================
# regression
# =============================================================================

def _sar_data(n=15, rho=0.5, seed=0, error_model=False):
    from spatialstats.regression import OLS  # noqa: F401
    _, w, xy = lattice(n)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(w.n, 2))
    eps = rng.normal(size=w.n)
    W = w.sparse.toarray()
    A = np.linalg.inv(np.eye(w.n) - rho * W)
    if error_model:
        y = 1 + X @ [1.0, -0.5] + A @ eps
    else:
        y = A @ (1 + X @ [1.0, -0.5] + eps)
    return y, X, w


def test_ols_matches_numpy_and_diagnostics():
    from spatialstats.regression import OLS
    y, X, w = _sar_data(rho=0.0)
    ols = OLS(y, X, w=w, names=["a", "b"])
    Xc = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    assert np.allclose(ols.beta, b)
    assert np.isclose(ols.r2, 1 - np.sum((y - Xc @ b) ** 2) / np.sum((y - y.mean()) ** 2))
    assert ols.names == ["const", "a", "b"]
    df = ols.diagnostics()
    assert {"Jarque-Bera (normality)", "Moran's I (residuals)", "LM-Lag"} <= set(df["diagnostic"])
    assert "Ordinary Least Squares" in ols.summary()


def test_input_validation():
    from spatialstats.regression import OLS, SpatialLag
    y, X, w = _sar_data(n=8)
    for bad in (lambda: OLS(y, np.column_stack([X, X[:, 0]])),                 # collinear
                lambda: OLS(y[:-1], X),                                          # length mismatch
                lambda: OLS(np.where(np.arange(len(y)) == 0, np.nan, y), X),   # NaN
                lambda: SpatialLag(y[:-3], X[:-3], w)):                         # W size mismatch
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_lm_tests_and_model_choice():
    from spatialstats.regression import OLS
    y, X, w = _sar_data(n=20, rho=0.6, seed=1)
    lm = OLS(y, X, w=w).lm_tests()
    assert lm["p_lm_lag"] < 1e-4 and lm["p_lm_error"] < 1e-4
    assert lm.recommendation().startswith("Spatial lag")
    y0, X0, _ = _sar_data(n=20, rho=0.0, seed=2)
    assert OLS(y0, X0, w=w).lm_tests().recommendation().startswith("OLS")
    ye, Xe, _ = _sar_data(n=20, rho=0.6, seed=3, error_model=True)
    assert OLS(ye, Xe, w=w).lm_tests().recommendation().startswith("Spatial error")


def test_spatial_lag_and_error_estimation():
    from spatialstats.regression import OLS, SpatialLag, SpatialError, compare_models, lr_test
    y, X, w = _sar_data(n=20, rho=0.6, seed=1)
    ols, slm, sem = OLS(y, X, w=w), SpatialLag(y, X, w), SpatialError(y, X, w)
    assert abs(slm.rho - 0.6) < 0.1 and slm.spatial_p < 1e-6
    assert np.allclose(slm.beta[1:], [1.0, -0.5], atol=0.3)
    assert slm.loglik > ols.loglik and slm.aic < ols.aic
    assert lr_test(ols, slm)["p_value"] < 1e-6
    tab = compare_models({"OLS": ols, "SLM": slm, "SEM": sem})
    assert tab.loc["OLS", "resid_moran_p"] < 1e-3 < tab.loc["SLM", "resid_moran_p"]   # dependence removed
    ye, Xe, _ = _sar_data(n=20, rho=0.6, seed=3, error_model=True)
    sem_e = SpatialError(ye, Xe, w)
    assert abs(sem_e.lam - 0.6) < 0.12 and sem_e.aic < SpatialLag(ye, Xe, w).aic + 5
    assert np.allclose(sem_e.beta[1:], [1.0, -0.5], atol=0.2)


def test_impacts_and_durbin():
    from spatialstats.regression import SpatialLag, SpatialDurbin, SLX
    y, X, w = _sar_data(n=15, rho=0.5, seed=4)
    slm = SpatialLag(y, X, w)
    imp = slm.impacts(nsim=100, seed=1)
    assert np.allclose(imp["total"], slm.beta[1:] / (1 - slm.rho))            # exact for row-standardised W
    assert np.allclose(imp["direct"] + imp["indirect"], imp["total"])
    assert (imp["indirect"].abs() > 0).all() and (imp["direct_lo"] <= imp["direct"]).all()
    sdm = SpatialDurbin(y, X, w)
    assert sdm.k == 5 and len(sdm.impacts(nsim=50, seed=1)) == 2
    assert SLX(y, X, w).k == 5


def test_spreg_crosscheck():
    spreg = need("spreg")
    libpysal = need("libpysal")
    from spatialstats.regression import OLS, SpatialLag, SpatialError
    gdf, w, _ = lattice(14)
    y, X, _ = _sar_data(n=14, rho=0.5, seed=5)
    lw = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    lw.transform = "r"
    ref = spreg.OLS(y.reshape(-1, 1), X, w=lw, spat_diag=True)
    ols = OLS(y, X, w=w)
    lm = ols.lm_tests().table.set_index("test")["statistic"]
    assert np.allclose([lm["LM-Lag"], lm["Robust LM-Lag"], lm["LM-Error"], lm["Robust LM-Error"], lm["LM-SARMA"]],
                       [ref.lm_lag[0], ref.rlm_lag[0], ref.lm_error[0], ref.rlm_error[0], ref.lm_sarma[0]], rtol=1e-6)
    rl = spreg.ML_Lag(y.reshape(-1, 1), X, w=lw)
    slm = SpatialLag(y, X, w)
    assert abs(slm.rho - rl.rho) < 1e-4 and abs(slm.loglik - rl.logll) < 1e-6
    assert np.allclose(slm.se, np.sqrt(np.diag(rl.vm))[:-1], rtol=1e-4)
    re = spreg.ML_Error(y.reshape(-1, 1), X, w=lw)
    sem = SpatialError(y, X, w)
    assert abs(sem.lam - re.lam) < 1e-3 and abs(sem.loglik - re.logll) < 1e-5


# =============================================================================
# cluster
# =============================================================================

def test_p_value_adjustment():
    from spatialstats.cluster import adjust_pvalues
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    assert np.allclose(adjust_pvalues(p, "fdr"), 0.05)
    assert np.allclose(adjust_pvalues(p, "bonferroni"), np.minimum(1, p * 5))
    assert np.array_equal(adjust_pvalues(p, "none"), p)


def test_conditional_permutation_properties():
    from spatialstats.cluster._perm import conditional_lag_simulations
    y = np.random.default_rng(0).normal(size=40)
    from scipy import sparse
    W = sparse.csr_matrix(np.diag(np.ones(39), 1) + np.diag(np.ones(39), -1))         # path graph
    sims = conditional_lag_simulations(y, W, 500, np.random.default_rng(1))
    assert sims.shape == (500, 40)
    # E[lag_i] = n_i * mean(other values) under conditional randomisation
    i = 10
    assert abs(sims[:, i].mean() - 2 * np.delete(y, i).mean()) < 0.15
    assert sims[:, 0].std() > 0


def test_gi_star_matches_definition_and_detects_hotspot():
    from spatialstats.cluster import getis_ord_gi
    from spatialstats import Queen
    gdf, _, xy = lattice(15)
    w = Queen(gdf, transform="binary")
    rng = np.random.default_rng(0)
    y = rng.normal(10, 1, len(gdf))
    hot = (xy[:, 0] < 4) & (xy[:, 1] < 4)
    y[hot] += 3
    res = getis_ord_gi(y, w, star=True, permutations=199, seed=1)
    # z-score by the closed form for binary Gi*: (Σ_j w y - ȳ W) / (s sqrt((nW2 - W²)/(n-1)))
    W = w.sparse.toarray() + np.eye(len(y))
    S, S2 = W.sum(1), (W ** 2).sum(1)
    z = (W @ y - y.mean() * S) / (y.std() * np.sqrt((len(y) * S2 - S ** 2) / (len(y) - 1)))
    assert np.allclose(res.z, z)
    assert res.hot[hot].mean() > 0.6 and res.hot[~hot].mean() < 0.05
    assert not res.cold.any()
    assert set(np.unique(res.labels)) <= {"Hot spot 99%", "Hot spot 95%", "Hot spot 90%", "Not significant"}


def test_gi_null_calibration():
    from spatialstats.cluster import getis_ord_gi
    _, w, _ = lattice(12)
    rng = np.random.default_rng(3)
    rates = [np.mean(getis_ord_gi(rng.normal(size=w.n), w, correction="none", permutations=0).p_analytic < 0.05)
             for _ in range(60)]
    assert 0.03 < np.mean(rates) < 0.08
    anyflag = np.mean([getis_ord_gi(rng.normal(size=w.n), w, permutations=0).hot.any() for _ in range(60)])
    assert anyflag < 0.15                              # FDR keeps false discoveries rare


def test_local_moran_clusters_quadrants():
    from spatialstats.cluster import local_moran_clusters
    _, w, xy = lattice(15)
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, w.n)
    hh = (xy[:, 0] < 4) & (xy[:, 1] < 4)
    ll = (xy[:, 0] > 10) & (xy[:, 1] > 10)
    y[hh] += 4
    y[ll] -= 4
    res = local_moran_clusters(y, w, permutations=999, seed=2, correction="none")
    lab = np.array(res.labels)
    assert (lab[hh] == "HH").mean() > 0.5 and (lab[ll] == "LL").mean() > 0.5
    assert res.stats["n_HH"] > 0 and res.stats["n_LL"] > 0
    q = np.array(res.quadrant)
    assert set(np.unique(q[lab == "HH"])) <= {1}


def test_fdr_resolution_warning():
    from spatialstats.cluster import local_moran_clusters
    _, w, _ = lattice(15)
    y = np.random.default_rng(0).normal(size=w.n)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        local_moran_clusters(y, w, permutations=99, seed=1)
    assert any("permutations" in str(x.message) for x in rec)


def test_esda_crosscheck_hotspots():
    libpysal, esda = need("libpysal"), need("esda")
    from spatialstats.cluster import getis_ord_gi
    from spatialstats import Queen
    gdf, _, _ = lattice(12)
    y = np.random.default_rng(0).normal(5, 1, len(gdf)) + 0.1 * np.arange(len(gdf)) / 12
    lw = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    lw.transform = "B"
    ref = esda.G_Local(y, lw, star=True, transform="B", permutations=0)
    mine = getis_ord_gi(y, Queen(gdf, transform="binary"), star=True, permutations=0)
    assert np.allclose(mine.z, ref.Zs, atol=1e-8)


def test_spatial_scan_statistic():
    from spatialstats.cluster import spatial_scan
    from spatialstats.cluster.scan import _llr
    c, e, C, E = 30.0, 10.0, 100.0, 100.0
    manual = c * np.log(c / e) + (C - c) * np.log((C - c) / (C - e))
    assert np.isclose(float(_llr(np.array(c), np.array(e), C, E)), manual)
    assert float(_llr(np.array(5.0), np.array(10.0), C, E, "high")) == 0.0        # deficit is not a high cluster
    _, _, xy = lattice(15)
    n = len(xy)
    E_ = np.full(n, 8.0)
    planted = np.flatnonzero((xy[:, 0] < 4) & (xy[:, 1] < 4))
    rng = np.random.default_rng(0)
    rr = np.ones(n)
    rr[planted] = 2.0
    res = spatial_scan(xy, rng.poisson(E_ * rr), expected=E_, max_pop_fraction=0.2, permutations=199, seed=1)
    top = res.clusters.iloc[0]
    assert top.significant and top.p_value <= 0.01 and top.rr > 1.5
    assert len(set(res.members[0]) & set(planted)) >= 0.7 * len(planted)
    assert set(res.members[0]) <= set(np.flatnonzero((xy[:, 0] < 7) & (xy[:, 1] < 7)))
    # under the null the top cluster is rarely significant
    fp = np.mean([spatial_scan(xy, rng.poisson(E_), expected=E_, max_pop_fraction=0.2, permutations=49,
                               seed=s).clusters.iloc[0].p_value <= 0.05 for s in range(60)])
    assert fp < 0.15
    lo = spatial_scan(xy, rng.poisson(E_ * np.where(np.isin(np.arange(n), planted), 0.3, 1.0)), expected=E_,
                      direction="low", max_pop_fraction=0.2, permutations=199, seed=3)
    assert lo.clusters.iloc[0].rr < 0.6 and lo.clusters.iloc[0].significant
    for bad in ({"cases": -np.ones(n)}, {"population": None, "expected": None}):
        try:
            spatial_scan(xy, bad.get("cases", np.ones(n)), population=bad.get("population", np.ones(n)),
                         expected=bad.get("expected"))
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_point_clusters():
    from spatialstats.cluster import dbscan_clusters, hdbscan_clusters, k_distance
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 10], [0, 10]])
    pts = np.vstack([c + rng.normal(0, 0.4, (80, 2)) for c in centres] + [rng.uniform(-3, 13, (30, 2))])
    r = dbscan_clusters(pts, eps=0.6, min_samples=8)
    assert r.stats["n_clusters"] == 3
    assert (np.bincount(r.labels[:240][r.labels[:240] >= 0]) > 60).all()
    assert r.summary_frame["n"].sum() + r.stats["n_noise"] == len(pts)
    assert hdbscan_clusters(pts, min_cluster_size=30).stats["n_clusters"] == 3
    kd = k_distance(pts, 5)
    assert len(kd) == len(pts) and np.all(np.diff(kd) <= 0)


def test_regionalisation():
    from spatialstats.cluster import skater, agglomerative, evaluate_regions, partition_agreement
    _, w, xy = lattice(12)
    rng = np.random.default_rng(0)
    left = xy[:, 0] < 6
    X = np.column_stack([np.where(left, 0.0, 5.0) + rng.normal(0, 0.3, w.n), rng.normal(0, 0.3, w.n)])
    s = skater(X, w, 2)
    assert s.is_contiguous(w) and abs(s.sizes[0] - s.sizes[1]) <= 2
    assert partition_agreement(s.labels, left.astype(int))["adjusted_rand"] > 0.9
    a = agglomerative(X, w, 2)
    assert partition_agreement(a.labels, left.astype(int))["adjusted_rand"] > 0.9
    s6 = skater(X, w, 6, min_size=10)
    assert s6.sizes.min() >= 10 and s6.is_contiguous(w)
    assert 0 < evaluate_regions(X, s6.labels)["r2"] < 1
    assert skater(X, w, 4).r2 > skater(X, w, 2).r2                       # more regions explain more
    fl = np.ones(w.n)
    sf = skater(X, w, 4, floor=fl, floor_value=20)
    assert np.bincount(sf.labels, weights=fl).min() >= 20
    try:
        skater(X, w, 1000)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_agreement_tools():
    from spatialstats.cluster import jaccard, agreement_matrix, flag_stability
    a, b = np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool)
    assert np.isclose(jaccard(a, b), 1 / 3) and jaccard(a, a) == 1.0
    m = agreement_matrix({"a": a, "b": b})
    assert m.loc["a", "b"] == m.loc["b", "a"]
    assert np.allclose(flag_stability([a, a, b]), [1, 2 / 3, 1 / 3, 0])


# =============================================================================
# bayes
# =============================================================================

def test_rates_and_intervals():
    from scipy import stats
    from spatialstats.bayes import expected_counts, smr, smr_confidence_interval, excess_pvalue, funnel_limits
    pop = np.array([1000.0, 2000.0, 3000.0])
    O = np.array([5, 10, 15])
    E = expected_counts(pop, observed=O)
    assert np.isclose(E.sum(), O.sum()) and np.allclose(E, pop * 30 / 6000)
    assert np.allclose(smr(O, E), O / E)
    ci = smr_confidence_interval(np.array([0, 10]), np.array([2.0, 10.0]))
    assert ci.loc[0, "lo"] == 0 and np.isclose(ci.loc[0, "hi"], -np.log(0.025) / 2.0, rtol=1e-6)
    assert np.isclose(ci.loc[1, "hi"], stats.chi2.ppf(0.975, 22) / 20)
    assert np.isclose(excess_pvalue(np.array([3]), np.array([1.0]))[0], stats.poisson.sf(2, 1.0))
    f = funnel_limits(np.array([1.0, 10.0, 100.0]))
    assert (f["hi_0.95"].diff().dropna() < 0).all()                 # limits narrow as E grows
    E2 = expected_counts(np.array([[100.0, 50.0]]), reference_rate=np.array([0.1, 0.2]))
    assert np.isclose(E2[0], 20.0)


def _disease_sim(seed=0, n=12, sigma=0.35, mean_e=8.0):
    from spatialstats.bayes import simulate_relative_risk, simulate_counts
    _, w, _ = lattice(n)
    rng = np.random.default_rng(seed)
    E = rng.gamma(1.5, mean_e / 1.5, w.n) + 0.5
    truth = simulate_relative_risk(w, sigma=sigma, seed=seed)
    return w, E, truth, simulate_counts(truth, E, seed=seed + 1)


def test_empirical_bayes_shrinkage():
    from spatialstats.bayes import eb_gamma_poisson, eb_local, eb_james_stein, evaluate_smoothing, smr
    w, E, truth, O = _disease_sim(0)
    raw = smr(O, E)
    for method in ("mle", "mom"):
        eb = eb_gamma_poisson(O, E, method=method)
        assert np.all((eb.shrinkage >= 0) & (eb.shrinkage <= 1))
        assert np.corrcoef(eb.shrinkage, E)[0, 1] > 0.9                          # bigger E → trust the data more
        lo, hi = np.minimum(raw, eb.prior_mean), np.maximum(raw, eb.prior_mean)
        assert np.all(eb.estimate >= lo - 1e-9) and np.all(eb.estimate <= hi + 1e-9)   # between raw and prior mean
    loc = eb_local(O, E, w)
    ev = evaluate_smoothing(truth, {"raw": raw, "eb": eb_gamma_poisson(O, E).estimate, "local": loc.estimate})
    assert ev.loc["eb", "rmse"] < ev.loc["raw", "rmse"]
    assert ev.loc["local", "rmse"] < ev.loc["raw", "rmse"]
    # closed forms
    Oo, Ee = np.array([2, 14, 1, 30, 3.0]), np.array([4, 5, 3, 6, 4.0])           # over-dispersed
    e = eb_gamma_poisson(Oo, Ee, method="mom")
    assert np.allclose(e.estimate, (e["shape"] + Oo) / (e["rate"] + Ee))
    assert np.allclose(e.shrinkage, Ee / (Ee + e["rate"]))
    # no over-dispersion -> the prior variance is 0 -> every area is shrunk to the overall mean
    flat = eb_gamma_poisson(np.array([2, 4, 1, 8, 3.0]), np.array([1, 2, 1, 3, 2.0]), method="mom")
    assert np.allclose(flat.estimate, 18 / 9) and np.allclose(flat.shrinkage, 0)
    js = eb_james_stein(np.array([1.0, 2, 3, 4, 5, 6]), np.full(6, 1.0), method="james-stein")
    expect = 1 - (6 - 3) * 1.0 / np.sum((np.arange(1, 7) - 3.5) ** 2)
    assert np.isclose(js["factor"], expect)
    nrm = eb_james_stein(np.array([0.0, 0, 0, 0, 0, 5.0]), np.array([0.5, 0.5, 0.5, 0.5, 0.5, 4.0]))
    assert nrm.shrinkage[5] < nrm.shrinkage[0]                                   # noisier estimate shrinks more


def test_mcmc_diagnostics():
    from spatialstats.bayes import rhat, ess
    rng = np.random.default_rng(0)
    iid = rng.normal(size=(4, 1000))
    assert abs(rhat(iid) - 1) < 0.02 and ess(iid) > 3000
    ar = np.zeros((4, 1000))
    for c in range(4):
        for t in range(1, 1000):
            ar[c, t] = 0.95 * ar[c, t - 1] + rng.normal()
    assert ess(ar) < 600
    stuck = np.stack([rng.normal(0, 1, 1000), rng.normal(5, 1, 1000)])
    assert rhat(stuck) > 1.5


def test_icar_structure():
    from spatialstats.bayes import icar_structure, icar_scaling_factor
    _, w, _ = lattice(8)
    st = icar_structure(w)
    A = st["A"]
    assert (A != A.T).nnz == 0 and A.diagonal().sum() == 0
    for idx in st["colors"]:                                          # colouring is proper
        assert A[idx][:, idx].nnz == 0
    assert st["n_comp"] == 1 and not st["island"].any()
    assert icar_scaling_factor(st) > 0


def test_bym_gibbs_recovers_risk_and_converges():
    from spatialstats.bayes import BYM, evaluate_smoothing, smr
    w, E, truth, O = _disease_sim(1, n=12)
    fit = BYM(O, E, w, draws=800, tune=1500, chains=4, thin=6, seed=3)
    assert fit.max_rhat < 1.1 and fit.min_ess > 30
    ev = evaluate_smoothing(truth, {"raw": smr(O, E), "bym": fit.rr_mean},
                            intervals={"bym": (fit.rr_lo, fit.rr_hi)})
    assert ev.loc["bym", "rmse"] < 0.75 * ev.loc["raw", "rmse"]
    assert ev.loc["bym", "coverage"] > 0.88
    assert np.all((fit.rr_lo <= fit.rr_median) & (fit.rr_median <= fit.rr_hi))
    assert np.all((fit.exceedance(1.0) >= 0) & (fit.exceedance(1.0) <= 1))
    assert np.corrcoef(fit.exceedance(1.0), truth)[0, 1] > 0.5
    assert 0 <= fit.stats["spatial_fraction"] <= 1 and np.isfinite(fit.dic)
    sh = fit.shrinkage()
    small = np.argsort(E)[:20]
    large = np.argsort(-E)[:20]
    assert np.nanmedian(np.abs(sh[small])) > np.nanmedian(np.abs(sh[large]))    # small areas shrink more
    same = BYM(O, E, w, draws=200, tune=300, chains=2, thin=2, seed=9)
    again = BYM(O, E, w, draws=200, tune=300, chains=2, thin=2, seed=9)
    assert np.allclose(same.rr_mean, again.rr_mean)                             # reproducible
    for m in ("icar", "iid"):
        assert BYM(O, E, w, model=m, draws=200, tune=300, chains=2, thin=2, seed=1).rr_mean.shape == (w.n,)


def test_bym_covariates_and_errors():
    from spatialstats.bayes import BYM
    w, E, truth, _ = _disease_sim(2, n=10)
    x = np.random.default_rng(0).normal(size=w.n)
    from spatialstats.bayes import simulate_counts
    O = simulate_counts(truth * np.exp(0.5 * x), E, seed=4)
    fit = BYM(O, E, w, X=x, names=["x"], draws=500, tune=1000, chains=3, thin=5, seed=1)
    assert fit.coef.loc[0, "lo"] < 0.5 < fit.coef.loc[0, "hi"] + 0.35 and fit.coef.loc[0, "mean"] > 0.2
    for kwargs in ({"chains": 1}, {"model": "nope"}, {"backend": "nope"}, {"model": "bym2"}):
        try:
            BYM(O, E, w, draws=10, tune=10, **kwargs)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_bym_pymc_backend_agrees():
    need("pymc")
    from spatialstats.bayes import BYM
    w, E, truth, O = _disease_sim(3, n=7)
    g = BYM(O, E, w, draws=1500, tune=1500, chains=4, thin=5, seed=1)
    p = BYM(O, E, w, backend="pymc", draws=500, tune=500, chains=2, seed=1)
    assert np.corrcoef(g.rr_mean, p.rr_mean)[0, 1] > 0.98
    assert np.abs(g.rr_mean - p.rr_mean).max() < 0.15


# =============================================================================
# sampling
# =============================================================================

def _frame(N=400, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (N, 2))
    y = np.sin(xy[:, 0] / 12) + np.cos(xy[:, 1] / 15) + rng.normal(0, 0.15, N)
    return xy, y


def test_basic_designs():
    from spatialstats.sampling import simple_random, systematic, stratified, two_stage, neyman_allocation
    xy, _ = _frame()
    s = simple_random(xy, 40, seed=1)
    assert s.n == 40 and len(set(s.indices)) == 40 and np.allclose(s.pi, 0.1) and np.allclose(s.weights, 10)
    assert np.array_equal(simple_random(xy, 40, seed=1).indices, s.indices)          # reproducible
    sy = systematic(xy, 40, seed=2)
    assert sy.n == 40 and len(set(sy.indices)) == 40
    strata = np.where(xy[:, 0] < 30, "a", "b")
    st = stratified(xy, strata, 40, seed=3)
    Na, Nb = (strata == "a").sum(), (strata == "b").sum()
    assert st.n == 40 and abs((st.strata == "a").sum() - 40 * Na / 400) <= 1
    assert np.isclose(st.pi[st.strata == "a"][0], (st.strata == "a").sum() / Na)
    assert set(stratified(xy, strata, {"a": 5, "b": 7}, seed=1).strata) == {"a", "b"}
    assert neyman_allocation([100, 100], [1.0, 3.0], 40).tolist() == [10, 30]
    assert stratified(xy, strata, 40, allocation="equal", seed=1).n == 40
    ts = two_stage(xy, (xy[:, 0] // 25).astype(int), n_clusters=2, n_per_cluster=10, seed=1)
    assert ts.n == 20 and len(np.unique(ts.clusters)) == 2
    for bad in (lambda: simple_random(xy, 1000), lambda: stratified(xy, strata[:-1], 10),
                lambda: stratified(xy, strata, 40, allocation="neyman")):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_grts_inclusion_probabilities_are_exact():
    from spatialstats.sampling import grts
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1, (60, 2))
    size = rng.uniform(1, 5, 60)
    counts = np.zeros(60)
    R = 3000
    g = np.random.default_rng(1)
    for _ in range(R):
        counts[grts(xy, 10, size=size, seed=g).indices] += 1
    pi = grts(xy, 10, size=size, seed=1).frame_pi
    assert np.isclose(pi.sum(), 10) and pi.max() <= 1
    assert np.abs(counts / R - pi).max() < 0.05
    eq = grts(xy, 10, seed=2)
    assert eq.n == 10 and np.allclose(eq.pi, 0.1667, atol=1e-3)


def test_grts_is_spatially_balanced_and_more_efficient():
    from spatialstats.sampling import simple_random, systematic, grts, spatial_balance, evaluate_design
    xy, y = _frame(600, seed=1)
    b_srs = np.mean([spatial_balance(simple_random(xy, 40, seed=s)) for s in range(25)])
    b_grts = np.mean([spatial_balance(grts(xy, 40, seed=s)) for s in range(25)])
    assert b_grts < 0.6 * b_srs
    e_srs = evaluate_design(y, lambda r: simple_random(xy, 40, r), n_reps=250, seed=5)
    e_grts = evaluate_design(y, lambda r: grts(xy, 40, seed=r), n_reps=250, seed=5)
    e_sys = evaluate_design(y, lambda r: systematic(xy, 40, "hilbert", r), n_reps=250, seed=5)
    assert e_grts["sd"] < 0.85 * e_srs["sd"] and e_sys["sd"] < 0.85 * e_srs["sd"]
    assert abs(e_srs["bias"]) < 3 * e_srs["sd"] / np.sqrt(250) + 1e-3
    for e in (e_srs, e_grts):
        assert 0.90 <= e["coverage"] <= 0.995
    # local-mean variance tracks the true (Monte Carlo) spread for a balanced design
    assert 0.7 < e_grts["mean_se"] / e_grts["sd"] < 1.4


def test_estimators_and_planning():
    from spatialstats.sampling import simple_random, stratified, estimate_mean, sample_size_mean, \
        effective_sample_size, space_filling, design_effect
    xy, y = _frame()
    s = simple_random(xy, 60, seed=4)
    r = estimate_mean(y, s)
    assert np.isclose(r.mean, y[s.indices].mean())
    assert np.isclose(r.se, np.sqrt((1 - 60 / 400) * y[s.indices].var(ddof=1) / 60))
    assert r.lo < y.mean() < r.hi or abs(r.mean - y.mean()) < 3 * r.se
    assert np.isclose(estimate_mean(y[s.indices], s).mean, r.mean)                    # sample-length y accepted
    st = stratified(xy, np.where(xy[:, 0] < 50, 0, 1), 60, seed=1)
    assert estimate_mean(y, st).variance_method == "stratified"
    try:
        estimate_mean(y, space_filling(xy, 20, "kmeans", seed=1))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert sample_size_mean(10, 2) == 97 and sample_size_mean(10, 2, N=100) < 97
    assert np.isclose(effective_sample_size(100, 0.5), 100 / 3) and np.isclose(design_effect(2, 4), 0.5)
    assert effective_sample_size(100, 0.5, "equicorrelated") < 2.1


def test_space_filling_and_clhs():
    from spatialstats.sampling import simple_random, space_filling, clhs, coverage_metrics
    xy, _ = _frame(500, seed=2)
    km, mm = space_filling(xy, 25, "kmeans", seed=1), space_filling(xy, 25, "maximin", seed=1)
    rnd = simple_random(xy, 25, seed=1)
    assert km.n == 25 and mm.n == 25
    assert coverage_metrics(km)["mean_dist_to_sample"] < coverage_metrics(rnd)["mean_dist_to_sample"]
    assert coverage_metrics(mm)["min_spacing"] > coverage_metrics(rnd)["min_spacing"]
    X = np.random.default_rng(0).normal(size=(500, 3))
    X[:, 1] += X[:, 0]
    cl = clhs(X, 30, iterations=4000, coords=xy, seed=1)
    edges = np.quantile(X, np.linspace(0, 1, 31)[1:-1], axis=0)
    def o1(sel):
        return sum(np.abs(np.bincount(np.searchsorted(edges[:, j], X[sel, j], side="right"), minlength=30) - 1).sum()
                   for j in range(3))
    assert o1(cl.indices) < 0.4 * o1(np.random.default_rng(3).choice(500, 30, replace=False))
    assert cl.n == 30 and np.isnan(cl.pi).all()


def test_two_stage_variance_coverage():
    from spatialstats.sampling import two_stage, evaluate_design
    xy, y = _frame(600, seed=3)
    cl = (xy[:, 0] // 12.5).astype(int) * 8 + (xy[:, 1] // 12.5).astype(int)
    e = evaluate_design(y, lambda r: two_stage(xy, cl, 20, 6, seed=r), n_reps=300, seed=2)
    assert 0.88 <= e["coverage"] <= 1.0 and 0.6 < e["mean_se"] / e["sd"] < 1.6


def test_grid_points():
    from spatialstats.sampling import grid_points
    from shapely.geometry import Polygon
    poly = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    g = grid_points(poly, spacing=1.0, seed=1)
    assert 55 <= len(g) <= 65
    d = np.sort(np.unique(np.round(np.diff(np.sort(np.unique(g[:, 0]))), 6)))
    assert np.allclose(d, 1.0)
    h = grid_points(poly, n=60, seed=1, hexagonal=True)
    assert 40 <= len(h) <= 80 and poly.contains(__import__("shapely").MultiPoint(h)) is True


# =============================================================================
# interpolate
# =============================================================================

def _field(n=150, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n, 2))
    z = 2 + 0.05 * xy[:, 0] - 0.03 * xy[:, 1] + np.sin(xy[:, 0] / 20)
    return xy, z


def test_deterministic_interpolators():
    from spatialstats.interpolate import IDW, NearestNeighbor, TIN, RBF, Spline, Thiessen, ThinPlateSpline
    xy, z = _field()
    for m in (IDW(2), IDW(3, k=None), NearestNeighbor(), Thiessen(), TIN(), RBF(), Spline(),
              ThinPlateSpline()):
        m.fit(xy, z)
        assert np.allclose(m.predict(xy), z, atol=1e-8), m
    q = np.array([[50.0, 50.0], [20.0, 80.0]])
    idw = IDW(2, k=None).fit(xy, z).predict(q)
    d = np.linalg.norm(xy[None] - q[:, None], axis=2)
    w = d ** -2.0
    assert np.allclose(idw, (w * z).sum(1) / w.sum(1))
    assert idw.min() >= z.min() and idw.max() <= z.max()                               # IDW never leaves the data range
    assert np.isnan(IDW(radius=1e-3).fit(xy, z).predict(np.array([[500.0, 500.0]]))[0])
    assert np.isfinite(TIN().fit(xy, z).predict(np.array([[500.0, 500.0]]))).all()
    assert np.isnan(TIN(fill="nan").fit(xy, z).predict(np.array([[500.0, 500.0]]))[0])
    lin = 3 + 0.5 * xy[:, 0] - 0.2 * xy[:, 1]                                          # thin plate reproduces planes
    assert np.allclose(RBF().fit(xy, lin).predict(q), 3 + 0.5 * q[:, 0] - 0.2 * q[:, 1], atol=1e-6)
    assert "IDW(power=3" in repr(IDW(3)) and IDW(3).clone().power == 3
    try:
        IDW().predict(q)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_trend_surface_and_thiessen():
    import shapely
    from spatialstats.interpolate import (
        TrendSurface, Thiessen, thiessen_polygons, NearestNeighbor, ThinPlateSpline,
    )
    xy, z = _field()
    q = np.array([[50.0, 50.0], [20.0, 80.0], [0.0, 0.0]])
    lin = 3 + 0.5 * xy[:, 0] - 0.2 * xy[:, 1]
    trend = TrendSurface(degree=1).fit(xy, lin)
    assert np.allclose(trend.predict(q), 3 + 0.5 * q[:, 0] - 0.2 * q[:, 1], atol=1e-6)
    assert trend.r2_ > 0.999 and trend.powers_[0] == (0, 0) and len(trend.coef_) == 3
    quad = lin + 0.01 * xy[:, 0] ** 2 - 0.002 * xy[:, 0] * xy[:, 1]
    pred = TrendSurface(2).fit(xy, quad).predict(q)
    truth = 3 + 0.5 * q[:, 0] - 0.2 * q[:, 1] + 0.01 * q[:, 0] ** 2 - 0.002 * q[:, 0] * q[:, 1]
    assert np.allclose(pred, truth, atol=1e-4)
    noisy = lin + np.random.default_rng(0).normal(0, 0.5, len(xy))
    assert TrendSurface(1).fit(xy, noisy).r2_ < 1.0
    assert "TrendSurface(degree=2)" in repr(TrendSurface(2)) and TrendSurface(2).clone().degree == 2
    for bad in (0, -1, 1.5, True):
        try:
            TrendSurface(bad)
            raise AssertionError(f"expected ValueError for degree={bad}")
        except ValueError:
            pass
    try:
        TrendSurface(4).fit(xy[:10], z[:10])          # 15 terms, 10 observations
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    line = np.column_stack([np.linspace(0, 10, 20), np.zeros(20)])
    try:
        TrendSurface(1).fit(line, line[:, 0])
        raise AssertionError("expected ValueError for collinear samples")
    except ValueError:
        pass

    th = Thiessen().fit(xy, z)
    assert np.allclose(th.predict(q), NearestNeighbor().fit(xy, z).predict(q))
    gdf = th.polygons(region=(0, 0, 100, 100))
    assert len(gdf) == len(xy) and {"x", "y", "value"} <= set(gdf.columns)
    assert np.isclose(gdf.geometry.area.sum(), 10_000)
    pts = shapely.points(xy[:, 0], xy[:, 1])
    assert shapely.covers(np.asarray(gdf.geometry), pts).all()
    assert np.allclose(gdf["value"].to_numpy(), z)
    # a query inside a cell takes that cell's value
    inside = np.array([[gdf.geometry.iloc[0].representative_point().x,
                        gdf.geometry.iloc[0].representative_point().y]])
    assert th.predict(inside)[0] == gdf["value"].iloc[0]
    bare = thiessen_polygons(xy, region=(0, 0, 100, 100))
    assert "value" not in bare.columns and len(bare) == len(xy)
    try:
        thiessen_polygons(np.vstack([xy[:5], xy[:1]]), np.arange(6))
        raise AssertionError("expected ValueError for duplicate coordinates")
    except ValueError:
        pass
    try:
        Thiessen().polygons()
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass

    tps = ThinPlateSpline().fit(xy, lin)
    assert np.allclose(tps.predict(q), 3 + 0.5 * q[:, 0] - 0.2 * q[:, 1], atol=1e-6)
    assert tps.clone().smoothing == 0.0 and "ThinPlateSpline" in repr(tps)


def test_variogram_recovery():
    from spatialstats.interpolate import VariogramModel, empirical_variogram, fit_variogram, fit_variogram_auto
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (300, 2))
    true = VariogramModel("spherical", nugget=0.1, psill=1.0, range=40.0)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    C = true.sill - true.gamma(D)
    np.fill_diagonal(C, true.sill)
    z = np.linalg.cholesky(C + 1e-8 * np.eye(300)) @ rng.normal(size=300)
    emp = empirical_variogram(xy, z, n_lags=15)
    fit = fit_variogram(emp, "spherical")
    assert 0.5 < fit.sill < 1.8 and 20 < fit.range < 80
    best, _ = fit_variogram_auto(xy, z)
    assert best.model in ("spherical", "exponential", "gaussian")
    assert true.gamma(np.array([0.0]))[0] == 0 and np.isclose(true.gamma(np.array([200.0]))[0], true.sill)
    assert VariogramModel("exponential", 0, 1, 30).gamma(np.array([30.0]))[0] > 0.94


def test_kriging_properties():
    from spatialstats.interpolate import OrdinaryKriging, VariogramModel
    xy, z = _field(80, seed=1)
    vg = VariogramModel("spherical", nugget=0.05, psill=1.0, range=45.0)
    ok = OrdinaryKriging(vg).fit(xy, z)
    m, sd = ok.predict(xy, return_std=True)
    assert np.allclose(m, z, atol=1e-8) and np.allclose(sd, 0, atol=1e-5)               # exact interpolator
    q = np.random.default_rng(2).uniform(0, 100, (30, 2))
    mg, sg = ok.predict(q, return_std=True)
    ml, sl = OrdinaryKriging(vg, n_neighbors=80).fit(xy, z).predict(q, return_std=True)
    assert np.allclose(mg, ml) and np.allclose(sg, sl)
    ms = OrdinaryKriging(vg, n_neighbors=12).fit(xy, z).predict(q)
    assert np.abs(ms - mg).max() < 0.35
    far = ok.predict(np.array([[300.0, 300.0]]), return_std=True)
    assert far[1][0] > sg.max()                                                          # uncertainty grows away from data
    assert np.isclose(far[0][0], z.mean(), atol=0.5)                                    # reverts to the (local) mean
    auto = OrdinaryKriging("auto").fit(xy, z)
    assert auto.variogram_.model in ("spherical", "exponential", "gaussian")
    ex = OrdinaryKriging("exponential").fit(xy, z)
    assert ex.variogram_.model == "exponential"


def test_simple_universal_cokriging_and_regression():
    from spatialstats.interpolate import (
        OrdinaryKriging, SimpleKriging, UniversalKriging, VariogramModel, CrossVariogramModel,
        Cokriging, MultivariateCokriging, ColocatedCokriging, RegressionKriging,
        empirical_cross_variogram, fit_cross_variogram, fit_lmc, check_lmc_validity,
        enforce_quantile_monotonicity, cross_validate,
    )
    rng = np.random.default_rng(2)
    xy, z = _field(60, seed=1)
    q = rng.uniform(0, 100, (20, 2))
    vg = VariogramModel("spherical", nugget=0.05, psill=1.0, range=40.0)
    vg2 = VariogramModel("spherical", nugget=0.05, psill=1.2, range=40.0)

    # simple kriging is exact, reverts to the known mean past the range, and
    # agrees with the global solution when every neighbour is used
    sk = SimpleKriging(vg, mean=5.0).fit(xy, z)
    m, sd = sk.predict(xy, return_std=True)
    assert np.allclose(m, z, atol=1e-8) and np.allclose(sd, 0, atol=1e-5)
    far, far_sd = sk.predict(np.array([[500.0, 500.0]]), return_std=True)
    assert np.isclose(far[0], 5.0, atol=1e-6) and np.isclose(far_sd[0], np.sqrt(vg.sill), atol=1e-6)
    assert np.allclose(sk.predict(q), SimpleKriging(vg, mean=5.0, n_neighbors=len(xy)).fit(xy, z).predict(q))
    local = SimpleKriging(vg, mean=5.0, n_neighbors=12).fit(xy, z).predict(q)
    assert np.isfinite(local).all() and np.abs(local - sk.predict(q)).max() < 1.0
    assert np.isclose(SimpleKriging(vg).fit(xy, z).mean_, z.mean())
    assert sk.clone().mean == 5.0
    g, gv, gs = sk.predict_grid((0, 0, 100, 100), n_cells=8, return_std=True)
    assert len(gv) == len(gs) == len(g.coords)

    # degree 0 is ordinary kriging; a polynomial drift is reproduced everywhere
    uk0 = UniversalKriging(vg, degree=0).fit(xy, z)
    ok = OrdinaryKriging(vg).fit(xy, z)
    mu, su = uk0.predict(q, return_std=True)
    mo, so = ok.predict(q, return_std=True)
    assert np.allclose(mu, mo, atol=1e-6) and np.allclose(su, so, atol=1e-6)
    plane_xy = rng.uniform(0, 100, (40, 2))
    plane = 1.0 + 2.0 * plane_xy[:, 0] - 0.5 * plane_xy[:, 1]
    uk = UniversalKriging(vg, degree=1).fit(plane_xy, plane)
    far_xy = np.array([[-50.0, 200.0], [10.0, 10.0]])
    assert np.allclose(uk.predict(plane_xy), plane, atol=1e-6)
    assert np.allclose(uk.predict(far_xy), 1.0 + 2.0 * far_xy[:, 0] - 0.5 * far_xy[:, 1], atol=1e-5)
    assert np.allclose(uk.predict(far_xy),
                       UniversalKriging(vg, degree=1, n_neighbors=len(plane_xy)).fit(plane_xy, plane).predict(far_xy))
    cv = cross_validate(SimpleKriging(vg, mean=float(z.mean())), xy, z, method="kfold", k=4, seed=0)
    assert np.isfinite(cv.rmse)

    # zero cross-covariance drops the secondary and matches ordinary kriging
    xy2 = xy + np.array([1.0, -1.0])
    z2 = 0.5 * z + 0.1
    cv0 = CrossVariogramModel("spherical", nugget=0.0, psill=0.0, range=40.0)
    ck0 = Cokriging(vg, vg2, cv0).fit(xy, z, xy2, z2)
    ckm, cks = ck0.predict(q, return_std=True)
    assert np.allclose(ckm, mo, atol=1e-6) and np.allclose(cks, so, atol=1e-5)
    linked = CrossVariogramModel("spherical", nugget=0.0, psill=0.4, range=40.0)
    ck = Cokriging(vg, vg2, linked).fit(xy, z, xy2, z2)
    assert np.allclose(ck.predict(xy), z, atol=1e-6)
    cksd = ck.predict(xy, return_std=True)[1]
    assert np.allclose(cksd, 0, atol=1e-5)
    mv = MultivariateCokriging([vg, vg2], {(0, 1): cv0}).fit([xy, xy2], [z, z2])
    assert np.allclose(mv.predict(q), mo, atol=1e-6)
    assert np.allclose(MultivariateCokriging([vg, vg2], {(1, 0): linked}).fit([xy, xy2], [z, z2]).predict(xy),
                       z, atol=1e-6)

    # colocated cokriging with no cross-correlation is ordinary kriging;
    # a strong positive cross-correlation moves the prediction
    colo0 = ColocatedCokriging(vg, [vg2], [cv0]).fit(xy, z, secondary_means=[0.0])
    sec = np.zeros((len(q), 1))
    assert np.allclose(colo0.predict(q, sec), mo, atol=1e-6)
    colo = ColocatedCokriging(vg, [vg], [CrossVariogramModel("spherical", 0.0, 0.8, 40.0)]).fit(
        xy, z, secondary_means=[float(z.mean())],
    )
    high = np.full(len(q), float(z.mean()) + 3.0)
    assert colo.predict(q, high).mean() > mo.mean()

    emp = empirical_cross_variogram(xy, z, z2, n_lags=8)
    assert {"lag", "gamma", "n_pairs"} <= set(emp.columns) and len(emp) >= 3
    fitted = fit_cross_variogram(emp, "spherical", cov0=float(np.cov(z, z2)[0, 1]))
    assert fitted.range > 0 and np.isfinite(fitted.psill)
    vgs, cross, info = fit_lmc(xy, [z, z2], names=["a", "b"], n_lags=8)
    assert info["lmc_check"]["valid"] and (0, 1) in cross
    assert np.allclose(np.diag(info["sill_matrix"]), [v.psill for v in vgs])
    assert check_lmc_validity([[1.0, 0.2], [0.2, 1.0]])["valid"]
    assert not check_lmc_validity([[1.0, 2.0], [2.0, 1.0]])["valid"]
    lmc = Cokriging(vgs[0], vgs[1], cross[(0, 1)]).fit(xy, z, xy, z2)
    assert np.allclose(lmc.predict(xy), z, atol=1e-5)

    # regression kriging recovers a pure regression and is exact at the samples
    xfeat = rng.normal(size=(len(xy), 1))
    y = 1.5 + 2.0 * xfeat[:, 0]
    rk = RegressionKriging(variogram=vg).fit(xfeat, y, xy)
    xnew = np.array([[0.0], [1.0], [-2.0]])
    assert np.allclose(rk.predict(xnew, q[:3]), 1.5 + 2.0 * xnew[:, 0], atol=1e-6)
    pred, std = rk.predict(xfeat, xy, return_std=True)
    assert np.allclose(pred, y, atol=1e-6) and np.allclose(std, 0, atol=1e-5)
    spatial = np.sin(xy[:, 0] / 15.0)
    rk2 = RegressionKriging(regressor="linear", variogram="spherical", n_lags=8).fit(
        xfeat, 2.0 * xfeat[:, 0] + spatial, xy,
    )
    assert rk2.variogram_.model == "spherical" and rk2.residuals_.shape == (len(xy),)
    try:
        _ = rk.feature_importances_
        raise AssertionError("expected AttributeError")
    except AttributeError:
        pass
    try:
        RegressionKriging(regressor="not-a-model")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    fixed = enforce_quantile_monotonicity({0.9: np.array([1.0, 3.0]), 0.1: np.array([2.0, 0.0])})
    assert np.allclose(fixed[0.1], [1.0, 0.0]) and np.allclose(fixed[0.9], [2.0, 3.0])


def test_indicator_kriging_and_etype():
    from spatialstats.interpolate import (
        IndicatorKriging, VariogramModel, empirical_variogram, empirical_indicator_variogram,
        fit_indicator_variogram, indicator_transform, etype_estimate,
    )
    xy, z = _field(50, seed=4)
    q = np.random.default_rng(1).uniform(0, 100, (12, 2))
    t = float(np.median(z))
    ind = indicator_transform(z, t)
    emp = empirical_indicator_variogram(xy, z, t, n_lags=8)
    direct = empirical_variogram(xy, ind, n_lags=8)
    assert np.allclose(emp["gamma"], direct["gamma"]) and np.all((emp["gamma"] >= 0) & (emp["gamma"] <= 0.5))
    # I(Z >= t) and I(Z <= t) share a variogram
    flipped = empirical_indicator_variogram(xy, z, t, n_lags=8, greater_equal=False)
    assert np.allclose(emp["gamma"], flipped["gamma"])
    vg, _ = fit_indicator_variogram(xy, z, t, model="spherical", n_lags=8)
    assert vg.model == "spherical" and vg.sill > 0 and vg.range > 0

    given = VariogramModel("spherical", nugget=0.01, psill=0.2, range=40.0)
    ik = IndicatorKriging(variogram=given, n_neighbors=12, regularization=0.0).fit(xy, z)
    at_data = ik.predict(xy, t)[t]
    assert np.allclose(at_data, ind, atol=1e-5)                                  # reproduces the indicator
    prob = ik.predict(q, [t, t + 1.0])
    assert prob[t].shape == (len(q),) and np.all((prob[t] >= 0) & (prob[t] <= 1))
    below = float(z.min() - 1.0)
    assert np.allclose(ik.predict(q, below)[below], 1.0)                         # every sample exceeds it
    wide = IndicatorKriging(variogram=given, n_neighbors=8, search_radius=1e9, regularization=0.0)
    local = IndicatorKriging(variogram=given, n_neighbors=8, regularization=0.0)
    assert np.allclose(wide.fit(xy, z).predict(q, t)[t], local.fit(xy, z).predict(q, t)[t], atol=1e-6)
    auto = IndicatorKriging(variogram="auto", n_lags=8).fit(xy, z)
    assert auto.predict(q, t)[t].shape == (len(q),) and t in auto.variograms_

    # Uniform(0, 1) survival integrates to the mean, 1/2
    cuts = np.array([0.25, 0.5, 0.75])
    surv = [np.array([0.75]), np.array([0.5]), np.array([0.25])]
    assert np.isclose(etype_estimate(cuts, surv, z_min=0.0, z_max=1.0)[0], 0.5)
    est = ik.etype(q, np.quantile(z, [0.3, 0.5, 0.7]), z_min=float(z.min()), z_max=float(z.max()))
    assert est.shape == (len(q),) and np.isfinite(est).all()
    try:
        etype_estimate([2.0, 1.0], [np.array([0.2]), np.array([0.8])])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        empirical_indicator_variogram(xy, z, z.max() + 10.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_pykrige_crosscheck():
    need("pykrige")
    from pykrige.ok import OrdinaryKriging as PK
    from spatialstats.interpolate import OrdinaryKriging, VariogramModel
    xy, z = _field(80, seed=1)
    q = np.random.default_rng(2).uniform(0, 100, (25, 2))
    for name in ("spherical", "exponential"):
        vg = VariogramModel(name, nugget=0.05, psill=0.9, range=45.0)
        mu, sd = OrdinaryKriging(vg).fit(xy, z).predict(q, return_std=True)
        # pykrige's first parameter is the TOTAL sill
        pm, pv = PK(xy[:, 0], xy[:, 1], z, variogram_model=name,
                    variogram_parameters=[0.95, 45.0, 0.05]).execute("points", q[:, 0], q[:, 1])
        assert np.allclose(mu, pm, atol=1e-8) and np.allclose(sd ** 2, pv, atol=1e-8)


def test_grid_and_cross_validation():
    from spatialstats.interpolate import IDW, RBF, OrdinaryKriging, make_grid, cross_validate, grid_search, \
        compare_methods
    xy, z = _field(120, seed=3)
    grid, v = IDW().fit(xy, z).predict_grid((0, 0, 100, 100), n_cells=20)
    assert grid.shape == (20, 20) and len(v) == 400 and grid.to_raster(v).shape == (20, 20)
    ok = OrdinaryKriging("spherical").fit(xy, z)
    g2, m2, s2 = ok.predict_grid(xy, n_cells=15, return_std=True)
    assert len(m2) == len(s2) == len(g2.coords)
    cv = cross_validate(RBF(), xy, z, method="loo")
    assert cv.rmse < 0.15 and cv.stats["n_failed"] == 0 and len(cv.to_frame()) == 120
    cvs = cross_validate(IDW(), xy, z, method="spatial_block", k=5, block_size=30, seed=1)
    cvl = cross_validate(IDW(), xy, z, method="loo")
    assert cvs.rmse >= cvl.rmse * 0.9                                                    # blocks are (at least) as hard
    gs = grid_search(IDW, {"power": [1, 2, 4], "k": [6, 12]}, xy, z, method="kfold", k=5, seed=1)
    assert len(gs) == 6 and gs["rmse"].is_monotonic_increasing
    cm = compare_methods({"idw": IDW(), "krig": OrdinaryKriging("auto"), "tps": RBF(smoothing=0.01)}, xy, z, seed=1)
    assert set(cm.index) == {"idw", "krig", "tps"} and cm["rmse"].is_monotonic_increasing
    try:
        cross_validate(IDW(), xy, z, method="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_kriging_variance_is_calibrated():
    from spatialstats.interpolate import OrdinaryKriging, VariogramModel
    rng = np.random.default_rng(0)
    n = 150
    xy = rng.uniform(0, 100, (n, 2))
    vg = VariogramModel("exponential", nugget=0.05, psill=1.0, range=40.0)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    C = vg.sill - vg.gamma(D)
    np.fill_diagonal(C, vg.sill)
    z = np.linalg.cholesky(C + 1e-8 * np.eye(n)) @ rng.normal(size=n)
    zs = []
    for i in range(n):
        tr = np.arange(n) != i
        m, s = OrdinaryKriging(vg).fit(xy[tr], z[tr]).predict(xy[i:i + 1], return_std=True)
        zs.append((z[i] - m[0]) / s[0])
    zs = np.array(zs)
    assert abs(zs.mean()) < 0.2 and 0.8 < zs.std() < 1.25


def _zones():
    import geopandas as gpd
    from shapely.geometry import box
    src = gpd.GeoDataFrame({"pop": [100.0, 300.0], "rate": [1.0, 3.0]},
                           geometry=[box(0, 0, 1, 2), box(1, 0, 3, 2)], crs=3857)
    tgt = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 1), box(0, 1, 2, 2), box(2, 0, 3, 2)], crs=3857)
    return src, tgt


def test_areal_weighting_and_dasymetric():
    import geopandas as gpd
    from spatialstats.interpolate import areal_weighting, dasymetric
    src, tgt = _zones()
    r = areal_weighting(src, tgt, extensive="pop", intensive="rate")
    # target 0 = left half of zone A (area 1 of 2) + left half of zone B's first unit column
    assert np.isclose(r["pop"].sum(), 400.0)
    assert np.allclose(r["pop"], [100 * 0.5 + 300 * 0.25, 100 * 0.5 + 300 * 0.25, 300 * 0.5])
    assert np.allclose(r["rate"], [(1 * 1 + 3 * 1) / 2, (1 * 1 + 3 * 1) / 2, 3.0])
    pts = gpd.GeoDataFrame({"w": [1.0, 1.0, 1.0, 1.0, 5.0]},
                           geometry=gpd.points_from_xy([0.5, 0.5, 1.5, 1.5, 2.5], [0.5, 1.5, 0.5, 1.5, 0.5]), crs=3857)
    d = dasymetric(src, tgt, pts, weight="w", extensive="pop")
    assert np.isclose(d["pop"].sum(), 400.0)
    # zone A (left col, weight 2): split equally between tgt0 and tgt1; zone B weights: tgt0 1, tgt1 1, tgt2 5
    assert np.allclose(d["pop"], [100 * 0.5 + 300 / 7, 100 * 0.5 + 300 / 7, 300 * 5 / 7])
    empty = pts.iloc[[4]]                                              # zone A has no points -> falls back to area
    d2 = dasymetric(src, tgt, empty, weight="w", extensive="pop")
    assert np.isclose(d2["pop"].sum(), 400.0)
    bad = tgt.set_crs(4326, allow_override=True)
    try:
        areal_weighting(src, bad, extensive="pop")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# =============================================================================
# package wiring
# =============================================================================

def test_packages_expose_public_api():
    import spatialstats as ss
    for mod, names in {
        "pointpattern": ["PointPattern", "Window", "ripley_k", "envelope", "kde", "clark_evans"],
        "regression": ["OLS", "SpatialLag", "SpatialError", "lm_tests", "compare_models"],
        "cluster": ["getis_ord_gi", "spatial_scan", "skater", "dbscan_clusters"],
        "bayes": ["BYM", "eb_gamma_poisson", "eb_local", "smr"],
        "sampling": ["grts", "stratified", "estimate_mean", "clhs"],
        "interpolate": ["IDW", "OrdinaryKriging", "SimpleKriging", "UniversalKriging",
                        "Cokriging", "RegressionKriging", "IndicatorKriging", "etype_estimate",
                        "areal_weighting", "cross_validate"],
    }.items():
        m = getattr(ss, mod)
        assert m.__all__, mod
        for n in names:
            assert hasattr(m, n) and n in m.__all__, (mod, n)


if __name__ == "__main__":
    import time
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = skipped = 0
    for name, fn in tests:
        t = time.time()
        try:
            fn()
            print(f"  PASS  {name}  ({time.time() - t:.1f}s)")
            passed += 1
        except unittest.SkipTest as e:
            print(f"  SKIP  {name}: {e}")
            skipped += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    raise SystemExit(1 if failed else 0)
