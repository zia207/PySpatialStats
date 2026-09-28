"""
Comprehensive test suite for PySpatialStats (gwmodel).
Tests all major modules: kernels, bandwidth, GWR, MGWR, GWSS,
GWRandomForest, GNNWR, GTNNWR, GWPCA, diagnostics, datasets.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

PASS = []
FAIL = []


def run_test(name, fn):
    try:
        fn()
        print(f"  ✅ PASS  {name}")
        PASS.append(name)
    except Exception as e:
        print(f"  ❌ FAIL  {name}: {e}")
        FAIL.append((name, str(e)))


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_data(n=80, p=3, seed=42):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n, 2))
    X = rng.standard_normal((n, p))
    # Spatially varying coefficients — shape (n, p)
    base = np.resize([1.0, -1.0, 0.5, 0.8, -0.3], p)  # works for any p
    spatial_var = 0.5 * np.sin(np.pi * coords[:, 0] / 100)[:, None]  # (n,1)
    coef_true = base[None, :] + spatial_var * np.ones((1, p))          # (n, p)
    y = np.sum(X * coef_true, axis=1) + rng.standard_normal(n) * 0.3
    return X, y, coords


# ─── Kernel Tests ────────────────────────────────────────────────────────────

def test_bisquare():
    from spatialstats.gwmodel.core.kernels import bisquare
    d = np.array([0, 5, 10, 15])
    w = bisquare(d, 10)
    assert w[0] == 1.0
    assert w[2] == 0.0  # d==h gives 0
    assert w[3] == 0.0
    assert w[1] > 0

def test_gaussian():
    from spatialstats.gwmodel.core.kernels import gaussian
    w = gaussian(np.array([0.0, 5.0, 10.0]), 5.0)
    assert w[0] == 1.0
    assert w[1] == np.exp(-1.0)
    assert 0 < w[2] < w[1]

def test_spatial_kernel_fixed():
    from spatialstats.gwmodel.core.kernels import SpatialKernel
    kern = SpatialKernel("bisquare", fixed=True, bandwidth=10.0)
    d = np.array([0, 5, 10, 20])
    w = kern(d)
    assert w[3] == 0.0
    assert w[0] == 1.0

def test_spatial_kernel_adaptive():
    from spatialstats.gwmodel.core.kernels import SpatialKernel
    kern = SpatialKernel("bisquare", fixed=False, bandwidth=5)
    d = np.array([0, 2, 4, 6, 8, 10])
    w = kern(d)
    assert w[0] == 1.0
    assert np.all(w >= 0)

def test_anisotropic_kernel():
    from spatialstats.gwmodel.core.kernels import AnisotropicKernel
    kern = AnisotropicKernel("bisquare", fixed=True, bandwidth=20, angle=45, ratio=2.0)
    focal = np.array([50.0, 50.0])
    all_coords = np.random.rand(20, 2) * 100
    w = kern.weights_for_point(focal, all_coords)
    assert w.shape == (20,)
    assert np.all(w >= 0)


# ─── Distance Tests ───────────────────────────────────────────────────────────

def test_euclidean_distance():
    from spatialstats.gwmodel.utils.distance import euclidean_distance_matrix
    coords = np.array([[0, 0], [3, 4], [6, 8]], dtype=float)
    D = euclidean_distance_matrix(coords)
    assert D.shape == (3, 3)
    np.testing.assert_almost_equal(D[0, 1], 5.0)
    assert D[0, 0] == 0.0

def test_normalize_coords():
    from spatialstats.gwmodel.utils.distance import normalize_coords
    coords = np.array([[0, 0], [10, 20], [5, 10]], dtype=float)
    cn = normalize_coords(coords)
    assert cn.min() >= 0.0
    assert cn.max() <= 1.0

def test_coords_from_geometry_array():
    from spatialstats.gwmodel.core.base import _parse_geometry
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = _parse_geometry(coords)
    np.testing.assert_array_equal(out, coords)


# ─── Stats Utilities ─────────────────────────────────────────────────────────

def test_aicc():
    from spatialstats.gwmodel.utils.stats import aicc
    score = aicc(100.0, 50, 5.0)
    assert np.isfinite(score)

def test_bh_correction():
    from spatialstats.gwmodel.utils.stats import bh_correction
    p = np.array([0.001, 0.05, 0.2, 0.5, 0.8])
    sig = bh_correction(p, alpha=0.05)
    assert sig[0]  # clearly significant
    assert not sig[4]  # clearly not

def test_local_r2():
    from spatialstats.gwmodel.utils.stats import local_r2
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yhat = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
    w = np.ones(5)
    r2 = local_r2(y, yhat, w)
    assert 0.99 < r2 <= 1.0


# ─── Bandwidth Selector ───────────────────────────────────────────────────────

def test_bandwidth_selector_golden():
    from spatialstats.gwmodel.core.bandwidth import BandwidthSelector
    def score(bw): return (bw - 15) ** 2 + 5
    sel = BandwidthSelector(score, fixed=True, bw_min=5, bw_max=50)
    bw, sc = sel.select("golden_section")
    assert abs(bw - 15) < 2.0
    assert len(sel.bw_history_) > 0

def test_bandwidth_selector_grid():
    from spatialstats.gwmodel.core.bandwidth import BandwidthSelector
    def score(bw): return abs(bw - 10)
    sel = BandwidthSelector(score, fixed=False, bw_min=5, bw_max=30)
    bw, sc = sel.select("grid")
    assert abs(bw - 10) <= 2.0


# ─── GWR Tests ────────────────────────────────────────────────────────────────

def test_gwr_fit_predict():
    from spatialstats.gwmodel.linear import GWR
    X, y, coords = make_data(n=60, p=2)
    model = GWR(bandwidth=20, fixed=True, kernel="bisquare", hat_matrix=True)
    model.fit(X, y, coords)
    assert model.coef_.shape == (60, 3)  # intercept + 2
    assert model.residuals_.shape == (60,)
    assert np.isfinite(model.aicc_)
    assert model.effective_df_ > 0
    preds = model.predict(X, coords)
    assert preds.shape == (60,)

def test_gwr_summary():
    from spatialstats.gwmodel.linear import GWR
    X, y, coords = make_data(n=50, p=2)
    model = GWR(bandwidth=25, fixed=True)
    model.fit(X, y, coords)
    s = model.summary()
    assert "GWR" in s
    assert "AICc" in s

def test_gwr_local_significance():
    from spatialstats.gwmodel.linear import GWR
    X, y, coords = make_data(n=50, p=2)
    model = GWR(bandwidth=25, fixed=True)
    model.fit(X, y, coords)
    sig = model.local_significance(alpha=0.05)
    assert sig.shape == (50, 3)
    assert sig.dtype == bool

def test_gwr_sklearn_score():
    from spatialstats.gwmodel.linear import GWR
    X, y, coords = make_data(n=60, p=2)
    model = GWR(bandwidth=20, fixed=True)
    model.fit(X, y, coords)
    r2 = model.score(X, y, coords)
    assert -1 < r2 <= 1.0

def test_gwr_get_set_params():
    from spatialstats.gwmodel.linear import GWR
    model = GWR(bandwidth=10, fixed=True, kernel="gaussian")
    params = model.get_params()
    assert params["bandwidth"] == 10
    model.set_params(bandwidth=20)
    assert model.bandwidth == 20

def test_gwr_adaptive_bandwidth():
    from spatialstats.gwmodel.linear import GWR
    X, y, coords = make_data(n=50, p=2)
    model = GWR(bandwidth=15, fixed=False, kernel="bisquare")
    model.fit(X, y, coords)
    assert model.coef_.shape[0] == 50


# ─── MGWR Tests ───────────────────────────────────────────────────────────────

def test_mgwr_fit():
    from spatialstats.gwmodel.linear import MGWR
    X, y, coords = make_data(n=50, p=2)
    model = MGWR(kernel="bisquare", fixed=True, tol=5.0, max_iter=3)
    model.fit(X, y, coords)
    assert model.coef_.shape == (50, 3)
    assert model.bandwidths_.shape == (3,)
    assert np.all(model.bandwidths_ > 0)

def test_mgwr_summary():
    from spatialstats.gwmodel.linear import MGWR
    X, y, coords = make_data(n=50, p=2)
    model = MGWR(fixed=True, tol=5.0, max_iter=2)
    model.fit(X, y, coords)
    s = model.summary()
    assert "MGWR" in s
    assert "bandwidths" in s.lower()

def test_mgwr_predict():
    from spatialstats.gwmodel.linear import MGWR
    X, y, coords = make_data(n=50, p=2)
    model = MGWR(fixed=True, tol=5.0, max_iter=2)
    model.fit(X, y, coords)
    preds = model.predict(X, coords)
    assert preds.shape == (50,)


# ─── GWSS Tests ──────────────────────────────────────────────────────────────

def test_gwss_fit():
    from spatialstats.gwmodel.linear import GWSS
    X, _, coords = make_data(n=50, p=3)
    ss = GWSS(bandwidth=20, fixed=True)
    ss.fit(X, coords, var_names=["A", "B", "C"])
    assert ss.local_mean_.shape == (50, 3)
    assert ss.local_std_.shape == (50, 3)
    assert "A-B" in ss.local_corr_

def test_gwss_quantile():
    from spatialstats.gwmodel.linear import GWSS
    X, _, coords = make_data(n=40, p=2)
    ss = GWSS(bandwidth=15, fixed=True, quantile=True)
    ss.fit(X, coords)
    assert hasattr(ss, "local_median_")
    assert ss.local_median_.shape == (40, 2)

def test_gwss_to_dataframe():
    from spatialstats.gwmodel.linear import GWSS
    X, _, coords = make_data(n=40, p=2)
    ss = GWSS(bandwidth=15, fixed=True)
    ss.fit(X, coords, var_names=["X1", "X2"])
    df = ss.to_dataframe()
    assert "mean_X1" in df.columns
    assert len(df) == 40


# ─── GWGLM Tests ─────────────────────────────────────────────────────────────

def test_gwglm_gaussian():
    from spatialstats.gwmodel.linear import GWGLM
    X, y, coords = make_data(n=50, p=2)
    model = GWGLM(family="gaussian", bandwidth=20, fixed=True)
    model.fit(X, y, coords)
    assert model.coef_.shape == (50, 3)

def test_gwglm_poisson():
    from spatialstats.gwmodel.linear import GWGLM
    X, _, coords = make_data(n=50, p=2)
    lam = np.exp(0.5 + X[:, 0])
    y = np.random.poisson(lam)
    model = GWGLM(family="poisson", bandwidth=20, fixed=True)
    model.fit(X, y, coords)
    preds = model.predict(X, coords)
    assert np.all(preds > 0)

def test_gwglm_binomial():
    from spatialstats.gwmodel.linear import GWGLM
    X, _, coords = make_data(n=50, p=2)
    prob = 1 / (1 + np.exp(-X[:, 0]))
    y = np.random.binomial(1, prob)
    model = GWGLM(family="binomial", bandwidth=20, fixed=True)
    model.fit(X, y, coords)
    preds = model.predict(X, coords)
    assert np.all((preds >= 0) & (preds <= 1))


# ─── Mixed GWR Tests ─────────────────────────────────────────────────────────

def test_mixed_gwr():
    from spatialstats.gwmodel.linear import MixedGWR
    X, y, coords = make_data(n=60, p=3)
    model = MixedGWR(fixed_vars=[0], varying_vars=[1, 2],
                      bandwidth=20, fixed=True)
    model.fit(X, y, coords)
    assert model.global_coef_.shape == (1,)
    assert model.coef_.shape == (60, 3)  # intercept + 2 varying


# ─── GTWR Tests ───────────────────────────────────────────────────────────────

def test_gtwr_fit():
    from spatialstats.gwmodel.spatiotemporal import GTWR
    n = 60
    X, y, coords = make_data(n=n, p=2)
    times = np.random.uniform(2000, 2020, n)
    model = GTWR(bandwidth_s=25, lambda_=0.5, fixed=True)
    model.fit(X, y, coords, times)
    assert model.coef_.shape == (n, 3)
    preds = model.predict(X, coords, times)
    assert preds.shape == (n,)


# ─── GW Random Forest Tests ──────────────────────────────────────────────────

def test_gw_random_forest_fit_predict():
    from spatialstats.gwmodel.ensemble import GWRandomForest
    X, y, coords = make_data(n=80, p=3)
    model = GWRandomForest(n_estimators=20, bandwidth=25, fixed=True,
                            random_state=42, n_jobs=2)
    model.fit(X, y, coords)
    assert model.feature_importances_local_.shape == (80, 3)
    preds = model.predict(X, coords)
    assert preds.shape == (80,)
    assert np.all(np.isfinite(preds))

def test_gw_random_forest_importance_map():
    from spatialstats.gwmodel.ensemble import GWRandomForest
    X, y, coords = make_data(n=60, p=3)
    model = GWRandomForest(n_estimators=15, bandwidth=20, fixed=True,
                            random_state=0, n_jobs=2)
    model.fit(X, y, coords)
    imp = model.feature_importance_map()
    assert imp.shape == (60, 3)
    assert np.all(imp >= 0)

def test_gw_random_forest_score():
    from spatialstats.gwmodel.ensemble import GWRandomForest
    X, y, coords = make_data(n=60, p=3)
    model = GWRandomForest(n_estimators=15, bandwidth=25, fixed=True,
                            random_state=0, n_jobs=2)
    model.fit(X, y, coords)
    r2 = model.score(X, y, coords)
    assert -1 < r2 <= 1.0


# ─── GW XGBoost Tests ────────────────────────────────────────────────────────

def test_gw_xgboost():
    try:
        import xgboost
    except ImportError:
        print("    (skipped: xgboost not installed)")
        return
    from spatialstats.gwmodel.ensemble import GWXGBoost
    X, y, coords = make_data(n=60, p=3)
    model = GWXGBoost(n_estimators=20, bandwidth=25, fixed=True, n_jobs=2)
    model.fit(X, y, coords)
    preds = model.predict(X, coords)
    assert preds.shape == (60,)

def test_gw_lightgbm():
    try:
        import lightgbm
    except ImportError:
        print("    (skipped: lightgbm not installed)")
        return
    from spatialstats.gwmodel.ensemble import GWLightGBM
    X, y, coords = make_data(n=60, p=3)
    model = GWLightGBM(n_estimators=20, bandwidth=25, fixed=True, n_jobs=2)
    model.fit(X, y, coords)
    preds = model.predict(X, coords)
    assert preds.shape == (60,)


# ─── GNNWR Tests ─────────────────────────────────────────────────────────────

def test_gnnwr_fit_predict():
    try:
        import torch
    except ImportError:
        print("    (skipped: torch not installed)")
        return
    from spatialstats.gwmodel.deep import GNNWR
    X, y, coords = make_data(n=80, p=3)
    model = GNNWR(hidden_layers=[16, 8], n_epochs=5, batch_size=32,
                   random_state=42, patience=3)
    model.fit(X, y, coords)
    assert model.coef_.shape == (80, 4)  # intercept + 3
    preds = model.predict(X, coords)
    assert preds.shape == (80,)
    assert np.all(np.isfinite(preds))

def test_gnnwr_loss_tracked():
    try:
        import torch
    except ImportError:
        return
    from spatialstats.gwmodel.deep import GNNWR
    X, y, coords = make_data(n=60, p=2)
    model = GNNWR(hidden_layers=[8], n_epochs=5, random_state=0)
    model.fit(X, y, coords)
    assert len(model.train_loss_) > 0


# ─── GTNNWR Tests ────────────────────────────────────────────────────────────

def test_gtnnwr_fit_predict():
    try:
        import torch
    except ImportError:
        return
    from spatialstats.gwmodel.deep import GTNNWR
    X, y, coords = make_data(n=70, p=2)
    times = np.random.rand(70) * 10
    model = GTNNWR(spatial_layers=[8], temporal_layers=[4],
                    fusion_layers=[8], n_epochs=5, random_state=1)
    model.fit(X, y, coords, times)
    assert model.coef_.shape == (70, 3)
    preds = model.predict(X, coords, times)
    assert preds.shape == (70,)


# ─── GWPCA Tests ─────────────────────────────────────────────────────────────

def test_gwpca_fit():
    from spatialstats.gwmodel.multivariate import GWPCA
    X, _, coords = make_data(n=60, p=4)
    model = GWPCA(n_components=2, bandwidth=20, fixed=True)
    model.fit(X, coords)
    assert model.local_loadings_.shape == (60, 4, 2)
    assert model.local_eigenvalues_.shape == (60, 2)
    assert model.local_pve_.shape == (60, 2)
    assert np.all(model.local_pve_ >= 0)
    assert np.all(model.local_pve_ <= 1)

def test_gwpca_robust():
    from spatialstats.gwmodel.multivariate import GWPCA
    X, _, coords = make_data(n=50, p=3)
    model = GWPCA(n_components=2, bandwidth=20, fixed=True, robust=True)
    model.fit(X, coords)
    assert model.local_loadings_.shape == (50, 3, 2)


# ─── Diagnostics Tests ───────────────────────────────────────────────────────

def test_moran_test():
    from spatialstats.gwmodel.diagnostics import moran_test
    _, y, coords = make_data(n=80, p=2)
    result = moran_test(y - y.mean(), coords, k=6)
    assert "I" in result
    assert "p_value" in result
    assert np.isfinite(result["I"])

def test_f3_test():
    from spatialstats.gwmodel.diagnostics import f3_test
    from spatialstats.gwmodel.linear import GWR
    X, y, coords = make_data(n=60, p=2)
    model = GWR(bandwidth=25, fixed=True)
    model.fit(X, y, coords)
    hat_diag = np.full(60, model.effective_df_ / 60)
    result = f3_test(model.coef_, model.residuals_, hat_diag)
    assert "statistic" in result
    assert result["statistic"].shape == (3,)

def test_monte_carlo_variability():
    from spatialstats.gwmodel.diagnostics import monte_carlo_variability
    coef = np.random.randn(50, 3)
    result = monte_carlo_variability(coef, n_sim=49, seed=0)
    assert "p_value" in result
    assert result["p_value"].shape == (3,)


# ─── Dataset Tests ────────────────────────────────────────────────────────────

def test_load_georgia():
    from spatialstats.gwmodel.datasets import load_georgia
    gdf, features, target = load_georgia()
    assert len(gdf) == 159
    assert target in gdf.columns
    assert all(f in gdf.columns for f in features)

def test_load_londonhouse():
    from spatialstats.gwmodel.datasets import load_londonhouse
    gdf, features, target = load_londonhouse()
    assert len(gdf) == 500
    assert target == "Price"

def test_load_beijing_pm25():
    from spatialstats.gwmodel.datasets import load_beijing_pm25
    gdf, features, target = load_beijing_pm25()
    assert len(gdf) == 300
    assert "PM25" in gdf.columns

def test_load_ncovid():
    from spatialstats.gwmodel.datasets import load_ncovid
    gdf, features, target = load_ncovid()
    assert len(gdf) == 400
    assert "time" in gdf.columns


# ─── Integration Tests ───────────────────────────────────────────────────────

def test_gwr_on_georgia():
    from spatialstats.gwmodel.linear import GWR
    from spatialstats.gwmodel.datasets import load_georgia
    gdf, features, target = load_georgia()
    coords = gdf[["x", "y"]].values
    X = gdf[features].values
    y = gdf[target].values
    model = GWR(bandwidth=5.0, fixed=True, kernel="bisquare")
    model.fit(X, y, coords)
    r2 = model.score(X, y, coords)
    assert np.isfinite(r2)
    assert model.coef_.shape == (159, 7)  # 6 features + intercept

def test_gwrf_on_londonhouse():
    from spatialstats.gwmodel.ensemble import GWRandomForest
    from spatialstats.gwmodel.datasets import load_londonhouse
    gdf, features, target = load_londonhouse()
    coords = gdf[["x", "y"]].values[:100]
    X = gdf[features].values[:100]
    y = gdf[target].values[:100]
    model = GWRandomForest(n_estimators=15, bandwidth=0.2, fixed=True,
                            random_state=42, n_jobs=2)
    model.fit(X, y, coords)
    preds = model.predict(X, coords)
    assert len(preds) == 100
    assert np.all(np.isfinite(preds))


# ─── Compute Config Tests ────────────────────────────────────────────────────

def test_compute_config_defaults():
    from spatialstats.gwmodel.utils.compute import get_compute_config, resolve_n_jobs
    cfg = get_compute_config()
    assert cfg.n_jobs == -1
    assert resolve_n_jobs(-1) >= 1

def test_set_compute_options():
    from spatialstats.gwmodel.utils.compute import set_compute_options, get_compute_config
    cfg = set_compute_options(n_jobs=2, parallel_backend="processes", device="cpu")
    assert cfg.resolved_n_jobs() == 2
    assert cfg.joblib_prefer() == "processes"
    set_compute_options(n_jobs=-1, parallel_backend="threads", device="auto")

def test_gwr_with_compute_options():
    from spatialstats.gwmodel import set_compute_options
    from spatialstats.gwmodel.linear import GWR
    set_compute_options(n_jobs=2, parallel_backend="threads")
    X, y, coords = make_data(n=40, p=2)
    model = GWR(bandwidth=30, fixed=True, n_jobs=2)
    model.fit(X, y, coords)
    assert model.coef_.shape == (40, 3)


# ─── Run All Tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  PySpatialStats (gwmodel) Test Suite")
    print("=" * 65)

    test_groups = [
        ("Kernels", [
            ("bisquare kernel", test_bisquare),
            ("gaussian kernel", test_gaussian),
            ("SpatialKernel fixed", test_spatial_kernel_fixed),
            ("SpatialKernel adaptive", test_spatial_kernel_adaptive),
            ("AnisotropicKernel", test_anisotropic_kernel),
        ]),
        ("Distance & Utils", [
            ("euclidean_distance_matrix", test_euclidean_distance),
            ("normalize_coords", test_normalize_coords),
            ("coords_from_geometry (array)", test_coords_from_geometry_array),
            ("aicc", test_aicc),
            ("bh_correction", test_bh_correction),
            ("local_r2", test_local_r2),
        ]),
        ("Bandwidth Selection", [
            ("golden section search", test_bandwidth_selector_golden),
            ("grid search", test_bandwidth_selector_grid),
        ]),
        ("GWR", [
            ("fit + predict", test_gwr_fit_predict),
            ("summary()", test_gwr_summary),
            ("local_significance()", test_gwr_local_significance),
            ("score() (R²)", test_gwr_sklearn_score),
            ("get/set params", test_gwr_get_set_params),
            ("adaptive bandwidth", test_gwr_adaptive_bandwidth),
        ]),
        ("MGWR", [
            ("fit", test_mgwr_fit),
            ("summary()", test_mgwr_summary),
            ("predict()", test_mgwr_predict),
        ]),
        ("GWSS", [
            ("fit", test_gwss_fit),
            ("quantile stats", test_gwss_quantile),
            ("to_dataframe()", test_gwss_to_dataframe),
        ]),
        ("GWGLM", [
            ("Gaussian family", test_gwglm_gaussian),
            ("Poisson family", test_gwglm_poisson),
            ("Binomial family", test_gwglm_binomial),
        ]),
        ("MixedGWR", [
            ("fit + predict", test_mixed_gwr),
        ]),
        ("GTWR", [
            ("fit + predict", test_gtwr_fit),
        ]),
        ("GW Random Forest", [
            ("fit + predict", test_gw_random_forest_fit_predict),
            ("feature_importance_map()", test_gw_random_forest_importance_map),
            ("score()", test_gw_random_forest_score),
        ]),
        ("GW Gradient Boosting", [
            ("GWXGBoost", test_gw_xgboost),
            ("GWLightGBM", test_gw_lightgbm),
        ]),
        ("GNNWR (Deep Learning)", [
            ("fit + predict", test_gnnwr_fit_predict),
            ("loss tracked", test_gnnwr_loss_tracked),
        ]),
        ("GTNNWR (Deep ST)", [
            ("fit + predict", test_gtnnwr_fit_predict),
        ]),
        ("GWPCA", [
            ("fit", test_gwpca_fit),
            ("robust", test_gwpca_robust),
        ]),
        ("Diagnostics", [
            ("moran_test", test_moran_test),
            ("f3_test", test_f3_test),
            ("monte_carlo_variability", test_monte_carlo_variability),
        ]),
        ("Datasets", [
            ("load_georgia", test_load_georgia),
            ("load_londonhouse", test_load_londonhouse),
            ("load_beijing_pm25", test_load_beijing_pm25),
            ("load_ncovid", test_load_ncovid),
        ]),
        ("Integration", [
            ("GWR on Georgia", test_gwr_on_georgia),
            ("GWRandomForest on LondonHouse", test_gwrf_on_londonhouse),
        ]),
        ("Compute", [
            ("defaults", test_compute_config_defaults),
            ("set_compute_options", test_set_compute_options),
            ("GWR with n_jobs", test_gwr_with_compute_options),
        ]),
    ]

    for group_name, tests in test_groups:
        print(f"\n{'─'*65}")
        print(f"  {group_name}")
        print(f"{'─'*65}")
        for test_name, test_fn in tests:
            run_test(test_name, test_fn)

    print(f"\n{'='*65}")
    print(f"  Results: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print(f"\n  Failed tests:")
        for name, err in FAIL:
            print(f"    ✗ {name}: {err}")
    print("=" * 65)
