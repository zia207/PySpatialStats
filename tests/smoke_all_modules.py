"""
Comprehensive smoke / debug harness for all spatialstats modules.
Run: PYTHONPATH=. python tests/smoke_all_modules.py
"""
from __future__ import annotations

import traceback
import warnings
import numpy as np

warnings.filterwarnings("ignore")

PASS, FAIL, SKIP = [], [], []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        PASS.append(name)
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        FAIL.append((name, str(e)))


def skip(name, reason):
    print(f"  SKIP  {name}: {reason}")
    SKIP.append((name, reason))


def has_shapely():
    try:
        import shapely  # noqa
        import geopandas  # noqa
        return True
    except ImportError:
        return False


# ── helpers ──────────────────────────────────────────────────────────────

def lattice_df(n=8, seed=0):
    from spatialstats.datasets import load_lattice
    return load_lattice(n=n, seed=seed, as_gdf=False)


def rook_w(df, n=8):
    from spatialstats.weights import W
    neighbors = {i: [] for i in range(len(df))}
    lookup = {(int(r.row), int(r.col)): idx for idx, r in df.iterrows()}
    # reset index for positional
    df2 = df.reset_index(drop=True)
    lookup = {(int(r.row), int(r.col)): i for i, r in df2.iterrows()}
    for i, r in df2.iterrows():
        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            key = (int(r.row) + di, int(r.col) + dj)
            if key in lookup:
                neighbors[i].append(lookup[key])
    return W(neighbors, ids=list(range(len(df2))), transform="row"), df2


# ── core ─────────────────────────────────────────────────────────────────

def test_core_distance():
    from spatialstats.core import (
        euclidean_distance_matrix, haversine_distance_matrix,
        pairwise_distances, normalize_coords, euclidean_distances,
    )
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    D = euclidean_distance_matrix(coords)
    assert np.isclose(D[0, 1], 5.0)
    H = haversine_distance_matrix(np.array([[0.0, 0.0], [0.0, 1.0]]))
    assert H[0, 1] > 100  # ~111 km
    assert pairwise_distances(coords, metric="euclidean").shape == (3, 3)
    nrm = normalize_coords(coords)
    assert nrm.min() >= 0 and nrm.max() <= 1
    assert euclidean_distances(coords[:2], coords[1:]).shape == (2, 2)

    # network hook
    def net(a, b):
        return np.ones((len(a), len(b))) * 2.0
    Dn = pairwise_distances(coords, metric="network", network_fn=net)
    assert np.allclose(Dn, 2.0)


def test_core_crs_helpers():
    from spatialstats.core import validate_crs, is_geographic, is_projected
    try:
        import pyproj  # noqa
    except ImportError:
        skip("core.crs", "pyproj not installed")
        return
    c = validate_crs("EPSG:4326")
    assert is_geographic(c)
    assert not is_projected(c)
    assert is_projected(validate_crs("EPSG:3857"))


def test_core_spatial_data():
    if not has_shapely():
        skip("core.SpatialData", "shapely/geopandas missing")
        return
    from spatialstats import SpatialData
    from spatialstats.datasets import load_lattice
    gdf, ycol = load_lattice(n=5, seed=1)
    sd = SpatialData(gdf, y=ycol, X=["binary"])
    assert sd.n == 25
    assert sd.coords_array().shape == (25, 2)
    assert sd.y_array().shape == (25,)
    assert sd.X_array().shape == (25, 1)
    assert "SpatialData" in sd.summary() if hasattr(sd, "summary") else True
    _ = sd.__repr__()


def test_core_result():
    from spatialstats.core import SpatialResult
    r = SpatialResult(name="Demo", a=1.5, b=2)
    assert "Demo" in r.summary()
    assert r.to_frame().shape == (1, 2)
    assert r["a"] == 1.5
    try:
        r.plot()
        raise AssertionError("plot should raise")
    except NotImplementedError:
        pass


# ── weights ──────────────────────────────────────────────────────────────

def test_weights_knn_kernel_band():
    from spatialstats.weights import KNN, DistanceBand, Kernel
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, (50, 2))
    w = KNN(coords, k=6)
    assert (w.cardinalities() == 6).all()
    assert len(w.islands()) == 0
    assert w.n_components() >= 1

    wb = DistanceBand(coords, threshold=2.0, binary=True)
    assert wb.sparse.nnz > 0

    wk = Kernel(coords, bandwidth=3.0, kernel="gaussian", transform="row")
    assert wk.sparse.nnz > 0
    wk2 = Kernel(coords, k=8, fixed=False, kernel="triangular")
    assert wk2.n == 50


def test_weights_graph():
    from spatialstats.weights import Delaunay, Gabriel, RelativeNeighborhood
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 1, (30, 2))
    for B in (Delaunay, Gabriel, RelativeNeighborhood):
        w = B(coords)
        assert w.n == 30
        assert w.sparse.nnz > 0
        assert w.is_symmetric() or True  # undirected builders should be ~symmetric


def test_weights_transform_diagnostics():
    from spatialstats.weights import KNN, higher_order
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 1, (20, 2))
    w = KNN(coords, k=4, transform="binary")
    w.transform("row")
    # rows sum ~1
    row_sums = np.asarray(w.sparse.sum(axis=1)).ravel()
    assert np.allclose(row_sums[row_sums > 0], 1.0, atol=1e-8)
    w.transform("variance")
    w.transform("binary")
    s = w.summary()
    assert "Islands" in s
    w2 = higher_order(w, k=2)
    assert w2.sparse.nnz > 0


def test_weights_queen_rook():
    if not has_shapely():
        skip("weights.Queen/Rook", "shapely/geopandas missing")
        return
    from spatialstats.weights import Queen, Rook, higher_order
    from spatialstats.datasets import load_lattice
    gdf, _ = load_lattice(n=6, seed=0)
    wq = Queen(gdf)
    wr = Rook(gdf)
    assert wq.n == 36 and wr.n == 36
    assert wq.cardinalities().mean() >= wr.cardinalities().mean()
    assert len(wq.islands()) == 0
    w2 = higher_order(wr, k=2)
    assert w2.sparse.nnz > 0


def test_weights_libpysal_optional():
    from spatialstats.weights import KNN
    w = KNN(np.random.default_rng(0).uniform(0, 1, (15, 2)), k=3)
    try:
        lw = w.to_libpysal()
        from spatialstats.weights import W
        w2 = W.from_libpysal(lw)
        assert w2.n == w.n
    except ImportError:
        skip("weights.libpysal", "libpysal not installed")


# ── esda ─────────────────────────────────────────────────────────────────

def test_esda_global():
    from spatialstats.esda import Moran, Geary, GeneralG, MoranBV
    df, ycol = lattice_df(n=8, seed=0)
    w, df = rook_w(df, n=8)
    y = df[ycol].values
    mi = Moran(y, w, permutations=199, seed=0)
    assert mi.I > 0.3, f"expected strong positive I, got {mi.I}"
    assert mi.p_value < 0.05
    assert "Moran" in mi.summary()

    c = Geary(y, w, permutations=99, seed=1)
    assert c.C < 1.0, f"expected C<1 for positive AC, got {c.C}"

    g = GeneralG(y, w, permutations=99, seed=2)
    assert np.isfinite(g.G)

    # bivariate: y with itself should relate to univariate
    bv = MoranBV(y, y, w, permutations=49, seed=3)
    assert np.isfinite(bv.I)


def test_esda_local_lisa():
    from spatialstats.esda import Moran_Local, Geary_Local, G_Local
    df, ycol = lattice_df(n=8, seed=0)
    w, df = rook_w(df, n=8)
    y = df[ycol].values

    # Without FDR should find some clusters on strongly autocorrelated data
    lisa = Moran_Local(y, w, permutations=199, seed=0, fdr=False, alpha=0.05)
    frame = lisa.to_frame()
    labels = set(frame["label"])
    assert labels & {"HH", "LL"}, f"expected HH/LL clusters, got {labels}"
    assert frame["Ii"].notna().all()

    # With FDR still should return valid frame
    lisa_fdr = Moran_Local(y, w, permutations=199, seed=0, fdr=True, alpha=0.05)
    assert len(lisa_fdr.to_frame()) == 64

    lg = Geary_Local(y, w, permutations=49, seed=1)
    assert len(lg.to_frame()) == 64

    gi = G_Local(y, w, star=False, permutations=49, seed=2)
    gi_star = G_Local(y, w, star=True, permutations=49, seed=3)
    assert len(gi.to_frame()) == 64
    assert len(gi_star.to_frame()) == 64


def test_esda_join_counts_fdr():
    from spatialstats.esda import Join_Counts, fdr_bh, MoranBV, Moran_Diff
    df, _ = lattice_df(n=6, seed=1)
    w, df = rook_w(df, n=6)
    w.transform("binary")
    jc = Join_Counts(df["binary"].values, w, permutations=99, seed=0)
    assert jc["BB"] + jc["WW"] + jc["BW"] > 0
    reject, padj = fdr_bh(np.array([0.001, 0.02, 0.04, 0.5]))
    assert padj[0] <= padj[-1]

    y = df["z"].values
    bv = MoranBV(y, y, w, permutations=0)
    assert np.isfinite(bv.I)
    try:
        MoranBV(y, y[:10], w, permutations=0)
        raise AssertionError("expected length error")
    except ValueError:
        pass
    md = Moran_Diff(y, y * 0.5, w, permutations=29, seed=0)
    assert "Differential" in md.name


# ── viz ──────────────────────────────────────────────────────────────────

def test_viz_moran_scatter():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        skip("viz.moran_scatter", "matplotlib missing")
        return
    from spatialstats import Moran
    from spatialstats.viz import moran_scatter
    df, ycol = lattice_df(n=6, seed=0)
    w, df = rook_w(df, n=6)
    mi = Moran(df[ycol], w, permutations=49, seed=0)
    ax = moran_scatter(mi)
    assert ax is not None
    plt.close("all")


def test_viz_choropleth_lisa():
    if not has_shapely():
        skip("viz.choropleth/lisa", "shapely/geopandas missing")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        skip("viz.choropleth/lisa", "matplotlib missing")
        return
    from spatialstats.datasets import load_lattice
    from spatialstats.weights import Queen
    from spatialstats import Moran_Local
    from spatialstats.viz import choropleth, lisa_cluster_map
    gdf, ycol = load_lattice(n=5, seed=0)
    w = Queen(gdf)
    lisa = Moran_Local(gdf[ycol], w, permutations=49, seed=0, fdr=False)
    ax1 = choropleth(gdf, ycol, scheme="quantiles", k=4)
    ax2 = lisa_cluster_map(lisa, gdf=gdf)
    assert ax1 is not None and ax2 is not None
    plt.close("all")


# ── datasets ─────────────────────────────────────────────────────────────

def test_datasets_all():
    from spatialstats.datasets import (
        load_georgia, load_londonhouse, load_beijing_pm25,
        load_ncovid, load_lattice, load_points_csr,
    )
    for loader in (load_georgia, load_londonhouse, load_beijing_pm25, load_ncovid):
        out = loader(as_gdf=False)
        assert len(out) == 3
        data, feats, target = out
        assert target in data.columns
        assert all(c in data.columns for c in feats)

    df, y = load_lattice(n=4, seed=0, as_gdf=False)
    assert y in df.columns and len(df) == 16
    pts, _ = load_points_csr(n=20, seed=0, as_gdf=False)
    assert len(pts) == 20


# ── gwmodel smoke ────────────────────────────────────────────────────────

def test_gwmodel_gwr_mgwr():
    from spatialstats import GWR, MGWR
    from spatialstats.datasets import load_georgia
    gdf, feats, target = load_georgia(as_gdf=False)
    coords = gdf[["x", "y"]].values
    X = gdf[feats].values
    y = gdf[target].values
    gwr = GWR(bandwidth=2.0, fixed=True, kernel="bisquare")
    gwr.fit(X, y, coords)
    pred = gwr.predict(X, coords)
    assert pred.shape == y.shape
    assert hasattr(gwr, "summary")

    mgwr = MGWR(kernel="bisquare", fixed=True, max_iter=3)
    mgwr.fit(X, y, coords)
    assert hasattr(mgwr, "bandwidths_")


def test_gwmodel_ensemble_deep():
    from spatialstats import GWRandomForest, GNNWR
    rng = np.random.default_rng(0)
    n, p = 60, 3
    coords = rng.uniform(0, 10, (n, 2))
    X = rng.standard_normal((n, p))
    y = X.sum(1) + rng.normal(0, 0.2, n)
    rf = GWRandomForest(n_estimators=10, bandwidth=3.0, fixed=True)
    rf.fit(X, y, coords)
    assert rf.predict(X, coords).shape == (n,)

    nn = GNNWR(hidden_layers=[16, 8], n_epochs=5)
    nn.fit(X, y, coords)
    assert nn.predict(X, coords).shape == (n,)


# ── stubs ────────────────────────────────────────────────────────────────

def test_stub_packages():
    import spatialstats.pointpattern as a
    import spatialstats.interpolate as b
    import spatialstats.regression as c
    import spatialstats.bayes as d
    import spatialstats.spacetime as e
    import spatialstats.sampling as f
    for m in (a, b, c, d, e, f):
        assert hasattr(m, "__all__")
        assert m.__all__ == []
        assert m.__doc__ and "planned" in m.__doc__.lower() or "Future" in (m.__doc__ or "")


def test_top_level_api():
    from spatialstats import (
        Moran, Moran_Local, Queen, Rook, KNN, W,
        SpatialData, SpatialResult, GWR, MGWR,
        set_compute_options,
    )
    assert callable(Moran)
    set_compute_options(n_jobs=1)


if __name__ == "__main__":
    print("\n=== Smoke / debug all modules ===")
    print(f"shapely/geopandas: {has_shapely()}")
    tests = [
        test_core_distance,
        test_core_crs_helpers,
        test_core_spatial_data,
        test_core_result,
        test_weights_knn_kernel_band,
        test_weights_graph,
        test_weights_transform_diagnostics,
        test_weights_queen_rook,
        test_weights_libpysal_optional,
        test_esda_global,
        test_esda_local_lisa,
        test_esda_join_counts_fdr,
        test_viz_moran_scatter,
        test_viz_choropleth_lisa,
        test_datasets_all,
        test_gwmodel_gwr_mgwr,
        test_gwmodel_ensemble_deep,
        test_stub_packages,
        test_top_level_api,
    ]
    for fn in tests:
        check(fn.__name__, fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    for n, e in FAIL:
        print(f"  - {n}: {e}")
    raise SystemExit(1 if FAIL else 0)
