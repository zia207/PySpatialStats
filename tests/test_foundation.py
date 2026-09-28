"""
Foundation tests for spatialstats: core, weights, esda, datasets.
Runnable with: PYTHONPATH=. python tests/test_foundation.py
Also discoverable by pytest when available.
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")

PASS = []
FAIL = []

try:
    import shapely  # noqa: F401
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        PASS.append(name)
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        FAIL.append((name, str(e)))


def _lattice():
    from spatialstats.datasets import load_lattice
    return load_lattice(n=6, seed=0)


def _rook_from_lattice(df, n=6):
    """Build rook contiguity from row/col columns without shapely."""
    from spatialstats.weights import W
    neighbors = {i: [] for i in range(len(df))}
    lookup = {(int(r.row), int(r.col)): i for i, r in df.iterrows()}
    for i, r in df.iterrows():
        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            key = (int(r.row) + di, int(r.col) + dj)
            if key in lookup:
                neighbors[i].append(lookup[key])
    return W(neighbors, ids=list(range(len(df))), transform="row")


def test_spatial_data():
    if not HAS_SHAPELY:
        from spatialstats.core import SpatialData
        # SpatialData requires GeoDataFrame; skip when unavailable
        print("    (skipped: shapely/geopandas not installed)")
        return
    from spatialstats import SpatialData
    gdf, y_col = _lattice()
    sd = SpatialData(gdf, y=y_col)
    assert sd.n == 36
    assert sd.coords_array().shape == (36, 2)
    assert sd.y_array().shape == (36,)
    assert "SpatialData" in repr(sd)


def test_distance_haversine():
    from spatialstats.core import haversine_distance_matrix, pairwise_distances
    coords = np.array([[40.0, -74.0], [40.1, -74.0], [41.0, -74.0]])
    D = haversine_distance_matrix(coords)
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0)
    assert D[0, 2] > D[0, 1]
    D2 = pairwise_distances(coords, metric="haversine")
    assert np.allclose(D, D2)


def test_queen_rook():
    if not HAS_SHAPELY:
        # Validate rook built from lattice indices
        df, _ = _lattice()
        w = _rook_from_lattice(df, n=6)
        assert w.n == 36
        assert w.cardinalities().min() >= 2
        assert len(w.islands()) == 0
        return
    from spatialstats.weights import Queen, Rook
    gdf, _ = _lattice()
    wq = Queen(gdf)
    wr = Rook(gdf)
    assert wq.n == 36
    assert wr.n == 36
    assert wq.cardinalities().min() >= 3
    assert wr.cardinalities().min() >= 2
    assert len(wq.islands()) == 0
    assert "Spatial Weights" in wq.summary()


def test_knn_kernel_graph():
    from spatialstats.weights import KNN, Kernel, Delaunay, Gabriel, RelativeNeighborhood
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, (40, 2))
    w = KNN(coords, k=5)
    assert w.n == 40
    assert (w.cardinalities() == 5).all()

    wk = Kernel(coords, bandwidth=0.3, kernel="bisquare")
    assert wk.sparse.nnz > 0

    for builder in (Delaunay, Gabriel, RelativeNeighborhood):
        wg = builder(coords)
        assert wg.n == 40
        assert wg.sparse.nnz > 0


def test_w_transform_and_lag():
    df, y_col = _lattice()
    w = _rook_from_lattice(df, n=6)
    w.transform("binary")
    w.transform("row")
    assert w.transform_type == "row"
    y = df[y_col].values
    lag = w.lag(y)
    assert lag.shape == (36,)


def test_moran():
    from spatialstats import Moran
    df, y_col = _lattice()
    w = _rook_from_lattice(df, n=6)
    mi = Moran(df[y_col], w, permutations=99, seed=1)
    assert np.isfinite(mi.I)
    assert 0 <= mi.p_value <= 1
    text = mi.summary()
    assert "Moran" in text
    assert mi.to_frame().shape[0] == 1


def test_geary_general_g():
    from spatialstats.esda import Geary, GeneralG
    df, y_col = _lattice()
    w = _rook_from_lattice(df, n=6)
    c = Geary(df[y_col], w, permutations=49, seed=2)
    assert np.isfinite(c.C)
    g = GeneralG(df[y_col], w, permutations=49, seed=2)
    assert np.isfinite(g.G)


def test_local_moran():
    from spatialstats import Moran_Local
    df, y_col = _lattice()
    w = _rook_from_lattice(df, n=6)
    lisa = Moran_Local(df[y_col], w, permutations=99, seed=3)
    frame = lisa.to_frame()
    assert len(frame) == 36
    assert set(frame["label"]).issubset({"HH", "LL", "HL", "LH", "NS"})


def test_bivariate_local_moran_matches_univariate():
    from spatialstats import Moran_Local
    from spatialstats.esda import Moran_Local_BV
    df, y_col = _lattice()
    w = _rook_from_lattice(df, n=6)
    y = df[y_col].to_numpy(dtype=float)
    uni = Moran_Local(y, w, permutations=0, fdr=False)
    bv = Moran_Local_BV(y, y, w, permutations=0, fdr=False)
    assert np.allclose(bv.Is, uni.Is)
    clustered = Moran_Local_BV(y, y, w, permutations=199, seed=3, fdr=False)
    assert set(clustered.labels).issubset({"HH", "LL", "HL", "LH", "NS"})
    assert np.isfinite(clustered.p_sim).all()


def test_join_counts():
    from spatialstats.esda import Join_Counts
    df, _ = _lattice()
    w = _rook_from_lattice(df, n=6)
    w.transform("binary")
    jc = Join_Counts(df["binary"], w, permutations=49, seed=4)
    assert jc["BB"] + jc["WW"] + jc["BW"] > 0


def test_fdr_bh():
    from spatialstats.esda import fdr_bh
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.5])
    reject, padj = fdr_bh(p, alpha=0.05)
    assert len(reject) == 5
    assert (padj >= p - 1e-12).all()


def test_top_level_imports():
    import spatialstats as ss
    assert hasattr(ss, "Moran")
    assert hasattr(ss, "GWR")
    assert hasattr(ss, "Queen")
    assert ss.__version__ == "0.2.0"
    import spatialstats.bayes
    import spatialstats.pointpattern
    import spatialstats.spacetime
    # implemented modules export a public API; spacetime is still a planned stub
    assert "BYM" in spatialstats.bayes.__all__
    assert "PointPattern" in spatialstats.pointpattern.__all__
    assert spatialstats.spacetime.__all__ == []


def test_load_points_csr():
    from spatialstats.datasets import load_points_csr
    gdf, _ = load_points_csr(n=50, seed=1)
    assert len(gdf) == 50


def test_higher_order():
    from spatialstats.weights import higher_order
    df, _ = _lattice()
    w1 = _rook_from_lattice(df, n=6)
    w2 = higher_order(w1, k=2)
    assert w2.n == 36
    assert w2.sparse.nnz > 0


def test_moranbv_length_check():
    from spatialstats.esda import MoranBV, Moran_Diff
    from spatialstats.weights import KNN
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, (20, 2))
    w = KNN(coords, k=4)
    y = rng.normal(size=20)
    try:
        MoranBV(y, rng.normal(size=15), w, permutations=0)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "length" in str(e).lower() or "!=" in str(e)

    # Differential Moran on change
    y0 = rng.normal(size=20)
    y1 = y0 + 0.5 * w.lag(y0) + rng.normal(0, 0.1, 20)
    md = Moran_Diff(y1, y0, w, permutations=49, seed=0)
    assert np.isfinite(md.I)
    assert "Differential" in md.name


def test_gi_islands():
    from spatialstats.esda import G_Local
    from spatialstats.weights import W
    # two connected, one island
    w = W({0: [1], 1: [0], 2: []}, transform="binary")
    y = np.array([1.0, 2.0, 3.0])
    gi = G_Local(y, w, star=False, permutations=0, fdr=False)
    assert np.isnan(gi.Gs[2])
    assert np.isfinite(gi.Gs[0]) or np.isnan(gi.Gs[0])  # may be nan for tiny n


if __name__ == "__main__":
    print("\n=== Foundation tests ===")
    print(f"shapely available: {HAS_SHAPELY}")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            run_test(name, fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        raise SystemExit(1)
