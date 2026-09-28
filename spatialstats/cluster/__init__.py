"""
spatialstats.cluster
====================
Spatial cluster analysis. "Cluster" means different things, so the module offers one family
of methods per question:

=========================================  ==================================================
Question                                   Tools
=========================================  ==================================================
Where are high (low) values grouped?       :func:`getis_ord_gi` (hot / cold spots),
                                           :func:`local_moran_clusters` (HH / LL / HL / LH)
Do *cases* exceed what population implies? :func:`spatial_scan` (Kulldorff scan statistic)
Where are *events* densely packed?         :func:`dbscan_clusters`, :func:`hdbscan_clusters`
Which areas form similar contiguous zones? :func:`skater`, :func:`agglomerative`
How much do methods / settings agree?      :func:`agreement_matrix`, :func:`flag_stability`
=========================================  ==================================================

Local statistics use *conditional* permutation and correct for multiple testing (FDR by default).
"""

from spatialstats.cluster.hotspot import (
    getis_ord_gi, HotSpotResult, local_moran_clusters, LisaClusterResult,
)
from spatialstats.cluster.scan import spatial_scan, ScanResult
from spatialstats.cluster.points import (
    dbscan_clusters, hdbscan_clusters, k_distance, PointClusterResult,
)
from spatialstats.cluster.regions import (
    skater, agglomerative, evaluate_regions, RegionResult,
)
from spatialstats.cluster.compare import (
    jaccard, agreement_matrix, partition_agreement, flag_stability,
)
from spatialstats.cluster._perm import adjust_pvalues

__all__ = [
    "getis_ord_gi", "HotSpotResult", "local_moran_clusters", "LisaClusterResult",
    "spatial_scan", "ScanResult",
    "dbscan_clusters", "hdbscan_clusters", "k_distance", "PointClusterResult",
    "skater", "agglomerative", "evaluate_regions", "RegionResult",
    "jaccard", "agreement_matrix", "partition_agreement", "flag_stability",
    "adjust_pvalues",
]
