"""
spatialstats.cluster.compare
============================
Agreement between cluster / hot-spot results and stability across settings.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def jaccard(a, b) -> float:
    """Jaccard index of two boolean masks (1 = identical sets of flagged areas; NaN if both empty)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else np.nan


def agreement_matrix(masks: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Jaccard indices between named boolean masks (e.g. 'Gi*' hot spots vs 'LISA' HH clusters)."""
    names = list(masks)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            out.loc[i, j] = jaccard(masks[i], masks[j])
    return out


def partition_agreement(labels_a, labels_b) -> dict:
    """Adjusted Rand index and normalised mutual information between two partitions (e.g. two regionalisations)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return {"adjusted_rand": float(adjusted_rand_score(labels_a, labels_b)),
            "nmi": float(normalized_mutual_info_score(labels_a, labels_b))}


def flag_stability(runs: Sequence[np.ndarray]) -> np.ndarray:
    """
    Share of runs in which each area is flagged. ``runs`` is a sequence of boolean masks obtained under
    different weights / seeds / thresholds; areas flagged in every run (1.0) are robust findings, those
    flagged in a few are artefacts of the settings.
    """
    return np.mean(np.stack([np.asarray(r, dtype=bool) for r in runs]), axis=0)
