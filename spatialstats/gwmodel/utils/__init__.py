from .distance import (euclidean_distance_matrix, haversine_distance_matrix,
                        pairwise_distances, coords_from_geometry, normalize_coords)
from .stats import aicc, aic, bic, hat_matrix_gwr, effective_df, local_r2, bh_correction, t_to_p
from .parallel import parallel_calibrate, chunk_indices, parallel_map
from .compute import (
    ComputeConfig, get_compute_config, set_compute_options,
    resolve_n_jobs, resolve_torch_device, configure_torch_threads,
)

__all__ = [
    "euclidean_distance_matrix", "haversine_distance_matrix", "pairwise_distances",
    "coords_from_geometry", "normalize_coords",
    "aicc", "aic", "bic", "hat_matrix_gwr", "effective_df", "local_r2",
    "bh_correction", "t_to_p",
    "parallel_calibrate", "chunk_indices", "parallel_map",
    "ComputeConfig", "get_compute_config", "set_compute_options",
    "resolve_n_jobs", "resolve_torch_device", "configure_torch_threads",
]
