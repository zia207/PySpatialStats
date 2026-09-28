"""
spatialstats.gwmodel.utils.compute
=======================
Unified CPU/GPU compute configuration for heavy workloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

ParallelBackend = Literal["threads", "processes"]
DeviceChoice = Literal["auto", "cpu", "cuda", "mps"]


def _cpu_count() -> int:
    return os.cpu_count() or 1


def resolve_n_jobs(n_jobs: int) -> int:
    """Map ``-1`` / ``0`` to the number of available CPUs."""
    if n_jobs <= 0:
        return _cpu_count()
    return n_jobs


def resolve_torch_device(device: DeviceChoice = "auto"):
    """
    Return a ``torch.device`` for deep-learning models.

    Priority when ``device='auto'``: CUDA → MPS (Apple) → CPU.
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def configure_torch_threads(num_threads: Optional[int] = None) -> int:
    """
    Set PyTorch / OpenMP thread counts for CPU inference/training.

    Returns the effective thread count.
    """
    import torch

    n = resolve_n_jobs(num_threads if num_threads is not None else -1)
    torch.set_num_threads(n)
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(env_var, str(n))
    return n


@dataclass
class ComputeConfig:
    """
    Global compute settings for PyGWmodel heavy workloads.

    Parameters
    ----------
    n_jobs : int
        Parallel workers for CPU-bound models (GWR, GW-RF, etc.).
        ``-1`` uses all available cores.
    parallel_backend : str
        Joblib backend: ``'threads'`` (default) or ``'processes'``.
    device : str
        PyTorch device for GNNWR / GTNNWR: ``'auto'``, ``'cpu'``,
        ``'cuda'``, or ``'mps'``.
    torch_threads : int or None
        CPU threads for PyTorch. ``None`` follows ``n_jobs``.
    xgboost_device : str or None
        XGBoost device (``'cuda'``, ``'cpu'``). ``None`` = auto-detect.
    lightgbm_device : str
        LightGBM device type (``'gpu'`` or ``'cpu'``).
    """

    n_jobs: int = -1
    parallel_backend: ParallelBackend = "threads"
    device: DeviceChoice = "auto"
    torch_threads: Optional[int] = None
    xgboost_device: Optional[str] = None
    lightgbm_device: str = "cpu"

    def resolved_n_jobs(self) -> int:
        return resolve_n_jobs(self.n_jobs)

    def resolved_torch_threads(self) -> int:
        return resolve_n_jobs(
            self.torch_threads if self.torch_threads is not None else self.n_jobs
        )

    def joblib_prefer(self) -> str:
        return self.parallel_backend

    def xgboost_params(self) -> dict:
        device = self.xgboost_device
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if device == "cuda":
            return {"device": "cuda", "tree_method": "hist"}
        return {"device": "cpu", "tree_method": "hist"}

    def lightgbm_params(self) -> dict:
        if self.lightgbm_device == "gpu":
            return {"device": "gpu"}
        return {"device": "cpu"}


# Module-level default; override with ``set_compute_options()``.
_DEFAULT = ComputeConfig()


def get_compute_config() -> ComputeConfig:
    """Return the active global compute configuration."""
    return _DEFAULT


def set_compute_options(
    n_jobs: int = -1,
    parallel_backend: ParallelBackend = "threads",
    device: DeviceChoice = "auto",
    torch_threads: Optional[int] = None,
    xgboost_device: Optional[str] = None,
    lightgbm_device: str = "cpu",
) -> ComputeConfig:
    """
    Configure global CPU/GPU options for all subsequent model fits.

    Examples
    --------
    >>> from spatialstats.gwmodel.utils.compute import set_compute_options
    >>> set_compute_options(n_jobs=-1, device="cuda")          # all CPU cores + GPU
    >>> set_compute_options(n_jobs=8, parallel_backend="processes")
    """
    global _DEFAULT
    _DEFAULT = ComputeConfig(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        device=device,
        torch_threads=torch_threads,
        xgboost_device=xgboost_device,
        lightgbm_device=lightgbm_device,
    )
    if device in ("cpu", "auto"):
        configure_torch_threads(_DEFAULT.resolved_torch_threads())
    return _DEFAULT


def apply_model_defaults(params: dict, *, use_device: bool = False) -> dict:
    """
    Merge global ``ComputeConfig`` into model constructor kwargs.

    Only fills keys that are absent from ``params``.
    """
    cfg = get_compute_config()
    out = dict(params)
    out.setdefault("n_jobs", cfg.n_jobs)
    if use_device:
        out.setdefault("device", cfg.device)
    return out
