"""
spatialstats.gwmodel.core.kernels
======================
Spatial kernel (weighting) functions for geographically weighted models.
"""

import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, float]


def gaussian(d: ArrayLike, h: float) -> np.ndarray:
    """Gaussian kernel: w = exp(-(d/h)^2)."""
    d = np.asarray(d, dtype=float)
    return np.exp(-(d / h) ** 2)


def bisquare(d: ArrayLike, h: float) -> np.ndarray:
    """Bi-square kernel: w = (1-(d/h)^2)^2 for d < h, else 0."""
    d = np.asarray(d, dtype=float)
    return np.where(d < h, (1.0 - (d / h) ** 2) ** 2, 0.0)


def tricube(d: ArrayLike, h: float) -> np.ndarray:
    """Tri-cube kernel: w = (1-(d/h)^3)^3 for d < h, else 0."""
    d = np.asarray(d, dtype=float)
    return np.where(d < h, (1.0 - (d / h) ** 3) ** 3, 0.0)


def exponential(d: ArrayLike, h: float) -> np.ndarray:
    """Exponential kernel: w = exp(-d/h)."""
    d = np.asarray(d, dtype=float)
    return np.exp(-d / h)


def boxcar(d: ArrayLike, h: float) -> np.ndarray:
    """Boxcar kernel: w = 1 for d <= h, else 0."""
    d = np.asarray(d, dtype=float)
    return np.where(d <= h, 1.0, 0.0)


def triangular(d: ArrayLike, h: float) -> np.ndarray:
    """Triangular kernel: w = 1 - d/h for d < h, else 0."""
    d = np.asarray(d, dtype=float)
    return np.where(d < h, 1.0 - d / h, 0.0)


KERNEL_FUNCTIONS = {
    "gaussian": gaussian,
    "bisquare": bisquare,
    "tricube": tricube,
    "exponential": exponential,
    "boxcar": boxcar,
    "triangular": triangular,
}


class SpatialKernel:
    """
    Unified spatial kernel interface.

    Parameters
    ----------
    kernel : str
        One of 'gaussian', 'bisquare', 'tricube', 'exponential', 'boxcar', 'triangular'.
    fixed : bool
        If True, bandwidth is a distance. If False, bandwidth is number of neighbours (adaptive).
    bandwidth : float or int
        Distance (fixed) or k nearest neighbours (adaptive).
    """

    def __init__(self, kernel: str = "bisquare", fixed: bool = False,
                 bandwidth: Union[float, int] = None):
        if kernel not in KERNEL_FUNCTIONS:
            raise ValueError(f"Unknown kernel '{kernel}'. Choose from {list(KERNEL_FUNCTIONS)}")
        self.kernel = kernel
        self.fixed = fixed
        self.bandwidth = bandwidth
        self._fn = KERNEL_FUNCTIONS[kernel]

    def __call__(self, distances: np.ndarray) -> np.ndarray:
        """Compute kernel weights from a pre-computed distance array."""
        if self.bandwidth is None:
            raise ValueError("bandwidth must be set before calling the kernel.")
        if self.fixed:
            h = float(self.bandwidth)
        else:
            k = int(self.bandwidth)
            sorted_d = np.sort(distances)
            k = min(k, len(sorted_d) - 1)
            h = sorted_d[k]
            if h == 0:
                h = sorted_d[min(k + 1, len(sorted_d) - 1)]
            if h == 0:
                h = 1.0
        return self._fn(distances, h)

    def weights_for_point(self, focal: np.ndarray, all_coords: np.ndarray) -> np.ndarray:
        """Compute weights from coordinates."""
        dists = np.sqrt(np.sum((all_coords - focal) ** 2, axis=1))
        return self(dists)

    def __repr__(self):
        return (f"SpatialKernel(kernel='{self.kernel}', fixed={self.fixed}, "
                f"bandwidth={self.bandwidth})")


class AnisotropicKernel(SpatialKernel):
    """
    Directional anisotropic kernel with elliptical decay.

    Parameters
    ----------
    angle : float  Rotation angle in degrees.
    ratio : float  Major/minor axis ratio (>= 1).
    """

    def __init__(self, kernel="bisquare", fixed=False, bandwidth=None,
                 angle=0.0, ratio=1.0):
        super().__init__(kernel, fixed, bandwidth)
        self.angle = angle
        self.ratio = ratio

    def weights_for_point(self, focal: np.ndarray, all_coords: np.ndarray) -> np.ndarray:
        theta = np.radians(self.angle)
        diff = all_coords - focal
        dx = diff[:, 0] * np.cos(theta) + diff[:, 1] * np.sin(theta)
        dy = -diff[:, 0] * np.sin(theta) + diff[:, 1] * np.cos(theta)
        dists = np.sqrt(dx ** 2 + (dy * self.ratio) ** 2)
        return self(dists)
