"""
spatialstats.gwmodel.datasets
=============================
Bundled example datasets for tutorials and benchmarking.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def _make_gdf(data: pd.DataFrame, x_col: str = "x", y_col: str = "y"):
    """Convert DataFrame with x/y columns to GeoDataFrame."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        geometry = [Point(xy) for xy in zip(data[x_col], data[y_col])]
        return gpd.GeoDataFrame(data, geometry=geometry)
    except ImportError:
        return data


def load_georgia(as_gdf: bool = True):
    """
    Simulated Georgia (USA) educational attainment dataset.

    Features: PctRural, PctBach, PctEld, PctFB, PctPov, PctBlack
    Target: TotPop90

    Returns (GeoDataFrame or DataFrame, feature_cols, target_col)
    """
    np.random.seed(0)
    n = 159  # Georgia counties
    lon = np.random.uniform(-85.6, -80.8, n)
    lat = np.random.uniform(30.4, 35.0, n)
    pct_rural = np.random.beta(2, 3, n)
    pct_bach = np.random.beta(1.5, 6, n)
    pct_eld = np.random.beta(3, 15, n)
    pct_fb = np.random.beta(1, 20, n)
    pct_pov = np.random.beta(2, 8, n) + 0.1 * (lat - 33)
    pct_black = np.random.beta(1.5, 5, n)
    tot_pop = (
        50000 + 10000 * pct_rural - 8000 * pct_pov +
        5000 * pct_bach + np.random.randn(n) * 5000
    )

    data = pd.DataFrame({
        "x": lon, "y": lat,
        "PctRural": pct_rural, "PctBach": pct_bach,
        "PctEld": pct_eld, "PctFB": pct_fb,
        "PctPov": pct_pov, "PctBlack": pct_black,
        "TotPop90": tot_pop,
    })
    feature_cols = ["PctRural", "PctBach", "PctEld", "PctFB", "PctPov", "PctBlack"]
    target_col = "TotPop90"
    if as_gdf:
        return _make_gdf(data), feature_cols, target_col
    return data, feature_cols, target_col


def load_londonhouse(as_gdf: bool = True):
    """
    Simulated London house prices dataset.

    Features: FloorArea, Bathrooms, Bedrooms, DistCenter, GreenSpace
    Target: Price (£ thousands)

    Returns (GeoDataFrame or DataFrame, feature_cols, target_col)
    """
    np.random.seed(1)
    n = 500
    lon = np.random.uniform(-0.51, 0.33, n)
    lat = np.random.uniform(51.28, 51.69, n)
    floor_area = np.random.gamma(3, 30, n) + 40
    bathrooms = np.random.poisson(1.5, n) + 1
    bedrooms  = np.random.poisson(2, n) + 1
    dist_center = np.sqrt((lon - (-0.09)) ** 2 + (lat - 51.505) ** 2) * 111
    green_space = np.random.beta(2, 5, n)

    # Local price effects (spatial non-stationarity)
    local_floor_coef = 1.5 + 2.0 * np.sin(np.pi * (lon + 0.09) / 0.42)
    price = (
        local_floor_coef * floor_area +
        40 * bathrooms + 25 * bedrooms -
        15 * dist_center + 50 * green_space + 50 +
        np.random.randn(n) * 30
    )

    data = pd.DataFrame({
        "x": lon, "y": lat,
        "FloorArea": floor_area, "Bathrooms": bathrooms,
        "Bedrooms": bedrooms, "DistCenter": dist_center,
        "GreenSpace": green_space, "Price": price,
    })
    feature_cols = ["FloorArea", "Bathrooms", "Bedrooms", "DistCenter", "GreenSpace"]
    target_col = "Price"
    if as_gdf:
        return _make_gdf(data), feature_cols, target_col
    return data, feature_cols, target_col


def load_beijing_pm25(as_gdf: bool = True):
    """
    Simulated Beijing PM2.5 dataset with meteorological covariates.

    Features: AOD, DEM, Wind, Humidity, Temperature
    Target: PM25 (μg/m³)

    Returns (GeoDataFrame or DataFrame, feature_cols, target_col)
    """
    np.random.seed(2)
    n = 300
    lon = np.random.uniform(115.7, 117.4, n)
    lat = np.random.uniform(39.4, 41.1, n)
    aod = np.random.gamma(2, 0.3, n)
    dem = np.random.gamma(2, 200, n)
    wind = np.random.gamma(2, 2, n)
    humidity = np.random.beta(3, 3, n)
    temperature = np.random.normal(15, 8, n)

    # Non-stationary relationship
    local_aod_coef = 30 + 20 * np.sin(np.pi * (lon - 115.7) / 1.7)
    pm25 = (
        local_aod_coef * aod - 5 * wind + 10 * humidity +
        0.01 * dem + np.random.randn(n) * 8 + 20
    )
    pm25 = np.maximum(pm25, 0)

    data = pd.DataFrame({
        "x": lon, "y": lat,
        "AOD": aod, "DEM": dem, "Wind": wind,
        "Humidity": humidity, "Temperature": temperature,
        "PM25": pm25,
    })
    feature_cols = ["AOD", "DEM", "Wind", "Humidity", "Temperature"]
    target_col = "PM25"
    if as_gdf:
        return _make_gdf(data), feature_cols, target_col
    return data, feature_cols, target_col


def load_ncovid(as_gdf: bool = True):
    """
    Simulated US county COVID mortality dataset for GTNNWR demonstration.

    Features: PopDensity, MedianAge, PctUninsured, HospBeds, Mobility
    Target: MortalityRate (per 100k)
    Also includes: time (wave number 1-4)

    Returns (GeoDataFrame or DataFrame, feature_cols, target_col)
    """
    np.random.seed(3)
    n = 400
    lon = np.random.uniform(-124, -70, n)
    lat = np.random.uniform(25, 49, n)
    time = np.random.randint(1, 5, n).astype(float)
    pop_density = np.exp(np.random.normal(5, 1.5, n))
    median_age = np.random.normal(38, 5, n)
    pct_uninsured = np.random.beta(2, 8, n)
    hosp_beds = np.random.gamma(2, 2, n)
    mobility = np.random.normal(0, 0.2, n)

    mortality = (
        0.5 * np.log1p(pop_density) + 2 * median_age / 38 +
        50 * pct_uninsured - 5 * hosp_beds - 20 * mobility +
        5 * time + np.random.randn(n) * 10 + 20
    )
    mortality = np.maximum(mortality, 0)

    data = pd.DataFrame({
        "x": lon, "y": lat, "time": time,
        "PopDensity": pop_density, "MedianAge": median_age,
        "PctUninsured": pct_uninsured, "HospBeds": hosp_beds,
        "Mobility": mobility, "MortalityRate": mortality,
    })
    feature_cols = ["PopDensity", "MedianAge", "PctUninsured", "HospBeds", "Mobility"]
    target_col = "MortalityRate"
    if as_gdf:
        return _make_gdf(data), feature_cols, target_col
    return data, feature_cols, target_col


__all__ = [
    "load_georgia", "load_londonhouse",
    "load_beijing_pm25", "load_ncovid",
]
