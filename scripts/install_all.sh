#!/usr/bin/env bash
# Install pyspatialstats with all optional dependencies.
# Usage: bash scripts/install_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install --upgrade pip setuptools wheel
if [[ -f dist/pyspatialstats-0.2.0-py3-none-any.whl ]]; then
  python -m pip install "dist/pyspatialstats-0.2.0-py3-none-any.whl[all]"
else
  python -m pip install -e ".[all]"
fi

python - <<'PY'
import importlib
mods = [
    "spatialstats", "numpy", "scipy", "pandas", "geopandas", "shapely",
    "sklearn", "matplotlib", "folium", "mapclassify", "libpysal", "networkx",
    "torch", "xgboost", "lightgbm", "shap",
]
print("Verification:")
for m in mods:
    name = "sklearn" if m == "sklearn" else m
    try:
        mod = importlib.import_module(name)
        print(f"  OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  MISS {m}: {e}")
PY
