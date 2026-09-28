# Example Notebooks

Jupyter notebooks demonstrating PyGWmodel usage.

| Notebook | Description |
|---|---|
| `getting_started.ipynb` | Hands-on tutorial (from TUTORIAL.md) using `data/` datasets; HTML render in `docs/getting_started.html` |
| `00_geographically_weighted_models_introductions.ipynb` | Introduction to geographically weighted models |
| `ch00_setup_and_foundations.ipynb` | **Spatial statistics, Chapter 0** (full single-file chapter) — installation, package map, spatial data types, `SpatialData`, CRS, distances, datasets, first maps; theory with mathematics, exercises with solutions. Executed HTML render: `docs/notebooks/ch00_setup_and_foundations.html` |
| `ch00_1_installation_and_package_map.ipynb` | **Chapter 0.1** — install, verify extras, optional-dependency pattern, package map |
| `ch00_2_spatial_data_types_and_containers.ipynb` | **Chapter 0.2** — fields / lattices / point patterns, `SpatialData` roles |
| `ch00_3_crs_and_distances.ipynb` | **Chapter 0.3** — CRS theory and audits, Buffalo case study, distance metrics |
| `ch00_4_datasets_maps_and_workflow.ipynb` | **Chapter 0.4** — data catalogue, choropleths, loading recipe, pitfalls, exercises |

Run notebooks from the repository root after installing the package:

```bash
pip install -e ".[all]"
jupyter notebook examples/notebooks/
```
