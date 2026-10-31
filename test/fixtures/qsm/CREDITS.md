# Credits: the measured trees

Demol, M., Gielen, B. & Verbeeck, H. (2021). QSMs, point cloud and harvest data from a
destructive forest biomass experiment in Belgium. Zenodo. https://doi.org/10.5281/zenodo.4557401.
CC-BY 4.0.

Converted to the model kit skeleton format by `tools/qsm_to_skeleton.py`; no other change.

| File | Species | Source file (`optimal_QSMs.zip`) | Height |
|---|---|---|---|
| `FSYL_15.skeleton.npz` | European beech (*Fagus sylvatica*) | `FSYL15.mat` | 18.0 m |
| `FSYL_13.skeleton.npz` | European beech (*Fagus sylvatica*) | `FSYL13.mat` | 23.8 m |
| `FEXC_5.skeleton.npz` | European ash (*Fraxinus excelsior*) | `FEXC5.mat` | 12.9 m |
| `FEXC_14.skeleton.npz` | European ash (*Fraxinus excelsior*) | `FEXC14.mat` | 22.7 m |

The four files span the set's heights for each species (shortest and tallest of 15 beech and 16
ash). They are leaf-off laser scans, so the skeletons hold no leaves.
