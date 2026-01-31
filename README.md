# Marine Benthic Cover Analysis Pipeline (SOM + Boundary Metrics)

This repository contains the full analysis pipeline used to characterize benthic community organization in Taiwan using a Manhattan-distance Self-Organizing Map (SOM), k-means clustering of SOM prototypes, and boundary strength metrics (MSS/RMSS). A PCA + k-means baseline is included for complementary comparison.

---

## 1) Overview

**Inputs**
- Transect-level benthic relative cover (major components + OTU/species cover)
- Sample metadata (site / transect / depth / coordinates)
- Environmental drivers to be added for future integration

**Core Outputs**
- Manhattan SOM (trained on standardized benthic cover)
- SOM-derived community states (k-means on node prototypes)
- U-matrix and cluster maps (SOM space)
- Boundary adjacency frequencies and **boundary strength** (MSS, RMSS)
- Spatial maps (study area + insets) of community states and major components
- PCA + k-means baseline visualizations (for comparison)

---

## 2) Repository Structure (recommended)

```text
.
├── data/
│   ├── raw/                 # raw input files (never edited)
│   ├── processed/           # cleaned + aggregated datasets
│   └── external/            # optional: external layers (shapefiles, env drivers)
├── configs/
│   └── config.yaml          # paths, variables, SOM/grid params, random seeds
src/
├── train_som.py              # (from minisom_base.py)
├── analyze_som.py            # (from analysis_minisom_base_v2.py)
├── som_plots.py              # U-matrix, hit maps, overlays
├── som_clustering.py         # kmeans on node weights + assign BMU clusters
├── som_alignment.py          # purity/entropy/disagreement + ARI/NMI (your analyze_alignment)
├── env_gradients.py          # environmental_gradient_analysis
├── spatial_plots.py          # Taiwan maps + insets + fallback scatter
└── clustering_evaluation.py  # keep as its own benchmark script
├── notebooks/               # optional: exploration / figure prototyping
├── results/
│   ├── models/              # saved SOM, scalers, labels
│   ├── figures/             # all exported figures
│   └── tables/              # MSS/RMSS tables, cluster summaries
├── environment.yml          # conda env
├── requirements.txt         # pip alternative
└── README.md
