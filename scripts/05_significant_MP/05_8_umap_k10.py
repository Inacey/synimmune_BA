import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/05_significant_MP/05_all_immune_cells/"
os.makedirs(OUT_DIR, exist_ok=True)

checkpoint("Loading data")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad")
checkpoint(f"Data loaded: {adata.n_obs} cells")

checkpoint("Loading mp_scores")
mp_scores_k10 = pd.read_csv("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_5_nmf_k10/mp_scores_k10.csv", index_col=0)
checkpoint(f"mp_scores loaded: {mp_scores_k10.shape}")

# Use all MPs
selected_mps = mp_scores_k10.columns.tolist()
checkpoint(f"Using all {len(selected_mps)} MPs: {selected_mps}")

scores_subset = mp_scores_k10.loc[adata.obs_names, selected_mps]
assert scores_subset.shape[0] == adata.n_obs, "Zellzahl-Mismatch zwischen adata und mp_scores!"

adata.obs["dominant_MP"] = scores_subset.idxmax(axis=1).values

# Assign dominant MP per cell
checkpoint("Assigning dominant MP per cell")
adata.obs["dominant_MP"] = scores_subset.idxmax(axis=1).values

adata.obs["dominant_MP"] = adata.obs["dominant_MP"].astype("category")

checkpoint(f"Distribution:\n{adata.obs['dominant_MP'].value_counts()}")

if "X_umap" not in adata.obsm:
    checkpoint("Computing UMAP")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    checkpoint("UMAP done")
else:
    checkpoint("UMAP already exists, skipping")

checkpoint("Plotting UMAP colored by 16 MP")
# Generate 35 distinct colors
def get_distinct_colors(n):
    colors = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.colormaps[cmap_name]
        colors += [cmap(i) for i in np.linspace(0, 1, 20)]
    return colors[:n]

all_cats = adata.obs["dominant_MP"].cat.categories.tolist()
colors = get_distinct_colors(len(all_cats))
palette = {cat: colors[i] for i, cat in enumerate(all_cats)}

sc.pl.umap(adata, color="dominant_MP", show=False, palette=palette)
plt.savefig(os.path.join(OUT_DIR, "umap_MP_k10.pdf"), bbox_inches="tight")
plt.close()

checkpoint("UMAP saved")