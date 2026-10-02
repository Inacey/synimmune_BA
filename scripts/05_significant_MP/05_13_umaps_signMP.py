import kaleidocell
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/05_significant_MP/05_all_immune_cells"
os.makedirs(OUT_DIR, exist_ok=True)

SELECTED_MPS = ["MP1", "MP2", "MP7", "MP8", "MP9", "MP11", "MP14", "MP15", "MP19", "MP20"]

checkpoint("Loading data")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad")
checkpoint(f"Data loaded: {adata.n_obs} cells")

checkpoint("Loading mp_scores")
mp_scores = pd.read_csv("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_3_nmf_all_gsea_c7/mp_scores.csv", index_col=0)
checkpoint(f"mp_scores loaded: {mp_scores.shape}")

# Add selected MP scores to adata.obs
checkpoint("Adding MP scores to adata")
for mp in SELECTED_MPS:
    col_name = f"{mp}_score"
    if col_name in mp_scores.columns:
        adata.obs[col_name] = mp_scores.loc[adata.obs_names, col_name].values
    else:
        checkpoint(f"WARNING: {col_name} not found in mp_scores")

# Compute UMAP if not present
if "X_umap" not in adata.obsm:
    checkpoint("Computing UMAP")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    checkpoint("UMAP done")
else:
    checkpoint("UMAP already exists, skipping")

# Plot all 11 MPs in one PDF
checkpoint("Plotting UMAPs")
n_cols = 5
n_rows = -(-len(SELECTED_MPS) // n_cols)  # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()

for i, mp in enumerate(SELECTED_MPS):
    col_name = f"{mp}_score"
    if col_name in adata.obs.columns:
        sc.pl.umap(adata, color=col_name, color_map="inferno", show=False, ax=axes[i], title=mp)
        axes[i].set_rasterized(True)
        checkpoint(f"Plotted {mp}")
    else:
        checkpoint(f"SKIPPED {mp} (not in adata.obs)")

# Hide empty subplots
for j in range(len(SELECTED_MPS), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_13_umaps_signMP_4.pdf"), bbox_inches="tight", dpi=200)
plt.close()
checkpoint("PDF saved")

checkpoint("ALL DONE!")