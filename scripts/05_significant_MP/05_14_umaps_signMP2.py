import scanpy as sc
import pandas as pd
import numpy as np
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
adata = sc.read_h5ad(
    "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad"
)
checkpoint(f"Data loaded: {adata.n_obs} cells")

if "X_umap" not in adata.obsm:
    raise ValueError("X_umap not found in adata.obsm — run/load UMAP first (e.g. from 05_14 script)")

checkpoint("Loading mp_scores")
mp_scores = pd.read_csv(
    "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_3_nmf_all_gsea_c7/mp_scores.csv",
    index_col=0,
)
mp_scores = mp_scores.loc[adata.obs_names]

# add each selected MP's score into adata.obs so sc.pl.umap can color by it directly
for mp in SELECTED_MPS:
    col = f"{mp}_score"
    if col not in mp_scores.columns:
        raise ValueError(f"{col} not found in mp_scores columns")
    adata.obs[mp] = mp_scores[col].values

checkpoint("Plotting per-MP UMAP grid (continuous scores)")
n_mps = len(SELECTED_MPS)
n_cols = 5
n_rows = int(np.ceil(n_mps / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = np.atleast_1d(axes).flatten()

for i, mp in enumerate(SELECTED_MPS):
    sc.pl.umap(
        adata,
        color=mp,
        ax=axes[i],
        show=False,
        title=mp,
        size=2.5,
        cmap="inferno",
        colorbar_loc=None,  # keep panels clean
    )
    axes[i].set_rasterized(True)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# hide any unused axes if n_mps doesn't fill the grid evenly
for j in range(n_mps, len(axes)):
    axes[j].axis("off")

scatter_collection = axes[0].collections[0]  # get the scatter collection from the first subplot

fig.subplots_adjust(right=0.88, wspace=0.15, hspace=0.25)

cax = fig.add_axes([0.9, 0.15, 0.015, 0.7])  # [left, bottom, width, height] in figure coordinates
cbar = fig.colorbar(scatter_collection, cax=cax, orientation="vertical")

out_path = os.path.join(OUT_DIR, "05_14_per_mp_umap_grid_inferno.pdf")
plt.savefig(out_path, bbox_inches="tight", dpi=300)
plt.close()
checkpoint(f"Grid PDF saved to {out_path}")

# ── Optional: also save each MP as its own individual PDF, useful for slides
# INDIV_DIR = os.path.join(OUT_DIR, "per_mp_umaps")
# os.makedirs(INDIV_DIR, exist_ok=True)
# checkpoint(f"Saving individual per-MP UMAPs to {INDIV_DIR}")

# for mp in SELECTED_MPS:
#     fig, ax = plt.subplots(figsize=(6, 5))
#     sc.pl.umap(
#         adata,
#         color=mp,
#         ax=ax,
#         show=False,
#         title=mp,
#         size=3,
#         cmap="viridis",
#     )
#     ax.set_rasterized(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(INDIV_DIR, f"05_16_{mp}_umap.pdf"), bbox_inches="tight", dpi=150)
#     plt.close()

checkpoint("ALL DONE!")