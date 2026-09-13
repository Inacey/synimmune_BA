import kaleidocell
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
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad")
checkpoint(f"Data loaded: {adata.n_obs} cells")

checkpoint("Loading mp_scores")
mp_scores = pd.read_csv("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_3_nmf_all_gsea_c7/mp_scores.csv", index_col=0)
checkpoint(f"mp_scores loaded: {mp_scores.shape}")

missing_cells = set(adata.obs_names) - set(mp_scores.index)
if missing_cells:
    raise ValueError(f"{len(missing_cells)} cells in adata not found in mp_scores index")

mp_scores = mp_scores.loc[adata.obs_names]
all_score_cols = [c for c in mp_scores.columns if c.endswith("_score")]
mp_scores_renamed = mp_scores[all_score_cols].copy()
mp_scores_renamed.columns = [c.replace("_score", "") for c in all_score_cols]

checkpoint("Computing dominant MP within SELECTED_MPS only")
scores_selected = mp_scores_renamed[SELECTED_MPS]

dominant_MP_all = scores_selected.idxmax(axis=1)
max_scores_all = scores_selected.max(axis=1)

threshold = np.percentile(max_scores_all, 55)
checkpoint(f"Using adaptive threshold (55th percentile): {threshold:.4f}")

dominant_MP_all = np.where(max_scores_all.values > threshold, dominant_MP_all.values, "Other")
adata.obs["dominant_MP_all"] = dominant_MP_all

checkpoint(f"Distribution:\n{adata.obs['dominant_MP_all'].value_counts()}")

if "X_umap" not in adata.obsm:
    checkpoint("Computing UMAP")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg)
    sc.tl.pca(adata_hvg)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.umap(adata)
    checkpoint("UMAP done")
else:
    checkpoint("UMAP already exists, skipping")

HIGHLIGHT_COLOR = "#ed1a72"
OTHER_COLOR = "#d3d3d3"


checkpoint("Plotting 11 individual dominant-MP UMAPs")
n_cols = 3
n_rows = -(-len(SELECTED_MPS) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()

for i, mp in enumerate(SELECTED_MPS):
    col_name = f"is_{mp}"
    values = np.where(adata.obs["dominant_MP_all"] == mp, mp, "Other")
    adata.obs[col_name] = pd.Categorical(values, categories=["Other", mp])

    sc.pl.umap(
        adata,
        color=col_name,
        palette={mp: HIGHLIGHT_COLOR, "Other": OTHER_COLOR},
        show=False,
        ax=axes[i],
        title=mp,
        size=3,
        legend_loc="none"
    )
    axes[i].set_rasterized(True)
    checkpoint(f"Plotted {mp}")

for j in range(len(SELECTED_MPS), len(axes)):
    axes[j].set_visible(False)


plt.tight_layout()
plt.title("Dominant MP (across selected MPs) per cell", fontsize=16)
plt.savefig(os.path.join(OUT_DIR, "05_15_dominant_mp_selected_umap_55.pdf"), bbox_inches="tight", dpi=300)
plt.close()

checkpoint("PDF saved")
checkpoint("ALL DONE!")