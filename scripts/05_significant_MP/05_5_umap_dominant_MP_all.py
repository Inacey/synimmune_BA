import kaleidocell
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

checkpoint("Loading data")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad")
checkpoint(f"Data loaded: {adata.n_obs} cells")

checkpoint("Loading results_mp")
results_mp = kaleidocell.load("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_nmf_all_gsea/results_mp.kc")

checkpoint("Loading mp_scores")
mp_scores = pd.read_csv("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_nmf_all_gsea/mp_scores.csv", index_col=0)
checkpoint(f"mp_scores loaded: {mp_scores.shape[1]} MPs")

#checkpoint("Computing MP scores")
#mp_scores = kaleidocell.compute_mp_scores(results_mp, adata)

# Save mp_scores for future use
#mp_scores.to_csv("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_nmf_all_gsea/mp_scores.csv")
#checkpoint("mp_scores saved to CSV")

# ── Select MPs with silhouette > 0.15 ───────────────────────────────────────
checkpoint("Selecting MPs with silhouette > 0.15")
metrics = results_mp["metrics"]
selected_mps = metrics[metrics["silhouette"] > 0.15].index.tolist()
checkpoint(f"Selected {len(selected_mps)} MPs: {selected_mps}")

selected_mp_cols = [f"{mp}_score" for mp in selected_mps]
scores_subset = mp_scores[selected_mp_cols]

checkpoint(f"Subset shape: {scores_subset.shape}")

# ── Inspect score distributions ─────────────────────────────────────────────
checkpoint("Inspecting score distributions")
print(scores_subset.describe())

scores_subset.hist(bins=50, figsize=(15, 10))
plt.savefig("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/05_significant_MP/05_all_immune_cells/mp_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
checkpoint("Score distribution plot saved")

# ── Assign dominant MP per cell ─────────────────────────────────────────────
checkpoint("Assigning dominant MP per cell")
adata.obs["dominant_MP"] = scores_subset.idxmax(axis=1).values
max_scores = scores_subset.max(axis=1).values

threshold = np.percentile(max_scores, 75)
checkpoint(f"Using adaptive threshold (75th percentile): {threshold:.4f}")

adata.obs["dominant_MP"] = np.where(max_scores > threshold, adata.obs["dominant_MP"], "Other")
adata.obs["dominant_MP"] = adata.obs["dominant_MP"].astype("category")

checkpoint(f"Distribution:\n{adata.obs['dominant_MP'].value_counts()}")

if "X_umap" not in adata.obsm:
    checkpoint("Computing UMAP")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    checkpoint("UMAP done")

checkpoint("Plotting single UMAP colored by dominant MP")
sc.pl.umap(adata, color="dominant_MP", show=False, palette="tab20")
plt.savefig("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/05_significant_MP/05_all_immune_cells/umap_dominant_MP_all.png", dpi=150, bbox_inches="tight")
plt.close()
checkpoint("Saved")

checkpoint("ALL DONE!")

