import scanpy as sc
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    
OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/05_significant_MP/05_all_immune_cells"
os.makedirs(OUT_DIR, exist_ok=True)

adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad")
mp_scores = pd.read_csv("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_nmf_all_gsea/mp_scores.csv", index_col=0)



# Compute mean MP score per cell type
mp_cols = mp_scores.columns.tolist()
adata_temp = adata.copy()
adata_temp.obs[mp_cols] = mp_scores.values

# Mean score per cell type per MP
mean_scores = adata_temp.obs.groupby("annotation_level_3")[mp_cols].mean()

# Plot heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(
    mean_scores.T,
    cmap="viridis",
    xticklabels=True,
    yticklabels=True,
    linewidths=0.5,
    cbar_kws={"label": "Mean MP score"}
)
plt.title("Mean MP score per cell type")
plt.xlabel("Cell type")
plt.ylabel("Meta-program")
plt.tight_layout()
plt.savefig("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/05_significant_MP/05_all_immune_cells/heatmap_MP_vs_celltype.pdf", bbox_inches="tight")
plt.close()
checkpoint("Heatmap saved")