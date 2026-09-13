import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/01_NMF_all_immune_cells"
os.makedirs(OUT_DIR, exist_ok=True)

checkpoint("Loading full GBMap")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/data/extendedGBmap.h5ad")
checkpoint(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
print(adata.obs.columns.tolist())
print(adata.obs["annotation_level_2"].value_counts())

# Label immune vs non-immune
checkpoint("Labeling immune cells")
immune_types = ["Myeloid", "Lymphoid"]
adata.obs["cell_focus"] = np.where(
    adata.obs["annotation_level_2"].isin(immune_types),
    adata.obs["annotation_level_2"],
    "Other"
)
adata.obs["cell_focus"] = adata.obs["cell_focus"].astype("category")

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

# Plot — gray for non-immune, colors for immune
checkpoint("Plotting UMAP")
palette = {
    "Myeloid": "#E64B35",
    "Lymphoid": "#4DBBD5",
    "Other": "#D9C4A8"
}

sc.pl.umap(
    adata,
    color="cell_focus",
    show=False,
    palette=palette,
    title="Extended GBMap — Immune cells highlighted",
    legend_loc="right margin",  # or "on data" to show labels on the UMAP itself
    legend_fontsize=12
)
plt.gca().set_rasterized(True)
plt.savefig(os.path.join(OUT_DIR, "01_umap_gbmap_immune_highlighted.pdf"), bbox_inches="tight", dpi=300)
plt.close()
checkpoint("UMAP saved")

checkpoint("ALL DONE!")