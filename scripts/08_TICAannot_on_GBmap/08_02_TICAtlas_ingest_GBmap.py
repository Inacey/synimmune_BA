import scanpy as sc
import matplotlib.pyplot as plt
import os
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/08_TICAtlas/"
os.makedirs(OUT_DIR, exist_ok=True)

# Load TICAtlas reference
checkpoint("Loading TICAtlas reference")
tica = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/TICAtlas.h5ad")
tica.var_names = tica.var["features"].values
checkpoint(f"TICAtlas loaded: {tica.n_obs} cells, {tica.n_vars} genes")

# Load GBMap immune cells
checkpoint("Loading GBMap immune cells")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad")
checkpoint(f"GBMap loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# Subset to common genes
checkpoint("Subsetting to common genes")
common_genes = list(set(tica.var_names) & set(adata.var_names))
tica_common = tica[:, common_genes].copy()
adata_common = adata[:, common_genes].copy()
checkpoint(f"Common genes: {len(common_genes)}")

# Preprocess reference
checkpoint("Preprocessing TICAtlas reference")
sc.pp.normalize_total(tica_common, target_sum=1e4)
sc.pp.log1p(tica_common)
sc.pp.highly_variable_genes(tica_common, n_top_genes=2000)
sc.pp.scale(tica_common)
sc.tl.pca(tica_common)
sc.pp.neighbors(tica_common)
sc.tl.umap(tica_common) 
checkpoint("TICAtlas preprocessing done")

# Preprocess query
checkpoint("Preprocessing GBMap query")
sc.pp.normalize_total(adata_common, target_sum=1e4)
sc.pp.log1p(adata_common)
checkpoint("GBMap preprocessing done")

# Ingest GBMap cells onto TICAtlas UMAP
checkpoint("Running sc.tl.ingest")
sc.tl.ingest(adata_common, tica_common, obs="lv1_annot")
checkpoint("Ingest done")

# Plot combined UMAP
checkpoint("Plotting combined UMAP")
tica_common.obs["source"] = "TICAtlas"
adata_common.obs["source"] = "GBMap"

import anndata
combined = anndata.concat([tica_common, adata_common])

fig = sc.pl.umap(combined, color=["source", "lv1_annot"], show=False, return_fig=True)

for ax in fig.axes:
    for artist in ax.collections:
        artist.set_rasterized(True)

fig.savefig(os.path.join(OUT_DIR, "umap_TICAtlas_vs_GBMap.pdf"), bbox_inches="tight")
plt.close(fig)
checkpoint("UMAP saved")

checkpoint("ALL DONE!")