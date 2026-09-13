import scanpy as sc
import matplotlib.pyplot as plt
import os
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/06_leiden/"
os.makedirs(OUT_DIR, exist_ok=True)

checkpoint("Loading data")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad")
checkpoint(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")

checkpoint("Selecting highly variable genes")
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

checkpoint("Scaling data")
sc.pp.scale(adata)

checkpoint("Running PCA")
sc.tl.pca(adata)

checkpoint("Computing neighbors")
sc.pp.neighbors(adata)
checkpoint("Neighbors done")

if "X_umap" not in adata.obsm:
    checkpoint("Computing UMAP")
    sc.tl.umap(adata)
    checkpoint("UMAP done")
else:
    checkpoint("UMAP already exists, skipping")

checkpoint("Running Leiden clustering (resolution=0.5)")
sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
checkpoint(f"Clusters found: {adata.obs['leiden'].nunique()}")
print(adata.obs["leiden"].value_counts())

checkpoint("Plotting UMAP")
sc.pl.umap(adata, color=["leiden", "annotation_level_2", "annotation_level_3"], show=False)
plt.savefig(os.path.join(OUT_DIR, "umap_leiden_res05_all.png"), dpi=150, bbox_inches="tight")
plt.close()
checkpoint(f"UMAP saved to {OUT_DIR}")

checkpoint("ALL DONE!")