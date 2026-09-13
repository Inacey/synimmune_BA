import scanpy as sc
import matplotlib.pyplot as plt
import os
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/06_leiden/per_celltype/"
os.makedirs(OUT_DIR, exist_ok=True)

SUBSET_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/hgnc_immune_subsets/"

# List all subset files
subset_files = [f for f in os.listdir(SUBSET_DIR) if f.endswith(".h5ad")]
checkpoint(f"Found {len(subset_files)} subset files: {subset_files}")

for filename in subset_files:
    cell_type = filename.replace("hgnc_", "").replace(".h5ad", "")
    checkpoint(f"Processing {cell_type}")
    
    subset = sc.read_h5ad(os.path.join(SUBSET_DIR, filename))
    checkpoint(f"{cell_type}: {subset.n_obs} cells")
    
    # Preprocessing
    sc.pp.highly_variable_genes(subset, n_top_genes=2000)
    sc.pp.scale(subset)
    sc.tl.pca(subset)
    sc.pp.neighbors(subset)
    sc.tl.umap(subset)
    
    # Leiden clustering
    sc.tl.leiden(subset, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
    checkpoint(f"{cell_type}: {subset.obs['leiden'].nunique()} clusters found")
    print(subset.obs["leiden"].value_counts())
    
    # Plot
    sc.pl.umap(subset, color=["leiden", "stage", "location"], show=False)
    safe_name = cell_type.replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(OUT_DIR, f"umap_leiden_{safe_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    checkpoint(f"{cell_type}: UMAP saved")

checkpoint("ALL DONE!")