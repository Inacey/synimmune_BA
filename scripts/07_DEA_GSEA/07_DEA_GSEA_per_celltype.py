import scanpy as sc
import pandas as pd
import gseapy
import os
import numpy as np
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

SUBSET_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/hgnc_immune_subsets/"
GMT_PATHS = {
    "hallmark": "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/h.all.v2026.1.Hs.symbols.gmt",
    "reactome": "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/c2.cp.reactome.v2026.1.Hs.symbols.gmt"
}
OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/07_DEA_GSEA/"
os.makedirs(OUT_DIR, exist_ok=True)

subset_files = [f for f in os.listdir(SUBSET_DIR) if f.endswith(".h5ad")]
checkpoint(f"Found {len(subset_files)} subset files")

for filename in subset_files:
    cell_type = filename.replace("hgnc_", "").replace(".h5ad", "")
    checkpoint(f"Processing {cell_type}")

    subset = sc.read_h5ad(os.path.join(SUBSET_DIR, filename))
    checkpoint(f"{cell_type}: {subset.n_obs} cells")

    # ── Preprocessing ──────────────────────────────────────────────────────
    sc.pp.normalize_total(subset, target_sum=1e4)
    sc.pp.log1p(subset)
    subset.raw = subset

    sc.pp.highly_variable_genes(subset, n_top_genes=2000)
    sc.pp.scale(subset)
    sc.tl.pca(subset)
    sc.pp.neighbors(subset)
    sc.tl.leiden(subset, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
    checkpoint(f"{cell_type}: {subset.obs['leiden'].nunique()} clusters found")

    # ── DEA — 1 vs all ─────────────────────────────────────────────────────
    checkpoint(f"{cell_type}: Running DEA (1 vs all, Wilcoxon)")
    sc.tl.rank_genes_groups(subset, groupby="leiden", method="wilcoxon", key_added="rank_genes", use_raw=True)
    checkpoint(f"{cell_type}: DEA done")

    # ── GSEA per cluster ───────────────────────────────────────────────────
    safe_name = cell_type.replace("/", "_").replace(" ", "_")
    gsea_out = os.path.join(OUT_DIR, safe_name)
    os.makedirs(gsea_out, exist_ok=True)

    clusters = subset.obs["leiden"].unique().tolist()


    for cluster in clusters:
        checkpoint(f"{cell_type} — cluster {cluster}: Running GSEA")

        # Get ranked genes with Wilcoxon scores for this cluster
        genes = sc.get.rank_genes_groups_df(subset, group=cluster, key="rank_genes")
        genes = genes.dropna(subset=["pvals", "logfoldchanges"])
        genes["rank_metric"] = -np.log10(genes["pvals"].clip(lower=1e-300)) * np.sign(genes["logfoldchanges"])
        genes = genes.sort_values("rank_metric", ascending=False)
        ranked = genes.set_index("names")["rank_metric"]

        for gmt_name, gmt_path in GMT_PATHS.items():
            try:
                res = gseapy.prerank(
                    rnk=ranked,
                    gene_sets=gmt_path,
                    outdir=None,
                    min_size=10,
                    max_size=500,
                    seed=42,
                    verbose=False
                )

                # Save significant results
                results_df = res.res2d[res.res2d["FDR q-val"] < 0.05].sort_values("FDR q-val")
                results_df.to_csv(os.path.join(gsea_out, f"gsea_{gmt_name}_cluster{cluster}.csv"), index=False)
                checkpoint(f"{cell_type} — cluster {cluster} ({gmt_name}): {len(results_df)} significant pathways")

            except Exception as e:
                checkpoint(f"{cell_type} — cluster {cluster} ({gmt_name}): GSEA failed — {e}")

    checkpoint(f"{cell_type}: DONE")

checkpoint("ALL DONE!")