import kaleidocell
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
from datetime import datetime

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad"
RESULTS_MP_PATH = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_nmf_all_gsea/results_mp.kc"
GSEA_CSV = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_nmf_all_gsea/gsea_C7.csv"
GMT_PATH = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/c7.all.v2026.1.Hs.symbols.gmt"
OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/05_significant_MP/05_all_immune_cells"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1 — Load data and results ─────────────────────────────────────────
checkpoint("Loading data")
adata = sc.read_h5ad(DATA_PATH)
checkpoint(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")

checkpoint("Loading results_mp")
results_mp = kaleidocell.load(RESULTS_MP_PATH)
checkpoint(f"results_mp loaded: {len(results_mp['mp_dict'])} MPs")

# Step 2 — Load significant terms────────────────────────────────────────────
gsea_df = pd.read_csv(GSEA_CSV)
checkpoint(f"Total significant terms: {len(gsea_df)}")

# Take top 10 terms per MP
top_per_mp = gsea_df.groupby("MP").head(10)
top_per_mp.to_csv(os.path.join(OUT_DIR, "top_gsea_per_mp.csv"), index=False)
checkpoint("Top GSEA terms saved")

# ── Step 3 — Parse GMT file ─────────────────────────────────────────────────
checkpoint("Parsing GMT file")
def parse_gmt(gmt_path):
    gene_sets = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            gene_sets[parts[0]] = parts[2:]
    return gene_sets

gene_sets = parse_gmt(GMT_PATH)
checkpoint(f"GMT loaded: {len(gene_sets)} gene sets")

# ── Step 4 — Score cells for top C7 signatures ─────────────────────────────
checkpoint("Scoring cells for top C7 signatures")
top_terms = top_per_mp["Term"].unique().tolist()


def sanitize(name):
    h = hashlib.md5(name.encode()).hexdigest()[:6]
    return name.replace(" ", "_").replace("/", "_")[:40] + "_" + h

term_name_map = {}

for term in top_terms:
    if term in gene_sets:
        genes = [g for g in gene_sets[term] if g in adata.var_names]
        if len(genes) >= 5:
            short_name = sanitize(term)
            term_name_map[short_name] = term
            try:
                sc.tl.score_genes(adata, gene_list=genes, score_name=short_name, use_raw=False)
                checkpoint(f"Scored: {term} ({len(genes)} genes)")
            except ValueError as e:
                checkpoint(f"SKIPPED {term}: {e}")

checkpoint("Cell scoring done")

# ── Step 5 — UMAP colored by top scores ────────────────────────────────────
checkpoint("Plotting UMAPs")
if "X_umap" not in adata.obsm:
    checkpoint("Computing neighbors and UMAP")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

for short_name in term_name_map.keys()[:5]:  # plot top 5
    if short_name in adata.obs.columns:
        sc.pl.umap(adata, color=short_name, show=False)
        plt.savefig(os.path.join(OUT_DIR, f"umap_{short_name[:50]}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        checkpoint(f"UMAP saved: {short_name}")



checkpoint("ALL DONE!")