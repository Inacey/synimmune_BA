# %% [markdown]
# # 08_03 — GBmap immune cell annotation via decoupler (TICA marker sets, wsum)
# 
# **Run with the `decoupler_gpu` conda env.**
# 
# Annotates all cells in `immune_cells_extended_gbmap_hgnc.h5ad` independently of GBmap's own
# cell-type labels, using canonical marker gene sets (TICA Supplementary Table 2 / Table 3)
# and decoupler's `wsum` method (via `dc.mt.waggr`).
# 
# 1. Load marker gene nets (Level 1: 25 cell types, Level 2: 32 cell types)
# 2. Load GBmap immune cell data
# 3. Runs in fixed-size cell chunks (CHUNK_SIZE) to avoid the hang observed when running
# `dc.mt.waggr` on the full ~634k-cell object in one call. Each chunk's scores are written
# to disk immediately, so the script is resumable: re-running it skips chunks whose CSV
# already exists.
# 4. Plot UMAP colored by annotation
# 5. Save annotated adata for future sessions

# %%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import decoupler as dc
from datetime import datetime
os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

BASE = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune"
H5AD = f"{BASE}/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad"
NET_LEVEL1_CSV = f"{BASE}/data/02_TICA/TICA_gene_markers_level1_table_s2.csv"
NET_LEVEL2_CSV = f"{BASE}/data/02_TICA/TICA_gene_markers_level2_table_s3.csv"
OUT_DIR = f"{BASE}/results/08_TICAtlas"
CHUNK_DIR = f"{OUT_DIR}/chunks"
ANNOTATED_H5AD_OUT = f"{BASE}/data/01_extGBmap/immune_cells_gbmap_TICAannot.h5ad"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

TMIN = 3  # minimum number of marker genes (present in adata) required per cell type
CHUNK_SIZE = 50_000

checkpoint(f"decoupler version: {dc.__version__}")
checkpoint("Setup complete.")

# %% [markdown]
# ## 1. Load decoupler networks (Level 1 and Level 2 marker sets)

# %%
def load_net(path):
    df = pd.read_csv(path)
    net = df.rename(columns={"cell_type": "source", "gene": "target"})[["source", "target"]].copy()
    net["weight"] = 1.0  # unweighted — all marker genes contribute equally
    return net


net_level1 = load_net(NET_LEVEL1_CSV)
net_level2 = load_net(NET_LEVEL2_CSV)

print(net_level1.head())
print(f"\nLevel 1 — sources: {net_level1['source'].nunique()}, targets: {net_level1['target'].nunique()} unique genes")
print(f"Level 2 — sources: {net_level2['source'].nunique()}, targets: {net_level2['target'].nunique()} unique genes")

# %% [markdown]
# ## 2. Load GBmap immune cell data

# %%
adata = sc.read_h5ad(H5AD)
checkpoint(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")
checkpoint(f"adata.X max value: {adata.X.max():.2f} (if this looks like raw counts, normalize first — decoupler expects log-normalised data)")
# Uncomment if adata.X still holds raw counts:
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)

# ── drop duplicate gene names (decoupler requires unique var_names) — defensive check,
# shouldn't trigger if the Ensembl→HGNC translation already enforced uniqueness
n_dup = adata.var_names.duplicated().sum()
if n_dup > 0:
    checkpoint(f"WARNING: {n_dup} duplicate var_names found — dropping (keeping first occurrence)")
    adata = adata[:, ~adata.var_names.duplicated()].copy()
else:
    checkpoint("No duplicate var_names — OK")

# %% [markdown]
# ## 3. Check marker gene coverage against adata.var_names

# %%
for name, net in [("Level 1", net_level1), ("Level 2", net_level2)]:
    present = net["target"].isin(adata.var_names)
    n_missing = (~present).sum()
    if n_missing > 0:
        missing_genes = sorted(net.loc[~present, "target"].unique())
        checkpoint(f"{name}: {n_missing}/{len(net)} marker rows reference genes NOT in adata.var_names: {missing_genes}")
    else:
        checkpoint(f"{name}: all marker genes found in adata.var_names")

# %% [markdown]
# ## 4. Chunked decoupler wsum (via waggr), resumable 

# %%

def run_chunked_waggr(adata_full, net, level_name, chunk_size):
    """Runs dc.mt.waggr in fixed-size chunks, writing/reading per-chunk CSVs for resume."""
    n_cells = adata_full.n_obs
    n_chunks = (n_cells + chunk_size - 1) // chunk_size
    checkpoint(f"[{level_name}] Total cells: {n_cells}, chunk size: {chunk_size}, n_chunks: {n_chunks}")

    for i in range(n_chunks):
        chunk_csv = f"{CHUNK_DIR}/{level_name}_chunk_{i:03d}.csv"
        if os.path.exists(chunk_csv):
            checkpoint(f"[{level_name}] Chunk {i+1}/{n_chunks} already done — skipping ({chunk_csv})")
            continue

        start = i * chunk_size
        end = min(start + chunk_size, n_cells)
        checkpoint(f"[{level_name}] Chunk {i+1}/{n_chunks}: cells {start}:{end}")

        chunk = adata_full[start:end].copy()
        dc.mt.waggr(data=chunk, net=net, fun="wsum", tmin=TMIN)

        acts = dc.pp.get_obsm(chunk, key="score_waggr")
        scores_df = acts.to_df()
        scores_df.index = chunk.obs_names
        scores_df.to_csv(chunk_csv)

        checkpoint(f"[{level_name}] Chunk {i+1}/{n_chunks} done -> {chunk_csv}")
        del chunk

    # reassemble all chunks for this level
    chunk_files = sorted(glob.glob(f"{CHUNK_DIR}/{level_name}_chunk_*.csv"))
    scores_full = pd.concat([pd.read_csv(f, index_col=0) for f in chunk_files])
    scores_full = scores_full.reindex(adata_full.obs_names)
    checkpoint(f"[{level_name}] Reassembled scores: {scores_full.shape}")
    return scores_full


checkpoint("Running decoupler wsum (waggr) — Level 1 (25 cell types), chunked")
scores_l1_df = run_chunked_waggr(adata, net_level1, "level1", CHUNK_SIZE)

adata.obs["decoupler_level1_celltype"] = scores_l1_df.idxmax(axis=1).values
adata.obs["decoupler_level1_score"] = scores_l1_df.max(axis=1).values

checkpoint("Level 1 dominant cell type distribution:")
print(adata.obs["decoupler_level1_celltype"].value_counts().to_string())


checkpoint("Running decoupler wsum (waggr) — Level 2 (32 cell types), chunked")
scores_l2_df = run_chunked_waggr(adata, net_level2, "level2", CHUNK_SIZE)

adata.obs["decoupler_level2_celltype"] = scores_l2_df.idxmax(axis=1).values
adata.obs["decoupler_level2_score"] = scores_l2_df.max(axis=1).values

checkpoint("Level 2 dominant cell type distribution:")
print(adata.obs["decoupler_level2_celltype"].value_counts().to_string())


# %% [markdown]
# ## 5. Save annotation table (CSV)

# %%
annot_df = adata.obs[[
    "decoupler_level1_celltype", "decoupler_level1_score",
    "decoupler_level2_celltype", "decoupler_level2_score",
]].copy()
annot_out = os.path.join(OUT_DIR, "08_03_decoupler_celltype_annotation.csv")
annot_df.to_csv(annot_out)
checkpoint(f"Annotation table saved to {annot_out}")

# %% [markdown]
# ## 6. UMAP colored by decoupler annotation (Level 1)

# %%
if "X_umap" in adata.obsm:
    fig, ax = plt.subplots(figsize=(9, 7))
    sc.pl.umap(adata, color="decoupler_level1_celltype", ax=ax, show=False, size=3, legend_fontsize=7)
    ax.set_rasterized(True)
    plt.tight_layout()
    out_umap = os.path.join(OUT_DIR, "08_03_decoupler_level1_umap.pdf")
    plt.savefig(out_umap, bbox_inches="tight", dpi=300)
    plt.show()
    checkpoint(f"UMAP saved: {out_umap}")
else:
    checkpoint("No X_umap found in adata — skipping UMAP plot")

# %% [markdown]
# ## 7. Save annotated adata for future sessions

# %%
adata.write(ANNOTATED_H5AD_OUT)
checkpoint(f"Annotated adata saved to {ANNOTATED_H5AD_OUT}")
checkpoint("ALL DONE!")


