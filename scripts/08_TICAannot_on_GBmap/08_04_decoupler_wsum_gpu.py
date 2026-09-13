"""
08_04 — GBmap immune cell annotation via decoupler-GPU (rapids-singlecell, TICA marker sets, wsum)

GPU-accelerated re-implementation of 08_03_decoupler_wsum_chunked.py. Uses
`rapids_singlecell.dcg.wsum` (decoupler-GPU) instead of `decoupler.mt.waggr` on CPU.
Same TICA marker nets, same tmin/output structure — only the compute backend changes.

Run in the `decoupler_gpu` conda env (rapids-singlecell + decoupler installed), inside tmux:
    tmux new -s decoupler_gpu_run
    export CUDA_VISIBLE_DEVICES=0   # GPU 0 was idle in nvidia-smi
    /path/to/decoupler_gpu/bin/python 08_04_decoupler_wsum_gpu.py 2>&1 | tee run_log_gpu.txt

No chunking should be necessary given A30 VRAM (24GB) vs. dataset size, but a chunked
fallback loop is included (commented) in case of an out-of-memory error on the GPU.
"""
import rmm
rmm.reinitialize(pool_allocator=True, managed_memory=True)
import os
# Must be set BEFORE any cupy/rapids_singlecell import — otherwise CuPy's JIT
# kernel compilation fails with "RuntimeError: Failed to find CUDA headers"
os.environ["CUDA_PATH"] = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/.environments/decoupler_gpu"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import decoupler as dc
import rapids_singlecell as rsc
from datetime import datetime

os.environ["TZ"] = "Europe/Berlin"
import time
time.tzset()


def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


BASE = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune"
H5AD = f"{BASE}/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad"
NET_LEVEL1_CSV = f"{BASE}/data/02_TICA/TICA_gene_markers_level1_table_s2.csv"
NET_LEVEL2_CSV = f"{BASE}/data/02_TICA/TICA_gene_markers_level2_table_s3.csv"
OUT_DIR = f"{BASE}/results/08_TICAtlas"
ANNOTATED_H5AD_OUT = f"{BASE}/data/01_extGBmap/immune_cells_gbmap_TICAannot_gpu_tmin2.h5ad"
os.makedirs(OUT_DIR, exist_ok=True)

TMIN = 2  # minimum number of marker genes (present in adata) required per cell type

checkpoint(f"decoupler version: {dc.__version__}")
checkpoint(f"rapids_singlecell version: {rsc.__version__}")
checkpoint("Setup complete.")


# ── 1. Load decoupler networks (Level 1 and Level 2 marker sets) ───────────────────────
def load_net(path):
    df = pd.read_csv(path)
    net = df.rename(columns={"cell_type": "source", "gene": "target"})[["source", "target"]].copy()
    net["weight"] = 1.0  # unweighted — all marker genes contribute equally
    return net


net_level1 = load_net(NET_LEVEL1_CSV)
net_level2 = load_net(NET_LEVEL2_CSV)

checkpoint(f"Level 1 — sources: {net_level1['source'].nunique()}, targets: {net_level1['target'].nunique()} unique genes")
checkpoint(f"Level 2 — sources: {net_level2['source'].nunique()}, targets: {net_level2['target'].nunique()} unique genes")


# ── 2. Load GBmap immune cell data ──────────────────────────────────────────────────────
adata = sc.read_h5ad(H5AD)

checkpoint(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")
checkpoint(f"adata.X max value: {adata.X.max():.2f} (should already be log-normalised)")

n_dup = adata.var_names.duplicated().sum()
if n_dup > 0:
    checkpoint(f"WARNING: {n_dup} duplicate var_names found — dropping (keeping first occurrence)")
    adata = adata[:, ~adata.var_names.duplicated()].copy()
else:
    checkpoint("No duplicate var_names — OK")

# GPU kernels expect float32
adata.X = adata.X.astype("float32")


# ── 3. Check marker gene coverage against adata.var_names ──────────────────────────────
for name, net in [("Level 1", net_level1), ("Level 2", net_level2)]:
    present = net["target"].isin(adata.var_names)
    n_missing = (~present).sum()
    if n_missing > 0:
        missing_genes = sorted(net.loc[~present, "target"].unique())
        checkpoint(f"{name}: {n_missing}/{len(net)} marker rows reference genes NOT in adata.var_names: {missing_genes}")
    else:
        checkpoint(f"{name}: all marker genes found in adata.var_names")


# ── 4. Move data to GPU ─────────────────────────────────────────────────────────────────
checkpoint("Moving adata.X to GPU...")
rsc.get.anndata_to_GPU(adata)
checkpoint("adata.X now resident on GPU.")


# ── 5. Run decoupler-GPU waggr(fun="wsum") — Level 1 ────────────────────────────────────
def sanitize_columns(df):
    """HDF5/h5ad keys can't contain '/', which some TICA cell type names do (e.g. 'Th17/helper')."""
    df = df.copy()
    df.columns = [str(c).replace("/", "_") for c in df.columns]
    return df

checkpoint("Running rsc.dcg.waggr (fun='wsum') — Level 1 (25 cell types)")
rsc.dcg.waggr(adata, net_level1, fun="wsum", tmin=TMIN)
checkpoint(f"Available obsm keys: {list(adata.obsm.keys())}")


scores_l1_df = sanitize_columns(adata.obsm.pop("score_waggr"))
padj_l1_df = sanitize_columns(adata.obsm.pop("padj_waggr"))
adata.obsm["level1_score_wsum"] = scores_l1_df
adata.obsm["level1_padj_wsum"] = padj_l1_df

if not isinstance(scores_l1_df, pd.DataFrame):
    scores_l1_df = pd.DataFrame(scores_l1_df, index=adata.obs_names)

adata.obs["decoupler_level1_celltype"] = scores_l1_df.idxmax(axis=1).values
adata.obs["decoupler_level1_score"] = scores_l1_df.max(axis=1).values

checkpoint("Level 1 dominant cell type distribution:")
print(adata.obs["decoupler_level1_celltype"].value_counts().to_string())


# ── 6. Run decoupler-GPU waggr(fun="wsum") — Level 2 ────────────────────────────────────
checkpoint("Running rsc.dcg.waggr (fun='wsum') — Level 2 (32 cell types)")
rsc.dcg.waggr(adata, net_level2, fun="wsum", tmin=TMIN)
checkpoint(f"Available obsm keys: {list(adata.obsm.keys())}")

scores_l2_df = sanitize_columns(adata.obsm.pop("score_waggr"))
padj_l2_df = sanitize_columns(adata.obsm.pop("padj_waggr"))
adata.obsm["level2_score_wsum"] = scores_l2_df
adata.obsm["level2_padj_wsum"] = padj_l2_df
if not isinstance(scores_l2_df, pd.DataFrame):
    scores_l2_df = pd.DataFrame(scores_l2_df, index=adata.obs_names)

adata.obs["decoupler_level2_celltype"] = scores_l2_df.idxmax(axis=1).values
adata.obs["decoupler_level2_score"] = scores_l2_df.max(axis=1).values

checkpoint("Level 2 dominant cell type distribution:")
print(adata.obs["decoupler_level2_celltype"].value_counts().to_string())


# ── 7. Move data back to CPU before saving/plotting ─────────────────────────────────────
checkpoint("Moving adata.X back to CPU...")
rsc.get.anndata_to_CPU(adata)
checkpoint("adata.X now resident on CPU.")


# ── 8. Save annotation table (CSV) ──────────────────────────────────────────────────────
annot_df = adata.obs[[
    "decoupler_level1_celltype", "decoupler_level1_score",
    "decoupler_level2_celltype", "decoupler_level2_score",
]].copy()
annot_out = os.path.join(OUT_DIR, "08_04_decoupler_gpu_celltype_annotation_tmin2.csv")
annot_df.to_csv(annot_out)
checkpoint(f"Annotation table saved to {annot_out}")


# ── 9. UMAP colored by decoupler annotation (Level 1) ───────────────────────────────────
if "X_umap" in adata.obsm:
    fig, ax = plt.subplots(figsize=(9, 7))
    sc.pl.umap(adata, color="decoupler_level1_celltype", ax=ax, show=False, size=3, legend_fontsize=7)
    ax.set_rasterized(True)
    plt.tight_layout()
    out_umap = os.path.join(OUT_DIR, "08_04_decoupler_gpu_level1_umap_tmin2.pdf")
    plt.savefig(out_umap, bbox_inches="tight", dpi=150)
    plt.close(fig)
    checkpoint(f"UMAP saved: {out_umap}")
else:
    checkpoint("No X_umap found in adata — skipping UMAP plot")


# ── 10. Save annotated adata for future sessions ────────────────────────────────────────
adata.write(ANNOTATED_H5AD_OUT)
checkpoint(f"Annotated adata saved to {ANNOTATED_H5AD_OUT}")
checkpoint("ALL DONE!")


# ── Fallback: chunked GPU loop (uncomment if a single call OOMs on the A30) ─────────────
# CHUNK_SIZE = 200_000
# n_chunks = (adata.n_obs + CHUNK_SIZE - 1) // CHUNK_SIZE
# all_scores = []
# for i in range(n_chunks):
#     start, end = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, adata.n_obs)
#     chunk = adata[start:end].copy()
#     rsc.get.anndata_to_GPU(chunk)
#     rsc.dcg.wsum(chunk, net_level1, tmin=TMIN)
#     scores = chunk.obsm["score_wsum"]
#     if not isinstance(scores, pd.DataFrame):
#         scores = pd.DataFrame(scores, index=chunk.obs_names)
#     all_scores.append(scores)
#     checkpoint(f"GPU chunk {i+1}/{n_chunks} done")
# scores_full = pd.concat(all_scores).reindex(adata.obs_names)
