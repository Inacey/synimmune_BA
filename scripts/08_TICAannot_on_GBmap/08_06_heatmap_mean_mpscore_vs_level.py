import scanpy as sc
import pandas as pd
import decoupler as dc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/08_TICAtlas"
os.makedirs(OUT_DIR, exist_ok=True)

SELECTED_MPS = ["MP1", "MP2", "MP7", "MP8", "MP9", "MP11", "MP14", "MP15", "MP19", "MP20"]

checkpoint("Loading annotated data (TICA/decoupler annotation)")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_gbmap_TICAannot_gpu.h5ad")
checkpoint(f"Data loaded: {adata.n_obs} cells")

for col in ["decoupler_level1_celltype", "decoupler_level2_celltype"]:
    if col not in adata.obs.columns:
        raise ValueError(f"'{col}' not found in adata.obs — did the decoupler annotation notebook run and save correctly?")




checkpoint("Loading mp_scores")
mp_scores = pd.read_csv("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_3_nmf_all_gsea_c7/mp_scores.csv",index_col=0)

# ── mp_scores.csv and adata (TICA-annotated) use different barcode suffix
# conventions (e.g. "-1-1-1" vs "-1-0") — restrict both to their common cell
# intersection BEFORE any downstream computation, instead of assuming a
# shared index via .loc[adata.obs_names] (which throws a KeyError otherwise).
common_cells = mp_scores.index.intersection(adata.obs_names)
n_dropped_mp = len(mp_scores) - len(common_cells)
n_dropped_adata = adata.n_obs - len(common_cells)
checkpoint(
    f"Common cells: {len(common_cells)} "
    f"(dropping {n_dropped_mp} mp_scores-only cells, {n_dropped_adata} adata-only cells)"
)
if len(common_cells) == 0:
    raise ValueError(
        "No overlapping cell barcodes between mp_scores.csv and adata — "
        "check barcode suffix conventions on both sides before proceeding."
    )


mp_scores = mp_scores.loc[common_cells]
# tica_scores = tica_scores.loc[common_cells]


mp_scores_sel = mp_scores[[f"{mp}_score" for mp in SELECTED_MPS]]
mp_scores_sel.columns = SELECTED_MPS


# Heatmap: mean (continuous) MP-Scores per TICA cell type.
# Uses the full score distribution instead of just the hard "dominant MP" assignment.

for level, col in [("level1", "decoupler_level1_celltype"), ("level2", "decoupler_level2_celltype")]:
    checkpoint(f"Calculate mean MP-Score per TICA cell type ({level})")
    obs_col_common = adata.obs.loc[common_cells, col]
    mean_scores = pd.DataFrame(index=sorted(adata.obs[col].unique()), columns=SELECTED_MPS, dtype=float)
    for mp in SELECTED_MPS:
        grouped = pd.Series(mp_scores_sel[mp].values, index=common_cells).groupby(obs_col_common).mean()
        mean_scores[mp] = grouped
    checkpoint(f"mean_scores shape: {mean_scores.shape} (erwartet: {adata.obs[col].nunique()} Zeilen)")
    
    mean_scores.to_csv(os.path.join(OUT_DIR, f"08_06_mp_mean_score_by_decoupler_{level}.csv"))
    checkpoint(f"Table ({level}) saved")

    fig, ax = plt.subplots(figsize=(8, max(6, mean_scores.shape[0] * 0.35)))
    sns.heatmap(
        mean_scores.astype(float), cmap="RdPu", annot=True, fmt=".2f",
        ax=ax, cbar_kws={"label": "mean MP-Score"},
    )
    ax.set_title(f"Mean MP-Score per TICA cell type ({level})")
    ax.set_ylabel("TICA cell type (decoupler wsum)")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"08_06_mp_mean_score_by_decoupler_{level}_heatmap.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()
    checkpoint(f"Heatmap ({level}) saved: {out_path}")

checkpoint("ALL DONE!")