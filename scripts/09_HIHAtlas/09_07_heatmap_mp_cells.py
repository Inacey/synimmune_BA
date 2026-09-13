import scanpy as sc
import decoupler as dc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
import os
from datetime import datetime
os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()


def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/09_HIHAtlas"
os.makedirs(OUT_DIR, exist_ok=True)


SELECTED_MPS = ["MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "MP7", "MP8"]


# Alternative mit mehr Grün-Anteilen
MP_PALETTE_HIHA_ALT = {
    "MP1": "#0caece",  # 
    "MP2": "#946da1",  # 
    "MP3": "#90e0ef",  # 
    "MP4": "#00c39f",  # 
    "MP5": "#655fdc",  # 
    "MP6": "#4cd276",  # 
    "MP7": "#6B157A",  # 
    "MP8": "#939393",  # 
}


LEVEL = "level1"  # "level1" (25 cell types) or "level2" (32 cell types)
STANDARDIZE_COLS = True  # min-max scale each TICA-celltype column — celltypes likely
                          # have different wsum score scales (different # marker genes),
                          # same reasoning as the earlier per-MP row standardization


checkpoint("Loading HIHA annotated data (TICA/decoupler)")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/03_HIHAtlas/HIHAtlas_ds_TICAannot_tmin2.h5ad")
checkpoint(f"Data loaded: {adata.n_obs} cells")


score_key = f"{LEVEL}_score_wsum"
if score_key not in adata.obsm:
    raise ValueError(f"'{score_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")


# use decoupler's own extraction to be safe against how obsm was serialized on save
acts = dc.pp.get_obsm(adata, key=score_key)
tica_scores = acts.to_df()  # cells x TICA cell types (continuous decoupler score)
checkpoint(f"TICA score matrix ({LEVEL}): {tica_scores.shape}")


checkpoint("Loading mp_scores")
mp_scores = pd.read_csv("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/09_HIHAtlas/09_03_nmf_downsampled/mp_scores_healthy.csv",index_col=0)


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
tica_scores = tica_scores.loc[common_cells]


mp_scores_sel = mp_scores[[f"{mp}_score" for mp in SELECTED_MPS]]
mp_scores_sel.columns = SELECTED_MPS


checkpoint("Computing dominant MP per cell")
order_map = {mp: i for i, mp in enumerate(SELECTED_MPS)}

dominant_MP = mp_scores_sel.idxmax(axis=1)
max_scores = mp_scores_sel.max(axis=1)
per_mp_threshold = mp_scores_sel.quantile(0.75, axis=0)  # here was before was a single threshold for all MPs, now per-MP 75th percentile
own_score_all = mp_scores_sel.values[np.arange(len(mp_scores_sel)), dominant_MP.map(order_map).values]
own_threshold = dominant_MP.map(per_mp_threshold.to_dict()).values

dominant_MP_selected = pd.Series(
    np.where(own_score_all > own_threshold, dominant_MP.values, "Other"),
    index=mp_scores_sel.index,
)


# ── just cells with dominant MP of 10 selected ("Other" cells is dropped)
keep_mask = dominant_MP_selected.isin(SELECTED_MPS)
checkpoint(f"Keeping {keep_mask.sum()} / {len(keep_mask)} cells (dropping 'Other')")


scores_kept = mp_scores_sel.loc[keep_mask]
groups_kept = dominant_MP_selected.loc[keep_mask]


# ── sortieren: nach dominant MP gruppiert, innerhalb jeder Gruppe absteigend
# nach dem jeweils eigenen MP-Score
own_score = scores_kept.values[np.arange(len(scores_kept)), groups_kept.map(order_map).values]
sort_df = pd.DataFrame({
    "group_order": groups_kept.map(order_map).values,
    "own_score": own_score,
}, index=scores_kept.index)
sort_order = sort_df.sort_values(["group_order", "own_score"], ascending=[True, False]).index
groups_sorted = groups_kept.loc[sort_order]

# ── pro dominant-MP-Gruppe auf max. MAX_CELLS_PER_MP Zellen kappen — da
# bereits absteigend nach own_score sortiert, behalten wir die Top-N je Gruppe
MAX_CELLS_PER_MP = 500


capped_parts = []
for mp in SELECTED_MPS:
    idx_mp = groups_sorted[groups_sorted == mp].index
    n_before = len(idx_mp)
    idx_mp_capped = idx_mp[:MAX_CELLS_PER_MP]
    checkpoint(f"  {mp}: {n_before} -> {len(idx_mp_capped)} cells")
    capped_parts.append(idx_mp_capped)
sort_order_capped = pd.Index(np.concatenate(capped_parts))
groups_sorted = groups_sorted.loc[sort_order_capped]
checkpoint(f"Total cells after per-MP cap ({MAX_CELLS_PER_MP}): {len(sort_order_capped)}")

# ── TICA-Scores auf dieselben (gefilterten, sortierten) Zellen ausrichten
tica_sorted = tica_scores.loc[sort_order_capped]
matrix = tica_sorted.values  # shape: (n_cells_kept, n_tica_celltypes)


if STANDARDIZE_COLS:
    col_min = matrix.min(axis=0, keepdims=True)
    col_max = matrix.max(axis=0, keepdims=True)
    matrix = (matrix - col_min) / np.maximum(col_max - col_min, 1e-9)


# ── TRANSPOSE: TICA-Zelltypen = Zeilen, Zellen = Spalten
matrix_t = matrix.T  # shape: (n_tica_celltypes, n_cells_kept)


checkpoint(f"Matrix shape (transposed): {matrix_t.shape} (TICA-cell types x Cells)")


# ═══════════════════════════════════════════════════════════════
# Plot: TICA-Zelltypen als Zeilen, Zellen als Spalten. Farbstreifen oben zeigt die dominant-MP-Gruppe jeder Spalte.
# ═══════════════════════════════════════════════════════════════
checkpoint("Plotting heatmap (gridspec, aligned)")
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[0.03, 1], width_ratios=[1, 0.03], hspace=0.02, wspace=0.05)
ax_group = fig.add_subplot(gs[0, 0])
ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_group)
ax_cbar = fig.add_subplot(gs[1, 1])

group_colors = np.array([MP_PALETTE_HIHA_ALT[mp] for mp in groups_sorted])
strip_rgb = np.array([to_rgb(c) for c in group_colors])[np.newaxis, :, :]

n_cells = matrix_t.shape[1]
ax_group.imshow(strip_rgb, aspect="auto", extent=[0, n_cells, 0, 1])
ax_group.set_yticks([])
for spine in ax_group.spines.values():
    spine.set_visible(False)

im = ax_heat.imshow(matrix_t, aspect="auto", cmap="inferno", interpolation="none",
                     extent=[0, n_cells, matrix_t.shape[0], 0])
ax_heat.set_xlim(0, n_cells)
ax_group.set_xlim(0, n_cells)
ax_heat.set_yticks(np.arange(tica_sorted.shape[1]) + 0.60)
ax_heat.set_yticklabels(tica_sorted.columns, fontsize=7, va="center")
ax_heat.set_xticks([])
ax_heat.set_xlabel(f"Cells (n={n_cells}, grouped by dominant MP, 75th percentile cutoff per MP)")
ax_heat.set_ylabel(f"TICA-cell type ({LEVEL})")

cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label("decoupler wsum score" + (" (col-scaled 0-1)" if STANDARDIZE_COLS else ""))

legend_handles = [Patch(facecolor=MP_PALETTE_HIHA_ALT[mp], label=mp) for mp in SELECTED_MPS]
ax_group.legend(
    handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.05),
    fontsize=7, ncol=len(SELECTED_MPS), frameon=False)

fig.suptitle(f"TICA-cell type scores by cell ({LEVEL}, max {MAX_CELLS_PER_MP} cells/MP)", y=0.95)
out_path = os.path.join(OUT_DIR, f"09_07_heatmap_healthy_tica_{LEVEL}_500cells_2.pdf")
plt.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close()
checkpoint(f"Heatmap gespeichert: {out_path}")
checkpoint("ALL DONE!")