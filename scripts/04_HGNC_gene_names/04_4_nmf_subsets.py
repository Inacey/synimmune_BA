
import kaleidocell
import scanpy as sc
import os
from datetime import datetime

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

SUBSET_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/hgnc_immune_subsets"
OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_4_nmf_subsets_c7"
GSEA_C7_PATH = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/c7.immunesigdb.v2026.1.Hs.symbols.gmt"

os.makedirs(OUT_DIR, exist_ok=True)

SUBSET_CONFIG = {
    "B_cell":     {"test_ranks": [2, 3, 4],          "min_donor_cells": None},
    "Mast":       {"test_ranks": [2, 3, 4],          "min_donor_cells": None},
    "Plasma_B":   {"test_ranks": [2, 3, 4],          "min_donor_cells": None},
    "DC":         {"test_ranks": [2, 3, 4, 5, 6],    "min_donor_cells": None},
    "Neutrophil": {"test_ranks": [2, 3, 4, 5, 6],    "min_donor_cells": None},
    "NK":         {"test_ranks": [2, 3, 4, 5, 6],    "min_donor_cells": None},
    "TAM-BDM":    {"test_ranks": [4, 5, 6, 7, 8, 9], "min_donor_cells": 50},
    "TAM-MG":     {"test_ranks": [4, 5, 6, 7, 8, 9], "min_donor_cells": 50},
    "CD4_CD8":    {"test_ranks": [4, 5, 6, 7, 8, 9], "min_donor_cells": 50},
    "Mono":       {"test_ranks": [4, 5, 6, 7, 8, 9], "min_donor_cells": 50},
}

def filter_donors_min_cells(adata, min_cells, batch_key="donor_id"):
    donor_counts = adata.obs[batch_key].value_counts()
    keep_donors = donor_counts[donor_counts >= min_cells].index
    n_before = adata.obs[batch_key].nunique()
    adata_filtered = adata[adata.obs[batch_key].isin(keep_donors)].copy()
    n_after = adata_filtered.obs[batch_key].nunique()
    checkpoint(
        f"  Donor filter (>= {min_cells} cells): "
        f"{n_before} -> {n_after} donors, "
        f"{adata.n_obs} -> {adata_filtered.n_obs} cells"
    )
    return adata_filtered

def find_subset_file(cell_type):
    safe_name = cell_type.replace("/", "_")
    candidates = [f"hgnc_{safe_name}.h5ad", f"hgnc_{cell_type}.h5ad"]
    for c in candidates:
        path = os.path.join(SUBSET_DIR, c)
        if os.path.exists(path):
            return path
    matches = [f for f in os.listdir(SUBSET_DIR) if safe_name in f and f.endswith(".h5ad")]
    if matches:
        return os.path.join(SUBSET_DIR, matches[0])
    raise FileNotFoundError(f"No file found for subset '{cell_type}' in {SUBSET_DIR}")

def run_subset(cell_type, config):
    safe_name = cell_type.replace("/", "_")
    checkpoint(f"=== Processing {cell_type} ===")

    filepath = find_subset_file(cell_type)
    checkpoint(f"Loading {filepath}")
    adata = sc.read_h5ad(filepath)
    checkpoint(f"{cell_type}: {adata.n_obs} cells, {adata.n_vars} genes loaded")

    if config["min_donor_cells"] is not None:
        adata = filter_donors_min_cells(adata, config["min_donor_cells"], batch_key="donor_id")

    if adata.n_obs == 0:
        checkpoint(f"{cell_type}: 0 cells remaining after filtering, skipping")
        return

    test_ranks = config["test_ranks"]
    subset_out_dir = os.path.join(OUT_DIR, safe_name)
    os.makedirs(subset_out_dir, exist_ok=True)

    checkpoint(f"{cell_type}: Running NMF with test_ranks={test_ranks}")
    results_nmf, _ = kaleidocell.multi_sample_nmf(
        adata,
        batch_key="donor_id",
        test_ranks=test_ranks,
        n_initializations=3,
        max_iterations=100,
        seed=442,
    )
    checkpoint(f"{cell_type}: NMF done")

    checkpoint(f"{cell_type}: Deriving meta-programs")
    results_mp = kaleidocell.derive_nmf_metaprograms(results_nmf)
    checkpoint(f"{cell_type}: Meta-programs done: {len(results_mp['mp_dict'])} MPs found")
    print(results_mp["metrics"])

    mp_save_path = os.path.join(subset_out_dir, f"results_mp_{safe_name}")
    checkpoint(f"{cell_type}: Saving results_mp to {mp_save_path}")
    kaleidocell.save(results_mp, mp_save_path)
    checkpoint(f"{cell_type}: results_mp saved")

    checkpoint(f"{cell_type}: Computing MP scores")
    mp_scores = kaleidocell.compute_mp_scores(results_mp, adata)
    mp_scores.to_csv(os.path.join(subset_out_dir, f"mp_scores_{safe_name}.csv"))
    checkpoint(f"{cell_type}: mp_scores saved")

    checkpoint(f"{cell_type}: Recomputing PCA and UMAP on adata in")
    kaleidocell.recompute_pca_umap(adata, n_pcs=50, umap_min_dist=0.5, umap_spread=1.0, random_state=442)


    checkpoint(f"{cell_type}: Generating HTML report with GSEA (C7)")
    html_out_dir = os.path.join(subset_out_dir, "gsea_c7")
    os.makedirs(html_out_dir, exist_ok=True)
    path = kaleidocell.get_html(
        results_mp,
        adata,
        mp_scores=mp_scores,
        obs=None,
        output_path=html_out_dir,
        gsea_sets={"C7": GSEA_C7_PATH},
    )
    checkpoint(f"{cell_type}: HTML report done: {path}")
    checkpoint(f"=== {cell_type} DONE ===\n")

# 
if __name__ == "__main__":
    checkpoint(f"Starting NMF subset run for {len(SUBSET_CONFIG)} subsets")
    checkpoint(f"Output directory: {OUT_DIR}")

    failed = []
    for cell_type, config in SUBSET_CONFIG.items():
        try:
            run_subset(cell_type, config)
        except Exception as e:
            checkpoint(f"ERROR processing {cell_type}: {e}")
            failed.append(cell_type)

    checkpoint("=== ALL SUBSETS PROCESSED ===")
    if failed:
        checkpoint(f"Failed subsets: {failed}")
    else:
        checkpoint("All subsets completed successfully")