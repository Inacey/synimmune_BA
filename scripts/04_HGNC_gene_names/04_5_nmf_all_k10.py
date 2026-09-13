import kaleidocell
import scanpy as sc
import os
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_5_nmf_k10/"
os.makedirs(OUT_DIR, exist_ok=True)

checkpoint("Loading data")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad")
checkpoint(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")

checkpoint("Running NMF with fixed rank 10")
results_nmf, _ = kaleidocell.multi_sample_nmf(
    adata,
    batch_key="donor_id",
    test_ranks=[10],  
    n_initializations=3,
    max_iterations=100,
    seed=42
)
checkpoint("NMF done")

checkpoint("Deriving meta-programs")
results_mp = kaleidocell.derive_nmf_metaprograms(results_nmf)
checkpoint(f"{len(results_mp['mp_dict'])} MPs found")
print(results_mp["metrics"])

checkpoint("Saving results_mp")
kaleidocell.save(results_mp, os.path.join(OUT_DIR, "results_mp_k10"))

checkpoint("Computing MP scores")
mp_scores = kaleidocell.compute_mp_scores(results_mp, adata)
mp_scores.to_csv(os.path.join(OUT_DIR, "mp_scores_k10.csv"))
checkpoint("MP scores saved")

checkpoint("Generating HTML report")
path = kaleidocell.get_html(
    results_mp,
    adata,
    mp_scores=mp_scores,
    obs=["annotation_level_3"],
    output_path=OUT_DIR,
    gsea_sets={"C7": "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/c7.immunesigdb.v2026.1.Hs.symbols.gmt"}
)
checkpoint(f"HTML report done: {path}")

checkpoint("ALL DONE!")