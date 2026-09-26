import kaleidocell
import scanpy as sc
import os
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/10_integration_GBM_HIHA/10_04_nmf_macro"
os.makedirs(OUT_DIR, exist_ok=True)


checkpoint("Loading data")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/10_integration_GBM_HIHA/10_subset_macro.h5ad")
checkpoint(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")


checkpoint("Running NMF")
results_nmf, _ = kaleidocell.multi_sample_nmf(
    adata,
    batch_key="donor_id",
    test_ranks=[4, 5, 6, 7, 8, 9],  
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
kaleidocell.save(results_mp, os.path.join(OUT_DIR, "results_mp"))

checkpoint("Computing MP scores")
mp_scores = kaleidocell.compute_mp_scores(results_mp, adata)
mp_scores.to_csv(os.path.join(OUT_DIR, "mp_scores.csv"))
checkpoint("MP scores saved")

checkpoint("Generating HTML report")
path = kaleidocell.get_html(
    results_mp,
    adata,
    mp_scores=mp_scores,
    output_path=OUT_DIR,
)

checkpoint(f"HTML report done: {path}")

checkpoint("ALL DONE!")
