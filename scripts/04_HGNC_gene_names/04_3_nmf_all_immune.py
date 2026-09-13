import kaleidocell
import scanpy as sc
import os
from datetime import datetime

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

checkpoint("Loading data")
adata = sc.read_h5ad('/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad')
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
checkpoint(f"Meta-programs done: {len(results_mp['mp_dict'])} MPs found")
print(results_mp["metrics"])

checkpoint("Saving results_mp")
kaleidocell.save(results_mp, "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/results_mp")
checkpoint("results_mp saved")

mp_scores = kaleidocell.compute_mp_scores(results_mp, adata)

os.makedirs("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_nmf_all_gsea", exist_ok=True)

checkpoint("Generating HTML report with GSEA")
path = kaleidocell.get_html(
    results_mp,
    adata,
    mp_scores=mp_scores,
    obs=None,
    output_path="/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_nmf_all_gsea",
    gsea_sets={"C7": "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/c7.all.v2026.1.Hs.symbols.gmt"}
)
checkpoint(f"HTML report done: {path}")