import kaleidocell
import scanpy as sc
import os
from datetime import datetime

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUTPUT = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_3_2_rerun_gsea_all"
os.makedirs(OUTPUT, exist_ok=True)

checkpoint("Loading data")
adata = sc.read_h5ad('/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad')
checkpoint(f"Data loaded: {adata.n_obs} cells")

checkpoint("Loading results_mp")
results_mp = kaleidocell.load("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_3_nmf_all_gsea_c7/results_mp.kc")
checkpoint(f"results_mp loaded")

checkpoint("Computing MP scores")
mp_scores = kaleidocell.compute_mp_scores(results_mp, adata)
checkpoint("MP scores done")

checkpoint("Generating HTML report")
path = kaleidocell.get_html(
    results_mp,
    adata,
    mp_scores=mp_scores,
    obs=None,
    output_path=OUTPUT
)
checkpoint(f"Done: {path}")