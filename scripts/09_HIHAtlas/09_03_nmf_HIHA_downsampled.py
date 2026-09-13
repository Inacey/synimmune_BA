import kaleidocell
import scanpy as sc
import numpy as np
import anndata as ad
import os
from datetime import datetime

os.environ['TZ'] = 'Europe/Berlin'
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/09_HIHAtlas/09_03_nmf_downsampled/"
os.makedirs(OUT_DIR, exist_ok=True)

checkpoint("Loading data")
adata = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/03_HIHAtlas/HIHAtlas_ds_TICAannot.h5ad")
adata_for_nmf = ad.AnnData(X=adata.raw.X.copy(),
                           obs=adata.obs,
                           var=adata.raw.var)
checkpoint(f"Data loaded: {adata_for_nmf.n_obs} cells, {adata_for_nmf.n_vars} genes")

checkpoint("log-normalizing adata_for_nmf")
sc.pp.normalize_total(adata_for_nmf, target_sum=1e4)
sc.pp.log1p(adata_for_nmf)

checkpoint("Running NMF with fixed rank 10")
results_nmf, _ = kaleidocell.multi_sample_nmf(
    adata_for_nmf,
    batch_key="donor_id",
    test_ranks=[4, 5, 6, 7, 8, 9, 10],  
    n_initializations=3,
    max_iterations=100,
    seed=48
)
checkpoint("NMF done")



checkpoint("inspecting results_nmf")
print(type(results_nmf))
print(dir(results_nmf))

for k, v in results_nmf.items():
    try:
        print(k, type(v), getattr(v, "shape", None))
    except Exception as e:
        print(k, type(v), "error:", e)


for sample_id, df in results_nmf.items():
    arr = df.values
    nan_cols = np.where(np.isnan(arr).any(axis=0))[0]
    zero_cols = np.where((arr.sum(axis=0) == 0))[0]
    if len(nan_cols) > 0 or len(zero_cols) > 0:
        print(f"{sample_id}: NaN cols={list(nan_cols)}, zero-sum cols={list(zero_cols)}")


checkpoint("Deriving meta-programs")
results_mp = kaleidocell.derive_nmf_metaprograms(results_nmf)
checkpoint(f"{len(results_mp['mp_dict'])} MPs found")
print(results_mp["metrics"])

checkpoint("Saving results_mp")
kaleidocell.save(results_mp, os.path.join(OUT_DIR, "results_mp_healthy"))

checkpoint("Computing MP scores")
mp_scores = kaleidocell.compute_mp_scores(results_mp, adata_for_nmf)
mp_scores.to_csv(os.path.join(OUT_DIR, "mp_scores_healthy.csv"))
checkpoint("MP scores saved")

checkpoint("Generating HTML report")
path = kaleidocell.get_html(
    results_mp,
    adata_for_nmf,
    mp_scores=mp_scores,
    obs=None,
    output_path=OUT_DIR,
    gene_name_col="feature_name",
    gsea_sets={"C7": "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/c7.immunesigdb.v2026.1.Hs.symbols.gmt"}
)
checkpoint(f"HTML report done: {path}")

checkpoint("ALL DONE!")