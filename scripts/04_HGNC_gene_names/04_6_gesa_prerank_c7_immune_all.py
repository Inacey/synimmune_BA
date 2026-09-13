import kaleidocell
import gseapy as gp
import pandas as pd
import os
from datetime import datetime

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_6_gsea_c7_immune_all/"
os.makedirs(OUT_DIR, exist_ok=True)

GMT_PATH = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/c7.immunesigdb.v2026.1.Hs.symbols.gmt"

checkpoint("Loading results_mp")
results_mp = kaleidocell.load("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_3_nmf_all_gsea_c7/results_mp.kc")
mp_dict = results_mp["mp_dict"]
checkpoint(f"Loaded {len(mp_dict)} MPs")

all_results = []

for mp_name, gene_scores in mp_dict.items():
    checkpoint(f"Running GSEA prerank for {mp_name}")
    rnk = gene_scores.sort_values(ascending=False)
    rnk = rnk[~rnk.index.duplicated(keep="first")]

    try:
        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets=GMT_PATH,
            min_size=5,
            max_size=1000,
            permutation_num=1000,
            seed=42,
            threads=4,
            outdir=None,
            no_plot=True
        )
        res_df = pre_res.res2d.copy()
        res_df["MP"] = mp_name
        all_results.append(res_df)
        checkpoint(f"{mp_name}: {len(res_df)} pathways tested")
    except Exception as e:
        checkpoint(f"{mp_name} FAILED: {e}")

checkpoint("Combining all results")
full_gsea_results = pd.concat(all_results, ignore_index=True)
full_gsea_results.to_csv(os.path.join(OUT_DIR, "04_gsea_c7_immunesigdb.csv"), index=False)
checkpoint("Saved full GSEA results with padj")