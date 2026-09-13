import kaleidocell
import scanpy as sc
import os
import pickle
from datetime import datetime

cd4_cd8_filtered = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_subsets/CD4_CD8_filtered.h5ad")

tam_bdm_filtered = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_subsets/TAM-BDM_filtered.h5ad")

tam_mg_filtered = sc.read_h5ad("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/data/01_extGBmap/immune_subsets/TAM-MG_filtered.h5ad")



def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

for name, dataset in [("CD4_CD8", cd4_cd8_filtered), ("TAM-BDM", tam_bdm_filtered), ("TAM-MG", tam_mg_filtered)]:
    checkpoint(f"Starting {name} ({dataset.n_obs} cells, {dataset.obs['donor_id'].nunique()} donors)")
    
    # NMF
    checkpoint(f"{name} — Running NMF")
    results_nmf, _ = kaleidocell.multi_sample_nmf(
        dataset,
        batch_key="donor_id",
        test_ranks=[4, 5, 6, 7, 8, 9],
        n_initializations=3,
        max_iterations=100,
        seed=42
    )
    checkpoint(f"{name} — NMF done")
    
    # Meta-programs
    checkpoint(f"{name} — Deriving meta-programs")
    results_mp = kaleidocell.derive_nmf_metaprograms(results_nmf)
    checkpoint(f"{name} — {len(results_mp['mp_dict'])} meta-programs found")
    
    # MP scores
    checkpoint(f"{name} — Computing MP scores")
    mp_scores = kaleidocell.compute_mp_scores(results_mp, dataset)
    checkpoint(f"{name} — MP scores done")
    
    # Save pickle
    checkpoint(f"{name} — Saving results")
    os.makedirs(f"/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/03_NMF_filtered/03_nmf_{name}/", exist_ok=True)
    with open(f"/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/03_NMF_filtered/03_nmf_{name}/results_mp.pkl", "wb") as f:
        pickle.dump(results_mp, f)
    checkpoint(f"{name} — Pickle saved")
    
    # HTML report
    checkpoint(f"{name} — Generating HTML report")
    path = kaleidocell.get_html(
        results_mp,
        dataset,
        mp_scores=mp_scores,
        obs=["donor_id", "stage"],
        output_path=f"/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/03_NMF_filtered/03_nmf_{name}/"
    )
    checkpoint(f"{name} — HTML report done: {path}")
    
checkpoint("ALL DONE!")