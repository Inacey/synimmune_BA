import os
import pandas as pd
import scanpy as sc
import decoupler as dc
from datetime import datetime

os.environ["TZ"] = "Europe/Berlin"
import time
time.tzset()

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

BASE = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune"
H5AD = f"{BASE}/data/01_extGBmap/immune_cells_extended_gbmap_hgnc.h5ad"
NET_LEVEL1_CSV = f"{BASE}/data/02_TICA/TICA_gene_markers_level1_table_s2.csv"
TMIN = 3

def load_net(path):
    df = pd.read_csv(path)
    net = df.rename(columns={"cell_type": "source", "gene": "target"})[["source", "target"]].copy()
    net["weight"] = 1.0
    return net

net_level1 = load_net(NET_LEVEL1_CSV)

checkpoint("Loading adata...")
adata = sc.read_h5ad(H5AD)
n_dup = adata.var_names.duplicated().sum()
if n_dup > 0:
    adata = adata[:, ~adata.var_names.duplicated()].copy()
checkpoint(f"Loaded: {adata.n_obs} cells")

for n in [10_000, 20_000, 30_000, 40_000]:
    checkpoint(f"Testing n={n}")
    test_chunk = adata[:n].copy()
    dc.mt.waggr(data=test_chunk, net=net_level1, fun="wsum", tmin=TMIN)
    checkpoint(f"n={n} OK")
    del test_chunk

checkpoint("ALL THRESHOLD TESTS DONE")