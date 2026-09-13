import kaleidocell
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

OUT_DIR = "/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/05_significant_MP/05_all_immune_cells/"
os.makedirs(OUT_DIR, exist_ok=True)

checkpoint("Loading results_mp")
results_mp = kaleidocell.load("/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/results/04_NMF_GSEAonHGNC/04_3_nmf_all_gsea_c7/results_mp.kc")
checkpoint(f"Loaded: {len(results_mp['mp_dict'])} MPs")

mp_dict = results_mp["mp_dict"]

checkpoint("Checking mp_dict structure")
example_mp = list(mp_dict.keys())[0]
print(type(mp_dict[example_mp]), mp_dict[example_mp])

mp_genes = {}
for mp, genes in mp_dict.items():
    if isinstance(genes, dict):
        mp_genes[mp] = set(genes.keys())
    elif isinstance(genes, pd.Series):
        mp_genes[mp] = set(genes.index)
    else:
        mp_genes[mp] = set(genes)

def jaccard(a, b):
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0

mps = list(mp_genes.keys())
checkpoint(f"Computing pairwise Jaccard similarity for {len(mps)} MPs")

records = []
for i, mp1 in enumerate(mps):
    for mp2 in mps[i+1:]:
        j = jaccard(mp_genes[mp1], mp_genes[mp2])
        records.append({"MP1": mp1, "MP2": mp2, "jaccard": j})

jaccard_df = pd.DataFrame(records)
jaccard_df.to_csv(os.path.join(OUT_DIR, "05_9_mp_jaccard_similarity.csv"), index=False)
checkpoint("Jaccard similarity table saved")

THRESHOLD = 0.1

checkpoint(f"Building network with threshold={THRESHOLD}")
G = nx.Graph()
G.add_nodes_from(mps)

for _, row in jaccard_df.iterrows():
    if row["jaccard"] >= THRESHOLD:
        G.add_edge(row["MP1"], row["MP2"], weight=row["jaccard"])

checkpoint(f"Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

pos = nx.spring_layout(G, seed=42, weight="weight", k=0.5)
weights = [G[u][v]["weight"] * 8 for u, v in G.edges()]

plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_size=900, node_color="lightblue", edgecolors="black")
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.4, edge_color="gray")
plt.axis("off")
plt.title(f"MP Gene-Set Jaccard Similarity Network (threshold={THRESHOLD})")
plt.savefig(os.path.join(OUT_DIR, "05_9_mp_jaccard_network.pdf"), bbox_inches="tight")
plt.close()

checkpoint("Network plot saved")
checkpoint("ALL DONE!")