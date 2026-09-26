# SynImmune

Identification of shared transcriptional Metaprograms (MPs) in glioblastoma-infiltrating immune cells, and their comparison to a healthy immune reference atlas.

## Overview

This repository analyzes tumor-associated immune cells from the extended GBMap dataset (Glioblastoma (GBM), ~633k immune cells across ~230 donors) to find gene expression programs that recur across patients within each immune cell type. Metaprograms are derived with [KaleidoCell](https://github.com/JeanRadig/KaleidoCell) — an NMF-based consensus method that factorizes each donor independently and clusters the resulting programs across donors — then filtered for quality, annotated with GSEA (GO), and compared against a healthy immune reference (Human Immune Health Atlas) to separate tumor-specific programs from baseline immune states.

## Data

| Dataset | Description |
|---|---|
| Extended GBMap | ~633k immune cells, 10 cell-type subsets (TAM-BDM, TAM-MG, CD4/CD8, Mono, DC, NK, B cell, Plasma B, Neutrophil, Mast), HGNC-annotated |
| Human Immune Health Atlas | Healthy immune reference atlas, used for cross-dataset MP comparison |
| Tumor Immune Cell Atlas marker tables | Level 1 (25 cell types) / Level 2 (32 cell types) marker sets, used for decoupler-based cell-type annotation |
| GO Biological Process | `c5.go.bp.v2026.1.Hs.symbols.gmt` , used for GSEA |

Raw and intermediate data are not tracked in git (too large); everything is located in `/net/bq-storage/ag-cherrmann/bq_cinac/projects/synimmune/` on the BioQuant cluster. 

## Environment

Two conda environments, under `.environments/` in the project root:

- **`kaleidocell_env`** (Python 3.11) — kaleidocell 0.1.2, decoupler 2.2.0, scanpy, leidenalg. Used for cNMF, clustering, and most analysis.
- **`decoupler_gpu`** (Python 3.12) — rapids_singlecell / cupy for GPU-accelerated cell-type scoring.


## Repository structure

```
scripts/
  01_all_immune_cells/         subsetted extended GBMap to only immune cells, full-cohort UMAP, cell-type overview
  02_NMF_immune_subsets/       per-cell-type cNMF (raw ranks) (exploratory, not used downstream)
  03_filtered_immune_cells/    donor-filtered subsets (>=50 cells/donor)
  04_HGNC_gene_names/          Ensembl -> HGNC conversion, cNMF on all GBM immune cells + GSEA
  05_significant_MP/           MP quality filtering, selection, visualization
  06_leiden/                   Leiden subclustering (exploratory, not used downstream)
  07_DEA_GSEA/                 Leiden-cluster DEA + GSEA (exploratory, not used downstream)
  08_TICAannot_on_GBmap/       TICA-based cell-type annotation on GBM immune cells (decoupler, GPU), cell type annotation comparison, visualization
  09_HIHAtlas/                 TICA-based cell-type annotation on Healthy immune cells (decoupler, GPU), full-cohort cNMF + GSEA, cross-dataset MP comparison (GBM vs. Healthy)
  10_integration_GB_HIHA/      Harmony-integrated GBM + Healthy, myeloid-focused joint MPs
results/
  <mirrors the scripts/ numbering above>
```

## Pipeline

1. **Subset & convert** — split immune cells by `annotation_level_3`, convert Ensembl → HGNC gene names.
2. **cNMF (KaleidoCell)** — `multi_sample_nmf` (`batch_key="donor_id"`, `test_ranks=[4-9]`, `n_initializations=3`, `seed=42`). Donors with <50 cells filtered out of the larger subsets before factorization.
3. **Consensus Metaprograms** — `derive_nmf_metaprograms` clusters programs across donors into MPs; quality tracked via `silhouette`, `sampleCoverage`, `meanSimilarity`, `n_genes`, `mt_fraction`.
4. **GSEA** — GO term enrichment per MP, run via kaleidocell's built-in GSEA pipeline.
5. **MP selection** — a combined rank over the core quality metrics (not a single threshold) picks the final MP set; known artifacts (heat-shock/dissociation signatures, mitochondrial-content programs, donor-driven programs) are excluded explicitly. See `results/05_significant_MP/` for the full metrics table and exclusion rationale.
6. **Annotation & comparison** — TICA marker-based cell-type scoring (decoupler `wsum`, GPU), and MP comparison, GBM against Healthy dataset, by gene-set overlap (Szymkiewicz–Simpson) and per-cell MP score correlation (Pearson and Spearman).
7. **Integrated myeloid analysis** — GBM cells downsampled, concatenated with Healthy Immune Cell Atlas, and Harmony-integrated; Healthy-derived MP scores are projected onto the combined object. The integrated set is subsetted to the myeloid/monocyte lineage (Macrophages SPP1, Macro. and Mono. Prolif., TAMs C1QC, TAMs proinflammatory, Monocytes) for a new KaleidoCell run, followed by DEG and GSEA — comparing tumor-associated macrophage states against their healthy counterparts in one integrated embedding.

GSEA methodology differs by step: scripts 1-9 use KaleidoCell's built-in GSEA on MP gene sets (GO terms); step 10 runs GSEA on ranked DEG scores instead of through KaleidoCell.



## Known issues

- NMF input must stay non-negative, log-normalized counts — never run it on `sc.pp.scale()`-transformed data.
- `mp_scores.csv` is wide (one column per MP); the top-genes-per-MP table is long (`gene, mp, score`).
- Cell barcodes can carry mismatched suffixes across files — reindex explicitly after intersecting, don't `.loc[]` directly.
- GPU env: `CUDA_PATH` must be set before the first `cupy`/`rapids_singlecell` import, or the kernel needs restarting.


## Contact

Ceyda Inac, AG Carl Herrmann Lab, Institute for Pharmacy and Molecular Biotechnology (IPMB), Heidelberg University.
Ceyda.inac@stud.uni-heidelberg.de