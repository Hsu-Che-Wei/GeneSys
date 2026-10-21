# Revision material

This folder holds what was generated during the revision of the manuscript. It is kept separate from the author's notebooks 0 to 13 in `../jupyter_notebook`, which define how the published analyses were done. Two of those were changed in the revision, and only in the labels of two LIMA centralities:

* Notebooks 12 (ten files, GeneSys and scRNA-seq): out-degree and in-degree were stored under each other's names. The two assignments are now exchanged, so that out-degree counts the targets a TF regulates (row of W) and in-degree the regulators acting on it (column of W). The graph and every other centrality are unchanged (`codes/regenerate_notebook12.py`, checked by `codes/check_edge_direction.py`).
* Notebook 13 (Figure 4) was re-executed on these tables. Every R50 is the same as before, apart from the out-degree and in-degree values trading places. The per-cell-type TF rankings it writes are identical to the earlier ones.

* `notebooks/`: four executed notebooks (with HTML copies). Each one reads the result tables in `results/`, recomputes the numbers that the manuscript quotes, checks each of them with an `assert`, and draws its figures from the same tables with `codes/exp24_manuscript_figures.py`.
* `codes/`: the scripts that produced the results, with one line per script in `codes/README.md`.
* `results/`: the small result tables and the figure images that the notebooks read.

| Notebook | Experiments in the Methods | Figures |
|---|---|---|
| `R1-GeneSys_model_validation_blueprint_external_classifiers_ablation` | 1 decoy blueprint, 2 external validation of generated transcriptomes, 3 ablation study | S6 |
| `R2-GeneSys_model_validation_random_init_extrapolation_noise_width_KL` | 4 random-initialization control, 8 extrapolation, 9 noise sensitivity, 10 latent-dimension sensitivity, 11 KL-divergence monitoring | S5E-F, S7, S14 |
| `R3-GeneSys_prioritization_benchmark_scenarios_and_nonlinear_criteria` | pseudobulk and scVI scenarios of the prioritization benchmark, 5 in silico knockout, 6 GRNBoost2, 7 genome-wide recall | 5, S11 |
| `R4-GeneSys_LIMA_identifiability_and_zscore_threshold` | 12 LIMA identifiability and regularization, 13 z-score threshold sensitivity | S15 |

## How the models were trained

Every model retrained for the revision (decoy blueprints, ablations, noise levels, widths, KL monitoring) is trained by the training function of the GeneSys package, `run_cycle` in `genesys/genesys_cli.py`, called unchanged through `codes/train_run_cycle.py`: one cycle of 30 epochs with the package defaults (AdamW at 1e-3, ReduceLROnPlateau with factor 0.5, patience 10 and threshold 0.05, gradient clipping at 5, best weights kept on lowest loss). Thirty epochs are used because each experiment compares variants under one training budget or follows the loss from the start of training.

What an experiment varies enters through the inputs of `run_cycle` only:

* the labels, in the convention of the package with time starting at 1 (`utils.load_root_data_blueprint`), time-reversed for the reversed decoy;
* the lineage blueprint table (correct, reversed, shuffled, random);
* the model object (noise level, width, a feedforward network in place of the LSTM, or the TD-VAE loss switched off).

With these inputs the blueprint class builds the same sequences as the `Root_Dataset` of the original notebooks: the stem cell at t0, then time bins 1 to 10 of the trajectory's cell type (`codes/check_blueprint_sequences.py`).

Analyses that use the published model (extrapolation, random-initialization control, LIMA identifiability, z-score threshold, GRNBoost2, in silico knockout) follow the procedures of the author's notebooks 4, 10 and 12. The GeneSys and scRNA-seq rows of Figure 5 are the values of Figure 4.

## Running the notebooks

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace R1-*.ipynb
```

The notebooks need numpy, pandas, scipy, matplotlib, Pillow and IPython, and read files with paths relative to `notebooks/`. The figures are written to `results/figures/` (PDF, 600 dpi PNG and TIFF) and are identical, pixel for pixel, to those in the manuscript.

## Figure 4 (notebook 13)

`../jupyter_notebook/13-GeneSys_Top_Regulators_Plots_zscore3_20240325.ipynb` is an R notebook (R 4.3 with Seurat, ComplexHeatmap, circlize, circular, ggnewscale, gplots, scater, scattermore, tidyverse, cowplot, patchwork, RColorBrewer). Its inputs are in `../supp_data/notebook13_inputs/`:

* `genesys/`: the centrality tables written by the ten notebooks 12 (`TF_GRN_centrality_t*_zscore3.csv`, `scRNAseq_TF_GRN_centrality_t*_zscore3.csv`), the DE marker tables of the GeneSys-generated and scRNA-seq data (`GeneSys_P_root_markers_ROC.csv`, `Input_X_root_markers_ROC.csv`), the TF list, the gold standards, the gene list and the STRING annotations used to build the gold standard;
* `celloracle/`: the CellOracle gene scores per cell type on GeneSys-generated (`P`) and scRNA-seq (`X`) data.

The notebook sets its working folder to `genesys/` and writes its tables there. The permutation baseline is an unseeded random draw and changes slightly from run to run (about 741 system-wide); every other R50 is reproduced exactly.
