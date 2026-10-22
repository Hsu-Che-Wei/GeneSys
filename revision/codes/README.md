# Scripts of the revision

These scripts produced the results that the revision notebooks R1 to R4 read from `../results/`. They were run as cluster jobs. They expect to sit in a folder that also holds a clone of this repository under `GeneSys/` (they import the model and the training function from `GeneSys/genesys`), the root atlas pickle and the published checkpoint; the paths are set in `utils.py`.

## Training

Every retrained model is trained by `run_cycle` of `genesys/genesys_cli.py`, called unchanged through `train_run_cycle.py`: one cycle of 30 epochs with the package defaults. What an experiment varies enters through the inputs of `run_cycle` only (labels, lineage table, model object). `utils.load_root_data_blueprint` supplies the labels in the convention of the package (time from 1). `check_blueprint_sequences.py` traces which cells each position of a sequence reads, and `smoke_run_cycle.py` is a two-epoch test of every training variant on synthetic data.

| Script | Role | Experiment in the Methods | Figure | Output files (in `../results/`) | Notebook |
|---|---|---|---|---|---|
| `train_run_cycle.py` | calls `genesys_cli.run_cycle`; loaders, model variants, KL logging wrappers | all retrained models | | | R1, R2 |
| `utils.py` | data and checkpoint paths, label conventions, `load_root_data_blueprint`, `reverse_time_labels` | | | | |
| `check_blueprint_sequences.py` | check of the sequences built by the blueprint class for each lineage table | 1 | | job log | R1 |
| `smoke_run_cycle.py` | two-epoch test of the four training variants | | | job log | |
| `rescue_train_and_eval.py` | trains one model (decoy, ablation or noise level) through `train_run_cycle.py` and writes its classification accuracy | 1, 3, 9 | S6, S14A-B | `exp1_decoy/`, `exp6_ablation/`, `exp4_noise/`: `*_eval_result.json` | R1, R2 |
| `exp1_decoy_blueprint.py` | lineage tables of the decoys (reversed, shuffled, random) | 1 | S6A-D | | R1 |
| `exp6_ablation.py` | model class of the No LSTM ablation | 3 | S6E-F | | R1 |
| `exp4_noise_sensitivity.py` | earlier driver of the noise experiment; no longer used for training | 9 | | | |
| `train_utils.py` | `evaluate_trained_model` (classification accuracy), used by `exp10_latent_sweep.py` | 10 | S14C-E | | R2 |
| `eval_recreation_accuracy_v2.py` | recreation accuracy (protocol of Figure S5) | 1, 3, 9 | S6, S14A-B | `*_recreation_acc_v2.json` | R1, R2 |
| `eval_recreation_accuracy.py` | autoregressive generation accuracy | 1, 3 | S6A, S6E-F | `*_recreation_acc.json` | R1 |
| `exp12_external_scoring.py` | external classifiers on generated transcriptomes | 2 | S6C, S6D | `exp12_external/*.csv` | R1 |
| `exp25_random_baseline_balanced.py` | random-initialization control | 4 | S7A, S7B | `exp25_random_baseline_balanced/*` | R2 |
| `exp11_trichoblast_forensics.py` | belief state under fully masked input | 4 | S7C | `exp11_trichoblast/*` | R2 |
| `exp7_extrapolation.py` | extrapolation with zero-padding (published model) | 8 | S5E-F | `exp7_extrapolation.csv` | R2 |
| `exp10_latent_sweep.py` | model width | 10 | S14C-E | `exp10_latent_sweep_summary.csv`, `exp10_latent_sweep_kl_logs.csv`, `exp10_latent_sweep/` | R2 |
| `exp3_kl_divergence_from_scratch.py` | KL-divergence monitoring | 11 | S14F-I | `exp3_kl_divergence_diagnostic.csv` | R2 |
| `exp16_lima_identifiability.py` | LIMA under the pipeline solver, the minimum-norm solution and ridge regression, at three sample sizes and three sampling spreads (one CPU job per condition with `--only`, then `--merge`) | 12 | S15A-C | `exp16_identifiability/exp16_conditions.csv`, `exp16_ranking_agreement.csv`, `exp16_ranks.csv` | R4 |
| `exp16b_solver_check.py` | singular spectrum and effective rank of the pipeline solver for one cell type | 12 | | `exp16_identifiability/exp16b_solver_check.json` | |
| `exp5_zscore_sensitivity.py`, `exp9_zscore_tf_recovery.py` | z-score threshold (matrix-wide z-score as in notebook 12): edge counts, gold-standard recall | 13 | S15D-F | `exp5_zscore_*.csv`, `exp9_zscore_*.csv` | R4 |
| `exp15_scvi_prepare.py` | scVI condition (training and denoised matrix) | benchmark scenarios | 5 | `prioritization/scvi_elbo_train.csv` | R3 |
| `exp14_condition_pipeline.py` | LIMA and centralities (`--mode lima`, port of notebook 12, `--n-reps 55`) and DE markers (`--mode de --seurat`) for a data condition | benchmark scenarios | 5 | `prioritization/markers/*_markers_ROC_fig4.csv` | R3 |
| `exp35_seurat_roc_markers.py`, `exp36_export_fig4_seurat.R` | port of the Figure 4 marker call and its validation on the cells of the Figure 4 Seurat objects | benchmark scenarios | 4, 5 | `prioritization/exp35_seurat_port_validation_objects.json` | R3 |
| `exp13_fig4_r50.py` | R50 scoring of centrality tables, port of the scoring of notebook 13 (DE strongest marker first; `--de-descending` gives the order used before the correction); permutation baseline | benchmark scenarios | 4, 5, S11A | `prioritization/exp13_*.csv` | R3 |
| `exp20_fig4_expanded.py` | benchmark table; R50 of the DE criteria, strongest marker first | benchmark scenarios | 5, S11A | `prioritization/exp20_*.csv` | R3 |
| `exp43_de_statistics.py` | what the DE statistics measure on GeneSys-generated and scRNA-seq data: detection, AUC, average difference, gold-standard TFs among the markers, separation of gold-standard TFs by each ordering | DE Methods | 4 | `prioritization/exp43_de_statistics.json` | R3 |
| `run_nb13.sh` | SLURM wrapper that executes notebook 13 (R kernel) | | 4 | `../supp_data/notebook13_outputs/` | |
| `regenerate_notebook12.py` | re-runs the network cells of a notebook 12 from its saved LIMA with the directed edge drawn from regulator to target (out-degree counts the targets of a TF, in-degree its regulators) and writes the centrality table | benchmark scenarios | 4 to 6, S15 | tables of notebook 12 | |
| `check_edge_direction.py` | checks on the LIMA and the centrality table of one cell type that out-degree equals the row count of W and in-degree the column count | benchmark scenarios | 4, 5 | `prioritization/check_edge_direction.json`, `check_edge_direction_before_fix.json` | R3 |
| `exchange_degree_columns.py` | applies the same correction to the centrality tables of the pseudobulk, scVI, scRNA-seq reproduction and GRNBoost2 runs | benchmark scenarios, 6 | 5, S11A | | |
| `exp38_illustrator_degree_labels.py` | writes the Illustrator files of Figures 4, 6 and S15 with the out-degree and in-degree labels of the LIMA centralities exchanged | | 4, 6, S12 | | |
| `exp21_genesys_pool.py`, `exp21b_pool_add_t0.py` | pool of GeneSys-generated cells (not used for the reported DE values, which are the Figure 4 tables) | | | | |
| `exp29_check_centrality_port.py` | check that the centrality code reproduces the published tables | | | | |
| `exp29_grnboost_states.py`, `exp30_grnboost_fit.py`, `exp31_condition_centrality.py` | GRNBoost2 in place of the least-squares fit | 6 | S11A | `prioritization/exp13_r50_grnboost_genesys.csv` | R3 |
| `exp17_knockout_ranking.py` | in silico knockout (`--protocol seeded`, reported; `--protocol rollout`, for comparison) | 5 | S11A, S11B | `prioritization/exp17_knockout/`, `prioritization/exp17_knockout_rollout/` | R3 |
| `exp26_gene_expression_stats.py`, `exp27_knockout_audit.py` | expression-level audit of the knockout ranking | 7 | | `prioritization/exp26_gene_expression_stats.csv`, `prioritization/exp27_knockout_audit.json` | R3 |
| `exp18_genomewide_recall.py` | genome-wide recall of the gold standard | 7 | S11B to S11D | `prioritization/exp18_genomewide_recall.csv` | R3 |
| `generate_sprint3_figures.py`, `exp24_manuscript_figures.py`, `exp41_supplementary_composites.py` | figures (exp24 draws Figure 4 from the outputs of notebook 13; exp41 joins new panels to Illustrator panels for Figures S5 and S10) | | 4, 5, S5 to S7, S10, S11, S14, S15 | `figures/*.png` | R1 to R4 |
| `build_revision_notebooks_R1_R2_R4.py`, `build_revision_notebook_R3.py` | write the notebooks (then executed with nbconvert) | | | `../notebooks/` | |
| `run_cpu.sh`, `run_gpu.sh`, `run_gpu_fast.sh`, `run_arboreto_cpu.sh`, `run_scvi_gpu.sh`, `run_r.sh`, `post_rc.sh`, `post_l55.sh`, `post_grnboost.sh` | SLURM wrappers; `post_rc.sh` scores the retrained models, `post_l55.sh` rescoring after the 55-batch LIMA runs | | | | |

Large intermediate files (checkpoints, LIMA arrays, the scVI-denoised matrix, the pool of generated cells) are not stored in this folder.
