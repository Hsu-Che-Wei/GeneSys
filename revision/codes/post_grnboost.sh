#!/bin/bash
# Score the GRNBoost2 centrality tables of exp31 (GeneSys-generated data) with the
# Figure 4 R50 code and rebuild the benchmark table and figures.
source /resnick/groups/NolanLab/cwhsu/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1
cd /resnick/groups/NolanLab/cwhsu/GeneSys_Revision
set -e
python experiments/exp13_fig4_r50.py --label grnboost_genesys \
    --lima-prefix genesys_GRNBoost2_TF_GRN_centrality \
    --lima-dir experiments/results/exp14_conditions
python experiments/exp20_fig4_expanded.py
python experiments/exp24_manuscript_figures.py
echo "[post_grnboost] Done!"
