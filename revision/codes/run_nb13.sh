#!/bin/bash
#SBATCH --job-name=nb13_de
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/resnick/groups/NolanLab/cwhsu/GeneSys_Revision/experiments/logs/%x_%j.log
#SBATCH --error=/resnick/groups/NolanLab/cwhsu/GeneSys_Revision/experiments/logs/%x_%j.err
# Execute notebook 13 (R kernel of the rverify environment) and refresh its HTML copy.
source /resnick/groups/NolanLab/cwhsu/miniconda3/etc/profile.d/conda.sh
conda activate rverify
cd /resnick/groups/NolanLab/cwhsu/GeneSys_Revision
EXP=/resnick/groups/NolanLab/cwhsu/GeneSys_Revision/experiments
export JUPYTER_PATH=/resnick/groups/NolanLab/cwhsu/miniconda3/envs/rverify/share/jupyter:$EXP/.nbtools/share/jupyter
export PYTHONPATH=$EXP/.nbtools
PY=/resnick/groups/NolanLab/cwhsu/miniconda3/bin/python
NB=$EXP/GeneSys/jupyter_notebook/13-GeneSys_Top_Regulators_Plots_zscore3_20240325.ipynb
set -e
$PY -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=ir "$NB"
$PY -m nbconvert --to html "$NB"
echo "[nb13] done"
