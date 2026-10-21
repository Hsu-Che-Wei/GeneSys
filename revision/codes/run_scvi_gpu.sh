#!/bin/bash
#SBATCH --job-name=scvi_exp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/logs/%x_%j.log
#SBATCH --error=experiments/logs/%x_%j.err

source /resnick/groups/NolanLab/cwhsu/miniconda3/etc/profile.d/conda.sh
conda activate scvi
export PYTHONUNBUFFERED=1

cd /resnick/groups/NolanLab/cwhsu/GeneSys_Revision
python "$@"
