#!/bin/bash
#SBATCH --job-name=genesys_exp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/logs/%x_%j.log
#SBATCH --error=experiments/logs/%x_%j.err

source /resnick/groups/NolanLab/cwhsu/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

cd /resnick/groups/NolanLab/cwhsu/GeneSys_Revision
python "$@"
