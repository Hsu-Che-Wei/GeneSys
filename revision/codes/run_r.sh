#!/bin/bash
#SBATCH --job-name=genesys_r
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/logs/%x_%j.log
#SBATCH --error=experiments/logs/%x_%j.err
source /resnick/groups/NolanLab/cwhsu/miniconda3/etc/profile.d/conda.sh
conda activate rverify
cd /resnick/groups/NolanLab/cwhsu/GeneSys_Revision
Rscript "$@"
