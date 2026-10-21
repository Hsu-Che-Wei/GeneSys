#!/bin/bash
#SBATCH --job-name=post_l55
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=experiments/logs/%x_%j.log
#SBATCH --error=experiments/logs/%x_%j.err
# Rescore the pseudobulk and scVI LIMA tables recomputed with 55 batches (as the scRNA-seq
# condition of Figure 4), rebuild the benchmark table and the genome-wide recall.
source /resnick/groups/NolanLab/cwhsu/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1
cd /resnick/groups/NolanLab/cwhsu/GeneSys_Revision
set -e
for c in pseudobulk scvi; do
  python experiments/exp13_fig4_r50.py --label $c --lima-prefix ${c}_TF_GRN_centrality \
      --lima-dir experiments/results/exp14_conditions
done
python experiments/exp20_fig4_expanded.py
python experiments/exp18_genomewide_recall.py
echo "[post_l55] Done!"
