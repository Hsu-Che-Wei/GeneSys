#!/bin/bash
#SBATCH --job-name=post_rc
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/logs/%x_%j.log
#SBATCH --error=experiments/logs/%x_%j.err
# Score the models trained through genesys_cli.run_cycle: autoregressive generation,
# recreation accuracy, external classifiers, extrapolation, width sweep summary, figures.
source /resnick/groups/NolanLab/cwhsu/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1
cd /resnick/groups/NolanLab/cwhsu/GeneSys_Revision
set -e
for spec in "exp1 correct" "exp1 reversed" "exp1 shuffled" "exp1 random" \
            "exp6 full" "exp6 no_tdvae" "exp6 no_lstm" "exp6 no_blueprint" \
            "exp4 0.0" "exp4 0.1" "exp4 0.2" "exp4 0.4"; do
  set -- $spec
  python experiments/eval_recreation_accuracy.py --experiment $1 --condition $2 --device cpu
  python experiments/eval_recreation_accuracy_v2.py --experiment $1 --condition $2 --device cpu
done
python experiments/exp12_external_scoring.py --device cpu
python experiments/exp7_extrapolation.py --device cpu
python experiments/exp10_latent_sweep.py --aggregate
python experiments/generate_sprint3_figures.py
echo "[post_rc] Done!"
