#!/usr/bin/env python3
"""
Experiment 25: Random-initialization baseline under the Figure 3 protocol.

Replaces the exp2 control. exp2 scored every validation cell of each terminal
type with no per-type cap, so its evaluation pool carried the natural
composition of the atlas (Lateral Root Cap = 16.7%) and a constant predictor
scored 16.7%. That does not match the published cell type recovery rate,
which is computed on blueprint-assembled test batches with one trajectory per
cell type sampled at equal rates (Root_Dataset in genesys_evaluate_v1.py,
prediction at the terminal time bin t10). Under that protocol every cell type
contributes equally and a constant predictor scores exactly 1/10.

This script evaluates the trained checkpoint and randomly initialized models
with the published protocol:
  * slot 0 = a random quiescent-center cell, slot k (1..10) = a random cell of
    the trajectory's cell type at pseudotime bin k (lineage blueprint);
  * N_PER_TRAJ trajectories per cell type (balanced), 2000 per evaluation;
  * gene masking applied to the whole sequence; prediction read at t10;
  * cell type recovery rate = fraction of correctly predicted terminal fates.

Outputs (experiments/results/exp25_random_baseline_balanced/):
  exp25_recovery.csv            trained vs random, per fraction and repeat
  exp25_seed_scores.csv         10 random seeds, recovery at 0% and 100% genes
  exp25_zero_input_predictions.csv  predicted-class distribution at 0% genes
  exp25_summary.json

Usage:
  sbatch --job-name=exp25_balanced experiments/slurm/run_cpu.sh \
         experiments/exp25_random_baseline_balanced.py
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from utils import (load_root_data, load_lineage, make_model, load_checkpoint,
                   RESULTS_DIR, ROOT_CKPT)

OUT_DIR = os.path.join(RESULTS_DIR, "exp25_random_baseline_balanced")
FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]


class BalancedSampler:
    """Blueprint-assembled trajectories, equal count per cell type."""

    def __init__(self, X, y_labels, lineage, rng):
        self.X = X
        self.rng = rng
        parts = np.array([lab.split('_', 1) for lab in y_labels])
        self.ct = parts[:, 0]
        self.tb = parts[:, 1].astype(float)
        self.lineage = lineage
        self.n_traj = lineage.shape[0]
        self.n_tb = lineage.shape[1]
        self.qc_idx = np.where(self.ct == 'Quiescent Center')[0]
        self.pool = {}
        for j in range(self.n_traj):
            for t in range(1, self.n_tb):
                ct = lineage.iloc[j, t]
                self.pool[(j, t)] = np.where((self.ct == ct) & (self.tb == float(t)))[0]
                if len(self.pool[(j, t)]) == 0:
                    raise RuntimeError(f'no cells for trajectory {j} ({ct}) at bin {t}')

    def batch(self, n_per_traj):
        n = n_per_traj * self.n_traj
        seq = np.zeros((n, self.n_tb, self.X.shape[1]), dtype=np.float32)
        y = np.zeros(n, dtype=int)
        row = 0
        for j in range(self.n_traj):
            for _ in range(n_per_traj):
                seq[row, 0] = self.X[self.rng.choice(self.qc_idx)]
                for t in range(1, self.n_tb):
                    seq[row, t] = self.X[self.rng.choice(self.pool[(j, t)])]
                y[row] = j
                row += 1
        return seq, y


def recovery(model, seq, y, keep_mask, t_pred, batch_size=250):
    """Cell type recovery rate at t_pred with genes outside keep_mask zeroed."""
    preds = []
    with torch.no_grad():
        for s in range(0, len(seq), batch_size):
            x = torch.from_numpy(seq[s:s + batch_size] * keep_mask[None, None, :])
            h = model.init_hidden(x.size(0))
            p, _ = model.predict_proba(x, h, t_pred)
            preds.append(p.argmax(dim=1).numpy())
    preds = np.concatenate(preds)
    return float((preds == y).mean()), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_per_traj', type=int, default=200)
    ap.add_argument('--n_repeats', type=int, default=10)
    ap.add_argument('--n_seeds', type=int, default=10)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_root_data()
    lineage = load_lineage()
    X, y = data['X_test'], data['y_test']
    n_genes = X.shape[1]
    n_traj, n_tb = lineage.shape
    t_pred = n_tb - 1
    terminal = [lineage.iloc[j, n_tb - 1] for j in range(n_traj)]
    print(f'[Exp 25] test cells {X.shape[0]}, genes {n_genes}, '
          f'{n_traj} trajectories x {args.n_per_traj} = {n_traj * args.n_per_traj} '
          f'sequences per evaluation, prediction at t{t_pred}', flush=True)

    rng = np.random.default_rng(0)
    sampler = BalancedSampler(X, y, lineage, rng)

    trained = make_model(n_genes, output_size=n_traj, device='cpu')
    trained = load_checkpoint(trained, ROOT_CKPT, device='cpu').eval()
    torch.manual_seed(0)
    random_model = make_model(n_genes, output_size=n_traj, device='cpu').eval()

    rows, zero_rows = [], []
    for frac in FRACTIONS:
        for rep in range(args.n_repeats):
            seq, yb = sampler.batch(args.n_per_traj)
            keep = np.zeros(n_genes, dtype=np.float32)
            if frac >= 1.0:
                keep[:] = 1.0
            elif frac > 0.0:
                keep[rng.choice(n_genes, int(frac * n_genes), replace=False)] = 1.0
            for name, model in (('trained', trained), ('random', random_model)):
                acc, preds = recovery(model, seq, yb, keep, t_pred)
                rows.append({'model': name, 'fraction': frac,
                             'n_genes_kept': int(keep.sum()), 'repeat': rep,
                             'accuracy': acc})
                print(f'  {name:8s} frac={frac:.2f} rep={rep} acc={acc:.4f}', flush=True)
                if frac == 0.0:
                    counts = np.bincount(preds, minlength=n_traj)
                    for j in range(n_traj):
                        zero_rows.append({'model': name, 'repeat': rep,
                                          'predicted_cell_type': terminal[j],
                                          'fraction_of_predictions': counts[j] / len(preds)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'exp25_recovery.csv'), index=False)
    pd.DataFrame(zero_rows).to_csv(
        os.path.join(OUT_DIR, 'exp25_zero_input_predictions.csv'), index=False)

    # Ten independent initializations, scored at 0% and 100% genes
    seed_rows = []
    seq, yb = sampler.batch(args.n_per_traj)
    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        m = make_model(n_genes, output_size=n_traj, device='cpu').eval()
        for frac in (0.0, 1.0):
            keep = np.full(n_genes, frac, dtype=np.float32)
            acc, preds = recovery(m, seq, yb, keep, t_pred)
            counts = np.bincount(preds, minlength=n_traj)
            seed_rows.append({'seed': seed, 'fraction': frac, 'accuracy': acc,
                              'dominant_class': terminal[int(counts.argmax())],
                              'dominant_share': counts.max() / len(preds)})
            print(f'  seed {seed} frac={frac:.0f} acc={acc:.4f} '
                  f'dominant={terminal[int(counts.argmax())]}', flush=True)
        del m
    sdf = pd.DataFrame(seed_rows)
    sdf.to_csv(os.path.join(OUT_DIR, 'exp25_seed_scores.csv'), index=False)

    summ = df.groupby(['model', 'fraction'])['accuracy'].agg(['mean', 'std']).reset_index()
    z = pd.DataFrame(zero_rows)
    ztr = z[z['model'] == 'trained'].groupby('predicted_cell_type')['fraction_of_predictions'].mean()
    zrd = z[z['model'] == 'random'].groupby('predicted_cell_type')['fraction_of_predictions'].mean()
    summary = {
        'protocol': 'blueprint-balanced test batches, prediction at t10, '
                    f'{args.n_per_traj} trajectories per cell type',
        'chance_level': 1.0 / n_traj,
        'trained_all_masked': float(summ[(summ.model == 'trained') & (summ.fraction == 0)]['mean'].iloc[0]),
        'random_all_masked': float(summ[(summ.model == 'random') & (summ.fraction == 0)]['mean'].iloc[0]),
        'trained_10pct': float(summ[(summ.model == 'trained') & (summ.fraction == 0.1)]['mean'].iloc[0]),
        'random_mean_over_fractions': float(df[df.model == 'random']['accuracy'].mean()),
        'random_max_over_fractions': float(df[df.model == 'random']['accuracy'].max()),
        'trained_zero_input_dominant_class': str(ztr.idxmax()),
        'trained_zero_input_dominant_share': float(ztr.max()),
        'random_zero_input_dominant_class': str(zrd.idxmax()),
        'random_zero_input_dominant_share': float(zrd.max()),
        'seed_dominant_classes_all_masked': sdf[sdf.fraction == 0].set_index('seed')['dominant_class'].to_dict(),
        'seed_accuracy_all_masked': sdf[sdf.fraction == 0].set_index('seed')['accuracy'].to_dict(),
        'seed_accuracy_all_genes': sdf[sdf.fraction == 1].set_index('seed')['accuracy'].to_dict(),
    }
    with open(os.path.join(OUT_DIR, 'exp25_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(summ.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print('[Exp 25] Done!')


if __name__ == '__main__':
    main()
