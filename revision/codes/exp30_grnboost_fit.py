#!/usr/bin/env python3
"""
Experiment 30: GRNBoost2 regression per cell type and transition, mirroring
the LIMA arm (one importance matrix per cell type and transition instead of
one least-squares matrix), on the states of exp29.

For each cell type, every TF at the later state is regressed on all other
TFs at the earlier state with arboreto's GRNBoost2 regression (the
GradientBoostingRegressor settings and early-stopping monitor of arboreto's
core module, orchestrated with joblib). The regulator-by-target importance
matrix (1,495 x 1,495) is stored; exp31 turns it into centralities with the
same z-score threshold and network code as LIMA.

Usage (one job per condition and transition; runs in the arboreto env):
  sbatch --job-name=gb_genesys_t1-t3 --cpus-per-task=32 --mem=64G \
      experiments/slurm/run_arboreto_cpu.sh experiments/exp30_grnboost_fit.py \
      --condition genesys --transition t1-t3 --n-jobs 32
"""
import os, sys, argparse, time
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
IN_DIR = os.path.join(RESULTS_DIR, "exp29_grnboost_states")
OUT_DIR = os.path.join(RESULTS_DIR, "exp30_grnboost")
CTS = ['Atrichoblast', 'Trichoblast', 'Cortex', 'Endodermis', 'Pericycle',
       'Procambium', 'Xylem', 'Phloem', 'Lateral Root Cap', 'Columella']


def fit_target(X, y, j, SGBM_KWARGS, EarlyStopMonitor, window):
    from sklearn.ensemble import GradientBoostingRegressor
    n = X.shape[1]
    imp = np.zeros(n, dtype=np.float32)
    if y.std() == 0:
        return imp
    features = np.delete(np.arange(n), j)          # a TF does not regulate itself
    model = GradientBoostingRegressor(random_state=666, **SGBM_KWARGS)
    model.fit(X[:, features], y, monitor=EarlyStopMonitor(window))
    imp[features] = model.feature_importances_
    return imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', required=True)
    ap.add_argument('--transition', required=True)
    ap.add_argument('--n-jobs', type=int, default=16)
    args = ap.parse_args()
    from arboreto.core import SGBM_KWARGS, EarlyStopMonitor, EARLY_STOP_WINDOW_LENGTH
    from joblib import Parallel, delayed

    z = np.load(os.path.join(IN_DIR, f'{args.condition}_{args.transition}.npz'))
    A, B, labels = z['A'], z['B'], z['labels'].astype(str)
    n_tf = A.shape[1]
    print(f'{args.condition} {args.transition}: rows {A.shape[0]}, TFs {n_tf}, '
          f'SGBM_KWARGS={SGBM_KWARGS}', flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{args.condition}_{args.transition}_importance.npz')
    imp_all = np.zeros((len(CTS), n_tf, n_tf), dtype=np.float32)
    for ci, ct in enumerate(CTS):
        rows = np.where(labels == ct)[0]
        X, Y = A[rows].astype(np.float32), B[rows].astype(np.float32)
        t0 = time.time()
        res = Parallel(n_jobs=args.n_jobs, batch_size=8)(
            delayed(fit_target)(X, Y[:, j], j, SGBM_KWARGS, EarlyStopMonitor,
                                EARLY_STOP_WINDOW_LENGTH) for j in range(n_tf))
        imp_all[ci] = np.stack(res, axis=1)       # [regulator, target]
        print(f'  {ct}: {len(rows)} rows, {time.time() - t0:.0f} s', flush=True)
        np.savez_compressed(out, importance=imp_all, cell_types=np.array(CTS),
                            tf_ids=z['tf_ids'], done=np.array(CTS[:ci + 1]))
    print(f'Saved {out}')
    print('[Exp 30] Done!')


if __name__ == '__main__':
    main()
