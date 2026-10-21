#!/usr/bin/env python3
"""
Experiment 21b: add the generated stem-cell state (t0) to the GeneSys-generated pool.

Figure 4 (notebook 4) generates t0 with generate_current(x, h, 0) and labels it as the
stem-cell cluster, so the Figure 4 marker tables hold 101 clusters. The exp21 pool held
bins t1 to t10 only. This script appends generated t0 cells (label 'Quiescent Center',
time 0) to the pool so that the DE schemes see the same 101 clusters in every condition.
The existing t1 to t10 cells are left untouched.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_root_data, load_lineage, make_model, load_checkpoint, ROOT_CKPT, RESULTS_DIR
from exp16_lima_identifiability import build_batch

POOL = os.path.join(RESULTS_DIR, "exp14_conditions", "genesys_gen_pool.npz")
N_T0 = 2200


def main():
    z = np.load(POOL, allow_pickle=True)
    X, names, times = z['X'], z['ct_names'], z['ct_times']
    if (names == 'Quiescent Center').any():
        print('pool already holds t0 cells; nothing to do'); return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.RandomState(0); torch.manual_seed(0)
    data = load_root_data(); lineage = load_lineage()
    model = make_model(data['X_test'].shape[1], output_size=lineage.shape[0], device=device)
    model = load_checkpoint(model, ROOT_CKPT, device=device)
    model.eval()
    x, _ = build_batch(data, lineage, N_T0, rng)
    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device)
        t0 = model.generate_current(x_t, model.init_hidden(x_t.shape[0]), 0).cpu().numpy()
    X = np.vstack([X, t0.astype(np.float32)])
    names = np.concatenate([names, np.array(['Quiescent Center'] * len(t0), dtype=object)])
    times = np.concatenate([times, np.zeros(len(t0), dtype=np.float32)])
    np.savez_compressed(POOL, X=X, ct_names=names, ct_times=times)
    print(f'Saved {POOL}: {X.shape}, t0 cells {len(t0)}, t0 mean {t0.mean():.4f}')
    print('[Exp 21b] Done!')


if __name__ == '__main__':
    main()
