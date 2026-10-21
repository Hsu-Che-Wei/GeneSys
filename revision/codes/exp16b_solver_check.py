#!/usr/bin/env python3
"""Diagnostic for Figure S21: why does the ridge solution at the smallest penalty agree
least with the unregularized (lstsq) solution?  One cell type, the first three batches of
Experiment 12 (same random stream), identical A and B for every solver.

Reports the singular spectrum of A, the effective rank used by torch.linalg.lstsq, and for
each solver the agreement with the lstsq solution on the TF-by-TF block: Pearson correlation
of the coefficients, Jaccard overlap of the |z| > 3 edges, Spearman of TF degree."""
import os, sys, json
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GeneSys", "genesys"))
from utils import load_root_data, load_lineage, make_model, load_checkpoint, RESULTS_DIR, ROOT_CKPT
import exp16_lima_identifiability as e16
from exp29_grnboost_states import tf_columns

torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
dev = 'cpu'
data = load_root_data(); lineage = load_lineage()
model = load_checkpoint(make_model(data['X_test'].shape[1], output_size=lineage.shape[0], device=dev), ROOT_CKPT, device=dev)
model.eval()
tf_idx, _ = tf_columns(); tf_idx = np.asarray(tf_idx)
rng = np.random.RandomState(0); torch.manual_seed(0)
CT = 'Atrichoblast'; NREP = 3
LAMS = [1e-6, 1e-4, 0.01, 0.1, 1.0, 10.0]
out = {'spectrum': [], 'agreement': []}
Wsum = {}
for rep in range(NREP):
    x, y = e16.build_batch(data, lineage, 2200, rng)
    xt = torch.from_numpy(x)
    A = e16.generate_states(model, xt, e16.FROM_SLOT, 1.0, dev)
    B = e16.generate_states(model, xt, e16.TO_SLOT, 1.0, dev)
    sel = np.where(y == CT)[0][:200]
    A = A[sel].contiguous(); B = B[sel].contiguous()
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    eps = torch.finfo(A.dtype).eps
    rcond = eps * max(A.shape)
    res = torch.linalg.lstsq(A, B)                      # the pipeline call (driver gelsy on CPU)
    rank_gelsy = int(res.rank) if res.rank.numel() else None
    res_d = torch.linalg.lstsq(A, B, driver='gelsd')    # SVD-based driver, same default rcond
    spec = {'rep': rep, 'dtype': str(A.dtype), 'rows': int(A.shape[0]),
            'S_max': float(S.max()), 'S_min': float(S.min()),
            'S_quantiles': [float(q) for q in torch.quantile(S, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))],
            'default_rcond': float(rcond), 'cutoff_abs': float(rcond * S.max()),
            'n_S_below_cutoff': int((S < rcond * S.max()).sum()),
            'rank_gelsy': rank_gelsy, 'rank_gelsd': int(res_d.rank) if res_d.rank.numel() else None,
            'n_S_below_sqrt_lam': {str(l): int((S < l ** 0.5).sum()) for l in LAMS}}
    out['spectrum'].append(spec); print(json.dumps(spec), flush=True)
    UB = U.T @ B
    sols = {'lstsq_gelsy': res.solution, 'lstsq_gelsd': res_d.solution,
            'pinv_all_directions': Vh.T @ ((1.0 / S)[:, None] * UB),
            'pinv_truncated_at_rcond': Vh.T @ (torch.where(S >= rcond * S.max(), 1.0 / S, torch.zeros_like(S))[:, None] * UB)}
    for l in LAMS:
        sols[f'ridge_{l}'] = Vh.T @ ((S / (S ** 2 + l))[:, None] * UB)
    for k, W in sols.items():
        Wsum[k] = W.clone() if k not in Wsum else Wsum[k] + W
    del sols, res, res_d

def tf_block_stats(W):
    W = W.numpy()
    z = (W - W.mean()) / W.std()
    sub = W[np.ix_(tf_idx, tf_idx)]; m = np.abs(z[np.ix_(tf_idx, tf_idx)]) > 3
    deg = m.sum(0) + m.sum(1)
    return sub, m, deg, float(np.linalg.norm(W))

base = tf_block_stats(Wsum['lstsq_gelsy'] / NREP)
for k in Wsum:
    sub, m, deg, nrm = tf_block_stats(Wsum[k] / NREP)
    r = float(np.corrcoef(base[0].ravel(), sub.ravel())[0, 1])
    jac = float((m & base[1]).sum() / max(1, (m | base[1]).sum()))
    rho = float(spearmanr(base[2], deg)[0])
    row = {'solver': k, 'pearson_tf_block': r, 'jaccard_edges_z3': jac, 'spearman_tf_degree': rho,
           'edges_tf_block': int(m.sum()), 'frobenius_norm_W': nrm}
    out['agreement'].append(row); print(json.dumps(row), flush=True)
os.makedirs(os.path.join(RESULTS_DIR, 'exp16_identifiability'), exist_ok=True)
json.dump(out, open(os.path.join(RESULTS_DIR, 'exp16_identifiability', 'exp16b_solver_check.json'), 'w'), indent=1)
print('done')
