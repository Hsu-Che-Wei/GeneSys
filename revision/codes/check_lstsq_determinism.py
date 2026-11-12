"""Is exp14's raw LIMA reproducible? Same seed, Xylem t1-t3, 5 reps, run twice in-process
and with 1 vs 8 threads; report max |dW| and |z|>3 mask flips."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp14_condition_pipeline as e
from utils import load_lineage
lineage = load_lineage()
X, ct_names, ct_times = e.load_cell_pool_from_pickle()
pools = e.state_indices(ct_names, ct_times, lineage)
slots = e.TRANSITION_SLOTS['t1-t3']
def run(threads, reps=5):
    torch.set_num_threads(threads)
    rng = np.random.RandomState(0)
    return e.lima_for_celltype(X, pools, 'Xylem', slots, reps, 200, rng, None)
def mask(W): return np.abs((W - W.mean()) / W.std()) > 3
A = run(8); B = run(8); C = run(1)
A0 = e.sample_state_rows(X, pools, 'Xylem', slots[0], 200, np.random.RandomState(0))
print('rank of one 200-row batch', np.linalg.matrix_rank(A0), 'unique rows', len(np.unique(A0, axis=0)))
for name, W in [('8 vs 8', B), ('1 vs 8', C)]:
    print(name, 'max|dW|', np.abs(W - A).max(), 'max|W|', np.abs(A).max(),
          'mask flips', (mask(W) != mask(A)).sum(), 'of', mask(A).sum())
