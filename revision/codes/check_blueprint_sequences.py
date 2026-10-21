#!/usr/bin/env python3
"""Check that the up-to-date blueprint class, fed labels in its own convention (time from 1),
builds the same sequences as the Root_Dataset of the original notebooks: slot 0 a stem cell,
slot k a cell of the trajectory's type at time bin k; and that the reversed decoy is reversed."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GeneSys", "genesys"))
from utils import load_root_data, load_root_data_blueprint, reverse_time_labels, load_lineage
from genesys_model import Cell_Lineage_Blueprint
from exp1_decoy_blueprint import make_reversed_lineage, make_random_lineage, make_shuffled_lineage

raw = load_root_data(); y0 = raw['y_val']; X = raw['X_val']
data = load_root_data_blueprint(); y1 = data['y_val']
lineage = load_lineage()
# tag every cell with its own row index in feature 0 so that the sampled cell can be traced
tag = np.zeros((len(y0), 2), dtype=np.float32); tag[:, 0] = np.arange(len(y0)) + 1
def trace(ds, n=300):
    ok = True; rows = []
    for _ in range(n):
        s = ds[0]; ids = s['x'][:, 0].astype(int) - 1
        rows.append([y0[i] if i >= 0 else 'ZERO' for i in ids])
    return rows
for name, lin, lab in [('correct', lineage, y1), ('old usage (labels not shifted)', lineage, y0),
                       ('reversed', make_reversed_lineage(lineage), reverse_time_labels(y1)),
                       ('shuffled', make_shuffled_lineage(lineage), y1),
                       ('random', make_random_lineage(lineage), y1)]:
    rows = trace(Cell_Lineage_Blueprint(tag, lab, lin))
    bins = [sorted(set(r[k].split('_')[1] if r[k] != 'ZERO' else 'ZERO' for r in rows)) for k in range(11)]
    types0 = sorted(set(r[0].split('_')[0] for r in rows))
    same_type = np.mean([len(set(x.split('_')[0] for x in r[1:] if x != 'ZERO')) == 1 for r in rows])
    print(f"[{name}] time bin of the pickle read at each slot: {bins}")
    print(f"    slot 0 cell types: {types0}; fraction of sequences with one cell type over slots 1-10: {same_type:.2f}")
print('[check] Done!')
