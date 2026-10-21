#!/usr/bin/env python3
"""
Experiment 29: paired state matrices for GRNBoost2, one file per data
condition and transition, built with the SAME sampling as the LIMA arm.

GeneSys-generated (GPU): 11 batches of 2,000 blueprint sequences from the
test split, states produced by the trained model exactly as in notebook 12
(t0 = generate_current at slot 0; t_k = generate_next at slot k-1), rows
labelled by the terminal cell type of their trajectory.
raw / pseudobulk / scVI (CPU): 11 batches of 200 rows per cell type from the
state pools of exp14 (pseudobulk rows = mean of 50 cells).

Only the 1,495 TF columns are stored, because the GRNBoost2 arm, like the
published LIMA centrality, is computed on the TF-by-TF subnetwork.

Usage:
  sbatch --job-name=exp29_genesys experiments/slurm/run_gpu.sh \
      experiments/exp29_grnboost_states.py --condition genesys
  sbatch --job-name=exp29_raw experiments/slurm/run_cpu.sh \
      experiments/exp29_grnboost_states.py --condition raw
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GeneSys", "genesys"))
from exp14_condition_pipeline import (TRANSITION_SLOTS, CTS, NAME_PATCHES, DATA_DIR,
                                      load_cell_pool_from_pickle, load_cell_pool_from_scvi,
                                      state_indices, sample_state_rows, OUT_DATA_DIR)
from utils import load_root_data, load_lineage, RESULTS_DIR

OUT_DIR = os.path.join(RESULTS_DIR, "exp29_grnboost_states")
N_REPS, ROWS_PER_REP, BATCH = 11, 200, 2000
# notebook 12: (generator for the from-state, slot), (generator for the to-state, slot)
GEN_SLOTS = {"t0-t1": (('current', 0), ('next', 0)), "t1-t3": (('next', 0), ('next', 2)),
             "t3-t5": (('next', 2), ('next', 4)), "t5-t7": (('next', 4), ('next', 6)),
             "t7-t9": (('next', 6), ('next', 8))}


def tf_columns():
    gene_list = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))
    wanted = pd.read_csv(os.path.join(DATA_DIR, 'Kay_TF_thalemine_annotations.csv'))
    for gid, name in NAME_PATCHES.items():
        wanted.loc[wanted['GeneID'] == gid, 'Name'] = name
    features = gene_list['features'].tolist()
    wanted_ids = set(wanted['GeneID'])
    tf_idx = np.sort(np.array([i for i, g in enumerate(features) if g in wanted_ids]))
    return tf_idx, np.array([features[i] for i in tf_idx])


def genesys_states(seed):
    import torch
    from utils import make_model, load_checkpoint, ROOT_CKPT
    from exp16_lima_identifiability import build_batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    data = load_root_data()
    lineage = load_lineage()
    n_genes = data['X_test'].shape[1]
    model = make_model(n_genes, output_size=lineage.shape[0], device=device)
    model = load_checkpoint(model, ROOT_CKPT, device=device)
    model.eval()
    tf_idx, tf_ids = tf_columns()
    for tr, ((gfrom, sfrom), (gto, sto)) in GEN_SLOTS.items():
        A, B, lab = [], [], []
        for rep in range(N_REPS):
            x, y_label = build_batch(data, lineage, BATCH, rng)
            x_t = torch.from_numpy(x).to(device)
            with torch.no_grad():
                h = model.init_hidden(BATCH)
                tfrom = (model.generate_current if gfrom == 'current' else model.generate_next)(x_t, h, sfrom)
                h = model.init_hidden(BATCH)
                tto = model.generate_next(x_t, h, sto)
            A.append(tfrom.cpu().numpy()[:, tf_idx].astype(np.float32))
            B.append(tto.cpu().numpy()[:, tf_idx].astype(np.float32))
            lab.append(y_label)
        save(tr, 'genesys', np.vstack(A), np.vstack(B), np.concatenate(lab), tf_ids)


def pool_states(condition, seed):
    lineage = load_lineage()
    if condition in ('raw', 'pseudobulk'):
        X, ct_names, ct_times = load_cell_pool_from_pickle()
        boot = 50 if condition == 'pseudobulk' else None
    else:
        X, ct_names, ct_times = load_cell_pool_from_scvi(
            os.path.join(OUT_DATA_DIR, 'scvi_denoised.npz'))
        boot = None
    pools = state_indices(ct_names, ct_times, lineage)
    rng = np.random.RandomState(seed)
    tf_idx, tf_ids = tf_columns()
    for tr, slots in TRANSITION_SLOTS.items():
        A, B, lab = [], [], []
        for ct in CTS:
            for rep in range(N_REPS):
                a = sample_state_rows(X, pools, ct, slots[0], ROWS_PER_REP, rng, boot)
                b = sample_state_rows(X, pools, ct, slots[1], ROWS_PER_REP, rng, boot)
                A.append(a[:, tf_idx]); B.append(b[:, tf_idx]); lab += [ct] * ROWS_PER_REP
        save(tr, condition, np.vstack(A), np.vstack(B), np.array(lab), tf_ids)


def save(tr, condition, A, B, labels, tf_ids):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{condition}_{tr}.npz')
    np.savez_compressed(out, A=A, B=B, labels=labels.astype(str), tf_ids=tf_ids)
    print(f'Saved {out}: A={A.shape} B={B.shape} '
          f'rows per cell type={pd.Series(labels).value_counts().to_dict()}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', required=True, choices=['genesys', 'raw', 'pseudobulk', 'scvi'])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.condition == 'genesys':
        genesys_states(args.seed)
    else:
        pool_states(args.condition, args.seed)
    print('[Exp 29] Done!')


if __name__ == '__main__':
    main()
