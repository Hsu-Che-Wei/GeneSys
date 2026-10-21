#!/usr/bin/env python3
"""
Experiment 31: TF-network centralities in the exp13 schema from either
  (a) the GRNBoost2 importance matrices of exp30, or
  (b) the LIMA matrices saved by exp14 (--save-ctw) for pseudobulk and scVI,
using exp14's network_centralities, which exp29a verified reproduces the
published TF_GRN_centrality tables exactly (global z-score per cell-type
matrix, |z| > 3, TF-by-TF subnetwork restricted to TFs that are both
regulator and target of a retained edge; entry [i, j] = regulator i at the
earlier state acting on target j at the later state).

Outputs {condition}_{source}_TF_GRN_centrality_{transition}_zscore3.csv into
results/exp14_conditions, ready for exp13/exp32 scoring.

Usage:
  sbatch --job-name=exp31_gb --mem=32G --time=4:00:00 experiments/slurm/run_cpu.sh \
      experiments/exp31_condition_centrality.py --source grnboost --condition genesys
  sbatch --job-name=exp31_pb --mem=48G --time=6:00:00 experiments/slurm/run_cpu.sh \
      experiments/exp31_condition_centrality.py --source lima --condition pseudobulk
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp14_condition_pipeline import (CTS, CT_PREFIX, NAME_PATCHES, DATA_DIR,
                                      TRANSITION_SLOTS, OUT_DATA_DIR, network_centralities)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ZTHRESH = 3.0


def tf_info():
    gene_list = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))
    wanted = pd.read_csv(os.path.join(DATA_DIR, 'Kay_TF_thalemine_annotations.csv'))
    for gid, name in NAME_PATCHES.items():
        wanted.loc[wanted['GeneID'] == gid, 'Name'] = name
    features = gene_list['features'].tolist()
    wanted_ids = set(wanted['GeneID'])
    tf_idx = np.sort(np.array([i for i, g in enumerate(features) if g in wanted_ids]))
    id_to_name = {}
    for _, row in wanted.iterrows():
        id_to_name.setdefault(row['GeneID'], row['Name'])
    tf_names = [id_to_name[features[i]] for i in tf_idx]
    return tf_idx, tf_names


def assemble(per_ct):
    """tf_occurance + concatenation in the notebook-12 column order."""
    tff = []
    for ct in CTS:
        df = per_ct[ct]
        col = f"{CT_PREFIX[ct]}_betweenness_centrality"
        if len(df):
            tff += df.index[df[col] > 0].tolist()
    tf_occ = pd.Series(tff, dtype=object).value_counts().rename('tf_occurance').to_frame()
    order = ['Atrichoblast', 'Trichoblast', 'Lateral Root Cap', 'Cortex', 'Endodermis',
             'Pericycle', 'Procambium', 'Xylem', 'Phloem', 'Columella']
    return pd.concat([tf_occ] + [per_ct[ct] for ct in order], axis=1).fillna(0)


def matrices(source, condition, tr, tf_idx):
    """Yield (cell type, TF-by-TF weight matrix) for one transition."""
    if source == 'grnboost':
        z = np.load(os.path.join(RESULTS_DIR, 'exp30_grnboost',
                                 f'{condition}_{tr}_importance.npz'))
        assert list(z['cell_types']) == CTS
        assert set(z['done']) == set(CTS), 'exp30 run incomplete'
        for ci, ct in enumerate(CTS):
            yield ct, np.asarray(z['importance'][ci], dtype=np.float64)
    else:
        arr = np.load(os.path.join(OUT_DATA_DIR, f'{condition}_ctw_{tr}.npy'), mmap_mode='r')
        for ci, ct in enumerate(CTS):
            W = np.asarray(arr[ci], dtype=np.float32)
            std = W.std()
            mask = np.abs((W - W.mean()) / std) > ZTHRESH if std > 0 else np.zeros_like(W, bool)
            yield ct, (W[np.ix_(tf_idx, tf_idx)].astype(np.float64),
                       mask[np.ix_(tf_idx, tf_idx)])
            del W, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=['grnboost', 'lima'])
    ap.add_argument('--condition', required=True)
    ap.add_argument('--transitions', nargs='*', default=list(TRANSITION_SLOTS))
    args = ap.parse_args()
    tf_idx, tf_names = tf_info()
    tag = 'GRNBoost2_TF_GRN_centrality' if args.source == 'grnboost' else 'TF_GRN_centrality'
    for tr in args.transitions:
        per_ct = {}
        for ct, M in matrices(args.source, args.condition, tr, tf_idx):
            if args.source == 'grnboost':
                # z-score threshold on the full importance matrix, as for LIMA
                std = M.std()
                mask = np.abs((M - M.mean()) / std) > ZTHRESH if std > 0 else np.zeros_like(M, bool)
                W = M
            else:
                W, mask = M
            df = network_centralities(W, mask, np.arange(W.shape[0]), tf_names)
            df.columns = [f"{CT_PREFIX[ct]}_{c}" for c in df.columns]
            per_ct[ct] = df
            print(f'[{args.condition} {tr}] {ct}: {int(mask.sum())} edges, {len(df)} TFs',
                  flush=True)
        out = assemble(per_ct)
        os.makedirs(OUT_DATA_DIR, exist_ok=True)
        path = os.path.join(OUT_DATA_DIR, f'{args.condition}_{tag}_{tr}_zscore3.csv')
        out.to_csv(path, index=True)
        print(f'Saved {path} {out.shape}', flush=True)
    print('[Exp 31] Done!')


if __name__ == '__main__':
    main()
