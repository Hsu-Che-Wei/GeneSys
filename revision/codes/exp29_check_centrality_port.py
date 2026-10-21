#!/usr/bin/env python3
"""
Experiment 29a: verify the centrality port against the published tables.

Loads one cell-type slice of the published GeneSys LIMA array
(genesys_ctw_t1-t3.npy, slice 0 = Atrichoblast), applies the notebook-12
z-score threshold, computes the six centralities with exp14's
network_centralities in both edge orientations, and compares them with the
atri_* columns of TF_GRN_centrality_t1-t3_zscore3.csv. Decides which
orientation reproduces the published in- and out-degree centralities.

Usage:
  sbatch --job-name=exp29a --mem=32G --time=2:00:00 experiments/slurm/run_cpu.sh \
      experiments/exp29_check_centrality_port.py
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp14_condition_pipeline import (network_centralities, NAME_PATCHES, DATA_DIR,
                                      CTS, CT_PREFIX)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
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

    pub = pd.read_csv(os.path.join(DATA_DIR, 'TF_GRN_centrality_t1-t3_zscore3.csv'), index_col=0)
    arr = np.load(os.path.join(DATA_DIR, 'genesys_ctw_t1-t3.npy'), mmap_mode='r')
    print('published table', pub.shape, 'array', arr.shape, flush=True)

    report = {}
    for ci, ct in enumerate(CTS[:2]):          # Atrichoblast, Trichoblast
        W = np.asarray(arr[ci], dtype=np.float32)
        z = (W - W.mean()) / W.std()
        mask = np.abs(z) > 3.0
        prefix = CT_PREFIX[ct]
        for orient, adj in [('as_ported (W)', W), ('transposed (W.T)', W.T)]:
            df = network_centralities(adj, mask if orient.startswith('as') else mask.T,
                                      tf_idx, tf_names)
            common = df.index.intersection(pub.index)
            res = {'n_nodes': int(len(df)), 'n_common_with_published': int(len(common))}
            for col in ['degree_centrality', 'out_centrality', 'in_centrality',
                        'betweenness_centrality', 'eigenvector_centrality']:
                a = df.loc[common, col].values
                b = pub.loc[common, f'{prefix}_{col}'].values
                res[col] = {'max_abs_diff': float(np.max(np.abs(a - b))),
                            'spearman': float(pd.Series(a).corr(pd.Series(b), method='spearman'))}
            report[f'{ct} | {orient}'] = res
            print(ct, orient, json.dumps(res, indent=1), flush=True)
        del W, z, mask
    os.makedirs(os.path.join(RESULTS_DIR, 'exp29_grnboost_states'), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'exp29_grnboost_states', 'centrality_port_check.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print('[Exp 29a] Done!')


if __name__ == '__main__':
    main()
