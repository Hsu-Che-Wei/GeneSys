#!/usr/bin/env python3
"""
Experiment 43: what the DE statistics of Figure 4 measure on GeneSys-generated and on scRNA-seq data.

Reads the two marker tables of Figure 4 (Seurat ROC test per cell type and time bin, notebook 13
inputs) and the expression matrices they were computed from, and reports:

  input        share of zero entries in the expression matrix given to Seurat (first 3,000 cells)
  markers      for the TF markers of each scenario: detection (pct.1, pct.2), AUC (median and the
               share at 0.95 or more), average expression difference and log2 fold change
  paired       for the TF markers found in the same cluster in both scenarios: median ratio
               (GeneSys / scRNA-seq) and Spearman correlation of AUC, average difference and
               log2 fold change
  thresholds   share of all markers within 0.05 of the AUC cut-off (0.7) and of the log2 fold-change
               cut-off (0.25), i.e. how often each filter decides
  coverage     TFs and gold-standard TFs among the markers; gold-standard TFs never ranked by DE
  ordering     R50 of the DE rankings (mean within-cluster rank, strongest markers first, as in
               notebook 13), R50 of the same TFs in random order (mean of 2,000 orders), and the
               ROC AUC with which each ordering separates gold-standard TFs from the other ranked TFs

Output: results/exp43_de_statistics.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp13_fig4_r50 import DATA_DIR, load_refs, r50_scan  # noqa: E402

RES = os.path.join(HERE, 'results')
SCEN = {'genesys': ('GeneSys_P_root_markers_ROC.csv', 'pred_X_WT.csv'),
        'scrnaseq': ('Input_X_root_markers_ROC.csv', 'input_X_root.csv')}


def main():
    wanted, exptf, gold, stringdb = load_refs()
    ex = set(exptf)
    out = {'input': {}, 'markers': {}, 'thresholds': {}, 'coverage': {}, 'ordering': {}}
    tables = {}
    for s, (markers, matrix) in SCEN.items():
        x = pd.read_csv(os.path.join(DATA_DIR, matrix), nrows=3000, index_col=0).values
        out['input'][s] = {'zero_fraction': round(float((x == 0).mean()), 3)}
        d = pd.read_csv(os.path.join(DATA_DIR, markers))
        tables[s] = d
        t = d[d['gene'].isin(ex)]
        out['markers'][s] = {
            'tf_marker_rows': int(len(t)),
            'pct1_median': float(t['pct.1'].median()), 'pct2_median': float(t['pct.2'].median()),
            'auc_median': round(float(t['myAUC'].median()), 3),
            'auc_ge_095': round(float((t['myAUC'] >= 0.95).mean()), 3),
            'avg_diff_median': round(float(t['avg_diff'].median()), 3),
            'log2fc_median': round(float(t['avg_log2FC'].median()), 3)}
        out['thresholds'][s] = {
            'auc_within_005_of_cutoff': round(float((d['myAUC'] < 0.75).mean()), 3),
            'log2fc_within_005_of_cutoff': round(float((d['avg_log2FC'] < 0.30).mean()), 3),
            'min_pct1': float(d['pct.1'].min())}
        agg = t.groupby('gene', as_index=False).agg(rA=('myAUC_rank', 'mean'), rD=('avg_diff_rank', 'mean'))
        agg['gold'] = agg['gene'].isin(gold)
        out['coverage'][s] = {'tfs_ranked': int(len(agg)), 'gold_ranked': int(agg['gold'].sum()),
                              'gold_not_ranked': int(len(gold) - agg['gold'].sum())}
        rng = np.random.default_rng(0)
        null = float(np.mean([r50_scan(rng.permutation(agg['gene'].values), gold, 70) for _ in range(2000)]))
        out['ordering'][s] = {
            'r50_mean_auc_rank': int(r50_scan(agg.sort_values('rA', kind='stable')['gene'].values, gold, 70)),
            'r50_mean_avg_diff_rank': int(r50_scan(agg.sort_values('rD', kind='stable')['gene'].values, gold, 70)),
            'r50_random_order_same_tfs': round(null, 1),
            'gold_vs_other_auc_mean_auc_rank': round(float(roc_auc_score(agg['gold'], -agg['rA'])), 3),
            'gold_vs_other_auc_mean_avg_diff_rank': round(float(roc_auc_score(agg['gold'], -agg['rD'])), 3)}
    m = tables['genesys'].merge(tables['scrnaseq'], on=['cluster', 'gene'], suffixes=('_g', '_r'))
    m = m[m['gene'].isin(ex)]
    out['paired'] = {'shared_tf_marker_rows': int(len(m))}
    for c in ('myAUC', 'avg_diff', 'avg_log2FC'):
        out['paired'][c] = {'median_ratio_genesys_over_scrnaseq': round(float((m[c + '_g'] / m[c + '_r']).median()), 3),
                            'spearman': round(float(spearmanr(m[c + '_g'], m[c + '_r'])[0]), 3)}
    with open(os.path.join(RES, 'exp43_de_statistics.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
