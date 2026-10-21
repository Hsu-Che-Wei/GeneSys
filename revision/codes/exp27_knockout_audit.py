#!/usr/bin/env python3
"""
Experiment 27: audit of the in-silico knockout regulator ranking (exp17).

Checks that the genome-wide gold-standard recall of the knockout generative
score is (i) statistically above chance (hypergeometric test at every K),
(ii) accompanied by enrichment for annotated TFs in general, and (iii) not a
by-product of expression level (a ranking by mean expression, detection rate
or variance of the same genes; exp26 statistics).

Usage: python experiments/exp27_knockout_audit.py
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp13_fig4_r50 import load_refs, DATA_DIR
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
KS = [100, 200, 300, 400]

wanted, exptf, gold, _ = load_refs()
gene_ids = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))['features'].values
tf_all = set(wanted['GeneID'])
N = len(gene_ids)
ko = pd.read_csv(os.path.join(RES, 'exp17_knockout', 'exp17_knockout_scores.csv')).set_index('GeneID')
ko = ko.loc[gene_ids]
out = {}


def top(scores, K):
    order = np.argsort(-np.asarray(scores), kind='stable')
    return set(gene_ids[order[:K]])


print(f'genes {N}, gold {len(gold)}, annotated TFs in gene list {len(tf_all & set(gene_ids))}')
for col in ['gen_effect', 'cls_effect', 'combined_effect']:
    rows = []
    for K in KS:
        t = top(ko[col].values, K)
        x_gold = len(t & gold)
        x_tf = len(t & tf_all)
        p_gold = hypergeom.sf(x_gold - 1, N, len(gold), K)
        n_tf = len(tf_all & set(gene_ids))
        p_tf = hypergeom.sf(x_tf - 1, N, n_tf, K)
        rows.append(dict(K=K, gold_hits=x_gold, gold_expected=K * len(gold) / N, p_gold=p_gold,
                         tf_hits=x_tf, tf_expected=K * n_tf / N, p_tf=p_tf))
    df = pd.DataFrame(rows)
    print(f'\n{col}\n{df.to_string(index=False)}')
    out[col] = df.to_dict('records')

stats_path = os.path.join(RES, 'exp26_gene_expression_stats.csv')
if os.path.exists(stats_path):
    st = pd.read_csv(stats_path).set_index('GeneID').loc[gene_ids]
    print('\nExpression-level confound check')
    for col in ['mean_expr', 'detection_rate', 'var_expr']:
        rows = []
        for K in KS:
            t = top(st[col].values, K)
            rows.append(dict(K=K, gold_hits=len(t & gold), tf_hits=len(t & tf_all)))
        rho = spearmanr(st[col].values, ko['gen_effect'].values).correlation
        print(f'  rank by {col}: {rows}   Spearman(gen_effect, {col}) = {rho:.3f}')
        out[f'confound_{col}'] = dict(rows=rows, spearman_with_gen_effect=float(rho))
    # gold-standard recall of the knockout score after removing expression level:
    # residual of log gen_effect on log mean expression (rank-based)
    from scipy.stats import rankdata
    r_ko = rankdata(ko['gen_effect'].values); r_ex = rankdata(st['mean_expr'].values)
    resid = r_ko - np.polyval(np.polyfit(r_ex, r_ko, 1), r_ex)
    rows = [dict(K=K, gold_hits=len(top(resid, K) & gold)) for K in KS]
    print(f'  gen_effect residualized on mean expression: {rows}')
    out['gen_effect_residualized_on_expression'] = rows
    # gold TFs: expression percentile
    gm = st.loc[[g for g in gold if g in st.index], 'mean_expr']
    print(f'  gold-standard TF mean-expression percentile (median): '
          f'{np.median([ (st["mean_expr"] < v).mean() for v in gm ])*100:.1f}')
else:
    print('\nexp26 statistics not found; expression confound check skipped')

json.dump(out, open(os.path.join(RES, 'exp27_knockout_audit.json'), 'w'), indent=2, default=float)
print('\nsaved exp27_knockout_audit.json')
