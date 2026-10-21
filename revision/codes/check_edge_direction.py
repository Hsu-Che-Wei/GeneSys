#!/usr/bin/env python3
"""Which side of the LIMA do the out- and in-degree centralities count?

Published LIMA of the t1 to t3 transition, Atrichoblast (genesys_ctw_t1-t3.npy, slice 0), and the
centrality table of notebook 12 (TF_GRN_centrality_t1-t3_zscore3.csv). W[i, j] is the coefficient
of gene i at the earlier state for gene j at the later state (lstsq(cfrom, cto) in notebook 12).
For every TF of the network, the number of retained entries in its row (later-state TFs it
predicts, its targets) and in its column (earlier-state TFs that predict it, its regulators) are
compared with the out- and in-degree centralities of the table. Out-degree has to equal the row
count and in-degree the column count."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp14_condition_pipeline import NAME_PATCHES, DATA_DIR

gene_list = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))
wanted = pd.read_csv(os.path.join(DATA_DIR, 'Kay_TF_thalemine_annotations.csv'))
for gid, name in NAME_PATCHES.items():
    wanted.loc[wanted['GeneID'] == gid, 'Name'] = name
features = gene_list['features'].tolist()
ids = set(wanted['GeneID'])
tf_idx = np.sort(np.array([i for i, g in enumerate(features) if g in ids]))
id2name = {}
for _, r in wanted.iterrows():
    id2name.setdefault(r['GeneID'], r['Name'])
names = np.array([id2name[features[i]] for i in tf_idx])

TABLE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, 'TF_GRN_centrality_t1-t3_zscore3.csv')
pub = pd.read_csv(TABLE, index_col=0)
W = np.asarray(np.load(os.path.join(DATA_DIR, 'genesys_ctw_t1-t3.npy'), mmap_mode='r')[0], dtype=np.float32)
mask = np.abs((W - W.mean()) / W.std()) > 3.0
m = mask[np.ix_(tf_idx, tf_idx)]
keep = np.where((m.sum(axis=1) >= 1) & (m.sum(axis=0) >= 1))[0]      # as in network(): reciprocal
m = m[np.ix_(keep, keep)]
n = len(keep)
row = m.sum(axis=1) / (n - 1)      # later-state TFs that this TF predicts
col = m.sum(axis=0) / (n - 1)      # earlier-state TFs that predict this TF
df = pd.DataFrame({'row_count_targets': row, 'column_count_regulators': col}, index=names[keep])
common = df.index.intersection(pub.index)
out_pub, in_pub = pub.loc[common, 'atri_out_centrality'], pub.loc[common, 'atri_in_centrality']
res = {'table': TABLE, 'n_tfs': int(n), 'n_common': int(len(common)),
       'max_abs_diff out-degree vs column count (regulators)': float((out_pub - df.loc[common, 'column_count_regulators']).abs().max()),
       'max_abs_diff out-degree vs row count (targets)': float((out_pub - df.loc[common, 'row_count_targets']).abs().max()),
       'max_abs_diff in-degree vs row count (targets)': float((in_pub - df.loc[common, 'row_count_targets']).abs().max()),
       'max_abs_diff in-degree vs column count (regulators)': float((in_pub - df.loc[common, 'column_count_regulators']).abs().max())}
ex = df.loc[common].assign(published_out=out_pub, published_in=in_pub).sort_values('row_count_targets', ascending=False).head(8)
print(json.dumps(res, indent=1)); print((ex * (n - 1)).round(0).astype(int).to_string())
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', sys.argv[2] if len(sys.argv) > 2 else 'check_edge_direction.json'), 'w'), indent=1)
