#!/usr/bin/env python3
"""
Experiment 35: the Figure 4 marker criterion, ported from the notebook call

    FindAllMarkers(seu, max.cells.per.ident = 10000, only.pos = T, test.use = "roc")

(DE_GeneSys_vs_scRNAseq.ipynb, Seurat v4 defaults: logfc.threshold = 0.25 on avg_log2FC,
min.pct = 0.1, return.thresh = 0.7 on myAUC for the ROC test), followed by the notebook's
within-cluster ranks (row_number after arrange(desc(.))).

`seurat_roc_markers` is the port. `--validate` runs it on the Figure 4 inputs
(input_X_root.csv, pred_X_WT.csv.gz) and compares the result with the published tables
(Input_X_root_markers_ROC.csv, GeneSys_P_root_markers_ROC.csv), so the port is accepted
only if it reproduces them.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata

DATA_DIR = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def seurat_roc_markers(mat, cluster_of_row, cluster_names, features,
                       logfc_threshold=0.25, min_pct=0.1, return_thresh=0.7):
    """mat: cells x genes (the Seurat `data` slot). Returns the marker table of
    FindAllMarkers(test.use='roc', only.pos=TRUE) with the notebook's rank columns."""
    mat = np.asarray(mat, dtype=np.float64)
    n_total = mat.shape[0]
    ranks = rankdata(mat, axis=0)
    expm = np.expm1(mat)
    tot_exp = expm.sum(axis=0)
    pos = (mat > 0)
    tot_pos = pos.sum(axis=0)
    out = []
    for ci, cname in enumerate(cluster_names):
        m = cluster_of_row == ci
        n1 = int(m.sum()); n2 = n_total - n1
        if n1 == 0:
            continue
        s1 = expm[m].sum(axis=0)
        mean1, mean2 = s1 / n1, (tot_exp - s1) / n2
        # Seurat v4 FoldChange on the data slot: log2(mean(expm1(x)) + 1)
        log2fc = np.log2(mean1 + 1) - np.log2(mean2 + 1)
        p1 = pos[m].sum(axis=0)
        pct1 = np.round(p1 / n1, 3); pct2 = np.round((tot_pos - p1) / n2, 3)
        auc = np.round((ranks[m].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n2), 3)
        # AUCMarkerTest: avg_diff = ExpMean(x1) - ExpMean(x2), natural log
        avg_diff = np.log(mean1 + 1) - np.log(mean2 + 1)
        keep = ((np.maximum(pct1, pct2) >= min_pct) & (log2fc >= logfc_threshold) &
                (auc > return_thresh))
        if not keep.any():
            continue
        sub = pd.DataFrame({'myAUC': auc[keep], 'avg_diff': avg_diff[keep],
                            'power': np.round(np.abs(auc[keep] - 0.5) * 2, 3),
                            'avg_log2FC': log2fc[keep], 'pct.1': pct1[keep],
                            'pct.2': pct2[keep], 'cluster': cname,
                            'gene': np.asarray(features)[keep]})
        sub['pct.diff'] = sub['pct.1'] - sub['pct.2']
        # notebook cell: arrange(desc(pct.diff)) -> row_number; arrange(desc(avg_diff)) ->
        # row_number; arrange(desc(myAUC)) -> row_number (stable sorts, ties keep prior order)
        sub = sub.sort_values('pct.diff', ascending=False, kind='stable')
        sub['pct.diff_rank'] = np.arange(1, len(sub) + 1)
        sub = sub.sort_values('avg_diff', ascending=False, kind='stable')
        sub['avg_diff_rank'] = np.arange(1, len(sub) + 1)
        sub = sub.sort_values('myAUC', ascending=False, kind='stable')
        sub['myAUC_rank'] = np.arange(1, len(sub) + 1)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def validate(tag, x_csv, y_csv, t_csv, published_csv):
    print(f'[{tag}] reading {x_csv}', flush=True)
    X = pd.read_csv(os.path.join(DATA_DIR, x_csv), header=None, dtype=np.float32).values[:, 1:]
    if X.shape[0] == len(pd.read_csv(os.path.join(DATA_DIR, y_csv))) + 1:
        X = X[1:]          # the file carries a header row
    y = pd.read_csv(os.path.join(DATA_DIR, y_csv), index_col=0).iloc[:, 0].astype(str).values
    t = pd.read_csv(os.path.join(DATA_DIR, t_csv), index_col=0).iloc[:, 0].astype(str).values
    feats = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))['features'].values
    print(f'[{tag}] X {X.shape}, labels {len(y)}', flush=True)
    lab = np.array([f'{a}_{b}' for a, b in zip(y, t)])
    names = list(pd.unique(lab))
    idx = np.array([names.index(l) for l in lab])
    mine = seurat_roc_markers(X, idx, names, feats)
    pub = pd.read_csv(os.path.join(DATA_DIR, published_csv))
    a = mine.set_index(['cluster', 'gene']); b = pub.set_index(['cluster', 'gene'])
    common = a.index.intersection(b.index)
    res = {'tag': tag, 'cells': int(X.shape[0]), 'clusters_mine': len(names),
           'clusters_published': int(pub['cluster'].nunique()),
           'rows_mine': len(a), 'rows_published': len(b), 'rows_common': len(common),
           'only_mine': int(len(a.index.difference(b.index))),
           'only_published': int(len(b.index.difference(a.index)))}
    for col in ['myAUC', 'avg_diff', 'avg_log2FC', 'pct.1', 'pct.2']:
        res[f'max_abs_diff_{col}'] = float(np.abs(a.loc[common, col].values -
                                                  b.loc[common, col].values).max())
    for col in ['myAUC_rank', 'avg_diff_rank']:
        res[f'identical_{col}'] = float((a.loc[common, col].values ==
                                         b.loc[common, col].values).mean())
    print(json.dumps(res, indent=1), flush=True)
    return res


def validate_objects():
    """Validation on exactly the cells stored in the Figure 4 Seurat objects
    (exported by exp36_export_fig4_seurat.R)."""
    O = os.path.join(RESULTS_DIR, 'exp36_fig4_objects')
    out = []
    for tag, label, pub_csv in [('X', 'scRNA-seq (Figure 4 object)', 'Input_X_root_markers_ROC.csv'),
                                ('P', 'GeneSys (Figure 4 object)', 'GeneSys_P_root_markers_ROC.csv')]:
        genes = [g.strip() for g in open(os.path.join(O, f'{tag}_genes.txt'))]
        cells = pd.read_csv(os.path.join(O, f'{tag}_cells.csv'))
        X = np.fromfile(os.path.join(O, f'{tag}_data_float32_genes_by_cells.bin'),
                        dtype=np.float32).reshape(len(cells), len(genes))
        lab = cells['celltype_timebin'].astype(str).values
        names = list(pd.unique(lab))
        idx = np.array([names.index(l) for l in lab])
        mine = seurat_roc_markers(X, idx, names, np.array(genes))
        mine.to_csv(os.path.join(O, f'{tag}_markers_port.csv'), index=False)
        pub = pd.read_csv(os.path.join(DATA_DIR, pub_csv))
        a = mine.set_index(['cluster', 'gene']); b = pub.set_index(['cluster', 'gene'])
        common = a.index.intersection(b.index)
        res = {'tag': label, 'cells': int(X.shape[0]), 'rows_mine': len(a),
               'rows_published': len(b), 'rows_common': len(common),
               'only_mine': int(len(a.index.difference(b.index))),
               'only_published': int(len(b.index.difference(a.index)))}
        for col in ['myAUC', 'avg_diff', 'avg_log2FC', 'pct.1', 'pct.2']:
            res[f'max_abs_diff_{col}'] = float(np.abs(a.loc[common, col].values -
                                                      b.loc[common, col].values).max())
        # R50 as in notebook 13, from the port table and from the published table
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from exp13_fig4_r50 import load_refs, r50_scan
        wanted, exptf, gold, _ = load_refs()
        for nm, tab in (('port', mine), ('published', pub)):
            tf = tab[tab['gene'].isin(set(exptf))]
            agg = tf.groupby('gene', as_index=False).agg(a=('myAUC_rank', 'mean'),
                                                         d=('avg_diff_rank', 'mean'))
            res[f'R50_{nm}'] = [int(r50_scan(agg.sort_values(c, ascending=False, kind='stable')
                                              ['gene'].values, gold, 70)) for c in ('a', 'd')]
        print(json.dumps(res, indent=1), flush=True)
        out.append(res)
    json.dump(out, open(os.path.join(RESULTS_DIR, 'exp35_seurat_port_validation_objects.json'), 'w'),
              indent=1)
    print('[Exp 35] Done!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--validate-objects', action='store_true')
    args = ap.parse_args()
    if args.validate_objects:
        validate_objects(); sys.exit(0)
    if args.validate:
        out = [validate('raw (Figure 4 scRNA-seq)', 'input_X_root.csv', 'input_Y_root.csv',
                        'input_T_root.csv', 'Input_X_root_markers_ROC.csv'),
               validate('GeneSys (Figure 4)', 'pred_X_WT.csv.gz', 'pred_Y_WT.csv.gz',
                        'pred_T_WT.csv.gz', 'GeneSys_P_root_markers_ROC.csv')]
        json.dump(out, open(os.path.join(RESULTS_DIR, 'exp35_seurat_port_validation.json'), 'w'),
                  indent=1)
        print('[Exp 35] Done!')
