#!/usr/bin/env python3
"""
Experiment 20: Expanded Figure 4 assembly [Round 3 headline figure]

Combines R50 results across all data conditions and methods into the
expanded Figure 4 panel and summary table:

  Data conditions: GeneSys-generated, raw scRNA-seq, pseudobulk (averaging
  control, R3.4), scVI-denoised (external denoiser, R1).
  Methods: LIMA centrality (5 strategies), DE (2 strategies), CellOracle
  (published values, GeneSys/raw only; inputs not recomputable, see G8),
  GRNBoost2 centrality (exp29-31, five centralities, GeneSys-generated data),
  in-silico knockout (exp17), permutation baseline.

Inputs (produced by exp13 scoring runs + exp17/exp19):
  results/exp13_gate_r50.csv          (genesys + scrnaseq)
  results/exp13_r50_pseudobulk.csv
  results/exp13_r50_scvi.csv
  results/exp17_knockout/exp17_summary.json
  results/exp19_grnboost/exp19b_summary.json

Usage:
  python experiments/exp20_fig4_expanded.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG4_DIR = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"

# Published CellOracle Fig 4A values (deg/out/in/betweenness/eigenvector);
# the CellOracle input CSVs were not migrated from the Duke cluster, so these
# bars carry over from the published figure unchanged.
CELLORACLE_PUBLISHED = {
    'genesys': [434, 466, 312, 335, 445],
    'scrnaseq': [470, 513, 517, 410, 453],
}
PERMUTATION_R50 = 743.4

COND_LABELS = {
    'genesys': 'GeneSys',
    'scrnaseq': 'scRNA-seq (raw)',
    'pseudobulk': 'Pseudobulk (averaging control)',
    'scvi': 'scVI-denoised',
}
COND_COLORS = {
    'genesys': '#2166ac',
    'scrnaseq': '#b2182b',
    'pseudobulk': '#f4a582',
    'scvi': '#92c5de',
}


def load_all():
    frames = []
    gate = os.path.join(RESULTS_DIR, 'exp13_gate_r50.csv')
    if os.path.exists(gate):
        frames.append(pd.read_csv(gate))
    for cond in ['pseudobulk', 'scvi']:
        p = os.path.join(RESULTS_DIR, f'exp13_r50_{cond}.csv')
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        else:
            print(f"  (missing {p})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig4_criterion_de_rows():
    """DE schemes scored exactly as in Figure 4. The marker tables are built by the port of
    the notebook call FindAllMarkers(test.use='roc', only.pos=T) (Seurat v4 defaults:
    avg_log2FC >= 0.25, min.pct = 0.1, myAUC > 0.7; exp35, validated against the published
    scRNA-seq table), with the notebook's within-cluster ranks, applied to all four data
    conditions. As in notebook 13, the TF rows are averaged per gene and ordered by
    ascending mean rank, strongest markers first (rank 1 is the strongest marker of a cluster).
    A condition whose markers contain fewer than 70 gold-standard
    TFs cannot reach R50 and is returned with R50 = NaN."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from exp13_fig4_r50 import load_refs, r50_scan
    wanted, exptf, gold_set, stringdb = load_refs()
    exptf_set = set(exptf)
    cond_dir = os.path.join(RESULTS_DIR, 'exp14_conditions')
    tables = {
        # GeneSys and scRNA-seq: the marker tables of Figure 4 themselves (notebook 13)
        'genesys': os.path.join(FIG4_DIR, 'GeneSys_P_root_markers_ROC.csv'),
        'scrnaseq': os.path.join(FIG4_DIR, 'Input_X_root_markers_ROC.csv'),
        # pseudobulk and scVI: the same marker call (exp35 port) on batches of the same composition
        'pseudobulk': os.path.join(cond_dir, 'pseudobulk_markers_ROC_fig4.csv'),
        'scvi': os.path.join(cond_dir, 'scvi_markers_ROC_fig4.csv'),
    }
    rows, coverage = [], []
    for cond, path in tables.items():
        if not os.path.exists(path):
            print(f"  (missing {path})")
            continue
        de = pd.read_csv(path)
        tf = de[de['gene'].isin(exptf_set)]
        agg = tf.groupby('gene', as_index=False).agg(
            myAUC_rank=('myAUC_rank', 'mean'),
            avg_diff_rank=('avg_diff_rank', 'mean'))
        n_gold = agg['gene'].isin(gold_set).sum()
        coverage.append({'condition': cond, 'markers': len(de),
                         'clusters': de['cluster'].nunique(),
                         'genes_per_cluster': round(len(de) / max(1, de['cluster'].nunique())),
                         'TFs': agg['gene'].nunique(), 'gold_TFs': int(n_gold)})
        for col in ['myAUC_rank', 'avg_diff_rank']:
            r50 = (r50_scan(agg.sort_values(col, ascending=True, kind='stable')
                            ['gene'].values, gold_set, 70)
                   if n_gold >= 70 else float('nan'))
            rows.append({'condition': cond, 'method': 'DE',
                         'strategy': col, 'R50': r50})
    pd.DataFrame(coverage).to_csv(
        os.path.join(RESULTS_DIR, 'exp20_de_marker_coverage.csv'), index=False)
    print(pd.DataFrame(coverage).to_string(index=False))
    return rows


def main():
    df = load_all()
    if df.empty:
        sys.exit("No exp13 outputs found yet.")
    main_panel = df[df['panel'] == 'All'].copy()

    rows = []
    # LIMA rows from the exp13 scoring runs; DE rows from fig4_criterion_de_rows, which applies
    # the Figure 4 marker call to all four conditions
    for _, r in main_panel.iterrows():
        if r['method'] == 'DE':
            continue
        rows.append({'condition': r['condition'], 'method': r['method'],
                     'strategy': r['strategy'], 'R50': r['R50']})
    rows += fig4_criterion_de_rows()
    for cond, vals in CELLORACLE_PUBLISHED.items():
        for strat, v in zip(['degree_centrality', 'out_centrality',
                             'in_centrality', 'betweenness_centrality',
                             'eigenvector_centrality'], vals):
            rows.append({'condition': cond, 'method': 'CellOracle',
                         'strategy': strat, 'R50': v})

    ko_path = os.path.join(RESULTS_DIR, 'exp17_knockout', 'exp17_summary.json')
    if os.path.exists(ko_path):
        ko = json.load(open(ko_path))
        for col in ['gen_effect', 'cls_effect', 'combined_effect']:
            rows.append({'condition': 'genesys', 'method': 'Knockout',
                         'strategy': col,
                         'R50': ko[col]['r50_within_tf']})
    # GRNBoost2 centrality (exp29-31): the same five centralities as LIMA, on
    # the same four conditions, scored by the exp13 R50 code.
    gb_any = False
    for cond in ['genesys']:
        p = os.path.join(RESULTS_DIR, f'exp13_r50_grnboost_{cond}.csv')
        if os.path.exists(p):
            gbdf = pd.read_csv(p)
            gbdf = gbdf[(gbdf['panel'] == 'All') & (gbdf['method'] == 'LIMA')]
            for _, r in gbdf.iterrows():
                rows.append({'condition': cond, 'method': 'GRNBoost2',
                             'strategy': r['strategy'], 'R50': r['R50']})
            gb_any = True
        else:
            print(f"  (missing {p})")
    if not gb_any:
        print('  GRNBoost2 centrality not yet available; no GRNBoost2 rows written')

    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(RESULTS_DIR, 'exp20_fig4_expanded_table.csv'),
                index=False)
    print(full.to_string(index=False))

    # ── Figure: grouped by method family, bars = condition, points = strategies
    fig, ax = plt.subplots(figsize=(13, 6))
    methods = [m for m in ['LIMA', 'CellOracle', 'DE', 'GRNBoost2', 'Knockout']
               if m in set(full['method'])]
    conds = [c for c in ['genesys', 'scrnaseq', 'pseudobulk', 'scvi']
             if c in set(full['condition'])]
    width = 0.8 / len(conds)
    for mi, method in enumerate(methods):
        for ci, cond in enumerate(conds):
            sub = full[(full['method'] == method) & (full['condition'] == cond)]
            if sub.empty:
                continue
            xpos = mi + (ci - (len(conds) - 1) / 2) * width
            vals = sub['R50'].dropna().values
            ax.bar(xpos, vals.mean(), width * 0.9,
                   color=COND_COLORS[cond],
                   label=COND_LABELS[cond] if mi == 0 else None)
            ax.scatter([xpos] * len(vals), vals, color='k', s=12, zorder=3)
    ax.axhline(PERMUTATION_R50, color='gray', linestyle='--',
               label=f'Permutation ({PERMUTATION_R50:.0f})')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel('R50 (lower is better)', fontsize=13)
    ax.set_title('TF prioritization across data conditions and methods '
                 '(140-TF gold standard)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f'exp20_fig4_expanded.{ext}'),
                    dpi=150, bbox_inches='tight')
    print("Saved exp20_fig4_expanded.pdf/.png")


if __name__ == '__main__':
    main()
