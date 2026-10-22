#!/usr/bin/env python3
"""
Experiment 13: Python port of the Figure 4 R50 benchmark [Round 3, R1 + R3.4]

Faithful port of notebook 13 (13-GeneSys_Top_Regulators_Plots_zscore3_20240325,
R kernel) so that new data conditions (pseudobulk, scVI) can be scored with the
IDENTICAL pipeline. Reproduction gate: with --gate, scores the two published
conditions (GeneSys-generated, raw scRNA-seq) from the on-disk centrality CSVs
and compares against the Figure 4A numbers of notebook 13.

Figure 4A values (notebook 13):
  LIMA (GeneSys):   degree 314, out 375, in 327, betweenness 335, eigen 432
  LIMA (scRNAseq):  degree 365, out 444, in 330, betweenness 447, eigen 536
  DE   (GeneSys):   myAUC 316, avg_diff 295
  DE   (scRNAseq):  myAUC 318, avg_diff 287
  Permutation:      743.4 (mean of 1000)

DE order: the DE ranks of the marker tables are 1 for the strongest marker of a cluster, so the
TFs are ordered by ascending mean rank (strongest markers first). Notebook 13 sorted them in
descending order until 2026-10-21 (310/331 and 380/403 in the first versions of Figure 4);
--de-descending reproduces that order.

Usage:
  python experiments/exp13_fig4_r50.py --gate
  python experiments/exp13_fig4_r50.py --lima-prefix pseudobulk_TF_GRN_centrality \
      --de-csv <markers.csv> --label pseudobulk
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

DATA_DIR = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TRANSITIONS = ["t0-t1", "t1-t3", "t3-t5", "t5-t7", "t7-t9"]
CT_PREFIXES = ['atri', 'tri', 'lrc', 'cor', 'end', 'per', 'pro', 'xyl', 'phl', 'col']
CENTRALITIES = ['degree_centrality', 'out_centrality', 'in_centrality',
                'betweenness_centrality', 'closeness_centrality',
                'eigenvector_centrality']
STRATEGIES = ['degree_centrality', 'out_centrality', 'in_centrality',
              'betweenness_centrality', 'eigenvector_centrality']

# The 31 hand patches from notebook 13 cell 6 / notebook 12 cell 17
NAME_PATCHES = {
    "AT2G33880": "WOX9", "AT2G45160": "SCL27", "AT5G04410": "NAC78",
    "AT3G29035": "ORS1", "AT2G02540": "ZHD3", "AT3G16500": "IAA26",
    "AT5G09740": "HAG5", "AT4G24660": "ZHD2", "AT5G46880": "HDG5",
    "AT1G28420": "RLT1", "AT1G14580": "BLJ", "AT3G45260": "BIB",
    "AT2G02070": "RVN", "AT2G28160": "FIT", "AT1G68360": "GIS3",
    "AT1G20640": "NLP4", "AT5G05550": "VFP5", "AT3G59470": "FRF1",
    "AT5G15150": "HAT7", "AT5G14750": "WER", "AT1G75710": "BRON",
    "AT1G74500": "TMO7", "AT2G12646": "RITF1", "AT3G48100": "ARR5",
    "AT4G16141": "GATA17L", "AT5G65640": "NFL", "AT1G62700": "VND5",
    "AT4G36160": "VND2", "AT5G66300": "VND3", "AT1G12260": "VND4",
    "AT5G62380": "VND6",
}

# Tissue sub-panels: (label, StringDB flag column [0-based], r50, celltype prefixes,
#                     DE cluster substrings)
SUBPANELS = [
    ('Stele', 4, 16, ['xyl', 'phl', 'pro'], ['Xylem', 'Phloem', 'Procambium']),
    ('Epidermis', 5, 13, ['atri', 'tri', 'lrc'],
     ['Atrichoblast', 'Trichoblast', 'Lateral Root Cap']),
    ('Xylem', 6, 10, ['xyl'], ['Xylem']),
    ('Trichoblast', 7, 5, ['tri'], ['Trichoblast']),
]

PUBLISHED = {
    # order: degree, out-degree, in-degree, betweenness, eigenvector. Out-degree is the number of
    # targets a TF regulates and in-degree the number of regulators acting on it (Figure 4B).
    ('LIMA', 'genesys'): [314, 375, 327, 335, 432],
    ('LIMA', 'scrnaseq'): [365, 444, 330, 447, 536],
    ('DE', 'genesys'): [316, 295],
    ('DE', 'scrnaseq'): [318, 287],
    'permutation': 743.4,
}


def min_max(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def load_refs():
    wanted = pd.read_csv(os.path.join(DATA_DIR, "Kay_TF_thalemine_annotations.csv"))
    for gid, name in NAME_PATCHES.items():
        wanted.loc[wanted['GeneID'] == gid, 'Name'] = name
    gene_list = pd.read_csv(os.path.join(DATA_DIR, "gene_list_1108.csv"))
    features = gene_list['features'].tolist()
    feat_set = set(features)
    # R intersect(gene_list$features, wanted$GeneID): order of first arg, unique
    wanted_ids = set(wanted['GeneID'])
    exptf = [g for g in dict.fromkeys(features) if g in wanted_ids]
    gold = pd.read_csv(os.path.join(
        DATA_DIR, "Gold_Standard_Root_TF_MINI_EX_StringDB.csv"))['GeneID'].tolist()
    stringdb = pd.read_csv(os.path.join(DATA_DIR, "Gold_Standard_Root_TF_StringDB.csv"))
    return wanted, exptf, set(gold), stringdb


def load_lima_table(prefix, data_dir=DATA_DIR):
    """Notebook 13 cells 11-28: load 5 transition centrality CSVs, min-max
    normalize the 60 centrality columns, outer-join on TF name, NA->0,
    suffix _1.._5, then map TF name -> GeneID."""
    frames = []
    for k, tr in enumerate(TRANSITIONS, start=1):
        path = os.path.join(data_dir, f"{prefix}_{tr}_zscore3.csv")
        df = pd.read_csv(path)
        tf_names = df.iloc[:, 0]
        cent = df.iloc[:, 2:62].apply(min_max, axis=0, raw=True)
        cent.columns = [f"{c}_{k}" for c in cent.columns]
        out = pd.concat([tf_names.rename('TF'), cent], axis=1)
        frames.append(out)
    dat = frames[0]
    for f in frames[1:]:
        dat = dat.merge(f, on='TF', how='outer')
    dat = dat.fillna(0)
    return dat


def attach_gene_ids(dat, wanted):
    """GeneID <- wanted$GeneID[match(dat$TF, wanted$Name)] (first match)."""
    name_to_id = {}
    for _, row in wanted.iterrows():
        if row['Name'] not in name_to_id:
            name_to_id[row['Name']] = row['GeneID']
    dat = dat.copy()
    dat.insert(0, 'GeneID', dat['TF'].map(name_to_id))
    return dat


def r50_scan(gene_ids, gold_set, r50):
    """Walk the ranked list; return 1-based rank of the r50-th gold hit."""
    count = 0
    for i, gid in enumerate(gene_ids, start=1):
        if isinstance(gid, str) and gid in gold_set:
            count += 1
            if count == r50:
                return i
    return None


def run_r50_lima(dat, strategy, gold_set, r50, ct_prefixes=None):
    """run_r50_genesys / _scRNAseq. strategy e.g. 'degree_centrality'.
    ct_prefixes limits the summed columns for tissue sub-panels."""
    if ct_prefixes is None:
        cols = [c for c in dat.columns if strategy in c]
    else:
        cols = [c for c in dat.columns
                if any(c.startswith(f"{p}_{strategy}") for p in ct_prefixes)]
    score = min_max(dat[cols].sum(axis=1).values)
    order = pd.Series(score).sort_values(ascending=False, kind='stable').index
    ranked_ids = dat['GeneID'].values[order]
    return r50_scan(ranked_ids, gold_set, r50)


def load_de_table(path, exptf, cluster_substrings=None):
    """Notebook 13 cells 91-101: mean myAUC_rank/avg_diff_rank per gene over
    (optionally filtered) clusters, restricted to expressed TFs."""
    de = pd.read_csv(path)
    if cluster_substrings:
        mask = pd.Series(False, index=de.index)
        for s in cluster_substrings:
            mask |= de['cluster'].astype(str).str.contains(s, regex=False)
        de = de[mask]
    de = de[de['gene'].isin(set(exptf))]
    dat = de.groupby('gene', as_index=False).agg(
        myAUC_rank=('myAUC_rank', 'mean'),
        avg_diff_rank=('avg_diff_rank', 'mean'))
    dat = dat.rename(columns={'gene': 'GeneID'})
    return dat


def run_r50_de(dat, col, gold_set, r50, ascending=True):
    """DE ranks are 1 for the strongest marker of a cluster: ascending mean rank puts the
    strongest markers first."""
    ranked = dat.sort_values(col, ascending=ascending, kind='stable')
    return r50_scan(ranked['GeneID'].values, gold_set, r50)


def permutation_r50(exptf, gold_set, r50, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    arr = np.array(exptf)
    is_gold = np.array([g in gold_set for g in arr])
    vals = []
    for _ in range(n):
        perm = rng.permutation(len(arr))
        gold_pos = np.where(is_gold[perm])[0]
        if len(gold_pos) >= r50:
            vals.append(int(gold_pos[r50 - 1]) + 1)
    return float(np.mean(vals))


def score_condition(label, lima_prefix, de_csv, wanted, exptf, gold_set,
                    stringdb, data_dir=DATA_DIR, de_ascending=True):
    """Score one data condition across LIMA strategies and DE, for the main
    panel and all tissue sub-panels."""
    rows = []
    dat = attach_gene_ids(load_lima_table(lima_prefix, data_dir), wanted) \
        if lima_prefix else None
    de_all = load_de_table(de_csv, exptf) if de_csv else None

    gold_main = gold_set
    r50_main = len(gold_main) // 2

    panels = [('All', None, None, gold_main, r50_main)]
    for name, flag_col, r50_sub, ct_prefixes, de_clusters in SUBPANELS:
        sub_ids = set(stringdb[stringdb.iloc[:, flag_col] == 'Yes']['GeneID'])
        gold_sub = sub_ids & gold_main
        panels.append((name, ct_prefixes, de_clusters, gold_sub, r50_sub))

    for pname, ct_prefixes, de_clusters, gold, r50 in panels:
        if dat is not None:
            for strat in STRATEGIES:
                val = run_r50_lima(dat, strat, gold, r50, ct_prefixes)
                rows.append({'condition': label, 'panel': pname,
                             'method': 'LIMA', 'strategy': strat, 'R50': val,
                             'n_gold': len(gold), 'r50_target': r50})
        if de_csv:
            de_dat = (load_de_table(de_csv, exptf, de_clusters)
                      if de_clusters else de_all)
            for col in ['myAUC_rank', 'avg_diff_rank']:
                val = run_r50_de(de_dat, col, gold, r50, ascending=de_ascending)
                rows.append({'condition': label, 'panel': pname,
                             'method': 'DE', 'strategy': col, 'R50': val,
                             'n_gold': len(gold), 'r50_target': r50})
    return pd.DataFrame(rows)


def gate(wanted, exptf, gold_set, stringdb):
    print(f"exptf: {len(exptf)} (expected 1495)")
    print(f"gold intersection: {len(gold_set)} (expected 140)")

    res_g = score_condition('genesys', 'TF_GRN_centrality',
                            os.path.join(DATA_DIR, 'GeneSys_P_root_markers_ROC.csv'),
                            wanted, exptf, gold_set, stringdb)
    res_r = score_condition('scrnaseq', 'scRNAseq_TF_GRN_centrality',
                            os.path.join(DATA_DIR, 'Input_X_root_markers_ROC.csv'),
                            wanted, exptf, gold_set, stringdb)
    perm = permutation_r50(exptf, gold_set, 70)

    print("\n=== Reproduction gate (panel All) ===")
    ok = True
    for label, res, key in [('genesys', res_g, ('LIMA', 'genesys')),
                            ('scrnaseq', res_r, ('LIMA', 'scrnaseq'))]:
        got = res[(res.panel == 'All') & (res.method == 'LIMA')]['R50'].tolist()
        exp = PUBLISHED[key]
        match = got == exp
        ok &= match
        print(f"LIMA {label}: got {got} expected {exp} {'MATCH' if match else 'MISMATCH'}")
    for label, res, key in [('genesys', res_g, ('DE', 'genesys')),
                            ('scrnaseq', res_r, ('DE', 'scrnaseq'))]:
        got = res[(res.panel == 'All') & (res.method == 'DE')]['R50'].tolist()
        exp = PUBLISHED[key]
        match = got == exp
        ok &= match
        print(f"DE {label}: got {got} expected {exp} {'MATCH' if match else 'MISMATCH'}")
    # the descending order used before the correction, for reference
    for label, csv in [('genesys', 'GeneSys_P_root_markers_ROC.csv'),
                       ('scrnaseq', 'Input_X_root_markers_ROC.csv')]:
        de_dat = load_de_table(os.path.join(DATA_DIR, csv), exptf)
        desc = [run_r50_de(de_dat, c, gold_set, 70, ascending=False)
                for c in ['myAUC_rank', 'avg_diff_rank']]
        print(f"DE {label} (descending, before the correction): {desc}")
    print(f"Permutation: {perm:.1f} expected ~{PUBLISHED['permutation']}")

    out = pd.concat([res_g, res_r], ignore_index=True)
    out.to_csv(os.path.join(RESULTS_DIR, 'exp13_gate_r50.csv'), index=False)
    print(f"\nGate {'PASSED' if ok else 'FAILED'}; details in exp13_gate_r50.csv")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Exp 13: Fig 4 R50 port")
    ap.add_argument('--gate', action='store_true',
                    help='Reproduce published GeneSys/raw numbers')
    ap.add_argument('--label', help='Condition label for scoring mode')
    ap.add_argument('--lima-prefix', default=None,
                    help='Centrality CSV prefix, e.g. pseudobulk_TF_GRN_centrality')
    ap.add_argument('--lima-dir', default=DATA_DIR)
    ap.add_argument('--de-csv', default=None, help='Markers ROC CSV path')
    ap.add_argument('--de-descending', action='store_true',
                    help='Sort DE mean ranks descending (weakest markers first), the order '
                         'notebook 13 used before the correction')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    wanted, exptf, gold_set, stringdb = load_refs()

    if args.gate:
        gate(wanted, exptf, gold_set, stringdb)
        return

    if not args.label:
        ap.error('--label required in scoring mode')
    res = score_condition(args.label, args.lima_prefix, args.de_csv,
                          wanted, exptf, gold_set, stringdb,
                          data_dir=args.lima_dir, de_ascending=not args.de_descending)
    out = args.out or os.path.join(RESULTS_DIR, f'exp13_r50_{args.label}.csv')
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
