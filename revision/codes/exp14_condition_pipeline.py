#!/usr/bin/env python3
"""
Experiment 14: New data conditions for the Figure 4 benchmark
[Round 3: R1 remaining point (scVI) and R3 point 4 (pseudobulk)]

Runs the IDENTICAL LIMA -> z-threshold -> TF-network centrality pipeline
(port of repo notebook 12) and the DE marker-table construction on new data
conditions, producing centrality CSVs with the same schema as
TF_GRN_centrality_t*_zscore3.csv so exp13 can score them with the same R50
code that reproduces the published Figure 4 numbers exactly.

Conditions:
  pseudobulk : plain averaging control (R3.4). Each LIMA row / DE pseudo-cell
               is the mean of `--boot-cells` real cells sampled from one
               (cell type, time slot) state. No network involved.
  scvi       : scVI-denoised per-cell profiles (R1). Requires the denoised
               matrix built by exp15_scvi_prepare.py.

State convention (matches the raw scRNA-seq condition, notebook
12-scRNAseq_*): transition tA-tB uses slot indices A and B directly; cells
for cell type ct at slot t are those labeled (lineage cell type at slot t,
time float t+1). LIMA uses torch.linalg.lstsq on CPU (default driver),
n_reps batches of ~200 rows per cell type, mean across batches, global
z-score per cell-type slice, |z| > 3.

Usage (one CPU job per transition for LIMA, one for DE):
  sbatch --job-name=pb_t0t1 experiments/slurm/run_cpu.sh \
      experiments/exp14_condition_pipeline.py --condition pseudobulk --mode lima --transition t0-t1
  ...
  sbatch --job-name=pb_de experiments/slurm/run_cpu.sh \
      experiments/exp14_condition_pipeline.py --condition pseudobulk --mode de
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_root_data, load_lineage, NUM2NAME, RESULTS_DIR

DATA_DIR = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
OUT_DATA_DIR = os.path.join(RESULTS_DIR, "exp14_conditions")

TRANSITION_SLOTS = {
    "t0-t1": (0, 1), "t1-t3": (1, 3), "t3-t5": (3, 5),
    "t5-t7": (5, 7), "t7-t9": (7, 9),
}
# Notebook 12 cell 9 ordering (must match column prefixes below)
CTS = ['Atrichoblast', 'Trichoblast', 'Cortex', 'Endodermis', 'Pericycle',
       'Procambium', 'Xylem', 'Phloem', 'Lateral Root Cap', 'Columella']
CT_PREFIX = {'Atrichoblast': 'atri', 'Trichoblast': 'tri', 'Cortex': 'cor',
             'Endodermis': 'end', 'Pericycle': 'per', 'Procambium': 'pro',
             'Xylem': 'xyl', 'Phloem': 'phl', 'Lateral Root Cap': 'lrc',
             'Columella': 'col'}
N_GENES = 17513

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


# ── Data sources ───────────────────────────────────────────────────────────
def load_cell_pool_from_pickle():
    """All-splits cell pool from the root pickle (matches the raw condition,
    which used vstack(train, val, test)). Returns (X, ct_names, ct_times)."""
    data = load_root_data()
    X = np.vstack([data['X_train'], data['X_val'], data['X_test']])
    labels = np.concatenate([data['y_train'], data['y_val'], data['y_test']])
    ct_names = np.array([lab.split('_', 1)[0] for lab in labels])
    ct_times = np.array([float(lab.split('_', 1)[1]) for lab in labels])
    return X, ct_names, ct_times


def load_cell_pool_from_scvi(scvi_file):
    """Denoised matrix built by exp15_scvi_prepare.py: npz with X (cells x
    17513, [0,1]-scaled), ct_names, ct_times."""
    z = np.load(scvi_file, allow_pickle=True)
    return z['X'], z['ct_names'].astype(str), z['ct_times'].astype(float)


def state_indices(ct_names, ct_times, lineage):
    """(cell type name, slot) -> cell indices, mirroring the ORIGINAL
    Root_Dataset used by the published LIMA notebooks
    (genesys_evaluate_v1.py): slot 0 is a Quiescent Center cell (any time
    bin), slot k for k >= 1 uses cells labeled (trajectory cell type at slot
    k, time bin == k)."""
    n_tb = lineage.shape[1]
    pools = {}
    qc_idx = np.where(ct_names == 'Quiescent Center')[0]
    terminal = {lineage.iloc[j, n_tb - 1]: j for j in range(lineage.shape[0])}
    for ct, j in terminal.items():
        pools[(ct, 0)] = qc_idx
        for t in range(1, n_tb):
            slot_ct = lineage.iloc[j, t]
            idx = np.where((ct_names == slot_ct) & (ct_times == float(t)))[0]
            pools[(ct, t)] = idx
    return pools


def sample_state_rows(X, pools, ct, slot, n_rows, rng, boot_cells=None):
    """One batch of state rows for (ct, slot).
    boot_cells=None  -> individual cells (scvi / raw style)
    boot_cells=k     -> each row is the mean of k cells sampled with
                        replacement (pseudobulk)."""
    idx = pools[(ct, slot)]
    if len(idx) == 0:
        return np.zeros((n_rows, X.shape[1]), dtype=np.float32)
    if boot_cells is None:
        chosen = rng.choice(idx, n_rows, replace=True)
        return X[chosen].astype(np.float32)
    rows = np.empty((n_rows, X.shape[1]), dtype=np.float32)
    for r in range(n_rows):
        chosen = rng.choice(idx, boot_cells, replace=True)
        rows[r] = X[chosen].mean(axis=0)
    return rows


# ── LIMA + centrality (port of notebook 12) ────────────────────────────────
def lima_for_celltype(X, pools, ct, slots, n_reps, rows_per_rep, rng,
                      boot_cells):
    """Mean least-squares W across n_reps batches for one cell type."""
    acc = np.zeros((N_GENES, N_GENES), dtype=np.float64)
    for rep in range(n_reps):
        A = sample_state_rows(X, pools, ct, slots[0], rows_per_rep, rng, boot_cells)
        B = sample_state_rows(X, pools, ct, slots[1], rows_per_rep, rng, boot_cells)
        W = torch.linalg.lstsq(torch.tensor(A), torch.tensor(B)).solution.numpy()
        acc += W
    return acc / n_reps


def network_centralities(adj_w, adj_mask, tf_idx, tf_names):
    """Port of the network function of notebook 12."""
    adj = (adj_w * adj_mask)[np.ix_(tf_idx, tf_idx)]
    adj_nw = adj_mask[np.ix_(tf_idx, tf_idx)]

    reg = np.unique(np.where(adj_nw)[0])
    tar = np.unique(np.where(adj_nw)[1])
    keep = np.sort(np.array(list(set(reg) & set(tar)), dtype=int))
    if len(keep) == 0:
        return pd.DataFrame()
    names = [tf_names[k] for k in keep]
    adj = adj[np.ix_(keep, keep)]

    n = adj.shape[0]
    src, dst = np.nonzero(adj)

    G_u = nx.Graph()
    for i, name in enumerate(names):
        G_u.add_node(i, name=name)
    for i, j in zip(src, dst):
        w = abs(adj[i, j])
        G_u.add_edge(j, i, weight=w, distance=1.0 / w)
    try:
        closeness = nx.closeness_centrality(G_u, distance='distance')
    except Exception:
        closeness = {i: 0.0 for i in range(n)}
    try:
        eigenvector = nx.eigenvector_centrality(G_u)
    except Exception:
        eigenvector = {i: 0.0 for i in range(n)}

    G_d = nx.MultiDiGraph()
    for i, name in enumerate(names):
        G_d.add_node(i, name=name)
    for i, j in zip(src, dst):
        G_d.add_edge(j, i, weight=adj[i, j])
    degree = nx.degree_centrality(G_d)
    # The edge of entry adj[i, j] (effect of gene i on gene j) is stored as (j, i), from the target
    # back to its regulator, as in notebook 12. The targets of a TF are therefore its incoming
    # edges in G_d and its regulators its outgoing edges. The graph is kept as it is because the
    # weighted betweenness below is not invariant to reversing edges with signed weights.
    out_c = nx.in_degree_centrality(G_d)       # number of targets the TF regulates
    in_c = nx.out_degree_centrality(G_d)       # number of regulators acting on the TF
    try:
        betweenness = nx.betweenness_centrality(G_d, weight='weight')
    except Exception:
        betweenness = {i: 0.0 for i in range(n)}

    df = pd.DataFrame({
        'degree_centrality': [degree.get(i, 0.0) for i in range(n)],
        'out_centrality': [out_c.get(i, 0.0) for i in range(n)],
        'in_centrality': [in_c.get(i, 0.0) for i in range(n)],
        'betweenness_centrality': [betweenness.get(i, 0.0) for i in range(n)],
        'closeness_centrality': [closeness.get(i, 0.0) for i in range(n)],
        'eigenvector_centrality': [eigenvector.get(i, 0.0) for i in range(n)],
    }, index=names)
    return df


def run_lima_mode(condition, transition, X, ct_names, ct_times, lineage,
                  n_reps, rows_per_rep, boot_cells, zthresh, seed, save_ctw):
    slots = TRANSITION_SLOTS[transition]
    pools = state_indices(ct_names, ct_times, lineage)
    rng = np.random.RandomState(seed)

    gene_list = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))
    wanted = pd.read_csv(os.path.join(DATA_DIR, 'Kay_TF_thalemine_annotations.csv'))
    for gid, name in NAME_PATCHES.items():
        wanted.loc[wanted['GeneID'] == gid, 'Name'] = name
    features = gene_list['features'].tolist()
    wanted_ids = set(wanted['GeneID'])
    tf_idx = np.sort(np.array(
        [i for i, g in enumerate(features) if g in wanted_ids]))
    id_to_name = {}
    for _, row in wanted.iterrows():
        id_to_name.setdefault(row['GeneID'], row['Name'])
    tf_names = [id_to_name[features[i]] for i in tf_idx]

    per_ct = {}
    ctw_store = []
    for ct in CTS:
        print(f"[{transition}] {ct}: LIMA over {n_reps} reps "
              f"(rows={rows_per_rep}, boot_cells={boot_cells})", flush=True)
        W = lima_for_celltype(X, pools, ct, slots, n_reps, rows_per_rep, rng,
                              boot_cells)
        std = W.std()
        if std == 0:
            mask = np.zeros_like(W, dtype=bool)
        else:
            z = (W - W.mean()) / std
            mask = np.abs(z) > zthresh
        df = network_centralities(W, mask, tf_idx, tf_names)
        prefix = CT_PREFIX[ct]
        df.columns = [f"{prefix}_{c}" for c in df.columns]
        per_ct[ct] = df
        if save_ctw:
            ctw_store.append(W.astype(np.float32))
        del W, mask

    # tf_occurance + concat (notebook 12 cell 33)
    tff = []
    for ct in CTS:
        df = per_ct[ct]
        col = f"{CT_PREFIX[ct]}_betweenness_centrality"
        if len(df):
            tff += df.index[df[col] > 0].tolist()
    tf_occ = pd.Series(tff).value_counts().rename('tf_occurance').to_frame()
    order = ['Atrichoblast', 'Trichoblast', 'Lateral Root Cap', 'Cortex',
             'Endodermis', 'Pericycle', 'Procambium', 'Xylem', 'Phloem',
             'Columella']
    tf_spec = pd.concat([tf_occ] + [per_ct[ct] for ct in order], axis=1)
    tf_spec = tf_spec.fillna(0)

    os.makedirs(OUT_DATA_DIR, exist_ok=True)
    out_csv = os.path.join(
        OUT_DATA_DIR, f"{condition}_TF_GRN_centrality_{transition}_zscore3.csv")
    tf_spec.to_csv(out_csv, index=True)
    print(f"Saved: {out_csv} ({tf_spec.shape})")

    if save_ctw:
        np.save(os.path.join(OUT_DATA_DIR, f"{condition}_ctw_{transition}.npy"),
                np.stack(ctw_store))


# ── DE marker table (Seurat ROC analog) ────────────────────────────────────
def auc_mann_whitney(x_in, x_out):
    """AUC via rank-sum, equivalent to Seurat's ROC AUC."""
    n1, n2 = len(x_in), len(x_out)
    if n1 == 0 or n2 == 0:
        return 0.5
    allv = np.concatenate([x_in, x_out])
    ranks = pd.Series(allv).rank().values
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    return u1 / (n1 * n2)


def run_de_mode(condition, X, ct_names, ct_times, lineage, boot_cells,
                de_cells_per_state, min_auc, seed, out_suffix='', seurat=False):
    """Build a markers-ROC table analogous to Input_X_root_markers_ROC.csv:
    clusters are {CellType}_t{bin}; per cluster, per gene: myAUC and avg_diff
    vs all other clusters; keep myAUC >= min_auc; rank within cluster
    (rank 1 = best)."""
    rng = np.random.RandomState(seed)
    gene_list = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))
    features = np.array(gene_list['features'])
    pools = state_indices(ct_names, ct_times, lineage)
    n_tb = lineage.shape[1]

    # Assemble per-cluster profile matrices (slots 1..10 -> t1..t10, plus the
    # shared Quiescent Center state as its own t0 cluster, matching the 101
    # clusters of the published marker tables)
    cluster_mats, cluster_names = [], []
    if seurat:
        # Same batch composition as the Figure 4 batch of notebook 4 (input_Y_root.csv):
        # 2,000 trajectories, the stem-cell profile of each at t0 and, for each cell type,
        # as many profiles per time bin as that batch holds (169 to 226).
        y4 = pd.read_csv(os.path.join(DATA_DIR, 'input_Y_root.csv'), index_col=0).iloc[:, 0]
        t4 = pd.read_csv(os.path.join(DATA_DIR, 'input_T_root.csv'), index_col=0).iloc[:, 0]
        n_fig4 = y4[t4 == 't1'].value_counts().to_dict()
        n_qc = int((t4 == 't0').sum())
        qc_name = 'Stem Cell_t0'
    else:
        n_fig4 = {ct: de_cells_per_state for ct in CTS}
        n_qc = de_cells_per_state
        qc_name = 'Quiescent Center_t0'
    bc = boot_cells if condition == 'pseudobulk' else None
    qc_rows = sample_state_rows(X, pools, CTS[0], 0, n_qc, rng, bc)
    if not np.allclose(qc_rows, 0):
        cluster_mats.append(qc_rows)
        cluster_names.append(qc_name)
    for ct in CTS:
        for t in range(1, n_tb):
            rows = sample_state_rows(X, pools, ct, t, int(n_fig4[ct]), rng, bc)
            if np.allclose(rows, 0):
                continue
            cluster_mats.append(rows)
            cluster_names.append(f"{ct}_t{t}")

    from scipy.stats import rankdata
    all_mat = np.vstack(cluster_mats)
    cluster_of_row = np.concatenate([
        np.full(len(m), i) for i, m in enumerate(cluster_mats)])

    if seurat:
        # Figure 4 marker criterion: the notebook call FindAllMarkers(test.use='roc',
        # only.pos=T) with Seurat defaults, ported and validated in exp35.
        from exp35_seurat_roc_markers import seurat_roc_markers
        de = seurat_roc_markers(all_mat, cluster_of_row, cluster_names, features)
        os.makedirs(OUT_DATA_DIR, exist_ok=True)
        out_csv = os.path.join(OUT_DATA_DIR, f"{condition}_markers_ROC_fig4.csv")
        de.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv} ({len(de)} rows, {de['cluster'].nunique()} clusters)")
        return

    # Because each cluster is compared against ALL other rows, the rank-sum
    # AUC can reuse one global per-gene ranking of the full matrix.
    print(f"  Ranking {all_mat.shape} ...", flush=True)
    ranks = rankdata(all_mat, axis=0)
    total_mean = all_mat.mean(axis=0)
    n_total = all_mat.shape[0]

    rows_out = []
    for ci, cname in enumerate(cluster_names):
        in_mask = cluster_of_row == ci
        n1 = int(in_mask.sum())
        n2 = n_total - n1
        r1 = ranks[in_mask].sum(axis=0)
        auc = (r1 - n1 * (n1 + 1) / 2) / (n1 * n2)
        mean_in = all_mat[in_mask].mean(axis=0)
        mean_out = (total_mean * n_total - mean_in * n1) / n2
        avg_diff = mean_in - mean_out

        keep = auc >= min_auc
        if keep.sum() == 0:
            continue
        sub = pd.DataFrame({
            'myAUC': auc[keep].round(3),
            'avg_diff': avg_diff[keep],
            'cluster': cname,
            'gene': features[keep],
        })
        sub['myAUC_rank'] = sub['myAUC'].rank(ascending=False, method='min').astype(int)
        sub['avg_diff_rank'] = sub['avg_diff'].rank(ascending=False, method='min').astype(int)
        rows_out.append(sub)
        print(f"  {cname}: {keep.sum()} markers", flush=True)

    de = pd.concat(rows_out, ignore_index=True)
    os.makedirs(OUT_DATA_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DATA_DIR, f"{condition}_markers_ROC{out_suffix}.csv")
    de.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(de)} rows, {de['cluster'].nunique()} clusters)")


def main():
    ap = argparse.ArgumentParser(description="Exp 14: new Fig 4 conditions")
    ap.add_argument('--condition', required=True,
                    choices=['pseudobulk', 'scvi', 'raw', 'genesys_gen'])
    ap.add_argument('--pool-file', default=None,
                    help='npz pool (X, ct_names, ct_times) overriding the '
                         'default data source')
    ap.add_argument('--mode', required=True, choices=['lima', 'de'])
    ap.add_argument('--transition', choices=list(TRANSITION_SLOTS))
    ap.add_argument('--scvi-file',
                    default=os.path.join(OUT_DATA_DIR, 'scvi_denoised.npz'))
    ap.add_argument('--n-reps', type=int, default=11)
    ap.add_argument('--rows-per-rep', type=int, default=200)
    ap.add_argument('--boot-cells', type=int, default=50)
    ap.add_argument('--de-cells-per-state', type=int, default=200)
    ap.add_argument('--min-auc', type=float, default=0.7)
    ap.add_argument('--zthresh', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-suffix', default='',
                    help='Suffix for the DE table name, e.g. _full for --min-auc 0 '
                         '(every gene in every cluster, no marker selection)')
    ap.add_argument('--seurat', action='store_true',
                    help='DE mode: apply the Figure 4 Seurat ROC marker criterion (exp35 port)')
    ap.add_argument('--save-ctw', action='store_true',
                    help='Also save the (10, G, G) LIMA array (float32, ~12 GB)')
    args = ap.parse_args()

    lineage = load_lineage()
    if args.pool_file:
        X, ct_names, ct_times = load_cell_pool_from_scvi(args.pool_file)
        boot = None
    elif args.condition in ('pseudobulk', 'raw'):
        X, ct_names, ct_times = load_cell_pool_from_pickle()
        boot = args.boot_cells if args.condition == 'pseudobulk' else None
    elif args.condition == 'genesys_gen':
        X, ct_names, ct_times = load_cell_pool_from_scvi(
            os.path.join(RESULTS_DIR, 'exp14_conditions', 'genesys_gen_pool.npz'))
        boot = None
    else:
        X, ct_names, ct_times = load_cell_pool_from_scvi(args.scvi_file)
        boot = None
    print(f"[Exp 14] condition={args.condition} mode={args.mode} "
          f"pool={X.shape}", flush=True)

    if args.mode == 'lima':
        if not args.transition:
            sys.exit('--transition required for lima mode')
        run_lima_mode(args.condition, args.transition, X, ct_names, ct_times,
                      lineage, args.n_reps, args.rows_per_rep, boot,
                      args.zthresh, args.seed, args.save_ctw)
    else:
        run_de_mode(args.condition, X, ct_names, ct_times, lineage,
                    args.boot_cells, args.de_cells_per_state, args.min_auc,
                    args.seed, args.out_suffix, args.seurat)
    print("[Exp 14] Done!")


if __name__ == '__main__':
    main()
