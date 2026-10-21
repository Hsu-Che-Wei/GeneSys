#!/usr/bin/env python3
"""
Experiment 5: LIMA Z-Score Threshold Sensitivity [R3.11]

Re-threshold existing LIMA matrices at different |z| cutoffs and evaluate
how TF recovery changes. No retraining needed — purely computational.

The LIMA matrices are gene×gene interaction matrices derived from
GeneSys-generated state transcriptomes via least-squares (AW=B).

Usage:
  sbatch --job-name=exp5_zscore experiments/slurm/run_cpu.sh experiments/exp5_zscore_sensitivity.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import RESULTS_DIR, save_figure, NUM2NAME

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
TABLES_DIR = "/resnick/groups/NolanLab/cwhsu/GeneSys_Revision/Tables"

# LIMA transition files
TRANSITIONS = [
    ("t0", "t1"),
    ("t1", "t3"),
    ("t3", "t5"),
    ("t5", "t7"),
    ("t7", "t9"),
]

# Gold-standard TFs from GitHub repo supplementary data
REPO_DIR = os.path.join(os.path.dirname(__file__), "GeneSys")
GOLD_STANDARD_MINIEX = os.path.join(REPO_DIR, "supp_data", "gold_standard_for_benchmarking",
                                     "Gold_Standard_Root_TF_MINI_EX.csv")
GOLD_STANDARD_STRINGDB = os.path.join(REPO_DIR, "supp_data", "gold_standard_for_benchmarking",
                                       "Gold_Standard_Root_TF_StringDB.csv")

# Cell types for which we have TF centrality results
CELL_TYPES = [
    "Atrichoblast", "Columella", "Cortex", "Endodermis",
    "Ground_Tissue", "LRC", "Pericycle", "Phloem",
    "Procambium", "Root_Cap", "Stele", "Trichoblast", "Xylem",
]


def load_lima_matrix(t_from, t_to):
    """Load a LIMA (gene×gene interaction) matrix for a temporal transition."""
    fname = f"genesys_ctw_{t_from}-{t_to}.npy"
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    print(f"  Loading {fname}...")
    return np.load(path)


def apply_zscore_threshold(W, threshold):
    """Z-score thresholding of one cell type's gene-by-gene LIMA, as in notebook 12
    (12-GeneSys_Find_Top_Regulators_*, "Calculate z-scores"): one mean and one standard
    deviation over all entries of the matrix, entries with |z| <= threshold set to zero.

    Returns: thresholded matrix, number of non-zero edges
    """
    W_thresh = W.astype(np.float32).copy()
    std = W_thresh.std()
    if std > 0:
        z = (W_thresh - W_thresh.mean()) / std
        W_thresh[~(np.abs(z) > threshold)] = 0
        del z
    else:
        W_thresh[:] = 0

    n_edges = np.count_nonzero(W_thresh)
    return W_thresh, n_edges


def compute_tf_centrality(W_thresh, gene_names=None):
    """
    Build a directed graph from thresholded W and compute centrality metrics.

    W[i,j] = influence of gene j on gene i.
    Edge: j → i (j regulates i).

    Returns DataFrame with centrality scores per gene.
    """
    n_genes = W_thresh.shape[0]

    # Build directed graph
    G = nx.DiGraph()
    for i in range(n_genes):
        for j in range(n_genes):
            if W_thresh[i, j] != 0:
                G.add_edge(j, i, weight=abs(W_thresh[i, j]))

    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    # Compute centralities
    dc = nx.degree_centrality(G)
    try:
        bc = nx.betweenness_centrality(G, weight='weight')
    except Exception:
        bc = {n: 0.0 for n in G.nodes()}
    try:
        cc = nx.closeness_centrality(G)
    except Exception:
        cc = {n: 0.0 for n in G.nodes()}

    df = pd.DataFrame({
        'gene_idx': list(G.nodes()),
        'degree_centrality': [dc.get(n, 0) for n in G.nodes()],
        'betweenness_centrality': [bc.get(n, 0) for n in G.nodes()],
        'closeness_centrality': [cc.get(n, 0) for n in G.nodes()],
    })

    # Composite score (mean of normalized centralities)
    for col in ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']:
        max_val = df[col].max()
        if max_val > 0:
            df[f'{col}_norm'] = df[col] / max_val
        else:
            df[f'{col}_norm'] = 0.0

    df['composite_centrality'] = (
        df['degree_centrality_norm'] +
        df['betweenness_centrality_norm'] +
        df['closeness_centrality_norm']
    ) / 3.0

    df = df.sort_values('composite_centrality', ascending=False).reset_index(drop=True)

    if gene_names is not None:
        df['gene_name'] = df['gene_idx'].map(
            lambda x: gene_names[x] if x < len(gene_names) else f"gene_{x}"
        )

    return df


def load_gold_standard_tfs():
    """
    Load gold-standard TFs from GitHub repo supplementary data.
    Combines MINI-EX and StringDB gold standards.
    """
    gs_tfs = set()

    # MINI-EX gold standard (just gene IDs)
    if os.path.exists(GOLD_STANDARD_MINIEX):
        df = pd.read_csv(GOLD_STANDARD_MINIEX)
        gs_tfs.update(df['GeneID'].dropna().tolist())
        print(f"  MINI-EX gold standard: {len(df)} TFs")

    # StringDB gold standard (gene IDs + annotations)
    if os.path.exists(GOLD_STANDARD_STRINGDB):
        df = pd.read_csv(GOLD_STANDARD_STRINGDB)
        gs_tfs.update(df['GeneID'].dropna().tolist())
        print(f"  StringDB gold standard: {len(df)} TFs")

    return gs_tfs


def recall_at_k(ranked_genes, gold_set, k=50):
    """Compute recall@k: fraction of gold-standard TFs found in top k."""
    if len(gold_set) == 0:
        return 0.0
    top_k = set(ranked_genes[:k])
    found = top_k & gold_set
    return len(found) / min(len(gold_set), k)


def main():
    parser = argparse.ArgumentParser(description="Exp 5: Z-score threshold sensitivity")
    parser.add_argument("--thresholds", nargs='+', type=float,
                        default=[2.0, 2.5, 3.0, 4.0])
    args = parser.parse_args()

    thresholds = args.thresholds
    print(f"[Exp 5] Thresholds: {thresholds}")

    # Check which LIMA matrices exist
    available_transitions = []
    for t_from, t_to in TRANSITIONS:
        path = os.path.join(DATA_DIR, f"genesys_ctw_{t_from}-{t_to}.npy")
        if os.path.exists(path):
            available_transitions.append((t_from, t_to))
            print(f"  Found: genesys_ctw_{t_from}-{t_to}.npy")
        else:
            print(f"  Missing: genesys_ctw_{t_from}-{t_to}.npy")

    if not available_transitions:
        print("ERROR: No LIMA matrices found. Exiting.")
        sys.exit(1)

    # Load gold-standard TFs
    print("\nLoading gold-standard TFs from existing analysis...")
    gs_tfs = load_gold_standard_tfs()
    print(f"  Gold-standard TF set: {len(gs_tfs)} TFs")

    # ── Main analysis ─────────────────────────────────────────────────────
    all_results = []
    edge_counts = []

    for t_from, t_to in available_transitions:
        print(f"\n=== Transition: {t_from} → {t_to} ===")
        W = load_lima_matrix(t_from, t_to)
        if W is None:
            continue

        print(f"  Raw matrix shape: {W.shape}")

        # Handle 3D matrices: (n_celltypes, n_genes, n_genes)
        if W.ndim == 3:
            n_ct, n_genes, n_genes2 = W.shape
            assert n_genes == n_genes2, f"Expected square gene matrices, got {W.shape}"
            print(f"  3D matrix: {n_ct} cell types × {n_genes} × {n_genes}")
        else:
            n_genes = W.shape[0]
            n_ct = 1
            W = W[np.newaxis, ...]  # make 3D for uniform handling

        for threshold in thresholds:
            print(f"\n  Threshold |z| > {threshold}")
            total_edges = 0
            total_possible = 0

            for ct_idx in range(n_ct):
                W_ct = W[ct_idx]
                W_thresh, n_edges = apply_zscore_threshold(W_ct, threshold)
                total_edges += n_edges
                total_possible += n_genes ** 2

                # Use degree proxy for large matrices (networkx too slow)
                gene_importance = np.sum(np.abs(W_thresh), axis=0) + np.sum(np.abs(W_thresh), axis=1)
                ranked_idx = np.argsort(gene_importance)[::-1]

                all_results.append({
                    'transition': f"{t_from}-{t_to}",
                    'cell_type_idx': ct_idx,
                    'threshold': threshold,
                    'metric': 'degree_proxy',
                    'top1_gene_idx': int(ranked_idx[0]),
                    'top1_score': float(gene_importance[ranked_idx[0]]),
                    'n_edges': n_edges,
                })

            frac = total_edges / total_possible if total_possible > 0 else 0
            print(f"    Total edges: {total_edges:,} / {total_possible:,} ({100*frac:.2f}%)")

            edge_counts.append({
                'transition': f"{t_from}-{t_to}",
                'threshold': threshold,
                'n_edges': total_edges,
                'n_possible': total_possible,
                'fraction': frac,
            })

    # ── Save results ──────────────────────────────────────────────────────
    df_edges = pd.DataFrame(edge_counts)
    csv_path = os.path.join(RESULTS_DIR, "exp5_zscore_edge_counts.csv")
    df_edges.to_csv(csv_path, index=False)
    print(f"\nSaved edge counts to {csv_path}")

    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "exp5_zscore_sensitivity.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # ── Plot: Edge retention vs threshold ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Edge count vs threshold
    ax = axes[0]
    for trans_name, group in df_edges.groupby('transition'):
        ax.plot(group['threshold'], group['n_edges'] / 1e6,
                'o-', linewidth=2, markersize=8, label=trans_name)
    ax.set_xlabel('Z-score threshold', fontsize=13)
    ax.set_ylabel('Non-zero edges (millions)', fontsize=13)
    ax.set_title('Network Density vs Z-Score Threshold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Fraction of edges retained
    ax = axes[1]
    for trans_name, group in df_edges.groupby('transition'):
        ax.plot(group['threshold'], group['fraction'] * 100,
                'o-', linewidth=2, markersize=8, label=trans_name)
    ax.set_xlabel('Z-score threshold', fontsize=13)
    ax.set_ylabel('Edges retained (%)', fontsize=13)
    ax.set_title('Fraction of Regulatory Edges Retained', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "exp5_zscore_sensitivity")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Summary: Edge counts by threshold ===")
    print(df_edges.pivot_table(index='transition', columns='threshold',
                               values='n_edges', aggfunc='mean').to_string())


if __name__ == "__main__":
    main()
