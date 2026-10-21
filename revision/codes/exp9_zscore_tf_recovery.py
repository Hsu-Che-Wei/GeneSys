#!/usr/bin/env python3
"""
Experiment 9: Gold-Standard TF Recovery at Different Z-Score Thresholds [R3.11]

Re-threshold existing LIMA matrices at different |z| cutoffs and measure how
many gold-standard TFs (from MINI-EX and StringDB) are recovered in the top-K
ranked genes by degree centrality.

The LIMA matrices are 3D (n_celltypes x n_genes x n_genes), ~24 GB each.
We use memory-mapped loading and process one cell-type slice at a time
to avoid out-of-memory issues.

Usage:
  python experiments/exp9_zscore_tf_recovery.py
  python experiments/exp9_zscore_tf_recovery.py --outdir experiments/results
  sbatch --job-name=exp9 experiments/slurm/run_cpu.sh experiments/exp9_zscore_tf_recovery.py
"""
import os, sys, argparse, gc
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import RESULTS_DIR, save_figure

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
TABLES_DIR = "/resnick/groups/NolanLab/cwhsu/GeneSys_Revision/Tables/Sub_tables"
GENE_LIST_CSV = os.path.join(DATA_DIR, "gene_list_1108.csv")

# Gold-standard TFs
REPO_DIR = os.path.join(os.path.dirname(__file__), "GeneSys")
GOLD_STANDARD_MINIEX = os.path.join(
    REPO_DIR, "supp_data", "gold_standard_for_benchmarking",
    "Gold_Standard_Root_TF_MINI_EX.csv"
)
GOLD_STANDARD_STRINGDB = os.path.join(
    REPO_DIR, "supp_data", "gold_standard_for_benchmarking",
    "Gold_Standard_Root_TF_StringDB.csv"
)

# LIMA transition files
TRANSITIONS = [
    ("t0", "t1"),
    ("t1", "t3"),
    ("t3", "t5"),
    ("t5", "t7"),
    ("t7", "t9"),
]

# Cell type ordering in the LIMA 3D array (dim 0), matching the original
# notebook that generated the ctw matrices
CELL_TYPES = [
    "Atrichoblast", "Trichoblast", "Cortex", "Endodermis",
    "Pericycle", "Procambium", "Xylem", "Phloem",
    "Lateral Root Cap", "Columella",
]

# Default thresholds and top-K values
DEFAULT_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]
DEFAULT_TOP_K = [100, 200, 300, 400]


def load_gene_names():
    """Load gene names from gene_list_1108.csv (17513 genes).

    Returns a list of gene IDs (e.g. 'AT1G05260') indexed by position,
    matching the LIMA matrix row/column order.
    """
    df = pd.read_csv(GENE_LIST_CSV, index_col=0)
    gene_names = df['features'].tolist()
    print(f"  Loaded {len(gene_names)} gene names from {os.path.basename(GENE_LIST_CSV)}")
    return gene_names


def load_gold_standard_tfs():
    """Load and combine gold-standard TF sets from MINI-EX and StringDB.

    Returns:
        gs_miniex: set of GeneIDs from MINI-EX
        gs_stringdb: set of GeneIDs from StringDB
        gs_union: combined set
    """
    gs_miniex = set()
    gs_stringdb = set()

    if os.path.exists(GOLD_STANDARD_MINIEX):
        df = pd.read_csv(GOLD_STANDARD_MINIEX)
        gs_miniex = set(df['GeneID'].dropna().tolist())
        print(f"  MINI-EX gold standard: {len(gs_miniex)} TFs")
    else:
        print(f"  WARNING: {GOLD_STANDARD_MINIEX} not found")

    if os.path.exists(GOLD_STANDARD_STRINGDB):
        df = pd.read_csv(GOLD_STANDARD_STRINGDB)
        gs_stringdb = set(df['GeneID'].dropna().tolist())
        print(f"  StringDB gold standard: {len(gs_stringdb)} TFs")
    else:
        print(f"  WARNING: {GOLD_STANDARD_STRINGDB} not found")

    gs_union = gs_miniex | gs_stringdb
    gs_intersection = gs_miniex & gs_stringdb
    print(f"  Union gold standard: {len(gs_union)} unique TFs")
    print(f"  Intersection gold standard: {len(gs_intersection)} TFs (validated by both sources)")
    return gs_miniex, gs_stringdb, gs_union, gs_intersection


def apply_zscore_threshold_row(W_slice, threshold):
    """Z-score thresholding of one cell type's gene-by-gene LIMA, as in notebook 12
    (12-GeneSys_Find_Top_Regulators_*, "Calculate z-scores"): one mean and one standard
    deviation over all entries of the matrix, entries with |z| <= threshold set to zero.
    (The function keeps its historical name; it no longer works row by row.)

    Returns: thresholded matrix (float32)
    """
    W = W_slice.astype(np.float32)
    std = W.std()
    if std > 0:
        z = (W - W.mean()) / std
        W[~(np.abs(z) > threshold)] = 0.0
        del z
    else:
        W[:] = 0.0
    return W


def compute_degree_centrality(W_thresh):
    """Compute degree centrality for each gene from a thresholded matrix.

    W[i,j] represents influence of gene j on gene i (j -> i).
    - Out-degree of gene j: sum of |W[:,j]| (column sum) = how much j influences others
    - In-degree of gene i: sum of |W[i,:]| (row sum) = how much others influence i
    - Total degree: out-degree + in-degree

    Returns: 1D array of total degree centrality per gene
    """
    abs_W = np.abs(W_thresh)
    out_degree = abs_W.sum(axis=0)  # column sums
    in_degree = abs_W.sum(axis=1)   # row sums
    total_degree = out_degree + in_degree
    return total_degree


def recall_at_k(ranked_gene_ids, gold_set, k):
    """Compute recall at top-K: fraction of gold-standard TFs found in top k genes."""
    if len(gold_set) == 0:
        return 0.0, 0
    top_k = set(ranked_gene_ids[:k])
    n_found = len(top_k & gold_set)
    recall = n_found / len(gold_set)
    return recall, n_found


def precision_at_k(ranked_gene_ids, gold_set, k):
    """Compute precision at top-K: fraction of top-k genes that are gold-standard TFs."""
    if k == 0:
        return 0.0
    top_k = set(ranked_gene_ids[:k])
    n_found = len(top_k & gold_set)
    return n_found / k


def process_transition(t_from, t_to, gene_names, gold_sets, thresholds, top_ks):
    """Process a single LIMA transition file across all thresholds and top-K values.

    Uses memory-mapped loading to avoid loading the full ~24 GB array into RAM.
    Processes one cell-type slice at a time.

    Returns: list of result dicts
    """
    fname = f"genesys_ctw_{t_from}-{t_to}.npy"
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return []

    transition_name = f"{t_from}-{t_to}"
    print(f"\n{'='*60}")
    print(f"Transition: {transition_name}")
    print(f"{'='*60}")

    # Memory-mapped load
    print(f"  Loading {fname} (memory-mapped)...")
    W_all = np.load(path, mmap_mode='r')
    print(f"  Shape: {W_all.shape}, dtype: {W_all.dtype}")

    if W_all.ndim == 3:
        n_ct, n_genes, n_genes2 = W_all.shape
        assert n_genes == n_genes2, f"Expected square gene matrices, got {W_all.shape}"
    elif W_all.ndim == 2:
        n_genes = W_all.shape[0]
        n_ct = 1
        W_all = W_all[np.newaxis, ...]
    else:
        print(f"  ERROR: Unexpected dimensionality {W_all.ndim}")
        return []

    gs_miniex, gs_stringdb, gs_union, gs_intersection = gold_sets
    results = []

    for threshold in thresholds:
        print(f"\n  Threshold |z| > {threshold}")

        # Accumulate degree centrality across all cell types for aggregate ranking
        aggregate_centrality = np.zeros(n_genes)

        for ct_idx in range(n_ct):
            ct_name = CELL_TYPES[ct_idx] if ct_idx < len(CELL_TYPES) else f"ct_{ct_idx}"

            # Load one cell-type slice as float32 (~1.2 GB instead of 2.3 GB)
            W_ct = np.array(W_all[ct_idx], dtype=np.float32)

            # Apply z-score threshold (returns float32, ~1.2 GB)
            W_thresh = apply_zscore_threshold_row(W_ct, threshold)
            del W_ct  # free immediately
            n_edges = np.count_nonzero(W_thresh)

            # Compute degree centrality
            centrality = compute_degree_centrality(W_thresh)
            del W_thresh  # free immediately
            gc.collect()
            aggregate_centrality += centrality

            # Rank genes for this cell type
            ranked_idx = np.argsort(centrality)[::-1]
            ranked_gene_ids = [gene_names[idx] for idx in ranked_idx]

            # Evaluate recovery at each top-K
            for k in top_ks:
                for gs_name, gs_set in [("MINI-EX", gs_miniex),
                                         ("StringDB", gs_stringdb),
                                         ("Union", gs_union),
                                         ("Intersection", gs_intersection)]:
                    rec, n_found = recall_at_k(ranked_gene_ids, gs_set, k)
                    prec = precision_at_k(ranked_gene_ids, gs_set, k)
                    results.append({
                        'transition': transition_name,
                        'cell_type': ct_name,
                        'threshold': threshold,
                        'K': k,
                        'gold_standard': gs_name,
                        'n_gold_total': len(gs_set),
                        'n_gold_recovered': n_found,
                        'recall': rec,
                        'precision': prec,
                        'n_edges': n_edges,
                    })

        # Also compute aggregate (all cell types combined) ranking
        ranked_idx = np.argsort(aggregate_centrality)[::-1]
        ranked_gene_ids = [gene_names[idx] for idx in ranked_idx]

        for k in top_ks:
            for gs_name, gs_set in [("MINI-EX", gs_miniex),
                                     ("StringDB", gs_stringdb),
                                     ("Union", gs_union),
                                     ("Intersection", gs_intersection)]:
                rec, n_found = recall_at_k(ranked_gene_ids, gs_set, k)
                prec = precision_at_k(ranked_gene_ids, gs_set, k)
                results.append({
                    'transition': transition_name,
                    'cell_type': 'ALL',
                    'threshold': threshold,
                    'K': k,
                    'gold_standard': gs_name,
                    'n_gold_total': len(gs_set),
                    'n_gold_recovered': n_found,
                    'recall': rec,
                    'precision': prec,
                    'n_edges': -1,  # aggregate, not meaningful
                })

    return results


def make_summary_plot(df_results, outdir):
    """Create summary plots of TF recovery vs z-score threshold."""
    # Filter to aggregate (ALL cell types) and Intersection gold standard (140 TFs)
    df_agg = df_results[
        (df_results['cell_type'] == 'ALL') &
        (df_results['gold_standard'] == 'Intersection')
    ].copy()

    if df_agg.empty:
        print("  No aggregate results to plot.")
        return

    transitions = df_agg['transition'].unique()
    ks = sorted(df_agg['K'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Recall vs threshold for different K, averaged over transitions
    ax = axes[0]
    for k in ks:
        sub = df_agg[df_agg['K'] == k]
        avg = sub.groupby('threshold')['recall'].mean()
        ax.plot(avg.index, avg.values, 'o-', linewidth=2, markersize=7, label=f'K={k}')
    ax.set_xlabel('Z-score threshold', fontsize=13)
    ax.set_ylabel('Recall (fraction of gold TFs recovered)', fontsize=13)
    ax.set_title('Gold-Standard TF Recovery vs Z-Score Threshold\n(averaged across transitions)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Recall vs threshold for different transitions at K=140
    ax = axes[1]
    target_k = 140 if 140 in ks else ks[len(ks)//2]
    sub = df_agg[df_agg['K'] == target_k]
    for trans in transitions:
        tsub = sub[sub['transition'] == trans]
        ax.plot(tsub['threshold'], tsub['recall'], 'o-', linewidth=2, markersize=7,
                label=trans)
    ax.set_xlabel('Z-score threshold', fontsize=13)
    ax.set_ylabel(f'Recall at K={target_k}', fontsize=13)
    ax.set_title(f'TF Recovery by Transition (K={target_k})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "exp9_zscore_tf_recovery")

    # Panel 3: Per-gold-standard comparison
    df_agg2 = df_results[df_results['cell_type'] == 'ALL'].copy()
    if df_agg2.empty:
        return

    fig2, axes2 = plt.subplots(1, len(ks), figsize=(5 * len(ks), 5), squeeze=False)
    for ki, k in enumerate(ks):
        ax = axes2[0, ki]
        sub = df_agg2[df_agg2['K'] == k]
        for gs_name in ['MINI-EX', 'StringDB', 'Union']:
            gs_sub = sub[sub['gold_standard'] == gs_name]
            avg = gs_sub.groupby('threshold')['recall'].mean()
            ax.plot(avg.index, avg.values, 'o-', linewidth=2, markersize=7, label=gs_name)
        ax.set_xlabel('Z-score threshold', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title(f'K = {k}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig2.suptitle('TF Recovery by Gold Standard Source', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig2, "exp9_zscore_tf_recovery_by_source")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 9: Gold-standard TF recovery at different z-score thresholds"
    )
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for CSV and figures (default: experiments/results)")
    parser.add_argument("--thresholds", nargs='+', type=float,
                        default=DEFAULT_THRESHOLDS,
                        help="Z-score thresholds to test")
    parser.add_argument("--top-k", nargs='+', type=int,
                        default=DEFAULT_TOP_K,
                        help="Top-K values for recall/precision")
    parser.add_argument("--transitions", nargs='+', type=str, default=None,
                        help="Specific transitions to process (e.g. t0-t1 t1-t3)")
    args = parser.parse_args()

    # Set output directory
    outdir = args.outdir if args.outdir else RESULTS_DIR
    os.makedirs(outdir, exist_ok=True)

    thresholds = sorted(args.thresholds)
    top_ks = sorted(args.top_k)

    print("=" * 60)
    print("Experiment 9: Gold-Standard TF Recovery vs Z-Score Threshold")
    print("=" * 60)
    print(f"  Thresholds: {thresholds}")
    print(f"  Top-K values: {top_ks}")
    print(f"  Output dir: {outdir}")

    # Load gene names
    print("\nLoading gene names...")
    gene_names = load_gene_names()

    # Load gold-standard TFs
    print("\nLoading gold-standard TFs...")
    gs_miniex, gs_stringdb, gs_union, gs_intersection = load_gold_standard_tfs()
    gold_sets = (gs_miniex, gs_stringdb, gs_union, gs_intersection)

    # Check overlap between gold standard and gene list
    gene_set = set(gene_names)
    miniex_in_data = gs_miniex & gene_set
    stringdb_in_data = gs_stringdb & gene_set
    union_in_data = gs_union & gene_set
    intersection_in_data = gs_intersection & gene_set
    print(f"\n  Gold TFs present in gene list:")
    print(f"    MINI-EX:       {len(miniex_in_data)} / {len(gs_miniex)}")
    print(f"    StringDB:      {len(stringdb_in_data)} / {len(gs_stringdb)}")
    print(f"    Union:         {len(union_in_data)} / {len(gs_union)}")
    print(f"    Intersection:  {len(intersection_in_data)} / {len(gs_intersection)}")

    # Determine which transitions to process
    if args.transitions:
        transitions = []
        for t in args.transitions:
            parts = t.split('-')
            transitions.append((parts[0], parts[1]))
    else:
        transitions = TRANSITIONS

    # Check availability
    available = []
    for t_from, t_to in transitions:
        path = os.path.join(DATA_DIR, f"genesys_ctw_{t_from}-{t_to}.npy")
        if os.path.exists(path):
            available.append((t_from, t_to))
            print(f"  Found: genesys_ctw_{t_from}-{t_to}.npy")
        else:
            print(f"  Missing: genesys_ctw_{t_from}-{t_to}.npy")

    if not available:
        print("\nERROR: No LIMA matrices found. Exiting.")
        sys.exit(1)

    # Process each transition
    all_results = []
    for t_from, t_to in available:
        results = process_transition(t_from, t_to, gene_names, gold_sets,
                                     thresholds, top_ks)
        all_results.extend(results)

    # Build results DataFrame
    df_results = pd.DataFrame(all_results)

    # Save full results
    csv_path = os.path.join(outdir, "exp9_zscore_tf_recovery.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved full results to {csv_path}")
    print(f"  {len(df_results)} rows")

    # Save aggregate-only summary (ALL cell types)
    df_agg = df_results[df_results['cell_type'] == 'ALL'].copy()
    csv_agg_path = os.path.join(outdir, "exp9_zscore_tf_recovery_aggregate.csv")
    df_agg.to_csv(csv_agg_path, index=False)
    print(f"Saved aggregate summary to {csv_agg_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Gold-Standard TF Recovery (Union, ALL cell types)")
    print("=" * 80)
    df_summary = df_agg[df_agg['gold_standard'] == 'Intersection'].pivot_table(
        index=['transition', 'threshold'],
        columns='K',
        values='n_gold_recovered',
        aggfunc='first'
    )
    print(df_summary.to_string())

    print("\n" + "-" * 80)
    print("RECALL (Intersection gold standard = 140 TFs, ALL cell types)")
    print("-" * 80)
    df_recall = df_agg[df_agg['gold_standard'] == 'Intersection'].pivot_table(
        index=['transition', 'threshold'],
        columns='K',
        values='recall',
        aggfunc='first'
    )
    print(df_recall.round(4).to_string())

    # Average recall across transitions
    print("\n" + "-" * 80)
    print("AVERAGE RECALL across transitions (Intersection gold standard = 140 TFs)")
    print("-" * 80)
    df_avg = df_agg[df_agg['gold_standard'] == 'Intersection'].groupby(
        ['threshold', 'K']
    )['recall'].mean().unstack('K')
    print(df_avg.round(4).to_string())

    # Generate plots
    print("\nGenerating plots...")
    make_summary_plot(df_results, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
