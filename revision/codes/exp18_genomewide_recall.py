#!/usr/bin/env python3
"""
Experiment 18: Genome-wide gold-standard recall for ALL Fig 4 conditions
[Round 3, R3 point 1]

R3: "Please either report Figure 4 genome-wide or justify the TF-restriction
and reconcile it with S18."

This computes the S18C metric (genome-wide recall@K over all 17,513 genes)
uniformly for every data condition and ranking method, so Figure 4 and
Figure S18 can be reconciled in one table:

  LIMA degree centrality (global z, |z|>3, notebook convention) for:
    - genesys      (genesys_ctw_*.npy, on disk)
    - scrnaseq     (genesys_raw_ctw_*.npy, on disk)
    - pseudobulk   (exp14 output, requires --save-ctw runs)
    - scvi         (exp14 output)
  DE mean-rank ranking, genome-wide (genes absent from the marker table are
  ranked last) for the same four conditions.
  Knockout perturbation ranking (exp17 output), already genome-wide.

If every method sits at chance genome-wide, the TF-restriction in Figure 4
is a property of the task (TF identity comes from domain annotation, not
expression), not a GeneSys-specific artifact.

Usage:
  sbatch --job-name=exp18_gw experiments/slurm/run_cpu.sh experiments/exp18_genomewide_recall.py
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from exp13_fig4_r50 import load_refs

DATA_DIR = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
COND_DIR = os.path.join(RESULTS_DIR, "exp14_conditions")
TRANSITIONS = ["t0-t1", "t1-t3", "t3-t5", "t5-t7", "t7-t9"]
KS = [100, 200, 300, 400]
ZTHRESH = 3.0

LIMA_SOURCES = {
    'genesys': os.path.join(DATA_DIR, "genesys_ctw_{tr}.npy"),
    'scrnaseq': os.path.join(DATA_DIR, "genesys_raw_ctw_{tr}.npy"),
    'pseudobulk': os.path.join(COND_DIR, "pseudobulk_ctw_{tr}.npy"),
    'scvi': os.path.join(COND_DIR, "scvi_ctw_{tr}.npy"),
}
DE_SOURCES = {
    # GeneSys and scRNA-seq: the Figure 4 marker tables; pseudobulk and scVI: the same
    # marker call (exp35 port) on batches of the same composition
    'genesys': os.path.join(DATA_DIR, "GeneSys_P_root_markers_ROC.csv"),
    'scrnaseq': os.path.join(DATA_DIR, "Input_X_root_markers_ROC.csv"),
    'pseudobulk': os.path.join(COND_DIR, "pseudobulk_markers_ROC_fig4.csv"),
    'scvi': os.path.join(COND_DIR, "scvi_markers_ROC_fig4.csv"),
}


def lima_genomewide_degree(path_tpl):
    """Sum unweighted degree (|z|>3, global z per cell-type slice, notebook
    convention) across all cell types and transitions."""
    deg = None
    for tr in TRANSITIONS:
        path = path_tpl.format(tr=tr)
        if not os.path.exists(path):
            print(f"  missing: {path}")
            return None
        arr = np.load(path, mmap_mode='r')
        n_ct = arr.shape[0]
        for i in range(n_ct):
            W = np.asarray(arr[i], dtype=np.float32)
            std = W.std()
            if std == 0:
                continue
            z = (W - W.mean()) / std
            mask = np.abs(z) > ZTHRESH
            d = mask.sum(axis=0) + mask.sum(axis=1)
            deg = d.astype(np.float64) if deg is None else deg + d
            del W, z, mask
        print(f"  {os.path.basename(path)} done", flush=True)
    return deg


def de_genomewide_ranking(path, gene_ids):
    """Mean myAUC per gene across clusters; genes not in the table get the
    worst score. Rank descending by mean myAUC."""
    de = pd.read_csv(path)
    agg = de.groupby('gene')['myAUC'].mean()
    scores = np.full(len(gene_ids), -1.0)
    idx = {g: i for i, g in enumerate(gene_ids)}
    for g, v in agg.items():
        if g in idx:
            scores[idx[g]] = v
    return scores


def recall_table(scores, gene_ids, gold_set):
    order = np.argsort(-scores, kind='stable')
    ranked = gene_ids[order]
    return {K: len(set(ranked[:K]) & gold_set) / len(gold_set) for K in KS}


def main():
    wanted, exptf, gold_set, stringdb = load_refs()
    gene_ids = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))['features'].values
    n_genes = len(gene_ids)

    rows = []
    for cond, tpl in LIMA_SOURCES.items():
        print(f"LIMA {cond}:", flush=True)
        deg = lima_genomewide_degree(tpl)
        if deg is None:
            continue
        rec = recall_table(deg, gene_ids, gold_set)
        rows.append({'condition': cond, 'method': 'LIMA_degree',
                     **{f'recall_at_{K}': v for K, v in rec.items()}})
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, 'exp18_genomewide_recall.csv'), index=False)

    for cond, path in DE_SOURCES.items():
        if not os.path.exists(path):
            print(f"DE {cond}: missing {path}")
            continue
        scores = de_genomewide_ranking(path, gene_ids)
        rec = recall_table(scores, gene_ids, gold_set)
        rows.append({'condition': cond, 'method': 'DE_myAUC',
                     **{f'recall_at_{K}': v for K, v in rec.items()}})

    ko_path = os.path.join(RESULTS_DIR, 'exp17_knockout', 'exp17_knockout_scores.csv')
    if os.path.exists(ko_path):
        ko = pd.read_csv(ko_path)
        for col in ['gen_effect', 'cls_effect', 'combined_effect']:
            scores = np.full(n_genes, -np.inf)
            idx = {g: i for i, g in enumerate(gene_ids)}
            for g, v in zip(ko['GeneID'], ko[col]):
                scores[idx[g]] = v
            rec = recall_table(scores, gene_ids, gold_set)
            rows.append({'condition': 'genesys', 'method': f'knockout_{col}',
                         **{f'recall_at_{K}': v for K, v in rec.items()}})

    rows.append({'condition': 'random', 'method': 'expected_by_chance',
                 **{f'recall_at_{K}': K / n_genes for K in KS}})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp18_genomewide_recall.csv'), index=False)
    print(df.to_string(index=False))
    print("[Exp 18] Done!")


if __name__ == '__main__':
    main()
