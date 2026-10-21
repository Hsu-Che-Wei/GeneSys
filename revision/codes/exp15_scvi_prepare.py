#!/usr/bin/env python3
"""
Experiment 15: scVI denoising of the root atlas [Round 3, R1 remaining point]

Reviewer 1 (round 3): "apply scVI (or DCA) purely as a denoiser, feed the
denoised expression matrices into the same TF prioritization pipeline used
for the GeneSys and raw-scRNA-seq conditions, and report R50 alongside the
existing conditions."

Trains scVI on the raw UMI counts of the root atlas
(Root_Atlas_RNA_20241022.h5ad, 110427 cells), restricted to the 17,513 genes
used throughout the manuscript, with cell labels rebuilt exactly as in
preprocessing notebook 1 (per-cell-type pseudotime qcut into 10 bins;
Quiescent Center = bin 0). Saves the denoised expression matrix
(get_normalized_expression, log1p, global min-max to [0,1] to match the
scale convention of the other conditions) for exp14 to consume.

Runs in the dedicated `scvi` conda env (see experiments/slurm/run_scvi_gpu.sh).

Usage:
  sbatch --job-name=exp15_scvi --gres=gpu:nvidia_h200:1 \
      experiments/slurm/run_scvi_gpu.sh experiments/exp15_scvi_prepare.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

ATLAS_H5AD = "/resnick/groups/NolanLab/cwhsu/scRNA-seq/Integrated_Objects/Root_Atlas_RNA_20241022.h5ad"
META_CSV = "/resnick/groups/NolanLab/cwhsu/scRNA-seq/Integrated_Objects/Root_Atlas_meta.csv"
GENE_LIST = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys/gene_list_1108.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "exp14_conditions")

CT_ORDER = ['Quiescent Center', 'Columella', 'Lateral Root Cap', 'Phloem',
            'Xylem', 'Procambium', 'Pericycle', 'Endodermis', 'Cortex',
            'Atrichoblast', 'Trichoblast']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-latent', type=int, default=30)
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'scvi_denoised.npz'))
    args = ap.parse_args()

    import scanpy as sc
    import scvi

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading atlas counts...", flush=True)
    adata = sc.read_h5ad(ATLAS_H5AD)
    print(f"  {adata.shape}", flush=True)

    meta = pd.read_csv(META_CSV, index_col=0)
    print(f"  meta: {meta.shape}, columns: {meta.columns.tolist()[:10]}", flush=True)

    # Align meta to counts by barcode
    common = adata.obs_names.intersection(meta.index)
    print(f"  common barcodes: {len(common)}", flush=True)
    adata = adata[common].copy()
    meta = meta.loc[adata.obs_names]
    adata.obs['celltype.anno'] = meta['celltype.anno'].values
    adata.obs['pseudotime'] = meta['pseudotime'].values

    # Keep only the 11 blueprint cell types (matches the cells the model
    # pipeline actually uses)
    keep = adata.obs['celltype.anno'].isin(CT_ORDER)
    adata = adata[keep].copy()
    print(f"  after cell-type filter: {adata.shape}", flush=True)

    # Rebuild time bins exactly as in preprocessing notebook 1
    tb = np.zeros(adata.n_obs)
    ct = adata.obs['celltype.anno'].values
    pt = adata.obs['pseudotime'].values.astype(float)
    for name in CT_ORDER[1:]:
        mask = ct == name
        tb[mask] = pd.qcut(pd.Series(pt[mask]), q=10,
                           labels=range(1, 11)).astype(float).to_numpy()
    tb[ct == 'Quiescent Center'] = 0.0
    adata.obs['time_bin'] = tb

    # Restrict to the manuscript's 17,513 genes
    gene_list = pd.read_csv(GENE_LIST)['features'].tolist()
    present = [g for g in gene_list if g in adata.var_names]
    missing = len(gene_list) - len(present)
    print(f"  genes: {len(present)} of {len(gene_list)} present "
          f"({missing} missing)", flush=True)
    adata = adata[:, present].copy()

    # scVI on raw counts
    scvi.model.SCVI.setup_anndata(adata, batch_key='orig.ident')
    model = scvi.model.SCVI(adata, n_latent=args.n_latent)
    print("Training scVI...", flush=True)
    model.train(max_epochs=args.max_epochs)
    print("Training done. Getting normalized expression...", flush=True)

    denoised = model.get_normalized_expression(library_size=1e4)
    X = np.log1p(np.asarray(denoised, dtype=np.float32))
    X = (X - X.min()) / (X.max() - X.min())

    # Reindex to the full 17,513-gene order (missing genes = zero columns)
    if missing:
        full = np.zeros((X.shape[0], len(gene_list)), dtype=np.float32)
        col_idx = {g: i for i, g in enumerate(gene_list)}
        for j, g in enumerate(present):
            full[:, col_idx[g]] = X[:, j]
        X = full

    np.savez_compressed(
        args.out,
        X=X,
        ct_names=adata.obs['celltype.anno'].values.astype(str),
        ct_times=adata.obs['time_bin'].values.astype(np.float32),
        barcodes=adata.obs_names.values.astype(str),
        n_missing_genes=missing,
    )
    print(f"Saved: {args.out} ({X.shape})", flush=True)

    hist = model.history.get('elbo_train')
    if hist is not None:
        hist.to_csv(os.path.join(OUT_DIR, 'scvi_elbo_train.csv'))
    print("[Exp 15] Done!")


if __name__ == '__main__':
    main()
