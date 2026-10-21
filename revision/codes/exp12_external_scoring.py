#!/usr/bin/env python3
"""
Experiment 12: External Scoring of Generated Transcriptomes [Round 3, R3 point 3]

Round 3 R3 objected that autoregressive generation accuracy is "your own
classifier scoring your own generated cells with no external reference."
This experiment scores the generated transcriptomes with references that are
independent of the GeneSys classifier:

  1. Centroid classifier: cosine similarity to per-(cell type, time bin)
     centroids computed from REAL training cells only. A generated
     transcriptome at bin t of trajectory j is scored correct if its nearest
     real centroid at time t matches the blueprint cell type lineage[j, t].
  2. Logistic-regression classifier trained on real training cells
     (cell-type labels), applied to generated transcriptomes.
  3. Cosine similarity of each generated transcriptome to its matching real
     centroid, and the margin over the best non-matching centroid.

Models scored: the published 600-epoch checkpoint, the exp1 decoy models
(correct / reversed / shuffled / random), and the exp6 ablations
(full / no_lstm / no_tdvae / no_blueprint). All generation follows the
Figure S7 autoregressive protocol (seed t0-t1 real, generate t2..t10).

Usage:
  sbatch --job-name=exp12_ext experiments/slurm/run_cpu.sh experiments/exp12_external_scoring.py
"""
import os, sys, json, glob, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint

from utils import (load_root_data, load_root_data_blueprint, load_lineage, load_checkpoint, make_model,
                   RESULTS_DIR, ROOT_CKPT, save_figure)
from train_utils import make_model_custom
from exp6_ablation import ClassifierMLP
from eval_recreation_accuracy import get_checkpoint_path

OUT_DIR = os.path.join(RESULTS_DIR, "exp12_external")

MODELS = [
    # (label, experiment, condition)
    ('published_600ep', 'direct', None),
    ('exp1_correct', 'exp1', 'correct'),
    ('exp1_reversed', 'exp1', 'reversed'),
    ('exp1_shuffled', 'exp1', 'shuffled'),
    ('exp1_random', 'exp1', 'random'),
    ('exp6_full', 'exp6', 'full'),
    ('exp6_no_lstm', 'exp6', 'no_lstm'),
    ('exp6_no_tdvae', 'exp6', 'no_tdvae'),
    ('exp6_no_blueprint', 'exp6', 'no_blueprint'),
]


def build_centroids(X, labels, n_tb):
    """Per-(cell type, time bin) centroids from real cells.
    Labels are '{type}_{time}' with time = bin index + 1 (float)."""
    ct_names = np.array([lab.split('_', 1)[0] for lab in labels])
    ct_times = np.array([float(lab.split('_', 1)[1]) for lab in labels])
    centroids = {}   # (type, bin) -> vector
    for t in range(n_tb):
        time_val = float(t + 1)
        for ct in np.unique(ct_names):
            mask = (ct_names == ct) & (ct_times == time_val)
            if mask.sum() >= 5:
                centroids[(ct, t)] = X[mask].mean(axis=0)
    return centroids


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def generate_autoregressive(model, loader, device, time_bins, batch_size,
                            n_batches):
    """Figure S7 autoregressive protocol, capturing generated transcriptomes.
    Returns: list of (traj_labels[batch], gen[t] arrays for t=2..time_bins-1)."""
    model.eval()
    captured = []
    with torch.no_grad():
        count = 0
        for sample in loader:
            if count >= n_batches:
                break
            x_real = sample['x'].to(device)
            y_true = sample['y'].cpu().numpy()
            if x_real.size(0) != batch_size:
                continue

            x = torch.stack(
                [x_real[:, 0, :], x_real[:, 1, :]] +
                [x_real[:, 2, :]] * (time_bins - 2), dim=1)
            h = model.init_hidden(batch_size)

            t_gen = [None] * time_bins
            t_gen[0] = model.generate_current(x, h, 0)
            t_gen[1] = model.generate_next(x, h, 0)
            _, h = model.predict_proba(x, h, 1)
            t_gen[2] = model.generate_next(x, h, 1)
            x = torch.stack([t_gen[0], t_gen[1], t_gen[2]] +
                            [t_gen[2]] * (time_bins - 3), dim=1)
            _, h = model.predict_proba(x, h, 2)

            for t in range(3, time_bins):
                t_gen[t] = model.generate_next(x, h, t - 1)
                components = [t_gen[k] for k in range(t + 1)]
                padding = [t_gen[t]] * (time_bins - t - 1)
                x = torch.stack(components + padding, dim=1)
                _, h = model.predict_proba(x, h, t)

            gen_np = {t: t_gen[t].cpu().numpy() for t in range(2, time_bins)}
            captured.append((y_true, gen_np))
            count += 1
    return captured


def score_generated(captured, lineage, centroids, lr_clf, n_tb):
    """Score generated transcriptomes against external references."""
    rows = []
    for y_true, gen_np in captured:
        for t in sorted(gen_np.keys()):
            G = gen_np[t]  # (batch, genes)
            # Blueprint label for each sample at this bin
            bp_labels = np.array([lineage.iloc[j, t] for j in y_true])

            # Centroid classifier at bin t
            cts = [ct for (ct, tb) in centroids if tb == t]
            if not cts:
                continue
            C = np.stack([centroids[(ct, t)] for ct in cts])  # (n_ct, genes)
            Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
            sims = Gn @ Cn.T                                  # (batch, n_ct)
            pred_ct = np.array(cts)[sims.argmax(axis=1)]
            centroid_acc = float((pred_ct == bp_labels).mean())

            # Cosine to matching centroid + margin
            match_sims, margins = [], []
            for i, bl in enumerate(bp_labels):
                if (bl, t) not in centroids:
                    continue
                ci = cts.index(bl)
                ms = sims[i, ci]
                others = np.delete(sims[i], ci)
                match_sims.append(ms)
                margins.append(ms - others.max() if len(others) else 0.0)

            # Logistic regression classifier (type-level)
            lr_acc = np.nan
            if lr_clf is not None:
                lr_pred = lr_clf.predict(G)
                lr_acc = float((lr_pred == bp_labels).mean())

            rows.append({
                'time_bin': t,
                'centroid_acc': centroid_acc,
                'lr_acc': lr_acc,
                'mean_cosine_to_match': float(np.mean(match_sims)) if match_sims else np.nan,
                'mean_margin': float(np.mean(margins)) if margins else np.nan,
                'n_samples': len(y_true),
            })
    df = pd.DataFrame(rows)
    return df.groupby('time_bin').agg(
        centroid_acc=('centroid_acc', 'mean'),
        lr_acc=('lr_acc', 'mean'),
        mean_cosine_to_match=('mean_cosine_to_match', 'mean'),
        mean_margin=('mean_margin', 'mean'),
    ).reset_index()


def load_model_for(label, experiment, condition, n_genes, n_traj, device):
    if experiment == 'direct':
        model = make_model(n_genes, output_size=n_traj, device=device)
        return load_checkpoint(model, ROOT_CKPT, device=device)
    ckpt_path = get_checkpoint_path(experiment, condition)
    if experiment == 'exp6' and condition == 'no_lstm':
        model = ClassifierMLP(n_genes, n_traj, 256, 256, 2, device).to(device)
    else:
        model = make_model_custom(n_genes, n_traj, device, sigma=0.2)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Exp 12: External scoring")
    parser.add_argument("--n_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr_subsample", type=int, default=20000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = args.device
    print(f"[Exp 12] Device: {device}")

    data = load_root_data_blueprint()
    lineage = load_lineage()
    n_genes = data['X_train'].shape[1]
    n_traj = lineage.shape[0]
    n_tb = lineage.shape[1]

    # ── External references from REAL training data only ──
    print("Building per-(type, bin) centroids from real training cells...")
    centroids = build_centroids(data['X_train'], data['y_train'], n_tb)
    print(f"  {len(centroids)} centroids")

    print(f"Training logistic-regression reference classifier "
          f"(subsample {args.lr_subsample})...")
    from sklearn.linear_model import LogisticRegression
    ct_names_train = np.array([lab.split('_', 1)[0] for lab in data['y_train']])
    rng = np.random.RandomState(0)
    idx = rng.choice(len(ct_names_train),
                     min(args.lr_subsample, len(ct_names_train)), replace=False)
    lr_clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr_clf.fit(data['X_train'][idx], ct_names_train[idx])
    holdout = rng.choice(len(data['y_val']), min(5000, len(data['y_val'])), replace=False)
    ct_names_val = np.array([lab.split('_', 1)[0] for lab in data['y_val']])
    lr_val_acc = float((lr_clf.predict(data['X_val'][holdout])
                        == ct_names_val[holdout]).mean())
    print(f"  LR reference accuracy on real held-out cells: {lr_val_acc:.4f}")

    # ── Score each model ──
    val_ds = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)

    all_rows = []
    for label, experiment, condition in MODELS:
        print(f"\n=== {label} ===")
        try:
            model = load_model_for(label, experiment, condition,
                                   n_genes, n_traj, device)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=True, drop_last=True)
        captured = generate_autoregressive(model, loader, device, n_tb,
                                           args.batch_size, args.n_batches)
        df = score_generated(captured, lineage, centroids, lr_clf, n_tb)
        df['model'] = label
        all_rows.append(df)
        print(df.to_string(index=False))
        del model

    result = pd.concat(all_rows, ignore_index=True)
    result['lr_reference_acc_on_real'] = lr_val_acc
    result.to_csv(os.path.join(OUT_DIR, "exp12_external_scores.csv"), index=False)

    # Summary: mean over bins t>=2
    summary = result.groupby('model').agg(
        centroid_acc=('centroid_acc', 'mean'),
        lr_acc=('lr_acc', 'mean'),
        mean_cosine_to_match=('mean_cosine_to_match', 'mean'),
        mean_margin=('mean_margin', 'mean'),
    ).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "exp12_external_summary.csv"), index=False)
    print("\n=== Summary (mean over generated bins) ===")
    print(summary.to_string(index=False))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [
            (axes[0], 'centroid_acc', 'Centroid classifier (external)'),
            (axes[1], 'lr_acc', 'Logistic regression (external)')]:
        for label in result['model'].unique():
            sub = result[result['model'] == label]
            ax.plot(sub['time_bin'], sub[metric], 'o-', label=label,
                    linewidth=2, markersize=4)
        ax.set_xlabel('Time bin'); ax.set_ylabel('Accuracy vs blueprint label')
        ax.set_title(title); ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    save_figure(fig, "exp12_external_scoring")
    print("\n[Exp 12] Done!")


if __name__ == "__main__":
    main()
