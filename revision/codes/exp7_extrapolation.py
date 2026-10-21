#!/usr/bin/env python3
"""
Experiment 7: Extrapolation Test [R3.2]

Test whether GeneSys can predict terminal differentiation states from
limited initial time points.

Provides progressively fewer seed time points (real data) and fills
the rest with zeros, then evaluates:
  - Classification accuracy at each generated time bin
  - Cosine similarity between generated and real transcriptomes

Usage:
  sbatch --job-name=exp7_extrap experiments/slurm/run_gpu.sh experiments/exp7_extrapolation.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.spatial.distance import cosine as cosine_dist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint

from utils import (
    load_root_data, load_root_data_blueprint, load_lineage, make_model, load_checkpoint,
    RESULTS_DIR, ROOT_CKPT, NUM2NAME, NAME2NUM, save_figure
)


def build_seed_input(X, y_labels, lineage, seed_bins, device, batch_size=128):
    """
    Build input tensors with real data for the first `seed_bins` time points
    and zeros for the rest.

    Returns:
        x_seeded: tensor (batch, time_bins, genes)
        x_full: tensor (batch, time_bins, genes)  - full real data for comparison
        y_traj: array of trajectory indices
    """
    label_parts = np.array([lab.split('_', 1) for lab in y_labels])
    ct_names = label_parts[:, 0]
    ct_times = label_parts[:, 1].astype(float)

    n_genes = X.shape[1]
    n_timebins = lineage.shape[1]
    n_traj = lineage.shape[0]

    all_seeded = []
    all_full = []
    all_traj = []

    for traj_idx in range(n_traj):
        traj_ct = lineage.iloc[traj_idx, -1]
        cell_mask = (ct_names == traj_ct)
        if cell_mask.sum() == 0:
            continue

        cell_indices = np.where(cell_mask)[0][:batch_size]  # cap per trajectory
        bs = len(cell_indices)

        seq_seeded = np.zeros((bs, n_timebins, n_genes), dtype=np.float32)
        seq_full = np.zeros((bs, n_timebins, n_genes), dtype=np.float32)

        for t in range(n_timebins):
            lineage_ct = lineage.iloc[traj_idx, t]
            source_mask = (ct_names == lineage_ct) & (ct_times == float(t + 1))
            source_idx = np.where(source_mask)[0]

            if len(source_idx) > 0:
                sampled = np.random.choice(source_idx, bs, replace=True)
                seq_full[:, t, :] = X[sampled, :]
                if t < seed_bins:
                    seq_seeded[:, t, :] = X[sampled, :]
                # else: seeded stays zeros

        all_seeded.append(seq_seeded)
        all_full.append(seq_full)
        all_traj.extend([traj_idx] * bs)

    X_seeded = np.concatenate(all_seeded, axis=0)
    X_full = np.concatenate(all_full, axis=0)
    y_traj = np.array(all_traj)

    return (torch.from_numpy(X_seeded).to(device),
            torch.from_numpy(X_full).to(device),
            y_traj)


def evaluate_extrapolation(model, X_seeded, X_full, y_traj, device, n_timebins):
    """
    For each time bin, evaluate:
      1. Classification accuracy (predict_proba → argmax vs true trajectory)
      2. Generation quality (generate_current/generate_next → cosine sim vs real)
    """
    model.eval()
    bs = X_seeded.size(0)
    results = []

    with torch.no_grad():
        h = model.init_hidden(bs)

        for t in range(n_timebins):
            # Classification accuracy
            p, h_new = model.predict_proba(X_seeded, h, t)
            preds = p.argmax(dim=1).cpu().numpy()
            acc = (preds == y_traj).mean()

            # Generate transcriptome at this time bin
            if t == 0:
                gen_x = model.generate_current(X_seeded, h, 0)
            else:
                gen_x = model.generate_next(X_seeded, h, t - 1)

            # Cosine similarity to real data
            gen_np = gen_x.cpu().numpy()
            real_np = X_full[:, t, :].cpu().numpy()

            cos_sims = []
            for i in range(bs):
                if np.linalg.norm(real_np[i]) > 0 and np.linalg.norm(gen_np[i]) > 0:
                    sim = 1.0 - cosine_dist(gen_np[i], real_np[i])
                    cos_sims.append(sim)

            mean_cos = np.mean(cos_sims) if cos_sims else 0.0

            results.append({
                'time_bin': t,
                'accuracy': float(acc),
                'cosine_similarity': float(mean_cos),
                'n_samples': bs,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 7: Extrapolation test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = args.device
    print(f"[Exp 7] Device: {device}")

    # Load data and model
    print("Loading root atlas data...")
    data = load_root_data_blueprint()
    lineage = load_lineage()

    input_size = data['X_val'].shape[1]
    n_traj = lineage.shape[0]
    n_timebins = lineage.shape[1]
    print(f"  input_size={input_size}, trajectories={n_traj}, time_bins={n_timebins}")

    model = make_model(input_size, output_size=n_traj, device=device)
    model = load_checkpoint(model, ROOT_CKPT, device=device)
    model.eval()

    X_eval = data['X_val']
    y_eval = data['y_val']

    # Test with different seed lengths
    seed_lengths = [n_timebins, 7, 5, 3, 2, 1, 0]
    all_results = []

    for seed_bins in seed_lengths:
        print(f"\n=== Seed bins: {seed_bins}/{n_timebins} ===")

        np.random.seed(42)  # reproducibility
        X_seeded, X_full, y_traj = build_seed_input(
            X_eval, y_eval, lineage, seed_bins, device, args.batch_size
        )

        results = evaluate_extrapolation(
            model, X_seeded, X_full, y_traj, device, n_timebins
        )

        for r in results:
            r['seed_bins'] = seed_bins
            # Mark whether this time bin was seeded or generated
            r['is_seeded'] = r['time_bin'] < seed_bins
        all_results.extend(results)

        # Print summary
        accs = [r['accuracy'] for r in results]
        coss = [r['cosine_similarity'] for r in results]
        print(f"  Mean accuracy: {np.mean(accs):.4f}")
        print(f"  Mean cosine sim: {np.mean(coss):.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "exp7_extrapolation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # ── Plot 1: Accuracy heatmap (seed_bins × time_bin) ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, title in [
        (axes[0], 'accuracy', 'Classification Accuracy'),
        (axes[1], 'cosine_similarity', 'Cosine Similarity to Real Data'),
    ]:
        pivot = df.pivot_table(index='seed_bins', columns='time_bin',
                               values=metric, aggfunc='mean')
        pivot = pivot.sort_index(ascending=False)

        im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n_timebins))
        ax.set_xticklabels([f't{t}' for t in range(n_timebins)])
        ax.set_yticks(range(len(seed_lengths)))
        ax.set_yticklabels([str(s) for s in sorted(seed_lengths, reverse=True)])
        ax.set_xlabel('Time bin', fontsize=12)
        ax.set_ylabel('# Seed time bins provided', fontsize=12)
        ax.set_title(title, fontsize=13)

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=8)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Experiment 7: Extrapolation — Effect of Seed Length on Generation Quality',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "exp7_extrapolation_heatmap")

    # ── Plot 2: Line plot — accuracy by seed length ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    # Show only non-seeded time bins (generated ones)
    df_gen = df[~df['is_seeded']]
    if len(df_gen) > 0:
        means = df_gen.groupby('seed_bins')['accuracy'].mean()
        ax.plot(means.index, means.values, 'o-', color='#2166ac', linewidth=2, markersize=8)
        ax.set_xlabel('Number of seed time bins', fontsize=13)
        ax.set_ylabel('Mean accuracy on GENERATED time bins', fontsize=13)
        ax.set_title('Extrapolation Accuracy vs Seed Length', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=1.0/n_traj, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    save_figure(fig, "exp7_extrapolation_accuracy")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Summary: Mean accuracy by seed length ===")
    summary = df.groupby('seed_bins')[['accuracy', 'cosine_similarity']].mean()
    print(summary)


if __name__ == "__main__":
    main()
