#!/usr/bin/env python3
"""
Experiment 4: Noise Sensitivity (sigma) [R3.14]

Train GeneSys with different Gaussian noise levels to justify sigma=0.2
and show it doesn't overwhelm sparse scRNA-seq signal.

4 conditions: sigma = {0.0, 0.1, 0.2, 0.4}

Also computes an analytical defense: since noise is RELATIVE (sigma * x),
genes with zero expression get zero noise. Produces a histogram showing
noise magnitude vs expression level.

Usage:
  sbatch --job-name=exp4_s0.0 experiments/slurm/run_gpu.sh experiments/exp4_noise_sensitivity.py --sigma=0.0
  sbatch --job-name=exp4_s0.1 experiments/slurm/run_gpu.sh experiments/exp4_noise_sensitivity.py --sigma=0.1
  sbatch --job-name=exp4_s0.2 experiments/slurm/run_gpu.sh experiments/exp4_noise_sensitivity.py --sigma=0.2
  sbatch --job-name=exp4_s0.4 experiments/slurm/run_gpu.sh experiments/exp4_noise_sensitivity.py --sigma=0.4

Or run all:
  sbatch --job-name=exp4_all --time=96:00:00 experiments/slurm/run_gpu.sh experiments/exp4_noise_sensitivity.py --sigma=all
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import Cell_Lineage_Blueprint

from utils import load_root_data, load_lineage, RESULTS_DIR, save_figure
from train_utils import make_model_custom, train_model, evaluate_trained_model


def analyze_noise_vs_sparsity(X, sigma=0.2):
    """
    Compute the expected noise magnitude for each gene across all cells.
    Shows that zero-expression genes receive zero noise (relative noise property).

    Returns DataFrame with per-gene stats.
    """
    mean_expr = X.mean(axis=0)
    std_expr = X.std(axis=0)
    sparsity = (X == 0).mean(axis=0)
    expected_noise = sigma * mean_expr  # noise scale = sigma * |x|

    df = pd.DataFrame({
        'gene_idx': range(len(mean_expr)),
        'mean_expression': mean_expr,
        'std_expression': std_expr,
        'sparsity': sparsity,
        'expected_noise_scale': expected_noise,
        'noise_to_signal': np.where(mean_expr > 0,
                                     expected_noise / mean_expr, 0),
    })
    return df


def run_sigma(sigma, data, lineage, device, batch_size, save_dir):
    """Train and evaluate one sigma condition."""
    print(f"\n{'='*60}")
    print(f"  SIGMA = {sigma}")
    print(f"{'='*60}")

    train_dataset = Cell_Lineage_Blueprint(data['X_train'], data['y_train'], lineage)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_dataset = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)

    input_size = data['X_train'].shape[1]
    n_traj = lineage.shape[0]
    n_timebins = lineage.shape[1]

    model = make_model_custom(input_size, n_traj, device, sigma=sigma)
    sigma_dir = os.path.join(save_dir, f"sigma_{sigma}")

    state, best_loss, history, logs = train_model(
        model, train_loader, val_loader, device,
        epochs_per_cycle=100, max_cycles=20, batch_size=batch_size,
        time_bins=n_timebins, save_dir=sigma_dir,
        save_prefix=f"exp4_sigma{sigma}",
    )

    model.load_state_dict(state)
    acc_per_bin = evaluate_trained_model(model, data, lineage, device, batch_size)

    result = {
        'sigma': sigma,
        'best_loss': best_loss,
        'n_cycles': len(history),
        'final_val_acc': history[-1]['best_val_acc'],
        'acc_per_bin': acc_per_bin,
        'mean_acc': np.mean(list(acc_per_bin.values())),
    }

    print(f"\nResults for sigma={sigma}:")
    print(f"  Best loss: {best_loss:.1f}")
    print(f"  Val acc: {result['final_val_acc']:.4f}")
    print(f"  Mean bin acc: {result['mean_acc']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Exp 4: Noise Sensitivity")
    parser.add_argument("--sigma", default="all",
                        help="Sigma value or 'all' for sweep")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"[Exp 4] Sigma: {args.sigma}, Device: {device}")

    data = load_root_data()
    lineage = load_lineage()

    save_dir = os.path.join(RESULTS_DIR, "exp4_noise")
    os.makedirs(save_dir, exist_ok=True)

    # ── Analytical defense: noise vs sparsity ──
    print("\n=== Noise vs Sparsity Analysis ===")
    noise_df = analyze_noise_vs_sparsity(data['X_train'], sigma=0.2)
    noise_csv = os.path.join(RESULTS_DIR, "exp4_noise_sparsity_analysis.csv")
    noise_df.to_csv(noise_csv, index=False)

    n_zero = (noise_df['mean_expression'] == 0).sum()
    n_total = len(noise_df)
    print(f"  Genes with zero mean expression: {n_zero}/{n_total} "
          f"({100*n_zero/n_total:.1f}%) — these get ZERO noise")
    print(f"  Noise-to-signal ratio for expressed genes: "
          f"{noise_df[noise_df['mean_expression'] > 0]['noise_to_signal'].mean():.4f}")

    # Plot sparsity analysis
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    ax.hist(noise_df['sparsity'], bins=50, color='steelblue', edgecolor='white')
    ax.set_xlabel('Sparsity (fraction of zeros)', fontsize=11)
    ax.set_ylabel('Number of genes', fontsize=11)
    ax.set_title('Gene Sparsity Distribution', fontsize=12)

    ax = axes[1]
    expressed = noise_df[noise_df['mean_expression'] > 0]
    ax.scatter(expressed['mean_expression'], expressed['expected_noise_scale'],
               alpha=0.1, s=1, color='steelblue')
    ax.set_xlabel('Mean expression', fontsize=11)
    ax.set_ylabel('Expected noise scale (σ × |x|)', fontsize=11)
    ax.set_title('Noise Scale vs Expression Level', fontsize=12)
    ax.plot([0, expressed['mean_expression'].max()],
            [0, 0.2 * expressed['mean_expression'].max()],
            'r--', alpha=0.5, label='σ=0.2 line')
    ax.legend()

    ax = axes[2]
    ax.hist(noise_df['expected_noise_scale'], bins=50, color='steelblue',
            edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Expected noise scale', fontsize=11)
    ax.set_ylabel('Number of genes', fontsize=11)
    ax.set_title(f'Noise Scale Distribution (σ=0.2)\n'
                 f'{n_zero} genes ({100*n_zero/n_total:.0f}%) get zero noise',
                 fontsize=12)

    plt.tight_layout()
    save_figure(fig, "exp4_noise_sparsity_analysis")

    # ── Training sweep ──
    if args.sigma == 'all':
        sigmas = [0.0, 0.1, 0.2, 0.4]
    else:
        sigmas = [float(args.sigma)]

    all_results = []
    for sigma in sigmas:
        result = run_sigma(sigma, data, lineage, device, args.batch_size, save_dir)
        all_results.append(result)
        with open(os.path.join(save_dir, f"exp4_sigma{sigma}_result.json"), 'w') as f:
            json.dump({k: v for k, v in result.items()
                       if k != 'acc_per_bin'}, f, indent=2)

    # ── Aggregate plot if all sigmas run ──
    if len(all_results) == 4:
        df = pd.DataFrame([{
            'sigma': r['sigma'],
            'best_loss': r['best_loss'],
            'val_acc': r['final_val_acc'],
            'mean_acc': r['mean_acc'],
        } for r in all_results])
        df.to_csv(os.path.join(RESULTS_DIR, "exp4_noise_summary.csv"), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot(df['sigma'], df['mean_acc'], 'o-', color='#2166ac',
                linewidth=2, markersize=10)
        ax.set_xlabel('Gaussian noise σ', fontsize=13)
        ax.set_ylabel('Mean recreation accuracy', fontsize=13)
        ax.set_title('Accuracy vs Noise Level', fontsize=14)
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5,
                    label='Default σ=0.2')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        ax = axes[1]
        ax.plot(df['sigma'], df['best_loss'], 'o-', color='#d6604d',
                linewidth=2, markersize=10)
        ax.set_xlabel('Gaussian noise σ', fontsize=13)
        ax.set_ylabel('Best training loss', fontsize=13)
        ax.set_title('Training Loss vs Noise Level', fontsize=14)
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5,
                    label='Default σ=0.2')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, "exp4_noise_sigma_comparison")

    print("\n[Exp 4] Done!")


if __name__ == "__main__":
    main()
