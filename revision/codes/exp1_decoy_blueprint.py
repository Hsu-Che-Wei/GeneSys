#!/usr/bin/env python3
"""
Experiment 1: Decoy Blueprint [R1.8, R3.1]

Train GeneSys with corrupted lineage blueprints to test whether the
model genuinely uses biological structure or just memorizes the blueprint.

4 conditions:
  - correct:  Original lineage blueprint
  - shuffled: Cell-type labels randomly permuted within each time bin column
  - reversed: Temporal order reversed (terminal→progenitor)
  - random:   Each entry is a uniformly random cell-type label

Usage:
  sbatch --job-name=exp1_correct experiments/slurm/run_gpu.sh experiments/exp1_decoy_blueprint.py --condition=correct
  sbatch --job-name=exp1_shuffled experiments/slurm/run_gpu.sh experiments/exp1_decoy_blueprint.py --condition=shuffled
  sbatch --job-name=exp1_reversed experiments/slurm/run_gpu.sh experiments/exp1_decoy_blueprint.py --condition=reversed
  sbatch --job-name=exp1_random experiments/slurm/run_gpu.sh experiments/exp1_decoy_blueprint.py --condition=random

Or run all at once (serial, slower):
  sbatch --job-name=exp1_all experiments/slurm/run_gpu.sh experiments/exp1_decoy_blueprint.py --condition=all
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

from utils import load_root_data, load_root_data_blueprint, load_lineage, RESULTS_DIR, save_figure
from train_utils import make_model_custom, train_model, evaluate_trained_model


def make_shuffled_lineage(lineage):
    """Shuffle cell-type labels within each time bin column."""
    shuffled = lineage.copy()
    rng = np.random.default_rng(42)
    for col in range(shuffled.shape[1]):
        vals = shuffled.iloc[:, col].values.copy()
        rng.shuffle(vals)
        shuffled.iloc[:, col] = vals
    return shuffled


def make_reversed_lineage(lineage):
    """Reverse temporal order: last time bin becomes first."""
    reversed_lin = lineage.copy()
    # Reverse column order (time bins)
    reversed_lin.columns = range(reversed_lin.shape[1])
    reversed_lin = reversed_lin.iloc[:, ::-1].copy()
    reversed_lin.columns = range(reversed_lin.shape[1])
    return reversed_lin


def make_random_lineage(lineage):
    """Each entry is a uniformly random cell-type label."""
    random_lin = lineage.copy()
    rng = np.random.default_rng(42)
    # the stem-cell column (t0) is kept; every later position draws one of the ten cell types
    all_cts = sorted(set(lineage.iloc[:, 1:].values.flatten()))
    for i in range(random_lin.shape[0]):
        for j in range(1, random_lin.shape[1]):
            random_lin.iloc[i, j] = rng.choice(all_cts)
    return random_lin


def run_condition(condition, data, lineage_orig, device, batch_size, save_dir):
    """Train and evaluate one decoy condition."""
    print(f"\n{'='*60}")
    print(f"  CONDITION: {condition}")
    print(f"{'='*60}")

    # Generate lineage
    if condition == 'correct':
        lineage = lineage_orig.copy()
    elif condition == 'shuffled':
        lineage = make_shuffled_lineage(lineage_orig)
    elif condition == 'reversed':
        lineage = make_reversed_lineage(lineage_orig)
    elif condition == 'random':
        lineage = make_random_lineage(lineage_orig)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    print(f"Lineage blueprint ({condition}):")
    print(lineage.to_string(index=False, header=False))

    # Data loaders
    train_dataset = Cell_Lineage_Blueprint(data['X_train'], data['y_train'], lineage)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_dataset = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)

    input_size = data['X_train'].shape[1]
    n_traj = lineage.shape[0]
    n_timebins = lineage.shape[1]

    # Train
    model = make_model_custom(input_size, n_traj, device)
    cond_dir = os.path.join(save_dir, condition)

    state, best_loss, history, logs = train_model(
        model, train_loader, val_loader, device,
        epochs_per_cycle=100, max_cycles=20, batch_size=batch_size,
        time_bins=n_timebins, save_dir=cond_dir,
        save_prefix=f"exp1_{condition}",
    )

    # Evaluate
    model.load_state_dict(state)
    acc_per_bin = evaluate_trained_model(model, data, lineage, device, batch_size)

    result = {
        'condition': condition,
        'best_loss': best_loss,
        'n_cycles': len(history),
        'final_val_acc': history[-1]['best_val_acc'],
        'acc_per_bin': acc_per_bin,
        'mean_acc': np.mean(list(acc_per_bin.values())),
    }

    print(f"\nResults for {condition}:")
    print(f"  Best loss: {best_loss:.1f}")
    print(f"  Final val_acc: {result['final_val_acc']:.4f}")
    print(f"  Mean acc across bins: {result['mean_acc']:.4f}")
    print(f"  Per-bin: {acc_per_bin}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Exp 1: Decoy Blueprint")
    parser.add_argument("--condition", default="all",
                        choices=["correct", "shuffled", "reversed", "random", "all"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"[Exp 1] Condition: {args.condition}, Device: {device}")

    # Load data
    print("Loading root atlas data...")
    data = load_root_data_blueprint()
    lineage_orig = load_lineage()
    print(f"  X_train: {data['X_train'].shape}")

    save_dir = os.path.join(RESULTS_DIR, "exp1_decoy")
    os.makedirs(save_dir, exist_ok=True)

    conditions = (['correct', 'shuffled', 'reversed', 'random']
                   if args.condition == 'all' else [args.condition])

    all_results = []
    for cond in conditions:
        result = run_condition(cond, data, lineage_orig, device,
                               args.batch_size, save_dir)
        all_results.append(result)

        # Save intermediate
        with open(os.path.join(save_dir, f"exp1_{cond}_result.json"), 'w') as f:
            json.dump({k: v for k, v in result.items()
                       if k != 'acc_per_bin'}, f, indent=2)

    # ── Aggregate if all conditions run ──
    if len(all_results) == 4:
        # Summary table
        df = pd.DataFrame([{
            'condition': r['condition'],
            'best_loss': r['best_loss'],
            'val_acc': r['final_val_acc'],
            'mean_acc': r['mean_acc'],
        } for r in all_results])
        csv_path = os.path.join(RESULTS_DIR, "exp1_decoy_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary to {csv_path}")

        # Per-bin accuracy table
        rows = []
        for r in all_results:
            for t, acc in r['acc_per_bin'].items():
                rows.append({
                    'condition': r['condition'],
                    'time_bin': t,
                    'accuracy': acc,
                })
        df_bins = pd.DataFrame(rows)
        df_bins.to_csv(os.path.join(RESULTS_DIR, "exp1_decoy_per_bin.csv"),
                       index=False)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart: overall accuracy
        ax = axes[0]
        colors = {'correct': '#2166ac', 'shuffled': '#d6604d',
                  'reversed': '#f4a582', 'random': '#b2182b'}
        for i, r in enumerate(all_results):
            ax.bar(i, r['mean_acc'], color=colors[r['condition']],
                   label=r['condition'].capitalize(), width=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels([r['condition'].capitalize() for r in all_results])
        ax.set_ylabel('Mean Recreation Accuracy', fontsize=13)
        ax.set_title('Effect of Blueprint Corruption', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5,
                    label='Chance')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Line plot: per-bin accuracy
        ax = axes[1]
        for r in all_results:
            bins = sorted(r['acc_per_bin'].keys())
            accs = [r['acc_per_bin'][b] for b in bins]
            ax.plot(bins, accs, 'o-', color=colors[r['condition']],
                    label=r['condition'].capitalize(), linewidth=2,
                    markersize=6)
        ax.set_xlabel('Time bin', fontsize=13)
        ax.set_ylabel('Accuracy', fontsize=13)
        ax.set_title('Accuracy by Time Bin', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        save_figure(fig, "exp1_decoy_blueprint")

    print("\n[Exp 1] Done!")


if __name__ == "__main__":
    main()
