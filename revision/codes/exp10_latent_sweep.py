#!/usr/bin/env python3
"""
Experiment 10: Latent-Dimension Sensitivity Sweep [Round 3, R3 point 6]

Reviewer 3 (round 3) noted that the latent-dimension sensitivity analysis was
declined on "would require retraining" grounds while other experiments were
retrained from scratch. This experiment closes that gap.

Trains GeneSys from scratch for 30 epochs at four model widths
(embedding_dim = hidden_dim = D for D in 64, 128, 256, 512; the TD-VAE latent
z has dimension 2*D since b_to_z outputs embedding_dim*2), then evaluates:

  1. Recreation accuracy (Figure S5 protocol, per time bin)
  2. Autoregressive generation accuracy (Figure S7 protocol)
  3. KL(z1), KL(z2), reconstruction loss per epoch (Figure S17 protocol)

Usage (one GPU job per dim):
  cd /resnick/groups/NolanLab/cwhsu/GeneSys_Revision
  sbatch --job-name=exp10_d64  experiments/slurm/run_gpu.sh experiments/exp10_latent_sweep.py --dim 64
  sbatch --job-name=exp10_d128 experiments/slurm/run_gpu.sh experiments/exp10_latent_sweep.py --dim 128
  sbatch --job-name=exp10_d256 experiments/slurm/run_gpu.sh experiments/exp10_latent_sweep.py --dim 256
  sbatch --job-name=exp10_d512 experiments/slurm/run_gpu.sh experiments/exp10_latent_sweep.py --dim 512

Aggregate after all four finish:
  python experiments/exp10_latent_sweep.py --aggregate
"""
import os, sys, json, math, copy, argparse, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint

from utils import load_root_data, load_root_data_blueprint, load_lineage, RESULTS_DIR, save_figure
from train_utils import evaluate_trained_model
from exp3_kl_divergence_from_scratch import calculate_loss_decomposed
from eval_recreation_accuracy import autoregressive_recreation_accuracy

DIMS = [64, 128, 256, 512]
SWEEP_DIR = os.path.join(RESULTS_DIR, "exp10_latent_sweep")


def run_dim(dim, data, lineage, device, batch_size, epochs):
    print(f"\n{'='*60}\n  LATENT SWEEP: dim={dim} (latent z = {2*dim})\n{'='*60}")
    input_size = data['X_train'].shape[1]
    n_traj = lineage.shape[0]
    n_tb = lineage.shape[1]

    model = ClassifierLSTM(
        input_size=input_size, output_size=n_traj,
        embedding_dim=dim, hidden_dim=dim, n_layers=2, device=device,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    import train_run_cycle as trc
    val_ds = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)
    train_loader, val_loader = trc.make_loaders(data['X_train'], data['y_train'],
                                                data['X_val'], data['y_val'], lineage, batch_size)
    kl = trc.KLLog(model, val_loader, n_tb, calculate_loss_decomposed)
    dim_dir = os.path.join(SWEEP_DIR, f"dim{dim}")
    best_state, _, _, logger = trc.train(model, train_loader, kl, device, dim_dir,
                                         f"exp10_dim{dim}", epochs=epochs,
                                         batch_size=batch_size, time_bins=n_tb)
    kl_log = [dict(r, train_loss=logger['tloss'][i], val_acc=logger['val_acc'][i])
              for i, r in enumerate(kl.rows)]

    model.load_state_dict(best_state)
    model.eval()

    # Classification accuracy (real cells at every time bin)
    rec_acc = evaluate_trained_model(model, data, lineage, device, batch_size)

    # Autoregressive generation accuracy (Figure S7 protocol)
    eval_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    ar = autoregressive_recreation_accuracy(
        model, eval_loader, device, time_bins=n_tb,
        batch_size=batch_size, n_batches=10)

    result = {
        'dim': dim,
        'latent_dim': 2 * dim,
        'n_params': n_params,
        'recreation_acc_per_bin': {str(k): v for k, v in rec_acc.items()},
        'recreation_mean_acc': float(np.mean(list(rec_acc.values()))),
        'autoregressive_acc_per_bin': {str(k): v for k, v in ar['acc_per_bin'].items()},
        'autoregressive_mean_acc': ar['mean_acc'],
        'final_kl_z1': kl_log[-1]['kl_z1'],
        'final_kl_z2': kl_log[-1]['kl_z2'],
        'kl_log': kl_log,
    }
    with open(os.path.join(dim_dir, f"exp10_dim{dim}_result.json"), 'w') as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(kl_log).to_csv(
        os.path.join(dim_dir, f"exp10_dim{dim}_kl_log.csv"), index=False)

    print(f"\n  dim={dim}: recreation={result['recreation_mean_acc']:.4f} "
          f"autoregressive={result['autoregressive_mean_acc']:.4f} "
          f"KL(z1)={result['final_kl_z1']:.1f}")
    return result


def aggregate():
    rows, kl_rows = [], []
    for dim in DIMS:
        path = os.path.join(SWEEP_DIR, f"dim{dim}", f"exp10_dim{dim}_result.json")
        if not os.path.exists(path):
            print(f"  Missing: {path} (skipping)")
            continue
        with open(path) as f:
            r = json.load(f)
        rows.append({
            'dim': r['dim'], 'latent_dim': r['latent_dim'],
            'n_params': r['n_params'],
            'recreation_mean_acc': r['recreation_mean_acc'],
            'autoregressive_mean_acc': r['autoregressive_mean_acc'],
            'final_kl_z1': r['final_kl_z1'], 'final_kl_z2': r['final_kl_z2'],
        })
        for e in r['kl_log']:
            kl_rows.append({'dim': r['dim'], **e})

    if not rows:
        print("No results found.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp10_latent_sweep_summary.csv"), index=False)
    pd.DataFrame(kl_rows).to_csv(
        os.path.join(RESULTS_DIR, "exp10_latent_sweep_kl_logs.csv"), index=False)
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    w = 0.35
    xs = np.arange(len(df))
    ax.bar(xs - w/2, df['recreation_mean_acc'], w, label='Recreation', color='#2166ac')
    ax.bar(xs + w/2, df['autoregressive_mean_acc'], w,
           label='Autoregressive generation', color='#d6604d')
    ax.set_xticks(xs)
    ax.set_xticklabels([f"D={d}\n(z={2*d})" for d in df['dim']])
    ax.set_ylabel('Mean accuracy', fontsize=13)
    ax.set_title('Accuracy vs model width', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    for dim, sub in pd.DataFrame(kl_rows).groupby('dim'):
        ax.plot(sub['epoch'], sub['kl_z1'], 'o-', label=f"D={dim}", markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('KL(z1) (nats)')
    ax.set_title('KL(z1) during training', fontsize=14)
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for dim, sub in pd.DataFrame(kl_rows).groupby('dim'):
        ax.plot(sub['epoch'], sub['kl_z2'], 'o-', label=f"D={dim}", markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('KL(z2) (nats)')
    ax.set_title('KL(z2) during training', fontsize=14)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "exp10_latent_sweep")


def main():
    parser = argparse.ArgumentParser(description="Exp 10: Latent-dimension sweep")
    parser.add_argument("--dim", type=int, choices=DIMS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
        return
    if args.dim is None:
        parser.error("Specify --dim or --aggregate")

    os.makedirs(SWEEP_DIR, exist_ok=True)
    data = load_root_data_blueprint()
    lineage = load_lineage()
    run_dim(args.dim, data, lineage, args.device, args.batch_size, args.epochs)
    print("\n[Exp 10] Done!")


if __name__ == "__main__":
    main()
