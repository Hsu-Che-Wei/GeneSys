#!/usr/bin/env python3
"""
Experiment 6: Ablation Study [R1.6, R2.1, R2.2]

Systematically remove each architectural component to measure its contribution:

  - full:          Full GeneSys (Bi-LSTM + TD-VAE + correct blueprint) — control
  - no_tdvae:      Remove TD-VAE loss (train with NLL only)
  - no_blueprint:  Replace correct blueprint with random (reuses Exp 1 random)
  - no_lstm:       Replace Bi-LSTM with feedforward MLP (no temporal context)

Usage:
  sbatch --job-name=exp6_full experiments/slurm/run_gpu.sh experiments/exp6_ablation.py --condition=full
  sbatch --job-name=exp6_notdvae experiments/slurm/run_gpu.sh experiments/exp6_ablation.py --condition=no_tdvae
  sbatch --job-name=exp6_noblue experiments/slurm/run_gpu.sh experiments/exp6_ablation.py --condition=no_blueprint
  sbatch --job-name=exp6_nolstm experiments/slurm/run_gpu.sh experiments/exp6_ablation.py --condition=no_lstm

Or all (serial):
  sbatch --job-name=exp6_all --time=96:00:00 experiments/slurm/run_gpu.sh experiments/exp6_ablation.py --condition=all
"""
import os, sys, argparse, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import (ClassifierLSTM, Cell_Lineage_Blueprint,
                           GaussianNoise, DBlock, Decoder)

from utils import load_root_data, load_root_data_blueprint, load_lineage, RESULTS_DIR, save_figure
from train_utils import make_model_custom, train_model, evaluate_trained_model
from exp1_decoy_blueprint import make_random_lineage


# ── No-LSTM ablation model ────────────────────────────────────────────────
class ClassifierMLP(nn.Module):
    """
    Ablation model: replaces bidirectional LSTM with per-timebin feedforward.
    Processes each time bin independently (no temporal context).
    Keeps TD-VAE and classifier for fair comparison.
    """
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim,
                 n_layers, device, drop_prob=0.2):
        super(ClassifierMLP, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            nn.Dropout(p=drop_prob),
            GaussianNoise(sigma=0.2),
        )

        # Replace LSTM with per-timebin feedforward
        self.temporal = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        self.fc = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim * 2, output_size),
        )

        self.dropout = nn.Dropout(drop_prob)

        # TD-VAE components (same as ClassifierLSTM)
        self.b_to_z = DBlock(hidden_dim * 2, hidden_dim, embedding_dim * 2)
        self.bz2_infer_z1 = DBlock(hidden_dim * 2 + embedding_dim * 2,
                                    hidden_dim, embedding_dim * 2)
        self.z1_to_z2 = DBlock(embedding_dim * 2, hidden_dim, embedding_dim * 2)
        self.z_to_x = Decoder(embedding_dim * 2, hidden_dim, input_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def predict(self, x, hidden, t):
        embeds = self.fc1(x)
        # Process each time bin independently
        temporal_out = self.temporal(embeds)
        out = self.dropout(temporal_out)
        out = self.fc(out)
        out = out[:, t, :]
        out = F.log_softmax(out, dim=1)
        return out, hidden  # hidden unused but kept for API compatibility

    def predict_proba(self, x, hidden, t):
        embeds = self.fc1(x)
        temporal_out = self.temporal(embeds)
        out = self.dropout(temporal_out)
        out = self.fc(out)
        out = out[:, t, :]
        out = F.softmax(out, dim=1)
        return out, hidden

    def get_belief(self, x, hidden):
        self.x = x
        embeds = self.fc1(x)
        self.b = self.temporal(embeds)  # "belief" = feedforward output

    def calculate_loss(self, t):
        z2_mu, z2_logsigma = self.b_to_z(self.b[:, t + 1, :])
        z2_epsilon = torch.randn_like(z2_mu)
        z2 = z2_mu + torch.exp(z2_logsigma) * z2_epsilon

        qs_z1_mu, qs_z1_logsigma = self.bz2_infer_z1(
            torch.cat((self.b[:, t, :], z2), dim=-1))
        qs_z1_epsilon = torch.randn_like(qs_z1_mu)
        qs_z1 = qs_z1_mu + torch.exp(qs_z1_logsigma) * qs_z1_epsilon

        pb_z1_mu, pb_z1_logsigma = self.b_to_z(self.b[:, t, :])
        t_z2_mu, t_z2_logsigma = self.z1_to_z2(qs_z1)
        x2 = self.z_to_x(z2)

        loss = 0.5 * torch.sum(
            ((pb_z1_mu - qs_z1) / torch.exp(pb_z1_logsigma)) ** 2, -1
        ) + torch.sum(pb_z1_logsigma, -1) - torch.sum(qs_z1_logsigma, -1)

        loss += torch.sum(
            -0.5 * z2_epsilon ** 2 - 0.5 * z2_epsilon.new_tensor(2 * np.pi)
            - z2_logsigma, dim=-1)
        loss += torch.sum(
            0.5 * ((z2 - t_z2_mu) / torch.exp(t_z2_logsigma)) ** 2
            + 0.5 * z2.new_tensor(2 * np.pi) + t_z2_logsigma, -1)
        loss += -torch.sum(
            self.x[:, t + 1, :] * torch.log(x2)
            + (1 - self.x[:, t + 1, :]) * torch.log(1 - x2), -1)

        return torch.mean(loss)

    def generate_current(self, x, hidden, t):
        self.get_belief(x, hidden)
        z_mu, z_logsigma = self.b_to_z(self.b[:, t, :])
        z = z_mu + torch.exp(z_logsigma) * torch.randn_like(z_mu)
        return self.z_to_x(z)

    def generate_next(self, x, hidden, t):
        self.get_belief(x, hidden)
        z_mu, z_logsigma = self.b_to_z(self.b[:, t, :])
        z = z_mu + torch.exp(z_logsigma) * torch.randn_like(z_mu)
        next_z_mu, next_z_logsigma = self.z1_to_z2(z)
        next_z = next_z_mu + torch.exp(next_z_logsigma) * torch.randn_like(next_z_mu)
        return self.z_to_x(next_z)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_().to(self.device),
                weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_().to(self.device))


def run_ablation(condition, data, lineage_orig, device, batch_size, save_dir):
    """Train and evaluate one ablation condition."""
    print(f"\n{'='*60}")
    print(f"  ABLATION: {condition}")
    print(f"{'='*60}")

    input_size = data['X_train'].shape[1]
    n_traj = lineage_orig.shape[0]
    n_timebins = lineage_orig.shape[1]

    # Determine lineage and loss mode
    if condition == 'no_blueprint':
        lineage = make_random_lineage(lineage_orig)
        loss_mode = 'standard'
    else:
        lineage = lineage_orig.copy()
        loss_mode = 'nll_only' if condition == 'no_tdvae' else 'standard'

    # Create data loaders
    train_dataset = Cell_Lineage_Blueprint(data['X_train'], data['y_train'], lineage)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_dataset = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)

    # Create model
    if condition == 'no_lstm':
        model = ClassifierMLP(input_size, n_traj, 256, 256, 2, device).to(device)
    else:
        model = make_model_custom(input_size, n_traj, device)

    cond_dir = os.path.join(save_dir, condition)

    state, best_loss, history, logs = train_model(
        model, train_loader, val_loader, device,
        epochs_per_cycle=100, max_cycles=20, batch_size=batch_size,
        time_bins=n_timebins, loss_mode=loss_mode,
        save_dir=cond_dir, save_prefix=f"exp6_{condition}",
    )

    model.load_state_dict(state)
    acc_per_bin = evaluate_trained_model(model, data, lineage, device, batch_size)

    result = {
        'condition': condition,
        'best_loss': best_loss,
        'n_cycles': len(history),
        'final_val_acc': history[-1]['best_val_acc'],
        'acc_per_bin': acc_per_bin,
        'mean_acc': np.mean(list(acc_per_bin.values())),
        'loss_mode': loss_mode,
    }

    print(f"\nResults for {condition}:")
    print(f"  Loss mode: {loss_mode}")
    print(f"  Best loss: {best_loss:.1f}")
    print(f"  Val acc: {result['final_val_acc']:.4f}")
    print(f"  Mean bin acc: {result['mean_acc']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Exp 6: Ablation Study")
    parser.add_argument("--condition", default="all",
                        choices=["full", "no_tdvae", "no_blueprint", "no_lstm", "all"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"[Exp 6] Condition: {args.condition}, Device: {device}")

    data = load_root_data_blueprint()
    lineage_orig = load_lineage()

    save_dir = os.path.join(RESULTS_DIR, "exp6_ablation")
    os.makedirs(save_dir, exist_ok=True)

    conditions = (['full', 'no_tdvae', 'no_blueprint', 'no_lstm']
                   if args.condition == 'all' else [args.condition])

    all_results = []
    for cond in conditions:
        result = run_ablation(cond, data, lineage_orig, device,
                               args.batch_size, save_dir)
        all_results.append(result)
        with open(os.path.join(save_dir, f"exp6_{cond}_result.json"), 'w') as f:
            json.dump({k: v for k, v in result.items()
                       if k != 'acc_per_bin'}, f, indent=2)

    # ── Aggregate ──
    if len(all_results) == 4:
        df = pd.DataFrame([{
            'condition': r['condition'],
            'best_loss': r['best_loss'],
            'val_acc': r['final_val_acc'],
            'mean_acc': r['mean_acc'],
            'loss_mode': r['loss_mode'],
        } for r in all_results])
        df.to_csv(os.path.join(RESULTS_DIR, "exp6_ablation_summary.csv"), index=False)

        # Per-bin
        rows = []
        for r in all_results:
            for t, acc in r['acc_per_bin'].items():
                rows.append({'condition': r['condition'], 'time_bin': t, 'accuracy': acc})
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, "exp6_ablation_per_bin.csv"), index=False)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        labels = {'full': 'Full GeneSys', 'no_tdvae': 'No TD-VAE',
                  'no_blueprint': 'No Blueprint', 'no_lstm': 'No LSTM'}
        colors = {'full': '#2166ac', 'no_tdvae': '#4393c3',
                  'no_blueprint': '#d6604d', 'no_lstm': '#b2182b'}

        # Bar chart
        ax = axes[0]
        for i, r in enumerate(all_results):
            ax.bar(i, r['mean_acc'], color=colors[r['condition']],
                   width=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels([labels[r['condition']] for r in all_results],
                           rotation=15, ha='right')
        ax.set_ylabel('Mean Recreation Accuracy', fontsize=13)
        ax.set_title('Ablation Study: Component Contributions', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Per-bin curves
        ax = axes[1]
        for r in all_results:
            bins = sorted(r['acc_per_bin'].keys())
            accs = [r['acc_per_bin'][b] for b in bins]
            ax.plot(bins, accs, 'o-', color=colors[r['condition']],
                    label=labels[r['condition']], linewidth=2, markersize=5)
        ax.set_xlabel('Time bin', fontsize=13)
        ax.set_ylabel('Accuracy', fontsize=13)
        ax.set_title('Accuracy by Time Bin', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        save_figure(fig, "exp6_ablation_comparison")

    print("\n[Exp 6] Done!")


if __name__ == "__main__":
    main()
