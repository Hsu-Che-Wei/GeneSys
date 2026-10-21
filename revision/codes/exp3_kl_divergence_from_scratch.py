#!/usr/bin/env python3
"""
Experiment 4 (revised): KL-Divergence Monitoring During Training From Scratch

Trains a GeneSys model from scratch for 30 epochs while logging all four
TD-VAE loss components at each epoch. This shows the actual KL trajectory
during learning, directly addressing whether posterior collapse occurs
during training (not just whether a converged model stays stable).

Output: results/exp3_kl_divergence_diagnostic.csv (overwrites previous)
Figure: Figure S17

Usage:
  python exp3_kl_divergence_from_scratch.py
  python exp3_kl_divergence_from_scratch.py --epochs 30
"""

import os, sys, math, copy, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint

from utils import load_root_data, load_root_data_blueprint, load_lineage, make_model, RESULTS_DIR


def calculate_loss_decomposed(model, t):
    """Same as ClassifierLSTM.calculate_loss(t) but returns individual KL terms.
    Manually recomputes the TD-VAE loss components to log them separately.
    """
    b = model.b
    x = model.x

    # Sample z2 from belief at t+1
    z2_mu, z2_logsigma = model.b_to_z(b[:, t + 1, :])
    z2_epsilon = torch.randn_like(z2_mu)
    z2 = z2_mu + torch.exp(z2_logsigma) * z2_epsilon

    # Infer z1 from belief at t and z2
    qs_z1_mu, qs_z1_logsigma = model.bz2_infer_z1(
        torch.cat((b[:, t, :], z2), dim=-1))
    qs_z1_epsilon = torch.randn_like(qs_z1_mu)
    qs_z1 = qs_z1_mu + torch.exp(qs_z1_logsigma) * qs_z1_epsilon

    # Belief prior for z1
    pb_z1_mu, pb_z1_logsigma = model.b_to_z(b[:, t, :])

    # Transition model z1 -> z2
    t_z2_mu, t_z2_logsigma = model.z1_to_z2(qs_z1)

    # Reconstruction
    x2 = model.z_to_x(z2)

    # KL(z1): analytic KL between two Gaussians
    kl_z1 = 0.5 * torch.sum(
        ((pb_z1_mu - qs_z1) / torch.exp(pb_z1_logsigma)) ** 2, -1
    ) + torch.sum(pb_z1_logsigma, -1) - torch.sum(qs_z1_logsigma, -1)
    kl_z1 = torch.mean(kl_z1)

    # KL(z2): estimated via samples
    log_q_z2 = torch.sum(
        -0.5 * z2_epsilon ** 2 - 0.5 * z2_epsilon.new_tensor(2 * np.pi) - z2_logsigma,
        dim=-1)
    log_p_z2 = torch.sum(
        -0.5 * ((z2 - t_z2_mu) / torch.exp(t_z2_logsigma)) ** 2
        - 0.5 * z2.new_tensor(2 * np.pi) - t_z2_logsigma,
        dim=-1)
    kl_z2 = torch.mean(log_q_z2 - log_p_z2)

    # Reconstruction loss: BCE
    recon = -torch.sum(
        x[:, t + 1, :] * torch.log(x2 + 1e-8) +
        (1 - x[:, t + 1, :]) * torch.log(1 - x2 + 1e-8),
        dim=-1)
    recon = torch.mean(recon)

    total = kl_z1 + kl_z2 + recon
    return total, float(kl_z1.item()), float(kl_z2.item()), float(recon.item())


def main():
    parser = argparse.ArgumentParser(description="KL divergence monitoring from scratch")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size
    print(f"[Exp 4] KL divergence from scratch, {args.epochs} epochs, device={device}")

    # Load data
    data = load_root_data_blueprint()
    lineage = load_lineage()
    input_size = data['X_train'].shape[1]
    n_traj = lineage.shape[0]
    n_tb = lineage.shape[1]

    # Create fresh model (no checkpoint loading)
    model = make_model(input_size, n_traj, device=device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params (randomly initialized)")

    # Training through genesys_cli.run_cycle (train_run_cycle.py); the loss components are
    # recorded by wrappers around model.calculate_loss and the validation loader.
    import train_run_cycle as trc
    train_loader, val_loader = trc.make_loaders(data['X_train'], data['y_train'],
                                                data['X_val'], data['y_val'], lineage, batch_size)
    kl = trc.KLLog(model, val_loader, n_tb, calculate_loss_decomposed)
    save_dir = os.path.join(RESULTS_DIR, "exp3_kl")
    best_state, best_loss, _, logger = trc.train(model, train_loader, kl, device, save_dir,
                                                 "exp3_kl_from_scratch", epochs=args.epochs,
                                                 batch_size=batch_size, time_bins=n_tb)
    rows = []
    for i, r in enumerate(kl.rows):
        rows.append({
            'epoch': i, 'train_loss': logger['tloss'][i], 'val_acc': logger['val_acc'][i],
            'train_kl_z1': r['train_kl_z1'], 'train_kl_z2': r['train_kl_z2'],
            'train_recon': r['train_recon'],
            # run_cycle logs the mean of TD-VAE loss + classifier NLL; the NLL is the difference
            'train_nll': logger['tloss'][i] - r['train_tdvae'],
            'train_total_tdvae': r['train_kl_z1'] + r['train_kl_z2'] + r['train_recon'],
            'val_kl_z1': r['val_kl_z1'], 'val_kl_z2': r['val_kl_z2'], 'val_recon': r['val_recon'],
            'val_total_tdvae': r['val_kl_z1'] + r['val_kl_z2'] + r['val_recon'],
        })

    # Save results
    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "exp3_kl_divergence_diagnostic.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Save best checkpoint
    save_dir = os.path.join(RESULTS_DIR, "exp3_kl")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"state_dict": best_state, "best_loss": best_loss},
               os.path.join(save_dir, "exp3_kl_from_scratch_best.pth"))
    print(f"Saved checkpoint: {save_dir}/exp3_kl_from_scratch_best.pth")


if __name__ == "__main__":
    main()
