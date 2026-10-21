"""
Shared training utilities for Sprint 2 experiments that require
retraining GeneSys models with modified configurations.

Wraps the GitHub genesys_cli.py training logic with support for:
  - Custom lineage blueprints (Exp 1: decoy)
  - Custom noise sigma (Exp 4: noise sensitivity)
  - Custom loss modes (Exp 6: ablation — no TD-VAE)
  - Post-training evaluation pipeline
"""
import os, sys, math, copy, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint, GaussianNoise

from utils import RESULTS_DIR, NUM2NAME, load_root_data, load_lineage, save_figure


def make_model_custom(input_size, output_size, device, sigma=0.2):
    """Create model with optional custom noise sigma."""
    model = ClassifierLSTM(
        input_size=input_size,
        output_size=output_size,
        embedding_dim=256,
        hidden_dim=256,
        n_layers=2,
        device=device,
    )

    # Override sigma if different from default
    if sigma != 0.2:
        for module in model.modules():
            if isinstance(module, GaussianNoise):
                module.sigma = sigma

    return model.to(device)


def train_model(model, train_loader, val_loader, device,
                epochs_per_cycle=100, max_cycles=20, batch_size=512,
                time_bins=11, lr=1e-3, clip=5, loss_mode='standard',
                save_dir=None, save_prefix='model', verbose=True):
    """
    Train a GeneSys model with configurable loss mode.

    loss_mode:
        'standard' — original: randomly pick tdvae_loss or nll_loss
        'nll_only' — ablation: only backprop classifier NLL loss
        'tdvae_only' — ablation: only backprop TD-VAE loss
    """
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.NLLLoss()
    overall_best_state = copy.deepcopy(model.state_dict())
    prev_cycle_best_loss = math.inf
    history = []
    all_logs = []

    for cycle in range(1, max_cycles + 1):
        if verbose:
            print(f"\n=== Cycle {cycle}/{max_cycles} ===")

        model.load_state_dict(overall_best_state)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, threshold=0.05
        )

        best_loss = math.inf
        best_val_acc = 0.0
        best_state = copy.deepcopy(model.state_dict())
        cycle_logs = {'tloss': [], 'val_acc': []}

        for epoch in range(epochs_per_cycle):
            # ── Train ──
            model.train()
            h = model.init_hidden(batch_size)
            running_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, leave=False,
                        desc=f"C{cycle} E{epoch:03d}") if verbose else train_loader
            for sample in pbar:
                h = tuple([each.data for each in h])
                x = sample['x'].to(device)
                y = sample['y'].to(device)

                if x.size(0) != batch_size:
                    continue

                optimizer.zero_grad(set_to_none=True)

                t = np.random.choice(max(1, time_bins - 1))
                p, h = model.predict(x, h, t)
                model.get_belief(x, h)

                tdvae_loss = model.calculate_loss(t)
                nll_loss = criterion(p, y)

                # Select loss based on mode
                if loss_mode == 'nll_only':
                    nll_loss.backward()
                elif loss_mode == 'tdvae_only':
                    tdvae_loss.backward()
                else:  # standard
                    wl = np.random.choice(2)
                    if wl == 0:
                        tdvae_loss.backward()
                    else:
                        nll_loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                total = float(tdvae_loss.item()) + float(nll_loss.item())
                running_loss += total
                n_batches += 1

                if verbose and hasattr(pbar, 'set_description'):
                    pbar.set_description(
                        f"C{cycle} E{epoch:03d} loss={total:.1f}")

            avg_loss = running_loss / max(n_batches, 1)
            cycle_logs['tloss'].append(avg_loss)
            scheduler.step(avg_loss)

            # ── Validate ──
            model.eval()
            val_h = model.init_hidden(batch_size)
            y_pred, y_true = [], []
            with torch.no_grad():
                for sample in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    x = sample['x'].to(device)
                    y = sample['y'].to(device)
                    if x.size(0) != batch_size:
                        continue
                    p, val_h = model.predict_proba(x, val_h, 1)
                    y_pred.append(p.cpu().numpy())
                    y_true.append(y.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            val_acc = (y_true == y_pred.argmax(axis=1)).mean()
            cycle_logs['val_acc'].append(float(val_acc))

            curr_lr = optimizer.param_groups[0]['lr']
            if verbose:
                print(f"  [{cycle}/{epoch+1:03d}] loss={avg_loss:.1f} "
                      f"val_acc={val_acc:.4f} lr={curr_lr:.2e}")

            improved = (avg_loss < best_loss) or (
                np.isclose(avg_loss, best_loss) and val_acc > best_val_acc)
            if improved:
                best_loss = avg_loss
                best_val_acc = float(val_acc)
                best_state = copy.deepcopy(model.state_dict())

        # Save cycle checkpoint
        ckpt_path = os.path.join(save_dir, f"{save_prefix}_cycle{cycle}.pth")
        torch.save({
            "state_dict": best_state,
            "best_loss": best_loss,
            "best_val_acc": best_val_acc,
            "cycle": cycle,
        }, ckpt_path)

        all_logs.append(cycle_logs)
        history.append({
            "cycle": cycle,
            "best_loss": best_loss,
            "best_val_acc": best_val_acc,
        })

        if verbose:
            print(f"  Cycle {cycle} best: loss={best_loss:.1f} "
                  f"val_acc={best_val_acc:.4f}")

        # Early stopping across cycles
        if cycle > 1 and not (best_loss < prev_cycle_best_loss):
            if verbose:
                print("No improvement over previous cycle; stopping.")
            break

        prev_cycle_best_loss = best_loss
        overall_best_state = copy.deepcopy(best_state)

    # Save final best
    final_path = os.path.join(save_dir, f"{save_prefix}_best.pth")
    torch.save({
        "state_dict": overall_best_state,
        "best_loss": prev_cycle_best_loss,
        "history": history,
    }, final_path)

    # Save training logs
    log_path = os.path.join(save_dir, f"{save_prefix}_logs.pkl")
    with open(log_path, 'wb') as f:
        pickle.dump(all_logs, f)

    if verbose:
        print(f"\nFinal best saved: {final_path} "
              f"(loss={prev_cycle_best_loss:.1f})")

    return overall_best_state, prev_cycle_best_loss, history, all_logs


def evaluate_trained_model(model, data, lineage, device, batch_size=512):
    """
    Quick evaluation: recreation accuracy at each time bin.
    Returns dict of {time_bin: accuracy}.
    """
    model.eval()
    test_dataset = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, drop_last=True)

    n_timebins = lineage.shape[1]
    all_preds = {t: [] for t in range(n_timebins)}
    all_true = []

    with torch.no_grad():
        for sample in test_loader:
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            if x.size(0) != batch_size:
                continue
            h = model.init_hidden(batch_size)
            for t in range(n_timebins):
                p, h = model.predict_proba(x, h, t)
                all_preds[t].append(p.argmax(dim=1).cpu().numpy())
            all_true.append(y.cpu().numpy())

    all_true = np.concatenate(all_true)
    results = {}
    for t in range(n_timebins):
        preds = np.concatenate(all_preds[t])
        results[t] = float((preds == all_true).mean())

    return results
