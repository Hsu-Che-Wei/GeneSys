#!/usr/bin/env python3
"""
Train one validation model through genesys_cli.run_cycle (train_run_cycle.py) and evaluate it.
One cycle of 30 epochs with the package defaults; the experiments differ only in the inputs
(labels, lineage blueprint, model object). Labels follow the package convention (time from 1).

Usage:
  python rescue_train_and_eval.py --experiment=exp1 --condition=reversed
  python rescue_train_and_eval.py --experiment=exp4 --condition=0.0
  python rescue_train_and_eval.py --experiment=exp6 --condition=full
  python rescue_train_and_eval.py --experiment=exp1 --condition=correct --eval_only
"""
import os, sys, argparse, json, copy, math, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint, GaussianNoise, DBlock, Decoder

from utils import (load_root_data, load_root_data_blueprint, reverse_time_labels, load_lineage, make_model, load_checkpoint,
                    RESULTS_DIR, ROOT_CKPT, NUM2NAME, NAME2NUM, save_figure)
from exp1_decoy_blueprint import make_shuffled_lineage, make_reversed_lineage, make_random_lineage
from exp6_ablation import ClassifierMLP


def make_model_with_sigma(input_size, output_size, device, sigma=0.2):
    model = ClassifierLSTM(input_size, output_size, 256, 256, 2, device)
    if sigma != 0.2:
        for m in model.modules():
            if isinstance(m, GaussianNoise):
                m.sigma = sigma
    return model.to(device)


def evaluate_model(model, data, lineage, device, batch_size=512):
    """Evaluate recreation accuracy per time bin."""
    model.eval()
    ds = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    n_tb = lineage.shape[1]
    preds = {t: [] for t in range(n_tb)}
    trues = []

    with torch.no_grad():
        for sample in loader:
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            if x.size(0) != batch_size:
                continue
            h = model.init_hidden(batch_size)
            for t in range(n_tb):
                p, h = model.predict_proba(x, h, t)
                preds[t].append(p.argmax(dim=1).cpu().numpy())
            trues.append(y.cpu().numpy())

    trues = np.concatenate(trues)
    results = {}
    for t in range(n_tb):
        p = np.concatenate(preds[t])
        results[t] = float((p == trues).mean())
    return results


def main():
    parser = argparse.ArgumentParser(description="Rescue: fast train + eval")
    parser.add_argument("--experiment", required=True, choices=["exp1", "exp4", "exp6"])
    parser.add_argument("--condition", required=True)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, just evaluate existing checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"[Rescue] {args.experiment} / {args.condition} "
          f"{'(eval only)' if args.eval_only else f'(train {args.epochs} epochs)'} "
          f"device={device}")

    data = load_root_data_blueprint()
    lineage_orig = load_lineage()
    input_size = data['X_train'].shape[1]
    n_traj = lineage_orig.shape[0]
    n_tb = lineage_orig.shape[1]

    # ── Configure experiment ──
    if args.experiment == "exp1":
        lineage_map = {
            'correct': lineage_orig.copy(),
            'shuffled': make_shuffled_lineage(lineage_orig),
            'reversed': make_reversed_lineage(lineage_orig),
            'random': make_random_lineage(lineage_orig),
        }
        lineage = lineage_map[args.condition]
        save_dir = os.path.join(RESULTS_DIR, "exp1_decoy", args.condition)
        prefix = f"exp1_{args.condition}"
        loss_mode = 'standard'
        model = make_model_with_sigma(input_size, n_traj, device)

    elif args.experiment == "exp4":
        sigma = float(args.condition)
        lineage = lineage_orig.copy()
        save_dir = os.path.join(RESULTS_DIR, "exp4_noise", f"sigma_{sigma}")
        prefix = f"exp4_sigma{sigma}"
        loss_mode = 'standard'
        model = make_model_with_sigma(input_size, n_traj, device, sigma=sigma)

    elif args.experiment == "exp6":
        loss_mode = 'nll_only' if args.condition == 'no_tdvae' else 'standard'
        if args.condition == 'no_blueprint':
            lineage = make_random_lineage(lineage_orig)
        else:
            lineage = lineage_orig.copy()
        save_dir = os.path.join(RESULTS_DIR, "exp6_ablation", args.condition)
        prefix = f"exp6_{args.condition}"
        if args.condition == 'no_lstm':
            model = ClassifierMLP(input_size, n_traj, 256, 256, 2, device).to(device)
        else:
            model = make_model_with_sigma(input_size, n_traj, device)

    os.makedirs(save_dir, exist_ok=True)

    # ── Check for existing checkpoint ──
    ckpt_path = os.path.join(save_dir, f"{prefix}_best.pth")
    cycle1_path = os.path.join(save_dir, f"{prefix}_cycle1.pth")
    resume_path = None
    if os.path.exists(ckpt_path):
        resume_path = ckpt_path
    elif os.path.exists(cycle1_path):
        resume_path = cycle1_path

    if args.eval_only:
        if resume_path is None:
            print(f"ERROR: No checkpoint found in {save_dir}")
            sys.exit(1)
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
    else:
        if resume_path:
            sys.exit(f"A checkpoint already exists ({resume_path}); training always starts from "
                     f"scratch. Move it away first.")

        # Reversed blueprint: column-reversed lineage table read against time-reversed labels,
        # so that the model is trained on t10, t9, ..., t1, stem cell.
        rev = (args.experiment == "exp1" and args.condition == "reversed")
        y_tr = reverse_time_labels(data['y_train']) if rev else data['y_train']
        y_va = reverse_time_labels(data['y_val']) if rev else data['y_val']
        import train_run_cycle as trc
        if args.experiment == "exp6" and args.condition == 'no_tdvae':
            trc.switch_off_tdvae(model)
        train_loader, val_loader = trc.make_loaders(data['X_train'], y_tr, data['X_val'], y_va,
                                                    lineage, args.batch_size)
        state, best_loss, best_acc, logs = trc.train(
            model, train_loader, val_loader, device, save_dir, prefix,
            epochs=args.epochs, batch_size=args.batch_size, time_bins=n_tb)
        model.load_state_dict(state)

    # ── Evaluate with CORRECT blueprint ──
    print(f"\n=== Evaluation (correct blueprint) ===")
    acc_per_bin = evaluate_model(model, data, lineage_orig, device, args.batch_size)
    mean_acc = np.mean(list(acc_per_bin.values()))
    print(f"  Mean accuracy: {mean_acc:.4f}")
    print(f"  Per-bin: {acc_per_bin}")

    # Also evaluate with the TRAINING blueprint (for decoy comparison)
    if args.experiment == "exp1" and args.condition != "correct":
        print(f"\n=== Evaluation (training blueprint: {args.condition}) ===")
        data_bp = dict(data)
        if args.condition == "reversed":
            data_bp['y_val'] = reverse_time_labels(data['y_val'])
        acc_train_bp = evaluate_model(model, data_bp, lineage, device, args.batch_size)
        mean_train = np.mean(list(acc_train_bp.values()))
        print(f"  Mean accuracy: {mean_train:.4f}")

    # Save result
    result = {
        'experiment': args.experiment,
        'condition': args.condition,
        'mean_acc_correct_bp': mean_acc,
        'acc_per_bin_correct_bp': acc_per_bin,
    }
    if args.experiment == "exp1" and args.condition != "correct":
        result['mean_acc_train_bp'] = mean_train
        result['acc_per_bin_train_bp'] = acc_train_bp

    with open(os.path.join(save_dir, f"{prefix}_eval_result.json"), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {save_dir}/{prefix}_eval_result.json")
    return result


if __name__ == "__main__":
    main()
