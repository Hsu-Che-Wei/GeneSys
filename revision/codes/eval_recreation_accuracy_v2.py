#!/usr/bin/env python3
"""
Evaluate recreation accuracy matching the EXACT protocol from the manuscript
(GitHub notebook 4, cell 257, Figure S5).

Protocol (NO autoregressive generation):
  For each time bin t (1..10):
    1. Build input: real cells at t0 through t-1, pad rest with last real cell
       x_k = [real_t0, real_t1, ..., real_t(k), real_t(k), ..., real_t(k)]
    2. Initialize hidden state FRESH
    3. predict_proba(x_k, fresh_h, k+1) → classify at bin k+1
    4. Compare prediction to ground truth trajectory index

  This measures: "given real cells up to time k, can the model classify
  the trajectory at time k+1?"

Usage:
  python eval_recreation_accuracy_v2.py --experiment exp6 --condition full
  python eval_recreation_accuracy_v2.py --all
  python eval_recreation_accuracy_v2.py --checkpoint /path/to/model.pth --label name
  python eval_recreation_accuracy_v2.py --original  # evaluate original manuscript checkpoint
"""

import argparse, os, sys, json, glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint

from utils import (load_root_data, load_root_data_blueprint, load_lineage, NUM2NAME, NAME2NUM, RESULTS_DIR)
from train_utils import make_model_custom
from exp6_ablation import ClassifierMLP


# Original Root_Dataset for evaluating the original checkpoint
class Root_Dataset(Dataset):
    """Original dataset class matching the training pipeline of the manuscript model."""
    def __init__(self, features, labels):
        super(Root_Dataset).__init__()
        self.features = features
        self.labels = labels
        self.celltype = np.array([i.split('_', 1)[0] for i in self.labels], dtype='int')
        self.timebin = np.array([i.split('_', 1)[1] for i in self.labels], dtype='float32')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        idx = np.random.randint(10)
        cell = np.copy(self.features[
            np.random.choice(np.where(self.celltype == -1)[0], 1), :
        ]).astype(np.float32).reshape(-1)

        cell_list = [cell]
        for k in range(1, 11):
            sample = np.copy(self.features[
                np.random.choice(
                    np.where((self.celltype == idx) & (self.timebin == k))[0], 1
                ), :
            ]).astype(np.float32).reshape(-1)
            cell_list.append(sample)

        return {'x': np.array(cell_list), 'y': idx}


def recreation_accuracy(model, loader, device, batch_size, n_timebins=11, n_batches=10):
    """
    Recreation accuracy matching GitHub notebook 4, cell 257.

    For each time bin k (0..n_timebins-2):
      - Build x_k: real cells at positions 0..k, pad rest with position k
      - Fresh hidden state
      - predict_proba(x_k, fresh_h, k+1) → classify at bin k+1
    """
    model.eval()

    # Collect predictions per target time bin (1..n_timebins-1)
    all_preds = {t: [] for t in range(1, n_timebins)}
    all_true = []

    with torch.no_grad():
        batch_count = 0
        for sample in loader:
            if batch_count >= n_batches:
                break

            x_real = sample['x'].to(device)  # (batch, n_timebins, n_genes)
            y = sample['y'].to(device)

            if x_real.size(0) != batch_size:
                continue

            # For each seed length k, classify at time k+1
            for k in range(0, n_timebins - 1):
                # Build x_k: [t0, t1, ..., t_k, t_k, t_k, ...]
                components = [x_real[:, i, :] for i in range(k + 1)]
                padding = [x_real[:, k, :]] * (n_timebins - k - 1)
                x_k = torch.stack(components + padding, dim=1)

                # Fresh hidden state (matching notebook: test_h = model.init_hidden())
                h = model.init_hidden(batch_size)

                # Classify at time bin k+1
                p, _ = model.predict_proba(x_k, h, k + 1)
                all_preds[k + 1].append(p.argmax(dim=1).cpu().numpy())

            all_true.append(y.cpu().numpy())
            batch_count += 1

    true = np.concatenate(all_true)
    acc_per_bin = {}
    for t in range(1, n_timebins):
        preds = np.concatenate(all_preds[t])
        acc_per_bin[str(t)] = float((preds == true).mean())

    mean_acc = float(np.mean(list(acc_per_bin.values())))

    return {
        'mean_acc': mean_acc,
        'acc_per_bin': acc_per_bin,
        'n_samples': int(len(true)),
    }


def get_checkpoint_path(experiment, condition):
    exp_dirs = {
        'exp1': f'exp1_decoy/{condition}',
        'exp4': f'exp4_noise/sigma_{condition}',
        'exp6': f'exp6_ablation/{condition}',
    }
    result_dir = os.path.join(RESULTS_DIR, exp_dirs.get(experiment, ''))
    for pattern in ['*_best.pth', '*_cycle1.pth', '*_ep30.pth', '*_ep20.pth']:
        matches = glob.glob(os.path.join(result_dir, pattern))
        if matches:
            return sorted(matches)[-1]
    raise FileNotFoundError(f"No checkpoint found in {result_dir}")


def get_all_conditions():
    conditions = []
    for cond in ['correct', 'reversed', 'shuffled', 'random']:
        conditions.append(('exp1', cond))
    for sigma in ['0.0', '0.1', '0.2', '0.4']:
        conditions.append(('exp4', sigma))
    for cond in ['full', 'no_lstm', 'no_tdvae', 'no_blueprint']:
        conditions.append(('exp6', cond))
    return conditions


def main():
    import pickle
    parser = argparse.ArgumentParser(
        description="Recreation accuracy (Figure S5 protocol)")
    parser.add_argument("--experiment", choices=["exp1", "exp4", "exp6"])
    parser.add_argument("--condition")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--original", action="store_true",
                        help="Evaluate original manuscript checkpoint with Root_Dataset")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_batches", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    if args.original:
        # ── Evaluate original checkpoint with Root_Dataset ────────────────
        print("\n=== Original manuscript checkpoint (Root_Dataset) ===")
        ORIG_CKPT = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys/workstation/genesys_model_trained_on_root_atlas_20240308_continue4.pth"
        ROOT_DATA = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys/genesys_root_data.pkl"

        with open(ROOT_DATA, 'rb') as f:
            data = pickle.load(f)

        ds = Root_Dataset(data['X_test'], data['y_test'])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

        n_genes = data['X_test'].shape[1]
        model = ClassifierLSTM(
            input_size=n_genes, output_size=10,
            embedding_dim=256, hidden_dim=256, n_layers=2, device=device
        ).to(device)
        ckpt = torch.load(ORIG_CKPT, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        print(f"  Checkpoint: {ORIG_CKPT}")
        print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

        result = recreation_accuracy(model, loader, device,
                                     batch_size=args.batch_size,
                                     n_timebins=11, n_batches=args.n_batches)

        print(f"\n  Mean recreation accuracy: {result['mean_acc']:.4f}")
        for t in sorted(result['acc_per_bin'].keys(), key=int):
            print(f"    t{t}: {result['acc_per_bin'][t]:.4f}")

        out_path = os.path.join(RESULTS_DIR, 'original_recreation_acc_v2.json')
        with open(out_path, 'w') as f:
            json.dump({
                'experiment': 'original', 'condition': 'original_600ep',
                'metric': 'recreation_accuracy_figS5_protocol',
                'pipeline': 'Root_Dataset',
                **result
            }, f, indent=2)
        print(f"  Saved: {out_path}")
        return

    # ── Retrained models with Cell_Lineage_Blueprint ─────────────────────
    if args.all:
        conditions = get_all_conditions()
    elif args.checkpoint:
        conditions = [('direct', args.label or 'checkpoint')]
    elif args.experiment and args.condition:
        conditions = [(args.experiment, args.condition)]
    else:
        parser.error("Specify --experiment/--condition, --all, --original, or --checkpoint")

    data = load_root_data_blueprint()
    lineage = load_lineage()
    n_genes = data['X_val'].shape[1]
    n_traj = lineage.shape[0]
    n_tb = lineage.shape[1]

    ds = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for experiment, condition in conditions:
        print(f"\n{'='*60}")
        print(f"[{experiment}/{condition}] Recreation accuracy (Fig S5 protocol)")
        print(f"{'='*60}")

        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            try:
                ckpt_path = get_checkpoint_path(experiment, condition)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue
        print(f"  Checkpoint: {ckpt_path}")

        if experiment == 'exp6' and condition == 'no_lstm':
            model = ClassifierMLP(n_genes, n_traj, 256, 256, 2, device).to(device)
        elif experiment == 'exp4':
            model = make_model_custom(n_genes, n_traj, device, sigma=float(condition))
        else:
            model = make_model_custom(n_genes, n_traj, device, sigma=0.2)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        result = recreation_accuracy(model, loader, device,
                                     batch_size=args.batch_size,
                                     n_timebins=n_tb, n_batches=args.n_batches)

        print(f"\n  Mean recreation accuracy: {result['mean_acc']:.4f}")
        for t in sorted(result['acc_per_bin'].keys(), key=int):
            print(f"    t{t}: {result['acc_per_bin'][t]:.4f}")

        # Save
        if args.checkpoint:
            out_dir = RESULTS_DIR
            out_file = os.path.join(out_dir, f'{condition}_recreation_acc_v2.json')
        else:
            exp_dirs = {
                'exp1': f'exp1_decoy/{condition}',
                'exp4': f'exp4_noise/sigma_{condition}',
                'exp6': f'exp6_ablation/{condition}',
            }
            out_dir = os.path.join(RESULTS_DIR, exp_dirs[experiment])
            out_file = os.path.join(out_dir, f'{experiment}_{condition}_recreation_acc_v2.json')

        with open(out_file, 'w') as f:
            json.dump({
                'experiment': experiment, 'condition': condition,
                'metric': 'recreation_accuracy_figS5_protocol',
                'pipeline': 'Cell_Lineage_Blueprint',
                **result
            }, f, indent=2)
        print(f"  Saved: {out_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
