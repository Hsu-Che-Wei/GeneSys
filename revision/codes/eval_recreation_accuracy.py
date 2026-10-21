#!/usr/bin/env python3
"""
Evaluate models using the ORIGINAL autoregressive recreation accuracy protocol
from the manuscript (Notebook 3, Figure S5, Figure 2).

Protocol:
  1. Sample a batch from the lineage blueprint (real cells at all 11 time bins)
  2. Seed input with real cells at t0, t1 and t2; fill t3-t10 with repeats of t2
  3. generate_current(x, h, 0) → t0_gen
  4. generate_next(x, h, 0) → t1_gen
  5. predict_proba(x, h, 1) → classify t1
  6. generate_next(x, h, 1) → t2_gen
  7. Update input: [t0_gen, t1_gen, t2_gen, t2_gen, ...]
  8. predict_proba(update_x, h, 2) → classify t2
  9. Continue autoregressively through t10
  10. Compare predicted trajectory labels to ground truth at each time bin

This tests BOTH the generative (TD-VAE) and classification (softmax) components.

For "No TD-VAE" ablation models (trained with NLL-only loss), generate_next()
will still run but produces lower-quality transcriptomes since the TD-VAE
latent space was not optimized during training. We evaluate it anyway to
show the effect.

Usage:
  python eval_recreation_accuracy.py --experiment exp6 --condition full
  python eval_recreation_accuracy.py --experiment exp1 --condition correct
  python eval_recreation_accuracy.py --experiment exp4 --condition sigma_0.2
  python eval_recreation_accuracy.py --all   # evaluate all 12 conditions
"""

import argparse, os, sys, json, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint

from utils import (load_root_data, load_root_data_blueprint, load_lineage, load_checkpoint,
                   NUM2NAME, NAME2NUM, RESULTS_DIR)
from train_utils import make_model_custom
from exp6_ablation import ClassifierMLP


def autoregressive_recreation_accuracy(model, loader, device, time_bins=11,
                                       batch_size=512, n_batches=10):
    """
    Evaluate using the original autoregressive recreation accuracy protocol.

    Returns:
        dict with keys:
          - 'acc_per_bin': {t: accuracy} for t in 1..10
          - 'mean_acc': mean across t1..t10 (t0 is always "Stem Cell", not classified)
          - 'pred_labels': list of per-bin predicted labels (for inspection)
    """
    model.eval()

    all_correct = {t: 0 for t in range(1, time_bins)}
    all_total = {t: 0 for t in range(1, time_bins)}

    with torch.no_grad():
        batch_count = 0
        for sample in loader:
            if batch_count >= n_batches:
                break

            x_real = sample['x'].to(device)  # (batch, 11, n_genes) — real cells
            y_true = sample['y'].to(device)  # trajectory index

            if x_real.size(0) != batch_size:
                continue

            # Ground truth labels for comparison
            y_true_np = y_true.cpu().numpy()

            # The Cell_Lineage_Blueprint dataset provides x_real with shape
            # (batch, n_timebins, n_genes). Due to the time indexing convention
            # (time == k+1), some positions may be zero vectors if no matching
            # cells exist in the data. The autoregressive evaluation uses the
            # real cells as available in x_real, matching the training pipeline.
            #
            # Protocol (matching original notebook cell 16):
            #   1. Seed x with real data at positions 0, 1, 2; pad rest with position 2
            #   2. generate_current at t=0, generate_next at t=0 to get t0_gen, t1_gen
            #   3. predict_proba at t=1 to classify and update hidden state
            #   4. generate_next at t=1 to get t2_gen
            #   5. Build [t0_gen, t1_gen, t2_gen, t2_gen, ...], classify at t=2
            #   6. Continue autoregressively through the last time bin

            x = torch.stack(
                [x_real[:, 0, :], x_real[:, 1, :]] +
                [x_real[:, 2, :]] * (time_bins - 2),
                dim=1)

            h = model.init_hidden(batch_size)

            t_gen = [None] * time_bins
            t_gen[0] = model.generate_current(x, h, 0)
            t_gen[1] = model.generate_next(x, h, 0)

            # Classify t1 using the SEED
            p1, h = model.predict_proba(x, h, 1)
            pred_t1 = p1.argmax(dim=1).cpu().numpy()
            all_correct[1] += (pred_t1 == y_true_np).sum()
            all_total[1] += len(y_true_np)

            # Generate t2
            t_gen[2] = model.generate_next(x, h, 1)
            x = torch.stack(
                [t_gen[0], t_gen[1], t_gen[2]] +
                [t_gen[2]] * (time_bins - 3),
                dim=1)

            # Classify t2
            p2, h = model.predict_proba(x, h, 2)
            pred_t2 = p2.argmax(dim=1).cpu().numpy()
            all_correct[2] += (pred_t2 == y_true_np).sum()
            all_total[2] += len(y_true_np)

            # Autoregressive loop: t3 through last time bin
            for t in range(3, time_bins):
                t_gen[t] = model.generate_next(x, h, t - 1)
                components = [t_gen[k] for k in range(t + 1)]
                padding = [t_gen[t]] * (time_bins - t - 1)
                x = torch.stack(components + padding, dim=1)

                p_t, h = model.predict_proba(x, h, t)
                pred_t = p_t.argmax(dim=1).cpu().numpy()
                all_correct[t] += (pred_t == y_true_np).sum()
                all_total[t] += len(y_true_np)

            batch_count += 1

    # Compute per-bin accuracy
    acc_per_bin = {}
    for t in range(1, time_bins):
        acc_per_bin[t] = float(all_correct[t]) / max(all_total[t], 1)

    mean_acc = np.mean(list(acc_per_bin.values()))

    return {
        'acc_per_bin': acc_per_bin,
        'mean_acc': float(mean_acc),
        'n_samples_per_bin': {t: int(all_total[t]) for t in range(1, time_bins)},
    }


def get_checkpoint_path(experiment, condition):
    """Find the best checkpoint for a given experiment/condition."""
    exp_dirs = {
        'exp1': f'exp1_decoy/{condition}',
        'exp4': f'exp4_noise/sigma_{condition}',
        'exp6': f'exp6_ablation/{condition}',
    }
    subdir = exp_dirs.get(experiment, '')
    result_dir = os.path.join(RESULTS_DIR, subdir)

    # Try common checkpoint names
    for pattern in ['*_best.pth', '*_cycle1.pth', '*_ep30.pth', '*_ep20.pth']:
        matches = glob.glob(os.path.join(result_dir, pattern))
        if matches:
            return sorted(matches)[-1]  # latest

    raise FileNotFoundError(f"No checkpoint found in {result_dir}")


def get_all_conditions():
    """Return list of (experiment, condition) tuples for all 12 runs."""
    conditions = []
    for cond in ['correct', 'reversed', 'shuffled', 'random']:
        conditions.append(('exp1', cond))
    for sigma in ['0.0', '0.1', '0.2', '0.4']:
        conditions.append(('exp4', sigma))
    for cond in ['full', 'no_lstm', 'no_tdvae', 'no_blueprint']:
        conditions.append(('exp6', cond))
    return conditions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate with original autoregressive recreation accuracy")
    parser.add_argument("--experiment", choices=["exp1", "exp4", "exp6"])
    parser.add_argument("--condition")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all 12 conditions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Evaluate a single checkpoint file directly")
    parser.add_argument("--label", type=str, default=None,
                        help="Label for --checkpoint mode (used in output filename)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_batches", type=int, default=10,
                        help="Number of batches to evaluate (default 10 = 5120 samples)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint:
        # Direct checkpoint mode — evaluate a single .pth file
        conditions = [('direct', args.label or 'checkpoint')]
    elif args.all:
        conditions = get_all_conditions()
    elif args.experiment and args.condition:
        conditions = [(args.experiment, args.condition)]
    else:
        parser.error("Specify --experiment and --condition, --all, or --checkpoint")

    device = args.device
    print(f"Device: {device}")

    # Load data and lineage
    data = load_root_data_blueprint()
    lineage = load_lineage()
    n_genes = data['X_val'].shape[1]
    n_traj = lineage.shape[0]
    n_tb = lineage.shape[1]

    # Create validation loader
    ds = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for experiment, condition in conditions:
        print(f"\n{'='*60}")
        print(f"[{experiment}/{condition}] Autoregressive recreation accuracy")
        print(f"{'='*60}")

        # Load checkpoint
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            try:
                ckpt_path = get_checkpoint_path(experiment, condition)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue
        print(f"  Checkpoint: {ckpt_path}")

        # Create model matching training configuration
        if experiment == 'exp6' and condition == 'no_lstm':
            model = ClassifierMLP(n_genes, n_traj, 256, 256, 2, device).to(device)
        elif experiment == 'exp4':
            sigma = float(condition)
            model = make_model_custom(n_genes, n_traj, device, sigma=sigma)
        else:
            model = make_model_custom(n_genes, n_traj, device, sigma=0.2)

        # Load weights
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        # Run autoregressive evaluation
        result = autoregressive_recreation_accuracy(
            model, loader, device,
            time_bins=n_tb,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
        )

        print(f"\n  Mean recreation accuracy: {result['mean_acc']:.4f}")
        print(f"  Per-bin accuracy:")
        for t in sorted(result['acc_per_bin'].keys()):
            print(f"    t{t}: {result['acc_per_bin'][t]:.4f}")
        print(f"  Samples per bin: {result['n_samples_per_bin'][1]}")

        # Save result
        if args.checkpoint:
            out_dir = RESULTS_DIR
            out_file = os.path.join(out_dir, f'{condition}_recreation_acc.json')
        else:
            exp_dirs = {
                'exp1': f'exp1_decoy/{condition}',
                'exp4': f'exp4_noise/sigma_{condition}',
                'exp6': f'exp6_ablation/{condition}',
            }
            out_dir = os.path.join(RESULTS_DIR, exp_dirs[experiment])
            out_file = os.path.join(out_dir, f'{experiment}_{condition}_recreation_acc.json')
        result_out = {
            'experiment': experiment,
            'condition': condition,
            'metric': 'autoregressive_recreation_accuracy',
            'mean_acc': result['mean_acc'],
            'acc_per_bin': result['acc_per_bin'],
            'n_samples_per_bin': result['n_samples_per_bin'],
            'protocol': 'Seed t0-t2 with real cells, generate t0-t2 from the seed and t3-t10 autoregressively '
                        'using generate_next(), classify at each step.',
        }
        with open(out_file, 'w') as f:
            json.dump(result_out, f, indent=2)
        print(f"  Saved: {out_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
