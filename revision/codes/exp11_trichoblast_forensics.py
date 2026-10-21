#!/usr/bin/env python3
"""
Experiment 11: Trichoblast Default Forensics [Round 3, R3 point 7]

Round 3 R3 flagged an internal contradiction: R3.6 said class imbalance is
not a factor (blueprint-balanced training), while R1.9/Figure S12 attributed
the random model's 16.7% floor to "class imbalance favouring trichoblast."
R3 also noted 16.7% is not chance for 10-11 balanced classes.

This experiment resolves the contradiction with three analyses:

  A. Evaluation-slot composition. The Figure 3 / S12 gene-masking protocol
     (exp2) evaluates every validation cell whose type matches a trajectory's
     terminal cell type, with no per-type cap. The expected score of a
     CONSTANT predictor that always outputs class c is therefore the
     frequency of class c among evaluated cells, i.e. the natural cell-type
     composition of the atlas, not 1/n_classes. We compute this frequency
     for every class to test whether trichoblast's share explains 16.7%.

  B. Random-initialization seeds. For N seeds, build an untrained model and
     record its prediction on an all-zero (fully masked) input at each time
     bin. If the default class varies across seeds, the trichoblast default
     is an initialization accident, not an architectural prior; if it is
     always trichoblast, there is a real prior to explain.

  C. Trained-model default. The published checkpoint's prediction on
     all-zero input, plus the distance from the zero-input belief state to
     each class's mean belief state, to characterize why the trained model
     lands where it lands under full masking.

Usage:
  sbatch --job-name=exp11_trich experiments/slurm/run_cpu.sh experiments/exp11_trichoblast_forensics.py
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from genesys_model import ClassifierLSTM

from utils import (load_root_data, load_lineage, make_model, load_checkpoint,
                   RESULTS_DIR, ROOT_CKPT, NUM2NAME, NAME2NUM)

OUT_DIR = os.path.join(RESULTS_DIR, "exp11_trichoblast")


def slot_composition(data, lineage):
    """Analysis A: expected constant-predictor score per class under the
    exp2 gene-masking evaluation protocol."""
    rows = []
    for split in ['y_train', 'y_val', 'y_test']:
        if split not in data:
            continue
        labels = data[split]
        ct_names = np.array([lab.split('_', 1)[0] for lab in labels])

        # Terminal cell type of each trajectory = evaluation groups
        terminal_types = [lineage.iloc[j, lineage.shape[1] - 1]
                          for j in range(lineage.shape[0])]
        counts = {ct: int((ct_names == ct).sum()) for ct in terminal_types}
        total = sum(counts.values())
        for j, ct in enumerate(terminal_types):
            rows.append({
                'split': split,
                'trajectory': j,
                'terminal_cell_type': ct,
                'n_cells': counts[ct],
                'fraction_of_evaluated': counts[ct] / total if total else 0.0,
            })
    df = pd.DataFrame(rows)
    return df


def random_seed_defaults(input_size, n_traj, n_timebins, n_seeds, device):
    """Analysis B: default class of untrained models across seeds."""
    rows = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make_model(input_size, output_size=n_traj, device=device)
        model.eval()

        x = torch.zeros(1, n_timebins, input_size, device=device)
        h = model.init_hidden(1)
        with torch.no_grad():
            for t in range(n_timebins):
                p, h = model.predict_proba(x, h, t)
                probs = p[0].cpu().numpy()
                rows.append({
                    'seed': seed,
                    'time_bin': t,
                    'pred_trajectory': int(probs.argmax()),
                    'pred_prob': float(probs.max()),
                })
        del model
    return pd.DataFrame(rows)


def trained_model_default(data, lineage, device):
    """Analysis C: published checkpoint behavior on all-zero input."""
    input_size = data['X_val'].shape[1]
    n_traj = lineage.shape[0]
    n_tb = lineage.shape[1]

    model = make_model(input_size, output_size=n_traj, device=device)
    model = load_checkpoint(model, ROOT_CKPT, device=device)
    model.eval()

    x = torch.zeros(1, n_tb, input_size, device=device)
    h = model.init_hidden(1)
    zero_rows = []
    with torch.no_grad():
        for t in range(n_tb):
            p, h = model.predict_proba(x, h, t)
            probs = p[0].cpu().numpy()
            zero_rows.append({
                'time_bin': t,
                'pred_trajectory': int(probs.argmax()),
                'pred_prob': float(probs.max()),
                'full_probs': probs.round(4).tolist(),
            })

    # Belief-state geometry: zero-input belief vs per-type mean belief
    # Sample up to 200 validation cells per terminal type, embed as
    # repeat-padded sequences, take belief at middle bin.
    labels = data['y_val']
    ct_names = np.array([lab.split('_', 1)[0] for lab in labels])
    X = data['X_val']
    terminal_types = [lineage.iloc[j, n_tb - 1] for j in range(n_traj)]

    mid = n_tb // 2
    with torch.no_grad():
        h0 = model.init_hidden(1)
        model.get_belief(x, h0)
        zero_belief = model.b[0, mid, :].cpu().numpy()

        dist_rows = []
        for j, ct in enumerate(terminal_types):
            idx = np.where(ct_names == ct)[0][:200]
            if len(idx) == 0:
                continue
            beliefs = []
            for ci in idx:
                seq = np.repeat(X[ci][None, None, :], n_tb, axis=1).astype(np.float32)
                xc = torch.from_numpy(seq).to(device)
                hc = model.init_hidden(1)
                model.get_belief(xc, hc)
                beliefs.append(model.b[0, mid, :].cpu().numpy())
            mean_b = np.mean(beliefs, axis=0)
            dist_rows.append({
                'trajectory': j,
                'terminal_cell_type': ct,
                'n_cells': len(idx),
                'euclidean_dist_zero_to_mean_belief':
                    float(np.linalg.norm(zero_belief - mean_b)),
                'cosine_sim_zero_to_mean_belief': float(
                    np.dot(zero_belief, mean_b) /
                    (np.linalg.norm(zero_belief) * np.linalg.norm(mean_b) + 1e-12)),
            })

    return pd.DataFrame(zero_rows), pd.DataFrame(dist_rows)


def main():
    parser = argparse.ArgumentParser(description="Exp 11: Trichoblast forensics")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = args.device
    print(f"[Exp 11] Device: {device}")

    data = load_root_data()
    lineage = load_lineage()
    input_size = data['X_val'].shape[1]
    n_traj = lineage.shape[0]
    n_tb = lineage.shape[1]
    print(f"  input_size={input_size}, trajectories={n_traj}, time_bins={n_tb}")

    # ── A: slot composition ──
    print("\n=== A: Evaluation-slot composition ===")
    comp = slot_composition(data, lineage)
    comp.to_csv(os.path.join(OUT_DIR, "exp11_slot_composition.csv"), index=False)
    val_comp = comp[comp['split'] == 'y_val'].sort_values(
        'fraction_of_evaluated', ascending=False)
    print(val_comp[['terminal_cell_type', 'n_cells',
                    'fraction_of_evaluated']].to_string(index=False))

    # ── B: random-seed defaults ──
    print(f"\n=== B: Random-init defaults across {args.n_seeds} seeds ===")
    seed_df = random_seed_defaults(input_size, n_traj, n_tb, args.n_seeds, device)
    seed_df.to_csv(os.path.join(OUT_DIR, "exp11_random_seed_defaults.csv"), index=False)
    mid = n_tb // 2
    mid_defaults = seed_df[seed_df['time_bin'] == mid]
    terminal_types = [lineage.iloc[j, n_tb - 1] for j in range(n_traj)]
    print("Default class at middle bin by seed:")
    for _, r in mid_defaults.iterrows():
        print(f"  seed {int(r['seed'])}: trajectory {int(r['pred_trajectory'])} "
              f"({terminal_types[int(r['pred_trajectory'])]}) "
              f"p={r['pred_prob']:.3f}")

    # ── C: trained-model default ──
    print("\n=== C: Trained model (published checkpoint) on zero input ===")
    zero_df, dist_df = trained_model_default(data, lineage, device)
    zero_df.to_csv(os.path.join(OUT_DIR, "exp11_trained_zero_input.csv"), index=False)
    dist_df.to_csv(os.path.join(OUT_DIR, "exp11_trained_belief_distances.csv"), index=False)
    print(zero_df[['time_bin', 'pred_trajectory', 'pred_prob']].to_string(index=False))
    print("\nBelief geometry (zero input vs per-type mean):")
    print(dist_df.to_string(index=False))

    # ── Summary ──
    tricho_traj = next((j for j, ct in enumerate(terminal_types)
                        if ct == 'Trichoblast'), None)
    tricho_frac = float(val_comp[val_comp['terminal_cell_type'] == 'Trichoblast']
                        ['fraction_of_evaluated'].iloc[0]) if tricho_traj is not None else None
    summary = {
        'trichoblast_trajectory_index': tricho_traj,
        'trichoblast_fraction_of_evaluated_val_cells': tricho_frac,
        'observed_random_model_floor': 0.167,
        'explanation_A': 'Constant predictor scores its class frequency among '
                         'evaluated cells (atlas composition), not 1/n_classes.',
        'seed_default_classes_mid_bin': {
            str(int(r['seed'])): terminal_types[int(r['pred_trajectory'])]
            for _, r in mid_defaults.iterrows()},
        'trained_zero_input_default': terminal_types[int(
            zero_df.iloc[mid]['pred_trajectory'])],
    }
    with open(os.path.join(OUT_DIR, "exp11_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    print("\n[Exp 11] Done!")


if __name__ == "__main__":
    main()
