#!/usr/bin/env python3
"""
Experiment 21: GeneSys-generated cell pool for the uniform DE builder.

Generates n cells per (cell type, time bin) state from the published
checkpoint (generate_next, stochastic latent sampling, notebook 12
convention) and saves them in the exp14 pool npz format, so the DE arm of
the expanded Figure 4 can be computed with ONE builder across all four
conditions (raw, pseudobulk, scVI, GeneSys-generated).

Usage:
  sbatch --job-name=exp21_pool --gres=gpu:nvidia_h200:1 --time=2:00:00 \
      experiments/slurm/run_gpu.sh experiments/exp21_genesys_pool.py
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_root_data, load_lineage, make_model, load_checkpoint, ROOT_CKPT, RESULTS_DIR
from exp16_lima_identifiability import build_batch, generate_states

OUT = os.path.join(RESULTS_DIR, "exp14_conditions", "genesys_gen_pool.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-state', type=int, default=220)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = args.device
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    data = load_root_data()
    lineage = load_lineage()
    n_genes = data['X_test'].shape[1]
    n_tb = lineage.shape[1]
    model = make_model(n_genes, output_size=lineage.shape[0], device=device)
    model = load_checkpoint(model, ROOT_CKPT, device=device)
    model.eval()

    X_out, names_out, times_out = [], [], []
    # generate_next(x, h, k) yields state k+1 -> covers bins 1..10 from k=0..9
    for k in range(0, n_tb - 1):
        # oversized batch so each trajectory contributes ~per_state cells
        x, y_label = build_batch(data, lineage, args.per_state * 10 + 200, rng)
        x_t = torch.from_numpy(x).to(device)
        gen = generate_states(model, x_t, k, 1.0, device).cpu().numpy()
        target_bin = k + 1
        for j in range(lineage.shape[0]):
            ct = lineage.iloc[j, target_bin]
            terminal = lineage.iloc[j, n_tb - 1]
            sel = np.where(y_label == terminal)[0][:args.per_state]
            X_out.append(gen[sel].astype(np.float32))
            names_out += [ct] * len(sel)
            times_out += [float(target_bin)] * len(sel)
        print(f"bin {target_bin} done", flush=True)

    X = np.vstack(X_out)
    np.savez_compressed(OUT, X=X,
                        ct_names=np.array(names_out, dtype=object),
                        ct_times=np.array(times_out, dtype=np.float32))
    print(f"Saved {OUT}: {X.shape}")


if __name__ == '__main__':
    main()
