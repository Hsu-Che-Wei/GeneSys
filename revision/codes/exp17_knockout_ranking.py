#!/usr/bin/env python3
"""
Experiment 17: In-silico knockout regulator ranking [Round 3, R3 points 5-6]

Fatecode-style perturbation analysis executed inside GeneSys: knock out each
gene in the input (zero its expression across all time bins), propagate
through the trained model, and measure how much the model's generated next-
state transcriptomes and cell-type predictions change. Genes are ranked by
perturbation effect and scored against the same 140-TF gold standard as
Figure 4 (R50 within the 1,495 expressed TFs) and genome-wide (recall@K over
all 17,513 genes, the Figure S18 protocol).

This provides the nonlinear, model-based regulator-ranking alternative to
LIMA's linear regression that R3 asked for (Fatecode/CLERA discussion):
if perturbation ranking recovers gold-standard TFs better than LIMA
centrality, the nonlinear pathway captures signal the linear readout loses.

Reported protocol (--protocol seeded, the default): fully seeded generation of notebook 4
(cells 144-147). The gene is set to zero at every time bin of the input, t0 is generated with
generate_current and each next state with generate_next from the real (knocked-out) input,
with predict_proba carrying the hidden state. --protocol rollout runs the knockout rollout of
notebook 4 cells 179-180 instead (seed t0 to t2, then autoregressive). In both, the random
generator is reset to the same seed for the unperturbed and for every knocked-out run, so
that they share the latent noise and differ only by the knockout.

Usage:
  sbatch --job-name=exp17_ko --gres=gpu:nvidia_h200:1 \
      experiments/slurm/run_gpu.sh experiments/exp17_knockout_ranking.py
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from utils import (load_root_data, load_lineage, make_model, load_checkpoint,
                   RESULTS_DIR, ROOT_CKPT)

DATA_DIR = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
OUT_DIR = os.path.join(RESULTS_DIR, "exp17_knockout")


def build_sequences(data, lineage, per_traj, rng):
    """Blueprint sequences from the test split, original Root_Dataset
    convention: slot 0 = QC cell (any bin), slot k = (trajectory cell type,
    time bin k)."""
    X = data['X_test']
    labels = data['y_test']
    ct_names = np.array([lab.split('_', 1)[0] for lab in labels])
    ct_times = np.array([float(lab.split('_', 1)[1]) for lab in labels])
    n_tb = lineage.shape[1]
    n_traj = lineage.shape[0]

    qc_idx = np.where(ct_names == 'Quiescent Center')[0]
    seqs, ys = [], []
    for j in range(n_traj):
        for _ in range(per_traj):
            seq = np.zeros((n_tb, X.shape[1]), dtype=np.float32)
            seq[0] = X[rng.choice(qc_idx)]
            for t in range(1, n_tb):
                ct = lineage.iloc[j, t]
                idx = np.where((ct_names == ct) & (ct_times == float(t)))[0]
                if len(idx):
                    seq[t] = X[rng.choice(idx)]
            seqs.append(seq)
            ys.append(j)
    return np.stack(seqs), np.array(ys)


@torch.no_grad()
def rollout(model, xm, seed):
    """Knockout rollout of notebook 4 (cells 179-180), written as a loop.
    xm: (B, 11, G) input with the knocked-out genes already set to zero.
    The model is seeded with t0, t1, t2 (t2 repeated), then generates t0 to t10 with
    generate_current / generate_next and classifies with predict_proba, carrying the hidden
    state as in the notebook. The random generator is reset to `seed` first, so that the
    unperturbed and the knocked-out rollouts draw the same latent noise and differ only by
    the knockout. Returns generated states (B, 11, G) and class probabilities (B, 10, n)."""
    torch.manual_seed(seed)
    B, T = xm.shape[0], xm.shape[1]
    x = torch.stack([xm[:, 0, :], xm[:, 1, :]] + [xm[:, 2, :]] * (T - 2), dim=1)
    pred_h = model.init_hidden(B)
    gen = [model.generate_current(x, pred_h, 0), model.generate_next(x, pred_h, 0)]
    y, pred_h = model.predict_proba(x, pred_h, 1)
    probs = [y]
    gen.append(model.generate_next(x, pred_h, 1))
    for t in range(2, T):
        update_x = torch.stack(gen + [gen[-1]] * (T - len(gen)), dim=1)
        y, pred_h = model.predict_proba(update_x, pred_h, t)
        probs.append(y)
        if t < T - 1:
            gen.append(model.generate_next(update_x, pred_h, t))
    return torch.stack(gen, dim=1), torch.stack(probs, dim=1)


@torch.no_grad()
def seeded(model, xm, seed):
    """Fully seeded generation of notebook 4 (cells 144-147, the protocol behind Figure 2),
    applied to an input in which the gene is zero at every time bin: t0 from
    generate_current, t(k+1) from generate_next(x, h, k) on the real (knocked-out) input, with
    predict_proba carrying the hidden state. Same seed for unperturbed and knocked-out runs.
    Returns generated states (B, 11, G) and class probabilities (B, 10, n) for t1 to t10."""
    torch.manual_seed(seed)
    B, T = xm.shape[0], xm.shape[1]
    pred_h = model.init_hidden(B)
    gen = [model.generate_current(xm, pred_h, 0)]
    _, pred_h = model.predict_proba(xm, pred_h, 0)
    probs = []
    for k in range(0, T - 1):
        gen.append(model.generate_next(xm, pred_h, k))
        y, pred_h = model.predict_proba(xm, pred_h, k + 1)
        probs.append(y)
    return torch.stack(gen, dim=1), torch.stack(probs, dim=1)


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description="Exp 17: knockout ranking")
    ap.add_argument('--per-traj', type=int, default=10)
    ap.add_argument('--protocol', choices=['rollout', 'seeded'], default='seeded',
                    help='rollout: notebook 4 cells 179-180 (seed t0-t2, autoregressive); '
                         'seeded: notebook 4 cells 144-147 (all time bins supplied)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = args.device
    rng = np.random.RandomState(args.seed)
    print(f"[Exp 17] device={device}")

    data = load_root_data()
    lineage = load_lineage()
    n_genes = data['X_test'].shape[1]
    n_traj = lineage.shape[0]

    model = make_model(n_genes, output_size=n_traj, device=device)
    model = load_checkpoint(model, ROOT_CKPT, device=device)
    model.eval()

    seqs, ys = build_sequences(data, lineage, args.per_traj, rng)
    x = torch.from_numpy(seqs).to(device)      # (B, 11, G)
    B = x.shape[0]
    print(f"Sequences: {tuple(x.shape)}")

    run = rollout if args.protocol == 'rollout' else seeded
    if args.protocol == 'rollout':
        OUT_DIR = OUT_DIR + '_rollout'
    if args.seed != 0:
        OUT_DIR = OUT_DIR + f'_seed{args.seed}'   # replicate runs; seed 0 is the reported one
    os.makedirs(OUT_DIR, exist_ok=True)
    xhat_wt, probs_wt = run(model, x, args.seed)
    xhat_wt_n = F.normalize(xhat_wt, dim=2)
    # the rollout is reproducible: a second unperturbed rollout must return the same output
    chk, _ = run(model, x, args.seed)
    assert torch.allclose(chk, xhat_wt), 'rollout is not reproducible under a fixed seed'

    gen_effect = np.zeros(n_genes, dtype=np.float64)
    cls_effect = np.zeros(n_genes, dtype=np.float64)

    for g in range(n_genes):
        xm = x.clone()
        xm[:, :, g] = 0
        xhat_ko, probs_ko = run(model, xm, args.seed)
        cos = (F.normalize(xhat_ko, dim=2) * xhat_wt_n).sum(dim=2)      # (B, 11)
        gen_effect[g] = float((1.0 - cos).mean())
        cls_effect[g] = float(0.5 * (probs_ko - probs_wt).abs().sum(dim=2).mean())
        if g % 1000 == 0:
            print(f"  {g}/{n_genes}", flush=True)

    gene_list = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))
    out = pd.DataFrame({
        'GeneID': gene_list['features'],
        'gen_effect': gen_effect,
        'cls_effect': cls_effect,
    })
    out['combined_effect'] = (
        (gen_effect - gen_effect.min()) / (gen_effect.max() - gen_effect.min()) +
        (cls_effect - cls_effect.min()) / (cls_effect.max() - cls_effect.min()))
    out_csv = os.path.join(OUT_DIR, 'exp17_knockout_scores.csv')
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # ── Scoring: R50 within expressed TFs + genome-wide recall@K ──
    sys.path.insert(0, os.path.dirname(__file__))
    from exp13_fig4_r50 import load_refs, r50_scan, permutation_r50
    wanted, exptf, gold_set, stringdb = load_refs()
    exptf_set = set(exptf)

    results = {}
    for col in ['gen_effect', 'cls_effect', 'combined_effect']:
        ranked_all = out.sort_values(col, ascending=False, kind='stable')
        # Within expressed TFs (Fig 4 universe, r50 = 70)
        ranked_tf = ranked_all[ranked_all['GeneID'].isin(exptf_set)]
        r50_tf = r50_scan(ranked_tf['GeneID'].values, gold_set, 70)
        # Genome-wide recall@K (Fig S18 universe)
        recalls = {}
        for K in [100, 200, 300, 400]:
            topk = set(ranked_all['GeneID'].head(K))
            recalls[K] = len(topk & gold_set) / len(gold_set)
        results[col] = {'r50_within_tf': r50_tf, 'genomewide_recall': recalls}
        print(f"{col}: R50(within-TF)={r50_tf}, recall@K={recalls}")

    results['permutation_r50'] = permutation_r50(exptf, gold_set, 70)
    results['random_recall_at_K'] = {
        K: K * len(gold_set) / n_genes / len(gold_set) for K in [100, 200, 300, 400]}
    with open(os.path.join(OUT_DIR, 'exp17_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("[Exp 17] Done!")


if __name__ == '__main__':
    main()
