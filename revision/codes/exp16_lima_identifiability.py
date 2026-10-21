#!/usr/bin/env python3
"""
Experiment 16: LIMA identifiability, regularization, and diversity [R3 point 2]

R3 argues: (a) the LIMA fit is unidentifiable (17,513 unknowns per gene from
~200 rows; lstsq returns the minimum-norm solution); (b) averaging replicates
stabilizes an arbitrary solution; (c) better denoising (less cell-to-cell
spread) makes the regression LESS reliable; (d) generating more cells cannot
help because they span a low-rank subspace.

(a) is mathematically correct and is conceded in the text. (b)-(d) contain
falsifiable empirical claims that this experiment tests directly on the
published checkpoint, transition t1-t3:

  1. Solver comparison: the pipeline call of notebook 12 (torch.linalg.lstsq on CPU
     tensors, which keeps only the singular directions above its default rank cutoff),
     the minimum-norm solution with every singular direction kept, and ridge regression
     with lambda spanning three orders of magnitude. Does the TF ranking (and its R50
     against the gold standard) depend on the solver?
  2. Row-count sweep (50 / 200 / 800 generated cells per state): does more
     generated data help, hurt, or nothing?
  3. Latent-spread sweep (epsilon scale 0.5 / 1 / 2 in TD-VAE sampling):
     directly manipulates cell-to-cell spread of the generated states.
     R3's prediction: less spread -> less reliable LIMA.
  4. Split-half stability: Spearman correlation of TF degree rankings
     between two independent replicate halves, per condition.

The effective rank of every pipeline fit is recorded (exp16_ranks.csv). The minimum-norm and
ridge solutions are obtained through the singular value decomposition. One job per condition
on CPU, then a merge job that writes the result tables.

Usage:
  for i in 0 1 2 3 4 5 6 7 8 9; do
    sbatch --job-name=e16c_$i --time=06:00:00 experiments/slurm/run_cpu.sh \
        experiments/exp16_lima_identifiability.py --only $i --device cpu
  done
  sbatch --job-name=e16c_merge --dependency=afterok:<the ten job ids> \
      experiments/slurm/run_cpu.sh experiments/exp16_lima_identifiability.py --merge
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GeneSys", "genesys"))
from utils import (load_root_data, load_lineage, make_model, load_checkpoint,
                   RESULTS_DIR, ROOT_CKPT)

DATA_DIR = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
OUT_DIR = os.path.join(RESULTS_DIR, "exp16_identifiability")
CTS = ['Atrichoblast', 'Trichoblast', 'Cortex', 'Endodermis', 'Pericycle',
       'Procambium', 'Xylem', 'Phloem', 'Lateral Root Cap', 'Columella']
FROM_SLOT, TO_SLOT = 0, 2      # generate_next(x,h,0) -> t1, generate_next(x,h,2) -> t3
N_REPS = 11        # notebook 12: the 11 batches of 2,000 trajectories of the test split
ZTHRESH = 3.0
KS = [100, 200, 300, 400]


def build_batch(data, lineage, batch_size, rng):
    """Blueprint sequences (original Root_Dataset convention) from test split."""
    X = data['X_test']
    labels = data['y_test']
    ct_names = np.array([lab.split('_', 1)[0] for lab in labels])
    ct_times = np.array([float(lab.split('_', 1)[1]) for lab in labels])
    n_tb = lineage.shape[1]
    qc_idx = np.where(ct_names == 'Quiescent Center')[0]

    traj = rng.randint(0, lineage.shape[0], size=batch_size)
    seq = np.zeros((batch_size, n_tb, X.shape[1]), dtype=np.float32)
    for i, j in enumerate(traj):
        seq[i, 0] = X[rng.choice(qc_idx)]
        for t in range(1, n_tb):
            ct = lineage.iloc[j, t]
            idx = np.where((ct_names == ct) & (ct_times == float(t)))[0]
            if len(idx):
                seq[i, t] = X[rng.choice(idx)]
    y_label = np.array([lineage.iloc[j, n_tb - 1] for j in traj])
    return seq, y_label


@torch.no_grad()
def generate_states(model, x, slot, eps_scale, device):
    """generate_next with controllable latent spread."""
    embeds = model.fc1(x)
    h = model.init_hidden(x.shape[0])
    b, _ = model.lstm(embeds, h)
    z1_mu, z1_ls = model.b_to_z(b[:, slot, :])
    z1 = z1_mu + torch.exp(z1_ls) * torch.randn_like(z1_mu) * eps_scale
    z2_mu, z2_ls = model.z1_to_z2(z1)
    z2 = z2_mu + torch.exp(z2_ls) * torch.randn_like(z2_mu) * eps_scale
    return model.z_to_x(z2)


@torch.no_grad()
def solve_w(A, B, lam, device):
    """Returns (W, rank). lam = 0: the pipeline call of notebook 12, torch.linalg.lstsq on CPU
    tensors (driver gelsy with its default rank cutoff, eps x max(m, n) of the largest pivot),
    whose effective rank is returned. lam = 'minnorm': the minimum-norm least-squares solution
    with every singular direction kept. lam > 0: the ridge solution, every direction kept."""
    if lam == 0:
        res = torch.linalg.lstsq(torch.tensor(A), torch.tensor(B))
        return res.solution.to(device), int(res.rank)
    A_t = torch.tensor(A, device=device)
    B_t = torch.tensor(B, device=device)
    U, S, Vh = torch.linalg.svd(A_t, full_matrices=False)
    inv = 1.0 / S if lam == 'minnorm' else S / (S ** 2 + lam)
    W = Vh.T @ (inv[:, None] * (U.T @ B_t))
    return W, int(len(S))


def tf_edge_mask(W_mean, zthresh):
    """|z| > zthresh edges of the TF-by-TF block (matrix-wide z-score), bit-packed."""
    from exp29_grnboost_states import tf_columns
    if 'idx' not in _TF:
        _TF['idx'], _ = tf_columns()
    idx = np.asarray(_TF['idx'])
    std = W_mean.std()
    z = (W_mean - W_mean.mean()) / (std if std > 0 else 1.0)
    return np.packbits(np.abs(z[np.ix_(idx, idx)]) > zthresh)


def degree_from_w(W_mean, zthresh):
    std = W_mean.std()
    if std == 0:
        return np.zeros(W_mean.shape[0])
    z = (W_mean - W_mean.mean()) / std
    mask = np.abs(z) > zthresh
    absw = np.abs(W_mean) * mask
    return (absw > 0).sum(axis=0) + (absw > 0).sum(axis=1)


_TF = {}


def tf_degree_from_w(W_mean, zthresh):
    """TF degree centrality as in notebook 12 (network()): matrix-wide z-score, TF-by-TF
    subnetwork, TFs with at least one edge in each direction, NetworkX degree centrality;
    min-max normalized per cell type as in the Figure 4 scoring. Returns a vector over the
    1,495 TFs (0 for TFs absent from the network)."""
    from exp14_condition_pipeline import network_centralities
    from exp29_grnboost_states import tf_columns
    if 'idx' not in _TF:
        _TF['idx'], _ = tf_columns()
    tf_idx = _TF['idx']
    out = np.zeros(len(tf_idx))
    std = W_mean.std()
    if std == 0:
        return out
    mask = np.abs((W_mean - W_mean.mean()) / std) > zthresh
    df = network_centralities(W_mean, mask, tf_idx, [str(k) for k in range(len(tf_idx))])
    if len(df) == 0:
        return out
    d = df['degree_centrality'].values
    rng_ = d.max() - d.min()
    out[df.index.astype(int)] = (d - d.min()) / rng_ if rng_ > 0 else 0.0
    return out


def run_condition(model, data, lineage, device, rng, n_rows, eps_scale, lam,
                  n_reps=N_REPS):
    """Returns the genome-wide degree vectors (summed over cell types) for the two replicate
    halves and for all replicates, the TF degree vectors, the effective rank of every fit and
    the TF-block edge mask of each cell type."""
    n_genes = data['X_test'].shape[1]
    deg_h1 = np.zeros(n_genes)
    deg_h2 = np.zeros(n_genes)
    deg_all = np.zeros(n_genes)
    tf_all, tf_h1, tf_h2 = 0.0, 0.0, 0.0
    ranks, masks = [], []
    batch_size = n_rows * 10 + 200   # oversample so each ct gets ~n_rows

    for ct in CTS:
        Ws = []
        for rep in range(n_reps):
            x, y_label = build_batch(data, lineage, batch_size, rng)
            x_t = torch.from_numpy(x).to(device)
            tfrom = generate_states(model, x_t, FROM_SLOT, eps_scale, device)
            tto = generate_states(model, x_t, TO_SLOT, eps_scale, device)
            sel = np.where(y_label == ct)[0][:n_rows]
            A = tfrom[sel].cpu().numpy()
            B = tto[sel].cpu().numpy()
            W, rank = solve_w(A, B, lam, device)
            ranks.append((ct, rep, len(sel), rank))
            Ws.append(W.cpu().numpy().astype(np.float32))
        Ws = np.stack(Ws)
        half = n_reps // 2
        W_all = Ws.mean(axis=0)
        deg_h1 += degree_from_w(Ws[:half].mean(axis=0), ZTHRESH)
        deg_h2 += degree_from_w(Ws[half:].mean(axis=0), ZTHRESH)
        deg_all += degree_from_w(W_all, ZTHRESH)
        tf_h1 = tf_h1 + tf_degree_from_w(Ws[:half].mean(axis=0), ZTHRESH)
        tf_h2 = tf_h2 + tf_degree_from_w(Ws[half:].mean(axis=0), ZTHRESH)
        tf_all = tf_all + tf_degree_from_w(W_all, ZTHRESH)
        masks.append(tf_edge_mask(W_all, ZTHRESH))
        del Ws, W_all
        print(f"    {ct} done", flush=True)
    return deg_all, deg_h1, deg_h2, tf_all, tf_h1, tf_h2, ranks, np.stack(masks)


def score(deg_all, deg_h1, deg_h2, gene_ids, gold_set, exptf_set, tf_all, tf_h1, tf_h2):
    order = np.argsort(-deg_all, kind='stable')
    ranked = gene_ids[order]
    recalls = {K: len(set(ranked[:K]) & gold_set) / len(gold_set) for K in KS}
    tf_mask = np.array([g in exptf_set for g in gene_ids])
    rho_all, _ = spearmanr(deg_h1, deg_h2)
    # TF ranking: degree centrality of the TF network of notebook 12
    rho_tf, _ = spearmanr(tf_h1, tf_h2)
    return recalls, float(rho_all), float(rho_tf), tf_all


CONDITIONS = (
    [('ridge', 200, 1.0, lam) for lam in [0, 0.01, 0.1, 1.0, 10.0]] +
    [('minnorm', 200, 1.0, 'minnorm')] +
    [('rows', r, 1.0, 0) for r in [50, 800]] +
    [('spread', 200, s, 0) for s in [0.5, 2.0]]
)
BASE = 'ridge_rows200_eps1.0_lam0'


def label_of(kind, n_rows, eps, lam):
    return f"{kind}_rows{n_rows}_eps{eps}_lam{lam}"


def merge():
    """Assemble the per-condition parts into the result tables."""
    parts = os.path.join(OUT_DIR, 'parts')
    P = {}
    for c in CONDITIONS:
        lab = label_of(*c)
        P[lab] = np.load(os.path.join(parts, lab + '.npz'), allow_pickle=True)
    rows_out, agree, ranks = [], [], []
    # R50 of the TF degree ranking of this transition among the 1,495 TFs (gold standard of
    # Figure 4); ties, which include the TFs outside the network, are broken at random and the
    # R50 is the mean over 200 draws. The permutation baseline is the mean over 1,000 orders.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from exp13_fig4_r50 import load_refs, r50_scan
    from exp29_grnboost_states import tf_columns
    _, _, gold_set, _ = load_refs()
    tf_ids = tf_columns()[1]
    r50_rng = np.random.RandomState(0)

    def r50_of(vec):
        vals = []
        for _ in range(200):
            order = np.lexsort((r50_rng.rand(len(vec)), -vec))
            vals.append(r50_scan(list(tf_ids[order]), gold_set, 70))
        return float(np.mean(vals))
    perm = float(np.mean([r50_scan(list(tf_ids[r50_rng.permutation(len(tf_ids))]), gold_set, 70)
                          for _ in range(1000)]))
    base_tf = P[BASE]['tf_vec']
    base_m = np.unpackbits(P[BASE]['masks'], axis=1).astype(bool)
    for c in CONDITIONS:
        lab = label_of(*c)
        p = P[lab]
        row = json.loads(str(p['row']))
        rk = p['ranks']
        r = np.array([int(x[3]) for x in rk])
        row.update({'rank_min': int(r.min()), 'rank_median': float(np.median(r)),
                    'rank_max': int(r.max()), 'r50_tf_degree': r50_of(p['tf_vec']),
                    'r50_permutation': perm, 'n_tf_in_network': int((p['tf_vec'] > 0).sum())})
        rows_out.append(row)
        for ct, rep, n, rank in rk:
            ranks.append({'label': lab, 'cell_type': ct, 'batch': int(rep), 'rows': int(n),
                          'rank': int(rank)})
        m = np.unpackbits(p['masks'], axis=1).astype(bool)
        jac = [(m[i] & base_m[i]).sum() / max(1, (m[i] | base_m[i]).sum()) for i in range(len(m))]
        rho, _ = spearmanr(base_tf, p['tf_vec'])
        agree.append({'label': lab, 'spearman_vs_lstsq_baseline': float(rho),
                      'edge_jaccard_vs_baseline_mean': float(np.mean(jac)),
                      'edge_jaccard_vs_baseline_min': float(np.min(jac)),
                      'edge_jaccard_vs_baseline_max': float(np.max(jac))})
    pd.DataFrame(rows_out).to_csv(os.path.join(OUT_DIR, 'exp16_conditions.csv'), index=False)
    pd.DataFrame(agree).to_csv(os.path.join(OUT_DIR, 'exp16_ranking_agreement.csv'), index=False)
    pd.DataFrame(ranks).to_csv(os.path.join(OUT_DIR, 'exp16_ranks.csv'), index=False)
    print(pd.DataFrame(rows_out).to_string(index=False))
    print(pd.DataFrame(agree).to_string(index=False))
    print("[Exp 16] merged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only', type=int, default=None, help='index of the one condition to run')
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT_DIR, 'parts'), exist_ok=True)
    if args.merge:
        return merge()
    device = args.device

    data = load_root_data()
    lineage = load_lineage()
    n_genes = data['X_test'].shape[1]
    model = make_model(n_genes, output_size=lineage.shape[0], device=device)
    model = load_checkpoint(model, ROOT_CKPT, device=device)
    model.eval()

    sys.path.insert(0, os.path.dirname(__file__))
    from exp13_fig4_r50 import load_refs
    wanted, exptf, gold_set, stringdb = load_refs()
    exptf_set = set(exptf)
    gene_ids = pd.read_csv(os.path.join(DATA_DIR, 'gene_list_1108.csv'))['features'].values

    todo = list(enumerate(CONDITIONS)) if args.only is None else [(args.only, CONDITIONS[args.only])]
    for i, (kind, n_rows, eps, lam) in todo:
        label = label_of(kind, n_rows, eps, lam)
        print(f"[{i+1}/{len(CONDITIONS)}] {label}", flush=True)
        rng = np.random.RandomState(args.seed)   # same sampling stream per condition
        torch.manual_seed(args.seed)
        deg_all, h1, h2, tfa, tf1, tf2, ranks, masks = run_condition(
            model, data, lineage, device, rng, n_rows, eps, lam)
        recalls, rho_all, rho_tf, tf_vec = score(deg_all, h1, h2, gene_ids,
                                                 gold_set, exptf_set, tfa, tf1, tf2)
        row = {'condition': kind, 'n_rows': n_rows, 'eps_scale': eps,
               'lambda': -1.0 if lam == 'minnorm' else lam, 'label': label,
               'splithalf_spearman_all': rho_all, 'splithalf_spearman_tf': rho_tf,
               **{f'recall_at_{K}': v for K, v in recalls.items()}}
        np.savez(os.path.join(OUT_DIR, 'parts', label + '.npz'), tf_vec=tf_vec, masks=masks,
                 ranks=np.array(ranks, dtype=object), row=json.dumps(row))
        print(json.dumps(row), flush=True)
    if args.only is None:
        merge()
    print("[Exp 16] Done!")


if __name__ == '__main__':
    main()
