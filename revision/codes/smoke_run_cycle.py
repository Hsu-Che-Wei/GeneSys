#!/usr/bin/env python3
"""Two-epoch smoke test of every training variant through run_cycle (tiny synthetic data),
to confirm that each call path works before the 30-epoch jobs are submitted."""
import os, sys, numpy as np, pandas as pd, torch, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_run_cycle as trc
from exp6_ablation import ClassifierMLP
from exp3_kl_divergence_from_scratch import calculate_loss_decomposed
from utils import load_lineage
lin = load_lineage(); G = 50; rng = np.random.RandomState(0)
labs = [f"Quiescent Center_1.0"] * 40 + [f"{ct}_{float(t)}" for ct in sorted(set(lin.iloc[:, 1])) for t in range(2, 12) for _ in range(6)]
X = rng.rand(len(labs), G).astype(np.float32); y = np.array(labs)
dev = 'cpu'; tmp = tempfile.mkdtemp()
for name in ['lstm', 'no_tdvae', 'mlp', 'kl']:
    if name == 'mlp':
        m = ClassifierMLP(G, 10, 16, 16, 2, dev).to(dev)
    else:
        m = trc.make_lstm(G, 10, dev, dim=16)
    if name == 'no_tdvae':
        before = {k: v.clone() for k, v in m.state_dict().items() if 'b_to_z' in k}
        trc.switch_off_tdvae(m)
    tl, vl = trc.make_loaders(X, y, X, y, lin, batch_size=32, workers=0)
    if name == 'kl':
        kl = trc.KLLog(m, vl, 11, calculate_loss_decomposed); vl = kl
    st, bl, ba, lg = trc.train(m, tl, vl, dev, tmp, f"smoke_{name}", epochs=2, batch_size=32, time_bins=11)
    print(f"[{name}] ok: best_loss={bl:.3f} val_acc={ba:.3f} epochs_logged={len(lg['tloss'])}", flush=True)
    if name == 'no_tdvae':
        same = all(torch.equal(before[k], st[k]) for k in before)
        print(f"    TD-VAE prior network untouched by training: {same}")
    if name == 'kl':
        print('    KL rows:', len(kl.rows), {k: round(float(v), 2) for k, v in kl.rows[-1].items() if k != 'epoch'})
print('[smoke] Done!')
