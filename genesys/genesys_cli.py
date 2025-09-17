#!/usr/bin/env python3
import os, sys, math, copy, json, pickle, argparse, datetime
import numpy as np
from pathlib import Path
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.linalg as linalg
import re
import anndata
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, silhouette_score
from itertools import product

# ---------------------------
# Your model import goes here
from genesys_model import *
# ---------------------------

# ---------- Training helpers (ReduceLROnPlateau / cycles) ----------
def make_optimizer_and_scheduler(model, base_lr=1e-3, factor=0.5, patience=10, threshold=0.05, verbose=True):
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, threshold=threshold, verbose=verbose
    )
    return optimizer, sch

def run_cycle(
    model, train_loader, val_loader, device,
    base_lr=1e-3, epochs=100, factor=0.5, patience=10, threshold=0.05, verbose=True,
    save_prefix="genesys_training", path="./", batch_size=128, time_bins=10, clip=5
):
    """
    One 100-epoch cycle: train+validate; return cycle-best (by loss), plus logs.
    """
    optimizer, sch = make_optimizer_and_scheduler(model, base_lr, factor, patience, threshold, verbose)
    logger = {'tloss': [], 'val_acc': []}

    best_loss = math.inf
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    val_stats_at_best = None
    criterion = torch.nn.NLLLoss()

    for epoch in range(epochs):
        # ---- TRAIN ----
        h = model.init_hidden(batch_size)
        c, s = 0, 0.0
        pBar = tqdm(train_loader, leave=False)
        model.train()

        for sample in pBar:
            h = tuple([each.data for each in h])
            x = sample['x'].to(device)
            y = sample['y'].to(device)

            optimizer.zero_grad(set_to_none=True)

            t = np.random.choice(max(1, (time_bins - 1)))
            p, h = model.predict(x, h, t)
            model.get_belief(x, h)
            tdvae_loss = model.calculate_loss(t)
            nll_loss = criterion(p, y)

            loss = tdvae_loss + nll_loss

            batch_size_eff = len(p) if hasattr(p, "__len__") else x.size(0)
            s = ((s * c) + (float(loss.item()) * batch_size_eff)) / (c + batch_size_eff)
            c += batch_size_eff

            pBar.set_description(f"Epoch {epoch:03d} Train: {s:.4f}")

            wl = np.random.choice(2)
            if wl == 0:
                tdvae_loss.backward()
            else:
                nll_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        logger['tloss'].append(s)
        sch.step(s)

        # ---- VALIDATE ----
        val_h = model.init_hidden(batch_size)
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for sample in val_loader:
                val_h = tuple([each.data for each in val_h])
                x = sample['x'].to(device)
                y = sample['y'].to(device)
                t = 1
                p, val_h = model.predict_proba(x, val_h, t)
                y_pred.append(p.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        val_acc = (y_true == y_pred.argmax(axis=1)).mean()
        logger['val_acc'].append(float(val_acc))

        if verbose:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"[Cycle] epoch {epoch+1:3d}/{epochs}  loss {s:.4f}  val_acc {val_acc:.4f}  lr {curr_lr:.2e}")

        improved = (s < best_loss) or (np.isclose(s, best_loss) and val_acc > best_val_acc)
        if improved:
            best_loss = s
            best_val_acc = float(val_acc)
            best_state = copy.deepcopy(model.state_dict())
            val_stats_at_best = {'pred': y_pred, 'true': y_true}

    ckpt_path = os.path.join(path, f"{save_prefix}_cycle_best.pth")
    torch.save({"state_dict": best_state, "best_loss": best_loss, "best_val_acc": best_val_acc}, ckpt_path)
    if verbose:
        print(f"Saved cycle-best -> {ckpt_path}  (loss={best_loss:.6f}, val_acc={best_val_acc:.4f})")

    with open(os.path.join(path, f"log_{save_prefix}_cycle.pkl"), 'wb') as f:
        pickle.dump(logger, f)

    return best_state, best_loss, best_val_acc, logger, val_stats_at_best

def train_until_no_improve(
    model, train_loader, val_loader, device,
    base_lr=1e-3, factor=0.5, patience=10, threshold=0.05,
    epochs_per_cycle=100, max_cycles=20, verbose=True,
    save_prefix="genesys_training", path="./", batch_size=128, time_bins=10, clip=5
):
    os.makedirs(path, exist_ok=True)

    overall_best_state = copy.deepcopy(model.state_dict())
    prev_cycle_best_loss = math.inf
    history = []

    for cycle in range(1, max_cycles + 1):
        if verbose:
            print(f"\n=== Cycle {cycle} (epochs={epochs_per_cycle}) ===")

        model.load_state_dict(overall_best_state)

        best_state, best_loss, best_val_acc, logger, val_stats = run_cycle(
            model, train_loader, val_loader, device,
            base_lr=base_lr, epochs=epochs_per_cycle,
            factor=factor, patience=patience, threshold=threshold, verbose=verbose,
            save_prefix=f"{save_prefix}_cycle{cycle}", path=path,
            batch_size=batch_size, time_bins=time_bins, clip=clip
        )

        cycle_ckpt = os.path.join(path, f"{save_prefix}_cycle{cycle}_best.pth")
        torch.save(
            {"state_dict": best_state, "best_loss": best_loss,
             "best_val_acc": best_val_acc, "cycle": cycle},
            cycle_ckpt
        )
        if verbose:
            print(f"Saved cycle {cycle} best -> {cycle_ckpt}  (loss={best_loss:.6f}, val_acc={best_val_acc:.4f})")

        history.append({"cycle": cycle, "best_loss": best_loss, "best_val_acc": best_val_acc})

        if cycle > 1 and not (best_loss < prev_cycle_best_loss):
            if verbose:
                print("No improvement over previous cycle’s best loss; stopping.")
            break

        prev_cycle_best_loss = best_loss
        overall_best_state = copy.deepcopy(best_state)

    overall_path = os.path.join(path, f"{save_prefix}_overall_best.pth")
    torch.save({"state_dict": overall_best_state, "best_loss": prev_cycle_best_loss, "history": history}, overall_path)
    if verbose:
        print(f"\nOverall best saved -> {overall_path}  (loss={prev_cycle_best_loss:.6f})")

    return overall_best_state, prev_cycle_best_loss, history

# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser(prog="genesys", description="GeneSys trainer/evaluator CLI")
    p.add_argument("--anndata", required=True, help="Path to .h5ad file")
    p.add_argument("--bprint", required=True, help="Path to lineage.txt file")
    p.add_argument("--train", action="store_true", help="Run training")
    p.add_argument("--raw_counts", action="store_true", help="If the provided X is raw UMI counts")
    p.add_argument("--anno", default="./annotation.txt")
    p.add_argument("--epochs", type=int, default=100, help="Epochs per cycle")
    p.add_argument("--max_cycles", type=int, default=20, help="Max training cycles")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--factor", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--path", default="./checkpoints")
    p.add_argument("--save_prefix", type=str, default="Data_Set_1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true", help="Verbose logs")
    return p

def main():
    args = build_argparser().parse_args()

    if not os.path.exists(args.anndata):
        print(f"[ERR] anndata or cell-by-gene matrix not found: {args.anndata}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.bprint):
        print(f"[ERR] cell lineage blueprint not found: {args.bprint}", file=sys.stderr)
        sys.exit(1)

    if args.train:
        if ".h5ad" in args.anndata: 
            adata = sc.read_h5ad(args.anndata)
            sc.pp.filter_genes(adata, min_cells=3)
        else:
            adata = sc.read_10x_mtx(args.anndata)
            anno = pd.read_csv(args.anno, sep="\t")
            anno = anno[["barcode", "label", "time"]]  # ensure required columns exist
            anno = anno.set_index("barcode") 
            adata.obs["label"] = pd.Categorical(anno["label"])
            adata.obs["time"]  = pd.to_numeric(anno["time"], errors="raise")
            sc.pp.filter_genes(adata, min_cells=3)   

        if args.raw_counts :
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, max_value=10)
            adata.X[adata.X < 0] = 0
            mmin = np.amin(adata.X)
            nor = (np.amax(adata.X) - mmin)
            adata.X = (adata.X - mmin) / nor
        else:
            sc.pp.scale(adata, max_value=10)
            adata.X[adata.X < 0] = 0
            mmin = np.amin(adata.X)
            nor = (np.amax(adata.X) - mmin)
            adata.X = (adata.X - mmin) / nor

        adata.obs['labels'] = adata.obs['label'].astype(str) + "_" + adata.obs['time'].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            np.array(adata.X), adata.obs['labels'], test_size=0.2,
            stratify=adata.obs['labels'], shuffle=True, random_state=0
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, shuffle=True, random_state=0
        )
        data = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }

        with open("./genesys_X.pkl", 'wb') as file_handle:
            pickle.dump(data, file_handle)

        lineage = pd.read_csv(args.bprint, sep="\t", header=None)

        input_size = data['X_train'].shape[1]
        output_size = lineage.shape[0]
        embedding_dim = 256
        hidden_dim = 256
        n_layers = 2
        clip = 5
        base_lr = 1e-3
        train_on_gpu = True
        time_bins = lineage.shape[1]
        device = args.device
        if args.verbose:
            print(f"[INFO] Using device: {device}")

        train_dataset = Cell_Lineage_Blueprint(data['X_train'], data['y_train'], lineage)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = Cell_Lineage_Blueprint(data['X_val'], data['y_val'], lineage)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        model = ClassifierLSTM(input_size, output_size, embedding_dim, hidden_dim, n_layers, device).to(device)

        overall_state, overall_best_loss, cycles_hist = train_until_no_improve(
            model, train_loader, val_loader, device,
            base_lr=args.lr, factor=args.factor, patience=args.patience, threshold=args.threshold,
            epochs_per_cycle=args.epochs, max_cycles=args.max_cycles, verbose=True,
            save_prefix="genesys_training", path=args.path,
            batch_size=args.batch_size, time_bins=time_bins, clip=clip
        )

        all_logs = []
        log_plot_path = os.path.join(args.path, "genesys_training_logs.pdf")

        for cycle_id, cycle_result in enumerate(cycles_hist, 1):
            log_path = os.path.join(args.path, f"log_genesys_training_cycle{cycle_id}_cycle.pkl")
            with open(log_path, "rb") as f:
                logger = pickle.load(f)

            log_df = pd.DataFrame(logger)
            log_df["cycle"] = cycle_id
            log_df["epoch"] = log_df.index + 1
            log_df["global_epoch"] = (cycle_id - 1) * len(log_df) + log_df["epoch"]
            all_logs.append(log_df)

        full_log = pd.concat(all_logs, ignore_index=True)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Global Epoch')
        ax1.set_ylabel('Total Loss (classifier + TD-VAE)', color=color)
        ax1.plot(full_log["global_epoch"], full_log["tloss"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Validation Accuracy (classifier)', color=color)
        ax2.plot(full_log["global_epoch"], full_log["val_acc"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        fig.savefig(log_plot_path, format="pdf")

    else:
        if ".h5ad" in args.anndata: 
            adata = sc.read_h5ad(args.anndata)
            # Filter genes expressed in fewer than 3 cells
            sc.pp.filter_genes(adata, min_cells=3)
        else:
            adata = sc.read_10x_mtx(args.anndata)
            anno = pd.read_csv(args.anno, sep="\t")
            anno = anno[["barcode", "label", "time"]]  # ensure required columns exist
            anno = anno.set_index("barcode") 
            adata.obs["label"] = pd.Categorical(anno["label"])
            adata.obs["time"]  = pd.to_numeric(anno["time"], errors="raise") 
            sc.pp.filter_genes(adata, min_cells=3)
            
        genes = [str(g) for g in adata.var.index]   # or adata.var_names
        with open("./genesys_X.pkl", 'rb') as file_handle:
            data = pickle.load(file_handle)

        lineage = pd.read_csv(args.bprint, sep="\t", header=None)

        input_size = data['X_train'].shape[1]
        output_size = lineage.shape[0]
        embedding_dim = 256
        hidden_dim = 256
        n_layers = 2
        clip = 5
        device = args.device
        if args.verbose:
            print(f"[INFO] Using device: {device}")

        test_dataset = Cell_Lineage_Blueprint(data['X_test'], data['y_test'], lineage)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        model = ClassifierLSTM(input_size, output_size, embedding_dim, hidden_dim, n_layers, device).to(device)
        checkpoint_path = os.path.join(args.path, "genesys_training_overall_best.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        sample = next(iter(test_loader))
        x = sample['x'].to(device)

        # x: (Batch, Time, Features) or similar
        T = x.size(1)

        pred_h = model.init_hidden(args.batch_size)

        t_blocks = []
        y_blocks = []
        t_tags = []

        for t in range(T):
            if t == 0:
                t_state = model.generate_current(x, pred_h, 0)
            else:
                t_state = model.generate_next(x, pred_h, t - 1)

            # predict probs at current time t
            y_prob, pred_h = model.predict_proba(x, pred_h, t)

            # class indices -> labels (stay in torch, then map)
            y_idx = y_prob.argmax(dim=1).cpu().tolist()
            y_lab = [lineage[t][i] for i in y_idx]

            # move t_state to CPU numpy for stacking later
            t_np = t_state.detach().to(device).numpy()

            # append
            t_blocks.append(t_np)
            y_blocks.append(y_lab)
            t_tags.extend([f"t{t}"] * t_np.shape[0])

        pred_X = np.vstack(t_blocks)
        pred_Y = np.concatenate(y_blocks).tolist()
        pred_T = t_tags

        obs = pd.DataFrame({
            "label": pred_Y,
            "time": pred_T
        })
        obs = obs.copy()
        if obs.index is None or not len(obs.index) or not isinstance(obs.index.dtype, pd.StringDtype):
             obs.index = pd.Index([f"cell_{i}" for i in range(len(obs))], name="cell_id").astype(str)
        n_features = pred_X.shape[1]
        var = pd.DataFrame(index=pd.Index(genes))

        pred_adata = anndata.AnnData(X=pred_X, obs=obs, var=var)

        sc.pp.scale(pred_adata, max_value=10)
        mask = np.any(np.isnan(pred_adata.X), axis=0)
        pred_adata = pred_adata[:, ~mask].copy()
        
        sc.tl.pca(pred_adata, svd_solver='arpack')
        sc.pp.neighbors(pred_adata, n_neighbors=30, n_pcs=50)
        sc.tl.leiden(pred_adata)
        sc.tl.paga(pred_adata)
        sc.pl.paga(pred_adata)
        sc.tl.umap(pred_adata, init_pos='paga')
        print("[INFO] Generating simulated transcriptomes")
        name_saved = ("./" + args.save_prefix + "_genesys_simulated.h5ad")
        pred_adata.write(name_saved)

if __name__ == "__main__":
    main()

#### Usage
##Train
#genesys --train --raw_counts --anndata ./Root_Atlas_RNA_downsampled_20000_cells.h5ad --bprint ./lineage.txt --epochs 100 --batch_size 512 --verbose

##Generate
#genesys --anndata ./Root_Atlas_RNA_downsampled_20000_cells.h5ad --bprint ./lineage.txt --batch_size 512 --verbose --device "cpu" --save_prefix "Root_Atlas_RNA_downsampled_20000_cells"
