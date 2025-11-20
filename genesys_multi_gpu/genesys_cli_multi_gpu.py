#!/usr/bin/env python3
# genesys_cli_multi_gpu.py — DDP-ready training script that preserves the
# single-GPU "train_until_no_improve" logic:
# - Each new cycle starts from the previous cycle's BEST model.
# - Early-stops when a cycle fails to improve previous best.
# - Rank-0 only I/O (checkpoints, logs, plots).
# - Optional SyncBatchNorm, AMP toggle, deterministic toggle.
# - Uses DistributedSampler under --dist.
#
# NOTE: This script focuses on TRAINING ONLY (no generation/eval here), per user request.

import os, sys, re, copy, argparse, pickle, random
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from genesys_model import *  # expects ClassifierLSTM, Cell_Lineage_Blueprint


# -------------------- Utilities --------------------
def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_deterministic(on: bool):
    if on:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def rank0() -> bool:
    return (dist.get_rank() == 0) if is_dist() else True

def ddp_init(enable: bool):
    if not enable:
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)

def ddp_cleanup():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()

def all_reduce_mean_tensor(t: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


# -------------------- Training core --------------------
def make_optimizer_and_scheduler(model: nn.Module, lr: float, factor: float, patience: int, threshold: float, verbose: bool):
    opt = optim.AdamW(unwrap(model).parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=factor, patience=patience, threshold=threshold, verbose=(rank0() and verbose)
    )
    return opt, sch


def train_one_cycle(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    factor: float,
    patience: int,
    threshold: float,
    clip: float,
    time_bins: int,
    amp_enabled: bool,
    verbose: bool,
):
    scaler = GradScaler(enabled=amp_enabled and device.startswith("cuda"))
    opt, sch = make_optimizer_and_scheduler(model, lr, factor, patience, threshold, verbose)
    criterion = nn.NLLLoss()

    best_loss = float("inf")
    best_val_acc = -1.0
    best_state = copy.deepcopy(unwrap(model).state_dict())
    log = {"tloss": [], "val_acc": []}

    for epoch in range(epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_sum = torch.zeros(1, device=device)
        running_cnt = torch.zeros(1, device=device)

        pbar = tqdm(train_loader, disable=not rank0(), leave=False)
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            h = unwrap(model).init_hidden(x.size(0))
            h = tuple(each.data for each in h)

            opt.zero_grad(set_to_none=True)
            t = np.random.choice(max(1, time_bins - 1))

            with autocast(enabled=amp_enabled and device.startswith("cuda")):
                p, h = unwrap(model).predict(x, h, t)
                unwrap(model).get_belief(x, h)
                tdvae_loss = unwrap(model).calculate_loss(t)
                nll_loss = criterion(p, y)
                loss = tdvae_loss + nll_loss

            # stochastic multi-head backprop option (parity with single-GPU)
            to_backprop = tdvae_loss if (np.random.randint(2) == 0) else nll_loss

            scaler.scale(to_backprop).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(unwrap(model).parameters(), clip)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                bs = x.size(0)
                running_sum += loss.detach() * bs
                running_cnt += torch.tensor([bs], device=device)

                show_sum = all_reduce_mean_tensor(running_sum)
                show_cnt = all_reduce_mean_tensor(running_cnt)
                curr = (show_sum / show_cnt).item()
                if rank0():
                    pbar.set_description(f"[Train] epoch {epoch+1}/{epochs} loss={curr:.4f}")

        # ---- validation ----
        model.eval()
        correct = torch.zeros(1, device=device)
        total = torch.zeros(1, device=device)
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                vh = unwrap(model).init_hidden(x.size(0))
                vh = tuple(each.data for each in vh)
                t = 1
                with autocast(enabled=amp_enabled and device.startswith("cuda")):
                    p, vh = unwrap(model).predict_proba(x, vh, t)
                pred = p.argmax(dim=1)
                correct += (pred == y).sum()
                total += torch.tensor([y.numel()], device=device)

        correct = all_reduce_mean_tensor(correct)
        total = all_reduce_mean_tensor(total)
        val_acc = (correct / total).item()
        tloss = (running_sum / running_cnt).item()
        log["tloss"].append(float(tloss))
        log["val_acc"].append(float(val_acc))

        if rank0() and verbose:
            print(f"[Train] epoch {epoch+1:3d}/{epochs}  loss {tloss:.4f}  val_acc {val_acc:.4f}  lr {opt.param_groups[0]['lr']:.2e}")

        sch.step(tloss)

        improved = (tloss < best_loss) or (np.isclose(tloss, best_loss) and val_acc > best_val_acc)
        if improved:
            best_loss = tloss
            best_val_acc = float(val_acc)
            best_state = copy.deepcopy(unwrap(model).state_dict())

    return best_loss, best_val_acc, log, best_state


def train_until_no_improve(
    model_ctor,
    data,
    lineage,
    device: str,
    save_dir: str,
    save_prefix: str,
    epochs: int,
    max_cycles: int,
    batch_size: int,
    time_bins: int,
    lr: float,
    factor: float,
    patience: int,
    threshold: float,
    clip: float,
    amp_enabled: bool,
    sync_bn: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    verbose: bool,
    resume_from: str = "",
    training_logs: str = ""):
    """
    Single-GPU behavior preserved under DDP:
    - Start cycle 1 from fresh model.
    - Before each new cycle, reload the PREVIOUS cycle's best state.
    - Save per-cycle best checkpoint: {save_prefix}_cycle{cycle}_best.pth
    - Save overall best after stop:   {save_prefix}_overall_best.pth
    - Early stop when no improvement vs previous cycle best.
    - Produce training log plot: genesys_training_logs.pdf
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Prepare datasets/loaders once (data splits already prepared)
    train_ds = Cell_Lineage_Blueprint(data["X_train"], data["y_train"], lineage)
    val_ds   = Cell_Lineage_Blueprint(data["X_val"],   data["y_val"],   lineage)

    if is_dist():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=True)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler   = None
        shuffle_train = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, drop_last=True,
        sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True,
        sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )

    # Build initial model
    def build_model():
        m = model_ctor().to(device)
        if sync_bn:
            m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
        if is_dist():
            local_rank = int(os.environ.get("LOCAL_RANK", 0)) if torch.cuda.is_available() else None
            m = DDP(m, device_ids=[local_rank] if local_rank is not None else None,
                    output_device=local_rank if local_rank is not None else None,
                    find_unused_parameters=True)
        return m

    net = build_model()

    # best state across cycles
    overall_best_state = copy.deepcopy(unwrap(net).state_dict())
    prev_cycle_best_loss = float("inf")
    
    # ---- optional warm-start from a prior cycle-best ----
    start_cycle = 1
    if bool(resume_from):
        if is_dist():
            dist.barrier()
        ckpt = torch.load(resume_from, map_location="cpu")
        # Expect only a 'state_dict' in the checkpoint
        if "state_dict" in ckpt:
            unwrap(net).load_state_dict(ckpt["state_dict"], strict=False)
            overall_best_state = ckpt["state_dict"]
        else:
            raise RuntimeError(f"--resume_from {resume_from} has no 'state_dict' key.")
        if is_dist():
            dist.barrier()

    # Determine which cycle to continue from and the previous best loss using the logs
    if bool(training_logs):
        last_c, last_loss = _infer_last_cycle_and_loss_from_log(training_logs)
        if last_c is not None:
            start_cycle = last_c + 1
        if last_loss is not None:
            prev_cycle_best_loss = float(last_loss)
        if rank0() and verbose:
            print(f"[Resume] training_logs={training_logs} -> last cycle {last_c}, best_loss {last_loss}; continuing at cycle {start_cycle}.")
    else:
        if rank0() and resume_from:
            print("[Resume] --training_logs was not provided; continuing at cycle 2 by default (set --training_logs to be precise).")
    # (Optional) sanity-check vs log file
    if bool(training_logs) and rank0():
        last_in_log = _infer_last_cycle_from_log(training_logs)
        if last_in_log is not None and (last_in_log + 1) != start_cycle:
            print(f"[Note] training_logs suggests next cycle {last_in_log+1}, but checkpoint implies {start_cycle}. Proceeding with checkpoint.")

    # for plotting across cycles
    all_logs = []   # list of DataFrames: tloss, val_acc, epoch, cycle, global_epoch

    for cycle in range(start_cycle, max_cycles + 1):
        if rank0() and verbose:
            print(f"\n===== Training Cycle {cycle}/{max_cycles} =====")

        # Reload previous best BEFORE training this cycle
        unwrap(net).load_state_dict(overall_best_state, strict=False)

        # Train one cycle
        best_loss, best_val_acc, log, best_state = train_one_cycle(
            net, train_loader, val_loader, device, epochs, lr, factor, patience, threshold, clip, time_bins, amp_enabled, verbose
        )

        # Save per-cycle artifacts (rank-0 only)
        if rank0():
            cycle_ckpt = os.path.join(save_dir, f"{save_prefix}_cycle{cycle}_best.pth")
            torch.save({"state_dict": best_state, "best_loss": best_loss, "best_val_acc": best_val_acc, "cycle": cycle}, cycle_ckpt)
            # Save this cycle's logs
            df = pd.DataFrame(log)
            df["epoch"] = np.arange(1, len(df) + 1)
            df["cycle"] = cycle
            all_logs.append(df)
            if verbose:
                print(f"[Train] cycle {cycle} best_loss={best_loss:.6f} val_acc={best_val_acc:.4f} -> saved {cycle_ckpt}")

        # Broadcast best_loss so all ranks agree for early stop decision
        loss_tensor = torch.tensor([best_loss], device=device)
        if is_dist():
            dist.broadcast(loss_tensor, src=0)
            best_loss = float(loss_tensor.item())

        # Early stop when not improved vs previous cycle's best
        if cycle > 1 and not (best_loss < prev_cycle_best_loss):
            if rank0() and verbose:
                print("[EarlyStop] No improvement over previous cycle’s best loss; stopping.")
            break

        # Carry forward best to next cycle
        prev_cycle_best_loss = best_loss
        overall_best_state = copy.deepcopy(best_state)

    # Final save of overall best (rank-0 only)
    if rank0():
        overall_path = os.path.join(save_dir, f"{save_prefix}_overall_best.pth")
        torch.save({"state_dict": overall_best_state, "best_loss": prev_cycle_best_loss}, overall_path)
        if verbose:
            print(f"[Final] Saved overall best -> {overall_path} (loss={prev_cycle_best_loss:.6f})")

        # Plot logs like single-GPU version
        if len(all_logs):
            full_log = pd.concat(all_logs, ignore_index=True)
            # global epoch across cycles
            full_log["global_epoch"] = (full_log["cycle"] - 1) * epochs + full_log["epoch"]

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
            log_plot_path = os.path.join(save_dir, "genesys_training_logs.pdf")
            fig.savefig(log_plot_path, format="pdf")
            if verbose:
                print(f"[Plot] Saved training curves -> {log_plot_path}")


# -------------------- Argparse & main --------------------
def build_argparser():
    p = argparse.ArgumentParser(description="GeneSys multi-GPU (DDP) training with single-GPU cycle logic")
    p.add_argument("--anndata", required=True, help=".h5ad or 10x mtx dir")
    p.add_argument("--bprint", required=True, help="lineage blueprint txt")
    p.add_argument("--anno", default="./annotation.txt", help="barcode/label/time table for 10x mtx")
    p.add_argument("--path", default="./checkpoints", help="output dir for ckpts and logs")
    p.add_argument("--save_prefix", type=str, default="genesys_training")

    # Modes
    p.add_argument("--train", action="store_true", help="Run training (otherwise only prepare data)")
    p.add_argument("--raw_counts", action="store_true", help="If the provided X is raw UMI counts")

    # Hyperparams
    p.add_argument("--epochs", type=int, default=100, help="Epochs per cycle")
    p.add_argument("--max_cycles", type=int, default=20, help="Max training cycles")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--factor", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=5.0)

    # DDP / loader
    p.add_argument("--dist", action="store_true", help="Enable DistributedDataParallel")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--sync_bn", action="store_true", help="Use synchronized BatchNorm for multi-GPU")

    # Repro / AMP
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Force deterministic algorithms")
    p.add_argument("--amp_off", action="store_true", help="Disable mixed precision (AMP)")

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume_from", type=str, default="", help="Path to a cycle-best or progress checkpoint to resume from.")
    p.add_argument("--training_logs", type=str, default="", help="Optional path to previous job log for sanity-checking resume cycle.")

    return p



def _infer_last_cycle_from_log(path: str):
    try:
        import re
        last = None
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = re.search(r"(?:^|\s)Cycle\s+(\d+)", line)
                if m:
                    last = int(m.group(1))
        return last
    except Exception:
        return None



def _infer_last_cycle_and_loss_from_log(path: str):
    """
    Parse the last occurrence of a line like:
    "[Train] cycle 9 best_loss=2217.422607 val_acc=1.0000 -> saved ..."
    Returns (last_cycle:int, last_best_loss:float) or (None, None).
    """
    try:
        import re
        last_c = None
        last_loss = None
        pat = re.compile(r"\[Train\]\s*cycle\s*(\d+)\s*best_loss=([0-9]+\.?[0-9]*)", re.I)
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    last_c = int(m.group(1))
                    try:
                        last_loss = float(m.group(2))
                    except Exception:
                        last_loss = None
        return last_c, last_loss
    except Exception:
        return None, None


def main():
    args = build_argparser().parse_args()

    # DDP setup
    ddp_init(args.dist)
    device = args.device
    if device == "cuda" and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"

    setup_seed(args.seed)
    set_deterministic(args.deterministic)
    amp_enabled = (not args.amp_off)

    if rank0() and args.verbose:
        ws = get_world_size()
        print(f"[INFO] device={device} dist={args.dist} world={ws} seed={args.seed} amp={amp_enabled}")

    # Load data
    X_path = os.path.join(args.path, "genesys_X.pkl")
    with open(X_path, 'rb') as file_handle:
            data = pickle.load(file_handle)

    # Blueprint
    lineage = pd.read_csv(args.bprint, sep="\t", header=None)
    input_size = data["X_train"].shape[1]
    output_size = lineage.shape[0]
    embedding_dim = 256
    hidden_dim = 256
    n_layers = 2

    def model_ctor():
        return ClassifierLSTM(input_size, output_size, embedding_dim, hidden_dim, n_layers, device)

    if args.train:
        train_until_no_improve(
            model_ctor=model_ctor,
            data=data,
            lineage=lineage,
            device=device,
            save_dir=args.path,
            save_prefix=args.save_prefix,
            epochs=args.epochs,
            max_cycles=args.max_cycles,
            batch_size=args.batch_size,
            time_bins=lineage.shape[1],
            lr=args.lr,
            factor=args.factor,
            patience=args.patience,
            threshold=args.threshold,
            clip=args.clip,
            amp_enabled=amp_enabled,
            sync_bn=args.sync_bn,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            verbose=args.verbose,
            resume_from=args.resume_from,
            training_logs=args.training_logs)

    if args.dist:
        ddp_cleanup()

if __name__ == "__main__":
    main()
