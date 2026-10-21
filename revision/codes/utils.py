"""
Shared utilities for GeneSys Sprint 2 experiments.
Handles data loading, label mapping, and model instantiation
using the GitHub version of the code.
"""
import os, sys, pickle, copy
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the GitHub repo's genesys package to path
REPO_DIR = os.path.join(os.path.dirname(__file__), "GeneSys")
sys.path.insert(0, os.path.join(REPO_DIR, "genesys"))

from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint, GaussianNoise

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
ROOT_DATA = os.path.join(DATA_DIR, "genesys_root_data.pkl")
ROOT_CKPT = os.path.join(DATA_DIR, "workstation", "genesys_model_trained_on_root_atlas_20240308_continue4.pth")
MOUSE_DATA = os.path.join(DATA_DIR,
    "genesys_mouse_embryo_11_clusters_500_cells_per_state_2500_GPC_UPC_data.pkl")
MOUSE_CKPT = os.path.join(DATA_DIR,
    "genesys_model_trained_on_mouse_embryo_500_cells_per_state_2500_GPC_UPC_20240303.pth")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Cell-type mapping (Root Atlas) ─────────────────────────────────────────
NUM2NAME = {
    -1: "Quiescent Center",
    0: "Columella",
    1: "Lateral Root Cap",
    2: "Phloem",
    3: "Xylem",
    4: "Procambium",
    5: "Pericycle",
    6: "Endodermis",
    7: "Cortex",
    8: "Atrichoblast",
    9: "Trichoblast",
}
NAME2NUM = {v: k for k, v in NUM2NAME.items()}

# ── Data loading ───────────────────────────────────────────────────────────
def load_root_data(path=ROOT_DATA):
    """Load root atlas pickle and remap integer labels to full string names.

    Old format: labels like "0_1.0" (celltype_int _ timebin_float)
    New format: labels like "Columella_1.0" (celltype_name _ timebin_float)
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    for key in ['y_train', 'y_val', 'y_test']:
        if key not in data:
            continue
        new_labels = []
        for lab in data[key]:
            ct_str, tb_str = lab.split('_', 1)
            ct_int = int(ct_str)
            ct_name = NUM2NAME[ct_int]
            new_labels.append(f"{ct_name}_{tb_str}")
        data[key] = np.array(new_labels)

    return data


def load_root_data_blueprint(path=ROOT_DATA, reverse_time=False):
    """Root atlas pickle with labels in the convention of the up-to-date GeneSys code
    (Cell_Lineage_Blueprint, genesys_cli): time starts at 1, so the stem cells (time bin 0 of
    the pickle) are time 1 and time bins 1 to 10 are times 2 to 11, as in toy_data/annotations.txt.
    Column k of the lineage blueprint then reads the cells of time k+1, i.e. time bin k.

    reverse_time=True inverts the temporal order of the labels (time -> 12 - time), for the
    reversed-blueprint decoy; it is used together with a column-reversed lineage table."""
    data = load_root_data(path)
    for key in ['y_train', 'y_val', 'y_test']:
        if key not in data:
            continue
        out = []
        for lab in data[key]:
            ct, tb = lab.split('_', 1)
            t = float(tb) + 1.0
            if reverse_time:
                t = 12.0 - t
            out.append(f"{ct}_{t}")
        data[key] = np.array(out)
    return data


def reverse_time_labels(labels):
    """Invert the temporal order of blueprint-convention labels (time -> 12 - time)."""
    out = []
    for lab in labels:
        ct, tb = lab.split('_', 1)
        out.append(f"{ct}_{12.0 - float(tb)}")
    return np.array(out)


def load_lineage(path=None):
    """Load the root atlas lineage blueprint TSV.
    Returns a pandas DataFrame (rows=trajectories, cols=time bins).
    """
    if path is None:
        path = os.path.join(REPO_DIR, "toy_data", "lineage.txt")
    return pd.read_csv(path, sep="\t", header=None)


# ── Model helpers ──────────────────────────────────────────────────────────
def make_model(input_size, output_size=10, embedding_dim=256, hidden_dim=256,
               n_layers=2, device="cuda"):
    """Instantiate ClassifierLSTM using the GitHub version."""
    model = ClassifierLSTM(
        input_size=input_size,
        output_size=output_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        device=device,
    )
    return model.to(device)


def load_checkpoint(model, ckpt_path, device="cuda"):
    """Load a checkpoint, handling both old (raw state_dict) and new (dict) formats."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # Old format: checkpoint IS the state dict
        model.load_state_dict(ckpt)
    return model


# ── Evaluation helpers ─────────────────────────────────────────────────────
def evaluate_recreation_accuracy(model, loader, device, time_bins=11):
    """Run classifier on each time bin and compute per-bin accuracy.
    Returns dict: {time_bin: accuracy}
    """
    model.eval()
    results = {}
    batch_size = loader.batch_size

    with torch.no_grad():
        all_preds = {t: [] for t in range(time_bins)}
        all_true = []

        for sample in loader:
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            h = model.init_hidden(x.size(0))

            for t in range(time_bins):
                p, h = model.predict_proba(x, h, t)
                all_preds[t].append(p.argmax(dim=1).cpu().numpy())

            all_true.append(y.cpu().numpy())

        all_true = np.concatenate(all_true)
        for t in range(time_bins):
            preds = np.concatenate(all_preds[t])
            acc = (preds == all_true).mean()
            results[t] = float(acc)

    return results


def evaluate_cell_recovery(model, data, device, gene_fractions=None,
                           n_repeats=10, lineage=None):
    """Evaluate cell-type recovery under gene masking.

    For each fraction, randomly select that fraction of genes to KEEP,
    mask the rest to 0, and measure classification accuracy.

    Returns DataFrame with columns: fraction, repeat, accuracy, per_class_accuracy
    """
    if gene_fractions is None:
        gene_fractions = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0]

    model.eval()
    X_test = data['X_test'] if 'X_test' in data else data['X_val']
    y_test = data['y_test'] if 'y_test' in data else data['y_val']

    # Parse labels to get cell-type index (trajectory index for prediction)
    y_int = []
    for lab in y_test:
        ct_name = lab.split('_', 1)[0]
        y_int.append(NAME2NUM[ct_name])
    y_int = np.array(y_int)

    n_genes = X_test.shape[1]

    # Build input sequences using lineage blueprint
    if lineage is None:
        lineage = load_lineage()
    n_trajectories = lineage.shape[0]
    n_timebins = lineage.shape[1]

    results = []

    for frac in gene_fractions:
        for rep in range(n_repeats):
            if frac == 0.0:
                mask = np.zeros(n_genes, dtype=bool)
            elif frac >= 1.0:
                mask = np.ones(n_genes, dtype=bool)
            else:
                n_keep = max(1, int(frac * n_genes))
                idx = np.random.choice(n_genes, n_keep, replace=False)
                mask = np.zeros(n_genes, dtype=bool)
                mask[idx] = True

            # Apply mask to test data
            X_masked = X_test.copy()
            X_masked[:, ~mask] = 0.0

            # Create a temporary dataset with masked data
            # We evaluate per trajectory: for each cell, build a sequence
            # following its trajectory in the lineage blueprint
            correct = 0
            total = 0

            # Group cells by cell type
            for traj_idx in range(n_trajectories):
                ct_name = lineage.iloc[traj_idx, -1]  # terminal cell type (last time bin)
                ct_int = NAME2NUM.get(ct_name, -1)
                if ct_int == -1:
                    continue

                cell_mask = (y_int == ct_int)
                if cell_mask.sum() == 0:
                    continue

                # Sample cells for this trajectory and build input sequence
                n_cells = min(cell_mask.sum(), 50)  # cap for speed
                cell_indices = np.where(cell_mask)[0][:n_cells]

                for ci in cell_indices:
                    # Build a single-sample sequence
                    seq = np.zeros((1, n_timebins, n_genes), dtype=np.float32)
                    for t in range(n_timebins):
                        seq[0, t, :] = X_masked[ci, :]  # same cell at all time points (simplified)

                    x_t = torch.from_numpy(seq).to(device)
                    h = model.init_hidden(1)
                    # Predict at middle time bin
                    p, h = model.predict_proba(x_t, h, n_timebins // 2)
                    pred = p.argmax(dim=1).item()

                    if pred == traj_idx:
                        correct += 1
                    total += 1

            acc = correct / total if total > 0 else 0.0
            results.append({
                'fraction': frac,
                'repeat': rep,
                'accuracy': acc,
                'n_genes_kept': int(frac * n_genes),
            })

            if frac == 0.0:
                # Only need one repeat for all-zeros
                break

    return pd.DataFrame(results)


# ── Plotting helpers ───────────────────────────────────────────────────────
def save_figure(fig, name, dpi=150):
    """Save figure to results directory as both PDF and PNG."""
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f"{name}.{ext}"),
                    dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}.pdf / .png")
