#!/usr/bin/env python3
"""
Training for every revision experiment through the up-to-date GeneSys code:
`run_cycle` of GeneSys/genesys/genesys_cli.py, called unchanged, one cycle of 30 epochs,
with its default optimizer and scheduler settings (AdamW 1e-3, ReduceLROnPlateau factor 0.5,
patience 10, threshold 0.05, gradient clip 5).

Everything an experiment varies enters through the inputs of run_cycle:
  - the labels (time convention of the package, time from 1; time-reversed for the reversed decoy)
  - the lineage blueprint table (correct, reversed, shuffled, random)
  - the model object (noise sigma, width, MLP in place of the LSTM, TD-VAE loss switched off)
KL logging does not touch the loop either: the training terms are recorded by wrapping
model.calculate_loss, the validation terms by a wrapper around the validation loader.

Compatibility shim: PyTorch >= 2.7 removed the `verbose` argument of ReduceLROnPlateau that
genesys_cli passes; the shim drops that argument and changes nothing else.
"""
import os, sys, copy, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GeneSys", "genesys"))
import genesys_cli
from genesys_model import ClassifierLSTM, Cell_Lineage_Blueprint, GaussianNoise

_RLROP = torch.optim.lr_scheduler.ReduceLROnPlateau


class _ReduceLROnPlateauCompat(_RLROP):
    def __init__(self, optimizer, *args, verbose=None, **kwargs):
        super().__init__(optimizer, *args, **kwargs)


import inspect
if 'verbose' not in inspect.signature(_RLROP.__init__).parameters:
    genesys_cli.optim.lr_scheduler.ReduceLROnPlateau = _ReduceLROnPlateauCompat

EPOCHS = 30
BATCH_SIZE = 512


def make_loaders(X_train, y_train, X_val, y_val, lineage, batch_size=BATCH_SIZE, workers=10):
    """Blueprint loaders as in genesys_cli.main (shuffle, drop_last); worker processes only
    speed up batch assembly."""
    train_ds = Cell_Lineage_Blueprint(X_train, y_train, lineage)
    val_ds = Cell_Lineage_Blueprint(X_val, y_val, lineage)
    kw = dict(batch_size=batch_size, shuffle=True, drop_last=True)
    if workers:
        kw.update(num_workers=workers, persistent_workers=True)
    return DataLoader(train_ds, **kw), DataLoader(val_ds, **kw)


def make_lstm(input_size, output_size, device, dim=256, sigma=0.2):
    model = ClassifierLSTM(input_size, output_size, dim, dim, 2, device)
    if sigma != 0.2:
        for m in model.modules():
            if isinstance(m, GaussianNoise):
                m.sigma = sigma
    return model.to(device)


def switch_off_tdvae(model):
    """No TD-VAE ablation. calculate_loss returns a constant that is not connected to any
    parameter, so the steps that run_cycle assigns to the TD-VAE loss leave the model
    unchanged and only the classifier loss trains it."""
    def constant_loss(t):
        return torch.zeros((), device=next(model.parameters()).device, requires_grad=True)
    model.calculate_loss = constant_loss
    return model


class KLLog:
    """Per-epoch TD-VAE loss components without touching run_cycle."""
    def __init__(self, model, val_loader, time_bins, decompose, n_val_batches=4):
        self.model, self.val_loader, self.time_bins = model, val_loader, time_bins
        self.decompose, self.n_val = decompose, n_val_batches
        self.rows, self._train = [], []
        orig = model.calculate_loss

        def logged(t):
            loss = orig(t)
            if model.training:
                with torch.no_grad():
                    _, k1, k2, rec = decompose(model, t)
                self._train.append((float(loss.item()), k1, k2, rec))
            return loss
        model.calculate_loss = logged

    def __len__(self):
        return len(self.val_loader)

    def __iter__(self):
        # run_cycle iterates the validation loader once per epoch, in eval mode under no_grad
        model = self.model
        v = []
        for i, sample in enumerate(self.val_loader):
            if i >= self.n_val:
                break
            x = sample['x'].to(next(model.parameters()).device)
            t = np.random.choice(max(1, self.time_bins - 1))
            model.get_belief(x, model.init_hidden(x.size(0)))
            _, k1, k2, rec = self.decompose(model, t)
            v.append((k1, k2, rec))
        tr = np.array(self._train) if self._train else np.zeros((1, 4))
        va = np.array(v)
        self.rows.append({'epoch': len(self.rows),
                          'train_tdvae': tr[:, 0].mean(), 'train_kl_z1': tr[:, 1].mean(),
                          'train_kl_z2': tr[:, 2].mean(), 'train_recon': tr[:, 3].mean(),
                          'val_kl_z1': va[:, 0].mean(), 'val_kl_z2': va[:, 1].mean(),
                          'val_recon': va[:, 2].mean(),
                          'kl_z1': va[:, 0].mean(), 'kl_z2': va[:, 1].mean(),
                          'recon': va[:, 2].mean()})
        self._train = []
        yield from self.val_loader


def train(model, train_loader, val_loader, device, save_dir, prefix,
          epochs=EPOCHS, batch_size=BATCH_SIZE, time_bins=11):
    """One cycle of run_cycle; returns (best_state, best_loss, best_val_acc, logger)."""
    os.makedirs(save_dir, exist_ok=True)
    best_state, best_loss, best_val_acc, logger, _ = genesys_cli.run_cycle(
        model, train_loader, val_loader, device, epochs=epochs, verbose=True,
        save_prefix=prefix, path=save_dir, batch_size=batch_size, time_bins=time_bins)
    torch.save({"state_dict": best_state, "best_loss": best_loss, "best_val_acc": best_val_acc,
                "epochs": epochs, "trainer": "genesys_cli.run_cycle"},
               os.path.join(save_dir, f"{prefix}_best.pth"))
    with open(os.path.join(save_dir, f"{prefix}_logs.pkl"), 'wb') as f:
        pickle.dump(logger, f)
    return best_state, best_loss, best_val_acc, logger
