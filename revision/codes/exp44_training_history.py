#!/usr/bin/env python3
"""
Experiment 44: the training histories of Figure S2, rebuilt on their true epoch index.

Figure S2 was drawn in notebooks 2 (root) and 6 (mouse embryo) from the logger saved after the
last training run. Notebook 6 removed epochs 200-499 of the mouse log and renumbered the rest
(cells 126 and 127: log[0:200] + log[500:1300], index set to 0-999), and both notebooks dropped
epoch 0 before plotting (root cell 123, mouse cell 129). The saved log files are not on Resnick.

Every training cell of those notebooks printed, for each epoch, the mean training loss in the
progress bar ("Epoch k Train: <loss>") and the validation accuracy ("Validation Accuracy <acc>"),
and, when the checkpoint of that run was saved, the new best loss on the next line. This script
reads those printed records in the order of the training cells, numbers the epochs from the start
of training, and marks the run each epoch belongs to and the epochs at which a checkpoint was
saved. A run whose printed output is incomplete keeps its missing epochs as gaps.

Outputs: results/exp44_training_history_{root,mouse}.csv (one row per epoch: epoch, run, run_epoch,
train_loss, val_acc, checkpoint_saved, checkpoint) and results/exp44_training_history.json
(runs, selected checkpoint, epochs trained, missing epochs, epochs whose training loss overflowed
to inf).
"""
import glob
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, 'GeneSys', 'jupyter_notebook')
RES = os.path.join(HERE, 'results')
# notebook, the checkpoint the analyses load (root: memory reference_root_checkpoint; mouse: notebook 7)
MODELS = {'root': ('2-Genesys_model_training_Root-20240308.ipynb', 'continue4'),
          'mouse': ('6-Genesys_model_training_Mouse_Embryo-20240302.ipynb', 'continue8')}
RUN_LEN = re.compile(r'range\((\d+)\)')


def runs_of(path):
    nb = json.load(open(path))
    out = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        texts = [''.join(o.get('text', [])) for o in c.get('outputs', []) if 'text' in o]
        if not any('Validation Accuracy' in t for t in texts):
            continue
        ckpt = re.findall(r'_(continue\d*)\.pth', src)
        n_epochs = re.findall(r'n_epochs\s*=\s*(\d+)', ''.join(''.join(x['source']) for x in nb['cells'][:i]))
        loss, acc, saved = {}, {}, set()
        last = None
        for t in texts:
            # the progress bar may be cut to 'h <k> Train' by a carriage return; an overflowed
            # epoch prints its loss as inf
            for k, v in re.findall(r'(\d+) Train: (inf|[\d.]+)', t):
                loss[int(k)] = float(v)
                last = int(k)
            for line in t.split('\n'):
                m = re.match(r'Validation Accuracy ([\d.]+)', line)
                if m and last is not None:
                    acc[last] = float(m.group(1))
                elif re.fullmatch(r'\d+\.\d+', line.strip()) and last is not None:
                    saved.add(last)
        out.append({'cell': i, 'checkpoint': ckpt[0] if ckpt else 'first', 'loss': loss, 'acc': acc,
                    'saved': saved, 'n_epochs': int(n_epochs[-1]) if n_epochs else None,
                    'execution_count': c.get('execution_count')})
    return out


def history(name):
    nb_file, selected = MODELS[name]
    runs = runs_of(os.path.join(NB, nb_file))
    rows, info, start = [], [], 0
    for r, run in enumerate(runs):
        # a run lasts n_epochs; when the printed output stops early, the epochs it does not show
        # are kept as gaps (the saved logger had them, see the module docstring)
        length = run['n_epochs'] or (max(run['loss']) + 1)
        length = max(length, max(run['loss']) + 1)
        missing = [k for k in range(length) if k not in run['acc']]
        overflow = [k for k, v in run['loss'].items() if v == float('inf')]
        for k in range(length):
            rows.append({'epoch': start + k, 'run': r + 1, 'run_epoch': k, 'checkpoint': run['checkpoint'],
                         'train_loss': run['loss'].get(k), 'val_acc': run['acc'].get(k),
                         'checkpoint_saved': k in run['saved']})
        info.append({'run': r + 1, 'cell': run['cell'], 'checkpoint': run['checkpoint'],
                     'first_epoch': start, 'last_epoch': start + length - 1, 'epochs': length,
                     'epochs_printed': len(run['acc']), 'missing_run_epochs': missing,
                     'loss_overflow_run_epochs': overflow,
                     'execution_count': run['execution_count']})
        start += length
    df = pd.DataFrame(rows)
    sel = df[(df['checkpoint'] == selected) & df['checkpoint_saved']]
    summary = {'notebook': nb_file, 'runs': info, 'epochs_trained': int(start),
               'selected_checkpoint': selected,
               'selected_checkpoint_epoch': int(sel['epoch'].max()) if len(sel) else None,
               'missing_epochs': [int(e) for e in df.loc[df['val_acc'].isna(), 'epoch']],
               'loss_overflow_epochs': [int(e) for e in df.loc[df['train_loss'] == float('inf'), 'epoch']]}
    return df, summary


if __name__ == '__main__':
    summ = {}
    for name in MODELS:
        df, s = history(name)
        df.to_csv(os.path.join(RES, f'exp44_training_history_{name}.csv'), index=False)
        summ[name] = s
        print(name, 'epochs', s['epochs_trained'], 'selected', s['selected_checkpoint'], 'at epoch',
              s['selected_checkpoint_epoch'], 'missing', s['missing_epochs'], 'overflow', s['loss_overflow_epochs'])
        for r in s['runs']:
            print('  ', r)
    with open(os.path.join(RES, 'exp44_training_history.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
