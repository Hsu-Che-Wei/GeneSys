#!/usr/bin/env python3
"""
Experiment 26: per-gene expression statistics of the root test split.

Small CSV used to audit whether the in-silico knockout ranking (exp17) is
confounded by expression level: mean expression, detection rate and variance
of every gene across the test cells.

Usage:
  sbatch --job-name=exp26_stats --qos=debug --time=0:20:00 \
         experiments/slurm/run_cpu.sh experiments/exp26_gene_expression_stats.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_root_data, RESULTS_DIR

data = load_root_data()
X = data['X_test']
gene_ids = pd.read_csv('/resnick/groups/NolanLab/cwhsu/CW_data/genesys/gene_list_1108.csv')['features'].values
assert len(gene_ids) == X.shape[1]
df = pd.DataFrame({'GeneID': gene_ids,
                   'mean_expr': X.mean(axis=0),
                   'detection_rate': (X > 0).mean(axis=0),
                   'var_expr': X.var(axis=0)})
out = os.path.join(RESULTS_DIR, 'exp26_gene_expression_stats.csv')
df.to_csv(out, index=False)
print(f'saved {out}: {df.shape}')
