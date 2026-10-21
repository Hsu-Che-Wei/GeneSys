#!/usr/bin/env python3
"""Regenerate the network analysis of one notebook 12 with out-degree and in-degree named by
their regulatory meaning.

The notebook computes the LIMA of a transition, saves it (np.save), reloads it (np.load) and
then builds the TF network and the centrality table. W[i, j] of a LIMA is the effect of gene i
at the earlier state on gene j at the later state. network() stores the edge of entry adj[i, j]
as (j, i), from the target back to its regulator, and read the out-degree of that graph as the
out-degree of the TF. The targets of a TF are the incoming edges of that graph, so the two
assignments are exchanged: out-degree is now the number of targets a TF regulates and in-degree
the number of regulators acting on it.

The graph itself is left as it is. Its edge weights are the signed LIMA coefficients, and
networkx computes weighted betweenness with Dijkstra's algorithm, which is not invariant to
reversing the edges when weights are negative. Drawing the edges the other way would therefore
change the betweenness values (and with them the combined centrality score), which is a change
of results and not of names. With the graph unchanged every other column of the table is
reproduced exactly.

Only the cells from the np.load of the saved LIMA to the end of the notebook are executed
again, with the gene list and the cell type list they need. The LIMA itself is not recomputed.
The outputs of the executed cells replace the stored ones, the cells before np.load keep
theirs, and the notebook writes its centrality table as before.

Usage (one CPU job per notebook, about 90 GB of memory):
  sbatch experiments/slurm/run_cpu.sh experiments/regenerate_notebook12.py <notebook.ipynb>
"""
import os, re, sys, json, copy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '.nbtools'))
import nbformat
from nbclient import NotebookClient

DATA_DIR = '/resnick/groups/NolanLab/cwhsu/CW_data/genesys'
OLD = ("    # Calculate outgoing centrality\n"
       "    out_centrality = nx.out_degree_centrality(G)\n"
       "    # Calculate incoming centrality\n"
       "    in_centrality = nx.in_degree_centrality(G)\n")
NEW = ("    # The edge of entry adj[i, j] (effect of gene i on gene j) is stored above as (j, i), from the\n"
       "    # target back to its regulator. The targets of a TF are therefore its incoming edges in G and\n"
       "    # its regulators its outgoing edges.\n"
       "    # Calculate outgoing centrality (number of targets the TF regulates)\n"
       "    out_centrality = nx.in_degree_centrality(G)\n"
       "    # Calculate incoming centrality (number of regulators acting on the TF)\n"
       "    in_centrality = nx.out_degree_centrality(G)\n")
SETUP = f'''import os
os.chdir({DATA_DIR!r})
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# The notebook was written for an older matplotlib, where plt.colorbar(mappable) took its space
# from the current axes. Newer versions ask for the axes, so the old default is restored here.
_colorbar = plt.colorbar
def _colorbar_on_current_axes(mappable=None, cax=None, ax=None, **kw):
    if cax is None and ax is None:
        ax = plt.gca()
    return _colorbar(mappable, cax=cax, ax=ax, **kw)
plt.colorbar = _colorbar_on_current_axes
'''


def main(path):
    nb = nbformat.read(path, as_version=4)
    src = [c.source for c in nb.cells]
    load = next(i for i, s in enumerate(src) if re.search(r"^ctw = np\.load\(", s, flags=re.M))
    net = next(i for i, s in enumerate(src) if 'def network(' in s)
    genes = next(i for i, s in enumerate(src) if "gene_list = pd.read_csv('./gene_list_1108.csv')" in s)
    cts = next(re.search(r"^cts = \[.*\]$", s, flags=re.M).group(0) for s in src
               if re.search(r"^cts = \[.*\]$", s, flags=re.M))
    assert nb.cells[net].source.count(OLD) == 1 or nb.cells[net].source.count(NEW) == 1
    nb.cells[net].source = nb.cells[net].source.replace(OLD, NEW)

    run = nbformat.v4.new_notebook()
    run.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    run.cells.append(nbformat.v4.new_code_cell(SETUP))
    run.cells.append(nbformat.v4.new_code_cell(nb.cells[genes].source))
    run.cells.append(nbformat.v4.new_code_cell(cts))
    order = [i for i in range(load, len(nb.cells))
             if nb.cells[i].cell_type == 'code' and nb.cells[i].source.strip()]
    for i in order:
        run.cells.append(nbformat.v4.new_code_cell(nb.cells[i].source))
    os.environ['PYTHONPATH'] = os.path.join(HERE, '.nbtools') + os.pathsep + os.environ.get('PYTHONPATH', '')
    NotebookClient(run, timeout=None, kernel_name='python3').execute()

    for k, i in enumerate(order):
        done = run.cells[3 + k]
        nb.cells[i].outputs = copy.deepcopy(done.outputs)
        nb.cells[i].execution_count = done.execution_count
    nbformat.write(nb, path)
    print('rewrote', path, '| cells executed:', len(order))


if __name__ == '__main__':
    main(sys.argv[1])
