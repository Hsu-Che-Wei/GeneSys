#!/usr/bin/env python3
"""
Experiment 24: manuscript figures for the round-3 revision.

Builds every figure that the round-3 manuscript adds or replaces, from the
existing result files. Nothing is recomputed here.

  fig5_benchmark_conditions Figure 5. TF-prioritization benchmark across the
                            four data scenarios (GeneSys, scRNA-seq, pseudobulk,
                            scVI), drawn in the style of Figure 4A-B.
  figS5EF_extrapolation     Figure S5E-F. Recreation from zero-padded input (exp7);
                            joined to the Illustrator panels A-D of Figure S5 by
                            exp41_supplementary_composites.py.
  figS_blueprint_controls   Figure S6. Decoy blueprints under recreation and
                            generation (exp1), external classifiers trained on
                            real cells only (exp12) and architectural ablations (exp6).
  figS_random_baseline      Figure S7. Random-initialization baseline under the
                            Figure 3 protocol (exp25), default class across seeds,
                            and belief-state geometry of the trained model (exp11).
  figS_genomewide           Figure S11. Genome-wide gold-standard TF recall for
                            every ranking method and data scenario (exp18).
  figS_model_sensitivity    Figure S14. Noise level (exp4), model width (exp10)
                            and TD-VAE loss components during training (exp3).
  figS_lima_sensitivity     Figure S15. LIMA sensitivity to the solver, row count
                            and generated-cell spread (exp16) and to the z-score
                            threshold (exp5, exp9).

Style follows Cell Press figure guidance and the original GeneSys figures:
Arial-compatible sans-serif (Liberation Sans), uppercase bold panel letters
outside the axes, no in-panel titles, half-open frames, no gridlines, a fixed
colour per data condition, and every label placed so that nothing overlaps.

Usage:
  python experiments/exp24_manuscript_figures.py

GENESYS_RESULTS sets the folder of result tables (default: results/ next to
this script) and GENESYS_FIGURES the folder the figures are written to
(default: the results folder). The revision notebooks set both.
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import font_manager

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('GENESYS_RESULTS', os.path.join(BASE, 'results'))
FIG = os.environ.get('GENESYS_FIGURES', RES)

# ── Style ────────────────────────────────────────────────────────────────
_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ('Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans')
             if f in _fonts), 'DejaVu Sans')
# Cell Press: two-column figures are 174 mm wide, type 6 to 8 pt at final size,
# colour/halftone art at 300 dpi or more, line art higher. Figures are drawn at
# final size, exported as vector PDF and as 600 dpi PNG and LZW TIFF.
W2COL = 174 / 25.4      # 6.85 in
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [FONT],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.grid': False,
    'lines.linewidth': 1.2,
    'figure.dpi': 200,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Data conditions: label, line/marker colour, pale row shading
COND_ORDER = ['genesys', 'scrnaseq', 'pseudobulk', 'scvi']
COND_LABEL = {'genesys': 'GeneSys', 'scrnaseq': 'scRNA-seq',
              'pseudobulk': 'Pseudobulk', 'scvi': 'scVI'}
COND_COLOR = {'genesys': '#0072B2', 'scrnaseq': '#D55E00',
              'pseudobulk': '#009E73', 'scvi': '#CC79A7'}
COND_SHADE = {'genesys': '#FBEADF', 'scrnaseq': '#E3F1FB',
              'pseudobulk': '#E6F4E9', 'scvi': '#EFE8F4', 'perm': '#E6E6E6'}
GREY = '#6E6E6E'          # every null, chance or reference line
MODEL = '#0072B2'         # the model / correct / trained condition
ALT = '#D55E00'           # the contrasting condition (reversed, untrained, spread)
NULLBAR = '#BDBDBD'       # baseline bars (recreation accuracy, non-highlighted rows)
PERMUTATION_R50 = 741

STRATEGY_LABELS = {
    'degree_centrality': 'degree centrality',
    'out_centrality': 'out centrality',
    'in_centrality': 'in centrality',
    'betweenness_centrality': 'betweenness centrality',
    'eigenvector_centrality': 'eigenvector centrality',
    'myAUC_rank': 'mean AUC rank',
    'avg_diff_rank': 'mean avg-diff rank',
    'importance_sum': 'importance sum',
    'gen_effect': 'generative effect',
    'cls_effect': 'classifier effect',
    'combined_effect': 'combined effect',
}
METHOD_ORDER = ['LIMA', 'DE']              # Figure 5: the schemes of Figure 4, on all four conditions
# Figure S11A: the linear criterion against the two nonlinear ones, on GeneSys-generated data
GENERATED_ORDER = ['LIMA', 'GRNBoost2', 'Knockout']
BENCH_TABLE = 'exp20_fig4_expanded_table.csv'   # exp13 R50 scoring of every scheme and condition
GW_TABLE = 'exp18_genomewide_recall.csv'
# Panels A and B are all one data condition, so they avoid the condition hues.


def _table(name):
    """Benchmark tables sit in results/ here and in results/prioritization/ in the repository."""
    path = os.path.join(RES, name)
    return path if os.path.exists(path) else os.path.join(RES, 'prioritization', name)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), bbox_inches='tight',
                    pad_inches=0.04)
    plt.close(fig)
    from PIL import Image
    with Image.open(os.path.join(FIG, f'{name}.png')) as im:
        im.save(os.path.join(FIG, f'{name}.tif'), compression='tiff_lzw', dpi=(600, 600))
        w, h = im.size
    print(f'  saved {name}.pdf / .png / .tif  ({w / 600:.2f} x {h / 600:.2f} in at 600 dpi)')


def letters(fig, axes, labels, dx=-0.012, dy=0.012, align_top=True):
    """Bold uppercase panel letters just outside each axes' top-left corner,
    measured in figure coordinates after layout (including tick labels and
    legends), all placed at the same height so they never sit on data or on a
    neighbouring panel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbs = [ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for ax in axes]
    top = max(bb.y1 for bb in bbs) if align_top else None
    for bb, lab in zip(bbs, labels):
        y = (top if align_top else bb.y1) + dy
        fig.text(bb.x0 + dx, y, lab, fontsize=10, fontweight='bold', ha='left', va='bottom')


def shade_rows(ax, ys, keys, half=0.5):
    for y, k in zip(ys, keys):
        ax.axhspan(y - half, y + half, color=COND_SHADE[k], lw=0, zorder=0)


# ══════════════════════════════════════════════════════════════════════════
# Figure 4E-F
# ══════════════════════════════════════════════════════════════════════════
def figure4_expanded():
    df = pd.read_csv(_table(BENCH_TABLE))
    # This figure compares data conditions under the schemes of Figure 4 (LIMA centrality
    # and DE). GRNBoost2 and the in silico knockout are computed on generated data alone and
    # belong to Figure S11.
    df = df[df['method'].isin(METHOD_ORDER)].copy()
    missing = df[df['R50'].isna()][['method', 'condition']].drop_duplicates()
    df = df.dropna(subset=['R50'])

    fig = plt.figure(figsize=(W2COL, 6.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=1.15)
    axE = fig.add_subplot(gs[0])
    axF = fig.add_subplot(gs[1])

    # ── E: one row per method family and data condition ──
    # Within each method family the conditions are ordered by their mean R50, best first,
    # so the figure reads in the order the text reports; a condition that never reaches
    # R50 closes its family.
    rows = []
    for m in METHOD_ORDER:
        fam = df[df['method'] == m]
        order = fam.groupby('condition')['R50'].mean().sort_values(kind='stable').index
        for c in order:
            rows.append((f'{m} ({COND_LABEL[c]})', c, fam[fam['condition'] == c]['R50'].values))
        for _, mr in missing[missing['method'] == m].iterrows():
            rows.append((f"{m} ({COND_LABEL[mr['condition']]})", mr['condition'], np.array([])))
    rows.append(('Permutation', 'perm', np.array([PERMUTATION_R50])))
    ys = np.arange(len(rows))[::-1]
    shade_rows(axE, ys, [r[1] for r in rows])
    box_props = dict(color='black', linewidth=0.7)
    for y, (_, _, vals) in zip(ys, rows):
        if len(vals) > 2:
            axE.boxplot([vals], positions=[y], vert=False, widths=0.55,
                        showfliers=False, whis=(0, 100), boxprops=box_props,
                        whiskerprops=box_props, capprops=box_props,
                        medianprops=dict(color='black', linewidth=1.2),
                        zorder=2)
        if len(vals):
            axE.scatter(vals, np.full(len(vals), y), s=9, color='black', zorder=3,
                        linewidths=0)
        else:
            axE.text(555, y, 'R50 not reached', fontsize=6, style='italic',
                     color=GREY, ha='center', va='center', zorder=3)
    axE.set_yticks(ys)
    axE.set_yticklabels([r[0] for r in rows])
    axE.set_ylim(-0.6, len(rows) - 0.4)
    xmax = max(830, int(np.nanmax(df['R50'])) + 40)
    axE.set_xlim(280, xmax)
    axE.set_xticks([t for t in range(300, xmax, 100)])
    axE.set_xlabel('R50', fontweight='bold')
    axE.tick_params(axis='y', length=0)
    handles = [Patch(facecolor=COND_SHADE[c], label=COND_LABEL[c]) for c in COND_ORDER]
    axE.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.35, -0.06),
               ncol=2, handlelength=1.4, borderaxespad=0.0, columnspacing=1.2,
               labelspacing=0.3)

    # ── F: every individual strategy, ranked best to worst ──
    d = df.copy()
    d['label'] = [f'{m} {STRATEGY_LABELS.get(s, s)} ({COND_LABEL[c]})'
                  for m, s, c in zip(d['method'], d['strategy'], d['condition'])]
    d = d.sort_values('R50', kind='stable').reset_index(drop=True)
    labels = list(d['label']) + ['Permutation']
    conds = list(d['condition']) + ['perm']
    vals = list(d['R50']) + [PERMUTATION_R50]
    yf = np.arange(len(labels))[::-1]
    shade_rows(axF, yf, conds)
    axF.scatter(vals, yf, s=11, color='black', zorder=3, linewidths=0)
    axF.set_yticks(yf)
    axF.set_yticklabels(labels, fontsize=6)
    axF.set_ylim(-0.6, len(labels) - 0.4)
    axF.set_xlim(280, xmax)
    axF.set_xticks([t for t in range(300, xmax, 100)])
    axF.set_xlabel('R50', fontweight='bold')
    axF.tick_params(axis='y', length=0)

    letters(fig, [axE, axF], ['A', 'B'], dx=-0.02)
    save(fig, 'fig5_benchmark_conditions')

    piv = df.groupby(['method', 'condition'])['R50'].mean().round(1).unstack()
    print(piv.reindex(index=[m for m in METHOD_ORDER if m in piv.index],
                      columns=[c for c in COND_ORDER if c in piv.columns]).to_string())


# ══════════════════════════════════════════════════════════════════════════
# Figure S6: decoy blueprints, external classifiers and architectural ablations
# ══════════════════════════════════════════════════════════════════════════
DECOYS = ['correct', 'reversed', 'shuffled', 'random']
DECOY_STYLE = {'correct': ('Correct', MODEL, '-'), 'reversed': ('Reversed', ALT, '-'),
               'shuffled': ('Shuffled', '#8C8C8C', '--'), 'random': ('Random', '#BDBDBD', '--')}
ABLATIONS = ['full', 'no_tdvae', 'no_lstm', 'no_blueprint']
ABLATION_STYLE = {'full': ('Full model', MODEL, '-'), 'no_tdvae': ('No TD-VAE', ALT, '-'),
                  'no_lstm': ('No LSTM', '#E69F00', '-'),
                  'no_blueprint': ('No blueprint', '#BDBDBD', '--')}


def _json(*parts):
    return json.load(open(os.path.join(RES, *parts)))


def _per_bin(d):
    bins = sorted(d['acc_per_bin'], key=int)
    return [int(b) for b in bins], [d['acc_per_bin'][b] * 100 for b in bins]


def figure_blueprint_controls():
    # recreation (_v2) and autoregressive generation of the four blueprint models (Experiment 1)
    rec = {c: _json('exp1_decoy', c, f'exp1_{c}_recreation_acc_v2.json') for c in DECOYS}
    gen = {c: _json('exp1_decoy', c, f'exp1_{c}_recreation_acc.json') for c in DECOYS}
    # classification, recreation and generation of the four ablation models (Experiment 3)
    a_cls = {c: _json('exp6_ablation', c, f'exp6_{c}_eval_result.json') for c in ABLATIONS}
    a_rec = {c: _json('exp6_ablation', c, f'exp6_{c}_recreation_acc_v2.json') for c in ABLATIONS}
    a_gen = {c: _json('exp6_ablation', c, f'exp6_{c}_recreation_acc.json') for c in ABLATIONS}
    scores = pd.read_csv(os.path.join(RES, 'exp12_external', 'exp12_external_scores.csv'))
    summary = pd.read_csv(os.path.join(RES, 'exp12_external',
                                       'exp12_external_summary.csv')).set_index('model')

    fig = plt.figure(figsize=(W2COL, 4.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.75, hspace=0.75)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    axD, axE, axF = (fig.add_subplot(gs[1, i]) for i in range(3))

    # A: recreation vs autoregressive generation, four blueprints
    x = np.arange(4)
    w = 0.38
    r_ = [rec[c]['mean_acc'] * 100 for c in DECOYS]
    g_ = [gen[c]['mean_acc'] * 100 for c in DECOYS]
    axA.bar(x - w / 2, r_, w, color=NULLBAR, label='Recreation accuracy')
    axA.bar(x + w / 2, g_, w, color=MODEL, label='Generation accuracy')
    # value labels sit above the bar, or above the chance line when the bar ends below it,
    # and are pushed apart by a fixed distance in points so that neighbours never touch
    for xs, vals, dx, col in ((x - w / 2, r_, -3.5, '0.3'), (x + w / 2, g_, 3.5, MODEL)):
        for xi, v in zip(xs, vals):
            axA.annotate(f'{v:.1f}', (xi, max(v, 10.5)), xytext=(dx, 1.5), textcoords='offset points',
                         ha='center', va='bottom', fontsize=5.2, color=col)
    axA.axhline(10, color=GREY, linestyle='--', linewidth=0.8, label='Chance')
    axA.set_xticks(x)
    axA.set_xticklabels([DECOY_STYLE[c][0] for c in DECOYS], rotation=30, ha='right',
                        rotation_mode='anchor')
    axA.set_xlim(-0.95, 3.6)
    axA.set_xlabel('Training blueprint')
    axA.set_ylabel('Accuracy (%)')
    axA.set_ylim(0, 126)
    axA.set_yticks([0, 20, 40, 60, 80, 100])
    axA.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.04), handlelength=1.2,
               labelspacing=0.25, fontsize=6)

    # B: recreation accuracy per time bin, four blueprints
    for c in DECOYS:
        lab, color, ls = DECOY_STYLE[c]
        bx, by = _per_bin(rec[c])
        axB.plot(bx, by, ls, marker='o', markersize=2.6, color=color, label=lab)
    axB.axhline(10, color=GREY, linestyle=':', linewidth=0.8)
    axB.set_xlabel('Time bin')
    axB.set_ylabel('Recreation accuracy (%)')
    axB.set_ylim(0, 105)
    axB.set_xticks([2, 4, 6, 8, 10])
    axB.set_xticklabels(['t2', 't4', 't6', 't8', 't10'])
    axB.legend(loc='center right', handlelength=1.5, labelspacing=0.2, fontsize=6)

    # C: mean external accuracy over generated bins, all eight models
    # the four blueprint models of Experiment 1, then the four ablation models of Experiment 3
    order = ['exp1_correct', 'exp1_reversed', 'exp1_shuffled', 'exp1_random',
             'exp6_full', 'exp6_no_lstm', 'exp6_no_tdvae', 'exp6_no_blueprint']
    nice = {'exp1_correct': 'Correct blueprint',
            'exp1_reversed': 'Reversed blueprint', 'exp6_full': 'Full model',
            'exp1_shuffled': 'Shuffled blueprint', 'exp1_random': 'Random blueprint',
            'exp6_no_lstm': 'No LSTM', 'exp6_no_tdvae': 'No TD-VAE',
            'exp6_no_blueprint': 'No blueprint'}
    present = [m for m in order if m in summary.index]
    lr = np.array([summary.loc[m, 'lr_acc'] * 100 for m in present])
    ce = np.array([summary.loc[m, 'centroid_acc'] * 100 for m in present])
    # two series kept apart: a gap and a heading for each, and a second bar colour
    is_abl = np.array([m.startswith('exp6') for m in present])
    n_dec, n_abl = int((~is_abl).sum()), int(is_abl.sum())
    yy = np.concatenate([n_abl + 1.6 + np.arange(n_dec)[::-1], np.arange(n_abl)[::-1]])
    cols = [ALT if a else MODEL for a in is_abl]
    axC.barh(yy, lr, 0.6, color=cols)
    axC.scatter(ce, yy, s=12, color='black', marker='D', zorder=3)
    for yi, a, b in zip(yy, lr, ce):
        axC.annotate(f'{a:.1f}', (max(a, b), yi), xytext=(5, 0), textcoords='offset points',
                     va='center', fontsize=6, color='0.3')
    axC.axvline(10, color=GREY, linestyle='--', linewidth=0.8)
    axC.set_yticks(yy)
    axC.set_yticklabels([nice[m] for m in present], fontsize=6)
    axC.set_xlabel('Mean external accuracy (%)')
    axC.set_xlim(0, 100)
    axC.set_ylim(-0.7, yy.max() + 1.35)
    axC.text(98, yy[~is_abl].max() + 0.72, 'Decoy blueprints', ha='right', va='center',
             fontsize=6.5, fontweight='bold', color=MODEL)
    axC.text(98, yy[is_abl].max() + 0.72, 'Ablations', ha='right', va='center',
             fontsize=6.5, fontweight='bold', color=ALT)
    axC.legend(handles=[Patch(facecolor='0.6', label='Logistic regression'),
                        Line2D([0], [0], marker='D', color='none', markerfacecolor='black',
                               markeredgecolor='black', markersize=4, label='Nearest centroid')],
               loc='center right', bbox_to_anchor=(1.03, 0.63), handlelength=0.9,
               handletextpad=0.4, fontsize=5.5)

    # D: external nearest-centroid accuracy per generated bin, four blueprints
    for c in DECOYS:
        lab, color, ls = DECOY_STYLE[c]
        sub = scores[scores['model'] == f'exp1_{c}'].sort_values('time_bin')
        axD.plot(sub['time_bin'], sub['centroid_acc'] * 100, ls, marker='o', markersize=2.6,
                 color=color, label=lab)
    axD.axhline(10, color=GREY, linestyle=':', linewidth=0.8)
    axD.set_xlabel('Generated time bin')
    axD.set_ylabel('Nearest-centroid accuracy (%)')
    axD.set_ylim(0, 100)
    axD.set_xticks([2, 4, 6, 8, 10])
    axD.set_xticklabels(['t2', 't4', 't6', 't8', 't10'])
    axD.legend(loc='upper right', bbox_to_anchor=(1.03, 1.03), handlelength=1.5,
               labelspacing=0.2, fontsize=6)

    # E: classification, recreation and generation accuracy of the ablation models
    x = np.arange(4)
    w = 0.27
    series = [('Classification', [a_cls[c]['mean_acc_correct_bp'] * 100 for c in ABLATIONS], '#E3E3E3'),
              ('Recreation', [a_rec[c]['mean_acc'] * 100 for c in ABLATIONS], '#9E9E9E'),
              ('Generation', [a_gen[c]['mean_acc'] * 100 for c in ABLATIONS], MODEL)]
    for k, (lab, vals, col) in enumerate(series):
        axE.bar(x + (k - 1) * w, vals, w, color=col, label=lab,
                edgecolor='0.55' if k == 0 else 'none', linewidth=0.4)
    for xi, v in zip(x + w, series[2][1]):
        axE.annotate(f'{v:.1f}', (xi, max(v, 10.5)), xytext=(0, 1.5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=5.2, color=MODEL)
    axE.axhline(10, color=GREY, linestyle='--', linewidth=0.8)
    axE.set_xticks(x)
    axE.set_xticklabels([ABLATION_STYLE[c][0] for c in ABLATIONS], rotation=30, ha='right',
                        rotation_mode='anchor')
    axE.set_xlim(-0.6, 3.6)
    axE.set_ylabel('Accuracy (%)')
    axE.set_ylim(0, 142)
    axE.set_yticks([0, 20, 40, 60, 80, 100])
    axE.legend(loc='upper right', bbox_to_anchor=(1.03, 1.04), handlelength=1.0,
               labelspacing=0.2, handletextpad=0.4, fontsize=6)

    # F: autoregressive generation accuracy per time bin, ablation models
    for c in ABLATIONS:
        lab, color, ls = ABLATION_STYLE[c]
        bx, by = _per_bin(a_gen[c])
        axF.plot(bx, by, ls, marker='o', markersize=2.6, color=color, label=lab)
    axF.axhline(10, color=GREY, linestyle=':', linewidth=0.8)
    axF.set_xlabel('Time bin')
    axF.set_ylabel('Generation accuracy (%)')
    axF.set_ylim(0, 105)
    axF.set_xticks([2, 4, 6, 8, 10])
    axF.set_xticklabels(['t2', 't4', 't6', 't8', 't10'])
    axF.legend(loc='upper right', bbox_to_anchor=(1.03, 1.03), handlelength=1.5,
               labelspacing=0.2, fontsize=6)

    letters(fig, [axA, axB, axC], ['A', 'B', 'C'])
    letters(fig, [axD, axE, axF], ['D', 'E', 'F'])
    save(fig, 'figS_blueprint_controls')


# ══════════════════════════════════════════════════════════════════════════
# Random-initialization baseline (Figure 3 protocol) and trichoblast default
# ══════════════════════════════════════════════════════════════════════════
def figure_random_baseline():
    d25 = os.environ.get('EXP25_DIR', os.path.join(RES, 'exp25_random_baseline_balanced'))
    rec = pd.read_csv(os.path.join(d25, 'exp25_recovery.csv'))
    seeds = pd.read_csv(os.path.join(d25, 'exp25_seed_scores.csv'))
    dist = pd.read_csv(os.path.join(RES, 'exp11_trichoblast',
                                    'exp11_trained_belief_distances.csv'))
    n_classes = dist.shape[0]

    fig = plt.figure(figsize=(W2COL, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=1.0)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    # A: recovery rate vs fraction of genes kept, categorical x axis
    fracs = sorted(rec['fraction'].unique())
    x = np.arange(len(fracs))
    for model, color, marker, lab in [('trained', MODEL, 'o', 'Trained model'),
                                      ('random', ALT, 's',
                                       'Randomly initialized model')]:
        g = rec[rec['model'] == model].groupby('fraction')['accuracy']
        m = np.array([g.mean()[f] for f in fracs])
        s = np.array([g.std()[f] if not np.isnan(g.std()[f]) else 0 for f in fracs])
        axA.errorbar(x, m, yerr=s, color=color, marker=marker, markersize=3.5,
                     capsize=2, linewidth=1.1, label=lab)
    axA.axhline(1.0 / n_classes, color=GREY, linestyle='--', linewidth=0.8,
                label='Chance (1/10)')
    axA.set_xticks(x)
    axA.set_xticklabels([f'{f * 100:g}' for f in fracs])
    axA.set_xlabel('Genes kept (%)')
    axA.set_ylabel('Cell type recovery rate')
    axA.set_ylim(-0.03, 1.08)
    axA.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.01), handlelength=1.4,
               labelspacing=0.25, fontsize=6)

    # B: dominant prediction of ten untrained models on fully masked input
    classes = list(dist['terminal_cell_type'])
    s0 = seeds[seeds['fraction'] == 0.0].sort_values('seed')
    yidx = [classes.index(c) for c in s0['dominant_class']]
    axB.scatter(s0['seed'], yidx, s=18, color=ALT, zorder=3, linewidths=0)
    axB.set_yticks(range(n_classes))
    axB.set_yticklabels(classes, fontsize=6)
    axB.set_ylim(-0.6, n_classes - 0.4)
    axB.set_xticks(range(len(s0)))
    axB.set_xlabel('Initialization seed')
    axB.set_ylabel('Predicted cell type\n(all genes masked)')
    acc0 = s0['accuracy'].mean() * 100
    axB.text(0.98, 0.98, f'recovery rate {acc0 / 100:.2f}\nfor every seed',
             transform=axB.transAxes, ha='right', va='top', fontsize=6, color='0.3')

    # C: cosine similarity, zero-input belief of the trained model vs class means
    dd = dist.sort_values('cosine_sim_zero_to_mean_belief')
    colors = [MODEL if ct == 'Trichoblast' else NULLBAR
              for ct in dd['terminal_cell_type']]
    yc = np.arange(len(dd))
    axC.barh(yc, dd['cosine_sim_zero_to_mean_belief'], 0.65, color=colors)
    axC.set_yticks(yc)
    axC.set_yticklabels(dd['terminal_cell_type'], fontsize=6)
    axC.set_xlim(0, 1.0)
    axC.set_xlabel('Cosine similarity to the mean\nbelief state of each cell type')
    axC.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    letters(fig, [axA, axB, axC], ['A', 'B', 'C'])
    save(fig, 'figS_random_baseline')


# ══════════════════════════════════════════════════════════════════════════
# Genome-wide gold-standard recall
# ══════════════════════════════════════════════════════════════════════════
def figure_genomewide():
    """Ranking within the annotated transcription factor set (A) against
    discovering transcription factors among all genes (B), for the same methods
    on the same data, with the condition check in C and D."""
    df = pd.read_csv(_table(GW_TABLE))
    bench = pd.read_csv(_table(BENCH_TABLE))
    bench = bench[bench['condition'] == 'genesys']
    ks = [100, 200, 300, 400]
    chance = df[df['method'] == 'expected_by_chance'].iloc[0]

    # Four panels of equal size on a 2 x 2 grid, so that every axis edge and every panel
    # letter lines up: A and B on top, C and D below.
    fig = plt.figure(figsize=(W2COL, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.42)
    axA = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    # ── A: R50 within the expressed transcription factor set ──
    labels = {'LIMA': 'LIMA\ncentrality', 'DE': 'DE',
              'GRNBoost2': 'GRNBoost2\ncentrality', 'Knockout': 'In silico\nknockout'}
    rows = [(m, bench[bench['method'] == m]['R50'].values) for m in GENERATED_ORDER]
    rows = [(m, v) for m, v in rows if len(v)]
    ys = np.arange(len(rows))[::-1]
    axA.axhspan(-0.6, len(rows) - 0.3, color=COND_SHADE['genesys'], zorder=0, lw=0)
    for y, (m, vals) in zip(ys, rows):
        if len(vals) > 2:
            axA.plot([vals.min(), vals.max()], [y, y], '-', color='black', linewidth=0.8,
                     zorder=2)
        axA.scatter(vals, np.full(len(vals), y), s=14, color='black', zorder=3,
                    linewidths=0)
    axA.axvline(PERMUTATION_R50, color=GREY, linestyle='--', linewidth=0.8)
    axA.text(PERMUTATION_R50 - 8, ys[0] + 0.55, 'permutation', fontsize=6, color=GREY,
             ha='right', va='center')
    axA.set_yticks(ys)
    axA.set_yticklabels([labels[m] for m, _ in rows])
    axA.set_ylim(-0.6, len(rows) - 0.3)
    xmaxA = max(860, int(np.nanmax(bench['R50'])) + 40)
    axA.set_xlim(280, xmaxA)
    axA.set_xticks([t for t in range(300, xmaxA, 100)])
    axA.set_xlabel('R50 (lower is better)')
    axA.tick_params(axis='y', length=0)

    # ── B: the same methods, genome-wide, on the same data ──
    axB = axes[0]
    gen = df[df['condition'] == 'genesys']
    # one colour per method, kept apart from the four scenario colours of C and D;
    # the three knockout scores share one brown hue in three shades
    gw = [('LIMA_degree', 'LIMA degree centrality', '#762A83', 'o', '-'),
          ('DE_myAUC', 'DE (mean AUC)', '#E69F00', 's', '-'),
          ('knockout_gen_effect', 'Knockout, generative', '#543005', '^', '-'),
          ('knockout_combined_effect', 'Knockout, combined', '#8C510A', 'D', '--'),
          ('knockout_cls_effect', 'Knockout, classifier', '#A0785A', 'v', ':')]
    for key, lab, color, mk, ls in gw:
        sub = gen[gen['method'] == key]
        if not len(sub):
            continue
        y = [sub.iloc[0][f'recall_at_{k}'] * 100 for k in ks]
        axB.plot(ks, y, ls, marker=mk, markersize=3, color=color, label=lab)
    axB.plot(ks, [chance[f'recall_at_{k}'] * 100 for k in ks], '--', color=GREY,
             linewidth=0.9, label='chance')
    axB.text(0.02, 0.97, 'GeneSys-generated data', transform=axB.transAxes, zorder=4,
             ha='left', va='top', fontsize=7.5)
    axB.legend(loc='upper left', bbox_to_anchor=(0.0, 0.9), handlelength=1.6,
               labelspacing=0.2, fontsize=6)

    # ── C, D: is the genome-wide picture specific to one data condition? ──
    markers = {'genesys': 'o', 'scrnaseq': 's', 'pseudobulk': '^', 'scvi': 'D'}
    for ax, (key, title) in zip(axes[1:], [('LIMA_degree', 'LIMA degree centrality'),
                                           ('DE_myAUC', 'DE (mean AUC)')]):
        sub = df[df['method'] == key]
        for _, row in sub.iterrows():
            y = [row[f'recall_at_{k}'] * 100 for k in ks]
            ax.plot(ks, y, '-', marker=markers[row['condition']], markersize=3,
                    color=COND_COLOR[row['condition']], label=COND_LABEL[row['condition']])
        ax.plot(ks, [chance[f'recall_at_{k}'] * 100 for k in ks], '--', color=GREY,
                linewidth=0.9, label='chance')
        ax.text(0.04, 0.97, title, transform=ax.transAxes, ha='left', va='top',
                fontsize=7.5)
    axes[1].legend(loc='upper left', bbox_to_anchor=(0.0, 0.9), handlelength=1.4,
                   labelspacing=0.2, fontsize=5.8)

    for ax in axes:
        ax.set_xticks(ks)
        ax.set_xlabel('K (top-ranked genes)')
        ax.set_ylim(0, max(11, float(df[[f'recall_at_{k}' for k in ks]].max().max()) * 115))
    for ax in axes:
        ax.set_ylabel('Gold-standard TF recall (%)')

    # Panel letters share one x per column and one y per row.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bb = {k: a.get_tightbbox(rend).transformed(inv)
          for k, a in zip('ABCD', [axA, axes[0], axes[1], axes[2]])}
    xl = min(bb['A'].x0, bb['C'].x0) - 0.012
    xr = min(bb['B'].x0, bb['D'].x0) - 0.012
    yt = max(bb['A'].y1, bb['B'].y1) + 0.012
    yb = max(bb['C'].y1, bb['D'].y1) + 0.012
    for lab, x, y in [('A', xl, yt), ('B', xr, yt), ('C', xl, yb), ('D', xr, yb)]:
        fig.text(x, y, lab, fontsize=10, fontweight='bold', ha='left', va='bottom')
    save(fig, 'figS_genomewide')


# ══════════════════════════════════════════════════════════════════════════
# Figure S5E-F: recreation from zero-padded input (joined to the Illustrator panels A-D
# of Figure S5 by experiments/exp41_supplementary_composites.py)
# ══════════════════════════════════════════════════════════════════════════
def figure_extrapolation_panels():
    df = pd.read_csv(os.path.join(RES, 'exp7_extrapolation.csv'))
    seeds = sorted(df['seed_bins'].unique(), reverse=True)
    bins = sorted(df['time_bin'].unique())
    fig, (axE, axF) = plt.subplots(1, 2, figsize=(W2COL, 1.9))
    fig.subplots_adjust(wspace=0.18)
    for ax, metric, lab in ((axE, 'accuracy', 'Recreation accuracy'),
                            (axF, 'cosine_similarity', 'Cosine similarity to real cells')):
        piv = df.pivot_table(index='seed_bins', columns='time_bin', values=metric).reindex(seeds)
        im = ax.imshow(piv.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        for i in range(len(seeds)):
            for j in range(len(bins)):
                v = piv.values[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=4.6,
                        color='white' if v < 0.5 else 'black')
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f't{t}' for t in bins])
        ax.set_yticks(range(len(seeds)))
        ax.set_yticklabels([str(s) for s in seeds])
        ax.set_xlabel('Time bin')
        ax.set_title(lab, fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axE.set_ylabel('Seed time bins provided')
    cb = fig.colorbar(im, ax=[axE, axF], fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    # letters sized as in the Illustrator panels A-D (10 pt at the 174 mm width)
    letters(fig, [axE, axF], ['E', 'F'], dy=0.03)
    save(fig, 'figS5EF_extrapolation')


# ══════════════════════════════════════════════════════════════════════════
# Figure S14: sensitivity to the noise level and the model width, and the
# TD-VAE loss components during training
# ══════════════════════════════════════════════════════════════════════════
SIGMAS = ['0.0', '0.1', '0.2', '0.4']
SIGMA_COLOR = {'0.0': '#BDBDBD', '0.1': '#8C8C8C', '0.2': MODEL, '0.4': ALT}


def figure_model_sensitivity():
    noise = {s: _json('exp4_noise', f'sigma_{s}', f'exp4_{s}_recreation_acc_v2.json') for s in SIGMAS}
    summ = pd.read_csv(os.path.join(RES, 'exp10_latent_sweep_summary.csv')).sort_values('dim')
    kl = pd.read_csv(os.path.join(RES, 'exp10_latent_sweep_kl_logs.csv'))
    diag = pd.read_csv(os.path.join(RES, 'exp3_kl_divergence_diagnostic.csv'))

    fig = plt.figure(figsize=(W2COL, 6.9))
    gs = fig.add_gridspec(3, 3, wspace=0.7, hspace=0.85)
    axA, axB, axC, axD, axE, axF, axG, axH, axI = (fig.add_subplot(gs[i // 3, i % 3])
                                                    for i in range(9))

    # A, B: recreation accuracy at four noise levels
    m = [noise[s]['mean_acc'] * 100 for s in SIGMAS]
    axA.bar(range(4), m, 0.6, color=[SIGMA_COLOR[s] for s in SIGMAS])
    for xi, v in enumerate(m):
        axA.annotate(f'{v:.1f}', (xi, v), xytext=(0, 1.5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=5.5, color='0.3')
    axA.axhline(10, color=GREY, linestyle='--', linewidth=0.8)
    axA.set_xticks(range(4))
    axA.set_xticklabels([f'{s}\n(default)' if s == '0.2' else s for s in SIGMAS])
    axA.set_xlabel('Noise level (σ)')
    axA.set_ylabel('Mean recreation accuracy (%)')
    axA.set_ylim(0, 108)
    axA.set_yticks([0, 20, 40, 60, 80, 100])
    for s in SIGMAS:
        bx, by = _per_bin(noise[s])
        axB.plot(bx, by, '-', marker='o', markersize=2.4, color=SIGMA_COLOR[s],
                 label=f'σ = {s}')
    axB.set_xlabel('Time bin')
    axB.set_ylabel('Recreation accuracy (%)')
    axB.set_ylim(0, 105)
    axB.set_xticks([2, 4, 6, 8, 10])
    axB.set_xticklabels(['t2', 't4', 't6', 't8', 't10'])
    axB.legend(loc='lower right', handlelength=1.5, labelspacing=0.2, fontsize=6)

    # C: classification and generation accuracy at four widths
    x = np.arange(len(summ))
    w = 0.30
    rec = summ['recreation_mean_acc'].values * 100
    gen = summ['autoregressive_mean_acc'].values * 100
    axC.bar(x - w / 2 - 0.015, rec, w, color=NULLBAR, label='Classification')
    axC.bar(x + w / 2 + 0.015, gen, w, color=MODEL, label='Generation')
    for xi, v in zip(x + w / 2 + 0.015, gen):
        axC.annotate(f'{v:.1f}', (xi, v), textcoords='offset points', xytext=(0, 2.5),
                     ha='center', va='bottom', fontsize=5.5, color=MODEL)
    axC.set_xticks(x)
    axC.set_xticklabels([f'{int(d)}\n{int(z)}' for d, z in
                         zip(summ['dim'], summ['latent_dim'])], fontsize=6.5)
    axC.set_xlabel('Model width (top) and\nlatent dimension (bottom)')
    axC.set_ylabel('Accuracy (%)')
    axC.set_ylim(0, 136)
    axC.set_yticks([0, 20, 40, 60, 80, 100])
    axC.set_xlim(-0.62, len(summ) - 0.38)
    axC.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.04), ncol=2, handlelength=1.0,
               columnspacing=0.8, handletextpad=0.3, fontsize=6)

    # D: KL(z1) across training at each width
    dim_colors = {64: '#9ECAE1', 128: '#4292C6', 256: '#08519C', 512: '#08306B'}
    kl_col = 'kl_z1' if 'kl_z1' in kl.columns else kl.columns[-1]
    for d in sorted(kl['dim'].unique()):
        sub = kl[kl['dim'] == d].sort_values('epoch')
        axD.plot(sub['epoch'], sub[kl_col], color=dim_colors.get(int(d), 'black'),
                 label=f'width {int(d)}')
    axD.set_xlabel('Training epoch')
    axD.set_ylabel('KL(z$_1$) (nats)')
    axD.set_ylim(0, 600)
    axD.set_xlim(0, 31)
    axD.legend(loc='center right', bbox_to_anchor=(1.0, 0.62), handlelength=1.5,
               labelspacing=0.2, fontsize=6)

    # E: KL(z1) per latent dimension at the end of training
    per_dim = summ['final_kl_z1'].values / summ['latent_dim'].values
    axE.plot(summ['latent_dim'], per_dim, 'o-', color=MODEL, markersize=3.5)
    for xd, v in zip(summ['latent_dim'], per_dim):
        axE.annotate(f'{v:.2f}', (xd, v), textcoords='offset points', xytext=(0, -9),
                     ha='center', va='top', fontsize=5.5, color=MODEL)
    axE.axhline(0, color=GREY, linestyle='--', linewidth=0.8)
    axE.annotate('posterior collapse', (0.5, 0), xycoords=('axes fraction', 'data'),
                 textcoords='offset points', xytext=(0, 3), ha='center', va='bottom',
                 fontsize=5.5, color=GREY)
    axE.set_xscale('log', base=2)
    axE.set_xticks(summ['latent_dim'].values)
    axE.set_xticklabels([str(int(v)) for v in summ['latent_dim'].values])
    axE.minorticks_off()
    axE.set_xlim(summ['latent_dim'].min() / 1.5, summ['latent_dim'].max() * 1.5)
    axE.set_xlabel('TD-VAE latent dimension')
    axE.set_ylabel('KL(z$_1$) per latent\ndimension (nats)')
    axE.set_ylim(-0.07, 0.72)

    # F to I: the four loss components of one model trained from scratch at the default width
    comps = [(axF, 'train_kl_z1', 'val_kl_z1', 'KL(z$_1$) (nats)'),
             (axG, 'train_kl_z2', 'val_kl_z2', 'KL(z$_2$) (nats)'),
             (axH, 'train_recon', 'val_recon', 'Reconstruction loss (nats)'),
             (axI, 'train_nll', None, 'Classifier NLL (nats)')]
    for ax, tr, va, lab in comps:
        ax.plot(diag['epoch'], diag[tr], '-', color=MODEL, label='Training')
        if va:
            ax.plot(diag['epoch'], diag[va], '--', color=ALT, label='Validation')
        ax.set_xlabel('Training epoch')
        ax.set_ylabel(lab)
        ax.set_xlim(0, diag['epoch'].max() + 1)
    axF.set_ylim(0, 320)
    axF.axhline(0, color=GREY, linestyle=':', linewidth=0.8)
    axF.legend(loc='lower right', handlelength=1.5, labelspacing=0.2, fontsize=6)

    letters(fig, [axA, axB, axC], ['A', 'B', 'C'])
    letters(fig, [axD, axE, axF], ['D', 'E', 'F'])
    letters(fig, [axG, axH, axI], ['G', 'H', 'I'])
    save(fig, 'figS_model_sensitivity')


# ══════════════════════════════════════════════════════════════════════════
# Figure S15: sensitivity of LIMA to the solver, the sample size, the
# generated-cell spread and the z-score threshold
# ══════════════════════════════════════════════════════════════════════════
def figure_lima_sensitivity():
    cond = pd.read_csv(os.path.join(RES, 'exp16_identifiability', 'exp16_conditions.csv')).set_index('label')
    agree = pd.read_csv(os.path.join(RES, 'exp16_identifiability',
                                     'exp16_ranking_agreement.csv')).set_index('label')
    edges = pd.read_csv(os.path.join(RES, 'exp5_zscore_edge_counts.csv'))
    tf = pd.read_csv(os.path.join(RES, 'exp9_zscore_tf_recovery_aggregate.csv'))
    tf = tf[(tf['gold_standard'] == 'Intersection') & (tf['cell_type'] == 'ALL')]
    BASE = 'ridge_rows200_eps1.0_lam0'
    MINN = 'minnorm_rows200_eps1.0_lamminnorm'
    RIDGE = [f'ridge_rows200_eps1.0_lam{l}' for l in ('0.01', '0.1', '1.0', '10.0')]
    # one colour family per experiment: purple for the solver, green for sample size and
    # spread, orange-brown for the z-score threshold; grey for the pipeline solution
    SOLVER, SOLVER_LT = '#762A83', '#AF8DC3'
    ROWS, SPREAD = '#1B7837', '#5AAE61'
    PIPE = '#4D4D4D'

    fig = plt.figure(figsize=(W2COL, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.35], wspace=0.75, hspace=0.8)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    axD, axE, axF = (fig.add_subplot(gs[1, i]) for i in range(3))

    # (A) agreement of the TF ranking with that of the pipeline solution
    labs = [MINN] + RIDGE
    yv_all = agree.loc[labs, 'spearman_vs_lstsq_baseline'].values
    xpos = np.arange(len(labs))
    half = cond.loc[BASE, 'splithalf_spearman_tf']
    axA.axhline(half, color=GREY, linestyle='--', linewidth=0.8)
    axA.text(len(labs) - 0.6, half + 0.006, 'two halves of the\npipeline replicates', fontsize=5.5,
             color=GREY, ha='right', va='bottom')
    axA.plot(xpos[:1], yv_all[:1], 's', color=SOLVER_LT, markersize=4)
    axA.plot(xpos[1:], yv_all[1:], 'o-', color=SOLVER, markersize=3.5)
    # labels sit to the upper left of each point, clear of the rising line
    for k, (xv, yv) in enumerate(zip(xpos, yv_all)):
        axA.annotate(f'{yv:.2f}', (xv, yv), textcoords='offset points',
                     xytext=(0, -7) if k == 0 else (4, -3) if k == len(labs) - 1 else (-3, 4),
                     ha='center' if k == 0 else 'left' if k == len(labs) - 1 else 'right',
                     va='bottom' if 0 < k < len(labs) - 1 else 'top',
                     fontsize=5.5, color='0.3')
    axA.set_xticks(xpos)
    axA.set_xticklabels(['all\ndirections', '0.01', '0.1', '1', '10'])
    axA.set_xlabel('Ridge penalty (lambda)')
    axA.set_ylabel('Spearman correlation with the\nTF ranking of the pipeline solution')
    axA.set_ylim(0.70, 1.02)
    axA.set_xlim(-0.5, len(labs) - 0.5)

    # (B) split-half stability against sample size and spread
    rows_l = ['rows_rows50_eps1.0_lam0', BASE, 'rows_rows800_eps1.0_lam0']
    spread_l = ['spread_rows200_eps0.5_lam0', BASE, 'spread_rows200_eps2.0_lam0']
    axB.plot(range(3), cond.loc[rows_l, 'splithalf_spearman_tf'], 'o-', color=ROWS,
             markersize=3.5, label='Rows per state\n(50 / 200 / 800)')
    axB.plot(range(3), cond.loc[spread_l, 'splithalf_spearman_tf'], 's--', color=SPREAD,
             markersize=3.5, label='Latent spread\n(0.5x / 1x / 2x)')
    axB.set_xticks(range(3))
    axB.set_xticklabels(['low', 'baseline', 'high'])
    axB.set_ylabel('Split-half Spearman correlation\nof the TF ranking')
    axB.set_ylim(0.90, 1.0)
    axB.legend(loc='lower right', handlelength=1.5, labelspacing=0.5, fontsize=6)

    # (C) recovery of the gold-standard TFs by the TF degree ranking of each condition
    order = [BASE, MINN] + RIDGE + ['rows_rows50_eps1.0_lam0', 'rows_rows800_eps1.0_lam0',
                                    'spread_rows200_eps0.5_lam0', 'spread_rows200_eps2.0_lam0']
    names = ['pipeline', 'all directions', 'lambda 0.01', 'lambda 0.1', 'lambda 1', 'lambda 10',
             '50 rows', '800 rows', '0.5x spread', '2x spread']
    xs = np.array([0, 1, 2, 3, 4, 5, 6.6, 7.6, 8.6, 9.6])
    vals = cond.loc[order, 'r50_tf_degree'].values
    cols = [PIPE, SOLVER_LT] + [SOLVER] * 4 + [ROWS] * 2 + [SPREAD] * 2
    axC.bar(xs, vals, color=cols, width=0.65)
    perm = float(cond['r50_permutation'].iloc[0])
    axC.axhline(perm, color=GREY, linestyle='--', linewidth=0.8)
    axC.text(xs[-1] + 0.45, perm + 12, 'permutation', fontsize=6, color=GREY, ha='right',
             va='bottom')
    axC.set_xlim(-0.7, xs[-1] + 0.7)
    axC.set_xticks(xs)
    axC.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axC.set_ylabel('R50 of the TF degree ranking')
    axC.set_ylim(0, 860)

    # (D, E) edges kept by the z-score threshold, one line per transition
    trans = sorted(edges['transition'].unique())
    t_col = dict(zip(trans, ['#FEC44F', '#FE9929', '#EC7014', '#CC4C02', '#8C2D04']))
    for t in trans:
        g = edges[edges['transition'] == t].sort_values('threshold')
        axD.plot(g['threshold'], g['n_edges'] / 1e6, 'o-', markersize=2.6, color=t_col[t], label=t)
        axE.plot(g['threshold'], g['fraction'] * 100, 'o-', markersize=2.6, color=t_col[t], label=t)
    axD.set_ylabel('Non-zero edges (millions)')
    axE.set_ylabel('Edges retained (%)')
    axD.legend(loc='upper right', handlelength=1.5, labelspacing=0.2, fontsize=6)

    # (F) genome-wide recall of the gold-standard TFs, mean over the five transitions
    k_col = {100: '#FEC44F', 200: '#FE9929', 300: '#CC4C02', 400: '#8C2D04'}
    for k in (100, 200, 300, 400):
        avg = tf[tf['K'] == k].groupby('threshold')['recall'].mean()
        axF.plot(avg.index, avg.values * 100, 'o-', markersize=2.6, color=k_col[k], label=f'K = {k}')
    axF.set_ylabel('Gold-standard TF recall (%)')
    axF.set_ylim(0, 7.0)
    axF.set_yticks(range(0, 7))
    axF.legend(loc='upper left', ncol=2, handlelength=1.5, labelspacing=0.2, columnspacing=0.8,
               fontsize=6)
    for ax in (axD, axE, axF):
        ax.set_xlabel('Z-score threshold (|z| >)')
        ax.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])

    letters(fig, [axA, axB, axC], ['A', 'B', 'C'])
    letters(fig, [axD, axE, axF], ['D', 'E', 'F'])
    save(fig, 'figS_lima_sensitivity')


if __name__ == '__main__':
    print(f'Font: {FONT}')
    print('Figure 5 (benchmark conditions)'); figure4_expanded()
    print('Figure S5E-F (zero-padded recreation)'); figure_extrapolation_panels()
    print('Figure S6 (blueprint controls)'); figure_blueprint_controls()
    print('Figure S11 (genome-wide recall)'); figure_genomewide()
    print('Figure S14 (noise, width, loss components)'); figure_model_sensitivity()
    print('Figure S15 (LIMA sensitivity)'); figure_lima_sensitivity()
    if os.path.exists(os.path.join(os.environ.get('EXP25_DIR', os.path.join(RES, 'exp25_random_baseline_balanced')), 'exp25_seed_scores.csv')):
        print('Figure S7 (random baseline)'); figure_random_baseline()
    else:
        print('Random baseline: exp25 results not found, skipped')
    print('Done.')
