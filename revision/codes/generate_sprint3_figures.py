"""
Generate supplementary figures for GeneSys revision.

Produces 7 supplementary figures (PDF + PNG):
  Fig S6  — exp2_decoy_blueprint       (Blueprint sensitivity)
  Fig S9  — exp3_random_vs_trained     (Random baseline)
  Fig S14 — exp1_ablation              (Ablation study)
  Fig S15 — exp7_extrapolation_heatmap (Extrapolation)
  Fig S16 — exp5_noise_sweep           (Noise sensitivity)
  Fig S17 — exp4_kl_divergence         (KL divergence)
  Fig S18 — exp6_zscore_sensitivity    (Z-score threshold)

Figure legends are in the manuscript DOCX, not embedded in the images.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Nolan Lab data-viz style (cowplot theme, Wilke principles) ────────────────
# Source: gitlab.caltech.edu/nolan/data-viz
_FONT_SIZE = 14
plt.rcParams.update({
    # Font: Arial, 14pt base (Wilke Ch. 24: "axis labels too small" is #1 defect)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': _FONT_SIZE,
    'axes.titlesize': _FONT_SIZE,
    'axes.labelsize': _FONT_SIZE,
    'xtick.labelsize': _FONT_SIZE * 0.85,
    'ytick.labelsize': _FONT_SIZE * 0.85,
    'legend.fontsize': _FONT_SIZE * 0.85,

    # Background: white, no grid (cowplot default)
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.grid': False,

    # Spines: half-open frame (bottom + left only)
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.linewidth': 0.5,

    # Ticks: outward
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,

    # Legend: frameless (cowplot default)
    'legend.frameon': False,

    # Export: editable text in Illustrator/Inkscape
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'ps.fonttype': 42,
    'mathtext.fontset': 'dejavusans',
})

# Okabe-Ito CVD-safe qualitative palette (Wilke's recommendation)
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']

def panel_label(fig, ax, label):
    """Add panel label (A, B, ...) matching cowplot draw_plot_label style."""
    ax.text(-0.08, 1.05, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', fontfamily='sans-serif',
            verticalalignment='top', horizontalalignment='left')


def save_figure(fig, name, dpi=300):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f"{name}.{ext}"),
                    dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")


def load_eval_result(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Fig S14: Ablation Study
# ═══════════════════════════════════════════════════════════════════════════
def plot_exp1_ablation():
    print("[Fig S14] Ablation Study")
    conditions = ['full', 'no_tdvae', 'no_lstm', 'no_blueprint']
    labels = ['Full\nModel', 'No TD-VAE\n(NLL only)', 'No LSTM\n(MLP only)', 'No Blueprint\n(random)']
    cond_colors = [OKABE_ITO[4], OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[5]]  # blue, orange, sky, red-orange

    cls_results, rec_results, gen_results = {}, {}, {}
    for cond in conditions:
        cls_results[cond] = load_eval_result(os.path.join(RESULTS_DIR, f"exp6_ablation/{cond}/exp6_{cond}_eval_result.json"))
        rec_results[cond] = load_eval_result(os.path.join(RESULTS_DIR, f"exp6_ablation/{cond}/exp6_{cond}_recreation_acc_v2.json"))
        gen_results[cond] = load_eval_result(os.path.join(RESULTS_DIR, f"exp6_ablation/{cond}/exp6_{cond}_recreation_acc.json"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(left=0.08, right=0.97, wspace=0.35, top=0.88, bottom=0.18)

    # Panel A: grouped bar chart
    ax = axes[0]
    x_pos = np.arange(4)
    width = 0.25
    cls_means = [cls_results[c]['mean_acc_correct_bp'] for c in conditions]
    rec_means = [rec_results[c]['mean_acc'] for c in conditions]
    gen_means = [gen_results[c]['mean_acc'] for c in conditions]

    # Three shades derived from Okabe-Ito blue
    ax.bar(x_pos - width, cls_means, width, color='#9ecae1', edgecolor='white', linewidth=0.5, label='Classification')
    ax.bar(x_pos, rec_means, width, color='#4292c6', edgecolor='white', linewidth=0.5, label='Recreation')
    ax.bar(x_pos + width, gen_means, width, color='#084594', edgecolor='white', linewidth=0.5, label='Generation')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    # Label every bar uniformly: same offset, font, rotation
    for bars, means in [(ax.containers[0], cls_means), (ax.containers[1], rec_means), (ax.containers[2], gen_means)]:
        for bar, val in zip(bars, means):
            txt = f'{val:.0%}' if val > 0.5 else f'{val:.1%}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, txt,
                    ha='left', va='bottom', fontsize=8, fontweight='bold', rotation=45)
    # Place legend above the plot area so it doesn't clash with labels or y-axis ticks
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=3, borderaxespad=0)

    # Panel B: line plot with Okabe-Ito per condition
    ax = axes[1]
    for cond, label, color in zip(conditions, labels, cond_colors):
        bins_dict = gen_results[cond]['acc_per_bin']
        bins = sorted(bins_dict.keys(), key=int)
        accs = [bins_dict[b] for b in bins]
        ax.plot([int(b) for b in bins], accs, 'o-', color=color,
                label=label.replace('\n', ' '), linewidth=2, markersize=5)
    ax.set_xlabel('Time Bin')
    ax.set_ylabel('Generation Accuracy')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.05, 1.05)

    panel_label(fig, axes[0], 'A')
    panel_label(fig, axes[1], 'B')
    save_figure(fig, "exp6_ablation")


# ═══════════════════════════════════════════════════════════════════════════
# Fig S9: Random Baseline
# ═══════════════════════════════════════════════════════════════════════════
def plot_exp3_random():
    print("[Fig S9] Random Baseline")
    csv = os.path.join(RESULTS_DIR, "exp2_random_vs_trained_recovery.csv")
    df = pd.read_csv(csv)

    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, color, marker in [('trained', OKABE_ITO[4], 'o'), ('random', OKABE_ITO[5], 's')]:
        sub = df[df['model'] == model_name]
        means = sub.groupby('fraction')['accuracy'].mean()
        stds = sub.groupby('fraction')['accuracy'].std().fillna(0)
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    label=model_name.capitalize() + ' model', color=color, marker=marker,
                    capsize=3, linewidth=2, markersize=6)

    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.set_xlabel('Fraction of genes retained')
    ax.set_ylabel('Cell-type recovery accuracy')
    ax.legend(loc='center right')
    ax.set_ylim(-0.05, 1.05)

    save_figure(fig, "exp2_random_vs_trained_recovery")


# ═══════════════════════════════════════════════════════════════════════════
# Fig S17: KL Divergence
# ═══════════════════════════════════════════════════════════════════════════
def plot_exp4_kl():
    print("[Fig S17] KL Divergence")
    csv = os.path.join(RESULTS_DIR, "exp3_kl_divergence_diagnostic.csv")
    df = pd.read_csv(csv)

    panels = [
        ('train_kl_z1', 'val_kl_z1',   'KL(z\u2081)', OKABE_ITO[5]),   # red-orange
        ('train_kl_z2', 'val_kl_z2',   'KL(z\u2082)', OKABE_ITO[0]),   # orange
        ('train_recon', 'val_recon',    'Reconstruction Loss', OKABE_ITO[2]),  # green
        ('train_nll',   None,           'Classifier NLL', OKABE_ITO[4]),  # blue
    ]
    panel_labels_list = ['A', 'B', 'C', 'D']

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.93, bottom=0.07)
    for ax, (train_col, val_col, title, color), plabel in zip(axes.flat, panels, panel_labels_list):
        ax.plot(df['epoch'], df[train_col], color=color, linewidth=2, label='Train')
        if val_col and val_col in df.columns:
            ax.plot(df['epoch'], df[val_col], color=color, linewidth=2,
                    linestyle='--', label='Validation', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (nats)')
        ax.set_title(title, loc='center')
        ax.legend(loc='upper right')
        if 'kl' in train_col:
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        # Use wider left offset for panel labels to avoid clashing with wide tick labels
        ax.text(-0.14, 1.05, plabel, transform=ax.transAxes,
                fontsize=16, fontweight='bold', fontfamily='sans-serif',
                verticalalignment='top', horizontalalignment='left')

    # Panel A (KL z1): extend y-axis so lines don't crowd the legend.
    # Use 320 so the top tick is 300, avoiding a "350" label that clashes with panel label A.
    axes[0, 0].set_ylim(0, 320)

    save_figure(fig, "exp3_kl_divergence")


# ═══════════════════════════════════════════════════════════════════════════
# Fig S15: Extrapolation Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_exp7_extrapolation():
    print("[Fig S15] Extrapolation")
    csv = os.path.join(RESULTS_DIR, "exp7_extrapolation.csv")
    df = pd.read_csv(csv)

    seed_lengths = sorted(df['seed_bins'].unique(), reverse=True)
    n_timebins = df['time_bin'].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.3, top=0.90, bottom=0.12)
    for ax, metric, title in [
        (axes[0], 'accuracy', 'Recreation Accuracy'),
        (axes[1], 'cosine_similarity', 'Cosine Similarity')
    ]:
        pivot = df.pivot_table(index='seed_bins', columns='time_bin', values=metric)
        pivot = pivot.reindex(sorted(seed_lengths, reverse=True))

        # Use viridis (perceptually uniform, CVD-safe sequential cmap)
        im = ax.imshow(pivot.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n_timebins))
        ax.set_xticklabels([f't{t}' for t in range(n_timebins)])
        ax.set_yticks(range(len(seed_lengths)))
        ax.set_yticklabels([str(s) for s in sorted(seed_lengths, reverse=True)])
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Seed time bins provided')
        ax.set_title(title, loc='center')
        # Heatmaps need all 4 spines
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        for i in range(len(seed_lengths)):
            for j in range(n_timebins):
                val = pivot.values[i, j]
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)

    panel_label(fig, axes[0], 'A')
    panel_label(fig, axes[1], 'B')
    save_figure(fig, "exp7_extrapolation_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# Fig S18: Z-Score Threshold Sensitivity
# ═══════════════════════════════════════════════════════════════════════════
def plot_exp6_zscore():
    print("[Fig S18] Z-Score Sensitivity")
    csv = os.path.join(RESULTS_DIR, "exp5_zscore_edge_counts.csv")
    ec = pd.read_csv(csv)

    # Load TF recovery data if available
    tf_csv = os.path.join(RESULTS_DIR, "exp9_zscore_tf_recovery_aggregate.csv")
    has_tf = os.path.exists(tf_csv)
    if has_tf:
        tf = pd.read_csv(tf_csv)
        # Filter to Union gold standard, ALL cell types aggregate, K=140
        tf_agg = tf[(tf['gold_standard'] == 'Intersection') & (tf['cell_type'] == 'ALL')]

    n_panels = 3 if has_tf else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.15)

    transitions = sorted(ec['transition'].unique())
    t_colors = {t: OKABE_ITO[i] for i, t in enumerate(transitions)}

    # Panel A: edge counts
    ax = axes[0]
    for name, group in ec.groupby('transition'):
        ax.plot(group['threshold'], group['n_edges'] / 1e6,
                'o-', label=name, color=t_colors[name], linewidth=2, markersize=5)
    ax.set_xlabel('Z-score threshold (|z| >)')
    ax.set_ylabel('Non-zero edges (millions)')
    ax.legend(loc='upper right')

    # Panel B: edges retained %
    ax = axes[1]
    for name, group in ec.groupby('transition'):
        ax.plot(group['threshold'], group['fraction'] * 100,
                'o-', label=name, color=t_colors[name], linewidth=2, markersize=5)
    ax.set_xlabel('Z-score threshold (|z| >)')
    ax.set_ylabel('Edges retained (%)')
    ax.legend(loc='upper right')

    panel_label(fig, axes[0], 'A')
    panel_label(fig, axes[1], 'B')

    # Panel C: Gold-standard TF recovery
    if has_tf:
        ax = axes[2]
        k_colors = {100: OKABE_ITO[0], 200: OKABE_ITO[1], 300: OKABE_ITO[4], 400: OKABE_ITO[5]}
        for k_val in [100, 200, 300, 400]:
            sub = tf_agg[tf_agg['K'] == k_val]
            # Average recall across transitions
            avg = sub.groupby('threshold')['recall'].mean()
            ax.plot(avg.index, avg.values * 100, 'o-', color=k_colors[k_val],
                    label=f'K={k_val}', linewidth=2, markersize=5)
        ax.set_xlabel('Z-score threshold (|z| >)')
        ax.set_ylabel('Gold-standard TF recall (%)')
        ax.set_ylim(0, 7.0)
        ax.set_yticks(range(0, 7))   # no tick at the top edge, where the panel letter sits
        ax.legend(loc='upper right')
        panel_label(fig, axes[2], 'C')

    save_figure(fig, "exp5_zscore_sensitivity")


# ═══════════════════════════════════════════════════════════════════════════
# Fig S16: Noise Sensitivity
# ═══════════════════════════════════════════════════════════════════════════
def plot_exp5_noise():
    print("[Fig S16] Noise Sensitivity")
    sigmas = ['0.0', '0.1', '0.2', '0.4']
    sigma_labels = ['\u03c3=0.0', '\u03c3=0.1', '\u03c3=0.2\n(default)', '\u03c3=0.4']
    n_colors = [OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[5], OKABE_ITO[6]]  # orange, sky, red-orange, pink

    results = {}
    for s in sigmas:
        results[s] = load_eval_result(os.path.join(RESULTS_DIR, f"exp4_noise/sigma_{s}/exp4_{s}_recreation_acc_v2.json"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=0.35, top=0.90, bottom=0.15)

    # Panel A: bar chart with direct value labels (no legend needed)
    ax = axes[0]
    means = [results[s]['mean_acc'] for s in sigmas]
    bars = ax.bar(range(4), means, color=n_colors, width=0.6, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(sigma_labels)
    ax.set_ylabel('Mean recreation accuracy')
    ax.set_ylim(0, 1.08)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)

    # Panel B: line plot
    ax = axes[1]
    markers = ['s', '^', 'o', 'D']
    for s, label, color, marker in zip(sigmas, sigma_labels, n_colors, markers):
        bins_dict = results[s]['acc_per_bin']
        bins = sorted(bins_dict.keys(), key=int)
        accs = [bins_dict[b] for b in bins]
        ax.plot([int(b) for b in bins], accs, f'{marker}-', color=color,
                label=label.replace('\n', ' '), linewidth=2, markersize=5)
    ax.set_xlabel('Time Bin')
    ax.set_ylabel('Recreation accuracy')
    ax.legend(loc='lower right')
    ax.set_ylim(-0.05, 1.05)

    panel_label(fig, axes[0], 'A')
    panel_label(fig, axes[1], 'B')
    save_figure(fig, "exp4_noise_sweep")


# ═══════════════════════════════════════════════════════════════════════════
# Fig S6: Decoy Blueprint
# ═══════════════════════════════════════════════════════════════════════════
def plot_exp2_decoy():
    print("[Fig S6] Decoy Blueprint")
    conditions = ['correct', 'reversed', 'shuffled', 'random']
    labels = ['Correct', 'Reversed', 'Shuffled', 'Random']
    d_colors = [OKABE_ITO[4], OKABE_ITO[1], OKABE_ITO[0], OKABE_ITO[5]]  # blue, sky, orange, red-orange

    results = {}
    for cond in conditions:
        results[cond] = load_eval_result(os.path.join(RESULTS_DIR, f"exp1_decoy/{cond}/exp1_{cond}_recreation_acc_v2.json"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=0.35, top=0.90, bottom=0.15)

    # Panel A: bar chart with direct value labels
    ax = axes[0]
    means = [results[c]['mean_acc'] for c in conditions]
    bars = ax.bar(range(4), means, color=d_colors, width=0.6, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean recreation accuracy')
    ax.set_ylim(0, 1.08)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    # Panel B: line plot (4 series, use legend)
    ax = axes[1]
    for cond, label, color in zip(conditions, labels, d_colors):
        bins_dict = results[cond]['acc_per_bin']
        bins = sorted(bins_dict.keys(), key=int)
        accs = [bins_dict[b] for b in bins]
        ax.plot([int(b) for b in bins], accs, 'o-', color=color, label=label,
                linewidth=2, markersize=5)
    ax.set_xlabel('Time Bin')
    ax.set_ylabel('Recreation accuracy')
    ax.legend(loc='center right')
    ax.set_ylim(-0.05, 1.05)

    panel_label(fig, axes[0], 'A')
    panel_label(fig, axes[1], 'B')

    # Also save summary CSV
    rows = []
    for cond in conditions:
        r = results[cond]
        row = {'condition': cond, 'mean_acc': r['mean_acc']}
        for t, acc in r['acc_per_bin'].items():
            row[f'bin_{t}'] = acc
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "exp1_decoy_summary.csv"), index=False)

    save_figure(fig, "exp1_decoy_blueprint")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Generating supplementary figures (publication style)")
    print("=" * 60)
    plot_exp2_decoy()       # S6
    plot_exp3_random()      # S9
    plot_exp1_ablation()    # S14
    plot_exp7_extrapolation()  # S15
    plot_exp5_noise()       # S16
    plot_exp4_kl()          # S17
    plot_exp6_zscore()      # S18
    print("\nAll 7 supplementary figures generated!")
