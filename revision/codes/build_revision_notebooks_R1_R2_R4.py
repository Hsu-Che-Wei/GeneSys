import sys, os
import nbformat as nbf
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks')

def make(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    for kind, src in cells:
        src = src.strip('\n')
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == 'md' else nbf.v4.new_code_cell(src))
    nbf.write(nb, os.path.join(OUT, name))

SETUP = '''
import os, json
import numpy as np
import pandas as pd
from IPython.display import Image, display

R = os.path.join('..', 'results')                           # result files of the revision folder
pct = lambda x: round(100 * float(x), 1)                    # accuracy as a percentage with one decimal

# Figures are drawn by codes/exp24_manuscript_figures.py from the result tables in R
import importlib.util
os.environ['GENESYS_RESULTS'] = os.path.abspath(R)
os.environ['GENESYS_FIGURES'] = os.path.abspath(os.path.join(R, 'figures'))
_spec = importlib.util.spec_from_file_location('figures', os.path.join('..', 'codes', 'exp24_manuscript_figures.py'))
figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figures)

def load_json(*parts):
    with open(os.path.join(R, *parts)) as fh:
        return json.load(fh)
'''
PROTO = '''
## The three metrics used below

* **Classification accuracy.** Real cells are supplied at every time bin and the classifier predicts the trajectory at each bin (`evaluate_model` in `rescue_train_and_eval.py`). Stored in `*_eval_result.json`.
* **Recreation accuracy** (protocol of Figure S5; `eval_recreation_accuracy_v2.py`). To predict time bin *t*, real cells are supplied at every bin before *t* and the remaining positions repeat the last real cell. No transcriptome is generated. Stored in `*_recreation_acc_v2.json`.
* **Autoregressive generation accuracy** (`eval_recreation_accuracy.py`). The input holds real cells at t0, t1 and t2, with t2 repeated in the later positions. The model generates t0, t1 and t2 from this seed, and from t3 onward each bin is generated from a sequence that contains only generated transcriptomes. The classifier is read at every bin from t1 to t10. Stored in `*_recreation_acc.json`.

File naming note: the files named `*_recreation_acc.json` hold the autoregressive generation accuracy, and the files named `*_recreation_acc_v2.json` hold the recreation accuracy.
'''


TRAINING = '''
## How the models of this notebook were trained

Every retrained model goes through the training function of the up-to-date GeneSys package, `run_cycle` in `genesys/genesys_cli.py`, called unchanged by `../codes/train_run_cycle.py`: one cycle of 30 epochs with the package defaults (AdamW, learning rate 1e-3, ReduceLROnPlateau with factor 0.5, patience 10 and threshold 0.05, gradient clipping at 5, best weights kept on lowest loss). Whatever an experiment varies enters through the inputs of `run_cycle` only:

* **labels** in the convention of the package, with time starting at 1 (`utils.load_root_data_blueprint`: the stem cells are time 1 and time bins 1 to 10 are times 2 to 11, as in `toy_data/annotations.txt`);
* **lineage blueprint table**: correct, reversed, shuffled or random. The reversed decoy combines the column-reversed table with time-reversed labels. The random table keeps the stem-cell column and draws one of the ten cell types at every later position;
* **model object**: noise level, width, a feedforward network in place of the LSTM, or the TD-VAE loss switched off. For the No TD-VAE ablation `calculate_loss` is replaced by a constant that is not connected to any parameter, so the steps that `run_cycle` assigns to the TD-VAE loss leave the model unchanged;
* **KL logging** is done by wrappers around `model.calculate_loss` and around the validation loader, not by editing the loop.

`../codes/check_blueprint_sequences.py` traces which cells each position of a sequence reads. With these inputs the correct blueprint reads the stem cell and then time bins 1 to 10, the same sequence as the `Root_Dataset` of the original notebooks, and the reversed decoy reads time bins 10 to 1 and then the stem cell.

All evaluations use the validation split, 10 batches of 512 trajectories.
'''

# ───────────────────────────── notebook R1 ─────────────────────────────
nb16 = [
('md', '''
# R1. Model validation: decoy blueprints, external classifiers and architectural ablations

This notebook reproduces, from saved result files, every number that the manuscript quotes for

* **Experiment 1, decoy blueprint** (Results, "GeneSys reconstructs root developmental trajectories in silico"; Figure S6A and S6B),
* **Experiment 2, external validation of generated transcriptomes** (Figure S6C and S6D),
* **Experiment 3, ablation study** (Figure S6E and S6F).

Training and evaluation were run as cluster jobs with the scripts in `../codes/`. The commands are shown in each section but are not executed here. Each number is followed by an `assert`, so the notebook fails if a result file and the manuscript disagree.

| Claim in the manuscript | Section of this notebook |
|---|---|
| Recreation accuracy 84.4% (correct), 84.1% (reversed), 4.0% (shuffled), 15.1% (random) | 1.2 |
| Autoregressive generation accuracy 61.9%, 35.8%, 8.0%, 12.9%; correct and reversed differ by 26 points | 1.3 |
| Logistic regression reaches 96.8% on real held-out cells | 2.2 |
| Generated transcriptomes of the correct-blueprint model match their blueprint labels at 57.9% and 52.4% under the two external classifiers | 2.2 |
| Corrupted blueprints and ablations fall to between 10% and 15% | 2.3 |
| Reversed blueprint, nearest-centroid accuracy 42.7% to 21.2%; correct blueprint 63.5% to 49.9% | 2.4 |
| Ablation recreation accuracy 85% to 88% for variants trained on the correct blueprint; No LSTM 88% | 3.2 |
| Full model 59.6% (61.9% for the correct-blueprint model of Experiment 1); no TD-VAE 20.4% and no LSTM 17.5% | 3.3 |
| No blueprint: classification 10.1%, recreation 11.9%, generation 12.2%, all at chance | 3.4 |
| 13,918,579 parameters in the full model | 3.5 |
'''),
('code', SETUP),
('md', TRAINING),
('md', PROTO),
('md', '''
## 1. Experiment 1: decoy blueprint

Four models with the identical architecture are trained on the same data and differ only in the lineage blueprint used to assemble the training batches (lineage tables from `exp1_decoy_blueprint.py`, training through `rescue_train_and_eval.py` and `train_run_cycle.py`):

* **correct**: the root blueprint (10 trajectories by 11 time bins);
* **reversed**: every trajectory is read from t10 to t0 (column-reversed table read against time-reversed labels);
* **shuffled**: within each time bin, the cell type labels are permuted across trajectories;
* **random**: every position after the stem cell holds a cell type drawn at random.

All four models are then evaluated against the correct blueprint on the validation split.
'''),
('code', '''
# Commands used on the cluster (not executed here).
# Training, one GPU job per blueprint (30 epochs through genesys_cli.run_cycle):
#   sbatch --job-name=rc_exp1_<blueprint> --gres=gpu:1 experiments/slurm/run_gpu_fast.sh \\
#       experiments/rescue_train_and_eval.py --experiment=exp1 --condition=<correct|reversed|shuffled|random> --epochs 30
# Evaluation of every saved model (experiments/slurm/post_rc.sh):
#   python experiments/eval_recreation_accuracy_v2.py --experiment exp1 --condition <blueprint> --device cpu   # recreation accuracy
#   python experiments/eval_recreation_accuracy.py    --experiment exp1 --condition <blueprint> --device cpu   # autoregressive generation accuracy
#   python experiments/generate_sprint3_figures.py                                                             # summary table
#   python experiments/exp24_manuscript_figures.py                                                             # Figure S6
'''),
('md', '### 1.2 Recreation accuracy (Figure S6A and S6B)'),
('code', '''
conds = ['correct', 'reversed', 'shuffled', 'random']
rec = {c: load_json('exp1_decoy', c, f'exp1_{c}_recreation_acc_v2.json') for c in conds}
recreation = pd.Series({c: pct(rec[c]['mean_acc']) for c in conds}, name='recreation accuracy (%)')
print(recreation.to_string())

assert recreation.to_dict() == {'correct': 84.4, 'reversed': 84.1, 'shuffled': 4.0, 'random': 15.1}

# the summary table of the decoy runs holds the same values
summary = pd.read_csv(os.path.join(R, 'exp1_decoy_summary.csv')).set_index('condition')
assert all(pct(summary.loc[c, 'mean_acc']) == recreation[c] for c in conds)
summary.round(3)
'''),
('md', '### 1.3 Autoregressive generation accuracy (Figure S6A)'),
('code', '''
gen = {c: load_json('exp1_decoy', c, f'exp1_{c}_recreation_acc.json') for c in conds}
generation = pd.Series({c: pct(gen[c]['mean_acc']) for c in conds}, name='generation accuracy (%)')
print(pd.concat([recreation, generation], axis=1).to_string())

assert generation.to_dict() == {'correct': 61.9, 'reversed': 35.8, 'shuffled': 8.0, 'random': 12.9}

# "autoregressive generation separates them by 26 percentage points" (legend of Figure S6)
gap = round(generation['correct'] - generation['reversed'], 1)
print('correct minus reversed:', gap, 'percentage points')
assert round(gap) == 26
# each model is scored on 5,120 trajectories per time bin
assert set(gen['correct']['n_samples_per_bin'].values()) == {5120}
'''),
('md', '''
## 2. Experiment 2: external validation of generated transcriptomes

The accuracies above come from the classifier of the same model that generated the cells. `exp12_external_scoring.py` scores the generated transcriptomes with two classifiers fitted on real cells only:

* a **multinomial logistic regression** (scikit-learn, 1,000 iterations, default L2 penalty) fitted on 20,000 cells drawn at random from the training split, with cell type as the label, and checked on 5,000 real cells of the validation split;
* a **nearest-centroid classifier**: for each cell type and time bin the mean transcriptome of the real training cells is computed, and a generated transcriptome at time bin *t* is assigned to the cell type whose centroid at *t* has the highest cosine similarity to it.

Transcriptomes are generated with the autoregressive protocol described above for the published model, the four decoy models and the four ablation models, and scored against the blueprint label at each of the generated time bins t2 to t10.
'''),
('code', '''
# Command used on the cluster (not executed here):
#   python experiments/exp12_external_scoring.py --device cpu        (part of experiments/slurm/post_rc.sh)
'''),
('md', '### 2.2 The correct-blueprint model under the two external classifiers'),
('code', '''
ext = pd.read_csv(os.path.join(R, 'exp12_external', 'exp12_external_summary.csv')).set_index('model')
scores = pd.read_csv(os.path.join(R, 'exp12_external', 'exp12_external_scores.csv'))

lr_on_real = pct(scores['lr_reference_acc_on_real'].iloc[0])
print('logistic regression on real held-out cells:', lr_on_real, '%')
assert lr_on_real == 96.8 and scores['lr_reference_acc_on_real'].nunique() == 1

correct = ext.loc['exp1_correct']
print('correct-blueprint model: logistic regression', pct(correct['lr_acc']), '% ; nearest centroid', pct(correct['centroid_acc']), '%')
assert pct(correct['lr_acc']) == 57.9 and pct(correct['centroid_acc']) == 52.4
(ext[['lr_acc', 'centroid_acc']] * 100).round(1)
'''),
('md', '### 2.3 Corrupted blueprints and ablations (Figure S6C)'),
('code', '''
collapsed = ['exp1_shuffled', 'exp1_random', 'exp6_no_tdvae', 'exp6_no_lstm', 'exp6_no_blueprint']
tab = (ext.loc[collapsed, ['lr_acc', 'centroid_acc']] * 100).round(1)
print(tab.to_string())
# "fell to between 10% and 15%": every value lies between 9.7% and 14.9%
assert tab.values.min() >= 9.5 and tab.values.max() <= 15

# the reversed-blueprint model lies between the correct-blueprint model and the collapsed models
for col in ['lr_acc', 'centroid_acc']:
    assert ext.loc[collapsed, col].max() < ext.loc['exp1_reversed', col] < ext.loc['exp1_correct', col]
print('reversed blueprint: logistic regression', pct(ext.loc['exp1_reversed', 'lr_acc']), '% ; nearest centroid',
      pct(ext.loc['exp1_reversed', 'centroid_acc']), '%')
'''),
('md', '### 2.4 Nearest-centroid accuracy along the generated trajectory (Figure S6D)'),
('code', '''
per_bin = (scores[scores['model'].isin(['exp1_correct', 'exp1_reversed'])]
           .pivot(index='time_bin', columns='model', values='centroid_acc') * 100).round(1)
print(per_bin.to_string())

assert list(per_bin.index) == [2, 3, 4, 5, 6, 7, 8, 9, 10]        # generated bins that are scored
assert (per_bin['exp1_reversed'].iloc[0], per_bin['exp1_reversed'].iloc[-1]) == (42.7, 21.2)
assert (per_bin['exp1_correct'].iloc[0], per_bin['exp1_correct'].iloc[-1]) == (63.5, 49.9)
'''),
('md', '''
## 3. Experiment 3: ablation study

Four variants are trained from scratch (model classes from `exp6_ablation.py`, training through `rescue_train_and_eval.py` and `train_run_cycle.py`):

* **full**: the standard model (bidirectional LSTM, TD-VAE, correct blueprint), trained with the alternating loss;
* **no_tdvae**: the TD-VAE loss is switched off (`calculate_loss` returns a constant that is not connected to any parameter), so only the classifier loss trains the model;
* **no_lstm**: the bidirectional LSTM is replaced by a feedforward network that processes each time bin independently;
* **no_blueprint**: trained on a blueprint with a random cell type at each position after the stem cell.

Each variant is evaluated with the correct blueprint on the validation split by the three metrics.
'''),
('code', '''
# Commands used on the cluster (not executed here):
#   sbatch --job-name=rc_exp6_<variant> --gres=gpu:1 experiments/slurm/run_gpu_fast.sh \\
#       experiments/rescue_train_and_eval.py --experiment=exp6 --condition=<full|no_tdvae|no_lstm|no_blueprint> --epochs 30
#   python experiments/eval_recreation_accuracy_v2.py --experiment exp6 --condition <variant> --device cpu
#   python experiments/eval_recreation_accuracy.py    --experiment exp6 --condition <variant> --device cpu
'''),
('code', '''
variants = ['full', 'no_tdvae', 'no_lstm', 'no_blueprint']
abl = pd.DataFrame({
    'classification (%)': {v: 100 * load_json('exp6_ablation', v, f'exp6_{v}_eval_result.json')['mean_acc_correct_bp'] for v in variants},
    'recreation (%)':     {v: 100 * load_json('exp6_ablation', v, f'exp6_{v}_recreation_acc_v2.json')['mean_acc'] for v in variants},
    'generation (%)':     {v: 100 * load_json('exp6_ablation', v, f'exp6_{v}_recreation_acc.json')['mean_acc'] for v in variants},
})
abl.round(2)
'''),
('md', '### 3.2 Recreation accuracy does not separate the variants trained on the correct blueprint'),
('code', '''
r = abl.loc[['full', 'no_tdvae', 'no_lstm'], 'recreation (%)']
print(r.round(1).to_string())
assert 84.5 <= r.min() and r.max() <= 88.5          # "between 85% and 88%"
assert round(r['no_lstm']) == 88                    # "even the No LSTM variant ... reached 88%"
'''),
('md', '### 3.3 Autoregressive generation separates them'),
('code', '''
g = abl['generation (%)'].round(1)
print(g.to_string())
assert g['full'] == 59.6
assert (g['no_tdvae'], g['no_lstm']) == (20.4, 17.5)
# the full model and the correct-blueprint model of Experiment 1 are independent runs of the same configuration
print('full model', g['full'], '% ; correct-blueprint model of Experiment 1', generation['correct'], '%')
assert generation['correct'] == 61.9
'''),
('md', '### 3.4 The model trained without the blueprint'),
('code', '''
nb = abl.loc['no_blueprint']
print(nb.round(2).to_string())
assert round(nb['classification (%)'], 1) == 10.1
assert round(nb['recreation (%)'], 1) == 11.9
assert round(nb['generation (%)'], 1) == 12.2
assert (nb.between(9, 13)).all()                      # all three at the chance level of 10%
'''),
('md', '### 3.5 Size of the full model'),
('code', '''
sweep = pd.read_csv(os.path.join(R, 'exp10_latent_sweep_summary.csv')).set_index('dim')
print('parameters of the model at width 256:', f"{int(sweep.loc[256, 'n_params']):,}")
assert int(sweep.loc[256, 'n_params']) == 13918579
'''),
('code', "figures.figure_blueprint_controls()   # Figure S6, drawn from the tables above\ndisplay(Image(os.path.join(R, 'figures', 'figS_blueprint_controls.png'), width=900))"),
]

# ───────────────────────────── notebook R2 ─────────────────────────────
nb17 = [
('md', '''
# R2. Model validation: random-initialization control, extrapolation, noise, model width and KL divergence

This notebook reproduces, from saved result files, every number that the manuscript quotes for

* **Experiment 4, random-initialization control** for the cell type recovery rate (Figure S7),
* **Experiment 8, extrapolation with zero-padding** (Figure S5E and S5F),
* **Experiment 9, noise sensitivity** (Figure S14A and S14B),
* **Experiment 10, latent-dimension sensitivity** (Figure S14C to S14E),
* **Experiment 11, KL-divergence monitoring** (Figure S14F to S14I).

The scripts are in `../codes/`. The cluster commands are shown but not executed. Each number is followed by an `assert`.

| Claim in the manuscript | Section |
|---|---|
| An untrained model scores exactly 0.10 with all genes masked | 1.2 |
| With genes unmasked it stays around chance: mean 0.09, individual evaluations 0.03 to 0.16 | 1.2 |
| The trained model reaches a recovery rate of 0.999 with 10% of the genes (1,751 of 17,513) | 1.2 |
| Six different default cell types over ten untrained seeds, none of them trichoblast | 1.3 |
| Trained model under fully masked input: all trichoblast; belief state cosine similarity 0.85 (trichoblast), 0.76 (atrichoblast), all others below 0.6 | 1.4 |
| Zero-padding: 78.1% with 7 seed bins, 21.5% with 3, chance with 0 to 1 | 2.2 |
| Cosine similarity of generated to real transcriptomes | 2.3 |
| Noise levels 0.0 to 0.4 all give 82.6% to 86.2% recreation accuracy | 3 |
| Generation accuracy rises with width and levels off from 256 (24.4%, 37.7%, 61.0%, 63.6%); KL(z1) is about 0.5 nats per latent dimension at every width | 4 |
| KL(z1) stays far from zero during 30 epochs of training from scratch | 5 |
'''),
('code', SETUP),
('md', TRAINING),
('md', '''
## 1. Experiment 4: random-initialization control (Figure S7)

`exp25_random_baseline_balanced.py` evaluates the trained model and a randomly initialized model of the identical architecture with the protocol of Figure 3. Test batches are assembled from the lineage blueprint with 200 trajectories for each of the ten cell types (2,000 per batch). For each gene fraction in {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1} a random subset of genes is kept and the rest set to zero across all time bins, and the terminal fate is read from the classifier at t10. Ten random gene subsets are evaluated per fraction. Because every cell type contributes the same number of trajectories, a model that always returns one cell type scores exactly 1/10.

The belief-state comparison comes from `exp11_trichoblast_forensics.py`: the belief state of the trained model under an all-zero input is compared, by cosine similarity, with the mean belief state of 200 real cells of each type.
'''),
('code', '''
# Commands used on the cluster (not executed here):
#   sbatch --job-name=exp25 experiments/slurm/run_cpu.sh experiments/exp25_random_baseline_balanced.py
#   sbatch --job-name=exp11 experiments/slurm/run_cpu.sh experiments/exp11_trichoblast_forensics.py
'''),
('md', '### 1.2 Recovery rate of the trained and the untrained model'),
('code', '''
E25 = 'exp25_random_baseline_balanced'
rec = pd.read_csv(os.path.join(R, E25, 'exp25_recovery.csv'))
s25 = load_json(E25, 'exp25_summary.json')
table = rec.groupby(['model', 'fraction'])['accuracy'].agg(['mean', 'std', 'count']).round(4).unstack(0)
print(table.to_string())

assert set(rec.groupby(['model', 'fraction']).size()) == {10}            # ten gene subsets per fraction
rnd = rec[rec['model'] == 'random']
trn = rec[rec['model'] == 'trained']

# untrained model, all genes masked: exactly 0.10 in every evaluation
assert (rnd[rnd['fraction'] == 0]['accuracy'] == 0.1).all()
# untrained model over all gene fractions: mean 0.09, individual evaluations 0.03 to 0.16
print('untrained model: mean', round(rnd['accuracy'].mean(), 2), ', min', round(rnd['accuracy'].min(), 2), ', max', round(rnd['accuracy'].max(), 2))
assert round(rnd['accuracy'].mean(), 2) == 0.09
assert (round(rnd['accuracy'].min(), 2), round(rnd['accuracy'].max(), 2)) == (0.03, 0.16)
# trained model with 10% of the genes
t10 = trn[trn['fraction'] == 0.10]
print('trained model, 10% of genes:', round(t10['accuracy'].mean(), 3), 'with', int(t10['n_genes_kept'].iloc[0]), 'genes')
assert round(t10['accuracy'].mean(), 3) == 0.999 == s25['trained_10pct']
assert int(t10['n_genes_kept'].iloc[0]) == 1751
# trained model with all genes masked also scores 0.10 (it returns one cell type for every trajectory)
assert (trn[trn['fraction'] == 0]['accuracy'] == 0.1).all()
'''),
('md', '### 1.3 The default cell type of an untrained model depends on the initialization'),
('code', '''
seeds = pd.read_csv(os.path.join(R, E25, 'exp25_seed_scores.csv'))
masked = seeds[seeds['fraction'] == 0]
print(masked.to_string(index=False))

assert len(masked) == 10 and (masked['accuracy'] == 0.1).all() and (masked['dominant_share'] == 1.0).all()
n_types = masked['dominant_class'].nunique()
print('different default cell types over ten seeds:', n_types)
assert n_types == 6
assert 'Trichoblast' not in set(masked['dominant_class'])
'''),
('md', '### 1.4 The trichoblast default of the trained model'),
('code', '''
assert s25['trained_zero_input_dominant_class'] == 'Trichoblast' and s25['trained_zero_input_dominant_share'] == 1.0

bel = pd.read_csv(os.path.join(R, 'exp11_trichoblast', 'exp11_trained_belief_distances.csv'))
bel = bel.sort_values('cosine_sim_zero_to_mean_belief', ascending=False)
print(bel[['terminal_cell_type', 'n_cells', 'cosine_sim_zero_to_mean_belief']].round(2).to_string(index=False))

cos = bel.set_index('terminal_cell_type')['cosine_sim_zero_to_mean_belief']
assert round(cos['Trichoblast'], 2) == 0.85 and round(cos['Atrichoblast'], 2) == 0.76
assert cos.drop(['Trichoblast', 'Atrichoblast']).max() < 0.6
'''),
('code', "figures.figure_random_baseline()   # Figure S7, drawn from the tables above\ndisplay(Image(os.path.join(R, 'figures', 'figS_random_baseline.png'), width=900))"),
('md', '''
## 2. Experiment 8: extrapolation with zero-padding (Figure S5E and S5F)

`exp7_extrapolation.py` uses the published model, with sequences built in the convention described above (stem cell at t0, then time bins 1 to 10). For each seed length *k* in {0, 1, 2, 3, 5, 7, 11} only the first *k* time bins of each input sequence hold real cells and the remaining positions are filled with zeros. Two quantities are recorded at every time bin: the accuracy of the classifier, and the cosine similarity between the transcriptome that the TD-VAE generates for that bin and the real transcriptome. The evaluation uses up to 128 validation cells for each of the ten trajectories, 1,280 sequences per seed length.
'''),
('code', '''
# Command used on the cluster (not executed here):
#   python experiments/exp7_extrapolation.py --device cpu        (part of experiments/slurm/post_rc.sh)
'''),
('md', '### 2.2 Accuracy at the zero-filled time bins'),
('code', '''
ex = pd.read_csv(os.path.join(R, 'exp7_extrapolation.csv'))
assert sorted(ex['seed_bins'].unique()) == [0, 1, 2, 3, 5, 7, 11] and set(ex['n_samples']) == {1280}
print((ex.pivot(index='seed_bins', columns='time_bin', values='accuracy') * 100).round(1).to_string())

unseen = ex[~ex['is_seeded']].groupby('seed_bins')['accuracy'].mean()
print('\\nmean accuracy over the zero-filled bins (%):')
print((unseen * 100).round(1).to_string())
assert pct(unseen[7]) == 78.1 and pct(unseen[3]) == 21.5
assert pct(unseen[0]) == 10.0 and pct(unseen[1]) == 10.0          # chance level with 0 or 1 seed bins
'''),
('md', '### 2.3 Cosine similarity between generated and real transcriptomes'),
('code', '''
cosm = ex.pivot(index='seed_bins', columns='time_bin', values='cosine_similarity').round(2)
print(cosm.to_string())

# With 7 seed bins the similarity is 0.38 to 0.61 across the seeded bins (t0 to t6) and 0.38 at the first zero-filled bin (t7).
assert (cosm.loc[7, 0:6].min(), cosm.loc[7, 0:6].max()) == (0.38, 0.61) and cosm.loc[7, 7] == 0.38
assert cosm.loc[7, 8:10].max() <= 0.21                                  # about 0.2 thereafter
# With 3 seed bins it is 0.37 to 0.49 across the seeded bins (t0 to t2) and 0.21 at the first zero-filled bin (t3).
assert (cosm.loc[3, 0:2].min(), cosm.loc[3, 0:2].max()) == (0.37, 0.49) and cosm.loc[3, 3] == 0.21
# The similarity at the zero-filled bins falls as fewer bins are seeded.
unseen_cos = ex[~ex['is_seeded']].groupby('seed_bins')['cosine_similarity'].mean()
print('\\nmean cosine similarity over the zero-filled bins:'); print(unseen_cos.round(2).to_string())
assert unseen_cos[7] > unseen_cos[5] > unseen_cos[3]
'''),
('code', "figures.figure_extrapolation_panels()   # Figure S5E and S5F, drawn from the tables above\ndisplay(Image(os.path.join(R, 'figures', 'figS5EF_extrapolation.png'), width=900))"),
('md', '''
## 3. Experiment 9: noise sensitivity (Figure S14A and S14B)

Four models are trained through `rescue_train_and_eval.py` and `train_run_cycle.py` that differ in the standard deviation of the Gaussian noise applied to the embedding, 0.0, 0.1, 0.2 (the default) and 0.4. The noise is relative: for each embedding dimension it is drawn with a standard deviation equal to sigma times the absolute value of that dimension, so a zero-valued dimension receives no noise. The models are compared by recreation accuracy on the validation split.
'''),
('code', '''
# Commands used on the cluster (not executed here):
#   sbatch --job-name=rc_exp4_<sigma> --gres=gpu:1 experiments/slurm/run_gpu_fast.sh \\
#       experiments/rescue_train_and_eval.py --experiment=exp4 --condition=<0.0|0.1|0.2|0.4> --epochs 30
#   python experiments/eval_recreation_accuracy_v2.py --experiment exp4 --condition <sigma> --device cpu
'''),
('code', '''
sigmas = ['0.0', '0.1', '0.2', '0.4']
noise = pd.Series({s: pct(load_json('exp4_noise', f'sigma_{s}', f'exp4_{s}_recreation_acc_v2.json')['mean_acc']) for s in sigmas},
                  name='recreation accuracy (%)')
print(noise.to_string())
assert (noise.min(), noise.max()) == (82.6, 86.2)             # "all conditions achieve 82.6 to 86.2%"
assert noise.max() - noise.min() < 4                           # "varying by less than four percentage points"
'''),
('md', '''
## 4. Experiment 10: latent-dimension sensitivity (Figure S14C to S14E)

`exp10_latent_sweep.py` trains the model from scratch for 30 epochs (through `train_run_cycle.py`) at four widths, with embedding and hidden dimension 64, 128, 256 and 512. The TD-VAE latent state has twice the embedding width, 128 to 1,024 dimensions. For each width the script records the KL terms at every epoch, the accuracy of the classifier, and the autoregressive generation accuracy.

In this script the column named `recreation_mean_acc` is computed by `evaluate_trained_model` with real cells supplied at every time bin. It is therefore the classification accuracy defined in notebook R1, not the partially seeded recreation accuracy of Figure S5, which is why it is close to 100% here and close to 85% in the other experiments.
'''),
('code', '''
# Commands used on the cluster (not executed here), one GPU job per width, then the aggregation:
#   sbatch --job-name=rc_exp10_64 --gres=gpu:1 experiments/slurm/run_gpu_fast.sh experiments/exp10_latent_sweep.py --dim 64 --epochs 30    (and 128, 256, 512)
#   python experiments/exp10_latent_sweep.py --aggregate
'''),
('code', '''
sw = pd.read_csv(os.path.join(R, 'exp10_latent_sweep_summary.csv')).set_index('dim')
sw['generation (%)'] = (100 * sw['autoregressive_mean_acc']).round(1)
sw['accuracy with real cells at every bin (%)'] = (100 * sw['recreation_mean_acc']).round(2)
sw['KL(z1) per latent dimension (nats)'] = (sw['final_kl_z1'] / sw['latent_dim']).round(2)
print(sw[['latent_dim', 'n_params', 'accuracy with real cells at every bin (%)', 'generation (%)', 'final_kl_z1',
          'KL(z1) per latent dimension (nats)']].round(1).to_string())

assert list(sw['latent_dim']) == [128, 256, 512, 1024]
assert list(sw['generation (%)']) == [24.4, 37.7, 61.0, 63.6]                     # rises with width and levels off from 256
assert sw['generation (%)'].is_monotonic_increasing and sw.loc[512, 'generation (%)'] - sw.loc[256, 'generation (%)'] < 3
assert (sw['recreation_mean_acc'] > 0.999).all()                                  # insensitive to width
assert (sw['KL(z1) per latent dimension (nats)'].between(0.49, 0.52)).all()       # about 0.5 nats per dimension
assert [round(v, 1) for v in sw['final_kl_z1']] == [64.0, 130.1, 259.3, 519.6]

kl = pd.read_csv(os.path.join(R, 'exp10_latent_sweep_kl_logs.csv'))
assert set(kl.groupby('dim')['epoch'].nunique()) == {30}                          # 30 epochs at every width
print('\\nKL(z1) over the 30 epochs, minimum and maximum per width:')
print(kl.groupby('dim')['kl_z1'].agg(['min', 'max']).round(1).to_string())
'''),
('md', '''
## 5. Experiment 11: KL-divergence monitoring (Figure S14F to S14I)

`exp3_kl_divergence_from_scratch.py` trains one model from scratch for 30 epochs (through `train_run_cycle.py`) and logs, on the training and the validation split, the four loss components at each epoch: KL(z1) between the inference posterior and the belief prior, KL(z2) between the belief-derived state and the transition prior, the reconstruction loss (binary cross-entropy) and the classifier loss.
'''),
('code', '''
# Command used on the cluster (not executed here):
#   sbatch --job-name=rc_exp3 --gres=gpu:1 experiments/slurm/run_gpu_fast.sh experiments/exp3_kl_divergence_from_scratch.py --epochs 30
'''),
('code', '''
kld = pd.read_csv(os.path.join(R, 'exp3_kl_divergence_diagnostic.csv'))
assert len(kld) == 30
print(kld[['epoch', 'train_kl_z1', 'val_kl_z1', 'train_kl_z2', 'val_kl_z2', 'train_recon', 'train_nll']].iloc[[0, 1, 9, 19, 29]].round(2).to_string(index=False))

# KL(z1) remains far from zero throughout training: it never falls below 256 nats (512 latent dimensions, about 0.5 nats each)
print('\\nminimum KL(z1): training', round(kld['train_kl_z1'].min(), 1), ', validation', round(kld['val_kl_z1'].min(), 1))
assert kld['train_kl_z1'].min() > 250 and kld['val_kl_z1'].min() > 250
'''),
('code', "figures.figure_model_sensitivity()   # Figure S14, drawn from the tables above\ndisplay(Image(os.path.join(R, 'figures', 'figS_model_sensitivity.png'), width=900))"),
]

# ───────────────────────────── notebook R4 ─────────────────────────────
nb19 = [
('md', '''
# R4. LIMA identifiability and z-score threshold sensitivity

This notebook reproduces, from saved result files, every number that the manuscript quotes for

* **Experiment 12, LIMA identifiability and regularization** (Materials and Methods, "LIMA construction and application for root development"; Figure S15A to S15C),
* **Experiment 13, z-score threshold sensitivity** (Figure S15D to S15F).

The scripts are in `../codes/`. The cluster commands are shown but not executed. Each number is followed by an `assert`.

| Claim in the manuscript | Section |
|---|---|
| The least-squares solver of the pipeline retains 4 to 32 of the 200 singular directions (median 11) | 1.1 |
| The TF ranking of the minimum-norm and ridge solutions (lambda = 0.01 to 10) correlates at 0.81 to 0.92 with that of the pipeline solution | 1.2 |
| 21% to 36% of the thresholded TF edges are shared with the pipeline solution | 1.2 |
| Changing the number of generated rows (50 to 800) or the latent sampling spread (0.5 to 2 times) changes split-half stability by no more than 0.02 | 1.3 |
| R50 of the TF degree ranking is 328 to 375 under every solver, against a permutation baseline of 741 | 1.4 |
| At a threshold of 3, 0.9 to 1.3% of the possible edges are retained | 2.2 |
| Recall of the gold standard is stable across thresholds, 0.7 to 5.0% for K = 100 to 400 | 2.3 |
'''),
('code', SETUP),
('md', '''
## 1. Experiment 12: LIMA identifiability and regularization (Figure S15A to S15C)

A LIMA estimates a 17,513 by 17,513 matrix from about 200 rows, so the least-squares problem is underdetermined. `exp16_lima_identifiability.py` tests how much the TF ranking depends on the solver. It uses the published model and the transition t1 to t3, and recomputes the LIMA

* with three kinds of **solver** (200 generated rows per state): the call of the LIMA pipeline (`torch.linalg.lstsq` on CPU tensors, as in notebook 12, which discards the singular directions below its default rank cutoff), the **minimum-norm** least-squares solution with every singular direction kept, and **ridge regression** with lambda = 0.01, 0.1, 1 and 10,
* with **50, 200 and 800 generated rows** per state (pipeline solver),
* with the **latent sampling spread** of the TD-VAE scaled by 0.5, 1 and 2 (pipeline solver).

The generated states follow notebook 12 (test split, batches of 2,000 trajectories, `generate_next` at the slots of the transition), with 11 batches per condition as in the notebook. The minimum-norm and ridge solutions are obtained through the singular value decomposition. The replicate matrices are averaged, the averaged matrix is thresholded at |z| > 3 (z-score over all entries of the matrix), and the TFs are ranked by the degree centrality of the TF network of notebook 12 (TF-by-TF subnetwork, TFs with at least one edge in each direction, NetworkX degree centrality, min-max normalized per cell type and summed over the ten cell types). Reported for each condition: the Spearman correlation of this TF ranking with that of the pipeline solution, the fraction of thresholded TF edges shared with the pipeline solution (Jaccard index, mean over the ten cell types), the split-half stability (Spearman correlation between the TF rankings obtained from the first five and the last six replicates), and the R50 of the ranking among the 1,495 TFs (ties broken at random, mean of 200 draws).
'''),
('code', '''
# Commands used on the cluster (not executed here): one CPU job per condition, then a merge job
#   for i in 0 1 2 3 4 5 6 7 8 9; do
#     sbatch --job-name=e16c_$i --time=06:00:00 experiments/slurm/run_cpu.sh experiments/exp16_lima_identifiability.py --only $i --device cpu
#   done
#   sbatch --job-name=e16c_merge --dependency=afterok:<job ids> experiments/slurm/run_cpu.sh experiments/exp16_lima_identifiability.py --merge
'''),
('md', '### 1.1 Singular directions retained by the pipeline solver'),
('code', '''
I = 'exp16_identifiability'
agree = pd.read_csv(os.path.join(R, I, 'exp16_ranking_agreement.csv')).set_index('label')
cond = pd.read_csv(os.path.join(R, I, 'exp16_conditions.csv')).set_index('label')
ranks = pd.read_csv(os.path.join(R, I, 'exp16_ranks.csv'))

BASE = 'ridge_rows200_eps1.0_lam0'                                   # the pipeline solution
MINNORM = 'minnorm_rows200_eps1.0_lamminnorm'
RIDGE = [f'ridge_rows200_eps1.0_lam{l}' for l in ['0.01', '0.1', '1.0', '10.0']]

rk = ranks[ranks['label'] == BASE]
assert len(rk) == 110 and rk['cell_type'].nunique() == 10            # 11 batches for each of the ten cell types
print('rows per fit:', rk['rows'].min(), 'to', rk['rows'].max())
print('directions retained by torch.linalg.lstsq: min', rk['rank'].min(), ', median', rk['rank'].median(), ', max', rk['rank'].max())
assert (rk['rank'].min(), rk['rank'].median(), rk['rank'].max()) == (4, 11, 32)
# the minimum-norm and ridge solutions keep every direction
full = ranks[ranks['label'].isin([MINNORM] + RIDGE)]
assert (full['rank'] == full['rows']).all()
'''),
('md', '''
On CPU tensors `torch.linalg.lstsq` treats the directions below its default cutoff as zero, so the solution of the pipeline uses only the dominant directions of variation among the sampled cells. This acts as an implicit regularization.
'''),
('md', '### 1.2 Agreement of the minimum-norm and ridge solutions with the pipeline solution (Figure S15A)'),
('code', '''
alt = [MINNORM] + RIDGE
tab = agree.loc[alt, ['spearman_vs_lstsq_baseline', 'edge_jaccard_vs_baseline_mean']]
tab.columns = ['Spearman vs pipeline', 'shared thresholded TF edges']
print(tab.round(3).to_string())

rho = tab['Spearman vs pipeline']
assert [round(v, 3) for v in rho] == [0.814, 0.823, 0.843, 0.890, 0.916]
assert (round(rho.min(), 2), round(rho.max(), 2)) == (0.81, 0.92)
assert rho[RIDGE].is_monotonic_increasing                            # agreement increases with the penalty
jac = tab['shared thresholded TF edges']
assert (round(100 * jac.min()), round(100 * jac.max())) == (21, 36)

half = cond.loc[BASE, 'splithalf_spearman_tf']
print('\\nsplit-half stability of the pipeline solution:', round(half, 3))
assert round(half, 2) == 0.94
'''),
('md', '''
The TF ranking of the minimum-norm solution and of the ridge solutions over a thousandfold range of the penalty correlates at 0.81 to 0.92 with the ranking of the pipeline solution. The agreement is lowest for the solutions that keep the weak directions and increases with the penalty. Two independent halves of the pipeline replicates correlate at 0.94. The individual edges are less stable than the ranking: 21% to 36% of the thresholded TF edges are shared with the pipeline solution.
'''),
('md', '### 1.3 Split-half stability across sample sizes and sampling spreads (Figure S15B)'),
('code', '''
base = cond.loc[BASE]
others = ['rows_rows50_eps1.0_lam0', 'rows_rows800_eps1.0_lam0', 'spread_rows200_eps0.5_lam0', 'spread_rows200_eps2.0_lam0']
stab = cond.loc[[BASE] + others, ['n_rows', 'eps_scale', 'splithalf_spearman_tf', 'splithalf_spearman_all']]
print(stab.round(3).to_string())

assert [round(v, 3) for v in stab['splithalf_spearman_tf']] == [0.943, 0.936, 0.960, 0.950, 0.941]
change = (cond.loc[others, 'splithalf_spearman_tf'] - base['splithalf_spearman_tf']).abs()
print('\\nlargest change from the default (200 rows, spread 1):', round(change.max(), 3))
assert round(change.max(), 3) == 0.017 and change.max() <= 0.02
# split-half stability of the minimum-norm and ridge solutions
rs = cond.loc[[MINNORM] + RIDGE, 'splithalf_spearman_tf']
print(rs.round(3).to_string())
assert rs.min() > 0.95
'''),
('md', '### 1.4 R50 of the TF degree ranking in each condition (Figure S15C)'),
('code', '''
r50 = cond[['condition', 'n_rows', 'eps_scale', 'lambda', 'r50_tf_degree', 'r50_permutation']].copy()
print(r50.round(0).to_string())

perm = cond['r50_permutation'].iloc[0]
assert round(perm) == 741 and cond['r50_permutation'].nunique() == 1
solvers = cond.loc[[BASE, MINNORM] + RIDGE, 'r50_tf_degree'].round().astype(int)
assert list(solvers) == [330, 374, 375, 359, 347, 328]
assert (solvers.min(), solvers.max()) == (328, 375) and solvers.max() < perm      # "R50 of 328 to 375 against 741"

rows_spread = cond.loc[['rows_rows50_eps1.0_lam0', 'rows_rows800_eps1.0_lam0',
                        'spread_rows200_eps0.5_lam0', 'spread_rows200_eps2.0_lam0'], 'r50_tf_degree'].round().astype(int)
assert list(rows_spread) == [330, 349, 356, 318]
assert cond['r50_tf_degree'].max() < 0.52 * perm                     # every condition recovers the gold standard well ahead of chance
'''),
('md', '''
The gold-standard TFs are recovered well ahead of the permutation baseline under every solver, sample size and spread. The R50 falls from 356 to 318 as the latent sampling spread goes from 0.5 to 2 times.
'''),
('md', '''
## 2. Experiment 13: z-score threshold sensitivity (Figure S15D to S15F)

The published LIMAs (one 17,513 by 17,513 matrix per cell type and transition) are thresholded again at |z| > 2.0, 2.5, 3.0 and 4.0. The z-score of an entry is taken against the mean and standard deviation of all entries of its matrix, as in notebook 12. `exp5_zscore_sensitivity.py` counts the retained edges, and `exp9_zscore_tf_recovery.py` ranks all 17,513 genes by degree (sum of the absolute weights of the retained in-coming and out-going edges, summed over the ten cell types) and measures the recall of the gold-standard TFs among the top K genes.
'''),
('code', '''
# Commands used on the cluster (not executed here):
#   sbatch --job-name=z_exp5 --cpus-per-task=8 experiments/slurm/run_cpu.sh experiments/exp5_zscore_sensitivity.py
#   sbatch --job-name=z_exp9 --cpus-per-task=8 experiments/slurm/run_cpu.sh experiments/exp9_zscore_tf_recovery.py
'''),
('md', '### 2.2 Fraction of edges retained (Figure S15D and S15E)'),
('code', '''
edges = pd.read_csv(os.path.join(R, 'exp5_zscore_edge_counts.csv'))
frac = (edges.pivot(index='transition', columns='threshold', values='fraction') * 100).round(2)
print(frac.to_string())

at3 = edges[edges['threshold'] == 3.0]['n_edges'] / edges[edges['threshold'] == 3.0]['n_possible'] * 100
print('retained at |z| > 3 (%):', [round(v, 2) for v in at3])
assert (round(at3.min(), 1), round(at3.max(), 1)) == (0.9, 1.3)      # "0.9 to 1.3% of edges are retained"
assert (frac.diff(axis=1).iloc[:, 1:] < 0).all().all()               # retention falls smoothly with the threshold

# The denominator: the ten cell-type matrices of a transition together hold 10 x 17,513^2 entries.
n_possible = int(edges['n_possible'].iloc[0])
print('\\npossible edges per transition:', f'{n_possible:,}', '= 10 x 17,513^2 =', f'{10 * 17513**2:,}')
assert n_possible == 10 * 17513 ** 2 and edges['n_possible'].nunique() == 1
'''),
('md', '''
The number of possible edges per transition is 10 x 17,513² = 3.07 x 10⁹, because the count runs over the ten cell-type matrices of a transition. One matrix holds 17,513² = 3.07 x 10⁸ entries.
'''),
('md', '### 2.3 Recall of the gold standard at each threshold (Figure S15F)'),
('code', '''
agg = pd.read_csv(os.path.join(R, 'exp9_zscore_tf_recovery_aggregate.csv'))
gold = agg[agg['gold_standard'] == 'Intersection']                   # the 140 TFs shared by MINI-EX and StringDB
assert set(gold['n_gold_total']) == {140} and gold['transition'].nunique() == 5

# mean over the five transitions
recall = (gold.groupby(['threshold', 'K'])['recall'].mean().unstack() * 100).round(1)
print(recall.to_string())

used = recall.loc[[2.0, 2.5, 3.0, 4.0]]                              # the thresholds named in the Methods
assert (used.values.min(), used.values.max()) == (0.7, 5.0)          # "0.7 to 5.0% at K = 100 to 400"
assert (used.max() - used.min()).max() <= 0.8                        # stable across thresholds, within one percentage point
print('\\nexpected by chance at K = 100, 200, 300, 400 (%):', [round(100 * k / 17513, 1) for k in (100, 200, 300, 400)])
'''),
('code', "figures.figure_lima_sensitivity()   # Figure S15, drawn from the tables above\ndisplay(Image(os.path.join(R, 'figures', 'figS_lima_sensitivity.png'), width=900))"),
]

make('R1-GeneSys_model_validation_blueprint_external_classifiers_ablation.ipynb', nb16)
make('R2-GeneSys_model_validation_random_init_extrapolation_noise_width_KL.ipynb', nb17)
make('R4-GeneSys_LIMA_identifiability_and_zscore_threshold.ipynb', nb19)
print('written')
