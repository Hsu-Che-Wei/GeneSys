#!/usr/bin/env python3
"""
Build Supplementary Table S3: TF prioritization lists of the cell types and tissues.

Each sheet is the candidate list that notebook 13 writes for one cell type or tissue
(<name>_TF_centrality_all_transition_zscore3.csv): TFs with their combined centrality score
(betweenness + out-degree + in-degree) summed over all transitions, per cell type and per
tissue, in the order of notebook 13. The Legends sheet is unchanged from the published table.

The TF annotation file carries one gene name converted to a date by Excel (AT2G03710, SEP4,
stored as "4-Sep"); it is restored from the GeneID here, and TF names are written as text.

Input : experiments/GeneSys/supp_data/notebook13_inputs/genesys/*_TF_centrality_all_transition_zscore3.csv
Output: Tables/Table_S3.xlsx

Usage:
  python experiments/build_table_s3.py                 # from the current notebook 13 lists (run from the project root)
  python experiments/build_table_s3.py --src DIR --out FILE
"""

import os
import argparse
from datetime import datetime

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

SRC = os.path.join('experiments', 'GeneSys', 'supp_data', 'notebook13_inputs', 'genesys')
OUT = os.path.join('Tables', 'Table_S3.xlsx')

# Sheet order of the published table
LISTS = ['Epidermis', 'Epidermis_LRC', 'Ground_Tissue', 'Stele', 'Root_Cap',
         'Atrichoblast', 'Trichoblast', 'Cortex', 'Endodermis', 'Pericycle',
         'Phloem', 'Xylem', 'Procambium', 'LRC', 'Columella']

# Gene names that Excel converted to dates in the TF annotation file
NAME_FIXES = {'AT2G03710': 'SEP4'}

_SCORE = 'Combined centrality score (betweenness, in-degree, and out-degree centrality) across all transitions'
LEGENDS = [
    ('Legends', None),
    ('TF', 'Transcription factor (gene name, or gene identifier where no name is assigned)'),
    ('all', f'{_SCORE} and cell types'),
    (None, None),
    ('atri', f'{_SCORE} for atrichoblast'),
    ('tri', f'{_SCORE} for trichoblast'),
    ('cor', f'{_SCORE} for cortex'),
    ('end', f'{_SCORE} for endodermis'),
    ('per', f'{_SCORE} for pericycle'),
    ('pro', f'{_SCORE} for procambium'),
    ('xyl', f'{_SCORE} for xylem'),
    ('phl', f'{_SCORE} for phloem'),
    ('lrc', f'{_SCORE} for lateral root cap'),
    ('col', f'{_SCORE} for columella'),
    (None, None),
    ('ground', f'{_SCORE} for ground tissue (cortex + endodermis)'),
    ('epi', f'{_SCORE} for epidermis (atrichoblast + trichoblast)'),
    ('stele', f'{_SCORE} for stele (pericycle + procambium + xylem + phloem)'),
    ('epilrc', f'{_SCORE} for epidermis + lateral root cap'),
    ('rc', f'{_SCORE} for root cap (columella + lateral root cap)'),
]

FONT = Font(name='Calibri', size=12)
BOLD = Font(name='Calibri', size=12, bold=True)


def read_list(src, name):
    df = pd.read_csv(os.path.join(src, f'{name}_TF_centrality_all_transition_zscore3.csv'))
    df = df.rename(columns={df.columns[0]: 'TF'})
    for gid, gname in NAME_FIXES.items():
        hit = df['GeneID'] == gid
        df.loc[hit, 'TF'] = gname
        df.loc[hit, 'GeneName'] = gname
    return df


def write_sheet(ws, rows, bold_first_row=True):
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row, start=1):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            cell = ws.cell(row=r, column=c, value=val)
            cell.font = BOLD if (r == 1 and bold_first_row) else FONT
            if isinstance(val, str):
                cell.number_format = '@'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    wb = Workbook()
    ws = wb.active
    ws.title = 'Legends'
    write_sheet(ws, LEGENDS)
    ws.column_dimensions['A'].width = 10.83
    ws.column_dimensions['B'].width = 121.0

    for name in LISTS:
        df = read_list(args.src, name)
        ws = wb.create_sheet(f'{name}_TF_prioritization')
        rows = [list(df.columns)] + [[v.item() if hasattr(v, 'item') else v for v in rec]
                                     for rec in df.itertuples(index=False)]
        write_sheet(ws, rows)
        print(f'{name:15s} {len(df):4d} TFs  top: {", ".join(df["TF"].astype(str).head(5))}')

    # Fixed document dates, so that the file only changes when the lists change
    stamp = datetime(2026, 10, 26)
    wb.properties.created = stamp
    wb.properties.modified = stamp
    wb.save(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
