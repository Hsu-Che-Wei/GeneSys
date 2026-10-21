#!/usr/bin/env python3
"""New Illustrator files for Figure 4, Figure 6 and Figure S15 with the out-degree and in-degree
labels of the LIMA centralities exchanged (out-degree counts the targets of a TF, in-degree its
regulators; see regenerate_notebook12.py).

The files of record before this change are in Figures/_before_degree_fix_20261002 and every run
starts from them. An Illustrator file holds the artwork twice, as PDF content and as private
Illustrator data. Only the PDF content can be edited here, so the private data are removed from
the new files and Illustrator opens them through their PDF content.

  Figure 6 (GeneSys_Fig5_2025.ai), Figure S15B (GeneSys_Fig5_sup_2025.ai): the labels are text.
      The two words are removed and written again in the other place, in Liberation Sans, which
      has the metrics of the Arial of the figure.
  Figure 4B (GeneSys_Fig4_2025.ai): the row labels are part of the plot image. The label strips
      of the LIMA out and in rows are exchanged within the image, for generated and for scRNA-seq
      data (each pair has the same row shading). CellOracle rows are not touched.
"""
import os, json
import numpy as np
import fitz

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, 'Figures')
OLD = os.path.join(FIG, '_before_degree_fix_20261002')
FONT = '/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf'


def rgb(c):
    return ((c >> 16) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255)


def spans(page, texts):
    out = []
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                if s['text'].strip() in texts:
                    out.append(s)
    return out


def relabel_text(doc, mapping, centred):
    page = doc[0]
    found = spans(page, set(mapping))
    assert found and len(found) % 2 == 0, [s['text'] for s in found]
    for s in found:
        page.add_redact_annot(fitz.Rect(s['bbox']), fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
    font = fitz.Font(fontfile=FONT)
    done = []
    for s in found:
        new = mapping[s['text'].strip()]
        x, y = s['origin']
        if centred:
            cx = (s['bbox'][0] + s['bbox'][2]) / 2
            x = cx - font.text_length(new, fontsize=s['size']) / 2
        page.insert_text((x, y), new, fontsize=s['size'], fontname='LiberationSans', fontfile=FONT,
                         color=rgb(s['color']))
        done.append({'old': s['text'], 'new': new, 'at': [round(x, 1), round(y, 1)]})
    return done


def label_strips(arr):
    """Label strips of the 25 rows of the plot in an RGB image: (x_left, x_right, [y0], pitch)."""
    dark = arr.max(axis=2) < 90
    axis_x = int(np.argmax(dark.sum(axis=0)))
    ys = np.flatnonzero(dark[:, axis_x])
    top, bottom = ys.min(), ys.max()
    x_right = axis_x - max(3, int(0.006 * arr.shape[1]))
    text = dark[top:bottom, :x_right]
    rows = text.any(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[0, rows.astype(int), 0]))
    bands = [(top + a, top + b) for a, b in zip(edges[::2], edges[1::2]) if b - a > 0.006 * arr.shape[0]]
    if len(bands) != 25:
        return None
    centres = np.array([(a + b) / 2 for a, b in bands])
    pitch = int(np.median(np.diff(centres)))
    x_left = max(0, int(np.flatnonzero(text.any(axis=0)).min()) - 4)
    return x_left, x_right, [int(round(c - pitch / 2)) for c in centres], pitch


def relabel_figure4(doc):
    page = doc[0]
    for img in page.get_images(full=True):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
        strips = label_strips(arr)
        if strips is None:
            continue
        x0, x1, y0s, pitch = strips
        # rows from the top (0 = DE myAUC rank, GeneSys): 3 and 9 are LIMA out and in on generated
        # data, 4 and 15 are LIMA out and in on scRNA-seq data
        for a, b in ((3, 9), (4, 15)):
            ya, yb = y0s[a], y0s[b]
            tmp = arr[ya:ya + pitch, x0:x1].copy()
            arr[ya:ya + pitch, x0:x1] = arr[yb:yb + pitch, x0:x1]
            arr[yb:yb + pitch, x0:x1] = tmp
        new = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, arr.tobytes(), False)
        page.replace_image(xref, pixmap=new)
        return [{'image_xref': xref, 'size': [pix.width, pix.height], 'rows_exchanged': [[3, 9], [4, 15]]}]
    raise SystemExit('panel B image not found')


def drop_private_data(doc):
    """Illustrator reads /PieceInfo (its own copy of the artwork) in preference to the page."""
    page = doc[0]
    doc.xref_set_key(page.xref, 'PieceInfo', 'null')


def main():
    report = {}
    jobs = {
        'GeneSys_Fig4_2025.ai': lambda d: relabel_figure4(d),
        'GeneSys_Fig5_2025.ai': lambda d: relabel_text(d, {'out degree': 'in degree', 'in degree': 'out degree'}, True),
        'GeneSys_Fig5_sup_2025.ai': lambda d: relabel_text(d, {'out-degree centrality': 'in-degree centrality',
                                                               'in-degree centrality': 'out-degree centrality'}, False),
    }
    for name, fn in jobs.items():
        doc = fitz.open(os.path.join(OLD, name), filetype='pdf')
        report[name] = fn(doc)
        drop_private_data(doc)
        out = os.path.join(FIG, name)
        doc.save(out + '.tmp', garbage=4, deflate=True)
        os.replace(out + '.tmp', out)
        print(name, json.dumps(report[name]), f'{os.path.getsize(out) / 1e6:.1f} MB (was {os.path.getsize(os.path.join(OLD, name)) / 1e6:.1f})')
    json.dump(report, open(os.path.join(OLD, 'changes.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
