#!/usr/bin/env python3
"""
Experiment 41: supplementary figures that join Illustrator panels and new panels.

The supplementary figures are grouped by content, so two of them combine artwork of the
round-2 manuscript with other panels:

  figS5_recreation_composite.png   Figure S5. Panels A-D are the Illustrator figure of
                                   recreation accuracy and time-bin annotations
                                   (Figures/highres/image11.png, re-rendered by exp28);
                                   panels E-F are the zero-padded recreation panels drawn
                                   by exp24 (figS5EF_extrapolation.png).
  figS10_gep_usage_composite.png   Figure S10. GEP usage across cell cycle phases (A) and
                                   across cell types and developmental stages (B), the two
                                   Illustrator heatmaps (Figures/highres/image16.png and
                                   image17.png), which carry no panel letters of their own.

The Illustrator panels are not resampled. The exp24 panels are scaled so that their type
has the size of the Illustrator type (both are drawn for the 174 mm figure width), and the
panel letters of the composites have the size and the left edge of the Illustrator letters.

Run after exp28 and exp24:  python experiments/exp41_supplementary_composites.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HI = os.path.join(BASE, 'Figures', 'highres')
RES = os.path.join(BASE, 'experiments', 'results')
BOLD = '/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf'
W2COL_IN = 174 / 25.4
EXP24_DPI = 600
MAX_QUANT_ERROR = 2.5       # mean |dRGB| out of 255 allowed from the palette conversion


def dark_bbox(a, thr=100):
    ys, xs = np.where(a < thr)
    return ys.min(), ys.max(), xs.min(), xs.max()


def letter_metrics(img):
    """Left edge, top and cap height of the panel letter A in the top-left corner."""
    y0, y1, x0, _ = dark_bbox(np.array(img.convert('L'))[:150, :200])
    return x0, y0, y1 - y0 + 1


def font_for_cap(cap_px):
    """Bold font whose capital A is cap_px pixels tall."""
    size = cap_px
    for _ in range(40):
        f = ImageFont.truetype(BOLD, size)
        box = f.getbbox('A')
        h = box[3] - box[1]
        if h == cap_px:
            return f
        size += 1 if h < cap_px else -1
    return f


def draw_letter(canvas, letter, x, y_top, font):
    d = ImageDraw.Draw(canvas)
    box = font.getbbox(letter)
    d.text((x - box[0], y_top - box[1]), letter, font=font, fill=(0, 0, 0))


def save(img, name):
    rgb = img.convert('RGB')
    pal = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    err = np.abs(np.asarray(pal.convert('RGB'), dtype=float) - np.asarray(rgb, dtype=float)).mean()
    out = pal if err <= MAX_QUANT_ERROR else rgb
    path = os.path.join(RES, name)
    out.save(path, optimize=True, dpi=(round(img.width / W2COL_IN),) * 2)
    print(f'  saved {name}: {img.width} x {img.height} px, {os.path.getsize(path) / 1e6:.1f} MB'
          f'{" (palette, mean |dRGB| %.2f)" % err if out is pal else ""}')


def figure_s5():
    top = Image.open(os.path.join(HI, 'image11.png')).convert('RGB')
    row = Image.open(os.path.join(RES, 'figS5EF_extrapolation.png')).convert('RGB')
    # Illustrator panels: width = 174 mm; exp24 panels: 600 dpi at their drawn size
    scale = top.width / W2COL_IN / EXP24_DPI
    row = row.resize((round(row.width * scale), round(row.height * scale)), Image.LANCZOS)
    x_a, _, _ = letter_metrics(top)
    x_e = dark_bbox(np.array(row.convert('L'))[:round(0.12 * row.height), :400])[2]   # letter E
    gap = round(0.12 * top.width / W2COL_IN)          # 0.12 in between the two parts
    canvas = Image.new('RGB', (top.width, top.height + gap + row.height), 'white')
    canvas.paste(top, (0, 0))
    canvas.paste(row, (x_a - x_e, top.height + gap))
    save(canvas, 'figS5_recreation_composite.png')


def figure_s10():
    a = Image.open(os.path.join(HI, 'image16.png')).convert('RGB')
    b = Image.open(os.path.join(HI, 'image17.png')).convert('RGB')
    ref = Image.open(os.path.join(HI, 'image11.png'))
    x_l, _, cap = letter_metrics(ref)
    cap = round(cap * max(a.width, b.width) / ref.width)
    font = font_for_cap(cap)
    band = 2 * cap                                     # white band that carries the letter
    width = max(a.width, b.width)
    canvas = Image.new('RGB', (width, band + a.height + band + b.height), 'white')
    canvas.paste(a, (0, band))
    canvas.paste(b, (0, band + a.height + band))
    draw_letter(canvas, 'A', x_l, round(cap * 0.4), font)
    draw_letter(canvas, 'B', x_l, band + a.height + round(cap * 0.4), font)
    save(canvas, 'figS10_gep_usage_composite.png')


if __name__ == '__main__':
    print('Figure S5'); figure_s5()
    print('Figure S10'); figure_s10()
