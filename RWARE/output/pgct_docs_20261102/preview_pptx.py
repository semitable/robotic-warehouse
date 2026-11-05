"""Approximate PNG preview of pgct_presentation.pptx (no PowerPoint/LibreOffice here), for layout checks only.

Draws shapes, connectors, pictures, tables, chart lines and wrapped text with Arial at the stored sizes, and
reports text whose wrapped height exceeds its frame. Usage: preview_pptx.py <pptx> <out_dir>
"""
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

DPI = 90
FONTS = {}


def font(size, bold):
    key = (round(size * DPI / 72), bold)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype("C:/Windows/Fonts/" + ("arialbd.ttf" if bold else "arial.ttf"), key[0])
    return FONTS[key]


def px(emu):
    return emu / 914400 * DPI


def wrap(draw, paragraphs, width):
    """paragraphs: [(alignment, [(text, font, colour)])] -> lines [(alignment, [(word, font, colour)], height)]"""
    lines = []
    for align, runs in paragraphs:
        words = [(w, f, c) for t, f, c in runs for w in t.replace("\n", " ").split(" ") if w]
        cur, curw, h = [], 0, max([f.size for _, f, _ in runs] or [10])
        for w, f, c in words:
            ww = draw.textlength(w + " ", font=f)
            if cur and curw + ww > width:
                lines.append((align, cur, h))
                cur, curw = [], 0
            cur.append((w, f, c))
            curw += ww
        lines.append((align, cur, h))
    return lines


def draw_text(draw, tf, x, y, w, h, issues, label):
    paras = []
    for p in tf.paragraphs:
        runs = []
        for r in p.runs:
            size = r.font.size.pt if r.font.size else 18
            col = tuple(r.font.color.rgb) if r.font.color and r.font.color.type is not None else (0, 0, 0)
            runs.append((r.text, font(size, bool(r.font.bold)), col))
        if runs:
            paras.append((p.alignment, runs))
    if not paras:
        return
    ml = px(tf.margin_left or 0)
    mt = px(tf.margin_top or 0)
    lines = wrap(draw, paras, w - 2 * ml)
    total = sum(hh * 1.18 for _, _, hh in lines)
    if total > h + 3:
        issues.append(f"{label}: text needs {total / DPI:.2f} in, frame {h / DPI:.2f} in: "
                      + " ".join(t for _, rr in paras for t, _, _ in rr)[:60])
    yy = y + mt + ((h - total) / 2 if tf.vertical_anchor == MSO_ANCHOR.MIDDLE else 0)
    for align, words, hh in lines:
        lw = sum(draw.textlength(wd + " ", font=f) for wd, f, _ in words)
        xx = x + ml + ((w - 2 * ml - lw) / 2 if align == PP_ALIGN.CENTER else (w - 2 * ml - lw) if align == PP_ALIGN.RIGHT else 0)
        for wd, f, c in words:
            draw.text((xx, yy), wd, font=f, fill=c)
            xx += draw.textlength(wd + " ", font=f)
        yy += hh * 1.18


def render(path, out):
    prs = Presentation(path)
    W, H = px(prs.slide_width), px(prs.slide_height)
    out.mkdir(parents=True, exist_ok=True)
    issues = []
    for n, slide in enumerate(prs.slides, 1):
        img = Image.new("RGB", (int(W), int(H)), "white")
        d = ImageDraw.Draw(img)
        for k, sh in enumerate(slide.shapes):
            x, y, w, h = px(sh.left), px(sh.top), px(sh.width), px(sh.height)
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                issues.append(f"slide {n}: shape {k} outside the slide")
            label = f"slide {n} shape {k}"
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                im = Image.open(sh.image._blob if False else __import__("io").BytesIO(sh.image.blob)).convert("RGBA")
                im = im.resize((max(1, int(w)), max(1, int(h))))
                img.paste(im, (int(x), int(y)), im)
            elif sh.__class__.__name__ == "Connector":
                x1, y1, x2, y2 = px(sh.begin_x), px(sh.begin_y), px(sh.end_x), px(sh.end_y)
                col = tuple(sh.line.color.rgb) if sh.line.color.type is not None else (0, 0, 0)
                d.line([(x1, y1), (x2, y2)], fill=col, width=2)
                if "tailEnd" in sh.line._get_or_add_ln().xml:
                    import math
                    a = math.atan2(y2 - y1, x2 - x1)
                    d.polygon([(x2, y2), (x2 - 9 * math.cos(a - .4), y2 - 9 * math.sin(a - .4)),
                               (x2 - 9 * math.cos(a + .4), y2 - 9 * math.sin(a + .4))], fill=col)
            elif sh.has_chart:
                d.rectangle([x, y, x + w, y + h], outline=(160, 160, 160))
                cs = sh.chart._chartSpace
                ns = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
                sers = cs.findall(".//c:ser", ns)
                vals = []
                for s in sers:
                    xs = [float(v.text) for v in s.findall("./c:xVal//c:v", ns)]
                    ys = [float(v.text) for v in s.findall("./c:yVal//c:v", ns)]
                    vals.append((xs, ys))
                ymax = max(max(ys) for _, ys in vals) or 1
                for (xs, ys), col in zip(vals, [(31, 78, 154), (192, 80, 77), (107, 107, 107)]):
                    pts = [(x + 60 + (xx / 5) * (w - 80), y + h - 70 - (yy / ymax) * (h - 100)) for xx, yy in zip(xs, ys)]
                    if len(pts) > 1:
                        d.line(pts, fill=col, width=2)
            elif sh.has_table:
                tbl = sh.table
                yy = y
                for r, row in enumerate(tbl.rows):
                    xx = x
                    rh = px(row.height)
                    for c, cell in enumerate(row.cells):
                        cw = px(tbl.columns[c].width)
                        d.rectangle([xx, yy, xx + cw, yy + rh], outline=(200, 200, 200),
                                    fill=(220, 231, 247) if r == 0 else None)
                        draw_text(d, cell.text_frame, xx + 4, yy + 2, cw - 8, rh - 4, issues, f"{label} cell {r},{c}")
                        xx += cw
                    yy += rh
            else:
                if sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
                    fill = tuple(sh.fill.fore_color.rgb) if sh.fill.type == 1 else None
                    line = tuple(sh.line.color.rgb) if sh.line.fill.type != 5 and sh.line.color.type is not None else None
                    if "OVAL" in str(sh.auto_shape_type):
                        d.ellipse([x, y, x + w, y + h], fill=fill, outline=line)
                    else:
                        d.rounded_rectangle([x, y, x + w, y + h], radius=5, fill=fill, outline=line, width=2 if line else 0)
                if sh.has_text_frame:
                    draw_text(d, sh.text_frame, x, y, w, h, issues, label)
        img.save(out / f"slide-{n:02d}.png")
    return issues


if __name__ == "__main__":
    found = render(sys.argv[1], Path(sys.argv[2]))
    print(f"{len(found)} issues")
    for line in found:
        print(" ", line)
