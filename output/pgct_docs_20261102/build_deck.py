"""Build presentation/pgct_beamer.tex, presentation/pgct_presentation.pptx and presentation/pgct_script_ko.md
from deck_spec.py. Every slide element has one geometry on a 13.333 x 7.5 inch canvas. PowerPoint draws it
with native shapes and Beamer draws the same element as a TikZ overlay node, so both decks match. The three
method figures are TikZ files in Beamer and are redrawn as native PowerPoint shapes from the same coordinates.
Run with a Python that has python-pptx (no matplotlib needed). Equation PNGs: render_deck_equations.py.
"""
from pathlib import Path
import csv
import sys

from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from deck_spec import EQUATIONS, METHODS, SLIDES  # noqa: E402

PRES = ROOT / "presentation"
FIG = PRES / "figures"
W, H = 13.333, 7.5
K = 16 / (W * 2.54)            # Beamer scale: 16 cm wide frame for a 13.333 inch canvas
COL = dict(pgblue="1F4E9A", pgorange="C0504D", pggreen="2E7D4F", pggrey="6B6B6B", pgsky="DCE7F7",
           pgsand="FBE9DA", pgmint="E2F2E8", pglight="F4F4F4", black="1A1A1A", white="FFFFFF")


def rgb(name):
    return RGBColor.from_string(COL[name])


SYMBOLS = {"►": r"\ensuremath{\blacktriangleright}", "▶": r"\ensuremath{\blacktriangleright}", "→": r"\ensuremath{\rightarrow}", "⇒": r"\ensuremath{\Rightarrow}",
           "≥": r"\ensuremath{\geq}", "≤": r"\ensuremath{\leq}", "≈": r"\ensuremath{\approx}", "×": r"\ensuremath{\times}",
           "−": r"\ensuremath{-}", "γ": r"\ensuremath{\gamma}", "λ": r"\ensuremath{\lambda}", "β": r"\ensuremath{\beta}",
           "η": r"\ensuremath{\eta}", "Δ": r"\ensuremath{\Delta}"}


def tex_escape(s):
    s = (s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
         .replace("\n", r"\\"))
    for k, v in SYMBOLS.items():
        s = s.replace(k, v)
    return s


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.tex = []
        self.slide = None

    # ---------- coordinates ----------
    @staticmethod
    def pt(x, y):  # canvas inch -> Beamer cm offset from the page's north west corner
        return f"([xshift={x * 2.54 * K:.3f}cm,yshift={-y * 2.54 * K:.3f}cm]current page.north west)"

    @staticmethod
    def fs(size):  # PowerPoint points -> Beamer points
        s = size * K
        return rf"\fontsize{{{s:.2f}}}{{{s * 1.22:.2f}}}\selectfont"

    def new_slide(self):
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = rgb("white")
        self.tex += [r"\begin{frame}[plain]", r"\begin{tikzpicture}[remember picture,overlay,pg]"]

    def end_slide(self, number):
        if number > 1:
            self.text(W - 1.2, H - 0.45, 0.9, 0.3, [(f"{number}", "pggrey", False)], 11, PP_ALIGN.RIGHT)
        self.tex += [r"\end{tikzpicture}", r"\end{frame}", ""]

    # ---------- primitives ----------
    def text(self, x, y, w, h, runs, size, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, beamer=True, italic=False):
        """runs: list of (text, colour, bold); '\n' starts a new paragraph."""
        tb = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        p = tf.paragraphs[0]
        p.alignment = align
        for text, colour, bold in runs:
            parts = text.split("\n")
            for k, part in enumerate(parts):
                if k:
                    p = tf.add_paragraph()
                    p.alignment = align
                if part:
                    r = p.add_run()
                    r.text = part
                    r.font.name, r.font.size, r.font.bold, r.font.italic = "Arial", Pt(size), bold, italic
                    r.font.color.rgb = rgb(colour)
        if beamer:
            body = "".join((r"\textbf{" if b else "") + rf"\textcolor{{{c}}}{{{tex_escape(t)}}}" + ("}" if b else "")
                           for t, c, b in runs)
            if italic:
                body = r"\itshape " + body
            al = {PP_ALIGN.LEFT: "left", PP_ALIGN.CENTER: "center", PP_ALIGN.RIGHT: "right"}[align]
            if anchor == MSO_ANCHOR.MIDDLE:
                pos, anc = self.pt(x + (w / 2 if al == "center" else 0 if al == "left" else w), y + h / 2), \
                    {"left": "west", "center": "center", "right": "east"}[al]
            else:
                pos, anc = self.pt(x + (w / 2 if al == "center" else 0 if al == "left" else w), y), \
                    {"left": "north west", "center": "north", "right": "north east"}[al]
            self.tex.append(rf"\node[anchor={anc},inner sep=0pt,text width={w * 2.54 * K:.3f}cm,align={al}] at {pos} "
                            rf"{{{self.fs(size)}{body}}};")
        return tb

    def bullet(self, x, y, w, head, body, colour, size):
        self.text(x, y + 0.04 * size / 20, 0.3, 0.3, [("►", "pgblue", False)], size * 0.62)
        runs = ([(head + ": ", colour, False)] if head else []) + [(body, "black", False)]
        lines = est_lines(head + ": " + body if head else body, w - 0.32, size)
        self.text(x + 0.32, y, w - 0.32, lines * size * 1.25 / 72, runs, size)
        return lines * size * 1.25 / 72

    def rect(self, x, y, w, h, line="pgblue", fill="white", lw=1.25, radius=True, dash=False):
        shp = self.slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                          Inches(x), Inches(y), Inches(w), Inches(h))
        if radius:
            shp.adjustments[0] = min(0.12, 0.06 / max(min(w, h), 0.1))
        if fill is None:
            shp.fill.background()
        else:
            shp.fill.solid()
            shp.fill.fore_color.rgb = rgb(fill)
        if line is None:
            shp.line.fill.background()
        else:
            shp.line.color.rgb = rgb(line)
            shp.line.width = Pt(lw)
            if dash:
                ln = shp.line._get_or_add_ln()
                d = OxmlElement("a:prstDash")
                d.set("val", "dash")
                ln.append(d)
        shp.shadow.inherit = False
        tf = shp.text_frame
        tf.text = ""
        opts = [f"line width={lw * K:.2f}pt" if line else "draw=none", f"draw={line}" if line else "",
                f"fill={fill}" if fill else "", "rounded corners=2pt" if radius else "", "dashed" if dash else ""]
        self.tex.append(rf"\draw[{','.join(o for o in opts if o)}] {self.pt(x, y)} rectangle {self.pt(x + w, y + h)};")
        return shp

    def box(self, x, y, w, h, title, body, line="pgblue", fill="white", size=14, tcol=None, lw=1.25, dash=False):
        self.rect(x, y, w, h, line, fill, lw, dash=dash)
        runs = []
        if title:
            runs.append((title + ("\n" if body else ""), tcol or line, True))
        if body:
            runs.append((body, "black", False))
        self.text(x + 0.06, y, w - 0.12, h, runs, size, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)

    def arrow(self, pts, colour="pgblue", lw=1.5, head=True):
        """Polyline through pts (canvas inches); arrow head at the end."""
        for k in range(len(pts) - 1):
            (x1, y1), (x2, y2) = pts[k], pts[k + 1]
            c = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            c.line.color.rgb = rgb(colour)
            c.line.width = Pt(lw)
            if head and k == len(pts) - 2:
                ln = c.line._get_or_add_ln()
                tail = OxmlElement("a:tailEnd")
                tail.set("type", "triangle")
                tail.set("w", "med")
                tail.set("len", "med")
                ln.append(tail)
        path = " -- ".join(self.pt(x, y) for x, y in pts)
        self.tex.append(rf"\draw[{'-{Stealth[length=2mm]},' if head else ''}draw={colour},line width={lw * K:.2f}pt] {path};")

    def picture(self, x, y, w, h, png, pdf=None):
        im = Image.open(png)
        ratio = im.size[1] / im.size[0]
        ww, hh = (w, w * ratio) if w * ratio <= h else (h / ratio, h)
        xx, yy = x + (w - ww) / 2, y + (h - hh) / 2
        self.slide.shapes.add_picture(str(png), Inches(xx), Inches(yy), Inches(ww), Inches(hh))
        if pdf is not None:
            self.tex.append(rf"\node[anchor=north west,inner sep=0pt] at {self.pt(xx, yy)} "
                            rf"{{\includegraphics[width={ww * 2.54 * K:.3f}cm]{{{pdf}}}}};")
        return xx, yy, ww, hh

    def equation(self, key, x, y, w, h, size=20):
        """PowerPoint: PNG rendered from mathtext. Beamer: the LaTeX form at the matching size."""
        png = FIG / f"eq_{key}.png"
        im = Image.open(png)
        scale = size / 22                            # PNGs were rendered at 22 pt, 300 dpi
        ww, hh = im.size[0] / 300 * scale, im.size[1] / 300 * scale
        if ww > w:
            ww, hh = w, hh * w / ww
        xx, yy = x + (w - ww) / 2, y + (h - hh) / 2
        self.slide.shapes.add_picture(str(png), Inches(xx), Inches(yy), Inches(ww), Inches(hh))
        self.tex.append(rf"\node[anchor=center,inner sep=0pt] at {self.pt(x + w / 2, y + h / 2)} "
                        rf"{{\resizebox{{{ww * 2.54 * K:.3f}cm}}{{!}}{{$\displaystyle {EQUATIONS[key][0]}$}}}};")

    def table(self, x, y, w, h, header, rows, widths, size, first_bold=False):
        shp = self.slide.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(h))
        tbl = shp.table
        tbl.first_row = True
        for j, cw in enumerate(widths):
            tbl.columns[j].width = Inches(cw)
        for i, row in enumerate([header] + rows):
            for j, val in enumerate(row):
                cell = tbl.cell(i, j)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb("pgsky" if i == 0 else "white")
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.margin_top = cell.margin_bottom = Inches(0.04)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf = cell.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                r = p.add_run()
                r.text = val
                r.font.name, r.font.size = "Arial", Pt(size)
                r.font.bold = i == 0 or (first_bold and j == 0)
                pend = val.lower() == "pending"
                r.font.italic = pend
                r.font.color.rgb = rgb("pgblue" if i == 0 else "pgorange" if pend else "black")
        rh = h / (len(rows) + 1)
        xs = [x]
        for cw in widths:
            xs.append(xs[-1] + cw)
        self.tex.append(rf"\fill[pgsky] {self.pt(x, y)} rectangle {self.pt(x + w, y + rh)};")
        for i in range(len(rows) + 2):
            self.tex.append(rf"\draw[pggrey!45,line width=.3pt] {self.pt(x, y + i * rh)} -- {self.pt(x + w, y + i * rh)};")
        for i, row in enumerate([header] + rows):
            for j, val in enumerate(row):
                pend = val.lower() == "pending"
                colour = "pgblue" if i == 0 else "pgorange" if pend else "black"
                bold = i == 0 or (first_bold and j == 0)
                self.tex.append(rf"\node[anchor=west,inner sep=0pt,text width={(widths[j] - 0.16) * 2.54 * K:.3f}cm] "
                                rf"at {self.pt(xs[j] + 0.08, y + (i + 0.5) * rh)} {{{self.fs(size)}"
                                + (r"\bfseries" if bold else "") + (r"\itshape" if pend else "")
                                + rf"\textcolor{{{colour}}}{{{tex_escape(val)}}}}};")

    def chart(self, x, y, w, h, series, xlabel, ylabel, ymax, pdf, legend=True, size=13):
        cd = XyChartData()
        for name, pts, _ in series:
            s = cd.add_series(name)
            for px, py in pts:
                s.add_data_point(px, py)
        gf = self.slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(x), Inches(y),
                                         Inches(w), Inches(h), cd)
        ch = gf.chart
        ch.has_legend = legend
        if legend:
            ch.legend.position = XL_LEGEND_POSITION.BOTTOM
            ch.legend.include_in_layout = False
            ch.legend.font.name, ch.legend.font.size = "Arial", Pt(size)
        for ser, (_, _, colour) in zip(ch.series, series):
            ser.format.line.color.rgb = rgb(colour)
            ser.format.line.width = Pt(2.25)
            ser.smooth = False
        for ax, label in ((ch.category_axis, xlabel), (ch.value_axis, ylabel)):
            ax.tick_labels.font.name, ax.tick_labels.font.size = "Arial", Pt(size)
            ax.has_title = True
            ax.axis_title.text_frame.text = label
            for p in ax.axis_title.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name, r.font.size, r.font.bold = "Arial", Pt(size + 1), False
        ch.category_axis.minimum_scale, ch.category_axis.maximum_scale, ch.category_axis.major_unit = 0, 5, 1
        ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, ymax
        ch.value_axis.has_major_gridlines = True
        for item in ch._chartSpace.iter():  # OOXML axis ids are unsigned
            if item.tag.rsplit("}", 1)[-1] in ("axId", "crossAx") and item.get("val") is not None:
                item.set("val", str(int(item.get("val")) & 0xFFFFFFFF))
        self.tex.append(rf"\node[anchor=center,inner sep=0pt] at {self.pt(x + w / 2, y + h / 2)} "
                        rf"{{\includegraphics[width={w * 2.54 * K:.3f}cm,height={h * 2.54 * K:.3f}cm,keepaspectratio]{{{pdf}}}}};")

    def tikzfig(self, name, x, y, w, bbox):
        """Beamer: the shared TikZ figure scaled to width w. Returns the cm->canvas-inch mapper for PowerPoint."""
        (x0, y0, x1, y1) = bbox
        s = w / (x1 - x0)                                   # canvas inch per TikZ cm
        hh = (y1 - y0) * s
        # The figure is a tikzpicture nested in an overlay node; reset the options it would otherwise inherit.
        self.tex.append(rf"\node[anchor=north west,inner sep=0pt] at {self.pt(x, y)} "
                        rf"{{\resizebox{{{w * 2.54 * K:.3f}cm}}{{!}}{{\tikzset{{anchor=center,inner sep=.3333em}}"
                        rf"\input{{../paper/figures/{name}.tex}}}}}};")
        return (lambda cx, cy: (x + (cx - x0) * s, y + (y1 - cy) * s)), s, hh


def est_lines(text, width_in, size):
    per_line = max(8, int(width_in * 72 / (size * 0.49)))
    words, lines, cur = text.split(), 1, 0
    for wd in words:
        if cur + len(wd) + (1 if cur else 0) > per_line:
            lines, cur = lines + 1, len(wd)
        else:
            cur += len(wd) + (1 if cur else 0)
    return lines


def read_csv(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------- native redraws of the three TikZ figures ----------------
CM = 1 / 2.54


def fig_architecture(d, m, s):
    fsz = 8.6 * s / CM            # TikZ \small at the figure's scale, in points
    st = [("Independent learning", "own rollout and own\nreward; log contacts", 0, 0),
          ("Individual reward\nprediction", "MLP fitted to own reward", 4.1, 0),
          ("Reward probe", "public events of i and j\nplus mirrored copies", 8.2, 0),
          ("Latent team\ndiscovery", "response distance → approval", 12.3, 0),
          ("Peer Critic\nselection", "approved and met\nin this rollout", 12.3, -3.1),
          ("Advantage\nestimation", "Max-GAE on own history", 8.2, -3.1),
          ("PPO update", "Actor: PGCT advantage\nCritic: own return only", 4.1, -3.1)]
    bw, bh = 3.45, 1.75
    x0, y0 = m(-2.0, 1.25)
    x1, y1 = m(14.1, -0.95)
    d.rect(x0, y0, x1 - x0, y1 - y0, None, "pgsky")
    x0, y0 = m(2.2, -2.05)
    x1, y1 = m(14.1, -4.15)
    d.rect(x0, y0, x1 - x0, y1 - y0, None, "pgsand")
    lx, ly = m(5.5, 1.6)
    d.text(lx, ly, 8.6 * s, 0.35 * s, [("Discover latent teammates from own reward", "pgblue", True)], fsz, PP_ALIGN.RIGHT)
    lx, ly = m(5.5, -4.2)
    d.text(lx, ly, 8.6 * s, 0.35 * s, [("Use discovered teammates in learning", "pgorange", True)], fsz, PP_ALIGN.RIGHT)
    for n, (t, b, cx, cy) in enumerate(st, 1):
        bx, by = m(cx - bw / 2, cy + bh / 2)
        d.box(bx, by, bw * s, bh * s, t, b, size=fsz)
        nx, ny = m(cx - bw / 2 - 0.22, cy + bh / 2 + 0.22)
        circ = d.slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(nx), Inches(ny), Inches(0.44 * s), Inches(0.44 * s))
        circ.fill.solid()
        circ.fill.fore_color.rgb = rgb("pgblue")
        circ.line.fill.background()
        tf = circ.text_frame
        for mg in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, mg, 0)
        r = tf.paragraphs[0].add_run()
        r.text = str(n)
        r.font.size, r.font.bold, r.font.name = Pt(fsz * 0.8), True, "Arial"
        r.font.color.rgb = rgb("white")
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    for (a, b) in ((0, 1), (1, 2), (2, 3), (5, 4), (6, 5)):
        ax_, ay_ = st[a][2], st[a][3]
        bx_, by_ = st[b][2], st[b][3]
        if a < b:
            d.arrow([m(ax_ + bw / 2, ay_), m(bx_ - bw / 2, by_)], head=True) if False else None
    for (a, b) in ((0, 1), (1, 2), (2, 3)):
        d.arrow([m(st[a][2] + bw / 2, 0), m(st[b][2] - bw / 2, 0)])
    d.arrow([m(12.3, -bh / 2), m(12.3, -3.1 + bh / 2)])
    for (a, b) in ((4, 5), (5, 6)):
        d.arrow([m(st[a][2] - bw / 2, -3.1), m(st[b][2] + bw / 2, -3.1)])
    d.arrow([m(4.1, -3.1 - bh / 2), m(4.1, -4.45), m(-2.32, -4.45), m(-2.32, 0), m(-bw / 2, 0)])
    tx, ty = m(-2.62, -0.6)
    d.text(tx - 0.05, ty, 0.25, 1.4 * s, [("next rollout", "pggrey", False)], fsz * 0.8, beamer=False)
    cx, cy = m(-1.62, -2.25)
    d.rect(cx, cy, 3.25 * s, 1.75 * s, "pggrey", "white", 0.75)
    d.text(cx + 0.06, cy + 0.04, 3.13 * s, 1.7 * s,
           [("Shared only on contact\n", "black", True), ("sent: ", "pggreen", True),
            ("public event views, predicted responses, Critic weights\n", "black", False),
            ("never sent: ", "pgorange", True), ("rewards, team labels, Actor weights", "black", False)],
           fsz * 0.8, beamer=False)


def fig_discovery(d, m, s):
    fsz = 8.4 * s / CM
    for cy, t, colour in ((1.5, "Agent i", "pgblue"), (-1.5, "Agent j", "pgorange")):
        bx, by = m(-1.55, cy + 0.75)
        d.box(bx, by, 3.1 * s, 1.5 * s, t, "own reward events\nreward MLP", line=colour, size=fsz)
    qx, qy = m(2.55, 1.45)
    d.rect(qx, qy, 3.2 * s, 2.9 * s, "pgblue", None)
    d.text(qx, qy - 0.35 * s, 3.2 * s, 0.35 * s, [("Common probes", "pgblue", True)], fsz * 0.9, PP_ALIGN.CENTER, beamer=False)
    for cy, t, f in ((0.85, "32 recent events of i", "pgsky"), (0.0, "32 recent events of j", "pgsand"),
                     (-0.85, "left–right mirrors", "pglight")):
        bx, by = m(2.85, cy + 0.27)
        d.box(bx, by, 2.6 * s, 0.54 * s, None, t, line="pggrey", fill=f, size=fsz * 0.85, lw=0.75)
    d.arrow([m(1.55, 1.5), m(2.55, 0.85)], "pggrey", 1.0)
    d.arrow([m(1.55, -1.5), m(2.55, -0.85)], "pggrey", 1.0)
    lx, ly = m(2.45, -1.75)
    d.text(lx, ly, 3.4 * s, 0.7 * s, [("public views and actions only;\nreward values stay private", "pggrey", False)],
           fsz * 0.8, PP_ALIGN.CENTER, beamer=False)
    rx, ry = m(7.1, 2.4)
    d.rect(rx, ry, 4.1 * s, 4.6 * s, "pggrey", None, 0.75, dash=True)
    for oy, lab, colour, pi, pj, dd in ((0.45, "same team", "pggreen", [.9, .8, .85, .75, 0, 0, 0, 0],
                                         [.85, .75, .9, .8, 0, .05, 0, 0], "D ≈ 0"),
                                        (-1.95, "different team", "pgorange", [.9, .8, 0, 0, 0, 0, .85, .8],
                                         [0, 0, .9, .85, .8, .85, 0, 0], "D ≈ 1")):
        tx, ty = m(7.25, oy + 1.75)
        d.text(tx, ty, 2.0 * s, 0.3 * s, [(lab, colour, True)], fsz * 0.8, beamer=False)
        for k, (hi, hj) in enumerate(zip(pi, pj)):
            for base, hgt, c in ((0.8, hi, "pgblue"), (0.0, hj, "pgorange")):
                if hgt > 0:
                    bx, by = m(7.3 + 0.27 * k, oy + base + 0.55 * hgt)
                    d.rect(bx, by, 0.17 * s, 0.55 * hgt * s, None, c, radius=False)
        for yy, lab2 in ((1.05, "agent i"), (0.25, "agent j")):
            tx, ty = m(9.5, oy + yy + 0.15)
            d.text(tx, ty, 0.9 * s, 0.3 * s, [(lab2, "black", False)], fsz * 0.8, beamer=False)
        tx, ty = m(10.3, oy + 0.85)
        d.text(tx, ty, 0.95 * s, 0.4 * s, [(dd, colour, True)], fsz, beamer=False)
    tx, ty = m(7.1, -2.3)
    d.text(tx, ty, 4.1 * s, 0.6 * s, [("predicted rewards on the same probes\n(illustration, not data)", "pggrey", False)],
           fsz * 0.75, PP_ALIGN.CENTER, beamer=False)
    d.arrow([m(5.85, 0.0), m(7.1, 0.0)])
    bx, by = m(11.95, 2.1)
    d.rect(bx, by, 4.2 * s, 3.5 * s, "pgblue", "white")
    d.text(bx, by + 0.12 * s, 4.2 * s, 0.4 * s, [("Response distance", "pgblue", True)], fsz, PP_ALIGN.CENTER, beamer=False)
    pic = FIG / "eq_distance.png"
    im = Image.open(pic)
    ew = 3.6 * s
    d.slide.shapes.add_picture(str(pic), Inches(bx + 0.3 * s), Inches(by + 0.6 * s), Inches(ew),
                               Inches(ew * im.size[1] / im.size[0]))
    d.text(bx, by + 1.75 * s, 4.2 * s, 1.6 * s,
           [("EMA over rounds\n", "black", False), ("approve", "black", True),
            (" if EMA < 0.5,\n≥ 3 rounds, both mature", "black", False)], fsz * 0.9, PP_ALIGN.CENTER, beamer=False)
    d.arrow([m(11.2, 0.35), m(11.95, 0.35)])
    cx, cy = m(12.0, -1.75)
    d.box(cx, cy, 4.1 * s, 0.85 * s, None, "fewer than 20 own events or no fitted predictor → abstain",
          line="pggrey", fill="pglight", size=fsz * 0.8, lw=0.75)
    # Beamer draws only the TikZ figure; drop the duplicated PowerPoint-only text nodes above
    return


def fig_critic(d, m, s):
    fsz = 8.2 * s / CM
    eqh = 0.32 * s

    def node(cx, cy, w, h, title, body, line="pgblue", tcol=None, lw=1.25):
        bx, by = m(cx - w / 2, cy + h / 2)
        d.box(bx, by, w * s, h * s, title, body, line=line, size=fsz, tcol=tcol, lw=lw)

    node(0, 0.8, 2.75, 1.45, "My own history", "my observations\nsteps 1 … T")
    node(4.4, 3.55, 3.85, 1.35, "Donor set", "approved and\nmet during this rollout", line="pggreen")
    node(4.4, 1.65, 3.85, 1.35, "Approved peer Critics", "teammate value of my history\nreplayed, frozen weights")
    node(4.4, -0.15, 3.85, 1.35, "Own Critic", "my own value")
    node(9.2, 1.65, 4.25, 1.35, "Positive uplift", "b = best teammate value − my value,\nif positive; f = my value + b")
    node(9.2, -0.15, 4.25, 1.35, "Self GAE", "own advantage and own return\nfrom own reward")
    node(14.0, 1.65, 3.65, 1.35, "Max-GAE", "advantage from f and own reward;\nmasked difference Δ")
    node(14.0, -0.15, 3.65, 1.35, "Actor advantage", "own advantage + η · s · Δ\ns ≤ 1 only shrinks Δ",
         line="pgorange", lw=1.6)
    node(14.0, -2.15, 3.6, 1.0, "Actor update", "PPO with PGCT advantage")
    node(9.2, -2.15, 3.6, 1.0, "Critic update", "own-return target only")
    d.arrow([m(1.375, 0.8), m(1.75, 0.8), m(1.75, 1.65), m(2.475, 1.65)])
    d.arrow([m(1.75, 0.8), m(1.75, -0.15), m(2.475, -0.15)])
    d.arrow([m(4.4, 2.875), m(4.4, 2.325)], "pggreen")
    d.arrow([m(6.325, 1.65), m(7.075, 1.65)])
    d.arrow([m(6.325, -0.15), m(7.075, -0.15)])
    d.arrow([m(6.7, -0.15), m(6.7, 1.25), m(7.075, 1.25)])
    d.arrow([m(11.325, 1.65), m(12.175, 1.65)])
    d.arrow([m(14.0, 0.975), m(14.0, 0.525)])
    d.arrow([m(11.325, -0.15), m(12.175, -0.15)])
    d.arrow([m(14.0, -0.825), m(14.0, -1.65)], "pgorange")
    d.arrow([m(9.2, -0.825), m(9.2, -1.65)])
    tx, ty = m(2.25, -1.2)
    d.text(tx, ty, 4.3 * s, 0.7 * s, [("peer values never enter Critic targets;\nno peer distillation loss", "pggrey", False)],
           fsz * 0.8, PP_ALIGN.CENTER, beamer=False)
    tx, ty = m(12.2, 3.95)
    d.text(tx, ty, 3.6 * s, 0.75 * s, [("no donor or no uplift\n→ independent PPO", "pggrey", False)],
           fsz * 0.8, PP_ALIGN.CENTER, beamer=False)


FIGS = {"fig_architecture": (fig_architecture, (-2.674, -4.751, 14.290, 1.607)),
        "fig_discovery": (fig_discovery, (-1.605, -3.274, 16.255, 2.442)),
        "fig_critic": (fig_critic, (-1.455, -2.664, 15.917, 4.273))}


# ---------------- small right-hand visuals (drawn identically in both decks) ----------------
def visual(d, name):
    if name == "background":
        d.rect(7.55, 1.4, 5.2, 3.75, "pgblue", "white")
        for gx in (8.2, 9.3, 10.4, 11.5):
            for gy in (1.75, 2.45, 3.15):
                d.rect(gx, gy, 0.75, 0.45, "pggrey", "pglight", 0.5, radius=False)
        d.box(7.7, 4.4, 1.25, 0.6, None, "Goal A", fill="pgsky", size=13, lw=0.75)
        d.box(11.35, 4.4, 1.25, 0.6, None, "Goal B", fill="pgsand", size=13, lw=0.75)
        for rx, ry in ((7.85, 2.0), (9.15, 3.85), (10.25, 2.9), (11.15, 1.55), (10.0, 4.45), (12.2, 3.6)):
            d.box(rx, ry, 0.42, 0.42, None, "?", line="pggrey", fill="white", size=14, lw=1.0)
        d.text(7.55, 5.35, 5.2, 0.45, [("Who is my teammate?", "pgblue", True)], 20, PP_ALIGN.CENTER)
        d.text(7.55, 5.85, 5.2, 0.35, [("robots look alike; team labels are hidden", "pggrey", False)], 13, PP_ALIGN.CENTER)
    elif name == "questions":
        d.box(7.7, 1.55, 5.0, 1.45, "Q1  Discovery (main)", "Who shares my reward rule?", line="pgorange", size=18, lw=2)
        d.arrow([(10.2, 3.0), (10.2, 3.7)])
        d.text(10.35, 3.15, 2.5, 0.4, [("approved teammates", "pggrey", False)], 13)
        d.box(7.7, 3.7, 5.0, 1.45, "Q2  Use (test)", "Does their Critic help my Actor?", size=18)
    elif name == "idea":
        steps = [("Own reward experience", "pgblue", "pgsky"), ("Reward responses on probes", "pgblue", "pgsky"),
                 ("Latent teammates", "pgblue", "pgsky"), ("Approved peer Critic", "pgorange", "pgsand"),
                 ("Actor advantage", "pgorange", "pgsand")]
        for k, (t, line, fill) in enumerate(steps):
            d.box(8.0, 1.35 + 0.9 * k, 4.2, 0.6, None, t, line=line, fill=fill, size=15)
            if k:
                d.arrow([(10.1, 1.35 + 0.9 * k - 0.3), (10.1, 1.35 + 0.9 * k)])
        d.text(12.3, 2.1, 0.6, 0.4, [("Q1", "pgblue", True)], 16)
        d.text(12.3, 4.45, 0.6, 0.4, [("Q2", "pgorange", True)], 16)
    elif name == "predictor":
        for x, w, t in ((7.55, 1.75, "public view before"), (9.45, 1.2, "action"), (10.8, 1.95, "public view after")):
            d.box(x, 1.45, w, 0.62, None, t, line="pggrey", fill="pglight", size=13, lw=0.75)
            d.arrow([(x + w / 2, 2.07), (10.15, 2.7)], "pggrey", 1.0)
        d.box(8.55, 2.7, 3.2, 0.95, "Reward MLP", "567 → 128 → 64 → 1", size=15)
        d.arrow([(10.15, 3.65), (10.15, 4.15)])
        d.box(8.55, 4.15, 3.2, 0.8, None, "predicted own reward", line="pgorange", size=15)
        d.text(7.55, 5.15, 5.2, 0.4, [("trained on own reward only", "pggrey", False)], 13, PP_ALIGN.CENTER)
    elif name == "probes":
        d.rect(7.45, 1.35, 3.35, 2.55, "pgblue", None)
        d.text(7.45, 1.42, 3.35, 0.35, [("Common probes", "pgblue", True)], 15, PP_ALIGN.CENTER)
        for k, (t, f) in enumerate((("32 events of i", "pgsky"), ("32 events of j", "pgsand"), ("left–right mirrors", "pglight"))):
            d.box(7.65, 1.85 + 0.65 * k, 2.95, 0.5, None, t, line="pggrey", fill=f, size=13, lw=0.75)
        for k, (gx, lab) in enumerate(((11.2, "event"), (12.05, "mirror"))):
            d.rect(gx - 0.05, 1.6 + 1.25 * k, 0.95, 0.95, "pggrey", "white", 0.75, radius=False)
        d.rect(11.15, 2.25, 0.3, 0.25, None, "pgblue", radius=False)
        d.rect(12.65, 3.5, 0.3, 0.25, None, "pgblue", radius=False)
        d.text(11.15, 1.25, 1.2, 0.3, [("goal left", "pggrey", False)], 11)
        d.text(11.95, 4.15, 1.2, 0.3, [("goal right", "pggrey", False)], 11)
        d.arrow([(11.65, 2.55), (11.65, 3.1), (12.0, 3.1)], "pggrey", 1.0)
        d.text(7.45, 4.25, 5.4, 0.9, [("The same probes go to both predictors.\nOnly public views and actions are shared.",
                                       "pggrey", False)], 13, PP_ALIGN.CENTER)
    elif name == "access":
        d.box(9.85, 1.3, 3.0, 0.85, None, "approved AND met in this rollout", line="pggreen", size=13)
        d.box(7.45, 2.6, 1.9, 1.0, "My history", "steps 1 … T", size=13)
        d.box(9.85, 2.6, 3.0, 1.0, "Donor Critic", "weights received on contact", size=13)
        d.box(9.85, 4.1, 3.0, 1.0, None, "teammate's value of my situation", line="pgorange", size=13)
        d.arrow([(11.35, 2.15), (11.35, 2.6)], "pggreen")
        d.arrow([(9.35, 3.1), (9.85, 3.1)])
        d.arrow([(11.35, 3.6), (11.35, 4.1)])
        d.text(7.45, 5.35, 5.4, 0.4, [("donor hidden state is never sent", "pggrey", False)], 13, PP_ALIGN.CENTER)
    elif name == "updates":
        d.box(7.45, 1.55, 2.4, 0.85, None, "PGCT advantage", line="pgorange", size=14)
        d.box(10.45, 1.55, 2.4, 0.85, "Actor", "PPO", size=14)
        d.box(7.45, 3.15, 2.4, 0.85, None, "own return", size=14)
        d.box(10.45, 3.15, 2.4, 0.85, "Critic", "value loss", size=14)
        d.arrow([(9.85, 1.975), (10.45, 1.975)], "pgorange")
        d.arrow([(9.85, 3.575), (10.45, 3.575)])
        d.text(7.45, 4.35, 5.4, 0.45, [("no peer value enters the Critic target", "pgorange", False)], 13, PP_ALIGN.CENTER)
    elif name == "setup_table":
        rows = [["Grid / agents", "11×10 / 6 (3 + 3)"], ["Sensor / contact radius", "5 / 3 cells"],
                ["Rollout / γ / λ", "1024 / 0.99 / 0.95"], ["Event / maturity", "r ≥ 0.5 / 20 events"],
                ["EMA β / threshold / rounds", "0.35 / 0.5 / 3"], ["Max-GAE η / scale", "0.5 / shrink only"],
                ["Evaluation", "20 episodes × 500 steps"]]
        d.table(7.5, 1.35, 5.35, 4.25, ["Setting", "Value"], rows, [2.75, 2.6], 13)


# ---------------- slides ----------------
def build():
    d = Deck()
    script = ["# PGCT 발표 대본 (슬라이드별)", "",
              "슬라이드 번호와 제목은 `pgct_beamer.pdf`, `pgct_presentation.pptx`와 같습니다. "
              "수식이 있는 슬라이드는 기호를 하나씩 읽기보다 식의 의미를 설명하도록 썼습니다. "
              "진행 중인 학습 비교는 결과가 나온 뒤 16번 슬라이드의 표와 함께 대본을 고쳐야 합니다.", ""]
    for n, sl in enumerate(SLIDES, 1):
        d.new_slide()
        kind = sl["kind"]
        if kind == "title":
            d.text(0.8, 1.55, W - 1.6, 0.8, [(sl["title"], "pgblue", True)], 38, PP_ALIGN.CENTER)
            d.text(0.8, 2.55, W - 1.6, 0.6, [(sl["subtitle"], "pgblue", False)], 24, PP_ALIGN.CENTER)
            d.rect(4.2, 3.45, W - 8.4, 0.02, None, "pgblue", radius=False)
            for k, line in enumerate(sl["lines"]):
                d.text(0.8, 4.0 + 0.55 * k, W - 1.6, 0.45, [(line, "black", False)], 18, PP_ALIGN.CENTER)
        else:
            d.text(0.45, 0.3, W - 0.9, 0.6, [(sl["title"], "pgblue", False)], 28)
        if kind == "outline":
            for k, item in enumerate(sl["items"]):
                d.bullet(1.3, 1.75 + 0.95 * k, 8, "", item, "pgblue", 24)
        elif kind in ("split", "conclusion"):
            wide = kind == "conclusion"
            x, y, w = 0.55, 1.35, (12.2 if wide else 6.65)
            size = 19 if wide else (18 if len(sl["items"]) >= 5 else 20)
            for head, body, colour in sl["items"]:
                hgt = d.bullet(x, y, w, head, body, colour, size)
                y += hgt + (0.32 if wide else 0.36)
            eqs = [k for k in ("equation", "equation2") if k in sl]
            for k, key in enumerate(eqs):
                d.equation(sl[key], 0.55, 5.55 + 0.68 * k - (0.3 if len(eqs) == 1 else 0.35), 6.65, 0.6)
            if "visual" in sl:
                visual(d, sl["visual"])
            if wide:
                d.text(0.5, 6.15, W - 1.0, 0.6, [("Thank you!   ", "pgblue", True), ("Questions?", "black", False)],
                       26, PP_ALIGN.CENTER)
        elif kind == "table":
            d.table(0.55, 1.35, 12.2, 3.9, sl["header"], sl["rows"], [4.6, 3.5, 4.1], 16, first_bold=False)
            d.text(0.55, 5.6, 12.2, 0.8, [("Research gap: ", "pgorange", False),
                                          (sl["gap"].split(": ", 1)[1], "black", False)], 20)
        elif kind == "figure":
            fn, bbox = FIGS[sl["visual"]]
            m, s, hh = d.tikzfig(sl["visual"], 0.45, 1.1, W - 0.9, bbox)
            n_tex = len(d.tex)
            fn(d, m, s)
            # PowerPoint-only drawing: keep the Beamer frame to the TikZ figure itself
            del d.tex[n_tex:]
            yb = 1.1 + hh + 0.15
            if "equation" in sl:
                d.equation(sl["equation"], 0.45, yb, W - 0.9, 0.55)
                yb += 0.65
            if "note" in sl:
                d.text(0.45, min(yb, 6.75), W - 0.9, 0.4, [(sl["note"], "pggrey", False)], 14, PP_ALIGN.CENTER)
        elif kind == "results":
            bins = read_csv("discovery_bins.csv")
            same = [(float(r["step_million"]), float(r["same_ema"])) for r in bins if r["same_ema"]]
            cross = [(float(r["step_million"]), float(r["cross_ema"])) for r in bins if r["cross_ema"]]
            d.chart(0.45, 1.15, 6.1, 3.95, [("Same team", same, "pgblue"), ("Different team", cross, "pgorange"),
                                            ("Threshold 0.5", [(0, .5), (5, .5)], "pggrey")],
                    "Training steps (million)", "Response distance (EMA)", 1.05, "figures/discovery_ema.pdf")
            alloc = read_csv("allocations_cumulative.csv")
            d.chart(6.8, 1.15, 6.1, 3.95,
                    [("Same team", [(float(r["step_million"]), float(r["same_team"])) for r in alloc], "pgblue"),
                     ("Different team", [(float(r["step_million"]), float(r["cross_team"])) for r in alloc], "pgorange")],
                    "Training steps (million)", "Cumulative approved arcs", 5500, "figures/discovery_alloc.pdf")
            y = 5.25
            for head, body, colour in sl["items"]:
                y += d.bullet(0.6, y, 12.2, head, body, colour, 17) + 0.08
            d.text(0.45, 6.85, W - 0.9, 0.35, [(sl["note"], "pggrey", False)], 12, PP_ALIGN.CENTER)
        elif kind == "results2":
            base = read_csv("baseline_eval.csv")
            series = [(f"Independent PPO, seed {s}", [(float(r["step_million"]), float(r["eval_deliveries"]))
                                                      for r in base if r["seed"] == str(s)], c)
                      for s, c in ((1, "pgblue"), (2, "pggreen"))]
            d.chart(0.45, 1.2, 6.3, 4.7, series, "Training steps (million)", "Deliveries per episode", 30,
                    "figures/baseline_ippo.pdf")
            d.rect(7.05, 1.25, 5.85, 1.55, "pggreen", "pgmint", 1.0)
            d.text(7.2, 1.33, 5.6, 0.35, [(sl["check"][0], "pggreen", True)], 15)
            for k, line in enumerate(sl["check"][1:]):
                d.text(7.2, 1.78 + 0.45 * k, 5.6, 0.4, [("► ", "pggreen", False), (line, "black", False)], 14)
            d.table(7.05, 3.05, 5.85, 2.85, sl["header"], [[mth, "Pending", "Pending"] for mth in METHODS],
                    [2.85, 1.5, 1.5], 13, first_bold=False)
            d.text(0.45, 6.4, W - 0.9, 0.5, [(sl["note"], "pggrey", False)], 13, PP_ALIGN.CENTER)
        d.end_slide(n)
        script += [f"## {n}. {sl['title']}", "", sl["script"], ""]
        notes = d.slide.notes_slide.notes_text_frame
        notes.text = sl["script"]

    preamble = r"""% Generated by output/pgct_docs_20261102/build_deck.py from deck_spec.py; edit the spec, not this file.
\documentclass[aspectratio=169,10pt]{beamer}
\usepackage{fontspec}
\setsansfont{Arial}
\setmainfont{Arial}
\usepackage{amsmath,amssymb,graphicx,tikz}
\input{../paper/figures/pgct_style.tex}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{background canvas}{\color{white}\rule{\paperwidth}{\paperheight}}
\usefonttheme{professionalfonts}
\begin{document}
"""
    (PRES / "pgct_beamer.tex").write_text(preamble + "\n".join(d.tex) + "\n\\end{document}\n", encoding="utf-8")
    d.prs.save(PRES / "pgct_presentation.pptx")
    (PRES / "pgct_script_ko.md").write_text("\n".join(script), encoding="utf-8")
    print("slides", len(SLIDES))


if __name__ == "__main__":
    build()
