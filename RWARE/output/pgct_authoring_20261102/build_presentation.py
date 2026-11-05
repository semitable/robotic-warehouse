"""Editable PPTX and matching Beamer source, authored from one slide specification.

User requested python-pptx/PptxGenJS where possible. Diagrams, charts, tables and
prose are native PPTX objects. Only mathematical equations use PNG fragments.
"""
from pathlib import Path
import json, math, re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.chart.data import XyChartData
from pptx.oxml.xmlchemy import OxmlElement
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'presentation'
S=json.loads((OUT/'slides.json').read_text(encoding='utf-8'))
D=json.loads((Path(__file__).parent/'evidence.json').read_text(encoding='utf-8'))
P=Presentation(); P.slide_width=Inches(13.333333); P.slide_height=Inches(7.5)
BLUE='3634B8'; INK='161616'; GRAY='656565'; ORANGE='B65B20'; LIGHT='E4E4EB'
tex=[r'\documentclass[aspectratio=169,10pt]{beamer}',r'\usepackage{fontspec,amsmath,amssymb,tikz,graphicx,booktabs}',
     r'\setsansfont{Arial}',r'\geometry{paperwidth=13.333333in,paperheight=7.5in}',
     r'\definecolor{pgctblue}{HTML}{3634B8}',r'\definecolor{pgctgray}{HTML}{656565}',
     r'\setbeamercolor{normal text}{fg=black,bg=white}',r'\setbeamertemplate{navigation symbols}{}',
     r'\setbeamertemplate{footline}{}',r'\setbeamertemplate{headline}{}',r'\begin{document}']

def esc(s):
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]: s=s.replace(a,b)
    return s.replace('≥',r'$\ge$').replace('→',r'$\rightarrow$').replace('×',r'$\times$').replace('▶',r'$\blacktriangleright$')

def node_tex(text,x,y,w,size,color='black',bold=False,align='left',h=None):
    text=esc(text).replace('\n',r'\\')
    return (r'\node[anchor=north west,inner sep=0pt,align='+align+',text width='+f'{w:.4f}in'
            +f',text={color}' +'] at ([xshift='+f'{x:.4f}in,yshift=-{y:.4f}in'+']current page.north west) {'
            +r'\fontsize{'+str(size)+'}{'+str(size*1.22)+r'}\selectfont\spaceskip=.22em\relax '+(r'\bfseries ' if bold else '')+text+'};')

def text(slide,s,x,y,w,h,size=21,color=INK,bold=False,align=PP_ALIGN.LEFT,beamer=True):
    box=slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf=box.text_frame; tf.word_wrap=True
    tf.margin_left=tf.margin_right=0; tf.margin_top=tf.margin_bottom=0
    for k,line in enumerate(s.split('\n')):
        p=tf.paragraphs[0] if not k else tf.add_paragraph(); p.text=line
        p.font.name='Arial'; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=RGBColor.from_string(color)
        p.alignment=align; p.space_after=Pt(5)
    if beamer: tex.append(node_tex(s,x,y,w,size,'pgctblue' if color==BLUE else ('pgctgray' if color==GRAY else 'black'),bold,
                                  'center' if align==PP_ALIGN.CENTER else 'left'))
    return box

def bullet(slide,s,x,y,w,size=22):
    text(slide,'▶',x,y,.28,.4,size-2,BLUE)
    text(slide,s,x+.35,y,w-.35,.95,size)

def rect(slide,label,x,y,w=2.2,h=.7,size=17,color=BLUE):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor(255,255,255)
    sh.line.color.rgb=RGBColor.from_string(color); sh.line.width=Pt(1)
    sh.adjustments[0]=.06
    text(slide,label,x+.09,y+.12,w-.18,h-.12,size,color,align=PP_ALIGN.CENTER,beamer=False)
    tex.append(r'\draw[draw=pgctblue,rounded corners=2pt,line width=.8pt] ([xshift='+str(x)+'in,yshift=-'+str(y)+'in]current page.north west) rectangle ++('+str(w)+'in,-'+str(h)+'in);')
    tex.append(node_tex(label,x+.09,y+.12,w-.18,size,'pgctblue',align='center'))

def arrow(slide,x1,y1,x2,y2):
    c=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
    c.line.color.rgb=RGBColor.from_string(BLUE); c.line.width=Pt(1.5)
    end=OxmlElement('a:tailEnd'); end.set('type','triangle'); c.line._get_or_add_ln().append(end)
    tex.append(r'\draw[->,draw=pgctblue,line width=1pt] ([xshift='+str(x1)+'in,yshift=-'+str(y1)+'in]current page.north west)--([xshift='+str(x2)+'in,yshift=-'+str(y2)+'in]current page.north west);')

def image(slide,path,x,y,w,h):
    iw,ih=Image.open(path).size; ratio=min(w/iw,h/ih); ww,hh=iw*ratio,ih*ratio
    slide.shapes.add_picture(str(path),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))
    return ww,hh

def equation(slide,eq,i,j,y,size=23):
    path=OUT/f'figures/equation_{i:02d}_{j}.png'; image(slide,path,.6,y,12.1,1.3 if i==10 else .85)
    tex.append(r'\node[anchor=north,inner sep=0pt] at ([yshift=-'+str(y+.1)+'in]current page.north) {'
               +r'\fontsize{'+str(size)+'}{'+str(size*1.2)+r'}\selectfont $\displaystyle '+eq+'$};')

def table(slide,data,x,y,w,h,widths=None,size=18):
    rows,cols=len(data),len(data[0]); sh=slide.shapes.add_table(rows,cols,Inches(x),Inches(y),Inches(w),Inches(h)); tb=sh.table
    if widths is None: widths=[w/cols]*cols
    for j,v in enumerate(widths): tb.columns[j].width=Inches(v)
    for i,row in enumerate(data):
        tb.rows[i].height=Inches(h/rows)
        for j,v in enumerate(row):
            c=tb.cell(i,j); c.text=v; c.margin_left=Inches(.09); c.margin_right=Inches(.06)
            c.margin_top=Inches(.09); c.margin_bottom=Inches(.02); c.vertical_anchor=MSO_ANCHOR.MIDDLE
            c.fill.solid(); c.fill.fore_color.rgb=RGBColor.from_string('EFEFF9' if i==0 else 'FFFFFF')
            for p in c.text_frame.paragraphs:
                p.font.name='Arial'; p.font.size=Pt(size); p.font.bold=i==0
                p.font.color.rgb=RGBColor.from_string(BLUE if i==0 else INK)
    # Reproduce a flat native-looking table as positioned text and rules in Beamer.
    rh=h/rows
    tex.append(r'\fill[blue!5] ([xshift='+str(x)+'in,yshift=-'+str(y)+'in]current page.north west) rectangle ++('+str(w)+'in,-'+str(rh)+'in);')
    for i,row in enumerate(data):
        xx=x
        for j,v in enumerate(row):
            tex.append(node_tex(v,xx+.09,y+i*rh+.12,widths[j]-.18,size,'pgctblue' if i==0 else 'black',i==0))
            xx+=widths[j]
        if i==0 or i==rows-1:
            yy=y+(i+1)*rh
            tex.append(r'\draw[gray!35] ([xshift='+str(x)+'in,yshift=-'+str(yy)+'in]current page.north west)--++('+str(w)+'in,0);')

def chart(slide,mode,x,y,w,h):
    cd=XyChartData()
    if mode=='discovery':
        rr=D['discovery']['bins']
        for key,name in [('same','Same team'),('cross','Different team')]:
            series=cd.add_series(name)
            for r in rr: series.add_data_point(r['step']/1e6,r[key])
        series=cd.add_series('Approval threshold'); series.add_data_point(0,.5); series.add_data_point(5,.5)
    else:
        series=cd.add_series('Independent PPO (seed 1)')
        for r in D['ippo']['curve']: series.add_data_point(r['step']/1e6,r['deliveries'])
    sh=slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch=sh.chart; ch.has_legend=True; ch.legend.position=XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout=False
    ch.legend.font.name='Arial'; ch.legend.font.size=Pt(13)
    for k,ser in enumerate(ch.series):
        ser.format.line.color.rgb=RGBColor.from_string([BLUE,ORANGE,GRAY][k]); ser.format.line.width=Pt(2)
    for ax in [ch.category_axis,ch.value_axis]:
        ax.tick_labels.font.name='Arial'; ax.tick_labels.font.size=Pt(13); ax.has_title=True
        for p in ax.axis_title.text_frame.paragraphs: p.font.name='Arial'; p.font.size=Pt(14)
    ch.category_axis.minimum_scale=0; ch.category_axis.maximum_scale=5; ch.category_axis.major_unit=1
    ch.category_axis.axis_title.text_frame.text='Training steps (million)'
    ch.value_axis.minimum_scale=0; ch.value_axis.maximum_scale=1.05 if mode=='discovery' else 30
    ch.value_axis.axis_title.text_frame.text='Response distance (EMA)' if mode=='discovery' else 'Deliveries / episode'
    ch.value_axis.has_major_gridlines=True
    # Chart axis identifiers are unsigned in OOXML. Normalize python-pptx's signed
    # defaults so strict importers can read the editable XY charts as well.
    for item in ch._chartSpace.iter():
        if item.tag.rsplit('}',1)[-1] in ('axId','crossAx') and item.get('val') is not None:
            item.set('val',str(int(item.get('val')) & 0xffffffff))
    # Beamer uses the exact same extracted observations, rendered as a vector PDF.
    asset='figures/discovery_chart.pdf' if mode=='discovery' else 'figures/learning_chart.pdf'
    tex.append(r'\node[anchor=north west,inner sep=0pt] at ([xshift='+str(x)+'in,yshift=-'+str(y)+r'in]current page.north west) {\includegraphics[width='+str(w)+'in,height='+str(h)+'in,keepaspectratio]{'+asset+'}};')

for idx,s in enumerate(S,1):
    sl=P.slides.add_slide(P.slide_layouts[6]); sl.background.fill.solid(); sl.background.fill.fore_color.rgb=RGBColor(255,255,255)
    tex.extend([r'\begin{frame}[plain]',r'\begin{tikzpicture}[remember picture,overlay]'])
    if s['kind']=='cover':
        text(sl,s['title'],.7,1,11.93,.8,35,BLUE,True,PP_ALIGN.CENTER)
        text(sl,s['bullets'][0],1.2,2.15,10.93,1.1,29,BLUE,align=PP_ALIGN.CENTER)
        text(sl,s['bullets'][1],1.5,4.5,10.33,.6,20,align=PP_ALIGN.CENTER)
    else:
        text(sl,s['title'],.3,.24,12.75,.7,28,BLUE)
        kind=s['kind']
        if kind=='bullets':
            for k,v in enumerate(s['bullets']): bullet(sl,v,.75,1.55+k*1.03,11.8,23)
        elif kind=='background':
            for k,v in enumerate(s['bullets']): bullet(sl,v,.55,1.55+k*1.22,6,23)
            rect(sl,'Shared workspace',7.3,1.7,4.7,2.7,24)
            rect(sl,'Goal A',7.6,3.4,1.7,.7,18); rect(sl,'Goal B',10,3.4,1.7,.7,18)
            text(sl,'Different personal objectives',7.2,4.7,4.9,.7,20,align=PP_ALIGN.CENTER)
            text(sl,'Task illustration, not a trajectory',7.2,5.6,4.9,.5,14,GRAY,align=PP_ALIGN.CENTER)
        elif kind=='related': table(sl,s['table'],.55,1.45,12.2,4.6,[2.7,3.3,6.2],19)
        elif kind=='architecture':
            nodes=[('Independent\nlearning',.65,1.7),('Individual reward\nprediction',3.85,1.7),('Public reward\nprobes',7.05,1.7),('Latent team\ndiscovery',10.25,1.7),
                   ('Selective peer\nCritic access',10.25,3.6),('Max-GAE\nadvantage',7.05,3.6),('Actor PPO\nupdate',3.85,3.6),('Own-return\nCritic update',.65,3.6)]
            for label,x,y in nodes: rect(sl,label,x,y,2.45,1.05,18)
            for x in [.65,3.85,7.05]: arrow(sl,x+2.45,2.225,x+3.2,2.225)
            arrow(sl,11.475,2.75,11.475,3.6)
            arrow(sl,10.25,4.125,9.5,4.125); arrow(sl,7.05,4.125,6.3,4.125); arrow(sl,1.875,2.75,1.875,3.6)
            text(sl,s['bullets'][1],1,5.55,11.4,.8,21,align=PP_ALIGN.CENTER)
        elif kind=='predictor':
            for k,v in enumerate(s['bullets'][1:]): bullet(sl,v,.7,1.45+k*.8,11.9,21)
            rect(sl,'Public before\n+ action + public after',1,3.3,3.1,1,18)
            rect(sl,'Personal MLP\n567 → 128 → 64 → 1',5.1,3.3,3.1,1,18)
            rect(sl,'Own predicted\nreward',9.2,3.3,3.1,1,18)
            arrow(sl,4.1,3.8,5.1,3.8); arrow(sl,8.2,3.8,9.2,3.8)
            equation(sl,s['equations'][0],idx,0,5.0)
        elif kind=='probes':
            for k,v in enumerate(s['bullets'][:2]): bullet(sl,v,.7,1.4+k*.8,11.9,22)
            rect(sl,'Public events\n+ mirror probes',.9,3.3,2.9,1.1,19)
            rect(sl,'Predictor i response',4.7,3,3,.65,17)
            rect(sl,'Predictor j response',4.7,3.85,3,.65,17)
            rect(sl,'Compare responses\n→ team approval',8.65,3.3,3.6,1.1,19)
            arrow(sl,3.8,3.6,4.7,3.325); arrow(sl,3.8,4.05,4.7,4.175)
            arrow(sl,7.7,3.325,8.65,3.6); arrow(sl,7.7,4.175,8.65,4.05)
            for k,v in enumerate(s['bullets'][2:]): bullet(sl,v,.7,4.7+k*.85,11.9,21)
        elif kind=='distance':
            equation(sl,s['equations'][0],idx,0,1.35,25)
            for k,v in enumerate(s['bullets']): bullet(sl,v,.7,2.8+k*.9,11.9,22)
            text(sl,'Maturity: ≥20 own events, fitted predictor, own-event response ≥0.3',1.1,6,11.3,.6,18,GRAY)
        elif kind=='critics':
            for k,v in enumerate(s['bullets']): bullet(sl,v,.7,1.4+k*.78,11.9,21)
            rect(sl,'My observation\nstream',.85,4.3,2.7,1.1,19)
            rect(sl,'Own Critic +\napproved peer Critics',4.5,4.3,3.2,1.1,19)
            rect(sl,'Positive uplift\n→ Actor advantage',8.65,4.3,3.7,1.1,19)
            arrow(sl,3.55,4.85,4.5,4.85); arrow(sl,7.7,4.85,8.65,4.85)
        elif kind=='maxgae':
            for k,eq in enumerate(s['equations']): equation(sl,eq,idx,k,1.12+k*.98,19 if k==1 else 21)
            for k,v in enumerate(s['bullets']): bullet(sl,v,.7,5.25+k*.47,11.9,17)
            text(sl,'γ = 0.99; λ = 0.95; first 7 cold-start steps: auxiliary term = 0',1.05,6.63,11.8,.3,14,GRAY)
        elif kind=='updates':
            for k,v in enumerate(s['bullets']): bullet(sl,v,.7,1.45+k*.83,11.9,22)
            equation(sl,s['equations'][0],idx,0,5.2,25)
        elif kind=='setup': table(sl,s['table'],.6,1.3,12.1,5.4,[4.5,7.6],18)
        elif kind=='discovery_chart':
            chart(sl,'discovery',.55,1.55,7.3,4.65)
            for k,v in enumerate(s['bullets']): bullet(sl,v,8.15,1.7+k*1.08,4.6,19)
        elif kind=='learning_chart':
            chart(sl,'learning',.55,1.55,7.3,4.65)
            table(sl,[['Comparison','Result'],['Final PGCT','Pending'],['True-team peer','Pending'],['Mature oracle','Pending'],['Count-matched peer','Pending']],8.25,1.75,4.45,2.85,[2.95,1.5],17)
            text(sl,s['bullets'][2],8.3,5.1,4.3,1.1,18,GRAY)
    if idx!=1:
        text(sl,s.get('footer','PGCT | Research draft'),.45,6.96,11.8,.35,10,GRAY)
    text(sl,f'{idx} / 17',12.3,7.03,.7,.25,10,GRAY,align=PP_ALIGN.RIGHT)
    notes=s['script']+'\n\nSources: '+ '; '.join(s.get('sources',[]))
    if idx==15: notes+='\n'+D['discovery']['source']+'; transfer.jsonl and metrics.jsonl. Allocation-weighted share, not pair accuracy. Component log used update-end contacts.'
    if idx==16: notes+='\n'+D['ippo']['source']+'; metrics.jsonl; eval/deliveries. Final PGCT result pending.'
    sl.notes_slide.notes_text_frame.text=notes
    tex.extend([r'\end{tikzpicture}',r'\end{frame}'])
tex.append(r'\end{document}')
P.save(OUT/'pgct_presentation.pptx')
(OUT/'pgct_beamer.tex').write_text('\n'.join(tex),encoding='utf-8')
print('Created 17-slide editable PPTX and matching Beamer source')
