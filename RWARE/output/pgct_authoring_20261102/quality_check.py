from pathlib import Path
import hashlib,json,re,zipfile,xml.etree.ElementTree as ET
from pypdf import PdfReader
from pptx import Presentation
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).parent
en=(ROOT/'paper/english/main.tex').read_text(encoding='utf-8')
ko=(ROOT/'paper/korean/main.tex').read_text(encoding='utf-8')
math_names=re.findall(r'\\newcommand\{(\\\w+Equation)\}',(ROOT/'paper/method_equations.tex').read_text())
en_math=[x for x in re.findall(r'\\\w+Equation',en) if x in math_names]
ko_math=[x for x in re.findall(r'\\\w+Equation',ko) if x in math_names]
assert en_math==ko_math and len(en_math)==11
cit=lambda s:sorted(set(x for a in re.findall(r'\\cite\w*\{([^}]+)\}',s) for x in a.split(',')))
assert cit(en)==cit(ko)
assert re.findall(r'\\label\{([^}]+)\}',en)==re.findall(r'\\label\{([^}]+)\}',ko)
S=json.loads((ROOT/'presentation/slides.json').read_text(encoding='utf-8'))
p=Presentation(ROOT/'presentation/pgct_presentation.pptx')
assert len(S)==len(p.slides)==17
beamer=(ROOT/'presentation/pgct_beamer.tex').read_text(encoding='utf-8')
assert beamer.count(r'\begin{frame}')==17
for i,(s,sl) in enumerate(zip(S,p.slides),1):
    words='\n'.join(sh.text for sh in sl.shapes if sh.has_text_frame)
    assert s['title'] in words
    assert s['script'] in sl.notes_slide.notes_text_frame.text
assert len(PdfReader(ROOT/'presentation/pgct_beamer.pdf').pages)==17
reports={}
for rel in ['paper/english/main','paper/korean/main','presentation/pgct_beamer']:
    log=(ROOT/(rel+'.log')).read_text(encoding='utf-8',errors='replace')
    errors=[l for l in log.splitlines() if any(x in l for x in ['Overfull','Missing character','Undefined control sequence','! LaTeX Error'])]
    assert not errors,(rel,errors)
    pp=PdfReader(ROOT/(rel+'.pdf'))
    reports[rel]=dict(pages=len(pp.pages),overfull_or_missing_glyphs=errors,
                     size_bytes=(ROOT/(rel+'.pdf')).stat().st_size)
evidence=json.loads((OUT/'evidence.json').read_text(encoding='utf-8'))
for name,old in evidence['sources'].items():
    if name.startswith('latent_team'):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==old,name
reports['consistency']=dict(shared_equation_blocks=en_math,citations=cit(en),matching_labels=True,
    matching_slide_script_order=True,slides=17,charts=sum(sh.has_chart for sl in p.slides for sh in sl.shapes),
    tables=sum(sh.has_table for sl in p.slides for sh in sl.shapes),research_code_hashes_unchanged=True,
    pptx_equation_image_slides=[8,10,12,13],audio_transcribed=False)
(OUT/'quality_report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
# Contact sheets support visual review; individual full-size renders remain available.
for family,paths in [('paper_en',sorted(OUT.glob('english-*.png'))),('paper_ko',sorted(OUT.glob('korean-*.png'))),
                     ('pptx',sorted((OUT/'pptx_render_final').glob('slide-*.png'),key=lambda p:int(p.stem.split('-')[-1]))),
                     ('beamer',sorted(OUT.glob('beamer_final-*.png')))]:
    for block in range(0,len(paths),6):
        group=paths[block:block+6]; canvas=Image.new('RGB',(1440,3*430),'#dddddd'); draw=ImageDraw.Draw(canvas)
        for k,path in enumerate(group):
            im=Image.open(path).convert('RGB'); im.thumbnail((700,390))
            x=(k%2)*720+(720-im.width)//2; y=(k//2)*430+28
            canvas.paste(im,(x,y)); draw.text(((k%2)*720+12,(k//2)*430+7),path.name,fill='black')
        canvas.save(OUT/f'{family}_contact_{block//6+1}.png')
print(json.dumps(reports,ensure_ascii=False,indent=2))
