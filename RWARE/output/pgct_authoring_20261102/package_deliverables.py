from pathlib import Path
import json,zipfile,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).parent
shutil.copyfile(OUT/'evidence.json',ROOT/'paper/evidence_manifest.json')
shutil.copyfile(OUT/'quality_report.json',ROOT/'presentation/quality_report.json')
target=ROOT/'PGCT_paper_presentation_20261102.zip'
files=[ROOT/'PGCT_DELIVERABLES_README.md']
extensions={'.tex','.pdf','.pptx','.bib','.csv','.svg','.png','.json','.md'}
for name in ['paper','presentation']:
    files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix in extensions)
for name in ['collect_evidence.py','plot_evidence.py','prepare_slides.py','render_equations.py',
             'build_presentation.py','compile_documents.py','quality_check.py','evidence.json','quality_report.json','compilation.json']:
    files.append(OUT/name)
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    required=['paper/english/main.tex','paper/english/main.pdf','paper/korean/main.tex','paper/korean/main.pdf',
              'paper/references.bib','presentation/pgct_beamer.tex','presentation/pgct_beamer.pdf',
              'presentation/pgct_presentation.pptx','presentation/pgct_script_ko.md']
    assert set(required)<=set(z.namelist())
manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT/'deliverable_hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('ZIP:',target,'files:',len(files),'bytes:',target.stat().st_size)
