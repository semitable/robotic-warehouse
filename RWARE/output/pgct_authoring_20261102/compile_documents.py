from pathlib import Path
import os, subprocess, json, re
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
tex=OUT/'texruntime/tectonic.exe'
env=os.environ.copy(); env['TECTONIC_CACHE_DIR']=str(OUT/'texcache')
reports=[]
for rel in ['paper/english/main.tex','paper/korean/main.tex','presentation/pgct_beamer.tex']:
    p=ROOT/rel
    if rel.startswith('paper/'):
        s=p.read_text(encoding='utf-8')
        s=re.sub(r'\\texttt\{([^{}]+)\}',lambda m:r'\path{'+m[1].replace(r'\_','_')+'}',s)
        s=s.replace(r'\begin{tabular}{lll}',r'\begin{tabular}{p{.4\columnwidth}p{.21\columnwidth}p{.2\columnwidth}}')
        p.write_text(s,encoding='utf-8')
    r=subprocess.run([str(tex),'--keep-logs',str(p)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    label=rel.replace('/','_').replace('.tex','')
    (OUT/f'{label}_build.txt').write_bytes(r.stdout)
    reports.append(dict(file=rel,returncode=r.returncode,pdf_exists=p.with_suffix('.pdf').exists()))
    print(json.dumps(reports[-1]),flush=True)
(OUT/'compilation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
