from pathlib import Path
import json, struct, sys
from pypdf import PdfReader
sys.stdout.reconfigure(encoding='utf-8')
out=Path(__file__).parent
src=Path('C:/Users/95101/OneDrive/Documents/카카오톡 받은 파일')
reader=PdfReader(src/'이기윤_ICCAS.pdf')
text=[]; fonts=set()
for i,p in enumerate(reader.pages):
    s=p.extract_text()
    text.append(f'\n=== SLIDE {i+1} ===\n{s}')
    for v in p.get('/Resources',{}).get('/Font',{}).values():
        fonts.add(str(v.get_object().get('/BaseFont')))
    print(i+1, tuple(float(x) for x in p.mediabox), s[:150].replace('\n',' | '))
(out/'reference_slides.txt').write_text('\n'.join(text),encoding='utf-8')
blob=(src/'음성 261023_135711.m4a').read_bytes()
def atoms(data,start=0,end=None):
    end=len(data) if end is None else end
    while start+8<=end:
        size,tag=struct.unpack_from('>I4s',data,start); header=8
        if size==1: size=struct.unpack_from('>Q',data,start+8)[0]; header=16
        if size==0: size=end-start
        if size<header: break
        yield tag,start+header,start+size
        start+=size
duration=None
for tag,a,b in atoms(blob):
    if tag==b'moov':
        for sub,c,d in atoms(blob,a,b):
            if sub==b'mvhd':
                if blob[c]==0: timescale,length=struct.unpack_from('>II',blob,c+12)
                else: timescale=struct.unpack_from('>I',blob,c+20)[0]; length=struct.unpack_from('>Q',blob,c+24)[0]
                duration=length/timescale
meta=dict(pdf_pages=len(reader.pages),pdf_fonts=sorted(fonts),audio_bytes=len(blob),audio_seconds=duration,
          audio_transcribed=False,audio_limit='No installed audio transcription tool/model or connector found.')
(out/'reference_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False))
