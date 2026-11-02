from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'CONVERSATION_LOG.md'
s=Path(__file__).with_name('T167_transcript.md').read_text(encoding='utf-8')
assert s.splitlines()[0] not in p.read_text(encoding='utf-8'), 'T167 already present'
with p.open('ab') as f: f.write(('\r\n'+s.replace('\r\n','\n').replace('\n','\r\n')).encode('utf-8'))
print('Appended T167 with CRLF; existing bytes preserved')
