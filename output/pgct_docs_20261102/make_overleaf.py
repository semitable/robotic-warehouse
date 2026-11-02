"""Flat, upload-ready copy of the IJCAI-27 English paper for Overleaf (pdfLaTeX): paper/overleaf_ijcai27/ and a zip.
All '../' paths are rewritten to the flat layout; sources in paper/ stay the single place to edit."""
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper"
OUT = SRC / "overleaf_ijcai27"
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "figures").mkdir(parents=True)
for name in ("main.tex", "supplementary.tex"):
    text = (SRC / "english" / name).read_text(encoding="utf-8")
    text = text.replace("{../figures/", "{figures/").replace("{../method_equations.tex}", "{method_equations.tex}")
    text = text.replace(r"\bibliography{../references}", r"\bibliography{references}")
    (OUT / name).write_text(text, encoding="utf-8")
for name in ("ijcai27.sty", "named.bst"):
    shutil.copy2(SRC / "english" / name, OUT / name)
for name in ("references.bib", "method_equations.tex"):
    shutil.copy2(SRC / name, OUT / name)
for name in ("pgct_style.tex", "fig_architecture.tex", "fig_discovery.tex", "fig_critic.tex",
             "discovery_log.pdf", "baseline_ippo.pdf"):
    shutil.copy2(SRC / "figures" / name, OUT / "figures" / name)
leftover = [p.name for p in OUT.glob("*.tex") if "../" in p.read_text(encoding="utf-8")]
assert not leftover, leftover
with zipfile.ZipFile(SRC / "overleaf_ijcai27.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())
print("files", sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file()))
