"""Check that the English and Korean papers share equations, citations, labels, figures and numbers,
and that the Beamer, PowerPoint and script decks have the same slide titles in the same order."""
from pathlib import Path
import re
import sys

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[2]
en = (ROOT / "paper/english/main.tex").read_text(encoding="utf-8")
ko = (ROOT / "paper/korean/main.tex").read_text(encoding="utf-8")
CHECKS = {"equations": r"\\(Model|RewardLoss|Probe|Distance|Approval|Donor|Max|GAE|Blend|PPO|Bound)Equation",
          "cites": r"\\citep?\{([^}]*)\}", "labels": r"\\label\{([^}]*)\}", "refs": r"\\ref\{([^}]*)\}",
          "figures": r"\{\.\./figures/([^}]*)\}"}
ok = True
for name, pat in CHECKS.items():
    a, b = sorted(set(re.findall(pat, en))), sorted(set(re.findall(pat, ko)))
    same = a == b
    ok &= same
    print(f"{name:10s}", "same" if same else f"DIFF en-only {set(a) - set(b)} ko-only {set(b) - set(a)}")


def numbers(s):
    s = re.sub(r"\\(label|ref|citep|input|includegraphics|path|bibliography|usepackage|documentclass)(\[[^]]*\])?\{[^}]*\}", "", s)
    s = s.replace("{,}", ",")
    return set(re.findall(r"(?<![A-Za-z_\\])\d+(?:[.,]\d+)*(?:--\d+(?:\.\d+)?)?", s))


a, b = numbers(en), numbers(ko)
print("numbers only in EN:", sorted(a - b))
print("numbers only in KO:", sorted(b - a))

beamer = (ROOT / "presentation/pgct_beamer.tex").read_text(encoding="utf-8")
prs = Presentation(ROOT / "presentation/pgct_presentation.pptx")
script = (ROOT / "presentation/pgct_script_ko.md").read_text(encoding="utf-8")
pptx_titles = []
for slide in prs.slides:
    texts = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    pptx_titles.append(texts[0] if texts else "")
script_titles = re.findall(r"^## \d+\. (.+)$", script, flags=re.M)
frames = beamer.count(r"\begin{frame}")
print("slides: beamer", frames, "pptx", len(prs.slides), "script", len(script_titles))
order_ok = pptx_titles == script_titles and frames == len(prs.slides)
print("titles in the same order:", order_ok)
missing = [t for t in script_titles if t.replace("&", r"\&") not in beamer]
print("titles missing from beamer:", missing)
sys.exit(0 if ok and order_ok and not missing else 1)
