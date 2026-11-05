from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
slides=json.loads((ROOT/'presentation/slides.json').read_text(encoding='utf-8'))
for i,s in enumerate(slides,1):
    for j,eq in enumerate(s.get('equations',[])):
        eq=eq.replace(r'\mathcal Q',r'\mathcal{Q}').replace(r'\mathcal G',r'\mathcal{G}')
        fig=plt.figure(figsize=(12,1.1),facecolor='none')
        fig.text(.5,.5,'$'+eq+'$',ha='center',va='center',fontsize=23,color='#171717')
        fig.savefig(ROOT/f'presentation/figures/equation_{i:02d}_{j}.png',dpi=210,transparent=True,bbox_inches='tight',pad_inches=.06)
        plt.close(fig)
