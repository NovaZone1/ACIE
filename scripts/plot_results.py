"""Plot only measured AP values; never fabricate or interpolate a result."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rows=json.loads(Path(a.summary).read_text());labels=sorted(set(f"{r['source']} → {r['target']}" for r in rows))
fig,ax=plt.subplots(figsize=(8,4))
for i,label in enumerate(labels):
    values=[r['AP'] for r in rows if f"{r['source']} → {r['target']}"==label]
    ax.scatter([i]*len(values),values,alpha=.7)
    ax.plot([i-.15,i+.15],[sum(values)/len(values)]*2,linewidth=2)
ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Average precision',ylim=(0,1))
ax.spines[['top','right']].set_visible(False);fig.tight_layout()
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out.with_suffix('.pdf'));fig.savefig(out.with_suffix('.png'),dpi=300)
