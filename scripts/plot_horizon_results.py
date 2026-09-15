#!/usr/bin/env python3
"""Plot measured E4 common-track AP curves from the frozen summary."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from acie.io import read_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--summary',type=Path,required=True);p.add_argument('--out-prefix',type=Path,required=True)
a=p.parse_args();s=read_json(a.summary)
fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),sharey=False)
for ax,(direction,result) in zip(axes,s['directions'].items(),strict=True):
    seconds=[h['cutoff_seconds'] for h in result['horizons']]
    for method,label,color,marker in [('geometry','Geometry','#666666','o'),('dual_no_pair','Geometry + behavior residual','#1769aa','s')]:
        ap=[h['sets']['common_tracks']['methods'][method]['ensemble_ranking']['AP'] for h in result['horizons']]
        ax.plot(seconds,ap,label=label,color=color,marker=marker,linewidth=2,markersize=5)
    ax.set_title(direction.replace('_to_',' → '))
    ax.set_xlabel('Nominal anchor lead time (s)')
    ax.set_ylabel('Average precision (common tracks)')
    ax.set_xticks(seconds);ax.grid(alpha=.25)
    ax.text(.02,.03,f"common n={result['common_track_count']}, positives={result['common_positives']}",transform=ax.transAxes,fontsize=8)
axes[0].legend(frameon=False,fontsize=8)
fig.tight_layout();a.out_prefix.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.out_prefix.with_suffix('.png'),dpi=300,bbox_inches='tight')
fig.savefig(a.out_prefix.with_suffix('.pdf'),bbox_inches='tight')
