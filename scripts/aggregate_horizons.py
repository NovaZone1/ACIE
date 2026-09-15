"""E4 analysis of separately exported, existing-label observation cutoffs.
Input: JSON list [{"seconds": 1.0, "predictions": "...jsonl"}, ...].
Same frozen source threshold must be used. No synthetic interpolation or re-labeling.
"""
import argparse
from collections import defaultdict
from pathlib import Path
from acie.io import read_json,read_jsonl,write_json
from acie.data import track_key
from acie.metrics import binary_metrics
p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--out',required=True);a=p.parse_args()
index=read_json(a.index);runs=[]
for item in index:
    rows=read_jsonl(item['predictions']);groups=defaultdict(list)
    for r in rows:groups[track_key(r)].append(r)
    if any(len(v)!=1 for v in groups.values()):raise ValueError('E4 expects exactly one deterministic prefix per track per horizon')
    runs.append((float(item['seconds']),{k:v[0] for k,v in groups.items()}))
common=set.intersection(*(set(r) for _,r in runs));result=[]
thresholds={r['threshold'] for _,rows in runs for r in rows.values()}
if len(thresholds)!=1:raise ValueError('Use one source-frozen threshold across horizon tests')
threshold=next(iter(thresholds))
for seconds,rows in sorted(runs):
    full=list(rows.values());paired=[rows[k] for k in sorted(common)]
    result.append({'seconds':seconds,'complete':binary_metrics([r['label'] for r in full],[r['score'] for r in full],threshold),
                   'common_tracks':binary_metrics([r['label'] for r in paired],[r['score'] for r in paired],threshold) if paired else None})
write_json(a.out,{'common_track_count':len(common),'horizons':result,'protocol':'supplemental horizon analysis, not native leaderboard'})
