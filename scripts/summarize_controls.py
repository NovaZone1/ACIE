"""Summarize fixed-split control runs, including paired seed and ensemble deltas."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from acie.data import group_key
from acie.io import read_json,read_jsonl,write_json
from acie.metrics import binary_metrics,bootstrap

METHODS=['geometry','fusion_mlp','matched_resampling','fusion_pair','dual_no_pair','full']
COMPARISONS=[('fusion_mlp','geometry'),('matched_resampling','fusion_mlp'),
             ('fusion_pair','fusion_mlp'),('dual_no_pair','geometry'),
             ('full','geometry'),('full','dual_no_pair')]


def ensemble(root,domain,method,seeds):
    tables=[]
    for seed in seeds:
        rows=read_jsonl(root/domain/method/f'seed{seed}'/'predictions.jsonl')
        tables.append({r['sample_id']:r for r in rows})
    ids=sorted(tables[0]);
    if any(set(t)!=set(ids) for t in tables):raise ValueError('Prediction IDs differ across seeds')
    y=np.array([tables[0][i]['label'] for i in ids]);scores=np.mean([[t[i]['score'] for i in ids] for t in tables],0)
    meta=[tables[0][i] for i in ids]
    return ids,y,scores,meta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
    rows=read_json(a.runs/'runs.json');domains=sorted({r['domain'] for r in rows});result={'domains':{}}
    for domain in domains:
        domain_rows=[r for r in rows if r['domain']==domain];seeds=sorted({r['seed'] for r in domain_rows})
        payload={'seeds':seeds,'methods':{},'comparisons':{},'ensemble':{}}
        by={(r['method'],r['seed']):r for r in domain_rows}
        for method in METHODS:
            rs=[by[method,seed] for seed in seeds]
            values=np.array([r['AP'] for r in rs]);payload['methods'][method]={
                'AP_mean':float(values.mean()),'AP_std':float(values.std(ddof=1)),
                'AUROC_mean':float(np.mean([r['AUROC'] for r in rs])),
                'pair_accuracy_mean':float(np.mean([r['pair_accuracy'] for r in rs])),
                'matching_coverage_mean':float(np.mean([r['matching_coverage'] for r in rs])),
                'n_pairs_mean':float(np.mean([r['n_pairs'] for r in rs]))}
        for first,second in COMPARISONS:
            delta=np.array([by[first,s]['AP']-by[second,s]['AP'] for s in seeds])
            payload['comparisons'][f'{first}_minus_{second}']={'AP_delta_by_seed':delta.tolist(),
                'mean':float(delta.mean()),'std':float(delta.std(ddof=1)),
                'positive_seeds':int((delta>0).sum()),'n_seeds':len(delta)}
        ensembles={}
        for method in METHODS:
            ids,y,scores,meta=ensemble(a.runs,domain,method,seeds);ensembles[method]=(ids,y,scores,meta)
            payload['ensemble'][method]={'metrics':binary_metrics(y,scores)}
        for first,second in COMPARISONS:
            _,y,scores,meta=ensembles[first];other=ensembles[second][2]
            payload['ensemble'][first]['vs_'+second]=bootstrap(
                y,scores,[group_key(m) for m in meta],a.bootstrap,other=other)
        result['domains'][domain]=payload
    write_json(a.out,result);print(f'Wrote {a.out}')


if __name__=='__main__':main()
