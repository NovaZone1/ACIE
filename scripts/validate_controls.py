"""Validate completeness and split identity of a six-control result directory."""
from __future__ import annotations
import argparse
from pathlib import Path
from acie.data import Bundle,resolve_split,group_key
from acie.io import read_json,read_jsonl,write_json

METHODS=['geometry','fusion_mlp','matched_resampling','fusion_pair','dual_no_pair','full']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True)
    p.add_argument('--runs',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    bundle=Bundle.load(a.data);rows=read_json(a.runs/'runs.json')
    keys={(r['domain'],r['method'],r['seed']) for r in rows}
    if len(keys)!=len(rows):raise ValueError('Duplicate run summaries')
    domains=sorted({r['domain'] for r in rows});seeds=sorted({r['seed'] for r in rows})
    expected={(d,m,s) for d in domains for m in METHODS for s in seeds}
    if keys!=expected:raise ValueError(f'Run matrix mismatch: missing={expected-keys}, extra={keys-expected}')
    splits={};checked=0
    for domain in domains:
        split=read_json(a.runs/domain/'split.json');ix=resolve_split(bundle,split)
        split_ids={name:{bundle.meta[i]['sample_id'] for i in values} for name,values in ix.items()}
        splits[domain]={name:{'n':len(values),'positives':int(bundle.y[values].sum()),
                              'groups':sorted({group_key(bundle.meta[i]) for i in values})}
                        for name,values in ix.items()}
        for method in METHODS:
            for seed in seeds:
                run=a.runs/domain/method/f'seed{seed}'
                state=read_json(run/'model'/'run.json')
                if state.get('status')!='complete':raise ValueError(f'Incomplete model: {run}')
                predictions=read_jsonl(run/'predictions.jsonl')
                if {r['sample_id'] for r in predictions}!=split_ids['test']:
                    raise ValueError(f'Prediction/test mismatch: {run}')
                if len(predictions)!=len(split_ids['test']):raise ValueError(f'Duplicate predictions: {run}')
                checked+=1
    write_json(a.out,{'schema':'acie.control-validation.v1','status':'passed','bundle_samples':len(bundle.y),
                      'domains':domains,'methods':METHODS,'seeds':seeds,'runs_checked':checked,'splits':splits})
    print(f'Validated {checked} complete runs')


if __name__=='__main__':main()
