"""Tune class-weighted pair loss on source validation only; never score test."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np
import yaml
from acie.data import Bundle
from acie.engine import train,predict
from acie.io import read_json,write_json


def summarize(rows,lambdas,domains,seeds):
    summaries=[]
    for domain in domains:
        zero={(r['seed']):r for r in rows if r['domain']==domain and r['lambda']==0.0}
        for lam in lambdas:
            subset=sorted([r for r in rows if r['domain']==domain and r['lambda']==lam],key=lambda x:x['seed'])
            if len(subset)!=len(seeds):continue
            ap=np.array([r['val_AP'] for r in subset]);delta=np.array([r['val_AP']-zero[r['seed']]['val_AP'] for r in subset])
            summaries.append({'domain':domain,'lambda':lam,'n_runs':len(subset),
                'val_AP_mean':float(ap.mean()),'val_AP_std':float(ap.std(ddof=1)),
                'delta_from_lambda0_mean':float(delta.mean()),'delta_by_seed':delta.tolist(),
                'positive_seeds_vs_lambda0':int((delta>0).sum()),
                'pair_accuracy_mean':float(np.mean([r['pair_accuracy'] for r in subset])),
                'evidence_pair_accuracy_mean':float(np.mean([r['evidence_pair_accuracy'] for r in subset]))})
    selected={}
    for domain in domains:
        available=[r for r in summaries if r['domain']==domain]
        if available:selected[domain]=max(available,key=lambda x:(x['val_AP_mean'],-x['lambda']))
    joint=[]
    for lam in lambdas:
        available=[r for r in summaries if r['lambda']==lam]
        if len(available)==len(domains):joint.append({'lambda':lam,
            'mean_domain_delta_from_lambda0':float(np.mean([r['delta_from_lambda0_mean'] for r in available]))})
    return summaries,selected,max(joint,key=lambda x:(x['mean_domain_delta_from_lambda0'],-x['lambda'])) if joint else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True)
    p.add_argument('--splits-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--config',default='configs/full.yaml',type=Path);p.add_argument('--domains',nargs='+',required=True)
    p.add_argument('--lambdas',nargs='+',type=float,default=[0.,.01,.03,.1,.3])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);p.add_argument('--bootstrap',type=int,default=300)
    a=p.parse_args();lambdas=sorted(set(a.lambdas))
    if 0. not in lambdas or any(x<0 for x in lambdas):raise ValueError('Lambda grid must contain 0 and no negatives')
    bundle=Bundle.load(a.data);base=yaml.safe_load(a.config.read_text());rows=[]
    if (a.out/'trials.json').is_file():rows=read_json(a.out/'trials.json')
    for domain in a.domains:
        split=read_json(a.splits_root/domain/'split.json')
        for lam in lambdas:
            for seed in a.seeds:
                run=a.out/domain/f'lambda{lam:g}'/f'seed{seed}';run.mkdir(parents=True,exist_ok=True)
                cfg=copy.deepcopy(base);cfg.update(kind='full',seed=seed,pair_lambda=lam,class_weighted=True)
                report_path=run/'model'/'training_report.json'
                if report_path.is_file() and (run/'model'/'best.pt').is_file():report=read_json(report_path)
                else:report=train(bundle,split,cfg,run/'model',resume=(run/'model'/'run.json').is_file())
                metrics_path=run/'val_predictions.metrics.json'
                if metrics_path.is_file():validation=read_json(metrics_path)
                else:validation=predict(bundle,run/'model'/'best.pt',run/'val_predictions.jsonl','val',repeats=a.bootstrap)
                rows=[r for r in rows if (r['domain'],r['lambda'],r['seed'])!=(domain,lam,seed)]
                rows.append({'domain':domain,'lambda':lam,'seed':seed,'class_weighted':True,
                    'val_AP':report['source_val']['AP'],'val_AUROC':report['source_val']['AUROC'],
                    'pair_accuracy':validation['pair_diagnostic']['final_accuracy'],
                    'evidence_pair_accuracy':validation['pair_diagnostic']['evidence_accuracy'],
                    'matching_coverage':validation['pair_diagnostic']['positive_coverage'],
                    'test_evaluated':False})
                summaries,selected,joint=summarize(rows,lambdas,a.domains,a.seeds)
                write_json(a.out/'trials.json',rows);write_json(a.out/'source_validation_summary.json',{
                    'protocol':'source validation only','test_evaluated':False,'class_weighted':True,
                    'lambdas':lambdas,'seeds':a.seeds,'summaries':summaries,
                    'selected_by_domain':selected,'joint_selection':joint})
                print(f'{domain} lambda={lam:g} seed={seed} val_AP={report["source_val"]["AP"]:.4f}',flush=True)


if __name__=='__main__':main()
