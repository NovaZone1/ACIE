"""Tune pair-radius quantiles on source validation only; never score test."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np
import yaml
from acie.data import Bundle
from acie.engine import train,predict
from acie.io import read_json,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True)
    p.add_argument('--splits-root',required=True,type=Path);p.add_argument('--lambda-root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--config',default='configs/full.yaml',type=Path)
    p.add_argument('--domains',nargs='+',required=True);p.add_argument('--quantiles',nargs='+',type=float,default=[.25,.5,.75,.9,1.])
    p.add_argument('--pair-lambda',type=float,default=.03);p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    p.add_argument('--bootstrap',type=int,default=300);a=p.parse_args();quantiles=sorted(set(a.quantiles))
    if any(not 0<=q<=1 for q in quantiles):raise ValueError('Quantiles must be in [0,1]')
    baseline_rows=read_json(a.lambda_root/'trials.json')
    baseline={(r['domain'],r['seed']):r['val_AP'] for r in baseline_rows if r['lambda']==0.0}
    bundle=Bundle.load(a.data);base=yaml.safe_load(a.config.read_text());rows=[]
    if (a.out/'trials.json').is_file():rows=read_json(a.out/'trials.json')
    for domain in a.domains:
        split=read_json(a.splits_root/domain/'split.json')
        for quantile in quantiles:
            for seed in a.seeds:
                run=a.out/domain/f'quantile{quantile:g}'/f'seed{seed}';run.mkdir(parents=True,exist_ok=True)
                cfg=copy.deepcopy(base);cfg.update(kind='full',seed=seed,pair_lambda=a.pair_lambda,class_weighted=True)
                cfg['matching']=copy.deepcopy(cfg['matching']);cfg['matching']['radius_quantile']=quantile
                report_path=run/'model'/'training_report.json'
                if report_path.is_file() and (run/'model'/'best.pt').is_file():report=read_json(report_path)
                else:report=train(bundle,split,cfg,run/'model',resume=(run/'model'/'run.json').is_file())
                metrics_path=run/'val_predictions.metrics.json'
                if metrics_path.is_file():validation=read_json(metrics_path)
                else:validation=predict(bundle,run/'model'/'best.pt',run/'val_predictions.jsonl','val',repeats=a.bootstrap)
                matching=read_json(run/'model'/'matching.json');val_ap=report['source_val']['AP']
                rows=[r for r in rows if (r['domain'],r['quantile'],r['seed'])!=(domain,quantile,seed)]
                rows.append({'domain':domain,'quantile':quantile,'pair_lambda':a.pair_lambda,'seed':seed,
                    'class_weighted':True,'val_AP':val_ap,'val_AUROC':report['source_val']['AUROC'],
                    'delta_from_lambda0':val_ap-baseline[domain,seed],
                    'train_pair_coverage':matching['positive_coverage'],'train_n_pairs':matching['n_pairs'],
                    'val_pair_coverage':validation['pair_diagnostic']['positive_coverage'],
                    'pair_accuracy':validation['pair_diagnostic']['final_accuracy'],
                    'evidence_pair_accuracy':validation['pair_diagnostic']['evidence_accuracy'],'test_evaluated':False})
                summaries=[]
                for d in a.domains:
                    for q in quantiles:
                        subset=[r for r in rows if r['domain']==d and r['quantile']==q]
                        if len(subset)!=len(a.seeds):continue
                        summaries.append({'domain':d,'quantile':q,'n_runs':len(subset),
                            'val_AP_mean':float(np.mean([r['val_AP'] for r in subset])),
                            'val_AP_std':float(np.std([r['val_AP'] for r in subset],ddof=1)),
                            'delta_from_lambda0_mean':float(np.mean([r['delta_from_lambda0'] for r in subset])),
                            'positive_seeds_vs_lambda0':int(sum(r['delta_from_lambda0']>0 for r in subset)),
                            'train_pair_coverage_mean':float(np.mean([r['train_pair_coverage'] for r in subset])),
                            'train_n_pairs_mean':float(np.mean([r['train_n_pairs'] for r in subset])),
                            'pair_accuracy_mean':float(np.mean([r['pair_accuracy'] for r in subset]))})
                selected={d:max([r for r in summaries if r['domain']==d],key=lambda x:x['val_AP_mean'])
                          for d in a.domains if any(r['domain']==d for r in summaries)}
                joint=[]
                for q in quantiles:
                    available=[r for r in summaries if r['quantile']==q]
                    if len(available)==len(a.domains):joint.append({'quantile':q,'mean_domain_delta_from_lambda0':
                        float(np.mean([r['delta_from_lambda0_mean'] for r in available]))})
                write_json(a.out/'trials.json',rows);write_json(a.out/'source_validation_summary.json',{
                    'protocol':'source validation only','test_evaluated':False,'class_weighted':True,
                    'pair_lambda':a.pair_lambda,'quantiles':quantiles,'seeds':a.seeds,'summaries':summaries,
                    'selected_by_domain':selected,'joint_selection':max(joint,key=lambda x:x['mean_domain_delta_from_lambda0']) if joint else None})
                print(f'{domain} q={quantile:g} seed={seed} val_AP={val_ap:.4f} coverage={matching["positive_coverage"]:.3f}',flush=True)


if __name__=='__main__':main()
