"""Run the six ACIE source-development controls on one fixed split per domain."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np
import yaml
from acie.data import Bundle,make_split
from acie.engine import train,predict
from acie.io import read_json,write_json

CONTROLS={
    'geometry':'configs/geometry.yaml',
    'fusion_mlp':'configs/fusion_mlp.yaml',
    'matched_resampling':'configs/resampled.yaml',
    'fusion_pair':'configs/fusion_pair.yaml',
    'dual_no_pair':'configs/dual_no_pair.yaml',
    'full':'configs/full.yaml',
}


def aggregate(rows):
    result=[]
    for domain in sorted({r['domain'] for r in rows}):
        for method in CONTROLS:
            subset=[r for r in rows if r['domain']==domain and r['method']==method]
            if not subset:continue
            item={'domain':domain,'method':method,'seeds':[r['seed'] for r in subset],'n_runs':len(subset)}
            for metric in ['AP','AUROC','false_positive_rate','recall']:
                values=np.array([r[metric] for r in subset if r[metric] is not None],dtype=float)
                item[metric+'_mean']=float(values.mean()) if len(values) else None
                item[metric+'_std']=float(values.std(ddof=1)) if len(values)>1 else 0.
            item['pair_accuracy_mean']=float(np.mean([r['pair_accuracy'] for r in subset
                                                      if r['pair_accuracy'] is not None]))
            result.append(item)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True)
    p.add_argument('--domains',nargs='+',required=True);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    p.add_argument('--split-seed',type=int,default=42);p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--val-fraction',type=float,default=.2);p.add_argument('--test-fraction',type=float,default=.2)
    a=p.parse_args();bundle=Bundle.load(a.data)
    rows=read_json(a.out/'runs.json') if (a.out/'runs.json').is_file() else []
    for domain in a.domains:
        split=make_split(bundle,domain,domain,a.split_seed,a.val_fraction,a.test_fraction)
        domain_out=a.out/domain;domain_out.mkdir(parents=True,exist_ok=True);write_json(domain_out/'split.json',split)
        for method,config_path in CONTROLS.items():
            base=yaml.safe_load(Path(config_path).read_text())
            for seed in a.seeds:
                cfg=copy.deepcopy(base);cfg['seed']=seed;run=domain_out/method/f'seed{seed}'
                run.mkdir(parents=True,exist_ok=True);write_json(run/'split.json',split)
                metrics_path=run/'predictions.metrics.json'
                if metrics_path.is_file() and (run/'model'/'best.pt').is_file():
                    result=read_json(metrics_path)
                else:
                    train(bundle,split,cfg,run/'model',resume=(run/'model'/'run.json').is_file())
                    result=predict(bundle,run/'model'/'best.pt',run/'predictions.jsonl',repeats=a.bootstrap)
                matching=read_json(run/'model'/'matching.json')
                rows=[r for r in rows if (r['domain'],r['method'],r['seed'])!=(domain,method,seed)]
                rows.append({'domain':domain,'method':method,'seed':seed,**result['metrics'],
                             'pair_accuracy':result['pair_diagnostic']['final_accuracy'],
                             'matching_coverage':matching['positive_coverage'],'n_pairs':matching['n_pairs']})
                write_json(a.out/'runs.json',rows);write_json(a.out/'summary.json',aggregate(rows))
                print(f'{domain} {method} seed={seed} AP={result["metrics"]["AP"]:.4f}')


if __name__=='__main__':main()
