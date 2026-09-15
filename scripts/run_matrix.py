"""Run four domain directions, fixed split and five configurable training seeds."""
import argparse
import copy
from pathlib import Path
import yaml
from acie.data import Bundle,make_split
from acie.engine import train,predict
from acie.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--domains',nargs=2,required=True)
    p.add_argument('--out',required=True);p.add_argument('--config',default='configs/full.yaml')
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);p.add_argument('--split-seed',type=int,default=42)
    a=p.parse_args();b=Bundle.load(a.data);cfg=yaml.safe_load(Path(a.config).read_text());summary=[]
    for source in a.domains:
        for target in a.domains:
            split=make_split(b,source,target,seed=a.split_seed)
            for seed in a.seeds:
                c=copy.deepcopy(cfg);c['seed']=seed;out=Path(a.out)/f'{source}_to_{target}'/f'seed{seed}'
                out.mkdir(parents=True,exist_ok=True);write_json(out/'split.json',split)
                train(b,split,c,out/'model');r=predict(b,out/'model'/'best.pt',out/'predictions.jsonl')
                summary.append({'source':source,'target':target,'seed':seed,**r['metrics'],'protocol':split['protocol']})
                write_json(Path(a.out)/'summary.json',summary)

if __name__=='__main__':main()
