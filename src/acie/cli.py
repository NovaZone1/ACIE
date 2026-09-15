from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from .io import read_json,write_json,read_jsonl


def main():
    p=argparse.ArgumentParser(description='ACIE research code. See README for protocol and data requirements.')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo',help='CPU synthetic integration test; no benchmark data');d.add_argument('--out',required=True)
    d=sub.add_parser('prepare');d.add_argument('--manifest',required=True);d.add_argument('--out',required=True)
    d=sub.add_parser('split');d.add_argument('--data',required=True);d.add_argument('--source',required=True)
    d.add_argument('--target',required=True);d.add_argument('--out',required=True);d.add_argument('--seed',type=int,default=42)
    d.add_argument('--val-fraction',type=float,default=.2)
    d=sub.add_parser('train');d.add_argument('--data',required=True);d.add_argument('--split',required=True)
    d.add_argument('--config',required=True);d.add_argument('--out',required=True);d.add_argument('--resume',action='store_true')
    d=sub.add_parser('predict');d.add_argument('--data',required=True);d.add_argument('--checkpoint',required=True)
    d.add_argument('--out',required=True);d.add_argument('--partition',choices=['val','test'],default='test')
    d.add_argument('--device',default='cpu');d.add_argument('--bootstrap',type=int,default=1000)
    d=sub.add_parser('tree');d.add_argument('--data',required=True);d.add_argument('--split',required=True)
    d.add_argument('--out',required=True);d.add_argument('--kind',choices=['histgb','random_forest'],default='histgb')
    d.add_argument('--seed',type=int,default=42)
    d=sub.add_parser('compare');d.add_argument('--first',required=True);d.add_argument('--second',required=True)
    d.add_argument('--out',required=True);d.add_argument('--bootstrap',type=int,default=1000)
    d=sub.add_parser('audit');d.add_argument('--data',required=True);d.add_argument('--split')
    a=p.parse_args()
    from .data import Bundle,prepare,make_split,resolve_split
    if a.command=='demo':
        from .demo import run_demo
        result=run_demo(a.out)
    elif a.command=='prepare':
        b=prepare(a.manifest,a.out);result={'samples':len(b.y),'a_shape':list(b.a.shape),'q_shape':list(b.q.shape)}
    elif a.command=='split':
        result=make_split(Bundle.load(a.data),a.source,a.target,a.seed,a.val_fraction);write_json(a.out,result)
    elif a.command=='train':
        from .engine import train
        config=yaml.safe_load(Path(a.config).read_text()) or {}
        result=train(Bundle.load(a.data),read_json(a.split),config,a.out,a.resume)
    elif a.command=='predict':
        from .engine import predict
        result=predict(Bundle.load(a.data),a.checkpoint,a.out,a.partition,a.device,a.bootstrap)
    elif a.command=='tree':
        from .baselines import run_tree
        result=run_tree(Bundle.load(a.data),read_json(a.split),a.out,a.kind,a.seed)
    elif a.command=='compare':
        from .engine import compare_predictions
        result=compare_predictions(a.first,a.second,a.bootstrap);write_json(a.out,result)
    elif a.command=='audit':
        b=Bundle.load(a.data);result={'samples':len(b.y),'domains':sorted(set(m['dataset'] for m in b.meta)),
                                    'positives':int(b.y.sum()),'provenance':b.provenance}
        if a.split:result['split_counts']={k:len(v) for k,v in resolve_split(b,read_json(a.split)).items()}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
