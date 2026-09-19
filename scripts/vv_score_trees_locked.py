#!/usr/bin/env python3
"""Score a complete locked VV tree matrix on target data."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from acie.data import Bundle
from acie.vv_trees import score_locked_tree
from acie.vv_provenance import validate_locked_checkpoint

def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1]
 p.add_argument('--root',type=Path,default=root);p.add_argument('--data',type=Path,default=Path('data/official_cross_domain.npz'));p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1'));p.add_argument('--experiment',default='r1')
 p.add_argument('--directions',nargs='+',default=['HUI360_to_SSUP-A','SSUP-A_to_HUI360']);p.add_argument('--models',nargs='+',choices=('histgb','random_forest'),default=['histgb','random_forest']);p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);p.add_argument('--lock',type=Path);p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();root=a.root.resolve();data=a.data if a.data.is_absolute() else root/a.data;output=a.output if a.output.is_absolute() else root/a.output
 plans=[]
 for direction in a.directions:
  for model in a.models:
   seeds=[a.seeds[0]] if model=='histgb' else a.seeds
   for seed in seeds:
    run=output/a.experiment/direction/model/'fixed'/str(seed);plans.append({'direction':direction,'model_id':model,'seed':seed,'checkpoint':run/'tree.joblib','prediction':run/'test_predictions.jsonl'})
 if a.dry_run:print(json.dumps({'training':False,'slots':len(plans),'plans':[{**p,'checkpoint':str(p['checkpoint']),'prediction':str(p['prediction'])} for p in plans]},indent=2));return
 missing=[str(p['checkpoint']) for p in plans if not p['checkpoint'].is_file()]
 if missing:raise FileNotFoundError(f'Refusing partial target scoring; {len(missing)} checkpoints missing. First: {missing[0]}')
 lock=a.lock if a.lock and a.lock.is_absolute() else root/a.lock if a.lock else output/a.experiment/'TARGET_SCORING_LOCK.json'
 b=Bundle.load(data)
 for plan in plans:
  validate_locked_checkpoint(lock,plan['checkpoint'],direction=plan['direction'],model_id=plan['model_id'],seed=plan['seed'])
  result=score_locked_tree(b,plan['checkpoint'],plan['prediction']);print(json.dumps({'checkpoint':str(plan['checkpoint']),'AP':result['metrics']['AP'],'status':'complete'}))
if __name__=='__main__':main()
