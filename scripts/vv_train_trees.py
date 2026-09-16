#!/usr/bin/env python3
"""Train/select VV tree references using source data only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from acie.data import Bundle
from acie.io import read_json
from acie.vv_trees import train_source_tree

DIRECTIONS={
 "HUI360_to_SSUP-A":"outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
 "SSUP-A_to_HUI360":"outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}

def main():
 p=argparse.ArgumentParser(); root=Path(__file__).resolve().parents[1]
 p.add_argument('--root',type=Path,default=root);p.add_argument('--data',type=Path,default=Path('data/official_cross_domain.npz'))
 p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1'));p.add_argument('--experiment',default='r1')
 p.add_argument('--directions',nargs='+',choices=tuple(DIRECTIONS),default=list(DIRECTIONS));p.add_argument('--models',nargs='+',choices=('histgb','random_forest'),default=['histgb','random_forest'])
 p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);p.add_argument('--protocol',choices=('r1','r2'),default='r1');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();root=a.root.resolve();data=a.data if a.data.is_absolute() else root/a.data;output=a.output if a.output.is_absolute() else root/a.output
 plans=[]
 for direction in a.directions:
  for model in a.models:
   seeds=[a.seeds[0]] if model=='histgb' else a.seeds
   for seed in seeds:plans.append((direction,model,seed,output/a.experiment/direction/model/'fixed'/str(seed)))
 if a.dry_run:
  print(json.dumps({'schema':'acie.vv-tree-dry-run.v1','target_scoring':False,'slots':len(plans),'plans':[{'direction':d,'model':m,'seed':s,'out':str(o)} for d,m,s,o in plans]},indent=2));return
 b=Bundle.load(data)
 for direction,model,seed,out in plans:
  result=train_source_tree(b,read_json(root/DIRECTIONS[direction]),out,model,seed,a.protocol,{'experiment':a.experiment})
  print(json.dumps({'direction':direction,'model':model,'seed':seed,'status':result['status'],'out':str(out)}))
if __name__=='__main__':main()
