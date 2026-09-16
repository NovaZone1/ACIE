#!/usr/bin/env python3
"""Score the complete locked R3 matrix on target data."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from acie.data import Bundle
from acie.io import read_json
from acie.vv_engine import score_locked_target
from acie.vv_trees import score_locked_tree
DIRECTIONS=('HUI360_to_SSUP-A','SSUP-A_to_HUI360');SPLITS=('A','B','C','D','E');SEEDS=(11,22,33,44,55)
def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1];p.add_argument('--root',type=Path,default=root);p.add_argument('--data',type=Path,default=Path('data/official_cross_domain.npz'));p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1/r3'));p.add_argument('--selection-root',type=Path,default=Path('outputs/vv_followup_v1/r2_cv'));p.add_argument('--device',default='cpu');a=p.parse_args();root=a.root.resolve();resolve=lambda x:x if x.is_absolute() else root/x;data,base,sel=map(resolve,(a.data,a.output,a.selection_root));lock=read_json(base/'TARGET_SCORING_LOCK.json')
 if lock.get('status')!='locked' or lock.get('checkpoint_count')!=210:raise ValueError('R3 target lock missing or incomplete')
 b=Bundle.load(data)
 for d in DIRECTIONS:
  selected=read_json(sel/d/'MODEL_SELECTION_LOCK.json');models=('G','F',selected['J_star'],selected['C_star'])
  for sp in SPLITS:
   for m in models:
    for seed in SEEDS:
     run=base/d/m/sp/str(seed);result=score_locked_target(b,run/'best.pt',run/'test_predictions.jsonl',a.device);print(json.dumps({'direction':d,'split':sp,'model':m,'seed':seed,'AP':result['metrics']['AP']}))
   tree=base/d/'histgb'/sp/'11';result=score_locked_tree(b,tree/'tree.joblib',tree/'test_predictions.jsonl');print(json.dumps({'direction':d,'split':sp,'model':'histgb','seed':11,'AP':result['metrics']['AP']}))
if __name__=='__main__':main()
