#!/usr/bin/env python3
"""Validate all R3 source-only runs before target scoring."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from acie.io import digest_file,read_json,write_json
from acie.vv_engine import _tensor_state_digest

DIRECTIONS=('HUI360_to_SSUP-A','SSUP-A_to_HUI360');SPLITS=('A','B','C','D','E');SEEDS=(11,22,33,44,55)
def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1];p.add_argument('--root',type=Path,default=root);p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1/r3'));p.add_argument('--selection-root',type=Path,default=Path('outputs/vv_followup_v1/r2_cv'));a=p.parse_args();root=a.root.resolve();resolve=lambda x:x if x.is_absolute() else root/x;base,sel=map(resolve,(a.output,a.selection_root));errors=[];checkpoints=[]
 for d in DIRECTIONS:
  lock=read_json(sel/d/'MODEL_SELECTION_LOCK.json');models=('G','F',lock['J_star'],lock['C_star'])
  for sp in SPLITS:
   for seed in SEEDS:
    gpath=base/d/'G'/sp/str(seed)/'best.pt'
    if not gpath.is_file():errors.append(f'missing {gpath}');continue
    g=torch.load(gpath,map_location='cpu',weights_only=True);gd=_tensor_state_digest(g['geometry_state']);gh=digest_file(gpath);shared=[]
    for m in models:
     run_dir=base/d/m/sp/str(seed);required=('run.json','best.pt','val_predictions.jsonl','metrics.json','timing.json','initial_state.json');missing=[x for x in required if not (run_dir/x).is_file()]
     if missing:errors.append(f'{d}/{sp}/{m}/{seed} missing {missing}');continue
     if (run_dir/'test_predictions.jsonl').exists():errors.append(f'{d}/{sp}/{m}/{seed} target prediction exists')
     run=read_json(run_dir/'run.json');ck=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=True)
     if run.get('status')!='complete' or run.get('target_scoring_performed') is not False:errors.append(f'{d}/{sp}/{m}/{seed} invalid status')
     if m in ('F','Jw','JwD'):
      ref=read_json(run_dir/'geometry_reference.json')
      if ref['sha256']!=gh or ref['state_sha256']!=gd:errors.append(f'{d}/{sp}/{m}/{seed} G mismatch')
      shared.append(read_json(run_dir/'initial_state.json'))
     if m=='F' and _tensor_state_digest(ck['geometry_state'])!=gd:errors.append(f'{d}/{sp}/F/{seed} frozen G changed')
     checkpoints.append({'direction':d,'split':sp,'model_id':m,'seed':seed,'path':str((run_dir/'best.pt').resolve()),'sha256':digest_file(run_dir/'best.pt')})
    if len(shared)==2:
     for field in ('behavior_state_sha256','first_batch_ids_sha256','first_batch_initial_logits_sha256'):
      if shared[0][field]!=shared[1][field]:errors.append(f'{d}/{sp}/{seed} F/J* differ for {field}')
   tree=base/d/'histgb'/sp/'11';required=('run.json','tree.joblib','val_predictions.jsonl','metrics.json','timing.json');missing=[x for x in required if not (tree/x).is_file()]
   if missing:errors.append(f'{d}/{sp}/histgb missing {missing}')
   elif (tree/'test_predictions.jsonl').exists():errors.append(f'{d}/{sp}/histgb target prediction exists')
   else:checkpoints.append({'direction':d,'split':sp,'model_id':'histgb','seed':11,'path':str((tree/'tree.joblib').resolve()),'sha256':digest_file(tree/'tree.joblib')})
 report={'schema':'acie.vv-r3-source-validation.v1','status':'passed' if not errors else 'failed','expected_slots':210,'observed_slots':len(checkpoints),'errors':errors};write_json(base/'source_validation.json',report)
 if errors:print(json.dumps(report,indent=2));raise SystemExit(1)
 lock={'schema':'acie.vv-r3-target-scoring-lock.v1','status':'locked','checkpoint_count':len(checkpoints),'checkpoints':checkpoints,'selection':'R2 identities and hyperparameters; five registered R3 source-date splits; no target selection'};write_json(base/'TARGET_SCORING_LOCK.json',lock);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
