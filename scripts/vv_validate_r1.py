#!/usr/bin/env python3
"""Validate the complete source-only R1 matrix and create the target scoring lock."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from acie.io import digest_file, digest_object, read_json, read_jsonl, write_json
from acie.vv_models import EXPECTED_PARAMETERS

DIRECTIONS=("HUI360_to_SSUP-A","SSUP-A_to_HUI360")
MODELS=("G","Q","C64","Cm","Jc","Jw","JwD","F")
SEEDS=(11,22,33,44,55)

def state_digest(state):
 h=hashlib.sha256()
 for name,value in sorted(state.items()):
  array=value.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1]
 p.add_argument('--root',type=Path,default=root);p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1'));p.add_argument('--experiment',default='r1');p.add_argument('--write-lock',action='store_true')
 a=p.parse_args();root=a.root.resolve();out=a.output if a.output.is_absolute() else root/a.output;base=out/a.experiment
 errors=[];checks=[];checkpoints=[]
 for direction in DIRECTIONS:
  val_ids_by_seed={}
  for seed in SEEDS:
   gdir=base/direction/'G'/'fixed'/str(seed);gpath=gdir/'best.pt'
   if not gpath.is_file():errors.append(f'missing {gpath}');continue
   g=torch.load(gpath,map_location='cpu',weights_only=True);g_digest=state_digest(g['geometry_state']);g_hash=digest_file(gpath)
   shared=[]
   for model_id in MODELS:
    run_dir=base/direction/model_id/'fixed'/str(seed);required=('run.json','config.yaml','split.json','scaler.json','initial_state.json','history.jsonl','best.pt','val_predictions.jsonl','metrics.json','timing.json')
    missing=[name for name in required if not (run_dir/name).is_file()]
    if missing:errors.append(f'{direction}/{model_id}/{seed}: missing {missing}');continue
    if (run_dir/'test_predictions.jsonl').exists():errors.append(f'{direction}/{model_id}/{seed}: target prediction exists before lock')
    run=read_json(run_dir/'run.json');ck=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=True);initial=read_json(run_dir/'initial_state.json');timing=read_json(run_dir/'timing.json')
    if run.get('status')!='complete' or run.get('target_scoring_performed') is not False:errors.append(f'{direction}/{model_id}/{seed}: invalid run status')
    if ck['model_spec']['model_id']!=model_id or ck['master_seed']!=seed:errors.append(f'{direction}/{model_id}/{seed}: checkpoint identity mismatch')
    if timing['effective_parameters']!=EXPECTED_PARAMETERS[model_id]:errors.append(f'{direction}/{model_id}/{seed}: parameter count mismatch')
    ids=[row['sample_id'] for row in read_jsonl(run_dir/'val_predictions.jsonl')]
    if seed not in val_ids_by_seed:val_ids_by_seed[seed]=ids
    elif ids!=val_ids_by_seed[seed]:errors.append(f'{direction}/{model_id}/{seed}: source val IDs/order differ')
    if model_id in ('F','Jw','JwD'):
     reference=read_json(run_dir/'geometry_reference.json')
     if reference['sha256']!=g_hash or reference['state_sha256']!=g_digest:errors.append(f'{direction}/{model_id}/{seed}: shared G mismatch')
     shared.append((model_id,initial))
     if model_id=='F':
      if state_digest(ck['geometry_state'])!=g_digest:errors.append(f'{direction}/F/{seed}: frozen G changed')
      if timing['trainable_final_stage_parameters']!=35393:errors.append(f'{direction}/F/{seed}: trainable count mismatch')
    checkpoints.append({'direction':direction,'model_id':model_id,'seed':seed,'path':str((run_dir/'best.pt').resolve()),'sha256':digest_file(run_dir/'best.pt')})
   if len(shared)==3:
    for field in ('behavior_state_sha256','first_batch_ids_sha256','first_batch_initial_logits_sha256','classification_batch_seed','classification_dropout_seed'):
     values={item[1][field] for item in shared}
     if len(values)!=1:errors.append(f'{direction}/{seed}: F/Jw/JwD differ for {field}')
  for model_id,seeds in (('histgb',(11,)),('random_forest',SEEDS)):
   for seed in seeds:
    run_dir=base/direction/model_id/'fixed'/str(seed)
    required=('run.json','split.json','scaler.json','tree.joblib','val_predictions.jsonl','candidate_results.json','metrics.json','timing.json')
    missing=[name for name in required if not (run_dir/name).is_file()]
    if missing:errors.append(f'{direction}/{model_id}/{seed}: missing {missing}');continue
    run=read_json(run_dir/'run.json')
    if run.get('status')!='complete' or run.get('target_scoring_performed') is not False or run.get('feature_dim')!=389:errors.append(f'{direction}/{model_id}/{seed}: invalid source-only tree run')
    if (run_dir/'test_predictions.jsonl').exists():errors.append(f'{direction}/{model_id}/{seed}: target prediction exists before lock')
    checkpoints.append({'direction':direction,'model_id':model_id,'seed':seed,'path':str((run_dir/'tree.joblib').resolve()),'sha256':digest_file(run_dir/'tree.joblib')})
 report={'schema':'acie.vv-r1-source-validation.v1','status':'passed' if not errors else 'failed','expected_neural_slots':80,'expected_tree_slots':12,'observed_checkpoint_slots':len(checkpoints),'errors':errors,'checks':{'target_predictions_absent':not any('target prediction exists' in e for e in errors),'shared_geometry_and_initialization':not any('shared G' in e or 'differ for' in e or 'frozen G' in e for e in errors),'source_only_runs_complete':len(checkpoints)==92 and not errors}}
 write_json(base/'source_validation.json',report)
 if errors:
  print(json.dumps(report,indent=2));raise SystemExit(1)
 if a.write_lock:
  protocol=root/'protocols/vv_followup_v1/protocol_lock.json'
  lock={'schema':'acie.vv-r1-target-scoring-lock.v1','status':'locked','experiment':a.experiment,'created_at':'2026-09-16','protocol_path':str(protocol.resolve()),'protocol_sha256':digest_file(protocol),'checkpoint_count':len(checkpoints),'checkpoints':checkpoints,'selection':'R1 fixed registered models/configurations; no target-based model selection','target_scoring_command':'scripts/vv_score_locked.py and scripts/vv_score_trees_locked.py'}
  lock['lock_digest']=digest_object(lock);write_json(base/'TARGET_SCORING_LOCK.json',lock)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
