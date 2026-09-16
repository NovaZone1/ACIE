#!/usr/bin/env python3
"""Run and select the registered R2 source-only neural development matrix."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from acie.data import Bundle
from acie.io import digest_file, read_json, write_json
from acie.vv_engine import train_source_model

DIRECTIONS=('HUI360_to_SSUP-A','SSUP-A_to_HUI360');MODELS=('G','Q','C64','Cm','Jc','Jw','JwD','F');FOLDS=('A','B','C');SEEDS=(11,22)
CANDIDATES=tuple({'order':i,'lr':lr,'weight_decay':wd} for i,(lr,wd) in enumerate(( (lr,wd) for lr in (.0003,.001,.003) for wd in (.00005,.0005) )))

def run_dir(base,direction,model,fold,candidate,seed):return base/direction/model/'cv'/fold/f"candidate_{candidate['order']}"/str(seed)
def select_candidate(base,direction,model):
 rows=[]
 for c in CANDIDATES:
  values=[]
  for fold in FOLDS:
   for seed in SEEDS:
    p=run_dir(base,direction,model,fold,c,seed)/'metrics.json'
    if not p.is_file():raise FileNotFoundError(p)
    values.append(float(read_json(p)['source_val']['AP']))
  rows.append({**c,'fold_seed_AP':values,'mean_source_CV_AP':float(np.mean(values))})
 best=max(r['mean_source_CV_AP'] for r in rows);selected=next(r for r in rows if best-r['mean_source_CV_AP']<=1e-6)
 return rows,selected

def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1]
 p.add_argument('--root',type=Path,default=root);p.add_argument('--data',type=Path,default=Path('data/official_cross_domain.npz'));p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1/r2_cv'));p.add_argument('--fold-root',type=Path,default=Path('protocols/vv_followup_v1/r2_folds'));p.add_argument('--phase',choices=('g','select-g','models','select-models'),required=True);p.add_argument('--directions',nargs='+',choices=DIRECTIONS,default=list(DIRECTIONS));p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS[1:]));p.add_argument('--device',default='cpu');p.add_argument('--num-threads',type=int,default=2);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();root=a.root.resolve();data=a.data if a.data.is_absolute() else root/a.data;base=a.output if a.output.is_absolute() else root/a.output;fold_root=a.fold_root if a.fold_root.is_absolute() else root/a.fold_root
 if a.phase in ('select-g','select-models'):
  for direction in a.directions:
   models=('G',) if a.phase=='select-g' else MODELS
   selections={};tables={}
   for model in models:
    rows,selected=select_candidate(base,direction,model);tables[model]=rows;selections[model]={k:selected[k] for k in ('order','lr','weight_decay','mean_source_CV_AP')}
   payload={'schema':'acie.vv-r2-neural-selection.v1','status':'locked','direction':direction,'selection_uses_target':False,'models':selections,'candidate_tables':tables}
   if a.phase=='select-models':
    def family(names,preference):
     best=max(selections[n]['mean_source_CV_AP'] for n in names)
     return next(n for n in preference if n in names and best-selections[n]['mean_source_CV_AP']<=1e-6)
    payload['J_star']=family(('Jw','JwD','Jc'),('Jw','JwD','Jc'));payload['C_star']=family(('C64','Cm'),('Cm','C64'))
   path=base/direction/('G_SELECTION_LOCK.json' if a.phase=='select-g' else 'MODEL_SELECTION_LOCK.json');write_json(path,payload);print(json.dumps({'direction':direction,'path':str(path),'selections':selections,'J_star':payload.get('J_star'),'C_star':payload.get('C_star')}))
  return
 models=('G',) if a.phase=='g' else tuple(dict.fromkeys(a.models))
 if a.phase=='models' and 'G' in models:raise ValueError('G is run and selected in its own phase')
 plans=[(direction,model,fold,c,seed) for direction in a.directions for model in models for fold in FOLDS for c in CANDIDATES for seed in SEEDS]
 if a.dry_run:
  print(json.dumps({'schema':'acie.vv-r2-cv-dry-run.v1','phase':a.phase,'target_scoring':False,'slots':len(plans),'plans':[{'direction':d,'model':m,'fold':f,'candidate':c,'seed':s,'out':str(run_dir(base,d,m,f,c,s))} for d,m,f,c,s in plans]},indent=2));return
 b=Bundle.load(data)
 for direction,model,fold,candidate,seed in plans:
  split_path=fold_root/direction/f'{fold}.json';split=read_json(split_path);geometry=None
  if model in ('F','Jw','JwD'):
   lock=read_json(base/direction/'G_SELECTION_LOCK.json');g_candidate=next(c for c in CANDIDATES if c['order']==lock['models']['G']['order']);geometry=run_dir(base,direction,'G',fold,g_candidate,seed)/'best.pt'
  config={'lr':candidate['lr'],'weight_decay':candidate['weight_decay'],'device':a.device,'num_threads':a.num_threads}
  out=run_dir(base,direction,model,fold,candidate,seed);context={'experiment':'r2_cv','split_id':f'inner_{fold}','fold_id':fold,'data_paths':[str(data.resolve()),str(split_path.resolve())],'data_hashes':[digest_file(data),digest_file(split_path)]}
  result=train_source_model(b,split,model,seed,out,config,geometry,a.resume,context)
  print(json.dumps({'direction':direction,'model':model,'fold':fold,'candidate':candidate['order'],'seed':seed,'status':result['status']}))
if __name__=='__main__':main()
