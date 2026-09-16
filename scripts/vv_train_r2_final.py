#!/usr/bin/env python3
"""Train the five-seed R2 neural matrix from source-only selection locks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from acie.data import Bundle
from acie.io import digest_file,read_json
from acie.vv_engine import train_source_model

DIRECTIONS={"HUI360_to_SSUP-A":"outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json","SSUP-A_to_HUI360":"outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json"}
MODELS=("G","Q","C64","Cm","Jc","Jw","JwD","F");SEEDS=(11,22,33,44,55)

def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1]
 p.add_argument('--root',type=Path,default=root);p.add_argument('--data',type=Path,default=Path('data/official_cross_domain.npz'));p.add_argument('--selection-root',type=Path,default=Path('outputs/vv_followup_v1/r2_cv'));p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1/r2_final'));p.add_argument('--directions',nargs='+',choices=tuple(DIRECTIONS),default=list(DIRECTIONS));p.add_argument('--device',default='cpu');p.add_argument('--num-threads',type=int,default=2);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();root=a.root.resolve();data=a.data if a.data.is_absolute() else root/a.data;sel=a.selection_root if a.selection_root.is_absolute() else root/a.selection_root;outroot=a.output if a.output.is_absolute() else root/a.output
 plans=[(d,m,s,outroot/d/m/'fixed'/str(s)) for d in a.directions for s in SEEDS for m in MODELS]
 if a.dry_run:print(json.dumps({'schema':'acie.vv-r2-final-dry-run.v1','target_scoring':False,'slots':len(plans),'plans':[{'direction':d,'model':m,'seed':s,'out':str(o)} for d,m,s,o in plans]},indent=2));return
 b=Bundle.load(data)
 for direction in a.directions:
  lock=read_json(sel/direction/'MODEL_SELECTION_LOCK.json');split_path=root/DIRECTIONS[direction];split=read_json(split_path)
  for seed in SEEDS:
   gpath=outroot/direction/'G'/'fixed'/str(seed)/'best.pt'
   for model in MODELS:
    selected=lock['models'][model];config={'lr':selected['lr'],'weight_decay':selected['weight_decay'],'device':a.device,'num_threads':a.num_threads};out=outroot/direction/model/'fixed'/str(seed);geometry=gpath if model in ('F','Jw','JwD') else None
    context={'experiment':'r2_final','split_id':'fixed','data_paths':[str(data.resolve()),str(split_path.resolve())],'data_hashes':[digest_file(data),digest_file(split_path)],'source_selection_lock':str((sel/direction/'MODEL_SELECTION_LOCK.json').resolve())}
    result=train_source_model(b,split,model,seed,out,config,geometry,a.resume,context);print(json.dumps({'direction':direction,'model':model,'seed':seed,'status':result['status']}))
if __name__=='__main__':main()
