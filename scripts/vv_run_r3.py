#!/usr/bin/env python3
"""Train R3 fixed-configuration date-sensitivity models using source data only."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import joblib
from acie.data import Bundle,resolve_split
from acie.features import SourceScaler
from acie.io import digest_file,digest_object,read_json,write_json,write_jsonl
from acie.metrics import binary_metrics,choose_threshold
from acie.vv_engine import train_source_model
from acie.vv_trees import _build,tree_features

DIRECTIONS=('HUI360_to_SSUP-A','SSUP-A_to_HUI360');SPLITS=('A','B','C','D','E');SEEDS=(11,22,33,44,55)
def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1];p.add_argument('--root',type=Path,default=root);p.add_argument('--data',type=Path,default=Path('data/official_cross_domain.npz'));p.add_argument('--split-root',type=Path,default=Path('protocols/vv_followup_v1/r3_splits'));p.add_argument('--selection-root',type=Path,default=Path('outputs/vv_followup_v1/r2_cv'));p.add_argument('--tree-selection-root',type=Path,default=Path('outputs/vv_followup_v1/r2_tree_cv'));p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1/r3'));p.add_argument('--directions',nargs='+',choices=DIRECTIONS,default=list(DIRECTIONS));p.add_argument('--device',default='cpu');p.add_argument('--num-threads',type=int,default=2);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args();root=a.root.resolve();resolve=lambda x:x if x.is_absolute() else root/x;data,splitroot,sel,treesel,outroot=map(resolve,(a.data,a.split_root,a.selection_root,a.tree_selection_root,a.output));plans=[]
 for d in a.directions:
  lock=read_json(sel/d/'MODEL_SELECTION_LOCK.json');models=('G','F',lock['J_star'],lock['C_star'])
  for sp in SPLITS:
   for seed in SEEDS:
    for m in models:plans.append((d,sp,m,seed))
   plans.append((d,sp,'histgb',11))
 if a.dry_run:print(json.dumps({'slots':len(plans),'target_scoring':False,'plans':[{'direction':d,'split':sp,'model':m,'seed':s} for d,sp,m,s in plans]},indent=2));return
 b=Bundle.load(data)
 for d in a.directions:
  lock=read_json(sel/d/'MODEL_SELECTION_LOCK.json');tree_lock=read_json(treesel/d/'TREE_SELECTION_LOCK.json');models=('G','F',lock['J_star'],lock['C_star'])
  for sp in SPLITS:
   split_path=splitroot/d/f'{sp}.json';split=read_json(split_path)
   for seed in SEEDS:
    gpath=outroot/d/'G'/sp/str(seed)/'best.pt'
    for m in models:
     chosen=lock['models'][m];config={'lr':chosen['lr'],'weight_decay':chosen['weight_decay'],'device':a.device,'num_threads':a.num_threads};geometry=gpath if m in ('F','Jw','JwD') else None;out=outroot/d/m/sp/str(seed);context={'experiment':'r3','split_id':sp,'data_paths':[str(data),str(split_path)],'data_hashes':[digest_file(data),digest_file(split_path)]};result=train_source_model(b,split,m,seed,out,config,geometry,a.resume,context);print(json.dumps({'direction':d,'split':sp,'model':m,'seed':seed,'status':result['status']}))
   out=outroot/d/'histgb'/sp/'11';out.mkdir(parents=True,exist_ok=True)
   if not (out/'tree.joblib').exists():
    ix=resolve_split(b,split);sc=SourceScaler.fit(b.a[ix['train']],b.q[ix['train']]);x=tree_features(sc,b.a,b.q);params=tree_lock['models']['histgb']['params'];start=time.perf_counter();model=_build('histgb',params,11);model.fit(x[ix['train']],b.y[ix['train']]);score=model.predict_proba(x[ix['val']])[:,1];threshold=choose_threshold(b.y[ix['val']],score);payload={'schema':'acie.vv-tree-checkpoint.v1','kind':'histgb','seed':11,'protocol':'r3','model':model,'scaler':sc.to_dict(),'split':split,'threshold':threshold,'selected_params':params,'run_id':digest_object({'direction':d,'split':sp,'params':params})};joblib.dump(payload,out/'tree.joblib');h=digest_file(out/'tree.joblib');write_json(out/'scaler.json',sc.to_dict());write_json(out/'split.json',split);write_json(out/'metrics.json',{'source_val':binary_metrics(b.y[ix['val']],score,threshold),'target':None});write_json(out/'timing.json',{'fit_seconds':time.perf_counter()-start});write_jsonl(out/'val_predictions.jsonl',[{**b.meta[i],'label':int(b.y[i]),'score':float(v),'model_id':'histgb','seed':11,'split_id':sp,'checkpoint_hash':h,'source_threshold':threshold} for i,v in zip(ix['val'],score)]);write_json(out/'run.json',{'status':'complete','target_scoring_performed':False,'feature_dim':389,'checkpoint_hash':h});print(json.dumps({'direction':d,'split':sp,'model':'histgb','seed':11,'status':'complete'}))
if __name__=='__main__':main()
