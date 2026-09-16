#!/usr/bin/env python3
"""R2 source-only tree CV, selection, and final source fitting."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import average_precision_score
from acie.data import Bundle,resolve_split
from acie.features import SourceScaler
from acie.io import digest_file,digest_object,read_json,write_json,write_jsonl
from acie.metrics import binary_metrics,choose_threshold
from acie.vv_trees import _build,candidate_grid,tree_features

DIRECTIONS={"HUI360_to_SSUP-A":"outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json","SSUP-A_to_HUI360":"outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json"};FOLDS=('A','B','C')

def cv_dir(base,direction,kind,fold,order,seed):return base/direction/kind/'cv'/fold/f'candidate_{order}'/str(seed)
def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1]
 p.add_argument('--root',type=Path,default=root);p.add_argument('--data',type=Path,default=Path('data/official_cross_domain.npz'));p.add_argument('--fold-root',type=Path,default=Path('protocols/vv_followup_v1/r2_folds'));p.add_argument('--cv-output',type=Path,default=Path('outputs/vv_followup_v1/r2_tree_cv'));p.add_argument('--final-output',type=Path,default=Path('outputs/vv_followup_v1/r2_final'));p.add_argument('--phase',choices=('cv','select','final'),required=True);p.add_argument('--directions',nargs='+',choices=tuple(DIRECTIONS),default=list(DIRECTIONS));p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();root=a.root.resolve();data=a.data if a.data.is_absolute() else root/a.data;foldroot=a.fold_root if a.fold_root.is_absolute() else root/a.fold_root;cvbase=a.cv_output if a.cv_output.is_absolute() else root/a.cv_output;finalbase=a.final_output if a.final_output.is_absolute() else root/a.final_output
 if a.phase=='select':
  for direction in a.directions:
   selected={};tables={}
   for kind in ('histgb','random_forest'):
    rows=[]
    for order,params in enumerate(candidate_grid(kind,'r2')):
     seeds=(11,) if kind=='histgb' else (11,22);values=[read_json(cv_dir(cvbase,direction,kind,fold,order,seed)/'metrics.json')['source_val_AP'] for fold in FOLDS for seed in seeds];rows.append({'order':order,'params':params,'fold_seed_AP':values,'mean_source_CV_AP':float(np.mean(values))})
    best=max(r['mean_source_CV_AP'] for r in rows);pick=next(r for r in rows if best-r['mean_source_CV_AP']<=1e-6);tables[kind]=rows;selected[kind]=pick
   lock={'schema':'acie.vv-r2-tree-selection.v1','status':'locked','direction':direction,'selection_uses_target':False,'models':selected,'candidate_tables':tables};write_json(cvbase/direction/'TREE_SELECTION_LOCK.json',lock);print(json.dumps({'direction':direction,'selected':selected}))
  return
 if a.phase=='cv':
  plans=[(d,k,f,o,p,s) for d in a.directions for k in ('histgb','random_forest') for f in FOLDS for o,p in enumerate(candidate_grid(k,'r2')) for s in ((11,) if k=='histgb' else (11,22))]
  if a.dry_run:print(json.dumps({'slots':len(plans),'target_scoring':False},indent=2));return
  b=Bundle.load(data)
  for direction,kind,fold,order,params,seed in plans:
   out=cv_dir(cvbase,direction,kind,fold,order,seed);metrics_path=out/'metrics.json'
   if metrics_path.exists():continue
   out.mkdir(parents=True,exist_ok=True);split=read_json(foldroot/direction/f'{fold}.json');ix=resolve_split(b,split);sc=SourceScaler.fit(b.a[ix['train']],b.q[ix['train']]);x=tree_features(sc,b.a,b.q);start=time.perf_counter();model=_build(kind,params,seed);model.fit(x[ix['train']],b.y[ix['train']]);score=model.predict_proba(x[ix['val']])[:,1];ap=float(average_precision_score(b.y[ix['val']],score));write_json(metrics_path,{'source_val_AP':ap,'fit_seconds':time.perf_counter()-start,'params':params,'seed':seed,'fold':fold,'target_scoring_performed':False});write_json(out/'run.json',{'status':'complete','kind':kind,'direction':direction,'fold':fold,'candidate':order,'seed':seed,'target_scoring_performed':False});print(json.dumps({'direction':direction,'kind':kind,'fold':fold,'candidate':order,'seed':seed,'AP':ap}))
  return
 b=Bundle.load(data)
 for direction in a.directions:
  lock=read_json(cvbase/direction/'TREE_SELECTION_LOCK.json');split=read_json(root/DIRECTIONS[direction]);ix=resolve_split(b,split)
  for kind in ('histgb','random_forest'):
   params=lock['models'][kind]['params'];seeds=(11,) if kind=='histgb' else (11,22,33,44,55)
   for seed in seeds:
    out=finalbase/direction/kind/'fixed'/str(seed);out.mkdir(parents=True,exist_ok=True)
    if (out/'tree.joblib').exists():continue
    sc=SourceScaler.fit(b.a[ix['train']],b.q[ix['train']]);x=tree_features(sc,b.a,b.q);start=time.perf_counter();model=_build(kind,params,seed);model.fit(x[ix['train']],b.y[ix['train']]);score=model.predict_proba(x[ix['val']])[:,1];threshold=choose_threshold(b.y[ix['val']],score);payload={'schema':'acie.vv-tree-checkpoint.v1','kind':kind,'seed':seed,'protocol':'r2','model':model,'scaler':sc.to_dict(),'split':split,'threshold':threshold,'selected_params':params,'run_id':digest_object({'direction':direction,'kind':kind,'seed':seed,'params':params})};joblib.dump(payload,out/'tree.joblib');h=digest_file(out/'tree.joblib');write_json(out/'scaler.json',sc.to_dict());write_json(out/'split.json',split);write_json(out/'candidate_results.json',[lock['models'][kind]]);write_json(out/'metrics.json',{'source_val':binary_metrics(b.y[ix['val']],score,threshold),'target':None});write_json(out/'timing.json',{'fit_selection_seconds':time.perf_counter()-start,'serialized_bytes':(out/'tree.joblib').stat().st_size});write_jsonl(out/'val_predictions.jsonl',[{**b.meta[i],'label':int(b.y[i]),'score':float(v),'model_id':kind,'seed':seed,'split_id':'fixed','checkpoint_hash':h,'source_threshold':threshold,'geometry_logit':None,'evidence_logit':None} for i,v in zip(ix['val'],score)]);write_json(out/'run.json',{'status':'complete','model_id':kind,'seed':seed,'feature_dim':389,'target_scoring_performed':False,'selected_params':params,'checkpoint_hash':h});print(json.dumps({'direction':direction,'kind':kind,'seed':seed,'status':'complete'}))
if __name__=='__main__':main()
