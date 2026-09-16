#!/usr/bin/env python3
"""Generate R1 tables and registered paired date/seed bootstrap intervals."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from acie.io import read_json, read_jsonl, write_json
from acie.vv_models import EXPECTED_PARAMETERS

DIRECTIONS=("HUI360_to_SSUP-A","SSUP-A_to_HUI360")
NEURAL=("G","Q","C64","Cm","Jc","Jw","JwD","F")
SEEDS=(11,22,33,44,55)

def load_predictions(base,direction,model,seed,tree=False):
 p=base/direction/model/'fixed'/str(seed)/'test_predictions.jsonl';rows=read_jsonl(p)
 return rows,np.array([r['label'] for r in rows],int),np.array([r['score'] for r in rows],float)

def interval(values,lo,hi):
 return [float(x) for x in np.quantile(np.asarray(values),[lo,hi])] if values else None

def two_level_bootstrap(y,groups,a,b,repeats=10000,seed=20260916):
 unique=np.unique(groups);group_rows={g:np.flatnonzero(groups==g) for g in unique};rng=np.random.default_rng(seed)
 cache={};joint=[];date_only=[];invalid=0
 def ap(method,seed_index,counts):
  key=(method,seed_index,counts)
  if key not in cache:
   ix=np.concatenate([group_rows[g] for g,count in zip(unique,counts) for _ in range(count)])
   score=a[seed_index] if method=='a' else b[seed_index]
   cache[key]=float(average_precision_score(y[ix],score[ix]))
  return cache[key]
 for _ in range(repeats):
  drawn=rng.integers(0,len(unique),len(unique));counts=tuple(int(x) for x in np.bincount(drawn,minlength=len(unique)))
  present=np.concatenate([group_rows[g] for g,count in zip(unique,counts) for _ in range(count)])
  if len(np.unique(y[present]))<2:
   invalid+=1;continue
  per_seed=np.array([ap('a',i,counts)-ap('b',i,counts) for i in range(len(a))])
  date_only.append(float(per_seed.mean()))
  seed_draw=rng.integers(0,len(a),len(a));joint.append(float(per_seed[seed_draw].mean()))
 def describe(values):return {'valid':len(values),'interval_95':interval(values,.025,.975),'interval_98_75':interval(values,.00625,.99375)}
 return {'groups':len(unique),'replicates_requested':repeats,'invalid_no_both_classes':invalid,'joint_date_and_seed':describe(joint),'date_only_fixed_seeds':describe(date_only),'warning':'fewer than 90% valid replicates' if len(joint)<.9*repeats else None}

def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1]
 p.add_argument('--root',type=Path,default=root);p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1'));p.add_argument('--experiment',default='r1');p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--selection-root',type=Path)
 a=p.parse_args();root=a.root.resolve();base=(a.output if a.output.is_absolute() else root/a.output)/a.experiment;selection_root=(a.selection_root if a.selection_root is None or a.selection_root.is_absolute() else root/a.selection_root)
 lock=read_json(base/'TARGET_SCORING_LOCK.json')
 if lock.get('status')!='locked' or lock.get('checkpoint_count')!=92:raise ValueError('R1 target scoring lock is absent or incomplete')
 main_rows=[];comparisons=[];date_rows=[];all_data={}
 for direction in DIRECTIONS:
  reference_ids=None;reference_y=None;reference_groups=None;scores={}
  model_seeds={**{m:SEEDS for m in NEURAL},'histgb':(11,),'random_forest':SEEDS}
  for model,seeds in model_seeds.items():
   model_scores=[];aps=[];aurocs=[]
   for seed in seeds:
    rows,y,score=load_predictions(base,direction,model,seed,model not in NEURAL);ids=[r['sample_id'] for r in rows];groups=np.array([str(r.get('group_id',r['recording'])) for r in rows])
    if reference_ids is None:reference_ids,reference_y,reference_groups=ids,y,groups
    elif ids!=reference_ids or not np.array_equal(y,reference_y):raise ValueError(f'{direction}/{model}/{seed}: target IDs or labels differ')
    aps.append(float(average_precision_score(y,score)));aurocs.append(float(roc_auc_score(y,score)));model_scores.append(score)
   array=np.stack(model_scores);scores[model]=array
   main_rows.append({'direction':direction,'model':model,'representation':'a' if model=='G' else 'q' if model=='Q' else 'a+q statistics' if model in ('histgb','random_forest') else 'a+q','seeds':len(seeds),'effective_parameters':EXPECTED_PARAMETERS.get(model),'AP_mean':float(np.mean(aps)),'AP_sample_SD':float(np.std(aps,ddof=1)) if len(aps)>1 else None,'AUROC_mean':float(np.mean(aurocs)),'AUROC_sample_SD':float(np.std(aurocs,ddof=1)) if len(aurocs)>1 else None,'ensemble_AP':float(average_precision_score(reference_y,array.mean(0)))})
  opponents=('Jw','C64')
  if selection_root is not None:
   selection=read_json(selection_root/direction/'MODEL_SELECTION_LOCK.json');opponents=(selection['J_star'],selection['C_star'])
  for opponent in opponents:
   f=scores['F'];o=scores[opponent];seed_delta=[float(average_precision_score(reference_y,f[i])-average_precision_score(reference_y,o[i])) for i in range(5)]
   boot=two_level_bootstrap(reference_y,reference_groups,f,o,a.bootstrap,20260916)
   item={'direction':direction,'comparison':f'F-{opponent}','seed_AP_differences':seed_delta,'mean_AP_difference':float(np.mean(seed_delta)),'positive_seeds':int(np.sum(np.asarray(seed_delta)>0)),'bootstrap':boot}
   comparisons.append(item)
   for group in np.unique(reference_groups):
    ix=np.flatnonzero(reference_groups==group);valid=len(np.unique(reference_y[ix]))==2
    delta=float(np.mean([average_precision_score(reference_y[ix],f[i,ix])-average_precision_score(reference_y[ix],o[i,ix]) for i in range(5)])) if valid else None
    keep=np.flatnonzero(reference_groups!=group);loo=float(np.mean([average_precision_score(reference_y[keep],f[i,keep])-average_precision_score(reference_y[keep],o[i,keep]) for i in range(5)])) if len(np.unique(reference_y[keep]))==2 else None
    date_rows.append({'direction':direction,'comparison':f'F-{opponent}','date_group':str(group),'n':len(ix),'positives':int(reference_y[ix].sum()),'mean_within_date_AP_difference':delta,'leave_one_date_out_AP_difference':loo})
  all_data[direction]={'n':len(reference_y),'positives':int(reference_y.sum()),'date_groups':len(np.unique(reference_groups))}

 def write_csv(path,rows):
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('w',newline='',encoding='utf-8') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 write_csv(base/'main_table.csv',main_rows)
 flat=[]
 for row in comparisons:
  j=row['bootstrap']['joint_date_and_seed'];d=row['bootstrap']['date_only_fixed_seeds']
  flat.append({'direction':row['direction'],'comparison':row['comparison'],'mean_AP_difference':row['mean_AP_difference'],'positive_seeds':row['positive_seeds'],'joint_95_low':j['interval_95'][0],'joint_95_high':j['interval_95'][1],'joint_98_75_low':j['interval_98_75'][0],'joint_98_75_high':j['interval_98_75'][1],'date_only_95_low':d['interval_95'][0],'date_only_95_high':d['interval_95'][1]})
 write_csv(base/'primary_differences.csv',flat);write_csv(base/'date_diagnostics.csv',date_rows)
 summary={'schema':'acie.vv-r2-summary.v1' if selection_root is not None else 'acie.vv-r1-summary.v1','data':all_data,'main_table':main_rows,'primary_comparisons':comparisons,'target_lock_digest':lock['lock_digest'],'bootstrap_repeats':a.bootstrap}
 write_json(base/'summary.json',summary)
 title='R2 source-selected results' if selection_root is not None else 'R1 fixed-configuration results'
 lines=[f'# {title}','','All values are generated from locked per-seed target predictions. Pilot results are excluded.','', '| Direction | Model | AP mean | AP SD | AUROC mean | Ensemble AP |','|---|---:|---:|---:|---:|---:|']
 for r in main_rows:lines.append(f"| {r['direction']} | {r['model']} | {r['AP_mean']:.6f} | {r['AP_sample_SD'] if r['AP_sample_SD'] is not None else 'NA'} | {r['AUROC_mean']:.6f} | {r['ensemble_AP']:.6f} |")
 lines+=['','## Registered primary differences','','| Direction | Comparison | Mean AP difference | Positive seeds | 95% joint interval | 98.75% joint interval |','|---|---:|---:|---:|---:|---:|']
 for r in comparisons:
  j=r['bootstrap']['joint_date_and_seed'];lines.append(f"| {r['direction']} | {r['comparison']} | {r['mean_AP_difference']:.6f} | {r['positive_seeds']}/5 | {j['interval_95']} | {j['interval_98_75']} |")
 (base/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'status':'complete','main_rows':len(main_rows),'primary_comparisons':len(comparisons),'out':str(base/'summary.json')},indent=2))
if __name__=='__main__':main()
