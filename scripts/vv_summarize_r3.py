#!/usr/bin/env python3
"""Summarize R3 per-split fixed-configuration sensitivity results."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from acie.io import read_json,read_jsonl,write_json
DIRECTIONS=('HUI360_to_SSUP-A','SSUP-A_to_HUI360');SPLITS=('A','B','C','D','E');SEEDS=(11,22,33,44,55)
def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1];p.add_argument('--root',type=Path,default=root);p.add_argument('--output',type=Path,default=Path('outputs/vv_followup_v1/r3'));p.add_argument('--result-out',type=Path);p.add_argument('--selection-root',type=Path,default=Path('outputs/vv_followup_v1/r2_cv'));a=p.parse_args();root=a.root.resolve();resolve=lambda x:x if x.is_absolute() else root/x;base,sel=map(resolve,(a.output,a.selection_root));result_out=resolve(a.result_out) if a.result_out else base;rows=[];seed_rows=[];summary=[]
 for d in DIRECTIONS:
  lock=read_json(sel/d/'MODEL_SELECTION_LOCK.json');models=('G','F',lock['J_star'],lock['C_star']);split_scores={}
  for sp in SPLITS:
   split_scores[sp]={}
   for m in models:
    aps=[]
    for seed in SEEDS:
     pred=read_jsonl(base/d/m/sp/str(seed)/'test_predictions.jsonl');ap=float(average_precision_score([r['label'] for r in pred],[r['score'] for r in pred]));aps.append(ap);seed_rows.append({'direction':d,'split':sp,'model':m,'seed':seed,'AP':ap})
    split_scores[sp][m]=aps;rows.append({'direction':d,'split':sp,'model':m,'AP_mean':float(np.mean(aps)),'AP_sample_SD':float(np.std(aps,ddof=1)),'seeds':5})
   pred=read_jsonl(base/d/'histgb'/sp/'11'/'test_predictions.jsonl');ap=float(average_precision_score([r['label'] for r in pred],[r['score'] for r in pred]));rows.append({'direction':d,'split':sp,'model':'histgb','AP_mean':ap,'AP_sample_SD':None,'seeds':1});seed_rows.append({'direction':d,'split':sp,'model':'histgb','seed':11,'AP':ap})
  for opponent in ('G',lock['J_star'],lock['C_star']):
   values=[]
   for sp in SPLITS:
    values.append(float(np.mean(np.asarray(split_scores[sp]['F'])-np.asarray(split_scores[sp][opponent]))))
   summary.append({'direction':d,'comparison':f'F-{opponent}','split_differences':dict(zip(SPLITS,values)),'positive_splits':int(sum(v>0 for v in values)),'range':[float(min(values)),float(max(values))],'worst_split':SPLITS[int(np.argmin(values))],'mean_across_splits':float(np.mean(values))})
 def csvout(path,data):
  with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 result_out.mkdir(parents=True,exist_ok=True);csvout(result_out/'r3_split_table.csv',rows);csvout(result_out/'r3_seed_AP.csv',seed_rows);csvout(result_out/'r3_difference_summary.csv',[{**x,'split_differences':json.dumps(x['split_differences']),'range':json.dumps(x['range'])} for x in summary]);write_json(result_out/'r3_summary.json',{'schema':'acie.vv-r3-summary.v2','per_seed':seed_rows,'per_split':rows,'differences':summary})
 print(json.dumps({'status':'complete','differences':summary},indent=2))
if __name__=='__main__':main()
