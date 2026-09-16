#!/usr/bin/env python3
"""Create and validate the registered R2 source-train date folds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from acie.data import Bundle, group_key, resolve_split
from acie.io import digest_file, digest_object, read_json, write_json

DIRECTIONS={
 'HUI360_to_SSUP-A':'outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json',
 'SSUP-A_to_HUI360':'outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json'}
FOLDS={
 'HUI360_to_SSUP-A':{
  'A':['2025-07-10','2025-07-22','2025-07-29','2025-10-20'],
  'B':['2025-07-11','2025-07-16','2025-07-24','2025-10-16'],
  'C':['2025-07-17','2025-07-18','2025-07-23','2025-10-15']},
 'SSUP-A_to_HUI360':{
  'A':['2022-10-12'],'B':['2022-09-21','2022-09-28'],'C':['2022-10-06']}}
EXPECTED={
 'HUI360_to_SSUP-A':{'A':(339,61),'B':(423,52),'C':(387,51)},
 'SSUP-A_to_HUI360':{'A':(1922,37),'B':(1943,40),'C':(1031,28)}}

def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1]
 p.add_argument('--root',type=Path,default=root);p.add_argument('--data',type=Path,default=Path('data/official_cross_domain.npz'));p.add_argument('--out',type=Path,default=Path('protocols/vv_followup_v1/r2_folds'));p.add_argument('--check-only',action='store_true')
 a=p.parse_args();root=a.root.resolve();data=a.data if a.data.is_absolute() else root/a.data;out=a.out if a.out.is_absolute() else root/a.out;b=Bundle.load(data);report={};id_to_index={m['sample_id']:i for i,m in enumerate(b.meta)}
 for direction,fixed_relative in DIRECTIONS.items():
  fixed=read_json(root/fixed_relative);ix=resolve_split(b,fixed);source_ids=set(fixed['train']);fold_ids={};report[direction]={}
  for fold,dates in FOLDS[direction].items():
   val=[b.meta[i]['sample_id'] for i in ix['train'] if str(b.meta[i].get('group_id',b.meta[i]['recording'])) in dates]
   train=[sample_id for sample_id in fixed['train'] if sample_id not in set(val)]
   observed=(len(val),sum(int(b.y[id_to_index[sample_id]]) for sample_id in val))
   if observed!=EXPECTED[direction][fold]:raise AssertionError(f'{direction}/{fold}: expected {EXPECTED[direction][fold]}, got {observed}')
   inner={**fixed,'seed':20260916,'protocol':'r2_source_train_date_fold','train':train,'val':val,'test':fixed['test'],'fold_id':fold,'validation_dates':dates}
   resolve_split(b,inner);fold_ids[fold]=set(val)
   path=out/direction/f'{fold}.json'
   if not a.check_only:write_json(path,inner)
   report[direction][fold]={'train_n':len(train),'train_positives':int(sum(b.y[id_to_index[s]] for s in train)),'val_n':observed[0],'val_positives':observed[1],'dates':dates,'split_object_sha256':digest_object(inner)}
  if set().union(*fold_ids.values())!=source_ids or sum(len(v) for v in fold_ids.values())!=len(source_ids):raise AssertionError(f'{direction}: folds do not partition fixed source train exactly')
 lock={'schema':'acie.vv-r2-fold-lock.v1','status':'locked_for_source_development','created_at':'2026-09-16','data_path':str(data.resolve()),'data_sha256':digest_file(data),'folds':report,'grid':[{'order':i,'lr':lr,'weight_decay':wd} for i,(lr,wd) in enumerate(( (lr,wd) for lr in (0.0003,0.001,0.003) for wd in (0.00005,0.0005) ))],'development_seeds':[11,22],'selection':'unweighted mean AP over three folds and two seeds; differences <=1e-6 tie by candidate order','target_scoring':False}
 if not a.check_only:write_json(out/'R2_FOLD_LOCK.json',lock)
 print(json.dumps({'status':'passed','directions':report,'lock_digest':digest_object(lock)},indent=2))
if __name__=='__main__':main()
