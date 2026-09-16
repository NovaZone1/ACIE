#!/usr/bin/env python3
"""Create the five registered source-date sensitivity splits for R3."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from acie.data import Bundle,resolve_split
from acie.io import digest_file,digest_object,read_json,write_json

DIRECTIONS={"HUI360_to_SSUP-A":"outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json","SSUP-A_to_HUI360":"outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json"}
GROUPS={
 "HUI360_to_SSUP-A":{"A":["2025-07-07","2025-07-18","2025-07-29"],"B":["2025-07-10","2025-07-22","2025-10-09"],"C":["2025-07-11","2025-07-23","2025-10-15"],"D":["2025-07-16","2025-07-24","2025-10-16"],"E":["2025-07-17","2025-07-25","2025-10-20"]},
 "SSUP-A_to_HUI360":{"A":["2022-09-21"],"B":["2022-09-26"],"C":["2022-09-28"],"D":["2022-10-06"],"E":["2022-10-12"]}}

def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1];p.add_argument('--root',type=Path,default=root);p.add_argument('--data',type=Path,default=Path('data/official_cross_domain.npz'));p.add_argument('--out',type=Path,default=Path('protocols/vv_followup_v1/r3_splits'));a=p.parse_args();root=a.root.resolve();data=a.data if a.data.is_absolute() else root/a.data;out=a.out if a.out.is_absolute() else root/a.out;b=Bundle.load(data);ids={m['sample_id']:i for i,m in enumerate(b.meta)};report={}
 for direction,relative in DIRECTIONS.items():
  fixed=read_json(root/relative);pool=fixed['train']+fixed['val'];report[direction]={};seen=[]
  for split_id,dates in GROUPS[direction].items():
   val=[s for s in pool if str(b.meta[ids[s]].get('group_id',b.meta[ids[s]]['recording'])) in dates];train=[s for s in pool if s not in set(val)];split={**fixed,'seed':20260916,'protocol':'r3_fixed_config_source_date_sensitivity','train':train,'val':val,'test':fixed['test'],'split_id':split_id,'validation_dates':dates};resolve_split(b,split);write_json(out/direction/f'{split_id}.json',split);seen+=val;report[direction][split_id]={'dates':dates,'train_n':len(train),'train_positives':int(sum(b.y[ids[s]] for s in train)),'val_n':len(val),'val_positives':int(sum(b.y[ids[s]] for s in val)),'split_sha256':digest_object(split)}
  if len(seen)!=len(pool) or set(seen)!=set(pool):raise AssertionError(f'{direction}: R3 groups do not partition the full source pool')
 lock={'schema':'acie.vv-r3-split-lock.v1','status':'locked','created_at':'2026-09-16','data_sha256':digest_file(data),'splits':report,'models':'G/F/J*/C* plus HistGB; identities and hyperparameters inherited from R2','target_scoring':False};write_json(out/'R3_SPLIT_LOCK.json',lock);print(json.dumps({'status':'passed','lock_digest':digest_object(lock),'splits':report},indent=2))
if __name__=='__main__':main()
