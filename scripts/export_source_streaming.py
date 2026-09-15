"""Export official legal windows one recording at a time to bound peak memory."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from acie.data import prepare
from acie.io import digest_file,read_jsonl,write_json,write_jsonl
from export_hui360 import export_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--split',choices=['train','val'],required=True)
    p.add_argument('--domain',choices=['HUI360','SSUP-A'],required=True);p.add_argument('--fps',required=True,type=float)
    p.add_argument('--panoramic',action=argparse.BooleanOptionalAction,required=True)
    p.add_argument('--group-by',choices=['recording','date'],default='recording');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--allow-missing-recording',action='append',default=[])
    a=p.parse_args()
    if a.fps<=0:raise ValueError('FPS must be positive')
    if (a.out/'manifest.jsonl').exists():raise FileExistsError('Choose a new export directory')
    upstream=a.upstream.resolve();sys.path.insert(0,str(upstream))
    from utils.loader_utils import load_hui_dataset
    cfg=yaml.safe_load(a.config.read_text());selected=cfg[f'include_recordings_{a.split}']
    cfg.setdefault('format_by_channel',False);cfg.setdefault('remove_joints',None)
    cfg.setdefault('perspective_reprojection',None)
    if selected=='all':raise ValueError('Streaming export requires explicit recordings')
    available=[]
    for recording in selected:
        if list(a.data_root.glob(f'data-{recording}-*.csv')) or list(a.data_root.glob(f'data-{recording}-*_light.csv')):
            available.append(recording)
    missing=sorted(set(selected)-set(available));unauthorized=sorted(set(missing)-set(a.allow_missing_recording))
    if unauthorized:raise FileNotFoundError(f'Missing selected recordings: {unauthorized}')
    a.out.mkdir(parents=True,exist_ok=True);rows=[]
    args=SimpleNamespace(preload_data=False,preload_only=False,verbose=False,
                         hf_local_dir=str(a.data_root.resolve()),offline_mode=True)
    for number,recording in enumerate(available):
        local=copy.deepcopy(cfg);local[f'include_recordings_{a.split}']=[recording]
        torch.manual_seed(a.seed);np.random.seed(a.seed)
        ds=load_hui_dataset(args,local,split=a.split,num_workers=a.workers)
        part=a.out/'parts'/f'{number:03d}'
        export_dataset(ds,part,a.domain,'train' if a.split=='train' else 'test',a.fps,a.panoramic,a.seed,a.group_by)
        part_rows=read_jsonl(part/'manifest.jsonl')
        for row in part_rows:row['window']=str(Path('parts')/f'{number:03d}'/row['window'])
        rows.extend(part_rows);print(f'EXPORT {number+1}/{len(available)} {recording}: {len(part_rows)} windows',flush=True)
        del ds;gc.collect()
    write_jsonl(a.out/'manifest.jsonl',rows)
    try:commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit='unavailable'
    provenance={'upstream_commit':commit,'config_sha256':digest_file(a.config),'resolved_upstream_config':cfg,
                'split':a.split,'domain':a.domain,'count':len(rows),'fps':a.fps,'group_by':a.group_by,
                'selected_recordings':selected,'available_recordings':available,'missing_recordings':missing,
                'execution':'official load_hui_dataset and legal proposals, one recording per process-lifetime object',
                'reason':'bounded-memory equivalent; no cross-recording rule is used by legal-window generation',
                'normalization':'raw visible prefix -> canonical image fractions; ACIE source-only scaling'}
    write_json(a.out/'export_provenance.json',provenance)
    bundle=prepare(a.out/'manifest.jsonl',a.out/'bundle.npz');bundle.provenance['export']=provenance
    bundle.save(a.out/'bundle.npz');print(f'Exported {len(rows)} windows to {a.out}')


if __name__=='__main__':main()
