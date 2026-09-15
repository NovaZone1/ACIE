"""Export official HUI360/SSUP-A sampled windows to the ACIE canonical schema.

Reads the official dataset object and its legal window proposals; does not re-label
tracks. Requires a local clone plus its dependencies and processed CSV data.
No upstream code is bundled here. Run --help before using an unreviewed revision.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from acie.features import JOINTS
from acie.data import prepare
from acie.io import write_json,write_jsonl,digest_file


def recording_group(recording,group_by):
    if group_by=='recording':return recording
    match=re.search(r'(20\d{2})[_-](\d{2})[_-](\d{2})',recording)
    if not match:raise ValueError(f'Cannot extract date from recording {recording!r}')
    return '-'.join(match.groups())


def export_dataset(ds,out,domain,pool,fps,panoramic,seed=42,group_by='recording'):
    out=Path(out);(out/'windows').mkdir(parents=True,exist_ok=True);rows=[]
    rng=np.random.default_rng(seed)
    required=['input_by_tracks','datasets_by_unique_track_identifier','idx_to_unique_track_identifier']
    if any(not hasattr(ds,k) for k in required):raise RuntimeError('Upstream dataset API changed; audit the adapter')
    for index,track in enumerate(ds.idx_to_unique_track_identifier):
        choices=ds.input_by_tracks[track]
        if ds.inputs_per_track_stride>0:choice=ds.idx_to_index_among_track[index]
        elif ds.fix_index_per_track:choice=ds.index_choice_by_unique_track_identifier[track]
        else:choice=int(rng.integers(len(choices)))
        start,length,label=choices[choice]
        track_df=ds.datasets_by_unique_track_identifier[track]
        frame_ix=list(range(start,start+length,ds.subsample_frames))
        df=track_df.iloc[frame_ix].copy()
        width=float(df['image_width'].iloc[0]);height=float(df['image_height'].iloc[0])
        if width<=0 or height<=0:raise ValueError('Invalid image dimensions')
        def column(base):
            if base in df:return df[base].to_numpy(dtype=np.float64)
            if base+'_meta' in df:return df[base+'_meta'].to_numpy(dtype=np.float64)
            raise ValueError(f'Missing required upstream column {base}')
        bbox=np.stack([column(k) for k in ['xmin','ymin','xmax','ymax']],1)
        # Retain wrap-around boxes instead of treating the seam as a large person.
        bbox[:,[0,2]]/=width;bbox[:,[1,3]]/=height
        if panoramic:
            raw_width=bbox[:,2]-bbox[:,0]
            bbox[:,0]%=1;bbox[:,2]%=1
            # A literal full-width box is not represented as an empty interval.
            full=np.isclose(np.abs(raw_width),1);bbox[full,0]=0;bbox[full,2]=1
        kp=np.stack([np.stack([column(f'vitpose_{j}_{c}') for c in ['x','y','score']],1) for j in JOINTS],1)
        kp[...,0]/=width;kp[...,1]/=height
        if panoramic:
            valid=kp[...,0]>=0;kp[...,0]=np.where(valid,kp[...,0]%1,kp[...,0])
        mask=column('mask_size')/(width*height)
        # Explicit FPS supplied by the caller; frame indices are preserved for audit.
        image_indexes=column('image_index')
        times=(image_indexes-image_indexes[0])/fps
        sid=hashlib.sha256(f'{domain}|{track}|{int(image_indexes[0])}|{int(image_indexes[-1])}'.encode()).hexdigest()[:24]
        np.savez_compressed(out/'windows'/f'{sid}.npz',keypoints=kp,bbox=bbox,mask_area=mask,times=times)
        recording=str(df['recording'].iloc[0])
        rows.append({'sample_id':f'{domain}:{sid}','dataset':domain,'recording':recording,
                     'group_id':recording_group(recording,group_by),
                     'track_id':str(track),'pool':pool,'label':int(label),'window':f'windows/{sid}.npz',
                     'panoramic':panoramic,'source_index':{'dataset_index':index,'choice':int(choice),
                     'frame_indexes':image_indexes.astype(int).tolist(),'fps':fps},'synthetic':False})
    write_jsonl(out/'manifest.jsonl',rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',required=True,type=Path);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--split',choices=['train','val'],required=True)
    p.add_argument('--domain',choices=['HUI360','SSUP-A'],required=True)
    p.add_argument('--fps',required=True,type=float,help='Confirm frame rate in your frozen data version')
    p.add_argument('--panoramic',action=argparse.BooleanOptionalAction,required=True)
    p.add_argument('--group-by',choices=['recording','date'],default='recording',
                   help='Conservative split/bootstrap unit recorded in the manifest')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--allow-download',action='store_true',help='Explicitly allow the official loader to fetch data')
    a=p.parse_args()
    if a.fps<=0:raise ValueError('FPS must be positive')
    if (a.out/'manifest.jsonl').exists():raise FileExistsError('Choose a new export directory')
    upstream=a.upstream.resolve();sys.path.insert(0,str(upstream))
    try:
        from utils.loader_utils import load_hui_dataset
    except ImportError as e:
        raise SystemExit('Install the official clone dependencies in a separate export environment. '+str(e))
    cfg=yaml.safe_load(a.config.read_text())
    # This upstream commit's loader expects newer optional keys that its shipped
    # MLP configs omit. Use the constructor defaults and record the resolved config.
    cfg.setdefault('format_by_channel',False)
    cfg.setdefault('remove_joints',None)
    cfg.setdefault('perspective_reprojection',None)
    if cfg.get('perspective_reprojection',{}):
        if cfg['perspective_reprojection'].get('do_perspective_reprojection'):
            raise ValueError('Reprojected upstream config not supported by raw-window export; use non-reprojected config')
    args=SimpleNamespace(preload_data=False,preload_only=False,verbose=False,
                         hf_local_dir=str(a.data_root.resolve()),offline_mode=not a.allow_download)
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    ds=load_hui_dataset(args,cfg,split=a.split,num_workers=a.workers)
    rows=export_dataset(ds,a.out,a.domain,'train' if a.split=='train' else 'test',a.fps,a.panoramic,a.seed,a.group_by)
    try:commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit='unavailable'
    write_json(a.out/'export_provenance.json',{'upstream_commit':commit,'config_sha256':digest_file(a.config),
                  'upstream_config':cfg,'split':a.split,'domain':a.domain,'count':len(rows),'fps':a.fps,
                  'group_by':a.group_by,
                  'normalization':'raw visible prefix -> canonical image fractions; ACIE source-only scaling',
                  'sampling':'official legal window proposals; selected indices stored in every manifest row',
                  'scope':'new feature implementation, not a bitwise reproduction of official normalized tensors'})
    b=prepare(a.out/'manifest.jsonl',a.out/'bundle.npz')
    b.provenance['export']=json.loads((a.out/'export_provenance.json').read_text());b.save(a.out/'bundle.npz')
    print(f'Exported {len(rows)} windows to {a.out}')

if __name__=='__main__':main()
