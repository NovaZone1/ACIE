"""Canonical data bundle and leakage-resistant group splits."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .features import extract, GEOMETRY_NAMES, BEHAVIOR_CHANNELS
from .io import read_jsonl,write_json,digest_file,inside,read_json

REQUIRED = {'sample_id','dataset','recording','track_id','pool','label','window','panoramic'}

@dataclass
class Bundle:
    a: np.ndarray
    q: np.ndarray
    y: np.ndarray
    meta: list[dict]
    provenance: dict

    def save(self,path:str|Path)->None:
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,a=self.a,q=self.q,y=self.y)
        write_json(path.with_suffix('.json'),{'schema':'acie.bundle.v1','meta':self.meta,'provenance':self.provenance,
                                             'npz_sha256':digest_file(path)})

    @classmethod
    def load(cls,path:str|Path)->'Bundle':
        path=Path(path); side=read_json(path.with_suffix('.json'))
        if side.get('schema')!='acie.bundle.v1':raise ValueError('Unknown bundle schema')
        if digest_file(path)!=side['npz_sha256']:raise ValueError('Bundle checksum mismatch')
        with np.load(path,allow_pickle=False) as f:
            b=cls(f['a'].astype(np.float32),f['q'].astype(np.float32),f['y'].astype(np.int64),side['meta'],side['provenance'])
        b.validate();return b

    def validate(self)->None:
        n=len(self.y)
        if n==0 or self.a.ndim!=2 or self.q.ndim!=3 or len(self.a)!=n or len(self.q)!=n or len(self.meta)!=n:
            raise ValueError('Inconsistent/empty bundle')
        if set(np.unique(self.y))- {0,1}:raise ValueError('Labels must be binary')
        if not np.isfinite(self.a).all() or not np.isfinite(self.q).all():raise ValueError('Non-finite bundle')
        ids=[m['sample_id'] for m in self.meta]
        if len(set(ids))!=n:raise ValueError('Duplicate sample_id')
        if self.q.shape[-1]!=119:raise ValueError('Expected 17 joints x 7 channels')


def prepare(manifest:str|Path,out:str|Path)->Bundle:
    manifest=Path(manifest);rows=read_jsonl(manifest);aa=[];qq=[];yy=[];meta=[];hashes=[]
    for row in rows:
        missing=REQUIRED-set(row)
        if missing:raise ValueError(f'Missing manifest fields {missing}')
        if row['pool'] not in ['train','test','all']:raise ValueError('pool must be train/test/all')
        if not isinstance(row['panoramic'],bool):raise ValueError('panoramic must be boolean')
        path=inside(manifest.parent,row['window'])
        with np.load(path,allow_pickle=False) as f:
            needed={'keypoints','bbox','mask_area','times'}
            if not needed.issubset(f.files):raise ValueError(f'{path}: missing {needed-set(f.files)}')
            a,q=extract(f['keypoints'],f['bbox'],f['mask_area'],f['times'],panoramic=row['panoramic'])
            t0=float(f['times'][0]); t1=float(f['times'][-1])
        if qq and q.shape!=qq[0].shape:raise ValueError('All windows need equal T; use the frozen official window length')
        if row['label'] not in [0,1]:raise ValueError('label must be 0/1')
        m={k:row[k] for k in ['sample_id','dataset','recording','track_id','pool','panoramic']}
        for k in ['onset_time','cutoff_seconds','source_index','synthetic','group_id']:
            if k in row:m[k]=row[k]  # metadata only: never concatenated with predictors
        m.update(start_time=t0,end_time=t1)
        if m.get('onset_time') is not None and t1>=float(m['onset_time']) and row['label']==1:
            raise ValueError('Positive observation reaches interaction onset')
        aa.append(a);qq.append(q);yy.append(row['label']);meta.append(m);hashes.append(digest_file(path))
    if not rows:raise ValueError('Empty manifest')
    b=Bundle(np.stack(aa),np.stack(qq),np.array(yy),meta,
             {'manifest_sha256':digest_file(manifest),'window_hashes':hashes,
              'geometry_names':GEOMETRY_NAMES,'behavior_channels':BEHAVIOR_CHANNELS,
              'synthetic':all(r.get('synthetic',False) for r in rows)})
    b.validate();b.save(out);return b


def group_key(m:dict)->str:
    """Return the conservative split/bootstrap unit when one is supplied."""
    return str(m['dataset'])+'::'+str(m.get('group_id',m['recording']))
def track_key(m:dict)->str:return str(m['dataset'])+'::'+str(m['track_id'])


def holdout(indices:np.ndarray,b:Bundle,fraction:float,seed:int)->tuple[np.ndarray,np.ndarray]:
    if not 0<fraction<1:raise ValueError('Holdout fraction must be inside (0,1)')
    groups=np.array([group_key(b.meta[i]) for i in indices])
    if len(set(groups))<2:raise ValueError('Need at least two recording groups; do not split overlapping frames')
    # Search group partitions using only source labels to ensure both classes.
    for tr,va in GroupShuffleSplit(n_splits=100,test_size=fraction,random_state=seed).split(indices,b.y[indices],groups):
        if len(np.unique(b.y[indices[tr]]))==2 and len(np.unique(b.y[indices[va]]))==2:
            return indices[tr],indices[va]
    raise ValueError('No group split with both classes. Inspect recordings; no row-wise fallback.')


def make_split(b:Bundle,source:str,target:str,seed:int=42,val_fraction:float=.2,
               test_fraction:float=.2)->dict:
    src=np.array([i for i,m in enumerate(b.meta) if m['dataset']==source],dtype=int)
    tgt=np.array([i for i,m in enumerate(b.meta) if m['dataset']==target],dtype=int)
    if len(src)==0 or len(tgt)==0:raise ValueError('Unknown/empty source or target domain')
    if source==target:
        # Intentionally a local grouped protocol, not a mislabeled official track split.
        src,test=holdout(src,b,test_fraction,seed+1009)
        protocol='local_recording_grouped_in_domain'
    else:
        src=np.array([i for i in src if b.meta[i]['pool'] in ['train','all']],dtype=int)
        test=np.array([i for i in tgt if b.meta[i]['pool'] in ['test','all']],dtype=int)
        protocol='strict_source_only_cross_domain'
    train,val=holdout(src,b,val_fraction,seed)
    d={'schema':'acie.split.v1','source':source,'target':target,'seed':seed,'protocol':protocol,
       'train':[b.meta[i]['sample_id'] for i in train],
       'val':[b.meta[i]['sample_id'] for i in val],'test':[b.meta[i]['sample_id'] for i in test]}
    resolve_split(b,d);return d


def resolve_split(b:Bundle,d:dict)->dict[str,np.ndarray]:
    if d.get('schema')!='acie.split.v1':raise ValueError('Unknown split schema')
    ids={m['sample_id']:i for i,m in enumerate(b.meta)}
    arrays={}
    for s in ['train','val','test']:
        if not d.get(s) or len(set(d[s]))!=len(d[s]):raise ValueError(f'Empty/duplicate split {s}')
        if set(d[s])-set(ids):raise ValueError('Split refers to unknown samples')
        arrays[s]=np.array([ids[x] for x in d[s]])
    for s,t in [('train','val'),('train','test'),('val','test')]:
        for fn in [group_key,track_key]:
            if {fn(b.meta[i]) for i in arrays[s]} & {fn(b.meta[i]) for i in arrays[t]}:
                raise ValueError(f'{s}/{t}: recording or track leakage')
    for s in ['train','val']:
        if any(b.meta[i]['dataset']!=d['source'] for i in arrays[s]):raise ValueError('Target data entered source fitting')
        if len(np.unique(b.y[arrays[s]]))<2:raise ValueError(f'{s} must contain both classes')
    if any(b.meta[i]['dataset']!=d['target'] for i in arrays['test']):raise ValueError('Unexpected test domain')
    return arrays
