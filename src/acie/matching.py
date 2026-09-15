"""Source-fitted, geometry-defined pairing with track exclusion and support audit."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .data import group_key,track_key

@dataclass
class MatchConfig:
    k:int=3
    radius_quantile:float=.75
    radius:float|None=None
    bandwidth:float=1.0
    max_abs_z:float=3.0
    max_negative_track_uses:int=12
    candidate_neighbors:int=128
    context_policy:str='any'

@dataclass
class Pairs:
    positive:np.ndarray
    negative:np.ndarray
    weight:np.ndarray
    distance:np.ndarray
    audit:dict

    def __len__(self):return len(self.positive)


def _context_allowed(positive:dict,negative:dict,policy:str)->bool:
    if policy=='any':return True
    same=group_key(positive)==group_key(negative)
    return same if policy=='same_group' else not same


def _candidates(a:np.ndarray,y:np.ndarray,meta:list[dict],cfg:MatchConfig):
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if not len(pos) or not len(neg):return pos,[]
    if cfg.context_policy!='any':
        result=[]
        for i in pos:
            legal=np.asarray([j for j in neg if track_key(meta[i])!=track_key(meta[j]) and
                              _context_allowed(meta[i],meta[j],cfg.context_policy)],dtype=int)
            if not len(legal):
                result.append([]);continue
            count=min(len(legal),cfg.candidate_neighbors)
            distances,neighbors=NearestNeighbors(n_neighbors=count).fit(a[legal]).kneighbors(a[i:i+1])
            keep=[]
            for distance,j in zip(distances[0],legal[neighbors[0]]):
                if np.max(np.abs(a[i]-a[j]))>cfg.max_abs_z:continue
                keep.append((int(j),float(distance/np.sqrt(a.shape[1]))))
            result.append(keep)
        return pos,result
    nn=NearestNeighbors(n_neighbors=min(len(neg),cfg.candidate_neighbors)).fit(a[neg])
    ds,js=nn.kneighbors(a[pos]); result=[]
    for i,dd,jj in zip(pos,ds,js):
        keep=[]
        for distance,j in zip(dd,neg[jj]):
            if track_key(meta[i])==track_key(meta[j]):continue
            if not _context_allowed(meta[i],meta[j],cfg.context_policy):continue
            if np.max(np.abs(a[i]-a[j]))>cfg.max_abs_z:continue
            keep.append((int(j),float(distance/np.sqrt(a.shape[1]))))
        result.append(keep)
    return pos,result


def fit_radius(a,y,meta,cfg:MatchConfig)->MatchConfig:
    if cfg.k<1 or cfg.bandwidth<=0 or cfg.max_negative_track_uses<1 or not 0<=cfg.radius_quantile<=1:
        raise ValueError('Invalid matching parameters')
    if cfg.context_policy not in ['any','same_group','different_group']:
        raise ValueError('Unknown matching context policy')
    cfg=MatchConfig(**asdict(cfg))
    if cfg.radius is None:
        _,choices=_candidates(a,y,meta,cfg)
        distances=[c[0][1] for c in choices if c]
        cfg.radius=float(np.quantile(distances,cfg.radius_quantile)) if distances else 0.0
    if cfg.radius<0:raise ValueError('Negative matching radius')
    return cfg


def match(a,y,meta,cfg:MatchConfig,seed:int=42,mode:str='geometry')->Pairs:
    if cfg.radius is None:raise ValueError('Matching radius must first be fit on source training data')
    if mode not in ['geometry','random']:raise ValueError('Unknown matching mode')
    if mode=='random':
        # Preserve the exact geometric control's positive counts, weights and negative-track usage.
        reference=match(a,y,meta,cfg,seed,mode='geometry')
        rng=np.random.default_rng(seed+1009);negative=reference.negative.copy()
        for _ in range(max(1,8*len(negative))):
            if len(negative)<2:break
            i,j=rng.integers(0,len(negative),size=2)
            pi,pj=reference.positive[i],reference.positive[j]
            if (track_key(meta[pi])==track_key(meta[negative[j]]) or
                    track_key(meta[pj])==track_key(meta[negative[i]]) or
                    not _context_allowed(meta[pi],meta[negative[j]],cfg.context_policy) or
                    not _context_allowed(meta[pj],meta[negative[i]],cfg.context_policy)):continue
            negative[i],negative[j]=negative[j],negative[i]
        distance=np.linalg.norm(a[reference.positive]-a[negative],axis=1)/np.sqrt(a.shape[1])
        audit={**reference.audit,'mode':'random_count_weight_reuse_matched',
               'mean_distance':float(distance.mean()) if len(distance) else None,
               'p95_distance':float(np.quantile(distance,.95)) if len(distance) else None,
               'changed_negative_fraction':float(np.mean(negative!=reference.negative)) if len(negative) else None,
               'weight_policy':'Reference geometric weights preserved to isolate pairing identity'}
        return Pairs(reference.positive.copy(),negative,reference.weight.copy(),distance,audit)
    pos,choices=_candidates(a,y,meta,cfg);rng=np.random.default_rng(seed)
    out=[];counts=Counter()
    for pidx in rng.permutation(len(pos)):
        i=int(pos[pidx]);candidates=choices[pidx]
        if mode=='geometry':
            candidates=[c for c in candidates if c[1]<=cfg.radius+1e-8]
        else:
            # Random-label-opposite control; same maximum pairs and track reuse constraint.
            neg=np.flatnonzero(y==0)
            candidates=[(int(j),float(np.linalg.norm(a[i]-a[j])/np.sqrt(a.shape[1])))
                        for j in rng.permutation(neg) if track_key(meta[i])!=track_key(meta[j])]
        count=0
        for j,d in candidates:
            track=track_key(meta[j])
            if counts[track]>=cfg.max_negative_track_uses:continue
            w=float(np.exp(-min((d/cfg.bandwidth)**2,60))) if mode=='geometry' else 1.
            out.append((i,j,w,d));counts[track]+=1;count+=1
            if count>=cfg.k:break
    ar=np.asarray(out,dtype=float).reshape(-1,4)
    coverage=len(set(ar[:,0].astype(int)))/max(len(pos),1)
    audit={'n_pairs':len(ar),'n_positive':len(pos),'positive_coverage':coverage,
           'mean_distance':float(ar[:,3].mean()) if len(ar) else None,
           'p95_distance':float(np.quantile(ar[:,3],.95)) if len(ar) else None,
           'unique_negative_tracks':len(counts),'max_negative_track_uses_observed':max(counts.values(),default=0),
           'config':asdict(cfg),'mode':mode,'seed':seed}
    return Pairs(ar[:,0].astype(int),ar[:,1].astype(int),ar[:,2].astype(np.float32),ar[:,3],audit)


def accuracy(scores:np.ndarray,pairs:Pairs)->float|None:
    if not len(pairs):return None
    diff=scores[pairs.positive]-scores[pairs.negative]
    return float(np.mean((diff>0)+.5*(diff==0)))
