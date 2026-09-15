"""Prefix-only feature extraction. All input coordinates/areas are image-normalized.

bbox is [xmin, ymin, xmax, ymax]; xmin>xmax is permitted only for panorama.
No labels, onset timestamps, track identifiers or target statistics are features.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

JOINTS = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder',
          'right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
          'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
GEOMETRY_CHANNELS = ['sin_x','cos_x','center_y','log_width','log_height',
                     'log_box_area','log_mask_area','angular_velocity']
GEOMETRY_NAMES = [f'{stat}:{name}' for stat in ['last','mean','std','slope'] for name in GEOMETRY_CHANNELS]
BEHAVIOR_CHANNELS = ['relative_x','relative_y','confidence','velocity_x','velocity_y','valid','velocity_valid']


def extract(keypoints: np.ndarray, bbox: np.ndarray, mask_area: np.ndarray,
            times: np.ndarray, *, panoramic: bool = True, confidence: float = .05) -> tuple[np.ndarray,np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    mask_area = np.asarray(mask_area, dtype=np.float64)
    t = len(times)
    if t < 2 or keypoints.shape != (t,17,3) or bbox.shape != (t,4) or mask_area.shape != (t,):
        raise ValueError('Expected T>=2, keypoints[T,17,3], bbox[T,4], mask_area[T], times[T]')
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('Times must be finite and strictly increasing, in seconds')
    if not np.all(np.isfinite(bbox)) or np.any(bbox < 0) or np.any(bbox > 1):
        raise ValueError('bbox must be finite and normalized into [0,1]')
    w = bbox[:,2]-bbox[:,0]
    if panoramic:
        w = np.where(w < 0, w+1, w)
    h = bbox[:,3]-bbox[:,1]
    if np.any(w <= 1e-6) or np.any(h <= 1e-6):
        raise ValueError('Degenerate or inverted bbox')
    if not np.all(np.isfinite(mask_area)) or np.any(mask_area < 0) or np.any(mask_area > 1):
        raise ValueError('mask_area must be finite image area fractions [0,1]')
    cx = bbox[:,0]+w/2
    if panoramic:
        cx %= 1
    cy = (bbox[:,1]+bbox[:,3])/2
    theta = 2*np.pi*cx
    unwrapped = np.unwrap(theta) if panoramic else theta
    av = np.zeros(t); av[1:] = np.diff(unwrapped)/np.diff(times)
    geom = np.stack([np.sin(theta),np.cos(theta),cy,np.log(w),np.log(h),
                     np.log(w*h),np.log(np.maximum(mask_area,1e-8)),av],axis=1)
    centered_time = times-times.mean()
    slope = (centered_time[:,None]*geom).sum(0)/np.square(centered_time).sum()
    a = np.concatenate([geom[-1],geom.mean(0),geom.std(0),slope]).astype(np.float32)
    valid = np.isfinite(keypoints).all(-1) & (keypoints[...,2] >= confidence)
    valid &= ((keypoints[...,:2] >= 0) & (keypoints[...,:2] <= 1)).all(-1)
    dx = keypoints[...,0]-cx[:,None]
    if panoramic:
        dx = (dx+.5)%1-.5
    xy = np.stack([dx/w[:,None],(keypoints[...,1]-cy[:,None])/h[:,None]],-1)
    xy = np.where(valid[...,None],xy,0)
    vel = np.zeros_like(xy)
    vv = np.zeros_like(valid)
    vv[1:] = valid[1:] & valid[:-1]
    vel[1:] = (xy[1:]-xy[:-1])/np.diff(times)[:,None,None]
    vel = np.where(vv[...,None],vel,0)
    conf = np.where(valid,np.clip(keypoints[...,2],0,1),0)
    q = np.concatenate([xy,conf[...,None],vel,valid[...,None],vv[...,None]],axis=-1)
    q = q.reshape(t,17*7).astype(np.float32)
    if not np.isfinite(a).all() or not np.isfinite(q).all():
        raise ValueError('Feature extraction produced non-finite values')
    return a,q


@dataclass
class SourceScaler:
    a_mean: np.ndarray
    a_scale: np.ndarray
    q_mean: np.ndarray
    q_scale: np.ndarray

    @classmethod
    def fit(cls,a:np.ndarray,q:np.ndarray) -> 'SourceScaler':
        if len(a)==0 or len(a)!=len(q):
            raise ValueError('Empty or inconsistent training arrays')
        # Fit valid coordinates and valid velocities only; confidence/masks stay unscaled.
        qq=q.reshape(*q.shape[:2],17,7)
        mean=np.zeros((17,7)); scale=np.ones((17,7))
        for j in range(17):
            for c in [0,1,3,4]:
                mask=qq[:,:,j,5 if c in [0,1] else 6]>.5
                values=qq[:,:,j,c][mask]
                if len(values):
                    mean[j,c]=values.mean(); scale[j,c]=max(float(values.std()),1e-5)
        return cls(a.mean(0),np.maximum(a.std(0),1e-5),mean.reshape(-1),scale.reshape(-1))

    def transform(self,a:np.ndarray,q:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        z=(q-self.q_mean)/self.q_scale
        shape=z.shape
        z=z.reshape(*shape[:2],17,7); raw=q.reshape(*shape[:2],17,7)
        z[...,:2]*=raw[...,5,None]
        z[...,3:5]*=raw[...,6,None]
        return ((a-self.a_mean)/self.a_scale).astype(np.float32),z.reshape(shape).astype(np.float32)

    def to_dict(self)->dict:
        return {k:v.tolist() for k,v in vars(self).items()}

    @classmethod
    def from_dict(cls,d:dict)->'SourceScaler':
        return cls(**{k:np.asarray(v,dtype=np.float32) for k,v in d.items()})
