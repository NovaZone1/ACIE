"""Deterministic synthetic integration fixtures. NEVER a research benchmark."""
from pathlib import Path
import numpy as np
from .io import write_json,write_jsonl
from .data import prepare,make_split
from .engine import train,predict


def make_fixture(root,seed=9,groups=10,per_group=12,frames=16):
    root=Path(root);(root/'windows').mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);rows=[]
    for domain in ['synthetic_A','synthetic_B']:
        for g in range(groups):
            # Each recording has both outcomes, with paired geometry by construction for testing.
            for k in range(per_group):
                label=k%2;times=np.arange(frames,dtype=float)/15
                pair=k//2;local_rng=np.random.default_rng(g*31+pair)
                cx=.5+.04*local_rng.normal()+.02*times
                height=.35+.03*local_rng.normal()+.05*times;width=height*.45
                cy=np.full(frames,.5);bbox=np.stack([cx-width/2,cy-height/2,cx+width/2,cy+height/2],1)
                kp=np.zeros((frames,17,3))
                base=np.stack([np.linspace(-.32,.32,17),np.linspace(-.4,.4,17)],1)
                relative=base[None]+rng.normal(0,.012,(frames,17,2))
                # Artificial local wrist pattern used only to test learnability and gradients.
                relative[:,9,0]+=(2*label-1)*.2*np.linspace(0,1,frames)
                relative[:,10,1]+=(2*label-1)*.15
                kp[...,0]=cx[:,None]+relative[...,0]*width[:,None]
                kp[...,1]=cy[:,None]+relative[...,1]*height[:,None];kp[...,2]=.95
                sid=f'{domain}_g{g}_t{k}'
                np.savez_compressed(root/'windows'/f'{sid}.npz',keypoints=kp,bbox=bbox,
                                    mask_area=width*height*.75,times=times)
                rows.append({'sample_id':sid,'dataset':domain,'recording':f'g{g}','track_id':sid,
                             'pool':'train' if g<groups-3 else 'test','label':label,
                             'window':f'windows/{sid}.npz','panoramic':True,'synthetic':True})
    write_jsonl(root/'manifest.jsonl',rows)
    return root/'manifest.jsonl'


def run_demo(out):
    out=Path(out)
    if (out/'model'/'run.json').exists():raise FileExistsError('Demo output exists; select a fresh --out')
    manifest=make_fixture(out/'fixture')
    b=prepare(manifest,out/'bundle.npz');split=make_split(b,'synthetic_A','synthetic_B')
    write_json(out/'split.json',split)
    report=train(b,split,{'epochs':5,'geometry_epochs':4,'hidden':24,'patience':5,
                         'matching':{'radius_quantile':1.,'max_abs_z':6.}},out/'model')
    result=predict(b,out/'model'/'best.pt',out/'predictions.jsonl',repeats=100)
    write_json(out/'DEMO_ONLY.json',{'purpose':'Synthetic end-to-end integration test, NOT HUI360/SSUP-A results',
                                    'training':report['status'],'test':result})
    return result
