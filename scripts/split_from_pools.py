"""Keep an exported official test pool; reserve source development recordings from train only.
This is a named derived protocol, not a claim of identical official validation selection.
"""
import argparse
import numpy as np
from acie.data import Bundle,holdout,resolve_split
from acie.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True)
    p.add_argument('--source',required=True);p.add_argument('--target',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--val-fraction',type=float,default=.2);a=p.parse_args()
    b=Bundle.load(a.data)
    train=np.array([i for i,m in enumerate(b.meta) if m['dataset']==a.source and m['pool']=='train'],dtype=int)
    test=np.array([i for i,m in enumerate(b.meta) if m['dataset']==a.target and m['pool']=='test'],dtype=int)
    tr,va=holdout(train,b,a.val_fraction,a.seed)
    result={'schema':'acie.split.v1','source':a.source,'target':a.target,'seed':a.seed,
            'protocol':'fixed_exported_test_pool_source_group_dev',
            'train':[b.meta[i]['sample_id'] for i in tr],'val':[b.meta[i]['sample_id'] for i in va],
            'test':[b.meta[i]['sample_id'] for i in test]}
    resolve_split(b,result);write_json(a.out,result)
    print({k:len(result[k]) for k in ['train','val','test']})

if __name__=='__main__':main()
