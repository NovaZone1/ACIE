"""Merge separately exported pools/domains without merging labels or identities."""
import argparse
import numpy as np
from acie.data import Bundle
from acie.io import digest_file
p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
bs=[Bundle.load(x) for x in a.inputs]
if len({b.q.shape[1:] for b in bs})!=1:raise ValueError('Mismatched frame/feature dimensions')
b=Bundle(np.concatenate([x.a for x in bs]),np.concatenate([x.q for x in bs]),np.concatenate([x.y for x in bs]),
         sum([x.meta for x in bs],[]),{'parents':[x.provenance for x in bs],
         'parent_sha256':[digest_file(x) for x in a.inputs],'synthetic':all(x.provenance.get('synthetic',False) for x in bs)})
b.validate();b.save(a.out)
print(f'Merged {len(b.y)} samples')
