"""Source-validation-only grid. Writes best checkpoint path; never calls test prediction."""
import argparse
import copy
from pathlib import Path
import yaml
from acie.data import Bundle
from acie.engine import train
from acie.io import read_json,write_json
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--split',required=True)
p.add_argument('--config',required=True);p.add_argument('--out',required=True)
p.add_argument('--lambdas',nargs='+',type=float,default=[0.,.1,.3,1.]);p.add_argument('--hidden',nargs='+',type=int,default=[32,64]);a=p.parse_args()
b=Bundle.load(a.data);sp=read_json(a.split);cfg=yaml.safe_load(Path(a.config).read_text());trials=[]
for h in a.hidden:
    for lam in a.lambdas:
        c=copy.deepcopy(cfg);c['hidden']=h;c['pair_lambda']=lam;out=Path(a.out)/f'h{h}_lambda{lam:g}'
        r=train(b,sp,c,out);trials.append({'hidden':h,'lambda':lam,'source_val_AP':r['source_val']['AP'],'checkpoint':str(out/'best.pt')})
write_json(Path(a.out)/'source_tuning.json',{'trials':trials,'selected':max(trials,key=lambda x:x['source_val_AP']),
                                          'test_evaluated':False})
