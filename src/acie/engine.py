"""Two-stage source-only training, checkpointing, inference, and diagnostic pairing."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import copy
import json
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import Bundle,resolve_split,group_key,track_key
from .features import SourceScaler
from .matching import MatchConfig,fit_radius,match,accuracy
from .models import Predictor,pair_loss
from .metrics import binary_metrics,choose_threshold,bootstrap
from .io import write_json,write_jsonl,digest_object,environment,read_jsonl

DEFAULT_CONFIG={'seed':42,'kind':'full','hidden':64,'dropout':.1,'epochs':50,'geometry_epochs':30,
                'batch_size':64,'pair_batch_size':64,'lr':.001,'weight_decay':.0005,'patience':10,
                'pair_lambda':.3,'resample_mix':.5,'class_weighted':False,
                'device':'cpu','num_threads':2,'matching':{}}


def matched_resampling_order(y,pairs,rng,mix=.5):
    """Resample matched endpoints while preserving the observed class counts exactly."""
    if not 0<=mix<=1:raise ValueError('resample_mix must be in [0,1]')
    counts=np.zeros(len(y),dtype=np.float64)
    np.add.at(counts,pairs.positive,1);np.add.at(counts,pairs.negative,1)
    parts=[]
    for label in [0,1]:
        ix=np.flatnonzero(y==label)
        if not len(ix):continue
        endpoint=counts[ix]
        endpoint=endpoint/endpoint.sum() if endpoint.sum() else np.full(len(ix),1/len(ix))
        probability=(1-mix)/len(ix)+mix*endpoint
        parts.append(rng.choice(ix,len(ix),replace=True,p=probability))
    return rng.permutation(np.concatenate(parts)).astype(int) if parts else np.empty(0,dtype=int)


def set_seed(seed:int,threads:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def _predict(model,a,q,ix,device,batch_size=256):
    model.eval();parts=[[],[],[]]
    with torch.inference_mode():
        for start in range(0,len(ix),batch_size):
            chunk=ix[start:start+batch_size]
            out=model(torch.from_numpy(a[chunk]).to(device),torch.from_numpy(q[chunk]).to(device))
            for slot,value in zip(parts,out):slot.append(value.cpu().numpy())
    return [np.concatenate(p) if p else np.empty(0,dtype=np.float32) for p in parts]


def _sigmoid(x):return 1/(1+np.exp(-np.clip(x,-60,60)))


def fit_stage(model,a,q,y,tr,va,cfg,pairs,out,stage,epochs,resume=False,track_keys=None):
    device=torch.device(cfg['device']);model.to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(cfg['seed']+(1 if stage=='evidence' else 0))
    best=-1.;best_state=copy.deepcopy(model.state_dict());bad=0;history=[];start_epoch=0
    last=Path(out)/(stage+'.last.pt')
    if resume and last.exists():
        state=torch.load(last,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch']+1;best=state['best'];best_state=state['best_state'];bad=state['bad']
        history=state['history'];rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
        if state.get('cuda_rng') is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
    n=len(tr);local_y=y[tr]
    track_keys=track_keys if track_keys is not None else list(range(n))
    paired_kinds=['full','fusion_pair','random_pair','hard_negative']
    use_pair=stage=='evidence' and cfg['kind'] in paired_kinds and cfg['pair_lambda']>0
    for epoch in range(start_epoch,epochs):
        if bad>=cfg['patience']:break
        model.train()
        if stage=='evidence' and cfg['kind'] in ['full','dual_no_pair','random_pair','hard_negative']:model.geometry.eval()
        order=rng.permutation(n)
        # Resampling control: class pairs affect BCE distribution only in this named baseline.
        if stage=='evidence' and cfg['kind']=='resampled' and len(pairs):
            order=matched_resampling_order(local_y,pairs,rng,cfg['resample_mix'])
        hard_neg=None
        if use_pair and cfg['kind']=='hard_negative':
            pred=_predict(model,a,q,tr,device)[0]
            hard_neg=np.flatnonzero(local_y==0)[np.argsort(pred[local_y==0])[::-1]]
            model.train()
        losses=[]
        for start in range(0,n,cfg['batch_size']):
            local=order[start:start+cfg['batch_size']];ix=tr[local]
            aa=torch.from_numpy(a[ix]).to(device);qq=torch.from_numpy(q[ix]).to(device)
            target=torch.from_numpy(y[ix].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            if stage=='geometry':logit=model.geometry(aa).squeeze(-1)
            else:logit=model(aa,qq)[0]
            pos_weight=None
            if (cfg['kind']=='weighted' or cfg['class_weighted']) and stage=='evidence':
                pos_weight=torch.tensor(float((local_y==0).sum()/max((local_y==1).sum(),1)),device=device)
            loss=F.binary_cross_entropy_with_logits(logit,target,pos_weight=pos_weight)
            if use_pair and (len(pairs) or hard_neg is not None):
                if hard_neg is not None:
                    pi=rng.choice(np.flatnonzero(local_y==1),cfg['pair_batch_size'],replace=True)
                    # Use several highest-scoring source negatives, never validation/test examples.
                    valid_pi=[];valid_ni=[]
                    for positive in pi:
                        legal=[j for j in hard_neg if track_keys[j]!=track_keys[positive]]
                        if legal:
                            valid_pi.append(positive)
                            valid_ni.append(rng.choice(legal[:cfg['pair_batch_size']]))
                    pi=np.asarray(valid_pi,dtype=int);ni=np.asarray(valid_ni,dtype=int)
                    ww=np.ones(len(pi),dtype=np.float32)
                else:
                    draw=rng.integers(0,len(pairs),size=min(cfg['pair_batch_size'],len(pairs)))
                    pi,ni,ww=pairs.positive[draw],pairs.negative[draw],pairs.weight[draw]
                def evidence(local_ix):
                    pos_ix=tr[local_ix]
                    return model(torch.from_numpy(a[pos_ix]).to(device),torch.from_numpy(q[pos_ix]).to(device))[2]
                if len(pi):loss=loss+cfg['pair_lambda']*pair_loss(evidence(pi),evidence(ni),torch.from_numpy(ww).to(device))
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step();losses.append(float(loss.detach().cpu()))
        if stage=='geometry':
            model.eval()
            with torch.inference_mode():
                val=np.concatenate([model.geometry(torch.from_numpy(a[va[s:s+256]]).to(device)).squeeze(-1).cpu().numpy()
                                    for s in range(0,len(va),256)])
        else:val=_predict(model,a,q,va,device)[0]
        metric=binary_metrics(y[va],_sigmoid(val))['AP']
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'source_val_AP':metric})
        if metric>best:
            best=metric;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        tmp=last.with_suffix('.tmp')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'best':best,'best_state':best_state,'bad':bad,'history':history,
                    'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},tmp)
        tmp.replace(last);write_json(Path(out)/(stage+'.history.json'),history)
    model.load_state_dict(best_state);return history


def train(b:Bundle,split:dict,config:dict,out:str|Path,resume:bool=False)->dict:
    cfg=copy.deepcopy(DEFAULT_CONFIG);cfg.update(config)
    unknown=set(cfg)-set(DEFAULT_CONFIG)
    if unknown:raise ValueError(f'Unknown configuration keys: {unknown}')
    if (cfg['epochs']<1 or cfg['geometry_epochs']<1 or cfg['patience']<1 or cfg['batch_size']<1
            or cfg['pair_lambda']<0 or not 0<=cfg['resample_mix']<=1):
        raise ValueError('Invalid training configuration')
    indices=resolve_split(b,split);tr,va=indices['train'],indices['val']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    run_key=digest_object({'config':cfg,'split':split,'data':b.provenance})
    lock=out/'run.json'
    if lock.exists():
        existing=json.loads(lock.read_text())
        if not resume:raise FileExistsError('Output already contains a run; choose a new directory or --resume')
        if existing['run_key']!=run_key:raise ValueError('Resume config/data/split mismatch')
    set_seed(cfg['seed'],cfg['num_threads'])
    if cfg['device'].startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    scaler=SourceScaler.fit(b.a[tr],b.q[tr]);a,q=scaler.transform(b.a,b.q)
    train_meta=[b.meta[i] for i in tr]
    mc=fit_radius(a[tr],b.y[tr],train_meta,MatchConfig(**cfg['matching']))
    pair_mode='random' if cfg['kind']=='random_pair' else 'geometry'
    pairs=match(a[tr],b.y[tr],train_meta,mc,seed=cfg['seed'],mode=pair_mode)
    write_json(out/'matching.json',pairs.audit)
    write_jsonl(out/'pairs.train.jsonl',[{'positive':b.meta[tr[i]]['sample_id'],'negative':b.meta[tr[j]]['sample_id'],
                                        'weight':float(w),'distance':float(d)}
                                       for i,j,w,d in zip(pairs.positive,pairs.negative,pairs.weight,pairs.distance)])
    model=Predictor(a.shape[1],q.shape[2],q.shape[1],cfg['kind'],cfg['hidden'],cfg['dropout'])
    write_json(lock,{'run_key':run_key,'config':cfg,'split':split,'environment':environment(),
                    'synthetic':b.provenance.get('synthetic',False),'status':'started',
                    'selection':'source validation AP only; no test scoring during training'})
    start=time.perf_counter()
    if cfg['kind'] in ['full','dual_no_pair','geometry','random_pair','hard_negative']:
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('geometry.'))
        fit_stage(model,a,q,b.y,tr,va,cfg,pairs,out,'geometry',cfg['geometry_epochs'],resume,[track_key(m) for m in train_meta])
    if cfg['kind']!='geometry':
        for name,p in model.named_parameters():p.requires_grad_(not name.startswith('geometry.'))
        fit_stage(model,a,q,b.y,tr,va,cfg,pairs,out,'evidence',cfg['epochs'],resume,[track_key(m) for m in train_meta])
    val_logits=_predict(model,a,q,va,torch.device(cfg['device']))[0];val_scores=_sigmoid(val_logits)
    threshold=choose_threshold(b.y[va],val_scores)
    payload={'schema':'acie.checkpoint.v1','spec':model.spec,'state':{k:v.cpu() for k,v in model.state_dict().items()},
             'scaler':scaler.to_dict(),'config':cfg,'matching':asdict(mc),'threshold':threshold,
             'split':split,'run_key':run_key,'synthetic':b.provenance.get('synthetic',False),
             'source_val':binary_metrics(b.y[va],val_scores,threshold)}
    torch.save(payload,out/'best.pt')
    report={k:v for k,v in payload.items() if k not in ['state','scaler']}
    report.update(training_seconds=time.perf_counter()-start,
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_final_stage=sum(p.numel() for p in model.parameters() if p.requires_grad),status='complete')
    write_json(out/'training_report.json',report)
    write_json(lock,{'run_key':run_key,'config':cfg,'split':split,'environment':environment(),
                    'synthetic':payload['synthetic'],'status':'complete'})
    return report


def predict(b:Bundle,checkpoint:str|Path,out:str|Path,split_name='test',device='cpu',repeats=500)->dict:
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck.get('schema')!='acie.checkpoint.v1':raise ValueError('Unsupported checkpoint')
    split=resolve_split(b,ck['split']);ix=split[split_name]
    model=Predictor(**ck['spec']);model.load_state_dict(ck['state']);model.to(device)
    scaler=SourceScaler.from_dict(ck['scaler']);a,q=scaler.transform(b.a,b.q)
    start=time.perf_counter();logit,g,r=_predict(model,a,q,ix,device);scores=_sigmoid(logit)
    seconds=time.perf_counter()-start
    rows=[]
    for k,i in enumerate(ix):
        row={**b.meta[i],'label':int(b.y[i]),'score':float(scores[k]),'logit':float(logit[k]),
             'geometry_logit':float(g[k]),'evidence':float(r[k]),'threshold':ck['threshold']}
        rows.append(row)
    write_jsonl(out,rows)
    pairs=match(a[ix],b.y[ix],[b.meta[i] for i in ix],MatchConfig(**ck['matching']),ck['config']['seed'])
    result={'metrics':binary_metrics(b.y[ix],scores,ck['threshold']),
            'cluster_bootstrap':bootstrap(b.y[ix],scores,[group_key(b.meta[i]) for i in ix],repeats),
            'pair_diagnostic':{**pairs.audit,'final_accuracy':accuracy(scores,pairs),
                               'geometry_accuracy':accuracy(g,pairs),'evidence_accuracy':accuracy(r,pairs)},
            'prediction_seconds':seconds,'synthetic':ck['synthetic'],'protocol':ck['split']['protocol'],
            'notice':'Synthetic integration test; not benchmark evidence' if ck['synthetic'] else 'New implementation under documented protocol'}
    write_json(Path(out).with_suffix('.metrics.json'),result);return result


def compare_predictions(first,second,repeats=1000)->dict:
    aa=read_jsonl(first);bb=read_jsonl(second)
    a={r['sample_id']:r for r in aa};b={r['sample_id']:r for r in bb}
    if len(a)!=len(aa) or len(b)!=len(bb) or set(a)!=set(b):raise ValueError('Predictions must contain exactly the same unique sample IDs')
    ids=sorted(a)
    if any(a[i]['label']!=b[i]['label'] for i in ids):raise ValueError('Label mismatch')
    return bootstrap([a[i]['label'] for i in ids],[a[i]['score'] for i in ids],
                     [group_key(a[i]) for i in ids],repeats,other=[b[i]['score'] for i in ids])
