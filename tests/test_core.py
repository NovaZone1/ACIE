import json
from pathlib import Path
import numpy as np
import pytest
import torch
from acie.demo import make_fixture
from acie.data import prepare,make_split,resolve_split,Bundle,group_key
from acie.features import extract,SourceScaler
from acie.matching import MatchConfig,fit_radius,match,accuracy
from acie.models import Predictor,pair_loss
from acie.metrics import binary_metrics,bootstrap
from acie.io import inside,read_jsonl

@pytest.fixture
def bundle(tmp_path):
    return prepare(make_fixture(tmp_path/'fixtures',groups=8,per_group=8),tmp_path/'bundle.npz')

def test_bundle_roundtrip(bundle,tmp_path):
    bundle.save(tmp_path/'copy.npz');b=Bundle.load(tmp_path/'copy.npz')
    np.testing.assert_array_equal(bundle.q,b.q)
    assert b.q.shape[-1]==119 and b.a.shape[-1]==32

def test_group_split_no_leakage(bundle):
    sp=make_split(bundle,'synthetic_A','synthetic_B');ix=resolve_split(bundle,sp)
    assert set(ix['train']).isdisjoint(ix['val'])
    assert all(bundle.meta[i]['dataset']=='synthetic_A' for i in ix['val'])

def test_optional_group_id_is_preserved_and_used(bundle):
    bundle.meta[0]['group_id']='shared-day'
    assert group_key(bundle.meta[0])=='synthetic_A::shared-day'

def test_corrupt_split_rejected(bundle):
    sp=make_split(bundle,'synthetic_A','synthetic_B');sp['val'][0]=sp['train'][0]
    with pytest.raises(ValueError,match='leakage'):resolve_split(bundle,sp)

def test_scaler_source_only(bundle):
    sp=make_split(bundle,'synthetic_A','synthetic_B');ix=resolve_split(bundle,sp)
    sc=SourceScaler.fit(bundle.a[ix['train']],bundle.q[ix['train']]);m=sc.a_mean.copy()
    bundle.a[ix['test']]+=100000
    sc2=SourceScaler.fit(bundle.a[ix['train']],bundle.q[ix['train']]);np.testing.assert_array_equal(m,sc2.a_mean)

def test_matching(bundle):
    tr=resolve_split(bundle,make_split(bundle,'synthetic_A','synthetic_B'))['train']
    sc=SourceScaler.fit(bundle.a[tr],bundle.q[tr]);a,_=sc.transform(bundle.a[tr],bundle.q[tr])
    meta=[bundle.meta[i] for i in tr];cfg=fit_radius(a,bundle.y[tr],meta,MatchConfig(radius_quantile=1.,max_abs_z=6))
    pairs=match(a,bundle.y[tr],meta,cfg)
    assert len(pairs)>0
    assert np.all(bundle.y[tr][pairs.positive]==1) and np.all(bundle.y[tr][pairs.negative]==0)
    assert all(meta[i]['track_id']!=meta[j]['track_id'] for i,j in zip(pairs.positive,pairs.negative))
    assert pairs.audit['max_negative_track_uses_observed']<=cfg.max_negative_track_uses
    assert accuracy(np.zeros(len(tr)),pairs)==.5

def test_empty_matching():
    a=np.zeros((3,2));y=np.zeros(3,dtype=int);m=[{'dataset':'d','track_id':str(i)} for i in range(3)]
    cfg=fit_radius(a,y,m,MatchConfig());assert len(match(a,y,m,cfg))==0

def test_matching_context_policy():
    a=np.array([[0.],[.05],[.1],[.15]],dtype=np.float32);y=np.array([1,0,1,0])
    m=[{'dataset':'d','recording':group,'track_id':str(i),'group_id':group}
       for i,group in enumerate(['a','a','b','b'])]
    for policy,expected_same in [('same_group',True),('different_group',False)]:
        cfg=fit_radius(a,y,m,MatchConfig(k=1,radius_quantile=1,max_abs_z=8,context_policy=policy))
        pairs=match(a,y,m,cfg)
        assert len(pairs)==2
        assert all((group_key(m[i])==group_key(m[j])) is expected_same
                   for i,j in zip(pairs.positive,pairs.negative))
    with pytest.raises(ValueError,match='context policy'):
        fit_radius(a,y,m,MatchConfig(context_policy='unknown'))

def test_pair_loss_gradient_and_empty():
    p=torch.tensor([0.],requires_grad=True);n=torch.tensor([1.],requires_grad=True)
    loss=pair_loss(p,n,torch.ones(1));loss.backward();assert p.grad.item()<0 and n.grad.item()>0
    assert pair_loss(torch.empty(0),torch.empty(0),torch.empty(0)).item()==0

@pytest.mark.parametrize('kind',['geometry','behavior','full','dual_no_pair','fusion_mlp','fusion_pair','random_pair','hard_negative','resampled','weighted','lstm','graph'])
def test_models(kind):
    torch.set_num_threads(1)
    model=Predictor(32,119,8,kind,hidden=8,dropout=0)
    out=model(torch.randn(3,32),torch.randn(3,8,119))
    assert all(v.shape==(3,) for v in out)
    out[0].mean().backward()

def test_metrics_degenerate_and_bootstrap():
    assert binary_metrics([0,0],[.1,.2])['AUROC'] is None
    assert binary_metrics([1,0],[.9,.1])['AP']==1
    r=bootstrap([1,0,1,0],[.9,.2,.8,.1],['a','a','b','b'],20)
    assert r['interval']==[1.,1.]

def test_path_escape(tmp_path):
    with pytest.raises(ValueError):inside(tmp_path,'../secret')

def test_prefix_and_missing_points():
    t=np.arange(4)/15;box=np.tile([.4,.2,.6,.8],(4,1));kp=np.tile([.5,.5,.9],(4,17,1))
    kp[0,0]=[np.nan,np.nan,0]
    a,q=extract(kp,box,np.ones(4)*.1,t);assert np.isfinite(q).all()
    assert q[0,5]==0 and q[1,6]==0
    with pytest.raises(ValueError):extract(kp,box,np.ones(4)*.1,np.zeros(4))

def test_panorama_seam():
    t=np.arange(4)/15;box=np.array([[.95,.2,.05,.8]]*4);kp=np.tile([.99,.5,.9],(4,17,1))
    a,q=extract(kp,box,np.ones(4)*.04,t,panoramic=True)
    assert np.abs(q[:,0]).max()<1
    with pytest.raises(ValueError):extract(kp,box,np.ones(4)*.04,t,panoramic=False)
