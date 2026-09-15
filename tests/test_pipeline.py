import copy
import numpy as np
import torch
import pytest
from acie.data import prepare,make_split,resolve_split
from acie.demo import make_fixture
from acie.engine import train,predict,compare_predictions
from acie.io import read_jsonl

def test_train_predict_and_resume(tmp_path):
    b=prepare(make_fixture(tmp_path/'fixture',groups=8,per_group=6),tmp_path/'bundle.npz')
    split=make_split(b,'synthetic_A','synthetic_B')
    cfg={'hidden':8,'epochs':2,'geometry_epochs':2,'patience':3,'num_threads':1,'class_weighted':True}
    report=train(b,split,cfg,tmp_path/'model')
    assert report['status']=='complete'
    result=predict(b,tmp_path/'model'/'best.pt',tmp_path/'predictions.jsonl',repeats=20)
    assert result['synthetic'] is True
    assert result['metrics']['n']==len(split['test'])
    train(b,split,cfg,tmp_path/'model',resume=True)
    predict(b,tmp_path/'model'/'best.pt',tmp_path/'predictions2.jsonl',repeats=20)
    a=read_jsonl(tmp_path/'predictions.jsonl');c=read_jsonl(tmp_path/'predictions2.jsonl')
    np.testing.assert_allclose([r['score'] for r in a],[r['score'] for r in c],atol=1e-7)
    assert compare_predictions(tmp_path/'predictions.jsonl',tmp_path/'predictions2.jsonl',20)['interval']==[0.,0.]
    with pytest.raises(ValueError):train(b,split,{**cfg,'hidden':10},tmp_path/'model',resume=True)

def test_metadata_does_not_change_features(tmp_path):
    from acie.io import read_jsonl,write_jsonl
    manifest=make_fixture(tmp_path/'fixture',groups=8,per_group=4)
    b1=prepare(manifest,tmp_path/'one.npz');rows=read_jsonl(manifest)
    for r in rows:r['oracle_label']=1-r['label'];r['future_interaction_secret']=1e9
    write_jsonl(manifest,rows);b2=prepare(manifest,tmp_path/'two.npz')
    np.testing.assert_array_equal(b1.a,b2.a);np.testing.assert_array_equal(b1.q,b2.q)
