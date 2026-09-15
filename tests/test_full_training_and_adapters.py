from pathlib import Path
from types import SimpleNamespace
import importlib.util
import numpy as np
import pytest
from acie.demo import make_fixture
from acie.data import Bundle,prepare,make_split
from acie.engine import train,predict
from acie.matching import fit_radius,match,MatchConfig
from acie.features import SourceScaler,JOINTS
from acie.io import read_jsonl
from acie.engine import matched_resampling_order

@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    root=tmp_path_factory.mktemp('train_matrix')
    bundle=prepare(make_fixture(root/'fixture',groups=7,per_group=6,frames=8),root/'bundle.npz')
    return bundle,make_split(bundle,'synthetic_A','synthetic_B'),root

@pytest.mark.parametrize('kind',['full','dual_no_pair','geometry','behavior','fusion_mlp','fusion_pair','random_pair',
                                 'hard_negative','resampled','weighted','lstm','graph'])
def test_each_training_branch(dataset,kind):
    bundle,split,root=dataset
    report=train(bundle,split,{'kind':kind,'epochs':1,'geometry_epochs':1,'hidden':8,'batch_size':16,'pair_batch_size':8,
        'matching':{'max_abs_z':8.,'radius_quantile':1.}},root/kind)
    assert report['status']=='complete'
    result=predict(bundle,root/kind/'best.pt',root/kind/'predictions.jsonl',repeats=5)
    assert np.isfinite(result['metrics']['AP']) and result['synthetic']

def test_random_control_exact_count_weight_reuse(dataset):
    from acie.data import resolve_split
    b,split,_=dataset;ix=resolve_split(b,split)['train'];sc=SourceScaler.fit(b.a[ix],b.q[ix]);a,_=sc.transform(b.a,b.q)
    meta=[b.meta[i] for i in ix];cfg=fit_radius(a[ix],b.y[ix],meta,MatchConfig(radius_quantile=1,max_abs_z=8.))
    base=match(a[ix],b.y[ix],meta,cfg);control=match(a[ix],b.y[ix],meta,cfg,mode='random')
    assert np.array_equal(base.positive,control.positive) and np.array_equal(base.weight,control.weight)
    assert sorted(base.negative)==sorted(control.negative)

def test_matched_resampling_preserves_class_counts(dataset):
    from acie.data import resolve_split
    b,split,_=dataset;ix=resolve_split(b,split)['train'];sc=SourceScaler.fit(b.a[ix],b.q[ix]);a,_=sc.transform(b.a,b.q)
    meta=[b.meta[i] for i in ix];cfg=fit_radius(a[ix],b.y[ix],meta,MatchConfig(radius_quantile=1,max_abs_z=8.))
    pairs=match(a[ix],b.y[ix],meta,cfg)
    order=matched_resampling_order(b.y[ix],pairs,np.random.default_rng(7),.5)
    assert len(order)==len(ix)
    np.testing.assert_array_equal(np.bincount(b.y[ix][order],minlength=2),np.bincount(b.y[ix],minlength=2))

@pytest.mark.parametrize('kind',['histgb','random_forest'])
def test_tree_strong_baselines(dataset,kind):
    from acie.baselines import run_tree
    b,split,root=dataset;result=run_tree(b,split,root/kind,kind)
    assert np.isfinite(result['metrics']['AP']) and result['synthetic']

def test_official_raw_window_adapter_contract(tmp_path):
    pd=pytest.importorskip('pandas')
    script=Path(__file__).resolve().parents[1]/'scripts/export_hui360.py'
    spec=importlib.util.spec_from_file_location('export_hui360',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    n=8;data={'image_width':[100]*n,'image_height':[100]*n,'recording':['r1']*n,'image_index':list(range(n)),
             'xmin':[30]*n,'xmax':[60]*n,'ymin':[15]*n,'ymax':[90]*n,'mask_size':[1800]*n}
    for j in JOINTS:
        for c,value in [('x',45),('y',50),('score',.9)]:data[f'vitpose_{j}_{c}']=[value]*n
    ds=SimpleNamespace(input_by_tracks={'t1':[(0,n,1)]},datasets_by_unique_track_identifier={'t1':pd.DataFrame(data)},
        idx_to_unique_track_identifier=['t1'],inputs_per_track_stride=-1,fix_index_per_track=True,
        index_choice_by_unique_track_identifier={'t1':0},subsample_frames=1)
    module.export_dataset(ds,tmp_path/'export','HUI360','train',15,True)
    b=prepare(tmp_path/'export/manifest.jsonl',tmp_path/'export/bundle.npz')
    assert len(b.y)==1 and b.y[0]==1 and b.q.shape==(1,n,119)
    assert b.meta[0]['source_index']['frame_indexes']==list(range(n))


def test_horizon_shift_contract():
    pd=pytest.importorskip('pandas')
    script=Path(__file__).resolve().parents[1]/'scripts/prepare_horizon_bundles.py'
    spec=importlib.util.spec_from_file_location('prepare_horizon_bundles',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    n=48
    data={'image_index':np.arange(n),'validity':['valid']*n,'mask_size':[2000.]*n}
    for joint in module.JOINTS:data[f'vitpose_{joint}_score']=[.9]*n
    track=pd.DataFrame(data)
    frame,reason=module.locate_shifted(track,list(range(8,40)),8)
    assert reason is None and frame['image_index'].tolist()==list(range(32))
    frame,reason=module.locate_shifted(track,list(range(8,40)),9)
    assert frame is None and reason=='insufficient_history'
    track.loc[3,'validity']='invalid'
    frame,reason=module.locate_shifted(track,list(range(8,40)),8)
    assert frame is None and reason=='invalid_flag'

def test_pair_identity_controls_match_full_architecture():
    from acie.models import Predictor
    full=Predictor(32,119,32,'full',64,.1)
    for kind in ['random_pair','hard_negative']:
        control=Predictor(32,119,32,kind,64,.1)
        assert list(control.state_dict())==list(full.state_dict())
        assert [tuple(value.shape) for value in control.state_dict().values()]==[tuple(value.shape) for value in full.state_dict().values()]
        assert sum(parameter.numel() for parameter in control.parameters())==sum(parameter.numel() for parameter in full.parameters())
