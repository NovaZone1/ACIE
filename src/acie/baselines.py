"""Source-selected tree baseline. No target early stopping. Own implementation."""
from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import average_precision_score
from .data import resolve_split,group_key
from .features import SourceScaler
from .metrics import choose_threshold,binary_metrics,bootstrap
from .io import write_json,write_jsonl


def run_tree(b,split,out,kind='histgb',seed=42,repeats=500):
    import joblib
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'tree.joblib').exists():raise FileExistsError('Choose a fresh tree output directory')
    ix=resolve_split(b,split);tr,va,te=ix['train'],ix['val'],ix['test']
    sc=SourceScaler.fit(b.a[tr],b.q[tr]);a,q=sc.transform(b.a,b.q)
    # Same available features; temporal statistics keep tree input tractable.
    x=np.concatenate([a,q[:,-1],q.mean(1),q.std(1)],1)
    candidates=[]; fit_start=time.perf_counter()
    for leaves in [7,15,31]:
        if kind=='histgb':
            model=HistGradientBoostingClassifier(max_leaf_nodes=leaves,max_iter=150,early_stopping=False,
                                                l2_regularization=1.,random_state=seed)
        elif kind=='random_forest':
            model=RandomForestClassifier(n_estimators=200,max_depth=leaves,class_weight='balanced',n_jobs=2,random_state=seed)
        else:raise ValueError('Unknown tree baseline')
        model.fit(x[tr],b.y[tr]);vs=model.predict_proba(x[va])[:,1]
        candidates.append((float(average_precision_score(b.y[va],vs)),model,vs,leaves))
    fit_seconds=time.perf_counter()-fit_start
    score,model,vs,leaves=max(candidates,key=lambda v:v[0]);threshold=choose_threshold(b.y[va],vs)
    joblib.dump({'model':model,'scaler':sc.to_dict(),'split':split,'threshold':threshold},out/'tree.joblib')
    predict_start=time.perf_counter();ps=model.predict_proba(x[te])[:,1];prediction_seconds=time.perf_counter()-predict_start
    rows=[{**b.meta[i],'label':int(b.y[i]),'score':float(s),'threshold':threshold} for i,s in zip(te,ps)]
    write_jsonl(out/'predictions.jsonl',rows)
    result={'source_val_AP':score,'selected_leaf_or_depth':leaves,'metrics':binary_metrics(b.y[te],ps,threshold),
            'bootstrap':bootstrap(b.y[te],ps,[group_key(b.meta[i]) for i in te],repeats),
            'fit_selection_seconds':fit_seconds,'prediction_seconds':prediction_seconds,
            'feature_dim':int(x.shape[1]),'serialized_bytes':int((out/'tree.joblib').stat().st_size),
            'candidate_source_val_AP':{str(item[3]):item[0] for item in candidates},
            'synthetic':b.provenance.get('synthetic',False),'protocol':split['protocol'],
            'warning':'joblib checkpoints must only be loaded from trusted sources'}
    write_json(out/'metrics.json',result);return result
