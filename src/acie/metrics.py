from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve


def binary_metrics(y,s,threshold=.5)->dict:
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    if len(y)==0 or len(y)!=len(s) or not np.isfinite(s).all():raise ValueError('Invalid prediction arrays')
    positives=int(y.sum());negatives=int(len(y)-positives)
    pred=s>=threshold;tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)))
    fn=int(np.sum((~pred)&(y==1)));tn=int(np.sum((~pred)&(y==0)))
    return {'n':len(y),'positives':positives,'prevalence':positives/len(y),
            'AP':float(average_precision_score(y,s)) if positives else None,
            'AUROC':float(roc_auc_score(y,s)) if positives and negatives else None,
            'threshold':float(threshold),'recall':tp/positives if positives else None,
            'false_positive_rate':fp/negatives if negatives else None,
            'precision':tp/(tp+fp) if tp+fp else 0.,
            'TP':tp,'FP':fp,'FN':fn,'TN':tn}


def choose_threshold(y,s)->float:
    p,r,t=precision_recall_curve(y,s)
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    return float(t[np.argmax(f)]) if len(t) else .5


def bootstrap(y,s,groups,repeats=1000,seed=42,other=None)->dict:
    y=np.asarray(y);s=np.asarray(s);groups=np.asarray(groups);unique=np.unique(groups)
    if len(unique)<2:return {'groups':len(unique),'interval':None,'reason':'fewer than two groups'}
    rng=np.random.default_rng(seed);rows={g:np.flatnonzero(groups==g) for g in unique};values=[]
    for _ in range(repeats):
        ix=np.concatenate([rows[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ix]))<2:continue
        value=average_precision_score(y[ix],s[ix])
        if other is not None:value-=average_precision_score(y[ix],np.asarray(other)[ix])
        values.append(float(value))
    return {'groups':len(unique),'replicates_requested':repeats,'replicates_valid':len(values),
            'interval':np.quantile(values,[.025,.975]).tolist() if values else None,
            'metric':'AP difference' if other is not None else 'AP',
            'warning':'Few groups: percentile interval may be unstable' if len(unique)<10 else None}
