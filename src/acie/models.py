"""Trainable reference implementations; not upstream official checkpoints."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def mlp(d:int,h:int,dropout:float=0.)->nn.Sequential:
    return nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(dropout),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))

class CausalConv(nn.Module):
    def __init__(self,cin,cout,dilation=1):
        super().__init__(); self.pad=2*dilation
        self.conv=nn.Conv1d(cin,cout,3,dilation=dilation)
    def forward(self,x):return self.conv(F.pad(x,(self.pad,0)))

class BehaviorTCN(nn.Module):
    def __init__(self,d:int,h:int,dropout:float):
        super().__init__()
        self.net=nn.Sequential(CausalConv(d,h),nn.ReLU(),nn.Dropout(dropout),CausalConv(h,h,2),nn.ReLU())
        self.head=nn.Linear(2*h,1)
    def forward(self,q):
        x=self.net(q.transpose(1,2));return self.head(torch.cat([x.mean(-1),x[:,:,-1]],-1)).squeeze(-1)

class SkeletonEncoder(nn.Module):
    """Small static-graph control, explicitly not the original ST-GCN implementation."""
    def __init__(self,h:int):
        super().__init__(); adjacency=torch.eye(17)
        for i,j in [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:
            adjacency[i,j]=adjacency[j,i]=1
        self.register_buffer('adjacency',adjacency/adjacency.sum(-1,keepdim=True))
        self.linear=nn.Linear(7,h);self.temporal=CausalConv(h,h)
    def forward(self,q):
        x=q.reshape(q.shape[0],q.shape[1],17,7)
        x=torch.einsum('vw,btwc->btvc',self.adjacency,x)
        x=torch.relu(self.linear(x)).mean(2).transpose(1,2)
        return torch.relu(self.temporal(x)).mean(-1)

class Predictor(nn.Module):
    def __init__(self,a_dim:int,q_dim:int,time_steps:int,kind:str='full',hidden:int=64,dropout:float=.1):
        super().__init__();self.kind=kind
        self.spec=dict(a_dim=a_dim,q_dim=q_dim,time_steps=time_steps,kind=kind,hidden=hidden,dropout=dropout)
        self.geometry=mlp(a_dim,hidden,dropout)
        if kind in ['full','dual_no_pair','behavior','random_pair','hard_negative']:
            self.behavior=BehaviorTCN(q_dim,hidden,dropout)
        elif kind in ['fusion_mlp','fusion_pair','resampled','weighted']:
            self.fusion=mlp(a_dim+q_dim*time_steps,hidden,dropout)
        elif kind=='lstm':
            self.encoder=nn.LSTM(q_dim,hidden,batch_first=True);self.head=mlp(a_dim+hidden,hidden,dropout)
        elif kind=='graph':
            self.encoder=SkeletonEncoder(hidden);self.head=mlp(a_dim+hidden,hidden,dropout)
        elif kind!='geometry':raise ValueError(f'Unknown model {kind}')

    def forward(self,a,q):
        g=self.geometry(a).squeeze(-1)
        if self.kind in ['full','dual_no_pair','random_pair','hard_negative']:
            r=self.behavior(q);return g+r,g,r
        if self.kind=='geometry':return g,g,torch.zeros_like(g)
        if self.kind=='behavior':
            r=self.behavior(q);return r,torch.zeros_like(r),r
        if self.kind=='lstm':
            x,_=self.encoder(q);p=self.head(torch.cat([a,x[:,-1]],-1)).squeeze(-1)
        elif self.kind=='graph':p=self.head(torch.cat([a,self.encoder(q)],-1)).squeeze(-1)
        else:p=self.fusion(torch.cat([a,q.flatten(1)],-1)).squeeze(-1)
        return p,torch.zeros_like(p),p


def pair_loss(pos:torch.Tensor,neg:torch.Tensor,weights:torch.Tensor)->torch.Tensor:
    if pos.numel()==0:return pos.sum()*0
    if torch.any(weights<0):raise ValueError('Negative pair weight')
    return (F.softplus(-(pos-neg))*weights).sum()/weights.sum().clamp_min(1e-12)
