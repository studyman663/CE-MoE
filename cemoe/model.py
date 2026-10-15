"""Explicit reference realization of the revised CE-MoE methodology."""
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU()); self.head=nn.Linear(h,1)
    def forward(self,x):
        h=self.net(x); return self.head(h),h


class Frequency(nn.Module):
    """One-level lifting analysis/synthesis with learned sigmoid detail gating."""
    def __init__(self,h):
        super().__init__()
        self.predict_raw=nn.Parameter(torch.zeros(1,9,1))
        self.update_raw=nn.Parameter(torch.zeros(1,9,1))
        self.detail_gate=nn.Conv1d(9,9,1,groups=9)
        self.c1=nn.Conv1d(9,h,3); self.c2=nn.Conv1d(h,h,3,dilation=2); self.head=nn.Linear(h,1)
    def analyze(self,x):
        p=2*torch.sigmoid(self.predict_raw); u=torch.sigmoid(self.update_raw)
        even,odd=x[...,::2],x[...,1::2]
        d=odd-p*even; a=even+u*d
        return a,d,p,u
    @staticmethod
    def synthesize(a,d,p,u):
        even=a-u*d; odd=d+p*even
        return torch.stack([even,odd],-1).flatten(-2)
    def forward(self,x):
        if x.shape[1]<2 or x.shape[1]%2: raise ValueError('Frequency branch requires an even temporal window.')
        a,d,p,u=self.analyze(x.transpose(1,2))
        clean=self.synthesize(a,torch.sigmoid(self.detail_gate(d))*d,p,u)
        h=F.relu(self.c1(F.pad(clean,(2,0))))
        h=F.relu(self.c2(F.pad(h,(4,0))))[...,-1]
        return self.head(h),h


class Axial(nn.Module):
    def __init__(self,h,threshold=60.,beta=.1):
        super().__init__(); self.h=h; self.threshold=threshold; self.beta=beta
        if h%2: raise ValueError('Hidden dimension must be even for two attention heads.')
        self.emb=nn.ModuleList([nn.Linear(3,h) for _ in range(3)])
        self.q,self.k,self.v=[nn.Linear(h,h) for _ in range(3)]
        self.head=nn.Linear(h,1)
    def forward(self,x,inc):
        axes=x.reshape(-1,3,3).transpose(1,2)
        tokens=torch.stack([e(axes[:,i]) for i,e in enumerate(self.emb)],1)
        def heads(t): return t.reshape(len(t),3,2,self.h//2).transpose(1,2)
        q,k,v=[heads(f(tokens)) for f in (self.q,self.k,self.v)]
        score=q@k.transpose(-1,-2)/(self.h//2)**.5
        bias=torch.zeros_like(score); bias[...,2]=-self.beta*F.relu(inc-self.threshold)[:,None,None]
        attention=(score+bias).softmax(-1)
        hidden=(attention@v).transpose(1,2).reshape(-1,3,self.h).mean(1)
        return self.head(hidden),hidden


class CEMoE(nn.Module):
    def __init__(self,cfg,ym=0.,ys=1.,variant='Full'):
        super().__init__(); self.cfg=cfg; self.variant=variant; self.temporal=cfg['window']>1
        h=cfg.get('hidden',24)
        self.frequency=Frequency(h) if self.temporal else Encoder(9,h)
        self.axial=Axial(h,cfg.get('theta0',60.),cfg.get('beta',.1))
        self.stage=nn.ModuleList([Encoder(12,h) for _ in range(4)])
        self.prototypes=nn.Parameter(torch.randn(4,12)*.1)
        self.router=nn.Sequential(nn.Linear(3*h,h),nn.ReLU(),nn.Linear(h,3))
        self.register_buffer('ym',torch.tensor(float(ym))); self.register_buffer('ys',torch.tensor(float(ys)))
    def forward(self,x,inc):
        last=x[:,-1]
        pf,hf=self.frequency(x[:,:,:9] if self.temporal else last[:,:9])
        pa,ha=self.axial(last[:,:9],inc)
        sg=(last@self.prototypes.T).softmax(-1)
        outputs=[e(last) for e in self.stage]
        ps=(torch.cat([a for a,b in outputs],1)*sg).sum(1,keepdim=True)
        hs=(torch.stack([b for a,b in outputs],1)*sg[:,:,None]).sum(1)
        experts=torch.cat([pf,pa,ps],1)
        hidden=[hf,ha,hs]
        if self.variant in ['noF','noA','noS']:
            index=['noF','noA','noS'].index(self.variant); hidden[index]=torch.zeros_like(hidden[index])
        q=self.router(torch.cat(hidden,1)).softmax(-1)
        distance=ps*self.ys+self.ym
        if self.cfg['target_mode']=='signed': distance=distance.abs()
        c=torch.sigmoid(-self.cfg.get('alpha',1.)*(distance-self.cfg.get('dmid',5.)))
        if self.variant=='noS': c=torch.zeros_like(c)
        prior=torch.cat([(1-c)/2,(1-c)/2,c],1)
        bias=self.cfg.get('fusion_bias',.25)
        q=(1-bias)*q+bias*prior
        if self.variant=='Uniform': q=torch.ones_like(q)/3
        if self.variant in ['noF','noA','noS']:
            keep=torch.ones_like(q); keep[:,['noF','noA','noS'].index(self.variant)]=0
            q=q*keep; q=q/q.sum(1,keepdim=True).clamp_min(1e-8)
        prediction=(experts*q).sum(1)
        if self.cfg['target_mode']!='signed': prediction=prediction.clamp_min(-self.ym/self.ys)
        return prediction,dict(experts=experts,weights=q,stage_weights=sg)


class MLP(nn.Module):
    def __init__(self,h=24,ym=0.,ys=1.):
        super().__init__(); self.encoder=Encoder(12,h)
        self.register_buffer('ym',torch.tensor(float(ym)));self.register_buffer('ys',torch.tensor(float(ys)))
    def forward(self,x,inc): return self.encoder(x[:,-1])[0][:,0].clamp_min(-self.ym/self.ys),{}


def count_parameters(model): return sum(p.numel() for p in model.parameters())


def make_model(name,cfg,ym,ys):
    if name == 'noM': return MLP(cfg.get('hidden',24),ym,ys)
    if name=='Capacity':
        target=count_parameters(CEMoE(cfg,ym,ys))
        # Nearest width budget, recorded rather than described as exactly matched.
        width=min(range(2,512),key=lambda h:abs((h*h+15*h+1)-target))
        return MLP(width,ym,ys)
    if name not in ['Full','noE','noF','noA','noS','Uniform','noSCF','noDRM','noCWC']:
        raise ValueError(f'Unknown model {name}')
    return CEMoE(cfg,ym,ys,name)
