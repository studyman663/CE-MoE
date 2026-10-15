"""Capped diversity fitness, density-guided mutation, and elitist search."""
import copy
import numpy as np
import torch
from torch.nn import functional as F


def tensor_pack(pack):
    return tuple(torch.as_tensor(pack[k],dtype=torch.float32) for k in ['x','inc','y'])


def gradient_fit(model,pack,epochs,lr,batch_size=128):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    x,inc,y=pack
    for _ in range(epochs):
        model.train()
        for ids in torch.randperm(len(y)).split(batch_size):
            optimizer.zero_grad(); p,_=model(x[ids],inc[ids]); loss=(p-y[ids]).square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
    return model


@torch.no_grad()
def fitness(model,buffer,cfg):
    model.eval(); x,inc,y=buffer; p,info=model(x,inc)
    v=info['experts'].var(1,unbiased=False)
    risk=inc>cfg.get('theta0',60.)
    distance=y*model.ys+model.ym
    if cfg['target_mode']=='signed':distance=distance.abs()
    near=distance<cfg.get('dmid',5.)
    comp=v.clamp(max=1.).mean()
    conflict=(v*risk).sum()/risk.sum().clamp(min=1)
    local=((p-y).abs()*near).sum()/near.sum().clamp(min=1)
    rmse=(p-y).square().mean().sqrt()
    return float(cfg.get('lambda_comp',.01)*comp-cfg.get('lambda_conf',.02)*conflict-cfg.get('lambda_local',.25)*local-rmse)


def operator_statistics(train,buffer,cfg,seed):
    rng=np.random.default_rng(seed)
    ref=train[0][:,:,:9].flatten(1)
    ref=ref[rng.choice(len(ref),min(len(ref),cfg.get('density_reference_size',128)),replace=False)]
    u=buffer[0][:,:,:9].flatten(1)
    h=cfg.get('density_bandwidth',float(u.shape[1]**.5))
    if h<=0:raise ValueError('Density bandwidth must be positive.')
    rho=torch.exp(-torch.cdist(u,ref).square()/(2*h*h)).mean(1)
    # Average per-window trigger probability for the globally shared filter parameters.
    scf=float((cfg.get('eta',.1)/(rho+1e-8)).clamp(max=1).mean())
    raw=buffer[0][:,:,:9]*torch.tensor(cfg['_x_scale'][:9])+torch.tensor(cfg['_x_mean'][:9])
    energy=raw.reshape(len(raw),raw.shape[1],3,3).square().sum((1,2))
    qz=float((energy[:,2]/(energy.sum(1)+1e-8)).mean())
    theta=float(buffer[1].mean())
    return dict(scf_probability=scf,mean_z_energy_fraction=qz,mean_inclination_degrees=theta)


@torch.no_grad()
def mutate(child,cfg,stats):
    sigma=cfg.get('mutation_sigma',.02); variant=child.variant
    if variant!='noSCF' and torch.rand(()).item()<stats['scf_probability']:
        targets=[child.frequency.predict_raw,child.frequency.update_raw] if child.temporal else list(child.frequency.parameters())
        for p in targets:p.add_(sigma*torch.randn_like(p))
    if variant!='noDRM' and stats['mean_z_energy_fraction']>cfg.get('tau_z',.4):
        amplitude=sigma*(1-np.cos(np.deg2rad(stats['mean_inclination_degrees'])))
        for emb in child.axial.emb[:2]:
            for p in emb.parameters():p.add_(amplitude*torch.randn_like(p))
    if variant!='noCWC':
        similarity=F.cosine_similarity(child.prototypes[2],child.prototypes[3],dim=0,eps=1e-8)
        probability=torch.sigmoid(cfg.get('kappa',1.)*similarity)
        for a,b in zip(child.stage[2].parameters(),child.stage[3].parameters()):
            b.copy_(torch.where(torch.rand_like(b)<probability,a,b))


def evolve(incumbent,train,buffer,cfg,seed):
    size=cfg.get('population',20); generations=cfg.get('generations',3)
    if size<2 or generations<0:raise ValueError('Population >=2 and generations >=0 required.')
    rng=np.random.default_rng(seed); population=[copy.deepcopy(incumbent)]
    for _ in range(size-1):
        model=copy.deepcopy(incumbent)
        with torch.no_grad():
            for p in model.parameters():p.add_(cfg.get('initial_sigma',.005)*torch.randn_like(p))
        population.append(model)
    stats=operator_statistics(train,buffer,cfg,seed)
    history=[]
    for generation in range(generations+1):
        if generation:
            children=[]
            for _ in range(cfg.get('offspring',size)):
                child=copy.deepcopy(population[int(rng.integers(len(population)))])
                mutate(child,cfg,stats);children.append(child)
            population+=children
        scores=[fitness(m,buffer,cfg) for m in population]
        order=np.argsort(scores)[::-1][:size]
        history.append(dict(generation=generation,best_fitness=float(scores[order[0]]),**stats))
        population=[population[i] for i in order]
    return population[0],history
