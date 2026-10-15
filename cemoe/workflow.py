"""Portable training, validation-only checkpoint selection and actual evaluation."""
import copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import read_frame,split_and_scale,metrics,TARGET,sha256
from .model import make_model,CEMoE,count_parameters
from .evolution import tensor_pack,evolve

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=dict(window=8,hidden=24,target_mode='absolute',protocol='group',group_column='well_id',
             test_group='inferred_4',val_group='inferred_3',sample_order_confirmed=True,
             epochs=30,lr=.001,batch_size=128,population=6,offspring=6,generations=3,
             fitness_rows=256,initial_sigma=.005,mutation_sigma=.02,theta0=60.,beta=.1,
             dmid=5.,alpha=1.,fusion_bias=.25,lambda_comp=.01,lambda_conf=.02,lambda_local=.25,
             eta=.1,tau_z=.4,kappa=1.,train_windows_per_well=800,val_windows_per_well=400,test_windows_per_well=600)
CASES=[('Case 1',30.,70.000001),('Case 2',10.,30.),('Case 3',5.,10.),('Case 4',1.,5.)]


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


def configuration(full=False,quick=False):
    c=DEFAULT.copy()
    if full:c.update(epochs=80,population=20,offspring=20,generations=10,
                     train_windows_per_well=0,val_windows_per_well=0,test_windows_per_well=0)
    if quick:c.update(epochs=3,population=4,offspring=4,generations=1,fitness_rows=64,
                      train_windows_per_well=200,val_windows_per_well=120,test_windows_per_well=160)
    return c


def load_data(cfg,path=None):
    path=Path(path or ROOT/'data/dataset.xlsx')
    frame=read_frame(path,'absolute');y=frame[TARGET].to_numpy()
    reset=np.r_[True,(y[:-1]<=1.)&(y[1:]>=30.)]
    frame['well_id']=['inferred_'+str(i) for i in reset.cumsum()]
    frame['time']=frame.groupby('well_id').cumcount()
    if frame.well_id.nunique()<3:raise ValueError('At least three trajectory groups required.')
    packs,meta=split_and_scale(frame,cfg,11)
    # Subsample ENDPOINTS after complete windows have been built, not raw window rows.
    # This retains true local adjacency inside each sampled window.
    for split,pack in zip(['train','val','test'],packs):
        cap=cfg.get(split+'_windows_per_well',0);selected=[]
        group=frame.iloc[pack['rows']].well_id.to_numpy()
        for well in pd.unique(group):
            ids=np.flatnonzero(group==well)
            selected.extend(ids[np.unique(np.linspace(0,len(ids)-1,min(len(ids),cap),dtype=int))] if cap else ids)
        selected=np.array(sorted(selected),dtype=int)
        for key in pack:pack[key]=pack[key][selected]
        pack['y_raw']=frame.iloc[pack['rows']][TARGET].to_numpy(float)
        meta['window_rows'][split]=[meta['window_rows'][split][i] for i in selected]
    meta['data_sha256']=sha256(path)
    meta['inferred_groups']=[dict(group=g,start_row=int(b.index.min()),end_row=int(b.index.max()),rows=len(b)) for g,b in frame.groupby('well_id',sort=False)]
    meta['sampling']='Absolute labels; distance-reset groups; index-ordered windows, physical time unverified.'
    return packs,meta


@torch.no_grad()
def predict(model,pack,batch_size=512):
    model.eval();pred=[];experts=[];weights=[]
    for start in range(0,len(pack['x']),batch_size):
        x=torch.as_tensor(pack['x'][start:start+batch_size],dtype=torch.float32)
        inc=torch.as_tensor(pack['inc'][start:start+batch_size],dtype=torch.float32)
        p,info=model(x,inc);pred.append(p.cpu().numpy())
        if 'experts' in info:experts.append(info['experts'].cpu().numpy());weights.append(info['weights'].cpu().numpy())
    return np.concatenate(pred),np.concatenate(experts) if experts else None,np.concatenate(weights) if weights else None


def train_model(cfg,seed=11,variant='Full',data_path=None):
    seed_all(seed);packs,meta=load_data(cfg,data_path);tr,va,te=packs;ym,ys=meta['y_mean'],meta['y_scale']
    model=make_model(variant,cfg,ym,ys);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    tx,ti,ty=tensor_pack(tr);history=[];best=float('inf');best_state=None;start=time.perf_counter()
    for epoch in range(cfg['epochs']):
        model.train()
        for ids in torch.randperm(len(ty)).split(cfg['batch_size']):
            optimizer.zero_grad();p,_=model(tx[ids],ti[ids]);loss=(p-ty[ids]).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        p,_,_=predict(model,va);score=float(np.sqrt(np.mean((p-va['y'])**2))*ys)
        history.append(dict(epoch=epoch+1,validation_rmse=score))
        if score<best:best=score;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);offline_state=copy.deepcopy(best_state);evolution=[]
    if isinstance(model,CEMoE) and variant!='noE':
        rng=np.random.default_rng(seed);ix=rng.choice(len(ty),min(len(ty),cfg['fitness_rows']),replace=False)
        buffer=tuple(v[ix] for v in (tx,ti,ty));ecfg={**cfg,'_x_mean':meta['x_mean'],'_x_scale':meta['x_scale']}
        candidate,evolution=evolve(model,(tx,ti,ty),buffer,ecfg,seed)
        p,_,_=predict(candidate,va);candidate_score=float(np.sqrt(np.mean((p-va['y'])**2))*ys)
        if candidate_score<best:model=candidate;best=candidate_score;selected='evolved'
        else:selected='offline'
    else:selected='offline'
    return dict(state=model.state_dict(),offline_state=offline_state,config=cfg,seed=seed,variant=variant,
                preprocessing=meta,validation_rmse=best,selected_stage=selected,history=history,
                evolution=evolution,fit_seconds=time.perf_counter()-start,parameters=count_parameters(model),
                source_sha256={p.name:sha256(p) for p in (ROOT/'cemoe').glob('*.py')})


def restore(bundle,offline=False):
    m=make_model(bundle['variant'],bundle['config'],bundle['preprocessing']['y_mean'],bundle['preprocessing']['y_scale'])
    m.load_state_dict(bundle['offline_state'] if offline else bundle['state']);return m


def save(bundle,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save(bundle,path)
    public={k:v for k,v in bundle.items() if k not in ['state','offline_state']}
    path.with_suffix('.json').write_text(json.dumps(public,indent=2))


def read_checkpoint(path=None):
    return torch.load(Path(path or ROOT/'checkpoint/best.pt'),map_location='cpu',weights_only=True)


def evaluate(bundle,pack):
    model=restore(bundle);p,e,w=predict(model,pack);meta=bundle['preprocessing'];ym,ys=meta['y_mean'],meta['y_scale']
    pred=pd.DataFrame(dict(row_id=pack['rows'],y_true=pack['y_raw'],y_pred=p*ys+ym))
    if e is not None:
        for i,k in enumerate('FAS'):pred['prediction_'+k]=e[:,i]*ys+ym;pred['weight_'+k]=w[:,i]
    return pred


def case_metrics(pred):
    rows=[dict(case='Overall',**metrics(pred.y_true.to_numpy(),pred.y_pred.to_numpy(),'absolute'))]
    for name,lo,hi in CASES:
        s=pred[(pred.y_true>=lo)&(pred.y_true<hi)]
        rows.append(dict(case=name,**metrics(s.y_true.to_numpy(),s.y_pred.to_numpy(),'absolute')) if len(s) else dict(case=name,n=0))
    return pd.DataFrame(rows)


def subset(pack,ids):return {k:v[ids] for k,v in pack.items()}


def perturbed(pack,condition,seed=73):
    pack={k:v.copy() for k,v in pack.items()};x=pack['x'];rng=np.random.default_rng(seed)
    if condition=='frequency':
        # Synthetic frequency perturbation, cycles per sample, not physical Hz.
        phase=np.arange(x.shape[1])/4
        x[:,:,:9]+=.5*np.sin(2*np.pi*phase)[None,:,None]
    elif condition=='axial':x[:,:,[2,5,8]]+=.5
    elif condition.startswith('noise_'):x[:,:,:9]+=rng.normal(0,float(condition[6:]),x[:,:,:9].shape)
    elif condition!='clean':raise ValueError(condition)
    return pack
