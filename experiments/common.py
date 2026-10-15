"""Actual CE-MoE experiments. Every figure is computed from data or measured runs."""
import argparse,copy,hashlib,json,os,platform,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'cemoe_reproduce_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cemoe.workflow import (ROOT,CASES,read_checkpoint,load_data,evaluate,case_metrics,train_model,
                           save,restore,predict,subset,perturbed,seed_all)
from cemoe.data import metrics,sha256
from cemoe.evolution import tensor_pack,evolve

plt.rcParams.update({'font.family':'serif','font.size':11,'pdf.fonttype':42})


def savefig(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=200);plt.close(fig)


def conditions(pack,meta):
    distance=pack['y_raw']
    return [('Clean',pack),('Frequency',perturbed(pack,'frequency')),('Axial',perturbed(pack,'axial')),
            ('Near [1,5) m',subset(pack,(distance>=1)&(distance<5)))]


def cached_fit(cfg,seed,variant,cache):
    # Cache identity includes implementation and dataset, preventing stale model reuse.
    identity=dict(config=cfg,seed=seed,variant=variant,data=sha256(ROOT/'data/dataset.xlsx'),
                  source={p.name:sha256(p) for p in (ROOT/'cemoe').glob('*.py')})
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    path=cache/f'{variant}_{seed}_{key}.pt'
    if path.exists():return read_checkpoint(path)
    print(f'Training {variant}, seed {seed}, epochs {cfg["epochs"]}',flush=True)
    bundle=train_model(cfg,seed,variant);save(bundle,path);return bundle


def performance(b,te,out,args):
    pred=evaluate(b,te);pred.to_csv(out/'predictions.csv',index=False)
    table=case_metrics(pred);table.to_csv(out/'case_metrics.csv',index=False);print(table.to_string(index=False))
    fig,axes=plt.subplots(2,2,figsize=(9,6));rows=[]
    for ax,(name,lo,hi) in zip(axes.flat,CASES):
        p=pred[(pred.y_true>=lo)&(pred.y_true<hi)].sort_values('row_id')
        scores=[]
        for start in range(0,len(p),args.batch_size):
            part=p.iloc[start:start+args.batch_size];score=metrics(part.y_true.to_numpy(),part.y_pred.to_numpy(),'absolute')
            rows.append(dict(case=name,batch=len(scores)+1,**score));scores.append(score['rmse'])
        ax.plot(np.arange(1,len(scores)+1),scores,'o-',label='CE-MoE');ax.set(title=name,xlabel='Batch within distance subset',ylabel='RMSE (m)');ax.legend()
    pd.DataFrame(rows).to_csv(out/'batch_rmse.csv',index=False);savefig(fig,out,'performance')


def tails(b,te,out,args):
    p=evaluate(b,te);err=np.sort(np.abs(p.y_true-p.y_pred));ecdf=np.arange(1,len(err)+1)/len(err)
    pd.DataFrame(dict(absolute_error_m=err,ecdf=ecdf)).to_csv(out/'ecdf.csv',index=False)
    case_metrics(p).to_csv(out/'tail_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4));ax.plot(err,ecdf,label='CE-MoE');ax.axvline(np.quantile(err,.95),color='gray',ls='--',label='95th percentile')
    ax.set(xlabel='Absolute error (m)',ylabel='Empirical probability');ax.legend();savefig(fig,out,'error_distribution')


def ablation(b,te,out,args):
    cfg=experiment_config(b,args);packs,meta=load_data(cfg);te=packs[2];rows=[]
    variants=['Full','noE','noM','noF','noA','noS','Uniform','Capacity','noSCF','noDRM','noCWC']
    for seed in range(11,11+(args.repeats or 1)):
        for variant in variants:
            run=cached_fit(cfg,seed,variant,ROOT/'results/cache')
            for label,pack in conditions(te,meta):
                if not len(pack['y']):continue
                p=evaluate(run,pack);rows.append(dict(seed=seed,variant=variant,condition=label,parameters=run['parameters'],**metrics(p.y_true.to_numpy(),p.y_pred.to_numpy(),'absolute')))
    table=pd.DataFrame(rows);table.to_csv(out/'ablation.csv',index=False)
    means=table.groupby(['variant','condition']).rmse.mean().unstack().reindex(variants)
    std=table.groupby(['variant','condition']).rmse.std().unstack().reindex(variants).fillna(0)
    deltas=means.subtract(means.loc['Full']);deltas.to_csv(out/'rmse_increase.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    deltas[['Clean','Frequency','Axial']].plot.bar(ax=axes[0]);axes[0].set(ylabel='RMSE increase versus Full (m)',xlabel='Variant')
    axes[1].bar(variants,means['Near [1,5) m'],yerr=std['Near [1,5) m'] if (args.repeats or 1)>1 else None,capsize=2);axes[1].set(ylabel='Near-range RMSE (m)');axes[1].tick_params(axis='x',rotation=60)
    savefig(fig,out,'expert_ablation')


def routing(b,te,out,args):
    rows=[];weights=[]
    for name,pack in conditions(te,b['preprocessing']):
        if not len(pack['y']):continue
        p=evaluate(b,pack)
        for expert in 'FAS':
            rows.append(dict(condition=name,expert=expert,rmse=float(np.sqrt(np.mean((p['prediction_'+expert]-p.y_true)**2))),mean_weight=float(p['weight_'+expert].mean())))
        p['condition']=name;weights.append(p)
    table=pd.DataFrame(rows);table.to_csv(out/'expert_summary.csv',index=False);pd.concat(weights).to_csv(out/'expert_predictions.csv',index=False)
    names=[x[0] for x in conditions(te,b['preprocessing'])]
    w=table.pivot(index='condition',columns='expert',values='mean_weight').reindex(names)[list('FAS')]
    e=table.pivot(index='condition',columns='expert',values='rmse').reindex(names)[list('FAS')]
    fig,axes=plt.subplots(1,2,figsize=(11,4));w.plot.bar(ax=axes[0]);axes[0].set(ylabel='Mean fusion weight',xlabel='Condition',ylim=(0,1));axes[0].legend(title='Expert',ncol=3,loc='upper center')
    axes[1].imshow(e.to_numpy(),aspect='auto',cmap='Blues');axes[1].set_xticks(range(3),list('FAS'));axes[1].set_yticks(range(len(e)),e.index)
    for i in range(len(e)):
        for j in range(3):axes[1].text(j,i,f'{e.iloc[i,j]:.3f}',ha='center',va='center',color='white' if e.iloc[i,j]>(e.to_numpy().min()+e.to_numpy().max())/2 else 'black')
    axes[1].set_title('Individual-expert RMSE (m)');savefig(fig,out,'expert_routing')


def evolution(b,te,out,args):
    h=pd.DataFrame(b['history']);e=pd.DataFrame(b['evolution']);h.to_csv(out/'training_history.csv',index=False);e.to_csv(out/'evolution_history.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(h.epoch,h.validation_rmse);axes[0].set(xlabel='Offline epoch',ylabel='Validation RMSE (m)')
    if len(e):axes[1].plot(e.generation,e.best_fitness,'o-')
    axes[1].set(xlabel='Evolution generation',ylabel='Best training-buffer fitness')
    savefig(fig,out,'evolution_process')


def experiment_config(b,args):
    cfg=b['config'].copy()
    if args.full:cfg.update(train_windows_per_well=0,val_windows_per_well=0,test_windows_per_well=0,population=20,offspring=20,generations=10,epochs=80)
    if args.quick:cfg.update(epochs=3,train_windows_per_well=200,val_windows_per_well=120,test_windows_per_well=160,population=4,offspring=4,generations=1,fitness_rows=64)
    if args.epochs is not None:cfg['epochs']=args.epochs
    return cfg


def stability(b,te,out,args):
    cfg=experiment_config(b,args);packs,_=load_data(cfg);rows=[]
    for seed in range(11,11+(args.repeats or (2 if args.quick else 10))):
        run=cached_fit(cfg,seed,'Full',ROOT/'results/cache');table=case_metrics(evaluate(run,packs[2]));table['seed']=seed;rows.append(table)
    table=pd.concat(rows);table.to_csv(out/'repeated_metrics.csv',index=False)
    groups=[table.loc[table.case==name,'rdb_percent'].dropna().to_numpy() for name,_,_ in CASES]
    fig,ax=plt.subplots(figsize=(7,4));ax.boxplot(groups,tick_labels=[x[0] for x in CASES],showmeans=True);ax.set(ylabel='RDB (%) across training seeds');savefig(fig,out,'stability')
    summary=table.groupby('case').agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),rdb_mean=('rdb_percent','mean'),rdb_sd=('rdb_percent','std'),repeats=('seed','count'))
    summary.to_csv(out/'summary.csv')


def robustness(b,te,out,args):
    rows=[]
    for sd in [0,.05,.1,.2,.4]:
        for repeat in range(args.repeats or (2 if args.quick else 10)):
            p=evaluate(b,perturbed(te,f'noise_{sd}',seed=73+repeat));rows.append(dict(noise_std=sd,repeat=repeat,**metrics(p.y_true.to_numpy(),p.y_pred.to_numpy(),'absolute')))
    table=pd.DataFrame(rows);table.to_csv(out/'robustness.csv',index=False);g=table.groupby('noise_std').rmse
    fig,ax=plt.subplots(figsize=(6,4));ax.errorbar(g.mean().index,g.mean(),yerr=g.std().fillna(0),fmt='o-',capsize=3)
    ax.set(xlabel='Magnetic noise SD (training-standardized units)',ylabel='RMSE (m)');savefig(fig,out,'noise_robustness')


def latency(b,te,out,args):
    m=restore(b).eval();rows=[]
    for size in [16,32,64,96,128]:
        if size>len(te['y']):continue
        x=torch.tensor(te['x'][:size]);inc=torch.tensor(te['inc'][:size]);times=[]
        with torch.no_grad():
            for _ in range(5):m(x,inc)
            for _ in range(20 if args.quick else 100):
                t=time.perf_counter();m(x,inc);times.append(1000*(time.perf_counter()-t))
        rows.append(dict(batch_size=size,median_ms=float(np.median(times)),mean_ms=float(np.mean(times)),sd_ms=float(np.std(times,ddof=1)),repeats=len(times)))
    table=pd.DataFrame(rows);table.to_csv(out/'latency.csv',index=False)
    (out/'hardware.json').write_text(json.dumps(dict(platform=platform.platform(),machine=platform.machine(),torch=torch.__version__,device='CPU',threads=torch.get_num_threads(),scope='forward pass, preloaded tensors, no training or data I/O'),indent=2))
    fig,ax=plt.subplots(figsize=(6,4));ax.errorbar(table.batch_size,table.median_ms,yerr=table.sd_ms,fmt='o-',capsize=3);ax.set(xlabel='Batch size',ylabel='Forward latency (ms)');savefig(fig,out,'latency')


def adaptation(b,te,out,args):
    seed_all(b['seed']);m=restore(b);frozen=restore(b,offline=True)
    cfg=b['config'].copy();cfg.update(population=4 if args.quick else 20,offspring=4 if args.quick else 20,generations=1)
    cfg.update(_x_mean=b['preprocessing']['x_mean'],_x_scale=b['preprocessing']['x_scale'])
    train,_=load_data(b['config']);tr=tensor_pack(train[0]);ym,ys=b['preprocessing']['y_mean'],b['preprocessing']['y_scale']
    size=args.batch_size;shift=args.change_batch;rows=[];predictions=[];released=[]
    observed={k:v.copy() for k,v in te.items()}
    # A declared synthetic persistent Z bias is the reproducible distribution change.
    start_shift=(shift-1)*size;observed['x'][start_shift:,:,[2,5,8]]+=.5
    num_batches=int(np.ceil(len(te['y'])/size))
    if shift>num_batches:raise ValueError('Not enough batches for requested change; reduce --batch-size.')
    for batch,start in enumerate(range(0,len(te['y']),size)):
        update=0.;deadline_exceeded=False
        release=batch-args.delay
        if release>=0:
            ids=np.arange(release*size,min((release+1)*size,len(te['y'])));released.extend(ids.tolist())
            buffer=tensor_pack(subset(observed,np.array(released[-128:],dtype=int)))
            before=time.perf_counter()
            # Measured best-effort budget: one generation can overrun 120 ms; recorded below.
            candidate,_=evolve(m,tr,buffer,cfg,b['seed']+batch)
            update=1000*(time.perf_counter()-before)
            if update<=args.budget_ms:m=candidate
            else:deadline_exceeded=True
        block=subset(observed,slice(start,start+size));truth=block['y_raw']
        for label,model in [('CE-MoE',m),('noE (frozen)',frozen)]:
            p,_,_=predict(model,block);p=p*ys+ym
            rows.append(dict(model=label,batch=batch+1,released_labels=len(released),update_ms=update if label=='CE-MoE' else 0.,update_rejected=deadline_exceeded if label=='CE-MoE' else False,**metrics(truth,p,'absolute')))
            for rid,y,yp in zip(block['rows'],truth,p):predictions.append(dict(model=label,batch=batch+1,row_id=int(rid),y_true=float(y),y_pred=float(yp)))
    table=pd.DataFrame(rows);table.to_csv(out/'adaptation.csv',index=False);pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    recovery=[];fig,ax=plt.subplots(figsize=(8,4))
    for label,g in table.groupby('model',sort=False):
        threshold=1.2*g[g.batch<shift].rmse.mean();post=g[g.batch>=shift];values=post.rmse.to_numpy();found=None
        for i in range(len(values)-2):
            if np.all(values[i:i+3]<=threshold):found=int(post.iloc[i].batch);break
        recovery.append(dict(model=label,threshold=float(threshold),first_recovery_batch=found,recovery_batches=None if found is None else found-shift+1))
        ax.plot(g.batch,g.rmse,label=label)
    pd.DataFrame(recovery).to_csv(out/'recovery.csv',index=False);ax.axvline(shift,color='gray',ls='--');ax.set(xlabel='Batch',ylabel='RMSE (m)');ax.legend();savefig(fig,out,'delayed_adaptation')
    (out/'protocol.json').write_text(json.dumps(dict(delay=args.delay,change_batch=shift,change='Persistent +0.5 training-SD Z-axis bias',batch_size=size,budget_ms=args.budget_ms,
       budget_policy='Over-budget candidate rejected; computation can still exceed deadline. Not a hard real-time guarantee.',
       recovery='First of three consecutive batches <=1.2 times own pre-change mean; batches counted inclusively.',
       note='Single-seed replay unless rerun manually. No invented standard-deviation shading; natural distance progression remains a confounder.'),indent=2))



def cross_validation(b,te,out,args):
    cfg=experiment_config(b,args);rows=[];predictions=[]
    for test_id in range(1,5):
        val_id=test_id%4+1
        fold={**cfg,'test_group':f'inferred_{test_id}','val_group':f'inferred_{val_id}'}
        packs,_=load_data(fold)
        for seed in range(11,11+(args.repeats or 1)):
            run=cached_fit(fold,seed,'Full',ROOT/'results/cache')
            pred=evaluate(run,packs[2]);table=case_metrics(pred)
            table['test_group']=test_id;table['seed']=seed;rows.append(table)
            pred['test_group']=test_id;pred['seed']=seed;predictions.append(pred)
    table=pd.concat(rows);table.to_csv(out/'fold_metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    summary=table.groupby('case').agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),rdb_mean=('rdb_percent','mean'),qr_mean=('qr_percent','mean'))
    summary.to_csv(out/'summary.csv');print(summary)
    overall=table[table.case=='Overall'];fig,ax=plt.subplots(figsize=(6,4))
    g=overall.groupby('test_group').rmse;ax.bar(g.mean().index,g.mean());ax.set(xlabel='Held-out inferred trajectory',ylabel='RMSE (m)');savefig(fig,out,'held_out_trajectories')


def main(kind):
    ap=argparse.ArgumentParser(description=f'CE-MoE {kind} experiment, computed from actual data.')
    ap.add_argument('--checkpoint',default=str(ROOT/'checkpoint/best.pt'));ap.add_argument('--out');ap.add_argument('--full',action='store_true');ap.add_argument('--quick',action='store_true')
    ap.add_argument('--epochs',type=int);ap.add_argument('--repeats',type=int);ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--delay',type=int,default=1);ap.add_argument('--change-batch',type=int,default=25);ap.add_argument('--budget-ms',type=float,default=120.)
    args=ap.parse_args()
    if args.batch_size<1 or args.delay<1 or args.change_batch<2:raise ValueError('Invalid replay/batch arguments.')
    seed_all(11);b=read_checkpoint(args.checkpoint);cfg=b['config'].copy()
    if args.full:cfg['test_windows_per_well']=0
    packs,meta=load_data(cfg)
    if meta['data_sha256']!=b['preprocessing']['data_sha256']:raise ValueError('Checkpoint/data mismatch.')
    out=Path(args.out or ROOT/'results'/kind);out.mkdir(parents=True,exist_ok=True)
    (out/'settings.json').write_text(json.dumps(dict(checkpoint=Path(args.checkpoint).name,checkpoint_sha256=sha256(args.checkpoint),
          selected_seed=b['seed'],dataset_sha256=meta['data_sha256'],training_config=experiment_config(b,args) if kind in ['ablation','stability','cross_validation'] else b['config'],
          arguments=vars(args)),indent=2))
    globals()[kind](b,packs[2],out,args);print('Saved:',out,flush=True)
