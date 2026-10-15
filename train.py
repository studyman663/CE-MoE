"""Train CE-MoE and select a checkpoint by validation RMSE, never test RMSE."""
import argparse
from pathlib import Path
import pandas as pd
from cemoe.workflow import ROOT,configuration,train_model,save,load_data,evaluate,case_metrics

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--full',action='store_true');ap.add_argument('--quick',action='store_true')
    ap.add_argument('--epochs',type=int);ap.add_argument('--seeds',type=int,nargs='+',default=[11,23]);ap.add_argument('--data');ap.add_argument('--out',default=str(ROOT/'checkpoint'))
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Use --out with a new directory to preserve the supplied checkpoint.')
    cfg=configuration(args.full,args.quick)
    if args.epochs is not None:cfg['epochs']=args.epochs
    best=None;rows=[]
    for seed in args.seeds:
        b=train_model(cfg,seed,data_path=args.data);save(b,out/f'seed_{seed}.pt')
        rows.append(dict(seed=seed,validation_rmse=b['validation_rmse'],selected_stage=b['selected_stage'],fit_seconds=b['fit_seconds']))
        print(rows[-1],flush=True)
        if best is None or b['validation_rmse']<best['validation_rmse']:best=b
    save(best,out/'best.pt');pd.DataFrame(rows).to_csv(out/'selection.csv',index=False)
    packs,_=load_data(cfg,args.data);pred=evaluate(best,packs[2]);pred.to_csv(out/'test_predictions.csv',index=False)
    result=case_metrics(pred);result.to_csv(out/'test_metrics.csv',index=False);print(result.to_string(index=False))
if __name__=='__main__':main()
