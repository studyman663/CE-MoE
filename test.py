"""Evaluate an existing checkpoint without training."""
import argparse
from pathlib import Path
from cemoe.workflow import ROOT,read_checkpoint,load_data,evaluate,case_metrics

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',default=str(ROOT/'checkpoint/best.pt'));ap.add_argument('--all-test',action='store_true');ap.add_argument('--data');ap.add_argument('--out',default=str(ROOT/'results/test'))
    args=ap.parse_args();b=read_checkpoint(args.checkpoint);cfg=b['config'].copy()
    if args.all_test:cfg['test_windows_per_well']=0
    packs,meta=load_data(cfg,args.data)
    if meta['data_sha256']!=b['preprocessing']['data_sha256']:raise ValueError('Dataset differs from checkpoint; use training/inference with an explicitly defined new-data protocol.')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);p=evaluate(b,packs[2]);p.to_csv(out/'predictions.csv',index=False)
    table=case_metrics(p);table.to_csv(out/'metrics.csv',index=False);print(table.to_string(index=False))
if __name__=='__main__':main()
