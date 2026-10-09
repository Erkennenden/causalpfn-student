"""V2: data/model/update sweeps and statistics across training seeds."""
import argparse
from dataclasses import fields,replace
import os
from pathlib import Path
import time

from scaling_v1.data import ROOT
from scaling_v1.runner import MODEL_SIZES,RunConfig,train,evaluate_checkpoint
from scaling_v1.__main__ import training_arguments
from .sweep import build_jobs,run_sweep,check_resume_runtime
from .report import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    smoke=commands.add_parser('smoke',help='20 updates on the original N1000 graph')
    training_arguments(smoke)
    smoke.set_defaults(model_size='small',train_tasks=8,validation_tasks=2,test_tasks=2,
        steps=20,batch_size=1,validation_every=10,checkpoint_every=10,log_every=5)
    training_arguments(commands.add_parser('train',help='One CEPO configuration'))
    sweep=commands.add_parser('sweep',help='One-factor-at-a-time sweep; 27 runs by default')
    for field in fields(RunConfig):
        if field.name in {'model_size','train_tasks','steps','train_seed'}:
            continue
        value=getattr(RunConfig(),field.name)
        sweep.add_argument('--'+field.name.replace('_','-'),type=type(value),default=value)
    sweep.add_argument('--task-counts',nargs='+',type=int,default=[256,1024,4096,16384])
    sweep.add_argument('--model-sizes',nargs='+',choices=tuple(MODEL_SIZES),default=list(MODEL_SIZES))
    sweep.add_argument('--step-budgets',nargs='+',type=int,default=[2000,8000,16000])
    sweep.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    sweep.add_argument('--anchor-tasks',type=int,default=4096)
    sweep.add_argument('--anchor-model',choices=tuple(MODEL_SIZES),default='medium')
    sweep.add_argument('--anchor-steps',type=int,default=8000)
    sweep.add_argument('--dry-run',action='store_true',help='Print the plan without generating data/training')
    sweep.add_argument('--resume',action='store_true')
    sweep.add_argument('--output',type=Path)
    report=commands.add_parser('report',help='Export per-run and across-seed summaries')
    report.add_argument('directory',type=Path)
    report.add_argument('--plot',action='store_true')
    evaluation=commands.add_parser('evaluate',help='Evaluate a compatible CEPO scaling checkpoint')
    evaluation.add_argument('--checkpoint',required=True,type=Path)
    evaluation.add_argument('--output',required=True,type=Path)
    evaluation.add_argument('--cache-dir',default=str(ROOT/'runs/scaling_cache'))
    evaluation.add_argument('--device',default='auto')
    evaluation.add_argument('--test-tasks',type=int,default=128)
    evaluation.add_argument('--threads',type=int,default=4)
    evaluation.add_argument('--query-chunk-size',type=int,default=16)
    args=parser.parse_args()
    if args.command=='report':
        summarize(args.directory,plot=args.plot)
        return
    if args.command=='evaluate':
        evaluate_checkpoint(args.checkpoint,args.output,cache_dir=args.cache_dir,device=args.device,
            test_tasks=args.test_tasks,threads=args.threads,query_chunk_size=args.query_chunk_size)
        return
    config=RunConfig(**{f.name:getattr(args,f.name) for f in fields(RunConfig) if hasattr(args,f.name)})
    output=args.output or ROOT/'runs'/f'scaling_v2_{args.command}_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    if args.command in {'smoke','train'}:
        if args.resume:
            check_resume_runtime(config,output)
        train(config,output,resume=args.resume)
        summarize(output)
        return
    config=replace(config,train_tasks=args.anchor_tasks,model_size=args.anchor_model,steps=args.anchor_steps)
    jobs=build_jobs(config,task_counts=args.task_counts,model_sizes=args.model_sizes,
        step_budgets=args.step_budgets,seeds=args.seeds)
    print(f'{len(jobs)} runs | total optimizer updates: {sum(j.config.steps for j in jobs):,}',flush=True)
    for job in jobs:
        seen=min(job.config.train_tasks,job.config.steps*job.config.batch_size)
        print(f'  {job.name}: planned distinct tasks seen {seen}/{job.config.train_tasks}',flush=True)
    if not args.dry_run:
        run_sweep(jobs,output,resume=args.resume)


if __name__=='__main__':
    main()
