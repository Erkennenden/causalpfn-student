"""python -m pfn_interference {diagnose,train,evaluate}."""
import argparse
from dataclasses import fields
from pathlib import Path
from .training import RunConfig,train
from .evaluation import evaluate,diagnose_prior
from .data import ROOT
from pfn_interference.baselines.localized_config import add_localized_arguments, LOCALIZED_DEFAULTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('train')
    defaults=RunConfig()
    for f in fields(defaults):
        value=getattr(defaults,f.name)
        p.add_argument('--'+f.name.replace('_','-'),type=type(value),default=value)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--resume',action='store_true')
    p=commands.add_parser('evaluate')
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='musa')
    p.add_argument('--test-tasks',type=int,default=2,help='Tasks PER SEED (default: 2)')
    seeds=p.add_mutually_exclusive_group()
    seeds.add_argument('--test-seeds',type=int,nargs='+',help='Default: 2025 2026 2027 2028 2029')
    seeds.add_argument('--test-seed',type=int,help='Evaluate one seed; use --test-tasks 1 to replay a single task')
    p.add_argument('--cache-dir')
    p.add_argument('--baseline-epochs',type=int,default=500)
    p.add_argument('--tnet-epochs',type=int,default=160)
    p.add_argument('--hypersci-arm-samples',type=int,default=4096)
    p.add_argument('--pfn-only',action='store_true',help='Diagnostic smoke only; formal run includes every baseline')
    add_localized_arguments(p)
    p=commands.add_parser('diagnose')
    p.add_argument('--graph-file',default=defaults.graph_file)
    p.add_argument('--cache-dir',default=defaults.cache_dir)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--tasks',type=int,default=128)
    p.add_argument('--data-seed',type=int,default=12345)
    p.add_argument('--noise-sd',type=float,default=4.)
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='train':
        output=args.pop('output');resume=args.pop('resume')
        train(RunConfig(**args),output,resume=resume)
    elif command=='evaluate':
        args['baselines']=not args.pop('pfn_only')
        args['localized_options']={key:args.pop(key) for key in LOCALIZED_DEFAULTS}
        evaluate(**args)
    else: diagnose_prior(**args)


if __name__=='__main__': main()
