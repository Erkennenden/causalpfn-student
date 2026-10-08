"""One-factor-at-a-time plans and resumable sequential execution."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import torch

from scaling_v1.runner import MODEL_SIZES, RunConfig, device_for, train
from scaling_v1.data import TaskBank, write_json
from . import VERSION


@dataclass(frozen=True)
class Job:
    name: str
    config: RunConfig


def build_jobs(base, *, task_counts, model_sizes, step_budgets, seeds):
    axes = (('task_counts', task_counts), ('model_sizes', model_sizes),
            ('step_budgets', step_budgets), ('seeds', seeds))
    for name, values in axes:
        if not values or len(set(values)) != len(values):
            raise ValueError(f'{name} must be nonempty and distinct.')
        if name == 'model_sizes':
            if any(v not in MODEL_SIZES for v in values):
                raise ValueError('Unknown model size.')
        elif any(not isinstance(v, int) or isinstance(v, bool) or
                 v < (0 if name == 'seeds' else 1) or v >= 2**63 for v in values):
            raise ValueError(f'Invalid {name}.')
    if (base.train_tasks not in task_counts or base.model_size not in model_sizes
            or base.steps not in step_budgets):
        raise ValueError('Each axis must include its anchor value.')
    triples = [(d, base.model_size, base.steps) for d in task_counts]
    triples += [(base.train_tasks, m, base.steps) for m in model_sizes]
    triples += [(base.train_tasks, base.model_size, s) for s in step_budgets]
    return [Job(f'{m}_D{d}_S{s}_seed{seed}',replace(base,train_tasks=d,model_size=m,steps=s,train_seed=seed))
            for d,m,s in dict.fromkeys(triples) for seed in seeds]


def config_identity(config):
    # Only operational settings may change while resuming an existing plan.
    return {k:v for k,v in config.items() if k not in {'graph_file','cache_dir','device','threads'}}


def plan_identity(plan):
    return [(j['name'],config_identity(j['config'])) for j in plan['jobs']]


def check_resume_runtime(config,output):
    """Keep accumulated timings under one runtime throughout a resumed run."""
    manifest_path=Path(output)/'manifest.json'
    if not manifest_path.exists():
        raise ValueError('Missing runtime manifest; use a new output directory.')
    previous=json.loads(manifest_path.read_text())
    expected=(str(device_for(config.device)),config.threads,str(torch.__version__))
    actual=(previous.get('device'),previous.get('config',{}).get('threads'),previous.get('torch_version'))
    if expected!=actual:
        raise ValueError('Cannot resume with a different runtime (device, threads or PyTorch version); '
                         'restore the original settings or use a new output directory.')


def run_sweep(jobs, output, *, resume=False):
    from .report import collect_runs, summarize
    if not jobs:
        raise ValueError('The plan is empty.')
    output=Path(output)
    plan=dict(version=VERSION,jobs=[dict(name=j.name,config=asdict(j.config)) for j in jobs])
    plan_path=output/'sweep_plan.json'
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError('Output contains files; use --resume or a new directory.')
        if not plan_path.exists():
            raise ValueError('No V2 plan found; choose a new output directory.')
        previous=json.loads(plan_path.read_text())
        if previous.get('version')!=VERSION or plan_identity(previous)!=plan_identity(plan):
            raise ValueError('Sweep plan mismatch; keep the same axes/seeds/settings or use a new directory.')
    cfg=jobs[0].config
    bank=TaskBank(cfg.graph_file,cfg.cache_dir,data_seed=cfg.data_seed,noise_sd=cfg.noise_sd)
    if plan_path.exists():
        previous=json.loads(plan_path.read_text())
        if previous.get('data_identity')!=bank.fingerprint:
            raise ValueError('Sweep data identity mismatch.')
    plan['data_identity']=bank.fingerprint
    write_json(plan_path,plan)
    for index,job in enumerate(jobs,1):
        target=output/job.name
        complete=collect_runs(target) if target.exists() else []
        if complete and complete[0][1]['data_identity']==bank.fingerprint:
            if config_identity(complete[0][1]['config'])!=config_identity(asdict(job.config)):
                raise ValueError(f'Completed run config mismatch: {target}')
            print(f'[{index}/{len(jobs)}] Already complete: {job.name}',flush=True)
            continue
        print(f'[{index}/{len(jobs)}] Training: {job.name}',flush=True)
        recoverable=any((target/name).exists() for name in ('model_last.pt','model_initial.pt'))
        if resume and recoverable:
            check_resume_runtime(job.config,target)
        # Invalidate aggregate results before this run resumes. Rebuild on failure too.
        if (target/'status.json').exists():
            write_json(target/'status.json',dict(status='resuming'))
        summarize(output)
        try:
            train(job.config,target,resume=resume and recoverable)
        finally:
            summarize(output)
    return summarize(output)
