"""Lazy task-bank CEPO training; one complete pass per epoch, optional MUSA DDP."""
from dataclasses import dataclass, asdict
from datetime import timedelta
from pathlib import Path
import json
import math
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .data import ROOT, TaskBank, atomic_save, write_json
from pfn_interference.cepo import PREDICTION_PROTOCOL, ARM_NAMES, cepo_queries, compute_cepo_losses, compute_cepo_metrics
from pfn_interference.runtime import MetricAccumulator
from pfn_interference.model import (ModelConfig, LocalNetworkQueryTransformer,
    initialize_distributed, cleanup_distributed, seed_everything, epoch_episode_indices,
    create_optimizer, create_plateau_scheduler, resolve_device)

VERSION='er_estimation_demo_linear_b_random_a_four_gmm_v1'


@dataclass(frozen=True)
class RunConfig:
    graph_file: str = str(ROOT/'data/fixed_er_N1000_p0.02_seed12345.json')
    cache_dir: str = str(ROOT/'runs/task_cache')
    device: str = 'musa'
    train_tasks: int = 40960
    validation_tasks: int = 5120
    epochs: int = 125
    batch_size: int = 16
    data_seed: int = 12345
    train_seed: int = 0
    noise_sd: float = 4.
    num_layers: int = 10
    d_model: int = 128
    num_heads: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    threads: int = 1
    def __post_init__(self):
        for name in ('train_tasks','validation_tasks','epochs','batch_size','num_layers','d_model','num_heads','threads'):
            if not isinstance(getattr(self,name),int) or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        if self.d_model % self.num_heads:
            raise ValueError('d_model must be divisible by num_heads')
        for name in ('noise_sd','weight_decay','learning_rate'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.learning_rate==0:
            raise ValueError('learning_rate must be positive')
        if any(not isinstance(s,int) or not 0<=s<2**63 for s in (self.data_seed,self.train_seed)):
            raise ValueError('Seeds must be nonnegative signed-64-bit integers')


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return type(value)(cpu_tree(v) for v in value)
    return value


def to_device(batch,device):
    return {k:v.to(device) for k,v in batch.items()}


def config_identity(config):
    return {k:v for k,v in asdict(config).items() if k not in ('epochs','device','threads','graph_file','cache_dir')}


def load_model(path,*,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('version')!=VERSION or state.get('prediction_protocol')!=PREDICTION_PROTOCOL:
        raise ValueError('Checkpoint is not the current random-a four-GMM release; train a new model for the new DGP and graph')
    model=LocalNetworkQueryTransformer(ModelConfig(**state['model_config'])).to(device)
    model.load_state_dict(state['model_state'])
    return model,state


def _validation(model,bank,count,batch_size,device,runtime):
    acc=MetricAccumulator()
    ids=list(range(runtime.rank,count,runtime.world_size))
    model.eval()
    with torch.no_grad():
        for start in range(0,len(ids),batch_size):
            b=to_device(bank.batch('validation',ids[start:start+batch_size]),device)
            prediction=model(b['tokens'],cepo_queries(b),b['adjacency'])
            pred_cpu={k:v.detach().cpu() for k,v in prediction.items()}
            truth=b['oracle_ite'].detach().cpu()
            metrics=compute_cepo_metrics(pred_cpu,{'cepo_target':b['cepo_target'].cpu(),'oracle_ite':truth},include_intervals=False)
            acc.add(metrics,truth.shape[0]*truth.shape[1],[float(truth[...,j].abs().double().mean()) for j in range(3)])
            if runtime.is_main and (start==0 or (start//batch_size+1)%20==0):
                print(f'validation {start+len(b["tokens"])}/{len(ids)} rank0 tasks',flush=True)
    if runtime.is_distributed:
        parts=[None]*runtime.world_size
        dist.all_gather_object(parts,acc.payload())
        acc=MetricAccumulator()
        for part in parts: acc.merge(part)
    return acc.result()


def train(config,output,*,resume=False):
    torch.set_num_threads(config.threads)
    runtime,device=initialize_distributed(config.device)
    try:
        return _train(config,Path(output),resume,runtime,device)
    finally:
        cleanup_distributed(runtime)


def _train(config,output,resume,runtime,device):
    global_batch=config.batch_size*runtime.world_size
    if config.train_tasks % global_batch:
        raise ValueError('train_tasks must be divisible by batch_size * world_size; no duplicated or dropped tasks')
    error=None
    if runtime.is_main:
        if resume and not (output/'model_last.pt').is_file(): error='Resume requires model_last.pt'
        elif not resume and output.exists() and any(output.iterdir()): error='Output is nonempty; choose a new output or --resume'
    if runtime.is_distributed:
        message=[error];dist.broadcast_object_list(message,src=0);error=message[0]
    if error: raise ValueError(error)
    output.mkdir(parents=True,exist_ok=True)
    bank=TaskBank(config.graph_file,config.cache_dir,data_seed=config.data_seed,noise_sd=config.noise_sd)
    seed_everything(config.train_seed)
    model_config=ModelConfig(num_layers=config.num_layers,d_model=config.d_model,
        num_heads=config.num_heads,ffn_dim=4*config.d_model,dropout=0.)
    raw=LocalNetworkQueryTransformer(model_config).to(device)
    optimizer=create_optimizer(raw.parameters(),learning_rate=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=create_plateau_scheduler(optimizer,factor=.5,patience=5)
    start_epoch=0;best=float('inf');best_epoch=0;previous_seconds=0.
    if resume:
        state=torch.load(output/'model_last.pt',map_location='cpu',weights_only=True)
        if state.get('version')!=VERSION or state.get('prediction_protocol')!=PREDICTION_PROTOCOL:
            raise ValueError('Incompatible checkpoint protocol')
        if state['data_identity']!=bank.fingerprint or state['config_identity']!=config_identity(config):
            raise ValueError('Resume data, model or training configuration mismatch')
        if state['world_size']!=runtime.world_size:
            raise ValueError('Resume must retain world_size and effective batch size')
        raw.load_state_dict(state['model_state']);optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        start_epoch=state['epoch'];best=state['best_validation_nll'];best_epoch=state['best_epoch']
        previous_seconds=state['elapsed_seconds']
        if config.epochs<start_epoch: raise ValueError('epochs is smaller than checkpoint epoch')
    group=None
    if runtime.is_distributed and device.type in ('musa','privateuseone'):
        torch.musa.synchronize()
        group=dist.new_group(backend='mccl',timeout=timedelta(seconds=int(os.environ.get('PFN_TRAIN_TIMEOUT','600'))))
    model=DistributedDataParallel(raw,device_ids=None if device.type=='cpu' else [device.index],
        output_device=None if device.type=='cpu' else device.index,broadcast_buffers=False,process_group=group) if runtime.is_distributed else raw
    history=output/'history.jsonl'
    if runtime.is_main:
        retained=[]
        if resume and history.exists():
            retained=[json.loads(line) for line in history.read_text().splitlines() if json.loads(line)['epoch']<=start_epoch]
        history.write_text(''.join(json.dumps(row)+'\n' for row in retained))
        write_json(output/'fixed_graph.json',bank.record)
        write_json(output/'manifest.json',dict(version=VERSION,prediction_protocol=PREDICTION_PROTOCOL,
            config=asdict(config),model_config=asdict(model_config),data_identity=bank.fingerprint,
            data_prior=bank.identity,world_size=runtime.world_size,effective_batch_size=global_batch,
            parameters=sum(p.numel() for p in raw.parameters()),model_name='PFN with interference'))
        write_json(output/'status.json',dict(status='training',epoch=start_epoch,target_epochs=config.epochs))
    started=time.perf_counter()
    for epoch in range(start_epoch+1,config.epochs+1):
        ids=epoch_episode_indices(config.train_tasks,training_seed=config.train_seed,epoch=epoch).tolist()
        local_ids=ids[runtime.rank::runtime.world_size]
        model.train();nll_sums={name:0. for name in ARM_NAMES};seen=0
        for start in range(0,len(local_ids),config.batch_size):
            b=to_device(bank.batch('train',local_ids[start:start+config.batch_size]),device)
            optimizer.zero_grad(set_to_none=True)
            prediction=model(b['tokens'],cepo_queries(b),b['adjacency'])
            losses=compute_cepo_losses(prediction,b)
            loss=losses['total_loss']
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite CEPO loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(raw.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            for name in ARM_NAMES:
                nll_sums[name]+=float(losses[name+'_gmm_nll'].detach().cpu())*len(b['tokens'])
            seen+=len(b['tokens'])
            if runtime.is_main and (start==0 or (start//config.batch_size+1)%20==0):
                print(f'epoch={epoch}/{config.epochs} step={start//config.batch_size+1}/{len(local_ids)//config.batch_size} CEPO_NLL={float(loss.detach().cpu()):.6f}',flush=True)
        count=torch.tensor([*[nll_sums[name] for name in ARM_NAMES],seen],dtype=torch.float64)
        if runtime.is_distributed: dist.all_reduce(count)
        train_metrics={name+'_gmm_nll':float(count[j]/count[-1]) for j,name in enumerate(ARM_NAMES)}
        train_metrics['total_loss']=sum(train_metrics.values())/len(ARM_NAMES)
        train_metrics['cepo_gmm_nll_macro']=train_metrics['total_loss']
        metrics=_validation(raw,bank,config.validation_tasks,config.batch_size,device,runtime)
        nll=metrics['validation_gmm_nll'];scheduler.step(nll)
        improved=nll<best
        if improved: best=nll;best_epoch=epoch
        if runtime.is_main:
            state=dict(version=VERSION,prediction_protocol=PREDICTION_PROTOCOL,config=asdict(config),
                config_identity=config_identity(config),model_config=asdict(model_config),
                data_identity=bank.fingerprint,graph_record=bank.record,model_state=cpu_tree(raw.state_dict()),
                optimizer_state=cpu_tree(optimizer.state_dict()),scheduler_state=scheduler.state_dict(),
                epoch=epoch,best_epoch=best_epoch,best_validation_nll=best,world_size=runtime.world_size,
                task_presentations=epoch*config.train_tasks,unique_train_tasks_seen=config.train_tasks,
                elapsed_seconds=previous_seconds+time.perf_counter()-started)
            if improved: atomic_save(state,output/'model_best.pt')
            atomic_save(state,output/'model_last.pt')
            row=dict(epoch=epoch,train=train_metrics,validation=metrics,
                lr=optimizer.param_groups[0]['lr'],elapsed_seconds=state['elapsed_seconds'],
                task_presentations=state['task_presentations'])
            with history.open('a') as handle: handle.write(json.dumps(row)+'\n')
            from pfn_interference.plotting import plot_history
            plot_history(history,output)
            write_json(output/'status.json',dict(status='training',epoch=epoch,target_epochs=config.epochs,best_epoch=best_epoch))
            print(f'epoch={epoch} validation_CEPO_NLL={nll:.6f} best_epoch={best_epoch}',flush=True)
        if runtime.is_distributed: dist.barrier()
    if runtime.is_main:
        write_json(output/'status.json',dict(status='training_complete',epoch=config.epochs,best_epoch=best_epoch))
    return output/'model_best.pt'
