#!/usr/bin/env python3
"""Small real-Torch integration gate before large MUSA data loading.

Checks exact CSV/cache tensor equivalence, streaming metrics, real two-process
DDP updates, unequal validation partitions and a rank with zero validation rows.
Never reads the production CSV or changes production checkpoints.
"""
from __future__ import annotations

import os
import sys
import math
import subprocess
import socket
import tempfile
import traceback
from pathlib import Path


def compare_metrics(actual, expected):
    for key,value in expected.items():
        if not math.isclose(actual[key],value,rel_tol=5e-5,abs_tol=5e-5):
            raise AssertionError(f"metric {key}: {actual[key]} != {value}")


def worker(root):
    import torch
    import torch.distributed as dist
    import pfn_interference.model as m
    runtime,device=m.initialize_distributed("cpu")
    def partition(batch):
        idx=torch.tensor(m.validation_indices(batch["tokens"].shape[0],runtime.rank,runtime.world_size),dtype=torch.long)
        return {k:v.index_select(0,idx) for k,v in batch.items()}
    full_train,_=m.load_cached_episode_csv(root/"train.csv",8)
    full_val,_=m.load_cached_episode_csv(root/"val.csv",5)
    full_train["cepo_target"]=m.cepo_targets(full_train,treatment_prob=.5)
    full_val["cepo_target"]=m.cepo_targets(full_val,treatment_prob=.5)
    local_train=partition(full_train)
    local_val=partition(full_val)
    verify_training_rmse(m, torch, runtime, device, local_train)
    model=m.train(
        data_config=m.DataConfig(n_units=10,er_edge_probability=.31),
        model_config=m.ModelConfig(d_model=16,num_heads=4,num_layers=1,ffn_dim=32),
        train_config=m.TrainConfig(epochs=2,batch_size=2,training_seed=0),
        train_batch=local_train,validation_batch=local_val,device=device,
        checkpoint_path=root/"model.pt",runtime=runtime,rank_sharded_train=True,
        distributed_validation=True,
    )
    params=torch.cat([p.detach().flatten() for p in model.parameters()])
    parts=[torch.empty_like(params) for _ in range(runtime.world_size)]
    dist.all_gather(parts,params)
    for p in parts: torch.testing.assert_close(p,params,rtol=0,atol=0)
    for count in (5,1):
        full={k:v[:count] for k,v in full_val.items()}
        metrics=m.evaluate(model,partition(full),batch_size=2,device=device,
                           runtime=runtime,distributed_validation=True)
        if runtime.is_main:
            model.eval()
            with torch.no_grad():
                pred=model(full["tokens"],m.cepo_queries(full),full["adjacency"])
                expected=m.compute_query_metrics(model,pred,full)
            compare_metrics(metrics,expected)
            print(f"真实 DDP：验证任务 {count} 个，全局指标与未分片计算一致。",flush=True)
        dist.barrier()
    m.cleanup_distributed(runtime)


def verify_training_rmse(m, torch, runtime, device, local_train):
    """Real DDP/Adam/GMM path with hand-checkable global squared errors."""
    class KnownGMM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = m.ModelConfig(gmm_n_components=1)
            self.offset = torch.nn.Parameter(torch.zeros(()))

        def forward(self, tokens, queries, adjacency):
            mu = tokens[..., :3].unsqueeze(-1) + self.offset
            return {"gmm_pi": torch.ones_like(mu), "gmm_mu": mu,
                    "gmm_sigma": torch.ones_like(mu)}

    # Two ranks, errors 1 and 3. The second case has a shorter final batch
    # on rank zero, but both ranks still take two optimizer steps.
    for counts, expected_mse in [((4, 4), 5.), ((3, 4), 39. / 7.)]:
        count = counts[runtime.rank]
        batch = {key: value[:count].clone() for key, value in local_train.items()}
        truth = torch.ones(count, batch["tokens"].shape[1], 3)
        batch["cepo_target"] = truth * torch.tensor([0.,1.,2.])
        batch["oracle_ite"] = truth * torch.tensor([1.,1.,2.])
        error = (1 + 2 * runtime.rank) * torch.tensor([0., 1., 3.])
        batch["tokens"][..., :3] = batch["cepo_target"] + error
        model = m.DistributedDataParallel(KnownGMM(), broadcast_buffers=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.)
        metrics = m.train_epoch(model, optimizer, batch, batch_size=2,
            training_seed=0, epoch=1, device=device, runtime=runtime,
            data_is_rank_sharded=True, return_metrics=True)
        for factor, effect in enumerate(("direct", "spillover", "total"), 1):
            expected = factor * math.sqrt(expected_mse)
            for alias in (effect, effect + "_effect"):
                assert math.isclose(metrics[alias + "_rmse"], expected, rel_tol=1e-6)
                assert math.isclose(metrics[alias + "_nrmse_pct"], 100 * expected/(2 if effect=="total" else 1), rel_tol=1e-6)
        assert math.isclose(metrics["effect_rmse_macro"], 2 * math.sqrt(expected_mse), rel_tol=1e-6)
        expected_nll = .5 * expected_mse * (10 / 3) + .5 * math.log(2 * math.pi)
        assert math.isclose(metrics["total_loss"], expected_nll, rel_tol=1e-6)
        if runtime.is_main:
            print(f"真实 DDP 训练指标：counts={counts} Direct RMSE={metrics['direct_rmse']:.6f}，全局汇总正确。", flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=="--worker":
        worker(Path(sys.argv[2]));return
    import torch
    import pfn_interference.model as m
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="pfn_cpu_verify_",dir=os.environ.get("TMPDIR")) as directory:
        root=Path(directory)
        cfg=m.DataConfig(n_units=10,er_edge_probability=.31)
        for name,count,seed in [("train",8,731),("val",5,732)]:
            batch=m.generate_batch(cfg,batch_size=count,seed=seed,device=torch.device("cpu"))
            m.save_episode_batch_csv(batch,root/f"{name}.csv")
        source=root/"train.csv"
        reference,ids=m._load_episode_batch_csv_reference(source,device=torch.device("cpu"),expected_dataset_count=8)
        fast,ids2=m.load_episode_batch_csv(source,device=torch.device("cpu"),expected_dataset_count=8)
        assert ids==ids2 and reference.keys()==fast.keys()
        for key in reference: torch.testing.assert_close(reference[key],fast[key],rtol=0,atol=0)
        for _ in range(2):
            cached,ids3=m.load_cached_episode_csv(source,8)
            assert ids==ids3
            for key in reference: torch.testing.assert_close(reference[key],cached[key],rtol=0,atol=0)
        print("原解析器、新解析器、缓存逐张量完全一致。",flush=True)
        # Corrupt only a temporary cache; the valid CSV must rebuild it.
        from pfn_interference.runtime import TENSOR_CACHE_VERSION
        source.with_name(source.name+f".tensor_v{TENSOR_CACHE_VERSION}.pt").write_bytes(b"invalid cache")
        rebuilt,_=m.load_cached_episode_csv(source,8)
        for key in reference: torch.testing.assert_close(reference[key],rebuilt[key],rtol=0,atol=0)
        # Mutating the temporary source must invalidate the old cached content.
        import csv
        with source.open(newline="") as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        rows[0]["y_obs"]=str(float(rows[0]["y_obs"])+1.)
        with source.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        refreshed,_=m.load_cached_episode_csv(source,8)
        assert refreshed["y_obs"][0,0]!=reference["y_obs"][0,0]
        m.save_episode_batch_csv(reference,source)
        # Full subprocess exercises actual Gloo collectives, not mocked calls.
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                 PFN_CONTROL_TIMEOUT="120",PFN_TRAIN_TIMEOUT="120")
        # Explicit loopback avoids hostname/IPv6 discovery on local test hosts.
        with socket.socket() as endpoint:
            endpoint.bind(("127.0.0.1", 0))
            port = endpoint.getsockname()[1]
        command=[sys.executable,"-u","-m","torch.distributed.run","--nnodes=1",
                 "--master-addr=127.0.0.1", f"--master-port={port}", "--nproc-per-node=2",
                 "--module","pfn_interference.experiments.verify_runtime_cpu","--worker",str(root)]
        process=subprocess.Popen(command,env=env,start_new_session=True)
        try:
            code=process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        if code: raise subprocess.CalledProcessError(code,command)
        print("CPU 数据与真实双进程训练回归检查全部通过。",flush=True)


if __name__=="__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
        if os.environ.get("WORLD_SIZE","1")!="1": os._exit(1)
        raise
