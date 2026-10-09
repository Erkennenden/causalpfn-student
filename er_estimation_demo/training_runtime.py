"""Data/metric plumbing only; model, estimands and GMM objective are unchanged."""
from __future__ import annotations

import csv
import math
import os
import json
import tempfile
from pathlib import Path

import numpy as np

TENSOR_CACHE_VERSION = 2


def cached_cpu_batch(path, loader, expected_dataset_count=None):
    """Atomic, source-validated CPU tensor cache; flock coordinates all ranks."""
    import fcntl
    import hashlib
    import torch
    path=Path(path)
    if os.environ.get("PFN_TENSOR_CACHE", "1") == "0":
        return loader(path, expected_dataset_count=expected_dataset_count)
    cache=path.with_name(path.name+f".tensor_v{TENSOR_CACHE_VERSION}.pt")
    with cache.with_suffix(cache.suffix+".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        stat=path.stat()
        hasher=hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda:source.read(8*1024*1024),b""):
                hasher.update(block)
        identity={"version":TENSOR_CACHE_VERSION,"sha256":hasher.hexdigest(),"size":stat.st_size,
                  "expected_count":expected_dataset_count}
        if cache.is_file():
            try:
                value=torch.load(cache,map_location="cpu",weights_only=True,mmap=True)
                if value["identity"] == identity:
                    print(f"复用张量缓存：{cache}",flush=True)
                    return value["batch"],value["ids"]
            except Exception as exc:
                print(f"缓存不可读，将从已校验 CSV 重建：{type(exc).__name__}",flush=True)
        print(f"首次建立张量缓存：{path}",flush=True)
        batch,ids=loader(path,expected_dataset_count=expected_dataset_count)
        fd,tmp=tempfile.mkstemp(prefix=cache.name+".",suffix=".tmp",dir=cache.parent)
        os.close(fd)
        try:
            torch.save({"identity":identity,"batch":batch,"ids":ids},tmp)
            os.replace(tmp,cache)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        # Return file-backed storage even to the rank which built the cache.
        del batch
        value=torch.load(cache,map_location="cpu",weights_only=True,mmap=True)
        print(f"张量缓存完成：{cache}",flush=True)
        return value["batch"],value["ids"]


def validation_indices(count, rank, world_size):
    if count < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid validation partition")
    return list(range(rank, count, world_size))


class MetricAccumulator:
    """Merge node-weighted metrics, reconstructing nonlinear ratios and RMSE."""
    def __init__(self):
        self.count = 0
        self.sums = {}
        self.abs_truth = [0., 0., 0.]

    def add(self, metrics, count, abs_truth_means):
        if count <= 0:
            return
        self.count += int(count)
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.) + float(value) * count
        for i, value in enumerate(abs_truth_means):
            self.abs_truth[i] += float(value) * count

    def payload(self):
        return {"count": self.count, "sums": self.sums, "abs_truth": self.abs_truth}

    def merge(self, payload):
        self.count += payload["count"]
        for key, value in payload["sums"].items():
            self.sums[key] = self.sums.get(key, 0.) + value
        for i, value in enumerate(payload["abs_truth"]):
            self.abs_truth[i] += value

    def result(self):
        if self.count == 0:
            raise ValueError("No samples in metric accumulator")
        out = {key: value / self.count for key, value in self.sums.items()}
        rmses=[]
        for i, effect in enumerate(("direct", "spillover", "total")):
            prefix=effect+"_effect"
            mse=out[prefix+"_mse"]
            rmse=math.sqrt(max(mse,0.))
            scale=max(self.abs_truth[i]/self.count,1e-8)
            for alias in (prefix,effect):
                out[alias+"_rmse"]=rmse
                out[alias+"_nmae_pct"]=out[prefix+"_mae"]/scale*100.
                out[alias+"_nrmse_pct"]=rmse/scale*100.
            rmses.append(rmse)
        out["effect_rmse_macro"]=sum(rmses)/3.
        for arm in ('mu00','mu01','mu10','mu11'):
            if arm+'_mse' in out:
                out[arm+'_rmse']=math.sqrt(max(out[arm+'_mse'],0.))
        return out


def read_csv_arrays(path, schema_version, token_dim, query_dim, query_names,
                    expected_dataset_count=None, progress=None):
    """Parse into NumPy buffers instead of millions of scalar torch writes.

    The same float32 representation and compact-CSV fields are retained.
    Adjacency decoding is vectorized; this function never touches an accelerator.
    """
    path=Path(path)
    if expected_dataset_count is not None and expected_dataset_count <= 0:
        raise ValueError("expected_dataset_count must be positive")
    with path.open(encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        try:
            first=next(reader)
        except StopIteration as exc:
            raise ValueError("CSV dataset is empty") from exc
        n=int(first["n_units"])
    if n <= 0:
        raise ValueError("n_units must be positive")
    if expected_dataset_count is None:
        with path.open(encoding="utf-8",newline="") as f:
            ids=sorted({int(row["dataset_id"]) for row in csv.DictReader(f)})
        index={v:i for i,v in enumerate(ids)}
        count=len(ids)
    else:
        count=int(expected_dataset_count)
        ids=[]
        index={}
    arrays={
        "tokens":np.empty((count,n,token_dim),np.float32),
        "queries":np.empty((count,n,3,query_dim),np.float32),
        "adjacency":np.empty((count,n,n),np.uint8),
        "graph_type":np.empty(count,np.int64),
        "star_center":np.empty(count,np.int64),
    }
    scalar_names=("x","observed_treatment","observed_exposure","degree","majority_arm",
                  "low_sampled_exposure","high_sampled_exposure","y_obs","structural_baseline",
                  "tau","gamma","eta")
    for name in scalar_names:
        arrays[name]=np.empty((count,n),np.int64 if name=="majority_arm" else np.float32)
    for name in ("outcome_a","outcome_b","query_effect","oracle_ite"):
        arrays[name]=np.empty((count,n,3),np.float32)
    seen=np.zeros((count,n),bool)
    meta_seen=np.zeros(count,bool)
    complete=0
    with path.open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if row["schema_version"] != schema_version:
                raise ValueError("CSV schema version is stale; regenerate the dataset.")
            if int(row["n_units"]) != n:
                raise ValueError("CSV rows contain inconsistent n_units values.")
            dataset_id=int(row["dataset_id"])
            if dataset_id not in index:
                if len(ids)>=count:
                    raise ValueError("CSV episode count exceeds expected_dataset_count")
                index[dataset_id]=len(ids);ids.append(dataset_id)
            b=index[dataset_id];u=int(row["unit_id"])
            if not 0<=u<n:
                raise ValueError(f"dataset_id={dataset_id} has unit_id={u} outside 0 through {n-1}")
            if seen[b,u]:
                raise ValueError(f"dataset_id={dataset_id} repeats unit_id={u}")
            seen[b,u]=True
            gt=int(row["graph_type"]);center=int(row["star_center"])
            if meta_seen[b] and (arrays["graph_type"][b]!=gt or arrays["star_center"][b]!=center):
                raise ValueError("Inconsistent episode metadata")
            arrays["graph_type"][b]=gt;arrays["star_center"][b]=center;meta_seen[b]=True
            for name in scalar_names:
                arrays[name][b,u]=int(row[name]) if name=="majority_arm" else float(row[name])
            arrays["tokens"][b,u]=[float(row[f"token_{j}"]) for j in range(token_dim)]
            encoded=int(row["adjacency_bits"],16)
            if encoded<0 or encoded>>n:
                raise ValueError("Encoded binary mask exceeds the requested length.")
            packed=np.frombuffer(encoded.to_bytes((n+7)//8,"little"),dtype=np.uint8)
            arrays["adjacency"][b,u]=np.unpackbits(packed,bitorder="little")[:n]
            for q,name in enumerate(query_names):
                arrays["queries"][b,u,q]=[float(row[f"{name}_query_{j}"]) for j in range(query_dim)]
                for dest,suffix in (("outcome_a","outcome_a"),("outcome_b","outcome_b"),("query_effect","effect")):
                    arrays[dest][b,u,q]=float(row[f"{name}_{suffix}"])
                arrays["oracle_ite"][b,u,q]=float(row[f"oracle_{name}_ite"])
            if u == n-1:
                complete+=1
                if progress and (complete%128==0 or complete==count):
                    progress(f"CSV 已解析 {complete}/{count} 个任务")
    if len(ids)!=count:
        raise ValueError("CSV episode count does not match expected_dataset_count")
    if not seen.all():
        raise ValueError(f"Each dataset must contain unit_id 0 through {n-1} exactly once")
    return arrays,ids
