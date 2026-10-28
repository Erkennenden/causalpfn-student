"""Registered independent calibration then held-out testing; frozen estimators.

All factors for all six families are frozen before generating any test data.
Test order: f103 pilot, then remaining families; no result-dependent tuning.
Small benchmark shards reuse the existing evaluator and can resume safely.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from benchmark import run_benchmark, configuration as frozen_configuration
from evaluate import context_metadata, evaluate_run
from make_table import EFFECTS, write_csv
from pfn_social.e12_configuration import FAMILY_BASE_IDS, generate_mechanism, protocol_record
from pfn_social.e12_interval_calibration import (
    METHOD, INTERVAL_KIND, conformal_factor, interval_scores, apply_factors,
)

PROTOCOL = 'e12_independent_interval_calibration_v1'
CALIBRATION_IDS = tuple(range(2000,2100))
TEST_IDS = tuple(range(3000,3200))
BUDGETS = (20,50,100)
TEST_METHODS = ('j','ht','hajek','f-z2','lin-z2','kernel-old','kernel-new')
ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def event(root,stage,**fields):
    record=dict(time_utc=stamp(),stage=stage,**fields)
    with (root/'events.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


def source_fingerprints():
    paths=[ROOT/'benchmark.py',ROOT/'predict.py',ROOT/'run_e12_calibration.py',ROOT/'evaluate.py']
    paths+=sorted((ROOT/'pfn_social').glob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def register(root):
    config=frozen_configuration()
    specifications=dict(name=PROTOCOL,families=list(FAMILY_BASE_IDS),
        calibration_ids=list(CALIBRATION_IDS),test_ids=list(TEST_IDS),
        excluded_previously_inspected_ids=list(range(8))+list(range(1000,1100)),
        calibration_repeat_ids=[0],test_repeat_ids=[0,1,2],observed_outcomes=list(BUDGETS),
        nominal_level=.95,target_empirical_coverage=.90,
        calibration='Per family/budget/effect: order ceil((K+1)*.95), max(1,q), K=100 independent mechanisms.',
        original_methods=list(TEST_METHODS),additional_method=METHOD,
        evaluation='All 54 cells reported. Requested interval failures count as misses. Repeat=0 also reported separately.',
        test_order='f103 pilot, then all other families; no threshold-based stopping or tuning.',
        roles='Calibration/test are registered by disjoint mechanism IDs in this extension. Underlying LargeNetworkConfig retains dev to preserve its canonical identity; not the upstream final split.',
        development='Original v1 and targeted replication are already inspected development evidence, excluded here.',
        source_sha256=source_fingerprints(),frozen_assets_sha256=config['frozen_assets_sha256'])
    root.mkdir(parents=True,exist_ok=True)
    path=root/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text())
        if existing['specification']!=specifications:
            raise ValueError('Registered protocol or source changed; do not overwrite an ongoing experiment')
    else:
        dump(path,dict(registered_utc=stamp(),specification=specifications))
    return digest(path)


def shard_protocol(stage,family,ids,protocol_hash):
    value=protocol_record([family],0,1 if stage=='calibration' else 3,BUDGETS)
    value.update(name=PROTOCOL,role=stage,mechanism_ids=list(ids),registered_protocol_sha256=protocol_hash,
                 coverage='Frozen J versus independent-mechanism calibrated J; factors frozen before test generation.')
    return value


def run_shard(job):
    stage,family,ids,path,protocol_hash=job
    path=Path(path)
    methods=['j'] if stage=='calibration' else list(TEST_METHODS)
    protocol=shard_protocol(stage,family,ids,protocol_hash)
    if (path/'execution.json').exists():
        saved=json.loads((path/'execution.json').read_text())
        if saved['extension_protocol']!=protocol:
            raise ValueError('Shard identity mismatch')
        return str(path)
    if path.exists():
        # Retain interrupted partial outputs for audit, never merge them.
        parent=path.parent.resolve()
        if not path.resolve().is_relative_to(parent):
            raise ValueError('Unexpected partial output path')
        path.rename(parent/(path.name+'_incomplete_'+str(time.time_ns())))
    args=argparse.Namespace(methods=methods,interval_level=.95,output=path,cohort=PROTOCOL)
    grid=[(family,mid,range(1 if stage=='calibration' else 3),BUDGETS) for mid in ids]
    with redirect_stdout(io.StringIO()):
        run_benchmark(args,case_grid=grid,mechanism_factory=generate_mechanism,protocol=protocol)
    return str(path)


def link_file(source,target):
    if target.exists():
        if digest(source)!=digest(target):
            raise ValueError(f'Conflicting saved output: {target}')
        return
    try:
        os.link(source,target)
    except OSError:
        shutil.copyfile(source,target)


def merge(shards,destination,stage,protocol_hash,families):
    destination.mkdir(parents=True,exist_ok=True)
    receipts,graphs,manifests=[],[],[]
    for name in ('factual','truth','predictions'):
        (destination/name).mkdir(exist_ok=True)
    for shard in sorted(map(Path,shards)):
        manifest=json.loads((shard/'execution.json').read_text())
        manifests.append(manifest)
        receipts.extend(manifest['receipts'])
        graphs.extend(json.loads((shard/'graph_statistics.json').read_text()))
        for name in ('factual','truth','predictions'):
            for file in (shard/name).glob('*.npz'):
                link_file(file,destination/name/file.name)
    ids=CALIBRATION_IDS if stage=='calibration' else TEST_IDS
    repeats=[0] if stage=='calibration' else [0,1,2]
    methods=['j'] if stage=='calibration' else [*TEST_METHODS,METHOD]
    expected={(f'f{f}_m{m}_r{r}_observed{b}',method) for f in families for m in ids
              for r in repeats for b in BUDGETS for method in methods}
    actual=[(r['context_id'],r['method']) for r in receipts]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Missing or duplicate merged context/method identities')
    protocol=protocol_record(families,0,len(repeats),BUDGETS)
    protocol.update(name=PROTOCOL,role=stage,mechanism_ids=list(ids),registered_protocol_sha256=protocol_hash,
                    coverage='Frozen J and independent-mechanism calibrated J, no held-out retuning.')
    merged=dict(manifests[0],receipts=receipts,methods=methods,extension_protocol=protocol,
                interval_methods=['j'] if stage=='calibration' else ['j',METHOD],
                model_load_seconds=sum(m['model_load_seconds'] for m in manifests),
                merged_shards=[str(p.relative_to(destination.parent.parent)) for p in map(Path,shards)])
    dump(destination/'execution.json',merged)
    dump(destination/'graph_statistics.json',graphs)
    evaluate_run(destination)
    return destination


def freeze_calibration(calibration,root,protocol_hash):
    """No test input argument: fit solely from the registered calibration run."""
    artifact=root/'frozen_calibration.json'
    manifest=json.loads((calibration/'execution.json').read_text())
    expected={f'f{f}_m{m}_r0_observed{b}' for f in FAMILY_BASE_IDS for m in CALIBRATION_IDS for b in BUDGETS}
    receipts=manifest['receipts']
    if ({r['context_id'] for r in receipts}!=expected or len(receipts)!=len(expected)
            or any(r['method']!='j' or r['failure'] or r['interval_failure'] for r in receipts)):
        raise ValueError('Calibration requires the complete registered J-only set without failures')
    if artifact.exists():
        saved=json.loads(artifact.read_text())
        if (saved['registered_protocol_sha256']!=protocol_hash
                or saved['calibration_execution_sha256']!=digest(calibration/'execution.json')):
            raise ValueError('Frozen calibration provenance mismatch')
        return artifact
    rows=[]
    for receipt in receipts:
        context=receipt['context_id'];meta=context_metadata(context)
        with np.load(calibration/'truth'/f"f{meta['family']}_m{meta['mechanism']}.npz") as f:
            mu=f['arm_means']
        truth=np.stack((mu[:,2]-mu[:,0],mu[:,3]-mu[:,2],mu[:,3]-mu[:,0]),-1).mean(0)
        with np.load(calibration/'predictions'/f'{context}__j.npz') as f:
            score=interval_scores(truth,f['ate_lower'],f['ate_upper'])
        rows.extend(dict(**meta,context_id=context,effect=effect,score=float(score[i])) for i,effect in enumerate(EFFECTS))
    factors=[]
    for family in FAMILY_BASE_IDS:
        for budget in BUDGETS:
            for effect in EFFECTS:
                members=[r for r in rows if (r['family'],r['m_observed'],r['effect'])==(family,budget,effect)]
                if len(members)!=100 or {r['mechanism'] for r in members}!=set(CALIBRATION_IDS):
                    raise ValueError('Calibration mechanisms missing, duplicated, or contaminated')
                factor=conformal_factor([r['score'] for r in members])
                if not np.isfinite(factor['factor']):
                    raise ValueError('Calibration threshold is not finite')
                factors.append(dict(family=family,m_observed=budget,effect=effect,**factor))
    write_csv(root/'calibration_scores.csv',rows)
    write_csv(root/'calibration_factors.csv',factors)
    dump(artifact,dict(frozen_utc=stamp(),registered_protocol_sha256=protocol_hash,
         calibration_execution_sha256=digest(calibration/'execution.json'),
         scores_sha256=digest(root/'calibration_scores.csv'),factors=factors,
         calibration_mechanism_ids=list(CALIBRATION_IDS),calibration_repeat=0,
         test_truth_used=False,interval_kind=INTERVAL_KIND))
    event(root,'calibration_frozen',sha256=digest(artifact),factors=len(factors))
    return artifact


def add_calibrated(run,artifact,expected_hash):
    """Truth-free application; preserve all original point estimates and receipts."""
    if digest(artifact)!=expected_hash:
        raise ValueError('Frozen calibration changed')
    calibration=json.loads(artifact.read_text())
    factors={(r['family'],r['m_observed'],r['effect']):r['factor'] for r in calibration['factors']}
    manifest=json.loads((run/'execution.json').read_text())
    if METHOD in manifest['methods']:
        if manifest.get('calibration_sha256')!=expected_hash:
            raise ValueError('Calibrated shard hash mismatch')
        return
    added=[]
    for receipt in manifest['receipts']:
        if receipt['method']!='j':
            continue
        context=receipt['context_id'];meta=context_metadata(context)
        with np.load(run/'predictions'/f'{context}__j.npz') as saved:
            pred={key:saved[key].copy() for key in saved.files}
        began=time.perf_counter();failure=receipt['interval_failure']
        try:
            if failure:
                raise ValueError('Original J interval failed')
            pred['ate_lower'],pred['ate_upper']=apply_factors(pred['ate_lower'],pred['ate_upper'],
                [factors[meta['family'],meta['m_observed'],e] for e in EFFECTS])
        except ValueError as error:
            pred['ate_lower']=np.full(3,np.nan);pred['ate_upper']=np.full(3,np.nan)
            failure=dict(type=type(error).__name__,message=str(error))
        elapsed=time.perf_counter()-began
        pred['interval_kind']=np.array(INTERVAL_KIND)
        np.savez_compressed(run/'predictions'/f'{context}__{METHOD}.npz',**pred)
        added.append(dict(receipt,method=METHOD,interval_failure=failure,
                          interval_seconds=receipt['interval_seconds']+elapsed,
                          calibration_application_seconds=elapsed,
                          timing_kind='Reuses measured J point/interval time plus actual calibration application; no duplicate fitting'))
    manifest['receipts'].extend(added)
    manifest['methods'].append(METHOD);manifest['interval_methods'].append(METHOD)
    manifest['calibration_sha256']=expected_hash
    dump(run/'execution.json',manifest)
    evaluate_run(run)


def run_stage(root,stage,families,workers,protocol_hash,artifact=None):
    ids=CALIBRATION_IDS if stage=='calibration' else TEST_IDS
    jobs=[]
    for family in families:
        for start in range(0,len(ids),25):
            chunk=ids[start:start+25]
            path=root/stage/'shards'/f'f{family}_m{chunk[0]}'
            jobs.append((stage,family,chunk,str(path),protocol_hash))
    artifact_hash=digest(artifact) if artifact else None
    event(root,'stage_started',role=stage,families=list(families),shards=len(jobs),workers=workers,
          frozen_calibration_sha256=artifact_hash)
    paths=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(run_shard,job):job for job in jobs}
        for future in as_completed(pending):
            path=Path(future.result())
            if artifact:
                add_calibrated(path,artifact,artifact_hash)
            paths.append(path)
            event(root,'shard_complete',role=stage,completed=len(paths),total=len(jobs),shard=path.name)
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:
        parser.error('Use 1..8 workers; each worker uses one BLAS/PyTorch thread')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]='1'
    root=args.output.resolve();protocol_hash=register(root)
    event(root,'run_started',registered_protocol_sha256=protocol_hash)
    cal=run_stage(root,'calibration',list(FAMILY_BASE_IDS),args.workers,protocol_hash)
    merged=merge(cal,root/'calibration/merged','calibration',protocol_hash,list(FAMILY_BASE_IDS))
    artifact=freeze_calibration(merged,root,protocol_hash);artifact_hash=digest(artifact)
    pilot=run_stage(root,'test',[103],args.workers,protocol_hash,artifact)
    merge(pilot,root/'test/pilot','test',protocol_hash,[103])
    event(root,'pilot_complete',calibration_sha256=artifact_hash,no_parameters_changed=True)
    others=run_stage(root,'test',[f for f in FAMILY_BASE_IDS if f!=103],args.workers,protocol_hash,artifact)
    merge(pilot+others,root/'test/merged','test',protocol_hash,list(FAMILY_BASE_IDS))
    if digest(artifact)!=artifact_hash:
        raise ValueError('Calibration changed after test generation')
    event(root,'complete',calibration_sha256=artifact_hash,test_contexts=10800,method_contexts=86400)


if __name__=='__main__':
    main()
