"""Bridge the task-random LPE mechanism into the OLD fixed-CSV raw-output trainer.

Only dataset construction and cache provenance live here. No model transforms,
residual targets, task regeneration during epochs, or extra losses are used.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import pfn_interference.model as demo
from pfn_interference.prior import PriorConfig, generate_tasks, PRIOR_VERSION

TRAIN_STREAM = 'random_lpe_csv_bank_v1'
BENCHMARK_STREAM = 'independent_c_lpe_sigma1_baselines_v1'
MANIFEST_VERSION = 'fixed_csv_raw_lpe_two_regime_v3'


def _task_maps(task_metadata):
    graphs, regimes = {}, {}
    for task in task_metadata:
        dataset_id, regime = int(task['task_id']), int(task['regime'])
        if dataset_id in graphs or regime not in (0, 3):
            raise ValueError('Expected unique tasks with coupled regime 0 or 3.')
        graphs[dataset_id], regimes[dataset_id] = int(task['graph_type']), regime
    return graphs, regimes


def _regime_validation_counts(regimes, validation_count):
    n0 = sum(value == 0 for value in regimes.values())
    n3 = len(regimes) - n0
    # Exact global count; when counts are odd, the closest feasible proportions.
    count0 = int(math.floor(validation_count * n0 / len(regimes) + .5))
    count0 = min(n0, max(validation_count - n3, count0))
    return {0: count0, 3: validation_count - count0}


def select_two_regime_split_ids(task_metadata, *, validation_fraction, split_seed,
                                stratify_by_graph_type=True):
    """Stratify first by mechanism, then graph family, without scanning the master."""
    graphs, regimes = _task_maps(task_metadata)
    total = len(graphs)
    if total < 2 or not 0 < validation_fraction < 1:
        raise ValueError('Need at least two tasks and a validation fraction in (0,1).')
    validation_count = min(max(int(round(total * validation_fraction)), 1), total - 1)
    allocation = _regime_validation_counts(regimes, validation_count)
    validation_ids = []
    for regime in (0, 3):
        group = {i: graphs[i] for i in graphs if regimes[i] == regime}
        count = allocation[regime]
        if count == 0:
            continue
        if count == len(group):
            validation_ids.extend(group)
            continue
        _, selected, _ = demo.select_episode_split_ids_from_graph_types(
            group, validation_fraction=count / len(group), split_seed=split_seed + regime,
            stratify_by_graph_type=stratify_by_graph_type)
        validation_ids.extend(selected)
    validation_ids = sorted(validation_ids)
    training_ids = sorted(set(graphs) - set(validation_ids))
    if len(validation_ids) != validation_count:
        raise RuntimeError('Two-regime split changed the exact validation total.')
    return training_ids, validation_ids, dict(total_datasets=total,
        training_datasets=len(training_ids), validation_datasets=len(validation_ids))


def _regime_split_counts(task_metadata, training_ids, validation_ids):
    _, regimes = _task_maps(task_metadata)
    return {name: {str(r): sum(regimes[i] == r for i in ids) for r in (0, 3)}
            for name, ids in (('all', regimes), ('train', training_ids),
                              ('validation', validation_ids))}


def _valid_regime_split(manifest):
    try:
        _, regimes = _task_maps(manifest['task_metadata'])
        train, val = manifest['training_dataset_ids'], manifest['validation_dataset_ids']
        if set(regimes) != set(train) | set(val):
            return False
        allocation = _regime_validation_counts(regimes, len(val))
        return all(sum(regimes[i] == r for i in val) == allocation[r] for r in (0, 3))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _write_selected_csv_splits(source, train_path, val_path, training_ids, validation_ids):
    """Write the selected whole-task split in one streaming pass."""
    expected = set(training_ids) | set(validation_ids)
    val_set, seen = set(validation_ids), set()
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline='', encoding='utf-8') as src, \
         train_path.open('w', newline='', encoding='utf-8') as train, \
         val_path.open('w', newline='', encoding='utf-8') as val:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError('Master CSV needs a header.')
        writers = [csv.DictWriter(f, fieldnames=reader.fieldnames) for f in (train, val)]
        for writer in writers:
            writer.writeheader()
        for row in reader:
            dataset_id = int(row['dataset_id'])
            if dataset_id not in expected:
                raise ValueError('Master CSV contains a task outside the split metadata.')
            seen.add(dataset_id)
            writers[int(dataset_id in val_set)].writerow(row)
    if seen != expected:
        raise ValueError('Master CSV is missing tasks from the split metadata.')


def prior_from_data(config: demo.DataConfig) -> PriorConfig:
    if config.dgp_version not in ('random_lpe_v1', 'a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1'):
        raise ValueError('Expected random_lpe_v1 or a_group_fixed_v1 data_config.')
    if config.graph_protocol == 'fixed_er_v1':
        from pfn_interference.graph import FixedERPriorConfig
        return FixedERPriorConfig(n_units=config.n_units,
            er_edge_probability=config.er_edge_probability, treatment_prob=config.treatment_prob)
    # Graph sampler implementations are untouched. Match the prior version used
    # by the recent LPE benchmark, not the C model or its training procedure.
    if (config.configuration_degree_sigma != 1.0 or config.configuration_max_degree != 25
            or config.sbm_between_probability != .006 or not config.rgg_torus):
        raise ValueError('random_lpe_v1 requires the original graph calibration settings.')
    if config.dgp_version in ('a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1'):
        from pfn_interference.simulation.coefficients import NOISE_SD
        # PriorConfig is reused only for graph calibration. Outcome coefficients
        # and functions are fixed by a_group_dgp, recorded separately in identity.
        return PriorConfig(n_units=config.n_units, graph_family=config.graph_family,
            treatment_prob=config.treatment_prob, nointerference_prob=0.,
            noise_min=NOISE_SD, noise_max=NOISE_SD,
            function_families=('linear', 'exponential'))
    return PriorConfig(n_units=config.n_units, graph_family=config.graph_family,
        treatment_prob=config.treatment_prob, nointerference_prob=config.random_nointerference_prob,
        noise_min=config.random_noise_sd, noise_max=config.random_noise_sd,
        function_families=('linear','polynomial','exponential'),
        config_version=config.random_prior_version)


def generate_random_batch(config, *, task_ids, seed, stream=TRAIN_STREAM,
                          device='cpu', graph_types=None, balanced=False):
    ids=list(task_ids)
    prior=prior_from_data(config)
    from pfn_interference.simulation.coefficients import gamma_kwargs
    if config.graph_protocol == 'fixed_er_v1':
        from pfn_interference.simulation.coefficients import generate_tasks as a_tasks
        from pfn_interference.graph import adjacency_from_config
        if graph_types is not None and (graph_types.shape != (len(ids),)
                                      or not bool((graph_types == demo.ER_GRAPH).all())):
            raise ValueError('Fixed ER protocol cannot override its graph family.')
        return a_tasks(prior, ids, seed=seed, stream=stream, device=device,
                       fixed_adjacency=adjacency_from_config(config),
                       interference_lambda=config.interference_lambda, **gamma_kwargs(config))
    generator = generate_tasks
    outcome_kwargs = {}
    if config.dgp_version in ('a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1'):
        from pfn_interference.simulation.coefficients import generate_tasks as generator
        outcome_kwargs['interference_lambda'] = config.interference_lambda
        outcome_kwargs.update(gamma_kwargs(config))
    if graph_types is None:
        return generator(prior,ids,seed=seed,stream=stream,device=device,balanced=balanced,**outcome_kwargs)
    if balanced:
        raise ValueError('Do not combine balanced gates/families with graph_types overrides.')
    if graph_types.shape != (len(ids),):
        raise ValueError('graph_types must have shape [batch].')
    parts=[]
    for task_id,kind in zip(ids,graph_types.detach().cpu().tolist()):
        if kind not in demo.GRAPH_TYPE_NAMES:
            raise ValueError(f'Unsupported graph type {kind}.')
        local=replace(prior,graph_family=demo.GRAPH_TYPE_NAMES[kind])
        parts.append(generator(local,[task_id],seed=seed,stream=stream,device=device,**outcome_kwargs))
    if not parts:
        raise ValueError('task_ids must be nonempty.')
    return {k:torch.cat([b[k] for b in parts],0) for k in parts[0]}


def manifest_path(data_csv: Path) -> Path:
    return Path(data_csv).with_suffix(Path(data_csv).suffix+'.manifest.json')


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def _identity(config, *, num_datasets, data_seed, validation_fraction, split_seed):
    if num_datasets < 2:
        raise ValueError('num_datasets must be at least 2.')
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be strictly in (0,1).')
    # JSON normalization also makes tuple/list representation stable on reload.
    info=dict(manifest_version=MANIFEST_VERSION, schema_version=demo.CSV_SCHEMA_VERSION,
        data_config=asdict(config),prior_config=asdict(prior_from_data(config)),
        num_datasets=num_datasets,data_seed=int(data_seed),stream=TRAIN_STREAM,
        validation_fraction=validation_fraction,split_seed=int(split_seed),
        prior_source_sha256=file_sha256((Path(__file__).parent/'../prior.py')))
    if config.dgp_version in ('a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1'):
        from pfn_interference.simulation.coefficients import outcome_spec_for_config
        info['outcome_dgp'] = outcome_spec_for_config(config)
        info['outcome_source_sha256'] = file_sha256((Path(__file__).parent/'coefficients.py'))
    if config.graph_protocol == 'fixed_er_v1':
        from pfn_interference.graph import graph_metadata
        info['fixed_graph'] = graph_metadata(config)
        info['graph_source_sha256'] = file_sha256((Path(__file__).parent/'../graph.py'))
    return json.loads(json.dumps(info,sort_keys=True,allow_nan=False))



def _master_identity(identity):
    """Identity of immutable generated tasks, excluding the derived split."""
    normalized=json.loads(json.dumps(identity,sort_keys=True,allow_nan=False))
    normalized.pop('validation_fraction',None)
    normalized.pop('split_seed',None)
    return normalized


def _write_manifest_atomic(data_csv: Path, manifest: dict) -> None:
    mp=manifest_path(Path(data_csv)); mt=mp.with_name(mp.name+'.building')
    mt.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    mt.replace(mp)


def _valid_split_sets(manifest: dict, num_datasets: int, expected_validation_count: int) -> bool:
    try:
        train={int(value) for value in manifest.get('training_dataset_ids',[])}
        val={int(value) for value in manifest.get('validation_dataset_ids',[])}
    except (TypeError,ValueError):
        return False
    return bool(train and val and not (train & val) and train|val==set(range(num_datasets))
                and len(val)==expected_validation_count)


def ensure_prepared_random_master_split(config, *, data_csv, num_datasets, data_seed,
                                        validation_fraction, split_seed, verify_hashes=False):
    """Reuse an immutable random-LPE master while safely changing only its split.

    The master task bank depends on the DGP/config/data seed, but not on
    validation_fraction or split_seed.  This function therefore permits a 20%
    bank to be re-split to 5% without regenerating the expensive 102.4k tasks.
    """
    data_csv=Path(data_csv)
    expected=_identity(config,num_datasets=num_datasets,data_seed=data_seed,
                       validation_fraction=validation_fraction,split_seed=split_seed)
    try:
        m=json.loads(manifest_path(data_csv).read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:
        raise ValueError('Missing/invalid random CSV manifest; cannot safely reuse the master bank.') from exc
    if m.get('complete') is not True or _master_identity(m.get('identity',{})) != _master_identity(expected):
        raise ValueError('Random CSV master identity/config mismatch; use a NEW path or explicit overwrite.')
    saved=m.get('files',{}).get('master',{})
    if not data_csv.is_file() or data_csv.stat().st_size != saved.get('size_bytes'):
        raise ValueError('CSV manifest file missing/truncated: master.')
    if not demo.csv_schema_matches(data_csv,expected_n_units=config.n_units):
        raise ValueError('CSV manifest schema/size mismatch: master.')
    if verify_hashes and file_sha256(data_csv)!=saved.get('sha256'):
        raise ValueError('CSV manifest SHA256 mismatch: master.')

    expected_validation_count=int(round(num_datasets*validation_fraction))
    expected_validation_count=min(max(expected_validation_count,1),num_datasets-1)
    if (m.get('identity')==expected and _valid_split_sets(m,num_datasets,expected_validation_count)
            and _valid_regime_split(m)):
        return m

    task_metadata=m.get('task_metadata',[])
    dataset_graph_type={}
    try:
        for index,item in enumerate(task_metadata):
            dataset_id=int(item.get('task_id',index))
            dataset_graph_type[dataset_id]=int(item['graph_type'])
    except (AttributeError,KeyError,TypeError,ValueError):
        dataset_graph_type={}
    if len(dataset_graph_type)!=num_datasets or set(dataset_graph_type)!=set(range(num_datasets)):
        raise ValueError('Random CSV manifest lacks complete episode graph metadata for safe re-splitting.')
    training_ids,validation_ids,summary=select_two_regime_split_ids(
        task_metadata,validation_fraction=validation_fraction,split_seed=split_seed,
        stratify_by_graph_type=(config.graph_family=='mixed'))
    if summary['total_datasets']!=num_datasets:
        raise ValueError('Master CSV episode count does not match requested num_datasets.')
    m['identity']=expected
    m['data_config']=expected['data_config']
    m['prior_config']=expected['prior_config']
    m['summary']=summary
    m['training_dataset_ids']=training_ids
    m['validation_dataset_ids']=validation_ids
    m['regime_counts']=_regime_split_counts(task_metadata,training_ids,validation_ids)
    # Any existing derived train/validation files belong to the old split until
    # explicitly rewritten.  Keep only the immutable master integrity record.
    m['files']={'master':saved}
    _write_manifest_atomic(data_csv,m)
    return m


def validate_prepared_random_master(config, *, data_csv, num_datasets, data_seed,
                                    validation_fraction, split_seed, verify_hashes=False):
    """Validate the immutable master bank + manifest without requiring split files."""
    data_csv=Path(data_csv)
    expected=_identity(config,num_datasets=num_datasets,data_seed=data_seed,
                       validation_fraction=validation_fraction,split_seed=split_seed)
    try:
        m=json.loads(manifest_path(data_csv).read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:
        raise ValueError('Missing/invalid random CSV manifest; prepare data in a NEW path or use --overwrite-data-csv explicitly.') from exc
    if m.get('identity') != expected or m.get('complete') is not True:
        raise ValueError('Random CSV manifest identity/config mismatch or incomplete preparation; use a NEW path or explicit --overwrite-data-csv.')
    saved=m.get('files',{}).get('master',{})
    if not data_csv.is_file() or data_csv.stat().st_size != saved.get('size_bytes'):
        raise ValueError('CSV manifest file missing/truncated: master.')
    if not demo.csv_schema_matches(data_csv,expected_n_units=config.n_units):
        raise ValueError('CSV manifest schema/size mismatch: master.')
    if verify_hashes and file_sha256(data_csv)!=saved.get('sha256'):
        raise ValueError('CSV manifest SHA256 mismatch: master.')
    train=set(m.get('training_dataset_ids',[])); val=set(m.get('validation_dataset_ids',[]))
    if not train or not val or train & val or train|val != set(range(num_datasets)):
        raise ValueError('CSV manifest has invalid/overlapping graph splits.')
    if not _valid_regime_split(m):
        raise ValueError('CSV manifest has an invalid two-regime split.')
    return m


def validate_prepared_random_csv(config, *, data_csv, num_datasets, data_seed,
                                validation_fraction, split_seed, verify_hashes=False):
    m=validate_prepared_random_master(config,data_csv=data_csv,num_datasets=num_datasets,
        data_seed=data_seed,validation_fraction=validation_fraction,split_seed=split_seed,
        verify_hashes=verify_hashes)
    data_csv=Path(data_csv)
    train_path,val_path=demo.episode_split_paths(data_csv)
    for role,path in (('train',train_path),('validation',val_path)):
        saved=m.get('files',{}).get(role,{})
        if not path.is_file() or path.stat().st_size != saved.get('size_bytes'):
            raise ValueError(f'CSV manifest file missing/truncated: {role}.')
        if not demo.csv_schema_matches(path,expected_n_units=config.n_units):
            raise ValueError(f'CSV manifest schema/size mismatch: {role}.')
        if verify_hashes and file_sha256(path)!=saved.get('sha256'):
            raise ValueError(f'CSV manifest SHA256 mismatch: {role}.')
    return m


def _split_ids(path):
    # Rows are saved in ascending dataset-id order; don't load tensors to inspect.
    import csv
    with path.open(newline='',encoding='utf-8') as f:
        return sorted({int(row['dataset_id']) for row in csv.DictReader(f)})


def prepare_random_csv(config, *, data_csv, num_datasets, data_seed,
                       validation_fraction=.2, split_seed=0, overwrite=False,
                       generation_batch_size=64):
    data_csv=Path(data_csv)
    if generation_batch_size<=0:
        raise ValueError('generation_batch_size must be positive.')
    identity=_identity(config,num_datasets=num_datasets,data_seed=data_seed,
                       validation_fraction=validation_fraction,split_seed=split_seed)
    train_path,val_path=demo.episode_split_paths(data_csv)
    if data_csv.exists() and not overwrite:
        m=ensure_prepared_random_master_split(
            config,data_csv=data_csv,num_datasets=num_datasets,data_seed=data_seed,
            validation_fraction=validation_fraction,split_seed=split_seed,verify_hashes=True)
        try:
            validated=validate_prepared_random_csv(
                config,data_csv=data_csv,num_datasets=num_datasets,data_seed=data_seed,
                validation_fraction=validation_fraction,split_seed=split_seed,verify_hashes=True)
            print(f'Reusing fixed CSV tasks: {data_csv} (no regeneration)',flush=True)
            return dict(data_csv=data_csv,train_path=train_path,validation_path=val_path,summary=validated['summary'])
        except ValueError:
            summary=m['summary']
            _write_selected_csv_splits(data_csv,train_path,val_path,
                m['training_dataset_ids'],m['validation_dataset_ids'])
            files={'master':m['files']['master']}
            for role,path in (('train',train_path),('validation',val_path)):
                files[role]=dict(name=path.name,size_bytes=path.stat().st_size,sha256=file_sha256(path))
            m['files']=files
            _write_manifest_atomic(data_csv,m)
            print(f'Reusing fixed CSV tasks with refreshed split: {data_csv}',flush=True)
            return dict(data_csv=data_csv,train_path=train_path,validation_path=val_path,summary=summary)
    data_csv.parent.mkdir(parents=True,exist_ok=True)
    # Remove the completion marker first; crashes must never validate as complete.
    manifest_path(data_csv).unlink(missing_ok=True)
    temp=data_csv.with_name(data_csv.name+'.building')
    temp.unlink(missing_ok=True)
    meta=[]
    for start in range(0,num_datasets,generation_batch_size):
        count=min(generation_batch_size,num_datasets-start)
        b=demo.generate_batch(config,count,seed=data_seed,device=torch.device('cpu'),
            task_ids=range(start,start+count),stream=TRAIN_STREAM)
        demo.save_episode_batch_csv(b,temp,dataset_id_offset=start,append=start>0)
        for j in range(count):
            values={k:v[j].tolist() for k,v in b.items()
                    if k in ('task_id','task_seed','regime','graph_type') or
                    (k.startswith('prior_') and v[j].numel()<=3)}
            meta.append(values)
        print(f'data_generation_progress={start+count}/{num_datasets} '
              f'({100*(start+count)/num_datasets:.1f}%)',flush=True)
    temp.replace(data_csv)
    training_ids,validation_ids,summary=select_two_regime_split_ids(meta,
        validation_fraction=validation_fraction,split_seed=split_seed,
        stratify_by_graph_type=config.graph_family=='mixed')
    _write_selected_csv_splits(data_csv,train_path,val_path,training_ids,validation_ids)
    if summary['total_datasets']!=num_datasets:
        raise RuntimeError('Incomplete random CSV; completion manifest was not written.')
    files={role:dict(name=path.name,size_bytes=path.stat().st_size,sha256=file_sha256(path))
           for role,path in (('master',data_csv),('train',train_path),('validation',val_path))}
    m=dict(identity=identity,data_config=identity['data_config'],prior_config=identity['prior_config'],
        complete=True,summary=summary,files=files,task_metadata=meta,
        training_dataset_ids=training_ids,validation_dataset_ids=validation_ids,
        regime_counts=_regime_split_counts(meta,training_ids,validation_ids),
        model_inputs=['tokens: X,T_obs,Y_obs,E_obs,degree/(N-1)','adjacency','query'],
        model_outputs='original-unit GMM; no residualization and no extra losses',
        training='fixed CSV per-graph split; every epoch reuses these same observations and labels')
    _write_manifest_atomic(data_csv,m)
    return dict(data_csv=data_csv,train_path=train_path,validation_path=val_path,summary=summary)


def reject_random_csv_for_legacy(config, data_csv, *, allow_overwrite=False):
    """Do not silently train random CSV data under a legacy fixed-DGP config."""
    if config.dgp_version in ('random_lpe_v1', 'a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1') or allow_overwrite:
        return
    path = manifest_path(Path(data_csv))
    if not path.exists():
        return
    try:
        info = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Invalid CSV manifest; inspect the data rather than assume a legacy DGP.') from exc
    if info.get('data_config', {}).get('dgp_version') in ('random_lpe_v1', 'a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1'):
        raise ValueError('Versioned random_lpe_v1/A CSV requires its matching --dgp-version; legacy config cannot reuse it.')
