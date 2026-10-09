"""Both neighbor mechanisms are switched together; evaluate each separately."""
import csv
import json
from collections import Counter
from dataclasses import asdict

import pytest
import torch
import csv_random_dgp as adapter
import train_local_network_interference as demo


def config():
    return demo.DataConfig(n_units=24, graph_family='mixed', dgp_version='random_lpe_v1')


def test_training_has_exactly_half_complete_null_and_half_both_neighbor_effects():
    batch = adapter.generate_random_batch(config(), task_ids=range(64), seed=42)
    assert Counter(batch['regime'].tolist()) == {0:32, 3:32}
    null = batch['regime'] == 0
    active = ~null
    assert (batch['prior_neighbor_coefficient'][null] == 0).all()
    assert (batch['prior_gamma'][null] == 0).all()
    assert (batch['prior_neighbor_coefficient'][active] != 0).all()
    assert (batch['prior_gamma'][active] < 0).all()
    assert torch.equal(batch['oracle_ite'][null,:,1], torch.zeros_like(batch['oracle_ite'][null,:,1]))
    assert torch.equal(batch['oracle_ite'][null,:,0], batch['oracle_ite'][null,:,2])


def test_training_and_validation_each_remain_half_after_csv_preparation_and_resplit(tmp_path):
    data=tmp_path/'data.csv'
    demo.prepare_episode_csv_files(config(), data_csv=data, num_datasets=40,
        data_seed=42, validation_fraction=.05, split_seed=7, generation_batch_size=7)
    manifest=json.loads(adapter.manifest_path(data).read_text())
    def check(m, train_count, val_count):
        regimes={r['task_id']:r['regime'] for r in m['task_metadata']}
        assert Counter(regimes[i] for i in m['training_dataset_ids']) == {0:train_count//2,3:train_count//2}
        assert Counter(regimes[i] for i in m['validation_dataset_ids']) == {0:val_count//2,3:val_count//2}
    check(manifest,38,2)
    before=adapter.file_sha256(data)
    new=adapter.ensure_prepared_random_master_split(config(), data_csv=data,
        num_datasets=40, data_seed=42, validation_fraction=.2, split_seed=42, verify_hashes=True)
    check(new,32,8)
    assert adapter.file_sha256(data)==before


def test_both_evaluators_have_half_of_each_regime_inside_every_family():
    from evaluate_matched_old_sampler import generate_matched_test_batch
    from benchmark_reference.prior import PriorConfig, generate_tasks
    cfg=config()
    matched=generate_matched_test_batch(cfg,range(64),seed=2026)
    other=generate_tasks(PriorConfig(**asdict(adapter.prior_from_data(cfg))),range(64),
        seed=2026,stream=adapter.BENCHMARK_STREAM,balanced=True)
    for batch in (matched,other):
        for kind in (2,3,4,5):
            assert Counter(batch['regime'][batch['graph_type']==kind].tolist()) == {0:8,3:8}


def test_old_independent_gate_checkpoint_is_not_silently_loaded():
    old={**asdict(config()), 'random_prior_version':'random_lpe_xnormal_v2_gh256'}
    with pytest.raises(ValueError,match='random_prior_version'):
        demo.data_config_from_mapping(old)


def test_regime_reports_use_actual_task_metadata_and_keep_common_support(tmp_path):
    from dual_sampler_reporting import write_family_reports
    # Deliberately reverse the usual regime order: metadata, not ID parity, is authoritative.
    regimes={i:(3 if i%2==0 else 0) for i in range(16)}
    manifest=dict(graphs=16,balanced=True,data_config={'graph_family':'mixed'},
        task_ids=list(range(16)),checkpoint_sha256='frozen',seed=2026)
    (tmp_path/'results.json').write_text(json.dumps(dict(manifest=manifest,
        input_hashes=[dict(task_id=i,regime=regimes[i]) for i in range(16)])))
    for level,filename in [('ite','ite_unit_results.csv'),('ate','ate_graph_results.csv')]:
        rows=[]
        for i in range(16):
            for method in ('PFN with interference','B'):
                rows.append(dict(dataset_id=i+1,unit_id=1,method=method,effect='direct',truth=0.,
                    estimate=(1. if regimes[i]==0 else 3.)*(1 if method=='PFN with interference' else 2),
                    supported=not(method=='B' and i==1)))
        with (tmp_path/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_family_reports(tmp_path,sampler='matched_old_sampler')
    for level in ('ite','ate'):
        with (tmp_path/f'{level}_by_regime.csv').open() as f:rows=list(csv.DictReader(f))
        selected={int(r['regime']):r for r in rows if r['method']=='PFN with interference' and r['graph_family']=='all'}
        assert float(selected[0]['rmse'])==1.
        assert float(selected[3]['rmse'])==3.
        assert int(selected[0]['n_graphs'])==8
        with (tmp_path/f'{level}_paired_by_regime.csv').open() as f: pairs=list(csv.DictReader(f))
        pair=next(r for r in pairs if int(r['regime'])==0 and r['graph_family']=='all')
        assert int(pair['n_paired'])==7
        assert float(pair['pfn_rmse'])==1.
        assert float(pair['method_rmse'])==2.


@pytest.mark.parametrize('seed',[12345,42])
def test_formal_102k_metadata_split_counts_are_exact_for_any_family_mix(seed):
    # No large graph bank needed: exercise the real splitter at formal scale.
    rng=torch.Generator().manual_seed(seed)
    graph_types=torch.randint(2,6,(102400,),generator=rng).tolist()
    first=torch.randint(0,2,(51200,),generator=rng).tolist()
    metadata=[dict(task_id=i,graph_type=graph_types[i],
                   regime=3*(first[i//2] if i%2==0 else 1-first[i//2])) for i in range(102400)]
    train,val,summary=adapter.select_two_regime_split_ids(metadata,
        validation_fraction=.05,split_seed=seed)
    assert summary==dict(total_datasets=102400,training_datasets=97280,validation_datasets=5120)
    assert not set(train)&set(val)
    assert set(train)|set(val)==set(range(102400))
    for ids,count in ((train,48640),(val,2560)):
        assert Counter(metadata[i]['regime'] for i in ids)=={0:count,3:count}
