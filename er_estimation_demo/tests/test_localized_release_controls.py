"""Exercise the actual CEPO evaluation CLI and its saved baseline configuration."""
import json
import subprocess
import sys


def test_evaluation_cli_applies_and_saves_localized_overrides(tmp_path):
    from causalfm_experiment.training import RunConfig, train
    from fixed_er_graph import load_or_create_graph
    graph=tmp_path/'graph.json'
    load_or_create_graph(graph,n_units=12,edge_probability=.4,seed=12)
    cfg=RunConfig(graph_file=str(graph),cache_dir=str(tmp_path/'cache'),device='cpu',
        train_tasks=2,validation_tasks=1,epochs=1,batch_size=1,num_layers=1,d_model=8,num_heads=2)
    checkpoint=train(cfg,tmp_path/'training')
    output=tmp_path/'evaluation'
    completed=subprocess.run([sys.executable,'-m','causalfm_experiment','evaluate',
        '--checkpoint',str(checkpoint),'--output',str(output),'--device','cpu',
        '--baseline-epochs','1','--tnet-epochs','1','--hypersci-arm-samples','16',
        '--localized-bandwidth','1.5','--localized-arm-samples','3',
        '--localized-nuisance-shrinkage','.02'],capture_output=True,text=True)
    assert completed.returncode==0, completed.stdout+completed.stderr
    result=json.loads((output/'results.json').read_text())
    assert result['test_seeds']==[2025,2026,2027,2028,2029]
    assert result['n_test_datasets']==10
    assert len(list((output/'linear_b_random_a').glob('task_*/benchmark_summary.json')))==10
    for row in result['summary']:
        assert row['n_datasets_total']==10
        assert row['n_total']==(10 if row['level']=='ATE' else 120)
    assert result['localized_options']['localized_bandwidth']==1.5
    assert result['localized_options']['localized_arm_samples']==3
    assert result['localized_options']['localized_nuisance_ridge']==.02
    report=json.loads((output/'linear_b_random_a/task_0000/benchmark_summary.json').read_text())
    diagnostic=report['localized_diagnostics']
    assert report['test_task']['test_seed']==2025
    assert report['test_task']['task_id']==0
    assert diagnostic['bandwidth']==1.5
    assert diagnostic['arm_samples']==3
    assert diagnostic['nuisance_shrinkage']==.02
    assert diagnostic['localization_geometry'][0]['near_uniform_kernel'] is True
