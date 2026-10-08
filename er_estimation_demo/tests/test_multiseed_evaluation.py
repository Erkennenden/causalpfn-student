"""Test independent dataset scheduling and pooled ATE/ITE evaluation."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from causalfm_experiment.evaluation import evaluate, summarize_rows


@pytest.mark.parametrize('options', [
    {'test_seeds': []}, {'test_seeds': [2025, 2025]},
    {'test_seeds': [2025], 'test_seed': 2026},
    {'test_seeds': [2025.5]}, {'test_tasks': 0}, {'test_tasks': 1.5},
])
def test_invalid_schedule_rejected_before_checkpoint_load(tmp_path, options):
    with pytest.raises(ValueError):
        evaluate(tmp_path/'missing.pt', tmp_path/'out', device='cpu', **options)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('level', ['ATE', 'ITE'])
def test_pooled_rmse_and_unsupported_dataset_counts(level):
    rows=[dict(dataset_id=i+1, method='example', effect='total', truth=0.,
               estimate=error, supported=i!=4)
          for i,error in enumerate([0.,2.,0.,4.,100.,float('nan')])]
    result=summarize_rows(rows,'random_functions',level)[0]
    assert result['rmse']==pytest.approx(math.sqrt(5))
    assert result['rmse']!=pytest.approx(1.5)  # mean of per-dataset absolute errors
    assert result['n_supported']==4 and result['n_total']==6
    assert result['n_datasets_supported']==4 and result['n_datasets_total']==6


@pytest.mark.parametrize('selection,expected', [
    ({}, ['--test-seeds','2025','2026','2027','2028','2029']),
    ({'TEST_SEEDS':'2030 2031'}, ['--test-seeds','2030','2031']),
    ({'TEST_SEED':'2026','TEST_TASKS':'1'}, ['--test-seed','2026']),
])
def test_evaluation_only_script_passes_schedule_without_training(tmp_path, selection, expected):
    root=Path(__file__).resolve().parents[1]
    capture=tmp_path/'args.json'
    stub=tmp_path/'python-stub'
    stub.write_text(f'#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n'
                    f'Path({str(capture)!r}).write_text(json.dumps(sys.argv[1:]))\n')
    stub.chmod(0o755)
    env={k:v for k,v in os.environ.items() if k not in ('TEST_SEED','TEST_SEEDS','TEST_TASKS')}
    env.update(PYTHON=str(stub),DEVICE='cpu',**selection)
    completed=subprocess.run(['bash',str(root/'evaluate_causalfm.sh'),
                              'checkpoint with spaces.pt','output with spaces'],
                             env=env,capture_output=True,text=True)
    assert completed.returncode==0, completed.stdout+completed.stderr
    args=json.loads(capture.read_text())
    assert args[:3]==['-m','causalfm_experiment','evaluate']
    pos=args.index(expected[0])
    assert args[pos:pos+len(expected)]==expected
    assert args[args.index('--test-tasks')+1]==selection.get('TEST_TASKS','2')
    assert Path(args[args.index('--checkpoint')+1])==Path.cwd()/'checkpoint with spaces.pt'


def test_cli_rejects_ambiguous_seed_arguments(tmp_path):
    result=subprocess.run([sys.executable,'-m','causalfm_experiment','evaluate',
        '--checkpoint','missing.pt','--output',str(tmp_path/'out'),
        '--test-seed','2026','--test-seeds','2025','2026'],capture_output=True,text=True)
    assert result.returncode==2
    assert 'not allowed with argument' in result.stderr
