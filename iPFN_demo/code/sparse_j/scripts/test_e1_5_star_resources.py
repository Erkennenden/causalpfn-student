"""CPU/PID/peak-memory cross-checks; short probes, no PFN inference."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.e1_5_star_resources import check_resource_consistency
from scripts.e1_5_star_timing import monitor_python


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.internal = dict(pid=12, process_cpu_seconds=2., native_cpu=dict(cpu_seconds=2.),
                             memory=dict(process_pid=12, peak_working_set_bytes=100))
        self.external = dict(process_pid=12, peak_working_set_bytes=101)

    def test_windows_accounting_quantum(self):
        self.assertTrue(check_resource_consistency(2.015625, self.internal, self.external)['passed'])

    def test_launcher_cpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CPU mismatch'):
            check_resource_consistency(.03125, self.internal, self.external)

    def test_wrong_pid_is_rejected(self):
        self.external['process_pid'] = 99
        with self.assertRaisesRegex(ValueError, 'PID mismatch'):
            check_resource_consistency(2., self.internal, self.external)

    def test_wrong_task_pid_is_rejected(self):
        task = dict(process_cpu_seconds=1., resources=dict(process_pid=99, peak_working_set_bytes=50))
        with self.assertRaisesRegex(ValueError, 'Task PID'):
            check_resource_consistency(2., self.internal, self.external, [task])

    def test_incomplete_peak_is_rejected(self):
        self.external['peak_working_set_bytes'] = 50
        with self.assertRaisesRegex(ValueError, 'peak'):
            check_resource_consistency(2., self.internal, self.external)


@unittest.skipUnless(os.name == 'nt', 'Native Windows monitor')
class ProcessTests(unittest.TestCase):
    def run_probe(self):
        with tempfile.TemporaryDirectory(prefix='e1_5_resource_unit_') as temporary:
            root = Path(temporary)
            probe = root/'probe.py'
            output = root/'run'
            probe.write_text('''import json, os, sys, time
from pathlib import Path
output = Path(sys.argv[1])
output.mkdir()
buffers = []
for size in (16, 8):
    buffers.append(bytearray(size*1024**2))
    began = time.process_time()
    while time.process_time()-began < .12:
        sum(range(2000))
# The second allocation/CPU phase represents output and acceptance AFTER tasks.
(output/'probe.json').write_text(json.dumps(dict(pid=os.getpid(), cpu=time.process_time())))
''', encoding='utf-8')
            report, samples = monitor_python([str(probe), str(output)], output)
            saved = json.loads((output/'probe.json').read_text())
            self.assertEqual(report['actual_pid'], saved['pid'])
            self.assertEqual({s['process_pid'] for s in samples}, {saved['pid']})
            self.assertGreaterEqual(report['cpu']['cpu_seconds'], saved['cpu']-.03125)
            self.assertGreater(report['cpu']['cpu_seconds'], .20)
            self.assertTrue(report['resource_consistency']['passed'])
            self.assertGreater(report['peak_working_set_bytes'], 24*1024**2)
            phases = {s['phase'] for s in samples}
            self.assertIn('before_imports', phases)
            self.assertIn('after_target_and_internal_acceptance', phases)
            return report

    def test_venv_launcher_uses_actual_worker_pid(self):
        report = self.run_probe()
        if report['actual_pid'] != report['launcher_pid']:
            self.assertEqual(report['worker_parent_pid'], report['launcher_pid'])
            self.assertGreater(report['cpu']['cpu_seconds'], report['launcher_cpu_excluded']['cpu_seconds'])

    def test_direct_interpreter_pid(self):
        executable = getattr(sys, '_base_executable', sys.executable)
        with patch('sys.executable', executable):
            report = self.run_probe()
        self.assertEqual(report['actual_pid'], report['launcher_pid'])


if __name__ == '__main__':
    unittest.main()
