"""Mock clocks/resources only; no model or GPU execution."""
from pathlib import Path
from types import SimpleNamespace
import inspect,tempfile,unittest
from unittest.mock import Mock,patch
import numpy as np
from star_v2 import runtime_monitor as r,model_adapters as common
from test_task_registry import registry_fixture

class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def advance(self,n):self.now+=n

def windows(clock,rss=lambda:100):
    return lambda label:r.RuntimeWindow(label,clock=clock,cpu_clock=lambda:clock.now/10,
                                        rss_reader=rss,sample_interval=None)

class RuntimeTests(unittest.TestCase):
    def test_wall_cpu_direct_segments(self):
        c=Clock();w=windows(c)("point")
        with w:c.advance(2);w.pause();c.advance(10);w.resume();c.advance(3)
        out=w.record()
        self.assertEqual(out["wall_seconds"],5.)
        self.assertAlmostEqual(out["cpu_seconds"],.5)
        self.assertEqual([s["wall_seconds"] for s in out["segments"]],[2.,3.])
    def test_rss_current_window_peak_and_count(self):
        c=Clock();values=iter([100,400,250]);w=windows(c,lambda:next(values))("point")
        with w:c.advance(1);w.sample();c.advance(1)
        out=w.record()["rss"]
        self.assertEqual(out["peak_rss_bytes"],400)
        self.assertEqual(out["sample_count"],3)
        self.assertEqual([s["seconds"] for s in out["samples"]],[0.,1.,2.])
        self.assertIn("int(value.WorkingSetSize)",inspect.getsource(r.windows_working_set))
    def test_cpu_only_gpu_null_not_zero(self):
        c=Clock();w=windows(c)("cpu")
        with w:c.advance(1)
        gpu=w.record()["gpu"]
        for key in ("peak_allocated_bytes","peak_reserved_bytes","device"):self.assertIsNone(gpu[key])
        self.assertIn("not used",gpu["reason"])
        self.assertFalse(gpu["synchronized"])
    def test_gpu_mock_sync_and_peak(self):
        cuda=SimpleNamespace(is_available=Mock(return_value=True),current_device=Mock(return_value=0),
            get_device_name=Mock(return_value="MOCK GPU"),synchronize=Mock(),
            reset_peak_memory_stats=Mock(),max_memory_allocated=Mock(return_value=123),
            max_memory_reserved=Mock(return_value=456))
        with patch.dict("sys.modules",{"torch":SimpleNamespace(cuda=cuda)}):
            probe=r.GPUProbe(used=True);probe.begin();out=probe.end()
        self.assertEqual(cuda.synchronize.call_count,2)
        cuda.reset_peak_memory_stats.assert_called_once()
        self.assertEqual((out["allocated_bytes"],out["reserved_bytes"],out["device"]),(123,456,"MOCK GPU"))
    def test_gpu_unavailable_and_rss_failure_keep_wall(self):
        with patch.dict("sys.modules",{"torch":SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))}):
            gpu=r.GPUProbe(used=True)
        c=Clock();w=r.RuntimeWindow("x",clock=c,cpu_clock=c,rss_reader=Mock(side_effect=OSError("RSS unavailable")),
                                   sample_interval=None,gpu=gpu)
        with w:c.advance(4)
        out=w.record()
        self.assertEqual(out["wall_seconds"],4.)
        self.assertIsNone(out["rss"]["peak_rss_bytes"])
        self.assertEqual(out["rss"]["status"],"UNAVAILABLE")
        self.assertIsNone(out["gpu"]["peak_allocated_bytes"])
        self.assertIn("unavailable",out["gpu"]["reason"])
    def test_j_entire_interval_separate_and_binding_restored(self):
        _,task,_,children=registry_fixture(outputs=("ate","interval"))
        c=Clock();original=common._interval_outputs;calls=Mock()
        def interval(**kw):c.advance(5);calls();return common.ModelIntervals(np.zeros(3),np.ones(3))
        def callback():
            c.advance(2);out=common._interval_outputs({},children[0],"j",interval);c.advance(3);return out
        result,record=r.measure_pair(callback,"j",window_factory=windows(c))
        calls.assert_called_once()
        self.assertEqual(record["pair_point"]["wall_seconds"],5.)
        self.assertEqual(len(record["pair_point"]["segments"]),2)
        self.assertEqual(record["context_intervals"][0]["wall_seconds"],5.)
        self.assertEqual(record["context_intervals"][0]["context_id"],task.context_ids[0])
        self.assertIs(common._interval_outputs,original)
        self.assertEqual(result[0]["interval"],("OK",)*3)
    def test_j_provenance_normalization_is_interval_time_not_point_time(self):
        from star_v2.j_adapter import JAdapter
        _,_,_,children=registry_fixture(outputs=("ate","interval"))
        c=Clock();original=JAdapter.__dict__["_normalize_context"]
        result=SimpleNamespace(request=children[0],context_ids=children[0].context_ids)
        def normalized(value):c.advance(4);return value
        def interval(**kw):c.advance(5);return common.ModelIntervals(np.zeros(3),np.ones(3))
        def callback():
            c.advance(2);common._interval_outputs({},children[0],"j",interval)
            JAdapter._normalize_context(result);c.advance(3);return result
        with patch.object(JAdapter,"_normalize_context",side_effect=normalized) as normalize:
            _,out=r.measure_pair(callback,"j",window_factory=windows(c))
            normalize.assert_called_once_with(result)
        self.assertEqual(out["runtime_seconds"],5.)
        self.assertEqual(out["context_intervals"][0]["wall_seconds"],9.)
        self.assertEqual(len(out["context_intervals"][0]["stages"]),2)
        self.assertIs(JAdapter.__dict__["_normalize_context"],original)
    def test_interval_failure_keeps_successful_point(self):
        _,_,_,children=registry_fixture(outputs=("ate","interval"));c=Clock()
        def interval(**kw):c.advance(7);raise ValueError("synthetic interval")
        def callback():
            c.advance(2);out=common._interval_outputs({},children[0],"j",interval)[0];c.advance(1)
            return {"point":"OK","intervals":out}
        result,record=r.measure_pair(callback,"j",window_factory=windows(c))
        self.assertEqual(result["point"],"OK")
        self.assertEqual(result["intervals"]["interval"],("FAILED",)*3)
        self.assertEqual(record["pair_point"]["wall_seconds"],3.)
        self.assertEqual(record["context_intervals"][0]["wall_seconds"],7.)
    def test_exception_preserves_resources_and_wrapper(self):
        c=Clock();original=common._interval_outputs
        def fail():c.advance(3);raise RuntimeError("synthetic point")
        with self.assertRaises(r.MeasuredFailure) as caught:r.measure_pair(fail,"j",window_factory=windows(c))
        self.assertEqual(caught.exception.record["pair_point"]["wall_seconds"],3.)
        self.assertEqual(caught.exception.record["exception"]["message"],"synthetic point")
        self.assertIs(common._interval_outputs,original)
    def test_clock_gpu_errors_keep_evidence_without_fake_zero(self):
        bad=Mock(side_effect=OSError("clock unavailable"))
        with self.assertRaises(r.MeasuredFailure) as caught:
            r.measure_pair(lambda:None,"ht",window_factory=lambda label:r.RuntimeWindow(
                label,clock=bad,rss_reader=lambda:1,sample_interval=None))
        self.assertIsNone(caught.exception.record["pair_point"]["wall_seconds"])
        self.assertEqual(caught.exception.record["pair_point"]["errors"][0]["metric"],"clock")
        c=Clock();gpu=r.GPUProbe();gpu.begin=Mock(side_effect=OSError("sync failed"))
        w=r.RuntimeWindow("gpu",clock=c,cpu_clock=c,rss_reader=lambda:3,sample_interval=None,gpu=gpu)
        with w:c.advance(1)
        self.assertEqual(w.record()["wall_seconds"],1.)
        self.assertIn("failed",w.record()["gpu"]["reason"])
    def test_separate_setup_data_evaluation_and_disk(self):
        c=Clock()
        for name in ("setup","data_generation","evaluation"):
            value,out=r.measure_phase(lambda:(c.advance(2),name)[1],name,window_factory=windows(c))
            self.assertEqual((value,out["label"],out["wall_seconds"]),(name,name,2.))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"one").write_bytes(b"123")
            (Path(directory)/"two").write_bytes(b"4567")
            self.assertEqual(r.disk_occupancy(directory)["disk_occupancy_bytes"],7)

if __name__=="__main__":unittest.main()
