"""Windowed wall/CPU/RSS/GPU observations; no profiling or model algorithms.

J point runtime is the sum of explicitly active wall-time segments, not total
minus interval time. A scoped transparent wrapper encloses the ENTIRE existing
model_adapters._interval_outputs helper; the original function/output is unchanged.
The wrapper is restored on every exit. Single-runner instrumentation only.
"""
from contextlib import contextmanager
from pathlib import Path
import ctypes
import os
import threading
import time

from . import model_adapters

_INTERVAL_LOCK = threading.RLock()


def windows_working_set():
    """Current WorkingSetSize, NOT PeakWorkingSetSize (lifetime history)."""
    if os.name != "nt":
        raise OSError("Native Windows current RSS sampler unavailable on this OS")
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    value = Counters();value.cb = ctypes.sizeof(value)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(value.WorkingSetSize)


class GPUProbe:
    def __init__(self, *, used=False):
        self.used = used
        self.torch = None
        self.reason = "GPU not used (CPU-only execution)"
        self.device = None
        if used:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                self.reason = None
                self.device = torch.cuda.get_device_name(torch.cuda.current_device())
            else:
                self.reason = "GPU requested but CUDA unavailable"

    def begin(self):
        if self.torch is not None:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()

    def end(self):
        if self.torch is None:
            return {"allocated_bytes": None, "reserved_bytes": None,
                    "device": self.device, "reason": self.reason, "synchronized": False}
        self.torch.cuda.synchronize()
        return {"allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
            "device": self.device, "reason": None, "synchronized": True}


class RuntimeWindow:
    def __init__(self, label, *, clock=time.perf_counter, cpu_clock=time.process_time,
                 rss_reader=windows_working_set, sample_interval=.01, gpu=None):
        if sample_interval is not None and sample_interval <= 0:
            raise ValueError("Positive sampling interval required")
        self.label=label;self.clock=clock;self.cpu_clock=cpu_clock
        self.rss_reader=rss_reader;self.sample_interval=sample_interval
        self.gpu=GPUProbe() if gpu is None else gpu
        self.segments=[];self.samples=[];self.errors=[];self.gpu_samples=[]
        self.active=False;self.started=False;self.closed=False
        self._lock=threading.RLock();self._stop=threading.Event();self._thread=None

    def sample(self):
        with self._lock:
            if not self.active:return
            try:
                value=self.rss_reader()
                if type(value) is not int or value < 0:raise ValueError("Invalid current RSS")
                self.samples.append({"seconds":self.clock(),"rss_bytes":value,
                                     "segment":len(self.segments)})
            except Exception as exc:
                self.errors.append({"metric":"rss","type":type(exc).__name__,"message":str(exc)})

    def _poll(self):
        while not self._stop.wait(self.sample_interval):
            self.sample()

    def resume(self):
        with self._lock:
            if self.active or self.closed:raise RuntimeError("Invalid runtime resume")
            try:self.gpu.begin()  # synchronization precedes the active wall-time start
            except Exception as exc:
                self.errors.append({"metric":"gpu","type":type(exc).__name__,"message":str(exc)})
                self.gpu=GPUProbe()
                self.gpu.reason="GPU synchronization/measurement failed: "+str(exc)
            self._wall=self.clock();self._cpu=self.cpu_clock();self.active=True
            self.sample()

    def pause(self):
        with self._lock:
            if not self.active:raise RuntimeError("Runtime window is not active")
            # Include GPU completion within the active window before ending it.
            try:self.gpu_samples.append(self.gpu.end())
            except Exception as exc:
                self.errors.append({"metric":"gpu","type":type(exc).__name__,"message":str(exc)})
                self.gpu.reason="GPU synchronization/measurement failed: "+str(exc)
            self.sample()
            end=self.clock();cpu_end=self.cpu_clock()
            self.segments.append({"start":self._wall,"end":end,"wall_seconds":end-self._wall,
                                  "cpu_seconds":cpu_end-self._cpu})
            self.active=False

    def __enter__(self):
        if self.started:raise RuntimeError("Runtime window is single-use")
        self.started=True
        try:self.resume()
        except Exception as exc:
            self.closed=True
            self.errors.append({"metric":"clock","type":type(exc).__name__,"message":str(exc)})
            raise
        if self.sample_interval is not None:
            self._thread=threading.Thread(target=self._poll,daemon=True)
            self._thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if kind is not None:
                self.errors.append({"metric":"execution","type":kind.__name__,"message":str(value)})
            if self.active:self.pause()
        finally:
            self.closed=True;self._stop.set()
            if self._thread is not None:self._thread.join()
        return False

    def record(self):
        if not self.closed:raise RuntimeError("Close runtime window before recording")
        good=[g for g in self.gpu_samples if g["allocated_bytes"] is not None]
        return {"label":self.label,"wall_seconds":sum(s["wall_seconds"] for s in self.segments) if self.segments else None,
            "cpu_seconds":sum(s["cpu_seconds"] for s in self.segments) if self.segments else None,
            "cpu_scope":"process_time over active segments; no CPU profiling",
            "segments":self.segments,"measurement_window":"active_segments_only",
            "rss":{"peak_rss_bytes":max((s["rss_bytes"] for s in self.samples),default=None),
                "status":"OBSERVED" if self.samples else "UNAVAILABLE",
                "reason":None if self.samples else "No successful current-RSS samples",
                "sampling_interval_seconds":self.sample_interval,"sample_count":len(self.samples),
                "samples":self.samples,"definition":"maximum sampled current Windows WorkingSetSize"},
            "gpu":{"peak_allocated_bytes":max((g["allocated_bytes"] for g in good),default=None),
                "peak_reserved_bytes":max((g["reserved_bytes"] for g in good),default=None),
                "device":self.gpu.device,"reason":None if good else self.gpu.reason,
                "synchronized":bool(good) and all(g["synchronized"] for g in good)},
            "errors":self.errors}


@contextmanager
def isolate_j_intervals(point_window, interval_records, *, window_factory=RuntimeWindow):
    """Wrap the complete frozen interval phase, including validation/provenance.

    This temporarily wraps a function binding, not a source file or algorithm.
    It delegates exactly once and restores the original binding in finally.
    No other runner/thread may use model_adapters during this scoped operation.
    """
    with _INTERVAL_LOCK:
        from .j_adapter import JAdapter
        original=model_adapters._interval_outputs
        normalize_descriptor=JAdapter.__dict__["_normalize_context"]
        original_normalize=JAdapter._normalize_context
        def measured_normalize(result):
            if "interval" not in result.request.requested_outputs:
                return original_normalize(result)
            point_window.pause()
            window=window_factory("j_interval_provenance:"+result.context_ids[0])
            try:
                with window:return original_normalize(result)
            finally:
                stage=window.record()
                found=next((item for item in interval_records
                            if item["context_id"]==result.context_ids[0]),None)
                if found is None:
                    interval_records.append({"context_id":result.context_ids[0],
                        **stage,"posterior_call_entered":False,
                        "stages":({"kind":"interval_provenance_normalization",**stage},)})
                else:
                    previous={k:v for k,v in found.items() if k not in ("context_id","stages")}
                    found["stages"]=({"kind":"posterior_interval_and_common_mapping",**previous},
                                     {"kind":"interval_provenance_normalization",**stage})
                    found["wall_seconds"]+=stage["wall_seconds"]
                    found["cpu_seconds"]+=stage["cpu_seconds"]
                    found["posterior_call_entered"]=True
                    rss_stages=(previous["rss"],stage["rss"])
                    peak=max((item["peak_rss_bytes"] for item in rss_stages
                              if item["peak_rss_bytes"] is not None),default=None)
                    found["rss"]={"definition":"maximum sampled current RSS across interval stages",
                        "peak_rss_bytes":peak,"status":"OBSERVED" if peak is not None else "UNAVAILABLE",
                        "reason":None if peak is not None else "No successful interval-stage RSS samples",
                        "sampling_interval_seconds":previous["rss"]["sampling_interval_seconds"],
                        "sample_count":sum(item["sample_count"] for item in rss_stages),
                        "samples":sum((item["samples"] for item in rss_stages),[]),"stages":rss_stages}
                    gpu_stages=(previous["gpu"],stage["gpu"])
                    valid_gpu=[item for item in gpu_stages if item["peak_allocated_bytes"] is not None]
                    found["gpu"]={"peak_allocated_bytes":max((item["peak_allocated_bytes"]
                        for item in valid_gpu),default=None),"peak_reserved_bytes":max((
                        item["peak_reserved_bytes"] for item in valid_gpu),default=None),
                        "device":previous["gpu"]["device"],
                        "reason":None if valid_gpu else previous["gpu"]["reason"],
                        "synchronized":bool(valid_gpu) and all(item["synchronized"] for item in valid_gpu)}
                point_window.resume()
        def measured(payload, request, method_id, interval_predictor):
            if method_id != "j" or "interval" not in request.requested_outputs:
                return original(payload,request,method_id,interval_predictor)
            point_window.pause()
            interval=window_factory("j_context_interval:"+request.context_ids[0])
            try:
                with interval:
                    return original(payload,request,method_id,interval_predictor)
            finally:
                interval_records.append({"context_id":request.context_ids[0],
                                         **interval.record()})
                point_window.resume()
        model_adapters._interval_outputs=measured
        JAdapter._normalize_context=staticmethod(measured_normalize)
        try:yield
        finally:
            model_adapters._interval_outputs=original
            JAdapter._normalize_context=normalize_descriptor


def measure_pair(callback, method_id, *, window_factory=RuntimeWindow):
    """Model setup MUST precede this call; callback is existing pair adapter."""
    point=window_factory("pair_ate:"+method_id);intervals=[];error=None;result=None
    try:
        with point:
            if method_id == "j":
                with isolate_j_intervals(point,intervals,window_factory=window_factory):
                    result=callback()
            else:
                result=callback()
    except Exception as exc:
        error=exc
    point_record=point.record()
    record={"schema_version":1,"method_id":method_id,
        "runtime_seconds":point_record["wall_seconds"],
        "peak_rss_bytes":point_record["rss"]["peak_rss_bytes"],
        "interval_runtime_seconds":tuple({"context_id":item["context_id"],"seconds":item["wall_seconds"]}
                                         for item in intervals),
        "boundary":"adapter+features+model calls+pair summary; excludes setup/data/evaluation/disk",
        "pair_point":point_record,"context_intervals":intervals,
        "exception":None if error is None else {"type":type(error).__name__,"message":str(error)}}
    if error is not None:
        raise MeasuredFailure(error,record) from error
    return result,record


class MeasuredFailure(RuntimeError):
    def __init__(self,original,record):
        self.original=original;self.record=record
        super().__init__(str(original))


def measure_phase(callback,label,*,window_factory=RuntimeWindow):
    window=window_factory(label)
    try:
        with window:result=callback()
    except Exception as exc:
        raise MeasuredFailure(exc,window.record()) from exc
    return result,window.record()


def disk_occupancy(directory):
    root=Path(directory).resolve()
    total=0
    for path in root.rglob("*"):
        if path.is_symlink() or getattr(path,"is_junction",lambda:False)():
            raise ValueError("No reparse paths in disk accounting")
        if path.is_file():total+=path.stat().st_size
    return {"disk_occupancy_bytes":total,"scope":str(root),
            "definition":"sum of current regular output file sizes; not allocated disk clusters"}
