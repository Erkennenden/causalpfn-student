"""Native Windows process measurements; no extra package or model imports."""
import ctypes
from ctypes import wintypes
import os
import statistics
from contextlib import contextmanager


def physical_memory():
    class Status(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load_percent", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ("total_bytes", "available_bytes", "total_pagefile",
                                                   "available_pagefile", "total_virtual", "available_virtual", "extended_virtual")]
    status = Status()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Status)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {name: int(getattr(status, name)) for name in ("total_bytes", "available_bytes", "load_percent")}


def memory_usage(handle=None):
    if os.name != "nt":
        raise RuntimeError("Resource measurement currently requires Windows")

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set_bytes", "working_set_bytes", "peak_paged_pool", "paged_pool",
                "peak_nonpaged_pool", "nonpaged_pool", "pagefile_bytes", "peak_pagefile", "private_bytes")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessId.argtypes = [wintypes.HANDLE]
    kernel.GetProcessId.restype = wintypes.DWORD
    process = kernel.GetCurrentProcess() if handle is None else wintypes.HANDLE(int(handle))
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    result = {name: int(getattr(counters, name)) for name in
              ("peak_working_set_bytes", "working_set_bytes", "private_bytes")}
    result["process_pid"] = int(kernel.GetProcessId(process))
    if not result["process_pid"]:
        raise ctypes.WinError(ctypes.get_last_error())
    return result


@contextmanager
def process_handle(pid):
    """Hold the actual interpreter handle through exit; never sum launcher CPU."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000 | 0x0010 | 0x100000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield handle
    finally:
        kernel.CloseHandle(handle)


def check_resource_consistency(external_cpu, internal, external_memory, tasks=()):
    """Allow two Windows 15.625 ms accounting ticks, or 1%, whichever is larger."""
    tolerance = max(.03125, .01*max(external_cpu, internal["process_cpu_seconds"]))
    differences = [abs(external_cpu-internal["process_cpu_seconds"]),
                   abs(internal["native_cpu"]["cpu_seconds"]-internal["process_cpu_seconds"])]
    if max(differences) > tolerance:
        raise ValueError(f"External/internal CPU mismatch: {differences}; tolerance={tolerance}")
    pid = internal["pid"]
    if external_memory["process_pid"] != pid or internal["memory"]["process_pid"] != pid:
        raise ValueError("External/internal PID mismatch")
    if external_memory["peak_working_set_bytes"] < internal["memory"]["peak_working_set_bytes"]:
        raise ValueError("External peak does not cover the final worker memory checkpoint")
    for task in tasks:
        if task["resources"]["process_pid"] != pid:
            raise ValueError("Task PID differs from actual monitored interpreter")
        if task["process_cpu_seconds"] > external_cpu+tolerance:
            raise ValueError("Final CPU total does not cover task CPU")
        if task["resources"]["peak_working_set_bytes"] > external_memory["peak_working_set_bytes"]:
            raise ValueError("Final peak memory does not cover task peak")
    return dict(passed=True, actual_pid=pid, cpu_tolerance_seconds=tolerance,
                external_minus_internal_cpu_seconds=external_cpu-internal["process_cpu_seconds"],
                internal_python_minus_native_cpu_seconds=internal["process_cpu_seconds"]-internal["native_cpu"]["cpu_seconds"],
                task_pid_checks=len(tasks),
                note="CPU totals are from one actual interpreter only; launcher and worker are never added together.")


def cpu_seconds(handle):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(wintypes.HANDLE(int(handle)), ctypes.byref(created), ctypes.byref(exited),
                                  ctypes.byref(system), ctypes.byref(user)):
        raise ctypes.WinError(ctypes.get_last_error())
    convert = lambda value: ((value.dwHighDateTime << 32) + value.dwLowDateTime) / 1e7
    return dict(cpu_seconds=convert(system)+convert(user), user_seconds=convert(user), kernel_seconds=convert(system))


def process_image(handle):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    size = wintypes.DWORD(32768)
    buffer = ctypes.create_unicode_buffer(size.value)
    if not kernel.QueryFullProcessImageNameW(wintypes.HANDLE(int(handle)), 0, buffer, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.value


def memory_trend(tasks):
    """Predefined diagnostic, not proof for/against a leak in a longer run."""
    values = [t["resources"]["private_bytes"] for t in tasks[4:]]
    if len(values) < 8:
        return dict(status="insufficient_tasks", warmup_tasks_excluded=4)
    quarter = max(2, len(values)//4)
    delta = statistics.median(values[-quarter:]) - statistics.median(values[:quarter])
    slope = statistics.linear_regression(range(len(values)), values).slope
    flag = delta > 32*1024**2 and slope > .5*1024**2
    return dict(status="investigate_growth" if flag else "no_sustained_growth_detected_at_this_scale",
                warmup_tasks_excluded=4, private_bytes_median_increase=delta,
                private_bytes_slope_per_task=slope,
                criterion="Flag if BOTH median last-vs-first-quarter increase >32 MiB and slope >0.5 MiB/task",
                limitation="A short 40-task pilot cannot exclude slow or later memory growth.")
