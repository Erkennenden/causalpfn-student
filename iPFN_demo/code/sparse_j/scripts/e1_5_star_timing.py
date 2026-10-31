"""Measure the actual interpreter, including imports, outputs and acceptance.

Explicit --mode smoke performs only two predictions; --mode pilot performs 40.
The wrapper is observed only as a parent, never added to interpreter CPU/RSS.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.e1_5_star_resources import (check_resource_consistency, cpu_seconds, memory_trend,
                                        memory_usage, physical_memory, process_handle, process_image)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def monitor_python(target, output):
    """Handshake with a stdlib bootstrap in the exact interpreter running target.

    No package imports or script code are executed in a second worker process.
    The open native handle fixes process identity even after exit/PID reuse.
    """
    output = Path(output)
    control = output.with_name(output.name + "_resources")
    if output.exists() or control.exists():
        raise FileExistsError("Output/control directory already exists; never overwrite or retry")
    control.mkdir(parents=True, exist_ok=False)
    system_memory = physical_memory()
    command = [sys.executable, "-B", str(ROOT / "scripts/e1_5_star_resource_worker.py"),
               "--control", str(control), "--", *map(str, target)]
    samples = []
    started = time.perf_counter()
    with (control / "stdout.txt").open("w", encoding="utf-8") as stdout, \
         (control / "stderr.txt").open("w", encoding="utf-8") as stderr, \
         subprocess.Popen(command, stdout=stdout, stderr=stderr) as child:
        while not (control / "ready.json").exists():
            if child.poll() is not None or time.perf_counter()-started > 30:
                raise RuntimeError(f"Worker PID handshake failed; inspect {control}")
            time.sleep(.005)
        ready = json.loads((control / "ready.json").read_text(encoding="utf-8"))
        actual_pid = ready["pid"]
        if not ((actual_pid == child.pid and ready["parent_pid"] == os.getpid()) or
                (actual_pid != child.pid and ready["parent_pid"] == child.pid)):
            (control / "start.signal").touch()
            (control / "finish.signal").touch()
            raise ValueError(f"Unexpected launcher/worker parent relationship: {ready}; launcher={child.pid}")
        with process_handle(actual_pid) as handle:
            image = process_image(handle)
            launcher_image = process_image(child._handle)

            def sample(phase):
                value = dict(wall_seconds=time.perf_counter()-started, phase=phase,
                             **memory_usage(handle), **cpu_seconds(handle))
                if value["process_pid"] != actual_pid:
                    raise ValueError("Native process handle does not match worker PID")
                samples.append(value)
                return value

            sample("before_imports")
            (control / "start.signal").touch()
            try:
                while not (control / "finished.json").exists():
                    if child.poll() is not None:
                        raise RuntimeError("Worker exited without its final resource checkpoint")
                    sample("running")
                    time.sleep(.02)
                internal = json.loads((control / "finished.json").read_text(encoding="utf-8"))
                paused = sample("after_target_and_internal_acceptance")
                check_resource_consistency(paused["cpu_seconds"], internal, paused)
            finally:
                (control / "finish.signal").touch()
            child.wait(timeout=30)
            wall = time.perf_counter()-started
            cpu = cpu_seconds(handle)
            launcher_cpu = cpu_seconds(child._handle)
            peak_note = "Native high-water mark through interpreter exit"
            peak_through_exit_verified = False
            try:
                ending = sample("after_exit")
                peak_through_exit_verified = ending["peak_working_set_bytes"] >= paused["peak_working_set_bytes"]
                if not peak_through_exit_verified:
                    peak_note = "Post-exit counters reset; high-water mark covers all target work through final checkpoint, excluding teardown"
            except OSError as exc:
                peak_note = f"Post-exit memory query unavailable ({exc}); final paused checkpoint covers all target work, excluding interpreter teardown"
            peak = max(samples, key=lambda s: s["peak_working_set_bytes"])
            # Compare CPU at the SAME boundary. Full exit CPU also includes
            # interpreter teardown, which can be substantial with torch.
            consistency = check_resource_consistency(paused["cpu_seconds"], internal, peak)
            if cpu["cpu_seconds"] + consistency["cpu_tolerance_seconds"] < paused["cpu_seconds"]:
                raise ValueError("Exit CPU total is smaller than the final checkpoint")
        if child.returncode != 0 or internal["exit_code"] != 0:
            raise RuntimeError(f"Worker failed ({child.returncode}); evidence retained in {control}")
    with (control / "resource_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    report = dict(status="resources_verified", command=command, target=list(map(str, target)),
                  launcher_pid=child.pid, actual_pid=actual_pid, worker_parent_pid=ready["parent_pid"],
                  actual_image=image, launcher_image=launcher_image,
                  cpu=cpu, launcher_cpu_excluded=launcher_cpu,
                  cpu_scope="Actual interpreter creation to exit; imports, target, internal acceptance, output and teardown included. Short-lived git subprocesses and launcher excluded.",
                  wall_seconds=wall, wall_scope="Launch to child exit, including PID handshake; parent post-process acceptance excluded",
                  final_internal_checkpoint=internal, resource_consistency=consistency,
                  checkpoint_external=paused,
                  cpu_after_checkpoint_seconds=cpu["cpu_seconds"]-paused["cpu_seconds"],
                  peak_working_set_bytes=peak["peak_working_set_bytes"], peak_note=peak_note,
                  peak_through_exit_verified=peak_through_exit_verified,
                  sample_interval_seconds=.02, samples=len(samples), physical_memory_at_start=system_memory,
                  platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count(),
                  control_directory=str(control))
    write_json(control / "process_measurement.json", report)
    return report, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This resource monitor requires Windows")
    mids, rids = (list(range(900, 904)), list(range(5))) if args.mode == "pilot" else ([900], [0])
    expected_tasks = 40 if args.mode == "pilot" else 2
    target = [str(ROOT / "scripts/e1_5_star.py"), "--output", str(args.output),
              "--mechanism-ids", *map(str, mids), "--repeat-ids", *map(str, rids), "--seed", "20261024",
              "--purpose", "timing_pilot" if args.mode == "pilot" else "smoke"]
    receipt, samples = monitor_python(target, args.output)
    receipt["monitor_command"] = sys.argv
    report_path = args.output / "runtime_measurement.json"
    write_json(report_path, receipt)
    from scripts.e1_5_star_analysis import validate_run
    acceptance = validate_run(args.output)
    write_json(args.output / "acceptance_report.json", acceptance)
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if len(tasks) != expected_tasks or manifest["mechanism_ids"] != mids or manifest["pair_repeat_ids"] != rids:
        raise ValueError("Pilot/smoke protocol count mismatch")
    checkpoint = receipt["final_internal_checkpoint"]
    receipt["resource_consistency"] = check_resource_consistency(receipt["checkpoint_external"]["cpu_seconds"], checkpoint,
            dict(process_pid=receipt["actual_pid"], peak_working_set_bytes=receipt["peak_working_set_bytes"]), tasks)
    receipt["memory_trend"] = memory_trend(tasks)
    inference = [t["inference_seconds"] for t in tasks]
    receipt["inference"] = dict(first_seconds=inference[0], subsequent_count=len(inference)-1,
                                subsequent_mean_seconds=statistics.mean(inference[1:]),
                                subsequent_median_seconds=statistics.median(inference[1:]),
                                subsequent_min_seconds=min(inference[1:]), subsequent_max_seconds=max(inference[1:]))
    generation = manifest["timing"]["mechanism_generation_seconds"]
    pair_times = manifest["timing"]["pair_seconds"]
    validation = manifest["acceptance"]["seconds"]
    fixed = max(0., receipt["wall_seconds"]-sum(generation)-sum(pair_times)-validation)
    receipt["estimate_400"] = dict(seconds=fixed+20*statistics.mean(generation)+200*statistics.mean(pair_times)+
                                   (400/expected_tasks)*validation,
                                   note="Extrapolation only; no formal tasks run. One process plus measured generation, pairs and acceptance scaling.")
    receipt["output_bytes_before_monitor_receipt"] = sum(p.stat().st_size for p in args.output.iterdir()
                                                         if p.is_file() and p != report_path)
    receipt["status"] = "accepted" if receipt["memory_trend"]["status"] != "investigate_growth" else "investigate_memory_growth"
    receipt["mode"] = args.mode
    receipt["tasks"] = len(tasks)
    write_json(report_path, receipt)
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "accepted":
        raise SystemExit("Memory growth diagnostic flagged; no retry performed")


if __name__ == "__main__":
    main()
