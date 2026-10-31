"""PID handshake before imports; final checkpoint after all runner output/acceptance.

The interpreter waits at both boundaries so the parent can open the correct
handle and independently sample the same live process. No scientific changes.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.e1_5_star_resources import cpu_seconds, memory_usage


def publish(directory, name, payload):
    temporary = directory / (name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.rename(directory / name)


def wait_for(path, timeout=60):
    began = time.monotonic()
    while not path.exists():
        if time.monotonic()-began > timeout:
            raise TimeoutError(f"Resource handshake timeout: {path.name}")
        time.sleep(.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target = args.target[1:] if args.target[:1] == ["--"] else args.target
    if not target:
        parser.error("A target Python script is required")
    publish(args.control, "ready.json", dict(pid=os.getpid(), parent_pid=os.getppid(), executable=sys.executable))
    wait_for(args.control / "start.signal")
    exit_code = 0
    try:
        sys.argv = target
        runpy.run_path(target[0], run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if exit_code:
            traceback.print_exc()
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    publish(args.control, "finished.json", dict(pid=os.getpid(), exit_code=exit_code,
            phase="after_target_return_including_imports_outputs_and_internal_acceptance",
            process_cpu_seconds=time.process_time(), native_cpu=cpu_seconds(-1), memory=memory_usage()))
    wait_for(args.control / "finish.signal")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
