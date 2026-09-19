#!/usr/bin/env python
"""Poll free RAM and kill a target PID (by name substring) if it drops below
a safety floor. Used as an external safety net around risky model loads,
since bitsandbytes/accelerate CPU offloading can spike memory unpredictably
during quantization - this guarantees the rest of the machine survives even
if that happens again, without relying on predicting *why* it happens.

    python scripts/ram_watchdog.py --floor-gb 6 --interval 1.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def free_ram_gb() -> float:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip()) / (1024 * 1024)


def kill_python_over(threshold_gb: float) -> list[int]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process python -ErrorAction SilentlyContinue | "
         "Where-Object { $_.WorkingSet64/1GB -gt %f } | Select-Object -ExpandProperty Id" % threshold_gb],
        capture_output=True, text=True, check=True,
    )
    pids = [int(x) for x in out.stdout.split()]
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    return pids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor-gb", type=float, default=6.0, help="kill if free RAM drops below this")
    parser.add_argument("--interval", type=float, default=1.5, help="poll interval, seconds")
    parser.add_argument("--max-seconds", type=float, default=3600, help="stop watching after this long")
    args = parser.parse_args()

    t0 = time.time()
    print(f"watchdog: floor={args.floor_gb}GB interval={args.interval}s", flush=True)
    while time.time() - t0 < args.max_seconds:
        free = free_ram_gb()
        if free < args.floor_gb:
            print(f"FREE RAM {free:.1f}GB < floor {args.floor_gb}GB - killing python processes over 3GB", flush=True)
            killed = kill_python_over(3.0)
            print(f"killed PIDs: {killed}", flush=True)
            sys.exit(1)
        time.sleep(args.interval)
    print("watchdog: max-seconds reached, exiting without incident", flush=True)


if __name__ == "__main__":
    main()
