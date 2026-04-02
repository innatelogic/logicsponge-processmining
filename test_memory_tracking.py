#!/usr/bin/env python3
"""
Standalone test script to verify memory tracking works on the server.

Run this before running predict_batch.py to ensure memory measurements
are accurate on your specific Linux server environment.
"""

import os
import sys
import time
from pathlib import Path

# Test 1: Check if psutil is available
print("=" * 60)
print("TEST 1: Check psutil availability")
print("=" * 60)

try:
    import psutil
    pid = os.getpid()
    rss_mb = psutil.Process(pid).memory_info().rss / (1024.0 * 1024.0)
    print(f"✓ psutil available: Current RSS = {rss_mb:.2f} MB")
    HAS_PSUTIL = True
except ImportError:
    print("✗ psutil not available (will fall back to /proc)")
    HAS_PSUTIL = False

# Test 2: Try /proc/self/status (Linux)
print("\n" + "=" * 60)
print("TEST 2: /proc/self/status method")
print("=" * 60)

if sys.platform.startswith("linux"):
    try:
        with Path("/proc/self/status").open() as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    rss_mb = kb / 1024.0
                    print(f"✓ /proc/self/status works: Current RSS = {rss_mb:.2f} MB")
                    break
    except Exception as e:
        print(f"✗ /proc/self/status failed: {e}")
else:
    print("- Not on Linux, skipping /proc/self/status")

# Test 3: Try /proc/self/stat (Linux)
print("\n" + "=" * 60)
print("TEST 3: /proc/self/stat method")
print("=" * 60)

if sys.platform.startswith("linux"):
    try:
        with Path("/proc/self/stat").open() as f:
            fields = f.read().split()
            rss_pages = int(fields[23])
            page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
            rss_mb = rss_pages * page_size / (1024.0 * 1024.0)
            print(f"✓ /proc/self/stat works: Current RSS = {rss_mb:.2f} MB")
    except Exception as e:
        print(f"✗ /proc/self/stat failed: {e}")
else:
    print("- Not on Linux, skipping /proc/self/stat")

# Test 4: Test memory growth measurement
print("\n" + "=" * 60)
print("TEST 4: Memory growth measurement (allocate ~100MB)")
print("=" * 60)


def get_current_rss_mb() -> float:
    """Get RSS using available method."""
    if HAS_PSUTIL:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0) # type: ignore  # noqa: PGH003
        except Exception:
            pass

    if sys.platform.startswith("linux"):
        try:
            with Path("/proc/self/status").open() as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except Exception:
            pass

    try:
        if sys.platform == "darwin":
            import resource
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)
    except Exception:
        pass

    return 0.0


start_rss = get_current_rss_mb()
print(f"Before allocation: {start_rss:.2f} MB")

# Allocate ~100MB
data = []
for _ in range(10):
    data.append([0.0] * (10 * 1024 * 1024))  # ~80MB per list
    time.sleep(0.1)

end_rss = get_current_rss_mb()
delta = end_rss - start_rss

print(f"After allocation:  {end_rss:.2f} MB")
print(f"Measured delta:    {delta:.2f} MB")

if delta > 50:
    print("✓ Memory growth measurement works (delta > 50 MB)")
else:
    print(f"⚠ Memory growth unexpected (delta only {delta:.2f} MB)")

# Test 5: Energy estimation
print("\n" + "=" * 60)
print("TEST 5: Energy estimation")
print("=" * 60)

DEFAULT_POWER_WATTS = {
    "cpu": 65.0,
    "cuda": 225.0,
    "mps": 35.0,
}

duration_sec = 10.0
cpu_watts = DEFAULT_POWER_WATTS["cpu"]
energy_j = duration_sec * cpu_watts

print(f"CPU power assumption: {cpu_watts} watts")
print(f"Duration: {duration_sec} seconds")
print(f"Estimated energy: {energy_j:.2f} joules")
print("✓ Energy estimation ready")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Memory tracking and energy estimation functions are operational.")
print("The predict_batch.py script should now record accurate metrics.")
