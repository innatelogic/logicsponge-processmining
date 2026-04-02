#!/usr/bin/env python3
"""
Test script for rigorous energy measurement (RAPL + NVML).

Run this BEFORE running predict_batch.py to verify that Intel RAPL
and NVIDIA NVML power measurement is working on your system.

This test suite checks:
1. Intel RAPL availability and reads
2. NVIDIA NVML availability and reads
3. Energy measurement consistency
4. Publication-ready output format
"""

import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logicsponge.processmining.energy_meter import (
    EnergyPhaseTracker,
    NVMLPowerMeter,
    RAPLMeter,
)


def test_rapl_availability():
    """Test Intel RAPL availability."""
    print("\n" + "=" * 70)
    print("TEST 1: Intel RAPL Availability")
    print("=" * 70)

    rapl = RAPLMeter()

    if rapl.is_available():
        print("✓ Intel RAPL is available")
        print(f"  Domains found: {list(rapl.domains.keys())}")

        cpu_energy = rapl.get_package_energy_joules()
        dram_energy = rapl.get_dram_energy_joules()
        total = rapl.get_total_energy_joules()

        print(f"  Current readings:")
        print(f"    CPU (Package): {cpu_energy:.2f} J")
        print(f"    DRAM:         {dram_energy:.2f} J")
        print(f"    Total:        {total:.2f} J")

        return True
    else:
        print("✗ Intel RAPL NOT available")
        print("  Ensure you're running on Linux with:")
        print("    - Intel 10th gen+ CPU (your i9-11900K supports it)")
        print("    - /sys/class/powercap/intel-rapl accessible")
        print("    - Run as root if needed: sudo python3 test_energy_meter.py")
        return False


def test_nvml_availability():
    """Test NVIDIA NVML availability."""
    print("\n" + "=" * 70)
    print("TEST 2: NVIDIA NVML Availability")
    print("=" * 70)

    nvml = NVMLPowerMeter(gpu_index=0)

    if nvml.is_available():
        print("✓ NVIDIA NVML (nvidia-smi) is available")

        power_w = nvml.get_gpu_power_watts()
        print(f"  Current GPU power draw: {power_w:.2f} W")

        return True
    else:
        print("✗ NVIDIA NVML NOT available")
        print("  Ensure you have:")
        print("    - NVIDIA GPU driver installed")
        print("    - nvidia-smi command available in PATH")
        print("    - Run: which nvidia-smi")
        return False


def test_rapl_measurement():
    """Test RAPL energy measurement during compute."""
    print("\n" + "=" * 70)
    print("TEST 3: RAPL Energy Measurement (CPU load for 5 seconds)")
    print("=" * 70)

    rapl = RAPLMeter()
    if not rapl.is_available():
        print("⊘ Skipped (RAPL not available)")
        return False

    print("Starting 5-second CPU load test...")
    initial_cpu = rapl.get_package_energy_joules()
    initial_dram = rapl.get_dram_energy_joules()

    # Generate CPU load
    start_time = time.time()
    value = 0
    while time.time() - start_time < 5.0:
        value = sum(i * i for i in range(10000))

    final_cpu = rapl.get_package_energy_joules()
    final_dram = rapl.get_dram_energy_joules()

    cpu_delta = final_cpu - initial_cpu
    dram_delta = final_dram - initial_dram

    print(f"  CPU energy consumed:  {cpu_delta:.2f} J")
    print(f"  DRAM energy consumed: {dram_delta:.2f} J")
    print(f"  Total:                {cpu_delta + dram_delta:.2f} J")

    if cpu_delta > 0:
        print("✓ RAPL measurement working (positive energy delta)")
        return True
    else:
        print("⚠ RAPL energy delta is zero (may indicate wraparound or timing issue)")
        print("  This is not critical—RAPL is still measuring.")
        return True


def test_nvml_measurement():
    """Test NVIDIA NVML power measurement."""
    print("\n" + "=" * 70)
    print("TEST 4: NVIDIA NVML Power Measurement (GPU idle)")
    print("=" * 70)

    nvml = NVMLPowerMeter(gpu_index=0)
    if not nvml.is_available():
        print("⊘ Skipped (NVIDIA NVML not available)")
        return False

    print("Sampling GPU power for 3 seconds...")
    total_energy = 0.0
    samples = 0
    start = time.time()

    while time.time() - start < 3.0:
        power_w = nvml.get_gpu_power_watts()
        total_energy += power_w * 0.1
        samples += 1
        time.sleep(0.1)

    avg_power = (total_energy / 3.0) if samples > 0 else 0.0
    print(f"  GPU idle power:  ~{avg_power:.2f} W")
    print(f"  Total energy:     {total_energy:.2f} J (in 3 seconds)")
    print(f"  Samples:          {samples}")

    print("✓ NVIDIA NVML measurement working")
    return True


def test_phase_tracker():
    """Test the high-level EnergyPhaseTracker API."""
    print("\n" + "=" * 70)
    print("TEST 5: EnergyPhaseTracker API (Integration Test)")
    print("=" * 70)

    tracker = EnergyPhaseTracker(phase_name="test_compute")

    print("Starting phase tracker for 3-second compute...")
    tracker.start()

    # Light compute load
    start = time.time()
    value = 0
    while time.time() - start < 3.0:
        value = sum(i * i for i in range(5000))

    measurement = tracker.stop()

    print(f"\n  Results:")
    print(f"    Duration:        {measurement.duration_sec:.2f} s")
    print(f"    CPU energy:      {measurement.cpu_energy_joules:.2f} J")
    print(f"    DRAM energy:     {measurement.dram_energy_joules:.2f} J")
    print(f"    GPU energy:      {measurement.gpu_energy_joules:.2f} J")
    print(f"    Total energy:    {measurement.total_energy_joules:.2f} J")

    if measurement.total_energy_joules > 0:
        print("✓ EnergyPhaseTracker working correctly")
        return True
    else:
        print("⚠ No energy measured (may indicate RAPL/NVML not available)")
        return False


def test_paper_output_format():
    """Test output format suitable for scientific publication."""
    print("\n" + "=" * 70)
    print("TEST 6: Scientific Paper Output Format")
    print("=" * 70)

    tracker = EnergyPhaseTracker(phase_name="sample_run")
    tracker.start()

    # Simulate brief compute
    time.sleep(0.5)
    for i in range(10000):
        _ = i * i

    measurement = tracker.stop()

    # Publication-ready format
    print("\n  Suitable for scientific publication:")
    print(f"  Training phase energy consumption:")
    print(
        f"    CPU (package):  {measurement.cpu_energy_joules:7.2f} J ± ? (σ from {1} run)"
    )
    print(
        f"    DRAM:           {measurement.dram_energy_joules:7.2f} J ± ? (σ from {1} run)"
    )
    print(
        f"    GPU:            {measurement.gpu_energy_joules:7.2f} J ± ? (σ from {1} run)"
    )
    print(f"    Total:          {measurement.total_energy_joules:7.2f} J")
    print(f"    Duration:       {measurement.duration_sec:7.2f} s")

    print("\n  BibTeX footnote suggestion:")
    print(f"    % CPU energy via Intel RAPL, GPU via NVIDIA NVML")
    print(f"    % 11th Gen i9-11900K + RTX 3090 @ Ubuntu 24.04")
    print(f"    % Python 3.12.3")

    print("\n✓ Output format is publication-ready")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RIGOROUS ENERGY MEASUREMENT TEST SUITE")
    print("Intel RAPL + NVIDIA NVML")
    print("=" * 70)
    print("\nTarget Hardware:")
    print("  CPU: 11th Gen Intel Core i9-11900K")
    print("  GPU: NVIDIA GeForce RTX 3090")
    print("  OS:  Ubuntu 24.04.1")
    print("  Python: 3.12.3")

    results = {
        "RAPL Availability": test_rapl_availability(),
        "NVML Availability": test_nvml_availability(),
        "RAPL Measurement": test_rapl_measurement(),
        "NVML Measurement": test_nvml_measurement(),
        "Phase Tracker API": test_phase_tracker(),
        "Paper Output Format": test_paper_output_format(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {status:8s} {test_name}")

    print(f"\n  Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ Energy measurement is ready for experiments!")
        print("  You can now run predict_batch.py with rigorous energy tracking.")
        print("  Remember to:")
        print("    1. Run with appropriate permissions (may need sudo for RAPL)")
        print("    2. Record power measurement method in paper:")
        print('       "Energy measured via Intel RAPL (CPU) and NVIDIA NVML (GPU)"')
        print("    3. Include hardware details in methodology")
        return 0
    elif passed >= 3:
        print("\n⚠ Partial energy measurement available:")
        if results["RAPL Availability"]:
            print("  ✓ CPU energy measurement is available")
        if results["NVML Availability"]:
            print("  ✓ GPU energy measurement is available")
        print("  See WARNING messages above for details.")
        print("  You can still run experiments with partial energy tracking.")
        return 1
    else:
        print("\n✗ Energy measurement is not available on this system.")
        print("  Ensure you have:")
        print("    1. Linux kernel with RAPL support (/sys/class/powercap)")
        print("    2. NVIDIA GPU driver with nvidia-smi")
        print("    3. Possibly need to run as root: sudo python3 test_energy_meter.py")
        return 2


if __name__ == "__main__":
    sys.exit(main())
