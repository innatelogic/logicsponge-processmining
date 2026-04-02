"""
Rigorous energy measurement for Intel RAPL (CPU) and NVIDIA NVML (GPU).

For publication-quality energy reporting on systems using:
- Intel 10th gen+ CPUs with RAPL support
- NVIDIA GPUs with NVML support

Energy measured via:
1. Intel RAPL: actual CPU/package energy from /sys/class/powercap/intel-rapl
2. NVIDIA NVML: actual GPU power draw from nvidia-smi
3. Continuous sampling at high frequency during execution phases
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RAPLDomain:
    """Represents an Intel RAPL energy domain."""

    name: str
    path: Path
    initial_energy_uj: float

    @classmethod
    def from_path(cls, path: Path) -> Optional[RAPLDomain]:
        """Create domain from RAPL sysfs path."""
        name_file = path / "name"
        energy_file = path / "energy_uj"

        if not name_file.exists() or not energy_file.exists():
            return None

        try:
            name = name_file.read_text().strip()
            energy_uj = float(energy_file.read_text().strip())
            return cls(name=name, path=path, initial_energy_uj=energy_uj)
        except (OSError, ValueError) as e:
            logger.debug(f"Failed to read RAPL domain {path}: {e}")
            return None

    def get_energy_joules(self) -> float:
        """Return energy consumed since initialization in joules."""
        try:
            current_energy_uj = float((self.path / "energy_uj").read_text().strip())
            delta_uj = current_energy_uj - self.initial_energy_uj
            # Handle wraparound on 32-bit counter (~65 seconds at 65W)
            if delta_uj < 0:
                delta_uj += 2**32
            return delta_uj / 1e6
        except (OSError, ValueError) as e:
            logger.debug(f"Failed to read RAPL energy from {self.path}: {e}")
            return 0.0


class RAPLMeter:
    """Intel RAPL energy meter (CPU package and DRAM)."""

    def __init__(self):
        """Initialize RAPL meter by scanning /sys/class/powercap."""
        self.domains: dict[str, RAPLDomain] = {}
        self._init_domains()

    def _init_domains(self) -> None:
        """Scan /sys/class/powercap/intel-rapl-* and initialize domains."""
        rapl_root = Path("/sys/class/powercap")
        if not rapl_root.exists():
            logger.warning("RAPL not available: /sys/class/powercap not found")
            return

        try:
            for item in rapl_root.glob("intel-rapl:*"):
                if item.is_dir():
                    domain = RAPLDomain.from_path(item)
                    if domain:
                        self.domains[domain.name] = domain
                        logger.debug(f"RAPL domain initialized: {domain.name}")
        except OSError as e:
            logger.warning(f"Error scanning RAPL domains: {e}")

    def is_available(self) -> bool:
        """Check if RAPL is available and initialized."""
        return len(self.domains) > 0

    def get_package_energy_joules(self) -> float:
        """Return CPU package energy in joules. Returns 0 if unavailable."""
        for name, domain in self.domains.items():
            if "package" in name.lower():
                return domain.get_energy_joules()
        return 0.0

    def get_dram_energy_joules(self) -> float:
        """Return DRAM energy in joules. Returns 0 if unavailable."""
        for name, domain in self.domains.items():
            if "dram" in name.lower():
                return domain.get_energy_joules()
        return 0.0

    def get_total_energy_joules(self) -> float:
        """Return total CPU+DRAM energy in joules."""
        return self.get_package_energy_joules() + self.get_dram_energy_joules()


class NVMLPowerMeter:
    """NVIDIA GPU power meter using nvidia-smi."""

    def __init__(self, gpu_index: int = 0):
        """Initialize NVIDIA power meter for specified GPU."""
        self.gpu_index = gpu_index
        self.has_nvidia_smi = self._check_nvidia_smi()

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def is_available(self) -> bool:
        """Check if NVIDIA GPU monitoring is available."""
        return self.has_nvidia_smi

    def get_gpu_power_watts(self) -> float:
        """Return current GPU power draw in watts."""
        if not self.has_nvidia_smi:
            return 0.0

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return 0.0

    def sample_gpu_energy_joules(
        self, duration_sec: float, interval_sec: float = 0.2
    ) -> float:
        """Sample GPU power over duration and return integrated energy."""
        if not self.has_nvidia_smi:
            return 0.0

        energy_j = 0.0
        samples = 0
        start_time = time.time()

        while time.time() - start_time < duration_sec:
            power_w = self.get_gpu_power_watts()
            if power_w > 0:
                energy_j += power_w * interval_sec
                samples += 1
            time.sleep(interval_sec)

        if samples > 0:
            logger.debug(
                f"GPU energy sampled: {energy_j:.2f} J ({samples} samples in {duration_sec:.2f}s)"
            )
        return energy_j


@dataclass
class EnergyMeasurement:
    """Results from a phase energy measurement."""

    duration_sec: float
    cpu_energy_joules: float
    dram_energy_joules: float
    gpu_energy_joules: float

    @property
    def total_energy_joules(self) -> float:
        """Total energy (CPU + DRAM + GPU)."""
        return self.cpu_energy_joules + self.dram_energy_joules + self.gpu_energy_joules

    def to_dict(self) -> dict[str, float]:
        """Return measurement as dictionary."""
        return {
            "duration_sec": self.duration_sec,
            "cpu_energy_joules": self.cpu_energy_joules,
            "dram_energy_joules": self.dram_energy_joules,
            "gpu_energy_joules": self.gpu_energy_joules,
            "total_energy_joules": self.total_energy_joules,
        }


class EnergyPhaseTracker:
    """Track energy consumption during a computational phase (train/eval)."""

    def __init__(self, phase_name: str = "phase"):
        """Initialize phase tracker."""
        self.phase_name = phase_name
        self.rapl_meter = RAPLMeter()
        self.nvml_meter = NVMLPowerMeter()
        self.start_time: Optional[float] = None
        self.start_cpu_energy_j: Optional[float] = None
        self.start_dram_energy_j: Optional[float] = None
        self.gpu_energy_j: float = 0.0
        self._sample_thread: Optional[threading.Thread] = None
        self._sampling: bool = False

        if self.rapl_meter.is_available():
            logger.info(
                f"RAPL available for {phase_name}: "
                f"Package energy tracking enabled"
            )
        else:
            logger.warning(f"RAPL not available for {phase_name}")

        if self.nvml_meter.is_available():
            logger.info(f"NVIDIA NVML available for {phase_name}: GPU energy tracking enabled")
        else:
            logger.warning(f"NVIDIA NVML not available for {phase_name}")

    def start(self) -> None:
        """Start tracking energy for this phase."""
        self.start_time = time.time()
        self.start_cpu_energy_j = self.rapl_meter.get_package_energy_joules()
        self.start_dram_energy_j = self.rapl_meter.get_dram_energy_joules()
        self.gpu_energy_j = 0.0
        self._sampling = True

        # Start GPU sampling thread
        if self.nvml_meter.is_available():
            self._sample_thread = threading.Thread(
                target=self._sample_gpu_continuous, daemon=True
            )
            self._sample_thread.start()

    def _sample_gpu_continuous(self) -> None:
        """Continuously sample GPU power in background thread."""
        interval = 0.1  # 100ms sampling interval
        while self._sampling:
            power_w = self.nvml_meter.get_gpu_power_watts()
            if power_w > 0:
                self.gpu_energy_j += power_w * interval
            time.sleep(interval)

    def stop(self) -> EnergyMeasurement:
        """Stop tracking and return energy measurement."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=2.0)

        if self.start_time is None:
            logger.warning(f"EnergyPhaseTracker.stop() called without start()")
            return EnergyMeasurement(0.0, 0.0, 0.0, 0.0)

        duration_sec = time.time() - self.start_time

        cpu_energy_j = 0.0
        dram_energy_j = 0.0

        if self.start_cpu_energy_j is not None:
            current_cpu = self.rapl_meter.get_package_energy_joules()
            cpu_energy_j = max(0.0, current_cpu - self.start_cpu_energy_j)

        if self.start_dram_energy_j is not None:
            current_dram = self.rapl_meter.get_dram_energy_joules()
            dram_energy_j = max(0.0, current_dram - self.start_dram_energy_j)

        measurement = EnergyMeasurement(
            duration_sec=duration_sec,
            cpu_energy_joules=cpu_energy_j,
            dram_energy_joules=dram_energy_j,
            gpu_energy_joules=self.gpu_energy_j,
        )

        logger.info(
            f"{self.phase_name} energy: "
            f"CPU={measurement.cpu_energy_joules:.2f}J, "
            f"DRAM={measurement.dram_energy_joules:.2f}J, "
            f"GPU={measurement.gpu_energy_joules:.2f}J, "
            f"Total={measurement.total_energy_joules:.2f}J "
            f"({duration_sec:.2f}s)"
        )

        return measurement
