"""
Synthetic dataset generator.

This module exposes a convenience function `generate_synthetic` which will
write a CSV into the repository `data/` folder. The generator can be used by
examples (like `predict_streaming.py`) or by the test data loader when a
synthetic CSV is missing.

The generation logic cycles through a provided pattern list. Each element of
the pattern should be an integer (e.g. 0, 1, 2...) and will be mapped to an
activity named `act_{value}`.
"""

from __future__ import annotations

import csv
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_MAX_NUM_CASES = 100
DEFAULT_TOTAL_ACTIVITIES = 200_000
DEFAULT_START_TIME = datetime(2013, 11, 7, 8, 0, 0, tzinfo=UTC)
DEFAULT_TIME_INCREMENT = timedelta(minutes=5)


def _ensure_data_dir() -> Path:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _normalize_pattern(pattern: Iterable[int] | str) -> list[int]:
    if isinstance(pattern, str):
        # Allow strings like "11100" or "1,1,1,0,0"
        if "," in pattern:
            return [int(p.strip()) for p in pattern.split(",") if p.strip()]
        return [int(c) for c in pattern.strip() if c.strip()]
    return [int(p) for p in pattern]


def generate_synthetic(  # noqa: PLR0913
    pattern: Iterable[int] | str = (1, 1, 1, 0, 0),
    save_path: Path | str | None = None,
    total_activities: int = DEFAULT_TOTAL_ACTIVITIES,
    max_num_cases: int = DEFAULT_MAX_NUM_CASES,
    start_time: datetime = DEFAULT_START_TIME,
    time_increment: timedelta = DEFAULT_TIME_INCREMENT,
) -> Path:
    """
    Generate a synthetic CSV dataset following `pattern`.

    Args:
        pattern: Sequence of integers describing the per-step activity ids.
        save_path: Optional path to write the CSV. If None, uses the pattern to
            name the file inside data/Synthetic{pattern}.csv inside the repo.
        total_activities: Total number of events to generate.
        max_num_cases: Maximum number of concurrent cases.
        start_time: Datetime for the first event.
        time_increment: Delta between consecutive global event timestamps.

    Returns:
        Path to the written CSV file.

    """
    pattern_list = _normalize_pattern(pattern)
    if not pattern_list:
        msg = "pattern must contain at least one element"
        raise ValueError(msg)

    data_dir = _ensure_data_dir()
    if save_path is None:
        pattern_str = "".join(map(str, pattern_list))
        # Name the file according to the exact selected pattern (e.g. Synthetic11100.csv)
        save_path = data_dir / f"Synthetic{pattern_str}.csv"
    save_path = Path(save_path)

    case_ids: list[str] = []
    case_counters: dict[str, int] = {}
    next_case_num = 1
    current_time = start_time

    with save_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case:concept:name", "concept:name", "time:timestamp"])

        for _ in range(int(total_activities)):
            num_cases = len(case_ids)
            # Decide whether to pick existing or create new case
            rand_val = random.random()

            if rand_val < (num_cases / max_num_cases) and num_cases > 0:
                case_id = random.choice(case_ids)
            else:
                case_id = f"case_{next_case_num}"
                case_ids.append(case_id)
                case_counters[case_id] = 0
                next_case_num += 1

            event_num = case_counters[case_id]
            cycle_position = event_num % len(pattern_list)
            activity = f"act_{pattern_list[cycle_position]}"

            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")
            writer.writerow([case_id, activity, timestamp])

            case_counters[case_id] += 1
            current_time += time_increment

    return save_path


if __name__ == "__main__":
    # Simple CLI so the script can be invoked directly for quick tests.
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic dataset CSV.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="11100",
        help="Pattern as a string (e.g. '11100' or '1,1,1,0,0').",
    )
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (optional)")
    parser.add_argument("--total", type=int, default=10000, help="Total activities to generate")
    args = parser.parse_args()

    p = generate_synthetic(pattern=args.pattern, save_path=args.out, total_activities=args.total)
    print(f"Wrote synthetic CSV to: {p}")

def generate_probabilistic_synthetic(  # noqa: PLR0913, PLR0915
    name: str,
    save_path: Path | str | None = None,
    total_activities: int = DEFAULT_TOTAL_ACTIVITIES,
    max_num_cases: int = DEFAULT_MAX_NUM_CASES,
    start_time: datetime = DEFAULT_START_TIME,
    time_increment: timedelta = DEFAULT_TIME_INCREMENT
) -> Path:
    """
    Generate a synthetic CSV dataset following `pattern`.

    Args:
        name: Name of the synthetic dataset.
        source: A Logicsponge SourceTerm that generates activity ids.
        save_path: Path to write the CSV.
        total_activities: Total number of events to generate.
        max_num_cases: Maximum number of concurrent cases.
        start_time: Datetime for the first event.
        time_increment: Delta between consecutive global event timestamps.

    Returns:
        Path to the written CSV file.

    """
    data_dir = _ensure_data_dir()
    if save_path is None:
        # Name the file according to the exact selected pattern (e.g. Synthetic11100.csv)
        save_path = data_dir / f"{name}.csv"
    save_path = Path(save_path)

    case_ids: list[str] = []
    next_case_num = 1
    current_time = start_time
    activities_added = 0


    with save_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case:concept:name", "concept:name", "time:timestamp"])

        while activities_added < int(total_activities):
            num_cases = len(case_ids)
            # Decide whether to pick existing or create new case
            rand_val = random.random()

            if rand_val < (num_cases / max_num_cases) and num_cases > 0:
                case_id = random.choice(case_ids)
            else:
                case_id = f"case_{next_case_num}"
                case_ids.append(case_id)
                next_case_num += 1

            match name:
                case "Lazy_Decrement":
                    prefix_length = random.randint(2, 5)
                    for i in range(prefix_length, 0, -1):
                        activity = f"act_{i}"
                        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")
                        writer.writerow([case_id, activity, timestamp])

                        activities_added += 1
                        current_time += time_increment
                case "Random_Decision_win2":
                    marker01 = random.choice([0, 1])
                    activity = f"act_{marker01}"
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")

                    writer.writerow([case_id, activity, timestamp])
                    activities_added += 1
                    current_time += time_increment
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")

                    writer.writerow([case_id, activity, timestamp])
                    activities_added += 1
                    current_time += time_increment
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")

                    activity = f"act_{(marker01+1)%2}"
                    writer.writerow([case_id, activity, timestamp])
                    activities_added += 1
                    current_time += time_increment
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")


                case "Random_Decision_win3":
                    marker012 = random.choice([0, 1, 2])
                    activity = f"act_{marker012}"
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")
                    writer.writerow([case_id, activity, timestamp])
                    activities_added += 1
                    current_time += time_increment
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")

                    writer.writerow([case_id, activity, timestamp])
                    activities_added += 1
                    current_time += time_increment
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")

                    remaining012 = [i for i in [0, 1, 2] if i != marker012]
                    random012 = random.choice(remaining012)
                    last012 = next(i for i in remaining012 if i != random012)
                    activity = f"act_{random012}"
                    writer.writerow([case_id, activity, timestamp])
                    activities_added += 1
                    current_time += time_increment
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")

                    activity = f"act_{last012}"
                    writer.writerow([case_id, activity, timestamp])
                    activities_added += 1
                    current_time += time_increment
                case _:
                    msg = f"Unknown synthetic dataset name: {name}"
                    raise ValueError(msg)

    return save_path


if __name__ == "__main__":
    # Simple CLI so the script can be invoked directly for quick tests.
    import argparse

    admissible_names = ["Lazy_Decrement", "Random_Decision_win2", "Random_Decision_win3"]

    for name in admissible_names:
        p = generate_probabilistic_synthetic(name=name)
        print(f"Wrote synthetic CSV to: {p}")
