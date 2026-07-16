"""Train and evaluate an integer-valued N-gram from a CSV file."""

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from logicsponge.processmining.numeric import NumericNGram, NumericNGramConfig, NumericNGramMiner

_MIN_SEQUENCE_LENGTH = 2


def parse_args() -> argparse.Namespace:
    """Parse numeric prediction options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--value-column", default="value")
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--estimator", choices=("mean", "median", "mode", "quantile", "trimmed_mean"), default="mean")
    parser.add_argument("--metric", choices=("smape", "mae", "mape", "mse", "rmse"), default="smape")
    parser.add_argument("--min-observations", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--trim-fraction", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--no-backoff", action="store_true")
    return parser.parse_args()


def load_sequences(path: Path, case_column: str) -> list[list[dict[str, str]]]:
    """Load CSV rows and group them by case in file order."""
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if case_column not in row:
                msg = f"Missing case column: {case_column}"
                raise KeyError(msg)
            grouped[row[case_column]].append(row)
    return list(grouped.values())


def chronological_split(
    sequences: Sequence[Sequence[dict[str, str]]],
    train_fraction: float,
) -> tuple[list[list[dict[str, str]]], list[list[dict[str, str]]]]:
    """Split every case chronologically, retaining its identity across partitions."""
    if not 0.0 < train_fraction < 1.0:
        msg = "train_fraction must be between 0 and 1."
        raise ValueError(msg)
    training: list[list[dict[str, str]]] = []
    testing: list[list[dict[str, str]]] = []
    for sequence in sequences:
        split_index = max(1, min(len(sequence) - 1, int(len(sequence) * train_fraction)))
        training.append(list(sequence[:split_index]))
        testing.append(list(sequence[split_index:]))
    return training, testing


def main() -> None:
    """Train from the first part of each case and evaluate the remainder."""
    args = parse_args()
    sequences = load_sequences(args.csv_path, args.case_column)
    if any(len(sequence) < _MIN_SEQUENCE_LENGTH for sequence in sequences):
        msg = "Every case needs at least two rows for a chronological train/test split."
        raise ValueError(msg)
    training, testing = chronological_split(sequences, args.train_fraction)
    config = NumericNGramConfig(
        window_length=args.window_length,
        estimator=args.estimator,
        min_observations=args.min_observations,
        backoff=not args.no_backoff,
        quantile=args.quantile,
        trim_fraction=args.trim_fraction,
        case_id_key=args.case_column,
        value_key=args.value_column,
        coerce_values=True,
    )
    miner = NumericNGramMiner(
        NumericNGram(config),
        error_metric=args.metric,
    )
    miner.fit(training)
    result = miner.evaluate(testing, warm_start_cases=True)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
