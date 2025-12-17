"""
Script to run batch experiments on multiple datasets by invoking predict_batch.py.

This script invokes the local `predict_batch.py` using the same Python
interpreter and a path relative to this file. That makes it robust when
executed from a different current working directory or on a server.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_experiments(datasets: list[str], data_prop: float) -> None:
    """
    Run predict_batch.py for each dataset in order.

    The call uses `sys.executable` and the full path to `predict_batch.py`
    (located in the same directory as this file) so it works regardless
    of the current working directory.
    """
    script_dir = Path(__file__).parent.resolve()
    predict_script = script_dir / "predict_batch.py"

    for ds in datasets:
        print(f"\n=== Running experiment on dataset: {ds} ===")
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(predict_script), "--data", ds, "--data_prop", str(data_prop)],
            capture_output=True,
            text=True,
            check=False,
        )

        print("--- Output ---")
        print(result.stdout)

        if result.stderr:
            print("--- Errors ---")
            print(result.stderr)

        print(f"=== Finished dataset: {ds} ===\n")


if __name__ == "__main__":
    synthetic_datasets = [
        "Synthetic111000",
        "Synthetic11100",
        "x1x0",
        "x10x01",
        "Random_Decision_win2",
        # "Random_Decision_win3",
        # "Interruption_5",
        # "Interruption_10",
        # "Interruption_20",
    ]

    real_life_datasets = [
        "Sepsis_Cases",
        # "Helpdesk",
        "BPI_Challenge_2012",
        "BPI_Challenge_2013",
        # "BPI_Challenge_2014",
        "BPI_Challenge_2017",
        "BPI_Challenge_2018",
        # "BPI_Challenge_2019",
    ]

    dataset_list = synthetic_datasets + real_life_datasets

    # parse argument --data_prop if provided
    parser = argparse.ArgumentParser(description="Run batch experiments on multiple datasets.")
    parser.add_argument(
        "--data_prop",
        type=float,
        default=1.0,
        help="Proportion of data to use for training (default: 1.0)",
    )
    args = parser.parse_args()

    run_experiments(dataset_list, data_prop=args.data_prop)
