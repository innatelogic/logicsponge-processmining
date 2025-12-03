"""Script to run batch experiments on multiple datasets by invoking predict_batch.py."""

import subprocess


def run_experiments(datasets: list[str]) -> None:
    """Run predict_batch.py for each dataset in order."""
    for ds in datasets:
        print(f"\n=== Running experiment on dataset: {ds} ===")
        result = subprocess.run(
            ["python", "predict_batch.py", "--data", ds],
            capture_output=True,
            text=True, check=False
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
        "Random_Decision_win3",
        "Interruption_5",
        "Interruption_10",
        "Interruption_20",
    ]

    real_life_datasets = [
        "Sepsis_Cases",
        "Helpdesk",
        "BPI_Challenge_2012",
        "BPI_Challenge_2013",
        "BPI_Challenge_2014",
        "BPI_Challenge_2017",
        "BPI_Challenge_2018",
    ]

    dataset_list = synthetic_datasets + real_life_datasets

    run_experiments(dataset_list)
