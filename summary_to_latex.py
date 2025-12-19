import pandas as pd
import sys
import os
from pathlib import Path

def aggregate_summaries_by_pattern(
    folder_path,
    patterns,
    models=None,
    model_name_map=None
):
    """
    Aggregate summary metrics across subfolders matching given patterns.

    Retrieve all subfolders matching given patterns, finds summary.csv files,
    aggregates metrics across matching subfolders, and returns averaged results.

    Args:
        folder_path: Path to the main folder (e.g., "final-fraction-1")
        patterns: List of patterns to match subfolder names (e.g., ["synthetic111000", "synthetic00"])
        models: List of model names to keep (if None, keeps all)
        model_name_map: Dictionary mapping model names to display names

    Returns:
        DataFrame with aggregated metrics (Model, Avg Pred Time, Avg Train Time)

    """
    folder_path = Path(folder_path)

    # Find all subfolders matching patterns
    matching_subfolders = []
    for subfolder in folder_path.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name.lower()
            if any(pattern.lower() in subfolder_name for pattern in patterns):
                matching_subfolders.append(subfolder)

    if not matching_subfolders:
        print(f"No subfolders matching patterns {patterns} found in {folder_path}")
        return pd.DataFrame()

    # Collect all summary CSV files
    summary_files = []
    for subfolder in matching_subfolders:
        for file in subfolder.iterdir():
            if file.name.endswith("summary.csv"):
                summary_files.append(file)

    if not summary_files:
        print(f"No summary.csv files found in matching subfolders")
        return pd.DataFrame()

    # Read and aggregate data
    all_data = []
    for csv_file in summary_files:
        df = pd.read_csv(csv_file)
        if models is not None:
            df = df[df["Model"].isin(models)]
        all_data.append(df)

    if not all_data:
        print("No data found after filtering models")
        return pd.DataFrame()

    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Group by model and calculate means
    aggregated = combined_df.groupby("Model")[["Pred Time", "Train Time"]].mean().reset_index()
    aggregated = aggregated.rename(columns={
        "Pred Time": "Avg Pred Time",
        "Train Time": "Avg Train Time"
    })

    # Filter to keep only requested models (if specified)
    if models is not None:
        aggregated = aggregated[aggregated["Model"].isin(models)]

    # Apply name mapping if provided
    if model_name_map:
        aggregated["Display Name"] = aggregated["Model"].map(
            lambda x: model_name_map.get(x, x) # type: ignore
        )

    return aggregated


def extract_models(
    csv_path,
    output_txt="output.txt",
    models=None,
    model_name_map=None,
    aggregated_df=None
):
    # Columns of interest in the CSV
    cols = [
        "Model",
        "Mean Accuracy (%)",
        "Top-3 (%)",
        "PP Harmo",
        "Pred Time",
        "Train Time",
    ]

    df = pd.read_csv(csv_path)

    # Keep only requested models
    if models is not None:
        df = df[df["Model"].isin(models)]

    # Keep only requested columns
    df = df[cols]

    with open(output_txt, "w") as f:
        # Write the main lines first
        for _, row in df.iterrows():
            model = row["Model"]
            display_name = model_name_map.get(model, model) if model_name_map else model

            line = (
                f"{display_name} & "
                f"{row['Mean Accuracy (%)']} & "
                f"{row['Top-3 (%)']} & "
                f"{row['PP Harmo']} & "
                f"{row['Pred Time']} & "
                f"{row['Train Time']}"
            )
            f.write(line + "\n")

        # Append aggregated data if provided
        if aggregated_df is not None and not aggregated_df.empty:
            f.write("\n")  # Add blank line separator
            for _, row in aggregated_df.iterrows():
                model = row["Model"]
                display_name = model_name_map.get(model, model) if model_name_map else model

                line = (
                    f"{display_name} & "
                    f"{row['Avg Pred Time']} & "
                    f"{row['Avg Train Time']}"
                )
                f.write(line + "\n")

    print(f"Extracted {len(df)} models to {output_txt}")


if __name__ == "__main__":
    # Example usage
    csv_path = sys.argv[1]

    list_of_window_sizes = [2, 4, 8, 16, 32]

    models_to_keep = [
        *[f"ngram_{k+1}" for k in list_of_window_sizes],
        "LSTM",
        *[f"LSTM_win{k}" for k in list_of_window_sizes],
        "transformer",
        *[f"transformer_win{k}" for k in list_of_window_sizes],
        "soft voting (2, 3, 4, 5)*",
        "soft voting (2, 3, 5, 8)*",
        "adaptive (2, 3, 4, 5) prob",
        "adaptive (2, 3, 5, 8) prob",
        "adaptive (2, 4, 6, 8, 12, 16, 24, 32) prob",
        "promotion (2, 3, 4, 5) prob",
        "promotion (2, 3, 5, 8) prob",
        "promotion (2, 4, 6, 8, 12, 16, 24, 32) prob",
    ]

    model_name_map = {
        **{f"ngram_{k+1}": f"{k+1}-gram" for k in list_of_window_sizes},
        **{f"LSTM_win{k}": f"LSTM[win:{k}]" for k in list_of_window_sizes},
        "transformer": "Transformer",
        **{f"transformer_win{k}": f"Transformer[win:{k}]" for k in list_of_window_sizes},
    }

    # Patterns to match subfolders for aggregation
    # aggregation_patterns = ["synthetic111000", "synthetic11100", "random_decision_win2", "x1x0", "x10x01"]
    aggregation_patterns = [
        "sepsis_cases", "bpi_challenge_2012", "bpi_challenge_2013", "bpi_challenge_2017", "bpi_challenge_2018"
    ]

    # Extract models from single CSV
    extract_models(
        csv_path,
        output_txt="selected_models.txt",
        models=models_to_keep,
        model_name_map=model_name_map,
    )

    # Aggregate summaries from subfolders matching patterns
    parent_folder = str(Path(csv_path).parent.parent)
    aggregated_df = aggregate_summaries_by_pattern(
        parent_folder,
        aggregation_patterns,
        models=models_to_keep,
        model_name_map=model_name_map,
    )

    # Append aggregated data to output file
    if not aggregated_df.empty:
        extract_models(
            csv_path,
            output_txt="selected_models.txt",
            models=models_to_keep,
            model_name_map=model_name_map,
            aggregated_df=aggregated_df,
        )
