import gzip
import logging
import os
import random
import shutil
from typing import cast

import pandas as pd
import pm4py
import requests

from logicsponge.core import DataItem
from logicsponge.processmining.globals import STATS, ActionName, CaseId

logger = logging.getLogger(__name__)

random.seed(123)


# ============================================================
# Data Transformation
# ============================================================


def shuffle_sequences(sequences, shuffle=True):  # noqa: FBT002
    """
    Takes a list of sequences (list of lists) and returns a shuffled version
    while preserving the order within each sequence.

    Parameters:
    sequences (list of lists): The sequences to be processed.
    shuffle (bool): Whether to shuffle the sequence selection or not.

    Returns:
    list of tuples: A list containing tuples of (sequence_index, value).
    """
    # Create a list of indices to track the sequences
    indices = list(range(len(sequences)))

    # Resulting shuffled dataset
    shuffled_dataset = []

    # While there are still sequences with elements left
    while indices:
        chosen_index = random.choice(indices) if shuffle else indices[0]  # noqa: S311

        # Pop the first element from the chosen sequence
        value = sequences[chosen_index].pop(0)
        shuffled_dataset.append((str(chosen_index), str(value)))

        # If the chosen sequence is now empty, remove its index from consideration
        if not sequences[chosen_index]:
            indices.remove(chosen_index)

    return shuffled_dataset


def add_input_symbols_sequence(sequence: list[ActionName], inp: str) -> list[tuple[str, ActionName]]:
    return [(inp, elem) for elem in sequence]  # Add (inp, elem) for each element


def add_input_symbols(data: list[list[ActionName]], inp: str) -> list[list[tuple[str, ActionName]]]:
    return [add_input_symbols_sequence(sequence, inp) for sequence in data]


def add_start_to_sequences(data: list[list[ActionName]], start: ActionName) -> list[list[ActionName]]:
    """
    Appends stop symbol to each sequence in the data.
    """
    return [[start, *seq] for seq in data]


def add_stop_to_sequences(data: list[list[ActionName]], stop: ActionName) -> list[list[ActionName]]:
    """
    Appends stop symbol to each sequence in the data.
    """
    return [[*seq, stop] for seq in data]


def transform_to_seqs(data: list[tuple[CaseId, ActionName]]) -> list[list[ActionName]]:
    """
    Transforms list of tuples (case_id, action) into list of sequences.
    """
    grouped_data = {}
    for case_id, action in data:
        if case_id not in grouped_data:
            grouped_data[case_id] = []
        grouped_data[case_id].append(action)

    return list(grouped_data.values())


def split_data(
    dataset: list[tuple[CaseId, ActionName]], test_size: float = 0.2
) -> tuple[list[tuple[CaseId, ActionName]], list[tuple[CaseId, ActionName]]]:
    """
    Splits the dataset into training and test sets, keeping actions grouped by case_id.
    """
    # Create a DataFrame from the dataset
    df = pd.DataFrame(dataset, columns=["case_id", "action_name"])  # type: ignore

    # Group the data by 'case_id'
    grouped = df.groupby("case_id")

    # Collect each case's actions as a list of tuples (case_id, action)
    grouped_dataset = [(case_id, list(group["action_name"])) for case_id, group in grouped]

    # Shuffle the case groups while keeping actions together within each case
    random.shuffle(grouped_dataset)

    # Flatten the shuffled list back into the original format (case_id, action_name)
    shuffled_dataset = [(case_id, action) for case_id, actions in grouped_dataset for action in actions]

    # Create a new DataFrame if needed
    shuffled_df = pd.DataFrame(shuffled_dataset, columns=["case_id", "action_name"])  # type: ignore

    # Get the unique case_ids
    unique_case_ids = shuffled_df["case_id"].unique()

    # Determine the number of case_ids for the training set (1 - test_size)
    train_size = int((1 - test_size) * len(unique_case_ids))

    # Split the case_ids into training and test sets
    train_case_ids = unique_case_ids[:train_size]
    test_case_ids = unique_case_ids[train_size:]

    # Split the original dataset based on these case_ids
    train_set_df = shuffled_df[shuffled_df["case_id"].isin(train_case_ids)]  # type: ignore
    test_set_df = shuffled_df[shuffled_df["case_id"].isin(test_case_ids)]  # type: ignore

    # Convert the training and test sets back to list of tuples (case_id, action_name)
    train_set = cast(
        list[tuple[CaseId, ActionName]],
        [(row.case_id, row.action_name) for row in train_set_df.itertuples(index=False)],  # type: ignore
    )
    test_set = cast(
        list[tuple[CaseId, ActionName]],
        [(row.case_id, row.action_name) for row in test_set_df.itertuples(index=False)],  # type: ignore
    )

    return train_set, test_set


# ============================================================
# Statistics
# ============================================================


def data_statistics(data: list[list[ActionName]]) -> None:
    total_length = sum(len(lst) for lst in data)
    average_length = total_length / len(data) if data else 0
    msg = f"Total number of actions in test set: {total_length}\nAverage length of test sequences: {average_length}"
    logger.info(msg)


def calculate_percentages(result_percentages: dict, result: dict, strategy_name: str) -> None:
    """
    Function to calculate percentages and average log loss for each strategy on the dictionary level.
    """
    # Extract names from STATS for the relevant categories
    correct_name = STATS["correct_count"]["name"]
    wrong_name = STATS["wrong_count"]["name"]
    within_top_k_name = STATS["within_top_k_count"]["name"]
    wrong_top_k_name = STATS["wrong_top_k_count"]["name"]
    unparseable_name = STATS["unparseable_count"]["name"]
    log_loss_name = STATS["log_loss"]["name"]

    # Total valid predictions (correct + wrong)
    total_valid = result[strategy_name][correct_name] + result[strategy_name][wrong_name]

    # Total predictions including unparseable sequences
    total_all = total_valid + result[strategy_name][unparseable_name]

    # Helper function to calculate percentage
    def calculate_percentage(value, total):
        return (value / total) * 100 if total > 0 else 0

    # Calculate percentages for valid predictions
    result_percentages[strategy_name][correct_name] = calculate_percentage(
        result[strategy_name][correct_name], total_all
    )
    result_percentages[strategy_name][wrong_name] = calculate_percentage(result[strategy_name][wrong_name], total_all)
    result_percentages[strategy_name][within_top_k_name] = calculate_percentage(
        result[strategy_name][within_top_k_name], total_all
    )
    result_percentages[strategy_name][wrong_top_k_name] = calculate_percentage(
        result[strategy_name][wrong_top_k_name], total_all
    )

    # Calculate percentages for total predictions including unparseable sequences
    result_percentages[strategy_name][unparseable_name] = calculate_percentage(
        result[strategy_name][unparseable_name], total_all
    )

    # Calculate the average log loss
    if total_valid > 0:
        avg_log_loss = result[strategy_name][log_loss_name] / total_valid
        result_percentages[strategy_name][log_loss_name] = avg_log_loss
    else:
        result_percentages[strategy_name][log_loss_name] = 0


# ============================================================
# File Download
# ============================================================


class FileDownloadAbortedError(Exception):
    """Custom exception to handle file download abortion."""


class FileHandler:
    def __init__(self, folder: str):
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def download_file(self, url: str, target_filename: str) -> str:
        """
        Downloads a file from the given URL and saves it in the specified folder with the target filename.
        """
        file_path = os.path.join(self.folder, target_filename)
        msg = f"Downloading from {url}..."
        logger.info(msg)
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(file_path, "wb") as file:
            file.write(response.content)
        msg = f"Downloaded and saved to {file_path}"
        logger.info(msg)
        return file_path

    def gunzip_file(self, gz_path: str, output_filename: str) -> str:
        """
        Decompresses a .gz file and returns the path of the decompressed file.
        """
        output_path = os.path.join(self.folder, output_filename)
        msg = f"Decompressing {gz_path}..."
        logger.info(msg)
        with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        msg = f"Decompressed to {output_path}"
        logger.info(msg)
        return output_path

    def process_xes_file(self, xes_path: str, csv_filename: str) -> str:
        """
        Converts an .xes file to a CSV file.
        """
        csv_path = os.path.join(self.folder, csv_filename)
        msg = f"Processing XES file: {xes_path}..."
        logger.info(msg)
        log = pm4py.read_xes(xes_path)

        if isinstance(log, pd.DataFrame):
            df = log
        else:
            msg = f"Unexpected log type: {type(log)}. Expected a DataFrame."
            raise TypeError(msg)

        df = df.sort_values(by="time:timestamp")
        df.to_csv(csv_path, index=True)
        msg = f"Converted XES to CSV and saved to {csv_path}"
        logger.info(msg)
        return csv_path

    @staticmethod
    def clean_up(*files: str) -> None:
        """
        Deletes the specified files.
        """
        for file in files:
            if os.path.exists(file):
                os.remove(file)
                msg = f"Removed file {file}"
                logger.info(msg)

    def handle_file(self, file_type: str, url: str, filename: str, doi: str | None = None) -> str:
        """
        Main method to handle downloading and processing files based on their type.
        Handles:
        - CSV: Direct download.
        - XES: Download and process.
        - XES.GZ: Download, unzip, and process.
        """
        file_path = os.path.join(self.folder, filename)

        # Check if the final file already exists
        if os.path.exists(file_path):
            msg = f"File {file_path} already exists."
            logger.info(msg)
            return file_path

        doi_message = f"Data DOI: {doi}" if doi else ""
        user_input = (
            input(f"File {file_path} does not exist.\n{doi_message}\nDownload data from {url}? (yes/no): ")
            .strip()
            .lower()
        )

        if user_input not in ["yes", "y"]:
            msg = "File download aborted by user."
            raise FileDownloadAbortedError(msg)

        if file_type == "csv":
            # Just download the CSV file
            self.download_file(url, filename)
            return file_path

        if file_type == "xes":
            # Download and process XES
            xes_filename = filename.replace(".csv", ".xes")
            xes_file_path = self.download_file(url, xes_filename)
            self.process_xes_file(xes_file_path, filename)
            self.clean_up(xes_file_path)  # Clean up XES file after processing
            return file_path

        if file_type == "xes.gz":
            # Download, unzip, and process XES.GZ
            gz_filename = filename.replace(".csv", ".xes.gz")
            xes_filename = filename.replace(".csv", ".xes")
            gz_file_path = self.download_file(url, gz_filename)
            xes_file_path = self.gunzip_file(gz_file_path, xes_filename)
            self.process_xes_file(xes_file_path, filename)
            self.clean_up(gz_file_path, xes_file_path)  # Clean up .gz and XES files after processing
            return file_path

        msg = f"Unsupported file type: {file_type}"
        raise ValueError(msg)


def handle_keys(keys: list[str | int], row: pd.Series | DataItem) -> str | int | tuple[str | int, ...]:
    """
    Handles the case and action keys, returning either a single value or a tuple of values.
    Ensures the return type matches the expected CaseId or ActionName.
    """
    if len(keys) == 1:
        return cast(str | int, row[keys[0]])  # Return the value directly if there's only one key

    # return tuple(cast(str | int, row[key]) for key in keys)
    return ", ".join(str(cast(str | int, row[key])) for key in keys)
