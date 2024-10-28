import gzip
import logging
import os
import random
import shutil
from collections import Counter
from typing import cast

import pandas as pd
import pm4py
import requests

from logicsponge.core import DataItem
from logicsponge.processmining.globals import ActionName, CaseId

logger = logging.getLogger(__name__)


# ============================================================
# Data Transformation
# ============================================================


def interleave_sequences(sequences: list[list[ActionName]], random_index=True) -> list[tuple[CaseId, ActionName]]:  # noqa: FBT002
    """
    Takes a list of sequences (list of lists) and returns a shuffled version
    while preserving the order within each sequence.
    """
    # Create a copy of sequences to avoid modifying the original list
    sequences_copy = [seq.copy() for seq in sequences]

    # Create a list of indices to track the sequences
    indices = list(range(len(sequences_copy)))

    # Resulting shuffled dataset
    shuffled_dataset = []

    # While there are still sequences with elements left
    while indices:
        chosen_index = random.choice(indices) if random_index else indices[0]  # noqa: S311

        # Pop the first element from the chosen sequence
        value = sequences_copy[chosen_index].pop(0)
        shuffled_dataset.append((str(chosen_index), str(value)))

        # If the chosen sequence is now empty, remove its index from consideration
        if not sequences_copy[chosen_index]:
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


def split_sequence_data(
    dataset: list[list[ActionName]],
    test_size: float = 0.2,
    random_shuffle: bool = False,
    seed: int | None = None,  # noqa: FBT001, FBT002
) -> tuple[list[list[ActionName]], list[list[ActionName]]]:
    dataset_copy = dataset.copy()

    if random_shuffle:
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        random.shuffle(dataset_copy)

    # Calculate the split index based on the test_size
    split_index = int(len(dataset_copy) * (1 - test_size))

    # Split the dataset into training and test sets
    train_set = dataset_copy[:split_index]
    test_set = dataset_copy[split_index:]

    return train_set, test_set


# ============================================================
# Statistics
# ============================================================


def data_statistics(data: list[list[ActionName]]) -> None:
    # Calculate total length of sequences and average length
    total_length = sum(len(lst) for lst in data)
    average_length = total_length / len(data) if data else 0

    # Flatten list of sequences and count the occurrences of each action
    flattened_data = [action for lst in data for action in lst]
    action_counter = Counter(flattened_data)

    # Extract unique actions and total number of occurrences
    unique_actions = list(action_counter.keys())
    action_occurrences = dict(action_counter)

    msg = (
        f"Number of cases: {len(data)}\n"
        f"Average length of case: {average_length}\n"
        f"Number of actions: {len(unique_actions)}\n"
        f"Number of events: {total_length}\n"
        f"Action occurrences: {action_occurrences}\n"
    )
    logger.info(msg)


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
