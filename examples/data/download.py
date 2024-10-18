import gzip
import logging
import os
import shutil

import pandas as pd
import pm4py
import requests

logger = logging.getLogger(__name__)


class FileDownloadAbortedError(Exception):
    """Custom exception to handle file download abortion."""


def download_file(url: str, download_folder: str, target_filename: str) -> str:
    """
    Downloads a file from the given URL and saves it as the specified target filename.
    Returns the full path of the downloaded file.
    """
    file_path = os.path.join(download_folder, target_filename)

    msg = f"Downloading from {url}..."
    logger.info(msg)
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    with open(file_path, "wb") as file:
        file.write(response.content)

    msg = f"Downloaded and saved to {file_path}"
    logger.info(msg)
    return file_path


def gunzip_file(gz_path: str, output_path: str) -> str:
    """
    Decompresses a .gz file and returns the path of the decompressed file.
    """
    msg = f"Decompressing {gz_path}..."
    logger.info(msg)
    with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    msg = f"Decompressed to {output_path}"
    logger.info(msg)
    return output_path


def process_xes_file(xes_path: str, csv_path: str) -> None:
    """
    Converts an .xes file to a sorted CSV file and saves it as csv_path.
    """
    msg = f"Processing XES file: {xes_path}..."
    logger.info(msg)
    log = pm4py.read_xes(xes_path)

    if isinstance(log, pd.DataFrame):
        df = log
    else:
        msg = f"Unexpected log type: {type(log)}. Expected a DataFrame."
        raise TypeError(msg)

        # Sort the DataFrame by 'time:timestamp'
    df = df.sort_values(by="time:timestamp")

    # Save to CSV
    df.to_csv(csv_path, index=True)
    msg = f"Converted XES to CSV and saved to {csv_path}"
    logger.info(msg)


def check_and_process_file(folder: str, filename: str, url: str, doi: str | None = None) -> None:
    """
    Checks if the CSV file exists in the given folder. If not, downloads the .xes.gz file,
    decompresses it, processes the .xes file, and saves it as the given filename in the folder.
    Finally, it cleans up the downloaded files.
    """
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Construct the full path of the final CSV file
    csv_file_path = os.path.join(folder, filename)

    # Check if the final CSV file exists
    if os.path.exists(csv_file_path):
        msg = f"File {csv_file_path} already exists."
        logger.info(msg)
        return

    doi_message = f"Data DOI: {doi}" if doi else ""

    # Ask the user if they want to download the file, include DOI in a new line
    user_input = (
        input(f"File {csv_file_path} does not exist.\n{doi_message}\nDownload data from {url}? (yes/no): ")
        .strip()
        .lower()
    )

    if user_input not in ["yes", "y"]:
        msg = "File download aborted by user."
        raise FileDownloadAbortedError(msg)

    # Derive dynamic filenames based on the CSV filename
    gz_filename = filename.replace(".csv", ".xes.gz")
    xes_filename = filename.replace(".csv", ".xes")

    # Step 1: Download the .xes.gz file
    gz_file_path = os.path.join(folder, gz_filename)
    download_file(url, folder, gz_filename)

    # Step 2: Decompress the .gz file
    xes_file_path = os.path.join(folder, xes_filename)
    gunzip_file(gz_file_path, xes_file_path)

    # Step 3: Process the .xes file and save it as CSV
    process_xes_file(xes_file_path, csv_file_path)

    # Step 4: Clean up the .gz and .xes files
    os.remove(gz_file_path)
    os.remove(xes_file_path)
    msg = "Removed the original .gz and .xes files."
    logger.info(msg)
