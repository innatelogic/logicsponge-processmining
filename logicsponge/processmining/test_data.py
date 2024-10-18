import os

import pandas as pd

from logicsponge.processmining.automata import PDFA
from logicsponge.processmining.data_utils import FileHandler, handle_keys, shuffle_sequences
from logicsponge.processmining.globals import ActionName, CaseId

FOLDERNAME = "data"
file_handler = FileHandler(folder=FOLDERNAME)


DATA = "file"
# DATA = "synthetic"
# DATA = "PDFA"


# ============================================================
# Data collection
# ============================================================

data_collection = {
    "incidents": {
        "url": "https://data.4tu.nl/file/0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c",
        "doi": "10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee",
        "filetype": "xes.gz",
        "target_filename": "incidents.csv",
        "target_foldername": "data",
        "case_keys": ["case:concept:name"],
        "action_keys": ["lifecycle:transition"],
        # "action_keys": ["concept:name", "lifecycle:transition"],
        "delimiter": ",",
    },
    "purchase": {
        "url": "https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828",
        "doi": "10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1",
        "filetype": "xes",
        "target_filename": "purchase.csv",
        "case_keys": ["case:Purchasing Document", "case:Item"],
        "action_keys": ["concept:name"],
        "delimiter": ",",
    },
}

# ============================================================
# File data loader
# ============================================================

if DATA == "file":
    data_name = "incidents"
    data = data_collection[data_name]
    file_path = os.path.join(FOLDERNAME, data["target_filename"])
    data["file_path"] = file_path
    file_handler.handle_file(
        file_type=data["filetype"], url=data["url"], filename=data["target_filename"], doi=data["doi"]
    )

    csv_file = pd.read_csv(data["file_path"], delimiter=data["delimiter"])

    dataset: list[tuple[CaseId, ActionName]] = [
        (
            handle_keys(data["case_keys"], row),
            handle_keys(data["action_keys"], row),
        )
        for index, row in csv_file.iterrows()
    ]


# ============================================================
# Synthetic data sets
# ============================================================

if DATA == "synthetic":
    sequences = []

    # Open the file and process it line by line
    with open(
        "/Users/bollig/innatelogic/git/circuits/innatelogic/circuits/process_mining/data/13.pautomac.train"
    ) as file:
        # Skip the first line (a header)
        next(file)

        # Process each line to extract the sequences
        for line in file:
            # Split the line into individual string numbers and convert them to integers
            numbers = list(map(int, line.split()))

            # Ignore the first element of each line
            if len(numbers) > 1:
                # As 0 is padding symbol in LSTMs, add 1 to each number in the sequence after ignoring the first element
                incremented_numbers = [num + 1 for num in numbers[1:]]

                # Store the modified sequence
                sequences.append(incremented_numbers)

    dataset = shuffle_sequences(sequences, shuffle=False)


# ============================================================
# PDFA simulation
# ============================================================


def translate_format(dataset: list[list[ActionName]]) -> list[tuple[CaseId, ActionName]]:
    translated = []
    # Enumerate over the dataset, starting from 1 for the sequence number
    for seq_number, action_list in enumerate(dataset, start=1):
        # For each action in the action_list, append a tuple (seq_number, action) to the result
        translated.extend((seq_number, action) for action in action_list)

    return translated


if DATA == "PDFA":
    pdfa = PDFA()

    pdfa.add_actions(["a", "b"])

    pdfa.create_states(3)
    pdfa.set_initial_state(0)

    pdfa.transitions[0]["a"] = 1
    pdfa.transitions[0]["b"] = 2
    pdfa.transitions[1]["a"] = 1
    pdfa.transitions[1]["b"] = 1
    pdfa.transitions[2]["a"] = 2
    pdfa.transitions[2]["b"] = 2

    pdfa.set_probs(0, [0.0, 0.5, 0.5])
    pdfa.set_probs(1, [0.1, 0.1, 0.8])
    pdfa.set_probs(2, [0.1, 0.8, 0.1])

    dataset = translate_format(pdfa.simulate(10000))


# pdfa.create_states(2)
# pdfa.set_initial_state(0)
#
# pdfa.transitions[0]["a"] = 1
# pdfa.transitions[1]["b"] = 0
#
# pdfa.set_probs(0, [0.1, 0.9, 0.0])
# pdfa.set_probs(1, [0.1, 0.0, 0.9])
