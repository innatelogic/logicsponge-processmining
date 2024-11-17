import os

import pandas as pd

from logicsponge.processmining.automata import PDFA
from logicsponge.processmining.data_utils import FileHandler, handle_keys, interleave_sequences
from logicsponge.processmining.globals import STOP, ActionName, CaseId

FOLDERNAME = "data"
file_handler = FileHandler(folder=FOLDERNAME)


# DATA = "file"
# DATA = "synthetic"
DATA = "PDFA"


# ============================================================
# Data collection
# ============================================================

data_collection = {
    "BPI_Challenge_2012": {
        "url": "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893",
        "doi": "10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f",
        "filetype": "xes.gz",
        "target_filename": "BPI_Challenge_2012.csv",
        "case_keys": ["case:concept:name"],
        "action_keys": ["concept:name"],
        "delimiter": ",",
    },
    "BPI_Challenge_2013": {
        "url": "https://data.4tu.nl/file/0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c",
        "doi": "10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee",
        "filetype": "xes.gz",
        "target_filename": "BPI_Challenge_2013.csv",
        "target_foldername": "data",
        "case_keys": ["case:concept:name"],
        "action_keys": ["lifecycle:transition"],
        "delimiter": ",",
    },
    "BPI_Challenge_2014": {
        "url": "https://data.4tu.nl/file/657fb1d6-b4c2-4adc-ba48-ed25bf313025/bd6cfa31-44f8-4542-9bad-f1f70c894728",
        "doi": "10.4121/uuid:86977bac-f874-49cf-8337-80f26bf5d2ef",
        "filetype": "csv",
        "target_filename": "BPI_Challenge_2014.csv",
        "target_foldername": "data",
        "case_keys": ["Incident ID"],
        "action_keys": ["IncidentActivity_Type"],
        "delimiter": ";",
    },
    "BPI_Challenge_2017": {
        "url": "https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c",
        "doi": "10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b",
        "filetype": "xes.gz",
        "target_filename": "BPI_Challenge_2017.csv",
        "target_foldername": "data",
        "case_keys": ["case:concept:name"],
        "action_keys": ["concept:name"],
        "delimiter": ",",
    },
    "BPI_Challenge_2018": {
        "url": "https://data.4tu.nl/file/443451fd-d38a-4464-88b4-0fc641552632/cd4fd2b8-6c95-47ae-aad9-dc1a085db364",
        "doi": "10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972",
        "filetype": "xes.gz",
        "target_filename": "BPI_Challenge_2018.csv",
        "target_foldername": "data",
        "case_keys": ["case:concept:name"],
        "action_keys": ["concept:name"],
        "delimiter": ",",
    },
    "BPI_Challenge_2019": {
        "url": "https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828",
        "doi": "10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1",
        "filetype": "xes",
        "target_filename": "BPI_Challenge_2019.csv",
        "case_keys": ["case:Purchasing Document", "case:Item"],
        "action_keys": ["concept:name"],
        "delimiter": ",",
    },
    "Sepsis_Cases": {
        "url": "https://data.4tu.nl/file/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/643dccf2-985a-459e-835c-a82bce1c0339",
        "doi": "10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460",
        "filetype": "xes.gz",
        "target_filename": "Sepsis_Cases.csv",
        "case_keys": ["case:concept:name"],
        "action_keys": ["concept:name"],
        "delimiter": ",",
    },
}

# ============================================================
# File data loader
# ============================================================

if DATA == "file":
    data_name = "Sepsis_Cases"
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
        "/Users/bollig/innatelogic/git/circuits/innatelogic/circuits/process_mining/data/10.pautomac.train"
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
                sequences.append([*incremented_numbers, STOP])

    dataset = interleave_sequences(sequences, random_index=False)


# ============================================================
# PDFA simulation
# ============================================================


# if DATA == "PDFA":
#     pdfa = PDFA()
#
#     pdfa.add_actions(["a", "b"])
#
#     pdfa.create_states(1)
#     pdfa.set_initial_state(0)
#
#     pdfa.transitions[0]["a"] = 0
#     pdfa.transitions[0]["b"] = 0
#
#     pdfa.set_probs(0, {STOP: 0.01, "a": 0.495, "b": 0.495})
#
#     dataset = interleave_sequences(pdfa.simulate(30))


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

    pdfa.set_probs(0, {STOP: 0.0, "a": 0.5, "b": 0.5})
    pdfa.set_probs(1, {STOP: 0.2, "a": 0.6, "b": 0.2})
    pdfa.set_probs(2, {STOP: 0.2, "a": 0.2, "b": 0.6})

    dataset = interleave_sequences(pdfa.simulate(1000))
