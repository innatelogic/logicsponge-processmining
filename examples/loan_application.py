import os

import datasponge.core as ds
from examples.data.download import check_and_process_file
from datasponge.core import file

FOLDER = "data"
FILENAME = "loan_application.csv"
URL = "https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c"
DOI = "10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f"
file_path = os.path.join(FOLDER, FILENAME)
check_and_process_file(FOLDER, FILENAME, URL, DOI)


# process_miner = ProcessMiner('case_id', 'activity')


streamer = file.CSVStreamer(file_path=file_path, delay=1, poll_delay=2)
circuit = streamer * ds.AddIndex(key="index") * ds.Print()
circuit.start()
