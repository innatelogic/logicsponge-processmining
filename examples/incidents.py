import os

import datasponge.core as ds
from examples.data.download import check_and_process_file
from datasponge.core import file

FOLDER = "data"
FILENAME = "incidents.csv"
URL = "https://data.4tu.nl/file/0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c"
DOI = "10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee"
file_path = os.path.join(FOLDER, FILENAME)
check_and_process_file(FOLDER, FILENAME, URL, DOI)


# process_miner = ProcessMiner('case_id', 'activity')


streamer = file.CSVStreamer(file_path=file_path, delay=1, poll_delay=2)
circuit = streamer * ds.AddIndex(key="index") * ds.Print()
circuit.start()
