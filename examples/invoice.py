import os

import datasponge.core as ds
from examples.data.download import check_and_process_file
from datasponge.core import file

FOLDER = "data"
FILENAME = "invoice.csv"
URL = "https://data.4tu.nl/file/443451fd-d38a-4464-88b4-0fc641552632/cd4fd2b8-6c95-47ae-aad9-dc1a085db364"
DOI = "10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1"
file_path = os.path.join(FOLDER, FILENAME)

check_and_process_file(FOLDER, FILENAME, URL, DOI)


# process_miner = ProcessMiner('case_id', 'activity')


streamer = file.CSVStreamer(file_path=file_path, delay=1, poll_delay=2)
circuit = streamer * ds.AddIndex(key="index") * ds.Print()
circuit.start()
