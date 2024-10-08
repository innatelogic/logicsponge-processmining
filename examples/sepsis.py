import os

import logicsponge.core as ls
from examples.data.download import check_and_process_file
from logicsponge.core import file

FOLDER = "data"
FILENAME = "sepsis.csv"
URL = "https://data.4tu.nl/file/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/643dccf2-985a-459e-835c-a82bce1c0339"
DOI = "10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460"
file_path = os.path.join(FOLDER, FILENAME)

check_and_process_file(FOLDER, FILENAME, URL, DOI)


class AddCaseID(ls.FunctionTerm):
    state: int

    def __init__(self):
        super().__init__()
        self.state: int = 0

    def f(self, item: ls.DataItem) -> ls.DataItem:
        if item["InfectionSuspected"] != "":
            self.state += 1

        item["case_id"] = self.state
        return item


streamer = file.CSVStreamer(file_path=file_path, delay=1, poll_delay=2)
circuit = streamer * AddCaseID() * ls.Print()
circuit.start()
