import logging
import time

import torch
from torch import nn, optim

import logicsponge.core as ls
from logicsponge.core import DataItem
from logicsponge.processmining.globals import probs_prediction
from logicsponge.processmining.models_error import (
    NeuralNetworkMiner,
    StreamingMiner,
)
from logicsponge.processmining.neural_networks import LSTMModel

# from logicsponge.processmining.test_data import data
from logicsponge.processmining.test_data import dataset

logger = logging.getLogger(__name__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    device = torch.device("cpu")
    logger.info("Using cpu.")

elif torch.cuda.is_available():
    msg = f"Using cuda: {torch.cuda.get_device_name(0)}."
    logger.info(msg)
    device = torch.device("cuda")

else:
    device = torch.device("cpu")
    logger.info("Using cpu.")

# ============================================================
# Function Terms
# ============================================================


class ListStreamer(ls.SourceTerm):
    """
    For streaming from list.
    """

    def __init__(self, *args, data_list: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_list = data_list
        self.remaining = len(data_list)

    def run(self):
        if self.remaining > 0:
            for case_id, action in self.data_list:
                out = DataItem({"case_id": case_id, "action": action})
                self.output(out)
                self.remaining -= 1
            logging.info("Finished streaming.")
        else:
            time.sleep(1000)


class AddStartSymbol(ls.FunctionTerm):
    """
    For streaming from list.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_ids = set()

    def run(self, ds_view: ls.DataStreamView):
        ds_view.next()
        item = ds_view[-1]
        case_id = item["case_id"]
        if case_id not in self.case_ids:
            out = DataItem({"case_id": case_id, "action": "start"})
            self.output(out)
            self.case_ids.add(case_id)
        self.output(item)


class StreamingActionPredictor(ls.FunctionTerm):
    strategy: StreamingMiner
    case_ids: set

    def __init__(self, *args, strategy: StreamingMiner, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.case_ids = set()

    # @profile
    def run(self, ds_view: ls.DataStreamView) -> None:
        ds_view.next()
        item = ds_view[-1]
        case_id = item["case_id"]
        if case_id not in self.case_ids:  # no prediction for __start__ symbol
            self.strategy.update(item["case_id"], item["action"])
            self.case_ids.add(case_id)
        else:
            start_time = time.time()

            probs = self.strategy.case_probs(item["case_id"])
            prediction = probs_prediction(probs, self.strategy.config)
            self.strategy.update(item["case_id"], item["action"])

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # latency in milliseconds (ms)

            out = DataItem(
                {
                    "case_id": item["case_id"],
                    "prediction": prediction,  # containing predicted action
                    "action": item["action"],  # actual action
                    "latency": latency,
                }
            )
            self.output(out)


# ====================================================
# Initialize process miners
# ====================================================

config = {
    "include_stop": False,
}

vocab_size = 50  # An upper bound on the number of activities
embedding_dim = 50
hidden_dim = 128
output_dim = vocab_size  # Predict the next activity
# model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, device=device)
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

lstm = StreamingActionPredictor(
    strategy=NeuralNetworkMiner(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=8,
        config=config,
    )
)

streamer = ListStreamer(data_list=dataset, delay=0.0)
sponge = (
    streamer
    * ls.KeyFilter(keys=["case_id", "action"])
    * AddStartSymbol()
    * lstm
    * ls.ToSingleStream(flatten=True)
    * ls.AddIndex(key="index")
    * ls.KeyFilter(keys=["index"])
    * ls.DataItemFilter(data_item_filter=lambda item: item["index"] % 100 == 0 or item["index"] >= len(dataset) - 10)
    * ls.Print()
)

sponge.start()
