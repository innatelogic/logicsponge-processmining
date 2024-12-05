import logging
import time

import torch
from torch import nn, optim

import logicsponge.core as ls
from logicsponge.core import DataItem  # , dashboard
from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.data_utils import handle_keys
from logicsponge.processmining.globals import probs_prediction
from logicsponge.processmining.models import (
    AdaptiveVoting,
    BasicMiner,
    Fallback,
    HardVoting,
    NeuralNetworkMiner,
    SoftVoting,
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


torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
            # to avoid busy waiting: if done sleep
            time.sleep(10)


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


class DataPreparation(ls.FunctionTerm):
    def __init__(self, *args, case_keys: list[str | int], action_keys: list[str | int], **kwargs):
        super().__init__(*args, **kwargs)
        self.case_keys = case_keys
        self.action_keys = action_keys

    def f(self, item: DataItem) -> DataItem:
        """
        Process the input DataItem to output a new DataItem containing only case and action keys.
        - Combines values from case_keys into a single case_id (as a tuple or single value).
        - Combines values from action_keys into a single action (as a tuple or single value).
        """
        # Construct the new DataItem with case_id and action values
        return DataItem({"case_id": handle_keys(self.case_keys, item), "action": handle_keys(self.action_keys, item)})


class StreamingActionPredictor(ls.FunctionTerm):
    def __init__(self, *args, strategy: StreamingMiner, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.case_ids = set()

    def run(self, ds_view: ls.DataStreamView):
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


class Evaluation(ls.FunctionTerm):
    def __init__(self, *args, top_actions: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.correct_predictions = 0
        self.total_predictions = 0
        self.missing_predictions = 0
        self.top_actions = top_actions
        self.latency_sum = 0
        self.latency_max = 0

    def f(self, item: DataItem) -> DataItem:
        self.latency_sum += item["latency"]
        self.latency_max = max(item["latency"], self.latency_max)

        if not item["prediction"]:
            self.missing_predictions += 1
        elif self.top_actions:
            if item["action"] in item["prediction"]["top_k_actions"]:
                self.correct_predictions += 1
        elif item["action"] == item["prediction"]["action"]:
            self.correct_predictions += 1

        self.total_predictions += 1

        accuracy = (
            self.correct_predictions / (self.total_predictions + self.missing_predictions) * 100
            if self.total_predictions > 0
            else 0
        )

        return DataItem(
            {
                "prediction": item["prediction"],
                "correct_predictions": self.correct_predictions,
                "total_predictions": self.total_predictions,
                "missing_predictions": self.missing_predictions,
                "accuracy": accuracy,
                "latency_mean": self.latency_sum / self.total_predictions,
                "latency_max": self.latency_max,
            }
        )


# ====================================================
# Initialize process miners
# ====================================================

config = {
    "include_stop": False,
}

fpt = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=FrequencyPrefixTree(), config=config),
)

bag = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=Bag(), config=config),
)

ngram_1 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=0), config=config),
)

ngram_2 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=1), config=config),
)

ngram_3 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=2), config=config),
)

ngram_4 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=3), config=config),
)

ngram_5 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=4), config=config),
)

ngram_6 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=5), config=config),
)

ngram_7 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=6), config=config),
)

ngram_8 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=7), config=config),
)

fallback = StreamingActionPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )
)

hard_voting = StreamingActionPredictor(
    strategy=HardVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2)),
            # BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
            # BasicMiner(algorithm=NGram(window_length=5)),
            BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )
)

soft_voting = StreamingActionPredictor(
    strategy=SoftVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2)),
            # BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
            # BasicMiner(algorithm=NGram(window_length=5)),
            BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )
)


adaptive_voting = StreamingActionPredictor(
    strategy=AdaptiveVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2)),
            # BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
            # BasicMiner(algorithm=NGram(window_length=5)),
            BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )
)


vocab_size = 50  # An upper bound on the number of activities
embedding_dim = 50
hidden_dim = 128
output_dim = vocab_size  # Predict the next activity
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


# ====================================================
# Sponge
# ====================================================

# Model names
models = [
    # "fpt",
    # "bag",
    "ngram_1",
    # "ngram_2",
    # "ngram_3",
    # "ngram_4",
    # "ngram_5",
    # "ngram_6",
    # "ngram_7",
    # "ngram_8",
    # "fallback",
    # "hard_voting",
    # "soft_voting",
    "adaptive_voting",
    # "lstm",
]

accuracy_list = [f"{model}.accuracy" for model in models]
latency_mean_list = [f"{model}.latency_mean" for model in models]
latency_max_list = [f"{model}.latency_max" for model in models]
all_attributes = ["index", *accuracy_list, *latency_mean_list]

# streamer = file.CSVStreamer(file_path=data["file_path"], delay=0, poll_delay=2)
streamer = ListStreamer(data_list=dataset, delay=0.0)

sponge = (
    streamer
    # Only for CSV files:
    # * DataPreparation(case_keys=data["case_keys"], action_keys=data["action_keys"])
    * ls.KeyFilter(keys=["case_id", "action"])
    * AddStartSymbol()
    * (
        # (fpt * Evaluation("fpt"))
        # | (bag * Evaluation("bag"))
        (ngram_1 * Evaluation("ngram_1"))
        # | (ngram_2 * Evaluation("ngram_2"))
        # | (ngram_3 * Evaluation("ngram_3"))
        # | (ngram_4 * Evaluation("ngram_4"))
        # | (ngram_5 * Evaluation("ngram_5"))
        # | (ngram_6 * Evaluation("ngram_6"))
        # | (ngram_7 * Evaluation("ngram_7"))
        # | (ngram_8 * Evaluation("ngram_8"))
        # | (fallback * Evaluation("fallback"))
        # | (hard_voting * Evaluation("hard_voting"))
        # | (soft_voting * Evaluation("soft_voting"))
        | (adaptive_voting * Evaluation("adaptive_voting"))
        # | (lstm * Evaluation("lstm"))
    )
    * ls.ToSingleStream(flatten=True)
    * ls.AddIndex(key="index")
    * ls.KeyFilter(keys=all_attributes)
    * ls.DataItemFilter(data_item_filter=lambda item: item["index"] % 100 == 0 or item["index"] >= len(dataset) - 10)
    * ls.Print()
    # * (dashboard.Plot("Accuracy (%)", x="index", y=accuracy_list))
    # * (dashboard.Plot("Latency Mean (ms)", x="index", y=latency_mean_list))
)


sponge.start()

# dashboard.show_stats(sponge)
# dashboard.run()
