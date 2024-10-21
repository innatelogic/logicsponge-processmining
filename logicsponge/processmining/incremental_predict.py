from torch import nn, optim

import logicsponge.core as ls
from logicsponge.core import DataItem, dashboard
from logicsponge.processmining.algorithms_and_structures import FrequencyPrefixTree, NGram
from logicsponge.processmining.data_utils import handle_keys
from logicsponge.processmining.globals import probs_prediction
from logicsponge.processmining.models import (
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

# ============================================================
# Function Terms
# ============================================================


class ListStreamer(ls.SourceTerm):
    """
    For streaming from list.
    """

    def __init__(self, *args, list_name: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_name = list_name

    def run(self):
        for case_id, action in self.list_name:
            out = DataItem({"case_id": case_id, "action": action})
            self.output(out)
            # time.sleep(0.001)


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


class RemoveStuttering(ls.FunctionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_action = {}

    def run(self, ds_view: ls.DataStreamView):
        ds_view.next()
        item = ds_view[-1]
        case_id = item["case_id"]
        if case_id not in self.last_action or self.last_action[case_id] != item["action"]:
            self.output(item)
        self.last_action[case_id] = item["action"]


class StreamingActionPredictor(ls.FunctionTerm):
    def __init__(self, *args, strategy: StreamingMiner, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.case_ids = set()

    def run(self, ds_view: ls.DataStreamView):
        ds_view.next()
        item = ds_view[-1]
        case_id = item["case_id"]
        if case_id not in self.case_ids:
            self.strategy.update(item["case_id"], item["action"])
            self.case_ids.add(case_id)
        else:
            probs = self.strategy.case_probs(item["case_id"])
            prediction = probs_prediction(probs, self.strategy.config)
            self.strategy.update(item["case_id"], item["action"])
            out = DataItem(
                {
                    "case_id": item["case_id"],
                    "prediction": prediction,
                    "action": item["action"],
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

    def f(self, item: DataItem) -> DataItem:
        if item["prediction"] is None:
            self.missing_predictions += 1
        elif self.top_actions:
            if item["action"] in item["prediction"][1]:
                self.correct_predictions += 1
        elif item["action"] == item["prediction"][0]:
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
            }
        )


# ====================================================
# Initialize process miners
# ====================================================

config = {
    "include_stop": True,
}

fpt = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=2), config=config),
)

ngram_2 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=2), config=config),
)

ngram_4 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=4), config=config),
)

fallback = StreamingActionPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=20)),
            BasicMiner(algorithm=NGram(window_length=3)),
        ],
        config=config,
    )
)

hard_voting = StreamingActionPredictor(
    strategy=HardVoting(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=20)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
        ],
        config=config,
    )
)

soft_voting = StreamingActionPredictor(
    strategy=SoftVoting(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=20)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
        ],
        config=config,
    )
)


vocab_size = 50  # Assume an upper bound on the number of activities, or adjust dynamically
embedding_dim = 50
hidden_dim = 128
output_dim = vocab_size  # Predict the next activity
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
# DataSponge
# ====================================================

acc_list = [
    "fpt.accuracy",
    "ngram_2.accuracy",
    "ngram_4.accuracy",
    "fallback.accuracy",
    "hard_voting.accuracy",
    "soft_voting.accuracy",
    "lstm.accuracy",
]

# streamer = file.CSVStreamer(file_path=data["file_path"], delay=0, poll_delay=2)
streamer = ListStreamer(list_name=dataset, delay=0.0)

sponge = (
    streamer
    # Only for CSV files:
    # * DataPreparation(case_keys=data["case_keys"], action_keys=data["action_keys"])
    * ls.KeyFilter(keys=["case_id", "action"])
    # * RemoveStuttering()
    * AddStartSymbol()
    # * ls.Print()
    * (
        (fpt * Evaluation("fpt"))
        | (ngram_2 * Evaluation("ngram_2"))
        | (ngram_4 * Evaluation("ngram_4"))
        | (fallback * Evaluation("fallback"))
        | (hard_voting * Evaluation("hard_voting"))
        | (soft_voting * Evaluation("soft_voting"))
        | (lstm * Evaluation("lstm"))
    )
    * ls.ToSingleStream(flatten=True)
    * ls.KeyFilter(keys=acc_list)
    * ls.AddIndex(key="index")
    # * ls.Print()
    * (dashboard.Plot("Accuracy", x="index", y=acc_list))
)


sponge.start()

dashboard.show_stats(sponge)
dashboard.run()
