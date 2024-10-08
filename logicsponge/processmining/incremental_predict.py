import logicsponge.core as ls
from logicsponge.core import DataItem, dashboard, file
from logicsponge.processmining.algorithms_and_structures import FrequencyPrefixTree, NGram
from logicsponge.processmining.data_utils import handle_keys
from logicsponge.processmining.models import BasicMiner, Fallback
from logicsponge.processmining.test_data import data

# ============================================================
# Function Terms
# ============================================================


class DataPreparation(ls.FunctionTerm):
    def __init__(self, *args, case_keys: list[str], action_keys: list[str], **kwargs):
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
    def __init__(self, *args, strategy, randomized: bool = True, top_k: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.randomized = randomized
        self.strategy = strategy

    def f(self, item: DataItem) -> DataItem:
        prediction = self.strategy.prediction_case(item["case_id"])
        self.strategy.update(item["case_id"], item["action"])
        return DataItem(
            {
                "case_id": item["case_id"],
                "prediction": prediction,
                "action": item["action"],
            }
        )


class Evaluation(ls.FunctionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correct_predictions = 0
        self.total_predictions = 0
        self.missing_predictions = 0

    def f(self, item: DataItem) -> DataItem:
        if item["prediction"] is None:
            self.missing_predictions += 1
        else:
            if item["prediction"][0] == item["action"]:
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

fpt_streamer = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=2)),
)

ngram_streamer = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=2)),
)

fallback_streamer = StreamingActionPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=2)),
            BasicMiner(algorithm=NGram(window_length=2, min_total_visits=2)),
        ]
    )
)


# ====================================================
# logicsponge
# ====================================================

streamer = file.CSVStreamer(file_path=data["file_path"], delay=0, poll_delay=2)
circuit = (
    streamer
    * DataPreparation(case_keys=data["case_keys"], action_keys=data["action_keys"])
    * ls.KeyFilter(keys=["case_id", "action"])
    * (
        (fpt_streamer * Evaluation("fpt"))
        | (ngram_streamer * Evaluation("ngram"))
        | (fallback_streamer * Evaluation("fallback"))
    )
    * ls.ToSingleStream(flatten=True)
    * ls.KeyFilter(keys=["fpt.accuracy", "ngram.accuracy", "fallback.accuracy"])
    * ls.AddIndex(key="index")
    # * ls.Print()
    * (dashboard.Plot("Accuracy", x="index", y=["fpt.accuracy", "ngram.accuracy", "fallback.accuracy"]))
)


circuit.start()

dashboard.show_stats(circuit)
dashboard.run()
