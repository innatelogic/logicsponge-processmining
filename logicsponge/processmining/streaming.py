import time
from collections.abc import Iterator

import logicsponge.core as ls
from logicsponge.core import DataItem  # , dashboard
from logicsponge.processmining.data_utils import handle_keys
from logicsponge.processmining.models import (
    StreamingMiner,
)
from logicsponge.processmining.types import ActivityName, Event
from logicsponge.processmining.utils import probs_prediction


class IteratorStreamer(ls.SourceTerm):
    """
    For streaming from iterator.
    """

    def __init__(self, *args, data_iterator: Iterator, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_iterator = data_iterator

    def run(self):
        for event in self.data_iterator:
            case_id = event["case_id"]
            activity = event["activity"]

            out = DataItem({"case_id": case_id, "activity": activity})
            self.output(out)

        # repeatedly sleep if done
        time.sleep(10)


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
            for event in self.data_list:
                case_id = event["case_id"]
                activity = event["activity"]

                out = DataItem({"case_id": case_id, "activity": activity})
                self.output(out)
                self.remaining -= 1
        else:
            # to avoid busy waiting: if done sleep
            time.sleep(10)


class AddStartSymbol(ls.FunctionTerm):
    """
    For streaming from list.
    """

    def __init__(self, *args, start_symbol: ActivityName, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_ids = set()
        self.start_symbol = start_symbol

    def run(self, ds_view: ls.DataStreamView):
        ds_view.next()
        item = ds_view[-1]
        case_id = item["case_id"]
        if case_id not in self.case_ids:
            out = DataItem({"case_id": case_id, "activity": self.start_symbol})
            self.output(out)
            self.case_ids.add(case_id)
        self.output(item)


class DataPreparation(ls.FunctionTerm):
    def __init__(self, *args, case_keys: list[str | int], activity_keys: list[str | int], **kwargs):
        super().__init__(*args, **kwargs)
        self.case_keys = case_keys
        self.activity_keys = activity_keys

    def f(self, item: DataItem) -> DataItem:
        """
        Process the input DataItem to output a new DataItem containing only case and activity keys.
        - Combines values from case_keys into a single case_id (as a tuple or single value).
        - Combines values from activity_keys into a single activity (as a tuple or single value).
        """
        # Construct the new DataItem with case_id and activity values
        return DataItem(
            {"case_id": handle_keys(self.case_keys, item), "activity": handle_keys(self.activity_keys, item)}
        )


class StreamingActivityPredictor(ls.FunctionTerm):
    def __init__(self, *args, strategy: StreamingMiner, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.case_ids = set()

    def run(self, ds_view: ls.DataStreamView):
        ds_view.next()
        item = ds_view[-1]

        start_time = time.time()

        probs = self.strategy.case_probs(item["case_id"])
        prediction = probs_prediction(probs, self.strategy.config)

        event: Event = {"case_id": item["case_id"], "activity": item["activity"]}

        self.strategy.update(event)

        end_time = time.time()
        latency = (end_time - start_time) * 1000  # latency in milliseconds (ms)

        out = DataItem(
            {
                "case_id": item["case_id"],
                "activity": item["activity"],  # actual activity
                "prediction": prediction,  # containing predicted activity
                "latency": latency,
            }
        )
        self.output(out)


class Evaluation(ls.FunctionTerm):
    def __init__(self, *args, top_activities: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_activities = top_activities
        self.correct_predictions = 0
        self.total_predictions = 0
        self.missing_predictions = 0
        self.latency_sum = 0
        self.latency_max = 0

    def f(self, item: DataItem) -> DataItem:
        self.latency_sum += item["latency"]
        self.latency_max = max(item["latency"], self.latency_max)

        if item["prediction"] is None:
            self.missing_predictions += 1
        elif self.top_activities:
            if item["activity"] in item["prediction"]["top_k_activities"]:
                self.correct_predictions += 1
        elif item["activity"] == item["prediction"]["activity"]:
            self.correct_predictions += 1

        self.total_predictions += 1

        accuracy = self.correct_predictions / self.total_predictions * 100 if self.total_predictions > 0 else 0

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
