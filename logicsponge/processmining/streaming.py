import logging
import time
from collections.abc import Iterator
from datetime import timedelta

import pandas as pd

import logicsponge.core as ls
from logicsponge.core import DataItem  # , dashboard
from logicsponge.processmining.data_utils import handle_keys
from logicsponge.processmining.models import (
    StreamingMiner,
)
from logicsponge.processmining.types import ActivityName, Event
from logicsponge.processmining.utils import metrics_prediction

logger = logging.getLogger(__name__)


class IteratorStreamer(ls.SourceTerm):
    """
    For streaming from iterator.
    """

    def __init__(self, *args, data_iterator: Iterator, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_iterator = data_iterator

    def run(self):
        while True:
            for event in self.data_iterator:
                case_id = event["case_id"]
                activity = event["activity"]
                timestamp = event["timestamp"]

                out = DataItem(
                    {
                        "case_id": case_id,
                        "activity": activity,
                        "timestamp": timestamp,
                    }
                )
                self.output(out)

            # repeatedly sleep if done
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
        while True:
            ds_view.next()
            item = ds_view[-1]
            case_id = item["case_id"]
            if case_id not in self.case_ids:
                out = DataItem(
                    {
                        "case_id": case_id,
                        "activity": self.start_symbol,
                        "timestamp": None,
                    }
                )
                self.output(out)
                self.case_ids.add(case_id)
            self.output(item)


class DataPreparation(ls.FunctionTerm):
    def __init__(self, *args, case_keys: list[str], activity_keys: list[str], **kwargs):
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
        self.last_timestamps = {}  # records last timestamps

    def run(self, ds_view: ls.DataStreamView):
        while True:
            ds_view.next()
            item = ds_view[-1]

            start_time = time.time()

            metrics = self.strategy.case_metrics(item["case_id"])
            prediction = metrics_prediction(metrics, self.strategy.config)

            # prediction = self.strategy.case_predictions.get(item["case_id"], None)

            event: Event = {
                "case_id": item["case_id"],
                "activity": item["activity"],
                "timestamp": item["timestamp"],
            }

            self.strategy.update(event)

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # latency in milliseconds (ms)

            if (
                prediction
                and item["timestamp"]
                and self.last_timestamps.get(item["case_id"], None)
                and item["case_id"] in self.last_timestamps
                and item["activity"] in prediction["predicted_delays"]
            ):
                predicted_delay = prediction["predicted_delays"][item["activity"]]
                actual_delay = item["timestamp"] - self.last_timestamps[item["case_id"]]
                delay_error = abs(predicted_delay - actual_delay)
            else:
                actual_delay = None
                delay_error = None
                predicted_delay = None

            self.last_timestamps[item["case_id"]] = item["timestamp"]

            out = DataItem(
                {
                    "case_id": item["case_id"],
                    "activity": item["activity"],  # actual activity
                    "prediction": prediction,  # containing predicted activity
                    "latency": latency,
                    "delay_error": delay_error,
                    "actual_delay": actual_delay,
                    "predicted_delay": predicted_delay,
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
        self.last_timestamps = {}  # records last timestamps for every case

        self.delay_count = 0
        self.actual_delay_sum = 0.0
        self.delay_error_sum = 0.0
        self.normalized_error_sum = 0.0

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

        actual_delay = item["actual_delay"]
        delay_error = item["delay_error"]
        predicted_delay = item["predicted_delay"]

        if actual_delay is not None and delay_error is not None:
            self.delay_count += 1
            self.delay_error_sum += delay_error.total_seconds()
            self.actual_delay_sum += actual_delay.total_seconds()
            if actual_delay.total_seconds() + predicted_delay.total_seconds() == 0:
                normalized_error = 0
            else:
                normalized_error = delay_error.total_seconds() / (
                    actual_delay.total_seconds() + predicted_delay.total_seconds()
                )
            self.normalized_error_sum += normalized_error

        if self.delay_count > 0:
            mean_delay_error = timedelta(seconds=self.delay_error_sum / self.delay_count)
            mean_actual_delay = timedelta(seconds=self.actual_delay_sum / self.delay_count)
            mean_normalized_error = self.normalized_error_sum / self.delay_count
        else:
            mean_delay_error = None
            mean_actual_delay = None
            mean_normalized_error = None

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
                "mean_delay_error": mean_delay_error,
                "mean_actual_delay": mean_actual_delay,
                "mean_normalized_error": mean_normalized_error,
                "delay_predictions": self.delay_count,
            }
        )


def eval_to_table(data: dict) -> pd.DataFrame:
    # Extract and display the index
    if "index" in data:
        msg = f"========== {data["index"]} =========="
        logger.info(msg)

    # Initialize a dictionary to hold the tabular data
    table_data = {}

    for key, value in data.items():
        if "." not in key:  # Skip keys without a dot (e.g., "index")
            continue

        row_name, attribute = key.split(".", 1)

        # Initialize row if it doesn't exist
        if row_name not in table_data:
            table_data[row_name] = {}

        # Process the value based on its type
        if isinstance(value, float):
            table_data[row_name][attribute] = round(value, 2)
        elif isinstance(value, timedelta):
            days = value.days
            hours, remainder = divmod(value.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            table_data[row_name][attribute] = f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
        else:
            table_data[row_name][attribute] = value  # Add as-is for other types

    # Convert to a DataFrame
    df = pd.DataFrame.from_dict(table_data, orient="index")

    # Reset index to make the names a column
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Name"}, inplace=True)

    return df


class PrintEval(ls.FunctionTerm):
    def run(self, ds_view: ls.DataStreamView):
        while True:
            ds_view.next()
            item = ds_view[-1]
            table = eval_to_table(item)
            logger.info(table)
