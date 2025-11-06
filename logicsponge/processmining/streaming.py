"""Module for streaming process mining tasks using LogicSponge."""

import csv
import logging
import random
import time
import typing
from collections import Counter
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import logicsponge.core as ls
from logicsponge.core import DataItem  # , dashboard
from logicsponge.processmining.data_utils import handle_keys
from logicsponge.processmining.miners import (
    StreamingMiner,
)
from logicsponge.processmining.types import ActivityName, Event
from logicsponge.processmining.utils import (
    compare_models_comparison,
    metrics_prediction,
    save_all_comparison_heatmaps,
)

logger = logging.getLogger(__name__)


class IteratorStreamer(ls.SourceTerm):
    """For streaming from iterator."""

    def __init__(self, *args: dict, data_iterator: Iterator, **kwargs: dict) -> None:
        """Create an IteratorStreamer."""
        super().__init__(*args, **kwargs)
        self.data_iterator = data_iterator

    def run(self) -> None:
        """Run the IteratorStreamer."""
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

class SynInfiniteStreamer(ls.SourceTerm):
    """For streaming synthetic infinite data."""

    def __init__(self, *args: dict, max_prefix_length=5, space=30, **kwargs: dict) -> None:
        """Create an IteratorStreamer."""
        super().__init__(*args, **kwargs)
        self.max_prefix_length = max_prefix_length
        self.space = space


    def run(self) -> None:
        """Run the IteratorStreamer."""
        start_seq = 12
        prefix_length = 3
        while True:
            prefix_length = random.randint(1, self.max_prefix_length)
            for i in range(start_seq, start_seq + prefix_length):
                case_id = f"case_{0}"
                activity = f"act_{i%self.space}"
                timestamp = pd.Timestamp.now()

                out = DataItem(
                    {
                        "case_id": case_id,
                        "activity": activity,
                        "timestamp": timestamp,
                    }
                )
                self.output(out)

            prefix_length = random.randint(1, self.max_prefix_length)
            start_seq = (2 * start_seq + prefix_length) % self.space
            # repeatedly sleep if done
            time.sleep(0.1)


class AddStartSymbol(ls.FunctionTerm):
    """For streaming from list."""

    def __init__(self, *args: dict, start_symbol: ActivityName, **kwargs: dict) -> None:
        """Initialize AddStartSymbol."""
        super().__init__(*args, **kwargs)
        self.case_ids = set()
        self.start_symbol = start_symbol

    def run(self, ds_view: ls.DataStreamView) -> None:
        """Run the AddStartSymbol function."""
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
    """Prepare data for streaming."""

    def __init__(self, *args: dict, case_keys: list[str], activity_keys: list[str], **kwargs: dict) -> None:
        """Prepare data for streaming."""
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
            {"case_id": handle_keys(self.case_keys, item), "activity": handle_keys(self.activity_keys, item)}  # type: ignore # noqa: PGH003
        )


class StreamingActivityPredictor(ls.FunctionTerm):
    """Streaming activity predictor."""

    def __init__(
            self, *args: dict,
            strategy: StreamingMiner, compute_metrics: bool = False,
            **kwargs: dict
        ) -> None:
        """Initialize the StreamingActivityPredictor."""
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        # self.case_ids = set()
        self.last_timestamps = {}  # records last timestamps

    def run(self, ds_view: ls.DataStreamView) -> None:
        """Run the StreamingActivityPredictor."""
        while True:
            ds_view.next()
            item = ds_view[-1]
            case_id = item["case_id"]

            start_time = time.time()
            metrics = self.strategy.case_metrics(case_id)
            prediction = metrics_prediction(metrics, self.strategy.config)
            predict_latency = time.time() - start_time  # time taken to compute prediction

            # pause_time = time.time()
            # likelihood = self.strategy.state_act_likelihood(metrics["state_id"], item["activity"])
            # start_time += time.time() - pause_time  # Adjust start time to account for the pause

            # prediction = self.strategy.case_predictions.get(item["case_id"], None)

            start_time_training = time.time()
            event: Event = {
                "case_id": item["case_id"],
                "activity": item["activity"],
                "timestamp": item["timestamp"],
            }

            self.strategy.update(event)
            training_latency = time.time() - start_time_training  # time taken to update the model

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
                    "likelihood": 0.0,
                    "latency": latency,
                    "predict_latency": predict_latency * 1_000_000,
                    "train_latency": training_latency * 1_000_000,
                    "delay_error": delay_error,
                    "actual_delay": actual_delay,
                    "predicted_delay": predicted_delay,
                }
            )
            self.output(out)


class Evaluation(ls.FunctionTerm):
    """Evaluate streaming predictions."""

    def __init__(self, *args, top_activities: bool = False, **kwargs) -> None:
        """Initialize Evaluation."""
        super().__init__(*args, **kwargs)
        self.top_activities = top_activities
        self.correct_predictions = 0
        self.top_k_correct_preds = 0
        self.total_predictions = 0
        self.missing_predictions = 0

        self.predict_latency_sum = 0
        self.train_latency_sum = 0

        self.latency_sum = 0
        self.latency_max = 0
        self.last_timestamps = {}  # records last timestamps for every case

        self.delay_count = 0
        self.actual_delay_sum = 0.0
        self.delay_error_sum = 0.0
        self.normalized_error_sum = 0.0

        self.likelihoods: dict[int, float] = {}
        self.sequence_lengths: dict[int, int] = {}
        # self.perplexities: dict[int, float] = {}

    def f(self, item: DataItem) -> DataItem:
        if item["case_id"] not in self.sequence_lengths:
            self.sequence_lengths[item["case_id"]] = 0
            self.likelihoods[item["case_id"]] = 0.0

        self.likelihoods[item["case_id"]] *= item["likelihood"]
        self.sequence_lengths[item["case_id"]] += 1

        # # Compute perplexity
        # normalized_likelihood = self.likelihoods[item["case_id"]] ** (1 / self.sequence_lengths[item["case_id"]])
        # self.perplexities[item["case_id"]] = compute_seq_perplexity(normalized_likelihood, log_likelihood=False)

        # perplexity_stats = compute_perplexity_stats(list(self.perplexities.values()))

        self.latency_sum += item["latency"]
        self.latency_max = max(item["latency"], self.latency_max)

        self.predict_latency_sum += item["predict_latency"]
        self.train_latency_sum += item["train_latency"]

        if item["prediction"] is None:
            self.missing_predictions += 1
        else:
            if (
                (self.top_activities and item["activity"] in item["prediction"]["top_k_activities"])
                or item["activity"] == item["prediction"]["activity"]
            ):
                self.correct_predictions += 1

            if item["activity"] in item["prediction"]["top_k_activities"]:
                self.top_k_correct_preds += 1

        self.total_predictions += 1

        # ######
        #         stats = strategy.stats

        # total = stats["total_predictions"]
        # correct_percentage = (stats["correct_predictions"] / total * 100) if total > 0 else 0
        # wrong_percentage = (stats["wrong_predictions"] / total * 100) if total > 0 else 0
        # empty_percentage = (stats["empty_predictions"] / total * 100) if total > 0 else 0

        # top_k_accuracies = (
        #     [(top_k_correct / total * 100) for top_k_correct in stats["top_k_correct_preds"]]
        #     if total > 0
        #     else [0] * len(stats["top_k_correct_preds"])
        # )

        # per_state_stats = stats.get("per_state_stats", {})
        # # Convert each value in the dictionary (PerStateStats) to a dict
        # for key, value in per_state_stats.items():
        #     per_state_stats[key] = value.to_dict()

        # stats_to_log.append(
        #     {
        #         "strategy": strategy_name,
        #         "strategy_accuracy": correct_percentage,
        #         "strategy_perplexity": stats["pp_harmonic_mean"],
        #         "strategy_eval_time": evaluation_time,
        #         "per_state_stats": per_state_stats
        #     }
        # )
        # #########

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
        top_k_accuracy = self.top_k_correct_preds / self.total_predictions * 100 if self.total_predictions > 0 else 0

        return DataItem(
            {
                "prediction": item["prediction"],
                "correct_predictions": self.correct_predictions,
                "total_predictions": self.total_predictions,
                "missing_predictions": self.missing_predictions,
                "top_k_correct_preds": self.top_k_correct_preds,
                "accuracy": accuracy,
                "top_k_accuracy": top_k_accuracy,
                "predict_latency_mean": self.predict_latency_sum / self.total_predictions,
                "train_latency_mean": self.train_latency_sum / self.total_predictions,
                "latency_mean": self.latency_sum / self.total_predictions,
                "latency_max": self.latency_max,
                "mean_delay_error": mean_delay_error,
                "mean_actual_delay": mean_actual_delay,
                "mean_normalized_error": mean_normalized_error,
                "delay_predictions": self.delay_count,
                # **perplexity_stats,
            }
        )


def eval_to_table(data: dict | ls.DataItem) -> pd.DataFrame:
    """Evaluate and add to table."""
    # Extract and display the index
    if "index" in data:
        msg = f"========== {data['index']} =========="
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
    dataframe = pd.DataFrame.from_dict(table_data, orient="index")

    # Reset index to make the names a column
    dataframe.reset_index() # inplace=True can be inconsistent
    dataframe.rename(columns={"index": "Name"}) # inplace=True can be inconsistent

    return dataframe


class PrintEval(ls.FunctionTerm):
    """Add to table and show the table."""

    def f(self, item: ls.DataItem) -> None:
        table = eval_to_table(item)
        logger.info(table)


class CSVStatsWriter(ls.FunctionTerm):
    """
    Write evaluation statistics from eval_to_table's DataFrame to a CSV file.

    Receives a DataItem, generates a DataFrame using eval_to_table,
    and writes each row of the DataFrame to the CSV file,
    optionally adding a batch index from the DataItem.
    """

    def __init__(
            self, *args: dict,
            csv_path: Path, append: bool = True, batch_index_col_name: str = "batch_index",
            **kwargs: dict
        ) -> None:
        """
        Initialize the CSV writer.

        Args:
            csv_path: Path to the CSV file where stats will be written.
            append: Whether to append to an existing file (default True).
                      If False, the file will be overwritten.
            batch_index_col_name: Name for the column storing the batch index from the DataItem.

        """
        super().__init__(*args, **kwargs)
        self.csv_path = csv_path
        self.append = append
        self.batch_index_col_name = batch_index_col_name

    def f(self, item: ls.DataItem) -> ls.DataItem:
        """Process incoming DataItem, generate DataFrame, and write stats to CSV."""
        # Generate DataFrame using eval_to_table (assuming eval_to_table is accessible)
        # This function is defined in the same file, so it should be accessible.
        df_to_save = eval_to_table(item)

        if not isinstance(df_to_save, pd.DataFrame) or df_to_save.empty:
            # logger.info("DataFrame from eval_to_table is empty or not a DataFrame. Nothing to write to CSV.")
            return item  # Return original item if no DataFrame to save

        # Prepare records from DataFrame
        records_to_write = df_to_save.to_dict("records")

        # Determine fieldnames from DataFrame columns
        final_fieldnames = df_to_save.columns.tolist()

        # Add batch index from the original DataItem if present
        batch_idx = item.get("index")  # 'index' is typically added by ls.AddIndex

        if batch_idx is not None:
            # Add batch index column name to fieldnames if not already there (e.g., from df_to_save)
            if self.batch_index_col_name not in final_fieldnames:
                final_fieldnames.insert(0, self.batch_index_col_name)  # Add to the beginning

            # Add batch_idx value to each record
            for record in records_to_write:
                record[self.batch_index_col_name] = batch_idx

        if not records_to_write:  # Should be caught by df_to_save.empty check
            return item

        # Determine file mode and if header needs to be written
        file_exists = self.csv_path.is_file()
        needs_header = False

        if self.append and file_exists:
            mode = "a"
            if self.csv_path.stat().st_size == 0:  # File exists but is empty
                needs_header = True
            # If appending to a non-empty file, we assume headers match or DictWriter will handle discrepancies.
        else:  # Overwrite mode or file doesn't exist
            mode = "w"
            needs_header = True

        try:
            # Ensure the directory exists
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            with self.csv_path.open(mode, newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames)

                if needs_header:
                    writer.writeheader()

                writer.writerows(records_to_write)  # type: ignore # noqa: PGH003
        except OSError:
            logger.exception("Error writing to CSV file %s.", self.csv_path)
        except Exception:
            logger.exception(
                "An unexpected error occurred in CSVStatsWriter while writing to %s.", self.csv_path
            )

        # Return the original DataItem, allowing it to continue in the pipeline
        return item


class PredictionCSVWriter(ls.FunctionTerm):
    """
    Write per-event prediction info for a given model to a CSV file.

    Columns:
    - step: incremental integer per writer instance (1-based)
    - case_id: case identifier
    - timestamp: event timestamp (if available)
    - actual: observed activity at this step
    - predicted: model's top-1 predicted activity (if any)
    - top_k: pipe-separated list of top-k predicted activities (if available)
    - correct: 1 if predicted == actual else 0 (only when predicted available)
    """

    def __init__(self, *args, csv_path: Path, model_name: str, append: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_path = csv_path
        self.model_name = model_name
        self.append = append
        self._step_counter = 0

    def f(self, item: DataItem) -> DataItem:
        self._step_counter += 1

        actual = item.get("activity")
        prediction = item.get("prediction")
        predicted_act = None
        top_k = None
        correct = None

        if isinstance(prediction, dict):
            predicted_act = prediction.get("activity")
            tk = prediction.get("top_k_activities")
            if isinstance(tk, (list, tuple)):
                top_k = "|".join(map(str, tk))
            correct = int(predicted_act == actual) if predicted_act is not None else None

        row = {
            "step": self._step_counter,
            "case_id": item.get("case_id"),
            "timestamp": item.get("timestamp"),
            "actual": actual,
            "predicted": predicted_act,
            "top_k": top_k,
            "correct": correct,
        }

        # Ensure parent directory exists
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.csv_path.is_file()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or not self.append:
                writer.writeheader()
            writer.writerow(row)

        return item


class ActualCSVWriter(ls.FunctionTerm):
    """
    Write the baseline actual activities to a CSV file once per event.

    This should be placed before branching so it records each event exactly once.
    Columns: step (1-based), case_id, timestamp, actual
    """

    def __init__(self, *args: dict, csv_path: Path, append: bool = True, **kwargs: dict) -> None:
        """Initialize the ActualCSVWriter."""
        super().__init__(*args, **kwargs)
        self.csv_path = csv_path
        self.append = append
        self._step_counter = 0

    def f(self, item: DataItem) -> DataItem:
        """Process incoming DataItem and write actual activity to CSV."""
        self._step_counter += 1
        row = {
            "step": self._step_counter,
            "case_id": item.get("case_id"),
            "timestamp": item.get("timestamp"),
            "actual": item.get("activity"),
        }

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_path.is_file()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or not self.append:
                writer.writeheader()
            writer.writerow(row)

        return item


class StreamAlteration(ls.FunctionTerm):
    """
    Alter the incoming event stream according to a chosen strategy.

    Parameters
    ----------
    - alteration_type: one of ["switch", "insert", "split", "delete"]
    - rate: controls stochastic application rate when transition > 1.0 (default 1.0)
    - alteration_start: iteration index (1-based step counter) at which alterations begin (default 5000)
    - transition: number of iterations over which to ramp up probability (default 1 => immediate)

    Behavior
    - switch: let A1 and A2 be the two most frequent activities observed since alteration_start; swap A1<->A2.
    - insert: insert a new activity named 'A_new' BEFORE the arriving item.
    - split: if arriving item is A1, emit it and then insert a new activity 'A1_post' AFTER it.
    - delete: if arriving item is A1, drop it; otherwise pass through.

    Notes
    -----
    - "Since alteration_start" is implemented as frequencies accumulated online for all
      items processed with step >= alteration_start.
    - Inserted activities reuse the current case_id and have timestamp=None.

    """

    VALID_TYPES: typing.ClassVar[set[str]] = {"switch", "insert", "split", "delete"}

    def __init__(
        self,
        *args: dict,
        alteration_type: str = "switch",
        rate: float = 1.0,
        alteration_start: int = 5000,
        transition: int = 1,
        **kwargs: dict,
    ) -> None:
        """Initialize the StreamAlteration term."""
        super().__init__(*args, **kwargs)
        if alteration_type not in self.VALID_TYPES:
            msg = f"alteration_type must be one of {sorted(self.VALID_TYPES)}"
            raise ValueError(msg)
        if transition < 1:
            msg = "transition must be >= 1"
            raise ValueError(msg)
        if alteration_start < 0:
            msg = "alteration_start must be >= 0"
            raise ValueError(msg)

        self.alteration_type = alteration_type
        self.rate = float(rate)
        self.alteration_start = int(alteration_start)
        self.transition = int(transition)

        self._step_counter = 0  # counts items seen (1-based for readability)
        self._freq: Counter[str] = Counter()
        self._top_a1: str | None = None
        self._top_a2: str | None = None
        self._rng = random.Random()
        self._started_logged = False
        self._last_logged_top: tuple[str | None, str | None] | None = None
        # total number of data items actually altered (switch/insert/split/delete)
        self._altered_count = 0

        logger.info(
            "StreamAlteration initialized: type=%s rate=%.3f start=%d transition=%d",
            self.alteration_type,
            self.rate,
            self.alteration_start,
            self.transition,
        )

    def _should_apply(self) -> bool:
        """
        Decide whether the alteration should be applied at the current step.

        - Before alteration_start: never apply.
        - If transition == 1: apply fully (deterministically) once started.
        - Else: pick X ~ Uniform(0, rate). Apply if X <= progress, where
          progress = (current_step - alteration_start) / transition, clipped to [0, 1].
        """
        if self._step_counter < self.alteration_start:
            return False

        x = self._rng.random()

        if self.transition == 1:
            return x < self.rate

        progress = (self._step_counter - self.alteration_start) / float(self.transition)
        progress = max(0.0, min(1.0, progress))

        return x < progress * self.rate

    def _update_frequencies(self, activity: str) -> None:
        """Update activity frequencies and top-2 activities since alteration_start."""
        if self._step_counter >= self.alteration_start:
            self._freq[activity] += 1
            # Update top two
            if len(self._freq) >= 2:
                mc = self._freq.most_common(2)
                self._top_a1, self._top_a2 = mc[0][0], mc[1][0]
            elif len(self._freq) == 1:
                self._top_a1 = next(iter(self._freq))
                self._top_a2 = None

            # Log updates to the top-2 activities (first time and whenever they change)
            current_top = (self._top_a1, self._top_a2)
            if current_top != self._last_logged_top and self._top_a1 is not None:
                c1 = self._freq.get(self._top_a1, 0)
                c2 = self._freq.get(self._top_a2, 0) if self._top_a2 is not None else 0
                logger.info(
                    "StreamAlteration top activities updated (since start): A1=%s(count=%d) A2=%s(count=%d)",
                    self._top_a1,
                    c1,
                    self._top_a2,
                    c2,
                )
                self._last_logged_top = current_top

    def run(self, ds_view: ls.DataStreamView) -> None:
        """Run the stream alteration process."""
        while True:
            ds_view.next()
            item = ds_view[-1]
            self._step_counter += 1

            # Log first step at or after alteration_start
            if (not self._started_logged) and self._step_counter >= self.alteration_start:
                logger.info(
                    "StreamAlteration started at step=%d (type=%s, transition=%d, rate=%.3f)",
                    self._step_counter,
                    self.alteration_type,
                    self.transition,
                    self.rate,
                )
                self._started_logged = True

            # Update frequencies with the current observed activity (since alteration_start)
            act = item.get("activity")
            if isinstance(act, str):
                self._update_frequencies(act)

            # Decide whether to apply alteration for this event
            if not self._should_apply():
                # pass-through
                self.output(item)
                logger.debug(
                    "StreamAlteration pass-through at step=%d case_id=%s activity=%s (no alteration applied)",
                    self._step_counter,
                    item.get("case_id"),
                    act,
                )
                continue

            # Apply alteration based on the selected type
            if self.alteration_type == "switch":
                a1, a2 = self._top_a1, self._top_a2
                if a1 is not None and a2 is not None and isinstance(act, str):
                    if act == a1:
                        logger.debug(
                            "StreamAlteration switch applied step=%d case_id=%s from=%s to=%s",
                            self._step_counter,
                            item.get("case_id"),
                            a1,
                            a2,
                        )
                        self.output(DataItem({**item, "activity": a2}))
                        # count and log alteration
                        self._altered_count += 1
                        logger.info(
                            "StreamAlteration total_altered=%d (switch at step=%d)",
                            self._altered_count, self._step_counter
                        )
                        continue
                    if act == a2:
                        logger.debug(
                            "StreamAlteration switch applied step=%d case_id=%s from=%s to=%s",
                            self._step_counter,
                            item.get("case_id"),
                            a2,
                            a1,
                        )
                        self.output(DataItem({**item, "activity": a1}))
                        # count and log alteration
                        self._altered_count += 1
                        logger.info(
                            "StreamAlteration total_altered=%d (switch at step=%d)",
                            self._altered_count, self._step_counter
                        )
                        continue
                # if no top-2 yet or activity is different, pass-through
                self.output(item)

            elif self.alteration_type == "insert":
                # Insert before current item
                try:
                    pre = DataItem({
                        "case_id": item.get("case_id"),
                        "activity": "A_new",
                        "timestamp": None,
                    })
                    logger.debug(
                        "StreamAlteration insert applied step=%d case_id=%s inserted=%s before=%s",
                        self._step_counter,
                        item.get("case_id"),
                        pre.get("activity"),
                        act,
                    )
                    self.output(pre)
                    # count and log alteration
                    self._altered_count += 1
                    logger.info(
                        "StreamAlteration total_altered=%d (insert at step=%d)", self._altered_count, self._step_counter
                    )
                except Exception:
                    # If anything goes wrong, fall back to no-pre insert
                    logger.exception("StreamAlteration: failed to create pre-insert event; skipping insert")
                self.output(item)

            elif self.alteration_type == "split":
                a1 = self._top_a1
                if a1 is not None and isinstance(act, str) and act == a1:
                    # emit original, then post item
                    self.output(item)
                    post = DataItem({
                        "case_id": item.get("case_id"),
                        "activity": f"{a1}_post",
                        "timestamp": None,
                    })
                    logger.info(
                        "StreamAlteration split applied step=%d case_id=%s original=%s inserted=%s",
                        self._step_counter,
                        item.get("case_id"),
                        act,
                        post.get("activity"),
                    )
                    self.output(post)
                    # count and log alteration
                    self._altered_count += 1
                    logger.info(
                        "StreamAlteration total_altered=%d (split at step=%d)",
                        self._altered_count, self._step_counter
                    )
                else:
                    self.output(item)

            elif self.alteration_type == "delete":
                a1 = self._top_a1
                if a1 is not None and isinstance(act, str) and act == a1:
                    # drop the event
                    logger.info(
                        "StreamAlteration delete applied step=%d case_id=%s activity=%s dropped",
                        self._step_counter,
                        item.get("case_id"),
                        act,
                    )
                    # count and log alteration
                    self._altered_count += 1
                    logger.info(
                        "StreamAlteration total_altered=%d (delete at step=%d)",
                        self._altered_count, self._step_counter
                    )
                    continue
                self.output(item)

            else:
                # Unknown type (shouldn't happen due to validation)
                self.output(item)


class StreamingComparisonComputer(ls.FunctionTerm):
    """
    Compute correlation, anticorrelation, and similarity across models during streaming.

    Reads per-model CSVs (including a baseline file named 'actual') from csv_dir
    and reconstructs a temporary prediction_vectors_memory to call compare_models_comparison.

    It returns the original DataItem enriched with keys suitable for eval_to_table(), e.g.:
        - correlation_vs_<reference>.<tested>
        - anticorrelation_vs_<reference>.<tested>
        - similarity_vs_<reference>.<tested>
    Values are percentages (0..100) or NaN when unavailable.
    """

    def __init__(
        self,
        *args: dict,
        models: list[str],
        csv_dir: Path,
        reference_model: str | None = None,
        run_id: str | None = None,
        save_matrices: bool = True,
        **kwargs: dict,
    ) -> None:
        """Initialize the StreamingComparisonComputer."""
        super().__init__(*args, **kwargs)
        self.models = models
        self.csv_dir = csv_dir
        self.reference_model = reference_model
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H-%M", time.localtime())
        self.save_matrices = save_matrices

    def _load_series(self, name: str) -> list:
        path = self.csv_dir / f"{name}.csv"
        if not path.is_file():
            return []
        try:
            predictions_df = pd.read_csv(path)
            if name == "actual":
                return predictions_df.get("actual", pd.Series(dtype=object)).tolist()
            return predictions_df.get("predicted", pd.Series(dtype=object)).tolist()
        except (OSError, ValueError, pd.errors.ParserError, FileNotFoundError):
            return []

    def _build_memory(self) -> dict:
        mem: dict[str, list] = {}
        # baseline
        actual_series = self._load_series("actual")
        if actual_series:
            mem["actual"] = [actual_series]
        # models
        for m in self.models:
            s = self._load_series(m)
            if s:
                mem[m] = [s]
        return mem

    def f(self, item: DataItem) -> DataItem:
        """Enrich item with streaming comparison metrics computed from CSVs."""
        mem = self._build_memory()
        model_keys = [k for k in mem if k != "actual"]
        if (not mem) or ("actual" not in mem) or (not model_keys):
            return item

        ref = self.reference_model
        if (ref is None) or (ref not in model_keys):
            # choose a default reference: prefer ngram_3, else first model
            ref = "ngram_3" if "ngram_3" in model_keys else model_keys[0]

        # Prepare dataframes to optionally save
        correlation_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
        anticorrelation_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
        similarity_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)

        enriched: dict[str, float] = {}
        for tested in model_keys:
            for reference in model_keys:
                try:
                    res = compare_models_comparison(
                        mem, tested_model=tested, reference_model=reference, baseline_model="actual"
                    )
                    correlation = res.get("correlation")
                    anticorrelation = res.get("anticorrelation")
                    similarity = res.get("similarity")
                except (ValueError, KeyError, TypeError, ZeroDivisionError, IndexError):
                    correlation = anticorrelation = similarity = None

                correlation_df.loc[tested, reference] = (correlation * 100.0) if correlation is not None else np.nan
                anticorrelation_df.loc[tested, reference] = (
                    (anticorrelation * 100.0) if anticorrelation is not None else np.nan
                )
                similarity_df.loc[tested, reference] = (similarity * 100.0) if similarity is not None else np.nan

            # also expose numbers vs chosen reference for PrintEval
            try:
                res_ref = compare_models_comparison(
                    mem, tested_model=tested, reference_model=ref, baseline_model="actual"
                )
                if (c := res_ref.get("correlation")) is not None:
                    enriched[f"correlation_vs_{ref}.{tested}"] = c * 100.0
                if (a := res_ref.get("anticorrelation")) is not None:
                    enriched[f"anticorrelation_vs_{ref}.{tested}"] = a * 100.0
                if (s := res_ref.get("similarity")) is not None:
                    enriched[f"similarity_vs_{ref}.{tested}"] = s * 100.0
            except (ValueError, KeyError, TypeError, ZeroDivisionError, IndexError):
                logger.debug("StreamingComparisonComputer: failed computing vs-ref for %s", tested)

        # Optionally save matrices to results directory
        if self.save_matrices and self.csv_dir.is_dir():
            try:
                correlation_csv_path = self.csv_dir.parent / f"{self.run_id}_correlation_matrix.csv"
                anticorrelation_csv_path = self.csv_dir.parent / f"{self.run_id}_anticorrelation_matrix.csv"
                similarity_csv_path = self.csv_dir.parent / f"{self.run_id}_similarity_matrix.csv"
                correlation_df.to_csv(correlation_csv_path)
                anticorrelation_df.to_csv(anticorrelation_csv_path)
                similarity_df.to_csv(similarity_csv_path)
            except (OSError, ValueError, RuntimeError):
                logger.debug("StreamingComparisonComputer: could not save matrices to CSV")

        # Every 1000 steps, generate PNG heatmaps from the saved matrices
        try:
            idx = item.get("index")
            if isinstance(idx, int) and idx % 1000 == 0 and self.csv_dir.is_dir():
                results_dir = self.csv_dir.parent
                save_all_comparison_heatmaps(results_dir, self.run_id, snapshot_idx=idx)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.debug("StreamingComparisonComputer: failed to save heatmap snapshot: %s", exc)

        # Merge enriched metrics into the item for PrintEval downstream
        merged = {**item, **enriched}
        return DataItem(merged)
