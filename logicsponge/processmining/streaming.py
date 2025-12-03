"""Module for streaming process mining tasks using LogicSponge."""

import csv
import logging
import random
import time
import typing
from collections import Counter, deque
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

    def generate(self) -> Iterator[DataItem]:
        """Generate data items from the iterator."""
        for item in self.data_iterator:
            # Normalize to DataItem instances. Many data sources yield dicts or
            # pandas.Series; ensure the flow_backend always receives DataItem.
            if isinstance(item, DataItem):
                yield item
            elif isinstance(item, dict):
                yield DataItem(item)
            else:
                # Try common conversion (pandas.Series has to_dict)
                try:
                    d = item.to_dict()  # type: ignore[attr-defined]
                    if isinstance(d, dict):
                        yield DataItem(d)
                        continue
                except Exception:
                    pass
                # Fallback: wrap non-dict value into a single-key DataItem
                yield DataItem({"value": item})

class AddStartSymbol(ls.FunctionTerm):
    """
    For streaming from list.

    Emits a start symbol before the first event of each case. Because `f()` is
    only allowed to return a single `DataItem` (or `None`), this term keeps a
    tiny internal buffer to hold the original event when it needs to emit the
    start symbol first.
    """

    def __init__(self, *args: dict, start_symbol: ActivityName, **kwargs: dict) -> None:
        """Initialize AddStartSymbol."""
        super().__init__(*args, **kwargs)
        self.case_ids = set()
        self.start_symbol = start_symbol
        self._buffer: deque[DataItem] = deque()

    def f(self, di: DataItem) -> DataItem | None:
        """Return at most one DataItem. Use internal buffer to emit start symbol before event."""
        # If we have a pending item from a previous call, emit it now and
        # process the current item into the buffer for the following calls.
        if self._buffer:
            to_emit = self._buffer.popleft()
            case_id = di["case_id"]
            if case_id not in self.case_ids:
                start = DataItem({"case_id": case_id, "activity": self.start_symbol, "timestamp": None})
                # ensure the start symbol is emitted before the original di
                self._buffer.append(start)
                self._buffer.append(di)
                self.case_ids.add(case_id)
            else:
                self._buffer.append(di)
            return to_emit

        # No pending item: decide what to emit for the current di
        case_id = di["case_id"]
        if case_id not in self.case_ids:
            start = DataItem({"case_id": case_id, "activity": self.start_symbol, "timestamp": None})
            # buffer the original so it follows the start symbol
            self._buffer.append(di)
            self.case_ids.add(case_id)
            return start

        return di


class DataPreparation(ls.FunctionTerm):
    """Prepare data for streaming."""

    def __init__(self, *args: dict, case_keys: list[str], activity_keys: list[str], **kwargs: dict) -> None:
        """Prepare data for streaming."""
        super().__init__(*args, **kwargs)
        self.case_keys = case_keys
        self.activity_keys = activity_keys

    def f(self, di: DataItem) -> DataItem:
        """
        Process the input DataItem to output a new DataItem containing only case and activity keys.

        - Combines values from case_keys into a single case_id (as a tuple or single value).
        - Combines values from activity_keys into a single activity (as a tuple or single value).
        """
        # Construct the new DataItem with case_id and activity values
        return DataItem(
            {"case_id": handle_keys(self.case_keys, di), "activity": handle_keys(self.activity_keys, di)}  # type: ignore # noqa: PGH003
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

    def f(self, di: DataItem) -> DataItem:
        """Process a single DataItem: compute prediction, update model, and return enriched DataItem."""
        case_id = di["case_id"]

        start_time = time.time()
        metrics = self.strategy.case_metrics(case_id)
        prediction = metrics_prediction(metrics, self.strategy.config)
        predict_latency = time.time() - start_time  # time taken to compute prediction

        start_time_training = time.time()
        event: Event = {
            "case_id": di["case_id"],
            "activity": di["activity"],
            "timestamp": di.get("timestamp"),
        }

        self.strategy.update(event)
        training_latency = time.time() - start_time_training  # time taken to update the model

        end_time = time.time()
        latency = (end_time - start_time) * 1000  # latency in milliseconds (ms)

        if (
            prediction
            and di.get("timestamp")
            and self.last_timestamps.get(di["case_id"], None)
            and di["activity"] in (prediction.get("predicted_delays") or {})
        ):
            predicted_delay = prediction["predicted_delays"][di["activity"]]
            actual_delay = di["timestamp"] - self.last_timestamps[di["case_id"]]
            delay_error = abs(predicted_delay - actual_delay)
        else:
            actual_delay = None
            delay_error = None
            predicted_delay = None

        self.last_timestamps[di["case_id"]] = di.get("timestamp")

        return DataItem(
            {
                "case_id": di["case_id"],
                "activity": di["activity"],  # actual activity
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

    def f(self, di: DataItem) -> DataItem:
        """Process incoming DataItem and update evaluation metrics."""
        if di["case_id"] not in self.sequence_lengths:
            self.sequence_lengths[di["case_id"]] = 0
            self.likelihoods[di["case_id"]] = 0.0

        self.likelihoods[di["case_id"]] *= di["likelihood"]
        self.sequence_lengths[di["case_id"]] += 1

        # # Compute perplexity
        # normalized_likelihood = self.likelihoods[di["case_id"]] ** (1 / self.sequence_lengths[di["case_id"]])
        # self.perplexities[di["case_id"]] = compute_seq_perplexity(normalized_likelihood, log_likelihood=False)

        # perplexity_stats = compute_perplexity_stats(list(self.perplexities.values()))

        self.latency_sum += di["latency"]
        self.latency_max = max(di["latency"], self.latency_max)

        self.predict_latency_sum += di["predict_latency"]
        self.train_latency_sum += di["train_latency"]
        if di["prediction"] is None:
            self.missing_predictions += 1
        else:
            if (
                (self.top_activities and di["activity"] in di["prediction"]["top_k_activities"])
                or di["activity"] == di["prediction"]["activity"]
            ):
                self.correct_predictions += 1

            if di["activity"] in di["prediction"]["top_k_activities"]:
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

        actual_delay = di["actual_delay"]
        delay_error = di["delay_error"]
        predicted_delay = di["predicted_delay"]

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
                "prediction": di["prediction"],
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

    def f(self, di: ls.DataItem) -> None:
        table = eval_to_table(di)
        logger.info(table)


class CSVStatsWriter(ls.FunctionTerm):
    """
    Write evaluation statistics from eval_to_table's DataFrame to a CSV file.

    Receives a DataItem, generates a DataFrame using eval_to_table,
    and writes each row of the DataFrame to the CSV file,
    optionally adding a batch index from the DataItem.
    """

    def __init__(  # noqa: D417
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

    def f(self, di: ls.DataItem) -> ls.DataItem:
        """Process incoming DataItem, generate DataFrame, and write stats to CSV."""
        # Generate DataFrame using eval_to_table (assuming eval_to_table is accessible)
        # This function is defined in the same file, so it should be accessible.
        df_to_save = eval_to_table(di)

        if not isinstance(df_to_save, pd.DataFrame) or df_to_save.empty:
            # logger.info("DataFrame from eval_to_table is empty or not a DataFrame. Nothing to write to CSV.")
            return di  # Return original di if no DataFrame to save

        # Prepare records from DataFrame
        records_to_write = df_to_save.to_dict("records")

        # Determine fieldnames from DataFrame columns
        final_fieldnames = df_to_save.columns.tolist()

        # Add batch index from the original DataItem if present
        batch_idx = di.get("index")  # 'index' is typically added by ls.AddIndex

        if batch_idx is not None:
            # Add batch index column name to fieldnames if not already there (e.g., from df_to_save)
            if self.batch_index_col_name not in final_fieldnames:
                final_fieldnames.insert(0, self.batch_index_col_name)  # Add to the beginning

            # Add batch_idx value to each record
            for record in records_to_write:
                record[self.batch_index_col_name] = batch_idx

        if not records_to_write:  # Should be caught by df_to_save.empty check
            return di

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
        return di


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

    def __init__(
            self, *args, csv_path: Path, model_name: str, append: bool = True, **kwargs
        ) -> None:
        """Initialize the PredictionCSVWriter."""
        super().__init__(*args, **kwargs)
        self.csv_path = csv_path
        self.model_name = model_name
        self.append = append
        self._step_counter = 0

    def f(self, di: DataItem) -> DataItem:
        self._step_counter += 1

        actual = di.get("activity")
        prediction = di.get("prediction")
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
            "case_id": di.get("case_id"),
            "timestamp": di.get("timestamp"),
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

        return di


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

    def f(self, di: DataItem) -> DataItem:
        """Process incoming DataItem and write actual activity to CSV."""
        self._step_counter += 1
        row = {
            "step": self._step_counter,
            "case_id": di.get("case_id"),
            "timestamp": di.get("timestamp"),
            "actual": di.get("activity"),
        }

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_path.is_file()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or not self.append:
                writer.writeheader()
            writer.writerow(row)

        return di


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
    - insert: insert a new activity named 'A_new' BEFORE the arriving di.
    - split: if arriving di is A1, emit it and then insert a new activity 'A1_post' AFTER it.
    - delete: if arriving di is A1, drop it; otherwise pass through.

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
        self._buffer: deque[DataItem] = deque()

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

    def f(self, di: DataItem) -> DataItem | None:
        """
        Process a single DataItem and return at most one DataItem (or None).

        Internally uses a small buffer to hold the next item(s) when an
        alteration requires emitting more than one logical item for a single
        incoming event (e.g. insert/split semantics).
        """
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

        act = di.get("activity")
        if isinstance(act, str):
            self._update_frequencies(act)

        def _buffer_current(item: DataItem) -> None:
            """Decide alteration for `item` and append resulting logical outputs to buffer."""
            # If the alteration should not apply, queue the original item
            if not self._should_apply():
                self._buffer.append(item)
                return

            # Apply alteration by appending to buffer rather than emitting
            if self.alteration_type == "switch":
                a1, a2 = self._top_a1, self._top_a2
                if a1 is not None and a2 is not None and isinstance(item.get("activity"), str):
                    if item.get("activity") == a1:
                        self._buffer.append(DataItem({**item, "activity": a2}))
                        self._altered_count += 1
                        logger.info(
                            "StreamAlteration total_altered=%d (switch at step=%d)",
                            self._altered_count, self._step_counter
                        )
                        return
                    if item.get("activity") == a2:
                        self._buffer.append(DataItem({**item, "activity": a1}))
                        self._altered_count += 1
                        logger.info(
                            "StreamAlteration total_altered=%d (switch at step=%d)",
                            self._altered_count, self._step_counter
                        )
                        return
                # fallback to no-op
                self._buffer.append(item)

            elif self.alteration_type == "insert":
                pre = DataItem({"case_id": item.get("case_id"), "activity": "A_new", "timestamp": None})
                self._buffer.append(pre)
                self._buffer.append(item)
                self._altered_count += 1
                logger.info(
                    "StreamAlteration total_altered=%d (insert at step=%d)", self._altered_count, self._step_counter
                )

            elif self.alteration_type == "split":
                a1 = self._top_a1
                if a1 is not None and isinstance(item.get("activity"), str) and item.get("activity") == a1:
                    post = DataItem({"case_id": item.get("case_id"), "activity": f"{a1}_post", "timestamp": None})
                    self._buffer.append(item)
                    self._buffer.append(post)
                    self._altered_count += 1
                    logger.info(
                        "StreamAlteration total_altered=%d (split at step=%d)", self._altered_count, self._step_counter
                    )
                else:
                    self._buffer.append(item)

            elif self.alteration_type == "delete":
                a1 = self._top_a1
                if a1 is not None and isinstance(item.get("activity"), str) and item.get("activity") == a1:
                    self._altered_count += 1
                    logger.info(
                        "StreamAlteration total_altered=%d (delete at step=%d)", self._altered_count, self._step_counter
                    )
                    # do not append anything -> effectively drop
                    return
                self._buffer.append(item)

            else:
                self._buffer.append(item)

        # If there is a pending item to emit (from previous processing), emit it
        # and process the current incoming item into the buffer for later.
        if self._buffer:
            to_emit = self._buffer.popleft()
            _buffer_current(di)
            return to_emit

        # No pending items: decide what to emit now (may also buffer follow-ups)
        # If alteration does not apply, just return the item
        if not self._should_apply():
            return di

        # Alteration should apply and buffer is empty -> we can emit one item now
        if self.alteration_type == "switch":
            a1, a2 = self._top_a1, self._top_a2
            if a1 is not None and a2 is not None and isinstance(act, str):
                if act == a1:
                    self._altered_count += 1
                    logger.info(
                        "StreamAlteration total_altered=%d (switch at step=%d)", self._altered_count, self._step_counter
                    )
                    return DataItem({**di, "activity": a2})
                if act == a2:
                    self._altered_count += 1
                    logger.info(
                        "StreamAlteration total_altered=%d (switch at step=%d)", self._altered_count, self._step_counter
                    )
                    return DataItem({**di, "activity": a1})
            return di

        if self.alteration_type == "insert":
            pre = DataItem({"case_id": di.get("case_id"), "activity": "A_new", "timestamp": None})
            # buffer original so it follows the inserted pre
            self._buffer.append(di)
            self._altered_count += 1
            logger.info(
                "StreamAlteration total_altered=%d (insert at step=%d)", self._altered_count, self._step_counter
            )
            return pre

        if self.alteration_type == "split":
            a1 = self._top_a1
            if a1 is not None and isinstance(act, str) and act == a1:
                post = DataItem({"case_id": di.get("case_id"), "activity": f"{a1}_post", "timestamp": None})
                # current original emitted now, post emitted later
                self._buffer.append(post)
                self._altered_count += 1
                logger.info(
                    "StreamAlteration total_altered=%d (split at step=%d)", self._altered_count, self._step_counter
                )
                return di
            return di

        if self.alteration_type == "delete":
            a1 = self._top_a1
            if a1 is not None and isinstance(act, str) and act == a1:
                self._altered_count += 1
                logger.info(
                    "StreamAlteration total_altered=%d (delete at step=%d)", self._altered_count, self._step_counter
                )
                return None
            return di

        # Fallback
        return di


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
            # Prefer unified 'predicted' column; fallback to legacy 'prediction' if needed
            series = predictions_df.get("predicted")
            if series is None:
                series = predictions_df.get("prediction")
            return (series if series is not None else pd.Series(dtype=object)).tolist()
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

    def f(self, di: DataItem) -> DataItem:
        """Enrich di with streaming comparison metrics computed from CSVs."""
        mem = self._build_memory()
        model_keys = [k for k in mem if k != "actual"]
        if (not mem) or ("actual" not in mem) or (not model_keys):
            return di

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
            idx = di.get("index")
            if isinstance(idx, int) and idx % 1000 == 0 and self.csv_dir.is_dir():
                results_dir = self.csv_dir.parent
                save_all_comparison_heatmaps(results_dir, self.run_id, snapshot_idx=idx)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.debug("StreamingComparisonComputer: failed to save heatmap snapshot: %s", exc)

        # Merge enriched metrics into the di for PrintEval downstream
        merged = {**di, **enriched}
        return DataItem(merged)
