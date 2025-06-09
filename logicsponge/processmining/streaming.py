import csv
import logging
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pandas as pd

import logicsponge.core as ls
from logicsponge.core import DataItem  # , dashboard
from logicsponge.processmining.data_utils import handle_keys
from logicsponge.processmining.models import (
    StreamingMiner,
)
from logicsponge.processmining.types import ActivityName, Event
from logicsponge.processmining.utils import compute_perplexity_stats, compute_seq_perplexity, metrics_prediction

logger = logging.getLogger(__name__)


class IteratorStreamer(ls.SourceTerm):
    """For streaming from iterator."""

    def __init__(self, *args, data_iterator: Iterator, **kwargs):
        """Create an IteratorStreamer."""
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
    """For streaming from list."""

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
        """Prepare data for streaming."""
        super().__init__(*args, **kwargs)
        self.case_keys = case_keys
        self.activity_keys = activity_keys

    def f(self, item: DataItem) -> DataItem:
        """Process the input DataItem to output a new DataItem containing only case and activity keys.

        - Combines values from case_keys into a single case_id (as a tuple or single value).
        - Combines values from activity_keys into a single activity (as a tuple or single value).
        """
        # Construct the new DataItem with case_id and activity values
        return DataItem(
            {"case_id": handle_keys(self.case_keys, item), "activity": handle_keys(self.activity_keys, item)} # type: ignore
        )


class StreamingActivityPredictor(ls.FunctionTerm):
    """Streaming activity predictor."""

    def __init__(
            self,
            *args,
            strategy: StreamingMiner,
            compute_metrics: bool = False,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        # self.case_ids = set()
        self.last_timestamps = {}  # records last timestamps

    def run(self, ds_view: ls.DataStreamView):
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
    def __init__(self, *args, top_activities: bool = False, **kwargs):
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
            if self.top_activities and item["activity"] in item["prediction"]["top_k_activities"]:
                    self.correct_predictions += 1
            elif item["activity"] == item["prediction"]["activity"]:
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
        top_k_accuracy = (
            self.top_k_correct_preds / self.total_predictions * 100 if self.total_predictions > 0 else 0
        )

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
    df = pd.DataFrame.from_dict(table_data, orient="index")

    # Reset index to make the names a column
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Name"}, inplace=True)

    return df


class PrintEval(ls.FunctionTerm):
    """Add to table and show the table."""

    def f(self, item: ls.DataItem):
        table = eval_to_table(item)
        logger.info(table)


class CSVStatsWriter(ls.FunctionTerm):
    """Write evaluation statistics from eval_to_table's DataFrame to a CSV file.

    Receives a DataItem, generates a DataFrame using eval_to_table,
    and writes each row of the DataFrame to the CSV file,
    optionally adding a batch index from the DataItem.
    """

    def __init__(
            self,
            *args,
            csv_path: Path,
            append: bool = True,
            batch_index_col_name: str = "batch_index",
            **kwargs
        ):
        """Initialize the CSV writer.

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

        if not records_to_write: # Should be caught by df_to_save.empty check
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

                writer.writerows(records_to_write) # type: ignore
        except OSError as e:
            logger.exception("Error writing to CSV file %s. %s", self.csv_path, e)
        except Exception as e:
            logger.exception(
                "An unexpected error occurred in CSVStatsWriter while writing to %s: %s",
                {self.csv_path},
                e
            )

        # Return the original DataItem, allowing it to continue in the pipeline
        return item
