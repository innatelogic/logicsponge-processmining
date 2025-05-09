"""Module for streaming miners.

This module contains the implementation of various streaming miners,
including the BasicMiner, MultiMiner, HardVoting, SoftVoting, AdaptiveVoting,
and Fallback classes.
"""

import itertools
import logging
import math
import random
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from datetime import timedelta
import time
from typing import Any

import matplotlib as mpl
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from logicsponge.processmining.config import update_config
from logicsponge.processmining.data_utils import add_input_symbols_sequence
from logicsponge.processmining.neural_networks import LSTMModel, RNNModel
from logicsponge.processmining.types import (
    ActivityDelays,
    ActivityName,
    CaseId,
    ComposedState,
    Event,
    Metrics,
    Prediction,
    ProbDistr,
    StateId,
    empty_metrics,
)
from logicsponge.processmining.utils import compute_perplexity_stats, metrics_prediction, probs_prediction

mpl.use("Agg")

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line-wrapping # noqa: FBT003

logger = logging.getLogger(__name__)

random.seed(123)


class PerStateStats:
    """Class to store statistics per state."""

    def __init__(self, state_id: StateId) -> None:
        """Initialize the statistics for a given state."""
        self.state_id = state_id
        self.total_predictions = 0
        self.correct_predictions = 0
        self.wrong_predictions = 0
        self.empty_predictions = 0
        self.level = 0
        self.visits = 0

    def update(self, result_str: str, level: int | None, visits: int | None) -> None:
        """Update the statistics based on the result string."""
        if result_str == "correct":
            self.correct_predictions += 1
        elif result_str == "wrong":
            self.wrong_predictions += 1
        elif result_str == "empty":
            self.empty_predictions += 1

        if level is not None:
            self.level = level
        if visits is not None:
            self.visits = visits

        self.total_predictions += 1

    def to_dict(self) -> dict:
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "state_id": self.state_id,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "wrong_predictions": self.wrong_predictions,
            "empty_predictions": self.empty_predictions,
            "level": self.level,
            "visits": self.visits,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerStateStats":
        """Create an object from a dictionary."""
        obj = cls(data["state_id"])
        obj.total_predictions = data["total_predictions"]
        obj.correct_predictions = data["correct_predictions"]
        obj.wrong_predictions = data["wrong_predictions"]
        obj.empty_predictions = data["empty_predictions"]
        obj.level = data["level"]
        obj.visits = data["visits"]
        return obj


class TrackedDict(dict):
    """A dictionary that tracks changes to its items."""

    def __setitem__(self, key: str, value: float | list) -> None:
        """Set an item in the dictionary and track the change."""
        if isinstance(value, float):
            old = self.get(key, None)
            if key == "perplexity":
                msg = f"[DEBUG] dict[{key!r}] changed from {old!r} to {value!r}"
                logger.debug(msg)
        super().__setitem__(key, value)


# ============================================================
# Base Streaming Miner (for streaming and batch mode)
# ============================================================


class StreamingMiner(ABC):
    """The Base Streaming Miner (for both streaming and batch mode)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the StreamingMiner with a configuration."""
        # Use CONFIG as a fallback if no specific config is provided
        self.config = update_config(config)

        # Set the initial state (or other initialization tasks)
        self.initial_state: ComposedState | None = None

        # Statistics for batch mode
        self.stats = TrackedDict(
            {
                "total_predictions": 0,
                "correct_predictions": 0,
                "wrong_predictions": 0,
                "empty_predictions": 0,
                "top_k_correct_preds": [0] * self.config["top_k"],
                "pp_harmonic_mean": 0.0,
                "pp_arithmetic_mean": 0.0,
                "pp_median": 0.0,
                "pp_q1": 0.0,
                "pp_q3": 0.0,
                # Per state analysis
                "per_state_stats": dict[StateId, PerStateStats](),
                # For delay predictions
                "delay_error_sum": 0,
                "actual_delay_sum": 0,
                "normalized_error_sum": 0,
                "num_delay_predictions": 0,
                "last_timestamps": {},  # last recorded timestamp for every case
            }
        )

        self.modified_cases = set()  # Records potentially modified cases (predictions) in last update

    def _increment_stat(
        self,
        state_id: StateId,
        result_str: str,
        level: int | None = None,
        visits: int | None = None,
    ) -> None:
        if state_id not in self.stats["per_state_stats"]:
            self.stats["per_state_stats"][state_id] = PerStateStats(state_id)

        state: PerStateStats = self.stats["per_state_stats"][state_id]
        state.update(result_str, level=level, visits=visits)

    def update_stats(self, event: Event, prediction: Prediction | None, state_id: StateId | None) -> None:
        """Update the statistics based on the actual activity, the prediction, and the top-k predictions."""
        case_id = event.get("case_id")
        actual_next_activity = event.get("activity")
        timestamp = event.get("timestamp")
        result_str = None

        self.stats["total_predictions"] += 1

        if prediction is None:
            self.stats["empty_predictions"] += 1
            result_str = "empty"
        else:
            predicted_activity = prediction["activity"]

            if actual_next_activity == predicted_activity:
                self.stats["correct_predictions"] += 1
                result_str = "correct"
                for i in range(len(self.stats["top_k_correct_preds"])):
                    self.stats["top_k_correct_preds"][i] += 1
            else:
                self.stats["wrong_predictions"] += 1
                for k in range(len(prediction["top_k_activities"])):
                    if actual_next_activity == prediction["top_k_activities"][k]:
                        for i in range(k, len(self.stats["top_k_correct_preds"])):
                            self.stats["top_k_correct_preds"][i] += 1
                        break

                result_str = "wrong"

        if state_id is not None:
            state_info = self.get_state_info(state_id) if hasattr(self, "get_state_info") else None

            if state_info is not None and isinstance(state_info, dict):
                self._increment_stat(state_id, result_str, level=state_info["level"], visits=state_info["total_visits"])

        # Update timing statistics
        if (
            prediction
            and case_id in self.stats["last_timestamps"]
            and actual_next_activity in prediction["predicted_delays"]
        ):
            predicted_delay = prediction["predicted_delays"][actual_next_activity]
            actual_delay = timestamp - self.stats["last_timestamps"][case_id]
            delay_error = abs(predicted_delay - actual_delay)
        else:
            actual_delay = None
            delay_error = None
            predicted_delay = None

        if actual_delay is not None and delay_error is not None and predicted_delay is not None:
            self.stats["num_delay_predictions"] += 1
            self.stats["delay_error_sum"] += delay_error.total_seconds()
            self.stats["actual_delay_sum"] += actual_delay.total_seconds()
            if actual_delay.total_seconds() + predicted_delay.total_seconds() == 0:
                normalized_error = 0
            else:
                normalized_error = delay_error.total_seconds() / (
                    actual_delay.total_seconds() + predicted_delay.total_seconds()
                )
            self.stats["normalized_error_sum"] += normalized_error

        self.stats["last_timestamps"][case_id] = timestamp

    @staticmethod
    def compute_seq_perplexity(normalized_likelihood: float, *, log_likelihood: bool) -> float:
        """Compute the perplexity of a sequence."""
        if normalized_likelihood is not None and normalized_likelihood > 0:
            return math.exp(-normalized_likelihood) if log_likelihood else 1.0 / normalized_likelihood
        return float("inf")

    def evaluate(self, data: list[list[Event]], mode: str = "incremental", *, log_likelihood: bool = False) -> float:
        """Evaluate in batch mode.

        Evaluate the dataset either incrementally or by full sequence.

        Modes: 'incremental' or 'sequence'.
        """
        # Evaluate the dataset using a Bayes classifier
        # self.initialize_bayes_classifier(data)
        # self.eval_bayes_classifier(data)

        # Initialize stats
        perplexities = []

        eval_start_time = time.time()
        pause_time = 0.0

        for sequence in tqdm(data, desc="Processing sequences"):
            pause_start_time = time.time()

            logger.debug(">>>>> Start Evaluating Sequence <<<<<")
            event_sequence = ""
            predicted_sequence = ""
            for event in sequence:
                event_sequence += event["activity"].__str__()
            logger.debug("Event sequence: %s", event_sequence.replace(self.config["stop_symbol"].__str__(), "S"))

            pause_time += time.time() - pause_start_time

            current_state = self.initial_state
            metrics = empty_metrics()
            likelihood = 0.0 if log_likelihood else 1.0

            for i in range(len(sequence)):
                if current_state is None:
                    self.stats["empty_predictions"] += 1
                    self.stats["total_predictions"] += 1
                    # If unparseable, count all remaining activities
                    # self.stats["empty_predictions"] += len(sequence) - i
                    # self.stats["total_predictions"] += len(sequence) - i
                    # break

                event = sequence[i]
                actual_next_activity = event.get("activity")

                if mode != "incremental":
                    msg = f"Invalid mode: {mode}. Use 'incremental' ('sequential' is deprecated)."
                    raise ValueError(msg)

                    # # Prediction for sequence mode (whole sequence)
                    # metrics = self.sequence_metrics(sequence[:i])
                    # prediction = metrics_prediction(metrics, config=self.config)

                # Prediction for incremental mode (step by step)
                metrics = self.state_metrics(current_state)
                prediction = metrics_prediction(metrics, config=self.config)
                predicted_sequence += prediction["activity"].__str__() if prediction else "-"

                pause_start_time = time.time()

                logger.debug("      [Before] Likelihood: %s", likelihood)
                if log_likelihood:
                    likelihood += math.log(self.state_act_likelihood(current_state, actual_next_activity))
                else:
                    likelihood *= self.state_act_likelihood(current_state, actual_next_activity)
                logger.debug("      [After] Likelihood: %s", likelihood)

                # likelihood *= (
                #     metrics["likelihood"]
                #     if metrics["likelihood"] is not None
                #     else 0.0
                # )

                logger.debug("State: %s", current_state)
                logger.debug("Actual next activity: %s", actual_next_activity)
                logger.debug("Prediction: %s", prediction)

                logger.debug("Metrics: %s", metrics)
                # Update statistics based on the prediction
                self.update_stats(event, prediction, current_state)

                pause_time += time.time() - pause_start_time

                # Move to the next state
                if i < len(sequence) - 1:
                    current_state = self.next_state(current_state, actual_next_activity)
                    logger.debug("Next state: %s", current_state)

            # Normalize by the length of the sequence
            if log_likelihood:
                normalized_likelihood = likelihood / len(sequence) if len(sequence) > 0 else likelihood
            else:
                normalized_likelihood = likelihood ** (1 / len(sequence)) if len(sequence) > 0 else likelihood

            seq_perplexity = StreamingMiner.compute_seq_perplexity(normalized_likelihood, log_likelihood=log_likelihood)
            perplexities.append(seq_perplexity)

            logger.debug("Pred. sequence: %s", predicted_sequence.replace(self.config["stop_symbol"].__str__(), "S"))
            logger.debug("Sequence likelihood: %s", likelihood)
            logger.debug("Sequence perplexity: %s", seq_perplexity)
            logger.debug("===== End Evaluating Sequence =====")

        perplexity_stats = compute_perplexity_stats(perplexities)
        logger.debug("Perplexity stats: %s", perplexity_stats)

        for key, value in perplexity_stats.items():
            self.stats[key] = value

        return time.time() - eval_start_time - pause_time

    @abstractmethod
    def get_state_info(self, state_id: StateId) -> ComposedState | None:
        """Return the state information of the algorithm."""

    @abstractmethod
    def get_modified_cases(self) -> set[CaseId]:
        """Retrieve, recursively, cases that have potentially been modified and whose prediction needs to be updated."""

    @abstractmethod
    def propagate_config(self) -> None:
        """Recursively propagates the config to all nested models."""

    @abstractmethod
    def update(self, event: Event) -> None:
        """Update Strategy."""

    @abstractmethod
    def next_state(self, current_state: ComposedState | None, activity: ActivityName) -> ComposedState | None:
        """Take a transition from the current state."""

    @abstractmethod
    def state_metrics(self, state: ComposedState | None) -> Metrics:
        """Return metrics dictionary based on state."""

    @abstractmethod
    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return metrics dictionary based on case."""

    @abstractmethod
    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return metrics dictionary based on sequence."""

    @abstractmethod
    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the given activity given a current state."""


# ============================================================
# Standard Streaming Miner (using one building block)
# ============================================================


class BasicMiner(StreamingMiner):
    """The Basic Miner is a wrapper for a single algorithm."""

    def __init__(self, *args: dict[str, Any], algorithm: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the BasicMiner with a specific algorithm."""
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm

        if self.algorithm is None:
            msg = "An algorithm must be specified."
            raise ValueError(msg)

        # Propagate self.config to the algorithm
        self.propagate_config()

        self.initial_state = self.algorithm.initial_state

    def __str__(self) -> str:
        """Return a string representation of the BasicMiner."""
        return f"BasicMiner({self.algorithm})"

    def get_num_states(self) -> int:
        """Return the number of states in the algorithm."""
        return len(self.algorithm.states)

    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the given activity given a current state."""
        if hasattr(self.algorithm, "state_act_likelihood"):
            return self.algorithm.state_act_likelihood(state, next_activity)
        logger.debug("[!!!]   No state_act_likelihood available for this miner: %s", self)
        return 0.0

    def get_modified_cases(self) -> set[CaseId]:
        """Retrieve, recursively, cases that have potentially been modified and whose prediction needs to be updated."""
        return self.algorithm.get_modified_cases()

    def propagate_config(self) -> None:
        """Recursively propagates the config to all nested models."""
        self.algorithm.config = self.config

    def update(self, event: Event) -> None:
        """Update the algorithm with the new event."""
        self.algorithm.update(event)
        self.modified_cases = self.algorithm.get_modified_cases()

    def next_state(self, current_state: ComposedState | None, activity: ActivityName) -> ComposedState | None:
        """Take a transition from the current state."""
        return self.algorithm.next_state(current_state, activity)

    def get_state_info(self, state_id: StateId | None) -> ComposedState | None:
        """Return the state information of the algorithm."""
        if state_id is None:
            return None
        return self.algorithm.state_info[state_id] if hasattr(self.algorithm, "state_info") else None

    def state_metrics(self, state: ComposedState | None) -> Metrics:
        """Return metrics dictionary based on state."""
        return self.algorithm.state_metrics(state)

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return metrics dictionary based on case."""
        return self.algorithm.case_metrics(case_id)

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return metrics dictionary based on sequence."""
        return self.algorithm.sequence_metrics(sequence)

    # def sequence_likelihood(self, sequence: list[Event]) -> float:
    #     """Return the likelihood of the sequence."""
    #     if hasattr(self.algorithm, "sequence_likelihood"):
    #         return self.algorithm.sequence_likelihood(sequence)
    #     logger.debug("No sequence likelihood available for this miner: %s", self)
    #     return 0.0


# ============================================================
# Multi Streaming Miner (using several building blocks)
# ============================================================


class MultiMiner(StreamingMiner, ABC):
    """The Multi Miner is a wrapper for several algorithms."""

    def __init__(
        self,
        *args: dict[str, Any] | None,
        models: list[StreamingMiner],
        delay_weights: list[float] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the MultiMiner with a list of models."""
        super().__init__(*args, **kwargs)
        self.models = models

        self.propagate_config()

        self.initial_state = tuple(model.initial_state for model in self.models)

        num_models = len(self.models)

        if delay_weights is not None:
            if len(delay_weights) != num_models or any(w < 0 for w in delay_weights):
                msg = "Delay weights do not meet specification."
                raise ValueError(msg)
        else:
            delay_weights = [1.0] * num_models  # Default to uniform weights

        self.delay_weights = delay_weights

    def get_modified_cases(self) -> set[CaseId]:
        """Retrieve, recursively, cases that have potentially been modified and whose prediction needs to be updated."""
        modified_cases = set()

        for model in self.models:
            # Add modified cases from the current model
            modified_cases.update(model.get_modified_cases())

        return modified_cases

    def propagate_config(self) -> None:
        """Recursively propagates the config to all nested models."""
        for model in self.models:
            model.config = self.config
            model.propagate_config()

    def voting_delays(self, delays_list: list[ActivityDelays]) -> ActivityDelays:
        """Return the weighted average of the predicted delays for each activity."""
        combined_delays = {}
        weight_sums = {}

        # Accumulate weighted delays
        for predicted_delays, weight in zip(delays_list, self.delay_weights, strict=True):
            for activity, delay in predicted_delays.items():
                if activity not in combined_delays:
                    combined_delays[activity] = timedelta(0)  # Initialize as timedelta
                    weight_sums[activity] = 0.0
                combined_delays[activity] += delay * weight
                weight_sums[activity] += weight

        # If there are no activities, return an empty dictionary
        if not combined_delays:
            return {}

        # Compute the weighted average delay for each activity
        return {
            activity: combined_delays[activity] / weight_sums[activity]
            for activity in combined_delays
            if weight_sums[activity] > 0
        }

    def get_state_info(self, state_id: ComposedState | None) -> ComposedState | None:
        """Return the state information of the algorithm."""
        ### QUESTION: WHAT IS THIS FOR? TESTED?
        if not isinstance(state_id, tuple) or state_id is None:
            return None
        return tuple(
            model.get_state_info(model_state) if hasattr(model, "get_state_info") else None
            for model, model_state in zip(self.models, state_id, strict=True)
        )

    def update(self, event: Event) -> None:
        """Update the algorithm with the new event."""
        self.modified_cases = set()

        for model in self.models:
            model.update(event)

        self.modified_cases = set()
        for model in self.models:
            self.modified_cases.update(model.get_modified_cases())

    def next_state(self, current_state: ComposedState | None, activity: ActivityName) -> ComposedState | None:
        """Take a transition from the current state."""
        if current_state is None:
            return None

        # Unpack the current state for each model
        next_states = [
            model.next_state(state, activity) for model, state in zip(self.models, current_state, strict=True)
        ]

        # If all next states are None, return None
        if all(ns is None for ns in next_states):
            return None

        # Otherwise, return the tuple of next states
        return tuple(next_states)


# ============================================================
# Ensemble Methods Derived from MultiMiner
# ============================================================


class HardVoting(MultiMiner):
    """The Hard Voting class implements a hard voting mechanism for ensemble learning."""

    def __init__(self, *args: dict[str, Any] | None, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the HardVoting class."""
        super().__init__(*args, **kwargs)

    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the sequence based on the metrics from the models.

        Calculates the total probability that the predicted activity will be the
        dominant outcome when considering all possible voting combinations
        weighted by their respective probabilities.
        """
        # We need to consider every possible combination of votes from all models
        # Each model can vote for any activity with its associated probability
        if state is None:
            return 0.0

        def get_winning_probability(vote_combination: tuple[ActivityName, ...]) -> float:
            """Determine if the predicted activity wins given a specific vote combination.

            Return the probability of this combination occurring.
            """
            # Count weighted votes for each activity
            vote_counter = Counter()
            combination_probability = 1.0

            for model_idx, activity in enumerate(vote_combination):
                vote_counter[activity] += 1
                # Multiply by the probability of this model voting for this activity
                combination_probability *= self.models[model_idx].state_act_likelihood(
                    state[model_idx] if state else None, activity
                )

            # Find activities with the most votes
            most_votes = max(vote_counter.values()) if vote_counter else 0
            winners = [activity for activity, count in vote_counter.items() if count == most_votes]

            # Check if the predicted activity wins (including tie resolution)
            if len(winners) == 1:
                # Unique winner case
                if next_activity == winners[0]:
                    return combination_probability
            else:
                # Tie case - resolve by model order
                # Sort winners based on first occurrence in the vote_combination
                winners_first_index = {winner: vote_combination.index(winner) for winner in winners}
                first_winner = min(winners, key=lambda w: winners_first_index[w])
                if next_activity == first_winner:
                    return combination_probability
            return 0.0

        # Generate all possible vote combinations using cartesian product
        # Each model can vote for any activity in its probability distribution
        model_activities = []
        metrics_list = [model.state_metrics(model_state) for model, model_state in zip(self.models, state, strict=True)]
        # logger.debug("Metrics list: %s", metrics_list)
        for metrics in metrics_list:
            # Only include activities with non-zero probability
            # logger.debug("Model metrics: %s", metrics)
            activities = [activity for activity, prob in metrics["probs"].items() if prob > 0.0]
            model_activities.append(activities)

        # If any model has no valid activities, return 0
        if any(not activities for activities in model_activities):
            # logger.debug("No valid activities for models: %s", model_activities)
            return 0.0

        # Calculate total probability across all possible voting combinations
        total_probability = 0.0
        for vote_combination in itertools.product(*model_activities):
            # logger.debug("New total probability: %s", total_probability)
            total_probability += get_winning_probability(vote_combination)

        return total_probability

    def voting_probs(self, probs_list: list[ProbDistr]) -> ProbDistr:
        """Perform hard voting based on the most frequent activity in the predictions and return the winning activity.

        As a probability dictionary with a probability of 1.0.
        If there is a tie, select the activity based on the first occurrence in the order of the models.
        """
        # Collect valid predictions
        valid_predictions = []

        for probs in probs_list:
            prediction = probs_prediction(probs, config=self.config)
            if prediction is not None:
                valid_predictions.append(prediction)

        if len(valid_predictions) == 0:
            return {}

        # Extract only the activity part of each valid prediction for voting
        activity_predictions = [pred["activity"] for pred in valid_predictions]

        # Count the frequency of each activity in the valid predictions
        activity_counter = Counter(activity_predictions)

        # Find the activity(s) with the highest count
        most_common = activity_counter.most_common()  # List of (activity, count) sorted by frequency

        # Get the highest count
        highest_count = most_common[0][1]
        most_voted_activities = [activity for activity, count in most_common if count == highest_count]

        selected_activity = self.config["stop_symbol"]

        # If there is only one activity with the highest count, select that activity
        if len(most_voted_activities) == 1:
            selected_activity = most_voted_activities[0]
        else:
            # In case of a tie, choose based on the first occurrence among the models' input
            for pred in valid_predictions:
                if pred["activity"] in most_voted_activities:
                    selected_activity = pred["activity"]
                    break

        # Create a result dictionary with only the selected activity
        return {self.config["stop_symbol"]: 0.0, selected_activity: 1.0}  # include STOP as an invariant

    def state_metrics(self, state: ComposedState | None) -> Metrics:
        """Return the majority vote."""
        if state is None:
            return empty_metrics()

        metrics_list = [model.state_metrics(model_state) for model, model_state in zip(self.models, state, strict=True)]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        delays_list = [
            model.state_metrics(model_state)["predicted_delays"]
            for model, model_state in zip(self.models, state, strict=True)
        ]

        return Metrics(
            probs=self.voting_probs(probs_list),
            predicted_delays=self.voting_delays(delays_list),
        )

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return the hard voting of predictions from the ensemble."""
        metrics_list = [model.case_metrics(case_id) for model in self.models]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        delays_list = [model.case_metrics(case_id)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=self.voting_probs(probs_list),
            predicted_delays=self.voting_delays(delays_list),
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the majority vote."""
        metrics_list = [model.sequence_metrics(sequence) for model in self.models]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        delays_list = [model.sequence_metrics(sequence)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=self.voting_probs(probs_list),
            predicted_delays=self.voting_delays(delays_list),
        )


class SoftVoting(MultiMiner):
    """Soft voting based on weighted probabilities."""

    def __init__(self, *args: dict[str, Any], prob_weights: list[float] | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the SoftVoting class with probability weights."""
        super().__init__(*args, **kwargs)

        # Validate the lengths of the weights if provided
        num_models = len(self.models)

        if prob_weights is not None:
            if len(prob_weights) != num_models or any(w < 0 for w in prob_weights):
                msg = "Probability weights do not meet specification."
                raise ValueError(msg)
        else:
            prob_weights = [1.0] * num_models  # Default to uniform weights

        self.prob_weights = prob_weights

    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the sequence based on the metrics from the models.

        Calculates the total probability that the predicted activity will be the
        dominant outcome when considering all possible voting combinations
        weighted by their respective probabilities.
        """
        if state is None:
            return 0.0
        metrics_list = [model.state_metrics(model_state) for model, model_state in zip(self.models, state, strict=True)]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        voting_probs = self.voting_probs(probs_list)
        return voting_probs.get(next_activity, 0.0)

    def voting_probs(self, probs_list: list[ProbDistr]) -> ProbDistr:
        """Return the weighted average of the predicted probabilities for each activity."""
        combined_probs = {}

        # Accumulate weighted probabilities
        for prob_dict, weight in zip(probs_list, self.prob_weights, strict=True):
            for activity, prob in prob_dict.items():
                if activity not in combined_probs:
                    combined_probs[activity] = 0.0
                combined_probs[activity] += weight * prob

        # If there are no activities, return an empty dictionary
        if not combined_probs:
            return {}

        # Normalize the combined probabilities so that they sum to 1
        total_prob = sum(combined_probs.values())
        if total_prob > 0:
            combined_probs = {activity: prob / total_prob for activity, prob in combined_probs.items()}

        return combined_probs

    def voting_likelihood(self, metrics: Metrics) -> float:
        """Return the likelihood of the sequence based on the metrics from the models.

        Calculates the total probability that the predicted activity will be the
        dominant outcome when considering all possible voting combinations
        weighted by their respective probabilities.
        """
        prediction = metrics_prediction(metrics, config=self.config)
        if prediction is None:
            return 0.0
        return prediction.get("probability", 0.0)

    def state_metrics(self, state: ComposedState | None) -> Metrics:
        """Return the majority vote."""
        if state is None:
            return empty_metrics()

        metrics_list = [model.state_metrics(model_state) for model, model_state in zip(self.models, state, strict=True)]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        voting_probs = self.voting_probs(probs_list)
        delays_list = [
            model.state_metrics(model_state)["predicted_delays"]
            for model, model_state in zip(self.models, state, strict=True)
        ]

        return Metrics(
            probs=voting_probs,
            predicted_delays=self.voting_delays(delays_list),
        )

        # output_metrics["likelihood"] = self.voting_likelihood(output_metrics)

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return the hard voting of predictions from the ensemble."""
        metrics_list = [model.case_metrics(case_id) for model in self.models]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        delays_list = [model.case_metrics(case_id)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=self.voting_probs(probs_list),
            predicted_delays=self.voting_delays(delays_list),
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the majority vote."""
        metrics_list = [model.sequence_metrics(sequence) for model in self.models]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        delays_list = [model.sequence_metrics(sequence)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=self.voting_probs(probs_list),
            predicted_delays=self.voting_delays(delays_list),
        )


class AdaptiveVoting(MultiMiner):
    """To be used only in streaming mode.

    In batch mode, it will stick to the model with the highest training accuracy.
    """

    total_predictions: int
    correct_predictions: list[int]

    def __init__(self, *args: dict[str, Any], **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the AdaptiveVoting class."""
        super().__init__(*args, **kwargs)
        # Initialize prediction tracking for each model
        self.total_predictions = 0
        self.correct_predictions = [0] * len(self.models)

    def update(self, event: Event) -> None:
        """Overwritten to account for keeping track of accuracies in streaming mode."""
        case_id = event["case_id"]
        activity = event["activity"]

        self.total_predictions += 1

        for i, model in enumerate(self.models):
            prediction = probs_prediction(model.case_metrics(case_id)["probs"], config=self.config)
            if prediction is not None and prediction["activity"] == activity:
                self.correct_predictions[i] += 1

            model.update(event)

        self.modified_cases = set()
        for model in self.models:
            self.modified_cases.update(model.get_modified_cases())

    def get_accuracies(self) -> list[float]:
        """Return the accuracy of each model as a list of floats."""
        total = self.total_predictions
        return [correct / total if total > 0 else 0.0 for correct in self.correct_predictions]

    def select_best_model(self) -> int:
        """Return the index of the model with the highest accuracy."""
        accuracies = self.get_accuracies()
        return accuracies.index(max(accuracies))

    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the sequence based on the metrics from the models.

        Calculates the total probability that the predicted activity will be the
        dominant outcome when considering all possible voting combinations
        weighted by their respective probabilities.
        """
        if state is None:
            return 0.0

        # Get the best model
        best_model_index = self.select_best_model()
        best_model = self.models[best_model_index]

        return best_model.state_act_likelihood(state[best_model_index], next_activity)

    def state_metrics(self, state: ComposedState | None) -> Metrics:
        """Return the probability distribution from the model with the best accuracy so far."""
        if state is None:
            return empty_metrics()

        # Get the best model
        best_model_index = self.select_best_model()
        best_model = self.models[best_model_index]

        best_model_state = state[best_model_index]

        delays_list = [
            model.state_metrics(model_state)["predicted_delays"]
            for model, model_state in zip(self.models, state, strict=True)
        ]

        return Metrics(
            probs=best_model.state_metrics(best_model_state)["probs"],
            predicted_delays=self.voting_delays(delays_list),
        )

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return the probability distribution from the model with the best accuracy so far."""
        # Get the best model
        best_model_index = self.select_best_model()
        best_model = self.models[best_model_index]

        delays_list = [model.case_metrics(case_id)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=best_model.case_metrics(case_id)["probs"],
            predicted_delays=self.voting_delays(delays_list),
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the probability distribution from the model with the best accuracy so far."""
        # Get the best model
        best_model_index = self.select_best_model()
        best_model = self.models[best_model_index]

        delays_list = [model.sequence_metrics(sequence)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=best_model.sequence_metrics(sequence)["probs"],
            predicted_delays=self.voting_delays(delays_list),
        )


# ============================================================
# Other Models Derived from Multi Streaming Miner
# ============================================================


class Fallback(MultiMiner):
    """Fallback model (Backoff)."""

    def __init__(self, *args: dict[str, Any], **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the Fallback model."""
        super().__init__(*args, **kwargs)

    def state_metrics(self, state: ComposedState | None) -> Metrics:
        """Return the first non-{} probabilities from the models, cascading through the models in order.

        Each model gets its corresponding state from the ComposedState.
        """
        if state is None:
            return empty_metrics()

        delays_list = [
            model.state_metrics(model_state)["predicted_delays"]
            for model, model_state in zip(self.models, state, strict=True)
        ]

        # Iterate through the models and their corresponding states
        for model, model_state in zip(self.models, state, strict=True):
            metrics = model.state_metrics(model_state)
            if metrics["probs"]:
                return Metrics(
                    probs=metrics["probs"],
                    predicted_delays=self.voting_delays(delays_list),
                )

        # If all models return empty metrics
        return empty_metrics()

    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the given activity given a current state."""
        if state is None:
            return 0.0

        for model, model_state in zip(self.models, state, strict=True):
            logger.debug("Trying model for likelihood")
            activity_likelihood = model.state_act_likelihood(model_state, next_activity)

            if activity_likelihood > 0:
                logger.debug("Chose model with likelihood %s", activity_likelihood)
                return activity_likelihood

        return 0.0

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return the first non-{} probabilities from the models, cascading through the models in order."""
        delays_list = [model.case_metrics(case_id)["predicted_delays"] for model in self.models]

        for model in self.models:
            metrics = model.case_metrics(case_id)
            if metrics["probs"]:
                msg = f"Fallback chooses model {model} with metrics {metrics}"
                logger.debug(msg)
                return Metrics(
                    probs=metrics["probs"],
                    predicted_delays=self.voting_delays(delays_list),
                )

        # If all models return {}
        return empty_metrics()

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the first non-empty probabilities from the models for the given sequence.

        Cascading through the models in order.
        """
        delays_list = [model.sequence_metrics(sequence)["predicted_delays"] for model in self.models]

        for model in self.models:
            metrics = model.sequence_metrics(sequence)
            logger.debug("Trying model with metrics %s", metrics)
            if metrics["probs"]:
                logger.debug("Chose model with metrics %s", metrics)
                return Metrics(
                    probs=metrics["probs"],
                    predicted_delays=self.voting_delays(delays_list),
                )

        # If all models return {}
        return empty_metrics()


class Relativize(MultiMiner):
    """Relativize the probabilities of two models."""

    def __init__(self, *args: dict[str, Any], **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the Relativize class."""
        super().__init__(*args, **kwargs)
        if len(self.models) != 2:  # noqa: PLR2004
            msg = "Class Relativize requires two models."
            raise ValueError(msg)

        self.model1 = self.models[0]
        self.model2 = self.models[1]

    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the given activity given a current state."""
        msg = "Not implemented for Relativize"
        raise NotImplementedError(msg)

    def state_metrics(self, state: ComposedState | None) -> Metrics:
        """Return the first non-{} probabilities from the models, cascading through the models in order."""
        if state is None:
            return empty_metrics()

        (state1, state2) = state

        probs = self.model1.state_metrics(state1)["probs"]

        if probs:
            probs = self.model2.state_metrics(state2)["probs"]

        delays_list = [
            model.state_metrics(model_state)["predicted_delays"]
            for model, model_state in zip(self.models, state, strict=True)
        ]

        return Metrics(
            probs=probs,
            predicted_delays=self.voting_delays(delays_list),
        )

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return the first non-{} probabilities from the models, cascading through the models in order."""
        probs = self.model1.case_metrics(case_id)["probs"]

        if probs:
            probs = self.model2.case_metrics(case_id)["probs"]

        delays_list = [model.case_metrics(case_id)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=probs,
            predicted_delays=self.voting_delays(delays_list),
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the first non-{} probabilities from the models for the given sequence."""
        probs = self.model1.sequence_metrics(sequence)["probs"]

        if probs:
            probs = self.model2.sequence_metrics(sequence)["probs"]

        delays_list = [model.sequence_metrics(sequence)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=probs,
            predicted_delays=self.voting_delays(delays_list),
        )


# ============================================================
# Alergia
# ============================================================


class Alergia(BasicMiner):
    """Alergia miner for probabilistic automata."""

    def __init__(self, *args: dict[str, Any], **kwargs: dict[str, Any]) -> None:
        """Initialize the Alergia miner."""
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        """Return a string representation of the Alergia miner."""
        return f"Alergia({self.algorithm})"

    @staticmethod
    def get_probability_distribution(state: ComposedState) -> ProbDistr:
        """Return the probability distribution of the state.

        The state should be a probabilistic automaton.
        """
        probability_distribution = {}

        for input_symbol, transitions in state.transitions.items():
            # Create a dictionary mapping output letters to probabilities for this input symbol
            output_probabilities = {transition[1]: transition[2] for transition in transitions}
            probability_distribution[input_symbol] = output_probabilities

        return probability_distribution["in"]

    def get_modified_cases(self) -> set[CaseId]:
        """Not implemented."""
        return set()

    def update(self, event: Event) -> None:
        """Methods not used in this subclass."""

    def state_metrics(self, state: ComposedState) -> Metrics:
        """Return the probability distribution of the state.

        The state should be a probabilistic automaton.
        """
        return Metrics(
            probs=self.get_probability_distribution(state),
            predicted_delays={},
        )

    def case_metrics(self, case_id: CaseId) -> Metrics:  # noqa: ARG002
        """Methods not used in this subclass."""
        return empty_metrics()

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the probability distribution of the state.

        The state should be a probabilistic automaton.
        """
        transformed_sequence = add_input_symbols_sequence(sequence, "in")

        self.algorithm.reset_to_initial()

        for symbol in transformed_sequence:
            self.algorithm.step_to(symbol[0], symbol[1])

        # Get probability distribution for the current state
        return Metrics(probs=self.get_probability_distribution(self.algorithm.current_state), predicted_delays={})

    def next_state(self, current_state: ComposedState, activity: ActivityName) -> ComposedState:
        """Take a transition from the current state."""
        self.algorithm.current_state = current_state
        self.algorithm.step_to("in", activity)
        return self.algorithm.current_state


# ============================================================
# Neural Network Streaming Miner (RNN and LSTM)
# ============================================================


class NeuralNetworkMiner(StreamingMiner):
    device: torch.device | None

    def __init__(self, *args, model: RNNModel | LSTMModel, batch_size: int, optimizer, criterion, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = model.device
        self.model = model.to(device=self.device)  # The neural network, make sure it's at the device
        self.optimizer = optimizer
        self.criterion = criterion

        self.sequences: OrderedDict[CaseId, list[int]] = OrderedDict()  # Ordered dictionary to maintain insertion order
        self.rr_index = 0  # Keeps track of the round-robin index
        self.batch_size = batch_size

        self.activity_index = {}
        self.index_activity = {}

    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the sequence based on the metrics from the models."""
        # WARNING: This method is not implemented in the neural network miner.
        return -1.0

    def get_state_info(self, state_id: StateId) -> ComposedState | None:
        """Return the state information of the algorithm."""
        ### QUESTION: WHAT IS THIS FOR? TESTED?
        return None

    def get_sequence(self, case_id: CaseId) -> list[int]:
        """Return the index sequence for a specific case_id."""
        return self.sequences.get(case_id, [])

    def get_modified_cases(self) -> set[CaseId]:
        """Not implemented."""
        return set()

    def update(self, event: Event) -> None:
        """Add an activity to the sequence corresponding to the case_id.

        Dynamically update the activity_to_idx mapping if a new activity is encountered.
        """
        case_id = event["case_id"]
        activity = event["activity"]

        # Dynamically update activity_to_idx if the activity is new
        if activity not in self.activity_index:
            current_idx = len(self.activity_index) + 1  # Get the next available index
            self.activity_index[activity] = current_idx
            self.index_activity[current_idx] = activity

        # Convert activity to its corresponding index
        activity_idx = self.activity_index[activity]

        # Add the activity index to the sequence for the given case_id
        if case_id not in self.sequences:
            self.sequences[case_id] = []  # New case added
        self.sequences[case_id].append(activity_idx)

        # Continue with the training step using the updated sequence
        batch = self.select_batch(case_id)

        # Ensure each sequence in the batch has at least two tokens
        if len(batch) == 0:
            msg = "Skipping training step because no valid sequences were found."
            logger.info(msg)
            return None

        # Set model to training mode
        self.model.train()

        # Convert the batch of sequences into tensors, padding them to the same length
        batch_sequences = [torch.tensor(seq, dtype=torch.long, device=self.device) for seq in batch]
        x_batch = pad_sequence(batch_sequences, batch_first=True, padding_value=0)

        # Input is all but the last token in each sequence, target is shifted by one position
        x_input = x_batch[:, :-1]  # Input sequence
        y_target = x_batch[:, 1:].reshape(-1)  # Flatten the target for CrossEntropyLoss

        self.optimizer.zero_grad()

        # Forward pass through the model
        outputs = self.model(x_input)

        # Reshape outputs to [batch_size * sequence_length, vocab_size] for loss calculation
        outputs = outputs.view(-1, outputs.shape[-1])

        # Create a mask to ignore padding (y_target == 0)
        mask = y_target != 0  # Mask out padding positions

        # Apply the mask
        outputs = outputs[mask]
        y_target = y_target[mask]

        # Compute loss
        loss = self.criterion(outputs, y_target)

        # Backward pass and gradient clipping
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def select_batch(self, case_id: CaseId) -> list[list[int]]:
        """Select a batch of sequences, using a round-robin approach.

        Only select sequences that have at least two tokens (input + target).
        """
        valid_case_ids = [cid for cid, sequence in self.sequences.items() if len(sequence) > 1]

        if len(valid_case_ids) < self.batch_size:
            msg = f"Not enough case_ids to form a full batch, using {len(valid_case_ids)} case_ids."
            logger.info(msg)
            return [self.get_sequence(cid) for cid in valid_case_ids]  # Return all valid sequences

        # Prepare the batch, starting with the current case_id
        batch_case_ids = [case_id] if len(self.sequences[case_id]) > 1 else []

        original_rr_index = self.rr_index  # Save the original index to detect when we complete a full cycle
        count = 0

        # Batch size - 1 if we've already added current case_id
        required_cases = self.batch_size - 1 if batch_case_ids else self.batch_size

        # Select additional case_ids in a round-robin manner, skipping the current case_id
        while count < required_cases:
            candidate_case_id = valid_case_ids[self.rr_index]

            # Skip the current case_id
            if candidate_case_id != case_id and len(self.sequences[candidate_case_id]) > 1:
                batch_case_ids.append(candidate_case_id)
                count += 1

            # Move to the next index, wrap around if necessary
            self.rr_index = (self.rr_index + 1) % len(valid_case_ids)

            # Stop if we've completed a full round (returning to original index)
            if self.rr_index == original_rr_index:
                break

        # batch = [self.get_sequence(cid) for cid in batch_case_ids]

        # Fetch the actual sequences based on the selected case_ids
        return [self.get_sequence(cid) for cid in batch_case_ids]

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Predict the next activity for a given case_id.

        Return the top-k most likely activities along with the probability
        of the top activity.
        Note that, here, a sequence is a sequence of activity indices (rather than activities).
        """
        # Get the sequence for the case_id
        index_sequence = self.get_sequence(case_id)

        if not index_sequence or len(index_sequence) < 1:
            return empty_metrics()

        return Metrics(
            probs=self.idx_sequence_probs(index_sequence),
            predicted_delays={},
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Predict the next activity for a given sequence of activities and return the top-k most likely activities...

        ...along with the
        probability of the top activity.
        """
        if not sequence or len(sequence) < 1:
            return empty_metrics()

        # Convert each activity name to its corresponding index, return None if any activity is unknown
        index_sequence = []
        for event in sequence:
            activity = event["activity"]
            activity_idx = self.activity_index.get(activity)
            if activity_idx is None:
                return empty_metrics()
            index_sequence.append(activity_idx)

        return Metrics(
            probs=self.idx_sequence_probs(index_sequence),
            predicted_delays={},
        )

    def idx_sequence_probs(self, index_sequence: list[int]) -> ProbDistr:
        """Predict the next activity for a given sequence of activity indices."""
        # Convert to a tensor and add a batch dimension
        input_sequence = torch.tensor(index_sequence, dtype=torch.long, device=self.device).unsqueeze(
            0
        )  # Shape [1, sequence_length]

        # Pass the sequence through the model to get the output
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_sequence)

        # Get the logits for the last time step (most recent activity in the sequence)
        logits = output[:, -1, :]  # Shape [1, vocab_size]

        # Apply softmax to get the probabilities
        probabilities = torch.softmax(logits, dim=-1)  # Shape [1, vocab_size]

        # Convert the tensor to a list of probabilities
        probabilities = probabilities.squeeze(0).tolist()  # Shape [vocab_size]

        return {
            self.index_activity[idx]: prob
            for idx, prob in enumerate(probabilities)
            if self.index_activity.get(idx) is not None
        }

    def next_state(self, *args: Any, **kwargs: Any) -> ComposedState:  # noqa: ANN401
        """Not implemented."""
        msg = "Not implemented for NeuralNetworkMiner."
        raise NotImplementedError(msg)

    def propagate_config(self) -> None:
        """Not implemented."""
        msg = "Not implemented for NeuralNetworkMiner."
        raise NotImplementedError(msg)

    def state_metrics(self, state: ComposedState | None) -> Metrics:
        """Not implemented."""
        msg = "Not implemented for NeuralNetworkMiner."
        raise NotImplementedError(msg)
