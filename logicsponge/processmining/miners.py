"""
Module for streaming miners.

This module contains the implementation of various streaming miners,
including the BasicMiner, MultiMiner, HardVoting, SoftVoting, AdaptiveVoting,
and Fallback classes.
"""

import itertools
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, defaultdict, deque
from datetime import timedelta
from typing import Any

import matplotlib as mpl
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from logicsponge.processmining.config import DEFAULT_CONFIG, update_config
from logicsponge.processmining.data_utils import add_input_symbols_sequence
from logicsponge.processmining.neural_networks import (
    GRUModel,
    LSTMModel,
    QNetwork,
    RNNModel,
    TransformerModel,
    # Use left-padding helper to align streaming NN batches with batch-mode training/eval
    _left_pad_stack,
)
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
from logicsponge.processmining.utils import (
    compute_perplexity_stats,
    compute_seq_perplexity,
    metrics_prediction,
    probs_prediction,
)

mpl.use("Agg")

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line-wrapping # noqa: FBT003

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
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

    def update_stats(self, event: Event, prediction: Prediction | None, state_id: StateId | None) -> None:  # noqa: C901, PLR0912
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

    def evaluate(  # noqa: C901, PLR0912, PLR0915
        self,
        data: list[list[Event]],
        mode: str = "incremental",  # noqa: ARG002
        *,
        log_likelihood: bool = False,
        compute_perplexity: bool = False,
        debug: bool = False,  # noqa: ARG002
    ) -> tuple[float, list[ActivityName]]:
        """
        Evaluate in batch mode.

        Evaluate the dataset either incrementally or by full sequence.

        Modes: 'incremental' or 'sequence'.

        NOTE: now returns a tuple (elapsed_time, predicted_vector) where predicted_vector is a
        flattened list of predicted next-activities in the same order they were produced.
        """
        # Initialize stats
        perplexities = []

        eval_start_time = time.time()
        pause_time = 0.0

        # Collect flattened predicted next-activities for incremental mode
        predicted_vector: list[ActivityName] = []

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
            likelihood = 0.0 if (log_likelihood or not compute_perplexity) else 1.0

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

                # Prediction for incremental mode (step by step)
                metrics = self.state_metrics(current_state)
                prediction = metrics_prediction(metrics, config=self.config)
                predicted_sequence += prediction["activity"].__str__() if prediction else "-"

                # Collect predicted activity (skip empty predictions) in order
                predicted_vector.append(
                    prediction["activity"] if prediction is not None else DEFAULT_CONFIG["empty_symbol"]
                )

                pause_start_time = time.time()

                if compute_perplexity:
                    logger.debug("      [Before] Likelihood: %s", likelihood)
                    if log_likelihood:
                        likelihood += math.log(self.state_act_likelihood(current_state, actual_next_activity))
                        # likelihood += metrics["likelihoods"].get(actual_next_activity, 0.0)
                    else:
                        likelihood *= self.state_act_likelihood(current_state, actual_next_activity)
                        # likelihood *= metrics["likelihoods"].get(actual_next_activity, 0.0)
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

                # ============================================================
                # Update statistics based on the prediction
                # -----------------------------------------
                self.update_stats(event, prediction, current_state)
                # ============================================================

                pause_time += time.time() - pause_start_time

                # Move to the next state
                if i < len(sequence) - 1:
                    current_state = self.next_state(current_state, actual_next_activity)
                    logger.debug("Next state: %s", current_state)

            if compute_perplexity:
                # Normalize by the length of the sequence
                if log_likelihood:
                    normalized_likelihood = likelihood / len(sequence) if len(sequence) > 0 else likelihood
                else:
                    normalized_likelihood = likelihood ** (1 / len(sequence)) if len(sequence) > 0 else likelihood

                seq_perplexity = compute_seq_perplexity(normalized_likelihood, log_likelihood=log_likelihood)
            else:
                seq_perplexity = float("nan")

            perplexities.append(seq_perplexity if compute_perplexity else likelihood)

            logger.debug("Pred. sequence: %s", predicted_sequence.replace(self.config["stop_symbol"].__str__(), "S"))
            logger.debug("Sequence likelihood: %s", likelihood)
            logger.debug("Sequence perplexity: %s", seq_perplexity)
            logger.debug("===== End Evaluating Sequence =====")

        perplexity_stats = compute_perplexity_stats(perplexities)
        logger.debug("Perplexity stats: %s", perplexity_stats)

        for key, value in perplexity_stats.items():
            self.stats[key] = value

        elapsed = time.time() - eval_start_time - pause_time
        # Return elapsed time and the flattened predicted vector (same-order predictions)
        return elapsed, predicted_vector

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
    def get_state_from_case(self, case_id: CaseId) -> ComposedState:
        """Return the state of the algorithm based on the case ID."""

    @abstractmethod
    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return metrics dictionary based on case."""

    # @abstractmethod
    # def sequence_metrics(self, sequence: list[Event]) -> Metrics:
    #     """Return metrics dictionary based on sequence."""

    @abstractmethod
    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the given activity given a current state."""

    def state_act_likelihoods(
        self, state: StateId | None, wanted_activities: list[ActivityName]
    ) -> dict[ActivityName, float]:
        """Return the likelihood of the wanted activities given a current state."""
        likelihoods = {}
        for activity in wanted_activities:
            likelihoods[activity] = self.state_act_likelihood(state, activity)
        return likelihoods


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

    def get_state_from_case(self, case_id: CaseId) -> StateId:
        """Return the state of the algorithm based on the case ID."""
        return self.algorithm.get_state_from_case(case_id)

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

    def get_state_from_case(self, case_id: CaseId) -> ComposedState:
        """Return the state of the algorithm based on the case ID."""
        # Get the state from each model
        return tuple([model.get_state_from_case(case_id) for model in self.models])

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
        """
        Return the likelihood of the sequence based on the metrics from the models.

        Calculates the total probability that the predicted activity will be the
        dominant outcome when considering all possible voting combinations
        weighted by their respective probabilities.
        """
        # We need to consider every possible combination of votes from all models
        # Each model can vote for any activity with its associated probability
        if state is None:
            return 0.0

        def get_winning_probability(vote_combination: tuple[ActivityName, ...]) -> float:
            """
            Determine if the predicted activity wins given a specific vote combination.

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
        """
        Perform hard voting based on the most frequent activity in the predictions and return the winning activity.

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
        # activity_list = [activity for metrics in metrics_list for activity in metrics["probs"]]

        return Metrics(
            state_id=tuple([metrics["state_id"] for metrics in metrics_list]),
            probs=self.voting_probs(probs_list),
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=self.state_act_likelihoods(state, activity_list)
        )

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return the hard voting of predictions from the ensemble."""
        # Non optimal, but it is convenient to call state_act_likelihood for the likelihood computation
        # So we need first to get the state of each model, although this is already done inside
        # each model.case_metrics() call
        # state = self.get_state_from_case(case_id)

        metrics_list = [model.case_metrics(case_id) for model in self.models]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        delays_list = [metrics["predicted_delays"] for metrics in metrics_list]

        # activity_list = [activity for metrics in metrics_list for activity in metrics["probs"]]

        return Metrics(
            state_id=tuple([metrics["state_id"] for metrics in metrics_list]),
            probs=self.voting_probs(probs_list),
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=self.state_act_likelihoods(state, activity_list)
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the majority vote."""
        msg = "Hard voting is not implemented for sequences."
        raise NotImplementedError(msg)
        metrics_list = [model.sequence_metrics(sequence) for model in self.models]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        delays_list = [metrics["predicted_delays"] for metrics in metrics_list]

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
        """
        Return the likelihood of the sequence based on the metrics from the models.

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
        """
        Return the likelihood of the sequence based on the metrics from the models.

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

        # activity_list = [activity for metrics in metrics_list for activity in metrics["probs"]]

        return Metrics(
            state_id=tuple([metrics["state_id"] for metrics in metrics_list]),
            probs=voting_probs,
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=self.state_act_likelihoods(state, activity_list)
        )

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return the hard voting of predictions from the ensemble."""
        # Non optimal, but it is convenient to call state_act_likelihood for the likelihood computation
        # So we need first to get the state of each model, although this is already done inside
        # each model.case_metrics() call
        # state = self.get_state_from_case(case_id)

        metrics_list = [model.case_metrics(case_id) for model in self.models]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        delays_list = [metrics["predicted_delays"] for metrics in metrics_list]

        # activity_list = [activity for metrics in metrics_list for activity in metrics["probs"]]

        return Metrics(
            state_id=tuple([metrics["state_id"] for metrics in metrics_list]),
            probs=self.voting_probs(probs_list),
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=self.state_act_likelihoods(state, activity_list)
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the majority vote."""
        msg = "Soft voting is not implemented for sequences."
        raise NotImplementedError(msg)
        metrics_list = [model.sequence_metrics(sequence) for model in self.models]
        probs_list = [metrics["probs"] for metrics in metrics_list]
        delays_list = [model.sequence_metrics(sequence)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=self.voting_probs(probs_list),
            predicted_delays=self.voting_delays(delays_list),
        )


class AdaptiveVoting(MultiMiner):
    """
    Selects the best model for each prediction.

    Args:
        select_best (str): The criterion to select the best model.
            - "acc": Selects the model with the highest accuracy.
            - "prob": Selects the model with the highest probability.
            - "prob x acc": Selects the model with the highest probability times accuracy.

    Note:
        - when using "acc", the model with the highest accuracy is selected.
        In streaming mode, this model can change over time, but in batch mode, it will
        stick to the model with the highest training accuracy.
        - when using "prob", the model with the highest probability of its prediction
        is selected (depending on the state).
        - when using "prob x acc", the model with the highest value of probability
        of its prediction, times the accuracy of the model, is selected.

    """

    total_predictions: int
    correct_predictions: list[int]
    offline_training: bool = False

    def __init__(self, *args: dict[str, Any], select_best: str = "acc", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the AdaptiveVoting class."""
        super().__init__(*args, **kwargs)
        # Initialize prediction tracking for each model
        self.total_predictions = 0
        self.correct_predictions = [0] * len(self.models)

        self.select_best = select_best
        accepted_options = ["acc", "prob", "prob x acc"]
        if self.select_best not in accepted_options:
            msg = f"select_best must be in {accepted_options}. Provided: {self.select_best}"
            raise ValueError(msg)

    def update(self, event: Event) -> None:
        """Overwritten to account for keeping track of accuracies in streaming mode."""
        case_id = event["case_id"]
        activity = event["activity"]

        if not self.offline_training:
            # Increment total and log before updating per-model counters
            prev_total = self.total_predictions
            self.total_predictions += 1
            logger.debug("AdaptiveVoting.update: total_predictions %d -> %d", prev_total, self.total_predictions)

        for i, model in enumerate(self.models):
            prediction = probs_prediction(model.case_metrics(case_id)["probs"], config=self.config)
            if not self.offline_training and prediction is not None and prediction["activity"] == activity:
                prev = self.correct_predictions[i]
                self.correct_predictions[i] += 1
                logger.debug(
                    "AdaptiveVoting.update: model %d predicted %s, actual %s, correct_predictions %d -> %d",
                    i,
                    prediction.get("activity"),
                    activity,
                    prev,
                    self.correct_predictions[i],
                )

            # still update the underlying model
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
        """
        Return the likelihood of the sequence based on the metrics from the models.

        Calculates the total probability that the predicted activity will be the
        dominant outcome when considering all possible voting combinations
        weighted by their respective probabilities.
        """
        if state is None:
            return 0.0

        # Get the best model
        best_model_index = (
            self.select_best_model() if self.select_best == "acc" else self.get_best_model_metrics(state)[0]
        )

        return (
            self.models[best_model_index].state_act_likelihood(state[best_model_index], next_activity)
            if best_model_index is not None
            else 0.0
        )

        # # Get the best model
        # best_model_index = self.select_best_model()
        # best_model = self.models[best_model_index]

        # return best_model.state_act_likelihood(state[best_model_index], next_activity)

    def get_best_model_metrics(self, state: ComposedState) -> tuple[int | None, Metrics]:
        """Return the metrics of the model with the best accuracy so far."""
        if state is None:
            return None, empty_metrics()

        all_state_metrics = [
            model.state_metrics(model_state) for model, model_state in zip(self.models, state, strict=True)
        ]

        # Select the metrics with the highest max probability and the index of the model
        return max(
            enumerate(all_state_metrics),
            key=lambda x: (
                (max(x[1]["probs"].values()) if x[1]["probs"] else 0.0)
                * (self.correct_predictions[x[0]] if self.select_best == "prob x acc" else 1.0)
            ),
        )

    def state_metrics(self, state: ComposedState | None) -> Metrics:
        """Return the probability distribution from the model with the best accuracy so far."""
        if state is None:
            return empty_metrics()

        # Get the best model
        if self.select_best == "acc":
            best_model_index = self.select_best_model()
            best_model = self.models[best_model_index]
            best_model_state = state[best_model_index]
            best_model_metrics = best_model.state_metrics(best_model_state)
        else:
            best_model_metrics = self.get_best_model_metrics(state)[1]

        delays_list = [
            model.state_metrics(model_state)["predicted_delays"]
            for model, model_state in zip(self.models, state, strict=True)
        ]

        return Metrics(
            state_id=state,
            probs=best_model_metrics["probs"],
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=best_model_metrics["likelihoods"]
        )

    def evaluate(  # noqa: C901, PLR0912, PLR0915
        self,
        data: list[list[Event]],
        mode: str = "incremental",  # noqa: ARG002
        *,
        log_likelihood: bool = False,
        compute_perplexity: bool = False,
        debug: bool = False,  # noqa: ARG002
    ) -> tuple[float, list[ActivityName]]:
        """
        Override evaluate to update per-model accuracy counters during batch evaluation.

        After each prediction this method queries every base model for its
        predicted top activity and increments `self.correct_predictions` for the
        models that predicted the real next activity. `self.total_predictions`
        is incremented accordingly so `get_accuracies()` reflects evaluation-time
        performance as the evaluation proceeds.
        """
        perplexities: list[float] = []

        eval_start_time = time.time()
        pause_time = 0.0

        predicted_vector: list[ActivityName] = []

        for sequence in tqdm(data, desc="Processing sequences"):
            pause_start_time = time.time()

            logger.debug(">>>>> Start Evaluating Sequence <<<<<")
            event_sequence = ""
            predicted_sequence = ""
            for event in sequence:
                event_sequence += event["activity"].__str__()
            logger.debug(
                "Event sequence: %s",
                event_sequence.replace(self.config["stop_symbol"].__str__(), "S"),
            )

            pause_time += time.time() - pause_start_time

            current_state = self.initial_state
            metrics = empty_metrics()
            likelihood = 0.0 if (log_likelihood or not compute_perplexity) else 1.0

            for i in range(len(sequence)):
                if current_state is None:
                    self.stats["empty_predictions"] += 1
                    self.stats["total_predictions"] += 1

                event = sequence[i]
                actual_next_activity = event.get("activity")

                # Prediction for incremental mode (step by step)
                metrics = self.state_metrics(current_state)
                prediction = metrics_prediction(metrics, config=self.config)
                predicted_sequence += prediction["activity"].__str__() if prediction else "-"

                # Collect predicted activity (skip empty predictions) in order
                predicted_vector.append(
                    prediction["activity"] if prediction is not None else DEFAULT_CONFIG["empty_symbol"]
                )

                # Update adaptive-selection counters: ask each base model for its
                # prediction at the same model-specific state and update correctness.
                prev_total = self.total_predictions
                self.total_predictions += 1
                logger.debug("AdaptiveVoting.evaluate: total_predictions %d -> %d", prev_total, self.total_predictions)

                if current_state is not None and isinstance(current_state, tuple):
                    for idx, (model, model_state) in enumerate(
                        zip(self.models, current_state, strict=True)
                    ):
                        m_metrics = model.state_metrics(model_state)
                        m_pred = metrics_prediction(m_metrics, config=self.config)
                        if m_pred is not None and m_pred.get("activity") == actual_next_activity:
                            prev = self.correct_predictions[idx]
                            self.correct_predictions[idx] += 1
                            logger.debug(
                                "AdaptiveVoting.evaluate: model %d predicted %s, actual %s, correct_predictions %d -> %d",
                                idx,
                                m_pred.get("activity"),
                                actual_next_activity,
                                prev,
                                self.correct_predictions[idx],
                            )

                else:
                    # Fall back to case_metrics when composed state is not available
                    for idx, model in enumerate(self.models):
                        m_metrics = model.case_metrics(event.get("case_id"))
                        m_pred = metrics_prediction(m_metrics, config=self.config)
                        if m_pred is not None and m_pred.get("activity") == actual_next_activity:
                            prev = self.correct_predictions[idx]
                            self.correct_predictions[idx] += 1
                            logger.debug(
                                "AdaptiveVoting.evaluate: model %d (case_metrics) predicted %s, actual %s, correct_predictions %d -> %d",
                                idx,
                                m_pred.get("activity"),
                                actual_next_activity,
                                prev,
                                self.correct_predictions[idx],
                            )


                pause_start_time = time.time()

                if compute_perplexity:
                    logger.debug("      [Before] Likelihood: %s", likelihood)
                    if log_likelihood:
                        likelihood += math.log(self.state_act_likelihood(current_state, actual_next_activity))
                    else:
                        likelihood *= self.state_act_likelihood(current_state, actual_next_activity)
                    logger.debug("      [After] Likelihood: %s", likelihood)

                logger.debug("State: %s", current_state)
                logger.debug("Actual next activity: %s", actual_next_activity)
                logger.debug("Prediction: %s", prediction)

                logger.debug("Metrics: %s", metrics)

                # Update shared statistics
                self.update_stats(event, prediction, current_state) # type: ignore  # noqa: PGH003

                pause_time += time.time() - pause_start_time

                if i < len(sequence) - 1:
                    current_state = self.next_state(current_state, actual_next_activity)
                    logger.debug("Next state: %s", current_state)

            if compute_perplexity:
                if log_likelihood:
                    normalized_likelihood = likelihood / len(sequence) if len(sequence) > 0 else likelihood
                else:
                    normalized_likelihood = likelihood ** (1 / len(sequence)) if len(sequence) > 0 else likelihood

                seq_perplexity = compute_seq_perplexity(normalized_likelihood, log_likelihood=log_likelihood)
            else:
                seq_perplexity = float("nan")

            perplexities.append(seq_perplexity if compute_perplexity else likelihood)

            logger.debug(
                "Pred. sequence: %s",
                predicted_sequence.replace(self.config["stop_symbol"].__str__(), "S"),
            )
            logger.debug("Sequence likelihood: %s", likelihood)
            logger.debug("Sequence perplexity: %s", seq_perplexity)
            logger.debug("===== End Evaluating Sequence =====")

        perplexity_stats = compute_perplexity_stats(perplexities)
        logger.debug("Perplexity stats: %s", perplexity_stats)

        for key, value in perplexity_stats.items():
            self.stats[key] = value

        elapsed = time.time() - eval_start_time - pause_time
        return elapsed, predicted_vector

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return the probability distribution from the model with the best accuracy so far."""
        state = self.get_state_from_case(case_id)

        # Get the best model
        if self.select_best == "accuracy":
            best_model_index = self.select_best_model()
            best_model = self.models[best_model_index]
            best_model_metrics = best_model.case_metrics(case_id)
        else:
            best_model_metrics = self.get_best_model_metrics(state)[1]

        delays_list = [model.case_metrics(case_id)["predicted_delays"] for model in self.models]

        return Metrics(
            state_id=state,
            probs=best_model_metrics["probs"],
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=best_model_metrics["likelihoods"]
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the probability distribution from the model with the best accuracy so far."""
        msg = "Adaptive voting is not implemented for sequences."
        raise NotImplementedError(msg)
        # Get the best model
        best_model_index = self.select_best_model()
        best_model = self.models[best_model_index]

        # best_model_metrics = best_model.sequence_metrics(sequence)

        delays_list = [model.sequence_metrics(sequence)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=best_model.sequence_metrics(sequence)["probs"],
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=best_model_metrics["likelihoods"]
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
        """
        Return the first non-{} probabilities from the models, cascading through the models in order.

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
                    state_id=state,
                    probs=metrics["probs"],
                    predicted_delays=self.voting_delays(delays_list),
                    # likelihoods=metrics["likelihoods"]
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
                    state_id=self.get_state_from_case(case_id),
                    probs=metrics["probs"],
                    predicted_delays=self.voting_delays(delays_list),
                    # likelihoods=metrics["likelihoods"]
                )

        # If all models return {}
        return empty_metrics()

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """
        Return the first non-empty probabilities from the models for the given sequence.

        Cascading through the models in order.
        """
        msg = "Fallback is not implemented for sequences."
        raise NotImplementedError(msg)

        delays_list = [model.sequence_metrics(sequence)["predicted_delays"] for model in self.models]

        for model in self.models:
            metrics = model.sequence_metrics(sequence)
            logger.debug("Trying model with metrics %s", metrics)
            if metrics["probs"]:
                logger.debug("Chose model with metrics %s", metrics)
                return Metrics(
                    probs=metrics["probs"],
                    predicted_delays=self.voting_delays(delays_list),
                    # likelihoods=metrics["likelihoods"]
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
        # likelihoods = self.model1.state_metrics(state1)["likelihoods"] # TO BE CHECKED

        if probs:
            probs = self.model2.state_metrics(state2)["probs"]
            # likelihoods = self.model2.state_metrics(state2)["likelihoods"] # TO BE CHECKED

        delays_list = [
            model.state_metrics(model_state)["predicted_delays"]
            for model, model_state in zip(self.models, state, strict=True)
        ]

        return Metrics(
            state_id=state,
            probs=probs,
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=likelihoods
        )

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return the first non-{} probabilities from the models, cascading through the models in order."""
        metrics = self.model1.case_metrics(case_id)
        probs = metrics["probs"]
        # likelihoods = metrics["likelihoods"]

        if probs:
            metrics = self.model2.case_metrics(case_id)
            probs = metrics["probs"]
            # likelihoods = metrics["likelihoods"]

        delays_list = [model.case_metrics(case_id)["predicted_delays"] for model in self.models]

        return Metrics(
            state_id=self.get_state_from_case(case_id),
            probs=probs,
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=likelihoods
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return the first non-{} probabilities from the models for the given sequence."""
        msg = "Relativize is not implemented for sequences."
        raise NotImplementedError(msg)
        metrics = self.model1.sequence_metrics(sequence)
        probs = metrics["probs"]
        # likelihoods = metrics["likelihoods"]

        if probs:
            metrics = self.model2.sequence_metrics(sequence)
            probs = metrics["probs"]
            # likelihoods = metrics["likelihoods"]

        delays_list = [model.sequence_metrics(sequence)["predicted_delays"] for model in self.models]

        return Metrics(
            probs=probs,
            predicted_delays=self.voting_delays(delays_list),
            # likelihoods=likelihoods
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
        """
        Return the probability distribution of the state.

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
        """
        Return the probability distribution of the state.

        The state should be a probabilistic automaton.
        """
        probs = self.get_probability_distribution(state)
        return Metrics(
            state_id=state,
            probs=probs,
            predicted_delays={},
            # likelihoods=probs
        )

    def case_metrics(self, case_id: CaseId) -> Metrics:  # noqa: ARG002
        """Methods not used in this subclass."""
        return empty_metrics()

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """
        Return the probability distribution of the state.

        The state should be a probabilistic automaton.
        """
        transformed_sequence = add_input_symbols_sequence(sequence, "in")

        self.algorithm.reset_to_initial()

        for symbol in transformed_sequence:
            self.algorithm.step_to(symbol[0], symbol[1])

        # Get probability distribution for the current state
        probs = self.get_probability_distribution(self.algorithm.current_state)
        return Metrics(
            state_id=self.algorithm.current_state,
            probs=probs,
            predicted_delays={},
            # likelihoods=probs
        )

    def next_state(self, current_state: ComposedState, activity: ActivityName) -> ComposedState:
        """Take a transition from the current state."""
        self.algorithm.current_state = current_state
        self.algorithm.step_to("in", activity)
        return self.algorithm.current_state


# ============================================================
# Neural Network Streaming Miner (RNN and LSTM)
# ============================================================


class NeuralNetworkMiner(StreamingMiner):
    """Neural Network based streaming miner."""

    device: torch.device | None

    STEP_OUTPUT_TUPLE_LEN = 2  # length of (logits, hidden) tuple returned by step()/forward

    def __init__(  # noqa: PLR0913
        self,
    model: RNNModel | LSTMModel | GRUModel | TransformerModel | QNetwork,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        *,
        criterion: nn.Module | None = None,
        mode: str = "incremental",
        sequence_buffer_length: int = 128,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the NeuralNetworkMiner class."""
        super().__init__(config=config)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.mode = mode
        # Prefer model.device when exposed; fallback to CPU to avoid None-device errors
        self.device = getattr(model, "device", None) or torch.device("cpu")
        self.sequence_buffer_length = sequence_buffer_length

        # Store per-case sequences as indices (int tokens). Mapping maintained below.
        # Ordered dictionary to maintain insertion order (used by round-robin batching)
        self.sequences: OrderedDict[CaseId, list[int]] = OrderedDict()

        # incremental caches
        self._case_rnn_state: dict[CaseId, Any] = {}  # stores hidden states for RNN/LSTM/GRU
        self._recent_tail: dict[CaseId, deque[int]] = defaultdict(lambda: deque(maxlen=self.sequence_buffer_length))

        self.rr_index = 0  # Keeps track of the round-robin index
        self.batch_size = batch_size

        # Bi-directional mapping between activity names and indices (0 reserved for padding)
        self.activity_index: dict[ActivityName, int] = {}
        self.index_activity: dict[int, ActivityName] = {}

    def get_state_from_case(self, case_id: CaseId) -> StateId:
        """Not implemented."""
        msg = "Not implemented for NeuralNetworkMiner."
        raise NotImplementedError(msg)

    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the sequence based on the metrics from the models."""
        # WARNING: This method is not implemented in the neural network miner.
        _, _ = state, next_activity  # Unused variables
        return -1.0

    def get_state_info(self, state_id: StateId) -> ComposedState | None:
        """Return the state information of the algorithm."""
        ### QUESTION: WHAT IS THIS FOR? TESTED?
        _ = state_id
        return None

    def get_sequence(self, case_id: CaseId) -> list[int]:
        """Return the token-index sequence for a specific case_id."""
        return self.sequences.get(case_id, [])

    def get_modified_cases(self) -> set[CaseId]:
        """Not implemented."""
        return set()

    def update(self, event: Event) -> None:
        """
        Add an activity to the sequence corresponding to the case_id.

        Dynamically update the activity_to_idx mapping if a new activity is encountered.
        """
        case_id = event["case_id"]
        activity = event["activity"]

        # Dynamically map activity to an index (0 is reserved for padding)
        if activity not in self.activity_index:
            new_idx = len(self.activity_index) + 1
            self.activity_index[activity] = new_idx
            self.index_activity[new_idx] = activity

        activity_idx = int(self.activity_index[activity])

        # Initialize sequence container for new case_id
        if case_id not in self.sequences:
            self.sequences[case_id] = []

        # Append index to per-case buffers
        self.sequences[case_id].append(activity_idx)
        self._recent_tail[case_id].append(activity_idx)

        # Optional incremental hidden-state update (no grad) for models exposing step()
        if self.mode == "incremental" and hasattr(self.model, "step"):
            last_token = torch.tensor([activity_idx], dtype=torch.long, device=self.device)
            hidden = self._case_rnn_state.get(case_id, None)
            try:
                with torch.no_grad():
                    _out = self.model.step(last_token, hidden)  # type: ignore[attr-defined]
                # Support both (logits, new_hidden) or just new_hidden (unlikely)
                if isinstance(_out, tuple) and len(_out) == self.STEP_OUTPUT_TUPLE_LEN:
                    _, new_hidden = _out
                else:
                    new_hidden = _out  # type: ignore[assignment]
                self._case_rnn_state[case_id] = new_hidden
            except Exception:
                logger.exception("Incremental step() failed; continuing without hidden-state cache.")

        # Continue with the training step using the updated sequence
        batch = self.select_batch(case_id)

        # Ensure each sequence in the batch has at least two tokens
        if len(batch) == 0:
            msg = "Skipping training step because no valid sequences were found."
            logger.debug(msg)
            return

        # Set model to training mode
        self.model.train()

        # Convert the batch of sequences into tensors, LEFT-padding them to the same length
        # to mirror batch-mode training/evaluation which uses left padding and source masks.
        batch_sequences = [torch.tensor(seq, dtype=torch.long, device=self.device) for seq in batch]
        x_batch = _left_pad_stack(batch_sequences, pad_value=0)

        # Input is all but the last token in each sequence, target is shifted by one position
        x_input = x_batch[:, :-1]  # Input sequence
        y_target = x_batch[:, 1:].reshape(-1)  # Flatten the target for CrossEntropyLoss

        self.optimizer.zero_grad()

        # Forward pass through the model
        outputs = self.model(x_input)
        # Support models returning either logits or (logits, hidden)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

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
        # loss.item()




    def train_on_batch(self, batch_case_ids: list[str]) -> torch.Tensor | None:
        """
        Construct a batch from case ids (their full sequences) and perform a training step.

        This method leaves incremental caches untouched; you may want to recompute caches after training.
        """
        sequences = [torch.tensor(self.sequences[cid], dtype=torch.long, device=self.device) for cid in batch_case_ids]

        # Pad sequences to the same length (left-padding with 0)
        batch = _left_pad_stack(sequences, pad_value=0)

        # teacher-forcing next-token prediction
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits, _ = self.model(x)
        logits = logits.reshape(-1, self.model.vocab_size)
        targets = y.reshape(-1)
        # mask padding targets
        mask = targets != 0
        if mask.sum().item() == 0:
            return None
        logits = logits[mask]
        targets = targets[mask]
        return self.criterion(logits, targets) # loss

    def select_batch(self, case_id: CaseId) -> list[list[int]]:
        """
        Select a batch of sequences, using a round-robin approach.

        Only select sequences that have at least two tokens (input + target).
        """
        # Only consider sequences that have at least two tokens (input + target)
        valid_case_ids = [cid for cid, sequence in self.sequences.items() if len(sequence) > 1]

        # If there are no valid cases, return empty batch
        if len(valid_case_ids) == 0:
            return []

        # If there are fewer valid cases than the desired batch size, return what we have
        if len(valid_case_ids) < self.batch_size:
            # Use DEBUG level to avoid noisy INFO spam during cold-start streaming
            if len(valid_case_ids) > 0:
                logger.debug(
                    "Not enough case_ids to form a full batch, using %d case_ids.",
                    len(valid_case_ids)
                )
            return [self.get_sequence(cid) for cid in valid_case_ids]

        # Prepare the batch, starting with the current case_id if it's valid
        batch_case_ids = []
        if case_id in valid_case_ids:
            batch_case_ids.append(case_id)

        # Prepare for round-robin selection
        rr_len = len(valid_case_ids)
        idx = self.rr_index % rr_len
        attempts = 0
        max_attempts = rr_len  # Prevent infinite loops

        # Select additional case_ids in a round-robin manner, skipping the current case_id
        while len(batch_case_ids) < self.batch_size and attempts < max_attempts:
            candidate_case_id = valid_case_ids[idx]

            # Add if not already in batch (handles current case_id and duplicates)
            if candidate_case_id not in batch_case_ids:
                batch_case_ids.append(candidate_case_id)

            # Move to the next index, wrap around
            idx = (idx + 1) % rr_len
            attempts += 1

        # Update the round-robin index for next call
        self.rr_index = idx

        # Fetch the actual sequences based on the selected case_ids
        return [self.get_sequence(cid) for cid in batch_case_ids]


    def case_metrics(self, case_id: CaseId) -> Metrics:
        """
        Predict the next activity for a given case_id.

        Return the top-k most likely activities along with the probability
        of the top activity.
        Note that, here, a sequence is a sequence of activity indices (rather than activities).
        """
        # Get the sequence for the case_id
        index_sequence = self.get_sequence(case_id)

        if not index_sequence or len(index_sequence) < 1:
            return empty_metrics()

        probs = self.idx_sequence_probs(index_sequence)
        return Metrics(
            state_id=-1,
            probs=probs,
            predicted_delays={},
            # likelihoods=probs
        )

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """
        Predict the next activity for a given sequence of activities and return the top-k most likely activities...

        ...along with the
        probability of the top activity.
        """
        msg = "NeuralNetworkMiner does not support sequence metrics."
        raise NotImplementedError(msg)
        # If the method were supported, below is the intended implementation template:
        # if not sequence or len(sequence) < 1:
        #     return empty_metrics()
        # index_sequence: list[int] = []
        # for event in sequence:
        #     activity = event["activity"]
        #     activity_idx = self.activity_index.get(activity)
        #     if activity_idx is None:
        #         return empty_metrics()
        #     index_sequence.append(activity_idx)
        # probs = self.idx_sequence_probs(index_sequence)
        # return Metrics(probs=probs, predicted_delays={})

    def idx_sequence_probs(self, index_sequence: list[int]) -> ProbDistr:
        """Predict next-activity probabilities from a given index sequence."""
        if not index_sequence:
            # Return uniform-zero over known activities (no context); empty dict is fine
            return {}

        # Convert to tensor [1, L]
        input_sequence = torch.as_tensor([index_sequence], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_sequence)
            # Support models that return (logits, hidden)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        # Get logits for last time step and softmax
        logits_last = outputs[:, -1, :]
        probabilities_t = torch.softmax(logits_last, dim=-1).squeeze(0)

        # Map back to activity names using index_activity
        probabilities = probabilities_t.tolist()
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


# ============================================================
# Windowed Neural Network Miner (limits context by window length)
# ============================================================


class WindowedNeuralNetworkMiner(NeuralNetworkMiner):
    """
    Neural Network miner that restricts context to the last N tokens per case.

    This mirrors the "window_size" behavior used in batch mode when training/evaluating
    LSTM/Transformer models on only the last ``window_size`` tokens. Here we enforce
    a sequence buffer per case so both training and prediction use at most the most
    recent ``sequence_buffer_length`` tokens.
    """

    def __init__(  # noqa: PLR0913
        self,
    model: RNNModel | LSTMModel | GRUModel | TransformerModel,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        *,
        criterion: nn.Module | None = None,
        # `sequence_buffer_length` is interpreted as the desired "window size"
        # (number of prefix tokens used to predict the next activity). Internally
        # we keep `window_size + 1` tokens so that training batches can contain
        # an input prefix of length `window_size` and the corresponding target
        # next-token. This ensures a user-provided window X yields prefix size X.
        sequence_buffer_length: int = 50,
        mode: str = "incremental",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the WindowedNeuralNetworkMiner class."""
        super().__init__(
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            mode=mode,
            sequence_buffer_length=sequence_buffer_length,
            config=config,
        )
        if sequence_buffer_length < 1:
            msg = "sequence_buffer_length must be >= 1"
            raise ValueError(msg)
        # Expose the user-requested window size separately
        self.window_size = int(sequence_buffer_length)
        # Internally keep window_size + 1 tokens so that x_input = seq[:, :-1]
        # produces an input prefix of length == window_size during training.
        self.sequence_buffer_length = self.window_size + 1

    def _trim_sequence(self, case_id: CaseId) -> None:
        """Keep only the last ``sequence_buffer_length`` tokens for the case."""
        if case_id in self.sequences:
            seq = self.sequences[case_id]
            if len(seq) > self.sequence_buffer_length:
                # Slice to the last window
                self.sequences[case_id] = seq[-self.sequence_buffer_length :]

    def update(self, event: Event) -> None:
        """Ensure window trimming before training step, otherwise same as parent."""
        case_id = event["case_id"]
        activity = event["activity"]

        # Dynamically update activity_to_idx if the activity is new
        if activity not in self.activity_index:
            current_idx = len(self.activity_index) + 1
            self.activity_index[activity] = current_idx
            self.index_activity[current_idx] = activity

        # Convert activity to its corresponding index
        activity_idx = self.activity_index[activity]

        # Add the activity index to the sequence for the given case_id
        if case_id not in self.sequences:
            self.sequences[case_id] = []
        self.sequences[case_id].append(activity_idx)

        # Enforce window before training
        self._trim_sequence(case_id)

        # Select batch and proceed as parent
        batch = self.select_batch(case_id)
        if len(batch) == 0:
            # Use DEBUG level to avoid noisy INFO spam during cold-start streaming.
            logger.debug("Skipping training step because no valid sequences were found.")
            return

        # Training step (same as parent)
        self.model.train()
        batch_sequences = [torch.tensor(seq, dtype=torch.long, device=self.device) for seq in batch]
        # Left-pad for windowed miner as well to match batch-mode behavior
        x_batch = _left_pad_stack(batch_sequences, pad_value=0)

        # Input is all but the last token in each sequence, target is shifted by one position
        x_input = x_batch[:, :-1]
        y_target = x_batch[:, 1:].reshape(-1)

        self.optimizer.zero_grad()
        outputs = self.model(x_input)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = outputs.view(-1, outputs.shape[-1])

        mask = y_target != 0
        outputs = outputs[mask]
        y_target = y_target[mask]

        loss = self.criterion(outputs, y_target)
        loss.backward()
        self.optimizer.step()

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Predict using only the last ``sequence_buffer_length`` tokens of the case."""
        index_sequence = self.get_sequence(case_id)
        if not index_sequence or len(index_sequence) < 1:
            return empty_metrics()
        # Use only the last `window_size` tokens as input context for prediction
        # (do not include the extra reserved token used for training targets).
        windowed = index_sequence[-self.window_size :]
        if len(windowed) < 1:
            return empty_metrics()
        probs = self.idx_sequence_probs(windowed)
        return Metrics(state_id=-1, probs=probs, predicted_delays={})

# New: Reinforcement-Learning-style miner (REINFORCE-like updates)
class RLMiner(NeuralNetworkMiner):
    """
    Reinforcement Learning Miner.

    Differences vs NeuralNetworkMiner:
    - Updates using rewards (REINFORCE-like): loss = - reward * log_prob(action)
    - sequence_buffer_length: maximum number of tokens stored per case (acts as a queue)
    - state encoder with (long_term_mem_size, short_term_mem_size) producing an embedding
      of size long_term_mem_size + short_term_mem_size (deterministic encoding).

    Expected event format for RL update:
    - event['case_id'], event['activity'] (the observed action)
    - optional: event['reward'] (numeric). If not present, reward defaults to 0.0 (no update).
    """

    def __init__(  # noqa: PLR0913
        self,
        model: QNetwork,
        optimizer: torch.optim.Optimizer,
        *,
        criterion: nn.Module | None = None,
        sequence_buffer_length: int = 50,
        long_term_mem_size: int = 10,
        short_term_mem_size: int | None = None,
        mode: str = "incremental",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the RLMiner class."""
        # Initialize parent
        super().__init__(
            model=model,
            batch_size=8,
            optimizer=optimizer,
            criterion=criterion,
            mode=mode,
            sequence_buffer_length=sequence_buffer_length,
            config=config,
        )

        # Buffering parameters
        if short_term_mem_size is None:
            short_term_mem_size = sequence_buffer_length
        if long_term_mem_size < 0 or short_term_mem_size < 0 or sequence_buffer_length < 1:
            msg = "Invalid memory sizes or sequence_buffer_length"
            raise ValueError(msg)

        self.sequence_buffer_length = sequence_buffer_length
        self.long_term_mem_size = long_term_mem_size
        self.short_term_mem_size = short_term_mem_size

        # Per-case last reward store (used during batch RL updates)
        self.case_rewards: dict[CaseId, float] = {}

        # reuse sequences, activity_index, index_activity from NeuralNetworkMiner
        # ... they are already initialized in the parent
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[RLMiner.__init__] buffer_len=%s, long_mem=%s, short_mem=%s, device=%s",
                self.sequence_buffer_length, self.long_term_mem_size, self.short_term_mem_size, self.device
            )

    def _enqueue_activity(self, case_id: CaseId, activity_idx: int) -> None:
        """Append activity index to the case sequence and enforce sequence_buffer_length."""
        if case_id not in self.sequences:
            self.sequences[case_id] = []
        self.sequences[case_id].append(activity_idx)

        if logger.isEnabledFor(logging.DEBUG):
            act_name = self.index_activity.get(activity_idx, f"<unk:{activity_idx}>")
            logger.debug("[RLMiner._enqueue_activity] case=%s enqueue idx=%s (%s)", case_id, activity_idx, act_name)

        # Enforce buffer length as a queue (discard oldest when exceeding)
        while len(self.sequences[case_id]) > self.sequence_buffer_length:
            removed = self.sequences[case_id].pop(0)
            if logger.isEnabledFor(logging.DEBUG):
                removed_name = self.index_activity.get(removed, f"<unk:{removed}>")
                logger.debug(
                    "[RLMiner._enqueue_activity] case=%s truncating oldest idx=%s (%s)", case_id, removed, removed_name
                )

    def update(self, event: Event) -> None:
        """
        Single-step RL-style update.

        - Use current sequence (before this event) as context to get the model prediction.
        - Reward r = 1.0 if predicted activity == observed activity, else 0.0 (or external event['reward'] if present).
        - Loss = - r * log_prob(observed_activity | current_seq)  (REINFORCE-style scaling).
        - Backpropagate immediately (no batch).
        - Finally, enqueue observed activity into the per-case sequence buffer.
        """
        case_id = event["case_id"]
        activity = event["activity"]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[RLMiner.update] event case=%s activity=%s", case_id, activity)

        # Current sequence BEFORE observing this activity
        current_seq = self.get_sequence(case_id)
        if logger.isEnabledFor(logging.DEBUG):
            seq_names = [self.index_activity.get(i, f"<unk:{i}>") for i in current_seq]
            logger.debug("[RLMiner.update] current_seq idx=%s", current_seq)
            logger.debug("[RLMiner.update] current_seq names=%s", seq_names)

        # Compute reward based on model prediction from current context (external reward overrides if provided)
        if current_seq and len(current_seq) >= 1:
            probs = self.idx_sequence_probs(current_seq)
            pred = probs_prediction(probs, config=self.config)

            # External reward (from event) takes precedence when provided.
            # If absent, fall back to a simple 0/1 reward computed from whether
            # the model predicted the observed activity.
            env_reward = event.get("reward", None)
            computed_reward = 1.0 if pred is not None and pred.get("activity") == activity else 0.0
            effective_reward = float(env_reward) if env_reward is not None else computed_reward

            # If an external reward was supplied, store it for potential batch RL updates
            if env_reward is not None:
                self.case_rewards[case_id] = effective_reward

            # Compute policy loss on the observed action with REINFORCE-style scaling
            # (loss = - reward * log_prob). If env reward is not provided, we use
            # the computed 0/1 reward from match vs observed activity.
            self.model.train()
            self.optimizer.zero_grad()

            x_input = torch.tensor(current_seq, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, L]
            outputs = self.model(x_input)  # [1, L, vocab] or (logits, hidden)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            logits_last = outputs[:, -1, :]  # [1, vocab]
            log_probs = torch.log_softmax(logits_last, dim=-1).squeeze(0)  # [vocab]

            # Ensure the activity has an index
            if activity not in self.activity_index:
                new_idx = len(self.activity_index) + 1
                self.activity_index[activity] = new_idx
                self.index_activity[new_idx] = activity
                logger.debug("[RLMiner.update] assigned new index to activity '%s' -> %s", activity, new_idx)
            target_idx = int(self.activity_index[activity])

            if target_idx < log_probs.shape[0]:
                lp = log_probs[target_idx]
                # If an external reward was provided, use REINFORCE-style scaling
                # (loss = - reward * log_prob). If no external reward is provided,
                # fall back to supervised negative log-likelihood loss (cross-entropy)
                # so the model learns from the observed action even when it did not
                # predict it correctly yet.
                loss = -effective_reward * lp if env_reward is not None else -lp

                logger.debug(
                    "[RLMiner.update] target_idx=%s activity=%s log_prob=%.6f reward=%s loss=%.6f vocab=%s",
                    target_idx,
                    activity,
                    lp.item(),
                    str(env_reward) if env_reward is not None else "<supervised>",
                    loss.item(),
                    log_probs.shape[0],
                )
                loss.backward()
                self.optimizer.step()
                logger.debug("[RLMiner.update] optimizer step completed")
            else:
                logger.error(
                    "[RLMiner.update] activity '%s' with idx=%s not in vocab (size=%s)",
                    activity, target_idx, log_probs.shape[0]
                )
        else:
            logger.debug("[RLMiner.update] no context (empty sequence). Skipping loss computation.")

        # Enqueue observed activity for future context
        if activity not in self.activity_index:
            new_idx = len(self.activity_index) + 1
            self.activity_index[activity] = new_idx
            self.index_activity[new_idx] = activity
            logger.debug("[RLMiner.update] assigned new index to activity '%s' -> %s (enqueue path)", activity, new_idx)

        self._enqueue_activity(case_id, self.activity_index[activity])

        # new_seq = self.get_sequence(case_id)
        # new_seq_names = [self.index_activity.get(i, f"<unk:{i}>") for i in new_seq]
        # logger.debug("[RLMiner.update] new_seq idx=%s", new_seq)
        # logger.debug("[RLMiner.update] new_seq names=%s", new_seq_names)

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Predict next activity probabilities for a case_id (like NeuralNetworkMiner.case_metrics)."""
        index_sequence = self.get_sequence(case_id)
        if not index_sequence or len(index_sequence) < 1:
            return empty_metrics()
        probs = self.idx_sequence_probs(index_sequence)
        if logger.isEnabledFor(logging.DEBUG):
            top_k = self.config.get("top_k", 5)
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            logger.debug("[RLMiner.case_metrics] case=%s seq_len=%s top-%s=%s",
                         case_id, len(index_sequence), top_k, sorted_probs[:top_k])
        return Metrics(state_id=-1, probs=probs, predicted_delays={})

    def idx_sequence_probs(self, index_sequence: list[int]) -> ProbDistr:
        """Predict next activity probabilities for given index sequence (like parent)."""
        # Convert to a tensor and add a batch dimension
        input_sequence = torch.as_tensor([index_sequence], dtype=torch.long, device=self.device)
        # input_sequence = torch.tensor(index_sequence, dtype=torch.long, device=self.device).unsqueeze(0)  # Shape [1, sequence_length] # noqa: E501


        # Pass the sequence through the model to get the output
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_sequence)
            if isinstance(output, tuple):
                output = output[0]
        # Get the logits for the last time step (most recent activity in the sequence)
        logits = output[:, -1, :]  # Shape [1, vocab_size]

        # Apply softmax to get the probabilities
        probabilities = torch.softmax(logits, dim=-1)  # Shape [1, vocab_size]

        # Convert the tensor to a list of probabilities
        probabilities = probabilities.squeeze(0).tolist()  # Shape [vocab_size]

        result = {
            self.index_activity[idx]: prob
            for idx, prob in enumerate(probabilities)
            if self.index_activity.get(idx) is not None
        }
        if logger.isEnabledFor(logging.DEBUG):
            top_k = self.config.get("top_k", 5)
            sorted_probs = sorted(result.items(), key=lambda x: x[1], reverse=True)
            logger.debug("[RLMiner.idx_sequence_probs] seq_len=%s top-%s=%s",
                         len(index_sequence), top_k, sorted_probs[:top_k])
        return result
