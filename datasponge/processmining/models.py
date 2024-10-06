import logging
import random
from abc import ABC, abstractmethod
from typing import Any

import matplotlib as mpl
import pandas as pd

from datasponge.processmining.data_utils import add_input_symbols_sequence
from datasponge.processmining.globals import (
    RANDOMIZED,
    STATS,
    TOP_K,
    ActionName,
    CaseId,
    ComposedState,
    Prediction,
)

mpl.use("Agg")

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line-wrapping

logger = logging.getLogger(__name__)

random.seed(123)


# ============================================================
# Base Streaming Miner (for streaming and batch mode)
# ============================================================


class BaseStreamingMiner(ABC):
    """
    The Base Streaming Miner (for both streaming and batch mode)
    """

    def __init__(self, randomized: bool = RANDOMIZED, top_k: int = TOP_K) -> None:  # noqa: FBT001
        self.randomized = randomized
        self.top_k = top_k

        self.algorithm = None
        self.initial_state = None

    @abstractmethod
    def update(self, case_id: CaseId, action: ActionName) -> None:
        """
        Updates Strategy.
        """

    @abstractmethod
    def next_state(self, current_state: ComposedState | None, action: ActionName) -> ComposedState | None:
        """
        Takes a transition from the current state.
        """

    @abstractmethod
    def prediction_case(self, case_id: CaseId) -> Prediction | None:
        """
        Makes a prediction for a given case id.
        """

    @abstractmethod
    def prediction_state(self, state: ComposedState | None) -> Prediction | None:
        """
        Makes a prediction for a given state.
        """

    @abstractmethod
    def prediction_sequence(self, sequence: list[ActionName]) -> Prediction | None:
        """
        Makes a prediction for a given sequence. Only for batch mode.
        """

    @staticmethod
    def update_stats(actual_next_action: ActionName, prediction: Prediction | None, stats: dict[str, int]) -> None:
        """
        Updates the statistics based on the actual action, the prediction, and the top-k predictions.
        """
        if prediction is None:
            # Call the unparseable count update
            # STATS["unparseable_count"]["update"](stats)
            stats["unparseable_count"] += 1
        else:
            predicted_action, top_k_actions, predicted_prob = prediction

            # Loop through each metric and call its update function
            for stat in STATS.values():
                if "update" in stat:
                    # Pass all arguments as keyword arguments
                    stat["update"](
                        stats,
                        actual=actual_next_action,
                        predicted=predicted_action,
                        top_k=top_k_actions,
                        predicted_prob=predicted_prob,
                    )

    def evaluate(self, data: list[list[ActionName]], mode: str = "incremental") -> dict[str, int]:
        """
        Evaluates the dataset either incrementally or by full sequence.
        Modes: 'incremental' or 'sequence'.
        """
        # Initialize stats
        stats = {key: value["init"] for key, value in STATS.items()}

        for sequence in data:
            current_state = self.initial_state

            for i in range(len(sequence)):
                if current_state is None:
                    # If unparseable, count all remaining actions
                    stats["unparseable_count"] += len(sequence) - i
                    break

                actual_next_action = sequence[i]

                if mode == "incremental":
                    # Prediction for incremental mode (step by step)
                    prediction = self.prediction_state(current_state)
                else:
                    # Prediction for sequence mode (whole sequence)
                    prediction = self.prediction_sequence(sequence[:i])

                # Update statistics based on the prediction
                self.update_stats(actual_next_action, prediction, stats)

                # Move to the next state
                current_state = self.next_state(current_state, actual_next_action)

        # Return summarized stats
        return {STATS[key]["name"]: value for key, value in stats.items()}

    def evaluate_incrementally(self, data: list[list[ActionName]]) -> dict[str, int]:
        """
        Processes batch, going through sequences incrementally. Only for batch mode.
        """
        # Initialize stats based on the STATS constant
        stats = {key: value["init"] for key, value in STATS.items()}

        for sequence in data:
            current_state = self.initial_state

            for i in range(len(sequence)):
                if current_state is None:
                    # Increment unparseable_count for the remaining unparseable sequence
                    stats["unparseable_count"] += len(sequence) - i
                    break

                actual_next_action = sequence[i]  # The actual action in the sequence

                # Get the prediction and top-k predictions
                result = self.prediction_state(current_state)

                # update statistics
                self.update_stats(actual_next_action, result, stats)

                # Move to the next state
                current_state = self.next_state(current_state, actual_next_action)

        # Return a summary of the statistics using the names from STATS
        return {STATS[key]["name"]: value for key, value in stats.items()}

    def evaluate_sequence(self, data: list[list[ActionName]]) -> dict[str, int]:
        """
        Evaluates batch processing whole sequences. Only for batch mode.
        """
        stats = {key: value["init"] for key, value in STATS.items()}

        # Iterate through each sequence in the test set
        for action_sequence in data:
            for i in range(len(action_sequence)):
                x = action_sequence[:i]
                actual_next_action = action_sequence[i]

                # Get the prediction and top-k predictions
                result = self.prediction_sequence(x)

                # update statistics
                self.update_stats(actual_next_action, result, stats)

        # Return a summary of the statistics using the names from STATS
        return {STATS[key]["name"]: value for key, value in stats.items()}


# ============================================================
# Standard Streaming Miner (using one building block)
# ============================================================


class BasicMiner(BaseStreamingMiner):
    # model is usually BaseStructure (apart from Alergia)
    def __init__(self, *args, algorithm: Any, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm

        if self.algorithm is None:
            msg = "An algorithm must be specified."
            raise ValueError(msg)

        self.initial_state = self.algorithm.initial_state

    def update(self, case_id: CaseId, action: ActionName) -> None:
        self.algorithm.update(case_id, action)

    def next_state(self, current_state: ComposedState | None, action: ActionName) -> ComposedState | None:
        return self.algorithm.next_state(current_state, action)

    def prediction_case(self, case_id: CaseId) -> Prediction | None:
        return self.algorithm.prediction_case(case_id)

    def prediction_state(self, state: ComposedState | None) -> Prediction | None:
        return self.algorithm.prediction_state(state)

    def prediction_sequence(self, sequence: list[ActionName]) -> Prediction | None:
        return self.algorithm.prediction_sequence(sequence)


# ============================================================
# Multi Streaming Miner (using several building blocks)
# ============================================================


class MultiMiner(BaseStreamingMiner, ABC):
    def __init__(self, *args, models: list[BaseStreamingMiner], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.models = models

        for model in self.models:
            model.randomized = self.randomized

        self.initial_state = self.initial_state = tuple(model.initial_state for model in self.models)

    def update(self, case_id: CaseId, action: ActionName) -> None:
        for model in self.models:
            model.update(case_id, action)

    def next_state(self, current_state: ComposedState | None, action: ActionName) -> ComposedState | None:
        if current_state is None:
            return None

        # Unpack the current state for each model
        next_states = [model.next_state(state, action) for model, state in zip(self.models, current_state, strict=True)]

        # If all next states are None, return None
        if all(ns is None for ns in next_states):
            return None

        # Otherwise, return the tuple of next states
        return tuple(next_states)


class Fallback(MultiMiner):
    def prediction_case(self, case_id: CaseId) -> Prediction | None:
        """
        Return the first non-None prediction from the models, cascading through the models in order.
        """
        for model in self.models:
            prediction = model.prediction_case(case_id)
            if prediction is not None:
                return prediction

        # If all models return None
        return None

    def prediction_state(self, state: ComposedState | None) -> Prediction | None:
        """
        Return the first non-None prediction from the models, cascading through the models in order.
        Each model gets its corresponding state from the ComposedState.
        """
        if state is None:
            return None

        # Iterate through the models and their corresponding states
        for model, model_state in zip(self.models, state, strict=True):
            prediction = model.prediction_state(model_state)
            if prediction is not None:
                return prediction

        # If all models return None
        return None

    def prediction_sequence(self, sequence: list[ActionName]) -> Prediction | None:
        """
        Return the first non-None prediction from the models for the given sequence,
        cascading through the models in order.
        """
        for model in self.models:
            prediction = model.prediction_sequence(sequence)
            if prediction is not None:
                return prediction

        # If all models return None
        return None


class Relativize(MultiMiner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if len(self.models) != 2:  # noqa: PLR2004
            msg = "Class Relativize requires two models."
            raise ValueError(msg)

        self.model1 = self.models[0]
        self.model2 = self.models[1]

    def prediction_case(self, case_id: CaseId) -> Prediction | None:
        prediction = self.model1.prediction_case(case_id)

        if prediction is not None:
            prediction = self.model2.prediction_case(case_id)

        return prediction

    def prediction_state(self, state: ComposedState | None) -> Prediction | None:
        if state is None:
            return None

        (state1, state2) = state

        prediction = self.model1.prediction_state(state1)

        if prediction is not None:
            prediction = self.model2.prediction_state(state2)

        return prediction

    def prediction_sequence(self, sequence: list[ActionName]) -> Prediction | None:
        prediction = self.model1.prediction_sequence(sequence)

        if prediction is not None:
            prediction = self.model2.prediction_sequence(sequence)

        return prediction


# ============================================================
# Alergia
# ============================================================


class Alergia(BasicMiner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_state = self.initial_state

    @staticmethod
    def get_probability_distribution(state: Any) -> dict[str, float]:
        probability_distribution = {}

        for input_symbol, transitions in state.transitions.items():
            # Create a dictionary mapping output letters to probabilities for this input symbol
            output_probabilities = {transition[1]: transition[2] for transition in transitions}
            probability_distribution[input_symbol] = output_probabilities

        return probability_distribution["in"]

    def prediction_probs(self, probs: dict[str, float]) -> Prediction:
        sorted_actions = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_k_actions = [action for action, _ in sorted_actions[: self.top_k]]

        # Select the next action
        if self.randomized:
            # Choose randomly based on the probabilities
            actions, probabilities = zip(*sorted_actions, strict=True)
            next_action = random.choices(actions, weights=probabilities, k=1)[0]  # noqa: S311
        else:
            # Choose the action with the highest probability
            next_action = top_k_actions[0]

        # Get the probability of the selected action from the original probs dict
        next_action_prob = probs[next_action]

        return next_action, top_k_actions, next_action_prob

    def prediction_case(self, case_id: CaseId) -> None:  # noqa: ARG002
        """
        This method is not used in this subclass.
        """
        msg = "This method is not implemented for this subclass."
        raise NotImplementedError(msg)

    def update(self, case_id: CaseId, action: ActionName) -> None:  # noqa: ARG002
        """
        This method is not used in this subclass.
        """
        msg = "This method is not implemented for this subclass."
        raise NotImplementedError(msg)

    def prediction_state(self, state: Any) -> Prediction | None:
        probs = self.get_probability_distribution(state)

        # Sort probabilities in descending order to get top-k actions
        return self.prediction_probs(probs)

    def prediction_sequence(self, sequence: list[ActionName]) -> Prediction | None:
        transformed_sequence = add_input_symbols_sequence(sequence, "in")

        self.algorithm.reset_to_initial()

        for symbol in transformed_sequence:
            self.algorithm.step_to(symbol[0], symbol[1])

        # Get probability distribution for the current state
        probs = self.get_probability_distribution(self.algorithm.current_state)

        return self.prediction_probs(probs)

    def step(self, action):
        self.algorithm.step_to("in", action)
        self.current_state = self.algorithm.current_state

    def next_state(self, current_state, action):
        self.algorithm.current_state = current_state
        self.algorithm.step_to("in", action)
        return self.algorithm.current_state
