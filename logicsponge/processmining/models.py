import logging
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import matplotlib as mpl
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from logicsponge.processmining.data_utils import add_input_symbols_sequence
from logicsponge.processmining.globals import (
    RANDOMIZED,
    STATS,
    TOP_K,
    ActionName,
    CaseId,
    ComposedState,
    Prediction,
)
from logicsponge.processmining.neural_networks import LSTMModel, RNNModel

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
                if i < len(sequence) - 1:
                    current_state = self.next_state(current_state, actual_next_action)

        # Return summarized stats
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


# ============================================================
# Neural Network Streaming Miner (RNN and LSTM)
# ============================================================


class NeuralNetworkMiner(BaseStreamingMiner, ABC):
    def __init__(self, *args, model: RNNModel | LSTMModel, batch_size: int = 1, optimizer, criterion, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = model  # The neural network
        self.optimizer = optimizer
        self.criterion = criterion

        self.sequences = OrderedDict()  # Ordered dictionary to maintain insertion order
        self.rr_index = 0  # Keeps track of the round-robin index
        self.batch_size = batch_size

        self.action_index = {}
        self.index_action = {}

    def get_sequences(self):
        """
        Return all sequences stored in the state.
        """
        return self.sequences

    def get_sequence(self, case_id):
        """
        Return the sequence for a specific case_id.
        """
        return self.sequences.get(case_id, [])

    def update(self, case_id: CaseId, action: ActionName) -> None:
        """
        Add an action to the sequence corresponding to the case_id.
        Dynamically update the activity_to_idx mapping if a new action is encountered.
        """
        # Dynamically update activity_to_idx if the action is new
        if action not in self.action_index:
            current_idx = len(self.action_index) + 1  # Get the next available index
            self.action_index[action] = current_idx
            self.index_action[current_idx] = action

        # Convert action to its corresponding index
        action_idx = self.action_index[action]

        # Add the action index to the sequence for the given case_id
        if case_id not in self.sequences:
            self.sequences[case_id] = []  # New case added
        self.sequences[case_id].append(action_idx)

        # Continue with the training step using the updated sequence
        batch = self.select_batch(case_id)

        # Ensure each sequence in the batch has at least two tokens
        if not batch:
            msg = "Skipping training step because no valid sequences were found."
            logger.info(msg)
            return None

        # Convert the batch of sequences into tensors, padding them to the same length
        batch_sequences = [torch.tensor(seq, dtype=torch.long) for seq in batch]
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def select_batch(self, case_id):
        """
        Select a batch of sequences, using a round-robin approach.
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

    def prediction_case(self, case_id: CaseId) -> Prediction | None:
        """
        Predict the next action for a given case_id and return the top-k most likely actions along with the probability
        of the top action.

        Note that, here, a sequence is a sequence of action indices (rather than actions).
        """

        # Get the sequence for the case_id
        index_sequence = self.get_sequence(case_id)

        if not index_sequence or len(index_sequence) < 1:
            return None

        return self.prediction_idx_sequence(index_sequence)

    def prediction_sequence(self, sequence: list[ActionName]) -> Prediction | None:
        """
        Predict the next action for a given sequence of actions and return the top-k most likely actions along with the
        probability of the top action.
        """
        if not sequence or len(sequence) < 1:
            return None

        # Convert each action name to its corresponding index, return None if any action is unknown
        index_sequence = []
        for action in sequence:
            action_idx = self.action_index.get(action)
            if action_idx is None:
                return None  # Return None if the action is not found in the index
            index_sequence.append(action_idx)

        return self.prediction_idx_sequence(index_sequence)

    def prediction_idx_sequence(self, index_sequence: list[int]) -> Prediction | None:
        """
        Predict the next action for a given sequence of action indices.
        """
        # Convert to a tensor and add a batch dimension
        input_sequence = torch.tensor(index_sequence, dtype=torch.long).unsqueeze(0)  # Shape [1, sequence_length]

        # Pass the sequence through the model to get the output
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_sequence)

        # Get the logits for the last time step (most recent action in the sequence)
        logits = output[:, -1, :]  # Shape [1, vocab_size]

        # Apply softmax to get the probabilities
        probabilities = torch.softmax(logits, dim=-1)  # Shape [1, vocab_size]

        # Get the top-k most likely actions and their probabilities
        top_k_results = torch.topk(probabilities, self.top_k, dim=1)
        top_k_indices = top_k_results.indices.squeeze(0).tolist()  # Shape [top_k]
        top_k_probs = top_k_results.values.squeeze(0).tolist()  # Shape [top_k]

        # Convert the top-k indices back to action names
        top_k_actions = [self.index_action.get(idx, None) for idx in top_k_indices]

        # If the most likely action is not found, return None
        if top_k_actions[0] is None:
            return None

        # Return a tuple with the most likely action, the top-k actions, and the probability of the top action
        return top_k_actions[0], top_k_actions, top_k_probs[0]

    def next_state(self, *args, **kwargs):
        pass  # Or return None, depending on your base class interface

    def prediction_state(self, *args, **kwargs):
        pass
