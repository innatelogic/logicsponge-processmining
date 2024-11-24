import copy
import logging
import random
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from typing import Any

import matplotlib as mpl
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from logicsponge.processmining.data_utils import add_input_symbols_sequence
from logicsponge.processmining.globals import (
    CONFIG,
    STOP,
    ActionName,
    CaseId,
    ComposedState,
    Prediction,
    ProbDistr,
    probs_prediction,
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


class StreamingMiner(ABC):
    """
    The Base Streaming Miner (for both streaming and batch mode)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Use CONFIG as a fallback if no specific config is provided
        self.config = copy.deepcopy(CONFIG)

        # If a specific configuration is provided, update the default copy with its values
        if config is not None:
            self.config.update(config)

        # Set the initial state (or other initialization tasks)
        self.initial_state: ComposedState | None = None

        # Statistics for batch mode
        self.stats = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "wrong_predictions": 0,
            "empty_predictions": 0,
            "log_loss": 0.0,  # not implemented yet
        }

    def update_stats(self, actual_next_action: ActionName, prediction: Prediction) -> None:
        """
        Updates the statistics based on the actual action, the prediction, and the top-k predictions.
        """
        self.stats["total_predictions"] += 1

        if not prediction:
            self.stats["empty_predictions"] += 1
        else:
            predicted_action = prediction["action"]

            if actual_next_action == predicted_action:
                self.stats["correct_predictions"] += 1

            if actual_next_action != predicted_action:
                self.stats["wrong_predictions"] += 1

    def evaluate(self, data: list[list[ActionName]], mode: str = "incremental") -> None:
        """
        Evaluation in batch mode.
        Evaluates the dataset either incrementally or by full sequence.
        Modes: 'incremental' or 'sequence'.
        """
        # Initialize stats
        for sequence in data:
            current_state = self.initial_state

            for i in range(len(sequence)):
                if current_state is None:
                    # If unparseable, count all remaining actions
                    self.stats["empty_predictions"] += len(sequence) - i
                    self.stats["total_predictions"] += len(sequence) - i
                    break

                actual_next_action = sequence[i]

                if mode == "incremental":
                    # Prediction for incremental mode (step by step)
                    probs = self.state_probs(current_state)
                    prediction = probs_prediction(probs, self.config)
                else:
                    # Prediction for sequence mode (whole sequence)
                    probs = self.sequence_probs(sequence[:i])
                    prediction = probs_prediction(probs, self.config)

                # Update statistics based on the prediction
                self.update_stats(actual_next_action, prediction)

                # Move to the next state
                if i < len(sequence) - 1:
                    current_state = self.next_state(current_state, actual_next_action)

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
    def state_probs(self, state: ComposedState | None) -> ProbDistr:
        """
        Returns probability dictionary based on state.
        """

    @abstractmethod
    def case_probs(self, case_id: CaseId) -> ProbDistr:
        """
        Returns probability dictionary based on case.
        """

    @abstractmethod
    def sequence_probs(self, sequence: list[ActionName]) -> ProbDistr:
        """
        Returns probability dictionary based on sequence.
        """


# ============================================================
# Standard Streaming Miner (using one building block)
# ============================================================


class BasicMiner(StreamingMiner):
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

    def state_probs(self, state: ComposedState | None) -> ProbDistr:
        return self.algorithm.state_probs(state)

    def case_probs(self, case_id: CaseId) -> ProbDistr:
        return self.algorithm.case_probs(case_id)

    def sequence_probs(self, sequence: list[ActionName]) -> ProbDistr:
        return self.algorithm.sequence_probs(sequence)


# ============================================================
# Multi Streaming Miner (using several building blocks)
# ============================================================


class MultiMiner(StreamingMiner, ABC):
    def __init__(self, *args, models: list[StreamingMiner], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.models = models

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


# ============================================================
# Ensemble Methods Derived from Multi Streaming Miner
# ============================================================


class HardVoting(MultiMiner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def voting_probs(self, probs_list: list[ProbDistr]) -> ProbDistr:
        """
        Perform hard voting based on the most frequent action in the predictions and return
        the winning action as a probability dictionary with a probability of 1.0.
        If there is a tie, select the action based on the first occurrence in the order of the models.
        """
        # Collect valid predictions
        valid_predictions = []
        for probs in probs_list:
            prediction = probs_prediction(probs, self.config)
            if prediction:
                valid_predictions.append(prediction)

        if not valid_predictions:
            return {}

        # Extract only the action part of each valid prediction for voting
        action_predictions = [pred["action"] for pred in valid_predictions]

        # Count the frequency of each action in the valid predictions
        action_counter = Counter(action_predictions)

        # Find the action(s) with the highest count
        most_common = action_counter.most_common()  # List of (action, count) sorted by frequency

        # Get the highest count
        highest_count = most_common[0][1]
        most_voted_actions = [action for action, count in most_common if count == highest_count]

        selected_action = STOP

        # If there is only one action with the highest count, select that action
        if len(most_voted_actions) == 1:
            selected_action = most_voted_actions[0]
        else:
            # In case of a tie, choose based on the first occurrence among the models' input
            for pred in valid_predictions:
                if pred["action"] in most_voted_actions:
                    selected_action = pred["action"]
                    break

        # Create a result dictionary with only the selected action
        return {STOP: 0.0, selected_action: 1.0}  # include STOP as an invariant

    def state_probs(self, state: ComposedState | None) -> ProbDistr:
        """
        Return the majority vote.
        """
        if state is None:
            return {}

        probs_list = [model.state_probs(model_state) for model, model_state in zip(self.models, state, strict=True)]

        return self.voting_probs(probs_list)

    def case_probs(self, case_id: CaseId) -> ProbDistr:
        """
        Return the hard voting of predictions from the ensemble.
        """
        probs_list = [model.case_probs(case_id) for model in self.models]

        return self.voting_probs(probs_list)

    def sequence_probs(self, sequence: list[ActionName]) -> ProbDistr:
        """
        Return the majority vote.
        """
        probs_list = [model.sequence_probs(sequence) for model in self.models]

        return self.voting_probs(probs_list)


class SoftVoting(MultiMiner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def voting_probs(probs_list: list[ProbDistr]) -> ProbDistr:
        combined_probs = {}

        # Iterate over all probability dictionaries and accumulate the probabilities for each action
        for prob_dict in probs_list:
            for action, prob in prob_dict.items():
                if action not in combined_probs:
                    combined_probs[action] = 0.0
                combined_probs[action] += prob

        # If there are no actions, return an empty dictionary
        if not combined_probs:
            return {}

        # Normalize the combined probabilities so that they sum to 1
        total_prob = sum(combined_probs.values())

        # Ensure we do not divide by zero (though combined_probs being empty is already checked)
        if total_prob > 0:
            combined_probs = {action: prob / total_prob for action, prob in combined_probs.items()}

        # Return the normalized probability dictionary
        return combined_probs

    def state_probs(self, state: ComposedState | None) -> ProbDistr:
        """
        Return the majority vote.
        """
        if state is None:
            return {}

        probs_list = [model.state_probs(model_state) for model, model_state in zip(self.models, state, strict=True)]

        return self.voting_probs(probs_list)

    def case_probs(self, case_id: CaseId) -> ProbDistr:
        """
        Return the hard voting of predictions from the ensemble.
        """
        probs_list = [model.case_probs(case_id) for model in self.models]

        return self.voting_probs(probs_list)

    def sequence_probs(self, sequence: list[ActionName]) -> ProbDistr:
        """
        Return the majority vote.
        """
        probs_list = [model.sequence_probs(sequence) for model in self.models]

        return self.voting_probs(probs_list)


class AdaptiveVoting(MultiMiner):
    """
    To be used only in streaming mode.
    In batch mode, it will stick to the model with the highest training accuracy.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Initialize prediction tracking for each model
        self.total_predictions = 0
        self.correct_predictions = [0] * len(self.models)

    def update(self, case_id: CaseId, action: ActionName) -> None:
        """
        Overwritten to account for keeping track of accuracies.
        """
        self.total_predictions += 1

        for i, model in enumerate(self.models):
            prediction = probs_prediction(model.case_probs(case_id), config=self.config)
            if prediction and prediction["action"] == action:
                self.correct_predictions[i] += 1

            model.update(case_id, action)

    def get_accuracies(self) -> list[float]:
        """
        Returns the accuracy of each model as a list of floats.
        """
        total = self.total_predictions
        return [correct / total if total > 0 else 0.0 for correct in self.correct_predictions]

    def select_best_model(self) -> int:
        """
        Returns the index of the model with the highest accuracy.
        """
        accuracies = self.get_accuracies()
        return accuracies.index(max(accuracies))

    def state_probs(self, state: ComposedState | None) -> ProbDistr:
        """
        Return the probability distribution from the model with the best accuracy so far.
        """
        if state is None:
            return {}

        # Get the best model
        best_model_index = self.select_best_model()
        best_model = self.models[best_model_index]

        best_model_state = state[best_model_index]

        return best_model.state_probs(best_model_state)

    def case_probs(self, case_id: CaseId) -> ProbDistr:
        """
        Return the probability distribution from the model with the best accuracy so far.
        """
        # Get the best model
        best_model_index = self.select_best_model()
        best_model = self.models[best_model_index]

        return best_model.case_probs(case_id)

    def sequence_probs(self, sequence: list[ActionName]) -> ProbDistr:
        """
        Return the probability distribution from the model with the best accuracy so far.
        """
        # Get the best model
        best_model_index = self.select_best_model()
        best_model = self.models[best_model_index]

        return best_model.sequence_probs(sequence)


# ============================================================
# Other Models Derived from Multi Streaming Miner
# ============================================================


class Fallback(MultiMiner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def state_probs(self, state: ComposedState | None) -> ProbDistr:
        """
        Return the first non-{} probabilities from the models, cascading through the models in order.
        Each model gets its corresponding state from the ComposedState.
        """
        if state is None:
            return {}

        # Iterate through the models and their corresponding states
        for model, model_state in zip(self.models, state, strict=True):
            probs = model.state_probs(model_state)
            if probs:
                return probs

        # If all models return {}
        return {}

    def case_probs(self, case_id: CaseId) -> ProbDistr:
        """
        Return the first non-{} probabilities from the models, cascading through the models in order.
        """
        for model in self.models:
            probs = model.case_probs(case_id)
            if probs:
                return probs

        # If all models return None
        return {}

    def sequence_probs(self, sequence: list[ActionName]) -> ProbDistr:
        """
        Return the first non-{} probabilities from the models for the given sequence,
        cascading through the models in order.
        """
        for model in self.models:
            probs = model.sequence_probs(sequence)
            if probs:
                return probs

        # If all models return None
        return {}


class Relativize(MultiMiner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if len(self.models) != 2:  # noqa: PLR2004
            msg = "Class Relativize requires two models."
            raise ValueError(msg)

        self.model1 = self.models[0]
        self.model2 = self.models[1]

    def state_probs(self, state: ComposedState | None) -> ProbDistr:
        if state is None:
            return {}

        (state1, state2) = state

        probs = self.model1.state_probs(state1)

        if probs:
            probs = self.model2.state_probs(state2)

        return probs

    def case_probs(self, case_id: CaseId) -> ProbDistr:
        probs = self.model1.case_probs(case_id)

        if probs:
            probs = self.model2.case_probs(case_id)

        return probs

    def sequence_probs(self, sequence: list[ActionName]) -> ProbDistr:
        probs = self.model1.sequence_probs(sequence)

        if probs:
            probs = self.model2.sequence_probs(sequence)

        return probs


# ============================================================
# Alergia
# ============================================================


class Alergia(BasicMiner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_state = self.initial_state

    @staticmethod
    def get_probability_distribution(state: Any) -> ProbDistr:
        probability_distribution = {}

        for input_symbol, transitions in state.transitions.items():
            # Create a dictionary mapping output letters to probabilities for this input symbol
            output_probabilities = {transition[1]: transition[2] for transition in transitions}
            probability_distribution[input_symbol] = output_probabilities

        return probability_distribution["in"]

    def case_probs(self, case_id: CaseId) -> ProbDistr:  # noqa: ARG002
        """
        This method is not used in this subclass.
        """
        return {}

    def update(self, case_id: CaseId, action: ActionName) -> None:
        """
        This method is not used in this subclass.
        """

    def state_probs(self, state: Any) -> ProbDistr:
        return self.get_probability_distribution(state)

    def sequence_probs(self, sequence: list[ActionName]) -> ProbDistr:
        transformed_sequence = add_input_symbols_sequence(sequence, "in")

        self.algorithm.reset_to_initial()

        for symbol in transformed_sequence:
            self.algorithm.step_to(symbol[0], symbol[1])

        # Get probability distribution for the current state
        return self.get_probability_distribution(self.algorithm.current_state)

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


class NeuralNetworkMiner(StreamingMiner):
    def __init__(self, *args, model: RNNModel | LSTMModel, batch_size: int, optimizer, criterion, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model  # The neural network
        self.optimizer = optimizer
        self.criterion = criterion

        self.sequences: OrderedDict[CaseId, list[ActionName]] = (
            OrderedDict()
        )  # Ordered dictionary to maintain insertion order
        self.rr_index = 0  # Keeps track of the round-robin index
        self.batch_size = batch_size

        self.action_index = {}
        self.index_action = {}

    def get_sequences(self):
        """
        Return all sequences stored in the state.
        """
        return self.sequences

    def get_sequence(self, case_id: CaseId) -> list[ActionName]:
        """
        Return the sequence for a specific case_id.
        """
        return self.sequences.get(case_id, [])

    def update(self, case_id: CaseId, action: ActionName) -> None:
        """
        Add an action to the sequence corresponding to the case_id.
        Dynamically update the activity_to_idx mapping if a new action is encountered.
        """

        # print("begin update")
        # update_start_time = time.time()

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
        # start_time = time.time()
        batch = self.select_batch(case_id)
        # end_time = time.time()
        # print(f"select batch, time needed: {(end_time - start_time) * 1000}")

        # Ensure each sequence in the batch has at least two tokens
        if len(batch) == 0:
            msg = "Skipping training step because no valid sequences were found."
            logger.info(msg)
            return None

        # Convert the batch of sequences into tensors, padding them to the same length
        # start_time = time.time()
        batch_sequences = [torch.tensor(seq, dtype=torch.long) for seq in batch]
        x_batch = pad_sequence(batch_sequences, batch_first=True, padding_value=0)
        # end_time = time.time()
        # print(f"convert batch, time needed: {(end_time - start_time) * 1000}")

        # Input is all but the last token in each sequence, target is shifted by one position
        x_input = x_batch[:, :-1]  # Input sequence
        y_target = x_batch[:, 1:].reshape(-1)  # Flatten the target for CrossEntropyLoss

        self.optimizer.zero_grad()

        # Forward pass through the model
        # start_time = time.time()
        outputs = self.model(x_input)
        # end_time = time.time()
        # print(f"forward pass, time needed: {(end_time - start_time) * 1000}")

        # start_time = time.time()
        # Reshape outputs to [batch_size * sequence_length, vocab_size] for loss calculation
        outputs = outputs.view(-1, outputs.shape[-1])

        # Create a mask to ignore padding (y_target == 0)
        mask = y_target != 0  # Mask out padding positions

        # Apply the mask
        outputs = outputs[mask]
        y_target = y_target[mask]
        # end_time = time.time()
        # print(f"reshape & mask, time needed: {(end_time - start_time) * 1000}")

        # Compute loss
        # start_time = time.time()
        loss = self.criterion(outputs, y_target)
        # end_time = time.time()
        # print(f"compute loss, time needed: {(end_time - start_time) * 1000}")

        # Backward pass and gradient clipping
        # start_time = time.time()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # end_time = time.time()
        # print(f"backward pass, time needed: {(end_time - start_time) * 1000}")

        # start_time = time.time()
        self.optimizer.step()
        # end_time = time.time()
        # print(f"optimizer step, time needed: {(end_time - start_time) * 1000}")

        # update_end_time = time.time()
        # print(f"end update, time needed: {(update_end_time - update_start_time) * 1000}")
        # print("==========")

        return loss.item()

    def select_batch(self, case_id: CaseId) -> list[list[ActionName]]:
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

    def case_probs(self, case_id: CaseId) -> ProbDistr:
        """
        Predict the next action for a given case_id and return the top-k most likely actions along with the probability
        of the top action.

        Note that, here, a sequence is a sequence of action indices (rather than actions).
        """

        # Get the sequence for the case_id
        index_sequence = self.get_sequence(case_id)

        if not index_sequence or len(index_sequence) < 1:
            return {}

        return self.idx_sequence_probs(index_sequence)

    def sequence_probs(self, sequence: list[ActionName]) -> ProbDistr:
        """
        Predict the next action for a given sequence of actions and return the top-k most likely actions along with the
        probability of the top action.
        """
        if not sequence or len(sequence) < 1:
            return {}

        # Convert each action name to its corresponding index, return None if any action is unknown
        index_sequence = []
        for action in sequence:
            action_idx = self.action_index.get(action)
            if action_idx is None:
                return {}  # Return None if the action is not found in the index
            index_sequence.append(action_idx)

        return self.idx_sequence_probs(index_sequence)

    def idx_sequence_probs(self, index_sequence: list[ActionName]) -> ProbDistr:
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

        # Convert the tensor to a list of probabilities
        probabilities = probabilities.squeeze(0).tolist()  # Shape [vocab_size]

        return {
            self.index_action[idx]: prob
            for idx, prob in enumerate(probabilities)
            if self.index_action.get(idx) is not None
        }

    def next_state(self, *args, **kwargs):
        pass  # Or return None, depending on your base class interface

    def state_probs(self, state: ComposedState | None) -> ProbDistr:  # noqa: ARG002
        return {}
