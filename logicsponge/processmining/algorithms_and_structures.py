import copy
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import timedelta
from typing import Any

from logicsponge.processmining.automata import PDFA
from logicsponge.processmining.types import (
    ActivityDelays,
    ActivityName,
    CaseId,
    Event,
    Metrics,
    ProbDistr,
    StateId,
    empty_metrics,
)
from logicsponge.processmining.utils import compute_perplexity_stats

logger = logging.getLogger(__name__)


# ============================================================
# Bayesian Classifier
# ============================================================


class BayesianClassifier:
    """Bayesian Classifier for predicting the next activity in a sequence of events.

    This class implements a simple Bayesian classifier that uses the frequency of
    activity sequences to predict the next activity. It is designed to work with
    event logs, where each event has an associated activity name.
    """

    def __init__(
            self,
            config: dict | None = None,
            *,
            single_occurence_allowed: bool = True,
        ) -> None:
        """Initialize the BayesianClassifier with the given configuration."""
        if config is None:
            config = {}
        self.memory: dict[tuple[ActivityName, ...], dict[ActivityName, int]] = {}

        self.config = (
            config
            if config
            else {
                "top_k": 1,
            }
        )

        # Statistics for batch mode
        self.stats = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "wrong_predictions": 0,
            "empty_predictions": 0,
            "top_k_correct_preds": [0] * self.config["top_k"],
            # For perplexity
            "pp_arithmetic_mean": None,
            "pp_harmonic_mean": None,
            "pp_median": None,
            "pp_q1": None,
            "pp_q3": None,
            # For delay predictions
            "delay_error_sum": 0,
            "actual_delay_sum": 0,
            "normalized_error_sum": 0,
            "num_delay_predictions": 0,
            "last_timestamps": {},  # last recorded timestamp for every case
        }

        self.single_occurence_allowed = single_occurence_allowed

    def initialize_memory(self, data: list[list[Event]]) -> None:
        """Initialize weights of the Bayes classifier."""
        mem_frequency: dict[tuple[ActivityName, ...], dict[ActivityName, int]] = {}
        for sequence in data:
            prefix = []
            for i in range(len(sequence)):
                event = sequence[i]
                activity = event.get("activity")

                if tuple(prefix) not in mem_frequency:
                    # Initialize the dictionary for the prefix if it doesn't exist
                    mem_frequency[tuple(prefix)] = {}
                if activity not in mem_frequency[tuple(prefix)]:
                    # Initialize the dictionary for the next activity if it doesn't exist
                    mem_frequency[tuple(prefix)][activity] = 0

                # Update the frequency of the next activity
                mem_frequency[tuple(prefix)][activity] += 1
                prefix.append(activity)

        self.memory = mem_frequency

    def evaluate(
        self,
        data: list[list[Event]],
        mode: str = "",
        *,
        log_likelihood: bool = False,
        debug: bool = False,
    ) -> float:
        """Evaluate the dataset using a Bayes classifier."""
        perplexities = []

        eval_start_time = time.time()
        pause_time = 0.0

        for sequence in data:
            prefix = []
            likelihood = 0.0 if log_likelihood else 1.0

            for i in range(len(sequence)):
                event = sequence[i]

                actual_activity = event.get("activity")

                bayes_prediction = self._get_bayes_prediction(prefix)

                pause_start_time = time.time()
                if log_likelihood:
                    likelihood += math.log(self._get_conditional_likelihood(prefix, actual_activity))
                else:
                    likelihood *= self._get_conditional_likelihood(prefix, actual_activity)

                if bayes_prediction is None:
                    self.stats["empty_predictions"] += 1
                elif bayes_prediction[0] == actual_activity:
                    self.stats["correct_predictions"] += 1
                    for indices_top_k in range(len(self.stats["top_k_correct_preds"])):
                        self.stats["top_k_correct_preds"][indices_top_k] += 1
                else:
                    self.stats["wrong_predictions"] += 1

                    for k in range(len(bayes_prediction)):
                        if actual_activity == bayes_prediction[k]:
                            for indices_top_k in range(k, len(self.stats["top_k_correct_preds"])):
                                self.stats["top_k_correct_preds"][indices_top_k] += 1
                            break

                pause_time += time.time() - pause_start_time

                self.stats["total_predictions"] += 1
                prefix.append(actual_activity)

            pause_start_time = time.time()
            # Normalize by the length of the sequence
            if log_likelihood:
                normalized_likelihood = likelihood / len(sequence) if len(sequence) > 0 else likelihood
            else:
                normalized_likelihood = likelihood ** (1 / len(sequence)) if len(sequence) > 0 else likelihood

            if normalized_likelihood is not None and normalized_likelihood > 0:
                seq_perplexity = math.exp(-normalized_likelihood) if log_likelihood else 1.0 / normalized_likelihood
            else:
                seq_perplexity = float("inf")

            perplexities.append(seq_perplexity)
            pause_time += time.time() - pause_start_time

        eval_time = time.time() - eval_start_time - pause_time

        perplexity_stats = compute_perplexity_stats(perplexities)
        logger.debug("Perplexity stats: %s", perplexity_stats)

        for key, value in perplexity_stats.items():
            self.stats[key] = value

        return eval_time

    def _get_conditional_likelihood(self, prefix: list[ActivityName], activity: ActivityName) -> float:
        """Return the predicted activity based on the Bayes classifier."""
        if not self.memory:
            msg = "Memory not initialized. Please call initialize_memory() first."
            raise ValueError(msg) from None

        prefix_act = tuple(prefix)
        if prefix_act in self.memory:
            if (not self.single_occurence_allowed) and (sum(self.memory[prefix_act].values()) == 1):
                return 0.0

            # Get the next activity with the highest frequency
            next_activities = self.memory[prefix_act]
            if activity not in next_activities:
                # If the activity is not in the next activities, return 0.0
                return 0.0

            # Calculate the conditional likelihood
            total_count = sum(next_activities.values())
            activity_count = next_activities[activity]

            return float(activity_count) / total_count

        return 0.0

    def _get_bayes_prediction(self, prefix: list[ActivityName]) -> list[ActivityName] | None:
        """Return the predicted activity based on the Bayes classifier."""
        if not self.memory:
            # Check if memory is initialized
            msg = "Memory not initialized. Please call initialize_memory() first."
            raise ValueError(msg) from None

        prefix_act = tuple(prefix)
        if prefix_act in self.memory:
            if (not self.single_occurence_allowed) and (sum(self.memory[prefix_act].values()) == 1):
                # If the priefix has only one occurrence, return None (no prediction)
                logger.debug(
                    "Prefix %s has only one occurrence. No prediction available (not allowed on single prefix).", prefix
                )
                return None

            # Get the next activity with the highest frequency
            next_activities = self.memory[prefix_act]
            sorted_activities: list[ActivityName] = sorted(
                next_activities, key=lambda activity: next_activities[activity], reverse=True
            )
            return sorted_activities[: self.config.get("top_k", 1)]

        logger.debug("No prediction available for prefix %s, %s.", prefix, len(prefix))
        return None


# ============================================================
# Base Structure
# ============================================================


class BaseStructure(PDFA, ABC):
    """Base structure for process mining."""

    case_info: dict[CaseId, Any]
    last_transition: tuple[StateId, ActivityName, StateId] | None
    min_total_visits: int
    min_max_prob: float
    modified_cases: set[CaseId]

    def __init__(
            self,
            *args,
            min_total_visits: int = 1,
            min_max_prob: float = 0.0,
            **kwargs
        ) -> None:
        """Initialize the BaseStructure."""
        super().__init__(*args, **kwargs)

        self.case_info = {}  # provides case info such as current state or last timestamp
        self.last_transition = None
        self.min_total_visits = min_total_visits
        self.min_max_prob = min_max_prob

        self.modified_cases = set()  # Records potentially modified cases (predictions) in last update

        # create initial state
        self.initial_state = self.create_state()
        self.state_info[self.initial_state]["access_string"] = ()
        self.state_info[self.initial_state]["level"] = 0

    def get_modified_cases(self) -> set[CaseId]:
        """Retrieve, recursively, cases that have potentially been modified and whose prediction needs to be updated."""
        return self.modified_cases

    @property
    def states(self) -> list[StateId]:
        """Returns a list of state IDs."""
        return list(self.state_info.keys())

    def create_state(self, state_id: StateId | None = None) -> StateId:
        """Overwrite Automata method.

        Creates and initializes a new state with the given state ID.
        If no state ID is provided, ID is assigned based on current number of states.
        """
        if state_id is None:
            state_id = len(self.state_info)

        self.state_info[state_id] = {}
        self.state_info[state_id]["total_visits"] = 0
        self.state_info[state_id]["active_visits"] = 0
        self.state_info[state_id]["active_cases"] = set()
        self.state_info[state_id]["activity_frequency"] = {}
        self.state_info[state_id]["time_info"] = {
            "delays": {},  # list of delays (as deque of floats in seconds) for every activity
            "rolling_sum": {},  # sum of delays for every activity
            "predicted_delay": {},  # predicted delay for every activity (currently based on mean of delays)
        }
        self.state_info[state_id]["access_string"] = None
        self.state_info[state_id]["level"] = None
        self.transitions[state_id] = {}

        return state_id

    def initialize_case(self, case_id: CaseId) -> None:
        """Initialize case-specific information for a given case ID."""
        self.case_info[case_id] = {}
        self.case_info[case_id]["state"] = self.initial_state
        self.case_info[case_id]["last_timestamp"] = None
        self.state_info[self.initial_state]["total_visits"] += 1
        self.state_info[self.initial_state]["active_visits"] += 1
        self.state_info[self.initial_state]["active_cases"].add(case_id)

    def initialize_activity(self, state_id: StateId, activity: ActivityName) -> None:
        """Initialize activity-specific information for a given state."""
        # Initialize activity frequency
        self.state_info[state_id]["activity_frequency"][activity] = 0

        # Initialize timing information
        self.state_info[state_id]["time_info"]["delays"][activity] = deque(maxlen=self.config.get("maxlen_delays", 500))
        self.state_info[state_id]["time_info"]["rolling_sum"][activity] = 0

    def parse_sequence(self, sequence: list[Event]) -> StateId | None:
        """Parse a sequence of events and returns the final state reached in the underlying (P)DFA."""
        current_state = self.initial_state

        # Follow the given sequence of activities through the underlying (P)DFA
        for event in sequence:
            activity = event["activity"]
            if activity in self.activities:
                if current_state in self.transitions and activity in self.transitions[current_state]:
                    current_state = self.transitions[current_state][activity]
                else:
                    # Sequence diverges, no matching transition
                    return None
            else:
                return None

        return current_state

    def get_probabilities(self, state_id: StateId) -> ProbDistr:
        """Return the probability distribution of activities for a given state."""
        total_visits = self.state_info[state_id]["total_visits"]
        probs = {self.config["stop_symbol"]: 0.0}  # Initialize the probabilities dictionary with STOP activity

        # Update the probability for each activity based on visits to successors
        for activity in self.activities:
            if activity in self.state_info[state_id]["activity_frequency"] and total_visits > 0:
                # Compute probability based on activity frequency and total visits
                probs[activity] = self.state_info[state_id]["activity_frequency"][activity] / total_visits
            else:
                # If activity is not present or there were no visits, set probability to 0
                probs[activity] = 0.0

        # Sum the probabilities for all activities (excluding STOP)
        activity_sum = sum(prob for activity, prob in probs.items() if activity != self.config["stop_symbol"])

        # Ensure that the probabilities are correctly normalized
        if activity_sum > 1:
            for activity in self.activities:
                # Adjust the probability proportionally so that their total sum is 1
                probs[activity] /= activity_sum

        # Compute the "STOP" probability as the remainder to ensure all probabilities sum to 1
        probs[self.config["stop_symbol"]] = max(0.0, 1.0 - activity_sum)

        return probs

    def get_predicted_delays(self, state: StateId) -> ActivityDelays:
        """Return the predicted delays for a given state."""
        return copy.deepcopy(self.state_info[state]["time_info"]["predicted_delay"])

    def get_metrics(self, state: StateId) -> Metrics:
        """Combine probabilities and delays for a given state into a single metrics dictionary."""
        probs = self.get_probabilities(state)
        # activity_list = list(probs)

        return Metrics(
            state_id=state,
            probs=probs,
            predicted_delays=self.get_predicted_delays(state),
            # likelihoods=self.state_act_likelihoods(state, activity_list)
        )

    @abstractmethod
    def update(self, event: Event) -> None:
        """Update DFA tree structure of the process miner object by adding a new activity to case."""

    def _sequence_likelihood(self, sequence: list[Event]) -> float:
        """Return the likelihood of a sequence of events.

        WARNING: INCORRECT IMPLEMENTATION!!!
        """
        msg = "This method is not correctly implemented."
        raise NotImplementedError(msg)
        logger.debug("  Call to _sequence_likelihood with sequence: %s", sequence)
        likelihood = 1.0

        current_state = self.initial_state

        for event in sequence:
            activity = event["activity"]

            # Check if the activity is valid for the current state
            if current_state in self.transitions and activity in self.transitions[current_state]:
                next_state = self.transitions[current_state][activity]
            elif activity == self.config["stop_symbol"]:
                logger.debug(
                    "     -> STOP activity found in sequence. State: %s. Dist: %s",
                    current_state,
                    self.state_info[current_state],
                )
                break
            else:
                logger.debug(
                    "     -> [!] Activity '%s' not found in state %s, at length %d.",
                    activity,
                    current_state,
                    len(sequence),
                )
                return 0.0
                # If not, follow the path to find the next state
                # next_state = self.follow_path([activity])

            likelihood *= float(self.state_info[current_state]["activity_frequency"].get(activity, 0)) / max(
                1, self.state_info[current_state]["total_visits"]
            )
            logger.debug(
                "     -> New Likelihood: %s. Activity: %s. State: %s. Dist: %s",
                likelihood,
                activity,
                current_state,
                self.state_info[current_state],
            )

            # likelihood *= (
            #     float(self.state_info[current_state]["activity_frequency"].get(activity, 0))
            #     / max(1, sum(self.state_info[current_state]["activity_frequency"].values()))
            # )

            logger.debug("Total visits: %s", self.state_info[current_state]["total_visits"])
            logger.debug("Likelihood: %s", likelihood)

            # Move to the next state
            current_state = next_state

        # Normalize likelihood by the length of the sequence (Geometric mean)
        likelihood = likelihood ** (1 / len(sequence)) if len(sequence) > 0 else likelihood
        logger.debug("Sequence Likelihood Fun: %s", likelihood)
        return likelihood

    def next_state(self, state: StateId | None, activity: ActivityName) -> StateId | None:
        """Return the next state based on the current state and activity."""
        if state is None or state not in self.transitions or activity not in self.transitions[state]:
            return None

        return self.transitions[state][activity]

    def update_info(self, event: Event, current_state: StateId, next_state: StateId) -> None:
        """Update state and timing information for a given transition."""
        case_id = event["case_id"]
        activity = event["activity"]
        timestamp = event["timestamp"]

        # Update state information
        self.case_info[case_id]["state"] = next_state
        self.state_info[next_state]["total_visits"] += 1
        self.state_info[current_state]["activity_frequency"][activity] += 1
        self.state_info[current_state]["active_visits"] -= 1
        self.state_info[next_state]["active_visits"] += 1
        self.state_info[current_state]["active_cases"].remove(case_id)
        self.state_info[next_state]["active_cases"].add(case_id)

        self.last_transition = (current_state, activity, next_state)  # for visualization

        # Update set of cases potentially modified
        self.modified_cases = set()
        for state in (current_state, next_state):
            for case in self.state_info[state]["active_cases"]:
                self.modified_cases.add(case)

        # Update timing information
        if self.config["include_time"]:
            last_timestamp = self.case_info[case_id].get("last_timestamp")

            if timestamp and last_timestamp:
                delay = (timestamp - last_timestamp).total_seconds()  # Convert timedelta to seconds
                time_info = self.state_info[current_state]["time_info"]

                # Cache dictionary lookups
                activity_delays = time_info["delays"][activity]
                activity_sum = time_info["rolling_sum"][activity]

                # Append delay to the deque and manage rolling sum
                if len(activity_delays) == activity_delays.maxlen:
                    activity_sum -= activity_delays[0]

                activity_delays.append(delay)
                activity_sum += delay

                # Update back into the dictionary
                time_info["rolling_sum"][activity] = activity_sum
                time_info["predicted_delay"][activity] = timedelta(seconds=activity_sum / len(activity_delays))

            # Update the last timestamp
            if timestamp:
                self.case_info[case_id]["last_timestamp"] = timestamp

    # def act_state_likelihood(self, activity: ActivityName, prev_state: StateId) -> float:
    #     """Return the likelihood of the given activity given a previous state."""
    #     # Check if the activity is valid for the next state
    #     if activity not in self.transitions[prev_state]:
    #         logger.debug("     -> [!] Activity %s not found in state %s, returning 0.0.", activity, prev_state)
    #         return 0.0
    #     return self.get_probabilities(prev_state).get(activity, 0.0)


    def state_act_likelihood(self, state: StateId | None, next_activity: ActivityName) -> float:
        """Return the likelihood of the given activity given a current state."""
        # Check if the activity is valid for the current state
        if state not in self.state_info:
            logger.debug("     -> [!] State %s not found in state info, returning 0.0.", state)
            return 0.0

        logger.debug("     -> State: %s. Activities: %s", state, self.state_info[state]["activity_frequency"])
        logger.debug("     -> Next activity: %s", next_activity)

        # BE CAREFUL: we need to use state_metrics (relies on self.get_probabilities(state)) and not state_info
        # otherwise the __stop__ frequencies are ignored!
        # Below is an example of the problem:
        # (self.state_info[state]["activity_frequency"].get(next_activity, 0))
        # / max(1, self.state_info[state]["total_visits"])

        return self.get_probabilities(state).get(next_activity, 0.0)

    def state_act_likelihoods(
        self,
        state: StateId | None,
        eligible_activities: list[ActivityName]
    ) -> dict[ActivityName, float]:
        """Return the likelihood of the given activities given a current state."""
        likelihoods = {}
        for activity in eligible_activities:
            likelihoods[activity] = self.state_act_likelihood(state, activity)
        return likelihoods

    def state_metrics(self, state: StateId | None) -> Metrics:
        """Return metrics based on state."""
        # Return {} if the current state is invalid or has insufficient visits
        if (
            state is None
            or self.state_info.get(state, {}).get("total_visits", 0) < self.min_total_visits
            or max(self.get_probabilities(state).values()) < self.min_max_prob
        ):
            # if state is None:
            #     logger.debug(f"State is None, returning empty metrics.\n")
            # elif self.state_info.get(state, {}).get("total_visits", 0) < self.min_total_visits:
            #     logger.debug(f"State {state} has less than
            # {self.min_total_visits} visits, returning empty metrics.\n")
            # elif max(self.get_probabilities(state).values()) < self.min_max_prob:
            #     logger.debug(f"State {state} has max prob less than {self.min_max_prob}, returning empty metrics.\n")
            return empty_metrics()

        return self.get_metrics(state)

    def get_state_from_case(self, case_id: CaseId) -> StateId:
        """Return the current state of a case."""
        if case_id not in self.case_info:
            return self.initial_state

        return self.case_info[case_id]["state"]

    def case_metrics(self, case_id: CaseId) -> Metrics:
        """Return metrics based on case."""
        state = self.get_state_from_case(case_id)

        return self.state_metrics(state)

    def sequence_metrics(self, sequence: list[Event]) -> Metrics:
        """Return probabilities based on sequence of events."""
        state = self.parse_sequence(sequence)

        metrics = self.state_metrics(state)
        logger.debug("[BaseStructure] State: %s", state)
        logger.debug("[BaseStructure] Metrics: %s", metrics)

        return metrics  # , state is not None


# ============================================================
# Frequency Prefix Tree
# ============================================================


class FrequencyPrefixTree(BaseStructure):
    """Frequency Prefix Tree structure for process mining."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the FrequencyPrefixTree structure."""
        super().__init__(*args, **kwargs)
        self.last_transition = None  # for visualization of frequency prefix tree

    def __str__(self) -> str:
        """Represent as a string of the FrequencyPrefixTree structure."""
        return f"FrequencyPrefixTree(min_total_visits={self.min_total_visits}, min_max_prob={self.min_max_prob})"

    def update(self, event: Event) -> None:
        """Update DFA tree structure of the process miner object by adding a new activity to case."""
        case_id = event["case_id"]
        activity = event["activity"]

        self.add_activity(activity)

        if case_id not in self.case_info:
            self.initialize_case(case_id)

        current_state = self.case_info[case_id]["state"]

        if current_state not in self.transitions:
            self.transitions[current_state] = {}

        if activity in self.transitions[current_state]:
            next_state = self.transitions[current_state][activity]
        else:
            self.initialize_activity(current_state, activity)
            next_state = self.create_state()
            self.transitions[current_state][activity] = next_state
            access_string = self.state_info[current_state]["access_string"] + (activity,)
            self.state_info[next_state]["access_string"] = access_string

        self.update_info(event, current_state, next_state)


# ============================================================
# N-Gram
# ============================================================


class NGram(BaseStructure):
    """NGram structure for process mining."""

    access_strings: dict[tuple[str, ...], StateId]

    def __init__(
        self,
        *args,  # noqa: ANN002
        window_length: int = 1,
        recover_lengths: list[int] | None = None,
        return_to_initial: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the NGram structure."""
        super().__init__(*args, **kwargs)
        self.window_length = window_length
        self.return_to_initial = return_to_initial

        self.recover_lengths = [self.window_length, *(sorted(recover_lengths, reverse=True) if recover_lengths else [])]
        logger.debug("Recover lengths: %s", self.recover_lengths)

        # Maps access string to its state; will be used to do backtracking in inference if transition is not possible.
        self.access_strings = {(): self.initial_state}

    def __str__(self) -> str:
        """Represent as a string of the NGram structure."""
        return (
            f"NGram(window_length={self.window_length}, "
            f"min_total_visits={self.min_total_visits}, "
            f"min_max_prob={self.min_max_prob})"
        )

    def follow_path(self, sequence: list[ActivityName]) -> StateId:
        """Follows the given activity sequence starting from the root (initial state).

        If necessary, creates new states along the path. Does not modify state
        and activity frequency counts.

        :param sequence: A list of activity names representing the path to follow.
        :return: The state of the final state reached after following the sequence.
        """
        current_state = self.initial_state

        for activity in sequence:
            # Initialize transitions for the current state if not already present
            if current_state not in self.transitions:
                self.transitions[current_state] = {}

            # Follow existing transitions, or create a new state and transition if necessary
            if activity in self.transitions[current_state]:
                next_state = self.transitions[current_state][activity]
            else:
                next_state = self.create_state()
                access_string = self.state_info[current_state]["access_string"] + (activity,)
                self.state_info[next_state]["access_string"] = access_string
                self.access_strings[access_string] = next_state
                self.state_info[next_state]["level"] = self.state_info[current_state]["level"] + 1
                self.transitions[current_state][activity] = next_state
                self.initialize_activity(current_state, activity)

            current_state = next_state

        return current_state

    def update(self, event: Event) -> None:
        """Update NGram by adding a new event."""
        case_id = event["case_id"]
        activity = event["activity"]

        self.add_activity(activity)

        if case_id not in self.case_info:
            self.initialize_case(case_id)
            self.case_info[case_id]["suffix"] = deque(maxlen=self.window_length)

        current_state = self.case_info[case_id]["state"]
        current_state_level = self.state_info[current_state]["level"]
        # Note: self.case_info[case_id]["suffix"] equals self.state_info[current_state]["access_string"]
        self.case_info[case_id]["suffix"].append(activity)

        if current_state not in self.transitions:
            self.transitions[current_state] = {}

        if activity in self.transitions[current_state]:
            next_state = self.transitions[current_state][activity]
        else:
            if current_state_level < self.window_length:
                next_state = self.create_state()
                self.state_info[next_state]["level"] = current_state_level + 1
                access_string = self.state_info[current_state]["access_string"] + (activity,)
                self.state_info[next_state]["access_string"] = access_string
                self.access_strings[access_string] = next_state
            else:
                next_state = self.follow_path(self.case_info[case_id]["suffix"])

            self.transitions[current_state][activity] = next_state
            self.initialize_activity(current_state, activity)

        self.update_info(event, current_state, next_state)

    def next_state(self, state: StateId | None, activity: ActivityName) -> StateId | None:
        """Overwrite next_state from superclass to implement backoff (backtracking)."""
        # When state is None (unknown prefix), we initiate the recovery process from the initial state
        state = state if state is not None else self.initial_state

        next_state = super().next_state(abs(state), activity)

        # Mark the recovered state (when not None) with a negative sign
        next_state = (
            -next_state
            if (
                next_state is not None
                and (state is None or state < 0)
                and self.state_info[next_state]["level"] < self.window_length
            )
            else next_state
        )

        if self.state_info[state]["level"] == self.window_length and next_state is None:
            full_access_string = self.state_info[state]["access_string"] + (activity,)
            access_string = full_access_string[-self.window_length :]

            next_state = self.access_strings.get(access_string, None)
            logger.debug("Recovered (window_length) State: %s - Activity: %s - %s", state, activity, next_state)

        return next_state

        # if next_state is not None and self.state_info[next_state]["total_visits"] == 0:
        #     return None
        # if next_state is None or next_state < 0:
        #     return next_state

        # Basic NGRAM state update (recover the new prefix)
        full_access_string = self.state_info[state]["access_string"] + (activity,)

        # access_string = () if self.window_length==0 else full_access_string[-self.window_length:]
        # next_state = self.access_strings.get(access_string, None)
        # if next_state is not None:# and self.state_info[next_state]["level"] == self.window_length:
        #     return next_state

        # Trying to recover
        # truncate recover lengths to the current level
        trunc_recover_lengths = [i for i in self.recover_lengths if i <= self.state_info[state]["level"]]
        # if self.state_info[state]["level"] == self.window_length:
        for i in trunc_recover_lengths:
            access_string = () if i == 0 else full_access_string[-i:]
            next_state = self.access_strings.get(access_string, None)
            logger.debug(f"[{i}] State: {state} - Activity: {activity} - {next_state}") # noqa: G004
            if next_state is not None and self.state_info[next_state]["total_visits"] > 0:
                return next_state

            logger.error(
                f"[!!!]   Recovery failed with access string {full_access_string}", # noqa: G004
                f"for state {state} and activity {activity}"
            )
        # Get number of active visits
        if len(self.recover_lengths) > 1:
            total_visits = self.state_info[state]["total_visits"]
            level = self.state_info[state]["level"]
            logger.debug(
                f"Total visits for state {state}: {total_visits}    -   Level: {level}  / {self.window_length}" # noqa: G004
            )

        logger.debug(f"Recovery failed with access string for state {state} and activity {activity}") # noqa: G004
        return None

    # def sequence_metrics(self, sequence: list[Event]) -> Metrics:
    #     """Return probabilities based on sequence of events."""
    #     # THIS IS VERY WRONG, we need the whole sequence
    #     # sequence = sequence[-self.window_length :] if self.window_length > 0 else []
    #     return super().sequence_metrics(sequence)


# ============================================================
# Bag Miner
# ============================================================


class Bag(BaseStructure):
    """Bag structure for process mining."""

    activity_sets: dict[frozenset, StateId]

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the Bag structure."""
        super().__init__(*args, **kwargs)
        initial_set: frozenset = frozenset()
        self.state_info[self.initial_state]["activity_set"] = frozenset()
        self.activity_sets = {initial_set: self.initial_state}

    def __str__(self) -> str:
        """Represent as a string of the Bag structure."""
        return f"Bag(min_total_visits={self.min_total_visits}, min_max_prob={self.min_max_prob})"

    def update(self, event: Event) -> None:
        """Update DFA tree structure of the process miner object by adding a new activity to case."""
        case_id = event["case_id"]
        activity = event["activity"]

        self.add_activity(activity)

        if case_id not in self.case_info:
            self.initialize_case(case_id)

        current_state = self.case_info[case_id]["state"]

        if current_state not in self.transitions:
            self.transitions[current_state] = {}

        if activity in self.transitions[current_state]:
            next_state = self.transitions[current_state][activity]
        else:
            self.initialize_activity(current_state, activity)

            current_set = self.state_info[current_state]["activity_set"]
            next_set = current_set.union({activity})
            if next_set in self.activity_sets:
                next_state = self.activity_sets[next_set]
            else:
                next_state = self.create_state()
                self.state_info[next_state]["activity_set"] = next_set
                self.activity_sets[next_set] = next_state

            self.transitions[current_state][activity] = next_state

        self.update_info(event, current_state, next_state)


# ============================================================
# Parikh Miner
# ============================================================


class Parikh(BaseStructure):
    """Parikh structure for process mining."""

    parikh_vectors: dict[str, StateId]

    def __init__(self, *args, upper_bound: int | None = None, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the Parikh structure."""
        super().__init__(*args, **kwargs)
        initial_vector: dict[ActivityName, int] = {}
        self.state_info[self.initial_state]["parikh_vector"] = {}
        self.parikh_vectors = {self.parikh_hash(initial_vector): self.initial_state}
        self.upper_bound = upper_bound

    @staticmethod
    def parikh_hash(d: dict) -> str:
        """Return a string representation of the Parikh vector for hashing."""
        return str(sorted(d.items()))

    def update(self, event: Event) -> None:
        """Update DFA tree structure of the process miner object by adding a new activity to case."""
        case_id = event["case_id"]
        activity = event["activity"]

        self.add_activity(activity)

        if case_id not in self.case_info:
            self.initialize_case(case_id)

        current_state = self.case_info[case_id]["state"]

        if current_state not in self.transitions:
            self.transitions[current_state] = {}

        if activity in self.transitions[current_state]:
            next_state = self.transitions[current_state][activity]
        else:
            self.initialize_activity(current_state, activity)

            current_vector = self.state_info[current_state]["parikh_vector"]
            next_vector = current_vector.copy()
            if activity in next_vector:
                if self.upper_bound is not None:
                    next_vector[activity] = min(next_vector[activity] + 1, self.upper_bound)
                else:
                    next_vector[activity] += 1
            elif self.upper_bound is not None:
                next_vector[activity] = min(1, self.upper_bound)
            else:
                next_vector[activity] = 1

            hashed_next_vector = self.parikh_hash(next_vector)
            if hashed_next_vector in self.parikh_vectors:
                next_state = self.parikh_vectors[hashed_next_vector]
            else:
                next_state = self.create_state()
                self.state_info[next_state]["parikh_vector"] = next_vector
                self.parikh_vectors[hashed_next_vector] = next_state

            self.transitions[current_state][activity] = next_state

        self.update_info(event, current_state, next_state)
