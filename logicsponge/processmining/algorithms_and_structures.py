import logging
from abc import ABC, abstractmethod
from collections import deque

from logicsponge.processmining.automata import PDFA, State
from logicsponge.processmining.types import (
    ActivityName,
    CaseId,
    Event,
    ProbDistr,
    StateId,
)

logger = logging.getLogger(__name__)


# ============================================================
# Base Structure
# ============================================================


class BaseStructure(PDFA, ABC):
    def __init__(self, *args, min_total_visits: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.case_info = {}  # provides state info

        self.last_transition = None

        self.min_total_visits = min_total_visits

        # create initial state
        self.initial_state = 0  # initial_state is always a valid StateId in BaseStructure
        initial_state_object = self.create_state(state_id=self.initial_state)
        self.set_initial_state(self.initial_state)
        self.state_info[self.initial_state]["object"] = initial_state_object
        self.state_info[self.initial_state]["access_string"] = ()

    @property
    def states(self) -> list[StateId]:
        return list(self.state_info.keys())

    def initialize_case(self, case_id: CaseId):
        self.case_info[case_id] = {}
        self.case_info[case_id]["state"] = self.initial_state
        self.state_info[self.initial_state]["total_visits"] += 1
        self.state_info[self.initial_state]["active_visits"] += 1

    def get_visit_statistics(self) -> tuple[int, float]:
        """
        Returns the maximum total visits and the average total visits over all states.
        :return: (max_total_visits, avg_total_visits)
        """
        total_visits_values = [state_info["total_visits"] for state_info in self.state_info.values()]

        if not total_visits_values:
            return 0, 0  # If no states have been visited yet, return 0 for both values

        # Calculate the maximum total visits
        max_total_visits = max(total_visits_values)

        # Calculate the average total visits
        avg_total_visits = sum(total_visits_values) / len(total_visits_values)

        return max_total_visits, avg_total_visits

    def parse_sequence(self, sequence: list[Event]) -> StateId | None:
        current_state = self.initial_state

        # Follow the given sequence of activities through the (P)DFA
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

    def create_state(self, state_id: StateId | None = None) -> State:
        """
        Creates and initializes a new state with the given name and state ID.
        If no state ID is provided, ID is assigned based on current number of states.
        """
        if state_id is None:
            state_id = len(self.state_info)
        new_state = State(state_id=state_id)

        self.state_info[state_id] = {}
        self.state_info[state_id]["object"] = new_state
        self.state_info[state_id]["total_visits"] = 0
        self.state_info[state_id]["activity_frequency"] = {}
        self.state_info[state_id]["active_visits"] = 0
        self.state_info[state_id]["level"] = 0
        self.state_info[state_id]["access_string"] = None

        self.transitions[state_id] = {}

        return new_state

    @abstractmethod
    def update(self, event: Event) -> None:
        """
        Updates DFA tree structure of the process miner object by adding a new activity to case
        """

    def next_state(self, state: StateId | None, activity: ActivityName) -> StateId | None:
        if state is None or state not in self.transitions or activity not in self.transitions[state]:
            return None

        return self.transitions[state][activity]

    def state_probs(self, state: StateId | None) -> ProbDistr:
        """
        Returns probabilities based on state.
        """
        # Return {} if the current state is invalid or has insufficient visits
        if state is None or self.state_info.get(state, {}).get("total_visits", 0) < self.min_total_visits:
            return {}

        return self.get_probabilities(state)

    def case_probs(self, case_id: CaseId) -> ProbDistr:
        """
        Returns probabilities based on case.
        """
        state = self.initial_state if case_id not in self.case_info else self.case_info[case_id].get("state", None)

        return self.state_probs(state)

    def sequence_probs(self, sequence: list[Event]) -> ProbDistr:
        """
        Returns probabilities based on sequence.
        """
        state = self.parse_sequence(sequence)

        return self.state_probs(state)


# ============================================================
# Frequency Prefix Tree
# ============================================================


class FrequencyPrefixTree(BaseStructure):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_transition = None

    def update(self, event: Event) -> None:
        """
        Updates DFA tree structure of the process miner object by adding a new activity to case
        """
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
            self.state_info[current_state]["activity_frequency"][activity] = 0
            next_state = self.create_state().state_id
            self.transitions[current_state][activity] = next_state
            access_string = self.state_info[current_state]["access_string"] + (activity,)
            self.state_info[next_state]["access_string"] = access_string

        self.case_info[case_id]["state"] = next_state
        self.state_info[next_state]["total_visits"] += 1
        self.state_info[current_state]["activity_frequency"][activity] += 1
        self.state_info[current_state]["active_visits"] -= 1
        self.state_info[next_state]["active_visits"] += 1

        self.last_transition = (current_state, activity, next_state)


# ============================================================
# N-Gram
# ============================================================


class NGram(BaseStructure):
    def __init__(self, *args, window_length: int = 1, recover_lengths: list[int] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window_length = window_length

        # recover lengths are by default [self.window_length, ..., 1, 0]
        self.recover_lengths = list(range(self.window_length, -1, -1)) if recover_lengths is None else recover_lengths

        # Maps access string to its state; will be used to do backtracking in inference if transition is not possible.
        self.access_strings = {(): self.initial_state}

    def follow_path(self, sequence: list[ActivityName]) -> StateId:
        """
        Follows the given activity_sequence starting from the root (initial state).
        If necessary, creates new states along the path. Does not modify state counts.

        :param sequence: A list of activity names representing the path to follow.
        :return: The state_id of the final state reached after following the sequence.
        """
        current_state = self.initial_state

        for activity in sequence:
            # Initialize transitions for the current state if not already present
            if current_state not in self.transitions:
                self.transitions[current_state] = {}

            # Follow existing transitions, or create a new state and transition if necessary
            if activity in self.transitions[current_state]:
                current_state = self.transitions[current_state][activity]
            else:
                next_state = self.create_state().state_id
                access_string = self.state_info[current_state]["access_string"] + (activity,)
                self.state_info[next_state]["access_string"] = access_string
                self.access_strings[access_string] = next_state
                self.state_info[next_state]["level"] = self.state_info[current_state]["level"] + 1
                self.transitions[current_state][activity] = next_state
                current_state = next_state

        return current_state

    def update(self, event: Event) -> None:
        """
        Updates DFA tree structure of the process miner object by adding a new activity to case
        """
        case_id = event["case_id"]
        activity = event["activity"]

        self.add_activity(activity)

        if case_id not in self.case_info:
            self.initialize_case(case_id)
            self.case_info[case_id]["suffix"] = deque(maxlen=self.window_length)

        self.case_info[case_id]["suffix"].append(activity)
        current_state = self.case_info[case_id]["state"]
        current_state_level = self.state_info[current_state]["level"]

        if current_state not in self.transitions:
            self.transitions[current_state] = {}

        if activity in self.transitions[current_state]:
            next_state = self.transitions[current_state][activity]
        else:
            if current_state_level < self.window_length:
                next_state = self.create_state().state_id
                self.state_info[next_state]["level"] = current_state_level + 1
                access_string = self.state_info[current_state]["access_string"] + (activity,)
                self.state_info[next_state]["access_string"] = access_string
                self.access_strings[access_string] = next_state
            else:
                next_state = self.follow_path(self.case_info[case_id]["suffix"])
            self.transitions[current_state][activity] = next_state

        self.case_info[case_id]["state"] = next_state
        self.state_info[next_state]["total_visits"] += 1
        if activity in self.state_info[current_state]["activity_frequency"]:
            self.state_info[current_state]["activity_frequency"][activity] += 1
        else:
            self.state_info[current_state]["activity_frequency"][activity] = 1
        self.state_info[current_state]["active_visits"] -= 1
        self.state_info[next_state]["active_visits"] += 1

        self.last_transition = (current_state, activity, next_state)

    def next_state(self, state: StateId | None, activity: ActivityName) -> StateId | None:
        """
        Overwrites next_state from superclass to implement backtracking.
        """
        if state is None:
            return None

        next_state = super().next_state(state, activity)

        if next_state is not None:
            return next_state

        # Trying to recover
        full_access_string = self.state_info[state]["access_string"] + (activity,)

        for i in self.recover_lengths:
            access_string = () if i == 0 else full_access_string[-i:]
            next_state = self.access_strings.get(access_string, None)
            if next_state is not None:
                return next_state

        return None


# ============================================================
# Bag Miner
# ============================================================


class Bag(BaseStructure):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        initial_set: frozenset = frozenset()
        self.state_info[self.initial_state]["activity_set"] = frozenset()
        self.activity_sets: dict[frozenset, StateId] = {initial_set: self.initial_state}

    def update(self, event: Event) -> None:
        """
        Updates DFA tree structure of the process miner object by adding a new activity to case
        """
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
            self.state_info[current_state]["activity_frequency"][activity] = 0

            current_set = self.state_info[current_state]["activity_set"]
            next_set = current_set.union({activity})
            if next_set in self.activity_sets:
                next_state = self.activity_sets[next_set]
            else:
                next_state = self.create_state().state_id
                self.state_info[next_state]["activity_set"] = next_set
                self.activity_sets[next_set] = next_state

            self.transitions[current_state][activity] = next_state

        self.case_info[case_id]["state"] = next_state
        self.state_info[next_state]["total_visits"] += 1
        self.state_info[current_state]["activity_frequency"][activity] += 1
        self.state_info[current_state]["active_visits"] -= 1
        self.state_info[next_state]["active_visits"] += 1

        self.last_transition = (current_state, activity, next_state)


# ============================================================
# Parikh Miner
# ============================================================


class Parikh(BaseStructure):
    def __init__(self, *args, upper_bound: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        initial_vector: dict[ActivityName, int] = {}
        self.state_info[self.initial_state]["parikh_vector"] = {}
        self.parikh_vectors: dict[str, StateId] = {self.parikh_hash(initial_vector): self.initial_state}
        self.upper_bound = upper_bound

    @staticmethod
    def parikh_hash(d: dict) -> str:
        return str(sorted(d.items()))

    def update(self, event: Event) -> None:
        """
        Updates DFA tree structure of the process miner object by adding a new activity to case
        """
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
            self.state_info[current_state]["activity_frequency"][activity] = 0

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
                next_state = self.create_state().state_id
                self.state_info[next_state]["parikh_vector"] = next_vector
                self.parikh_vectors[hashed_next_vector] = next_state

            self.transitions[current_state][activity] = next_state

        self.case_info[case_id]["state"] = next_state
        self.state_info[next_state]["total_visits"] += 1
        self.state_info[current_state]["activity_frequency"][activity] += 1
        self.state_info[current_state]["active_visits"] -= 1
        self.state_info[next_state]["active_visits"] += 1

        self.last_transition = (current_state, activity, next_state)
