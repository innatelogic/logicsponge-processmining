import logging
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import ot
from sortedcontainers import SortedSet
from tqdm import tqdm

from logicsponge.processmining.automata import PDFA, State
from logicsponge.processmining.globals import (
    DISCOUNT,
    RANDOMIZED,
    STOP,
    TOP_K,
    ActionName,
    CaseId,
    Prediction,
    StateId,
)

logger = logging.getLogger(__name__)


# ============================================================
# Base Structure
# ============================================================


class BaseStructure(PDFA, ABC):
    def __init__(
        self, *args, min_total_visits: int = 0, randomized: bool = RANDOMIZED, top_k: int = TOP_K, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.randomized = randomized
        self.top_k = top_k

        self.case_info = {}  # provides state info
        # keys: state, suffix

        self.last_transition = None

        self.min_total_visits = min_total_visits

        # create initial state
        initial_state = 0
        initial_state_object = self.create_state(state_id=initial_state)
        self.set_initial_state(initial_state)
        self.state_info[initial_state]["object"] = initial_state_object

        self.initial_state = 0

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

    def parse_sequence(self, sequence: list[ActionName]) -> StateId | None:
        current_state = self.initial_state

        # Follow the given sequence of actions through the (P)DFA
        for action in sequence:
            if action in self.action_index:
                if current_state in self.transitions and action in self.transitions[current_state]:
                    current_state = self.transitions[current_state][action]
                else:
                    # Sequence diverges, no matching transition
                    return None
            else:
                return None

        return current_state

    def update_state_probs(self, state_id: StateId) -> None:
        """
        Updates the probability distribution for actions originating from a given state.
        The probabilities are based on the ratio of visits to successor states relative
        to the total visits to the current state. They are normalized if
        the sum of action probabilities exceeds 1, and the STOP probability is set
        to the remainder to ensure all probabilities sum to 1.
        """
        total_visits = self.state_info[state_id]["total_visits"]
        probs = [0.0] * (len(self.action_index) + 1)

        # Update the probability for each action based on visits to successors
        for action, action_idx in self.action_index.items():
            if action in self.state_info[state_id]["action_frequency"] and total_visits > 0:
                probs[action_idx] = self.state_info[state_id]["action_frequency"][action] / total_visits
            else:
                probs[action_idx] = 0

        # Sum the probabilities for actions (excluding STOP)
        action_sum = sum(probs[1:])

        # Normalize probabilities if the sum is greater than 1
        if action_sum > 1:
            for i in range(1, len(probs)):
                probs[i] = probs[i] / action_sum

        # Compute the "STOP" probability as the remainder, ensuring it's not negative due to rounding errors
        probs[0] = max(0.0, 1.0 - sum(probs[1:]))

        # Save the updated probability vector
        self.state_info[state_id]["probs"] = probs

    @abstractmethod
    def create_state(self, state_id: StateId | None = None) -> State:
        """
        Creates and initializes a new state with the given state ID.
        If no state ID is provided, ID is assigned based on current number of states.
        """

    @abstractmethod
    def update(self, case_id: CaseId, action: ActionName) -> None:
        """
        Updates DFA tree structure of the process miner object by adding a new action to case
        """

    def prediction_probs(self, probs: list[float] | None) -> Prediction | None:
        """
        Transforms probability vector into a pair (prediction action, top k predicted actions).
        """
        if probs is None:
            return None

        # Get top-k predictions using np.argsort (to handle ties consistently)
        top_k_indices = np.argsort(probs)[-self.top_k :][::-1]  # Get the top k indices with the highest probabilities

        # For random selection, choose based on probabilities; otherwise, choose the max
        if self.randomized:  # noqa: SIM108
            next_action_idx = np.random.choice(len(probs), p=probs)
        else:
            next_action_idx = top_k_indices[0]  # Use the highest probability

        predicted_action = self.index_action[next_action_idx] if next_action_idx != 0 else STOP

        # If the next action is 0, return STOP, otherwise return the corresponding action name
        top_k_actions = [self.index_action[idx] if idx != 0 else STOP for idx in top_k_indices]
        return predicted_action, top_k_actions, probs[next_action_idx]

    def prediction_case(self, case_id: CaseId) -> Prediction | None:
        """
        Makes prediction based for given case_id. Predicts STOP if case_id not present.
        """
        if case_id not in self.case_info:
            return self.prediction_state(self.initial_state)

        if "state" not in self.case_info[case_id]:
            return None

        return self.prediction_state(self.case_info[case_id]["state"])

    def prediction_state(self, current_state: StateId | None) -> Prediction | None:
        """
        Makes prediction based on current state.
        """
        if (
            current_state is None
            or self.state_info.get(current_state, {}).get("total_visits", 0) < self.min_total_visits
        ):
            return None

        default_probs = [1.0] + [0.0] * len(self.action_index)
        probs = self.state_info.get(current_state, {}).get("probs", default_probs)

        return self.prediction_probs(probs)

    def prediction_sequence(self, sequence: list[ActionName]) -> Prediction | None:
        """
        Makes prediction based on sequence.
        """
        current_state = self.parse_sequence(sequence)

        return self.prediction_state(current_state)

    def next_state(self, state: StateId | None, action: ActionName) -> StateId | None:
        if state is None or state not in self.transitions or action not in self.transitions[state]:
            return None

        return self.transitions[state][action]


# ============================================================
# Frequency Prefix Tree
# ============================================================


class FrequencyPrefixTree(BaseStructure):
    def __init__(self, *args, depth: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_transition = None

        self.depth = depth

        self.state_probs = {}

        self.distance_mask = None  # helper matrix to compute probabilistic bisimilarity distance

    def initialize_distance_mask(self) -> None:
        """
        Initializes a square distance mask matrix with size equal to the number
        of actions plus one (for STOP at index 0). All distances are set to 1,
        except for the diagonal, which is set to 0 (distance to self).
        """
        num_actions = len(self.action_index) + 1  # Includes STOP at position 0
        self.distance_mask = np.ones((num_actions, num_actions))
        np.fill_diagonal(self.distance_mask, 0)  # Initially, self-distances are set to 1 except for identity

    def hash_state(self, state_id: StateId, k: int | None = None) -> int:
        """
        Recursively computes a hash for the state, factoring in its transitions
        and depth. If no transitions or depth limit is reached, hashes the state's
        label (labels are not implemented yet). Otherwise, hashes its child states
        and transitions.
        :param state_id: state id of state whose subtree to be hashed
        :param k: custom depth
        """
        if k is None:
            k = self.depth

        no_transitions = state_id not in self.transitions or len(self.transitions[state_id]) == 0
        insufficient_visits = self.state_info[state_id]["total_visits"] < self.min_total_visits
        depth_reached = k <= 0

        # Base case: no transitions, insufficient visits, or depth reached
        if no_transitions or insufficient_visits or depth_reached:
            return hash(0)

        child_hashes = []
        # Recursively hash child states and their transitions, decrementing depth
        for action, next_state in self.transitions[state_id].items():
            child_hashes.append((action, self.hash_state(next_state, k - 1)))

        # Sort child hashes by action name for consistency, then hash the current state
        return hash((0, tuple(sorted(child_hashes))))  # later use 0 as placeholder for label-specific information

    def hashing(self) -> None:
        """
        Updates the hash value for every state.
        """
        for state_id in self.state_info:
            self.state_info[state_id]["hash"] = self.hash_state(state_id)

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
        self.state_info[state_id]["action_frequency"] = {}
        self.state_info[state_id]["active_visits"] = 0
        self.state_info[state_id]["parent"] = None
        self.state_info[state_id]["hash"] = None
        self.state_info[state_id]["leaf"] = True
        self.state_info[state_id]["probs"] = [1.0] + [0.0 for _ in self.action_index]

        self.transitions[state_id] = {}

        return new_state

    def update(self, case_id: CaseId, action: ActionName) -> None:
        """
        Updates DFA tree structure of the process miner object by adding a new action to case
        """
        if self.initial_state is None:
            msg = "Initial state is not set. Cannot update the automaton."
            raise ValueError(msg)

        if action not in self.action_index:
            idx = len(self.action_index) + 1
            self.action_index[action] = idx
            self.index_action[idx] = action

            for state in self.state_info:
                self.state_info[state]["probs"].append(0)

            self.initialize_distance_mask()

        if case_id not in self.case_info:
            self.case_info[case_id] = {}
            self.case_info[case_id]["state"] = self.initial_state
            self.state_info[self.initial_state]["total_visits"] += 1
            self.state_info[self.initial_state]["active_visits"] += 1

        current_state = self.case_info[case_id]["state"]

        if current_state not in self.transitions:
            self.transitions[current_state] = {}

        if action in self.transitions[current_state]:
            next_state = self.transitions[current_state][action]
        else:
            self.state_info[current_state]["action_frequency"][action] = 0
            next_state = self.create_state().state_id
            self.transitions[current_state][action] = next_state
            self.state_info[next_state]["parent"] = current_state

        self.case_info[case_id]["state"] = next_state
        self.state_info[next_state]["total_visits"] += 1
        self.state_info[current_state]["action_frequency"][action] += 1
        self.state_info[current_state]["active_visits"] -= 1
        self.state_info[next_state]["active_visits"] += 1

        self.update_state_probs(current_state)
        self.update_state_probs(next_state)
        self.state_info[current_state]["leaf"] = False

        self.last_transition = (current_state, action, next_state)

    def compute_state_distances_bottom_up(self, discount: float = DISCOUNT) -> np.ndarray:
        """
        Compute distances between states in a bottom-up fashion using memoization and optimal transport.
        Args: discount (float): A discount factor used in computing distances (default is DISCOUNT).
        Returns: np.ndarray: A matrix of distances between states.
        """
        num_states = len(self.state_info)
        dist_matrix: np.ndarray = np.zeros((num_states, num_states))

        # Step 1: Initialize memoization dictionary for precomputed distances
        memo: dict[tuple[int, int], float] = {(state, state): 0 for state in range(num_states)}

        # Function to compute the distance between two states recursively with memoization
        def compute_distance(state1: StateId, state2: StateId) -> float:
            # If distance already computed, return the memoized result
            if (state1, state2) in memo:
                return memo[(state1, state2)]

            # Otherwise, compute the distance (based on your distance computation logic)
            # dist = self.compute_distance_between_states(state1, state2)
            probs1 = self.state_info[state1]["probs"]
            probs2 = self.state_info[state2]["probs"]

            num_actions = len(probs1)

            if self.distance_mask is None:
                msg = "Distance mask has not ben initialized yet."
                raise ValueError(msg)

            d_matrix = np.copy(self.distance_mask)  # Create a copy of the distance mask

            # Compute distances for actions between the successors
            for i in range(1, num_actions):
                successor1 = self.transitions.get(state1, {}).get(self.index_action.get(i))
                successor2 = self.transitions.get(state2, {}).get(self.index_action.get(i))

                if successor1 is not None and successor2 is not None:
                    # Recursive call with depth decreased by 1
                    d_matrix[i, i] = memo[(successor1, successor2)]
                else:
                    d_matrix[i, i] = 1  # If there's no successor, distance is 1

            # Solve optimal transport problem for the coupling using POT
            dist = self.solve_optimal_transport_pot(probs1, probs2, d_matrix, discount)

            # Memoize the result for future use
            memo[(state1, state2)] = dist
            memo[(state2, state1)] = dist  # Since the matrix is symmetric

            return dist

        leaf_states = [state_id for state_id, state_info in self.state_info.items() if state_info["leaf"]]

        # Step 2: Compute distances between leaf states first
        for leaf1 in leaf_states:
            for leaf2 in leaf_states:
                dist_matrix[leaf1][leaf2] = compute_distance(leaf1, leaf2)

        # Step 3: Move up the tree, computing distances for parent states
        visited = set(leaf_states)  # Track already visited states (start with leaf states)

        # Progress bar setup
        total_steps = num_states - len(leaf_states)
        with tqdm(total=total_steps, desc="Computing state distances") as pbar:
            # While not all states are visited, keep moving upwards through the tree
            while len(visited) < num_states:
                next_level_states = SortedSet()  # Use SortedSet for efficient insertion and lookup

                # Identify next-level states (parents of already visited states)
                for state in visited:
                    parent_state = self.state_info[state].get("parent")
                    if (
                        parent_state is not None
                        and parent_state not in visited
                        and all(target in visited for target in self.transitions.get(parent_state, {}).values())
                    ):
                        next_level_states.add(parent_state)

                # Compute distances between all pairs of next-level states
                for state1 in next_level_states:
                    for state2 in next_level_states:
                        dist_matrix[state1][state2] = compute_distance(state1, state2)
                        dist_matrix[state2][state1] = compute_distance(state1, state2)

                    # Also compute distances between the next-level state and all visited states
                    for visited_state in visited:
                        dist_matrix[state1][visited_state] = compute_distance(state1, visited_state)
                        dist_matrix[visited_state][state1] = dist_matrix[state1][visited_state]  # Symmetry

                # Add next-level states to the visited set
                visited.update(next_level_states)

                # Update the progress bar for each level
                pbar.update(len(next_level_states))

        return dist_matrix

    @staticmethod
    def solve_optimal_transport_pot(
        ps: np.ndarray | list[float], pt: np.ndarray | list[float], d_matrix: np.ndarray, discount: float = DISCOUNT
    ) -> float:
        """
        Solves the optimal transport problem using POT's emd function and computes the optimal transport distance.
        Returns: float: The computed optimal transport distance.
        """
        ps_array = np.array(ps) if isinstance(ps, list) else ps
        pt_array = np.array(pt) if isinstance(pt, list) else pt

        # Use POT's emd function to compute the optimal transport matrix
        c_matrix = ot.emd(ps_array, pt_array, d_matrix)  # Compute the optimal transport matrix

        result = c_matrix * (discount * d_matrix)
        result = np.array(result)

        # Compute the optimal transport distance
        return np.sum(result)

    def print_state_distances(self) -> None:
        dist_matrix = self.compute_state_distances_bottom_up()

        # Get the number of states
        num_states = len(self.state_info)

        # Print header (state indices)
        header = "      " + "  ".join(f"{i:<4}" for i in range(num_states))
        logger.info(header)

        # Print each row of the matrix with the row header (state index)
        for i, row in enumerate(dist_matrix):
            row_str = "  ".join(f"{dist:.2f}" for dist in row)
            msg = f"{i:<4} {row_str}"
            logger.info(msg)


# ============================================================
# N-Gram
# ============================================================


class NGram(BaseStructure):
    def __init__(self, *args, window_length: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window_length = window_length

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
        self.state_info[state_id]["action_frequency"] = {}
        self.state_info[state_id]["active_visits"] = 0
        self.state_info[state_id]["level"] = 0
        self.state_info[state_id]["probs"] = [1.0] + [0.0 for _ in self.action_index]

        self.transitions[state_id] = {}

        return new_state

    def follow_path(self, sequence: list[ActionName]) -> StateId:
        """
        Follows the given action_sequence starting from the root (initial state).
        If necessary, creates new states along the path. Does not modify state counts.

        :param sequence: A list of action names representing the path to follow.
        :return: The state_id of the final state reached after following the sequence.
        """
        current_state = self.initial_state

        for action in sequence:
            # Initialize transitions for the current state if not already present
            if current_state not in self.transitions:
                self.transitions[current_state] = {}

            # Follow existing transitions, or create a new state and transition if necessary
            if action in self.transitions[current_state]:
                current_state = self.transitions[current_state][action]
            else:
                next_state = self.create_state().state_id
                self.state_info[next_state]["level"] = self.state_info[current_state]["level"] + 1
                # Update the transition dictionary instead of overwriting
                self.transitions[current_state][action] = next_state
                current_state = next_state

        return current_state

    def update(self, case_id: CaseId, action: ActionName):
        """
        Updates DFA tree structure of the process miner object by adding a new action to case
        """
        if self.initial_state is None:
            msg = "Initial state is not set. Cannot update the automaton."
            raise ValueError(msg)

        if action not in self.action_index:
            idx = len(self.action_index) + 1
            self.action_index[action] = idx
            self.index_action[idx] = action

            for state in self.state_info:
                self.state_info[state]["probs"].append(0.0)

        if case_id not in self.case_info:
            self.case_info[case_id] = {}
            self.case_info[case_id]["state"] = self.initial_state
            self.case_info[case_id]["suffix"] = deque(maxlen=self.window_length)
            self.state_info[self.initial_state]["total_visits"] += 1
            self.state_info[self.initial_state]["active_visits"] += 1

        self.case_info[case_id]["suffix"].append(action)
        current_state = self.case_info[case_id]["state"]
        current_state_level = self.state_info[current_state]["level"]

        if current_state not in self.transitions:
            self.transitions[current_state] = {}

        if action in self.transitions[current_state]:
            next_state = self.transitions[current_state][action]
        else:
            if current_state_level < self.window_length:
                next_state = self.create_state().state_id
                self.state_info[next_state]["level"] = current_state_level + 1
            else:
                next_state = self.follow_path(self.case_info[case_id]["suffix"])
            self.transitions[current_state][action] = next_state

        self.case_info[case_id]["state"] = next_state
        self.state_info[next_state]["total_visits"] += 1
        if action in self.state_info[current_state]["action_frequency"]:
            self.state_info[current_state]["action_frequency"][action] += 1
        else:
            self.state_info[current_state]["action_frequency"][action] = 1
        self.state_info[current_state]["active_visits"] -= 1
        self.state_info[next_state]["active_visits"] += 1

        self.update_state_probs(current_state)
        self.update_state_probs(next_state)

        self.last_transition = (current_state, action, next_state)


# ============================================================
# Frequency Prefix Tree
# ============================================================


class Alergia(FrequencyPrefixTree):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
