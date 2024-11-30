import random
from collections import OrderedDict
from typing import Any

from logicsponge.processmining.globals import STOP, ActionName, ProbDistr, StateId


class State:
    def __init__(self, state_id: StateId = 0, name: str = "state") -> None:
        self.state_id = state_id
        self.name = name


class Automaton:
    name: str
    state_info: dict[StateId, Any]
    transitions: dict[StateId, dict[ActionName, Any]]
    initial_state: StateId | None
    actions: OrderedDict[ActionName, bool]

    def __init__(self, name: str = "Automaton") -> None:
        self.name = name
        self.state_info = {}
        self.transitions = {}
        self.initial_state = None
        self.actions = OrderedDict()  # maps actions (excluding STOP) to dummy value True

    def add_action(self, action: ActionName) -> None:
        if action != STOP:
            self.actions[action] = True

    def add_actions(self, actions: list[ActionName]) -> None:
        for action in actions:
            self.add_action(action)

    def set_initial_state(self, state_id: StateId) -> None:
        self.initial_state = state_id

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

        self.transitions[state_id] = {}

        return new_state

    def create_states(self, n_states: int) -> None:
        for _ in range(n_states):
            self.create_state()

    def add_transition(self, *args, **kwargs) -> None:
        """
        Abstract method to add a transition between states.

        Parameters:
        - source: The state from which the transition originates.
        - action: The symbol triggering the transition.
        - target: The state or states to which the transition leads (type varies by subclass).
        """
        raise NotImplementedError


class PDFA(Automaton):
    def set_probs(self, state, probs):
        if state not in self.state_info:
            self.state_info[state] = {}

        self.state_info[state]["probs"] = probs

    def simulate(self, n_runs: int) -> list[list[ActionName]]:
        dataset = []

        for _ in range(n_runs):
            current_state = self.initial_state
            sequence = []

            while True:
                # Get the probabilities for the current state as a dictionary
                probs: ProbDistr = self.state_info[current_state]["probs"]

                # If there are no probabilities, stop the simulation
                if not probs:
                    break

                # Extract actions and their corresponding probabilities, sorted for consistency
                actions, probabilities = zip(*probs.items(), strict=True)

                # Choose an action based on the probabilities
                action_choice: ActionName = random.choices(actions, weights=probabilities, k=1)[0]  # noqa: S311

                # If the chosen action is STOP, we stop (this is the stopping action)
                if action_choice == STOP:
                    break

                # Append the action to the sequence
                sequence.append(action_choice)

                # Transition to the next state based on the chosen action
                current_state = self.transitions[current_state][action_choice]

            # Add the generated sequence to the dataset
            dataset.append(sequence)

        return dataset
