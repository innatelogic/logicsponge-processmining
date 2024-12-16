import random
from collections import OrderedDict
from typing import Any

from logicsponge.processmining.config import update_config
from logicsponge.processmining.types import ActivityName, Event, ProbDistr, StateId


class State:
    def __init__(self, state_id: StateId = 0, name: str = "state") -> None:
        self.state_id = state_id
        self.name = name


class Automaton:
    name: str
    state_info: dict[StateId, Any]
    transitions: dict[StateId, dict[ActivityName, Any]]
    initial_state: StateId
    activities: OrderedDict[ActivityName, bool]

    def __init__(self, name: str = "Automaton", config: dict | None = None) -> None:
        self.name = name
        self.config = update_config(config)  # Merge provided config with defaults
        self.state_info = {}
        self.transitions = {}
        self.initial_state = 0  # dummy value, will be overwritten when initial state is set
        self.activities = OrderedDict()  # maps activities (excluding STOP) to dummy value True

    def add_activity(self, activity: ActivityName) -> None:
        if activity != self.config["stop_symbol"]:
            self.activities[activity] = True

    def add_activities(self, activities: list[ActivityName]) -> None:
        for activity in activities:
            self.add_activity(activity)

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

        self.state_info[state_id] = {"object": new_state}

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
        - activity: The symbol triggering the transition.
        - target: The state or states to which the transition leads (type varies by subclass).
        """
        raise NotImplementedError


class PDFA(Automaton):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def set_probs(self, state: StateId, probs: ProbDistr):
        if state not in self.state_info:
            self.state_info[state] = {}

        self.state_info[state]["probs"] = probs

    def simulate(self, n_runs: int) -> list[list[Event]]:
        dataset = []

        for i in range(n_runs):
            current_state = self.initial_state
            sequence = []

            while True:
                probs: ProbDistr = self.state_info[current_state]["probs"]

                if not probs:
                    break

                # Extract activities and their corresponding probabilities, sorted for consistency
                activities, probabilities = zip(*probs.items(), strict=True)

                activity_choice: ActivityName = random.choices(activities, weights=probabilities, k=1)[0]  # noqa: S311

                if activity_choice == self.config["stop_symbol"]:
                    break

                event = {"case_id": str(i), "activity": activity_choice}
                sequence.append(event)

                current_state = self.transitions[current_state][activity_choice]

            dataset.append(sequence)

        return dataset
