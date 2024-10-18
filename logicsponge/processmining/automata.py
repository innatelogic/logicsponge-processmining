import random

from logicsponge.processmining.globals import ActionName, StateId


class State:
    def __init__(self, state_id: StateId = 0, name: str = "state") -> None:
        self.state_id = state_id
        self.name = name


class Automaton:
    def __init__(self, name: str = "Automaton") -> None:
        self.name = name
        self.state_info = {}
        self.transitions = {}
        self.initial_state = None

        self.action_index = {}
        self.index_action = {}

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
        self.state_info[state_id]["probs"] = [1.0] + [0.0 for _ in self.action_index]

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


class DFA(Automaton):
    def add_transition(self, source: StateId, action: ActionName, target: StateId) -> None:
        if source not in self.transitions:
            self.transitions[source] = {}

        # In DFA, each symbol leads to exactly one state
        self.transitions[source][action] = target

    def __str__(self) -> str:
        result = [f"DFA with {len(self.state_info)} states."]
        return "\n".join(result)


class PDFA(DFA):
    def add_actions(self, actions: list) -> None:
        for action in actions:
            idx = len(self.action_index) + 1
            self.action_index[action] = idx
            self.index_action[idx] = action

    def set_probs(self, state, probs):
        # Check if the length of probs is equal to len(self.action_index) + 1
        if len(probs) == len(self.action_index) + 1:
            # Ensure state exists in state_info
            if state not in self.state_info:
                self.state_info[state] = {}

            # Set the probabilities for the state
            self.state_info[state]["probs"] = probs
        else:
            # Raise an error or handle the case where the lengths don't match
            msg = "Length of probs and action_index do not match."
            raise ValueError(msg)

    def simulate(self, n_runs: int) -> list[list[ActionName]]:
        dataset = []

        for _ in range(n_runs):
            current_state = self.initial_state
            sequence = []

            while True:
                # Get the probabilities for the current state
                probs = self.state_info[current_state]["probs"]

                # Choose an action or stop based on the probabilities
                action_choice = random.choices(range(len(probs)), weights=probs, k=1)[0]  # noqa: S311

                # If the chosen action is 0, we stop (this is the stopping action)
                if action_choice == 0:
                    break

                # Map the index back to the action using self.index_action
                action: ActionName = self.index_action[action_choice]

                # Append the action to the sequence
                sequence.append(action)

                # Transition to the next state based on the chosen action
                current_state = self.transitions[current_state][action]

            # Add the generated sequence to the dataset
            dataset.append(sequence)

        return dataset
