from typing import Any

import numpy as np

np.random.seed(123)

# ============================================================
# Types
# ============================================================

CaseId = str | int | tuple[str | int, ...]

StateId = int
ComposedState = Any

ActionName = str | int | tuple[str | int, ...]

Prediction = dict[str, Any]

ProbDistr = dict[ActionName, float]


# ============================================================
# Constants
# ============================================================

START: ActionName = "__start__"  # start action
STOP: ActionName = "__stop__"  # stop action

DISCOUNT = 0.9

# prob_prediction config
CONFIG = {
    "randomized": False,
    "top_k": 3,
    "include_stop": True,
}


# ============================================================
# Prediction
# ==========================================================


def probs_prediction(probs: ProbDistr, config: dict[str, Any] | None = None) -> Prediction:
    """
    Returns the top-k actions based on their probabilities.
    If STOP has a probability of 1.0 and there are no other actions, return None.
    If STOP has a probability of 1.0 and there are other actions, give a uniform distribution to these other actions.
    If STOP is present but with a probability less than 1.0 and include_stop is False, remove it and normalize the rest.
    """
    # Set up the final configuration by merging user-provided config with defaults
    final_config = CONFIG.copy()
    if config is not None:
        final_config.update(config)

    # If there are no probabilities, return None
    if not probs:
        return {}

    # Create a copy of probs to avoid modifying the original dictionary
    probs_copy = probs.copy()

    # Handle the case where include_stop is False
    if not final_config["include_stop"] and STOP in probs_copy:  # STOP will always be a key
        stop_probability = probs_copy.get(STOP, 0.0)

        # If STOP has a probability of 1 and there are no other actions available, return None
        if stop_probability == 1.0 and len(probs_copy) == 1:
            return {}

        # If STOP has a probability of 1 but there are other actions, give a uniform distribution to the other actions
        if stop_probability == 1.0 and len(probs_copy) > 1:
            del probs_copy[STOP]  # Remove STOP from consideration

            # Verify STOP is indeed deleted
            if STOP in probs_copy:
                msg = "STOP was not successfully removed from probabilities."
                raise ValueError(msg)

            # Distribute the remaining probability uniformly among other actions
            num_actions = len(probs_copy)
            uniform_prob = 1.0 / num_actions
            probs_copy = {action: uniform_prob for action in probs_copy}

        # If STOP has less than 1.0 probability, remove it and normalize the rest
        elif stop_probability < 1.0:
            del probs_copy[STOP]

            # Verify STOP is indeed deleted
            if STOP in probs_copy:
                msg = "STOP was not successfully removed from probabilities."
                raise ValueError(msg)

            # Normalize the remaining probabilities so that they sum to 1
            total_prob = sum(probs_copy.values())
            if total_prob > 0:
                probs_copy = {action: prob / total_prob for action, prob in probs_copy.items()}

    # If there are no probabilities after filtering, return None
    if not probs_copy:
        return {}

    # Convert dictionary to a sorted list of items (actions and probabilities) for consistency
    sorted_probs = sorted(probs_copy.items(), key=lambda x: (-x[1], x[0]))

    # Extract actions and probabilities in a consistent way
    actions, probabilities = zip(*sorted_probs, strict=True)

    # Convert the probabilities to a numpy array
    probabilities_array = np.array(probabilities)

    # Get the indices of the top-k elements, sorted in descending order
    top_k_indices = np.argsort(probabilities_array)[-final_config["top_k"] :][::-1]

    # Use the indices to get the top-k actions
    top_k_actions = [actions[i] for i in top_k_indices]

    # Determine the predicted action
    if final_config["randomized"]:
        # Randomly choose an action based on the given probability distribution
        next_action_idx = np.random.choice(len(probabilities_array), p=probabilities_array / probabilities_array.sum())
        predicted_action = actions[next_action_idx]
    else:
        # Get the most probable action deterministically
        predicted_action = top_k_actions[0]

    # Get the highest probability corresponding to the predicted action
    highest_probability = float(probabilities_array[actions.index(predicted_action)])

    # Return the predicted action, top-k actions, and the probability of the predicted action
    return {
        "action": predicted_action,
        "top_k_actions": top_k_actions,
        "probability": highest_probability,
        "probs": probs_copy,
    }
