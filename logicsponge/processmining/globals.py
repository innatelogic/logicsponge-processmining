from math import log
from typing import Any

import numpy as np

np.random.seed(123)

# ============================================================
# Types
# ============================================================

CaseId = str | int | tuple[str | int, ...]

StateId = int
ActionName = str | int | tuple[str | int, ...]

Prediction = tuple[ActionName, list[ActionName], float]

Probs = dict[ActionName, float]

# ComposedState = StateId | tuple[StateId]
ComposedState = Any


# ============================================================
# Constants
# ============================================================

START: ActionName = "start"  # start action
STOP: ActionName = "stop"  # stop action

DISCOUNT = 0.9


# ============================================================
# Prediction
# ==========================================================


def probs_prediction(probs: Probs) -> Prediction | None:
    """
    Returns the top-k actions based on their probabilities.
    """
    config = {
        "randomized": False,
        "top_k": 3,
        "include_stop": True,
    }

    # probs = dict(sorted(probs.items()))
    # If there are no probabilities, return an empty list
    if not probs:
        return None

    # Convert dictionary to a list of items (actions and probabilities)
    actions, probabilities = zip(*probs.items(), strict=True)

    # Convert the probabilities to a numpy array
    probabilities_array = np.array(probabilities)

    # Get the indices of the top-k elements, sorted in descending order
    top_k_indices = np.argsort(probabilities_array)[-config["top_k"] :][::-1]

    # Use the indices to get the top-k actions
    top_k_actions = [actions[i] for i in top_k_indices]

    return top_k_actions[0], [], 0.0


# def probs_prediction(probs: dict[str, float], config: dict | None = None) -> tuple | None:
#     """
#     Transforms probability dictionary into a prediction consisting of:
#     - The most probable action.
#     - Top-k predicted actions.
#     - Probability of the most probable action.
#     """
#     if not probs:
#         return None
#
#     if config is None:
#         config = CONFIG
#
#     include_stop = config.get("include_stop", True)
#
#     # Handle the case where include_stop is False by filtering out STOP from probs
#     if not include_stop:
#         probs = {action: prob for action, prob in probs.items() if action != STOP}
#
#     # If the resulting dictionary is empty, return None
#     if not probs:
#         return None
#
#     # Get the top-k actions with the highest probabilities
#     top_k_items = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:config["top_k"]]
#
#     # Extract the top-k actions and their probabilities
#     top_k_actions = [action for action, prob in top_k_items]
#
#     # Find the action with the highest probability
#     if config["randomized"]:
#         # Create a list of probabilities in order of the keys in the dict
#         actions, probabilities = zip(*probs.items(), strict=True)
#         next_action_idx = np.random.choice(len(probabilities), p=np.array(probabilities) / sum(probabilities))
#         predicted_action = actions[next_action_idx]
#     else:
#         # Get the most probable action deterministically
#         predicted_action = top_k_actions[0]
#
#     # If the most probable action has a probability of 0, return None
#     highest_probability = probs[predicted_action]
#     if highest_probability == 0.0:
#         return None
#
#     # Return the most probable action, the top-k actions, and the probability of the predicted action
#     return predicted_action, top_k_actions, highest_probability


# ============================================================
# Constants
# ============================================================


def update_correct_count(stats, actual, predicted=None, **kwargs):  # noqa: ARG001
    if actual == predicted:
        stats["correct_count"] += 1


def update_wrong_count(stats, actual, predicted=None, **kwargs):  # noqa: ARG001
    if actual != predicted:
        stats["wrong_count"] += 1


def update_unparseable_count(stats, **kwargs):
    pass


def update_within_top_k_count(stats, actual, top_k=None, **kwargs):  # noqa: ARG001
    if actual in top_k:
        stats["within_top_k_count"] += 1


def update_wrong_top_k_count(stats, actual, top_k=None, **kwargs):  # noqa: ARG001
    if actual not in top_k:
        stats["wrong_top_k_count"] += 1


def update_log_loss(stats, predicted_prob=None, **kwargs):  # noqa: ARG001
    if predicted_prob is not None and predicted_prob > 0:
        stats["log_loss"] += -log(predicted_prob)
    else:
        stats["log_loss"] += float("inf")


# Define the STATS dictionary with references to the update functions
STATS = {
    "correct_count": {"init": 0, "name": "Correct Predictions", "update": update_correct_count},
    "wrong_count": {"init": 0, "name": "Wrong Predictions", "update": update_wrong_count},
    "unparseable_count": {"init": 0, "name": "Unparseable Positions", "update": update_unparseable_count},
    "within_top_k_count": {"init": 0, "name": "Within Top k Predictions", "update": update_within_top_k_count},
    "wrong_top_k_count": {"init": 0, "name": "Wrong Top k Predictions", "update": update_wrong_top_k_count},
    "log_loss": {"init": 0, "name": "Logarithmic Loss", "update": update_log_loss},
}
