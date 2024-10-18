from math import log
from typing import Any

# ============================================================
# Types
# ============================================================

CaseId = str | int | tuple[str | int, ...]

StateId = int
ActionName = str | int | tuple[str | int, ...]

Prediction = tuple[str, list[str], float]

# ComposedState = StateId | tuple[StateId]
ComposedState = Any


# ============================================================
# Constants
# ============================================================

START = "start"  # start action
STOP = "stop"  # stop action
RANDOMIZED = False
TOP_K = 3

DISCOUNT = 0.9


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
