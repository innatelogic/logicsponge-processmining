"""Types for process mining."""

from datetime import datetime, timedelta
from typing import Any, TypedDict

# ============================================================
# Types
# ============================================================

CaseId = str | tuple[str, ...]


class StateId(int):
    """A class representing a state identifier."""

    in_recovery: bool = False

    def __new__(cls, value: int) -> "StateId":
        """Create a new StateId instance."""
        obj = int.__new__(cls, value)
        obj.in_recovery = False
        return obj

    @classmethod
    def with_recovery(cls, value: int) -> "StateId":
        """Create a new StateId instance with recovery mode enabled."""
        obj = int.__new__(cls, value)
        obj.in_recovery = True
        return obj

    def __repr__(self) -> str:
        """Return a string representation of the StateId."""
        return f"StateId({int(self)}, in_recovery={self.in_recovery})"


ComposedState = Any  # QUESTION: Is there a way to write this? ComposedState = StateId | tuple[ComposedState, ...]

ActivityName = str | tuple[str, ...]  # QUESTION: why tuple[str, ...]?

Prediction = dict[str, Any]

ProbDistr = dict[ActivityName, float]

ActivityDelays = dict[ActivityName, timedelta]


class Metrics(TypedDict):
    """
    A dictionary type for storing metrics related to process mining.

    Attributes:
        probs (ProbDistr): Probability distribution of activities.
        predicted_delays (ActivityDelays): Predicted delays for activities.

    """

    state_id: ComposedState
    probs: ProbDistr
    predicted_delays: ActivityDelays
    # likelihoods: dict[ActivityName, float]


def empty_metrics() -> Metrics:
    """Return an empty metrics object."""
    return Metrics(state_id=None, probs={}, predicted_delays={})  # , likelihoods={})


class Config(TypedDict, total=True):
    """Configuration for process mining."""

    # Process mining core configuration
    start_symbol: ActivityName
    stop_symbol: ActivityName
    empty_symbol: ActivityName
    discount_factor: float
    randomized: bool
    top_k: int
    include_stop: bool
    include_time: bool
    maxlen_delays: int


class RequiredEvent(TypedDict):
    """
    A dictionary type for storing required event attributes.

    Attributes:
        case_id (CaseId): Unique identifier for the case.
        activity (ActivityName): Name of the activity.
        timestamp (datetime | None): Timestamp of the event, can be None.

    """

    case_id: CaseId
    activity: ActivityName
    timestamp: datetime | None


class Event(RequiredEvent, total=False):
    """
    A dictionary type for storing event attributes.

    Attributes:
        attributes (dict[str, Any]): Additional attributes of the event.

    """

    attributes: dict[str, Any]
