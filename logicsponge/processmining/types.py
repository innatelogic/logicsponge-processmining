"""Types for process mining."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, Self, TypedDict

# ============================================================
# Types
# ============================================================

CaseId = str | tuple[str, ...]


class StateId(int):
    """A class representing a state identifier."""

    in_recovery: bool = False

    def __new__(cls, value: int) -> Self:
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


@dataclass(frozen=True, slots=True)
class OrderedModelState:
    """States for the previous, current, and next models in an ordered strategy."""

    previous: ComposedState | None
    current: ComposedState
    next: ComposedState | None


ActivityName = str | tuple[str, ...]  # QUESTION: why tuple[str, ...]?

Prediction = dict[str, Any]

ProbDistr = dict[ActivityName, float]

ActivityDelays = dict[ActivityName, timedelta]

NumericEstimatorName = Literal["mean", "median", "mode", "quantile", "trimmed_mean"]
NumericErrorName = Literal["smape", "mae", "mape", "mse", "rmse"]
NumericAggregator = Callable[[Mapping[int, int]], float]
NumericErrorFunction = Callable[[float, float], float]


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


class RequiredNumericEvent(TypedDict):
    """Required fields for an integer-valued sequence event."""

    case_id: CaseId
    value: int
    timestamp: datetime | None


class NumericEvent(RequiredNumericEvent, total=False):
    """Canonical event for integer-valued sequence prediction."""

    attributes: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NumericMetrics:
    """Prediction and learned next-value distribution for a numeric context."""

    state_id: tuple[int, ...]
    prediction: float | None
    distribution: dict[int, float]
    estimator: str


@dataclass(frozen=True, slots=True)
class NumericEvaluationResult:
    """Aggregate and per-observation results from numeric evaluation."""

    metric: str
    score: float | None
    total_observations: int
    evaluated_observations: int
    missing_predictions: int
    predictions: list[float | None]
    actuals: list[int]
    errors: list[float | None]
