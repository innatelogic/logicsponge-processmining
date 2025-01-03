from datetime import datetime, timedelta
from typing import Any, TypedDict

# ============================================================
# Types
# ============================================================

CaseId = str | tuple[str, ...]

StateId = int
ComposedState = Any

ActivityName = str | tuple[str, ...]

Prediction = dict[str, Any]

ProbDistr = dict[ActivityName, float]

ActivityDelays = dict[ActivityName, timedelta]


class Metrics(TypedDict):
    probs: ProbDistr
    predicted_delays: ActivityDelays


def empty_metrics():
    return Metrics(probs={}, predicted_delays={})


class Config(TypedDict, total=True):
    # Process mining core configuration
    start_symbol: ActivityName
    stop_symbol: ActivityName
    discount_factor: float
    randomized: bool
    top_k: int
    include_stop: bool
    include_time: bool
    maxlen_delays: int


class RequiredEvent(TypedDict):
    case_id: CaseId
    activity: ActivityName
    timestamp: datetime | None


class Event(RequiredEvent, total=False):
    attributes: dict[str, Any]
