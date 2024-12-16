from datetime import datetime
from typing import Any, TypedDict

# ============================================================
# Types
# ============================================================

CaseId = str | int | tuple[str | int, ...]

StateId = int
ComposedState = Any

ActivityName = str | int | tuple[str | int, ...]

Prediction = dict[str, Any]

ProbDistr = dict[ActivityName, float]


class Config(TypedDict, total=True):
    # Process mining core configuration
    start_symbol: ActivityName
    stop_symbol: ActivityName
    discount_factor: float
    randomized: bool
    top_k: int
    include_stop: bool


class RequiredEvent(TypedDict):
    case_id: CaseId
    activity: ActivityName


class Event(RequiredEvent, total=False):
    timestamp: datetime
    attributes: dict[str, Any]
