from typing import Any

# ============================================================
# Types
# ============================================================

CaseId = str | int | tuple[str | int, ...]

StateId = int
ComposedState = Any

ActionName = str | int | tuple[str | int, ...]

Prediction = dict[str, Any]

ProbDistr = dict[ActionName, float]
