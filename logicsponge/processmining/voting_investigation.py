"""Standalone investigation of the accuracy gap between voting ensembles and an oracle."""

# ruff: noqa: ANN401, D102, D107, PLC0415, PLR0913, TC003

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from heapq import nsmallest
from pathlib import Path
from typing import Any

from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.data_utils import (
    add_stop_to_sequences,
    interleave_sequences,
    split_sequence_data,
    transform_to_seqs,
)
from logicsponge.processmining.miners import AdaptiveVoting, BasicMiner, SoftVoting, StreamingMiner
from logicsponge.processmining.types import ComposedState, Event, Metrics
from logicsponge.processmining.utils import metrics_prediction, resolve_dataset_from_args

logger = logging.getLogger(__name__)
VISIT_FEW_MAX = 2
VISIT_MANY_MAX = 9
TRIMMED_POOL_MIN_MODELS = 3
HEDGE_RESCALE_MAX = 1e6
HEDGE_RESCALE_MIN = 1e-6
MIN_INTEGRATION_SEQUENCES = 3
MIN_SPECIALIST_OVERRIDES = 5
MIN_BRANCHING_SEQUENCES = 4
MIN_NGRAM_WINDOW = 2
TRANSIENT_BOOST_TRIGGER_LABELS = {
    "generalist was correct": "after generalist-correct error",
    "generalist was wrong": "after generalist-wrong error",
    "all soft errors": "after any soft error",
}
DEFAULT_RESULTS_ROOT = Path("results/voting-investigation")
DEFAULT_BENCHMARK_DATASETS = (
    "Sepsis_Cases",
    "Helpdesk",
    "BPI_Challenge_2012",
    "BPI_Challenge_2013",
    "BPI_Challenge_2014",
    "BPI_Challenge_2017",
    "BPI_Challenge_2018",
    "BPI_Challenge_2019",
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Description and constructor for one constituent model."""

    name: str
    complexity: int
    factory: Callable[[], StreamingMiner]
    model_type: str = "generic"
    window_size: int | None = None


def default_model_specs(windows: Iterable[int] = (2, 3, 4)) -> list[ModelSpec]:
    """Return the default Bag/N-gram model family used by voting benchmarks."""
    window_lengths = tuple(windows)
    if any(window < MIN_NGRAM_WINDOW for window in window_lengths):
        msg = "N-gram windows must be at least 2, preserving Bag and making streak / (window - 1) well-defined."
        raise ValueError(msg)
    specs = [
        ModelSpec("bag", 0, lambda: BasicMiner(algorithm=Bag()), model_type="bag"),
    ]
    specs.extend(
        ModelSpec(
            f"ngram_{window}",
            window,
            lambda window=window: BasicMiner(algorithm=NGram(window_length=window)),
            model_type="ngram",
            window_size=window,
        )
        for window in window_lengths
    )
    return specs


def _display(value: Any) -> str:
    """Return a stable display representation for activities and states."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return repr(value)


def _normalized_distribution(probs: dict[Any, float]) -> dict[str, float]:
    """Convert a probability mapping to stable display keys and normalize it."""
    combined: defaultdict[str, float] = defaultdict(float)
    for activity, probability in probs.items():
        if probability > 0:
            combined[_display(activity)] += float(probability)
    total = sum(combined.values())
    return {activity: probability / total for activity, probability in combined.items()} if total else {}


def _distribution_statistics(distribution: dict[str, float]) -> dict[str, float | int]:
    """Return entropy, top margin, support, and top-three probability mass."""
    probabilities = sorted(distribution.values(), reverse=True)
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0)
    normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    margin = (
        probabilities[0] - probabilities[1] if len(probabilities) > 1 else (probabilities[0] if probabilities else 0.0)
    )
    return {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "margin": margin,
        "support": len(probabilities),
        "top3_mass": sum(probabilities[:3]),
    }


def _jensen_shannon_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    """Return Jensen-Shannon divergence between two normalized distributions."""
    activities = set(left) | set(right)
    mixture = {activity: (left.get(activity, 0.0) + right.get(activity, 0.0)) / 2 for activity in activities}

    def divergence(source: dict[str, float]) -> float:
        return sum(
            probability * math.log(probability / mixture[activity])
            for activity, probability in source.items()
            if probability > 0 and mixture[activity] > 0
        )

    return (divergence(left) + divergence(right)) / 2


def _value_bin(value: float, edges: tuple[float, ...]) -> str:
    """Assign a numeric value to a stable categorical interval."""
    for index, edge in enumerate(edges):
        if value <= edge:
            return str(index)
    return str(len(edges))


def _accuracy(rows: list[dict[str, Any]], prediction_key: str) -> float:
    """Compute accuracy for one prediction field."""
    return sum(row[prediction_key] == row["actual"] for row in rows) / len(rows) if rows else 0.0


def _best_model(rows: list[dict[str, Any]], *, minimum_support: int = 1) -> str:
    """Return the most accurate constituent model on labeled rows."""
    if not rows:
        return ""
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        for model in row["models"]:
            totals[model["name"]] += 1
            correct[model["name"]] += int(model["correct"])
    eligible = [name for name, total in totals.items() if total >= minimum_support]
    if not eligible:
        eligible = list(totals)
    return max(eligible, key=lambda name: (correct[name] / totals[name], totals[name], -list(totals).index(name)))


class DecisionRule:
    """Base interface for model-selection hypotheses."""

    name = "decision_rule"
    family = "base"
    description = "Selects a prediction source from calibration evidence."
    interpretation = "Its choices indicate which model representation is locally most trustworthy."
    selection_policy = "Derive the prediction source from event metrics or calibration; never name a fixed model."

    def fit(self, rows: list[dict[str, Any]]) -> None:
        """Fit rule parameters from labeled calibration events."""

    def observe(self, row: dict[str, Any], selected_model: str) -> None:
        """
        Consume feedback after an event has been scored.

        Rules may override this for delayed-feedback adaptation. ``evaluate_hypotheses``
        calls it only after recording the current prediction, so an adaptive rule
        cannot inspect the current label before selecting the current model.
        """

    def select(self, row: dict[str, Any]) -> str:
        """Select a constituent model name for an unlabeled event."""
        raise NotImplementedError

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        """Return the selected source label and its activity prediction."""
        model_name = self.select(row)
        return model_name, _model_prediction(row, model_name)


class HighestConfidenceRule(DecisionRule):
    """Select the model with the highest top-prediction probability."""

    name = "highest confidence"
    family = "confidence"
    description = "Selects the constituent with the greatest current top-prediction probability."
    interpretation = "Tests whether sharper model distributions are reliable enough to route predictions directly."
    selection_policy = "Choose the current model with the greatest top-prediction probability."

    def select(self, row: dict[str, Any]) -> str:
        return max(row["models"], key=lambda model: (model["confidence"], -model["index"]))["name"]


class GlobalAccuracyRule(DecisionRule):
    """Always select the best model on calibration data."""

    name = "best calibration accuracy"
    family = "accuracy"
    description = "Uses the constituent with the greatest accuracy on the complete calibration split."
    interpretation = "A reference for whether conditional routing adds value beyond one globally dominant model."
    selection_policy = "Choose the model with the greatest global calibration accuracy."

    def __init__(self) -> None:
        self.model_name = ""

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.model_name = _best_model(rows)

    def select(self, row: dict[str, Any]) -> str:  # noqa: ARG002
        return self.model_name


class GroupedAccuracyRule(DecisionRule):
    """Select the best calibrated model for a feature-defined event group."""

    selection_policy = "Choose the most accurate calibrated model in the current feature group."
    description = "Selects the most accurate calibrated constituent inside a supported feature-defined group."
    interpretation = "Shows whether model competence changes systematically across coarse process regimes."

    def __init__(
        self,
        *,
        name: str,
        family: str,
        feature: Callable[[dict[str, Any]], str],
        minimum_support: int,
    ) -> None:
        self.name = name
        self.family = family
        self.feature = feature
        self.minimum_support = minimum_support
        self.default_model = ""
        self.model_by_group: dict[str, str] = {}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.default_model = _best_model(rows)
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[self.feature(row)].append(row)
        self.model_by_group = {
            group: _best_model(group_rows)
            for group, group_rows in grouped.items()
            if len(group_rows) >= self.minimum_support
        }

    def select(self, row: dict[str, Any]) -> str:
        return self.model_by_group.get(self.feature(row), self.default_model)


class StateAccuracyRule(DecisionRule):
    """Choose the model with the best calibrated accuracy in its current state."""

    family = "state accuracy"
    description = "Compares each model's calibrated accuracy in its own current learned process state."
    interpretation = "Tests whether recurrent miner states expose stable local subprocess competence."
    selection_policy = "Choose the model with the greatest supported accuracy in its current learned state."

    def __init__(self, minimum_support: int) -> None:
        self.minimum_support = minimum_support
        self.name = f"per-state accuracy (support {minimum_support})"
        self.default_model = ""
        self.state_scores: dict[tuple[str, str], tuple[int, int]] = {}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.default_model = _best_model(rows)
        scores: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            for model in row["models"]:
                key = (model["name"], model["state"])
                scores[key][0] += int(model["correct"])
                scores[key][1] += 1
        self.state_scores = {key: (value[0], value[1]) for key, value in scores.items()}

    def select(self, row: dict[str, Any]) -> str:
        candidates: list[tuple[float, int, str]] = []
        for model in row["models"]:
            correct, total = self.state_scores.get((model["name"], model["state"]), (0, 0))
            if total >= self.minimum_support:
                candidates.append((correct / total, total, model["name"]))
        return max(candidates)[2] if candidates else self.default_model


class AgreementRule(DecisionRule):
    """Trust consensus above a threshold, otherwise use a calibrated fallback."""

    family = "agreement"
    description = "Uses the highest-confidence consensus member above a threshold and a metric-based fallback below it."
    interpretation = "Measures when cross-model corroboration is more trustworthy than a global or confidence fallback."
    selection_policy = (
        "Choose the highest-confidence model in the consensus; otherwise use the configured metric-based fallback."
    )

    def __init__(self, threshold: int, fallback: str) -> None:
        self.threshold = threshold
        self.fallback = fallback
        self.name = f"agreement ≥ {threshold}, else {fallback}"
        self.default_model = ""

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.default_model = _best_model(rows)

    def select(self, row: dict[str, Any]) -> str:
        if row["agreement_count"] >= self.threshold:
            matching = [model for model in row["models"] if model["prediction"] == row["consensus_prediction"]]
            if matching:
                return max(matching, key=lambda model: (model["confidence"], -model["index"]))["name"]
        if self.fallback == "confidence":
            return max(row["models"], key=lambda model: (model["confidence"], -model["index"]))["name"]
        return self.default_model


def _confidence_bin(value: float, bins: int = 5) -> str:
    """Discretize a confidence value without assuming cross-model calibration."""
    return str(min(bins - 1, max(0, int(value * bins))))


def _visit_bin(visits: int) -> str:
    """Use robust visit-count bands instead of treating raw counts as numeric."""
    if visits <= 0:
        return "0"
    if visits <= VISIT_FEW_MAX:
        return "1-2"
    if visits <= VISIT_MANY_MAX:
        return "3-9"
    return "10+"


def _relative_rank(models: list[dict[str, Any]], model: dict[str, Any], field: str, *, reverse: bool = True) -> str:
    """Return a label-invariant ordinal rank for one model diagnostic."""
    ordered = sorted(
        models,
        key=lambda candidate: (float(candidate[field]), -int(candidate["index"])),
        reverse=reverse,
    )
    return str(next(index for index, candidate in enumerate(ordered, start=1) if candidate is model))


def _prediction_partition(row: dict[str, Any]) -> tuple[str, ...]:
    """
    Canonicalize model agreement without retaining any activity identity.

    For example, predictions ``A, B, B, C`` become ``0, 1, 1, 2``. This lets
    selectors learn recurring disagreement topology even if every activity in
    another dataset has a different name.
    """
    groups: dict[str, int] = {}
    partition = []
    for model in row["models"]:
        prediction = model["prediction"] or "<empty>"
        groups.setdefault(prediction, len(groups))
        partition.append(str(groups[prediction]))
    return tuple(partition)


def _soft_rank_of_model(row: dict[str, Any], model: dict[str, Any], maximum_rank: int = 4) -> str:
    """Describe a model prediction by its rank in the soft distribution, not its label."""
    prediction = model["prediction"]
    for rank, item in enumerate(row.get("soft_ranked_predictions", []), start=1):
        if item["activity"] == prediction:
            return str(min(rank, maximum_rank))
    return "missing"


def _sequence_stage(row: dict[str, Any]) -> str:
    position = float(row["relative_position"])
    return "early" if position <= 1 / 3 else "middle" if position <= 2 / 3 else "late"


def _structural_regime_contexts(row: dict[str, Any]) -> list[tuple[str, ...]]:
    """
    Return activity-label-invariant descriptions of the current process regime.

    The contexts describe how the constituent miners *read* the prefix: which
    ones agree, how much evidence their states contain, whether longer n-grams
    are mature, and how concentrated their probability distributions are.
    """
    models = row["models"]
    topology = _prediction_partition(row)
    visits = tuple(_visit_bin(int(model["state_visits"])) for model in models)
    maturity = tuple(str(int(row["position"]) >= int(model["complexity"])) for model in models)
    confidence_order = tuple(
        model["name"] for model in sorted(models, key=lambda item: (-float(item["confidence"]), item["index"]))
    )
    entropy_order = tuple(
        model["name"] for model in sorted(models, key=lambda item: (float(item["normalized_entropy"]), item["index"]))
    )
    margin = _value_bin(float(row.get("soft_margin", 0.0)), (0.01, 0.03, 0.07, 0.15))
    entropy = _value_bin(float(row.get("soft_normalized_entropy", 0.0)), (0.25, 0.5, 0.7, 0.85))
    spread = _value_bin(float(row.get("model_confidence_spread", 0.0)), (0.1, 0.25, 0.5, 0.75))
    previous = "unknown" if row.get("previous_soft_correct") is None else str(row["previous_soft_correct"])
    return [
        ("joint-state-evidence", *visits, *topology, margin),
        ("order-maturity", _sequence_stage(row), *maturity, *topology),
        ("distribution-order", *confidence_order, *entropy_order, margin, entropy),
        (
            "uncertainty-topology",
            str(row["agreement_count"]),
            str(row["distinct_prediction_count"]),
            str(row.get("empty_prediction_count", 0)),
            margin,
            entropy,
            spread,
        ),
        ("feedback-regime", previous, _sequence_stage(row), str(row["agreement_count"]), margin),
        ("topology", *topology, str(row["agreement_count"])),
        ("global",),
    ]


def _smoothed_rate(correct: float, total: float, prior: float, prior_weight: float = 4.0) -> float:
    """Return a shrinkage estimate that protects sparse contexts."""
    return (correct + prior * prior_weight) / (total + prior_weight)


class HierarchicalReliabilityRule(DecisionRule):
    """
    Select a model using fine-to-coarse reliability contexts.

    Contexts back off from exact learned state and joint evidence topology to
    relative evidence roles, agreement, and finally global model reliability.
    Every score is shrunk toward the model's global calibration accuracy.
    """

    family = "hierarchical"
    description = (
        "Selects the locally reliable model through progressively broader state-evidence, n-gram maturity, "
        "agreement-topology, and uncertainty contexts."
    )
    interpretation = (
        "Frequent selection of longer-order models indicates stable local routing; backoff to general models "
        "indicates sparse states, branching, or concept drift."
    )
    selection_policy = "Choose the model with the greatest smoothed local reliability estimate."

    def __init__(self, minimum_support: int = 3, prior_weight: float = 4.0) -> None:
        self.minimum_support = minimum_support
        self.prior_weight = prior_weight
        self.name = f"hierarchical reliability (support {minimum_support})"
        self.global_scores: dict[str, tuple[int, int]] = {}
        self.context_scores: list[dict[tuple[str, ...], tuple[int, int]]] = []
        self.default_model = ""

    def _contexts(self, row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        model_name = model["name"]
        support = _visit_bin(int(model["state_visits"]))
        support_rank = _relative_rank(row["models"], model, "state_visits")
        confidence_rank = _relative_rank(row["models"], model, "confidence")
        entropy_rank = _relative_rank(row["models"], model, "normalized_entropy", reverse=False)
        soft_rank = _soft_rank_of_model(row, model)
        topology = _prediction_partition(row)
        mature = str(int(row["position"]) >= int(model["complexity"]))
        return [
            (
                model_name,
                model["state"],
                support,
                soft_rank,
                *topology,
            ),
            (model_name, support, support_rank, confidence_rank, entropy_rank, soft_rank, mature, *topology),
            (model_name, support, support_rank, confidence_rank, soft_rank, mature),
            (model_name, support, soft_rank, str(row["agreement_count"])),
            (model_name, support, mature),
            (model_name,),
        ]

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.global_scores = {}
        context_count = len(self._contexts(rows[0], rows[0]["models"][0])) if rows else 0
        self.context_scores = [defaultdict(lambda: [0, 0]) for _ in range(context_count)]
        for row in rows:
            for model in row["models"]:
                name = model["name"]
                correct, total = self.global_scores.get(name, (0, 0))
                self.global_scores[name] = (correct + int(model["correct"]), total + 1)
                for level, context in enumerate(self._contexts(row, model)):
                    old_correct, old_total = self.context_scores[level].get(context, (0, 0))
                    self.context_scores[level][context] = (old_correct + int(model["correct"]), old_total + 1)
        self.default_model = _best_model(rows)

    def _score(self, row: dict[str, Any], model: dict[str, Any]) -> tuple[float, int, int]:
        global_correct, global_total = self.global_scores.get(model["name"], (0, 0))
        prior = global_correct / global_total if global_total else 0.0
        for level, context in enumerate(self._contexts(row, model)):
            if level >= len(self.context_scores):
                break
            correct, total = self.context_scores[level].get(context, (0, 0))
            if total >= self.minimum_support:
                return _smoothed_rate(correct, total, prior, self.prior_weight), total, -level
        return (
            _smoothed_rate(global_correct, global_total, prior, self.prior_weight),
            global_total,
            -len(self.context_scores),
        )

    def _select_from(self, row: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return self.default_model
        return max(
            candidates,
            key=lambda model: (*self._score(row, model), -model["index"]),
        )["name"]

    def select(self, row: dict[str, Any]) -> str:
        return self._select_from(row, row["models"])


class ConsensusHierarchicalRule(HierarchicalReliabilityRule):
    """Use hierarchical reliability inside a consensus-aware branch."""

    family = "hierarchical agreement"

    def __init__(self, threshold: int = 3, minimum_support: int = 3) -> None:
        super().__init__(minimum_support=minimum_support)
        self.threshold = threshold
        self.name = f"consensus hierarchy ≥ {threshold} (support {minimum_support})"

    def select(self, row: dict[str, Any]) -> str:
        if row["agreement_count"] >= self.threshold:
            consensus_models = [model for model in row["models"] if model["prediction"] == row["consensus_prediction"]]
            if consensus_models:
                return self._select_from(row, consensus_models)
        return super().select(row)


class ConfidenceStateReliabilityRule(HierarchicalReliabilityRule):
    """Calibrate reliability by confidence, state visits, and agreement."""

    family = "confidence-state"
    description = "Calibrates model reliability by confidence band, state-support band, and agreement count."
    interpretation = "Reveals whether confidence becomes trustworthy only after a process state has repeated enough."

    def __init__(self, minimum_support: int = 5) -> None:
        super().__init__(minimum_support=minimum_support)
        self.name = f"confidence/state reliability (support {minimum_support})"

    def _contexts(self, row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        name = model["name"]
        confidence = _confidence_bin(model["confidence"])
        visits = _visit_bin(model["state_visits"])
        return [
            (name, confidence, visits, str(row["agreement_count"])),
            (name, confidence, visits),
            (name, confidence),
            (name,),
        ]


class DisagreementProfileRule(HierarchicalReliabilityRule):
    """Learn which model wins for each observable disagreement profile."""

    family = "disagreement profile"
    description = "Learns model reliability from canonical agreement topology, support, soft rank, and divergence role."
    interpretation = "Identifies recurring branches where a particular abstraction is a reliable contrarian."

    def __init__(self, minimum_support: int = 4) -> None:
        super().__init__(minimum_support=minimum_support)
        self.name = f"disagreement profile reliability (support {minimum_support})"

    def _contexts(self, row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        profile = _prediction_partition(row)
        name = model["name"]
        return [
            (
                name,
                *profile,
                _visit_bin(int(model["state_visits"])),
                _soft_rank_of_model(row, model),
                _relative_rank(row["models"], model, "soft_divergence", reverse=False),
            ),
            (name, *profile, _soft_rank_of_model(row, model)),
            (name, *profile),
            (name, str(row["agreement_count"]), _visit_bin(int(model["state_visits"]))),
            (name,),
        ]


class DelayedFeedbackAdaptiveRule(HierarchicalReliabilityRule):
    """
    Adapt model reliability from labels observed after prior predictions.

    This is an online selector: calibration provides priors, then each scored
    test event updates decayed global/state/context reliability for the next
    event. It uses delayed feedback only and never the current event's label.
    """

    family = "adaptive"

    def __init__(self, minimum_support: int = 2, decay: float = 0.94) -> None:
        super().__init__(minimum_support=minimum_support)
        self.decay = decay
        self.name = f"delayed-feedback adaptive (decay {decay:g})"

    def fit(self, rows: list[dict[str, Any]]) -> None:
        super().fit(rows)
        # Give calibration evidence a bounded prior so recent feedback can win.
        context_count = len(self._contexts(rows[0], rows[0]["models"][0])) if rows else 0
        self.context_scores = [defaultdict(lambda: [0.0, 0.0]) for _ in range(context_count)]
        for row in rows:
            for model in row["models"]:
                prior_correct = float(model["correct"])
                prior_total = 1.0
                for level, context in enumerate(self._contexts(row, model)):
                    correct, total = self.context_scores[level].get(context, (0.0, 0.0))
                    self.context_scores[level][context] = (
                        correct + prior_correct,
                        total + prior_total,
                    )

    def observe(self, row: dict[str, Any], selected_model: str) -> None:  # noqa: ARG002
        # Result-file migration can evaluate a rule without a calibration
        # partition.  Lazily create the structural levels from the first
        # labeled event so delayed feedback remains safe in that mode.
        if not self.context_scores and row["models"]:
            level_count = len(self._contexts(row, row["models"][0]))
            self.context_scores = [defaultdict(lambda: [0.0, 0.0]) for _ in range(level_count)]
        for model in row["models"]:
            for level, context in enumerate(self._contexts(row, model)):
                correct, total = self.context_scores[level].get(context, (0.0, 0.0))
                self.context_scores[level][context] = (
                    self.decay * correct + float(model["correct"]),
                    self.decay * total + 1.0,
                )


class PreviousErrorCorrectSetRule(HierarchicalReliabilityRule):
    """After a soft-vote error, listen only to models that predicted that event correctly."""

    family = "adaptive"
    description = (
        "If soft voting failed on the preceding event, restricts the next decision to constituents that were "
        "correct on that event; otherwise it uses the normal hierarchical selector."
    )
    interpretation = (
        "Tests short-lived competence persistence after a routing error: a model that just recognized the true "
        "transition may be better aligned with the case's current subprocess for one more step."
    )
    selection_policy = (
        "After a recoverable previous soft error, choose the most hierarchically reliable member of the previous "
        "correct-model set; fall back to all models when that set is empty or no error occurred."
    )

    def __init__(self, minimum_support: int = 3) -> None:
        super().__init__(minimum_support=minimum_support)
        self.name = f"previous-error correct-set persistence (support {minimum_support})"

    def select(self, row: dict[str, Any]) -> str:
        if row.get("previous_soft_correct") is False:
            previously_correct = set(row.get("previous_correct_models", []))
            eligible = [model for model in row["models"] if model["name"] in previously_correct]
            if eligible:
                return self._select_from(row, eligible)
        return super().select(row)


class TransientGeneralizationBoostRule(DecisionRule):
    """Temporarily route to the least-complex model after a soft error."""

    family = "adaptive recovery"
    description = (
        "After a soft-voting error, routes the next three predictions directly through the least-complex "
        "constituent. The rule is deterministic and has no fitted strength or decay parameters."
    )
    interpretation = (
        "Tests whether an error signals local over-specialization: briefly favoring the most general process "
        "view may help the ensemble recover at a branch or unfamiliar continuation."
    )
    selection_policy = (
        "Identify the unique minimum-complexity constituent and enforce its prediction for the configured "
        "recovery horizon after eligible previous soft errors."
    )

    def __init__(self, horizon: int = 3, trigger_mode: str = "generalist was correct") -> None:
        if trigger_mode not in TRANSIENT_BOOST_TRIGGER_LABELS:
            msg = f"Unknown transient-boost trigger: {trigger_mode}"
            raise ValueError(msg)
        self.horizon = horizon
        self.trigger_mode = trigger_mode
        self.name = f"transient generalization boost {TRANSIENT_BOOST_TRIGGER_LABELS[trigger_mode]} ({horizon} steps)"
        self._age_by_sequence: dict[int, int] = {}
        self.fitted_parameters: dict[str, Any] = {}

    @staticmethod
    def _generalist_names(row: dict[str, Any]) -> set[str]:
        minimum_complexity = min((int(model["complexity"]) for model in row["models"]), default=0)
        return {model["name"] for model in row["models"] if int(model["complexity"]) == minimum_complexity}

    def _error_triggers(self, row: dict[str, Any], trigger_mode: str) -> bool:
        if row.get("previous_soft_correct") is not False:
            return False
        generalist_was_correct = bool(self._generalist_names(row) & set(row.get("previous_correct_models", [])))
        if trigger_mode == "generalist was correct":
            return generalist_was_correct
        if trigger_mode == "generalist was wrong":
            return not generalist_was_correct
        return True

    def _next_age(self, row: dict[str, Any], ages: dict[int, int], trigger_mode: str) -> int | None:
        sequence = int(row["sequence_index"])
        if int(row["position"]) == 0:
            ages.pop(sequence, None)
        previous_age = ages.get(sequence)
        if self._error_triggers(row, trigger_mode):
            age = 0
        elif previous_age is not None and previous_age + 1 < self.horizon:
            age = previous_age + 1
        else:
            ages.pop(sequence, None)
            return None
        ages[sequence] = age
        return age

    def fit(self, rows: list[dict[str, Any]]) -> None:
        """Reset stream state and record the structural target without fitting labels."""
        self._age_by_sequence = {}
        generalists = sorted(self._generalist_names(rows[0])) if rows else []
        minimum_complexity = (
            min(int(model["complexity"]) for model in rows[0]["models"]) if rows and rows[0]["models"] else None
        )
        self.fitted_parameters = {
            "target": "minimum model complexity",
            "target_models": generalists,
            "minimum_complexity": minimum_complexity,
            "calibrated": False,
            "enforcement": "direct constituent prediction",
            "horizon": self.horizon,
            "trigger": self.trigger_mode,
        }

    def select(self, row: dict[str, Any]) -> str:  # noqa: ARG002
        return "minimum-complexity model"

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        age = self._next_age(row, self._age_by_sequence, self.trigger_mode)
        diagnostics = row.setdefault("rule_diagnostics", {})
        if age is None:
            diagnostics[self.name] = {
                "active": False,
                "recovery_age": None,
                "target_model": None,
                "target_complexity": None,
                "prediction_matches_target": None,
            }
            return "soft voting", row["soft_prediction"]
        generalists = [model for model in row["models"] if model["name"] in self._generalist_names(row)]
        # The default ensemble has one structural generalist (Bag). The stable
        # index tie-break keeps custom ensembles deterministic if they declare
        # several models at the same minimum complexity.
        target = min(generalists, key=lambda model: int(model["index"]))
        prediction = target["prediction"]
        diagnostics[self.name] = {
            "active": True,
            "recovery_age": age,
            "target_model": target["name"],
            "target_complexity": int(target["complexity"]),
            "prediction_matches_target": prediction == target["prediction"],
        }
        return target["name"], prediction


class TransientBagFavoritismRule(TransientGeneralizationBoostRule):
    """Temporarily upweight the structural generalist after it survives a soft error."""

    family = "independent model boost"
    description = (
        "After soft voting fails while the minimum-complexity constituent was correct, temporarily increases "
        "that constituent's distribution weight for three decisions."
    )
    interpretation = (
        "A correct generalist during an ensemble error is evidence that the case may have entered a transition "
        "where broad process memory is temporarily more reliable than specialized context."
    )
    selection_policy = (
        "Multiply only the unique minimum-complexity model's distribution by a positive exponentially decaying "
        "factor; leave every other model at its normal weight."
    )

    def __init__(
        self,
        horizon: int = 3,
        per_competing_model_boost: float = 1.0,
        decay: float = 0.5,
        trigger_mode: str = "generalist was correct",
    ) -> None:
        if trigger_mode not in {"generalist was correct", "generalist was wrong"}:
            msg = f"Unknown transient Bag trigger: {trigger_mode}"
            raise ValueError(msg)
        if horizon < 1:
            msg = "Transient Bag horizon must be positive."
            raise ValueError(msg)
        if trigger_mode == "generalist was wrong" and per_competing_model_boost == 1.0:
            per_competing_model_boost = 0.75
        if per_competing_model_boost < 0 or not 0 < decay <= 1:
            msg = "Per-model Bag boost must be non-negative and decay must be in (0, 1]."
            raise ValueError(msg)
        super().__init__(horizon=horizon, trigger_mode=trigger_mode)
        self.per_competing_model_boost = per_competing_model_boost
        self.decay = decay
        self.name = f"transient Bag favoritism {TRANSIENT_BOOST_TRIGGER_LABELS[trigger_mode]} ({horizon} steps)"
        if trigger_mode == "generalist was wrong":
            self.description = (
                "After soft voting and the minimum-complexity constituent both fail, temporarily increases that "
                "constituent's distribution weight for three decisions."
            )
            self.interpretation = (
                "Tests whether the broad process view remains the best short-term recovery bias even when it did "
                "not identify the error event itself."
            )

    @staticmethod
    def _target(row: dict[str, Any]) -> dict[str, Any]:
        minimum = min(int(model["complexity"]) for model in row["models"])
        return min(
            (model for model in row["models"] if int(model["complexity"]) == minimum),
            key=lambda model: int(model["index"]),
        )

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self._age_by_sequence = {}
        target = self._target(rows[0]) if rows else None
        self.fitted_parameters = {
            "calibrated": False,
            "target": "minimum model complexity",
            "target_models": [target["name"]] if target else [],
            "minimum_complexity": int(target["complexity"]) if target else None,
            "horizon": self.horizon,
            "trigger": (
                "previous soft error and minimum-complexity model was correct"
                if self.trigger_mode == "generalist was correct"
                else "previous soft error and minimum-complexity model was wrong"
            ),
            "per_competing_model_boost": self.per_competing_model_boost,
            "decay": self.decay,
            "multiplier_schedule": (
                [
                    1.0 + self.per_competing_model_boost * (len(rows[0]["models"]) - 1) * self.decay**age
                    for age in range(self.horizon)
                ]
                if rows
                else []
            ),
            "initial_multiplier": "1 + per_competing_model_boost * (model_count - 1)",
            "composition": "positive per-model multiplier",
        }

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        age = self._next_age(row, self._age_by_sequence, self.trigger_mode)
        target = self._target(row)
        competing_models = len(row["models"]) - 1
        multiplier = (
            1.0 + self.per_competing_model_boost * competing_models * self.decay**age if age is not None else 1.0
        )
        model_multipliers = {target["name"]: multiplier} if multiplier > 1.0 else {}
        prediction = _weighted_model_distribution_prediction(row, model_multipliers)
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": bool(model_multipliers),
            "recovery_age": age,
            "target_model": target["name"],
            "target_complexity": int(target["complexity"]),
            "multiplier": multiplier,
            "model_multipliers": model_multipliers,
            "prediction_matches_target": prediction == target["prediction"] if model_multipliers else None,
        }
        source = "transient generalist-weighted pool" if model_multipliers else "soft voting"
        return source, prediction if model_multipliers else row["soft_prediction"]


class CalibratedGeneralizationRecoveryRule(TransientGeneralizationBoostRule):
    """Gate or soften transient generalist recovery using calibration evidence."""

    family = "calibrated recovery"
    MODES = frozenset({"age gate", "structural gate", "distribution blend"})

    def __init__(self, calibration_mode: str, *, horizon: int = 3, minimum_support: int = 5) -> None:
        if calibration_mode not in self.MODES:
            msg = f"Unknown recovery calibration mode: {calibration_mode}"
            raise ValueError(msg)
        super().__init__(horizon=horizon, trigger_mode="all soft errors")
        self.calibration_mode = calibration_mode
        self.minimum_support = minimum_support
        self.name = f"transient recovery calibrated by {calibration_mode}"
        self.description = {
            "age gate": (
                "Uses Bag only at recovery ages where calibration measured positive paired gain over soft voting."
            ),
            "structural gate": (
                "Uses Bag only in supported recovery-age and uncertainty contexts with positive calibrated gain."
            ),
            "distribution blend": (
                "Calibrates a finite Bag-distribution multiplier independently for each recovery age."
            ),
        }[calibration_mode]
        self.interpretation = (
            "Tests whether post-error generalization is useful only at particular recovery ages or process regimes, "
            "instead of assuming every soft-voting error justifies a Bag override."
        )
        self.selection_policy = self.description
        self._enabled_ages: set[int] = set()
        self._supported_contexts: set[tuple[str, ...]] = set()
        self._enabled_contexts: set[tuple[str, ...]] = set()
        self._multiplier_by_age: dict[int, float] = {}

    @staticmethod
    def _context(row: dict[str, Any], age: int) -> tuple[str, ...]:
        return (
            str(age),
            _sequence_stage(row),
            str(row["agreement_count"]),
            _value_bin(float(row["soft_margin"]), (0.01, 0.03, 0.07, 0.15)),
            _value_bin(float(row["soft_normalized_entropy"]), (0.25, 0.5, 0.7, 0.85)),
        )

    @staticmethod
    def _generalist_prediction(row: dict[str, Any]) -> tuple[str, str, int]:
        minimum = min(int(model["complexity"]) for model in row["models"])
        target = min(
            (model for model in row["models"] if int(model["complexity"]) == minimum),
            key=lambda model: int(model["index"]),
        )
        return target["name"], target["prediction"], minimum

    @classmethod
    def _blended_prediction(cls, row: dict[str, Any], multiplier: float) -> str:
        generalists = cls._generalist_names(row)
        scores: defaultdict[str, float] = defaultdict(float)
        for model in row["models"]:
            weight = multiplier if model["name"] in generalists else 1.0
            for activity, probability in model["distribution"].items():
                scores[activity] += weight * float(probability)
        return max(sorted(scores), key=scores.get) if scores else row["soft_prediction"]  # type: ignore[arg-type]

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self._age_by_sequence = {}
        self._enabled_ages = set()
        self._supported_contexts = set()
        self._enabled_contexts = set()
        self._multiplier_by_age = {}
        age_rows: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        context_rows: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        ages: dict[int, int] = {}
        for row in rows:
            age = self._next_age(row, ages, self.trigger_mode)
            if age is None:
                continue
            age_rows[age].append(row)
            context_rows[self._context(row, age)].append(row)

        def positive_gain(group: list[dict[str, Any]]) -> bool:
            return (
                len(group) >= self.minimum_support
                and sum(
                    int(self._generalist_prediction(row)[1] == row["actual"]) - int(row["soft_correct"])
                    for row in group
                )
                > 0
            )

        self._enabled_ages = {age for age, group in age_rows.items() if positive_gain(group)}
        if self.calibration_mode == "structural gate":
            self._supported_contexts = {
                context for context, group in context_rows.items() if len(group) >= self.minimum_support
            }
            self._enabled_contexts = {context for context, group in context_rows.items() if positive_gain(group)}
        if self.calibration_mode == "distribution blend":
            for age, group in age_rows.items():
                candidates = (1.0, 1.25, 1.5, 2.0, 3.0)
                self._multiplier_by_age[age] = max(
                    candidates,
                    key=lambda multiplier: (
                        sum(self._blended_prediction(row, multiplier) == row["actual"] for row in group),
                        -multiplier,
                    ),
                )
        self._age_by_sequence = {}
        self.fitted_parameters = {
            "calibrated": True,
            "calibration_available": bool(rows),
            "technique": self.calibration_mode,
            "minimum_support": self.minimum_support,
            "enabled_ages": sorted(self._enabled_ages),
            "enabled_contexts": len(self._enabled_contexts),
            "supported_contexts": len(self._supported_contexts),
            "multiplier_by_age": {str(age): value for age, value in self._multiplier_by_age.items()},
            "horizon": self.horizon,
        }

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        age = self._next_age(row, self._age_by_sequence, self.trigger_mode)
        diagnostics = row.setdefault("rule_diagnostics", {})
        target_name, target_prediction, target_complexity = self._generalist_prediction(row)
        applied = False
        multiplier = 1.0
        if age is not None:
            if self.calibration_mode == "age gate":
                applied = age in self._enabled_ages
            elif self.calibration_mode == "structural gate":
                context = self._context(row, age)
                applied = (
                    context in self._enabled_contexts
                    if context in self._supported_contexts
                    else age in self._enabled_ages
                )
            else:
                multiplier = self._multiplier_by_age.get(age, 1.0)
                applied = multiplier > 1.0
        if age is None or not applied:
            prediction = row["soft_prediction"]
            source = "soft voting"
        elif self.calibration_mode == "distribution blend":
            prediction = self._blended_prediction(row, multiplier)
            source = "calibrated generalist distribution blend"
        else:
            prediction = target_prediction
            source = target_name
        diagnostics[self.name] = {
            "active": applied,
            "recovery_age": age,
            "target_model": target_name,
            "target_complexity": target_complexity,
            "calibration_mode": self.calibration_mode,
            "calibration_allowed": applied,
            "multiplier": multiplier,
            "prediction_matches_target": prediction == target_prediction if applied else None,
        }
        return source, prediction


class CalibratedTransientGeneralistPoolRule(DecisionRule):
    """Select a transient generalist-pool policy from an internal case holdout."""

    family = "calibrated recovery"
    description = (
        "Selects the trigger, horizon, strength, and decay of a minimum-complexity distribution boost from "
        "out-of-fit calibration cases."
    )
    interpretation = (
        "Separates a generalist-correct branch from a complete generalist failure: the latter can still signal "
        "that specialized states are temporarily unreliable, but requires a different recovery schedule."
    )
    selection_policy = (
        "Use only the transient policy whose paired selector-holdout lower bound exceeds a small positive gain; "
        "otherwise retain soft voting for every event."
    )
    policies = tuple(
        (trigger, horizon, boost, decay)
        for trigger in ("generalist was correct", "generalist was wrong")
        for horizon in (1, 2, 3, 4, 5)
        for boost in (0.5, 1.0, 1.5, 2.0, 3.0)
        for decay in (0.2, 0.4, 0.6, 1.0)
    )

    def __init__(self, minimum_lower_bound: float = 0.005, confidence_z: float = 0.5) -> None:
        self.minimum_lower_bound = minimum_lower_bound
        self.confidence_z = confidence_z
        self.name = "calibrated transient generalist pool"
        self.selected_policy: tuple[str, int, float, float] | None = None
        self.selected_rule: TransientBagFavoritismRule | None = None
        self.fitted_parameters: dict[str, Any] = {}

    @staticmethod
    def _paired_lower_bound(predictions: list[str], rows: list[dict[str, Any]], confidence_z: float) -> tuple[float, float]:
        outcomes = [
            int(prediction == row["actual"]) - int(row["soft_correct"])
            for prediction, row in zip(predictions, rows, strict=True)
        ]
        if not outcomes:
            return float("-inf"), 0.0
        mean = sum(outcomes) / len(outcomes)
        variance = sum((outcome - mean) ** 2 for outcome in outcomes) / max(1, len(outcomes) - 1)
        return mean - confidence_z * math.sqrt(variance / len(outcomes)), mean

    @staticmethod
    def _rule(policy: tuple[str, int, float, float]) -> TransientBagFavoritismRule:
        trigger, horizon, boost, decay = policy
        return TransientBagFavoritismRule(
            horizon=horizon,
            per_competing_model_boost=boost,
            decay=decay,
            trigger_mode=trigger,
        )

    def fit(self, rows: list[dict[str, Any]]) -> None:
        fitting_rows, selector_rows = split_integration_calibration(rows)
        evidence: list[tuple[float, float, tuple[str, int, float, float]]] = []
        for policy in self.policies:
            candidate = self._rule(policy)
            candidate.fit(fitting_rows)
            predictions = [candidate.choose(row)[1] for row in selector_rows]
            lower_bound, mean = self._paired_lower_bound(predictions, selector_rows, self.confidence_z)
            evidence.append((lower_bound, mean, policy))
        best = max(evidence, default=(float("-inf"), 0.0, None), key=lambda item: (item[0], item[1]))
        self.selected_policy = best[2] if best[0] >= self.minimum_lower_bound else None
        self.selected_rule = self._rule(self.selected_policy) if self.selected_policy is not None else None
        if self.selected_rule is not None:
            self.selected_rule.fit(rows)
        self.fitted_parameters = {
            "calibrated": True,
            "selector_events": len(selector_rows),
            "confidence_z": self.confidence_z,
            "minimum_lower_bound": self.minimum_lower_bound,
            "best_selector_lower_bound": best[0],
            "best_selector_mean_gain": best[1],
            "selected_policy": (
                {
                    "trigger": self.selected_policy[0],
                    "horizon": self.selected_policy[1],
                    "per_competing_model_boost": self.selected_policy[2],
                    "decay": self.selected_policy[3],
                }
                if self.selected_policy is not None
                else None
            ),
        }

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        if self.selected_rule is None:
            row.setdefault("rule_diagnostics", {})[self.name] = {"active": False, "selected_policy": None}
            return "soft voting", row["soft_prediction"]
        source, prediction = self.selected_rule.choose(row)
        underlying = row.get("rule_diagnostics", {}).get(self.selected_rule.name, {})
        row.setdefault("rule_diagnostics", {})[self.name] = {
            **underlying,
            "selected_policy": self.fitted_parameters["selected_policy"],
        }
        return source, prediction


def _ngram_window(model: dict[str, Any]) -> int | None:
    """Return structural N-gram order, with a compatibility fallback for old result files."""
    if model.get("model_type") == "ngram" and model.get("window_size") is not None:
        return int(model["window_size"])
    name = str(model.get("name", ""))
    if name.startswith("ngram_"):
        suffix = name.removeprefix("ngram_")
        return int(suffix) if suffix.isdigit() else None
    return None


def _weighted_model_distribution_prediction(
    row: dict[str, Any],
    model_multipliers: dict[str, float],
) -> str:
    """Merge full model distributions after applying non-negative local multipliers."""
    if any(multiplier < 0.0 for multiplier in model_multipliers.values()):
        msg = "Model multipliers must be non-negative."
        raise ValueError(msg)
    scores: defaultdict[str, float] = defaultdict(float)
    for model in row["models"]:
        multiplier = model_multipliers.get(model["name"], 1.0)
        for activity, probability in model["distribution"].items():
            scores[activity] += multiplier * float(probability)
    return max(sorted(scores), key=scores.get) if scores else row["soft_prediction"]  # type: ignore[arg-type]


class NGramCorrectStreakBoostRule(DecisionRule):
    """Upweight N-grams whose recent predictions have remained correct."""

    name = "ngram correctness-streak multiplier"
    family = "independent model boost"
    exponential_rate = 4.0
    minimum_multiplier = 0.1
    description = (
        "After a soft-voting error, independently downweights each eligible N-gram distribution from its delayed correctness streak."
    )
    interpretation = (
        "A sustained correct streak suggests that an N-gram's context has become mature and is currently aligned "
        "with the active local process path, but a failed soft vote calls for temporarily deferring to the generalist."
    )
    selection_policy = (
        "For N-gram order x, track delayed consecutive correctness and use "
        "a complexity-scaled bounded exponential multiplier: 0.1 at zero streak, 1 at half-window progress, and "
        "at least 2 near maturity; a previously wrong N-gram prediction receives multiplier 0."
    )

    def __init__(
        self,
        *,
        boost_scale: float = 1.5,
    ) -> None:
        self.boost_scale = boost_scale
        self._streaks_by_sequence: dict[int, dict[str, int]] = {}
        self._last_correct_by_sequence: dict[int, dict[str, bool]] = {}
        self.fitted_parameters: dict[str, Any] = {}

    @staticmethod
    def boost_ratio(streak: int, window_size: int) -> float:
        """Exponentially favor the longest pre-maturity correctness streaks."""
        if streak >= window_size + 1:
            return 0.0
        linear_ratio = min(1.0, max(0.0, streak / (window_size - 1)))
        return (math.exp(NGramCorrectStreakBoostRule.exponential_rate * linear_ratio) - 1.0) / (
            math.exp(NGramCorrectStreakBoostRule.exponential_rate) - 1.0
        )

    @classmethod
    def streak_multiplier(cls, streak: int, window_size: int, minimum_window_size: int | None = None) -> float:
        """Return the bounded exponential N-gram weight for a pre-maturity streak."""
        if streak >= window_size + 1:
            return 1.0
        progress = min(1.0, max(0.0, streak / (window_size - 1)))
        if progress <= 0.5:
            base_multiplier = cls.minimum_multiplier * 10.0 ** (2.0 * progress)
        else:
            base_multiplier = 2.0 ** (2.0 * progress - 1.0)
        return base_multiplier * math.sqrt(window_size / (minimum_window_size or window_size))

    def fit(self, rows: list[dict[str, Any]]) -> None:  # noqa: ARG002
        """Reset delayed state; all parameters are fixed and label-independent."""
        self._streaks_by_sequence = {}
        self._last_correct_by_sequence = {}
        self.fitted_parameters = {
            "calibrated": False,
            "formula": "(exp(4 * min(1, max(0, streak / (window_size - 1)))) - 1) / (exp(4) - 1) if streak < window_size + 1 else 0",
            "eligibility_streak": "window_size",
            "exponential_rate": self.exponential_rate,
            "ngram_multiplier": "0.1 * 10 ** (2 * progress) if progress <= 0.5 else 2 ** (2 * progress - 1)",
            "multiplier_progress": "min(1, max(0, streak / (window_size - 1)))",
            "multiplier_schedule": {"zero_streak": 0.1, "half_window": 1.0, "full_window": 2.0},
            "complexity_scaling": "sqrt(window_size / minimum_ngram_window_size)",
            "wrong_previous_prediction_multiplier": 0.0,
            "boost_scale": self.boost_scale,
            "activation": "previous soft-voting prediction was wrong",
            "composition": "non-negative per-model multiplier",
        }

    def _reset_if_needed(self, row: dict[str, Any]) -> dict[str, int]:
        sequence = int(row["sequence_index"])
        if int(row["position"]) == 0:
            self._streaks_by_sequence.pop(sequence, None)
            self._last_correct_by_sequence.pop(sequence, None)
        return self._streaks_by_sequence.setdefault(sequence, {})

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        streaks = self._reset_if_needed(row)
        last_correct = self._last_correct_by_sequence.setdefault(int(row["sequence_index"]), {})
        follows_soft_error = row.get("previous_soft_correct") is False
        scores: defaultdict[str, float] = defaultdict(float)
        ratios: dict[str, float] = {}
        eligible: dict[str, bool] = {}
        multipliers: dict[str, float] = {}
        ngram_windows = [_ngram_window(model) for model in row["models"]]
        minimum_window_size = min((window for window in ngram_windows if window is not None), default=1)
        for model in row["models"]:
            window_size = _ngram_window(model)
            streak = streaks.get(model["name"], 0)
            ratio = self.boost_ratio(streak, window_size) if window_size else 0.0
            if window_size and last_correct.get(model["name"]) is False:
                multiplier = 0.0
            elif follows_soft_error and window_size:
                multiplier = self.streak_multiplier(streak, window_size, minimum_window_size)
            else:
                multiplier = 1.0
            if window_size:
                ratios[model["name"]] = ratio
                eligible[model["name"]] = 0 < streak < window_size + 1
                multipliers[model["name"]] = multiplier
            for activity, probability in model["distribution"].items():
                scores[activity] += multiplier * float(probability)
        modified_models = [name for name, multiplier in multipliers.items() if multiplier != 1.0]
        prediction = max(sorted(scores), key=scores.get) if scores else row["soft_prediction"]  # type: ignore[arg-type]
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": bool(modified_models),
            "follows_soft_error": follows_soft_error,
            "streaks": {name: streaks.get(name, 0) for name in ratios},
            "previous_prediction_correct": {name: last_correct.get(name) for name in ratios},
            "eligible": eligible,
            "boost_ratios": ratios,
            "multipliers": multipliers,
            "model_multipliers": {name: multiplier for name, multiplier in multipliers.items() if multiplier != 1.0},
            "modified_models": modified_models,
        }
        return ("ngram streak-weighted pool", prediction) if modified_models else ("soft voting", row["soft_prediction"])

    def select(self, row: dict[str, Any]) -> str:  # noqa: ARG002
        return "ngram streak-weighted pool"

    def observe(self, row: dict[str, Any], selected_model: str) -> None:  # noqa: ARG002
        streaks = self._streaks_by_sequence.setdefault(int(row["sequence_index"]), {})
        last_correct = self._last_correct_by_sequence.setdefault(int(row["sequence_index"]), {})
        for model in row["models"]:
            if _ngram_window(model) is None:
                continue
            streaks[model["name"]] = streaks.get(model["name"], 0) + 1 if model["correct"] else 0
            last_correct[model["name"]] = bool(model["correct"])


class LargestNGramDisagreementMultiplierRule(DecisionRule):
    """Slightly favor the largest N-gram only when streak weights are comparable."""

    name = "largest N-gram disagreement multiplier"
    family = "independent model boost"
    description = "When N-grams disagree, slightly favors the largest order only if their streak multipliers are comparable."
    interpretation = "When no N-gram order has a clear streak advantage, the largest context gets a small tie-breaking preference."
    selection_policy = (
        "If N-gram top predictions disagree and max(streak_multiplier) / min(streak_multiplier) <= 1.1, "
        "multiply every largest-order N-gram by 1.1; otherwise preserve the streak multiplier as the dominant signal."
    )

    def __init__(self, *, multiplier: float = 1.1, comparability_ratio: float = 1.1) -> None:
        self.multiplier = multiplier
        self.comparability_ratio = comparability_ratio
        self.fitted_parameters: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> None:  # noqa: ARG002
        self.fitted_parameters = {
            "calibrated": False,
            "largest_ngram_multiplier": self.multiplier,
            "maximum_streak_multiplier_ratio": self.comparability_ratio,
            "activation": "N-grams disagree and their streak multipliers are comparable",
            "composition": "strictly positive per-model multiplier",
        }

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        ngrams = [(model, _ngram_window(model)) for model in row["models"]]
        ngrams = [(model, window) for model, window in ngrams if window is not None]
        streak_diagnostic = row.get("rule_diagnostics", {}).get(NGramCorrectStreakBoostRule.name, {})
        streak_multipliers = streak_diagnostic.get("multipliers", {})
        weights = [float(streak_multipliers.get(model["name"], 1.0)) for model, _ in ngrams]
        predictions = {model["prediction"] for model, _ in ngrams if model["prediction"]}
        comparable = bool(weights) and min(weights) > 0.0 and max(weights) / min(weights) <= self.comparability_ratio
        disagree = len(predictions) > 1
        largest_window = max((window for _, window in ngrams), default=None)
        targets = [model["name"] for model, window in ngrams if window == largest_window] if largest_window else []
        active = disagree and comparable and bool(targets)
        model_multipliers = {name: self.multiplier for name in targets} if active else {}
        prediction = _weighted_model_distribution_prediction(row, model_multipliers)
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": active,
            "ngram_predictions_disagree": disagree,
            "streak_multipliers": {model["name"]: weight for (model, _), weight in zip(ngrams, weights, strict=True)},
            "streak_multipliers_comparable": comparable,
            "largest_window": largest_window,
            "model_multipliers": model_multipliers,
        }
        return ("largest ngram weighted pool", prediction) if active else ("soft voting", row["soft_prediction"])


def _ranked_soft_activity(row: dict[str, Any], rank: int) -> str:
    ranked = row.get("soft_ranked_predictions", [])
    return ranked[rank - 1]["activity"] if len(ranked) >= rank else ""


def _distribution_contexts(row: dict[str, Any]) -> list[tuple[str, ...]]:
    """Describe probability regimes without encoding activity identities."""
    previous_outcome = "unknown" if row["previous_soft_correct"] is None else str(row["previous_soft_correct"])
    margin = _value_bin(float(row["soft_margin"]), (0.01, 0.03, 0.07, 0.15))
    entropy = _value_bin(float(row["soft_normalized_entropy"]), (0.25, 0.5, 0.7, 0.85))
    spread = _value_bin(float(row["model_confidence_spread"]), (0.1, 0.25, 0.5, 0.75))
    topology = _prediction_partition(row)
    visits = tuple(_visit_bin(int(model["state_visits"])) for model in row["models"])
    soft_ranks = tuple(_soft_rank_of_model(row, model) for model in row["models"])
    return [
        (
            "feedback-evidence",
            previous_outcome,
            str(row["previous_wrong_model_count"]),
            str(row["previous_empty_prediction_count"]),
            margin,
            *topology,
        ),
        (
            "state-availability",
            str(row["empty_prediction_count"]),
            str(row["agreement_count"]),
            entropy,
            margin,
            *visits,
        ),
        ("rank-topology", *soft_ranks, *topology, margin),
        ("distribution", entropy, margin, spread, str(row["agreement_count"]), *topology),
        ("stage", _sequence_stage(row), previous_outcome, entropy, margin),
        ("global",),
    ]


class CalibratedSoftRankRule(DecisionRule):
    """Prefer the second or third soft activity in calibrated contexts."""

    family = "soft rank"
    description = "Uses a lower-ranked soft-distribution activity only in structural regimes where it beat rank one."
    interpretation = "Tests whether uncertainty shape exposes systematic top-rank reversals without choosing a model."
    selection_policy = "Choose a calibrated soft-distribution rank, independent of constituent model identity."

    def __init__(self, rank: int, minimum_support: int = 2) -> None:
        self.rank = rank
        self.minimum_support = minimum_support
        self.name = f"calibrated soft rank {rank} override (support {minimum_support})"
        self.context_decisions: dict[tuple[str, ...], bool] = {}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if _ranked_soft_activity(row, self.rank):
                for context in _distribution_contexts(row):
                    grouped[context].append(row)
        self.context_decisions = {}
        for context, context_rows in grouped.items():
            if len(context_rows) < self.minimum_support:
                continue
            rank_correct = sum(_ranked_soft_activity(row, self.rank) == row["actual"] for row in context_rows)
            soft_correct = sum(row["soft_correct"] for row in context_rows)
            gain = (rank_correct - soft_correct) / len(context_rows)
            self.context_decisions[context] = gain > 0

    def select(self, row: dict[str, Any]) -> str:  # noqa: ARG002
        return f"soft rank {self.rank}"

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        candidate = _ranked_soft_activity(row, self.rank)
        if candidate:
            for context in _distribution_contexts(row):
                if context in self.context_decisions:
                    return (
                        (f"soft rank {self.rank}", candidate)
                        if self.context_decisions[context]
                        else ("soft voting", row["soft_prediction"])
                    )
        return "soft voting", row["soft_prediction"]


class CalibratedModelRankRule(DecisionRule):
    """Prefer the globally second- or third-ranked model in favorable contexts."""

    family = "model rank"
    description = "Routes to a calibration-accuracy rank only in structural regimes where that rank beat soft voting."
    interpretation = "Tests whether a globally weaker abstraction becomes a useful specialist in some process regimes."
    selection_policy = "Choose the requested rank in the calibration-accuracy ordering computed for this run."

    def __init__(self, rank: int, minimum_support: int = 8) -> None:
        self.rank = rank
        self.minimum_support = minimum_support
        self.name = f"calibrated model rank {rank} switch (support {minimum_support})"
        self.model_name = ""
        self.context_decisions: dict[tuple[str, ...], bool] = {}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        model_names = [model["name"] for model in rows[0]["models"]]
        accuracies = {
            name: sum(_model_prediction(row, name) == row["actual"] for row in rows) / len(rows) for name in model_names
        }
        ranked_models = sorted(model_names, key=lambda name: (accuracies[name], -model_names.index(name)), reverse=True)
        self.model_name = ranked_models[min(self.rank - 1, len(ranked_models) - 1)]
        grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for context in _distribution_contexts(row):
                grouped[context].append(row)
        self.context_decisions = {}
        for context, context_rows in grouped.items():
            if len(context_rows) < self.minimum_support:
                continue
            model_correct = sum(_model_prediction(row, self.model_name) == row["actual"] for row in context_rows)
            self.context_decisions[context] = model_correct > sum(row["soft_correct"] for row in context_rows)

    def select(self, row: dict[str, Any]) -> str:
        for context in _distribution_contexts(row):
            if context in self.context_decisions:
                return self.model_name if self.context_decisions[context] else "soft voting"
        return "soft voting"

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        source = self.select(row)
        return (source, row["soft_prediction"]) if source == "soft voting" else (source, _model_prediction(row, source))


class CalibratedLoneDissenterRule(DecisionRule):
    """Override soft voting only for a calibration-verified single contrarian."""

    family = "calibrated contrarian"
    description = (
        "Overrides soft voting only when one dissenting model has a supported, uncertainty-adjusted calibration "
        "advantage in the current disagreement regime."
    )
    interpretation = (
        "Identifies a model that repeatedly recognizes a local process transition despite a competing model consensus."
    )
    selection_policy = (
        "Keep soft voting unless exactly one model dissents from a consensus and that model's shrunken paired "
        "advantage over soft voting has a positive lower confidence bound."
    )

    def __init__(self, minimum_support: int = 2, confidence_z: float = 0.5, prior_weight: float = 4.0) -> None:
        self.minimum_support = minimum_support
        self.confidence_z = confidence_z
        self.prior_weight = prior_weight
        self.name = f"calibrated lone-dissenter override (support {minimum_support})"
        self.context_decisions: dict[tuple[str, ...], dict[str, float | int | bool]] = {}
        self.global_advantage: dict[str, float] = {}
        self.fitted_parameters: dict[str, Any] = {}

    @staticmethod
    def _dissenter(row: dict[str, Any]) -> dict[str, Any] | None:
        """Return the unique non-empty top-1 dissenter, if a genuine consensus exists."""
        consensus = row.get("consensus_prediction", "")
        dissenters = [
            model
            for model in row["models"]
            if model["prediction"] and model["prediction"] != consensus
        ]
        if not consensus or int(row["agreement_count"]) < 2 or len(dissenters) != 1:
            return None
        return dissenters[0]

    @staticmethod
    def _contexts(row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        """Describe a contrarian opportunity without retaining activity identities."""
        confidence = _value_bin(float(model["confidence"]), (0.2, 0.4, 0.6, 0.8))
        margin = _value_bin(float(model["margin"]), (0.03, 0.1, 0.25, 0.5))
        divergence = _value_bin(float(model["soft_divergence"]), (0.01, 0.03, 0.08, 0.15))
        soft_margin = _value_bin(float(row["soft_margin"]), (0.01, 0.03, 0.07, 0.15))
        visits = _visit_bin(int(model["state_visits"]))
        consensus_size = str(row["agreement_count"])
        topology = _prediction_partition(row)
        return [
            (
                model["name"],
                model["state"],
                visits,
                confidence,
                margin,
                divergence,
                consensus_size,
                soft_margin,
                *topology,
            ),
            (model["name"], visits, confidence, margin, divergence, consensus_size, soft_margin, *topology),
            (model["name"], confidence, margin, divergence, consensus_size, soft_margin),
            (model["name"], divergence, consensus_size, soft_margin),
            (model["name"], consensus_size),
        ]

    def fit(self, rows: list[dict[str, Any]]) -> None:
        totals: Counter[str] = Counter()
        advantages: Counter[str] = Counter()
        for row in rows:
            for model in row["models"]:
                totals[model["name"]] += 1
                advantages[model["name"]] += int(model["correct"]) - int(row["soft_correct"])
        self.global_advantage = {
            name: advantages[name] / total for name, total in totals.items() if total
        }

        grouped: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        for row in rows:
            model = self._dissenter(row)
            if model is None:
                continue
            delta = int(model["correct"]) - int(row["soft_correct"])
            for context in self._contexts(row, model):
                grouped[context].append(delta)

        self.context_decisions = {}
        for context, observations in grouped.items():
            support = len(observations)
            if support < self.minimum_support:
                continue
            model_name = context[0]
            raw_advantage = sum(observations) / support
            shrunken_advantage = (
                sum(observations) + self.prior_weight * self.global_advantage.get(model_name, 0.0)
            ) / (support + self.prior_weight)
            variance = (
                sum((delta - raw_advantage) ** 2 for delta in observations) / (support - 1)
                if support > 1
                else 1.0
            )
            lower_bound = shrunken_advantage - self.confidence_z * math.sqrt(variance / support)
            self.context_decisions[context] = {
                "support": support,
                "raw_advantage": raw_advantage,
                "shrunken_advantage": shrunken_advantage,
                "lower_bound": lower_bound,
                "active": lower_bound > 0.0,
            }
        self.fitted_parameters = {
            "calibrated": True,
            "minimum_support": self.minimum_support,
            "confidence_z": self.confidence_z,
            "prior_weight": self.prior_weight,
            "activation": "unique dissenter with positive shrunken paired-advantage lower bound",
        }

    def select(self, row: dict[str, Any]) -> str:
        model = self._dissenter(row)
        if model is None:
            return "soft voting"
        for context in self._contexts(row, model):
            decision = self.context_decisions.get(context)
            if decision is not None:
                return model["name"] if decision["active"] else "soft voting"
        return "soft voting"

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        model = self._dissenter(row)
        decision: dict[str, float | int | bool] | None = None
        if model is not None:
            decision = next(
                (self.context_decisions[context] for context in self._contexts(row, model) if context in self.context_decisions),
                None,
            )
        source = model["name"] if model is not None and decision and decision["active"] else "soft voting"
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": source != "soft voting",
            "candidate_model": model["name"] if model is not None else "",
            "consensus_size": int(row["agreement_count"]),
            "decision": decision or {},
        }
        return (source, _model_prediction(row, source)) if source != "soft voting" else (source, row["soft_prediction"])


class CalibratedLoneDissenterSecondRankRule(DecisionRule):
    """Use a lone dissenter's second activity only in calibration-verified risk regimes."""

    family = "calibrated contrarian"
    description = (
        "Uses the unique dissenter's second-ranked activity only after observable failure-risk signals and "
        "a supported calibration advantage over soft voting."
    )
    interpretation = (
        "Tests whether the contrarian model's next-best transition is informative when the case is sparse, "
        "variable, or follows a complete ensemble miss."
    )
    selection_policy = (
        "Keep soft voting unless one model dissents, its distinct second activity is available, a failure-risk "
        "signal is present, and its shrunken paired-advantage lower bound is positive."
    )

    def __init__(self, minimum_support: int = 2, confidence_z: float = 0.5, prior_weight: float = 4.0) -> None:
        self.minimum_support = minimum_support
        self.confidence_z = confidence_z
        self.prior_weight = prior_weight
        self.name = f"calibrated lone-dissenter rank 2 override (support {minimum_support})"
        self.context_decisions: dict[tuple[str, ...], dict[str, float | int | bool]] = {}
        self.global_advantage = 0.0
        self.fitted_parameters: dict[str, Any] = {}

    @staticmethod
    def _candidate(row: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
        model = CalibratedLoneDissenterRule._dissenter(row)
        if model is None or len(model.get("ranked_predictions", [])) < 2:
            return None
        candidate = str(model["ranked_predictions"][1]["activity"])
        if not candidate or candidate == row["soft_prediction"]:
            return None
        return model, candidate

    @staticmethod
    def _risk_signals(row: dict[str, Any], model: dict[str, Any]) -> tuple[str, ...]:
        """Return observable signals that make the ordinary consensus less dependable."""
        signals = []
        if row.get("previous_soft_correct") is False and not row.get("previous_correct_models", []):
            signals.append("previous_all_models_failed")
        if int(model["state_visits"]) <= VISIT_FEW_MAX:
            signals.append("low_representation")
        if (
            float(row["soft_normalized_entropy"]) >= 0.7
            or float(row["soft_margin"]) <= 0.07
            or float(row["model_confidence_spread"]) >= 0.5
        ):
            signals.append("high_variability")
        return tuple(signals)

    @classmethod
    def _contexts(cls, row: dict[str, Any], model: dict[str, Any], signals: tuple[str, ...]) -> list[tuple[str, ...]]:
        visits = _visit_bin(int(model["state_visits"]))
        confidence = _value_bin(float(model["confidence"]), (0.2, 0.4, 0.6, 0.8))
        margin = _value_bin(float(model["margin"]), (0.03, 0.1, 0.25, 0.5))
        soft_margin = _value_bin(float(row["soft_margin"]), (0.01, 0.03, 0.07, 0.15))
        return [
            (
                model["name"],
                model["state"],
                _sequence_stage(row),
                visits,
                confidence,
                margin,
                soft_margin,
                *signals,
                *_prediction_partition(row),
            ),
            (model["name"], _sequence_stage(row), visits, confidence, soft_margin, *signals),
            (model["name"], visits, soft_margin, *signals),
            (model["name"], *signals),
        ]

    def fit(self, rows: list[dict[str, Any]]) -> None:
        grouped: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        all_deltas: list[int] = []
        for row in rows:
            candidate = self._candidate(row)
            if candidate is None:
                continue
            model, prediction = candidate
            signals = self._risk_signals(row, model)
            if not signals:
                continue
            delta = int(prediction == row["actual"]) - int(row["soft_correct"])
            all_deltas.append(delta)
            for context in self._contexts(row, model, signals):
                grouped[context].append(delta)
        self.global_advantage = sum(all_deltas) / len(all_deltas) if all_deltas else 0.0
        self.context_decisions = {}
        for context, observations in grouped.items():
            support = len(observations)
            if support < self.minimum_support:
                continue
            raw_advantage = sum(observations) / support
            shrunken_advantage = (sum(observations) + self.prior_weight * self.global_advantage) / (
                support + self.prior_weight
            )
            variance = (
                sum((delta - raw_advantage) ** 2 for delta in observations) / (support - 1)
                if support > 1
                else 1.0
            )
            lower_bound = shrunken_advantage - self.confidence_z * math.sqrt(variance / support)
            self.context_decisions[context] = {
                "support": support,
                "raw_advantage": raw_advantage,
                "shrunken_advantage": shrunken_advantage,
                "lower_bound": lower_bound,
                "active": lower_bound > 0.0,
            }
        self.fitted_parameters = {
            "calibrated": True,
            "minimum_support": self.minimum_support,
            "confidence_z": self.confidence_z,
            "prior_weight": self.prior_weight,
            "risk_signals": ["previous_all_models_failed", "low_representation", "high_variability"],
        }

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        candidate = self._candidate(row)
        model: dict[str, Any] | None = None
        prediction = ""
        signals: tuple[str, ...] = ()
        decision: dict[str, float | int | bool] | None = None
        if candidate is not None:
            model, prediction = candidate
            signals = self._risk_signals(row, model)
            if signals:
                decision = next(
                    (self.context_decisions[context] for context in self._contexts(row, model, signals) if context in self.context_decisions),
                    None,
                )
        active = bool(decision and decision["active"])
        source = f"{model['name']} rank 2" if active and model is not None else "soft voting"
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": active,
            "candidate_model": model["name"] if model is not None else "",
            "candidate_prediction": prediction,
            "risk_signals": list(signals),
            "decision": decision or {},
        }
        return (source, prediction) if active else (source, row["soft_prediction"])


class CalibratedComplexityContrastExceptionRule(DecisionRule):
    """Find calibration-verified exceptions by contrasting high- and low-complexity distributions."""

    family = "complexity contrast"
    description = (
        "When the top choice is not individually predominant, continuously contrasts complexity-weighted probability evidence to "
        "identify a well-supported alternative."
    )
    interpretation = (
        "Tests whether generalist models mask a specific exception that richer process contexts retain as a "
        "strong second or third possibility."
    )
    selection_policy = (
        "Keep soft voting unless a high-complexity top-three alternative survives low-complexity subtraction, "
        "is close enough to the complexity-weighted consensus probability, and exceeds the consensus contrast score."
    )

    def __init__(
        self,
        minimum_support: int = 4,
        confidence_z: float = 0.5,
        consensus_ratio: float = 0.75,
        alternative_ratio: float = 0.5,
        minimum_contrast: float = 0.08,
    ) -> None:
        # Retained as compatibility parameters for prior experiment calls. The
        # rule is now systematic once its probability safeguards are satisfied.
        _ = minimum_support, confidence_z, consensus_ratio
        self.alternative_ratio = alternative_ratio
        self.minimum_contrast = minimum_contrast
        self.name = "complexity-contrast exception override"
        self.context_decisions: dict[tuple[str, ...], dict[str, float | int | bool]] = {}
        self.global_advantage = 0.0
        self.fitted_parameters: dict[str, Any] = {}

    def _candidate(self, row: dict[str, Any]) -> dict[str, Any] | None:
        models = [model for model in row["models"] if model["prediction"] and model["distribution"]]
        model_count = len(models)
        consensus = row["soft_prediction"]
        if model_count < 3 or any(model["distribution"].get(consensus, 0.0) >= 0.6 for model in models):
            return None
        complexities = [int(model["complexity"]) for model in models]
        minimum_complexity = min(complexities)
        maximum_complexity = max(complexities)
        if minimum_complexity == maximum_complexity:
            return None
        complexity_range = maximum_complexity - minimum_complexity
        positive_weights = {
            model["name"]: (int(model["complexity"]) - minimum_complexity) / complexity_range
            for model in models
        }
        negative_weights = {
            model["name"]: (maximum_complexity - int(model["complexity"])) / complexity_range
            for model in models
        }

        def weighted_distribution(weights: dict[str, float]) -> dict[str, float]:
            activities = set().union(*(model["distribution"].keys() for model in models))
            total_weight = sum(weights.values())
            return {
                activity: sum(weights[model["name"]] * model["distribution"].get(activity, 0.0) for model in models)
                / total_weight
                for activity in activities
            }

        high_distribution = weighted_distribution(positive_weights)
        # Negative evidence is deliberately limited to each model's own top
        # activity. A low-complexity model should discount the broad rule it
        # proposes, not erase every low-probability alternative it retains.
        negative_total = sum(negative_weights.values())
        low_distribution: defaultdict[str, float] = defaultdict(float)
        for model in models:
            confidence = float(
                model.get(
                    "confidence",
                    model["ranked_predictions"][0]["probability"] if model.get("ranked_predictions") else 0.0,
                )
            )
            low_distribution[model["prediction"]] += (
                negative_weights[model["name"]] * confidence / negative_total
            )
        high_consensus_probability = high_distribution.get(consensus, 0.0)
        consensus_contrast = high_consensus_probability - low_distribution.get(consensus, 0.0)
        top_three = sorted(high_distribution, key=lambda activity: (-high_distribution[activity], activity))[:3]
        candidates = []
        required_top_three_support = max(2, math.ceil(model_count * 0.6))
        for activity in top_three:
            if activity == consensus:
                continue
            high_probability = high_distribution[activity]
            low_probability = low_distribution.get(activity, 0.0)
            contrast = high_probability - low_probability
            top_three_support = sum(
                activity in {item["activity"] for item in model.get("ranked_predictions", [])[:3]}
                for model in models
            )
            if (
                high_probability >= self.alternative_ratio * high_consensus_probability
                and contrast >= self.minimum_contrast
                and contrast > consensus_contrast
                and top_three_support >= required_top_three_support
            ):
                candidates.append((activity, high_probability, low_probability, contrast, top_three_support))
        if not candidates:
            return None
        activity, high_probability, low_probability, contrast, top_three_support = max(
            candidates,
            key=lambda item: (item[3], item[1], item[4], item[0]),
        )
        return {
            "prediction": activity,
            "positive_complexity_weights": positive_weights,
            "negative_complexity_weights": negative_weights,
            "high_probability": high_probability,
            "high_consensus_probability": high_consensus_probability,
            "low_probability": low_probability,
            "contrast": contrast,
            "consensus_contrast": consensus_contrast,
            "top_three_support": top_three_support,
            "model_count": model_count,
            "consensus_ratio": int(row["agreement_count"]) / model_count,
        }

    @staticmethod
    def _contexts(row: dict[str, Any], candidate: dict[str, Any]) -> list[tuple[str, ...]]:
        contrast = _value_bin(float(candidate["contrast"]), (0.08, 0.15, 0.25, 0.4))
        alternative_ratio = _value_bin(
            float(candidate["high_probability"]) / max(float(candidate["high_consensus_probability"]), 1e-9),
            (0.5, 0.65, 0.8, 0.95),
        )
        support = str(candidate["top_three_support"])
        consensus = _value_bin(float(candidate["consensus_ratio"]), (0.75, 0.85, 0.95))
        soft_margin = _value_bin(float(row["soft_margin"]), (0.01, 0.03, 0.07, 0.15))
        entropy = _value_bin(float(row["soft_normalized_entropy"]), (0.25, 0.5, 0.7, 0.85))
        return [
            ("exception", _sequence_stage(row), consensus, support, alternative_ratio, contrast, soft_margin, entropy),
            ("exception", consensus, support, alternative_ratio, contrast, soft_margin),
            ("exception", support, alternative_ratio, contrast),
            ("exception", contrast),
        ]

    def fit(self, rows: list[dict[str, Any]]) -> None:
        # This is a systematic distribution transformation once its observable
        # applicability condition is met; calibration is intentionally not used
        # to enable or suppress an individual event.
        self.context_decisions = {}
        self.fitted_parameters = {
            "calibrated": False,
            "applicability": "every model assigns the soft-vote top activity less than 0.5 probability",
            "complexity_weighting": "continuous positive high-complexity and negative low-complexity weights",
            "minimum_alternative_to_consensus_ratio": self.alternative_ratio,
            "minimum_contrast": self.minimum_contrast,
        }

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        candidate = self._candidate(row)
        active = candidate is not None
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": active,
            "candidate": candidate or {},
            "decision": {"systematic": active},
        }
        return ("complexity-contrast alternative", str(candidate["prediction"])) if active and candidate else (
            "soft voting",
            row["soft_prediction"],
        )


class DistributionShapeReliabilityRule(HierarchicalReliabilityRule):
    """Calibrate each model by entropy, margin, divergence, and availability."""

    family = "distribution shape"
    description = (
        "Calibrates constituent reliability from entropy, margin, soft divergence, agreement, and availability."
    )
    interpretation = "Distinguishes an informative confident contrarian from diffuse disagreement at a process branch."
    selection_policy = "Choose the model with the best calibrated reliability for the current distribution shape."

    def __init__(self, minimum_support: int = 5) -> None:
        super().__init__(minimum_support=minimum_support)
        self.name = f"distribution-shape reliability (support {minimum_support})"

    def _contexts(self, row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        entropy = _value_bin(float(model["normalized_entropy"]), (0.25, 0.5, 0.7, 0.85))
        margin = _value_bin(float(model["margin"]), (0.03, 0.1, 0.25, 0.5))
        divergence = _value_bin(float(model["soft_divergence"]), (0.01, 0.03, 0.08, 0.15))
        agrees = str(model["prediction"] == row["soft_prediction"])
        name = model["name"]
        return [
            (name, entropy, margin, divergence, agrees, str(row["empty_prediction_count"])),
            (name, entropy, margin, divergence, agrees),
            (name, entropy, margin, agrees),
            (name, divergence, agrees),
            (name, agrees),
            (name,),
        ]


def _pooled_distribution(row: dict[str, Any], mode: str) -> dict[str, float]:
    distributions = [model["distribution"] for model in row["models"]]
    # Keep equal-probability outcomes reproducible across interpreter processes.
    activities = sorted(set().union(*(distribution.keys() for distribution in distributions)))
    pooled: dict[str, float] = {}
    for activity in activities:
        values = sorted(distribution.get(activity, 0.0) for distribution in distributions)
        if mode == "median":
            middle = len(values) // 2
            pooled[activity] = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
        elif mode == "trimmed" and len(values) >= TRIMMED_POOL_MIN_MODELS:
            pooled[activity] = sum(values[1:-1]) / (len(values) - 2)
        elif mode == "product":
            pooled[activity] = math.exp(sum(math.log(max(value, 1e-9)) for value in values) / len(values))
        else:
            pooled[activity] = sum(values) / len(values)
    return pooled


class DistributionPoolRule(DecisionRule):
    """Predict from a robust median, trimmed mean, or product probability pool."""

    family = "distribution pool"
    description = "Merges all constituent activity probabilities with a symmetric robust pooling operator."
    interpretation = (
        "Tests whether corroborated probability mass is more useful than selecting one process abstraction."
    )
    selection_policy = "Merge every model distribution symmetrically; no constituent model is selected by name."

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.name = f"{mode} probability pool"

    def select(self, row: dict[str, Any]) -> str:  # noqa: ARG002
        return self.name

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        distribution = _pooled_distribution(row, self.mode)
        prediction = max(distribution, key=distribution.get, default="")  # type: ignore[arg-type]
        return self.name, prediction


def _rank_aggregation_activities(row: dict[str, Any]) -> list[str]:
    """Return every activity ranked by at least one constituent model."""
    return sorted(
        set(row.get("soft_distribution", {}))
        | {
            activity
            for model in row["models"]
            for activity in model.get("distribution", {})
        }
    )


def _rank_aggregation_winner(
    row: dict[str, Any],
    scores: dict[str, float],
) -> str:
    """Resolve an aggregate-score tie with soft probability, then stable lexical order."""
    if not scores:
        return row["soft_prediction"]
    return min(
        scores,
        key=lambda activity: (
            -scores[activity],
            -float(row.get("soft_distribution", {}).get(activity, 0.0)),
            activity != row["soft_prediction"],
            activity,
        ),
    )


def _pairwise_model_preferences(
    row: dict[str, Any],
    activities: list[str],
) -> dict[tuple[str, str], int]:
    """
    Count strict constituent-model preferences between every activity pair.

    Equal probabilities, including two absent activities, are treated as an
    abstention. This adapts the paper's linear ballots to the weak rankings
    naturally produced by probability distributions without inventing an order.
    """
    preferences: Counter[tuple[str, str]] = Counter()
    for model in row["models"]:
        distribution = model.get("distribution", {})
        for left_index, left in enumerate(activities):
            for right in activities[left_index + 1 :]:
                left_probability = float(distribution.get(left, 0.0))
                right_probability = float(distribution.get(right, 0.0))
                if left_probability > right_probability:
                    preferences[left, right] += 1
                elif right_probability > left_probability:
                    preferences[right, left] += 1
    return dict(preferences)


class BordaRankAggregationRule(DecisionRule):
    """Aggregate constituent probability rankings with a tie-aware Borda score."""

    name = "Borda rank aggregation"
    family = "social-choice rank aggregation"
    description = (
        "Treats constituent models as voters and activities as alternatives, then sums positional Borda scores."
    )
    interpretation = (
        "Tests whether consistently high activity ranks contain useful evidence that equal-weight probability "
        "averaging loses."
    )
    selection_policy = (
        "Give each activity one point per strictly lower-ranked alternative and half a point per tied alternative "
        "within every model; choose the largest total."
    )
    def __init__(self) -> None:
        self.fitted_parameters = {
            "calibrated": False,
            "source": "Brandt, Conitzer, and Endriss (2012), pp. 7-8 and 18",
            "weak_ranking_adaptation": "equal probabilities split positional credit",
            "tie_break": "soft-voting probability, then stable activity label",
        }

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        activities = _rank_aggregation_activities(row)
        scores = dict.fromkeys(activities, 0.0)
        for model in row["models"]:
            distribution = model.get("distribution", {})
            for activity in activities:
                probability = float(distribution.get(activity, 0.0))
                scores[activity] += sum(
                    1.0
                    if probability > float(distribution.get(other, 0.0))
                    else 0.5
                    if probability == float(distribution.get(other, 0.0))
                    else 0.0
                    for other in activities
                    if other != activity
                )
        prediction = _rank_aggregation_winner(row, scores)
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": prediction != row["soft_prediction"],
            "winner": prediction,
            "top_scores": sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:5],
        }
        return self.name, prediction


class CopelandRankAggregationRule(DecisionRule):
    """Choose the activity with the strongest pairwise-majority record."""

    name = "Copeland pairwise rank aggregation"
    family = "social-choice rank aggregation"
    description = (
        "Treats constituent models as voters and scores each activity by pairwise majority wins plus half-points "
        "for ties."
    )
    interpretation = (
        "Tests whether an activity supported across direct rank comparisons is more reliable than the soft-vote leader."
    )
    selection_policy = (
        "For every activity pair, count models assigning higher probability to each side; award one point for a "
        "pairwise win and half a point for a tie."
    )
    def __init__(self) -> None:
        self.fitted_parameters = {
            "calibrated": False,
            "source": "Brandt, Conitzer, and Endriss (2012), pp. 19-20",
            "weak_ranking_adaptation": "equal probabilities abstain in that pairwise contest",
            "tie_break": "soft-voting probability, then stable activity label",
        }

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        activities = _rank_aggregation_activities(row)
        preferences = _pairwise_model_preferences(row, activities)
        scores = dict.fromkeys(activities, 0.0)
        for left_index, left in enumerate(activities):
            for right in activities[left_index + 1 :]:
                left_support = preferences.get((left, right), 0)
                right_support = preferences.get((right, left), 0)
                if left_support > right_support:
                    scores[left] += 1.0
                elif right_support > left_support:
                    scores[right] += 1.0
                else:
                    scores[left] += 0.5
                    scores[right] += 0.5
        prediction = _rank_aggregation_winner(row, scores)
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": prediction != row["soft_prediction"],
            "winner": prediction,
            "top_scores": sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:5],
        }
        return self.name, prediction


class MaximinRankAggregationRule(DecisionRule):
    """Choose the activity with the best worst pairwise-majority margin."""

    name = "maximin pairwise rank aggregation"
    family = "social-choice rank aggregation"
    description = (
        "Treats constituent models as voters and maximizes each activity's worst pairwise support margin."
    )
    interpretation = (
        "Tests a conservative rank consensus: the selected activity is the one least exposed to a strong pairwise "
        "defeat."
    )
    selection_policy = (
        "For each activity, compute its vote margin against every rival and choose the activity with the largest "
        "worst-case margin."
    )
    def __init__(self) -> None:
        self.fitted_parameters = {
            "calibrated": False,
            "source": "Brandt, Conitzer, and Endriss (2012), p. 20",
            "weak_ranking_adaptation": "equal probabilities abstain in that pairwise contest",
            "tie_break": "soft-voting probability, then stable activity label",
        }

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        activities = _rank_aggregation_activities(row)
        preferences = _pairwise_model_preferences(row, activities)
        scores = {
            activity: min(
                (
                    preferences.get((activity, other), 0)
                    - preferences.get((other, activity), 0)
                    for other in activities
                    if other != activity
                ),
                default=0,
            )
            for activity in activities
        }
        prediction = _rank_aggregation_winner(row, scores)
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": prediction != row["soft_prediction"],
            "winner": prediction,
            "top_scores": sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:5],
        }
        return self.name, prediction


class CalibratedDistributionPoolRule(DecisionRule):
    """Choose a probability-pooling method by calibrated distribution regime."""

    family = "distribution pool"
    description = "Selects soft, median, trimmed, or product pooling from calibrated structural regimes."
    interpretation = "Shows whether different process regimes require averaging, outlier resistance, or corroboration."
    selection_policy = "Choose the best calibrated distribution-merging operator for the current structural regime."
    modes = ("soft", "median", "trimmed", "product")

    def __init__(self, minimum_support: int = 10) -> None:
        self.minimum_support = minimum_support
        self.name = f"calibrated probability pool (support {minimum_support})"
        self.mode_by_context: dict[tuple[str, ...], str] = {}

    @staticmethod
    def _prediction(row: dict[str, Any], mode: str) -> str:
        if mode == "soft":
            return row["soft_prediction"]
        distribution = _pooled_distribution(row, mode)
        return max(distribution, key=distribution.get, default="")  # type: ignore[arg-type]

    def fit(self, rows: list[dict[str, Any]]) -> None:
        grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for context in _distribution_contexts(row):
                grouped[context].append(row)
        self.mode_by_context = {}
        for context, context_rows in grouped.items():
            if len(context_rows) < self.minimum_support:
                continue
            correct_by_mode = {
                mode: sum(self._prediction(row, mode) == row["actual"] for row in context_rows) for mode in self.modes
            }
            best_mode = max(self.modes, key=lambda mode: (correct_by_mode[mode], -self.modes.index(mode)))
            self.mode_by_context[context] = (
                best_mode if correct_by_mode[best_mode] > correct_by_mode["soft"] else "soft"
            )

    def select(self, row: dict[str, Any]) -> str:  # noqa: ARG002
        return self.name

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        for context in _distribution_contexts(row):
            if context in self.mode_by_context:
                mode = self.mode_by_context[context]
                return f"{mode} probability pool", self._prediction(row, mode)
        return "soft voting", row["soft_prediction"]


class StateEvidenceTopologyRule(HierarchicalReliabilityRule):
    """
    Select a miner from state evidence and cross-model disagreement topology.

    Unlike a suffix or activity rule, the calibrated contexts remain meaningful
    after an arbitrary renaming of all activities. Exact state identifiers are
    used only as learned process states, followed by structural backoff levels.
    """

    family = "state evidence topology"
    description = (
        "Compares models through joint state identity, per-state occurrence counts, n-gram order maturity, "
        "relative confidence/entropy, and canonical agreement topology."
    )
    interpretation = (
        "A deep model winning in a well-visited joint state suggests a stable local subprocess; a general model "
        "winning while deep-state support collapses suggests branching or insufficient repeated context."
    )
    selection_policy = "Choose the model with the strongest calibrated state-evidence/topology reliability."

    def __init__(self, minimum_support: int = 5) -> None:
        super().__init__(minimum_support=minimum_support, prior_weight=6.0)
        self.name = f"state-evidence topology (support {minimum_support})"

    def _contexts(self, row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        models = row["models"]
        name = model["name"]
        topology = _prediction_partition(row)
        joint_states = tuple(candidate["state"] for candidate in models)
        visits = tuple(_visit_bin(int(candidate["state_visits"])) for candidate in models)
        role = (
            _relative_rank(models, model, "state_visits"),
            _relative_rank(models, model, "confidence"),
            _relative_rank(models, model, "normalized_entropy", reverse=False),
            _relative_rank(models, model, "soft_divergence", reverse=False),
            _soft_rank_of_model(row, model),
        )
        mature = str(int(row["position"]) >= int(model["complexity"]))
        return [
            (name, "joint-state", *joint_states, *topology),
            (name, "state-evidence", model["state"], _visit_bin(int(model["state_visits"])), *role),
            (name, "evidence-curve", *visits, *topology, *role, mature),
            (name, "relative-role", *role, mature, str(row["agreement_count"])),
            (name, "support-role", _visit_bin(int(model["state_visits"])), role[0], role[4], mature),
            (name,),
        ]


class EvidenceWeightedDistributionRule(DecisionRule):
    """Blend full model distributions using locally calibrated evidence weights."""

    family = "evidence-weighted distribution"
    description = (
        "Learns each model's smoothed logarithmic probability quality in activity-invariant structural regimes, "
        "then blends the full distributions with local evidence weights."
    )
    interpretation = (
        "The learned weights expose which process abstraction explains a regime: broad models dominate uncertain "
        "branch points, while longer n-grams dominate repeated, well-supported subprocesses."
    )
    selection_policy = "Weight every full distribution by locally calibrated probability quality and state evidence."

    def __init__(self, minimum_support: int = 12, prior_weight: float = 16.0, temperature: float = 0.6) -> None:
        self.minimum_support = minimum_support
        self.prior_weight = prior_weight
        self.temperature = temperature
        self.name = f"evidence-weighted distribution mixture (support {minimum_support})"
        self.global_quality: dict[str, tuple[float, int]] = {}
        self.context_quality: dict[tuple[tuple[str, ...], str], tuple[float, int]] = {}

    @staticmethod
    def _log_quality(row: dict[str, Any], model: dict[str, Any]) -> float:
        # A bounded log score rewards calibrated probability on the observed
        # transition, not merely an occasionally correct top-1 prediction.
        return math.log(max(float(model["distribution"].get(row["actual"], 0.0)), math.exp(-8.0)))

    def fit(self, rows: list[dict[str, Any]]) -> None:
        global_quality: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        context_quality: defaultdict[tuple[tuple[str, ...], str], list[float]] = defaultdict(lambda: [0.0, 0.0])
        for row in rows:
            contexts = _structural_regime_contexts(row)
            for model in row["models"]:
                quality = self._log_quality(row, model)
                global_quality[model["name"]][0] += quality
                global_quality[model["name"]][1] += 1
                for context in contexts:
                    context_quality[(context, model["name"])][0] += quality
                    context_quality[(context, model["name"])][1] += 1
        self.global_quality = {name: (values[0], int(values[1])) for name, values in global_quality.items()}
        self.context_quality = {key: (values[0], int(values[1])) for key, values in context_quality.items()}

    def _weights(self, row: dict[str, Any]) -> dict[str, float]:
        local_context = None
        for context in _structural_regime_contexts(row):
            if all(
                self.context_quality.get((context, model["name"]), (0.0, 0))[1] >= self.minimum_support
                for model in row["models"]
            ):
                local_context = context
                break
        scores = {}
        for model in row["models"]:
            name = model["name"]
            global_sum, global_total = self.global_quality.get(name, (0.0, 0))
            global_mean = global_sum / global_total if global_total else -8.0
            local_sum, local_total = self.context_quality.get((local_context, name), (0.0, 0))
            local_mean = (
                (local_sum + self.prior_weight * global_mean) / (local_total + self.prior_weight)
                if local_context is not None
                else global_mean
            )
            # State support is a modest evidence multiplier; calibration still
            # controls the dominant reliability term and can favor backoff.
            support = 1.0 + min(math.log1p(int(model["state_visits"])), 4.0) / 8.0
            scores[name] = local_mean + math.log(support)
        maximum = max(scores.values(), default=0.0)
        return {name: math.exp((score - maximum) / self.temperature) for name, score in scores.items()}

    def select(self, row: dict[str, Any]) -> str:  # noqa: ARG002
        return self.name

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        weights = self._weights(row)
        pooled: Counter[str] = Counter()
        for model in row["models"]:
            weight = weights.get(model["name"], 0.0)
            for activity, probability in model["distribution"].items():
                pooled[activity] += weight * float(probability)
        prediction = max(sorted(pooled), key=pooled.get) if pooled else row["soft_prediction"]  # type: ignore[arg-type]
        return self.name, prediction


class StructuralBranchingEnsembleRule(DecisionRule):
    """Gate between model selection and distribution-merging experts by regime."""

    family = "structural branching ensemble"
    description = (
        "Uses an internal sequence-level calibration holdout to branch between state-topology selection, "
        "evidence-weighted blending, robust probability pools, and soft voting."
    )
    interpretation = (
        "Its chosen branch characterizes the process regime: state selection marks a recurring subprocess, robust "
        "pooling marks noisy outlier models, and soft fallback marks unresolved or weakly supported branching."
    )
    selection_policy = "Choose the expert with validated local gain; retain soft voting when no branch is reliable."

    def __init__(self, minimum_support: int = 8, confidence_z: float = 0.75) -> None:
        self.minimum_support = minimum_support
        self.confidence_z = confidence_z
        self.name = f"structural branching ensemble (support {minimum_support})"
        self.experts: list[DecisionRule] = []
        self.expert_by_name: dict[str, DecisionRule] = {}
        self.branch_by_context: dict[tuple[str, ...], str] = {}

    @staticmethod
    def _new_experts() -> list[DecisionRule]:
        return [
            StateEvidenceTopologyRule(minimum_support=4),
            EvidenceWeightedDistributionRule(minimum_support=8),
            DistributionPoolRule("median"),
            DistributionPoolRule("trimmed"),
            DistributionPoolRule("product"),
        ]

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.experts = self._new_experts()
        self.expert_by_name = {expert.name: expert for expert in self.experts}
        if not rows:
            self.branch_by_context = {}
            return
        sequence_ids = list(dict.fromkeys(row["sequence_index"] for row in rows))
        validation_ids = set(sequence_ids[::4]) if len(sequence_ids) >= MIN_BRANCHING_SEQUENCES else set()
        fitting = [row for row in rows if row["sequence_index"] not in validation_ids] or rows
        validation = [row for row in rows if row["sequence_index"] in validation_ids] or rows[::4] or rows
        for expert in self.experts:
            expert.fit(fitting)

        grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in validation:
            for context in _structural_regime_contexts(row):
                grouped[context].append(row)
        self.branch_by_context = {}
        for context, context_rows in grouped.items():
            total = len(context_rows)
            if total < self.minimum_support:
                continue
            soft_correct = sum(row["soft_correct"] for row in context_rows)
            correct = {
                expert.name: sum(expert.choose(row)[1] == row["actual"] for row in context_rows)
                for expert in self.experts
            }
            best_name = max(correct, key=lambda name: (correct[name], -list(correct).index(name)))
            paired_effective = correct[best_name] + soft_correct
            uncertainty = self.confidence_z * math.sqrt(max(1, paired_effective))
            if correct[best_name] - soft_correct > uncertainty:
                self.branch_by_context[context] = best_name

        # Refit experts on every calibration event after the internal gate has
        # been chosen; the held-out test split remains completely untouched.
        for expert in self.experts:
            expert.fit(rows)

    def select(self, row: dict[str, Any]) -> str:
        for context in _structural_regime_contexts(row):
            if context in self.branch_by_context:
                return self.branch_by_context[context]
        return "soft voting"

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        expert_name = self.select(row)
        expert = self.expert_by_name.get(expert_name)
        return expert.choose(row) if expert is not None else ("soft voting", row["soft_prediction"])


class CalibratedCandidateRouterRule(DecisionRule):
    """
    Route to a ranked ensemble candidate when it has a supported local gain.

    The router deliberately has no knowledge of activity labels or model names.
    A candidate is described only by its *role* in the current ensemble: the
    soft rank, or the rank emitted by a constituent at a relative complexity
    position.  This lets the same rule work when the ensemble has a different
    number of N-grams, their names change, or a non-N-gram model is added.

    It is also stateful at inference time.  Once the label for an event is
    available, the case-local reliability of every candidate role is updated
    for subsequent events in that case.  The current label is never used to
    select the current prediction.
    """

    family = "calibrated candidate router"
    description = (
        "Routes among soft and constituent ranked candidates using paired, shrunk gains in state-support, "
        "uncertainty, and agreement regimes, with delayed within-case feedback."
    )
    interpretation = (
        "An override identifies a candidate role that is repeatedly superior to the soft leader in the current "
        "process regime; the within-case term captures a temporary local subprocess."
    )
    selection_policy = (
        "Keep soft voting unless an activity-invariant candidate role has enough calibration support and a positive "
        "lower confidence bound for paired gain over soft voting."
    )

    def __init__(
        self,
        minimum_support: int = 12,
        prior_weight: float = 12.0,
        confidence_z: float = 0.5,
        online_weight: float = 3.0,
        decay: float = 0.75,
    ) -> None:
        self.minimum_support = minimum_support
        self.prior_weight = prior_weight
        self.confidence_z = confidence_z
        self.online_weight = online_weight
        self.decay = decay
        self.name = f"calibrated candidate router (support {minimum_support})"
        self.global_gains: dict[str, tuple[float, float, int]] = {}
        self.context_gains: list[dict[tuple[str, ...], tuple[float, float, int]]] = []
        self.case_scores: dict[str, tuple[float, float]] = {}
        self.active_sequence = ""

    @staticmethod
    def _complexity_positions(row: dict[str, Any]) -> dict[str, int]:
        ordered = sorted(row["models"], key=lambda model: (int(model["complexity"]), int(model["index"])))
        return {model["name"]: position for position, model in enumerate(ordered)}

    def _candidates(self, row: dict[str, Any]) -> list[tuple[str, str, dict[str, Any] | None, int]]:
        """Return role, activity, source model, and source probability rank."""
        candidates: list[tuple[str, str, dict[str, Any] | None, int]] = []
        for rank, item in enumerate(row.get("soft_ranked_predictions", [])[:3], start=1):
            candidates.append((f"soft-rank-{rank}", str(item["activity"]), None, rank))
        positions = self._complexity_positions(row)
        for model in row["models"]:
            position = positions[model["name"]]
            for rank, item in enumerate(model.get("ranked_predictions", [])[:2], start=1):
                candidates.append((f"model-rank-{rank}-complexity-{position}", str(item["activity"]), model, rank))
        # A deterministic fallback keeps the rule usable for an empty
        # distribution, while normal events always include soft-rank-1.
        return candidates or [("soft-rank-1", row["soft_prediction"], None, 1)]

    def _contexts(
        self,
        row: dict[str, Any],
        role: str,
        activity: str,
        model: dict[str, Any] | None,
        rank: int,
    ) -> list[tuple[str, ...]]:
        prediction_support = sum(candidate["prediction"] == activity for candidate in row["models"])
        topology = _prediction_partition(row)
        soft_shape = (
            _value_bin(float(row["soft_normalized_entropy"]), (0.35, 0.6, 0.8)),
            _value_bin(float(row["soft_margin"]), (0.03, 0.1, 0.25)),
            str(row["agreement_count"]),
            str(prediction_support),
        )
        if model is None:
            source = ("soft", str(rank))
        else:
            source = (
                "model",
                str(rank),
                _visit_bin(int(model["state_visits"])),
                str(_relative_rank(row["models"], model, "confidence")),
                str(_soft_rank_of_model(row, model)),
            )
        return [
            (role, *source, *soft_shape, *topology),
            (role, *source, *soft_shape),
            (role, *source, str(row["agreement_count"]), str(prediction_support)),
            (role, *source),
            (role,),
        ]

    def fit(self, rows: list[dict[str, Any]]) -> None:
        levels = 5
        totals: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        contexts: list[defaultdict[tuple[str, ...], list[float]]] = [
            defaultdict(lambda: [0.0, 0.0, 0.0]) for _ in range(levels)
        ]
        for row in rows:
            soft_outcome = int(row["soft_correct"])
            for role, activity, model, rank in self._candidates(row):
                gain = int(activity == row["actual"]) - soft_outcome
                values = totals[role]
                values[0] += gain
                values[1] += gain * gain
                values[2] += 1
                for level, context in enumerate(self._contexts(row, role, activity, model, rank)):
                    values = contexts[level][context]
                    values[0] += gain
                    values[1] += gain * gain
                    values[2] += 1
        self.global_gains = {role: (values[0], values[1], int(values[2])) for role, values in totals.items()}
        self.context_gains = [
            {context: (values[0], values[1], int(values[2])) for context, values in level.items()}
            for level in contexts
        ]
        self.case_scores = {}
        self.active_sequence = ""
        self.fitted_parameters = {
            "calibrated": True,
            "candidate_roles": "soft ranks 1-3 and constituent ranks 1-2 indexed by relative complexity",
            "minimum_support": self.minimum_support,
            "confidence_z": self.confidence_z,
            "online_feedback": "decayed candidate-role gain after the preceding labeled event in the same case",
        }

    @staticmethod
    def _mean_variance(values: tuple[float, float, int]) -> tuple[float, float, int]:
        total, squares, count = values
        if not count:
            return 0.0, 0.0, 0
        mean = total / count
        variance = max(0.0, (squares - count * mean * mean) / max(1, count - 1))
        return mean, variance, count

    def _gain_score(
        self,
        row: dict[str, Any],
        role: str,
        activity: str,
        model: dict[str, Any] | None,
        rank: int,
    ) -> tuple[float, float, int]:
        global_values = self.global_gains.get(role, (0.0, 0.0, 0))
        global_mean, _, global_count = self._mean_variance(global_values)
        for level, context in enumerate(self._contexts(row, role, activity, model, rank)):
            values = self.context_gains[level].get(context, (0.0, 0.0, 0))
            mean, variance, count = self._mean_variance(values)
            if count >= self.minimum_support:
                shrunk = (count * mean + self.prior_weight * global_mean) / (count + self.prior_weight)
                uncertainty = self.confidence_z * math.sqrt(variance / max(1, count))
                return shrunk - uncertainty, shrunk, count
        # Global roles are a valid final backoff, but only if they have the
        # same support requirement as every structural context.
        if global_count >= self.minimum_support:
            _, variance, _ = self._mean_variance(global_values)
            return global_mean - self.confidence_z * math.sqrt(variance / global_count), global_mean, global_count
        return float("-inf"), 0.0, 0

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        if row["sequence_id"] != self.active_sequence:
            self.active_sequence = row["sequence_id"]
            self.case_scores = {}
        best_role, best_activity, best_score = "soft-rank-1", row["soft_prediction"], 0.0
        for role, activity, model, rank in self._candidates(row):
            lower_bound, _mean, _support = self._gain_score(row, role, activity, model, rank)
            online_gain, online_count = self.case_scores.get(role, (0.0, 0.0))
            online_mean = online_gain / online_count if online_count else 0.0
            # Online evidence only refines an already calibrated positive
            # route; it cannot create an unvalidated override by itself.
            score = lower_bound + self.online_weight * online_mean / (self.online_weight + online_count)
            if activity != row["soft_prediction"] and lower_bound > 0 and score > best_score:
                best_role, best_activity, best_score = role, activity, score
        if best_activity != row["soft_prediction"]:
            return best_role, best_activity
        return "soft voting", row["soft_prediction"]

    def observe(self, row: dict[str, Any], selected_model: str) -> None:  # noqa: ARG002
        soft_outcome = int(row["soft_correct"])
        for role, activity, _model, _rank in self._candidates(row):
            gain, count = self.case_scores.get(role, (0.0, 0.0))
            self.case_scores[role] = (
                self.decay * gain + int(activity == row["actual"]) - soft_outcome,
                self.decay * count + 1.0,
            )


class CompleteMissStateRecoveryRule(DecisionRule):
    """Recover after a complete ensemble miss using state-calibrated candidates."""

    family = "complete-miss recovery"
    description = (
        "After the preceding event defeated every constituent, compares the minimum-complexity and adaptive "
        "candidates using their state-specific paired gain over soft voting."
    )
    interpretation = (
        "A complete miss marks a possible regime change.  A recovery is allowed only when one of the ensemble's "
        "generalist or online-adaptive views has repeatedly handled the following state better."
    )
    selection_policy = (
        "After a previous complete miss, replace soft voting only with a candidate whose supported state-level "
        "paired lower confidence bound is positive; otherwise keep soft voting."
    )

    def __init__(self, minimum_support: int = 8, confidence_z: float = 0.5) -> None:
        self.minimum_support = minimum_support
        self.confidence_z = confidence_z
        self.name = f"complete-miss state recovery (support {minimum_support})"
        self.global_gains: dict[str, tuple[float, float, int]] = {}
        self.context_gains: list[dict[tuple[str, ...], tuple[float, float, int]]] = []

    @staticmethod
    def _complete_previous_miss(row: dict[str, Any]) -> bool:
        return row.get("previous_soft_correct") is False and not row.get("previous_correct_models", [])

    @staticmethod
    def _minimum_complexity_model(row: dict[str, Any]) -> dict[str, Any] | None:
        return min(row["models"], key=lambda model: (int(model["complexity"]), int(model["index"])), default=None)

    def _candidates(self, row: dict[str, Any]) -> list[tuple[str, str, dict[str, Any] | None]]:
        minimum = self._minimum_complexity_model(row)
        adaptive_index = row.get("adaptive_model_index")
        adaptive_model = (
            next((model for model in row["models"] if model["index"] == adaptive_index), None)
            if adaptive_index is not None
            else None
        )
        candidates = [
            ("minimum-complexity", minimum["prediction"], minimum) if minimum is not None else None,
            ("adaptive", row.get("adaptive_prediction", ""), adaptive_model),
        ]
        return [candidate for candidate in candidates if candidate is not None and candidate[1]]

    def _contexts(
        self,
        row: dict[str, Any],
        role: str,
        model: dict[str, Any] | None,
    ) -> list[tuple[str, ...]]:
        source_state = model["state"] if model is not None else ""
        support = _visit_bin(int(model["state_visits"])) if model is not None else "adaptive"
        shape = (
            str(row["agreement_count"]),
            str(row["empty_prediction_count"]),
            _value_bin(float(row["soft_margin"]), (0.03, 0.1, 0.25, 0.5)),
            *_prediction_partition(row),
        )
        return [
            (role, source_state, support, *shape),
            (role, source_state),
            (role, support, *shape),
            (role, *shape[:3]),
            (role,),
        ]

    @staticmethod
    def _mean_variance(values: tuple[float, float, int]) -> tuple[float, float, int]:
        total, squares, count = values
        if not count:
            return 0.0, 0.0, 0
        mean = total / count
        variance = max(0.0, (squares - count * mean * mean) / max(1, count - 1))
        return mean, variance, count

    def fit(self, rows: list[dict[str, Any]]) -> None:
        totals: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        contexts: list[defaultdict[tuple[str, ...], list[float]]] = [
            defaultdict(lambda: [0.0, 0.0, 0.0]) for _ in range(5)
        ]
        for row in rows:
            if not self._complete_previous_miss(row):
                continue
            for role, prediction, model in self._candidates(row):
                gain = int(prediction == row["actual"]) - int(row["soft_correct"])
                totals[role][0] += gain
                totals[role][1] += gain * gain
                totals[role][2] += 1
                for level, context in enumerate(self._contexts(row, role, model)):
                    contexts[level][context][0] += gain
                    contexts[level][context][1] += gain * gain
                    contexts[level][context][2] += 1
        self.global_gains = {role: (value[0], value[1], int(value[2])) for role, value in totals.items()}
        self.context_gains = [
            {context: (value[0], value[1], int(value[2])) for context, value in level.items()}
            for level in contexts
        ]
        self.fitted_parameters = {
            "calibrated": True,
            "trigger": "previous soft error with no correct constituent",
            "candidates": "minimum-complexity and adaptive-selected constituent, by relative role",
            "minimum_support": self.minimum_support,
            "confidence_z": self.confidence_z,
        }

    def _lower_bound(self, row: dict[str, Any], role: str, model: dict[str, Any] | None) -> tuple[float, int]:
        for level, context in enumerate(self._contexts(row, role, model)):
            values = self.context_gains[level].get(context, (0.0, 0.0, 0))
            mean, variance, count = self._mean_variance(values)
            if count >= self.minimum_support:
                return mean - self.confidence_z * math.sqrt(variance / count), count
        values = self.global_gains.get(role, (0.0, 0.0, 0))
        mean, variance, count = self._mean_variance(values)
        if count >= self.minimum_support:
            return mean - self.confidence_z * math.sqrt(variance / count), count
        return float("-inf"), 0

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        if not self._complete_previous_miss(row):
            return "soft voting", row["soft_prediction"]
        best = (0.0, 0, "soft voting", row["soft_prediction"])
        diagnostics = []
        for role, prediction, model in self._candidates(row):
            lower_bound, support = self._lower_bound(row, role, model)
            diagnostics.append({"role": role, "prediction": prediction, "lower_bound": lower_bound, "support": support})
            if prediction != row["soft_prediction"] and lower_bound > best[0]:
                best = lower_bound, support, role, prediction
        row.setdefault("rule_diagnostics", {})[self.name] = {
            "active": best[2] != "soft voting",
            "candidates": diagnostics,
            "previous_complete_miss": True,
        }
        return best[2], best[3]


def _new_advanced_hypotheses() -> list[DecisionRule]:
    """
    Return activity-label-invariant higher-capacity selectors.

    These implementations are retained for explicit archived experiments.
    """
    return [
        HierarchicalReliabilityRule(),
        ConsensusHierarchicalRule(),
        DisagreementProfileRule(),
        StateEvidenceTopologyRule(),
        EvidenceWeightedDistributionRule(),
        StructuralBranchingEnsembleRule(),
        CalibratedDistributionPoolRule(minimum_support=5),
        NearestCalibrationBehaviorRule(),
        StackedRulePortfolio(),
        DelayedFeedbackAdaptiveRule(),
    ]


def default_hypotheses() -> list[DecisionRule]:
    """Return the compact, interpretable deployed rule set; N-gram rules remain optional experiments."""
    return [
        # These complementary rules are deliberately kept small and explainable:
        # Distribution evidence and a calibrated alternative to the soft-vote
        # leader complement the single short-lived Bag recovery rule below.
        EvidenceWeightedDistributionRule(minimum_support=12),
        ConfidenceStateReliabilityRule(minimum_support=5),
        DelayedFeedbackAdaptiveRule(minimum_support=2, decay=0.94),
        CompleteMissStateRecoveryRule(minimum_support=8),
        CalibratedTransientGeneralistPoolRule(),
        CalibratedSoftRankRule(rank=2),
        CalibratedLoneDissenterRule(),
        CalibratedLoneDissenterSecondRankRule(),
        CalibratedComplexityContrastExceptionRule(),
        BordaRankAggregationRule(),
        CopelandRankAggregationRule(),
        MaximinRankAggregationRule(),
        TransientBagFavoritismRule(per_competing_model_boost=3.0),
    ]


def archived_hypotheses() -> list[DecisionRule]:
    """Return retained exploratory rules excluded from the normal benchmark."""
    rules: list[DecisionRule] = [
        HighestConfidenceRule(),
        GlobalAccuracyRule(),
        TransientGeneralizationBoostRule(trigger_mode="all soft errors"),
        TransientGeneralizationBoostRule(trigger_mode="generalist was correct"),
        CalibratedGeneralizationRecoveryRule("age gate"),
        CalibratedGeneralizationRecoveryRule("structural gate"),
        CalibratedGeneralizationRecoveryRule("distribution blend"),
        *_new_advanced_hypotheses(),
    ]
    rules.extend(
        GroupedAccuracyRule(
            name=f"position bucket {width} (support 5)",
            family="position",
            feature=lambda row, width=width: str(row["position"] // width),
            minimum_support=5,
        )
        for width in (2, 5, 10)
    )
    rules.extend(StateAccuracyRule(support) for support in (3, 5, 10))
    rules.extend(
        AgreementRule(threshold, fallback)
        for threshold in (2, 3)
        for fallback in ("confidence", "accuracy")
    )
    rules.extend(
        [
            CalibratedCandidateRouterRule(minimum_support=6),
            CalibratedSoftRankRule(3),
            CalibratedModelRankRule(2),
            CalibratedModelRankRule(3, minimum_support=5),
            DistributionShapeReliabilityRule(minimum_support=8),
            DistributionPoolRule("median"),
            DistributionPoolRule("trimmed"),
            DistributionPoolRule("product"),
            PreviousErrorCorrectSetRule(),
        ]
    )
    deployed_names = {rule.name for rule in default_hypotheses()}
    return [rule for rule in rules if rule.name not in deployed_names]


class VotingInvestigator:
    """Train constituent models and produce event-level oracle-gap diagnostics."""

    def __init__(self, specs: list[ModelSpec] | None = None, config: dict[str, Any] | None = None) -> None:
        self.specs = specs or default_model_specs()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.models = [spec.factory() for spec in self.specs]
        for model in self.models:
            model.config = self.config
            model.propagate_config()
        self.soft_voting = SoftVoting(models=self.models, config=self.config)

    def train(self, sequences: list[list[Event]]) -> None:
        """Train all constituent models once on the same event stream."""
        for event in interleave_sequences(sequences, random_index=False):
            for model in self.models:
                model.update(event)

    def diagnose(self, sequences: list[list[Event]], *, split: str) -> list[dict[str, Any]]:
        """Collect predictions, agreement, state, pattern, and correctness per event."""
        rows: list[dict[str, Any]] = []
        adaptive_voting = AdaptiveVoting(models=self.models, select_best="acc", config=self.config)
        for sequence_index, sequence in enumerate(sequences):
            if not sequence:
                continue
            states: list[ComposedState | None] = [model.initial_state for model in self.models]
            prefix: list[str] = []
            sequence_id = _display(sequence[0]["case_id"])
            previous_soft_correct: bool | None = None
            previous_correct_models: list[str] = []
            previous_empty_prediction_count = 0
            for position, event in enumerate(sequence):
                actual = _display(event["activity"])
                metrics_list = [model.state_metrics(state) for model, state in zip(self.models, states, strict=True)]
                model_rows = self._model_rows(metrics_list, actual)
                predictions = [model["prediction"] for model in model_rows if model["prediction"]]
                counts = Counter(predictions)
                consensus = counts.most_common(1)[0][0] if counts else ""
                agreement_count = counts[consensus] if consensus else 0

                soft_probs = self.soft_voting.voting_probs([metrics["probs"] for metrics in metrics_list])
                soft_distribution = _normalized_distribution(soft_probs)
                soft_prediction = self._distribution_prediction(soft_probs)
                soft_ranked = sorted(soft_distribution.items(), key=lambda item: item[1], reverse=True)
                if soft_prediction and any(activity == soft_prediction for activity, _ in soft_ranked):
                    soft_ranked.sort(key=lambda item: (item[0] != soft_prediction, -item[1]))
                soft_stats = _distribution_statistics(soft_distribution)
                adaptive_metrics = adaptive_voting.state_metrics(tuple(states))
                adaptive_prediction = self._distribution_prediction(adaptive_metrics["probs"])
                adaptive_model_index = adaptive_voting.last_selected_model_index
                for model_row in model_rows:
                    model_row["soft_divergence"] = _jensen_shannon_divergence(
                        model_row["distribution"], soft_distribution
                    )
                    model_row["probability_on_soft_prediction"] = model_row["distribution"].get(soft_prediction, 0.0)
                correct_models = [model["name"] for model in model_rows if model["correct"]]
                oracle_prediction = actual if correct_models else soft_prediction
                empty_prediction_count = sum(not model["prediction"] for model in model_rows)
                confidences = [float(model["confidence"]) for model in model_rows]

                padded_prefix = [*prefix]
                row = {
                    "split": split,
                    "sequence_id": sequence_id,
                    "sequence_index": sequence_index,
                    "sequence_length": len(sequence),
                    "position": position,
                    "position_1based": position + 1,
                    "relative_position": (position + 1) / len(sequence),
                    "prefix": padded_prefix,
                    "prefix_text": " → ".join(padded_prefix),
                    "suffix_1": " → ".join(padded_prefix[-1:]),
                    "suffix_2": " → ".join(padded_prefix[-2:]),
                    "suffix_3": " → ".join(padded_prefix[-3:]),
                    "actual": actual,
                    "models": model_rows,
                    "correct_models": correct_models,
                    "correct_model_count": len(correct_models),
                    "consensus_prediction": consensus,
                    "agreement_count": agreement_count,
                    "distinct_prediction_count": len(counts),
                    "empty_prediction_count": empty_prediction_count,
                    "previous_soft_correct": previous_soft_correct,
                    "previous_actual": padded_prefix[-1] if padded_prefix else "",
                    "previous_correct_models": previous_correct_models,
                    "previous_wrong_model_count": len(model_rows) - len(previous_correct_models)
                    if previous_soft_correct is not None
                    else 0,
                    "previous_empty_prediction_count": previous_empty_prediction_count,
                    "soft_distribution": soft_distribution,
                    "soft_ranked_predictions": [
                        {"activity": activity, "probability": probability} for activity, probability in soft_ranked
                    ],
                    "soft_entropy": soft_stats["entropy"],
                    "soft_normalized_entropy": soft_stats["normalized_entropy"],
                    "soft_margin": soft_stats["margin"],
                    "soft_support": soft_stats["support"],
                    "soft_top3_mass": soft_stats["top3_mass"],
                    "model_confidence_mean": sum(confidences) / len(confidences) if confidences else 0.0,
                    "model_confidence_spread": max(confidences) - min(confidences) if confidences else 0.0,
                    "model_entropy_mean": sum(float(model["normalized_entropy"]) for model in model_rows)
                    / len(model_rows),
                    "soft_prediction": soft_prediction,
                    "soft_correct": soft_prediction == actual,
                    "adaptive_prediction": adaptive_prediction,
                    "adaptive_correct": adaptive_prediction == actual,
                    "adaptive_model_index": adaptive_model_index,
                    "adaptive_model": (
                        model_rows[adaptive_model_index]["name"]
                        if adaptive_model_index is not None and adaptive_model_index < len(model_rows)
                        else ""
                    ),
                    "oracle_prediction": oracle_prediction,
                    "oracle_correct": oracle_prediction == actual,
                    "oracle_model": correct_models[0] if correct_models else "",
                    "oracle_gap": oracle_prediction == actual and soft_prediction != actual,
                }
                rows.append(row)
                adaptive_voting.total_predictions += 1
                for index, model_row in enumerate(model_rows):
                    if model_row["correct"]:
                        adaptive_voting.correct_predictions[index] += 1
                previous_soft_correct = soft_prediction == actual
                previous_correct_models = correct_models
                previous_empty_prediction_count = empty_prediction_count
                prefix.append(actual)
                states = [
                    model.next_state(state, event["activity"]) for model, state in zip(self.models, states, strict=True)
                ]
        return rows

    def _model_rows(self, metrics_list: list[Metrics], actual: str) -> list[dict[str, Any]]:
        model_rows = []
        for index, (spec, model, metrics) in enumerate(zip(self.specs, self.models, metrics_list, strict=True)):
            prediction = metrics_prediction(metrics, config=self.config)
            predicted = _display(prediction["activity"]) if prediction is not None else ""
            state = metrics["state_id"]
            state_info = model.get_state_info(state) if state is not None else None
            visits = state_info.get("total_visits", 0) if isinstance(state_info, dict) else 0
            access_string = state_info.get("access_string", state) if isinstance(state_info, dict) else state
            per_state_stats = model.stats.get("per_state_stats", {}).get(state)
            state_total_predictions = int(getattr(per_state_stats, "total_predictions", 0))
            state_correct_predictions = int(getattr(per_state_stats, "correct_predictions", 0))
            state_accuracy = state_correct_predictions / state_total_predictions if state_total_predictions else None
            distribution = _normalized_distribution(metrics["probs"])
            distribution_stats = _distribution_statistics(distribution)
            ranked_predictions = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
            model_rows.append(
                {
                    "index": index,
                    "name": spec.name,
                    "complexity": spec.complexity,
                    "model_type": spec.model_type,
                    "window_size": spec.window_size,
                    "state": _display(state),
                    "state_access_string": _display(access_string),
                    "state_visits": int(visits),
                    "state_accuracy": state_accuracy,
                    "state_correct_predictions": state_correct_predictions,
                    "state_total_predictions": state_total_predictions,
                    "prediction": predicted,
                    "confidence": float(prediction.get("probability", 0.0)) if prediction is not None else 0.0,
                    "distribution": distribution,
                    "ranked_predictions": [
                        {"activity": activity, "probability": probability}
                        for activity, probability in ranked_predictions
                    ],
                    **distribution_stats,
                    "correct": predicted == actual,
                }
            )
        return model_rows

    def _ensemble_prediction(self, ensemble: SoftVoting, metrics_list: list[Metrics]) -> str:
        probs = ensemble.voting_probs([metrics["probs"] for metrics in metrics_list])
        return self._distribution_prediction(probs)

    def _distribution_prediction(self, probs: dict[Any, float]) -> str:
        prediction = metrics_prediction(
            Metrics(state_id=None, probs=probs, predicted_delays={}),
            config=self.config,
        )
        return _display(prediction["activity"]) if prediction is not None else ""


def evaluate_hypotheses(
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    rules: list[DecisionRule] | None = None,
) -> list[dict[str, Any]]:
    """Fit rules without test leakage and attach their predictions to test events."""
    selected_rules = rules or default_hypotheses()
    summaries: list[dict[str, Any]] = []
    for row in test_rows:
        row["rule_predictions"] = {}
        row["rule_models"] = {}
        row["rule_diagnostics"] = {}

    for rule in selected_rules:
        rule.fit(calibration_rows)
        # This is deliberately an in-sample score: it shows how well the rule
        # explains the data used to fit its selector.  Re-fit below before the
        # test pass so feedback-consuming rules start the held-out evaluation
        # with no state carried over from this diagnostic pass.
        calibration_correct = 0
        for row in calibration_rows:
            model_name, prediction = rule.choose(row)
            calibration_correct += int(prediction == row["actual"])
            rule.observe(row, model_name)
        rule.fit(calibration_rows)
        correct = 0
        selected_counts: Counter[str] = Counter()
        for row in test_rows:
            model_name, prediction = rule.choose(row)
            row["rule_predictions"][rule.name] = prediction
            row["rule_models"][rule.name] = model_name
            correct += int(prediction == row["actual"])
            selected_counts[model_name] += 1
            # Delayed-feedback rules may update only after this event has been scored.
            rule.observe(row, model_name)
        summaries.append(
            {
                "name": rule.name,
                "family": rule.family,
                "description": rule.description,
                "interpretation": rule.interpretation,
                "selection_policy": rule.selection_policy,
                "parameters": getattr(rule, "fitted_parameters", {}),
                "accuracy": correct / len(test_rows) if test_rows else 0.0,
                "correct": correct,
                "total": len(test_rows),
                "calibration_accuracy": (
                    calibration_correct / len(calibration_rows) if calibration_rows else None
                ),
                "calibration_correct": calibration_correct,
                "calibration_total": len(calibration_rows),
                "selected_models": dict(selected_counts),
            }
        )
    return sorted(summaries, key=lambda item: item["accuracy"], reverse=True)


def _model_prediction(row: dict[str, Any], model_name: str) -> str:
    """Return one constituent prediction from an event diagnostic."""
    selected = next((model for model in row["models"] if model["name"] == model_name), None)
    return selected["prediction"] if selected is not None else ""


def _override_impact(
    rows: list[dict[str, Any]],
    predictions: list[str],
    *,
    name: str,
    family: str,
) -> dict[str, Any]:
    """Measure the effect of replacing soft voting with candidate predictions."""
    total = len(rows)
    soft_errors = sum(not row["soft_correct"] for row in rows)
    triggered = sum(prediction != row["soft_prediction"] for row, prediction in zip(rows, predictions, strict=True))
    recoveries = sum(
        not row["soft_correct"] and prediction == row["actual"]
        for row, prediction in zip(rows, predictions, strict=True)
    )
    harms = sum(
        row["soft_correct"] and prediction != row["actual"] for row, prediction in zip(rows, predictions, strict=True)
    )
    unchanged_errors = sum(
        not row["soft_correct"] and prediction != row["actual"]
        for row, prediction in zip(rows, predictions, strict=True)
    )
    candidate_correct = sum(prediction == row["actual"] for row, prediction in zip(rows, predictions, strict=True))
    decisive = recoveries + harms
    return {
        "name": name,
        "family": family,
        "triggered": triggered,
        "coverage": triggered / total if total else 0.0,
        "soft_errors": soft_errors,
        "recoveries": recoveries,
        "harms": harms,
        "unchanged_errors": unchanged_errors,
        "net_correct": recoveries - harms,
        "net_accuracy_delta": (recoveries - harms) / total if total else 0.0,
        "conditional_delta": (recoveries - harms) / triggered if triggered else 0.0,
        "soft_error_recall": recoveries / soft_errors if soft_errors else 0.0,
        "override_precision": recoveries / triggered if triggered else 0.0,
        "decisive_precision": recoveries / decisive if decisive else 0.0,
        "resulting_accuracy": candidate_correct / total if total else 0.0,
        "total": total,
    }


@cache
def _rule_family(rule_name: str) -> str:
    """Infer a persisted rule family from its stable display name."""
    prefixes = {
        "agreement": "agreement",
        "suffix": "pattern",
        "position": "position",
        "per-state": "state accuracy",
        "hierarchical": "hierarchical",
        "consensus hierarchy": "hierarchical",
        "confidence/state": "confidence-state",
        "disagreement profile": "disagreement profile",
        "state-evidence topology": "state evidence topology",
        "evidence-weighted distribution": "evidence-weighted distribution",
        "structural branching": "structural branching ensemble",
        "prediction/state": "prediction outcome",
        "prefix/state": "prefix",
        "calibrated decision": "decision list",
        "composite decision": "decision list",
        "nearest calibration": "nearest behavior",
        "stacked rule": "stacked portfolio",
        "delayed-feedback": "adaptive",
        "complete-miss": "complete-miss recovery",
        "calibrated soft rank": "soft rank",
        "calibrated model rank": "model rank",
        "previous-outcome": "previous outcome",
        "distribution-shape": "distribution shape",
        "median probability": "distribution pool",
        "trimmed probability": "distribution pool",
        "product probability": "distribution pool",
        "calibrated probability": "distribution pool",
        "Borda rank": "social-choice rank aggregation",
        "Copeland pairwise": "social-choice rank aggregation",
        "maximin pairwise": "social-choice rank aggregation",
        "calibrated transient generalist": "calibrated recovery",
        "confusion residual": "residual activity",
        "run-cycle residual": "residual activity",
        "state residual": "residual activity",
        "rank residual": "residual activity",
        "online within-case motif": "adaptive residual",
        "hashed rank router": "learned rank router",
        "ngram correctness-streak": "adaptive recovery",
        "transient recovery calibrated": "calibrated recovery",
        "transient generalization boost": "uncalibrated recovery",
    }
    for prefix, family in prefixes.items():
        if rule_name.startswith(prefix):
            return family
    if "confidence" in rule_name:
        return "confidence"
    return "accuracy"


def analyze_rule_impacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank calibrated decision rules by their held-out impact relative to soft voting."""
    if not rows:
        return []
    rule_names = list(rows[0].get("rule_predictions", {}))
    impacts = []
    for rule_name in rule_names:
        family = _rule_family(rule_name)
        impacts.append(
            _override_impact(
                rows,
                [row["rule_predictions"].get(rule_name, "") for row in rows],
                name=rule_name,
                family=family,
            )
        )
    return sorted(impacts, key=lambda item: (item["net_accuracy_delta"], item["recoveries"]), reverse=True)


def _condition_features(row: dict[str, Any]) -> dict[str, str]:
    """Extract activity-invariant features used to discover conditional rules."""
    model_count = len(row["models"])
    agreement = int(row["agreement_count"])
    if agreement <= model_count / 2:
        consensus_strength = "low/no majority"
    elif agreement < model_count:
        consensus_strength = "majority"
    else:
        consensus_strength = "unanimous"
    relative_position = float(row["relative_position"])
    stage = "early" if relative_position <= 1 / 3 else "middle" if relative_position <= 2 / 3 else "late"
    features = {
        "consensus strength": consensus_strength,
        "agreement count": str(agreement),
        "prediction diversity": str(row["distinct_prediction_count"]),
        "sequence stage": stage,
        "position bucket 2": str(int(row["position"]) // 2),
        "position bucket 5": str(int(row["position"]) // 5),
        "prediction topology": "-".join(_prediction_partition(row)),
        "state evidence profile": " | ".join(_visit_bin(int(model["state_visits"])) for model in row["models"]),
        "n-gram maturity profile": " | ".join(
            str(int(row["position"]) >= int(model["complexity"])) for model in row["models"]
        ),
    }
    if "soft_margin" in row:
        features.update(
            {
                "previous soft correctness": "unknown"
                if row.get("previous_soft_correct") is None
                else str(row["previous_soft_correct"]),
                "previous wrong model count": str(row.get("previous_wrong_model_count", 0)),
                "previous empty prediction count": str(row.get("previous_empty_prediction_count", 0)),
                "current empty prediction count": str(row.get("empty_prediction_count", 0)),
                "soft margin bin": _value_bin(float(row["soft_margin"]), (0.01, 0.03, 0.07, 0.15)),
                "soft entropy bin": _value_bin(float(row["soft_normalized_entropy"]), (0.25, 0.5, 0.7, 0.85)),
                "model confidence spread bin": _value_bin(
                    float(row["model_confidence_spread"]), (0.1, 0.25, 0.5, 0.75)
                ),
            }
        )

    consensus_models = [model["name"] for model in row["models"] if model["prediction"] == row["consensus_prediction"]]
    disagreeing_with_soft = [model["name"] for model in row["models"] if model["prediction"] != row["soft_prediction"]]
    features["consensus model set"] = " + ".join(consensus_models) or "none"
    features["soft-disagreement model set"] = " + ".join(disagreeing_with_soft) or "none"
    for model in row["models"]:
        features[f"{model['name']} vs soft"] = (
            "agrees" if model["prediction"] == row["soft_prediction"] else "disagrees"
        )
        features[f"{model['name']} vs consensus"] = (
            "agrees" if model["prediction"] == row["consensus_prediction"] else "disagrees"
        )
        features[f"{model['name']} process state"] = model["state"] or "empty"
        features[f"{model['name']} state support"] = _visit_bin(int(model["state_visits"]))
        features[f"{model['name']} support rank"] = _relative_rank(row["models"], model, "state_visits")
        features[f"{model['name']} confidence rank"] = _relative_rank(row["models"], model, "confidence")
        features[f"{model['name']} entropy rank"] = _relative_rank(
            row["models"], model, "normalized_entropy", reverse=False
        )
        features[f"{model['name']} soft-prediction rank"] = _soft_rank_of_model(row, model)
    return features


def _condition_family(feature: str) -> str:
    if any(term in feature for term in ("margin", "entropy", "confidence", "empty prediction")):
        family = "distribution"
    elif feature.startswith("previous"):
        family = "previous outcome"
    elif "state" in feature or "n-gram" in feature or "support" in feature:
        family = "state evidence"
    elif "position" in feature or feature == "sequence stage":
        family = "position"
    elif "agreement" in feature or "consensus" in feature or "disagreement" in feature or " vs " in feature:
        family = "agreement"
    else:
        family = "other"
    return family


def discover_condition_hypotheses(
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    minimum_support: int = 10,
    maximum_hypotheses: int = 250,
) -> tuple[list[dict[str, Any]], str]:
    """Learn favorable feature conditions on calibration data and score them on test data."""
    source_rows = calibration_rows or test_rows
    source = "calibration" if calibration_rows else "test-derived exploratory"
    if not source_rows:
        return [], source
    model_names = [model["name"] for model in source_rows[0]["models"]]
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_rows:
        model_correct = {model["name"]: int(model["correct"]) for model in row["models"]}
        for feature, value in _condition_features(row).items():
            stats = grouped.setdefault(
                (feature, value),
                {"total": 0, "soft_correct": 0, "model_correct": Counter()},
            )
            stats["total"] += 1
            stats["soft_correct"] += int(row["soft_correct"])
            stats["model_correct"].update(model_correct)

    candidates: list[dict[str, Any]] = []
    for (feature, value), stats in grouped.items():
        total = stats["total"]
        if total < minimum_support:
            continue
        model_accuracies = {model_name: stats["model_correct"][model_name] / total for model_name in model_names}
        recommended_model = max(model_names, key=lambda name: (model_accuracies[name], -model_names.index(name)))
        soft_accuracy = stats["soft_correct"] / total
        calibration_gain = model_accuracies[recommended_model] - soft_accuracy
        if calibration_gain <= 0:
            continue
        condition_id = f"{feature}={value}"
        candidates.append(
            {
                "id": condition_id,
                "name": (
                    f"When {feature} is {value}, use the locally most accurate calibrated model "
                    f"(selected {recommended_model} in this run)"
                ),
                "family": _condition_family(feature),
                "feature": feature,
                "value": value,
                "recommended_model": recommended_model,
                "selection_policy": "highest calibration accuracy within the supported condition",
                "calibration_support": total,
                "calibration_soft_accuracy": soft_accuracy,
                "calibration_model_accuracy": model_accuracies[recommended_model],
                "calibration_gain": calibration_gain,
                "weight": calibration_gain * math.sqrt(total),
            }
        )

    candidates.sort(key=lambda item: (item["weight"], item["calibration_gain"]), reverse=True)
    candidates = candidates[:maximum_hypotheses]
    candidates_by_condition = {(candidate["feature"], candidate["value"]): candidate for candidate in candidates}
    impact_by_id: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in test_rows:
        feature_values = _condition_features(row)
        matching = [
            candidates_by_condition[(feature, value)]
            for feature, value in feature_values.items()
            if (feature, value) in candidates_by_condition
        ]
        row["condition_matches"] = [candidate["id"] for candidate in matching]
        for candidate in matching:
            prediction = _model_prediction(row, candidate["recommended_model"])
            impact = impact_by_id[candidate["id"]]
            impact["test_support"] += 1
            impact["soft_errors"] += int(not row["soft_correct"])
            impact["triggered"] += int(prediction != row["soft_prediction"])
            impact["recoveries"] += int(not row["soft_correct"] and prediction == row["actual"])
            impact["harms"] += int(row["soft_correct"] and prediction != row["actual"])

    for candidate in candidates:
        impact = impact_by_id[candidate["id"]]
        support = impact["test_support"]
        recoveries = impact["recoveries"]
        harms = impact["harms"]
        decisive = recoveries + harms
        candidate.update(
            {
                "test_support": support,
                "soft_errors": impact["soft_errors"],
                "recoveries": recoveries,
                "harms": harms,
                "net_correct": recoveries - harms,
                "conditional_delta": (recoveries - harms) / support if support else 0.0,
                "soft_error_recall": recoveries / impact["soft_errors"] if impact["soft_errors"] else 0.0,
                "decisive_precision": recoveries / decisive if decisive else 0.0,
            }
        )
    return sorted(candidates, key=lambda item: (item["net_correct"], item["recoveries"]), reverse=True), source


class NearestCalibrationBehaviorRule(DecisionRule):
    """Transfer model reliability from the nearest observed calibration behaviors."""

    family = "nearest behavior"
    description = (
        "Transfers model reliability from calibration events with similar agreement topology, state evidence, "
        "n-gram maturity, and distribution geometry."
    )
    interpretation = (
        "Neighbor composition reveals whether the current prefix resembles a stable repeated state, a sparse "
        "long-context state, or a contested branch across model abstractions."
    )
    selection_policy = "Choose the model with the best smoothed accuracy among structurally nearest calibration events."

    def __init__(self, neighbors: int = 32) -> None:
        self.neighbors = neighbors
        self.name = f"nearest calibration behavior (k {neighbors})"
        self.rows: list[dict[str, Any]] = []
        self.global_accuracy: dict[str, float] = {}
        self.default_model = ""
        self.indexes: list[defaultdict[tuple[Any, ...], list[dict[str, Any]]]] = []
        self.global_sample: list[dict[str, Any]] = []

    @staticmethod
    def _bucket_keys(row: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
        topology = _prediction_partition(row)
        visits = tuple(_visit_bin(int(model["state_visits"])) for model in row["models"])
        ranks = tuple(_soft_rank_of_model(row, model) for model in row["models"])
        margin = _value_bin(float(row.get("soft_margin", 0.0)), (0.01, 0.03, 0.07, 0.15))
        entropy = _value_bin(float(row.get("soft_normalized_entropy", 0.0)), (0.25, 0.5, 0.7, 0.85))
        return (
            ("topology", *topology, str(row["agreement_count"])),
            ("state-evidence", *visits, *topology),
            ("soft-ranks", *ranks, margin),
            ("distribution", margin, entropy, str(row["agreement_count"]), _sequence_stage(row)),
        )

    @staticmethod
    def _sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if len(rows) <= limit:
            return rows
        stride = len(rows) / limit
        return [rows[min(int(index * stride), len(rows) - 1)] for index in range(limit)]

    @staticmethod
    def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        distance = abs(float(left["relative_position"]) - float(right["relative_position"]))
        distance += 0.5 * abs(int(left["agreement_count"]) - int(right["agreement_count"]))
        distance += 0.5 * abs(int(left["distinct_prediction_count"]) - int(right["distinct_prediction_count"]))
        distance += 0.8 * (_prediction_partition(left) != _prediction_partition(right))
        for left_model, right_model in zip(left["models"], right["models"], strict=True):
            distance += 0.4 * (
                _visit_bin(int(left_model["state_visits"])) != _visit_bin(int(right_model["state_visits"]))
            )
            distance += 0.3 * (_soft_rank_of_model(left, left_model) != _soft_rank_of_model(right, right_model))
            distance += 0.25 * abs(left_model["confidence"] - right_model["confidence"])
            distance += 0.2 * abs(left_model["normalized_entropy"] - right_model["normalized_entropy"])
        return distance

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.default_model = _best_model(rows)
        self.global_accuracy = {}
        for model in rows[0]["models"] if rows else []:
            name = model["name"]
            total = sum(1 for row in rows for candidate in row["models"] if candidate["name"] == name)
            correct = sum(
                int(candidate["correct"]) for row in rows for candidate in row["models"] if candidate["name"] == name
            )
            self.global_accuracy[name] = correct / total if total else 0.0
        self.indexes = [defaultdict(list) for _ in range(4)]
        for candidate in rows:
            for index, key in enumerate(self._bucket_keys(candidate)):
                self.indexes[index][key].append(candidate)
        self.global_sample = self._sample(rows, max(128, self.neighbors * 4))

    def _candidates(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        limit = max(128, self.neighbors * 4)
        candidates: dict[tuple[Any, Any], dict[str, Any]] = {}
        for index, key in enumerate(self._bucket_keys(row)):
            pool = self.indexes[index].get(key, [])
            for candidate in self._sample(pool, limit):
                candidates[(candidate["sequence_index"], candidate["position"])] = candidate
            if len(candidates) >= limit:
                break
        if len(candidates) < self.neighbors:
            for candidate in self.global_sample:
                candidates[(candidate["sequence_index"], candidate["position"])] = candidate
        return list(candidates.values())

    def select(self, row: dict[str, Any]) -> str:
        if not self.rows:
            return self.default_model
        nearest = nsmallest(
            self.neighbors,
            ((self._distance(row, candidate), candidate) for candidate in self._candidates(row)),
            key=lambda item: item[0],
        )
        scores: dict[str, float] = {}
        for model in row["models"]:
            weighted_correct = 0.0
            weighted_total = 0.0
            for distance, neighbor in nearest:
                weight = 1.0 / (1.0 + distance)
                neighbor_model = next(item for item in neighbor["models"] if item["name"] == model["name"])
                weighted_correct += weight * float(neighbor_model["correct"])
                weighted_total += weight
            scores[model["name"]] = _smoothed_rate(
                weighted_correct,
                weighted_total,
                self.global_accuracy.get(model["name"], 0.0),
                prior_weight=8.0,
            )
        return max(row["models"], key=lambda model: (scores[model["name"]], -model["index"]))["name"]


class StackedRulePortfolio(DecisionRule):
    """Select among heterogeneous rules using an internal calibration holdout."""

    family = "stacked portfolio"
    description = "Uses internal validation to select among heterogeneous structural rules with contextual backoff."
    interpretation = "Indicates which reasoning family best explains each state-evidence and disagreement regime."
    selection_policy = "Choose the rule with the best internal-validation reliability in the current structural regime."

    def __init__(self, minimum_support: int = 5) -> None:
        self.minimum_support = minimum_support
        self.name = f"stacked rule portfolio (support {minimum_support})"
        self.rules: list[DecisionRule] = []
        self.rule_by_name: dict[str, DecisionRule] = {}
        self.rule_by_context: dict[tuple[str, ...], str] = {}
        self.global_rule = ""

    @staticmethod
    def _base_rules() -> list[DecisionRule]:
        return [
            HighestConfidenceRule(),
            GlobalAccuracyRule(),
            StateAccuracyRule(3),
            AgreementRule(3, "confidence"),
            HierarchicalReliabilityRule(3),
            ConsensusHierarchicalRule(3),
            ConfidenceStateReliabilityRule(5),
            DisagreementProfileRule(4),
            StateEvidenceTopologyRule(4),
            EvidenceWeightedDistributionRule(8),
            DistributionShapeReliabilityRule(8),
        ]

    @staticmethod
    def _contexts(row: dict[str, Any]) -> list[tuple[str, ...]]:
        return _structural_regime_contexts(row)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.rules = self._base_rules()
        self.rule_by_name = {rule.name: rule for rule in self.rules}
        if not rows:
            self.global_rule = ""
            return
        validation = [row for index, row in enumerate(rows) if index % 4 == 0]
        fitting = [row for index, row in enumerate(rows) if index % 4 != 0] or rows
        for rule in self.rules:
            rule.fit(fitting)

        grouped: defaultdict[tuple[str, ...], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        global_scores: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in validation:
            for rule in self.rules:
                prediction = rule.choose(row)[1]
                correct = int(prediction == row["actual"])
                global_scores[rule.name][0] += correct
                global_scores[rule.name][1] += 1
                for context in self._contexts(row):
                    grouped[context][rule.name].append(correct)
        self.global_rule = max(
            global_scores,
            key=lambda name: (
                global_scores[name][0] / global_scores[name][1],
                -list(global_scores).index(name),
            ),
            default="",
        )
        self.rule_by_context = {}
        for context, scores in grouped.items():
            eligible = {name: values for name, values in scores.items() if len(values) >= self.minimum_support}
            if eligible:
                self.rule_by_context[context] = max(
                    eligible,
                    key=lambda name: (sum(eligible[name]) / len(eligible[name]), -list(eligible).index(name)),
                )
        for rule in self.rules:
            rule.fit(rows)

    def _selected_rule(self, row: dict[str, Any]) -> DecisionRule:
        rule_name = self.global_rule
        for context in self._contexts(row):
            if context in self.rule_by_context:
                rule_name = self.rule_by_context[context]
                break
        return self.rule_by_name.get(rule_name, self.rules[0])

    def select(self, row: dict[str, Any]) -> str:
        return self._selected_rule(row).select(row)

    def choose(self, row: dict[str, Any]) -> tuple[str, str]:
        return self._selected_rule(row).choose(row)


def evaluate_weighted_conditions(
    rows: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    *,
    selected_ids: set[str] | None = None,
    weight_mode: str = "calibrated",
) -> dict[str, Any]:
    """Combine fulfilled favorable conditions as weighted votes for constituent models."""
    selected = [condition for condition in conditions if selected_ids is None or condition["id"] in selected_ids]
    by_id = {condition["id"]: condition for condition in selected}
    predictions: list[str] = []
    selected_models: Counter[str] = Counter()
    overridden = 0
    for row in rows:
        model_order = [model["name"] for model in row["models"]]
        votes: defaultdict[str, float] = defaultdict(float)
        for condition_id in row.get("condition_matches", []):
            condition = by_id.get(condition_id)
            if condition is None:
                continue
            if weight_mode == "equal":
                weight = 1.0
            elif weight_mode == "gain":
                weight = float(condition["calibration_gain"])
            else:
                weight = float(condition["weight"])
            votes[condition["recommended_model"]] += weight
        if votes:
            model_name = max(model_order, key=lambda name: (votes[name], -model_order.index(name)))
            prediction = _model_prediction(row, model_name)
            selected_models[model_name] += 1
            overridden += int(prediction != row["soft_prediction"])
        else:
            model_name = "soft voting"
            prediction = row["soft_prediction"]
            selected_models[model_name] += 1
        predictions.append(prediction)
    result = _override_impact(rows, predictions, name="weighted favorable conditions", family="weighted")
    result["selected_models"] = dict(selected_models)
    result["conditions"] = len(selected)
    result["overridden"] = overridden
    return result


def build_soft_failure_analysis(
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build rule impacts, conditional hypotheses, and calibrated weighted presets."""
    conditions, source = discover_condition_hypotheses(calibration_rows, test_rows)
    by_priority = sorted(conditions, key=lambda item: item["weight"], reverse=True)
    weighted_presets = []
    for requested_count in (5, 10, 20, 50):
        chosen = by_priority[:requested_count]
        if not chosen:
            continue
        result = evaluate_weighted_conditions(test_rows, conditions, selected_ids={item["id"] for item in chosen})
        result["name"] = f"top {len(chosen)} calibrated conditions"
        weighted_presets.append(result)
    soft_failures = [row for row in test_rows if not row["soft_correct"]]
    recoverable = [row for row in soft_failures if row["correct_models"]]
    return {
        "condition_source": source,
        "soft_failures": len(soft_failures),
        "recoverable_soft_failures": len(recoverable),
        "recoverable_soft_failure_rate": len(recoverable) / len(soft_failures) if soft_failures else 0.0,
        "rule_impacts": analyze_rule_impacts(test_rows),
        "condition_hypotheses": conditions,
        "weighted_presets": weighted_presets,
    }


CORE_RULE_SET = (
    "agreement ≥ 3, else confidence",
    "consensus hierarchy ≥ 3 (support 3)",
    "hierarchical reliability (support 3)",
    "confidence/state reliability (support 5)",
    "state-evidence topology (support 5)",
    "evidence-weighted distribution mixture (support 12)",
    "structural branching ensemble (support 8)",
    "calibrated soft rank 2 override (support 2)",
    "distribution-shape reliability (support 8)",
    "calibrated probability pool (support 5)",
    "delayed-feedback adaptive (decay 0.94)",
    "complete-miss state recovery (support 8)",
    "calibrated transient generalist pool",
)
ADVANCED_RULE_SET = (
    "consensus hierarchy ≥ 3 (support 3)",
    "hierarchical reliability (support 3)",
    "confidence/state reliability (support 5)",
    "disagreement profile reliability (support 4)",
    "state-evidence topology (support 5)",
    "evidence-weighted distribution mixture (support 12)",
    "structural branching ensemble (support 8)",
    "calibrated soft rank 2 override (support 2)",
    "calibrated soft rank 3 override (support 2)",
    "calibrated model rank 2 switch (support 8)",
    "calibrated model rank 3 switch (support 5)",
    "distribution-shape reliability (support 8)",
    "median probability pool",
    "trimmed probability pool",
    "product probability pool",
    "calibrated probability pool (support 5)",
    "stacked rule portfolio (support 5)",
    "delayed-feedback adaptive (decay 0.94)",
)


def _rule_consensus_prediction(row: dict[str, Any], rule_names: list[str], minimum_votes: int) -> str:
    predictions = [row.get("rule_predictions", {}).get(name, "") for name in rule_names]
    counts = Counter(prediction for prediction in predictions if prediction)
    if not counts:
        return row["soft_prediction"]
    highest_count = max(counts.values())
    winners = [prediction for prediction, count in counts.items() if count == highest_count]
    if highest_count < minimum_votes or len(winners) != 1:
        return row["soft_prediction"]
    return winners[0]


def evaluate_rule_scenarios(
    rows: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate fixed rule-set consensus and diagnostic oracle scenarios."""
    if not rows:
        return []
    available = set(rows[0].get("rule_predictions", {}))
    all_rules = [hypothesis["name"] for hypothesis in hypotheses if hypothesis["name"] in available]
    core_rules = [name for name in CORE_RULE_SET if name in available]
    advanced_rules = [name for name in ADVANCED_RULE_SET if name in available]
    definitions = [
        ("core rules: consensus ≥ 2", core_rules, 2, False),
        ("core rules: consensus ≥ 3", core_rules, 3, False),
        ("advanced rules: consensus ≥ 2", advanced_rules, 2, False),
        ("all rules: consensus ≥ 3", all_rules, 3, False),
        ("core rule-set oracle ceiling", core_rules, 1, True),
        ("all rule-set oracle ceiling", all_rules, 1, True),
    ]
    soft_correct = sum(row["soft_correct"] for row in rows)
    cheating_correct = sum(row["oracle_correct"] for row in rows)
    recoverable_gap = cheating_correct - soft_correct
    scenarios = []
    for name, rule_names, minimum_votes, oracle in definitions:
        if not rule_names:
            continue
        if oracle:
            predictions = [
                row["actual"]
                if any(row["rule_predictions"].get(rule_name) == row["actual"] for rule_name in rule_names)
                else row["soft_prediction"]
                for row in rows
            ]
        else:
            predictions = [_rule_consensus_prediction(row, rule_names, minimum_votes) for row in rows]
            for row, prediction in zip(rows, predictions, strict=True):
                row.setdefault("scenario_predictions", {})[name] = prediction
        correct = sum(prediction == row["actual"] for row, prediction in zip(rows, predictions, strict=True))
        net_correct = correct - soft_correct
        scenarios.append(
            {
                "name": name,
                "family": "rule-set oracle" if oracle else "rule-set consensus",
                "rule_names": rule_names,
                "minimum_votes": minimum_votes,
                "oracle": oracle,
                "accuracy": correct / len(rows),
                "correct": correct,
                "total": len(rows),
                "net_correct": net_correct,
                "net_accuracy_delta": net_correct / len(rows),
                "gap_recovered_fraction": net_correct / recoverable_gap if recoverable_gap else 0.0,
            }
        )
    return sorted(scenarios, key=lambda item: item["accuracy"], reverse=True)


def _integration_contexts(row: dict[str, Any]) -> list[tuple[str, ...]]:
    """Return activity-invariant contexts from specific to general for meta-selection."""
    return _structural_regime_contexts(row)


def _rule_scores(rows: list[dict[str, Any]], rule_names: list[str]) -> dict[str, float]:
    """Measure rule accuracy on the selector-calibration partition."""
    if not rows:
        return dict.fromkeys(rule_names, 0.0)
    return {
        name: sum(row["rule_predictions"].get(name) == row["actual"] for row in rows) / len(rows) for name in rule_names
    }


def _family_weighted_prediction(
    row: dict[str, Any],
    rule_names: list[str],
    weights: dict[str, float],
) -> str:
    """Combine rule votes while giving each rule family equal total influence."""
    family_votes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    family_totals: Counter[str] = Counter()
    for name in rule_names:
        prediction = row.get("rule_predictions", {}).get(name, "")
        weight = max(0.0, weights.get(name, 0.0))
        if not prediction or weight <= 0:
            continue
        family = _rule_family(name)
        family_votes[family][prediction] += weight
        family_totals[family] += weight
    scores: Counter[str] = Counter()
    for family, votes in family_votes.items():
        total = family_totals[family]
        for prediction, weight in votes.items():
            scores[prediction] += weight / total
    if not scores:
        return row["soft_prediction"]
    highest = max(scores.values())
    winners = sorted(prediction for prediction, score in scores.items() if math.isclose(score, highest))
    return row["soft_prediction"] if row["soft_prediction"] in winners else winners[0]


def _specialist_override_prediction(
    row: dict[str, Any],
    rule_names: list[str],
    weights: dict[str, float],
) -> str:
    active_weights = {
        name: weight
        for name, weight in weights.items()
        if row["rule_predictions"].get(name, row["soft_prediction"]) != row["soft_prediction"]
    }
    return _family_weighted_prediction(row, rule_names, active_weights)


def _family_prediction_choices(row: dict[str, Any], rule_names: list[str]) -> dict[str, str]:
    predictions_by_family: defaultdict[str, list[str]] = defaultdict(list)
    for name in rule_names:
        prediction = row["rule_predictions"].get(name, "")
        if prediction:
            predictions_by_family[_rule_family(name)].append(prediction)
    choices = {}
    for family, predictions in predictions_by_family.items():
        counts = Counter(predictions)
        highest = max(counts.values())
        winners = sorted(prediction for prediction, count in counts.items() if count == highest)
        if len(winners) == 1:
            choices[family] = winners[0]
    return choices


def _prediction_role(row: dict[str, Any], prediction: str) -> str:
    """Encode a candidate by structural role rather than activity identity."""
    if prediction == row["soft_prediction"]:
        return "soft-top"
    for rank, item in enumerate(row.get("soft_ranked_predictions", []), start=1):
        if item["activity"] == prediction:
            return f"soft-rank-{min(rank, 4)}"
    groups = _prediction_partition(row)
    matching_groups = [groups[index] for index, model in enumerate(row["models"]) if model["prediction"] == prediction]
    return "model-group-" + (matching_groups[0] if matching_groups else "outside")


def _fit_confusion_router(
    rows: list[dict[str, Any]],
    rule_names: list[str],
    *,
    minimum_support: int = 5,
    confidence_z: float = 0.5,
) -> dict[tuple[str, str, tuple[str, ...]], float]:
    gains: defaultdict[tuple[str, str, tuple[str, ...]], list[int]] = defaultdict(list)
    for row in rows:
        soft = row["soft_prediction"]
        for family, candidate in _family_prediction_choices(row, rule_names).items():
            if candidate != soft:
                key = (family, _prediction_role(row, candidate), _prediction_partition(row))
                gains[key].append(int(candidate == row["actual"]) - int(soft == row["actual"]))
    weights = {}
    for key, outcomes in gains.items():
        total = len(outcomes)
        if total < minimum_support:
            continue
        mean = sum(outcomes) / total
        variance = sum((outcome - mean) ** 2 for outcome in outcomes) / max(1, total - 1)
        lower_bound = mean - confidence_z * math.sqrt(variance / total)
        if lower_bound > 0:
            weights[key] = lower_bound * math.sqrt(total)
    return weights


def _confusion_router_prediction(
    row: dict[str, Any],
    rule_names: list[str],
    weights: dict[tuple[str, str, tuple[str, ...]], float],
) -> str:
    scores: Counter[str] = Counter()
    soft = row["soft_prediction"]
    for family, candidate in _family_prediction_choices(row, rule_names).items():
        if candidate != soft:
            key = (family, _prediction_role(row, candidate), _prediction_partition(row))
            scores[candidate] += weights.get(key, 0.0)
    return max(sorted(scores), key=scores.get) if scores and max(scores.values()) > 0 else soft  # type: ignore[arg-type]


def _fit_expert_overlay(
    rows: list[dict[str, Any]],
    rule_names: list[str],
    base_rule: str,
    *,
    minimum_support: int = 5,
    confidence_z: float = 0.5,
) -> dict[tuple[str, str, str, tuple[str, ...]], float]:
    outcomes_by_route: defaultdict[tuple[str, str, str, tuple[str, ...]], list[int]] = defaultdict(list)
    for row in rows:
        base = row["rule_predictions"].get(base_rule, row["soft_prediction"])
        for name in rule_names:
            candidate = row["rule_predictions"].get(name, base)
            if candidate != base:
                route = (
                    name,
                    _prediction_role(row, base),
                    _prediction_role(row, candidate),
                    _prediction_partition(row),
                )
                outcomes_by_route[route].append(int(candidate == row["actual"]) - int(base == row["actual"]))
    weights = {}
    for route, outcomes in outcomes_by_route.items():
        if len(outcomes) < minimum_support:
            continue
        mean = sum(outcomes) / len(outcomes)
        variance = sum((outcome - mean) ** 2 for outcome in outcomes) / max(1, len(outcomes) - 1)
        lower_bound = mean - confidence_z * math.sqrt(variance / len(outcomes))
        if lower_bound > 0:
            weights[route] = lower_bound * math.sqrt(len(outcomes))
    return weights


def _expert_overlay_prediction(
    row: dict[str, Any],
    rule_names: list[str],
    base_rule: str,
    weights: dict[tuple[str, str, str, tuple[str, ...]], float],
) -> str:
    base = row["rule_predictions"].get(base_rule, row["soft_prediction"])
    family_scores: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for name in rule_names:
        candidate = row["rule_predictions"].get(name, base)
        route = (
            name,
            _prediction_role(row, base),
            _prediction_role(row, candidate),
            _prediction_partition(row),
        )
        weight = weights.get(route, 0.0)
        if candidate != base and weight > 0:
            family_scores[_rule_family(name)][candidate] += weight
    scores: Counter[str] = Counter()
    for votes in family_scores.values():
        total = sum(votes.values())
        for candidate, weight in votes.items():
            scores[candidate] += weight / total
    return max(sorted(scores), key=scores.get) if scores else base  # type: ignore[arg-type]


def _fit_context_experts(
    rows: list[dict[str, Any]],
    rule_names: list[str],
    minimum_support: int,
) -> dict[tuple[str, ...], str]:
    """Fit the best rule in each observable context, retaining soft when safer."""
    grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for context in _integration_contexts(row):
            grouped[context].append(row)
    experts: dict[tuple[str, ...], str] = {}
    for context, context_rows in grouped.items():
        if len(context_rows) < minimum_support:
            continue
        soft_correct = sum(row["soft_correct"] for row in context_rows)
        correct = {
            name: sum(row["rule_predictions"].get(name) == row["actual"] for row in context_rows) for name in rule_names
        }
        best = max(rule_names, key=lambda name: (correct[name], -rule_names.index(name)))
        experts[context] = best if correct[best] > soft_correct else "soft voting"
    return experts


def _context_expert_prediction(row: dict[str, Any], experts: dict[tuple[str, ...], str]) -> str:
    for context in _integration_contexts(row):
        if context in experts:
            source = experts[context]
            return row["soft_prediction"] if source == "soft voting" else row["rule_predictions"].get(source, "")
    return row["soft_prediction"]


def _fit_risk_controlled_experts(
    rows: list[dict[str, Any]],
    rule_names: list[str],
    *,
    minimum_support: int = 10,
    confidence_z: float = 1.0,
) -> dict[tuple[str, ...], str]:
    """Retain contextual overrides only when paired gain clears uncertainty."""
    grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for context in _integration_contexts(row):
            grouped[context].append(row)
    experts: dict[tuple[str, ...], str] = {}
    for context, context_rows in grouped.items():
        total = len(context_rows)
        if total < minimum_support:
            continue
        gains_by_rule = {
            name: [
                int(row["rule_predictions"].get(name) == row["actual"]) - int(row["soft_correct"])
                for row in context_rows
            ]
            for name in rule_names
        }
        best = max(rule_names, key=lambda name: (sum(gains_by_rule[name]), -rule_names.index(name)))
        gains = gains_by_rule[best]
        mean_gain = sum(gains) / total
        variance = sum((gain - mean_gain) ** 2 for gain in gains) / max(1, total - 1)
        lower_bound = mean_gain - confidence_z * math.sqrt(variance / total)
        experts[context] = best if lower_bound > 0 else "soft voting"
    return experts


def _fit_context_reliability(
    rows: list[dict[str, Any]],
    rule_names: list[str],
) -> dict[tuple[tuple[str, ...], str], tuple[int, int]]:
    totals: Counter[tuple[str, ...]] = Counter()
    correct: Counter[tuple[tuple[str, ...], str]] = Counter()
    for row in rows:
        contexts = _integration_contexts(row)
        for context in contexts:
            totals[context] += 1
            for name in rule_names:
                correct[(context, name)] += int(row["rule_predictions"].get(name) == row["actual"])
    reliability: dict[tuple[tuple[str, ...], str], tuple[int, int]] = {}
    for context, total in totals.items():
        for name in rule_names:
            reliability[(context, name)] = (correct[(context, name)], total)
    return reliability


def _contextual_vote_prediction(
    row: dict[str, Any],
    rule_names: list[str],
    global_scores: dict[str, float],
    reliability: dict[tuple[tuple[str, ...], str], tuple[int, int]],
    *,
    minimum_support: int = 8,
    prior_strength: float = 12.0,
) -> str:
    weights: dict[str, float] = {}
    contexts = _integration_contexts(row)
    for name in rule_names:
        estimate = global_scores[name]
        for context in contexts:
            correct, total = reliability.get((context, name), (0, 0))
            if total >= minimum_support:
                estimate = (correct + prior_strength * global_scores[name]) / (total + prior_strength)
                break
        weights[name] = max(estimate - 0.5, 0.001)
    return _family_weighted_prediction(row, rule_names, weights)


def _behavior_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Activity-invariant distance between state evidence and disagreement behavior."""
    distance = 2.0 * abs(float(left["relative_position"]) - float(right["relative_position"]))
    distance += 0.35 * abs(int(left["agreement_count"]) - int(right["agreement_count"]))
    distance += 0.25 * abs(int(left["distinct_prediction_count"]) - int(right["distinct_prediction_count"]))
    distance += 1.5 * abs(float(left.get("soft_margin", 0.0)) - float(right.get("soft_margin", 0.0)))
    distance += abs(float(left.get("soft_normalized_entropy", 0.0)) - float(right.get("soft_normalized_entropy", 0.0)))
    distance += 0.3 * (left.get("previous_soft_correct") != right.get("previous_soft_correct"))
    distance += 0.9 * (_prediction_partition(left) != _prediction_partition(right))
    for left_model, right_model in zip(left["models"], right["models"], strict=True):
        distance += 0.3 * (_visit_bin(int(left_model["state_visits"])) != _visit_bin(int(right_model["state_visits"])))
        distance += 0.25 * (_soft_rank_of_model(left, left_model) != _soft_rank_of_model(right, right_model))
        distance += 0.2 * abs(float(left_model["soft_divergence"]) - float(right_model["soft_divergence"]))
    return distance


def _local_competence_prediction(
    row: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    rule_names: list[str],
    global_scores: dict[str, float],
    *,
    neighbors: int = 32,
    prior_strength: float = 12.0,
) -> str:
    nearest = sorted(calibration_rows, key=lambda candidate: _behavior_distance(row, candidate))[:neighbors]
    weights = {}
    for name in rule_names:
        correct = sum(candidate["rule_predictions"].get(name) == candidate["actual"] for candidate in nearest)
        estimate = (correct + prior_strength * global_scores[name]) / (len(nearest) + prior_strength)
        weights[name] = max(estimate - 0.5, 0.001)
    return _family_weighted_prediction(row, rule_names, weights)


def _family_consensus_prediction(
    row: dict[str, Any],
    rule_names: list[str],
    *,
    maximum_margin: float,
    minimum_families: int,
) -> str:
    """Use an alternative only when independent rule families agree under uncertainty."""
    if float(row.get("soft_margin", 0.0)) > maximum_margin:
        return row["soft_prediction"]
    by_family: defaultdict[str, list[str]] = defaultdict(list)
    for name in rule_names:
        prediction = row["rule_predictions"].get(name, "")
        if prediction:
            by_family[_rule_family(name)].append(prediction)
    family_choices = []
    for predictions in by_family.values():
        counts = Counter(predictions)
        highest = max(counts.values())
        winners = sorted(prediction for prediction, count in counts.items() if count == highest)
        if len(winners) == 1:
            family_choices.append(winners[0])
    counts = Counter(family_choices)
    alternatives = [(count, prediction) for prediction, count in counts.items() if prediction != row["soft_prediction"]]
    if not alternatives:
        return row["soft_prediction"]
    best_count, best_prediction = max(alternatives, key=lambda item: (item[0], item[1]))
    tied = sum(count == best_count for count, _ in alternatives) > 1
    return best_prediction if best_count >= minimum_families and not tied else row["soft_prediction"]


def _hedge_predictions(
    rows: list[dict[str, Any]],
    rule_names: list[str],
    initial_scores: dict[str, float],
    learning_rate: float,
) -> list[str]:
    """Run delayed-feedback exponential expert weighting over an event stream."""
    weights = {name: math.exp(4 * (initial_scores[name] - 0.5)) for name in rule_names}
    predictions: list[str] = []
    for row in rows:
        predictions.append(_family_weighted_prediction(row, rule_names, weights))
        for name in rule_names:
            correct = row["rule_predictions"].get(name) == row["actual"]
            weights[name] *= math.exp(learning_rate * (1 if correct else -1))
        scale = max(weights.values(), default=1.0)
        if scale > HEDGE_RESCALE_MAX or scale < HEDGE_RESCALE_MIN:
            weights = {name: weight / scale for name, weight in weights.items()}
    return predictions


def _sequence_hedge_predictions(
    rows: list[dict[str, Any]],
    rule_names: list[str],
    initial_scores: dict[str, float],
    learning_rate: float,
) -> list[str]:
    """Run independent delayed-feedback expert portfolios inside each case."""
    initial = {name: math.exp(4 * (initial_scores[name] - 0.5)) for name in rule_names}
    weights_by_sequence: dict[str, dict[str, float]] = {}
    predictions = []
    for row in rows:
        weights = weights_by_sequence.setdefault(row["sequence_id"], initial.copy())
        predictions.append(_family_weighted_prediction(row, rule_names, weights))
        for name in rule_names:
            correct = row["rule_predictions"].get(name) == row["actual"]
            weights[name] *= math.exp(learning_rate * (1 if correct else -1))
    return predictions


def _integration_result(
    rows: list[dict[str, Any]],
    predictions: list[str],
    *,
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    for row, prediction in zip(rows, predictions, strict=True):
        row.setdefault("integration_predictions", {})[name] = prediction
    impact = _override_impact(rows, predictions, name=name, family="smart integration")
    return {
        **impact,
        "accuracy": impact["resulting_accuracy"],
        "correct": round(impact["resulting_accuracy"] * len(rows)),
        "description": description,
        "parameters": parameters,
        "deployable": True,
    }


INDEPENDENT_BOOST_STACK_NAME = "independent Bag + N-gram boost stack"


def evaluate_independent_boost_stack(
    rows: list[dict[str, Any]],
    rule_names: list[str],
) -> dict[str, Any] | None:
    """Compose every active rule as independent non-negative model multipliers."""
    boost_rules = [
        name
        for name in rule_names
        if any(row.get("rule_diagnostics", {}).get(name, {}).get("model_multipliers") for row in rows)
        or name.startswith("transient Bag favoritism")
        or name == NGramCorrectStreakBoostRule.name
    ]
    if not rows or not boost_rules:
        return None

    predictions: list[str] = []
    bag_events = 0
    ngram_events = 0
    simultaneous_events = 0
    overlap_events = 0
    bag_rule_overlap_events = 0
    negative_multiplier_events = 0
    for row in rows:
        combined = {model["name"]: 1.0 for model in row["models"]}
        targets_by_rule: dict[str, list[str]] = {}
        bag_modifiers: defaultdict[str, list[float]] = defaultdict(list)
        for name in boost_rules:
            modifiers = row.get("rule_diagnostics", {}).get(name, {}).get("model_multipliers", {})
            targets_by_rule[name] = sorted(modifiers)
            for model_name, raw_multiplier in modifiers.items():
                multiplier = float(raw_multiplier)
                if multiplier < 1.0:
                    negative_multiplier_events += 1
                if name.startswith("transient Bag favoritism"):
                    bag_modifiers[model_name].append(multiplier)
                else:
                    combined[model_name] = combined.get(model_name, 1.0) * multiplier
        # The two Bag rules are complementary trigger variants of one
        # mechanism. Their recovery windows can overlap after consecutive
        # errors, so coalesce them with max instead of counting the same
        # structural evidence twice.
        for model_name, modifiers in bag_modifiers.items():
            combined[model_name] = max(combined.get(model_name, 1.0), *modifiers)

        bag_rule_targets = [
            set(targets) for name, targets in targets_by_rule.items() if name.startswith("transient Bag favoritism")
        ]
        bag_targets = set().union(*bag_rule_targets) if bag_rule_targets else set()
        bag_rule_overlap_events += int(sum(bool(targets) for targets in bag_rule_targets) > 1)
        ngram_targets = set(targets_by_rule.get(NGramCorrectStreakBoostRule.name, []))
        bag_events += int(bool(bag_targets))
        ngram_events += int(bool(ngram_targets))
        simultaneous_events += int(bool(bag_targets and ngram_targets))
        overlap_events += int(bool(bag_targets & ngram_targets))
        prediction = _weighted_model_distribution_prediction(row, combined)
        predictions.append(prediction)
        row.setdefault("integration_diagnostics", {})[INDEPENDENT_BOOST_STACK_NAME] = {
            "model_multipliers": combined,
            "rule_targets": targets_by_rule,
            "simultaneous": bool(bag_targets and ngram_targets),
            "target_overlap": sorted(bag_targets & ngram_targets),
            "bag_rule_overlap": sum(bool(targets) for targets in bag_rule_targets) > 1,
            "all_multipliers_non_negative": all(value >= 0.0 for value in combined.values()),
        }

    result = _integration_result(
        rows,
        predictions,
        name=INDEPENDENT_BOOST_STACK_NAME,
        description=(
            "Starts every constituent at weight 1.0, coalesces overlapping transient Bag variants by their "
            "strongest positive multiplier, "
            "downweights each eligible N-gram from its own correctness streak, and merges all distributions "
            "once. There is no routing hierarchy, calibration gate, or rule suppression."
        ),
        parameters={
            "composition": "non-negative model multipliers; max within Bag variants, product across model families",
            "base_weight": 1.0,
            "rules": boost_rules,
            "calibrated": False,
            "priority_order": [],
            "enforcement_audit": {
                "bag_boost_events": bag_events,
                "ngram_boost_events": ngram_events,
                "simultaneous_boost_events": simultaneous_events,
                "cross_rule_target_overlap_events": overlap_events,
                "bag_rule_overlap_events": bag_rule_overlap_events,
                "negative_multiplier_events": negative_multiplier_events,
            },
        },
    )
    result["deployment_priority"] = True
    cheating_gap = (sum(row["oracle_correct"] for row in rows) - sum(row["soft_correct"] for row in rows)) / len(rows)
    result["gap_recovered_fraction"] = result["net_accuracy_delta"] / cheating_gap if cheating_gap else 0.0
    return result


def _priority_adaptive_prediction(
    row: dict[str, Any],
    base_rule: str,
    ngram_rule: str,
    recovery_rule: str,
) -> str:
    """Apply collective N-gram evidence before transient generalist recovery."""
    ngram_diagnostic = row.get("rule_diagnostics", {}).get(ngram_rule, {})
    if ngram_rule and ngram_diagnostic.get("collectively_strong"):
        return row["rule_predictions"][ngram_rule]
    recovery_diagnostic = row.get("rule_diagnostics", {}).get(recovery_rule, {})
    if recovery_rule and recovery_diagnostic.get("active"):
        return row["rule_predictions"][recovery_rule]
    if ngram_rule and ngram_diagnostic.get("active"):
        return row["rule_predictions"][ngram_rule]
    if base_rule == "soft voting":
        return row["soft_prediction"]
    return row["rule_predictions"].get(base_rule, row["soft_prediction"])


def evaluate_adaptive_recovery_overlay(
    test_rows: list[dict[str, Any]],
    rule_names: list[str],
    *,
    recovery_rule: str | None = None,
    base_rule: str = "soft voting",
    deployment_priority: bool = True,
    calibration_accuracy: float | None = None,
    enable_ngram: bool = True,
    ngram_selector_gain: int | None = None,
) -> dict[str, Any] | None:
    """Build and audit an adaptive stack around one selected recovery policy."""
    if recovery_rule is None:
        recovery_rule = next((name for name in rule_names if "after any soft error" in name), "")
    ngram_rule = next((name for name in rule_names if name == NGramCorrectStreakBoostRule.name), "")
    if not enable_ngram:
        ngram_rule = ""
    if not test_rows or not recovery_rule:
        return None
    ngram_diagnostic = next(
        (
            row.get("rule_diagnostics", {}).get(ngram_rule, {})
            for row in test_rows
            if ngram_rule and ngram_rule in row.get("rule_diagnostics", {})
        ),
        {},
    )
    priority_result = _integration_result(
        test_rows,
        [_priority_adaptive_prediction(row, base_rule, ngram_rule, recovery_rule) for row in test_rows],
        name=("calibration-selected adaptive recovery" if deployment_priority else "uncalibrated adaptive recovery"),
        description=(
            "Lets collective N-gram evidence override the selected recovery policy; otherwise applies that "
            "policy only when its own gate is active, then any remaining N-gram boost and the base prediction."
        ),
        parameters={
            "priority_order": [
                f"{ngram_rule}: collective gate" if ngram_rule else "no ngram streak rule",
                recovery_rule,
                f"{ngram_rule}: ordinary activation" if ngram_rule else "no ngram streak rule",
                base_rule,
            ],
            "selected_recovery_rule": recovery_rule,
            "selector_calibration_accuracy": calibration_accuracy,
            "ngram_overlay_enabled": enable_ngram and bool(ngram_rule),
            "ngram_selector_net_gain": ngram_selector_gain,
            "base_rule": base_rule,
            "calibration_selected": deployment_priority,
            "ngram_collective_gate": {
                "minimum_models": ngram_diagnostic.get("collective_min_models", 2),
                "minimum_ratio_each": ngram_diagnostic.get("collective_min_ratio", 0.75),
            },
        },
    )
    priority_result["deployment_priority"] = deployment_priority
    active_rows = [row for row in test_rows if row.get("rule_diagnostics", {}).get(recovery_rule, {}).get("active")]
    collective_ngram_rows = [
        row
        for row in test_rows
        if ngram_rule and row.get("rule_diagnostics", {}).get(ngram_rule, {}).get("collectively_strong")
    ]
    collective_row_ids = {id(row) for row in collective_ngram_rows}
    active_row_ids = {id(row) for row in active_rows}
    recovery_enforced_rows = [row for row in active_rows if id(row) not in collective_row_ids]
    priority_result["parameters"]["enforcement_audit"] = {
        "recovery_candidate_events": sum(
            row.get("rule_diagnostics", {}).get(recovery_rule, {}).get("recovery_age") is not None for row in test_rows
        ),
        "recovery_applied_events": len(active_rows),
        "recovery_rejected_events": sum(
            row.get("rule_diagnostics", {}).get(recovery_rule, {}).get("recovery_age") is not None
            and not row.get("rule_diagnostics", {}).get(recovery_rule, {}).get("active")
            for row in test_rows
        ),
        "overlay_prediction_mismatches": sum(
            row["integration_predictions"][priority_result["name"]] != row["rule_predictions"][recovery_rule]
            for row in recovery_enforced_rows
        ),
        "collective_ngram_override_events": len(collective_ngram_rows),
        "collective_ngram_over_recovery_events": sum(id(row) in active_row_ids for row in collective_ngram_rows),
        "collective_ngram_precedence_mismatches": sum(
            row["integration_predictions"][priority_result["name"]] != row["rule_predictions"][ngram_rule]
            for row in collective_ngram_rows
        ),
    }
    cheating_gap = (
        sum(row["oracle_correct"] for row in test_rows) - sum(row["soft_correct"] for row in test_rows)
    ) / len(test_rows)
    priority_result["gap_recovered_fraction"] = (
        priority_result["net_accuracy_delta"] / cheating_gap if cheating_gap else 0.0
    )
    return priority_result


ARCHIVED_INTEGRATION_NAMES = frozenset(
    {
        "calibration-best rule",
        "calibration-selected adaptive recovery",
        "uncalibrated adaptive recovery",
        "risk-gated specialist overlay",
        "family-balanced reliability vote",
        "positive-gain family vote",
        "risk-gated override specialists",
        "structural disagreement family router",
        "contextual best expert (support 8)",
        "contextual best expert (support 20)",
        "risk-controlled contextual expert",
        "contextual Bayesian family vote",
        "nearest-behavior local competence",
        "uncertainty-gated family consensus",
        "delayed-feedback Hedge portfolio",
        "sequence-local delayed-feedback Hedge",
    }
)


def evaluate_rule_integrations(  # noqa: C901, PLR0912, PLR0915
    selector_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    rule_names: list[str],
    *,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    """Compose active boosts directly; retain former selectors only for archived ablations."""
    if not test_rows or not rule_names:
        return []
    rule_names = [name for name in rule_names if name in test_rows[0].get("rule_predictions", {})]
    if not rule_names:
        return []
    if not include_archived:
        stack = evaluate_independent_boost_stack(test_rows, rule_names)
        active_names = {stack["name"]} if stack else set()
        for row in test_rows:
            row["integration_predictions"] = {
                name: prediction
                for name, prediction in row.get("integration_predictions", {}).items()
                if name in active_names
            }
            row["integration_diagnostics"] = {
                name: diagnostics
                for name, diagnostics in row.get("integration_diagnostics", {}).items()
                if name in active_names
            }
        return [stack] if stack else []
    if not selector_rows:
        return []
    rule_names = [name for name in rule_names if name in selector_rows[0].get("rule_predictions", {})]
    if not rule_names:
        return []
    scores = _rule_scores(selector_rows, rule_names)
    soft_score = sum(row["soft_correct"] for row in selector_rows) / len(selector_rows)
    results: list[dict[str, Any]] = []

    best_rule = max(rule_names, key=lambda name: (scores[name], -rule_names.index(name)))
    results.append(
        _integration_result(
            test_rows,
            [row["rule_predictions"].get(best_rule, row["soft_prediction"]) for row in test_rows],
            name="calibration-best rule",
            description="Selects the single rule with the highest accuracy on the selector-calibration partition.",
            parameters={"rule": best_rule, "calibration_accuracy": scores[best_rule]},
        )
    )

    calibrated_recovery_rules = [name for name in rule_names if name.startswith("transient recovery calibrated by")]
    if calibrated_recovery_rules:
        selected_recovery = max(
            calibrated_recovery_rules,
            key=lambda name: (scores[name], -rule_names.index(name)),
        )
        ngram_rule = next((name for name in rule_names if name == NGramCorrectStreakBoostRule.name), "")
        with_ngram = [
            _priority_adaptive_prediction(row, "soft voting", ngram_rule, selected_recovery) for row in selector_rows
        ]
        without_ngram = [
            _priority_adaptive_prediction(row, "soft voting", "", selected_recovery) for row in selector_rows
        ]
        ngram_selector_gain = sum(
            int(candidate == row["actual"]) - int(baseline == row["actual"])
            for row, candidate, baseline in zip(selector_rows, with_ngram, without_ngram, strict=True)
        )
        priority_result = evaluate_adaptive_recovery_overlay(
            test_rows,
            rule_names,
            recovery_rule=selected_recovery,
            calibration_accuracy=scores[selected_recovery],
            enable_ngram=ngram_selector_gain > 0,
            ngram_selector_gain=ngram_selector_gain,
        )
        if priority_result:
            results.append(priority_result)

    overlay_weights = _fit_expert_overlay(selector_rows, rule_names, best_rule)
    results.append(
        _integration_result(
            test_rows,
            [_expert_overlay_prediction(row, rule_names, best_rule, overlay_weights) for row in test_rows],
            name="risk-gated specialist overlay",
            description=(
                "Starts from the strongest calibration-selected rule rather than soft voting, then applies only "
                "structural prediction-role corrections with positive paired lower-bound gain."
            ),
            parameters={
                "base_rule": best_rule,
                "minimum_support": 5,
                "confidence_z": 0.5,
                "routes": len(overlay_weights),
            },
        )
    )

    global_weights = {name: max(scores[name] - 0.5, 0.001) for name in rule_names}
    results.append(
        _integration_result(
            test_rows,
            [_family_weighted_prediction(row, rule_names, global_weights) for row in test_rows],
            name="family-balanced reliability vote",
            description=(
                "Weights rules by calibration accuracy and normalizes each rule family so many similar variants "
                "cannot dominate by duplication."
            ),
            parameters={"weight": "max(calibration accuracy - 0.5, 0.001)"},
        )
    )

    positive_weights = {name: max(scores[name] - soft_score, 0.0) for name in rule_names}
    results.append(
        _integration_result(
            test_rows,
            [_family_weighted_prediction(row, rule_names, positive_weights) for row in test_rows],
            name="positive-gain family vote",
            description=(
                "Votes only with rule families whose selector-calibration accuracy exceeded soft voting; "
                "otherwise it falls back to soft voting."
            ),
            parameters={"selector_soft_accuracy": soft_score},
        )
    )

    specialist_weights = {}
    for name in rule_names:
        outcomes = [
            int(row["rule_predictions"].get(name) == row["actual"]) - int(row["soft_correct"])
            for row in selector_rows
            if row["rule_predictions"].get(name, row["soft_prediction"]) != row["soft_prediction"]
        ]
        if len(outcomes) < MIN_SPECIALIST_OVERRIDES:
            specialist_weights[name] = 0.0
            continue
        mean = sum(outcomes) / len(outcomes)
        variance = sum((outcome - mean) ** 2 for outcome in outcomes) / max(1, len(outcomes) - 1)
        lower_bound = mean - 0.5 * math.sqrt(variance / len(outcomes))
        specialist_weights[name] = max(0.0, lower_bound * math.sqrt(len(outcomes)))
    results.append(
        _integration_result(
            test_rows,
            [_specialist_override_prediction(row, rule_names, specialist_weights) for row in test_rows],
            name="risk-gated override specialists",
            description=(
                "Activates only rules that disagree with soft voting and gives weight only to specialists whose "
                "paired calibration gain remains positive after an uncertainty penalty."
            ),
            parameters={"minimum_overrides": 5, "confidence_z": 0.5},
        )
    )

    confusion_weights = _fit_confusion_router(selector_rows, rule_names)
    results.append(
        _integration_result(
            test_rows,
            [_confusion_router_prediction(row, rule_names, confusion_weights) for row in test_rows],
            name="structural disagreement family router",
            description=(
                "Learns which independent rule families reliably correct each canonical prediction rank and "
                "cross-model disagreement topology, without using activity identities."
            ),
            parameters={"minimum_support": 5, "confidence_z": 0.5, "routes": len(confusion_weights)},
        )
    )

    if include_archived:
        for support in (8, 20):
            experts = _fit_context_experts(selector_rows, rule_names, support)
            results.append(
                _integration_result(
                    test_rows,
                    [_context_expert_prediction(row, experts) for row in test_rows],
                    name=f"contextual best expert (support {support})",
                    description=(
                        "Chooses a rule only in calibration-backed state-evidence, n-gram maturity, uncertainty, "
                        "feedback, or agreement-topology regimes where it beat soft voting, with hierarchical "
                        "backoff."
                    ),
                    parameters={"minimum_support": support, "learned_contexts": len(experts)},
                )
            )

    risk_experts = _fit_risk_controlled_experts(selector_rows, rule_names)
    results.append(
        _integration_result(
            test_rows,
            [_context_expert_prediction(row, risk_experts) for row in test_rows],
            name="risk-controlled contextual expert",
            description=(
                "Uses paired rule-versus-soft outcomes inside each context and overrides only when the estimated "
                "gain remains positive after a sampling-uncertainty penalty."
            ),
            parameters={"minimum_support": 10, "confidence_z": 1.0, "learned_contexts": len(risk_experts)},
        )
    )

    reliability = _fit_context_reliability(selector_rows, rule_names)
    results.append(
        _integration_result(
            test_rows,
            [_contextual_vote_prediction(row, rule_names, scores, reliability) for row in test_rows],
            name="contextual Bayesian family vote",
            description=(
                "Estimates each rule's local reliability with shrinkage toward its global calibration accuracy, "
                "then combines predictions with family-balanced voting."
            ),
            parameters={"minimum_support": 8, "prior_strength": 12.0},
        )
    )

    if include_archived:
        behavior_index = NearestCalibrationBehaviorRule(neighbors=32)
        behavior_index.fit(selector_rows)
        results.append(
            _integration_result(
                test_rows,
                [
                    _local_competence_prediction(
                        row,
                        behavior_index._candidates(row),  # noqa: SLF001
                        rule_names,
                        scores,
                    )
                    for row in test_rows
                ],
                name="nearest-behavior local competence",
                description=(
                    "Finds calibration events with similar state evidence, sequence stage, uncertainty, n-gram "
                    "maturity, and disagreement topology, then weights rules by smoothed local accuracy."
                ),
                parameters={"neighbors": 32, "prior_strength": 12.0},
            )
        )

    gate_candidates = [(margin, families) for margin in (0.01, 0.03, 0.07, 0.15, 0.3) for families in (2, 3, 4)]
    minimum_gate_lower_bound = 0.005

    def gate_evidence(candidate: tuple[float, int]) -> tuple[float, float, int]:
        outcomes = [
            int(
                _family_consensus_prediction(
                    row,
                    rule_names,
                    maximum_margin=candidate[0],
                    minimum_families=candidate[1],
                )
                == row["actual"]
            )
            - int(row["soft_correct"])
            for row in selector_rows
        ]
        mean = sum(outcomes) / len(outcomes) if outcomes else 0.0
        variance = sum((outcome - mean) ** 2 for outcome in outcomes) / max(1, len(outcomes) - 1)
        lower_bound = mean - 0.5 * math.sqrt(variance / max(1, len(outcomes)))
        return lower_bound, mean, sum(outcome == 1 for outcome in outcomes)

    supported_gates = [
        (candidate, *gate_evidence(candidate))
        for candidate in gate_candidates
        if gate_evidence(candidate)[0] >= minimum_gate_lower_bound
    ]
    selected_gate = (
        max(supported_gates, key=lambda item: (item[2], item[0][0], -item[0][1]))
        if supported_gates
        else None
    )
    best_gate = selected_gate[0] if selected_gate else None
    results.append(
        _integration_result(
            test_rows,
            [
                (
                    _family_consensus_prediction(
                        row,
                        rule_names,
                        maximum_margin=best_gate[0],
                        minimum_families=best_gate[1],
                    )
                    if best_gate is not None
                    else row["soft_prediction"]
                )
                for row in test_rows
            ],
            name="uncertainty-gated family consensus",
            description=(
                "Overrides soft voting only when its probability margin is small and several distinct rule "
                "families independently agree on the same alternative."
            ),
            parameters={
                "maximum_soft_margin": best_gate[0] if best_gate is not None else None,
                "minimum_families": best_gate[1] if best_gate is not None else None,
                "selector_lower_bound": selected_gate[1] if selected_gate else 0.0,
                "minimum_selector_lower_bound": minimum_gate_lower_bound,
                "active": best_gate is not None,
            },
        )
    )

    learning_rates = (0.03, 0.07, 0.15, 0.3)
    best_rate = max(
        learning_rates,
        key=lambda rate: sum(
            prediction == row["actual"]
            for prediction, row in zip(
                _hedge_predictions(selector_rows, rule_names, scores, rate),
                selector_rows,
                strict=True,
            )
        ),
    )
    results.append(
        _integration_result(
            test_rows,
            _hedge_predictions(test_rows, rule_names, scores, best_rate),
            name="delayed-feedback Hedge portfolio",
            description=(
                "Starts from calibration reliability and exponentially reweights rules after each actual outcome "
                "becomes available; the current label is never used for the current prediction."
            ),
            parameters={"learning_rate": best_rate},
        )
    )

    best_sequence_rate = max(
        learning_rates,
        key=lambda rate: sum(
            prediction == row["actual"]
            for prediction, row in zip(
                _sequence_hedge_predictions(selector_rows, rule_names, scores, rate),
                selector_rows,
                strict=True,
            )
        ),
    )
    results.append(
        _integration_result(
            test_rows,
            _sequence_hedge_predictions(test_rows, rule_names, scores, best_sequence_rate),
            name="sequence-local delayed-feedback Hedge",
            description=(
                "Maintains a separate expert portfolio for each case, adapting to case-specific behavior without "
                "allowing unrelated cases to overwrite its recent reliability."
            ),
            parameters={"learning_rate": best_sequence_rate},
        )
    )

    cheating_gap = (
        sum(row["oracle_correct"] for row in test_rows) - sum(row["soft_correct"] for row in test_rows)
    ) / len(test_rows)
    for result in results:
        result["gap_recovered_fraction"] = result["net_accuracy_delta"] / cheating_gap if cheating_gap else 0.0
    if not include_archived:
        results = [result for result in results if result["name"] not in ARCHIVED_INTEGRATION_NAMES]
        active_names = {result["name"] for result in results}
        for row in test_rows:
            row["integration_predictions"] = {
                name: prediction
                for name, prediction in row.get("integration_predictions", {}).items()
                if name in active_names
            }
    return sorted(results, key=lambda item: item["accuracy"], reverse=True)


def evaluate_archived_rule_integrations(
    selector_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    rule_names: list[str],
) -> list[dict[str, Any]]:
    """Evaluate only integration methods retained for explicit ablation studies."""
    results = evaluate_rule_integrations(selector_rows, test_rows, rule_names, include_archived=True)
    return [result for result in results if result["name"] in ARCHIVED_INTEGRATION_NAMES]


def split_integration_calibration(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reserve whole calibration sequences for fitting the meta-selectors."""
    sequence_ids = list(dict.fromkeys(row["sequence_index"] for row in rows))
    if len(sequence_ids) < MIN_INTEGRATION_SEQUENCES:
        return rows, []
    selector_ids = set(sequence_ids[::3])
    base_rows = [row for row in rows if row["sequence_index"] not in selector_ids]
    selector_rows = [row for row in rows if row["sequence_index"] in selector_ids]
    return base_rows, selector_rows


def select_best_deployable(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the mandatory deployment stack when present, otherwise the accuracy leader."""
    priority = [candidate for candidate in candidates if candidate.get("deployment_priority")]
    pool = priority or candidates
    return max(pool, key=lambda item: item["accuracy"], default=None)


def _persisted_candidate_prediction(row: dict[str, Any], candidate_name: str) -> str | None:
    """Return a candidate prediction already attached during evaluation."""
    for field in ("rule_predictions", "integration_predictions", "scenario_predictions"):
        if candidate_name in row.get(field, {}):
            return row[field][candidate_name]
    return None


def build_summary(
    *,
    dataset_name: str,
    specs: list[ModelSpec],
    train_rows: int,
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    rule_integrations: list[dict[str, Any]] | None = None,
    selector_calibration_events: int = 0,
) -> dict[str, Any]:
    """Build aggregate results and oracle-gap slices for persistence and display."""
    def scored_strategy(name: str, prediction_key: str) -> dict[str, Any]:
        calibration_correct = sum(row[prediction_key] == row["actual"] for row in calibration_rows)
        test_correct = sum(row[prediction_key] == row["actual"] for row in test_rows)
        return {
            "name": name,
            "accuracy": test_correct / len(test_rows) if test_rows else 0.0,
            "correct": test_correct,
            "total": len(test_rows),
            "calibration_accuracy": calibration_correct / len(calibration_rows) if calibration_rows else None,
            "calibration_correct": calibration_correct,
            "calibration_total": len(calibration_rows),
        }

    strategies = [
        scored_strategy("soft voting", "soft_prediction"),
        scored_strategy("adaptive voting", "adaptive_prediction"),
        scored_strategy("cheating voting", "oracle_prediction"),
    ]
    per_model = []
    for spec in specs:
        correct = sum(
            next(model["correct"] for model in row["models"] if model["name"] == spec.name) for row in test_rows
        )
        calibration_correct = sum(
            next(model["correct"] for model in row["models"] if model["name"] == spec.name)
            for row in calibration_rows
        )
        per_model.append(
            {
                "name": spec.name,
                "accuracy": correct / len(test_rows) if test_rows else 0.0,
                "correct": correct,
                "total": len(test_rows),
                "calibration_accuracy": calibration_correct / len(calibration_rows) if calibration_rows else None,
                "calibration_correct": calibration_correct,
                "calibration_total": len(calibration_rows),
            }
        )
    gap_rows = [row for row in test_rows if row["oracle_gap"]]
    rule_scenarios = evaluate_rule_scenarios(test_rows, hypotheses)
    integrations = rule_integrations or []
    candidate_results = [
        *hypotheses,
        *(scenario for scenario in rule_scenarios if not scenario["oracle"]),
        *integrations,
    ]
    best_candidate = select_best_deployable(candidate_results)
    best_candidate_name = best_candidate["name"] if best_candidate else "soft voting"
    for row in test_rows:
        best_prediction = (
            _persisted_candidate_prediction(row, best_candidate_name) if best_candidate else row["soft_prediction"]
        )
        if best_prediction is None:
            msg = f"Missing persisted prediction for best deployable candidate {best_candidate_name!r}."
            raise RuntimeError(msg)
        row["best_deployable_method"] = best_candidate_name
        row["best_deployable_prediction"] = best_prediction
    soft_accuracy = strategies[0]["accuracy"]
    oracle_accuracy = strategies[2]["accuracy"]
    recoverable_gap = oracle_accuracy - soft_accuracy
    recovered_accuracy = max(0.0, best_candidate["accuracy"] - soft_accuracy) if best_candidate else 0.0
    minimum_complexity = min((spec.complexity for spec in specs), default=None)
    minimum_complexity_models = [spec.name for spec in specs if spec.complexity == minimum_complexity]
    mandatory_integration = next(
        (result for result in integrations if result.get("deployment_priority")),
        None,
    )
    enforcement_audit = (
        mandatory_integration.get("parameters", {}).get("enforcement_audit", {}) if mandatory_integration else {}
    )
    boost_contract = {
        "minimum_complexity": minimum_complexity,
        "minimum_complexity_models": minimum_complexity_models,
        "unique_minimum_complexity_model": len(minimum_complexity_models) == 1,
        "default_generalist_is_bag": minimum_complexity_models == ["bag"],
        "independent_stack_present": mandatory_integration is not None,
        "independent_stack_is_best_deployable": bool(
            mandatory_integration and best_candidate_name == mandatory_integration["name"]
        ),
        **enforcement_audit,
    }
    return {
        "dataset": dataset_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": [spec.name for spec in specs],
        "train_events": train_rows,
        "calibration_events": len(calibration_rows),
        "test_events": len(test_rows),
        "strategies": strategies,
        "per_model": per_model,
        "hypotheses": hypotheses,
        "rule_scenarios": rule_scenarios,
        "rule_integrations": integrations,
        "selector_calibration_events": selector_calibration_events,
        "best_rule_result": best_candidate,
        "independent_boost_contract": boost_contract,
        # Kept as a result-schema compatibility alias for existing consumers.
        "adaptive_recovery_contract": boost_contract,
        "recoverable_gap": recoverable_gap,
        "recovered_gap_with_best_rule": recovered_accuracy,
        "recovered_gap_fraction": recovered_accuracy / recoverable_gap if recoverable_gap else 0.0,
        "oracle_gap_events": len(gap_rows),
        "oracle_gap_rate": len(gap_rows) / len(test_rows) if test_rows else 0.0,
        "soft_failures": sum(not row["soft_correct"] for row in test_rows),
        "recoverable_soft_failures": len(gap_rows),
        "correct_models_when_soft_fails": dict(
            Counter(model_name for row in gap_rows for model_name in row["correct_models"])
        ),
    }


def save_results(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Persist compact summary JSON and event-level JSON Lines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "events.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


def run_investigation(
    *,
    dataset_name: str,
    data_prop: float,
    windows: tuple[int, ...],
    output_dir: Path,
    seed: int = 0,
) -> Path:
    """Load data, train models, calibrate hypotheses, evaluate, and save results."""
    args = argparse.Namespace(data=dataset_name)
    resolved_name, dataset, _ = resolve_dataset_from_args(args)
    sequences = transform_to_seqs(dataset)
    if not 0 < data_prop <= 1:
        msg = "data_prop must be in (0, 1]."
        raise ValueError(msg)
    sequences = sequences[: max(1, int(len(sequences) * data_prop))]
    train_sequences, remainder = split_sequence_data(sequences, test_size=0.3, random_shuffle=True, seed=seed)
    calibration_sequences, test_sequences = split_sequence_data(
        remainder,
        test_size=0.5,
        random_shuffle=True,
        seed=seed,
    )
    train_sequences = add_stop_to_sequences(train_sequences, DEFAULT_CONFIG["stop_symbol"])
    calibration_sequences = add_stop_to_sequences(calibration_sequences, DEFAULT_CONFIG["stop_symbol"])
    test_sequences = add_stop_to_sequences(test_sequences, DEFAULT_CONFIG["stop_symbol"])

    specs = default_model_specs(windows)
    investigator = VotingInvestigator(specs=specs, config={"top_k": 3, "include_stop": True})
    investigator.train(train_sequences)
    calibration_rows = investigator.diagnose(calibration_sequences, split="calibration")
    test_rows = investigator.diagnose(test_sequences, split="test")

    # Reserve complete cases for the meta-rule.  The constituent rules learn
    # their parameters on the remaining calibration cases; their resulting
    # selector predictions are then genuinely out-of-fit evidence for choosing
    # an integration policy.  Finally, constituents are refit on every
    # calibration case before touching the test set.
    rule_fit_rows, selector_rows = split_integration_calibration(calibration_rows)
    selector_hypotheses = evaluate_hypotheses(rule_fit_rows, selector_rows)
    hypotheses = evaluate_hypotheses(calibration_rows, test_rows)
    integration_rule_names = [
        hypothesis["name"]
        for hypothesis in selector_hypotheses
        # The transient-policy selector already aggregates a family of 200
        # schedules.  Giving that aggregate another family vote would double
        # count the same recovery signal.
        if hypothesis["name"] != "calibrated transient generalist pool"
    ]
    selector_integrations = evaluate_archived_rule_integrations(
        selector_rows,
        test_rows,
        integration_rule_names,
    )
    # This is the most conservative meta-policy: it changes the soft leader
    # only when several independent rule families agree and soft voting is
    # uncertain.  It is selected from out-of-fit calibration cases, not from
    # held-out accuracy.  Other integration methods remain inspection-only.
    active_integration_names = {"uncertainty-gated family consensus"}
    rule_integrations = [
        result for result in selector_integrations if result["name"] in active_integration_names
    ]
    for row in test_rows:
        row["integration_predictions"] = {
            name: prediction
            for name, prediction in row.get("integration_predictions", {}).items()
            if name in active_integration_names
        }
        row["integration_diagnostics"] = {
            name: diagnostics
            for name, diagnostics in row.get("integration_diagnostics", {}).items()
            if name in active_integration_names
        }
    soft_failure_analysis = build_soft_failure_analysis(calibration_rows, test_rows)
    summary = build_summary(
        dataset_name=resolved_name,
        specs=specs,
        train_rows=sum(len(sequence) for sequence in train_sequences),
        calibration_rows=calibration_rows,
        test_rows=test_rows,
        hypotheses=hypotheses,
        rule_integrations=rule_integrations,
        selector_calibration_events=len(selector_rows),
    )
    summary["soft_failure_analysis"] = soft_failure_analysis
    summary["run_config"] = {
        "data_prop": data_prop,
        "windows": list(windows),
        "seed": seed,
    }
    save_results(output_dir, summary, test_rows)
    return output_dir


def dataset_result_dir(results_root: Path, dataset_name: str) -> Path:
    """Return the stable result directory assigned to one dataset."""
    safe_name = "".join(character for character in dataset_name if character.isalnum() or character in "-_")
    if not safe_name:
        msg = f"Dataset name {dataset_name!r} cannot be used as a result directory."
        raise ValueError(msg)
    return results_root / safe_name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run an investigation and save event-level results.")
    run_parser.add_argument("--data", default="Sepsis_Cases")
    run_parser.add_argument("--data-prop", type=float, default=1.0)
    run_parser.add_argument("--windows", default="2,3,4")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--dashboard", action="store_true")
    run_parser.add_argument("--port", type=int, default=8050)
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run multiple datasets into DATASET-named directories under one result root.",
    )
    benchmark_parser.add_argument("--data", nargs="+", default=list(DEFAULT_BENCHMARK_DATASETS))
    benchmark_parser.add_argument("--data-prop", type=float, default=0.9)
    benchmark_parser.add_argument("--windows", default="2,3,4,5,6")
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    benchmark_parser.add_argument("--skip-existing", action="store_true")
    dashboard_parser = subparsers.add_parser("dashboard", help="Open a dashboard for saved results.")
    dashboard_parser.add_argument("results", type=Path)
    dashboard_parser.add_argument("--port", type=int, default=8050)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the standalone investigation CLI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    if args.command == "dashboard":
        from logicsponge.processmining.voting_dashboard import run_dashboard

        run_dashboard(args.results, port=args.port)
        return

    if args.command == "benchmark":
        windows = tuple(int(value) for value in args.windows.split(",") if value.strip())
        dataset_names = [name for value in args.data for name in value.split(",") if name]
        for dataset_name in dataset_names:
            output_dir = dataset_result_dir(args.output_root, dataset_name)
            if args.skip_existing and (output_dir / "summary.json").exists():
                logger.info("Skipping existing result for %s at %s", dataset_name, output_dir.resolve())
                continue
            logger.info("Running voting investigation for %s", dataset_name)
            run_investigation(
                dataset_name=dataset_name,
                data_prop=args.data_prop,
                windows=windows,
                output_dir=output_dir,
                seed=args.seed,
            )
        logger.info("Cross-dataset results saved under %s", args.output_root.resolve())
        return

    windows = tuple(int(value) for value in args.windows.split(",") if value.strip())
    output_dir = args.output or dataset_result_dir(DEFAULT_RESULTS_ROOT, args.data)
    result_path = run_investigation(
        dataset_name=args.data,
        data_prop=args.data_prop,
        windows=windows,
        output_dir=output_dir,
        seed=args.seed,
    )
    logger.info("Investigation saved to %s", result_path.resolve())
    if args.dashboard:
        from logicsponge.processmining.voting_dashboard import run_dashboard

        run_dashboard(result_path, port=args.port)


if __name__ == "__main__":
    main()
