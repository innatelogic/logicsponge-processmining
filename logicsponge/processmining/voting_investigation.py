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
from itertools import combinations
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
from logicsponge.processmining.miners import BasicMiner, SoftVoting, StreamingMiner
from logicsponge.processmining.types import ComposedState, Event, Metrics
from logicsponge.processmining.utils import metrics_prediction, resolve_dataset_from_args

logger = logging.getLogger(__name__)
VISIT_FEW_MAX = 2
VISIT_MANY_MAX = 9


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Description and constructor for one constituent model."""

    name: str
    complexity: int
    factory: Callable[[], StreamingMiner]


def default_model_specs(windows: Iterable[int] = (2, 3, 4)) -> list[ModelSpec]:
    """Return the default Bag/FPT/N-gram model family used by voting benchmarks."""
    specs = [
        ModelSpec("bag", 0, lambda: BasicMiner(algorithm=Bag())),
        ModelSpec("fpt", 1, lambda: BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10))),
    ]
    specs.extend(
        ModelSpec(
            f"ngram_{window}",
            window,
            lambda window=window: BasicMiner(algorithm=NGram(window_length=window)),
        )
        for window in windows
    )
    return specs


def _display(value: Any) -> str:
    """Return a stable display representation for activities and states."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return repr(value)


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


class HighestConfidenceRule(DecisionRule):
    """Select the model with the highest top-prediction probability."""

    name = "highest confidence"
    family = "confidence"

    def select(self, row: dict[str, Any]) -> str:
        return max(row["models"], key=lambda model: (model["confidence"], -model["index"]))["name"]


class GlobalAccuracyRule(DecisionRule):
    """Always select the best model on calibration data."""

    name = "best calibration accuracy"
    family = "accuracy"

    def __init__(self) -> None:
        self.model_name = ""

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.model_name = _best_model(rows)

    def select(self, row: dict[str, Any]) -> str:  # noqa: ARG002
        return self.model_name


class GroupedAccuracyRule(DecisionRule):
    """Select the best calibrated model for a feature-defined event group."""

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


def _smoothed_rate(correct: float, total: float, prior: float, prior_weight: float = 4.0) -> float:
    """Return a shrinkage estimate that protects sparse contexts."""
    return (correct + prior * prior_weight) / (total + prior_weight)


class HierarchicalReliabilityRule(DecisionRule):
    """
    Select a model using fine-to-coarse reliability contexts.

    Contexts back off from state + recent pattern + agreement + position to
    state, agreement, and finally global model reliability. Every score is
    shrunk toward the model's global calibration accuracy.
    """

    family = "hierarchical"

    def __init__(self, minimum_support: int = 3, prior_weight: float = 4.0) -> None:
        self.minimum_support = minimum_support
        self.prior_weight = prior_weight
        self.name = f"hierarchical reliability (support {minimum_support})"
        self.global_scores: dict[str, tuple[int, int]] = {}
        self.context_scores: list[dict[tuple[str, ...], tuple[int, int]]] = []
        self.default_model = ""

    def _contexts(self, row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        model_name = model["name"]
        return [
            (
                model_name,
                model["state"],
                row["suffix_3"],
                str(row["agreement_count"]),
                str(int(row["position"]) // 5),
            ),
            (model_name, model["state"], row["suffix_2"], str(row["agreement_count"])),
            (model_name, model["state"], str(row["agreement_count"])),
            (model_name, model["state"]),
            (model_name, str(row["agreement_count"])),
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

    def __init__(self, minimum_support: int = 4) -> None:
        super().__init__(minimum_support=minimum_support)
        self.name = f"disagreement profile reliability (support {minimum_support})"

    def _contexts(self, row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        profile = ";".join(
            f"{candidate['name']}="
            f"{'soft' if candidate['prediction'] == row['soft_prediction'] else 'off-soft'}"
            f"/{'consensus' if candidate['prediction'] == row['consensus_prediction'] else 'off-consensus'}"
            for candidate in row["models"]
        )
        name = model["name"]
        return [
            (name, profile, row["suffix_2"], str(row["agreement_count"])),
            (name, profile, str(row["agreement_count"])),
            (name, profile),
            (name, str(row["agreement_count"])),
            (name,),
        ]


class PredictionOutcomeReliabilityRule(HierarchicalReliabilityRule):
    """Use a model's own predicted activity as a calibrated outcome feature."""

    family = "prediction outcome"

    def __init__(self, minimum_support: int = 3) -> None:
        super().__init__(minimum_support=minimum_support)
        self.name = f"prediction/state reliability (support {minimum_support})"

    def _contexts(self, row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        name = model["name"]
        prediction = model["prediction"]
        confidence = _confidence_bin(model["confidence"])
        return [
            (name, model["state"], prediction, confidence, str(row["agreement_count"])),
            (name, model["state"], prediction),
            (name, prediction, confidence),
            (name, prediction),
            (name, model["state"]),
            (name,),
        ]


class PrefixBackoffRule(HierarchicalReliabilityRule):
    """Match repeated full prefixes before backing off to suffix contexts."""

    family = "prefix"

    def __init__(self, minimum_support: int = 2) -> None:
        super().__init__(minimum_support=minimum_support)
        self.name = f"prefix/state reliability (support {minimum_support})"

    def _contexts(self, row: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, ...]]:
        name = model["name"]
        return [
            (name, row["prefix_text"]),
            (name, row["prefix_text"], str(row["agreement_count"])),
            (name, row["suffix_3"], str(row["position"] // 5)),
            (name, row["suffix_2"]),
            (name, model["state"]),
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
        for model in row["models"]:
            for level, context in enumerate(self._contexts(row, model)):
                correct, total = self.context_scores[level].get(context, (0.0, 0.0))
                self.context_scores[level][context] = (
                    self.decay * correct + float(model["correct"]),
                    self.decay * total + 1.0,
                )


class CalibratedDecisionListRule(DecisionRule):
    """Apply the strongest calibrated condition, then a hierarchical fallback."""

    family = "decision list"

    def __init__(self, minimum_support: int = 10) -> None:
        self.minimum_support = minimum_support
        self.name = f"calibrated decision list (support {minimum_support})"
        self.conditions: list[dict[str, Any]] = []
        self.fallback = HierarchicalReliabilityRule(minimum_support=3)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.fallback.fit(rows)
        self.conditions, _ = discover_condition_hypotheses(
            rows,
            [],
            minimum_support=self.minimum_support,
        )
        self.conditions.sort(
            key=lambda condition: (
                condition["calibration_gain"] * math.sqrt(condition["calibration_support"]),
                condition["calibration_gain"],
            ),
            reverse=True,
        )

    def select(self, row: dict[str, Any]) -> str:
        features = _condition_features(row)
        for condition in self.conditions:
            if features.get(condition["feature"]) == condition["value"]:
                return condition["recommended_model"]
        return self.fallback.select(row)


def _new_advanced_hypotheses() -> list[DecisionRule]:
    """Return higher-capacity selectors kept separate from legacy rules."""
    return [
        HierarchicalReliabilityRule(),
        ConsensusHierarchicalRule(),
        ConfidenceStateReliabilityRule(),
        DisagreementProfileRule(),
        PredictionOutcomeReliabilityRule(),
        PrefixBackoffRule(),
        CalibratedDecisionListRule(),
        CompositeDecisionListRule(),
        NearestCalibrationBehaviorRule(),
        StackedRulePortfolio(),
        DelayedFeedbackAdaptiveRule(),
    ]


def default_hypotheses() -> list[DecisionRule]:
    """Return an initial, intentionally broad set of selection hypotheses."""
    rules: list[DecisionRule] = [HighestConfidenceRule(), GlobalAccuracyRule()]
    rules.extend(
        GroupedAccuracyRule(
            name=f"position bucket {width} (support 5)",
            family="position",
            feature=lambda row, width=width: str(row["position"] // width),
            minimum_support=5,
        )
        for width in (2, 5, 10)
    )
    rules.extend(
        GroupedAccuracyRule(
            name=f"suffix length {length} (support {support})",
            family="pattern",
            feature=lambda row, length=length: row[f"suffix_{length}"],
            minimum_support=support,
        )
        for length in (1, 2, 3)
        for support in (3, 10)
    )
    rules.extend(StateAccuracyRule(support) for support in (3, 5, 10))
    rules.extend(AgreementRule(threshold, fallback) for threshold in (2, 3) for fallback in ("confidence", "accuracy"))
    rules.extend(_new_advanced_hypotheses())
    return rules


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
        for sequence_index, sequence in enumerate(sequences):
            if not sequence:
                continue
            states: list[ComposedState | None] = [model.initial_state for model in self.models]
            prefix: list[str] = []
            sequence_id = _display(sequence[0]["case_id"])
            for position, event in enumerate(sequence):
                actual = _display(event["activity"])
                metrics_list = [model.state_metrics(state) for model, state in zip(self.models, states, strict=True)]
                model_rows = self._model_rows(metrics_list, actual)
                predictions = [model["prediction"] for model in model_rows if model["prediction"]]
                counts = Counter(predictions)
                consensus = counts.most_common(1)[0][0] if counts else ""
                agreement_count = counts[consensus] if consensus else 0

                soft_prediction = self._ensemble_prediction(self.soft_voting, metrics_list)
                correct_models = [model["name"] for model in model_rows if model["correct"]]
                oracle_prediction = actual if correct_models else soft_prediction

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
                    "soft_prediction": soft_prediction,
                    "soft_correct": soft_prediction == actual,
                    "oracle_prediction": oracle_prediction,
                    "oracle_correct": oracle_prediction == actual,
                    "oracle_model": correct_models[0] if correct_models else "",
                    "oracle_gap": oracle_prediction == actual and soft_prediction != actual,
                }
                rows.append(row)
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
            model_rows.append(
                {
                    "index": index,
                    "name": spec.name,
                    "complexity": spec.complexity,
                    "state": _display(state),
                    "state_visits": int(visits),
                    "prediction": predicted,
                    "confidence": float(prediction.get("probability", 0.0)) if prediction is not None else 0.0,
                    "correct": predicted == actual,
                }
            )
        return model_rows

    def _ensemble_prediction(self, ensemble: SoftVoting, metrics_list: list[Metrics]) -> str:
        probs = ensemble.voting_probs([metrics["probs"] for metrics in metrics_list])
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

    for rule in selected_rules:
        rule.fit(calibration_rows)
        correct = 0
        selected_counts: Counter[str] = Counter()
        for row in test_rows:
            model_name = rule.select(row)
            selected_model = next((model for model in row["models"] if model["name"] == model_name), None)
            prediction = selected_model["prediction"] if selected_model is not None else ""
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
                "accuracy": correct / len(test_rows) if test_rows else 0.0,
                "correct": correct,
                "total": len(test_rows),
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
        "prediction/state": "prediction outcome",
        "prefix/state": "prefix",
        "calibrated decision": "decision list",
        "composite decision": "decision list",
        "nearest calibration": "nearest behavior",
        "stacked rule": "stacked portfolio",
        "delayed-feedback": "adaptive",
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
    """Extract pre-label categorical features used to discover conditional rules."""
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
    }
    for length in (1, 2, 3):
        suffix = row[f"suffix_{length}"]
        if suffix:
            features[f"last {length} activit{'y' if length == 1 else 'ies'}"] = suffix

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
    return features


def _condition_family(feature: str) -> str:
    if "activit" in feature:
        return "pattern"
    if "position" in feature or feature == "sequence stage":
        return "position"
    if "agreement" in feature or "consensus" in feature or "disagreement" in feature or " vs " in feature:
        return "agreement"
    return "other"


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
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        for feature, value in _condition_features(row).items():
            grouped[(feature, value)].append(row)

    candidates: list[dict[str, Any]] = []
    for (feature, value), group_rows in grouped.items():
        if len(group_rows) < minimum_support:
            continue
        model_names = [model["name"] for model in group_rows[0]["models"]]
        model_accuracies = {
            model_name: sum(_model_prediction(row, model_name) == row["actual"] for row in group_rows) / len(group_rows)
            for model_name in model_names
        }
        recommended_model = max(model_names, key=lambda name: (model_accuracies[name], -model_names.index(name)))
        soft_accuracy = sum(row["soft_correct"] for row in group_rows) / len(group_rows)
        calibration_gain = model_accuracies[recommended_model] - soft_accuracy
        if calibration_gain <= 0:
            continue
        condition_id = f"{feature}={value}"
        candidates.append(
            {
                "id": condition_id,
                "name": f"When {feature} is {value}, prefer {recommended_model}",
                "family": _condition_family(feature),
                "feature": feature,
                "value": value,
                "recommended_model": recommended_model,
                "calibration_support": len(group_rows),
                "calibration_soft_accuracy": soft_accuracy,
                "calibration_model_accuracy": model_accuracies[recommended_model],
                "calibration_gain": calibration_gain,
                "weight": calibration_gain * math.sqrt(len(group_rows)),
            }
        )

    candidates.sort(key=lambda item: (item["weight"], item["calibration_gain"]), reverse=True)
    candidates = candidates[:maximum_hypotheses]
    for row in test_rows:
        feature_values = _condition_features(row)
        row["condition_matches"] = [
            candidate["id"]
            for candidate in candidates
            if feature_values.get(candidate["feature"]) == candidate["value"]
        ]

    for candidate in candidates:
        matching_rows = [row for row in test_rows if candidate["id"] in row["condition_matches"]]
        predictions = [_model_prediction(row, candidate["recommended_model"]) for row in matching_rows]
        impact = _override_impact(
            matching_rows,
            predictions,
            name=candidate["name"],
            family=candidate["family"],
        )
        candidate.update(
            {
                "test_support": len(matching_rows),
                "soft_errors": impact["soft_errors"],
                "recoveries": impact["recoveries"],
                "harms": impact["harms"],
                "net_correct": impact["net_correct"],
                "conditional_delta": impact["net_correct"] / len(matching_rows) if matching_rows else 0.0,
                "soft_error_recall": impact["soft_error_recall"],
                "decisive_precision": impact["decisive_precision"],
            }
        )
    return sorted(candidates, key=lambda item: (item["net_correct"], item["recoveries"]), reverse=True), source


def _discover_composite_conditions(
    rows: list[dict[str, Any]],
    *,
    minimum_support: int = 8,
    maximum_hypotheses: int = 250,
) -> list[dict[str, Any]]:
    """Learn positive two-clause conditions from calibration events."""
    if not rows:
        return []
    feature_names = sorted(_condition_features(rows[0]))
    grouped: defaultdict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        features = _condition_features(row)
        for feature_a, feature_b in combinations(feature_names, 2):
            value_a = features[feature_a]
            value_b = features[feature_b]
            if not value_a or not value_b:
                continue
            grouped[((feature_a, value_a), (feature_b, value_b))].append(row)

    candidates: list[dict[str, Any]] = []
    for clauses, group_rows in grouped.items():
        if len(group_rows) < minimum_support:
            continue
        model_names = [model["name"] for model in group_rows[0]["models"]]
        accuracies = {
            name: sum(_model_prediction(row, name) == row["actual"] for row in group_rows) / len(group_rows)
            for name in model_names
        }
        recommended_model = max(model_names, key=lambda name: (accuracies[name], -model_names.index(name)))
        soft_accuracy = sum(row["soft_correct"] for row in group_rows) / len(group_rows)
        gain = accuracies[recommended_model] - soft_accuracy
        if gain <= 0:
            continue
        candidates.append(
            {
                "id": " AND ".join(f"{feature}={value}" for feature, value in clauses),
                "name": "When "
                + " and ".join(f"{feature} is {value}" for feature, value in clauses)
                + f", prefer {recommended_model}",
                "family": "composite",
                "clauses": clauses,
                "recommended_model": recommended_model,
                "calibration_support": len(group_rows),
                "calibration_gain": gain,
                "weight": gain * math.sqrt(len(group_rows)),
            }
        )
    candidates.sort(key=lambda item: (item["weight"], item["calibration_gain"]), reverse=True)
    return candidates[:maximum_hypotheses]


class CompositeDecisionListRule(DecisionRule):
    """Use calibrated two-clause interactions before a hierarchical fallback."""

    family = "composite"

    def __init__(self, minimum_support: int = 8) -> None:
        self.minimum_support = minimum_support
        self.name = f"composite decision list (support {minimum_support})"
        self.conditions: list[dict[str, Any]] = []
        self.fallback = HierarchicalReliabilityRule(minimum_support=3)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.fallback.fit(rows)
        self.conditions = _discover_composite_conditions(rows, minimum_support=self.minimum_support)

    def select(self, row: dict[str, Any]) -> str:
        features = _condition_features(row)
        for condition in self.conditions:
            if all(features.get(feature) == value for feature, value in condition["clauses"]):
                return condition["recommended_model"]
        return self.fallback.select(row)


class NearestCalibrationBehaviorRule(DecisionRule):
    """Transfer model reliability from the nearest observed calibration behaviors."""

    family = "nearest behavior"

    def __init__(self, neighbors: int = 32) -> None:
        self.neighbors = neighbors
        self.name = f"nearest calibration behavior (k {neighbors})"
        self.rows: list[dict[str, Any]] = []
        self.global_accuracy: dict[str, float] = {}
        self.default_model = ""

    @staticmethod
    def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        distance = abs(float(left["relative_position"]) - float(right["relative_position"]))
        distance += 0.5 * abs(int(left["agreement_count"]) - int(right["agreement_count"]))
        distance += 0.5 * abs(int(left["distinct_prediction_count"]) - int(right["distinct_prediction_count"]))
        distance += 1.0 * (left["suffix_3"] != right["suffix_3"])
        distance += 0.5 * (left["suffix_2"] != right["suffix_2"])
        for left_model, right_model in zip(left["models"], right["models"], strict=True):
            distance += 0.8 * (left_model["prediction"] != right_model["prediction"])
            distance += 0.25 * (left_model["state"] != right_model["state"])
            distance += 0.25 * abs(left_model["confidence"] - right_model["confidence"])
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

    def select(self, row: dict[str, Any]) -> str:
        if not self.rows:
            return self.default_model
        nearest = sorted(
            ((self._distance(row, candidate), candidate) for candidate in self.rows),
            key=lambda item: item[0],
        )[: self.neighbors]
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
            PredictionOutcomeReliabilityRule(3),
            DisagreementProfileRule(4),
        ]

    @staticmethod
    def _contexts(row: dict[str, Any]) -> list[tuple[str, ...]]:
        features = _condition_features(row)
        return [
            (features.get("last 2 activities", ""), features["agreement count"], str(int(row["position"]) // 5)),
            (features.get("last 2 activities", ""), features["agreement count"]),
            (features["consensus strength"], features["sequence stage"]),
            (features["agreement count"],),
            ("global",),
        ]

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
                selected = _model_prediction(row, rule.select(row))
                correct = int(selected == row["actual"])
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

    def select(self, row: dict[str, Any]) -> str:
        rule_name = self.global_rule
        for context in self._contexts(row):
            if context in self.rule_by_context:
                rule_name = self.rule_by_context[context]
                break
        rule = self.rule_by_name.get(rule_name)
        return rule.select(row) if rule is not None else self.rules[0].select(row)


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
    "delayed-feedback adaptive (decay 0.94)",
)
ADVANCED_RULE_SET = (
    "consensus hierarchy ≥ 3 (support 3)",
    "hierarchical reliability (support 3)",
    "confidence/state reliability (support 5)",
    "disagreement profile reliability (support 4)",
    "prediction/state reliability (support 3)",
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


def build_summary(
    *,
    dataset_name: str,
    specs: list[ModelSpec],
    train_rows: int,
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build aggregate results and oracle-gap slices for persistence and display."""
    strategies = [
        {"name": "soft voting", "accuracy": _accuracy(test_rows, "soft_prediction")},
        {"name": "cheating voting", "accuracy": _accuracy(test_rows, "oracle_prediction")},
    ]
    per_model = []
    for spec in specs:
        correct = sum(
            next(model["correct"] for model in row["models"] if model["name"] == spec.name) for row in test_rows
        )
        per_model.append(
            {"name": spec.name, "accuracy": correct / len(test_rows) if test_rows else 0.0, "correct": correct}
        )
    gap_rows = [row for row in test_rows if row["oracle_gap"]]
    rule_scenarios = evaluate_rule_scenarios(test_rows, hypotheses)
    candidate_results = [*hypotheses, *(scenario for scenario in rule_scenarios if not scenario["oracle"])]
    best_candidate = max(candidate_results, key=lambda item: item["accuracy"], default=None)
    soft_accuracy = strategies[0]["accuracy"]
    oracle_accuracy = strategies[1]["accuracy"]
    recoverable_gap = oracle_accuracy - soft_accuracy
    recovered_accuracy = max(0.0, best_candidate["accuracy"] - soft_accuracy) if best_candidate else 0.0
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
        "best_rule_result": best_candidate,
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
    hypotheses = evaluate_hypotheses(calibration_rows, test_rows)
    soft_failure_analysis = build_soft_failure_analysis(calibration_rows, test_rows)
    summary = build_summary(
        dataset_name=resolved_name,
        specs=specs,
        train_rows=sum(len(sequence) for sequence in train_sequences),
        calibration_rows=calibration_rows,
        test_rows=test_rows,
        hypotheses=hypotheses,
    )
    summary["soft_failure_analysis"] = soft_failure_analysis
    save_results(output_dir, summary, test_rows)
    return output_dir


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

    windows = tuple(int(value) for value in args.windows.split(",") if value.strip())
    output_dir = args.output or Path("results") / f"voting-investigation-{time.strftime('%Y%m%d-%H%M%S')}"
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
