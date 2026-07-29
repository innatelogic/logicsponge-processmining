"""Interactive Dash application for saved voting-investigation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html

from logicsponge.processmining.voting_investigation import (
    ARCHIVED_INTEGRATION_NAMES,
    analyze_rule_impacts,
    archived_hypotheses,
    build_soft_failure_analysis,
    default_hypotheses,
    evaluate_hypotheses,
    evaluate_rule_integrations,
    evaluate_rule_scenarios,
    evaluate_weighted_conditions,
    select_best_deployable,
)

OVERVIEW_GRAPH_CONFIG = {"responsive": True, "displayModeBar": False}
ARCHIVED_RULE_NAMES = frozenset(rule.name for rule in archived_hypotheses())
ACTIVE_INTEGRATION_NAMES = frozenset({"uncertainty-gated family consensus"})
CONDITION_FEATURE_DESCRIPTIONS = {
    "consensus strength": "Whether the most common constituent top prediction has no majority, a majority, or unanimity.",
    "agreement count": "How many constituent models share the most common top prediction.",
    "prediction diversity": "How many distinct non-empty constituent top predictions appear at this event.",
    "sequence stage": "Relative location in the case: early (first third), middle (second third), or late (final third).",
    "position bucket 2": "Zero-based event position grouped into consecutive two-event buckets.",
    "position bucket 5": "Zero-based event position grouped into consecutive five-event buckets.",
    "prediction topology": "Model predictions encoded by equality pattern, not activity names; equal codes mean the models predicted the same activity.",
    "state evidence profile": "Training-visit support bands for each constituent's current learned process state, in model order.",
    "n-gram maturity profile": "For each model, whether the observed prefix is at least as long as that model's complexity/order.",
    "previous soft correctness": "Whether the preceding soft-voting prediction in the same case was correct. It is unknown at the first event.",
    "previous wrong model count": "Number of constituent top predictions that were wrong at the preceding event in the same case.",
    "previous empty prediction count": "Number of constituents that had no top prediction at the preceding event in the same case.",
    "current empty prediction count": "Number of constituents that have no current top prediction.",
    "soft margin bin": "Bin for the soft vote's top-probability minus second-probability gap; smaller means a less decisive soft prediction.",
    "soft entropy bin": "Bin for normalized entropy of the soft distribution; larger means probability is spread across more activities.",
    "model confidence spread bin": "Bin for the range between the highest and lowest constituent top-prediction confidences.",
    "consensus model set": "The constituent models whose top prediction equals the consensus prediction.",
    "soft-disagreement model set": "The constituent models whose top prediction differs from soft voting's top prediction.",
}
RULE_DESCRIPTIONS = {
    "transient Bag favoritism": (
        "After a soft-vote error that Bag got right, applies a three-step positive, decaying multiplier only to "
        "the structurally simplest model."
    ),
    "transient recovery calibrated by age": (
        "Enables generalist recovery only at recovery ages with positive paired calibration gain over soft voting."
    ),
    "transient recovery calibrated by structural": (
        "Uses supported recovery-age, sequence-stage, agreement, margin, and entropy contexts to gate recovery."
    ),
    "transient recovery calibrated by distribution": (
        "Selects a finite generalist-distribution multiplier separately for each recovery age."
    ),
    "ngram correctness-streak": (
        "Uses delayed consecutive correctness to reweight N-grams after a soft-vote error: low streaks are "
        "suppressed, half-window streaks are neutral, and near-mature streaks receive a positive boost."
    ),
    "state-evidence topology": (
        "Compares joint process states, state occurrence counts, n-gram maturity, relative confidence/entropy, "
        "and canonical model-agreement topology without keying on activity names."
    ),
    "evidence-weighted distribution": (
        "Learns local model weights from smoothed probability quality in structural regimes, then merges the full "
        "next-activity distributions."
    ),
    "structural branching ensemble": (
        "Uses an internal calibration holdout to gate between state-based model selection, locally weighted "
        "distribution merging, robust pooling, and soft-voting fallback."
    ),
    "calibrated soft rank": (
        "Uses the second- or third-ranked soft-voting activity only in calibration contexts where that rank "
        "outperformed the normal soft prediction; otherwise it keeps soft voting."
    ),
    "calibrated model rank": (
        "Switches to the globally second- or third-best constituent model only in calibration contexts where "
        "that specialist beat soft voting."
    ),
    "previous-outcome": (
        "Uses the previous observed activity, previous soft correctness, and which models were previously correct "
        "to select a current specialist."
    ),
    "distribution-shape": (
        "Scores models from calibrated entropy, top-probability margin, divergence from the soft distribution, "
        "agreement, and empty-prediction regime."
    ),
    "median probability": "Pools each activity probability with a robust median across constituent models.",
    "trimmed probability": "Drops the smallest and largest per-activity probabilities, then averages the rest.",
    "product probability": "Uses a geometric probability pool, rewarding activities supported across models.",
    "calibrated probability": (
        "Chooses among soft, median, trimmed-mean, and product pooling according to a calibration-learned "
        "distribution regime."
    ),
    "Borda positional rank aggregation": (
        "Treats models as voters, activities as alternatives, and aggregates their probability orderings with "
        "tie-aware positional Borda scores."
    ),
    "Copeland pairwise rank aggregation": (
        "Scores activities by pairwise majority wins across constituent model rankings, with half a point for ties."
    ),
    "Maximin pairwise rank aggregation": (
        "Selects the activity whose worst pairwise majority margin across model rankings is strongest."
    ),
    "confusion residual": (
        "Corrects recurring soft-prediction/activity confusion pairs only in supported suffix, transition, and "
        "uncertainty contexts."
    ),
    "run-cycle residual": (
        "Uses repeated-activity run length and compressed cycle phase to detect systematic next-activity errors."
    ),
    "state residual": (
        "Learns direct activity corrections for supported joint constituent-state and prediction signatures."
    ),
    "rank residual": "Chooses a calibrated rank-two or rank-three activity in supported probability regimes.",
    "online within-case motif": (
        "Updates repeated motif outcomes only after each actual event is observed, allowing case-local adaptation."
    ),
    "hashed rank router": (
        "Uses a regularized hashed feature model to decide whether soft rank one, two, or three should be used."
    ),
    "hierarchical": "Backs off from detailed state/pattern/agreement reliability to broader calibrated contexts.",
    "consensus hierarchy": "Restricts selection to consensus models when agreement is strong, then uses hierarchy.",
    "delayed-feedback": "Updates decayed reliability only after the previous event's actual activity is observed.",
}
DASHBOARD_EXPLANATIONS = {
    "cheating": "Uses a correct constituent top prediction when one exists; otherwise soft voting.",
    "overview": (
        "The cheating baseline selects a correct constituent top prediction when possible. "
        "Second/third-rank and probability-pooling rules can exceed it by predicting activities "
        "that are not any constituent's top choice."
    ),
    "hypotheses": (
        "Hover over bars for rule details. Rank rules may predict the second or third activity "
        "directly; model-selection rules choose a constituent source."
    ),
    "scenarios": (
        "Consensus scenarios are deployable because they use only rule outputs. Oracle scenarios "
        "inspect the label and show whether the correct answer already exists in the rule portfolio."
    ),
    "scenario_gap": (
        "Gap recovered is measured relative to the soft-to-cheating-baseline accuracy gap. Values "
        "over 100% are possible when rank or pooling rules beat that top-1 baseline."
    ),
    "impact": (
        "Recoveries repair a soft error; harms replace a correct soft prediction with a wrong result. "
        "Net impact is recoveries minus harms."
    ),
    "conditions": (
        "These are data-mined feature/value conditions, not decision rules. Each row asks whether one constituent "
        "was more accurate than soft voting in that calibration condition, then reports its held-out impact."
    ),
    "weighted": (
        "Each fulfilled condition votes for its calibrated model. Choose equal weight, calibration "
        "gain, or gain adjusted by support."
    ),
    "integration": (
        "The active integration retains soft voting unless it is uncertain and at least two distinct rule families agree "
        "on an alternative with positive held-out selector evidence."
    ),
    "headline": (
        "This table puts the main baselines, best individual rule, deployable combinations, and diagnostic ceilings "
        "in one place. Oracle rows use the true label only to measure remaining opportunity."
    ),
}


def _condition_feature_description(feature: str) -> str:
    """Explain an activity-invariant condition feature, including per-model variants."""
    if feature in CONDITION_FEATURE_DESCRIPTIONS:
        return CONDITION_FEATURE_DESCRIPTIONS[feature]
    if feature.endswith(" vs soft"):
        return "Whether this constituent's top prediction agrees with soft voting's current top prediction."
    if feature.endswith(" vs consensus"):
        return "Whether this constituent's top prediction agrees with the most common constituent top prediction."
    if feature.endswith(" process state"):
        return "The constituent's learned process state for the current prefix; it is a model state, not an activity label."
    if feature.endswith(" state support"):
        return "Training-visit support band for this constituent's current learned process state."
    if feature.endswith(" support rank"):
        return "This constituent's relative rank by current-state training support within the ensemble."
    if feature.endswith(" confidence rank"):
        return "This constituent's relative rank by its top-prediction confidence within the ensemble."
    if feature.endswith(" entropy rank"):
        return "This constituent's relative rank by normalized distribution entropy within the ensemble."
    if feature.endswith(" soft-prediction rank"):
        return "The rank at which this constituent places the activity selected by soft voting."
    return "Activity-invariant feature extracted from the observable ensemble state at prediction time."


def _rule_description(name: str) -> str:
    for prefix, description in RULE_DESCRIPTIONS.items():
        if name.startswith(prefix):
            return description
    return "Calibration-fitted candidate selector; inspect its family, selected sources, recoveries, and harms."


def _info(text: str) -> html.Details:
    """Return a compact clickable explanation control."""
    return html.Details(
        [html.Summary("i", title="Show explanation"), html.P(text)],
        className="info-button",
    )


def _section_heading(title: str, explanation: str) -> html.Div:
    return html.Div([html.H2(title), _info(explanation)], className="section-heading")


def _hide_archived_results(summary: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> None:
    """Remove archived candidates from current views while preserving saved files unchanged."""
    summary["hypotheses"] = [rule for rule in summary.get("hypotheses", []) if rule["name"] not in ARCHIVED_RULE_NAMES]
    summary["rule_integrations"] = [
        integration
        for integration in summary.get("rule_integrations", [])
        if integration["name"] not in ARCHIVED_INTEGRATION_NAMES or integration["name"] in ACTIVE_INTEGRATION_NAMES
    ]
    soft_analysis = summary.get("soft_failure_analysis")
    if soft_analysis:
        soft_analysis["rule_impacts"] = [
            impact for impact in soft_analysis.get("rule_impacts", []) if impact["name"] not in ARCHIVED_RULE_NAMES
        ]
    for row in rows or []:
        for field in ("rule_predictions", "rule_models", "rule_diagnostics"):
            row[field] = {name: value for name, value in row.get(field, {}).items() if name not in ARCHIVED_RULE_NAMES}
        row["integration_predictions"] = {
            name: value
            for name, value in row.get("integration_predictions", {}).items()
            if name not in ARCHIVED_INTEGRATION_NAMES or name in ACTIVE_INTEGRATION_NAMES
        }
        row["integration_diagnostics"] = {
            name: value
            for name, value in row.get("integration_diagnostics", {}).items()
            if name not in ARCHIVED_INTEGRATION_NAMES or name in ACTIVE_INTEGRATION_NAMES
        }


def _restore_calibration_scores(rebuilt: list[dict[str, Any]], persisted: list[dict[str, Any]]) -> None:
    """Keep calibration scores when test-only event data is rebuilt for the dashboard."""
    stored_by_name = {item["name"]: item for item in persisted}
    fields = ("calibration_accuracy", "calibration_correct", "calibration_total")
    for candidate in rebuilt:
        stored = stored_by_name.get(candidate["name"], {})
        candidate.update({field: stored[field] for field in fields if field in stored})


def load_results(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a saved summary and its event-level diagnostics."""
    summary_path = results_dir / "summary.json"
    events_path = results_dir / "events.jsonl"
    if not summary_path.exists() or not events_path.exists():
        msg = f"Expected summary.json and events.jsonl in {results_dir}"
        raise FileNotFoundError(msg)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    _hide_archived_results(summary, rows)
    for row in rows:
        row.pop("hard_prediction", None)
        row.pop("hard_correct", None)
        row["oracle_prediction"] = row["actual"] if row["correct_models"] else row["soft_prediction"]
        row["oracle_correct"] = row["oracle_prediction"] == row["actual"]
        row["oracle_gap"] = row["oracle_correct"] and not row["soft_correct"]
    # Current result files already persist the calibrated rule and integration
    # predictions.  Preserve them: recomputing an integration from test rows
    # would replace its nested selector policy with a different, uncalibrated
    # policy and make the displayed recovered gap incorrect.  The fallback is
    # only for legacy files that have no persisted hypotheses at all.
    persisted_hypotheses = summary.get("hypotheses", [])
    if not persisted_hypotheses:
        summary["hypotheses"] = evaluate_hypotheses([], rows, rules=default_hypotheses())
    else:
        summary["hypotheses"] = persisted_hypotheses
    integrations = summary.get("rule_integrations", [])
    if not integrations:
        # Legacy result files did not persist an active integration.  Retain
        # the previous compatibility reconstruction only for those files.
        integrations = evaluate_rule_integrations(
            [],
            rows,
            [hypothesis["name"] for hypothesis in summary["hypotheses"]],
        )
        summary["rule_integrations"] = integrations
    mandatory = next((item for item in integrations if item.get("deployment_priority")), None)
    if summary.get("soft_failure_analysis") is not None:
        summary["soft_failure_analysis"]["rule_impacts"] = analyze_rule_impacts(rows)
    soft_accuracy = sum(row["soft_correct"] for row in rows) / len(rows) if rows else 0.0
    oracle_accuracy = sum(row["oracle_correct"] for row in rows) / len(rows) if rows else 0.0
    persisted_strategies = summary.get("strategies", [])
    summary["strategies"] = [
        {"name": "soft voting", "accuracy": soft_accuracy},
        {"name": "cheating voting", "accuracy": oracle_accuracy},
    ]
    _restore_calibration_scores(summary["strategies"], persisted_strategies)
    summary["rule_scenarios"] = evaluate_rule_scenarios(rows, summary.get("hypotheses", []))
    available_hypotheses = [
        hypothesis
        for hypothesis in summary.get("hypotheses", [])
        if hypothesis.get("parameters", {}).get("calibration_available", True)
    ]
    available_names = {hypothesis["name"] for hypothesis in available_hypotheses}
    candidates = [
        *available_hypotheses,
        *(
            scenario
            for scenario in summary["rule_scenarios"]
            if not scenario["oracle"] and set(scenario.get("rule_names", [])) <= available_names
        ),
        *summary.get("rule_integrations", []),
        {"name": "soft voting", "accuracy": soft_accuracy, "deployable": True},
    ]
    best = select_best_deployable(candidates)
    if best:
        for row in rows:
            prediction = (
                row["soft_prediction"] if best["name"] == "soft voting" else _candidate_prediction(row, best["name"])
            )
            if prediction is not None:
                row["best_deployable_method"] = best["name"]
                row["best_deployable_prediction"] = prediction
    model_rows = rows[0].get("models", []) if rows else []
    minimum_complexity = min((int(model["complexity"]) for model in model_rows), default=None)
    minimum_models = [model["name"] for model in model_rows if int(model["complexity"]) == minimum_complexity]
    audit = mandatory.get("parameters", {}).get("enforcement_audit", {}) if mandatory else {}
    boost_contract = {
        "minimum_complexity": minimum_complexity,
        "minimum_complexity_models": minimum_models,
        "unique_minimum_complexity_model": len(minimum_models) == 1,
        "default_generalist_is_bag": minimum_models == ["bag"],
        "independent_stack_present": mandatory is not None,
        "independent_stack_is_best_deployable": bool(mandatory and best and mandatory["name"] == best["name"]),
        **audit,
    }
    summary["independent_boost_contract"] = boost_contract
    summary["adaptive_recovery_contract"] = boost_contract
    gap = oracle_accuracy - soft_accuracy
    recovered = max(0.0, best["accuracy"] - soft_accuracy) if best else 0.0
    summary["best_rule_result"] = best
    summary["recoverable_gap"] = gap
    summary["recovered_gap_with_best_rule"] = recovered
    summary["recovered_gap_fraction"] = recovered / gap if gap else 0.0
    return summary, rows


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _accuracy_figure(summary: dict[str, Any]) -> go.Figure:
    strategies = [*summary["strategies"], *summary["hypotheses"][:8]]
    frame = pd.DataFrame(strategies)
    figure = px.bar(
        frame,
        x="accuracy",
        y="name",
        orientation="h",
        color="family" if "family" in frame else None,
        text=frame["accuracy"].map(_percent),
        labels={"accuracy": "Accuracy", "name": "Strategy", "family": "Rule family"},
    )
    figure.update_layout(
        height=460,
        autosize=True,
        yaxis={"categoryorder": "total ascending"},
        showlegend=True,
        margin={"l": 170, "r": 20, "t": 20, "b": 40},
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1])
    return figure


def _calibration_generalization_figure(summary: dict[str, Any]) -> go.Figure:
    """Compare in-sample calibration with held-out test accuracy."""
    candidates = [*summary.get("strategies", []), *summary.get("per_model", []), *summary.get("hypotheses", [])]
    rows = [candidate for candidate in candidates if candidate.get("calibration_accuracy") is not None]
    if not rows:
        figure = go.Figure()
        figure.add_annotation(
            text="This saved run has no calibration scores. Rerun the investigation to compare calibration and test.",
            showarrow=False,
        )
        figure.update_layout(height=260)
        return figure
    frame = pd.DataFrame(rows)
    frame["generalization gap"] = frame["calibration_accuracy"] - frame["accuracy"]
    frame["kind"] = frame.apply(
        lambda candidate: "rule" if "family" in candidate and pd.notna(candidate.get("family")) else "baseline/model",
        axis=1,
    )
    figure = go.Figure()
    for _, candidate in frame.sort_values("generalization gap", ascending=False).iterrows():
        figure.add_trace(
            go.Scatter(
                x=[candidate["accuracy"], candidate["calibration_accuracy"]],
                y=[candidate["name"], candidate["name"]],
                mode="lines+markers",
                line={"color": "#7f8c8d"},
                marker={"size": 9},
                showlegend=False,
                hovertemplate=(
                    "%{y}<br>test: %{x:.1%}<extra></extra>"
                    if candidate["accuracy"] == candidate["calibration_accuracy"]
                    else "%{y}<br>test / calibration: %{x:.1%}<extra></extra>"
                ),
            )
        )
    figure.add_trace(
        go.Scatter(
            x=frame["accuracy"],
            y=frame["name"],
            mode="markers",
            name="Held-out test",
            marker={"symbol": "circle", "size": 10},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["calibration_accuracy"],
            y=frame["name"],
            mode="markers",
            name="Calibration (fit data)",
            marker={"symbol": "diamond", "size": 10},
        )
    )
    figure.update_layout(
        height=max(360, 28 * len(frame) + 120),
        margin={"l": 240, "r": 25, "t": 20, "b": 45},
        yaxis={"categoryorder": "array", "categoryarray": frame["name"].tolist()},
        legend={"orientation": "h", "y": 1.05},
    )
    figure.update_xaxes(title="Accuracy", tickformat=".0%", range=[0, 1])
    return figure


def _agreement_figure(rows: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby("agreement_count", as_index=False)
        .agg(events=("oracle_gap", "size"), oracle_gap_rate=("oracle_gap", "mean"))
        .sort_values("agreement_count")
    )
    figure = px.bar(
        grouped,
        x="agreement_count",
        y="oracle_gap_rate",
        text=grouped["oracle_gap_rate"].map(_percent),
        hover_data={"events": True},
        labels={"agreement_count": "Models agreeing", "oracle_gap_rate": "Recoverable soft-vote failure rate"},
    )
    figure.update_yaxes(tickformat=".0%", range=[0, max(0.05, grouped["oracle_gap_rate"].max() * 1.15)])
    figure.update_layout(height=360, autosize=True, margin={"l": 60, "r": 15, "t": 20, "b": 50})
    return figure


def _position_figure(rows: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(rows)
    frame["position_bucket"] = (frame["relative_position"] * 10).clip(upper=9.999).astype(int) * 10
    grouped = frame.groupby("position_bucket", as_index=False).agg(
        events=("oracle_gap", "size"),
        oracle_gap_rate=("oracle_gap", "mean"),
    )
    grouped["position_label"] = grouped["position_bucket"].map(lambda value: f"{value}-{value + 10}%")
    figure = px.line(
        grouped,
        x="position_label",
        y="oracle_gap_rate",
        markers=True,
        hover_data={"events": True},
        labels={"position_label": "Relative sequence position", "oracle_gap_rate": "Soft-to-oracle gap rate"},
    )
    figure.update_yaxes(tickformat=".0%", range=[0, max(0.05, grouped["oracle_gap_rate"].max() * 1.15)])
    figure.update_layout(height=360, autosize=True, margin={"l": 60, "r": 15, "t": 20, "b": 50})
    return figure


def _rule_impact_figure(impacts: list[dict[str, Any]]) -> go.Figure:
    """Show recovered soft errors against newly introduced errors."""
    selected = impacts[:15]
    figure = go.Figure()
    figure.add_bar(
        x=[item["recoveries"] for item in selected],
        y=[item["name"] for item in selected],
        orientation="h",
        name="Soft errors recovered",
    )
    figure.add_bar(
        x=[-item["harms"] for item in selected],
        y=[item["name"] for item in selected],
        orientation="h",
        name="Soft-correct events harmed",
    )
    figure.update_layout(
        barmode="relative",
        height=max(430, 34 * len(selected)),
        yaxis={"categoryorder": "array", "categoryarray": [item["name"] for item in reversed(selected)]},
        xaxis_title="Events gained (+) or lost (-) versus soft voting",
        margin={"l": 230, "r": 20, "t": 20, "b": 50},
    )
    return figure


def _scenario_figure(summary: dict[str, Any]) -> go.Figure:
    """Compare deployable rule-set scenarios and diagnostic ceilings."""
    scenarios = summary.get("rule_scenarios", [])
    if not scenarios:
        figure = go.Figure()
        figure.update_layout(height=430, xaxis_title="Accuracy", yaxis_title="Rule-set scenario")
        return figure
    frame = pd.DataFrame(scenarios)
    frame["description"] = frame["oracle"].map(
        {
            False: "Deployable consensus override; falls back to soft voting without sufficient unique agreement.",
            True: "Diagnostic ceiling that checks whether any rule in the set was correct.",
        }
    )
    figure = px.bar(
        frame,
        x="accuracy",
        y="name",
        orientation="h",
        color="family" if not frame.empty else None,
        text=frame["accuracy"].map(_percent) if not frame.empty else None,
        hover_data={"description": True, "net_accuracy_delta": ":+.2%", "gap_recovered_fraction": ":.1%"},
        labels={"accuracy": "Accuracy", "name": "Rule-set scenario", "family": "Scenario type"},
    )
    figure.update_layout(
        height=max(430, 58 * len(scenarios)),
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 230, "r": 20, "t": 20, "b": 50},
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1])
    return figure


def _headline_results(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the compact benchmark comparison shown prominently in Overview."""
    soft = next(item for item in summary["strategies"] if item["name"] == "soft voting")
    cheating = next(item for item in summary["strategies"] if item["name"] == "cheating voting")
    gap = cheating["accuracy"] - soft["accuracy"]
    hypotheses = summary.get("hypotheses", [])
    integrations = summary.get("rule_integrations", [])
    scenarios = summary.get("rule_scenarios", [])
    selected: list[tuple[str, str, dict[str, Any]]] = [("Soft voting", "baseline", soft)]
    if hypotheses:
        selected.append(("Best individual rule", "deployable", max(hypotheses, key=lambda item: item["accuracy"])))
    if integrations:
        selected.append(("Best smart integration", "deployable", max(integrations, key=lambda item: item["accuracy"])))
    for label, prefix in (
        ("Core consensus", "core rules: consensus"),
        ("Core rule-set oracle", "core rule-set oracle"),
        ("All rule-set oracle", "all rule-set oracle"),
    ):
        matches = [scenario for scenario in scenarios if scenario["name"].startswith(prefix)]
        if matches:
            selected.append((label, "diagnostic" if matches[0]["oracle"] else "deployable", matches[0]))
    selected.append(("Cheating voting", "baseline", cheating))
    rows = []
    for label, kind, result in selected:
        gain = result["accuracy"] - soft["accuracy"]
        rows.append(
            {
                "scenario": label,
                "selected method": result["name"],
                "type": kind,
                "accuracy": result["accuracy"],
                "accuracy display": _percent(result["accuracy"]),
                "gain vs soft": gain,
                "gap recovered": gain / gap if gap else 0.0,
            }
        )
    return rows


def _headline_figure(rows: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(rows)
    figure = px.bar(
        frame,
        x="accuracy",
        y="scenario",
        orientation="h",
        color="type",
        text="accuracy display",
        hover_data={
            "selected method": True,
            "gain vs soft": ":+.2%",
            "gap recovered": ":.1%",
            "accuracy display": False,
        },
        labels={"accuracy": "Accuracy", "scenario": "Result", "type": "Result type"},
    )
    figure.update_layout(
        height=max(450, 54 * len(rows)),
        yaxis={"categoryorder": "array", "categoryarray": [row["scenario"] for row in reversed(rows)]},
        margin={"l": 170, "r": 20, "t": 20, "b": 50},
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1])
    return figure


def _integration_figure(
    integrations: list[dict[str, Any]],
    *,
    soft_accuracy: float,
    cheating_accuracy: float,
) -> go.Figure:
    if not integrations:
        return go.Figure().update_layout(height=430)
    frame = pd.DataFrame(integrations)
    frame["accuracy display"] = frame["accuracy"].map(_percent)
    figure = px.bar(
        frame,
        x="accuracy",
        y="name",
        orientation="h",
        text="accuracy display",
        color="net_correct",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        hover_data={
            "description": True,
            "recoveries": True,
            "harms": True,
            "net_correct": True,
            "gap_recovered_fraction": ":.1%",
            "accuracy display": False,
        },
        labels={"accuracy": "Accuracy", "name": "Integration method", "net_correct": "Net events"},
    )
    figure.update_layout(
        height=max(460, 58 * len(integrations)),
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 245, "r": 20, "t": 20, "b": 50},
    )
    figure.add_vline(
        x=soft_accuracy,
        line_dash="dash",
        line_color="#607080",
        annotation_text="Soft voting",
        annotation_position="bottom right",
    )
    figure.add_vline(
        x=cheating_accuracy,
        line_dash="dot",
        line_color="#145c73",
        annotation_text="Cheating baseline",
        annotation_position="top right",
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1])
    return figure


def _condition_table_rows(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "family": condition["family"],
            "when feature": condition["feature"],
            "has value": condition["value"],
            "learned choice": condition["recommended_model"],
            "calibration events": condition["calibration_support"],
            "calibration advantage": _percent(condition["calibration_gain"]),
            "test events": condition["test_support"],
            "recoveries": condition["recoveries"],
            "harms": condition["harms"],
            "net improvement": condition["net_correct"],
            "helpful when decisive": _percent(condition["decisive_precision"]),
        }
        for condition in conditions
    ]


def _condition_tooltips(conditions: list[dict[str, Any]]) -> list[dict[str, dict[str, str]]]:
    """Provide hover help for each data-mined condition row."""
    return [
        {
            "when feature": {"value": _condition_feature_description(condition["feature"]), "type": "markdown"},
            "has value": {
                "value": "Observed category or bin for this feature. Click the row to keep its explanation visible below.",
                "type": "markdown",
            },
        }
        for condition in conditions
    ]


def _condition_label(condition: dict[str, Any]) -> str:
    """Return a concise human-readable condition without repeating its learned policy."""
    return f"{condition['feature']} = {condition['value']} → {condition['recommended_model']}"


def _soft_failure_rows(
    rows: list[dict[str, Any]],
    conditions_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    def ranked_activity(row: dict[str, Any], rank: int) -> str:
        ranked = row.get("soft_ranked_predictions", [])
        return ranked[rank - 1].get("activity", "") if len(ranked) >= rank else ""

    table_rows = []
    for row in rows:
        if row["soft_correct"]:
            continue
        active_rules = [
            name for name, prediction in row.get("rule_predictions", {}).items() if prediction != row["soft_prediction"]
        ]
        recovering_rules = [
            name for name, prediction in row.get("rule_predictions", {}).items() if prediction == row["actual"]
        ]
        matched_conditions = [
            _condition_label(conditions_by_id[condition_id])
            for condition_id in row.get("condition_matches", [])
            if condition_id in conditions_by_id
        ]
        table_rows.append(
            {
                "sequence": row["sequence_id"],
                "position": row["position_1based"],
                "actual": row["actual"],
                "soft prediction": row["soft_prediction"],
                "soft rank 2": ranked_activity(row, 2),
                "soft rank 3": ranked_activity(row, 3),
                "soft margin": round(float(row.get("soft_margin", 0.0)), 4),
                "soft entropy": round(float(row.get("soft_normalized_entropy", 0.0)), 4),
                "previous soft correct": row.get("previous_soft_correct"),
                "previous actual": row.get("previous_actual", ""),
                "empty models": row.get("empty_prediction_count", 0),
                "correct models": ", ".join(row["correct_models"]),
                "agreement": row["agreement_count"],
                "last activities": row["suffix_3"],
                "rules that override soft": "; ".join(active_rules),
                "rules that recover error": "; ".join(recovering_rules),
                "favorable conditions fulfilled": "; ".join(matched_conditions),
            }
        )
    return table_rows


def _candidate_prediction(row: dict[str, Any], candidate_name: str) -> str | None:
    """Return a persisted event prediction for any deployable candidate type."""
    if candidate_name == "soft voting":
        return row["soft_prediction"]
    for field in ("rule_predictions", "integration_predictions", "scenario_predictions"):
        if candidate_name in row.get(field, {}):
            return row[field][candidate_name]
    if row.get("best_deployable_method") == candidate_name:
        return row.get("best_deployable_prediction")
    return None


def _best_available_sequence_candidate(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the strongest deployable result whose event predictions are available."""
    available_hypotheses = [
        hypothesis
        for hypothesis in summary.get("hypotheses", [])
        if hypothesis.get("parameters", {}).get("calibration_available", True)
    ]
    available_rule_names = {hypothesis["name"] for hypothesis in available_hypotheses}
    candidates = [
        *available_hypotheses,
        *summary.get("rule_integrations", []),
        *(
            scenario
            for scenario in summary.get("rule_scenarios", [])
            if not scenario.get("oracle", False) and set(scenario.get("rule_names", [])) <= available_rule_names
        ),
    ]
    if summary.get("best_rule_result", {}).get("name") == "soft voting":
        candidates.append(summary["best_rule_result"])
    if not rows:
        return None
    available = [candidate for candidate in candidates if _candidate_prediction(rows[0], candidate["name"]) is not None]
    return select_best_deployable(available)


def _sequence_mismatch_count(sequence_rows: list[dict[str, Any]], best_deployable: dict[str, Any] | None) -> int:
    """Count positions where the selected deployable result differs from cheating voting."""
    if best_deployable is None:
        return 0
    return sum(
        (_candidate_prediction(row, best_deployable["name"]) or "") != row["oracle_prediction"] for row in sequence_rows
    )


def _sequence_options(rows: list[dict[str, Any]], best_deployable: dict[str, Any] | None) -> list[dict[str, str]]:
    """Label each sequence with its deployable-versus-cheating mismatch count."""
    sequence_ids = list(dict.fromkeys(row["sequence_id"] for row in rows))
    return [
        {
            "label": f"{sequence_id} · {_sequence_mismatch_count([row for row in rows if row['sequence_id'] == sequence_id], best_deployable)} mismatches",
            "value": sequence_id,
        }
        for sequence_id in sequence_ids
    ]


def _top_distribution_text(distribution: dict[str, float]) -> str:
    """Format the top three activities in a probability distribution for a hover panel."""
    return ", ".join(
        f"{activity}={probability:.2f}"
        for activity, probability in sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[:3]
    )


def _best_deployable_distribution(row: dict[str, Any], best_deployable: dict[str, Any] | None) -> dict[str, float]:
    """Reconstruct an integration's final normalized distribution from its stored model multipliers."""
    if best_deployable is None:
        return dict(row.get("soft_distribution", {}))
    multipliers = (
        row.get("integration_diagnostics", {})
        .get(best_deployable["name"], {})
        .get("model_multipliers", {})
    )
    if not multipliers:
        return dict(row.get("soft_distribution", {}))
    scores: dict[str, float] = {}
    for model in row["models"]:
        multiplier = float(multipliers.get(model["name"], 1.0))
        for activity, probability in model["distribution"].items():
            scores[activity] = scores.get(activity, 0.0) + multiplier * float(probability)
    total = sum(scores.values())
    return {activity: probability / total for activity, probability in scores.items()} if total else {}


def _rule_driven_change_text(row: dict[str, Any], best_deployable: dict[str, Any] | None) -> str:
    """Attribute a Best-deployable deviation from soft voting to its active rule modifiers."""
    if best_deployable is None:
        return ""
    prediction = _candidate_prediction(row, best_deployable["name"])
    if prediction is None or prediction == row["soft_prediction"]:
        return ""
    integration = row.get("integration_diagnostics", {}).get(best_deployable["name"], {})
    rule_names = [name for name, targets in integration.get("rule_targets", {}).items() if targets]
    if not rule_names and best_deployable["name"] in row.get("rule_predictions", {}):
        rule_names = [best_deployable["name"]]
    return ", ".join(rule_names) if rule_names else "deployable integration"


def _sequence_figure(
    sequence_rows: list[dict[str, Any]],
    model_names: list[str],
    best_deployable: dict[str, Any] | None = None,
    tracked_rules: list[str] | None = None,
) -> go.Figure:
    best_lane = ["best_deployable"] if best_deployable else []
    lanes = ["actual", "soft_prediction", "adaptive_prediction", *best_lane, "oracle_prediction", *model_names]
    labels = {
        "actual": "Actual",
        "soft_prediction": "Soft voting",
        "adaptive_prediction": "Adaptive voting",
        "best_deployable": "Best deployable",
        "oracle_prediction": "Cheating voting",
        **{name: name for name in model_names},
    }
    figure = go.Figure()
    for lane_index, lane in enumerate(lanes):
        predictions: list[str] = []
        correct: list[bool] = []
        for row in sequence_rows:
            if lane == "best_deployable":
                prediction = _candidate_prediction(row, best_deployable["name"]) or ""  # type: ignore[index]
            elif lane in {"actual", "soft_prediction", "adaptive_prediction", "oracle_prediction"}:
                prediction = row.get(lane, "")
            else:
                prediction = next(model["prediction"] for model in row["models"] if model["name"] == lane)
            predictions.append(prediction)
            correct.append(lane == "actual" or prediction == row["actual"])
        rule_driven_changes = [_rule_driven_change_text(row, best_deployable) for row in sequence_rows]
        model_lane = lane not in {"actual", "soft_prediction", "adaptive_prediction", "best_deployable", "oracle_prediction"}
        line_style: dict[str, Any] = {"width": 1.25}
        marker_size = 11
        if lane == "best_deployable":
            line_style = {"width": 3}
            marker_size = 13
        elif lane == "oracle_prediction":
            line_style = {"width": 2.25, "dash": "dot"}
            marker_size = 12
        figure.add_trace(
            go.Scatter(
                x=[row["position_1based"] for row in sequence_rows],
                y=[lane_index] * len(sequence_rows),
                mode="lines+markers+text",
                name=labels[lane],
                text=predictions,
                textposition="top center",
                marker={
                    "size": marker_size,
                    "color": ["#16865b" if is_correct else "#d1495b" for is_correct in correct],
                    "symbol": ["circle" if is_correct else "x" for is_correct in correct],
                    "line": (
                        {
                            "color": ["#c28a00" if change else "rgba(0,0,0,0)" for change in rule_driven_changes],
                            "width": [3 if change else 0 for change in rule_driven_changes],
                        }
                        if lane == "best_deployable"
                        else {"width": 0}
                    ),
                },
                customdata=[
                    [
                        row["actual"],
                        row["agreement_count"],
                        row["prefix_text"],
                        next(
                            (
                                diagnostic.get("target_model") or ""
                                for diagnostic in row.get("rule_diagnostics", {}).values()
                                if diagnostic.get("active")
                            ),
                            "",
                        ),
                        next(
                            (
                                diagnostic.get("recovery_age")
                                for diagnostic in row.get("rule_diagnostics", {}).values()
                                if diagnostic.get("active")
                            ),
                            "",
                        ),
                        (
                            ", ".join(
                                f"{name}={multiplier:.2f}"
                                for name, multiplier in row.get("integration_diagnostics", {})
                                .get(best_deployable["name"], {})  # type: ignore[index]
                                .get("model_multipliers", {})
                                .items()
                                if multiplier != 1.0
                            )
                            if best_deployable
                            else ""
                        ),
                        next(
                            (
                                ", ".join(
                                    f"{name}={multiplier:.2f}"
                                    for name, multiplier in diagnostic.get("model_multipliers", {}).items()
                                    if multiplier != 1.0
                                )
                                for name, diagnostic in row.get("rule_diagnostics", {}).items()
                                if name == "ngram correctness-streak multiplier"
                            ),
                            "",
                        ),
                        (
                            next(
                                model.get("state_access_string", model["state"])
                                for model in row["models"]
                                if model["name"] == lane
                            )
                            if model_lane
                            else ""
                        ),
                        (
                            next(model["state_visits"] for model in row["models"] if model["name"] == lane)
                            if model_lane
                            else ""
                        ),
                        next(
                            (
                                (
                                    f"{float(model['state_accuracy']):.1%} "
                                    f"({model.get('state_correct_predictions', 0)}/{model.get('state_total_predictions', 0)})"
                                )
                                if model.get("state_accuracy") is not None
                                else "not yet measured"
                            )
                            for model in row["models"]
                            if model["name"] == lane
                        )
                        if model_lane
                        else "",
                        (
                            ", ".join(
                                f"{item['activity']}={item['probability']:.2f}"
                                for item in next(
                                    model["ranked_predictions"]
                                    for model in row["models"]
                                    if model["name"] == lane
                                )[:3]
                            )
                            if model_lane
                            else ""
                        ),
                        _top_distribution_text(row.get("soft_distribution", {})),
                        _top_distribution_text(_best_deployable_distribution(row, best_deployable)),
                        rule_driven_changes[row_index],
                    ]
                    for row_index, row in enumerate(sequence_rows)
                ],
                hovertemplate=(
                    f"{labels[lane]}: %{{text}}<br>Actual: %{{customdata[0]}}<br>"
                    "Agreement: %{customdata[1]}<br>"
                    + (
                        "State access string: %{customdata[7]}<br>State frequency: %{customdata[8]}<br>"
                        "State prediction accuracy: %{customdata[9]}<br>"
                        "Next activities (top 3): %{customdata[10]}<extra></extra>"
                        if model_lane
                        else (
                            "Final soft distribution (top 3): %{customdata[11]}<extra></extra>"
                            if lane == "soft_prediction"
                            else "<extra></extra>"
                        )
                    )
                ),
                line=line_style,
            )
        )
    if best_deployable:
        best_lane_index = lanes.index("best_deployable")
        cheating_lane_index = lanes.index("oracle_prediction")
        for row in sequence_rows:
            best_prediction = _candidate_prediction(row, best_deployable["name"])
            if best_prediction is None or best_prediction == row["oracle_prediction"]:
                continue
            figure.add_shape(
                type="rect",
                x0=row["position_1based"] - 0.38,
                x1=row["position_1based"] + 0.38,
                y0=min(best_lane_index, cheating_lane_index) - 0.42,
                y1=max(best_lane_index, cheating_lane_index) + 0.42,
                fillcolor="rgba(255, 215, 0, 0.22)",
                line={"color": "rgba(218, 165, 32, 0.9)", "width": 1},
                layer="below",
            )
    for rule_name in tracked_rules or []:
        active_rows = [row for row in sequence_rows if row.get("rule_diagnostics", {}).get(rule_name, {}).get("active")]
        if active_rows:
            figure.add_trace(
                go.Scatter(
                    x=[row["position_1based"] for row in active_rows],
                    y=[-0.35] * len(active_rows),
                    mode="markers",
                    name=f"Rule active: {rule_name}",
                    marker={"size": 11, "symbol": "diamond"},
                    customdata=[[row["position_1based"], rule_name] for row in active_rows],
                    hovertemplate="Rule active: %{customdata[1]}<br>Position: %{customdata[0]}<extra></extra>",
                )
            )
    figure.update_layout(
        height=max(430, 70 * len(lanes)),
        margin={"l": 120, "r": 30, "t": 20, "b": 50},
        showlegend=False,
        yaxis={"tickmode": "array", "tickvals": list(range(len(lanes))), "ticktext": [labels[lane] for lane in lanes]},
        xaxis_title="Position in sequence",
    )
    if best_deployable:
        figure.update_traces(
            selector={"name": "Best deployable"},
            hovertemplate=(
                f"Best deployable: {best_deployable['name']}<br>Prediction: %{{text}}<br>"
                "Actual: %{customdata[0]}<br>Agreement: %{customdata[1]}<br>"
                "Recovery target: %{customdata[3]}<br>"
                "Recovery age: %{customdata[4]}<br>N-gram streak multipliers: %{customdata[6]}<br>"
                "Combined multipliers: %{customdata[5]}<br>"
                "Baseline soft distribution (top 3): %{customdata[11]}<br>"
                "Final modifier-weighted distribution (top 3): %{customdata[12]}<br>"
                "<b>Rule-driven change: %{customdata[13]}</b><extra></extra>"
            ),
        )
    return figure


def discover_result_sets(results_root: Path) -> dict[str, Path]:
    """Discover standard DATASET/summary.json result directories."""
    discovered: dict[str, Path] = {}
    candidates = [results_root / "summary.json", *(results_root.glob("*/summary.json"))]
    for summary_path in candidates:
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dataset = str(summary.get("dataset") or summary_path.parent.name)
        current = discovered.get(dataset)
        if current is None or summary_path.stat().st_mtime > (current / "summary.json").stat().st_mtime:
            discovered[dataset] = summary_path.parent
    return dict(sorted(discovered.items()))


def _load_comparison_summary(results_dir: Path) -> dict[str, Any]:
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    summary.setdefault("hypotheses", [])
    summary.setdefault("rule_scenarios", [])
    summary.setdefault("rule_integrations", [])
    summary.setdefault("run_config", {})
    _hide_archived_results(summary)
    return summary


def _global_comparison_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, summary in summaries.items():
        strategies = {item["name"]: item["accuracy"] for item in summary.get("strategies", [])}
        soft = strategies.get("soft voting", 0.0)
        cheating = strategies.get("cheating voting", soft)
        individual = max(summary["hypotheses"], key=lambda item: item["accuracy"], default=None)
        integration = max(summary["rule_integrations"], key=lambda item: item["accuracy"], default=None)
        deployable_candidates = [
            *summary["hypotheses"],
            *summary["rule_integrations"],
            *(scenario for scenario in summary["rule_scenarios"] if not scenario.get("oracle", False)),
        ]
        best = select_best_deployable(deployable_candidates)
        best_accuracy = best["accuracy"] if best else soft
        gap = cheating - soft
        rows.append(
            {
                "dataset": dataset,
                "test events": summary.get("test_events", 0),
                "soft voting": soft,
                "best individual": individual["accuracy"] if individual else soft,
                "best integration": integration["accuracy"] if integration else soft,
                "best deployable": best_accuracy,
                "best method": best["name"] if best else "soft voting",
                "cheating baseline": cheating,
                "accuracy gain": best_accuracy - soft,
                "gap recovered": (best_accuracy - soft) / gap if gap else 0.0,
            }
        )
    return rows


def _global_accuracy_figure(rows: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(rows).melt(
        id_vars=["dataset"],
        value_vars=["soft voting", "best deployable", "cheating baseline"],
        var_name="strategy",
        value_name="accuracy",
    )
    figure = px.bar(
        frame,
        x="dataset",
        y="accuracy",
        color="strategy",
        barmode="group",
        text=frame["accuracy"].map(_percent),
        labels={"dataset": "Dataset", "accuracy": "Accuracy", "strategy": "Strategy"},
    )
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    figure.update_layout(height=470, margin={"l": 55, "r": 20, "t": 20, "b": 100})
    return figure


def _global_gap_figure(rows: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(rows)
    figure = px.bar(
        frame,
        x="dataset",
        y="gap recovered",
        text=frame["gap recovered"].map(_percent),
        hover_data={"best method": True, "accuracy gain": ":+.2%"},
        labels={"dataset": "Dataset", "gap recovered": "Soft-to-cheating gap recovered"},
    )
    figure.update_yaxes(tickformat=".0%")
    figure.update_layout(height=400, margin={"l": 65, "r": 20, "t": 20, "b": 100})
    return figure


def _global_rule_impact_figure(summaries: dict[str, dict[str, Any]]) -> go.Figure:
    """Compare each deployed rule's held-out net-event impact across datasets."""
    rows = [
        {
            "dataset": dataset,
            "rule": impact["name"],
            "net events": impact.get("net_correct", 0),
            "recoveries": impact.get("recoveries", 0),
            "harms": impact.get("harms", 0),
        }
        for dataset, summary in summaries.items()
        for impact in (summary.get("soft_failure_analysis") or {}).get("rule_impacts", [])
    ]
    if not rows:
        return go.Figure().update_layout(title="No saved rule-impact data")
    frame = pd.DataFrame(rows)
    figure = px.bar(
        frame,
        x="rule",
        y="net events",
        color="dataset",
        barmode="group",
        hover_data={"recoveries": True, "harms": True},
        labels={"rule": "Rule", "net events": "Net events versus soft voting", "dataset": "Dataset"},
    )
    figure.update_layout(height=430, margin={"l": 65, "r": 20, "t": 20, "b": 150})
    return figure


def _comparison_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row["dataset"],
            "test events": f"{row['test events']:,}",
            "soft voting": _percent(row["soft voting"]),
            "best individual": _percent(row["best individual"]),
            "best integration": _percent(row["best integration"]),
            "best deployable": _percent(row["best deployable"]),
            "best method": row["best method"],
            "cheating baseline": _percent(row["cheating baseline"]),
            "accuracy gain": f"{row['accuracy gain']:+.2%}",
            "gap recovered": _percent(row["gap recovered"]),
        }
        for row in rows
    ]


def create_comparison_dashboard(results_root: Path, result_sets: dict[str, Path]) -> Dash:
    """Create the root dashboard for selecting and comparing datasets."""
    summaries = {dataset: _load_comparison_summary(path) for dataset, path in result_sets.items()}
    comparison = _global_comparison_rows(summaries)
    table_rows = _comparison_table_rows(comparison)
    datasets = list(summaries)
    default_dataset = next((dataset for dataset in datasets if dataset.casefold() == "sepsis_cases"), datasets[0])
    # Event diagnostics can be very large for the BPI datasets. Keep only the
    # currently inspected dataset in memory, and load it when its detail tab is
    # needed instead of making the global comparison dashboard expensive.
    detailed_cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}

    def detailed_results(dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if dataset not in detailed_cache:
            detailed_cache.clear()
            detailed_cache[dataset] = load_results(result_sets[dataset])
        return detailed_cache[dataset]

    app = Dash(
        __name__,
        title="Cross-dataset voting investigation",
        assets_folder=str(Path(__file__).with_name("assets")),
    )
    app.layout = html.Main(
        className="investigation-shell",
        children=[
            html.Header(
                [
                    html.Div(
                        [
                            html.H1("Cross-dataset voting investigation"),
                            html.P(f"{len(datasets)} datasets · shared rule-selection benchmark"),
                        ]
                    ),
                    html.Code(str(results_root)),
                ],
                className="investigation-header",
            ),
            html.Label(
                [
                    "Dataset",
                    dcc.Dropdown(
                        [{"label": dataset, "value": dataset} for dataset in datasets],
                        default_dataset,
                        clearable=False,
                        id="global-dataset-selector",
                    ),
                ],
                className="dataset-selector",
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Global comparison",
                        children=html.Div(
                            [
                                html.P(
                                    "All values come from each dataset's held-out test split. Best deployable "
                                    "uses the independent positive-multiplier stack and excludes diagnostic oracle "
                                    "scenarios.",
                                    className="analysis-notice",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Accuracy across datasets",
                                            (
                                                "Compare soft voting, the strongest deployable selector, and "
                                                "cheating voting."
                                            ),
                                        ),
                                        dcc.Graph(
                                            figure=_global_accuracy_figure(comparison),
                                            config=OVERVIEW_GRAPH_CONFIG,
                                            className="global-comparison-chart",
                                        ),
                                    ],
                                    className="global-comparison-chart-panel",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Recovered oracle gap",
                                            "Accuracy gained over soft voting divided by the soft-to-cheating gap.",
                                        ),
                                        dcc.Graph(
                                            figure=_global_gap_figure(comparison),
                                            config=OVERVIEW_GRAPH_CONFIG,
                                            className="global-comparison-chart",
                                        ),
                                    ],
                                    className="global-comparison-chart-panel",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Per-rule impact across datasets",
                                            "Net held-out events gained or lost versus soft voting; each color is a dataset.",
                                        ),
                                        dcc.Graph(
                                            figure=_global_rule_impact_figure(summaries),
                                            config=OVERVIEW_GRAPH_CONFIG,
                                            className="global-comparison-chart",
                                        ),
                                    ],
                                    className="global-comparison-chart-panel",
                                ),
                                html.Section(
                                    [
                                        html.H2("Exact cross-dataset results"),
                                        dash_table.DataTable(
                                            data=table_rows,
                                            columns=[{"name": column, "id": column} for column in table_rows[0]],
                                            sort_action="native",
                                            filter_action="native",
                                            style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ]
                                ),
                            ],
                            className="tab-content",
                        ),
                    ),
                    dcc.Tab(
                        label="Selected dataset",
                        children=html.Div(
                            [
                                html.P(id="selected-dataset-config", className="analysis-notice"),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Data split and generalization check",
                                            "Training events fit the constituent models. Calibration events fit and "
                                            "select candidate rules. Held-out test events provide the final accuracy. "
                                            "A large calibration-to-test drop flags possible overfitting.",
                                        ),
                                        html.P(id="selected-split-summary", className="selection-summary"),
                                        dcc.Graph(
                                            id="selected-calibration-chart",
                                            config=OVERVIEW_GRAPH_CONFIG,
                                        ),
                                    ]
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Headline benchmark results",
                                            DASHBOARD_EXPLANATIONS["headline"],
                                        ),
                                        dcc.Graph(id="selected-headline-chart", config=OVERVIEW_GRAPH_CONFIG),
                                        dash_table.DataTable(
                                            id="selected-headline-table",
                                            style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ]
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Smart integration methods",
                                            DASHBOARD_EXPLANATIONS["integration"],
                                        ),
                                        dcc.Graph(id="selected-integration-chart", config=OVERVIEW_GRAPH_CONFIG),
                                        dash_table.DataTable(
                                            id="selected-integration-table",
                                            page_size=15,
                                            sort_action="native",
                                            filter_action="native",
                                            style_cell={
                                                "whiteSpace": "normal",
                                                "height": "auto",
                                                "textAlign": "left",
                                                "minWidth": "110px",
                                                "maxWidth": "420px",
                                            },
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ]
                                ),
                            ],
                            className="tab-content selected-dataset-content",
                        ),
                    ),
                    dcc.Tab(
                        label="Detailed investigation",
                        children=html.Div(
                            [
                                html.P(
                                    id="detail-dataset-notice",
                                    className="analysis-notice",
                                ),
                                html.Section(
                                    [
                                        html.Article(
                                            [html.Span("Soft voting"), html.Strong(id="detail-soft-accuracy")]
                                        ),
                                        html.Article(
                                            [html.Span("Cheating baseline"), html.Strong(id="detail-cheating-accuracy")]
                                        ),
                                        html.Article([html.Span("Oracle gap"), html.Strong(id="detail-oracle-gap")]),
                                        html.Article(
                                            [html.Span("Best deployable"), html.Strong(id="detail-best-deployable")]
                                        ),
                                        html.Article(
                                            [html.Span("Oracle gap closed"), html.Strong(id="detail-gap-closed")]
                                        ),
                                    ],
                                    className="metric-strip detail-metrics",
                                ),
                                dcc.Tabs(
                                    [
                                        dcc.Tab(
                                            label="Sequence explorer",
                                            children=html.Div(
                                                [
                                                    html.Label(
                                                        [
                                                            "Sequence",
                                                            dcc.Dropdown(id="detail-sequence-id", clearable=False),
                                                        ],
                                                        className="sequence-control",
                                                    ),
                                                    html.Label(
                                                        [
                                                            "Track active rules",
                                                            dcc.Checklist(id="detail-sequence-rule-tracking", inline=True),
                                                        ],
                                                        className="sequence-control",
                                                    ),
                                                    dcc.Graph(
                                                        id="detail-sequence-timeline", config=OVERVIEW_GRAPH_CONFIG
                                                    ),
                                                    html.P(id="detail-sequence-summary", className="selection-summary"),
                                                ],
                                                className="tab-content",
                                            ),
                                        ),
                                        dcc.Tab(
                                            label="Rule impact",
                                            children=html.Div(
                                                [
                                                    html.Section(
                                                        [
                                                            _section_heading(
                                                                "Rule impact relative to soft voting",
                                                                DASHBOARD_EXPLANATIONS["impact"],
                                                            ),
                                                            dcc.Graph(
                                                                id="detail-rule-impact-chart",
                                                                config=OVERVIEW_GRAPH_CONFIG,
                                                            ),
                                                            dash_table.DataTable(
                                                                id="detail-rule-impact-table",
                                                                page_size=15,
                                                                sort_action="native",
                                                                style_cell={
                                                                    "whiteSpace": "normal",
                                                                    "height": "auto",
                                                                    "textAlign": "left",
                                                                },
                                                                style_table={"overflowX": "auto"},
                                                            ),
                                                        ]
                                                    )
                                                ],
                                                className="tab-content",
                                            ),
                                        ),
                                        dcc.Tab(
                                            label="Data-mined conditions",
                                            children=html.Div(
                                                [
                                                    html.Section(
                                                        [
                                                            _section_heading(
                                                                "Data-mined favorable conditions",
                                                                DASHBOARD_EXPLANATIONS["conditions"],
                                                            ),
                                                            dash_table.DataTable(
                                                                id="detail-condition-table",
                                                                page_size=15,
                                                                sort_action="native",
                                                                filter_action="native",
                                                                style_cell={
                                                                    "whiteSpace": "normal",
                                                                    "height": "auto",
                                                                    "textAlign": "left",
                                                                },
                                                                style_table={"overflowX": "auto"},
                                                            ),
                                                            html.P(
                                                                id="detail-condition-explanation",
                                                                className="selection-summary condition-explanation",
                                                            ),
                                                        ]
                                                    )
                                                ],
                                                className="tab-content",
                                            ),
                                        ),
                                        dcc.Tab(
                                            label="Soft-vote errors",
                                            children=html.Div(
                                                [
                                                    html.Section(
                                                        [
                                                            html.H2("Soft-voting errors and fulfilled hypotheses"),
                                                            dash_table.DataTable(
                                                                id="detail-soft-failure-table",
                                                                page_size=15,
                                                                sort_action="native",
                                                                filter_action="native",
                                                                style_cell={
                                                                    "whiteSpace": "normal",
                                                                    "height": "auto",
                                                                    "textAlign": "left",
                                                                    "minWidth": "110px",
                                                                    "maxWidth": "360px",
                                                                },
                                                                style_table={"overflowX": "auto"},
                                                            ),
                                                        ]
                                                    )
                                                ],
                                                className="tab-content",
                                            ),
                                        ),
                                    ],
                                    className="detail-subtabs",
                                ),
                            ],
                            className="tab-content soft-analysis",
                        ),
                    ),
                ],
                parent_className="dashboard-tabs",
            ),
        ],
    )

    @app.callback(
        Output("selected-dataset-config", "children"),
        Output("selected-split-summary", "children"),
        Output("selected-calibration-chart", "figure"),
        Output("selected-headline-chart", "figure"),
        Output("selected-headline-table", "data"),
        Output("selected-headline-table", "columns"),
        Output("selected-integration-chart", "figure"),
        Output("selected-integration-table", "data"),
        Output("selected-integration-table", "columns"),
        Input("global-dataset-selector", "value"),
    )
    def update_selected_dataset(dataset: str) -> tuple[Any, ...]:
        summary = summaries[dataset]
        headline = _headline_results(summary)
        headline_table = [
            {
                "scenario": item["scenario"],
                "selected method": item["selected method"],
                "type": item["type"],
                "accuracy": item["accuracy display"],
                "gain vs soft": f"{item['gain vs soft']:+.2%}",
                "gap recovered": _percent(item["gap recovered"]),
            }
            for item in headline
        ]
        integrations = summary["rule_integrations"]
        strategies = {item["name"]: item["accuracy"] for item in summary["strategies"]}
        integration_table = [
            {
                "method": item["name"],
                "accuracy": _percent(item["accuracy"]),
                "net events": item["net_correct"],
                "recoveries": item["recoveries"],
                "harms": item["harms"],
                "gap recovered": _percent(item.get("gap_recovered_fraction", 0.0)),
                "parameters": json.dumps(item.get("parameters", {}), sort_keys=True),
                "description": item.get("description", ""),
            }
            for item in integrations
        ]
        config = summary.get("run_config", {})
        config_text = (
            f"{dataset} · {summary.get('test_events', 0):,} test events · "
            f"data proportion {config.get('data_prop', 'legacy')} · windows {config.get('windows', 'legacy')} · "
            f"seed {config.get('seed', 'legacy')} · stored in {result_sets[dataset]}"
        )
        return (
            config_text,
            (
                f"{summary.get('train_events', 0):,} train events → "
                f"{summary.get('calibration_events', 0):,} calibration events → "
                f"{summary.get('test_events', 0):,} held-out test events. "
                "Diamonds show calibration accuracy; circles show test accuracy."
            ),
            _calibration_generalization_figure(summary),
            _headline_figure(headline),
            headline_table,
            [{"name": column, "id": column} for column in headline_table[0]] if headline_table else [],
            _integration_figure(
                integrations,
                soft_accuracy=strategies.get("soft voting", 0.0),
                cheating_accuracy=strategies.get("cheating voting", 0.0),
            ),
            integration_table,
            [{"name": column, "id": column} for column in integration_table[0]] if integration_table else [],
        )

    @app.callback(
        Output("detail-dataset-notice", "children"),
        Output("detail-soft-accuracy", "children"),
        Output("detail-cheating-accuracy", "children"),
        Output("detail-oracle-gap", "children"),
        Output("detail-best-deployable", "children"),
        Output("detail-gap-closed", "children"),
        Output("detail-sequence-id", "options"),
        Output("detail-sequence-id", "value"),
        Output("detail-sequence-rule-tracking", "options"),
        Output("detail-rule-impact-chart", "figure"),
        Output("detail-rule-impact-table", "data"),
        Output("detail-rule-impact-table", "columns"),
        Output("detail-condition-table", "data"),
        Output("detail-condition-table", "columns"),
        Output("detail-condition-table", "tooltip_data"),
        Output("detail-soft-failure-table", "data"),
        Output("detail-soft-failure-table", "columns"),
        Input("global-dataset-selector", "value"),
    )
    def update_detailed_dataset(dataset: str) -> tuple[Any, ...]:
        summary, rows = detailed_results(dataset)
        soft_analysis = summary.get("soft_failure_analysis") or build_soft_failure_analysis([], rows)
        conditions = soft_analysis.get("condition_hypotheses", [])
        conditions_by_id = {condition["id"]: condition for condition in conditions}
        impacts = soft_analysis.get("rule_impacts", [])
        hypotheses_by_name = {hypothesis["name"]: hypothesis for hypothesis in summary.get("hypotheses", [])}
        sequences = list(dict.fromkeys(row["sequence_id"] for row in rows))
        sequence_candidate = _best_available_sequence_candidate(summary, rows)
        strategies = {item["name"]: item["accuracy"] for item in summary["strategies"]}
        soft_accuracy = strategies.get("soft voting", 0.0)
        cheating_accuracy = strategies.get("cheating voting", soft_accuracy)
        best_deployable = summary.get("best_rule_result") or {"accuracy": soft_accuracy}
        oracle_gap = cheating_accuracy - soft_accuracy
        gap_closed = (best_deployable["accuracy"] - soft_accuracy) / oracle_gap if oracle_gap else 0.0
        impact_rows = [
            {
                "rule": impact["name"],
                "family": impact["family"],
                "accuracy": _percent(impact["resulting_accuracy"]),
                "recoveries": impact["recoveries"],
                "harms": impact["harms"],
                "net correct": impact["net_correct"],
                "coverage": _percent(impact["coverage"]),
                "soft-error recall": _percent(impact["soft_error_recall"]),
                "description": hypotheses_by_name.get(impact["name"], {}).get(
                    "description", _rule_description(impact["name"])
                ),
                "selection policy": hypotheses_by_name.get(impact["name"], {}).get("selection_policy", ""),
                "process interpretation": hypotheses_by_name.get(impact["name"], {}).get("interpretation", ""),
            }
            for impact in impacts
        ]
        condition_rows = _condition_table_rows(conditions)
        failure_rows = _soft_failure_rows(rows, conditions_by_id)
        notice = (
            f"Detailed held-out event diagnostics for {dataset}. The event file is loaded only for the selected "
            "dataset, so cross-dataset comparison remains lightweight."
        )
        return (
            notice,
            _percent(soft_accuracy),
            _percent(cheating_accuracy),
            _percent(oracle_gap),
            _percent(best_deployable["accuracy"]),
            _percent(gap_closed),
            _sequence_options(rows, sequence_candidate),
            sequences[0] if sequences else None,
            [{"label": rule["name"], "value": rule["name"]} for rule in summary.get("hypotheses", [])],
            _rule_impact_figure(impacts),
            impact_rows,
            [{"name": column, "id": column} for column in impact_rows[0]] if impact_rows else [],
            condition_rows,
            [{"name": column, "id": column} for column in condition_rows[0]] if condition_rows else [],
            _condition_tooltips(conditions),
            failure_rows,
            [{"name": column, "id": column} for column in failure_rows[0]] if failure_rows else [],
        )

    @app.callback(
        Output("detail-condition-explanation", "children"),
        Input("detail-condition-table", "active_cell"),
        State("detail-condition-table", "data"),
    )
    def explain_selected_condition(active_cell: dict[str, Any] | None, data: list[dict[str, Any]] | None) -> str:
        """Keep a clicked condition's meaning visible in addition to cell hover help."""
        if not active_cell or not data:
            return "Hover a feature for a definition, or click a condition row to keep its explanation here."
        row_index = active_cell.get("row")
        if not isinstance(row_index, int) or not 0 <= row_index < len(data):
            return "Hover a feature for a definition, or click a condition row to keep its explanation here."
        feature = str(data[row_index].get("when feature", ""))
        value = str(data[row_index].get("has value", ""))
        return f"{feature} = {value}: {_condition_feature_description(feature)}"

    @app.callback(
        Output("detail-sequence-timeline", "figure"),
        Output("detail-sequence-summary", "children"),
        Input("global-dataset-selector", "value"),
        Input("detail-sequence-id", "value"),
        Input("detail-sequence-rule-tracking", "value"),
    )
    def update_detailed_sequence(
        dataset: str, sequence_id: str | None, tracked_rules: list[str]
    ) -> tuple[go.Figure, str]:
        summary, rows = detailed_results(dataset)
        selected = [row for row in rows if row["sequence_id"] == sequence_id]
        gap_count = sum(row["oracle_gap"] for row in selected)
        best_deployable = _best_available_sequence_candidate(summary, rows)
        method = best_deployable["name"] if best_deployable else "not available in this saved run"
        return (
            _sequence_figure(selected, summary["models"], best_deployable, tracked_rules),
            f"Best deployable shown: {method}. {gap_count} soft-voting failures were recoverable in this sequence.",
        )

    return app


def _event_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sequence": row["sequence_id"],
            "position": row["position_1based"],
            "prefix": row["prefix_text"],
            "actual": row["actual"],
            "soft": row["soft_prediction"],
            "correct models": ", ".join(row["correct_models"]),
            "agreement": row["agreement_count"],
            "pattern": row["suffix_3"],
        }
        for row in rows
    ]


def create_dashboard(results_dir: Path) -> Dash:  # noqa: PLR0915
    """Create a dashboard for one completed investigation."""
    if not (results_dir / "summary.json").exists():
        result_sets = discover_result_sets(results_dir)
        if result_sets:
            return create_comparison_dashboard(results_dir, result_sets)
    summary, rows = load_results(results_dir)
    gap_rows = [row for row in rows if row["oracle_gap"]]
    model_names = summary["models"]
    sequence_ids = list(dict.fromkeys(row["sequence_id"] for row in rows))
    best_deployable = _best_available_sequence_candidate(summary, rows)
    rule_families = sorted({rule["family"] for rule in summary["hypotheses"]})
    trackable_rule_names = [rule["name"] for rule in summary.get("hypotheses", [])]
    oracle_accuracy = next(item["accuracy"] for item in summary["strategies"] if item["name"] == "cheating voting")
    soft_accuracy = next(item["accuracy"] for item in summary["strategies"] if item["name"] == "soft voting")
    best_individual = max(summary.get("hypotheses", []), key=lambda item: item["accuracy"], default=None)
    integrations = summary.get("rule_integrations", [])
    best_integration = max(integrations, key=lambda item: item["accuracy"], default=None)
    integration_figure = _integration_figure(
        integrations,
        soft_accuracy=soft_accuracy,
        cheating_accuracy=oracle_accuracy,
    )
    integration_height = int(integration_figure.layout.height or 430)
    integration_rows = [
        {
            "method": result["name"],
            "accuracy": _percent(result["accuracy"]),
            "gain vs soft": f"{result['net_accuracy_delta']:+.2%}",
            "gap recovered": _percent(result.get("gap_recovered_fraction", 0.0)),
            "recoveries": result["recoveries"],
            "harms": result["harms"],
            "net events": result["net_correct"],
            "parameters and audit": json.dumps(result.get("parameters", {}), sort_keys=True),
            "description": result["description"],
        }
        for result in integrations
    ]
    headline_results = _headline_results(summary)
    headline_figure = _headline_figure(headline_results)
    headline_height = int(headline_figure.layout.height or 450)
    headline_rows = [
        {
            "scenario": result["scenario"],
            "selected method": result["selected method"],
            "type": result["type"],
            "accuracy": result["accuracy display"],
            "gain vs soft": f"{result['gain vs soft']:+.2%}",
            "gap recovered": _percent(result["gap recovered"]),
        }
        for result in headline_results
    ]
    calibration_figure = _calibration_generalization_figure(summary)
    calibration_height = int(calibration_figure.layout.height or 360)
    scenario_figure = _scenario_figure(summary)
    scenario_height = int(scenario_figure.layout.height or 430)
    scenario_rows = [
        {
            "scenario": scenario["name"],
            "type": scenario["family"],
            "accuracy": _percent(scenario["accuracy"]),
            "gain vs soft": f"{scenario['net_accuracy_delta']:+.2%}",
            "gap recovered": _percent(scenario["gap_recovered_fraction"]),
            "minimum agreeing rules": scenario["minimum_votes"],
            "description": (
                "Diagnostic best-case: succeeds when any included rule is correct."
                if scenario["oracle"]
                else "Uses a unique rule consensus; otherwise retains the soft-voting prediction."
            ),
            "rules": "; ".join(scenario["rule_names"]),
        }
        for scenario in summary.get("rule_scenarios", [])
    ]
    soft_analysis = summary.get("soft_failure_analysis")
    if soft_analysis is None:
        soft_analysis = build_soft_failure_analysis([], rows)
    conditions = soft_analysis["condition_hypotheses"]
    conditions_by_id = {condition["id"]: condition for condition in conditions}
    default_condition_ids = [
        condition["id"] for condition in sorted(conditions, key=lambda item: item["weight"], reverse=True)[:10]
    ]
    condition_table_rows = _condition_table_rows(conditions)
    soft_failure_table_rows = _soft_failure_rows(rows, conditions_by_id)
    rule_impacts = soft_analysis["rule_impacts"]
    best_rule = rule_impacts[0] if rule_impacts else None
    rule_impact_figure = _rule_impact_figure(rule_impacts)
    rule_impact_height = int(rule_impact_figure.layout.height or 430)
    condition_notice = (
        "Conditions were learned on calibration events and scored on held-out test events."
        if soft_analysis["condition_source"] == "calibration"
        else (
            "Exploratory fallback: this saved run predates calibration-based condition output, "
            "so conditions shown here were discovered on the test events. Rerun the investigation "
            "before treating their ranking as predictive evidence."
        )
    )
    condition_description = (
        "Read each row as: when the feature has this value, calibration favored the learned choice over soft "
        "voting. Calibration advantage selects the condition; recoveries, harms, and net improvement measure "
        "what happened on held-out test events. These rows are separate from the active and archived rule grids."
    )

    app = Dash(
        __name__,
        title=f"Voting investigation · {summary['dataset']}",
        assets_folder=str(Path(__file__).with_name("assets")),
    )
    app.layout = html.Main(
        className="investigation-shell",
        children=[
            html.Header(
                [
                    html.Div(
                        [
                            html.H1("Voting model-selection investigation"),
                            html.P(
                                f"{summary['dataset']} · {summary['test_events']:,} test events · "
                                f"{len(sequence_ids):,} sequences"
                            ),
                        ]
                    ),
                    html.Code(str(results_dir)),
                ],
                className="investigation-header",
            ),
            html.Section(
                [
                    html.Article(
                        [
                            html.Span(
                                "Soft voting",
                                title="Equal-weight average of all constituent probability distributions.",
                            ),
                            html.Strong(_percent(soft_accuracy)),
                        ]
                    ),
                    html.Article(
                        [
                            html.Span(
                                "Cheating baseline",
                                title=DASHBOARD_EXPLANATIONS["cheating"],
                            ),
                            html.Strong(_percent(oracle_accuracy)),
                        ]
                    ),
                    html.Article(
                        [
                            html.Span(
                                "Best individual rule",
                                title=best_individual["name"] if best_individual else "No candidate rule available.",
                            ),
                            html.Strong(_percent(best_individual["accuracy"]) if best_individual else "n/a"),
                        ]
                    ),
                    html.Article(
                        [
                            html.Span(
                                "Best smart integration",
                                title=best_integration["name"]
                                if best_integration
                                else "Rerun to benchmark integrations.",
                            ),
                            html.Strong(_percent(best_integration["accuracy"]) if best_integration else "n/a"),
                        ]
                    ),
                    html.Article(
                        [
                            html.Span(
                                "Baseline gap recovered",
                                title="Accuracy gained over soft voting divided by the cheating-baseline gap.",
                            ),
                            html.Strong(_percent(summary.get("recovered_gap_fraction", 0.0))),
                        ]
                    ),
                ],
                className="metric-strip",
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Overview",
                        children=html.Div(
                            [
                                html.Section(
                                    [
                                        _section_heading(
                                            "Data split and generalization check",
                                            "Training events update the constituent process models and are never "
                                            "reported as an accuracy score. Calibration events fit and select rules; "
                                            "their score is therefore in-sample. Test events are kept untouched until "
                                            "the final evaluation. A large calibration-to-test drop is evidence that a "
                                            "selector may be overfitting its calibration data.",
                                        ),
                                        html.P(
                                            f"{summary['train_events']:,} train events → "
                                            f"{summary['calibration_events']:,} calibration events → "
                                            f"{summary['test_events']:,} held-out test events. "
                                            "Circles are held-out test accuracy; diamonds are calibration accuracy.",
                                            className="selection-summary",
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                figure=calibration_figure,
                                                config=OVERVIEW_GRAPH_CONFIG,
                                                style={"height": f"{calibration_height}px"},
                                                className="analysis-fixed-graph",
                                            ),
                                            style={"height": f"{calibration_height}px"},
                                            className="analysis-graph-frame",
                                        ),
                                    ],
                                    className="analysis-graph-panel",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Headline benchmark results",
                                            DASHBOARD_EXPLANATIONS["headline"],
                                        ),
                                        html.P(
                                            "The results requested for comparison are shown here and remain "
                                            "available in the detailed scenario and integration tabs.",
                                            className="selection-summary",
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                figure=headline_figure,
                                                config=OVERVIEW_GRAPH_CONFIG,
                                                style={"height": f"{headline_height}px"},
                                                className="analysis-fixed-graph",
                                            ),
                                            style={"height": f"{headline_height}px"},
                                            className="analysis-graph-frame",
                                        ),
                                        dash_table.DataTable(
                                            data=headline_rows,
                                            columns=(
                                                [{"name": column, "id": column} for column in headline_rows[0]]
                                                if headline_rows
                                                else []
                                            ),
                                            style_cell={
                                                "whiteSpace": "normal",
                                                "height": "auto",
                                                "textAlign": "left",
                                            },
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ],
                                    className="analysis-graph-panel",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Baselines and candidate rules",
                                            DASHBOARD_EXPLANATIONS["overview"],
                                        ),
                                        dcc.Graph(
                                            figure=_accuracy_figure(summary),
                                            config=OVERVIEW_GRAPH_CONFIG,
                                            style={"height": "460px"},
                                        ),
                                    ],
                                    className="overview-panel overview-main-panel",
                                ),
                                html.Div(
                                    [
                                        html.Section(
                                            [
                                                html.H2("Gap by model agreement"),
                                                dcc.Graph(
                                                    figure=_agreement_figure(rows),
                                                    config=OVERVIEW_GRAPH_CONFIG,
                                                    style={"height": "360px"},
                                                ),
                                            ],
                                            className="overview-panel",
                                        ),
                                        html.Section(
                                            [
                                                html.H2("Gap by sequence position"),
                                                dcc.Graph(
                                                    figure=_position_figure(rows),
                                                    config=OVERVIEW_GRAPH_CONFIG,
                                                    style={"height": "360px"},
                                                ),
                                            ],
                                            className="overview-panel",
                                        ),
                                    ],
                                    className="two-column",
                                ),
                            ],
                            className="tab-content",
                        ),
                    ),
                    dcc.Tab(
                        label="Recoverable soft failures",
                        children=html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            [
                                                "Correct model",
                                                dcc.Dropdown(["All", *model_names], "All", id="gap-model"),
                                            ]
                                        ),
                                        html.Label(
                                            [
                                                "Minimum agreement",
                                                dcc.Slider(
                                                    1,
                                                    len(model_names),
                                                    1,
                                                    value=1,
                                                    id="gap-agreement",
                                                    marks=None,
                                                    tooltip={"always_visible": True},
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="control-row",
                                ),
                                html.P(id="gap-count", className="selection-summary"),
                                dash_table.DataTable(
                                    id="gap-table",
                                    page_size=15,
                                    sort_action="native",
                                    filter_action="native",
                                    style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                                    style_table={"overflowX": "auto"},
                                ),
                            ],
                            className="tab-content",
                        ),
                    ),
                    dcc.Tab(
                        label="Sequence explorer",
                        children=html.Div(
                            [
                                html.Label(
                                    [
                                        "Sequence",
                                        dcc.Dropdown(
                                            [
                                                option
                                                for option in _sequence_options(rows, best_deployable)
                                            ],
                                            sequence_ids[0] if sequence_ids else None,
                                            id="sequence-id",
                                        ),
                                    ],
                                    className="sequence-control",
                                ),
                                html.Label(
                                    [
                                        "Track active rules",
                                        dcc.Checklist(
                                            options=[{"label": name, "value": name} for name in trackable_rule_names],
                                            value=[],
                                            id="sequence-rule-tracking",
                                            inline=True,
                                        ),
                                    ],
                                    className="sequence-control",
                                ),
                                dcc.Graph(id="sequence-timeline"),
                                html.P(id="sequence-summary", className="selection-summary"),
                            ],
                            className="tab-content",
                        ),
                    ),
                    dcc.Tab(
                        label="Hypotheses",
                        children=html.Div(
                            [
                                html.P(
                                    DASHBOARD_EXPLANATIONS["hypotheses"],
                                    className="analysis-notice",
                                ),
                                html.Label(
                                    ["Rule family", dcc.Dropdown(["All", *rule_families], "All", id="rule-family")],
                                    className="sequence-control",
                                ),
                                dcc.Graph(id="hypothesis-chart"),
                                dash_table.DataTable(
                                    id="hypothesis-table",
                                    page_size=15,
                                    sort_action="native",
                                    style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                                    style_table={"overflowX": "auto"},
                                ),
                            ],
                            className="tab-content",
                        ),
                    ),
                    dcc.Tab(
                        label="Rule-set scenarios",
                        children=html.Div(
                            [
                                html.Section(
                                    [
                                        _section_heading(
                                            "Performance of selected rule combinations",
                                            DASHBOARD_EXPLANATIONS["scenarios"],
                                        ),
                                        html.P(
                                            "Consensus scenarios fall back to soft voting when too few rules agree. "
                                            "Oracle scenarios are diagnostic ceilings, not deployable selectors.",
                                            className="selection-summary",
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                figure=scenario_figure,
                                                config=OVERVIEW_GRAPH_CONFIG,
                                                style={"height": f"{scenario_height}px"},
                                                className="analysis-fixed-graph",
                                            ),
                                            style={"height": f"{scenario_height}px"},
                                            className="analysis-graph-frame",
                                        ),
                                    ],
                                    className="analysis-graph-panel",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Scenario definitions and recovered gap",
                                            DASHBOARD_EXPLANATIONS["scenario_gap"],
                                        ),
                                        dash_table.DataTable(
                                            data=scenario_rows,
                                            columns=(
                                                [{"name": column, "id": column} for column in scenario_rows[0]]
                                                if scenario_rows
                                                else []
                                            ),
                                            page_size=12,
                                            sort_action="native",
                                            filter_action="native",
                                            style_cell={
                                                "whiteSpace": "normal",
                                                "height": "auto",
                                                "textAlign": "left",
                                            },
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ]
                                ),
                            ],
                            className="tab-content",
                        ),
                    ),
                    dcc.Tab(
                        label="Integration methods",
                        children=html.Div(
                            [
                                html.P(
                                    "The active stack uses fixed, positive, model-local multipliers and is evaluated "
                                    "on the held-out test split. Reserved calibration events do not gate it.",
                                    className="analysis-notice",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Smart rule-integration performance",
                                            DASHBOARD_EXPLANATIONS["integration"],
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                figure=integration_figure,
                                                config=OVERVIEW_GRAPH_CONFIG,
                                                style={"height": f"{integration_height}px"},
                                                className="analysis-fixed-graph",
                                            ),
                                            style={"height": f"{integration_height}px"},
                                            className="analysis-graph-frame",
                                        ),
                                    ],
                                    className="analysis-graph-panel",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Method definitions, fitted parameters, and impact",
                                            (
                                                "Recoveries are repaired soft-voting errors; harms are previously "
                                                "correct soft predictions changed to a wrong answer."
                                            ),
                                        ),
                                        dash_table.DataTable(
                                            data=integration_rows,
                                            columns=(
                                                [{"name": column, "id": column} for column in integration_rows[0]]
                                                if integration_rows
                                                else []
                                            ),
                                            page_size=12,
                                            sort_action="native",
                                            filter_action="native",
                                            style_cell={
                                                "whiteSpace": "normal",
                                                "height": "auto",
                                                "textAlign": "left",
                                                "minWidth": "110px",
                                                "maxWidth": "420px",
                                            },
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ]
                                ),
                            ],
                            className="tab-content",
                        ),
                    ),
                    dcc.Tab(
                        label="Soft-vote analysis",
                        children=html.Div(
                            [
                                html.P(condition_notice, className="analysis-notice"),
                                html.Section(
                                    [
                                        html.Article(
                                            [
                                                html.Span("Soft-voting errors"),
                                                html.Strong(f"{soft_analysis['soft_failures']:,}"),
                                            ]
                                        ),
                                        html.Article(
                                            [
                                                html.Span("Recoverable by a constituent"),
                                                html.Strong(f"{soft_analysis['recoverable_soft_failures']:,}"),
                                            ]
                                        ),
                                        html.Article(
                                            [
                                                html.Span("Best rule net impact"),
                                                html.Strong(
                                                    f"{best_rule['net_correct']:+d} events" if best_rule else "n/a"
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="metric-strip analysis-metrics",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Candidate rule impact relative to soft voting",
                                            DASHBOARD_EXPLANATIONS["impact"],
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                figure=rule_impact_figure,
                                                config=OVERVIEW_GRAPH_CONFIG,
                                                style={"height": f"{rule_impact_height}px"},
                                                className="analysis-fixed-graph",
                                            ),
                                            style={"height": f"{rule_impact_height}px"},
                                            className="analysis-graph-frame",
                                        ),
                                    ],
                                    className="analysis-graph-panel",
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Data-mined favorable conditions",
                                            DASHBOARD_EXPLANATIONS["conditions"],
                                        ),
                                        html.P(condition_description, className="selection-summary"),
                                        dash_table.DataTable(
                                            data=condition_table_rows,
                                            columns=(
                                                [{"name": column, "id": column} for column in condition_table_rows[0]]
                                                if condition_table_rows
                                                else []
                                            ),
                                            page_size=15,
                                            sort_action="native",
                                            filter_action="native",
                                            style_cell={
                                                "whiteSpace": "normal",
                                                "height": "auto",
                                                "textAlign": "left",
                                            },
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ]
                                ),
                                html.Section(
                                    [
                                        _section_heading(
                                            "Weighted combination of favorable conditions",
                                            DASHBOARD_EXPLANATIONS["weighted"],
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    [
                                                        "Conditions",
                                                        dcc.Dropdown(
                                                            [
                                                                {
                                                                    "label": _condition_label(condition),
                                                                    "value": condition["id"],
                                                                }
                                                                for condition in sorted(
                                                                    conditions,
                                                                    key=lambda item: item["weight"],
                                                                    reverse=True,
                                                                )
                                                            ],
                                                            default_condition_ids,
                                                            multi=True,
                                                            id="weighted-conditions",
                                                        ),
                                                    ]
                                                ),
                                                html.Label(
                                                    [
                                                        "Weighting",
                                                        dcc.Dropdown(
                                                            [
                                                                {
                                                                    "label": "Calibration gain x sqrt(support)",
                                                                    "value": "calibrated",
                                                                },
                                                                {"label": "Calibration gain", "value": "gain"},
                                                                {"label": "Equal vote", "value": "equal"},
                                                            ],
                                                            "calibrated",
                                                            clearable=False,
                                                            id="weight-mode",
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="control-row weighted-controls",
                                        ),
                                        html.P(id="weighted-summary", className="selection-summary"),
                                        html.Div(
                                            dcc.Graph(
                                                id="weighted-model-chart",
                                                config=OVERVIEW_GRAPH_CONFIG,
                                                style={"height": "330px"},
                                                className="analysis-fixed-graph",
                                            ),
                                            style={"height": "330px"},
                                            className="analysis-graph-frame",
                                        ),
                                    ],
                                    className="analysis-graph-panel",
                                ),
                                html.Section(
                                    [
                                        html.H2("Every soft-voting error and its fulfilled hypotheses"),
                                        dash_table.DataTable(
                                            data=soft_failure_table_rows,
                                            columns=(
                                                [
                                                    {"name": column, "id": column}
                                                    for column in soft_failure_table_rows[0]
                                                ]
                                                if soft_failure_table_rows
                                                else []
                                            ),
                                            page_size=15,
                                            sort_action="native",
                                            filter_action="native",
                                            style_cell={
                                                "whiteSpace": "normal",
                                                "height": "auto",
                                                "textAlign": "left",
                                                "minWidth": "110px",
                                                "maxWidth": "360px",
                                            },
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ]
                                ),
                            ],
                            className="tab-content soft-analysis",
                        ),
                    ),
                ]
            ),
        ],
    )

    @callback(
        Output("gap-table", "data"),
        Output("gap-table", "columns"),
        Output("gap-count", "children"),
        Input("gap-model", "value"),
        Input("gap-agreement", "value"),
    )
    def update_gap_table(
        model_name: str,
        minimum_agreement: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], str]:
        filtered = [
            row
            for row in gap_rows
            if row["agreement_count"] >= minimum_agreement
            and (model_name == "All" or model_name in row["correct_models"])
        ]
        table_rows = _event_table_rows(filtered)
        columns = [{"name": column, "id": column} for column in table_rows[0]] if table_rows else []
        return table_rows, columns, f"{len(filtered):,} recoverable soft-voting failures match these filters."

    @callback(
        Output("sequence-timeline", "figure"),
        Output("sequence-summary", "children"),
        Input("sequence-id", "value"),
        Input("sequence-rule-tracking", "value"),
    )
    def update_sequence(sequence_id: str, tracked_rules: list[str]) -> tuple[go.Figure, str]:
        selected = [row for row in rows if row["sequence_id"] == sequence_id]
        gap_count = sum(row["oracle_gap"] for row in selected)
        best_deployable = _best_available_sequence_candidate(summary, rows)
        method = best_deployable["name"] if best_deployable else "not available in this saved run"
        return (
            _sequence_figure(selected, model_names, best_deployable, tracked_rules),
            f"Best deployable shown: {method}. {gap_count} soft-voting failures were recoverable in this sequence.",
        )

    @callback(
        Output("hypothesis-chart", "figure"),
        Output("hypothesis-table", "data"),
        Output("hypothesis-table", "columns"),
        Input("rule-family", "value"),
    )
    def update_hypotheses(
        family: str,
    ) -> tuple[go.Figure, list[dict[str, str]], list[dict[str, str]]]:
        selected = [rule for rule in summary["hypotheses"] if family == "All" or rule["family"] == family]
        frame = pd.DataFrame(selected)
        if not frame.empty:
            frame["description"] = frame.apply(
                lambda rule: rule.get("description") or _rule_description(rule["name"]), axis=1
            )
        figure = px.bar(
            frame,
            x="accuracy",
            y="name",
            color="family",
            orientation="h",
            text=frame["accuracy"].map(_percent) if not frame.empty else None,
            hover_data={
                "description": True,
                "correct": True,
                "total": True,
                "calibration_accuracy": ":.1%",
                "calibration_correct": True,
                "calibration_total": True,
            }
            if not frame.empty
            else None,
            labels={"accuracy": "Held-out test accuracy", "name": "Hypothesis", "family": "Family"},
        )
        figure.update_layout(yaxis={"categoryorder": "total ascending"}, margin={"l": 220, "r": 20})
        figure.update_xaxes(tickformat=".0%", range=[0, 1])
        table_rows = [
            {
                "hypothesis": rule["name"],
                "family": rule["family"],
                "description": rule.get("description") or _rule_description(rule["name"]),
                "selection policy": rule.get("selection_policy", ""),
                "process interpretation": rule.get("interpretation", ""),
                "calibration accuracy (fit data)": (
                    _percent(rule["calibration_accuracy"])
                    if rule.get("calibration_accuracy") is not None
                    else "not recorded"
                ),
                "held-out test accuracy": _percent(rule["accuracy"]),
                "generalization gap": (
                    f"{rule['calibration_accuracy'] - rule['accuracy']:+.1%}"
                    if rule.get("calibration_accuracy") is not None
                    else "not recorded"
                ),
                "calibration correct / total": (
                    f"{rule.get('calibration_correct', 0)} / {rule.get('calibration_total', 0)}"
                ),
                "test correct / total": f"{rule['correct']} / {rule['total']}",
                "selected models": ", ".join(f"{name}: {count}" for name, count in rule["selected_models"].items()),
            }
            for rule in selected
        ]
        columns = [{"name": column, "id": column} for column in table_rows[0]] if table_rows else []
        return figure, table_rows, columns

    @callback(
        Output("weighted-summary", "children"),
        Output("weighted-model-chart", "figure"),
        Input("weighted-conditions", "value"),
        Input("weight-mode", "value"),
    )
    def update_weighted_conditions(selected_ids: list[str] | None, weight_mode: str) -> tuple[str, go.Figure]:
        result = evaluate_weighted_conditions(
            rows,
            conditions,
            selected_ids=set(selected_ids or []),
            weight_mode=weight_mode,
        )
        summary_text = (
            f"{result['conditions']} conditions · {result['triggered']} prediction changes · "
            f"{result['recoveries']} soft errors recovered · {result['harms']} correct soft predictions harmed · "
            f"net {result['net_correct']:+d} events ({result['net_accuracy_delta']:+.2%}) · "
            f"resulting accuracy {result['resulting_accuracy']:.2%}."
        )
        selected_models = pd.DataFrame(
            [{"model": model_name, "events": count} for model_name, count in result["selected_models"].items()]
        )
        figure = px.bar(
            selected_models,
            x="model",
            y="events",
            text="events",
            labels={"model": "Selected predictor", "events": "Events"},
        )
        figure.update_layout(height=330, margin={"l": 50, "r": 20, "t": 20, "b": 50})
        return summary_text, figure

    return app


def run_dashboard(results_dir: Path, *, port: int = 8050, debug: bool = False) -> None:
    """Serve the interactive investigation dashboard."""
    app = create_dashboard(results_dir)
    app.run(debug=debug, port=port)
