"""Interactive Dash application for saved voting-investigation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dash_table, dcc, html

from logicsponge.processmining.voting_investigation import (
    build_soft_failure_analysis,
    evaluate_rule_scenarios,
    evaluate_weighted_conditions,
)

OVERVIEW_GRAPH_CONFIG = {"responsive": True, "displayModeBar": False}


def load_results(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a saved summary and its event-level diagnostics."""
    summary_path = results_dir / "summary.json"
    events_path = results_dir / "events.jsonl"
    if not summary_path.exists() or not events_path.exists():
        msg = f"Expected summary.json and events.jsonl in {results_dir}"
        raise FileNotFoundError(msg)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row.pop("hard_prediction", None)
        row.pop("hard_correct", None)
        row["oracle_prediction"] = row["actual"] if row["correct_models"] else row["soft_prediction"]
        row["oracle_correct"] = row["oracle_prediction"] == row["actual"]
        row["oracle_gap"] = row["oracle_correct"] and not row["soft_correct"]
    soft_accuracy = sum(row["soft_correct"] for row in rows) / len(rows) if rows else 0.0
    oracle_accuracy = sum(row["oracle_correct"] for row in rows) / len(rows) if rows else 0.0
    summary["strategies"] = [
        {"name": "soft voting", "accuracy": soft_accuracy},
        {"name": "cheating voting", "accuracy": oracle_accuracy},
    ]
    summary["rule_scenarios"] = evaluate_rule_scenarios(rows, summary.get("hypotheses", []))
    candidates = [
        *summary.get("hypotheses", []),
        *(scenario for scenario in summary["rule_scenarios"] if not scenario["oracle"]),
    ]
    best = max(candidates, key=lambda item: item["accuracy"], default=None)
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
    figure = px.bar(
        frame,
        x="accuracy",
        y="name",
        orientation="h",
        color="family" if not frame.empty else None,
        text=frame["accuracy"].map(_percent) if not frame.empty else None,
        labels={"accuracy": "Accuracy", "name": "Rule-set scenario", "family": "Scenario type"},
    )
    figure.update_layout(
        height=max(430, 58 * len(scenarios)),
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 230, "r": 20, "t": 20, "b": 50},
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1])
    return figure


def _condition_table_rows(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition["name"],
            "family": condition["family"],
            "recommended model": condition["recommended_model"],
            "calibration support": condition["calibration_support"],
            "calibration model accuracy": _percent(condition["calibration_model_accuracy"]),
            "calibration gain": _percent(condition["calibration_gain"]),
            "calibrated weight": round(condition["weight"], 3),
            "test support": condition["test_support"],
            "soft errors": condition["soft_errors"],
            "recoveries": condition["recoveries"],
            "harms": condition["harms"],
            "net correct": condition["net_correct"],
            "conditional delta": _percent(condition["conditional_delta"]),
            "soft-error recall": _percent(condition["soft_error_recall"]),
            "decisive precision": _percent(condition["decisive_precision"]),
        }
        for condition in conditions
    ]


def _soft_failure_rows(
    rows: list[dict[str, Any]],
    conditions_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
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
            conditions_by_id[condition_id]["name"]
            for condition_id in row.get("condition_matches", [])
            if condition_id in conditions_by_id
        ]
        table_rows.append(
            {
                "sequence": row["sequence_id"],
                "position": row["position_1based"],
                "actual": row["actual"],
                "soft prediction": row["soft_prediction"],
                "correct models": ", ".join(row["correct_models"]),
                "agreement": row["agreement_count"],
                "last activities": row["suffix_3"],
                "rules that override soft": "; ".join(active_rules),
                "rules that recover error": "; ".join(recovering_rules),
                "favorable conditions fulfilled": "; ".join(matched_conditions),
            }
        )
    return table_rows


def _sequence_figure(sequence_rows: list[dict[str, Any]], model_names: list[str]) -> go.Figure:
    lanes = ["actual", "soft_prediction", "oracle_prediction", *model_names]
    labels = {
        "actual": "Actual",
        "soft_prediction": "Soft voting",
        "oracle_prediction": "Cheating voting",
        **{name: name for name in model_names},
    }
    figure = go.Figure()
    for lane_index, lane in enumerate(lanes):
        predictions: list[str] = []
        correct: list[bool] = []
        for row in sequence_rows:
            if lane in {"actual", "soft_prediction", "oracle_prediction"}:
                prediction = row[lane]
            else:
                prediction = next(model["prediction"] for model in row["models"] if model["name"] == lane)
            predictions.append(prediction)
            correct.append(lane == "actual" or prediction == row["actual"])
        figure.add_trace(
            go.Scatter(
                x=[row["position_1based"] for row in sequence_rows],
                y=[lane_index] * len(sequence_rows),
                mode="lines+markers+text",
                name=labels[lane],
                text=predictions,
                textposition="top center",
                marker={
                    "size": 11,
                    "color": ["#16865b" if is_correct else "#d1495b" for is_correct in correct],
                    "symbol": ["circle" if is_correct else "x" for is_correct in correct],
                },
                customdata=[[row["actual"], row["agreement_count"], row["prefix_text"]] for row in sequence_rows],
                hovertemplate=(
                    f"{labels[lane]}: %{{text}}<br>Actual: %{{customdata[0]}}<br>"
                    "Agreement: %{customdata[1]}<br>Prefix: %{customdata[2]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        height=max(430, 70 * len(lanes)),
        margin={"l": 120, "r": 30, "t": 20, "b": 50},
        showlegend=False,
        yaxis={"tickmode": "array", "tickvals": list(range(len(lanes))), "ticktext": [labels[lane] for lane in lanes]},
        xaxis_title="Position in sequence",
    )
    return figure


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


def create_dashboard(results_dir: Path) -> Dash:
    """Create a dashboard for one completed investigation."""
    summary, rows = load_results(results_dir)
    gap_rows = [row for row in rows if row["oracle_gap"]]
    model_names = summary["models"]
    sequence_ids = list(dict.fromkeys(row["sequence_id"] for row in rows))
    rule_families = sorted({rule["family"] for rule in summary["hypotheses"]})
    oracle_accuracy = next(item["accuracy"] for item in summary["strategies"] if item["name"] == "cheating voting")
    soft_accuracy = next(item["accuracy"] for item in summary["strategies"] if item["name"] == "soft voting")
    best_rule_result = summary.get("best_rule_result")
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
        "Recommendations are learned from the calibration split. Test net impact is recoveries "
        "minus harms when the condition is fulfilled."
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
                    html.Article([html.Span("Soft voting"), html.Strong(_percent(soft_accuracy))]),
                    html.Article([html.Span("Oracle ceiling"), html.Strong(_percent(oracle_accuracy))]),
                    html.Article(
                        [
                            html.Span("Best selected rule"),
                            html.Strong(_percent(best_rule_result["accuracy"]) if best_rule_result else "n/a"),
                        ]
                    ),
                    html.Article(
                        [
                            html.Span("Oracle gap recovered"),
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
                                        html.H2("Accuracy ceiling and candidate rules"),
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
                                                {"label": sequence_id, "value": sequence_id}
                                                for sequence_id in sequence_ids
                                            ],
                                            sequence_ids[0] if sequence_ids else None,
                                            id="sequence-id",
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
                                        html.H2("Performance of selected rule combinations"),
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
                                        html.H2("Scenario definitions and recovered gap"),
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
                                        html.H2("Candidate rule impact relative to soft voting"),
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
                                        html.H2("Favorable conditional hypotheses"),
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
                                        html.H2("Weighted combination of favorable conditions"),
                                        html.Div(
                                            [
                                                html.Label(
                                                    [
                                                        "Conditions",
                                                        dcc.Dropdown(
                                                            [
                                                                {
                                                                    "label": condition["name"],
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
        Output("sequence-timeline", "figure"), Output("sequence-summary", "children"), Input("sequence-id", "value")
    )
    def update_sequence(sequence_id: str) -> tuple[go.Figure, str]:
        selected = [row for row in rows if row["sequence_id"] == sequence_id]
        gap_count = sum(row["oracle_gap"] for row in selected)
        return _sequence_figure(
            selected, model_names
        ), f"{gap_count} soft-voting failures were recoverable in this sequence."

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
        figure = px.bar(
            frame,
            x="accuracy",
            y="name",
            color="family",
            orientation="h",
            text=frame["accuracy"].map(_percent) if not frame.empty else None,
            labels={"accuracy": "Accuracy", "name": "Hypothesis", "family": "Family"},
        )
        figure.update_layout(yaxis={"categoryorder": "total ascending"}, margin={"l": 220, "r": 20})
        figure.update_xaxes(tickformat=".0%", range=[0, 1])
        table_rows = [
            {
                "hypothesis": rule["name"],
                "family": rule["family"],
                "accuracy": _percent(rule["accuracy"]),
                "correct / total": f"{rule['correct']} / {rule['total']}",
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
