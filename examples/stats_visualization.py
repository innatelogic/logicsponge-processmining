"""Module to visualize the correctness of states based on various factors.

This module reads state statistics data and visualizes correctness metrics
based on visits, level, and state ID.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class PerStateStats:
    """Class to store statistics per state."""

    def __init__(self, state_id: int) -> None:
        """Initialize the statistics for a given state."""
        self.state_id = state_id
        self.total_predictions = 0
        self.correct_predictions = 0
        self.wrong_predictions = 0
        self.empty_predictions = 0
        self.level = 0
        self.visits = 0


def get_latest_stats_file(folder: Path = Path("results")) -> tuple[Path, str]:
    """Get the latest stats file from the specified folder.

    It considers files matching "stats_batch_*.json" or "stats_streaming_*.json".

    Args:
        folder (Path): The folder to search for stats files.

    Returns:
        tuple[Path, str]: A tuple containing the latest stats file path
                           and its mode ("batch" or "streaming").

    Raises:
        FileNotFoundError: If no stats files are found.

    """
    batch_files = list(folder.glob("stats_batch_*.json"))
    streaming_files = list(folder.glob("stats_streaming_*.csv"))
    stats_files = batch_files + streaming_files

    if not stats_files:
        msg = f"No stats files (stats_batch_*.json or stats_streaming_*.json) found in {folder}"
        raise FileNotFoundError(msg)

    # Match and extract mode and numeric ID
    def extract_info(file: Path) -> tuple[str | None, int]:
        match = re.search(r"stats_(batch|streaming)_(\d+)\.(json|csv)", file.name)
        if match:
            return match.group(1), int(match.group(2))
        return None, -1  # Should not happen if glob is correct

    latest_file = None
    latest_id = -1
    latest_mode = ""

    for file in stats_files:
        mode, file_id = extract_info(file)
        if mode is not None and file_id > latest_id:
            latest_id = file_id
            latest_file = file
            latest_mode = mode

    if latest_file is None:
        # This case should ideally not be reached if stats_files is not empty
        # and regex works as expected.
        msg = f"Could not determine the latest stats file from found files in {folder}"
        raise FileNotFoundError(msg)

    return latest_file, latest_mode


def load_stats(file_path: Path) -> list[dict]:
    """Load stats from the specified JSON file.

    Args:
        file_path (Path): The path to the JSON file.

    Returns:
        list[dict]: A list of dictionaries containing the stats for each model.

    """
    with file_path.open("r") as file:
        return json.load(file)


def prepare_data_from_per_state_stats(per_state_stats: dict[int, PerStateStats]) -> pd.DataFrame:
    """Convert per_state_stats dictionary to a pandas DataFrame.

    Args:
        per_state_stats: dictionary mapping state IDs to PerStateStats objects.

    Returns:
        pd.DataFrame: DataFrame with state statistics.

    """
    data = []
    for stats in per_state_stats.values():
        # Calculate correctness rate
        correctness_rate = stats.correct_predictions / stats.total_predictions if stats.total_predictions > 0 else 0
        # Calculate failure rate
        failure_rate = stats.wrong_predictions / stats.total_predictions if stats.total_predictions > 0 else 0

        data.append(
            {
                "state_id": stats.state_id,
                "visits": stats.visits,
                "level": stats.level,
                "total_predictions": stats.total_predictions,
                "correct_predictions": stats.correct_predictions,
                "wrong_predictions": stats.wrong_predictions,
                "empty_predictions": stats.empty_predictions,
                "correctness_rate": correctness_rate,
                "failure_rate": failure_rate,
            }
        )

    return pd.DataFrame(data)


def extract_global_stats_from_log_data(
        log_data: list[dict]
    ) -> dict[str, dict[str, float]]:
    """Extract global stats from loaded log data.

    Args:
        log_data: List of dictionaries containing model stats as loaded from JSON.

    Returns:
        dict[str, dict]: Dictionary mapping strategy names to their global stats.

    """
    global_stats: dict[str, dict[str, float]] = {}

    for model_data in log_data:
        strategy_name = model_data["strategy"]

        # Handle accuracy:
        # Batch JSON provides "strategy_accuracy" as 0-100.
        # Streaming CSV (transformed) might provide "accuracy" as 0-1.
        # We want final accuracy in global_stats to be 0-1 for plotting.
        final_accuracy = 0.0
        strategy_acc_val = model_data.get("strategy_accuracy") # From batch (0-100)
        if strategy_acc_val is not None:
            final_accuracy = strategy_acc_val / 100.0
        else:
            direct_acc_val = model_data.get("accuracy") # From streaming (0-1)
            if direct_acc_val is not None:
                final_accuracy = direct_acc_val
            # else final_accuracy remains 0.0

        global_stats[strategy_name] = {
            "accuracy": final_accuracy,
            "perplexity": model_data.get(
                "strategy_perplexity",
                model_data.get("perplexity", 0.0)
            ), # Also check direct perplexity
            "evaluation_time": model_data.get(
                "strategy_eval_time",
                model_data.get("evaluation_time", 0.0)
            ), # Also check direct eval_time
        }

    return global_stats

def extract_per_index_stats_from_log_data(log_data: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Extract per_index_stats dictionaries from loaded log data."""
    per_index_stats: dict[str, dict[str, list[float]]] = {}

    for index_data in log_data:
        strategy_name = index_data["strategy"]

        # Handle accuracy:
        # Batch JSON provides "strategy_accuracy" as 0-100.
        # Streaming CSV (transformed) might provide "accuracy" as 0-1.
        # We want final accuracy in global_stats to be 0-1 for plotting.
        final_accuracy = 0.0
        strategy_acc_val = index_data.get("strategy_accuracy") # From batch (0-100)
        if strategy_acc_val is not None:
            final_accuracy = strategy_acc_val / 100.0
        else:
            direct_acc_val = index_data.get("accuracy") # From streaming (0-1)
            if direct_acc_val is not None:
                final_accuracy = direct_acc_val
            # else final_accuracy remains 0.0

        if strategy_name not in per_index_stats:
            per_index_stats[strategy_name] = {
                "batch_index": [],
                "accuracy": [],
                "perplexity": [],
                "evaluation_time": [],
            }
        per_index_stats[strategy_name]["batch_index"].append(
            index_data.get("batch_index", 0)
        )
        per_index_stats[strategy_name]["accuracy"].append(final_accuracy)
        per_index_stats[strategy_name]["perplexity"].append(index_data.get(
            "strategy_perplexity",
            index_data.get("pp_harmonic_mean", 0.0)
        ))
        per_index_stats[strategy_name]["evaluation_time"].append(index_data.get(
            "strategy_eval_time",
            index_data.get("evaluation_time", 0.0)
        ))
    return per_index_stats

def extract_per_state_stats_from_log_data(log_data: list[dict]) -> dict[str, dict[int, PerStateStats]]:
    """Extract per_state_stats dictionaries from loaded log data.

    Args:
        log_data: List of dictionaries containing model stats as loaded from JSON.

    Returns:
        dict[str, dict[int, PerStateStats]]: Dictionary mapping strategy names to their per_state_stats.

    """
    per_strategy_stats = {}

    for model_data in log_data:
        strategy_name = model_data["strategy"]
        per_state_stats = {}

        # Extract state stats for each state
        # Assuming the log_data contains per_state_stats or something we can convert
        if "per_state_stats" in model_data:
            state_stats_data = model_data["per_state_stats"]

            for state_id_str, stats_dict in state_stats_data.items():
                state_id = int(state_id_str)
                state_stats = PerStateStats(state_id)

                # Copy stats from dict to object
                state_stats.total_predictions = stats_dict.get("total_predictions", 0)
                state_stats.correct_predictions = stats_dict.get("correct_predictions", 0)
                state_stats.wrong_predictions = stats_dict.get("wrong_predictions", 0)
                state_stats.empty_predictions = stats_dict.get("empty_predictions", 0)
                state_stats.level = stats_dict.get("level", 0)
                state_stats.visits = stats_dict.get("visits", 0)

                per_state_stats[state_id] = state_stats

        per_strategy_stats[strategy_name] = per_state_stats

    return per_strategy_stats


def add_trend_line(input_df: pd.DataFrame, color: str, strategy_name: str, group_key: str) -> go.Scatter:
    """Add a trend line to the figure based on the input DataFrame."""
    # Group by "group_key" and calculate weighted mean correctness rate
    visit_groups = input_df.groupby(group_key)
    total_predictions = (
        visit_groups["correct_predictions"].sum()
        + visit_groups["wrong_predictions"].sum()
        + visit_groups["empty_predictions"].sum()
    )
    correct_sum = visit_groups["correct_predictions"].sum()  # Sum of correct predictions

    # Calculate the weighted mean correctness rate
    mean_correctness = correct_sum / total_predictions.astype(float)
    mean_data = pd.DataFrame({group_key: mean_correctness.index, "weighted_correctness": mean_correctness.to_numpy()})

    # Add a scatter plot of the weighted means
    return go.Scatter(
        x=mean_data[group_key],
        y=mean_data["weighted_correctness"],
        mode="lines+markers",
        line={"color": color, "width": 2},
        name=f"{strategy_name} (weighted mean)",
        showlegend=False,
        hoverinfo="text",
        hovertext=[
            f"Visits: {v}, Mean correctness: {c:.3f}, n={total_predictions[v]}"
            for v, c in zip(mean_data[group_key], mean_data["weighted_correctness"], strict=False)
        ],
        visible=True,
    )


def plot_stats(  # noqa: C901, PLR0915, PLR0912
    log_data: list[dict],
    # stats_to_plot: list[str] | None = None,
    *,
    per_index_df: pd.DataFrame | None = None,
    visible_models: list[str] | None = None,
    add_by_state_id: bool = False,
) -> None:
    """Plot state correctness statistics and global model performance based on log data.

    Args:
        log_data: list of dictionaries containing model stats as loaded from JSON.
        stats_to_plot: Optional list of specific stats to plot (not used in current implementation).
        visible_models: Optional list of model names to include in visualization.
        per_index_df: DataFrame containing per-batch-index statistics (optional, only for streaming).
        add_by_state_id: Whether to add a third row for state ID visualization.

    """
    # Filter models if visible_models is specified
    if visible_models:
        filtered_log_data = [model for model in log_data if model["strategy"] in visible_models]
    else:
        filtered_log_data = log_data

    # Extract global stats for each strategy
    global_stats_per_strategy = extract_global_stats_from_log_data(filtered_log_data)

    # # Extract per_index_stats for each strategy
    # per_index_stats = extract_per_index_stats_from_log_data(filtered_log_data)

    # Extract per_state_stats for each strategy
    per_strategy_stats = extract_per_state_stats_from_log_data(filtered_log_data)

    if not per_strategy_stats and not global_stats_per_strategy:
        print("No valid model data found for visualization.")
        return

    plot_configs = {
        "visits": {
            "title": "Correctness Rate vs. Number of Visits",
            "xaxis_title": "Number of Visits",
            "yaxis_title": "Correctness Rate",
            "yaxis_range": [0, 1],
            "data_type": "per_state",
        },
        "level": {
            "title": "Correctness Rate vs. Level",
            "xaxis_title": "Level",
            "yaxis_title": "Correctness Rate",
            "yaxis_range": [0, 1],
            "data_type": "per_state",
        }
    }
    if add_by_state_id:
        plot_configs["state_id"] = {
            "title": "Correctness Rate vs. State ID",
            "xaxis_title": "State ID",
            "yaxis_title": "Correctness Rate",
            "yaxis_range": [0, 1],
            "data_type": "per_state",
        }
    plot_configs["global_accuracy"] = {
        "title": "Global Accuracy vs. Evaluation Time",
        "xaxis_title": "Evaluation Time (s)",
        "yaxis_title": "Accuracy",
        "yaxis_range": [0, 1],
        "data_type": "global",
    }
    plot_configs["global_perplexity"] = {
        "title": "Global Perplexity vs. Evaluation Time",
        "xaxis_title": "Evaluation Time (s)",
        "yaxis_title": "Perplexity",
        "yaxis_autorange": True,
        "data_type": "global",
    }
    plot_configs["per_index_accuracy"] = {
        "title": "Accuracy vs. Index",
        "xaxis_title": "Index",
        "yaxis_title": "Accuracy",
        "yaxis_autorange": True,
        "data_type": "global",
    }
    plot_configs["per_index_pp_harmonic_mean"] = {
        "title": "Perplexity vs. Index",
        "xaxis_title": "Index",
        "yaxis_title": "Perplexity",
        "yaxis_autorange": True,
        "data_type": "global",
    }

    initial_plot_key = "visits" # Default plot to show

    fig = make_subplots(rows=1, cols=1)

    colors = [
        "rgb(31, 119, 180)", "rgb(255, 127, 14)", "rgb(44, 160, 44)", "rgb(214, 39, 40)",
        "rgb(148, 103, 189)", "rgb(140, 86, 75)", "rgb(227, 119, 194)", "rgb(127, 127, 127)",
        "rgb(188, 189, 34)", "rgb(23, 190, 207)",
    ]

    all_traces_info = [] # Stores {"trace": trace_object, "strategy": name, "plot_key": key}

    all_unique_strategy_names = sorted(
        set(per_strategy_stats.keys()) | set(global_stats_per_strategy.keys())
    )
    strategy_color_map = {
        name: colors[j % len(colors)] for j, name in enumerate(all_unique_strategy_names)
    }

    # Add Per-State Traces
    for strategy_name, current_per_state_stats in per_strategy_stats.items():
        if not current_per_state_stats:
            continue
        data_per_state = prepare_data_from_per_state_stats(current_per_state_stats)
        df_with_predictions = data_per_state[data_per_state["total_predictions"] > 0]
        if df_with_predictions.empty:
            continue

        sizes = 5 + (df_with_predictions["total_predictions"] / df_with_predictions["total_predictions"].max() * 15)
        color = strategy_color_map.get(strategy_name, colors[-1])

        # Visits Plot
        trace_visits_scatter = go.Scatter(
            x=df_with_predictions["visits"], y=df_with_predictions["correctness_rate"], mode="markers",
            name=strategy_name, legendgroup=strategy_name, showlegend=True,
            marker={"size": sizes, "color": color, "opacity": 0.6},
            text=[f"Strategy: {strategy_name}, State ID: {sid}, Level: {lvl}, Visits: {vis}<br>"
                  f"Correct: {cor}, Wrong: {wrg}, Empty: {emp}, Rate: {rate:.2f}, Failure: {failure:.2f}"
                  for sid, lvl, vis, cor, wrg, emp, rate, failure in zip(
                      df_with_predictions["state_id"], df_with_predictions["level"], df_with_predictions["visits"],
                      df_with_predictions["correct_predictions"], df_with_predictions["wrong_predictions"],
                      df_with_predictions["empty_predictions"], df_with_predictions["correctness_rate"],
                      df_with_predictions["failure_rate"], strict=False)],
            hoverinfo="text", visible=(initial_plot_key == "visits"))
        fig.add_trace(trace_visits_scatter, row=1, col=1)
        all_traces_info.append(
            {
                "trace": trace_visits_scatter,
                "strategy": strategy_name,
                "plot_key": "visits",
                "is_legend_item": True
            }
        )

        if len(df_with_predictions) > 1:
            trend_visits = add_trend_line(df_with_predictions, color, strategy_name, "visits")
            trend_visits.legendgroup = strategy_name
            trend_visits.visible = (initial_plot_key == "visits")
            fig.add_trace(trend_visits, row=1, col=1)
            all_traces_info.append(
                {
                    "trace": trend_visits,
                    "strategy": strategy_name,
                    "plot_key": "visits",
                    "is_legend_item": False
                }
            )

        # Level Plot
        trace_level_scatter = go.Scatter(
            x=df_with_predictions["level"], y=df_with_predictions["correctness_rate"], mode="markers",
            name=strategy_name, legendgroup=strategy_name, showlegend=True,
            marker={"size": sizes, "color": color, "opacity": 0.6},
            text=[f"Strategy: {strategy_name}, State ID: {sid}, Level: {lvl}, Visits: {vis}<br>"
                  f"Correct: {cor}, Wrong: {wrg}, Empty: {emp}, Rate: {rate:.2f}"
                  for sid, lvl, vis, cor, wrg, emp, rate in zip(
                      df_with_predictions["state_id"], df_with_predictions["level"], df_with_predictions["visits"],
                      df_with_predictions["correct_predictions"], df_with_predictions["wrong_predictions"],
                      df_with_predictions["empty_predictions"], df_with_predictions["correctness_rate"], strict=False)],
            hoverinfo="text", visible=(initial_plot_key == "level"))
        fig.add_trace(trace_level_scatter, row=1, col=1)
        all_traces_info.append(
            {
                "trace": trace_level_scatter,
                "strategy": strategy_name,
                "plot_key": "level",
                "is_legend_item": False
            }
        )

        if len(df_with_predictions) > 1:
            trend_level = add_trend_line(df_with_predictions, color, strategy_name, "level")
            trend_level.legendgroup = strategy_name
            trend_level.visible = (initial_plot_key == "level")
            fig.add_trace(trend_level, row=1, col=1)
            all_traces_info.append(
                {
                    "trace": trend_level,
                    "strategy": strategy_name,
                    "plot_key": "level",
                    "is_legend_item": False
                }
            )

        # State ID Plot (if applicable)
        if add_by_state_id:
            trace_state_id_scatter = go.Scatter(
                x=df_with_predictions["state_id"], y=df_with_predictions["correctness_rate"], mode="markers",
                name=strategy_name, legendgroup=strategy_name, showlegend=True,
                marker={"size": sizes, "color": color, "opacity": 0.6},
                text=
                    [f"Strategy: {strategy_name}, State ID: {sid}, Level: {lvl}, Visits: {vis}<br>"
                      f"Correct: {cor}, Wrong: {wrg}, Empty: {emp}, Rate: {rate:.2f}"
                      for sid, lvl, vis, cor, wrg, emp, rate in zip(
                          df_with_predictions["state_id"],
                          df_with_predictions["level"],
                          df_with_predictions["visits"],
                          df_with_predictions["correct_predictions"],
                          df_with_predictions["wrong_predictions"],
                          df_with_predictions["empty_predictions"],
                          df_with_predictions["correctness_rate"],
                          strict=False)],
                hoverinfo="text", visible=(initial_plot_key == "state_id"))
            fig.add_trace(trace_state_id_scatter, row=1, col=1)
            all_traces_info.append(
                {
                    "trace": trace_state_id_scatter, "strategy": strategy_name, "plot_key": "per_state_id",
                    "is_legend_item": False
                }
            )

            if len(df_with_predictions) > 1:
                trend_state_id = add_trend_line(df_with_predictions, color, strategy_name, "state_id")
                trend_state_id.legendgroup = strategy_name
                trend_state_id.visible = (initial_plot_key == "state_id")
                fig.add_trace(trend_state_id, row=1, col=1)
                all_traces_info.append(
                    {
                        "trace": trend_state_id,
                        "strategy": strategy_name,
                        "plot_key": "state_id",
                        "is_legend_item": False
                    }
                )

    # Add Global Stats Traces
    for strategy_name, stats in global_stats_per_strategy.items():
        color = strategy_color_map.get(strategy_name, colors[-1])
        eval_time = stats.get("evaluation_time", 0)
        accuracy = stats.get("accuracy", 0)
        perplexity = stats.get("perplexity", 0)

        # Global Accuracy Plot
        trace_global_acc = go.Scatter(
            x=[eval_time],
            y=[accuracy],
            mode="markers",
            name=strategy_name, legendgroup=strategy_name,
            showlegend=True,
            marker={"size": 10, "color": color},
            text=[f"Strategy: {strategy_name}<br>Eval Time: {eval_time:.2f}s<br>Accuracy: {accuracy:.3f}"],
            hoverinfo="text", visible=(initial_plot_key == "global_accuracy"))
        fig.add_trace(trace_global_acc, row=1, col=1)
        all_traces_info.append(
            {
                "trace": trace_global_acc,
                "strategy": strategy_name,
                "plot_key": "global_accuracy",
                "is_legend_item": False
            }
        )

        # Global Perplexity Plot
        trace_global_perp = go.Scatter(
            x=[eval_time],
            y=[perplexity],
            mode="markers",
            name=strategy_name,
            legendgroup=strategy_name,
            showlegend=True,
            marker={"size": 10, "color": color},
            text=[f"Strategy: {strategy_name}<br>Eval Time: {eval_time:.2f}s<br>Perplexity: {perplexity:.3f}"],
            hoverinfo="text", visible=(initial_plot_key == "global_perplexity"))
        fig.add_trace(trace_global_perp, row=1, col=1)
        all_traces_info.append({
            "trace": trace_global_perp,
            "strategy": strategy_name,
            "plot_key": "global_perplexity",
            "is_legend_item": False,
        })


    # Add Per Index Stats Traces
    if per_index_df is not None:
        for strategy_name, group in per_index_df.groupby("Name"):
            color = strategy_color_map.get(str(strategy_name), colors[-1])

            x = group["batch_index"]

            for i, metric in enumerate(["accuracy", "pp_harmonic_mean"]):
                if metric in group.columns:
                    y = group[metric]
                    trace_acc_per_index = go.Scatter(
                            x=x,
                            y=y,
                            mode="lines+markers",
                            name=strategy_name if i == 0 else None,  # Show legend only in first subplot
                            legendgroup=strategy_name,
                            marker={"size": 6, "color": color},
                            text=[
                                f"Strategy: {strategy_name}<br>Batch: {b}<br>{metric}: {v:.3f}"
                                for b, v in zip(x, y, strict=False)
                            ],
                            hoverinfo="text"
                        )
                    fig.add_trace(
                        trace_acc_per_index
                    )
                    all_traces_info.append(
                        {
                            "trace": trace_acc_per_index,
                            "strategy": strategy_name,
                            "plot_key": f"per_index_{metric}",
                            "is_legend_item": False
                        }
                    )


    # Initial Layout
    initial_config = plot_configs[initial_plot_key]
    fig.update_layout(
        title=initial_config["title"],
        height=700, # Adjusted height for single plot
        legend_title="Strategies",
        hovermode="closest",
        xaxis_title=initial_config["xaxis_title"],
        yaxis_title=initial_config["yaxis_title"],
    )
    if "yaxis_range" in initial_config:
        fig.update_yaxes(range=initial_config["yaxis_range"], autorange=None)
    if "yaxis_autorange" in initial_config:
        fig.update_yaxes(autorange=initial_config["yaxis_autorange"], range=None)


    # Create buttons for plot type selection
    plot_type_buttons = []
    for key, config in plot_configs.items():
        visibility_list = [info["plot_key"] == key for info in all_traces_info]

        # Ensure legend items for the selected plot type are correctly shown/hidden
        # For the selected plot_key, the first trace of each strategy should show in legend.
        # Other traces (like trendlines or other plot_keys) should not.
        # This logic is complex with "update" method if not handled carefully.
        # The current trace addition already sets showlegend=True only for the first scatter of each per-state plot.
        # Global plots have showlegend=False.
        # We need to ensure that when a plot type is selected, its primary traces (those with showlegend=True)
        # are indeed shown in the legend, and others are not.
        # The "visible" array correctly hides/shows traces.
        # Plotly handles legend items based on trace visibility and their "showlegend" property.
        # However, we need to ensure that only one set of legend items (for one strategy) is active.
        # The `legendgroup` attribute handles grouping.
        # The first trace added for a `legendgroup` with `showlegend=True` defines the legend item.

        # Update legend visibility: only the main scatter for "visits" (or the first per-state plot type)
        # should manage legend items.
        # For other plot types, we might need to explicitly hide legend items from other types if they were primary.
        # This is tricky. A simpler way: ensure only one trace per strategy ever has showlegend=True.
        # The current code does this for "visits" scatter.
        # If we want other plot types to also show legend items when active, we"d need to update showlegend properties.
        # For now, let"s assume the legend is primarily driven by the "visits" plot"s main scatter traces.
        # Or, more robustly, ensure that for the *active* plot_key,
        # the designated legend-bearing traces have showlegend=True,
        # and for *inactive* plot_keys, their legend-bearing traces have showlegend=False.
        # This is too complex for a simple visibility update.
        # The current setup: "visits" scatter has showlegend=True. Others have showlegend=False.
        # This means legend items are always for "visits" data points, even if "level" plot is shown. This is not ideal.

        # Revised approach for legend handling with plot type switching:
        # Each plot_key"s primary scatter trace should have showlegend=True.
        # When switching plot_types, the visibility array handles showing/hiding traces.
        # Plotly"s default behavior with legendgroup should then correctly show legend for visible items.
        # Let"s re-verify trace setup:
        # trace_visits_scatter: showlegend=True
        # trace_level_scatter: showlegend=False (but should be True if it"s the active plot type"s main representation)
        # trace_state_id_scatter: showlegend=False (same as above)
        # trace_global_acc: showlegend=False
        # (global stats usually don"t need separate legend items if strategies are already listed)
        #
        # To fix legend: all primary scatters should have showlegend=True initially.
        # The visibility control will then determine which ones appear.
        # This is done by changing showlegend=False to showlegend=True for level and state_id scatters,
        # and ensuring global plots still have showlegend=False (or True if they are to be distinctly listed).
        # For simplicity, let"s keep showlegend=True only for the first trace of each strategy (visits scatter).
        # The legend will always refer to the strategy, and the plot title indicates what data is shown.

        args_list = [{"visible": visibility_list}]
        layout_update = {
            "title": config["title"],
            "xaxis.title.text": config["xaxis_title"],
            "yaxis.title.text": config["yaxis_title"],
        }
        if "yaxis_range" in config:
            layout_update["yaxis.range"] = config["yaxis_range"]
            layout_update["yaxis.autorange"] = None
        if "yaxis_autorange" in config:
            layout_update["yaxis.autorange"] = config["yaxis_autorange"]
            layout_update["yaxis.range"] = None

        args_list.append(layout_update)

        plot_type_buttons.append({
            "label": config["title"], # Use full title for clarity in dropdown
            "method": "update",
            "args": args_list
        })

    # Create buttons for strategy selection
    strategy_buttons = []
    # Button for showing all strategies
    strategy_buttons.append({
        "label": "All Strategies",
        "method": "update",
        "args": [
            {
                "visible": [
                    info["plot_key"] == initial_plot_key for info in all_traces_info
                ]
            },
            {},  # Default to initial plot key
        ]
    }) # This needs to be smarter, respecting current plot view

    # Revised strategy buttons:
    # The strategy buttons should only affect visibility of traces that match the *currently selected plot type*.
    # This is complex because the button args are static.
    # A simpler approach: strategy buttons toggle visibility for *all* traces of a strategy, across all plot types.
    # The plot type dropdown then filters which of these are actually displayed.

    # Current strategy button logic (from original code, adapted):
    # This makes all traces of a strategy visible/hidden.
    strategy_buttons = []
    all_traces_visible_true = [True] * len(all_traces_info)
    strategy_buttons.append(
        {
            "label": "All Strategies",
            "method": "update",
            "args": [{"visible": all_traces_visible_true}]
        }
    )

    unique_strategies = sorted(all_unique_strategy_names)
    for strategy in unique_strategies:
        visibility = [info["strategy"] == strategy for info in all_traces_info]
        strategy_buttons.append({"label": strategy, "method": "update", "args": [{"visible": visibility}]})

    # X-axis scale toggle configuration
    xaxis_log_config = {"xaxis.type": "log"}
    xaxis_linear_config = {"xaxis.type": "linear"}

    fig.update_layout(
        updatemenus=[
            # Plot Type selection dropdown
            {
                "buttons": plot_type_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.5, # Centered
                "y": 1.15,
                "xanchor": "center",
                "yanchor": "top",
                "name": "Select Plot Type"
            },
            # Strategy selection dropdown
            {
                "buttons": strategy_buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.0, # Right
                "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
            },
            # X-axis scale toggle
            {
                "type": "buttons",
                "direction": "right",
                "x": 0, # Left
                "y": 1.15,
                "buttons": [
                    {"label": "X Linear", "method": "relayout", "args": [xaxis_linear_config]},
                    {"label": "Toggle X Log", "method": "relayout", "args": [xaxis_log_config]},
                ],
                "showactive": True,
                "xanchor": "left",
                "yanchor": "top",
            },
        ]
    )

    fig.show()


if __name__ == "__main__":
    stats_file, stats_mode = get_latest_stats_file()
    print(f"Using latest {stats_mode} stats file: {stats_file}")

    if stats_mode == "batch":
        log_data = load_stats(stats_file)

        stats_to_plot = [
            "states_of_total_predictions",
            "states_of_empty_predictions",
            "states_of_correct_predictions",
            "states_of_wrong_predictions",
            "nb_correct_pred_per_state",
            "nb_wrong_pred_per_state",
        ]

        WINDOW_RANGE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15]

        NGRAM_NAMES = (
            [f"ngram_{i}" for i in WINDOW_RANGE]
            + [f"ngram_{i}_recovery" for i in WINDOW_RANGE]
            + [f"ngram_{i}_shorts" for i in WINDOW_RANGE]
        )

        visible_models = [ngram_name for ngram_name in NGRAM_NAMES if "shorts" not in ngram_name]
        visible_models += ["soft voting", "fallback ngram_8->bag", "bayesian train"]

        plot_stats(log_data) #, visible_models=visible_models)

    elif stats_mode == "streaming":
        try:
            df_streaming = pd.read_csv(stats_file)
            # Clean "inf" strings and convert to numeric
            df_streaming.replace("inf", float("inf"))
            for col in df_streaming.columns:
                if col not in ["batch_index", "Name"]:
                    df_streaming[col] = pd.to_numeric(df_streaming[col], errors="coerce")
        except FileNotFoundError:
            print(f"Error: The file {stats_file} was not found.")
            sys.exit()
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file: {e}")
            sys.exit()
        except ValueError as e:
            print(f"Value error while reading CSV file: {e}")
            sys.exit()

        if "Name" not in df_streaming.columns or "batch_index" not in df_streaming.columns:
            print("Error: Streaming CSV file must contain 'Name' and 'batch_index' columns.")
            sys.exit()

        log_data_streaming = []
        visible_models = df_streaming["Name"].unique().tolist()

        for model_name in visible_models:
            model_df = df_streaming[df_streaming["Name"] == model_name]
            if model_df.empty:
                continue

            # Get data from the latest batch_index for this model
            latest_entry = model_df.loc[model_df["batch_index"].idxmax()]

            model_stats = {
                "strategy": model_name,
                "strategy_accuracy": latest_entry.get("accuracy", 0),
                # Use latency_mean as evaluation_time, default to 0 if not present
                "evaluation_time": latest_entry.get("latency_mean", 0),
                "perplexity": latest_entry.get("pp_harmonic_mean"),
                "per_state_stats": {},  # No per-state stats in streaming CSVs
            }
            log_data_streaming.append(model_stats)

        if not log_data_streaming:
            print(f"No data processed from streaming file {stats_file} for the dashboard.")
        else:
            print(f"Generating dashboard for streaming data from {stats_file}...")
            plot_stats(log_data_streaming, per_index_df=df_streaming) #, visible_models=visible_models)

    else:
        print(f"Unknown stats mode: {stats_mode}. Expected 'batch' or 'streaming'.")
        msg = f"Unknown stats mode: {stats_mode}. Expected 'batch' or 'streaming'."
        raise ValueError(msg)
