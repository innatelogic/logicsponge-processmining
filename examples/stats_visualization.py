"""Module to visualize the correctness of states based on various factors.

This module reads state statistics data and visualizes correctness metrics
based on visits, level, and state ID.
"""

import json
import re
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


def get_latest_stats_file(folder: Path = Path("results")) -> Path:
    """Get the latest stats file from the specified folder.

    Args:
        folder (Path): The folder to search for stats files.

    Returns:
        Path: The latest stats file path.

    Raises:
        FileNotFoundError: If no stats files are found.

    """
    stats_files = list(folder.glob("stats_*.json"))

    if not stats_files:
        msg = f"No stats files found in {folder}"
        raise FileNotFoundError(msg)

    # Match and extract numeric ID
    def extract_id(file: Path) -> int:
        match = re.search(r"stats_(\d+)\.json", file.name)
        return int(match.group(1)) if match else -1

    return max(stats_files, key=extract_id)


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
    # Group by 'group_key' and calculate weighted mean correctness rate
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

    # fig.add_trace(means_line, row=1, col=1)
    # all_traces.append(means_line)
    # trace_info.append((strategy_name, "visits", 1))


def plot_stats(  # noqa: C901, PLR0915, PLR0912
    log_data: list[dict],
    # stats_to_plot: list[str] | None = None,
    *,
    visible_models: list[str] | None = None,
    add_by_state_id: bool = False,
) -> None:
    """Plot state correctness statistics based on log data.

    Args:
        log_data: list of dictionaries containing model stats as loaded from JSON.
        stats_to_plot: Optional list of specific stats to plot (not used in current implementation).
        visible_models: Optional list of model names to include in visualization.
        add_by_state_id: Whether to add a third row for state ID visualization.

    """
    # Filter models if visible_models is specified
    if visible_models:
        filtered_log_data = [model for model in log_data if model["strategy"] in visible_models]
    else:
        filtered_log_data = log_data

    # Extract per_state_stats for each strategy
    per_strategy_stats = extract_per_state_stats_from_log_data(filtered_log_data)

    # If there are no valid strategies with data, exit
    if not per_strategy_stats:
        print("No valid model data found for visualization.")
        return

    # Create a 3x1 subplot figure for our main views
    fig = make_subplots(
        rows=2 if not add_by_state_id else 3,
        cols=1,
        subplot_titles=(
            "Correctness Rate vs. Number of Visits",
            "Correctness Rate vs. Level",
            "Correctness Rate vs. State ID",
        ),
        vertical_spacing=0.15,
    )

    # Define colors for different strategies
    colors = [
        "rgb(31, 119, 180)",
        "rgb(255, 127, 14)",
        "rgb(44, 160, 44)",
        "rgb(214, 39, 40)",
        "rgb(148, 103, 189)",
        "rgb(140, 86, 75)",
        "rgb(227, 119, 194)",
        "rgb(127, 127, 127)",
        "rgb(188, 189, 34)",
        "rgb(23, 190, 207)",
    ]

    # Track all traces for visibility control
    all_traces = []
    trace_info = []  # [(model_name, x_factor, row), ...]

    # Create traces for each strategy and factor
    for i, (strategy_name, per_state_stats) in enumerate(per_strategy_stats.items()):
        # Skip if no state stats data
        if not per_state_stats:
            continue

        # Convert to DataFrame
        data_per_state = prepare_data_from_per_state_stats(per_state_stats)
        df_with_predictions = data_per_state[data_per_state["total_predictions"] > 0]

        # Skip if no valid data points
        if df_with_predictions.empty:
            continue

        # Calculate marker size based on total predictions (min 5, max 20)
        sizes = 5 + (df_with_predictions["total_predictions"] / df_with_predictions["total_predictions"].max() * 15)

        color = colors[i % len(colors)]

        # Factor 1: Visits (Row 1)
        trace1 = go.Scatter(
            x=df_with_predictions["visits"],
            y=df_with_predictions["correctness_rate"],
            mode="markers",
            name=strategy_name,
            marker={
                "size": sizes,
                "color": color,
                "opacity": 0.6,
                # "line": {"width": 1, "color": "black"},
            },
            text=[
                f"Strategy: {strategy_name}, State ID: {sid}, Level: {lvl}, Visits: {vis}<br>"
                f"Correct: {cor}, Wrong: {wrg}, Empty: {emp}, Rate: {rate:.2f}, Failure: {failure:.2f}"
                for sid, lvl, vis, cor, wrg, emp, rate, failure in zip(
                    df_with_predictions["state_id"],
                    df_with_predictions["level"],
                    df_with_predictions["visits"],
                    df_with_predictions["correct_predictions"],
                    df_with_predictions["wrong_predictions"],
                    df_with_predictions["empty_predictions"],
                    df_with_predictions["correctness_rate"],
                    df_with_predictions["failure_rate"],
                    strict=False,
                )
            ],
            hoverinfo="text",
            visible=True,  # Initially visible
        )

        fig.add_trace(trace1, row=1, col=1)
        all_traces.append(trace1)
        trace_info.append((strategy_name, "visits", 1))

        # trace1_bis = go.Scatter(
        #     x=df_with_predictions["visits"],
        #     y=df_with_predictions["failure_rate"],
        #     mode="markers",
        #     name=strategy_name,
        #     marker={
        #         "size": sizes,
        #         "symbol": "x",
        #         "color": color,
        #         "opacity": 0.6,
        #         # "line": {"width": 1, "color": "black"},
        #     },
        #     text=[
        #         f"Strategy: {strategy_name}, State ID: {sid}, Level: {lvl}, Visits: {vis}<br>"
        #         f"Correct: {cor}, Wrong: {wrg}, Empty: {emp}, Rate: {rate:.2f}, Failure: {failure:.2f}"
        #         for sid, lvl, vis, cor, wrg, emp, rate, failure in zip(
        #             df_with_predictions["state_id"],
        #             df_with_predictions["level"],
        #             df_with_predictions["visits"],
        #             df_with_predictions["correct_predictions"],
        #             df_with_predictions["wrong_predictions"],
        #             df_with_predictions["empty_predictions"],
        #             df_with_predictions["correctness_rate"],
        #             df_with_predictions["failure_rate"],
        #             strict=False,
        #         )
        #     ],
        #     hoverinfo="text",
        #     visible=True,  # Initially visible
        # )
        # fig.add_trace(trace1_bis, row=1, col=1)
        # all_traces.append(trace1_bis)
        # trace_info.append((strategy_name, "visits(X)", 1))

        # Add weighted trend line for visits
        if len(df_with_predictions) > 1:
            # Add a scatter plot of the means
            means_line = add_trend_line(df_with_predictions, color, strategy_name, "visits")

            fig.add_trace(means_line, row=1, col=1)
            all_traces.append(means_line)
            trace_info.append((strategy_name, "visits", 1))

        # Factor 2: Level (Row 2)
        trace2 = go.Scatter(
            x=df_with_predictions["level"],
            y=df_with_predictions["correctness_rate"],
            mode="markers",
            name=strategy_name,
            marker={
                "size": sizes,
                "color": color,
                "opacity": 0.6,
                # "line": {"width": 1, "color": "black"}
            },
            text=[
                f"Strategy: {strategy_name}, State ID: {sid}, Level: {lvl}, Visits: {vis}<br>"
                f"Correct: {cor}, Wrong: {wrg}, Empty: {emp}, Rate: {rate:.2f}"
                for sid, lvl, vis, cor, wrg, emp, rate in zip(
                    df_with_predictions["state_id"],
                    df_with_predictions["level"],
                    df_with_predictions["visits"],
                    df_with_predictions["correct_predictions"],
                    df_with_predictions["wrong_predictions"],
                    df_with_predictions["empty_predictions"],
                    df_with_predictions["correctness_rate"],
                    strict=False,
                )
            ],
            hoverinfo="text",
            visible=True,  # Initially visible
        )
        fig.add_trace(trace2, row=2, col=1)
        all_traces.append(trace2)
        trace_info.append((strategy_name, "level", 2))

        # After trace2 addition (row 2 - level)
        # Add weighted trend line for level
        if len(df_with_predictions) > 1:
            # Add a scatter plot of the means
            means_line = add_trend_line(df_with_predictions, color, strategy_name, "level")

            fig.add_trace(means_line, row=2, col=1)
            all_traces.append(means_line)
            trace_info.append((strategy_name, "level", 2))

        # Factor 3: State ID (Row 3)
        if add_by_state_id:
            trace3 = go.Scatter(
                x=df_with_predictions["state_id"],
                y=df_with_predictions["correctness_rate"],
                mode="markers",
                name=strategy_name,
                marker={
                    "size": sizes,
                    "color": color,
                    "opacity": 0.6,
                    # "line": {"width": 1, "color": "black"}
                },
                text=[
                    f"Strategy: {strategy_name}, State ID: {sid}, Level: {lvl}, Visits: {vis}<br>"
                    f"Correct: {cor}, Wrong: {wrg}, Empty: {emp}, Rate: {rate:.2f}"
                    for sid, lvl, vis, cor, wrg, emp, rate in zip(
                        df_with_predictions["state_id"],
                        df_with_predictions["level"],
                        df_with_predictions["visits"],
                        df_with_predictions["correct_predictions"],
                        df_with_predictions["wrong_predictions"],
                        df_with_predictions["empty_predictions"],
                        df_with_predictions["correctness_rate"],
                        strict=False,
                    )
                ],
                hoverinfo="text",
                visible=True,  # Initially visible
            )
            fig.add_trace(trace3, row=3, col=1)
            all_traces.append(trace3)
            trace_info.append((strategy_name, "state_id", 3))

            # After trace3 addition (row 3 - state_id)
            # Add weighted trend line for state_id
            if len(df_with_predictions) > 1:
                # Add a scatter plot of the means
                means_line = add_trend_line(df_with_predictions, color, strategy_name, "state_id")

                fig.add_trace(means_line, row=3, col=1)
                all_traces.append(means_line)
                trace_info.append((strategy_name, "state_id", 3))

    # Update layout
    fig.update_layout(
        title="State Correctness Analysis by Visits, Level, and State ID",
        height=900,
        legend_title="Strategies",
        hovermode="closest",
    )

    # X-axis titles
    fig.update_xaxes(title_text="Number of Visits", row=1, col=1)
    fig.update_xaxes(title_text="Level", row=2, col=1)
    if add_by_state_id:
        # Only update x-axis title for state_id if it's added
        fig.update_xaxes(title_text="State ID", row=3, col=1)

    # Y-axis titles (same for all)
    for i in range(1, 4):
        fig.update_yaxes(title_text="Correctness Rate", range=[0, 1], row=i, col=1)

    # Create buttons for strategy selection
    buttons = []

    # Button for showing all strategies
    all_visible = [True] * len(all_traces)
    buttons.append({"label": "All Strategies", "method": "update", "args": [{"visible": all_visible}, {}]})

    # Button for each strategy
    unique_strategies = sorted({info[0] for info in trace_info})
    for strategy in unique_strategies:
        visibility = [info[0] == strategy for info in trace_info]
        buttons.append({"label": strategy, "method": "update", "args": [{"visible": visibility}, {}]})

    # Add toggle buttons for log scales and strategy selection
    fig.update_layout(
        updatemenus=[
            # Strategy selection dropdown
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.0,
                "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
            },
            # X-axis scale toggle
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.15,
                "buttons": [
                    {
                        "label": "Toggle X Log",
                        "method": "relayout",
                        "args": [{"xaxis.type": "log", "xaxis2.type": "log", "xaxis3.type": "log"}],
                    },
                    {
                        "label": "X Linear",
                        "method": "relayout",
                        "args": [{"xaxis.type": "linear", "xaxis2.type": "linear", "xaxis3.type": "linear"}],
                    },
                ],
                "showactive": True,
            },
        ]
    )

    fig.show()


if __name__ == "__main__":
    stats_file = get_latest_stats_file()
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

    plot_stats(log_data, visible_models=visible_models)
