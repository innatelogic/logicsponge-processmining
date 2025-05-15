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
        # Batch JSON provides 'strategy_accuracy' as 0-100.
        # Streaming CSV (transformed) might provide 'accuracy' as 0-1.
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
            "perplexity": model_data.get("strategy_perplexity", model_data.get("perplexity", 0.0)), # Also check direct perplexity
            "evaluation_time": model_data.get("strategy_eval_time", model_data.get("evaluation_time", 0.0)), # Also check direct eval_time
        }

    return global_stats


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
    """Plot state correctness statistics and global model performance based on log data.

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

    # Extract global stats for each strategy
    global_stats_per_strategy = extract_global_stats_from_log_data(filtered_log_data)

    # Extract per_state_stats for each strategy
    per_strategy_stats = extract_per_state_stats_from_log_data(filtered_log_data)

    # If there are no valid strategies with data, exit
    if not per_strategy_stats and not global_stats_per_strategy:
        print("No valid model data found for visualization.")
        return

    # Determine number of rows for per-state plots
    num_per_state_rows = 2
    if add_by_state_id:
        num_per_state_rows = 3

    # Total rows including global stats plots (2 new rows for global stats)
    total_rows = num_per_state_rows + 2

    # Dynamically generate subplot titles
    subplot_titles_list = [
        "Correctness Rate vs. Number of Visits",
        "Correctness Rate vs. Level",
    ]
    if add_by_state_id:
        subplot_titles_list.append("Correctness Rate vs. State ID")

    subplot_titles_list.extend([
        "Global Accuracy vs. Evaluation Time",
        "Global Perplexity vs. Evaluation Time",
    ])
    final_subplot_titles = tuple(subplot_titles_list)

    # Create a subplot figure
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        subplot_titles=final_subplot_titles,
        vertical_spacing=0.1,  # Adjusted for potentially more rows
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

    # Create a strategy-to-color map for consistent coloring
    all_unique_strategy_names = sorted(
        set(per_strategy_stats.keys()) | set(global_stats_per_strategy.keys())
    )
    strategy_color_map = {
        name: colors[j % len(colors)] for j, name in enumerate(all_unique_strategy_names)
    }

    # Create traces for each strategy and factor (per-state stats)
    for strategy_name, current_per_state_stats in per_strategy_stats.items():
        # Skip if no state stats data
        if not current_per_state_stats:
            continue

        # Convert to DataFrame
        data_per_state = prepare_data_from_per_state_stats(current_per_state_stats)
        df_with_predictions = data_per_state[data_per_state["total_predictions"] > 0]

        # Skip if no valid data points
        if df_with_predictions.empty:
            continue

        # Calculate marker size based on total predictions (min 5, max 20)
        sizes = 5 + (df_with_predictions["total_predictions"] / df_with_predictions["total_predictions"].max() * 15)

        color = strategy_color_map.get(strategy_name, colors[len(colors) -1]) # Fallback color

        # Factor 1: Visits (Row 1)
        trace1 = go.Scatter(
            x=df_with_predictions["visits"],
            y=df_with_predictions["correctness_rate"],
            mode="markers",
            name=strategy_name,
            legendgroup=strategy_name,
            showlegend=True, # Show this primary trace in legend
            marker={
                "size": sizes,
                "color": color,
                "opacity": 0.6,
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
            means_line_visits = add_trend_line(df_with_predictions, color, strategy_name, "visits")
            means_line_visits.legendgroup = strategy_name # Ensure trend line is part of the group
            # means_line_visits.showlegend = False # Already set in add_trend_line
            fig.add_trace(means_line_visits, row=1, col=1)
            all_traces.append(means_line_visits)
            trace_info.append((strategy_name, "visits_trend", 1)) # Differentiate trend in trace_info if needed

        # Factor 2: Level (Row 2)
        trace2 = go.Scatter(
            x=df_with_predictions["level"],
            y=df_with_predictions["correctness_rate"],
            mode="markers",
            name=strategy_name, # Name for hover, legend handled by group
            legendgroup=strategy_name,
            showlegend=False, # Don't repeat in legend
            marker={
                "size": sizes,
                "color": color,
                "opacity": 0.6,
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

        # Add weighted trend line for level
        if len(df_with_predictions) > 1:
            means_line_level = add_trend_line(df_with_predictions, color, strategy_name, "level")
            means_line_level.legendgroup = strategy_name
            # means_line_level.showlegend = False
            fig.add_trace(means_line_level, row=2, col=1)
            all_traces.append(means_line_level)
            trace_info.append((strategy_name, "level_trend", 2))

        # Factor 3: State ID (Row 3, if applicable)
        if add_by_state_id:
            trace3 = go.Scatter(
                x=df_with_predictions["state_id"],
                y=df_with_predictions["correctness_rate"],
                mode="markers",
                name=strategy_name, # Name for hover
                legendgroup=strategy_name,
                showlegend=False, # Don't repeat in legend
                marker={
                    "size": sizes,
                    "color": color,
                    "opacity": 0.6,
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

            # Add weighted trend line for state_id
            if len(df_with_predictions) > 1:
                means_line_stateid = add_trend_line(df_with_predictions, color, strategy_name, "state_id")
                means_line_stateid.legendgroup = strategy_name
                # means_line_stateid.showlegend = False
                fig.add_trace(means_line_stateid, row=3, col=1)
                all_traces.append(means_line_stateid)
                trace_info.append((strategy_name, "state_id_trend", 3))

    # Add Global Stats Plots
    global_stats_accuracy_row = num_per_state_rows + 1
    global_stats_perplexity_row = num_per_state_rows + 2

    for strategy_name, stats in global_stats_per_strategy.items():
        color = strategy_color_map.get(strategy_name, colors[len(colors) -1]) # Fallback color
        eval_time = stats.get("evaluation_time", 0)
        accuracy = stats.get("accuracy", 0)
        perplexity = stats.get("perplexity", 0)

        # print(f"Strategy: {strategy_name}, Accuracy: {accuracy:.3f}, Perplexity: {perplexity:.3f}") # Removed debug print

        # Plot for Accuracy vs Eval Time
        trace_acc_global = go.Scatter(
            x=[eval_time],
            y=[accuracy],
            mode="markers",
            name=strategy_name, # Name for hover
            legendgroup=strategy_name,
            showlegend=False, # Already in legend from per-state plots
            marker={"size": 10, "color": color},
            text=[f"Strategy: {strategy_name}<br>Eval Time: {eval_time:.2f}s<br>Accuracy: {accuracy:.3f}"],
            hoverinfo="text",
            visible=True,
        )
        fig.add_trace(trace_acc_global, row=global_stats_accuracy_row, col=1)
        all_traces.append(trace_acc_global)
        trace_info.append((strategy_name, "global_accuracy", global_stats_accuracy_row))

        # Plot for Perplexity vs Eval Time
        trace_perp_global = go.Scatter(
            x=[eval_time],
            y=[perplexity],
            mode="markers",
            name=strategy_name, # Name for hover
            legendgroup=strategy_name,
            showlegend=False, # Already in legend
            marker={"size": 10, "color": color},
            text=[f"Strategy: {strategy_name}<br>Eval Time: {eval_time:.2f}s<br>Perplexity: {perplexity:.3f}"],
            hoverinfo="text",
            visible=True,
        )
        fig.add_trace(trace_perp_global, row=global_stats_perplexity_row, col=1)
        all_traces.append(trace_perp_global)
        trace_info.append((strategy_name, "global_perplexity", global_stats_perplexity_row))

    # Update layout
    fig.update_layout(
        title="State Correctness Analysis & Global Model Performance",
        height=max(900, 220 * total_rows), # Dynamic height
        legend_title="Strategies",
        hovermode="closest",
    )

    # X-axis titles
    fig.update_xaxes(title_text="Number of Visits", row=1, col=1)
    fig.update_xaxes(title_text="Level", row=2, col=1)
    if add_by_state_id:
        fig.update_xaxes(title_text="State ID", row=num_per_state_rows, col=1)
    fig.update_xaxes(title_text="Evaluation Time (s)", row=global_stats_accuracy_row, col=1)
    fig.update_xaxes(title_text="Evaluation Time (s)", row=global_stats_perplexity_row, col=1)


    # Y-axis titles
    for r in range(1, num_per_state_rows + 1):
        fig.update_yaxes(title_text="Correctness Rate", range=[0, 1], row=r, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=global_stats_accuracy_row, col=1)
    fig.update_yaxes(title_text="Perplexity", row=global_stats_perplexity_row, col=1) # Auto-range for perplexity

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
    # Dynamically create arguments for X-axis scale toggle
    xaxis_log_config = {}
    xaxis_linear_config = {}
    xaxis_log_config["xaxis.type"] = "log"
    xaxis_linear_config["xaxis.type"] = "linear"
    for r_idx in range(2, total_rows + 1):
        xaxis_log_config[f"xaxis{r_idx}.type"] = "log"
        xaxis_linear_config[f"xaxis{r_idx}.type"] = "linear"

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
                "y": 1.15 if total_rows <= 3 else 1.08, # Adjust y to prevent overlap if many rows
                "buttons": [
                    {
                        "label": "Toggle X Log",
                        "method": "relayout",
                        "args": [xaxis_log_config],
                    },
                    {
                        "label": "X Linear",
                        "method": "relayout",
                        "args": [xaxis_linear_config],
                    },
                ],
                "showactive": True,
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

        plot_stats(log_data, visible_models=visible_models)

    elif stats_mode == "streaming":
        try:
            df_streaming = pd.read_csv(stats_file)
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
                "accuracy": latest_entry.get("accuracy", 0),
                # Use latency_mean as evaluation_time, default to 0 if not present
                "evaluation_time": latest_entry.get("latency_mean", 0),
                "perplexity": 0,  # Perplexity is not typically in streaming CSVs
                "per_state_stats": {},  # No per-state stats in streaming CSVs
            }
            log_data_streaming.append(model_stats)

        if not log_data_streaming:
            print(f"No data processed from streaming file {stats_file} for the dashboard.")
        else:
            print(f"Generating dashboard for streaming data from {stats_file}...")
            plot_stats(log_data_streaming, visible_models=visible_models)

    else:
        print(f"Unknown stats mode: {stats_mode}. Expected 'batch' or 'streaming'.")
        msg = f"Unknown stats mode: {stats_mode}. Expected 'batch' or 'streaming'."
        raise ValueError(msg)
