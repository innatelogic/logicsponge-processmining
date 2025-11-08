"""Utility functions for process mining."""

import copy
import json
import logging
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap

from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.types import Config, Event, Metrics, Prediction, ProbDistr

RED_TO_GREEN_CMAP = LinearSegmentedColormap.from_list("rg",["r", "w", "g"], N=256)


def save_run_config(config: dict, dest: Path) -> bool:
    """
    Save a run configuration (JSON) to the specified path.

    Returns True if write succeeded, False otherwise (and logs debug info).
    """
    logger = logging.getLogger(__name__)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w") as _f:
            json.dump(config, _f, indent=2)
    except OSError as e:
        logger.debug("Could not write run config to %s: %s", dest, e)
        return False
    else:
        return True


def add_file_log_handler(log_file_path: Path, fmt: str = "%(message)s") -> logging.Handler | None:
    """
    Add a FileHandler to the root logger writing to `log_file_path`.

    Returns the handler on success, or None on failure. Does not remove existing
    handlers; caller can manage handlers if needed.
    """
    logger = logging.getLogger(__name__)
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter(fmt)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
    except OSError as e:
        logger.debug("Could not create log file %s; continuing with console logging. %s", log_file_path, e)
        return None
    else:
        return file_handler

def extract_event_fields(event: Event) -> Event:
    """Extract the required fields from an event."""
    return event


# ============================================================
# Probabilities
# ============================================================

stop_symbol = DEFAULT_CONFIG["stop_symbol"]


def probs_prediction(probs: ProbDistr, config: Config) -> Prediction | None: # noqa: C901
    """
    Return the top-k activities based on their probabilities.

    If stop_symbol has a probability of 1.0 and there are no other activities, return None.
    If stop_symbol has a probability of 1.0 and there are other activities, give a uniform distribution to these
    other activities.
    If stop_symbol is present but with a probability less than 1.0 and include_stop is False, remove it and
    normalize the rest.
    """
    # If there are no probabilities, return None
    if not probs:
        return None


    def compute_highest_probability(probs_input: dict) -> Prediction:
        """Get the highest probability of a given activity."""
        # Convert dictionary to a sorted list of items (activities and probabilities) for consistency
        sorted_probs = sorted(probs_input.items(), key=lambda x: (-x[1], x[0]))

        # Extract activities and probabilities in a consistent way
        activities, probabilities = zip(*sorted_probs, strict=True)

        # Convert the probabilities to a numpy array
        probabilities_array = np.array(probabilities)

        # Get the indices of the top-k elements, sorted in descending order
        top_k_indices = np.argsort(probabilities_array)[-config["top_k"] :][::-1]

        # Use the indices to get the top-k activities
        top_k_activities = [activities[i] for i in top_k_indices if probs_input[activities[i]] > 0]

        # Determine the predicted activity
        if config["randomized"]:
            # Randomly choose an activity based on the given probability distribution
            next_activity_idx = np.random.choice(  # noqa: NPY002
                len(probabilities_array), p=probabilities_array / probabilities_array.sum()
            )
            predicted_activity = activities[next_activity_idx]
        else:
            # Get the most probable activity deterministically
            predicted_activity = top_k_activities[0]

        # Get the highest probability corresponding to the predicted activity
        highest_probability = float(probabilities_array[activities.index(predicted_activity)])

        # Return the predicted activity, top-k activities, and the probability of the predicted activity
        return {
            "activity": predicted_activity,
            "top_k_activities": top_k_activities,
            "probability": highest_probability,
            "probs": probs_input,
        }

    # Handle the case where include_stop is False
    if not config["include_stop"] and stop_symbol in probs:  # stop_symbol will always be a key
        # Create a copy of probs to avoid modifying the original dictionary
        probs_copy = probs.copy()

    if not config["include_stop"] and stop_symbol in probs:  # stop_symbol will always be a key
        # Create a copy of probs to avoid modifying the original dictionary
        probs_copy = probs.copy()

        stop_probability = probs_copy.get(stop_symbol, 0.0)

        # If stop_symbol has a probability of 1 and there are no other activities available, return None
        if stop_probability >= 1.0 and len(probs_copy) == 1:
            return None

        # If stop has probability 1 but there are other activities, give a uniform distribution to the other activities
        if stop_probability >= 1.0 and len(probs_copy) > 1:
            del probs_copy[stop_symbol]  # Remove stop_symbol from consideration

            # Verify stop_symbol is indeed deleted
            if stop_symbol in probs_copy:
                msg = "stop_symbol was not successfully removed from probabilities."
                raise ValueError(msg)

            # Distribute the remaining probability uniformly among other activities
            num_activities = len(probs_copy)
            uniform_prob = 1.0 / num_activities
            probs_copy = dict.fromkeys(probs_copy, uniform_prob)

        # If stop_symbol has less than 1.0 probability, remove it and normalize the rest
        elif stop_probability < 1.0:
            del probs_copy[stop_symbol]

            # Verify stop_symbol is indeed deleted
            if stop_symbol in probs_copy:
                msg = "stop_symbol was not successfully removed from probabilities."
                raise ValueError(msg)

            # Normalize the remaining probabilities so that they sum to 1
            total_prob = sum(probs_copy.values())
            if total_prob > 0:
                probs_copy = {activity: prob / total_prob for activity, prob in probs_copy.items()}

        # If there are no probabilities after filtering, return None
        if probs_copy == {}:
            return None
        # If there are no probabilities after filtering, return None
        if probs_copy == {}:
            return None

        return compute_highest_probability(probs_copy)

    return compute_highest_probability(probs)

def metrics_prediction(metrics: Metrics, config: Config) -> Prediction | None:
    """Return prediction including time delays."""
    probs = metrics["probs"]
    delays = metrics["predicted_delays"]

    # If there are no probabilities, return None
    if not probs:
        return None

    # Generate the probability-based prediction
    prediction = probs_prediction(probs, config=config)
    if prediction:
        prediction["predicted_delays"] = copy.deepcopy(delays)

    return prediction


def compute_perplexity_stats(perplexities: list[float]) -> dict[str, float]:
    """Compute and return perplexity statistics."""
    res = {}

    arithmetic_mean_perplexity = sum(perplexities) / len(perplexities) if perplexities else None

    inverted_sum = sum(((1.0 / p) if p > 0 else float("inf")) for p in perplexities) if perplexities else float("inf")
    harmonic__mean_perplexity = len(perplexities) / inverted_sum if inverted_sum > 0 else float("inf")

    sorted_perplexities = sorted(perplexities)
    q1_perplexity = sorted_perplexities[int(0.25 * len(sorted_perplexities))] if sorted_perplexities else None
    q3_perplexity = sorted_perplexities[int(0.75 * len(sorted_perplexities))] if sorted_perplexities else None
    median_perplexity = sorted_perplexities[int(0.5 * len(sorted_perplexities))] if sorted_perplexities else None

    res["pp_arithmetic_mean"] = arithmetic_mean_perplexity
    res["pp_harmonic_mean"] = harmonic__mean_perplexity
    res["pp_median"] = median_perplexity
    res["pp_q1"] = q1_perplexity
    res["pp_q3"] = q3_perplexity

    return res


def compute_seq_perplexity(normalized_likelihood: float, *, log_likelihood: bool) -> float:
    """Compute the perplexity of a sequence."""
    if normalized_likelihood is not None and normalized_likelihood > 0:
        return math.exp(-normalized_likelihood) if log_likelihood else 1.0 / normalized_likelihood
    return float("inf")


def compare_models_comparison( # noqa: C901, PLR0915, PLR0912
    prediction_vectors_memory: dict,
    tested_model: str,
    reference_model: str,
    baseline_model: str = "actual",
    *,
    include_empty: bool = False,
) -> dict:
    """
    For each iteration stored in prediction_vectors_memory, compute correlation, anticorrelation and similarity.

    Correlation:
        (# positions where reference == baseline AND tested == baseline)
        / (# positions where reference == baseline)
    Anticorrelation:
        (# positions where reference != baseline AND tested == baseline)
        / (# positions where reference != baseline)
    Similarity:
        (# positions where tested == reference)
        / (# total positions)

    Returns a dict with:
      - "per_iteration": list of ratios (float) or None when denominator is 0 for that iteration
      - "correlation": aggregated ratio across iterations (float) or None if no reference-correct positions
      - "anticorrelation": aggregated ratio across iterations (float) or None if no reference-incorrect positions
      - "similarity": aggregated ratio across iterations (float) or None if no total positions
      - "counts": dict with totals {'total_ref_correct', 'total_both_correct', 'iterations_used'}
      - "notes": optional notes about length mismatches
    """
    # Validate presence
    if tested_model not in prediction_vectors_memory:
        msg = f"tested_model '{tested_model}' not found in prediction_vectors_memory"
        raise KeyError(msg)
    if reference_model not in prediction_vectors_memory:
        msg = f"reference_model '{reference_model}' not found in prediction_vectors_memory"
        raise KeyError(msg)
    if baseline_model not in prediction_vectors_memory:
        msg = f"baseline_model '{baseline_model}' not found in prediction_vectors_memory"
        raise KeyError(msg)

    tested_list = prediction_vectors_memory[tested_model]
    reference_list = prediction_vectors_memory[reference_model]
    baseline_list = prediction_vectors_memory[baseline_model]

    iterations = min(len(tested_list), len(reference_list), len(baseline_list))
    per_iteration = []
    notes: list[str] = []

    # Aggregated counters
    total_ref_correct = 0  # reference == baseline
    total_both_correct = 0  # reference == baseline AND tested == baseline
    total_ref_incorrect = 0  # reference != baseline
    total_tested_correct_when_ref_incorrect = 0  # reference != baseline AND tested == baseline
    total_positions = 0
    total_similarity_matches = 0  # tested == reference

    for it in range(iterations):
        tested_vec = tested_list[it]
        ref_vec = reference_list[it]
        base_vec = baseline_list[it]

        # align lengths conservatively
        n = min(len(tested_vec), len(ref_vec), len(base_vec))
        if len(tested_vec) != len(ref_vec) or len(ref_vec) != len(base_vec):
            notes.append(f"iter_{it}: length_mismatch tested={len(tested_vec)} ref={len(ref_vec)} base={len(base_vec)}")

        # Optionally remove positions where tested or reference produced an empty prediction
        if not include_empty:
            empty_symbol = DEFAULT_CONFIG.get("empty_symbol")
            if empty_symbol is not None:
                # Build filtered vectors keeping only indices where neither tested nor reference is empty
                indices = [i for i in range(n) if tested_vec[i] != empty_symbol and ref_vec[i] != empty_symbol]
                if len(indices) != n:
                    # Note the filtering
                    notes.append(f"iter_{it}: filtered_out_empty_positions removed={n - len(indices)}")
                tested_vec = [tested_vec[i] for i in indices]
                ref_vec = [ref_vec[i] for i in indices]
                base_vec = [base_vec[i] for i in indices]
                n = len(tested_vec)

        ref_correct = 0
        both_correct = 0
        ref_incorrect = 0
        tested_correct_when_ref_incorrect = 0
        similarity_matches = 0

        for i in range(n):
            ref_equal_base = (ref_vec[i] == base_vec[i])
            tested_equal_base = (tested_vec[i] == base_vec[i])
            tested_equal_ref = (tested_vec[i] == ref_vec[i])

            if ref_equal_base:
                ref_correct += 1
                if tested_equal_base:
                    both_correct += 1
            else:
                ref_incorrect += 1
                if tested_equal_base:
                    tested_correct_when_ref_incorrect += 1

            if tested_equal_ref:
                similarity_matches += 1

        # Compute per-iteration ratios (None if denominator zero)
        corr = (both_correct / ref_correct) if ref_correct > 0 else None
        anticorr = (tested_correct_when_ref_incorrect / ref_incorrect) if ref_incorrect > 0 else None
        similarity = (similarity_matches / n) if n > 0 else None

        per_iteration.append(
            {
                "iteration": it,
                "positions": n,
                "correlation": corr,
                "anticorrelation": anticorr,
                "similarity": similarity,
                "counts": {
                    "ref_correct": ref_correct,
                    "both_correct": both_correct,
                    "ref_incorrect": ref_incorrect,
                    "tested_correct_when_ref_incorrect": tested_correct_when_ref_incorrect,
                    "similarity_matches": similarity_matches,
                },
            }
        )

        # Update aggregates
        total_ref_correct += ref_correct
        total_both_correct += both_correct
        total_ref_incorrect += ref_incorrect
        total_tested_correct_when_ref_incorrect += tested_correct_when_ref_incorrect
        total_positions += n
        total_similarity_matches += similarity_matches

    # Compute overall aggregated ratios (None when denominator 0)
    overall_correlation = (total_both_correct / total_ref_correct) if total_ref_correct > 0 else None
    overall_anticorrelation = (
        total_tested_correct_when_ref_incorrect / total_ref_incorrect if total_ref_incorrect > 0 else None
    )
    overall_similarity = (total_similarity_matches / total_positions) if total_positions > 0 else None

    # Return aggregated metrics as top-level keys matching the docstring:
    return {
        "per_iteration": per_iteration,
        "correlation": overall_correlation,
        "anticorrelation": overall_anticorrelation,
        "similarity": overall_similarity,
        "counts": {
            "total_ref_correct": total_ref_correct,
            "total_both_correct": total_both_correct,
            "total_ref_incorrect": total_ref_incorrect,
            "total_tested_correct_when_ref_incorrect": total_tested_correct_when_ref_incorrect,
            "total_positions": total_positions,
            "total_similarity_matches": total_similarity_matches,
            "iterations_used": iterations,
        },
        "notes": notes,
    }


class HeatmapOptions(NamedTuple):
    """Options for show_comparison_heatmap to group optional visualization parameters."""

    # `cmap` may be either a matplotlib colormap name (str) or a Colormap instance
    # (e.g. returned by LinearSegmentedColormap.from_list). Passing the actual
    # Colormap object avoids lookup errors when a custom colormap name is not
    # registered in matplotlib's global colormap registry.
    # Use matplotlib's 'RdYlGn' divergent colormap as the default (red->yellow->green).
    cmap: str | Colormap = "RdYlGn"
    annotate: bool = True
    fmt: str = ".1f"


def show_comparison_heatmap(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] | None = None,
    options: HeatmapOptions | None = None,
) -> None:
    """
    Load a comparison CSV (as saved by this script) and show/save a seaborn heatmap.

    Parameters
    ----------
    csv_path:
        Path to the CSV file produced by this script.
    output_path:
        If provided, the heatmap image will be saved to this path.
    figsize:
        Optional tuple (w, h) in inches. If None, size is inferred from shape.
    options:
        HeatmapOptions namedtuple grouping optional visualization parameters:
            - cmap: matplotlib colormap name (default 'viridis').
            - annotate: whether to annotate cells with values (default True).
            - fmt: annotation format string (default '.1f').

    """
    opts = options or HeatmapOptions()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        msg = f"CSV file not found: {csv_path}"
        raise FileNotFoundError(msg)

    comparision_df = pd.read_csv(csv_path, index_col=0)
    # Convert to numeric and keep NaNs
    comparision_df = comparision_df.apply(pd.to_numeric, errors="coerce")

    nrows, ncols = comparision_df.shape
    if figsize is None:
        figsize = (max(8, ncols * 0.5), max(6, nrows * 0.5))  # noqa: PGH003 # type: ignore

    plt.figure(figsize=figsize)
    sns.heatmap(comparision_df, annot=opts.annotate, fmt=opts.fmt, cmap=opts.cmap, cbar_kws={"label": "Percent"})
    plt.xlabel("reference model")
    plt.ylabel("tested model")
    plt.title("Comparison heatmap (tested vs reference)")

    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outp, dpi=150)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def save_all_comparison_heatmaps(
    results_dir: str | Path,
    run_id: str,
    *,
    snapshot_idx: int | None = None,
    cmap: LinearSegmentedColormap = RED_TO_GREEN_CMAP,
    annotate: bool = False,
) -> dict[str, Path]:
    """
    Render and save correlation/anticorrelation/similarity heatmaps from CSV matrices.

    Looks for CSVs produced by the comparison step:
      - {results_dir}/{run_id}_correlation_matrix.csv
      - {results_dir}/{run_id}_anticorrelation_matrix.csv
      - {results_dir}/{run_id}_similarity_matrix.csv

    Saves PNGs alongside them. If ``snapshot_idx`` is provided, it will be appended
    to the filenames for periodic snapshots during streaming.

    Returns a mapping with keys: 'correlation', 'anticorrelation', 'similarity' pointing to the PNG paths.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{snapshot_idx}" if snapshot_idx is not None else ""

    correlation_csv = results_dir / f"{run_id}_correlation_matrix.csv"
    anticorrelation_csv = results_dir / f"{run_id}_anticorrelation_matrix.csv"
    similarity_csv = results_dir / f"{run_id}_similarity_matrix.csv"

    correlation_png = results_dir / f"{run_id}_correlation_matrix{suffix}.png"
    anticorrelation_png = results_dir / f"{run_id}_anticorrelation_matrix{suffix}.png"
    similarity_png = results_dir / f"{run_id}_similarity_matrix{suffix}.png"

    # Pass the colormap directly (may be a Colormap instance or a name)
    opts = HeatmapOptions(cmap=cmap, annotate=annotate, fmt=".1f")

    # Render each if CSV exists
    if correlation_csv.exists():
        show_comparison_heatmap(correlation_csv, output_path=correlation_png, options=opts)
    if anticorrelation_csv.exists():
        show_comparison_heatmap(anticorrelation_csv, output_path=anticorrelation_png, options=opts)
    if similarity_csv.exists():
        show_comparison_heatmap(similarity_csv, output_path=similarity_png, options=opts)

    return {
        "correlation": correlation_png,
        "anticorrelation": anticorrelation_png,
        "similarity": similarity_png,
    }


def discover_window_columns(df: pd.DataFrame, prefix: str) -> list[tuple[int, str]]:
    """
    Discover columns in a DataFrame whose names start with `prefix`.

    Parse the integer window sizes from the suffix.

    Returns a sorted list of (window_size, column_name). Any column whose suffix
    cannot be parsed to int will be skipped and an exception logged.
    """
    logger = logging.getLogger(__name__)
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    windows: list[tuple[int, str]] = []
    for c in cols:
        try:
            windows.append((int(c.split(prefix)[-1]), c))
        except Exception:
            logger.exception("Could not parse window size from column: %s", c)

    windows.sort()
    return windows


def plot_ngrams_vs_prefix(  # noqa: C901, D417, PLR0913, PLR0915
    df: pd.DataFrame,
    ngram_names: list[str],
    prefix: str,
    *,
    xlabel: str,
    title: str,
    out_png: str | Path,
    grid_alpha: float = 0.25,
    baseline_tested: str | None = None,
    baseline_label: str | None = None,
) -> bool:
    """
    Plot "NGrams vs {prefix} windows".

    Parameters
    ----------
    - df: DataFrame indexed by ngram name, columns are model names including columns
      with names like '<prefix><window>' where <window> is an integer.
    - ngram_names: list of ngram labels (rows to plot)
    - prefix: column name prefix to discover window columns (e.g. 'qlearning_win')
    - xlabel: x-axis label (e.g. 'Q-learning window size')
    - title: short title used in the figure title
    - out_png: path to save the resulting PNG
    - grid_alpha: alpha transparency for the grid lines (default 0.25)

    Returns True if a file was created, False if nothing was plotted (no columns / data).

    """
    logger = logging.getLogger(__name__)
    out_path = Path(out_png)

    windows = discover_window_columns(df, prefix)
    if not windows:
        logger.debug("No %s columns found for ngram vs %s plots; skipping.", prefix, title)
        return False

    windows_sorted, cols_sorted = zip(*windows, strict=True)

    cmap = plt.get_cmap("tab10")
    linestyles = ["-", "--", "-.", ":"]

    plt.figure(figsize=(8, 6))
    plotted_any = False
    for i, ngram in enumerate(ngram_names):
        if ngram not in df.index:
            continue
        y: list[float] = []
        for col in cols_sorted:
            try:
                val = df.loc[ngram, col] if col in df.columns else np.nan
                y.append(float(val) if not pd.isna(val) else np.nan)  # type: ignore # noqa: PGH003
            except Exception:
                # keep alignment even if a conversion fails
                logger.exception("Failed converting value for row=%s col=%s", ngram, col)
                y.append(np.nan)

        if all(np.isnan(v) for v in y):
            continue

        color = cmap(i % cmap.N)
        ls = linestyles[i % len(linestyles)]
        plt.plot(list(windows_sorted), y, label=ngram, color=color, linestyle=ls, marker="o")
        plotted_any = True

    # Optionally overlay a baseline (non-windowed) tested model across the same reference windows
    if baseline_tested is not None and baseline_tested in df.index:
        try:
            y_base: list[float] = []
            for col in cols_sorted:
                try:
                    val = df.loc[baseline_tested, col] if col in df.columns else np.nan
                    y_base.append(float(val) if not pd.isna(val) else np.nan)  # type: ignore # noqa: PGH003
                except (ValueError, TypeError):
                    y_base.append(np.nan)

            if not all(np.isnan(v) for v in y_base):
                label = baseline_label or f"baseline: {baseline_tested}"
                plt.plot(
                    list(windows_sorted),
                    y_base,
                    label=label,
                    color="black",
                    linestyle="-",
                    marker="s",
                    linewidth=2.5,
                )
                plotted_any = True
        except (ValueError, TypeError, KeyError):
            logging.getLogger(__name__).exception("Failed plotting baseline tested model: %s", baseline_tested)

    if not plotted_any:
        logger.debug("No data plotted for %s vs %s; skipping file generation.", title, prefix)
        plt.close()
        return False

    plt.xlabel(xlabel)
    plt.ylabel("Percent")
    plt.title(f"{title}: NGrams vs {prefix} windows")
    plt.grid(alpha=grid_alpha)
    plt.legend(title="Ngram", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved ngrams vs %s %s to: %s", prefix, title, out_path.resolve())
    return True

