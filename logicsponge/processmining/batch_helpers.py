"""
Helpers for batch prediction/evaluation scripts.

These utilities encapsulate common bookkeeping for iteration-level records
without changing outputs or behavior.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.miners import BasicMiner, Fallback, HardVoting, SoftVoting
from logicsponge.processmining.utils import (
    RED_TO_GREEN_CMAP,
    compare_models_comparison,
    save_all_comparison_heatmaps,
)

# Global default for left-padding behavior (align sequence ends)
LEFT_PAD_DEFAULT = True

def _left_pad_window(prefix: torch.Tensor, window_size: int) -> torch.Tensor:
    """Crop prefix to last window_size tokens then left-pad if shorter."""
    if prefix.shape[1] > window_size:
        return prefix[:, -window_size:]
    if prefix.shape[1] < window_size:
        pad_len = window_size - prefix.shape[1]
        pad = torch.zeros((prefix.shape[0], pad_len), dtype=prefix.dtype, device=prefix.device)
        return torch.cat([pad, prefix], dim=1)
    return prefix

if TYPE_CHECKING:
    import logging


def _pick_perplexity_value(
    key: str,
    perplexity_stats: dict[str, Any] | None,
    fallback_stats: dict[str, Any],
) -> float | None:
    """
    Return the perplexity value to record for a given key.

    Preference order:
    1) Provided perplexity_stats (e.g., from compute_perplexity_stats)
    2) Fallback from model stats (miners expose pp_* inside stats)
    3) None
    """
    if perplexity_stats is not None:
        return perplexity_stats.get(key)
    return fallback_stats.get(key)


def record_model_results(  # noqa: PLR0913, PLR0915
    *,
    display_name: str,
    stats: dict[str, Any],
    perplexity_stats: dict[str, Any] | None,
    eval_time_micro: float,
    train_time_micro: float,
    num_states: int | None,
    top_k: int,
    iteration_data: dict[str, list[Any]],
    all_metrics: dict[str, dict[str, list[Any]]],
    stats_to_log: list[dict[str, Any]],
    mean_delay_error: float | None = None,
    mean_actual_delay: float | None = None,
    mean_normalized_error: float | None = None,
    num_delay_predictions: int | None = None,
    per_state_stats: dict[str, Any] | None = None,
) -> None:
    """
    Update iteration_data, all_metrics and stats_to_log consistently.

    Mirrors original inline behavior without changing output semantics.
    """
    total = int(stats.get("total_predictions", 0))
    correct = int(stats.get("correct_predictions", 0))

    # Accuracy: prefer provided, else derive from counts
    if "accuracy" in stats and isinstance(stats["accuracy"], (int, float)):
        accuracy = float(stats["accuracy"])  # already in [0,1]
    else:
        accuracy = (correct / total) if total > 0 else 0.0

    # Percentages: if wrong/empty available, use exact counts; else fallback to RL/NN behavior
    if "wrong_predictions" in stats and "empty_predictions" in stats:
        wrong_percentage = (stats["wrong_predictions"] / total * 100) if total > 0 else 0.0
        empty_percentage = (stats["empty_predictions"] / total * 100) if total > 0 else 0.0
    else:
        wrong_percentage = 100.0 - (accuracy * 100.0)
        empty_percentage = 0.0

    # Top-k percentages derived from counts
    top_k_percentages: list[float] = []
    for k in range(1, top_k):
        if (
            total > 0
            and "top_k_correct_preds" in stats
            and isinstance(stats["top_k_correct_preds"], list)
            and k < len(stats["top_k_correct_preds"])
        ):
            val = stats["top_k_correct_preds"][k] / total * 100.0
        else:
            val = 0.0
        top_k_percentages.append(val)

    # Perplexity fields
    pp_harmo = _pick_perplexity_value("pp_harmonic_mean", perplexity_stats, stats)
    pp_arith = _pick_perplexity_value("pp_arithmetic_mean", perplexity_stats, stats)
    pp_median = _pick_perplexity_value("pp_median", perplexity_stats, stats)
    pp_q1 = _pick_perplexity_value("pp_q1", perplexity_stats, stats)
    pp_q3 = _pick_perplexity_value("pp_q3", perplexity_stats, stats)

    # ----------------- iteration_data -----------------
    iteration_data["Model"].append(display_name)

    iteration_data["PP Harmo"].append(pp_harmo)
    iteration_data["PP Arithm"].append(pp_arith)
    iteration_data["PP Median"].append(pp_median)
    iteration_data["PP Q1"].append(pp_q1)
    iteration_data["PP Q3"].append(pp_q3)

    iteration_data["Correct (%)"].append(accuracy * 100.0)
    iteration_data["Wrong (%)"].append(wrong_percentage)
    iteration_data["Empty (%)"].append(empty_percentage)

    for idx, val in enumerate(top_k_percentages, start=2):
        iteration_data[f"Top-{idx}"].append(val)

    iteration_data["Pred Time"].append(eval_time_micro)
    iteration_data["Train Time"].append(train_time_micro)

    iteration_data["Good Preds"].append(correct)
    iteration_data["Tot Preds"].append(total)
    iteration_data["Nb States"].append(num_states)

    # ----------------- stats_to_log -----------------
    entry = {
        "strategy": display_name,
        "strategy_accuracy": accuracy * 100.0,
        "strategy_perplexity": pp_harmo,
        "strategy_eval_time": eval_time_micro,
    }
    if per_state_stats is not None:
        entry["per_state_stats"] = per_state_stats
    stats_to_log.append(entry)

    # ----------------- all_metrics -----------------
    if display_name not in all_metrics:
        all_metrics[display_name] = {
            "accuracies": [],
            "pp_arithmetic_mean": [],
            "pp_harmonic_mean": [],
            "pp_median": [],
            "pp_q1": [],
            "pp_q3": [],
            "top-2": [],
            "top-3": [],
            "num_states": [],
            "train_time": [],
            "pred_time": [],
            "mean_delay_error": [],
            "mean_actual_delay": [],
            "mean_normalized_error": [],
            "num_delay_predictions": [],
        }

    all_metrics[display_name]["accuracies"].append(accuracy)
    all_metrics[display_name]["pp_arithmetic_mean"].append(pp_arith)
    all_metrics[display_name]["pp_harmonic_mean"].append(pp_harmo)
    all_metrics[display_name]["pp_median"].append(pp_median)
    all_metrics[display_name]["pp_q1"].append(pp_q1)
    all_metrics[display_name]["pp_q3"].append(pp_q3)

    # Fill top-2/top-3 using the computed percentages where available
    for idx, val in enumerate(top_k_percentages, start=2):
        if idx in (2, 3):
            all_metrics[display_name][f"top-{idx}"].append(val)

    all_metrics[display_name]["pred_time"].append(eval_time_micro)
    all_metrics[display_name]["train_time"].append(train_time_micro)

    all_metrics[display_name]["num_states"].append(num_states)
    all_metrics[display_name]["mean_delay_error"].append(mean_delay_error)
    all_metrics[display_name]["mean_actual_delay"].append(mean_actual_delay)
    all_metrics[display_name]["mean_normalized_error"].append(mean_normalized_error)
    all_metrics[display_name]["num_delay_predictions"].append(num_delay_predictions)


def write_prediction_vectors(  # noqa: C901, PLR0912
    *,
    prediction_vectors_memory: dict[str, list[Any]],
    run_id: str,
    predictions_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Persist per-iteration prediction vectors and combined CSVs.

    Mirrors previous in-file try/except behavior to avoid changing outputs.
    """
    try:
        for model_name, iterations in prediction_vectors_memory.items():
            combined_rows = []
            for it_idx, pred_vec in enumerate(iterations, start=1):
                actual_iters = prediction_vectors_memory.get("actual", [])
                actual_vec = None
                if model_name != "actual" and len(actual_iters) >= it_idx:
                    actual_vec = actual_iters[it_idx - 1]

                try:
                    preds = list(pred_vec)

                    if actual_vec is not None:
                        actuals = list(actual_vec)
                        min_len = min(len(preds), len(actuals))
                        preds = preds[:min_len]
                        actuals = actuals[:min_len]
                        df_iter = pd.DataFrame({
                            "index": list(range(len(preds))),
                            "predicted": preds,
                            "actual": actuals,
                        })
                    # Baseline 'actual' case has no separate actual_vec; treat preds as ground truth
                    elif model_name == "actual":
                        df_iter = pd.DataFrame({
                            "index": list(range(len(preds))),
                            "actual": preds,
                        })
                    else:
                        df_iter = pd.DataFrame({
                            "index": list(range(len(preds))),
                            "predicted": preds,
                        })
                except (ValueError, TypeError, OSError):
                    if actual_vec is not None:
                        try:
                            actuals = list(actual_vec)
                            df_iter = pd.DataFrame({"predicted": [str(pred_vec)], "actual": [str(actuals)]})
                        except (ValueError, TypeError, OSError):
                            df_iter = pd.DataFrame({"predicted": [str(pred_vec)]})
                    elif model_name == "actual":
                        df_iter = pd.DataFrame({"actual": [str(pred_vec)]})
                    else:
                        df_iter = pd.DataFrame({"predicted": [str(pred_vec)]})

                iter_csv_path = predictions_dir / f"{run_id}_{model_name}_iter{it_idx}.csv"
                df_iter.to_csv(iter_csv_path, index=False)

                df_iter_insert = df_iter.copy()
                df_iter_insert.insert(0, "iteration", it_idx)
                combined_rows.append(df_iter_insert)

            if combined_rows:
                combined_df = pd.concat(combined_rows, ignore_index=True)
                combined_csv_path = predictions_dir / f"{model_name}.csv"
                combined_df.to_csv(combined_csv_path, index=False)
        logger.info("Saved prediction vectors for %d strategies to %s", len(prediction_vectors_memory), predictions_dir)
    except (OSError, RuntimeError, ValueError, TypeError) as e:
        logger.debug("Could not write prediction vectors to CSV: %s", e, exc_info=True)


def prefix_evaluate_rnn(  # noqa: C901, PLR0912, PLR0913
    *,
    model: torch.nn.Module,
    sequences: torch.Tensor,
    idx_to_activity: dict[int, Any] | None,
    max_k: int,
    window_size: int,
    left_pad: bool = LEFT_PAD_DEFAULT,
) -> tuple[dict[str, Any], list[float], float, list[str]]:
    """
    Evaluate an RNN/Transformer in prefix mode (one prediction per prefix).

    Mirrors inline logic previously in predict_batch.process_neural_model:
    - iterate each padded sequence (zero is padding)
    - for each prefix k=1..valid_len-1, crop last `window_size` tokens
    - run model and use last timestep logits for next-token prediction
    - collect top-k correctness and a flattened prediction vector (activity names)

    Returns (stats, perplexities_list, eval_time_seconds, prediction_vector)
    with perplexities_list intentionally empty (kept for signature parity).
    """
    eval_start = time.time()
    correct = 0
    total = 0
    top_k_correct = [0] * max_k
    prediction_vector: list[str] = []

    model.eval()
    model_device = getattr(model, "device", None)

    with torch.no_grad():
        for i in range(sequences.shape[0]):
            seq = sequences[i]
            valid_len = int((seq != 0).sum().item())
            # Skip sequences too short to have a prefix
            if valid_len < 2:  # noqa: PLR2004
                continue
            # iterate prefixes (k = 1..valid_len-1) to produce one pred per event
            for k in range(1, valid_len):
                prefix = seq[:k].unsqueeze(0)  # [1, k]
                if left_pad:
                    prefix = _left_pad_window(prefix, window_size)
                elif prefix.shape[1] > window_size:
                    prefix = prefix[:, -window_size:]
                if model_device is not None:
                    prefix = prefix.to(device=model_device)

                outputs = model(prefix)
                # Some model forward() implementations (e.g., LSTMModel/GRUModel/QNetwork)
                # return a tuple (logits, hidden). We only need the logits here.
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                # outputs shape: [batch=1, seq_len(<=window_size), vocab] or [1, vocab]
                last_logits = outputs[:, -1, :].squeeze(0) if outputs.dim() == 3 else outputs.squeeze(0)  # noqa: PLR2004

                topk = torch.topk(last_logits, k=max_k)
                pred_idx = int(topk.indices[0].item())

                # Map to activity names when available
                if idx_to_activity and pred_idx in idx_to_activity:
                    prediction_vector.append(str(idx_to_activity[pred_idx]))
                else:
                    prediction_vector.append(str(pred_idx))

                target_idx = int(seq[k].item())
                total += 1
                if pred_idx == target_idx:
                    correct += 1
                    for j in range(max_k):
                        top_k_correct[j] += 1
                else:
                    # count inclusion in top-k
                    for j in range(max_k):
                        if target_idx in {int(x) for x in topk.indices[: j + 1].tolist()}:
                            top_k_correct[j] += 1

    eval_time = time.time() - eval_start
    stats = {
        "accuracy": (correct / total) if total > 0 else 0.0,
        "total_predictions": total,
        "correct_predictions": correct,
        "top_k_correct_preds": top_k_correct,
    }
    perplexities: list[float] = []
    return stats, perplexities, eval_time, prediction_vector


def build_strategies(
    *,
    config: dict[str, Any],
    test_set_transformed: list[list[Any]],
    ngram_names: list[str],
    voting_ngrams: list[tuple[int, ...]],
) -> dict[str, tuple[BasicMiner | Fallback | HardVoting | SoftVoting, list[list[Any]]]]:
    """
    Construct the strategies dict exactly as in predict_batch without NN/RL/Bayesian variants.

    Includes:
      - 'fpt', 'bag'
      - ngram models based on provided ngram_names (no recovery, matching current behavior)
      - 'fallback fpt->ngram' (FPT( min_total_visits=10 ) -> NGram(window=4))
      - 'hard voting' with Bag, FPT(min_total_visits=10), NGram(2,3,4)
      - soft voting variants for each grams in voting_ngrams:
          'soft voting {grams}' (plain NGram windows)
          'soft voting {grams}*' (NGram with min_total_visits=10)
    """
    # Base miners
    fpt = BasicMiner(algorithm=FrequencyPrefixTree(), config=config)
    bag = BasicMiner(algorithm=Bag(), config=config)

    # NGram models (no recovery by default, as in current behavior)
    ngram_models: dict[str, BasicMiner] = {}
    for ngram_name in ngram_names:
        window_length = int(ngram_name.split("_")[1]) - 1
        ngram_models[ngram_name] = BasicMiner(
            algorithm=NGram(window_length=window_length, recover_lengths=[]),
            config=config,
        )

    # Fallback FPT -> NGram(4)
    fallback = Fallback(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=4)),
        ],
        config=config,
    )

    # Hard voting
    hard_voting = HardVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
        ],
        config=config,
    )

    # Soft voting variants
    optional_num_ngrams = 3
    soft_voting_plain = []
    soft_voting_star = []
    for grams in voting_ngrams:
        models_plain = [
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=grams[0])),
            BasicMiner(algorithm=NGram(window_length=grams[1])),
            BasicMiner(algorithm=NGram(window_length=grams[2])),
        ] + (
            [BasicMiner(algorithm=NGram(window_length=grams[optional_num_ngrams]))]
            if len(grams) > optional_num_ngrams
            else []
        )
        models_star = [
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=grams[0], min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=grams[1], min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=grams[2], min_total_visits=10)),
        ] + (
            [BasicMiner(algorithm=NGram(window_length=grams[optional_num_ngrams], min_total_visits=10))]
            if len(grams) > optional_num_ngrams
            else []
        )
        soft_voting_plain.append((grams, SoftVoting(models=models_plain, config=config)))
        soft_voting_star.append((grams, SoftVoting(models=models_star)))

    # Build final strategies mapping
    strategies: dict[str, tuple[BasicMiner | Fallback | HardVoting | SoftVoting, list[list[Any]]]] = {
        "fpt": (fpt, test_set_transformed),
        "bag": (bag, test_set_transformed),
        **{
            ngram_name: (ngram_model, test_set_transformed)
            for ngram_name, ngram_model in ngram_models.items()
        },
        "fallback fpt->ngram": (fallback, test_set_transformed),
        "hard voting": (hard_voting, test_set_transformed),
        **{f"soft voting {grams}": (sv, test_set_transformed) for grams, sv in soft_voting_plain},
        **{f"soft voting {grams}*": (sv, test_set_transformed) for grams, sv in soft_voting_star},
    }

    return strategies


def build_and_save_comparison_matrices(
    *,
    prediction_vectors_memory: dict[str, list[Any]],
    run_id: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Build correlation/anticorrelation/similarity matrices...

    ...from prediction_vectors_memory, save CSVs and render heatmaps.
    """
    try:
        model_keys = list(prediction_vectors_memory.keys())

        correlation_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
        anticorrelation_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
        similarity_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
        iterations_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)

        for tested in model_keys:
            for reference in model_keys:
                res = compare_models_comparison(
                    prediction_vectors_memory,
                    tested_model=tested,
                    reference_model=reference,
                    baseline_model="actual",
                    include_empty=False,
                )
                correlation = res.get("correlation")
                anticorrelation = res.get("anticorrelation")
                similarity = res.get("similarity")
                counts = res.get("counts", {})
                iter_used = counts.get("iterations_used")

                correlation_df.loc[tested, reference] = (correlation * 100.0) if correlation is not None else np.nan
                anticorrelation_df.loc[tested, reference] = (
                    (anticorrelation * 100.0) if anticorrelation is not None else np.nan
                )
                similarity_df.loc[tested, reference] = (similarity * 100.0) if similarity is not None else np.nan
                iterations_df.loc[tested, reference] = iter_used if iter_used is not None else np.nan

        # Save CSVs
        out_dir.mkdir(parents=True, exist_ok=True)
        correlation_csv_path = out_dir / f"{run_id}_correlation_matrix.csv"
        anticorrelation_csv_path = out_dir / f"{run_id}_anticorrelation_matrix.csv"
        similarity_csv_path = out_dir / f"{run_id}_similarity_matrix.csv"
        correlation_df.to_csv(correlation_csv_path)
        anticorrelation_df.to_csv(anticorrelation_csv_path)
        similarity_df.to_csv(similarity_csv_path)
        logger.info(
            "[COMPARISON] Saved cross-reference CSV to: %s, %s, %s",
            correlation_csv_path.resolve(),
            anticorrelation_csv_path.resolve(),
            similarity_csv_path.resolve(),
        )

        # Render and save heatmaps using shared utility (ensures consistent style)
        save_all_comparison_heatmaps(results_dir=out_dir, run_id=run_id, cmap=RED_TO_GREEN_CMAP, annotate=False)

        # Log summary table
        # logger.info(
        #     "\n[COMPARISON] Cross-reference (percent) - rows=tested, cols=reference:\n%s",
        #     correlation_df.round(2),
        # )
        # logger.debug("\n[COMPARISON DEBUG] Iterations used per cell:\n%s", iterations_df)

    except (ValueError, KeyError, TypeError, ZeroDivisionError, IndexError, OSError, RuntimeError):
        logger.exception("Failed to build cross-reference comparison table")




def save_results_summary(csv_path: str | Path, data: pd.DataFrame, logger: logging.Logger) -> None:
    """Save a results summary dictionary as a JSON file."""
    if isinstance(csv_path, str):
        csv_path = Path(csv_path)
    try:
        data.to_csv(csv_path, index=False)
        logger.info("Saved final summary CSV: %s", csv_path)
    except Exception:
        logger.exception("Failed to save final summary CSV.")

    with Path.open(csv_path) as f:
        reader = csv.reader(f)
        # Capture the rendered table as a string so we can both log it and save it to a file
        rows = list(reader)
        table_str = tabulate(rows, headers="firstrow", tablefmt="github")
        logger.info(table_str)

        # Also save the table string next to the CSV for easy reference
        try:
            csv_p = Path(csv_path)
            out_path = csv_p.with_name(f"{csv_p.stem}_table.md")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as out_f:
                out_f.write(table_str)
        except OSError:
            logger.exception("Failed to write tabulated summary to file")


def plot_accuracy_by_window(  # noqa: C901, PLR0913, PLR0915
    *,
    all_metrics: dict[str, dict[str, list[Any]]],
    window_sizes: list[int],
    run_id: str,
    out_dir: Path,
    logger: logging.Logger,
    baseline_curve: list[tuple[int, float]] | None = None,
) -> None:
    """
    Plot accuracy vs window size for three families: NGrams, Transformers and LSTMs.

    For each provided window size `n` the function looks up the corresponding keys
    in `all_metrics` as:
      - NGram -> f"ngram_{n+1}"
      - Transformer -> f"transformer_win{n}"
      - LSTM -> f"LSTM_win{n}"

    The `all_metrics` structure is expected to match the one produced by
    `record_model_results` (accuracies stored as floats in [0,1]). The plotted
    values are percentages (0-100).

    The plot is saved as `{run_id}_accuracy_by_window.png` inside `out_dir`.
    """
    try:
        # Helper to compute mean accuracy (percentage) for a given model key
        def _mean_acc_pct(key: str) -> float:
            if key not in all_metrics:
                return float("nan")
            vals = all_metrics[key].get("accuracies", [])
            if not vals:
                return float("nan")
            # filter None and convert
            nums = [float(v) for v in vals if v is not None]
            if len(nums) == 0:
                return float("nan")
            return float(np.nanmean(nums) * 100.0)

        ngram_vals: list[float] = []
        transformer_vals: list[float] = []
        lstm_vals: list[float] = []

        for n in window_sizes:
            ngram_key = f"ngram_{n+1}"
            transformer_key = f"transformer_win{n}"
            lstm_key = f"LSTM_win{n}"

            ngram_vals.append(_mean_acc_pct(ngram_key))
            transformer_vals.append(_mean_acc_pct(transformer_key))
            lstm_vals.append(_mean_acc_pct(lstm_key))

        # Prepare output dir
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / f"{run_id}_accuracy_by_window.png"

        # Sort by window size to ensure monotonic x (useful if input isn't already sorted)
        paired = sorted(zip(window_sizes, ngram_vals, transformer_vals, lstm_vals, strict=True), key=lambda x: x[0])
        xs, ngram_vals_sorted, transformer_vals_sorted, lstm_vals_sorted = zip(*paired, strict=True)

        xs = np.array(xs, dtype=float)

        # Plot on a logarithmic x-axis (base 2) so exponentially-distributed windows are spaced sensibly.
        plt.figure(figsize=(10, 5))
        plt.plot(xs, ngram_vals_sorted, marker="o", label="NGram")
        plt.plot(xs, transformer_vals_sorted, marker="s", label="Transformer")
        plt.plot(xs, lstm_vals_sorted, marker="^", label="LSTM")
        # If a baseline curve is provided, plot it as a black curve labelled 'Theoretical Best'
        baseline_xs = None
        baseline_ys = None
        if baseline_curve:
            # baseline_curve is list[tuple[int, float]] -> sort and unzip
            baseline_sorted = sorted(baseline_curve, key=lambda t: t[0])
            baseline_xs = np.array([t[0] for t in baseline_sorted], dtype=float)
            baseline_ys = np.array([t[1] for t in baseline_sorted], dtype=float)
            plt.plot(
                baseline_xs,
                baseline_ys,
                marker="D",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor="k",
                color="k",
                linestyle="--",
                label="Theoretical Best",
            )

        # Use log scale to spread out exponential window sizes (base 2 looks natural for powers-of-two)
        plt.xscale("log", base=2)

        # Compute powers-of-two ticks that lie within the plotted x-range.
        # This yields a "normal" log plot appearance: only powers-of-two shown
        # as major x-ticks, and vertical gridlines only at those ticks.
        # Consider baseline xs too when computing axis ticks

        all_xs_for_ticks = np.concatenate([xs, baseline_xs]) if baseline_xs is not None and baseline_xs.size > 0 else xs

        if all_xs_for_ticks.size > 0:
            min_x = float(all_xs_for_ticks.min())
            max_x = float(all_xs_for_ticks.max())
            # Guard against non-positive values (log axis requires positive)
            min_x = max(min_x, 1.0)
            # Compute exponent bounds and create ticks
            min_exp = int(np.floor(np.log2(min_x)))
            max_exp = int(np.ceil(np.log2(max_x)))
            pow2_ticks = [2 ** e for e in range(min_exp, max_exp + 1) if 2 ** e >= min_x and 2 ** e <= max_x]
        else:
            pow2_ticks = []

        # If we have power-of-two ticks, show them; otherwise fall back to default ticking
        if pow2_ticks:
            plt.xticks(pow2_ticks, [str(int(x)) for x in pow2_ticks])
        # Turn off minor ticks so there are no extra vertical grid lines
        plt.minorticks_off()

        plt.xlabel("Window size")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.title("Model accuracy vs window size (log-scale)")
        # Draw horizontal grid lines and vertical grid lines only at the major ticks (powers of two)
        plt.grid(axis="y", alpha=0.3, which="major", linestyle="--")
        plt.grid(axis="x", alpha=0.3, which="major", linestyle="--")
        plt.legend()
        plt.tight_layout()
        # Save raster PNG
        fig = plt.gcf()
        fig.savefig(fig_path)
        # Also save a vectorized SVG for later modifications
        try:
            vec_path = out_dir / f"{run_id}_accuracy_by_window.svg"
            fig.savefig(vec_path)
            logger.info("Saved accuracy-by-window plot to %s and %s", fig_path, vec_path)
        except Exception:
            # If SVG save fails, at least keep the PNG and log the error
            logger.exception("Failed to save vectorized SVG for accuracy-by-window plot")
        plt.close()
    except Exception:
        logger.exception("Failed to generate accuracy-by-window plot")


def evaluate_strategy(  # noqa: ANN201, D103, PLR0913
        strategy_name: str,
        strategy: BasicMiner | Fallback | HardVoting | SoftVoting,
        test_data: list[list[Any]],
        prediction_vectors_memory: dict[str, list[Any]],
        logger: logging.Logger,
        iteration: int,
        *,
        debug: bool,
        compute_prediction_vectors: bool,
        test_events_arg: int,
        sec_to_micro_arg: float,
    ):
    msg = f"Evaluating {strategy_name}..."
    logger.info(msg)

    evaluation_time, prediction_vector = strategy.evaluate(
        test_data,
        mode="incremental",
        debug=debug,
        compute_perplexity=("hard" not in strategy_name and "qlearning" not in strategy_name),
    )
    evaluation_time *= sec_to_micro_arg / test_events_arg

    # Store prediction vector for this strategy and iteration (keep ordering across iterations)
    if compute_prediction_vectors:
        prediction_vectors_memory.setdefault(strategy_name, []).append(prediction_vector)
    else:
        logger.debug(
            "Skipping storage of prediction vector for strategy '%s' (iteration %d) due to dataset size",
            strategy_name,
            iteration + 1,
        )

    # Print debugging info about the stored prediction vector
    try:
        sample = prediction_vector[:10]
    except (TypeError, AttributeError, IndexError):
        sample = repr(prediction_vector)[:200]
    logger.debug(
        "Stored preds | strategy=%s | iteration=%d | preds_len=%d | sample=%s",
        strategy_name,
        iteration + 1,
        len(prediction_vector) if hasattr(prediction_vector, "__len__") else -1,
        sample,
    )
    logger.debug("prediction_vector repr (first 400 chars): %s", repr(prediction_vector)[:400])

    stats = strategy.stats

    per_state_stats = stats.get("per_state_stats", {})
    # Convert each value in the dictionary (PerStateStats) to a dict
    for key, value in per_state_stats.items():
        per_state_stats[key] = value.to_dict()

    return evaluation_time, stats, per_state_stats, prediction_vectors_memory


def baseline_curve(data_name: str) -> list[tuple[int, float]]:
    """
    Return a baseline accuracy-by-window curve for reference in plots.

    Currently uses hardcoded values for known datasets.
    """
    baseline_data = {
        "Synthetic111000": [
            (1, float(2/3 * 100)),
            (2, float(2/3 * 100)),
            (3, 100.0),
            (256, 100.0),
        ],
        "Synthetic11100": [
            (1, 60.0),
            (2, 80.0),
            (3, 100.0),
            (256, 100.0),
        ],
        "Random_Decision_win2": [
            (1, 50.0),
            (2, float(2/3 * 100)),
            (3, float(2/3 * 100)),
            (4, float(5/6 * 100)),
            (256, float(5/6 * 100)),
        ],
        "x1x0": [
            (1, 50.0),
            (2, float(7/12 * 100)),
            (3, float(13/16 * 100)),
            (4, float(13/16 * 100)),
            (5, float(7/8 * 100)),
            (256, float(7/8 * 100)),
        ],
        "x10x01": [
            (1, float(7/12 * 100)),
            (2, float(7/12 * 100)),
            (3, float(5/6 * 100)),
            (4, float(5/6 * 100)),
            (5, float(7/8 * 100)),
            (6, float(11/12 * 100)),
            *[(x, float(11/12 * 100)) for x in [2**(3+i) for i in range(6)]],
        ],
        "Interruption_5": [
            *[
                (x, compute_interruption_best_acc(interruption_length=5, window_size=x))
                for x in [2**i for i in range(9)]
            ]
        ],
        "Interruption_10": [
            *[
                (x, compute_interruption_best_acc(interruption_length=10, window_size=x))
                for x in [2**i for i in range(9)]
            ]
        ],
        "Interruption_20": [
            *[
                (x, compute_interruption_best_acc(interruption_length=20, window_size=x))
                for x in [2**i for i in range(9)]
            ]
        ],
        "Interruption_30": [
            *[
                (x, compute_interruption_best_acc(interruption_length=30, window_size=x))
                for x in [2**i for i in range(9)]
            ]
        ],
    }
    return baseline_data.get(data_name, [])


def sample_datasets(
    *,
    data: list[list[Any]],
    data_test: list[list[Any]],
    fraction: float,
    seed: int = 123,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """
    Return sampled copies of `data` and `data_test` using the provided fraction.

    The sampling is deterministic given the `seed` and uses a single RNG
    instance to mirror previous inline behavior.
    """
    if not (0.0 < fraction <= 1.0):
        msg = "fraction must be in (0.0, 1.0]"
        raise ValueError(msg)
    if fraction >= 1.0:
        return data, data_test

    rng = np.random.default_rng(seed)

    def _sample_list(lst: list[list[Any]]) -> list[list[Any]]:
        if not lst:
            return lst
        n = max(1, int(len(lst) * fraction))
        idx = rng.choice(len(lst), size=n, replace=False)
        return [lst[i] for i in sorted(idx)]

    return _sample_list(data), _sample_list(data_test)


def persist_state_mining(
    state_mining_iteration: dict[str, Any], run_results_dir: Path, iteration: int, logger: logging.Logger
) -> None:
    """
    Persist state-mining information for the given iteration.

    Saves to `run_results_dir/state_mining.json` by merging with any existing file.
    """
    try:
        state_mining_path = run_results_dir / "state_mining.json"
        if state_mining_path.exists():
            with state_mining_path.open("r") as _f:
                existing = json.load(_f)
        else:
            existing = {}

        existing[f"iteration_{iteration}"] = state_mining_iteration

        with state_mining_path.open("w") as _f:
            json.dump(existing, _f, indent=2)

        logger.info("Saved state mining info to %s", state_mining_path)
    except Exception:
        logger.exception("Failed to write state mining file %s", run_results_dir / "state_mining.json")


def compute_interruption_best_acc(interruption_length: int, window_size: int) -> float:
    """Compute the best achievable accuracy for the Interruption dataset."""
    accuracy = (interruption_length - 3) / interruption_length

    addon_1 = 1 if window_size >= interruption_length + 1 else 0

    addon_2 = 2 * max(0, interruption_length + 1 - min(3, window_size)) / (interruption_length - 1)

    accuracy += (addon_1 + addon_2) / interruption_length

    return float(accuracy * 100.0)


if __name__ == "__main__":
    print(baseline_curve("Interruption_10"))  # noqa: T201
