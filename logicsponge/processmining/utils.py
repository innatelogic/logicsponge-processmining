"""Utility functions for process mining."""

import copy
import math

import numpy as np

from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.types import Config, Event, Metrics, Prediction, ProbDistr


def extract_event_fields(event: Event) -> Event:
    """Extract the required fields from an event."""
    return event


# ============================================================
# Probabilities
# ============================================================

stop_symbol = DEFAULT_CONFIG["stop_symbol"]


def probs_prediction(probs: ProbDistr, config: Config) -> Prediction | None:
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
            next_activity_idx = np.random.choice(
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

        return compute_highest_probability(probs_copy)

    return compute_highest_probability(probs)

# def simple_activity_prediction(probs: ProbDistr, config: Config) -> dict | None:
#     """
#     Simplified prediction: return only the 'activity' with highest probability.

#     Does not use randomization or top_k from config. Still respects include_stop
#     and stop_symbol handling similar to probs_prediction.
#     """
#     if not probs:
#         return None

#     probs_copy = probs.copy()

#     if not config["include_stop"] and stop_symbol in probs_copy:
#         stop_probability = probs_copy.get(stop_symbol, 0.0)

#         if stop_probability >= 1.0 and len(probs_copy) == 1:
#             return None

#         if stop_probability >= 1.0 and len(probs_copy) > 1:
#             del probs_copy[stop_symbol]
#             num_activities = len(probs_copy)
#             uniform_prob = 1.0 / num_activities
#             probs_copy = dict.fromkeys(probs_copy, uniform_prob)

#         elif stop_probability < 1.0:
#             del probs_copy[stop_symbol]
#             total_prob = sum(probs_copy.values())
#             if total_prob > 0:
#                 probs_copy = {activity: prob / total_prob for activity, prob in probs_copy.items()}

#     if not probs_copy:
#         return None

#     # choose the activity with highest probability; break ties by activity name
#     sorted_probs = sorted(probs_copy.items(), key=lambda x: (-x[1], x[0]))
#     predicted_activity = sorted_probs[0][0]

#     return {"activity": predicted_activity}


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


def compare_models_prediction_ratio(
    prediction_vectors_memory: dict,
    tested_model: str,
    reference_model: str,
    baseline_model: str = "actual",
) -> dict:
    """
    For each iteration stored in prediction_vectors_memory, compute the ratio:
      (# positions where reference == baseline AND tested == baseline)
      / (# positions where reference == baseline)
    Ignore positions where reference != baseline.

    Returns a dict with:
      - "per_iteration": list of ratios (float) or None when denominator is 0 for that iteration
      - "overall": aggregated ratio across iterations (float) or None if no reference-correct positions
      - "counts": dict with totals {'total_ref_correct', 'total_both_correct', 'iterations_used'}
      - "notes": optional notes about length mismatches
    """

    # Validate presence
    if tested_model not in prediction_vectors_memory:
        raise KeyError(f"tested_model '{tested_model}' not found in prediction_vectors_memory")
    if reference_model not in prediction_vectors_memory:
        raise KeyError(f"reference_model '{reference_model}' not found in prediction_vectors_memory")
    if baseline_model not in prediction_vectors_memory:
        raise KeyError(f"baseline_model '{baseline_model}' not found in prediction_vectors_memory")

    tested_list = prediction_vectors_memory[tested_model]
    reference_list = prediction_vectors_memory[reference_model]
    baseline_list = prediction_vectors_memory[baseline_model]

    iterations = min(len(tested_list), len(reference_list), len(baseline_list))
    per_iteration = []
    total_ref_correct = 0
    total_both_correct = 0
    notes = []

    for it in range(iterations):
        tested_vec = tested_list[it]
        ref_vec = reference_list[it]
        base_vec = baseline_list[it]

        # align lengths conservatively
        n = min(len(tested_vec), len(ref_vec), len(base_vec))
        if len(tested_vec) != len(ref_vec) or len(ref_vec) != len(base_vec):
            notes.append(f"iter_{it}: length_mismatch tested={len(tested_vec)} ref={len(ref_vec)} base={len(base_vec)}")

        ref_correct = 0
        both_correct = 0
        for i in range(n):
            if ref_vec[i] == base_vec[i]:
                ref_correct += 1
                if tested_vec[i] == base_vec[i]:
                    both_correct += 1

        per_ratio = (both_correct / ref_correct) if ref_correct > 0 else None
        per_iteration.append(per_ratio)
        total_ref_correct += ref_correct
        total_both_correct += both_correct

    overall = (total_both_correct / total_ref_correct) if total_ref_correct > 0 else None

    return {
        "per_iteration": per_iteration,
        "overall": overall,
        "counts": {
            "total_ref_correct": total_ref_correct,
            "total_both_correct": total_both_correct,
            "iterations_used": iterations,
        },
        "notes": notes,
    }
