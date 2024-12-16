import numpy as np

from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.types import Config, Event, Prediction, ProbDistr


def extract_event_fields(event: Event) -> Event:
    return event


# ============================================================
# Probabilities
# ============================================================

stop_symbol = DEFAULT_CONFIG["stop_symbol"]


def probs_prediction(probs: ProbDistr, config: Config) -> Prediction | None:
    """
    Returns the top-k activities based on their probabilities.
    If stop_symbol has a probability of 1.0 and there are no other activities, return None.
    If stop_symbol has a probability of 1.0 and there are other activities, give a uniform distribution to these other activities.
    If stop_symbol is present but with a probability less than 1.0 and include_stop is False, remove it and normalize the rest.
    """
    # If there are no probabilities, return None
    if not probs:
        return None

    # Create a copy of probs to avoid modifying the original dictionary
    probs_copy = probs.copy()

    # Handle the case where include_stop is False
    if not config["include_stop"] and stop_symbol in probs_copy:  # stop_symbol will always be a key
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
            probs_copy = {activity: uniform_prob for activity in probs_copy}

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

    # Convert dictionary to a sorted list of items (activities and probabilities) for consistency
    sorted_probs = sorted(probs_copy.items(), key=lambda x: (-x[1], x[0]))

    # Extract activities and probabilities in a consistent way
    activities, probabilities = zip(*sorted_probs, strict=True)

    # Convert the probabilities to a numpy array
    probabilities_array = np.array(probabilities)

    # Get the indices of the top-k elements, sorted in descending order
    top_k_indices = np.argsort(probabilities_array)[-config["top_k"] :][::-1]

    # Use the indices to get the top-k activities
    top_k_activities = [activities[i] for i in top_k_indices]

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
        "probs": probs_copy,
    }
