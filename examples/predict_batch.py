import logging
import time
from datetime import timedelta

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from aalpy.learning_algs import run_Alergia
from torch import nn, optim

from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram, Parikh
from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.data_utils import (
    add_input_symbols,
    add_start_to_sequences,
    add_stop_to_sequences,
    data_statistics,
    interleave_sequences,
    split_sequence_data,
    transform_to_seqs,
)
from logicsponge.processmining.models import Alergia, BasicMiner, Fallback, HardVoting, Relativize, SoftVoting
from logicsponge.processmining.neural_networks import LSTMModel, PreprocessData, evaluate_rnn, train_rnn
from logicsponge.processmining.test_data import dataset

mpl.use("Agg")

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line-wrapping

logging.basicConfig(
    format="%(levelname)s: %(message)s",  # Only log level and message, no date
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

if torch.backends.mps.is_available():
    # device = torch.device("mps")
    device = torch.device("cpu")
    logger.info("Using cpu.")

elif torch.cuda.is_available():
    msg = f"Using cuda: {torch.cuda.get_device_name(0)}."
    logger.info(msg)
    device = torch.device("cuda")

else:
    device = torch.device("cpu")
    logger.info("Using cpu.")


torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NN_training = True


# ============================================================
# Determine start and stop symbols
# ============================================================

start_symbol = DEFAULT_CONFIG["start_symbol"]
stop_symbol = DEFAULT_CONFIG["stop_symbol"]

# ============================================================
# Data preparation
# ============================================================

nn_processor = PreprocessData()
data = transform_to_seqs(dataset)
data_statistics(data)

# ============================================================
# Define the number of iterations
# ============================================================

n_iterations = 5
# n_iterations = 1

# Store metrics across iterations
all_metrics = {
    name: {
        "accuracies": [],
        "num_states": [],
        "mean_delay_error": [],
        "mean_actual_delay": [],
        "mean_normalized_error": [],
        "num_delay_predictions": [],
    }
    for name in [
        "fpt",
        "bag",
        "ngram_1",
        "ngram_2",
        "ngram_3",
        "ngram_4",
        "ngram_5",
        "ngram_6",
        "ngram_7",
        "ngram_8",
        "fallback fpt->ngram",
        "hard voting",
        "soft voting",
        "alergia",
        "LSTM",
    ]
}

# Repeat the experiment n_iterations times
for iteration in range(n_iterations):
    msg = f"Starting iteration {iteration + 1}/{n_iterations}..."
    logger.info(msg)

    # ============================================================
    # Data Splitting
    # ============================================================

    train_set_transformed, remainder = split_sequence_data(data, 0.3, random_shuffle=True, seed=iteration)
    val_set_transformed, test_set_transformed = split_sequence_data(remainder, 0.5, random_shuffle=True, seed=iteration)

    # Train set for process miners
    train_set = interleave_sequences(train_set_transformed, False)

    # Append STOP symbol
    train_set_transformed = add_stop_to_sequences(train_set_transformed, stop_symbol)
    val_set_transformed = add_stop_to_sequences(val_set_transformed, stop_symbol)
    test_set_transformed = add_stop_to_sequences(test_set_transformed, stop_symbol)

    # data_statistics(test_set_transformed)

    alergia_train_set_transformed = add_input_symbols(train_set_transformed, "in")

    # ============================================================
    # Initialize Process Miners
    # ============================================================

    config = {
        "include_stop": True,
    }

    fpt = BasicMiner(algorithm=FrequencyPrefixTree(), config=config)

    bag = BasicMiner(algorithm=Bag(), config=config)

    parikh = BasicMiner(algorithm=Parikh(upper_bound=2), config=config)

    ngram_1 = BasicMiner(algorithm=NGram(window_length=0), config=config)

    ngram_2 = BasicMiner(algorithm=NGram(window_length=1), config=config)

    ngram_3 = BasicMiner(algorithm=NGram(window_length=2), config=config)

    ngram_4 = BasicMiner(algorithm=NGram(window_length=3), config=config)

    ngram_5 = BasicMiner(algorithm=NGram(window_length=4), config=config)

    ngram_6 = BasicMiner(algorithm=NGram(window_length=5), config=config)

    ngram_7 = BasicMiner(algorithm=NGram(window_length=6), config=config)

    ngram_8 = BasicMiner(algorithm=NGram(window_length=7), config=config)

    fallback = Fallback(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=4)),
        ],
        config=config,
    )

    hard_voting = HardVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
            # BasicMiner(algorithm=NGram(window_length=5)),
            # BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )

    soft_voting = SoftVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
            # BasicMiner(algorithm=NGram(window_length=5)),
            # BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )

    relativize = Relativize(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=3)),
        ],
        config=config,
    )

    # Train Process Miners
    start_time = time.time()
    for event in train_set:
        fpt.update(event)
        bag.update(event)
        ngram_1.update(event)
        ngram_2.update(event)
        ngram_3.update(event)
        ngram_4.update(event)
        ngram_5.update(event)
        ngram_6.update(event)
        ngram_7.update(event)
        ngram_8.update(event)
        fallback.update(event)
        hard_voting.update(event)
        soft_voting.update(event)
    end_time = time.time()
    elapsed_time = end_time - start_time
    msg = f"Total training time for process miners: {elapsed_time:.4f} seconds"
    logger.info(msg)

    # Train Alergia
    start_time = time.time()
    algorithm = run_Alergia(alergia_train_set_transformed, automaton_type="smm", eps=0.5, print_info=True)
    smm = Alergia(algorithm=algorithm)
    end_time = time.time()
    elapsed_time = end_time - start_time
    msg = f"Training time for Alergia: {elapsed_time:.4f} seconds"
    logger.info(msg)

    # ============================================================
    # Evaluation
    # ============================================================

    # All strategies (without LSTM)
    strategies = {
        "fpt": (fpt, test_set_transformed),
        "bag": (bag, test_set_transformed),
        "ngram_1": (ngram_1, test_set_transformed),
        "ngram_2": (ngram_2, test_set_transformed),
        "ngram_3": (ngram_3, test_set_transformed),
        "ngram_4": (ngram_4, test_set_transformed),
        "ngram_5": (ngram_5, test_set_transformed),
        "ngram_6": (ngram_6, test_set_transformed),
        "ngram_7": (ngram_7, test_set_transformed),
        "ngram_8": (ngram_8, test_set_transformed),
        "fallback fpt->ngram": (fallback, test_set_transformed),
        "hard voting": (hard_voting, test_set_transformed),
        "soft voting": (soft_voting, test_set_transformed),
        "alergia": (smm, test_set_transformed),
    }

    # Store the statistics for each iteration and also print them out
    iteration_data = {
        "Model": [],
        "Correct (%)": [],
        "Wrong (%)": [],
        "Empty (%)": [],
        "Correct (Total)": [],
        "Total Predictions": [],
        "Number of States": [],
        # "Mean Delay Error": [],
        # "Mean Actual Delay": [],
        # "Mean Normalized Error": [],
        # "Delay Predictions": [],
    }

    for strategy_name, (strategy, test_data) in strategies.items():
        strategy.evaluate(test_data, mode="incremental")
        stats = strategy.stats

        total = stats["total_predictions"]
        correct_percentage = (stats["correct_predictions"] / total * 100) if total > 0 else 0
        wrong_percentage = (stats["wrong_predictions"] / total * 100) if total > 0 else 0
        empty_percentage = (stats["empty_predictions"] / total * 100) if total > 0 else 0

        num_states = len(strategy.algorithm.states) if isinstance(strategy, BasicMiner) else None

        # Append data to the iteration data dictionary
        iteration_data["Model"].append(strategy_name)
        iteration_data["Correct (%)"].append(correct_percentage)
        iteration_data["Wrong (%)"].append(wrong_percentage)
        iteration_data["Empty (%)"].append(empty_percentage)
        iteration_data["Correct (Total)"].append(stats["correct_predictions"])
        iteration_data["Total Predictions"].append(total)
        iteration_data["Number of States"].append(num_states)

        # Timing information
        delay_count = stats["num_delay_predictions"]
        if delay_count > 0:
            mean_delay_error = stats["delay_error_sum"] / delay_count
            mean_actual_delay = stats["actual_delay_sum"] / delay_count
            mean_normalized_error = stats["normalized_error_sum"] / delay_count
        else:
            mean_delay_error = None
            mean_actual_delay = None
            mean_normalized_error = None

        # iteration_data["Mean Delay Error"].append(mean_delay_error)
        # iteration_data["Mean Actual Delay"].append(mean_actual_delay)
        # iteration_data["Mean Normalized Error"].append(mean_normalized_error)
        # iteration_data["Delay Predictions"].append(delay_count)

        # Calculate and append accuracy to all_metrics for final statistics
        accuracy = stats["correct_predictions"] / total if total > 0 else 0

        all_metrics[strategy_name]["accuracies"].append(accuracy)
        all_metrics[strategy_name]["num_states"].append(num_states)

        all_metrics[strategy_name]["mean_delay_error"].append(mean_delay_error)
        all_metrics[strategy_name]["mean_actual_delay"].append(mean_actual_delay)
        all_metrics[strategy_name]["mean_normalized_error"].append(mean_normalized_error)
        all_metrics[strategy_name]["num_delay_predictions"].append(delay_count)

    # Create a DataFrame for the iteration and log it
    iteration_df = pd.DataFrame(iteration_data)
    msg = f"\nIteration {iteration + 1} stats:\n{iteration_df}"
    logger.info(msg)

    # LSTM Evaluation
    if NN_training:
        # For RNNs: Append START symbol
        nn_train_set_transformed = add_start_to_sequences(train_set_transformed, start_symbol)
        nn_val_set_transformed = add_start_to_sequences(val_set_transformed, start_symbol)
        nn_test_set_transformed = add_start_to_sequences(test_set_transformed, start_symbol)

        nn_train_set_transformed = nn_processor.preprocess_data(nn_train_set_transformed)
        nn_val_set_transformed = nn_processor.preprocess_data(nn_val_set_transformed)
        nn_test_set_transformed = nn_processor.preprocess_data(nn_test_set_transformed)

        vocab_size = 50  # Assume an upper bound on the number of activities

        # Initialize the model, criterion, and optimizer
        embedding_dim = 50
        hidden_dim = 128
        output_dim = vocab_size  # Output used to predict the next activity

        model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the LSTM on the train set with batch size and sequence-to-sequence targets
        model = train_rnn(
            model, nn_train_set_transformed, nn_val_set_transformed, criterion, optimizer, batch_size=8, epochs=20
        )

        lstm_accuracy = evaluate_rnn(model, nn_test_set_transformed, dataset_type="Test")
        all_metrics["LSTM"]["accuracies"].append(lstm_accuracy)

# ============================================================
# Calculate and Show Final Results
# ============================================================

results = {
    "Model": [],
    "Mean Accuracy (%)": [],
    "Std": [],
    "States": [],
    "Delay Error": [],
    "Actual Delay": [],
    "Normalized Error": [],
    "Delay Predictions": [],
}

for model_name, stats in all_metrics.items():
    results["Model"].append(model_name)

    if len(stats["accuracies"]) > 0:
        mean_acc = np.mean(stats["accuracies"]) * 100
        std_acc = np.std(stats["accuracies"]) * 100
    else:
        mean_acc = None
        std_acc = None

    results["Mean Accuracy (%)"].append(mean_acc)
    results["Std"].append(std_acc)

    if len(stats["num_states"]) > 0 and None not in stats["num_states"]:
        mean_num_states = np.mean(stats["num_states"])
    else:
        mean_num_states = None

    results["States"].append(mean_num_states)

    if len(stats["mean_delay_error"]) > 0 and None not in stats["mean_delay_error"]:
        mean_delay_error = timedelta(seconds=float(np.mean(stats["mean_delay_error"])))
    else:
        mean_delay_error = None

    if len(stats["mean_actual_delay"]) > 0 and None not in stats["mean_actual_delay"]:
        mean_actual_delay = timedelta(seconds=float(np.mean(stats["mean_actual_delay"])))
    else:
        mean_actual_delay = None

    if len(stats["mean_normalized_error"]) > 0 and None not in stats["mean_normalized_error"]:
        mean_normalized_error = np.mean(stats["mean_normalized_error"])
    else:
        mean_normalized_error = None

    if len(stats["num_delay_predictions"]) > 0 and None not in stats["num_delay_predictions"]:
        num_delay_predictions = np.mean(stats["num_delay_predictions"])
    else:
        num_delay_predictions = None

    def format_timedelta(td: timedelta | None) -> str | None:
        if td is None:
            return None
        days = td.days
        hours, rem = divmod(td.seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"

    # Assuming mean_delay_error and mean_actual_delay are timedelta objects
    results["Delay Error"].append(format_timedelta(mean_delay_error))
    results["Actual Delay"].append(format_timedelta(mean_actual_delay))

    # Normalized Error and Delay Predictions (assuming they are not timedelta)
    results["Normalized Error"].append(mean_normalized_error)  # Round to two decimals
    results["Delay Predictions"].append(num_delay_predictions)  # Keep as-is

# Create a DataFrame and print it
df = pd.DataFrame(results)
msg = "\n" + str(df)
logger.info(msg)
