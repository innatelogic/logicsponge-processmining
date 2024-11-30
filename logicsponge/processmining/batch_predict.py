import logging
import time

import matplotlib as mpl
import numpy as np
import pandas as pd
from aalpy.learning_algs import run_Alergia
import torch
from torch import nn, optim

from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram, Parikh
from logicsponge.processmining.data_utils import (
    add_input_symbols,
    add_start_to_sequences,
    add_stop_to_sequences,
    data_statistics,
    interleave_sequences,
    split_sequence_data,
    transform_to_seqs,
)
from logicsponge.processmining.globals import START, STOP
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

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NN_training = False

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
    name: {"accuracies": [], "num_states": []}
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
        "fallback fpt->ngram_6",
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

    # Append STOP action
    train_set_transformed = add_stop_to_sequences(train_set_transformed, STOP)
    val_set_transformed = add_stop_to_sequences(val_set_transformed, STOP)
    test_set_transformed = add_stop_to_sequences(test_set_transformed, STOP)

    data_statistics(test_set_transformed)

    alergia_train_set_transformed = add_input_symbols(train_set_transformed, "in")

    # ============================================================
    # Initialize Process Miners
    # ============================================================

    config = {
        "include_stop": True,
    }

    fpt = BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=1), config=config)

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
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=1)),
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
    for case_id, action_name in train_set:
        fpt.update(case_id, action_name)
        bag.update(case_id, action_name)
        ngram_1.update(case_id, action_name)
        ngram_2.update(case_id, action_name)
        ngram_3.update(case_id, action_name)
        ngram_4.update(case_id, action_name)
        ngram_5.update(case_id, action_name)
        ngram_6.update(case_id, action_name)
        ngram_7.update(case_id, action_name)
        ngram_8.update(case_id, action_name)
        fallback.update(case_id, action_name)
        hard_voting.update(case_id, action_name)
        soft_voting.update(case_id, action_name)
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
        "fallback fpt->ngram_6": (fallback, test_set_transformed),
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

        # Calculate and append accuracy to all_metrics for final statistics
        accuracy = stats["correct_predictions"] / total if total > 0 else 0

        all_metrics[strategy_name]["accuracies"].append(accuracy)
        all_metrics[strategy_name]["num_states"].append(num_states)

    # Create a DataFrame for the iteration and log it
    iteration_df = pd.DataFrame(iteration_data)
    msg = f"\nIteration {iteration + 1} stats:\n{iteration_df}"
    logger.info(msg)

    # LSTM Evaluation
    if NN_training:
        # For RNNs: Append START action
        nn_train_set_transformed = add_start_to_sequences(train_set_transformed, START)
        nn_val_set_transformed = add_start_to_sequences(val_set_transformed, START)
        nn_test_set_transformed = add_start_to_sequences(test_set_transformed, START)

        nn_train_set_transformed = nn_processor.preprocess_data(nn_train_set_transformed)
        nn_val_set_transformed = nn_processor.preprocess_data(nn_val_set_transformed)
        nn_test_set_transformed = nn_processor.preprocess_data(nn_test_set_transformed)

        vocab_size = 50  # Assume an upper bound on the number of activities, or adjust dynamically

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
    "Std Dev Accuracy (%)": [],
    "Mean Number of States": [],
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
    results["Std Dev Accuracy (%)"].append(std_acc)

    if len(stats["num_states"]) > 0 and None not in stats["num_states"]:
        mean_num_states = np.mean(stats["num_states"])
    else:
        mean_num_states = None

    results["Mean Number of States"].append(mean_num_states)

# Create a DataFrame and print it
df = pd.DataFrame(results)
msg = "\n" + str(df)
logger.info(msg)
