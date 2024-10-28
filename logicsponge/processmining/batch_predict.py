import logging
import random
import time

import matplotlib as mpl
import pandas as pd
from torch import nn, optim

from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.data_utils import (
    add_input_symbols,
    add_start_to_sequences,
    add_stop_to_sequences,
    data_statistics,
    shuffle_sequences,
    split_sequence_data,
    transform_to_seqs,
)
from logicsponge.processmining.globals import START, STOP
from logicsponge.processmining.models import BasicMiner, Fallback, HardVoting, Relativize, SoftVoting
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

random.seed(123)


NN_training = True

# ============================================================
# Data preparation
# ============================================================

# nn_processor = PreprocessData()
#
# # Split dataset into train, validation (for RNNs), and test set
# train_set, remainder = split_data(dataset, 0.3)
# val_set, test_set = split_data(remainder, 0.5)
#
# # Transform into sequences
# train_set_transformed = transform_to_seqs(train_set)
# val_set_transformed = transform_to_seqs(val_set)
# test_set_transformed = transform_to_seqs(test_set)
#
# # Append STOP action
# train_set_transformed = add_stop_to_sequences(train_set_transformed, STOP)
# val_set_transformed = add_stop_to_sequences(val_set_transformed, STOP)
# test_set_transformed = add_stop_to_sequences(test_set_transformed, STOP)
#
# # For Alergia: Transform action into pair ("in", action)
# alergia_train_set_transformed = add_input_symbols(train_set_transformed, "in")
#
# data_statistics(test_set_transformed)


# ============================================================
# Alternative data preparation
# ============================================================

nn_processor = PreprocessData()

data = transform_to_seqs(dataset)
data_statistics(data)

train_set_transformed, remainder = split_sequence_data(data, 0.3)
val_set_transformed, test_set_transformed = split_sequence_data(remainder, 0.5)

# train_set for process miners
train_set = shuffle_sequences(train_set_transformed, False)

# Append STOP action
train_set_transformed = add_stop_to_sequences(train_set_transformed, STOP)
val_set_transformed = add_stop_to_sequences(val_set_transformed, STOP)
test_set_transformed = add_stop_to_sequences(test_set_transformed, STOP)

# For Alergia: Transform action into pair ("in", action)
alergia_train_set_transformed = add_input_symbols(train_set_transformed, "in")


# ============================================================
# Initialize process miners
# ============================================================


config = {
    "include_stop": True,
}

fpt = BasicMiner(algorithm=FrequencyPrefixTree(), config=config)

bag = BasicMiner(algorithm=Bag(), config=config)

ngram_0 = BasicMiner(algorithm=NGram(window_length=0), config=config)

ngram_2 = BasicMiner(algorithm=NGram(window_length=2), config=config)

ngram_3 = BasicMiner(algorithm=NGram(window_length=3), config=config)

ngram_4 = BasicMiner(algorithm=NGram(window_length=4), config=config)

fallback = Fallback(
    models=[
        BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=20)),
        BasicMiner(algorithm=NGram(window_length=3)),
    ],
    config=config,
)

hard_voting = HardVoting(
    models=[
        BasicMiner(algorithm=Bag()),
        BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=20)),
        BasicMiner(algorithm=NGram(window_length=2)),
        BasicMiner(algorithm=NGram(window_length=3)),
        BasicMiner(algorithm=NGram(window_length=4)),
    ],
    config=config,
)

soft_voting = SoftVoting(
    models=[
        # BasicMiner(algorithm=Bag()),
        BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=20)),
        BasicMiner(algorithm=NGram(window_length=2)),
        BasicMiner(algorithm=NGram(window_length=3)),
        BasicMiner(algorithm=NGram(window_length=4)),
    ],
    config=config,
)

relativize = Relativize(
    models=[
        BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=5)),
        BasicMiner(algorithm=NGram(window_length=3)),
    ],
    config=config,
)

# Alergia is initialized and trained below in one go.


# ============================================================
# Training of process miners
# ============================================================

for case_id, action_name in train_set:
    fpt.update(case_id, action_name)
    bag.update(case_id, action_name)
    ngram_0.update(case_id, action_name)
    ngram_2.update(case_id, action_name)
    ngram_3.update(case_id, action_name)
    ngram_4.update(case_id, action_name)
    fallback.update(case_id, action_name)
    hard_voting.update(case_id, action_name)
    soft_voting.update(case_id, action_name)
    relativize.update(case_id, action_name)

# algorithm = run_Alergia(alergia_train_set_transformed, automaton_type="smm", eps=0.5, print_info=True)
# smm = Alergia(algorithm=algorithm)

strategies = {
    "fpt": (fpt, test_set_transformed),
    "bag": (bag, test_set_transformed),
    "ngram_0": (ngram_0, test_set_transformed),
    "ngram_2": (ngram_2, test_set_transformed),
    "ngram_3": (ngram_3, test_set_transformed),
    "ngram_4": (ngram_4, test_set_transformed),
    "fallback fpt->ngram_3": (fallback, test_set_transformed),
    "relativize fpt->ngram_3": (relativize, test_set_transformed),
    "hard voting": (hard_voting, test_set_transformed),
    "soft voting": (soft_voting, test_set_transformed),
    # "alergia": (smm, test_set_transformed),
}


# ============================================================
# Run evaluation
# ============================================================

start_time = time.time()
for strategy, data in strategies.values():
    strategy.evaluate(data, mode="incremental")
end_time = time.time()

elapsed_time = end_time - start_time
msg = f"Time taken: {elapsed_time:.4f} seconds"
logger.info(msg)


# ============================================================
# Show results
# ============================================================

data = {
    "Model": [],
    "Correct (%)": [],
    "Wrong (%)": [],
    "Empty (%)": [],
    "Correct (Total)": [],
    "Total Predictions": [],
}

for name, (strategy, _) in strategies.items():
    stats = strategy.stats
    total = stats["total_predictions"]

    correct_percentage = (stats["correct_predictions"] / total * 100) if total > 0 else 0
    wrong_percentage = (stats["wrong_predictions"] / total * 100) if total > 0 else 0
    empty_percentage = (stats["empty_predictions"] / total * 100) if total > 0 else 0

    data["Model"].append(name)
    data["Correct (%)"].append(correct_percentage)
    data["Wrong (%)"].append(wrong_percentage)
    data["Empty (%)"].append(empty_percentage)
    data["Correct (Total)"].append(stats["correct_predictions"])
    data["Total Predictions"].append(stats["total_predictions"])

# Create a DataFrame and print it
df = pd.DataFrame(data)
logger.info(df)

# ============================================================
# RNN/LSTM Training and Evaluation
# ============================================================

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

    # Evaluate model with test set
    msg = "\nFinished training, evaluating accuracy on test set..."
    logger.info(msg)
    evaluate_rnn(model, nn_test_set_transformed, dataset_type="Test")
