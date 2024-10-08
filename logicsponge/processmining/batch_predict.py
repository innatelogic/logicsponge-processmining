import logging
import random
import time

import matplotlib as mpl
import pandas as pd
from aalpy.learning_algs import run_Alergia

from logicsponge.processmining.algorithms_and_structures import FrequencyPrefixTree, NGram
from logicsponge.processmining.data_utils import (
    add_input_symbols,
    add_stop_to_sequences,
    calculate_percentages,
    data_statistics,
    split_data,
    transform_to_seqs,
)
from logicsponge.processmining.globals import STATS, STOP
from logicsponge.processmining.models import Alergia, BasicMiner, Fallback, Relativize
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


# ============================================================
# Data preparation
# ============================================================

train_set, test_set = split_data(dataset, 0.1)
train_set_transformed = transform_to_seqs(train_set)
test_set_transformed = transform_to_seqs(test_set)
train_set_transformed = add_stop_to_sequences(train_set_transformed, STOP)
test_set_transformed = add_stop_to_sequences(test_set_transformed, STOP)
in_train_set_transformed = add_input_symbols(train_set_transformed, "in")

data_statistics(test_set_transformed)

# ============================================================
# Initialize process miners
# ============================================================

pm_strategy = BasicMiner(algorithm=FrequencyPrefixTree())

ngram_strategy = BasicMiner(algorithm=NGram(window_length=2))

fallback_strategy = Fallback(
    models=[
        BasicMiner(algorithm=FrequencyPrefixTree()),
        # BasicMiner(algorithm=NGram(window_length=3)),
        BasicMiner(algorithm=NGram(window_length=2)),
        # BasicMiner(algorithm=NGram(window_length=1)),
        # BasicMiner(algorithm=NGram(window_length=0)),
    ]
)

relativize_strategy = Relativize(
    models=[
        BasicMiner(algorithm=FrequencyPrefixTree()),
        BasicMiner(algorithm=NGram(window_length=2)),
    ]
)

# Alergia is initialized and trained below in one go.


# ============================================================
# Training of process miners
# ============================================================

for case_id, action_name in train_set:
    pm_strategy.update(case_id, action_name)
    ngram_strategy.update(case_id, action_name)
    fallback_strategy.update(case_id, action_name)
    relativize_strategy.update(case_id, action_name)

smm = run_Alergia(in_train_set_transformed, automaton_type="smm", eps=0.5, print_info=True)
smm_strategy = Alergia(algorithm=smm)

strategies = {
    "pm": (pm_strategy, test_set_transformed),
    "ngram": (ngram_strategy, test_set_transformed),
    "fallback pm->ngram": (fallback_strategy, test_set_transformed),
    "relativize pm->ngram": (relativize_strategy, test_set_transformed),
    "alergia": (smm_strategy, test_set_transformed),
}


# ============================================================
# Run evaluation
# ============================================================

start_time = time.time()
result = {name: strategy.evaluate(data, mode="seq") for name, (strategy, data) in strategies.items()}
end_time = time.time()
elapsed_time = end_time - start_time
msg = f"Time taken: {elapsed_time:.4f} seconds"
logger.info(msg)


# ============================================================
# Show results
# ============================================================

# Convert result to DataFrame and ensure columns are floats for percentage calculation
# Initialize the result_percentages dictionary
result_percentages = {strategy: {key["name"]: 0 for key in STATS.values()} for strategy in strategies}

# Apply percentage calculation for each strategy
for strategy_name in strategies:
    calculate_percentages(result_percentages, result, strategy_name)

# Convert result_percentages to DataFrame for logging/printing
result_percentages_df = pd.DataFrame(result_percentages).T.astype(float)

# Print full count table (Table 1)
logger.info("Count Table:")
logger.info(pd.DataFrame(result).T.astype(float))

# Print the percentage table (Table 2)
logger.info("\nResults:")
logger.info(result_percentages_df[[value["name"] for value in STATS.values()]])
