"""Streaming prediction example for process mining using LogicSponge."""

import gc
import logging
import time
from pathlib import Path

# ruff: noqa: E402
import torch
from torch import nn, optim

logging.basicConfig(
    level=logging.DEBUG,
    # format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Ensure module-specific debug logs are visible
logging.getLogger("logicsponge.processmining.models").setLevel(logging.INFO)
logging.getLogger("logicsponge.processmining.streaming").setLevel(logging.INFO)
logging.getLogger("logicsponge.processmining").setLevel(logging.INFO)


import logicsponge.core as ls  # type: ignore # noqa: PGH003
from logicsponge.core import DataItem, DataItemFilter  # type: ignore # noqa: PGH003

# from logicsponge.core import dashboard
from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.models import (
    AdaptiveVoting,
    BasicMiner,
    Fallback,
    HardVoting,
    NeuralNetworkMiner,
    RLMiner,  # added RLMiner
    SoftVoting,
)
from logicsponge.processmining.neural_networks import LSTMModel, QNetwork, TransformerModel
from logicsponge.processmining.streaming import (
    ActualCSVWriter,
    AddStartSymbol,
    CSVStatsWriter,
    Evaluation,
    IteratorStreamer,
    PredictionCSVWriter,
    PrintEval,
    StreamAlteration,
    StreamingActivityPredictor,
)
from logicsponge.processmining.test_data import data_name, dataset

logger = logging.getLogger(__name__)
RUN_ID = time.strftime("%Y-%m-%d_%H-%M", time.localtime()) + f"_{data_name}"
stats_to_log = []
# create a run-specific results directory: results/{RUN_ID}
run_results_dir = Path(f"results/{RUN_ID}")
run_results_dir.mkdir(parents=True, exist_ok=True)

# stats and predictions live inside the run folder
stats_file_path = run_results_dir / f"{RUN_ID}_stats_streaming.csv"
predictions_dir = run_results_dir / "predictions"
predictions_dir.mkdir(parents=True, exist_ok=True)

# Add a file handler so logs are written into the run folder as well
log_file_path = run_results_dir / f"{RUN_ID}_log.txt"
try:
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
except OSError:
    logger.debug("Could not create log file %s; continuing with console logging.", log_file_path)

# disable circular gc here, since a phase 2 may take minutes
gc.disable()

# def gb_callback_example(phase, info: dict):
#     print("gc", phase, info)
# gc.callbacks.append(gb_callback_example)

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

# ====================================================
# Initialize models
# ====================================================

config = {
    "include_stop": False,
    "include_time": False,  # True is default
}

start_symbol = DEFAULT_CONFIG["start_symbol"]


# ============================================================
# Generate a list of ngrams to test
# ============================================================
SOFT_VOTING_NGRAMS = [
    (2, 3, 5, 8),
    (2, 3, 4, 5),
]  # (2, 3, 6, 8), (2, 3, 5, 6), (2, 3, 4, 6), (2, 3, 6, 7), (2, 3, 7, 8), (2, 3, 6, 8)

WINDOW_RANGE = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # , 9, 10, 12, 14, 16]

NGRAM_NAMES = [f"ngram_{i + 1}" for i in WINDOW_RANGE]
# ] + [
#     f"ngram_{i+1}_recovery" for i in WINDOW_RANGE
# ]
# ] + [
#     f"ngram_{i+1}_shorts" for i in WINDOW_RANGE
# ]

NGRAM_RETURN_TO_INITIAL = True
# ============================================================

fpt = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=FrequencyPrefixTree(), config=config),
)

bag = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=Bag(), config=config),
)

# NGram models
NGRAM_MODELS: dict[str, StreamingActivityPredictor] = {}
for ngram_name in NGRAM_NAMES:
    window_length = int(ngram_name.split("_")[1]) - 1
    # Assuming recovery options are not used in streaming for simplicity, matching current streaming file structure
    NGRAM_MODELS[ngram_name] = StreamingActivityPredictor(
        strategy=BasicMiner(
            algorithm=NGram(window_length=window_length, recover_lengths=[]),
            config=config,
        )
    )


fallback = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=4)),
        ],
        config=config,
    )
)

fallback_ngram8to2 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )
)

fallback_ngram8to3 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=2)),
        ],
        config=config,
    )
)

fallback_ngram8to4 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=3)),
        ],
        config=config,
    )
)

fallback_ngram10to2 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=9)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )
)

fallback_ngram13to2 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=12)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )
)

fallback_ngram8to_ooo = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=6)),
            BasicMiner(algorithm=NGram(window_length=5)),
            BasicMiner(algorithm=NGram(window_length=4)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=1)),
            BasicMiner(algorithm=NGram(window_length=0)),
        ],
        config=config,
    )
)

complex_fallback = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=9, min_total_visits=10, min_max_prob=0.9)),
            BasicMiner(algorithm=NGram(window_length=8, min_total_visits=10, min_max_prob=0.9)),
            BasicMiner(algorithm=NGram(window_length=7, min_total_visits=10, min_max_prob=0.8)),
            BasicMiner(algorithm=NGram(window_length=6, min_total_visits=10, min_max_prob=0.7)),
            BasicMiner(algorithm=NGram(window_length=5, min_total_visits=10, min_max_prob=0.6)),
            BasicMiner(algorithm=NGram(window_length=4, min_total_visits=10, min_max_prob=0.0)),
            BasicMiner(algorithm=NGram(window_length=3, min_total_visits=10, min_max_prob=0.0)),
            BasicMiner(algorithm=NGram(window_length=2, min_total_visits=10, min_max_prob=0.0)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )
)

hard_voting = StreamingActivityPredictor(
    strategy=HardVoting(
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
)

soft_voting = StreamingActivityPredictor(
    strategy=SoftVoting(
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
)

soft_voting_star = StreamingActivityPredictor(
    strategy=SoftVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2, min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=3, min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=4, min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=5, min_total_visits=10)),
            # BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )
)

# list_grams = [(2, 3, 6, 8), (2, 3, 5, 6), (2, 3, 5, 8), (2, 3, 4, 6), (2, 3, 6, 7), (2, 3, 7, 8), (2, 3, 6, 8)]
# Use SOFT_VOTING_NGRAMS for consistency
soft_voting_predictors = {
    f"soft_voting_{grams}": StreamingActivityPredictor(
        strategy=SoftVoting(
            models=[
                BasicMiner(algorithm=Bag()),
                BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
                BasicMiner(algorithm=NGram(window_length=grams[0])),
                BasicMiner(algorithm=NGram(window_length=grams[1])),
                BasicMiner(algorithm=NGram(window_length=grams[2])),
                BasicMiner(algorithm=NGram(window_length=grams[3])),
            ],
            config=config,
        )
    )
    for grams in SOFT_VOTING_NGRAMS
}

adaptive_voting = StreamingActivityPredictor(
    strategy=AdaptiveVoting(
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
)

# Initialize LSTMs
vocab_size = 50  # An upper bound on the number of activities
embedding_dim = 50
hidden_dim = 128
output_dim = vocab_size
model_lstm = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, device=device, use_one_hot=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)

lstm = StreamingActivityPredictor(
    strategy=NeuralNetworkMiner(
        model=model_lstm,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=8,
        config=config,
    )
)


# Initialize transformer model
hidden_dim = 128
output_dim = vocab_size  # Output used to predict the next activity

model_transformer = TransformerModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    use_one_hot=True,
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_transformer.parameters(), lr=0.0001)  # Lower learning rate for transformer

transformer = StreamingActivityPredictor(
    strategy=NeuralNetworkMiner(
        model=model_transformer,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=8,
        config=config,
    )
)

# instantiate q-learning model and wrap in RLMiner
model_qlearning = QNetwork(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    device=device
).to(device)
criterion_q = nn.MSELoss()  # placeholder; RLMiner uses log-prob based updates
optimizer_q = optim.Adam(model_qlearning.parameters(), lr=0.001)

qlearning_vanilla = StreamingActivityPredictor(
    strategy=RLMiner(
        model=model_qlearning,
        criterion=criterion_q,
        optimizer=optimizer_q,
        batch_size=8,
        config=config,
        sequence_buffer_length=24, # has to be enough to cover short_term_mem_size
        long_term_mem_size=8,
        short_term_mem_size=16, # ~n in n-gram
    )
)

# ====================================================
# Sponge
# ====================================================

# Model names
models = [
    "fpt",
    "bag",
    *NGRAM_NAMES,  # Add all NGRAM_NAMES
    "fallback",
    "fallback_ngram8to2",
    "fallback_ngram8to3",
    "fallback_ngram8to4",
    "fallback_ngram10to2",
    "fallback_ngram13to2",
    "fallback_ngram8to_ooo",
    "complex_fallback",
    "hard_voting",
    "adaptive_voting",
    "soft_voting",
    "soft_voting_star",
    *list(soft_voting_predictors.keys()),  # Add all soft_voting predictor names
    "lstm",
    "transformer",
    "qlearning_vanilla",
]



metrics_attributes = [
    "accuracy",
    "top_k_accuracy",
    "pp_arithmetic_mean",
    "pp_harmonic_mean",
    "pp_median",
    "pp_q1",
    "pp_q3",
]
metrics_list = [f"{model}.{attribute}" for model in models for attribute in metrics_attributes]

train_latency_list = [f"{model}.train_latency_mean" for model in models]
predict_latency_list = [f"{model}.predict_latency_mean" for model in models]

latency_mean_list = [f"{model}.latency_mean" for model in models]

delay_attributes = [
    "mean_delay_error",
    "mean_actual_delay",
    "mean_normalized_error",
    "delay_predictions",
]

delay_list = [f"{model}.{attribute}" for model in models for attribute in delay_attributes]

all_attributes = [
    "index",
    *metrics_list,
    *train_latency_list,
    *predict_latency_list,
    *latency_mean_list,
]  # , *delay_list]



streamer = IteratorStreamer(data_iterator=dataset)


def start_filter(item: DataItem) -> bool:
    """Filter function to check if the activity is not the start symbol."""
    return item["activity"] != start_symbol


if data_name == "Helpdesk":
    len_dataset = 53_500
elif data_name == "Sepsis":
    len_dataset = 15_214
elif data_name == "BPI_Challenge_2012":
    len_dataset = 262_200
elif data_name == "BPI_Challenge_2013":
    len_dataset = 65_533
elif data_name == "BPI_Challenge_2014":
    len_dataset = 466_737
elif data_name == "BPI_Challenge_2017":
    len_dataset = 1_202_267
elif data_name == "BPI_Challenge_2018":
    len_dataset = 2_514_266
elif data_name == "BPI_Challenge_2019":
    len_dataset = 1_595_923
else:
    len_dataset = 100_000  # Default value


sponge = (
    streamer
    * StreamAlteration(
        alteration_type="switch",
        rate=1.0,
        alteration_start=5000,
        transition=1,
    )
    * StreamAlteration(
        alteration_type="split",
        rate=1.0,
        alteration_start=5000,
        transition=1000,
    )
    * ls.KeyFilter(keys=["case_id", "activity", "timestamp"])
    * ActualCSVWriter(csv_path=predictions_dir / "actual.csv")
    * (
        (fpt * PredictionCSVWriter(csv_path=predictions_dir / "fpt.csv", model_name="fpt") * Evaluation("fpt"))
        | (bag * PredictionCSVWriter(csv_path=predictions_dir / "bag.csv", model_name="bag") * Evaluation("bag"))
        # Add all NGRAM_MODELS to the pipeline
        | (
            (
                NGRAM_MODELS[name]
                * PredictionCSVWriter(csv_path=predictions_dir / f"{name}.csv", model_name=name)
                * Evaluation(name)
            )
            for name in NGRAM_NAMES
        )
        | (
            fallback
            * PredictionCSVWriter(csv_path=predictions_dir / "fallback.csv", model_name="fallback")
            * Evaluation("fallback")
        )
        # | (fallback_ngram8to2 * Evaluation("fallback_ngram8to2"))
        # | (fallback_ngram8to3 * Evaluation("fallback_ngram8to3"))
        # | (fallback_ngram8to4 * Evaluation("fallback_ngram8to4"))
        # | (fallback_ngram10to2 * Evaluation("fallback_ngram10to2"))
        # | (fallback_ngram13to2 * Evaluation("fallback_ngram13to2"))
        # | (fallback_ngram8to_ooo * Evaluation("fallback_ngram8to_ooo"))
        # | (complex_fallback * Evaluation("complex_fallback"))
        | (
            hard_voting
            * PredictionCSVWriter(csv_path=predictions_dir / "hard_voting.csv", model_name="hard_voting")
            * Evaluation("hard_voting")
        )
        # | (adaptive_voting * Evaluation("adaptive_voting"))
        | (
            soft_voting
            * PredictionCSVWriter(csv_path=predictions_dir / "soft_voting.csv", model_name="soft_voting")
            * Evaluation("soft_voting")
        )
        | (
            soft_voting_star
            * PredictionCSVWriter(csv_path=predictions_dir / "soft_voting_star.csv", model_name="soft_voting_star")
            * Evaluation("soft_voting_star")
        )
        # Add all soft_voting_predictors to the pipeline
        | (
            (
                predictor
                * PredictionCSVWriter(csv_path=predictions_dir / f"{name}.csv", model_name=name)
                * Evaluation(name)
            )
            for name, predictor in soft_voting_predictors.items()
        )
        | (
            AddStartSymbol(start_symbol=start_symbol)
            * lstm
            * DataItemFilter(data_item_filter=start_filter)
            * PredictionCSVWriter(csv_path=predictions_dir / "lstm.csv", model_name="lstm")
            * Evaluation("lstm")
        )
        | (
            AddStartSymbol(start_symbol=start_symbol)
            * transformer
            * DataItemFilter(data_item_filter=start_filter)
            * PredictionCSVWriter(csv_path=predictions_dir / "transformer.csv", model_name="transformer")
            * Evaluation("transformer")
        )
        | (
            AddStartSymbol(start_symbol=start_symbol)
            * qlearning_vanilla
            * DataItemFilter(data_item_filter=start_filter)
            * PredictionCSVWriter(
                csv_path=predictions_dir / "qlearning_vanilla.csv", model_name="qlearning_vanilla"
            )
            * Evaluation("qlearning_vanilla")
        )
    )
    * ls.MergeToSingleStream()
    * ls.Flatten()
    * ls.AddIndex(key="index", index=1)
    * ls.KeyFilter(keys=all_attributes)
    * ls.DataItemFilter(data_item_filter=lambda item: item["index"] % 100 == 0 or item["index"] == len_dataset - 1)
    * (PrintEval() | CSVStatsWriter(csv_path=stats_file_path))
    # * ls.Print()
    # * (dashboard.Plot("Accuracy (%)", x="index", y=accuracy_list))
    # * (dashboard.Plot("Latency Mean (ms)", x="index", y=latency_mean_list))
)


sponge.start()

# dashboard.show_stats(sponge)
# dashboard.run()
