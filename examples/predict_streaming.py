"""Streaming prediction example for process mining using LogicSponge."""



import gc
import json
import logging
import time
from pathlib import Path

# ruff: noqa: E402
import torch
from torch import nn, optim

import logicsponge.core as ls  # type: ignore # noqa: PGH003
from logicsponge.core import DataItem, DataItemFilter  # type: ignore # noqa: PGH003

# from logicsponge.core import dashboard
from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.miners import (
    AdaptiveVoting,
    BasicMiner,
    Fallback,
    HardVoting,
    NeuralNetworkMiner,
    RLMiner,  # added RLMiner
    SoftVoting,
    WindowedNeuralNetworkMiner,
)
from logicsponge.processmining.neural_networks import GRUModel, LSTMModel, QNetwork, TransformerModel
from logicsponge.processmining.streaming import (
    ActualCSVWriter,
    AddStartSymbol,
    CSVStatsWriter,
    CustomStreamer,
    Evaluation,
    InfiniteDiscriminerSource,  # noqa: F401
    IteratorStreamer,
    PredictionCSVWriter,
    PrintEval,
    StreamingActivityPredictor,
    SynInfiniteStreamer,  # noqa: F401
)
from logicsponge.processmining.test_data import data_name, dataset
from logicsponge.processmining.utils import add_file_log_handler, save_run_config

logging.basicConfig(
    level=logging.DEBUG,
    # format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Ensure module-specific debug logs are visible
logging.getLogger("logicsponge.processmining.models").setLevel(logging.INFO)
logging.getLogger("logicsponge.processmining.streaming").setLevel(logging.INFO)
logging.getLogger("logicsponge.processmining").setLevel(logging.INFO)

CUSTOM_GENERATOR_PATTERN = [1, 1, 1, 0, 0, 0]
# [0, 1, 0, 1, 2, 1, 0, 1, 2, 3, 2, 1]
# [1, 1, 1, 0, 0, 0]
str_pattern = s = "".join(map(str, CUSTOM_GENERATOR_PATTERN))

logger = logging.getLogger(__name__)
RUN_ID = (
    time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    + f"_{str_pattern if CUSTOM_GENERATOR_PATTERN else data_name}"
)
stats_to_log = []
# create a run-specific results directory: results/{RUN_ID}
run_results_dir = Path(f"results/{RUN_ID}_streaming")
run_results_dir.mkdir(parents=True, exist_ok=True)

# stats and predictions live inside the run folder
stats_file_path = run_results_dir / f"{RUN_ID}_stats_streaming.csv"
predictions_dir = run_results_dir / "predictions"
predictions_dir.mkdir(parents=True, exist_ok=True)

# --- Run configuration (defaults + writing config file like predict_batch.py)
config_file_path = Path(__file__).parent / "predict_config.json"
MAGIC_VALUE = 8
HIDDEN_DIM_DEFAULT = 128
default_run_config = {
    "nn": {"lr": 0.001, "batch_size": 8, "epochs": 20},
    "transf": {"lr": 0.0006, "batch_size": 8, "epochs": 20},
    "rl": {"lr": 0.001, "batch_size": 8, "epochs": 20, "gamma": 0.99},
    "lstm": {
        "vocab_size": MAGIC_VALUE,
        "embedding_dim": MAGIC_VALUE,
        "hidden_dim": HIDDEN_DIM_DEFAULT,
    },
    "gru": {
        "vocab_size": MAGIC_VALUE,
        "embedding_dim": MAGIC_VALUE,
        "hidden_dim": HIDDEN_DIM_DEFAULT,
    },
    "transformer": {
        "seq_input_dim": 32,
        "vocab_size": MAGIC_VALUE,
        "embedding_dim": MAGIC_VALUE,
        "hidden_dim": HIDDEN_DIM_DEFAULT,
    },
    "qlearning": {
        "vocab_size": MAGIC_VALUE,
        "embedding_dim": MAGIC_VALUE,
        "hidden_dim": HIDDEN_DIM_DEFAULT,
    },
}
try:
    with config_file_path.open("w") as _f:
        json.dump(default_run_config, _f, indent=2)
    run_config = default_run_config
except OSError as _e:
    logger.debug("Could not write default config to %s: %s", config_file_path, _e)
    run_config = default_run_config

# Persist the resolved run configuration into the run-specific results folder
try:
    save_run_config(run_config, run_results_dir / "predict_config.json")
except Exception:
    logger.exception("Could not write run config copy to %s; continuing without saving.", run_results_dir)

# Add a file handler so logs are written into the run folder as well
log_file_path = run_results_dir / f"{RUN_ID}_log.txt"
try:
    # remove existing file handlers (if any) to avoid duplicate file logging
    for handler in logging.root.handlers[:]:
        try:
            if isinstance(handler, logging.FileHandler):
                logging.root.removeHandler(handler)
        except Exception:
            logger.exception("Error removing existing file handler.")
    # use helper to add a file handler
    fh = add_file_log_handler(log_file_path, fmt="%(asctime)s %(levelname)s %(name)s: %(message)s")
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

# Select which ML components to train/evaluate (parity with predict_batch.py)
ML_TRAINING = True
NN_TRAINING = True
RL_TRAINING = True
ALERGIA_TRAINING = False
SHOW_DELAYS = False

# Model selector: enable/disable specific models and their variants
MODEL_SELECTOR = {
    # Base NN models
    "lstm": True,
    "gru": False,
    "transformer": True,
    # Attention-head variants for transformer (transformer_2heads, transformer_4heads, ...)
    "transformer_heads": False,
    # transformer variants with different positional encodings
    "transformer_pos_encodings": False,
    # Windowed NN variants
    "window": True,
    # RL models
    "qlearning": False,
}

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

WINDOW_RANGE = [1, 2, 3, 4, 5, 6, 7]  # 0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 12, 14, 16]

# NN/RL window range aligned with predict_batch.py
NN_WINDOW_RANGE = WINDOW_RANGE # [1, 2, 3, 4, 5, 6, 7, 8]

NGRAM_NAMES = [f"ngram_{i + 1}" for i in WINDOW_RANGE]
# ] + [
#     f"ngram_{i+1}_recovery" for i in WINDOW_RANGE
# ]
# ] + [
#     f"ngram_{i+1}_shorts" for i in WINDOW_RANGE
# ]

NGRAM_RETURN_TO_INITIAL = True
# ============================================================

# RL (QNetwork) window configurations (align to batch-style windows)
RL_WINDOWS = NN_WINDOW_RANGE
RL_NAMES = [f"qlearning_linear_win{w}" for w in RL_WINDOWS] + [f"qlearning_gru_win{w}" for w in RL_WINDOWS]
# Two non-windowed baselines (user request): one GRU architecture and one linear architecture
RL_BASELINE_GRU_NAME = "qlearning_gru"
RL_BASELINE_LINEAR_NAME = "qlearning_linear"

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

# Initialize LSTMs/GRUs (base models without window constraint) using run_config
nn_cfg = run_config.get("nn", {})
transf_cfg = run_config.get("transf", {})
lstm_cfg = run_config.get("lstm", {})
gru_cfg = run_config.get("gru", {})
vocab_size = lstm_cfg.get("vocab_size", 50)  # An upper bound on the number of activities
embedding_dim = lstm_cfg.get("embedding_dim", 50)
hidden_dim = lstm_cfg.get("hidden_dim", 128)
num_layers = lstm_cfg.get("num_layers", 2)
model_lstm = LSTMModel(
    vocab_size,
    embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, device=device, use_one_hot=True
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=nn_cfg.get("lr", 0.001))

lstm = StreamingActivityPredictor(
    strategy=NeuralNetworkMiner(
        model=model_lstm,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=nn_cfg.get("batch_size", 8),
        config=config,
    )
)

# GRU model (parity with LSTM settings)
gru_hidden_dim = gru_cfg.get("hidden_dim", hidden_dim)
gru_embedding_dim = gru_cfg.get("embedding_dim", embedding_dim)
gru_vocab = gru_cfg.get("vocab_size", vocab_size)
model_gru = GRUModel(
    gru_vocab,
    embedding_dim=gru_embedding_dim,
    hidden_dim=gru_hidden_dim,
    device=device,
    use_one_hot=True,
)
criterion_gru = nn.CrossEntropyLoss()
optimizer_gru = optim.Adam(model_gru.parameters(), lr=nn_cfg.get("lr", 0.001))

gru = StreamingActivityPredictor(
    strategy=NeuralNetworkMiner(
        model=model_gru,
        criterion=criterion_gru,
        optimizer=optimizer_gru,
        batch_size=nn_cfg.get("batch_size", 8),
        config=config,
    )
)


# Initialize transformer model (base model without window constraint) using run_config
transformer_cfg = run_config.get("transformer", {})
hidden_dim_tr = transformer_cfg.get("hidden_dim", hidden_dim)
model_transformer = TransformerModel(
    vocab_size=transformer_cfg.get("vocab_size", vocab_size),
    embedding_dim=transformer_cfg.get("embedding_dim", embedding_dim),
    hidden_dim=hidden_dim_tr,
    attention_heads=1,
    use_one_hot=True,
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_transformer.parameters(), lr=transf_cfg.get("lr", 0.0001))

transformer = StreamingActivityPredictor(
    strategy=NeuralNetworkMiner(
        model=model_transformer,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=transf_cfg.get("batch_size", 8),
        config=config,
    )
)






# Windowed LSTM/Transformer models (like batch mode NN windowing)
LSTM_MODELS: dict[str, StreamingActivityPredictor] = {}
GRU_MODELS: dict[str, StreamingActivityPredictor] = {}
TRANSFORMER_MODELS: dict[str, StreamingActivityPredictor] = {}

# Positional encodings available for transformer variants. We build both
# windowed (inside the NN_WINDOW_RANGE loop) and base (non-windowed)
# variants using this list.
POS_ENCODINGS = [
    # ("sinusoidal", None),
    # ("periodic", None),
    # ("sharp_relative", "square"),
    # ("learnable_backward_relative", None),
    # ("learnable_relative", None),
]

TRANSFORMER_POSENC_MODELS: dict[str, StreamingActivityPredictor] = {}
TRANSFORMER_POSENC_NAMES: list[str] = []

LSTM_WIN_NAMES = [f"lstm_win{w}" for w in NN_WINDOW_RANGE]
GRU_WIN_NAMES = [f"gru_win{w}" for w in NN_WINDOW_RANGE]
TRANSFORMER_WIN_NAMES = [f"transformer_win{w}" for w in NN_WINDOW_RANGE]

for w in NN_WINDOW_RANGE:
    # LSTM windowed variant
    model_lstm_w = LSTMModel(
        vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device, use_one_hot=True
    )
    criterion_l = nn.CrossEntropyLoss()
    optimizer_l = optim.Adam(model_lstm_w.parameters(), lr=nn_cfg.get("lr", 0.01))
    LSTM_MODELS[f"lstm_win{w}"] = StreamingActivityPredictor(
        strategy=WindowedNeuralNetworkMiner(
            model=model_lstm_w,
            criterion=criterion_l,
            optimizer=optimizer_l,
            batch_size=nn_cfg.get("batch_size", 8),
            sequence_buffer_length=w,
            config=config,
        )
    )

    # GRU windowed variant
    model_gru_w = GRUModel(
        vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        device=device,
        use_one_hot=True,
    )
    criterion_g = nn.CrossEntropyLoss()
    optimizer_g = optim.Adam(model_gru_w.parameters(), lr=nn_cfg.get("lr", 0.001))
    GRU_MODELS[f"gru_win{w}"] = StreamingActivityPredictor(
        strategy=WindowedNeuralNetworkMiner(
            model=model_gru_w,
            criterion=criterion_g,
            optimizer=optimizer_g,
            batch_size=nn_cfg.get("batch_size", 8),
            sequence_buffer_length=w,
            config=config,
        )
    )


    # Create windowed transformer variants for every positional encoding
    for name_enc, _ in POS_ENCODINGS:
        model_name = f"transformer_pos_{name_enc}_win{w}"
        model_tr_pe_w = TransformerModel(
            vocab_size=transformer_cfg.get("vocab_size", vocab_size),
            embedding_dim=transformer_cfg.get("embedding_dim", embedding_dim),
            hidden_dim=hidden_dim_tr,
            attention_heads=1,
            use_one_hot=True,
            pos_encoding_type=name_enc,
        )
        criterion_pe_w = nn.CrossEntropyLoss()
        optimizer_pe_w = optim.Adam(model_tr_pe_w.parameters(), lr=transf_cfg.get("lr", 0.0001))
        TRANSFORMER_POSENC_MODELS[model_name] = StreamingActivityPredictor(
            strategy=WindowedNeuralNetworkMiner(
                model=model_tr_pe_w,
                criterion=criterion_pe_w,
                optimizer=optimizer_pe_w,
                batch_size=transf_cfg.get("batch_size", 8),
                sequence_buffer_length=w,
                config=config,
            )
        )
        TRANSFORMER_POSENC_NAMES.append(model_name)

    # (windowed default transformer handled below once per window)



    # Transformer windowed variant (default positional encoding)
    model_tr_w = TransformerModel(
        vocab_size=transformer_cfg.get("vocab_size", vocab_size),
        embedding_dim=transformer_cfg.get("embedding_dim", embedding_dim),
        hidden_dim=hidden_dim_tr,
        attention_heads=1,
        use_one_hot=True,
        pos_encoding_type="learnable_backward_relative",
    )

    criterion_t = nn.CrossEntropyLoss()
    optimizer_t = optim.Adam(model_tr_w.parameters(), lr=transf_cfg.get("lr", 0.001))
    TRANSFORMER_MODELS[f"transformer_win{w}"] = StreamingActivityPredictor(
        strategy=WindowedNeuralNetworkMiner(
            model=model_tr_w,
            criterion=criterion_t,
            optimizer=optimizer_t,
            batch_size=transf_cfg.get("batch_size", 8),
            sequence_buffer_length=w,
            config=config,
        )
    )


ATTENTION_HEADS = [1, 2, 4]
TRANSFORMER_HEAD_MODELS: dict[str, StreamingActivityPredictor] = {}
TRANSFORMER_HEAD_NAMES: list[str] = []

for heads in ATTENTION_HEADS:
    if heads == 1:
        TRANSFORMER_HEAD_NAMES.append("transformer")
        continue

    model_tr_h = TransformerModel(
        vocab_size=transformer_cfg.get("vocab_size", vocab_size),
        embedding_dim=transformer_cfg.get("embedding_dim", embedding_dim),
        hidden_dim=hidden_dim_tr,
        attention_heads=heads,
        use_one_hot=True,
    )
    criterion_h = nn.CrossEntropyLoss()
    optimizer_h = optim.Adam(model_tr_h.parameters(), lr=transf_cfg.get("lr", 0.0001))

    name = f"transformer_{heads}heads"
    TRANSFORMER_HEAD_MODELS[name] = StreamingActivityPredictor(
        strategy=NeuralNetworkMiner(
            model=model_tr_h,
            criterion=criterion_h,
            optimizer=optimizer_h,
            batch_size=transf_cfg.get("batch_size", 8),
            config=config,
        )
    )
    TRANSFORMER_HEAD_NAMES.append(name)
# POS_ENCODINGS and TRANSFORMER_POSENC_MODELS / NAMES are defined above.
# Build base (non-windowed) variants below using the shared POS_ENCODINGS list.
for name_enc, _ in POS_ENCODINGS:
    if name_enc == "sharp_relative":
        continue
    model_name = f"transformer_pos_{name_enc}"
    model_tr_pe = TransformerModel(
        vocab_size=transformer_cfg.get("vocab_size", vocab_size),
        embedding_dim=transformer_cfg.get("embedding_dim", embedding_dim),
        hidden_dim=hidden_dim_tr,
        attention_heads=1,
        use_one_hot=True,
        pos_encoding_type=name_enc,
    )
    criterion_pe = nn.CrossEntropyLoss()
    optimizer_pe = optim.Adam(model_tr_pe.parameters(), lr=transf_cfg.get("lr", 0.0001))
    TRANSFORMER_POSENC_MODELS[model_name] = StreamingActivityPredictor(
        strategy=NeuralNetworkMiner(
            model=model_tr_pe,
            criterion=criterion_pe,
            optimizer=optimizer_pe,
            batch_size=transf_cfg.get("batch_size", 8),
            config=config,
        )
    )
    TRANSFORMER_POSENC_NAMES.append(model_name)

# RL (QNetwork) models built in a loop
RL_MODELS: dict[str, StreamingActivityPredictor] = {}
for w in RL_WINDOWS:
    q_cfg = run_config.get("qlearning", {})
    model_qlearning = QNetwork(
        vocab_size=q_cfg.get("vocab_size", vocab_size),
        embedding_dim=q_cfg.get("embedding_dim", embedding_dim),
        hidden_dim=q_cfg.get("hidden_dim", hidden_dim),
        device=device,
        model_architecture="gru"
    ).to(device)
    criterion_q = nn.MSELoss()
    optimizer_q = optim.Adam(model_qlearning.parameters(), lr=run_config.get("rl", {}).get("lr", 0.001))

    RL_MODELS[f"qlearning_gru_win{w}"] = StreamingActivityPredictor(
        strategy=RLMiner(
            model=model_qlearning,
            criterion=criterion_q,
            optimizer=optimizer_q,
            config=config,
            sequence_buffer_length=w,  # has to be enough to cover short_term_mem_size
            long_term_mem_size=8,
            short_term_mem_size=16,  # ~n in n-gram
        )
    )

    model_qlearning = QNetwork(
        vocab_size=q_cfg.get("vocab_size", vocab_size),
        embedding_dim=q_cfg.get("embedding_dim", embedding_dim),
        hidden_dim=q_cfg.get("hidden_dim", hidden_dim),
        device=device,
        model_architecture="linear"
    ).to(device)
    criterion_q = nn.MSELoss()
    optimizer_q = optim.Adam(model_qlearning.parameters(), lr=run_config.get("rl", {}).get("lr", 0.001))

    RL_MODELS[f"qlearning_linear_win{w}"] = StreamingActivityPredictor(
        strategy=RLMiner(
            model=model_qlearning,
            criterion=criterion_q,
            optimizer=optimizer_q,
            config=config,
            sequence_buffer_length=w,  # has to be enough to cover short_term_mem_size
            long_term_mem_size=8,
            short_term_mem_size=16,  # ~n in n-gram
        )
    )

"""Add two non-windowed baselines (large buffer approximates "no window")"""
# GRU baseline
model_qlearning_gru_base = QNetwork(
    vocab_size=run_config.get("qlearning", {}).get("vocab_size", vocab_size),
    embedding_dim=run_config.get("qlearning", {}).get("embedding_dim", embedding_dim),
    hidden_dim=run_config.get("qlearning", {}).get("hidden_dim", hidden_dim),
    device=device,
    model_architecture="gru",
).to(device)
criterion_q_gru_base = nn.MSELoss()
optimizer_q_gru_base = optim.Adam(model_qlearning_gru_base.parameters(), lr=run_config.get("rl", {}).get("lr", 0.001))

RL_MODELS[RL_BASELINE_GRU_NAME] = StreamingActivityPredictor(
    strategy=RLMiner(
        model=model_qlearning_gru_base,
        criterion=criterion_q_gru_base,
        optimizer=optimizer_q_gru_base,
        config=config,
        sequence_buffer_length=10000,
        long_term_mem_size=8,
        short_term_mem_size=16,
    )
)

# Linear baseline
model_qlearning_linear_base = QNetwork(
    vocab_size=run_config.get("qlearning", {}).get("vocab_size", vocab_size),
    embedding_dim=run_config.get("qlearning", {}).get("embedding_dim", embedding_dim),
    hidden_dim=run_config.get("qlearning", {}).get("hidden_dim", hidden_dim),
    device=device,
    model_architecture="linear",
).to(device)
criterion_q_linear_base = nn.MSELoss()
optimizer_q_linear_base = optim.Adam(
    model_qlearning_linear_base.parameters(),
    lr=run_config.get("rl", {}).get("lr", 0.001),
)

RL_MODELS[RL_BASELINE_LINEAR_NAME] = StreamingActivityPredictor(
    strategy=RLMiner(
        model=model_qlearning_linear_base,
        criterion=criterion_q_linear_base,
        optimizer=optimizer_q_linear_base,
        config=config,
        sequence_buffer_length=10000,
        long_term_mem_size=8,
        short_term_mem_size=16,
    )
)

# ====================================================
# Sponge
# ====================================================

# Model names (filtered by MODEL_SELECTOR)
models = [
    "fpt",
    "bag",
    *NGRAM_NAMES,
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
    *list(soft_voting_predictors.keys()),
]

if MODEL_SELECTOR.get("lstm", False):
    models.append("lstm")
    if MODEL_SELECTOR.get("window", False):
        models.extend(LSTM_WIN_NAMES)
if MODEL_SELECTOR.get("gru", False):
    models.append("gru")
    if MODEL_SELECTOR.get("window", False):
        models.extend(GRU_WIN_NAMES)
if MODEL_SELECTOR.get("transformer", False):
    models.append("transformer")
    if MODEL_SELECTOR.get("window", False):
        models.extend(TRANSFORMER_WIN_NAMES)
if MODEL_SELECTOR.get("qlearning", False):
    # Add both baselines
    models.append(RL_BASELINE_GRU_NAME)
    # models.append(RL_BASELINE_LINEAR_NAME)
    if MODEL_SELECTOR.get("window", False):
        models.extend(RL_NAMES)
if MODEL_SELECTOR.get("transformer_heads", False):
    models.extend(TRANSFORMER_HEAD_NAMES)
if MODEL_SELECTOR.get("transformer_pos_encodings", False):
    models.extend(TRANSFORMER_POSENC_NAMES)

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

# streamer = SynInfiniteStreamer(max_prefix_length=10)
# streamer = InfiniteDiscriminerSource()
streamer = CustomStreamer(sequence = CUSTOM_GENERATOR_PATTERN)

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



# Build the prediction sub-pipeline conditionally based on ML/NN flags
prediction_group = (
    (fpt * PredictionCSVWriter(csv_path=predictions_dir / "fpt.csv", model_name="fpt") * Evaluation("fpt"))
    | (bag * PredictionCSVWriter(csv_path=predictions_dir / "bag.csv", model_name="bag") * Evaluation("bag"))
)

# Add NGRAM models
for name in NGRAM_NAMES:
    prediction_group = prediction_group | (
        NGRAM_MODELS[name]
        * PredictionCSVWriter(csv_path=predictions_dir / f"{name}.csv", model_name=name)
        * Evaluation(name)
    )

# Add voting and fallback/hard predictors
prediction_group = prediction_group | (
    hard_voting
    * PredictionCSVWriter(
        csv_path=predictions_dir / "hard_voting.csv",
        model_name="hard_voting",
    )
    * Evaluation("hard_voting")
)

for name, predictor in soft_voting_predictors.items():
    prediction_group = prediction_group | (
        predictor * PredictionCSVWriter(csv_path=predictions_dir / f"{name}.csv", model_name=name) * Evaluation(name)
    )

# Optionally include NN models (only if NN_TRAINING is enabled)
if NN_TRAINING and ML_TRAINING:
    if MODEL_SELECTOR.get("lstm", False):
        prediction_group = prediction_group | (
            AddStartSymbol(start_symbol=start_symbol)
            * lstm
            * DataItemFilter(data_item_filter=start_filter)
            * PredictionCSVWriter(csv_path=predictions_dir / "lstm.csv", model_name="lstm")
            * Evaluation("lstm")
        )

    if MODEL_SELECTOR.get("gru", False):
        prediction_group = prediction_group | (
            AddStartSymbol(start_symbol=start_symbol)
            * gru
            * DataItemFilter(data_item_filter=start_filter)
            * PredictionCSVWriter(csv_path=predictions_dir / "gru.csv", model_name="gru")
            * Evaluation("gru")
        )

    if MODEL_SELECTOR.get("transformer", False):
        prediction_group = prediction_group | (
            AddStartSymbol(start_symbol=start_symbol)
            * transformer
            * DataItemFilter(data_item_filter=start_filter)
            * PredictionCSVWriter(csv_path=predictions_dir / "transformer.csv", model_name="transformer")
            * Evaluation("transformer")
        )

    # Add predictions for all transformer attention-head variants.
    # `TRANSFORMER_HEAD_NAMES` contains the model names; for the
    # special case "transformer" we reuse the already-created
    # `transformer` predictor object. Other names are stored in
    # `TRANSFORMER_HEAD_MODELS` created earlier.
    if MODEL_SELECTOR.get("transformer_heads", False):
        for tname in TRANSFORMER_HEAD_NAMES:
            # The base 'transformer' predictor was already added above; skip re-adding it
            if tname == "transformer":
                continue
            predictor_obj = TRANSFORMER_HEAD_MODELS[tname]
            prediction_group = prediction_group | (
                AddStartSymbol(start_symbol=start_symbol)
                * predictor_obj
                * DataItemFilter(data_item_filter=start_filter)
                * PredictionCSVWriter(csv_path=predictions_dir / f"{tname}.csv", model_name=tname)
                * Evaluation(tname)
            )

    # Add predictions for transformer positional-encoding variants
    if MODEL_SELECTOR.get("transformer_pos_encodings", False):
        for tname in TRANSFORMER_POSENC_NAMES:
            predictor_obj = TRANSFORMER_POSENC_MODELS[tname]
            prediction_group = prediction_group | (
                AddStartSymbol(start_symbol=start_symbol)
                * predictor_obj
                * DataItemFilter(data_item_filter=start_filter)
                * PredictionCSVWriter(csv_path=predictions_dir / f"{tname}.csv", model_name=tname)
                * Evaluation(tname)
            )

    # prediction_group = prediction_group | (
    #     AddStartSymbol(start_symbol=start_symbol)
    #     * transformer_auto
    #     * DataItemFilter(data_item_filter=start_filter)
    #     * PredictionCSVWriter(csv_path=predictions_dir / "transformer_auto.csv", model_name="transformer_auto")
    #     * Evaluation("transformer_auto")
    # )

    if MODEL_SELECTOR.get("lstm", False) and MODEL_SELECTOR.get("window", False):
        for name in LSTM_WIN_NAMES:
            prediction_group = prediction_group | (
                AddStartSymbol(start_symbol=start_symbol)
                * LSTM_MODELS[name]
                * DataItemFilter(data_item_filter=start_filter)
                * PredictionCSVWriter(csv_path=predictions_dir / f"{name}.csv", model_name=name)
                * Evaluation(name)
            )

    if MODEL_SELECTOR.get("gru", False) and MODEL_SELECTOR.get("window", False):
        for name in GRU_WIN_NAMES:
            prediction_group = prediction_group | (
                AddStartSymbol(start_symbol=start_symbol)
                * GRU_MODELS[name]
                * DataItemFilter(data_item_filter=start_filter)
                * PredictionCSVWriter(csv_path=predictions_dir / f"{name}.csv", model_name=name)
                * Evaluation(name)
            )

    if MODEL_SELECTOR.get("transformer", False) and MODEL_SELECTOR.get("window", False):
        for name in TRANSFORMER_WIN_NAMES:
            prediction_group = prediction_group | (
                AddStartSymbol(start_symbol=start_symbol)
                * TRANSFORMER_MODELS[name]
                * DataItemFilter(data_item_filter=start_filter)
                * PredictionCSVWriter(csv_path=predictions_dir / f"{name}.csv", model_name=name)
                * Evaluation(name)
            )

# Optionally include RL models (if ML_TRAINING enabled)
if RL_TRAINING and ML_TRAINING:
    if MODEL_SELECTOR.get("qlearning", False) and MODEL_SELECTOR.get("window", False):
        for name in RL_NAMES:
            prediction_group = prediction_group | (
                AddStartSymbol(start_symbol=start_symbol)
                * RL_MODELS[name]
                * DataItemFilter(data_item_filter=start_filter)
                * PredictionCSVWriter(csv_path=predictions_dir / f"{name}.csv", model_name=name)
                * Evaluation(name)
            )

    if MODEL_SELECTOR.get("qlearning", False):
        # GRU baseline predictions
        prediction_group = prediction_group | (
            AddStartSymbol(start_symbol=start_symbol)
            * RL_MODELS[RL_BASELINE_GRU_NAME]
            * DataItemFilter(data_item_filter=start_filter)
            * PredictionCSVWriter(
                csv_path=predictions_dir / f"{RL_BASELINE_GRU_NAME}.csv", model_name=RL_BASELINE_GRU_NAME
            )
            * Evaluation(RL_BASELINE_GRU_NAME)
        )
        # Linear baseline predictions
        prediction_group = prediction_group | (
            AddStartSymbol(start_symbol=start_symbol)
            * RL_MODELS[RL_BASELINE_LINEAR_NAME]
            * DataItemFilter(data_item_filter=start_filter)
            * PredictionCSVWriter(
                csv_path=predictions_dir / f"{RL_BASELINE_LINEAR_NAME}.csv", model_name=RL_BASELINE_LINEAR_NAME
            )
            * Evaluation(RL_BASELINE_LINEAR_NAME)
        )

# `set_history_bound` is a method on DataStream. The source term `streamer`
# exposes its output DataStream as `_output` (see SourceTerm in logicsponge).
# Call the method on that DataStream instance.
streamer._output.set_history_bound(ls.NumberBound(1))  # noqa: SLF001

# Assemble the full sponge
sponge = (
    streamer
    # * StreamAlteration(alteration_type="switch", rate=1.0, alteration_start=5000, transition=1)
    # * StreamAlteration(alteration_type="split", rate=1.0, alteration_start=5000, transition=1000)
    * ls.KeyFilter(keys=["case_id", "activity", "timestamp"])
    * ActualCSVWriter(csv_path=predictions_dir / "actual.csv")
    * prediction_group
    * ls.MergeToSingleStream()
    * ls.Flatten()
    * ls.AddIndex(key="index", index=1)
    * ls.KeyFilter(keys=all_attributes)
    * ls.DataItemFilter(data_item_filter=lambda item: item["index"] % 100 == 0 or item["index"] == len_dataset - 1)
    * (PrintEval() | CSVStatsWriter(csv_path=stats_file_path))
)



sponge.start()

# dashboard.show_stats(sponge)
# dashboard.run()
