# logicsponge-processmining context

Date: 2026-02-16
Workspace: logicsponge-processmining

## Purpose
Library for process-mining tasks built on logicsponge-core. Focus: event-log prediction in batch and streaming modes, plus stream visualization.

## Quick start
- Streaming example: examples/predict_streaming.py
- Batch example: examples/predict_batch.py
- Results default: results/

## Key modules (processmining)
- logicsponge/processmining/miners.py
  - Core models and ensembles.
  - StreamingMiner: base class for batch + streaming evaluation and stats.
  - Ensemble miners include BasicMiner, HardVoting, SoftVoting, AdaptiveVoting, Fallback (see file for full set).
  - Stats tracked: accuracy, top-k, perplexity, per-state stats, delay prediction error.

- logicsponge/processmining/streaming.py
  - Streaming pipeline terms for logicsponge.core.
  - IteratorStreamer: adapts iterators to DataItem stream.
  - AddStartSymbol, DataPreparation, StreamingActivityPredictor, Evaluation.
  - Handles per-item prediction + online update with latency metrics.

- logicsponge/processmining/batch_helpers.py
  - Helpers for batch eval scripts: record metrics, write prediction vectors, evaluate RNN/Transformer prefixes.

- logicsponge/processmining/data_utils.py
  - Dataset utilities: interleave, add start/stop, split, stats, file download/convert (XES -> CSV).

- logicsponge/processmining/config.py
  - DEFAULT_CONFIG and update_config().

- logicsponge/processmining/types.py
  - Core typed dicts and aliases (Event, Metrics, Config, StateId, etc.).

## Data
- data/ contains CSV event logs (BPI Challenge datasets, Sepsis, Helpdesk, synthetic sets).
- models/ and results/ contain saved models and evaluation outputs.

## Dependencies
- Python >= 3.11 (pyproject.toml)
- Key libs: torch, pandas, numpy, matplotlib, tqdm, pm4py (see requirements.txt).

## Notes
- StreamingActivityPredictor calls strategy.case_metrics() for prediction, then strategy.update() to train online.
- Batch evaluation in StreamingMiner.evaluate() collects perplexity and top-k stats and also returns flattened predictions.
- Delay prediction uses timestamps; if missing timestamps, delay stats are skipped.
