# Voting model-selection investigation

This document describes the standalone voting investigation implemented in:

- `logicsponge/processmining/voting_investigation.py`: experiment engine,
  decision-rule framework, command-line interface, and result persistence;
- `logicsponge/processmining/voting_dashboard.py`: interactive dashboard;
- `logicsponge/processmining/assets/voting_investigation.css`: dashboard styles;
- `tests/test_voting_investigation.py`: focused behavioral tests.

The investigation is independent from `examples/predict_batch.py`. It trains
only a small family of constituent process-mining models and compares:

1. soft voting;
2. cheating/oracle voting with a soft-voting fallback;
3. candidate rules and rule-set scenarios that try to select the right constituent model without
   seeing the test label.

Its purpose is not merely to report that oracle voting is better. It records
the event-level evidence required to understand *where* ordinary voting loses
accuracy and to test hypotheses for closing that gap.

## Quick start

Run these commands from the repository root. Activate the existing virtual
environment first:

```bash
source .venv/bin/activate
```

Run an investigation on half of the Sepsis cases:

```bash
python -m logicsponge.processmining.voting_investigation run \
  --data Sepsis_Cases \
  --data-prop 0.5 \
  --windows 2,3,4 \
  --output results/sepsis-voting-investigation
```

Open the saved results in the dashboard:

```bash
python -m logicsponge.processmining.voting_investigation dashboard \
  results/sepsis-voting-investigation
```

Then open:

```text
http://127.0.0.1:8050
```

Stop the dashboard with `Ctrl+C`.

To run the investigation and immediately open its dashboard:

```bash
python -m logicsponge.processmining.voting_investigation run \
  --data Sepsis_Cases \
  --data-prop 0.5 \
  --dashboard
```

## Command-line reference

### Running an experiment

```text
python -m logicsponge.processmining.voting_investigation run [OPTIONS]
```

| Option | Default | Meaning |
|---|---:|---|
| `--data` | `Sepsis_Cases` | Dataset name resolved through the repository's dataset utilities. |
| `--data-prop` | `1.0` | Fraction of grouped cases retained before train/calibration/test shuffling. Must be in `(0, 1]`. |
| `--windows` | `2,3,4` | Comma-separated N-gram window lengths. Bag and FPT are always included. |
| `--seed` | `0` | Seed used for deterministic case-level split shuffling. |
| `--output` | timestamped directory | Directory receiving `summary.json` and `events.jsonl`. |
| `--dashboard` | disabled | Start the dashboard after analysis finishes. |
| `--port` | `8050` | Dashboard port when `--dashboard` is used. |

If `--output` is omitted, results are stored under:

```text
results/voting-investigation-YYYYMMDD-HHMMSS/
```

Examples:

```bash
# Quick smoke experiment
python -m logicsponge.processmining.voting_investigation run \
  --data Sepsis_Cases --data-prop 0.05

# Larger N-gram family
python -m logicsponge.processmining.voting_investigation run \
  --data Sepsis_Cases --windows 1,2,3,4,5,6 \
  --output results/sepsis-many-ngrams

# Reproducible alternate split
python -m logicsponge.processmining.voting_investigation run \
  --data Helpdesk --seed 7 --output results/helpdesk-seed-7
```

### Opening existing results

```text
python -m logicsponge.processmining.voting_investigation dashboard RESULTS [--port PORT]
```

`RESULTS` must be a directory containing both `summary.json` and
`events.jsonl`.

```bash
python -m logicsponge.processmining.voting_investigation dashboard \
  results/sepsis-voting-investigation --port 8060
```

Open `http://127.0.0.1:8060` in that case.

## Constituent models

The default model order is significant and deterministic:

1. `bag`;
2. `fpt`, configured with `min_total_visits=10`;
3. `ngram_2`;
4. `ngram_3`;
5. `ngram_4`.

The N-gram entries change according to `--windows`. For example,
`--windows 2,4,8` produces `ngram_2`, `ngram_4`, and `ngram_8` after Bag and
FPT.

Every constituent is trained once on the same ordered training-event stream.
Soft voting and the oracle baseline are computed from the same trained
constituent predictions. This avoids differences caused by training separate
copies of the models.

The `complexity` value recorded for a model is:

- `0` for Bag;
- `1` for FPT;
- the window length for an N-gram.

It is recorded as an input for future generalization/complexity hypotheses. The
initial built-in rules do not yet select directly from this value.

## Experimental protocol

### 1. Grouping and sampling

Input events are grouped by case ID into complete sequences. `--data-prop` is
then applied by keeping the first requested fraction of grouped cases.

This means `--data-prop 0.5` does **not** randomly sample half of the cases. It
keeps the first half and randomizes those retained cases during splitting.

### 2. Case-level split

Retained cases are shuffled using `--seed` and split approximately into:

- 70% model training;
- 15% rule calibration;
- 15% final testing.

Integer rounding can make the exact percentages differ slightly on small
datasets. Entire cases remain together; events from one case are never divided
between splits.

### 3. Stop events

The configured stop event, `__stop__`, is appended to every training,
calibration, and test sequence. Stop prediction is enabled. Consequently:

- event counts include one stop event per case;
- all reported accuracies include stop-event predictions;
- sequence positions and relative positions include the appended stop event.

### 4. Model training

Only the 70% training split updates the constituent models. Calibration and
test diagnostics perform state transitions through each sequence but do not
update model parameters or frequency tables.

### 5. Rule calibration

Decision rules are fitted using labeled events from the 15% calibration split.
Examples include learning the best model after a suffix pattern or in a
position bucket.

### 6. Final evaluation

The fitted rules select models for the held-out 15% test split. Test labels are
used only after selection to score the resulting predictions. Built-in rule
selection therefore does not use the current test event's true activity.

Only test-event rows are written to `events.jsonl`. Calibration aggregates are
represented indirectly by the fitted rule behavior and hypothesis summaries.

## Exact voting definitions

Let there be `M` constituent models. For test event `i`, let:

- `y_i` be the actual activity;
- `p_im(a)` be model `m`'s probability for activity `a`;
- `ŷ_im` be model `m`'s top-1 prediction after normal prediction processing.

### Constituent-model correctness

Model `m` is correct for event `i` exactly when:

```text
ŷ_im == y_i
```

Top-k inclusion does not count as correctness in this investigation.

### Soft voting

Soft voting sums the full probability distributions with equal model weights,
normalizes the result, and selects its top activity:

```text
soft_score_i(a) = Σ_m p_im(a)
soft_probability_i(a) = soft_score_i(a) / Σ_b soft_score_i(b)
soft_prediction_i = argmax_a soft_probability_i(a)
```

All models currently have weight `1.0`. Soft voting can select an activity that
is not the top-1 prediction of any individual model because it uses every
probability in each distribution.

### Cheating/oracle voting

The oracle selects the true activity if at least one constituent model's top-1
prediction is correct:

```text
constituent_hit_i = any(ŷ_im == y_i for m in models)
oracle_prediction_i = y_i if constituent_hit_i else soft_prediction_i
oracle_correct_i = (oracle_prediction_i == y_i)
```

When one or more models are correct, the recorded oracle prediction is the
actual activity. `oracle_model` is the first correct model in configured model
order, while `correct_models` contains every correct model.

When no constituent is correct, the oracle falls back to soft voting. This
matters because soft voting may recover an activity from non-top probabilities
even when no constituent top-1 prediction is correct.

This is a ceiling for **selecting among constituent top-1 predictions**. It is
not a universal ensemble ceiling. In particular, soft voting can theoretically
recover an activity from non-top probabilities even when no constituent top-1
prediction is correct.

## Accuracy and gap metrics

Unless explicitly stated otherwise, the denominator is the total number of
test events, including appended stop events.

### Strategy accuracy

For soft voting, oracle voting, any constituent model, or any hypothesis:

```text
accuracy = number of events where prediction == actual / number of test events
```

Values in JSON are fractions in `[0, 1]`. The dashboard formats them as
percentages.

### Soft failures

```text
soft_failures = count(soft_prediction_i != y_i)
```

This includes both recoverable and unrecoverable soft-voting errors.

### Oracle-gap event

An event belongs to the oracle gap exactly when soft voting is wrong and the
oracle is correct:

```text
oracle_gap_i = (soft_prediction_i != y_i) and oracle_correct_i
```

These are the events for which a perfect model selector would repair soft
voting without inventing a new prediction.

### Oracle-gap event count

```text
oracle_gap_events = count(oracle_gap_i)
recoverable_soft_failures = oracle_gap_events
```

These two summary fields currently contain the same value under different
interpretations.

### Oracle-gap rate

```text
oracle_gap_rate = oracle_gap_events / number of test events
```

This is also the absolute accuracy improvement of cheating/oracle voting over
soft voting.

Do not confuse this with the fraction of soft failures that are recoverable.
That conditional fraction can be computed as:

```text
recoverable_failure_fraction = recoverable_soft_failures / soft_failures
```

### Correct models when soft voting fails

`correct_models_when_soft_fails` counts how often each model is correct on an
oracle-gap event.

The counts are not mutually exclusive. If two models are correct on one soft
failure, both receive one count. Therefore, the values can sum to more than
`oracle_gap_events`.

### Dashboard recovered-gap metrics

The total recoverable gap is:

```text
recoverable_gap = oracle_accuracy - soft_accuracy
```

For the best non-oracle rule or rule-set scenario, the dashboard also displays:

```text
recovered_gap_with_best_rule = best_rule_accuracy - soft_accuracy
recovered_gap_fraction = recovered_gap_with_best_rule / recoverable_gap
```

The first value is an absolute accuracy change; the fraction says how much of
the available soft-to-oracle opportunity was actually closed.

## Event-level diagnostic metrics

Each line of `events.jsonl` is one test event.

### Sequence fields

| Field | Exact meaning |
|---|---|
| `sequence_id` | Display representation of the case ID. |
| `sequence_index` | Zero-based index of the test sequence after splitting. |
| `sequence_length` | Number of events in the test sequence, including `__stop__`. |
| `position` | Zero-based position of the current event. |
| `position_1based` | `position + 1`, used in the dashboard. |
| `relative_position` | `(position + 1) / sequence_length`; therefore it lies in `(0, 1]`. |
| `prefix` | Actual activities observed before the current event. The current true activity is excluded. |
| `prefix_text` | Prefix joined with ` → ` for display. |
| `suffix_1` | Last one activity from the prefix. |
| `suffix_2` | Last two activities from the prefix. |
| `suffix_3` | Last three activities from the prefix. |
| `actual` | True current activity used only for evaluation. |

At the first sequence position, the prefix and every suffix are empty. For a
position with fewer than three preceding events, `suffix_3` contains all
available preceding events.

### Agreement fields

| Field | Exact meaning |
|---|---|
| `consensus_prediction` | Most frequent non-empty constituent top-1 prediction. Ties follow first occurrence in model order. |
| `agreement_count` | Number of models predicting `consensus_prediction`; it is the size of the largest top-1 agreement group. |
| `distinct_prediction_count` | Number of distinct non-empty constituent top-1 predictions. |
| `correct_model_count` | Number of constituent models whose top-1 prediction equals the actual activity. |

Models with an empty prediction do not contribute to agreement or distinct
prediction counts.

### Per-model fields

The nested `models` list contains one object per constituent:

| Field | Exact meaning |
|---|---|
| `index` | Constituent position used for deterministic tie-breaking. |
| `name` | Stable model label such as `bag` or `ngram_3`. |
| `complexity` | Metadata described in “Constituent models”. |
| `state` | Display representation of the model state used for this prediction. |
| `state_visits` | Training-time `total_visits` for that state when exposed by the model; otherwise `0`. |
| `prediction` | Processed top-1 activity, or an empty string if no prediction exists. |
| `confidence` | Probability assigned to the processed top-1 prediction. |
| `correct` | Whether `prediction == actual`. |

`confidence` values are model-specific and need not be calibrated across
models. A model producing `0.8` is not necessarily more reliable than another
model producing `0.6`; testing that assumption is the purpose of the
highest-confidence hypothesis.

### Ensemble and oracle fields

| Field | Exact meaning |
|---|---|
| `soft_prediction`, `soft_correct` | Soft-voting result and correctness. |
| `oracle_prediction`, `oracle_correct` | Oracle result with soft fallback, and correctness of that final result. |
| `oracle_model` | First correct constituent in model order, or an empty string. |
| `correct_models` | Names of all correct constituents. |
| `oracle_gap` | Whether this is a recoverable soft-voting failure. |

### Hypothesis fields

| Field | Exact meaning |
|---|---|
| `rule_models` | Mapping from hypothesis name to the constituent selected for this event. |
| `rule_predictions` | Mapping from hypothesis name to that constituent's top-1 prediction. |
| `condition_matches` | IDs of favorable conditional hypotheses fulfilled before this event's label is revealed. |

## Built-in decision hypotheses

All calibration-based accuracy below means top-1 accuracy measured only on
calibration events.

### Highest confidence

Selects the constituent with the greatest `confidence` for the current event.
Confidence ties choose the earlier model in configured order. No fitting is
required.

### Best calibration accuracy

Computes each model's overall calibration accuracy and always selects the best
one. Accuracy ties prefer the earlier configured model.

This is a useful lower-complexity reference: a conditional rule should ideally
beat it to justify its extra decisions.

### Position buckets

Default bucket widths are `2`, `5`, and `10`. A rule with width `W` defines:

```text
bucket = zero_based_position // W
```

For each bucket containing at least five calibration events, the rule stores
the model with the highest accuracy in that bucket. Unseen or insufficiently
supported buckets fall back to the globally best calibration model.

### Suffix-pattern rules

Rules are generated for suffix lengths `1`, `2`, and `3`, with minimum support
`3` and `10`.

For each suffix appearing at least the required number of times in calibration
data, the rule selects the model with the best accuracy after that suffix.
Unseen or low-support suffixes fall back to the globally best calibration
model.

The suffix contains only activities preceding the predicted event, so the rule
does not use the current true activity.

### Per-state accuracy

Rules are generated with minimum per-model-state support `3`, `5`, and `10`.
For the current event, the rule looks up each model's calibration accuracy in
that model's current state. It selects the candidate with the highest supported
state accuracy.

If no model has sufficient calibration support for its current state, the rule
falls back to the globally best calibration model.

### Agreement rules

Rules are generated for agreement thresholds `2` and `3`, each with two
fallbacks.

When `agreement_count >= threshold`, the rule selects a model predicting the
consensus activity. If several models predict it, the highest-confidence one is
selected, with model order breaking confidence ties.

Below the threshold:

- `else confidence` selects the highest-confidence model;
- `else accuracy` selects the globally best calibration model.

### Hypothesis summary fields

| Field | Meaning |
|---|---|
| `name` | Unique human-readable rule name. |
| `family` | Group used by dashboard filtering. |
| `accuracy` | Correct rule predictions divided by all test events. |
| `correct` | Number of correct test predictions. |
| `total` | Number of test events. |
| `selected_models` | Number of test events assigned to each constituent. Counts sum to `total`. |

## Advanced model-selection rules

The default hypothesis grid now includes higher-capacity selectors. They are
still evaluated without exposing the current test label: calibration labels
fit the selector, and only previous-event feedback is allowed for adaptive
rules.

### Hierarchical reliability

`hierarchical reliability (support 3)` estimates each model's probability of
being correct using a fine-to-coarse context cascade:

1. model + current state + last three activities + agreement + position bucket;
2. model + state + last two activities + agreement;
3. model + state + agreement;
4. model + state;
5. model + agreement;
6. model-global calibration accuracy.

The first context with sufficient support is selected. Its estimate is shrunk
toward the model-global rate using:

```text
(context_correct + global_rate × prior_weight) /
(context_events + prior_weight)
```

This prevents a tiny state/pattern group from replacing a well-supported
global model. `consensus hierarchy ≥ 3` adds a branch: when at least three
models agree, it chooses only among models making that consensus prediction;
otherwise it uses the full hierarchical candidate set.

### Confidence, state, and prediction-behavior reliability

`confidence/state reliability` calibrates correctness by discretized confidence
(five bins), state-visit band (`0`, `1–2`, `3–9`, `10+`), and agreement. It is
intended to test whether confidence means the same thing for different models
and states.

`prediction/state reliability` adds the model's own predicted activity to the
state context. This can reveal that a model is reliable for one transition but
systematically wrong for another, even when its state-level accuracy looks
acceptable.

`disagreement profile reliability` encodes the full observable profile of which
models agree with soft voting and which agree with top-1 consensus. It tests
pairwise/model-specific disagreement patterns rather than only the number of
agreeing models.

### Prefix and composite decision lists

`prefix/state reliability` first matches repeated full prefixes, then backs off
to suffix, position, state, and global contexts. It is intentionally strict:
full-prefix matches need repeated calibration evidence to be used.

`calibrated decision list` learns favorable single conditions and applies the
highest-weight matching condition, with hierarchical reliability as fallback.
`composite decision list` extends this to two-clause interactions such as:

```text
last-2-activities = Leucocytes → CRP
AND agreement-count = 3
```

Only combinations with positive calibration gain and sufficient support are
kept. This tests whether a pattern is useful only under a particular model
agreement regime.

### Nearest calibration behavior

`nearest calibration behavior (k 32)` is a non-parametric sequence matcher. It
finds calibration events close to the current event using relative position,
agreement/diversity, suffixes, each model's prediction, state, and confidence.
For each constituent, correctness is inverse-distance weighted across the
nearest neighbors and shrunk toward its global rate. No test labels are used
to define the distance or the selected model.

### Stacked rule portfolio

`stacked rule portfolio` is a hierarchical composition of selectors. It first
fits a portfolio of confidence, state, agreement, hierarchical, and
prediction-behavior rules on three quarters of calibration events. The
remaining quarter is an internal validation set used to learn which rule is
most reliable for each context (suffix/agreement/position, then broader
contexts). Constituent rules are finally refit on all calibration events before
test evaluation.

This internal holdout is important: choosing the best rule on the same events
used to fit that rule would overstate the portfolio's accuracy.

### Delayed-feedback adaptive rule

`delayed-feedback adaptive` starts with calibration priors and updates decayed
model/state/context reliability after each test event has been scored. The
update uses the observed actual activity only for the *next* event. Its decay
parameter (`0.94` by default) controls how quickly recent behavior replaces
older evidence.

This is valid for a streaming deployment where the previous event's outcome is
known. It is not valid for a batch setting in which labels are unavailable
between predictions. It also cannot repair the first event of a stream before
any feedback exists.

### Interpreting the expanded grid

The extra rules are a search space, not a guarantee of improvement. On the
Sepsis split used by the project (`data-prop=0.5`, seed `0`, windows `2,3,4`),
the strongest added selector was `consensus hierarchy ≥ 3`, while ordinary
`agreement ≥ 3, else confidence` remained stronger. This is useful evidence:
the state/pattern rules are not yet closing the entire oracle gap, so the next
experiments should vary context support, split seeds, and model families rather
than treating one split's leaderboard as a final gating policy.

The meaningful target is not merely the highest selector accuracy. Inspect
each selector's recoveries, harms, and `net_correct` relative to soft voting,
then confirm any positive rule on an untouched final holdout.

## Soft-voting error and impact analysis

The soft-vote analysis answers two separate questions:

1. on which soft-voting errors did an existing calibrated rule or constituent
   have the correct alternative; and
2. which observable, pre-label conditions suggest that a particular model
   should replace soft voting?

Every soft-voting error is retained in the dashboard table. For each one, the
table reports the correct constituents, built-in rules that produced a
different prediction, rules that recovered the error, and all favorable
conditions fulfilled by the event.

### Override-impact metrics

For a candidate selector, let `q_i` be its prediction and `s_i` the soft-vote
prediction. The following counts are evaluated on held-out test events:

| Metric | Exact definition |
|---|---|
| `triggered` | `count(q_i != s_i)`. Events where the candidate actually changes the soft-vote decision. |
| `coverage` | `triggered / total test events`. |
| `soft_errors` | `count(s_i != y_i)`. |
| `recoveries` | `count(s_i != y_i and q_i == y_i)`. Soft errors repaired by the candidate. |
| `harms` | `count(s_i == y_i and q_i != y_i)`. Correct soft predictions broken by the candidate. |
| `unchanged_errors` | `count(s_i != y_i and q_i != y_i)`. It includes changed predictions that remain wrong. |
| `net_correct` | `recoveries - harms`. Change in the number of correct predictions. |
| `net_accuracy_delta` | `net_correct / total test events`. Absolute accuracy change relative to soft voting. |
| `conditional_delta` | `net_correct / triggered`. Net improvement per actual override. Zero if no override occurs. |
| `soft_error_recall` | `recoveries / soft_errors`. Fraction of all soft errors repaired. |
| `override_precision` | `recoveries / triggered`. Fraction of all changed decisions that repair a soft error. |
| `decisive_precision` | `recoveries / (recoveries + harms)`. Fraction of correctness-changing decisions that help rather than hurt. |
| `resulting_accuracy` | `count(q_i == y_i) / total test events`. |

`net_correct` is the primary impact ranking because it penalizes a hypothesis
that recovers many errors by also destroying correct soft-voting decisions.
`recoveries` alone is not sufficient evidence of improvement.

### Favorable conditional hypotheses

The engine enumerates categorical conditions using only information available
at prediction time:

- consensus strength (`low/no majority`, `majority`, or `unanimous`), exact
  agreement count, and prediction diversity;
- early, middle, or late sequence stage and position buckets of width 2 and 5;
- exact last-activity patterns of length 1, 2, and 3;
- the set of models forming the consensus;
- the set of models disagreeing with soft voting;
- whether each particular model agrees or disagrees with soft voting and with
  the top-1 consensus.

For each condition supported by at least 10 calibration events, the engine
calculates every constituent's calibration accuracy and recommends the best
one. A condition is retained only when that constituent beats soft voting on
the same calibration events:

```text
calibration_gain = recommended_model_accuracy - soft_voting_accuracy
weight = calibration_gain * sqrt(calibration_support)
```

The square-root support factor rewards repeated evidence without letting very
common, weak conditions dominate linearly. At most 250 favorable conditions
are retained. Their recommendation and weight are fixed from calibration;
their recoveries, harms, and net impact are then measured on test events.

The condition table is ranked by held-out `net_correct`, with recoveries as the
tie-breaker. This ranking describes the current test split and must not be used
to refit the same reported test result. Confirm promising conditions on other
seeds or a new final holdout.

### Weighted favorable-condition selector

At an event, every selected condition that is fulfilled votes for its
recommended constituent. Votes are summed by constituent, and the model with
the largest total supplies the prediction. Model order breaks equal vote
totals. If none of the selected conditions is fulfilled, the selector keeps
the soft-voting prediction.

The dashboard exposes three weights:

- **Calibration gain x sqrt(support)**: the default `weight` above;
- **Calibration gain**: ignores support after the minimum threshold;
- **Equal vote**: assigns weight 1 to every fulfilled condition.

The selector can combine correlated conditions, so the result is not a causal
estimate and conditions are not statistically independent. Use the displayed
model-selection counts and override count to detect combinations dominated by
one model or by several near-duplicate conditions.

### Recoverable soft failures

```text
recoverable_soft_failures =
    count(soft_prediction_i != y_i and any constituent prediction equals y_i)

recoverable_soft_failure_rate = recoverable_soft_failures / soft_failures
```

The rate uses soft failures—not all test events—as its denominator. It is the
fraction of soft-voting mistakes that a perfect top-1 constituent selector
could repair.

## Result directory

Every completed run contains:

```text
results/sepsis-voting-investigation/
├── summary.json
└── events.jsonl
```

The dashboard reads these files directly; it does not retrain models.

### `summary.json`

Top-level fields:

| Field | Meaning |
|---|---|
| `dataset` | Resolved dataset name. |
| `created_at` | Local timestamp with timezone offset. |
| `models` | Constituent model names in decision/tie-breaking order. |
| `train_events` | Training-event count including appended stop events. |
| `calibration_events` | Calibration-event count including stop events. |
| `test_events` | Test-event count including stop events. |
| `strategies` | Soft and cheating-voting accuracies. |
| `per_model` | Constituent accuracy and correct-event count. |
| `hypotheses` | Rules sorted from highest to lowest test accuracy. |
| `rule_scenarios` | Fixed rule-set consensus scenarios and diagnostic rule-set ceilings. |
| `best_rule_result` | Highest-accuracy non-oracle individual rule or rule-set scenario. |
| `recoverable_gap` | Cheating-voting accuracy minus soft-voting accuracy. |
| `recovered_gap_with_best_rule` | Best candidate accuracy minus soft-voting accuracy. |
| `recovered_gap_fraction` | Fraction of the soft-to-oracle gap closed by the best candidate. |
| `oracle_gap_events` | Recoverable soft-voting failure count. |
| `oracle_gap_rate` | Recoverable failure count divided by all test events. |
| `soft_failures` | Total soft-voting error count. |
| `recoverable_soft_failures` | Same count as `oracle_gap_events`. |
| `correct_models_when_soft_fails` | Per-model occurrence counts on oracle-gap events. |
| `soft_failure_analysis` | Rule impacts, favorable conditional hypotheses, weighted presets, and recoverable soft-error counts. |

### `events.jsonl`

This is JSON Lines rather than one large JSON array. Each non-empty line is an
independent JSON object representing one test event. It is suitable for
streaming and command-line processing.

Example inspection:

```bash
# First event, formatted with jq
head -n 1 results/sepsis-voting-investigation/events.jsonl | jq

# Count recoverable soft-vote failures
jq -s '[.[] | select(.oracle_gap)] | length' \
  results/sepsis-voting-investigation/events.jsonl

# See which models were correct on those failures
jq -r 'select(.oracle_gap) | .correct_models[]' \
  results/sepsis-voting-investigation/events.jsonl | sort | uniq -c | sort -nr
```

File size grows approximately with:

```text
test events × (constituent models + hypotheses)
```

Large datasets and wide model/rule grids can therefore produce substantial
event files and slower initial dashboard loading.

Result directories created before the soft-failure analysis was introduced
remain loadable. For those older files, the dashboard discovers conditions
from the saved test events and labels them **test-derived exploratory**. That
fallback is useful for exploration but is not leakage-safe evidence. Rerun the
investigation into the result directory to obtain calibration-learned
conditions and persist `condition_matches` in `events.jsonl`.

## Dashboard guide

### Header and summary strip

The header identifies the dataset, event count, sequence count, and loaded
result directory. The summary strip shows soft accuracy, oracle accuracy, the
best selected rule/scenario accuracy, and the fraction of the oracle gap that
candidate recovered.

### Overview tab

The accuracy chart compares:

- soft voting;
- cheating voting;
- the eight highest-ranked hypotheses.

“Gap by model agreement” groups all test events by `agreement_count` and plots:

```text
oracle-gap events in group / all test events in group
```

Hovering also shows the number of events in the group. Low-count groups should
not be interpreted as stable evidence.

“Gap by sequence position” divides `relative_position` into ten percentage
buckets and plots the oracle-gap rate within each bucket.

### Recoverable soft failures tab

This table contains only `oracle_gap == true` events: soft voting was wrong but
the oracle could recover the event.

Controls allow filtering by:

- a model that must appear in `correct_models`;
- minimum `agreement_count`.

The table itself supports native sorting and text filtering. Use it to find
recurring prefixes, suffix patterns, positions, and correct-model combinations.

### Sequence explorer tab

Select a case ID to display aligned prediction lanes for:

- actual activity;
- soft voting;
- cheating voting;
- every constituent model.

Green circles are correct predictions. Red crosses are wrong predictions. Text
above each point is the predicted activity. Hovering shows the actual activity,
agreement count, and preceding prefix.

This view is intended to reveal temporal behavior such as:

- a model becoming reliable only after a particular prefix;
- short N-grams working early and longer N-grams working later;
- consensus becoming misleading after a branch or repeated pattern;
- clusters of recoverable failures within one case.

### Hypotheses tab

Filter the leaderboard by rule family. The chart and table show test accuracy,
correct-event count, and how frequently each model was selected.

A rule that almost always selects one model may have high accuracy without
having learned useful conditional behavior. Compare its result against “best
calibration accuracy” and inspect `selected_models` before drawing conclusions.

### Rule-set scenarios tab

This tab compares several fixed sets of rule predictions:

- **core consensus ≥ 2/3** combines the strongest agreement, hierarchy,
  confidence/state, and delayed-feedback rules;
- **advanced consensus ≥ 2** combines the higher-capacity selectors;
- **all rules consensus ≥ 3** uses every available candidate rule;
- **core/all rule-set oracle ceilings** report whether any rule in the set was
  correct, with soft voting as fallback.

Deployable consensus scenarios retain soft voting unless one alternative has
the required unique rule agreement. The rule-set oracle scenarios inspect the
true label and are therefore opportunity ceilings only. The table reports
accuracy, gain versus soft voting, the recovered fraction of the full
soft-to-oracle gap, and the exact participating rule names.

For the reference Sepsis run (`data-prop=0.5`, seed `0`, windows `2,3,4`), the
current scenario results are:

| Scenario | Accuracy | Soft-to-oracle gap recovered |
|---|---:|---:|
| Soft voting | 65.07% | 0% |
| Best individual rule (`agreement ≥ 3, else confidence`) | 65.94% | 10.20% |
| Core-rule consensus | 65.68% | 7.14% |
| Core rule-set oracle ceiling | 70.03% | 58.16% |
| All rule-set oracle ceiling | 72.46% | 86.73% |
| Cheating-voting ceiling | 73.59% | 100% |

This distinction is central: the rule portfolio already contains the correct
answer for most of the recoverable gap, but the deployable consensus mechanism
does not yet identify when to trust each rule. The remaining research problem
is therefore primarily gating/selection, not lack of candidate predictions.

### Soft-vote analysis tab

This tab contains four linked views:

1. a diverging bar chart for each built-in rule, with recoveries above zero and
   harms below zero;
2. the favorable-condition leaderboard, including calibration support/gain
   and held-out recoveries, harms, net impact, error recall, and decisive
   precision;
3. controls for selecting conditions and a weighting method, followed by the
   combined selector's accuracy impact and selected-model distribution;
4. one row per wrong soft-voting prediction, showing which rules and
   conditions applied and which models were actually correct.

Start with high positive `net_correct`, then inspect support, harms, and
whether the effect repeats across several sequences. A high conditional delta
on a tiny condition is weaker evidence than a slightly smaller effect with
substantial calibration and test support.

## How to interpret findings

Recommended investigation order:

1. Measure the soft-to-oracle gap. If it is very small, model selection has
   limited upside for that model family.
2. Compute or inspect the recoverable fraction of soft failures. This separates
   selection errors from errors shared by every model.
3. Use the recoverable-soft-failure table to identify which models recover errors and under
   which prefixes, positions, or agreement levels.
4. Use the sequence explorer to determine whether the behavior is temporally
   coherent or isolated noise.
5. Compare simple calibrated rules against soft voting and the global-best
   model.
6. Prefer improvements that repeat across seeds and datasets, not a single
   split.
7. Only then add more flexible rules or learned gating models.

Run several seeds before treating a rule as reliable:

```bash
for seed in 0 1 2 3 4; do
  python -m logicsponge.processmining.voting_investigation run \
    --data Sepsis_Cases --seed "$seed" \
    --output "results/sepsis-voting-seed-$seed"
done
```

The current module does not aggregate seeds automatically; each result
directory represents one split.

## Adding a hypothesis

Subclass `DecisionRule`. `fit` may use labeled calibration rows. `select` must
choose a model using only information that would be available before revealing
the current test label.

```python
from typing import Any

from logicsponge.processmining.voting_investigation import DecisionRule


class PreferShortNGramEarly(DecisionRule):
    name = "ngram 2 before position 5, then ngram 4"
    family = "position"

    def select(self, row: dict[str, Any]) -> str:
        return "ngram_2" if row["position"] < 5 else "ngram_4"
```

Evaluate it with other rules:

```python
summaries = evaluate_hypotheses(
    calibration_rows,
    test_rows,
    rules=[PreferShortNGramEarly()],
)
```

Useful pre-label inputs available to `select` include:

- `position`, `relative_position`, prefix, and suffix patterns;
- each model's state, visits, confidence, prediction, and complexity;
- agreement count, consensus prediction, and distinct prediction count;
- calibration statistics stored by the rule during `fit`.

Do **not** inspect these fields inside `select`:

- `actual`;
- any model's `correct` field;
- `correct_models`, `correct_model_count`, or `oracle_model`;
- `soft_correct`, `oracle_correct`, or `oracle_gap`.

Using them would turn the candidate rule itself into another cheating rule and
invalidate its reported test accuracy.

To change the default grid, edit `default_hypotheses()` in
`voting_investigation.py`. To use a different constituent model family, provide
custom `ModelSpec` instances when constructing `VotingInvestigator`.

## Important limitations

- The oracle is defined over constituent top-1 predictions, not top-k sets or
  arbitrary probability combinations.
- The experiment reports accuracy, not likelihood or perplexity. Oracle
  likelihood would not represent a deployable probabilistic predictor.
- The dashboard is descriptive. A visually convincing pattern is not proof
  that a rule generalizes.
- The built-in pattern and position rules use discrete exact groups. They do
  not smooth across similar patterns or neighboring positions.
- Calibration groups with the minimum allowed support can still be noisy.
- Confidence values are compared directly even though constituent models may
  not be calibrated to the same probability scale.
- Appended stop events can materially influence accuracy, especially on short
  cases.
- `state_visits=0` can mean either zero visits or that a model does not expose a
  compatible `total_visits` field.
- `--data-prop` keeps an initial case prefix before shuffling; it is not a
  uniform subsample of the complete dataset.
- The standalone defaults may differ from the current constituent set used by
  `predict_batch.py`. Pass matching `--windows` values when comparing runs.

## Validation and troubleshooting

Run the relevant tests:

```bash
pytest -q tests/test_voting_investigation.py tests/test_cheating_miner.py
```

Run lint and formatting checks:

```bash
ruff check \
  logicsponge/processmining/voting_investigation.py \
  logicsponge/processmining/voting_dashboard.py \
  tests/test_voting_investigation.py

ruff format --check \
  logicsponge/processmining/voting_investigation.py \
  logicsponge/processmining/voting_dashboard.py \
  tests/test_voting_investigation.py
```

### Dashboard does not start

Confirm that the chosen result directory contains both required files and that
Dash is installed in the active environment:

```bash
ls results/sepsis-voting-investigation/{summary.json,events.jsonl}
python -c "import dash; print(dash.__version__)"
```

### Port already in use

Choose another port:

```bash
python -m logicsponge.processmining.voting_investigation dashboard \
  results/sepsis-voting-investigation --port 8060
```

### Matplotlib cache warning

Some imports initialize Matplotlib. If the default cache directory is not
writable, set a temporary cache directory:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
python -m logicsponge.processmining.voting_investigation run \
  --data Sepsis_Cases --data-prop 0.1
```

### Empty or unexpectedly small splits

Increase `--data-prop`. Very small retained datasets can round to an empty
calibration or test case split, making hypothesis results uninformative even if
the command completes.
