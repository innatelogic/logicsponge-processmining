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

Run the requested Sepsis investigation:

```bash
python -m logicsponge.processmining.voting_investigation run \
  --data Sepsis_Cases \
  --data-prop 0.9 \
  --windows 2,3,4,5,6 \
  --seed 0
```

Without `--output`, this is saved in the stable dataset-named directory
`results/voting-investigation/Sepsis_Cases/`.

Open the saved results in the dashboard:

```bash
python -m logicsponge.processmining.voting_investigation dashboard \
  results/voting-investigation/Sepsis_Cases
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
| `--data` | `Sepsis_Cases` | Dataset name resolved through the repository's dataset utilities. Choose among: Sepsis_Cases, Helpdesk, BPI_Challenge_2012, BPI_Challenge_2013, BPI_Challenge_2014, BPI_Challenge_2017, BPI_Challenge_2018, BPI_Challenge_2019|
| `--data-prop` | `1.0` | Fraction of grouped cases retained before train/calibration/test shuffling. Must be in `(0, 1]`. |
| `--windows` | `2,3,4` | Comma-separated N-gram window lengths of at least 2. Bag is always included; smaller values are rejected so `streak / (window_size - 1)` is defined. |
| `--seed` | `0` | Seed used for deterministic case-level split shuffling. |
| `--output` | `results/voting-investigation/DATASET/` | Optional exact directory receiving `summary.json` and `events.jsonl`. |
| `--dashboard` | disabled | Start the dashboard after analysis finishes. |
| `--port` | `8050` | Dashboard port when `--dashboard` is used. |

If `--output` is omitted, results are stored by dataset name under:

```text
results/voting-investigation/<DatasetName>/
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

### Preparing several datasets

The `benchmark` command runs datasets sequentially and always stores each one
under its own name:

```bash
python -m logicsponge.processmining.voting_investigation benchmark \
  --data Sepsis_Cases Helpdesk BPI_Challenge_2012 BPI_Challenge_2013 \
  --data-prop 0.9 \
  --windows 2,3,4,5,6 \
  --seed 0 \
  --output-root results/voting-investigation \
  --skip-existing
```

`--data` accepts space-separated names or comma-separated groups. The default
list contains Sepsis, Helpdesk, and BPI Challenges 2012, 2013, 2014, 2017,
2018, and 2019. `--skip-existing` avoids replacing a dataset that already has a
`summary.json`. The command is orchestration only: it does not change the
train/calibration/test protocol used by an individual run.

### Opening existing results

```text
python -m logicsponge.processmining.voting_investigation dashboard RESULTS [--port PORT]
```

`RESULTS` can be either:

- one dataset directory containing `summary.json` and `events.jsonl`; or
- a root containing dataset-named child directories, each with a
  `summary.json`.

```bash
python -m logicsponge.processmining.voting_investigation dashboard \
  results/voting-investigation --port 8060
```

Open `http://127.0.0.1:8060` in that case.

## Constituent models

The default model order is significant and deterministic:

1. `bag`;
2. `ngram_2`;
3. `ngram_3`;
4. `ngram_4`.

The N-gram entries change according to `--windows`. For example,
`--windows 2,4,8` produces Bag followed by `ngram_2`, `ngram_4`, and
`ngram_8`. FPT is not part of the current default family.

Every constituent is trained once on the same ordered training-event stream.
Soft voting and the oracle baseline are computed from the same trained
constituent predictions. This avoids differences caused by training separate
copies of the models.

The `complexity` value recorded for a model is:

- `0` for Bag;
- the window length for an N-gram.

It is recorded as an input for generalization/complexity hypotheses. The
state-evidence, structural-regime, and branching rules use it to determine
whether a model's context order is mature at the current sequence position.
Each saved model row also records `model_type` and `window_size`. The streak
rule uses these structural fields to identify N-grams and their orders without
depending on names such as `ngram_6`. Older event files are supported through a
name-based compatibility fallback during dashboard migration.

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

### 5. Calibration, delayed feedback, and nested selection

Rules are fitted using calibration cases, never the final test cases. A rule may
use the previous event's observed activity only after that event has completed;
it cannot use the label of the event it is currently predicting. This permits
stateful rules such as delayed-feedback and transient recovery while preserving
the online prediction order.

Most calibrated overrides compare a candidate with soft voting on the same
calibration rows. Their basic quantity is the paired gain:

```text
gain = 1(candidate prediction == actual) - 1(soft prediction == actual)
```

They require adequate support and a positive conservative lower confidence
bound on mean gain. If either condition is absent, they retain soft voting.
This is deliberately stricter than choosing the largest raw calibration
accuracy: a small, lucky context is not allowed to override the baseline.

The transient-generalist pool also searches its trigger, horizon, boost, and
decay on a nested split of calibration *cases*. Roughly one third of calibration
sequences (`sequence_index[::3]`) is reserved as selector evidence; the other
two thirds fit the candidate schedules. A schedule is enabled only when its
held-out paired lower bound is at least 0.005; otherwise it is inert.

The final smart integration follows the same principle. Constituent rules are
fit on the two-thirds calibration subset and scored on the reserved sequences.
Only then is the uncertainty-gated family consensus admitted. Afterwards the
individual rules are refit on all calibration rows before evaluating the test
set. The transient-generalist pool is excluded from this family vote because it
already aggregates many related recovery schedules and would otherwise count
that same signal twice.

`calibration_accuracy` in a rule summary is an in-sample diagnostic explaining
the fitted selector; it is not a performance claim. The nested selector rows
and the final test rows are the safeguards against selecting a meta-policy by
its apparent fit.

### 6. Final evaluation

The fitted rules select models for the held-out 15% test split. Test labels are
used only after selection to score the resulting predictions. Built-in rule
selection therefore does not use the current test event's true activity.

Only test-event rows are written to `events.jsonl`. Calibration aggregates and
the number of reserved selector events are represented in `summary.json`.

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
the available soft-to-cheating-baseline opportunity was actually closed.

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
| `empty_prediction_count` | Number of constituents without a current top prediction. |
| `previous_soft_correct` | Whether the previous soft prediction was correct, or `null` at sequence start. |
| `previous_actual` | Previously observed actual activity, or an empty string at sequence start. |
| `previous_correct_models` | Models whose previous top prediction was correct. |
| `previous_wrong_model_count` | Number of models wrong on the previous event. |
| `previous_empty_prediction_count` | Number of empty constituent predictions on the previous event. |

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
| `distribution` | Full normalized activity-probability mapping. |
| `ranked_predictions` | Full distribution ordered from greatest to smallest probability. |
| `entropy`, `normalized_entropy` | Raw and support-normalized distribution entropy. |
| `margin` | Probability difference between the first and second ranked activities. |
| `support` | Number of activities with positive probability. |
| `top3_mass` | Total probability assigned to the three highest-ranked activities. |
| `soft_divergence` | Jensen-Shannon divergence from the aggregated soft distribution. |
| `probability_on_soft_prediction` | Probability this model assigns to the soft-voting top activity. |
| `correct` | Whether `prediction == actual`. |

`confidence` values are model-specific and need not be calibrated across
models. A model producing `0.8` is not necessarily more reliable than another
model producing `0.6`; testing that assumption is the purpose of the
highest-confidence hypothesis.

### Ensemble and oracle fields

| Field | Exact meaning |
|---|---|
| `soft_prediction`, `soft_correct` | Soft-voting result and correctness. |
| `soft_distribution`, `soft_ranked_predictions` | Full normalized soft distribution and its ordered activities. |
| `soft_entropy`, `soft_normalized_entropy` | Raw and normalized soft-distribution entropy. |
| `soft_margin` | Soft top-one minus top-two probability. |
| `soft_support`, `soft_top3_mass` | Positive support size and top-three probability mass. |
| `model_confidence_mean`, `model_confidence_spread` | Mean and range of constituent top probabilities. |
| `model_entropy_mean` | Mean normalized entropy across constituent distributions. |
| `oracle_prediction`, `oracle_correct` | Oracle result with soft fallback, and correctness of that final result. |
| `oracle_model` | First correct constituent in model order, or an empty string. |
| `correct_models` | Names of all correct constituents. |
| `oracle_gap` | Whether this is a recoverable soft-voting failure. |

### Hypothesis fields

| Field | Exact meaning |
|---|---|
| `rule_models` | Mapping from hypothesis name to the selected constituent or direct source such as `soft rank 2`. |
| `rule_predictions` | Mapping from hypothesis name to the activity predicted by its selected model, rank, pool, or meta-rule. |
| `rule_diagnostics` | Per-rule enforcement trace. For transient rules this records whether recovery is active, its age, the minimum-complexity target, and exact target-prediction agreement. |
| `integration_predictions` | Mapping from each active smart-integration name to its final event prediction. |
| `scenario_predictions` | Mapping from each deployable rule-consensus scenario to its event prediction; diagnostic oracle scenarios are excluded. |
| `best_deployable_method`, `best_deployable_prediction` | Explicit final deployment policy and prediction selected for this event. |
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

### Activity-label invariance policy

Suffix-identity, previous-activity, activity-confusion, motif-residual, and
hashed activity-token rules have been removed from the investigation engine.
New default rules must make the same decision after any consistent renaming of
the dataset's activities. Previously saved JSON remains readable by the
dashboard, but reruns use the structural grid.

The default benchmark may still use an algorithm's process-state identifier:
that is a learned state of the fitted miner, not a source-code rule naming a
particular activity. Every exact-state context backs off to state-support,
n-gram-maturity, agreement-topology, and distribution contexts when support is
insufficient.

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
| `description` | Operational definition of the selector. |
| `interpretation` | Process-mining meaning suggested by its routing behavior. |
| `selection_policy` | Metric or learned ordering used to choose a source. This makes explicit that a displayed model name is fitted, not hard-coded. |
| `parameters` | Fixed or fitted settings used by parameterized rules, including an explicit `calibrated` flag for the active adaptive rules. |
| `accuracy` | Correct rule predictions divided by all test events. |
| `correct` | Number of correct test predictions. |
| `total` | Number of test events. |
| `selected_models` | Number of test events assigned to each selected source: a constituent model or a direct derived source such as a distribution rank/pool. Counts sum to `total`. |

## Advanced model-selection rules

The active hypothesis grid contains thirteen compact, interpretable rules. It tests
calibrated distribution blending, confidence/state routing, delayed feedback,
complete-miss recovery, calibrated transient generalist recovery, ranked
alternatives, contrarian disagreement, complexity contrast, three social-choice
rank aggregators, and one fixed short-lived generalist boost. The N-gram
multiplier rules remain available for explicit experiments but are disabled
from the deployed default grid. An event's correctness becomes available only
after that event is predicted.

### Current default rule inventory

The table below is the authoritative default grid produced by
`default_hypotheses()`. “Best” always means computed from the current run's
calibration data or current event metrics; no rule body returns `bag`, `fpt`,
or a particular `ngram_N` name.

| Rule or generated variants | Selection policy |
|---|---|
| Evidence-weighted distribution mixture (support 12) | Blends every model's probability distribution, weighting models by their calibration evidence in the current structural regime; sparse regimes fall back safely. |
| Confidence/state reliability (support 5) | Routes to the constituent with the best smoothed calibration reliability for its confidence and learned-state support bands. |
| Delayed-feedback adaptive (decay 0.94) | Starts from the calibration reliability contexts and updates them after each labeled event, for use on later events in that case. |
| Complete-miss state recovery (support 8) | After every constituent missed the preceding event, uses the state-calibrated minimum-complexity or adaptive candidate only when it has a positive paired gain over soft voting. |
| Calibrated transient generalist pool | Chooses the trigger type, horizon, boost strength, and decay from a nested calibration holdout; weak evidence disables the transient policy entirely. |
| Calibrated soft rank 2 override (support 2) | Uses soft voting's second-ranked activity only in contexts where calibration showed it reliably beats the usual first choice; otherwise retains soft voting. |
| Calibrated lone-dissenter override (support 2) | Keeps soft voting unless exactly one model opposes a consensus of at least two and its supported, uncertainty-adjusted calibration advantage is positive. |
| Calibrated lone-dissenter rank 2 override (support 2) | Uses the dissenter's second activity only after an observable failure-risk signal and calibration evidence that it beats soft voting. |
| Complexity-contrast exception override | When every model gives the soft-vote top activity less than 50% probability, systematically discounts lower-complexity models' top activities to surface a stronger alternative. |
| Borda rank aggregation | Treats models as voters and activities as alternatives; sums tie-aware positional scores over every model probability ranking. |
| Copeland pairwise rank aggregation | Awards an activity one point for each pairwise majority win over another activity and half a point for a tie. |
| Maximin pairwise rank aggregation | Selects the activity whose worst pairwise vote margin against any rival is largest. |
| Transient Bag favoritism after a generalist-correct error | After a soft-voting error that the minimum-complexity model predicted correctly, applies a model-count-scaled multiplier with `1.0` extra weight for every competing constituent, then halves that extra weight on each of the next two events. |

Model identity is necessarily retained as the key under which calibration
statistics are accumulated. This is not a fixed choice: renaming every model
consistently and refitting produces the same source role and prediction. The
test suite checks this invariance across the complete default grid.

### How to read the rules

The rules are probes of different kinds of ensemble complementarity, rather
than ten independent claims that each model has a universal priority.

| Rule family | What it tests about the ensemble | Safe fallback |
|---|---|---|
| Evidence mixture; confidence/state reliability | Whether a model's probability and confidence are trustworthy in this structural regime, rather than globally. | Soft vote or the best supported calibrated evidence. |
| Delayed-feedback adaptive | Whether recently revealed outcomes indicate a temporary shift in which structural view is reliable. Adaptive voting is a routing signal, not an additional constituent model. | Its calibration reliability routing. |
| Complete-miss recovery | Whether an event after a total ensemble miss is a regime boundary where the broadest model or the adaptive-selected model has a supported advantage. | Soft vote unless paired gain is positive. |
| Transient generalist rules | Whether a recent generalist success or failure predicts a short-lived recovery period, and how quickly that signal decays. | No boost when the nested evidence is weak. |
| Soft-rank and lone-dissenter rules | Whether a correlated majority is masking a useful second probability or a specialist's alternative. | Soft leader unless the alternative has supported gain. |
| Complexity contrast | Whether models disagree because broad context smooths away a locally discriminative path. | Soft vote when the conflict condition is absent. |
| Social-choice rank aggregation | Whether rank consensus across model distributions contains signal that probability averaging loses. | Soft probability breaks aggregate-score ties; empty profiles retain soft voting. |

### Social-choice rank aggregation

The theoretical source is Felix Brandt, Vincent Conitzer, and Ulle Endriss,
“[Computational Social Choice](https://eprints.illc.uva.nl/id/eprint/446/1/PP-2012-04.text.pdf),”
ILLC Prepublication Series PP-2012-04 (2012).

The paper makes the model-ensemble analogy unusually explicit: combining ranked
results from several search engines is described as preference aggregation, while
also warning that modern applications may require classical assumptions to be
altered (pp. 7-8). Here:

- a constituent process model is a voter;
- a possible next activity is an alternative;
- the model's descending probability order is its ballot.

Three deterministic, label-invariant candidates follow:

1. **Borda rank aggregation.** Positional scoring gives an alternative credit
   for every lower-ranked alternative, and Borda uses the score vector
   `(m-1, m-2, ..., 0)` (p. 18). Equal model probabilities split positional
   credit, producing a weak-ranking extension instead of inventing an order.
2. **Copeland pairwise rank aggregation.** Copeland awards one point for each
   pairwise majority win and half a point for a pairwise tie (pp. 19-20).
   A model votes for activity `a` over `b` exactly when `P(a) > P(b)`.
3. **Maximin pairwise rank aggregation.** Maximin evaluates an alternative by
   its worst pairwise defeat and prefers the least severe worst case (p. 20).
   The implementation maximizes the minimum signed model-vote margin.

The paper assumes linear ballots in the formal presentation (p. 18), whereas
process-model distributions routinely contain ties and omit activities.
Accordingly, equal probabilities abstain in pairwise contests, and two omitted
activities are never ordered. All three rules are symmetric in model identity
and activity identity until a tied aggregate score must be made resolute. That
final engineering tie is broken by soft-voting probability and then by a stable
activity label; it is documented in each result's `parameters`.

These are uncalibrated candidate rules: they never read calibration or test
labels to choose an activity. Their calibration accuracy is still reported as a
diagnostic comparison, but no parameter is selected from it. The principal
limitation is correlated voters: neighbouring N-gram orders are nested views,
not independent opinions. Therefore a positive result is evidence for useful
rank aggregation in this ensemble, not a claim that the classical voting-rule
axioms hold literally for process predictors.

N-grams are deliberately not treated as independent votes. Their contexts are
nested suffixes, so neighbouring orders often make the same error for the same
reason. Bag is a broad, low-complexity view of activity frequency; it can remain
useful at sparse or boundary states where longer contexts have insufficient
support, but it is not globally strongest. Medium N-grams can exploit common
local pathways; high-order N-grams can surface rare, specific continuations but
also become sparse. Soft voting reduces variance across these related views,
yet can bury a correct minority prediction when several correlated N-grams
agree. The family-consensus integration requires agreement from distinct rule
families, not merely several variants of the same model signal, precisely to
avoid mistaking correlation for independent evidence.

### Sepsis interpretation and the WW trace

On the saved full Sepsis run with windows 2–6 and seed 0, soft voting scored
62.70%, cheating/oracle voting 75.32%, and the uncertainty-gated family
consensus 64.24% (39 net additional correct events, or 12.23% of the oracle
gap). These are one split's held-out observations, not a claim of stable
clinical performance; repeat seeds and, ideally, a temporally separate cohort
are required before deployment.

The constituent results support the complementarity hypothesis. The best
single constituent in that split was `ngram_4` (61.00%), while Bag was only
56.96%; nevertheless, oracle voting shows that some of Bag's low-frequency,
broad-state predictions repair errors from the more accurate models. Adaptive
voting alone was 60.36%, so its value is not as a global replacement for soft
voting. Its current selected constituent can still be useful in the narrowly
defined post-complete-miss state, which is why the complete-miss rule compares
both relative candidates on paired calibration gains instead of promoting
either one universally.

Case `WW` illustrates the distinction. Among four oracle-gap errors in the
recorded trace, the minimum-complexity model correctly supplied `LacticAcid`
where soft voting chose `CRP`; adaptive routing supplied `CRP` where soft chose
`LacticAcid`; and the minimum-complexity model supplied `Leucocytes` and
`Return ER` where soft chose `Admission NC` and `__stop__`, respectively. The
complete-miss rule corrects the latter three eligible post-miss cases in that
trace. It deliberately does **not** force the first: its matching calibration
state had negative paired evidence (−2 over seven observations). This is the
central calibration principle: an appealing single-case correction becomes a
rule only when the same observable situation has a supported positive gain.

### Archived exploratory rules

`archived_hypotheses()` contains every selector outside this compact grid. This
includes both former direct-routing transient variants, all three Bag
calibrations, global and state-accuracy routing, highest-confidence routing,
agreement routing, alternative position/state variants, hierarchical and contextual
selectors, the relative calibrated candidate router, other calibrated ranks
and probability pools, and the previous-error correct-set selector.

They are not fitted, evaluated, written to results, or displayed by default.
They can still be passed explicitly to `evaluate_hypotheses()` for a focused
experiment. Archiving reduces benchmark noise without deleting implementations
or making older experiments irreproducible.

The dashboard also filters these names while loading older result files. Their
saved JSON is left untouched, but archived rules are omitted from accuracy
charts, impact tables, rule scenarios, and event-level rule annotations.

### Hierarchical reliability

`hierarchical reliability (support 3)` estimates each model's probability of
being correct using a fine-to-coarse context cascade:

1. model + current state + state support + soft rank + canonical agreement topology;
2. model + support/confidence/entropy ranks + n-gram maturity + topology;
3. model + relative evidence ranks + maturity;
4. model + state-support band + soft rank + agreement;
5. model + support band + maturity;
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

### Confidence, state, and structural disagreement reliability

`confidence/state reliability` calibrates correctness by discretized confidence
(five bins), state-visit band (`0`, `1–2`, `3–9`, `10+`), and agreement. It is
intended to test whether confidence means the same thing for different models
and states.

`disagreement profile reliability` canonicalizes predictions into equivalence
classes. For example, model outputs `A, B, B, C` become topology `0, 1, 1, 2`.
It combines that topology with state support, soft-distribution rank, and
relative divergence. The learned behavior therefore transfers across activity
renamings and focuses on how models disagree.

### Transient Bag favoritism

The first variant, `transient Bag favoritism after generalist-correct error (3
steps)`, activates only when the preceding soft vote was wrong and the
minimum-complexity model was among the models that predicted that event
correctly. The target is found structurally:

```text
target = model with minimum complexity
```

With the standard model specifications this is Bag (`complexity = 0`), but the
rule never checks the name `bag`. A stable model-index tie break is used for a
custom ensemble with several minimum-complexity models.

For recovery age `a = 0, 1, 2`, the target receives:

```text
multiplier(a) = 1 + 3.0 × (model_count - 1) × 0.5^a
```

For example, a three-model ensemble receives `7`, `4`, `2.5`, then `1`;
a five-model ensemble receives `13`, `7`, `4`, then `1`. A new eligible error
restarts the schedule. The rule merges every constituent's full distribution;
it does not directly route to Bag's top prediction. All non-target models stay
at weight `1`, so favoritism cannot suppress or negatively weight them.

The former `after generalist-wrong error` variant is no longer part of the
deployed experiment grid. The retained Bag rule tests whether broad process
memory remains the safest immediate
recovery bias even when it did not identify the error event itself.

The previous event's truth is consumed only after its prediction. Therefore it
can affect the next decision but never the already-scored decision. The
process interpretation is a short recovery from over-specialization. The first
variant represents demonstrated generalist competence; the second is an
explicit fallback hypothesis after a joint ensemble/generalist error.

Event diagnostics contain `active`, `recovery_age`, `target_model`,
`target_complexity`, `multiplier`, and the model-local `model_multipliers` map.
The fitted parameters state `calibrated: false` and persist the complete
multiplier schedule.

The former direct-routing variants and the three recovery calibration methods
are retained by `archived_hypotheses()` for reproducibility. They are neither
evaluated nor displayed in a normal run.

### N-gram correctness-streak multiplier

For every model structurally marked as an N-gram of window size `x`, the rule
tracks consecutive correct predictions within the current case. A multiplier
is applied only when `previous_soft_correct == false`; otherwise the latent
streak remains visible but the multiplier stays `1`. The state is
updated only by `observe()` after the current prediction is scored, so the
current true activity can never affect its own selection. A wrong N-gram
prediction resets only that N-gram's streak to zero; a new case resets all
streaks.

The requested pre-maturity ratio is first computed literally, then mapped
through a normalized exponential curve (`rate = 4`) so that the greatest
streaks receive much more of the boost:

```text
linear_ratio = (
    min(1, max(0, streak / (window_size - 1)))
    if streak < window_size + 1
    else 0
)

boost_ratio = (exp(4 * linear_ratio) - 1) / (exp(4) - 1)
```

The linear precursor reaches `1` at streak `x - 1` and remains capped at `1`
through streak `x`; at streak `x + 1` it drops to `0` and remains zero for
longer streaks. The exponential mapping is convex: it stays near zero for
short streaks and rises sharply toward `1` for the greatest streaks. This makes
the boost a temporary pre-maturity intervention. Eligibility and the current
numerical ratio are preserved separately in event diagnostics through the
`eligible` and `boost_ratios` mappings. Window sizes below 2 are rejected to
keep the formula well-defined.

After a soft-voting error, each N-gram's full probability distribution receives
a bounded exponential multiplier. With
`progress = min(1, max(0, streak / (window_size - 1)))`, it is

```text
distribution_multiplier = (
    0.1 × 10 ** (2 × progress)   if progress <= 0.5
    2 ** (2 × progress - 1)      otherwise
)
```

This gives base weight `0.1` at zero streak, `1` at half-window progress, and
`2` near the window size. It is then multiplied by
`sqrt(window_size / minimum_ngram_window_size)`, so two N-grams that both
reach their window size give the larger-window N-gram the higher multiplier.
Regardless of the soft-error gate, an N-gram whose immediately previous
prediction was wrong receives multiplier `0` and contributes no probability
mass for that event. The scale remains fixed rather than calibrated.

### Largest N-gram disagreement multiplier

When at least two N-grams make different predictions, this fixed tie-breaker
multiplies the largest order by `1.1`. It activates only when the ratio of the
largest to smallest current streak multiplier is at most `1.1`. Therefore it
cannot override a material streak-based preference; it only resolves close
streak-weight ties in favor of the richer context.

The rule records per event:

- delayed streak for every N-gram;
- `x - 1` eligibility;
- boost ratio and resulting multiplier;
- boosted model names;
- the positive `model_multipliers` contributed to the combined stack.

No collective gate or priority rule is active. After a soft error, if two
N-grams have ratios `1` and `0.75`, both multipliers are applied in the same
merge. One of the two complementary transient Bag variants is active at that
event too, and its Bag multiplier is applied in the same merge.

### State-evidence topology

`state-evidence topology (support 5)` is the most process-state-oriented
selector. Its context cascade combines:

- the joint current state vector across miners;
- each state's occurrence band (`0`, `1–2`, `3–9`, `10+`);
- the evidence curve across increasing n-gram orders;
- whether the current prefix is long enough to make each order mature;
- relative ranks of support, confidence, entropy, and soft divergence;
- canonical model-agreement topology and each model's soft rank.

Interpretation: selection of a longer-order n-gram in a well-visited state is
evidence for a stable local subprocess or routing pattern. Selection of Bag,
FPT, or a shorter n-gram while longer states are sparse indicates branching,
novel context, or insufficient repetition for the specific model.

### Evidence-weighted distribution mixture

`evidence-weighted distribution mixture (support 12)` scores the probability
that each model assigned to the observed calibration transition with a bounded
logarithmic score. Scores are estimated inside structural regimes and shrunk
toward each model's global score. At prediction time they become positive
weights for merging the complete model distributions; state visit counts add a
small evidence multiplier but cannot override calibrated quality.

This is a weighted ponderation, not a top-1 model switch. Broad models receiving
high weight near uncertain branches suggests useful generalization; a deep
n-gram receiving high weight in supported regimes suggests a repeatable local
path.

### Structural branching ensemble

`structural branching ensemble (support 8)` reserves whole calibration
sequences as an internal gate-validation split. Within each activity-invariant
regime it compares five experts:

1. state-evidence topology selection;
2. evidence-weighted distribution merging;
3. median pooling;
4. trimmed pooling;
5. product pooling.

It installs a branch only when the best expert beats soft voting by more than a
sampling-uncertainty penalty; otherwise it keeps soft voting. Experts are then
refitted on all calibration events. A state-selection branch identifies a
recurring subprocess, robust pooling identifies noisy/outlier models, product
pooling identifies cross-model corroboration, and fallback identifies weak or
ambiguous evidence.

### Second/third-choice and model-rank rules

`calibrated soft rank 2/3 override` can predict the second- or third-ranked
activity in the aggregated soft distribution. It groups calibration events by
well-defined contexts combining:

- whether the previous soft prediction was correct;
- previous wrong/empty model counts and current empty model count;
- agreement count;
- soft top-two margin and normalized entropy;
- spread between constituent top probabilities.
- canonical prediction topology, state-evidence curve, soft ranks, and n-gram
  maturity.

A rank is enabled for a context only when it has at least eight calibration
events and beats the ordinary soft prediction there. Otherwise the rule keeps
soft voting. These rules may predict an activity that is not any constituent's
top-1 choice.

`calibrated model rank 2/3 switch` instead ranks constituent models by global
calibration accuracy, then learns contexts where the second- or third-ranked
model beats soft voting. It tests the hypothesis that a globally weaker model
is a useful conditional specialist.

### Calibrated lone-dissenter override

`calibrated lone-dissenter override (support 2)` is a deliberately narrow
contrarian rule. It considers an override only when exactly one model predicts
an activity different from a non-empty consensus of at least two models. It
then searches, from a detailed to a broader structural context, for calibration
evidence about that same model acting as the lone dissenter. Contexts use the
model's learned state and visit band, confidence, top-two margin, divergence
from soft voting, consensus size, soft-vote margin, and canonical disagreement
topology. No activity label is used as a feature.

For a context with `n` calibration events, each event contributes the paired
difference:

```text
delta = 1(model is correct) - 1(soft vote is correct)
```

The observed mean is shrunk toward the model's overall paired advantage, using
a prior weight of four events. An override is installed only if the lower bound
below is strictly positive:

```text
lower_bound = shrunken_mean(delta) - 0.5 × standard_error(delta)
```

The default requires at least six comparable calibration events. A missing,
sparse, tied, negative, or statistically uncertain context always retains soft
voting. Event diagnostics record the candidate model, support, raw and
shrunken advantage, lower bound, and whether the override was active.

### Lone-dissenter second-choice override

`calibrated lone-dissenter rank 2 override (support 2)` tests the dissenter's
second-ranked activity, rather than its top activity. It is narrower than the
ordinary lone-dissenter rule: the second activity must be distinct from soft
voting, and at least one observable risk signal must be present:

- **Previous all-model failure:** on the preceding event every constituent
  top prediction was wrong. This is delayed feedback, never knowledge of the
  current event's outcome.
- **Low representation:** the dissenter's current learned state has been seen
  at most twice in training, so the ordinary consensus may be extrapolating.
- **High variability:** the soft distribution is diffuse or nearly tied, or
  constituent confidence is widely spread.

Within those risk regimes, calibration groups examples by the dissenter's
learned state, sequence stage, support band, confidence and margin, soft-vote
margin, disagreement topology, and the active risk signals. This is the
case-dependent component: it learns which observable case signatures have
actually made the rank-two activity useful. It needs four comparable events
and the same positive, shrunken lower-bound test as the ordinary contrarian
rule; otherwise it retains soft voting. The diagnostic records every active
risk signal and the complete calibration decision.

### Complexity-contrast exception override

`complexity-contrast exception override` searches for exceptions when the
ensemble's apparent top choice is not individually predominant: every model
must assign the soft-vote top activity **less than 50%** probability. When that
condition holds, the transformation is systematic; it does not wait for a
calibration gate. It has no hard “lowest 20%” cutoff: every model contributes
to both sides of a continuous complexity contrast. Higher-complexity models
receive more positive weight; lower-complexity models receive more subtractive
weight. The rule declines to act when every model has the same complexity,
because there is then no complexity signal to contrast.

For every activity in the richer pool's top three, other than the consensus
activity, it computes:

```text
contrast(activity) = complexity-weighted probability(activity)
                   - inverse-complexity-weighted top-prediction reward(activity)
```

The negative term applies only to each model's own top activity. Therefore a
low-complexity model discounts the broad rule it actively proposes but does not
erase every alternative it assigns a small probability to. Repeated low-model
top predictions add their negative rewards together.

The candidate must retain at least 8 percentage points after subtraction, be
at least half as probable as the complexity-weighted consensus activity, exceed
the consensus activity's contrast score, and appear in the top three of at
least 60% of all constituent models (with a minimum of two models). This
prevents a single high-order model's speculative tail from causing a switch.

If the probability, contrast, or top-three-support safeguards fail, the rule
retains soft voting. Diagnostics preserve each model's positive and subtractive
complexity weights, candidate probabilities, contrast, support, and the
systematic applicability decision.

No rule uses whether the *current* predictions are correct. That information
requires the current label and is forbidden. Delayed previous correctness is
valid only because it has already been observed before the next prediction.

### Distribution-shape specialist

`distribution-shape reliability` scores each model from calibrated versions of:

- normalized entropy of its full distribution;
- its top-one versus top-two probability margin;
- Jensen-Shannon divergence from the aggregated soft distribution;
- whether its top activity agrees with soft voting;
- the current number of empty constituent predictions.

This distinguishes, for example, a confident contrarian model from a diffuse
contrarian model instead of treating every disagreement equally.

### Probability-distribution pools

Three direct ensemble rules use all probability values rather than only model
top predictions:

- `median probability pool` takes the per-activity median across models;
- `trimmed probability pool` removes the smallest and largest per-activity
  values, then averages the remaining models;
- `product probability pool` uses a geometric mean, rewarding activities that
  receive support across several distributions.

`calibrated probability pool` learns which of soft, median, trimmed, or product
pooling works best for each state-evidence, agreement-topology, sequence-stage,
and distribution regime. It retains soft voting when calibration does not show
a positive gain.

### Data-driven conditional hypotheses

The favorable-condition analysis now discovers only structural conditions:
process state, state-support band, support/confidence/entropy rank, n-gram
maturity, canonical prediction topology, agreement, position, delayed previous
correctness, and distribution uncertainty. It no longer proposes conditions
such as “after activity X, prefer model Y.”

### Nearest calibration behavior

`nearest calibration behavior (k 32)` is a non-parametric sequence matcher. It
finds calibration events close to the current event using relative position,
agreement topology, state-support bands, soft ranks, entropy, divergence, and
confidence.
For each constituent, correctness is inverse-distance weighted across the
nearest neighbors and shrunk toward its global rate. No test labels are used
to define the distance or the selected model.

### Stacked rule portfolio

`stacked rule portfolio` is a hierarchical composition of selectors. It first
fits a portfolio of confidence, state, agreement, topology, and
distribution-mixture rules on three quarters of calibration events. The
remaining quarter is an internal validation set used to learn which rule is
most reliable for each structural regime, with hierarchical backoff.
Constituent rules are finally refit on all calibration events before
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

### Smart rule-integration methods

The normal investigation now reserves whole calibration cases for a nested
meta-policy.  The constituent rules fit on the remaining calibration cases;
their predictions on the reserved cases choose the uncertainty-gated family
consensus.  On test events it changes soft voting only when the soft margin is
within the calibrated threshold and at least the calibrated number of
independent rule families agree on the same alternative.  This prevents the
meta-policy from choosing its gate on the same cases used to fit its experts.
It additionally requires a positive paired lower confidence bound of at least
0.5 percentage points on the reserved cases; otherwise it deterministically
falls back to soft voting.

The sole active integration is `uncertainty-gated family consensus`. The
current defaults use a maximum soft margin of `0.30`, require two rule
families, and require a selector lower bound of at least `0.005`. It is an
interleaving layer, not a new predictor: it asks whether independently derived
structural rules agree on an alternative precisely when the soft vote is
uncertain. If selector evidence is insufficient, it is inert and returns soft
voting. `best_rule_result` records this defined deployment policy rather than
retrospectively choosing a test-set winner.

Earlier multiplier-stack integrations remain archived ablations. They are not
persisted as active integrations in a normal run.

#### Archived integration methods

Former calibration-best, adaptive-recovery routing, specialist overlay,
family-voting, contextual routing, consensus, nearest-behavior, and Hedge
methods remain available through `evaluate_archived_rule_integrations()` for
explicit ablation studies. They are excluded from normal result files and the
dashboard because their selection gates and precedence rules obscure the two
model-local effects being tested. Older result files remain readable, but the
current dashboard hides those archived integration rows.

### Interpreting the expanded grid

The extra rules are a search space, not a guarantee of improvement. Existing
saved result files retain the rule grid with which they were produced; they are
not silently rewritten. Rerun an investigation to benchmark the new structural
rules and compare their recoveries, harms, and gap closed across datasets and
seeds. Old numeric examples are intentionally omitted here because they refer
to the retired activity-sensitive default grid.

Direct rank and pooling rules expand the prediction space beyond constituent
top-1 activities. Consequently, a rule-set oracle can exceed the
cheating-voting baseline; cheating voting remains the requested top-1 model
selection baseline, not an absolute ceiling for these expanded rules.

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

### Data-mined favorable conditions

These conditions are **not members of the decision-rule grid**. They are an
additional descriptive mining pass over pre-label event features. Consequently,
they are unrelated to `default_hypotheses()` and `archived_hypotheses()` and
should not be read as archived rules returning through another dashboard view.

The engine enumerates categorical conditions using only information available
at prediction time:

- consensus strength (`low/no majority`, `majority`, or `unanimous`), exact
  agreement count, and prediction diversity;
- early, middle, or late sequence stage and position buckets of width 2 and 5;
- canonical prediction topology, which preserves which models agree without
  preserving the activity labels;
- each model's current process state, state-support band, and relative support
  rank;
- the evidence curve and maturity profile across configured n-gram orders;
- the set of models forming the consensus;
- the set of models disagreeing with soft voting;
- whether each particular model agrees or disagrees with soft voting and with
  the top-1 consensus;
- previous soft correctness and previous wrong or empty model counts;
- current empty prediction count, soft margin/entropy bins, constituent
  confidence spread, and per-model confidence/entropy/soft-rank order.

For each condition supported by at least 10 calibration events, the engine
calculates every constituent's calibration accuracy and recommends the best
one. The persisted wording is “locally most accurate calibrated model,” followed
by the model selected in that particular run. That displayed model is an output
of `fit`, never a model encoded in the condition implementation. A condition is
retained only when that constituent beats soft voting on the same calibration
events:

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

The dashboard renders the persisted fields in a shorter sentence-like form:
“when feature” + “has value” → “learned choice.” It shows the calibration event
count and advantage beside held-out recoveries, harms, net improvement, and
decisive precision. Less essential intermediate metrics remain available in
`summary.json` rather than widening the table.

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
results/voting-investigation/
├── Sepsis_Cases/
│   ├── summary.json
│   └── events.jsonl
├── Helpdesk/
│   ├── summary.json
│   └── events.jsonl
└── BPI_Challenge_2012/
    ├── summary.json
    └── events.jsonl
```

The stable dataset directory is replaced when the same dataset is run again.
Use an explicit `--output` directory when retaining several seeds or parameter
sets for one dataset.

The dashboard reads these files directly; it does not retrain models.
It loads them when the dashboard process starts. After regenerating a result
directory, stop and restart an already-running dashboard process to see the new
summary fields and integration comparisons.

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
| `run_config` | Data proportion, N-gram windows, and split seed used for the run. |
| `strategies` | Soft-voting, adaptive-voting, and cheating/oracle accuracies. |
| `per_model` | Constituent accuracy and correct-event count. |
| `hypotheses` | Rules sorted by test accuracy, including description, process interpretation, and model-independent selection policy. |
| `rule_scenarios` | Fixed rule-set consensus scenarios and diagnostic rule-set ceilings. |
| `rule_integrations` | Active uncertainty-gated family consensus, with accuracy, impact, and selector-gate parameters. |
| `selector_calibration_events` | Events belonging to complete calibration cases reserved for the nested uncertainty-gated family consensus. |
| `best_rule_result` | Highest-accuracy deployable individual rule, rule-set scenario, or smart integration. |
| `independent_boost_contract` | Legacy schema-compatible integration-audit field; it is not the definition of the current consensus policy. |
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
head -n 1 results/voting-investigation/Sepsis_Cases/events.jsonl | jq

# Count recoverable soft-vote failures
jq -s '[.[] | select(.oracle_gap)] | length' \
  results/voting-investigation/Sepsis_Cases/events.jsonl

# See which models were correct on those failures
jq -r 'select(.oracle_gap) | .correct_models[]' \
  results/voting-investigation/Sepsis_Cases/events.jsonl | sort | uniq -c | sort -nr
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

The dashboard also recognizes files created before the current independent
stack. At load time it recomputes both transient Bag variants, N-gram streaks, and
the combined positive multipliers from stored model predictions, complexities,
N-gram names/metadata, and delayed previous-event outcomes. This is a
label-safe in-memory compatibility migration because neither active rule needs
calibration. Former fitted integration rows are hidden, and the reconstructed
independent stack becomes the displayed deployable policy. The saved files on
disk are not rewritten.

## Dashboard guide

### Root dashboard and dataset selector

Opening `results/voting-investigation/` starts the cross-dataset dashboard.
The dataset selector at the top controls the **Selected dataset** tab. The
selected view shows that dataset's run configuration, headline scenario chart,
exact result table, smart-integration comparison, recoveries, harms, and fitted
parameters.

The selector also controls **Detailed investigation**. That view restores the
large soft/cheating/gap/best-deployable/gap-closed values and contains nested
subpanels for sequence exploration, per-rule impact and selection policy,
conditional hypotheses, and every soft-voting error. Only the selected
dataset's event file is held in memory.

The **Global comparison** tab is intentionally independent of the selector and
shows all discovered datasets simultaneously:

- grouped soft-voting, best-deployable, and cheating-baseline accuracy;
- the fraction of each soft-to-cheating gap recovered by its best deployable
  method; and
- an exact sortable table with event count, best individual rule, best smart
  integration, enforced deployment policy, accuracy gain, and gap recovery.

Diagnostic oracle scenarios are excluded when choosing `best deployable`.
The independent positive-multiplier stack takes deployment priority even if a
comparison row has higher retrospective held-out accuracy.
Only direct child directories containing a readable `summary.json` are
discovered, which prevents unrelated archived runs from being mixed into the
comparison accidentally.

Legacy custom names such as `results/sepsis-voting-investigation/` still open
as single-dataset dashboards. Passing their parent (for example `results/`)
also discovers them when they are direct children; new runs use the standardized
dataset-name layout by default.

Opening a specific dataset directory instead retains the detailed single-log
dashboard described below.

### Header and summary strip

The header identifies the dataset, event count, sequence count, and loaded
result directory. The summary strip shows soft accuracy, oracle accuracy, the
best individual-rule accuracy, best smart-integration accuracy, and the
fraction of the cheating-baseline gap recovered by the best deployable
candidate.

### Overview tab

The first Overview panel makes the requested benchmark figures directly
visible without navigating to another tab. Its chart and exact-value table
compare soft voting, the best individual rule, the best smart integration,
core consensus, core/all rule-set diagnostic ceilings, and cheating voting.
Hovering shows the selected method, gain over soft, and recovered gap.

The candidate-rule accuracy chart compares:

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
recurring states, evidence profiles, positions, and correct-model combinations.

### Sequence explorer tab

Select a case ID to display aligned prediction lanes for:

- actual activity;
- soft voting;
- cheating voting;
- every constituent model.

Green circles are correct predictions. Red crosses are wrong predictions. Text
above each point is the predicted activity. Hovering shows the actual activity,
agreement count, and preceding prefix.

The timeline contains aligned lanes for the actual activity, soft voting, the
best deployable candidate, cheating voting, and every constituent model. When a
candidate carries `deployment_priority`, “Best deployable” means that enforced
deployment policy; otherwise it is the highest-accuracy non-oracle individual
rule, smart integration, or consensus scenario. Its hover text and the summary
below the chart give the exact method name.

New investigations persist `integration_predictions` and deployable
`scenario_predictions` in `events.jsonl`, alongside `rule_predictions`, so the
winning method can be inspected event by event. For an older result file that
lacks predictions for its aggregate best integration, the explorer displays
the highest-accuracy deployable candidate whose event predictions are actually
available and states that method explicitly below the chart. It never
reconstructs or invents missing predictions from aggregate accuracy.

For new runs, hovering the Best deployable lane additionally shows the active
recovery target, zero-based recovery age, N-gram streak ratios, and the exact
combined model multipliers. A blank target means transient Bag
favoritism is inactive at that event.

This view is intended to reveal temporal behavior such as:

- a model becoming reliable only after a particular prefix;
- short N-grams working early and longer N-grams working later;
- consensus becoming misleading after a branch or repeated pattern;
- clusters of recoverable failures within one case.

### Hypotheses tab

Filter the leaderboard by rule family. The chart and table show test accuracy,
correct-event count, how frequently each model was selected, its selection
policy, and its process interpretation.

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

Saved results created with an older rule grid remain visible, but their numeric
values do not describe the current structural selectors. Rerun the dataset
before comparing a rule named in this document. A rule-set oracle can exceed
the top-1 cheating baseline because direct distribution-rank and pooling rules
can predict an activity that is not any constituent's top-1 prediction.

### Integration methods tab

This tab shows the active uncertainty-gated family consensus against soft
voting and the cheating baseline. Color represents net events gained or lost
relative to soft voting; hovering shows the description, recoveries, harms,
and recovered gap. Inspect the soft-margin gate, minimum-family requirement,
and selector lower bound before interpreting a gain as a robust deployment
improvement.

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
the current test label. Prefer relative policies over literal model names. This
example chooses the most specific currently supported model and breaks ties by
confidence; it works for any configured model family:

```python
from typing import Any

from logicsponge.processmining.voting_investigation import DecisionRule


class MostSpecificSupportedModel(DecisionRule):
    name = "most specific supported model"
    family = "state evidence"
    selection_policy = "greatest complexity with at least five state visits, then greatest confidence"

    def select(self, row: dict[str, Any]) -> str:
        supported = [model for model in row["models"] if model["state_visits"] >= 5]
        candidates = supported or row["models"]
        return max(
            candidates,
            key=lambda model: (model["complexity"], model["confidence"], -model["index"]),
        )["name"]
```

Evaluate it with other rules:

```python
summaries = evaluate_hypotheses(
    calibration_rows,
    test_rows,
    rules=[MostSpecificSupportedModel()],
)
```

Useful pre-label inputs available to `select` include:

- `position`, `relative_position`, prefix, and suffix patterns;
- each model's state, visits, confidence, prediction, and complexity;
- agreement count, consensus prediction, and distinct prediction count;
- calibration statistics stored by the rule during `fit`.

Avoid returning names such as `ngram_6` from a rule body. A model name may be
stored after `fit` only when it is the result of a documented policy—for
example, highest calibration accuracy in a supported state. The built-in grid
is tested under consistent model renaming to enforce this distinction.

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
- Position buckets and exact process-state conditions remain discrete. The
  hierarchical, nearest-behavior, and structural-regime rules provide broader
  backoff or similarity-based alternatives.
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
ls results/voting-investigation/Sepsis_Cases/{summary.json,events.jsonl}
python -c "import dash; print(dash.__version__)"
```

### Port already in use

Choose another port:

```bash
python -m logicsponge.processmining.voting_investigation dashboard \
  results/voting-investigation --port 8060
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
