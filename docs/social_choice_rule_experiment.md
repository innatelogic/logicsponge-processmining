# Social-choice rank aggregation experiment

## Question and theoretical source

Can next-activity prediction improve when constituent model distributions are
treated as ranked ballots rather than averaged probabilities?

The source is Felix Brandt, Vincent Conitzer, and Ulle Endriss,
“[Computational Social Choice](https://eprints.illc.uva.nl/id/eprint/446/1/PP-2012-04.text.pdf),”
ILLC Prepublication Series PP-2012-04 (2012). The paper explicitly describes
combining several search-engine rankings as a preference-aggregation problem
and notes that classical assumptions may need adaptation in computational
applications (pp. 7-8). It defines positional scoring and Borda aggregation on
p. 18, and Copeland and maximin pairwise-majority rules on pp. 19-20.

The implemented mapping is:

| Social-choice concept | Prediction concept |
|---|---|
| voter | constituent Bag or N-gram model |
| alternative | possible next activity |
| ballot | descending next-activity probability order |
| election winner | predicted next activity |

Probability ties create weak rather than linear rankings. Borda splits
positional credit for ties; Copeland and maximin treat equal probabilities as
pairwise abstentions. Aggregate-score ties use soft-voting probability and then
a stable activity label. No label from calibration or test data is used by any
of the three rules.

## Experimental design

Five full-data runs used N-gram windows 2-6:

- Sepsis Cases, seeds 0, 1, and 7;
- Helpdesk, seed 0;
- BPI Challenge 2013, seed 0.

Entire cases stay within train, calibration, or held-out test partitions. The
three rules have no fitted parameters, so their calibration scores are
diagnostic only. Held-out effects are paired against soft voting:

- a **recovery** is a soft error corrected by the rule;
- a **harm** is a correct soft prediction replaced by an error;
- **net** is recoveries minus harms;
- **exact p** is a two-sided exact sign/McNemar test over those discordant
  events.

The p-values are descriptive. They are not corrected for multiple rules,
datasets, or seeds, and runs on different seeds of the same log are not
independent replications.

Reproduce the result table with:

```bash
python examples/evaluate_social_choice_rules.py \
  results/social-choice-rank-experiments
```

## Held-out results

| Dataset | Seed | Rule | Calibration | Test | Delta vs soft | Overrides | Recoveries | Harms | Net | Exact p |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BPI Challenge 2013 | 0 | Borda | 73.58% | 73.85% | +0.04% | 190 | 66 | 61 | +5 | 0.7228 |
| BPI Challenge 2013 | 0 | Copeland | 73.62% | 73.85% | +0.04% | 186 | 65 | 60 | +5 | 0.7207 |
| BPI Challenge 2013 | 0 | Maximin | 73.62% | 73.85% | +0.04% | 181 | 63 | 58 | +5 | 0.7163 |
| Helpdesk | 0 | Copeland | 85.54% | 84.79% | -0.03% | 8 | 3 | 4 | -1 | 1.0000 |
| Helpdesk | 0 | Borda | 85.56% | 84.76% | -0.05% | 7 | 2 | 4 | -2 | 0.6875 |
| Helpdesk | 0 | Maximin | 85.51% | 84.76% | -0.05% | 7 | 2 | 4 | -2 | 0.6875 |
| Sepsis Cases | 0 | Copeland | 64.00% | 63.61% | +0.91% | 146 | 58 | 35 | +23 | 0.0220 |
| Sepsis Cases | 0 | Maximin | 63.92% | 63.45% | +0.75% | 139 | 54 | 35 | +19 | 0.0558 |
| Sepsis Cases | 0 | Borda | 63.46% | 62.90% | +0.20% | 128 | 49 | 44 | +5 | 0.6785 |
| Sepsis Cases | 1 | Copeland | 64.96% | 65.24% | +0.69% | 90 | 40 | 23 | +17 | 0.0430 |
| Sepsis Cases | 1 | Maximin | 64.88% | 65.12% | +0.57% | 90 | 39 | 25 | +14 | 0.1034 |
| Sepsis Cases | 1 | Borda | 64.47% | 64.07% | -0.49% | 87 | 23 | 35 | -12 | 0.1480 |
| Sepsis Cases | 7 | Copeland | 63.31% | 63.60% | +0.00% | 107 | 32 | 32 | +0 | 1.0000 |
| Sepsis Cases | 7 | Maximin | 63.27% | 63.39% | -0.21% | 98 | 26 | 31 | -5 | 0.5966 |
| Sepsis Cases | 7 | Borda | 63.14% | 62.85% | -0.74% | 94 | 22 | 40 | -18 | 0.0300 |

Soft-voting test accuracies were 73.81% for BPI Challenge 2013, 84.81% for
Helpdesk, and 62.70%, 64.55%, and 63.60% for the three Sepsis splits.

## Evaluation and decision

Copeland is the only consistently promising candidate:

- across the three Sepsis splits it produced 130 recoveries and 90 harms,
  a total net gain of 40 events and a mean accuracy gain of 0.53 percentage
  points;
- it improved two Sepsis splits and tied the third;
- it was effectively neutral on BPI Challenge 2013 and Helpdesk.

This is evidence for a dataset-specific rank-consensus signal, not a general
improvement. The two positive Sepsis p-values are unadjusted, the third seed is
null, and neither additional dataset shows a meaningful effect.

Maximin follows the same direction but is weaker and reverses on Sepsis seed 7.
Borda is unstable: its aggregate Sepsis effect is negative and seed 7 has
significantly more harms than recoveries in the wrong direction.

All three implementations remain visible as candidate hypotheses so their
behavior can be inspected and reproduced. None is promoted to the mandatory
deployment integration. Copeland merits a later calibration-gated experiment
based only on structural features; Borda and maximin should not be promoted
without new cross-dataset evidence.

