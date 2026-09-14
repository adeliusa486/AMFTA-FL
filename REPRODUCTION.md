# Reproducing the manuscript's tables

This repository could not produce the configuration the manuscript describes.
The gaps are listed below, together with what was added to close them and how
to run the study.

## What was missing, and what now exists

| Manuscript element | Status before | Added |
|---|---|---|
| Logistic-regression model | absent (only `LocalMLP`) | `amfta/models/logistic.py` |
| NormClip-Only ablation arm | absent | `amfta/aggregation/extra_baselines.py` |
| Coordinate-wise Median | absent | same |
| Multi-Krum (as a selectable method) | flag only | same |
| FoolsGold | absent | same |
| Adaptive / AGR-tailored attack | absent | `amfta/attacks/adaptive.py` |
| Dirichlet α = 0.1 condition | not runnable (fixed α = 0.5 partitions) | `repartition` flag on `RunConfig` |
| Study driver for every table | absent | `experiments/run_paper_study.py` |
| Equivalence tests, Holm, trend test | absent | `experiments/paper_stats.py` |

AMFTA-ND (`amfta_noq`) and median-norm rescaling were already implemented; they
had simply never been run, since no result file for either exists in the repo
history.

The two runner patches are idempotent and were applied by `patch_runner.py`
and `patch_runner2.py`. Both default to the previous behaviour
(`model_class="mlp"`, `repartition=False`), so nothing that worked before
changes.

## Running it

```bash
python experiments/run_paper_study.py --block clean       --model logistic
python experiments/run_paper_study.py --block labelflip   --model logistic
python experiments/run_paper_study.py --block gaussian    --model logistic
python experiments/run_paper_study.py --block adaptive    --model logistic
python experiments/run_paper_study.py --block extras      --model logistic
python experiments/run_paper_study.py --block normclip    --model logistic
python experiments/run_paper_study.py --block alpha01     --model logistic
python experiments/run_paper_study.py --block scalability --model logistic
```

Then:

```bash
python experiments/paper_stats.py --results results_paper --out tables
```

Blocks are resumable: a configuration already present in the block's CSV is
skipped, so the study can be run in pieces and interrupted safely. The default
seed list has ten entries and begins with the manuscript's three
(42, 123, 456), so a three-seed run is a strict prefix of a ten-seed one.

## Paper ↔ repository correspondence

Every number in the manuscript is produced by one of the following. Nothing in
the tables is typed by hand.

| Manuscript element | Produced by | Needs |
|---|---|---|
| Tables 6, 7 (accuracy), 8 (detection), 11 (per-seed) | `sustainability/rebuild_tables_from_logs.py --results results/` | logs only |
| Table 9 (degradation, TOST) | `sustainability/equivalence_tests.py` | nothing |
| Table 10 (pairwise, Holm, Hedges' g) | `sustainability/equivalence_tests.py` + `experiments/paper_stats.py` | logs only |
| Tables 13–19 (energy, sensitivity, projection, Pareto, scheduling, p_min) | `sustainability/resource_model.py` | nothing |
| Table 12, Figure 8 (aggregation timing) | `sustainability/make_scalability_figure.py` | nothing, no GPU |
| Figures 6, 7, 9 | `sustainability/make_figures.py` | logs only |

**Standard-deviation convention.** All across-seed dispersion is the *sample*
standard deviation (`ddof = 1`). An earlier version of both the manuscript and
`equivalence_tests.py` used the population standard deviation, which understates
every interval by about 18%. Correcting it moves two verdicts: Trimmed Mean
under Gaussian noise from *degrades* (p = 0.044) to *inconclusive* (p = 0.064),
and FedDBC under Gaussian noise survives only at p = 0.047. AMFTA-ND's
equivalence verdicts hold in both families.

**Runs not reported in the manuscript.** `results/` also contains ρ = 0.40
label-flipping runs for the six baselines and sign-flipping runs at ρ = 0.30 for
six methods. Neither appears in the paper because AMFTA-ND was never run in
either configuration. Where more than one log exists for a
(method, ρ, attack, seed) tuple, the table builders take the most recent
non-empty file.

## Three things to settle before the tables are trustworthy

**1. The model dimension.** The manuscript now states 41 NetFlow features
plus a bias, d = 42 parameters, and records separately that the resource model
of Section 3.2 is evaluated at d = 41 (a 2.4% underestimate of payload that
changes no ordering). A later regeneration of the preprocessing produced 45
feature columns, i.e. d = 46; that pipeline is **not** the one that produced the
released logs and is not shipped here (`data/` contains only a README).
`LogisticRegression.num_parameters()` reports the true count for whatever is
actually built, and any re-run should fill the table from it rather than
asserting the value.

**2. The class balance — resolved for the released logs.** The manuscript
previously stated "54.1% normal / 45.9% attack" and reasoned from those figures
about constant classifiers. That was wrong. The evaluation split behind every
file in `results/` is **72.6% class 1 (attack)**: in runs where a model predicts
the positive class everywhere, precision and accuracy are both exactly 0.7258,
and in the four collapsed Gaussian-noise seeds, where the model predicts the
negative class everywhere, accuracy is exactly 0.2741 with precision, recall and
F1 at zero. The manuscript now reports 27.4% / 72.6% and a 72.6% majority-class
baseline, and its constant-predictor arithmetic has been rewritten accordingly.

A later regeneration of the preprocessing produced a 62.0% positive split. That
is a different pipeline and does not reproduce the released logs; do not mix the
two. Note the consequence the manuscript now states: at ρ = 0.30 under label
flipping, FedAvg (73.8%), FLTrust (72.1%) and FedDBC (72.0%) sit at or below the
majority-class baseline.

**3. The dataset may be close to separable.** With this preprocessing, methods
reach 99%+ accuracy on the released logs even under 30% label flipping. A
benchmark on which undefended FedAvg is barely degraded does not discriminate
between aggregation rules, and reviewers will notice. Before re-running the
full study it is worth checking how much of the label information sits in one
or two features, and whether the deduplication step has left near-duplicate
rows spanning the train/test split.

## What the new components do and do not guarantee

The aggregators were unit-tested on controlled updates: with 70 honest clients
sharing a direction and 30 colluding clients pushing against it, FoolsGold,
coordinate-wise Median and NormClip-Only all recover the honest direction
(cosine +0.99 or better) and reject the coalition. That verifies the mechanics,
not the empirical claims.

The adaptive attack implements the coalition-level formulation the manuscript
describes, including a binary search on the perturbation magnitude against an
acceptance oracle. Without an oracle it defaults to the largest magnitude that
median-norm rescaling leaves untouched. Its strength depends on `gamma_init`,
`jitter` and `knowledge`, all of which must be reported in the paper — the
manuscript's own limitations section asks for exactly that specification.

Nothing here reproduces the numbers currently in the manuscript. It makes it
possible to generate numbers, which was not previously the case.
