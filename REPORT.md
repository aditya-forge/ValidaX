# CSE303 Capstone Project Report: Digital Payments Fraud Investigation Engine

> **Status of this report.** Every number below comes from completed runs on the real
> IEEE-CIS data:
>
> - the main pipeline run saved in `outputs/` (`model_summary.json`, `threshold_sweep.csv`,
>   `cost_sensitive_threshold.csv`, `daily_capacity_simulation.csv`,
>   `feature_importance.csv`, `scored_transactions.csv`, `shap_summary.png`);
> - `notebooks/04_model_comparison.ipynb`, which wrote `outputs/model_comparison.csv`
>   (Section 5);
> - `notebooks/05_tuning_and_benchmark.ipynb`, which wrote `outputs/xgb_tuning_trials.csv`,
>   `outputs/rf_tuning_trials.csv`, `outputs/tuning_benchmark.csv` and
>   `outputs/alert_clusters.csv` (Section 6).
>
> All sections are complete. No results have been estimated or extrapolated.

## 1. Summary

We built a fraud triage pipeline on the IEEE-CIS (Vesta) dataset, using a chronological
split of **472,432 training** and **118,108 test** transactions. On the test window, the
XGBoost model reaches **PR-AUC 0.488** and **ROC-AUC 0.891**, compared with **0.143** and
**0.757** for a Logistic Regression baseline.

Two operating results matter most:

- Under the scenario costs ($500 per missed fraud, $10 per false positive), the
  cost-minimizing alert threshold is a **0.24 probability threshold**. It catches 86.4% of
  fraud at 10.8% precision.
- With a capacity of **500 alerts per day**, investigators would catch on average
  **79.3%** of each day's fraud.

Across the full CSE303 model ladder (Section 5), XGBoost stays the clear leader. In the
re-run in notebook 04 it scored PR-AUC 0.4862, against a no-skill baseline of 0.034. Next
come the feed-forward ANN (0.3949) and a single decision tree (0.3369). The linear models
land between 0.14 and 0.21, and Gaussian Naive Bayes is weakest (0.0869).

The model flags **elevated risk that warrants review**. A flag is not a determination that
a transaction is fraudulent.

## 2. Problem and framing

Fraud makes up about 3.5% of the traffic, so accuracy is misleading: a model that never
alerts would score about 96.5%. We therefore treat the task as ranking. The primary metric
is PR-AUC, and the operating point is chosen from costs and investigator capacity rather
than a default 0.5 cut-off. The $500 and $10 costs and the 500 alerts/day capacity are
**scenario assumptions**, not industry figures. The full framing is in `PROPOSAL.md`.

## 3. Data and setup

- **Source:** IEEE-CIS Fraud Detection (Kaggle, data from Vesta Corporation). The data is
  real and anonymized. `train_transaction.csv` (590,540 labeled rows) is left-joined to
  `train_identity.csv` on `TransactionID`.
- **Split:** chronological 80/20 on `TransactionDT`: **472,432** training rows and
  **118,108** test rows. The test window has **4,064** frauds, a fraud rate of **3.44%**.
  Kaggle's unlabeled test files are not used.
- **Features:** **451** model features. They are interpretable business features
  (including point-in-time `card1` velocity features computed only from earlier
  transactions), the masked columns used as opaque inputs, and missingness counts per
  masked family.
- **Masked feature families.** The exact meaning of these columns is withheld by Vesta,
  and this project never assigns one. They are always labeled by family:

  | Family | Generic label used |
  |---|---|
  | `C1-C14` | masked counting feature |
  | `D1-D15` | masked time-delta feature |
  | `M1-M9` | masked match flag |
  | `V1-V339` | masked Vesta-engineered feature |
  | `id_*` | masked identity/device signal |

## 4. Models (completed run)

| Model | Features | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression (class-weighted) | interpretable only | 0.757 | 0.143 |
| **XGBoost** (`scale_pos_weight`, native categoricals and missing values) | all 451 | **0.891** | **0.488** |
| No-skill reference | — | 0.500 | 0.034 |

The XGBoost PR-AUC of 0.488 falls inside the project's own 0.45–0.50 target band, and it
is more than three times the baseline's PR-AUC. The Kaggle competition was scored on
ROC-AUC against a different, unlabeled test set, so these figures cannot be compared
directly with the leaderboard.

## 5. Model comparison (completed run, `notebooks/04_model_comparison.ipynb`)

All models are scored on the same chronological test window (118,108 transactions, 3.44%
fraud). Results are from `outputs/model_comparison.csv`, ranked by PR-AUC:

| Rank | Model | PR-AUC | ROC-AUC | Training rows | Fit time |
|---|---|---|---|---|---|
| 1 | **XGBoost (champion)**, all 451 features | **0.4862** | **0.8855** | 472,432 | — |
| 2 | Feed-forward ANN (MLP 128-64) | 0.3949 | 0.8467 | 120,000 | 12.6 s |
| 3 | Decision Tree (depth 8) | 0.3369 | 0.8248 | 150,000 | 13.8 s |
| 4 | Logistic Regression, all encoded columns | 0.2129 | 0.8356 | 150,000 | 30.5 s |
| 5 | Linear SVM (`dual=False`, sigmoid-calibrated) | 0.2020 | 0.8272 | 120,000 | 803.5 s |
| 6 | Logistic Regression, top-40 tree-ranked columns | 0.1704 | 0.8196 | 150,000 | 1.1 s |
| 7 | PCA (95% variance) + Logistic Regression | 0.1671 | 0.8216 | 150,000 | 4.8 s |
| 8 | Logistic Regression baseline, interpretable features only | 0.1428 | 0.7577 | 472,432 | — |
| 9 | Gaussian Naive Bayes | 0.0869 | 0.7620 | 150,000 | 0.8 s |
| — | No-skill reference (fraud rate) | 0.034 | 0.500 | — | — |

**Setup.**

- XGBoost and the baseline are trained on the full training window.
- Models 3, 4, 6, 7 and 9 use a 150,000-row random sample of the training window, with
  median imputation, standardisation and one-hot encoding (626 encoded columns).
- The SVM and ANN train on the earlier 120,000 rows of that sample. The latest 30,000 rows
  (by `TransactionDT`) are held back to calibrate the SVM and to early-stop the ANN.
- The test window is never used for fitting, calibration or early stopping.

**XGBoost in two runs.** Notebook 04 retrains XGBoost with the same settings as the main
pipeline, under XGBoost 3.4.1. It scores PR-AUC 0.4862 / ROC-AUC 0.8855, close to the
main pipeline run's 0.488 / 0.891 in Section 4. The deployed model (`app/model.pkl`) and
all operating results in Section 7 come from the main pipeline run.

**Key findings.**

- **XGBoost leads clearly.** Its PR-AUC is about 14 times the no-skill baseline and 3.4
  times the interpretable-only baseline. It is the only model inside the project's
  0.45–0.50 target band.
- **The feed-forward ANN is the strongest comparison model** (PR-AUC 0.3949). Training was
  capped at 10 epochs, with early stopping on validation loss and patience 3. It stopped at
  epoch 7 and kept the weights from epoch 4 (validation log-loss 0.1077).
- **A single decision tree (0.3369) beats every linear model.** This points to non-linear
  interactions that linear models cannot express. The tree uses 49 of the 451 source
  columns. Its top splits are on masked columns, reported by family label only:
  - masked Vesta-engineered features (V258), (V70) and (V317);
  - masked counting features (C14) and (C8);
  - masked time-delta feature (D2).
- **Linear models cluster between 0.14 and 0.21 PR-AUC:**
  - Adding all encoded columns lifts Logistic Regression from 0.1428 (baseline) to 0.2129.
  - Keeping only the columns of the top-40 tree-ranked features (50 encoded columns) keeps
    most of that gain (0.1704).
  - PCA needs 156 components to keep 95% of the variance, and reaches 0.1671.
  - The calibrated linear SVM (0.2020) performs like Logistic Regression and is by far
    the slowest model to fit.
- **Gaussian Naive Bayes is weakest** (0.0869) and badly calibrated: 98.5% of its test
  scores fall below 0.01 or above 0.99, against 1.8% for XGBoost. Its independence and
  normality assumptions don't fit hundreds of correlated, heavy-tailed, partly binary
  columns.
- **ROC-AUC hides most of these differences.** All nine models fall between 0.76 and 0.89
  ROC-AUC, while PR-AUC ranges from 0.09 to 0.49. This confirms PR-AUC as the
  primary metric for this imbalanced problem.

Part of the gap between XGBoost and the sampled models reflects training volume (472,432
vs. 120,000–150,000 rows). Part reflects missing-value handling: XGBoost handles missing
values natively, while the others see median-imputed values.

## 6. Tuning, benchmarking and clustering (completed run, `notebooks/05_tuning_and_benchmark.ipynb`)

**Protocol.**

- **Model choice uses validation only.** Hyperparameters and the operating model are
  chosen on a **chronological validation slice**: the last 20% of the training window
  (94,487 transactions, days 109–141), with models fit on the earlier 377,945. The test
  window is used only for reporting.
- **Fast settings.** The notebook ran on its fast profile:
  - 3 XGBoost combinations, each capped at 300 trees with early stopping after 20 rounds;
  - the 6-trial Random Forest search, reused from an earlier run of the same notebook on
    the same data, split and seed;
  - 2 seeds (42, 43) for the test-window benchmark;
  - 2 forward-chaining CV folds.

  Standard deviations use `ddof=1`. With only 2 seeds and 2 folds they are rough estimates
  of spread.

### 6.1 Tuning (validation PR-AUC; `xgb_tuning_trials.csv`, `rf_tuning_trials.csv`)

| Candidate | Settings | Validation PR-AUC | Validation ROC-AUC |
|---|---|---|---|
| **XGBoost, default parameters (selected)** | 400 trees, learning rate 0.05, depth 6 (`src/model.py` defaults) | **0.5497** | — |
| XGBoost, best fast trial ("tuned") | learning rate 0.15, depth 6; early-stopped at 197 trees, refit with 246 | 0.5384 | 0.8939 |
| XGBoost, fast trial | learning rate 0.08, depth 6; early-stopped at 124 trees | 0.5262 | 0.9020 |
| XGBoost, fast trial | learning rate 0.12, depth 4; early-stopped at 122 trees | 0.5084 | 0.8951 |
| Random Forest, best of 6 trials ("tuned") | 300 trees, `max_depth=16`, `min_samples_leaf=50`, `max_features=0.2` | 0.5229 | 0.8995 |

**Outcome.**

- The fast search did not beat the default XGBoost parameters on validation, so the
  **default XGBoost was retained as the operating model**.
- The notebook's promotion step was off (`PROMOTE_TUNED_MODEL = False`), and **`app/model.pkl`
  was not changed**. The app and all results in Sections 7–8 still come from the main
  pipeline model.

### 6.2 Benchmark: mean ± std (`tuning_benchmark.csv`)

| Model | Test PR-AUC (2 seeds) | Test ROC-AUC (2 seeds) | Forward-chaining CV PR-AUC (2 folds) | Forward-chaining CV ROC-AUC (2 folds) |
|---|---|---|---|---|
| **XGBoost, default (selected)** | **0.4827 ± 0.0049** | 0.8861 ± 0.0008 | not run* | not run* |
| XGBoost, tuned | 0.4884 ± 0.0101 | 0.8803 ± 0.0006 | 0.4652 ± 0.0232 | 0.8422 ± 0.0302 |
| Random Forest, tuned | 0.4593 ± 0.0010 | 0.8826 ± 0.0001 | 0.4957 ± 0.0003 | 0.8825 ± 0.0080 |

\* The notebook's CV step covers the two tuned candidates only.

- **Default vs. tuned XGBoost:** the tuned model's test mean is 0.0057 higher, but that is
  smaller than its own seed spread (± 0.0101). The two are indistinguishable at this
  sample size, and the tuned model scored lower on validation, which is the selection
  criterion.
- **Random Forest:** the most seed-stable model (± 0.0010), but about 0.02–0.03 PR-AUC
  below XGBoost on the test window. In the forward-chaining folds (which lie inside the
  training window) it was both higher and steadier than tuned XGBoost (0.4957 ± 0.0003
  vs. 0.4652 ± 0.0232). That suggests its quality varies less across time periods, at a
  lower peak on the most recent window.
- **Target band:** both XGBoost variants stay within the 0.45–0.50 PR-AUC target, and the
  Random Forest sits just inside its lower edge.

### 6.3 Operating analysis for the selected model (scenario: $500 missed fraud, $10 false positive; 500 alerts/day)

These results are for the selected default XGBoost, retrained in notebook 05 (seed 42).
It is the same retrain as in notebook 04 (test PR-AUC 0.4862), not the deployed pipeline
model. That is why its cost-optimal threshold differs slightly from the 0.24 in Section 7.

| Model | Cost-optimal threshold | Precision | Recall | Total scenario cost | Avg. daily capture at 500/day | Avg. daily precision at 500/day |
|---|---|---|---|---|---|---|
| **XGBoost, default (selected)** | **0.26** | **11.2%** | **84.6%** | **$583,810** | **78.9%** | **15.3%** |
| XGBoost, tuned | 0.22 | 12.1% | 82.8% | $593,870 | 79.0% | 15.3% |
| Random Forest, tuned | 0.24 | 9.7% | 86.5% | $601,120 | 75.6% | 14.6% |

- **Fixed capacity:** for the selected model, reviewing the top 500 scores each day
  catches on average **78.9%** of that day's fraud, ranging from **57.1% to 95.5%** across
  the 42 test days, at **15.3%** average precision. A single global top-500 cut over the
  whole window reaches **93.0% precision** but only **11.4% recall**.
- **Consistency with Section 7:** the main pipeline model gave 0.24, 79.3% daily capture
  and $566,840. The operating picture is consistent across retrains:
  - a cost-optimal threshold of about 0.22–0.26, well below the default 0.5;
  - about 79% daily capture at 500 alerts/day.
- **Cost ranking:** under the $500 / $10 scenario the selected model also has the lowest
  total cost of the three candidates.

### 6.4 Alert segments (`alert_clusters.csv`)

**Method.**

- **Input:** the 30,621 test-window alerts at the selected model's cost-optimal threshold
  (0.26), clustered with K-means into k = 4 segments; k was chosen by sampled silhouette
  score.
- **Features:** interpretable features only (`m.CLUSTER_FEATURES`). The notebook asserts
  that no masked column (`C1-C14`, `D1-D15`, `M1-M9`, `V1-V339`, `id_*`) is an input, and the
  output file contains none.
- **`isFraud`** was used only to profile the segments afterwards, never as an input.
- **Typical amount** is the back-transformed mean of the log amount, so roughly a
  geometric mean.
- **Time** is a position in the relative `TransactionDT` cycle, not a clock time.

| Segment | Alerts (share) | Fraud rate | Frauds (share of alert frauds) | Typical amount | Hours 0–5 of relative cycle | Free email domain | Identity data | Mobile device | First-seen card |
|---|---|---|---|---|---|---|---|---|---|
| **1** | 3,219 (10.5%) | **28.1%** | 906 (26.3%) | ≈ $43 | 34% | 90% | 100% | 99% | 0% |
| **2** | 184 (0.6%) | **20.1%** | 37 (1.1%) | ≈ $113 | 25% | 83% | 57% | 23% | 100% |
| **3** | 5,316 (17.4%) | **10.6%** | 561 (16.3%) | ≈ $90 | 100% | 75% | 23% | 0% | 0% |
| **4** | 21,902 (71.5%) | **8.8%** | 1,936 (56.3%) | ≈ $112 | 0% | 72% | 13% | 0% | 0% |

Together the four segments hold 3,440 frauds among 30,621 alerts (11.2%), matching the
cost-optimal precision above.

- **Segment 1: mobile, identity-linked, free-email, low-amount alerts.** The richest
  segment: it is 10.5% of the alert queue but holds over a quarter of its fraud, at
  2.5 times the queue's average fraud rate. It is a natural first-review lane.
- **Segment 2: first transaction seen for the card.** A small, high-rate segment. These
  cards have no history: the "time since this card's last transaction" average of −1 is
  the "no prior transaction" placeholder, and the amount-vs-typical ratio (about 180×) is
  inflated because there is no prior average to compare against. Both should be read as
  "no history", not as real values.
- **Segment 3: hours 0–5 of the relative cycle, non-mobile.** Fraud rate close to the
  queue average.
- **Segment 4: rest of the relative cycle, mostly no identity data.** The bulk of the queue
  (71.5% of alerts) at the lowest fraud rate. It still contains over half of all alert
  fraud, so it can't simply be deprioritised.

`card1` is a shared card-attribute code used as a pseudo account key, not a unique card.
So the "prior transaction count" averages (about 2,100–2,400 in segments 1, 3 and 4) count
earlier transactions sharing that code, not one card's history.

Segments describe groups of alerts in interpretable terms to help organise review. They
are **not fraud verdicts**: every alert still indicates elevated risk warranting review,
not a determination of fraud.

## 7. Operating the model (completed run)

### 7.1 Threshold trade-off (`threshold_sweep.csv`)

| Threshold | Alerts | Precision | Recall |
|---|---|---|---|
| 0.1 | 71,848 | 5.4% | 95.3% |
| 0.2 | 40,346 | 9.0% | 89.0% |
| 0.3 | 24,379 | 13.7% | 82.0% |
| 0.4 | 16,199 | 18.9% | 75.4% |
| 0.5 | 11,106 | 24.9% | 68.1% |
| 0.6 | 7,545 | 32.6% | 60.5% |
| 0.7 | 4,820 | 43.2% | 51.2% |
| 0.8 | 2,948 | 55.9% | 40.6% |
| 0.9 | 1,530 | 75.2% | 28.3% |

### 7.2 Cost-sensitive threshold (`cost_sensitive_threshold.csv`)

Scenario assumptions: missed fraud = $500, false positive = $10.

| Threshold | Alerts | Frauds caught | Frauds missed | False positives | Precision | Recall | Total scenario cost |
|---|---|---|---|---|---|---|---|
| **0.24 (cost-optimal)** | 32,545 | 3,511 | 553 | 29,034 | 10.8% | 86.4% | **$566,840** |
| 0.50 (default) | 11,106 | 2,767 | 1,297 | 8,339 | 24.9% | 68.1% | $731,890 |

At this 50:1 cost ratio the cheapest threshold favours recall. Moving from 0.5 to the
0.24 probability threshold lowers total scenario cost by $165,050 (22.6%) on the test
window, even though it creates about three times as many alerts.

### 7.3 Fixed capacity of 500 alerts/day (`daily_capacity_simulation.csv`)

Across the **42 test days** (days 141–182), every day had more than 500 transactions. When
investigators review only the 500 highest-scored transactions each day:

- They catch on average **79.3%** of the day's fraud. The daily rate ranges from 60.8% to
  93.2%.
- Average precision is **15.4%**.

For comparison, a single top-500 cut over the whole test window (`model_summary.json`)
reaches 93.8% precision but catches only 469 of 4,064 frauds (11.5% recall). That is why
we apply the capacity per day.

### 7.4 Risk tiers (`scored_transactions.csv`)

| Tier | Score range | Transactions | Frauds | Observed fraud rate |
|---|---|---|---|---|
| Critical | ≥ 0.8 | 2,948 | 1,649 | 55.9% |
| High | 0.6–0.8 | 4,597 | 808 | 17.6% |
| Medium | 0.3–0.6 | 16,834 | 876 | 5.2% |
| Low | < 0.3 | 93,729 | 731 | 0.8% |

All 24,379 Medium-or-higher transactions carry three SHAP reason codes. Together they
contain 3,333 of the 4,064 test-window frauds (82.0%). The tier cut-offs are project
operating rules, not validated benchmarks.

## 8. Explainability (completed run)

`outputs/shap_summary.png` shows TreeSHAP values on a 20,000-transaction sample of the
test window. The top drivers by mean |SHAP| (`feature_importance.csv`) are:

| Rank | Feature | Label used in reason codes | Mean \|SHAP\| |
|---|---|---|---|
| 1 | `P_emaildomain` | purchaser email domain | 0.181 |
| 2 | `C14` | masked counting feature (C14) | 0.179 |
| 3 | `C13` | masked counting feature (C13) | 0.177 |
| 4 | `card6` | card type (credit/debit) | 0.173 |
| 5 | `TransactionAmt` | transaction amount | 0.166 |
| 6 | `C1` | masked counting feature (C1) | 0.152 |
| 7 | `V70` | masked Vesta-engineered feature (V70) | 0.133 |
| 8 | `C11` | masked counting feature (C11) | 0.132 |
| 9 | `R_emaildomain` | recipient email domain | 0.125 |
| 10 | `card_avg_amount_prior` | this card's average past transaction amount | 0.119 |
| 11 | `V294` | masked Vesta-engineered feature (V294) | 0.118 |
| 12 | `id_31` | masked identity/device signal (id_31) | 0.116 |
| 13 | `D2` | masked time-delta feature (D2) | 0.113 |
| 14 | `M4` | masked match flag (M4) | 0.104 |
| 15 | `card_max_amount_prior` | this card's largest past transaction amount | 0.090 |

All five masked families appear in the top 15. That is a real property of this dataset,
and we report it as such. We do not interpret what any individual masked column means.
Interpretable signals are also among the strongest drivers: email domains, card type,
amount, and the two `card1` velocity features built in this project.

## 9. Deployment

Run `streamlit run app/app.py` from the repository root. The app loads `app/model.pkl` and
the files in `outputs/`, never writes to them, and has four pages:

- **Fraud overview:** test-window metrics, a risk-ladder view of the tiers, and the
  global SHAP drivers from `feature_importance.csv` with family labels.
  `shap_summary.png` is available in a collapsed section. A model benchmarks section shows
  `model_comparison.csv` (Section 5) and `tuning_benchmark.csv` (Section 6).
- **Alert queue:** a filterable triage list from `scored_transactions.csv`, plus the alert
  segments from `alert_clusters.csv` (Section 6.4), with interpretable features only.
- **Transaction investigation:** pick one of the top 500 alerts or look up any
  test-window ID. It shows a SHAP contribution chart and the top five reason codes, each
  marked as raising or lowering risk.
- **Threshold simulator:** cost and capacity trade-offs, with the 0.24 threshold marked.

A banner on every page states the operating context: FinTech / CSE303 (F20), the $500 /
$10 scenario costs and the 500 alerts/day capacity. Every page labels masked features by
family only and never shows a masked column's value. Every score is presented as elevated
risk that warrants review, not as a determination of fraud.

## 10. Limitations

- The data comes from a single provider (Vesta) and covers a fixed window of about six
  months. Model behaviour will not transfer directly to another payments product,
  region or period, and fraud tactics drift.
- `TransactionDT` is a relative time offset, not a calendar timestamp. Day-of-week and
  hour features come from modulo buckets, not real dates.
- Several of the strongest predictors are masked by the data provider. This project
  deliberately does not guess their meaning, which limits how much of the model's
  behaviour can be explained to a reviewer.
- The $500 / $10 costs, the 500 alerts/day capacity and the risk-tier cut-offs
  (0.8 / 0.6 / 0.3) are project assumptions, not validated industry benchmarks.
- There is no live transaction stream and no investigator feedback loop.
- **Operational disclaimer:** the model's output indicates **elevated risk warranting
  review**. It is not a determination that a transaction is fraudulent, and it should not
  be used to block transactions automatically.

## 11. Next steps

1. ~~Run `notebooks/04_model_comparison.ipynb` on the real data and fill in Section 5.~~
   Done.
2. ~~Run `notebooks/05_tuning_and_benchmark.ipynb` on the real data and fill in
   Section 6.~~ Done (fast profile; default XGBoost retained, `app/model.pkl` unchanged).
3. **Optional:** re-run notebook 05 with more XGBoost trials, seeds and folds. That would
   narrow the mean ± std estimates and test whether a broader search can beat the
   default parameters on validation.
4. Possible extensions: client-level grouping beyond `card1`, rolling retraining with
   drift monitoring, and an investigator feedback loop.
