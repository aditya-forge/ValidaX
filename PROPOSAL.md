# CSE303 Capstone Project Proposal: Digital Payments Fraud Investigation Engine

**Domain / catalog:** FinTech / CSE303 (F20). This is a transaction risk prioritization
system: a score indicates elevated risk warranting review, not a determination of fraud.

## Name / Roll Number

- **Name:** _[YOUR NAME]_
- **Roll Number:** _[YOUR ROLL NUMBER]_

## Problem statement

This project detects fraudulent e-commerce transactions in real card-not-present traffic
where only about 3.5% of transactions are fraud, so a model that never raises an alert
would look 96.5% accurate while catching nothing. The goal is to rank transactions by
fraud risk so investigators review the most suspicious ones first. That means balancing
how much fraud is caught against how many legitimate transactions are sent for review.

## Business framing

As scenario assumptions (not industry figures), a missed fraud costs **$500** and a false
positive investigation costs **$10**. In the prototype run, minimizing total cost under
these assumptions gives a **0.24 probability threshold**. That is far below the default
0.5, because missing a fraud is 50 times more expensive than an unnecessary review.
Investigators can handle a fixed **500 alerts per day** (also a scenario assumption), so
the order of the alert queue matters as much as the threshold.

## Related work

The IEEE-CIS Fraud Detection competition on Kaggle (2019, data from Vesta Corporation)
ranked submissions by **ROC-AUC** on an unlabeled test set, and its top solutions were
gradient-boosted tree ensembles built on heavy feature engineering. Because the
competition did not use PR-AUC, there is no published PR-AUC benchmark for this dataset.
The **0.45–0.50 PR-AUC** range used here is this project's own target on a chronological
holdout, not an industry standard. It is roughly 13–15 times the no-skill baseline of
0.034, which equals the fraud rate in the chronological test window. PR-AUC is the primary metric because precision-recall curves are more
informative than ROC curves when the positive class is rare (Saito and Rehmsmeier, 2015).

> _[TO VERIFY BEFORE SUBMISSION: winning team, final private-leaderboard ROC-AUC, and
> solution summary, checked against the official Kaggle leaderboard and write-ups.]_

## Dataset

- **Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
  (Kaggle competition, data provided by Vesta Corporation). These are real, anonymized
  transactions.
- **Size:** **590,540 labeled transactions** (`train_transaction.csv`), left-joined on
  `TransactionID` to identity and device signals (`train_identity.csv`). About 24% of
  transactions have identity data.
- **Split:** **chronological 80/20 on `TransactionDT`.** The first 472,432 transactions by
  time are used for training and the last 118,108 for testing. No random split is used,
  so the model never trains on transactions from the period it is tested on.
- **No test-label usage:** Kaggle's `test_transaction.csv` and `test_identity.csv` have no
  fraud labels and are not used. Hyperparameter tuning uses a chronological validation
  slice inside the training window, so the test window is never used to choose a model.

## Planned approach

The models follow the CSE303 ladder. Every model is evaluated on the same chronological
test window with the same metrics.

| Ladder step | Method | Notebook |
|---|---|---|
| Regression / baseline | Class-weighted **Logistic Regression** on interpretable features | `03_baseline_model` |
| Trees / feature selection | **Decision Tree**; tree-based feature ranking with a top-K selection test; **PCA** (95% variance) + Logistic Regression; SHAP-based global ranking | `04_model_comparison`, `03` |
| Probabilistic / SVM | **Gaussian Naive Bayes**; **linear SVM** on standardised features, sigmoid-calibrated on a chronological validation slice | `04_model_comparison` |
| Neural network | **Feed-forward ANN** (MLP, 128-64 hidden units, at most 10 epochs, early stopping on validation loss) | `04_model_comparison` |
| Ensembles / clustering | **XGBoost** on interpretable + masked features; **Random Forest**; **K-means** segments of the alert queue on interpretable features only | `03`, `05_tuning_and_benchmark` |
| Tuning / benchmark | Random search for Random Forest and XGBoost on a chronological validation slice; mean ± std across seeds and forward-chaining CV folds; threshold, 500 alerts/day capacity and $500 / $10 cost analysis | `05_tuning_and_benchmark` |

## Deployment plan

A multi-page **Streamlit** app in `app/app.py` loads the trained model bundle from
`app/model.pkl` and the analysis files in `outputs/`:

- **Fraud overview:** test-window metrics, the global SHAP drivers with family labels,
  and `outputs/shap_summary.png`. Model benchmarks from notebooks 04/05 appear when their
  outputs exist.
- **Alert queue:** a filterable triage list from `outputs/scored_transactions.csv`.
- **Transaction investigation:** look up a single transaction to see its risk score,
  tier, SHAP contribution chart and reason codes.
- **Threshold simulator:** shows the cost and capacity trade-offs, with the 0.24
  threshold marked.

Every page carries an operating-context banner: FinTech / CSE303 (F20), elevated risk
warranting review rather than a determination of fraud, the $500 / $10 scenario costs and
the 500 alerts/day capacity.

Masked feature families (`C1-C14`, `D1-D15`, `M1-M9`, `V1-V339` and `id_*`) are labeled
generically by family everywhere in the app, for example "masked counting feature
(C14)". The app never assigns a specific meaning to an individual masked column and never
shows a masked column's value in a reason code.

## Success criteria

| Criterion | Target |
|---|---|
| PR-AUC on the chronological test window (primary) | 0.45–0.50 (no-skill baseline 0.034) |
| Improvement over the Logistic Regression baseline | Clear PR-AUC gain on the same split |
| ROC-AUC (secondary) | Reported for every model |
| Cost-sensitive operating point | Total scenario cost at the 0.24 threshold reported and compared with the default 0.5 |
| Fixed capacity | Average daily fraud capture and precision at 500 alerts/day reported |
| Model ladder | Every ladder step evaluated on the same split and metrics, with mean ± std across seeds and forward-chaining folds for the tuned ensembles |
| Explainability | Every alert has SHAP reason codes; masked features keep family labels only |
| Leakage control | No random split, no future information in features, no test labels or test window used for tuning |

## "Make it yours" twist

1. **Point-in-time `card1` velocity features.** For each transaction we compute the
   card's prior transaction count, its average and largest past amounts, the time since
   its last transaction, and how this amount compares with its usual amount. All of
   these use expanding windows ordered by `TransactionDT`, so each transaction only sees
   earlier activity.
2. **Fixed-capacity and cost-sensitive threshold decisioning.** The alert threshold
   minimizes total scenario cost, which gives the 0.24 probability threshold. A daily
   simulation then shows how much fraud a team limited to 500 reviews per day would
   actually catch.
3. **SHAP reason codes with strict masked-family labeling.** Each alert gets
   plain-English reasons from its SHAP values. Documented columns get specific labels,
   such as "purchaser email domain". Masked columns get family labels only, such as
   "masked Vesta-engineered feature (V70)". The project never guesses what a masked
   column means and never shows a masked column's value in a reason code. Reason codes
   explain why a transaction warrants review; they are not a determination of fraud.
