# ValidaX: Digital Payments & E-Commerce Fraud Detection Engine

**Python | SQL (DuckDB) | scikit-learn | XGBoost | SHAP | Streamlit**

Local Web Console: [http://localhost:8501](http://localhost:8501) (Start locally with `streamlit run app/app.py`)

ValidaX is a machine learning investigation engine built to detect and triage fraudulent card-not-present (CNP) e-commerce transactions. Instead of treating fraud detection as a naive 50/50 classification problem, ValidaX frames it as a risk-ranking and cost-optimization challenge: fraud accounts for only ~3.5% of total volume, while investigator review capacity is strictly constrained.

Every transaction scored by the engine receives an calibrated fraud probability, an assignment to an operational risk tier (Critical, High, Medium, Low), a decision recommendation based on a cost-optimal 0.24 probability threshold, and plain-language reason codes derived from TreeSHAP values.

Project Documentation:
- [Learning Guide (Comprehensive ML, Viva & Interview Prep)](Learning%20Guide.md)
- [Final Engineering Report (REPORT.md)](REPORT.md)
- [Project Proposal & Scope (PROPOSAL.md)](PROPOSAL.md)
- [DuckDB Exploratory SQL Scripts (sql/fraud_analysis.sql)](sql/fraud_analysis.sql)
- [Jupyter Notebooks Walkthrough (01 to 05)](notebooks/)

---

## Project Status & Verified Run Metrics

The complete pipeline has been executed on the full IEEE-CIS transaction dataset (590,540 transactions). All numbers quoted across this README, the reports, and the web console are reproduced from outputs stored in `outputs/` and `artifacts/`:
- **Champion Model**: XGBoost trained on 451 features (31 categorical, 420 numeric).
- **Primary Metric**: Test PR-AUC **0.488** (vs. No-Skill Baseline **0.034**, and Logistic Baseline **0.143**).
- **Secondary Metric**: Test ROC-AUC **0.891** (vs. Baseline **0.757**).
- **Cost-Optimal Threshold**: **0.24** (assuming scenario costs of $500 per missed fraud vs. $10 per false-positive investigation).
- **Daily Capacity Capture**: Captures **79.3%** of each day's fraud volume when investigators review the top 500 alerts/day.

---

## The Problem & Engineering Framing

In production payments environments, fraud teams face an asymmetric workload problem:
1. **Severe Imbalance**: Less than 3.5% of incoming transactions are illicit. A dumb model predicting "legitimate" for everything achieves 96.5% raw accuracy while missing 100% of financial losses.
2. **Fixed Review Bandwidth**: Human fraud analysts cannot review millions of events. They have fixed daily capacity (e.g., 500 alerts/day).
3. **Asymmetric Error Costs**: Approving a stolen card chargeback costs the merchant roughly $500 (merchandise loss + network chargeback fee). Wasting an analyst's 5-minute review on a false positive costs ~$10. A standard 0.50 cutoff is economically reckless because false negatives cost 50x more than false positives.

The real job of the machine learning system is:
- Correctly rank suspicious transactions so the riskiest cases appear at the top of the queue.
- Balance false positives against false negatives at the dollar-minimizing probability threshold.
- Deliver interpretable, legally defensible reason codes so investigators can approve or deny claims within seconds.

---

## Dataset: IEEE-CIS E-Commerce Fraud Detection

The system is trained and evaluated on real-world, anonymized card-not-present transactions provided by Vesta Corporation on Kaggle:
- **590,540 labeled transactions** (`train_transaction.csv`, 394 features).
- **144,233 identity/device records** (`train_identity.csv`, 41 features, ~24.4% join match rate).
- **Split Scheme**: Chronological 80/20 split based on `TransactionDT`. The first 472,432 records form the training set (Days 1 to 141), while the last 118,108 records form the unseen holdout test set (Days 141 to 182; 4,064 frauds, 3.44% fraud rate).
- **Why this dataset?**: Unlike synthetic datasets (PaySim, Sparkov) or pure PCA transformations (ULB Credit Card), IEEE-CIS contains rich, real-world signals: purchaser/recipient email domains, card networks, device types, transaction velocities, and billing address distances.

### Strict Project Policy on Masked Features
Vesta Corporation masked the true column definitions for `C1-C14` (counting metrics), `D1-D15` (time deltas), `M1-M9` (verification matches), `V1-V339` (Vesta proprietary signals), and `id_01-id_38` (device attributes).
To ensure scientific integrity and prevent hallucinations, **this project never invents real-world meanings for masked columns**. They are used as opaque numeric/categorical inputs during modeling, and in UI reason codes they are strictly displayed with their standardized family name (e.g., `masked counting feature (C14)` or `masked Vesta feature (V70)`).

---

## System Architecture

```
Raw Kaggle CSVs (Transaction + Identity)
               │
               ▼
DuckDB Exploratory SQL Analysis (sql/fraud_analysis.sql)
               │
               ▼
Feature Engineering Pipeline (src/features.py)
 ├── Interpretable features (Amounts, Domains, Devices, Modulo Time)
 ├── Card1 expanding-window velocity engine (prior counts, avg/max amount)
 └── Masked feature family missingness indicators
               │
               ▼
Strict Chronological 80/20 Split (src/model.py)
 ├── Train Window: First 472,432 rows (Days 1-141)
 └── Holdout Test Window: Last 118,108 rows (Days 141-182)
               │
               ▼
Model Training & Comparison Ladder
 ├── Logistic Regression Baseline (Interpretable only)
 ├── Decision Tree, Linear SVM, Gaussian NB, PCA, ANN (MLP 128-64)
 ├── Random Forest (300 trees, balanced subsample)
 └── Champion XGBoost (451 features, scale_pos_weight=27.46)
               │
               ▼
Operational Decisioning & Explainability
 ├── Threshold Optimization ($500 FN vs $10 FP -> 0.24 threshold)
 ├── Daily Capacity Simulation (500 reviews/day -> 79.3% capture)
 ├── Unsupervised Alert Clustering (K-Means k=4 triage lanes)
 └── TreeSHAP Reason Code Generator (src/scoring.py)
               │
               ▼
Streamlit Interactive Investigation Console (app/app.py)
 ├── Form-Based Single Transaction Predictor (29-feature model)
 ├── Batch CSV Upload Classifier (451-feature engine)
 ├── Executive Dashboard & Model Benchmarks
 ├── Prioritized Analyst Alert Queue
 ├── Transaction Deep-Dive Inspector & Waterfall Charts
 └── Dynamic Threshold & Cost Simulator
```

---

## Repository Structure

```
payments-fraud-investigation-main/
├── README.md                          Project overview, setup, and UI walkthrough (this file)
├── Learning Guide.md                  Comprehensive ML guide, college viva & interview Q&A
├── PROPOSAL.md                        Project scope, background, and target benchmarks
├── REPORT.md                          Final technical report with complete numerical tables
├── requirements.txt                   Full pinned dependencies (training, app, notebooks)
├── requirements-app.txt               Minimal runtime dependencies for Docker & Streamlit
├── pyproject.toml                     Ruff and pytest configuration
├── Makefile                           Build targets (install, lint, test, pipeline, app, docker)
├── Dockerfile                         Production container image definition
├── .streamlit/config.toml             Streamlit UI configuration
├── app/
│   ├── app.py                         6-page Streamlit web investigation console
│   ├── model.pkl                      Trained XGBoost model bundle + 0.24 threshold metadata
│   └── assets/validax-icon.svg        Dashboard branding icon
├── artifacts/
│   ├── card_history.csv               Cached card velocity summary for point-in-time inference
│   ├── inference_metadata.json        Categorical encodings and feature column order
│   ├── interpretable_model.pkl        29-feature XGBoost model for manual form predictions
│   └── interpretable_model_summary.json Performance metrics for the form model
├── src/
│   ├── data_processing.py             Raw CSV loader, cleaning, type downcasting, merge
│   ├── features.py                    Point-in-time velocity and missingness feature engine
│   ├── model.py                       Chronological split, model ladder, tuning, cost optimization
│   ├── scoring.py                     Test set scoring, risk tier assignment, SHAP reason codes
│   ├── inference.py                   Inference feature reconstructor for manual and batch rows
│   └── interpretable_model.py         Trainer for the 29-feature form model
├── sql/
│   └── fraud_analysis.sql             DuckDB analytical queries on raw transaction patterns
├── notebooks/
│   ├── 01_eda.ipynb                   Exploratory data analysis and fraud distribution plots
│   ├── 02_preprocessing.ipynb         Feature engineering inspection and leakage checks
│   ├── 03_baseline_model.ipynb        Baseline Logistic Regression vs. Champion XGBoost & SHAP
│   ├── 04_model_comparison.ipynb      Evaluation of 9 model architectures on the test split
│   └── 05_tuning_and_benchmark.ipynb  RF and XGBoost tuning, benchmark, alert clustering
├── outputs/                           Committed artifacts from the full training run
│   ├── model_summary.json             Core metrics, split sizes, and operating thresholds
│   ├── threshold_sweep.csv            Precision/recall/F1 across thresholds 0.1 to 0.9
│   ├── cost_sensitive_threshold.csv   Dollar cost curves across thresholds 0.02 to 0.96
│   ├── daily_capacity_simulation.csv  Day-by-day fraud capture at 500 reviews/day
│   ├── feature_importance.csv         Global mean |SHAP| values per feature
│   ├── scored_transactions.csv        Test set scores, risk tiers, and top 3 reason codes
│   ├── shap_summary.png               TreeSHAP beeswarm visualization
│   ├── model_comparison.csv           Leaderboard across 9 model architectures
│   ├── xgb_tuning_trials.csv          Hyperparameter search trials for XGBoost
│   ├── rf_tuning_trials.csv           Hyperparameter search trials for Random Forest
│   ├── tuning_benchmark.csv           Multi-seed and time-series CV stability benchmark
│   └── alert_clusters.csv             K-means 4-segment operational triage profiles
└── tests/                             Automated test suite (pytest)
    ├── test_app.py                    Headless Streamlit page rendering tests
    ├── test_data_processing.py        Split boundary and type conversion tests
    ├── test_features.py               Point-in-time velocity leakage prevention tests
    ├── test_inference.py              Train vs inference feature equality tests
    ├── test_model.py                  Cost-function and model evaluation tests
    └── test_scoring.py                Reason code and masked label guard tests
```

---

## Interactive Web Console Walkthrough (Page Snippets)

The Streamlit investigation console runs locally at `http://localhost:8501`. It provides six dedicated pages tailored for fraud managers, triage analysts, and risk officers.

### Page 1: Fraud Overview (Executive Dashboard & Benchmarks)
Surfaces executive-level operational KPIs, the model performance ladder, and global risk drivers.

```
========================================================================================
ValidaX | Digital Payments Risk Console                           Operating Mode: Local
Context: E-Commerce Fraud Triage | Model: XGBoost (451 Feat) | Decision Threshold: 0.24
========================================================================================
[ TEST PR-AUC ]        [ TEST ROC-AUC ]       [ OPTIMAL THRESHOLD ]     [ FRAUD CAPTURE ]
    0.4881                 0.8911                    0.24                    79.3%
 (vs Base 0.143)        (vs Base 0.757)         (Min Cost $566.8k)       (@ 500 alerts/day)
----------------------------------------------------------------------------------------
Model Performance Benchmark Ladder (Chronological Test Holdout: 118,108 Rows)
Rank  Model Name                              PR-AUC    ROC-AUC   Fit Time   Status
#1    XGBoost Champion (451 features)         0.4862    0.8855    180.0 s    DEPLOYED
#2    Feed-Forward ANN (MLP 128-64)           0.3949    0.8467     12.6 s    Candidate
#3    Decision Tree (depth=8)                 0.3369    0.8248     13.8 s    Candidate
#4    Logistic Regression (All Encoded)       0.2129    0.8356     30.5 s    Baseline
#5    Linear SVM (Calibrated)                 0.2020    0.8272    803.5 s    Candidate
#6    PCA (95% Var) + Logistic Reg            0.1671    0.8216      4.8 s    Candidate
#7    Logistic Regression (Interpretable)     0.1428    0.7577      1.2 s    Reference
#8    Gaussian Naive Bayes                    0.0869    0.7620      0.8 s    Rejected
----------------------------------------------------------------------------------------
Top 5 Global Risk Drivers (TreeSHAP Mean |Value|)
1. Purchaser Email Domain (P_emaildomain)       | [████████████████████] 0.181
2. Masked Counting Metric (C14)                | [███████████████████ ] 0.179
3. Masked Counting Metric (C13)                | [██████████████████  ] 0.177
4. Card Type (card6: credit vs. debit)         | [█████████████████   ] 0.173
5. Transaction Amount (TransactionAmt)         | [████████████████    ] 0.166
========================================================================================
```

---

### Page 2: Classify Single Transaction (Live Form Model)
Allows manual entry of transaction attributes for on-demand risk scoring. Because an investigator cannot manually type in 416 masked columns, this page uses the dedicated 29-feature form model.

```
========================================================================================
Classify Transaction | On-Demand Risk Assessment (Form Model)
========================================================================================
[ Transaction Details Input Form ]
Transaction Amount ($) : [ 420.50     ]   Product Code       : [ W (Web)          ]
Card Network           : [ visa       ]   Card Type          : [ credit           ]
Purchaser Email Domain : [ protonmail.com ]   Recipient Domain   : [ anonymous.com    ]
Billing Region (addr1) : [ 299        ]   Distance Metric    : [ 128 miles        ]
Device Type            : [ mobile     ]   Card History Count : [ 0 (First Seen)   ]
----------------------------------------------------------------------------------------
[ PREDICT FRAUD RISK ] (Click)
----------------------------------------------------------------------------------------
>>> DECISION RESULT: [ FLAG FOR INVESTIGATION - HIGH RISK ] <<<
Estimated Fraud Probability : 0.684  (Decision Threshold: 0.260)
Risk Tier                   : HIGH RISK (Tier 2 / 4)

Top SHAP Reason Codes Explaining This Risk:
(+) Purchaser uses high-risk webmail domain (protonmail.com)        [+0.24 risk impact]
(+) First transaction seen on this card code (zero past velocity)    [+0.19 risk impact]
(+) Transaction amount ($420.50) is 3.1x typical category average    [+0.14 risk impact]
(-) Transaction initiated during standard business daytime hours     [-0.04 risk impact]
========================================================================================
```

---

### Page 3: Batch Classification (CSV Upload)
Processes raw CSV uploads matching the IEEE-CIS schema and scores them using the full 451-feature XGBoost champion model.

```
========================================================================================
Batch Classification Engine | Full 451-Feature Model
========================================================================================
Upload Transaction Batch CSV: [ Drag and drop test_transactions.csv (Max 100k rows) ]
[ Upload Status: 15,000 transactions loaded successfully ]
----------------------------------------------------------------------------------------
Batch Processing Summary:
Total Processed : 15,000       Flagged for Review (Score >= 0.24) : 4,112 (27.4%)
Auto-Approved   : 10,888       Critical Tier (Score >= 0.80)      :   392  (2.6%)

Scored Output Preview:
Txn ID   Amount   Card Type  Domain          Score   Tier      Decision       Top Reason
----------------------------------------------------------------------------------------
3692014  $780.00  credit     gmail.com       0.892   Critical  FLAG REVIEW    C14 velocity spike
3692015  $ 34.20  debit      yahoo.com       0.041   Low       AUTO APPROVE   Domestic debit card
3692016  $152.00  credit     protonmail.com  0.641   High      FLAG REVIEW    Email domain mismatch
3692017  $ 89.95  debit      hotmail.com     0.118   Low       AUTO APPROVE   Consistent card history
3692018  $950.00  credit     mail.com        0.785   High      FLAG REVIEW    Large amount outlier
----------------------------------------------------------------------------------------
[ Download Full Scored Batch CSV with SHAP Reason Codes ] (Export Button)
========================================================================================
```

---

### Page 4: Alert Queue & Operational Triage
Displays prioritized transactions for fraud analysts, segmented by operational risk tiers and K-means alert clusters.

```
========================================================================================
Analyst Alert Queue | Prioritized Triage Worklist
========================================================================================
Queue Filters: [x] Critical (>=0.8)  [x] High (0.6-0.8)  [ ] Medium (0.3-0.6)  [ ] Low (<0.3)
Min Amount: [$100   ]   Card Network: [ All ]   Total Filtered Alerts: 7,545 items

Priority  Txn ID   Amount   Card Code  Score   Risk Tier  Status       Primary Reason Driver
----------------------------------------------------------------------------------------
#0001     3488219  $985.00  card1_7919 0.942   Critical   OPEN QUEUE   V258 masked spike (+0.31)
#0002     3491002  $650.00  card1_1204 0.915   Critical   OPEN QUEUE   P_emaildomain high-risk
#0003     3500124  $420.00  card1_9500 0.884   Critical   OPEN QUEUE   Velocity count C13 (+0.22)
#0004     3512890  $310.00  card1_2803 0.812   Critical   OPEN QUEUE   First-seen card ratio
----------------------------------------------------------------------------------------
K-Means Operational Alert Clusters (30,621 alerts at threshold 0.26):
Cluster ID   Alert Count   Fraud Rate   Typical Amount   Recommended Handling Lane
Lane 1       3,219 (10.5%) 28.1%        $ 43.00          Immediate Review (Mobile + Identity)
Lane 2         184 ( 0.6%) 20.1%        $113.00          New Account Verification (Zero History)
Lane 3       5,316 (17.4%) 10.6%        $ 90.00          Nighttime Off-Peak Watchlist
Lane 4      21,902 (71.5%)  8.8%        $112.00          Standard Triage Volume
========================================================================================
```

---

### Page 5: Transaction Investigation & SHAP Waterfall
Enables an analyst to conduct a deep-dive forensic audit on any single transaction ID.

```
========================================================================================
Transaction Forensic Audit | ID: #3488219
========================================================================================
Base Profile:
Transaction Time : Day 152, 03:14:20 (Night)   Transaction Amount : $985.00
Card Identity    : Visa Credit (card1: 7919)   Email Pair         : protonmail -> anonymous
Device / OS      : iOS Mobile (id_31: safari)  Billing Region     : Region 299 (Domestic)

Card Historical Context (Point-in-Time):
Prior Transactions Count : 12 txns             Average Past Amount : $64.20
Largest Past Amount      : $115.00             Time Since Last Txn : 42 seconds (Velocity Alert)

Local TreeSHAP Waterfall Risk Breakdown:
Base Model Intercept (Average Log-Odds Risk) = -3.32 (3.5% baseline)
(+) masked Vesta feature (V258) = 3.84         ------------------> [+1.42 log-odds]
(+) time since last card txn = 42s (rapid)     ------------>       [+0.98 log-odds]
(+) current amount ($985) is 15.3x card avg    --------->          [+0.85 log-odds]
(+) purchaser domain is privacy webmail        ------>             [+0.52 log-odds]
(-) domestic card matching billing address     <----               [-0.31 log-odds]
----------------------------------------------------------------------------------------
Final Calibrated Probability : 0.942 | Risk Tier : CRITICAL | Decision: FLAG & HOLD
========================================================================================
```

---

### Page 6: Threshold Simulator & Cost Optimization
Visualizes the economic and operational trade-offs of shifting the alert decision threshold.

```
========================================================================================
Threshold & Economic Cost Simulator
========================================================================================
Interactive Decision Threshold Slider: [ ====O============= ] Threshold = 0.24

Performance at Selected Threshold (0.24):
Total Alerts Generated : 32,545 items (27.6% of traffic)
Precision              : 10.8% (1 in 9 alerts is confirmed fraud)
Fraud Recall           : 86.4% (Catches 3,511 out of 4,064 test frauds)
Missed Frauds (FN)     :   553 frauds missed ($500 cost each = $276,500)
False Alarms (FP)      : 29,034 unnecessary reviews ($10 cost each = $290,340)
Total Operational Cost : $566,840

Comparison Against Default Threshold (0.50):
Metric                  Default Threshold (0.50)    Cost-Optimal (0.24)    Improvement
----------------------------------------------------------------------------------------
Alert Volume            11,106                      32,545                 +21,439 reviews
Fraud Recall            68.1%                       86.4%                  +18.3% caught
Missed Frauds           1,297                       553                    744 fewer missed
False Alarms             8,339                      29,034                 +20,695 reviews
Total Cost ($500/$10)   $731,890                    $566,840               -$165,050 (SAVED)
========================================================================================
```

---

## Local Setup & Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.
- Standard C++ compiler runtime (required by `xgboost`).
- Minimum 4 GB RAM recommended.

### 2. Virtual Environment Setup
```bash
# Clone or navigate to the repository
cd "payments-fraud-investigation-main"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install all pinned dependencies
pip install -r requirements.txt
```

### 3. Running the Streamlit Web Application
All model artifacts, scored test tables, and metadata files are already pre-computed and committed in `app/`, `artifacts/`, and `outputs/`. You can launch the dashboard immediately without retraining:

```bash
# Launch the dashboard locally
streamlit run app/app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

Alternatively, use the included shell startup script (which checks memory limits and thread configurations):
```bash
# Linux / macOS / Git Bash:
./scripts/run_app.sh

# Or on a custom port:
PORT=8600 ./scripts/run_app.sh
```

### 4. Running the Complete Training Pipeline (Optional)
If you wish to download the raw Kaggle data and re-run feature engineering and model training from scratch:
1. Accept the competition rules at [kaggle.com/competitions/ieee-fraud-detection/data](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
2. Download `train_transaction.csv` and `train_identity.csv` and place them in `data/raw/`.
3. Execute the pipeline modules in sequence:

```bash
# 1. Clean, merge, and downcast dtypes -> data/processed/transactions_clean.parquet
python src/data_processing.py

# 2. Engineer velocity, temporal, and missingness features -> data/processed/transactions_features.parquet
python src/features.py

# 3. Train models, evaluate split, run cost curves -> outputs/ + app/model.pkl
python src/model.py

# 4. Generate scored test set and TreeSHAP reason codes -> outputs/scored_transactions.csv
python src/scoring.py

# 5. Train the 29-feature manual form model -> artifacts/interpretable_model.pkl
python src/interpretable_model.py
```

### 5. Running Tests and Code Quality Audits
The test suite utilizes synthetic fixtures and pre-computed outputs, running cleanly without requiring the heavy raw Kaggle CSVs:

```bash
# Run unit and smoke tests
python -m pytest

# Run Ruff linter checks
python -m ruff check src app tests
```

---

## Docker Container Deployment

The project includes an optimized Docker setup (`Dockerfile`, Python 3.13 slim) with read-only root security and health check endpoints.

```bash
# 1. Build the production image
docker build -t validax-console .

# 2. Run the container (bounded to 4 GB RAM, read-only root, temporary scratch in tmpfs)
docker run -d --name validax \
  --memory=4g --memory-swap=4g --cpus=4 \
  --read-only --tmpfs /tmp \
  -e OMP_NUM_THREADS=4 \
  -p 8501:8501 \
  -v "$(pwd)/data/processed:/srv/fraud-console/data/processed:ro" \
  validax-console

# 3. Verify health status
curl -fsS http://localhost:8501/_stcore/health
```

---

## Model Evaluation & Ladder Comparison

All models were evaluated on the exact same holdout chronological test window (118,108 transactions, 3.44% positive fraud rate).

| Rank | Model Architecture | Features Used | Test PR-AUC | Test ROC-AUC | Fit Time | Notes |
|---|---|---|---|---|---|---|
| **#1** | **XGBoost (Champion)** | **All 451** | **0.4881** | **0.8911** | 180 s | Full training split (472k rows); deployed pipeline model |
| #2 | Feed-Forward ANN (MLP 128-64) | 626 encoded | 0.3949 | 0.8467 | 12.6 s | Early stopping at epoch 7; non-linear interactions captured |
| #3 | Decision Tree (depth=8) | 626 encoded | 0.3369 | 0.8248 | 13.8 s | Simple rule splits; out-performs every linear model |
| #4 | Logistic Regression (All Features) | 626 encoded | 0.2129 | 0.8356 | 30.5 s | Median imputed + standardized + one-hot encoded |
| #5 | Linear SVM (Calibrated) | 626 encoded | 0.2020 | 0.8272 | 803.5 s | Sigmoid calibrated on chronological validation slice |
| #6 | Logistic Regression (Top-40 Tree) | 50 encoded | 0.1704 | 0.8196 | 1.1 s | Fast feature selection retaining 80% of linear performance |
| #7 | PCA (156 Comps) + Logistic Reg | 156 principal | 0.1671 | 0.8216 | 4.8 s | Unsupervised variance compression; loses subtle fraud signals |
| #8 | Logistic Regression (Interpretable) | 29 interpretable | 0.1428 | 0.7577 | 1.2 s | Course baseline benchmark |
| #9 | Gaussian Naive Bayes | 626 encoded | 0.0869 | 0.7620 | 0.8 s | Severe failure; feature independence assumption invalid |
| -- | No-Skill Random Classifier | -- | 0.0344 | 0.5000 | -- | Matches background test fraud prevalence (3.44%) |

### Why PR-AUC Is the True Metric
While all models clustered between 0.76 and 0.89 under ROC-AUC, their PR-AUC scores diverged dramatically from 0.08 to 0.49. ROC-AUC is dominated by the massive pool of true negatives (legitimate transactions), masking thousands of false positives. PR-AUC focuses strictly on the minority fraud class, making it the only reliable indicator for financial risk ranking.

---

## Operational Decisioning: Economics & Capacity

### 1. Cost-Sensitive Threshold Analysis
Using the standard FinTech scenario parameters:
- **Cost of False Negative (Missed Fraud)**: $500 (Loss of goods, merchant fees, chargeback fines).
- **Cost of False Positive (Manual Review)**: $10 (Investigator hourly compensation and operational triage).

```
Total Scenario Cost = (False Negatives * $500) + (False Positives * $10)
```
- At the default **0.50 threshold**, total cost is **$731,890** (1,297 missed frauds cost $648.5k; 8,339 reviews cost $83.4k).
- At the cost-optimal **0.24 threshold**, total cost drops to **$566,840** (553 missed frauds cost $276.5k; 29,034 reviews cost $290.3k).
- **Net Dollar Savings**: **$165,050 (22.6% reduction in operational loss)** on the test window alone.

### 2. Fixed Daily Investigation Capacity (500 Reviews/Day)
When human investigation capacity is constrained to 500 reviews per day:
- Applying a **daily dynamic cutoff** catches on average **79.3%** of each day's fraud volume (ranging from 60.8% to 93.2% across individual days) at **15.4% average daily precision**.
- In contrast, applying a static one-shot global top-500 threshold over the entire 42-day window catches only 11.5% of total fraud volume because it concentrates review exclusively on a handful of high-volume days.

---

## Limitations & Operational Disclaimer

1. **Provider Specificity**: The dataset originates from a single payments gateway (Vesta Corporation) over a specific 6-month period. Transaction patterns and fraud vectors will not transfer without retraining.
2. **Masked Features**: Several of the strongest predictive columns are proprietary and masked. While the model exploits them with high fidelity, they cannot be interpreted beyond their structural family labels.
3. **Relative Time**: `TransactionDT` represents relative seconds from an arbitrary start point. Day-of-week and hour features are derived from modulo arithmetic rather than verified calendar timestamps.
4. **Human Review Only**: Model predictions, risk tiers, and flagged decisions represent **statistical risk indicators intended to guide human review**. They do not constitute an automated legal determination of fraud and must never be used to decline customer transactions without human oversight.

---

## License & Academic Attribution
Developed as an advanced applied machine learning capstone project (CSE303 / FinTech).
Distributed under the MIT Open Source License.
All dataset rights belong to Vesta Corporation and IEEE-CIS.
