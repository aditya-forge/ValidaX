"""Train, evaluate, and analyze the fraud model.

Pipeline:
  1. Chronological train/test split on TransactionDT (first 80% -> train,
     last 20% -> test) -- NOT a random split, so the model is evaluated on
     transactions strictly *after* everything it trained on, which is the
     realistic way to validate a fraud model meant to score future traffic.
  2. Baseline: Logistic Regression on the interpretable feature set only.
  3. Main model: XGBoost on the full feature set (interpretable + masked
     C/D/M/V/id_ columns as opaque inputs), using native categorical support
     and native missing-value handling.
  4. Threshold sweep (precision/recall/alert volume at each cutoff).
  5. Fixed-capacity analysis: Precision@K / Fraud-Capture@K for a
     500-alert/day investigator review capacity, simulated per day across
     the test period.
  6. Cost-sensitive threshold: minimizes an assumed cost function
     (false negative = $500, false positive review cost = $10 -- explicitly
     scenario assumptions, not real industry costs).
  7. SHAP global feature importance + per-transaction local explanations.

It also holds the model-comparison, tuning, and clustering helpers used by
notebooks/04_model_comparison.ipynb and notebooks/05_tuning_and_benchmark.ipynb,
so the notebooks stay narrative and the logic lives in one place.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from features import (
    INTERPRETABLE_MODEL_FEATURES,
    get_model_feature_columns,
)

DATA_PATH = Path("data/processed/transactions_features.parquet")
MODEL_PATH = Path("app/model.pkl")
OUTPUTS_DIR = Path("outputs")

RANDOM_STATE = 42  # single seed for sampling, every model, KMeans and the tuning search
TRAIN_FRAC = 0.8

# Tuning never replaces the deployed app/model.pkl unless this is switched on
# (see promote_tuned_model and the matching flag in notebook 05).
PROMOTE_TUNED_MODEL = False
FALSE_NEGATIVE_COST = 500  # scenario assumption, not a real industry figure
FALSE_POSITIVE_COST = 10   # scenario assumption, not a real industry figure
DAILY_REVIEW_CAPACITY = 500

# The comparison models (SVM, ANN, ...) don't scale to 470k rows x 450
# features the way XGBoost does, so they're fit on a random sample of the
# training window. Sampling only from train keeps the test window untouched.
COMPARISON_SAMPLE_SIZE = 150_000

DEFAULT_XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_lambda": 1.5,
}

XGB_SEARCH_SPACE = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.02, 0.05, 0.1],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.4, 0.5, 0.7],
    "reg_lambda": [1.0, 1.5, 3.0, 5.0],
}

# Interpretable-only inputs for alert clustering, so every cluster profile
# can be described to an investigator without touching masked columns.
CLUSTER_FEATURES = [
    "log_amount",
    "transaction_hour",
    "is_night",
    "purchaser_email_is_free",
    "has_identity_data",
    "is_mobile_device",
    "card_prior_txn_count",
    "amount_vs_card_avg_ratio",
    "card_is_first_seen",
    "seconds_since_card_last_txn",
]
_CLUSTER_LOG_COLUMNS = (
    "card_prior_txn_count",
    "amount_vs_card_avg_ratio",
    "seconds_since_card_last_txn",
)


def chronological_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    cutoff = int(len(df) * train_frac)
    train, test = df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
    print(
        f"Train: {len(train):,} rows (days {train['transaction_day'].min()}-{train['transaction_day'].max()}), "
        f"fraud rate {train['isFraud'].mean():.4%}"
    )
    print(
        f"Test:  {len(test):,} rows (days {test['transaction_day'].min()}-{test['transaction_day'].max()}), "
        f"fraud rate {test['isFraud'].mean():.4%}"
    )
    return train, test


def split_feature_types(df: pd.DataFrame, feature_cols: list) -> tuple[list, list]:
    """Return (numeric_cols, categorical_cols) for the given feature list."""
    cat_cols = [c for c in feature_cols if df[c].dtype.name in ("object", "category")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return num_cols, cat_cols


def make_preprocessor(num_cols: list, cat_cols: list, dense: bool = False) -> ColumnTransformer:
    """Median-impute + scale numerics, constant-impute + one-hot categoricals.

    `dense=True` forces a dense float32 output, which GaussianNB, PCA and the
    MLP need; the logistic-regression baseline keeps the sparse default.
    """
    onehot_kwargs = {"handle_unknown": "ignore", "max_categories": 20}
    if dense:
        onehot_kwargs.update(sparse_output=False, dtype=np.float32)
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(**onehot_kwargs)),
            ]), cat_cols),
        ],
        sparse_threshold=0.0 if dense else 0.3,
    )


def score_probabilities(y_true, proba) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def train_baseline_logreg(train: pd.DataFrame, test: pd.DataFrame):
    feature_cols = [c for c in INTERPRETABLE_MODEL_FEATURES if c in train.columns]
    num_cols, cat_cols = split_feature_types(train, feature_cols)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("clf", clf)])

    X_train, y_train = train[feature_cols], train["isFraud"]
    X_test, y_test = test[feature_cols], test["isFraud"]

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = score_probabilities(y_test, proba)
    print(f"[Logistic Regression baseline] ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")
    return pipe, proba, metrics


def make_xgb(scale_pos_weight: float, **overrides) -> xgb.XGBClassifier:
    params = {**DEFAULT_XGB_PARAMS, **overrides}
    return xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _scale_pos_weight(y: pd.Series) -> float:
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list, params: dict = None):
    X_train, y_train = train[feature_cols], train["isFraud"]
    X_test, y_test = test[feature_cols], test["isFraud"]

    model = make_xgb(_scale_pos_weight(y_train), **(params or {}))
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = score_probabilities(y_test, proba)
    print(f"[XGBoost] ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")
    return model, proba, metrics


# ------------------------------------------------------- model comparison --
def comparison_models(random_state: int = RANDOM_STATE) -> dict:
    """The classical / neural models benchmarked against the XGBoost champion.

    Every model sees the same preprocessed matrix (median-imputed, scaled
    numerics + capped one-hot categoricals), so differences in PR-AUC come
    from the learner, not from feature handling.
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=200, class_weight="balanced",
            random_state=random_state,
        ),
        "PCA + Logistic Regression": Pipeline([
            ("pca", PCA(n_components=0.95, random_state=random_state)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Gaussian Naive Bayes": GaussianNB(),
        # LinearSVC has no predict_proba; sigmoid calibration gives scores
        # that PR-AUC, thresholds and the cost curve can all use.
        "Linear SVM (calibrated)": CalibratedClassifierCV(
            LinearSVC(C=0.05, class_weight="balanced", random_state=random_state),
            method="sigmoid", cv=3,
        ),
        "Feed-forward ANN (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=1024,
            learning_rate_init=1e-3, max_iter=50, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=5, random_state=random_state,
        ),
    }


def train_comparison_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list,
    sample_size: int = COMPARISON_SAMPLE_SIZE,
    models: dict = None,
    random_state: int = RANDOM_STATE,
):
    """Fit each comparison model on a train-window sample, score the full test window.

    Returns (results DataFrame, {name: test probabilities}, fitted preprocessor,
    {name: fitted model}).
    """
    models = models if models is not None else comparison_models(random_state)
    fit_df = train.sample(n=min(sample_size, len(train)), random_state=random_state)

    num_cols, cat_cols = split_feature_types(fit_df, feature_cols)
    prep = make_preprocessor(num_cols, cat_cols, dense=True)
    X_fit = prep.fit_transform(fit_df[feature_cols]).astype(np.float32, copy=False)
    X_test = prep.transform(test[feature_cols]).astype(np.float32, copy=False)
    y_fit, y_test = fit_df["isFraud"].to_numpy(), test["isFraud"].to_numpy()
    print(f"Comparison matrix: {X_fit.shape[0]:,} train rows x {X_fit.shape[1]} encoded columns")

    rows, probas, fitted = [], {}, {}
    for name, estimator in models.items():
        start = time.perf_counter()
        estimator.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - start
        proba = estimator.predict_proba(X_test)[:, 1]

        metrics = score_probabilities(y_test, proba)
        rows.append({"model": name, **metrics, "train_rows": len(fit_df), "fit_seconds": fit_seconds})
        probas[name], fitted[name] = proba, estimator
        print(f"[{name}] ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
              f"({fit_seconds:,.0f}s)")

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return results, probas, prep, fitted


# ----------------------------------------------------------------- tuning --
def tune_xgboost(
    train: pd.DataFrame,
    feature_cols: list,
    n_iter: int = 12,
    val_frac: float = 0.2,
    max_estimators: int = 2000,
    early_stopping_rounds: int = 100,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Random search over XGB_SEARCH_SPACE, scored on a chronological validation slice.

    The validation slice is the *last* `val_frac` of the training window, so
    tuning never looks at the test window and still respects time order.
    Returns one row per trial, best validation PR-AUC first.
    """
    fit, val = chronological_split(train, train_frac=1 - val_frac)
    X_fit, y_fit = fit[feature_cols], fit["isFraud"]
    X_val, y_val = val[feature_cols], val["isFraud"]
    spw = _scale_pos_weight(y_fit)

    rng = np.random.default_rng(random_state)
    rows = []
    for trial in range(n_iter):
        params = {k: v[rng.integers(len(v))] for k, v in XGB_SEARCH_SPACE.items()}
        model = make_xgb(
            spw, n_estimators=max_estimators,
            early_stopping_rounds=early_stopping_rounds, **params,
        )
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        # The sklearn wrapper predicts with the best iteration once early stopping fires.
        metrics = score_probabilities(y_val, model.predict_proba(X_val)[:, 1])
        rows.append({
            "trial": trial, **params, "best_iteration": int(model.best_iteration),
            "val_roc_auc": metrics["roc_auc"], "val_pr_auc": metrics["pr_auc"],
        })
        print(f"trial {trial:>2}: val PR-AUC={metrics['pr_auc']:.4f}  "
              f"trees={model.best_iteration + 1}  {params}")

    return pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)


def best_params_from_search(search: pd.DataFrame, val_frac: float = 0.2) -> dict:
    """Turn the top search row into params for a refit on the full training window.

    The tree count is scaled up by 1 / (1 - val_frac) because the refit sees
    proportionally more data than the early-stopped trial did.
    """
    # Read column-wise: a row Series of mixed numerics upcasts ints to float,
    # and XGBoost rejects a float max_depth.
    params = {k: search[k].iloc[0].item() for k in XGB_SEARCH_SPACE}
    best_iteration = int(search["best_iteration"].iloc[0])
    params["n_estimators"] = int(round((best_iteration + 1) / (1 - val_frac)))
    return params


# ------------------------------------------------------------- clustering --
def cluster_matrix(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Interpretable cluster inputs with heavy-tailed counts/ratios log-scaled.

    seconds_since_card_last_txn uses -1 for "no prior transaction", so it is
    clipped at 0 before the log transform; card_is_first_seen carries that
    information separately.
    """
    cols = [c for c in (cols or CLUSTER_FEATURES) if c in df.columns]
    X = df[cols].astype("float64").copy()
    for col in _CLUSTER_LOG_COLUMNS:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))
    return X


def kmeans_k_sweep(X: pd.DataFrame, ks=range(2, 9), sample_size: int = 10_000,
                   random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Inertia + (sampled) silhouette score per k, for choosing the cluster count."""
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    Xs = prep.fit_transform(X)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Xs)
        sil = silhouette_score(
            Xs, km.labels_, sample_size=min(sample_size, len(Xs)), random_state=random_state,
        )
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def cluster_alerts(df: pd.DataFrame, n_clusters: int = 5, cols: list = None,
                   random_state: int = RANDOM_STATE):
    """K-means segments of alert-tier transactions on interpretable features.

    Returns (labels, per-cluster profile, fitted pipeline). The profile shows
    each segment's size, observed fraud rate and feature means in original
    units, so segments can be named in investigator terms.
    """
    X = cluster_matrix(df, cols)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)),
    ])
    labels = pipe.fit_predict(X)

    raw_cols = [c for c in (cols or CLUSTER_FEATURES) if c in df.columns]
    profile = df[raw_cols].assign(cluster=labels)
    agg = {c: "mean" for c in raw_cols}
    if "isFraud" in df.columns:
        profile["isFraud"] = df["isFraud"].to_numpy()
        agg["isFraud"] = "mean"
    profile = (
        profile.groupby("cluster")
        .agg(**{"transactions": (raw_cols[0], "size")}, **{c: (c, f) for c, f in agg.items()})
        .rename(columns={"isFraud": "fraud_rate"})
        .sort_values("fraud_rate" if "isFraud" in df.columns else "transactions", ascending=False)
    )
    return labels, profile, pipe


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "alerts": int(preds.sum()),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def threshold_sweep(y_true, proba, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.1)
    rows = [metrics_at_threshold(y_true, proba, t) for t in thresholds]
    return pd.DataFrame(rows)


def capacity_at_k(y_true: np.ndarray, proba: np.ndarray, k: int) -> dict:
    order = np.argsort(-proba)[:k]
    top_y = np.asarray(y_true)[order]
    fraud_total = int(np.asarray(y_true).sum())
    captured = int(top_y.sum())
    return {
        "k": k,
        "fraud_captured": captured,
        "fraud_total": fraud_total,
        "precision_at_k": captured / k if k else 0.0,
        "recall_at_k": captured / fraud_total if fraud_total else 0.0,
    }


def daily_capacity_simulation(
    test: pd.DataFrame, proba: np.ndarray, capacity: int = DAILY_REVIEW_CAPACITY
) -> pd.DataFrame:
    """Simulate investigators who can only review `capacity` alerts per day.

    Each day, the top-`capacity` scored transactions are "reviewed"; fraud
    outside that cutoff is missed that day. This is a more realistic
    operational view than a single global top-K cut, since it's bound by a
    per-day review capacity rather than a capacity over the whole test window.
    """
    t = test[["transaction_day", "isFraud"]].copy()
    t["fraud_probability"] = proba
    rows = []
    for day, grp in t.groupby("transaction_day"):
        grp_sorted = grp.sort_values("fraud_probability", ascending=False)
        reviewed = grp_sorted.head(capacity)
        rows.append({
            "day": day,
            "transactions": len(grp),
            "fraud_in_day": int(grp["isFraud"].sum()),
            "alerts_reviewed": len(reviewed),
            "fraud_captured": int(reviewed["isFraud"].sum()),
            "precision": reviewed["isFraud"].mean() if len(reviewed) else 0.0,
        })
    daily = pd.DataFrame(rows).sort_values("day")
    daily["recall"] = daily["fraud_captured"] / daily["fraud_in_day"].replace(0, np.nan)
    return daily


def cost_sensitive_threshold(y_true, proba, fn_cost=FALSE_NEGATIVE_COST, fp_cost=FALSE_POSITIVE_COST) -> pd.DataFrame:
    thresholds = np.arange(0.02, 0.98, 0.02)
    rows = []
    for t in thresholds:
        m = metrics_at_threshold(y_true, proba, t)
        total_cost = m["false_negatives"] * fn_cost + m["false_positives"] * fp_cost
        rows.append({**m, "total_cost": total_cost})
    return pd.DataFrame(rows)


def risk_tier(p: float) -> str:
    if p >= 0.8:
        return "Critical"
    elif p >= 0.6:
        return "High"
    elif p >= 0.3:
        return "Medium"
    return "Low"


def compute_shap(model: xgb.XGBClassifier, X: pd.DataFrame, sample_size: int = None):
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=RANDOM_STATE)
    else:
        X_sample = X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values, X_sample


def global_feature_importance(shap_values, feature_names: list) -> pd.DataFrame:
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    mean_abs = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


# ------------------------------------------------------- model bundle I/O --
def build_model_bundle(
    xgb_model: xgb.XGBClassifier,
    logreg_pipe: Pipeline,
    feature_cols: list,
    cost_optimal_threshold: float,
    interpretable_cols: list = None,
    xgb_params: dict = None,
) -> dict:
    """The dict saved to app/model.pkl and read by src/scoring.py and app/app.py."""
    bundle = {
        "xgb_model": xgb_model,
        "logreg_pipe": logreg_pipe,
        "feature_cols": list(feature_cols),
        "interpretable_cols": (
            list(interpretable_cols) if interpretable_cols is not None
            else [c for c in INTERPRETABLE_MODEL_FEATURES if c in feature_cols]
        ),
        "cost_optimal_threshold": float(cost_optimal_threshold),
    }
    if xgb_params is not None:
        bundle["xgb_params"] = dict(xgb_params)
    return bundle


def save_model_bundle(bundle: dict, path: Path = MODEL_PATH, overwrite: bool = False) -> Path:
    """Write a model bundle, refusing to replace an existing file unless asked to."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace the deployed model."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"Saved model bundle -> {path}")
    return path


def promote_tuned_model(bundle: dict, path: Path = MODEL_PATH,
                        enabled: bool = PROMOTE_TUNED_MODEL) -> bool:
    """Replace the deployed bundle with a tuned one, only when explicitly enabled.

    Off by default (PROMOTE_TUNED_MODEL = False) so tuning runs never
    overwrite app/model.pkl by accident. Returns True if the bundle was saved.
    """
    if not enabled:
        print(f"Tuned model NOT promoted (PROMOTE_TUNED_MODEL is off); {path} left unchanged.")
        return False
    save_model_bundle(bundle, path, overwrite=True)
    return True


def main():
    df = pd.read_parquet(DATA_PATH)
    train, test = chronological_split(df)

    feature_cols = get_model_feature_columns(df, include_masked=True)
    print(f"Training XGBoost on {len(feature_cols)} features "
          f"({len(INTERPRETABLE_MODEL_FEATURES)} interpretable + masked families)")

    logreg_pipe, logreg_proba, logreg_metrics = train_baseline_logreg(train, test)
    xgb_model, xgb_proba, xgb_metrics = train_xgboost(train, test, feature_cols)

    y_test = test["isFraud"].values

    OUTPUTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    # --- threshold sweep ---
    sweep = threshold_sweep(y_test, xgb_proba)
    sweep.to_csv(OUTPUTS_DIR / "threshold_sweep.csv", index=False)
    print("\nThreshold sweep:\n", sweep.round(4).to_string(index=False))

    # --- fixed capacity (single global top-K, as a headline number) ---
    cap500 = capacity_at_k(y_test, xgb_proba, k=DAILY_REVIEW_CAPACITY)
    print(f"\nFraud Capture@{DAILY_REVIEW_CAPACITY}: {cap500}")

    # --- realistic per-day capacity simulation ---
    daily = daily_capacity_simulation(test, xgb_proba, capacity=DAILY_REVIEW_CAPACITY)
    daily.to_csv(OUTPUTS_DIR / "daily_capacity_simulation.csv", index=False)

    # --- cost-sensitive threshold ---
    cost_curve = cost_sensitive_threshold(y_test, xgb_proba)
    cost_curve.to_csv(OUTPUTS_DIR / "cost_sensitive_threshold.csv", index=False)
    best_row = cost_curve.loc[cost_curve["total_cost"].idxmin()]
    print(f"\nCost-minimizing threshold: {best_row['threshold']:.2f} "
          f"(total scenario cost ${best_row['total_cost']:,.0f}, "
          f"precision={best_row['precision']:.3f}, recall={best_row['recall']:.3f})")

    # --- SHAP ---
    X_test = test[feature_cols]
    explainer, shap_values, X_sample = compute_shap(xgb_model, X_test, sample_size=20000)
    importance = global_feature_importance(shap_values, feature_cols)
    importance.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    print("\nTop 15 features by mean |SHAP|:\n", importance.head(15).to_string(index=False))

    # --- persist artifacts ---
    # Running this script is an explicit retrain, so it replaces the deployed bundle.
    bundle = build_model_bundle(
        xgb_model, logreg_pipe, feature_cols, best_row["threshold"],
        interpretable_cols=[c for c in INTERPRETABLE_MODEL_FEATURES if c in df.columns],
    )
    save_model_bundle(bundle, MODEL_PATH, overwrite=True)

    summary = {
        "logreg": logreg_metrics,
        "xgboost": xgb_metrics,
        "fraud_capture_at_500": cap500,
        "cost_optimal_threshold": float(best_row["threshold"]),
        "cost_optimal_precision": float(best_row["precision"]),
        "cost_optimal_recall": float(best_row["recall"]),
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features": len(feature_cols),
    }
    with open(OUTPUTS_DIR / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model_summary.json -> {OUTPUTS_DIR / 'model_summary.json'}")

    return {
        "train": train, "test": test, "xgb_model": xgb_model, "xgb_proba": xgb_proba,
        "feature_cols": feature_cols, "explainer": explainer, "importance": importance,
    }


if __name__ == "__main__":
    main()
