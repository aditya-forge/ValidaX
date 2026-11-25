"""Fraud risk operations console -- Streamlit dashboard.

Four pages: Fraud overview, Alert queue, Transaction investigation and
Threshold simulator. Built on the artifacts produced by src/model.py and
src/scoring.py; launch from the repo root with `streamlit run app/app.py`.

Data note: IEEE-CIS Fraud Detection is real, anonymized e-commerce
transaction data (not synthetic). TransactionDT is a relative time offset
from an undisclosed reference point, not a calendar timestamp. Many columns
(C1-C14, D1-D15, M1-M9, V1-V339, id_*) are masked by the data provider
(Vesta); this console never claims to know their meaning. They are shown with
generic family labels only, and their raw values are never displayed.

Operational context (FinTech / CSE303 F20): this is a transaction risk
prioritization console. A score indicates elevated risk warranting review,
not a determination of fraud. Cost and capacity figures are scenario
assumptions imported from src/model.py.

Reads: app/model.pkl and outputs/ (never written). Optional results from
notebooks 04/05 are shown when present and skipped with a note when not.
"""
import collections
import html
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import (  # noqa: E402
    DAILY_REVIEW_CAPACITY,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    chronological_split,
)
from scoring import FEATURE_LABELS, is_masked_column, label_for_feature, reason_text  # noqa: E402

OUTPUTS_DIR = ROOT / "outputs"
SHAP_SUMMARY_PATH = OUTPUTS_DIR / "shap_summary.png"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
FEATURES_PATH = ROOT / "data" / "processed" / "transactions_features.parquet"

# Core artifacts from src/model.py + src/scoring.py: the console needs all of them.
CORE_OUTPUTS = {
    "scored_transactions.csv": "python src/scoring.py",
    "model_summary.json": "python src/model.py",
    "threshold_sweep.csv": "python src/model.py",
    "cost_sensitive_threshold.csv": "python src/model.py",
    "daily_capacity_simulation.csv": "python src/model.py",
    "feature_importance.csv": "python src/model.py",
}
# Optional results from notebooks 04/05: shown when present.
OPTIONAL_OUTPUTS = {
    "model_comparison.csv": "notebooks/04_model_comparison.ipynb",
    "tuning_benchmark.csv": "notebooks/05_tuning_and_benchmark.ipynb",
    "alert_clusters.csv": "notebooks/05_tuning_and_benchmark.ipynb",
}

PAGES = ["Fraud overview", "Alert queue", "Transaction investigation", "Threshold simulator"]
TIERS = ["Critical", "High", "Medium", "Low"]

st.set_page_config(page_title="Fraud risk operations console", layout="wide")

# ---------------------------------------------------------------- design --
INK = "#1B2430"
MUTED = "#5B6676"
RULE = "#D5DBE3"
PANEL = "#F4F6F9"
DATA = "#2F5D8A"
RISK_COLORS = {"Critical": "#B42318", "High": "#D9730D", "Medium": "#C9A227", "Low": "#8A94A6"}
INCREASES_RISK, DECREASES_RISK = RISK_COLORS["Critical"], DATA
FONT_STACK = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp {{ font-family: {FONT_STACK}; color: {INK}; font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top: 1.6rem; max-width: 1440px; }}
h1 {{ font-size: 1.55rem !important; font-weight: 600 !important; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.12rem !important; font-weight: 600 !important; }}
h3 {{ font-size: 0.98rem !important; font-weight: 600 !important; }}
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {RULE}; }}
[data-testid="stMetric"] {{
    border: 1px solid {RULE}; border-radius: 2px; padding: 0.55rem 0.8rem; background: #FFFFFF;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.8rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.4rem; font-weight: 600; color: {INK}; }}
[data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stPlotlyChart"] {{
    border: 1px solid {RULE}; border-radius: 2px;
}}
[data-testid="stExpander"] details {{ border: 1px solid {RULE}; border-radius: 2px; }}
hr {{ border-color: {RULE} !important; }}
.ctx {{
    border: 1px solid {RULE}; border-left: 3px solid {DATA}; border-radius: 2px;
    background: {PANEL}; padding: 0.55rem 0.85rem; margin-bottom: 1rem;
    font-size: 0.86rem; line-height: 1.5; color: {INK};
}}
.ctx strong {{ font-weight: 600; }}
.reasons {{ margin: 0.2rem 0 0 0; padding-left: 1.1rem; font-size: 0.9rem; line-height: 1.6; }}
.reasons .up {{ color: {INCREASES_RISK}; }}
.reasons .down {{ color: {DECREASES_RISK}; }}
</style>
"""

PLOT_LAYOUT = dict(
    font=dict(family=FONT_STACK, size=12, color=INK),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=12, r=12, t=44, b=12),
    title=dict(x=0.01, xanchor="left", font=dict(size=14, color=INK)),
    xaxis=dict(gridcolor="#E6EAF0", linecolor=RULE, zeroline=False),
    yaxis=dict(gridcolor="#E6EAF0", linecolor=RULE, zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    hoverlabel=dict(font_family=FONT_STACK),
)


def styled(fig: go.Figure, title: str, height: int = 340, **layout) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_layout(title_text=title, height=height, **layout)
    return fig


def md_safe(text: str) -> str:
    """Escape '$' so Streamlit markdown never renders amounts as LaTeX."""
    return str(text).replace("$", "\\$")


def html_safe(text: str) -> str:
    """HTML-escape text and encode '$' as an entity for st.html blocks."""
    return html.escape(str(text)).replace("$", "&#36;")


def operational_context_banner() -> None:
    st.html(
        '<div class="ctx"><strong>FinTech / CSE303 (F20) · Transaction risk prioritization '
        "console.</strong> Scores rank transactions for investigator review: a high score "
        "indicates <strong>elevated risk warranting review, not a determination of fraud</strong>. "
        f"Scenario assumptions: missed fraud &#36;{FALSE_NEGATIVE_COST:,}, false-positive review "
        f"&#36;{FALSE_POSITIVE_COST:,}; review capacity {DAILY_REVIEW_CAPACITY:,} alerts/day. "
        "Masked C/D/M/V/id_ columns appear with generic family labels only.</div>"
    )


# ------------------------------------------------------ feature labelling --
# src/scoring.py owns the labels: masked columns map to their family label
# ("masked counting feature (C14)") and never carry a value in reason codes.
display_label = label_for_feature

# Reason codes stored in scored_transactions.csv by an earlier scoring run may
# use older time wording; map it to the current relative-time labels.
LEGACY_LABELS = {
    "hour of day": FEATURE_LABELS["transaction_hour"],
    "day of week": FEATURE_LABELS["transaction_dow"],
    "nighttime transaction (12am-6am)": FEATURE_LABELS["is_night"],
    "weekend transaction": FEATURE_LABELS["is_weekend"],
}
_MASKED_REASON = re.compile(
    r"^(?P<label>masked [A-Za-z/ -]+ \((?:[CDMV]\d+|id_\w+)\)).*?(?P<dir>increased|decreased) risk$"
)
_RAW_MASKED_REASON = re.compile(r"^(?P<col>[CDMV]\d+|id_\w+)\b.*?(?P<dir>increased|decreased) risk$")


def safe_reason(text) -> str:
    """Normalise a stored reason code so masked columns show a family label only."""
    if not isinstance(text, str) or not text.strip():
        return ""
    match = _MASKED_REASON.match(text)
    if match:  # drop anything between the family label and the direction
        return f"{match['label']} {match['dir']} risk"
    match = _RAW_MASKED_REASON.match(text)
    if match and is_masked_column(match["col"]):
        return f"{display_label(match['col'])} {match['dir']} risk"
    for old, new in LEGACY_LABELS.items():
        if text.startswith(old):
            return new + text[len(old):]
    return text


# ------------------------------------------------------------ card history --
# card1 velocity features use -1 for "no prior transaction". A first-seen card
# also has no prior average, so its amount-vs-typical ratio equals the raw
# amount and is not a meaningful comparison.
NO_PRIOR_TXN = "No prior transaction (first seen)"
FIRST_SEEN_RATIO_NOTE = "not meaningful: first-seen card has no prior average"


def humanize_seconds(seconds) -> str:
    if pd.isna(seconds):
        return "missing"
    seconds = float(seconds)
    if seconds < 0:
        return NO_PRIOR_TXN
    if seconds < 60:
        return f"{seconds:,.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:,.0f} min"
    if seconds < 86400:
        return f"{seconds / 3600:,.1f} h"
    return f"{seconds / 86400:,.1f} days"


def card_is_first_seen(raw_row) -> bool:
    flag = raw_row.get("card_is_first_seen")
    if pd.notna(flag):
        return bool(flag == 1)
    gap = raw_row.get("seconds_since_card_last_txn")
    return pd.notna(gap) and float(gap) < 0


def investigation_reason(feature: str, shap_value: float, raw_value, first_seen: bool) -> str:
    """Reason code with card-history placeholders spelled out; masked handling stays in scoring.py."""
    direction = "increased" if shap_value > 0 else "decreased"
    label = display_label(feature)
    if feature == "seconds_since_card_last_txn" and pd.notna(raw_value) and float(raw_value) < 0:
        return f"{label}: {NO_PRIOR_TXN.lower()}; {direction} risk"
    if feature == "card_is_first_seen" and pd.notna(raw_value):
        return f"{label} ({'yes' if raw_value == 1 else 'no'}) {direction} risk"
    if first_seen and feature == "amount_vs_card_avg_ratio":
        return f"{label} ({FIRST_SEEN_RATIO_NOTE}) {direction} risk"
    if first_seen and feature in ("card_avg_amount_prior", "card_max_amount_prior"):
        return f"{label} (none, first seen) {direction} risk"
    return reason_text(feature, shap_value, raw_value)


# ---------------------------------------------------------------- data I/O --
def missing_core_outputs() -> list:
    return [name for name in CORE_OUTPUTS if not (OUTPUTS_DIR / name).is_file()]


@st.cache_data
def load_scored() -> pd.DataFrame:
    scored = pd.read_csv(OUTPUTS_DIR / "scored_transactions.csv", low_memory=False)
    for col in ("top_reason_1", "top_reason_2", "top_reason_3"):
        if col in scored.columns:
            scored[col] = scored[col].map(safe_reason)
    return scored


@st.cache_data
def load_summary() -> dict:
    with open(OUTPUTS_DIR / "model_summary.json") as f:
        return json.load(f)


@st.cache_data
def load_output_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / name)


@st.cache_data
def load_optional_csv(name: str):
    """Return (frame, None) if the optional artifact loads, else (None, reason)."""
    path, notebook = OUTPUTS_DIR / name, OPTIONAL_OUTPUTS[name]
    if not path.is_file():
        return None, f"Not available yet: run `{notebook}` to create `outputs/{name}`."
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # unreadable or half-written file
        return None, f"Couldn't read `outputs/{name}` ({type(exc).__name__}). Re-run `{notebook}`."
    if frame.empty:
        return None, f"`outputs/{name}` is empty. Re-run `{notebook}`."
    return frame, None


# Optional results (notebooks 04/05) are shown read-only, with masked columns
# removed and any masked name in a text cell replaced by its family label.
def drop_masked_columns(frame: pd.DataFrame) -> tuple:
    masked = [c for c in frame.columns if is_masked_column(str(c))]
    return frame.drop(columns=masked), masked


def label_masked_cells(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].map(lambda v: display_label(v) if isinstance(v, str) and is_masked_column(v) else v)
    return out


def mean_pm_std(frame: pd.DataFrame, prefix: str, digits: int = 4):
    """'0.4880 ± 0.0021' strings from <prefix>_mean / <prefix>_std, or None if absent."""
    if f"{prefix}_mean" not in frame:
        return None
    std = frame[f"{prefix}_std"] if f"{prefix}_std" in frame else pd.Series(float("nan"), index=frame.index)
    return [
        "–" if pd.isna(m) else (f"{m:.{digits}f}" if pd.isna(s) else f"{m:.{digits}f} ± {s:.{digits}f}")
        for m, s in zip(frame[f"{prefix}_mean"], std, strict=True)
    ]


def render_model_comparison(comparison: pd.DataFrame) -> None:
    table, _ = drop_masked_columns(label_masked_cells(comparison))
    keep = [c for c in ("model", "pr_auc", "roc_auc", "train_rows", "fit_seconds") if c in table]
    if "pr_auc" in table:
        table = table.sort_values("pr_auc", ascending=False)
    st.dataframe(
        table[keep], hide_index=True, width="stretch",
        column_config={
            "model": "Model",
            "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
            "train_rows": st.column_config.NumberColumn("Training rows", format="localized"),
            "fit_seconds": st.column_config.NumberColumn("Fit time (s)", format="%.0f"),
        },
    )
    st.caption("Test window, chronological split. Comparison models were fit on a training-window "
               "sample; the baseline and XGBoost on the full training window.")


BENCHMARK_COLUMNS = [  # (column in tuning_benchmark.csv, header, number format)
    ("val_pr_auc", "Validation PR-AUC", "%.4f"),
    ("cost_optimal_threshold", "Cost-optimal threshold", "%.2f"),
    ("precision_at_cost_opt", "Precision at threshold", "percent"),
    ("recall_at_cost_opt", "Recall at threshold", "percent"),
    ("scenario_cost_at_cost_opt", "Scenario cost", "dollar"),
    ("avg_daily_recall_at_500", f"Daily capture at {DAILY_REVIEW_CAPACITY}/day", "percent"),
    ("avg_daily_precision_at_500", f"Daily precision at {DAILY_REVIEW_CAPACITY}/day", "percent"),
]


def render_tuning_benchmark(benchmark: pd.DataFrame) -> None:
    source, _ = drop_masked_columns(label_masked_cells(benchmark))
    view = pd.DataFrame({"Model": source.get("model", pd.Series(dtype=str))})
    if "selected" in source:
        view["Selected"] = source["selected"].map(lambda v: "yes" if str(v).strip().lower() in ("true", "1") else "")
    for prefix, header in [("test_pr_auc", "Test PR-AUC, mean ± std"),
                           ("test_roc_auc", "Test ROC-AUC, mean ± std"),
                           ("cv_pr_auc", "Forward-chaining CV PR-AUC, mean ± std")]:
        values = mean_pm_std(source, prefix)
        if values is not None:
            view[header] = values
    config = {}
    for col, header, fmt in BENCHMARK_COLUMNS:
        if col in source:
            view[header] = pd.to_numeric(source[col], errors="coerce")
            config[header] = st.column_config.NumberColumn(header, format=fmt)
    st.dataframe(view, hide_index=True, width="stretch", column_config=config)
    st.caption("Scenario cost uses the assumed missed-fraud and false-positive costs; "
               "mean ± std across seeds (test) and forward-chaining folds (CV).")


CLUSTER_FORMATS = {  # profile columns from notebook 05; shares of 0/1 flags show as percent
    "log_amount": "%.2f", "transaction_hour": "%.1f", "is_night": "percent",
    "purchaser_email_is_free": "percent", "has_identity_data": "percent", "is_mobile_device": "percent",
    "card_prior_txn_count": "%.1f", "card_is_first_seen": "percent",
}


def segment_view(table: pd.DataFrame) -> pd.DataFrame:
    """Readable card-history columns: the -1 placeholder and first-seen ratios become text."""
    view = table.copy()
    first_seen = view["card_is_first_seen"] if "card_is_first_seen" in view else pd.Series(0.0, index=view.index)
    if "seconds_since_card_last_txn" in view:
        view["seconds_since_card_last_txn"] = [
            NO_PRIOR_TXN if share >= 0.999
            else humanize_seconds(s) + (" (mean pulled down by first-seen cards)" if share > 0 else "")
            for s, share in zip(view["seconds_since_card_last_txn"], first_seen, strict=True)
        ]
    if "amount_vs_card_avg_ratio" in view:
        view["amount_vs_card_avg_ratio"] = [
            f"n/a ({FIRST_SEEN_RATIO_NOTE})" if share >= 0.999
            else f"{r:,.2f}x" + (" (inflated by first-seen cards)" if share > 0 else "")
            for r, share in zip(view["amount_vs_card_avg_ratio"], first_seen, strict=True)
        ]
    return view


def render_alert_segments(clusters: pd.DataFrame, threshold=None) -> None:
    table, dropped = drop_masked_columns(clusters)  # segments are described without masked columns
    if "fraud_rate" in table:
        table = table.sort_values("fraud_rate", ascending=False)
    table = table.reset_index(drop=True)
    table.insert(0, "segment", range(1, len(table) + 1))  # ranked by fraud rate; K-means ids are arbitrary
    table = segment_view(table.drop(columns=[c for c in ("cluster",) if c in table]))

    alerts = int(table["transactions"].sum()) if "transactions" in table else 0
    at = f" at threshold {threshold:.2f}" if threshold is not None else ""
    st.markdown(f"**{len(table)} segments** over **{alerts:,} alerts**{at}, ranked by observed fraud rate.")

    config = {
        "segment": st.column_config.NumberColumn("Segment", format="%d"),
        "transactions": st.column_config.NumberColumn("Alerts", format="localized"),
        "fraud_rate": st.column_config.NumberColumn("Observed fraud rate", format="percent"),
        "seconds_since_card_last_txn": st.column_config.TextColumn(display_label("seconds_since_card_last_txn")),
        "amount_vs_card_avg_ratio": st.column_config.TextColumn(display_label("amount_vs_card_avg_ratio")),
    }
    for col in table.columns:
        if col not in config:
            config[col] = st.column_config.NumberColumn(display_label(col), format=CLUSTER_FORMATS.get(col, "%.2f"))
    st.dataframe(table, hide_index=True, width="stretch", column_config=config)
    note = ("Segment averages over interpretable features only; segments are not fraud verdicts. "
            "Time is a position in the relative TransactionDT cycle. A first-seen card has no prior "
            "transaction and no prior average, so its time gap is shown as text and its "
            "amount-vs-typical ratio is not meaningful.")
    if dropped:
        note += f" {len(dropped)} masked column(s) were removed before display."
    st.caption(note)


def _register_sklearn_pickle_compat() -> None:
    """Let bundles pickled with scikit-learn 1.5/1.6 load on newer releases.

    The bundle's logistic-regression ColumnTransformer references
    `_RemainderColsList`, a private list wrapper that later scikit-learn
    versions removed. Restoring it as a plain UserList is enough to
    unpickle; the app only uses the bundle's XGBoost model and feature list.
    """
    import sklearn.compose._column_transformer as column_transformer

    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(collections.UserList):
            pass

        column_transformer._RemainderColsList = _RemainderColsList


@st.cache_resource
def load_model_bundle():
    _register_sklearn_pickle_compat()
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_test_features(feature_cols: tuple) -> pd.DataFrame:
    """Test-window rows and model columns only, streamed to keep memory low.

    Reading the whole 590k x 461 feature table (one row group) and then
    splitting it needs several copies at peak and is OOM-killed under a 4 GB
    container limit. Instead: take the exact test IDs from the split keys
    (same chronological_split as training), stream only the model columns in
    batches, and keep the test rows. Categorical columns are pinned to the
    categories a full read produces, so the model's category codes match.
    """
    keys = pd.read_parquet(FEATURES_PATH, columns=["TransactionID", "TransactionDT", "transaction_day", "isFraud"])
    _, test_keys = chronological_split(keys)
    test_ids = set(test_keys["TransactionID"])
    del keys, test_keys

    columns = ["TransactionID", *[c for c in feature_cols if c != "TransactionID"]]
    parts = []
    for batch in pq.ParquetFile(FEATURES_PATH).iter_batches(batch_size=50_000, columns=columns):
        frame = batch.to_pandas()
        parts.append(frame[frame["TransactionID"].isin(test_ids)])
    test = pd.concat(parts, ignore_index=True)

    for col in test.columns:
        if isinstance(test[col].dtype, pd.CategoricalDtype) or test[col].dtype == object:
            full = pd.read_parquet(FEATURES_PATH, columns=[col])[col]
            if isinstance(full.dtype, pd.CategoricalDtype):
                test[col] = pd.Categorical(test[col], categories=full.cat.categories, ordered=full.cat.ordered)
    return test.set_index("TransactionID", drop=False)


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# ------------------------------------------------------------- analytics --
def cost_at_threshold(scored: pd.DataFrame, threshold: float) -> dict:
    alerted = scored["fraud_probability"] >= threshold
    fraud = scored["isFraud"] == 1
    tp, fp = int((alerted & fraud).sum()), int((alerted & ~fraud).sum())
    fn, tn = int((~alerted & fraud).sum()), int((~alerted & ~fraud).sum())
    alerts = tp + fp
    precision = tp / alerts if alerts else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "alerts": alerts, "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "true_negatives": tn, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "estimated_cost": fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST,
    }


def classification_table(m: dict) -> pd.DataFrame:
    """Fraudulent vs. legitimate outcome counts at a threshold."""
    return pd.DataFrame({
        "Actual class": ["Fraudulent", "Legitimate"],
        "Flagged for review": [m["true_positives"], m["false_positives"]],
        "Not flagged": [m["false_negatives"], m["true_negatives"]],
    })


def tier_table(scored: pd.DataFrame) -> pd.DataFrame:
    table = (
        scored.groupby("risk_tier")
        .agg(transactions=("TransactionID", "size"), frauds=("isFraud", "sum"))
        .reindex(TIERS).fillna(0)
    )
    table["share"] = table["transactions"] / table["transactions"].sum()
    table["fraud_rate"] = table["frauds"] / table["transactions"].where(table["transactions"] > 0)
    return table.reset_index().rename(columns={"risk_tier": "tier"})


# ------------------------------------------------------------------- pages --
def page_overview(scored: pd.DataFrame, summary: dict):
    st.title("Fraud overview")
    st.caption(
        "Chronological test window: the most recent 20% of transactions by TransactionDT "
        "offset, never seen in training."
    )

    flagged = int((scored["risk_tier"] != "Low").sum())
    fraud_value = scored.loc[scored["isFraud"] == 1, "TransactionAmt"].sum()
    cols = st.columns(6)
    cols[0].metric("Transactions", f"{len(scored):,}")
    cols[1].metric("Flagged, medium and above", f"{flagged:,}")
    cols[2].metric("Fraud rate", f"{scored['isFraud'].mean():.2%}")
    cols[3].metric("Fraud value", f"${fraud_value:,.0f}")
    cols[4].metric("XGBoost PR-AUC", f"{summary['xgboost']['pr_auc']:.3f}")
    cols[5].metric("XGBoost ROC-AUC", f"{summary['xgboost']['roc_auc']:.3f}")

    op = summary.get("cost_optimal_threshold")
    if op is not None:
        m = cost_at_threshold(scored, float(op))
        st.markdown(md_safe(
            f"At the deployed threshold **{op:.2f}**: **{m['alerts']:,}** transactions flagged for review "
            f"({m['true_positives']:,} fraudulent, {m['false_positives']:,} legitimate) and "
            f"{m['false_negatives']:,} fraudulent transactions not flagged: precision {m['precision']:.1%}, "
            f"recall {m['recall']:.1%}, scenario cost ${m['estimated_cost']:,.0f}."
        ))

    # Risk ladder: the alert queue's shape in one bar.
    tiers = tier_table(scored)
    ladder = go.Figure()
    for _, row in tiers.iterrows():
        ladder.add_trace(go.Bar(
            x=[row["transactions"]], y=[""], orientation="h", name=row["tier"],
            marker=dict(color=RISK_COLORS[row["tier"]], line=dict(color="#FFFFFF", width=1)),
            hovertemplate=(f"{row['tier']}: {row['transactions']:,.0f} transactions "
                           f"({row['share']:.1%}), observed fraud rate {row['fraud_rate']:.1%}<extra></extra>"),
        ))
    styled(ladder, "Risk ladder: test-window transactions by tier", height=150, barmode="stack",
           xaxis_title=None, yaxis=dict(showticklabels=False), margin=dict(l=12, r=12, t=44, b=8))
    left, right = st.columns([3, 2])
    left.plotly_chart(ladder, width="stretch")
    right.dataframe(
        tiers, hide_index=True, width="stretch",
        column_config={
            "tier": "Tier",
            "transactions": st.column_config.NumberColumn("Transactions", format="%d"),
            "frauds": st.column_config.NumberColumn("Frauds", format="%d"),
            "share": st.column_config.NumberColumn("Share", format="percent"),
            "fraud_rate": st.column_config.NumberColumn("Observed fraud rate", format="percent"),
        },
    )

    c1, c2 = st.columns(2)
    by_product = (scored.groupby("ProductCD")["isFraud"].agg(["mean", "size"])
                  .sort_values("mean").reset_index())
    fig = go.Figure(go.Bar(
        x=by_product["mean"], y=by_product["ProductCD"], orientation="h", marker_color=DATA,
        text=[f"{v:.1%}" for v in by_product["mean"]], textposition="outside",
        customdata=by_product["size"], hovertemplate="%{y}: %{x:.2%} of %{customdata:,} transactions<extra></extra>",
    ))
    c1.plotly_chart(styled(fig, "Fraud rate by product code", xaxis_tickformat=".0%"), width="stretch")

    trend = scored.groupby("transaction_day")["isFraud"].agg(["sum", "size"]).reset_index()
    fig = go.Figure(go.Scatter(
        x=trend["transaction_day"], y=trend["sum"], mode="lines", line=dict(color=DATA, width=2),
        hovertemplate="relative day %{x}: %{y} frauds<extra></extra>",
    ))
    c2.plotly_chart(styled(fig, "Frauds per relative day (TransactionDT offset)",
                           xaxis_title="relative day", yaxis_title="frauds"), width="stretch")

    st.subheader("Global risk drivers")
    importance = load_output_csv("feature_importance.csv").head(15).copy()
    importance["label"] = importance["feature"].map(display_label)
    importance["family"] = importance["feature"].map(
        lambda f: "masked" if is_masked_column(f) else "documented / engineered")
    ranked = importance.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=ranked["mean_abs_shap"], y=ranked["label"], orientation="h",
        marker_color=[RISK_COLORS["Low"] if fam == "masked" else DATA for fam in ranked["family"]],
        hovertemplate="%{y}: mean |SHAP| %{x:.3f}<extra></extra>",
    ))
    st.plotly_chart(
        styled(fig, "Top 15 features by mean |SHAP| (grey = masked column, family label only)",
               height=460, xaxis_title="mean |SHAP|"),
        width="stretch",
    )
    if SHAP_SUMMARY_PATH.exists():
        with st.expander("Static SHAP summary plot from notebook 03"):
            st.image(str(SHAP_SUMMARY_PATH), width="stretch")
            st.caption("This plot is an image from notebook 03 and shows raw column names; masked "
                       "names (C*, D*, M*, V*, id_*) are identifiers only, with no meaning implied. "
                       "Use the chart above for family labels.")

    st.subheader("Model benchmarks")
    st.markdown("**Model comparison**")
    comparison, note = load_optional_csv("model_comparison.csv")
    if comparison is None:
        st.info(note)
    else:
        render_model_comparison(comparison)

    st.markdown("**Tuning and operating benchmark**")
    benchmark, note = load_optional_csv("tuning_benchmark.csv")
    if benchmark is None:
        st.info(note)
    else:
        render_tuning_benchmark(benchmark)


def page_alert_queue(scored: pd.DataFrame):
    st.title("Alert queue")
    st.caption("Highest risk first. Filter to build a review list.")

    products = sorted(scored["ProductCD"].dropna().unique().tolist())
    with st.sidebar:
        st.subheader("Queue filters")
        tiers = st.multiselect("Risk tier", TIERS, default=TIERS[:3])
        chosen_products = st.multiselect("Product code", products, default=products)
        min_score, max_score = st.slider("Risk score", 0.0, 1.0, (0.3, 1.0), 0.01)
        amt_lo, amt_hi = float(scored["TransactionAmt"].min()), float(scored["TransactionAmt"].max())
        amt_range = st.slider("Amount (\\$)", amt_lo, amt_hi, (amt_lo, amt_hi))

    queue = scored[
        scored["risk_tier"].isin(tiers)
        & scored["ProductCD"].isin(chosen_products)
        & scored["fraud_probability"].between(min_score, max_score)
        & scored["TransactionAmt"].between(*amt_range)
    ].sort_values("fraud_probability", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("In queue", f"{len(queue):,}")
    c2.metric(f"Days of review at {DAILY_REVIEW_CAPACITY:,}/day",
              f"{len(queue) / max(scored['transaction_day'].nunique(), 1) / DAILY_REVIEW_CAPACITY:.2f}")
    c3.metric("Median risk score", f"{queue['fraud_probability'].median():.3f}" if len(queue) else "–")

    columns = ["TransactionID", "fraud_probability", "risk_tier", "TransactionAmt", "ProductCD",
               "card4", "card6", "P_emaildomain", "DeviceType", "top_reason_1"]
    st.dataframe(
        queue[columns], hide_index=True, width="stretch", height=560,
        column_config={
            "TransactionID": st.column_config.NumberColumn("Transaction ID", format="%d"),
            "fraud_probability": st.column_config.NumberColumn("Risk score", format="%.3f"),
            "risk_tier": "Tier",
            "TransactionAmt": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "ProductCD": "Product",
            "card4": "Card network",
            "card6": "Card type",
            "P_emaildomain": "Purchaser email domain",
            "DeviceType": "Device",
            "top_reason_1": st.column_config.TextColumn("Top reason", width="large"),
        },
    )

    clusters, note = load_optional_csv("alert_clusters.csv")
    with st.expander("Alert segments (K-means on interpretable features)"):
        if clusters is None:
            st.info(note)
        else:
            benchmark, _ = load_optional_csv("tuning_benchmark.csv")
            threshold = None
            if benchmark is not None and {"selected", "cost_optimal_threshold"} <= set(benchmark.columns):
                chosen = benchmark[benchmark["selected"].astype(str).str.lower().isin(["true", "1"])]
                if len(chosen):
                    threshold = float(chosen["cost_optimal_threshold"].iloc[0])
            render_alert_segments(clusters, threshold)


def page_investigation(scored: pd.DataFrame, bundle: dict):
    st.title("Transaction investigation")

    top_alerts = scored.nlargest(500, "fraud_probability")
    # 4 decimals: many top alerts score above 0.999, so 3 decimals would show them all as 1.000.
    top_scores = dict(zip(top_alerts["TransactionID"], top_alerts["fraud_probability"], strict=True))
    pick_col, lookup_col = st.columns([3, 2])
    picked = pick_col.selectbox(
        "Top 500 alerts by risk score",
        options=list(top_scores),
        format_func=lambda t: f"{t}  ·  score {top_scores[t]:.4f}",
    )
    typed = lookup_col.text_input("Or look up any transaction ID in the test window", placeholder="e.g. 3577539")

    txn_id = picked
    if typed.strip():
        if not typed.strip().isdigit():
            st.error("Transaction IDs are whole numbers. Showing the selected alert instead.")
        elif int(typed) not in set(scored["TransactionID"]):
            st.error(f"Transaction {typed.strip()} is not in the test window. Showing the selected alert instead.")
        else:
            txn_id = int(typed)

    row = scored.loc[scored["TransactionID"] == txn_id].iloc[0]
    threshold = float(bundle.get("cost_optimal_threshold", 0.5))
    flagged = row["fraud_probability"] >= threshold
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk score", f"{row['fraud_probability']:.4f}")  # same precision as the picker
    c2.metric("Tier", row["risk_tier"])
    c3.metric(f"Decision at {threshold:.2f}", "Flag for review" if flagged else "No alert")
    c4.metric("Amount", f"${row['TransactionAmt']:,.2f}")
    c5.metric("Relative day", f"{int(row['transaction_day'])}")
    st.caption(md_safe(
        f"Operating threshold {threshold:.2f} is cost-optimal for the scenario "
        f"(missed fraud ${FALSE_NEGATIVE_COST:,}, false-positive review ${FALSE_POSITIVE_COST:,}). "
        "\"Flag for review\" means elevated risk warranting review, not a determination of fraud; "
        "\"No alert\" means the transaction is treated as legitimate unless other evidence arises."
    ))

    features_ready = FEATURES_PATH.is_file()
    test_features = load_test_features(tuple(bundle["feature_cols"])) if features_ready else None
    raw_row = None
    if test_features is not None and txn_id in test_features.index:
        raw_row = test_features.loc[txn_id]
    first_seen = raw_row is not None and card_is_first_seen(raw_row)

    details, why = st.columns([2, 3])
    with details:
        st.subheader("Transaction details")
        st.dataframe(
            pd.DataFrame({
                "Field": ["Transaction ID", "Product code", "Card network", "Card type",
                          "Purchaser email domain", "Device type"],
                "Value": [str(row["TransactionID"]), row["ProductCD"], row["card4"], row["card6"],
                          row["P_emaildomain"], row["DeviceType"]],
            }).fillna("missing").astype(str),
            hide_index=True, width="stretch",
        )
        if raw_row is not None:
            st.subheader("Card history (card1 key)")
            ratio = raw_row.get("amount_vs_card_avg_ratio")
            st.dataframe(
                pd.DataFrame({
                    "Field": ["Prior transactions", "Time since last transaction", "Average past amount",
                              "Largest past amount", "Amount vs. typical"],
                    "Value": [
                        f"{int(raw_row['card_prior_txn_count']):,}",
                        humanize_seconds(raw_row["seconds_since_card_last_txn"]),
                        "none (first seen)" if first_seen else f"${raw_row['card_avg_amount_prior']:,.2f}",
                        "none (first seen)" if first_seen else f"${raw_row['card_max_amount_prior']:,.2f}",
                        f"n/a ({FIRST_SEEN_RATIO_NOTE})" if first_seen or pd.isna(ratio) else f"{ratio:,.2f}x",
                    ],
                }),
                hide_index=True, width="stretch",
            )
            if first_seen:
                st.caption("First transaction seen for this card key. With no prior average, the stored "
                           "amount-vs-typical ratio equals the raw amount and is inflated, so it is not shown.")
            else:
                st.caption("card1 is a shared card-attribute code used as an account key, so the history "
                           "can include other cards with the same code.")
        st.caption(f"Demo only: ground truth, not known at review time. isFraud = {int(row['isFraud'])}")

    with why:
        st.subheader("Why flagged" if flagged else "Main risk drivers")
        if not features_ready:
            st.info("Live SHAP needs data/processed/transactions_features.parquet. Build it with "
                    "`python src/data_processing.py` and `python src/features.py`.")
            return
        if raw_row is None:
            st.info("This transaction is not in the test-window feature table.")
            return

        model, feature_cols = bundle["xgb_model"], bundle["feature_cols"]
        x_row = test_features.loc[[txn_id], feature_cols]
        values = get_explainer(model)(x_row).values[0]
        feature_row = x_row.iloc[0]

        contrib = (pd.DataFrame({"feature": feature_cols, "shap_value": values})
                   .assign(abs_shap=lambda d: d["shap_value"].abs())
                   .nlargest(10, "abs_shap"))
        contrib["label"] = contrib["feature"].map(display_label)
        reasons = [investigation_reason(f, v, feature_row[f], first_seen)
                   for f, v in zip(contrib["feature"], contrib["shap_value"], strict=True)]

        ordered = contrib.sort_values("shap_value")
        fig = go.Figure(go.Bar(
            x=ordered["shap_value"], y=ordered["label"], orientation="h",
            marker_color=[INCREASES_RISK if v > 0 else DECREASES_RISK for v in ordered["shap_value"]],
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        ))
        st.plotly_chart(styled(fig, "Top 10 SHAP contributions (red raises risk, blue lowers it)",
                               height=380, xaxis_title="SHAP value"), width="stretch")

        items = "".join(
            f'<li class="{"up" if v > 0 else "down"}">{html_safe(r)}</li>'
            for r, v in zip(reasons[:5], contrib["shap_value"][:5], strict=True)
        )
        st.html(f'<ol class="reasons">{items}</ol>')
        st.caption("Risk score from outputs/scored_transactions.csv; contributions recomputed live "
                   "from app/model.pkl. Masked columns show a family label and never a value.")


def page_threshold_simulator(scored: pd.DataFrame, summary: dict):
    st.title("Threshold simulator")
    st.caption(md_safe(
        f"Scenario costs: missed fraud ${FALSE_NEGATIVE_COST:,}, false-positive review "
        f"${FALSE_POSITIVE_COST:,}. These are assumptions, not industry figures."
    ))

    sweep = load_output_csv("threshold_sweep.csv")
    cost_curve = load_output_csv("cost_sensitive_threshold.csv")
    best_t = summary.get("cost_optimal_threshold")
    threshold = st.slider("Review threshold", 0.0, 1.0, float(best_t if best_t is not None else 0.5), 0.01)

    m = cost_at_threshold(scored, threshold)
    n_days = max(int(scored["transaction_day"].nunique()), 1)
    alerts_per_day = m["alerts"] / n_days
    cols = st.columns(4)
    cols[0].metric("Alerts", f"{m['alerts']:,}")
    cols[1].metric("Frauds caught", f"{m['true_positives']:,}")
    cols[2].metric("Frauds missed", f"{m['false_negatives']:,}")
    cols[3].metric("Scenario cost", f"${m['estimated_cost']:,.0f}")
    cols = st.columns(4)
    cols[0].metric("Precision", f"{m['precision']:.1%}")
    cols[1].metric("Recall", f"{m['recall']:.1%}")
    cols[2].metric("F1", f"{m['f1']:.3f}")
    cols[3].metric(f"Alerts per day vs. {DAILY_REVIEW_CAPACITY:,} capacity",
                   f"{alerts_per_day:,.0f}  ({alerts_per_day / DAILY_REVIEW_CAPACITY:.1f}x)")

    table_col, note_col = st.columns([3, 2])
    table_col.dataframe(
        classification_table(m), hide_index=True, width="stretch",
        column_config={
            "Flagged for review": st.column_config.NumberColumn(format="localized"),
            "Not flagged": st.column_config.NumberColumn(format="localized"),
        },
    )
    note_col.caption(md_safe(
        f"Fraudulent vs. legitimate outcomes on the {len(scored):,}-transaction test window at threshold "
        f"{threshold:.2f}. Missed fraud costs ${FALSE_NEGATIVE_COST:,} each and a legitimate transaction "
        f"flagged for review costs ${FALSE_POSITIVE_COST:,} (scenario assumptions). Above "
        f"{DAILY_REVIEW_CAPACITY:,} alerts/day the team cannot review every alert; the capacity chart "
        "below shows what the top 500 per day catch."
    ))

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["precision"], name="Precision",
                             line=dict(color=DATA, width=2)))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["recall"], name="Recall",
                             line=dict(color=INK, width=2, dash="dot")))
    fig.add_vline(x=threshold, line_dash="dash", line_color=MUTED)
    c1.plotly_chart(styled(fig, "Precision and recall by threshold", xaxis_title="threshold",
                           yaxis_tickformat=".0%"), width="stretch")

    fig = go.Figure(go.Scatter(x=cost_curve["threshold"], y=cost_curve["total_cost"],
                               name="Scenario cost", line=dict(color=INK, width=2)))
    fig.add_vline(x=threshold, line_dash="dash", line_color=MUTED)
    if best_t is not None:
        fig.add_vline(x=best_t, line_color=DATA, annotation_text=f"cost-optimal {best_t:.2f}",
                      annotation_position="top right")
    c2.plotly_chart(styled(fig, "Scenario cost by threshold", xaxis_title="threshold",
                           yaxis_tickprefix="$", yaxis_tickformat=",.0f"), width="stretch")

    st.subheader(f"Fixed capacity: {DAILY_REVIEW_CAPACITY:,} alerts per day")
    daily = load_output_csv("daily_capacity_simulation.csv")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["day"], y=daily["fraud_in_day"], name="Frauds that day", marker_color=RULE))
    fig.add_trace(go.Bar(x=daily["day"], y=daily["fraud_captured"], name="Caught within capacity",
                         marker_color=DATA))
    st.plotly_chart(styled(fig, "Daily fraud capture under a fixed review capacity", barmode="overlay",
                           xaxis_title="relative day", yaxis_title="frauds"), width="stretch")
    st.caption(
        f"Reviewing the top {DAILY_REVIEW_CAPACITY:,} scores each day catches on average "
        f"{daily['recall'].mean():.1%} of that day's fraud (range {daily['recall'].min():.1%} to "
        f"{daily['recall'].max():.1%}), at {daily['precision'].mean():.1%} average precision."
    )


# --------------------------------------------------------------------- main --
def main():
    st.html(STYLE)
    st.sidebar.title("Risk operations")
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(
        "Data: IEEE-CIS Fraud Detection (Kaggle, Vesta Corporation). Real anonymized "
        "transactions; masked fields carry family labels only."
    )

    operational_context_banner()

    missing = missing_core_outputs()
    if missing:
        commands = sorted({CORE_OUTPUTS[name] for name in missing})
        st.error(
            "The console can't start because these outputs are missing: "
            + ", ".join(f"`outputs/{name}`" for name in missing)
            + ". From the repository root, run: " + " then ".join(f"`{c}`" for c in commands) + "."
        )
        st.stop()

    scored, summary = load_scored(), load_summary()

    if page == "Fraud overview":
        page_overview(scored, summary)
    elif page == "Alert queue":
        page_alert_queue(scored)
    elif page == "Transaction investigation":
        if not MODEL_PATH.is_file():
            st.error(f"`app/{MODEL_PATH.name}` is missing. Rebuild it with `python src/model.py`.")
            return
        try:
            bundle = load_model_bundle()
        except Exception as exc:  # incompatible pickle or corrupt file
            st.error(f"`app/{MODEL_PATH.name}` could not be loaded ({type(exc).__name__}: {exc}). "
                     "Rebuild it with `python src/model.py` in the current environment.")
            return
        page_investigation(scored, bundle)
    else:
        page_threshold_simulator(scored, summary)


if __name__ == "__main__":
    main()
