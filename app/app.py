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
import base64
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

import inference as inf  # noqa: E402
from model import (  # noqa: E402
    DAILY_REVIEW_CAPACITY,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    chronological_split,
    risk_tier,
)
from scoring import FEATURE_LABELS, is_masked_column, label_for_feature, reason_text  # noqa: E402

OUTPUTS_DIR = ROOT / "outputs"
SHAP_SUMMARY_PATH = OUTPUTS_DIR / "shap_summary.png"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
FEATURES_PATH = ROOT / "data" / "processed" / "transactions_features.parquet"
FORM_MODEL_PATH = inf.ARTIFACTS_DIR / "interpretable_model.pkl"
MAX_BATCH_ROWS = 100_000      # protects memory on small hosts (e.g. Streamlit Community Cloud)
BATCH_REASON_ROWS = 300       # SHAP reason codes for the highest-scored fraudulent rows

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

LANDING = "ValidaX Landing"
PAGES = [LANDING, "Classify a transaction", "Batch classification", "Fraud overview", "Alert queue",
         "Transaction investigation", "Threshold simulator"]
PAGE_SLUGS = {LANDING: "home", "Classify a transaction": "classify", "Batch classification": "batch",
              "Fraud overview": "overview", "Alert queue": "queue",
              "Transaction investigation": "investigate", "Threshold simulator": "threshold"}
TIERS = ["Critical", "High", "Medium", "Low"]

st.set_page_config(page_title="ValidaX | Autonomous Transaction Validation", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------- design --
# Dark design system (also set in .streamlit/config.toml so native widgets,
# menus and data tables render dark): canvas, card, border, text, muted, accent.
CANVAS = "#0E1117"
CARD = "#161B22"
RULE = "#30363D"
INK = "#F0F6FC"
BODY = "#C9D1D9"
MUTED = "#8B949E"
ACCENT = "#238636"
POSITIVE = "#3FB950"
DATA = "#58A6FF"
PANEL = CARD
RISK_COLORS = {"Critical": "#F85149", "High": "#DB6D28", "Medium": "#D29922", "Low": "#6E7681"}
INCREASES_RISK, DECREASES_RISK = RISK_COLORS["Critical"], DATA
FONT_STACK = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

_LOGO_MARKUP = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none">'
    f'<path d="M12 2.5 4 5.6v6.1c0 4.9 3.4 8.9 8 9.8 4.6-.9 8-4.9 8-9.8V5.6l-8-3.1Z" '
    f'fill="{ACCENT}" stroke="{POSITIVE}" stroke-width="1.2"/>'
    f'<path d="m8.4 12.2 2.5 2.5 4.9-5" stroke="{INK}" stroke-width="2" stroke-linecap="round" '
    f'stroke-linejoin="round"/></svg>'
)
# st.html sanitises inline <svg>, so the mark is embedded as an image.
LOGO_SVG = ('<img alt="ValidaX" width="30" height="30" src="data:image/svg+xml;base64,'
            + base64.b64encode(_LOGO_MARKUP.encode()).decode() + '">')

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
.stApp {{ font-family: {FONT_STACK}; background: {CANVAS}; color: {INK}; font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top: 4.4rem; max-width: 1440px; }}
h1 {{ font-size: 1.6rem !important; font-weight: 600 !important; letter-spacing: -0.015em; color: {INK} !important; }}
h2 {{ font-size: 1.15rem !important; font-weight: 600 !important; color: {INK} !important; }}
h3 {{ font-size: 1rem !important; font-weight: 600 !important; color: {INK} !important; }}
.stApp p, .stApp li, .stApp label {{ color: {BODY}; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: {MUTED} !important; }}
[data-testid="stWidgetLabel"] p {{ color: {INK} !important; font-weight: 500; font-size: 0.86rem; }}

/* Sidebar: crisp text on the canvas, nav as a list with a green accent on the active page. */
[data-testid="stSidebar"] {{ background: {CANVAS}; border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{ color: {INK}; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: {MUTED} !important; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    width: 100%; margin: 0; padding: 0.5rem 0.7rem; border-left: 3px solid transparent;
    border-radius: 0 6px 6px 0; cursor: pointer;
}}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ color: {MUTED}; font-size: 0.9rem; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {CARD}; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover p {{ color: {INK}; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: {CARD}; border-left-color: {ACCENT};
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: {INK}; font-weight: 600; }}
/* Hide only the drawn circle; the real (visually hidden) radio input stays for keyboard/screen readers. */
[data-testid="stSidebar"] [role="radiogroup"] label > div > div:first-child:not([data-testid]) {{ display: none; }}
.vx-nav-title {{ color: {MUTED}; font-size: 0.72rem; letter-spacing: 0.06em; margin: 0.2rem 0 0.4rem 0.2rem; }}

/* Cards: metrics, tables, charts, expanders, bordered containers, forms. */
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {RULE}; border-radius: 8px; padding: 0.7rem 0.9rem;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: 0.8rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 600; color: {INK}; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {{ white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; line-height: 1.25; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius: 8px; }}
[data-testid="stPlotlyChart"] {{
    background: {CARD}; border: 1px solid {RULE}; border-radius: 8px; overflow: hidden;
}}
[data-testid="stExpander"] details {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 8px; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{ border-radius: 8px; }}
[data-testid="stForm"] {{ border: none; padding: 0; }}
[data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border-color: {RULE}; }}
hr {{ border-color: {RULE} !important; }}

/* Brand bar and context banner. */
.vx-brand {{
    display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
    background: {CARD}; border: 1px solid {RULE}; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.7rem;
}}
.vx-brand-left {{ display: flex; align-items: center; gap: 0.7rem; }}
.vx-brand-name {{ color: {INK}; font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; }}
.vx-brand-name span {{ color: {MUTED}; font-weight: 400; }}
.vx-brand-tag {{ color: {MUTED}; font-size: 0.85rem; }}
.ctx {{
    width: 100%; box-sizing: border-box; background: {CARD}; border: 1px solid {RULE};
    border-left: 3px solid {ACCENT}; border-radius: 8px; padding: 0.65rem 0.95rem; margin-bottom: 1.1rem;
    color: {INK}; font-size: 0.86rem; line-height: 1.55; white-space: normal; overflow-wrap: anywhere;
}}
.ctx strong {{ font-weight: 600; color: {INK}; }}
.ctx .muted {{ color: {MUTED}; }}

/* Form section headers. */
.vx-section {{
    color: {INK}; font-weight: 600; font-size: 0.95rem; padding-bottom: 0.45rem; margin-bottom: 0.2rem;
    border-bottom: 1px solid {RULE};
}}

/* Landing page. */
.vx-hero {{ padding: 1.6rem 0 1.2rem 0; max-width: 860px; }}
.vx-hero h1 {{ font-size: 2.3rem !important; line-height: 1.15; margin: 0 0 0.7rem 0; }}
.vx-hero p {{ color: {BODY}; font-size: 1.02rem; line-height: 1.6; margin: 0; max-width: 72ch; }}
.vx-tiles {{
    display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.75rem; margin: 0.4rem 0 1.4rem 0;
}}
.vx-tile {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 8px; padding: 0.85rem 1rem; }}
.vx-tile .v {{ color: {INK}; font-size: 1.55rem; font-weight: 600; }}
.vx-tile .l {{ color: {MUTED}; font-size: 0.8rem; margin-top: 0.15rem; }}
@media (max-width: 1100px) {{ .vx-tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.vx-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 0.75rem; margin: 0.4rem 0 1rem 0;
}}
.vx-step {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 8px; padding: 0.9rem 1rem; }}
.vx-step .n {{ color: {POSITIVE}; font-size: 0.78rem; font-weight: 600; }}
.vx-step .t {{ color: {INK}; font-weight: 600; margin: 0.15rem 0 0.35rem 0; }}
.vx-step .d {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.5; }}

.reasons {{ margin: 0.2rem 0 0 0; padding-left: 1.1rem; font-size: 0.9rem; line-height: 1.65; }}
.reasons .up {{ color: {INCREASES_RISK}; }}
.reasons .down {{ color: {DECREASES_RISK}; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
"""

PLOT_LAYOUT = dict(
    font=dict(family=FONT_STACK, size=12, color=BODY),
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    margin=dict(l=14, r=14, t=46, b=14),
    title=dict(x=0.01, xanchor="left", font=dict(size=14, color=INK)),
    xaxis=dict(gridcolor="#21262D", linecolor=RULE, zeroline=False, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor="#21262D", linecolor=RULE, zeroline=False, tickfont=dict(color=MUTED)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="",
                font=dict(color=BODY)),
    hoverlabel=dict(font_family=FONT_STACK, bgcolor=CANVAS, bordercolor=RULE, font_color=INK),
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


def brand_bar() -> None:
    st.html(
        f'<div class="vx-brand"><div class="vx-brand-left">{LOGO_SVG}'
        '<div class="vx-brand-name">ValidaX <span>| Autonomous Transaction Validation</span></div></div>'
        '<div class="vx-brand-tag">Precision Risk Scoring. Zero Black-Box Decisions.</div></div>'
    )


def operational_context_banner() -> None:
    st.html(
        '<div class="ctx"><strong>FinTech / CSE303 (F20) · Transaction risk prioritization '
        "console.</strong> Scores rank transactions for investigator review: a high score "
        "indicates <strong>elevated risk warranting review, not a determination of fraud</strong>. "
        f"<span class=\"muted\">Scenario assumptions:</span> missed fraud &#36;{FALSE_NEGATIVE_COST:,}, "
        f"false-positive review &#36;{FALSE_POSITIVE_COST:,}; review capacity "
        f"{DAILY_REVIEW_CAPACITY:,} alerts/day. Masked C/D/M/V/id_ columns appear with generic family "
        "labels only.</div>"
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
def get_explainer(_model, key: str):
    """TreeExplainer per model. `_model` is not hashed by Streamlit, so `key` keeps the cache per model."""
    return shap.TreeExplainer(_model)


@st.cache_resource
def load_form_model():
    return joblib.load(FORM_MODEL_PATH)


@st.cache_resource
def load_inference_assets():
    """(metadata, card history) for preparing new transactions."""
    return inf.load_metadata(), inf.load_card_history()


def inference_assets_missing() -> list:
    return [str(p.relative_to(ROOT)) for p in (inf.METADATA_PATH, inf.CARD_HISTORY_PATH) if not p.is_file()]


def get_main_bundle():
    """The deployed bundle, or None after showing a clear error."""
    if not MODEL_PATH.is_file():
        st.error(f"`app/{MODEL_PATH.name}` is missing. Rebuild it with `python src/model.py`.")
        return None
    try:
        return load_model_bundle()
    except Exception as exc:  # incompatible pickle or corrupt file
        st.error(f"`app/{MODEL_PATH.name}` could not be loaded ({type(exc).__name__}: {exc}). "
                 "Rebuild it with `python src/model.py` in the current environment.")
        return None


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
            marker=dict(color=RISK_COLORS[row["tier"]], line=dict(color=CANVAS, width=2)),
            hovertemplate=(f"{row['tier']}: {row['transactions']:,.0f} transactions "
                           f"({row['share']:.1%}), observed fraud rate {row['fraud_rate']:.1%}<extra></extra>"),
        ))
    styled(ladder, "Risk ladder: test-window transactions by tier", height=190, barmode="stack",
           xaxis_title=None, yaxis=dict(showticklabels=False), margin=dict(l=14, r=14, t=46, b=10),
           legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0, title_text="",
                       font=dict(color=BODY)))
    left, right = st.columns([1, 1], gap="medium")
    left.plotly_chart(ladder, width="stretch")
    right.dataframe(
        tiers, hide_index=True, width="stretch",
        column_config={
            "tier": "Tier",
            "transactions": st.column_config.NumberColumn("Transactions", format="%d"),
            "frauds": st.column_config.NumberColumn("Frauds", format="%d"),
            "share": st.column_config.NumberColumn("Share", format="percent"),
            "fraud_rate": st.column_config.NumberColumn("Fraud rate", format="percent",
                                                        help="Observed fraud rate in the tier"),
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
    fig.update_traces(cliponaxis=False)
    c1.plotly_chart(styled(fig, "Fraud rate by product code", xaxis_tickformat=".0%",
                           xaxis_range=[0, float(by_product["mean"].max()) * 1.18]), width="stretch")

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
        values = get_explainer(model, "main")(x_row).values[0]
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
    fig.add_trace(go.Bar(x=daily["day"], y=daily["fraud_in_day"], name="Frauds that day", marker_color="#484F58"))
    fig.add_trace(go.Bar(x=daily["day"], y=daily["fraud_captured"], name="Caught within capacity",
                         marker_color=DATA))
    st.plotly_chart(styled(fig, "Daily fraud capture under a fixed review capacity", barmode="overlay",
                           xaxis_title="relative day", yaxis_title="frauds"), width="stretch")
    st.caption(
        f"Reviewing the top {DAILY_REVIEW_CAPACITY:,} scores each day catches on average "
        f"{daily['recall'].mean():.1%} of that day's fraud (range {daily['recall'].min():.1%} to "
        f"{daily['recall'].max():.1%}), at {daily['precision'].mean():.1%} average precision."
    )


# --------------------------------------------------------- classification --
def verdict_banner(label: str, probability: float, threshold: float) -> None:
    fraud = label == inf.FRAUDULENT
    color = RISK_COLORS["Critical"] if fraud else POSITIVE
    meaning = ("flag for review: elevated risk, not a final determination of fraud" if fraud
               else "below the review threshold: no alert")
    st.html(
        f'<div style="background:{CARD};border:1px solid {RULE};border-left:4px solid {color};'
        f'border-radius:8px;padding:0.9rem 1.1rem;margin:0.4rem 0 0.9rem 0;">'
        f'<div style="font-size:0.8rem;color:{MUTED};">Prediction</div>'
        f'<div style="font-size:1.6rem;font-weight:600;color:{color};">{html_safe(label)}</div>'
        f'<div style="font-size:0.88rem;color:{INK};">Fraud probability {probability:.4f} vs. threshold '
        f'{threshold:.2f}, {html_safe(meaning)}.</div></div>'
    )


def explain_rows(model, key: str, X: pd.DataFrame, top_n: int):
    """Top-|SHAP| (feature, value) pairs per row, highest first."""
    values = get_explainer(model, key)(X).values
    order = (-abs(values)).argsort(axis=1)[:, :top_n]
    return [[(X.columns[j], float(row[j])) for j in idx] for row, idx in zip(values, order, strict=True)]


def page_classify_transaction(main_summary: dict):
    st.title("Classify a transaction")
    st.caption("Enter one e-commerce transaction to classify it as fraudulent or legitimate. Time is a "
               "position in the dataset's relative cycle (TransactionDT is an offset, not a date).")

    missing = inference_assets_missing()
    if not FORM_MODEL_PATH.is_file():
        missing.append(str(FORM_MODEL_PATH.relative_to(ROOT)))
    if missing:
        st.error("This page needs " + ", ".join(f"`{p}`" for p in missing) + ". Build them with "
                 "`python src/inference.py` and `python src/interpretable_model.py`.")
        return
    bundle = load_form_model()
    meta, _ = load_inference_assets()
    cats = meta["categories"]

    def pick(options, preferred):
        return options.index(preferred) if preferred in options else 0

    def section(title: str) -> None:
        st.html(f'<div class="vx-section">{html_safe(title)}</div>')

    defaults = meta.get("form_defaults", {})
    none = "(not provided)"
    with st.form("classify_transaction", border=False):
        top_left, top_right = st.columns(2, gap="medium")
        with top_left.container(border=True):
            section("Payment")
            c1, c2 = st.columns(2)
            amount = c1.number_input("Amount (\\$)", min_value=0.01, value=117.00, step=1.0, format="%.2f")
            product = c2.selectbox("Product code", cats["ProductCD"], index=pick(cats["ProductCD"], "W"))
            c1, c2 = st.columns(2)
            card4 = c1.selectbox("Card network", cats["card4"], index=pick(cats["card4"], "visa"))
            card6 = c2.selectbox("Card type", cats["card6"], index=pick(cats["card6"], "debit"))

        with top_right.container(border=True):
            section("Customer & Device")
            c1, c2 = st.columns(2)
            p_email = c1.selectbox("Purchaser email domain", [none, *cats["P_emaildomain"]],
                                   index=pick([none, *cats["P_emaildomain"]], "gmail.com"))
            r_email = c2.selectbox("Recipient email domain", [none, *cats["R_emaildomain"]])
            c1, c2, c3 = st.columns(3)
            device = c1.selectbox("Device type", ["(unknown)", *cats["DeviceType"]])
            hour = c2.number_input("Hour (relative, 0-23)", min_value=0, max_value=23, value=14)
            dow = c3.number_input("Day (relative, 0-6)", min_value=0, max_value=6, value=2)

        bottom_left, bottom_right = st.columns(2, gap="medium")
        with bottom_left.container(border=True):
            section("Card History")
            first_seen = st.checkbox("First transaction seen for this card (history below is ignored)")
            c1, c2 = st.columns(2)
            prior = c1.number_input("Earlier transactions", min_value=0, value=12, step=1)
            avg_prior = c2.number_input("Average past amount (\\$)", min_value=0.0, value=95.0, step=1.0)
            c1, c2 = st.columns(2)
            max_prior = c1.number_input("Largest past amount (\\$)", min_value=0.0, value=300.0, step=1.0)
            gap_hours = c2.number_input("Hours since last transaction", min_value=0.0, value=30.0, step=1.0)

        with bottom_right.container(border=True):
            section("Billing Address")
            c1, c2 = st.columns(2)
            addr1 = c1.number_input("addr1 (billing region code)", value=defaults.get("addr1"), step=1.0)
            addr2 = c2.number_input("addr2 (billing country code)", value=defaults.get("addr2"), step=1.0)
            c1, c2 = st.columns(2)
            dist1 = c1.number_input("dist1 (distance, optional)", value=None, step=1.0)
            dist2 = c2.number_input("dist2 (distance, optional)", value=None, step=1.0)
            st.caption("Address codes are prefilled with the most common values in the data. Clear them only "
                       "if the address is really unknown: a missing billing address is itself a strong fraud "
                       "signal (11.8% fraud vs. 2.5% when present). Distances are usually missing.")
        submitted = st.form_submit_button("Classify transaction", type="primary")

    if not submitted:
        st.info("Fill in the transaction and select **Classify transaction**.")
        return

    inputs = {
        "TransactionAmt": amount, "ProductCD": product, "card4": card4, "card6": card6,
        "P_emaildomain": None if p_email == none else p_email,
        "R_emaildomain": None if r_email == none else r_email,
        "DeviceType": None if device == "(unknown)" else device,
        "addr1": addr1, "addr2": addr2, "dist1": dist1, "dist2": dist2,
        "transaction_hour": hour, "transaction_dow": dow,
        "card_prior_txn_count": 0 if first_seen else prior,
        "card_avg_amount_prior": avg_prior, "card_max_amount_prior": max_prior,
        "seconds_since_card_last_txn": None if first_seen else gap_hours * 3600,
    }
    X = inf.prepare_single(inputs, meta, bundle["feature_cols"])
    probability = float(bundle["xgb_model"].predict_proba(X)[:, 1][0])
    threshold = float(bundle["threshold"])
    label = str(inf.classify([probability], threshold)[0])
    verdict_banner(label, probability, threshold)

    contributions = explain_rows(bundle["xgb_model"], "form", X, top_n=8)[0]
    row = X.iloc[0]
    is_first = bool(row["card_is_first_seen"] == 1)
    chart = pd.DataFrame({"label": [display_label(f) for f, _ in contributions],
                          "shap_value": [v for _, v in contributions]}).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=chart["shap_value"], y=chart["label"], orientation="h",
        marker_color=[INCREASES_RISK if v > 0 else DECREASES_RISK for v in chart["shap_value"]],
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    left, right = st.columns([3, 2])
    left.plotly_chart(styled(fig, "What drove this prediction (red raises risk, blue lowers it)",
                             height=360, xaxis_title="SHAP value"), width="stretch")
    with right:
        st.markdown("**Top reasons**")
        items = "".join(
            f'<li class="{"up" if v > 0 else "down"}">{html_safe(investigation_reason(f, v, row[f], is_first))}</li>'
            for f, v in contributions[:5]
        )
        st.html(f'<ol class="reasons">{items}</ol>')

    s = bundle["summary"]["test"]
    main_pr = main_summary["xgboost"]["pr_auc"]
    st.caption(md_safe(
        f"Form model: XGBoost on {bundle['summary']['features']} interpretable fields (no masked columns can be "
        f"typed in). On the 118,108-transaction test window it reaches PR-AUC {s['pr_auc']:.3f} / ROC-AUC "
        f"{s['roc_auc']:.3f}, and at threshold {threshold:.2f} catches {s['at_threshold']['recall']:.1%} of fraud "
        f"at {s['at_threshold']['precision']:.1%} precision. The full 451-feature model (PR-AUC {main_pr:.3f}) is "
        "more accurate; use Batch classification for rows in the IEEE-CIS format. Threshold chosen for the "
        f"${FALSE_NEGATIVE_COST:,} missed-fraud / ${FALSE_POSITIVE_COST:,} false-positive scenario."
    ))


def page_batch_classification(bundle: dict):
    st.title("Batch classification")
    threshold = float(bundle["cost_optimal_threshold"])
    st.caption(md_safe(
        "Upload transactions in the IEEE-CIS format (the columns of Kaggle's train/test_transaction.csv, "
        f"optionally with the identity file) to classify every row as fraudulent or legitimate with the main "
        f"451-feature model at threshold {threshold:.2f} (cost-optimal for ${FALSE_NEGATIVE_COST:,} missed fraud "
        f"/ ${FALSE_POSITIVE_COST:,} false positive). Files are processed in memory and never stored."
    ))
    missing = inference_assets_missing()
    if missing:
        st.error("This page needs " + ", ".join(f"`{p}`" for p in missing) + ". Build them with "
                 "`python src/inference.py`.")
        return
    meta, history = load_inference_assets()

    template = ",".join(["TransactionID", "TransactionDT", "TransactionAmt", "card1",
                         *[c for c in meta["raw_columns"] if c not in inf.REQUIRED_BATCH_COLUMNS]]) + "\n"
    c1, c2 = st.columns(2)
    tx_file = c1.file_uploader("Transactions CSV (required)", type="csv")
    id_file = c2.file_uploader("Identity CSV (optional, merged on TransactionID)", type="csv")
    st.download_button("Download a blank column template", template, "validax_template.csv", "text/csv")
    st.caption(f"Required columns: {', '.join(inf.REQUIRED_BATCH_COLUMNS)}. Any other IEEE-CIS column that is "
               f"missing is treated as missing data. Up to {MAX_BATCH_ROWS:,} rows per upload. If an `isFraud` "
               "column is included, the page also scores the predictions against it.")
    if tx_file is None:
        st.info("Upload a transactions CSV to classify it.")
        return

    try:
        raw = inf.read_batch_csv(tx_file, meta)
        if len(raw) > MAX_BATCH_ROWS:
            st.error(f"The file has {len(raw):,} rows; the limit is {MAX_BATCH_ROWS:,}. Split it and upload in parts.")
            return
        if id_file is not None:
            identity = inf.read_batch_csv(id_file, meta)
            if "TransactionID" not in identity or identity["TransactionID"].duplicated().any():
                st.error("The identity file needs a unique TransactionID column.")
                return
            overlap = [c for c in identity.columns if c in raw.columns and c != "TransactionID"]
            raw = raw.merge(identity.drop(columns=overlap), on="TransactionID", how="left")
        X = inf.prepare_batch(raw, meta, history, list(bundle["feature_cols"]))
    except ValueError as exc:
        st.error(f"Couldn't read the upload: {exc}")
        return
    except Exception as exc:  # malformed CSV, wrong encoding, ...
        st.error(f"Couldn't read the upload ({type(exc).__name__}). Check that it is a comma-separated "
                 f"IEEE-CIS file: {exc}")
        return

    model = bundle["xgb_model"]
    proba = model.predict_proba(X)[:, 1]
    labels = inf.classify(proba, threshold)
    raw_by_id = raw.set_index("TransactionID").loc[X.index]
    results = pd.DataFrame({
        "TransactionID": X.index.to_numpy(),
        "fraud_probability": proba,
        "prediction": labels,
        "risk_tier": [risk_tier(p) for p in proba],
        "TransactionAmt": raw_by_id["TransactionAmt"].to_numpy(),
        "ProductCD": raw_by_id["ProductCD"].to_numpy() if "ProductCD" in raw_by_id else None,
        "card4": raw_by_id["card4"].to_numpy() if "card4" in raw_by_id else None,
        "card6": raw_by_id["card6"].to_numpy() if "card6" in raw_by_id else None,
    })

    in_window = float((raw_by_id["TransactionDT"] <= meta["history_end_dt"]).mean())
    if in_window > 0:
        st.warning(f"{in_window:.0%} of rows fall inside the labeled IEEE-CIS time window (TransactionDT ≤ "
                   f"{meta['history_end_dt']:,}). Their card-history features include later transactions from "
                   "that window, so treat those predictions as indicative only. Rows after the window "
                   "(e.g. Kaggle's test files) are scored point-in-time correctly.")

    fraud_mask = results["prediction"] == inf.FRAUDULENT
    cols = st.columns(4)
    cols[0].metric("Rows classified", f"{len(results):,}")
    cols[1].metric("Fraudulent", f"{int(fraud_mask.sum()):,}  ({fraud_mask.mean():.1%})")
    cols[2].metric("Legitimate", f"{int((~fraud_mask).sum()):,}")
    cols[3].metric("Amount in fraudulent rows", f"${results.loc[fraud_mask, 'TransactionAmt'].sum():,.0f}")

    if "isFraud" in raw_by_id and raw_by_id["isFraud"].notna().all():
        truth = raw_by_id["isFraud"].astype(int).to_numpy()
        m = cost_at_threshold(pd.DataFrame({"fraud_probability": proba, "isFraud": truth}), threshold)
        st.markdown(md_safe(
            f"**Against the uploaded `isFraud` labels:** precision {m['precision']:.1%}, recall {m['recall']:.1%}, "
            f"F1 {m['f1']:.3f}, scenario cost ${m['estimated_cost']:,.0f}."
        ))
        st.dataframe(classification_table(m), hide_index=True, width="stretch")

    # SHAP reason codes for the highest-scored fraudulent rows.
    for i in (1, 2, 3):
        results[f"top_reason_{i}"] = ""
    flagged = results[fraud_mask].nlargest(BATCH_REASON_ROWS, "fraud_probability")
    if len(flagged):
        X_flagged = X.loc[flagged["TransactionID"].to_numpy()]
        for row_pos, contributions in zip(flagged.index, explain_rows(model, "main", X_flagged, top_n=3), strict=True):
            feature_row = X.loc[results.at[row_pos, "TransactionID"]]
            first = bool(feature_row["card_is_first_seen"] == 1)
            for i, (f, v) in enumerate(contributions, 1):
                results.at[row_pos, f"top_reason_{i}"] = investigation_reason(f, v, feature_row[f], first)

    shown = results.sort_values("fraud_probability", ascending=False)
    st.dataframe(
        shown, hide_index=True, width="stretch", height=480,
        column_config={
            "TransactionID": st.column_config.NumberColumn("Transaction ID", format="%d"),
            "fraud_probability": st.column_config.NumberColumn("Fraud probability", format="%.4f"),
            "prediction": "Prediction",
            "risk_tier": "Tier",
            "TransactionAmt": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "ProductCD": "Product", "card4": "Card network", "card6": "Card type",
            "top_reason_1": st.column_config.TextColumn("Top reason", width="large"),
            "top_reason_2": "Reason 2", "top_reason_3": "Reason 3",
        },
    )
    st.caption(f"Reason codes are computed for the {min(len(flagged), BATCH_REASON_ROWS):,} highest-scored "
               "fraudulent rows; masked columns show family labels only. \"Fraudulent\" means the row warrants "
               "investigator review; it is not a final determination of fraud.")
    st.download_button("Download results (CSV)", shown.to_csv(index=False), "validax_predictions.csv", "text/csv")


# ----------------------------------------------------------------- landing --
def go_to(page: str) -> None:
    """Button callback: switch the sidebar navigation before the next run."""
    st.session_state["nav"] = page


def page_landing(scored: pd.DataFrame, summary: dict) -> None:
    rows = int(summary.get("train_rows", 0)) + int(summary.get("test_rows", 0))
    threshold = float(summary.get("cost_optimal_threshold", 0.5))
    curve = load_output_csv("cost_sensitive_threshold.csv")
    at_threshold = curve.loc[(curve["threshold"] - threshold).abs().idxmin()]
    daily = load_output_csv("daily_capacity_simulation.csv")
    pr_auc = summary["xgboost"]["pr_auc"]
    baseline_pr = summary["logreg"]["pr_auc"]
    no_skill = float(scored["isFraud"].mean())
    comparison, _ = load_optional_csv("model_comparison.csv")
    ladder = (f"the strongest of {len(comparison)} models compared" if comparison is not None
              else "the strongest model compared")

    st.html(
        '<div class="vx-hero"><h1>Classify every e-commerce transaction as fraudulent or legitimate, '
        'and show why.</h1>'
        "<p>ValidaX scores each payment with a gradient-boosted model trained on "
        f"{rows:,} real IEEE-CIS transactions, flags the ones above a cost-optimal threshold for "
        "investigator review, and explains every decision with plain-language reason codes. Masked "
        "provider fields are only ever shown by family, never interpreted.</p></div>"
    )

    c1, c2, c3, _ = st.columns([1.1, 1, 1, 2.2])
    c1.button("Open investigation portal", type="primary", on_click=go_to, args=("Transaction investigation",),
              width="stretch")
    c2.button("Classify a transaction", on_click=go_to, args=("Classify a transaction",), width="stretch")
    c3.button("Upload a batch", on_click=go_to, args=("Batch classification",), width="stretch")

    tiles = [
        (f"{rows:,}", "labeled transactions (IEEE-CIS)"),
        (f"{threshold:.2f}", "cost-optimal decision threshold"),
        (f"&#36;{at_threshold['total_cost']:,.0f}", "scenario cost at that threshold"),
        (f"{pr_auc:.3f}", f"PR-AUC ({pr_auc / no_skill:.0f}x the {no_skill:.3f} no-skill baseline)"),
        (f"{daily['recall'].mean():.1%}", f"fraud caught at {DAILY_REVIEW_CAPACITY:,} reviews/day"),
    ]
    st.html('<div class="vx-tiles">' + "".join(
        f'<div class="vx-tile"><div class="v">{v}</div><div class="l">{html_safe(label)}</div></div>'
        for v, label in tiles) + "</div>")

    st.subheader("How ValidaX decides")
    steps = [
        ("Data", f"{rows:,} labeled transactions joined with identity signals, split by time: the model is "
                 f"tested only on the most recent {int(summary.get('test_rows', 0)):,}."),
        ("Features", "451 inputs: documented fields, point-in-time card velocity (earlier transactions only) and "
                     "masked C/D/M/V/id_ families used as opaque signals."),
        ("Model", f"XGBoost with class weighting: PR-AUC {pr_auc:.3f} against {baseline_pr:.3f} for the "
                  f"logistic baseline, {ladder}."),
        ("Decision", f"Fraudulent if the score is at least {threshold:.2f}: the threshold that minimises "
                     f"cost when a missed fraud costs &#36;{FALSE_NEGATIVE_COST:,} and a review "
                     f"&#36;{FALSE_POSITIVE_COST:,}, under a {DAILY_REVIEW_CAPACITY:,}/day review cap."),
        ("Explanation", "TreeSHAP reason codes for every alert. Documented fields get plain labels; masked "
                        "columns keep generic family labels with no values shown."),
        ("Delivery", "This Streamlit app, a read-only Docker image and CI. Single-transaction form, CSV batch "
                     "upload and an investigation console."),
    ]
    st.html('<div class="vx-grid">' + "".join(
        f'<div class="vx-step"><div class="n">{i:02d}</div><div class="t">{title}</div>'
        f'<div class="d">{text}</div></div>' for i, (title, text) in enumerate(steps, 1)) + "</div>")
    st.caption("Every Fraudulent label means elevated risk warranting human review, not an automated "
               "determination of fraud. Costs and capacity are scenario assumptions.")


# --------------------------------------------------------------------- main --
def main():
    st.html(STYLE)
    if "nav" not in st.session_state:  # deep links: ?page=investigate
        slug = st.query_params.get("page")
        st.session_state["nav"] = next((p for p, s in PAGE_SLUGS.items() if s == slug), LANDING)
    st.sidebar.html(f'<div class="vx-brand-left" style="margin:0.2rem 0 0.9rem 0.1rem;">{LOGO_SVG}'
                    '<div class="vx-brand-name">ValidaX</div></div>'
                    '<div class="vx-nav-title">NAVIGATION</div>')
    page = st.sidebar.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
    st.query_params["page"] = PAGE_SLUGS[page]
    st.sidebar.divider()
    st.sidebar.caption(
        "Data: IEEE-CIS Fraud Detection (Kaggle, Vesta Corporation). Real anonymized "
        "transactions; masked fields carry family labels only."
    )

    brand_bar()
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

    if page == LANDING:
        page_landing(scored, summary)
    elif page == "Classify a transaction":
        page_classify_transaction(summary)
    elif page == "Batch classification":
        bundle = get_main_bundle()
        if bundle is not None:
            page_batch_classification(bundle)
    elif page == "Fraud overview":
        page_overview(scored, summary)
    elif page == "Alert queue":
        page_alert_queue(scored)
    elif page == "Transaction investigation":
        bundle = get_main_bundle()
        if bundle is not None:
            page_investigation(scored, bundle)
    else:
        page_threshold_simulator(scored, summary)


if __name__ == "__main__":
    main()
