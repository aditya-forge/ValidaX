"""ValidaX: payment fraud risk console (Streamlit).

Pages: Overview, Classify, Alerts, Investigate and Simulator. Reads the
artifacts produced by src/model.py and src/scoring.py. Run from the repo root
with `streamlit run app/app.py`.

Data note: IEEE-CIS Fraud Detection is real, anonymized e-commerce
transaction data (not synthetic). TransactionDT is a relative time offset
from an undisclosed reference point, not a calendar timestamp. Many columns
(C1-C14, D1-D15, M1-M9, V1-V339, id_*) are masked by the data provider
(Vesta); this console never claims to know their meaning. They are shown with
generic family labels only, and their raw values are never displayed.

Operating context: scores prioritise transactions for review. A score
indicates elevated risk warranting review, not a determination of fraud. Cost and capacity figures are scenario
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

TIERS = ["Critical", "High", "Medium", "Low"]

st.set_page_config(page_title="ValidaX | Payment Fraud Intelligence",
                   page_icon=str(Path(__file__).resolve().parent / "assets" / "validax-icon.svg"), layout="wide")

# Design system
# Flat, rectangular UI in two themes: one canvas, one surface, 1px rules, 4px corners,
# no shadows or gradients. Streamlit sets `color-scheme` on the app container for the
# active theme, so every custom colour is a CSS light-dark() pair and follows a theme
# switch without a rerun. .streamlit/config.toml holds the same values for native
# widgets and tables ([theme.light] and [theme.dark]).
PALETTE = {  # token: (light, dark)
    "canvas": ("#F8FAFC", "#050608"),       # near-black canvas from the reference
    "surface": ("#FFFFFF", "#0D1016"),
    "rule": ("#E2E8F0", "#242833"),
    "ink": ("#0F172A", "#E8EDF5"),
    "body": ("#334155", "#B5C0D0"),
    "muted": ("#64748B", "#8D9AAF"),
    "accent": ("#1F6FEB", "#4C8DF6"),       # brand blue for text, indicators, focus
    "accent_hover": ("#1A5FCC", "#6BA1F8"),
    "button": ("#1F6FEB", "#2F6FE0"),       # primary button fill: white text stays above 4.5:1
    "button_hover": ("#1A5FCC", "#3B7CF0"),
    "on_accent": ("#FFFFFF", "#FFFFFF"),
    "flag": ("#E11D48", "#F43F5E"),
    "clear": ("#15803D", "#22C55E"),
}


def ld(token: str) -> str:
    light, dark = PALETTE[token]
    return f"light-dark({light}, {dark})"


CANVAS, CARD, RULE = ld("canvas"), ld("surface"), ld("rule")
INK, BODY, MUTED = ld("ink"), ld("body"), ld("muted")
ACCENT, ACCENT_HOVER, ON_ACCENT = ld("accent"), ld("accent_hover"), ld("on_accent")
BUTTON, BUTTON_HOVER = ld("button"), ld("button_hover")
FLAG, CLEAR = ld("flag"), ld("clear")
RADIUS = "4px"

# Chart marks are drawn by Plotly, which cannot read CSS, so they use mid-tones that
# hold contrast on both canvases. Fonts, grids and hover labels come from Streamlit's
# Plotly theme, which tracks the active theme in the browser.
BRAND = "#2F81F7"  # logo and favicon shield; the same in both themes
TRACE = "#3B82F6"
NEUTRAL = "#64748B"
FAINT = "rgba(100, 116, 139, 0.35)"
RISK_COLORS = {"Critical": "#F43F5E", "High": "#F97316", "Medium": "#EAB308", "Low": "#64748B"}
INCREASES_RISK, DECREASES_RISK = RISK_COLORS["Critical"], TRACE
FONT_STACK = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO_STACK = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Consolas, monospace"


# Logo: solid blue shield with a white geometric X (also app/assets/validax-icon.svg,
# the favicon). Fixed colours read on both canvases, so it is one image; st.html strips
# inline <svg>. The wordmark is text and follows the theme.
LOGO_MARK_PATH = Path(__file__).resolve().parent / "assets" / "validax-icon.svg"
LOGO_HTML = ('<span class="vx-logo" role="img" aria-label="ValidaX">'
             '<img class="vx-shield" alt="" src="data:image/svg+xml;base64,'
             + base64.b64encode(LOGO_MARK_PATH.read_bytes()).decode() + '">'
             '<span class="vx-word" aria-hidden="true">Valida<b>X</b></span></span>')

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp {{ font-family: {FONT_STACK}; background: {CANVAS}; color: {INK}; font-variant-numeric: tabular-nums; }}
/* Streamlit Community Cloud adds a fixed host bar above the app content. */
.block-container {{ padding-top: 5.5rem; max-width: 1440px; }}
h1 {{ font-size: 1.55rem !important; font-weight: 600 !important; letter-spacing: -0.01em; color: {INK} !important; }}
h2 {{ font-size: 1.12rem !important; font-weight: 600 !important; color: {INK} !important; }}
h3 {{ font-size: 1rem !important; font-weight: 600 !important; color: {INK} !important; }}
.stApp p, .stApp li, .stApp label {{ color: {BODY}; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: {MUTED} !important;
    opacity: 1 !important; }}
[data-testid="stWidgetLabel"] p {{ color: {INK} !important; font-weight: 500; font-size: 0.85rem; }}

/* Hide Streamlit's empty native bar; the branded row below is the app header. */
[data-testid="stHeader"] {{ display: none !important; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
/* Native navigation is replaced by the compact branded navigation row below. */
[data-testid="stTopNav"], [data-testid="stNavigation"] {{ display: none !important; }}
[data-testid="stTopNavLink"] {{ background: transparent !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent; }}
[data-testid="stTopNavLink"] p {{ color: {MUTED} !important; font-weight: 500; }}
[data-testid="stTopNavLink"]:hover p {{ color: {INK} !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ border-bottom-color: {ACCENT}; }}
[data-testid="stTopNavLink"][aria-current="page"] p {{ color: {INK} !important; }}
[data-testid="stBaseButton-header"] {{ color: {MUTED}; }}

/* Native buttons: rectangular; primary is the accent with contrasting text. */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{
    background: {BUTTON} !important; border: 1px solid {BUTTON} !important; color: {ON_ACCENT} !important; }}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p {{
    color: {ON_ACCENT} !important; font-weight: 600; }}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {{
    background: {BUTTON_HOVER} !important; border-color: {BUTTON_HOVER} !important; }}
[data-testid="stBaseButton-secondary"] {{ background: {CARD}; border: 1px solid {RULE}; }}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: {ACCENT}; color: {INK}; }}
.stApp a:focus-visible, .stApp button:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}

/* Rectangular surfaces: metrics, tables, charts, expanders, bordered containers. */
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {RULE}; border-radius: {RADIUS}; padding: 0.65rem 0.85rem;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: 0.78rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.35rem; font-weight: 600; color: {INK}; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {{ white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; line-height: 1.25; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius: {RADIUS}; }}
[data-testid="stPlotlyChart"] {{
    background: {CARD}; border: 1px solid {RULE}; border-radius: {RADIUS}; overflow: hidden;
}}
[data-testid="stExpander"] details {{ background: {CARD}; border: 1px solid {RULE}; border-radius: {RADIUS}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: {RADIUS} !important; }}
[data-testid="stForm"] {{ border: none; padding: 0; }}
[data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"],
.stTabs [data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border-color: {RULE}; }}
.stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid {RULE}; gap: 0; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 500; color: {MUTED}; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; }}
hr {{ border-color: {RULE} !important; }}

/* Brand row: logo, product line, status badge, theme switch. */
.st-key-vx_brand {{ border-bottom: 1px solid {RULE}; padding: 0.55rem 0; margin: 0 0 1.35rem 0;
    background: {CANVAS}; }}
.vx-mark {{ display: flex; align-items: center; gap: 0.75rem; }}
.vx-logo {{ display: inline-flex; align-items: center; gap: 0.5rem; }}
.vx-shield {{ display: block; width: 32px; height: 32px; flex: none; }}
.vx-word {{ color: {INK}; font-size: 1.2rem; font-weight: 600; letter-spacing: -0.02em; line-height: 1; }}
.vx-word b {{ color: {ACCENT}; font-weight: 600; }}
.vx-mark .desc {{ display: none; }}
.vx-header-nav {{ display: flex; align-items: center; justify-content: center; gap: clamp(0.7rem, 2vw, 2rem); }}
.vx-header-nav a {{ color: {MUTED}; text-decoration: none; font-size: 0.88rem; font-weight: 500;
    padding: 0.85rem 0 0.72rem; border-bottom: 2px solid transparent; white-space: nowrap; }}
.vx-header-nav a:hover, .vx-header-nav a[aria-current="page"] {{ color: {INK}; }}
.vx-header-nav a[aria-current="page"] {{ border-bottom-color: {ACCENT}; }}
.vx-header-actions {{ display: flex; align-items: center; justify-content: flex-end; gap: 0.75rem; }}
.vx-header-launch {{ display: inline-flex; align-items: center; justify-content: center; padding: 0.58rem 0.9rem;
    background: {BUTTON}; border: 1px solid {BUTTON}; color: {ON_ACCENT} !important; text-decoration: none;
    font-size: 0.84rem; font-weight: 600; white-space: nowrap; border-radius: {RADIUS}; }}
.vx-header-launch:hover {{ background: {BUTTON_HOVER}; border-color: {BUTTON_HOVER}; }}

/* Overview: title lockup, command callout, context box, telemetry strip. */
.vx-lockup h1 {{ font-size: 1.9rem !important; line-height: 1.2; margin: 0.2rem 0 0.5rem 0; }}
.vx-lockup p {{ color: {BODY}; font-size: 0.98rem; line-height: 1.6; margin: 0; max-width: 78ch; }}
.vx-box {{
    background: {CARD}; border: 1px solid {RULE}; border-radius: {RADIUS}; height: 100%;
    box-sizing: border-box;
}}
.vx-box .head {{
    color: {INK}; font-size: 0.82rem; font-weight: 600; padding: 0.55rem 0.9rem;
    border-bottom: 1px solid {RULE};
}}
.vx-box .body {{ padding: 0.7rem 0.9rem; color: {BODY}; font-size: 0.86rem; line-height: 1.55; }}
.vx-box pre {{ margin: 0; font-family: {MONO_STACK}; font-size: 0.8rem; line-height: 1.7; color: {BODY};
    white-space: pre-wrap; overflow-wrap: anywhere; background: transparent; }}
.vx-box pre .c {{ color: {MUTED}; }}
.vx-box pre .p {{ color: {ACCENT}; }}
.vx-kv {{ display: grid; grid-template-columns: max-content 1fr; column-gap: 1rem; row-gap: 0.3rem; margin-top: 0.7rem;
    padding-top: 0.65rem; border-top: 1px solid {RULE}; font-size: 0.82rem; }}
.vx-kv .k {{ color: {MUTED}; }}
.vx-kv .v {{ color: {INK}; font-family: {MONO_STACK}; font-size: 0.8rem; }}
.ctx {{
    background: {CARD}; border: 1px solid {RULE}; border-radius: {RADIUS}; padding: 0.6rem 0.9rem;
    margin: 0 0 1.1rem 0;
    color: {BODY}; font-size: 0.84rem; line-height: 1.55; white-space: normal; overflow-wrap: anywhere; }}
.ctx strong {{ color: {INK}; font-weight: 600; }}
.ctx .muted {{ color: {MUTED}; }}
.vx-strip {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1px; background: {RULE};
    border: 1px solid {RULE}; border-radius: {RADIUS}; overflow: hidden; margin: 1.1rem 0 1.6rem 0; }}
.vx-cell {{ background: {CARD}; padding: 0.75rem 0.95rem; }}
.vx-cell .k {{ color: {MUTED}; font-size: 0.76rem; }}
.vx-cell .v {{ color: {INK}; font-size: 1.35rem; font-weight: 600; margin-top: 0.15rem; }}
.vx-cell .s {{ color: {MUTED}; font-size: 0.74rem; margin-top: 0.1rem; }}
@media (max-width: 1100px) {{ .vx-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 850px) {{ .vx-header-nav {{ gap: 0.55rem; }} .vx-header-nav a {{ font-size: 0.78rem; }}
    .vx-header-launch {{ display: none; }} }}
@media (max-width: 650px) {{ .vx-header-nav {{ overflow-x: auto; justify-content: flex-start; }}
    .st-key-vx_brand [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 0.3rem; }} }}

/* Classification result: decision first. */
.vx-result {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: {RULE};
    border: 1px solid {RULE}; border-radius: {RADIUS}; overflow: hidden; margin: 1rem 0 0.5rem 0; }}
.vx-result.is-flag {{ border-left: 3px solid {FLAG}; }}
.vx-result.is-clear {{ border-left: 3px solid {CLEAR}; }}
.vx-result .v {{ font-size: 1.3rem; }}
.vx-result .v .flag {{ color: {FLAG}; }}
.vx-result .v .clear {{ color: {CLEAR}; }}
@media (max-width: 1100px) {{ .vx-result {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.vx-note {{ color: {MUTED}; font-size: 0.82rem; line-height: 1.5; }}

/* Rectangular actions: page links and anchor links styled as flat buttons. */
.vx-actions {{ display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 1rem 0 1.1rem 0; }}
.vx-btn {{ display: inline-block; padding: 0.5rem 1rem; border-radius: {RADIUS}; font-size: 0.88rem; font-weight: 500;
    text-decoration: none !important; border: 1px solid {RULE}; color: {INK} !important; background: {CARD}; }}
.vx-btn:hover {{ border-color: {ACCENT}; }}
.vx-btn.primary {{ background: {BUTTON}; border-color: {BUTTON}; color: {ON_ACCENT} !important; font-weight: 600; }}
.vx-btn.primary:hover {{ background: {BUTTON_HOVER}; border-color: {BUTTON_HOVER}; }}

/* Form section headers. */
.vx-section {{ color: {INK}; font-weight: 600; font-size: 0.92rem; padding-bottom: 0.45rem; margin-bottom: 0.2rem;
    border-bottom: 1px solid {RULE}; }}

.reasons {{ margin: 0.2rem 0 0 0; padding-left: 1.1rem; font-size: 0.9rem; line-height: 1.65; }}
.reasons .up {{ color: {FLAG}; }}
.reasons .down {{ color: {ACCENT}; }}
.vx-status-line {{ color: {MUTED}; font-size: 0.82rem; border-left: 2px solid {RULE}; padding: 0.1rem 0 0.1rem 0.6rem;
    margin: 0.2rem 0 0.6rem 0; }}
</style>
"""

# Transparent backgrounds let the surrounding card show through in either theme.
PLOT_LAYOUT = dict(
    font=dict(family=FONT_STACK, size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=14, r=14, t=46, b=14),
    title=dict(x=0.01, xanchor="left", font=dict(size=14)),
    xaxis=dict(zeroline=False),
    yaxis=dict(zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
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


# Theme switch. Streamlit keeps the chosen theme in localStorage under
# stActiveTheme-<path>-v2 ("Light" or "Dark") and applies it on load. The switch runs
# in a same-origin st.iframe: it writes that key for every page path and reloads the
# app. Its own colours are read from the app canvas, so it always matches the theme.
_THEME_SWITCH = """
<style>
:root { color-scheme: normal; }  /* match the host iframe so Chrome keeps the frame transparent */
html, body { margin: 0; background: transparent; }
body { display: flex; justify-content: flex-end; }
.seg { display: flex; border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; background: var(--surface); }
button { all: unset; box-sizing: border-box; width: 34px; height: 32px; display: grid; place-items: center;
         cursor: pointer; color: var(--muted); }
button + button { border-left: 1px solid var(--rule); }
button:hover { color: var(--ink); }
button[aria-pressed="true"] { color: var(--ink); background: var(--raise); cursor: default; }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 1.6; stroke-linecap: round; }
</style>
<div class="seg" role="group" aria-label="Theme">
  <button type="button" data-theme="Light" aria-label="Light theme" title="Light theme">
    <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2
      M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/></svg>
  </button><button type="button" data-theme="Dark" aria-label="Dark theme" title="Dark theme">
    <svg viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" stroke-linejoin="round"/></svg>
  </button>
</div>
<script>
const app = window.parent, COLORS = __COLORS__, PAGES = __PAGES__;
function active() {
  const root = app.document.querySelector(".stApp");
  const rgb = root ? app.getComputedStyle(root).backgroundColor.match(/\\d+/g) : null;
  if (!rgb) return "Dark";
  const [r, g, b] = rgb.map(Number);
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 > 0.5 ? "Light" : "Dark";
}
function paint() {
  const theme = active();
  for (const [k, v] of Object.entries(COLORS[theme])) document.documentElement.style.setProperty("--" + k, v);
  document.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.theme === theme));
}
function choose(theme) {
  if (theme === active()) return;
  const path = app.location.pathname;
  const base = path.replace(new RegExp("(" + PAGES.join("|") + ")/?$"), "");
  new Set([path, base, ...PAGES.map(p => base + p)]).forEach(
    p => app.localStorage.setItem("stActiveTheme-" + p + "-v2", JSON.stringify(theme)));
  app.location.reload();
}
document.querySelectorAll("button").forEach(b => b.addEventListener("click", () => choose(b.dataset.theme)));
paint();
setInterval(paint, 800);
</script>
"""


def theme_switch_html() -> str:
    raise_ = {"Light": "#F1F5F9", "Dark": "#191D26"}  # active button surface
    colors = {name: {"surface": PALETTE["surface"][i], "rule": PALETTE["rule"][i], "muted": PALETTE["muted"][i],
                     "ink": PALETTE["ink"][i], "accent": PALETTE["accent"][i], "raise": raise_[name]}
              for i, name in enumerate(("Light", "Dark"))}
    return (_THEME_SWITCH.replace("__COLORS__", json.dumps(colors))
            .replace("__PAGES__", json.dumps([path for _, _, path in NAV_PAGES])))


def brand_header() -> None:
    """Compact branded navigation row matching the reference header layout."""
    with st.container(key="vx_brand"):
        left, middle, right = st.columns([2.2, 5.2, 2.4], vertical_alignment="center")
        left.html(f'<div class="vx-mark">{LOGO_HTML}</div>')
        nav_html = '<nav class="vx-header-nav" aria-label="Main navigation">' + "".join(
            f'<a href="./{path}" data-page="{path}">{title}</a>' for title, _, path in NAV_PAGES
        ) + '</nav><script>\n'
        nav_html += """const p = window.parent.location.pathname.replace(/\\/$/, '').split('/').pop();
document.querySelectorAll('.vx-header-nav a').forEach(a => {
  if ((p || 'overview') === a.dataset.page) a.setAttribute('aria-current', 'page');
});
</script>"""
        middle.html(nav_html)
        with right:
            theme, launch = st.columns([1, 1.8], vertical_alignment="center")
            with theme:
                st.iframe(theme_switch_html(), height=34, alt="Theme switch: light or dark")
            launch.html('<a class="vx-header-launch" href="./classify" target="_self">Launch Workspace</a>')


def operational_context_banner() -> None:
    st.html(
        '<div class="ctx"><strong>Operating context:</strong> scores rank transactions for review. '
        "A flag means <strong>elevated risk warranting review, not a determination of fraud</strong>. "
        f'<span class="muted">Scenario:</span> missed fraud &#36;{FALSE_NEGATIVE_COST:,} | false positive '
        f"&#36;{FALSE_POSITIVE_COST:,} | capacity {DAILY_REVIEW_CAPACITY:,} alerts/day. Masked C/D/M/V/id_ "
        "columns: family labels only.</div>"
    )


# Section: Feature labelling
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


# Section: Card history
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


# Section: Data I/O
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
        "n/a" if pd.isna(m) else (f"{m:.{digits}f}" if pd.isna(s) else f"{m:.{digits}f} ± {s:.{digits}f}")
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
    note = ("Averages over interpretable features. Segments are not fraud verdicts. Time is a position "
            "in the relative TransactionDT cycle. First-seen cards have no prior transaction or average: "
            "time gap shown as text, amount ratio not meaningful.")
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


# Section: Analytics
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


# Section: Pages
def page_overview(scored: pd.DataFrame, summary: dict, embedded: bool = False):
    if embedded:
        st.subheader("Detection performance")
    else:
        st.title("Fraud overview")
    st.caption(
        "Test window: latest 20% of transactions by TransactionDT, held out from training. "
        "Deployed model."
    )

    flagged = int((scored["risk_tier"] != "Low").sum())
    fraud_value = scored.loc[scored["isFraud"] == 1, "TransactionAmt"].sum()
    cols = st.columns(6)
    cols[0].metric("Transactions", f"{len(scored):,}")
    cols[1].metric("Flagged, medium and above", f"{flagged:,}")
    cols[2].metric("Fraud rate", f"{scored['isFraud'].mean():.2%}")
    cols[3].metric("Fraud value", f"${fraud_value:,.0f}")
    cols[4].metric("Deployed PR-AUC", f"{summary['xgboost']['pr_auc']:.4f}")
    cols[5].metric("Deployed ROC-AUC", f"{summary['xgboost']['roc_auc']:.4f}")

    op = summary.get("cost_optimal_threshold")
    if op is not None:
        m = cost_at_threshold(scored, float(op))
        st.markdown(md_safe(
            f"Operating Threshold: **{op:.2f}** | flagged **{m['alerts']:,}** ({m['true_positives']:,} "
            f"fraudulent, {m['false_positives']:,} legitimate) | missed {m['false_negatives']:,} | precision "
            f"{m['precision']:.1%} | recall {m['recall']:.1%} | scenario cost ${m['estimated_cost']:,.0f}"
        ))

    # Risk ladder: the alert queue's shape in one bar.
    tiers = tier_table(scored)
    ladder = go.Figure()
    for _, row in tiers.iterrows():
        ladder.add_trace(go.Bar(
            x=[row["transactions"]], y=[""], orientation="h", name=row["tier"],
            marker=dict(color=RISK_COLORS[row["tier"]], line=dict(width=0)),
            hovertemplate=(f"{row['tier']}: {row['transactions']:,.0f} transactions "
                           f"({row['share']:.1%}), observed fraud rate {row['fraud_rate']:.1%}<extra></extra>"),
        ))
    styled(ladder, "Risk ladder: test-window transactions by tier", height=190, barmode="stack",
           xaxis_title=None, yaxis=dict(showticklabels=False), margin=dict(l=14, r=14, t=46, b=10),
           legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0, title_text=""))
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
        x=by_product["mean"], y=by_product["ProductCD"], orientation="h", marker_color=TRACE,
        text=[f"{v:.1%}" for v in by_product["mean"]], textposition="outside",
        customdata=by_product["size"], hovertemplate="%{y}: %{x:.2%} of %{customdata:,} transactions<extra></extra>",
    ))
    fig.update_traces(cliponaxis=False)
    c1.plotly_chart(styled(fig, "Fraud rate by product code", xaxis_tickformat=".0%",
                           xaxis_range=[0, float(by_product["mean"].max()) * 1.18]), width="stretch")

    trend = scored.groupby("transaction_day")["isFraud"].agg(["sum", "size"]).reset_index()
    fig = go.Figure(go.Scatter(
        x=trend["transaction_day"], y=trend["sum"], mode="lines", line=dict(color=TRACE, width=2),
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
        marker_color=[RISK_COLORS["Low"] if fam == "masked" else TRACE for fam in ranked["family"]],
        hovertemplate="%{y}: mean |SHAP| %{x:.3f}<extra></extra>",
    ))
    st.plotly_chart(
        styled(fig, "Top 15 features by mean |SHAP| (grey = masked column, family label only)",
               height=460, xaxis_title="mean |SHAP|"),
        width="stretch",
    )
    if SHAP_SUMMARY_PATH.exists():
        with st.expander("SHAP summary plot (notebook 03)"):
            st.image(str(SHAP_SUMMARY_PATH), width="stretch")
            st.caption("Static image with raw column names. Masked names (C*, D*, M*, V*, id_*) are "
                       "identifiers only. The chart above uses family labels.")

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
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.2, 1.3, 1, 1.3], gap="medium")
        tiers = f1.multiselect("Risk tier", TIERS, default=TIERS[:3])
        chosen_products = f2.multiselect("Product code", products, default=products)
        min_score, max_score = f3.slider("Risk score", 0.0, 1.0, (0.3, 1.0), 0.01)
        amt_lo, amt_hi = float(scored["TransactionAmt"].min()), float(scored["TransactionAmt"].max())
        amt_range = f4.slider("Amount (\\$)", amt_lo, amt_hi, (amt_lo, amt_hi))

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
    c3.metric("Median risk score", f"{queue['fraud_probability'].median():.4f}" if len(queue) else "n/a")

    columns = ["TransactionID", "fraud_probability", "risk_tier", "TransactionAmt", "ProductCD",
               "card4", "card6", "P_emaildomain", "DeviceType", "top_reason_1"]
    st.dataframe(
        queue[columns], hide_index=True, width="stretch", height=560,
        column_config={
            "TransactionID": st.column_config.NumberColumn("Transaction ID", format="%d"),
            "fraud_probability": st.column_config.NumberColumn("Risk score", format="%.4f"),
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


def stored_reasons(row: pd.Series, status: str) -> None:
    """Reason codes saved by the scoring run, for hosts without the feature table."""
    st.html(f'<div class="vx-status-line">{html_safe(status)}</div>')
    reasons = [r for r in (row.get(f"top_reason_{i}") for i in (1, 2, 3)) if isinstance(r, str) and r]
    if not reasons:
        st.html('<div class="vx-status-line">No stored reason codes for this transaction.</div>')
        return
    items = "".join(f'<li class="{"down" if "decreased risk" in r else "up"}">{html_safe(r)}</li>' for r in reasons)
    st.html(f'<ol class="reasons">{items}</ol>')
    st.caption("Masked columns: family label only, no values.")


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
    c3.metric(f"Verdict at {threshold:.2f}", verdict(row["fraud_probability"], threshold))
    c4.metric("Amount", f"${row['TransactionAmt']:,.2f}")
    c5.metric("Relative day", f"{int(row['transaction_day'])}")
    st.caption(md_safe(
        f"Cost-Optimal Threshold: {threshold:.2f} (missed fraud ${FALSE_NEGATIVE_COST:,} | false positive "
        f"${FALSE_POSITIVE_COST:,}). Flagged: elevated risk warranting review, not a determination of fraud. "
        "Legitimate: below threshold, no review queued."
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
        st.caption(f"Ground truth (not available at review time): isFraud = {int(row['isFraud'])}")

    with why:
        st.subheader("Why flagged" if flagged else "Main risk drivers")
        if raw_row is None:
            stored_reasons(row, "Live SHAP: feature table not loaded on this host. Showing reason codes "
                                "stored by the scoring run." if not features_ready else
                                "Transaction not in the test-window feature table. Showing stored reason codes.")
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
        st.caption("Score: outputs/scored_transactions.csv. Contributions: live TreeSHAP on app/model.pkl. "
                   "Masked columns: family label only, no values.")


def page_threshold_simulator(scored: pd.DataFrame, summary: dict):
    st.title("Threshold simulator")
    st.caption(md_safe(
        f"Scenario costs: missed fraud ${FALSE_NEGATIVE_COST:,} | false positive ${FALSE_POSITIVE_COST:,}. "
        "Assumed values, not industry benchmarks."
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
        f"Outcomes on {len(scored):,} test-window transactions at threshold {threshold:.2f}. Each missed "
        f"fraud costs ${FALSE_NEGATIVE_COST:,}; each legitimate transaction flagged costs "
        f"${FALSE_POSITIVE_COST:,}. Above {DAILY_REVIEW_CAPACITY:,} alerts/day not every alert gets reviewed: "
        f"see the capacity chart below."
    ))

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["precision"], name="Precision",
                             line=dict(color=TRACE, width=2)))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["recall"], name="Recall",
                             line=dict(color=NEUTRAL, width=2, dash="dot")))
    fig.add_vline(x=threshold, line_dash="dash", line_color=NEUTRAL)
    c1.plotly_chart(styled(fig, "Precision and recall by threshold", xaxis_title="threshold",
                           yaxis_tickformat=".0%"), width="stretch")

    fig = go.Figure(go.Scatter(x=cost_curve["threshold"], y=cost_curve["total_cost"],
                               name="Scenario cost", line=dict(color=NEUTRAL, width=2)))
    fig.add_vline(x=threshold, line_dash="dash", line_color=NEUTRAL)
    if best_t is not None:
        fig.add_vline(x=best_t, line_color=TRACE, annotation_text=f"cost-optimal {best_t:.2f}",
                      annotation_position="top right")
    c2.plotly_chart(styled(fig, "Scenario cost by threshold", xaxis_title="threshold",
                           yaxis_tickprefix="$", yaxis_tickformat=",.0f"), width="stretch")

    st.subheader(f"Fixed capacity: {DAILY_REVIEW_CAPACITY:,} alerts per day")
    daily = load_output_csv("daily_capacity_simulation.csv")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["day"], y=daily["fraud_in_day"], name="Frauds that day", marker_color=FAINT))
    fig.add_trace(go.Bar(x=daily["day"], y=daily["fraud_captured"], name="Caught within capacity",
                         marker_color=TRACE))
    st.plotly_chart(styled(fig, "Daily fraud capture under a fixed review capacity", barmode="overlay",
                           xaxis_title="relative day", yaxis_title="frauds"), width="stretch")
    st.caption(
        f"Reviewing the top {DAILY_REVIEW_CAPACITY:,} scores each day catches on average "
        f"{daily['recall'].mean():.1%} of that day's fraud (range {daily['recall'].min():.1%} to "
        f"{daily['recall'].max():.1%}), at {daily['precision'].mean():.1%} average precision."
    )


# Section: Classification
def explain_rows(model, key: str, X: pd.DataFrame, top_n: int):
    """Top-|SHAP| (feature, value) pairs per row, highest first."""
    values = get_explainer(model, key)(X).values
    order = (-abs(values)).argsort(axis=1)[:, :top_n]
    return [[(X.columns[j], float(row[j])) for j in idx] for row, idx in zip(values, order, strict=True)]


# Form state. Every field has a session-state key so presets can fill the form.
NOT_PROVIDED, UNKNOWN_DEVICE = "(not provided)", "(unknown)"
FORM_DEFAULTS = {
    "cls_amount": 117.0, "cls_product": "W", "cls_card4": "visa", "cls_card6": "debit",
    "cls_p_email": "gmail.com", "cls_r_email": NOT_PROVIDED, "cls_device": UNKNOWN_DEVICE,
    "cls_hour": 14, "cls_dow": 2,
    "cls_first_seen": False, "cls_prior": 12, "cls_avg": 95.0, "cls_max": 300.0, "cls_gap": 30.0,
    "cls_addr1": 299.0, "cls_addr2": 87.0, "cls_dist1": None, "cls_dist2": None,
}
# Illustrative inputs (not real customers). Each is applied on top of FORM_DEFAULTS.
FORM_PRESETS = {
    "Typical repeat card": {},
    "First-seen card": {"cls_first_seen": True},
    "High-risk spike": {
        "cls_amount": 450.0, "cls_product": "H", "cls_card6": "credit", "cls_p_email": "outlook.com",
        "cls_r_email": "outlook.com", "cls_device": "mobile", "cls_hour": 3,
        "cls_prior": 2, "cls_avg": 40.0, "cls_max": 60.0, "cls_gap": 0.07,
    },
}


def init_form_state(meta: dict) -> None:
    defaults = {**FORM_DEFAULTS, **{f"cls_{k}": v for k, v in meta.get("form_defaults", {}).items()}}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset(name: str, meta: dict) -> None:
    """Button callback: reset the form, apply the preset, classify on the next run."""
    defaults = {**FORM_DEFAULTS, **{f"cls_{k}": v for k, v in meta.get("form_defaults", {}).items()}}
    st.session_state.update({**defaults, **FORM_PRESETS[name]})
    st.session_state["cls_run"] = True
    st.session_state["cls_preset"] = name


def verdict(probability: float, threshold: float) -> str:
    return "Flagged" if probability >= threshold else "Legitimate"


def result_block(probability: float, threshold: float, form_threshold: float, preset) -> None:
    """Verdict at the operating threshold, 4-decimal score and scenario first; links to drivers and tools."""
    decision = verdict(probability, threshold)
    tone = "flag" if decision == "Flagged" else "clear"
    source = f"Preset: {preset}" if preset else "Manual input"
    cells = [
        ("Verdict", f'<span class="{tone}">{decision}</span>',
         "route to review" if tone == "flag" else f"below {threshold:.2f}"),
        ("Risk score", f"{probability:.4f}", "form model probability"),
        ("Operating threshold", f"{threshold:.2f}", f"cost-optimal | form model optimum {form_threshold:.2f}"),
        ("Input", html_safe(source), "edit any field and reclassify"),
    ]
    st.html(
        f'<div class="vx-result is-{tone}">' + "".join(
            f'<div class="vx-cell"><div class="k">{k}</div><div class="v">{v}</div>'
            f'<div class="s">{html_safe(s)}</div></div>'
            for k, v, s in cells) + "</div>"
        f'<div class="vx-note">Scenario: missed fraud &#36;{FALSE_NEGATIVE_COST:,} | false positive '
        f"&#36;{FALSE_POSITIVE_COST:,} | capacity {DAILY_REVIEW_CAPACITY:,} alerts/day. "
        "Flagged: elevated risk warranting review, not a determination of fraud.</div>"
        '<div class="vx-actions">'
        '<a class="vx-btn" href="#risk-drivers" target="_self">Inspect drivers ↓</a>'
        '<a class="vx-btn" href="./simulator" target="_self">Test thresholds →</a>'
        '<a class="vx-btn" href="./investigate" target="_self">Investigate scored transactions →</a></div>'
    )


def page_classify_transaction(main_summary: dict, embedded: bool = False):
    if not embedded:
        st.title("Classify a transaction")

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
    init_form_state(meta)

    preset_cols = st.columns([1, 1, 1, 3])
    for col, name in zip(preset_cols[:3], FORM_PRESETS, strict=False):
        col.button(name, on_click=apply_preset, args=(name, meta), width="stretch", key=f"preset_{name}")
    preset_cols[3].caption("Presets load illustrative inputs and classify on click. Hour and day are "
                           "positions in the relative TransactionDT cycle, not clock time.")

    def section(title: str) -> None:
        st.html(f'<div class="vx-section">{html_safe(title)}</div>')

    with st.form("classify_transaction", border=False):
        top_left, top_right = st.columns(2, gap="medium")
        with top_left.container(border=True):
            section("Payment")
            c1, c2 = st.columns(2)
            c1.number_input("Amount (\\$)", min_value=0.01, step=1.0, format="%.2f", key="cls_amount")
            c2.selectbox("Product code", cats["ProductCD"], key="cls_product")
            c1, c2 = st.columns(2)
            c1.selectbox("Card network", cats["card4"], key="cls_card4")
            c2.selectbox("Card type", cats["card6"], key="cls_card6")

        with top_right.container(border=True):
            section("Customer & Device")
            c1, c2 = st.columns(2)
            c1.selectbox("Purchaser email domain", [NOT_PROVIDED, *cats["P_emaildomain"]], key="cls_p_email")
            c2.selectbox("Recipient email domain", [NOT_PROVIDED, *cats["R_emaildomain"]], key="cls_r_email")
            c1, c2, c3 = st.columns(3)
            c1.selectbox("Device type", [UNKNOWN_DEVICE, *cats["DeviceType"]], key="cls_device")
            c2.number_input("Hour (relative, 0-23)", min_value=0, max_value=23, key="cls_hour")
            c3.number_input("Day (relative, 0-6)", min_value=0, max_value=6, key="cls_dow")

        bottom_left, bottom_right = st.columns(2, gap="medium")
        with bottom_left.container(border=True):
            section("Card History")
            st.checkbox("First transaction seen for this card (history below is ignored)", key="cls_first_seen")
            c1, c2 = st.columns(2)
            c1.number_input("Earlier transactions", min_value=0, step=1, key="cls_prior")
            c2.number_input("Average past amount (\\$)", min_value=0.0, step=1.0, key="cls_avg")
            c1, c2 = st.columns(2)
            c1.number_input("Largest past amount (\\$)", min_value=0.0, step=1.0, key="cls_max")
            c2.number_input("Hours since last transaction", min_value=0.0, step=0.5, format="%.2f", key="cls_gap")

        with bottom_right.container(border=True):
            section("Billing Address")
            c1, c2 = st.columns(2)
            c1.number_input("addr1 (billing region code)", step=1.0, key="cls_addr1")
            c2.number_input("addr2 (billing country code)", step=1.0, key="cls_addr2")
            c1, c2 = st.columns(2)
            c1.number_input("dist1 (distance, optional)", step=1.0, key="cls_dist1")
            c2.number_input("dist2 (distance, optional)", step=1.0, key="cls_dist2")
            st.caption("Address codes default to the most common values in the data. Clear them only if the "
                       "address is unknown: a missing billing address is itself a strong fraud signal "
                       "(11.8% fraud vs. 2.5% when present). Distances are usually missing.")
        submitted = st.form_submit_button("Classify transaction", type="primary")

    run_from_preset = st.session_state.pop("cls_run", False)
    if submitted:
        st.session_state["cls_preset"] = None
    if not (submitted or run_from_preset):
        st.info("Pick a preset or fill in the transaction, then select **Classify transaction**.")
        return

    state = st.session_state
    first_seen = bool(state["cls_first_seen"])
    inputs = {
        "TransactionAmt": state["cls_amount"], "ProductCD": state["cls_product"],
        "card4": state["cls_card4"], "card6": state["cls_card6"],
        "P_emaildomain": None if state["cls_p_email"] == NOT_PROVIDED else state["cls_p_email"],
        "R_emaildomain": None if state["cls_r_email"] == NOT_PROVIDED else state["cls_r_email"],
        "DeviceType": None if state["cls_device"] == UNKNOWN_DEVICE else state["cls_device"],
        "addr1": state["cls_addr1"], "addr2": state["cls_addr2"],
        "dist1": state["cls_dist1"], "dist2": state["cls_dist2"],
        "transaction_hour": state["cls_hour"], "transaction_dow": state["cls_dow"],
        "card_prior_txn_count": 0 if first_seen else state["cls_prior"],
        "card_avg_amount_prior": state["cls_avg"], "card_max_amount_prior": state["cls_max"],
        "seconds_since_card_last_txn": None if first_seen else state["cls_gap"] * 3600,
    }
    X = inf.prepare_single(inputs, meta, bundle["feature_cols"])
    probability = float(bundle["xgb_model"].predict_proba(X)[:, 1][0])
    threshold = float(bundle["threshold"])  # the form model's own cost-optimal cutoff
    operating = float(main_summary.get("cost_optimal_threshold", 0.5))
    result_block(probability, operating, threshold, st.session_state.get("cls_preset"))

    st.subheader("Risk drivers")
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
    left, right = st.columns([3, 2], gap="medium")
    left.plotly_chart(styled(fig, "SHAP contributions (red raises risk, blue lowers it)",
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
        f"Form model: XGBoost, {bundle['summary']['features']} interpretable fields (masked columns are not "
        f"form inputs). Test window: PR-AUC {s['pr_auc']:.3f} | ROC-AUC {s['roc_auc']:.3f} | at its own "
        f"{threshold:.2f} cutoff, recall {s['at_threshold']['recall']:.1%} and precision "
        f"{s['at_threshold']['precision']:.1%}. Deployed 451-feature model: PR-AUC {main_pr:.3f}. "
        "Use Batch upload for IEEE-CIS rows."
    ))


def page_batch_classification(bundle: dict, embedded: bool = False):
    if not embedded:
        st.title("Batch classification")
    threshold = float(bundle["cost_optimal_threshold"])
    st.caption(md_safe(
        "Upload IEEE-CIS rows (columns of Kaggle's train/test_transaction.csv, identity file optional). "
        f"Model: deployed 451-feature XGBoost. Cost-Optimal Threshold: {threshold:.2f} (missed fraud "
        f"${FALSE_NEGATIVE_COST:,} | false positive ${FALSE_POSITIVE_COST:,}). Files stay in memory and are "
        "not stored."
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


# Section: Overview
def telemetry_strip(summary: dict) -> None:
    """Five system counters in one ruled strip, all read from outputs/."""
    threshold = float(summary.get("cost_optimal_threshold", 0.5))
    curve = load_output_csv("cost_sensitive_threshold.csv")
    cost = float(curve.loc[(curve["threshold"] - threshold).abs().idxmin(), "total_cost"])

    pr_value, pr_sub = f"{summary['xgboost']['pr_auc']:.4f}", "deployed model, test window"
    benchmark, _ = load_optional_csv("tuning_benchmark.csv")
    if benchmark is not None and {"selected", "test_pr_auc_mean"} <= set(benchmark.columns):
        chosen = benchmark[benchmark["selected"].astype(str).str.lower().isin(["true", "1"])]
        if len(chosen):
            row = chosen.iloc[0]
            pr_value = f"{row['test_pr_auc_mean']:.4f}"
            std = row.get("test_pr_auc_std")
            pr_sub = f"± {std:.4f} · benchmark mean, 2 seeds" if pd.notna(std) else "benchmark mean"

    clusters, _ = load_optional_csv("alert_clusters.csv")
    if clusters is None:
        k_value, k_sub = "n/a", "run notebook 05 to create"
    else:
        k_value = f"k={len(clusters)}"
        k_sub = (f"{int(clusters['transactions'].sum()):,} alerts clustered" if "transactions" in clusters
                 else "K-means segments")

    cells = [
        ("PR-AUC", pr_value, pr_sub),
        ("Cost-optimal cutoff", f"{threshold:.2f}", "minimises scenario cost"),
        ("Scenario cost", f"&#36;{cost:,.0f}", f"at cutoff {threshold:.2f}, test window"),
        ("Review capacity", f"{DAILY_REVIEW_CAPACITY:,}/day", "investigator alerts per day"),
        ("Alert segments", k_value, k_sub),
    ]
    st.html('<div class="vx-strip">' + "".join(
        f'<div class="vx-cell"><div class="k">{html_safe(k)}</div><div class="v">{v}</div>'
        f'<div class="s">{html_safe(sub)}</div></div>' for k, v, sub in cells) + "</div>")


def page_home(scored: pd.DataFrame, summary: dict) -> None:
    rows = int(summary.get("train_rows", 0)) + int(summary.get("test_rows", 0))
    threshold = float(summary.get("cost_optimal_threshold", 0.5))
    st.html(
        '<div class="vx-lockup"><h1>Transaction fraud scoring with reason codes</h1>'
        f"<p>XGBoost classifier trained on {rows:,} labeled IEEE-CIS e-commerce transactions. "
        f"Cost-Optimal Threshold: {threshold:.2f}. Every alert carries TreeSHAP reason codes.</p></div>"
        '<div class="vx-actions"><a class="vx-btn primary" href="./classify" target="_self">Launch Workspace →</a>'
        '<a class="vx-btn" href="#model-benchmarks" target="_self">Model Benchmarks</a></div>'
    )

    train_rows, test_rows = int(summary.get("train_rows", 0)), int(summary.get("test_rows", 0))
    params = [
        ("Model", f"XGBoost · {summary.get('n_features', 451)} features"),
        ("Split", f"{train_rows:,} train / {test_rows:,} test, chronological"),
        ("Cutoff", f"{threshold:.2f}, cost-optimal"),
        ("Costs", f"FN &#36;{FALSE_NEGATIVE_COST:,} · FP &#36;{FALSE_POSITIVE_COST:,} (scenario)"),
        ("Capacity", f"{DAILY_REVIEW_CAPACITY:,} alerts/day"),
        ("Masking", "C/D/M/V/id_ by family only"),
    ]
    command = (
        '<div class="vx-box"><div class="head">Run locally</div><div class="body"><pre>'
        '<span class="c"># install, test, serve</span>\n'
        '<span class="p">&#36;</span> pip install -r requirements.txt\n'
        '<span class="p">&#36;</span> make test\n'
        '<span class="p">&#36;</span> streamlit run app/app.py\n'
        '<span class="c"># container: 4 GB, 4 CPUs, read-only root filesystem</span>\n'
        '<span class="p">&#36;</span> make docker-build &amp;&amp; make docker-run</pre>'
        '<div class="vx-kv">' + "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in params)
        + "</div></div></div>"
    )
    context = (
        '<div class="vx-box"><div class="head">Operating context</div><div class="body">'
        "Scores rank transactions for review. Flagged: <strong>elevated risk warranting review, not a "
        "determination of fraud</strong>.<br><br>"
        f"Scenario: missed fraud &#36;{FALSE_NEGATIVE_COST:,} | false positive &#36;{FALSE_POSITIVE_COST:,} | "
        f"capacity {DAILY_REVIEW_CAPACITY:,} alerts/day.<br><br>"
        "Masked C1-C14, D1-D15, M1-M9, V1-V339 and id_* columns: generic family labels only, no values "
        "shown.</div></div>"
    )
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.html(command)
    with right:
        st.html(context)

    telemetry_strip(summary)
    page_overview(scored, summary, embedded=True)


# Section: Pages
def _data():
    return load_scored(), load_summary()


def render_overview() -> None:
    page_home(*_data())


def render_classify() -> None:
    _, summary = _data()
    operational_context_banner()
    st.title("Classify")
    single, batch = st.tabs(["Single transaction", "Batch upload (CSV)"])
    with single:
        page_classify_transaction(summary, embedded=True)
    with batch:
        bundle = get_main_bundle()
        if bundle is not None:
            page_batch_classification(bundle, embedded=True)


def render_alerts() -> None:
    scored, _ = _data()
    operational_context_banner()
    page_alert_queue(scored)


def render_investigate() -> None:
    scored, _ = _data()
    operational_context_banner()
    bundle = get_main_bundle()
    if bundle is not None:
        page_investigation(scored, bundle)


def render_simulator() -> None:
    scored, summary = _data()
    operational_context_banner()
    page_threshold_simulator(scored, summary)


NAV_PAGES = [
    ("Overview", render_overview, "overview"),
    ("Classify", render_classify, "classify"),
    ("Alerts", render_alerts, "alerts"),
    ("Investigate", render_investigate, "investigate"),
    ("Simulator", render_simulator, "simulator"),
]


# Section: Main
def main():
    st.html(STYLE)
    pages = [st.Page(fn, title=title, url_path=path, default=(i == 0))
             for i, (title, fn, path) in enumerate(NAV_PAGES)]
    navigation = st.navigation(pages, position="hidden")

    missing = missing_core_outputs()
    if missing:
        brand_header()
        commands = sorted({CORE_OUTPUTS[name] for name in missing})
        st.error(
            "The console can't start because these outputs are missing: "
            + ", ".join(f"`outputs/{name}`" for name in missing)
            + ". From the repository root, run: " + " then ".join(f"`{c}`" for c in commands) + "."
        )
        st.stop()

    brand_header()
    navigation.run()


if __name__ == "__main__":
    main()
