"""
Self-Healing Agentic ML Pipeline — Streamlit Dashboard
=======================================================
A structured visualization and control interface for the goal-aware,
learning multi-agent self-healing ML pipeline.
"""

import io
import sys
import traceback

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Self-Healing ML Agent — Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1421 40%, #0a1628 100%);
        color: #e2e8f0;
    }

    /* ── Header ── */
    .dashboard-header {
        background: linear-gradient(135deg, #1a2744 0%, #0f1c35 100%);
        border: 1px solid rgba(99, 179, 237, 0.25);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .dashboard-title { font-size: 1.9rem; font-weight: 700; color: #90cdf4; margin: 0; }
    .dashboard-subtitle { font-size: 0.92rem; color: #718096; margin-top: 4px; }

    /* ── Section headers ── */
    .section-header {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #63b3ed;
        border-bottom: 1px solid rgba(99,179,237,0.18);
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a2744 0%, #162238 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: rgba(99, 179, 237, 0.5); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #90cdf4; font-family: 'JetBrains Mono', monospace; }
    .metric-label { font-size: 0.77rem; color: #718096; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }

    /* ── Status badge ── */
    .badge-success { background: rgba(72,187,120,0.18); color: #68d391; border: 1px solid rgba(72,187,120,0.4); padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .badge-failure { background: rgba(252,129,74,0.18); color: #fc8149; border: 1px solid rgba(252,129,74,0.4); padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .badge-neutral { background: rgba(160,174,192,0.18); color: #a0aec0; border: 1px solid rgba(160,174,192,0.35); padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

    /* ── Status pill ── */
    .status-SUCCESS { background: rgba(72,187,120,0.15); border: 1px solid rgba(72,187,120,0.4); color: #68d391; padding: 8px 20px; border-radius: 24px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
    .status-CLOSE   { background: rgba(246,224,94,0.12); border: 1px solid rgba(246,224,94,0.4); color: #f6e05e; padding: 8px 20px; border-radius: 24px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
    .status-FAILED  { background: rgba(252,129,74,0.15); border: 1px solid rgba(252,129,74,0.4); color: #fc8149; padding: 8px 20px; border-radius: 24px; font-weight: 700; font-size: 1.1rem; display: inline-block; }

    /* ── Timeline card ── */
    .timeline-card {
        background: linear-gradient(135deg, #1a2744 0%, #162238 100%);
        border: 1px solid rgba(99,179,237,0.15);
        border-left: 4px solid #63b3ed;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.89rem;
    }
    .timeline-card.success { border-left-color: #68d391; }
    .timeline-card.failure { border-left-color: #fc8149; }
    .timeline-attempt { font-size: 0.72rem; color: #718096; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }
    .timeline-action  { font-size: 1rem; font-weight: 600; color: #90cdf4; }
    .timeline-meta    { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 6px; font-size: 0.82rem; color: #a0aec0; }
    .timeline-meta span { font-family: 'JetBrains Mono', monospace; }

    /* ── Decision panel ── */
    .decision-panel {
        background: linear-gradient(135deg, #1a2744 0%, #162238 100%);
        border: 1px solid rgba(246,224,94,0.3);
        border-radius: 12px;
        padding: 22px 26px;
    }
    .decision-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #718096; margin-bottom: 4px; }
    .decision-value { font-size: 0.95rem; color: #e2e8f0; font-weight: 500; line-height: 1.5; }
    .decision-action-chip {
        display: inline-block;
        background: rgba(246,224,94,0.12);
        border: 1px solid rgba(246,224,94,0.4);
        color: #f6e05e;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Panel wrapper ── */
    .panel {
        background: linear-gradient(135deg, #1a2744 0%, #162238 100%);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
    }

    /* ── Confidence bar ── */
    .conf-bar-outer { background: rgba(255,255,255,0.06); border-radius: 4px; height: 8px; margin-top: 6px; }
    .conf-bar-inner { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #63b3ed, #90cdf4); }

    /* ── Streamlit widget overrides ── */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stFileUploader"] label { color: #a0aec0 !important; font-size: 0.85rem !important; }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #2b6cb0 0%, #1a4e8a 100%) !important;
        color: white !important;
        border: 1px solid rgba(99,179,237,0.4) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 28px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%) !important;
        border-color: rgba(99,179,237,0.7) !important;
    }

    .stExpander { border: 1px solid rgba(99,179,237,0.15) !important; border-radius: 10px !important; background: #111c30 !important; }
    .stExpander summary { color: #90cdf4 !important; font-weight: 500 !important; }

    /* ── Matplotlib / plotly backgrounds ── */
    .js-plotly-plot, .plotly { background: transparent !important; }

    /* ── Info/dividers ── */
    hr { border-color: rgba(99,179,237,0.12) !important; }
    .empty-state { text-align: center; color: #4a5568; padding: 48px 0; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

for key in ("result", "run_logs", "uploaded_df", "is_running"):
    if key not in st.session_state:
        st.session_state[key] = None

if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (mirrored from pipeline.py to avoid import side-effects)
# ─────────────────────────────────────────────────────────────────────────────

F1_THRESHOLD = 0.80
CLOSE_ENOUGH_DELTA = 0.06   # near-goal band
FAILURE_TYPES = [
    "none",
    "MISSING_COLUMN",
    "TYPE_MISMATCH",
    "LOW_PERFORMANCE",
    "COMPOUND",
    "PARTIAL_CORRUPTION",
    "NEAR_GOAL",
]

ACTION_COLORS = {
    "REBALANCE_DATA":      "#63b3ed",
    "CHANGE_MODEL":        "#9f7aea",
    "FEATURE_SELECTION":   "#68d391",
    "ADD_MISSING_COLUMNS": "#fc8149",
    "CAST_DATATYPE":       "#f6e05e",
    "TUNE_HYPERPARAMETERS":"#76e4f7",
    "STOP":                "#718096",
    "":                    "#4a5568",
}

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────

def apply_dark_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor("#0d1421")
    ax.set_facecolor("#111c30")
    ax.tick_params(colors="#718096", labelsize=9)
    ax.xaxis.label.set_color("#a0aec0")
    ax.yaxis.label.set_color("#a0aec0")
    ax.title.set_color("#90cdf4")
    for spine in ax.spines.values():
        spine.set_edgecolor((0.39, 0.70, 0.93, 0.18))   # #63b3ed at 18% alpha as RGBA tuple
        spine.set_linewidth(0.8)
    ax.grid(color="#1e3a5f", linewidth=0.5, linestyle="--", alpha=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 0 — HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown(
        """
        <div class="dashboard-header">
            <div>
                <div style="font-size:2.4rem; line-height:1;">🤖</div>
            </div>
            <div>
                <p class="dashboard-title">Self-Healing Agentic ML Pipeline</p>
                <p class="dashboard-subtitle">
                    Goal-aware · LangGraph + Groq LLM · 7-Agent architecture ·
                    Real-time reasoning transparency dashboard
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — PIPELINE CONTROL
# ─────────────────────────────────────────────────────────────────────────────

# CSV → failure mode pairing guide (for the info box)
_CSV_GUIDE = {
    "clean_baseline.csv":       "none",
    "missing_column.csv":       "MISSING_COLUMN",
    "type_mismatch.csv":        "TYPE_MISMATCH",
    "low_performance.csv":      "LOW_PERFORMANCE",
    "compound.csv":             "COMPOUND",
    "near_goal.csv":            "NEAR_GOAL",
    "partial_corruption.csv":   "PARTIAL_CORRUPTION",
}

def render_control_panel() -> tuple:
    st.markdown('<div class="section-header">⚙️ Pipeline Control Panel</div>', unsafe_allow_html=True)

    col_upload, col_failure, col_run = st.columns([3, 2, 1], gap="medium")

    uploaded_df   = None
    suggested_mode = None

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload CSV dataset (optional — uses synthetic data if omitted)",
            type=["csv"],
            key="csv_upload",
            help="Upload one of the test CSVs generated by generate_test_datasets.py",
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                uploaded_df = df
                st.success(
                    f"✅ Loaded **{uploaded_file.name}** — "
                    f"{len(df):,} rows × {len(df.columns)} cols"
                )
                # Auto-suggest the matching failure mode
                suggested_mode = _CSV_GUIDE.get(uploaded_file.name)
                if suggested_mode:
                    st.markdown(
                        f'<div style="margin-top:6px; padding:7px 12px; '
                        f'background:rgba(99,179,237,0.08); border:1px solid rgba(99,179,237,0.25); '
                        f'border-radius:8px; font-size:0.82rem; color:#90cdf4;">'
                        f'💡 Suggested failure mode for this file: '
                        f'<strong>{suggested_mode}</strong></div>',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
        else:
            # Clear stale upload from session
            st.session_state.uploaded_df = None

    with col_failure:
        # Pre-select suggested mode if we just detected one
        default_idx = 0
        if suggested_mode and suggested_mode in FAILURE_TYPES:
            default_idx = FAILURE_TYPES.index(suggested_mode)
        failure_type = st.selectbox(
            "Failure injection mode",
            FAILURE_TYPES,
            index=default_idx,
            key="failure_select",
            help=(
                "When a CSV is uploaded: the agent trains on that CSV directly.\n"
                "When no CSV is uploaded: the pipeline generates synthetic data "
                "and injects this failure type internally."
            ),
        )

    with col_run:
        st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
        run_clicked = st.button("▶ Run Pipeline", key="run_btn", use_container_width=True)

    # ── Usage hint strip ──
    has_upload = st.session_state.get("uploaded_df") is not None
    if has_upload:
        st.markdown(
            '<div style="font-size:0.8rem; color:#68d391; margin-top:6px;">'
            '📂 <strong>CSV mode</strong> — the agent will train on your uploaded data. '
            'The dropdown is informational only.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.8rem; color:#718096; margin-top:6px;">'
            '🔬 <strong>Synthetic mode</strong> — pipeline generates 500-row data internally '
            'and injects the selected failure. Upload a CSV to use your own data.</div>',
            unsafe_allow_html=True,
        )

    return run_clicked, failure_type, uploaded_df


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_safe(failure_type: str, uploaded_df: "pd.DataFrame | None" = None):
    """
    Call run_pipeline(), optionally injecting an uploaded DataFrame.

    If `uploaded_df` is provided the pipeline skips its internal synthetic
    data generator and operates directly on the uploaded CSV instead.
    The failure_type dropdown still controls what the agent is told to
    expect (and is used when NO csv is uploaded).
    """
    from pipeline import (
        run_pipeline,
        build_pipeline,
        generate_synthetic_data,
        inject_failures,
        F1_THRESHOLD,
    )  # local import to avoid top-level side-effects

    capture   = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = capture

    result    = None
    error_msg = None
    try:
        ft = None if failure_type == "none" else failure_type

        if uploaded_df is not None:
            # ── Use the uploaded CSV as the dataset ──────────────────────
            # We bypass run_pipeline() and call build_pipeline() directly
            # so we can inject our own dataframe.
            print("=" * 65)
            print("  GOAL-AWARE SELF-HEALING ML PIPELINE  —  v3")
            print("  Dataset source : *** UPLOADED CSV ***")
            if ft:
                print(f"  Failure mode   : {ft} (from dashboard dropdown)")
            else:
                print("  Failure mode   : none (clean run on uploaded data)")
            print("=" * 65)

            df = uploaded_df.copy()

            initial_state = {
                "df":                   df,
                "error":                "",
                "action":               "",
                "f1":                   0.0,
                "best_f1":              0.0,
                "attempt":              1,
                "history":              [],
                "done":                 False,
                "model_type":           "logistic",
                "model_params":         {},
                "root_cause":           "",
                "reasoning":            "",
                "confidence":           0.0,
                "goal_distance":        F1_THRESHOLD,
                "near_goal":            False,
                "action_scores":        {},
                "action_success_count": {},
                "action_failure_count": {},
                "failed_actions":       [],
                "best_action":          "",
            }
            result = build_pipeline().invoke(initial_state)
        else:
            # ── Normal run with synthetic data + optional failure injection ──
            result = run_pipeline(failure_type=ft)

    except Exception:
        error_msg = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    logs = capture.getvalue()
    return result, logs, error_msg


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — LIVE PIPELINE TRACE
# ─────────────────────────────────────────────────────────────────────────────

def render_pipeline_trace(history: list) -> None:
    st.markdown('<div class="section-header">📋 Live Pipeline Trace</div>', unsafe_allow_html=True)

    if not history:
        st.markdown('<div class="empty-state">No execution history yet. Run the pipeline first.</div>', unsafe_allow_html=True)
        return

    for entry in history:
        attempt   = entry.get("attempt", "?")
        action    = entry.get("action", "—")
        f1        = entry.get("f1", 0.0)
        delta     = entry.get("improvement", 0.0)
        success   = entry.get("success", False)
        root_cause = entry.get("root_cause", "—")
        confidence = entry.get("confidence", 0.0)
        error     = entry.get("error", "")

        card_class = "success" if success else "failure"
        badge      = '<span class="badge-success">✅ SUCCESS</span>' if success else '<span class="badge-failure">❌ NO IMPROVEMENT</span>'
        delta_color = "#68d391" if delta > 0 else ("#fc8149" if delta < 0 else "#a0aec0")
        action_color = ACTION_COLORS.get(action, "#90cdf4")
        conf_pct = int(confidence * 100)

        st.markdown(
            f"""
            <div class="timeline-card {card_class}">
                <div class="timeline-attempt">Attempt #{attempt} &nbsp;·&nbsp; {error or "No structural error"}</div>
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:6px;">
                    <div class="timeline-action" style="color:{action_color};">{action}</div>
                    {badge}
                </div>
                <div class="timeline-meta">
                    <span>F1&nbsp;<strong style="color:#90cdf4;">{f1:.4f}</strong></span>
                    <span>Δ&nbsp;<strong style="color:{delta_color};">{delta:+.4f}</strong></span>
                    <span>Conf&nbsp;<strong style="color:#a0aec0;">{confidence:.0%}</strong></span>
                </div>
                <div style="margin-top:8px; font-size:0.82rem; color:#718096;">
                    🔍&nbsp;<em>{root_cause or "—"}</em>
                </div>
                <div class="conf-bar-outer" style="margin-top:8px;">
                    <div class="conf-bar-inner" style="width:{conf_pct}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — PERFORMANCE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def render_performance_graph(history: list) -> None:
    st.markdown('<div class="section-header">📈 F1 Score vs. Attempt</div>', unsafe_allow_html=True)

    if not history:
        st.markdown('<div class="empty-state">Run the pipeline to see the performance graph.</div>', unsafe_allow_html=True)
        return

    attempts = [h["attempt"] for h in history]
    f1s      = [h["f1"]      for h in history]
    successes = [h.get("success", False) for h in history]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    apply_dark_theme(fig, ax)

    # Goal line
    ax.axhline(y=F1_THRESHOLD, color="#f6e05e", linewidth=1.2, linestyle="--", alpha=0.7, label=f"Goal (F1={F1_THRESHOLD})")
    # Near-goal band
    ax.axhspan(F1_THRESHOLD - CLOSE_ENOUGH_DELTA, F1_THRESHOLD, color="#f6e05e", alpha=0.04, label="Near-goal zone")

    # Trend line
    ax.plot(attempts, f1s, color="#63b3ed", linewidth=2.2, zorder=2, linestyle="-", marker="", alpha=0.8)

    # Scatter: colour by success
    for i, (att, f1, ok) in enumerate(zip(attempts, f1s, successes)):
        color = "#68d391" if ok else "#fc8149"
        ax.scatter(att, f1, color=color, s=72, zorder=3, edgecolors="white", linewidths=0.6)

    # Best F1 annotation
    best_f1 = max(f1s)
    best_idx = f1s.index(best_f1)
    ax.annotate(
        f" Best: {best_f1:.4f}",
        xy=(attempts[best_idx], best_f1),
        fontsize=8, color="#90cdf4",
        xytext=(4, 6), textcoords="offset points",
    )

    ax.set_xlabel("Attempt", fontsize=10)
    ax.set_ylabel("F1 Score (weighted)", fontsize=10)
    ax.set_title("Performance Trajectory", fontsize=11, pad=10)
    ax.set_xticks(attempts)
    ax.set_ylim(max(0, min(f1s) - 0.08), min(1.0, max(f1s) + 0.08))

    legend = ax.legend(fontsize=8, facecolor="#111c30", edgecolor="#2d4a7a", labelcolor="#a0aec0")
    # Success/failure legend patches
    legend.legend_handles.extend([
        mpatches.Patch(color="#68d391", label="Improved"),
        mpatches.Patch(color="#fc8149", label="No improvement"),
    ])
    ax.legend(handles=legend.legend_handles + [
        mpatches.Patch(color="#68d391", label="Improved"),
        mpatches.Patch(color="#fc8149", label="No improvement"),
    ], fontsize=8, facecolor="#111c30", edgecolor="#2d4a7a", labelcolor="#a0aec0")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4 — AGENT DECISION PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_decision_panel(result: dict) -> None:
    st.markdown('<div class="section-header">🧠 Latest Agent Decision</div>', unsafe_allow_html=True)

    root_cause = result.get("root_cause", "—") or "—"
    reasoning  = result.get("reasoning",  "—") or "—"
    confidence = result.get("confidence", 0.0)
    action     = result.get("action",     "—") or "—"
    near_goal  = result.get("near_goal",  False)
    conf_pct   = int(confidence * 100)
    conf_color = "#68d391" if confidence >= 0.7 else ("#f6e05e" if confidence >= 0.5 else "#fc8149")

    st.markdown(
        f"""
        <div class="decision-panel">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px; margin-bottom:18px;">
                <div>
                    <div class="decision-label">Action Selected</div>
                    <div><span class="decision-action-chip">{action}</span></div>
                </div>
                <div style="text-align:right;">
                    <div class="decision-label">Confidence</div>
                    <div style="font-size:1.5rem; font-weight:700; color:{conf_color}; font-family:'JetBrains Mono',monospace;">{confidence:.0%}</div>
                    <div class="conf-bar-outer" style="width:100px; margin-left:auto; margin-top:4px;">
                        <div class="conf-bar-inner" style="width:{conf_pct}%; background:linear-gradient(90deg,{conf_color},{conf_color}88);"></div>
                    </div>
                </div>
            </div>
            <div style="margin-bottom:14px;">
                <div class="decision-label">Root Cause</div>
                <div class="decision-value">🔍 {root_cause}</div>
            </div>
            <div style="margin-bottom:14px;">
                <div class="decision-label">Reasoning</div>
                <div class="decision-value" style="color:#cbd5e0; font-size:0.88rem; line-height:1.65;">💭 {reasoning}</div>
            </div>
            {"<div style='margin-top:10px; padding:8px 14px; background:rgba(246,224,94,0.08); border:1px solid rgba(246,224,94,0.25); border-radius:8px; font-size:0.83rem; color:#f6e05e;'>🎯 <strong>Near-goal mode active</strong> — conservative actions preferred (TUNE_HYPERPARAMETERS)</div>" if near_goal else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 5 — MEMORY + LEARNING PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_memory_panel(result: dict) -> None:
    st.markdown('<div class="section-header">🧬 Memory & Learning</div>', unsafe_allow_html=True)

    action_scores  = result.get("action_scores",  {})
    best_action    = result.get("best_action",    "—")
    failed_actions = result.get("failed_actions", [])

    col_chart, col_info = st.columns([2, 1], gap="large")

    with col_chart:
        if action_scores:
            actions = list(action_scores.keys())
            scores  = list(action_scores.values())
            colors  = [ACTION_COLORS.get(a, "#63b3ed") for a in actions]

            fig, ax = plt.subplots(figsize=(7, max(2.5, len(actions) * 0.55)))
            apply_dark_theme(fig, ax)

            bars = ax.barh(actions, scores, color=colors, height=0.55, edgecolor="none")

            # Zero reference line
            ax.axvline(0, color="#718096", linewidth=0.8, linestyle="-")

            # Value labels
            for bar, val in zip(bars, scores):
                x_pos = val + (0.002 if val >= 0 else -0.002)
                ha = "left" if val >= 0 else "right"
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}", va="center", ha=ha,
                        fontsize=8.5, color="#e2e8f0", fontfamily="monospace")

            ax.set_xlabel("Cumulative Score (F1 delta impact)", fontsize=9)
            ax.set_title("Action Score Leaderboard", fontsize=10, pad=8)
            ax.tick_params(axis="y", labelsize=9)
            ax.set_facecolor("#111c30")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.markdown('<div class="empty-state" style="padding:20px">No action scores yet.</div>', unsafe_allow_html=True)

    with col_info:
        # Best action
        best_color = ACTION_COLORS.get(best_action, "#90cdf4")
        st.markdown(
            f"""
            <div class="panel" style="margin-bottom:14px;">
                <div class="decision-label">🏆 Best Action</div>
                <div style="font-size:1.1rem; font-weight:700; color:{best_color}; font-family:'JetBrains Mono',monospace; margin-top:6px;">{best_action or "—"}</div>
                <div style="font-size:0.78rem; color:#718096; margin-top:4px;">Highest cumulative F1 contribution</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Failed / Ineffective actions
        st.markdown('<div class="decision-label" style="margin-bottom:8px;">🚫 Ineffective Actions</div>', unsafe_allow_html=True)
        if failed_actions:
            for fa in failed_actions:
                st.markdown(
                    f'<div style="background:rgba(252,129,74,0.1); border:1px solid rgba(252,129,74,0.3); '
                    f'border-radius:8px; padding:6px 12px; margin-bottom:6px; '
                    f'font-family:JetBrains Mono,monospace; font-size:0.87rem; color:#fc8149;">'
                    f'❌&nbsp;{fa}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#4a5568; font-size:0.85rem; font-style:italic;">None marked ineffective</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 6 — DATA HEALTH PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_data_health_panel(result: dict) -> None:
    st.markdown('<div class="section-header">🩺 Data Health</div>', unsafe_allow_html=True)

    df: pd.DataFrame = result.get("df", None)
    if df is None or not isinstance(df, pd.DataFrame):
        st.markdown('<div class="empty-state">Data frame not available in result.</div>', unsafe_allow_html=True)
        return

    required_cols = ["age", "gender", "income", "target"]
    target_col    = "target"

    col_left, col_mid, col_right = st.columns(3, gap="medium")

    # ── Left: Missing columns ──
    with col_left:
        st.markdown('<div class="decision-label">Missing Required Columns</div>', unsafe_allow_html=True)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            for m in missing:
                st.markdown(f'<span class="badge-failure">❌ {m}</span><br>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-success">✅ All columns present</span>', unsafe_allow_html=True)

    # ── Mid: Null counts ──
    with col_mid:
        st.markdown('<div class="decision-label">Null Counts per Column</div>', unsafe_allow_html=True)
        null_counts = df.isnull().sum()
        null_df = pd.DataFrame({
            "Column": null_counts.index,
            "Nulls":  null_counts.values,
            "Rate":   (null_counts.values / len(df) * 100).round(1),
        })
        null_df["Rate"] = null_df["Rate"].astype(str) + "%"
        st.dataframe(
            null_df.style.applymap(
                lambda v: "color: #fc8149;" if isinstance(v, (int, float)) and v > 0 else "color: #68d391;",
                subset=["Nulls"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    # ── Right: Class distribution ──
    with col_right:
        st.markdown('<div class="decision-label">Target Class Distribution</div>', unsafe_allow_html=True)
        if target_col in df.columns:
            dist = df[target_col].value_counts()
            total = len(df)
            for cls, cnt in dist.items():
                pct = cnt / total * 100
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; font-size:0.87rem; margin-bottom:4px;">'
                    f'<span style="color:#a0aec0;">Class {cls}</span>'
                    f'<span style="color:#90cdf4; font-family:JetBrains Mono,monospace;">{cnt}&nbsp;({pct:.1f}%)</span>'
                    f'</div>'
                    f'<div class="conf-bar-outer" style="margin-bottom:8px;">'
                    f'<div class="conf-bar-inner" style="width:{pct}%;"></div></div>',
                    unsafe_allow_html=True,
                )
            ratio = dist.min() / dist.max() if len(dist) > 1 and dist.max() > 0 else 1.0
            if ratio < 0.4:
                st.markdown(
                    '<div style="font-size:0.8rem; color:#fc8149; margin-top:6px;">⚠️ Class imbalance detected</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div style="color:#4a5568; font-size:0.85rem;">Target column not found</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 7 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _compute_status(final_f1: float, goal_distance: float) -> str:
    if goal_distance <= 0:
        return "SUCCESS"
    if goal_distance <= CLOSE_ENOUGH_DELTA:
        return "CLOSE"
    return "FAILED"


def render_final_summary(result: dict) -> None:
    st.markdown('<div class="section-header">🏁 Final Summary</div>', unsafe_allow_html=True)

    final_f1      = result.get("f1",           0.0)
    best_f1       = result.get("best_f1",       0.0)
    goal_distance = result.get("goal_distance", F1_THRESHOLD)
    history       = result.get("history",       [])
    total_attempts = len(history)
    final_action  = history[-1]["action"] if history else "—"
    status        = _compute_status(final_f1, goal_distance)

    dist_color = "#68d391" if goal_distance <= 0 else ("#f6e05e" if goal_distance <= CLOSE_ENOUGH_DELTA else "#fc8149")

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    for col, label, value, fmt in [
        (col1, "Final F1",      final_f1,      f"{final_f1:.4f}"),
        (col2, "Best F1",       best_f1,       f"{best_f1:.4f}"),
        (col3, "Total Attempts",total_attempts, str(total_attempts)),
        (col4, "Goal Distance", goal_distance, f"{goal_distance:+.4f}"),
    ]:
        with col:
            color = dist_color if label == "Goal Distance" else "#90cdf4"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color};">{fmt}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Final action + status
    st.markdown("<div style='margin-top:18px; display:flex; gap:16px; align-items:center; flex-wrap:wrap;'>", unsafe_allow_html=True)
    action_color = ACTION_COLORS.get(final_action, "#90cdf4")
    st.markdown(
        f"""
        <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-top:18px;">
            <div>
                <div class="decision-label">Final Action</div>
                <span class="decision-action-chip" style="color:{action_color}; border-color:{action_color}44; background:{action_color}14;">{final_action}</span>
            </div>
            <div>
                <div class="decision-label">Pipeline Status</div>
                <span class="status-{status}">
                    {"✅ SUCCESS — Goal Reached" if status == "SUCCESS" else
                     "🎯 CLOSE — Near Goal" if status == "CLOSE" else
                     "❌ FAILED — Below Goal"}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 8 — RUN LOGS  (collapsible)
# ─────────────────────────────────────────────────────────────────────────────

def render_run_logs(logs: str) -> None:
    if not logs:
        return
    with st.expander("📜 Full Agent Execution Logs", expanded=False):
        st.code(logs, language="text")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    render_header()

    # ── Control Panel ──
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    run_clicked, failure_type, uploaded_df = render_control_panel()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Run pipeline when button clicked ──
    if run_clicked:
        st.session_state.result    = None
        st.session_state.run_logs  = None

        # Prefer freshly uploaded df; fall back to one stored in session
        df_to_use = uploaded_df if uploaded_df is not None else st.session_state.get("uploaded_df")

        with st.spinner("🤖 Agent is diagnosing and healing the pipeline…  (this may take 30–90 s)"):
            result, logs, err = run_pipeline_safe(failure_type, uploaded_df=df_to_use)

        if err:
            st.error(f"**Pipeline execution error:**\n\n```\n{err}\n```")
            st.session_state.run_logs = logs
        else:
            st.session_state.result   = result
            st.session_state.run_logs = logs
            if df_to_use is not None:
                st.success("✅ Pipeline run complete! (trained on uploaded CSV)")
            else:
                st.success("✅ Pipeline run complete! (trained on synthetic data)")

    result   = st.session_state.result
    run_logs = st.session_state.run_logs

    # ── Empty state ──
    if result is None:
        st.markdown(
            """
            <div class="empty-state" style="padding:80px 0;">
                <div style="font-size:3rem; margin-bottom:16px;">🤖</div>
                <div style="font-size:1.15rem; color:#4a5568; margin-bottom:8px;">No pipeline run yet</div>
                <div style="font-size:0.9rem; color:#2d3748;">
                    Select a failure mode above and click <strong>▶ Run Pipeline</strong> to start the agent.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if run_logs:
            render_run_logs(run_logs)
        return

    history = result.get("history", [])

    # ── Layout: Trace + Performance + Decision ──
    st.markdown("---")
    col_trace, col_right = st.columns([1, 1], gap="large")

    with col_trace:
        render_pipeline_trace(history)

    with col_right:
        render_performance_graph(history)
        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        render_decision_panel(result)

    # ── Memory + Learning ──
    st.markdown("---")
    render_memory_panel(result)

    # ── Data Health ──
    st.markdown("---")
    render_data_health_panel(result)

    # ── Final Summary ──
    st.markdown("---")
    render_final_summary(result)

    # ── Execution Logs ──
    st.markdown("---")
    render_run_logs(run_logs)


if __name__ == "__main__":
    main()
