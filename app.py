"""Self-serve data-product UI for a non-technical business stakeholder.

Run with:  streamlit run app.py

Zero API keys required: the business-question box always works via the
deterministic, grounded composer in src/insight_agent.py. If
ANTHROPIC_API_KEY is set in the environment, answers are phrased by Claude
(still strictly from the same computed numbers) - see insight_agent.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.insight_agent import (
    answer_business_question,
    churn_rate_by_segment,
    completion_rate_by_genre,
    top_churn_drivers,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# --- chart styling: one validated palette used consistently across charts ---
INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BAR_HUE = "#2a78d6"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.edgecolor": GRIDLINE,
        "axes.labelcolor": MUTED_INK,
        "text.color": INK,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)

st.set_page_config(page_title="Subscriber Engagement Data Product", layout="wide")


@st.cache_data
def load_overview():
    subscribers = pd.read_csv(DATA_DIR / "subscribers.csv")
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return subscribers, sessions, metrics


def bar_chart(labels, values, value_fmt="{:.1%}", horizontal=False, figsize=(6, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    positions = range(len(labels))
    if horizontal:
        ax.barh(positions, values, color=BAR_HUE, height=0.6)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        for i, v in enumerate(values):
            ax.text(v, i, f"  {value_fmt.format(v)}", va="center", color=INK, fontsize=9)
    else:
        ax.bar(positions, values, color=BAR_HUE, width=0.55)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels)
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(values):
            ax.text(i, v, value_fmt.format(v), ha="center", va="bottom", color=INK, fontsize=9)
    ax.grid(axis="x" if horizontal else "y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


st.title("Subscriber Engagement Data Product")
st.caption(
    "A self-serve dashboard over a real movie catalog (MovieLens) and a "
    "synthetic subscriber/engagement layer. See README.md 'Honest scope' for "
    "exactly what is real data vs. simulated."
)

subscribers, sessions, metrics = load_overview()

# --------------------------------------------------------------------------
# KPI tiles
# --------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Subscribers", f"{len(subscribers):,}")
col2.metric("Overall churn rate", f"{subscribers['churn'].mean():.1%}")
col3.metric("Avg completion rate", f"{sessions['pct_completed'].mean():.1%}")
col4.metric("Model ROC-AUC (churn)", f"{metrics['roc_auc']:.3f}")

st.divider()

# --------------------------------------------------------------------------
# Top churn drivers
# --------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Top churn drivers")
    st.caption(f"Attribution method: {metrics['attribution_method']}")
    drivers = metrics["top_churn_drivers"][:5]
    labels = [d["feature"] for d in drivers][::-1]
    values = [d["mean_abs_shap"] for d in drivers][::-1]
    fig = bar_chart(labels, values, value_fmt="{:.3f}", horizontal=True, figsize=(6, 3.2))
    st.pyplot(fig)

with right:
    st.subheader("Churn rate by segment")
    segment = st.selectbox("Segment", ["plan_tier", "region"], format_func=lambda s: s.replace("_", " ").title())
    seg_result = churn_rate_by_segment(segment)
    seg_df = pd.DataFrame(seg_result["by_group"]).sort_values("churn_rate", ascending=False)
    fig = bar_chart(seg_df["segment_value"].tolist(), seg_df["churn_rate"].tolist())
    st.pyplot(fig)

st.divider()

st.subheader("Completion rate by genre")
genre_result = completion_rate_by_genre(top_n=8)
genre_df = pd.DataFrame(genre_result["by_genre"])
fig = bar_chart(genre_df["genre"].tolist(), genre_df["avg_completion"].tolist(), figsize=(10, 3.2))
st.pyplot(fig)

st.divider()

# --------------------------------------------------------------------------
# Business-question box
# --------------------------------------------------------------------------
st.subheader("Ask a business question")
st.caption(
    "Answers are computed from this app's own data (pandas/DuckDB), never "
    "invented. If a question doesn't map to a supported analysis, the "
    "assistant says so explicitly. See PLAYBOOK.md for example questions."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. Which genre has the highest completion rate?",
)

if question:
    result = answer_business_question(question)
    if result["supported"]:
        st.success(result["answer"])
        with st.expander("Show supporting numbers"):
            st.json(result["grounded_data"])
    else:
        st.warning(result["answer"])
