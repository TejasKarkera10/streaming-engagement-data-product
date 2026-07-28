"""Self-serve data-product UI for a non-technical business stakeholder.

Run with:  streamlit run app.py

Zero API keys required: the business-question box always works via the
deterministic, grounded composer in src/insight_agent.py. If
ANTHROPIC_API_KEY is set in the environment, answers are phrased by Claude
(still strictly from the same computed numbers) - see insight_agent.py.

Chart styling follows a validated, colorblind-safe palette (one accent hue
per metric, hairline recessive gridlines, direct value labels, real hover
tooltips) rather than framework defaults - see the project README for the
full "Honest scope" section this page summarizes inline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.insight_agent import (
    answer_business_question,
    churn_rate_by_segment,
    completion_rate_by_genre,
    engagement_trend_over_time,
    top_churn_drivers,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPO_URL = "https://github.com/TejasKarkera10/streaming-engagement-data-product"

# --- validated chart palette: one accent hue per metric, text stays in ink
# tokens (never the series color), gridlines one step off the surface. ---
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
ACCENT = "#2a78d6"
SURFACE = "#fcfcfb"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

st.set_page_config(page_title="Subscriber Engagement Data Product", page_icon="📺", layout="wide")


# ---------------------------------------------------------------------------
# Chart helpers - shared layout/spec so every chart reads as one system.
# ---------------------------------------------------------------------------
def _base_layout(height: int) -> dict:
    return dict(
        height=height,
        margin=dict(l=8, r=36, t=8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=SECONDARY_INK, size=13),
        showlegend=False,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRIDLINE, font=dict(family=FONT, color=INK, size=13)),
    )


def bar_chart(labels, values, value_fmt="{:.1%}", horizontal=False, height=320):
    """Single-series bar/column: one accent hue, direct labels at the tip,
    hairline gridlines on the value axis only, real hover tooltip."""
    text = [value_fmt.format(v) for v in values]
    headroom = (max(values) if values else 1) * 1.22

    if horizontal:
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=ACCENT, cornerradius=4),
                text=text,
                textposition="outside",
                textfont=dict(color=INK, size=13),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed", showgrid=False, color=MUTED_INK, ticks="")
        fig.update_xaxes(range=[0, headroom], showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=False, color=MUTED_INK)
        fig.update_layout(bargap=0.45, **_base_layout(height))
    else:
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=ACCENT, cornerradius=4),
                text=text,
                textposition="outside",
                textfont=dict(color=INK, size=13),
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>",
            )
        )
        fig.update_xaxes(showgrid=False, color=MUTED_INK, ticks="")
        fig.update_yaxes(range=[0, headroom], showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=False, color=MUTED_INK, showticklabels=False)
        fig.update_layout(bargap=0.4, **_base_layout(height))
    return fig


def line_chart(x, y, value_fmt="{:.1%}", height=260):
    """Single-series trend line: 2px line, ringed end-markers, one hue."""
    text = [value_fmt.format(v) for v in y]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8, color=ACCENT, line=dict(width=2, color=SURFACE)),
            text=text,
            hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=False, color=MUTED_INK)
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=False, color=MUTED_INK)
    fig.update_layout(**_base_layout(height))
    return fig


@st.cache_data
def load_overview():
    subscribers = pd.read_csv(DATA_DIR / "subscribers.csv")
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return subscribers, sessions, metrics


def _set_question(q: str) -> None:
    st.session_state["question_input"] = q


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Subscriber Engagement Data Product")
st.markdown(
    "A self-serve analytics tool over a **real movie catalog** (MovieLens) and a "
    "**synthetic subscriber-engagement layer**, with a business-question box that "
    "only ever answers from numbers computed in this app - never invented."
)

badge_cols = st.columns(4)
badge_cols[0].markdown("🎬 &nbsp;**9,742** real movies")
badge_cols[1].markdown("⭐ &nbsp;**100,836** real ratings")
badge_cols[2].markdown("👥 &nbsp;**3,000** synthetic subscribers")
badge_cols[3].markdown("🔑 &nbsp;**Zero API keys** required")

with st.expander("What is this, and what's real vs. simulated?"):
    st.markdown(
        f"""
This is a **solo demo data product** - not a production system with real users
or measured business impact. It exists to show the full loop a data-product
role runs day to day: build the data, engineer features, train and evaluate a
model, turn that into a constrained GenAI question-answering layer with a
grounding guardrail, and ship it as something a business stakeholder can use
unassisted.

- **Real:** the MovieLens `ml-latest-small` catalog and ratings (9,742 movies,
  real titles/genres; 100,836 real user ratings - GroupLens Research).
- **Synthetic, explicitly disclosed:** the subscriber base, viewing sessions,
  and churn label. Generated by a documented, seeded rule tying engagement
  (session frequency, completion, recency) to churn probability plus noise -
  it is realistic and learnable, but **not real subscriber behavior**.
- **No fabricated impact:** every number below is either a real MovieLens
  statistic or a model-evaluation metric on a held-out split - never an
  invented percent-improvement claim.

Full disclosure, including why the model's ROC-AUC is partly circular by
construction: see the [`README` "Honest scope"]({REPO_URL}) section.
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------
subscribers, sessions, metrics = load_overview()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Subscribers", f"{len(subscribers):,}")
col1.caption("Synthetic subscriber base, seeded and reproducible.")
col2.metric("Overall churn rate", f"{subscribers['churn'].mean():.1%}")
col2.caption("Share of subscribers labeled churned (a static snapshot label).")
col3.metric("Avg completion rate", f"{sessions['pct_completed'].mean():.1%}")
col3.caption("Average share of a title watched per session.")
col4.metric("Model ROC-AUC (churn)", f"{metrics['roc_auc']:.3f}")
col4.caption("Held-out test split; see 'Honest scope' above for caveats.")

st.divider()

# ---------------------------------------------------------------------------
# Top churn drivers + churn by segment
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Top churn drivers")
    st.caption(
        f"Attribution: {metrics['attribution_method']} on the held-out test set. "
        "Longer bars mean the model leans on that signal more heavily when "
        "predicting churn - an attribution of the model, not a causal claim."
    )
    drivers = metrics["top_churn_drivers"][:5]
    labels = [d["feature"] for d in drivers][::-1]
    values = [d["mean_abs_shap"] for d in drivers][::-1]
    st.plotly_chart(bar_chart(labels, values, value_fmt="{:.3f}", horizontal=True), width="stretch")

with right:
    st.subheader("Churn rate by segment")
    segment = st.selectbox("Segment", ["plan_tier", "region"], format_func=lambda s: s.replace("_", " ").title())
    seg_result = churn_rate_by_segment(segment)
    seg_df = pd.DataFrame(seg_result["by_group"]).sort_values("churn_rate", ascending=False)
    spread = seg_df["churn_rate"].max() - seg_df["churn_rate"].min()
    st.caption(
        f"Churn ranges from {seg_df['churn_rate'].min():.1%} to {seg_df['churn_rate'].max():.1%} "
        f"across {segment.replace('_', ' ')} groups (a {spread:.1%}-point spread), against an "
        f"overall rate of {seg_result['overall_churn_rate']:.1%}."
    )
    st.plotly_chart(
        bar_chart(seg_df["segment_value"].tolist(), seg_df["churn_rate"].tolist()),
        width="stretch",
    )

st.divider()

# ---------------------------------------------------------------------------
# Engagement trend over time (two single-axis charts - never dual-axis)
# ---------------------------------------------------------------------------
st.subheader("Engagement trend")
st.caption(
    "Monthly session volume and average completion, computed directly from "
    "the session log. This dataset stores churn as one static label per "
    "subscriber, not a time series - so this is an engagement trend, not a "
    "churn trend (the assistant explains this distinction if you ask about a "
    "churn 'spike' below)."
)
trend = engagement_trend_over_time(months=6)["by_month"]
trend_df = pd.DataFrame(trend)

t_left, t_right = st.columns(2)
with t_left:
    st.caption("Sessions per month")
    st.plotly_chart(
        bar_chart(trend_df["month"].tolist(), trend_df["session_count"].tolist(), value_fmt="{:,.0f}"),
        width="stretch",
    )
with t_right:
    st.caption("Avg completion rate per month")
    st.plotly_chart(
        line_chart(trend_df["month"].tolist(), trend_df["avg_completion"].tolist(), height=320),
        width="stretch",
    )

st.divider()

# ---------------------------------------------------------------------------
# Completion rate by genre
# ---------------------------------------------------------------------------
st.subheader("Completion rate by genre")
genre_result = completion_rate_by_genre(top_n=8)
genre_df = pd.DataFrame(genre_result["by_genre"])
genre_spread = genre_df["avg_completion"].max() - genre_df["avg_completion"].min()
if genre_spread < 0.03:
    st.caption(
        f"Completion barely varies by genre in this dataset (top-to-bottom spread: "
        f"{genre_spread:.1%}) - consistent with 'genre diversity' ranking lowest among "
        "the churn model's top drivers above. In this simulation, how much someone "
        "watches is driven by their overall engagement level, not what genre it is."
    )
else:
    st.caption(
        f"{genre_df.iloc[0]['genre']} leads at {genre_df.iloc[0]['avg_completion']:.1%} "
        f"average completion; spread top-to-bottom is {genre_spread:.1%}."
    )
st.plotly_chart(
    bar_chart(genre_df["genre"].tolist(), genre_df["avg_completion"].tolist(), height=340),
    width="stretch",
)

st.divider()

# ---------------------------------------------------------------------------
# Business-question box
# ---------------------------------------------------------------------------
st.subheader("Ask a business question")
st.caption(
    "Answers are computed from this app's own data (pandas/DuckDB), never "
    "invented. If a question doesn't map to a supported analysis, the "
    "assistant says so explicitly instead of guessing."
)

example_questions = [
    "What is the churn rate by plan tier?",
    "What are the top churn drivers?",
    "Which genre has the highest completion rate?",
    "What is the engagement trend over the last few months?",
    "Why did churn spike for Premium subscribers last quarter?",
]
st.caption("Try one of these, or type your own question below:")
ex_cols = st.columns(len(example_questions))
for i, q in enumerate(example_questions):
    ex_cols[i].button(q, key=f"example_{i}", on_click=_set_question, args=(q,), width="stretch")

question = st.text_input(
    "Your question",
    key="question_input",
    placeholder="e.g. Which genre has the highest completion rate?",
)

if question:
    result = answer_business_question(question)
    if result["supported"]:
        st.success(result["answer"])
        st.caption(f"Computed via the `{result['intent']}` analysis - see PLAYBOOK.md for what each one covers.")
        with st.expander("Show supporting numbers"):
            st.json(result["grounded_data"])
    else:
        st.warning(result["answer"])

st.divider()
st.caption(
    f"[Source on GitHub]({REPO_URL}) · [PLAYBOOK.md]({REPO_URL}/blob/main/PLAYBOOK.md) "
    f"(metric definitions & example questions) · [PRODUCT_BRIEF.md]({REPO_URL}/blob/main/PRODUCT_BRIEF.md) "
    "(scope, KPIs, governance)"
)
