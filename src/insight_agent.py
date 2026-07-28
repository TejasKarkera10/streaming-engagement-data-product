"""Answer natural-language business questions with grounded computation.

This is a constrained business-question layer, not an open-ended chat
agent. It supports exactly four analyses, each backed by a real pandas/
DuckDB computation over data/*.csv and models/metrics.json:

  1. churn_rate_by_segment   - churn rate grouped by plan_tier or region
  2. top_churn_drivers       - precomputed SHAP feature importances from the
                               trained model (src/model.py, models/metrics.json)
  3. completion_rate_by_genre - avg pct_completed grouped by MovieLens genre
  4. engagement_trend_over_time - monthly session volume / completion rate

`answer_business_question(question)` is the single entry point used by the
Streamlit app and the MCP server. It:
  - routes the question to one of the four analyses with a small rule-based
    keyword router (deterministic and unit-testable - no LLM required to
    decide WHAT to compute),
  - runs the real computation and keeps the numeric result as `grounded_data`,
  - composes a natural-language answer STRICTLY from those numbers, either
    with a deterministic template (default, zero API keys) or, if
    ANTHROPIC_API_KEY is set, by asking Claude to phrase (not invent) the
    same numbers via langchain-anthropic - mirroring the grounded-composer
    pattern in the sibling clinical-trials-genai-assistant project's
    backend/app/planner.py.

Grounding guardrail: if a question does not match any of the four supported
analyses, `answer_business_question` returns `supported: False` and an
explicit "I can't answer that with the analyses this tool supports" message
instead of guessing. This is the single most important property of this
module - every number in a returned answer traces back to a pandas/DuckDB
computation in this file, never to LLM invention.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SEGMENT_COLUMNS = {"plan_tier", "region"}
KNOWN_REGIONS = {"north america", "europe", "apac", "latin america"}
KNOWN_TIERS = {"basic", "standard", "premium"}


# --------------------------------------------------------------------------
# Data access (real pandas/DuckDB computation - no fabricated numbers below)
# --------------------------------------------------------------------------
def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from scripts.build_data import ensure_movielens_catalog

    ensure_movielens_catalog()
    subscribers = pd.read_csv(DATA_DIR / "subscribers.csv")
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    movies = pd.read_csv(DATA_DIR / "movielens" / "movies.csv")
    return subscribers, sessions, movies


def churn_rate_by_segment(segment: str = "plan_tier") -> dict:
    """Real churn rate per group, computed with DuckDB SQL over subscribers.csv."""
    if segment not in SEGMENT_COLUMNS:
        raise ValueError(f"segment must be one of {sorted(SEGMENT_COLUMNS)}, got {segment!r}")
    subscribers, _, _ = _load_frames()
    query = f"""
        SELECT {segment} AS segment_value,
               COUNT(*) AS subscriber_count,
               AVG(churn) AS churn_rate
        FROM subscribers
        GROUP BY {segment}
        ORDER BY churn_rate DESC
    """
    result = duckdb.query(query).to_df()
    overall = float(subscribers["churn"].mean())
    return {
        "segment": segment,
        "overall_churn_rate": overall,
        "by_group": result.to_dict(orient="records"),
    }


def top_churn_drivers(top_n: int = 5) -> dict:
    """Top churn-driving features from the trained model's precomputed SHAP
    attribution (models/metrics.json, written by src/model.py). Not
    recomputed here - SHAP is a training-time-only dependency by design (see
    src/model.py docstring)."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "models/metrics.json not found - run `python -m src.model` first."
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    drivers = metrics.get("top_churn_drivers", [])[:top_n]
    return {
        "drivers": drivers,
        "attribution_method": metrics.get("attribution_method", "unknown"),
        "model_roc_auc": metrics.get("roc_auc"),
    }


def completion_rate_by_genre(top_n: int = 10) -> dict:
    """Avg pct_completed per MovieLens genre, computed with DuckDB by
    exploding the pipe-delimited genres column and joining to sessions."""
    _, sessions, movies = _load_frames()
    genres_exploded = movies.assign(genre=movies["genres"].str.split("|")).explode("genre")
    genres_exploded = genres_exploded[genres_exploded["genre"] != "(no genres listed)"]
    query = """
        SELECT genre,
               COUNT(*) AS session_count,
               AVG(pct_completed) AS avg_completion
        FROM sessions s
        JOIN genres_exploded g ON s.movie_id = g.movieId
        GROUP BY genre
        HAVING COUNT(*) >= 20
        ORDER BY avg_completion DESC
        LIMIT ?
    """
    result = duckdb.query(query, params=[top_n]).to_df()
    return {"by_genre": result.to_dict(orient="records")}


def engagement_trend_over_time(months: int = 6) -> dict:
    """Monthly session volume and avg completion rate, computed with DuckDB
    by truncating watch_date to month. Note: this dataset has NO per-period
    churn events (churn is a single static label per subscriber - see
    README "Honest scope"), so this trend is over engagement, not churn."""
    _, sessions, _ = _load_frames()
    query = """
        SELECT strftime(CAST(watch_date AS DATE), '%Y-%m') AS month,
               COUNT(*) AS session_count,
               AVG(pct_completed) AS avg_completion
        FROM sessions
        GROUP BY month
        ORDER BY month DESC
        LIMIT ?
    """
    result = duckdb.query(query, params=[months]).to_df().sort_values("month")
    return {"by_month": result.to_dict(orient="records")}


# --------------------------------------------------------------------------
# Rule-based router: question -> (intent, analysis args)
# --------------------------------------------------------------------------
def route(question: str) -> tuple[str | None, dict]:
    """Deterministic keyword router. Returns (intent, args); intent is None
    if the question doesn't map to a supported analysis (grounding
    guardrail - see module docstring)."""
    q = question.lower()

    wants_driver = any(w in q for w in ["driver", "why do", "why does", "why did",
                                          "cause", "factor", "predict"])
    wants_genre = "genre" in q
    wants_trend = any(w in q for w in ["trend", "over time", "last quarter",
                                        "last month", "spike", "month over month"])
    wants_churn = "churn" in q

    segment = None
    for tier in KNOWN_TIERS:
        if tier in q:
            segment = "plan_tier"
            break
    if segment is None:
        for region in KNOWN_REGIONS:
            if region in q:
                segment = "region"
                break
    if segment is None and "region" in q:
        segment = "region"
    if segment is None and ("plan" in q or "tier" in q):
        segment = "plan_tier"

    # Order matters: a question can mention "churn", "why"/"spike" (driver
    # language), and a specific segment all at once (e.g. "why did churn
    # spike for Premium subscribers last quarter"). A named segment (a plan
    # tier or region) signals the user wants a segment cut, so that takes
    # priority over generic driver language; this dataset has no
    # time-indexed churn events, so a segment cut plus an explicit trend
    # caveat is the honest, closest-supported answer rather than a
    # fabricated spike explanation. Only when no segment is named do we fall
    # back to the model's general top drivers.
    if wants_genre:
        return "completion_by_genre", {}
    if wants_churn and segment is not None:
        return "churn_by_segment", {"segment": segment, "trend_requested": wants_trend}
    if wants_driver and wants_churn:
        return "top_drivers", {}
    if wants_churn:
        return "churn_by_segment", {"segment": "plan_tier", "trend_requested": wants_trend}
    if wants_trend:
        return "engagement_trend", {}
    return None, {}


SUPPORTED_ANALYSES_DESCRIPTION = (
    "churn rate by plan tier or region, top churn drivers from the trained "
    "model, completion rate by movie genre, and engagement (session volume / "
    "completion) trend over time"
)


# --------------------------------------------------------------------------
# Deterministic, grounded composer (default) + optional Claude phrasing
# --------------------------------------------------------------------------
def _compose_template(intent: str, args: dict, result: dict) -> str:
    if intent == "churn_by_segment":
        rows = result["by_group"]
        overall = result["overall_churn_rate"]
        lines = [f"Overall churn rate across all subscribers is {overall:.1%}."]
        lines.append(
            "By " + args["segment"].replace("_", " ") + ": "
            + "; ".join(f"{r['segment_value']} {r['churn_rate']:.1%} (n={r['subscriber_count']})"
                        for r in rows) + "."
        )
        if args.get("trend_requested"):
            lines.append(
                "Note: this dataset stores churn as a single static label per "
                "subscriber, not a time series, so a quarter-over-quarter "
                "'spike' can't be measured here - the figures above are the "
                "current snapshot churn rate by segment, the closest "
                "supported analysis."
            )
        return " ".join(lines)

    if intent == "top_drivers":
        drivers = result["drivers"]
        method = result["attribution_method"]
        auc = result["model_roc_auc"]
        ranked = ", ".join(f"{d['feature']} ({d['mean_abs_shap']:.3f})" for d in drivers)
        return (
            f"Top churn drivers from the trained model (ROC-AUC {auc:.3f}, "
            f"attribution: {method}): {ranked}. Higher values mean the "
            f"feature contributes more to pushing predictions toward churn."
        )

    if intent == "completion_by_genre":
        rows = result["by_genre"]
        top = rows[0] if rows else None
        if not top:
            return "No genre had enough sessions (20+) to report a reliable completion rate."
        lines = [
            f"{top['genre']} has the highest average completion rate at "
            f"{top['avg_completion']:.1%} across {top['session_count']} sessions."
        ]
        if len(rows) > 1:
            others = "; ".join(
                f"{r['genre']} {r['avg_completion']:.1%}" for r in rows[1:5]
            )
            lines.append(f"Next highest: {others}.")
        return " ".join(lines)

    if intent == "engagement_trend":
        rows = result["by_month"]
        if not rows:
            return "No session data available to compute a trend."
        lines = [
            f"{r['month']}: {r['session_count']} sessions, "
            f"{r['avg_completion']:.1%} avg completion" for r in rows
        ]
        return ("Monthly engagement trend (most recent months): " + "; ".join(lines) + ". "
                "Note: this is a viewing-engagement trend, not a churn trend - "
                "churn is a static per-subscriber label in this dataset.")

    return "Here is what I found."


def _maybe_llm_compose():
    """Optional Claude phrasing of the same grounded numbers - mirrors the
    key-free/live-LLM pattern in the sibling clinical-trials project's
    backend/app/planner.py. Returns None if no key or package available."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from langchain_anthropic import ChatAnthropic
    except Exception:
        return None

    model = ChatAnthropic(model="claude-sonnet-5", max_tokens=400)

    def _compose(intent: str, args: dict, result: dict, question: str) -> str:
        prompt = (
            "You are a business-analytics assistant for a streaming subscription "
            "product. Using ONLY the computed numbers in the JSON below, answer "
            "the user's question in under 100 words. Do NOT invent any number "
            "that is not present in the JSON. If the JSON contains a note about "
            "a data limitation, include it.\n\n"
            f"Question: {question}\nIntent: {intent}\nComputed data: {json.dumps(result)}"
        )
        return model.invoke(prompt).content

    return _compose


def compose_answer(intent: str, args: dict, result: dict, question: str) -> str:
    llm_compose = _maybe_llm_compose()
    if llm_compose is not None:
        try:
            return llm_compose(intent, args, result, question)
        except Exception:
            pass
    return _compose_template(intent, args, result)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def answer_business_question(question: str) -> dict:
    """Route -> compute -> compose. Single entry point for the Streamlit app
    and the MCP server's `answer_business_question` tool."""
    intent, args = route(question)
    if intent is None:
        return {
            "question": question,
            "supported": False,
            "intent": None,
            "grounded_data": None,
            "answer": (
                "I can't answer that with the analyses this tool supports. "
                f"Supported analyses: {SUPPORTED_ANALYSES_DESCRIPTION}. "
                "Try rephrasing around one of those."
            ),
        }

    if intent == "churn_by_segment":
        result = churn_rate_by_segment(args["segment"])
    elif intent == "top_drivers":
        result = top_churn_drivers()
    elif intent == "completion_by_genre":
        result = completion_rate_by_genre()
    elif intent == "engagement_trend":
        result = engagement_trend_over_time()
    else:  # pragma: no cover - defensive, route() only returns known intents
        raise ValueError(f"Unhandled intent: {intent}")

    answer = compose_answer(intent, args, result, question)
    return {
        "question": question,
        "supported": True,
        "intent": intent,
        "grounded_data": result,
        "answer": answer,
    }
