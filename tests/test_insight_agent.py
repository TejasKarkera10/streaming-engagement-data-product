"""Grounding tests for src/insight_agent.py - the most important tests in
this project. For each example business question, we assert the numbers
returned by the agent match numbers computed directly via pandas on the
same underlying data, which is what proves "no hallucination": every
number in an answer traces back to a real computation, never an LLM guess.
(No ANTHROPIC_API_KEY is set in this test environment, so these exercise
the deterministic template composer - the zero-API-key path.)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.insight_agent import (
    answer_business_question,
    churn_rate_by_segment,
    completion_rate_by_genre,
    route,
    top_churn_drivers,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def test_churn_rate_by_segment_matches_direct_pandas():
    subscribers = pd.read_csv(DATA_DIR / "subscribers.csv")
    expected = subscribers.groupby("plan_tier")["churn"].mean().to_dict()

    result = churn_rate_by_segment("plan_tier")
    got = {row["segment_value"]: row["churn_rate"] for row in result["by_group"]}

    assert got.keys() == expected.keys()
    for tier in expected:
        assert got[tier] == pytest.approx(expected[tier])
    assert result["overall_churn_rate"] == pytest.approx(subscribers["churn"].mean())


def test_completion_rate_by_genre_matches_direct_pandas():
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    movies = pd.read_csv(DATA_DIR / "movielens" / "movies.csv")
    exploded = movies.assign(genre=movies["genres"].str.split("|")).explode("genre")
    exploded = exploded[exploded["genre"] != "(no genres listed)"]
    merged = sessions.merge(exploded[["movieId", "genre"]], left_on="movie_id", right_on="movieId")
    expected = (
        merged.groupby("genre")["pct_completed"]
        .agg(["mean", "count"])
        .query("count >= 20")
        .sort_values("mean", ascending=False)
    )

    result = completion_rate_by_genre(top_n=10)
    top_row = result["by_genre"][0]
    assert top_row["genre"] == expected.index[0]
    assert top_row["avg_completion"] == pytest.approx(expected.iloc[0]["mean"])
    assert top_row["session_count"] == int(expected.iloc[0]["count"])


def test_top_churn_drivers_matches_saved_metrics():
    import json

    with open(ROOT / "models" / "metrics.json") as f:
        metrics = json.load(f)
    result = top_churn_drivers(top_n=3)
    assert result["drivers"] == metrics["top_churn_drivers"][:3]
    assert result["model_roc_auc"] == metrics["roc_auc"]


def test_answer_business_question_numbers_match_grounded_data():
    """The answer text's headline number must equal the grounded_data number
    - not merely 'close', but the same value the composer was given."""
    result = answer_business_question("What is the churn rate by plan tier?")
    assert result["supported"] is True
    overall_pct = f"{result['grounded_data']['overall_churn_rate']:.1%}"
    assert overall_pct in result["answer"]


def test_answer_business_question_genre_matches_grounded_data():
    result = answer_business_question("Which genre has the highest completion rate?")
    assert result["supported"] is True
    top = result["grounded_data"]["by_genre"][0]
    assert top["genre"] in result["answer"]
    assert f"{top['avg_completion']:.1%}" in result["answer"]


def test_unsupported_question_triggers_grounding_guardrail():
    result = answer_business_question("What's the weather forecast for tomorrow?")
    assert result["supported"] is False
    assert result["grounded_data"] is None
    assert route("What's the weather forecast for tomorrow?") == (None, {})


def test_router_maps_known_question_types():
    assert route("Which genre has the highest completion rate?")[0] == "completion_by_genre"
    assert route("What are the top churn drivers?")[0] == "top_drivers"
    assert route("What is the churn rate by region?")[0] == "churn_by_segment"
