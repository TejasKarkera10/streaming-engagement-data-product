"""Data-pipeline schema/row-count tests. These assume `python
scripts/build_data.py` has already been run (see Makefile `data` target)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def test_subscribers_schema_and_row_count():
    df = pd.read_csv(DATA_DIR / "subscribers.csv")
    assert set(df.columns) == {"subscriber_id", "signup_date", "plan_tier", "region", "churn"}
    assert len(df) == 3000
    assert df["subscriber_id"].is_unique
    assert set(df["plan_tier"].unique()) <= {"Basic", "Standard", "Premium"}
    assert set(df["churn"].unique()) <= {0, 1}


def test_sessions_schema_references_real_catalog():
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    movies = pd.read_csv(DATA_DIR / "movielens" / "movies.csv")
    assert set(sessions.columns) == {
        "subscriber_id", "movie_id", "watch_date", "pct_completed", "rating",
    }
    assert len(sessions) > 1000
    assert sessions["pct_completed"].between(0, 1).all()
    assert sessions["rating"].between(0.5, 5.0).all()
    # Every session must reference a movie that actually exists in the catalog.
    assert set(sessions["movie_id"]).issubset(set(movies["movieId"]))


def test_movielens_catalog_looks_real():
    """Sanity check that the real MovieLens download succeeded rather than
    silently falling back to the placeholder catalog (see build_data.py)."""
    movies = pd.read_csv(DATA_DIR / "movielens" / "movies.csv")
    assert {"movieId", "title", "genres"} <= set(movies.columns)
    assert len(movies) > 1000
    titles = " ".join(movies["title"].astype(str))
    assert "[FALLBACK-PLACEHOLDER]" not in titles
    # A well-known real title should be present in the real ml-latest-small set.
    assert movies["title"].str.contains("Toy Story", case=False).any()
