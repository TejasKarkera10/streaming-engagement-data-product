"""Tests for the feature-engineering table shared by data generation and
model training (src/features.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import FEATURE_COLUMNS, compute_feature_table

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _load():
    subscribers = pd.read_csv(DATA_DIR / "subscribers.csv")
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    movies = pd.read_csv(DATA_DIR / "movielens" / "movies.csv")
    return subscribers, sessions, movies


def test_feature_table_has_expected_columns_and_row_count():
    subscribers, sessions, movies = _load()
    features = compute_feature_table(sessions, subscribers, movies)
    assert set(FEATURE_COLUMNS) <= set(features.columns)
    assert len(features) == len(subscribers)
    assert features["subscriber_id"].is_unique


def test_never_watched_subscriber_gets_zero_completion_and_recency_equals_tenure():
    subscribers, sessions, movies = _load()
    features = compute_feature_table(sessions, subscribers, movies)
    never_watched = features[features["n_sessions"] == 0]
    if len(never_watched):
        assert (never_watched["avg_pct_completed"] == 0.0).all()
        assert (never_watched["recency_days"] == never_watched["tenure_days"]).all()


def test_sessions_per_week_is_nonnegative_and_finite():
    subscribers, sessions, movies = _load()
    features = compute_feature_table(sessions, subscribers, movies)
    assert (features["sessions_per_week"] >= 0).all()
    assert features["sessions_per_week"].notna().all()
