"""Feature engineering: raw sessions/subscribers -> a per-subscriber feature table.

This module is the single source of truth for what an "engagement feature"
means in this project. `scripts/build_data.py` imports `compute_feature_table`
to derive the synthetic churn label from real engagement signals (see its
module docstring), and `src/model.py` imports the exact same function to
build the model's training matrix. There is one feature definition, used in
both places, so the label-generation logic and the model-training logic can
never silently drift apart.

Feature definitions (see PLAYBOOK.md for the business-facing explanation):
- sessions_per_week: viewing sessions / (tenure in weeks). A frequency signal.
- avg_pct_completed: mean fraction of a title watched per session (0-1).
  Subscribers with zero sessions get 0.0 (maximally disengaged, not missing).
- recency_days: days between REFERENCE_DATE and the subscriber's most recent
  session. Subscribers with zero sessions get recency_days = tenure_days
  (they have been "not watching" for their entire tenure).
- genre_diversity: count of distinct MovieLens genres appearing among a
  subscriber's watched titles. A proxy for how broadly they use the catalog.
- tenure_days: days between signup_date and REFERENCE_DATE.
"""

from __future__ import annotations

import pandas as pd

# Fixed "as-of" date for this snapshot dataset. Every recency/tenure
# calculation is relative to this date, not to wall-clock "today", so the
# generated data and derived features are fully reproducible.
REFERENCE_DATE = pd.Timestamp("2025-06-30")

FEATURE_COLUMNS = [
    "tenure_days",
    "sessions_per_week",
    "avg_pct_completed",
    "recency_days",
    "genre_diversity",
]


def _genre_lookup(movies_df: pd.DataFrame) -> dict[int, list[str]]:
    """movieId -> list of genre strings (MovieLens pipe-delimited genres)."""
    out: dict[int, list[str]] = {}
    for row in movies_df.itertuples(index=False):
        genres = [] if row.genres == "(no genres listed)" else row.genres.split("|")
        out[row.movieId] = genres
    return out


def compute_feature_table(
    sessions_df: pd.DataFrame,
    subscribers_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    reference_date: pd.Timestamp = REFERENCE_DATE,
) -> pd.DataFrame:
    """Build one row per subscriber_id with the engagement features above.

    Pure function of the three input tables - no hidden state, no randomness -
    so it is deterministic and safe to call from both the data-generation
    step and the training step.
    """
    sessions_df = sessions_df.copy()
    sessions_df["watch_date"] = pd.to_datetime(sessions_df["watch_date"])
    genre_map = _genre_lookup(movies_df)

    agg = (
        sessions_df.groupby("subscriber_id")
        .agg(
            n_sessions=("movie_id", "count"),
            avg_pct_completed=("pct_completed", "mean"),
            last_session=("watch_date", "max"),
        )
        .reset_index()
    )

    genre_counts = (
        sessions_df.assign(
            genres=sessions_df["movie_id"].map(lambda m: genre_map.get(m, []))
        )
        .explode("genres")
        .dropna(subset=["genres"])
        .groupby("subscriber_id")["genres"]
        .nunique()
        .rename("genre_diversity")
        .reset_index()
    )

    subs = subscribers_df[["subscriber_id", "signup_date", "plan_tier", "region"]].copy()
    subs["signup_date"] = pd.to_datetime(subs["signup_date"])
    subs["tenure_days"] = (reference_date - subs["signup_date"]).dt.days

    feat = subs.merge(agg, on="subscriber_id", how="left").merge(
        genre_counts, on="subscriber_id", how="left"
    )

    feat["n_sessions"] = feat["n_sessions"].fillna(0).astype(int)
    feat["avg_pct_completed"] = feat["avg_pct_completed"].fillna(0.0)
    feat["genre_diversity"] = feat["genre_diversity"].fillna(0).astype(int)

    tenure_weeks = (feat["tenure_days"] / 7.0).clip(lower=1.0)
    feat["sessions_per_week"] = feat["n_sessions"] / tenure_weeks

    # Recency: days since last session. Never-watched subscribers are treated
    # as maximally stale (recency = their whole tenure), not as missing data.
    has_session = feat["last_session"].notna()
    feat["recency_days"] = 0
    feat.loc[has_session, "recency_days"] = (
        reference_date - feat.loc[has_session, "last_session"]
    ).dt.days
    feat.loc[~has_session, "recency_days"] = feat.loc[~has_session, "tenure_days"]

    return feat[
        ["subscriber_id", "plan_tier", "region", "signup_date", "n_sessions"]
        + FEATURE_COLUMNS
    ].reset_index(drop=True)
