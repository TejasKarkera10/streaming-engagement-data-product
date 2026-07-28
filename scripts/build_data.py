"""Build the full dataset for this project: one real catalog + one synthetic
subscriber/engagement layer on top of it.

Run with:  python scripts/build_data.py

What this script does, in order
--------------------------------
1. Downloads the REAL, public, no-auth MovieLens "ml-latest-small" dataset
   (https://files.grouplens.org/datasets/movielens/ml-latest-small.zip,
   GroupLens Research, ~1MB, real movie titles/genres and real user ratings)
   directly over HTTPS and unzips it into data/movielens/. No API key needed.
   If the download fails for any reason (offline sandbox, DNS, etc.) this
   script falls back to a small locally-generated placeholder catalog with
   the *same schema* (movieId, title, genres) - that fallback path is
   clearly logged and flagged, see `_fallback_movie_catalog()` below. Prefer
   re-running this script with network access to get the real data.
2. Generates a SYNTHETIC subscriber base (data/subscribers.csv) and a
   SYNTHETIC viewing-session log (data/sessions.csv) that references real
   movieIds from step 1. This is explicitly fake data standing in for a
   production subscriber database - see the "Synthetic churn-label logic"
   section below and the project README's "Honest scope" section.
3. Derives a per-subscriber churn label from the generated sessions using a
   documented, transparent rule (engagement features -> logistic score ->
   Bernoulli draw), via the SAME feature-engineering code
   (`src.features.compute_feature_table`) the model is trained on later, so
   there is one definition of "engagement" in the whole codebase.

Everything is seeded (RANDOM_SEED) so re-running this script reproduces byte-
identical subscribers.csv / sessions.csv given the same MovieLens download.

Synthetic churn-label logic (why these features drive churn)
--------------------------------------------------------------
Real subscription-churn research consistently finds three behavioral signals
predict cancellation: falling usage frequency, low content completion (people
who bail on what they start are not getting value), and long recency gaps
(inactive users churn). We simulate that mechanism explicitly and honestly:

  1. Each synthetic subscriber gets a latent "engagement_propensity" in (0,1)
     drawn from Beta(2,2). This single latent value independently drives:
       - how many viewing sessions they generate (higher propensity -> more
         sessions per week, via a Poisson process),
       - how much of each title they complete (higher propensity -> higher
         mean pct_completed, with per-session noise),
       - how recent their last session is (higher propensity -> sessions
         skew toward the recent end of their tenure window).
     This is what makes engaged subscribers *look* engaged across every
     signal at once, exactly as in real usage data.
  2. We then compute the same engagement features the model will later train
     on (sessions_per_week, avg_pct_completed, recency_days, genre_diversity,
     tenure_days) via `src.features.compute_feature_table` - NOT the hidden
     propensity itself - and turn them into a churn probability with a
     documented logistic rule (see `_assign_churn` below): low frequency,
     low completion, and long recency each independently push churn risk up;
     Premium subscribers get a small negative (protective) adjustment
     reflecting higher commitment/sunk cost; genre diversity gets a small
     protective adjustment (broader catalog use ~ more engaged use).
  3. A Gaussian noise term is added to the logit before the Bernoulli draw
     specifically so the label is NOT a deterministic function of the
     features - otherwise a model would trivially achieve ~perfect
     separation, which would misrepresent what a churn model can do in
     practice. The noise std was chosen (see NOISE_STD) so the trained
     model lands in a realistic, non-trivial ROC-AUC range.

This is a fully disclosed, seeded, rule-based-plus-noise simulation. It is
NOT real subscriber behavior - see README.md "Honest scope".
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MOVIELENS_DIR = DATA_DIR / "movielens"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

RANDOM_SEED = 42
N_SUBSCRIBERS = 3000
NOISE_STD = 1.35  # logit-scale noise; tuned so trained ROC-AUC lands ~0.75-0.9

sys.path.insert(0, str(ROOT))
from src.features import compute_feature_table  # noqa: E402


# --------------------------------------------------------------------------
# Step 1: real MovieLens catalog (with a clearly-flagged fallback)
# --------------------------------------------------------------------------
def download_movielens() -> bool:
    """Download + unzip the real ml-latest-small dataset. Returns True on
    success. Never raises - a failure here triggers the fallback catalog."""
    try:
        print(f"Downloading real MovieLens dataset from {MOVIELENS_URL} ...")
        with urllib.request.urlopen(MOVIELENS_URL, timeout=30) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(DATA_DIR)
        extracted = DATA_DIR / "ml-latest-small"
        if extracted.exists():
            if MOVIELENS_DIR.exists():
                import shutil

                shutil.rmtree(MOVIELENS_DIR)
            extracted.rename(MOVIELENS_DIR)
        print(f"OK: real MovieLens data unzipped to {MOVIELENS_DIR}")
        return True
    except Exception as exc:  # noqa: BLE001 - any failure -> fallback path
        print(f"WARNING: MovieLens download failed ({exc!r}); using fallback catalog.")
        return False


def ensure_movielens_catalog() -> None:
    """Idempotent: guarantee data/movielens/movies.csv exists before it's
    read at runtime (by src/insight_agent.py, app.py, src/mcp_server.py).

    data/movielens/ is gitignored (see README "Honest scope" - the raw
    MovieLens files are re-downloadable, not redistributed), so a fresh
    clone (e.g. Streamlit Community Cloud) won't have it even though
    data/subscribers.csv, data/sessions.csv, and models/ ARE committed.
    Without this, every runtime read of movies.csv would crash on first
    deploy. Safe/cheap to call on every process start: no-ops if the file
    is already there.
    """
    if (MOVIELENS_DIR / "movies.csv").exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not download_movielens():
        _fallback_movie_catalog()


def _fallback_movie_catalog() -> None:
    """LAST-RESORT FALLBACK ONLY - used only if the real MovieLens download
    fails (e.g. no network in this sandbox). This is NOT the real MovieLens
    dataset: it is a small, locally-generated placeholder with the same
    schema (movieId, title, genres) so the rest of the pipeline still runs.
    Flagged loudly here and in README.md "Honest scope" if it is ever used.
    """
    MOVIELENS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    genre_pool = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War",
    ]
    rows = []
    for movie_id in range(1, 501):
        n_genres = rng.integers(1, 3)
        genres = "|".join(sorted(rng.choice(genre_pool, size=n_genres, replace=False)))
        rows.append(
            {
                "movieId": movie_id,
                # Clearly labeled as placeholder, not a real title.
                "title": f"[FALLBACK-PLACEHOLDER] Movie {movie_id:04d} ({1970 + movie_id % 50})",
                "genres": genres,
            }
        )
    pd.DataFrame(rows).to_csv(MOVIELENS_DIR / "movies.csv", index=False)
    # Minimal ratings.csv so popularity weighting still has something to read.
    rating_rows = []
    for movie_id in range(1, 501):
        for _ in range(int(rng.integers(1, 20))):
            rating_rows.append(
                {
                    "userId": int(rng.integers(1, 200)),
                    "movieId": movie_id,
                    "rating": float(rng.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])),
                    "timestamp": 0,
                }
            )
    pd.DataFrame(rating_rows).to_csv(MOVIELENS_DIR / "ratings.csv", index=False)
    print(
        "FALLBACK CATALOG WRITTEN: data/movielens/movies.csv is a synthetic "
        "placeholder, NOT the real MovieLens dataset. Re-run with network "
        "access to replace it with real data."
    )


# --------------------------------------------------------------------------
# Step 2: synthetic subscriber base + viewing sessions
# --------------------------------------------------------------------------
REFERENCE_DATE = pd.Timestamp("2025-06-30")
SIGNUP_WINDOW_DAYS = (730, 14)  # (earliest signup, latest signup) days before reference
PLAN_TIERS = ["Basic", "Standard", "Premium"]
PLAN_WEIGHTS = [0.40, 0.35, 0.25]
REGIONS = ["North America", "Europe", "APAC", "Latin America"]
REGION_WEIGHTS = [0.42, 0.28, 0.20, 0.10]


def _make_subscribers(rng: np.random.Generator) -> pd.DataFrame:
    ids = [f"sub_{i:05d}" for i in range(1, N_SUBSCRIBERS + 1)]
    offsets = rng.integers(SIGNUP_WINDOW_DAYS[1], SIGNUP_WINDOW_DAYS[0], size=N_SUBSCRIBERS)
    signup_dates = [REFERENCE_DATE - pd.Timedelta(days=int(o)) for o in offsets]
    plan_tier = rng.choice(PLAN_TIERS, size=N_SUBSCRIBERS, p=PLAN_WEIGHTS)
    region = rng.choice(REGIONS, size=N_SUBSCRIBERS, p=REGION_WEIGHTS)
    # Latent engagement propensity - NOT written to disk, only used to drive
    # session generation below. The churn label is derived later from the
    # *observable* engagement features, not from this latent value directly.
    propensity = rng.beta(2, 2, size=N_SUBSCRIBERS)
    return pd.DataFrame(
        {
            "subscriber_id": ids,
            "signup_date": [d.date().isoformat() for d in signup_dates],
            "plan_tier": plan_tier,
            "region": region,
            "propensity": propensity,  # latent only; dropped before writing subscribers.csv
        }
    )


def _make_sessions(subscribers: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame,
                    rng: np.random.Generator) -> pd.DataFrame:
    movie_ids = movies["movieId"].to_numpy()
    # Weight movie selection by real MovieLens rating popularity so commonly
    # rated titles are also commonly "watched" in the synthetic session log.
    popularity = ratings["movieId"].value_counts()
    weights = np.array([popularity.get(m, 1) for m in movie_ids], dtype=float)
    weights = weights / weights.sum()

    session_rows = []
    for row in subscribers.itertuples(index=False):
        tenure_days = (REFERENCE_DATE - pd.Timestamp(row.signup_date)).days
        propensity = row.propensity
        weeks_tenure = tenure_days / 7.0
        base_rate_per_week = 0.3 + propensity * 2.2
        n_sessions = int(rng.poisson(base_rate_per_week * weeks_tenure))
        n_sessions = min(n_sessions, 400)
        if n_sessions == 0:
            continue

        chosen_movies = rng.choice(movie_ids, size=n_sessions, p=weights, replace=True)

        # Recency skew: higher propensity -> watch dates skew toward "now".
        alpha = 1 + propensity * 3
        beta_param = 1 + (1 - propensity) * 3
        frac = rng.beta(alpha, beta_param, size=n_sessions)
        watch_dates = [
            pd.Timestamp(row.signup_date) + pd.Timedelta(days=float(f) * tenure_days)
            for f in frac
        ]

        completion_mean = np.clip(0.35 + propensity * 0.55, 0.05, 0.98)
        pct_completed = np.clip(
            rng.normal(completion_mean, 0.14, size=n_sessions), 0.02, 1.0
        )
        # Subscriber's own rating of the title: loosely tracks how much they
        # completed, plus noise - people who finish something tend to rate it
        # higher, but not deterministically.
        raw_rating = pct_completed * 4.0 + 1.0 + rng.normal(0, 0.6, size=n_sessions)
        star_rating = np.clip(np.round(raw_rating * 2) / 2, 0.5, 5.0)

        for movie_id, wd, pct, rating in zip(chosen_movies, watch_dates, pct_completed, star_rating):
            session_rows.append(
                {
                    "subscriber_id": row.subscriber_id,
                    "movie_id": int(movie_id),
                    "watch_date": wd.date().isoformat(),
                    "pct_completed": round(float(pct), 3),
                    "rating": float(rating),
                }
            )

    return pd.DataFrame(session_rows)


def _assign_churn(features: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Documented rule-based-plus-noise churn label. See module docstring."""

    def z(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0.0

    z_sessions = z(features["sessions_per_week"])
    z_completion = z(features["avg_pct_completed"])
    z_recency = z(features["recency_days"])
    z_diversity = z(features["genre_diversity"])

    plan_adjustment = features["plan_tier"].map(
        {"Basic": 0.15, "Standard": 0.0, "Premium": -0.15}
    ).fillna(0.0)

    logit = (
        -0.35
        - 1.10 * z_sessions
        - 0.90 * z_completion
        + 1.00 * z_recency
        - 0.30 * z_diversity
        + plan_adjustment
        + rng.normal(0, NOISE_STD, size=len(features))
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    return (rng.random(len(features)) < churn_prob).astype(int)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not download_movielens():
        _fallback_movie_catalog()

    movies = pd.read_csv(MOVIELENS_DIR / "movies.csv")
    ratings = pd.read_csv(MOVIELENS_DIR / "ratings.csv")

    rng = np.random.default_rng(RANDOM_SEED)
    subscribers = _make_subscribers(rng)
    sessions = _make_sessions(subscribers, movies, ratings, rng)

    features = compute_feature_table(
        sessions, subscribers.drop(columns=["propensity"]), movies
    )
    churn = _assign_churn(features, rng)
    churn_by_id = dict(zip(features["subscriber_id"], churn))

    subscribers_out = subscribers.drop(columns=["propensity"]).copy()
    subscribers_out["churn"] = subscribers_out["subscriber_id"].map(churn_by_id).fillna(0).astype(int)

    subscribers_out.to_csv(DATA_DIR / "subscribers.csv", index=False)
    sessions.to_csv(DATA_DIR / "sessions.csv", index=False)

    print(f"Wrote {len(subscribers_out)} subscribers -> data/subscribers.csv")
    print(f"Wrote {len(sessions)} sessions -> data/sessions.csv")
    print(f"Overall synthetic churn rate: {subscribers_out['churn'].mean():.3f}")
    print(f"Subscribers with zero sessions: {(subscribers_out['subscriber_id'].map(lambda s: s not in set(sessions['subscriber_id']))).sum()}")


if __name__ == "__main__":
    main()
