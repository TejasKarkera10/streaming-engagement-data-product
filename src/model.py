"""Train a churn classifier on the engagement feature table and save it,
along with an evaluation report and a ranked list of top churn drivers.

Run with:  python -m src.model     (idempotent - re-running retrains cleanly)

Model choice
------------
LightGBM's `LGBMClassifier` (gradient-boosted trees). Both scikit-learn's
`GradientBoostingClassifier` and `lightgbm` installed cleanly in this
environment; LightGBM was chosen because it trains noticeably faster on the
~3k-row / 5-feature table used here (leaf-wise growth) and its native
`predict_proba` plus tree structure works directly with SHAP's
`TreeExplainer` (see below) with no extra wrapping.

Top-driver attribution: SHAP vs. feature_importances_
-------------------------------------------------------
We use SHAP's `TreeExplainer` on the trained model, not the raw
`feature_importances_` split-gain values, because SHAP gives per-feature,
signed, comparable-magnitude attributions (mean |SHAP value|) that map
directly onto "how much did this feature push predictions toward churn,"
which is what a business stakeholder actually wants from a "driver" chart.
`feature_importances_` (split gain) was considered as the documented
lighter-weight fallback (per this project's own design goal of only using
what's actually installed and fast) but SHAP installed cleanly here and
`TreeExplainer` is fast on tree ensembles (no KernelExplainer slowness), so
we use it. SHAP is therefore only a training-time dependency: this module
computes it ONCE and writes the result into models/metrics.json; the
Streamlit app and insight agent read that precomputed json and never import
shap themselves, so the (slightly heavy) shap import cost is paid only when
running `make train`, not on every app load.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLUMNS, compute_feature_table  # noqa: E402

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CATEGORICAL_COLUMNS = ["plan_tier", "region"]
RANDOM_SEED = 42


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subscribers = pd.read_csv(DATA_DIR / "subscribers.csv")
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    movies = pd.read_csv(DATA_DIR / "movielens" / "movies.csv")
    return subscribers, sessions, movies


def build_training_table(
    subscribers: pd.DataFrame, sessions: pd.DataFrame, movies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Feature table (one-hot encoded) + churn label, aligned by subscriber_id."""
    features = compute_feature_table(sessions, subscribers, movies)
    features = features.merge(
        subscribers[["subscriber_id", "churn"]], on="subscriber_id", how="left"
    )
    encoded = pd.get_dummies(features, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS)
    dummy_cols = [
        c for c in encoded.columns
        if any(c.startswith(p + "_") for p in CATEGORICAL_COLUMNS)
    ]
    x_cols = FEATURE_COLUMNS + dummy_cols
    X = encoded[x_cols].astype(float)
    y = encoded["churn"].astype(int)
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate_test": float(y_test.mean()),
    }
    return model, metrics, (X_test, y_test)


def top_drivers(model, X: pd.DataFrame) -> list[dict]:
    """Mean |SHAP value| per feature, using shap.TreeExplainer. See module
    docstring for why SHAP over feature_importances_, and why this only runs
    at training time."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # some shap/lightgbm combos return per-class list
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(X.columns, mean_abs), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "mean_abs_shap": float(v)} for f, v in ranked]


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    subscribers, sessions, movies = load_data()
    X, y = build_training_table(subscribers, sessions, movies)
    model, metrics, (X_test, y_test) = train_and_evaluate(X, y)

    drivers = top_drivers(model, X_test)
    metrics["top_churn_drivers"] = drivers
    metrics["feature_columns"] = list(X.columns)
    metrics["model"] = "LGBMClassifier"
    metrics["attribution_method"] = "shap.TreeExplainer (mean |SHAP value| on the held-out test set)"

    with open(MODELS_DIR / "churn_model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_columns": list(X.columns)}, f)

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ROC-AUC: {metrics['roc_auc']:.3f}  Precision: {metrics['precision']:.3f}  "
          f"Recall: {metrics['recall']:.3f}")
    print("Top churn drivers (mean |SHAP|):")
    for d in drivers[:5]:
        print(f"  {d['feature']:<25} {d['mean_abs_shap']:.4f}")
    print(f"Saved model -> {MODELS_DIR / 'churn_model.pkl'}")
    print(f"Saved metrics -> {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
