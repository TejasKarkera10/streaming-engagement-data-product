"""Tests for src/model.py: the model trains and produces a reasonable,
sanity-bounded ROC-AUC (not a strict target - this is a demo with a
partially-synthetic-by-construction label, see README 'Honest scope')."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.model import build_training_table, load_data, train_and_evaluate

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def test_model_trains_with_reasonable_auc():
    subscribers, sessions, movies = load_data()
    X, y = build_training_table(subscribers, sessions, movies)
    model, metrics, _ = train_and_evaluate(X, y)
    # Sanity bound, not a strict target: a real model should clearly beat
    # random (0.5); it should also not be perfectly separable (an
    # ~1.0 AUC would indicate a labeling leak).
    assert 0.6 < metrics["roc_auc"] < 0.99
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_saved_metrics_json_matches_expected_shape():
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    assert "roc_auc" in metrics and "top_churn_drivers" in metrics
    assert len(metrics["top_churn_drivers"]) >= 5
    assert metrics["roc_auc"] > 0.6


def test_training_table_has_no_missing_values():
    subscribers, sessions, movies = load_data()
    X, y = build_training_table(subscribers, sessions, movies)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert isinstance(subscribers, pd.DataFrame)
