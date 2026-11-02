"""
Unit test untuk modul evaluasi (src.evaluation).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB

from src.evaluation import (
    analyze_errors,
    evaluate_predictions,
    run_stratified_cv,
    save_experiment_results,
)


def test_evaluate_predictions_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    res = evaluate_predictions(y_true, y_pred)

    cm = res["confusion_matrix"]
    assert cm["TN"] == 1
    assert cm["FP"] == 1
    assert cm["FN"] == 1
    assert cm["TP"] == 1

    assert np.isclose(res["accuracy"], 0.5)
    assert np.isclose(res["precision"], 0.5)
    assert np.isclose(res["recall"], 0.5)
    assert np.isclose(res["f1_score"], 0.5)


def test_run_stratified_cv():
    np.random.seed(42)
    X = pd.DataFrame({"x1": np.random.randn(60), "x2": np.random.randn(60)})
    y = pd.Series([0] * 30 + [1] * 30)

    clf = GaussianNB()
    cv_summary = run_stratified_cv(clf, X, y, n_splits=3, random_state=42)

    assert "accuracy" in cv_summary
    assert "precision" in cv_summary
    assert "recall" in cv_summary
    assert "f1" in cv_summary

    assert 0.0 <= cv_summary["accuracy"]["mean"] <= 1.0
    assert cv_summary["accuracy"]["std"] >= 0.0


def test_analyze_errors():
    X_test = pd.DataFrame({"val": [10, 20, 30, 40]})
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    err_analysis = analyze_errors(X_test, y_true, y_pred)

    assert err_analysis["total_test_samples"] == 4
    assert err_analysis["total_errors"] == 2
    assert np.isclose(err_analysis["error_rate"], 0.5)
    assert err_analysis["false_positives"]["count"] == 1
    assert err_analysis["false_negatives"]["count"] == 1


def test_save_experiment_results(tmp_path: Path):
    data = {"status": "success", "metric": 0.85}
    dest_path = tmp_path / "test_metrics" / "out.json"

    saved = save_experiment_results(data, dest_path)
    assert saved.exists()

    with open(saved, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metric"] == 0.85
