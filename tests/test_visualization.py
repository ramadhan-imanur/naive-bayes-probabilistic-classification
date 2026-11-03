"""
Unit test untuk modul visualisasi (src.visualization).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization import (
    plot_confusion_matrix,
    plot_cross_validation_stability,
    plot_feature_correlations,
    plot_likelihood_distributions,
    plot_roc_and_pr_curves,
    plot_threshold_tuning,
)


@pytest.fixture
def dummy_vis_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Glucose": np.random.normal(120, 25, n),
        "BMI": np.random.normal(30, 6, n),
        "Age": np.random.randint(20, 80, n),
        "Outcome": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })
    y_true = df["Outcome"].values
    y_proba = np.clip(np.random.normal(0.4, 0.3, n), 0.0, 1.0)
    return df, y_true, y_proba


def test_plot_confusion_matrix(tmp_path: Path):
    cm = {"TN": 84, "FP": 16, "FN": 22, "TP": 32}
    save_file = tmp_path / "cm.png"
    fig = plot_confusion_matrix(cm, save_path=save_file)

    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_roc_and_pr_curves(dummy_vis_data, tmp_path: Path):
    _, y_true, y_proba = dummy_vis_data
    save_file = tmp_path / "roc.png"
    fig = plot_roc_and_pr_curves(y_true, y_proba, save_path=save_file)

    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_likelihood_distributions(dummy_vis_data, tmp_path: Path):
    df, _, _ = dummy_vis_data
    save_file = tmp_path / "dist.png"
    fig = plot_likelihood_distributions(df, ["Glucose", "BMI"], save_path=save_file)

    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_feature_correlations(dummy_vis_data, tmp_path: Path):
    df, _, _ = dummy_vis_data
    save_file = tmp_path / "corr.png"
    fig = plot_feature_correlations(df, ["Glucose", "BMI", "Age"], save_path=save_file)

    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_threshold_tuning(dummy_vis_data, tmp_path: Path):
    _, y_true, y_proba = dummy_vis_data
    save_file = tmp_path / "thresh.png"
    fig, df_thresh = plot_threshold_tuning(y_true, y_proba, save_path=save_file)

    assert isinstance(fig, plt.Figure)
    assert isinstance(df_thresh, pd.DataFrame)
    assert "precision" in df_thresh.columns
    assert "recall" in df_thresh.columns
    assert save_file.exists()
    plt.close(fig)


def test_plot_cross_validation_stability(tmp_path: Path):
    cv_summary = {
        "accuracy": {"mean": 0.7448, "std": 0.0312},
        "precision": {"mean": 0.6475, "std": 0.0512},
        "recall": {"mean": 0.5937, "std": 0.0644},
        "f1": {"mean": 0.6180, "std": 0.0511},
    }
    save_file = tmp_path / "cv.png"
    fig = plot_cross_validation_stability(cv_summary, save_path=save_file)

    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)
