"""
Probabilistic Machine Learning - Gaussian Naïve Bayes Clinical Modeling
"""

from .data import load_dataset, clean_clinical_data, prepare_splits
from .models import GaussianNBClassifier
from .evaluation import (
    evaluate_predictions,
    run_stratified_cv,
    analyze_errors,
    save_experiment_results,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_likelihood_distributions,
    plot_feature_correlations,
    plot_threshold_tuning,
    plot_cross_validation_stability,
)

__all__ = [
    "load_dataset",
    "clean_clinical_data",
    "prepare_splits",
    "GaussianNBClassifier",
    "evaluate_predictions",
    "run_stratified_cv",
    "analyze_errors",
    "save_experiment_results",
    "plot_confusion_matrix",
    "plot_roc_and_pr_curves",
    "plot_likelihood_distributions",
    "plot_feature_correlations",
    "plot_threshold_tuning",
    "plot_cross_validation_stability",
]

__version__ = "1.0.0"
