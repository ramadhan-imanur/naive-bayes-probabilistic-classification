"""
Probabilistic Machine Learning - Gaussian Naïve Bayes Clinical Modeling
Mata Kuliah: Kecerdasan Buatan (Tugas 3: Project-Based Learning)
Program Studi S-1 Matematika FMIPA Universitas Sebelas Maret
"""

from .data import load_dataset, clean_clinical_data, prepare_splits
from .models import GaussianNBClassifier
from .evaluation import (
    evaluate_predictions,
    run_stratified_cv,
    analyze_errors,
    save_experiment_results,
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
]

__version__ = "1.0.0"
