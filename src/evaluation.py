"""
Modul Evaluasi Kinerja Model & Analisis Galat.

Menyediakan fungsi komputasi untuk:
1. Evaluasi metrik klasifikasi (Confusion Matrix, Akurasi, Presisi, Recall, F1-Score).
2. K-Fold Stratified Cross-Validation untuk estimasi kestabilan statistik.
3. Analisis galat klasifikasi (False Positive & False Negative) pada konteks klinis.
4. Serialisasi artefak eksperimen ke format JSON terstruktur.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_predictions(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Menghitung matriks konfusi dan metrik evaluasi klasifikasi standar.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    }


def run_stratified_cv(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Melakukan K-Fold Stratified Cross-Validation untuk mengukur generalisasi model.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1"]
    cv_res = cross_validate(estimator, X, y, cv=skf, scoring=scoring)

    summary: Dict[str, Dict[str, float]] = {}
    for metric in scoring:
        key = f"test_{metric}"
        scores = cv_res[key]
        summary[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
    return summary


def analyze_errors(
    X_test: pd.DataFrame,
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Menganalisis sampel data yang mengalami misklasifikasi (FP dan FN).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    fp_mask = (y_true_arr == 0) & (y_pred_arr == 1)
    fn_mask = (y_true_arr == 1) & (y_pred_arr == 0)

    total_samples = len(y_true_arr)
    fp_count = int(np.sum(fp_mask))
    fn_count = int(np.sum(fn_mask))
    total_errors = fp_count + fn_count

    return {
        "total_test_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": float(total_errors / total_samples) if total_samples > 0 else 0.0,
        "false_positives": {
            "count": fp_count,
            "description": "Pasien sehat didiagnosis mengidap diabetes (False Alarm)",
        },
        "false_negatives": {
            "count": fn_count,
            "description": "Pasien diabetes gagal terdeteksi (Critical Clinical Risk)",
        },
    }


def save_experiment_results(
    results_data: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Menyimpan ringkasan parameter dan hasil evaluasi ke file JSON.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    return dest
