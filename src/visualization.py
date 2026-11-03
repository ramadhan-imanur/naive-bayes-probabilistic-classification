"""
Modul Visualisasi Evaluasi Model & Analisis Data Klinis.

Menyediakan fungsi visualisasi standar industri untuk:
1. Matriks Konfusi Beranotasi (Annotated Confusion Matrix).
2. Kurva ROC dan Precision-Recall dengan penanda ambang batas klinis.
3. Distribusi Likelihood Normal Gauss per Fitur (Kelas 0 vs Kelas 1).
4. Matriks Korelasi Fitur (Uji Pelanggaran Asumsi Independensi).
5. Plot Stabilitas Metrik Validasi Silang (5-Fold Cross-Validation).
6. Analisis Trade-off Ambang Batas Keputusan (Threshold Tuning).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Mengatur konfigurasi gaya visual profesional
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_confusion_matrix(
    cm: Union[np.ndarray, Dict[str, int]],
    class_labels: Tuple[str, str] = ("Non-Diabetes (0)", "Diabetes (1)"),
    title: str = "Matriks Konfusi - Gaussian Naïve Bayes",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Menghasilkan heatmap matriks konfusi dengan anotasi kuantitatif dan persentase.
    """
    if isinstance(cm, dict):
        matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    else:
        matrix = np.asarray(cm)

    total = np.sum(matrix)
    annot = np.empty_like(matrix, dtype=object)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = (count / total) * 100
            annot[i, j] = f"{labels[i][j]}\n{count:,}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Prediksi Model", fontsize=11, labelpad=8)
    ax.set_ylabel("Status Aktual Pasien", fontsize=11, labelpad=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_roc_and_pr_curves(
    y_true: Union[pd.Series, np.ndarray],
    y_proba: np.ndarray,
    clinical_threshold: float = 0.35,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Memplot kurva ROC (dengan AUC) dan Precision-Recall curve berdampingan.
    """
    y_true_arr = np.asarray(y_true)
    fpr, tpr, roc_thresh = roc_curve(y_true_arr, y_proba)
    roc_auc = auc(fpr, tpr)

    precisions, recalls, pr_thresh = precision_recall_curve(y_true_arr, y_proba)
    pr_auc = auc(recalls, precisions)

    # Identifikasi titik ambang batas default (0.5) dan klinis (clinical_threshold)
    idx_50 = np.argmin(np.abs(roc_thresh - 0.50))
    idx_clin = np.argmin(np.abs(roc_thresh - clinical_threshold))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # 1. Plot ROC Curve
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=1.2, label="Random Guess (AUC = 0.500)")
    ax1.scatter(
        fpr[idx_50], tpr[idx_50], color="#d62728", s=80, zorder=5,
        label=f"Default (τ = 0.50): Sens={tpr[idx_50]:.2f}, FPR={fpr[idx_50]:.2f}"
    )
    ax1.scatter(
        fpr[idx_clin], tpr[idx_clin], color="#2ca02c", s=80, marker="^", zorder=5,
        label=f"Klinis (τ = {clinical_threshold:.2f}): Sens={tpr[idx_clin]:.2f}, FPR={fpr[idx_clin]:.2f}"
    )
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Spesifisitas)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivitas / Recall)", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)

    # 2. Plot Precision-Recall Curve
    no_skill = np.sum(y_true_arr == 1) / len(y_true_arr)
    ax2.plot(recalls, precisions, color="#ff7f0e", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.3f})")
    ax2.plot([0, 1], [no_skill, no_skill], color="#888888", linestyle="--", lw=1.2, label=f"Baseline Prevalensi ({no_skill:.2f})")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel("Recall (Sensitivitas)", fontsize=11)
    ax2.set_ylabel("Presisi (Positive Predictive Value)", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_likelihood_distributions(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "Outcome",
    class_labels: Tuple[str, str] = ("Non-Diabetes", "Diabetes"),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Menampilkan kurva densitas KDE untuk setiap fitur kontinu, membedakan Kelas 0 dan Kelas 1.
    """
    n_features = len(features)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), dpi=120)
    axes = np.array(axes).flatten()

    colors = {0: "#1f77b4", 1: "#d62728"}

    for idx, feat in enumerate(features):
        ax = axes[idx]
        for class_val in [0, 1]:
            subset = df[df[target_col] == class_val][feat]
            sns.kdeplot(
                subset,
                ax=ax,
                color=colors[class_val],
                label=f"{class_labels[class_val]} (n={len(subset)})",
                fill=True,
                alpha=0.25,
                linewidth=2.0,
            )
        ax.set_title(f"Distribusi: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel(feat, fontsize=9.5)
        ax.set_ylabel("Kerapatan Peluang (PDF)", fontsize=9.5)
        if idx == 0:
            ax.legend(fontsize=8.5, loc="upper right")
        else:
            ax.legend().remove()

    # Hapus axis berlebih jika jumlah subplot tidak habis dibagi 3
    for extra_idx in range(n_features, len(axes)):
        fig.delaxes(axes[extra_idx])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_feature_correlations(
    df: pd.DataFrame,
    features: List[str],
    title: str = "Matriks Korelasi Pearson Fitur Klinis",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Menghasilkan heatmap korelasi bivariat untuk menguji asumsi independensi bersyarat.
    """
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=120)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.3,
        vmax=0.8,
        center=0.0,
        linewidths=1.0,
        linecolor="white",
        annot_kws={"fontsize": 9.5, "fontweight": "medium"},
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_threshold_tuning(
    y_true: Union[pd.Series, np.ndarray],
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Menganalisis pergerakan metrik (Precision, Recall, F1-Score) terhadap variasi ambang batas keputusan tau.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    y_true_arr = np.asarray(y_true)
    records = []

    for tau in thresholds:
        y_pred = (y_proba >= tau).astype(int)
        tp = np.sum((y_true_arr == 1) & (y_pred == 1))
        fp = np.sum((y_true_arr == 0) & (y_pred == 1))
        fn = np.sum((y_true_arr == 1) & (y_pred == 0))
        tn = np.sum((y_true_arr == 0) & (y_pred == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(y_true_arr)

        records.append({
            "threshold": tau,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "fn_count": fn,
            "fp_count": fp,
        })

    df_thresh = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    ax.plot(df_thresh["threshold"], df_thresh["accuracy"], label="Akurasi", color="#7f7f7f", linestyle="--", lw=1.8)
    ax.plot(df_thresh["threshold"], df_thresh["precision"], label="Presisi", color="#1f77b4", lw=2.2)
    ax.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall (Sensitivitas)", color="#2ca02c", lw=2.5)
    ax.plot(df_thresh["threshold"], df_thresh["f1_score"], label="F1-Score", color="#d62728", lw=2.2)

    ax.axvline(0.50, color="#888888", linestyle=":", lw=1.5, label="Default τ = 0.50")
    ax.axvline(0.35, color="#2ca02c", linestyle=":", lw=1.5, label="Optimal Klinis τ = 0.35")

    ax.set_title("Pengaruh Ambang Batas Keputusan (τ) terhadap Metrik Klinis", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Ambang Batas Probabilitas Keputusan (τ)", fontsize=10.5)
    ax.set_ylabel("Nilai Metrik Evaluasi [0, 1]", fontsize=10.5)
    ax.set_xlim([0.08, 0.92])
    ax.set_ylim([0.20, 1.02])
    ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig, df_thresh


def plot_cross_validation_stability(
    cv_summary: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Membuat diagram batang kestabilan performa validasi silang 5-Fold beserta interval simpangan baku.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Akurasi", "Presisi", "Recall", "F1-Score"]
    means = [cv_summary[m]["mean"] * 100 for m in metrics]
    stds = [cv_summary[m]["std"] * 100 for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=["#4e79a7", "#59a14f", "#edc948", "#e15759"], alpha=0.85, edgecolor="#333333", lw=1.2)

    ax.set_ylim([0, 100])
    ax.set_ylabel("Skor Performa Rata-rata (%)", fontsize=10.5)
    ax.set_title("Stabilitas Kinerja: 5-Fold Stratified Cross-Validation", fontsize=12, fontweight="bold", pad=12)

    for bar, mean_val, std_val in zip(bars, means, stds):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval / 2.0,
            f"{mean_val:.2f}%\n± {std_val:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig
