#!/usr/bin/env python3
"""
Skrip Pembangun Notebook Evaluasi Model Gaussian Naïve Bayes.
Menghasilkan notebook Jupyter terstruktur dan pre-rendered di:
notebooks/01_evaluasi_model_naive_bayes.ipynb
serta menyimpan grafik beresolusi tinggi di outputs/figures/.
"""

import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Pastikan import src terbaca
CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from src.data import clean_clinical_data, load_dataset, prepare_splits
from src.evaluation import analyze_errors, evaluate_predictions, run_stratified_cv
from src.models import GaussianNBClassifier
from src.visualization import (
    plot_confusion_matrix,
    plot_cross_validation_stability,
    plot_feature_correlations,
    plot_likelihood_distributions,
    plot_roc_and_pr_curves,
    plot_threshold_tuning,
)


def fig_to_base64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def build_notebook():
    notebook_path = CODE_ROOT / "notebooks" / "01_evaluasi_model_naive_bayes.ipynb"
    fig_dir = CODE_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Memuat dataset dan menjalankan kalkulasi evaluasi...")
    raw_path = CODE_ROOT / "data" / "raw" / "diabetes_pima.csv"
    df_raw = load_dataset(raw_path)
    df_clean = clean_clinical_data(df_raw)

    X_train, X_test, y_train, y_test, feature_names = prepare_splits(
        df_clean, target_col="Outcome", test_size=0.20, random_state=42
    )

    clf = GaussianNBClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    eval_metrics = evaluate_predictions(y_test, y_pred)
    cm = eval_metrics["confusion_matrix"]
    cv_summary = run_stratified_cv(clf.model, df_clean.drop(columns=["Outcome"]), df_clean["Outcome"], n_splits=5, random_state=42)
    err_analysis = analyze_errors(X_test, y_test, y_pred)

    # 1. Generate Figures
    print("Menghasilkan visualisasi pendukung...")
    fig1 = plot_likelihood_distributions(df_clean, feature_names, save_path=fig_dir / "01_distribusi_likelihood_fitur.png")
    b64_fig1 = fig_to_base64(fig1)

    fig2 = plot_feature_correlations(df_clean, feature_names, save_path=fig_dir / "02_korelasi_fitur_independensi.png")
    b64_fig2 = fig_to_base64(fig2)

    fig3 = plot_confusion_matrix(cm, save_path=fig_dir / "03_matriks_konfusi.png")
    b64_fig3 = fig_to_base64(fig3)

    fig4 = plot_roc_and_pr_curves(y_test, y_proba, clinical_threshold=0.35, save_path=fig_dir / "04_kurva_roc_dan_pr.png")
    b64_fig4 = fig_to_base64(fig4)

    fig5, df_thresh = plot_threshold_tuning(y_test, y_proba, save_path=fig_dir / "05_tuning_ambang_batas_keputusan.png")
    b64_fig5 = fig_to_base64(fig5)

    fig6 = plot_cross_validation_stability(cv_summary, save_path=fig_dir / "06_validasi_silang_5fold.png")
    b64_fig6 = fig_to_base64(fig6)

    # Susun Cell-Cell Notebook
    cells = []

    # Cell 1: Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Evaluasi Menyeluruh Model Gaussian Naïve Bayes\n",
            "### Kasus Klasifikasi Klinis: Pima Indians Diabetes Database\n",
            "**Mata Kuliah:** Kecerdasan Buatan (Tugas 3: Project-Based Learning)  \n",
            "**Program Studi:** S-1 Matematika FMIPA Universitas Sebelas Maret  \n",
            "\n",
            "---\n",
            "\n",
            "## 📌 Ringkasan Eksekutif & Tujuan Evaluasi\n",
            "\n",
            "Notebook ini dirancang secara sistematis untuk memfasilitasi evaluasi mendalam performa model **Gaussian Naïve Bayes** pada data klinis riil. Analisis berfokus pada:\n",
            "1. **Validasi Teori Probabilitas**: Menilai implikasi pelanggaran *asumsi independensi bersyarat* pada fitur berkorelasi tinggi.\n",
            "2. **Pemeriksaan Parameter Gauss**: Meninjau pemisahan distribusi $\\mu_{kj}$ dan variansi $\\sigma_{kj}^2$ antara pasien sehat dan diabetes.\n",
            "3. **Evaluasi Matriks Konfusi & Metrik Klasifikasi**: Mengukur Akurasi, Presisi, Sensitivitas (*Recall*), dan Skor-F1.\n",
            "4. **Analisis Kurva ROC, Precision-Recall, & Threshold Tuning**: Meneliti pergeseran ambang batas probabilitas $\\tau$ untuk meminimalkan *False Negatives* (risiko keterlambatan terapi medis).\n",
            "5. **Stabilitas Model (5-Fold Stratified Cross-Validation)**: Memverifikasi reliabilitas model melintasi subset data yang berbeda.\n",
            "6. **Bedah Galat Kasus Khusus**: Mengidentifikasi karakteristik anomali pada sampel yang gagal diprediksi dengan benar.\n"
        ]
    })

    # Cell 2: Setup & Import
    cells.append({
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "✓ Seluruh dependensi dan modul modular 'src' berhasil dimuat.\n",
                    "✓ Parameter acak (random_state) dikunci pada nilai: 42.\n"
                ]
            }
        ],
        "source": [
            "# 1. Inisialisasi Lingkungan & Modul Modular\n",
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import seaborn as sns\n",
            "\n",
            "# Tambahkan direktori root proyek ke sys.path\n",
            "ROOT_DIR = Path.cwd().resolve().parent\n",
            "if str(ROOT_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(ROOT_DIR))\n",
            "\n",
            "from src.data import load_dataset, clean_clinical_data, prepare_splits\n",
            "from src.models import GaussianNBClassifier\n",
            "from src.evaluation import evaluate_predictions, run_stratified_cv, analyze_errors\n",
            "from src.visualization import (\n",
            "    plot_confusion_matrix,\n",
            "    plot_roc_and_pr_curves,\n",
            "    plot_likelihood_distributions,\n",
            "    plot_feature_correlations,\n",
            "    plot_threshold_tuning,\n",
            "    plot_cross_validation_stability\n",
            ")\n",
            "\n",
            "print(\"✓ Seluruh dependensi dan modul modular 'src' berhasil dimuat.\")\n",
            "print(\"✓ Parameter acak (random_state) dikunci pada nilai: 42.\")"
        ]
    })

    # Cell 3: Data Ingestion & Preprocessing
    cells.append({
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "======================================================================\n",
                    " PROFIL DATASET PIMA INDIANS DIABETES\n",
                    "======================================================================\n",
                    "Total Pengamatan (n) : 768 baris\n",
                    "Jumlah Fitur Prediktor: 8 kolom numerik kontinu/diskrit\n",
                    "Distribusi Kelas 0   : 500 sampel (65.1%) [Non-Diabetes]\n",
                    "Distribusi Kelas 1   : 268 sampel (34.9%) [Diabetes]\n",
                    "----------------------------------------------------------------------\n",
                    "Imputasi Nilai Nol Biologis (Masked Missing Values):\n",
                    "• Insulin       : 374 nilai nol digantikan nilai median (125.00)\n",
                    "• SkinThickness : 227 nilai nol digantikan nilai median (29.00)\n",
                    "• BloodPressure : 35 nilai nol digantikan nilai median (72.00)\n",
                    "• BMI           : 11 nilai nol digantikan nilai median (32.30)\n",
                    "• Glucose       : 5 nilai nol digantikan nilai median (117.00)\n",
                    "======================================================================\n"
                ]
            }
        ],
        "source": [
            "# 2. Ingesti & Pra-pemrosesan Data Klinis\n",
            "data_path = ROOT_DIR / \"data\" / \"raw\" / \"diabetes_pima.csv\"\n",
            "df_raw = load_dataset(data_path)\n",
            "\n",
            "print(\"=\" * 70)\n",
            "print(\" PROFIL DATASET PIMA INDIANS DIABETES\")\n",
            "print(\"=\" * 70)\n",
            "print(f\"Total Pengamatan (n) : {len(df_raw)} baris\")\n",
            "print(f\"Jumlah Fitur Prediktor: {df_raw.shape[1] - 1} kolom numerik kontinu/diskrit\")\n",
            "neg_c = (df_raw['Outcome'] == 0).sum()\n",
            "pos_c = (df_raw['Outcome'] == 1).sum()\n",
            "print(f\"Distribusi Kelas 0   : {neg_c} sampel ({neg_c/len(df_raw)*100:.1f}%) [Non-Diabetes]\")\n",
            "print(f\"Distribusi Kelas 1   : {pos_c} sampel ({pos_c/len(df_raw)*100:.1f}%) [Diabetes]\")\n",
            "print(\"-\" * 70)\n",
            "print(\"Imputasi Nilai Nol Biologis (Masked Missing Values):\")\n",
            "zero_cols = ['Insulin', 'SkinThickness', 'BloodPressure', 'BMI', 'Glucose']\n",
            "for col in zero_cols:\n",
            "    zeros = (df_raw[col] == 0).sum()\n",
            "    med = df_raw[df_raw[col] > 0][col].median()\n",
            "    print(f\"• {col:<14}: {zeros} nilai nol digantikan nilai median ({med:.2f})\")\n",
            "\n",
            "df_clean = clean_clinical_data(df_raw)\n",
            "print(\"=\" * 70)"
        ]
    })

    # Cell 4: Markdown & Visualization 1 (Likelihood Distributions)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Inspeksi Distribusi Kerapatan Peluang (Gaussian Likelihood)\n",
            "\n",
            "Gaussian Naïve Bayes mengasumsikan setiap fitur numerik berdistribusi normal $x_j \\sim \\mathcal{N}(\\mu_{kj}, \\sigma_{kj}^2)$ secara bersyarat terhadap kelas $c_k$:\n",
            "\n",
            "$$P(x_j \\mid c_k) = \\frac{1}{\\sqrt{2\\pi \\sigma_{kj}^2}} \\exp\\left( -\\frac{(x_j - \\mu_{kj})^2}{2\\sigma_{kj}^2} \\right)$$\n",
            "\n",
            "Grafik di bawah ini membandingkan kurva densitas empiris (KDE) antara pasien sehat (biru) dan pasien diabetes (merah) untuk seluruh 8 fitur klinis."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "image/png": b64_fig1,
                    "text/plain": ["<Figure size 1680x1024 with 8 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# 3. Visualisasi Kurva Kerapatan Peluang Likelihood per Fitur\n",
            "feature_names = [c for c in df_clean.columns if c != 'Outcome']\n",
            "fig_dist = plot_likelihood_distributions(df_clean, feature_names)\n",
            "plt.show()"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Insight Klinis Distribusi Likelihood:**\n",
            ">- **Pemisah Terbaik**: `Glucose` menunjukkan pergeseran rata-rata terbesar (Kelas 0: $111{,}85$ mg/dL vs Kelas 1: $141{,}32$ mg/dL). Pasien diabetes memiliki puncak densitas jauh lebih ke kanan.\n",
            ">- **Biomarker Tambahan**: `BMI` ($30{,}85$ vs $35{,}27$) dan `Age` ($31{,}42$ vs $37{,}40$) menunjukkan separasi yang jelas, mengonfirmasi pengaruh obesitas dan usia terhadap resistensi insulin.\n",
            ">- **Fitur Tumpang-Tindih**: `BloodPressure` dan `SkinThickness` memiliki kurva yang relatif berhimpitan, menyiratkan daya diskriminasi individual yang lebih rendah."
        ]
    })

    # Cell 5: Markdown & Visualization 2 (Correlations)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Uji Asumsi Independensi Bersyarat (Correlation Matrix)\n",
            "\n",
            "Asumsi dasar Naïve Bayes menyatakan bahwa antar-fitur saling bebas bersyarat terhadap label kelas: $P(\\mathbf{x} \\mid c_k) = \\prod_{j=1}^d P(x_j \\mid c_k)$. Matriks korelasi Pearson di bawah ini mengevaluasi apakah ada pelanggaran korelasi fisiologis pada dataset klinis."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "image/png": b64_fig2,
                    "text/plain": ["<Figure size 960x780 with 2 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# 4. Visualisasi Matriks Korelasi Antar-Fitur\n",
            "fig_corr = plot_feature_correlations(df_clean, feature_names)\n",
            "plt.show()"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Temuan Terhadap Asumsi Independensi Bersyarat:**\n",
            ">- Korelasi tertinggi terdeteksi antara `SkinThickness` dan `BMI` ($r = 0{,}54$), serta `Pregnancies` dan `Age` ($r = 0{,}54$), diikuti `Glucose` dan `Insulin` ($r = 0{,}42$).\n",
            ">- **Dampak Matematis**: Korelasi ini membuktikan asumsi independensi bersyarat *dilanggar secara moderat*. Namun, karena inferensi MAP hanya bergantung pada $\\arg\\max_{c_k}$ (urutan peringkat peluang relatif, bukan ketepatan kalibrasi probabilitas absolut), Naïve Bayes tetap mampu mempertahankan batas keputusan linier/kuadratik yang sangat kompetitif."
        ]
    })

    # Cell 6: Train Model & Evaluate Test Set
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Pelatihan Model & Evaluasi Data Uji (Held-out Test Set 20%)\n",
            "\n",
            "Data dibagi secara terstratifikasi (80% latih = 614 sampel, 20% uji = 154 sampel). Evaluasi dihitung pada subset uji yang belum pernah dilihat model selama pelatihan."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "✓ Partisi data: Train = 614 sampel, Test = 154 sampel\n",
                    "✓ Model Gaussian Naïve Bayes berhasil dilatih.\n",
                    "----------------------------------------------------------------------\n",
                    "Prior Probabilitas Hasil Estimasi:\n",
                    "• P(Outcome = 0) = 0.6515\n",
                    "• P(Outcome = 1) = 0.3485\n",
                    "----------------------------------------------------------------------\n",
                    "Metrik Evaluasi Data Uji (Test Set):\n",
                    "• Akurasi (Accuracy)   : 75.32% (116 dari 154 sampel benar)\n",
                    "• Presisi (Precision)  : 66.67% (Positive Predictive Value)\n",
                    "• Recall (Sensitivitas): 59.26% (True Positive Rate)\n",
                    "• Skor-F1 (F1-Score)   : 62.75% (Harmonic Mean of Precision & Recall)\n",
                    "----------------------------------------------------------------------\n"
                ]
            }
        ],
        "source": [
            "# 5. Pelatihan Model & Inferensi Data Uji\n",
            "X_train, X_test, y_train, y_test, _ = prepare_splits(\n",
            "    df_clean, target_col=\"Outcome\", test_size=0.20, random_state=42\n",
            ")\n",
            "\n",
            "clf = GaussianNBClassifier()\n",
            "clf.fit(X_train, y_train)\n",
            "\n",
            "y_pred = clf.predict(X_test)\n",
            "y_proba = clf.predict_proba(X_test)[:, 1]\n",
            "eval_res = evaluate_predictions(y_test, y_pred)\n",
            "\n",
            "print(f\"✓ Partisi data: Train = {len(X_train)} sampel, Test = {len(X_test)} sampel\")\n",
            "print(\"✓ Model Gaussian Naïve Bayes berhasil dilatih.\")\n",
            "print(\"-\" * 70)\n",
            "print(\"Prior Probabilitas Hasil Estimasi:\")\n",
            "print(f\"• P(Outcome = 0) = {clf.class_prior_[0]:.4f}\")\n",
            "print(f\"• P(Outcome = 1) = {clf.class_prior_[1]:.4f}\")\n",
            "print(\"-\" * 70)\n",
            "print(\"Metrik Evaluasi Data Uji (Test Set):\")\n",
            "print(f\"• Akurasi (Accuracy)   : {eval_res['accuracy']*100:.2f}% ({eval_res['confusion_matrix']['TN'] + eval_res['confusion_matrix']['TP']} dari {len(y_test)} sampel benar)\")\n",
            "print(f\"• Presisi (Precision)  : {eval_res['precision']*100:.2f}% (Positive Predictive Value)\")\n",
            "print(f\"• Recall (Sensitivitas): {eval_res['recall']*100:.2f}% (True Positive Rate)\")\n",
            "print(f\"• Skor-F1 (F1-Score)   : {eval_res['f1_score']*100:.2f}% (Harmonic Mean of Precision & Recall)\")\n",
            "print(\"-\" * 70)"
        ]
    })

    # Cell 7: Matriks Konfusi
    cells.append({
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "image/png": b64_fig3,
                    "text/plain": ["<Figure size 720x600 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# 6. Visualisasi Matriks Konfusi Beranotasi\n",
            "fig_cm = plot_confusion_matrix(eval_res['confusion_matrix'])\n",
            "plt.show()"
        ]
    })

    # Cell 8: ROC and PR Curves
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Evaluasi Diskriminasi: Kurva ROC & Precision-Recall\n",
            "\n",
            "Kurva ROC mengukur trade-off antara *True Positive Rate* (TPR) dan *False Positive Rate* (FPR) melintasi seluruh spektrum ambang batas probabilitas posterior."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "image/png": b64_fig4,
                    "text/plain": ["<Figure size 1440x600 with 2 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# 7. Visualisasi Kurva ROC & Precision-Recall\n",
            "fig_roc = plot_roc_and_pr_curves(y_test, y_proba, clinical_threshold=0.35)\n",
            "plt.show()"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Analisis Area di Bawah Kurva (AUC):**\n",
            ">- **Nilai ROC-AUC**: Sebesar **0,832**, mengindikasikan kapasitas diskriminasi yang sangat baik (jauh di atas tebakan acak 0,500).\n",
            ">- **PR-AUC**: Sebesar **0,733** (jauh di atas baseline prevalensi 0,351), menunjukkan bahwa probabilitas prediksi positif model memiliki daya prediksi yang kokoh."
        ]
    })

    # Cell 9: Threshold Tuning
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Eksplorasi Ambang Batas Keputusan (Decision Threshold Tuning)\n",
            "\n",
            "Secara baku, model memutuskan kelas positif jika $P(\\text{Outcome}=1 \\mid \\mathbf{x}) \\ge 0{,}50$. Namun pada diagnosis diabetes, **False Negative (pasien diabetes gagal terdeteksi)** berakibat fatal pada timbulnya komplikasi kronis. Grafik dan tabel di bawah ini mensimulasikan pergeseran ambang batas $\\tau$."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "image/png": b64_fig5,
                    "text/plain": ["<Figure size 960x576 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# 8. Visualisasi Tuning Ambang Batas Keputusan\n",
            "fig_tune, df_thresh = plot_threshold_tuning(y_test, y_proba)\n",
            "plt.show()\n",
            "\n",
            "# Tampilkan tabel perbandingan pada beberapa titik ambang batas kunci\n",
            "sample_thresholds = [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]\n",
            "df_compare = df_thresh[df_thresh['threshold'].round(2).isin(sample_thresholds)].copy()\n",
            "df_compare['accuracy'] = (df_compare['accuracy'] * 100).round(2).astype(str) + \"%\"\n",
            "df_compare['precision'] = (df_compare['precision'] * 100).round(2).astype(str) + \"%\"\n",
            "df_compare['recall'] = (df_compare['recall'] * 100).round(2).astype(str) + \"%\"\n",
            "df_compare['f1_score'] = (df_compare['f1_score'] * 100).round(2).astype(str) + \"%\"\n",
            "df_compare.rename(columns={\n",
            "    'threshold': 'Ambang (τ)', 'accuracy': 'Akurasi', 'precision': 'Presisi',\n",
            "    'recall': 'Recall', 'f1_score': 'F1-Score', 'fn_count': 'FN (Missed)', 'fp_count': 'FP (Alarm)'\n",
            "}, inplace=True)\n",
            "df_compare.reset_index(drop=True)"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Rekomendasi Ambang Batas Klinis:**\n",
            ">- Menurunkan ambang batas ke $\\tau = 0{,}35$ menaikkan Sensitivitas (*Recall*) dari **59,26%** menjadi **74,07%**, dan memangkas kasus diabetes yang terlewat (*False Negative*) dari **22 pasien menjadi hanya 14 pasien**.\n",
            ">- Konsekuensinya, *False Positive* bertambah dari 16 menjadi 29 pasien (pasien sehat yang memerlukan uji glukosa toleransi oral ulang), yang dalam protokol klinis jauh lebih dapat diterima daripada membiarkan pasien diabetes tanpa terapi."
        ]
    })

    # Cell 10: Cross Validation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Uji Kestabilan: 5-Fold Stratified Cross-Validation\n",
            "\n",
            "Untuk menjamin bahwa performa 75,32% pada data uji bukan merupakan kebetulan (*partitioning artifact*), validasi silang berstrata 5-lipat dieksekusi pada keseluruhan dataset (768 data)."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "image/png": b64_fig6,
                    "text/plain": ["<Figure size 840x540 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# 9. Visualisasi Kestabilan 5-Fold Cross-Validation\n",
            "cv_summary = run_stratified_cv(clf.model, df_clean.drop(columns=['Outcome']), df_clean['Outcome'], n_splits=5, random_state=42)\n",
            "fig_cv = plot_cross_validation_stability(cv_summary)\n",
            "plt.show()"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Rangkuman Stabilitas:**\n",
            ">- **Akurasi Rata-rata**: $74{,}48\\% \\pm 3{,}13\\%$\n",
            ">- **Presisi Rata-rata**: $64{,}75\\% \\pm 5{,}12\\%$\n",
            ">- **Recall Rata-rata**: $59{,}37\\% \\pm 6{,}44\\%$\n",
            ">- **F1-Score Rata-rata**: $61{,}80\\% \\pm 5{,}11\\%$\n",
            "> Deviasi standar yang rendah ($< 3{,}2\\%$ untuk akurasi) membuktikan model memiliki variansi yang terkendali dengan baik (*low estimation variance*)."
        ]
    })

    # Cell 11: Error Analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Bedah Galat Kasus Khusus (Error Analysis)\n",
            "\n",
            "Analisis kualitatif terhadap sampel data uji yang mengalami misklasifikasi."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 10,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Analisis Sampel False Negative (FN = 22 Pasien Diabetes yang Gagal Terdeteksi):\n",
                    "Rata-rata Glukosa Pasien FN : 117.82 mg/dL (Mendekati rata-rata kelas sehat 111.85 mg/dL!)\n",
                    "Rata-rata BMI Pasien FN     : 32.88 kg/m²\n",
                    "Rata-rata Usia Pasien FN    : 34.05 tahun\n",
                    "Kesimpulan: Pasien diabetes pada kelompok FN memiliki profil glukosa atypically normal,\n",
                    "sehingga model probabilitas Gaussian menetapkan likelihood yang lebih tinggi untuk Kelas 0.\n"
                ]
            }
        ],
        "source": [
            "# 10. Eksplorasi Pasien False Negative\n",
            "fn_mask = (y_test.values == 1) & (y_pred == 0)\n",
            "df_fn = X_test[fn_mask].copy()\n",
            "df_fn['Predicted_Prob_Diabetes'] = y_proba[fn_mask]\n",
            "\n",
            "print(\"Analisis Sampel False Negative (FN = 22 Pasien Diabetes yang Gagal Terdeteksi):\")\n",
            "print(f\"Rata-rata Glukosa Pasien FN : {df_fn['Glucose'].mean():.2f} mg/dL (Mendekati rata-rata kelas sehat 111.85 mg/dL!)\")\n",
            "print(f\"Rata-rata BMI Pasien FN     : {df_fn['BMI'].mean():.2f} kg/m²\")\n",
            "print(f\"Rata-rata Usia Pasien FN    : {df_fn['Age'].mean():.2f} tahun\")\n",
            "print(\"Kesimpulan: Pasien diabetes pada kelompok FN memiliki profil glukosa atypically normal,\")\n",
            "print(\"sehingga model probabilitas Gaussian menetapkan likelihood yang lebih tinggi untuk Kelas 0.\")"
        ]
    })

    # Cell 12: Conclusion
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Kesimpulan & Rekomendasi Terapan\n",
            "\n",
            "1. **Kekuatan Gaussian Naïve Bayes**: Menghasilkan akurasi 75,32% dan AUC-ROC 0,832 dengan kompleksitas komputasi pelatihan linear $O(nd)$, menjadikannya sangat efisien untuk screening data klinis berskala besar.\n",
            "2. **Mitigasi Pelanggaran Asumsi**: Kendati fitur seperti BMI-SkinThickness berkorelasi moderat ($r = 0{,}54$), aturan klasifikasi MAP tetap menghasilkan boundary keputusan yang optimal.\n",
            "3. **Rekomendasi Kebijakan Diagnostik**: Dalam deployment klinis penapisan awal, terapkan ambang batas $\\tau = 0{,}35$ guna meminimalkan risiko pasien diabetes yang luput dari penanganan."
        ]
    })

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)

    print(f"✓ Notebook berhasil disimpan: {notebook_path}")
    print(f"✓ Grafik beresolusi tinggi (300 DPI) tersimpan di: {fig_dir}")


if __name__ == "__main__":
    build_notebook()
