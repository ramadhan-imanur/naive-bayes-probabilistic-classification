#!/usr/bin/env python3
"""
Eksperimen Komputasi Pemodelan Gaussian Naïve Bayes Berbasis Dataset Publik.

Orkestrasi Pipeline Machine Learning End-to-End:
1. Ingesti dan pra-pemrosesan data klinis (imputasi nilai nol biologis tidak valid).
2. Partisi data terstratifikasi (Train 80% : Test 20%).
3. Pelatihan Gaussian Naïve Bayes & estimasi parameter (Prior, Mean, Variansi).
4. Evaluasi performa pada data uji (Confusion Matrix, Akurasi, Presisi, Recall, F1-Score).
5. Validasi kestabilan model (5-Fold Stratified Cross-Validation).
6. Analisis galat klinis (False Positive vs False Negative).
7. Serialisasi artefak hasil eksperimen ke format JSON.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Menambahkan direktori root proyek ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    clean_clinical_data,
    load_dataset,
    prepare_splits,
    resolve_default_dataset_path,
)
from src.evaluation import (
    analyze_errors,
    evaluate_predictions,
    run_stratified_cv,
    save_experiment_results,
)
from src.models import GaussianNBClassifier


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline Eksperimen Gaussian Naïve Bayes pada Dataset Klinis Publik."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path ke berkas dataset CSV (default: auto-detect data/raw/diabetes_pima.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "metrics"),
        help="Direktori penyimpanan hasil metrik JSON (default: outputs/metrics/)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(PROJECT_ROOT / "models"),
        help="Direktori penyimpanan bobot model terlatik (default: models/)",
    )
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Simpan dataset hasil pra-pemrosesan ke data/processed/",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporsi data uji (default: 0.20 atau 20%)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Jumlah fold untuk Stratified Cross-Validation (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed bilangan acak untuk reprodusibilitas (default: 42)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Jangan simpan hasil eksperimen ke berkas JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # 1. Resolusi Path Dataset
    if args.data_path:
        dataset_path = Path(args.data_path)
    else:
        dataset_path = resolve_default_dataset_path()

    print("=" * 75)
    print(" EKSPERIMEN GAUSSIAN NAÏVE BAYES: PIMA INDIANS DIABETES DATASET")
    print(" Repositori: Harvard Dataverse (doi:10.7910/DVN/HIBCFL) & Kaggle")
    print(f" Target Berkas: {dataset_path}")
    print("=" * 75)

    if not dataset_path.exists():
        print(f"\n[ERROR] Berkas dataset tidak ditemukan pada: {dataset_path}")
        sys.exit(1)

    # 2. Ingesti Data
    df_raw = load_dataset(dataset_path)
    n_samples, n_cols = df_raw.shape
    n_features = n_cols - 1
    neg_count = int((df_raw["Outcome"] == 0).sum())
    pos_count = int((df_raw["Outcome"] == 1).sum())

    print("\n1. PROFIL DATASET:")
    print(f"   Jumlah Sampel (n) : {n_samples}")
    print(f"   Jumlah Fitur (d)  : {n_features}")
    print(
        f"   Distribusi Kelas  : 0 (Negatif) = {neg_count} sampel ({neg_count/n_samples*100:.1f}%), "
        f"1 (Positif) = {pos_count} sampel ({pos_count/n_samples*100:.1f}%)"
    )

    # 3. Partisi Data & Pra-pemrosesan Bebas Kebocoran (Zero Data Leakage)
    # Partisi dilakukan pada data mentah terlebih dahulu; median dihitung HANYA dari data latih
    X_train, X_test, y_train, y_test, feature_names = prepare_splits(
        df_raw,
        target_col="Outcome",
        test_size=args.test_size,
        random_state=args.random_state,
        impute_missing=True,
    )

    print(f"\n2. PARTISI DATA ({int((1-args.test_size)*100)}:{int(args.test_size*100)} STRATIFIKASI TANPA KEBOCORAN):")
    print(f"   Data Latih (Train) : {len(X_train)} sampel")
    print(f"   Data Uji (Test)    : {len(X_test)} sampel")

    # 4. Pelatihan Gaussian Naïve Bayes
    clf = GaussianNBClassifier()
    clf.fit(X_train, y_train)

    prior_0 = float(clf.class_prior_[0])
    prior_1 = float(clf.class_prior_[1])
    params_dict = clf.get_parameters_dict(feature_names)
    epsilon_val = clf.epsilon_

    print("\n3. PARAMETER PELUANG HASIL PELATIHAN:")
    print(f"   Prior P(Outcome = 0) : {prior_0:.4f} (ln = {np.log(prior_0):.6f})")
    print(f"   Prior P(Outcome = 1) : {prior_1:.4f} (ln = {np.log(prior_1):.6f})")
    print(f"   Variance Smoothing epsilon_ : {epsilon_val:.6e}")
    print("\n   Parameter Likelihood Normal Gauss per Fitur (4 Desimal):")
    print(f"   {'Fitur':<25} | {'Mean (Class 0)':<15} | {'Mean (Class 1)':<15} | {'Var (Class 0)':<15} | {'Var (Class 1)':<15}")
    print("   " + "-" * 90)
    for feat in feature_names:
        m0 = params_dict[feat]["mean_0"]
        m1 = params_dict[feat]["mean_1"]
        v0 = params_dict[feat]["var_0"]
        v1 = params_dict[feat]["var_1"]
        print(f"   {feat:<25} | {m0:<15.4f} | {m1:<15.4f} | {v0:<15.4f} | {v1:<15.4f}")

    # 5. Evaluasi pada Data Uji
    y_pred = clf.predict(X_test)
    eval_metrics = evaluate_predictions(y_test, y_pred)
    cm = eval_metrics["confusion_matrix"]

    print("\n4. HASIL EVALUASI DATA UJI (TEST SET):")
    print("   Matriks Konfusi (Confusion Matrix):")
    print("                     Prediksi 0 (Negatif)   Prediksi 1 (Positif)")
    print(f"   Aktual 0 (Negatif)      TN = {cm['TN']:<18} FP = {cm['FP']}")
    print(f"   Aktual 1 (Positif)      FN = {cm['FN']:<18} TP = {cm['TP']}")
    print("\n   Metrik Kinerja:")
    print(f"   - Akurasi (Accuracy)   : {eval_metrics['accuracy']:.4f} ({eval_metrics['accuracy']*100:.2f}%)")
    print(f"   - Presisi (Precision)  : {eval_metrics['precision']:.4f} ({eval_metrics['precision']*100:.2f}%)")
    print(f"   - Recall (Sensitivitas): {eval_metrics['recall']:.4f} ({eval_metrics['recall']*100:.2f}%)")
    print(f"   - F1-Score             : {eval_metrics['f1_score']:.4f} ({eval_metrics['f1_score']*100:.2f}%)")

    # 6. Validasi Silang Bebas Kebocoran (K-Fold Pipeline Stratified Cross-Validation)
    X_raw = df_raw.drop(columns=["Outcome"])
    y_raw = df_raw["Outcome"]
    cv_summary = run_stratified_cv(
        clf.model,
        X_raw,
        y_raw,
        n_splits=args.cv_folds,
        random_state=args.random_state,
        use_pipeline=True,
    )

    print(f"\n5. VALIDASI SILANG {args.cv_folds}-FOLD STRATIFIKASI DENGAN PIPELINE (BEBAS LEAKAGE, ddof=1):")
    print(f"   - Akurasi Rata-rata    : {cv_summary['accuracy']['mean']:.4f} +/- {cv_summary['accuracy']['std']:.4f}")
    print(f"   - Presisi Rata-rata    : {cv_summary['precision']['mean']:.4f} +/- {cv_summary['precision']['std']:.4f}")
    print(f"   - Recall Rata-rata     : {cv_summary['recall']['mean']:.4f} +/- {cv_summary['recall']['std']:.4f}")
    print(f"   - F1-Score Rata-rata   : {cv_summary['f1']['mean']:.4f} +/- {cv_summary['f1']['std']:.4f}")

    # 7. Analisis Galat (Error Analysis)
    err_analysis = analyze_errors(X_test, y_test, y_pred)
    fp_count = err_analysis["false_positives"]["count"]
    fn_count = err_analysis["false_negatives"]["count"]

    print("\n6. ANALISIS GALAT (MISKLASIFIKASI):")
    print(f"   - Total Galat Prediksi        : {err_analysis['total_errors']} dari {err_analysis['total_test_samples']} sampel ({err_analysis['error_rate']*100:.2f}%)")
    print(f"   - False Positive (FP)         : {fp_count} sampel ({err_analysis['false_positives']['description']})")
    print(f"   - False Negative (FN)         : {fn_count} sampel ({err_analysis['false_negatives']['description']})")
    print("   Catatan Kritis: Pada diagnosis medis, meminimalkan FN sangat krusial guna mencegah keterlambatan intervensi klinis.")

    # 8. Simpan Hasil ke JSON & Serialisasi Model
    if not args.no_save:
        export_payload = {
            "dataset": "Pima Indians Diabetes",
            "source": "Harvard Dataverse (doi:10.7910/DVN/HIBCFL) & Kaggle (uciml/pima-indians-diabetes-database)",
            "n_samples": n_samples,
            "n_features": n_features,
            "class_distribution": {
                "0_negative": neg_count,
                "1_positive": pos_count,
            },
            "prior": {
                "class_0": prior_0,
                "class_1": prior_1,
            },
            "parameters": params_dict,
            "test_metrics": eval_metrics,
            "cross_validation": cv_summary,
            "error_analysis": err_analysis,
        }
        output_file = Path(args.output_dir) / "hasil_eksperimen.json"
        save_experiment_results(export_payload, output_file)
        print(f"\n[OUTPUT] Artefak metrik berhasil disimpan ke: {output_file}")

        # Serialisasi model menggunakan joblib
        try:
            import joblib
            model_dir = Path(args.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file = model_dir / "gaussian_nb.joblib"
            joblib.dump(clf.model, model_file)
            print(f"[MODEL] Model terlatih berhasil diserialisasi ke: {model_file}")
        except Exception as exc:
            print(f"[WARN] Gagal menyimpan model joblib: {exc}")

        # Simpan processed dataset jika diminta (imputasi berbasis statistik data latih)
        if args.save_processed:
            proc_dir = PROJECT_ROOT / "data" / "processed"
            proc_dir.mkdir(parents=True, exist_ok=True)
            proc_path = proc_dir / "diabetes_pima_cleaned.csv"
            df_clean = pd.concat([
                pd.concat([X_train, y_train], axis=1),
                pd.concat([X_test, y_test], axis=1)
            ]).sort_index()
            df_clean.to_csv(proc_path, index=False)
            print(f"[DATA] Data bersih (tanpa kebocoran) tersimpan ke: {proc_path}")

    print("=" * 75)


if __name__ == "__main__":
    main()
