#!/usr/bin/env python3
"""
Eksperimen Komputasi Pemodelan Naïve Bayes Berbasis Dataset Publik (Pima Indians Diabetes)
Mata Kuliah: Kecerdasan Buatan (Tugas 3: Project-Based Learning)
Kelompok 3 - S-1 Matematika FMIPA UNS

Modul ini menjalankan:
1. Prapemrosesan data (imputasi nilai nol biologis tidak valid, stratifikasi partisi data 80:20).
2. Pemodelan Gaussian Naïve Bayes dan estimasi parameter empiris (Prior, Mean, Variansi).
3. Evaluasi kinerja pengujian: Confusion Matrix, Akurasi, Presisi, Recall, F1-Score.
4. Validasi kestabilan model melalui 5-Fold Stratified Cross-Validation.
5. Analisis galat (Error Analysis) pada sampel misklasifikasi (False Positive & False Negative).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'diabetes_pima.csv')
    
    print("=" * 70)
    print(" EKSPERIMEN GAUSSIAN NAÏVE BAYES: PIMA INDIANS DIABETES DATASET")
    print(" Repositori: Harvard Dataverse (doi:10.7910/DVN/HIBCFL) & Kaggle")
    print("=" * 70)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset tidak ditemukan pada {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"\n1. PROFIL DATASET:")
    print(f"   Jumlah Sampel (n) : {df.shape[0]}")
    print(f"   Jumlah Fitur (d)  : {df.shape[1] - 1}")
    print(f"   Distribusi Kelas  : 0 (Negatif) = {(df['Outcome'] == 0).sum()} sampel ({(df['Outcome'] == 0).mean()*100:.1f}%), "
          f"1 (Positif) = {(df['Outcome'] == 1).sum()} sampel ({(df['Outcome'] == 1).mean()*100:.1f}%)")

    # 2. Pra-pemrosesan Data
    # Pada data medis Pima, nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI adalah missing values secara fisiologis
    zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_clean = df.copy()
    for col in zero_as_missing:
        median_val = df_clean[df_clean[col] > 0][col].median()
        df_clean[col] = df_clean[col].replace(0, median_val)
        
    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']
    
    # 3. Partisi Data (80% Train, 20% Test) Terstratifikasi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n2. PARTISI DATA (80:20 STRATIFIKASI):")
    print(f"   Data Latih (Train) : {X_train.shape[0]} sampel")
    print(f"   Data Uji (Test)    : {X_test.shape[0]} sampel")

    # 4. Pelatihan Model Gaussian Naïve Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    print(f"\n3. PARAMETER PELUANG HASIL PELATIHAN:")
    print(f"   Prior P(Outcome = 0) : {gnb.class_prior_[0]:.4f}")
    print(f"   Prior P(Outcome = 1) : {gnb.class_prior_[1]:.4f}")
    print("\n   Parameter Likelihood Normal Gauss per Fitur:")
    print(f"   {'Fitur':<25} | {'Mean (Class 0)':<15} | {'Mean (Class 1)':<15} | {'Var (Class 0)':<15} | {'Var (Class 1)':<15}")
    print("   " + "-" * 90)
    for i, col in enumerate(X.columns):
        m0 = gnb.theta_[0][i]
        m1 = gnb.theta_[1][i]
        v0 = gnb.var_[0][i]
        v1 = gnb.var_[1][i]
        print(f"   {col:<25} | {m0:<15.2f} | {m1:<15.2f} | {v0:<15.2f} | {v1:<15.2f}")

    # 5. Prediksi dan Evaluasi pada Data Uji
    y_pred = gnb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n4. HASIL EVALUASI DATA UJI (TEST SET):")
    print(f"   Matriks Konfusi (Confusion Matrix):")
    print(f"                     Prediksi 0 (Negatif)   Prediksi 1 (Positif)")
    print(f"   Aktual 0 (Negatif)      TN = {tn:<18} FP = {fp}")
    print(f"   Aktual 1 (Positif)      FN = {fn:<18} TP = {tp}")
    print(f"\n   Metrik Kinerja:")
    print(f"   - Akurasi (Accuracy)  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   - Presisi (Precision) : {prec:.4f} ({prec*100:.2f}%)")
    print(f"   - Recall (Sensitivitas): {rec:.4f} ({rec*100:.2f}%)")
    print(f"   - F1-Score            : {f1:.4f} ({f1*100:.2f}%)")

    # 6. Validasi Silang (5-Fold Stratified Cross-Validation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(gnb, X, y, cv=skf, scoring=scoring)
    
    print(f"\n5. VALIDASI SILANG 5-FOLD STRATIFIKASI (KESTABILAN MODEL):")
    print(f"   - Akurasi Rata-rata   : {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
    print(f"   - Presisi Rata-rata  : {cv_results['test_precision'].mean():.4f} +/- {cv_results['test_precision'].std():.4f}")
    print(f"   - Recall Rata-rata    : {cv_results['test_recall'].mean():.4f} +/- {cv_results['test_recall'].std():.4f}")
    print(f"   - F1-Score Rata-rata  : {cv_results['test_f1'].mean():.4f} +/- {cv_results['test_f1'].std():.4f}")

    # 7. Analisis Galat (Error Analysis)
    fp_mask = (y_test.values == 0) & (y_pred == 1)
    fn_mask = (y_test.values == 1) & (y_pred == 0)
    
    print(f"\n6. ANALISIS GALAT (MISKLASIFIKASI):")
    print(f"   - Total Sampel Salah Prediksi : {fp + fn} dari {len(y_test)} sampel (Tingkat Galat: {(fp+fn)/len(y_test)*100:.2f}%)")
    print(f"   - False Positive (FP)         : {fp} sampel (Pasien sehat salah didiagnosis diabetes)")
    print(f"   - False Negative (FN)         : {fn} sampel (Pasien diabetes gagal terdeteksi)")
    print(f"   Catatan Kritis: Pada diagnosis medis, meminimalkan FN sangat krusial agar penderita segera memperoleh perawatan medis.")
    print("=" * 70)

if __name__ == "__main__":
    main()
