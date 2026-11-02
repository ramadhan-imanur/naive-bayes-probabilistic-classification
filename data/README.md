# Dokumentasi Dataset Publik (Tugas 3: Naïve Bayes)

Direktori ini memuat dataset publik riil yang telah diunduh dari repositori ilmiah terbuka **Harvard Dataverse** dan memiliki padanan resmi pada **Kaggle Datasets**. Dataset ini disiapkan khusus untuk eksperimen pemodelan klasifikasi **Naïve Bayes**, evaluasi kinerja (*Confusion Matrix*, Akurasi, Presisi, *Recall*, *F1-Score*), dan analisis kritis pada Tugas 3.

---

## 1. Dataset Utama: Pima Indians Diabetes (`diabetes_pima.csv`)

- **Nama Dataset**: Pima Indians Diabetes Database
- **Sumber Repositori Ilmiah**:
  - **Harvard Dataverse**: OpenXAI Benchmark (`doi:10.7910/DVN/HIBCFL`), File ID: 8550937 (Train) & 8550938 (Test).
  - **Kaggle**: [Pima Indians Diabetes Database (UCI Machine Learning)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Dimensi**: 768 baris pengamatan, 9 kolom (8 fitur prediktor + 1 target kelas).
- **Format Berkas**: Comma-Separated Values (`diabetes_pima.csv`).
- **Tipe Tugas**: Klasifikasi Biner (*Binary Classification*).
- **Distribusi Target (`Outcome`)**:
  - `0` (Negatif Diabetes): 500 sampel (65,1%)
  - `1` (Positif Diabetes): 268 sampel (34,9%)

### Definisi Atribut / Fitur:
1. `Pregnancies`: Jumlah kehamilan (numerik diskrit).
2. `Glucose`: Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa oral (mg/dL).
3. `BloodPressure`: Tekanan darah diastolik (mm Hg).
4. `SkinThickness`: Ketebalan lipatan kulit trisep (mm).
5. `Insulin`: Serum insulin 2 jam ($\mu$U/mL).
6. `BMI`: Indeks massa tubuh ($\text{berat dalam kg}/(\text{tinggi dalam m})^2$).
7. `DiabetesPedigreeFunction`: Skor riwayat diabetes silsilah keluarga (kontinu).
8. `Age`: Usia pasien (tahun).
9. `Outcome` (**Target**): Variabel kelas ($0$ jika tidak menderita diabetes, $1$ jika terindikasi menderita diabetes).

### Relevansi untuk Algoritme Naïve Bayes:
- Sangat cocok untuk **Gaussian Naïve Bayes**, di mana fungsi kemungkinan (*likelihood*) dimodelkan dengan kurva normal Gauss $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$.
- Memiliki korelasi klinis alami (misal: Glucose dengan Insulin, BMI dengan SkinThickness, Age dengan Pregnancies) yang menjadi materi analisis kritis sangat kaya untuk menguji dampak pelanggaran asumsi independensi bersyarat (*conditional independence assumption*).
- Menghadirkan konsekuensi klinis nyata pada evaluasi matriks konfusi: *False Negative* (pasien diabetes salah didiagnosis sehat) memiliki risiko fatal, sehingga memberikan justifikasi kuat perlunya memprioritaskan metrik *Recall* di samping *Precision* dan *F1-Score*.

---

## 2. Dataset Pendamping / Komparasi

### A. Breast Cancer Wisconsin (`breast_cancer_wisconsin.csv`)
- **Sumber**: Harvard Dataverse (`doi:10.7910/DVN/OJU7WZ`, File ID: 13211133) & Kaggle (`uciml/breast-cancer-wisconsin-data`).
- **Dimensi**: 569 baris, 31 kolom (30 fitur kontinu + 1 label target).
- **Distribusi Target (`target`)**: `1` (Jinak/Benign: 357 sampel), `0` (Ganas/Malignant: 212 sampel).
- **Karakteristik**: Fitur berdimensi tinggi ($d=30$) yang merepresentasikan karakteristik visual inti sel kanker payudara.

### B. Framingham Heart Disease (`heart_disease.csv`)
- **Sumber**: Harvard Dataverse OpenXAI (`doi:10.7910/DVN/HIBCFL`, File ID: 8550932 & 8550935) & Kaggle (`dileep070/heart-disease-prediction-using-logistic-regression`).
- **Dimensi**: 3.658 baris, 16 kolom (kombinasi fitur kontinu dan kategorikal).
- **Distribusi Target (`TenYearCHD`)**: `0` (Tidak berisiko PJK 10 tahun: 3.101 sampel), `1` (Berisiko PJK 10 tahun: 557 sampel).

---

## 3. Cara Penggunaan dalam Skrip Pemodelan
Dataset dapat dibaca langsung menggunakan modul `src.data` atau pustaka `pandas` dari direktori `data/raw/`:
```python
import pandas as pd
from src.data import load_dataset, clean_clinical_data

# Memuat dan membersihkan dataset Pima Indians Diabetes
df = load_dataset('data/raw/diabetes_pima.csv')
df_clean = clean_clinical_data(df)

X = df_clean.drop(columns=['Outcome'])
y = df_clean['Outcome']
```
