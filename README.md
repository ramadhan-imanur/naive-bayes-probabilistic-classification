# Klasifikasi Probabilistik Gaussian Naïve Bayes pada Dataset Klinis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Lisensi](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Data-Harvard%20Dataverse-orange.svg)](https://doi.org/10.7910/DVN/HIBCFL)

Repositori ini memuat penurunan matematis formal, implementasi komputasi modular, evaluasi kinerja empiris, dan analisis risiko diagnostik klinis pengklasifikasi Gaussian Naïve Bayes pada dataset *Pima Indians Diabetes* (*Harvard Dataverse* & *Kaggle*), disertai pengujian unit otomatis dan modul simulasi diskrit.

Proyek ini disusun sebagai pemenuhan **Tugas 3 (Project-Based Learning: Pemodelan Naïve Bayes)** pada mata kuliah **Kecerdasan Buatan**, Program Studi S-1 Matematika, Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Sebelas Maret.

---

## Daftar Isi

1. [Ringkasan Eksekutif](#ringkasan-eksekutif)
2. [Formulasi Matematis Deduktif](#formulasi-matematis-deduktif)
   - [Peluang Posterior dan Teorema Bayes](#peluang-posterior-dan-teorema-bayes)
   - [Asumsi Independensi Bersyarat](#asumsi-independensi-bersyarat)
   - [Aturan Keputusan MAP Log-Likelihood](#aturan-keputusan-map-log-likelihood)
   - [Fungsi Kemungkinan Kontinu Gaussian](#fungsi-kemungkinan-kontinu-gaussian)
3. [Dataset dan Pra-pemrosesan Data](#dataset-dan-pra-pemrosesan-data)
   - [Deskripsi Data dan Konteks Biologis](#deskripsi-data-dan-konteks-biologis)
   - [Imputasi Berbasis Pertimbangan Fisiologis](#imputasi-berbasis-pertimbangan-fisiologis)
4. [Hasil Eksperimen dan Evaluasi](#hasil-eksperimen-dan-evaluasi)
   - [Distribusi Parameter yang Dipelajari](#distribusi-parameter-yang-dipelajari)
   - [Evaluasi Data Uji (Partisi Terstratifikasi 80/20)](#evaluasi-data-uji-partisi-terstratifikasi-8020)
   - [Stabilitas Kinerja (Validasi Silang Terstratifikasi 5-Lipat)](#stabilitas-kinerja-validasi-silang-terstratifikasi-5-lipat)
5. [Analisis Kritis dan Risiko Diagnostik Klinis](#analisis-kritis-dan-risiko-diagnostik-klinis)
   - [Distorsi Kalibrasi vs Ketangguhan Batas Keputusan](#distorsi-kalibrasi-vs-ketangguhan-batas-keputusan)
   - [Efek Reduksi Variansi pada Sampel Terbatas](#efek-reduksi-variansi-pada-sampel-terbatas)
   - [Trade-off Penapisan Medis: Positif Palsu vs Negatif Palsu](#trade-off-penapisan-medis-positif-palsu-vs-negatif-palsu)
6. [Struktur Direktori Repositori](#struktur-direktori-repositori)
7. [Panduan Instalasi dan Eksekusi](#panduan-instalasi-dan-eksekusi)
   - [Prasyarat Sistem](#prasyarat-sistem)
   - [Instalasi Dependensi](#instalasi-dependensi)
   - [Menjalankan Pipeline Eksperimen](#menjalankan-pipeline-eksperimen)
   - [Menjalankan Pengujian Unit Otomatis](#menjalankan-pengujian-unit-otomatis)
8. [Daftar Rujukan Ilmiah](#daftar-rujukan-ilmiah)
9. [Lisensi](#lisensi)

---

## Ringkasan Eksekutif

Pengklasifikasi Naïve Bayes beroperasi di bawah asumsi penyederhanaan bahwa setiap fitur prediktor bersifat independen secara bersyarat (*conditionally independent*) terhadap fitur lainnya jika diberikan label kelas. Pada data rekam medis dan diagnostik fisiologis, asumsi ini secara empiris dilanggar karena mekanisme metabolisme manusia saling terhubung erat (misalnya hubungan antara kadar glukosa plasma dengan sekresi insulin, atau indeks massa tubuh dengan ketebalan jaringan adiposa).

Kendati terdapat pertentangan teoretis tersebut, model Gaussian Naïve Bayes secara konsisten menunjukkan kinerja diskriminasi yang tangguh dan efisien dalam penapisan (*screening*) medis. Repositori ini menyajikan implementasi *end-to-end* yang mencakup:

* **Penurunan matematis analitis** aturan keputusan *Maximum A Posteriori* (MAP) berbasis transformasi log-kebolehjadian (*log-likelihood*) untuk menjamin stabilitas numerik terhadap *arithmetic underflow*.
* **Pra-pemrosesan data berwawasan klinis** melalui imputasi nilai median terarah untuk menangani pencatatan nilai nol yang tidak masuk akal secara fisiologis (*physiologically non-viable zeros*).
* **Evaluasi kinerja komprehensif** menggunakan partisi terstratifikasi 80/20 dan validasi silang 5-lipat (*5-fold stratified cross-validation*), mencakup matriks konfusi eksak, akurasi, presisi, sensitivitas (*recall*), dan skor-F1.
* **Analisis galat diagnostik (*error analysis*)** yang membedah trade-off antara *False Positive* (alarm keliru) dan *False Negative* (kegagalan deteksi pasien sakit berisiko komplikasi) serta menjustifikasi kestabilan batas keputusan linier/kuadratik model.

---

## Formulasi Matematis Deduktif

### Peluang Posterior dan Teorema Bayes
Diberikan suatu vektor observasi klinis $d$-dimensi $\mathbf{x} = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d$ dan variabel kelas diskrit $c_k \in \mathcal{C} = \{c_1, c_2, \dots, c_K\}$, peluang posterior bahwa observasi $\mathbf{x}$ merupakan anggota kelas $c_k$ diformulasikan berdasarkan Teorema Bayes:

$$P(c_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c_k) P(c_k)}{P(\mathbf{x})} = \frac{P(\mathbf{x} \mid c_k) P(c_k)}{\sum_{j=1}^K P(c_j) P(\mathbf{x} \mid c_j)}$$

di mana:
* $P(c_k \mid \mathbf{x})$ merepresentasikan probabilitas posterior (*posterior probability*).
* $P(\mathbf{x} \mid c_k)$ merepresentasikan fungsi kemungkinan (*likelihood*).
* $P(c_k)$ adalah peluang prior kelas (*prior probability*).
* $P(\mathbf{x})$ adalah probabilitas bukti marginal (*marginal evidence*) yang berfungsi sebagai konstanta normalisasi.

### Asumsi Independensi Bersyarat
Estimasi fungsi kemungkinan gabungan penuh $P(\mathbf{x} \mid c_k)$ pada ruang dimensi tinggi membutuhkan sampel data yang luar biasa besar untuk mengestimasi matriks kovariansi secara akurat. Pengklasifikasi Naïve Bayes menyederhanakan estimasi ini melalui asumsi bahwa setiap fitur $x_j$ bersifat independen secara bersyarat terhadap fitur lainnya jika diberikan kelas $c_k$:

$$P(\mathbf{x} \mid c_k) = \prod_{j=1}^d P(x_j \mid c_k)$$

### Aturan Keputusan MAP Log-Likelihood
Klasifikasi dilakukan dengan memilih kelas $c_k$ yang memaksimalkan peluang posterior, yang dikenal sebagai aturan keputusan *Maximum A Posteriori* (MAP):

$$\hat{y} = \arg\max_{c_k \in \mathcal{C}} P(c_k \mid \mathbf{x}) = \arg\max_{c_k \in \mathcal{C}} \left[ P(c_k) \prod_{j=1}^d P(x_j \mid c_k) \right]$$

Karena perkalian berulang dari sejumlah nilai probabilitas yang bernilai kecil ($P(x_j \mid c_k) \ll 1$) menyebabkan galat pembulatan komputasi (*floating-point arithmetic underflow*), fungsi objektif ditransformasikan menggunakan logaritma natural ($\ln$). Karena fungsi logaritma bersifat monotonik naik tegas (*strictly monotonically increasing*), transformasi ini mempertahankan urutan kelas optimal tanpa mengubah hasil klasifikasi:

$$\hat{y} = \arg\max_{c_k \in \mathcal{C}} \left[ \ln P(c_k) + \sum_{j=1}^d \ln P(x_j \mid c_k) \right]$$

### Fungsi Kemungkinan Kontinu Gaussian
Untuk fitur berskala kontinu, densitas peluang bersyarat kelas dimodelkan mengikuti distribusi normal (Gaussian) univariat dengan parameter rata-rata kelas $\mu_{kj}$ dan variansi $\sigma_{kj}^2$:

$$P(x_j \mid c_k) = \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left( -\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2} \right)$$

Substitusi persamaan densitas Gaussian ke dalam bentuk log-likelihood menghasilkan persamaan analitis eksplisit:

$$\ln P(x_j \mid c_k) = -\frac{1}{2} \ln(2\pi) - \frac{1}{2} \ln(\sigma_{kj}^2) - \frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}$$

Pendekatan parametrik ini hanya memerlukan estimasi $2d$ parameter per kelas (vektor rata-rata dan vektor variansi untuk setiap fitur), mengeliminasi kebutuhan estimasi $d(d+1)/2$ kovariansi dari matriks kovariansi penuh. Hal ini memberikan keuntungan krusial berupa ketahanan terhadap fenomena *curse of dimensionality* dan mencegah *overfitting* saat ukuran data terbatas.

---

## Dataset dan Pra-pemrosesan Data

### Deskripsi Data dan Konteks Biologis
Eksperimen utama menggunakan **Pima Indians Diabetes Database** yang bersumber dari repositori ilmiah terbuka *Harvard Dataverse* ([DOI: 10.7910/DVN/HIBCFL](https://doi.org/10.7910/DVN/HIBCFL)) dan *Kaggle*.

* **Jumlah Pengamatan**: 768 baris rekam medis
* **Jumlah Fitur Prediktor**: 8 variabel klinis kuantitatif
* **Variabel Target**: `Outcome` biner (0 = Negatif [500 sampel, 65,1%], 1 = Positif [268 sampel, 34,9%])

| Fitur | Satuan / Format | Peran & Konteks Fisiologis Klinis |
| :--- | :--- | :--- |
| `Pregnancies` | Hitungan diskrit | Beban metabolik gestasional (resistensi insulin selama kehamilan) |
| `Glucose` | mg/dL | Konsentrasi glukosa plasma 2 jam pada tes toleransi glukosa oral (OGTT); penanda utama diabetes |
| `BloodPressure` | mm Hg | Tekanan darah diastolik; indikator kesehatan vaskular dan kardiovaskular |
| `SkinThickness` | mm | Ketebalan lipatan kulit trisep; representasi distribusi lemak subkutan perifer |
| `Insulin` | $\mu\text{U/mL}$ | Serum insulin 2 jam; indikator respon sekresi endogen terhadap beban glukosa |
| `BMI` | $\text{kg/m}^2$ | Indeks Massa Tubuh (rasio berat terhadap kuadrat tinggi badan); indikator obesitas sentral |
| `DiabetesPedigreeFunction` | Skor kontinu | Fungsi silsilah keluarga; estimasi risiko genetik bawaan |
| `Age` | Tahun | Usia pasien; degradasi fungsi sel beta pankreas akibat proses penuaan |

### Imputasi Berbasis Pertimbangan Fisiologis
Pada data mentah, ditemukan nilai $0$ pada fitur `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI`. Secara klinis, nilai nol pada variabel tersebut mustahil terjadi pada manusia yang hidup (*physiologically non-viable*). Nilai nol ini merupakan representasi data hilang (*missing values*) akibat kegagalan pencatatan atau tes yang tidak dilakukan.

Pipeline pra-pemrosesan (`src/data.py`) menerapkan strategi imputasi terarah:
1. Mengidentifikasi dan mengubah nilai $0$ pada kelima fitur tersebut menjadi nilai penanda *NaN* (*Not a Number*).
2. Menghitung nilai median dari entri non-nol yang valid pada masing-masing fitur.
3. Melakukan imputasi nilai median tersebut untuk menggantikan nilai yang hilang, menjaga distribusi data tetap kokoh terhadap pengaruh *outliers* tanpa merusak integritas statistik data latih.

---

## Hasil Eksperimen dan Evaluasi

### Distribusi Parameter yang Dipelajari
Estimasi parameter pada subset data latih menghasilkan nilai rata-rata dan deviasi standar ($\mu \pm \sigma$) yang membedakan profil metabolik antara pasien sehat dan terindikasi diabetes:

| Fitur Klinis | Kelas 0 (Non-Diabetes) $\mu \pm \sigma$ | Kelas 1 (Diabetes) $\mu \pm \sigma$ |
| :--- | :---: | :---: |
| `Glucose` (mg/dL) | $111,85 \pm 24,38$ | $141,32 \pm 29,88$ |
| `BMI` ($\text{kg/m}^2$) | $30,85 \pm 6,62$ | $35,27 \pm 6,48$ |
| `Age` (tahun) | $31,42 \pm 11,96$ | $37,40 \pm 11,08$ |
| `Insulin` ($\mu\text{U/mL}$) | $127,08 \pm 66,38$ | $156,43 \pm 87,45$ |
| `BloodPressure` (mm Hg) | $71,13 \pm 11,82$ | $75,48 \pm 12,17$ |
| `SkinThickness` (mm) | $27,50 \pm 8,59$ | $31,52 \pm 8,76$ |

### Evaluasi Data Uji (Partisi Terstratifikasi 80/20)
Data uji independen (*held-out test set*) berukuran 154 data (100 negatif dan 54 positif) menghasilkan matriks konfusi:

```text
                        Prediksi Negatif (0)    Prediksi Positif (1)
Aktual Negatif (0)            TN = 84                 FP = 16
Aktual Positif (1)            FN = 22                 TP = 32
```

Metrik evaluasi kinerja klasifikasi:
* **Akurasi**: $75,32\%$
* **Presisi**: $66,67\%$
* **Sensitivitas / Recall**: $59,26\%$
* **Skor-F1**: $62,75\%$

### Stabilitas Kinerja (Validasi Silang Terstratifikasi 5-Lipat)
Validasi silang berstrata 5-lipat (*5-fold stratified cross-validation*) pada keseluruhan dataset memastikan generalisasi model bebas dari bias partisi tunggal:

* **Akurasi Rata-rata**: $74,48\% \pm 3,13\%$
* **Presisi Rata-rata**: $64,75\% \pm 5,12\%$
* **Sensitivitas / Recall Rata-rata**: $59,37\% \pm 6,44\%$
* **Skor-F1 Rata-rata**: $61,80\% \pm 5,11\%$

---

## Analisis Kritis dan Risiko Diagnostik Klinis

### Distorsi Kalibrasi vs Ketangguhan Batas Keputusan
Korelasi fisiologis alami antar-fitur (seperti `BMI` dengan `SkinThickness`, atau `Glucose` dengan `Insulin`) melanggar asumsi independensi bersyarat. Dampak matematis dari pelanggaran ini adalah *overcounting of evidence*, yang menyebabkan estimasi peluang posterior terdistorsi mendekati nilai ekstrem $0$ atau $1$ (probabilitas menjadi tidak terkalibrasi secara akurat).

Namun, ketidakakuratan nilai probabilitas posterior absolut tidak serta-merta menurunkan akurasi klasifikasi. Selama probabilitas posterior kelas yang benar tetap berada di atas probabilitas kelas saingannya, operasi $\arg\max$ akan tetap menghasilkan prediksi kelas yang tepat. Batas keputusan (*decision boundary*) yang dibentuk model tetap berada di wilayah optimal pemisah antarkelas.

### Efek Reduksi Variansi pada Sampel Terbatas
Jika estimasi dilakukan menggunakan model generatif penuh dengan matriks kovariansi kovarian bebas, maka diperlukan perhitungan $\frac{8 \times 9}{2} = 36$ nilai kovariansi per kelas (total 72 parameter kovariansi untuk 2 kelas). Dengan ukuran sampel positif yang hanya berjumlah 268 observasi, matriks kovariansi empiris menjadi sangat bising (*noisy*) dan rentan terhadap *overfitting*.

Dengan membatasi matriks kovariansi pada diagonalnya (asumsi independensi bersyarat), model hanya mengestimasi 8 parameter variansi per kelas. Pengurangan derajat kebebasan (*degrees of freedom*) ini secara signifikan memangkas variansi estimasi model (*variance reduction*), memberikan trade-off bias-variansi yang menguntungkan stabilitas prediksi pada data uji.

### Trade-off Penapisan Medis: Positif Palsu vs Negatif Palsu
Pada pengujian data uji, terjadi dua bentuk galat klasifikasi dengan implikasi klinis yang asimetris:
1. **Positif Palsu (*False Positive*, 16 kasus)**: Pasien sehat salah didiagnosis mengidap diabetes (*false alarm*). Pasien mengalami kecemasan psikologis dan diarahkan pada pemeriksaan diagnostik konfirmasi lanjutan (misalnya tes HbA1c ulang), namun tidak membahayakan keselamatan jiwa secara langsung.
2. **Negatif Palsu (*False Negative*, 22 kasus)**: Pasien yang secara patologis menderita diabetes salah diprediksi sebagai sehat. Pasien kehilangan jendela intervensi terapi dini, membiarkan hiperglikemia kronis berkembang menjadi komplikasi mikrovaskular dan makrovaskular (neuropati, nefropati, retinopati, dan penyakit jantung koroner).

Oleh karena itu, dalam konteks sistem penapisan primer (*mass screening*), ambang batas probabilitas keputusan $\tau = 0{,}5$ dapat digeser ke nilai yang lebih rendah (misalnya $\tau = 0{,}35$) sepanjang kurva *Receiver Operating Characteristic* (ROC) guna mendongkrak metrik sensitivitas (*recall*), meminimalkan risiko pasien diabetes yang luput dari deteksi medis.

---

## Struktur Direktori Repositori

Arsitektur repositori disusun secara modular, terisolasi, dan mengikuti standar kode bersih (*clean architecture*):

```text
code/
├── config/
│   └── params.yaml                    # Konfigurasi hyperparameter, jalur data & random seed
├── data/
│   ├── raw/                           # Dataset mentah (diabetes_pima.csv, breast_cancer, heart_disease)
│   ├── processed/                     # Dataset bersih hasil pra-pemrosesan dan imputasi
│   │   └── diabetes_pima_cleaned.csv
│   └── README.md                      # Dokumentasi komprehensif dataset klinis & DOI
├── models/                            # Artefak model & serializer bobot
│   └── gaussian_nb.joblib             # Model Gaussian Naïve Bayes terlatih
├── notebooks/                         # Notebook eksplorasi & evaluasi visual interaktif
│   └── 01_evaluasi_model_naive_bayes.ipynb # Analisis evaluasi lengkap & visualisasi kurva ROC/PR/Likelihood
├── outputs/
│   ├── figures/                       # Grafik visualisasi resolusi tinggi (300 DPI)
│   │   ├── 01_distribusi_likelihood_fitur.png
│   │   ├── 02_korelasi_fitur_independensi.png
│   │   ├── 03_matriks_konfusi.png
│   │   ├── 04_kurva_roc_dan_pr.png
│   │   ├── 05_tuning_ambang_batas_keputusan.png
│   │   └── 06_validasi_silang_5fold.png
│   └── metrics/
│       └── hasil_eksperimen.json      # Serialisasi metrik evaluasi & parameter model
├── scripts/
│   ├── generate_evaluation_notebook.py# Otomasi rendering notebook evaluasi & grafis
│   └── simulate_categorical_nb.py     # Simulasi Naïve Bayes kategorikal & Laplace smoothing
├── src/
│   ├── __init__.py                    # Inisialisasi paket Python modular
│   ├── data.py                        # Pipeline pemuatan, imputasi biologis, dan partisi data
│   ├── models.py                      # Abstraksi parametrik Gaussian Naïve Bayes
│   ├── evaluation.py                  # Evaluasi metrik, cross-validation, dan analisis galat
│   └── visualization.py               # Pembuatan heatmap matriks konfusi, kurva ROC/PR, & KDE
├── tests/
│   ├── __init__.py
│   ├── test_data.py                   # Pengujian unit pra-pemrosesan data
│   ├── test_models.py                 # Pengujian unit model Gaussian Naïve Bayes
│   ├── test_evaluation.py             # Pengujian unit metrik evaluasi dan analisis galat
│   └── test_visualization.py          # Pengujian unit visualisasi grafis
├── LICENSE                            # Lisensi sumber terbuka MIT
├── pyproject.toml                     # Konfigurasi build sistem & dependensi proyek
├── README.md                          # Dokumentasi utama proyek
├── requirements.txt                   # Spesifikasi pustaka dependensi lingkungan kerja
└── run_experiment.py                  # Skrip orkestrasi pipeline eksekusi end-to-end
```

---

## Panduan Instalasi dan Eksekusi

### Prasyarat Sistem
* Python 3.10 atau versi yang lebih tinggi.
* Manajer paket `pip` dan pustaka `virtualenv` (disarankan).

### Instalasi Dependensi
Buka terminal pada direktori `Tugas 3/code/` dan siapkan lingkungan kerja virtual:

```bash
# 1. Pindah ke direktori kode proyek
cd "/media/ramadhan/0C6A-1ABD/University/Kecerdasan Buatan/Tugas 3/code"

# 2. Buat lingkungan virtual Python (opsional namun disarankan)
python3 -m venv venv
source venv/bin/activate

# 3. Pasang dependensi yang diperlukan
pip install -r requirements.txt
```

### Menjalankan Pipeline Eksperimen
Jalankan skrip orkestrasi utama untuk memuat data, mengeksekusi pra-pemrosesan, melatih model, mengevaluasi data uji, menjalankan 5-fold cross validation, dan menyimpan metrik ke `outputs/metrics/hasil_eksperimen.json`:

```bash
# Menjalankan eksperimen end-to-end secara otomatis
python3 run_experiment.py
```

Skrip ini juga mendukung berbagai argumen baris perintah (*command-line interface*):

```bash
# Menjalankan dengan konfigurasi parameter khusus
python3 run_experiment.py \
    --data-path data/raw/diabetes_pima.csv \
    --test-size 0.20 \
    --cv-folds 5 \
    --random-state 42 \
    --output-dir outputs/metrics
```

### Menjalankan Simulasi Diskrit (Laplace Smoothing)
Untuk meninjau demonstrasi komputasi manual langkah-demi-langkah pengklasifikasi Naïve Bayes kategorikal beserta penyelesaian masalah frekuensi nol (*zero-frequency problem*) menggunakan *Laplace smoothing*:

```bash
python3 scripts/simulate_categorical_nb.py
```

### Menjalankan Pengujian Unit Otomatis
Seluruh fungsionalitas modul pada direktori `src/` dilindungi oleh pengujian unit otomatis berbasis `pytest`:

```bash
pytest -v
```

---

## Daftar Rujukan Ilmiah

1. **OpenXAI Benchmark**: Agarwal, C., Krishna, S., Saxena, E., et al. (2022). *OpenXAI: Towards a Comprehensive Evaluation Benchmark for Explainable AI*. Harvard Dataverse, DOI: [10.7910/DVN/HIBCFL](https://doi.org/10.7910/DVN/HIBCFL).
2. **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)**: *Pima Indians Diabetes Database*. UCI Machine Learning Repository & Kaggle.
3. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer, New York.
4. **Mitchell, T. M.** (1997). *Machine Learning*. McGraw-Hill, New York.
5. **Hand, D. J., & Yu, K.** (2001). *Idiot's Bayes—Not So Stupid After All?*. International Statistical Review, 69(3), 385–398.
6. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd Edition, Springer, New York.

---

## Lisensi

Proyek ini didistribusikan di bawah lisensi terbuka [MIT License](LICENSE). Bebas digunakan, dimodifikasi, dan didistribusikan untuk keperluan akademis dan penelitian dengan tetap menyertakan atribusi orisinal.
