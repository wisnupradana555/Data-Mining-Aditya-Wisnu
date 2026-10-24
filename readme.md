# Portofolio GitHub: 3 Proyek Machine Learning BPS Jawa Tengah (2025)

Koleksi **3 Proyek Machine Learning** berbasis data resmi **Badan Pusat Statistik (BPS) Provinsi Jawa Tengah Tahun 2025** (mencakup seluruh **35 Kabupaten/Kota**) yang disusun mengikuti kurikulum **IBM Machine Learning Professional Certificate**:

1. **Project 1 — Exploratory Data Analysis for Machine Learning (`01_eda_for_machine_learning.py`)**
2. **Project 2 — Supervised Machine Learning: Regression (`02_supervised_ml_regression.py`)**
3. **Project 3 — Supervised Machine Learning: Classification (`03_supervised_ml_classification.py`)**

## Opsi Struktur Folder di VS Code & GitHub

**1 Repositori Portofolio Terpadu** (dengan 3 *script* terpisah) atau membaginya menjadi **3 Folder/Repositori GitHub Mandiri** seperti berikut:

```text
ml-portfolio-bps-jateng/
├── bps_jawa_tengah_2025.csv                  # Dataset utama 35 Kabupaten/Kota BPS Jawa Tengah 2025
├── 01_eda_for_machine_learning.py            # Project 1: Course 1 - Exploratory Data Analysis for ML
├── 02_supervised_ml_regression.py            # Project 2: Course 2 - Supervised ML: Regression
├── 03_supervised_ml_classification.py        # Project 3: Course 3 - Supervised ML: Classification
└── README.md                                 # Dokumentasi portofolio lengkap ketiga proyek
```

*(Catatan: Ketiga script Python telah dirancang cerdas untuk mencari file `bps_jawa_tengah_2025.csv` baik di folder yang sama, di dalam subfolder `data/`, maupun satu level folder di atasnya `../bps_jawa_tengah_2025.csv` jika memisahkannya ke dalam 3 sub-folder berbeda di VS Code).*

## Sumber Data Resmi BPS Jawa Tengah (2025)

Dataset `bps_jawa_tengah_2025.csv` terdiri dari **35 baris (29 Kabupaten dan 6 Kota)** serta **12 variabel utama** dari publikasi resmi BPS:

| Nama Kolom | Satuan | Deskripsi & Tautan Tabel Statistik Resmi BPS |
| :--- | :--- | :--- |
| `kode_bps` | Kode | Kode wilayah administrasi BPS (`3301` s.d. `3376`) |
| `nama_wilayah` | Teks | Nama Kabupaten/Kota di Provinsi Jawa Tengah |
| `tipe_wilayah` | Kategori | Status administrasi wilayah (`Kabupaten` atau `Kota`) |
| `ipm_2025` | Indeks | [Indeks Pembangunan Manusia (IPM) Menurut Kabupaten/Kota di Provinsi Jawa Tengah 2025](https://jateng.bps.go.id/en/statistics-table/3/V25GaFNHaExaMnhITm1sWmRrUlJZelJzYUc1SGR6MDkjMw==/indeks-pembangunan-manusia-menurut-kabupaten-kota-di-provinsi-jawa-tengah--2019.html) |
| `persentase_penduduk_miskin_2025` | Persen (%) | [Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota 2025](https://www.bps.go.id/id/statistics-table/2/NjIxIzI=/persentase-penduduk-miskin-menurut-kabupaten-kota.html) |
| `pdrb_adhb_2025_triliun_rp` | Triliun Rp | [Nilai PDRB Atas Dasar Harga Berlaku (ADHB) Menurut Kabupaten/Kota se-Jawa Tengah 2025](https://jateng.bps.go.id/id/statistics-table/2/Mzc0MSMy/nilai-kontribusi-dan-peringkat-pdrb-adhb-menurut-kabupaten-kota-se-jawa-tengah.html) |
| `kontribusi_pdrb_jateng_persen` | Persen (%) | [Kontribusi PDRB Kabupaten/Kota terhadap Total Provinsi Jawa Tengah 2025](https://jateng.bps.go.id/id/statistics-table/2/Mzc0MSMy/nilai-kontribusi-dan-peringkat-pdrb-adhb-menurut-kabupaten-kota-se-jawa-tengah.html) |
| `peringkat_pdrb_2025` | Peringkat | Peringkat nilai PDRB ADHB se-Jawa Tengah tahun 2025 (1 s.d. 35) |
| `hls_laki_laki_2025` | Tahun | [Angka Harapan Lama Sekolah (HLS) Laki-laki Menurut Kabupaten/Kota 2025](https://www.bps.go.id/id/statistics-table/2/NDU3IzI=/angka-harapan-lama-sekolah--hls--menurut-jenis-kelamin--tahun-.html) |
| `hls_perempuan_2025` | Tahun | [Angka Harapan Lama Sekolah (HLS) Perempuan Menurut Kabupaten/Kota 2025](https://www.bps.go.id/id/statistics-table/2/NDU3IzI=/angka-harapan-lama-sekolah--hls--menurut-jenis-kelamin--tahun-.html) |
| `uhh_laki_laki_2025` | Tahun | [Umur Harapan Hidup Saat Lahir (UHH) Laki-laki Hasil SP2020 LF 2025](https://www.bps.go.id/id/statistics-table/2/MjI3MCMy/umur-harapan-hidup-saat-lahir-menurut-jenis-kelamin--menggunakan-uhh-hasilsp2020lf-.html) |
| `uhh_perempuan_2025` | Tahun | [Umur Harapan Hidup Saat Lahir (UHH) Perempuan Hasil SP2020 LF 2025](https://www.bps.go.id/id/statistics-table/2/MjI3MCMy/umur-harapan-hidup-saat-lahir-menurut-jenis-kelamin--menggunakan-uhh-hasilsp2020lf-.html) |

## Project 1: Exploratory Data Analysis for Machine Learning (`01_eda_for_machine_learning.py`)

Proyek pertama berfokus pada penyiapan data, analisis statistik, rekayasa fitur, dan uji hipotesis sebelum pemodelan *Machine Learning*:

1. **Data Cleaning & Outlier Detection (Metode IQR)**:
   - Memverifikasi `0` *missing values* dan `0` baris duplikat pada seluruh 35 Kabupaten/Kota.
   - Mendeteksi *outlier* pada 7 variabel numerik menggunakan *Interquartile Range (IQR)*, termasuk 3 wilayah pusat ekonomi dengan PDRB di atas ambang $Q_3 + 1,5 \times \text{IQR}$ (`Rp 104,55 Triliun`), yaitu **Kota Semarang** (`Rp 288,05 Triliun`), **Kabupaten Cilacap** (`Rp 142,71 Triliun`), dan **Kabupaten Kudus** (`Rp 132,32 Triliun`).
2. **Analisis Skewness & Transformasi Fitur**:
   - Membandingkan kecondongan (*Fisher-Pearson skewness*) distribusi PDRB mentah (`+3,2432`, sangat *right-skewed*), transformasi akar kuadrat $\sqrt{\text{PDRB}}$ (`+1,9886`), dan transformasi logaritma natural $\ln(\text{PDRB})$ (`+0,6491`, mendekati simetris/normal).
3. **Feature Engineering & Feature Scaling**:
   - Membentuk 7 fitur baru: `hls_rata_rata`, `uhh_rata_rata`, `gap_hls_gender`, `gap_uhh_gender`, `rasio_pendidikan_kesehatan`, `log_pdrb`, dan `is_kota`.
   - Membandingkan normalisasi `StandardScaler` ($\mu = 0, \sigma = 1$) terhadap `MinMaxScaler` ($[0, 1]$).
4. **Statistical Hypothesis Testing (Welch's Two-Sample T-Test)**:
   - Menguji perbedaan rata-rata antara **6 Kota** vs. **29 Kabupaten** di Jawa Tengah:
     - **IPM**: Rata-rata Kota (`82,60`) vs. Kabupaten (`74,20`) menghasilkan $t = +5,1060$ ($p < 0,0001$, signifikan secara statistik).
     - **Persentase Kemiskinan**: Rata-rata Kota (`5,80%`) vs. Kabupaten (`9,89%`) menghasilkan $t = -5,1499$ ($p < 0,0001$, signifikan secara statistik).

## Project 2: Supervised Machine Learning: Regression (`02_supervised_ml_regression.py`)

Proyek kedua mengimplementasikan dan membandingkan **5 algoritma regresi** (termasuk regularisasi L1, L2, dan *ElasticNet*) untuk memprediksi **Indeks Pembangunan Manusia (`ipm_2025`)** serta mendiagnosis anomali kemiskinan daerah:

### 1. Tabel Perbandingan 5 Model Regresi (Target: `ipm_2025`)

| Nama Model Regresi | $R^2$ Score (Test) | MAE (Test) | RMSE (Test) | LOOCV $R^2$ (35 Wilayah) | LOOCV MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Linear Regression (OLS)** | 0,9845 | **0,4342** | 0,5155 | 0,9439 | 0,8370 |
| **2. Ridge Regression (L2, $\alpha=1,0$)** | 0,9808 | 0,4987 | 0,5746 | 0,9464 | **0,8359** |
| **3. Lasso Regression (L1, $\alpha=0,08$)** | **0,9853** | 0,4863 | **0,5032** | 0,9465 | 0,8373 |
| **4. ElasticNet (L1+L2, $\alpha=0,10$)** | 0,9791 | 0,5480 | 0,5990 | **0,9473** | 0,8383 |
| **5. Random Forest Regressor** | 0,8188 | 1,4897 | 1,7647 | 0,8759 | 1,2059 |

### 2. Efek Regularisasi Koefisien (OLS vs. Ridge L2 vs. Lasso L1)
- **Penyusutan Koefisien (*Shrinkage*)**: Regularisasi *Lasso (L1)* menyusutkan bobot fitur yang kurang dominan seperti `log_pdrb` dari `+0,2228` (pada OLS) menjadi `+0,0806`, sekaligus mempertahankan prediktor utama `hls_rata_rata` (`+1,9134`) dan `uhh_rata_rata` (`+1,2687`).
- **Hyperparameter Tuning $\alpha$**: Eksperimen *grid search* pada $\alpha \in \{0,01; 0,10; 0,50; 1,00; 2,00; 5,00\}$ menunjukkan generalisasi LOOCV optimal dicapai pada $\alpha = 1,00 - 2,00$ untuk Ridge ($R^2 = 0,9475$) dan $\alpha = 0,05$ untuk Lasso ($\text{MAE} = 0,8281$).

### 3. Diagnostik Residual Kemiskinan Wilayah
- **Residual Positif Tertinggi** (Kemiskinan aktual jauh di atas ekspektasi kapasitas IPM/PDRB): **Kabupaten Kebumen** (`+3,36%`), **Kabupaten Brebes** (`+3,23%`), dan **Kota Surakarta** (`+2,88%`).
- **Residual Negatif Terendah** (Kemiskinan aktual jauh lebih rendah dari ekspektasi model): **Kabupaten Tegal** (`-4,50%`), **Kabupaten Jepara** (`-3,78%`), dan **Kabupaten Batang** (`-3,00%`).

## Project 3: Supervised Machine Learning: Classification (`03_supervised_ml_classification.py`)

Proyek ketiga membandingkan **5 algoritma klasifikasi** (*Logistic Regression*, *K-Nearest Neighbors*, *Support Vector Machine*, *Decision Tree Gini*, dan *Random Forest Classifier*) menggunakan evaluasi *Leave-One-Out Cross-Validation (LOOCV)* pada seluruh 35 Kabupaten/Kota:

### 1. Skenario 1: Klasifikasi Status Kemiskinan Wilayah (`> 9,48%` vs. `<= 9,48%`)

| Algoritma Klasifikasi | LOOCV Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Logistic Regression (L2)** | 65,71% | 61,11% | 68,75% | 63,16% | 64,71% | 0,7599 |
| **2. K-Nearest Neighbors ($k=5$)** | 65,71% | 61,11% | 68,75% | 63,16% | 64,71% | 0,6480 |
| **3. Support Vector Machine (SVM)** | 62,86% | 57,89% | 68,75% | 57,89% | 62,86% | 0,7171 |
| **4. Decision Tree (Gini)** | 71,43% | 66,67% | 75,00% | 68,42% | 70,59% | 0,5921 |
| **5. Random Forest Classifier** | **74,29%** | **68,42%** | **81,25%** | **68,42%** | **74,29%** | **0,7697** |

- **Tuning Nilai $k$ pada KNN**: Meningkatkan jumlah tetangga dari $k=3$ (`62,86%`) ke **$k=7$** mendongkrak akurasi LOOCV menjadi **74,29%** dan *F1-Score* menjadi **75,68%** ($\text{ROC-AUC} = 0,7401$).

### 2. Skenario 2: Klasifikasi Klaster Pembangunan Wilayah (`IPM >= 75,0` vs. `< 75,0`)

| Algoritma Klasifikasi | LOOCV Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Logistic Regression (L2)** | 74,29% | 75,00% | 70,59% | 77,78% | 72,73% | 0,8791 |
| **2. K-Nearest Neighbors ($k=5$)** | 74,29% | 75,00% | 70,59% | 77,78% | 72,73% | 0,8170 |
| **3. Support Vector Machine (SVM)** | 74,29% | 83,33% | 58,82% | **88,89%** | 68,97% | 0,8954 |
| **4. Decision Tree (Gini)** | **88,57%** | **88,24%** | **88,24%** | **88,89%** | **88,24%** | 0,8693 |
| **5. Random Forest Classifier** | 80,00% | 81,25% | 76,47% | 83,33% | 78,79% | **0,9118** |

## Cara Menjalankan Ketiga Proyek di VS Code

1. Simpan `bps_jawa_tengah_2025.csv` bersama ketiga file script Python di dalam folder proyek.
2. Buka terminal VS Code (`Ctrl + ~`), lalu jalankan masing-masing proyek secara mandiri:
   ```bash
   # Menjalankan Project 1: Exploratory Data Analysis for Machine Learning
   python 01_eda_for_machine_learning.py

   # Menjalankan Project 2: Supervised Machine Learning - Regression
   python 02_supervised_ml_regression.py

   # Menjalankan Project 3: Supervised Machine Learning - Classification
   python 03_supervised_ml_classification.py
   ```
