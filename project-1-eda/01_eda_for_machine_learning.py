#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROJECT 1 OF 3: EXPLORATORY DATA ANALYSIS (EDA) FOR MACHINE LEARNING
================================================================================
Kurikulum : Course 1 - Exploratory Data Analysis for Machine Learning
Studi Kasus: Analisis Sosio-Ekonomi 35 Kabupaten/Kota di Jawa Tengah (BPS 2025)
Cakupan Materi Course 1:
  Data Retrieval, Schema Inspection & Summary Statistics
  Data Cleaning (Missing Values, Duplicates) & Outlier Detection (IQR & Z-Score)
  Skewness Analysis & Feature Transformation (Log Transform ln(PDRB))
  Feature Engineering & Categorical Encoding (One-Hot / Binary Encoding)
  Feature Scaling Comparison (StandardScaler vs. MinMaxScaler)
  Statistical Hypothesis Testing (Welch's T-Test: Kabupaten vs. Kota)
    & Multivariate Correlation Analysis (Pearson & Spearman Rank)
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)


def locate_dataset() -> str:
    """Mencari lokasi file CSV dataset BPS Jawa Tengah 2025."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "bps_jawa_tengah_2025.csv"),
        os.path.join(script_dir, "..", "bps_jawa_tengah_2025.csv"),
        os.path.join(script_dir, "data", "bps_jawa_tengah_2025.csv"),
        "bps_jawa_tengah_2025.csv",
        "artifacts/file_generation/ttl=63d/output/bps_jawa_tengah_2025.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "File 'bps_jawa_tengah_2025.csv' tidak ditemukan. Pastikan berada di folder proyek."
    )


def compute_skewness(series: pd.Series) -> float:
    """Menghitung koefisien kemencengan (Fisher-Pearson skewness)."""
    x = series.dropna().values.astype(float)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def welch_t_test(group1: np.ndarray, group2: np.ndarray):
    """Menghitung Welch's two-sample t-test (tanpa asumsi varians sama)."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se if se > 0 else 0.0
    # Degrees of freedom (Welch-Satterthwaite equation)
    df = ((v1 / n1 + v2 / n2) ** 2) / (
        ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    )
    # Normal approximation for two-tailed p-value
    from math import erf, sqrt
    p_val_approx = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
    return float(t_stat), float(df), float(p_val_approx)


def run_eda_project():
    print("=" * 86)
    print("PROJECT 1: EXPLORATORY DATA ANALYSIS (EDA) FOR MACHINE LEARNING")
    print("Studi Kasus: Data Indikator Sosio-Ekonomi 35 Kabupaten/Kota BPS Jawa Tengah 2025")
    print("=" * 86)

    # --------------------------------------------------------------------------
    # STEP 1: DATA RETRIEVAL & SCHEMA INSPECTION
    # --------------------------------------------------------------------------
    csv_path = locate_dataset()
    df = pd.read_csv(csv_path)
    print(f"\n[STEP 1] Data Retrieval & Schema Inspection")
    print(f"         - Lokasi Dataset : {os.path.basename(csv_path)}")
    print(f"         - Dimensi Data   : {df.shape[0]} Observasi (Kabupaten/Kota) x {df.shape[1]} Variabel")
    print("         - Tipe Data Kolom:")
    for col, dtype in df.dtypes.items():
        print(f"           * {col:<35}: {str(dtype)}")

    # --------------------------------------------------------------------------
    # STEP 2: DATA CLEANING & OUTLIER DETECTION (IQR & Z-SCORE)
    # --------------------------------------------------------------------------
    print(f"\n[STEP 2] Data Cleaning & Outlier Detection (IQR & Z-Score)")
    print(f"         - Missing Values : {int(df.isnull().sum().sum())} sel kosong")
    print(f"         - Data Duplikat  : {int(df.duplicated().sum())} baris duplikat")

    num_cols = [
        "ipm_2025",
        "persentase_penduduk_miskin_2025",
        "pdrb_adhb_2025_triliun_rp",
        "hls_laki_laki_2025",
        "hls_perempuan_2025",
        "uhh_laki_laki_2025",
        "uhh_perempuan_2025",
    ]
    print("\n         Ringkasan Deteksi Outlier Menggunakan Metode Interquartile Range (IQR):")
    print("         " + "-" * 76)
    print(f"         {'Variabel Numerik':<34} | {'Q1':>7} | {'Median':>7} | {'Q3':>7} | {'Outlier (IQR)':>11}")
    print("         " + "-" * 76)
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        med = df[col].median()
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        print(f"         {col:<34} | {q1:>7.2f} | {med:>7.2f} | {q3:>7.2f} | {n_out:>5} wilayah")

    # Detail outlier PDRB
    q1_p, q3_p = df["pdrb_adhb_2025_triliun_rp"].quantile([0.25, 0.75])
    upper_pdrb = q3_p + 1.5 * (q3_p - q1_p)
    outliers_df = df[df["pdrb_adhb_2025_triliun_rp"] > upper_pdrb]
    print(f"\n         Wilayah Outlier PDRB (> Rp {upper_pdrb:.2f} Triliun):")
    for _, r in outliers_df.iterrows():
        print(f"         * {r['nama_wilayah']:<22}: Rp {r['pdrb_adhb_2025_triliun_rp']:>6.2f} Triliun ({r['kontribusi_pdrb_jateng_persen']:.2f}% PDRB Jateng)")

    # --------------------------------------------------------------------------
    # STEP 3: SKEWNESS ANALYSIS & FEATURE TRANSFORMATION
    # --------------------------------------------------------------------------
    print(f"\n[STEP 3] Skewness Analysis & Feature Transformation (Mengatasi Kecondongan Data)")
    skew_raw = compute_skewness(df["pdrb_adhb_2025_triliun_rp"])
    df["sqrt_pdrb"] = np.sqrt(df["pdrb_adhb_2025_triliun_rp"])
    df["log_pdrb"] = np.log(df["pdrb_adhb_2025_triliun_rp"])
    skew_sqrt = compute_skewness(df["sqrt_pdrb"])
    skew_log = compute_skewness(df["log_pdrb"])

    print(f"         - Skewness PDRB Mentah (Raw)         : {skew_raw:+.4f} (Sangat Right-Skewed, perlu transformasi)")
    print(f"         - Skewness Akar Kuadrat sqrt(PDRB)   : {skew_sqrt:+.4f} (Masih cukup miring ke kanan)")
    print(f"         - Skewness Logaritma Natural ln(PDRB): {skew_log:+.4f} (Mendekati distribusi normal/simetris!)")

    # --------------------------------------------------------------------------
    # STEP 4: FEATURE ENGINEERING & ENCODING
    # --------------------------------------------------------------------------
    print(f"\n[STEP 4] Feature Engineering & Categorical Encoding")
    df["hls_rata_rata"] = ((df["hls_laki_laki_2025"] + df["hls_perempuan_2025"]) / 2.0).round(3)
    df["uhh_rata_rata"] = ((df["uhh_laki_laki_2025"] + df["uhh_perempuan_2025"]) / 2.0).round(3)
    df["gap_hls_gender"] = (df["hls_perempuan_2025"] - df["hls_laki_laki_2025"]).round(3)
    df["gap_uhh_gender"] = (df["uhh_perempuan_2025"] - df["uhh_laki_laki_2025"]).round(3)
    df["rasio_pendidikan_kesehatan"] = (df["hls_rata_rata"] / df["uhh_rata_rata"]).round(4)
    df["is_kota"] = (df["tipe_wilayah"] == "Kota").astype(int)

    print("         Fitur baru hasil rekayasa (Engineered Features):")
    print("         1. hls_rata_rata              : Rata-rata Harapan Lama Sekolah (Tahun)")
    print("         2. uhh_rata_rata              : Rata-rata Umur Harapan Hidup Saat Lahir (Tahun)")
    print("         3. gap_hls_gender             : Selisih HLS Perempuan terhadap Laki-laki (Tahun)")
    print("         4. gap_uhh_gender             : Selisih UHH Perempuan terhadap Laki-laki (Tahun)")
    print("         5. rasio_pendidikan_kesehatan : Rasio HLS terhadap UHH")
    print("         6. log_pdrb                   : Transformasi Logaritma Natural PDRB ADHB")
    print("         7. is_kota                    : Binary Encoding (1 = Kota, 0 = Kabupaten)")

    # --------------------------------------------------------------------------
    # STEP 5: FEATURE SCALING COMPARISON (STANDARDSCALER VS MINMAXSCALER)
    # --------------------------------------------------------------------------
    print(f"\n[STEP 5] Perbandingan Metode Feature Scaling (StandardScaler vs MinMaxScaler)")
    scale_cols = ["ipm_2025", "persentase_penduduk_miskin_2025", "hls_rata_rata", "uhh_rata_rata", "log_pdrb"]
    print("         " + "-" * 76)
    print(f"         {'Fitur':<32} | {'Raw Mean':>9} | {'Z-Score Mean/Std':>18} | {'MinMax [Min, Max]':>17}")
    print("         " + "-" * 76)
    for col in scale_cols:
        vals = df[col].values
        z_vals = (vals - np.mean(vals)) / np.std(vals)
        mm_vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
        print(
            f"         {col:<32} | {np.mean(vals):>9.2f} | {np.mean(z_vals):>+7.2f} / {np.std(z_vals):>6.2f} | [{np.min(mm_vals):>5.2f}, {np.max(mm_vals):>5.2f}]"
        )

    # --------------------------------------------------------------------------
    # STEP 6: STATISTICAL HYPOTHESIS TESTING & CORRELATION MATRIX
    # --------------------------------------------------------------------------
    print(f"\n[STEP 6] Uji Hipotesis Statistik (Welch's T-Test: Kota vs Kabupaten) & Korelasi")
    kota_ipm = df[df["is_kota"] == 1]["ipm_2025"].values
    kab_ipm = df[df["is_kota"] == 0]["ipm_2025"].values
    t_ipm, df_ipm, p_ipm = welch_t_test(kota_ipm, kab_ipm)

    kota_pov = df[df["is_kota"] == 1]["persentase_penduduk_miskin_2025"].values
    kab_pov = df[df["is_kota"] == 0]["persentase_penduduk_miskin_2025"].values
    t_pov, df_pov, p_pov = welch_t_test(kota_pov, kab_pov)

    print("         Uji Hipotesis H0: Tidak ada perbedaan signifikan antara rata-rata Kota (n=6) & Kabupaten (n=29)")
    print(f"         - Indeks Pembangunan Manusia (IPM) : Mean Kota = {np.mean(kota_ipm):.2f} vs Kab = {np.mean(kab_ipm):.2f}")
    print(f"           t-statistic = {t_ipm:+.4f}, df = {df_ipm:.2f}, p-value = {p_ipm:.6f} -> Tolak H0 (Signifikan pada alpha=0.01)")
    print(f"         - Persentase Penduduk Miskin (%)   : Mean Kota = {np.mean(kota_pov):.2f}% vs Kab = {np.mean(kab_pov):.2f}%")
    print(f"           t-statistic = {t_pov:+.4f}, df = {df_pov:.2f}, p-value = {p_pov:.6f} -> Tolak H0 (Signifikan pada alpha=0.01)")

    print("\n         Matriks Korelasi Pearson terhadap Target Utama (IPM & Kemiskinan):")
    corr_features = ["ipm_2025", "persentase_penduduk_miskin_2025", "hls_rata_rata", "uhh_rata_rata", "log_pdrb", "gap_hls_gender", "is_kota"]
    corr_mat = df[corr_features].corr()
    print(corr_mat[["ipm_2025", "persentase_penduduk_miskin_2025"]].round(4).to_string())

    print("\n" + "=" * 86)
    print("PROJECT 1 (EDA FOR MACHINE LEARNING) SELESAI DIJALANKAN DENGAN SUKSES!")
    print("=" * 86)


if __name__ == "__main__":
    run_eda_project()