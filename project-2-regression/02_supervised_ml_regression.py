#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROJECT 2 OF 3: SUPERVISED MACHINE LEARNING - REGRESSION
================================================================================
Kurikulum : Course 2 - Supervised Machine Learning: Regression
Studi Kasus: Prediksi IPM & Pemodelan Regresi Kemiskinan Jawa Tengah (BPS 2025)
Cakupan Materi Course 2:
  Persiapan Fitur, Train-Test Split & Standarisasi Tanpa Data Leakage
  Ordinary Least Squares (OLS) Multiple Linear Regression
  Regularized Regression:
     - Ridge Regression (L2 Regularization) + Hyperparameter Tuning Alpha
     - Lasso Regression (L1 Regularization & Automatic Feature Selection)
     - ElasticNet Regression (Kombinasi Penalti L1 + L2)
  Non-Linear Ensemble Regression (Random Forest Regressor)
  Evaluasi Model (R2, Adjusted R2, MAE, RMSE, Leave-One-Out CV) &
     Analisis Diagnostik Residual (Deteksi Anomali Daerah)
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


# ==============================================================================
# IMPLEMENTASI ALGORITMA REGRESI (OLS, RIDGE L2, LASSO L1, ELASTICNET, RF)
# ==============================================================================
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LinearRegressionOLS:
    """Multiple Linear Regression menggunakan Ordinary Least Squares (OLS)."""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


class RidgeRegressionL2:
    """Ridge Regression dengan penalti L2: min ||Xw + b - y||^2 + alpha * ||w||_2^2."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        X_b = np.c_[np.ones((n, 1)), X]
        I = np.eye(p + 1)
        I[0, 0] = 0.0
        theta = np.linalg.solve(X_b.T @ X_b + self.alpha * I, X_b.T @ y)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


class ElasticNetCoordinateDescent:
    """
    Lasso (l1_ratio=1.0) & ElasticNet (0 < l1_ratio < 1) via Coordinate Descent.
    Objektif: (1 / (2n)) * ||y - Xw - b||_2^2 + alpha * l1_ratio * ||w||_1
              + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2
    """
    def __init__(self, alpha=0.1, l1_ratio=1.0, max_iter=2000, tol=1e-6):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0

    @staticmethod
    def _soft_threshold(rho, lam):
        if rho < -lam:
            return rho + lam
        if rho > lam:
            return rho - lam
        return 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.intercept_ = float(np.mean(y))
        y_centered = y - self.intercept_
        w = np.zeros(p, dtype=float)
        l1_pen = self.alpha * self.l1_ratio
        l2_pen = self.alpha * (1.0 - self.l1_ratio)

        for _ in range(self.max_iter):
            w_old = w.copy()
            for j in range(p):
                residual = y_centered - (X @ w - X[:, j] * w[j])
                rho = float(np.dot(X[:, j], residual) / n)
                denom = float(np.sum(X[:, j] ** 2) / n + l2_pen)
                w[j] = self._soft_threshold(rho, l1_pen) / denom if denom > 0 else 0.0
            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


class _SimpleRegTree:
    def __init__(self, max_depth=4, min_samples_split=3, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = rng or np.random.default_rng(42)
        self.root = None
        self.feature_importances_ = None

    def _build(self, X, y, depth):
        n, p = X.shape
        if depth >= self.max_depth or n < self.min_samples_split or len(np.unique(y)) == 1:
            return {"leaf": True, "val": float(np.mean(y))}

        parent_var = float(np.var(y))
        best_gain, best_f, best_t = 0.0, None, None
        for f in range(p):
            vals = np.unique(X[:, f])
            if len(vals) <= 1:
                continue
            for t in (vals[:-1] + vals[1:]) / 2.0:
                left = X[:, f] <= t
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue
                gain = parent_var - (np.sum(left) / n) * float(np.var(y[left])) - (np.sum(right) / n) * float(np.var(y[right]))
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, float(t)

        if best_f is None:
            return {"leaf": True, "val": float(np.mean(y))}

        self.feature_importances_[best_f] += best_gain * n
        left_mask = X[:, best_f] <= best_t
        return {
            "leaf": False,
            "f": best_f,
            "t": best_t,
            "l": self._build(X[left_mask], y[left_mask], depth + 1),
            "r": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.root = self._build(X, y, 0)
        tot = np.sum(self.feature_importances_)
        if tot > 0:
            self.feature_importances_ /= tot
        return self

    def _pred_one(self, row, node):
        if node["leaf"]:
            return node["val"]
        return self._pred_one(row, node["l"] if row[node["f"]] <= node["t"] else node["r"])

    def predict(self, X):
        return np.array([self._pred_one(r, self.root) for r in np.asarray(X, dtype=float)])


class RandomForestRegressorEnsemble:
    def __init__(self, n_estimators=150, max_depth=4, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        imp = np.zeros(X.shape[1], dtype=float)
        n = len(X)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            t = _SimpleRegTree(max_depth=self.max_depth, rng=rng).fit(X[idx], y[idx])
            self.trees_.append(t)
            imp += t.feature_importances_
        self.feature_importances_ = imp / np.sum(imp) if np.sum(imp) > 0 else imp
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees_], axis=0)


def calc_metrics(y_true, y_pred, n_features=5):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, (n - n_features - 1))
    return r2, adj_r2, mae, rmse


def loocv_eval(factory, X_df, y_ser, scale=True):
    X = X_df.values
    y = y_ser.values
    n = len(y)
    preds = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[~mask]
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        model = factory().fit(X_tr, y_tr)
        preds[i] = model.predict(X_te)[0]
    return preds


def run_regression_project():
    print("=" * 88)
    print("PROJECT 2: SUPERVISED MACHINE LEARNING - REGRESSION")
    print("Studi Kasus: Prediksi Indeks Pembangunan Manusia (IPM) & Regresi Kemiskinan Jawa Tengah")
    print("=" * 88)

    df = pd.read_csv(locate_dataset())
    df["hls_rata_rata"] = (df["hls_laki_laki_2025"] + df["hls_perempuan_2025"]) / 2.0
    df["uhh_rata_rata"] = (df["uhh_laki_laki_2025"] + df["uhh_perempuan_2025"]) / 2.0
    df["gap_hls_gender"] = df["hls_perempuan_2025"] - df["hls_laki_laki_2025"]
    df["log_pdrb"] = np.log(df["pdrb_adhb_2025_triliun_rp"])
    df["is_kota"] = (df["tipe_wilayah"] == "Kota").astype(int)

    features = [
        "hls_rata_rata",
        "uhh_rata_rata",
        "persentase_penduduk_miskin_2025",
        "log_pdrb",
        "gap_hls_gender",
        "is_kota",
    ]
    X = df[features]
    y = df["ipm_2025"]

    # Train-Test Split (80% Train = 28 Wilayah, 20% Test = 7 Wilayah)
    rng = np.random.default_rng(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    test_idx, train_idx = idx[:7], idx[7:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --------------------------------------------------------------------------
    # STEP 1: PERBANDINGAN 5 MODEL REGRESI (OLS, RIDGE, LASSO, ELASTICNET, RF)
    # --------------------------------------------------------------------------
    models = [
        ("1. Linear Regression (OLS)", lambda: LinearRegressionOLS(), True),
        ("2. Ridge Regression (L2, a=1.0)", lambda: RidgeRegressionL2(alpha=1.0), True),
        ("3. Lasso Regression (L1, a=0.08)", lambda: ElasticNetCoordinateDescent(alpha=0.08, l1_ratio=1.0), True),
        ("4. ElasticNet (L1+L2, a=0.10)", lambda: ElasticNetCoordinateDescent(alpha=0.10, l1_ratio=0.5), True),
        ("5. Random Forest Regressor", lambda: RandomForestRegressorEnsemble(n_estimators=150, max_depth=4), False),
    ]

    print("\n[STEP 1] Evaluasi Perbandingan 5 Model Regresi (Target: Indeks Pembangunan Manusia 2025)")
    print("-" * 88)
    print(f"{'Nama Model Regresi':<34} | {'R2 Test':>8} | {'MAE Test':>8} | {'RMSE Test':>9} | {'LOOCV R2':>8} | {'LOOCV MAE':>9}")
    print("-" * 88)

    fitted = {}
    for name, factory, use_scale in models:
        m = factory()
        if use_scale:
            m.fit(X_train_s, y_train.values)
            preds_te = m.predict(X_test_s)
        else:
            m.fit(X_train.values, y_train.values)
            preds_te = m.predict(X_test.values)

        r2_t, _, mae_t, rmse_t = calc_metrics(y_test.values, preds_te, len(features))
        loo_preds = loocv_eval(factory, X, y, scale=use_scale)
        r2_l, _, mae_l, rmse_l = calc_metrics(y.values, loo_preds, len(features))
        fitted[name] = (m, preds_te, loo_preds)
        print(f"{name:<34} | {r2_t:>8.4f} | {mae_t:>8.4f} | {rmse_t:>9.4f} | {r2_l:>8.4f} | {mae_l:>9.4f}")

    # --------------------------------------------------------------------------
    # STEP 2: HYPERPARAMETER TUNING ALPHA PADA RIDGE & LASSO REGRESSION
    # --------------------------------------------------------------------------
    print("\n[STEP 2] Hyperparameter Tuning Regularization Strength (Alpha) via LOOCV")
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    print("         " + "-" * 68)
    print(f"         {'Alpha':>8} | {'Ridge LOOCV R2':>15} | {'Ridge LOOCV MAE':>16} | {'Lasso LOOCV MAE':>16}")
    print("         " + "-" * 68)
    for a in alphas:
        r_loo = loocv_eval(lambda: RidgeRegressionL2(alpha=a), X, y, scale=True)
        l_loo = loocv_eval(lambda: ElasticNetCoordinateDescent(alpha=a * 0.1, l1_ratio=1.0), X, y, scale=True)
        r_r2, _, r_mae, _ = calc_metrics(y.values, r_loo)
        _, _, l_mae, _ = calc_metrics(y.values, l_loo)
        print(f"         {a:>8.2f} | {r_r2:>15.4f} | {r_mae:>16.4f} | {l_mae:>16.4f}")

    # --------------------------------------------------------------------------
    # STEP 3: EFEK REGULARISASI & SELEKSI FITUR OTOMATIS OLEH LASSO (L1)
    # --------------------------------------------------------------------------
    print("\n[STEP 3] Perbandingan Koefisien Regresi (OLS vs Ridge L2 vs Lasso L1 Feature Selection)")
    ols_c = fitted["1. Linear Regression (OLS)"][0].coef_
    ridge_c = fitted["2. Ridge Regression (L2, a=1.0)"][0].coef_
    lasso_c = fitted["3. Lasso Regression (L1, a=0.08)"][0].coef_
    print("         " + "-" * 76)
    print(f"         {'Fitur Prediktor':<34} | {'Koef OLS':>10} | {'Koef Ridge':>11} | {'Koef Lasso (L1)':>15}")
    print("         " + "-" * 76)
    for feat, c_o, c_r, c_l in zip(features, ols_c, ridge_c, lasso_c):
        status = " (Eliminasi L1)" if abs(c_l) < 1e-4 else ""
        print(f"         {feat:<34} | {c_o:>+10.4f} | {c_r:>+11.4f} | {c_l:>+10.4f}{status}")

    # --------------------------------------------------------------------------
    # STEP 4: DIAGNOSTIK RESIDUAL & DETEKSI ANOMALI KEMISKINAN WILAYAH
    # --------------------------------------------------------------------------
    print("\n[STEP 4] Analisis Diagnostik Residual & Deteksi Anomali Kemiskinan Wilayah")
    X_pov = df[["ipm_2025", "hls_rata_rata", "uhh_rata_rata", "log_pdrb", "is_kota"]]
    y_pov = df["persentase_penduduk_miskin_2025"]
    X_pov_s = StandardScaler().fit_transform(X_pov)
    pov_model = RidgeRegressionL2(alpha=1.0).fit(X_pov_s, y_pov.values)
    df["pred_kemiskinan"] = pov_model.predict(X_pov_s)
    df["residual_kemiskinan"] = df["persentase_penduduk_miskin_2025"] - df["pred_kemiskinan"]

    print("         Top 3 Wilayah dengan Kemiskinan Aktual di Atas Ekspektasi Model (Positive Residual):")
    for _, r in df.sort_values("residual_kemiskinan", ascending=False).head(3).iterrows():
        print(f"         * {r['nama_wilayah']:<22}: Aktual {r['persentase_penduduk_miskin_2025']:>5.2f}% vs Prediksi {r['pred_kemiskinan']:>5.2f}% (Residual {r['residual_kemiskinan']:>+5.2f}%)")

    print("\n         Top 3 Wilayah dengan Kemiskinan Aktual Lebih Rendah dari Ekspektasi Model (Negative Residual):")
    for _, r in df.sort_values("residual_kemiskinan", ascending=True).head(3).iterrows():
        print(f"         * {r['nama_wilayah']:<22}: Aktual {r['persentase_penduduk_miskin_2025']:>5.2f}% vs Prediksi {r['pred_kemiskinan']:>5.2f}% (Residual {r['residual_kemiskinan']:>+5.2f}%)")

    print("\n" + "=" * 88)
    print("PROJECT 2 (SUPERVISED MACHINE LEARNING: REGRESSION) SELESAI DENGAN SUKSES!")
    print("=" * 88)


if __name__ == "__main__":
    run_regression_project()